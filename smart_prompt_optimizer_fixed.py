#!/usr/bin/env python3
"""
Smart Prompt Optimizer - PERFORMANCE OPTIMIZED
==============================================
🚀 Fixed major performance bottlenecks from critic feedback
⚡ Vector embeddings for instant similarity (not LLM calls)
🎯 Pre-computed embeddings for O(1) similarity search
📊 Streamlined optimization with implicit pattern recognition

Critical Fixes:
1. Replace 400+ LLM calls with vector similarity (milliseconds vs minutes)
2. Pre-compute embeddings offline for instant inference
3. Let LLM infer patterns implicitly rather than explicit extraction
4. Add data quality assessment and fallback strategies
"""

import json
import requests
import time
import sys
import csv
import numpy as np
from typing import Dict, List, Optional, Tuple
from pathlib import Path
from dataclasses import dataclass
import statistics
import subprocess
import pickle
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

@dataclass
class TrainingExample:
    """A training example with actual validation scores and pre-computed embedding"""
    original: str
    optimized: str
    score: float
    episode: int
    step: int
    strategy: str
    embedding: Optional[np.ndarray] = None  # Pre-computed for fast similarity

@dataclass
class FastMatch:
    """Fast similarity match using vector embeddings"""
    training_example: TrainingExample
    similarity_score: float

class FastTrainingDataLoader:
    """High-performance training data loader with pre-computed embeddings"""
    
    def __init__(self, csv_file: str = "prompt_score_log.csv", 
                 cache_file: str = "training_embeddings.pkl"):
        self.csv_file = Path(csv_file)
        self.cache_file = Path(cache_file)
        self.training_examples = []
        self.high_score_examples = []
        
        # Initialize sentence transformer for embeddings
        try:
            self.encoder = SentenceTransformer('all-MiniLM-L6-v2')
            print("✅ SentenceTransformer loaded successfully")
        except ImportError:
            print("❌ sentence-transformers not available - install with: pip install sentence-transformers")
            self.encoder = None
            
        self._load_training_data_with_embeddings()
        self._assess_data_quality()
    
    def _load_training_data_with_embeddings(self):
        """Load training data and compute embeddings efficiently"""
        
        # Try to load from cache first
        if self.cache_file.exists() and self.csv_file.exists():
            csv_mtime = self.csv_file.stat().st_mtime
            cache_mtime = self.cache_file.stat().st_mtime
            
            if cache_mtime > csv_mtime:
                print(f"📊 Loading cached embeddings from {self.cache_file}")
                try:
                    with open(self.cache_file, 'rb') as f:
                        cached_data = pickle.load(f)
                        self.training_examples = cached_data['training_examples']
                        self.high_score_examples = cached_data['high_score_examples']
                        print(f"   ⚡ Loaded {len(self.training_examples)} cached examples")
                        return
                except Exception as e:
                    print(f"   ⚠️ Cache loading failed: {e}, rebuilding...")
        
        # Load from CSV and compute embeddings
        if not self.csv_file.exists():
            print(f"⚠️ No training data found at {self.csv_file}")
            print(f"💡 Run your training script first to generate actual data")
            return
        
        if not self.encoder:
            print(f"❌ Cannot compute embeddings without sentence-transformers")
            return
            
        print(f"📊 Loading training data from {self.csv_file}")
        print(f"🔄 Computing embeddings (one-time cost)...")
        
        examples_to_encode = []
        
        with open(self.csv_file, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    example = TrainingExample(
                        original=row['original'],
                        optimized=row['custom'],
                        score=float(row['score']),
                        episode=int(row['episode']),
                        step=int(row['step']),
                        strategy=row.get('strategy', 'unknown')
                    )
                    
                    self.training_examples.append(example)
                    examples_to_encode.append(example.original)
                    
                    # Keep high-scoring examples (>0.85)
                    if example.score >= 0.85:
                        self.high_score_examples.append(example)
                        
                except (ValueError, KeyError) as e:
                    print(f"   ⚠️ Skipping malformed row: {e}")
        
        # Batch compute embeddings for efficiency
        if examples_to_encode and self.encoder:
            print(f"   🧮 Computing {len(examples_to_encode)} embeddings...")
            embeddings = self.encoder.encode(examples_to_encode, batch_size=32, show_progress_bar=True)
            
            # Assign embeddings to examples
            for example, embedding in zip(self.training_examples, embeddings):
                example.embedding = embedding
        
        print(f"   📈 Loaded {len(self.training_examples)} training examples")
        print(f"   💎 Found {len(self.high_score_examples)} high-scoring examples (≥0.85)")
        
        # Sort high-scoring examples by score
        self.high_score_examples.sort(key=lambda x: x.score, reverse=True)
        
        # Cache the results
        self._cache_embeddings()
    
    def _cache_embeddings(self):
        """Cache embeddings to disk for fast future loading"""
        try:
            cache_data = {
                'training_examples': self.training_examples,
                'high_score_examples': self.high_score_examples
            }
            with open(self.cache_file, 'wb') as f:
                pickle.dump(cache_data, f)
            print(f"   💾 Cached embeddings to {self.cache_file}")
        except Exception as e:
            print(f"   ⚠️ Caching failed: {e}")
    
    def _assess_data_quality(self):
        """Assess quality and diversity of training data"""
        if not self.high_score_examples:
            print("⚠️ DATA QUALITY WARNING: No high-scoring examples found!")
            return
        
        print(f"\n📊 DATA QUALITY ASSESSMENT:")
        
        # Score distribution
        scores = [ex.score for ex in self.high_score_examples]
        print(f"   📈 Score range: {min(scores):.3f} - {max(scores):.3f}")
        print(f"   📊 Average score: {statistics.mean(scores):.3f}")
        print(f"   🎯 Ultra scores (≥0.96): {sum(1 for s in scores if s >= 0.96)}")
        
        # Diversity assessment (simple keyword-based)
        categories = self._categorize_examples(self.high_score_examples)
        print(f"   🌈 Categories found: {len(categories)}")
        for category, count in categories.items():
            print(f"     {category}: {count} examples")
        
        # Quality warnings
        if len(categories) < 3:
            print("   ⚠️ WARNING: Low diversity - consider training on more object types")
        
        if len(self.high_score_examples) < 10:
            print("   ⚠️ WARNING: Very few high-scoring examples - may need more training")
        
        print()
    
    def _categorize_examples(self, examples: List[TrainingExample]) -> Dict[str, int]:
        """Simple categorization for diversity assessment"""
        categories = {}
        
        for example in examples:
            original = example.original.lower()
            category = "other"
            
            if any(word in original for word in ['glass', 'crystal', 'transparent', 'clear']):
                category = "glass/crystal"
            elif any(word in original for word in ['metal', 'steel', 'iron', 'weapon', 'spear']):
                category = "metal/weapons"
            elif any(word in original for word in ['food', 'drink', 'beverage', 'cake', 'lemonade']):
                category = "food/beverage"
            elif any(word in original for word in ['fabric', 'silk', 'cloth', 'textile']):
                category = "textiles"
            elif any(word in original for word in ['creature', 'character', 'animal']):
                category = "creatures"
            elif any(word in original for word in ['jewelry', 'pendant', 'ring', 'gem']):
                category = "jewelry"
            
            categories[category] = categories.get(category, 0) + 1
        
        return categories

class VectorSimilarityEngine:
    """High-performance similarity using vector embeddings"""
    
    def __init__(self):
        try:
            self.encoder = SentenceTransformer('all-MiniLM-L6-v2')
            print("⚡ Vector similarity engine ready")
        except ImportError:
            print("❌ sentence-transformers not available")
            self.encoder = None
    
    def find_similar_examples_fast(self, user_prompt: str, 
                                 training_examples: List[TrainingExample], 
                                 top_k: int = 5) -> List[FastMatch]:
        """Find similar examples using lightning-fast vector similarity"""
        
        if not self.encoder:
            print("   ❌ Vector encoder not available - falling back to slow method")
            return []
        
        print(f"   ⚡ Finding matches using vector similarity...")
        start_time = time.time()
        
        # Encode user prompt
        user_embedding = self.encoder.encode([user_prompt])[0]
        
        # Get pre-computed embeddings
        training_embeddings = []
        valid_examples = []
        
        for example in training_examples:
            if example.embedding is not None:
                training_embeddings.append(example.embedding)
                valid_examples.append(example)
        
        if not training_embeddings:
            print("   ❌ No pre-computed embeddings found")
            return []
        
        # Fast cosine similarity computation
        training_matrix = np.array(training_embeddings)
        similarities = cosine_similarity([user_embedding], training_matrix)[0]
        
        # Get top matches
        top_indices = np.argsort(similarities)[-top_k:][::-1]  # Highest first
        
        matches = []
        for idx in top_indices:
            if similarities[idx] > 0.3:  # Relevance threshold
                matches.append(FastMatch(
                    training_example=valid_examples[idx],
                    similarity_score=float(similarities[idx])
                ))
        
        elapsed = time.time() - start_time
        print(f"   ⚡ Found {len(matches)} matches in {elapsed:.3f}s")
        
        return matches

class OptimizedPromptOptimizer:
    """High-performance optimizer with streamlined architecture"""
    
    def __init__(self, training_csv: str = "prompt_score_log.csv", 
                 ollama_url: str = "http://localhost:11434"):
        self.ollama_url = ollama_url
        self.model = "llama3.2:3b"
        
        # Load training data with pre-computed embeddings
        self.training_loader = FastTrainingDataLoader(training_csv)
        self.similarity_engine = VectorSimilarityEngine()
        
        print(f"🚀 OPTIMIZED PROMPT OPTIMIZER READY")
        print(f"   📊 Training examples: {len(self.training_loader.training_examples)}")
        print(f"   💎 High-scoring examples: {len(self.training_loader.high_score_examples)}")
    
    def optimize(self, user_prompt: str, use_validation: bool = False) -> Dict:
        """
        High-performance prompt optimization
        
        Args:
            user_prompt: Prompt to optimize
            use_validation: Whether to run actual validation (slow but accurate)
        
        Returns:
            Optimization result with training-informed confidence
        """
        start_time = time.time()
        
        print(f"\n🎯 OPTIMIZING: '{user_prompt}'")
        
        # Fast similarity search using vectors
        similar_matches = self.similarity_engine.find_similar_examples_fast(
            user_prompt, self.training_loader.high_score_examples, top_k=5
        )
        
        if not similar_matches:
            print("   ⚠️ No similar training examples found - using fallback")
            return self._fallback_optimization(user_prompt, start_time)
        
        print(f"   📊 Found {len(similar_matches)} similar training examples")
        for i, match in enumerate(similar_matches[:3]):
            ex = match.training_example
            print(f"     {i+1}. {ex.original} (sim: {match.similarity_score:.3f}, score: {ex.score:.3f})")
        
        # Streamlined optimization with implicit pattern recognition
        optimized_prompt = self._generate_streamlined_optimization(user_prompt, similar_matches)
        
        # Fast confidence calculation
        confidence = self._calculate_fast_confidence(user_prompt, optimized_prompt, similar_matches)
        
        # Optional validation
        actual_score = None
        if use_validation:
            actual_score = self._validate_prompt(optimized_prompt)
        
        processing_time = time.time() - start_time
        
        result = {
            'original_prompt': user_prompt,
            'optimized_prompt': optimized_prompt,
            'confidence': confidence,
            'processing_time': processing_time,
            'similar_examples_count': len(similar_matches),
            'best_similar_score': similar_matches[0].training_example.score if similar_matches else 0.0,
            'actual_validation_score': actual_score,
            'data_quality_score': self._assess_local_data_quality(similar_matches)
        }
        
        print(f"✅ RESULT: {optimized_prompt}")
        print(f"⏱️  Time: {processing_time:.2f}s | Confidence: {confidence:.1%}")
        if actual_score:
            print(f"🎯 Validation Score: {actual_score:.3f}")
        
        return result
    
    def _generate_streamlined_optimization(self, user_prompt: str, 
                                         similar_matches: List[FastMatch]) -> str:
        """Streamlined optimization with implicit pattern recognition"""
        
        # Build system prompt with successful transformations
        system_prompt = """You are an expert prompt optimizer targeting validation scores >0.96.

Analyze the successful transformations below. Identify the principles that made them score highly and apply those same principles to the new user prompt.

CRITICAL RULES:
1. MUST start with "wbgmsst," and end with ", white background"
2. Preserve the core object - enhance, don't replace
3. Focus on what made the examples below successful

SUCCESSFUL TRANSFORMATIONS FROM TRAINING DATA:
"""
        
        # Add top training examples
        for i, match in enumerate(similar_matches[:4]):  # Top 4 for context
            ex = match.training_example
            system_prompt += f"""
Example {i+1} (Score: {ex.score:.3f}, Similarity: {match.similarity_score:.3f}):
  Before: "{ex.original}"
  After:  "{ex.optimized}"
"""
        
        system_prompt += f"""
NEW PROMPT TO OPTIMIZE: "{user_prompt}"

Analyze the successful patterns above and apply similar enhancement principles to create an optimization that will score >0.96.

OUTPUT: Only the optimized prompt - no explanations."""
        
        try:
            response = self._query_llama_fast(system_prompt)
            return self._clean_prompt(response)
        except Exception as e:
            print(f"   ❌ Optimization failed: {e}")
            return self._generate_fast_fallback(user_prompt, similar_matches)
    
    def _calculate_fast_confidence(self, original: str, optimized: str, 
                                 similar_matches: List[FastMatch]) -> float:
        """Fast confidence calculation based on vector similarity"""
        
        if not similar_matches:
            return 0.3
        
        # Base confidence on vector similarity to successful examples
        best_similarity = similar_matches[0].similarity_score
        base_confidence = best_similarity * 0.8  # Strong correlation with similarity
        
        # Boost for high-scoring similar examples
        avg_similar_score = statistics.mean([m.training_example.score for m in similar_matches])
        score_boost = min((avg_similar_score - 0.8) * 0.5, 0.15)  # Up to 0.15 boost
        
        # Diversity boost (more examples = higher confidence)
        diversity_boost = min(len(similar_matches) * 0.02, 0.1)
        
        # Data quality assessment
        quality_score = self._assess_local_data_quality(similar_matches)
        quality_boost = quality_score * 0.1
        
        total_confidence = base_confidence + max(score_boost, 0) + diversity_boost + quality_boost
        
        return min(max(total_confidence, 0.1), 0.95)
    
    def _assess_local_data_quality(self, matches: List[FastMatch]) -> float:
        """Assess quality of the matched training data"""
        if not matches:
            return 0.0
        
        # Score consistency (are all matches high-scoring?)
        scores = [m.training_example.score for m in matches]
        score_consistency = min(scores) / max(scores) if max(scores) > 0 else 0
        
        # Similarity consistency (are matches actually similar?)
        similarities = [m.similarity_score for m in matches]
        similarity_quality = statistics.mean(similarities)
        
        # Combine factors
        return (score_consistency + similarity_quality) / 2
    
    def _fallback_optimization(self, user_prompt: str, start_time: float) -> Dict:
        """High-quality fallback when no training data is available"""
        
        print("   🔧 Using intelligent fallback optimization...")
        
        # Use LLM to generate fallback without training data
        system_prompt = """You are an expert prompt optimizer for 3D generation.

Optimize the user's prompt to achieve maximum validation scores by:
1. Adding appropriate material and quality details
2. Including relevant craftsmanship descriptors
3. Enhancing visual clarity and appeal

RULES:
- Start with "wbgmsst," and end with ", white background"
- Preserve the core object
- Add contextually appropriate enhancements"""
        
        user_opt_prompt = f'Optimize: "{user_prompt}"'
        
        try:
            optimized = self._query_llama_fast(system_prompt + "\n\n" + user_opt_prompt)
            optimized = self._clean_prompt(optimized)
        except:
            # Ultimate fallback
            optimized = f"wbgmsst, premium-quality {user_prompt}, exquisite craftsmanship, white background"
        
        return {
            'original_prompt': user_prompt,
            'optimized_prompt': optimized,
            'confidence': 0.4,  # Lower confidence without training data
            'processing_time': time.time() - start_time,
            'similar_examples_count': 0,
            'best_similar_score': 0.0,
            'actual_validation_score': None,
            'data_quality_score': 0.0
        }
    
    def _generate_fast_fallback(self, user_prompt: str, similar_matches: List[FastMatch]) -> str:
        """Fast fallback using best available match"""
        if not similar_matches:
            return f"wbgmsst, premium-quality {user_prompt}, refined details, white background"
        
        # Use the transformation pattern from the best match
        best_match = similar_matches[0]
        best_example = best_match.training_example
        
        # Simple pattern application (preserve structure, change object)
        best_optimized = best_example.optimized
        
        # Replace original object with user's object in the successful pattern
        try:
            # Find the core enhancement pattern by removing the original object
            pattern_parts = best_optimized.replace(best_example.original, "{OBJECT}")
            final_prompt = pattern_parts.replace("{OBJECT}", user_prompt)
            return final_prompt
        except:
            return f"wbgmsst, high-quality {user_prompt}, premium finish, white background"
    
    def _validate_prompt(self, prompt: str) -> float:
        """Run actual validation"""
        try:
            print("   🔍 Running actual validation...")
            cmd = [sys.executable, "subnet_accurate_validator.py", prompt]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
            
            if result.returncode == 0:
                with open("subnet_validation_results.json", 'r') as f:
                    data = json.load(f)
                    return data.get("validation_engine_score", 0.0)
            return 0.0
        except Exception as e:
            print(f"   ❌ Validation failed: {e}")
            return 0.0
    
    def _clean_prompt(self, prompt: str) -> str:
        """Clean and format the prompt"""
        lines = prompt.split('\n')
        
        # Find wbgmsst line
        for line in lines:
            if 'wbgmsst' in line.lower():
                prompt = line.strip().replace('"', '')
                break
        else:
            # Use first substantial line
            for line in lines:
                if len(line.strip()) > 10:
                    prompt = line.strip().replace('"', '')
                    break
            else:
                prompt = f"wbgmsst, {prompt}, white background"
        
        # Ensure proper format
        if not prompt.startswith('wbgmsst'):
            prompt = f"wbgmsst, {prompt}"
        if not prompt.endswith('white background'):
            prompt = prompt.rstrip(', ') + ", white background"
        
        return prompt
    
    def _query_llama_fast(self, full_prompt: str) -> str:
        """Fast LLaMA query with optimized parameters"""
        data = {
            "model": self.model,
            "messages": [
                {"role": "user", "content": full_prompt}  # Simplified - no separate system/user
            ],
            "stream": False,
            "options": {
                "temperature": 0.6,
                "num_predict": 150,  # Shorter for speed
                "top_p": 0.9
            }
        }
        
        response = requests.post(f"{self.ollama_url}/api/chat", json=data, timeout=20)
        response.raise_for_status()
        return response.json()["message"]["content"].strip()

def main():
    """Command line interface"""
    if len(sys.argv) < 2:
        print("Usage: python smart_prompt_optimizer_fixed.py \"your prompt here\" [--validate]")
        print("\nOptions:")
        print("  --validate    Run actual validation (slow but accurate)")
        print("\nExample:")
        print("python smart_prompt_optimizer_fixed.py \"tall glass of layered lemonade\"")
        return
    
    user_prompt = sys.argv[1]
    use_validation = "--validate" in sys.argv
    
    print("🚀 OPTIMIZED PROMPT OPTIMIZER - PERFORMANCE FIXED")
    print("=" * 60)
    print("✅ Vector embeddings for instant similarity")
    print("✅ Pre-computed embeddings for fast inference")
    print("✅ Streamlined LLM calls")
    print("✅ Data quality assessment")
    print("=" * 60)
    
    try:
        optimizer = OptimizedPromptOptimizer("rl_checkpoints_v3/prompt_score_log.csv")
        result = optimizer.optimize(user_prompt, use_validation=use_validation)
        
        print(f"\n📋 OPTIMIZATION COMPLETE:")
        print(f"   Original: {result['original_prompt']}")
        print(f"   Optimized: {result['optimized_prompt']}")
        print(f"   Confidence: {result['confidence']:.1%} (vector-informed)")
        print(f"   Similar examples: {result['similar_examples_count']}")
        print(f"   Best similar score: {result['best_similar_score']:.3f}")
        print(f"   Data quality: {result['data_quality_score']:.2f}")
        print(f"   Processing time: {result['processing_time']:.2f}s")
        
        if result['actual_validation_score'] is not None:
            print(f"   🎯 Actual validation: {result['actual_validation_score']:.3f}")
        
    except KeyboardInterrupt:
        print(f"\n⚠️ Interrupted by user")
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 