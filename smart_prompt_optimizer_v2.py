#!/usr/bin/env python3
"""
Smart Prompt Optimizer V2 - LEARNED PATTERN OPTIMIZATION
=========================================================
🧠 LLM actually learns from training data patterns for unseen examples
⚡ Still uses vector embeddings for performance
🎯 Fallback LLM is trained on successful optimization patterns
📊 Confidence based on pattern similarity to training successes

Key Improvement: When no similar examples exist, the LLM uses learned 
patterns from ALL successful training examples, not generic descriptions.
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
import re
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
    embedding: Optional[np.ndarray] = None

@dataclass
class FastMatch:
    """Fast similarity match using vector embeddings"""
    training_example: TrainingExample
    similarity_score: float

@dataclass
class LearnedPattern:
    """A learned optimization pattern extracted from training data"""
    pattern_type: str
    description: str
    example_original: str
    example_optimized: str
    success_rate: float
    avg_score: float
    frequency: int

class PatternLearner:
    """Learns optimization patterns from successful training examples"""
    
    def __init__(self, training_examples: List[TrainingExample]):
        self.training_examples = training_examples
        self.learned_patterns = []
        self._extract_patterns()
    
    def _extract_patterns(self):
        """Extract patterns from successful examples"""
        if not self.training_examples:
            return
        
        print("🧠 Learning optimization patterns from training data...")
        
        # Group examples by success level
        high_success = [ex for ex in self.training_examples if ex.score >= 0.9]
        medium_success = [ex for ex in self.training_examples if 0.85 <= ex.score < 0.9]
        
        # Extract enhancement patterns
        enhancement_patterns = self._extract_enhancement_patterns(high_success)
        structure_patterns = self._extract_structure_patterns(high_success + medium_success)
        material_patterns = self._extract_material_patterns(high_success)
        
        self.learned_patterns = enhancement_patterns + structure_patterns + material_patterns
        
        print(f"   📖 Learned {len(self.learned_patterns)} optimization patterns")
        for pattern in self.learned_patterns[:3]:  # Show top 3
            print(f"     • {pattern.pattern_type}: {pattern.description} (score: {pattern.avg_score:.3f})")
    
    def _extract_enhancement_patterns(self, examples: List[TrainingExample]) -> List[LearnedPattern]:
        """Extract what types of enhancements lead to high scores"""
        patterns = []
        
        # Common enhancement words/phrases that appear in high-scoring examples
        enhancement_keywords = {
            'material_quality': ['precision', 'artisanal', 'premium', 'exquisite', 'masterwork'],
            'visual_detail': ['vibrant', 'lustrous', 'gleaming', 'shimmering', 'detailed'],
            'craftsmanship': ['hand-crafted', 'intricately', 'delicate', 'refined', 'elegant'],
            'setting_context': ['suspended', 'resting', 'placed', 'nestled', 'positioned']
        }
        
        for category, keywords in enhancement_keywords.items():
            matching_examples = []
            scores = []
            
            for example in examples:
                opt_lower = example.optimized.lower()
                if any(keyword in opt_lower for keyword in keywords):
                    matching_examples.append(example)
                    scores.append(example.score)
            
            if matching_examples:
                avg_score = statistics.mean(scores)
                best_example = max(matching_examples, key=lambda x: x.score)
                
                pattern = LearnedPattern(
                    pattern_type=category,
                    description=f"Use {category.replace('_', ' ')} enhancing words like: {', '.join(keywords[:3])}",
                    example_original=best_example.original,
                    example_optimized=best_example.optimized,
                    success_rate=len(matching_examples) / len(examples),
                    avg_score=avg_score,
                    frequency=len(matching_examples)
                )
                patterns.append(pattern)
        
        return patterns
    
    def _extract_structure_patterns(self, examples: List[TrainingExample]) -> List[LearnedPattern]:
        """Extract structural patterns from successful optimizations"""
        patterns = []
        
        # Analyze common structural elements
        for example in examples:
            if not example.optimized.startswith('wbgmsst'):
                continue
                
            # Extract the middle part (between wbgmsst and white background)
            content = example.optimized[8:]  # Remove 'wbgmsst,'
            if ', white background' in content:
                content = content.replace(', white background', '')
            
            # Look for patterns like "a [adjective] [object] [description]"
            if ' filled with ' in content:
                patterns.append(LearnedPattern(
                    pattern_type="filled_structure",
                    description="Use 'filled with [detailed description]' pattern",
                    example_original=example.original,
                    example_optimized=example.optimized,
                    success_rate=1.0,
                    avg_score=example.score,
                    frequency=1
                ))
            
            if ' suspended in ' in content or ' floating ' in content:
                patterns.append(LearnedPattern(
                    pattern_type="floating_element",
                    description="Add suspended/floating elements for visual interest",
                    example_original=example.original,
                    example_optimized=example.optimized,
                    success_rate=1.0,
                    avg_score=example.score,
                    frequency=1
                ))
        
        return patterns
    
    def _extract_material_patterns(self, examples: List[TrainingExample]) -> List[LearnedPattern]:
        """Extract material-specific patterns"""
        patterns = []
        
        # Track what materials get enhanced how
        material_enhancements = {}
        
        for example in examples:
            original_words = set(example.original.lower().split())
            optimized_words = set(example.optimized.lower().split())
            
            # Find material words
            materials = original_words.intersection({
                'glass', 'crystal', 'metal', 'steel', 'silver', 'gold', 'wood', 'stone'
            })
            
            for material in materials:
                if material not in material_enhancements:
                    material_enhancements[material] = []
                
                # Find enhancement words that were added
                enhancement_words = optimized_words - original_words
                material_enhancements[material].append({
                    'enhancements': list(enhancement_words),
                    'score': example.score,
                    'example': example
                })
        
        # Create patterns for materials with enough data
        for material, data in material_enhancements.items():
            if len(data) >= 2:  # Need at least 2 examples
                avg_score = statistics.mean([d['score'] for d in data])
                best_example = max(data, key=lambda x: x['score'])['example']
                
                # Find most common enhancements
                all_enhancements = []
                for d in data:
                    all_enhancements.extend(d['enhancements'])
                
                common_enhancements = [word for word in set(all_enhancements) 
                                     if all_enhancements.count(word) >= 2][:3]
                
                if common_enhancements:
                    pattern = LearnedPattern(
                        pattern_type=f"{material}_enhancement",
                        description=f"For {material} objects, add: {', '.join(common_enhancements)}",
                        example_original=best_example.original,
                        example_optimized=best_example.optimized,
                        success_rate=1.0,
                        avg_score=avg_score,
                        frequency=len(data)
                    )
                    patterns.append(pattern)
        
        return patterns
    
    def get_applicable_patterns(self, prompt: str) -> List[LearnedPattern]:
        """Get patterns that might apply to the given prompt"""
        applicable = []
        prompt_lower = prompt.lower()
        prompt_words = set(prompt_lower.split())
        
        for pattern in self.learned_patterns:
            # Material-specific patterns
            if pattern.pattern_type.endswith('_enhancement'):
                material = pattern.pattern_type.replace('_enhancement', '')
                if material in prompt_lower:
                    applicable.append(pattern)
            
            # Structure patterns - generally applicable
            elif pattern.pattern_type in ['filled_structure', 'floating_element']:
                applicable.append(pattern)
            
            # Enhancement patterns - always applicable
            elif pattern.pattern_type in ['material_quality', 'visual_detail', 'craftsmanship', 'setting_context']:
                applicable.append(pattern)
        
        # Sort by success rate and score
        applicable.sort(key=lambda x: (x.success_rate, x.avg_score), reverse=True)
        return applicable[:5]  # Top 5 most applicable

class FastTrainingDataLoader:
    """High-performance training data loader with pattern learning"""
    
    def __init__(self, csv_file: str = "prompt_score_log.csv", 
                 cache_file: str = "training_embeddings_v2.pkl"):
        self.csv_file = Path(csv_file)
        self.cache_file = Path(cache_file)
        self.training_examples = []
        self.high_score_examples = []
        self.pattern_learner = None
        
        # Initialize sentence transformer
        try:
            self.encoder = SentenceTransformer('all-MiniLM-L6-v2')
            print("✅ SentenceTransformer loaded successfully")
        except ImportError:
            print("❌ sentence-transformers not available")
            self.encoder = None
            
        self._load_training_data_with_embeddings()
        self._learn_patterns()
    
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
            return
        
        if not self.encoder:
            print(f"❌ Cannot compute embeddings without sentence-transformers")
            return
            
        print(f"📊 Loading training data from {self.csv_file}")
        
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
                    
                    if example.score >= 0.85:
                        self.high_score_examples.append(example)
                        
                except (ValueError, KeyError) as e:
                    print(f"   ⚠️ Skipping malformed row: {e}")
        
        # Batch compute embeddings
        if examples_to_encode and self.encoder:
            print(f"   🧮 Computing {len(examples_to_encode)} embeddings...")
            embeddings = self.encoder.encode(examples_to_encode, batch_size=32, show_progress_bar=True)
            
            for example, embedding in zip(self.training_examples, embeddings):
                example.embedding = embedding
        
        print(f"   📈 Loaded {len(self.training_examples)} training examples")
        print(f"   💎 Found {len(self.high_score_examples)} high-scoring examples (≥0.85)")
        
        self.high_score_examples.sort(key=lambda x: x.score, reverse=True)
        self._cache_embeddings()
    
    def _cache_embeddings(self):
        """Cache embeddings to disk"""
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
    
    def _learn_patterns(self):
        """Learn patterns from training data"""
        if self.high_score_examples:
            self.pattern_learner = PatternLearner(self.high_score_examples)
        else:
            print("⚠️ No high-scoring examples to learn patterns from")

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
            return []
        
        print(f"   ⚡ Finding matches using vector similarity...")
        start_time = time.time()
        
        user_embedding = self.encoder.encode([user_prompt])[0]
        
        training_embeddings = []
        valid_examples = []
        
        for example in training_examples:
            if example.embedding is not None:
                training_embeddings.append(example.embedding)
                valid_examples.append(example)
        
        if not training_embeddings:
            return []
        
        training_matrix = np.array(training_embeddings)
        similarities = cosine_similarity([user_embedding], training_matrix)[0]
        
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        
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

class LearnedPatternOptimizer:
    """High-performance optimizer that actually learns from training data"""
    
    def __init__(self, training_csv: str = "prompt_score_log.csv", 
                 ollama_url: str = "http://localhost:11434"):
        self.ollama_url = ollama_url
        self.model = "llama3.2:3b"
        
        # Load training data with pattern learning
        self.training_loader = FastTrainingDataLoader(training_csv)
        self.similarity_engine = VectorSimilarityEngine()
        
        print(f"🧠 LEARNED PATTERN OPTIMIZER READY")
        print(f"   📊 Training examples: {len(self.training_loader.training_examples)}")
        print(f"   💎 High-scoring examples: {len(self.training_loader.high_score_examples)}")
        if self.training_loader.pattern_learner:
            print(f"   🎓 Learned patterns: {len(self.training_loader.pattern_learner.learned_patterns)}")
    
    def optimize(self, user_prompt: str, use_validation: bool = False) -> Dict:
        """Optimize using learned patterns for both similar and unseen examples"""
        start_time = time.time()
        
        print(f"\n🎯 OPTIMIZING: '{user_prompt}'")
        
        # Try vector similarity first
        similar_matches = self.similarity_engine.find_similar_examples_fast(
            user_prompt, self.training_loader.high_score_examples, top_k=5
        )
        
        if similar_matches:
            # Use similar examples
            print(f"   📊 Found {len(similar_matches)} similar training examples")
            optimized_prompt = self._optimize_with_similar_examples(user_prompt, similar_matches)
            confidence = self._calculate_similarity_confidence(similar_matches)
        else:
            # Use learned patterns for unseen examples
            print(f"   🧠 No similar examples - using learned patterns")
            optimized_prompt = self._optimize_with_learned_patterns(user_prompt)
            confidence = self._calculate_pattern_confidence(user_prompt)
        
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
            'optimization_method': 'similarity' if similar_matches else 'learned_patterns'
        }
        
        print(f"✅ RESULT: {optimized_prompt}")
        print(f"⏱️  Time: {processing_time:.2f}s | Confidence: {confidence:.1%} | Method: {result['optimization_method']}")
        
        return result
    
    def _optimize_with_similar_examples(self, user_prompt: str, matches: List[FastMatch]) -> str:
        """Optimize using similar training examples"""
        
        system_prompt = """You are an expert prompt optimizer with proven training data.

Analyze the successful transformations below and apply those same principles to the new prompt.

PROVEN SUCCESSFUL TRANSFORMATIONS:
"""
        
        for i, match in enumerate(matches[:3]):
            ex = match.training_example
            system_prompt += f"""
Example {i+1} (Score: {ex.score:.3f}, Similarity: {match.similarity_score:.3f}):
  Before: "{ex.original}"
  After:  "{ex.optimized}"
"""
        
        system_prompt += f"""
NEW PROMPT: "{user_prompt}"

Apply the successful patterns above. Must start with "wbgmsst," and end with ", white background".
OUTPUT: Only the optimized prompt."""
        
        try:
            response = self._query_llama_fast(system_prompt)
            return self._clean_prompt(response)
        except Exception as e:
            print(f"   ❌ Optimization failed: {e}")
            return f"wbgmsst, {user_prompt}, white background"
    
    def _optimize_with_learned_patterns(self, user_prompt: str) -> str:
        """Optimize using learned patterns from training data"""
        
        if not self.training_loader.pattern_learner:
            return self._fallback_optimization(user_prompt)
        
        # Get applicable patterns
        applicable_patterns = self.training_loader.pattern_learner.get_applicable_patterns(user_prompt)
        
        if not applicable_patterns:
            return self._fallback_optimization(user_prompt)
        
        print(f"   🎓 Applying {len(applicable_patterns)} learned patterns")
        
        # Build system prompt with learned patterns
        system_prompt = f"""You are an expert prompt optimizer trained on validation data.

I have learned these optimization patterns from successful high-scoring examples (0.85-0.92 validation scores):

LEARNED SUCCESS PATTERNS:
"""
        
        for i, pattern in enumerate(applicable_patterns[:4]):
            system_prompt += f"""
Pattern {i+1}: {pattern.description}
  Success rate: {pattern.success_rate:.1%}, Avg score: {pattern.avg_score:.3f}
  Example transformation:
    Before: "{pattern.example_original}"
    After:  "{pattern.example_optimized}"
"""
        
        system_prompt += f"""
NEW PROMPT TO OPTIMIZE: "{user_prompt}"

Apply the learned patterns above to create a high-scoring optimization. The patterns above are proven to achieve 0.85-0.92 validation scores.

RULES:
- Must start with "wbgmsst," and end with ", white background"
- Apply multiple applicable patterns
- Focus on what made the examples above score highly

OUTPUT: Only the optimized prompt."""
        
        try:
            response = self._query_llama_fast(system_prompt)
            return self._clean_prompt(response)
        except Exception as e:
            print(f"   ❌ Pattern optimization failed: {e}")
            return self._fallback_optimization(user_prompt)
    
    def _calculate_similarity_confidence(self, matches: List[FastMatch]) -> float:
        """Calculate confidence based on similarity matches"""
        if not matches:
            return 0.3
        
        best_similarity = matches[0].similarity_score
        avg_score = statistics.mean([m.training_example.score for m in matches])
        
        confidence = (best_similarity * 0.6) + ((avg_score - 0.5) * 0.4)
        return min(max(confidence, 0.1), 0.95)
    
    def _calculate_pattern_confidence(self, user_prompt: str) -> float:
        """Calculate confidence based on learned patterns"""
        if not self.training_loader.pattern_learner:
            return 0.4
        
        applicable_patterns = self.training_loader.pattern_learner.get_applicable_patterns(user_prompt)
        
        if not applicable_patterns:
            return 0.4
        
        # Confidence based on pattern quality and applicability
        avg_pattern_score = statistics.mean([p.avg_score for p in applicable_patterns])
        avg_success_rate = statistics.mean([p.success_rate for p in applicable_patterns])
        pattern_count_factor = min(len(applicable_patterns) / 5, 1.0)
        
        confidence = (avg_pattern_score - 0.5) * 0.5 + avg_success_rate * 0.3 + pattern_count_factor * 0.2
        
        return min(max(confidence, 0.5), 0.85)  # Higher baseline for learned patterns
    
    def _fallback_optimization(self, user_prompt: str) -> str:
        """Ultimate fallback"""
        return f"wbgmsst, premium-quality {user_prompt}, exquisite craftsmanship, white background"
    
    def _clean_prompt(self, prompt: str) -> str:
        """Clean and format the prompt"""
        lines = prompt.split('\n')
        
        for line in lines:
            if 'wbgmsst' in line.lower():
                prompt = line.strip().replace('"', '')
                break
        else:
            for line in lines:
                if len(line.strip()) > 10:
                    prompt = line.strip().replace('"', '')
                    break
        
        if not prompt.startswith('wbgmsst'):
            prompt = f"wbgmsst, {prompt}"
        if not prompt.endswith('white background'):
            prompt = prompt.rstrip(', ') + ", white background"
        
        return prompt
    
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
    
    def _query_llama_fast(self, full_prompt: str) -> str:
        """Fast LLaMA query"""
        data = {
            "model": self.model,
            "messages": [
                {"role": "user", "content": full_prompt}
            ],
            "stream": False,
            "options": {
                "temperature": 0.6,
                "num_predict": 150,
                "top_p": 0.9
            }
        }
        
        response = requests.post(f"{self.ollama_url}/api/chat", json=data, timeout=20)
        response.raise_for_status()
        return response.json()["message"]["content"].strip()

def main():
    """Command line interface"""
    if len(sys.argv) < 2:
        print("Usage: python smart_prompt_optimizer_v2.py \"your prompt here\" [--validate]")
        return
    
    user_prompt = sys.argv[1]
    use_validation = "--validate" in sys.argv
    
    print("🧠 LEARNED PATTERN OPTIMIZER - TRAINS ON SUCCESS PATTERNS")
    print("=" * 60)
    print("✅ Learns actual patterns from training data")
    print("✅ Handles unseen examples intelligently")
    print("✅ Vector similarity for known patterns")
    print("✅ Learned patterns for novel objects")
    print("=" * 60)
    
    try:
        optimizer = LearnedPatternOptimizer("rl_checkpoints_v3/prompt_score_log.csv")
        result = optimizer.optimize(user_prompt, use_validation=use_validation)
        
        print(f"\n📋 OPTIMIZATION COMPLETE:")
        print(f"   Original: {result['original_prompt']}")
        print(f"   Optimized: {result['optimized_prompt']}")
        print(f"   Method: {result['optimization_method']}")
        print(f"   Confidence: {result['confidence']:.1%}")
        print(f"   Processing time: {result['processing_time']:.2f}s")
        
    except KeyboardInterrupt:
        print(f"\n⚠️ Interrupted by user")
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 