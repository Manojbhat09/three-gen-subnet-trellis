#!/usr/bin/env python3
"""
Smart Prompt Optimizer - Training-Informed Inference
===================================================
🧠 Uses actual training data to inform optimization decisions
⚡ No redundant RL agent - LLM makes all strategic decisions
🎯 Semantic pattern matching without hardcoded keywords
📊 Real confidence scoring based on training data similarity

Core Design:
1. Load actual prompt-score pairs from training CSV logs
2. Use LLM to determine optimization strategy based on semantic similarity
3. Apply learned patterns semantically, not via keyword matching
4. Calculate confidence based on similarity to successful training examples
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
import re
import subprocess

@dataclass
class TrainingExample:
    """A training example with actual validation scores"""
    original: str
    optimized: str
    score: float
    episode: int
    step: int
    strategy: str

@dataclass
class SemanticMatch:
    """Semantic similarity match with training data"""
    training_example: TrainingExample
    similarity_score: float
    applicable_patterns: List[str]

class TrainingDataLoader:
    """Loads and processes actual training data from CSV logs"""
    
    def __init__(self, csv_file: str = "prompt_score_log.csv"):
        self.csv_file = Path(csv_file)
        self.training_examples = []
        self.high_score_examples = []
        self._load_training_data()
    
    def _load_training_data(self):
        """Load training data from CSV logs"""
        
        if not self.csv_file.exists():
            print(f"⚠️ No training data found at {self.csv_file}")
            print(f"💡 Run your training script first to generate actual data")
            return
        
        print(f"📊 Loading training data from {self.csv_file}")
        
        with open(self.csv_file, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    example = TrainingExample(
                        original=row['original_prompt'],
                        optimized=row['custom_prompt'],
                        score=float(row['score']),
                        episode=int(row['episode']),
                        step=int(row['step']),
                        strategy=row.get('strategy', 'unknown')
                    )
                    
                    self.training_examples.append(example)
                    
                    # Keep high-scoring examples (>0.85)
                    if example.score >= 0.85:
                        self.high_score_examples.append(example)
                        
                except (ValueError, KeyError) as e:
                    print(f"   ⚠️ Skipping malformed row: {e}")
        
        print(f"   📈 Loaded {len(self.training_examples)} training examples")
        print(f"   💎 Found {len(self.high_score_examples)} high-scoring examples (≥0.85)")
        
        # Sort high-scoring examples by score
        self.high_score_examples.sort(key=lambda x: x.score, reverse=True)

class SemanticSimilarityEngine:
    """Semantic similarity engine using LLM instead of hardcoded keywords"""
    
    def __init__(self, ollama_url: str = "http://localhost:11434"):
        self.ollama_url = ollama_url
        self.model = "llama3.2:3b"
    
    def find_similar_training_examples(self, user_prompt: str, 
                                     training_examples: List[TrainingExample], 
                                     top_k: int = 5) -> List[SemanticMatch]:
        """Find semantically similar training examples using LLM"""
        
        print(f"   🔍 Finding semantic matches for: '{user_prompt}'")
        
        # Create batches of examples for comparison
        matches = []
        
        # Use LLM to assess semantic similarity
        for example in training_examples:
            similarity = self._assess_semantic_similarity(user_prompt, example.original)
            if similarity > 0.3:  # Threshold for relevance
                patterns = self._extract_applicable_patterns(user_prompt, example)
                matches.append(SemanticMatch(
                    training_example=example,
                    similarity_score=similarity,
                    applicable_patterns=patterns
                ))
        
        # Sort by similarity and return top matches
        matches.sort(key=lambda x: x.similarity_score, reverse=True)
        return matches[:top_k]
    
    def _assess_semantic_similarity(self, prompt1: str, prompt2: str) -> float:
        """Use LLM to assess semantic similarity between prompts"""
        
        system_prompt = """You are an expert at assessing semantic similarity between object descriptions.

Rate the semantic similarity between two prompts on a scale of 0.0 to 1.0:
- 1.0: Nearly identical objects (e.g., "red apple" vs "crimson apple")
- 0.8: Same object type, different details (e.g., "glass cup" vs "crystal wine glass")
- 0.6: Similar object category (e.g., "wooden chair" vs "leather sofa")
- 0.4: Related but different (e.g., "car" vs "bicycle")
- 0.2: Loosely related (e.g., "tree" vs "flower")
- 0.0: Completely unrelated (e.g., "mountain" vs "smartphone")

Respond with ONLY the similarity score (0.0-1.0), no explanation."""
        
        user_prompt = f"""Prompt 1: "{prompt1}"
Prompt 2: "{prompt2}"

Similarity score:"""
        
        try:
            response = self._query_llama(system_prompt, user_prompt, temperature=0.1)
            # Extract numeric score
            score_match = re.search(r'(\d+\.?\d*)', response.strip())
            if score_match:
                score = float(score_match.group(1))
                return min(max(score, 0.0), 1.0)  # Clamp to [0, 1]
            return 0.0
        except Exception as e:
            print(f"     ⚠️ Similarity assessment failed: {e}")
            return 0.0
    
    def _extract_applicable_patterns(self, user_prompt: str, training_example: TrainingExample) -> List[str]:
        """Extract patterns that might apply to the user prompt"""
        
        system_prompt = """You are an expert at identifying applicable enhancement patterns.

Given a user prompt and a successful training example, identify which enhancement patterns from the training example could apply to the user prompt.

Focus on transferable techniques like:
- Material quality improvements (e.g., "crystal-clear", "polished")
- Craftsmanship details (e.g., "handcrafted", "precision-made")
- Visual enhancements (e.g., "lustrous", "gleaming")
- Contextual details (e.g., lighting, setting)

List 2-3 applicable patterns, each on a new line starting with "-"."""
        
        user_prompt_text = f"""User prompt: "{user_prompt}"
Training example:
  Original: "{training_example.original}"
  Optimized: "{training_example.optimized}"
  Score: {training_example.score:.3f}

Applicable patterns:"""
        
        try:
            response = self._query_llama(system_prompt, user_prompt_text, temperature=0.3)
            # Extract patterns (lines starting with -)
            patterns = []
            for line in response.split('\n'):
                line = line.strip()
                if line.startswith('-'):
                    patterns.append(line[1:].strip())
            return patterns[:3]  # Limit to top 3
        except Exception as e:
            print(f"     ⚠️ Pattern extraction failed: {e}")
            return []
    
    def _query_llama(self, system_prompt: str, user_prompt: str, temperature: float = 0.5) -> str:
        """Query LLaMA"""
        data = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": 100
            }
        }
        
        response = requests.post(f"{self.ollama_url}/api/chat", json=data, timeout=15)
        response.raise_for_status()
        return response.json()["message"]["content"].strip()

class IntelligentPromptOptimizer:
    """Intelligent optimizer that uses training data semantically"""
    
    def __init__(self, training_csv: str = "prompt_score_log.csv", 
                 ollama_url: str = "http://localhost:11434"):
        self.ollama_url = ollama_url
        self.model = "llama3.2:3b"
        
        # Load actual training data
        self.training_loader = TrainingDataLoader(training_csv)
        self.similarity_engine = SemanticSimilarityEngine(ollama_url)
        
        print(f"🧠 INTELLIGENT PROMPT OPTIMIZER READY")
        print(f"   📊 Training examples: {len(self.training_loader.training_examples)}")
        print(f"   💎 High-scoring examples: {len(self.training_loader.high_score_examples)}")
    
    def optimize(self, user_prompt: str, use_validation: bool = False) -> Dict:
        """
        Intelligently optimize prompt using training data
        
        Args:
            user_prompt: Prompt to optimize
            use_validation: Whether to run actual validation (slow but accurate)
        
        Returns:
            Optimization result with training-informed confidence
        """
        start_time = time.time()
        
        print(f"\n🎯 OPTIMIZING: '{user_prompt}'")
        
        # Find semantically similar training examples
        similar_matches = self.similarity_engine.find_similar_training_examples(
            user_prompt, self.training_loader.high_score_examples, top_k=5
        )
        
        if not similar_matches:
            print("   ⚠️ No similar training examples found - using fallback")
            return self._fallback_optimization(user_prompt, start_time)
        
        print(f"   📊 Found {len(similar_matches)} similar training examples")
        for i, match in enumerate(similar_matches[:3]):
            print(f"     {i+1}. {match.training_example.original} (sim: {match.similarity_score:.2f}, score: {match.training_example.score:.3f})")
        
        # Generate optimization using best training examples
        optimized_prompt = self._generate_intelligent_optimization(user_prompt, similar_matches)
        
        # Calculate training-informed confidence
        confidence = self._calculate_training_confidence(user_prompt, optimized_prompt, similar_matches)
        
        # Optionally run actual validation
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
            'actual_validation_score': actual_score
        }
        
        print(f"✅ RESULT: {optimized_prompt}")
        print(f"⏱️  Time: {processing_time:.2f}s | Confidence: {confidence:.1%}")
        if actual_score:
            print(f"🎯 Validation Score: {actual_score:.3f}")
        
        return result
    
    def _generate_intelligent_optimization(self, user_prompt: str, 
                                         similar_matches: List[SemanticMatch]) -> str:
        """Generate optimization using semantic training examples"""
        
        # Build intelligent system prompt with similar examples
        system_prompt = f"""You are an expert prompt optimizer with access to training data showing what optimizations achieve high scores (>0.9).

TARGET: Optimize the user's prompt to achieve a validation score >0.96.

CRITICAL RULES:
1. MUST start with "wbgmsst," and end with ", white background"
2. Preserve the core object - enhance, don't replace
3. Use insights from the high-scoring training examples below

HIGH-SCORING TRAINING EXAMPLES (SIMILAR TO USER'S PROMPT):
"""
        
        # Add the most relevant training examples
        for i, match in enumerate(similar_matches[:3]):
            example = match.training_example
            system_prompt += f"""
Example {i+1} (Score: {example.score:.3f}, Similarity: {match.similarity_score:.2f}):
  Original: "{example.original}"
  Optimized: "{example.optimized}"
  Key Patterns: {', '.join(match.applicable_patterns)}
"""
        
        system_prompt += """
STRATEGY: Analyze the successful patterns above and apply similar enhancement principles to the user's prompt. Focus on what made these examples score highly.

OUTPUT: Only the optimized prompt - no explanations."""
        
        user_optimization_prompt = f"""OPTIMIZE: "{user_prompt}"

Apply the successful patterns from the training examples above to create an optimization that will score >0.96."""
        
        try:
            response = self._query_llama(system_prompt, user_optimization_prompt)
            return self._clean_prompt(response)
        except Exception as e:
            print(f"   ❌ Optimization failed: {e}")
            return self._generate_pattern_fallback(user_prompt, similar_matches)
    
    def _calculate_training_confidence(self, original: str, optimized: str, 
                                     similar_matches: List[SemanticMatch]) -> float:
        """Calculate confidence based on similarity to successful training examples"""
        
        if not similar_matches:
            return 0.3
        
        # Base confidence on similarity to high-scoring examples
        best_match = similar_matches[0]
        base_confidence = best_match.similarity_score * 0.7  # Start with similarity
        
        # Boost confidence based on training example scores
        avg_similar_score = statistics.mean([m.training_example.score for m in similar_matches])
        score_boost = min(avg_similar_score - 0.5, 0.4)  # Up to 0.4 boost for high scores
        
        # Boost for having multiple similar examples
        diversity_boost = min(len(similar_matches) * 0.05, 0.2)  # Up to 0.2 for many examples
        
        # Quality indicators (still useful as a secondary signal)
        quality_words = ['artisanal', 'precision', 'premium', 'exquisite', 'crystal-clear', 'flawless']
        quality_boost = min(sum(1 for word in quality_words if word in optimized.lower()) * 0.03, 0.1)
        
        total_confidence = base_confidence + score_boost + diversity_boost + quality_boost
        
        return min(max(total_confidence, 0.1), 0.95)  # Clamp to reasonable range
    
    def _fallback_optimization(self, user_prompt: str, start_time: float) -> Dict:
        """Fallback when no training data is available"""
        
        # Simple contextual enhancement
        prompt_lower = user_prompt.lower()
        if any(word in prompt_lower for word in ['glass', 'crystal', 'transparent']):
            optimized = f"wbgmsst, crystal-clear {user_prompt}, pristine transparency, white background"
        elif any(word in prompt_lower for word in ['metal', 'steel', 'weapon']):
            optimized = f"wbgmsst, precision-forged {user_prompt}, masterwork craftsmanship, white background"
        else:
            optimized = f"wbgmsst, premium-quality {user_prompt}, exquisite detail, white background"
        
        return {
            'original_prompt': user_prompt,
            'optimized_prompt': optimized,
            'confidence': 0.4,  # Low confidence without training data
            'processing_time': time.time() - start_time,
            'similar_examples_count': 0,
            'best_similar_score': 0.0,
            'actual_validation_score': None
        }
    
    def _generate_pattern_fallback(self, user_prompt: str, similar_matches: List[SemanticMatch]) -> str:
        """Generate fallback using patterns from similar matches"""
        if not similar_matches:
            return f"wbgmsst, premium-quality {user_prompt}, exquisite craftsmanship, white background"
        
        # Use patterns from best match
        best_match = similar_matches[0]
        patterns = best_match.applicable_patterns
        
        if patterns:
            # Apply the first applicable pattern
            pattern = patterns[0]
            return f"wbgmsst, {pattern} {user_prompt}, white background"
        else:
            return f"wbgmsst, high-quality {user_prompt}, refined details, white background"
    
    def _validate_prompt(self, prompt: str) -> float:
        """Run actual validation (optional, slow)"""
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
            # Use first non-empty line
            for line in lines:
                if line.strip():
                    prompt = line.strip().replace('"', '')
                    break
        
        # Ensure proper format
        if not prompt.startswith('wbgmsst'):
            prompt = f"wbgmsst, {prompt}"
        if not prompt.endswith('white background'):
            prompt = prompt.rstrip(', ') + ", white background"
        
        return prompt
    
    def _query_llama(self, system_prompt: str, user_prompt: str, temperature: float = 0.6) -> str:
        """Query LLaMA"""
        data = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": 200
            }
        }
        
        response = requests.post(f"{self.ollama_url}/api/chat", json=data, timeout=25)
        response.raise_for_status()
        return response.json()["message"]["content"].strip()


def main():
    """Command line interface"""
    if len(sys.argv) < 2:
        print("Usage: python smart_prompt_optimizer.py \"your prompt here\" [--validate]")
        print("\nOptions:")
        print("  --validate    Run actual validation (slow but accurate)")
        print("\nExample:")
        print("python smart_prompt_optimizer.py \"tall glass of layered lemonade\"")
        return
    
    user_prompt = sys.argv[1]
    use_validation = "--validate" in sys.argv
    
    print("🧠 SMART PROMPT OPTIMIZER - TRAINING-INFORMED INFERENCE")
    print("=" * 60)
    print("✅ Uses actual training data semantically")
    print("✅ No redundant RL agent")
    print("✅ LLM-based semantic similarity")
    print("✅ Training-informed confidence scoring")
    print("=" * 60)
    
    try:
        optimizer = IntelligentPromptOptimizer()
        result = optimizer.optimize(user_prompt, use_validation=use_validation)
        
        print(f"\n📋 OPTIMIZATION COMPLETE:")
        print(f"   Original: {result['original_prompt']}")
        print(f"   Optimized: {result['optimized_prompt']}")
        print(f"   Confidence: {result['confidence']:.1%} (training-informed)")
        print(f"   Similar examples: {result['similar_examples_count']}")
        print(f"   Best similar score: {result['best_similar_score']:.3f}")
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