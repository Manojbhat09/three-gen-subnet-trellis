#!/usr/bin/env python3
"""
RL + LLaMA Optimizer v5.0 - FIXED ARCHITECTURE
==============================================
✅ Addresses core architectural problems from v3.1_9
✅ Maintains high-score objective (>0.96)
✅ Uses actual validation scores from subnet_accurate_validator.py
✅ Semantic embedding-based state representation
✅ LLM-based pattern discovery instead of hardcoded keywords
✅ Streamlined RL agent that focuses on strategy coordination

Core Fixes:
1. Replace hardcoded keyword patterns with LLM-based semantic pattern extraction
2. Replace simplistic state with semantic embeddings of prompt content
3. Keep RL agent for high-level strategy coordination while letting LLM do detailed work
4. Use actual validation scores for training and golden example extraction
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import random
from collections import deque, namedtuple
import json
import subprocess
import sys
import time
import pickle
import os
import requests
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, asdict
from pathlib import Path
import signal
import datetime
import statistics
import csv
from sentence_transformers import SentenceTransformer

Experience = namedtuple('Experience', ['state', 'action', 'reward', 'next_state', 'done'])

@dataclass
class LLaMAInstruction:
    """High-level strategy instructions for LLaMA"""
    strategy_name: str
    creativity_level: float
    focus_directive: str  # High-level instruction, not hardcoded categories
    risk_tolerance: str
    enhancement_approach: str

@dataclass
class SemanticPattern:
    """LLM-discovered semantic pattern"""
    pattern_description: str
    applicable_contexts: List[str]
    enhancement_principle: str
    success_score: float
    example_transformations: List[Dict]

class SemanticStateEncoder:
    """Encodes prompt content into semantic embeddings instead of keyword matching"""
    
    def __init__(self):
        # Use a lightweight sentence transformer for semantic understanding
        self.encoder = SentenceTransformer('all-MiniLM-L6-v2')
        print("🧠 Semantic encoder initialized")
    
    def encode_prompt(self, prompt: str) -> np.ndarray:
        """Encode prompt into semantic embedding"""
        # Get semantic embedding
        embedding = self.encoder.encode([prompt])[0]
        
        # Add some basic structural features
        structural_features = np.array([
            len(prompt.split()),  # Word count
            len(prompt),  # Character count
            prompt.count(','),  # Complexity indicator
            1.0 if any(word in prompt.lower() for word in ['glass', 'crystal', 'transparent']) else 0.0,
            1.0 if any(word in prompt.lower() for word in ['metal', 'steel', 'iron', 'gold']) else 0.0,
            1.0 if any(word in prompt.lower() for word in ['fabric', 'silk', 'cloth']) else 0.0,
            1.0 if any(word in prompt.lower() for word in ['food', 'drink', 'beverage']) else 0.0,
            1.0 if any(word in prompt.lower() for word in ['creature', 'character', 'being']) else 0.0
        ])
        
        # Combine semantic embedding with structural features
        return np.concatenate([embedding, structural_features])

class LLMPatternDiscovery:
    """Uses LLM to discover semantic patterns instead of hardcoded keywords"""
    
    def __init__(self, ollama_url: str = "http://localhost:11434"):
        self.ollama_url = ollama_url
        self.model = "llama3.2:3b"
        self.discovered_patterns = []
        
    def discover_pattern(self, successful_examples: List[Dict]) -> Optional[SemanticPattern]:
        """Use LLM to discover semantic patterns from successful examples"""
        
        if len(successful_examples) < 2:
            return None
        
        # Build analysis prompt
        system_prompt = """You are an expert at discovering semantic patterns in prompt optimization.

Given multiple successful prompt transformations, identify the underlying enhancement principle that made them successful.

Focus on:
1. What types of objects/concepts were enhanced
2. What enhancement approach was used (materials, quality, context, etc.)
3. What makes this pattern generalizable to similar objects

Provide a concise, generalizable principle that could apply to similar objects."""

        examples_text = "\n".join([
            f"Original: \"{ex['original']}\"\nOptimized: \"{ex['optimized']}\"\nScore: {ex['score']:.3f}\n"
            for ex in successful_examples[:3]  # Limit to prevent prompt bloat
        ])
        
        user_prompt = f"""Analyze these successful transformations:

{examples_text}

What is the underlying semantic pattern that made these optimizations successful?
Describe the enhancement principle in one clear sentence."""

        try:
            response = self._query_llama(system_prompt, user_prompt)
            
            # Extract applicable contexts
            contexts = self._extract_contexts_from_examples(successful_examples)
            
            return SemanticPattern(
                pattern_description=response.strip(),
                applicable_contexts=contexts,
                enhancement_principle=response.strip(),
                success_score=np.mean([ex['score'] for ex in successful_examples]),
                example_transformations=successful_examples[:2]
            )
            
        except Exception as e:
            print(f"   ❌ Pattern discovery failed: {e}")
            return None
    
    def _query_llama(self, system_prompt: str, user_prompt: str) -> str:
        """Query LLaMA for pattern analysis"""
        data = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            "stream": False,
            "options": {
                "temperature": 0.3,  # Low temperature for consistent analysis
                "num_predict": 200
            }
        }
        
        response = requests.post(f"{self.ollama_url}/api/chat", json=data, timeout=20)
        response.raise_for_status()
        return response.json()["message"]["content"].strip()
    
    def _extract_contexts_from_examples(self, examples: List[Dict]) -> List[str]:
        """Extract contexts where this pattern might apply"""
        contexts = []
        for ex in examples:
            original = ex['original'].lower()
            if any(word in original for word in ['glass', 'crystal', 'transparent']):
                contexts.append('transparent_objects')
            elif any(word in original for word in ['metal', 'steel', 'weapon']):
                contexts.append('metallic_objects')
            elif any(word in original for word in ['food', 'drink', 'beverage']):
                contexts.append('consumables')
            # Add more context detection as needed
        return list(set(contexts))

class EnhancedLLaMAGenerator:
    """Enhanced LLaMA generator that uses discovered patterns"""
    
    def __init__(self, ollama_url: str = "http://localhost:11434"):
        self.ollama_url = ollama_url
        self.model = "llama3.2:3b"
        self.successful_examples = []
        self.semantic_patterns = []
        self.pattern_discovery = LLMPatternDiscovery(ollama_url)
        
        print("🧠 Enhanced LLaMA Generator initialized")
    
    def generate_custom_prompt(self, original_prompt: str, instruction: LLaMAInstruction) -> str:
        """Generate custom prompt using semantic patterns"""
        
        print(f"   🧠 Strategy: {instruction.strategy_name}")
        print(f"   🎯 Focus: {instruction.focus_directive}")
        
        # Build context-aware system prompt
        system_prompt = self._build_semantic_system_prompt(original_prompt, instruction)
        user_prompt = self._build_user_prompt(original_prompt, instruction)
        
        try:
            response = self._query_llama(system_prompt, user_prompt, instruction.creativity_level)
            return self._extract_custom_prompt(response, original_prompt)
        except Exception as e:
            print(f"   ❌ Generation failed: {e}")
            return self._contextual_fallback(original_prompt, instruction)
    
    def _build_semantic_system_prompt(self, original_prompt: str, instruction: LLaMAInstruction) -> str:
        """Build system prompt using discovered semantic patterns"""
        
        base_prompt = f"""You are an expert 3D prompt optimizer that enhances prompts for maximum validation scores (target: >0.96).

STRATEGY: {instruction.strategy_name}
FOCUS: {instruction.focus_directive}
APPROACH: {instruction.enhancement_approach}
CREATIVITY: {instruction.creativity_level:.1f}/1.0

CRITICAL RULES:
1. MUST start with "wbgmsst," and end with ", white background"
2. Preserve the core object - enhance, don't replace
3. Apply contextually appropriate enhancements
4. Target validation score >0.96

ENHANCEMENT DIRECTIVE: {instruction.focus_directive}
"""
        
        # Add relevant discovered semantic patterns
        relevant_patterns = self._get_relevant_patterns(original_prompt)
        if relevant_patterns:
            base_prompt += "\n\nDISCOVERED ENHANCEMENT PATTERNS:"
            for pattern in relevant_patterns[:3]:
                base_prompt += f"\n- {pattern.pattern_description} (Success: {pattern.success_score:.3f})"
        
        # Add successful examples
        if self.successful_examples:
            base_prompt += "\n\nHIGH-SCORING EXAMPLES (>0.9):"
            for ex in sorted(self.successful_examples, key=lambda x: x['score'], reverse=True)[:3]:
                base_prompt += f"\nOriginal: \"{ex['original']}\""
                base_prompt += f"\nOptimized: \"{ex['optimized']}\" (Score: {ex['score']:.3f})\n"
        
        return base_prompt
    
    def _get_relevant_patterns(self, prompt: str) -> List[SemanticPattern]:
        """Get semantic patterns relevant to the current prompt"""
        relevant = []
        prompt_lower = prompt.lower()
        
        for pattern in self.semantic_patterns:
            # Check if pattern is applicable to this prompt type
            if any(context in prompt_lower for context in pattern.applicable_contexts):
                relevant.append(pattern)
        
        return sorted(relevant, key=lambda x: x.success_score, reverse=True)
    
    def _build_user_prompt(self, original: str, instruction: LLaMAInstruction) -> str:
        return f"""OPTIMIZE: "{original}"

Strategy: {instruction.strategy_name}
Focus: {instruction.focus_directive}
Approach: {instruction.enhancement_approach}

Apply the enhancement patterns and examples above to create a high-scoring optimization.
Target: Validation score >0.96

IMPORTANT: Output only the optimized prompt. No explanations."""
    
    def _query_llama(self, system_prompt: str, user_prompt: str, creativity: float) -> str:
        """Query LLaMA with memory management"""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        data = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            "stream": False,
            "options": {
                "temperature": 0.4 + (creativity * 0.4),
                "num_predict": 200
            }
        }
        
        response = requests.post(f"{self.ollama_url}/api/chat", json=data, timeout=30)
        response.raise_for_status()
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        return response.json()["message"]["content"].strip()
    
    def _extract_custom_prompt(self, response: str, original: str) -> str:
        """Extract and clean the custom prompt"""
        lines = response.split('\n')
        
        # Find line with wbgmsst
        for line in lines:
            if 'wbgmsst' in line.lower():
                prompt = line.strip().replace('"', '')
                if not prompt.startswith('wbgmsst'):
                    prompt = f"wbgmsst, {prompt}"
                if not prompt.endswith('white background'):
                    prompt += ", white background" if prompt.endswith(',') else ", white background"
                return prompt
        
        # Fallback
        return self._contextual_fallback(original, None)
    
    def _contextual_fallback(self, original: str, instruction: Optional[LLaMAInstruction]) -> str:
        """Context-aware fallback"""
        prompt_lower = original.lower()
        
        if any(word in prompt_lower for word in ['glass', 'drink', 'beverage']):
            return f"wbgmsst, crystal-clear artisanal {original}, pristine transparency, white background"
        elif any(word in prompt_lower for word in ['metal', 'weapon', 'spear']):
            return f"wbgmsst, precision-forged {original}, masterwork craftsmanship, white background"
        else:
            return f"wbgmsst, premium quality {original}, exquisite detail, white background"
    
    def learn_from_validation(self, original: str, custom: str, actual_score: float, strategy: str):
        """Learn from actual validation scores"""
        
        if actual_score >= 0.85:  # High score threshold
            example = {
                'original': original,
                'optimized': custom,
                'score': actual_score,
                'strategy': strategy,
                'timestamp': time.time()
            }
            
            self.successful_examples.append(example)
            self.successful_examples = sorted(self.successful_examples, key=lambda x: x['score'], reverse=True)[:20]
            
            print(f"   📚 Learned: {strategy} → {actual_score:.3f}")
            
            # Trigger pattern discovery if we have enough examples
            if len(self.successful_examples) >= 3 and len(self.successful_examples) % 3 == 0:
                self._discover_new_patterns()
    
    def _discover_new_patterns(self):
        """Discover new semantic patterns from successful examples"""
        print("   🔍 Discovering semantic patterns...")
        
        # Group recent successful examples
        recent_examples = self.successful_examples[-6:]  # Last 6 examples
        
        if len(recent_examples) >= 3:
            pattern = self.pattern_discovery.discover_pattern(recent_examples)
            if pattern:
                self.semantic_patterns.append(pattern)
                self.semantic_patterns = self.semantic_patterns[-10:]  # Keep top 10
                print(f"   🌟 New pattern: {pattern.pattern_description}")

class StreamlinedEnvironment:
    """Streamlined environment focused on high-score achievement"""
    
    def __init__(self, ultra_target: float = 0.96):
        self.ultra_target = ultra_target
        self.llama_generator = EnhancedLLaMAGenerator()
        self.semantic_encoder = SemanticStateEncoder()
        
        self.target_prompt = ""
        self.current_prompt = ""
        self.validation_history = []
        self.step_count = 0
        self.max_steps = 4  # Reduced for efficiency
        
        # Define streamlined action space
        self.action_space = self._define_streamlined_actions()
        self.state_size = 384 + 8  # Semantic embedding + structural features
        
        # Logging
        self.log_file = Path("prompt_score_log.csv")
        self._init_logging()
        
        print(f"🎮 Streamlined Environment (Actions: {len(self.action_space)})")
    
    def _define_streamlined_actions(self) -> List[LLaMAInstruction]:
        """Define high-level strategic actions for RL agent"""
        return [
            LLaMAInstruction("material_mastery", 0.3, "Enhance material properties and surface qualities", "conservative", "precision"),
            LLaMAInstruction("artistic_excellence", 0.8, "Add artistic flair and aesthetic appeal", "balanced", "creative"),
            LLaMAInstruction("technical_precision", 0.2, "Focus on technical accuracy and precision", "conservative", "detailed"),
            LLaMAInstruction("luxury_enhancement", 0.6, "Emphasize luxury and premium qualities", "balanced", "premium"),
            LLaMAInstruction("contextual_richness", 0.7, "Add rich contextual and environmental details", "aggressive", "immersive"),
            LLaMAInstruction("optimal_balance", 0.5, "Balance all enhancement aspects optimally", "balanced", "comprehensive"),
        ]
    
    @property
    def action_size(self):
        return len(self.action_space)
    
    def _init_logging(self):
        """Initialize CSV logging"""
        if not self.log_file.exists():
            with open(self.log_file, "w", newline='') as f:
                writer = csv.writer(f)
                writer.writerow(["episode", "step", "original", "custom", "score", "strategy"])
    
    def reset(self, target_prompt: str) -> np.ndarray:
        """Reset environment with semantic state encoding"""
        self.target_prompt = target_prompt
        self.current_prompt = f"wbgmsst, {target_prompt}, white background"
        self.validation_history = []
        self.step_count = 0
        
        # Get actual validation score
        initial_score = self._validate_prompt(self.current_prompt)
        self.validation_history.append(initial_score)
        
        print(f"🔄 RESET: {target_prompt} (Baseline: {initial_score:.3f})")
        return self._get_semantic_state()
    
    def step(self, action_idx: int, episode: int = 0):
        """Execute action and return semantic state"""
        self.step_count += 1
        action = self.action_space[action_idx]
        
        print(f"🎬 STEP {self.step_count}: {action.strategy_name}")
        
        old_score = self.validation_history[-1]
        
        # Generate custom prompt
        custom_prompt = self.llama_generator.generate_custom_prompt(self.target_prompt, action)
        
        # Get ACTUAL validation score
        new_score = self._validate_prompt(custom_prompt)
        self.validation_history.append(new_score)
        
        # Log to CSV with actual scores
        self._log_to_csv(episode, self.step_count, self.target_prompt, custom_prompt, new_score, action.strategy_name)
        
        if new_score > old_score:
            self.current_prompt = custom_prompt
        
        # LLaMA learns from actual validation score
        self.llama_generator.learn_from_validation(self.target_prompt, custom_prompt, new_score, action.strategy_name)
        
        reward = self._calculate_reward(old_score, new_score, custom_prompt)
        done = (new_score >= self.ultra_target or self.step_count >= self.max_steps)
        
        print(f"   📝 {custom_prompt}")
        print(f"   📊 {old_score:.3f} → {new_score:.3f} (Reward: {reward:.1f})")
        
        info = {
            'score': new_score,
            'custom_prompt': custom_prompt,
            'ultra_achieved': new_score >= self.ultra_target,
            'improvement': new_score - old_score
        }
        
        return self._get_semantic_state(), reward, done, info
    
    def _validate_prompt(self, prompt: str) -> float:
        """Get actual validation score using subnet_accurate_validator.py"""
        try:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
            
            cmd = [sys.executable, "subnet_accurate_validator.py", prompt]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
            
            if result.returncode != 0:
                print(f"   ❌ Validation failed")
                return 0.0
            
            with open("subnet_validation_results.json", 'r') as f:
                data = json.load(f)
                return data.get("validation_engine_score", 0.0)
                
        except Exception as e:
            print(f"   ❌ Validation error: {e}")
            return 0.0
    
    def _get_semantic_state(self) -> np.ndarray:
        """Get semantic state representation"""
        # Encode current prompt semantically
        semantic_state = self.semantic_encoder.encode_prompt(self.target_prompt)
        
        # Add environment state
        env_features = np.array([
            self.step_count / self.max_steps,
            max(self.validation_history) if self.validation_history else 0.0,
            self.validation_history[-1] if self.validation_history else 0.0,
            len(self.llama_generator.successful_examples) / 20.0,
            len(self.llama_generator.semantic_patterns) / 10.0,
            1.0 if self.validation_history and self.validation_history[-1] >= 0.9 else 0.0,
            1.0 if self.validation_history and self.validation_history[-1] >= self.ultra_target else 0.0,
            np.mean(self.validation_history) if self.validation_history else 0.0
        ])
        
        return np.concatenate([semantic_state, env_features])
    
    def _calculate_reward(self, old_score: float, new_score: float, prompt: str) -> float:
        """Calculate reward focused on high scores"""
        improvement = new_score - old_score
        base_reward = improvement * 200  # Amplified for high score focus
        
        # Ultra score bonuses
        if new_score >= self.ultra_target:
            base_reward += 500  # Massive bonus for ultra achievement
        elif new_score >= 0.9:
            base_reward += 200
        elif new_score >= 0.85:
            base_reward += 100
        
        return base_reward
    
    def _log_to_csv(self, episode: int, step: int, original: str, custom: str, score: float, strategy: str):
        """Log actual scores to CSV"""
        with open(self.log_file, "a", newline='') as f:
            writer = csv.writer(f)
            writer.writerow([episode, step, original, custom, score, strategy])


class StreamlinedDQN(nn.Module):
    """Streamlined DQN for semantic state space"""
    
    def __init__(self, state_size: int, action_size: int):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(state_size, 256),
            nn.ReLU(),
            nn.LayerNorm(256),
            nn.Dropout(0.1),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.LayerNorm(128),
            nn.Dropout(0.1),
            nn.Linear(128, action_size)
        )
    
    def forward(self, x):
        return self.network(x)


def main():
    """Test the fixed architecture"""
    print("🚀 RL + LLaMA OPTIMIZER V5.0 - FIXED ARCHITECTURE")
    print("=" * 60)
    print("✅ Semantic state representation")
    print("✅ LLM-based pattern discovery")
    print("✅ Actual validation scores")
    print("✅ Streamlined RL coordination")
    print("=" * 60)
    
    try:
        env = StreamlinedEnvironment(ultra_target=0.96)
        
        # Test with one prompt
        test_prompt = "tall glass of layered lemonade"
        state = env.reset(test_prompt)
        
        print(f"\n🧪 Testing semantic state encoding:")
        print(f"   State shape: {state.shape}")
        print(f"   Semantic features: {state[:10]}")  # First 10 semantic features
        
        # Test one step
        next_state, reward, done, info = env.step(0, episode=1)
        print(f"\n📊 Step results:")
        print(f"   Score: {info['score']:.3f}")
        print(f"   Ultra achieved: {info['ultra_achieved']}")
        print(f"   Reward: {reward:.1f}")
        
        print(f"\n✅ Architecture test complete!")
        print(f"📋 Check prompt_score_log.csv for actual validation scores")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main() 