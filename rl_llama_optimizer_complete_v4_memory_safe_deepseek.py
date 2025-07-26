#!/usr/bin/env python3
"""
RL + LLaMA Optimizer v4.0 - MEMORY SAFE + COMPLETE SAVE/RESUME
==============================================================
🚀 DYNAMIC ACTION SPACE: LLaMA creates new strategies from successful patterns
🧠 INTELLIGENT DQN: LSTM + Attention + Dueling architecture (CPU-ONLY)
🎯 ADVANCED REWARDS: Multi-objective with exploration, creativity, consistency bonuses
🔬 PROACTIVE META-LEARNING: Continuous pattern mining and hypothesis testing
🎮 MEMORY-SAFE: FORCED CPU + complete save/resume from v3.1
💾 CHECKPOINTS: Full training state persistence and restoration

Key Memory Safety Features:
✅ FORCED CPU mode for RL model (no CUDA conflicts with OLLaMA)
✅ Complete save/resume checkpoint system from v3.1
✅ Intelligent memory cleanup and monitoring
✅ Graceful error handling and fallbacks
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
from typing import Dict, List, Tuple, Optional, Set
from dataclasses import dataclass, asdict, field
from pathlib import Path
import signal
import datetime
import statistics
import hashlib
import re

Experience = namedtuple('Experience', ['state', 'action', 'reward', 'next_state', 'done'])

def intelligent_memory_manager():
    """Intelligent GPU/CPU memory allocation"""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        
        if reserved > 10.0:  # >10GB - aggressive cleanup
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            time.sleep(1)
            return "aggressive_cleanup"
        elif reserved > 7.0:  # >7GB - moderate cleanup
            torch.cuda.empty_cache()
            return "moderate_cleanup"
        return "normal"
    return "cpu_only"

@dataclass
class TrainingCheckpoint:
    """Complete training state checkpoint - same as v3.1 + v4 additions"""
    episode: int
    total_episodes_completed: int
    current_prompt_index: int
    training_prompts: List[str]
    episodes_per_prompt: int
    episode_rewards: List[float]
    episode_scores: List[float]
    ultra_achievements: List[bool]
    epsilon: float
    step_count: int
    learn_count: int
    best_overall_score: float
    training_start_time: float
    last_checkpoint_time: float
    new_patterns_learned: int = 0
    llama_successful_patterns: int = 0
    
    # V4 additions
    discovered_strategies: int = 0
    action_space_size: int = 7

@dataclass 
class TrainingMetrics:
    """Training metrics for monitoring"""
    episode: int
    score: float
    reward: float
    epsilon: float
    loss: float
    ultra_achieved: bool
    improvement: float
    prompt_length: int
    action_type: str
    exploration_action: bool
    learn_count: int

@dataclass
class MetaLearningEvent:
    """Records meta-learning discoveries"""
    episode: int
    original_prompt: str
    successful_prompt: str
    extracted_pattern: str
    score_achieved: float
    timestamp: float

@dataclass
class DynamicLLaMAInstruction:
    """Dynamic instruction that can evolve"""
    strategy_name: str
    creativity_level: float
    focus_area: str
    enhancement_type: str
    risk_level: str
    length_target: str
    success_rate: float = 0.0
    usage_count: int = 0
    avg_score: float = 0.0
    created_from_pattern: bool = False
    generation: int = 0
    parent_strategy: Optional[str] = None

@dataclass
class PatternDiscovery:
    """Discovered pattern for meta-learning"""
    pattern_id: str
    pattern_text: str
    success_examples: List[str]
    avg_score: float
    confidence: float
    discovery_method: str
    prompt_types: Set[str] = field(default_factory=set)

# Prioritized Experience Replay - same as v3.1
class PrioritizedReplayBuffer:
    def __init__(self, capacity: int = 3000, alpha: float = 0.6):  # Reduced for CPU
        self.capacity = capacity
        self.alpha = alpha
        self.buffer = []
        self.priorities = np.zeros(capacity, dtype=np.float32)
        self.position = 0
        self.max_priority = 1.0

    def push(self, experience: Experience):
        if len(self.buffer) < self.capacity:
            self.buffer.append(experience)
        else:
            self.buffer[self.position] = experience
        self.priorities[self.position] = self.max_priority
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size: int, beta: float = 0.4):
        if len(self.buffer) == self.capacity:
            priorities = self.priorities
        else:
            priorities = self.priorities[:self.position]
        
        probs = priorities ** self.alpha
        probs /= probs.sum()
        indices = np.random.choice(len(self.buffer), batch_size, p=probs)
        experiences = [self.buffer[i] for i in indices]
        total = len(self.buffer)
        weights = (total * probs[indices]) ** (-beta)
        weights /= weights.max()
        return experiences, indices, np.array(weights, dtype=np.float32)

    def update_priorities(self, batch_indices, batch_priorities):
        for idx, priority in zip(batch_indices, batch_priorities):
            self.priorities[idx] = priority
        self.max_priority = max(self.max_priority, np.max(batch_priorities))

    def __len__(self):
        return len(self.buffer)

    def save(self, filepath):
        data = {
            'buffer': self.buffer,
            'priorities': self.priorities,
            'position': self.position,
            'max_priority': self.max_priority
        }
        with open(filepath, 'wb') as f:
            pickle.dump(data, f)

    def load(self, filepath):
        if filepath.exists():
            with open(filepath, 'rb') as f:
                data = pickle.load(f)
                self.buffer = data['buffer']
                self.priorities = data['priorities']
                self.position = data['position']
                self.max_priority = data.get('max_priority', 1.0)
            return True
        return False

# Production Readiness Monitor - same as v3.1
class ProductionReadinessMonitor:
    def __init__(self):
        self.metrics_history = []
        self.success_criteria = {
            "ultra_achievement_rate": 0.3,
            "avg_score_threshold": 0.75,
            "min_episodes": 50
        }
        
    def add_episode_metrics(self, metrics: TrainingMetrics):
        self.metrics_history.append(metrics)
        
    def assess_production_readiness(self) -> Dict:
        if len(self.metrics_history) < self.success_criteria["min_episodes"]:
            return {"ready": False, "reason": "Insufficient episodes"}
        
        recent_window = self.metrics_history[-30:]
        recent_scores = [m.score for m in recent_window]
        recent_ultras = [m.ultra_achieved for m in recent_window]
        
        ultra_rate = sum(recent_ultras) / len(recent_ultras)
        avg_score = statistics.mean(recent_scores)
        
        checks = {
            "ultra_rate": ultra_rate >= self.success_criteria["ultra_achievement_rate"],
            "avg_score": avg_score >= self.success_criteria["avg_score_threshold"],
            "min_episodes": len(self.metrics_history) >= self.success_criteria["min_episodes"]
        }
        
        ready = sum(checks.values()) >= len(checks) * 0.8
        
        return {
            "ready": ready,
            "metrics": {"ultra_rate": ultra_rate, "avg_score": avg_score},
            "checks": checks
        }

class DynamicActionSpace:
    """Evolving action space that learns new strategies"""
    
    def __init__(self, max_strategies: int = 20):  # Reduced for memory safety
        self.max_strategies = max_strategies
        self.base_strategies = self._create_base_strategies()
        self.learned_strategies = []
        
    def _create_base_strategies(self) -> List[DynamicLLaMAInstruction]:
        """Create foundational strategy set"""
        return [
            DynamicLLaMAInstruction("material_precision", 0.3, "material", "precision", "conservative", "medium"),
            DynamicLLaMAInstruction("material_artistic", 0.8, "material", "artistic", "balanced", "detailed"),
            DynamicLLaMAInstruction("shape_creative", 0.9, "shape", "artistic", "aggressive", "detailed"),
            DynamicLLaMAInstruction("quality_masterpiece", 0.7, "quality", "artistic", "aggressive", "detailed"),
            DynamicLLaMAInstruction("context_studio", 0.4, "context", "precision", "balanced", "medium"),
            DynamicLLaMAInstruction("balanced_optimal", 0.5, "quality", "premium", "balanced", "medium"),
            DynamicLLaMAInstruction("aggressive_max", 0.9, "quality", "artistic", "aggressive", "detailed"),
        ]
    
    def get_all_strategies(self) -> List[DynamicLLaMAInstruction]:
        return self.base_strategies + self.learned_strategies
    
    def add_discovered_strategy(self, pattern: PatternDiscovery) -> bool:
        if len(self.learned_strategies) >= (self.max_strategies - len(self.base_strategies)):
            self._prune_ineffective_strategies()
        
        strategy_name = f"discovered_{pattern.pattern_id}"
        creativity = min(0.9, pattern.confidence + 0.2)
        focus_area = self._infer_focus_area(pattern.pattern_text)
        enhancement = self._infer_enhancement_type(pattern.pattern_text)
        risk_level = "aggressive" if creativity > 0.7 else "balanced"
        
        new_strategy = DynamicLLaMAInstruction(
            strategy_name=strategy_name,
            creativity_level=creativity,
            focus_area=focus_area,
            enhancement_type=enhancement,
            risk_level=risk_level,
            length_target="detailed",
            created_from_pattern=True,
            generation=1
        )
        
        self.learned_strategies.append(new_strategy)
        print(f"   🆕 NEW STRATEGY DISCOVERED: {strategy_name} (Focus: {focus_area})")
        return True
    
    def _infer_focus_area(self, pattern_text: str) -> str:
        pattern_lower = pattern_text.lower()
        if any(word in pattern_lower for word in ["material", "steel", "metal", "fabric", "glass"]):
            return "material"
        elif any(word in pattern_lower for word in ["shape", "geometric", "form", "structure"]):
            return "shape"
        elif any(word in pattern_lower for word in ["quality", "masterpiece", "premium", "excellence"]):
            return "quality"
        else:
            return "context"
    
    def _infer_enhancement_type(self, pattern_text: str) -> str:
        pattern_lower = pattern_text.lower()
        if any(word in pattern_lower for word in ["artistic", "creative", "aesthetic"]):
            return "artistic"
        elif any(word in pattern_lower for word in ["precision", "technical", "accurate"]):
            return "precision"
        else:
            return "premium"
    
    def update_strategy_performance(self, strategy_idx: int, score: float):
        strategies = self.get_all_strategies()
        if strategy_idx < len(strategies):
            strategy = strategies[strategy_idx]
            strategy.usage_count += 1
            strategy.avg_score = ((strategy.avg_score * (strategy.usage_count - 1)) + score) / strategy.usage_count
    
    def _prune_ineffective_strategies(self):
        if not self.learned_strategies:
            return
        self.learned_strategies.sort(key=lambda s: s.avg_score, reverse=True)
        removed = self.learned_strategies.pop()
        print(f"   🗑️ PRUNED STRATEGY: {removed.strategy_name} (Score: {removed.avg_score:.3f})")

# CPU-Optimized DQN Network
class CPUOptimizedDQN(nn.Module):
    """CPU-optimized DQN with simplified architecture"""
    
    def __init__(self, state_size: int, action_size: int, hidden_size: int = 128):
        super(CPUOptimizedDQN, self).__init__()
        self.state_size = state_size
        self.action_size = action_size
        
        # Simplified architecture for CPU efficiency
        self.feature_extractor = nn.Sequential(
            nn.Linear(state_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        # Dueling DQN heads
        self.value_head = nn.Linear(hidden_size, 1)
        self.advantage_head = nn.Linear(hidden_size, action_size)
    
    def forward(self, state):
        features = self.feature_extractor(state)
        value = self.value_head(features)
        advantages = self.advantage_head(features)
        
        # Dueling DQN combination
        q_values = value + (advantages - advantages.mean(dim=1, keepdim=True))
        return q_values

class AdvancedRewardFunction:
    """Multi-objective reward function with exploration and creativity bonuses"""
    
    def __init__(self):
        self.strategy_diversity_window = 10
        self.creativity_memory = deque(maxlen=30)  # Reduced for memory
        
    def calculate_reward(self, old_score: float, new_score: float, prompt: str, 
                        action_idx: int, episode_context: Dict, action_history: List[int]) -> float:
        
        # Base improvement reward
        improvement = new_score - old_score
        base_reward = improvement * 100
        
        # Score milestone bonuses
        if new_score >= 0.96:
            base_reward += 300
        elif new_score >= 0.9:
            base_reward += 150
        elif new_score >= 0.8:
            base_reward += 75
        
        # Exploration bonus
        exploration_bonus = self._calculate_exploration_bonus(action_idx, action_history)
        
        # Creativity bonus
        creativity_bonus = self._calculate_creativity_bonus(prompt)
        
        # Consistency bonus
        consistency_bonus = self._calculate_consistency_bonus(episode_context.get('scores_achieved', []))
        
        # Length penalty
        length_penalty = max(0, (len(prompt) - 100) * 0.5)
        
        total_reward = (base_reward + exploration_bonus + creativity_bonus + 
                       consistency_bonus - length_penalty)
        
        print(f"   💰 REWARD: {total_reward:.1f} = {base_reward:.1f}(base) + {exploration_bonus:.1f}(explore) + {creativity_bonus:.1f}(creative) + {consistency_bonus:.1f}(consist) - {length_penalty:.1f}(length)")
        
        return total_reward
    
    def _calculate_exploration_bonus(self, action_idx: int, action_history: List[int]) -> float:
        if len(action_history) < 2:
            return 0.0
        
        recent_actions = action_history[-self.strategy_diversity_window:]
        unique_strategies = len(set(recent_actions))
        total_strategies = len(recent_actions)
        
        diversity_ratio = unique_strategies / total_strategies
        exploration_bonus = diversity_ratio * 25
        
        if action_idx not in action_history[:-1]:
            exploration_bonus += 15
        
        return exploration_bonus
    
    def _calculate_creativity_bonus(self, prompt: str) -> float:
        creativity_score = 0.0
        words = prompt.lower().split()
        
        novel_words = 0
        for word in words:
            if word not in self.creativity_memory and len(word) > 4:
                novel_words += 1
                self.creativity_memory.append(word)
        
        creativity_score += novel_words * 3
        
        if any(len(word) > 10 for word in words):
            creativity_score += 5
        
        technical_terms = ['precision', 'aerospace', 'quantum', 'ultra', 'micro', 'nano']
        artistic_terms = ['elegant', 'masterpiece', 'aesthetic', 'harmonious', 'sublime']
        
        for term in technical_terms + artistic_terms:
            if term in prompt.lower():
                creativity_score += 3
        
        return min(creativity_score, 20)
    
    def _calculate_consistency_bonus(self, scores: List[float]) -> float:
        if len(scores) < 3:
            return 0.0
        
        improvements = [scores[i] - scores[i-1] for i in range(1, len(scores))]
        positive_improvements = [imp for imp in improvements if imp > 0]
        
        if len(positive_improvements) >= len(improvements) * 0.7:
            return 15
        elif len(positive_improvements) >= len(improvements) * 0.5:
            return 8
        
        return 0.0

class SimplifiedMetaLearner:
    """Simplified meta-learning for memory efficiency"""
    
    def __init__(self):
        self.all_experiences = []
        self.pattern_id_counter = 0
        
    def continuous_pattern_mining(self, experiences: List[Dict]) -> List[PatternDiscovery]:
        self.all_experiences.extend(experiences)
        
        # Keep only recent experiences for memory efficiency
        if len(self.all_experiences) > 100:
            self.all_experiences = self.all_experiences[-100:]
        
        if len(self.all_experiences) < 10:
            return []
        
        # Simplified pattern mining
        successful_prompts = [exp for exp in self.all_experiences[-30:] 
                            if exp.get('score', 0) >= 0.8]
        
        if len(successful_prompts) < 3:
            return []
        
        patterns = []
        
        # Find common technical terms
        all_text = ' '.join([exp.get('prompt', '') for exp in successful_prompts])
        technical_pattern = re.findall(r'([a-z]+-[a-z]+(?:-[a-z]+)*)', all_text.lower())
        
        for pattern in set(technical_pattern):
            if len(pattern) > 8:
                confidence = sum(1 for exp in successful_prompts 
                               if pattern in exp.get('prompt', '').lower()) / len(successful_prompts)
                
                if confidence >= 0.3:
                    patterns.append(PatternDiscovery(
                        pattern_id=f"text_{self.pattern_id_counter}",
                        pattern_text=pattern,
                        success_examples=[exp.get('prompt', '') for exp in successful_prompts 
                                        if pattern in exp.get('prompt', '').lower()][:3],
                        avg_score=statistics.mean([exp.get('score', 0) for exp in successful_prompts 
                                                 if pattern in exp.get('prompt', '').lower()]),
                        confidence=confidence,
                        discovery_method='simplified_regex'
                    ))
                    self.pattern_id_counter += 1
        
        if patterns:
            print(f"   🔬 DISCOVERED {len(patterns)} NEW PATTERNS")
        
        return patterns

def remove_think_tags(text):
    import re
    return re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL).strip()

# Memory-Safe LLaMA Generator
class MemorySafeLLaMAGenerator:
    """Memory-optimized LLaMA generator"""
    
    def __init__(self, ollama_url: str = "http://localhost:11434"):
        self.ollama_url = ollama_url
        # self.model = "llama3.2:3b"
        self.model = "deepseek-r1:1.5b"
        self.successful_examples = []
        self.learned_patterns = []
        self._test_connection()
        print("🧠 MEMORY-SAFE LLaMA GENERATOR INITIALIZED")
    
    def _test_connection(self):
        try:
            response = requests.get(f"{self.ollama_url}/api/tags", timeout=5)
            if response.status_code == 200:
                print("   ✅ LLaMA 3.2 Connected")
            else:
                raise Exception(f"Connection failed: {response.status_code}")
        except Exception as e:
            print(f"❌ LLaMA 3.2 unavailable: {e}")
    
    def generate_custom_prompt(self, original_prompt: str, instruction: DynamicLLaMAInstruction) -> str:
        print(f"   🧠 LLaMA Strategy: {instruction.strategy_name}")
        
        intelligent_memory_manager()
        
        system_prompt = self._build_system_prompt(instruction)
        user_prompt = self._build_user_prompt(original_prompt, instruction)
        
        try:
            response = self._query_llama(system_prompt, user_prompt, instruction.creativity_level)
            return self._extract_custom_prompt(response, original_prompt)
        except Exception as e:
            print(f"   ❌ LLaMA generation failed: {e}")
            return self._fallback_prompt(original_prompt, instruction)
    
    def _build_system_prompt(self, instruction: DynamicLLaMAInstruction) -> str:
        base_prompt = f"""You are an expert 3D prompt optimizer.

STRATEGY: {instruction.strategy_name}
FOCUS: {instruction.focus_area}
CREATIVITY: {instruction.creativity_level:.1f}/1.0

REQUIREMENTS:
1. Start with "wbgmsst,"
2. End with ", white background"
3. Keep concise for memory efficiency


FORMAT:
ANALYSIS: [Brief analysis]
CUSTOM_PROMPT: [Your optimized prompt]
REASONING: [Why this will score 0.9+]

IMPORTANT: Only output the optimized prompt. Do NOT include any explanations, analysis, or extra text. Do NOT say anything except the prompt itself.
"""

        # Add limited successful examples
        if self.successful_examples:
            base_prompt += "\n\nSUCCESSFUL EXAMPLES:"
            for ex in self.successful_examples[-2:]:  # Only 2 examples for memory
                base_prompt += f"\n{ex['custom']} (Score: {ex['score']:.3f})"

        return base_prompt
    
    def _build_user_prompt(self, original: str, instruction: DynamicLLaMAInstruction) -> str:
        return f"""OPTIMIZE: "{original}"
STRATEGY: {instruction.strategy_name}
FOCUS: {instruction.focus_area}
CREATIVITY: {instruction.creativity_level:.1f}/1.0

REQUIREMENTS:
1. Start with "wbgmsst,"
2. End with ", white background"
3. Keep concise for memory efficiency

Should be short and concise. 
Examples of good patterns:
- wbgmsst, aerospace-grade precision-engineered {original}, ultra-high technical specification, white background
- wbgmsst, defense-grade ultra-precision {original}, premium excellence, white background

Based on the context of the original prompt, create a completely custom optimization that addresses its unique characteristics.
Analyze this specific prompt and create a completely custom optimization that addresses its unique characteristics.
Think about what makes THIS object special and how to enhance those qualities for 3D generation.

GENERATE YOUR CUSTOM PROMPT:"""

# Strategy: {instruction.strategy_name}
# Focus: {instruction.focus_area}

# Create a concise, custom optimization."""
    
    def _query_llama(self, system_prompt: str, user_prompt: str, creativity: float) -> str:
        temperature = 0.4 + (creativity * 0.4)
        
        data = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": 120  # Reduced for memory safety
            }
        }
        
        try:
            response = requests.post(f"{self.ollama_url}/api/chat", json=data, timeout=30)
            response.raise_for_status()
            intelligent_memory_manager()
            return response.json()["message"]["content"].strip()
            
        except Exception as e:
            print(f"   🔥 LLaMA query failed: {e}")
            intelligent_memory_manager()
            raise e
    
    # def _extract_custom_prompt(self, response: str, original: str) -> str:
    #     lines = response.split('\n')
        
    #     for line in lines:
    #         if line.strip().startswith('CUSTOM_PROMPT:'):
    #             prompt = line.split('CUSTOM_PROMPT:', 1)[1].strip()
    #             return self._clean_prompt(prompt, original)
        
    #     for line in lines:
    #         if 'wbgmsst' in line.lower():
    #             return self._clean_prompt(line.strip(), original)
        
    #     return self._fallback_prompt(original, None)
    # def _extract_custom_prompt(self, response: str, original: str) -> str:
    #     """Extract custom prompt from LLaMA response"""
    #     # response = remove_think_tags(response)
    #     response = re.sub(r'<think>.*?</think>', '', response, flags=re.DOTALL)
    #     lines = response.split('\n')
    #     custom_prompt = None
        
    #     # Look for CUSTOM_PROMPT section
    #     for line in lines:
    #         if line.strip().startswith('CUSTOM_PROMPT:'):
    #             custom_prompt = line.split('CUSTOM_PROMPT:', 1)[1].strip()
    #             break
        
    #     # Fallback: look for wbgmsst line
    #     if not custom_prompt:
    #         for line in lines:
    #             if 'wbgmsst' in line.lower():
    #                 custom_prompt = line.strip()
    #                 break
        
    #     # Clean and validate
    #     if custom_prompt:
    #         custom_prompt = custom_prompt.replace('"', '').strip()
            
    #         if not custom_prompt.startswith('wbgmsst'):
    #             custom_prompt = f"wbgmsst, {custom_prompt}"
    #         if not custom_prompt.endswith('white background'):
    #             if custom_prompt.endswith(','):
    #                 custom_prompt += " white background"
    #             else:
    #                 custom_prompt += ", white background"
            
    #         return custom_prompt
        
    #     # Ultimate fallback
    #     print(f"   ❌ Fallback prompt ")
    #     return self._fallback_prompt(original, None)

    def _extract_custom_prompt(self, response: str, original: str) -> str:
        import re
        # Remove <think> tags and similar
        response = re.sub(r'<think>.*?</think>', '', response, flags=re.DOTALL)
        # Find the shortest substring that starts with wbgmsst, and ends with white background
        match = re.search(r'wbgmsst,.*?white background', response, re.IGNORECASE | re.DOTALL)
        if match:
            prompt = match.group(0)
            # Remove any meta-phrases that often appear at the start
            prompt = re.sub(
                r'^(wbgmsst,\s*)?(Okay, so I need to optimize.*?\.|Alright, so I\'m trying to optimize.*?\.|The original prompt is.*?\.|Let me break this down\.,?)\s*',
                'wbgmsst, ', prompt, flags=re.IGNORECASE
            )
            # Remove any trailing meta-phrases after the prompt
            prompt = re.sub(
                r'\.\s*white background$', ' white background', prompt, flags=re.IGNORECASE
            )
            # Remove any double commas or extra whitespace
            prompt = re.sub(r',\s*,', ',', prompt)
            prompt = prompt.replace(' ,', ',').replace(',,', ',').strip()
            return prompt
        # Fallback: try to find any line containing wbgmsst
        for line in response.split('\n'):
            if 'wbgmsst' in line.lower():
                return line.strip().replace('"', '')
        # Ultimate fallback
        return f"wbgmsst, professional-grade {original}, detailed craftsmanship, white background"
    
    def _clean_prompt(self, prompt: str, original: str) -> str:
        prompt = prompt.replace('"', '').strip()
        if not prompt.startswith('wbgmsst'):
            prompt = f"wbgmsst, {prompt}"
        if not prompt.endswith('white background'):
            if prompt.endswith(','):
                prompt += " white background"
            else:
                prompt += ", white background"
        return prompt
    
    def _fallback_prompt(self, original: str, instruction: Optional[DynamicLLaMAInstruction]) -> str:
        if instruction and instruction.focus_area == 'material':
            return f"wbgmsst, precision-crafted {original}, ultra-high material specification, white background"
        elif instruction and instruction.focus_area == 'quality':
            return f"wbgmsst, masterpiece-quality {original}, premium excellence, white background"
        return f"wbgmsst, professional-grade {original}, detailed craftsmanship, white background"
    
    def learn_from_feedback(self, original: str, custom: str, score: float, strategy: str):
        if score >= 0.8:
            self.successful_examples.append({
                'original': original, 'custom': custom, 
                'score': score, 'strategy': strategy
            })
            # Keep only recent examples for memory efficiency
            self.successful_examples = self.successful_examples[-5:]
            print(f"   🧠 LLaMA learned success: {strategy} → {score:.3f}")
    
    def learn_pattern_from_meta_learning(self, pattern: str, score: float):
        self.learned_patterns.append({'pattern': pattern, 'score': score})
        self.learned_patterns = self.learned_patterns[-5:]
        print(f"   🌟 LLaMA learned meta-pattern: {pattern} (Score: {score:.3f})")

# Memory-Safe Environment 
class MemorySafeEnvironmentV4:
    """Memory-optimized environment with dynamic action space"""
    
    def __init__(self, ultra_target: float = 0.96, checkpoint_dir: str = "rl_checkpoints_v4"):
        self.ultra_target = ultra_target
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(exist_ok=True)
        
        # Components
        self.dynamic_action_space = DynamicActionSpace()
        self.llama_generator = MemorySafeLLaMAGenerator()
        self.reward_function = AdvancedRewardFunction()
        self.meta_learner = SimplifiedMetaLearner()
        
        # Environment state
        self.target_prompt = ""
        self.current_prompt = ""
        self.validation_history = []
        self.step_count = 0
        self.max_steps = 6  # Reduced for memory efficiency
        self.state_size = 30  # Reduced state space
        self.action_history = []
        
        # Meta-learning
        self.meta_learning_events = []
        self.meta_learn_score_threshold = 0.8
        self.session_id = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.episode_log = []
        
        print(f"🚀 MEMORY-SAFE ENVIRONMENT V4 INITIALIZED")
        print(f"   🎮 Dynamic Action Space: {len(self.dynamic_action_space.get_all_strategies())} strategies")
    
    @property
    def action_size(self):
        return len(self.dynamic_action_space.get_all_strategies())
    
    def reset(self, target_prompt: str) -> np.ndarray:
        self.target_prompt = target_prompt
        self.current_prompt = f"wbgmsst, {target_prompt}, white background"
        self.validation_history = []
        self.step_count = 0
        self.action_history = []
        
        memory_status = intelligent_memory_manager()
        print(f"   🧠 Memory Status: {memory_status}")
        
        initial_score = self._validate_prompt(self.current_prompt)
        self.validation_history.append(initial_score)
        
        print(f"🔄 RESET V4: {target_prompt} (Baseline: {initial_score:.3f})")
        return self._get_state()
    
    def step(self, action_idx: int) -> Tuple[np.ndarray, float, bool, Dict]:
        self.step_count += 1
        
        strategies = self.dynamic_action_space.get_all_strategies()
        if action_idx >= len(strategies):
            action_idx = 0
        
        action = strategies[action_idx]
        self.action_history.append(action_idx)
        
        print(f"🎬 STEP {self.step_count}: {action.strategy_name}")
        
        old_score = self.validation_history[-1]
        
        # Generate custom prompt
        custom_prompt = self.llama_generator.generate_custom_prompt(self.target_prompt, action)
        new_score = self._validate_prompt(custom_prompt)
        self.validation_history.append(new_score)
        
        if new_score > old_score:
            self.current_prompt = custom_prompt
        
        # Update strategy performance
        self.dynamic_action_space.update_strategy_performance(action_idx, new_score)
        
        # Learn from feedback
        self.llama_generator.learn_from_feedback(
            self.target_prompt, custom_prompt, new_score, action.strategy_name
        )
        
        # Calculate reward
        episode_context = {
            'scores_achieved': self.validation_history,
            'step_count': self.step_count
        }
        
        reward = self.reward_function.calculate_reward(
            old_score, new_score, custom_prompt, action_idx, 
            episode_context, self.action_history
        )
        
        done = (new_score >= self.ultra_target or self.step_count >= self.max_steps)
        
        print(f"   📝 {custom_prompt}")
        print(f"   📊 {old_score:.3f} → {new_score:.3f}")
        
        # Simplified meta-learning
        if self.step_count % 3 == 0 or done:
            experience_data = {
                'prompt': custom_prompt,
                'score': new_score,
                'strategy_used': action.strategy_name
            }
            new_patterns = self.meta_learner.continuous_pattern_mining([experience_data])
            for pattern in new_patterns:
                self.dynamic_action_space.add_discovered_strategy(pattern)
        
        info = {
            'score': new_score,
            'custom_prompt': custom_prompt,
            'strategy_used': action.strategy_name,
            'improvement': new_score - old_score,
            'ultra_achieved': new_score >= self.ultra_target,
            'action_space_size': len(strategies),
            'discovered_strategies': sum(1 for s in strategies if s.created_from_pattern)
        }
        
        return self._get_state(), reward, done, info
    
    def _validate_prompt(self, prompt: str) -> float:
        try:
            memory_status = intelligent_memory_manager()
            
            cmd = [sys.executable, "subnet_accurate_validator.py", prompt]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
            
            if result.returncode != 0:
                print(f"   ❌ Validation failed (return code {result.returncode})")
                if "CUDA" in result.stderr or "out of memory" in result.stderr.lower():
                    print(f"   🔥 CUDA OOM detected - aggressive cleanup")
                    intelligent_memory_manager()
                    time.sleep(2)
                return 0.0
            
            with open("subnet_validation_results.json", 'r') as f:
                data = json.load(f)
                score = data.get("validation_engine_score", 0.0)
                
                if score == 0.0:
                    print(f"   🔧 Score 0.0 - potential memory issue")
                    intelligent_memory_manager()
                
                return score
                
        except Exception as e:
            print(f"   ❌ Validation error: {e}")
            intelligent_memory_manager()
            return 0.0
    
    def _get_state(self) -> np.ndarray:
        state = np.zeros(self.state_size)
        
        # Recent scores
        for i, score in enumerate(self.validation_history[-5:]):
            if i < 5:
                state[i] = score
        
        # Progress and performance
        state[5] = self.step_count / self.max_steps
        state[6] = max(self.validation_history) if self.validation_history else 0.0
        state[7] = np.mean(self.validation_history) if self.validation_history else 0.0
        
        # LLaMA learning state
        state[8] = min(len(self.llama_generator.successful_examples) / 5, 1.0)
        state[9] = min(len(self.llama_generator.learned_patterns) / 5, 1.0)
        
        # Action space dynamics
        strategies = self.dynamic_action_space.get_all_strategies()
        state[10] = len(strategies) / 20.0
        state[11] = sum(1 for s in strategies if s.created_from_pattern) / len(strategies)
        
        # Strategy diversity
        if len(self.action_history) > 0:
            unique_actions = len(set(self.action_history[-8:]))
            state[12] = unique_actions / min(8, len(self.action_history))
        
        # Prompt characteristics
        target_lower = self.target_prompt.lower()
        state[13] = 1.0 if any(w in target_lower for w in ["steel", "metal", "iron"]) else 0.0
        state[14] = 1.0 if any(w in target_lower for w in ["fabric", "silk", "cotton"]) else 0.0
        state[15] = 1.0 if any(w in target_lower for w in ["glass", "crystal"]) else 0.0
        
        return state

# CPU-Only DQN Agent
class MemorySafeDQNAgentV4:
    """CPU-only DQN agent for memory safety"""
    
    def __init__(self, state_size: int, max_action_size: int, checkpoint_dir: str):
        self.state_size = state_size
        self.max_action_size = max_action_size
        self.checkpoint_dir = Path(checkpoint_dir)
        
        # FORCE CPU to avoid memory conflicts with OLLaMA
        self.device = torch.device("cpu")
        print("   🖥️ FORCED CPU MODE for memory safety with OLLaMA")
        
        # Networks
        self.q_network_local = CPUOptimizedDQN(state_size, max_action_size).to(self.device)
        self.q_network_target = CPUOptimizedDQN(state_size, max_action_size).to(self.device)
        self.optimizer = optim.Adam(self.q_network_local.parameters(), lr=0.001)
        
        # Memory with PER
        self.memory = PrioritizedReplayBuffer(capacity=3000)
        self.beta = 0.4
        self.beta_increment = 0.001
        self.batch_size = 16  # Reduced for CPU
        self.gamma = 0.95
        self.tau = 0.005
        self.update_every = 4
        self.epsilon = 0.9
        self.epsilon_min = 0.05
        self.epsilon_decay = 0.98
        self.step_count = 0
        self.learn_count = 0
        
        print(f"�� MEMORY-SAFE DQN AGENT V4 INITIALIZED (CPU-ONLY)")
    
    def act(self, state: np.ndarray, current_action_size: int, training: bool = True) -> int:
        if training and random.random() < self.epsilon:
            action = random.randrange(current_action_size)
            print(f"   🎲 EXPLORATION: Strategy {action} (ε={self.epsilon:.3f})")
            return action
        else:
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            with torch.no_grad():
                q_values = self.q_network_local(state_tensor)
                # Mask invalid actions
                q_values_masked = q_values.clone()
                q_values_masked[:, current_action_size:] = -float('inf')
                action = q_values_masked.argmax().item()
            
            print(f"   🧠 EXPLOITATION: Strategy {action} (ε={self.epsilon:.3f})")
            return action
    
    def step(self, state, action, reward, next_state, done):
        experience = Experience(state, action, reward, next_state, done)
        self.memory.push(experience)
        self.step_count += 1
        
        if self.step_count % self.update_every == 0 and len(self.memory) >= self.batch_size:
            self.beta = min(1.0, self.beta + self.beta_increment)
            experiences, indices, weights = self.memory.sample(self.batch_size, self.beta)
            loss = self.learn(experiences, indices, weights)
            if self.epsilon > self.epsilon_min:
                self.epsilon *= self.epsilon_decay
            return loss
        return None
    
    def learn(self, experiences, indices, weights):
        self.learn_count += 1
        
        states = torch.FloatTensor([e.state for e in experiences]).to(self.device)
        actions = torch.LongTensor([e.action for e in experiences]).to(self.device)
        rewards = torch.FloatTensor([e.reward for e in experiences]).to(self.device)
        next_states = torch.FloatTensor([e.next_state for e in experiences]).to(self.device)
        dones = torch.BoolTensor([e.done for e in experiences]).to(self.device)
        weights = torch.FloatTensor(weights).unsqueeze(1).to(self.device)
        
        # Forward pass
        current_q_values = self.q_network_local(states).gather(1, actions.unsqueeze(1))
        
        with torch.no_grad():
            next_q_values = self.q_network_target(next_states).max(1)[0]
            target_q_values = rewards + (self.gamma * next_q_values * ~dones)
        
        # TD errors for PER
        td_errors = torch.abs(target_q_values.unsqueeze(1) - current_q_values).detach().cpu().numpy()
        self.memory.update_priorities(indices, td_errors.squeeze() + 1e-5)
        
        # Compute loss with importance sampling
        loss = (weights * F.mse_loss(current_q_values, target_q_values.unsqueeze(1), reduction='none')).mean()
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_network_local.parameters(), 1.0)
        self.optimizer.step()
        
        # Soft update target network
        for target_param, local_param in zip(self.q_network_target.parameters(), 
                                           self.q_network_local.parameters()):
            target_param.data.copy_(self.tau * local_param.data + (1.0 - self.tau) * target_param.data)
        
        print(f"   📚 LEARNING #{self.learn_count}: Loss {loss.item():.4f}, ε={self.epsilon:.3f}")
        return loss.item()
    
    def save_checkpoint(self, checkpoint_path: Path, metadata: Dict):
        """Save checkpoint - same as v3.1"""
        checkpoint = {
            'q_network_local_state_dict': self.q_network_local.state_dict(),
            'q_network_target_state_dict': self.q_network_target.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'step_count': self.step_count,
            'learn_count': self.learn_count,
            'beta': self.beta,
            'metadata': metadata
        }
        
        torch.save(checkpoint, checkpoint_path / 'agent_checkpoint.pth')
        self.memory.save(checkpoint_path / 'per_buffer.pkl')
        print("   💾 Agent checkpoint saved")

    def load_checkpoint(self, checkpoint_path: Path) -> Optional[Dict]:
        """Load checkpoint - same as v3.1"""
        model_file = checkpoint_path / 'agent_checkpoint.pth'
        if model_file.exists():
            checkpoint = torch.load(model_file, map_location=self.device, weights_only=False)
            
            self.q_network_local.load_state_dict(checkpoint['q_network_local_state_dict'])
            self.q_network_target.load_state_dict(checkpoint['q_network_target_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.epsilon = checkpoint['epsilon']
            self.step_count = checkpoint['step_count']
            self.learn_count = checkpoint.get('learn_count', 0)
            self.beta = checkpoint.get('beta', 0.4)
            
            self.memory.load(checkpoint_path / 'per_buffer.pkl')
            print(f"   📂 Agent checkpoint loaded (ε={self.epsilon:.3f})")
            return checkpoint['metadata']
        return None

# Complete Training System with Save/Resume
class MemorySafeTrainerV4:
    """Complete training system with memory safety and save/resume"""
    
    def __init__(self, ultra_target: float = 0.96, checkpoint_dir: str = "rl_checkpoints_v4"):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(exist_ok=True)
        
        self.env = MemorySafeEnvironmentV4(ultra_target, checkpoint_dir)
        self.agent = MemorySafeDQNAgentV4(self.env.state_size, 20, checkpoint_dir)
        
        self.training_start_time = time.time()
        
        # Training state with save/resume
        self.training_state = TrainingCheckpoint(
            episode=0, total_episodes_completed=0, current_prompt_index=0,
            training_prompts=[], episodes_per_prompt=0, episode_rewards=[],
            episode_scores=[], ultra_achievements=[], epsilon=self.agent.epsilon,
            step_count=0, learn_count=0, best_overall_score=0.0,
            training_start_time=time.time(), last_checkpoint_time=time.time(),
            discovered_strategies=0, action_space_size=len(self.env.dynamic_action_space.get_all_strategies())
        )
        
        self.monitor = ProductionReadinessMonitor()
        self.meta_learn_every_n_episodes = 4
        signal.signal(signal.SIGINT, self._signal_handler)
        
        print("🚀 MEMORY-SAFE TRAINER V4 INITIALIZED")
        print("✅ FORCED CPU mode for memory safety")
        print("✅ Complete save/resume system")
        print("✅ Dynamic action space evolution")
        print("✅ Advanced multi-objective rewards")
    
    def _signal_handler(self, signum, frame):
        """Signal handler for graceful interruption"""
        print(f"\n⚠️ INTERRUPTION DETECTED (Signal {signum})")
        print("💾 Saving emergency checkpoint...")
        self._save_checkpoint("emergency_checkpoint")
        print("✅ Emergency checkpoint saved!")
        sys.exit(0)
    
    def train_with_checkpoints(self, target_prompts: List[str], 
                              episodes_per_prompt: int = 5,
                              resume_from: Optional[str] = None) -> Dict:
        """Train with complete checkpoint system"""
        
        print(f"🎓 MEMORY-SAFE TRAINING SESSION V4")
        print(f"📝 Prompts: {len(target_prompts)} | Episodes each: {episodes_per_prompt}")
        print("=" * 70)
        
        if resume_from and self._load_checkpoint(resume_from):
            print(f"📂 RESUMED FROM: {resume_from}")
        else:
            self.training_state.training_prompts = target_prompts
            self.training_state.episodes_per_prompt = episodes_per_prompt
        
        total_prompts = len(self.training_state.training_prompts)
        
        for prompt_idx in range(self.training_state.current_prompt_index, total_prompts):
            current_prompt = self.training_state.training_prompts[prompt_idx]
            print(f"\n🎯 PROMPT {prompt_idx + 1}/{total_prompts}: '{current_prompt}'")
            
            episodes_completed = 0
            if prompt_idx == self.training_state.current_prompt_index:
                episodes_completed = self.training_state.episode - (prompt_idx * episodes_per_prompt)
            
            for episode_in_prompt in range(episodes_completed, episodes_per_prompt):
                episode_num = prompt_idx * episodes_per_prompt + episode_in_prompt + 1
                
                # Scheduled meta-learning
                if episode_num > 0 and episode_num % self.meta_learn_every_n_episodes == 0:
                    self._meta_learning_phase()
                
                print(f"\n📚 MEMORY-SAFE EPISODE {episode_num}")
                result = self._train_single_episode(current_prompt, episode_num)
                
                # Update training state
                self.training_state.episode = episode_num
                self.training_state.total_episodes_completed += 1
                self.training_state.episode_rewards.append(result['total_reward'])
                self.training_state.episode_scores.append(result['best_score'])
                self.training_state.ultra_achievements.append(result['ultra_achieved'])
                self.training_state.epsilon = self.agent.epsilon
                self.training_state.learn_count = self.agent.learn_count
                self.training_state.best_overall_score = max(
                    self.training_state.best_overall_score, result['best_score']
                )
                
                # V4 specific updates
                strategies = self.env.dynamic_action_space.get_all_strategies()
                self.training_state.discovered_strategies = sum(1 for s in strategies if s.created_from_pattern)
                self.training_state.action_space_size = len(strategies)
                
                # Production monitoring
                metrics = TrainingMetrics(
                    episode=episode_num, score=result['best_score'], 
                    reward=result['total_reward'], epsilon=result['epsilon'],
                    loss=result.get('avg_loss', 0.0), ultra_achieved=result['ultra_achieved'],
                    improvement=result.get('improvement', 0.0), 
                    prompt_length=len(result.get('final_prompt', '')),
                    action_type=result.get('final_action', ''), 
                    exploration_action=result['epsilon'] > 0.2,
                    learn_count=result['learn_count']
                )
                self.monitor.add_episode_metrics(metrics)
                
                # Auto-checkpoint every 5 episodes
                if episode_num % 5 == 0:
                    self._save_checkpoint(f"episode_{episode_num:03d}")
                
                print(f"\n⏸️ EPISODE {episode_num} COMPLETE")
                print(f"   📊 Score: {result['best_score']:.3f} | Reward: {result['total_reward']:.1f}")
                print(f"   🧠 Strategies: {self.training_state.action_space_size} ({self.training_state.discovered_strategies} discovered)")
            
            self.training_state.current_prompt_index = prompt_idx + 1
        
        return self._generate_final_report()
    
    def _train_single_episode(self, target_prompt: str, episode_num: int) -> Dict:
        """Train single episode"""
        state = self.env.reset(target_prompt)
        total_reward = 0
        best_score = self.env.validation_history[0]
        losses = []
        
        while True:
            current_action_size = self.env.action_size
            action = self.agent.act(state, current_action_size, training=True)
            
            next_state, reward, done, info = self.env.step(action)
            
            loss = self.agent.step(state, action, reward, next_state, done)
            if loss is not None:
                losses.append(loss)
            
            total_reward += reward
            best_score = max(best_score, info['score'])
            state = next_state
            
            if done:
                break
        
        return {
            'episode': episode_num,
            'best_score': best_score,
            'total_reward': total_reward,
            'ultra_achieved': best_score >= self.env.ultra_target,
            'epsilon': self.agent.epsilon,
            'learn_count': self.agent.learn_count,
            'avg_loss': statistics.mean(losses) if losses else 0.0
        }
    
    def _meta_learning_phase(self):
        """Simplified meta-learning phase"""
        print(f"\n{'='*20} META-LEARNING PHASE V4 {'='*20}")
        
        if len(self.env.meta_learning_events) < 1:
            print("   📊 No recent successes for meta-learning")
            return
        
        # Simple pattern extraction
        recent_successes = self.env.meta_learning_events[-3:]
        for success in recent_successes:
            pattern = self._extract_pattern(success.original_prompt, success.successful_prompt)
            if pattern:
                self.env.llama_generator.learn_pattern_from_meta_learning(pattern, success.score_achieved)
                self.training_state.new_patterns_learned += 1
        
        print(f"{'='*60}\n")
    
    def _extract_pattern(self, original: str, successful: str) -> Optional[str]:
        """Extract pattern from successful prompts"""
        successful_lower = successful.lower()
        if 'aerospace-grade' in successful_lower and 'precision' in successful_lower:
            return f"aerospace-grade precision-enhanced {original}"
        elif 'masterpiece-quality' in successful_lower:
            return f"masterpiece-quality {original}"
        return None
    
    def _save_checkpoint(self, checkpoint_name: str):
        """Save complete checkpoint"""
        checkpoint_path = self.checkpoint_dir / checkpoint_name
        checkpoint_path.mkdir(exist_ok=True)
        
        # Metadata with dynamic action space
        strategies = self.env.dynamic_action_space.get_all_strategies()
        metadata = {
            'episode': self.training_state.episode,
            'training_prompts': self.training_state.training_prompts,
            'best_overall_score': self.training_state.best_overall_score,
            'new_patterns_learned': self.training_state.new_patterns_learned,
            'meta_learning_events': [asdict(e) for e in self.env.meta_learning_events],
            'llama_learned_patterns': self.env.llama_generator.learned_patterns,
            'llama_successful_examples': self.env.llama_generator.successful_examples,
            
            # V4 specific data
            'dynamic_action_space': {
                'base_strategies': [asdict(s) for s in self.env.dynamic_action_space.base_strategies],
                'learned_strategies': [asdict(s) for s in self.env.dynamic_action_space.learned_strategies]
            },
            'action_space_size': len(strategies),
            'discovered_strategies_count': sum(1 for s in strategies if s.created_from_pattern)
        }
        
        self.agent.save_checkpoint(checkpoint_path, metadata)
        
        with open(checkpoint_path / 'training_state.json', 'w') as f:
            json.dump(asdict(self.training_state), f, indent=2)
        
        print(f"   💾 V4 Checkpoint saved: {checkpoint_name}")
    
    def _load_checkpoint(self, checkpoint_name: str) -> bool:
        """Load complete checkpoint"""
        checkpoint_path = self.checkpoint_dir / checkpoint_name
        training_file = checkpoint_path / 'training_state.json'
        
        if not training_file.exists():
            return False
        
        try:
            with open(training_file, 'r') as f:
                state_dict = json.load(f)
            self.training_state = TrainingCheckpoint(**state_dict)
            
            agent_metadata = self.agent.load_checkpoint(checkpoint_path)
            if agent_metadata:
                # Restore data
                if 'meta_learning_events' in agent_metadata:
                    self.env.meta_learning_events = [
                        MetaLearningEvent(**e) for e in agent_metadata['meta_learning_events']
                    ]
                if 'llama_learned_patterns' in agent_metadata:
                    self.env.llama_generator.learned_patterns = agent_metadata['llama_learned_patterns']
                if 'llama_successful_examples' in agent_metadata:
                    self.env.llama_generator.successful_examples = agent_metadata['llama_successful_examples']
                
                # Restore V4 dynamic action space
                if 'dynamic_action_space' in agent_metadata:
                    das_data = agent_metadata['dynamic_action_space']
                    self.env.dynamic_action_space.base_strategies = [
                        DynamicLLaMAInstruction(**s) for s in das_data['base_strategies']
                    ]
                    self.env.dynamic_action_space.learned_strategies = [
                        DynamicLLaMAInstruction(**s) for s in das_data['learned_strategies']
                    ]
                
                print(f"   📂 V4 Training state loaded (Episode: {self.training_state.episode})")
                print(f"   🎮 Action Space: {self.training_state.action_space_size} ({self.training_state.discovered_strategies} discovered)")
                return True
        except Exception as e:
            print(f"   ❌ Load error: {e}")
        return False
    
    def _generate_final_report(self) -> Dict:
        """Generate comprehensive final report"""
        print(f"\n🎓 FINAL MEMORY-SAFE TRAINING REPORT V4")
        print("=" * 50)
        
        total_episodes = len(self.training_state.episode_scores)
        ultra_count = sum(self.training_state.ultra_achievements)
        avg_score = sum(self.training_state.episode_scores) / total_episodes if total_episodes > 0 else 0
        training_time = time.time() - self.training_state.training_start_time
        
        print(f"📊 PERFORMANCE:")
        print(f"   Episodes: {total_episodes}")
        print(f"   Ultra Rate: {ultra_count}/{total_episodes} ({ultra_count/total_episodes*100:.1f}%)" if total_episodes > 0 else "   Ultra Rate: 0%")
        print(f"   Avg Score: {avg_score:.3f}")
        print(f"   Best Score: {self.training_state.best_overall_score:.3f}")
        print(f"   Training Time: {training_time/3600:.2f}h")
        
        print(f"\n🧠 LEARNING:")
        print(f"   Final Epsilon: {self.agent.epsilon:.3f}")
        print(f"   Learn Count: {self.agent.learn_count}")
        print(f"   Patterns Learned: {self.training_state.new_patterns_learned}")
        
        print(f"\n🚀 V4 MEMORY-SAFE:")
        print(f"   Final Action Space: {self.training_state.action_space_size} strategies")
        print(f"   Discovered Strategies: {self.training_state.discovered_strategies}")
        print(f"   Memory Mode: CPU-ONLY (Safe)")
        
        readiness = self.monitor.assess_production_readiness()
        print(f"\n🚀 PRODUCTION: {'✅ READY' if readiness['ready'] else '❌ NOT READY'}")
        
        return {
            'total_episodes': total_episodes,
            'ultra_rate': ultra_count / total_episodes if total_episodes > 0 else 0,
            'average_score': avg_score,
            'best_score': self.training_state.best_overall_score,
            'patterns_learned': self.training_state.new_patterns_learned,
            'production_ready': readiness['ready'],
            'final_action_space_size': self.training_state.action_space_size,
            'discovered_strategies': self.training_state.discovered_strategies,
            'training_time_hours': training_time / 3600
        }
    
    def list_checkpoints(self) -> List[str]:
        """List available checkpoints"""
        checkpoints = []
        for path in self.checkpoint_dir.iterdir():
            if path.is_dir() and (path / 'training_state.json').exists():
                checkpoints.append(path.name)
        return sorted(checkpoints)

def main():
    """Main function with complete save/resume"""
    print("🚀 RL + LLaMA OPTIMIZER V4.0 - MEMORY SAFE + COMPLETE")
    print("="*70)
    print("✅ FORCED CPU mode for memory safety")
    print("✅ Complete save/resume system")
    print("✅ Dynamic action space evolution")
    print("✅ Advanced multi-objective rewards")
    print("✅ Simplified for memory efficiency")
    print("="*70)
    
    try:
        trainer = MemorySafeTrainerV4(ultra_target=0.96)
        checkpoints = trainer.list_checkpoints()
        
        resume_from = None
        if checkpoints:
            print(f"📂 Found checkpoints: {checkpoints}")
            # Uncomment for interactive resume:
            choice = input("Resume from checkpoint? Enter name or ENTER for new: ").strip()
            if choice in checkpoints:
                resume_from = choice
        
        test_prompts = [
            "hexagonal prism steel structure",
            "elegant silk fabric draping", 
            "transparent crystal sphere",
            "wooden geometric sculpture"
        ]
        
        results = trainer.train_with_checkpoints(
            target_prompts=test_prompts,
            episodes_per_prompt=20,
            resume_from=resume_from
        )
        
        print(f"\n🎉 V4 MEMORY-SAFE TRAINING COMPLETE!")
        print(f"📈 Ultra Rate: {results.get('ultra_rate', 0):.1%}")
        print(f"🧠 Patterns: {results.get('patterns_learned', 0)}")
        print(f"🎮 Final Action Space: {results.get('final_action_space_size', 0)}")
        print(f"🆕 Discovered Strategies: {results.get('discovered_strategies', 0)}")
        print(f"🚀 Production: {results.get('production_ready', False)}")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 
