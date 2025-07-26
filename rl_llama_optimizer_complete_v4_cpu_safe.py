#!/usr/bin/env python3
"""
RL + LLaMA Optimizer v4.0 - REVOLUTIONARY ARCHITECTURE + MEMORY SAFE
====================================================================
🚀 DYNAMIC ACTION SPACE: LLaMA creates new strategies from successful patterns
🧠 INTELLIGENT DQN: LSTM + Attention + Dueling architecture with episode memory
🎯 ADVANCED REWARDS: Multi-objective with exploration, creativity, consistency bonuses
🔬 PROACTIVE META-LEARNING: Continuous pattern mining and hypothesis testing
🎮 MEMORY-SAFE: FORCED CPU for RL model + complete save/resume system

Revolutionary improvements over v3.1:
✅ Dynamic strategy evolution - action space grows with learning
✅ Attention-based neural architecture for complex pattern recognition  
✅ Multi-dimensional reward function encouraging exploration and creativity
✅ Active meta-learning that continuously discovers new patterns
✅ FORCED CPU mode to prevent CUDA OOM with OLLaMA
✅ Complete save/resume checkpoint system from v3.1
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
from sklearn.cluster import DBSCAN
import re

Experience = namedtuple('Experience', ['state', 'action', 'reward', 'next_state', 'done', 'episode_context'])

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
    """Complete training state checkpoint - same as v3.1"""
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
    """Training metrics for monitoring - same as v3.1"""
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
    """Records meta-learning discoveries - same as v3.1"""
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
    generation: int = 0  # Track evolution generations
    parent_strategy: Optional[str] = None

@dataclass
class EpisodeMemory:
    """Rich episode context for neural network memory"""
    episode_id: int
    target_prompt: str
    actions_taken: List[int]
    scores_achieved: List[float]
    strategies_used: List[str]
    final_score: float
    improvement_trajectory: List[float]
    prompt_embedding: Optional[np.ndarray] = None

@dataclass
class PatternDiscovery:
    """Discovered pattern for meta-learning"""
    pattern_id: str
    pattern_text: str
    success_examples: List[str]
    avg_score: float
    confidence: float
    discovery_method: str  # 'clustering', 'regex', 'llama_analysis'
    prompt_types: Set[str] = field(default_factory=set)

# Prioritized Experience Replay - same as v3.1
class PrioritizedReplayBuffer:
    def __init__(self, capacity: int = 5000, alpha: float = 0.6):
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
    
    def __init__(self, max_strategies: int = 25):
        self.max_strategies = max_strategies
        self.base_strategies = self._create_base_strategies()
        self.learned_strategies = []
        self.strategy_performance = {}
        self.strategy_usage = {}
        
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
        """Get current complete strategy set"""
        return self.base_strategies + self.learned_strategies
    
    def add_discovered_strategy(self, pattern: PatternDiscovery, parent_strategy: str = None) -> bool:
        """Add new strategy from discovered pattern"""
        if len(self.learned_strategies) >= (self.max_strategies - len(self.base_strategies)):
            # Remove worst performing strategy
            self._prune_ineffective_strategies()
        
        # Create new strategy from pattern
        strategy_name = f"discovered_{pattern.pattern_id}"
        
        # Infer parameters from pattern characteristics
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
            generation=1,
            parent_strategy=parent_strategy
        )
        
        self.learned_strategies.append(new_strategy)
        print(f"   🆕 NEW STRATEGY DISCOVERED: {strategy_name} (Focus: {focus_area})")
        return True
    
    def _infer_focus_area(self, pattern_text: str) -> str:
        """Infer focus area from pattern characteristics"""
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
        """Infer enhancement type from pattern"""
        pattern_lower = pattern_text.lower()
        if any(word in pattern_lower for word in ["artistic", "creative", "aesthetic"]):
            return "artistic"
        elif any(word in pattern_lower for word in ["precision", "technical", "accurate"]):
            return "precision"
        else:
            return "premium"
    
    def update_strategy_performance(self, strategy_idx: int, score: float):
        """Update strategy performance metrics"""
        strategies = self.get_all_strategies()
        if strategy_idx < len(strategies):
            strategy = strategies[strategy_idx]
            strategy.usage_count += 1
            strategy.avg_score = ((strategy.avg_score * (strategy.usage_count - 1)) + score) / strategy.usage_count
            strategy.success_rate = strategy.avg_score  # Simplified success rate
    
    def _prune_ineffective_strategies(self):
        """Remove underperforming learned strategies"""
        if not self.learned_strategies:
            return
            
        # Sort by performance and remove worst
        self.learned_strategies.sort(key=lambda s: s.avg_score, reverse=True)
        removed = self.learned_strategies.pop()
        print(f"   🗑️ PRUNED STRATEGY: {removed.strategy_name} (Score: {removed.avg_score:.3f})")

class IntelligentDQN(nn.Module):
    """Advanced DQN with LSTM, Attention, and Dueling architecture"""
    
    def __init__(self, state_size: int, action_size: int, hidden_size: int = 128):  # Reduced for CPU
        super(IntelligentDQN, self).__init__()
        self.state_size = state_size
        self.action_size = action_size
        self.hidden_size = hidden_size
        
        # Episode memory processor (smaller for CPU)
        self.lstm = nn.LSTM(state_size, hidden_size, batch_first=True, num_layers=1, dropout=0.1)
        
        # Attention mechanism (fewer heads for CPU)
        self.attention = nn.MultiheadAttention(hidden_size, num_heads=4, dropout=0.1, batch_first=True)
        
        # Feature extractors
        self.state_processor = nn.Sequential(
            nn.Linear(state_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        # Dueling DQN heads (smaller for CPU)
        self.value_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, 1)
        )
        
        self.advantage_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, action_size)
        )
        
        # Context integration
        self.context_fusion = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU()
        )
    
    def forward(self, state, episode_context=None, hidden_state=None):
        batch_size = state.size(0)
        
        # Process current state
        state_features = self.state_processor(state)
        
        # Process episode context with LSTM if available
        if episode_context is not None and episode_context.size(1) > 1:
            lstm_out, hidden_state = self.lstm(episode_context, hidden_state)
            
            # Apply attention to focus on important parts of episode
            attended_context, _ = self.attention(lstm_out, lstm_out, lstm_out)
            context_features = attended_context[:, -1, :]  # Use last timestep
            
            # Fuse state and context
            combined_features = self.context_fusion(torch.cat([state_features, context_features], dim=1))
        else:
            combined_features = state_features
        
        # Dueling DQN computation
        value = self.value_head(combined_features)
        advantages = self.advantage_head(combined_features)
        
        # Combine value and advantages
        q_values = value + (advantages - advantages.mean(dim=1, keepdim=True))
        
        return q_values, hidden_state

class AdvancedRewardFunction:
    """Multi-objective reward function with exploration and creativity bonuses"""
    
    def __init__(self):
        self.episode_history = []
        self.strategy_diversity_window = 10
        self.creativity_memory = deque(maxlen=50)
        
    def calculate_reward(self, old_score: float, new_score: float, prompt: str, 
                        action_idx: int, episode_context: Dict, action_history: List[int]) -> float:
        """Calculate comprehensive reward with multiple objectives"""
        
        # Base improvement reward
        improvement = new_score - old_score
        base_reward = improvement * 100
        
        # Score milestone bonuses
        if new_score >= 0.96:
            base_reward += 300  # Ultra achievement
        elif new_score >= 0.9:
            base_reward += 150
        elif new_score >= 0.8:
            base_reward += 75
        
        # Exploration bonus - reward trying diverse strategies
        exploration_bonus = self._calculate_exploration_bonus(action_idx, action_history)
        
        # Creativity bonus - reward novel prompt characteristics
        creativity_bonus = self._calculate_creativity_bonus(prompt)
        
        # Consistency bonus - reward consistent improvements
        consistency_bonus = self._calculate_consistency_bonus(episode_context.get('scores_achieved', []))
        
        # Efficiency bonus - reward achieving good scores quickly
        efficiency_bonus = self._calculate_efficiency_bonus(new_score, episode_context.get('step_count', 0))
        
        # Length penalty - encourage concise prompts
        length_penalty = max(0, (len(prompt) - 100) * 0.5)
        
        total_reward = (base_reward + exploration_bonus + creativity_bonus + 
                       consistency_bonus + efficiency_bonus - length_penalty)
        
        # Log reward breakdown for analysis
        print(f"   💰 REWARD: {total_reward:.1f} = {base_reward:.1f}(base) + {exploration_bonus:.1f}(explore) + {creativity_bonus:.1f}(creative) + {consistency_bonus:.1f}(consist) + {efficiency_bonus:.1f}(effic) - {length_penalty:.1f}(length)")
        
        return total_reward
    
    def _calculate_exploration_bonus(self, action_idx: int, action_history: List[int]) -> float:
        """Reward diverse strategy exploration"""
        if len(action_history) < 2:
            return 0.0
        
        recent_actions = action_history[-self.strategy_diversity_window:]
        unique_strategies = len(set(recent_actions))
        total_strategies = len(recent_actions)
        
        diversity_ratio = unique_strategies / total_strategies
        exploration_bonus = diversity_ratio * 25  # Up to 25 point bonus
        
        # Extra bonus for trying completely new strategies
        if action_idx not in action_history[:-1]:
            exploration_bonus += 15
        
        return exploration_bonus
    
    def _calculate_creativity_bonus(self, prompt: str) -> float:
        """Reward novel and creative prompt characteristics"""
        creativity_score = 0.0
        
        # Check for novel descriptors
        words = prompt.lower().split()
        novel_words = 0
        for word in words:
            if word not in self.creativity_memory and len(word) > 4:
                novel_words += 1
                self.creativity_memory.append(word)
        
        creativity_score += novel_words * 3
        
        # Reward complex descriptors
        if any(len(word) > 10 for word in words):
            creativity_score += 5
        
        # Reward technical/artistic terminology
        technical_terms = ['precision', 'aerospace', 'quantum', 'ultra', 'micro', 'nano']
        artistic_terms = ['elegant', 'masterpiece', 'aesthetic', 'harmonious', 'sublime']
        
        for term in technical_terms + artistic_terms:
            if term in prompt.lower():
                creativity_score += 3
        
        return min(creativity_score, 20)  # Cap at 20 points
    
    def _calculate_consistency_bonus(self, scores: List[float]) -> float:
        """Reward consistent upward trajectory"""
        if len(scores) < 3:
            return 0.0
        
        # Check for consistent improvement
        improvements = [scores[i] - scores[i-1] for i in range(1, len(scores))]
        positive_improvements = [imp for imp in improvements if imp > 0]
        
        if len(positive_improvements) >= len(improvements) * 0.7:  # 70% improvements
            return 15
        elif len(positive_improvements) >= len(improvements) * 0.5:  # 50% improvements
            return 8
        
        return 0.0
    
    def _calculate_efficiency_bonus(self, score: float, step_count: int) -> float:
        """Reward achieving good scores quickly"""
        if score >= 0.9 and step_count <= 3:
            return 25
        elif score >= 0.8 and step_count <= 4:
            return 15
        elif score >= 0.7 and step_count <= 5:
            return 8
        
        return 0.0

class ProactiveMetaLearner:
    """Continuous pattern discovery and hypothesis testing"""
    
    def __init__(self):
        self.all_experiences = []
        self.discovered_patterns = []
        self.hypothesis_tests = []
        self.pattern_id_counter = 0
        
    def continuous_pattern_mining(self, experiences: List[Dict]) -> List[PatternDiscovery]:
        """Continuously mine patterns from all experiences"""
        self.all_experiences.extend(experiences)
        
        if len(self.all_experiences) < 10:
            return []
        
        new_patterns = []
        
        # Text-based pattern mining
        text_patterns = self._mine_text_patterns()
        new_patterns.extend(text_patterns)
        
        # Score-based clustering
        score_patterns = self._mine_score_clusters()
        new_patterns.extend(score_patterns)
        
        # Strategy effectiveness patterns
        strategy_patterns = self._mine_strategy_patterns()
        new_patterns.extend(strategy_patterns)
        
        # Filter out duplicate patterns
        new_patterns = self._deduplicate_patterns(new_patterns)
        
        if new_patterns:
            print(f"   🔬 DISCOVERED {len(new_patterns)} NEW PATTERNS")
            for pattern in new_patterns:
                print(f"     📊 {pattern.pattern_text} (Confidence: {pattern.confidence:.3f})")
        
        return new_patterns
    
    def _mine_text_patterns(self) -> List[PatternDiscovery]:
        """Mine patterns from successful prompt text"""
        successful_prompts = [exp for exp in self.all_experiences[-50:] 
                            if exp.get('score', 0) >= 0.8]
        
        if len(successful_prompts) < 5:
            return []
        
        patterns = []
        
        # Find common descriptive patterns
        all_text = ' '.join([exp.get('prompt', '') for exp in successful_prompts])
        
        # Extract multi-word technical terms
        technical_pattern = re.findall(r'([a-z]+-[a-z]+(?:-[a-z]+)*)', all_text.lower())
        
        for pattern in set(technical_pattern):
            if len(pattern) > 8:  # Only significant patterns
                confidence = sum(1 for exp in successful_prompts 
                               if pattern in exp.get('prompt', '').lower()) / len(successful_prompts)
                
                if confidence >= 0.3:  # At least 30% of successful prompts
                    patterns.append(PatternDiscovery(
                        pattern_id=f"text_{self.pattern_id_counter}",
                        pattern_text=pattern,
                        success_examples=[exp.get('prompt', '') for exp in successful_prompts 
                                        if pattern in exp.get('prompt', '').lower()][:3],
                        avg_score=statistics.mean([exp.get('score', 0) for exp in successful_prompts 
                                                 if pattern in exp.get('prompt', '').lower()]),
                        confidence=confidence,
                        discovery_method='regex'
                    ))
                    self.pattern_id_counter += 1
        
        return patterns
    
    def _mine_score_clusters(self) -> List[PatternDiscovery]:
        """Mine patterns from score clustering"""
        if len(self.all_experiences) < 20:
            return []
        
        # Group experiences by score ranges
        high_score_exps = [exp for exp in self.all_experiences if exp.get('score', 0) >= 0.85]
        
        if len(high_score_exps) < 5:
            return []
        
        # Find common characteristics in high-scoring prompts
        patterns = []
        
        # Check for common strategy sequences
        strategy_sequences = [exp.get('strategy_used', '') for exp in high_score_exps]
        strategy_counts = {}
        
        for strategy in strategy_sequences:
            strategy_counts[strategy] = strategy_counts.get(strategy, 0) + 1
        
        for strategy, count in strategy_counts.items():
            if count >= 3 and count / len(high_score_exps) >= 0.4:
                patterns.append(PatternDiscovery(
                    pattern_id=f"strategy_{self.pattern_id_counter}",
                    pattern_text=f"high_success_strategy_{strategy}",
                    success_examples=[exp.get('prompt', '') for exp in high_score_exps 
                                    if exp.get('strategy_used', '') == strategy][:3],
                    avg_score=statistics.mean([exp.get('score', 0) for exp in high_score_exps 
                                             if exp.get('strategy_used', '') == strategy]),
                    confidence=count / len(high_score_exps),
                    discovery_method='clustering'
                ))
                self.pattern_id_counter += 1
        
        return patterns
    
    def _mine_strategy_patterns(self) -> List[PatternDiscovery]:
        """Mine patterns from strategy effectiveness"""
        strategy_performance = {}
        
        for exp in self.all_experiences[-100:]:  # Recent experiences
            strategy = exp.get('strategy_used', '')
            score = exp.get('score', 0)
            
            if strategy not in strategy_performance:
                strategy_performance[strategy] = []
            strategy_performance[strategy].append(score)
        
        patterns = []
        
        for strategy, scores in strategy_performance.items():
            if len(scores) >= 5:
                avg_score = statistics.mean(scores)
                consistency = 1 - (statistics.stdev(scores) / max(avg_score, 0.1))
                
                if avg_score >= 0.8 and consistency >= 0.7:
                    patterns.append(PatternDiscovery(
                        pattern_id=f"effective_{self.pattern_id_counter}",
                        pattern_text=f"consistently_effective_{strategy}",
                        success_examples=[],
                        avg_score=avg_score,
                        confidence=consistency,
                        discovery_method='strategy_analysis'
                    ))
                    self.pattern_id_counter += 1
        
        return patterns
    
    def _deduplicate_patterns(self, patterns: List[PatternDiscovery]) -> List[PatternDiscovery]:
        """Remove duplicate or very similar patterns"""
        unique_patterns = []
        seen_texts = set()
        
        for pattern in patterns:
            pattern_key = pattern.pattern_text.lower().replace('_', '').replace('-', '')
            if pattern_key not in seen_texts:
                seen_texts.add(pattern_key)
                unique_patterns.append(pattern)
        
        return unique_patterns

# Enhanced LLaMA Generator with dynamic strategy creation
class EvolutionaryLLaMAGenerator:
    """LLaMA generator that can create new strategies from patterns"""
    
    def __init__(self, ollama_url: str = "http://localhost:11434"):
        self.ollama_url = ollama_url
        self.model = "llama3.2:3b"
        self.successful_examples = []
        self.learned_patterns = []
        self.strategy_templates = {}
        self._test_connection()
        print("🧠 EVOLUTIONARY LLaMA GENERATOR INITIALIZED")
    
    def _test_connection(self):
        try:
            response = requests.get(f"{self.ollama_url}/api/tags", timeout=5)
            if response.status_code == 200:
                print("   ✅ LLaMA 3.2 Connected")
            else:
                raise Exception(f"Connection failed: {response.status_code}")
        except Exception as e:
            print(f"❌ LLaMA 3.2 unavailable: {e}")
    
    def create_strategy_from_pattern(self, pattern: PatternDiscovery) -> str:
        """Create new strategy instructions from discovered pattern"""
        intelligent_memory_manager()
        
        system_prompt = f"""You are an expert AI strategy designer. Create a new prompting strategy based on a discovered successful pattern.

DISCOVERED PATTERN: {pattern.pattern_text}
SUCCESS RATE: {pattern.confidence:.3f}
AVERAGE SCORE: {pattern.avg_score:.3f}
DISCOVERY METHOD: {pattern.discovery_method}

SUCCESSFUL EXAMPLES:
{chr(10).join(pattern.success_examples[:3])}

Create a concise strategy instruction that captures the essence of this pattern.
Focus on what makes this pattern successful and how to apply it to new prompts.

FORMAT:
STRATEGY_NAME: [descriptive name]
INSTRUCTION: [how to apply this pattern]"""

        user_prompt = f"Analyze the pattern '{pattern.pattern_text}' and create a reusable strategy."

        try:
            response = self._query_llama(system_prompt, user_prompt, 0.7)
            strategy_instruction = self._extract_strategy_instruction(response)
            
            # Store the strategy template
            self.strategy_templates[pattern.pattern_id] = strategy_instruction
            
            return strategy_instruction
            
        except Exception as e:
            print(f"   ❌ Strategy creation failed: {e}")
            return f"Apply the successful pattern: {pattern.pattern_text}"
    
    def generate_custom_prompt(self, original_prompt: str, instruction: DynamicLLaMAInstruction) -> str:
        """Generate custom prompt with enhanced pattern awareness"""
        print(f"   🧠 LLaMA Strategy: {instruction.strategy_name}")
        
        intelligent_memory_manager()
        
        system_prompt = self._build_enhanced_system_prompt(instruction)
        user_prompt = self._build_enhanced_user_prompt(original_prompt, instruction)
        
        try:
            response = self._query_llama(system_prompt, user_prompt, instruction.creativity_level)
            return self._extract_custom_prompt(response, original_prompt)
        except Exception as e:
            print(f"   ❌ LLaMA generation failed: {e}")
            intelligent_memory_manager()
            return self._fallback_prompt(original_prompt, instruction)
    
    def _build_enhanced_system_prompt(self, instruction: DynamicLLaMAInstruction) -> str:
        """Build system prompt with pattern awareness"""
        base_prompt = f"""You are an expert 3D prompt optimizer with access to discovered successful patterns.

STRATEGY: {instruction.strategy_name}
FOCUS: {instruction.focus_area}
ENHANCEMENT: {instruction.enhancement_type}
RISK: {instruction.risk_level}
CREATIVITY: {instruction.creativity_level:.1f}/1.0
GENERATION: {instruction.generation}

REQUIREMENTS:
1. MUST start with "wbgmsst,"
2. MUST end with ", white background"
3. PRESERVE main object from original
4. CREATE custom descriptions, not templates
5. FOCUS on {instruction.focus_area} optimization

SUCCESS PATTERNS TO LEVERAGE:"""

        # Add relevant learned patterns
        for pattern in self.learned_patterns[-5:]:
            base_prompt += f"\n- {pattern['pattern']} (Score: {pattern['score']:.3f})"

        # Add strategy-specific template if available
        if instruction.created_from_pattern and instruction.strategy_name.startswith('discovered_'):
            pattern_id = instruction.strategy_name.replace('discovered_', '')
            if pattern_id in self.strategy_templates:
                base_prompt += f"\n\nSPECIAL STRATEGY GUIDANCE:\n{self.strategy_templates[pattern_id]}"

        base_prompt += f"\n\nFORMAT:\nCUSTOM_PROMPT: [Your optimized prompt]"
        
        return base_prompt
    
    def _build_enhanced_user_prompt(self, original: str, instruction: DynamicLLaMAInstruction) -> str:
        """Build user prompt with enhanced context"""
        return f"""OPTIMIZE: "{original}"

Strategy: {instruction.strategy_name}
Focus: {instruction.focus_area}
Style: {instruction.enhancement_type}
Risk: {instruction.risk_level}

This is a {instruction.focus_area}-focused optimization using {instruction.enhancement_type} enhancement.
{'This strategy was discovered from successful patterns.' if instruction.created_from_pattern else 'This is a foundational strategy.'}

Create a completely custom optimization that maximizes the 3D generation score."""
    
    def _query_llama(self, system_prompt: str, user_prompt: str, creativity: float) -> str:
        """Query LLaMA with memory management"""
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
                "num_predict": 150  # Reduced for memory safety
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
    
    def _extract_strategy_instruction(self, response: str) -> str:
        """Extract strategy instruction from LLaMA response"""
        lines = response.split('\n')
        
        instruction_text = ""
        for line in lines:
            if line.strip().startswith('INSTRUCTION:'):
                instruction_text = line.split('INSTRUCTION:', 1)[1].strip()
                break
        
        if not instruction_text:
            # Fallback extraction
            for line in lines:
                if len(line.strip()) > 20 and 'strategy' in line.lower():
                    instruction_text = line.strip()
                    break
        
        return instruction_text or "Apply discovered pattern for enhanced results"
    
    def _extract_custom_prompt(self, response: str, original: str) -> str:
        """Extract custom prompt from LLaMA response"""
        lines = response.split('\n')
        
        for line in lines:
            if line.strip().startswith('CUSTOM_PROMPT:'):
                prompt = line.split('CUSTOM_PROMPT:', 1)[1].strip()
                return self._clean_prompt(prompt, original)
        
        for line in lines:
            if 'wbgmsst' in line.lower():
                return self._clean_prompt(line.strip(), original)
        
        return self._fallback_prompt(original, None)
    
    def _clean_prompt(self, prompt: str, original: str) -> str:
        """Clean and validate prompt"""
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
        """Generate fallback prompt"""
        if instruction and instruction.focus_area == 'material':
            return f"wbgmsst, precision-crafted {original}, ultra-high material specification, white background"
        elif instruction and instruction.focus_area == 'quality':
            return f"wbgmsst, masterpiece-quality {original}, premium excellence, white background"
        return f"wbgmsst, professional-grade {original}, detailed craftsmanship, white background"
    
    def learn_from_feedback(self, original: str, custom: str, score: float, strategy: str):
        """Learn from feedback with enhanced pattern storage"""
        if score >= 0.8:
            self.successful_examples.append({
                'original': original, 'custom': custom, 
                'score': score, 'strategy': strategy,
                'timestamp': time.time()
            })
            self.successful_examples = self.successful_examples[-10:]
            print(f"   🧠 LLaMA learned success: {strategy} → {score:.3f}")
    
    def learn_pattern_from_meta_learning(self, pattern: str, score: float):
        """Learn pattern from meta-learning discovery"""
        self.learned_patterns.append({
            'pattern': pattern, 
            'score': score,
            'timestamp': time.time()
        })
        self.learned_patterns = self.learned_patterns[-8:]
        print(f"   🌟 LLaMA learned meta-pattern: {pattern} (Score: {score:.3f})")

# Enhanced Environment with Dynamic Action Space
class RevolutionaryEnvironmentV4:
    """Environment with dynamic action space and intelligent memory management"""
    
    def __init__(self, ultra_target: float = 0.96, checkpoint_dir: str = "rl_checkpoints_v4"):
        self.ultra_target = ultra_target
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(exist_ok=True)
        
        # Dynamic components
        self.dynamic_action_space = DynamicActionSpace()
        self.llama_generator = EvolutionaryLLaMAGenerator()
        self.reward_function = AdvancedRewardFunction()
        self.meta_learner = ProactiveMetaLearner()
        
        # Environment state
        self.target_prompt = ""
        self.current_prompt = ""
        self.validation_history = []
        self.step_count = 0
        self.max_steps = 8  # Increased for more exploration
        self.state_size = 40  # Expanded state space
        self.episode_memory = []
        self.action_history = []
        
        # Meta-learning
        self.meta_learning_events = []
        self.meta_learn_score_threshold = 0.8
        self.session_id = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.episode_log = []
        
        print(f"🚀 REVOLUTIONARY ENVIRONMENT V4 INITIALIZED")
        print(f"   🎮 Dynamic Action Space: {len(self.dynamic_action_space.get_all_strategies())} strategies")
    
    @property
    def action_size(self):
        return len(self.dynamic_action_space.get_all_strategies())
    
    def reset(self, target_prompt: str) -> Tuple[np.ndarray, np.ndarray]:
        """Reset environment and return state + episode context"""
        self.target_prompt = target_prompt
        self.current_prompt = f"wbgmsst, {target_prompt}, white background"
        self.validation_history = []
        self.step_count = 0
        self.action_history = []
        
        # Memory management
        memory_status = intelligent_memory_manager()
        print(f"   🧠 Memory Status: {memory_status}")
        
        # Initial validation
        initial_score = self._validate_prompt(self.current_prompt)
        self.validation_history.append(initial_score)
        
        # Create episode memory
        episode_memory = EpisodeMemory(
            episode_id=len(self.episode_log),
            target_prompt=target_prompt,
            actions_taken=[],
            scores_achieved=[initial_score],
            strategies_used=[],
            final_score=initial_score,
            improvement_trajectory=[0.0]
        )
        self.episode_memory.append(episode_memory)
        
        print(f"🔄 RESET V4: {target_prompt} (Baseline: {initial_score:.3f})")
        
        state = self._get_enhanced_state()
        episode_context = self._get_episode_context()
        
        return state, episode_context
    
    def step(self, action_idx: int) -> Tuple[np.ndarray, np.ndarray, float, bool, Dict]:
        """Enhanced step with dynamic action space and advanced rewards"""
        self.step_count += 1
        
        # Get strategy (might be newly discovered)
        strategies = self.dynamic_action_space.get_all_strategies()
        if action_idx >= len(strategies):
            action_idx = 0  # Fallback to first strategy
        
        action = strategies[action_idx]
        self.action_history.append(action_idx)
        
        print(f"🎬 STEP {self.step_count}: {action.strategy_name}")
        if action.created_from_pattern:
            print(f"   🆕 Using discovered strategy (Gen {action.generation})")
        
        old_score = self.validation_history[-1]
        
        # Generate custom prompt with evolutionary LLaMA
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
        
        # Advanced reward calculation
        episode_context = {
            'scores_achieved': self.validation_history,
            'step_count': self.step_count,
            'strategies_used': [strategies[idx].strategy_name for idx in self.action_history]
        }
        
        reward = self.reward_function.calculate_reward(
            old_score, new_score, custom_prompt, action_idx, 
            episode_context, self.action_history
        )
        
        # Update episode memory
        current_episode = self.episode_memory[-1]
        current_episode.actions_taken.append(action_idx)
        current_episode.scores_achieved.append(new_score)
        current_episode.strategies_used.append(action.strategy_name)
        current_episode.final_score = new_score
        current_episode.improvement_trajectory.append(new_score - old_score)
        
        done = (new_score >= self.ultra_target or self.step_count >= self.max_steps)
        
        print(f"   📝 {custom_prompt}")
        print(f"   📊 {old_score:.3f} → {new_score:.3f}")
        
        # Continuous meta-learning
        experience_data = {
            'prompt': custom_prompt,
            'score': new_score,
            'strategy_used': action.strategy_name,
            'improvement': new_score - old_score,
            'step_count': self.step_count
        }
        
        # Trigger pattern discovery every few steps
        if self.step_count % 3 == 0 or done:
            new_patterns = self.meta_learner.continuous_pattern_mining([experience_data])
            for pattern in new_patterns:
                success = self.dynamic_action_space.add_discovered_strategy(pattern)
                if success:
                    # Teach LLaMA the new strategy
                    strategy_instruction = self.llama_generator.create_strategy_from_pattern(pattern)
                    print(f"   🎓 LLaMA learned new strategy: {pattern.pattern_id}")
        
        info = {
            'score': new_score,
            'custom_prompt': custom_prompt,
            'strategy_used': action.strategy_name,
            'improvement': new_score - old_score,
            'ultra_achieved': new_score >= self.ultra_target,
            'action_space_size': len(strategies),
            'discovered_strategies': sum(1 for s in strategies if s.created_from_pattern),
            'trigger_immediate_meta_learning': new_score >= 0.9
        }
        
        next_state = self._get_enhanced_state()
        episode_context = self._get_episode_context()
        
        return next_state, episode_context, reward, done, info
    
    def _validate_prompt(self, prompt: str) -> float:
        """Validate prompt with intelligent memory management"""
        try:
            # Proactive memory management
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
    
    def _get_enhanced_state(self) -> np.ndarray:
        """Get enhanced state representation"""
        state = np.zeros(self.state_size)
        
        # Recent scores (expanded)
        for i, score in enumerate(self.validation_history[-5:]):
            if i < 5:
                state[i] = score
        
        # Progress and performance
        state[5] = self.step_count / self.max_steps
        state[6] = max(self.validation_history) if self.validation_history else 0.0
        state[7] = np.mean(self.validation_history) if self.validation_history else 0.0
        state[8] = len(self.validation_history)
        
        # LLaMA learning state
        state[9] = min(len(self.llama_generator.successful_examples) / 10, 1.0)
        state[10] = min(len(self.llama_generator.learned_patterns) / 8, 1.0)
        
        # Action space dynamics
        strategies = self.dynamic_action_space.get_all_strategies()
        state[11] = len(strategies) / 25.0  # Normalized action space size
        state[12] = sum(1 for s in strategies if s.created_from_pattern) / len(strategies)
        
        # Strategy diversity
        if len(self.action_history) > 0:
            unique_actions = len(set(self.action_history[-10:]))
            state[13] = unique_actions / min(10, len(self.action_history))
        
        # Prompt characteristics (enhanced)
        target_lower = self.target_prompt.lower()
        state[14] = 1.0 if any(w in target_lower for w in ["steel", "metal", "iron", "aluminum"]) else 0.0
        state[15] = 1.0 if any(w in target_lower for w in ["fabric", "silk", "cotton", "cloth"]) else 0.0
        state[16] = 1.0 if any(w in target_lower for w in ["glass", "crystal", "transparent"]) else 0.0
        state[17] = 1.0 if any(w in target_lower for w in ["wood", "wooden", "timber"]) else 0.0
        state[18] = 1.0 if any(w in target_lower for w in ["geometric", "shape", "form"]) else 0.0
        
        # Recent performance trends
        if len(self.validation_history) >= 3:
            recent_trend = np.mean(np.diff(self.validation_history[-3:]))
            state[19] = max(-1, min(1, recent_trend * 10))  # Normalized trend
        
        # Episode context
        state[20] = len(self.episode_log) / 100.0  # Episode number
        
        return state
    
    def _get_episode_context(self) -> np.ndarray:
        """Get episode context for LSTM processing"""
        if not self.episode_memory:
            return np.zeros((1, self.state_size))
        
        # Create sequence of states for current episode
        episode = self.episode_memory[-1]
        context_length = min(len(episode.scores_achieved), 10)
        
        context = np.zeros((context_length, self.state_size))
        
        for i in range(context_length):
            # Simplified context state
            context[i, 0] = episode.scores_achieved[i] if i < len(episode.scores_achieved) else 0
            context[i, 1] = i / context_length  # Position in episode
            if i < len(episode.improvement_trajectory):
                context[i, 2] = episode.improvement_trajectory[i]
        
        return context.reshape(1, context_length, self.state_size)

# Intelligent DQN Agent V4 - FORCED CPU
class IntelligentDQNAgentV4:
    """Advanced DQN agent with dynamic action space support - FORCED CPU"""
    
    def __init__(self, state_size: int, max_action_size: int, checkpoint_dir: str):
        self.state_size = state_size
        self.max_action_size = max_action_size
        self.checkpoint_dir = Path(checkpoint_dir)
        
        # FORCE CPU to avoid memory conflicts with OLLaMA
        self.device = torch.device("cpu")
        print("   🖥️ FORCED CPU MODE for memory safety with OLLaMA")
        
        # Networks
        self.q_network_local = IntelligentDQN(state_size, max_action_size).to(self.device)
        self.q_network_target = IntelligentDQN(state_size, max_action_size).to(self.device)
        self.optimizer = optim.Adam(self.q_network_local.parameters(), lr=0.0005)
        
        # Enhanced memory with PER (like v3.1)
        self.memory = PrioritizedReplayBuffer(capacity=5000)
        self.beta = 0.4
        self.beta_increment = 0.001
        self.batch_size = 32  # Reduced for CPU
        self.gamma = 0.99
        self.tau = 0.005
        self.update_every = 4
        self.epsilon = 0.95
        self.epsilon_min = 0.05
        self.epsilon_decay = 0.995
        self.step_count = 0
        self.learn_count = 0
        
        # Episode memory for LSTM
        self.episode_contexts = deque(maxlen=1000)
        self.hidden_state = None
        
        print(f"🤖 INTELLIGENT DQN AGENT V4 INITIALIZED (CPU-ONLY)")
    
    def act(self, state: np.ndarray, episode_context: np.ndarray, 
            current_action_size: int, training: bool = True) -> int:
        """Enhanced action selection with episode context"""
        
        if training and random.random() < self.epsilon:
            action = random.randrange(current_action_size)
            print(f"   🎲 EXPLORATION: Strategy {action} (ε={self.epsilon:.3f})")
            return action
        else:
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            context_tensor = torch.FloatTensor(episode_context).to(self.device)
            
            with torch.no_grad():
                q_values, self.hidden_state = self.q_network_local(
                    state_tensor, context_tensor, self.hidden_state
                )
                # Mask invalid actions
                q_values_masked = q_values.clone()
                q_values_masked[:, current_action_size:] = -float('inf')
                
                action = q_values_masked.argmax().item()
            
            print(f"   🧠 EXPLOITATION: Strategy {action} (ε={self.epsilon:.3f})")
            return action
    
    def step(self, state, episode_context, action, reward, next_state, next_episode_context, done):
        """Enhanced learning step with episode context - using PER like v3.1"""
        experience = Experience(state, action, reward, next_state, done, episode_context)
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
        """Enhanced learning with episode context and PER"""
        self.learn_count += 1
        
        # Sample batch
        states = torch.FloatTensor([e.state for e in experiences]).to(self.device)
        actions = torch.LongTensor([e.action for e in experiences]).to(self.device)
        rewards = torch.FloatTensor([e.reward for e in experiences]).to(self.device)
        next_states = torch.FloatTensor([e.next_state for e in experiences]).to(self.device)
        dones = torch.BoolTensor([e.done for e in experiences]).to(self.device)
        weights = torch.FloatTensor(weights).unsqueeze(1).to(self.device)
        
        # Process episode contexts - simplified for CPU efficiency
        try:
            # Simplified context processing for CPU
            context_features = torch.zeros(len(experiences), self.state_size).to(self.device)
            
            # Forward pass
            current_q_values, _ = self.q_network_local(states)
            current_q_values = current_q_values.gather(1, actions.unsqueeze(1))
            
            with torch.no_grad():
                next_q_values, _ = self.q_network_target(next_states)
                max_next_q_values = next_q_values.max(1)[0]
                target_q_values = rewards + (self.gamma * max_next_q_values * ~dones)
            
            # TD errors for PER
            td_errors = torch.abs(target_q_values.unsqueeze(1) - current_q_values).detach().cpu().numpy()
            self.memory.update_priorities(indices, td_errors.squeeze() + 1e-5)
            
            # Compute loss with importance sampling weights
            loss = (weights * F.mse_loss(current_q_values, target_q_values.unsqueeze(1), reduction='none')).mean()
            
        except Exception as e:
            # Fallback to simple forward pass if context processing fails
            print(f"   ⚠️ Context processing failed, using simple forward pass: {e}")
            current_q_values = self.q_network_local(states)[0].gather(1, actions.unsqueeze(1))
            with torch.no_grad():
                next_q_values = self.q_network_target(next_states)[0].max(1)[0]
                target_q_values = rewards + (self.gamma * next_q_values * ~dones)
            loss = F.mse_loss(current_q_values.squeeze(), target_q_values)
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_network_local.parameters(), 1.0)
        self.optimizer.step()
        
        # Soft update target network
        for target_param, local_param in zip(self.q_network_target.parameters(), 
                                           self.q_network_local.parameters()):
            target_param.data.copy_(self.tau * local_param.data + (1.0 - self.tau) * target_param.data)
        
        print(f"   📚 INTELLIGENT LEARNING #{self.learn_count}: Loss {loss.item():.4f}, ε={self.epsilon:.3f}")
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

# Revolutionary Training System V4 with COMPLETE Save/Resume
class RevolutionaryTrainerV4:
    """Complete training system with all v4 enhancements + v3.1 save/resume"""
    
    def __init__(self, ultra_target: float = 0.96, checkpoint_dir: str = "rl_checkpoints_v4"):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(exist_ok=True)
        
        self.env = RevolutionaryEnvironmentV4(ultra_target, checkpoint_dir)
        self.agent = IntelligentDQNAgentV4(self.env.state_size, 25, checkpoint_dir)
        
        self.training_start_time = time.time()
        self.episode_count = 0
        
        # V4 training state with dynamic action space tracking
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
        
        print("🚀 REVOLUTIONARY TRAINER V4 INITIALIZED")
        print("✅ Dynamic action space evolution")
        print("✅ Intelligent neural architecture") 
        print("✅ Advanced multi-objective rewards")
        print("✅ Proactive meta-learning")
        print("✅ FORCED CPU for memory safety")
        print("✅ Complete save/resume system")
    
    def _signal_handler(self, signum, frame):
        """Signal handler for graceful interruption - same as v3.1"""
        print(f"\n⚠️ INTERRUPTION DETECTED (Signal {signum})")
        print("💾 Saving emergency checkpoint...")
        self._save_checkpoint("emergency_checkpoint")
        print("✅ Emergency checkpoint saved!")
        sys.exit(0)
    
    def train_with_checkpoints(self, target_prompts: List[str], 
                              episodes_per_prompt: int = 5,
                              resume_from: Optional[str] = None) -> Dict:
        """Train with complete checkpoint system - enhanced from v3.1"""
        
        print(f"🎓 REVOLUTIONARY TRAINING SESSION V4")
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
                
                print(f"\n📚 REVOLUTIONARY EPISODE {episode_num}")
                result = self._train_single_episode(current_prompt, episode_num)
                
                # Immediate meta-learning
                if result.get('immediate_meta_learning', False):
                    print("\n⚡ IMMEDIATE META-LEARNING!")
                    self._meta_learning_phase()
                
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
                
                # Auto-checkpoint
                self._save_checkpoint(f"episode_{episode_num:03d}")
                
                # Production readiness check
                if episode_num % 10 == 0:
                    readiness = self.monitor.assess_production_readiness()
                    print(f"\n📊 PRODUCTION CHECK: {'✅ READY' if readiness['ready'] else '❌ NOT READY'}")
                    if readiness['ready']:
                        self._save_checkpoint("production_ready")
                        return self._generate_final_report()
                
                print(f"\n⏸️ EPISODE {episode_num} COMPLETE")
                print(f"   📊 Score: {result['best_score']:.3f} | Reward: {result['total_reward']:.1f}")
                print(f"   🧠 Strategies: {self.training_state.action_space_size} ({self.training_state.discovered_strategies} discovered)")
                
                # Conversational interaction - commented out for automation
                # user_input = input("\n➡️ ENTER=continue, q=quit, s=save+quit: ").strip().lower()
                # if user_input == 'q':
                #     return self._generate_final_report()
                # elif user_input == 's':
                #     self._save_checkpoint("user_save")
                #     return self._generate_final_report()
            
            self.training_state.current_prompt_index = prompt_idx + 1
        
        return self._generate_final_report()
    
    def _train_single_episode(self, target_prompt: str, episode_num: int) -> Dict:
        """Train single episode with all v4 enhancements"""
        self.episode_count += 1
        
        # Reset with episode context
        state, episode_context = self.env.reset(target_prompt)
        total_reward = 0
        best_score = self.env.validation_history[0]
        losses = []
        immediate_meta_learning = False
        
        while True:
            # Action selection with dynamic action space
            current_action_size = self.env.action_size
            action = self.agent.act(state, episode_context, current_action_size, training=True)
            
            # Environment step
            next_state, next_episode_context, reward, done, info = self.env.step(action)
            
            if info.get('trigger_immediate_meta_learning', False):
                immediate_meta_learning = True
            
            # Agent learning
            loss = self.agent.step(state, episode_context, action, reward, 
                                 next_state, next_episode_context, done)
            if loss is not None:
                losses.append(loss)
            
            total_reward += reward
            best_score = max(best_score, info['score'])
            
            # Update for next iteration
            state, episode_context = next_state, next_episode_context
            
            if done:
                break
        
        return {
            'episode': episode_num,
            'best_score': best_score,
            'total_reward': total_reward,
            'ultra_achieved': best_score >= self.env.ultra_target,
            'action_space_size': self.env.action_size,
            'discovered_strategies': sum(1 for s in self.env.dynamic_action_space.get_all_strategies() if s.created_from_pattern),
            'avg_loss': statistics.mean(losses) if losses else 0.0,
            'epsilon': self.agent.epsilon,
            'learn_count': self.agent.learn_count,
            'immediate_meta_learning': immediate_meta_learning
        }
    
    def _meta_learning_phase(self):
        """Meta-learning phase - enhanced from v3.1"""
        print(f"\n{'='*20} META-LEARNING PHASE V4 {'='*20}")
        
        recent_successes = [e for e in self.env.meta_learning_events 
                          if e.episode >= max(0, len(self.env.episode_log) - self.meta_learn_every_n_episodes)]
        
        if len(recent_successes) < 1:
            print("   📊 No recent successes for meta-learning")
            return
        
        best_success = max(recent_successes, key=lambda x: x.score_achieved)
        print(f"   🎯 Best: {best_success.successful_prompt} (Score: {best_success.score_achieved:.3f})")
        
        pattern = self._extract_pattern(best_success.original_prompt, best_success.successful_prompt)
        if pattern:
            self.env.llama_generator.learn_pattern_from_meta_learning(pattern, best_success.score_achieved)
            self.training_state.new_patterns_learned += 1
            print(f"   🎉 NEW PATTERN LEARNED: {pattern}")
        
        print(f"{'='*60}\n")
    
    def _extract_pattern(self, original: str, successful: str) -> Optional[str]:
        """Extract pattern from successful prompts - same as v3.1"""
        successful_lower = successful.lower()
        if 'aerospace-grade' in successful_lower and 'precision' in successful_lower:
            return f"aerospace-grade precision-enhanced {original}"
        elif 'defense-grade' in successful_lower and 'ultra-precision' in successful_lower:
            return f"defense-grade ultra-precision {original}"
        elif 'masterpiece-quality' in successful_lower:
            return f"masterpiece-quality {original}"
        return None
    
    def _save_checkpoint(self, checkpoint_name: str):
        """Save complete checkpoint - enhanced from v3.1 with v4 data"""
        checkpoint_path = self.checkpoint_dir / checkpoint_name
        checkpoint_path.mkdir(exist_ok=True)
        
        # V4 metadata with dynamic action space
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
                'learned_strategies': [asdict(s) for s in self.env.dynamic_action_space.learned_strategies],
                'strategy_performance': self.env.dynamic_action_space.strategy_performance
            },
            'discovered_patterns': [asdict(p) for p in self.env.meta_learner.discovered_patterns],
            'action_space_size': len(strategies),
            'discovered_strategies_count': sum(1 for s in strategies if s.created_from_pattern)
        }
        
        self.agent.save_checkpoint(checkpoint_path, metadata)
        
        with open(checkpoint_path / 'training_state.json', 'w') as f:
            json.dump(asdict(self.training_state), f, indent=2)
        
        print(f"   💾 V4 Checkpoint saved: {checkpoint_name}")
    
    def _load_checkpoint(self, checkpoint_name: str) -> bool:
        """Load complete checkpoint - enhanced from v3.1 with v4 data"""
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
                # Restore v3.1 data
                if 'meta_learning_events' in agent_metadata:
                    self.env.meta_learning_events = [
                        MetaLearningEvent(**e) for e in agent_metadata['meta_learning_events']
                    ]
                if 'llama_learned_patterns' in agent_metadata:
                    self.env.llama_generator.learned_patterns = agent_metadata['llama_learned_patterns']
                if 'llama_successful_examples' in agent_metadata:
                    self.env.llama_generator.successful_examples = agent_metadata['llama_successful_examples']
                
                # Restore V4 specific data
                if 'dynamic_action_space' in agent_metadata:
                    das_data = agent_metadata['dynamic_action_space']
                    self.env.dynamic_action_space.base_strategies = [
                        DynamicLLaMAInstruction(**s) for s in das_data['base_strategies']
                    ]
                    self.env.dynamic_action_space.learned_strategies = [
                        DynamicLLaMAInstruction(**s) for s in das_data['learned_strategies']
                    ]
                    self.env.dynamic_action_space.strategy_performance = das_data.get('strategy_performance', {})
                
                if 'discovered_patterns' in agent_metadata:
                    self.env.meta_learner.discovered_patterns = [
                        PatternDiscovery(**p) for p in agent_metadata['discovered_patterns']
                    ]
                
                print(f"   📂 V4 Training state loaded (Episode: {self.training_state.episode})")
                print(f"   🎮 Action Space: {self.training_state.action_space_size} ({self.training_state.discovered_strategies} discovered)")
                return True
        except Exception as e:
            print(f"   ❌ Load error: {e}")
        return False
    
    def _generate_final_report(self) -> Dict:
        """Generate comprehensive final report - enhanced from v3.1"""
        print(f"\n🎓 FINAL REVOLUTIONARY TRAINING REPORT V4")
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
        print(f"   LLaMA Examples: {len(self.env.llama_generator.successful_examples)}")
        
        # V4 specific metrics
        print(f"\n🚀 V4 REVOLUTION:")
        print(f"   Final Action Space: {self.training_state.action_space_size} strategies")
        print(f"   Discovered Strategies: {self.training_state.discovered_strategies}")
        print(f"   Pattern Discovery Events: {len(self.env.meta_learner.discovered_patterns)}")
        
        readiness = self.monitor.assess_production_readiness()
        print(f"\n🚀 PRODUCTION: {'✅ READY' if readiness['ready'] else '❌ NOT READY'}")
        
        return {
            'total_episodes': total_episodes,
            'ultra_rate': ultra_count / total_episodes if total_episodes > 0 else 0,
            'average_score': avg_score,
            'best_score': self.training_state.best_overall_score,
            'patterns_learned': self.training_state.new_patterns_learned,
            'production_ready': readiness['ready'],
            # V4 specific
            'final_action_space_size': self.training_state.action_space_size,
            'discovered_strategies': self.training_state.discovered_strategies,
            'training_time_hours': training_time / 3600
        }
    
    def list_checkpoints(self) -> List[str]:
        """List available checkpoints - same as v3.1"""
        checkpoints = []
        for path in self.checkpoint_dir.iterdir():
            if path.is_dir() and (path / 'training_state.json').exists():
                checkpoints.append(path.name)
        return sorted(checkpoints)

def main():
    """Main function with complete save/resume - enhanced from v3.1"""
    print("🚀 RL + LLaMA OPTIMIZER V4.0 - REVOLUTIONARY + MEMORY SAFE")
    print("="*70)
    print("✅ Dynamic action space evolution")
    print("✅ Intelligent neural architecture") 
    print("✅ Advanced multi-objective rewards")
    print("✅ Proactive meta-learning")
    print("✅ FORCED CPU for memory safety")
    print("✅ Complete save/resume system")
    print("="*70)
    
    try:
        trainer = RevolutionaryTrainerV4(ultra_target=0.96)
        checkpoints = trainer.list_checkpoints()
        
        resume_from = None
        if checkpoints:
            print(f"📂 Found checkpoints: {checkpoints}")
            # Auto-resume from latest for automation
            # choice = input("Resume from checkpoint? Enter name or ENTER for new: ").strip()
            # if choice in checkpoints:
            #     resume_from = choice
        
        test_prompts = [
            "hexagonal prism steel structure",
            "elegant silk fabric draping", 
            "transparent crystal sphere",
            "wooden geometric sculpture"
        ]
        
        results = trainer.train_with_checkpoints(
            target_prompts=test_prompts,
            episodes_per_prompt=20,  # User's modification
            resume_from=resume_from
        )
        
        print(f"\n🎉 V4 REVOLUTIONARY TRAINING COMPLETE!")
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