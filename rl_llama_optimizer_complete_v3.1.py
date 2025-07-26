#!/usr/bin/env python3
"""
RL + LLaMA Optimizer v3.1 - COMPLETE WITH ALL V2 FEATURES
=========================================================
✅ LLaMA 3.2 generates CUSTOM prompts instead of templates
✅ RL agent learns optimal instruction strategies for LLaMA  
✅ Uses subnet_accurate_validator.py directly
✅ ALL v2 features: PER, checkpointing, meta-learning, conversational
✅ TrainingCheckpoint, TrainingMetrics, ProductionReadinessMonitor
✅ Signal handlers, graceful interruption, save/resume
✅ Meta-learning phases that teach LLaMA new patterns
✅ Complete training management and progress tracking

Revolutionary approach: RL learns WHEN/HOW to instruct LLaMA for optimal results
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

Experience = namedtuple('Experience', ['state', 'action', 'reward', 'next_state', 'done'])

def check_gpu_memory():
    """Monitor GPU memory usage"""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3  # GB
        reserved = torch.cuda.memory_reserved() / 1024**3    # GB
        print(f"   🔍 GPU Memory: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved")
        
        # If memory usage is high, trigger cleanup
        if reserved > 8.0:  # More than 8GB reserved
            print(f"   🔧 High memory usage detected - cleaning up")
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
    else:
        print(f"   💾 Using CPU memory (CUDA not available)")

@dataclass
class TrainingCheckpoint:
    """Complete training state checkpoint"""
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
class LLaMAInstruction:
    """Instructions for LLaMA custom prompt generation"""
    strategy_name: str
    creativity_level: float
    focus_area: str
    enhancement_type: str
    risk_level: str
    length_target: str

# Prioritized Experience Replay
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

# Production Readiness Monitor
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

# LLaMA Custom Generator
class LLaMACustomGenerator:
    def __init__(self, ollama_url: str = "http://localhost:11434"):
        self.ollama_url = ollama_url
        self.model = "llama3.2:3b"
        self.successful_examples = []
        self.failed_examples = []
        self.learned_patterns = []
        self._test_connection()
        print("🧠 LLaMA CUSTOM GENERATOR INITIALIZED")
    
    def _test_connection(self):
        try:
            response = requests.get(f"{self.ollama_url}/api/tags", timeout=5)
            if response.status_code == 200:
                print("   ✅ LLaMA 3.2 Connected")
            else:
                raise Exception(f"Connection failed: {response.status_code}")
        except Exception as e:
            print(f"❌ LLaMA 3.2 unavailable: {e}")
    
    def generate_custom_prompt(self, original_prompt: str, instruction: LLaMAInstruction) -> str:
        print(f"   🧠 LLaMA Strategy: {instruction.strategy_name}")
        print(f"   🎯 Focus: {instruction.focus_area}, Risk: {instruction.risk_level}")
        
        system_prompt = self._build_system_prompt(instruction)
        user_prompt = self._build_user_prompt(original_prompt, instruction)
        
        try:
            response = self._query_llama(system_prompt, user_prompt, instruction.creativity_level)
            return self._extract_custom_prompt(response, original_prompt)
        except Exception as e:
            print(f"   ❌ LLaMA generation failed: {e}")
            return self._fallback_prompt(original_prompt, instruction)
    
    def _build_system_prompt(self, instruction: LLaMAInstruction) -> str:
        base_prompt = f"""You are an expert 3D prompt optimizer that creates CUSTOM prompts for maximum scores.

STRATEGY: {instruction.strategy_name}
FOCUS: {instruction.focus_area}
ENHANCEMENT: {instruction.enhancement_type}
RISK: {instruction.risk_level}
CREATIVITY: {instruction.creativity_level:.1f}/1.0

REQUIREMENTS:
1. MUST start with "wbgmsst,"
2. MUST end with ", white background"
3. PRESERVE main object from original
4. CREATE custom descriptions, not templates
5. FOCUS on {instruction.focus_area} optimization

FORMAT:
ANALYSIS: [Brief analysis]
CUSTOM_PROMPT: [Your optimized prompt]
REASONING: [Why this will score 0.9+]"""

        if self.successful_examples:
            base_prompt += "\n\nSUCCESSFUL EXAMPLES:"
            for ex in self.successful_examples[-3:]:
                base_prompt += f"\n{ex['custom']} (Score: {ex['score']:.3f})"
        
        if self.learned_patterns:
            base_prompt += "\n\nLEARNED PATTERNS:"
            for pattern in self.learned_patterns[-3:]:
                base_prompt += f"\n{pattern['pattern']} (Score: {pattern['score']:.3f})"

        return base_prompt
    
    def _build_user_prompt(self, original: str, instruction: LLaMAInstruction) -> str:
        return f"""OPTIMIZE: "{original}"

Strategy: {instruction.strategy_name}
Focus: {instruction.focus_area}
Style: {instruction.enhancement_type}
Risk: {instruction.risk_level}

Should be short and concise. 
Examples of good patterns:
- wbgmsst, aerospace-grade precision-engineered {original}, ultra-high technical specification, white background
- wbgmsst, defense-grade ultra-precision {original}, premium excellence, white background

Based on the context of the original prompt, create a completely custom optimization that addresses its unique characteristics.
Analyze this specific prompt and create a completely custom optimization that addresses its unique characteristics.
Think about what makes THIS object special and how to enhance those qualities for 3D generation.

IMPORTANT: Only output the optimized prompt. Do NOT include any explanations, analysis, or extra text. Do NOT say anything except the prompt itself.
GENERATE YOUR CUSTOM PROMPT:"""
    
    def _query_llama(self, system_prompt: str, user_prompt: str, creativity: float) -> str:
        # Clear CUDA cache before LLaMA query
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
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
                "num_predict": 200
            }
        }
        
        try:
            response = requests.post(f"{self.ollama_url}/api/chat", json=data, timeout=30)
            response.raise_for_status()
            
            # Clear cache after successful query
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                
            return response.json()["message"]["content"].strip()
            
        except Exception as e:
            print(f"   🔥 LLaMA query failed (possible CUDA OOM): {e}")
            # Aggressive cleanup on failure
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
                time.sleep(1)
            raise e
    
    # def _extract_custom_prompt(self, response: str, original: str) -> str:
    #     lines = response.split('\n')
        
    #     # Look for CUSTOM_PROMPT line
    #     for line in lines:
    #         if line.strip().startswith('CUSTOM_PROMPT:'):
    #             prompt = line.split('CUSTOM_PROMPT:', 1)[1].strip()
    #             return self._clean_prompt(prompt, original)
        
    #     # Fallback: find wbgmsst line
    #     for line in lines:
    #         if 'wbgmsst' in line.lower():
    #             return self._clean_prompt(line.strip(), original)
        
    #     return self._fallback_prompt(original, None)
    def _extract_custom_prompt(self, response: str, original: str) -> str:
        """Extract custom prompt from LLaMA response"""
        
        lines = response.split('\n')
        custom_prompt = None
        
        # Look for CUSTOM_PROMPT section
        for line in lines:
            if line.strip().startswith('CUSTOM_PROMPT:'):
                custom_prompt = line.split('CUSTOM_PROMPT:', 1)[1].strip()
                break
        
        # Fallback: look for wbgmsst line
        if not custom_prompt:
            for line in lines:
                if 'wbgmsst' in line.lower():
                    custom_prompt = line.strip()
                    break
        
        # Clean and validate
        if custom_prompt:
            custom_prompt = custom_prompt.replace('"', '').strip()
            
            if not custom_prompt.startswith('wbgmsst'):
                custom_prompt = f"wbgmsst, {custom_prompt}"
            if not custom_prompt.endswith('white background'):
                if custom_prompt.endswith(','):
                    custom_prompt += " white background"
                else:
                    custom_prompt += ", white background"
            
            return custom_prompt
        
        # Ultimate fallback
        print(f"   ❌ Fallback prompt ")
        return self._fallback_prompt(original, None)
    
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
    
    def _fallback_prompt(self, original: str, instruction: Optional[LLaMAInstruction]) -> str:
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
            self.successful_examples = self.successful_examples[-8:]
            print(f"   🧠 LLaMA learned success: {strategy} → {score:.3f}")
        elif score < 0.5:
            self.failed_examples.append({
                'original': original, 'custom': custom,
                'score': score, 'strategy': strategy
            })
            self.failed_examples = self.failed_examples[-4:]
    
    def learn_pattern_from_meta_learning(self, pattern: str, score: float):
        self.learned_patterns.append({'pattern': pattern, 'score': score})
        self.learned_patterns = self.learned_patterns[-5:]
        print(f"   🌟 LLaMA learned meta-pattern: {pattern} (Score: {score:.3f})")

# Environment
class CustomPromptEnvironmentV3:
    def __init__(self, ultra_target: float = 0.96, checkpoint_dir: str = "rl_checkpoints_v3"):
        self.ultra_target = ultra_target
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(exist_ok=True)
        
        self.llama_generator = LLaMACustomGenerator()
        self.target_prompt = ""
        self.current_prompt = ""
        self.validation_history = []
        self.step_count = 0
        self.max_steps = 6
        self.action_space = self._define_action_space()
        self.state_size = 25
        self.meta_learning_events = []
        self.meta_learn_score_threshold = 0.8
        self.session_id = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.episode_log = []
        
        print(f"🎮 ENVIRONMENT V3 INITIALIZED (Actions: {len(self.action_space)})")
    
    @property
    def action_size(self):
        return len(self.action_space)
    
    def _define_action_space(self) -> List[LLaMAInstruction]:
        return [
            LLaMAInstruction("material_precision", 0.3, "material", "precision", "conservative", "medium"),
            LLaMAInstruction("material_artistic", 0.8, "material", "artistic", "balanced", "detailed"),
            LLaMAInstruction("material_technical", 0.2, "material", "technical", "conservative", "concise"),
            LLaMAInstruction("shape_geometric", 0.4, "shape", "precision", "balanced", "medium"),
            LLaMAInstruction("shape_creative", 0.9, "shape", "artistic", "aggressive", "detailed"),
            LLaMAInstruction("quality_premium", 0.5, "quality", "premium", "balanced", "medium"),
            LLaMAInstruction("quality_masterpiece", 0.7, "quality", "artistic", "aggressive", "detailed"),
            LLaMAInstruction("quality_professional", 0.3, "quality", "technical", "conservative", "medium"),
            LLaMAInstruction("context_studio", 0.4, "context", "precision", "balanced", "medium"),
            LLaMAInstruction("context_artistic", 0.8, "context", "artistic", "aggressive", "detailed"),
            LLaMAInstruction("conservative_safe", 0.2, "quality", "precision", "conservative", "concise"),
            LLaMAInstruction("balanced_optimal", 0.5, "quality", "premium", "balanced", "medium"),
            LLaMAInstruction("aggressive_max", 0.9, "quality", "artistic", "aggressive", "detailed"),
        ]
    
    def reset(self, target_prompt: str) -> np.ndarray:
        self.target_prompt = target_prompt
        self.current_prompt = f"wbgmsst, {target_prompt}, white background"
        self.validation_history = []
        self.step_count = 0
        
        initial_score = self._validate_prompt(self.current_prompt)
        self.validation_history.append(initial_score)
        
        self.episode_log.append({
            'timestamp': time.time(),
            'target_prompt': target_prompt,
            'initial_score': initial_score,
            'session_id': self.session_id
        })
        
        print(f"🔄 RESET: {target_prompt} (Baseline: {initial_score:.3f})")
        return self._get_state()
    
    def step(self, action_idx: int):
        self.step_count += 1
        action = self.action_space[action_idx]
        
        print(f"🎬 STEP {self.step_count}: {action.strategy_name}")
        
        old_score = self.validation_history[-1]
        
        # LLaMA generates custom prompt
        custom_prompt = self.llama_generator.generate_custom_prompt(self.target_prompt, action)
        new_score = self._validate_prompt(custom_prompt)
        self.validation_history.append(new_score)
        
        if new_score > old_score:
            self.current_prompt = custom_prompt
        
        # LLaMA learns from feedback
        self.llama_generator.learn_from_feedback(
            self.target_prompt, custom_prompt, new_score, action.strategy_name
        )
        
        reward = self._calculate_reward(old_score, new_score, custom_prompt)
        done = (new_score >= self.ultra_target or self.step_count >= self.max_steps)
        
        print(f"   📝 {custom_prompt}")
        print(f"   📊 {old_score:.3f} → {new_score:.3f} (Reward: {reward:.1f})")
        
        # Meta-learning tracking
        info = {
            'score': new_score,
            'custom_prompt': custom_prompt,
            'strategy_used': action.strategy_name,
            'improvement': new_score - old_score,
            'ultra_achieved': new_score >= self.ultra_target,
            'trigger_immediate_meta_learning': False
        }
        
        if new_score >= self.ultra_target:
            self._record_success_for_meta_learning(self.target_prompt, custom_prompt, new_score)
            info['trigger_immediate_meta_learning'] = True
        elif new_score >= self.meta_learn_score_threshold:
            should_trigger = self._record_high_score_for_meta_learning(custom_prompt, new_score)
            info['trigger_immediate_meta_learning'] = should_trigger
        
        return self._get_state(), reward, done, info
    
    def _record_success_for_meta_learning(self, original: str, successful: str, score: float):
        event = MetaLearningEvent(
            episode=len(self.episode_log),
            original_prompt=original,
            successful_prompt=successful,
            extracted_pattern="",
            score_achieved=score,
            timestamp=time.time()
        )
        self.meta_learning_events.append(event)
        print(f"   🌟 SUCCESS RECORDED for meta-learning (Score: {score:.3f})")
    
    def _record_high_score_for_meta_learning(self, prompt: str, score: float):
        # Record the success first
        self._record_success_for_meta_learning(self.target_prompt, prompt, score)
        
        current_episode = len(self.episode_log)
        if not hasattr(self, '_last_immediate_trigger') or self._last_immediate_trigger != current_episode:
            # Trigger on first 0.85+ score OR significant improvement
            if score >= 0.85:
                print(f"   🚀 IMMEDIATE META-LEARNING: High score achieved {score:.3f}")
                self._last_immediate_trigger = current_episode
                return True
            elif len(self.meta_learning_events) >= 2:
                latest_score = score
                previous_best = max([e.score_achieved for e in self.meta_learning_events[:-1]], default=0.0)
                if latest_score > previous_best + 0.03:
                    print(f"   🚀 IMMEDIATE META-LEARNING: Score improvement {previous_best:.3f} → {latest_score:.3f}")
                    self._last_immediate_trigger = current_episode
                    return True
        return False
    
    def _validate_prompt(self, prompt: str) -> float:
        try:
            # Clear CUDA cache before validation to prevent OOM
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
            
            cmd = [sys.executable, "subnet_accurate_validator.py", prompt]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
            
            if result.returncode != 0:
                print(f"   ❌ Validation failed (return code {result.returncode})")
                if "CUDA" in result.stderr or "out of memory" in result.stderr.lower():
                    print(f"   🔥 CUDA OOM detected in validation - clearing cache")
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                        time.sleep(2)  # Brief pause for memory cleanup
                return 0.0
            
            with open("subnet_validation_results.json", 'r') as f:
                data = json.load(f)
                score = data.get("validation_engine_score", 0.0)
                
                # If score is 0.0, might be OOM - try cleanup
                if score == 0.0 and torch.cuda.is_available():
                    print(f"   🔧 Score 0.0 - clearing CUDA cache")
                    torch.cuda.empty_cache()
                
                return score
                
        except Exception as e:
            print(f"   ❌ Validation error: {e}")
            # Clean up memory on any error
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            return 0.0
    
    def _calculate_reward(self, old_score: float, new_score: float, prompt: str) -> float:
        improvement = new_score - old_score
        base_reward = improvement * 100
        
        # Score bonuses
        if new_score >= self.ultra_target:
            base_reward += 200
        elif new_score >= 0.9:
            base_reward += 100
        elif new_score >= 0.8:
            base_reward += 50
        
        # Length penalty
        if len(prompt) > 120:
            base_reward -= 10
        
        # Personal best bonus
        if new_score > max(self.validation_history[:-1]) if len(self.validation_history) > 1 else 0:
            base_reward += 30
        
        return base_reward
    
    def _get_state(self) -> np.ndarray:
        state = np.zeros(self.state_size)
        
        # Recent scores
        for i, score in enumerate(self.validation_history[-3:]):
            if i < 3:
                state[i] = score
        
        # Progress
        state[3] = self.step_count / self.max_steps
        state[4] = max(self.validation_history) if self.validation_history else 0.0
        
        # LLaMA learning state
        state[5] = min(len(self.llama_generator.successful_examples) / 5, 1.0)
        state[6] = min(len(self.llama_generator.learned_patterns) / 5, 1.0)
        
        # Prompt characteristics
        target_lower = self.target_prompt.lower()
        state[7] = 1.0 if any(w in target_lower for w in ["steel", "metal"]) else 0.0
        state[8] = 1.0 if any(w in target_lower for w in ["fabric", "silk"]) else 0.0
        state[9] = 1.0 if any(w in target_lower for w in ["glass", "crystal"]) else 0.0
        
        return state

# DQN Agent
class CustomDQNAgentV3:
    def __init__(self, state_size: int, action_size: int, checkpoint_dir: str):
        self.state_size = state_size
        self.action_size = action_size
        self.checkpoint_dir = Path(checkpoint_dir)
        # Force CPU to avoid CUDA OOM conflicts with OLLaMA
        self.device = torch.device("cpu")
        
        self.q_network_local = self._build_network().to(self.device)
        self.q_network_target = self._build_network().to(self.device)
        self.optimizer = optim.Adam(self.q_network_local.parameters(), lr=0.001)
        
        self.memory = PrioritizedReplayBuffer(capacity=5000)
        self.beta = 0.4
        self.beta_increment = 0.001
        self.batch_size = 32
        self.gamma = 0.95
        self.tau = 0.005
        self.update_every = 2
        self.epsilon = 0.9
        self.epsilon_min = 0.05
        self.epsilon_decay = 0.98
        self.step_count = 0
        self.learn_count = 0
        
        print(f"🤖 DQN AGENT V3 INITIALIZED ({self.device}) - CPU mode for GPU memory safety")
    
    def _build_network(self):
        return nn.Sequential(
            nn.Linear(self.state_size, 128),
            nn.ReLU(),
            nn.LayerNorm(128),
            nn.Dropout(0.1),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.LayerNorm(128),
            nn.Dropout(0.1),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, self.action_size)
        )
    
    def act(self, state: np.ndarray, training: bool = True) -> int:
        if training and random.random() < self.epsilon:
            action = random.randrange(self.action_size)
            print(f"   🎲 EXPLORATION: Strategy {action} (ε={self.epsilon:.3f})")
            return action
        else:
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            with torch.no_grad():
                q_values = self.q_network_local(state_tensor)
            action = q_values.argmax().item()
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
        actions = torch.LongTensor([e.action for e in experiences]).unsqueeze(1).to(self.device)
        rewards = torch.FloatTensor([e.reward for e in experiences]).to(self.device)
        next_states = torch.FloatTensor([e.next_state for e in experiences]).to(self.device)
        dones = torch.BoolTensor([e.done for e in experiences]).to(self.device)
        weights = torch.FloatTensor(weights).unsqueeze(1).to(self.device)

        current_q_values = self.q_network_local(states).gather(1, actions)
        next_q_values = self.q_network_target(next_states).detach().max(1)[0]
        target_q_values = rewards + (self.gamma * next_q_values * ~dones)

        td_errors = torch.abs(target_q_values.unsqueeze(1) - current_q_values).detach().cpu().numpy()
        self.memory.update_priorities(indices, td_errors.squeeze() + 1e-5)

        loss = (weights * F.mse_loss(current_q_values, target_q_values.unsqueeze(1), reduction='none')).mean()
        
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_network_local.parameters(), 1.0)
        self.optimizer.step()
        
        # Soft update
        for target_param, local_param in zip(self.q_network_target.parameters(), self.q_network_local.parameters()):
            target_param.data.copy_(self.tau * local_param.data + (1.0 - self.tau) * target_param.data)
        
        print(f"   📚 LEARNING #{self.learn_count}: Loss {loss.item():.4f}, ε={self.epsilon:.3f}")
        return loss.item()
    
    def save_checkpoint(self, checkpoint_path: Path, metadata: Dict):
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



# Main Optimizer
class RLLLaMAOptimizerV3:
    def __init__(self, ultra_target: float = 0.96, checkpoint_dir: str = "rl_checkpoints_v3"):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(exist_ok=True)
        
        self.env = CustomPromptEnvironmentV3(ultra_target, checkpoint_dir)
        self.agent = CustomDQNAgentV3(self.env.state_size, self.env.action_size, checkpoint_dir)
        
        self.meta_learn_every_n_episodes = 4
        self.training_state = TrainingCheckpoint(
            episode=0, total_episodes_completed=0, current_prompt_index=0,
            training_prompts=[], episodes_per_prompt=0, episode_rewards=[],
            episode_scores=[], ultra_achievements=[], epsilon=self.agent.epsilon,
            step_count=0, learn_count=0, best_overall_score=0.0,
            training_start_time=time.time(), last_checkpoint_time=time.time()
        )
        
        self.monitor = ProductionReadinessMonitor()
        signal.signal(signal.SIGINT, self._signal_handler)
        
        print("🚀 RL + LLaMA OPTIMIZER V3.1 INITIALIZED")
        print("✅ Custom prompts, PER, checkpointing, meta-learning, production monitoring")
    
    def _signal_handler(self, signum, frame):
        print(f"\n⚠️ INTERRUPTION DETECTED (Signal {signum})")
        print("💾 Saving emergency checkpoint...")
        self._save_checkpoint("emergency_checkpoint")
        print("✅ Emergency checkpoint saved!")
        sys.exit(0)

    def _meta_learning_phase(self):
        print(f"\n{'='*20} META-LEARNING PHASE {'='*20}")
        
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
        successful_lower = successful.lower()
        if 'aerospace-grade' in successful_lower and 'precision' in successful_lower:
            return f"aerospace-grade precision-enhanced {original}"
        elif 'defense-grade' in successful_lower and 'ultra-precision' in successful_lower:
            return f"defense-grade ultra-precision {original}"
        elif 'masterpiece-quality' in successful_lower:
            return f"masterpiece-quality {original}"
        return None

    def train_with_checkpoints(self, target_prompts: List[str], 
                              episodes_per_prompt: int = 5,
                              resume_from: Optional[str] = None,
                              continuous_learning: bool = True) -> Dict:
        
        print(f"🎓 TRAINING SESSION V3.1")
        print(f"📝 Prompts: {len(target_prompts)} | Episodes each: {episodes_per_prompt}")
        
        if continuous_learning:
            print("🔄 CONTINUOUS LEARNING: Will cycle through prompts indefinitely")
        print("=" * 60)
        
        if resume_from and self._load_checkpoint(resume_from):
            print(f"📂 RESUMED FROM: {resume_from}")
        else:
            self.training_state.training_prompts = target_prompts
            self.training_state.episodes_per_prompt = episodes_per_prompt
        
        total_prompts = len(self.training_state.training_prompts)
        
        # Start from current prompt index, but support cycling
        prompt_idx = self.training_state.current_prompt_index
        
        while True:  # Continuous learning loop
            # If we've completed all prompts and continuous learning is enabled
            if prompt_idx >= total_prompts:
                if continuous_learning:
                    print(f"\n🔄 COMPLETED CYCLE! Starting new learning cycle from prompt 1")
                    print(f"   📈 Knowledge preserved: ε={self.agent.epsilon:.3f}, Patterns={self.training_state.new_patterns_learned}")
                    prompt_idx = 0  # Reset to first prompt
                    self.training_state.current_prompt_index = 0
                    # Keep episode numbering continuous for better tracking
                else:
                    break  # Exit if not continuous learning
            
            current_prompt = self.training_state.training_prompts[prompt_idx]
            print(f"\n🎯 PROMPT {prompt_idx + 1}/{total_prompts}: '{current_prompt}'")
            
            episodes_completed = 0
            if prompt_idx == self.training_state.current_prompt_index:
                # For resumed training, calculate episodes done in current prompt
                total_episodes_done = self.training_state.episode
                episodes_in_current_prompt = total_episodes_done % episodes_per_prompt
                if total_episodes_done > 0 and episodes_in_current_prompt == 0:
                    # If we just completed a prompt, start fresh with next prompt
                    episodes_completed = 0
                else:
                    episodes_completed = episodes_in_current_prompt
            
            
             # Check if this prompt is already completed
            if episodes_completed >= episodes_per_prompt:
                print(f"   ✅ Prompt {prompt_idx + 1} already completed, moving to next")
                prompt_idx += 1
                self.training_state.current_prompt_index = prompt_idx
                continue
                
            for episode_in_prompt in range(episodes_completed, episodes_per_prompt):
                episode_num = self.training_state.episode + 1 
                
                # Scheduled meta-learning
                if episode_num > 0 and episode_num % self.meta_learn_every_n_episodes == 0:
                    self._meta_learning_phase()
                
                print(f"\n📚 EPISODE {episode_num} (Prompt {prompt_idx + 1}, Episode {episode_in_prompt + 1}/{episodes_per_prompt})")
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
                        if not continuous_learning:
                            return self._generate_final_report()
                
                print(f"\n⏸️ EPISODE {episode_num} COMPLETE")
                print(f"   📊 Score: {result['best_score']:.3f} | Reward: {result['total_reward']:.1f}")
                print(f"   🧠 Patterns: {self.training_state.new_patterns_learned}")
                
                # Conversational
                # user_input = input("\n➡️ ENTER=continue, q=quit, s=save+quit: ").strip().lower()
                # user_input = 'e'

                # Conversational controls
                if continuous_learning:
                    # In continuous mode, show status and options
                    status = self.get_current_status()
                    print(f"   🎯 Prompt {status['current_prompt_index'] + 1}/{status['total_prompts']}: {status['current_prompt']}")
                    user_input = input("\n➡️ ENTER=continue, q=quit, s=save+quit, r=reset to prompt 1: ").strip().lower()
                else:
                    user_input = input("\n➡️ ENTER=continue, q=quit, s=save+quit: ").strip().lower()
                
                if user_input == 'q':
                    return self._generate_final_report()
                elif user_input == 's':
                    self._save_checkpoint("user_save")
                    return self._generate_final_report()
                elif user_input == 'r' and continuous_learning:
                    self.reset_to_prompt_cycle_start()
                    prompt_idx = 0  # Reset loop to start from prompt 1
                    self.training_state.current_prompt_index = 0
                    break  # Break out of episode loop to sta
 
            
            prompt_idx += 1
            self.training_state.current_prompt_index = prompt_idx
        
        return self._generate_final_report()

    def _train_single_episode(self, target_prompt: str, episode_num: int) -> Dict:
        state = self.env.reset(target_prompt)
        total_reward = 0
        best_score = self.env.validation_history[0]
        losses = []
        immediate_meta_learning = False
        
        while True:
            action = self.agent.act(state, training=True)
            next_state, reward, done, info = self.env.step(action)
            
            if info.get('trigger_immediate_meta_learning', False):
                immediate_meta_learning = True
            
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
            'avg_loss': statistics.mean(losses) if losses else 0.0,
            'immediate_meta_learning': immediate_meta_learning
        }

    def _save_checkpoint(self, checkpoint_name: str):
        checkpoint_path = self.checkpoint_dir / checkpoint_name
        checkpoint_path.mkdir(exist_ok=True)
        
        metadata = {
            'episode': self.training_state.episode,
            'training_prompts': self.training_state.training_prompts,
            'best_overall_score': self.training_state.best_overall_score,
            'new_patterns_learned': self.training_state.new_patterns_learned,
            'meta_learning_events': [asdict(e) for e in self.env.meta_learning_events],
            'llama_learned_patterns': self.env.llama_generator.learned_patterns,
            'llama_successful_examples': self.env.llama_generator.successful_examples
        }
        
        self.agent.save_checkpoint(checkpoint_path, metadata)
        
        with open(checkpoint_path / 'training_state.json', 'w') as f:
            json.dump(asdict(self.training_state), f, indent=2)
        
        print(f"   💾 Checkpoint saved: {checkpoint_name}")

    def _load_checkpoint(self, checkpoint_name: str) -> bool:
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
                if 'meta_learning_events' in agent_metadata:
                    self.env.meta_learning_events = [
                        MetaLearningEvent(**e) for e in agent_metadata['meta_learning_events']
                    ]
                if 'llama_learned_patterns' in agent_metadata:
                    self.env.llama_generator.learned_patterns = agent_metadata['llama_learned_patterns']
                if 'llama_successful_examples' in agent_metadata:
                    self.env.llama_generator.successful_examples = agent_metadata['llama_successful_examples']
                
                print(f"   📂 Training state loaded (Episode: {self.training_state.episode})")
                return True
        except Exception as e:
            print(f"   ❌ Load error: {e}")
        return False

    def _generate_final_report(self) -> Dict:
        print(f"\n🎓 FINAL TRAINING REPORT V3.1")
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
        
        readiness = self.monitor.assess_production_readiness()
        print(f"\n🚀 PRODUCTION: {'✅ READY' if readiness['ready'] else '❌ NOT READY'}")
        
        return {
            'total_episodes': total_episodes,
            'ultra_rate': ultra_count / total_episodes if total_episodes > 0 else 0,
            'average_score': avg_score,
            'best_score': self.training_state.best_overall_score,
            'patterns_learned': self.training_state.new_patterns_learned,
            'production_ready': readiness['ready']
        }

    def list_checkpoints(self) -> List[str]:
        checkpoints = []
        for path in self.checkpoint_dir.iterdir():
            if path.is_dir() and (path / 'training_state.json').exists():
                checkpoints.append(path.name)
        return sorted(checkpoints)

    def reset_to_prompt_cycle_start(self, save_checkpoint: bool = True):
        """Reset to prompt 1 while preserving all learned knowledge"""
        if save_checkpoint:
            self._save_checkpoint("cycle_reset")
        
        # Reset prompt index but keep all learning
        old_prompt_idx = self.training_state.current_prompt_index
        self.training_state.current_prompt_index = 0
        
        print(f"🔄 RESET TO PROMPT 1 (was at prompt {old_prompt_idx + 1})")
        print(f"   📈 Knowledge preserved: ε={self.agent.epsilon:.3f}")
        print(f"   🧠 Patterns learned: {self.training_state.new_patterns_learned}")
        print(f"   📊 LLaMA examples: {len(self.env.llama_generator.successful_examples)}")
        
    def get_current_status(self) -> Dict:
        """Get current training status"""
        return {
            'current_episode': self.training_state.episode,
            'current_prompt_index': self.training_state.current_prompt_index,
            'total_prompts': len(self.training_state.training_prompts),
            'current_prompt': self.training_state.training_prompts[self.training_state.current_prompt_index] if self.training_state.training_prompts else None,
            'epsilon': self.agent.epsilon,
            'patterns_learned': self.training_state.new_patterns_learned,
            'best_score': self.training_state.best_overall_score,
            'llama_examples': len(self.env.llama_generator.successful_examples)
        }

def main():
    print("🚀 RL + LLaMA OPTIMIZER V3.1 - COMPLETE")
    print("="*50)
    print("✅ LLaMA custom prompts + RL strategy learning")
    print("✅ All v2 features: PER, checkpoints, meta-learning")
    print("✅ Production monitoring + conversational interaction")
    print("✅ CONTINUOUS LEARNING: Cycles through prompts indefinitely")
    print("="*50)
    
    try:
        optimizer = RLLLaMAOptimizerV3(ultra_target=0.96)
        checkpoints = optimizer.list_checkpoints()
        
        resume_from = None
        if checkpoints:
            print(f"📂 Found checkpoints: {checkpoints}")
            choice = input("Resume from checkpoint? Enter name or ENTER for new: ").strip()
            if choice in checkpoints:
                resume_from = choice

        # Ask about continuous learning mode
        continuous_mode = True
        mode_choice = input("Enable continuous learning? (Y/n): ").strip().lower()
        if mode_choice in ['n', 'no']:
            continuous_mode = False
        
        
        test_prompts = [
            "hexagonal prism steel structure",
            "elegant silk fabric draping", 
            "transparent glass sphere"
        ]

        print(f"\n{'🔄 CONTINUOUS' if continuous_mode else '📚 SINGLE CYCLE'} TRAINING MODE")
        if continuous_mode:
            print("⚠️  Training will cycle through prompts indefinitely")
            print("💡 Use Ctrl+C to save and exit at any time")
        
        
        results = optimizer.train_with_checkpoints(
            target_prompts=test_prompts,
            episodes_per_prompt=10,
            resume_from=resume_from,
            continuous_learning=continuous_mode
        )
        
        print(f"\n🎉 TRAINING COMPLETE!")
        print(f"📈 Ultra Rate: {results.get('ultra_rate', 0):.1%}")
        print(f"🧠 Patterns: {results.get('patterns_learned', 0)}")
        print(f"🚀 Production: {results.get('production_ready', False)}")
    
    except KeyboardInterrupt:
        print(f"\n⚠️ Training interrupted by user")
        print(f"💾 Emergency checkpoint should be saved")
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 