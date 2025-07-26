#!/usr/bin/env python3
"""
RL + LLaMA Optimizer v3.1 - MEMORY-SAFE VERSION
==============================================
FIXES:
✅ RL Agent uses CPU to avoid CUDA OOM with OLLaMA
✅ CUDA memory cleanup after validation
✅ Memory monitoring and error handling
✅ Graceful fallback on GPU memory issues

Addresses the validation score 0.0 issue caused by GPU memory conflicts.
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
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        print(f"   🔍 GPU Memory: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved")
        if reserved > 8.0:
            print(f"   🔧 High memory usage - cleaning up")
            torch.cuda.empty_cache()

def cleanup_gpu_memory():
    """Aggressive GPU memory cleanup"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        time.sleep(0.5)

@dataclass
class TrainingCheckpoint:
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
    episode: int
    original_prompt: str
    successful_prompt: str
    extracted_pattern: str
    score_achieved: float
    timestamp: float

@dataclass
class LLaMAInstruction:
    strategy_name: str
    creativity_level: float
    focus_area: str
    enhancement_type: str
    risk_level: str
    length_target: str

# Memory-safe LLaMA Generator
class LLaMACustomGenerator:
    def __init__(self, ollama_url: str = "http://localhost:11434"):
        self.ollama_url = ollama_url
        self.model = "llama3.2:3b"
        self.successful_examples = []
        self.failed_examples = []
        self.learned_patterns = []
        self._test_connection()
        print("🧠 LLaMA CUSTOM GENERATOR INITIALIZED (Memory-Safe)")
    
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
        
        # Clean memory before generation
        cleanup_gpu_memory()
        
        system_prompt = self._build_system_prompt(instruction)
        user_prompt = self._build_user_prompt(original_prompt, instruction)
        
        try:
            response = self._query_llama(system_prompt, user_prompt, instruction.creativity_level)
            return self._extract_custom_prompt(response, original_prompt)
        except Exception as e:
            print(f"   ❌ LLaMA generation failed (possible CUDA OOM): {e}")
            cleanup_gpu_memory()
            return self._fallback_prompt(original_prompt, instruction)
    
    def _build_system_prompt(self, instruction: LLaMAInstruction) -> str:
        base_prompt = f"""You are an expert 3D prompt optimizer.

STRATEGY: {instruction.strategy_name}
FOCUS: {instruction.focus_area}
CREATIVITY: {instruction.creativity_level:.1f}/1.0

REQUIREMENTS:
1. Start with "wbgmsst,"
2. End with ", white background"
3. Keep it concise to avoid GPU memory issues

FORMAT:
CUSTOM_PROMPT: [Your optimized prompt]"""

        if self.successful_examples:
            base_prompt += "\n\nSUCCESSFUL EXAMPLES:"
            for ex in self.successful_examples[-2:]:  # Limit examples to save memory
                base_prompt += f"\n{ex['custom']} (Score: {ex['score']:.3f})"

        return base_prompt
    
    def _build_user_prompt(self, original: str, instruction: LLaMAInstruction) -> str:
        return f"""OPTIMIZE: "{original}"

Strategy: {instruction.strategy_name}
Focus: {instruction.focus_area}

Create a concise, custom optimization."""
    
    def _query_llama(self, system_prompt: str, user_prompt: str, creativity: float) -> str:
        # Clean memory before query
        cleanup_gpu_memory()
        
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
                "num_predict": 150  # Reduced to save memory
            }
        }
        
        try:
            response = requests.post(f"{self.ollama_url}/api/chat", json=data, timeout=30)
            response.raise_for_status()
            
            # Clean memory after query
            cleanup_gpu_memory()
            
            return response.json()["message"]["content"].strip()
            
        except Exception as e:
            print(f"   🔥 LLaMA query failed: {e}")
            cleanup_gpu_memory()
            raise e
    
    def _extract_custom_prompt(self, response: str, original: str) -> str:
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
            self.successful_examples = self.successful_examples[-5:]  # Limit memory
            print(f"   🧠 LLaMA learned success: {strategy} → {score:.3f}")

# Memory-safe Environment
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
        
        print(f"🎮 ENVIRONMENT V3 INITIALIZED (Memory-Safe)")
    
    @property
    def action_size(self):
        return len(self.action_space)
    
    def _define_action_space(self) -> List[LLaMAInstruction]:
        return [
            LLaMAInstruction("material_precision", 0.3, "material", "precision", "conservative", "medium"),
            LLaMAInstruction("material_artistic", 0.8, "material", "artistic", "balanced", "detailed"),
            LLaMAInstruction("shape_geometric", 0.4, "shape", "precision", "balanced", "medium"),
            LLaMAInstruction("quality_premium", 0.5, "quality", "premium", "balanced", "medium"),
            LLaMAInstruction("quality_masterpiece", 0.7, "quality", "artistic", "aggressive", "detailed"),
            LLaMAInstruction("conservative_safe", 0.2, "quality", "precision", "conservative", "concise"),
            LLaMAInstruction("balanced_optimal", 0.5, "quality", "premium", "balanced", "medium"),
            LLaMAInstruction("aggressive_max", 0.9, "quality", "artistic", "aggressive", "detailed"),
        ]
    
    def reset(self, target_prompt: str) -> np.ndarray:
        self.target_prompt = target_prompt
        self.current_prompt = f"wbgmsst, {target_prompt}, white background"
        self.validation_history = []
        self.step_count = 0
        
        # Clean memory before validation
        check_gpu_memory()
        
        initial_score = self._validate_prompt(self.current_prompt)
        self.validation_history.append(initial_score)
        
        print(f"🔄 RESET: {target_prompt} (Baseline: {initial_score:.3f})")
        return self._get_state()
    
    def step(self, action_idx: int):
        self.step_count += 1
        action = self.action_space[action_idx]
        
        print(f"🎬 STEP {self.step_count}: {action.strategy_name}")
        
        old_score = self.validation_history[-1]
        
        # Generate custom prompt
        custom_prompt = self.llama_generator.generate_custom_prompt(self.target_prompt, action)
        
        # Validate with memory safety
        new_score = self._validate_prompt(custom_prompt)
        self.validation_history.append(new_score)
        
        if new_score > old_score:
            self.current_prompt = custom_prompt
        
        self.llama_generator.learn_from_feedback(
            self.target_prompt, custom_prompt, new_score, action.strategy_name
        )
        
        reward = self._calculate_reward(old_score, new_score, custom_prompt)
        done = (new_score >= self.ultra_target or self.step_count >= self.max_steps)
        
        print(f"   📝 {custom_prompt}")
        print(f"   📊 {old_score:.3f} → {new_score:.3f} (Reward: {reward:.1f})")
        
        info = {
            'score': new_score,
            'custom_prompt': custom_prompt,
            'strategy_used': action.strategy_name,
            'improvement': new_score - old_score,
            'ultra_achieved': new_score >= self.ultra_target,
            'trigger_immediate_meta_learning': new_score >= 0.85
        }
        
        return self._get_state(), reward, done, info
    
    def _validate_prompt(self, prompt: str) -> float:
        try:
            # Clean memory before validation
            cleanup_gpu_memory()
            check_gpu_memory()
            
            cmd = [sys.executable, "subnet_accurate_validator.py", prompt]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
            
            if result.returncode != 0:
                print(f"   ❌ Validation failed (return code {result.returncode})")
                if "CUDA" in result.stderr or "out of memory" in result.stderr.lower():
                    print(f"   🔥 CUDA OOM detected in validation")
                    cleanup_gpu_memory()
                    time.sleep(2)
                return 0.0
            
            with open("subnet_validation_results.json", 'r') as f:
                data = json.load(f)
                score = data.get("validation_engine_score", 0.0)
                
                if score == 0.0:
                    print(f"   🔧 Score 0.0 - potential memory issue, cleaning up")
                    cleanup_gpu_memory()
                
                return score
                
        except Exception as e:
            print(f"   ❌ Validation error: {e}")
            cleanup_gpu_memory()
            return 0.0
    
    def _calculate_reward(self, old_score: float, new_score: float, prompt: str) -> float:
        improvement = new_score - old_score
        base_reward = improvement * 100
        
        if new_score >= self.ultra_target:
            base_reward += 200
        elif new_score >= 0.9:
            base_reward += 100
        elif new_score >= 0.8:
            base_reward += 50
        
        if len(prompt) > 120:
            base_reward -= 10
        
        if new_score > max(self.validation_history[:-1]) if len(self.validation_history) > 1 else 0:
            base_reward += 30
        
        return base_reward
    
    def _get_state(self) -> np.ndarray:
        state = np.zeros(self.state_size)
        
        for i, score in enumerate(self.validation_history[-3:]):
            if i < 3:
                state[i] = score
        
        state[3] = self.step_count / self.max_steps
        state[4] = max(self.validation_history) if self.validation_history else 0.0
        state[5] = min(len(self.llama_generator.successful_examples) / 5, 1.0)
        
        target_lower = self.target_prompt.lower()
        state[7] = 1.0 if any(w in target_lower for w in ["steel", "metal"]) else 0.0
        state[8] = 1.0 if any(w in target_lower for w in ["fabric", "silk"]) else 0.0
        state[9] = 1.0 if any(w in target_lower for w in ["glass", "crystal"]) else 0.0
        
        return state

# CPU-Only DQN Agent (Memory Safe)
class CustomDQNAgentV3:
    def __init__(self, state_size: int, action_size: int, checkpoint_dir: str):
        self.state_size = state_size
        self.action_size = action_size
        self.checkpoint_dir = Path(checkpoint_dir)
        
        # FORCE CPU to avoid CUDA OOM with OLLaMA
        self.device = torch.device("cpu")
        
        self.q_network_local = self._build_network().to(self.device)
        self.q_network_target = self._build_network().to(self.device)
        self.optimizer = optim.Adam(self.q_network_local.parameters(), lr=0.001)
        
        # Simplified memory buffer to save RAM
        self.memory = deque(maxlen=2000)  # Reduced from 5000
        self.batch_size = 16  # Reduced from 32
        self.gamma = 0.95
        self.tau = 0.005
        self.update_every = 4  # Less frequent updates
        self.epsilon = 0.9
        self.epsilon_min = 0.05
        self.epsilon_decay = 0.98
        self.step_count = 0
        self.learn_count = 0
        
        print(f"🤖 DQN AGENT V3 INITIALIZED (CPU-ONLY for memory safety)")
    
    def _build_network(self):
        return nn.Sequential(
            nn.Linear(self.state_size, 64),  # Smaller network
            nn.ReLU(),
            nn.Linear(64, 64),
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
        self.memory.append((state, action, reward, next_state, done))
        self.step_count += 1
        
        if self.step_count % self.update_every == 0 and len(self.memory) >= self.batch_size:
            self.learn()
            if self.epsilon > self.epsilon_min:
                self.epsilon *= self.epsilon_decay
    
    def learn(self):
        self.learn_count += 1
        
        # Simple random sampling (no PER to save memory)
        batch = random.sample(self.memory, self.batch_size)
        
        states = torch.FloatTensor([e[0] for e in batch]).to(self.device)
        actions = torch.LongTensor([e[1] for e in batch]).to(self.device)
        rewards = torch.FloatTensor([e[2] for e in batch]).to(self.device)
        next_states = torch.FloatTensor([e[3] for e in batch]).to(self.device)
        dones = torch.BoolTensor([e[4] for e in batch]).to(self.device)

        current_q_values = self.q_network_local(states).gather(1, actions.unsqueeze(1))
        next_q_values = self.q_network_target(next_states).detach().max(1)[0]
        target_q_values = rewards + (self.gamma * next_q_values * ~dones)

        loss = F.mse_loss(current_q_values.squeeze(), target_q_values)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # Soft update
        for target_param, local_param in zip(self.q_network_target.parameters(), self.q_network_local.parameters()):
            target_param.data.copy_(self.tau * local_param.data + (1.0 - self.tau) * target_param.data)
        
        print(f"   📚 LEARNING #{self.learn_count}: Loss {loss.item():.4f}, ε={self.epsilon:.3f}")

def main():
    print("🚀 RL + LLaMA OPTIMIZER V3.1 - MEMORY-SAFE")
    print("="*50)
    print("✅ CPU-only RL agent to avoid CUDA OOM")
    print("✅ GPU memory cleanup and monitoring")
    print("✅ Graceful handling of validation failures")
    print("="*50)
    
    try:
        # Simple training without complex checkpointing to test memory safety
        from CustomPromptEnvironmentV3 import CustomPromptEnvironmentV3
        from CustomDQNAgentV3 import CustomDQNAgentV3
        
        env = CustomPromptEnvironmentV3(ultra_target=0.96)
        agent = CustomDQNAgentV3(env.state_size, env.action_size, "rl_checkpoints_v3")
        
        test_prompts = ["hexagonal prism steel structure", "elegant silk fabric draping"]
        
        for prompt in test_prompts:
            print(f"\n🎯 TESTING: {prompt}")
            
            state = env.reset(prompt)
            total_reward = 0
            
            for step in range(6):
                action = agent.act(state)
                next_state, reward, done, info = env.step(action)
                agent.step(state, action, reward, next_state, done)
                
                total_reward += reward
                state = next_state
                
                # Monitor memory
                check_gpu_memory()
                
                if done:
                    break
            
            print(f"   🏁 Episode complete: Reward {total_reward:.1f}, Best Score {max(env.validation_history):.3f}")
        
        print("\n✅ Memory-safe testing complete!")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        cleanup_gpu_memory()

if __name__ == "__main__":
    main() 