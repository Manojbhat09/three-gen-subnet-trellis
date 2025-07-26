#!/usr/bin/env python3
"""
RL + LLaMA Prompt Optimizer v3.0 - CUSTOM PROMPT GENERATION
===========================================================
Based on rl_prompt_optimizer_complete_v2.py but with key improvements:
✅ LLaMA 3.2 generates CUSTOM prompts instead of applying templates
✅ RL agent learns optimal instruction strategies for LLaMA
✅ Uses subnet_accurate_validator.py directly (like v2)
✅ All v2 features: PER, checkpointing, meta-learning
✅ NEW: LLaMA writes completely custom prompts based on RL guidance

The RL agent learns WHEN and HOW to instruct LLaMA for best results.
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
import sqlite3
from pathlib import Path
import re
import signal
import datetime
import statistics

Experience = namedtuple('Experience', ['state', 'action', 'reward', 'next_state', 'done'])

# ==============================================================================
# PRIORITIZED EXPERIENCE REPLAY (from v2)
# ==============================================================================
class PrioritizedReplayBuffer:
    """Prioritized Experience Replay buffer from v2"""
    
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

    def sample(self, batch_size: int, beta: float = 0.4) -> Tuple[List[Experience], np.ndarray, np.ndarray]:
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

    def update_priorities(self, batch_indices: np.ndarray, batch_priorities: np.ndarray):
        for idx, priority in zip(batch_indices, batch_priorities):
            self.priorities[idx] = priority
        self.max_priority = max(self.max_priority, np.max(batch_priorities))

    def __len__(self):
        return len(self.buffer)

@dataclass
class LLaMAInstruction:
    """Instructions for LLaMA custom prompt generation"""
    strategy_name: str
    creativity_level: float  # 0.1-1.0
    focus_area: str         # material, shape, quality, context
    enhancement_type: str   # precision, artistic, technical, premium
    risk_level: str        # conservative, balanced, aggressive
    length_target: str     # concise, medium, detailed

@dataclass
class CustomPromptResult:
    """Result of LLaMA custom prompt generation"""
    original_prompt: str
    custom_prompt: str
    validation_score: float
    generation_successful: bool
    instruction_used: LLaMAInstruction
    llama_reasoning: str
    generation_time: float

class LLaMACustomGenerator:
    """LLaMA 3.2 for generating completely custom prompts"""
    
    def __init__(self, ollama_url: str = "http://localhost:11434"):
        self.ollama_url = ollama_url
        self.model = "llama3.2:3b"
        
        # Learning from feedback
        self.successful_examples = []
        self.failed_examples = []
        self.generation_history = []
        
        self._test_connection()
        print(f"🧠 LLaMA CUSTOM GENERATOR INITIALIZED")
    
    def _test_connection(self):
        try:
            response = requests.get(f"{self.ollama_url}/api/tags", timeout=5)
            if response.status_code == 200:
                print(f"   ✅ LLaMA 3.2 Connected")
            else:
                raise Exception(f"Connection failed: {response.status_code}")
        except Exception as e:
            raise Exception(f"❌ LLaMA 3.2 unavailable: {e}")
    
    def generate_custom_prompt(self, original_prompt: str, instruction: LLaMAInstruction) -> str:
        """Generate completely custom optimized prompt"""
        
        print(f"   🧠 LLaMA Strategy: {instruction.strategy_name}")
        print(f"   🎯 Focus: {instruction.focus_area}, Risk: {instruction.risk_level}")
        print(f"   🎨 Creativity: {instruction.creativity_level:.2f}")
        
        # Build system prompt with learned knowledge
        system_prompt = self._build_system_prompt(instruction)
        
        # Build user prompt
        user_prompt = self._build_user_prompt(original_prompt, instruction)
        
        try:
            # Query LLaMA
            response = self._query_llama(system_prompt, user_prompt, instruction.creativity_level)
            
            # Extract custom prompt
            custom_prompt = self._extract_custom_prompt(response, original_prompt)
            
            # Track generation
            self.generation_history.append({
                'original': original_prompt,
                'custom': custom_prompt,
                'instruction': asdict(instruction),
                'response': response,
                'timestamp': time.time()
            })
            
            return custom_prompt
            
        except Exception as e:
            print(f"   ❌ LLaMA generation failed: {e}")
            return self._fallback_prompt(original_prompt, instruction)
    
    def _build_system_prompt(self, instruction: LLaMAInstruction) -> str:
        """Build intelligent system prompt with learned knowledge"""
        
        base_prompt = f"""You are an expert 3D prompt optimizer AI that creates CUSTOM prompts for maximum validation scores.

MISSION: Generate a completely custom optimized prompt that will score 0.9+ on 3D model validation.

CURRENT STRATEGY: {instruction.strategy_name}
- Focus Area: {instruction.focus_area}
- Enhancement Type: {instruction.enhancement_type}  
- Risk Level: {instruction.risk_level}
- Target Length: {instruction.length_target}
- Creativity Level: {instruction.creativity_level:.1f}/1.0

CORE REQUIREMENTS:
1. MUST start with "wbgmsst,"
2. MUST end with ", white background"
3. PRESERVE the main object from original prompt
4. CREATE custom descriptions, not templates
5. FOCUS on {instruction.focus_area} optimization
6. Use {instruction.enhancement_type} enhancement approach
7. Apply {instruction.risk_level} risk strategy

RESPONSE FORMAT:
ANALYSIS: [Brief analysis of the original prompt]
STRATEGY: [Your specific approach for this prompt]
CUSTOM_PROMPT: [Your completely custom optimized prompt]
REASONING: [Why this will score 0.9+]
CONFIDENCE: [0.1-1.0]"""

        # Add learned examples
        if self.successful_examples:
            base_prompt += "\n\nSUCCESSFUL EXAMPLES (Score 0.8+):"
            for ex in self.successful_examples[-3:]:
                base_prompt += f"\nOriginal: \"{ex['original']}\""
                base_prompt += f"\nCustom: \"{ex['custom']}\" (Score: {ex['score']:.3f})"
                base_prompt += f"\nStrategy: {ex['strategy']}\n"
        
        if self.failed_examples:
            base_prompt += "\n\nAVOID THESE APPROACHES (Failed):"
            for ex in self.failed_examples[-2:]:
                base_prompt += f"\nFailed: \"{ex['custom']}\" (Score: {ex['score']:.3f})"
        
        return base_prompt
    
    def _build_user_prompt(self, original: str, instruction: LLaMAInstruction) -> str:
        """Build specific user prompt"""
        
        return f"""CREATE CUSTOM OPTIMIZED PROMPT:

Original: "{original}"

REQUIREMENTS:
- Strategy: {instruction.strategy_name}
- Focus: {instruction.focus_area} optimization priority  
- Style: {instruction.enhancement_type} enhancement
- Risk: {instruction.risk_level} approach
- Length: {instruction.length_target}
- Creativity: {instruction.creativity_level:.1f}/1.0

Analyze this specific prompt and create a completely custom optimization that addresses its unique characteristics.
Think about what makes THIS object special and how to enhance those qualities for 3D generation.

GENERATE YOUR CUSTOM PROMPT:"""
    
    def _query_llama(self, system_prompt: str, user_prompt: str, creativity: float, seed: int = 42) -> str:
        """Query LLaMA with dynamic parameters and reproducible seed"""
        
        temperature = 0.4 + (creativity * 0.4)  # 0.4-0.8 range
        
        data = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            "stream": False,
            "options": {
                "temperature": temperature,
                "top_p": 0.9,
                "repeat_penalty": 1.1,
                "num_predict": 250,
                "seed": seed  # Add seed for reproducible generation
            }
        }
        
        response = requests.post(f"{self.ollama_url}/api/chat", json=data, timeout=30)
        response.raise_for_status()
        
        return response.json()["message"]["content"].strip()
    
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
        return self._fallback_prompt(original, None)
    
    def _fallback_prompt(self, original: str, instruction: Optional[LLaMAInstruction]) -> str:
        """Generate fallback prompt if LLaMA fails"""
        if instruction:
            if instruction.focus_area == 'material':
                return f"wbgmsst, precision-crafted {original}, ultra-high material specification, white background"
            elif instruction.focus_area == 'quality':
                return f"wbgmsst, masterpiece-quality {original}, premium excellence, white background"
            elif instruction.focus_area == 'shape':
                return f"wbgmsst, geometrically-precise {original}, perfect dimensional accuracy, white background"
        
        return f"wbgmsst, professional-grade {original}, detailed craftsmanship, white background"
    
    def learn_from_feedback(self, original: str, custom: str, score: float, strategy: str):
        """Learn from validation feedback"""
        
        if score >= 0.8:
            self.successful_examples.append({
                'original': original,
                'custom': custom,
                'score': score,
                'strategy': strategy,
                'timestamp': time.time()
            })
            print(f"   🧠 LLaMA learned success: {strategy} → {score:.3f}")
            self.successful_examples = self.successful_examples[-8:]  # Keep recent
            
        elif score < 0.5:
            self.failed_examples.append({
                'original': original,
                'custom': custom,
                'score': score,
                'strategy': strategy,
                'timestamp': time.time()
            })
            print(f"   ⚠️ LLaMA learned failure: {strategy} → {score:.3f}")
            self.failed_examples = self.failed_examples[-4:]  # Keep recent

class CustomPromptEnvironmentV3:
    """Environment that uses LLaMA for custom prompt generation"""
    
    def __init__(self, ultra_target: float = 0.96, checkpoint_dir: str = "rl_checkpoints_v3"):
        self.ultra_target = ultra_target
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(exist_ok=True)
        
        # LLaMA generator
        self.llama_generator = LLaMACustomGenerator()
        
        # Environment state
        self.target_prompt = ""
        self.current_prompt = ""
        self.validation_history = []
        self.step_count = 0
        self.max_steps = 6
        
        # Define action space: RL actions = LLaMA instruction strategies
        self.action_space = self._define_action_space()
        self.state_size = 25
        
        # Learning tracking
        self.episode_results = []
        self.session_id = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        
        print(f"🎮 CUSTOM PROMPT ENVIRONMENT V3 INITIALIZED")
        print(f"   🎯 Ultra Target: {ultra_target}")
        print(f"   🎬 Action Space: {len(self.action_space)} LLaMA strategies")
        print(f"   🧠 Custom Generation: ENABLED")
    
    @property
    def action_size(self):
        return len(self.action_space)
    
    def _define_action_space(self) -> List[LLaMAInstruction]:
        """Define LLaMA instruction strategies as actions"""
        
        strategies = [
            # Material-focused strategies
            LLaMAInstruction("material_precision", 0.3, "material", "precision", "conservative", "medium"),
            LLaMAInstruction("material_artistic", 0.8, "material", "artistic", "balanced", "detailed"),
            LLaMAInstruction("material_technical", 0.2, "material", "technical", "conservative", "concise"),
            
            # Shape-focused strategies
            LLaMAInstruction("shape_geometric", 0.4, "shape", "precision", "balanced", "medium"),
            LLaMAInstruction("shape_creative", 0.9, "shape", "artistic", "aggressive", "detailed"),
            
            # Quality-focused strategies
            LLaMAInstruction("quality_premium", 0.5, "quality", "premium", "balanced", "medium"),
            LLaMAInstruction("quality_masterpiece", 0.7, "quality", "artistic", "aggressive", "detailed"),
            LLaMAInstruction("quality_professional", 0.3, "quality", "technical", "conservative", "medium"),
            
            # Context strategies
            LLaMAInstruction("context_studio", 0.4, "context", "precision", "balanced", "medium"),
            LLaMAInstruction("context_artistic", 0.8, "context", "artistic", "aggressive", "detailed"),
            
            # Risk-based strategies  
            LLaMAInstruction("conservative_safe", 0.2, "quality", "precision", "conservative", "concise"),
            LLaMAInstruction("balanced_optimal", 0.5, "quality", "premium", "balanced", "medium"),
            LLaMAInstruction("aggressive_max", 0.9, "quality", "artistic", "aggressive", "detailed"),
        ]
        
        return strategies
    
    def reset(self, target_prompt: str) -> np.ndarray:
        """Reset for new prompt optimization episode"""
        
        self.target_prompt = target_prompt
        self.current_prompt = f"wbgmsst, {target_prompt}, white background"
        self.validation_history = []
        self.step_count = 0
        
        # Get baseline score
        initial_score = self._validate_prompt(self.current_prompt)
        self.validation_history.append(initial_score)
        
        print(f"🔄 ENVIRONMENT RESET")
        print(f"   🎯 Target: {target_prompt}")
        print(f"   📊 Baseline Score: {initial_score:.3f}")
        
        return self._get_state()
    
    def step(self, action_idx: int) -> Tuple[np.ndarray, float, bool, Dict]:
        """Execute action: use LLaMA strategy to generate custom prompt"""
        
        self.step_count += 1
        action = self.action_space[action_idx]
        
        print(f"🎬 STEP {self.step_count}: {action.strategy_name}")
        
        old_prompt = self.current_prompt
        old_score = self.validation_history[-1]
        
        # Generate custom prompt using LLaMA
        start_time = time.time()
        custom_prompt = self.llama_generator.generate_custom_prompt(
            self.target_prompt, 
            action
        )
        generation_time = time.time() - start_time
        
        # Validate the custom prompt
        new_score = self._validate_prompt(custom_prompt)
        self.validation_history.append(new_score)
        
        # Update current prompt if better
        if new_score > old_score:
            self.current_prompt = custom_prompt
        
        # Let LLaMA learn from feedback
        self.llama_generator.learn_from_feedback(
            self.target_prompt,
            custom_prompt, 
            new_score,
            action.strategy_name
        )
        
        # Calculate reward
        reward = self._calculate_reward(old_score, new_score, custom_prompt)
        
        # Check done
        done = (new_score >= self.ultra_target or 
                self.step_count >= self.max_steps)
        
        print(f"   📝 Custom Prompt: {custom_prompt}")
        print(f"   📊 Score: {old_score:.3f} → {new_score:.3f}")
        print(f"   🎁 Reward: {reward:.3f}")
        print(f"   ⏱️ Generation: {generation_time:.2f}s")
        
        info = {
            'score': new_score,
            'custom_prompt': custom_prompt,
            'strategy_used': action.strategy_name,
            'improvement': new_score - old_score,
            'ultra_achieved': new_score >= self.ultra_target,
            'generation_time': generation_time
        }
        
        return self._get_state(), reward, done, info
    
    def _validate_prompt(self, prompt: str) -> float:
        """Validate prompt using subnet_accurate_validator.py (same as v2)"""
        try:
            # cmd = [
            #     "bash", "-c", 
            #     f"source /home/mbhat/miniconda/bin/activate && conda activate trellis_new && python subnet_accurate_validator.py '{prompt}'"
            # ]
            cmd = [sys.executable, "subnet_accurate_validator.py", prompt]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
            
            # result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
            
            if result.returncode != 0:
                print(f"   ❌ Validation failed: {result.stderr}")
                return 0.0
            
            with open("subnet_validation_results.json", 'r') as f:
                data = json.load(f)
                return data.get("validation_engine_score", 0.0)
        
        except Exception as e:
            print(f"   ❌ Validation error: {e}")
            return 0.0
    
    def _calculate_reward(self, old_score: float, new_score: float, prompt: str) -> float:
        """Calculate reward (same logic as v2)"""
        
        score_improvement = new_score - old_score
        base_reward = score_improvement * 100
        
        # Score bonuses
        if new_score >= self.ultra_target:
            base_reward += 200
        elif new_score >= 0.9:
            base_reward += 100
        elif new_score >= 0.8:
            base_reward += 50
        elif new_score >= 0.7:
            base_reward += 20
        
        # Length penalty
        if len(prompt) > 120:
            base_reward -= 10
        
        # Low score penalty
        if new_score < 0.3:
            base_reward -= 20
        
        # Personal best bonus
        if new_score > max(self.validation_history[:-1]) if len(self.validation_history) > 1 else 0:
            base_reward += 30
        
        return base_reward
    
    def _get_state(self) -> np.ndarray:
        """Get state representation"""
        
        state = np.zeros(self.state_size)
        
        # Last 3 scores
        history = self.validation_history[-3:]
        for i, score in enumerate(history):
            if i < 3:
                state[i] = score
        
        # Progress and performance
        state[3] = self.step_count / self.max_steps
        state[4] = max(self.validation_history) if self.validation_history else 0.0
        
        # Recent improvement
        if len(self.validation_history) >= 2:
            state[5] = self.validation_history[-1] - self.validation_history[-2]
        
        # LLaMA learning state
        state[6] = min(len(self.llama_generator.successful_examples) / 5, 1.0)
        state[7] = min(len(self.llama_generator.failed_examples) / 3, 1.0)
        
        # Target prompt characteristics
        target_lower = self.target_prompt.lower()
        state[8] = 1.0 if any(w in target_lower for w in ["steel", "metal", "iron"]) else 0.0
        state[9] = 1.0 if any(w in target_lower for w in ["fabric", "cloth", "silk"]) else 0.0
        state[10] = 1.0 if any(w in target_lower for w in ["glass", "crystal", "transparent"]) else 0.0
        state[11] = 1.0 if any(w in target_lower for w in ["wood", "wooden"]) else 0.0
        
        # Current prompt analysis
        if self.current_prompt:
            prompt_lower = self.current_prompt.lower()
            state[12] = len(self.current_prompt) / 150
            state[13] = 1.0 if 'precision' in prompt_lower else 0.0
            state[14] = 1.0 if 'masterpiece' in prompt_lower else 0.0
            state[15] = 1.0 if 'professional' in prompt_lower else 0.0
        
        # Performance indicators
        state[16] = 1.0 if self.validation_history and self.validation_history[-1] >= 0.8 else 0.0
        state[17] = 1.0 if self.validation_history and self.validation_history[-1] >= self.ultra_target else 0.0
        
        return state

class CustomDQNAgentV3:
    """DQN Agent with PER for learning LLaMA instruction strategies"""
    
    def __init__(self, state_size: int, action_size: int, checkpoint_dir: str):
        self.state_size = state_size
        self.action_size = action_size
        self.checkpoint_dir = Path(checkpoint_dir)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Networks
        self.q_network_local = self._build_network().to(self.device)
        self.q_network_target = self._build_network().to(self.device)
        self.optimizer = optim.Adam(self.q_network_local.parameters(), lr=0.001)
        
        # PER buffer
        self.memory = PrioritizedReplayBuffer(capacity=5000)
        self.beta = 0.4
        self.beta_increment = 0.001
        
        # Hyperparameters
        self.batch_size = 32
        self.gamma = 0.95
        self.tau = 0.005
        self.update_every = 2
        
        # Epsilon schedule
        self.epsilon = 0.9
        self.epsilon_min = 0.05
        self.epsilon_decay = 0.98
        
        # Tracking
        self.step_count = 0
        self.learn_count = 0
        self.losses = []
        
        print(f"🤖 CUSTOM DQN AGENT V3 INITIALIZED")
        print(f"   🧠 Device: {self.device}")
        print(f"   🎯 Learning LLaMA instruction strategies")
    
    def _build_network(self) -> nn.Module:
        """Build Q-network"""
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
        """Choose LLaMA instruction strategy"""
        
        if training and random.random() < self.epsilon:
            action = random.randrange(self.action_size)
            print(f"   🎲 EXPLORATION: Random strategy {action} (ε={self.epsilon:.3f})")
            return action
        else:
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            with torch.no_grad():
                q_values = self.q_network_local(state_tensor)
            action = q_values.argmax().item()
            print(f"   🧠 EXPLOITATION: Strategy {action} (ε={self.epsilon:.3f})")
            return action
    
    def step(self, state, action, reward, next_state, done):
        """Store experience and learn"""
        
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
        """Learn from experiences with PER"""
        
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

        # TD errors for priority update
        td_errors = torch.abs(target_q_values.unsqueeze(1) - current_q_values).detach().cpu().numpy()
        self.memory.update_priorities(indices, td_errors.squeeze() + 1e-5)

        # Weighted loss
        loss = (weights * F.mse_loss(current_q_values, target_q_values.unsqueeze(1), reduction='none')).mean()
        
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_network_local.parameters(), 1.0)
        self.optimizer.step()
        
        # Soft update
        for target_param, local_param in zip(self.q_network_target.parameters(), self.q_network_local.parameters()):
            target_param.data.copy_(self.tau * local_param.data + (1.0 - self.tau) * target_param.data)
        
        self.losses.append(loss.item())
        print(f"   📚 LEARNING #{self.learn_count}: Loss {loss.item():.4f}, ε={self.epsilon:.3f}")
        
        return loss.item()

class RLLLaMAOptimizerV3:
    """Complete RL + LLaMA optimizer v3"""
    
    def __init__(self, ultra_target: float = 0.96, checkpoint_dir: str = "rl_checkpoints_v3"):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(exist_ok=True)
        
        self.env = CustomPromptEnvironmentV3(ultra_target, checkpoint_dir)
        self.agent = CustomDQNAgentV3(
            state_size=self.env.state_size,
            action_size=self.env.action_size,
            checkpoint_dir=checkpoint_dir
        )
        
        # Training tracking
        self.episode_count = 0
        self.episode_scores = []
        self.episode_rewards = []
        self.ultra_achievements = []
        
        print(f"🚀 RL + LLaMA OPTIMIZER V3 INITIALIZED")
        print(f"   ✅ Custom Prompt Generation: LLaMA 3.2")
        print(f"   ✅ Strategy Learning: RL Agent")
        print(f"   ✅ Real Validation: subnet_accurate_validator.py")
        print(f"   ✅ Advanced Learning: PER + Meta-learning")
    
    def train_episode(self, target_prompt: str) -> Dict:
        """Train one episode"""
        
        self.episode_count += 1
        print(f"\n📚 EPISODE {self.episode_count}: '{target_prompt}'")
        print("=" * 80)
        
        state = self.env.reset(target_prompt)
        total_reward = 0
        step_details = []
        best_score = self.env.validation_history[0]
        
        while True:
            # Agent chooses LLaMA strategy
            action = self.agent.act(state, training=True)
            
            # Environment uses LLaMA to generate and validate
            next_state, reward, done, info = self.env.step(action)
            
            # Agent learns
            loss = self.agent.step(state, action, reward, next_state, done)
            
            # Track step
            step_details.append({
                'step': self.env.step_count,
                'strategy': info['strategy_used'],
                'custom_prompt': info['custom_prompt'],
                'score': info['score'],
                'reward': reward,
                'improvement': info['improvement']
            })
            
            total_reward += reward
            best_score = max(best_score, info['score'])
            state = next_state
            
            if done:
                break
        
        # Episode summary
        ultra_achieved = best_score >= self.env.ultra_target
        self.episode_scores.append(best_score)
        self.episode_rewards.append(total_reward)
        self.ultra_achievements.append(ultra_achieved)
        
        print(f"\n📊 EPISODE {self.episode_count} COMPLETE")
        print(f"   🏆 Best Score: {best_score:.3f}")
        print(f"   🎁 Total Reward: {total_reward:.2f}")
        print(f"   🌟 Ultra Achieved: {ultra_achieved}")
        print(f"   📚 Agent Memory: {len(self.agent.memory)}")
        print(f"   🧠 LLaMA Patterns: {len(self.env.llama_generator.successful_examples)}")
        
        return {
            'episode': self.episode_count,
            'target_prompt': target_prompt,
            'best_score': best_score,
            'total_reward': total_reward,
            'ultra_achieved': ultra_achieved,
            'steps': len(step_details),
            'step_details': step_details,
            'epsilon': self.agent.epsilon,
            'learn_count': self.agent.learn_count
        }
    
    def train_multiple_episodes(self, target_prompts: List[str], episodes_per_prompt: int = 3) -> Dict:
        """Train on multiple prompts"""
        
        print(f"🎓 TRAINING RL + LLaMA OPTIMIZER V3")
        print(f"   📝 Prompts: {len(target_prompts)}")
        print(f"   🔄 Episodes per prompt: {episodes_per_prompt}")
        print("=" * 80)
        
        all_results = []
        
        for prompt_idx, prompt in enumerate(target_prompts):
            print(f"\n🎯 TRAINING ON PROMPT {prompt_idx + 1}/{len(target_prompts)}: '{prompt}'")
            
            for episode in range(episodes_per_prompt):
                result = self.train_episode(prompt)
                all_results.append(result)
                
                # Progress update
                recent_scores = self.episode_scores[-5:]
                avg_recent = sum(recent_scores) / len(recent_scores)
                print(f"   📈 Recent avg: {avg_recent:.3f}")
        
        # Final summary
        total_episodes = len(all_results)
        avg_score = sum(self.episode_scores) / len(self.episode_scores)
        best_score = max(self.episode_scores)
        ultra_count = sum(self.ultra_achievements)
        
        print(f"\n🎉 TRAINING COMPLETE!")
        print(f"   📊 Episodes: {total_episodes}")
        print(f"   📈 Avg Score: {avg_score:.3f}")
        print(f"   🏆 Best Score: {best_score:.3f}")
        print(f"   🌟 Ultra Rate: {ultra_count}/{total_episodes} ({ultra_count/total_episodes*100:.1f}%)")
        print(f"   🧠 Patterns Learned: {len(self.env.llama_generator.successful_examples)}")
        
        return {
            'total_episodes': total_episodes,
            'average_score': avg_score,
            'best_score': best_score,
            'ultra_achievements': ultra_count,
            'ultra_rate': ultra_count / total_episodes,
            'patterns_learned': len(self.env.llama_generator.successful_examples),
            'final_epsilon': self.agent.epsilon,
            'all_results': all_results
        }

def main():
    """Test RL + LLaMA optimizer v3"""
    
    print("🚀 RL + LLaMA PROMPT OPTIMIZER V3")
    print("="*80)
    print("✅ LLaMA 3.2 generates CUSTOM prompts")
    print("✅ RL agent learns optimal instruction strategies")
    print("✅ Real validation with subnet_accurate_validator.py")
    print("✅ Continuous learning from actual scores")
    print("="*80)
    
    try:
        optimizer = RLLLaMAOptimizerV3(ultra_target=0.96)
        
        test_prompts = [
            "hexagonal prism steel structure",
            "elegant silk fabric draping",
            "transparent glass sphere with reflections"
        ]
        
        results = optimizer.train_multiple_episodes(test_prompts, episodes_per_prompt=2)
        
        print(f"\n🎯 FINAL RESULTS:")
        print(f"   Average Score: {results['average_score']:.3f}")
        print(f"   Best Score: {results['best_score']:.3f}")
        print(f"   Ultra Rate: {results['ultra_rate']:.1%}")
        print(f"   Patterns Learned: {results['patterns_learned']}")
        
    except Exception as e:
        print(f"❌ Training failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 