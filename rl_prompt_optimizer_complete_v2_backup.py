#!/usr/bin/env python3
"""
Complete RL Prompt Optimizer v2.0 - FULLY PATCHED
=================================================
Combines ALL features from v1 PLUS two major architectural upgrades:

PATCH 1: ACCELERATED LEARNING (Prioritized Experience Replay)
✅ Prioritized Experience Replay (PER) for 3-5x faster learning
✅ Focuses on "surprising" experiences with high TD-errors
✅ Importance sampling weights for unbiased learning

PATCH 2: DYNAMIC ACTION SPACE & CREATIVE LEARNING  
✅ Meta-learning phase that discovers new strategies
✅ LLM-powered Knowledge Engineer extracts patterns from successes
✅ Dynamic neural network resizing for new actions
✅ Self-improving action space that grows over time

RETAINED FEATURES:
✅ Fixed rewards, epsilon decay, proper learning
✅ Checkpointing, save/load, graceful interruption
✅ Production readiness monitoring
✅ Complete state persistence
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
# PATCH 1: PRIORITIZED EXPERIENCE REPLAY FOR FASTER LEARNING
# ==============================================================================
class PrioritizedReplayBuffer:
    """
    Prioritized Experience Replay (PER) Buffer
    Stores experiences with priorities based on TD-errors.
    High-error (surprising) experiences are sampled more frequently.
    """
    
    def __init__(self, capacity: int = 5000, alpha: float = 0.6):
        self.capacity = capacity
        self.alpha = alpha  # Controls prioritization strength
        self.buffer = []
        self.priorities = np.zeros(capacity, dtype=np.float32)
        self.position = 0
        self.max_priority = 1.0
        
        print(f"   📚 Prioritized Replay Buffer initialized (capacity: {capacity}, alpha: {alpha})")

    def push(self, experience: Experience):
        """Add experience with maximum priority to ensure sampling"""
        if len(self.buffer) < self.capacity:
            self.buffer.append(experience)
        else:
            self.buffer[self.position] = experience
        
        # New experiences get max priority
        self.priorities[self.position] = self.max_priority
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size: int, beta: float = 0.4) -> Tuple[List[Experience], np.ndarray, np.ndarray]:
        """Sample experiences based on priorities"""
        if len(self.buffer) == self.capacity:
            priorities = self.priorities
        else:
            priorities = self.priorities[:self.position]
        
        # Calculate sampling probabilities
        probs = priorities ** self.alpha
        probs /= probs.sum()

        # Sample indices
        indices = np.random.choice(len(self.buffer), batch_size, p=probs)
        experiences = [self.buffer[i] for i in indices]

        # Importance sampling weights for unbiased learning
        total = len(self.buffer)
        weights = (total * probs[indices]) ** (-beta)
        weights /= weights.max()
        
        return experiences, indices, np.array(weights, dtype=np.float32)

    def update_priorities(self, batch_indices: np.ndarray, batch_priorities: np.ndarray):
        """Update priorities based on TD-errors"""
        for idx, priority in zip(batch_indices, batch_priorities):
            self.priorities[idx] = priority
        self.max_priority = max(self.max_priority, np.max(batch_priorities))

    def __len__(self):
        return len(self.buffer)
        
    def save(self, filepath: Path):
        """Save PER buffer with priorities"""
        data = {
            'buffer': self.buffer,
            'priorities': self.priorities,
            'position': self.position,
            'max_priority': self.max_priority
        }
        with open(filepath, 'wb') as f:
            pickle.dump(data, f)
        print(f"   💾 PER buffer saved: {len(self.buffer)} experiences")

    def load(self, filepath: Path):
        """Load PER buffer with priorities"""
        if filepath.exists():
            with open(filepath, 'rb') as f:
                data = pickle.load(f)
                self.buffer = data['buffer']
                self.priorities = data['priorities']
                self.position = data['position']
                self.max_priority = data.get('max_priority', 1.0)
            print(f"   📂 PER buffer loaded: {len(self.buffer)} experiences")
            return True
        return False

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
    # PATCH 2: Meta-learning tracking
    new_actions_learned: int = 0
    action_space_size: int = 18

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

class CompletePromptEnvironmentV2:
    """Enhanced environment with dynamic action space"""
    
    def __init__(self, ultra_target: float = 0.96, checkpoint_dir: str = "rl_checkpoints_v2"):
        self.ultra_target = ultra_target
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(exist_ok=True)
        
        self.target_prompt = ""
        self.current_prompt = ""
        self.validation_history = []
        self.step_count = 0
        self.max_steps = 8
        
        # PATCH 2: Dynamic action space
        self.action_space = self._define_initial_action_space()
        self.state_size = 25  # Slightly expanded for meta-learning features
        
        # Meta-learning tracking
        self.meta_learning_events = []
        self.meta_learn_score_threshold = 0.8  # High score threshold for meta-learning
        
        # Session tracking
        self.session_id = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.episode_log = []
        
        print(f"🎮 COMPLETE RL ENVIRONMENT V2 INITIALIZED")
        print(f"   🎯 Target Score: {ultra_target}")
        print(f"   💾 Checkpoint Dir: {self.checkpoint_dir}")
        print(f"   🎬 Initial Action Space Size: {len(self.action_space)}")
        print(f"   ✨ Meta-learning: ENABLED")

    @property
    def action_size(self):
        """Dynamic property for action space size"""
        return len(self.action_space)

    def _define_initial_action_space(self) -> List[Tuple]:
        """Define initial action space (same as before)"""
        
        actions = []
        
        # Proven patterns
        proven_patterns = [
            "defense-grade ultra-precision {target}, ultra-high technical specification",
            "aerospace-grade precision-engineered {target}, advanced engineering design", 
            "military-spec ultra-detailed {target}, premium manufacturing excellence",
            "laboratory-grade precision-forged {target}, aerospace-engineering excellence",
            "ultra-precision masterpiece-quality {target}, ultra-high technical specification",
            "precision-aerospace {target}, defense-grade excellence",
            "ultra-military-spec {target}, precision-engineering design"
        ]
        
        for pattern in proven_patterns:
            actions.append(('APPLY_PATTERN', pattern, 'full_replace'))
        
        # Quality upgrades
        quality_upgrades = [
            ('UPGRADE_AUTHORITY', 'aerospace-grade', 'replace'),
            ('UPGRADE_AUTHORITY', 'defense-grade', 'replace'),
            ('UPGRADE_AUTHORITY', 'military-spec', 'replace'),
            ('UPGRADE_PROCESS', 'ultra-precision', 'replace'),
            ('UPGRADE_PROCESS', 'precision-engineered', 'replace'),
            ('UPGRADE_PROCESS', 'masterpiece-quality', 'replace'),
            ('UPGRADE_QUALITY', 'ultra-high technical specification', 'replace'),
            ('UPGRADE_QUALITY', 'advanced engineering design', 'replace')
        ]
        
        actions.extend(quality_upgrades)
        
        # Simplification actions
        simplify_actions = [
            ('SIMPLIFY', 'remove_duplicates', 'clean'),
            ('SIMPLIFY', 'keep_best_only', 'clean'),
            ('SIMPLIFY', 'ultra_minimal', 'clean')
        ]
        
        actions.extend(simplify_actions)
        
        return actions

    def add_new_action(self, new_action: Tuple, source_info: str = ""):
        """PATCH 2: Dynamically add new action to action space"""
        if new_action not in self.action_space:
            self.action_space.append(new_action)
            print(f"   ✨ NEW ACTION LEARNED! '{new_action[1][:50]}...' (Total: {self.action_size})")
            print(f"      Source: {source_info}")
            return True
        return False

    def reset(self, target_prompt: str) -> np.ndarray:
        """Reset environment with enhanced logging"""
        
        self.target_prompt = target_prompt
        self.current_prompt = f"wbgmsst, {target_prompt}, white background"
        self.validation_history = []
        self.step_count = 0
        
        initial_score = self._validate_prompt(self.current_prompt)
        self.validation_history.append(initial_score)
        
        # Enhanced episode logging
        episode_start = {
            'timestamp': time.time(),
            'target_prompt': target_prompt,
            'initial_prompt': self.current_prompt,
            'initial_score': initial_score,
            'session_id': self.session_id,
            'action_space_size': self.action_size
        }
        self.episode_log.append(episode_start)
        
        print(f"🔄 ENVIRONMENT RESET")
        print(f"   🎯 Target: {target_prompt}")
        print(f"   📊 Initial Score: {initial_score:.3f}")
        print(f"   🎬 Actions Available: {self.action_size}")
        
        return self._get_state()

    def step(self, action_idx: int) -> Tuple[np.ndarray, float, bool, Dict]:
        """Enhanced step function with meta-learning tracking"""
        
        self.step_count += 1
        
        # Handle dynamic action space
        if action_idx >= len(self.action_space):
            print(f"   ⚠️ Invalid action index {action_idx}, using random action")
            action_idx = random.randrange(len(self.action_space))
        
        action = self.action_space[action_idx]
        
        print(f"🎬 STEP {self.step_count}: {action[0]}")
        
        # Apply action
        old_prompt = self.current_prompt
        old_score = self.validation_history[-1]
        
        self.current_prompt = self._apply_smart_action(action)
        
        # Validate new prompt
        new_score = self._validate_prompt(self.current_prompt)
        self.validation_history.append(new_score)
        
        # FIXED: Proper reward calculation
        reward = self._calculate_proper_reward(old_score, new_score, old_prompt)
        
        # Check done
        done = (new_score >= self.ultra_target or self.step_count >= self.max_steps)
        
        print(f"   📝 Prompt: {self.current_prompt}")
        print(f"   📊 Score: {old_score:.3f} → {new_score:.3f}")
        print(f"   🎁 Reward: {reward:.3f}")
        
        # Enhanced step logging
        step_data = {
            'step': self.step_count,
            'action_idx': action_idx,
            'action': action,
            'old_prompt': old_prompt,
            'new_prompt': self.current_prompt,
            'old_score': old_score,
            'new_score': new_score,
            'reward': reward,
            'done': done,
            'ultra_achieved': new_score >= self.ultra_target
        }
        self.episode_log[-1].setdefault('steps', []).append(step_data)
        
        # Create info dictionary first
        info = {
            'score': new_score,
            'prompt': self.current_prompt,
            'action_taken': action[0],
            'step': self.step_count,
            'improvement': new_score - old_score,
            'ultra_achieved': new_score >= self.ultra_target,
            'trigger_immediate_meta_learning': False  # Default to False
        }
        
        # PATCH 2: Track both ultra achievements AND high scores for meta-learning
        if new_score >= self.ultra_target:
            self._record_success_for_meta_learning(self.target_prompt, self.current_prompt, new_score)
            # Always trigger immediate meta-learning for ultra achievements
            info['trigger_immediate_meta_learning'] = True
        else:
            # Also track high scores (0.8+) for meta-learning
            should_trigger = self._record_high_score_for_meta_learning(self.target_prompt, self.current_prompt, new_score)
            info['trigger_immediate_meta_learning'] = should_trigger
        
        return self._get_state(), reward, done, info

    def _record_success_for_meta_learning(self, original: str, successful: str, score: float):
        """PATCH 2: Record successful prompts for meta-learning"""
        event = MetaLearningEvent(
            episode=len(self.episode_log),
            original_prompt=original,
            successful_prompt=successful,
            extracted_pattern="",  # Will be filled by meta-learning
            score_achieved=score,
            timestamp=time.time()
        )
        self.meta_learning_events.append(event)
        print(f"   🌟 SUCCESS RECORDED for meta-learning (Score: {score:.3f})")
        
    def _should_trigger_immediate_meta_learning(self) -> bool:
        """PATCH 2: Check if we should trigger meta-learning immediately (less aggressive)"""
        
        # Only trigger once per episode to avoid spam
        current_episode = len(self.episode_log)
        if hasattr(self, '_last_meta_trigger_episode') and self._last_meta_trigger_episode == current_episode:
            return False
        
        # Trigger if we achieved a significant new personal best (0.05+ improvement)
        if len(self.meta_learning_events) > 0:
            latest_event = self.meta_learning_events[-1]
            if latest_event.episode == current_episode:  # Just happened this episode
                previous_best = max([e.score_achieved for e in self.meta_learning_events[:-1]], default=0.0)
                if latest_event.score_achieved > previous_best + 0.05:  # Significant improvement
                    print(f"   🚀 IMMEDIATE META-LEARNING TRIGGER: New personal best {latest_event.score_achieved:.3f}!")
                    self._last_meta_trigger_episode = current_episode
                    return True
        
        return False

    def _record_high_score_for_meta_learning(self, original: str, successful: str, score: float):
        """PATCH 2: Also record high scores (0.8+) for meta-learning"""
        if score >= self.meta_learn_score_threshold:
            self._record_success_for_meta_learning(original, successful, score)
            print(f"   📈 HIGH SCORE RECORDED for meta-learning (Score: {score:.3f})")
            
            # Check for immediate meta-learning trigger
            if self._should_trigger_immediate_meta_learning():
                return True  # Signal to trigger meta-learning immediately
        return False

    def _apply_smart_action(self, action: Tuple) -> str:
        """Smart action application (same as before)"""
        
        action_type, modifier, mode = action
        
        if action_type == 'APPLY_PATTERN':
            pattern = modifier.replace('{target}', self.target_prompt)
            return f"wbgmsst, {pattern}, white background"
        
        elif action_type in ['UPGRADE_AUTHORITY', 'UPGRADE_PROCESS', 'UPGRADE_QUALITY']:
            parts = self.current_prompt.split(', ')
            if len(parts) >= 3:
                middle = parts[1]
                
                # Remove conflicting descriptors
                if action_type == 'UPGRADE_AUTHORITY':
                    middle = re.sub(r'\b(aerospace-grade|military-spec|defense-grade|aviation-standard|laboratory-grade)\b\s*', '', middle)
                elif action_type == 'UPGRADE_PROCESS':
                    middle = re.sub(r'\b(ultra-precision|precision-engineered|masterpiece-quality|ultra-detailed)\b\s*', '', middle)
                
                # Add new descriptor
                middle = f"{modifier} {middle}".strip()
                parts[1] = middle
                
                return ', '.join(parts)
        
        elif action_type == 'SIMPLIFY':
            if modifier == 'remove_duplicates':
                words = self.current_prompt.split()
                seen = set()
                cleaned_words = []
                for word in words:
                    if word not in seen:
                        cleaned_words.append(word)
                        seen.add(word)
                return ' '.join(cleaned_words)
            
            elif modifier == 'keep_best_only':
                return f"wbgmsst, defense-grade ultra-precision {self.target_prompt}, ultra-high technical specification, white background"
            
            elif modifier == 'ultra_minimal':
                return f"wbgmsst, aerospace-grade {self.target_prompt}, precision-engineered excellence, white background"
        
        return self.current_prompt

    def _validate_prompt(self, prompt: str) -> float:
        """Validate prompt with error handling"""
        try:
            cmd = [
                "bash", "-c", 
                f"source /home/mbhat/miniconda/bin/activate && conda activate trellis_new && python subnet_accurate_validator.py '{prompt}'"
            ]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
            
            if result.returncode != 0:
                return 0.0
            
            with open("subnet_validation_results.json", 'r') as f:
                data = json.load(f)
                return data.get("validation_engine_score", 0.0)
        
        except Exception as e:
            print(f"   ❌ Validation error: {e}")
            return 0.0

    def _calculate_proper_reward(self, old_score: float, new_score: float, old_prompt: str) -> float:
        """FIXED: Proper reward function"""
        
        # Base reward: Direct score improvement
        score_improvement = new_score - old_score
        base_reward = score_improvement * 100
        
        # Absolute score bonuses
        if new_score >= self.ultra_target:
            base_reward += 200  # Ultra achievement
        elif new_score >= 0.9:
            base_reward += 100  # Excellent
        elif new_score >= 0.8:
            base_reward += 50   # Good
        elif new_score >= 0.7:
            base_reward += 20   # Decent
        
        # Prompt length penalty
        if len(self.current_prompt) > 120:
            base_reward -= 10
        elif len(self.current_prompt) > 100:
            base_reward -= 5
        
        # Penalty for very low scores
        if new_score < 0.3:
            base_reward -= 20
        
        # Bonus for reaching new personal best
        if new_score > max(self.validation_history[:-1]) if len(self.validation_history) > 1 else 0:
            base_reward += 30
        
        return base_reward

    def _get_state(self) -> np.ndarray:
        """Enhanced state representation"""
        
        state = np.zeros(self.state_size)
        
        # Last 3 scores
        history = self.validation_history[-3:]
        for i, score in enumerate(history):
            if i < 3:
                state[i] = score
        
        # Current step progress
        state[3] = self.step_count / self.max_steps
        
        # Best score so far
        state[4] = max(self.validation_history) if self.validation_history else 0.0
        
        # Recent improvement trend
        if len(self.validation_history) >= 2:
            state[5] = self.validation_history[-1] - self.validation_history[-2]
        
        # Prompt characteristics
        prompt_lower = self.current_prompt.lower()
        state[6] = 1.0 if 'aerospace' in prompt_lower else 0.0
        state[7] = 1.0 if 'defense' in prompt_lower else 0.0
        state[8] = 1.0 if 'military' in prompt_lower else 0.0
        state[9] = 1.0 if 'ultra-precision' in prompt_lower else 0.0
        state[10] = 1.0 if 'precision-engineered' in prompt_lower else 0.0
        state[11] = 1.0 if 'masterpiece' in prompt_lower else 0.0
        state[12] = len(self.current_prompt) / 150
        
        # Target object characteristics
        target_lower = self.target_prompt.lower()
        state[13] = 1.0 if any(word in target_lower for word in ["steel", "metal"]) else 0.0
        state[14] = 1.0 if any(word in target_lower for word in ["fabric", "cloth"]) else 0.0
        state[15] = 1.0 if any(word in target_lower for word in ["glass", "crystal"]) else 0.0
        
        # Performance indicators
        state[16] = 1.0 if self.validation_history[-1] >= 0.8 else 0.0
        state[17] = 1.0 if self.validation_history[-1] >= self.ultra_target else 0.0
        
        # PATCH 2: Meta-learning features
        state[18] = len(self.meta_learning_events) / 10  # Normalized success count
        state[19] = self.action_size / 50  # Normalized action space size
        
        # Recent ultra achievement rate
        if len(self.episode_log) >= 5:
            recent_ultras = sum(1 for ep in self.episode_log[-5:] 
                              if any(step.get('ultra_achieved', False) for step in ep.get('steps', [])))
            state[20] = recent_ultras / 5
        
        # Action space diversity (ratio of new to original actions)
        original_actions = 18  # Initial action space size
        state[21] = min((self.action_size - original_actions) / 10, 1.0)
        
        return state

    def save_episode_log(self, checkpoint_path: Path):
        """Save detailed episode log with meta-learning events"""
        log_file = checkpoint_path / f"episode_log_{self.session_id}.json"
        
        # Include meta-learning events in the log
        full_log = {
            'episodes': self.episode_log,
            'meta_learning_events': [asdict(event) for event in self.meta_learning_events],
            'final_action_space_size': self.action_size,
            'session_id': self.session_id
        }
        
        with open(log_file, 'w') as f:
            json.dump(full_log, f, indent=2)

class CompleteDQNV2(nn.Module):
    """Enhanced DQN with dynamic resizing capabilities"""
    
    def __init__(self, state_size: int, action_size: int):
        super(CompleteDQNV2, self).__init__()
        
        self.fc1 = nn.Linear(state_size, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 64)
        self.output_layer = nn.Linear(64, action_size)  # This layer will be resized
        
        self.dropout = nn.Dropout(0.1)
        self.layer_norm1 = nn.LayerNorm(128)
        self.layer_norm2 = nn.LayerNorm(128)

    def forward(self, x):
        x = F.relu(self.layer_norm1(self.fc1(x)))
        x = self.dropout(x)
        x = F.relu(self.layer_norm2(self.fc2(x)))
        x = self.dropout(x)
        x = F.relu(self.fc3(x))
        x = self.output_layer(x)
        return x

class CompleteDQNAgentV2:
    """PATCHED: DQN Agent with PER and dynamic resizing"""
    
    def __init__(self, state_size: int, action_size: int, checkpoint_dir: str, lr: float = 0.001):
        self.state_size = state_size
        self.action_size = action_size
        self.checkpoint_dir = Path(checkpoint_dir)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Networks
        self.q_network_local = CompleteDQNV2(state_size, action_size).to(self.device)
        self.q_network_target = CompleteDQNV2(state_size, action_size).to(self.device)
        self.optimizer = optim.Adam(self.q_network_local.parameters(), lr=lr)
        
        # PATCH 1: Prioritized Experience Replay
        self.memory = PrioritizedReplayBuffer(capacity=5000)
        self.beta = 0.4  # Importance sampling parameter
        self.beta_increment_per_sampling = 0.001
        
        # Optimized hyperparameters
        self.batch_size = 32
        self.gamma = 0.95
        self.tau = 0.005
        self.update_every = 2  # Learn more frequently
        
        # FIXED: Proper epsilon schedule
        self.epsilon = 0.9
        self.epsilon_min = 0.05
        self.epsilon_decay = 0.98
        
        # Learning tracking
        self.step_count = 0
        self.learn_count = 0
        self.losses = []
        self.q_values_history = []
        
        print(f"🤖 COMPLETE DQN AGENT V2 INITIALIZED")
        print(f"   🧠 Device: {self.device}")
        print(f"   📚 Prioritized Experience Replay: ENABLED")
        print(f"   🎲 Initial Epsilon: {self.epsilon}")
        print(f"   ✨ Dynamic Action Space: SUPPORTED")

    def act(self, state: np.ndarray, training: bool = True) -> int:
        """Enhanced action selection with Q-value tracking"""
        
        if training and random.random() < self.epsilon:
            action = random.randrange(self.action_size)
            print(f"   🎲 EXPLORATION: Random action {action} (ε={self.epsilon:.3f})")
            return action
        else:
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            
            self.q_network_local.eval()
            with torch.no_grad():
                q_values = self.q_network_local(state_tensor)
            self.q_network_local.train()
            
            action = q_values.argmax().item()
            max_q = q_values.max().item()
            self.q_values_history.append(max_q)
            
            print(f"   🧠 EXPLOITATION: Action {action} (Q={max_q:.2f}, ε={self.epsilon:.3f})")
            return action

    def step(self, state: np.ndarray, action: int, reward: float, 
             next_state: np.ndarray, done: bool):
        """PATCH 1: Enhanced step with PER"""
        
        # Store experience
        experience = Experience(state, action, reward, next_state, done)
        self.memory.push(experience)
        
        self.step_count += 1
        
        # Learn more frequently with PER
        if self.step_count % self.update_every == 0 and len(self.memory) >= self.batch_size:
            # Gradually increase beta (importance sampling correction)
            self.beta = np.min([1., self.beta + self.beta_increment_per_sampling])
            
            # Sample from PER buffer
            experiences, indices, weights = self.memory.sample(self.batch_size, self.beta)
            loss = self.learn(experiences, indices, weights)
            
            # FIXED: Decay epsilon after learning
            if self.epsilon > self.epsilon_min:
                self.epsilon *= self.epsilon_decay
                
            return loss
        
        return None

    def learn(self, experiences: List[Experience], indices: np.ndarray, weights: np.ndarray) -> float:
        """PATCH 1: Enhanced learning with PER weights and priority updates"""
        
        self.learn_count += 1
        
        # Convert to tensors
        states = torch.FloatTensor([e.state for e in experiences]).to(self.device)
        actions = torch.LongTensor([e.action for e in experiences]).unsqueeze(1).to(self.device)
        rewards = torch.FloatTensor([e.reward for e in experiences]).to(self.device)
        next_states = torch.FloatTensor([e.next_state for e in experiences]).to(self.device)
        dones = torch.BoolTensor([e.done for e in experiences]).to(self.device)
        weights = torch.FloatTensor(weights).unsqueeze(1).to(self.device)

        # Get Q-values
        current_q_values = self.q_network_local(states).gather(1, actions)
        next_q_values = self.q_network_target(next_states).detach().max(1)[0]
        target_q_values = rewards + (self.gamma * next_q_values * ~dones)

        # Compute TD errors for priority updates
        td_errors = torch.abs(target_q_values.unsqueeze(1) - current_q_values).detach().cpu().numpy()
        
        # Update priorities in PER buffer
        self.memory.update_priorities(indices, td_errors.squeeze() + 1e-5)

        # Compute weighted loss (importance sampling)
        loss = (weights * F.mse_loss(current_q_values, target_q_values.unsqueeze(1), reduction='none')).mean()
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_network_local.parameters(), 1.0)
        self.optimizer.step()
        
        # Soft update target network
        self.soft_update()
        
        # Track loss
        self.losses.append(loss.item())
        
        print(f"   📚 PER LEARNING #{self.learn_count}: Loss {loss.item():.4f}, Buffer {len(self.memory)}, β={self.beta:.3f}, ε={self.epsilon:.3f}")
        return loss.item()

    def resize_action_space(self, new_action_size: int):
        """PATCH 2: Dynamically resize neural network for new actions"""
        
        if new_action_size <= self.action_size:
            return  # No need to resize if not growing
        
        print(f"   🧠 Resizing neural networks: {self.action_size} → {new_action_size} actions")
        
        old_action_size = self.action_size
        self.action_size = new_action_size
        
        # Resize both networks
        for network in [self.q_network_local, self.q_network_target]:
            old_layer = network.output_layer
            new_layer = nn.Linear(old_layer.in_features, new_action_size).to(self.device)
            
            # Copy existing weights
            new_layer.weight.data[:old_action_size, :] = old_layer.weight.data
            new_layer.bias.data[:old_action_size] = old_layer.bias.data
            
            # Initialize new action weights
            with torch.no_grad():
                nn.init.xavier_uniform_(new_layer.weight.data[old_action_size:, :])
                new_layer.bias.data[old_action_size:] = 0
            
            network.output_layer = new_layer

        # Re-create optimizer with new parameters
        self.optimizer = optim.Adam(self.q_network_local.parameters(), lr=self.optimizer.param_groups[0]['lr'])
        
        print(f"   ✅ Networks resized, optimizer recreated")

    def soft_update(self):
        """Soft update target network"""
        for target_param, local_param in zip(self.q_network_target.parameters(),
                                           self.q_network_local.parameters()):
            target_param.data.copy_(self.tau * local_param.data + 
                                  (1.0 - self.tau) * target_param.data)

    def save_checkpoint(self, checkpoint_path: Path, metadata: Dict):
        """Enhanced checkpoint saving with PER and dynamic action size"""
        
        checkpoint = {
            'q_network_local_state_dict': self.q_network_local.state_dict(),
            'q_network_target_state_dict': self.q_network_target.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'step_count': self.step_count,
            'learn_count': self.learn_count,
            'losses': self.losses,
            'q_values_history': self.q_values_history,
            'action_size': self.action_size,  # Save current action size
            'beta': self.beta,  # Save PER parameter
            'metadata': metadata
        }
        
        model_file = checkpoint_path / 'agent_checkpoint.pth'
        torch.save(checkpoint, model_file)
        
        # Save PER buffer
        buffer_file = checkpoint_path / 'per_buffer.pkl'
        self.memory.save(buffer_file)
        
        print(f"   💾 Enhanced agent checkpoint saved (action_size: {self.action_size})")

    def load_checkpoint(self, checkpoint_path: Path) -> Optional[Dict]:
        """Enhanced checkpoint loading with dynamic resizing"""
        
        model_file = checkpoint_path / 'agent_checkpoint.pth'
        buffer_file = checkpoint_path / 'per_buffer.pkl'
        
        if model_file.exists():
            checkpoint = torch.load(model_file, map_location=self.device, weights_only=False)
            
            # Handle dynamic action size
            loaded_action_size = checkpoint.get('action_size', self.action_size)
            if loaded_action_size != self.action_size:
                self.resize_action_space(loaded_action_size)
            
            # Load state dicts
            self.q_network_local.load_state_dict(checkpoint['q_network_local_state_dict'])
            self.q_network_target.load_state_dict(checkpoint['q_network_target_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            
            # Load training state
            self.epsilon = checkpoint['epsilon']
            self.step_count = checkpoint['step_count']
            self.learn_count = checkpoint.get('learn_count', 0)
            self.losses = checkpoint['losses']
            self.q_values_history = checkpoint.get('q_values_history', [])
            self.beta = checkpoint.get('beta', 0.4)
            
            # Load PER buffer
            self.memory.load(buffer_file)
            
            print(f"   📂 Enhanced agent checkpoint loaded")
            print(f"      🎬 Action size: {self.action_size}")
            print(f"      🎲 Epsilon: {self.epsilon:.3f}")
            print(f"      📚 Learn count: {self.learn_count}")
            print(f"      📊 Buffer size: {len(self.memory)}")
            print(f"      🎯 Beta (PER): {self.beta:.3f}")
            
            return checkpoint['metadata']
        
        return None

class ProductionReadinessMonitor:
    """Production readiness monitoring (same as before)"""
    
    def __init__(self):
        self.metrics_history = []
        self.success_criteria = {
            "ultra_achievement_rate": 0.3,
            "avg_score_threshold": 0.75,
            "improvement_rate": 0.7,
            "epsilon_stability": 0.05,
            "min_episodes": 50
        }
        
    def add_episode_metrics(self, metrics: TrainingMetrics):
        self.metrics_history.append(metrics)
        
    def assess_production_readiness(self) -> Dict:
        if len(self.metrics_history) < self.success_criteria["min_episodes"]:
            return {"ready": False, "reason": "Insufficient episodes"}
        
        recent_window = self.metrics_history[-30:]
        
        # Calculate metrics
        recent_scores = [m.score for m in recent_window]
        recent_ultras = [m.ultra_achieved for m in recent_window]
        recent_epsilons = [m.epsilon for m in recent_window[-10:]]
        
        ultra_rate = sum(recent_ultras) / len(recent_ultras)
        avg_score = statistics.mean(recent_scores)
        epsilon_stable = statistics.variance(recent_epsilons) < 0.001 if len(recent_epsilons) > 1 else False
        
        # Check criteria
        checks = {
            "ultra_rate": ultra_rate >= self.success_criteria["ultra_achievement_rate"],
            "avg_score": avg_score >= self.success_criteria["avg_score_threshold"],
            "epsilon_stable": epsilon_stable,
            "min_episodes": len(self.metrics_history) >= self.success_criteria["min_episodes"]
        }
        
        ready = sum(checks.values()) >= len(checks) * 0.8
        
        return {
            "ready": ready,
            "metrics": {
                "ultra_rate": ultra_rate,
                "avg_score": avg_score,
                "epsilon_stable": epsilon_stable,
                "total_episodes": len(self.metrics_history)
            },
            "checks": checks
        }

# ==============================================================================
# PATCH 2: META-LEARNING & DYNAMIC ACTION DISCOVERY
# ==============================================================================
class KnowledgeEngineer:
    """LLM-powered knowledge engineer for extracting patterns from successes"""
    
    def __init__(self, ollama_url: str = "http://localhost:11434", model: str = "llama3.2:3b"):
        self.ollama_url = ollama_url
        self.model = model
        
        print(f"🧠 Knowledge Engineer initialized (Model: {model})")

    def distill_pattern_from_success(self, original_prompt: str, successful_prompt: str, score: float) -> Optional[str]:
        """Extract reusable pattern from successful prompt"""
        
        system_prompt = """You are a Knowledge Engineer AI specialized in prompt optimization patterns.

Your task is to analyze successful prompt optimizations and extract reusable patterns.

CRITICAL REQUIREMENTS:
1. Extract the CORE PATTERN that made the prompt successful
2. Replace the specific object with {target} placeholder
3. Keep the essential descriptors and structure
4. Make it generic enough to work with different objects
5. Preserve the key elements that led to the high score

If no clear pattern can be extracted, respond with "NO_PATTERN"."""

        user_prompt = f"""Analyze this successful optimization:

ORIGINAL TARGET: "{original_prompt}"
SUCCESSFUL PROMPT: "{successful_prompt}"
VALIDATION SCORE: {score:.3f} (This is a high score!)

Extract a reusable pattern template that captures the essence of what made this prompt successful.

Examples:
- If successful prompt was "wbgmsst, ultra-realistic metallic steel beam, precision-manufactured finish, white background"
- Good template: "ultra-realistic metallic {{target}}, precision-manufactured finish"

- If successful prompt was "wbgmsst, aerospace-grade precision-engineered hexagonal prism, ultra-high technical specification, white background"  
- Good template: "aerospace-grade precision-engineered {{target}}, ultra-high technical specification"

Respond with ONLY the template pattern. Use {{target}} as the placeholder."""

        try:
            print(f"   🔬 Analyzing pattern: {original_prompt} → {successful_prompt}")
            
            data = {
                "model": self.model,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                "stream": False,
                "options": {
                    "temperature": 0.3,  # Low temperature for consistent patterns
                    "num_predict": 200
                }
            }
            
            response = requests.post(f"{self.ollama_url}/api/chat", json=data, timeout=30)
            response.raise_for_status()
            
            pattern = response.json()["message"]["content"].strip()
            pattern = pattern.strip('"\'')
            
            # Validate pattern
            if "NO_PATTERN" in pattern.upper() or "{target}" not in pattern:
                print(f"   ❌ No valid pattern extracted")
                return None
            
            # Clean up pattern
            pattern = pattern.replace("{{target}}", "{target}")
            
            print(f"   ✅ Pattern extracted: {pattern}")
            return pattern
            
        except Exception as e:
            print(f"   ❌ Knowledge Engineer failed: {e}")
            return None

class CompleteRLOptimizerV2:
    """COMPLETE: Fixed + Persistent + PER + Meta-Learning RL optimizer"""
    
    def __init__(self, ultra_target: float = 0.96, checkpoint_dir: str = "rl_checkpoints_v2"):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(exist_ok=True)
        
        self.env = CompletePromptEnvironmentV2(ultra_target, checkpoint_dir)
        self.agent = CompleteDQNAgentV2(
            state_size=self.env.state_size,
            action_size=self.env.action_size,
            checkpoint_dir=checkpoint_dir
        )
        
        # PATCH 2: Meta-learning components
        self.knowledge_engineer = KnowledgeEngineer()
        self.meta_learn_every_n_episodes = 4   # Check for new patterns every 4 episodes (much more frequent!)
        self.successful_episodes_threshold = 1  # Need only 1 success to start learning
        self.meta_learn_score_threshold = 0.8   # Also trigger on high scores (not just ultra)
        
        # Training state with meta-learning tracking
        self.training_state = TrainingCheckpoint(
            episode=0,
            total_episodes_completed=0,
            current_prompt_index=0,
            training_prompts=[],
            episodes_per_prompt=0,
            episode_rewards=[],
            episode_scores=[],
            ultra_achievements=[],
            epsilon=self.agent.epsilon,
            step_count=0,
            learn_count=0,
            best_overall_score=0.0,
            training_start_time=time.time(),
            last_checkpoint_time=time.time(),
            new_actions_learned=0,
            action_space_size=self.env.action_size
        )
        
        # Production readiness monitor
        self.monitor = ProductionReadinessMonitor()
        
        # Graceful shutdown handling
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        
        print(f"🚀 COMPLETE RL OPTIMIZER V2 INITIALIZED")
        print(f"   ✅ PATCH 1: Prioritized Experience Replay")
        print(f"   ✅ PATCH 2: Dynamic Action Discovery with LLM")
        print(f"   ✅ Fixed: Proper rewards, epsilon decay, batch learning")
        print(f"   ✅ Persistent: Checkpointing, save/load, resume")
        print(f"   ✅ Monitoring: Production readiness assessment")

    def _signal_handler(self, signum, frame):
        """Handle interruption gracefully"""
        print(f"\n⚠️ INTERRUPTION DETECTED (Signal {signum})")
        print("💾 Saving emergency checkpoint...")
        self._save_checkpoint("emergency_checkpoint")
        print("✅ Emergency checkpoint saved!")
        sys.exit(0)

    # ==============================================================================
    # PATCH 2: META-LEARNING PHASE
    # ==============================================================================
    def _meta_learning_phase(self):
        """PATCH 2: Periodically discover new action patterns from successes"""
        
        print(f"\n{'='*25} META-LEARNING PHASE {'='*25}")
        print(f"🔬 Analyzing recent successes for new action patterns...")
        
        # Get recent successful events
        recent_successes = [event for event in self.env.meta_learning_events 
                          if event.episode >= max(0, len(self.env.episode_log) - self.meta_learn_every_n_episodes)]
        
        if len(recent_successes) < self.successful_episodes_threshold:
            print(f"   📊 Only {len(recent_successes)} recent successes, need {self.successful_episodes_threshold}+ for meta-learning")
            print(f"{'='*72}\n")
            return
        
        print(f"   📊 Found {len(recent_successes)} recent ultra achievements")
        
        # Try to extract pattern from the best recent success
        best_success = max(recent_successes, key=lambda x: x.score_achieved)
        
        print(f"   🎯 Analyzing best success (Score: {best_success.score_achieved:.3f})")
        print(f"      Original: {best_success.original_prompt}")
        print(f"      Successful: {best_success.successful_prompt}")
        
        # Use Knowledge Engineer to extract pattern
        pattern = self.knowledge_engineer.distill_pattern_from_success(
            best_success.original_prompt,
            best_success.successful_prompt,
            best_success.score_achieved
        )
        
        if pattern:
            # Create new action
            new_action = ('APPLY_PATTERN', pattern, 'full_replace')
            
            # Check if truly new
            existing_patterns = [action[1] for action in self.env.action_space if action[0] == 'APPLY_PATTERN']
            
            if pattern not in existing_patterns:
                # Add to environment
                source_info = f"Meta-learned from episode {best_success.episode} (score: {best_success.score_achieved:.3f})"
                if self.env.add_new_action(new_action, source_info):
                    
                    # Resize agent's neural network
                    self.agent.resize_action_space(self.env.action_size)
                    
                    # Update tracking
                    self.training_state.new_actions_learned += 1
                    self.training_state.action_space_size = self.env.action_size
                    
                    # Mark this success as processed
                    best_success.extracted_pattern = pattern
                    
                    print(f"   🎉 NEW ACTION LEARNED AND INTEGRATED!")
                    print(f"      📈 Total actions: {self.env.action_size}")
                    print(f"      🧠 Agent network resized")
                    print(f"      🎊 Total learned: {self.training_state.new_actions_learned}")
                else:
                    print(f"   ⚠️ Action already exists in action space")
            else:
                print(f"   ⚠️ Pattern similar to existing action, skipping")
        else:
            print(f"   ❌ Could not extract reusable pattern from this success")
        
        print(f"{'='*72}\n")

    def train_with_checkpoints(self, target_prompts: List[str], 
                              episodes_per_prompt: int = 5,
                              resume_from: Optional[str] = None) -> Dict:
        """Enhanced training with meta-learning integration"""
        
        print(f"🎓 COMPLETE RL TRAINING SESSION V2")
        print(f"📝 Prompts: {len(target_prompts)}")
        print(f"🔄 Episodes per prompt: {episodes_per_prompt}")
        print(f"🧠 Meta-learning: Every {self.meta_learn_every_n_episodes} episodes + immediate triggers")
        print(f"📈 High score threshold: {self.meta_learn_score_threshold}")
        print(f"🎯 Ultra target: {self.env.ultra_target}")
        print("=" * 80)
        
        # Load previous state if resuming
        if resume_from:
            if self._load_checkpoint(resume_from):
                print(f"📂 RESUMED FROM CHECKPOINT: {resume_from}")
            else:
                print(f"❌ Could not load checkpoint: {resume_from}")
        
        # Initialize training state
        if not self.training_state.training_prompts:
            self.training_state.training_prompts = target_prompts
            self.training_state.episodes_per_prompt = episodes_per_prompt
        
        total_prompts = len(self.training_state.training_prompts)
        
        # Training loop with meta-learning
        for prompt_idx in range(self.training_state.current_prompt_index, total_prompts):
            current_prompt = self.training_state.training_prompts[prompt_idx]
            
            print(f"\n🎯 TRAINING ON PROMPT {prompt_idx + 1}/{total_prompts}: '{current_prompt}'")
            
            # Episodes for this prompt
            episodes_completed = 0
            if prompt_idx == self.training_state.current_prompt_index:
                episodes_completed = self.training_state.episode - (prompt_idx * episodes_per_prompt)
            
            for episode_in_prompt in range(episodes_completed, episodes_per_prompt):
                episode_num = prompt_idx * episodes_per_prompt + episode_in_prompt + 1
                
                # PATCH 2: Enhanced meta-learning triggers (scheduled first)
                should_trigger_scheduled = episode_num > 0 and episode_num % self.meta_learn_every_n_episodes == 0
                
                if should_trigger_scheduled:
                    print(f"\n🕒 SCHEDULED META-LEARNING (Episode {episode_num})")
                    self._meta_learning_phase()
                
                print(f"\n📚 EPISODE {episode_num} (Prompt {prompt_idx + 1}, Episode {episode_in_prompt + 1}/{episodes_per_prompt})")
                print(f"   🎬 Actions Available: {self.env.action_size}")
                print(f"   🧠 Learned Actions: {self.training_state.new_actions_learned}")
                
                # Train episode
                result = self._train_single_episode(current_prompt, episode_num)
                
                # PATCH 2: Check for immediate meta-learning trigger after episode
                should_trigger_immediate = result.get('immediate_meta_learning', False)
                if should_trigger_immediate:
                    print(f"\n⚡ IMMEDIATE META-LEARNING TRIGGERED!")
                    self._meta_learning_phase()
                
                # Update training state
                self.training_state.episode = episode_num
                self.training_state.total_episodes_completed += 1
                self.training_state.episode_rewards.append(result['total_reward'])
                self.training_state.episode_scores.append(result['best_score'])
                self.training_state.ultra_achievements.append(result['ultra_achieved'])
                self.training_state.epsilon = self.agent.epsilon
                self.training_state.learn_count = self.agent.learn_count
                self.training_state.action_space_size = self.env.action_size
                self.training_state.best_overall_score = max(
                    self.training_state.best_overall_score, 
                    result['best_score']
                )
                
                # Add to production readiness monitor
                metrics = TrainingMetrics(
                    episode=episode_num,
                    score=result['best_score'],
                    reward=result['total_reward'],
                    epsilon=result['epsilon'],
                    loss=result.get('avg_loss', 0.0),
                    ultra_achieved=result['ultra_achieved'],
                    improvement=result.get('improvement', 0.0),
                    prompt_length=len(result.get('final_prompt', '')),
                    action_type=result.get('final_action', 'unknown'),
                    exploration_action=result['epsilon'] > 0.2,
                    learn_count=result['learn_count']
                )
                self.monitor.add_episode_metrics(metrics)
                
                # Auto-checkpoint
                checkpoint_name = f"episode_{episode_num:03d}"
                self._save_checkpoint(checkpoint_name)
                
                # Check production readiness
                if episode_num % 10 == 0:
                    readiness = self.monitor.assess_production_readiness()
                    print(f"\n📊 PRODUCTION READINESS CHECK:")
                    print(f"   Status: {'✅ READY' if readiness['ready'] else '❌ NOT READY'}")
                    if 'metrics' in readiness:
                        print(f"   Ultra Rate: {readiness['metrics']['ultra_rate']:.1%}")
                        print(f"   Avg Score: {readiness['metrics']['avg_score']:.3f}")
                    else:
                        print(f"   Reason: {readiness.get('reason', 'Unknown')}")
                    
                    if readiness['ready']:
                        print(f"\n🎉 MODEL IS READY FOR PRODUCTION!")
                        self._save_checkpoint("production_ready")
                        return self._generate_final_report()
                
                # Enhanced episode summary
                print(f"\n⏸️ EPISODE {episode_num} COMPLETE")
                print(f"   📊 Score: {result['best_score']:.3f}")
                print(f"   🎁 Reward: {result['total_reward']:.2f}")
                print(f"   🎉 Ultra: {result['ultra_achieved']}")
                print(f"   🎲 Epsilon: {result['epsilon']:.3f}")
                print(f"   📚 Learn Count: {result['learn_count']}")
                print(f"   🎬 Action Space: {self.env.action_size}")
                print(f"   🧠 New Actions: {self.training_state.new_actions_learned}")
                
                # user_input = input("\n➡️ Press ENTER to continue, 'q' to quit, 's' to save and quit: ").strip().lower()
                user_input = "e"    
                if user_input == 'q':
                    print("🛑 Training stopped by user")
                    return self._generate_final_report()
                elif user_input == 's':
                    print("💾 Saving and exiting...")
                    self._save_checkpoint("user_save")
                    return self._generate_final_report()
            
            # Update prompt index
            self.training_state.current_prompt_index = prompt_idx + 1
        
        print(f"\n🎉 ALL TRAINING COMPLETE!")
        return self._generate_final_report()

    def _train_single_episode(self, target_prompt: str, episode_num: int) -> Dict:
        """Train one episode with enhanced tracking"""
        
        state = self.env.reset(target_prompt)
        total_reward = 0
        step = 0
        best_score = self.env.validation_history[0]
        losses = []
        immediate_meta_learning = False
        
        while True:
            action = self.agent.act(state, training=True)
            next_state, reward, done, info = self.env.step(action)
            
            # Check for immediate meta-learning trigger
            if info.get('trigger_immediate_meta_learning', False):
                immediate_meta_learning = True
            
            loss = self.agent.step(state, action, reward, next_state, done)
            if loss is not None:
                losses.append(loss)
            
            state = next_state
            total_reward += reward
            step += 1
            best_score = max(best_score, info['score'])
            
            if done:
                break
        
        ultra_achieved = best_score >= self.env.ultra_target
        
        return {
            'episode': episode_num,
            'target_prompt': target_prompt,
            'best_score': best_score,
            'total_reward': total_reward,
            'steps': step,
            'ultra_achieved': ultra_achieved,
            'epsilon': self.agent.epsilon,
            'learn_count': self.agent.learn_count,
            'avg_loss': statistics.mean(losses) if losses else 0.0,
            'final_prompt': info.get('prompt', ''),
            'improvement': best_score - self.env.validation_history[0],
            'final_action': info.get('action_taken', ''),
            'immediate_meta_learning': immediate_meta_learning  # Pass the trigger flag
        }

    def _save_checkpoint(self, checkpoint_name: str):
        """Enhanced checkpoint saving with meta-learning state"""
        
        checkpoint_path = self.checkpoint_dir / checkpoint_name
        checkpoint_path.mkdir(exist_ok=True)
        
        # Enhanced metadata with meta-learning info
        metadata = {
            'episode': self.training_state.episode,
            'total_episodes_completed': self.training_state.total_episodes_completed,
            'current_prompt_index': self.training_state.current_prompt_index,
            'training_prompts': self.training_state.training_prompts,
            'episodes_per_prompt': self.training_state.episodes_per_prompt,
            'best_overall_score': self.training_state.best_overall_score,
            'training_start_time': self.training_state.training_start_time,
            'checkpoint_time': time.time(),
            'new_actions_learned': self.training_state.new_actions_learned,
            'action_space_size': self.training_state.action_space_size,
            'action_space': self.env.action_space,  # Save current action space
            'meta_learning_events': [asdict(event) for event in self.env.meta_learning_events]
        }
        
        self.agent.save_checkpoint(checkpoint_path, metadata)
        
        # Save training state
        training_file = checkpoint_path / 'training_state.json'
        with open(training_file, 'w') as f:
            state_dict = asdict(self.training_state)
            state_dict['last_checkpoint_time'] = time.time()
            json.dump(state_dict, f, indent=2)
        
        # Save environment log with meta-learning events
        self.env.save_episode_log(checkpoint_path)
        
        print(f"   💾 ENHANCED CHECKPOINT SAVED: {checkpoint_name}")
        print(f"      🎬 Action space size: {self.env.action_size}")
        print(f"      🧠 New actions learned: {self.training_state.new_actions_learned}")

    def _load_checkpoint(self, checkpoint_name: str) -> bool:
        """Enhanced checkpoint loading with meta-learning state"""
        
        checkpoint_path = self.checkpoint_dir / checkpoint_name
        training_file = checkpoint_path / 'training_state.json'
        
        if not training_file.exists():
            return False
        
        try:
            # Load training state
            with open(training_file, 'r') as f:
                state_dict = json.load(f)
            
            self.training_state = TrainingCheckpoint(**state_dict)
            
            # Load agent state (this will handle action space resizing)
            agent_metadata = self.agent.load_checkpoint(checkpoint_path)
            
            if agent_metadata:
                # Restore action space if saved
                if 'action_space' in agent_metadata:
                    self.env.action_space = agent_metadata['action_space']
                    print(f"   🎬 Restored action space: {len(self.env.action_space)} actions")
                
                # Restore meta-learning events
                if 'meta_learning_events' in agent_metadata:
                    self.env.meta_learning_events = [
                        MetaLearningEvent(**event) 
                        for event in agent_metadata['meta_learning_events']
                    ]
                    print(f"   🧠 Restored meta-learning events: {len(self.env.meta_learning_events)}")
                
                print(f"   📂 ENHANCED TRAINING STATE LOADED")
                print(f"      📊 Episode: {self.training_state.episode}")
                print(f"      🎯 Prompt: {self.training_state.current_prompt_index + 1}/{len(self.training_state.training_prompts)}")
                print(f"      🏆 Best Score: {self.training_state.best_overall_score:.3f}")
                print(f"      🧠 Actions Learned: {self.training_state.new_actions_learned}")
                return True
        
        except Exception as e:
            print(f"   ❌ Error loading enhanced checkpoint: {e}")
        
        return False

    def _generate_final_report(self) -> Dict:
        """Enhanced final report with meta-learning statistics"""
        
        print(f"\n🎓 COMPLETE TRAINING REPORT V2")
        print("=" * 80)
        
        total_episodes = len(self.training_state.episode_scores)
        ultra_count = sum(self.training_state.ultra_achievements)
        avg_score = sum(self.training_state.episode_scores) / total_episodes if total_episodes > 0 else 0
        avg_reward = sum(self.training_state.episode_rewards) / total_episodes if total_episodes > 0 else 0
        
        training_time = time.time() - self.training_state.training_start_time
        
        print(f"📊 TRAINING PERFORMANCE:")
        print(f"   Total Episodes: {total_episodes}")
        print(f"   Ultra Achievements: {ultra_count}/{total_episodes} ({ultra_count/total_episodes*100:.1f}%)" if total_episodes > 0 else "   Ultra Achievements: 0/0")
        print(f"   Average Score: {avg_score:.3f}")
        print(f"   Best Score: {self.training_state.best_overall_score:.3f}")
        print(f"   Average Reward: {avg_reward:.2f}")
        print(f"   Training Time: {training_time/3600:.2f} hours")
        
        print(f"\n🧠 LEARNING METRICS:")
        print(f"   Final Epsilon: {self.agent.epsilon:.3f}")
        print(f"   Replay Buffer: {len(self.agent.memory)} experiences")
        print(f"   Total Learning Steps: {self.agent.step_count}")
        print(f"   Total Learning Updates: {self.agent.learn_count}")
        print(f"   PER Beta (final): {self.agent.beta:.3f}")
        
        print(f"\n✨ META-LEARNING ACHIEVEMENTS:")
        print(f"   New Actions Learned: {self.training_state.new_actions_learned}")
        print(f"   Final Action Space Size: {self.env.action_size}")
        print(f"   Meta-learning Events: {len(self.env.meta_learning_events)}")
        print(f"   Action Space Growth: {((self.env.action_size - 18) / 18 * 100):.1f}%")
        
        # Production readiness
        readiness = self.monitor.assess_production_readiness()
        print(f"\n🚀 PRODUCTION READINESS:")
        print(f"   Status: {'✅ READY' if readiness['ready'] else '❌ NOT READY'}")
        print(f"   Ultra Rate: {readiness['metrics']['ultra_rate']:.1%}")
        print(f"   Avg Score: {readiness['metrics']['avg_score']:.3f}")
        
        # Show learned actions
        if self.training_state.new_actions_learned > 0:
            print(f"\n🎊 LEARNED ACTIONS:")
            learned_actions = self.env.action_space[18:]  # Actions beyond the original 18
            for i, action in enumerate(learned_actions[:5], 1):  # Show first 5
                print(f"   {i}. {action[1][:70]}...")
            if len(learned_actions) > 5:
                print(f"   ... and {len(learned_actions) - 5} more")
        
        return {
            'total_episodes': total_episodes,
            'ultra_achievements': ultra_count,
            'ultra_rate': ultra_count / total_episodes if total_episodes > 0 else 0,
            'average_score': avg_score,
            'best_score': self.training_state.best_overall_score,
            'training_time_hours': training_time / 3600,
            'final_epsilon': self.agent.epsilon,
            'learn_count': self.agent.learn_count,
            'new_actions_learned': self.training_state.new_actions_learned,
            'final_action_space_size': self.env.action_size,
            'meta_learning_events': len(self.env.meta_learning_events),
            'production_ready': readiness['ready'],
            'per_beta_final': self.agent.beta
        }

    def list_checkpoints(self) -> List[str]:
        """List available checkpoints"""
        
        checkpoints = []
        for path in self.checkpoint_dir.iterdir():
            if path.is_dir() and (path / 'training_state.json').exists():
                checkpoints.append(path.name)
        
        checkpoints.sort()
        return checkpoints

def main():
    """Test complete RL optimizer v2 with all patches"""
    
    print("🚀 COMPLETE RL PROMPT OPTIMIZER V2")
    print("✅ PATCH 1: Prioritized Experience Replay (3-5x faster learning)")
    print("✅ PATCH 2: Dynamic Action Discovery with LLM Knowledge Engineer")
    print("✅ Fixed: Rewards, epsilon decay, learning, actions")
    print("✅ Persistent: Checkpointing, save/load, resume")
    print("✅ Monitoring: Production readiness assessment")
    print("=" * 80)
    
    # Check for existing checkpoints
    optimizer = CompleteRLOptimizerV2(ultra_target=0.96)
    checkpoints = optimizer.list_checkpoints()
    
    if checkpoints:
        print(f"📂 FOUND EXISTING CHECKPOINTS:")
        for i, checkpoint in enumerate(checkpoints, 1):
            print(f"   {i}. {checkpoint}")
        
        choice = input(f"\n🔄 Resume from checkpoint? Enter number (1-{len(checkpoints)}) or press ENTER for new training: ").strip()
        
        resume_from = None
        if choice.isdigit() and 1 <= int(choice) <= len(checkpoints):
            resume_from = checkpoints[int(choice) - 1]
            print(f"📂 Will resume from: {resume_from}")
    else:
        resume_from = None
        print("🆕 No existing checkpoints found. Starting fresh training.")
    
    # Training prompts
    training_prompts = [
        "hexagonal prism steel structure",
        "elegant silk fabric draping",
        "rusty metal gear mechanism"
    ]
    
    # Start enhanced training
    results = optimizer.train_with_checkpoints(
        target_prompts=training_prompts,
        episodes_per_prompt=8,  # More episodes to see meta-learning in action
        resume_from=resume_from
    )
    
    print(f"\n🎉 COMPLETE RL TRAINING V2 SESSION COMPLETE!")
    print(f"🚀 Production ready: {results.get('production_ready', False)}")
    print(f"🧠 New actions learned: {results.get('new_actions_learned', 0)}")
    print(f"🎬 Final action space: {results.get('final_action_space_size', 18)}")

if __name__ == "__main__":
    main() 