#!/usr/bin/env python3
"""
Complete RL Prompt Optimizer
============================
Combines ALL features:
✅ FIXED: Proper reward function, epsilon decay, batch learning, smart actions
✅ PERSISTENT: Checkpointing, save/load, resume training, graceful interruption
✅ MONITORING: Training progress tracking and production readiness assessment
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
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, asdict
import sqlite3
from pathlib import Path
import re
import signal
import datetime
import statistics

Experience = namedtuple('Experience', ['state', 'action', 'reward', 'next_state', 'done'])

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

class CompletePromptEnvironment:
    """Complete environment with fixes and persistence"""
    
    def __init__(self, ultra_target: float = 0.96, checkpoint_dir: str = "rl_checkpoints_complete"):
        self.ultra_target = ultra_target
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(exist_ok=True)
        
        self.target_prompt = ""
        self.current_prompt = ""
        self.validation_history = []
        self.step_count = 0
        self.max_steps = 8  # Shorter episodes for faster learning
        
        # FIXED: Smart action space
        self.action_space = self._define_smart_action_space()
        self.state_size = 20
        self.action_size = len(self.action_space)
        
        # Session tracking
        self.session_id = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.episode_log = []
        
        print(f"🎮 COMPLETE RL ENVIRONMENT INITIALIZED")
        print(f"   🎯 Target Score: {ultra_target}")
        print(f"   💾 Checkpoint Dir: {self.checkpoint_dir}")
        print(f"   🆔 Session ID: {self.session_id}")
        print(f"   🎬 Action Space Size: {self.action_size}")

    def _define_smart_action_space(self) -> List[Tuple]:
        """FIXED: Smart action space that creates better prompts"""
        
        actions = []
        
        # Proven high-scoring patterns (REPLACE entire middle section)
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
        
        # Quality upgrades (REPLACE existing descriptors)
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

    def reset(self, target_prompt: str) -> np.ndarray:
        """Reset environment with logging"""
        
        self.target_prompt = target_prompt
        self.current_prompt = f"wbgmsst, {target_prompt}, white background"
        self.validation_history = []
        self.step_count = 0
        
        initial_score = self._validate_prompt(self.current_prompt)
        self.validation_history.append(initial_score)
        
        # Log episode start
        episode_start = {
            'timestamp': time.time(),
            'target_prompt': target_prompt,
            'initial_prompt': self.current_prompt,
            'initial_score': initial_score,
            'session_id': self.session_id
        }
        self.episode_log.append(episode_start)
        
        print(f"🔄 ENVIRONMENT RESET")
        print(f"   🎯 Target: {target_prompt}")
        print(f"   📊 Initial Score: {initial_score:.3f}")
        
        return self._get_state()

    def step(self, action_idx: int) -> Tuple[np.ndarray, float, bool, Dict]:
        """FIXED: Better step function with proper rewards and logging"""
        
        self.step_count += 1
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
        
        # Log step
        step_data = {
            'step': self.step_count,
            'action': action,
            'old_prompt': old_prompt,
            'new_prompt': self.current_prompt,
            'old_score': old_score,
            'new_score': new_score,
            'reward': reward,
            'done': done
        }
        self.episode_log[-1].setdefault('steps', []).append(step_data)
        
        info = {
            'score': new_score,
            'prompt': self.current_prompt,
            'action_taken': action[0],
            'step': self.step_count,
            'improvement': new_score - old_score
        }
        
        return self._get_state(), reward, done, info

    def _apply_smart_action(self, action: Tuple) -> str:
        """FIXED: Smart action application"""
        
        action_type, modifier, mode = action
        
        if action_type == 'APPLY_PATTERN':
            # Replace with proven pattern
            pattern = modifier.replace('{target}', self.target_prompt)
            return f"wbgmsst, {pattern}, white background"
        
        elif action_type in ['UPGRADE_AUTHORITY', 'UPGRADE_PROCESS', 'UPGRADE_QUALITY']:
            # Intelligent upgrade
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
        base_reward = score_improvement * 100  # Scale up improvements
        
        # Absolute score bonuses
        if new_score >= self.ultra_target:
            base_reward += 200  # Huge bonus for ultra
        elif new_score >= 0.9:
            base_reward += 100  # Big bonus for excellent
        elif new_score >= 0.8:
            base_reward += 50   # Good bonus for high
        elif new_score >= 0.7:
            base_reward += 20   # Small bonus for decent
        
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
        """FIXED: Simplified but effective state representation"""
        
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
        
        return state

    def save_episode_log(self, checkpoint_path: Path):
        """Save detailed episode log"""
        log_file = checkpoint_path / f"episode_log_{self.session_id}.json"
        with open(log_file, 'w') as f:
            json.dump(self.episode_log, f, indent=2)

class CompleteDQN(nn.Module):
    """Complete DQN with proper architecture"""
    
    def __init__(self, state_size: int, action_size: int):
        super(CompleteDQN, self).__init__()
        
        self.fc1 = nn.Linear(state_size, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, action_size)
        
        self.dropout = nn.Dropout(0.1)
        self.layer_norm1 = nn.LayerNorm(128)
        self.layer_norm2 = nn.LayerNorm(128)

    def forward(self, x):
        x = F.relu(self.layer_norm1(self.fc1(x)))
        x = self.dropout(x)
        x = F.relu(self.layer_norm2(self.fc2(x)))
        x = self.dropout(x)
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return x

class PersistentReplayBuffer:
    """Replay buffer with save/load capabilities"""
    
    def __init__(self, capacity: int = 5000):
        self.buffer = deque(maxlen=capacity)
        self.capacity = capacity
        
    def push(self, experience: Experience):
        self.buffer.append(experience)
    
    def sample(self, batch_size: int) -> List[Experience]:
        return random.sample(self.buffer, batch_size)
    
    def __len__(self):
        return len(self.buffer)
    
    def save(self, filepath: Path):
        """Save replay buffer"""
        with open(filepath, 'wb') as f:
            pickle.dump(list(self.buffer), f)
        print(f"   💾 Replay buffer saved: {len(self.buffer)} experiences")
    
    def load(self, filepath: Path):
        """Load replay buffer"""
        if filepath.exists():
            with open(filepath, 'rb') as f:
                experiences = pickle.load(f)
                self.buffer = deque(experiences, maxlen=self.capacity)
            print(f"   📂 Replay buffer loaded: {len(self.buffer)} experiences")
            return True
        return False

class CompleteDQNAgent:
    """COMPLETE: Fixed DQN Agent with persistence"""
    
    def __init__(self, state_size: int, action_size: int, checkpoint_dir: str, lr: float = 0.001):
        self.state_size = state_size
        self.action_size = action_size
        self.checkpoint_dir = Path(checkpoint_dir)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Networks
        self.q_network_local = CompleteDQN(state_size, action_size).to(self.device)
        self.q_network_target = CompleteDQN(state_size, action_size).to(self.device)
        self.optimizer = optim.Adam(self.q_network_local.parameters(), lr=lr)
        
        # Replay buffer
        self.memory = PersistentReplayBuffer(capacity=5000)
        
        # FIXED: Proper hyperparameters
        self.batch_size = 32
        self.gamma = 0.95
        self.tau = 0.005
        self.update_every = 2
        
        # FIXED: Proper epsilon schedule
        self.epsilon = 0.9       # Start high but not 1.0
        self.epsilon_min = 0.05  # Don't go to zero
        self.epsilon_decay = 0.98  # Faster decay
        
        # Learning tracking
        self.step_count = 0
        self.learn_count = 0
        self.losses = []
        self.q_values_history = []
        
        print(f"🤖 COMPLETE DQN AGENT INITIALIZED")
        print(f"   🧠 Device: {self.device}")
        print(f"   🎲 Initial Epsilon: {self.epsilon}")
        print(f"   💾 Checkpoint Dir: {self.checkpoint_dir}")

    def act(self, state: np.ndarray, training: bool = True) -> int:
        """FIXED: Proper epsilon-greedy with decay"""
        
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
        """FIXED: Proper learning with epsilon decay"""
        
        # Store experience
        experience = Experience(state, action, reward, next_state, done)
        self.memory.push(experience)
        
        self.step_count += 1
        
        # Learn more frequently
        if self.step_count % self.update_every == 0 and len(self.memory) >= self.batch_size:
            experiences = self.memory.sample(self.batch_size)
            loss = self.learn(experiences)
            
            # FIXED: Decay epsilon after learning
            if self.epsilon > self.epsilon_min:
                self.epsilon *= self.epsilon_decay
                
            return loss
        
        return None

    def learn(self, experiences: List[Experience]) -> float:
        """FIXED: Actual learning implementation"""
        
        self.learn_count += 1
        
        # Convert to tensors
        states = torch.FloatTensor([e.state for e in experiences]).to(self.device)
        actions = torch.LongTensor([e.action for e in experiences]).to(self.device)
        rewards = torch.FloatTensor([e.reward for e in experiences]).to(self.device)
        next_states = torch.FloatTensor([e.next_state for e in experiences]).to(self.device)
        dones = torch.BoolTensor([e.done for e in experiences]).to(self.device)
        
        # Current Q values
        current_q_values = self.q_network_local(states).gather(1, actions.unsqueeze(1))
        
        # Next Q values from target network
        next_q_values = self.q_network_target(next_states).detach().max(1)[0]
        target_q_values = rewards + (self.gamma * next_q_values * ~dones)
        
        # Compute loss
        loss = F.mse_loss(current_q_values.squeeze(), target_q_values)
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_network_local.parameters(), 1.0)
        self.optimizer.step()
        
        # Soft update target network
        self.soft_update()
        
        # Track loss
        self.losses.append(loss.item())
        
        print(f"   📚 LEARNING #{self.learn_count}: Loss {loss.item():.4f}, Buffer {len(self.memory)}, ε={self.epsilon:.3f}")
        return loss.item()

    def soft_update(self):
        """Soft update target network"""
        for target_param, local_param in zip(self.q_network_target.parameters(),
                                           self.q_network_local.parameters()):
            target_param.data.copy_(self.tau * local_param.data + 
                                  (1.0 - self.tau) * target_param.data)

    def save_checkpoint(self, checkpoint_path: Path, metadata: Dict):
        """Save complete agent state"""
        
        checkpoint = {
            'q_network_local_state_dict': self.q_network_local.state_dict(),
            'q_network_target_state_dict': self.q_network_target.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'step_count': self.step_count,
            'learn_count': self.learn_count,
            'losses': self.losses,
            'q_values_history': self.q_values_history,
            'metadata': metadata
        }
        
        model_file = checkpoint_path / 'agent_checkpoint.pth'
        torch.save(checkpoint, model_file)
        
        # Save replay buffer separately
        buffer_file = checkpoint_path / 'replay_buffer.pkl'
        self.memory.save(buffer_file)
        
        print(f"   💾 Agent checkpoint saved")

    def load_checkpoint(self, checkpoint_path: Path) -> Optional[Dict]:
        """Load complete agent state"""
        
        model_file = checkpoint_path / 'agent_checkpoint.pth'
        buffer_file = checkpoint_path / 'replay_buffer.pkl'
        
        if model_file.exists():
            checkpoint = torch.load(model_file, map_location=self.device, weights_only=False)
            
            self.q_network_local.load_state_dict(checkpoint['q_network_local_state_dict'])
            self.q_network_target.load_state_dict(checkpoint['q_network_target_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.epsilon = checkpoint['epsilon']
            self.step_count = checkpoint['step_count']
            self.learn_count = checkpoint.get('learn_count', 0)
            self.losses = checkpoint['losses']
            self.q_values_history = checkpoint.get('q_values_history', [])
            
            # Load replay buffer
            self.memory.load(buffer_file)
            
            print(f"   📂 Agent checkpoint loaded")
            print(f"      🎲 Epsilon: {self.epsilon:.3f}")
            print(f"      🔢 Step count: {self.step_count}")
            print(f"      📚 Learn count: {self.learn_count}")
            print(f"      📊 Buffer size: {len(self.memory)}")
            
            return checkpoint['metadata']
        
        return None

class ProductionReadinessMonitor:
    """Monitor production readiness during training"""
    
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
        """Add episode metrics"""
        self.metrics_history.append(metrics)
        
    def assess_production_readiness(self) -> Dict:
        """Check if ready for production"""
        
        if len(self.metrics_history) < self.success_criteria["min_episodes"]:
            return {"ready": False, "reason": "Insufficient episodes"}
        
        recent_window = self.metrics_history[-30:]  # Last 30 episodes
        
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
        
        ready = sum(checks.values()) >= len(checks) * 0.8  # 80% of checks pass
        
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

class CompleteRLOptimizer:
    """COMPLETE: Fixed + Persistent RL optimizer"""
    
    def __init__(self, ultra_target: float = 0.96, checkpoint_dir: str = "rl_checkpoints_complete"):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(exist_ok=True)
        
        self.env = CompletePromptEnvironment(ultra_target, checkpoint_dir)
        self.agent = CompleteDQNAgent(
            state_size=self.env.state_size,
            action_size=self.env.action_size,
            checkpoint_dir=checkpoint_dir
        )
        
        # Training state
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
            last_checkpoint_time=time.time()
        )
        
        # Production readiness monitor
        self.monitor = ProductionReadinessMonitor()
        
        # Graceful shutdown handling
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        
        print(f"🚀 COMPLETE RL OPTIMIZER INITIALIZED")
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

    def train_with_checkpoints(self, target_prompts: List[str], 
                              episodes_per_prompt: int = 5,
                              resume_from: Optional[str] = None) -> Dict:
        """Complete training with checkpoints and monitoring"""
        
        print(f"🎓 COMPLETE RL TRAINING SESSION")
        print(f"📝 Prompts: {len(target_prompts)}")
        print(f"🔄 Episodes per prompt: {episodes_per_prompt}")
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
        
        # Training loop
        for prompt_idx in range(self.training_state.current_prompt_index, total_prompts):
            current_prompt = self.training_state.training_prompts[prompt_idx]
            
            print(f"\n🎯 TRAINING ON PROMPT {prompt_idx + 1}/{total_prompts}: '{current_prompt}'")
            
            # Episodes for this prompt
            episodes_completed = 0
            if prompt_idx == self.training_state.current_prompt_index:
                episodes_completed = self.training_state.episode - (prompt_idx * episodes_per_prompt)
            
            for episode_in_prompt in range(episodes_completed, episodes_per_prompt):
                episode_num = prompt_idx * episodes_per_prompt + episode_in_prompt + 1
                
                print(f"\n📚 EPISODE {episode_num} (Prompt {prompt_idx + 1}, Episode {episode_in_prompt + 1}/{episodes_per_prompt})")
                
                # Train episode
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
                    print(f"   Ultra Rate: {readiness['metrics']['ultra_rate']:.1%}")
                    print(f"   Avg Score: {readiness['metrics']['avg_score']:.3f}")
                    
                    if readiness['ready']:
                        print(f"\n🎉 MODEL IS READY FOR PRODUCTION!")
                        self._save_checkpoint("production_ready")
                        return self._generate_final_report()
                
                # Pause for user input
                print(f"\n⏸️ EPISODE {episode_num} COMPLETE")
                print(f"   📊 Score: {result['best_score']:.3f}")
                print(f"   🎁 Reward: {result['total_reward']:.2f}")
                print(f"   🎉 Ultra: {result['ultra_achieved']}")
                print(f"   🎲 Epsilon: {result['epsilon']:.3f}")
                print(f"   📚 Learn Count: {result['learn_count']}")
                
                user_input = input("\n➡️ Press ENTER to continue, 'q' to quit, 's' to save and quit: ").strip().lower()
                
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
        """Train one episode with complete tracking"""
        
        state = self.env.reset(target_prompt)
        total_reward = 0
        step = 0
        best_score = 0
        losses = []
        
        while True:
            action = self.agent.act(state, training=True)
            next_state, reward, done, info = self.env.step(action)
            
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
            'final_prompt': info['prompt']
        }

    def _save_checkpoint(self, checkpoint_name: str):
        """Save complete training checkpoint"""
        
        checkpoint_path = self.checkpoint_dir / checkpoint_name
        checkpoint_path.mkdir(exist_ok=True)
        
        # Save agent state
        metadata = {
            'episode': self.training_state.episode,
            'total_episodes_completed': self.training_state.total_episodes_completed,
            'current_prompt_index': self.training_state.current_prompt_index,
            'training_prompts': self.training_state.training_prompts,
            'episodes_per_prompt': self.training_state.episodes_per_prompt,
            'best_overall_score': self.training_state.best_overall_score,
            'training_start_time': self.training_state.training_start_time,
            'checkpoint_time': time.time()
        }
        
        self.agent.save_checkpoint(checkpoint_path, metadata)
        
        # Save training state
        training_file = checkpoint_path / 'training_state.json'
        with open(training_file, 'w') as f:
            state_dict = asdict(self.training_state)
            state_dict['last_checkpoint_time'] = time.time()
            json.dump(state_dict, f, indent=2)
        
        # Save environment log
        self.env.save_episode_log(checkpoint_path)
        
        print(f"   💾 CHECKPOINT SAVED: {checkpoint_name}")

    def _load_checkpoint(self, checkpoint_name: str) -> bool:
        """Load complete training checkpoint"""
        
        checkpoint_path = self.checkpoint_dir / checkpoint_name
        training_file = checkpoint_path / 'training_state.json'
        
        if not training_file.exists():
            return False
        
        try:
            # Load training state
            with open(training_file, 'r') as f:
                state_dict = json.load(f)
            
            self.training_state = TrainingCheckpoint(**state_dict)
            
            # Load agent state
            agent_metadata = self.agent.load_checkpoint(checkpoint_path)
            
            if agent_metadata:
                print(f"   📂 TRAINING STATE LOADED")
                print(f"      📊 Episode: {self.training_state.episode}")
                print(f"      🎯 Prompt: {self.training_state.current_prompt_index + 1}/{len(self.training_state.training_prompts)}")
                print(f"      🏆 Best Score: {self.training_state.best_overall_score:.3f}")
                return True
        
        except Exception as e:
            print(f"   ❌ Error loading checkpoint: {e}")
        
        return False

    def _generate_final_report(self) -> Dict:
        """Generate comprehensive final report"""
        
        print(f"\n🎓 COMPLETE TRAINING REPORT")
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
        
        # Production readiness
        readiness = self.monitor.assess_production_readiness()
        print(f"\n🚀 PRODUCTION READINESS:")
        print(f"   Status: {'✅ READY' if readiness['ready'] else '❌ NOT READY'}")
        print(f"   Ultra Rate: {readiness['metrics']['ultra_rate']:.1%}")
        print(f"   Avg Score: {readiness['metrics']['avg_score']:.3f}")
        
        return {
            'total_episodes': total_episodes,
            'ultra_achievements': ultra_count,
            'ultra_rate': ultra_count / total_episodes if total_episodes > 0 else 0,
            'average_score': avg_score,
            'best_score': self.training_state.best_overall_score,
            'training_time_hours': training_time / 3600,
            'final_epsilon': self.agent.epsilon,
            'learn_count': self.agent.learn_count,
            'production_ready': readiness['ready']
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
    """Test complete RL optimizer"""
    
    print("🚀 COMPLETE RL PROMPT OPTIMIZER")
    print("✅ Fixed: Rewards, epsilon decay, learning, actions")
    print("✅ Persistent: Checkpointing, save/load, resume")
    print("✅ Monitoring: Production readiness assessment")
    print("=" * 80)
    
    # Check for existing checkpoints
    optimizer = CompleteRLOptimizer(ultra_target=0.96)
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
        "elegant silk fabric draping"
    ]
    
    # Start complete training
    results = optimizer.train_with_checkpoints(
        target_prompts=training_prompts,
        episodes_per_prompt=5,
        resume_from=resume_from
    )
    
    print(f"\n🎉 COMPLETE RL TRAINING SESSION COMPLETE!")
    print(f"🚀 Production ready: {results.get('production_ready', False)}")

if __name__ == "__main__":
    main() 