#!/usr/bin/env python3
"""
Persistent Reinforcement Learning Prompt Optimizer
=================================================
Features:
- Automatic checkpointing after each episode
- Save/load model weights, replay buffer, and training state
- Resume training from any checkpoint
- Persistent learning across sessions
- Graceful interruption handling
- Training progress tracking and visualization
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

# Experience tuple for replay buffer
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
    best_overall_score: float
    training_start_time: float
    last_checkpoint_time: float
    model_state_dict: Dict
    optimizer_state_dict: Dict
    target_model_state_dict: Dict

class PersistentPromptEnvironment:
    """Enhanced environment with state persistence"""
    
    def __init__(self, ultra_target: float = 0.96, checkpoint_dir: str = "rl_checkpoints"):
        self.ultra_target = ultra_target
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(exist_ok=True)
        
        self.target_prompt = ""
        self.current_prompt = ""
        self.validation_history = []
        self.step_count = 0
        self.max_steps = 10
        
        # Action space definition
        self.action_space = self._define_action_space()
        self.state_size = 25
        self.action_size = len(self.action_space)
        
        # Session tracking
        self.session_id = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.episode_log = []
        
        print(f"🎮 PERSISTENT RL ENVIRONMENT INITIALIZED")
        print(f"   🎯 Target Score: {ultra_target}")
        print(f"   💾 Checkpoint Dir: {self.checkpoint_dir}")
        print(f"   🆔 Session ID: {self.session_id}")
        print(f"   🎬 Action Space Size: {self.action_size}")

    def _define_action_space(self) -> List:
        """Define comprehensive action space"""
        
        actions = []
        
        # Authority descriptors
        authority_descriptors = [
            "aerospace-grade", "military-spec", "defense-grade", "aviation-standard",
            "laboratory-grade", "precision-aerospace", "ultra-military-spec"
        ]
        
        # Process descriptors  
        process_descriptors = [
            "precision-engineered", "ultra-precision", "masterpiece-quality", 
            "ultra-detailed", "precision-forged", "ultra-refined"
        ]
        
        # Quality descriptors
        quality_descriptors = [
            "ultra-high technical specification", "advanced engineering design",
            "premium manufacturing excellence", "aerospace-engineering excellence"
        ]
        
        # Build action space
        for position in ['prefix', 'middle', 'suffix']:
            for desc in authority_descriptors:
                actions.append(('ADD_AUTHORITY', desc, '', position))
            for desc in process_descriptors:
                actions.append(('ADD_PROCESS', desc, '', position))
            for desc in quality_descriptors:
                actions.append(('ADD_QUALITY', desc, '', position))
        
        # Special combination actions
        proven_combos = [
            "defense-grade ultra-precision",
            "aerospace-grade precision-engineered", 
            "military-spec ultra-detailed"
        ]
        for combo in proven_combos:
            actions.append(('APPLY_COMBO', combo, '', 'prefix'))
        
        return actions

    def reset(self, target_prompt: str) -> np.ndarray:
        """Reset environment and log episode start"""
        
        self.target_prompt = target_prompt
        self.current_prompt = f"wbgmsst, {target_prompt}, white background"
        self.validation_history = []
        self.step_count = 0
        
        # Get initial score
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
        """Execute action with logging"""
        
        self.step_count += 1
        action = self.action_space[action_idx]
        
        print(f"🎬 STEP {self.step_count}: {action[0]}")
        print(f"   🔧 Action: {action[1]} ({action[3]})")
        
        # Apply action
        old_prompt = self.current_prompt
        self.current_prompt = self._apply_action(action)
        
        # Validate
        new_score = self._validate_prompt(self.current_prompt)
        self.validation_history.append(new_score)
        
        # Calculate reward
        reward = self._calculate_reward(new_score, old_prompt)
        
        # Check done
        done = (new_score >= self.ultra_target or self.step_count >= self.max_steps)
        
        print(f"   📝 New Prompt: {self.current_prompt}")
        print(f"   📊 Score: {new_score:.3f}")
        print(f"   🎁 Reward: {reward:.3f}")
        
        # Log step
        step_data = {
            'step': self.step_count,
            'action': action,
            'old_prompt': old_prompt,
            'new_prompt': self.current_prompt,
            'score': new_score,
            'reward': reward,
            'done': done
        }
        self.episode_log[-1].setdefault('steps', []).append(step_data)
        
        info = {
            'score': new_score,
            'prompt': self.current_prompt,
            'action_taken': action[0],
            'step': self.step_count
        }
        
        return self._get_state(), reward, done, info

    def _apply_action(self, action: Tuple) -> str:
        """Apply action to modify prompt"""
        
        action_type, descriptor, old_descriptor, position = action
        prompt = self.current_prompt
        
        if action_type == "ADD_AUTHORITY" or action_type == "ADD_PROCESS":
            parts = prompt.split(', ')
            if len(parts) >= 3:
                if position == "prefix":
                    parts[1] = f"{descriptor} {parts[1]}"
                elif position == "middle":
                    parts.insert(-1, descriptor)
                elif position == "suffix":
                    parts[-2] = f"{parts[-2]}, {descriptor}"
            prompt = ', '.join(parts)
            
        elif action_type == "ADD_QUALITY":
            parts = prompt.split(', white background')
            if len(parts) == 2:
                prompt = f"{parts[0]}, {descriptor}, white background"
                
        elif action_type == "APPLY_COMBO":
            # Apply proven combination at prefix
            parts = prompt.split(', ')
            if len(parts) >= 2:
                parts[1] = f"{descriptor} {parts[1]}"
            prompt = ', '.join(parts)
        
        # Clean up
        prompt = re.sub(r',\s*,', ',', prompt)
        prompt = re.sub(r'\s+', ' ', prompt)
        
        return prompt.strip()

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

    def _calculate_reward(self, new_score: float, old_prompt: str) -> float:
        """Enhanced reward calculation"""
        
        prev_score = self.validation_history[-2] if len(self.validation_history) > 1 else 0.0
        score_improvement = new_score - prev_score
        reward = score_improvement * 10
        
        # Bonuses
        if new_score >= self.ultra_target:
            reward += 100
        elif new_score >= 0.9:
            reward += 50
        elif new_score >= 0.8:
            reward += 20
        
        # Penalties
        if len(self.current_prompt) > 150:
            reward -= 5
        
        return reward

    def _get_state(self) -> np.ndarray:
        """Get state representation"""
        
        state = np.zeros(self.state_size)
        
        # Validation history (last 5 scores)
        history = self.validation_history[-5:]
        for i, score in enumerate(history):
            if i < 5:
                state[i] = score
        
        # Current step normalized
        state[5] = self.step_count / self.max_steps
        
        # Best score so far
        state[6] = max(self.validation_history) if self.validation_history else 0.0
        
        # Prompt length normalized
        state[7] = len(self.current_prompt) / 150
        
        # Descriptor counts
        authority_count = sum(1 for desc in ["aerospace", "military", "defense", "aviation"] 
                             if desc in self.current_prompt.lower())
        process_count = sum(1 for desc in ["precision", "ultra", "masterpiece", "detailed"] 
                           if desc in self.current_prompt.lower())
        quality_count = sum(1 for desc in ["specification", "engineering", "excellence"] 
                           if desc in self.current_prompt.lower())
        
        state[8] = min(authority_count / 3, 1.0)
        state[9] = min(process_count / 3, 1.0)
        state[10] = min(quality_count / 2, 1.0)
        
        # Target characteristics
        target_lower = self.target_prompt.lower()
        state[11] = 1.0 if any(word in target_lower for word in ["steel", "metal"]) else 0.0
        state[12] = 1.0 if any(word in target_lower for word in ["fabric", "cloth"]) else 0.0
        state[13] = 1.0 if any(word in target_lower for word in ["glass", "crystal"]) else 0.0
        
        return state

    def save_episode_log(self, checkpoint_path: Path):
        """Save detailed episode log"""
        log_file = checkpoint_path / f"episode_log_{self.session_id}.json"
        with open(log_file, 'w') as f:
            json.dump(self.episode_log, f, indent=2)

class PersistentReplayBuffer:
    """Replay buffer with save/load capabilities"""
    
    def __init__(self, capacity: int = 10000):
        self.buffer = deque(maxlen=capacity)
        self.capacity = capacity
        
    def push(self, experience: Experience):
        self.buffer.append(experience)
    
    def sample(self, batch_size: int) -> List[Experience]:
        return random.sample(self.buffer, batch_size)
    
    def __len__(self):
        return len(self.buffer)
    
    def save(self, filepath: Path):
        """Save replay buffer to file"""
        with open(filepath, 'wb') as f:
            pickle.dump(list(self.buffer), f)
        print(f"   💾 Replay buffer saved: {len(self.buffer)} experiences")
    
    def load(self, filepath: Path):
        """Load replay buffer from file"""
        if filepath.exists():
            with open(filepath, 'rb') as f:
                experiences = pickle.load(f)
                self.buffer = deque(experiences, maxlen=self.capacity)
            print(f"   📂 Replay buffer loaded: {len(self.buffer)} experiences")
            return True
        return False

class PersistentDQN(nn.Module):
    """DQN with enhanced architecture"""
    
    def __init__(self, state_size: int, action_size: int, hidden_size: int = 256):
        super(PersistentDQN, self).__init__()
        
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, hidden_size)
        self.fc4 = nn.Linear(hidden_size, action_size)
        
        self.dropout = nn.Dropout(0.2)
        self.layer_norm1 = nn.LayerNorm(hidden_size)
        self.layer_norm2 = nn.LayerNorm(hidden_size)

    def forward(self, x):
        x = F.relu(self.layer_norm1(self.fc1(x)))
        x = self.dropout(x)
        x = F.relu(self.layer_norm2(self.fc2(x)))
        x = self.dropout(x)
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return x

class PersistentDQNAgent:
    """DQN Agent with comprehensive save/load functionality"""
    
    def __init__(self, state_size: int, action_size: int, checkpoint_dir: str, lr: float = 0.001):
        self.state_size = state_size
        self.action_size = action_size
        self.checkpoint_dir = Path(checkpoint_dir)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Networks
        self.q_network_local = PersistentDQN(state_size, action_size).to(self.device)
        self.q_network_target = PersistentDQN(state_size, action_size).to(self.device)
        self.optimizer = optim.Adam(self.q_network_local.parameters(), lr=lr)
        
        # Replay buffer
        self.memory = PersistentReplayBuffer(capacity=10000)
        
        # Training parameters
        self.batch_size = 64
        self.gamma = 0.99
        self.tau = 0.001
        self.update_every = 4
        
        # Exploration
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        
        # Tracking
        self.step_count = 0
        self.losses = []
        self.q_values_history = []
        
        print(f"🤖 PERSISTENT DQN AGENT INITIALIZED")
        print(f"   🧠 Device: {self.device}")
        print(f"   💾 Checkpoint Dir: {self.checkpoint_dir}")

    def act(self, state: np.ndarray, training: bool = True) -> int:
        """Choose action with Q-value tracking"""
        
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
            
            print(f"   🧠 EXPLOITATION: Action {action} (Q={max_q:.3f}, ε={self.epsilon:.3f})")
            return action

    def step(self, state: np.ndarray, action: int, reward: float, 
             next_state: np.ndarray, done: bool):
        """Enhanced step with learning tracking"""
        
        experience = Experience(state, action, reward, next_state, done)
        self.memory.push(experience)
        
        self.step_count += 1
        if self.step_count % self.update_every == 0:
            if len(self.memory) > self.batch_size:
                experiences = self.memory.sample(self.batch_size)
                loss = self.learn(experiences)
                return loss
        return None

    def learn(self, experiences: List[Experience]) -> float:
        """Learn with loss tracking"""
        
        states = torch.FloatTensor([e.state for e in experiences]).to(self.device)
        actions = torch.LongTensor([e.action for e in experiences]).to(self.device)
        rewards = torch.FloatTensor([e.reward for e in experiences]).to(self.device)
        next_states = torch.FloatTensor([e.next_state for e in experiences]).to(self.device)
        dones = torch.BoolTensor([e.done for e in experiences]).to(self.device)
        
        current_q_values = self.q_network_local(states).gather(1, actions.unsqueeze(1))
        next_q_values = self.q_network_target(next_states).detach().max(1)[0]
        target_q_values = rewards + (self.gamma * next_q_values * ~dones)
        
        loss = F.mse_loss(current_q_values.squeeze(), target_q_values)
        
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_network_local.parameters(), 1.0)
        self.optimizer.step()
        
        self.losses.append(loss.item())
        self.soft_update()
        
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        
        print(f"   📚 LEARNING: Loss {loss.item():.4f}, Buffer {len(self.memory)}")
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
            checkpoint = torch.load(model_file, map_location=self.device)
            
            self.q_network_local.load_state_dict(checkpoint['q_network_local_state_dict'])
            self.q_network_target.load_state_dict(checkpoint['q_network_target_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.epsilon = checkpoint['epsilon']
            self.step_count = checkpoint['step_count']
            self.losses = checkpoint['losses']
            self.q_values_history = checkpoint.get('q_values_history', [])
            
            # Load replay buffer
            self.memory.load(buffer_file)
            
            print(f"   📂 Agent checkpoint loaded")
            print(f"      🎲 Epsilon: {self.epsilon:.3f}")
            print(f"      🔢 Step count: {self.step_count}")
            print(f"      📚 Buffer size: {len(self.memory)}")
            
            return checkpoint['metadata']
        
        return None

class PersistentRLOptimizer:
    """Main RL optimizer with comprehensive persistence"""
    
    def __init__(self, ultra_target: float = 0.96, checkpoint_dir: str = "rl_checkpoints"):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(exist_ok=True)
        
        self.env = PersistentPromptEnvironment(ultra_target, checkpoint_dir)
        self.agent = PersistentDQNAgent(
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
            best_overall_score=0.0,
            training_start_time=time.time(),
            last_checkpoint_time=time.time(),
            model_state_dict={},
            optimizer_state_dict={},
            target_model_state_dict={}
        )
        
        # Graceful shutdown handling
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        
        print(f"🚀 PERSISTENT RL OPTIMIZER INITIALIZED")
        print(f"   💾 Auto-checkpointing enabled")
        print(f"   🔄 Resume capability enabled")

    def _signal_handler(self, signum, frame):
        """Handle interruption gracefully"""
        print(f"\n⚠️ INTERRUPTION DETECTED (Signal {signum})")
        print("💾 Saving checkpoint before exit...")
        self._save_checkpoint("emergency_checkpoint")
        print("✅ Emergency checkpoint saved!")
        sys.exit(0)

    def train_with_checkpoints(self, target_prompts: List[str], 
                              episodes_per_prompt: int = 5,
                              resume_from: Optional[str] = None) -> Dict:
        """Train with automatic checkpointing and resume capability"""
        
        print(f"🎓 PERSISTENT RL TRAINING SESSION")
        print(f"📝 Prompts: {len(target_prompts)}")
        print(f"🔄 Episodes per prompt: {episodes_per_prompt}")
        print("=" * 80)
        
        # Load previous state if resuming
        if resume_from:
            if self._load_checkpoint(resume_from):
                print(f"📂 RESUMED FROM CHECKPOINT: {resume_from}")
            else:
                print(f"❌ Could not load checkpoint: {resume_from}")
                print("🔄 Starting fresh training...")
        
        # Initialize or update training state
        if not self.training_state.training_prompts:
            self.training_state.training_prompts = target_prompts
            self.training_state.episodes_per_prompt = episodes_per_prompt
            self.training_state.training_start_time = time.time()
        
        total_prompts = len(self.training_state.training_prompts)
        
        # Continue from where we left off
        for prompt_idx in range(self.training_state.current_prompt_index, total_prompts):
            current_prompt = self.training_state.training_prompts[prompt_idx]
            
            print(f"\n🎯 TRAINING ON PROMPT {prompt_idx + 1}/{total_prompts}: '{current_prompt}'")
            
            # Calculate episodes completed for this prompt
            episodes_completed_this_prompt = 0
            if prompt_idx == self.training_state.current_prompt_index:
                episodes_completed_this_prompt = self.training_state.episode - (prompt_idx * episodes_per_prompt)
            
            # Continue episodes for this prompt
            for episode_in_prompt in range(episodes_completed_this_prompt, episodes_per_prompt):
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
                self.training_state.best_overall_score = max(
                    self.training_state.best_overall_score, 
                    result['best_score']
                )
                
                # Auto-checkpoint after each episode
                checkpoint_name = f"episode_{episode_num:03d}"
                self._save_checkpoint(checkpoint_name)
                
                # Pause for user input after each episode
                print(f"\n⏸️ EPISODE {episode_num} COMPLETE")
                print(f"   📊 Score: {result['best_score']:.3f}")
                print(f"   🎁 Reward: {result['total_reward']:.2f}")
                print(f"   🎉 Ultra: {result['ultra_achieved']}")
                
                user_input = input("\n➡️ Press ENTER to continue, 'q' to quit, 's' to save and quit: ").strip().lower()
                
                if user_input == 'q':
                    print("🛑 Training stopped by user")
                    return self._generate_final_report()
                elif user_input == 's':
                    print("💾 Saving and exiting...")
                    self._save_checkpoint("user_save")
                    return self._generate_final_report()
            
            # Update prompt index after completing all episodes for current prompt
            self.training_state.current_prompt_index = prompt_idx + 1
            
            # Evaluate after completing this prompt
            print(f"\n🔬 EVALUATING PROMPT: '{current_prompt}'")
            eval_result = self._evaluate_episode(current_prompt)
            
            print(f"📈 PROMPT '{current_prompt}' TRAINING COMPLETE")
            prompt_scores = self.training_state.episode_scores[-episodes_per_prompt:]
            print(f"   📊 Training Scores: {[f'{s:.3f}' for s in prompt_scores]}")
            print(f"   🏆 Best Training: {max(prompt_scores):.3f}")
            print(f"   🔬 Evaluation: {eval_result['best_score']:.3f}")
        
        print(f"\n🎉 ALL TRAINING COMPLETE!")
        return self._generate_final_report()

    def _train_single_episode(self, target_prompt: str, episode_num: int) -> Dict:
        """Train one episode with detailed tracking"""
        
        state = self.env.reset(target_prompt)
        total_reward = 0
        step = 0
        best_score = 0
        
        while True:
            action = self.agent.act(state, training=True)
            next_state, reward, done, info = self.env.step(action)
            
            loss = self.agent.step(state, action, reward, next_state, done)
            
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
            'final_prompt': info['prompt']
        }

    def _evaluate_episode(self, target_prompt: str) -> Dict:
        """Evaluate without training"""
        
        state = self.env.reset(target_prompt)
        total_reward = 0
        step = 0
        best_score = 0
        
        while True:
            action = self.agent.act(state, training=False)
            next_state, reward, done, info = self.env.step(action)
            
            state = next_state
            total_reward += reward
            step += 1
            best_score = max(best_score, info['score'])
            
            if done:
                break
        
        return {
            'target_prompt': target_prompt,
            'best_score': best_score,
            'total_reward': total_reward,
            'steps': step,
            'ultra_achieved': best_score >= self.env.ultra_target,
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
            # Convert to serializable format
            state_dict = asdict(self.training_state)
            state_dict['last_checkpoint_time'] = time.time()
            json.dump(state_dict, f, indent=2)
        
        # Save environment episode log
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
        """Generate comprehensive training report"""
        
        print(f"\n🎓 FINAL TRAINING REPORT")
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
        
        if len(self.training_state.episode_scores) >= 10:
            early_scores = self.training_state.episode_scores[:5]
            recent_scores = self.training_state.episode_scores[-5:]
            improvement = sum(recent_scores)/len(recent_scores) - sum(early_scores)/len(early_scores)
            print(f"   Learning Progress: {improvement:+.3f}")
        
        return {
            'total_episodes': total_episodes,
            'ultra_achievements': ultra_count,
            'ultra_rate': ultra_count / total_episodes if total_episodes > 0 else 0,
            'average_score': avg_score,
            'best_score': self.training_state.best_overall_score,
            'training_time_hours': training_time / 3600,
            'final_epsilon': self.agent.epsilon,
            'replay_buffer_size': len(self.agent.memory)
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
    """Test persistent RL optimizer"""
    
    print("🚀 PERSISTENT REINFORCEMENT LEARNING OPTIMIZER")
    print("💾 Features: Auto-checkpointing, Resume, Graceful interruption")
    print("=" * 80)
    
    # Check for existing checkpoints
    optimizer = PersistentRLOptimizer(ultra_target=0.96)
    checkpoints = optimizer.list_checkpoints()
    
    if checkpoints:
        print(f"📂 FOUND EXISTING CHECKPOINTS:")
        for i, checkpoint in enumerate(checkpoints, 1):
            print(f"   {i}. {checkpoint}")
        
        choice = input("\n🔄 Resume from checkpoint? Enter number (1-{}) or press ENTER for new training: ".format(len(checkpoints))).strip()
        
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
        "transparent glass sphere with reflections"
    ]
    
    # Start training with checkpoints
    results = optimizer.train_with_checkpoints(
        target_prompts=training_prompts,
        episodes_per_prompt=3,
        resume_from=resume_from
    )
    
    print(f"\n🎉 PERSISTENT RL TRAINING SESSION COMPLETE!")

if __name__ == "__main__":
    main() 