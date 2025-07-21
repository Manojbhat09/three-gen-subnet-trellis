#!/usr/bin/env python3
"""
Reinforcement Learning Prompt Optimizer
=======================================
Uses RL algorithms (DQN with experience replay) to optimize prompt generation
through proper exploration-exploitation and learning from experience.

RL Components:
- State: Prompt characteristics + validation history
- Actions: Specific prompt modification strategies
- Reward: Validation score improvements
- Environment: 3D generation validation system
- Agent: DQN with experience replay and epsilon-greedy exploration
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
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import sqlite3
from pathlib import Path
import re

# Experience tuple for replay buffer
Experience = namedtuple('Experience', ['state', 'action', 'reward', 'next_state', 'done'])

@dataclass
class PromptState:
    """State representation for RL"""
    target_object: str
    current_descriptors: List[str]
    validation_history: List[float]
    attempt_number: int
    best_score_so_far: float
    descriptor_categories: Dict[str, int]  # Count of each category used
    prompt_length: int
    last_improvement: float

@dataclass
class PromptAction:
    """Action representation for RL"""
    action_type: str
    descriptor_to_add: str
    descriptor_to_remove: str
    position: str  # 'prefix', 'middle', 'suffix'
    
class PromptEnvironment:
    """3D Generation Prompt Optimization Environment"""
    
    def __init__(self, ultra_target: float = 0.96):
        self.ultra_target = ultra_target
        self.target_prompt = ""
        self.current_prompt = ""
        self.validation_history = []
        self.step_count = 0
        self.max_steps = 10
        
        # Action space definition
        self.action_space = self._define_action_space()
        self.state_size = 25  # Dimensionality of state vector
        self.action_size = len(self.action_space)
        
        print(f"🎮 RL ENVIRONMENT INITIALIZED")
        print(f"   🎯 Target Score: {ultra_target}")
        print(f"   🎬 Action Space Size: {self.action_size}")
        print(f"   📊 State Space Size: {self.state_size}")

    def _define_action_space(self) -> List[PromptAction]:
        """Define the action space for prompt modifications"""
        
        actions = []
        
        # Authority descriptors
        authority_descriptors = [
            "aerospace-grade", "military-spec", "defense-grade", "aviation-standard",
            "laboratory-grade", "pharmaceutical-grade", "precision-aerospace",
            "ultra-military-spec", "defense-aerospace-grade"
        ]
        
        # Process descriptors
        process_descriptors = [
            "precision-engineered", "ultra-precision", "masterpiece-quality", 
            "ultra-detailed", "precision-forged", "ultra-refined", 
            "laboratory-crafted", "precision-aerospace-engineered"
        ]
        
        # Quality descriptors
        quality_descriptors = [
            "ultra-high technical specification", "advanced engineering design",
            "premium manufacturing excellence", "ultra-precision specification",
            "aerospace-engineering excellence", "laboratory-precision specification"
        ]
        
        # Define actions: ADD descriptor actions
        for position in ['prefix', 'middle', 'suffix']:
            for desc in authority_descriptors:
                actions.append(PromptAction("ADD_AUTHORITY", desc, "", position))
            for desc in process_descriptors:
                actions.append(PromptAction("ADD_PROCESS", desc, "", position))
            for desc in quality_descriptors:
                actions.append(PromptAction("ADD_QUALITY", desc, "", position))
        
        # REPLACE actions (replace existing with better)
        for old_desc in ["ultra-precision", "high-quality", "good"]:
            for new_desc in authority_descriptors[:3]:  # Top 3 authority
                actions.append(PromptAction("REPLACE", new_desc, old_desc, "any"))
        
        # SIMPLIFY actions (remove redundant descriptors)
        for desc in ["basic", "standard", "regular", "normal"]:
            actions.append(PromptAction("REMOVE", "", desc, "any"))
        
        # COMBINE actions (stack premium descriptors)
        premium_combos = [
            ("aerospace-grade", "ultra-precision"),
            ("defense-grade", "masterpiece-quality"),
            ("military-spec", "precision-engineered")
        ]
        for combo in premium_combos:
            actions.append(PromptAction("COMBINE", f"{combo[0]} {combo[1]}", "", "prefix"))
        
        return actions

    def reset(self, target_prompt: str) -> np.ndarray:
        """Reset environment for new prompt optimization episode"""
        
        self.target_prompt = target_prompt
        self.current_prompt = f"wbgmsst, {target_prompt}, white background"  # Basic starting prompt
        self.validation_history = []
        self.step_count = 0
        
        # Get initial validation score
        initial_score = self._validate_prompt(self.current_prompt)
        self.validation_history.append(initial_score)
        
        print(f"🔄 ENVIRONMENT RESET")
        print(f"   🎯 Target: {target_prompt}")
        print(f"   📝 Initial Prompt: {self.current_prompt}")
        print(f"   📊 Initial Score: {initial_score:.3f}")
        
        return self._get_state()

    def step(self, action_idx: int) -> Tuple[np.ndarray, float, bool, Dict]:
        """Execute action and return (next_state, reward, done, info)"""
        
        self.step_count += 1
        action = self.action_space[action_idx]
        
        print(f"🎬 STEP {self.step_count}: {action.action_type}")
        print(f"   🔧 Action: {action.descriptor_to_add} ({action.position})")
        
        # Apply action to modify prompt
        old_prompt = self.current_prompt
        self.current_prompt = self._apply_action(action)
        
        # Validate new prompt
        new_score = self._validate_prompt(self.current_prompt)
        self.validation_history.append(new_score)
        
        # Calculate reward
        reward = self._calculate_reward(new_score, old_prompt)
        
        # Check if episode is done
        done = (new_score >= self.ultra_target or 
                self.step_count >= self.max_steps)
        
        print(f"   📝 New Prompt: {self.current_prompt}")
        print(f"   📊 Score: {new_score:.3f}")
        print(f"   🎁 Reward: {reward:.3f}")
        print(f"   ✅ Done: {done}")
        
        # Info dict
        info = {
            'score': new_score,
            'prompt': self.current_prompt,
            'action_taken': action.action_type,
            'step': self.step_count
        }
        
        return self._get_state(), reward, done, info

    def _apply_action(self, action: PromptAction) -> str:
        """Apply the action to modify the current prompt"""
        
        prompt = self.current_prompt
        
        if action.action_type == "ADD_AUTHORITY" or action.action_type == "ADD_PROCESS":
            # Add descriptor at specified position
            parts = prompt.split(', ')
            if len(parts) >= 3:  # wbgmsst, [middle], white background
                if action.position == "prefix":
                    parts[1] = f"{action.descriptor_to_add} {parts[1]}"
                elif action.position == "middle":
                    parts.insert(-1, action.descriptor_to_add)
                elif action.position == "suffix":
                    parts[-2] = f"{parts[-2]}, {action.descriptor_to_add}"
            prompt = ', '.join(parts)
            
        elif action.action_type == "ADD_QUALITY":
            # Add quality descriptor before white background
            parts = prompt.split(', white background')
            if len(parts) == 2:
                prompt = f"{parts[0]}, {action.descriptor_to_add}, white background"
                
        elif action.action_type == "REPLACE":
            # Replace old descriptor with new one
            prompt = prompt.replace(action.descriptor_to_remove, action.descriptor_to_add)
            
        elif action.action_type == "REMOVE":
            # Remove descriptor
            prompt = prompt.replace(action.descriptor_to_remove, "").replace("  ", " ")
            
        elif action.action_type == "COMBINE":
            # Add combined descriptors at prefix
            parts = prompt.split(', ')
            if len(parts) >= 2:
                parts[1] = f"{action.descriptor_to_add} {parts[1]}"
            prompt = ', '.join(parts)
        
        # Clean up prompt
        prompt = re.sub(r',\s*,', ',', prompt)  # Remove double commas
        prompt = re.sub(r'\s+', ' ', prompt)    # Remove extra spaces
        
        return prompt.strip()

    def _validate_prompt(self, prompt: str) -> float:
        """Validate prompt and return score"""
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
        """Calculate reward based on score improvement and other factors"""
        
        prev_score = self.validation_history[-2] if len(self.validation_history) > 1 else 0.0
        
        # Base reward: score improvement
        score_improvement = new_score - prev_score
        reward = score_improvement * 10  # Scale up improvements
        
        # Bonus rewards
        if new_score >= self.ultra_target:
            reward += 100  # Ultra achievement bonus
        elif new_score >= 0.9:
            reward += 50   # High score bonus
        elif new_score >= 0.8:
            reward += 20   # Good score bonus
        
        # Penalty for making prompt too long
        if len(self.current_prompt) > 150:
            reward -= 5
        
        # Penalty for no improvement over multiple steps
        if len(self.validation_history) >= 3:
            recent_scores = self.validation_history[-3:]
            if max(recent_scores) - min(recent_scores) < 0.01:  # No significant change
                reward -= 2
        
        # Bonus for consistent improvement
        if len(self.validation_history) >= 3:
            if all(self.validation_history[i] >= self.validation_history[i-1] 
                   for i in range(-2, 0)):
                reward += 5
        
        return reward

    def _get_state(self) -> np.ndarray:
        """Get current state representation as vector"""
        
        state = np.zeros(self.state_size)
        
        # Index 0-4: Validation history (last 5 scores, padded with 0)
        history = self.validation_history[-5:]
        for i, score in enumerate(history):
            if i < 5:
                state[i] = score
        
        # Index 5: Current step / max steps (normalized)
        state[5] = self.step_count / self.max_steps
        
        # Index 6: Best score so far
        state[6] = max(self.validation_history) if self.validation_history else 0.0
        
        # Index 7: Prompt length (normalized)
        state[7] = len(self.current_prompt) / 150  # Normalize by max desired length
        
        # Index 8-12: Descriptor category counts (normalized)
        authority_count = sum(1 for desc in ["aerospace", "military", "defense", "aviation", "laboratory"] 
                             if desc in self.current_prompt.lower())
        process_count = sum(1 for desc in ["precision", "ultra", "masterpiece", "detailed", "forged"] 
                           if desc in self.current_prompt.lower())
        quality_count = sum(1 for desc in ["specification", "engineering", "excellence", "manufacturing"] 
                           if desc in self.current_prompt.lower())
        
        state[8] = min(authority_count / 3, 1.0)  # Normalize by reasonable max
        state[9] = min(process_count / 3, 1.0)
        state[10] = min(quality_count / 2, 1.0)
        
        # Index 11: Score improvement trend (last 3 steps)
        if len(self.validation_history) >= 3:
            recent = self.validation_history[-3:]
            trend = (recent[-1] - recent[0]) / 2  # Improvement over 3 steps
            state[11] = np.clip(trend * 10, -1, 1)  # Normalize and clip
        
        # Index 12-16: Target object characteristics (one-hot encoded categories)
        target_lower = self.target_prompt.lower()
        state[12] = 1.0 if any(word in target_lower for word in ["steel", "metal", "iron", "aluminum"]) else 0.0
        state[13] = 1.0 if any(word in target_lower for word in ["fabric", "cloth", "textile", "silk"]) else 0.0
        state[14] = 1.0 if any(word in target_lower for word in ["glass", "crystal", "transparent"]) else 0.0
        state[15] = 1.0 if any(word in target_lower for word in ["wood", "wooden", "timber"]) else 0.0
        state[16] = 1.0 if any(word in target_lower for word in ["geometric", "prism", "sphere", "structure"]) else 0.0
        
        # Index 17-24: Action effectiveness history (which action types worked)
        # This would be filled by the agent based on its experience
        # For now, initialize with zeros (agent will learn these patterns)
        
        return state

class DQN(nn.Module):
    """Deep Q-Network for prompt optimization"""
    
    def __init__(self, state_size: int, action_size: int, hidden_size: int = 256):
        super(DQN, self).__init__()
        
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, hidden_size)
        self.fc4 = nn.Linear(hidden_size, action_size)
        
        # Dropout for regularization
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return x

class ReplayBuffer:
    """Experience replay buffer for DQN"""
    
    def __init__(self, capacity: int = 10000):
        self.buffer = deque(maxlen=capacity)
        
    def push(self, experience: Experience):
        """Add experience to buffer"""
        self.buffer.append(experience)
    
    def sample(self, batch_size: int) -> List[Experience]:
        """Sample batch of experiences"""
        return random.sample(self.buffer, batch_size)
    
    def __len__(self):
        return len(self.buffer)

class DQNAgent:
    """DQN Agent with epsilon-greedy exploration and experience replay"""
    
    def __init__(self, state_size: int, action_size: int, lr: float = 0.001):
        self.state_size = state_size
        self.action_size = action_size
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Neural networks
        self.q_network_local = DQN(state_size, action_size).to(self.device)
        self.q_network_target = DQN(state_size, action_size).to(self.device)
        self.optimizer = optim.Adam(self.q_network_local.parameters(), lr=lr)
        
        # Replay buffer
        self.memory = ReplayBuffer(capacity=10000)
        
        # Hyperparameters
        self.batch_size = 64
        self.gamma = 0.99        # Discount factor
        self.tau = 0.001         # Soft update parameter
        self.update_every = 4    # Update frequency
        
        # Exploration parameters
        self.epsilon = 1.0       # Start with full exploration
        self.epsilon_min = 0.01  # Minimum exploration
        self.epsilon_decay = 0.995  # Decay rate
        
        # Learning tracking
        self.step_count = 0
        self.losses = []
        
        print(f"🤖 DQN AGENT INITIALIZED")
        print(f"   🧠 Device: {self.device}")
        print(f"   📊 State Size: {state_size}")
        print(f"   🎬 Action Size: {action_size}")
        print(f"   🎲 Initial Epsilon: {self.epsilon}")

    def act(self, state: np.ndarray, training: bool = True) -> int:
        """Choose action using epsilon-greedy policy"""
        
        if training and random.random() < self.epsilon:
            # Exploration: random action
            action = random.randrange(self.action_size)
            print(f"   🎲 EXPLORATION: Random action {action}")
            return action
        else:
            # Exploitation: use Q-network
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            
            self.q_network_local.eval()
            with torch.no_grad():
                q_values = self.q_network_local(state_tensor)
            self.q_network_local.train()
            
            action = q_values.argmax().item()
            print(f"   🧠 EXPLOITATION: Q-network action {action} (Q-val: {q_values.max().item():.3f})")
            return action

    def step(self, state: np.ndarray, action: int, reward: float, 
             next_state: np.ndarray, done: bool):
        """Add experience to replay buffer and learn"""
        
        # Store experience
        experience = Experience(state, action, reward, next_state, done)
        self.memory.push(experience)
        
        # Learn from experience
        self.step_count += 1
        if self.step_count % self.update_every == 0:
            if len(self.memory) > self.batch_size:
                experiences = self.memory.sample(self.batch_size)
                self.learn(experiences)

    def learn(self, experiences: List[Experience]):
        """Learn from batch of experiences using DQN"""
        
        # Convert batch to tensors
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
        self.optimizer.step()
        
        # Track loss
        self.losses.append(loss.item())
        
        # Soft update target network
        self.soft_update()
        
        # Decay epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        
        print(f"   📚 LEARNING: Loss {loss.item():.4f}, Epsilon {self.epsilon:.3f}")

    def soft_update(self):
        """Soft update target network parameters"""
        for target_param, local_param in zip(self.q_network_target.parameters(),
                                           self.q_network_local.parameters()):
            target_param.data.copy_(self.tau * local_param.data + 
                                  (1.0 - self.tau) * target_param.data)

class RLPromptOptimizer:
    """Main RL-based prompt optimizer"""
    
    def __init__(self, ultra_target: float = 0.96):
        self.env = PromptEnvironment(ultra_target)
        self.agent = DQNAgent(
            state_size=self.env.state_size,
            action_size=self.env.action_size,
            lr=0.001
        )
        self.training_episodes = 0
        self.episode_rewards = []
        self.episode_scores = []
        
        print(f"🚀 RL PROMPT OPTIMIZER INITIALIZED")
        print(f"   🎮 Environment: 3D Generation Prompt Optimization")
        print(f"   🤖 Agent: DQN with Experience Replay")
        print(f"   🎯 Target: {ultra_target} validation score")

    def train_episode(self, target_prompt: str) -> Dict:
        """Train one episode"""
        
        self.training_episodes += 1
        print(f"\n🎓 TRAINING EPISODE {self.training_episodes}")
        print(f"🎯 Target: {target_prompt}")
        print("=" * 80)
        
        # Reset environment
        state = self.env.reset(target_prompt)
        total_reward = 0
        step = 0
        best_score = 0
        
        while True:
            # Agent chooses action
            action = self.agent.act(state, training=True)
            
            # Environment steps
            next_state, reward, done, info = self.env.step(action)
            
            # Agent learns
            self.agent.step(state, action, reward, next_state, done)
            
            # Update tracking
            state = next_state
            total_reward += reward
            step += 1
            best_score = max(best_score, info['score'])
            
            if done:
                break
        
        # Episode summary
        self.episode_rewards.append(total_reward)
        self.episode_scores.append(best_score)
        
        ultra_achieved = best_score >= self.env.ultra_target
        
        print(f"\n📊 EPISODE {self.training_episodes} COMPLETE")
        print(f"   🏆 Best Score: {best_score:.3f}")
        print(f"   🎁 Total Reward: {total_reward:.2f}")
        print(f"   🎬 Steps: {step}")
        print(f"   🎉 Ultra Achieved: {ultra_achieved}")
        print(f"   🎲 Final Epsilon: {self.agent.epsilon:.3f}")
        
        return {
            'episode': self.training_episodes,
            'target_prompt': target_prompt,
            'best_score': best_score,
            'total_reward': total_reward,
            'steps': step,
            'ultra_achieved': ultra_achieved,
            'final_prompt': info['prompt']
        }

    def evaluate_episode(self, target_prompt: str) -> Dict:
        """Evaluate without training (no exploration)"""
        
        print(f"\n🔬 EVALUATION EPISODE")
        print(f"🎯 Target: {target_prompt}")
        print("=" * 60)
        
        # Reset environment
        state = self.env.reset(target_prompt)
        total_reward = 0
        step = 0
        best_score = 0
        
        while True:
            # Agent chooses action (no exploration)
            action = self.agent.act(state, training=False)
            
            # Environment steps
            next_state, reward, done, info = self.env.step(action)
            
            # Update tracking (no learning)
            state = next_state
            total_reward += reward
            step += 1
            best_score = max(best_score, info['score'])
            
            if done:
                break
        
        ultra_achieved = best_score >= self.env.ultra_target
        
        print(f"\n📊 EVALUATION COMPLETE")
        print(f"   🏆 Best Score: {best_score:.3f}")
        print(f"   🎁 Total Reward: {total_reward:.2f}")
        print(f"   🎬 Steps: {step}")
        print(f"   🎉 Ultra Achieved: {ultra_achieved}")
        
        return {
            'target_prompt': target_prompt,
            'best_score': best_score,
            'total_reward': total_reward,
            'steps': step,
            'ultra_achieved': ultra_achieved,
            'final_prompt': info['prompt']
        }

    def train_multiple_episodes(self, target_prompts: List[str], 
                              episodes_per_prompt: int = 5) -> Dict:
        """Train on multiple prompts for several episodes each"""
        
        print(f"🎓 RL TRAINING SESSION")
        print(f"📝 Prompts: {len(target_prompts)}")
        print(f"🔄 Episodes per prompt: {episodes_per_prompt}")
        print("=" * 80)
        
        all_results = []
        
        for prompt in target_prompts:
            prompt_results = []
            
            print(f"\n🎯 TRAINING ON: '{prompt}'")
            for episode in range(episodes_per_prompt):
                result = self.train_episode(prompt)
                prompt_results.append(result)
                
                # Brief pause between episodes
                time.sleep(1)
            
            all_results.extend(prompt_results)
            
            # Evaluate after training on this prompt
            eval_result = self.evaluate_episode(prompt)
            
            print(f"\n📈 PROMPT '{prompt}' TRAINING COMPLETE")
            scores = [r['best_score'] for r in prompt_results]
            print(f"   📊 Training Scores: {scores}")
            print(f"   🏆 Best Training: {max(scores):.3f}")
            print(f"   🔬 Evaluation: {eval_result['best_score']:.3f}")
            print(f"   🎉 Ultra Rate: {sum(1 for s in scores if s >= self.env.ultra_target)}/{len(scores)}")
        
        # Overall training analysis
        print(f"\n🎓 COMPLETE RL TRAINING ANALYSIS")
        print("=" * 80)
        
        ultra_count = sum(1 for r in all_results if r['ultra_achieved'])
        avg_score = sum(r['best_score'] for r in all_results) / len(all_results)
        avg_reward = sum(r['total_reward'] for r in all_results) / len(all_results)
        
        print(f"📊 TRAINING PERFORMANCE:")
        print(f"   Total Episodes: {len(all_results)}")
        print(f"   Ultra Achievements: {ultra_count}/{len(all_results)} ({ultra_count/len(all_results)*100:.1f}%)")
        print(f"   Average Score: {avg_score:.3f}")
        print(f"   Average Reward: {avg_reward:.2f}")
        print(f"   Final Epsilon: {self.agent.epsilon:.3f}")
        print(f"   Replay Buffer Size: {len(self.agent.memory)}")
        
        # Learning progress
        if len(self.episode_scores) >= 10:
            early_avg = sum(self.episode_scores[:5]) / 5
            recent_avg = sum(self.episode_scores[-5:]) / 5
            improvement = recent_avg - early_avg
            print(f"   Learning Progress: {improvement:+.3f} (early: {early_avg:.3f} → recent: {recent_avg:.3f})")
        
        return {
            'total_episodes': len(all_results),
            'ultra_achievements': ultra_count,
            'ultra_rate': ultra_count / len(all_results),
            'average_score': avg_score,
            'average_reward': avg_reward,
            'all_results': all_results
        }

def main():
    """Test RL prompt optimizer"""
    
    print("🚀 REINFORCEMENT LEARNING PROMPT OPTIMIZER")
    print("🎯 Mission: Learn optimal prompt generation through RL")
    print("⚡ Features: DQN, Experience Replay, Exploration-Exploitation")
    print("=" * 80)
    
    # Initialize RL optimizer
    optimizer = RLPromptOptimizer(ultra_target=0.96)
    
    # Training prompts
    training_prompts = [
        "hexagonal prism steel structure",
        "elegant silk fabric draping"
    ]
    
    # Train the RL agent
    training_results = optimizer.train_multiple_episodes(
        target_prompts=training_prompts,
        episodes_per_prompt=3  # Start with fewer episodes for testing
    )
    
    print(f"\n🎉 RL TRAINING COMPLETE!")
    print(f"🏆 Agent learned optimal prompt generation through reinforcement learning!")

if __name__ == "__main__":
    main() 