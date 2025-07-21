#!/usr/bin/env python3
"""
Fixed Reinforcement Learning Prompt Optimizer
=============================================
FIXES:
1. Proper reward function (positive for good scores)
2. Epsilon decay working correctly  
3. Batch learning actually happening
4. Smarter action space (replace vs add)
5. Transition from exploration to exploitation
6. Shorter, better prompts
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

Experience = namedtuple('Experience', ['state', 'action', 'reward', 'next_state', 'done'])

class FixedPromptEnvironment:
    """Fixed environment with proper rewards and actions"""
    
    def __init__(self, ultra_target: float = 0.96, checkpoint_dir: str = "rl_checkpoints_fixed"):
        self.ultra_target = ultra_target
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(exist_ok=True)
        
        self.target_prompt = ""
        self.current_prompt = ""
        self.validation_history = []
        self.step_count = 0
        self.max_steps = 8  # Shorter episodes
        
        # FIXED: Smarter action space
        self.action_space = self._define_smart_action_space()
        self.state_size = 20  # Simplified state
        self.action_size = len(self.action_space)
        
        # Session tracking
        self.session_id = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        
        print(f"🎮 FIXED RL ENVIRONMENT INITIALIZED")
        print(f"   🎯 Target Score: {ultra_target}")
        print(f"   🎬 Action Space Size: {self.action_size}")

    def _define_smart_action_space(self) -> List[Tuple]:
        """FIXED: Smart action space that replaces rather than just adds"""
        
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
        
        # Single descriptor improvements (REPLACE existing descriptors)
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
        
        # Simplification actions (when prompt gets too complex)
        simplify_actions = [
            ('SIMPLIFY', 'remove_duplicates', 'clean'),
            ('SIMPLIFY', 'keep_best_only', 'clean'),
            ('SIMPLIFY', 'ultra_minimal', 'clean')
        ]
        
        actions.extend(simplify_actions)
        
        return actions

    def reset(self, target_prompt: str) -> np.ndarray:
        """Reset with better initial prompt"""
        
        self.target_prompt = target_prompt
        # Start with basic but clean prompt
        self.current_prompt = f"wbgmsst, {target_prompt}, white background"
        self.validation_history = []
        self.step_count = 0
        
        initial_score = self._validate_prompt(self.current_prompt)
        self.validation_history.append(initial_score)
        
        print(f"🔄 ENVIRONMENT RESET")
        print(f"   🎯 Target: {target_prompt}")
        print(f"   📊 Initial Score: {initial_score:.3f}")
        
        return self._get_state()

    def step(self, action_idx: int) -> Tuple[np.ndarray, float, bool, Dict]:
        """FIXED: Better step function with proper rewards"""
        
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
        
        info = {
            'score': new_score,
            'prompt': self.current_prompt,
            'action_taken': action[0],
            'step': self.step_count,
            'improvement': new_score - old_score
        }
        
        return self._get_state(), reward, done, info

    def _apply_smart_action(self, action: Tuple) -> str:
        """FIXED: Smart action application that creates better prompts"""
        
        action_type, modifier, mode = action
        
        if action_type == 'APPLY_PATTERN':
            # Replace with proven pattern
            pattern = modifier.replace('{target}', self.target_prompt)
            return f"wbgmsst, {pattern}, white background"
        
        elif action_type in ['UPGRADE_AUTHORITY', 'UPGRADE_PROCESS', 'UPGRADE_QUALITY']:
            # Intelligent upgrade of existing prompt
            parts = self.current_prompt.split(', ')
            if len(parts) >= 3:  # wbgmsst, [middle], white background
                middle = parts[1]
                
                # Remove any existing similar descriptors before adding new one
                if 'aerospace' in middle or 'military' in middle or 'defense' in middle:
                    # Remove old authority terms
                    middle = re.sub(r'\b(aerospace-grade|military-spec|defense-grade|aviation-standard|laboratory-grade)\b\s*', '', middle)
                
                if 'precision' in middle or 'ultra' in middle or 'masterpiece' in middle:
                    # Remove old process terms for process upgrades
                    if action_type == 'UPGRADE_PROCESS':
                        middle = re.sub(r'\b(ultra-precision|precision-engineered|masterpiece-quality|ultra-detailed)\b\s*', '', middle)
                
                # Add new descriptor at the beginning
                middle = f"{modifier} {middle}".strip()
                parts[1] = middle
                
                return ', '.join(parts)
        
        elif action_type == 'SIMPLIFY':
            if modifier == 'remove_duplicates':
                # Remove duplicate words
                words = self.current_prompt.split()
                seen = set()
                cleaned_words = []
                for word in words:
                    if word not in seen:
                        cleaned_words.append(word)
                        seen.add(word)
                return ' '.join(cleaned_words)
            
            elif modifier == 'keep_best_only':
                # Keep only the target and best descriptors
                return f"wbgmsst, defense-grade ultra-precision {self.target_prompt}, ultra-high technical specification, white background"
            
            elif modifier == 'ultra_minimal':
                # Ultra minimal but effective
                return f"wbgmsst, aerospace-grade {self.target_prompt}, precision-engineered excellence, white background"
        
        return self.current_prompt

    def _validate_prompt(self, prompt: str) -> float:
        """Validate with error handling"""
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
        """FIXED: Proper reward function that makes sense"""
        
        # Base reward: Direct score improvement (this is the main signal!)
        score_improvement = new_score - old_score
        base_reward = score_improvement * 100  # Scale up improvements
        
        # Absolute score bonuses (reward good scores regardless of improvement)
        if new_score >= self.ultra_target:
            base_reward += 200  # Huge bonus for ultra achievement
        elif new_score >= 0.9:
            base_reward += 100  # Big bonus for excellent scores
        elif new_score >= 0.8:
            base_reward += 50   # Good bonus for high scores
        elif new_score >= 0.7:
            base_reward += 20   # Small bonus for decent scores
        
        # Prompt length penalty (encourage concise prompts)
        if len(self.current_prompt) > 120:
            base_reward -= 10
        elif len(self.current_prompt) > 100:
            base_reward -= 5
        
        # Penalty for very low scores (discourage bad actions)
        if new_score < 0.3:
            base_reward -= 20
        
        # Bonus for reaching new personal best
        if new_score > max(self.validation_history[:-1]) if len(self.validation_history) > 1 else 0:
            base_reward += 30
        
        return base_reward

    def _get_state(self) -> np.ndarray:
        """FIXED: Simplified but effective state representation"""
        
        state = np.zeros(self.state_size)
        
        # Last 3 scores (most important for learning trends)
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
        state[12] = len(self.current_prompt) / 150  # Normalized length
        
        # Target object type (helps with strategy selection)
        target_lower = self.target_prompt.lower()
        state[13] = 1.0 if any(word in target_lower for word in ["steel", "metal"]) else 0.0
        state[14] = 1.0 if any(word in target_lower for word in ["fabric", "cloth"]) else 0.0
        state[15] = 1.0 if any(word in target_lower for word in ["glass", "crystal"]) else 0.0
        
        # Performance indicators
        state[16] = 1.0 if self.validation_history[-1] >= 0.8 else 0.0  # High score indicator
        state[17] = 1.0 if self.validation_history[-1] >= self.ultra_target else 0.0  # Ultra indicator
        
        return state

class FixedDQN(nn.Module):
    """Simplified but effective DQN"""
    
    def __init__(self, state_size: int, action_size: int):
        super(FixedDQN, self).__init__()
        
        self.fc1 = nn.Linear(state_size, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, action_size)
        
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return x

class FixedDQNAgent:
    """FIXED: DQN Agent that actually learns"""
    
    def __init__(self, state_size: int, action_size: int, lr: float = 0.001):
        self.state_size = state_size
        self.action_size = action_size
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Networks
        self.q_network_local = FixedDQN(state_size, action_size).to(self.device)
        self.q_network_target = FixedDQN(state_size, action_size).to(self.device)
        self.optimizer = optim.Adam(self.q_network_local.parameters(), lr=lr)
        
        # Replay buffer
        self.memory = deque(maxlen=5000)  # Smaller buffer for faster learning
        
        # FIXED: Proper hyperparameters
        self.batch_size = 32
        self.gamma = 0.95        # Discount factor
        self.tau = 0.005         # Soft update
        self.update_every = 2    # Learn more frequently
        
        # FIXED: Proper epsilon schedule
        self.epsilon = 0.9       # Start high but not 1.0
        self.epsilon_min = 0.05  # Don't go to zero
        self.epsilon_decay = 0.98  # Faster decay
        
        # Learning tracking
        self.step_count = 0
        self.learn_count = 0
        self.losses = []
        
        print(f"🤖 FIXED DQN AGENT INITIALIZED")
        print(f"   🎲 Initial Epsilon: {self.epsilon}")
        print(f"   📚 Batch Size: {self.batch_size}")

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
            
            print(f"   🧠 EXPLOITATION: Action {action} (Q={max_q:.2f}, ε={self.epsilon:.3f})")
            return action

    def step(self, state: np.ndarray, action: int, reward: float, 
             next_state: np.ndarray, done: bool):
        """FIXED: Proper learning with epsilon decay"""
        
        # Store experience
        experience = Experience(state, action, reward, next_state, done)
        self.memory.append(experience)
        
        self.step_count += 1
        
        # Learn more frequently and ensure we have enough experiences
        if self.step_count % self.update_every == 0 and len(self.memory) >= self.batch_size:
            experiences = random.sample(self.memory, self.batch_size)
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

class FixedRLOptimizer:
    """FIXED: Main RL optimizer that actually works"""
    
    def __init__(self, ultra_target: float = 0.96):
        self.env = FixedPromptEnvironment(ultra_target)
        self.agent = FixedDQNAgent(
            state_size=self.env.state_size,
            action_size=self.env.action_size
        )
        
        self.episode_rewards = []
        self.episode_scores = []
        self.ultra_achievements = []
        
        print(f"🚀 FIXED RL PROMPT OPTIMIZER")
        print(f"   ✅ Proper reward function")
        print(f"   ✅ Epsilon decay working")
        print(f"   ✅ Batch learning enabled")
        print(f"   ✅ Smart action space")

    def train_episode(self, target_prompt: str, episode_num: int) -> Dict:
        """Train one episode with proper learning"""
        
        print(f"\n🎓 EPISODE {episode_num}")
        print("=" * 60)
        
        state = self.env.reset(target_prompt)
        total_reward = 0
        steps = 0
        best_score = 0
        
        while True:
            # Agent acts
            action = self.agent.act(state, training=True)
            
            # Environment responds
            next_state, reward, done, info = self.env.step(action)
            
            # Agent learns
            loss = self.agent.step(state, action, reward, next_state, done)
            
            # Update tracking
            state = next_state
            total_reward += reward
            steps += 1
            best_score = max(best_score, info['score'])
            
            if done:
                break
        
        # Episode summary
        ultra_achieved = best_score >= self.env.ultra_target
        
        self.episode_rewards.append(total_reward)
        self.episode_scores.append(best_score)
        self.ultra_achievements.append(ultra_achieved)
        
        print(f"\n📊 EPISODE {episode_num} RESULTS:")
        print(f"   🏆 Best Score: {best_score:.3f}")
        print(f"   🎁 Total Reward: {total_reward:.1f}")
        print(f"   🎬 Steps: {steps}")
        print(f"   🎉 Ultra: {'YES' if ultra_achieved else 'NO'}")
        print(f"   🎲 Epsilon: {self.agent.epsilon:.3f}")
        print(f"   📚 Learned: {self.agent.learn_count} times")
        
        return {
            'episode': episode_num,
            'best_score': best_score,
            'total_reward': total_reward,
            'steps': steps,
            'ultra_achieved': ultra_achieved,
            'epsilon': self.agent.epsilon,
            'learn_count': self.agent.learn_count
        }

    def train_session(self, target_prompts: List[str], episodes_per_prompt: int = 5):
        """Train on multiple prompts"""
        
        print(f"🎓 FIXED RL TRAINING SESSION")
        print(f"📝 Prompts: {len(target_prompts)}")
        print("=" * 80)
        
        all_results = []
        
        for prompt_idx, prompt in enumerate(target_prompts, 1):
            print(f"\n🎯 TRAINING ON PROMPT {prompt_idx}/{len(target_prompts)}: '{prompt}'")
            
            for episode in range(1, episodes_per_prompt + 1):
                episode_num = (prompt_idx - 1) * episodes_per_prompt + episode
                
                result = self.train_episode(prompt, episode_num)
                all_results.append(result)
                
                # Pause after each episode
                if episode < episodes_per_prompt:
                    user_input = input(f"\n➡️ Continue to episode {episode + 1}? (ENTER/q): ").strip()
                    if user_input.lower() == 'q':
                        break
        
        # Final analysis
        self._show_learning_analysis(all_results)
        
        return all_results

    def _show_learning_analysis(self, results: List[Dict]):
        """Show learning progress analysis"""
        
        print(f"\n🧠 LEARNING ANALYSIS")
        print("=" * 60)
        
        scores = [r['best_score'] for r in results]
        rewards = [r['total_reward'] for r in results]
        epsilons = [r['epsilon'] for r in results]
        
        print(f"📊 PERFORMANCE METRICS:")
        print(f"   Episodes: {len(results)}")
        print(f"   Ultra Rate: {sum(self.ultra_achievements)}/{len(results)} ({sum(self.ultra_achievements)/len(results)*100:.1f}%)")
        print(f"   Avg Score: {sum(scores)/len(scores):.3f}")
        print(f"   Best Score: {max(scores):.3f}")
        print(f"   Final Epsilon: {epsilons[-1]:.3f}")
        print(f"   Total Learning: {self.agent.learn_count} updates")
        
        # Learning progress
        if len(scores) >= 6:
            early_avg = sum(scores[:3]) / 3
            late_avg = sum(scores[-3:]) / 3
            improvement = late_avg - early_avg
            print(f"   Learning Progress: {improvement:+.3f} (early: {early_avg:.3f} → late: {late_avg:.3f})")
        
        # Show trend
        print(f"\n📈 SCORE PROGRESSION:")
        for i, result in enumerate(results, 1):
            status = "🎉" if result['ultra_achieved'] else "📈" if result['best_score'] >= 0.8 else "📊"
            print(f"   Episode {i:2d}: {status} {result['best_score']:.3f} (ε={result['epsilon']:.2f})")

def main():
    """Test the fixed RL optimizer"""
    
    print("🚀 TESTING FIXED RL OPTIMIZER")
    print("🎯 Fixes: Proper rewards, epsilon decay, batch learning, smart actions")
    print("=" * 80)
    
    optimizer = FixedRLOptimizer(ultra_target=0.96)
    
    test_prompts = [
        "hexagonal prism steel structure",
        "elegant silk fabric draping"
    ]
    
    results = optimizer.train_session(
        target_prompts=test_prompts,
        episodes_per_prompt=5
    )
    
    print(f"\n🎉 FIXED RL TRAINING COMPLETE!")
    print("🧠 Agent should now show proper learning behavior!")

if __name__ == "__main__":
    main() 