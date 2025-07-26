#!/usr/bin/env python3
"""
RL + LLaMA Optimizer v3.0 - INTELLIGENT PROMPT GENERATION
========================================================
Revolutionary approach that combines:
✅ LLaMA 3.2 as the PROMPT GENERATOR (not template selector)
✅ RL environment that learns from ACTUAL generation scores
✅ Custom prompt writing instead of rigid templates
✅ Real feedback loop with generation server validation
✅ Continuous learning from real-world performance

The RL agent learns WHEN and HOW to instruct LLaMA to generate better prompts.
LLaMA writes custom prompts based on RL agent's learned strategies.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import random
import requests
import json
import time
import hashlib
from collections import deque, namedtuple
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, asdict
from pathlib import Path
import pickle
import signal
import sys

Experience = namedtuple('Experience', ['state', 'action', 'reward', 'next_state', 'done'])

@dataclass
class GenerationResult:
    """Result from actual 3D generation"""
    prompt_used: str
    generation_successful: bool
    validation_score: float
    generation_time: float
    file_size_bytes: int
    error_message: Optional[str] = None

@dataclass
class LLaMAInstruction:
    """Instruction for LLaMA prompt generation"""
    strategy_type: str
    creativity_level: float  # 0.1-1.0
    technical_focus: str     # material, shape, quality, context
    style_directive: str     # descriptive instruction
    length_target: str       # short, medium, long
    risk_tolerance: str      # conservative, balanced, aggressive

class LLaMAPromptGenerator:
    """LLaMA 3.2 powered intelligent prompt generator"""
    
    def __init__(self, ollama_url: str = "http://localhost:11434"):
        self.ollama_url = ollama_url
        self.model = "llama3.2:3b"
        self.generation_history = []
        
        # Learning from feedback
        self.successful_patterns = []
        self.failed_patterns = []
        
        self._test_connection()
        print(f"🧠 LLaMA PROMPT GENERATOR INITIALIZED")
    
    def _test_connection(self):
        """Test LLaMA connection"""
        try:
            response = requests.get(f"{self.ollama_url}/api/tags", timeout=5)
            if response.status_code == 200:
                print(f"   ✅ LLaMA 3.2 Connected")
            else:
                raise Exception(f"Connection failed: {response.status_code}")
        except Exception as e:
            raise Exception(f"❌ LLaMA 3.2 unavailable: {e}")
    
    def generate_prompt(self, original_prompt: str, instruction: LLaMAInstruction, seed: int = 42) -> str:
        """Generate custom optimized prompt using LLaMA intelligence with reproducible seed"""
        
        # Build intelligent system prompt based on learned patterns
        system_prompt = self._build_system_prompt(instruction)
        
        # Build user prompt with context and requirements
        user_prompt = self._build_user_prompt(original_prompt, instruction)
        
        # Query LLaMA for custom prompt with seed
        try:
            response = self._query_llama(system_prompt, user_prompt, instruction.creativity_level, seed)
            optimized_prompt = self._extract_prompt_from_response(response, original_prompt)
            
            # Track generation
            self.generation_history.append({
                'original': original_prompt,
                'optimized': optimized_prompt,
                'instruction': asdict(instruction),
                'seed': seed,
                'timestamp': time.time()
            })
            
            return optimized_prompt
            
        except Exception as e:
            print(f"❌ LLaMA generation failed: {e}")
            return self._fallback_optimization(original_prompt, instruction)
    
    def _build_system_prompt(self, instruction: LLaMAInstruction) -> str:
        """Build intelligent system prompt based on learned patterns"""
        
        base_prompt = f"""You are an expert 3D prompt optimizer with deep learning from real validation feedback.

MISSION: Generate a custom optimized prompt for 3D model generation that will score 0.9+ on validation.

STRATEGY: {instruction.strategy_type}
CREATIVITY: {instruction.creativity_level:.1f}/1.0 (higher = more innovative approaches)
FOCUS: {instruction.technical_focus} optimization priority
STYLE: {instruction.style_directive}
TARGET LENGTH: {instruction.length_target}
RISK LEVEL: {instruction.risk_tolerance}

CRITICAL REQUIREMENTS:
1. MUST start with "wbgmsst,"
2. MUST end with ", white background"
3. PRESERVE the core object from the original prompt
4. CREATE a custom solution, not template application
5. FOCUS on {instruction.technical_focus} enhancements
6. BALANCE technical precision with creative specificity

PROVEN SUCCESS PATTERNS:"""
        
        # Add learned successful patterns
        if self.successful_patterns:
            base_prompt += "\nSUCCESSFUL APPROACHES THAT SCORED 0.8+:"
            for pattern in self.successful_patterns[-5:]:  # Last 5 successful patterns
                base_prompt += f"\n- Score {pattern['score']:.3f}: {pattern['approach']}"
        
        # Add failure patterns to avoid
        if self.failed_patterns:
            base_prompt += "\n\nAVOID THESE FAILED APPROACHES:"
            for pattern in self.failed_patterns[-3:]:  # Last 3 failures
                base_prompt += f"\n- Score {pattern['score']:.3f}: {pattern['approach']} (FAILED)"
        
        base_prompt += f"""

RESPONSE FORMAT:
ANALYSIS: [Brief analysis of original prompt and optimization opportunities]
STRATEGY: [Your chosen approach for this specific prompt]
OPTIMIZED: [The complete optimized prompt - ready to use]
REASONING: [Why you expect this to score 0.9+]
CONFIDENCE: [0.1-1.0 your confidence in success]"""

        return base_prompt
    
    def _build_user_prompt(self, original: str, instruction: LLaMAInstruction) -> str:
        """Build contextual user prompt"""
        
        return f"""OPTIMIZE THIS PROMPT:

Original: "{original}"

OPTIMIZATION REQUIREMENTS:
- Strategy: {instruction.strategy_type}
- Focus on: {instruction.technical_focus}
- Style: {instruction.style_directive}
- Creativity level: {instruction.creativity_level:.1f}/1.0
- Risk tolerance: {instruction.risk_tolerance}

Generate a completely custom optimized prompt that addresses the specific characteristics of this object.
Think about what makes THIS particular object unique and how to enhance those qualities for 3D generation.

PROVIDE YOUR OPTIMIZATION:"""
    
    def _query_llama(self, system_prompt: str, user_prompt: str, creativity: float, seed: int = 42) -> str:
        """Query LLaMA with dynamic creativity and reproducible seed"""
        
        # Adjust temperature based on creativity level
        temperature = 0.3 + (creativity * 0.5)  # 0.3-0.8 range
        
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
                "num_predict": 300,
                "seed": seed  # Add seed for reproducible generation
            }
        }
        
        response = requests.post(f"{self.ollama_url}/api/chat", json=data, timeout=30)
        response.raise_for_status()
        
        return response.json()["message"]["content"].strip()
    
    def _extract_prompt_from_response(self, response: str, original: str) -> str:
        """Extract optimized prompt from LLaMA response"""
        
        # Look for OPTIMIZED: section
        lines = response.split('\n')
        optimized_prompt = None
        
        for line in lines:
            if line.strip().startswith('OPTIMIZED:'):
                optimized_prompt = line.split('OPTIMIZED:', 1)[1].strip()
                break
        
        # Fallback: look for any line with wbgmsst
        if not optimized_prompt:
            for line in lines:
                if 'wbgmsst' in line.lower():
                    optimized_prompt = line.strip()
                    break
        
        # Clean up the prompt
        if optimized_prompt:
            optimized_prompt = optimized_prompt.replace('"', '').strip()
            
            # Ensure proper format
            if not optimized_prompt.startswith('wbgmsst'):
                optimized_prompt = f"wbgmsst, {optimized_prompt}"
            if not optimized_prompt.endswith('white background'):
                if optimized_prompt.endswith(','):
                    optimized_prompt += " white background"
                else:
                    optimized_prompt += ", white background"
            
            return optimized_prompt
        
        # Fallback if extraction fails
        return self._fallback_optimization(original, None)
    
    def _fallback_optimization(self, original: str, instruction: Optional[LLaMAInstruction]) -> str:
        """Fallback optimization if LLaMA fails"""
        if instruction and instruction.technical_focus == 'material':
            return f"wbgmsst, precision-engineered {original}, ultra-high material specification, white background"
        elif instruction and instruction.technical_focus == 'quality':
            return f"wbgmsst, masterpiece-quality {original}, premium craftsmanship excellence, white background"
        else:
            return f"wbgmsst, detailed {original}, professional 3D rendering, white background"
    
    def learn_from_feedback(self, original: str, optimized: str, validation_score: float):
        """Learn from actual validation feedback"""
        
        if validation_score >= 0.8:
            # Extract approach pattern
            approach = self._extract_approach_pattern(original, optimized)
            self.successful_patterns.append({
                'original': original,
                'optimized': optimized,
                'score': validation_score,
                'approach': approach,
                'timestamp': time.time()
            })
            print(f"🧠 LLaMA learned successful pattern: {approach} (Score: {validation_score:.3f})")
            
            # Keep only recent successes
            self.successful_patterns = self.successful_patterns[-10:]
            
        elif validation_score < 0.5:
            # Learn from failures
            approach = self._extract_approach_pattern(original, optimized)
            self.failed_patterns.append({
                'original': original,
                'optimized': optimized,
                'score': validation_score,
                'approach': approach,
                'timestamp': time.time()
            })
            print(f"⚠️ LLaMA learned failed pattern: {approach} (Score: {validation_score:.3f})")
            
            # Keep only recent failures
            self.failed_patterns = self.failed_patterns[-5:]
    
    def _extract_approach_pattern(self, original: str, optimized: str) -> str:
        """Extract the key approach/pattern from the optimization"""
        
        # Remove common parts to see what was added
        original_words = set(original.lower().split())
        optimized_words = set(optimized.lower().split())
        
        added_words = optimized_words - original_words
        key_additions = [word for word in added_words if len(word) > 3 and word not in ['wbgmsst', 'white', 'background']]
        
        if key_additions:
            return f"Added: {', '.join(key_additions[:3])}"
        else:
            return "Structural enhancement"

class RLLLaMAEnvironment:
    """RL Environment that uses LLaMA for prompt generation and learns from real validation"""
    
    def __init__(self, generation_server_url: str = "http://localhost:8096", 
                 validation_server_url: str = "http://localhost:10006"):
        
        self.generation_server_url = generation_server_url
        self.validation_server_url = validation_server_url
        
        # LLaMA generator
        self.llama_generator = LLaMAPromptGenerator()
        
        # RL state management
        self.current_original_prompt = ""
        self.current_optimized_prompt = ""
        self.validation_history = []
        self.step_count = 0
        self.max_steps = 5  # Max optimization attempts per prompt
        
        # Learning tracking
        self.episode_results = []
        self.best_score_achieved = 0.0
        
        # State representation
        self.state_size = 20
        
        print(f"🎮 RL-LLaMA ENVIRONMENT INITIALIZED")
        print(f"   🎨 Generation Server: {generation_server_url}")
        print(f"   📊 Validation Server: {validation_server_url}")
        
        self._test_servers()
    
    def _test_servers(self):
        """Test server connections"""
        try:
            # Test generation server
            response = requests.get(f"{self.generation_server_url}/health/", timeout=5)
            if response.status_code == 200:
                print(f"   ✅ Generation Server: Connected")
            else:
                print(f"   ⚠️ Generation Server: HTTP {response.status_code}")
        except Exception as e:
            print(f"   ❌ Generation Server: {e}")
        
        try:
            # Test validation server  
            response = requests.get(f"{self.validation_server_url}/", timeout=5)
            print(f"   ✅ Validation Server: Available")
        except Exception as e:
            print(f"   ❌ Validation Server: {e}")
    
    def reset(self, original_prompt: str) -> np.ndarray:
        """Reset environment for new prompt optimization"""
        
        self.current_original_prompt = original_prompt
        self.current_optimized_prompt = original_prompt
        self.validation_history = []
        self.step_count = 0
        
        # Get baseline score
        print(f"🔄 ENVIRONMENT RESET")
        print(f"   🎯 Original Prompt: {original_prompt}")
        
        baseline_result = self._generate_and_validate(original_prompt)
        self.validation_history.append(baseline_result)
        
        print(f"   📊 Baseline Score: {baseline_result.validation_score:.3f}")
        
        return self._get_state()
    
    def step(self, action_idx: int) -> Tuple[np.ndarray, float, bool, Dict]:
        """Execute RL action: instruct LLaMA to generate optimized prompt"""
        
        self.step_count += 1
        
        # Convert action to LLaMA instruction
        instruction = self._action_to_instruction(action_idx)
        
        print(f"🎬 STEP {self.step_count}: {instruction.strategy_type}")
        print(f"   🎯 Focus: {instruction.technical_focus}")
        print(f"   🎨 Creativity: {instruction.creativity_level:.2f}")
        print(f"   🎭 Style: {instruction.style_directive}")
        
        # Get LLaMA to generate optimized prompt
        start_time = time.time()
        optimized_prompt = self.llama_generator.generate_prompt(
            self.current_original_prompt, 
            instruction
        )
        generation_time = time.time() - start_time
        
        print(f"   📝 Generated: {optimized_prompt}")
        print(f"   ⏱️ Generation Time: {generation_time:.2f}s")
        
        # Test with actual 3D generation and validation
        result = self._generate_and_validate(optimized_prompt)
        self.validation_history.append(result)
        
        # Calculate reward based on actual performance
        reward = self._calculate_reward(result)
        
        # Update current prompt if this was better
        old_score = self.validation_history[-2].validation_score if len(self.validation_history) > 1 else 0.0
        if result.validation_score > old_score:
            self.current_optimized_prompt = optimized_prompt
            self.best_score_achieved = max(self.best_score_achieved, result.validation_score)
        
        # Let LLaMA learn from this feedback
        self.llama_generator.learn_from_feedback(
            self.current_original_prompt,
            optimized_prompt,
            result.validation_score
        )
        
        # Check if done
        done = (result.validation_score >= 0.9 or 
                self.step_count >= self.max_steps or
                not result.generation_successful)
        
        print(f"   📊 Validation Score: {result.validation_score:.3f}")
        print(f"   🎁 Reward: {reward:.3f}")
        print(f"   ✅ Done: {done}")
        
        info = {
            'score': result.validation_score,
            'prompt_used': optimized_prompt,
            'generation_successful': result.generation_successful,
            'generation_time': result.generation_time,
            'improvement': result.validation_score - old_score,
            'instruction_used': asdict(instruction)
        }
        
        return self._get_state(), reward, done, info
    
    def _action_to_instruction(self, action_idx: int) -> LLaMAInstruction:
        """Convert RL action index to LLaMA instruction"""
        
        # Define action space - different strategies for LLaMA
        strategies = [
            # Material focus strategies
            ("material_precision", 0.4, "material", "Add precise material specifications and engineering terms", "medium", "conservative"),
            ("material_creative", 0.8, "material", "Create innovative material descriptions with artistic flair", "long", "aggressive"),
            ("material_technical", 0.3, "material", "Focus on technical material properties and manufacturing", "medium", "conservative"),
            
            # Shape/geometry focus
            ("geometric_enhancement", 0.5, "shape", "Enhance geometric precision and dimensional accuracy", "medium", "balanced"),
            ("shape_creative", 0.9, "shape", "Reimagine shape descriptions with creative geometry", "long", "aggressive"),
            
            # Quality focus strategies  
            ("quality_premium", 0.4, "quality", "Emphasize premium quality and craftsmanship", "medium", "conservative"),
            ("quality_artistic", 0.7, "quality", "Elevate to artistic masterpiece level", "long", "balanced"),
            ("quality_technical", 0.3, "quality", "Focus on technical quality standards", "short", "conservative"),
            
            # Context strategies
            ("context_professional", 0.4, "context", "Add professional product photography context", "medium", "balanced"),
            ("context_creative", 0.8, "context", "Create compelling scene and lighting context", "long", "aggressive"),
            
            # Risk strategies
            ("minimal_safe", 0.2, "material", "Make minimal safe improvements", "short", "conservative"),
            ("balanced_approach", 0.5, "quality", "Balanced enhancement across all aspects", "medium", "balanced"),
            ("maximum_enhancement", 0.9, "quality", "Go all-out for maximum scoring potential", "long", "aggressive"),
        ]
        
        # Clamp action index
        action_idx = action_idx % len(strategies)
        strategy_type, creativity, focus, directive, length, risk = strategies[action_idx]
        
        return LLaMAInstruction(
            strategy_type=strategy_type,
            creativity_level=creativity,
            technical_focus=focus,
            style_directive=directive,
            length_target=length,
            risk_tolerance=risk
        )
    
    def _generate_and_validate(self, prompt: str) -> GenerationResult:
        """Actually generate 3D model and validate it"""
        
        start_time = time.time()
        
        try:
            # Step 1: Generate 3D model using TRELLIS server
            print(f"   🎨 Generating 3D model...")
            
            generation_data = {
                'prompt': prompt,
                'seed': 42,  # Fixed seed for reproducibility
                'return_compressed': True
            }
            
            gen_response = requests.post(
                f"{self.generation_server_url}/generate/",
                data=generation_data,
                timeout=120  # 2 minutes max
            )
            
            if gen_response.status_code != 200:
                return GenerationResult(
                    prompt_used=prompt,
                    generation_successful=False,
                    validation_score=0.0,
                    generation_time=time.time() - start_time,
                    file_size_bytes=0,
                    error_message=f"Generation failed: HTTP {gen_response.status_code}"
                )
            
            ply_data = gen_response.content
            generation_time = time.time() - start_time
            
            print(f"   ✅ Generation successful ({len(ply_data):,} bytes, {generation_time:.1f}s)")
            
            # Step 2: Validate using subnet validation
            print(f"   📊 Validating prompt...")
            
            validation_start = time.time()
            
            try:
                # Decompress for validation
                import pyspz
                decompressed_data = pyspz.decompress(ply_data)
            except Exception as e:
                print(f"   ⚠️ Decompression failed, using raw data: {e}")
                decompressed_data = ply_data
            
            # Encode for validation
            import base64
            encoded_data = base64.b64encode(decompressed_data).decode('utf-8')
            
            validation_data = {
                "prompt": prompt,
                "data": encoded_data,
                "compression": 0,
                "generate_preview": False,
                "preview_score_threshold": 0.8
            }
            
            val_response = requests.post(
                f"{self.validation_server_url}/validate_txt_to_3d_ply/",
                json=validation_data,
                timeout=60
            )
            
            validation_time = time.time() - validation_start
            
            if val_response.status_code == 200:
                val_result = val_response.json()
                score = val_result.get("score", 0.0)
                
                print(f"   ✅ Validation complete ({validation_time:.1f}s)")
                print(f"   📊 Score: {score:.4f}")
                
                return GenerationResult(
                    prompt_used=prompt,
                    generation_successful=True,
                    validation_score=score,
                    generation_time=generation_time + validation_time,
                    file_size_bytes=len(ply_data)
                )
            else:
                print(f"   ❌ Validation failed: HTTP {val_response.status_code}")
                return GenerationResult(
                    prompt_used=prompt,
                    generation_successful=False,
                    validation_score=0.0,
                    generation_time=generation_time,
                    file_size_bytes=len(ply_data),
                    error_message=f"Validation failed: HTTP {val_response.status_code}"
                )
        
        except Exception as e:
            error_time = time.time() - start_time
            print(f"   ❌ Generation/Validation error: {e}")
            
            return GenerationResult(
                prompt_used=prompt,
                generation_successful=False,
                validation_score=0.0,
                generation_time=error_time,
                file_size_bytes=0,
                error_message=str(e)
            )
    
    def _calculate_reward(self, result: GenerationResult) -> float:
        """Calculate reward based on actual performance"""
        
        if not result.generation_successful:
            return -50.0  # Heavy penalty for generation failure
        
        # Base reward from validation score
        score_reward = result.validation_score * 100
        
        # Bonus for high scores
        if result.validation_score >= 0.9:
            score_reward += 100  # Ultra bonus
        elif result.validation_score >= 0.8:
            score_reward += 50   # High score bonus
        elif result.validation_score >= 0.7:
            score_reward += 20   # Good score bonus
        
        # Improvement bonus
        if len(self.validation_history) > 1:
            improvement = result.validation_score - self.validation_history[-2].validation_score
            score_reward += improvement * 50  # Bonus for improvement
        
        # Time penalty (encourage efficiency)
        if result.generation_time > 60:  # More than 1 minute
            score_reward -= 10
        
        # Personal best bonus
        if result.validation_score > self.best_score_achieved:
            score_reward += 30
        
        return score_reward
    
    def _get_state(self) -> np.ndarray:
        """Get current state representation"""
        
        state = np.zeros(self.state_size)
        
        # Current scores (last 3)
        for i, result in enumerate(self.validation_history[-3:]):
            if i < 3:
                state[i] = result.validation_score
        
        # Step progress
        state[3] = self.step_count / self.max_steps
        
        # Best score so far
        state[4] = self.best_score_achieved
        
        # Recent trend
        if len(self.validation_history) >= 2:
            state[5] = self.validation_history[-1].validation_score - self.validation_history[-2].validation_score
        
        # Success rate
        successful_generations = sum(1 for r in self.validation_history if r.generation_successful)
        state[6] = successful_generations / max(1, len(self.validation_history))
        
        # Average generation time (normalized)
        if self.validation_history:
            avg_time = sum(r.generation_time for r in self.validation_history) / len(self.validation_history)
            state[7] = min(avg_time / 120, 1.0)  # Normalize by 2 minutes
        
        # LLaMA learning indicators
        state[8] = min(len(self.llama_generator.successful_patterns) / 10, 1.0)
        state[9] = min(len(self.llama_generator.failed_patterns) / 5, 1.0)
        
        # Prompt characteristics (simple analysis)
        if self.current_original_prompt:
            prompt_lower = self.current_original_prompt.lower()
            state[10] = 1.0 if any(word in prompt_lower for word in ["steel", "metal", "iron"]) else 0.0
            state[11] = 1.0 if any(word in prompt_lower for word in ["fabric", "cloth", "silk"]) else 0.0
            state[12] = 1.0 if any(word in prompt_lower for word in ["glass", "crystal", "transparent"]) else 0.0
            state[13] = len(self.current_original_prompt) / 50  # Normalized length
        
        # Performance indicators
        state[14] = 1.0 if self.validation_history and self.validation_history[-1].validation_score >= 0.8 else 0.0
        state[15] = 1.0 if self.validation_history and self.validation_history[-1].validation_score >= 0.9 else 0.0
        
        return state
    
    @property
    def action_size(self):
        """Number of available actions (LLaMA instruction strategies)"""
        return 13  # Number of strategies defined in _action_to_instruction

class RLLLaMAAgent:
    """RL Agent that learns when and how to instruct LLaMA"""
    
    def __init__(self, state_size: int, action_size: int):
        self.state_size = state_size
        self.action_size = action_size
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Neural networks
        self.q_network = self._build_network().to(self.device)
        self.target_network = self._build_network().to(self.device)
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=0.001)
        
        # Replay buffer
        self.memory = deque(maxlen=10000)
        
        # Hyperparameters
        self.batch_size = 32
        self.gamma = 0.95
        self.epsilon = 0.9
        self.epsilon_min = 0.1
        self.epsilon_decay = 0.995
        self.tau = 0.005
        
        # Training tracking
        self.step_count = 0
        self.learn_count = 0
        
        print(f"🤖 RL-LLaMA AGENT INITIALIZED")
        print(f"   🧠 Device: {self.device}")
        print(f"   🎯 Action Space: {action_size} LLaMA instruction strategies")
    
    def _build_network(self) -> nn.Module:
        """Build Q-network"""
        return nn.Sequential(
            nn.Linear(self.state_size, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, self.action_size)
        )
    
    def act(self, state: np.ndarray, training: bool = True) -> int:
        """Choose action using epsilon-greedy policy"""
        
        if training and random.random() < self.epsilon:
            action = random.randrange(self.action_size)
            print(f"   🎲 EXPLORATION: Random strategy {action} (ε={self.epsilon:.3f})")
            return action
        else:
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            with torch.no_grad():
                q_values = self.q_network(state_tensor)
            action = q_values.argmax().item()
            print(f"   🧠 EXPLOITATION: Strategy {action} (ε={self.epsilon:.3f})")
            return action
    
    def remember(self, state, action, reward, next_state, done):
        """Store experience in replay buffer"""
        experience = Experience(state, action, reward, next_state, done)
        self.memory.append(experience)
    
    def learn(self):
        """Train the network on a batch of experiences"""
        if len(self.memory) < self.batch_size:
            return
        
        batch = random.sample(self.memory, self.batch_size)
        states = torch.FloatTensor([e.state for e in batch]).to(self.device)
        actions = torch.LongTensor([e.action for e in batch]).to(self.device)
        rewards = torch.FloatTensor([e.reward for e in batch]).to(self.device)
        next_states = torch.FloatTensor([e.next_state for e in batch]).to(self.device)
        dones = torch.BoolTensor([e.done for e in batch]).to(self.device)
        
        current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1))
        next_q_values = self.target_network(next_states).max(1)[0].detach()
        target_q_values = rewards + (self.gamma * next_q_values * ~dones)
        
        loss = F.mse_loss(current_q_values.squeeze(), target_q_values)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        self.learn_count += 1
        
        # Soft update target network
        for target_param, local_param in zip(self.target_network.parameters(), self.q_network.parameters()):
            target_param.data.copy_(self.tau * local_param.data + (1.0 - self.tau) * target_param.data)
        
        # Decay epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        
        print(f"   📚 LEARNING: Loss {loss.item():.4f}, Memory {len(self.memory)}, ε={self.epsilon:.3f}")

class RLLLaMAOptimizer:
    """Main optimizer combining RL + LLaMA v3"""
    
    def __init__(self, generation_server_url: str = "http://localhost:8096"):
        self.env = RLLLaMAEnvironment(generation_server_url)
        self.agent = RLLLaMAAgent(self.env.state_size, self.env.action_size)
        
        # Training tracking
        self.episode_count = 0
        self.total_rewards = []
        self.best_scores = []
        
        print(f"🚀 RL-LLaMA OPTIMIZER V3 INITIALIZED")
        print(f"   🎮 Environment: Real generation + validation feedback")
        print(f"   🤖 Agent: RL strategy learning")
        print(f"   🧠 Generator: LLaMA 3.2 custom prompt creation")
    
    def optimize_prompt(self, original_prompt: str, max_steps: int = 5) -> Dict[str, Any]:
        """Optimize a single prompt using RL + LLaMA"""
        
        self.episode_count += 1
        print(f"\n🎓 EPISODE {self.episode_count}: Optimizing '{original_prompt}'")
        print("=" * 80)
        
        # Reset environment
        state = self.env.reset(original_prompt)
        total_reward = 0
        step_results = []
        
        for step in range(max_steps):
            # Agent chooses strategy for LLaMA
            action = self.agent.act(state, training=True)
            
            # Environment uses LLaMA to generate and validate
            next_state, reward, done, info = self.env.step(action)
            
            # Store experience
            self.agent.remember(state, action, reward, next_state, done)
            
            # Track results
            step_results.append({
                'step': step + 1,
                'action': action,
                'prompt_generated': info['prompt_used'],
                'score': info['score'],
                'reward': reward,
                'successful': info['generation_successful']
            })
            
            total_reward += reward
            state = next_state
            
            if done:
                break
        
        # Learn from this episode
        if len(self.agent.memory) >= self.agent.batch_size:
            self.agent.learn()
        
        # Track episode results
        best_score = max(result['score'] for result in step_results)
        self.total_rewards.append(total_reward)
        self.best_scores.append(best_score)
        
        # Find best prompt from this episode
        best_step = max(step_results, key=lambda x: x['score'])
        
        print(f"\n📊 EPISODE {self.episode_count} COMPLETE")
        print(f"   🏆 Best Score: {best_score:.3f}")
        print(f"   🎁 Total Reward: {total_reward:.2f}")
        print(f"   ✨ Best Prompt: {best_step['prompt_generated']}")
        print(f"   📚 Agent Memory: {len(self.agent.memory)} experiences")
        
        return {
            'original_prompt': original_prompt,
            'best_optimized_prompt': best_step['prompt_generated'],
            'best_score': best_score,
            'total_reward': total_reward,
            'steps_taken': len(step_results),
            'step_details': step_results,
            'episode': self.episode_count
        }
    
    def train_on_prompts(self, prompts: List[str], episodes_per_prompt: int = 3) -> Dict[str, Any]:
        """Train the RL-LLaMA system on multiple prompts"""
        
        print(f"🎓 TRAINING RL-LLaMA OPTIMIZER V3")
        print(f"   📝 Prompts: {len(prompts)}")
        print(f"   🔄 Episodes per prompt: {episodes_per_prompt}")
        print("=" * 80)
        
        all_results = []
        
        for prompt_idx, prompt in enumerate(prompts):
            print(f"\n🎯 TRAINING ON PROMPT {prompt_idx + 1}/{len(prompts)}: '{prompt}'")
            
            for episode in range(episodes_per_prompt):
                result = self.optimize_prompt(prompt)
                all_results.append(result)
                
                # Show progress
                avg_score = sum(self.best_scores[-5:]) / min(5, len(self.best_scores))
                print(f"   📈 Recent average score: {avg_score:.3f}")
        
        # Training summary
        total_episodes = len(all_results)
        avg_score = sum(self.best_scores) / len(self.best_scores)
        best_overall = max(self.best_scores)
        ultra_achievements = sum(1 for score in self.best_scores if score >= 0.9)
        
        print(f"\n🎉 TRAINING COMPLETE!")
        print(f"   📊 Total Episodes: {total_episodes}")
        print(f"   📈 Average Score: {avg_score:.3f}")
        print(f"   🏆 Best Score: {best_overall:.3f}")
        print(f"   🌟 Ultra Achievements: {ultra_achievements}/{total_episodes} ({ultra_achievements/total_episodes*100:.1f}%)")
        print(f"   🧠 LLaMA Learned Patterns: {len(self.env.llama_generator.successful_patterns)}")
        
        return {
            'total_episodes': total_episodes,
            'average_score': avg_score,
            'best_score': best_overall,
            'ultra_achievements': ultra_achievements,
            'ultra_rate': ultra_achievements / total_episodes,
            'llama_patterns_learned': len(self.env.llama_generator.successful_patterns),
            'all_results': all_results
        }

def main():
    """Test the RL + LLaMA v3 optimizer"""
    
    print("🚀 RL + LLaMA OPTIMIZER V3 - INTELLIGENT PROMPT GENERATION")
    print("="*80)
    print("✅ LLaMA 3.2 generates custom prompts (not templates)")
    print("✅ RL agent learns optimal LLaMA instruction strategies") 
    print("✅ Real feedback from generation server validation")
    print("✅ Continuous learning from actual performance")
    print("="*80)
    
    try:
        # Initialize optimizer
        optimizer = RLLLaMAOptimizer(generation_server_url="http://localhost:8096")
        
        # Test prompts
        test_prompts = [
            "hexagonal prism steel structure",
            "elegant silk fabric draping",
            "transparent glass sphere with reflections"
        ]
        
        # Train the system
        results = optimizer.train_on_prompts(test_prompts, episodes_per_prompt=2)
        
        print(f"\n🎯 SYSTEM PERFORMANCE:")
        print(f"   Average Score: {results['average_score']:.3f}")
        print(f"   Best Score: {results['best_score']:.3f}")
        print(f"   Ultra Rate: {results['ultra_rate']:.1%}")
        print(f"   Patterns Learned: {results['llama_patterns_learned']}")
        
    except Exception as e:
        print(f"❌ Demo failed: {e}")
        print("   Ensure generation and validation servers are running")

if __name__ == "__main__":
    main() 