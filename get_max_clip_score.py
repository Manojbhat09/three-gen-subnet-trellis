#!/usr/bin/env python3
"""
CLIP Score Maximizer - Advanced 3D-Ready Optimization System v4 with RL Learning Loop
====================================================================================

This script implements an advanced optimization strategy focused on generating 2D images
that are ideal inputs for image-to-3D conversion models. It uses a modern CLIP model and
self-correcting optimization loop to achieve significantly higher scores.

Key Features:
- **Modern CLIP Model**: Uses ViT-H-14 with LAION-2B training for superior evaluation
- **Clay Render Focus**: Optimizes for "clay render" style prompts that emphasize geometric form
- **Negative Prompts**: Critically uses negative prompts to remove shadows, complex backgrounds, and artistic effects
- **Self-Correction**: Automatic reset and mutation when stuck in local maxima
- **Sanity Checks**: Ensures prompts still contain the original subject
- **Multi-Candidate Hill-Climbing**: Evaluates multiple candidates per iteration
- **Adaptive Keyword Weighting**: Learns which technical descriptors work best
- **Creative Mutations**: Occasional creative jumps to escape local maxima
- **🔄 RL Learning Loop**: True reinforcement learning with strategy performance tracking
- **🎯 Score-driven Learning**: Agent adjusts strategies based on CLIP score feedback
- **🧠 Multi-round Conversations**: Iterative optimization with learned insights
- **⚡ Continuous Improvement**: Exploration and exploitation based on performance

Advanced Algorithm:
1. Generate dynamic technical adjectives specific to the input object
2. Generate 5+ candidate prompts per iteration:
   - 1 LLM refinement (builds on best so far with clay render focus)
   - 1 creative mutation (15% chance, fresh technical approach)
   - 3-4 weighted templated variations (favors successful technical keywords)
3. Each candidate includes both positive and negative prompts
4. Sanity check ensures prompts contain the original subject
5. Evaluate all candidates with modern CLIP scoring
6. Update keyword scores based on performance
7. Self-correct: reset to best known prompt if stuck for 3 iterations
8. Select the best performer as the seed for the next iteration
9. Repeat until convergence or max iterations reached

Revolutionary RL Loop:
1. Agent makes optimization attempt
2. Gets CLIP score feedback  
3. Reflects on what worked/didn't work
4. Updates strategy preferences based on scores
5. Makes improved attempt using learned insights
6. Repeats until convergence or max rounds

This approach optimizes for 3D model quality using a superior evaluator and intelligent self-correction.
"""

import torch
import open_clip
import numpy as np
import time
import random
import argparse
import json
from typing import List, Dict, Tuple, Optional, Any, Union
from loguru import logger
import requests
from PIL import Image
import io
import base64
import subprocess
import sys
import logging
import os
import re
import statistics
from pathlib import Path
from dataclasses import dataclass, asdict
from datetime import datetime

@dataclass
class OptimizationAttempt:
    """Single optimization attempt within an RL loop"""
    attempt_number: int
    strategy_used: str
    exploration_type: str
    optimized_prompt: str
    predicted_confidence: float
    validation_score: Optional[float]
    agent_reasoning: str
    timestamp: float
    clip_score: float
    negative_prompt: str
    image: Optional[str] = None  # Store the generated image

@dataclass
class RLOptimizationSession:
    """Complete RL session with multiple learning attempts"""
    session_id: str
    original_prompt: str
    attempts: List[OptimizationAttempt]
    final_best_prompt: str
    final_best_score: float
    total_rounds: int
    convergence_achieved: bool
    learned_insights: List[str]
    strategy_performance_updates: Dict[str, float]
    session_duration: float

@dataclass
class StrategyPerformance:
    """Dynamic strategy performance that updates based on scores"""
    strategy_name: str
    success_count: int
    total_attempts: int
    avg_score: float
    recent_scores: List[float]  # Last 10 scores
    confidence_in_strategy: float
    last_used: float
    improvement_trend: float  # Positive = getting better, negative = getting worse

def extract_true_prompt(original_prompt: str) -> str:
    """Extract the true prompt from potentially formatted input"""
    # Try to find "Original Prompt: ..." at the end
    match = re.search(r'Original Prompt:\s*(.*)', original_prompt)
    if match:
        return match.group(1).strip()
    # Otherwise, just return the string as-is
    return original_prompt.strip()

class RLLoopAgent:
    """RL agent that learns through iterative CLIP score optimization loops"""
    
    def __init__(self, 
                 clip_maximizer: 'CLIPScoreMaximizer',
                 ollama_url: str = "http://localhost:11434",
                 memory_file: str = "clip_rl_loop_memory.json",
                 api_base_url: str = "https://openrouter.ai/api/v1"):
        
        self.clip_maximizer = clip_maximizer
        self.ollama_url = ollama_url
        self.model = None  # Will be set based on user input
        self.memory_file = Path(memory_file)
        self.api_base_url = api_base_url
        self.use_openrouter = False
        self.provider = None
        self.api_key = None
        self.site_url = "http://localhost"
        self.app_name = "CLIP RL Optimizer"
        self._choose_llm_provider()

        # RL Loop parameters
        self.max_optimization_rounds = 100
        self.min_rounds_before_convergence = 4
        self.convergence_threshold = 0.015
        self.min_score_threshold = 0.85
        self.explore_performance_threshold = 0.05
        self.convergence_improvement_threshold = 0.08

        # Learning state
        self.strategy_performance: Dict[str, StrategyPerformance] = {}
        self.optimization_sessions: List[RLOptimizationSession] = []
        self.global_insights: List[str] = []

        # RL parameters
        self.epsilon = 0.6
        self.epsilon_decay = 0.98
        self.epsilon_min = 0.3
        self.explore_scores: List[float] = []
        self.exploit_scores: List[float] = []

        self.logger = logging.getLogger(__name__)
        self._load_memory()
        self._initialize_strategies()
        self.logger.info(f"🔄 CLIP RL LOOP AGENT INITIALIZED")
        self.logger.info(f"   Strategy tracking: {len(self.strategy_performance)} strategies")
        self.logger.info(f"   Past sessions: {len(self.optimization_sessions)}")
        self.logger.info(f"   Exploration rate: {self.epsilon:.2f}")
        self.logger.info(f"   Max rounds per optimization: {self.max_optimization_rounds}")
    
    def _choose_llm_provider(self):
        """Choose LLM provider for RL agent"""
        print("\nWhich LLM provider do you want to use for RL learning?")
        print("1. Local Ollama (default: llama3.2:3b)")
        print("2. OpenRouter (cloud, supports many models)")
        choice = input("Enter 1 for Ollama or 2 for OpenRouter: ").strip()
        if choice == "2":
            self.use_openrouter = True
            self.api_key = os.getenv("OPENROUTER_API_KEY")
            if not self.api_key:
                print("\nOPENROUTER_API_KEY environment variable is not set.")
                key = input("Please enter your OpenRouter API key: ").strip()
                if not key:
                    raise ValueError("🚨 No API key provided. Cannot use OpenRouter.")
                os.environ["OPENROUTER_API_KEY"] = key
                self.api_key = key
                print("API key set for this session.")
            print("\nAvailable OpenRouter models (examples):")
            print("- meta-llama/llama-3.3-70b-instruct:free (default)")
            print("- openai/gpt-4-turbo")
            print("- anthropic/claude-3-opus")
            model = input("Enter OpenRouter model (or press Enter for default): ").strip()
            if not model:
                model = "meta-llama/llama-3.3-70b-instruct:free"
            self.model = model
            provider = input("Enter OpenRouter provider (or press Enter for default 'venice/fp8'): ").strip()
            if not provider:
                provider = "venice/fp8"
            self.provider = provider
            print(f"Using OpenRouter model: {self.model} (provider: {self.provider})")
        else:
            self.use_openrouter = False
            self.model = "llama3.2:3b"
            print(f"Using local Ollama model: {self.model}")

    def _load_memory(self):
        """Load RL loop memory"""
        if self.memory_file.exists():
            try:
                with open(self.memory_file, 'r') as f:
                    data = json.load(f)
                
                # Load strategy performance
                strategies_data = data.get('strategy_performance', {})
                self.strategy_performance = {
                    name: StrategyPerformance(**perf)
                    for name, perf in strategies_data.items()
                }
                
                # Load recent sessions
                sessions_data = data.get('optimization_sessions', [])
                self.optimization_sessions = []
                for session_data in sessions_data[-50:]:
                    attempts = []
                    for attempt_data in session_data.get('attempts', []):
                        attempt = OptimizationAttempt(**attempt_data)
                        attempts.append(attempt)
                    
                    session_data['attempts'] = attempts
                    session = RLOptimizationSession(**session_data)
                    self.optimization_sessions.append(session)
                
                self.global_insights = data.get('global_insights', [])
                self.epsilon = data.get('epsilon', self.epsilon)
                
                self.logger.info(f"📚 Loaded CLIP RL memory: {len(self.strategy_performance)} strategies, {len(self.optimization_sessions)} sessions")
                
            except Exception as e:
                self.logger.warning(f"⚠️ Failed to load CLIP RL memory: {e}")
                self._initialize_fresh()
        else:
            self._initialize_fresh()
    
    def _initialize_fresh(self):
        """Initialize fresh RL agent"""
        self.strategy_performance = {}
        self.optimization_sessions = []
        self.global_insights = []
        self.logger.info("📄 Starting fresh CLIP RL loop memory")
    
    def _initialize_strategies(self):
        """Initialize strategy performance tracking for CLIP optimization"""
        default_strategies = [
            "conservative_enhancement",
            "aggressive_transformation", 
            "material_focus",
            "artistic_elaboration",
            "technical_precision",
            "contextual_scene_building",
            "minimalist_refinement",
            "clay_render_focus",
            "geometric_detail",
            "hybrid_approach"
        ]
        
        for strategy in default_strategies:
            if strategy not in self.strategy_performance:
                self.strategy_performance[strategy] = StrategyPerformance(
                    strategy_name=strategy,
                    success_count=0,
                    total_attempts=0,
                    avg_score=0.5,
                    recent_scores=[],
                    confidence_in_strategy=0.5,
                    last_used=0.0,
                    improvement_trend=0.0
                )

    def get_best_previous_session_for_prompt(self, prompt: str) -> Optional[RLOptimizationSession]:
        """Get best previous session for the same prompt"""
        best_session = None
        for session in self.optimization_sessions:
            session_prompt = extract_true_prompt(session.original_prompt)
            if session_prompt == prompt:
                if (best_session is None or 
                    session.final_best_score > best_session.final_best_score):
                    best_session = session
        return best_session

    def get_best_previous_attempt_for_prompt(self, prompt: str) -> Optional[Dict]:
        """Get best previous attempt for the same prompt"""
        best_attempt = None
        for session in self.optimization_sessions:
            session_prompt = extract_true_prompt(session.original_prompt)
            if session_prompt == prompt:
                for attempt in session.attempts:
                    if attempt.validation_score is not None:
                        if best_attempt is None or attempt.validation_score > best_attempt['validation_score']:
                            best_attempt = {
                                'optimized_prompt': attempt.optimized_prompt,
                                'validation_score': attempt.validation_score,
                                'clip_score': attempt.clip_score,
                                'strategy_used': attempt.strategy_used,
                                'session_id': session.session_id,
                                'attempt_number': attempt.attempt_number,
                                'negative_prompt': attempt.negative_prompt
                            }
        return best_attempt

    def optimize_with_rl_loop(self, prompt: str, seed: Optional[int] = None) -> Dict[str, Any]:
        """Optimize prompt using RL learning loop with CLIP scoring"""
        session_id = f"clip_rl_session_{int(time.time())}_{random.randint(1000, 9999)}"
        start_time = time.time()
        self.logger.info(f"\n🔄 CLIP RL LOOP OPTIMIZATION: '{prompt}'")
        self.logger.info(f"   Session: {session_id}")
        self.logger.info(f"   Max rounds: {self.max_optimization_rounds}")
        self.logger.info(f"   Target score: {self.min_score_threshold}")
        
        attempts: List[OptimizationAttempt] = []
        best_prompt = prompt
        best_score = 0.0
        initial_score = None
        convergence_achieved = False
        convergence_reason = "Not converged"
        
        # Check for previous best attempts
        best_prev_session = self.get_best_previous_session_for_prompt(prompt)
        best_prev_attempt = self.get_best_previous_attempt_for_prompt(prompt)
        
        if best_prev_attempt:
            best_prompt = best_prev_attempt['optimized_prompt']
            best_score = best_prev_attempt['validation_score']
            initial_score = best_score
            self.logger.info(f"      🎯 Found previous best attempt: {best_prompt} (score: {best_score:.4f})")
            self.logger.info(f"      🎯 Strategy used: {best_prev_attempt['strategy_used']}")
            
            if best_prev_session:
                self.logger.info(f"      🎯 Past session insights: {', '.join(best_prev_session.learned_insights)}")

        # Main RL optimization loop
        for round_num in range(1, self.max_optimization_rounds + 1):
            self.logger.info(f"\n   🔄 RL Round {round_num}/{self.max_optimization_rounds}")
            
            strategy, exploration_type = self._select_strategy_for_rl()
            if convergence_achieved:
                exploration_type = "exploit"
                convergence_achieved = False
            attempt = self._make_optimization_attempt(
                prompt, strategy, exploration_type, round_num, attempts, seed
            )
            
            # Validate with CLIP scoring
            attempt.validation_score, attempt.clip_score, attempt.negative_prompt, attempt.image = self._validate_with_clip(
                prompt, attempt.optimized_prompt, seed
            )
            
            self.logger.info(f"      📊 CLIP score: {attempt.clip_score:.4f}")
            self.logger.info(f"      📊 Validation score: {attempt.validation_score:.4f}")
            
            attempts.append(attempt)
            
            if initial_score is None and attempt.validation_score is not None:
                initial_score = attempt.validation_score
            
            if attempt.validation_score:
                if exploration_type == 'explore':
                    self.explore_scores.append(attempt.validation_score)
                else:
                    self.exploit_scores.append(attempt.validation_score)
                
                if len(self.explore_scores) > 50:
                    self.explore_scores = self.explore_scores[-50:]
                if len(self.exploit_scores) > 50:
                    self.exploit_scores = self.exploit_scores[-50:]
            
            if attempt.validation_score and attempt.validation_score > best_score:
                best_score = attempt.validation_score
                best_prompt = attempt.optimized_prompt
                self.logger.info(f"      🎯 New best score: {best_score:.4f}")
            
            self._update_strategy_performance(strategy, attempt.validation_score or 0.0)
            
            # Fix convergence bug and add stuck detection
            should_converge, reason = self._should_converge(attempts, round_num)
            if should_converge:
                convergence_achieved = True
                convergence_reason = reason
                self.logger.info(f"      ✅ Convergence achieved: {reason}")
                # break  # Actually break when convergence is achieved
            
            # Add stuck detection - if score hasn't changed for 10 rounds, force stop
            if len(attempts) >= 10:
                recent_scores = [a.validation_score or 0.0 for a in attempts[-10:]]
                if all(abs(score - recent_scores[0]) < 0.001 for score in recent_scores):
                    convergence_achieved = True
                    convergence_reason = f"Stuck at score {recent_scores[0]:.4f} for 10 rounds"
                    self.logger.info(f"      🛑 Forced convergence: {convergence_reason}")
                    # break
            
            if round_num < self.max_optimization_rounds:
                self._inter_round_learning(attempts)
        
        learned_insights = self._extract_session_insights(attempts)
        strategy_updates = self._calculate_strategy_updates(attempts)
        
        session = RLOptimizationSession(
            session_id=session_id,
            original_prompt=prompt,
            attempts=attempts,
            final_best_prompt=best_prompt,
            final_best_score=best_score,
            total_rounds=len(attempts),
            convergence_achieved=convergence_achieved,
            learned_insights=learned_insights,
            strategy_performance_updates=strategy_updates,
            session_duration=time.time() - start_time
        )
        
        self.optimization_sessions.append(session)
        self.global_insights.extend(learned_insights)
        self._decay_exploration()
        self._save_memory()
        
        # Get the best image from the best attempt
        best_image = None
        if attempts:
            best_attempt = max(attempts, key=lambda x: x.validation_score or 0.0)
            if best_attempt.image:
                best_image = best_attempt.image
        
        result = {
            'session_id': session_id,
            'original_prompt': prompt,
            'final_optimized_prompt': best_prompt,
            'final_score': best_score,
            'initial_score': initial_score if initial_score is not None else (attempts[0].validation_score if attempts else 0.0),
            'total_rounds': len(attempts),
            'convergence_achieved': convergence_achieved,
            'convergence_reason': convergence_reason,
            'learned_insights': learned_insights,
            'strategy_updates': strategy_updates,
            'processing_time': session.session_duration,
            'score_progression': [a.validation_score for a in attempts],
            'clip_score_progression': [a.clip_score for a in attempts],
            'strategy_sequence': [a.strategy_used for a in attempts],
            'exploration_ratio': sum(1 for a in attempts if a.exploration_type == 'explore') / len(attempts),
            'best_image': best_image
        }
        
        self.logger.info(f"\n🎯 CLIP RL LOOP COMPLETE:")
        self.logger.info(f"   Best prompt: {best_prompt}")
        self.logger.info(f"   Best score: {best_score:.4f}")
        self.logger.info(f"   Rounds: {len(attempts)}")
        self.logger.info(f"   Convergence: {convergence_achieved}")
        self.logger.info(f"   Exploration ratio: {result['exploration_ratio']:.1%}")
        self.logger.info(f"   Insights learned: {len(learned_insights)}")
        self.logger.info(f"   Total time: {session.session_duration:.2f}s")
        
        return result

    def _should_converge(self, attempts: List[OptimizationAttempt], current_round: int) -> Tuple[bool, str]:
        """Determine if optimization should converge"""
        if current_round < self.min_rounds_before_convergence:
            return False, f"Below minimum rounds ({current_round}/{self.min_rounds_before_convergence})"
        
        # Check if we achieved target score
        if attempts and attempts[-1].validation_score and attempts[-1].validation_score >= self.min_score_threshold:
            return True, f"Target score achieved ({attempts[-1].validation_score:.3f})"
        
        # Check for convergence based on improvement
        if len(attempts) >= 2:
            current_score = attempts[-1].validation_score or 0.0
            previous_score = attempts[-2].validation_score or 0.0
            improvement = current_score - previous_score
            adaptive_threshold = self.convergence_threshold
            
            if self.explore_scores and self.exploit_scores:
                explore_avg = statistics.mean(self.explore_scores[-10:]) if len(self.explore_scores) >= 10 else 0.5
                exploit_avg = statistics.mean(self.exploit_scores[-10:]) if len(self.exploit_scores) >= 10 else 0.5
                if explore_avg > exploit_avg + self.explore_performance_threshold:
                    adaptive_threshold = self.convergence_threshold * 0.5
                    self.logger.info(f"      🔍 Explore performing better - stricter convergence threshold: {adaptive_threshold:.3f}")
            
            if abs(improvement) < adaptive_threshold:
                explore_attempts = sum(1 for a in attempts if a.exploration_type == 'explore')
                total_attempts = len(attempts)
                explore_ratio = explore_attempts / total_attempts
                if explore_ratio < 0.3:
                    return False, f"Insufficient exploration ({explore_ratio:.1%}) for convergence"
                return True, f"Convergence threshold met (improvement: {improvement:.3f})"
        
        return False, "No convergence criteria met"

    def _select_strategy_for_rl(self) -> Tuple[str, str]:
        """Select strategy for RL optimization"""
        if self.explore_scores and self.exploit_scores:
            recent_explore = statistics.mean(self.explore_scores[-5:]) if len(self.explore_scores) >= 5 else 0.5
            recent_exploit = statistics.mean(self.exploit_scores[-5:]) if len(self.exploit_scores) >= 5 else 0.5
            if recent_explore > recent_exploit + self.explore_performance_threshold:
                self.epsilon = min(0.8, self.epsilon + 0.05)
                self.logger.info(f"      🔍 Explore performing better - increasing epsilon to {self.epsilon:.2f}")
            elif recent_exploit > recent_explore + self.explore_performance_threshold:
                self.epsilon = max(self.epsilon_min, self.epsilon - 0.02)
        
        if random.random() < self.epsilon:
            exploration_type = "explore"
            strategy_scores = []
            for name, perf in self.strategy_performance.items():
                if perf.total_attempts == 0:
                    uncertainty = float('inf')
                else:
                    uncertainty = 1.0 / (perf.total_attempts + 1)
                    if perf.improvement_trend < 0:
                        uncertainty += 0.5
                strategy_scores.append((name, uncertainty))
            strategy_scores.sort(key=lambda x: x[1], reverse=True)
            selected_strategy = strategy_scores[0][0]
        else:
            exploration_type = "exploit"
            best_strategy = None
            best_combined_score = -1
            for name, perf in self.strategy_performance.items():
                if perf.total_attempts == 0:
                    continue
                combined_score = (perf.avg_score * 0.5 + 
                                perf.confidence_in_strategy * 0.3 + 
                                max(0, perf.improvement_trend) * 0.2)
                if combined_score > best_combined_score:
                    best_combined_score = combined_score
                    best_strategy = name
            selected_strategy = best_strategy or "conservative_enhancement"
        
        return selected_strategy, exploration_type

    def _make_optimization_attempt(self, prompt: str, strategy: str, exploration_type: str, 
                                 round_num: int, previous_attempts: List[OptimizationAttempt], 
                                 seed: Optional[int] = None, max_retries: int = 3) -> OptimizationAttempt:
        """Make a single optimization attempt with context from previous rounds"""
        self.logger.info(f"      🎯 Strategy: {strategy} ({exploration_type})")
        
        previous_context = self._build_previous_attempts_context(previous_attempts)
        strategy_context = self._build_strategy_context(strategy)
        
        system_prompt = f"""You are an RL agent learning to optimize prompts for CLIP score maximization. This is round {round_num} of iterative optimization.

ORIGINAL PROMPT: "{prompt}"
STRATEGY: {strategy} ({exploration_type} mode)

{strategy_context}

{previous_context}

TASK: Create an optimized prompt that improves CLIP scores based on learned insights.

🎯 CRITICAL OBJECTIVE: Beat the current episode's best score shown above!
- If there's a best score to beat, you MUST achieve a higher score
- Focus on the strategy that worked best so far
- Learn from what made the best attempt successful
- Be {'experimental' if exploration_type == 'explore' else 'systematic'} in your approach

Rules:
- Learn from previous rounds' CLIP scores and approaches
- Focus on technical details that improve CLIP alignment
- Consider material properties, geometric features, and visual clarity
- If a strategy worked well, build upon it
- If scores are declining, try a different approach

**Prime Example:**
* **ORIGINAL:** `tall glass of layered lemonade`
* **OPTIMIZED (Score: 0.9443):** `a slender glass of layered lemonade suspended in mid-air over a serene lake with lotus flowers and gentle ripples on the surface`

If the original prompt is about an object, don't focus on the scene or background but refine details of the object itself.
Example:
* **ORIGINAL:** `small wooden hammer with screws`
* **OPTIMIZED (Score: 0.0 ⚠️):** `a weathered small wooden hammer resting on a worn leather workbench amidst tools of various trades in a cozy, rustic workshop filled with natural light and the scent of sawdust`
* **OPTIMIZED (Score: 0.8287):** `small wooden hammer with screws` 

RESPONSE FORMAT:
REASONING: [Your reasoning considering previous attempts and how to beat the best score]

OPTIMIZATION: {{
  "optimized_prompt": "[full optimized prompt]",
  "confidence": [0.0-1.0],
  "key_changes": ["change1", "change2"],
  "expected_score": [0.0-1.0],
  "learning_applied": ["insight1", "insight2"]
}}"""

        retries = 0
        while retries < max_retries:
            try:
                response = self._query_llm(system_prompt) if self.use_openrouter else self._query_ollama(system_prompt)
                structured_output = self._parse_optimization_response(response, prompt)
                
                return OptimizationAttempt(
                    attempt_number=round_num,
                    strategy_used=strategy,
                    exploration_type=exploration_type,
                    optimized_prompt=structured_output.get('optimized_prompt', prompt),
                    predicted_confidence=structured_output.get('confidence', 0.5),
                    validation_score=None,  # Will be filled later
                    agent_reasoning=response,
                    timestamp=time.time(),
                    clip_score=0.0,  # Will be filled later
                    negative_prompt="blurry, shadows, artistic, grainy, low-resolution",
                    image=None  # Will be filled later
                )
            except Exception as e:
                self.logger.error(f"      ❌ Optimization attempt failed: {e}")
                retries += 1
                if retries < max_retries:
                    time.sleep(5)
        
        self.logger.error(f"      ❌ All retries failed for RL round {round_num}. Skipping this round.")
        return OptimizationAttempt(
            attempt_number=round_num,
            strategy_used=strategy,
            exploration_type=exploration_type,
            optimized_prompt=prompt,
            predicted_confidence=0.5,
            validation_score=None,
            agent_reasoning=f"Fallback due to repeated error",
            timestamp=time.time(),
            clip_score=0.0,
            negative_prompt="blurry, shadows, artistic, grainy, low-resolution",
            image=None
        )

    def _build_previous_attempts_context(self, previous_attempts: List[OptimizationAttempt]) -> str:
        """Build context from previous optimization attempts"""
        if not previous_attempts:
            return "PREVIOUS ATTEMPTS: None - this is your first attempt."
        
        # Find current episode's best score and attempt
        valid_attempts = [a for a in previous_attempts if a.validation_score is not None]
        if valid_attempts:
            best_attempt = max(valid_attempts, key=lambda x: x.validation_score)
            best_score = best_attempt.validation_score
            best_clip_score = best_attempt.clip_score
            best_prompt = best_attempt.optimized_prompt
            best_strategy = best_attempt.strategy_used
        else:
            best_score = 0.0
            best_clip_score = 0.0
            best_prompt = "None yet"
            best_strategy = "None"
        
        context = f"""🎯 CURRENT EPISODE BEST TO BEAT:
   Best Validation Score: {best_score:.4f}
   Best CLIP Score: {best_clip_score:.4f}
   Best Strategy Used: {best_strategy}
   Best Prompt: "{best_prompt[:80]}..."
   
📊 PREVIOUS ATTEMPTS IN THIS SESSION:\n"""
        
        for attempt in previous_attempts:
            score_text = f"{attempt.validation_score:.4f}" if attempt.validation_score else "pending"
            clip_text = f"{attempt.clip_score:.4f}" if attempt.clip_score else "pending"
            is_best = attempt.validation_score == best_score if attempt.validation_score else False
            best_marker = " 🏆" if is_best else ""
            
            context += f"Round {attempt.attempt_number}: {attempt.strategy_used}{best_marker}\n"
            context += f"  CLIP Score: {clip_text} | Validation Score: {score_text} | Confidence: {attempt.predicted_confidence:.2f}\n"
            context += f"  Prompt: {attempt.optimized_prompt[:100]}...\n"
        
        # Add insights
        if len(previous_attempts) > 1:
            scores = [a.validation_score for a in previous_attempts if a.validation_score]
            clip_scores = [a.clip_score for a in previous_attempts if a.clip_score]
            
            if len(scores) > 1:
                trend = "improving" if scores[-1] > scores[-2] else "declining"
                context += f"\nTREND: Validation scores are {trend}. "
                
                if trend == "declining":
                    context += "Consider why the previous approach worked better."
                else:
                    context += "Build on what's working well."
            
            if len(clip_scores) > 1:
                clip_trend = "improving" if clip_scores[-1] > clip_scores[-2] else "declining"
                context += f"\nCLIP TREND: CLIP scores are {clip_trend}. "
                
                if clip_trend == "declining":
                    context += "Focus on improving CLIP alignment by making prompts more specific."
                else:
                    context += "CLIP alignment is improving - continue with this approach."
        
        # Add improvement requirement
        if best_score > 0:
            context += f"\n\n🚨 IMPROVEMENT REQUIREMENT:"
            context += f"\n   You MUST achieve a score > {best_score:.4f} (current episode best)"
            context += f"\n   Target improvement: +0.01 or better"
            context += f"\n   Focus on beating the best strategy: {best_strategy}"
        
        return context

    def _build_strategy_context(self, strategy: str) -> str:
        """Build context for specific strategy"""
        perf = self.strategy_performance.get(strategy)
        if not perf or perf.total_attempts == 0:
            return f"STRATEGY {strategy}: No prior experience - explore freely."
        
        context = f"STRATEGY {strategy} PERFORMANCE:\n"
        context += f"  Average score: {perf.avg_score:.3f}\n"
        context += f"  Success rate: {perf.success_count / perf.total_attempts:.1%}\n"
        context += f"  Confidence: {perf.confidence_in_strategy:.2f}\n"
        
        if perf.recent_scores:
            recent_avg = statistics.mean(perf.recent_scores[-3:])
            context += f"  Recent performance: {recent_avg:.3f}\n"
        
        if perf.improvement_trend > 0:
            context += "  This strategy is improving - leverage it!\n"
        elif perf.improvement_trend < -0.1:
            context += "  This strategy is declining - try a different approach.\n"
        
        return context

    def _validate_with_clip(self, original_prompt: str, optimized_prompt: str, seed: Optional[int] = None) -> Tuple[float, float, str, Optional[str]]:
        """Validate prompt using CLIP scoring"""
        try:
            self.logger.info(f"      🔍 Validating with CLIP...")
            
            # Generate image with optimized prompt
            image = self.clip_maximizer.generate_dit_image(optimized_prompt, seed)
            if not image:
                self.logger.warning("      Image generation failed")
                return 0.0, 0.0, "blurry, shadows, artistic, grainy, low-resolution", None
            
            # Compute CLIP score between original prompt and generated image
            clip_score = self.clip_maximizer.compute_clip_score(original_prompt, image)
            
            # For validation score, we can use the CLIP score directly or apply some transformation
            validation_score = clip_score  # Simple mapping for now
            
            self.logger.info(f"      📊 CLIP score: {clip_score:.4f}")
            self.logger.info(f"      📊 Validation score: {validation_score:.4f}")
            
            return validation_score, clip_score, "blurry, shadows, artistic, grainy, low-resolution", image
            
        except Exception as e:
            self.logger.error(f"      ❌ CLIP validation error: {e}")
            return 0.0, 0.0, "blurry, shadows, artistic, grainy, low-resolution", None

    def _inter_round_learning(self, attempts: List[OptimizationAttempt]):
        """Learn between rounds to improve next attempt"""
        if len(attempts) < 2:
            return
        
        current = attempts[-1]
        previous = attempts[-2]
        
        # Quick analysis for next round
        if current.validation_score and previous.validation_score:
            if current.validation_score > previous.validation_score:
                self.logger.info(f"      📈 Improvement detected: {current.strategy_used} working better")
            else:
                self.logger.info(f"      📉 Score declined: Consider different approach")
        
        # Update epsilon based on performance
        if current.validation_score and current.validation_score < 0.5:
            self.epsilon = min(0.8, self.epsilon + 0.1)  # Explore more if doing poorly
            self.logger.info(f"      🔍 Increasing exploration to {self.epsilon:.2f}")

    def _update_strategy_performance(self, strategy: str, score: float):
        """Update strategy performance based on new score"""
        perf = self.strategy_performance[strategy]
        
        # Update basic stats
        perf.total_attempts += 1
        if score >= 0.7:  # Success threshold
            perf.success_count += 1
        
        # Update average score with exponential moving average
        alpha = 0.2
        perf.avg_score = (1 - alpha) * perf.avg_score + alpha * score
        
        # Update recent scores
        perf.recent_scores.append(score)
        if len(perf.recent_scores) > 10:
            perf.recent_scores.pop(0)
        
        # Calculate improvement trend
        if len(perf.recent_scores) >= 3:
            recent_scores = perf.recent_scores[-3:]
            trend = (recent_scores[-1] - recent_scores[0]) / 2
            perf.improvement_trend = trend
        
        # Update confidence in strategy
        if len(perf.recent_scores) >= 2:
            consistency = 1.0 - statistics.stdev(perf.recent_scores[-5:]) if len(perf.recent_scores) >= 2 else 0.5
            perf.confidence_in_strategy = (perf.avg_score + consistency) / 2
        
        perf.last_used = time.time()

    def _extract_session_insights(self, attempts: List[OptimizationAttempt]) -> List[str]:
        """Extract insights from the optimization session"""
        insights = []
        
        if len(attempts) <= 1:
            return insights
        
        # Score progression insights
        scores = [a.validation_score for a in attempts if a.validation_score]
        if len(scores) > 1:
            if scores[-1] > scores[0]:
                insights.append(f"Iterative improvement successful: {scores[0]:.3f} → {scores[-1]:.3f}")
            
            best_idx = scores.index(max(scores))
            best_strategy = attempts[best_idx].strategy_used
            insights.append(f"Best performing strategy in session: {best_strategy}")
        
        # Strategy insights
        strategy_counts = {}
        for attempt in attempts:
            strategy_counts[attempt.strategy_used] = strategy_counts.get(attempt.strategy_used, 0) + 1
        
        if len(strategy_counts) > 1:
            insights.append(f"Explored {len(strategy_counts)} different strategies")
        
        return insights

    def _calculate_strategy_updates(self, attempts: List[OptimizationAttempt]) -> Dict[str, float]:
        """Calculate how much each strategy's performance changed"""
        updates = {}
        
        for attempt in attempts:
            if attempt.validation_score:
                strategy = attempt.strategy_used
                if strategy not in updates:
                    updates[strategy] = []
                updates[strategy].append(attempt.validation_score)
        
        # Calculate average performance for each strategy in this session
        strategy_updates = {}
        for strategy, scores in updates.items():
            avg_session_score = statistics.mean(scores)
            current_avg = self.strategy_performance[strategy].avg_score
            improvement = avg_session_score - current_avg
            strategy_updates[strategy] = improvement
        
        return strategy_updates

    def _decay_exploration(self):
        """Decay exploration rate"""
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def _parse_optimization_response(self, response: str, original_prompt: str) -> Dict[str, Any]:
        """Parse optimization response with robust fallbacks"""
        try:
            # Try multiple JSON patterns
            json_patterns = [
                r'OPTIMIZATION:\s*(\{.*?\})',
                r'\{[^{}]*"optimized_prompt"[^{}]*\}',
                r'\{[^{}]*optimized_prompt[^{}]*\}'
            ]
            
            for pattern in json_patterns:
                json_match = re.search(pattern, response, re.DOTALL | re.IGNORECASE)
                if json_match:
                    json_str = json_match.group(1)
                    # Clean up the JSON string
                    json_str = re.sub(r'[\n\r\t]', ' ', json_str)
                    json_str = re.sub(r'\s+', ' ', json_str)
                    
                    parsed = json.loads(json_str)
                    if 'optimized_prompt' in parsed:
                        return parsed
        except Exception as e:
            pass
        
        # Fallback parsing - extract optimized prompt from response
        optimized_prompt = original_prompt
        confidence = 0.5
        
        # Extract optimized prompt from the response
        opt_patterns = [
            r'optimized_prompt[":\s]*"([^"]+)"',
            r'"optimized_prompt":\s*"([^"]+)"',
            r'optimized_prompt[":\s]*([^,\n}]+)',
        ]
        
        for pattern in opt_patterns:
            match = re.search(pattern, response, re.IGNORECASE)
            if match:
                extracted_prompt = match.group(1).strip()
                # Clean up the extracted prompt
                if extracted_prompt and extracted_prompt != original_prompt:
                    optimized_prompt = extracted_prompt
                    break
        
        # Extract confidence
        conf_match = re.search(r'confidence["\s:]*([0-9.]+)', response, re.IGNORECASE)
        if conf_match:
            try:
                confidence_str = conf_match.group(1)
                if confidence_str in ['.', '', 'nan']:
                    confidence = 0.7
                else:
                    confidence = max(0.0, min(1.0, float(confidence_str)))
            except (ValueError, AttributeError):
                confidence = 0.7
        
        return {
            "optimized_prompt": optimized_prompt,
            "confidence": confidence,
            "key_changes": ["parsed from unstructured response"],
            "expected_score": confidence,
            "learning_applied": []
        }

    def _query_llm(self, prompt: str) -> str:
        """Query the configured LLM via OpenRouter."""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": self.site_url,
            "X-Title": self.app_name,
        }
        data = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.7,
            "top_p": 0.9,
            "max_tokens": 350,
            "providers": [self.provider]
        }
        response = requests.post(f"{self.api_base_url}/chat/completions", headers=headers, json=data, timeout=60)
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"].strip()

    def _query_ollama(self, prompt: str) -> str:
        """Query local Ollama LLM"""
        data = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "stream": False,
            "options": {
                "temperature": 0.7,
                "num_predict": 350,
                "top_p": 0.9
            }
        }
        response = requests.post(f"{self.ollama_url}/api/chat", json=data, timeout=60)
        response.raise_for_status()
        return response.json()["message"]["content"].strip()

    def _save_memory(self):
        """Save RL loop memory"""
        try:
            data = {
                'strategy_performance': {
                    name: asdict(perf) for name, perf in self.strategy_performance.items()
                },
                'optimization_sessions': [
                    asdict(session) for session in self.optimization_sessions[-50:]
                ],
                'global_insights': self.global_insights[-100:],
                'epsilon': self.epsilon,
                'last_updated': time.time()
            }
            
            with open(self.memory_file, 'w') as f:
                json.dump(data, f, indent=2)
                
        except Exception as e:
            self.logger.warning(f"⚠️ Failed to save CLIP RL memory: {e}")

    def get_rl_insights(self) -> Dict[str, Any]:
        """Get RL learning insights"""
        if not self.optimization_sessions:
            return {"message": "No CLIP RL learning sessions yet"}
        
        recent_sessions = self.optimization_sessions[-10:]
        
        insights = {
            "total_rl_sessions": len(self.optimization_sessions),
            "current_exploration_rate": self.epsilon,
            "average_rounds_per_session": statistics.mean([s.total_rounds for s in recent_sessions]),
            "convergence_rate": len([s for s in recent_sessions if s.convergence_achieved]) / len(recent_sessions),
            "average_score_improvement": 0.0,
            "strategy_performance": []
        }
        
        # Calculate average improvement per session
        improvements = []
        for session in recent_sessions:
            if len(session.attempts) > 1:
                first_score = session.attempts[0].validation_score or 0
                best_score = session.final_best_score
                improvements.append(best_score - first_score)
        
        if improvements:
            insights["average_score_improvement"] = statistics.mean(improvements)
        
        # Add strategy performance
        sorted_strategies = sorted(
            self.strategy_performance.items(),
            key=lambda x: x[1].avg_score,
            reverse=True
        )
        
        for name, perf in sorted_strategies:
            if perf.total_attempts > 0:
                insights["strategy_performance"].append({
                    "strategy": name,
                    "avg_score": perf.avg_score,
                    "attempts": perf.total_attempts,
                    "success_rate": perf.success_count / perf.total_attempts,
                    "confidence": perf.confidence_in_strategy,
                    "improvement_trend": perf.improvement_trend
                })
        
        return insights

class CLIPScoreMaximizer:
    """Maximizes CLIP scores using iterative LLM feedback"""
    
    def __init__(self, 
                 dit_server_url: str = "http://localhost:8096",
                 max_iterations: int = 5,
                 target_score: float = 0.85,
                 min_improvement: float = 0.01):
        
        self.dit_server_url = dit_server_url
        self.max_iterations = max_iterations
        self.target_score = target_score
        self.min_improvement = min_improvement
        self.mutation_chance = 0.15  # 15% chance to try a creative mutation
        
        # Add these new properties for adaptive optimization
        self.keyword_scores = {}
        self.dynamic_adjectives = []
        
        # Initialize CLIP model
        self.clip_model = None
        self.clip_processor = None
        self.clip_tokenizer = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._model_loaded = False
        
        # Initialize attempt history
        self.attempt_history = []
        
        logger.info(f"🔧 CLIP Score Maximizer initialized")
        logger.info(f"   Max iterations: {max_iterations}")
        logger.info(f"   Target score: {target_score}")
        logger.info(f"   Min improvement: {min_improvement}")
    
    def load_clip_model(self):
        """Load a modern, high-performance CLIP model."""
        if self._model_loaded:
            return
        
        model_name = "ViT-H-14"
        pretrained_dataset = "laion2b_s32b_b79k"
        
        logger.info(f"📥 Loading advanced CLIP model: {model_name} ({pretrained_dataset})...")
        
        self.clip_model, _, self.clip_processor = open_clip.create_model_and_transforms(
            model_name, pretrained=pretrained_dataset, device=self.device
        )
        self.clip_tokenizer = open_clip.get_tokenizer(model_name)
        self.clip_model.eval()
        self._model_loaded = True
        logger.info("✅ Advanced CLIP model loaded.")
    
    def generate_dit_image(self, prompt: str, seed: Optional[int] = None, negative_prompt: str = "") -> Optional[str]:
        """Generate image with DiT server, now with negative prompt support for 3D-ready images."""
        if seed is None:
            seed = random.randint(0, 2**31 - 1)
        
        try:
            # Use form data instead of JSON for FastAPI Form endpoints
            payload = {
                "prompt": prompt,
                # "negative_prompt": negative_prompt,  # Add negative prompt to the payload
                "seed": seed,
                "num_inference_steps": 7,  # Slightly more steps for better detail
                "guidance_scale": 3.5       # Higher guidance scale for stricter prompt adherence
            }
            
            response = requests.post(
                f"{self.dit_server_url}", # /generate_image/tf2_style/
                data=payload,  # Use data instead of json for form data
                timeout=45     # Increase timeout slightly for more steps
            )
            
            if response.status_code == 200:
                result = response.json()
                return result.get('image')
            else:
                logger.warning(f"DiT server error: {response.status_code}")
                if response.status_code == 422:
                    logger.warning(f"Request validation error: {response.text}")
                return None
                
        except Exception as e:
            logger.warning(f"DiT generation failed: {e}")
            return None
    
    def generate_templated_variations(self, base_prompt: str, num_variations: int = 3) -> List[str]:
        """Generates reliable prompt variations using a weighted choice of keywords."""
        # Use dynamic adjectives if available, otherwise fall back to a static list
        default_adjectives = [
            "weathered", "polished", "intricate", "smooth", "textured", "layered",
            "stacked", "carved", "gleaming", "ornate", "masterfully crafted",
            "rough", "porous", "dense", "etched", "patterned", "interlocking",
            "jagged", "angular", "rounded", "refined", "luxurious", "elegant"
        ]
        adjectives = self.dynamic_adjectives if self.dynamic_adjectives else default_adjectives

        # Use keyword scores to influence adjective choice
        # Give unscored keywords a default weight of 1
        weights = [self.keyword_scores.get(adj, 1.0) for adj in adjectives]
        
        base_prompt = base_prompt.split(",")[0].strip()
        variations = []
        for _ in range(num_variations):
            try:
                num_adjectives = random.randint(1, 2)
                # Use weights to make the random choice
                selected_adjectives = random.choices(adjectives, weights=weights, k=num_adjectives)
                variation = f"{' '.join(selected_adjectives)} {base_prompt}"
                variations.append(variation)
            except IndexError:
                # Fallback if adjective list is too small
                selected_adjectives = random.sample(default_adjectives, 1)
                variations.append(f"{' '.join(selected_adjectives)} {base_prompt}")

        return list(set(variations))
    
    def compute_clip_score(self, prompt: str, image_base64: str) -> float:
        """Compute CLIP score between prompt and image"""
        if not self._model_loaded:
            self.load_clip_model()
        
        try:
            # Decode and preprocess image
            image_data = base64.b64decode(image_base64)
            image = Image.open(io.BytesIO(image_data))
            image_tensor = self.clip_processor(image).unsqueeze(0).to(self.device)
            
            # Tokenize prompt
            text_tokens = self.clip_tokenizer([prompt]).to(self.device)
            
            with torch.no_grad():
                # Encode
                image_features = self.clip_model.encode_image(image_tensor)
                text_features = self.clip_model.encode_text(text_tokens)
                
                # Normalize and compute similarity
                image_features /= image_features.norm(dim=-1, keepdim=True)
                text_features /= text_features.norm(dim=-1, keepdim=True)
                similarity = (image_features @ text_features.T).cpu().numpy()[0][0]
                
                return float(np.clip(similarity, 0, 1))
                
        except Exception as e:
            logger.warning(f"CLIP scoring failed: {e}")
            return 0.0
    
    def query_llm_for_improvement(self, original_prompt: str, attempt_history: List[Dict]) -> str:
        """Query LLM to improve prompt based on 3D-ready technical requirements."""
        
        system_prompt = f"""You are a Technical Director for a 3D modeling pipeline, creating prompts for a 'clay render' pass. The goal is an image that perfectly describes the object's geometry for 3D conversion.

### Core Principles:
1.  **Form Over Artistry:** The prompt must generate an image that looks like a high-quality, detailed clay sculpture.
2.  **Technical Language:** Use terms that describe physical shape and surface properties.
3.  **Neutral Presentation:** The object must be isolated on a neutral gray or white background.

### The Clay Render Prompt Formula:
`[Object Name], [Key Material], [Surface Texture], [Specific Geometric Detail], clay model, 3d sculpt, high poly, uniform neutral lighting, centered, product shot, neutral background`

### The Negative Prompt (Crucial for Clarity):
A strong negative prompt is essential to remove all unwanted details.
`shadows, artistic, noisy, text, watermark, color, textures, photo, realistic, blurry, grainy, complex background`

### Example:
-   **User Request:** "a stone wall"
-   **Your JSON Output:**
    ```json
    {{
      "optimized_prompt": "stone wall, carved granite, rough textured, interlocking blocks, clay model, 3d sculpt, high poly, uniform neutral lighting, centered, product shot, neutral background",
      "negative_prompt": "shadows, artistic, noisy, text, watermark, color, textures, photo, realistic, blurry, grainy, complex background",
      "reasoning": "Specified granite material and interlocking geometry. The prompt is tailored for a clay render to maximize geometric clarity."
    }}
    ```

### Final Instruction:
Return your response in this exact JSON format. The `negative_prompt` is REQUIRED.
```json
{{
  "optimized_prompt": "your new prompt following the formula",
  "negative_prompt": "your negative prompt for a clean clay render",
  "reasoning": "brief explanation of technical choices"
}}
```

**CRITICAL:** 
- Return ONLY valid JSON, no other text
- The "optimized_prompt" should be the final prompt without quotes
- The "negative_prompt" is REQUIRED and must include the standard elements
- "reasoning" should be a brief explanation of what changes were made
"""

        user_prompt = f"Original idea: '{original_prompt}'"
        if len(attempt_history) > 1:
            best_attempt = max(attempt_history, key=lambda x: x['score'])
            user_prompt += f"\nBest attempt so far had a score of {best_attempt['score']:.4f} with prompt: '{best_attempt['prompt']}'. Improve upon this with better technical specifications."

        try:
            logger.debug(f"    Calling _query_ollama...")
            raw_response = self._query_ollama(system_prompt, user_prompt)
            logger.debug(f"    _query_ollama returned: {raw_response[:100]}...")
            # Parse the structured response
            return self._parse_structured_response(raw_response)
        except Exception as e:
            logger.warning(f"LLM query failed: {e}")
            return ""
    
    def get_dynamic_adjectives(self, base_prompt: str):
        """Query the LLM to get a list of adjectives relevant to the base prompt."""
        logger.info("    Generating dynamic adjectives for templates...")
        system_prompt = """You are an expert prompt component generator. Generate a list of 20-30 descriptive adjectives for a given object. Focus on material, texture, shape, and style. Return ONLY a JSON object like this: {"adjectives": ["word1", "word2", "word3"]}"""
        user_prompt = f"Generate adjectives for: '{base_prompt}'"
        
        try:
            response = self._query_ollama(system_prompt, user_prompt)
            logger.debug(f"    Raw LLM response for adjectives: {response[:200]}...")
            
            # Try to extract JSON from the response
            import re
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
                # Fix double curly braces that the LLM might return
                json_str = json_str.replace('{{', '{').replace('}}', '}')
                parsed_response = json.loads(json_str)
                self.dynamic_adjectives = parsed_response.get("adjectives", [])
            else:
                # Fallback: extract adjectives from conversational text
                logger.warning("    JSON not found, extracting adjectives from text...")
                words = response.lower().split()
                # Look for common adjective patterns
                adjectives = []
                for word in words:
                    word = word.strip('.,!?()[]{}":;')
                    if len(word) > 3 and word.isalpha():
                        # Simple heuristic: if it looks like an adjective, include it
                        if word not in ['the', 'and', 'or', 'but', 'for', 'with', 'from', 'this', 'that', 'these', 'those', 'here', 'are', 'descriptive', 'adjectives', 'given', 'object', 'focus', 'material', 'texture', 'shape', 'style', 'return', 'only', 'json', 'object', 'like', 'this']:
                            adjectives.append(word)
                self.dynamic_adjectives = adjectives[:25]  # Limit to 25 adjectives
            
            if self.dynamic_adjectives:
                logger.info(f"    Generated {len(self.dynamic_adjectives)} dynamic adjectives (e.g., {self.dynamic_adjectives[:3]})")
            else:
                logger.warning("    No dynamic adjectives extracted, will use static list.")
        except Exception as e:
            logger.warning(f"    Failed to get dynamic adjectives: {e}. Using static list.")
            self.dynamic_adjectives = []
    
    def _extract_patterns(self, prompts: List[str]) -> Dict[str, int]:
        """Extract common patterns from a list of prompts, focusing on the winning formula"""
        patterns = {}
        
        # Core winning formula components based on RL training
        formula_components = {
            "adjectives": [
                "sleek", "intricate", "classic", "glowing", "radiant", "delicate",
                "slender", "ornate", "weathered", "polished", "smooth", "textured",
                "luxurious", "elegant", "refined", "masterfully", "breathtakingly"
            ],
            "colors": [
                "blue", "green", "black", "yellow", "red", "white", "silver", "gold",
                "turquoise", "sapphire", "emerald", "ruby", "bronze", "copper"
            ],
            "features": [
                "pointed tip", "curved handle", "ornate frame", "intricate filigree",
                "radiant stone", "gleaming surface", "ethereal glow", "catches light",
                "suspended", "surrounded by", "topped with", "set against"
            ],
            "contexts": [
                "serene lake", "velvet cushion", "soft glow", "white background",
                "studio setting", "elegant scene", "thematic environment"
            ]
        }
        
        for prompt in prompts:
            prompt_lower = prompt.lower()
            
            # Check for formula components
            for category, terms in formula_components.items():
                for term in terms:
                    if term in prompt_lower:
                        key = f"{category}:{term}"
                        patterns[key] = patterns.get(key, 0) + 1
            
            # Check for winning formula structure
            if "with" in prompt_lower:
                patterns["formula_structure:with"] = patterns.get("formula_structure:with", 0) + 1
            
            # Check for brevity (5-12 words)
            word_count = len(prompt.split())
            if 5 <= word_count <= 12:
                patterns["brevity:optimal_length"] = patterns.get("brevity:optimal_length", 0) + 1
            elif word_count > 12:
                patterns["brevity:too_long"] = patterns.get("brevity:too_long", 0) + 1
        
        return patterns
    
    def _format_patterns(self, patterns: Dict[str, int]) -> str:
        """Format patterns for LLM prompt"""
        if not patterns:
            return "- No clear patterns identified"
        
        sorted_patterns = sorted(patterns.items(), key=lambda x: x[1], reverse=True)
        return "\n".join([f"- '{pattern}': {count} occurrences" for pattern, count in sorted_patterns[:10]])
    
    def _query_ollama(self, system_prompt: str, user_prompt: str) -> str:
        """Query Ollama LLM and return raw response"""
        try:
            cmd = [
                "ollama", "run", "llama2", 
                f"System: {system_prompt}\n\nUser: {user_prompt}\n\nAssistant:"
            ]
            
            result = subprocess.run(
                cmd, 
                capture_output=True, 
                text=True, 
                timeout=30
            )
            
            if result.returncode == 0:
                response = result.stdout.strip()
                logger.debug(f"   Raw LLM response: {response}")
                return response
            else:
                raise Exception(f"Ollama command failed: {result.stderr}")
                
        except Exception as e:
            logger.warning(f"Ollama query failed: {e}")
            raise e
    
    def _parse_structured_response(self, response: str) -> str:
        """Parse structured JSON response from LLM that includes negative_prompt"""
        try:
            # Try to extract JSON from the response
            import json
            import re
            
            # Look for JSON pattern in the response
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
                
                # Fix double curly braces that the LLM might return
                json_str = json_str.replace('{{', '{').replace('}}', '}')
                
                parsed = json.loads(json_str)
                
                if 'optimized_prompt' in parsed:
                    optimized_prompt = parsed['optimized_prompt']
                    negative_prompt = parsed.get('negative_prompt', 'blurry, shadows, artistic, grainy, low-resolution')
                    reasoning = parsed.get('reasoning', 'No reasoning provided')
                    
                    logger.info(f"   Optimized prompt: {optimized_prompt}")
                    logger.info(f"   Negative prompt: {negative_prompt}")
                    logger.info(f"   Reasoning: {reasoning}")
                    
                    # Return a special format that includes both prompts
                    return json.dumps({
                        'optimized_prompt': optimized_prompt,
                        'negative_prompt': negative_prompt,
                        'reasoning': reasoning
                    })
            
            # Fallback: if JSON parsing fails, try to extract just the prompt
            logger.warning("   JSON parsing failed, falling back to text extraction")
            return self._fallback_text_extraction(response)
            
        except Exception as e:
            logger.warning(f"   Structured parsing failed: {e}")
            logger.warning(f"   Response was: {response[:200]}...")
            return self._fallback_text_extraction(response)
    
    def _fallback_text_extraction(self, response: str) -> str:
        """Fallback method to extract prompt from unstructured response"""
        # Simple extraction - take the first reasonable line that looks like a prompt
        lines = response.split('\n')
        for line in lines:
            line = line.strip()
            if line and len(line.split()) <= 15 and not any(word in line.lower() for word in ['here', 'optimized', 'prompt', 'task', 'analysis', 'json']):
                return line
        
        # If no good line found, return a simple fallback
        return "optimized prompt"
    
    def maximize_clip_score(self, original_prompt: str, seed: Optional[int] = None) -> Dict:
        """Maximize CLIP score through intelligent multi-candidate optimization"""
        start_time = time.time()
        
        logger.info(f"🚀 Starting CLIP score maximization for: '{original_prompt}'")
        logger.info(f"   📝 Original prompt will be used for CLIP scoring")
        logger.info(f"   🎨 Optimized prompts will be used for image generation")

        best_prompt = original_prompt
        best_score = 0.0
        best_image = None
        best_negative_prompt = "blurry, shadows, artistic, grainy, low-resolution"
        consecutive_failures = 0  # Track consecutive failures for self-correction
        
        # Reset attempt history
        self.attempt_history = []
        
        # --- New: Get Dynamic Adjectives at the Start ---
        self.get_dynamic_adjectives(original_prompt)

        # Initial evaluation
        logger.info("📊 Evaluating original prompt...")
        initial_image = self.generate_dit_image(original_prompt, seed, negative_prompt=best_negative_prompt)
        if initial_image:
            initial_score = self.compute_clip_score(original_prompt, initial_image)
            logger.info(f"   Original CLIP score: {initial_score:.4f}")
            best_score = initial_score
            best_image = initial_image
            
            self.attempt_history.append({
                'prompt': original_prompt,  # For initial evaluation, both are the same
                'original_prompt': original_prompt,  # For consistency
                'score': initial_score,
                'iteration': 0,
                # 'image': initial_image
            })
        
        # --- Main Optimization Loop ---
        for iteration in range(1, self.max_iterations + 1):
            logger.info(f"\n🔄 Iteration {iteration}/{self.max_iterations} (Current Best Score: {best_score:.4f})")

            # --- Step 1: Generate Candidates ---
            candidates = []
            
            # New: Creative Mutation Step
            if random.random() < self.mutation_chance:
                logger.info("    💥 Performing a creative mutation...")
                mutation_system_prompt = """You are a creative Technical Director for 3D modeling. Generate a completely new, technical prompt for 3D asset creation based on an idea. Focus on a fresh but geometrically clear approach.

You MUST return ONLY a JSON object in this exact format:
```json
{
  "optimized_prompt": "your new technical prompt here",
  "negative_prompt": "shadows, artistic, noisy, text, watermark, color, textures, photo, realistic, blurry, grainy, complex background"
}
```

Do not include any explanations, just the JSON object."""
                try:
                    mutation_response = self._query_ollama(mutation_system_prompt, f"Original idea: {original_prompt}")
                    if mutation_response:
                        # Parse the mutation response
                        try:
                            # Extract JSON from the response
                            import re
                            json_match = re.search(r'\{.*\}', mutation_response, re.DOTALL)
                            if json_match:
                                json_str = json_match.group(0).replace('{{', '{').replace('}}', '}')
                                mutation_json = json.loads(json_str)
                                candidates.append(mutation_json)
                                logger.info(f"    Generated mutation: '{mutation_json.get('optimized_prompt', '')}'")
                            else:
                                logger.warning("    Mutation response did not contain valid JSON")
                        except Exception as parse_error:
                            logger.warning(f"    Mutation parsing failed: {parse_error}")
                            logger.debug(f"    Raw mutation response: {mutation_response[:200]}...")
                except Exception as e:
                    logger.warning(f"    Mutation failed: {e}")

            # Add LLM-based suggestion for refinement
            try:
                refined_suggestion = self.query_llm_for_improvement(original_prompt, self.attempt_history)
                if refined_suggestion:
                    # Parse the refined suggestion
                    try:
                        refined_json = json.loads(refined_suggestion.replace('{{', '{').replace('}}', '}'))
                        candidates.append(refined_json)
                        logger.info(f"    Generated refinement: '{refined_json.get('optimized_prompt', '')}'")
                    except:
                        logger.warning("    Refinement parsing failed")
            except Exception as e:
                logger.warning(f"    Refinement suggestion failed: {e}")

            # Add templated variations (these will have default negative prompts)
            templated_variations = self.generate_templated_variations(best_prompt, num_variations=4)
            for variation in templated_variations:
                candidates.append({
                    'optimized_prompt': variation,
                    'negative_prompt': 'blurry, shadows, artistic, grainy, low-resolution, complex background, photograph, realistic, noisy, text, watermark'
                })
            logger.info(f"    Generated {len(templated_variations)} templated variations")
            
            # --- New: Sanity Check on Candidates ---
            def sanity_check(prompt):
                """Check if the prompt still contains the original subject"""
                if not prompt:
                    return False
                original_words = original_prompt.lower().split()
                prompt_lower = prompt.lower()
                # Check if any of the original words are in the prompt
                return any(word in prompt_lower for word in original_words if len(word) > 2)
            
            # Remove duplicates and prompts we've already tried, plus sanity check
            seen_prompts = {a['prompt'] for a in self.attempt_history}
            unique_candidates = [c for c in candidates if c.get('optimized_prompt') and 
                               c.get('optimized_prompt') not in seen_prompts and 
                               sanity_check(c.get('optimized_prompt'))]
            
            if not unique_candidates:
                logger.warning("    No valid new candidates after sanity check, stopping optimization.")
                break

            logger.info(f"    Testing {len(unique_candidates)} new candidate prompts...")

            # --- Step 2: Evaluate All Candidates ---
            iteration_results = []
            for i, candidate_data in enumerate(unique_candidates):
                prompt_candidate = candidate_data.get('optimized_prompt', '')
                negative_prompt = candidate_data.get('negative_prompt', 'blurry, shadows, artistic, grainy, low-resolution')
                
                logger.info(f"      - Testing candidate {i+1}/{len(unique_candidates)}: '{prompt_candidate}'")
                logger.info(f"        Negative: '{negative_prompt}'")
                
                try:
                    image = self.generate_dit_image(prompt_candidate, seed, negative_prompt=negative_prompt)
                    if not image:
                        logger.warning("        Image generation failed, skipping candidate.")
                        continue
                    
                    # Compute CLIP score between original prompt and the image generated from optimized prompt
                    score = self.compute_clip_score(original_prompt, image)
                    logger.info(f"        CLIP Score (original prompt vs optimized image): {score:.4f}")
                    
                    iteration_results.append({
                        'prompt': prompt_candidate, 
                        'score': score, 
                        'image': image,
                        'negative_prompt': negative_prompt
                    })
                except Exception as e:
                    logger.warning(f"        Failed to evaluate candidate: {e}")
                    continue

            if not iteration_results:
                logger.warning("    No candidates were successfully evaluated in this iteration.")
                consecutive_failures += 1
                continue

            # --- Step 3: Select the Best Candidate and Update History ---
            best_of_iteration = max(iteration_results, key=lambda x: x['score'])
            
            # Add all evaluated attempts to history
            for result in iteration_results:
                self.attempt_history.append({
                    'prompt': result['prompt'],  # This is the optimized prompt used for generation
                    'original_prompt': original_prompt,  # This is the original prompt used for scoring
                    'score': result['score'],
                    'iteration': iteration,
                    'image': result['image']
                })
            
            # Check if the best of this iteration is better than the overall best score
            if best_of_iteration['score'] > best_score:
                improvement = best_of_iteration['score'] - best_score
                best_score = best_of_iteration['score']
                best_prompt = best_of_iteration['prompt']
                best_image = best_of_iteration['image']
                best_negative_prompt = best_of_iteration['negative_prompt']
                logger.info(f"    🏆 New best score found in iteration {iteration}: {best_score:.4f} (+{improvement:.4f})")
                consecutive_failures = 0  # Reset failure counter on success
            else:
                logger.info(f"    ⏸️  No improvement in this iteration. Best score remains {best_score:.4f}.")
                consecutive_failures += 1  # Increment failure counter

            # --- New: Automatic Reset Logic ---
            if consecutive_failures >= 3:
                logger.warning("    🚨 Score has not improved for 3 consecutive iterations. Performing a reset.")
                # Revert to the best known prompt before the decline
                best_attempt_before_decline = max(self.attempt_history[:-len(iteration_results)], key=lambda x: x['score'])
                best_prompt = best_attempt_before_decline['prompt']
                best_score = best_attempt_before_decline['score']
                logger.warning(f"    ⏪ Resetting to best prompt: '{best_prompt}' (Score: {best_score:.4f})")
                # Force a strong mutation on the next iteration
                self.mutation_chance = 1.0 
                consecutive_failures = 0  # Reset counter after action
            else:
                self.mutation_chance = 0.15  # Reset mutation chance if not resetting

            # --- New: Step 4: Update Keyword Scores ---
            logger.info("    Updating keyword scores based on iteration results...")
            base_words = set(original_prompt.split())
            for result in iteration_results:
                added_words = set(result['prompt'].split()) - base_words
                # Reward improvement, penalize regression
                score_delta = result['score'] - best_score 
                for word in added_words:
                    if word in (self.dynamic_adjectives or []):  # Only track our adjectives
                        # A simple learning rate of 0.1
                        update = 0.1 * score_delta
                        self.keyword_scores[word] = self.keyword_scores.get(word, 1.0) + update
                        # Ensure scores don't go below a minimum threshold
                        self.keyword_scores[word] = max(0.1, self.keyword_scores[word])
            
            # Log top performing keywords
            if self.keyword_scores:
                top_keywords = sorted(self.keyword_scores.items(), key=lambda x: x[1], reverse=True)[:5]
                logger.info(f"    Top performing keywords: {top_keywords}")

            # Check for convergence (only after minimum iterations)
            try:
                if iteration >= 3:  # Require at least 3 iterations before convergence
                    # Compare current best score against the best score from previous iterations
                    previous_best = max(a['score'] for a in self.attempt_history[:-len(iteration_results)])
                    recent_improvement = best_score - previous_best
                    
                    if recent_improvement < self.min_improvement:
                        logger.info(f"    ⏸️  Minimal improvement over previous best ({recent_improvement:.4f}), stopping early")
                        break
            except Exception as e:
                logger.error(f"    ❌ Error checking convergence in iteration {iteration}: {e}")
                import traceback
                logger.error(f"    Full traceback: {traceback.format_exc()}")
        
        optimization_time = time.time() - start_time
        # Calculate improvement percentage safely
        if self.attempt_history and self.attempt_history[0]['score'] > 0:
            improvement_percent = ((best_score - self.attempt_history[0]['score']) / 
                                  self.attempt_history[0]['score'] * 100)
        else:
            improvement_percent = 0.0
        
        logger.info(f"\n✅ CLIP score maximization completed!")
        logger.info(f"   Original score: {self.attempt_history[0]['score']:.4f}")
        logger.info(f"   Best score: {best_score:.4f} (+{improvement_percent:.1f}%)")
        logger.info(f"   Best generation prompt: '{best_prompt}'")
        logger.info(f"   Scoring prompt: '{original_prompt}'")
        logger.info(f"   Best negative prompt: '{best_negative_prompt}'")
        logger.info(f"   Total time: {optimization_time:.2f}s")
        logger.info(f"   Iterations: {len(self.attempt_history) - 1}")
        
        # Log final keyword performance
        if self.keyword_scores:
            logger.info(f"   Final keyword scores: {dict(sorted(self.keyword_scores.items(), key=lambda x: x[1], reverse=True)[:10])}")
        
        return {
            'original_prompt': original_prompt,
            'best_prompt': best_prompt,
            'best_negative_prompt': best_negative_prompt,
            'original_score': self.attempt_history[0]['score'],
            'best_score': best_score,
            'improvement_percent': improvement_percent,
            'iterations': len(self.attempt_history) - 1,
            'optimization_time': optimization_time,
            'attempt_history': self.attempt_history,
            'best_image': best_image,
            'keyword_scores': self.keyword_scores,
            'dynamic_adjectives': self.dynamic_adjectives
        }
    
    def cleanup(self):
        """Cleanup resources"""
        if self._model_loaded:
            del self.clip_model, self.clip_tokenizer, self.clip_processor
            self.clip_model = self.clip_tokenizer = self.clip_processor = None
            self._model_loaded = False
            if self.device.type == 'cuda':
                torch.cuda.empty_cache()
    
    def maximize_clip_score_with_rl(self, original_prompt: str, seed: Optional[int] = None) -> Dict:
        """Maximize CLIP score using RL learning loop"""
        logger.info(f"🚀 Starting CLIP score maximization with RL learning for: '{original_prompt}'")
        
        # Initialize RL agent
        rl_agent = RLLoopAgent(self)
        
        # Run RL optimization
        result = rl_agent.optimize_with_rl_loop(original_prompt, seed)
        
        # Add RL insights to result
        rl_insights = rl_agent.get_rl_insights()
        result['rl_insights'] = rl_insights
        
        logger.info(f"✅ RL-based CLIP score maximization completed!")
        logger.info(f"   Final optimized prompt: '{result['final_optimized_prompt']}'")
        logger.info(f"   Final score: {result['final_score']:.4f}")
        logger.info(f"   Total rounds: {result['total_rounds']}")
        logger.info(f"   Convergence achieved: {result['convergence_achieved']}")
        
        return result

def main():
    """Main function"""
    parser = argparse.ArgumentParser(
        description="Maximize CLIP score for a prompt using traditional optimization or RL learning loop",
        epilog="""
Examples:
  # Traditional optimization
  python get_max_clip_score.py "a red car"
  
  # RL learning loop optimization
  python get_max_clip_score.py "a red car" --rl-mode
  
  # Show RL learning insights
  python get_max_clip_score.py dummy --insights
  
  # RL mode with custom parameters
  python get_max_clip_score.py "a red car" --rl-mode --seed 42 --save-results results.json
        """
    )
    parser.add_argument("prompt", help="Input prompt to optimize (use 'dummy' with --insights)")
    parser.add_argument("--dit-server", default="http://localhost:8096", help="DiT server URL")
    parser.add_argument("--max-iterations", type=int, default=5, help="Maximum optimization iterations")
    parser.add_argument("--target-score", type=float, default=0.85, help="Target CLIP score")
    parser.add_argument("--min-improvement", type=float, default=0.01, help="Minimum improvement threshold")
    parser.add_argument("--mutation-chance", type=float, default=0.15, help="Probability of creative mutations (0.0-1.0)")
    parser.add_argument("--seed", type=int, help="Random seed for reproducibility")
    parser.add_argument("--save-results", help="Save results to JSON file")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    parser.add_argument("--log-file", help="Save logs to file")
    parser.add_argument("--rl-mode", action="store_true", help="Use RL learning loop instead of traditional optimization")
    parser.add_argument("--insights", action="store_true", help="Show RL learning insights")
    
    args = parser.parse_args()
    
    # Set up logging
    log_format = '%(asctime)s | %(levelname)-8s | %(name)s:%(funcName)s:%(lineno)d - %(message)s'
    
    if args.debug:
        log_level = logging.DEBUG
        print("🔍 Debug mode enabled")
    else:
        log_level = logging.INFO
    
    # Configure logging
    logging.basicConfig(
        level=log_level,
        format=log_format,
        handlers=[
            logging.StreamHandler(),  # Console output
        ]
    )
    
    # Add file handler if log file specified or create default
    if args.log_file:
        log_file_path = args.log_file
    else:
        # Create default log file with timestamp
        import datetime
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file_path = f"clip_optimizer_{timestamp}.log"
    
    file_handler = logging.FileHandler(log_file_path)
    file_handler.setLevel(log_level)
    file_handler.setFormatter(logging.Formatter(log_format))
    logging.getLogger().addHandler(file_handler)
    print(f"📝 Logs will be saved to: {log_file_path}")
    
    # Handle special commands
    if args.insights:
        # Show RL insights
        try:
            rl_agent = RLLoopAgent(CLIPScoreMaximizer(dit_server_url=args.dit_server))
            insights = rl_agent.get_rl_insights()
            print("\n🔄 CLIP RL LEARNING INSIGHTS:")
            print("=" * 50)
            print(json.dumps(insights, indent=2))
            return
        except Exception as e:
            print(f"\n❌ Error getting RL insights: {e}")
            return
    
    # Initialize maximizer
    maximizer = CLIPScoreMaximizer(
        dit_server_url=args.dit_server,
        max_iterations=args.max_iterations,
        target_score=args.target_score,
        min_improvement=args.min_improvement
    )
    
    # Set mutation chance from command line
    maximizer.mutation_chance = args.mutation_chance
    
    try:
        # Choose optimization mode
        if args.rl_mode:
            print("🔄 CLIP RL LEARNING MODE")
            print("=" * 60)
            print("✅ Multi-round optimization with CLIP score feedback")
            print("✅ Strategy performance updates based on results")
            print("✅ Convergence detection and early stopping")
            print("✅ Exploration/exploitation based on performance")
            print("=" * 60)
            
            # Use RL learning loop
            result = maximizer.maximize_clip_score_with_rl(args.prompt, seed=args.seed)
        else:
            # Use traditional optimization
            result = maximizer.maximize_clip_score(args.prompt, seed=args.seed)
        
        # Print summary based on mode
        if args.rl_mode:
            print(f"\n📊 CLIP RL LOOP RESULTS")
            print(f"=" * 50)
            print(f"Original prompt: '{result['original_prompt']}'")
            print(f"Final optimized prompt: '{result['final_optimized_prompt']}'")
            print(f"Initial score: {result['initial_score']:.4f}")
            print(f"Final score: {result['final_score']:.4f}")
            print(f"Total rounds: {result['total_rounds']}")
            print(f"Convergence achieved: {result['convergence_achieved']}")
            print(f"Convergence reason: {result['convergence_reason']}")
            print(f"Exploration ratio: {result['exploration_ratio']:.1%}")
            print(f"Score progression: {[f'{s:.3f}' for s in result['score_progression']]}")
            print(f"Strategy sequence: {result['strategy_sequence']}")
            print(f"Processing time: {result['processing_time']:.2f}s")
            
            # Show RL insights
            if result.get('rl_insights'):
                insights = result['rl_insights']
                print(f"\n🧠 RL LEARNING INSIGHTS:")
                print(f"   Total RL sessions: {insights.get('total_rl_sessions', 0)}")
                print(f"   Current exploration rate: {insights.get('current_exploration_rate', 0):.2f}")
                
                # Handle cases where there might not be enough data
                avg_rounds = insights.get('average_rounds_per_session', 0)
                if avg_rounds > 0:
                    print(f"   Average rounds per session: {avg_rounds:.1f}")
                
                convergence_rate = insights.get('convergence_rate', 0)
                if convergence_rate > 0:
                    print(f"   Convergence rate: {convergence_rate:.1%}")
                
                avg_improvement = insights.get('average_score_improvement', 0)
                if avg_improvement != 0:
                    print(f"   Average score improvement: {avg_improvement:.3f}")
                
                # Show top strategies
                strategies = insights.get('strategy_performance', [])
                if strategies:
                    print(f"\n🏆 TOP STRATEGIES:")
                    for i, strategy in enumerate(strategies[:5]):
                        print(f"   {i+1}. {strategy['strategy']}: {strategy['avg_score']:.3f} (confidence: {strategy['confidence']:.2f})")
        else:
            print(f"\n📊 TRADITIONAL OPTIMIZATION RESULTS")
            print(f"=" * 50)
            print(f"Original prompt (for scoring): '{result['original_prompt']}'")
            print(f"Best generation prompt: '{result['best_prompt']}'")
            print(f"Best negative prompt: '{result['best_negative_prompt']}'")
            print(f"Original score: {result['original_score']:.4f}")
            print(f"Best score: {result['best_score']:.4f}")
            print(f"Improvement: +{result['improvement_percent']:.1f}%")
            print(f"Iterations: {result['iterations']}")
            print(f"Time: {result['optimization_time']:.2f}s")
            
            # Show advanced features info
            if result.get('keyword_scores'):
                top_keywords = sorted(result['keyword_scores'].items(), key=lambda x: x[1], reverse=True)[:5]
                print(f"\n🏆 Top performing keywords: {dict(top_keywords)}")
            
            if result.get('dynamic_adjectives'):
                print(f"🎯 Dynamic adjectives generated: {len(result['dynamic_adjectives'])} words")
                print(f"   Examples: {result['dynamic_adjectives'][:5]}")
        
        # Save results if requested
        if args.save_results:
            # Remove image data for JSON serialization
            save_data = result.copy()
            
            # Handle different result formats (traditional vs RL mode)
            if 'best_image' in save_data:
                save_data['best_image'] = save_data['best_image'][:100] + "..." if save_data['best_image'] else None
            if 'attempt_history' in save_data:
                for attempt in save_data['attempt_history']:
                    attempt['image'] = attempt['image'][:100] + "..." if attempt['image'] else None
            
            with open(args.save_results, 'w') as f:
                json.dump(save_data, f, indent=2)
            print(f"\n💾 Results saved to {args.save_results}")
        
    except KeyboardInterrupt:
        print("\n🛑 Optimization interrupted by user")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        sys.exit(1)
    finally:
        maximizer.cleanup()

if __name__ == "__main__":
    main() 