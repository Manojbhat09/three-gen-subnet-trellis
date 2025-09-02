#!/usr/bin/env python3
"""
Smart Prompt Optimizer V4.1 - TRUE RL LEARNING LOOP
===================================================
🔄 Real RL loop: Agent learns through iterative optimization cycles
🎯 Score-driven learning: Agent adjusts strategies based on validation feedback
🧠 Multi-round conversations with score-based strategy updates
⚡ Continuous improvement through exploration and exploitation
🎓 Learns principles from score patterns across multiple attempts

Revolutionary RL Loop:
1. Agent makes optimization attempt
2. Gets validation score feedback  
3. Reflects on what worked/didn't work
4. Updates strategy preferences based on scores
5. Makes improved attempt using learned insights
6. Repeats until convergence or max rounds

front view, accurate, complete, white background
"""

import json
import requests
import time
import sys
import random
import re
from typing import Dict, List, Optional, Tuple, Any, Union
from pathlib import Path
from dataclasses import dataclass, asdict
import statistics
import subprocess
from datetime import datetime
import torch
import logging
import os

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
    # Try to find "Original Prompt: ..." at the end
    match = re.search(r'Original Prompt:\s*(.*)', original_prompt)
    if match:
        return match.group(1).strip()
    # Otherwise, just return the string as-is
    return original_prompt.strip()

class RLLoopAgent:
    """RL agent that learns through iterative optimization loops (with improved convergence and exploration)"""
    
    def __init__(self, ollama_url: str = "http://localhost:11434",
                 memory_file: str = "rl_loop_memory.json",
                 api_base_url: str = "https://openrouter.ai/api/v1",
                 trellis_server_url_w_port: str = "http://localhost:8096",
                 endpoint: str = "generate/",
                 ollama_coordinator=None,  # New parameter
                 ollama_model: str = None,
                 port: int = 8096,  # Add port parameter
                 use_vllm: bool = False,  # New parameter for vLLM
                 vllm_url: str = "http://localhost:9000",  # New parameter for vLLM URL
                 vllm_model: str = "llama-3-2-3b-it",  # New parameter for vLLM model
                 disable_convergence: bool = False,  # New parameter to disable convergence
                 use_separated_validation: bool = False,  # New parameter for separated validation
                 cache_models: bool = False):  # New parameter for model caching
        self.ollama_url = ollama_url
        if ollama_model:
            self.model = ollama_model
        # self.model = None  # Will be set based on user input
        self.memory_file = Path(memory_file)
        self.api_base_url = api_base_url
        self.use_openrouter = False
        self.use_vllm = use_vllm  # Store vLLM preference
        self.vllm_url = vllm_url  # Store vLLM URL
        self.vllm_model = vllm_model  # Store vLLM model
        self.disable_convergence = disable_convergence  # Store convergence preference
        self.use_separated_validation = use_separated_validation  # Store separated validation preference
        self.cache_models = cache_models  # Store model caching preference
        self.provider = None
        self.api_key = None
        self.site_url = "http://localhost"
        self.app_name = "RL Prompt Optimizer"
        self.trellis_server_url_w_port = trellis_server_url_w_port
        self.ollama_coordinator = ollama_coordinator  # Store coordinator reference
        self.port = port  # Store port for validation calls
        self._choose_llm_provider()
        self.endpoint = endpoint
        # RL Loop parameters (improved)
        self.max_optimization_rounds = 15
        self.min_rounds_before_convergence = 5  # NEW: Minimum rounds before convergence
        self.convergence_threshold = 0.01  # Reduced for more sensitivity
        self.min_score_threshold = 0.85   # Target score to achieve
        self.explore_performance_threshold = 0.05  # If explore is 5% better, increase exploration
        self.convergence_improvement_threshold = 0.1  # Minimum improvement to allow convergence

        # Learning state
        self.strategy_performance: Dict[str, StrategyPerformance] = {}
        self.optimization_sessions: List[RLOptimizationSession] = []
        self.global_insights: List[str] = []

        # Improved RL parameters
        self.epsilon = 0.6  # Increased from 0.4 based on analysis
        self.epsilon_decay = 0.98  # Slower decay
        self.epsilon_min = 0.3  # Higher minimum
        self.explore_scores: List[float] = []
        self.exploit_scores: List[float] = []

        self.logger = logging.getLogger(__name__)
        self._load_memory()
        self._initialize_strategies()
        
        # Print LLM provider information prominently
        print("\n" + "="*60)
        print("🤖 LLM PROVIDER CONFIGURATION")
        print("="*60)
        if self.use_vllm:
            print(f"✅ Using vLLM: {self.vllm_url}")
            print(f"   Model: {self.vllm_model}")
            print(f"   Status: ACTIVE")
        elif self.use_openrouter:
            print(f"✅ Using OpenRouter: {self.api_base_url}")
            print(f"   Model: {self.model}")
            print(f"   Provider: {self.provider}")
            print(f"   Status: ACTIVE")
        else:
            print(f"✅ Using Ollama: {self.ollama_url}")
            print(f"   Model: {self.model}")
            print(f"   Status: ACTIVE")
        print("="*60)
        
        self.logger.info(f"🔄 RL LOOP AGENT INITIALIZED (IMPROVED)")
        self.logger.info(f"   Strategy tracking: {len(self.strategy_performance)} strategies")
        self.logger.info(f"   Past sessions: {len(self.optimization_sessions)}")
        self.logger.info(f"   Exploration rate: {self.epsilon:.2f}")
        self.logger.info(f"   Max rounds per optimization: {self.max_optimization_rounds}")
        self.logger.info(f"   Min rounds before convergence: {self.min_rounds_before_convergence}")
        self.logger.info(f"   Adaptive convergence: {'DISABLED - Force all rounds' if self.disable_convergence else 'ENABLED'}")
        self.logger.info(f"   Validation method: {'SEPARATED (two-stage)' if self.use_separated_validation else 'DIRECT (single-stage)'}")
        self.logger.info(f"   Model caching: {'ENABLED (uses more GPU memory)' if self.cache_models else 'DISABLED (safer for GPU memory)'}")
        if self.use_vllm:
            self.logger.info(f"   Using vLLM: {self.vllm_url} with model {self.vllm_model}")
        elif self.use_openrouter:
            self.logger.info(f"   Using OpenRouter: {self.api_base_url} with model {self.model}")
        else:
            self.logger.info(f"   Using Ollama: {self.ollama_url} with model {self.model}")
    
    def _choose_llm_provider(self):
        # If vLLM is specified, skip the interactive choice
        if self.use_vllm:
            self.model = self.vllm_model
            print(f"\n🚀 AUTO-CONFIGURED: Using vLLM model: {self.model} at {self.vllm_url}")
            return
            
        print("\nWhich LLM provider do you want to use?")
        print("1. Local Ollama (default: llama3.2:3b)")
        print("2. OpenRouter (cloud, supports many models)")
        print("3. vLLM (fast local inference)")
        choice = input("Enter 1 for Ollama, 2 for OpenRouter, or 3 for vLLM: ").strip()
        if choice == "3":
            self.use_vllm = True
            self.vllm_url = input("Enter vLLM server URL (default: http://localhost:9000): ").strip()
            if not self.vllm_url:
                self.vllm_url = "http://localhost:9000"
            self.vllm_model = input("Enter vLLM model name (default: llama-3-2-3b-it): ").strip()
            if not self.vllm_model:
                self.vllm_model = "llama-3-2-3b-it"
            self.model = self.vllm_model
            print(f"🚀 CONFIGURED: Using vLLM: {self.vllm_url} with model {self.vllm_model}")
        elif choice == "2":
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
            print("- google/gemini-pro")
            model = input("Enter OpenRouter model (or press Enter for default): ").strip()
            if not model:
                model = "meta-llama/llama-3.3-70b-instruct:free"
            self.model = model
            provider = input("Enter OpenRouter provider (or press Enter for default 'venice/fp8'): ").strip()
            if not provider:
                provider = "venice/fp8"
            self.provider = provider
            print(f"🚀 CONFIGURED: Using OpenRouter model: {self.model} (provider: {self.provider})")
        else:
            self.use_openrouter = False
            model = input("Enter Ollama model (or press Enter for default): ").strip()
            if not model:
                model = "llama3.2:3b"
            self.model = model
            # self.model = "llama3.2:3b"
            print(f"🚀 CONFIGURED: Using local Ollama model: {self.model}")

    def _load_memory(self):
        """Load RL loop memory from appended data"""
        if self.memory_file.exists():
            try:
                with open(self.memory_file, 'r') as f:
                    data = json.load(f)
                
                # Handle both old format (dict) and new format (list of sessions)
                if isinstance(data, dict):
                    # Old format - convert to new format
                    self._load_from_old_format(data)
                else:
                    # New format - load from most recent session
                    self._load_from_new_format(data)
                
                self.logger.info(f"📚 Loaded RL memory: {len(self.strategy_performance)} strategies, {len(self.optimization_sessions)} sessions")
                
            except Exception as e:
                self.logger.warning(f"⚠️ Failed to load RL memory: {e}")
                self._initialize_fresh()
        else:
            self._initialize_fresh()
    
    def _initialize_fresh(self):
        """Initialize fresh RL agent"""
        self.strategy_performance = {}
        self.optimization_sessions = []
        self.global_insights = []
        self.logger.info("📄 Starting fresh RL loop memory")
    
    def _initialize_strategies(self):
        """Initialize strategy performance tracking"""
        default_strategies = [
            "conservative_enhancement",
            "aggressive_transformation", 
            "material_focus",
            "artistic_elaboration",
            "technical_precision",
            "contextual_scene_building",
            "minimalist_refinement",
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

    def get_best_previous_session_for_prompt(self, prompt: str) -> dict:
        best_session = None
        for session in self.optimization_sessions:
            session_prompt = extract_true_prompt(session.original_prompt)
            if session_prompt == prompt:
                if (best_session is None or 
                    session.final_best_score > best_session.final_best_score):
                    best_session = session
        return best_session

    def get_best_previous_attempt_for_prompt(self, prompt: str) -> dict:
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
                                'strategy_used': attempt.strategy_used,
                                'session_id': session.session_id,
                                'attempt_number': attempt.attempt_number
                            }
        return best_attempt
    
    def optimize_with_rl_loop(self, prompt: str, use_validation: bool = True, prompt_with_context=None, endpoint: str = None) -> Dict[str, Any]:
        # Check if TRELLIS server is busy before starting optimization
        if endpoint:
            # override the endpoint
            self.logger.info(f"      🔧 Overriding endpoint: {self.endpoint}")
            self.logger.info(f"      🔧 Using endpoint: {endpoint}")
            self.endpoint = endpoint
        try:
            server_status_url = f"http://127.0.0.1:{self.port}/job/status/"
            resp = requests.get(server_status_url, timeout=5)
            if resp.status_code == 200:
                status_json = resp.json()
                job_status = status_json.get('status', 'unknown')
                if job_status not in ('idle', 'completed'):
                    self.logger.info(f"⏳ TRELLIS server busy (status: {job_status}), aborting RL optimization loop.")
                    return {'aborted': True, 'reason': f'TRELLIS server busy: {job_status}'}
            else:
                self.logger.warning(f"⚠️ Could not check TRELLIS server status (HTTP {resp.status_code}), proceeding anyway.")
        except Exception as e:
            self.logger.warning(f"⚠️ Exception checking TRELLIS server status: {e}, proceeding anyway.")

        session_id = f"rl_session_{int(time.time())}_{random.randint(1000, 9999)}"
        start_time = time.time()
        self.logger.info(f"\n🔄 RL LOOP OPTIMIZATION: '{prompt}'")
        self.logger.info(f"   Session: {session_id}")
        self.logger.info(f"   Max rounds: {self.max_optimization_rounds}")
        self.logger.info(f"   Min rounds before convergence: {self.min_rounds_before_convergence}")
        self.logger.info(f"   Target score: {self.min_score_threshold}")
        if self.disable_convergence:
            self.logger.info(f"   🚫 Convergence: DISABLED - will run for all {self.max_optimization_rounds} rounds")
        else:
            self.logger.info(f"   ✅ Convergence: ENABLED - may stop early on convergence")
        attempts: List[OptimizationAttempt] = []
        # Only add "white background" if it's not already present
        if "white background" not in prompt.lower():
            best_prompt = f"{prompt}, white background"
        else:
            best_prompt = prompt
        best_score = 0.0
        initial_score = None
        convergence_achieved = False
        convergence_reason = "Not converged"
        best_prev_session = self.get_best_previous_session_for_prompt(prompt) # past sessions from memory
        best_prev_attempt = self.get_best_previous_attempt_for_prompt(prompt) # past attempts from memory
        '''
        {
          "attempt_number": 6,
          "strategy_used": "material_focus",
          "exploration_type": "exploit",
          "optimized_prompt": "an ornate heart-shaped pendant, masterfully crafted with a polished sterling silver frame, intricate gold filigree, and a vibrant central turquoise stone, clean white background",
          "predicted_confidence": 0.95,
          "validation_score": 0.8223,
          "alignment_score": 0.3848,
          "quality_score": 0.9714,
          "demo_fidelity_score": 1.0,
          "task_fidelity_score": 0.8223,
          "validation_passed": true,
          "result_file": "subnet_validation_results.json",
          "timestamp": 1753383335.0,
          "agent_reasoning": "Marked as previous win for curriculum learning. This attempt achieved perfect fidelity and passed all thresholds."
        },
        {
          "attempt_number": 7,
          "strategy_used": "material_focus",
          "exploration_type": "exploit",
          "optimized_prompt": "an exquisite, ornate heart-shaped pendant, masterfully crafted from gleaming sterling silver with intricate gold filigree that catches the light, featuring a vibrant central turquoise stone that emits a subtle inner glow, set against a soft, luminous white background",
          "predicted_confidence": 0.98,
          "validation_score": 0.8743,
          "alignment_score": 0.5599,
          "quality_score": 0.9917,
          "demo_fidelity_score": 1.0,
          "task_fidelity_score": 0.8743,
          "validation_passed": true,
          "result_file": "subnet_validation_results.json",
          "timestamp": 1753383528.0,
          "agent_reasoning": "Marked as previous win for curriculum learning. This attempt achieved perfect fidelity and passed all thresholds."
        }
        '''
        if best_prev_attempt:
            best_prompt = best_prev_attempt['optimized_prompt']
            best_score = best_prev_attempt['validation_score']
            initial_score = best_score
            self.logger.info(f"      🎯 Found previous best attempt: {best_prompt} (score: {best_score:.4f})")
            self.logger.info(f"      🎯 Strategy used: {best_prev_attempt['strategy_used']}")
            self.logger.info(f"      🎯 Session ID: {best_prev_attempt['session_id']}")
            self.logger.info(f"      🎯 Past final best prompt: {best_prev_session.final_best_prompt}")
            self.logger.info(f"      🎯 Past final best score: {best_prev_session.final_best_score:.3f}")
            self.logger.info(f"      🎯 Past session ID: {best_prev_session.session_id}")
            self.logger.info(f"      🎯 Past learned insights: {', '.join(best_prev_session.learned_insights)}")
            self.logger.info(f"      🎯 Past strategy performance updates: {best_prev_session.strategy_performance_updates}")
            self.logger.info(f"      🎯 Past final best prompt: {best_prev_session.final_best_prompt}")
            self.logger.info(f"      🎯 Past final best score: {best_prev_session.final_best_score:.3f}")
            
            past_data = (
                    f"--- BEST IN PREVIOUS SESSIONS ---\n"
                    f"Final Best Prompt: '{best_prev_session.final_best_prompt}'\n"
                    f"Final Best Score: {best_prev_session.final_best_score:.3f}\n"
                    f"Session: {best_prev_session.session_id}\n"
                    f"Learned Insights: {', '.join(best_prev_session.learned_insights)}\n"
                    f"Strategy Performance Updates: {best_prev_session.strategy_performance_updates}\n"
                    "Try to beat this previous best, or explain why it cannot be improved.\n"
                )
            if prompt_with_context:
                prompt_with_context += past_data
            else:
                prompt_with_context = past_data


        for round_num in range(1, self.max_optimization_rounds + 1):
            self.logger.info(f"\n   🔄 RL Round {round_num}/{self.max_optimization_rounds}")
            strategy, exploration_type = self._select_strategy_for_rl()
            attempt = self._make_optimization_attempt(
                prompt, strategy, exploration_type, round_num, attempts, 
                prompt_with_context=prompt_with_context
            )
            if use_validation:
                attempt.validation_score = self._validate_prompt(
                    prompt, 
                    attempt.optimized_prompt, 
                    endpoint=endpoint, 
                    use_direct=not self.use_separated_validation,
                    use_separated=self.use_separated_validation,
                    cache_models=self.cache_models  # Use stored cache_models parameter
                )
                self.logger.info(f"      📊 Validation score: {attempt.validation_score:.4f}")
            else:
                attempt.validation_score = attempt.predicted_confidence
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
            
            should_converge, reason = self._should_converge(attempts, round_num)
            if should_converge:
                convergence_achieved = True
                convergence_reason = reason
                self.logger.info(f"      ✅ Convergence achieved: {reason}")
                break  # Actually break when convergence is achieved
            
            # Add stuck detection - if score hasn't changed for 8 rounds, force stop
            # Only apply if convergence is enabled
            if not self.disable_convergence and len(attempts) >= 8:
                recent_scores = [a.validation_score or 0.0 for a in attempts[-8:]]
                if all(abs(score - recent_scores[0]) < 0.001 for score in recent_scores):
                    convergence_achieved = True
                    convergence_reason = f"Stuck at score {recent_scores[0]:.4f} for 8 rounds"
                    self.logger.info(f"      🛑 Forced convergence: {convergence_reason}")
                    break
            
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
            'strategy_sequence': [a.strategy_used for a in attempts],
            'exploration_ratio': sum(1 for a in attempts if a.exploration_type == 'explore') / len(attempts)
        }
        self.logger.info(f"\n🎯 RL LOOP COMPLETE:")
        self.logger.info(f"   Best prompt: {best_prompt}")
        self.logger.info(f"   Best score: {best_score:.4f}")
        self.logger.info(f"   Rounds: {len(attempts)}")
        self.logger.info(f"   Convergence: {convergence_achieved}")
        self.logger.info(f"   Convergence reason: {convergence_reason}")
        self.logger.info(f"   Exploration ratio: {result['exploration_ratio']:.1%}")
        self.logger.info(f"   Insights learned: {len(learned_insights)}")
        self.logger.info(f"   Total time: {session.session_duration:.2f}s")
        return result
    
    def _should_converge(self, attempts: List[OptimizationAttempt], current_round: int) -> Tuple[bool, str]:
        """Improved convergence logic with minimum rounds and adaptive thresholds"""
        # If convergence is disabled, never converge (force all rounds)
        if self.disable_convergence:
            return False, "Convergence disabled - forcing all rounds"
            
        if current_round < self.min_rounds_before_convergence:
            return False, f"Below minimum rounds ({current_round}/{self.min_rounds_before_convergence})"
        if attempts and attempts[-1].validation_score and attempts[-1].validation_score >= self.min_score_threshold:
            return True, f"Target score achieved ({attempts[-1].validation_score:.3f})"
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
        """Improved strategy selection with better exploration/exploitation balance"""
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
                                 round_num: int, previous_attempts: List[OptimizationAttempt], max_retries: int = 3, 
                                 prompt_with_context=None) -> OptimizationAttempt:
        """Make a single optimization attempt with context from previous rounds, with retry-on-failure logic."""
        self.logger.info(f"      🎯 Strategy: {strategy} ({exploration_type})")
        
        # Print LLM provider info for this round
        if self.use_vllm:
            print(f"      🤖 [ROUND {round_num}] Using vLLM: {self.vllm_url} | Model: {self.vllm_model}")
        elif self.use_openrouter:
            print(f"      🤖 [ROUND {round_num}] Using OpenRouter: {self.api_base_url} | Model: {self.model}")
        else:
            print(f"      🤖 [ROUND {round_num}] Using Ollama: {self.ollama_url} | Model: {self.model}")
        
        # Request Ollama access for this round if coordinator is available
        ollama_request_id = None
        if self.ollama_coordinator and not self.use_openrouter and not self.use_vllm:
            try:
                ollama_request_id = self.ollama_coordinator.request_access(
                    priority=2,  # MEDIUM priority for RL optimization
                    description=f"RL Round {round_num} - {strategy} ({exploration_type})"
                )
                
                if not self.ollama_coordinator.wait_for_access(ollama_request_id):
                    self.logger.warning(f"      ⏰ Ollama access timeout for round {round_num}, using fallback")
                    # Continue with fallback prompt
                    return OptimizationAttempt(
                        attempt_number=round_num,
                        strategy_used=strategy,
                        exploration_type=exploration_type,
                        optimized_prompt=f"{prompt}, front view, accurate, complete" + (", white background" if "white background" not in prompt.lower() else ""),
                        predicted_confidence=0.5,
                        validation_score=None,
                        agent_reasoning="Fallback due to Ollama access timeout",
                        timestamp=time.time()
                    )
                
                self.logger.info(f"      ✅ Ollama access granted for round {round_num}")
                
            except Exception as e:
                self.logger.warning(f"      ⚠️ Ollama coordination failed: {e}, proceeding without coordination")
        
        try:
            previous_context = self._build_previous_attempts_context(previous_attempts)
            strategy_context = self._build_strategy_context(strategy)
            system_prompt = f"""You are an RL agent learning to optimize prompts. This is round {round_num} of iterative optimization."""
            system_prompt += f"""

ORIGINAL PROMPT: "{prompt}"
STRATEGY: {strategy} ({exploration_type} mode)

{strategy_context}

{previous_context}

TASK: Create an optimized prompt that improves on previous attempts based on learned insights.

Rules:
- Learn from previous rounds' scores and approaches
- Be {'experimental' if exploration_type == 'explore' else 'systematic'} in your approach

**Prime Example:**
* **ORIGINAL:** `tall glass of layered lemonade`
* **OPTIMIZED (Score: 0.9443):** `a slender glass of layered lemonade suspended in mid-air over a serene lake with lotus flowers and gentle ripples on the surface`

If the original prompt is about an object, don't focus on the scene or background but refine details of the object itself.
Example:
* **ORIGINAL:** `small wooden hammer with screws`
* **OPTIMIZED (Score: 0.0 ⚠️):** `a weathered small wooden hammer resting on a worn leather workbench amidst tools of various trades in a cozy, rustic workshop filled with natural light and the scent of sawdust`
* **OPTIMIZED (Score: 0.8287):** `small wooden hammer with screws` # keep it simple for object-focused prompts 

"""
            if prompt_with_context:
                system_prompt += f"\n\n{prompt_with_context}"
            system_prompt += """
RESPONSE FORMAT:
REASONING: [Your reasoning considering previous attempts and scores]

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
                    self.logger.info(f"      🚀 Making LLM request for round {round_num}")
                    # Choose the appropriate query method based on provider
                    if self.use_vllm:
                        print(f"      🚀 [vLLM] Sending request to {self.vllm_url}...")
                        response = self._query_vllm(system_prompt)
                    elif self.use_openrouter:
                        print(f"      🚀 [OpenRouter] Sending request to {self.api_base_url}...")
                        response = self._query_llm(system_prompt)
                    else:
                        print(f"      🚀 [Ollama] Sending request to {self.ollama_url}...")
                        response = self._query_llama(system_prompt)
                    self.logger.info(f"      ✅ LLM request completed for round {round_num}")
                    structured_output = self._parse_optimization_response(response, prompt)
                    
                    # Release Ollama access immediately after use
                    if ollama_request_id and self.ollama_coordinator:
                        self.ollama_coordinator.release_access(ollama_request_id)
                        self.logger.info(f"      🔓 Released Ollama access for round {round_num}")
                    
                    return OptimizationAttempt(
                        attempt_number=round_num,
                        strategy_used=strategy,
                        exploration_type=exploration_type,
                        optimized_prompt=structured_output.get('optimized_prompt', f"{prompt}, front view, accurate, complete" + (", white background" if "white background" not in prompt.lower() else "")),
                        predicted_confidence=structured_output.get('confidence', 0.5),
                        validation_score=None,  # Will be filled later
                        agent_reasoning=response,
                        timestamp=time.time()
                    )
                except Exception as e:
                    self.logger.error(f"      ❌ Optimization attempt failed: {e}")
                    # Check for transient errors (429, 500, etc.)
                    err_str = str(e)
                    if '429' in err_str or 'Too Many Requests' in err_str:
                        wait_time = 30
                    elif '500' in err_str or 'Internal Server Error' in err_str:
                        wait_time = 10
                    else:
                        wait_time = 5
                    self.logger.info(f"      ⏳ Waiting {wait_time}s before retrying RL round (retry {retries+1}/{max_retries})")
                    time.sleep(wait_time)
                    retries += 1
            
            self.logger.error(f"      ❌ All retries failed for RL round {round_num}. Skipping this round.")
            return OptimizationAttempt(
                attempt_number=round_num,
                strategy_used=strategy,
                exploration_type=exploration_type,
                optimized_prompt=f"{prompt}, front view, accurate, complete" + (", white background" if "white background" not in prompt.lower() else ""),
                predicted_confidence=0.5,
                validation_score=None,  
                agent_reasoning=f"Fallback due to repeated error",
                timestamp=time.time()
            )
            
        except Exception as e:
            self.logger.error(f"      ❌ Unexpected error in optimization attempt: {e}")
            # Release Ollama access on error
            if ollama_request_id and self.ollama_coordinator:
                self.ollama_coordinator.release_access(ollama_request_id)
                self.logger.info(f"      🔓 Released Ollama access for round {round_num} (error case)")
            
            return OptimizationAttempt(
                attempt_number=round_num,
                strategy_used=strategy,
                exploration_type=exploration_type,
                optimized_prompt=f"{prompt}, front view, accurate, complete" + (", white background" if "white background" not in prompt.lower() else ""),
                predicted_confidence=0.5,
                validation_score=None,
                agent_reasoning=f"Fallback due to error: {e}",
                timestamp=time.time()
            )
    
    def _build_previous_attempts_context(self, previous_attempts: List[OptimizationAttempt]) -> str:
        """Build context from previous optimization attempts"""
        if not previous_attempts:
            return "PREVIOUS ATTEMPTS: None - this is your first attempt."
        
        context = "PREVIOUS ATTEMPTS IN THIS SESSION:\n"
        for attempt in previous_attempts:
            score_text = f"{attempt.validation_score:.4f}" if attempt.validation_score else "pending"
            context += f"Round {attempt.attempt_number}: {attempt.strategy_used}\n"
            context += f"  Score: {score_text} | Confidence: {attempt.predicted_confidence:.2f}\n"
            context += f"  Prompt: {attempt.optimized_prompt[:100]}...\n"
        
        # Add insights
        if len(previous_attempts) > 1:
            scores = [a.validation_score for a in previous_attempts if a.validation_score]
            if len(scores) > 1:
                trend = "improving" if scores[-1] > scores[-2] else "declining"
                context += f"\nTREND: Scores are {trend}. "
                
                if trend == "declining":
                    context += "Consider why the previous approach worked better."
                else:
                    context += "Build on what's working well."
        
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
    
    def _clear_trellis_gpu_cache(self):
        """Send a request to the TRELLIS server to clear GPU cache."""
        try:
            url = f"http://127.0.0.1:{self.port}/clear_cache/"
            resp = requests.post(url, timeout=10)
            if resp.status_code == 200:
                self.logger.info(f"[TRELLIS] GPU cache cleared: {resp.json()}")
            else:
                self.logger.warning(f"[TRELLIS] Failed to clear GPU cache: HTTP {resp.status_code}")
        except Exception as e:
            self.logger.warning(f"[TRELLIS] Exception clearing GPU cache: {e}")

    def _validate_prompt(self, original_prompt: str, optimized_prompt: str = None, endpoint: str = "generate/", 
                        use_direct: bool = False, use_separated: bool = False, cache_models: bool = False) -> float:
        """Run validation with conda environment, clearing GPU cache first via TRELLIS server.

        Args:
            original_prompt: Original prompt for validation scoring
            optimized_prompt: Optimized prompt for generation (optional)
            endpoint: Generation endpoint
            use_direct: If True, use direct validation function instead of subprocess
            use_separated: If True, use separated two-stage validation to avoid GPU memory conflicts
        """
        try:
            if use_separated:
                return self._validate_prompt_separated(original_prompt, optimized_prompt, endpoint, cache_models)
            elif use_direct:
                return self._validate_prompt_direct(original_prompt, optimized_prompt, endpoint)
            else:
                return self._validate_prompt_subprocess(original_prompt, optimized_prompt, endpoint)
        except Exception as e:
            self.logger.error(f"   ❌ Validation failed: {e}")
            return 0.0

    def _validate_prompt_direct(self, original_prompt: str, optimized_prompt: str = None, endpoint: str = "generate/") -> float:
        """Direct validation using cached models for speed."""
        try:
            self.logger.info("      🔍 Direct validating...")
            self._clear_trellis_gpu_cache()  # Clear GPU cache before validation

            # Import the direct validation function
            sys.path.append('/home/mbhat/three-gen-subnet-trellis')
            from subnet_accurate_validator_multigpu import validate_prompt_direct

            # Use optimized prompt for generation if provided, otherwise use original
            if optimized_prompt and optimized_prompt != original_prompt:
                self.logger.info(f"      📝 Using optimized prompt for generation: '{optimized_prompt}...'")
                self.logger.info(f"      🎯 Computing scores against original prompt: '{original_prompt}...'")
            else:
                self.logger.info(f"      📝 Using same prompt for generation and validation: '{original_prompt}...'")

            # Call direct validation function
            result = validate_prompt_direct(
                original_prompt=original_prompt,
                optimized_prompt=optimized_prompt,
                endpoint=endpoint,
                port=self.port
            )

            if not result:
                self.logger.warning("   ❌ Direct validation returned no result")
                return 0.0

            # Log results
            if result.get('endpoint_type') == 'image':
                self.logger.info(f"      🖼️ Image validation - tt_clip: {result.get('tt_clip', 0.0):.4f}, ti_clip: {result.get('ti_clip', 0.0):.4f}")
                score = result.get('ti_clip', 0.0)  # Use text-image CLIP for image endpoint
            else:
                score = result.get('validation_engine_score', 0.0)
                self.logger.info(f"      🏆 Validation score: {score:.4f}")
                self.logger.info(f"      🤝 Alignment score: {result.get('alignment_score', 0.0):.4f}")

            return float(score)

        except Exception as e:
            self.logger.error(f"   ❌ Direct validation failed: {e}")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            return 0.0

    def _validate_prompt_separated(self, original_prompt: str, optimized_prompt: str = None, endpoint: str = "generate/", cache_models: bool = False) -> float:
        """Separated validation using two-stage approach to avoid GPU memory conflicts."""
        try:
            self.logger.info("      🔍 Separated validation (two-stage)...")
            self._clear_trellis_gpu_cache()  # Clear GPU cache before validation

            # Import the separated validation functions
            sys.path.append('/home/mbhat/three-gen-subnet-trellis')
            from subnet_accurate_validator_multigpu import generate_and_get_ply_data, validate_ply_data

            # Use optimized prompt for generation if provided, otherwise use original
            generation_prompt = optimized_prompt if optimized_prompt else original_prompt
            if optimized_prompt and optimized_prompt != original_prompt:
                self.logger.info(f"      📝 Using optimized prompt for generation: '{generation_prompt}...'")
                self.logger.info(f"      🎯 Computing scores against original prompt: '{original_prompt}...'")
            else:
                self.logger.info(f"      📝 Using same prompt for generation and validation: '{generation_prompt[:50]}...'")

            # Stage 1: Generate PLY data directly in memory
            self.logger.info(f"      🎨 Stage 1: Generating PLY data...")
            generation_start = time.time()
            
            # Generate PLY data directly in memory (no temporary file)
            ply_data = generate_and_get_ply_data(
                prompt=generation_prompt,
                endpoint=endpoint,
                port=self.port
            )
            generation_time = time.time() - generation_start
            self.logger.info(f"      ⏱️ PLY generation: {generation_time:.2f}s")
            
            # Stage 2: Validate PLY data directly in memory
            self.logger.info(f"      🔍 Stage 2: Validating PLY data...")
            validation_start = time.time()
            
            result = validate_ply_data(
                ply_data=ply_data,
                prompt=original_prompt,
                use_cached_models=False,  # Always use non-cached for clean memory management
                force_cpu=False
            )
            validation_time = time.time() - validation_start
            self.logger.info(f"      ⏱️ PLY validation: {validation_time:.2f}s")
            
            # Extract score from result
            score = result.get('validation_engine_score', 0.0)
            alignment_score = result.get('alignment_score', 0.0)
            
            self.logger.info(f"      ✅ Separated validation score: {score:.4f}")
            self.logger.info(f"      🤝 Alignment score: {alignment_score:.4f}")
            
            return float(score)

        except Exception as e:
            self.logger.error(f"   ❌ Separated validation failed: {e}")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            return 0.0

    def _validate_prompt_subprocess(self, original_prompt: str, optimized_prompt: str = None, endpoint: str = "generate/") -> float:
        """Original subprocess-based validation."""
        try:
            self._clear_trellis_gpu_cache()  # Clear GPU cache before validation
            self.logger.info("      🔍 Validating...")

            # Use optimized prompt for generation if provided, otherwise use original
            if optimized_prompt and optimized_prompt != original_prompt:
                self.logger.info(f"      📝 Using optimized prompt for generation: '{optimized_prompt}...'")
                self.logger.info(f"      🎯 Computing scores against original prompt: '{original_prompt}...'")
                cmd = [
                    "bash", "-c",
                    f"source /home/mbhat/miniconda/bin/activate && conda activate trellis_new && python subnet_accurate_validator_multigpu.py \"{original_prompt}\" \"{optimized_prompt}\" --endpoint \"{endpoint}\" --port {self.port}"
                ]
            else:
                self.logger.info(f"      📝 Using same prompt for generation and validation: '{original_prompt}...'")
                cmd = [
                    "bash", "-c",
                    f"source /home/mbhat/miniconda/bin/activate && conda activate trellis_new && python subnet_accurate_validator_multigpu.py \"{original_prompt}\" --endpoint \"{endpoint}\" --port {self.port}"
                ]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
            if result.returncode != 0:
                self.logger.warning(f"   ❌ Validation failed (return code {result.returncode})")
                if "CUDA" in result.stderr or "out of memory" in result.stderr.lower():
                    self.logger.warning(f"   🔥 CUDA OOM detected in validation - clearing cache")
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                        time.sleep(2)  # Brief pause for memory cleanup
                return 0.0
            # Try to read from the port-specific file first, then fallback to generic file
            validation_files = [
                f"subnet_validation_results_{self.port}.json",
                f"subnet_validation_results_{self.trellis_server_url_w_port.split(':')[-1]}.json",
                "subnet_validation_results.json"
            ]

            data = None
            for validation_file in validation_files:
                try:
                    with open(validation_file, 'r') as f:
                        data = json.load(f)
                        self.logger.info(f"      📄 Reading validation results from: {validation_file}")
                        break
                except FileNotFoundError:
                    self.logger.debug(f"      📄 Validation file not found: {validation_file}")
                    continue

            if data is None:
                self.logger.warning(f"   ❌ No validation results file found")
                return 0.0

            score = data.get("validation_engine_score", 0.0)
            if score == 0.0 and torch.cuda.is_available():
                self.logger.warning(f"   🔧 Score 0.0 - clearing CUDA cache")
                torch.cuda.empty_cache()
        except Exception as e:
            self.logger.error(f"      ❌ Validation error: {e}")
            return 0.0
        return score
    
    def _parse_optimization_response(self, response: str, original_prompt: str) -> Dict[str, Any]:
        """Parse optimization response with robust fallbacks"""
        try:
            json_match = re.search(r'OPTIMIZATION:\s*(\{.*?\})', response, re.DOTALL | re.IGNORECASE)
            if json_match:
                return json.loads(json_match.group(1))
        except:
            pass
        
        # Fallback parsing
        optimized_prompt = original_prompt
        confidence = 0.5
        
        # Extract optimized prompt
        opt_patterns = [
            r'optimized_prompt[":\s]*"([^"]+)"',
            r'front view, accurate, complete, white background[^"]*([^"]+)',
        ]
        
        for pattern in opt_patterns:
            match = re.search(pattern, response, re.IGNORECASE)
            if match:
                optimized_prompt = match.group(1).strip()
                break
        
        # Ensure proper format - only add suffix if not already present
        if not optimized_prompt.endswith('front view, accurate, complete, white background'):
            # Only add "white background" if it's not already present
            if "white background" not in optimized_prompt.lower():
                optimized_prompt = optimized_prompt.rstrip(', ') + ", front view, accurate, complete, white background"
            else:
                optimized_prompt = optimized_prompt.rstrip(', ') + ", front view, accurate, complete"
        
        # Extract confidence
        conf_match = re.search(r'confidence["\s:]*([0-9.]+)', response, re.IGNORECASE)
        if conf_match:
            try:
                confidence_str = conf_match.group(1)
                # Handle edge cases like lone periods or invalid floats
                if confidence_str in ['.', '', 'nan']:
                    confidence = 0.7  # Default confidence
                else:
                    confidence = max(0.0, min(1.0, float(confidence_str)))
            except (ValueError, AttributeError):
                confidence = 0.7  # Default confidence on parse error
        
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
        print(f"      📤 [OpenRouter] Request sent to {self.api_base_url} | Model: {self.model}")
        response = requests.post(f"{self.api_base_url}/chat/completions", headers=headers, json=data, timeout=120)
        response.raise_for_status()
        content = response.json()["choices"][0]["message"]["content"].strip()
        print(f"      📥 [OpenRouter] Response received: {len(content)} characters")
        return content

    def _query_llama(self, prompt: str) -> str:
        """Query local Ollama LLM"""
        data = {
            # "model": self.model,
            "model": "my-model",
            "messages": [{"role": "user", "content": prompt}],
            "stream": False,
            "options": {
                "temperature": 0.7,
                "num_predict": 350,
                "top_p": 0.9
            }
        }
        self.logger.info(f"      📤 Sending Ollama request to {self.ollama_url}")
        self.logger.info(f"      📝 Prompt length: {len(prompt)} characters")
        print(f"      📤 [Ollama] Request sent to {self.ollama_url} | Model: {self.model}")
        response = requests.post(f"{self.ollama_url}/api/chat", json=data, timeout=120)
        response.raise_for_status()
        content = response.json()["message"]["content"].strip()
        self.logger.info(f"      📥 Received response: {len(content)} characters")
        print(f"      📥 [Ollama] Response received: {len(content)} characters")
        return content

    def _query_vllm(self, prompt: str) -> str:
        """Query vLLM server"""
        data = {
            "model": self.vllm_model,
            "messages": [{"role": "user", "content": prompt}],
            "stream": False,
            "temperature": 0.7,
            "max_tokens": 350,
            "top_p": 0.9
        }
        self.logger.info(f"      📤 Sending vLLM request to {self.vllm_url}")
        self.logger.info(f"      📝 Prompt length: {len(prompt)} characters")
        print(f"      📤 [vLLM] Request sent to {self.vllm_url} | Model: {self.vllm_model}")
        response = requests.post(f"{self.vllm_url}/v1/chat/completions", json=data, timeout=120)
        response.raise_for_status()
        content = response.json()["choices"][0]["message"]["content"].strip()
        self.logger.info(f"      📥 Received response: {len(content)} characters")
        print(f"      📥 [vLLM] Response received: {len(content)} characters")
        return content

    def _save_memory(self):
        """Save RL loop memory by appending to avoid conflicts with simultaneous runs"""
        try:
            # Create a unique session identifier for this run
            session_id = f"session_{int(time.time())}_{os.getpid()}"
            
            # Prepare the data to append
            data_to_append = {
                'session_id': session_id,
                'timestamp': time.time(),
                'pid': os.getpid(),
                'strategy_performance': {
                    name: asdict(perf) for name, perf in self.strategy_performance.items()
                },
                'optimization_sessions': [
                    asdict(session) for session in self.optimization_sessions[-50:]
                ],
                'global_insights': self.global_insights[-100:],  # Keep last 100
                'epsilon': self.epsilon,
                'last_updated': time.time()
            }
            
            # Append to memory file instead of overwriting
            memory_file_path = str(self.memory_file)
            
            # Load existing data if file exists
            existing_data = []
            if os.path.exists(memory_file_path):
                try:
                    with open(memory_file_path, 'r') as f:
                        existing_data = json.load(f)
                        if not isinstance(existing_data, list):
                            # If it's the old format, convert to new format
                            existing_data = [existing_data]
                except (json.JSONDecodeError, FileNotFoundError):
                    existing_data = []
            
            # Append new session data
            existing_data.append(data_to_append)
            
            # Keep only the last 10 sessions to prevent file from growing too large
            if len(existing_data) > 10:
                existing_data = existing_data[-10:]
            
            # Write back the combined data
            with open(memory_file_path, 'w') as f:
                json.dump(existing_data, f, indent=2)
                
            self.logger.info(f"📚 Appended memory for session {session_id} (PID: {os.getpid()})")
                
        except Exception as e:
            self.logger.warning(f"⚠️ Failed to append RL memory: {e}")
    
    def _load_from_old_format(self, data):
        """Load memory from old single-session format"""
        # Load strategy performance
        strategies_data = data.get('strategy_performance', {})
        self.strategy_performance = {
            name: StrategyPerformance(**perf)
            for name, perf in strategies_data.items()
        }
        
        # Load recent sessions with proper reconstruction
        sessions_data = data.get('optimization_sessions', [])
        self.optimization_sessions = []
        for session_data in sessions_data[-50:]:
            # Reconstruct attempts properly
            attempts = []
            for attempt_data in session_data.get('attempts', []):
                attempt = OptimizationAttempt(**attempt_data)
                attempts.append(attempt)
            
            session_data['attempts'] = attempts
            session = RLOptimizationSession(**session_data)
            self.optimization_sessions.append(session)
        
        self.global_insights = data.get('global_insights', [])
        self.epsilon = data.get('epsilon', self.epsilon)
    
    def _load_from_new_format(self, sessions_data):
        """Load memory from new multi-session format"""
        if not sessions_data:
            self._initialize_fresh()
            return
        
        # Find the most recent session for this PID, or the most recent overall
        current_pid = os.getpid()
        current_session = None
        
        # First try to find a session from the current PID
        for session in reversed(sessions_data):
            if session.get('pid') == current_pid:
                current_session = session
                break
        
        # If no session from current PID, use the most recent overall
        if not current_session:
            current_session = sessions_data[-1]
        
        # Load from the selected session
        strategies_data = current_session.get('strategy_performance', {})
        self.strategy_performance = {
            name: StrategyPerformance(**perf)
            for name, perf in strategies_data.items()
        }
        
        # Load recent sessions with proper reconstruction
        sessions_data = current_session.get('optimization_sessions', [])
        self.optimization_sessions = []
        for session_data in sessions_data[-50:]:
            # Reconstruct attempts properly
            attempts = []
            for attempt_data in session_data.get('attempts', []):
                attempt = OptimizationAttempt(**attempt_data)
                attempts.append(attempt)
            
            session_data['attempts'] = attempts
            session = RLOptimizationSession(**session_data)
            self.optimization_sessions.append(session)
        
        self.global_insights = current_session.get('global_insights', [])
        self.epsilon = current_session.get('epsilon', self.epsilon)
        
        self.logger.info(f"📚 Loaded memory from session {current_session.get('session_id', 'unknown')} (PID: {current_session.get('pid', 'unknown')})")
    
    def get_rl_insights(self) -> Dict[str, Any]:
        """Get RL learning insights"""
        if not self.optimization_sessions:
            return {"message": "No RL learning sessions yet"}
        
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
    
    def get_multi_session_insights(self) -> Dict[str, Any]:
        """Get insights across all sessions in the memory file"""
        try:
            if not self.memory_file.exists():
                return {"message": "No memory file found"}
            
            with open(self.memory_file, 'r') as f:
                all_sessions = json.load(f)
            
            if not isinstance(all_sessions, list):
                # Old format - convert to list
                all_sessions = [all_sessions]
            
            insights = {
                "total_sessions_in_file": len(all_sessions),
                "sessions_by_pid": {},
                "overall_strategy_performance": {},
                "cross_session_learning": {}
            }
            
            # Analyze each session
            for session_data in all_sessions:
                pid = session_data.get('pid', 'unknown')
                session_id = session_data.get('session_id', 'unknown')
                
                if pid not in insights["sessions_by_pid"]:
                    insights["sessions_by_pid"][pid] = []
                
                insights["sessions_by_pid"][pid].append({
                    "session_id": session_id,
                    "timestamp": session_data.get('timestamp', 0),
                    "strategies_count": len(session_data.get('strategy_performance', {})),
                    "sessions_count": len(session_data.get('optimization_sessions', [])),
                    "epsilon": session_data.get('epsilon', 0)
                })
                
                # Aggregate strategy performance across sessions
                for strategy_name, perf_data in session_data.get('strategy_performance', {}).items():
                    if strategy_name not in insights["overall_strategy_performance"]:
                        insights["overall_strategy_performance"][strategy_name] = {
                            "total_attempts": 0,
                            "total_success": 0,
                            "avg_score": 0.0,
                            "sessions_used": 0
                        }
                    
                    current = insights["overall_strategy_performance"][strategy_name]
                    current["total_attempts"] += perf_data.get('total_attempts', 0)
                    current["total_success"] += perf_data.get('success_count', 0)
                    current["sessions_used"] += 1
            
            # Calculate overall averages
            for strategy_name, data in insights["overall_strategy_performance"].items():
                if data["total_attempts"] > 0:
                    data["overall_success_rate"] = data["total_success"] / data["total_attempts"]
                    data["avg_attempts_per_session"] = data["total_attempts"] / data["sessions_used"]
            
            # Sort strategies by overall success rate
            insights["overall_strategy_performance"] = dict(
                sorted(
                    insights["overall_strategy_performance"].items(),
                    key=lambda x: x[1]["overall_success_rate"],
                    reverse=True
                )
            )
            
            return insights
            
        except Exception as e:
            return {"error": f"Failed to get multi-session insights: {e}"}
    
    def merge_memories_from_file(self) -> bool:
        """Merge memories from all sessions in the file into current agent state"""
        try:
            if not self.memory_file.exists():
                return False
            
            with open(self.memory_file, 'r') as f:
                all_sessions = json.load(f)
            
            if not isinstance(all_sessions, list):
                # Old format - nothing to merge
                return False
            
            if len(all_sessions) <= 1:
                # Only one session - nothing to merge
                return False
            
            # Start with current state
            merged_strategies = dict(self.strategy_performance)
            merged_sessions = list(self.optimization_sessions)
            merged_insights = list(self.global_insights)
            
            # Merge from other sessions
            for session_data in all_sessions:
                if session_data.get('pid') == os.getpid():
                    # Skip current session
                    continue
                
                # Merge strategy performance
                for strategy_name, perf_data in session_data.get('strategy_performance', {}).items():
                    if strategy_name in merged_strategies:
                        # Combine performance data
                        current = merged_strategies[strategy_name]
                        other = StrategyPerformance(**perf_data)
                        
                        # Weighted average based on attempt counts
                        total_attempts = current.total_attempts + other.total_attempts
                        if total_attempts > 0:
                            current.avg_score = (current.avg_score * current.total_attempts + 
                                               other.avg_score * other.total_attempts) / total_attempts
                            current.total_attempts = total_attempts
                            current.success_count += other.success_count
                            
                            # Combine recent scores (keep last 10 from each)
                            combined_scores = current.recent_scores + other.recent_scores
                            current.recent_scores = combined_scores[-10:]
                    else:
                        # Add new strategy
                        merged_strategies[strategy_name] = StrategyPerformance(**perf_data)
                
                # Merge optimization sessions (avoid duplicates)
                for session_data in session_data.get('optimization_sessions', []):
                    session = RLOptimizationSession(**session_data)
                    # Check if session already exists
                    if not any(s.session_id == session.session_id for s in merged_sessions):
                        merged_sessions.append(session)
                
                # Merge global insights (avoid duplicates)
                for insight in session_data.get('global_insights', []):
                    if insight not in merged_insights:
                        merged_insights.append(insight)
            
            # Update current state
            self.strategy_performance = merged_strategies
            self.optimization_sessions = merged_sessions[-100:]  # Keep last 100
            self.global_insights = merged_insights[-100:]  # Keep last 100
            
            self.logger.info(f"🔄 Merged memories from {len(all_sessions)} sessions")
            return True
            
        except Exception as e:
            self.logger.warning(f"⚠️ Failed to merge memories: {e}")
            return False

def main():
    """Command line interface"""
    if len(sys.argv) < 2 or sys.argv[1] in ('-h', '--help'):
        print("Usage: python smart_prompt_optimizer_v5_rl_loop_lora.py \"prompt\" [--validate] [--endpoint] [--port] [--insights] [--vllm] [--vllm-url] [--vllm-model] [--separated] [--cache-models]")
        print("\nCommands:")
        print("  \"prompt\"        Optimize with RL learning loop")
        print("  dummy --insights Show RL learning insights")
        print("  --merge-memories Merge memories from all sessions in file")
        print("  --multi-insights Show insights across all sessions")
        print("\nOptions:")
        print("  --validate       Use real validation scores (recommended)")
        print("  --separated      Use separated two-stage validation (avoids GPU memory conflicts)")
        print("  --cache-models   Enable model caching (uses more GPU memory but faster validation)")
        print("  --endpoint       Use specific endpoint (default: generate/)")
        print("  --port           Port for TRELLIS server (default: 8096)")
        print("  --vllm           Use vLLM instead of Ollama")
        print("  --vllm-url       vLLM server URL (default: http://localhost:9000)")
        print("  --vllm-model     vLLM model name (default: llama-3-2-3b-it)")
        return
    
    agent = None
    try:
        if "--insights" in sys.argv:
            agent = RLLoopAgent()
            insights = agent.get_rl_insights()
            print("\n🤖 RL LEARNING INSIGHTS")
            print("="*50)
            print(json.dumps(insights, indent=2))
            return
        elif "--multi-insights" in sys.argv:
            agent = RLLoopAgent()
            insights = agent.get_multi_session_insights()
            print("\n🤖 MULTI-SESSION INSIGHTS")
            print("="*50)
            print(json.dumps(insights, indent=2))
            return
        elif "--merge-memories" in sys.argv:
            agent = RLLoopAgent()
            success = agent.merge_memories_from_file()
            if success:
                print("✅ Successfully merged memories from all sessions")
                # Show updated insights
                insights = agent.get_rl_insights()
                print("\n🤖 UPDATED RL LEARNING INSIGHTS (After Merge)")
                print("="*50)
                print(json.dumps(insights, indent=2))
            else:
                print("❌ Failed to merge memories or no sessions to merge")
            return
        
        user_prompt = sys.argv[1]
        use_validation = "--validate" in sys.argv
        use_separated = "--separated" in sys.argv
        cache_models = "--cache-models" in sys.argv
        
        # Parse command line arguments
        endpoint = "generate/"
        port = 8096
        use_vllm = "--vllm" in sys.argv
        vllm_url = "http://localhost:9000"
        vllm_model = "llama-3-2-3b-it"
        
        for i, arg in enumerate(sys.argv):
            if arg == "--endpoint" and i + 1 < len(sys.argv):
                endpoint = sys.argv[i + 1]
            elif arg == "--port" and i + 1 < len(sys.argv):
                try:
                    port = int(sys.argv[i + 1])
                except ValueError:
                    print(f"⚠️ Invalid port number: {sys.argv[i + 1]}, using default 8096")
                    port = 8096
            elif arg == "--vllm-url" and i + 1 < len(sys.argv):
                vllm_url = sys.argv[i + 1]
            elif arg == "--vllm-model" and i + 1 < len(sys.argv):
                vllm_model = sys.argv[i + 1]
        
        print("🔄 RL LOOP AGENT - TRUE ITERATIVE LEARNING")
        print("=" * 60)
        print("✅ Multi-round optimization with score feedback")
        print("✅ Strategy performance updates based on results")
        print("✅ Convergence detection and early stopping")
        print("✅ Exploration/exploitation based on performance")
        print(f"✅ Using TRELLIS server on port {port}")
        print(f"✅ Using endpoint: {endpoint}")
        if use_separated:
            print("✅ Using SEPARATED two-stage validation (avoids GPU memory conflicts)")
        else:
            print("✅ Using DIRECT single-stage validation")
        if cache_models:
            print("✅ Model caching ENABLED (uses more GPU memory but faster validation)")
        else:
            print("✅ Model caching DISABLED (safer for GPU memory)")
        if use_vllm:
            print(f"✅ Using vLLM: {vllm_url} with model {vllm_model}")
        else:
            print("✅ Using Ollama (default)")
        print("=" * 60)
        
        agent = RLLoopAgent(port=port, use_vllm=use_vllm, vllm_url=vllm_url, vllm_model=vllm_model, use_separated_validation=use_separated, cache_models=cache_models)
        result = agent.optimize_with_rl_loop(user_prompt, use_validation=use_validation, endpoint=endpoint)
        
        print("\n" + "="*20 + " RL LOOP SUMMARY " + "="*20)
        print(f"   Original: {result['original_prompt']}")
        print(f"   Final Best: {result['final_optimized_prompt']}")
        print(f"   Final Score: {result['final_score']:.4f}")
        print(f"   Total Rounds: {result['total_rounds']}")
        print(f"   Converged: {result['convergence_achieved']}")
        print(f"   Convergence reason: {result['convergence_reason']}")
        print(f"   Exploration ratio: {result['exploration_ratio']:.1%}")
        print(f"   Score Progression: {[f'{s:.3f}' for s in result['score_progression']]}")
        print(f"   Strategy Sequence: {result['strategy_sequence']}")
        print(f"   Processing Time: {result['processing_time']:.2f}s")
        print("="*60)
        
    except KeyboardInterrupt:
        print("\n⚠️ Interrupted by user")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if agent:
            agent._save_memory()

if __name__ == "__main__":
    main() 
