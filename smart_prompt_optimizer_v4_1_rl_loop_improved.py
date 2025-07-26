#!/usr/bin/env python3
"""
Smart Prompt Optimizer V4.1 - IMPROVED RL LEARNING LOOP
======================================================
🔄 Enhanced RL loop with adaptive convergence and better exploration
🎯 Addresses premature convergence issues identified in analysis
🧠 Minimum rounds before convergence, adaptive thresholds
⚡ Better exploration/exploitation balance based on performance data
🎓 Learns from explore vs exploit performance patterns

Key Improvements:
1. Minimum rounds before convergence (prevents early stopping)
2. Adaptive convergence threshold based on exploration success
3. Increased exploration rate when explore performs better
4. Convergence only allowed after sufficient exploration
5. Dynamic epsilon adjustment based on performance patterns
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
    convergence_reason: str  # New field to track why convergence happened

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
    explore_performance: float  # Average score when used for exploration
    exploit_performance: float  # Average score when used for exploitation

class ImprovedRLLoopAgent:
    """Improved RL agent with adaptive convergence and better exploration"""
    
    def __init__(self, ollama_url: str = "http://localhost:11434",
                 memory_file: str = "rl_loop_memory_improved.json"):
        self.ollama_url = ollama_url
        self.model = "llama3.2:3b"
        self.memory_file = Path(memory_file)
        
        # Improved RL Loop parameters
        self.max_optimization_rounds = 15  # Increased from 5
        self.min_rounds_before_convergence = 5  # NEW: Minimum rounds before convergence
        self.convergence_threshold = 0.01  # Reduced from 0.02 for more sensitivity
        self.min_score_threshold = 0.85   # Target score to achieve
        
        # Adaptive convergence parameters
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
        
        # Performance tracking
        self.explore_scores: List[float] = []
        self.exploit_scores: List[float] = []
        
        self.logger = logging.getLogger(__name__)
        
        self._load_memory()
        self._initialize_strategies()
        
        self.logger.info(f"🔄 IMPROVED RL LOOP AGENT INITIALIZED")
        self.logger.info(f"   Strategy tracking: {len(self.strategy_performance)} strategies")
        self.logger.info(f"   Past sessions: {len(self.optimization_sessions)}")
        self.logger.info(f"   Exploration rate: {self.epsilon:.2f}")
        self.logger.info(f"   Max rounds per optimization: {self.max_optimization_rounds}")
        self.logger.info(f"   Min rounds before convergence: {self.min_rounds_before_convergence}")
        self.logger.info(f"   Adaptive convergence: ENABLED")
    
    def _load_memory(self):
        """Load RL loop memory with improved data structure"""
        if self.memory_file.exists():
            try:
                with open(self.memory_file, 'r') as f:
                    data = json.load(f)
                
                # Load strategy performance with new fields
                strategies_data = data.get('strategy_performance', {})
                self.strategy_performance = {}
                for name, perf in strategies_data.items():
                    # Handle old format
                    if 'explore_performance' not in perf:
                        perf['explore_performance'] = 0.5
                    if 'exploit_performance' not in perf:
                        perf['exploit_performance'] = 0.5
                    
                    self.strategy_performance[name] = StrategyPerformance(**perf)
                
                # Load recent sessions
                sessions_data = data.get('optimization_sessions', [])
                self.optimization_sessions = []
                for session_data in sessions_data[-50:]:
                    attempts = []
                    for attempt_data in session_data.get('attempts', []):
                        attempt = OptimizationAttempt(**attempt_data)
                        attempts.append(attempt)
                    
                    session_data['attempts'] = attempts
                    # Handle old format
                    if 'convergence_reason' not in session_data:
                        session_data['convergence_reason'] = 'unknown'
                    
                    session = RLOptimizationSession(**session_data)
                    self.optimization_sessions.append(session)
                
                self.global_insights = data.get('global_insights', [])
                self.epsilon = data.get('epsilon', self.epsilon)
                self.explore_scores = data.get('explore_scores', [])
                self.exploit_scores = data.get('exploit_scores', [])
                
                self.logger.info(f"📚 Loaded improved RL memory: {len(self.strategy_performance)} strategies, {len(self.optimization_sessions)} sessions")
                
            except Exception as e:
                self.logger.warning(f"⚠️ Failed to load improved RL memory: {e}")
                self._initialize_fresh()
        else:
            self._initialize_fresh()
    
    def _initialize_fresh(self):
        """Initialize fresh improved RL agent"""
        self.strategy_performance = {}
        self.optimization_sessions = []
        self.global_insights = []
        self.explore_scores = []
        self.exploit_scores = []
        self.logger.info("📄 Starting fresh improved RL loop memory")
    
    def _initialize_strategies(self):
        """Initialize strategy performance tracking with improved metrics"""
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
                    improvement_trend=0.0,
                    explore_performance=0.5,
                    exploit_performance=0.5
                )
    
    def _should_converge(self, attempts: List[OptimizationAttempt], current_round: int) -> Tuple[bool, str]:
        """Improved convergence logic with minimum rounds and adaptive thresholds"""
        
        # Never converge before minimum rounds
        if current_round < self.min_rounds_before_convergence:
            return False, f"Below minimum rounds ({current_round}/{self.min_rounds_before_convergence})"
        
        # Check if we hit target score
        if attempts and attempts[-1].validation_score and attempts[-1].validation_score >= self.min_score_threshold:
            return True, f"Target score achieved ({attempts[-1].validation_score:.3f})"
        
        # Check for convergence based on improvement
        if len(attempts) >= 2:
            current_score = attempts[-1].validation_score or 0.0
            previous_score = attempts[-2].validation_score or 0.0
            improvement = current_score - previous_score
            
            # Adaptive convergence threshold based on exploration performance
            adaptive_threshold = self.convergence_threshold
            if self.explore_scores and self.exploit_scores:
                explore_avg = statistics.mean(self.explore_scores[-10:]) if len(self.explore_scores) >= 10 else 0.5
                exploit_avg = statistics.mean(self.exploit_scores[-10:]) if len(self.exploit_scores) >= 10 else 0.5
                
                if explore_avg > exploit_avg + self.explore_performance_threshold:
                    # If explore is performing better, be more strict about convergence
                    adaptive_threshold = self.convergence_threshold * 0.5
                    self.logger.info(f"      🔍 Explore performing better - stricter convergence threshold: {adaptive_threshold:.3f}")
            
            if abs(improvement) < adaptive_threshold:
                # Only converge if we've had sufficient exploration
                explore_attempts = sum(1 for a in attempts if a.exploration_type == 'explore')
                total_attempts = len(attempts)
                explore_ratio = explore_attempts / total_attempts
                
                if explore_ratio < 0.3:  # Need at least 30% exploration
                    return False, f"Insufficient exploration ({explore_ratio:.1%}) for convergence"
                
                return True, f"Convergence threshold met (improvement: {improvement:.3f})"
        
        return False, "No convergence criteria met"
    
    def _select_strategy_for_rl(self) -> Tuple[str, str]:
        """Improved strategy selection with better exploration/exploitation balance"""
        
        # Calculate current explore vs exploit performance
        if self.explore_scores and self.exploit_scores:
            recent_explore = statistics.mean(self.explore_scores[-5:]) if len(self.explore_scores) >= 5 else 0.5
            recent_exploit = statistics.mean(self.exploit_scores[-5:]) if len(self.exploit_scores) >= 5 else 0.5
            
            # Adjust epsilon based on relative performance
            if recent_explore > recent_exploit + self.explore_performance_threshold:
                self.epsilon = min(0.8, self.epsilon + 0.05)  # Increase exploration
                self.logger.info(f"      🔍 Explore performing better - increasing epsilon to {self.epsilon:.2f}")
            elif recent_exploit > recent_explore + self.explore_performance_threshold:
                self.epsilon = max(self.epsilon_min, self.epsilon - 0.02)  # Decrease exploration
        
        if random.random() < self.epsilon:
            # EXPLORE: Enhanced exploration logic
            exploration_type = "explore"
            
            # Prefer strategies with high uncertainty or poor recent performance
            strategy_scores = []
            for name, perf in self.strategy_performance.items():
                if perf.total_attempts == 0:
                    uncertainty = float('inf')
                else:
                    # Consider both uncertainty and explore performance
                    uncertainty = 1.0 / (perf.total_attempts + 1)
                    if perf.explore_performance < 0.6:  # Prefer strategies that need exploration
                        uncertainty += 0.3
                    if perf.improvement_trend < 0:  # Getting worse
                        uncertainty += 0.2
                
                strategy_scores.append((name, uncertainty))
            
            strategy_scores.sort(key=lambda x: x[1], reverse=True)
            selected_strategy = strategy_scores[0][0]
            
        else:
            # EXPLOIT: Enhanced exploitation logic
            exploration_type = "exploit"
            
            best_strategy = None
            best_combined_score = -1
            
            for name, perf in self.strategy_performance.items():
                if perf.total_attempts == 0:
                    continue
                
                # Consider both exploit performance and overall performance
                combined_score = (perf.exploit_performance * 0.4 + 
                                perf.avg_score * 0.3 + 
                                perf.confidence_in_strategy * 0.2 + 
                                max(0, perf.improvement_trend) * 0.1)
                
                if combined_score > best_combined_score:
                    best_combined_score = combined_score
                    best_strategy = name
            
            selected_strategy = best_strategy or "conservative_enhancement"
        
        return selected_strategy, exploration_type
    
    def optimize_with_rl_loop(self, prompt: str, use_validation: bool = True) -> Dict[str, Any]:
        """Main RL optimization loop with improved convergence logic"""
        session_id = f"rl_session_{int(time.time())}_{random.randint(1000, 9999)}"
        start_time = time.time()
        
        self.logger.info(f"\n🔄 IMPROVED RL LOOP OPTIMIZATION: '{prompt}'")
        self.logger.info(f"   Session: {session_id}")
        self.logger.info(f"   Max rounds: {self.max_optimization_rounds}")
        self.logger.info(f"   Min rounds before convergence: {self.min_rounds_before_convergence}")
        self.logger.info(f"   Target score: {self.min_score_threshold}")
        
        attempts: List[OptimizationAttempt] = []
        best_prompt = f"wbgmsst, {prompt}, white background"
        best_score = 0.0
        convergence_achieved = False
        convergence_reason = "Not converged"
        
        # RL Loop: Multiple optimization rounds
        for round_num in range(1, self.max_optimization_rounds + 1):
            self.logger.info(f"\n   🔄 RL Round {round_num}/{self.max_optimization_rounds}")
            
            # Select strategy based on current performance and exploration
            strategy, exploration_type = self._select_strategy_for_rl()
            
            # Create optimization attempt
            attempt = self._make_optimization_attempt(
                prompt, strategy, exploration_type, round_num, attempts
            )
            
            # Validate if requested
            if use_validation:
                attempt.validation_score = self._validate_prompt(attempt.optimized_prompt)
                self.logger.info(f"      📊 Validation score: {attempt.validation_score:.4f}")
            else:
                attempt.validation_score = attempt.predicted_confidence
            
            attempts.append(attempt)
            
            # Track explore vs exploit performance
            if attempt.validation_score:
                if exploration_type == 'explore':
                    self.explore_scores.append(attempt.validation_score)
                else:
                    self.exploit_scores.append(attempt.validation_score)
                
                # Keep only recent scores
                if len(self.explore_scores) > 50:
                    self.explore_scores = self.explore_scores[-50:]
                if len(self.exploit_scores) > 50:
                    self.exploit_scores = self.exploit_scores[-50:]
            
            # Update best if this is better
            if attempt.validation_score and attempt.validation_score > best_score:
                best_score = attempt.validation_score
                best_prompt = attempt.optimized_prompt
                self.logger.info(f"      🎯 New best score: {best_score:.4f}")
            
            # Update strategy performance immediately
            self._update_strategy_performance(strategy, attempt.validation_score or 0.0, exploration_type)
            
            # Check for convergence with improved logic
            should_converge, reason = self._should_converge(attempts, round_num)
            if should_converge:
                convergence_achieved = True
                convergence_reason = reason
                self.logger.info(f"      ✅ Convergence achieved: {reason}")
                break
            
            # Learn from this round for next round
            if round_num < self.max_optimization_rounds:
                self._inter_round_learning(attempts)
        
        # Final learning and insights extraction
        learned_insights = self._extract_session_insights(attempts)
        strategy_updates = self._calculate_strategy_updates(attempts)
        
        # Create session record
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
            session_duration=time.time() - start_time,
            convergence_reason=convergence_reason
        )
        
        # Update global learning
        self.optimization_sessions.append(session)
        self.global_insights.extend(learned_insights)
        self._decay_exploration()
        self._save_memory()
        
        # Prepare result
        result = {
            'session_id': session_id,
            'original_prompt': prompt,
            'final_optimized_prompt': best_prompt,
            'final_score': best_score,
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
        
        self.logger.info(f"\n🎯 IMPROVED RL LOOP COMPLETE:")
        self.logger.info(f"   Best prompt: {best_prompt}")
        self.logger.info(f"   Best score: {best_score:.4f}")
        self.logger.info(f"   Rounds: {len(attempts)}")
        self.logger.info(f"   Convergence: {convergence_achieved}")
        self.logger.info(f"   Convergence reason: {convergence_reason}")
        self.logger.info(f"   Exploration ratio: {result['exploration_ratio']:.1%}")
        self.logger.info(f"   Insights learned: {len(learned_insights)}")
        self.logger.info(f"   Total time: {session.session_duration:.2f}s")
        
        return result
    
    def _update_strategy_performance(self, strategy: str, score: float, exploration_type: str):
        """Update strategy performance with explore/exploit tracking"""
        perf = self.strategy_performance[strategy]
        
        # Update basic stats
        perf.total_attempts += 1
        if score >= 0.7:  # Success threshold
            perf.success_count += 1
        
        # Update average score with exponential moving average
        alpha = 0.2
        perf.avg_score = (1 - alpha) * perf.avg_score + alpha * score
        
        # Update explore/exploit performance separately
        if exploration_type == 'explore':
            perf.explore_performance = (1 - alpha) * perf.explore_performance + alpha * score
        else:
            perf.exploit_performance = (1 - alpha) * perf.exploit_performance + alpha * score
        
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
    
    def _make_optimization_attempt(self, prompt: str, strategy: str, exploration_type: str, 
                                 round_num: int, previous_attempts: List[OptimizationAttempt]) -> OptimizationAttempt:
        """Make a single optimization attempt with enhanced context"""
        
        self.logger.info(f"      🎯 Strategy: {strategy} ({exploration_type})")
        
        # Build context from previous attempts
        previous_context = self._build_previous_attempts_context(previous_attempts)
        strategy_context = self._build_strategy_context(strategy)
        
        # Enhanced system prompt with exploration guidance
        exploration_guidance = ""
        if exploration_type == 'explore':
            exploration_guidance = """
EXPLORATION MODE: You are in exploration mode. Be creative and experimental:
- Try different approaches and perspectives
- Consider unusual combinations or descriptions
- Don't be afraid to take risks
- Focus on discovering new optimization patterns
"""
        else:
            exploration_guidance = """
EXPLOITATION MODE: You are in exploitation mode. Be systematic and refined:
- Build on what has worked well
- Make incremental improvements
- Focus on consistency and reliability
- Leverage proven optimization patterns
"""
        
        system_prompt = f"""You are an improved RL agent learning to optimize prompts. This is round {round_num} of iterative optimization.

ORIGINAL PROMPT: "{prompt}"
STRATEGY: {strategy} ({exploration_type} mode)

{exploration_guidance}

{strategy_context}

{previous_context}

TASK: Create an optimized prompt that improves on previous attempts based on learned insights.

Rules:
- Start with "wbgmsst," and end with ", white background"
- Learn from previous rounds' scores and approaches
- Consider both exploration and exploitation patterns

RESPONSE FORMAT:
REASONING: [Your reasoning considering previous attempts and scores]

OPTIMIZATION: {{
  "optimized_prompt": "[full optimized prompt]",
  "confidence": [0.0-1.0],
  "key_changes": ["change1", "change2"],
  "expected_score": [0.0-1.0],
  "learning_applied": ["insight1", "insight2"]
}}"""

        try:
            response = self._query_llama(system_prompt)
            structured_output = self._parse_optimization_response(response, prompt)
            
            return OptimizationAttempt(
                attempt_number=round_num,
                strategy_used=strategy,
                exploration_type=exploration_type,
                optimized_prompt=structured_output.get('optimized_prompt', f"wbgmsst, {prompt}, white background"),
                predicted_confidence=structured_output.get('confidence', 0.5),
                validation_score=None,  # Will be filled later
                agent_reasoning=response,
                timestamp=time.time()
            )
            
        except Exception as e:
            self.logger.error(f"      ❌ Optimization attempt failed: {e}")
            return OptimizationAttempt(
                attempt_number=round_num,
                strategy_used=strategy,
                exploration_type=exploration_type,
                optimized_prompt=f"wbgmsst, {prompt}, white background",
                predicted_confidence=0.5,
                validation_score=None,
                agent_reasoning=f"Fallback due to error: {e}",
                timestamp=time.time()
            )
    
    def _build_previous_attempts_context(self, previous_attempts: List[OptimizationAttempt]) -> str:
        """Build enhanced context from previous optimization attempts"""
        if not previous_attempts:
            return "PREVIOUS ATTEMPTS: None - this is your first attempt."
        
        context = "PREVIOUS ATTEMPTS IN THIS SESSION:\n"
        for attempt in previous_attempts:
            score_text = f"{attempt.validation_score:.4f}" if attempt.validation_score else "pending"
            context += f"Round {attempt.attempt_number}: {attempt.strategy_used} ({attempt.exploration_type})\n"
            context += f"  Score: {score_text} | Confidence: {attempt.predicted_confidence:.2f}\n"
            context += f"  Prompt: {attempt.optimized_prompt[:100]}...\n"
        
        # Add exploration insights
        explore_attempts = [a for a in previous_attempts if a.exploration_type == 'explore']
        exploit_attempts = [a for a in previous_attempts if a.exploration_type == 'exploit']
        
        if explore_attempts and exploit_attempts:
            explore_scores = [a.validation_score for a in explore_attempts if a.validation_score]
            exploit_scores = [a.validation_score for a in exploit_attempts if a.validation_score]
            
            if explore_scores and exploit_scores:
                explore_avg = statistics.mean(explore_scores)
                exploit_avg = statistics.mean(exploit_scores)
                
                if explore_avg > exploit_avg:
                    context += f"\nINSIGHT: Exploration is performing better (explore: {explore_avg:.3f} vs exploit: {exploit_avg:.3f})"
                else:
                    context += f"\nINSIGHT: Exploitation is performing better (exploit: {exploit_avg:.3f} vs explore: {explore_avg:.3f})"
        
        return context
    
    def _build_strategy_context(self, strategy: str) -> str:
        """Build enhanced context for specific strategy"""
        perf = self.strategy_performance.get(strategy)
        if not perf or perf.total_attempts == 0:
            return f"STRATEGY {strategy}: No prior experience - explore freely."
        
        context = f"STRATEGY {strategy} PERFORMANCE:\n"
        context += f"  Average score: {perf.avg_score:.3f}\n"
        context += f"  Explore performance: {perf.explore_performance:.3f}\n"
        context += f"  Exploit performance: {perf.exploit_performance:.3f}\n"
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
        """Enhanced learning between rounds"""
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
        
        # Update epsilon based on performance patterns
        if current.validation_score and current.validation_score < 0.5:
            self.epsilon = min(0.8, self.epsilon + 0.1)  # Explore more if doing poorly
            self.logger.info(f"      🔍 Increasing exploration to {self.epsilon:.2f}")
    
    def _extract_session_insights(self, attempts: List[OptimizationAttempt]) -> List[str]:
        """Extract enhanced insights from the optimization session"""
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
        
        # Exploration insights
        explore_attempts = [a for a in attempts if a.exploration_type == 'explore']
        exploit_attempts = [a for a in attempts if a.exploration_type == 'exploit']
        
        if explore_attempts and exploit_attempts:
            explore_scores = [a.validation_score for a in explore_attempts if a.validation_score]
            exploit_scores = [a.validation_score for a in exploit_attempts if a.validation_score]
            
            if explore_scores and exploit_scores:
                explore_avg = statistics.mean(explore_scores)
                exploit_avg = statistics.mean(exploit_scores)
                
                if explore_avg > exploit_avg:
                    insights.append(f"Exploration outperformed exploitation: {explore_avg:.3f} vs {exploit_avg:.3f}")
                else:
                    insights.append(f"Exploitation outperformed exploration: {exploit_avg:.3f} vs {explore_avg:.3f}")
        
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
        """Slower decay of exploration rate"""
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
    
    def _validate_prompt(self, prompt: str) -> float:
        """Run validation with conda environment"""
        try:
            self.logger.info("      🔍 Validating...")
            cmd = [
                "bash", "-c",
                f"source /home/mbhat/miniconda/bin/activate && conda activate trellis_new && python subnet_accurate_validator.py \"{prompt}\""
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

            with open("subnet_validation_results.json", 'r') as f:
                data = json.load(f)
                score = data.get("validation_engine_score", 0.0)
                
                # If score is 0.0, might be OOM - try cleanup
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
        optimized_prompt = f"wbgmsst, {original_prompt}, white background"
        confidence = 0.5
        
        # Extract optimized prompt
        opt_patterns = [
            r'optimized_prompt[":\s]*"([^"]+)"',
            r'wbgmsst[^"]*([^"]+white background)',
        ]
        
        for pattern in opt_patterns:
            match = re.search(pattern, response, re.IGNORECASE)
            if match:
                optimized_prompt = match.group(1).strip()
                break
        
        # Ensure proper format
        if not optimized_prompt.startswith('wbgmsst'):
            optimized_prompt = f"wbgmsst, {optimized_prompt}"
        if not optimized_prompt.endswith('white background'):
            optimized_prompt = optimized_prompt.rstrip(', ') + ", white background"
        
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
    
    def _query_llama(self, prompt: str) -> str:
        """Query LLM"""
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
        """Save improved RL loop memory"""
        try:
            data = {
                'strategy_performance': {
                    name: asdict(perf) for name, perf in self.strategy_performance.items()
                },
                'optimization_sessions': [
                    asdict(session) for session in self.optimization_sessions[-50:]
                ],
                'global_insights': self.global_insights[-100:],  # Keep last 100
                'epsilon': self.epsilon,
                'explore_scores': self.explore_scores,
                'exploit_scores': self.exploit_scores,
                'last_updated': time.time()
            }
            
            with open(self.memory_file, 'w') as f:
                json.dump(data, f, indent=2)
                
        except Exception as e:
            self.logger.warning(f"⚠️ Failed to save improved RL memory: {e}")
    
    def get_rl_insights(self) -> Dict[str, Any]:
        """Get improved RL learning insights"""
        if not self.optimization_sessions:
            return {"message": "No RL learning sessions yet"}
        
        recent_sessions = self.optimization_sessions[-10:]
        
        insights = {
            "total_rl_sessions": len(self.optimization_sessions),
            "current_exploration_rate": self.epsilon,
            "average_rounds_per_session": statistics.mean([s.total_rounds for s in recent_sessions]),
            "convergence_rate": len([s for s in recent_sessions if s.convergence_achieved]) / len(recent_sessions),
            "average_score_improvement": 0.0,
            "exploration_performance": statistics.mean(self.explore_scores[-20:]) if len(self.explore_scores) >= 20 else 0.0,
            "exploitation_performance": statistics.mean(self.exploit_scores[-20:]) if len(self.exploit_scores) >= 20 else 0.0,
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
                    "explore_performance": perf.explore_performance,
                    "exploit_performance": perf.exploit_performance,
                    "attempts": perf.total_attempts,
                    "success_rate": perf.success_count / perf.total_attempts,
                    "confidence": perf.confidence_in_strategy,
                    "improvement_trend": perf.improvement_trend
                })
        
        return insights

def main():
    """Command line interface"""
    if len(sys.argv) < 2 or sys.argv[1] in ('-h', '--help'):
        print("Usage: python smart_prompt_optimizer_v4_1_rl_loop_improved.py \"prompt\" [--validate] [--insights]")
        print("\nCommands:")
        print("  \"prompt\"        Optimize with improved RL learning loop")
        print("  dummy --insights Show improved RL learning insights")
        print("\nOptions:")
        print("  --validate       Use real validation scores (recommended)")
        return
    
    agent = None
    try:
        if "--insights" in sys.argv:
            agent = ImprovedRLLoopAgent()
            insights = agent.get_rl_insights()
            print("\n🔄 IMPROVED RL LOOP AGENT INSIGHTS:")
            print("=" * 50)
            print(json.dumps(insights, indent=2))
            return
        
        user_prompt = sys.argv[1]
        use_validation = "--validate" in sys.argv
        
        print("🔄 IMPROVED RL LOOP AGENT - ADAPTIVE CONVERGENCE")
        print("=" * 60)
        print("✅ Minimum rounds before convergence")
        print("✅ Adaptive convergence thresholds")
        print("✅ Enhanced exploration/exploitation balance")
        print("✅ Performance-based epsilon adjustment")
        print("=" * 60)
        
        agent = ImprovedRLLoopAgent()
        result = agent.optimize_with_rl_loop(user_prompt, use_validation=use_validation)
        
        print("\n" + "="*20 + " IMPROVED RL LOOP SUMMARY " + "="*20)
        print(f"   Original: {result['original_prompt']}")
        print(f"   Final Best: {result['final_optimized_prompt']}")
        print(f"   Final Score: {result['final_score']:.4f}")
        print(f"   Total Rounds: {result['total_rounds']}")
        print(f"   Converged: {result['convergence_achieved']}")
        print(f"   Convergence Reason: {result['convergence_reason']}")
        print(f"   Exploration Ratio: {result['exploration_ratio']:.1%}")
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