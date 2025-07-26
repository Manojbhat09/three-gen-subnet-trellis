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

class RLLoopAgent:
    """RL agent that learns through iterative optimization loops"""
    
    def __init__(self, ollama_url: str = "http://localhost:11434",
                 memory_file: str = "rl_loop_memory.json"):
        self.ollama_url = ollama_url
        self.model = "llama3.2:3b"
        self.memory_file = Path(memory_file)
        
        # RL Loop parameters
        self.max_optimization_rounds = 5
        self.convergence_threshold = 0.02  # Stop if improvement < 2%
        self.min_score_threshold = 0.85   # Target score to achieve
        
        # Learning state
        self.strategy_performance: Dict[str, StrategyPerformance] = {}
        self.optimization_sessions: List[RLOptimizationSession] = []
        self.global_insights: List[str] = []
        
        # RL parameters
        self.epsilon = 0.4  # Higher exploration for RL loop
        self.epsilon_decay = 0.95
        self.epsilon_min = 0.1
        
        self.logger = logging.getLogger(__name__)

        self._load_memory()
        self._initialize_strategies()
        
        self.logger.info(f"🔄 RL LOOP AGENT INITIALIZED")
        self.logger.info(f"   Strategy tracking: {len(self.strategy_performance)} strategies")
        self.logger.info(f"   Past sessions: {len(self.optimization_sessions)}")
        self.logger.info(f"   Exploration rate: {self.epsilon:.2f}")
        self.logger.info(f"   Max rounds per optimization: {self.max_optimization_rounds}")
    
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
    
    def optimize_with_rl_loop(self, prompt: str, use_validation: bool = True) -> Dict[str, Any]:
        """Main RL optimization loop with iterative learning"""
        session_id = f"rl_session_{int(time.time())}_{random.randint(1000, 9999)}"
        start_time = time.time()
        
        self.logger.info(f"\n🔄 RL LOOP OPTIMIZATION: '{prompt}'")
        self.logger.info(f"   Session: {session_id}")
        self.logger.info(f"   Max rounds: {self.max_optimization_rounds}")
        self.logger.info(f"   Target score: {self.min_score_threshold}")
        
        attempts: List[OptimizationAttempt] = []
        best_prompt = f"wbgmsst, {prompt}, white background"
        best_score = 0.0
        convergence_achieved = False
        
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
            
            # Update best if this is better
            if attempt.validation_score and attempt.validation_score > best_score:
                best_score = attempt.validation_score
                best_prompt = attempt.optimized_prompt
                self.logger.info(f"      🎯 New best score: {best_score:.4f}")
            
            # Update strategy performance immediately
            self._update_strategy_performance(strategy, attempt.validation_score or 0.0)
            
            # Check for convergence
            if round_num > 1:
                score_improvement = (attempt.validation_score or 0.0) - (attempts[-2].validation_score or 0.0)
                if abs(score_improvement) < self.convergence_threshold:
                    self.logger.info(f"      ✅ Convergence achieved (improvement: {score_improvement:.4f})")
                    convergence_achieved = True
                    break
                
                # Early stop if we hit target
                if attempt.validation_score and attempt.validation_score >= self.min_score_threshold:
                    self.logger.info(f"      🎯 Target score achieved: {attempt.validation_score:.4f}")
                    convergence_achieved = True
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
            session_duration=time.time() - start_time
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
            'learned_insights': learned_insights,
            'strategy_updates': strategy_updates,
            'processing_time': session.session_duration,
            'score_progression': [a.validation_score for a in attempts],
            'strategy_sequence': [a.strategy_used for a in attempts]
        }
        
        self.logger.info(f"\n🎯 RL LOOP COMPLETE:")
        self.logger.info(f"   Best prompt: {best_prompt}")
        self.logger.info(f"   Best score: {best_score:.4f}")
        self.logger.info(f"   Rounds: {len(attempts)}")
        self.logger.info(f"   Convergence: {convergence_achieved}")
        self.logger.info(f"   Insights learned: {len(learned_insights)}")
        self.logger.info(f"   Total time: {session.session_duration:.2f}s")
        
        return result
    
    def _select_strategy_for_rl(self) -> Tuple[str, str]:
        """Select strategy for RL with performance-based selection"""
        if random.random() < self.epsilon:
            # EXPLORE: Try strategies with uncertainty or poor recent performance
            exploration_type = "explore"
            
            # Prefer strategies with high uncertainty or recent poor performance
            strategy_scores = []
            for name, perf in self.strategy_performance.items():
                if perf.total_attempts == 0:
                    uncertainty = float('inf')
                else:
                    # Combine uncertainty with recent performance trend
                    uncertainty = 1.0 / (perf.total_attempts + 1)
                    if perf.improvement_trend < 0:  # Getting worse
                        uncertainty += 0.5
                
                strategy_scores.append((name, uncertainty))
            
            strategy_scores.sort(key=lambda x: x[1], reverse=True)
            selected_strategy = strategy_scores[0][0]
            
        else:
            # EXPLOIT: Use best performing strategy
            exploration_type = "exploit"
            
            best_strategy = None
            best_combined_score = -1
            
            for name, perf in self.strategy_performance.items():
                if perf.total_attempts == 0:
                    continue
                
                # Combine average score with recent trend and confidence
                combined_score = (perf.avg_score * 0.5 + 
                                perf.confidence_in_strategy * 0.3 + 
                                max(0, perf.improvement_trend) * 0.2)
                
                if combined_score > best_combined_score:
                    best_combined_score = combined_score
                    best_strategy = name
            
            selected_strategy = best_strategy or "conservative_enhancement"
        
        return selected_strategy, exploration_type
    
    def _make_optimization_attempt(self, prompt: str, strategy: str, exploration_type: str, 
                                 round_num: int, previous_attempts: List[OptimizationAttempt]) -> OptimizationAttempt:
        """Make a single optimization attempt with context from previous rounds"""
        
        self.logger.info(f"      🎯 Strategy: {strategy} ({exploration_type})")
        
        # Build context from previous attempts
        previous_context = self._build_previous_attempts_context(previous_attempts)
        strategy_context = self._build_strategy_context(strategy)
        
        system_prompt = f"""You are an RL agent learning to optimize prompts. This is round {round_num} of iterative optimization.

ORIGINAL PROMPT: "{prompt}"
STRATEGY: {strategy} ({exploration_type} mode)

{strategy_context}

{previous_context}

TASK: Create an optimized prompt that improves on previous attempts based on learned insights.

Rules:
- Start with "wbgmsst," and end with ", white background"
- Learn from previous rounds' scores and approaches
- Be {'experimental' if exploration_type == 'explore' else 'systematic'} in your approach

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
        """Save RL loop memory"""
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
                'last_updated': time.time()
            }
            
            with open(self.memory_file, 'w') as f:
                json.dump(data, f, indent=2)
                
        except Exception as e:
            self.logger.warning(f"⚠️ Failed to save RL memory: {e}")
    
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

def main():
    """Command line interface"""
    if len(sys.argv) < 2 or sys.argv[1] in ('-h', '--help'):
        print("Usage: python smart_prompt_optimizer_v4_1_rl_loop.py \"prompt\" [--validate] [--insights]")
        print("\nCommands:")
        print("  \"prompt\"        Optimize with RL learning loop")
        print("  dummy --insights Show RL learning insights")
        print("\nOptions:")
        print("  --validate       Use real validation scores (recommended)")
        return
    
    agent = None
    try:
        if "--insights" in sys.argv:
            agent = RLLoopAgent()
            insights = agent.get_rl_insights()
            print("\n🔄 RL LOOP AGENT INSIGHTS:")
            print("=" * 50)
            print(json.dumps(insights, indent=2))
            return
        
        user_prompt = sys.argv[1]
        use_validation = "--validate" in sys.argv
        
        print("🔄 RL LOOP AGENT - TRUE ITERATIVE LEARNING")
        print("=" * 60)
        print("✅ Multi-round optimization with score feedback")
        print("✅ Strategy performance updates based on results")
        print("✅ Convergence detection and early stopping")
        print("✅ Exploration/exploitation based on performance")
        print("=" * 60)
        
        agent = RLLoopAgent()
        result = agent.optimize_with_rl_loop(user_prompt, use_validation=use_validation)
        
        print("\n" + "="*20 + " RL LOOP SUMMARY " + "="*20)
        print(f"   Original: {result['original_prompt']}")
        print(f"   Final Best: {result['final_optimized_prompt']}")
        print(f"   Final Score: {result['final_score']:.4f}")
        print(f"   Total Rounds: {result['total_rounds']}")
        print(f"   Converged: {result['convergence_achieved']}")
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