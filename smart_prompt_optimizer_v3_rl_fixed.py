#!/usr/bin/env python3
"""
Smart Prompt Optimizer V3 - LLM AS RL AGENT
===========================================
🧠 LLM itself is the RL agent that learns through in-context examples
🔄 Real-time exploration, exploitation, and self-correction
🎯 No hardcoded patterns - pure in-context learning and adaptation
⚡ Self-improving through success/failure feedback loops

Revolutionary Design:
1. LLM maintains dynamic memory of successful/failed optimizations
2. Uses exploration vs exploitation trade-offs in real-time
3. Self-corrects based on validation feedback
4. Learns optimal strategies through multi-turn reasoning
5. Adapts its optimization approach based on accumulated experience
"""

import json
import requests
import time
import sys
import random
import numpy as np
import re
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path
from dataclasses import dataclass, asdict
import statistics
import subprocess
from datetime import datetime

@dataclass
class OptimizationMemory:
    """A memory of an optimization attempt with outcome"""
    original_prompt: str
    optimized_prompt: str
    validation_score: Optional[float]
    confidence_predicted: float
    strategy_used: str
    timestamp: float
    exploration_type: str  # 'explore' or 'exploit'
    success: bool
    feedback_reason: Optional[str] = None

@dataclass
class StrategyPerformance:
    """Performance tracking for different optimization strategies"""
    strategy_name: str
    success_count: int
    total_attempts: int
    avg_score: float
    confidence_accuracy: float
    last_used: float

class LLMRLAgent:
    """LLM-based RL agent that learns to optimize prompts through experience"""

    def __init__(self, ollama_url: str = "http://localhost:11434",
                 memory_file: str = "llm_rl_memory.json"):
        self.ollama_url = ollama_url
        self.model = "llama3" # Using a standard model name
        self.memory_file = Path(memory_file)

        # RL parameters
        self.epsilon = 0.3  # Exploration rate (starts high, decays)
        self.epsilon_decay = 0.99
        self.epsilon_min = 0.1

        # Memory systems
        self.optimization_memories: List[OptimizationMemory] = []
        self.strategy_performance: Dict[str, StrategyPerformance] = {}
        self.context_window_size = 10  # Number of recent memories to include

        # Learning state
        self.total_optimizations = 0
        self.successful_optimizations = 0
        self.current_session_start = time.time()

        self._load_memory()
        self._initialize_strategies()

        print("🧠 LLM RL AGENT INITIALIZED")
        print(f"   Memory: {len(self.optimization_memories)} past optimizations")
        print(f"   Strategies: {len(self.strategy_performance)} tracked")
        print(f"   Exploration rate: {self.epsilon:.2f}")

    def _load_memory(self):
        """Load persistent memory from disk"""
        if self.memory_file.exists():
            try:
                with open(self.memory_file, 'r') as f:
                    data = json.load(f)

                self.optimization_memories = [
                    OptimizationMemory(**mem) for mem in data.get('memories', [])
                ]

                strategy_data = data.get('strategies', {})
                self.strategy_performance = {
                    name: StrategyPerformance(**perf)
                    for name, perf in strategy_data.items()
                }

                self.total_optimizations = data.get('total_optimizations', 0)
                self.successful_optimizations = data.get('successful_optimizations', 0)
                self.epsilon = data.get('epsilon', self.epsilon)

                print(f"📚 Loaded {len(self.optimization_memories)} memories and {len(self.strategy_performance)} strategies from disk.")

            except (json.JSONDecodeError, TypeError) as e:
                print(f"⚠️ Failed to load or parse memory file: {e}. Starting fresh.")
                self.optimization_memories = []
                self.strategy_performance = {}
        else:
            print("📄 No memory file found. Starting with a fresh memory.")

    def _save_memory(self):
        """Save memory to disk"""
        try:
            data = {
                'memories': [asdict(mem) for mem in self.optimization_memories[-1000:]],  # Keep last 1000
                'strategies': {name: asdict(perf) for name, perf in self.strategy_performance.items()},
                'total_optimizations': self.total_optimizations,
                'successful_optimizations': self.successful_optimizations,
                'epsilon': self.epsilon,
                'last_updated': time.time()
            }

            with open(self.memory_file, 'w') as f:
                json.dump(data, f, indent=2)

        except Exception as e:
            print(f"⚠️ Failed to save memory: {e}")

    def _initialize_strategies(self):
        """Initialize strategy tracking if not loaded"""
        default_strategies = [
            "conservative_enhancement",
            "aggressive_transformation",
            "material_focus",
            "artistic_elaboration",
            "technical_precision",
            "contextual_scene_building",
            "minimalist_refinement"
        ]

        for strategy in default_strategies:
            if strategy not in self.strategy_performance:
                self.strategy_performance[strategy] = StrategyPerformance(
                    strategy_name=strategy,
                    success_count=0,
                    total_attempts=0,
                    avg_score=0.5,
                    confidence_accuracy=0.5,
                    last_used=0.0
                )

    def _select_strategy(self) -> Tuple[str, str]:
        """Select optimization strategy using epsilon-greedy with UCB"""
        if random.random() < self.epsilon or self.total_optimizations < len(self.strategy_performance):
            # EXPLORE: Try less-used or random strategies
            exploration_type = "explore"
            
            strategy_scores = []
            for name, perf in self.strategy_performance.items():
                if perf.total_attempts == 0:
                    exploration_bonus = float('inf')
                else:
                    # UCB1 exploration bonus
                    uncertainty = np.sqrt(2 * np.log(max(1, self.total_optimizations)) / perf.total_attempts)
                    exploration_bonus = uncertainty
                
                strategy_scores.append((name, exploration_bonus))
            
            strategy_scores.sort(key=lambda x: x[1], reverse=True)
            selected_strategy = strategy_scores[0][0]
        else:
            # EXPLOIT: Use best performing strategy
            exploration_type = "exploit"
            
            best_strategy = None
            best_score = -1
            
            for name, perf in self.strategy_performance.items():
                if perf.total_attempts == 0:
                    continue
                
                success_rate = perf.success_count / perf.total_attempts
                # Combine success rate and average score for a more nuanced choice
                combined_score = (success_rate * 0.7) + (perf.avg_score * 0.3)
                
                if combined_score > best_score:
                    best_score = combined_score
                    best_strategy = name
            
            selected_strategy = best_strategy or "conservative_enhancement"
        
        return selected_strategy, exploration_type

    def _build_context_memory(self, current_prompt: str) -> str:
        """Build in-context memory for the LLM"""
        if not self.optimization_memories:
            return "No previous optimization experience available."

        recent_memories = self.optimization_memories[-self.context_window_size:]
        successes = [mem for mem in recent_memories if mem.success and mem.validation_score is not None]
        failures = [mem for mem in recent_memories if not mem.success]

        context = "OPTIMIZATION EXPERIENCE MEMORY:\n\n"
        if successes:
            context += "--- SUCCESSFUL OPTIMIZATIONS (Examples to learn from) ---\n"
            for mem in successes[-5:]:
                context += (f"Original: {mem.original_prompt}\n"
                            f"Optimized: {mem.optimized_prompt}\n"
                            f"Strategy: {mem.strategy_used}, Score: {mem.validation_score:.3f}\n\n")
        
        if failures:
            context += "--- FAILED OPTIMIZATIONS (Examples to avoid) ---\n"
            for mem in failures[-3:]:
                context += (f"Original: {mem.original_prompt}\n"
                            f"Optimized: {mem.optimized_prompt}\n"
                            f"Strategy: {mem.strategy_used}, Reason: {mem.feedback_reason or 'Low score'}\n\n")

        context += "--- STRATEGY PERFORMANCE INSIGHTS ---\n"
        sorted_strategies = sorted(
            self.strategy_performance.items(),
            key=lambda x: (x[1].success_count / max(1, x[1].total_attempts), x[1].avg_score),
            reverse=True
        )
        for name, perf in sorted_strategies[:3]:
            if perf.total_attempts > 0:
                success_rate = perf.success_count / perf.total_attempts
                context += f"- {name}: {success_rate:.1%} success, avg score {perf.avg_score:.3f}\n"

        return context

    def _generate_optimization(self, prompt: str, strategy: str, exploration_type: str) -> Tuple[str, float]:
        """Generate optimization using LLM with strategy and memory context"""
        memory_context = self._build_context_memory(prompt)
        strategy_instructions = self._get_strategy_instructions(strategy, exploration_type)

        system_prompt = f"""You are an expert prompt optimization RL agent. Your goal is to refine a user's prompt to make it better for an image generation model. You learn from your past successes and failures.

{memory_context}

--- CURRENT TASK ---
Original Prompt: "{prompt}"
Chosen Strategy: {strategy} ({exploration_type} mode)

{strategy_instructions}

Based on your experience, generate an optimized prompt and predict your confidence in its success.

--- RESPONSE FORMAT ---
OPTIMIZED: [your optimized prompt here, starting with "wbgmsst," and ending with ", white background"]
CONFIDENCE: [a single float value between 0.0 and 1.0]
REASONING: [briefly explain your changes based on the strategy and memory]"""

        try:
            response = self._query_llama(system_prompt)
            return self._parse_optimization_response(response, prompt)
        except Exception as e:
            print(f"❌ Optimization generation failed: {e}")
            return f"wbgmsst, {prompt}, white background", 0.5

    def _get_strategy_instructions(self, strategy: str, exploration_type: str) -> str:
        """Get strategy-specific instructions"""
        base_strategies = {
            "conservative_enhancement": "Apply minimal but high-impact enhancements. Focus on adding 1-2 proven descriptive elements from successful memories.",
            "aggressive_transformation": "Be bold. Comprehensively rewrite the prompt, adding rich detail, context, and multiple enhancement layers.",
            "material_focus": "Zoom in on materials. Emphasize textures, finishes (e.g., matte, glossy), and physical properties (e.g., solid, translucent).",
            "artistic_elaboration": "Add artistic and aesthetic elements. Specify lighting (e.g., cinematic, soft), mood, and a specific artistic style (e.g., photorealistic, concept art).",
            "technical_precision": "Focus on technical details. Use terms that convey high quality, precision, and craftsmanship (e.g., 'intricate details', '4k').",
            "contextual_scene_building": "Build a scene. Place the object in a simple, non-distracting environment to give it context.",
            "minimalist_refinement": "Less is more. Refine existing words for clarity and impact rather than adding new concepts. Shorten where possible."
        }
        instruction = f"STRATEGY INSTRUCTIONS: {base_strategies.get(strategy, base_strategies['conservative_enhancement'])}"
        if exploration_type == "explore":
            instruction += "\nSince you are EXPLORING, feel free to experiment with novel combinations or variations on successful patterns."
        else:
            instruction += "\nSince you are EXPLOITING, stick closely to patterns that have proven successful in your memory."
        return instruction

    def _parse_optimization_response(self, response: str, original_prompt: str) -> Tuple[str, float]:
        """Parse LLM response to extract optimization and confidence using regex for robustness."""
        optimized_prompt = ""
        confidence = 0.5

        # Use regex to find OPTIMIZED line, case-insensitive, multiline
        opt_match = re.search(r"OPTIMIZED:\s*(.*)", response, re.IGNORECASE | re.DOTALL)
        if opt_match:
            optimized_prompt = opt_match.group(1).strip().split('\n')[0] # Take first line after match

        # Use regex to find CONFIDENCE line
        conf_match = re.search(r"CONFIDENCE:\s*([0-9.]+)", response, re.IGNORECASE)
        if conf_match:
            try:
                confidence = float(conf_match.group(1))
                confidence = max(0.0, min(1.0, confidence))  # Clamp to [0,1]
            except (ValueError, IndexError):
                confidence = 0.5

        # Fallback if regex fails
        if not optimized_prompt:
            # Use original prompt as a safe fallback
            optimized_prompt = original_prompt

        # --- Robust Prompt Cleaning and Formatting ---
        # 1. Remove surrounding quotes
        core_content = optimized_prompt.strip().strip('"')
        
        # 2. Remove potential prefixes/suffixes added by the LLM
        core_content = re.sub(r"^\s*wbgmsst\s*,?", "", core_content, flags=re.IGNORECASE).strip()
        core_content = re.sub(r",?\s*white background\s*$", "", core_content, flags=re.IGNORECASE).strip()
        
        # 3. Rebuild the prompt with the required structure
        final_prompt = f"wbgmsst, {core_content}, white background"

        return final_prompt, confidence

    def optimize(self, prompt: str, use_validation: bool = False) -> Dict[str, Any]:
        """Main optimization method with RL learning"""
        start_time = time.time()
        print(f"\n🧠 RL AGENT OPTIMIZING: '{prompt}'")

        strategy, exploration_type = self._select_strategy()
        print(f"   🎯 Strategy: {strategy} ({exploration_type})")

        optimized_prompt, predicted_confidence = self._generate_optimization(
            prompt, strategy, exploration_type
        )

        actual_score = None
        if use_validation:
            actual_score = self._validate_prompt(optimized_prompt)

        success = self._evaluate_success(predicted_confidence, actual_score)
        feedback = self._get_feedback_reason(predicted_confidence, actual_score)

        memory = OptimizationMemory(
            original_prompt=prompt,
            optimized_prompt=optimized_prompt,
            validation_score=actual_score,
            confidence_predicted=predicted_confidence,
            strategy_used=strategy,
            timestamp=time.time(),
            exploration_type=exploration_type,
            success=success,
            feedback_reason=feedback
        )

        self._update_learning(memory)
        processing_time = time.time() - start_time

        result = {
            'original_prompt': prompt,
            'optimized_prompt': optimized_prompt,
            'predicted_confidence': predicted_confidence,
            'actual_validation_score': actual_score,
            'strategy_used': strategy,
            'exploration_type': exploration_type,
            'success': success,
            'processing_time': processing_time,
            'total_experience': len(self.optimization_memories),
            'success_rate': self.successful_optimizations / max(1, self.total_optimizations),
            'current_epsilon': self.epsilon
        }

        print(f"✅ RESULT: {optimized_prompt}")
        print(f"⏱️   Time: {processing_time:.2f}s | Confidence: {predicted_confidence:.1%}")
        if actual_score is not None:
            print(f"📊   Validation Score: {actual_score:.3f} -> Success: {success}")
        else:
            print(f"👍   Success (confidence-based): {success}")
        print(f"📈   Experience: {len(self.optimization_memories)} memories | Overall Success Rate: {result['success_rate']:.1%}")

        return result

    def _evaluate_success(self, predicted_confidence: float, actual_score: Optional[float]) -> bool:
        """Evaluate if the optimization was successful"""
        if actual_score is not None:
            # With validation, success is a high score
            return actual_score >= 0.7
        else:
            # Without validation, success is based on high confidence
            return predicted_confidence >= 0.65

    def _get_feedback_reason(self, predicted_confidence: float, actual_score: Optional[float]) -> Optional[str]:
        """Generate feedback reason for learning"""
        if actual_score is None:
            return "Success based on confidence only" if predicted_confidence >= 0.65 else "Low confidence"
        
        if actual_score < 0.5:
            return "Low validation score"
        elif abs(predicted_confidence - actual_score) > 0.4:
            return f"Poor confidence calibration (predicted {predicted_confidence:.2f}, got {actual_score:.2f})"
        elif actual_score >= 0.8:
            return "High validation score"
        else:
            return "Moderate performance"

    def _update_learning(self, memory: OptimizationMemory):
        """Update agent's learning from new experience"""
        self.optimization_memories.append(memory)
        self.total_optimizations += 1

        strategy_perf = self.strategy_performance[memory.strategy_used]
        strategy_perf.total_attempts += 1
        strategy_perf.last_used = memory.timestamp

        if memory.success:
            strategy_perf.success_count += 1
            self.successful_optimizations += 1

        if memory.validation_score is not None:
            alpha = 0.1  # Learning rate for moving averages
            strategy_perf.avg_score = (1 - alpha) * strategy_perf.avg_score + alpha * memory.validation_score
            
            confidence_error = abs(memory.confidence_predicted - memory.validation_score)
            accuracy = max(0, 1 - confidence_error)
            strategy_perf.confidence_accuracy = (1 - alpha) * strategy_perf.confidence_accuracy + alpha * accuracy

        # Decay exploration rate after each step
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

        # Save memory after every single update to ensure persistence
        self._save_memory()
        print(f"🧠   Learning updated: ε={self.epsilon:.3f}. Memory saved.")

    def _validate_prompt(self, prompt: str) -> float:
        """Run actual validation by calling an external script."""
        try:
            print("   🔍 Running validation...")
            # Ensure the validator script is executable and in the right path
            cmd = [sys.executable, "subnet_accurate_validator.py", prompt]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300, check=False)
            
            if result.returncode == 0:
                # Assuming validator script outputs JSON to a file
                with open("subnet_validation_results.json", 'r') as f:
                    data = json.load(f)
                    score = data.get("validation_engine_score", 0.0)
                    return float(score)
            else:
                print(f"   ⚠️ Validator script returned error (code {result.returncode}): {result.stderr}")
                return 0.0
        except FileNotFoundError:
            print("   ❌ Validation script 'subnet_accurate_validator.py' not found.")
            return 0.0
        except Exception as e:
            print(f"   ❌ Validation failed: {e}")
            return 0.0

    def _query_llama(self, prompt: str) -> str:
        """Query the LLM API."""
        data = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "stream": False,
            "options": {
                "temperature": 0.7,
                "num_predict": 300,
                "top_p": 0.9
            }
        }
        response = requests.post(f"{self.ollama_url}/api/chat", json=data, timeout=60)
        response.raise_for_status()
        return response.json()["message"]["content"].strip()

    def get_learning_insights(self) -> Dict[str, Any]:
        """Get insights about the agent's learning progress"""
        if not self.optimization_memories:
            return {"message": "No learning experience yet"}

        insights = {
            "total_experience": len(self.optimization_memories),
            "overall_success_rate": self.successful_optimizations / max(1, self.total_optimizations),
            "current_exploration_rate": self.epsilon,
            "strategy_performance": []
        }

        for name, perf in sorted(self.strategy_performance.items(), key=lambda item: item[1].total_attempts, reverse=True):
            if perf.total_attempts > 0:
                insights["strategy_performance"].append({
                    "name": name,
                    "success_rate": perf.success_count / perf.total_attempts,
                    "avg_score": perf.avg_score,
                    "confidence_accuracy": perf.confidence_accuracy,
                    "attempts": perf.total_attempts
                })
        
        return insights

def main():
    """Command line interface"""
    if len(sys.argv) < 2 or sys.argv[1] in ('-h', '--help'):
        print("Usage: python smart_prompt_optimizer_v3_rl_fixed.py \"your prompt here\" [--validate] [--insights]")
        print("\nCommands:")
        print("  \"your prompt\"  The prompt to optimize.")
        print("  dummy --insights  Show learning insights from memory.")
        print("\nOptions:")
        print("  --validate      Run external validation for more accurate learning.")
        return

    agent = None
    try:
        if "--insights" in sys.argv:
            agent = LLMRLAgent()
            insights = agent.get_learning_insights()
            print("\n📊 RL AGENT LEARNING INSIGHTS:")
            print("=" * 50)
            print(json.dumps(insights, indent=2))
            return

        user_prompt = sys.argv[1]
        use_validation = "--validate" in sys.argv

        print("=" * 60)
        print("      🧠 LLM RL AGENT - SELF-IMPROVING PROMPT OPTIMIZER 🧠")
        print("=" * 60)
        
        agent = LLMRLAgent()
        result = agent.optimize(user_prompt, use_validation=use_validation)
        
        print("\n" + "="*25 + " FINAL SUMMARY " + "="*24)
        print(f"   Original: {result['original_prompt']}")
        print(f"   Optimized: {result['optimized_prompt']}")
        print(f"   Strategy: {result['strategy_used']} ({result['exploration_type']})")
        print(f"   Confidence: {result['predicted_confidence']:.1%}")
        if result['actual_validation_score'] is not None:
            print(f"   Validation Score: {result['actual_validation_score']:.3f}")
        print("="*60)

    except KeyboardInterrupt:
        print(f"\n⚠️ Interrupted by user.")
    except Exception as e:
        print(f"\n❌ An unexpected error occurred: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if agent:
            print("\n💾 Attempting to save final memory state...")
            agent._save_memory()
            print("✅ Memory saved.")

if __name__ == "__main__":
    main() 