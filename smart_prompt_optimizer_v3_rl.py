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
import csv
import random
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path
from dataclasses import dataclass, asdict
import statistics
import subprocess
import pickle
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
        self.model = "llama3.2:3b"
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
                
                print(f"📚 Loaded {len(self.optimization_memories)} memories from disk")
                
            except Exception as e:
                print(f"⚠️ Failed to load memory: {e}")
                self.optimization_memories = []
                self.strategy_performance = {}
    
    def _save_memory(self):
        """Save memory to disk"""
        try:
            data = {
                'memories': [asdict(mem) for mem in self.optimization_memories[-1000:]],  # Keep last 1000
                'strategies': {name: asdict(perf) for name, perf in self.strategy_performance.items()},
                'total_optimizations': self.total_optimizations,
                'successful_optimizations': self.successful_optimizations,
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
        
        # Exploration vs Exploitation decision
        if random.random() < self.epsilon:
            # EXPLORE: Try less-used or random strategies
            exploration_type = "explore"
            
            # Choose strategy that hasn't been used recently or has high uncertainty
            strategy_scores = []
            for name, perf in self.strategy_performance.items():
                # UCB1-like exploration bonus
                if perf.total_attempts == 0:
                    exploration_bonus = float('inf')
                else:
                    time_bonus = max(0, time.time() - perf.last_used) / 3600  # Hours since last use
                    uncertainty = np.sqrt(2 * np.log(self.total_optimizations + 1) / perf.total_attempts)
                    exploration_bonus = uncertainty + time_bonus * 0.1
                
                strategy_scores.append((name, exploration_bonus))
            
            # Select strategy with highest exploration bonus
            strategy_scores.sort(key=lambda x: x[1], reverse=True)
            selected_strategy = strategy_scores[0][0]
            
        else:
            # EXPLOIT: Use best performing strategy
            exploration_type = "exploit"
            
            if not self.strategy_performance:
                selected_strategy = "conservative_enhancement"
            else:
                # Select strategy with best success rate, weighted by confidence
                best_strategy = None
                best_score = -1
                
                for name, perf in self.strategy_performance.items():
                    if perf.total_attempts == 0:
                        continue
                    
                    success_rate = perf.success_count / perf.total_attempts
                    confidence_weight = perf.confidence_accuracy
                    score_weight = (perf.avg_score - 0.5) * 2  # Normalize around 0.5
                    
                    combined_score = (success_rate * 0.5 + 
                                    confidence_weight * 0.3 + 
                                    score_weight * 0.2)
                    
                    if combined_score > best_score:
                        best_score = combined_score
                        best_strategy = name
                
                selected_strategy = best_strategy or "conservative_enhancement"
        
        return selected_strategy, exploration_type
    
    def _build_context_memory(self, current_prompt: str) -> str:
        """Build in-context memory for the LLM"""
        
        if not self.optimization_memories:
            return "No previous optimization experience available."
        
        # Get relevant recent memories
        recent_memories = self.optimization_memories[-self.context_window_size:]
        
        # Separate successes and failures
        successes = [mem for mem in recent_memories if mem.success and mem.validation_score is not None]
        failures = [mem for mem in recent_memories if not mem.success]
        
        context = "OPTIMIZATION EXPERIENCE MEMORY:\n\n"
        
        # Add successful examples
        if successes:
            context += "SUCCESSFUL OPTIMIZATIONS (high validation scores):\n"
            for i, mem in enumerate(successes[-5:], 1):  # Last 5 successes
                context += f"Success {i}:\n"
                context += f"  Original: {mem.original_prompt}\n"
                context += f"  Optimized: {mem.optimized_prompt}\n"
                context += f"  Score: {mem.validation_score:.3f}\n"
                context += f"  Strategy: {mem.strategy_used}\n"
                context += f"  Confidence predicted: {mem.confidence_predicted:.2f}\n\n"
        
        # Add failure examples for learning
        if failures:
            context += "FAILED OPTIMIZATIONS (learn what to avoid):\n"
            for i, mem in enumerate(failures[-3:], 1):  # Last 3 failures
                context += f"Failure {i}:\n"
                context += f"  Original: {mem.original_prompt}\n"
                context += f"  Optimized: {mem.optimized_prompt}\n"
                context += f"  Strategy: {mem.strategy_used}\n"
                context += f"  Reason: {mem.feedback_reason or 'Unknown'}\n\n"
        
        # Add strategy performance insights
        if self.strategy_performance:
            context += "STRATEGY PERFORMANCE INSIGHTS:\n"
            sorted_strategies = sorted(
                self.strategy_performance.items(),
                key=lambda x: x[1].success_count / max(1, x[1].total_attempts),
                reverse=True
            )
            
            for name, perf in sorted_strategies[:3]:  # Top 3 strategies
                if perf.total_attempts > 0:
                    success_rate = perf.success_count / perf.total_attempts
                    context += f"  {name}: {success_rate:.1%} success, avg score {perf.avg_score:.3f}\n"
        
        return context
    
    def _generate_optimization(self, prompt: str, strategy: str, exploration_type: str) -> Tuple[str, float]:
        """Generate optimization using LLM with strategy and memory context"""
        
        # Build context from memory
        memory_context = self._build_context_memory(prompt)
        
        # Create strategy-specific instructions
        strategy_instructions = self._get_strategy_instructions(strategy, exploration_type)
        
        # Build the full system prompt
        system_prompt = f"""You are an expert prompt optimization RL agent that learns from experience.

{memory_context}

CURRENT TASK:
Optimize this prompt: "{prompt}"

STRATEGY TO USE: {strategy}
EXPLORATION MODE: {exploration_type}

{strategy_instructions}

You must learn from the successful examples above and avoid patterns that led to failures.

CRITICAL REQUIREMENTS:
1. Start with "wbgmsst," and end with ", white background"
2. Analyze the successful patterns in your memory
3. Apply the chosen strategy while considering past successes/failures
4. Predict your confidence (0.0-1.0) in this optimization

RESPONSE FORMAT:
OPTIMIZED: [your optimized prompt here]
CONFIDENCE: [0.0-1.0]
REASONING: [why you chose this approach based on your experience]"""

        try:
            response = self._query_llama(system_prompt)
            return self._parse_optimization_response(response)
        except Exception as e:
            print(f"❌ Optimization generation failed: {e}")
            return f"wbgmsst, {prompt}, white background", 0.5
    
    def _get_strategy_instructions(self, strategy: str, exploration_type: str) -> str:
        """Get strategy-specific instructions"""
        
        base_strategies = {
            "conservative_enhancement": """
Apply minimal but high-impact enhancements. Focus on proven successful patterns from memory.
Add 1-2 key descriptive elements that have worked well before.""",
            
            "aggressive_transformation": """
Apply bold, comprehensive transformations. Experiment with new descriptive approaches.
Add rich detail, context, and multiple enhancement layers.""",
            
            "material_focus": """
Emphasize material properties, textures, and physical characteristics.
Focus on words that enhance the tactile and visual qualities of the object.""",
            
            "artistic_elaboration": """
Add artistic and aesthetic elements. Focus on visual appeal, composition, and artistic style.
Include lighting, mood, and artistic presentation elements.""",
            
            "technical_precision": """
Emphasize precision, craftsmanship, and technical excellence.
Use words that convey skill, accuracy, and technical mastery.""",
            
            "contextual_scene_building": """
Build context and environment around the object.
Add setting, placement, and environmental details.""",
            
            "minimalist_refinement": """
Make subtle but impactful improvements. Focus on quality over quantity.
Refine existing elements rather than adding many new ones."""
        }
        
        strategy_instruction = base_strategies.get(strategy, base_strategies["conservative_enhancement"])
        
        if exploration_type == "explore":
            strategy_instruction += "\n\nEXPLORATION MODE: Try new approaches, experiment with variations of successful patterns."
        else:
            strategy_instruction += "\n\nEXPLOITATION MODE: Stick to proven successful patterns from your memory."
        
        return strategy_instruction
    
    def _parse_optimization_response(self, response: str) -> Tuple[str, float]:
        """Parse LLM response to extract optimization and confidence"""
        lines = response.strip().split('\n')
        
        optimized_prompt = ""
        confidence = 0.5
        
        for line in lines:
            line = line.strip()
            if line.startswith("OPTIMIZED:"):
                optimized_prompt = line.replace("OPTIMIZED:", "").strip()
            elif line.startswith("CONFIDENCE:"):
                try:
                    confidence = float(line.replace("CONFIDENCE:", "").strip())
                    confidence = max(0.0, min(1.0, confidence))  # Clamp to [0,1]
                except:
                    confidence = 0.5
        
        # Clean up the prompt
        if not optimized_prompt:
            # Extract from first substantial line that looks like a prompt
            for line in lines:
                line_clean = line.strip()
                if len(line_clean) > 20 and ('wbgmsst' in line_clean.lower() or 
                                           not any(x in line_clean.lower() for x in ['reasoning:', 'confidence:', 'strategy:'])):
                    optimized_prompt = line_clean
                    break
        
        # Remove quotes and clean
        optimized_prompt = optimized_prompt.replace('"', '').strip()
        
        # Fix double wbgmsst issue
        while optimized_prompt.count('wbgmsst') > 1:
            parts = optimized_prompt.split('wbgmsst')
            optimized_prompt = 'wbgmsst' + ''.join(parts[1:])
        
        # Ensure proper format
        if not optimized_prompt.startswith('wbgmsst'):
            optimized_prompt = f"wbgmsst, {optimized_prompt}"
        
        # Fix multiple white background
        optimized_prompt = optimized_prompt.replace(', white background, white background', ', white background')
        if not optimized_prompt.endswith(', white background'):
            optimized_prompt = optimized_prompt.rstrip(', ') + ", white background"
        
        return optimized_prompt, confidence
    
    def optimize(self, prompt: str, use_validation: bool = False) -> Dict[str, Any]:
        """Main optimization method with RL learning"""
        start_time = time.time()
        
        print(f"\n🧠 RL AGENT OPTIMIZING: '{prompt}'")
        
        # Select strategy using epsilon-greedy
        strategy, exploration_type = self._select_strategy()
        print(f"   🎯 Strategy: {strategy} ({exploration_type})")
        
        # Generate optimization
        optimized_prompt, predicted_confidence = self._generate_optimization(
            prompt, strategy, exploration_type
        )
        
        # Validate if requested
        actual_score = None
        if use_validation:
            actual_score = self._validate_prompt(optimized_prompt)
        
        # Determine success and create memory
        success = self._evaluate_success(predicted_confidence, actual_score)
        
        memory = OptimizationMemory(
            original_prompt=prompt,
            optimized_prompt=optimized_prompt,
            validation_score=actual_score,
            confidence_predicted=predicted_confidence,
            strategy_used=strategy,
            timestamp=time.time(),
            exploration_type=exploration_type,
            success=success,
            feedback_reason=self._get_feedback_reason(predicted_confidence, actual_score)
        )
        
        # Update agent's learning
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
        print(f"⏱️  Time: {processing_time:.2f}s | Confidence: {predicted_confidence:.1%}")
        print(f"🎯 Strategy: {strategy} | Success: {success}")
        if actual_score:
            print(f"📊 Validation: {actual_score:.3f}")
        print(f"📈 Experience: {len(self.optimization_memories)} memories | Success rate: {result['success_rate']:.1%}")
        
        return result
    
    def _evaluate_success(self, predicted_confidence: float, actual_score: Optional[float]) -> bool:
        """Evaluate if the optimization was successful"""
        if actual_score is not None:
            # Success if score is above threshold and confidence was reasonably accurate
            score_success = actual_score >= 0.7
            confidence_accuracy = abs(predicted_confidence - actual_score) < 0.3
            return score_success and confidence_accuracy
        else:
            # Without validation, consider success if confidence is reasonable
            return predicted_confidence >= 0.6
    
    def _get_feedback_reason(self, predicted_confidence: float, actual_score: Optional[float]) -> Optional[str]:
        """Generate feedback reason for learning"""
        if actual_score is None:
            return None
        
        if actual_score < 0.5:
            return "Low validation score - optimization may have been too aggressive or inappropriate"
        elif abs(predicted_confidence - actual_score) > 0.4:
            return f"Poor confidence calibration - predicted {predicted_confidence:.2f}, got {actual_score:.2f}"
        elif actual_score >= 0.8:
            return "High validation score - this approach worked well"
        else:
            return "Moderate performance - approach was okay but could be improved"
    
    def _update_learning(self, memory: OptimizationMemory):
        """Update agent's learning from new experience"""
        
        # Add to memory
        self.optimization_memories.append(memory)
        
        # Update strategy performance
        strategy_perf = self.strategy_performance[memory.strategy_used]
        strategy_perf.total_attempts += 1
        strategy_perf.last_used = memory.timestamp
        
        if memory.success:
            strategy_perf.success_count += 1
            self.successful_optimizations += 1
        
        if memory.validation_score is not None:
            # Update average score with exponential moving average
            alpha = 0.1
            strategy_perf.avg_score = (1 - alpha) * strategy_perf.avg_score + alpha * memory.validation_score
            
            # Update confidence accuracy
            confidence_error = abs(memory.confidence_predicted - memory.validation_score)
            accuracy = max(0, 1 - confidence_error)
            strategy_perf.confidence_accuracy = (1 - alpha) * strategy_perf.confidence_accuracy + alpha * accuracy
        
        self.total_optimizations += 1
        
        # Decay exploration rate
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        
        # Save memory periodically
        if len(self.optimization_memories) % 5 == 0:  # Every 5 optimizations
            self._save_memory()
        
        print(f"🧠 Learning updated: ε={self.epsilon:.3f}, Strategy {memory.strategy_used} performance updated")
    
    def _validate_prompt(self, prompt: str) -> float:
        """Run actual validation"""
        try:
            print("   🔍 Running validation...")
            cmd = [sys.executable, "subnet_accurate_validator.py", prompt]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
            
            if result.returncode == 0:
                with open("subnet_validation_results.json", 'r') as f:
                    data = json.load(f)
                    return data.get("validation_engine_score", 0.0)
            return 0.0
        except Exception as e:
            print(f"   ❌ Validation failed: {e}")
            return 0.0
    
    def _query_llama(self, prompt: str) -> str:
        """Query LLaMA with the optimization prompt"""
        data = {
            "model": self.model,
            "messages": [
                {"role": "user", "content": prompt}
            ],
            "stream": False,
            "options": {
                "temperature": 0.7,  # Higher for exploration
                "num_predict": 300,
                "top_p": 0.9
            }
        }
        
        response = requests.post(f"{self.ollama_url}/api/chat", json=data, timeout=30)
        response.raise_for_status()
        return response.json()["message"]["content"].strip()
    
    def get_learning_insights(self) -> Dict[str, Any]:
        """Get insights about the agent's learning progress"""
        if not self.optimization_memories:
            return {"message": "No learning experience yet"}
        
        recent_memories = self.optimization_memories[-20:]  # Last 20
        
        insights = {
            "total_experience": len(self.optimization_memories),
            "success_rate": self.successful_optimizations / self.total_optimizations,
            "current_exploration_rate": self.epsilon,
            "recent_performance": {
                "successes": len([m for m in recent_memories if m.success]),
                "total": len(recent_memories),
                "avg_confidence": statistics.mean([m.confidence_predicted for m in recent_memories]),
            },
            "best_strategies": []
        }
        
        # Analyze strategy performance
        for name, perf in self.strategy_performance.items():
            if perf.total_attempts > 0:
                insights["best_strategies"].append({
                    "name": name,
                    "success_rate": perf.success_count / perf.total_attempts,
                    "avg_score": perf.avg_score,
                    "attempts": perf.total_attempts
                })
        
        insights["best_strategies"].sort(key=lambda x: x["success_rate"], reverse=True)
        
        return insights

def main():
    """Command line interface"""
    if len(sys.argv) < 2:
        print("Usage: python smart_prompt_optimizer_v3_rl.py \"your prompt here\" [--validate] [--insights]")
        print("\nOptions:")
        print("  --validate    Run actual validation for RL learning")
        print("  --insights    Show learning insights")
        return
    
    if "--insights" in sys.argv:
        agent = LLMRLAgent()
        insights = agent.get_learning_insights()
        print("\n📊 RL AGENT LEARNING INSIGHTS:")
        print("=" * 50)
        print(json.dumps(insights, indent=2))
        return
    
    user_prompt = sys.argv[1]
    use_validation = "--validate" in sys.argv
    
    print("🧠 LLM RL AGENT - SELF-IMPROVING PROMPT OPTIMIZER")
    print("=" * 60)
    print("🔄 Real-time exploration, exploitation, and learning")
    print("🎯 Self-correcting through experience")
    print("💡 In-context learning with memory")
    print("📈 Strategy performance tracking")
    print("=" * 60)
    
    try:
        agent = LLMRLAgent()
        result = agent.optimize(user_prompt, use_validation=use_validation)
        
        print(f"\n📋 OPTIMIZATION COMPLETE:")
        print(f"   Original: {result['original_prompt']}")
        print(f"   Optimized: {result['optimized_prompt']}")
        print(f"   Strategy: {result['strategy_used']} ({result['exploration_type']})")
        print(f"   Confidence: {result['predicted_confidence']:.1%}")
        print(f"   Success: {result['success']}")
        print(f"   Experience: {result['total_experience']} memories")
        print(f"   Overall success rate: {result['success_rate']:.1%}")
        print(f"   Exploration rate: {result['current_epsilon']:.3f}")
        
        if result['actual_validation_score'] is not None:
            print(f"   🎯 Validation: {result['actual_validation_score']:.3f}")
        
    except KeyboardInterrupt:
        print(f"\n⚠️ Interrupted by user")
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 