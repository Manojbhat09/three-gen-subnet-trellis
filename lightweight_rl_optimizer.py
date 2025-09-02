#!/usr/bin/env python3
"""
Lightweight 3-Round RL Optimizer for TRELLIS Orchestrator
=========================================================
A simplified RL system that runs exactly 3 rounds when initial score < 0.6
Integrates with the existing orchestrator validation system.
"""

import json
import time
import random
import re
import logging
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import statistics

@dataclass
class RLOptimizationAttempt:
    """Single optimization attempt within the 3-round RL loop"""
    attempt_number: int
    strategy_used: str
    optimized_prompt: str
    predicted_confidence: float
    validation_score: Optional[float]
    agent_reasoning: str
    timestamp: float

class LightweightRLOptimizer:
    """Lightweight 3-round RL optimizer for score improvement"""
    
    def __init__(self, logger: logging.Logger, trellis_server_url: str = "http://localhost:8096",
                 use_vllm: bool = True, vllm_url: str = "http://localhost:11300", 
                 vllm_model: str = "llama-3-2-3b-it", ollama_url: str = "http://localhost:11434"):
        self.logger = logger
        self.trellis_server_url = trellis_server_url
        
        # LLM configuration
        self.use_vllm = use_vllm
        self.vllm_url = vllm_url
        self.vllm_model = vllm_model
        self.ollama_url = ollama_url
        
        # RL parameters for 3-round system
        self.max_rounds = 3
        self.target_score = 0.7
        self.trigger_threshold = 0.7  # Only run if initial score < this
        
        # Strategy definitions (simplified from full RL system)
        self.strategies = [
            "conservative_enhancement",
            "aggressive_transformation", 
            "material_focus",
            "artistic_elaboration",
            "technical_precision",
            "contextual_scene_building",
            "minimalist_refinement"
        ]
        
        # Strategy performance tracking (simplified)
        self.strategy_performance = {strategy: 0.5 for strategy in self.strategies}
        
        llm_provider = "vLLM" if use_vllm else "Ollama"
        self.logger.info(f"🔄 Lightweight RL Optimizer initialized (3 rounds, target: {self.target_score}, LLM: {llm_provider})")
    
    def should_trigger_rl_optimization(self, initial_score: float) -> bool:
        """Check if RL optimization should be triggered based on initial score"""
        return initial_score < self.trigger_threshold
    
    def optimize_with_3_rounds(self, original_prompt: str, initial_score: float, 
                              validation_callback, endpoint: str = "generate/") -> Dict[str, Any]:
        """
        Run 3-round RL optimization to improve score above 0.6
        
        Args:
            original_prompt: The original prompt to optimize
            initial_score: The initial validation score
            validation_callback: Function to validate prompts (should return score)
            endpoint: TRELLIS endpoint to use for validation
            
        Returns:
            Dict with optimization results
        """
        self.logger.info(f"\n🔄 Starting 3-round RL optimization")
        self.logger.info(f"   Original prompt: '{original_prompt}'")
        self.logger.info(f"   Initial score: {initial_score:.4f}")
        self.logger.info(f"   Target score: {self.target_score}")
        
        attempts: List[RLOptimizationAttempt] = []
        best_prompt = original_prompt
        best_score = initial_score
        
        for round_num in range(1, self.max_rounds + 1):
            self.logger.info(f"\n   🔄 RL Round {round_num}/{self.max_rounds}")
            
            # Select strategy for this round
            strategy = self._select_strategy_for_round(round_num, attempts)
            
            # Make optimization attempt
            attempt = self._make_optimization_attempt(
                original_prompt, strategy, round_num, attempts
            )
            
            # Validate the attempt
            try:
                validation_result = validation_callback(
                    original_prompt, attempt.optimized_prompt, endpoint
                )
                
                # Handle both single score and enhanced validation results
                if isinstance(validation_result, dict):
                    attempt.validation_score = validation_result.get("validation_engine_score", 0.0)
                    attempt.clip_score = validation_result.get("clip_score", 0.0)
                    attempt.alignment_score = validation_result.get("alignment_score", 0.0)
                    self.logger.info(f"      📊 Validation score: {attempt.validation_score:.4f}")
                    self.logger.info(f"      🖼️ CLIP score: {attempt.clip_score:.4f}")
                    self.logger.info(f"      🤝 Alignment score: {attempt.alignment_score:.4f}")
                else:
                    # Backward compatibility with single score
                    # If we're using alignment score for RL, treat single score as alignment score
                    attempt.validation_score = validation_result
                    attempt.clip_score = None
                    attempt.alignment_score = validation_result  # Use the same score for alignment
                    self.logger.info(f"      📊 Validation score: {attempt.validation_score:.4f}")
                    self.logger.info(f"      🤝 Alignment score: {attempt.alignment_score:.4f}")
                    
            except Exception as e:
                self.logger.error(f"      ❌ Validation failed: {e}")
                attempt.validation_score = 0.0
                attempt.clip_score = 0.0
                attempt.alignment_score = 0.0
            
            attempts.append(attempt)
            
            # Determine which score to use for comparison (CLIP score if available, then alignment score, then validation score)
            if attempt.clip_score is not None and attempt.clip_score > 0:
                comparison_score = attempt.clip_score
            elif attempt.alignment_score is not None:
                comparison_score = attempt.alignment_score
            else:
                comparison_score = attempt.validation_score
            
            # Update best if improved
            if comparison_score and comparison_score > best_score:
                best_score = comparison_score
                best_prompt = attempt.optimized_prompt
                if attempt.clip_score is not None and attempt.clip_score > 0:
                    score_type = "CLIP"
                elif attempt.alignment_score is not None:
                    score_type = "alignment"
                else:
                    score_type = "validation"
                self.logger.info(f"      🎯 New best {score_type} score: {best_score:.4f}")
            
            # Update strategy performance
            self._update_strategy_performance(strategy, comparison_score or 0.0)
            
            # Early exit if target achieved
            if comparison_score and comparison_score >= self.target_score:
                if attempt.clip_score is not None and attempt.clip_score > 0:
                    score_type = "CLIP"
                elif attempt.alignment_score is not None:
                    score_type = "alignment"
                else:
                    score_type = "validation"
                self.logger.info(f"      ✅ Target {score_type} score achieved in round {round_num}!")
                break
        
        # Calculate results
        improvement = best_score - initial_score
        success = best_score >= self.target_score
        
        result = {
            'success': success,
            'original_prompt': original_prompt,
            'final_optimized_prompt': best_prompt,
            'initial_score': initial_score,
            'final_score': best_score,
            'improvement': improvement,
            'rounds_used': len(attempts),
            'attempts': [
                {
                    'round': a.attempt_number,
                    'strategy': a.strategy_used,
                    'prompt': a.optimized_prompt,
                    'score': a.validation_score,
                    'clip_score': a.clip_score,
                    'alignment_score': a.alignment_score,
                    'confidence': a.predicted_confidence
                }
                for a in attempts
            ],
            'strategy_performance': self.strategy_performance.copy()
        }
        
        self.logger.info(f"\n🎯 3-Round RL Optimization Complete:")
        self.logger.info(f"   Success: {success}")
        self.logger.info(f"   Improvement: {improvement:+.4f}")
        self.logger.info(f"   Final score: {best_score:.4f}")
        self.logger.info(f"   Rounds used: {len(attempts)}")
        
        return result
    
    def _select_strategy_for_round(self, round_num: int, previous_attempts: List[RLOptimizationAttempt]) -> str:
        """Select strategy for current round based on round number and previous performance"""
        
        # Round 1: Use best performing strategy
        if round_num == 1:
            best_strategy = max(self.strategy_performance.items(), key=lambda x: x[1])[0]
            return best_strategy
        
        # Round 2: Use second best or explore
        elif round_num == 2:
            sorted_strategies = sorted(self.strategy_performance.items(), key=lambda x: x[1], reverse=True)
            if len(sorted_strategies) > 1:
                return sorted_strategies[1][0]  # Second best
            else:
                return random.choice(self.strategies)
        
        # Round 3: Use strategy that hasn't been tried yet, or best available
        else:
            used_strategies = {a.strategy_used for a in previous_attempts}
            unused_strategies = [s for s in self.strategies if s not in used_strategies]
            
            if unused_strategies:
                return random.choice(unused_strategies)
            else:
                # All strategies used, pick best performing
                return max(self.strategy_performance.items(), key=lambda x: x[1])[0]
    
    def _make_optimization_attempt(self, original_prompt: str, strategy: str, 
                                 round_num: int, previous_attempts: List[RLOptimizationAttempt]) -> RLOptimizationAttempt:
        """Make a single optimization attempt using the specified strategy"""
        
        self.logger.info(f"      🎯 Strategy: {strategy}")
        
        # Build context from previous attempts
        previous_context = self._build_previous_attempts_context(previous_attempts)
        
        # Build strategy-specific prompt
        system_prompt = f"""You are optimizing a prompt for 3D model generation. This is round {round_num} of 3 rounds.

ORIGINAL PROMPT: "{original_prompt}"
STRATEGY: {strategy}

{previous_context}

STRATEGY GUIDANCE:
- conservative_enhancement: Make subtle improvements, add quality descriptors
- aggressive_transformation: Completely rephrase with enhanced details
- material_focus: Emphasize materials, textures, and physical properties
- artistic_elaboration: Add artistic style, lighting, and composition details
- technical_precision: Focus on technical accuracy and specifications
- contextual_scene_building: Add environmental context and scene details
- minimalist_refinement: Simplify and focus on core elements

TASK: Create an optimized prompt that improves on the original.

Rules:
- Keep the core subject intact
- Add quality descriptors (high quality, detailed, etc.)
- Ensure the prompt is clear and specific
- Add "front view, accurate, complete, white background" at the end

RESPONSE FORMAT:
REASONING: [Your reasoning for this optimization approach]

OPTIMIZATION: {{
  "optimized_prompt": "[full optimized prompt]",
  "confidence": [0.0-1.0],
  "key_changes": ["change1", "change2"],
  "expected_score": [0.0-1.0]
}}"""

        try:
            # Use LLM to generate optimization
            if self.use_vllm:
                response = self._query_vllm(system_prompt)
            else:
                response = self._query_ollama(system_prompt)
            
            # Parse the response
            structured_output = self._parse_optimization_response(response, original_prompt)
            
            return RLOptimizationAttempt(
                attempt_number=round_num,
                strategy_used=strategy,
                optimized_prompt=structured_output.get('optimized_prompt', f"{original_prompt}, front view, accurate, complete, white background"),
                predicted_confidence=structured_output.get('confidence', 0.7),
                validation_score=None,  # Will be filled by validation
                agent_reasoning=response,
                timestamp=time.time()
            )
            
        except Exception as e:
            self.logger.error(f"      ❌ LLM optimization failed: {e}")
            # Fallback to rule-based optimization
            optimized_prompt = self._apply_strategy_optimization(original_prompt, strategy)
            
            return RLOptimizationAttempt(
                attempt_number=round_num,
                strategy_used=strategy,
                optimized_prompt=optimized_prompt,
                predicted_confidence=0.5,  # Lower confidence for fallback
                validation_score=None,  # Will be filled by validation
                agent_reasoning=f"Fallback optimization due to LLM error: {e}",
                timestamp=time.time()
            )
    
    def _apply_strategy_optimization(self, original_prompt: str, strategy: str) -> str:
        """Apply strategy-specific optimization rules"""
        
        # Ensure proper suffix
        if not original_prompt.endswith('front view, accurate, complete, white background'):
            base_prompt = original_prompt.rstrip(', ')
        else:
            base_prompt = original_prompt.replace(', front view, accurate, complete, white background', '').rstrip(', ')
        
        # Apply strategy-specific transformations
        if strategy == "conservative_enhancement":
            optimized = f"{base_prompt}, high quality, detailed, front view, accurate, complete, white background"
        
        elif strategy == "aggressive_transformation":
            optimized = f"a highly detailed and meticulously crafted {base_prompt}, professional quality, intricate details, front view, accurate, complete, white background"
        
        elif strategy == "material_focus":
            optimized = f"{base_prompt}, with realistic materials and textures, front view, accurate, complete, white background"
        
        elif strategy == "artistic_elaboration":
            optimized = f"{base_prompt}, artistically rendered with perfect lighting and composition, front view, accurate, complete, white background"
        
        elif strategy == "technical_precision":
            optimized = f"{base_prompt}, technically accurate with precise specifications, front view, accurate, complete, white background"
        
        elif strategy == "contextual_scene_building":
            optimized = f"{base_prompt}, in a clean studio environment, front view, accurate, complete, white background"
        
        elif strategy == "minimalist_refinement":
            optimized = f"{base_prompt}, clean and simple, front view, accurate, complete, white background"
        
        else:
            # Default fallback
            optimized = f"{base_prompt}, high quality, front view, accurate, complete, white background"
        
        return optimized
    
    def _build_previous_attempts_context(self, previous_attempts: List[RLOptimizationAttempt]) -> str:
        """Build context from previous optimization attempts"""
        if not previous_attempts:
            return "PREVIOUS ATTEMPTS: None - this is your first attempt."
        
        context = "PREVIOUS ATTEMPTS IN THIS SESSION:\n"
        for attempt in previous_attempts:
            score_text = f"{attempt.validation_score:.4f}" if attempt.validation_score else "pending"
            context += f"Round {attempt.attempt_number}: {attempt.strategy_used}\n"
            context += f"  Score: {score_text} | Confidence: {attempt.predicted_confidence:.2f}\n"
            context += f"  Prompt: {attempt.optimized_prompt[:100]}...\n"
        
        # Add trend analysis
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
    
    def _update_strategy_performance(self, strategy: str, score: float):
        """Update strategy performance based on new score"""
        # Simple exponential moving average
        alpha = 0.3
        current_perf = self.strategy_performance[strategy]
        self.strategy_performance[strategy] = (1 - alpha) * current_perf + alpha * score
        
        self.logger.debug(f"      📈 Updated {strategy} performance: {self.strategy_performance[strategy]:.3f}")
    
    def get_strategy_insights(self) -> Dict[str, Any]:
        """Get insights about strategy performance"""
        sorted_strategies = sorted(
            self.strategy_performance.items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        return {
            "best_strategy": sorted_strategies[0][0] if sorted_strategies else None,
            "strategy_rankings": sorted_strategies,
            "average_performance": statistics.mean(self.strategy_performance.values()) if self.strategy_performance else 0.0
        }
    
    def _query_vllm(self, prompt: str) -> str:
        """Query vLLM for optimization"""
        import requests
        
        headers = {
            "Content-Type": "application/json",
        }
        
        data = {
            "model": self.vllm_model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.7,
            "top_p": 0.9,
            "max_tokens": 350,
            "stream": False
        }
        
        response = requests.post(f"{self.vllm_url}/v1/chat/completions", headers=headers, json=data, timeout=60)
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"].strip()
    
    def _query_ollama(self, prompt: str) -> str:
        """Query Ollama for optimization"""
        import requests
        
        data = {
            "model": "llama3.2:3b",
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
    
    def _parse_optimization_response(self, response: str, original_prompt: str) -> Dict[str, Any]:
        """Parse optimization response with robust fallbacks"""
        try:
            # Try to extract JSON from the response
            import re
            json_match = re.search(r'OPTIMIZATION:\s*(\{.*?\})', response, re.DOTALL | re.IGNORECASE)
            if json_match:
                import json
                return json.loads(json_match.group(1))
        except:
            pass
        
        # Fallback parsing
        optimized_prompt = original_prompt
        confidence = 0.5
        
        # Extract optimized prompt
        opt_patterns = [
            r'optimized_prompt[":\s]*"([^"]+)"',
            r'"([^"]*front view, accurate, complete, white background[^"]*)"',
        ]
        
        for pattern in opt_patterns:
            match = re.search(pattern, response, re.IGNORECASE)
            if match:
                optimized_prompt = match.group(1).strip()
                break
        
        # Ensure proper format - only add suffix, no prefix
        if not optimized_prompt.endswith('front view, accurate, complete, white background'):
            optimized_prompt = optimized_prompt.rstrip(', ') + ", front view, accurate, complete, white background"
        
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
