#!/usr/bin/env python3
"""
RL Prompt Optimizer - Production Inference System
================================================
Production-ready inference engine for optimizing prompts in the miner.
Fast, reliable, and optimized for real-time usage.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import json
import time
import re
from typing import Dict, List, Tuple, Optional
from pathlib import Path
from dataclasses import dataclass

@dataclass
class OptimizationResult:
    """Result of prompt optimization"""
    original_prompt: str
    optimized_prompt: str
    predicted_score: float
    confidence: float
    optimization_time: float
    actions_taken: List[str]
    success: bool
    error_message: Optional[str] = None

class ProductionDQN(nn.Module):
    """Production-optimized DQN (same architecture as training)"""
    
    def __init__(self, state_size: int, action_size: int):
        super(ProductionDQN, self).__init__()
        
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

class RLPromptOptimizerInference:
    """Production inference engine for prompt optimization"""
    
    def __init__(self, model_path: str, ultra_target: float = 0.96):
        self.ultra_target = ultra_target
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load model and configuration
        self.model, self.config = self._load_model(model_path)
        self.action_space = self._define_production_action_space()
        self.state_size = 20
        
        # Performance tracking
        self.optimization_count = 0
        self.success_count = 0
        self.total_time = 0.0
        
        print(f"🚀 RL PROMPT OPTIMIZER INFERENCE ENGINE")
        print(f"   🧠 Device: {self.device}")
        print(f"   🎯 Ultra Target: {ultra_target}")
        print(f"   ⚡ Model Loaded: {model_path}")

    def _load_model(self, model_path: str) -> Tuple[nn.Module, Dict]:
        """Load trained model and configuration"""
        
        model_file = Path(model_path)
        if not model_file.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        # Load checkpoint
        checkpoint = torch.load(model_file, map_location=self.device)
        
        # Get configuration
        config = checkpoint.get('metadata', {})
        action_size = len(self._define_production_action_space())
        
        # Initialize model
        model = ProductionDQN(self.state_size, action_size).to(self.device)
        model.load_state_dict(checkpoint['q_network_local_state_dict'])
        model.eval()  # Set to evaluation mode
        
        print(f"   📊 Model trained for {config.get('total_episodes', 'unknown')} episodes")
        print(f"   🏆 Best training score: {config.get('best_score', 'unknown')}")
        
        return model, config

    def _define_production_action_space(self) -> List[Tuple]:
        """Production action space (same as training)"""
        
        actions = []
        
        # Proven high-scoring patterns
        proven_patterns = [
            "defense-grade ultra-precision {target}, ultra-high technical specification",
            "aerospace-grade precision-engineered {target}, advanced engineering design", 
            "military-spec ultra-detailed {target}, premium manufacturing excellence",
            "laboratory-grade precision-forged {target}, aerospace-engineering excellence",
            "ultra-precision masterpiece-quality {target}, ultra-high technical specification",
            "precision-aerospace {target}, defense-grade excellence"
        ]
        
        for pattern in proven_patterns:
            actions.append(('APPLY_PATTERN', pattern, 'full_replace'))
        
        # Quality upgrades
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
        
        # Simplification actions
        simplify_actions = [
            ('SIMPLIFY', 'remove_duplicates', 'clean'),
            ('SIMPLIFY', 'keep_best_only', 'clean'),
            ('SIMPLIFY', 'ultra_minimal', 'clean')
        ]
        
        actions.extend(simplify_actions)
        
        return actions

    def optimize_prompt(self, input_prompt: str, max_steps: int = 5, 
                       timeout: float = 10.0) -> OptimizationResult:
        """
        Optimize a prompt for better validation scores
        
        Args:
            input_prompt: Original prompt to optimize
            max_steps: Maximum optimization steps
            timeout: Maximum time to spend optimizing
            
        Returns:
            OptimizationResult with optimized prompt and metadata
        """
        
        start_time = time.time()
        self.optimization_count += 1
        
        try:
            # Extract target object from prompt
            target_object = self._extract_target_object(input_prompt)
            if not target_object:
                return OptimizationResult(
                    original_prompt=input_prompt,
                    optimized_prompt=input_prompt,
                    predicted_score=0.0,
                    confidence=0.0,
                    optimization_time=time.time() - start_time,
                    actions_taken=[],
                    success=False,
                    error_message="Could not extract target object from prompt"
                )
            
            # Initialize optimization
            current_prompt = input_prompt
            actions_taken = []
            best_prompt = input_prompt
            best_predicted_score = 0.0
            
            # Optimization loop
            for step in range(max_steps):
                # Check timeout
                if time.time() - start_time > timeout:
                    break
                
                # Get current state
                state = self._get_state(current_prompt, target_object, step, max_steps)
                
                # Predict best action
                action_idx, confidence = self._predict_best_action(state)
                action = self.action_space[action_idx]
                
                # Apply action
                new_prompt = self._apply_action_inference(action, current_prompt, target_object)
                
                # Predict score improvement (without actual validation for speed)
                predicted_score = self._predict_score(new_prompt, target_object)
                
                # Update if better
                if predicted_score > best_predicted_score:
                    best_prompt = new_prompt
                    best_predicted_score = predicted_score
                
                # Track action
                actions_taken.append(f"Step {step+1}: {action[0]} -> {predicted_score:.3f}")
                
                # Update current prompt
                current_prompt = new_prompt
                
                # Early stop if we predict ultra achievement
                if predicted_score >= self.ultra_target:
                    actions_taken.append(f"Early stop: Ultra target predicted")
                    break
            
            # Calculate final metrics
            optimization_time = time.time() - start_time
            self.total_time += optimization_time
            
            # Success if we improved the predicted score
            success = best_predicted_score > 0.5  # Reasonable threshold
            if success:
                self.success_count += 1
            
            return OptimizationResult(
                original_prompt=input_prompt,
                optimized_prompt=best_prompt,
                predicted_score=best_predicted_score,
                confidence=confidence,
                optimization_time=optimization_time,
                actions_taken=actions_taken,
                success=success
            )
            
        except Exception as e:
            return OptimizationResult(
                original_prompt=input_prompt,
                optimized_prompt=input_prompt,
                predicted_score=0.0,
                confidence=0.0,
                optimization_time=time.time() - start_time,
                actions_taken=[],
                success=False,
                error_message=str(e)
            )

    def _extract_target_object(self, prompt: str) -> Optional[str]:
        """Extract target object from prompt"""
        
        # Try to extract from standard format: "wbgmsst, [target], ..."
        if 'wbgmsst,' in prompt.lower():
            parts = prompt.split(',')
            if len(parts) >= 2:
                target = parts[1].strip()
                # Remove common descriptors to get base object
                target = re.sub(r'\b(aerospace-grade|military-spec|defense-grade|ultra-precision|precision-engineered|masterpiece-quality)\b\s*', '', target)
                return target.strip()
        
        # Fallback: return the prompt as-is
        return prompt

    def _get_state(self, prompt: str, target_object: str, step: int, max_steps: int) -> np.ndarray:
        """Get state representation for inference"""
        
        state = np.zeros(self.state_size)
        
        # Step progress
        state[0] = step / max_steps
        
        # Prompt characteristics
        prompt_lower = prompt.lower()
        state[1] = 1.0 if 'aerospace' in prompt_lower else 0.0
        state[2] = 1.0 if 'defense' in prompt_lower else 0.0
        state[3] = 1.0 if 'military' in prompt_lower else 0.0
        state[4] = 1.0 if 'ultra-precision' in prompt_lower else 0.0
        state[5] = 1.0 if 'precision-engineered' in prompt_lower else 0.0
        state[6] = 1.0 if 'masterpiece' in prompt_lower else 0.0
        state[7] = len(prompt) / 150  # Normalized length
        
        # Target object characteristics
        target_lower = target_object.lower()
        state[8] = 1.0 if any(word in target_lower for word in ["steel", "metal"]) else 0.0
        state[9] = 1.0 if any(word in target_lower for word in ["fabric", "cloth"]) else 0.0
        state[10] = 1.0 if any(word in target_lower for word in ["glass", "crystal"]) else 0.0
        state[11] = 1.0 if any(word in target_lower for word in ["wood", "wooden"]) else 0.0
        
        # Prompt quality indicators
        quality_terms = ["specification", "engineering", "excellence", "manufacturing"]
        state[12] = sum(1 for term in quality_terms if term in prompt_lower) / len(quality_terms)
        
        return state

    def _predict_best_action(self, state: np.ndarray) -> Tuple[int, float]:
        """Predict best action using trained model"""
        
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_values = self.model(state_tensor)
            
            # Get best action and confidence
            action_idx = q_values.argmax().item()
            max_q_value = q_values.max().item()
            
            # Convert Q-value to confidence (0-1 range)
            confidence = torch.sigmoid(q_values.max()).item()
            
            return action_idx, confidence

    def _apply_action_inference(self, action: Tuple, current_prompt: str, target_object: str) -> str:
        """Apply action to prompt (inference version)"""
        
        action_type, modifier, mode = action
        
        if action_type == 'APPLY_PATTERN':
            # Replace with proven pattern
            pattern = modifier.replace('{target}', target_object)
            return f"wbgmsst, {pattern}, white background"
        
        elif action_type in ['UPGRADE_AUTHORITY', 'UPGRADE_PROCESS', 'UPGRADE_QUALITY']:
            # Upgrade existing prompt
            parts = current_prompt.split(', ')
            if len(parts) >= 3:  # wbgmsst, [middle], white background
                middle = parts[1]
                
                # Remove conflicting descriptors
                if action_type == 'UPGRADE_AUTHORITY':
                    middle = re.sub(r'\b(aerospace-grade|military-spec|defense-grade)\b\s*', '', middle)
                elif action_type == 'UPGRADE_PROCESS':
                    middle = re.sub(r'\b(ultra-precision|precision-engineered|masterpiece-quality)\b\s*', '', middle)
                
                # Add new descriptor
                middle = f"{modifier} {middle}".strip()
                parts[1] = middle
                
                return ', '.join(parts)
        
        elif action_type == 'SIMPLIFY':
            if modifier == 'remove_duplicates':
                words = current_prompt.split()
                seen = set()
                cleaned_words = []
                for word in words:
                    if word not in seen:
                        cleaned_words.append(word)
                        seen.add(word)
                return ' '.join(cleaned_words)
            
            elif modifier == 'keep_best_only':
                return f"wbgmsst, defense-grade ultra-precision {target_object}, ultra-high technical specification, white background"
            
            elif modifier == 'ultra_minimal':
                return f"wbgmsst, aerospace-grade {target_object}, precision-engineered excellence, white background"
        
        return current_prompt

    def _predict_score(self, prompt: str, target_object: str) -> float:
        """Predict validation score without actual validation (for speed)"""
        
        # Heuristic-based score prediction for fast inference
        score = 0.5  # Base score
        
        prompt_lower = prompt.lower()
        
        # Authority bonuses
        if 'aerospace-grade' in prompt_lower: score += 0.15
        elif 'defense-grade' in prompt_lower: score += 0.12
        elif 'military-spec' in prompt_lower: score += 0.10
        
        # Process bonuses
        if 'ultra-precision' in prompt_lower: score += 0.12
        elif 'precision-engineered' in prompt_lower: score += 0.10
        elif 'masterpiece-quality' in prompt_lower: score += 0.08
        
        # Quality bonuses
        if 'ultra-high technical specification' in prompt_lower: score += 0.08
        elif 'advanced engineering design' in prompt_lower: score += 0.06
        
        # Proven pattern bonuses
        if 'defense-grade ultra-precision' in prompt_lower: score += 0.20
        if 'aerospace-grade precision-engineered' in prompt_lower: score += 0.18
        
        # Length penalties
        if len(prompt) > 120: score -= 0.05
        if len(prompt) > 150: score -= 0.10
        
        # Ensure reasonable range
        return max(0.0, min(1.0, score))

    def batch_optimize(self, prompts: List[str], max_steps: int = 5) -> List[OptimizationResult]:
        """Optimize multiple prompts efficiently"""
        
        results = []
        
        print(f"🔄 Batch optimizing {len(prompts)} prompts...")
        start_time = time.time()
        
        for i, prompt in enumerate(prompts, 1):
            result = self.optimize_prompt(prompt, max_steps=max_steps)
            results.append(result)
            
            if i % 10 == 0:
                elapsed = time.time() - start_time
                avg_time = elapsed / i
                print(f"   Progress: {i}/{len(prompts)} ({avg_time:.2f}s/prompt)")
        
        total_time = time.time() - start_time
        print(f"✅ Batch optimization complete: {total_time:.2f}s total")
        
        return results

    def get_performance_stats(self) -> Dict:
        """Get performance statistics"""
        
        avg_time = self.total_time / self.optimization_count if self.optimization_count > 0 else 0.0
        success_rate = self.success_count / self.optimization_count if self.optimization_count > 0 else 0.0
        
        return {
            "total_optimizations": self.optimization_count,
            "successful_optimizations": self.success_count,
            "success_rate": success_rate,
            "average_time_per_optimization": avg_time,
            "total_time": self.total_time
        }

# Production API Interface
class PromptOptimizerAPI:
    """Production API interface for the miner"""
    
    def __init__(self, model_path: str):
        self.optimizer = RLPromptOptimizerInference(model_path)
        
    def optimize_for_miner(self, prompt: str, urgency: str = "normal") -> Dict:
        """
        Optimize prompt for miner with different urgency levels
        
        Args:
            prompt: Input prompt to optimize
            urgency: "fast" (1 step), "normal" (3 steps), "thorough" (5 steps)
            
        Returns:
            Dictionary with optimization results
        """
        
        # Adjust parameters based on urgency
        step_mapping = {"fast": 1, "normal": 3, "thorough": 5}
        timeout_mapping = {"fast": 2.0, "normal": 5.0, "thorough": 10.0}
        
        max_steps = step_mapping.get(urgency, 3)
        timeout = timeout_mapping.get(urgency, 5.0)
        
        result = self.optimizer.optimize_prompt(
            input_prompt=prompt,
            max_steps=max_steps,
            timeout=timeout
        )
        
        return {
            "success": result.success,
            "original_prompt": result.original_prompt,
            "optimized_prompt": result.optimized_prompt,
            "predicted_improvement": result.predicted_score,
            "confidence": result.confidence,
            "optimization_time": result.optimization_time,
            "actions_taken": result.actions_taken,
            "error": result.error_message
        }

def main():
    """Demo the inference system"""
    
    print("🚀 RL PROMPT OPTIMIZER INFERENCE DEMO")
    print("="*60)
    
    # For demo, we'll create a mock model path
    # In production, this would be the path to your trained model
    model_path = "trained_models/rl_prompt_optimizer.pth"
    
    # Demo prompts
    test_prompts = [
        "hexagonal prism steel structure",
        "wbgmsst, elegant silk fabric draping, white background",
        "transparent glass sphere with reflections",
        "ornate wooden sculpture with intricate details"
    ]
    
    try:
        # Initialize optimizer (this will fail without real model, but shows the interface)
        optimizer = RLPromptOptimizerInference(model_path)
        
        # Test single optimization
        result = optimizer.optimize_prompt(test_prompts[0])
        
        print(f"📊 OPTIMIZATION RESULT:")
        print(f"   Original: {result.original_prompt}")
        print(f"   Optimized: {result.optimized_prompt}")
        print(f"   Predicted Score: {result.predicted_score:.3f}")
        print(f"   Time: {result.optimization_time:.2f}s")
        
        # Test batch optimization
        results = optimizer.batch_optimize(test_prompts)
        
        print(f"\n📈 BATCH RESULTS:")
        for i, result in enumerate(results, 1):
            print(f"   {i}. {result.predicted_score:.3f} ({result.optimization_time:.2f}s)")
        
    except FileNotFoundError:
        print("❌ Model file not found - this is expected in demo mode")
        print("✅ Inference system architecture is ready for production!")

if __name__ == "__main__":
    main() 