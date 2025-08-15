#!/usr/bin/env python3
"""
Live Prompt Optimizer - Fast CLIP-based optimization for 0.0 fidelity recovery

Instead of slow RL (5-15 rounds × 30-40s = 2.5-10 minutes), this uses:
1. Fast CLIP scoring (~0.1s per prompt)
2. Multiple prompt candidates generated in parallel
3. Best candidate selected via CLIP alignment
4. Total time: ~2-3 seconds vs 2.5-10 minutes

Usage:
  optimizer = LivePromptOptimizer()
  
  # When you get 0.0 fidelity score:
  optimized_prompt = optimizer.optimize_for_zero_recovery(
      original_prompt="steel spade",
      lora_used="baolei", 
      strategies=["material_focus", "scene_removal", "technical_detail"]
  )
"""

import time
import logging
from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np
import torch
import clip
from PIL import Image
import requests
import json
import threading
from pathlib import Path

@dataclass
class OptimizationCandidate:
    """Single optimization candidate"""
    original_prompt: str
    optimized_prompt: str
    strategy_used: str
    confidence_score: float
    clip_score: Optional[float] = None
    generation_time: Optional[float] = None

class FastLLMOptimizer:
    """Fast LLM-based prompt optimization strategies"""
    
    def __init__(self):
        self.strategies = {
            "material_focus": self._material_focus_strategy,
            "scene_removal": self._scene_removal_strategy, 
            "technical_detail": self._technical_detail_strategy,
            "simplification": self._simplification_strategy,
            "object_clarity": self._object_clarity_strategy,
            "lighting_fix": self._lighting_fix_strategy,
            "viewpoint_fix": self._viewpoint_fix_strategy,
            "size_specification": self._size_specification_strategy
        }
    
    def generate_candidates(self, prompt: str, strategies: List[str], lora_context: str = None) -> List[OptimizationCandidate]:
        """Generate multiple optimization candidates quickly"""
        candidates = []
        
        for strategy in strategies:
            if strategy in self.strategies:
                try:
                    optimized = self.strategies[strategy](prompt, lora_context)
                    candidates.append(OptimizationCandidate(
                        original_prompt=prompt,
                        optimized_prompt=optimized,
                        strategy_used=strategy,
                        confidence_score=0.8  # Default confidence
                    ))
                except Exception as e:
                    logging.warning(f"Strategy {strategy} failed: {e}")
                    
        return candidates
    
    def _material_focus_strategy(self, prompt: str, lora_context: str = None) -> str:
        """Focus on material and texture details"""
        # Fast rule-based transformations
        words = prompt.lower().split()
        
        # Add material descriptors based on object type
        if any(word in prompt.lower() for word in ['spade', 'shovel', 'tool']):
            return f"polished steel {prompt} with wooden handle, metallic finish"
        elif any(word in prompt.lower() for word in ['gem', 'crystal', 'stone']):
            return f"faceted crystal {prompt} with smooth surface, transparent"
        elif any(word in prompt.lower() for word in ['robot', 'mechanical']):
            return f"metallic {prompt} with brushed aluminum surface"
        else:
            return f"detailed {prompt} with refined surface texture"
    
    def _scene_removal_strategy(self, prompt: str, lora_context: str = None) -> str:
        """Remove scene context, focus on object"""
        # Remove scene words
        scene_words = ['background', 'environment', 'scene', 'setting', 'landscape', 
                      'room', 'table', 'floor', 'wall', 'outdoor', 'indoor']
        
        words = prompt.split()
        filtered_words = [w for w in words if not any(sw in w.lower() for sw in scene_words)]
        
        base_prompt = ' '.join(filtered_words)
        return f"{base_prompt}, isolated object, white background"
    
    def _technical_detail_strategy(self, prompt: str, lora_context: str = None) -> str:
        """Add technical precision details"""
        return f"precise {prompt}, accurate proportions, detailed construction, high fidelity"
    
    def _simplification_strategy(self, prompt: str, lora_context: str = None) -> str:
        """Simplify complex prompts"""
        # Keep only key nouns and adjectives
        words = prompt.split()
        if len(words) > 6:
            # Take first 4-5 most important words
            key_words = words[:5]
            return ' '.join(key_words)
        return prompt
    
    def _object_clarity_strategy(self, prompt: str, lora_context: str = None) -> str:
        """Improve object clarity and definition"""
        return f"clearly defined {prompt}, sharp edges, well-formed structure"
    
    def _lighting_fix_strategy(self, prompt: str, lora_context: str = None) -> str:
        """Add proper lighting specification"""
        return f"{prompt}, even lighting, no harsh shadows, soft illumination"
    
    def _viewpoint_fix_strategy(self, prompt: str, lora_context: str = None) -> str:
        """Specify clear viewpoint"""
        return f"{prompt}, front view, centered, clear perspective"
    
    def _size_specification_strategy(self, prompt: str, lora_context: str = None) -> str:
        """Add size/scale specification"""
        if 'small' in prompt.lower():
            return f"compact {prompt}, miniature scale"
        elif 'large' in prompt.lower():
            return f"substantial {prompt}, full size"
        else:
            return f"standard {prompt}, appropriate scale"

class CLIPScorer:
    """Fast CLIP-based prompt scoring"""
    
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = None
        self.preprocess = None
        
    def load_model(self):
        """Load CLIP model for scoring"""
        if self.model is None:
            self.model, self.preprocess = clip.load("ViT-B/32", device=self.device)
            logging.info(f"CLIP model loaded on {self.device}")
    
    def score_text_alignment(self, text1: str, text2: str) -> float:
        """Score alignment between two text prompts"""
        if self.model is None:
            self.load_model()
            
        with torch.no_grad():
            text1_tokens = clip.tokenize([text1]).to(self.device)
            text2_tokens = clip.tokenize([text2]).to(self.device)
            
            text1_features = self.model.encode_text(text1_tokens)
            text2_features = self.model.encode_text(text2_tokens)
            
            # Normalize features
            text1_features = text1_features / text1_features.norm(dim=-1, keepdim=True)
            text2_features = text2_features / text2_features.norm(dim=-1, keepdim=True)
            
            # Cosine similarity
            similarity = torch.matmul(text1_features, text2_features.T)
            return similarity.item()
    
    def score_prompt_quality(self, prompt: str) -> float:
        """Score prompt quality based on CLIP text encoder confidence"""
        if self.model is None:
            self.load_model()
            
        # Generate multiple quality reference prompts
        quality_refs = [
            "high quality detailed object",
            "precise accurate 3D model", 
            "well-defined clear structure",
            "professional render"
        ]
        
        scores = []
        for ref in quality_refs:
            score = self.score_text_alignment(prompt, ref)
            scores.append(score)
        
        return np.mean(scores)

class LivePromptOptimizer:
    """Main live optimization system"""
    
    def __init__(self, server_url: str = "http://localhost:8096"):
        self.server_url = server_url
        self.llm_optimizer = FastLLMOptimizer()
        self.clip_scorer = CLIPScorer()
        self.logger = logging.getLogger(__name__)
        
        # Strategy performance tracking for adaptive selection
        self.strategy_performance: Dict[str, List[float]] = {}
        
        # Lock for thread safety
        self.lock = threading.Lock()
        
        logging.basicConfig(level=logging.INFO)
        self.logger.info("🚀 Live Prompt Optimizer initialized")
    
    def optimize_for_zero_recovery(self, 
                                 original_prompt: str, 
                                 lora_used: str = None,
                                 strategies: List[str] = None,
                                 max_candidates: int = 5) -> str:
        """
        Fast optimization for 0.0 fidelity score recovery
        
        Args:
            original_prompt: The prompt that got 0.0 score
            lora_used: LoRA that was used (for context)
            strategies: List of strategies to try
            max_candidates: Maximum candidates to generate
            
        Returns:
            Best optimized prompt
        """
        start_time = time.time()
        
        if strategies is None:
            strategies = self._select_best_strategies_for_lora(lora_used)
        
        self.logger.info(f"🔧 Optimizing '{original_prompt[:50]}...' for {lora_used}")
        self.logger.info(f"   Strategies: {strategies}")
        
        # Generate candidates
        candidates = self.llm_optimizer.generate_candidates(
            original_prompt, 
            strategies[:max_candidates], 
            lora_context=lora_used
        )
        
        if not candidates:
            # Fallback
            return f"{original_prompt}, high quality, detailed"
        
        # Score candidates with CLIP
        self.clip_scorer.load_model()
        
        for candidate in candidates:
            # Score against original for semantic preservation
            semantic_score = self.clip_scorer.score_text_alignment(
                candidate.optimized_prompt, 
                original_prompt
            )
            
            # Score for general quality
            quality_score = self.clip_scorer.score_prompt_quality(
                candidate.optimized_prompt
            )
            
            # Combined score (favor quality but preserve semantics)
            candidate.clip_score = 0.7 * quality_score + 0.3 * semantic_score
        
        # Select best candidate
        best_candidate = max(candidates, key=lambda c: c.clip_score or 0.0)
        
        # Update strategy performance tracking
        with self.lock:
            strategy = best_candidate.strategy_used
            if strategy not in self.strategy_performance:
                self.strategy_performance[strategy] = []
            self.strategy_performance[strategy].append(best_candidate.clip_score)
        
        optimization_time = time.time() - start_time
        
        self.logger.info(f"✅ Optimization complete in {optimization_time:.2f}s")
        self.logger.info(f"   Best strategy: {best_candidate.strategy_used}")
        self.logger.info(f"   CLIP score: {best_candidate.clip_score:.3f}")
        self.logger.info(f"   Result: '{best_candidate.optimized_prompt[:80]}...'")
        
        return best_candidate.optimized_prompt
    
    def _select_best_strategies_for_lora(self, lora: str) -> List[str]:
        """Select best strategies based on LoRA characteristics and past performance"""
        
        # LoRA-specific strategy preferences (from your analysis)
        lora_strategies = {
            'baolei': ['scene_removal', 'simplification', 'object_clarity'],  # Avoid complex scenes
            'cartoon_3d': ['material_focus', 'lighting_fix', 'technical_detail'],
            'isometric_3d': ['viewpoint_fix', 'technical_detail', 'size_specification'],
            'tf2_style': ['object_clarity', 'material_focus', 'lighting_fix'],
            'sd15_game_icon': ['simplification', 'object_clarity', 'scene_removal']
        }
        
        base_strategies = lora_strategies.get(lora, ['scene_removal', 'object_clarity', 'material_focus'])
        
        # Add best performing strategies based on history
        with self.lock:
            strategy_scores = {}
            for strategy, scores in self.strategy_performance.items():
                if scores:
                    strategy_scores[strategy] = np.mean(scores[-10:])  # Recent performance
            
            # Add top performing strategies not already included
            top_strategies = sorted(strategy_scores.items(), key=lambda x: x[1], reverse=True)
            for strategy, score in top_strategies[:2]:
                if strategy not in base_strategies:
                    base_strategies.append(strategy)
        
        return base_strategies[:5]  # Max 5 strategies
    
    def optimize_batch(self, prompts: List[str], lora_used: str = None) -> List[str]:
        """Optimize multiple prompts in batch for efficiency"""
        start_time = time.time()
        
        results = []
        with ThreadPoolExecutor(max_workers=3) as executor:
            future_to_prompt = {
                executor.submit(self.optimize_for_zero_recovery, prompt, lora_used): prompt 
                for prompt in prompts
            }
            
            for future in as_completed(future_to_prompt):
                prompt = future_to_prompt[future]
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    self.logger.error(f"Error optimizing '{prompt}': {e}")
                    results.append(f"{prompt}, high quality")
        
        batch_time = time.time() - start_time
        self.logger.info(f"📊 Batch optimization: {len(prompts)} prompts in {batch_time:.2f}s")
        
        return results
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get optimization performance statistics"""
        with self.lock:
            stats = {}
            for strategy, scores in self.strategy_performance.items():
                if scores:
                    stats[strategy] = {
                        'avg_score': np.mean(scores),
                        'recent_avg': np.mean(scores[-5:]) if len(scores) >= 5 else np.mean(scores),
                        'count': len(scores),
                        'trend': 'improving' if len(scores) >= 3 and scores[-1] > scores[-3] else 'stable'
                    }
            
            return {
                'strategy_performance': stats,
                'total_optimizations': sum(len(scores) for scores in self.strategy_performance.values()),
                'best_strategy': max(stats.keys(), key=lambda k: stats[k]['avg_score']) if stats else None
            }

# Integration with existing task processing
class TaskOptimizationMixin:
    """Mixin to add live optimization to your existing task processor"""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.live_optimizer = LivePromptOptimizer()
        self.logger = logging.getLogger(__name__)
    
    async def handle_zero_fidelity_task(self, task, lora_used: str) -> Optional[Dict[str, Any]]:
        """Handle task that got 0.0 fidelity score with live optimization"""
        
        self.logger.warning(f"🚨 Zero fidelity detected for: '{task.prompt}'")
        self.logger.info(f"   LoRA used: {lora_used}")
        
        # Fast optimization (2-3 seconds vs 2.5-10 minutes for RL)
        optimized_prompt = self.live_optimizer.optimize_for_zero_recovery(
            original_prompt=task.prompt,
            lora_used=lora_used
        )
        
        self.logger.info(f"🔧 Retrying with optimized prompt: '{optimized_prompt}'")
        
        # Update task with optimized prompt and retry
        task.optimized_prompt = optimized_prompt
        
        # Retry generation with optimized prompt
        try:
            result = await self._generate_with_optimized_prompt(task, lora_used, optimized_prompt)
            
            if result and result.get('task_fidelity_score', 0.0) > 0.0:
                self.logger.info(f"✅ Recovery successful! New score: {result['task_fidelity_score']:.3f}")
                return result
            else:
                self.logger.warning(f"⚠️ Optimization didn't fully recover. Score: {result.get('task_fidelity_score', 0.0):.3f}")
                return result
                
        except Exception as e:
            self.logger.error(f"❌ Optimized retry failed: {e}")
            return None
    
    def log_optimization_performance(self, task, original_score: float, optimized_score: float):
        """Log optimization performance for analysis"""
        improvement = optimized_score - original_score
        
        self.logger.info(f"📈 Optimization performance:")
        self.logger.info(f"   Original score: {original_score:.3f}")
        self.logger.info(f"   Optimized score: {optimized_score:.3f}") 
        self.logger.info(f"   Improvement: {improvement:+.3f}")
        
        # Add to your task processing CSV
        optimization_log = {
            'timestamp': time.time(),
            'original_prompt': task.prompt,
            'optimized_prompt': getattr(task, 'optimized_prompt', ''),
            'original_score': original_score,
            'optimized_score': optimized_score,
            'improvement': improvement,
            'lora_used': getattr(task, 'lora_used', ''),
            'optimization_successful': improvement > 0.1
        }
        
        # Save to CSV for analysis
        self._save_optimization_log(optimization_log)

# Example usage for your existing orchestrator
"""
class SmartOrchestrator(TaskOptimizationMixin):
    async def generate_3d_model(self, task):
        # Your existing generation logic
        result = await super().generate_3d_model(task)
        
        # Check for zero fidelity and optimize if needed
        if result and result.get('task_fidelity_score', 0.0) == 0.0:
            lora_used = self._extract_lora_from_endpoint(self.current_endpoint)
            optimized_result = await self.handle_zero_fidelity_task(task, lora_used)
            
            if optimized_result:
                self.log_optimization_performance(
                    task, 
                    0.0, 
                    optimized_result.get('task_fidelity_score', 0.0)
                )
                return optimized_result
        
        return result
"""

if __name__ == "__main__":
    # Test the live optimizer
    optimizer = LivePromptOptimizer()
    
    # Test cases from your data
    test_prompts = [
        "steel long-handled spade",  # Known to get 0.0 with baolei
        "robot that is orange and has pointed head",
        "large dark purple pyramid shaped gemstone"
    ]
    
    for prompt in test_prompts:
        print(f"\n🧪 Testing: '{prompt}'")
        
        # Test with different LoRAs
        for lora in ['baolei', 'cartoon_3d']:
            optimized = optimizer.optimize_for_zero_recovery(prompt, lora)
            print(f"   {lora}: '{optimized}'")
    
    # Show performance stats
    stats = optimizer.get_performance_stats()
    print(f"\n📊 Performance: {stats}")
