#!/usr/bin/env python3
"""
Miner Integration Example
========================
Shows how to integrate the RL prompt optimizer with the miner for improved fidelity scores.
"""

import time
from typing import Dict, List, Optional
from rl_prompt_optimizer_inference import PromptOptimizerAPI

class MinerWithRLOptimizer:
    """Enhanced miner with RL prompt optimization"""
    
    def __init__(self, model_path: str, enable_optimization: bool = True):
        self.enable_optimization = enable_optimization
        
        if enable_optimization:
            self.optimizer = PromptOptimizerAPI(model_path)
            print("🚀 Miner initialized with RL prompt optimization")
        else:
            self.optimizer = None
            print("📝 Miner initialized without optimization")
        
        # Performance tracking
        self.optimization_stats = {
            "total_prompts": 0,
            "optimized_prompts": 0,
            "score_improvements": 0,
            "average_improvement": 0.0,
            "optimization_time": 0.0
        }

    def generate_3d_model(self, prompt: str, optimize: bool = True) -> Dict:
        """
        Generate 3D model with optional prompt optimization
        
        Args:
            prompt: Input prompt
            optimize: Whether to optimize the prompt first
            
        Returns:
            Generation result with optimization metadata
        """
        
        start_time = time.time()
        original_prompt = prompt
        optimization_metadata = {}
        
        self.optimization_stats["total_prompts"] += 1
        
        # Optimize prompt if enabled
        if optimize and self.enable_optimization:
            print(f"🔧 Optimizing prompt: {prompt}")
            
            optimization_result = self.optimizer.optimize_for_miner(
                prompt=prompt,
                urgency="normal"  # Can be "fast", "normal", or "thorough"
            )
            
            if optimization_result["success"]:
                prompt = optimization_result["optimized_prompt"]
                self.optimization_stats["optimized_prompts"] += 1
                self.optimization_stats["optimization_time"] += optimization_result["optimization_time"]
                
                predicted_improvement = optimization_result["predicted_improvement"]
                if predicted_improvement > 0.7:  # Threshold for counting as improvement
                    self.optimization_stats["score_improvements"] += 1
                
                optimization_metadata = {
                    "optimized": True,
                    "original_prompt": original_prompt,
                    "optimized_prompt": prompt,
                    "predicted_score": predicted_improvement,
                    "confidence": optimization_result["confidence"],
                    "optimization_time": optimization_result["optimization_time"],
                    "actions_taken": optimization_result["actions_taken"]
                }
                
                print(f"✅ Optimization complete: {predicted_improvement:.3f} predicted score")
            else:
                optimization_metadata = {
                    "optimized": False,
                    "error": optimization_result.get("error", "Optimization failed")
                }
                print(f"❌ Optimization failed: {optimization_result.get('error', 'Unknown error')}")
        
        # Simulate 3D model generation (replace with actual generation code)
        generation_result = self._simulate_3d_generation(prompt)
        
        # Combine results
        total_time = time.time() - start_time
        
        result = {
            "prompt_used": prompt,
            "generation_result": generation_result,
            "optimization_metadata": optimization_metadata,
            "total_time": total_time,
            "validation_score": generation_result.get("validation_score", 0.0)
        }
        
        return result

    def _simulate_3d_generation(self, prompt: str) -> Dict:
        """Simulate 3D model generation (replace with actual TRELLIS code)"""
        
        # This is where you'd call your actual 3D generation pipeline
        # For example:
        # result = trellis_generate(prompt)
        # validation_score = validate_model(result)
        
        # Simulated result for demo
        time.sleep(0.1)  # Simulate generation time
        
        return {
            "model_path": f"generated_models/model_{int(time.time())}.ply",
            "validation_score": 0.75 + (len(prompt) % 20) * 0.01,  # Simulated score
            "generation_time": 30.5,  # Simulated time
            "success": True
        }

    def batch_generate_with_optimization(self, prompts: List[str], 
                                       optimize_all: bool = True) -> List[Dict]:
        """Generate 3D models for multiple prompts with batch optimization"""
        
        print(f"🎯 Batch generation for {len(prompts)} prompts")
        
        results = []
        start_time = time.time()
        
        for i, prompt in enumerate(prompts, 1):
            print(f"\n📝 Processing prompt {i}/{len(prompts)}: {prompt[:50]}...")
            
            result = self.generate_3d_model(prompt, optimize=optimize_all)
            results.append(result)
            
            # Progress update
            if i % 5 == 0:
                elapsed = time.time() - start_time
                avg_time = elapsed / i
                eta = avg_time * (len(prompts) - i)
                print(f"   Progress: {i}/{len(prompts)} (ETA: {eta:.1f}s)")
        
        total_time = time.time() - start_time
        print(f"\n✅ Batch generation complete: {total_time:.2f}s total")
        
        # Batch analysis
        self._analyze_batch_performance(results)
        
        return results

    def _analyze_batch_performance(self, results: List[Dict]):
        """Analyze batch performance and optimization effectiveness"""
        
        optimized_results = [r for r in results if r["optimization_metadata"].get("optimized", False)]
        unoptimized_results = [r for r in results if not r["optimization_metadata"].get("optimized", False)]
        
        if optimized_results:
            opt_scores = [r["validation_score"] for r in optimized_results]
            avg_opt_score = sum(opt_scores) / len(opt_scores)
            
            print(f"\n📊 BATCH PERFORMANCE ANALYSIS:")
            print(f"   Total prompts: {len(results)}")
            print(f"   Optimized: {len(optimized_results)}")
            print(f"   Average optimized score: {avg_opt_score:.3f}")
            
            if unoptimized_results:
                unopt_scores = [r["validation_score"] for r in unoptimized_results]
                avg_unopt_score = sum(unopt_scores) / len(unopt_scores)
                improvement = avg_opt_score - avg_unopt_score
                print(f"   Average unoptimized score: {avg_unopt_score:.3f}")
                print(f"   Optimization benefit: {improvement:+.3f}")
            
            ultra_count = sum(1 for score in opt_scores if score >= 0.96)
            print(f"   Ultra achievements: {ultra_count}/{len(opt_scores)} ({ultra_count/len(opt_scores)*100:.1f}%)")

    def get_optimization_stats(self) -> Dict:
        """Get comprehensive optimization statistics"""
        
        stats = self.optimization_stats.copy()
        
        if stats["optimized_prompts"] > 0:
            stats["optimization_rate"] = stats["optimized_prompts"] / stats["total_prompts"]
            stats["improvement_rate"] = stats["score_improvements"] / stats["optimized_prompts"]
            stats["avg_optimization_time"] = stats["optimization_time"] / stats["optimized_prompts"]
        else:
            stats["optimization_rate"] = 0.0
            stats["improvement_rate"] = 0.0
            stats["avg_optimization_time"] = 0.0
        
        return stats

    def enable_adaptive_optimization(self, score_threshold: float = 0.6):
        """Enable adaptive optimization based on prompt performance"""
        
        def adaptive_optimize(prompt: str) -> bool:
            # Quick score prediction to decide if optimization is needed
            if len(prompt) < 50:  # Short prompts likely need optimization
                return True
            if "aerospace" in prompt or "precision" in prompt:  # Already optimized
                return False
            return True  # Default to optimize
        
        self.adaptive_optimize = adaptive_optimize
        print(f"🧠 Adaptive optimization enabled (threshold: {score_threshold})")

def main():
    """Demo miner integration"""
    
    print("🚀 MINER + RL OPTIMIZER INTEGRATION DEMO")
    print("="*60)
    
    # For demo purposes, we'll simulate the integration
    # In production, use the actual trained model path
    model_path = "trained_models/rl_prompt_optimizer.pth"
    
    # Test prompts
    test_prompts = [
        "hexagonal prism steel structure",
        "elegant silk fabric draping",
        "transparent glass sphere with reflections",
        "ornate wooden sculpture",
        "modern geometric lamp design"
    ]
    
    try:
        # Initialize miner with RL optimization
        miner = MinerWithRLOptimizer(model_path, enable_optimization=True)
        
        # Test single generation
        print("\n🎯 Testing single generation with optimization:")
        result = miner.generate_3d_model(test_prompts[0])
        
        print(f"📊 GENERATION RESULT:")
        print(f"   Original: {result['optimization_metadata'].get('original_prompt', 'N/A')}")
        print(f"   Optimized: {result['prompt_used']}")
        print(f"   Score: {result['validation_score']:.3f}")
        print(f"   Time: {result['total_time']:.2f}s")
        
        # Test batch generation
        print(f"\n🔄 Testing batch generation:")
        batch_results = miner.batch_generate_with_optimization(test_prompts[:3])
        
        # Show statistics
        stats = miner.get_optimization_stats()
        print(f"\n📈 OPTIMIZATION STATISTICS:")
        print(f"   Optimization Rate: {stats['optimization_rate']:.1%}")
        print(f"   Improvement Rate: {stats['improvement_rate']:.1%}")
        print(f"   Avg Optimization Time: {stats['avg_optimization_time']:.2f}s")
        
    except FileNotFoundError:
        print("❌ Model file not found - this is expected in demo mode")
        print("✅ Integration architecture is ready for production!")
        
        # Show what the integration would look like
        print(f"\n🔧 PRODUCTION INTEGRATION STEPS:")
        print(f"1. Train RL model: python rl_prompt_optimizer_fixed.py")
        print(f"2. Save production model: python save_production_model.py") 
        print(f"3. Initialize miner: miner = MinerWithRLOptimizer('trained_model.pth')")
        print(f"4. Generate with optimization: result = miner.generate_3d_model(prompt)")

if __name__ == "__main__":
    main() 