#!/usr/bin/env python3
"""
Smart Mining Integration with Intelligent LLaMA Optimization
===========================================================
Replaces rigid RL patterns with intelligent LLaMA 3.2 optimization that:
- Analyzes each prompt individually
- Learns from successful optimizations
- Adapts strategies per prompt category
- Provides custom solutions, not templates

This is the NEW way to integrate with continuous_trellis_orchestrator.py
"""

from intelligent_llama_optimizer import IntelligentLLaMAOptimizer, OptimizationResult
from typing import Dict, Any, Optional
import time
import logging

class SmartMiningOptimizer:
    """Smart mining optimizer using intelligent LLaMA 3.2"""
    
    def __init__(self, ollama_url: str = "http://localhost:11434", 
                 enable_learning: bool = True):
        """
        Initialize smart mining optimizer
        
        Args:
            ollama_url: Ollama server URL
            enable_learning: Enable learning from validation feedback
        """
        self.optimizer = IntelligentLLaMAOptimizer(ollama_url)
        self.enable_learning = enable_learning
        
        # Mining-specific settings
        self.optimization_timeout = 15.0  # 15 seconds max per optimization
        self.min_confidence_threshold = 0.4  # Minimum confidence to use optimization
        
        # Performance tracking
        self.mining_stats = {
            'prompts_optimized': 0,
            'optimizations_used': 0,
            'optimizations_skipped': 0,
            'feedback_learned': 0,
            'avg_score_improvement': 0.0
        }
        
        print(f"🚀 SMART MINING OPTIMIZER INITIALIZED")
        print(f"   🧠 Intelligent LLaMA: ENABLED")
        print(f"   📚 Learning: {'ENABLED' if enable_learning else 'DISABLED'}")
        print(f"   ⏱️ Timeout: {self.optimization_timeout}s")
    
    def optimize_prompt(self, prompt: str) -> str:
        """
        Optimize prompt for mining with intelligent analysis
        
        Args:
            prompt: Original prompt from validator
            
        Returns:
            Optimized prompt ready for 3D generation
        """
        self.mining_stats['prompts_optimized'] += 1
        
        try:
            # Check if prompt is already well-optimized
            if self._is_already_optimized(prompt):
                print(f"✅ Prompt already optimized: {prompt[:50]}...")
                return prompt
            
            # Get intelligent optimization
            start_time = time.time()
            result = self.optimizer.optimize_prompt(prompt)
            
            # Check if optimization is confident enough to use
            if result.confidence >= self.min_confidence_threshold:
                self.mining_stats['optimizations_used'] += 1
                print(f"🧠 Intelligent optimization applied:")
                print(f"   📝 Original: {prompt}")
                print(f"   ✨ Optimized: {result.optimized_prompt}")
                print(f"   🎯 Strategy: {result.strategy_used}")
                print(f"   📊 Confidence: {result.confidence:.3f}")
                print(f"   ⏱️ Time: {result.optimization_time:.2f}s")
                
                return result.optimized_prompt
            else:
                self.mining_stats['optimizations_skipped'] += 1
                print(f"⚠️ Low confidence optimization skipped ({result.confidence:.3f})")
                return self._apply_basic_optimization(prompt)
                
        except Exception as e:
            print(f"❌ Intelligent optimization failed: {e}")
            return self._apply_basic_optimization(prompt)
    
    def _is_already_optimized(self, prompt: str) -> bool:
        """Check if prompt is already well-optimized"""
        prompt_lower = prompt.lower()
        
        # Check for optimization indicators
        has_prefix = prompt_lower.startswith('wbgmsst')
        has_suffix = 'white background' in prompt_lower
        has_quality_terms = any(term in prompt_lower for term in [
            'aerospace', 'defense', 'military', 'precision', 'ultra', 'masterpiece',
            'professional', 'high-quality', 'detailed', 'technical'
        ])
        
        return has_prefix and has_suffix and has_quality_terms
    
    def _apply_basic_optimization(self, prompt: str) -> str:
        """Apply basic optimization as fallback"""
        # Simple rule-based optimization
        if not prompt.lower().startswith('wbgmsst'):
            optimized = f"wbgmsst, {prompt}"
        else:
            optimized = prompt
            
        if not optimized.lower().endswith('white background'):
            if optimized.endswith(','):
                optimized += " white background"
            else:
                optimized += ", white background"
        
        # Add basic quality enhancement
        if not any(term in optimized.lower() for term in ['detailed', 'high-quality', 'precision']):
            parts = optimized.split(', ')
            if len(parts) >= 2:
                parts.insert(-1, "detailed 3D object")
                optimized = ', '.join(parts)
        
        return optimized
    
    def learn_from_validation(self, original_prompt: str, optimized_prompt: str, 
                             validation_score: float):
        """Learn from validation feedback"""
        if not self.enable_learning:
            return
        
        try:
            # Only learn from the intelligent optimizations we made
            if optimized_prompt != original_prompt and validation_score > 0.0:
                # Create a result object for learning
                result = OptimizationResult(
                    original_prompt=original_prompt,
                    optimized_prompt=optimized_prompt,
                    strategy_used="mining_optimization",
                    reasoning="Applied during mining",
                    confidence=0.8,  # Assume good confidence if we used it
                    predicted_score=validation_score,
                    optimization_time=0.0,
                    category_detected=self.optimizer.categorizer.categorize_prompt(original_prompt),
                    key_changes=[],
                    memory_used=[]
                )
                
                # Let the intelligent optimizer learn
                self.optimizer.learn_from_feedback(result, validation_score)
                self.mining_stats['feedback_learned'] += 1
                
                print(f"📚 Learned from validation: {validation_score:.3f}")
                
        except Exception as e:
            print(f"⚠️ Learning from feedback failed: {e}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get mining optimization statistics"""
        total_optimized = max(1, self.mining_stats['prompts_optimized'])
        
        return {
            'mining_stats': self.mining_stats,
            'optimization_usage_rate': self.mining_stats['optimizations_used'] / total_optimized,
            'skip_rate': self.mining_stats['optimizations_skipped'] / total_optimized,
            'learning_rate': self.mining_stats['feedback_learned'] / total_optimized,
            'intelligent_optimizer_stats': self.optimizer.get_stats()
        }

# ========================================================================================
# INTEGRATION FUNCTIONS FOR continuous_trellis_orchestrator.py
# ========================================================================================

def create_smart_optimizer(ollama_url: str = "http://localhost:11434",
                          enable_learning: bool = True) -> SmartMiningOptimizer:
    """
    Factory function to create the smart optimizer for mining integration
    
    Args:
        ollama_url: Ollama server URL (default localhost)
        enable_learning: Enable learning from validation feedback
        
    Returns:
        SmartMiningOptimizer ready for production mining
    """
    return SmartMiningOptimizer(ollama_url, enable_learning)

class SmartOptimizerAdapter:
    """Adapter to make SmartMiningOptimizer compatible with existing orchestrator code"""
    
    def __init__(self, smart_optimizer: SmartMiningOptimizer):
        self.smart_optimizer = smart_optimizer
    
    def optimize_prompt(self, prompt: str, aggressive: bool = False) -> Dict[str, Any]:
        """
        Adapter method to match the existing orchestrator's expected interface
        
        Args:
            prompt: Original prompt
            aggressive: Ignored (intelligent optimizer decides strategy)
            
        Returns:
            Dictionary matching the expected format from TrellisPromptOptimizer
        """
        # Get intelligent optimization
        optimized_prompt = self.smart_optimizer.optimize_prompt(prompt)
        
        # Determine if improvement is expected
        improvement_expected = optimized_prompt != prompt
        
        # Build response in expected format
        return {
            'analysis': {
                'risk_level': 'LOW',  # Intelligent optimizer handles risk assessment
                'risk_factors': []
            },
            'improvement_expected': improvement_expected,
            'optimized_prompt': optimized_prompt,
            'applied_strategies': ['intelligent_llama_optimization']
        }

# ========================================================================================
# USAGE EXAMPLES
# ========================================================================================

def integrate_with_orchestrator_example():
    """Example of how to integrate with continuous_trellis_orchestrator.py"""
    
    print("🔧 INTEGRATION EXAMPLE:")
    print("="*50)
    
    # Method 1: Direct replacement (recommended)
    print("METHOD 1: Direct Replacement")
    print("""
# In continuous_trellis_orchestrator.py, replace around line 460:

# OLD:
from prompt_optimizer import TrellisPromptOptimizer
self.prompt_optimizer = TrellisPromptOptimizer()

# NEW:
from smart_mining_integration import create_smart_optimizer, SmartOptimizerAdapter
smart_optimizer = create_smart_optimizer(enable_learning=True)
self.prompt_optimizer = SmartOptimizerAdapter(smart_optimizer)

# The rest of your code works exactly the same!
# The optimize_prompt_for_generation method doesn't need changes.
    """)
    
    # Method 2: Hybrid approach
    print("\nMETHOD 2: Hybrid with Learning")
    print("""
# In continuous_trellis_orchestrator.py:

from smart_mining_integration import create_smart_optimizer

class ContinuousTrellisOrchestrator:
    def __init__(self, config):
        # ... existing code ...
        self.smart_optimizer = create_smart_optimizer(enable_learning=True)
        
    def optimize_prompt_for_generation(self, task: TaskRecord) -> str:
        # Use intelligent optimization
        optimized = self.smart_optimizer.optimize_prompt(task.prompt)
        return optimized
        
    async def submit_result(self, task: TaskRecord, generation_result: Dict) -> bool:
        success = await super().submit_result(task, generation_result)
        
        # Learn from validation feedback
        if success and task.task_fidelity_score is not None:
            self.smart_optimizer.learn_from_validation(
                task.prompt, 
                optimized_prompt_used,  # You'll need to track this
                task.task_fidelity_score
            )
            
        return success
    """)

def main():
    """Demo the smart mining integration"""
    print("🚀 SMART MINING INTEGRATION DEMO")
    print("="*60)
    
    try:
        # Create smart optimizer
        smart_optimizer = create_smart_optimizer(enable_learning=True)
        
        # Test mining-style optimization
        mining_prompts = [
            "hexagonal prism steel structure",  # From validators
            "elegant silk fabric draping",
            "transparent glass sphere",
            "rusty metal gear mechanism", 
            "wooden sculpture details",
            "plastic toy robot figure",
            "ceramic vase with patterns"
        ]
        
        print(f"\n🎯 Testing smart mining optimization:")
        
        for i, prompt in enumerate(mining_prompts, 1):
            print(f"\n--- MINING TASK {i} ---")
            
            # Optimize as if from mining
            optimized = smart_optimizer.optimize_prompt(prompt)
            
            # Simulate validation feedback
            simulated_score = 0.82 + (i * 0.02)  # Simulate improving scores
            smart_optimizer.learn_from_validation(prompt, optimized, simulated_score)
            
            print(f"📊 Validation Score: {simulated_score:.3f}")
        
        # Show mining statistics
        stats = smart_optimizer.get_stats()
        print(f"\n📊 SMART MINING STATISTICS:")
        print(f"   Prompts Processed: {stats['mining_stats']['prompts_optimized']}")
        print(f"   Optimizations Used: {stats['optimization_usage_rate']:.1%}")
        print(f"   Learning Events: {stats['mining_stats']['feedback_learned']}")
        print(f"   Categories Discovered: {stats['intelligent_optimizer_stats']['categories_encountered']}")
        
        # Test adapter compatibility
        print(f"\n🔧 Testing Adapter Compatibility:")
        adapter = SmartOptimizerAdapter(smart_optimizer)
        result = adapter.optimize_prompt("test metal object")
        print(f"   Adapter Result: {result['improvement_expected']}")
        
    except Exception as e:
        print(f"❌ Demo failed: {e}")
        print("   Ensure Ollama is running with llama3.2:3b model")

if __name__ == "__main__":
    main() 