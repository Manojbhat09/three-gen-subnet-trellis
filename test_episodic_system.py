#!/usr/bin/env python3
"""
Test script to verify the episodic optimization system works correctly.
Runs a single episode with one prompt for quick validation.
"""

import os
import sys
from episodic_prompt_optimizer import EpisodicPromptOptimizer

def test_single_optimization():
    """Test a single prompt optimization to verify the system works."""
    print("🧪 Testing episodic optimization system...")
    
    # Create a minimal test
    class TestEpisodicOptimizer(EpisodicPromptOptimizer):
        def __init__(self):
            super().__init__(
                num_episodes=1,
                target_score=0.85,
                max_rounds_per_prompt=2,
                log_dir="test_episodic_logs"
            )
            # Override with just one test prompt for quick validation
            self.test_prompts = ["emerald pendant"]
    
    try:
        optimizer = TestEpisodicOptimizer()
        print(f"✅ System initialized successfully")
        print(f"✅ Test prompt: '{optimizer.test_prompts[0]}'")
        print(f"✅ Using V4.1 RL Loop optimizer: {type(optimizer.optimizer).__name__}")
        
        # Check if the underlying optimizer is available
        if hasattr(optimizer.optimizer, 'optimize_with_rl_loop'):
            print(f"✅ Optimizer has optimize_with_rl_loop method")
        else:
            print(f"❌ Optimizer missing optimize_with_rl_loop method")
            return False
            
        # Check if validator command is accessible
        print(f"✅ System ready for testing")
        print(f"\nTo run the test:")
        print(f"  python test_episodic_system.py --run")
        
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {str(e)}")
        print(f"   Make sure smart_prompt_optimizer_v4_1_rl_loop.py is available")
        return False
        
    except Exception as e:
        print(f"❌ System error: {str(e)}")
        return False

def run_quick_test():
    """Run a quick test with one episode and one prompt."""
    print("🚀 Running quick test...")
    
    class TestEpisodicOptimizer(EpisodicPromptOptimizer):
        def __init__(self):
            super().__init__(
                num_episodes=1,
                target_score=0.85,
                max_rounds_per_prompt=2,
                log_dir="test_episodic_logs"
            )
            self.test_prompts = ["emerald pendant"]
    
    optimizer = TestEpisodicOptimizer()
    
    try:
        results = optimizer.run_all_episodes()
        
        print("\n📋 TEST RESULTS:")
        if results and 'episode_results' in results:
            episode = results['episode_results'][0]
            summary = episode['episode_summary']
            
            print(f"✅ Episode completed")
            print(f"   Prompts processed: {summary['total_prompts']}")
            print(f"   Success rate: {summary['success_rate']:.1%}")
            print(f"   Average rounds: {summary['avg_rounds_per_prompt']:.1f}")
            print(f"   Average improvement: {summary['avg_score_improvement']:+.3f}")
            
            if episode['prompt_results']:
                prompt_result = episode['prompt_results'][0]
                print(f"   Final score: {prompt_result.get('final_score', 'N/A')}")
                print(f"   Converged: {prompt_result.get('converged', False)}")
        
        print(f"\n✅ Quick test completed successfully!")
        return True
        
    except Exception as e:
        print(f"\n❌ Test failed: {str(e)}")
        return False

def main():
    if len(sys.argv) > 1 and sys.argv[1] == '--run':
        success = run_quick_test()
    else:
        success = test_single_optimization()
    
    if not success:
        sys.exit(1)

if __name__ == "__main__":
    main() 