#!/usr/bin/env python3
"""
Test script for CLIP RL Learning Integration
"""

import sys
import os

# Add the current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from get_max_clip_score import CLIPScoreMaximizer, RLLoopAgent

def test_rl_integration():
    """Test the RL integration with a simple prompt"""
    print("🧪 Testing CLIP RL Learning Integration")
    print("=" * 50)
    
    # Initialize the CLIP maximizer
    maximizer = CLIPScoreMaximizer(
        dit_server_url="http://localhost:8096",
        max_iterations=3,  # Reduced for testing
        target_score=0.8,
        min_improvement=0.01
    )
    
    # Test prompt
    test_prompt = "a wooden chair"
    
    print(f"📝 Test prompt: '{test_prompt}'")
    print(f"🔧 CLIP maximizer initialized")
    
    try:
        # Test RL learning mode
        print("\n🔄 Testing RL learning mode...")
        result = maximizer.maximize_clip_score_with_rl(test_prompt, seed=42)
        
        print(f"\n✅ RL test completed!")
        print(f"   Final optimized prompt: '{result['final_optimized_prompt']}'")
        print(f"   Final score: {result['final_score']:.4f}")
        print(f"   Total rounds: {result['total_rounds']}")
        print(f"   Convergence achieved: {result['convergence_achieved']}")
        
        # Test insights
        print("\n🧠 Testing RL insights...")
        rl_agent = RLLoopAgent(maximizer)
        insights = rl_agent.get_rl_insights()
        print(f"   Total RL sessions: {insights.get('total_rl_sessions', 0)}")
        print(f"   Current exploration rate: {insights.get('current_exploration_rate', 0):.2f}")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_traditional_mode():
    """Test the traditional optimization mode"""
    print("\n🧪 Testing Traditional Optimization Mode")
    print("=" * 50)
    
    # Initialize the CLIP maximizer
    maximizer = CLIPScoreMaximizer(
        dit_server_url="http://localhost:8096",
        max_iterations=2,  # Reduced for testing
        target_score=0.8,
        min_improvement=0.01
    )
    
    # Test prompt
    test_prompt = "a red apple"
    
    print(f"📝 Test prompt: '{test_prompt}'")
    
    try:
        # Test traditional mode
        print("\n🔄 Testing traditional optimization...")
        result = maximizer.maximize_clip_score(test_prompt, seed=42)
        
        print(f"\n✅ Traditional test completed!")
        print(f"   Best prompt: '{result['best_prompt']}'")
        print(f"   Best score: {result['best_score']:.4f}")
        print(f"   Iterations: {result['iterations']}")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Traditional test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("🚀 CLIP RL Learning Integration Test")
    print("=" * 60)
    
    # Test traditional mode first
    traditional_success = test_traditional_mode()
    
    # Test RL mode
    rl_success = test_rl_integration()
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 TEST SUMMARY")
    print("=" * 60)
    print(f"Traditional optimization: {'✅ PASSED' if traditional_success else '❌ FAILED'}")
    print(f"RL learning integration: {'✅ PASSED' if rl_success else '❌ FAILED'}")
    
    if traditional_success and rl_success:
        print("\n🎉 All tests passed! RL learning integration is working correctly.")
    else:
        print("\n⚠️ Some tests failed. Check the error messages above.")
        sys.exit(1) 