#!/usr/bin/env python3
"""
Test script to verify the RL fix for the 'best_image' error
"""

import sys
import os

# Add the current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from get_max_clip_score import CLIPScoreMaximizer, RLLoopAgent

def test_rl_fix():
    """Test that RL mode now includes best_image in results"""
    print("🧪 Testing RL Fix for 'best_image' Error")
    print("=" * 50)
    
    # Initialize the CLIP maximizer
    maximizer = CLIPScoreMaximizer(
        dit_server_url="http://localhost:8096",
        max_iterations=2,  # Very short for testing
        target_score=0.8,
        min_improvement=0.01
    )
    
    # Test prompt
    test_prompt = "a simple cube"
    
    print(f"📝 Test prompt: '{test_prompt}'")
    
    try:
        # Test RL learning mode
        print("\n🔄 Testing RL learning mode...")
        result = maximizer.maximize_clip_score_with_rl(test_prompt, seed=42)
        
        # Check if best_image is present
        if 'best_image' in result:
            print(f"✅ 'best_image' field is present in result")
            if result['best_image']:
                print(f"   Image data length: {len(result['best_image'])} characters")
            else:
                print(f"   Image data is None (no successful generation)")
        else:
            print(f"❌ 'best_image' field is missing from result")
            return False
        
        # Check other required fields
        required_fields = [
            'session_id', 'original_prompt', 'final_optimized_prompt', 
            'final_score', 'total_rounds', 'convergence_achieved'
        ]
        
        missing_fields = []
        for field in required_fields:
            if field not in result:
                missing_fields.append(field)
        
        if missing_fields:
            print(f"❌ Missing required fields: {missing_fields}")
            return False
        else:
            print(f"✅ All required fields are present")
        
        print(f"\n✅ RL fix test completed successfully!")
        print(f"   Final optimized prompt: '{result['final_optimized_prompt']}'")
        print(f"   Final score: {result['final_score']:.4f}")
        print(f"   Total rounds: {result['total_rounds']}")
        print(f"   Convergence achieved: {result['convergence_achieved']}")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("🚀 RL Fix Test")
    print("=" * 60)
    
    success = test_rl_fix()
    
    print("\n" + "=" * 60)
    print("📊 TEST SUMMARY")
    print("=" * 60)
    print(f"RL fix test: {'✅ PASSED' if success else '❌ FAILED'}")
    
    if success:
        print("\n🎉 RL fix is working correctly! The 'best_image' error has been resolved.")
    else:
        print("\n⚠️ Test failed. The fix may need further investigation.")
        sys.exit(1) 