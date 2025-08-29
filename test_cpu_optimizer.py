#!/usr/bin/env python3
"""
CPU-based CLIP Optimizer Test Script
=====================================

This script tests the CPU-based CLIP optimizer with vLLM integration.
Make sure you have:
1. vLLM running on port 11300
2. Image generation server running on port 8096
3. Required Python packages installed
"""

import sys
import os
sys.path.append('/home/mbhat/three-gen-subnet-trellis')

from clip_episodic_optimizer import MultiGeneratorCLIPOptimizer

def main():
    print("🧪 Testing CPU-based CLIP Optimizer")
    print("=" * 50)

    # Test prompts
    test_prompts = [
        "small wooden hammer",
        "red sports car",
        "crystal vase"
    ]

    # Create optimizer
    optimizer = MultiGeneratorCLIPOptimizer(
        num_episodes=1,
        target_score=0.7,
        max_rounds_per_episode=5,  # Quick test
        enable_router=False,
        use_cpu=True,
        vllm_url="http://localhost:11300"
    )

    try:
        results = optimizer.run_all_episodes(test_prompts)
        print("\n✅ Test completed successfully!")
        print(f"Results: {results.get('total_episodes', 0)} episodes run")

    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()

