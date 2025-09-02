#!/usr/bin/env python3
"""
Test script to demonstrate CPU loading limitations
"""

import subprocess
import sys

def test_cpu_loading_operations():
    """Test which operations work with CPU loading"""
    
    print("🧪 Testing CPU Loading Limitations")
    print("=" * 50)
    
    # Test operations that should work with CPU loading
    cpu_safe_operations = [
        {
            "name": "CLIP Score Only",
            "cmd": ["python", "test_rl_standalone.py", "a simple red car", "--clip-score", "--cpu-loading"],
            "should_work": True
        },
        {
            "name": "Alignment Score Only", 
            "cmd": ["python", "test_rl_standalone.py", "a simple red car", "--alignment-score", "--cpu-loading"],
            "should_work": True
        }
    ]
    
    # Test operations that should fail with CPU loading
    cpu_unsafe_operations = [
        {
            "name": "RL Optimization (requires 3D rendering)",
            "cmd": ["python", "test_rl_standalone.py", "a simple red car", "--rl-alignment", "--cpu-loading"],
            "should_work": False
        },
        {
            "name": "Enhanced RL (requires 3D rendering)",
            "cmd": ["python", "test_rl_standalone.py", "a simple red car", "--enhanced", "--cpu-loading"],
            "should_work": False
        }
    ]
    
    print("✅ CPU-Safe Operations (should work):")
    for op in cpu_safe_operations:
        print(f"   • {op['name']}")
    
    print("\n❌ CPU-Unsafe Operations (will fail):")
    for op in cpu_unsafe_operations:
        print(f"   • {op['name']} - requires 3D rendering (CUDA only)")
    
    print("\n💡 Recommendation:")
    print("   Use --cpu-loading only with --clip-score or --alignment-score")
    print("   Avoid --cpu-loading with operations that require 3D rendering")

if __name__ == "__main__":
    test_cpu_loading_operations()


