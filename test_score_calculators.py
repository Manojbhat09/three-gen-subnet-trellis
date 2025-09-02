#!/usr/bin/env python3
"""
Test script for the new score calculator functionality
Demonstrates the different scoring options available
"""

import subprocess
import sys
import time

def run_command(cmd):
    """Run a command and return the output"""
    print(f"🚀 Running: {' '.join(cmd)}")
    start_time = time.time()
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        end_time = time.time()
        
        print(f"⏱️ Command completed in {end_time - start_time:.2f}s")
        print(f"📤 Return code: {result.returncode}")
        
        if result.stdout:
            print("📋 Output:")
            print(result.stdout)
        
        if result.stderr:
            print("⚠️ Errors:")
            print(result.stderr)
            
        return result.returncode == 0
        
    except subprocess.TimeoutExpired:
        print("⏰ Command timed out after 300 seconds")
        return False
    except Exception as e:
        print(f"❌ Command failed: {e}")
        return False

def test_score_calculators():
    """Test all the new score calculator options"""
    
    test_prompt = "a red sports car"
    
    print("🧪 Testing Score Calculators")
    print("=" * 60)
    print(f"Test prompt: '{test_prompt}'")
    print("=" * 60)
    
    # Test alignment score only
    print("\n1️⃣ Testing Alignment Score Only")
    print("-" * 40)
    success = run_command([
        "python", "test_rl_standalone.py", test_prompt, 
        "--alignment-score", "--endpoint", "generate_both/cinema/"
    ])
    print(f"✅ Alignment score test: {'PASSED' if success else 'FAILED'}")
    
    # Test CLIP score only
    print("\n2️⃣ Testing CLIP Score Only")
    print("-" * 40)
    success = run_command([
        "python", "test_rl_standalone.py", test_prompt, 
        "--clip-score", "--endpoint", "generate_both/cinema/"
    ])
    print(f"✅ CLIP score test: {'PASSED' if success else 'FAILED'}")
    
    # Test both scores
    print("\n3️⃣ Testing Both Scores")
    print("-" * 40)
    success = run_command([
        "python", "test_rl_standalone.py", test_prompt, 
        "--both-scores", "--endpoint", "generate_both/cinema/"
    ])
    print(f"✅ Both scores test: {'PASSED' if success else 'FAILED'}")
    
    print("\n🎉 Score calculator tests completed!")

if __name__ == "__main__":
    test_score_calculators()


