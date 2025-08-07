#!/usr/bin/env python3
"""
Debug Validation
Purpose: Debug the validation step to identify the issue
"""

import subprocess
import json
import time
import sys

def test_validation_step():
    """Test the validation step with detailed output"""
    print("🔍 Debugging Validation Step")
    print("=" * 50)
    
    # Test with a simple case first
    original_prompt = "greek amphora scene detail"
    optimized_prompt = "Isometric 3D, greek amphora scene detail"
    
    print(f"Original prompt: '{original_prompt}'")
    print(f"Optimized prompt: '{optimized_prompt}'")
    
    # Check if we have a PLY file to validate
    import os
    if os.path.exists("benchmark_outputs"):
        files = os.listdir("benchmark_outputs")
        print(f"Found {len(files)} files in benchmark_outputs:")
        for f in files:
            print(f"  - {f}")
    
    # Try running validation with more verbose output
    cmd = [
        "bash", "-c",
        f"source /home/mbhat/miniconda/bin/activate && conda activate trellis_new && python subnet_accurate_validator.py \"{original_prompt}\" \"{optimized_prompt}\" 2>&1"
    ]
    
    print(f"\nRunning command: {' '.join(cmd)}")
    print("=" * 50)
    
    try:
        # Run with timeout and capture all output
        result = subprocess.run(
            cmd, 
            capture_output=True, 
            text=True, 
            timeout=120  # 2 minutes timeout
        )
        
        print(f"Return code: {result.returncode}")
        print(f"stdout length: {len(result.stdout)}")
        print(f"stderr length: {len(result.stderr)}")
        
        if result.stdout:
            print("\nSTDOUT:")
            print(result.stdout)
        
        if result.stderr:
            print("\nSTDERR:")
            print(result.stderr)
        
        if result.returncode == 0:
            # Check if validation results file was created
            if os.path.exists("subnet_validation_results.json"):
                with open("subnet_validation_results.json", 'r') as f:
                    data = json.load(f)
                    print("\n✅ Validation Results:")
                    print(json.dumps(data, indent=2))
            else:
                print("\n❌ Validation results file not found")
        else:
            print(f"\n❌ Validation failed with return code {result.returncode}")
            
    except subprocess.TimeoutExpired:
        print("\n❌ Validation timed out after 2 minutes")
    except Exception as e:
        print(f"\n❌ Exception during validation: {e}")

def test_simple_validation():
    """Test validation with just the original prompt"""
    print("\n🔍 Testing Simple Validation (original prompt only)")
    print("=" * 50)
    
    original_prompt = "greek amphora scene detail"
    
    cmd = [
        "bash", "-c",
        f"source /home/mbhat/miniconda/bin/activate && conda activate trellis_new && python subnet_accurate_validator.py \"{original_prompt}\" 2>&1"
    ]
    
    try:
        result = subprocess.run(
            cmd, 
            capture_output=True, 
            text=True, 
            timeout=120
        )
        
        print(f"Return code: {result.returncode}")
        
        if result.stdout:
            print("\nSTDOUT:")
            print(result.stdout[-500:])  # Last 500 chars
        
        if result.stderr:
            print("\nSTDERR:")
            print(result.stderr[-500:])  # Last 500 chars
            
    except subprocess.TimeoutExpired:
        print("❌ Simple validation timed out")
    except Exception as e:
        print(f"❌ Exception: {e}")

if __name__ == "__main__":
    test_validation_step()
    test_simple_validation() 