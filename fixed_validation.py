#!/usr/bin/env python3
"""
Fixed Validation Script
Purpose: Use the correct endpoint for validation
"""

import json
import requests
import time
import subprocess
import base64
from pathlib import Path

def validate_with_correct_endpoint(original_prompt: str, optimized_prompt: str = None):
    """Validate using the correct endpoint"""
    print("🔍 Fixed Validation with Correct Endpoint")
    print("=" * 50)
    
    # Use the correct endpoint
    url = "http://127.0.0.1:8096/generate/"
    
    # Use optimized prompt for generation if provided
    generation_prompt = optimized_prompt if optimized_prompt else original_prompt
    
    print(f"🎨 Generating 3D model for: '{generation_prompt}'")
    
    try:
        with requests.post(url, data={'prompt': generation_prompt}, timeout=300, stream=False) as response:
            response.raise_for_status()
            
            compression = response.headers.get('x-compression', 'none')
            content_length = len(response.content)
            
            print(f"📦 Response received: {content_length:,} bytes (compression: {compression})")
            
            # Save the file for validation
            output_dir = Path("./validation_outputs")
            output_dir.mkdir(exist_ok=True)
            
            filename = f"validation_{original_prompt.replace(' ', '_')}_{int(time.time())}.ply.spz"
            filepath = output_dir / filename
            
            with open(filepath, 'wb') as f:
                f.write(response.content)
            
            print(f"💾 Saved to: {filepath}")
            
            # Now run validation using the subnet_accurate_validator
            print("\n🔍 Running validation...")
            
            cmd = [
                "bash", "-c",
                f"source /home/mbhat/miniconda/bin/activate && conda activate trellis_new && python subnet_accurate_validator.py \"{original_prompt}\" \"{generation_prompt}\" 2>&1"
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
            
            if result.returncode == 0:
                # Parse validation results
                with open("subnet_validation_results.json", 'r') as f:
                    data = json.load(f)
                    score = data.get("validation_engine_score", 0.0)
                    alignment_score = data.get("alignment_score", 0.0)
                    quality_score = data.get("quality_score", 0.0)
                    
                    print(f"\n✅ Validation Results:")
                    print(f"📊 Validation Score: {score:.4f}")
                    print(f"📊 Alignment Score: {alignment_score:.4f}")
                    print(f"📊 Quality Score: {quality_score:.4f}")
                    
                    return score, alignment_score, quality_score
            else:
                print(f"❌ Validation failed (return code {result.returncode})")
                print(f"STDOUT: {result.stdout}")
                print(f"STDERR: {result.stderr}")
                return 0.0, 0.0, 0.0
                
    except requests.exceptions.RequestException as e:
        print(f"❌ Generation request failed: {e}")
        return 0.0, 0.0, 0.0
    except Exception as e:
        print(f"❌ Validation failed: {e}")
        return 0.0, 0.0, 0.0

def test_lora_validation():
    """Test validation with a LoRA"""
    print("🎨 Testing LoRA Validation")
    print("=" * 50)
    
    # Test with isometric_3d LoRA
    original_prompt = "greek amphora scene detail"
    optimized_prompt = "Isometric 3D, greek amphora scene detail"
    
    print(f"Original prompt: '{original_prompt}'")
    print(f"LoRA enhanced prompt: '{optimized_prompt}'")
    
    # First, generate with LoRA
    url = "http://127.0.0.1:8096/generate/isometric_3d/"
    
    try:
        response = requests.post(
            url,
            data={
                'prompt': optimized_prompt,
                'seed': 42,
                'return_compressed': True
            },
            timeout=300
        )
        
        if response.status_code == 200:
            file_size = len(response.content)
            print(f"✅ LoRA generation successful: {file_size:,} bytes")
            
            # Save the file
            output_dir = Path("./validation_outputs")
            output_dir.mkdir(exist_ok=True)
            
            filename = f"lora_validation_{original_prompt.replace(' ', '_')}_{int(time.time())}.ply.spz"
            filepath = output_dir / filename
            
            with open(filepath, 'wb') as f:
                f.write(response.content)
            
            print(f"💾 Saved to: {filepath}")
            
            # Now validate
            print("\n🔍 Running validation...")
            
            cmd = [
                "bash", "-c",
                f"source /home/mbhat/miniconda/bin/activate && conda activate trellis_new && python subnet_accurate_validator.py \"{original_prompt}\" \"{optimized_prompt}\" 2>&1"
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
            
            if result.returncode == 0:
                # Parse validation results
                with open("subnet_validation_results.json", 'r') as f:
                    data = json.load(f)
                    score = data.get("validation_engine_score", 0.0)
                    alignment_score = data.get("alignment_score", 0.0)
                    quality_score = data.get("quality_score", 0.0)
                    
                    print(f"\n✅ LoRA Validation Results:")
                    print(f"📊 Validation Score: {score:.4f}")
                    print(f"📊 Alignment Score: {alignment_score:.4f}")
                    print(f"📊 Quality Score: {quality_score:.4f}")
                    
                    return score, alignment_score, quality_score
            else:
                print(f"❌ Validation failed (return code {result.returncode})")
                print(f"STDOUT: {result.stdout}")
                print(f"STDERR: {result.stderr}")
                return 0.0, 0.0, 0.0
                
        else:
            print(f"❌ LoRA generation failed: HTTP {response.status_code}")
            return 0.0, 0.0, 0.0
            
    except Exception as e:
        print(f"❌ LoRA validation failed: {e}")
        return 0.0, 0.0, 0.0

if __name__ == "__main__":
    print("🚀 Fixed Validation Test")
    print("=" * 60)
    
    # Test 1: Basic validation
    print("\n1️⃣ Testing basic validation...")
    score1, align1, qual1 = validate_with_correct_endpoint("greek amphora scene detail")
    
    # Test 2: LoRA validation
    print("\n2️⃣ Testing LoRA validation...")
    score2, align2, qual2 = test_lora_validation()
    
    print("\n" + "="*60)
    print("📊 COMPARISON RESULTS")
    print("="*60)
    print(f"Basic Generation:")
    print(f"  Validation Score: {score1:.4f}")
    print(f"  Alignment Score: {align1:.4f}")
    print(f"  Quality Score: {qual1:.4f}")
    print(f"\nLoRA Generation:")
    print(f"  Validation Score: {score2:.4f}")
    print(f"  Alignment Score: {align2:.4f}")
    print(f"  Quality Score: {qual2:.4f}")
    
    if score2 > score1:
        print(f"\n✅ LoRA improved score by {score2 - score1:.4f}")
    elif score2 < score1:
        print(f"\n❌ LoRA decreased score by {score1 - score2:.4f}")
    else:
        print(f"\n🟡 LoRA had no effect on score") 