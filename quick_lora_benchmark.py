#!/usr/bin/env python3
"""
Quick LoRA Benchmark Test
Purpose: Test one LoRA with one prompt to verify the benchmark approach
"""

import json
import requests
import time
import subprocess
from pathlib import Path

def test_single_lora_benchmark():
    """Test a single LoRA with one prompt"""
    print("🚀 Quick LoRA Benchmark Test")
    print("=" * 50)
    
    # Test configuration
    trellis_server_url = "http://localhost:8096"
    test_prompt = "greek amphora scene detail"
    lora_config = {
        'name': 'Flux Isometric 3D',
        'endpoint': '/generate/isometric_3d/',
        'trigger_prefix': 'Isometric 3D,'
    }
    
    # Test server health
    try:
        response = requests.get(f"{trellis_server_url}/health/", timeout=10)
        if response.status_code == 200:
            print("✅ Server is healthy")
        else:
            print(f"❌ Server health check failed: {response.status_code}")
            return
    except Exception as e:
        print(f"❌ Cannot connect to server: {e}")
        return
    
    # Switch to FLUX model
    try:
        response = requests.post(
            f"{trellis_server_url}/config/model/",
            data={'model': 'flux'},
            timeout=10
        )
        if response.status_code == 200:
            print("✅ Switched to FLUX model")
        else:
            print(f"❌ Failed to switch to FLUX: {response.status_code}")
            return
    except Exception as e:
        print(f"❌ Error switching to FLUX: {e}")
        return
    
    # Generate with LoRA
    print(f"\n🎨 Testing {lora_config['name']} with prompt: '{test_prompt}'")
    
    start_time = time.time()
    try:
        # Apply trigger prefix
        enhanced_prompt = f"{lora_config['trigger_prefix']} {test_prompt}"
        
        response = requests.post(
            f"{trellis_server_url}{lora_config['endpoint']}",
            data={
                'prompt': enhanced_prompt,
                'seed': 42,
                'return_compressed': True
            },
            timeout=300
        )
        
        generation_time = time.time() - start_time
        
        if response.status_code == 200:
            file_size = len(response.content)
            print(f"✅ Generation successful: {file_size:,} bytes in {generation_time:.2f}s")
            
            # Save the generated file
            output_dir = Path("./benchmark_outputs")
            output_dir.mkdir(exist_ok=True)
            
            filename = f"quick_test_{test_prompt.replace(' ', '_')}_{int(time.time())}.ply.spz"
            filepath = output_dir / filename
            
            with open(filepath, 'wb') as f:
                f.write(response.content)
            
            print(f"💾 Saved to: {filepath}")
            
            # Validate with subnet_accurate_validator
            print("\n🔍 Validating with subnet_accurate_validator...")
            
            cmd = [
                "bash", "-c",
                f"source /home/mbhat/miniconda/bin/activate && conda activate trellis_new && python subnet_accurate_validator.py \"{test_prompt}\" \"{enhanced_prompt}\""
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
            
            if result.returncode == 0:
                # Parse validation results
                with open("subnet_validation_results.json", 'r') as f:
                    data = json.load(f)
                    score = data.get("validation_engine_score", 0.0)
                    alignment_score = data.get("alignment_score", 0.0)
                    quality_score = data.get("quality_score", 0.0)
                    
                    print(f"📊 Validation Score: {score:.4f}")
                    print(f"📊 Alignment Score: {alignment_score:.4f}")
                    print(f"📊 Quality Score: {quality_score:.4f}")
                    
                    print(f"\n✅ Quick benchmark completed successfully!")
                    print(f"   LoRA: {lora_config['name']}")
                    print(f"   Prompt: '{test_prompt}'")
                    print(f"   Enhanced: '{enhanced_prompt}'")
                    print(f"   Validation Score: {score:.4f}")
                    print(f"   Generation Time: {generation_time:.2f}s")
                    print(f"   File Size: {file_size:,} bytes")
                    
            else:
                print(f"❌ Validation failed (return code {result.returncode})")
                print(f"   stderr: {result.stderr}")
                
        else:
            print(f"❌ Generation failed: HTTP {response.status_code}")
            
    except Exception as e:
        generation_time = time.time() - start_time
        print(f"❌ Generation error: {e}")

if __name__ == "__main__":
    test_single_lora_benchmark() 