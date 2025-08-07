#!/usr/bin/env python3
"""
Comprehensive LoRA System Test
Purpose: Test all models and their respective LoRAs properly
"""

import requests
import json
import time
from pathlib import Path

def test_server_health():
    """Test if the server is running"""
    print("🔍 Testing server health...")
    
    try:
        response = requests.get("http://127.0.0.1:8096/health/", timeout=10)
        if response.status_code == 200:
            print("✅ Server is healthy")
            return True
        else:
            print(f"❌ Server health check failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Cannot connect to server: {e}")
        return False

def test_model_and_loras():
    """Test each model with its respective LoRAs"""
    print("\n🎨 Testing Models and LoRAs")
    print("=" * 50)
    
    # Test configurations
    test_configs = [
        {
            'model': 'flux',
            'loras': ['isometric_3d', 'live_3d', 'game_assets', 'patched_realism', 'tf2_style', 'baolei', 'cartoon_3d', 'cinema'],
            'test_prompt': 'a blue ceramic vase'
        },
        {
            'model': 'sd15',
            'loras': ['game_icon'],
            'test_prompt': 'a blue ceramic vase'
        }
    ]
    
    results = []
    
    for config in test_configs:
        model = config['model']
        loras = config['loras']
        test_prompt = config['test_prompt']
        
        print(f"\n🔧 Testing {model.upper()} model...")
        
        # Switch to model
        try:
            response = requests.post(
                "http://127.0.0.1:8096/config/model/",
                data={'model': model},
                timeout=10
            )
            
            if response.status_code == 200:
                print(f"   ✅ Switched to {model.upper()}")
            else:
                print(f"   ❌ Failed to switch to {model.upper()}")
                continue
                
        except Exception as e:
            print(f"   ❌ Error switching to {model.upper()}: {e}")
            continue
        
        # Load model pipeline by doing a basic generation
        print(f"   🔄 Loading {model.upper()} pipeline...")
        try:
            response = requests.post(
                "http://127.0.0.1:8096/generate/",
                data={
                    'prompt': test_prompt,
                    'seed': 42,
                    'return_compressed': True
                },
                timeout=300
            )
            
            if response.status_code == 200:
                print(f"   ✅ {model.upper()} pipeline loaded successfully")
            else:
                print(f"   ❌ Failed to load {model.upper()} pipeline")
                continue
                
        except Exception as e:
            print(f"   ❌ Error loading {model.upper()} pipeline: {e}")
            continue
        
        # Test each LoRA
        for lora in loras:
            print(f"   🎨 Testing {model.upper()} LoRA: {lora}")
            
            try:
                # Load LoRA
                response = requests.post(
                    f"http://127.0.0.1:8096/loras/load/{lora}",
                    timeout=30
                )
                
                if response.status_code == 200:
                    print(f"      ✅ LoRA loaded successfully")
                    
                    # Test generation with LoRA
                    if model == 'flux':
                        endpoint = f"/generate/{lora}/"
                    elif model == 'sd15':
                        endpoint = f"/generate/sd15_{lora}/"
                    else:
                        endpoint = f"/generate/{lora}/"
                    
                    response = requests.post(
                        f"http://127.0.0.1:8096{endpoint}",
                        data={
                            'prompt': test_prompt,
                            'seed': 42,
                            'return_compressed': True
                        },
                        timeout=300
                    )
                    
                    if response.status_code == 200:
                        file_size = len(response.content)
                        print(f"      ✅ LoRA generation successful! Size: {file_size:,} bytes")
                        
                        # Save the result
                        output_dir = Path("./test_outputs")
                        output_dir.mkdir(exist_ok=True)
                        
                        filename = f"{model}_{lora}_{int(time.time())}.ply.spz"
                        filepath = output_dir / filename
                        
                        with open(filepath, 'wb') as f:
                            f.write(response.content)
                        
                        print(f"      💾 Saved to: {filepath}")
                        
                        results.append({
                            'model': model,
                            'lora': lora,
                            'success': True,
                            'file_size': file_size,
                            'filepath': str(filepath)
                        })
                        
                    else:
                        print(f"      ❌ LoRA generation failed: {response.status_code}")
                        results.append({
                            'model': model,
                            'lora': lora,
                            'success': False,
                            'error': f"HTTP {response.status_code}"
                        })
                        
                else:
                    print(f"      ❌ Failed to load LoRA: {response.status_code}")
                    results.append({
                        'model': model,
                        'lora': lora,
                        'success': False,
                        'error': f"LoRA load failed: HTTP {response.status_code}"
                    })
                    
            except Exception as e:
                print(f"      ❌ Error testing LoRA: {e}")
                results.append({
                    'model': model,
                    'lora': lora,
                    'success': False,
                    'error': str(e)
                })
    
    return results

def test_validation_prompts():
    """Test with the validation prompts"""
    print("\n🎯 Testing with Validation Prompts")
    print("=" * 50)
    
    validation_prompts = [
        "greek amphora scene detail",
        "plastic straw of drink",
        "small yellow triangular wooden kitchen knife",
        "enormous black robot with round body",
        "rose gold locket necklace with floral"
    ]
    
    # Test with FLUX model and isometric_3d LoRA
    print("🔧 Testing FLUX + isometric_3d LoRA with validation prompts...")
    
    # Switch to FLUX
    try:
        response = requests.post(
            "http://127.0.0.1:8096/config/model/",
            data={'model': 'flux'},
            timeout=10
        )
        
        if response.status_code == 200:
            print("   ✅ Switched to FLUX")
        else:
            print("   ❌ Failed to switch to FLUX")
            return
    except Exception as e:
        print(f"   ❌ Error switching to FLUX: {e}")
        return
    
    # Load FLUX pipeline
    try:
        response = requests.post(
            "http://127.0.0.1:8096/generate/",
            data={
                'prompt': 'test',
                'seed': 42,
                'return_compressed': True
            },
            timeout=300
        )
        
        if response.status_code == 200:
            print("   ✅ FLUX pipeline loaded")
        else:
            print("   ❌ Failed to load FLUX pipeline")
            return
    except Exception as e:
        print(f"   ❌ Error loading FLUX pipeline: {e}")
        return
    
    # Load isometric_3d LoRA
    try:
        response = requests.post(
            "http://127.0.0.1:8096/loras/load/isometric_3d",
            timeout=30
        )
        
        if response.status_code == 200:
            print("   ✅ isometric_3d LoRA loaded")
        else:
            print("   ❌ Failed to load isometric_3d LoRA")
            return
    except Exception as e:
        print(f"   ❌ Error loading isometric_3d LoRA: {e}")
        return
    
    # Test each validation prompt
    for i, prompt in enumerate(validation_prompts):
        print(f"   🎨 Testing prompt {i+1}: '{prompt}'")
        
        try:
            response = requests.post(
                "http://127.0.0.1:8096/generate/isometric_3d/",
                data={
                    'prompt': prompt,
                    'seed': 42,
                    'return_compressed': True
                },
                timeout=300
            )
            
            if response.status_code == 200:
                file_size = len(response.content)
                print(f"      ✅ Generation successful! Size: {file_size:,} bytes")
                
                # Save the result
                output_dir = Path("./test_outputs")
                output_dir.mkdir(exist_ok=True)
                
                filename = f"validation_prompt_{i+1}_{prompt.replace(' ', '_')}_{int(time.time())}.ply.spz"
                filepath = output_dir / filename
                
                with open(filepath, 'wb') as f:
                    f.write(response.content)
                
                print(f"      💾 Saved to: {filepath}")
                
            else:
                print(f"      ❌ Generation failed: {response.status_code}")
                
        except Exception as e:
            print(f"      ❌ Error: {e}")

def print_summary(results):
    """Print test summary"""
    print("\n📊 Test Summary")
    print("=" * 50)
    
    total_tests = len(results)
    successful_tests = len([r for r in results if r['success']])
    failed_tests = total_tests - successful_tests
    
    print(f"Total Tests: {total_tests}")
    print(f"Successful: {successful_tests}")
    print(f"Failed: {failed_tests}")
    print(f"Success Rate: {(successful_tests/total_tests*100):.1f}%")
    
    if successful_tests > 0:
        print("\n✅ Successful Tests:")
        for result in results:
            if result['success']:
                print(f"   - {result['model'].upper()} + {result['lora']}: {result['file_size']:,} bytes")
    
    if failed_tests > 0:
        print("\n❌ Failed Tests:")
        for result in results:
            if not result['success']:
                print(f"   - {result['model'].upper()} + {result['lora']}: {result['error']}")

def main():
    """Main test function"""
    print("🚀 Comprehensive LoRA System Test")
    print("=" * 80)
    
    # Test server health
    if not test_server_health():
        print("❌ Server is not available. Please start the server first.")
        return
    
    # Test models and LoRAs
    results = test_model_and_loras()
    
    # Test validation prompts
    test_validation_prompts()
    
    # Print summary
    print_summary(results)
    
    print("\n" + "=" * 80)
    print("✅ Comprehensive LoRA system test completed!")
    print("📁 Check './test_outputs/' for generated files")

if __name__ == "__main__":
    main() 