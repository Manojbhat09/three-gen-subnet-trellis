#!/usr/bin/env python3
"""
Test LoRA System
Purpose: Test the LoRA system to identify any issues
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

def test_model_switching():
    """Test model switching functionality"""
    print("\n🔄 Testing model switching...")
    
    models = ['flux', 'sdxl', 'sd15']
    
    for model in models:
        try:
            print(f"   Switching to {model.upper()}...")
            response = requests.post(
                "http://127.0.0.1:8096/config/model/",
                data={'model': model},
                timeout=10
            )
            
            if response.status_code == 200:
                result = response.json()
                print(f"   ✅ Switched to {model.upper()}: {result['message']}")
            else:
                print(f"   ❌ Failed to switch to {model.upper()}: {response.status_code}")
                
        except Exception as e:
            print(f"   ❌ Error switching to {model.upper()}: {e}")

def test_lora_management():
    """Test LoRA management endpoints"""
    print("\n🎨 Testing LoRA management...")
    
    try:
        # Get available LoRAs
        print("   Getting available LoRAs...")
        response = requests.get("http://127.0.0.1:8096/loras/", timeout=10)
        
        if response.status_code == 200:
            loras = response.json()
            print(f"   ✅ Available LoRAs: {list(loras['loras'].keys())}")
            
            # Try to load a LoRA if available
            if loras['loras']:
                first_lora = list(loras['loras'].keys())[0]
                print(f"   Loading LoRA: {first_lora}")
                
                load_response = requests.post(
                    f"http://127.0.0.1:8096/loras/load/{first_lora}",
                    timeout=30
                )
                
                if load_response.status_code == 200:
                    print(f"   ✅ LoRA loaded successfully")
                else:
                    print(f"   ❌ Failed to load LoRA: {load_response.status_code}")
                    
        else:
            print(f"   ❌ Failed to get LoRAs: {response.status_code}")
            
    except Exception as e:
        print(f"   ❌ Error in LoRA management: {e}")

def test_generation():
    """Test basic generation"""
    print("\n🎯 Testing generation...")
    
    test_prompt = "a simple blue vase"
    
    try:
        print(f"   Generating: '{test_prompt}'")
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
            print(f"   ✅ Generation successful! Size: {len(response.content):,} bytes")
            
            # Save the result
            output_dir = Path("./test_outputs")
            output_dir.mkdir(exist_ok=True)
            
            filename = f"test_generation_{int(time.time())}.ply.spz"
            filepath = output_dir / filename
            
            with open(filepath, 'wb') as f:
                f.write(response.content)
            
            print(f"   💾 Saved to: {filepath}")
            return True
        else:
            print(f"   ❌ Generation failed: {response.status_code}")
            print(f"   Error: {response.text}")
            return False
            
    except Exception as e:
        print(f"   ❌ Generation error: {e}")
        return False

def test_lora_generation():
    """Test LoRA-specific generation"""
    print("\n🎨 Testing LoRA generation...")
    
    # Test with isometric_3d LoRA
    test_prompt = "a simple blue vase"
    
    try:
        print(f"   Generating with isometric_3d LoRA: '{test_prompt}'")
        response = requests.post(
            "http://127.0.0.1:8096/generate/isometric_3d/",
            data={
                'prompt': test_prompt,
                'seed': 42,
                'return_compressed': True
            },
            timeout=300
        )
        
        if response.status_code == 200:
            print(f"   ✅ LoRA generation successful! Size: {len(response.content):,} bytes")
            
            # Save the result
            output_dir = Path("./test_outputs")
            output_dir.mkdir(exist_ok=True)
            
            filename = f"lora_generation_{int(time.time())}.ply.spz"
            filepath = output_dir / filename
            
            with open(filepath, 'wb') as f:
                f.write(response.content)
            
            print(f"   💾 Saved to: {filepath}")
            return True
        else:
            print(f"   ❌ LoRA generation failed: {response.status_code}")
            print(f"   Error: {response.text}")
            return False
            
    except Exception as e:
        print(f"   ❌ LoRA generation error: {e}")
        return False

def test_server_status():
    """Test server status endpoint"""
    print("\n📊 Testing server status...")
    
    try:
        response = requests.get("http://127.0.0.1:8096/status/", timeout=10)
        
        if response.status_code == 200:
            status = response.json()
            print(f"   ✅ Server status: {status['status']}")
            print(f"   📈 Total generations: {status['metrics']['total_generations']}")
            print(f"   ✅ Successful generations: {status['metrics']['successful_generations']}")
            print(f"   ❌ Failed generations: {status['metrics']['failed_generations']}")
            
            if status['metrics']['total_generations'] > 0:
                success_rate = (status['metrics']['successful_generations'] / status['metrics']['total_generations']) * 100
                print(f"   📊 Success rate: {success_rate:.1f}%")
            
            return True
        else:
            print(f"   ❌ Failed to get status: {response.status_code}")
            return False
            
    except Exception as e:
        print(f"   ❌ Status error: {e}")
        return False

def main():
    """Main test function"""
    print("🚀 LoRA System Test")
    print("=" * 50)
    
    # Test server health
    if not test_server_health():
        print("❌ Server is not available. Please start the server first.")
        return
    
    # Test model switching
    test_model_switching()
    
    # Test LoRA management
    test_lora_management()
    
    # Test basic generation
    if test_generation():
        print("✅ Basic generation test passed")
    else:
        print("❌ Basic generation test failed")
    
    # Test LoRA generation
    if test_lora_generation():
        print("✅ LoRA generation test passed")
    else:
        print("❌ LoRA generation test failed")
    
    # Test server status
    test_server_status()
    
    print("\n" + "=" * 50)
    print("✅ LoRA system test completed!")
    print("📁 Check './test_outputs/' for generated files")

if __name__ == "__main__":
    main() 