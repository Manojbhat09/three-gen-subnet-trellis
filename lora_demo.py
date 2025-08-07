#!/usr/bin/env python3
"""
LoRA Demo Script
Purpose: Demonstrate how to use the LoRA endpoints for generation and testing
"""

import requests
import json
import time
from pathlib import Path

def test_flux_lora_endpoints():
    """Test FLUX LoRA endpoints"""
    base_url = "http://127.0.0.1:8096"
    
    print("🎨 Testing FLUX LoRA Endpoints")
    print("=" * 50)
    
    # Test prompts
    test_prompts = [
        "greek amphora scene detail",
        "plastic straw of drink"
    ]
    
    # FLUX LoRA endpoints
    flux_endpoints = [
        ("isometric_3d", "/generate/isometric_3d/"),
        ("live_3d", "/generate/live_3d/"),
        ("game_assets", "/generate/game_assets/"),
        ("patched_realism", "/generate/patched_realism/"),
        ("tf2_style", "/generate/tf2_style/"),
        ("baolei", "/generate/baolei/"),
        ("cartoon_3d", "/generate/cartoon_3d/"),
        ("cinema", "/generate/cinema/")
    ]
    
    for lora_name, endpoint in flux_endpoints:
        print(f"\n🔧 Testing {lora_name} LoRA...")
        
        for prompt in test_prompts:
            print(f"   Prompt: '{prompt}'")
            
            try:
                # Generate with LoRA
                response = requests.post(
                    f"{base_url}{endpoint}",
                    data={
                        'prompt': prompt,
                        'seed': 42,
                        'return_compressed': True
                    },
                    timeout=300
                )
                
                if response.status_code == 200:
                    print(f"   ✅ Generation successful! Size: {len(response.content):,} bytes")
                    
                    # Save the generated PLY file
                    output_dir = Path("./lora_demo_outputs")
                    output_dir.mkdir(exist_ok=True)
                    
                    filename = f"{lora_name}_{prompt.replace(' ', '_')}_{42}.ply.spz"
                    filepath = output_dir / filename
                    
                    with open(filepath, 'wb') as f:
                        f.write(response.content)
                    
                    print(f"   💾 Saved to: {filepath}")
                    
                else:
                    print(f"   ❌ Generation failed: {response.status_code}")
                    print(f"   Error: {response.text}")
                    
            except Exception as e:
                print(f"   ❌ Request failed: {e}")
            
            print()  # Empty line for readability

def test_sdxl_lora_endpoints():
    """Test SDXL LoRA endpoints"""
    base_url = "http://127.0.0.1:8097"
    
    print("🎨 Testing SDXL LoRA Endpoints")
    print("=" * 50)
    
    # Test prompts
    test_prompts = [
        "greek amphora scene detail",
        "plastic straw of drink"
    ]
    
    # SDXL LoRA endpoints
    sdxl_endpoints = [
        ("game_icon", "/generate/game_icon/")
    ]
    
    for lora_name, endpoint in sdxl_endpoints:
        print(f"\n🔧 Testing {lora_name} LoRA...")
        
        for prompt in test_prompts:
            print(f"   Prompt: '{prompt}'")
            
            try:
                # Generate with LoRA
                response = requests.post(
                    f"{base_url}{endpoint}",
                    data={
                        'prompt': prompt,
                        'seed': 42,
                        'return_compressed': True
                    },
                    timeout=300
                )
                
                if response.status_code == 200:
                    print(f"   ✅ Generation successful! Size: {len(response.content):,} bytes")
                    
                    # Save the generated PLY file
                    output_dir = Path("./lora_demo_outputs")
                    output_dir.mkdir(exist_ok=True)
                    
                    filename = f"sdxl_{lora_name}_{prompt.replace(' ', '_')}_{42}.ply.spz"
                    filepath = output_dir / filename
                    
                    with open(filepath, 'wb') as f:
                        f.write(response.content)
                    
                    print(f"   💾 Saved to: {filepath}")
                    
                else:
                    print(f"   ❌ Generation failed: {response.status_code}")
                    print(f"   Error: {response.text}")
                    
            except Exception as e:
                print(f"   ❌ Request failed: {e}")
            
            print()  # Empty line for readability

def test_lora_management():
    """Test LoRA management endpoints"""
    print("🔧 Testing LoRA Management Endpoints")
    print("=" * 50)
    
    # Test FLUX server
    flux_url = "http://127.0.0.1:8096"
    
    try:
        # Get available LoRAs
        response = requests.get(f"{flux_url}/loras/")
        if response.status_code == 200:
            loras = response.json()
            print("✅ FLUX Available LoRAs:")
            for key, lora in loras['loras'].items():
                print(f"   - {key}: {lora['name']} ({'loaded' if lora['loaded'] else 'not loaded'})")
        else:
            print(f"❌ Failed to get FLUX LoRAs: {response.status_code}")
    
    except Exception as e:
        print(f"❌ FLUX LoRA management test failed: {e}")
    
    print()
    
    # Test SDXL server
    sdxl_url = "http://127.0.0.1:8097"
    
    try:
        # Get available LoRAs
        response = requests.get(f"{sdxl_url}/loras/")
        if response.status_code == 200:
            loras = response.json()
            print("✅ SDXL Available LoRAs:")
            for key, lora in loras['loras'].items():
                print(f"   - {key}: {lora['name']} ({'loaded' if lora['loaded'] else 'not loaded'})")
        else:
            print(f"❌ Failed to get SDXL LoRAs: {response.status_code}")
    
    except Exception as e:
        print(f"❌ SDXL LoRA management test failed: {e}")

def main():
    """Main function"""
    print("🚀 LoRA Demo Script")
    print("=" * 80)
    
    # Test LoRA management first
    test_lora_management()
    
    print("\n" + "=" * 80)
    
    # Test FLUX LoRA endpoints
    test_flux_lora_endpoints()
    
    print("\n" + "=" * 80)
    
    # Test SDXL LoRA endpoints
    test_sdxl_lora_endpoints()
    
    print("\n✅ Demo completed!")
    print("📁 Check './lora_demo_outputs/' for generated files")

if __name__ == "__main__":
    main() 