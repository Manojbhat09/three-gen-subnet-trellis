#!/usr/bin/env python3
"""
Quick LoRA Test
Purpose: Quick test of LoRA system
"""

import requests
import time

def test_basic_functionality():
    """Test basic server functionality"""
    print("🚀 Quick LoRA Test")
    print("=" * 50)
    
    # Test server health
    try:
        response = requests.get("http://127.0.0.1:8096/health/", timeout=200)
        if response.status_code == 200:
            print("✅ Server is healthy")
        else:
            print(f"❌ Server health check failed: {response.status_code}")
            return
    except Exception as e:
        print(f"❌ Cannot connect to server: {e}")
        return
    
    # Test FLUX model switching
    try:
        response = requests.post("http://127.0.0.1:8096/config/model/", data={'model': 'flux'}, timeout=200)
        if response.status_code == 200:
            print("✅ Switched to FLUX model")
        else:
            print(f"❌ Failed to switch to FLUX: {response.status_code}")
            return
    except Exception as e:
        print(f"❌ Error switching to FLUX: {e}")
        return
    
    # Test basic generation
    try:
        response = requests.post(
            "http://127.0.0.1:8096/generate/",
            data={'prompt': 'test', 'seed': 42, 'return_compressed': True},
            timeout=200
        )
        if response.status_code == 200:
            print("✅ Basic generation successful")
        else:
            print(f"❌ Basic generation failed: {response.status_code}")
            return
    except Exception as e:
        print(f"❌ Error in basic generation: {e}")
        return
    
    # Test LoRA loading
    try:
        response = requests.post("http://127.0.0.1:8096/loras/load/isometric_3d", timeout=200)
        if response.status_code == 200:
            print("✅ isometric_3d LoRA loaded successfully")
        else:
            print(f"❌ isometric_3d LoRA failed: {response.status_code}")
            return
    except Exception as e:
        print(f"❌ Error loading isometric_3d LoRA: {e}")
        return
    
    # Test LoRA generation
    try:
        response = requests.post(
            "http://127.0.0.1:8096/generate/isometric_3d/",
            data={'prompt': 'a blue ceramic vase', 'seed': 42, 'return_compressed': True},
            timeout=120
        )
        if response.status_code == 200:
            file_size = len(response.content)
            print(f"✅ LoRA generation successful! Size: {file_size:,} bytes")
        else:
            print(f"❌ LoRA generation failed: {response.status_code}")
            return
    except Exception as e:
        print(f"❌ Error in LoRA generation: {e}")
        return
    
    print("\n" + "=" * 50)
    print("✅ Quick LoRA test completed successfully!")

if __name__ == "__main__":
    test_basic_functionality() 