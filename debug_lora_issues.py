#!/usr/bin/env python3
"""
Debug LoRA Issues
Purpose: Test specific LoRAs and identify problems
"""

import requests
import json
import time

def test_server_health():
    """Test if server is responding"""
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

def test_lora_loading():
    """Test LoRA loading for each model"""
    print("\n🔍 Testing LoRA Loading")
    print("=" * 50)
    
    # Test FLUX LoRAs
    print("🎨 Testing FLUX LoRAs...")
    
    # Switch to FLUX
    try:
        response = requests.post("http://127.0.0.1:8096/config/model/", data={'model': 'flux'}, timeout=10)
        if response.status_code == 200:
            print("   ✅ Switched to FLUX")
        else:
            print("   ❌ Failed to switch to FLUX")
            return
    except Exception as e:
        print(f"   ❌ Error switching to FLUX: {e}")
        return
    
    # Get available FLUX LoRAs
    try:
        response = requests.get("http://127.0.0.1:8096/loras/", timeout=10)
        if response.status_code == 200:
            loras = response.json()
            flux_loras = list(loras['loras'].keys())
            print(f"   📋 Available FLUX LoRAs: {flux_loras}")
            
            # Test each LoRA
            for lora in flux_loras:
                print(f"   🎨 Testing {lora}...")
                try:
                    response = requests.post(f"http://127.0.0.1:8096/loras/load/{lora}", timeout=30)
                    if response.status_code == 200:
                        print(f"      ✅ {lora} loaded successfully")
                    else:
                        print(f"      ❌ {lora} failed: {response.status_code}")
                        try:
                            error_data = response.json()
                            print(f"         Error: {error_data.get('detail', 'Unknown error')}")
                        except:
                            print(f"         Error: {response.text}")
                except Exception as e:
                    print(f"      ❌ {lora} error: {e}")
        else:
            print(f"   ❌ Failed to get LoRAs: {response.status_code}")
    except Exception as e:
        print(f"   ❌ Error getting LoRAs: {e}")
    
    # Test SD1.5 LoRAs
    print("\n🎨 Testing SD1.5 LoRAs...")
    
    # Switch to SD1.5
    try:
        response = requests.post("http://127.0.0.1:8096/config/model/", data={'model': 'sd15'}, timeout=10)
        if response.status_code == 200:
            print("   ✅ Switched to SD1.5")
        else:
            print("   ❌ Failed to switch to SD1.5")
            return
    except Exception as e:
        print(f"   ❌ Error switching to SD1.5: {e}")
        return
    
    # Get available SD1.5 LoRAs
    try:
        response = requests.get("http://127.0.0.1:8096/loras/", timeout=10)
        if response.status_code == 200:
            loras = response.json()
            sd15_loras = list(loras['loras'].keys())
            print(f"   📋 Available SD1.5 LoRAs: {sd15_loras}")
            
            # Test each LoRA
            for lora in sd15_loras:
                print(f"   🎨 Testing {lora}...")
                try:
                    response = requests.post(f"http://127.0.0.1:8096/loras/load/{lora}", timeout=30)
                    if response.status_code == 200:
                        print(f"      ✅ {lora} loaded successfully")
                    else:
                        print(f"      ❌ {lora} failed: {response.status_code}")
                        try:
                            error_data = response.json()
                            print(f"         Error: {error_data.get('detail', 'Unknown error')}")
                        except:
                            print(f"         Error: {response.text}")
                except Exception as e:
                    print(f"      ❌ {lora} error: {e}")
        else:
            print(f"   ❌ Failed to get LoRAs: {response.status_code}")
    except Exception as e:
        print(f"   ❌ Error getting LoRAs: {e}")

def test_endpoints():
    """Test specific endpoints"""
    print("\n🔗 Testing Endpoints")
    print("=" * 50)
    
    # Test FLUX endpoints
    flux_endpoints = [
        '/generate/isometric_3d/',
        '/generate/live_3d/',
        '/generate/game_assets/',
        '/generate/patched_realism/',
        '/generate/tf2_style/',
        '/generate/baolei/',
        '/generate/cartoon_3d/',
        '/generate/cinema/'
    ]
    
    print("🎨 Testing FLUX LoRA endpoints...")
    for endpoint in flux_endpoints:
        try:
            response = requests.post(
                f"http://127.0.0.1:8096{endpoint}",
                data={'prompt': 'test', 'seed': 42, 'return_compressed': True},
                timeout=5
            )
            if response.status_code == 200:
                print(f"   ✅ {endpoint} - OK")
            elif response.status_code == 404:
                print(f"   ❌ {endpoint} - Not Found")
            else:
                print(f"   ⚠️ {endpoint} - {response.status_code}")
        except Exception as e:
            print(f"   ❌ {endpoint} - Error: {e}")
    
    # Test SD1.5 endpoints
    sd15_endpoints = [
        '/generate/sd15_game_icon/'
    ]
    
    print("\n🎨 Testing SD1.5 LoRA endpoints...")
    for endpoint in sd15_endpoints:
        try:
            response = requests.post(
                f"http://127.0.0.1:8096{endpoint}",
                data={'prompt': 'test', 'seed': 42, 'return_compressed': True},
                timeout=5
            )
            if response.status_code == 200:
                print(f"   ✅ {endpoint} - OK")
            elif response.status_code == 404:
                print(f"   ❌ {endpoint} - Not Found")
            else:
                print(f"   ⚠️ {endpoint} - {response.status_code}")
        except Exception as e:
            print(f"   ❌ {endpoint} - Error: {e}")

def main():
    """Main debug function"""
    print("🚀 Debug LoRA Issues")
    print("=" * 80)
    
    if not test_server_health():
        print("❌ Server is not available. Please start the server first.")
        return
    
    test_lora_loading()
    test_endpoints()
    
    print("\n" + "=" * 80)
    print("✅ Debug completed!")

if __name__ == "__main__":
    main() 