#!/usr/bin/env python3
"""
Simple LoRA Test
Purpose: Test the LoRA system with available LoRAs
"""

import requests
import json
import time
from pathlib import Path

def test_available_loras():
    """Test what LoRAs are available"""
    print("🔍 Testing available LoRAs...")
    
    try:
        # Test server health
        response = requests.get("http://127.0.0.1:8096/health/", timeout=10)
        if response.status_code != 200:
            print("❌ Server not responding")
            return
        
        print("✅ Server is healthy")
        
        # Get available LoRAs for FLUX
        response = requests.post("http://127.0.0.1:8096/config/model/", data={'model': 'flux'}, timeout=10)
        if response.status_code == 200:
            print("✅ Switched to FLUX model")
        
        response = requests.get("http://127.0.0.1:8096/loras/", timeout=10)
        if response.status_code == 200:
            loras = response.json()
            print(f"✅ Available FLUX LoRAs: {list(loras['loras'].keys())}")
            
            # Test each LoRA
            for lora_key in loras['loras'].keys():
                print(f"   Testing {lora_key}...")
                try:
                    response = requests.post(f"http://127.0.0.1:8096/loras/load/{lora_key}", timeout=30)
                    if response.status_code == 200:
                        print(f"      ✅ {lora_key} loaded successfully")
                    else:
                        print(f"      ❌ {lora_key} failed to load: {response.status_code}")
                except Exception as e:
                    print(f"      ❌ {lora_key} error: {e}")
        
        # Get available LoRAs for SD1.5
        response = requests.post("http://127.0.0.1:8096/config/model/", data={'model': 'sd15'}, timeout=10)
        if response.status_code == 200:
            print("✅ Switched to SD1.5 model")
        
        response = requests.get("http://127.0.0.1:8096/loras/", timeout=10)
        if response.status_code == 200:
            loras = response.json()
            print(f"✅ Available SD1.5 LoRAs: {list(loras['loras'].keys())}")
            
    except Exception as e:
        print(f"❌ Error: {e}")

def main():
    """Main test function"""
    print("🚀 Simple LoRA Test")
    print("=" * 50)
    
    test_available_loras()
    
    print("\n" + "=" * 50)
    print("✅ Simple LoRA test completed!")

if __name__ == "__main__":
    main() 