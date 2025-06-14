#!/usr/bin/env python3
"""
Automated GPU Coordination Test
Tests the complete automated GPU memory coordination between validation and generation servers
"""

import requests
import time
import json
import subprocess
import signal
import os
import sys
from pathlib import Path

# Server configurations
VALIDATION_SERVER_URL = "http://localhost:10006"
GENERATION_SERVER_URL = "http://localhost:8095"

def check_server_health(url, name):
    """Check if server is running and healthy"""
    try:
        response = requests.get(f"{url}/health/", timeout=5)
        if response.status_code == 200:
            print(f"✅ {name} server is healthy")
            return True
        else:
            print(f"❌ {name} server returned status {response.status_code}")
            return False
    except requests.exceptions.RequestException as e:
        print(f"❌ {name} server is not responding: {e}")
        return False

def get_gpu_status(url, name):
    """Get GPU memory status from server"""
    try:
        response = requests.get(f"{url}/gpu_status/", timeout=10)
        if response.status_code == 200:
            status = response.json()
            print(f"🧠 {name} GPU Status:")
            print(f"   Used: {status.get('memory_used_gb', 0):.1f}GB")
            print(f"   Free: {status.get('memory_free_gb', 0):.1f}GB")
            print(f"   Total: {status.get('memory_total_gb', 0):.1f}GB")
            print(f"   Usage: {status.get('memory_usage_percent', 0):.1f}%")
            return status
        else:
            print(f"❌ Failed to get {name} GPU status: {response.status_code}")
            return None
    except requests.exceptions.RequestException as e:
        print(f"❌ Failed to get {name} GPU status: {e}")
        return None

def test_validation_unload_reload():
    """Test validation server unload/reload functionality"""
    print("\n🧪 Testing validation server unload/reload...")
    
    # Get initial status
    print("📊 Initial GPU status:")
    initial_status = get_gpu_status(VALIDATION_SERVER_URL, "Validation")
    if not initial_status:
        return False
    
    # Test unload
    print("\n🔄 Testing model unload...")
    try:
        response = requests.post(f"{VALIDATION_SERVER_URL}/unload_models/", timeout=30)
        if response.status_code == 200:
            unload_result = response.json()
            print(f"✅ Unload successful: freed {unload_result.get('memory_freed_gb', 0):.2f}GB")
            
            # Check GPU status after unload
            print("📊 GPU status after unload:")
            unload_status = get_gpu_status(VALIDATION_SERVER_URL, "Validation")
            
        else:
            print(f"❌ Unload failed: {response.status_code}")
            return False
    except requests.exceptions.RequestException as e:
        print(f"❌ Unload request failed: {e}")
        return False
    
    # Wait a moment
    time.sleep(2)
    
    # Test reload
    print("\n🔄 Testing model reload...")
    try:
        response = requests.post(f"{VALIDATION_SERVER_URL}/reload_models/", timeout=60)
        if response.status_code == 200:
            reload_result = response.json()
            print(f"✅ Reload successful in {reload_result.get('reload_time', 0):.2f}s")
            
            # Check GPU status after reload
            print("📊 GPU status after reload:")
            reload_status = get_gpu_status(VALIDATION_SERVER_URL, "Validation")
            
        else:
            print(f"❌ Reload failed: {response.status_code}")
            print(f"Response: {response.text}")
            return False
    except requests.exceptions.RequestException as e:
        print(f"❌ Reload request failed: {e}")
        return False
    
    return True

def test_generation_coordination():
    """Test generation server with automatic validation coordination"""
    print("\n🧪 Testing automated generation with GPU coordination...")
    
    # Get initial GPU status from both servers
    print("📊 Initial GPU status:")
    val_status = get_gpu_status(VALIDATION_SERVER_URL, "Validation")
    gen_status = get_gpu_status(GENERATION_SERVER_URL, "Generation")
    
    if not val_status or not gen_status:
        print("❌ Cannot get initial GPU status from both servers")
        return False
    
    # Test generation with coordination
    test_prompt = "a red cube"
    print(f"\n🎯 Starting automated generation: '{test_prompt}'")
    
    try:
        # Make generation request
        data = {
            "prompt": test_prompt,
            "seed": 42,
            "return_compressed": True
        }
        
        print("⏳ Sending generation request (this will take time)...")
        response = requests.post(
            f"{GENERATION_SERVER_URL}/generate/", 
            data=data,
            timeout=300  # 5 minutes timeout
        )
        
        if response.status_code == 200:
            result = response.json()
            print("✅ Generation completed successfully!")
            print(f"   Generation time: {result.get('generation_time', 0):.2f}s")
            print(f"   PLY size: {result.get('ply_size_bytes', 0)} bytes")
            
            # Check final GPU status
            print("\n📊 Final GPU status:")
            final_val_status = get_gpu_status(VALIDATION_SERVER_URL, "Validation")
            final_gen_status = get_gpu_status(GENERATION_SERVER_URL, "Generation")
            
            return True
            
        else:
            print(f"❌ Generation failed: {response.status_code}")
            print(f"Response: {response.text}")
            return False
            
    except requests.exceptions.RequestException as e:
        print(f"❌ Generation request failed: {e}")
        return False

def test_multiple_generations():
    """Test multiple generations to ensure coordination is stable"""
    print("\n🧪 Testing multiple generations for stability...")
    
    prompts = [
        "a blue sphere",
        "a green pyramid", 
        "a yellow cylinder"
    ]
    
    success_count = 0
    
    for i, prompt in enumerate(prompts, 1):
        print(f"\n🎯 Generation {i}/{len(prompts)}: '{prompt}'")
        
        try:
            data = {
                "prompt": prompt,
                "seed": 42 + i,
                "return_compressed": True
            }
            
            response = requests.post(
                f"{GENERATION_SERVER_URL}/generate/", 
                data=data,
                timeout=300
            )
            
            if response.status_code == 200:
                result = response.json()
                print(f"✅ Generation {i} successful in {result.get('generation_time', 0):.2f}s")
                success_count += 1
            else:
                print(f"❌ Generation {i} failed: {response.status_code}")
                
        except requests.exceptions.RequestException as e:
            print(f"❌ Generation {i} request failed: {e}")
        
        # Small delay between generations
        if i < len(prompts):
            print("⏳ Waiting before next generation...")
            time.sleep(5)
    
    print(f"\n📊 Multiple generation results: {success_count}/{len(prompts)} successful")
    return success_count == len(prompts)

def main():
    """Main test function"""
    print("🚀 Starting Automated GPU Coordination Test")
    print("=" * 60)
    
    # Check if both servers are running
    print("🔍 Checking server health...")
    val_healthy = check_server_health(VALIDATION_SERVER_URL, "Validation")
    gen_healthy = check_server_health(GENERATION_SERVER_URL, "Generation")
    
    if not val_healthy or not gen_healthy:
        print("\n❌ One or both servers are not running!")
        print("Please start both servers before running this test:")
        print("1. Validation server: cd validation && python serve.py --host 0.0.0.0 --port 10006")
        print("2. Generation server: python flux_hunyuan_sugar_generation_server.py")
        return False
    
    print("\n✅ Both servers are healthy!")
    
    # Test 1: Validation server unload/reload
    test1_success = test_validation_unload_reload()
    
    # Test 2: Single generation with coordination
    test2_success = test_generation_coordination()
    
    # Test 3: Multiple generations for stability
    test3_success = test_multiple_generations()
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 TEST SUMMARY")
    print("=" * 60)
    print(f"✅ Validation unload/reload: {'PASS' if test1_success else 'FAIL'}")
    print(f"✅ Single generation coordination: {'PASS' if test2_success else 'FAIL'}")
    print(f"✅ Multiple generation stability: {'PASS' if test3_success else 'FAIL'}")
    
    all_passed = test1_success and test2_success and test3_success
    
    if all_passed:
        print("\n🎉 ALL TESTS PASSED! GPU coordination is working properly.")
        print("The automated system successfully manages GPU memory between servers.")
    else:
        print("\n❌ SOME TESTS FAILED! Check the logs above for details.")
        print("The GPU coordination system needs debugging.")
    
    return all_passed

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 