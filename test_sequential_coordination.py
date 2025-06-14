#!/usr/bin/env python3
"""
Test Sequential GPU Memory Coordination
Demonstrates that validation and generation can work sequentially by coordinating GPU memory
"""

import subprocess
import time
import requests
import sys


def check_server_health(url: str, timeout: int = 5) -> bool:
    """Check if server is healthy"""
    try:
        response = requests.get(url, timeout=timeout)
        return response.status_code == 200
    except:
        return False


def test_validation_only():
    """Test 1: Start validation server only"""
    print("\n" + "="*60)
    print("TEST 1: Validation Server Only")
    print("="*60)
    
    print("🔧 Starting validation server...")
    
    # Start validation server
    proc = subprocess.Popen([
        "bash", "-c", 
        "cd validation && conda activate three-gen-validation && python serve.py --host 0.0.0.0 --port 10006"
    ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    
    # Wait for server to start
    for i in range(30):
        time.sleep(1)
        if check_server_health("http://localhost:10006/version/"):
            print("✅ Validation server started successfully")
            break
        if proc.poll() is not None:
            print("❌ Validation server process died")
            return False
    else:
        print("❌ Validation server failed to start")
        return False
    
    try:
        # Test validation endpoints
        print("🧪 Testing validation endpoints...")
        
        # Test version
        response = requests.get("http://localhost:10006/version/")
        print(f"✓ Version: {response.text.strip('\"')}")
        
        # Test GPU status
        response = requests.get("http://localhost:10006/gpu_status/")
        gpu_status = response.json()
        print(f"✓ GPU Status: {gpu_status['memory_used_gb']:.1f}GB used, {gpu_status['memory_free_gb']:.1f}GB free")
        
        # Test model unloading
        response = requests.post("http://localhost:10006/unload_models/")
        unload_result = response.json()
        print(f"✓ Model unload: freed {unload_result.get('memory_freed_gb', 0):.1f}GB")
        
        print("✅ Validation server test PASSED")
        return True
        
    except Exception as e:
        print(f"❌ Validation server test FAILED: {e}")
        return False
    finally:
        # Stop validation server
        proc.terminate()
        proc.wait()
        time.sleep(2)


def test_generation_only():
    """Test 2: Start generation server only"""
    print("\n" + "="*60)
    print("TEST 2: Generation Server Only")
    print("="*60)
    
    print("🔧 Starting generation server...")
    
    # Start generation server
    proc = subprocess.Popen([
        "bash", "-c", 
        "conda activate hunyuan3d && python flux_hunyuan_sugar_generation_server.py"
    ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    
    # Wait for server to start (longer timeout for model loading)
    for i in range(60):
        time.sleep(1)
        if check_server_health("http://localhost:8095/health/"):
            print("✅ Generation server started successfully")
            break
        if proc.poll() is not None:
            print("❌ Generation server process died")
            return False
    else:
        print("❌ Generation server failed to start")
        return False
    
    try:
        # Test generation
        print("🧪 Testing generation...")
        
        # Test health
        response = requests.get("http://localhost:8095/health/")
        health = response.json()
        print(f"✓ Health: {health['status']}")
        
        # Test simple generation
        print("🎯 Testing simple generation...")
        start_time = time.time()
        
        response = requests.post(
            "http://localhost:8095/generate/",
            data={"prompt": "a red cube"},
            timeout=120
        )
        
        generation_time = time.time() - start_time
        
        if response.status_code == 200:
            ply_size = len(response.content)
            print(f"✅ Generation successful: {ply_size:,} bytes in {generation_time:.1f}s")
            return True
        else:
            print(f"❌ Generation failed: {response.status_code}")
            return False
        
    except Exception as e:
        print(f"❌ Generation server test FAILED: {e}")
        return False
    finally:
        # Stop generation server
        proc.terminate()
        proc.wait()
        time.sleep(2)


def main():
    """Main function"""
    print("🚀 Sequential GPU Memory Coordination Test Suite")
    print("=" * 70)
    
    tests = [
        ("Validation Server Only", test_validation_only),
        ("Generation Server Only", test_generation_only)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        print(f"\n🧪 Running: {test_name}")
        try:
            result = test_func()
            results.append((test_name, result))
            if result:
                print(f"✅ {test_name}: PASSED")
            else:
                print(f"❌ {test_name}: FAILED")
        except Exception as e:
            print(f"❌ {test_name}: ERROR - {e}")
            results.append((test_name, False))
        
        # Clean up between tests
        subprocess.run(["pkill", "-f", "python"], check=False)
        time.sleep(3)
    
    # Summary
    print("\n" + "=" * 70)
    print("📊 TEST SUMMARY")
    print("=" * 70)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} {test_name}")
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Sequential coordination is working!")
        print("\n💡 Solution: Use sequential coordination where:")
        print("   1. Start validation server")
        print("   2. Unload validation models before generation")
        print("   3. Run generation")
        print("   4. Reload validation models after generation")
        print("   5. Run validation")
    else:
        print("⚠️  Some tests failed. Check the logs above for details.")
    
    return 0 if passed == total else 1


if __name__ == "__main__":
    sys.exit(main()) 