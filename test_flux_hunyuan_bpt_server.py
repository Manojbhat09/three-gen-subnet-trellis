#!/usr/bin/env python3
"""
Test script for Flux-Hunyuan-BPT Generation Server
Purpose: Comprehensive testing of the enhanced generation server
"""

import asyncio
import aiohttp
import json
import time
import sys
import os
from pathlib import Path
from typing import Dict, Any, Optional, List
import traceback

# Test Configuration
SERVER_BASE_URL = "http://127.0.0.1:8095"
TEST_OUTPUT_DIR = "test_outputs_flux_hunyuan_bpt"
TEST_PROMPTS = [
    "a simple red cube",
    "a blue sphere with smooth surface",
    "a wooden chair with four legs",
    "a small house with a red roof",
    "a cartoon-style car",
    "a green apple",
    "a coffee mug",
    "a computer mouse"
]

# Test timeouts
HEALTH_CHECK_TIMEOUT = 10
GENERATION_TIMEOUT = 300  # 5 minutes instead of 15 minutes
STATUS_CHECK_TIMEOUT = 5

class TestResults:
    def __init__(self):
        self.passed = 0
        self.failed = 0
        self.errors = []
        self.generation_times = []
        self.file_sizes = []
        self.bpt_usage = []

def create_test_output_dir():
    """Create test output directory."""
    Path(TEST_OUTPUT_DIR).mkdir(exist_ok=True)
    print(f"✓ Test output directory created: {TEST_OUTPUT_DIR}")

async def test_server_health() -> bool:
    """Test server health endpoint."""
    print("\n=== Testing Server Health ===")
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(
                f"{SERVER_BASE_URL}/health/",
                timeout=aiohttp.ClientTimeout(total=HEALTH_CHECK_TIMEOUT)
            ) as response:
                if response.status == 200:
                    data = await response.json()
                    print(f"✓ Health check passed: {data}")
                    return True
                else:
                    print(f"✗ Health check failed with status: {response.status}")
                    return False
    except Exception as e:
        print(f"✗ Health check failed with error: {e}")
        return False

async def test_server_status() -> Optional[Dict[str, Any]]:
    """Test server status endpoint."""
    print("\n=== Testing Server Status ===")
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(
                f"{SERVER_BASE_URL}/status/",
                timeout=aiohttp.ClientTimeout(total=STATUS_CHECK_TIMEOUT)
            ) as response:
                if response.status == 200:
                    data = await response.json()
                    print(f"✓ Status check passed")
                    print(f"  Device: {data.get('device', 'unknown')}")
                    print(f"  BPT Enabled: {data.get('bpt_enabled', False)}")
                    
                    models_loaded = data.get('models_loaded', {})
                    print(f"  Models Loaded:")
                    for model_name, loaded in models_loaded.items():
                        status = "✓" if loaded else "✗"
                        print(f"    {status} {model_name}")
                    
                    metrics = data.get('metrics', {})
                    print(f"  Metrics:")
                    for key, value in metrics.items():
                        print(f"    {key}: {value}")
                    
                    return data
                else:
                    print(f"✗ Status check failed with status: {response.status}")
                    return None
    except Exception as e:
        print(f"✗ Status check failed with error: {e}")
        return None

async def test_server_config() -> Optional[Dict[str, Any]]:
    """Test server config endpoint."""
    print("\n=== Testing Server Config ===")
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(
                f"{SERVER_BASE_URL}/config/",
                timeout=aiohttp.ClientTimeout(total=STATUS_CHECK_TIMEOUT)
            ) as response:
                if response.status == 200:
                    data = await response.json()
                    print(f"✓ Config retrieved successfully")
                    print(f"  Output Directory: {data.get('output_dir', 'unknown')}")
                    print(f"  Use BPT: {data.get('use_bpt', False)}")
                    print(f"  BPT Temperature: {data.get('bpt_temperature', 'unknown')}")
                    print(f"  Device: {data.get('device', 'unknown')}")
                    return data
                else:
                    print(f"✗ Config retrieval failed with status: {response.status}")
                    return None
    except Exception as e:
        print(f"✗ Config retrieval failed with error: {e}")
        return None

async def test_single_generation(prompt: str, use_bpt: bool = None, seed: int = None) -> Dict[str, Any]:
    """Test a single 3D model generation."""
    print(f"\n--- Testing Generation ---")
    print(f"Prompt: '{prompt}'")
    print(f"BPT: {use_bpt}")
    print(f"Seed: {seed}")
    
    result = {
        "prompt": prompt,
        "use_bpt": use_bpt,
        "seed": seed,
        "success": False,
        "generation_time": 0.0,
        "file_size": 0,
        "error": None
    }
    
    try:
        payload = {"prompt": prompt}
        if use_bpt is not None:
            payload["use_bpt"] = use_bpt
        if seed is not None:
            payload["seed"] = seed
        
        start_time = time.time()
        
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{SERVER_BASE_URL}/generate/",
                data=payload,
                timeout=aiohttp.ClientTimeout(total=GENERATION_TIMEOUT)
            ) as response:
                generation_time = time.time() - start_time
                result["generation_time"] = generation_time
                
                if response.status == 200:
                    ply_data = await response.read()
                    file_size = len(ply_data)
                    result["file_size"] = file_size
                    result["success"] = True
                    
                    # Save the generated file
                    bpt_suffix = "_bpt" if use_bpt else "_no_bpt"
                    filename = f"test_{prompt.replace(' ', '_')}{bpt_suffix}_{int(time.time())}.ply"
                    filepath = Path(TEST_OUTPUT_DIR) / filename
                    
                    with open(filepath, 'wb') as f:
                        f.write(ply_data)
                    
                    result["filepath"] = str(filepath)
                    
                    print(f"✓ Generation successful!")
                    print(f"  Time: {generation_time:.2f}s")
                    print(f"  File size: {file_size:,} bytes")
                    print(f"  Saved to: {filepath}")
                    
                else:
                    error_text = await response.text()
                    result["error"] = f"HTTP {response.status}: {error_text}"
                    print(f"✗ Generation failed: {result['error']}")
                    
    except asyncio.TimeoutError:
        result["error"] = f"Timeout after {GENERATION_TIMEOUT}s"
        print(f"✗ Generation timed out after {GENERATION_TIMEOUT}s")
    except Exception as e:
        result["error"] = str(e)
        print(f"✗ Generation failed with error: {e}")
        traceback.print_exc()
    
    return result

async def test_cache_clear() -> bool:
    """Test cache clearing functionality."""
    print("\n=== Testing Cache Clear ===")
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{SERVER_BASE_URL}/clear_cache/",
                timeout=aiohttp.ClientTimeout(total=10)
            ) as response:
                if response.status == 200:
                    data = await response.json()
                    print(f"✓ Cache clear successful: {data}")
                    return True
                else:
                    print(f"✗ Cache clear failed with status: {response.status}")
                    return False
    except Exception as e:
        print(f"✗ Cache clear failed with error: {e}")
        return False

async def run_comprehensive_tests():
    """Run comprehensive tests on the generation server."""
    print("🚀 Starting Comprehensive Flux-Hunyuan-BPT Server Tests")
    print("=" * 60)
    
    create_test_output_dir()
    results = TestResults()
    
    # Test 1: Health Check
    if await test_server_health():
        results.passed += 1
    else:
        results.failed += 1
        results.errors.append("Health check failed")
        print("❌ Server health check failed. Please ensure the server is running.")
        return results
    
    # Test 2: Status Check
    status_data = await test_server_status()
    if status_data:
        results.passed += 1
        
        # Check if BPT is enabled
        bpt_enabled = status_data.get('bpt_enabled', False)
        models_loaded = status_data.get('models_loaded', {})
        
        if not bpt_enabled:
            print("⚠️  BPT is not enabled on the server")
        
        if not models_loaded.get('bpt', False):
            print("⚠️  BPT model is not loaded")
        
    else:
        results.failed += 1
        results.errors.append("Status check failed")
    
    # Test 3: Config Check
    config_data = await test_server_config()
    if config_data:
        results.passed += 1
    else:
        results.failed += 1
        results.errors.append("Config check failed")
    
    # Test 4: Cache Clear
    if await test_cache_clear():
        results.passed += 1
    else:
        results.failed += 1
        results.errors.append("Cache clear failed")
    
    # Test 5: Single Generation Tests
    print("\n=== Testing 3D Model Generation ===")
    
    # Test with different configurations - reduced to avoid server overload
    test_configs = [
        {"use_bpt": False, "description": "without BPT enhancement"},  # Test non-BPT first
        {"use_bpt": None, "description": "with default settings"},
    ]
    
    for i, test_prompt in enumerate(TEST_PROMPTS[:2]):  # Test only first 2 prompts
        for config in test_configs:
            print(f"\n--- Test {i+1}.{test_configs.index(config)+1}: {test_prompt} ({config['description']}) ---")
            
            result = await test_single_generation(
                prompt=test_prompt,
                use_bpt=config["use_bpt"],
                seed=42 + i  # Consistent seeds for reproducibility
            )
            
            if result["success"]:
                results.passed += 1
                results.generation_times.append(result["generation_time"])
                results.file_sizes.append(result["file_size"])
                results.bpt_usage.append(config["use_bpt"])
            else:
                results.failed += 1
                results.errors.append(f"Generation failed for '{test_prompt}': {result['error']}")
                
            # Add small delay between tests to help server recover
            await asyncio.sleep(2)
    
    # Test 6: Performance Test
    print("\n=== Performance Test ===")
    print("Testing generation with a complex prompt...")
    
    complex_prompt = "a detailed medieval castle with towers, walls, and a drawbridge"
    perf_result = await test_single_generation(
        prompt=complex_prompt,
        use_bpt=True,
        seed=123
    )
    
    if perf_result["success"]:
        results.passed += 1
        results.generation_times.append(perf_result["generation_time"])
        results.file_sizes.append(perf_result["file_size"])
        print(f"✓ Performance test passed: {perf_result['generation_time']:.2f}s")
    else:
        results.failed += 1
        results.errors.append(f"Performance test failed: {perf_result['error']}")
    
    # Test 7: Stress Test (optional)
    print("\n=== Quick Stress Test ===")
    print("Testing rapid consecutive generations...")
    
    stress_results = []
    for i in range(2):  # Small stress test
        stress_result = await test_single_generation(
            prompt=f"test object {i+1}",
            use_bpt=False,  # Faster without BPT
            seed=1000 + i
        )
        stress_results.append(stress_result)
        
        if stress_result["success"]:
            results.passed += 1
        else:
            results.failed += 1
            results.errors.append(f"Stress test {i+1} failed: {stress_result['error']}")
    
    return results

def print_test_summary(results: TestResults):
    """Print comprehensive test results."""
    print("\n" + "=" * 60)
    print("🏁 TEST SUMMARY")
    print("=" * 60)
    
    total_tests = results.passed + results.failed
    success_rate = (results.passed / total_tests * 100) if total_tests > 0 else 0
    
    print(f"Total Tests: {total_tests}")
    print(f"Passed: {results.passed} ✓")
    print(f"Failed: {results.failed} ✗")
    print(f"Success Rate: {success_rate:.1f}%")
    
    if results.generation_times:
        avg_time = sum(results.generation_times) / len(results.generation_times)
        min_time = min(results.generation_times)
        max_time = max(results.generation_times)
        print(f"\nGeneration Performance:")
        print(f"  Average Time: {avg_time:.2f}s")
        print(f"  Min Time: {min_time:.2f}s")
        print(f"  Max Time: {max_time:.2f}s")
    
    if results.file_sizes:
        avg_size = sum(results.file_sizes) / len(results.file_sizes)
        min_size = min(results.file_sizes)
        max_size = max(results.file_sizes)
        print(f"\nFile Sizes:")
        print(f"  Average Size: {avg_size:,.0f} bytes")
        print(f"  Min Size: {min_size:,} bytes")
        print(f"  Max Size: {max_size:,} bytes")
    
    if results.bpt_usage:
        bpt_count = sum(1 for x in results.bpt_usage if x is True)
        no_bpt_count = sum(1 for x in results.bpt_usage if x is False)
        print(f"\nBPT Usage:")
        print(f"  With BPT: {bpt_count}")
        print(f"  Without BPT: {no_bpt_count}")
    
    if results.errors:
        print(f"\n❌ Errors Encountered:")
        for i, error in enumerate(results.errors, 1):
            print(f"  {i}. {error}")
    
    # Overall assessment
    print(f"\n{'🎉 OVERALL: SUCCESS' if success_rate >= 80 else '⚠️  OVERALL: NEEDS ATTENTION'}")
    
    if success_rate >= 90:
        print("✅ Server is ready for production use!")
    elif success_rate >= 70:
        print("⚠️  Server has some issues but is functional")
    else:
        print("❌ Server has significant issues that need to be addressed")
    
    print(f"\nTest outputs saved to: {TEST_OUTPUT_DIR}")

async def main():
    """Main test function."""
    try:
        # Check if server is accessible
        print("Checking server accessibility...")
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(f"{SERVER_BASE_URL}/health/", timeout=5) as response:
                    pass
        except Exception as e:
            print(f"❌ Cannot connect to server at {SERVER_BASE_URL}")
            print(f"Error: {e}")
            print("\nPlease ensure the Flux-Hunyuan-BPT server is running:")
            print(f"python flux_hunyuan_bpt_generation_server.py")
            return False
        
        # Run comprehensive tests
        results = await run_comprehensive_tests()
        
        # Print summary
        print_test_summary(results)
        
        # Return success/failure
        total_tests = results.passed + results.failed
        success_rate = (results.passed / total_tests * 100) if total_tests > 0 else 0
        return success_rate >= 80
        
    except KeyboardInterrupt:
        print("\n⚠️  Tests interrupted by user")
        return False
    except Exception as e:
        print(f"\n❌ Test suite failed with error: {e}")
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("🧪 Flux-Hunyuan-BPT Generation Server Test Suite")
    print(f"Server URL: {SERVER_BASE_URL}")
    print(f"Output Directory: {TEST_OUTPUT_DIR}")
    
    success = asyncio.run(main())
    sys.exit(0 if success else 1) 