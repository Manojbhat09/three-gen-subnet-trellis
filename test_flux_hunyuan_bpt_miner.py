#!/usr/bin/env python3
"""
Test script for Flux-Hunyuan-BPT Miner
Purpose: Local testing of the enhanced miner functionality
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
import random
import base64

# Mock data for testing
MOCK_TASKS = [
    {"id": "test_001", "prompt": "a red cube"},
    {"id": "test_002", "prompt": "a blue sphere"},
    {"id": "test_003", "prompt": "a wooden chair"},
    {"id": "test_004", "prompt": "a small house"},
    {"id": "test_005", "prompt": "a green apple"},
]

# Test Configuration
GENERATION_SERVER_URL = "http://127.0.0.1:8095"
LOCAL_VALIDATION_URL = "http://127.0.0.1:8094/validate_txt_to_3d_ply/"
TEST_OUTPUT_DIR = "test_outputs_miner"
GENERATION_TIMEOUT = 600  # 10 minutes

class MinerTestResults:
    def __init__(self):
        self.tests_run = 0
        self.tests_passed = 0
        self.tests_failed = 0
        self.errors = []
        self.generation_times = []
        self.validation_scores = []
        self.bpt_usage = []

def create_test_output_dir():
    """Create test output directory."""
    Path(TEST_OUTPUT_DIR).mkdir(exist_ok=True)
    print(f"✓ Test output directory created: {TEST_OUTPUT_DIR}")

async def check_generation_server_health() -> bool:
    """Check if generation server is available."""
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(f"{GENERATION_SERVER_URL}/health/", timeout=5) as response:
                return response.status == 200
    except Exception:
        return False

async def check_validation_server_health() -> bool:
    """Check if validation server is available."""
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(f"{LOCAL_VALIDATION_URL.replace('/validate_txt_to_3d_ply/', '/health/')}", timeout=5) as response:
                return response.status == 200
    except Exception:
        return False

async def test_generation_request(prompt: str, use_bpt: bool = True, seed: int = None) -> Dict[str, Any]:
    """Test a single generation request."""
    print(f"\n--- Testing Generation ---")
    print(f"Prompt: '{prompt}'")
    print(f"BPT: {use_bpt}")
    
    result = {
        "prompt": prompt,
        "use_bpt": use_bpt,
        "success": False,
        "generation_time": 0.0,
        "file_size": 0,
        "ply_data": None,
        "error": None
    }
    
    try:
        payload = {
            "prompt": prompt,
            "use_bpt": use_bpt,
            "seed": seed or random.randint(0, 2**31 - 1)
        }
        
        start_time = time.time()
        
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{GENERATION_SERVER_URL}/generate/",
                data=payload,
                timeout=aiohttp.ClientTimeout(total=GENERATION_TIMEOUT)
            ) as response:
                generation_time = time.time() - start_time
                result["generation_time"] = generation_time
                
                if response.status == 200:
                    ply_data = await response.read()
                    result["file_size"] = len(ply_data)
                    result["ply_data"] = ply_data
                    result["success"] = True
                    
                    print(f"✓ Generation successful!")
                    print(f"  Time: {generation_time:.2f}s")
                    print(f"  File size: {len(ply_data):,} bytes")
                    
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
    
    return result

async def test_validation_request(prompt: str, ply_data: bytes) -> Dict[str, Any]:
    """Test local validation."""
    print(f"\n--- Testing Validation ---")
    print(f"Prompt: '{prompt}'")
    print(f"PLY data size: {len(ply_data):,} bytes")
    
    result = {
        "prompt": prompt,
        "success": False,
        "score": 0.0,
        "error": None
    }
    
    try:
        # Compress PLY data (mock compression for now)
        import pyspz
        compressed_ply = pyspz.compress(ply_data)
        
        form_data = aiohttp.FormData()
        form_data.add_field('prompt', prompt)
        form_data.add_field('ply_file', compressed_ply, filename='model.ply.spz')
        
        async with aiohttp.ClientSession() as session:
            async with session.post(
                LOCAL_VALIDATION_URL,
                data=form_data,
                timeout=aiohttp.ClientTimeout(total=60)
            ) as response:
                if response.status == 200:
                    validation_result = await response.json()
                    score = validation_result.get('score', 0.0)
                    result["score"] = score
                    result["success"] = True
                    
                    print(f"✓ Validation successful!")
                    print(f"  Score: {score:.3f}")
                    
                else:
                    error_text = await response.text()
                    result["error"] = f"HTTP {response.status}: {error_text}"
                    print(f"✗ Validation failed: {result['error']}")
                    
    except Exception as e:
        result["error"] = str(e)
        print(f"✗ Validation failed with error: {e}")
    
    return result

async def test_full_mining_pipeline(task: Dict[str, str], use_bpt: bool = True) -> Dict[str, Any]:
    """Test the full mining pipeline for a single task."""
    print(f"\n{'='*50}")
    print(f"Testing Full Mining Pipeline")
    print(f"Task ID: {task['id']}")
    print(f"Prompt: {task['prompt']}")
    print(f"{'='*50}")
    
    pipeline_result = {
        "task_id": task["id"],
        "prompt": task["prompt"],
        "success": False,
        "generation_result": None,
        "validation_result": None,
        "overall_score": 0.0,
        "total_time": 0.0,
        "error": None
    }
    
    pipeline_start = time.time()
    
    try:
        # Step 1: Generation
        print("\n🎯 Step 1: Generate 3D Model")
        generation_result = await test_generation_request(task["prompt"], use_bpt=use_bpt)
        pipeline_result["generation_result"] = generation_result
        
        if not generation_result["success"]:
            pipeline_result["error"] = f"Generation failed: {generation_result['error']}"
            return pipeline_result
        
        # Step 2: Validation
        print("\n✅ Step 2: Validate Generated Model")
        validation_result = await test_validation_request(task["prompt"], generation_result["ply_data"])
        pipeline_result["validation_result"] = validation_result
        
        if not validation_result["success"]:
            pipeline_result["error"] = f"Validation failed: {validation_result['error']}"
            return pipeline_result
        
        # Step 3: Save result
        print("\n💾 Step 3: Save Generated Asset")
        try:
            timestamp = int(time.time())
            bpt_suffix = "_bpt" if use_bpt else "_no_bpt"
            filename = f"{task['id']}_{task['prompt'].replace(' ', '_')}{bpt_suffix}_{timestamp}.ply"
            filepath = Path(TEST_OUTPUT_DIR) / filename
            
            with open(filepath, 'wb') as f:
                f.write(generation_result["ply_data"])
            
            print(f"✓ Asset saved: {filepath}")
            
        except Exception as e:
            print(f"⚠️  Failed to save asset: {e}")
        
        # Calculate overall results
        pipeline_result["success"] = True
        pipeline_result["overall_score"] = validation_result["score"]
        pipeline_result["total_time"] = time.time() - pipeline_start
        
        print(f"\n🎉 Pipeline completed successfully!")
        print(f"  Total time: {pipeline_result['total_time']:.2f}s")
        print(f"  Validation score: {pipeline_result['overall_score']:.3f}")
        print(f"  BPT used: {use_bpt}")
        
    except Exception as e:
        pipeline_result["error"] = f"Pipeline error: {str(e)}"
        print(f"\n❌ Pipeline failed: {e}")
        traceback.print_exc()
    
    pipeline_result["total_time"] = time.time() - pipeline_start
    return pipeline_result

async def test_miner_health_checks():
    """Test health checks that the miner would perform."""
    print("\n=== Testing Miner Health Checks ===")
    
    results = {
        "generation_server": False,
        "validation_server": False,
        "overall_health": False
    }
    
    # Test generation server
    print("Checking generation server...")
    gen_health = await check_generation_server_health()
    results["generation_server"] = gen_health
    print(f"{'✓' if gen_health else '✗'} Generation server: {'Healthy' if gen_health else 'Unhealthy'}")
    
    # Test validation server (if available)
    print("Checking validation server...")
    val_health = await check_validation_server_health()
    results["validation_server"] = val_health
    print(f"{'✓' if val_health else '✗'} Validation server: {'Healthy' if val_health else 'Unhealthy/Not Required'}")
    
    # Overall health
    results["overall_health"] = gen_health  # Validation server is optional
    print(f"\n{'✅' if results['overall_health'] else '❌'} Overall health: {'Ready' if results['overall_health'] else 'Not Ready'}")
    
    return results

async def test_server_status_check():
    """Test server status endpoint."""
    print("\n=== Testing Server Status Check ===")
    
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(f"{GENERATION_SERVER_URL}/status/", timeout=5) as response:
                if response.status == 200:
                    data = await response.json()
                    print("✓ Server status retrieved successfully")
                    
                    # Check key information
                    bpt_enabled = data.get('bpt_enabled', False)
                    models_loaded = data.get('models_loaded', {})
                    metrics = data.get('metrics', {})
                    
                    print(f"  BPT Enabled: {bpt_enabled}")
                    print(f"  Models Loaded: {models_loaded}")
                    print(f"  Server Metrics: {metrics}")
                    
                    return data
                else:
                    print(f"✗ Status check failed with status: {response.status}")
                    return None
                    
    except Exception as e:
        print(f"✗ Status check failed: {e}")
        return None

async def run_comprehensive_miner_tests():
    """Run comprehensive miner tests."""
    print("🚀 Starting Comprehensive Flux-Hunyuan-BPT Miner Tests")
    print("=" * 60)
    
    create_test_output_dir()
    results = MinerTestResults()
    
    # Test 1: Health Checks
    print("\n📋 Test 1: Health Checks")
    health_results = await test_miner_health_checks()
    
    if health_results["overall_health"]:
        results.tests_passed += 1
        print("✓ Health checks passed")
    else:
        results.tests_failed += 1
        results.errors.append("Health checks failed")
        print("❌ Health checks failed - cannot proceed with generation tests")
        return results
    
    results.tests_run += 1
    
    # Test 2: Server Status
    print("\n📊 Test 2: Server Status Check")
    status_data = await test_server_status_check()
    results.tests_run += 1
    
    if status_data:
        results.tests_passed += 1
        print("✓ Server status check passed")
    else:
        results.tests_failed += 1
        results.errors.append("Server status check failed")
    
    # Test 3: Mining Pipeline Tests
    print("\n🔄 Test 3: Mining Pipeline Tests")
    
    # Test with BPT enabled
    for i, task in enumerate(MOCK_TASKS[:3]):  # Test first 3 tasks
        print(f"\n--- Pipeline Test {i+1}/3 (BPT Enabled) ---")
        
        pipeline_result = await test_full_mining_pipeline(task, use_bpt=True)
        results.tests_run += 1
        
        if pipeline_result["success"]:
            results.tests_passed += 1
            results.generation_times.append(pipeline_result["total_time"])
            results.validation_scores.append(pipeline_result["overall_score"])
            results.bpt_usage.append(True)
            print("✓ BPT pipeline test passed")
        else:
            results.tests_failed += 1
            results.errors.append(f"BPT pipeline test failed for {task['id']}: {pipeline_result['error']}")
            print("✗ BPT pipeline test failed")
    
    # Test with BPT disabled (faster)
    print(f"\n--- Pipeline Test (BPT Disabled) ---")
    pipeline_result = await test_full_mining_pipeline(MOCK_TASKS[0], use_bpt=False)
    results.tests_run += 1
    
    if pipeline_result["success"]:
        results.tests_passed += 1
        results.generation_times.append(pipeline_result["total_time"])
        results.validation_scores.append(pipeline_result["overall_score"])
        results.bpt_usage.append(False)
        print("✓ Non-BPT pipeline test passed")
    else:
        results.tests_failed += 1
        results.errors.append(f"Non-BPT pipeline test failed: {pipeline_result['error']}")
        print("✗ Non-BPT pipeline test failed")
    
    # Test 4: Stress Test (Optional)
    print("\n⚡ Test 4: Mini Stress Test")
    stress_tasks = MOCK_TASKS[3:5]  # Last 2 tasks
    
    for i, task in enumerate(stress_tasks):
        print(f"\n--- Stress Test {i+1}/2 ---")
        
        pipeline_result = await test_full_mining_pipeline(task, use_bpt=False)  # Faster without BPT
        results.tests_run += 1
        
        if pipeline_result["success"]:
            results.tests_passed += 1
            results.generation_times.append(pipeline_result["total_time"])
            results.validation_scores.append(pipeline_result["overall_score"])
            results.bpt_usage.append(False)
            print("✓ Stress test passed")
        else:
            results.tests_failed += 1
            results.errors.append(f"Stress test failed for {task['id']}: {pipeline_result['error']}")
            print("✗ Stress test failed")
    
    return results

def print_miner_test_summary(results: MinerTestResults):
    """Print comprehensive miner test results."""
    print("\n" + "=" * 60)
    print("🏁 MINER TEST SUMMARY")
    print("=" * 60)
    
    success_rate = (results.tests_passed / results.tests_run * 100) if results.tests_run > 0 else 0
    
    print(f"Total Tests: {results.tests_run}")
    print(f"Passed: {results.tests_passed} ✓")
    print(f"Failed: {results.tests_failed} ✗")
    print(f"Success Rate: {success_rate:.1f}%")
    
    if results.generation_times:
        avg_time = sum(results.generation_times) / len(results.generation_times)
        min_time = min(results.generation_times)
        max_time = max(results.generation_times)
        print(f"\nPipeline Performance:")
        print(f"  Average Time: {avg_time:.2f}s")
        print(f"  Min Time: {min_time:.2f}s")
        print(f"  Max Time: {max_time:.2f}s")
    
    if results.validation_scores:
        avg_score = sum(results.validation_scores) / len(results.validation_scores)
        min_score = min(results.validation_scores)
        max_score = max(results.validation_scores)
        print(f"\nValidation Scores:")
        print(f"  Average Score: {avg_score:.3f}")
        print(f"  Min Score: {min_score:.3f}")
        print(f"  Max Score: {max_score:.3f}")
    
    if results.bpt_usage:
        bpt_count = sum(1 for x in results.bpt_usage if x)
        no_bpt_count = sum(1 for x in results.bpt_usage if not x)
        print(f"\nBPT Usage:")
        print(f"  With BPT: {bpt_count}")
        print(f"  Without BPT: {no_bpt_count}")
    
    if results.errors:
        print(f"\n❌ Errors Encountered:")
        for i, error in enumerate(results.errors, 1):
            print(f"  {i}. {error}")
    
    # Overall assessment
    print(f"\n{'🎉 MINER STATUS: READY' if success_rate >= 80 else '⚠️  MINER STATUS: NEEDS ATTENTION'}")
    
    if success_rate >= 90:
        print("✅ Miner is ready for deployment!")
        print("💡 Recommendations:")
        print("  - Start the miner with: python flux_hunyuan_bpt_miner.py")
        print("  - Monitor logs in the ./logs directory")
        print("  - Check mining outputs in flux_hunyuan_bpt_mining_outputs/")
    elif success_rate >= 70:
        print("⚠️  Miner has some issues but may be functional")
        print("💡 Recommendations:")
        print("  - Review errors above and fix issues")
        print("  - Consider running with BPT disabled if BPT tests failed")
        print("  - Monitor performance closely")
    else:
        print("❌ Miner has significant issues that need to be addressed")
        print("💡 Recommendations:")
        print("  - Fix all errors before deployment")
        print("  - Ensure generation server is properly configured")
        print("  - Check GPU memory and dependencies")
    
    print(f"\nTest outputs saved to: {TEST_OUTPUT_DIR}")

async def main():
    """Main test function."""
    try:
        # Check prerequisites
        print("Checking prerequisites...")
        
        if not await check_generation_server_health():
            print(f"❌ Generation server not available at {GENERATION_SERVER_URL}")
            print("\nPlease ensure the Flux-Hunyuan-BPT server is running:")
            print("python flux_hunyuan_bpt_generation_server.py")
            return False
        
        print("✓ Generation server is available")
        
        # Run comprehensive tests
        results = await run_comprehensive_miner_tests()
        
        # Print summary
        print_miner_test_summary(results)
        
        # Return success/failure
        success_rate = (results.tests_passed / results.tests_run * 100) if results.tests_run > 0 else 0
        return success_rate >= 80
        
    except KeyboardInterrupt:
        print("\n⚠️  Tests interrupted by user")
        return False
    except Exception as e:
        print(f"\n❌ Test suite failed with error: {e}")
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("🧪 Flux-Hunyuan-BPT Miner Test Suite")
    print(f"Generation Server: {GENERATION_SERVER_URL}")
    print(f"Output Directory: {TEST_OUTPUT_DIR}")
    
    success = asyncio.run(main())
    sys.exit(0 if success else 1) 