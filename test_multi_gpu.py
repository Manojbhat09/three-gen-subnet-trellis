#!/usr/bin/env python3
"""
Test script for Multi-GPU FLUX + TRELLIS Generation Server
Demonstrates parallel generation across 8 GPUs
"""

import requests
import time
import json
from typing import List, Dict, Any

# Server configuration
SERVER_URL = "http://localhost:8096"

def test_gpu_status():
    """Test GPU status endpoint"""
    print("🔍 Testing GPU Status...")
    try:
        response = requests.get(f"{SERVER_URL}/gpu_status/")
        if response.status_code == 200:
            status = response.json()
            print(f"✅ GPU Status:")
            print(f"   Total GPUs: {status['num_gpus']}")
            print(f"   Available: {status['available_gpus']}")
            print(f"   Busy: {status['busy_gpus']}")
            print(f"   Total Jobs: {status['total_jobs']}")
            return status
        else:
            print(f"❌ Failed to get GPU status: {response.status_code}")
            return None
    except Exception as e:
        print(f"❌ Error getting GPU status: {e}")
        return None

def test_gpu_health():
    """Test GPU health endpoint"""
    print("🏥 Testing GPU Health...")
    try:
        response = requests.get(f"{SERVER_URL}/gpu_health/")
        if response.status_code == 200:
            health = response.json()
            print(f"✅ GPU Health:")
            
            for gpu_id, gpu_info in health['gpu_health'].items():
                if 'error' in gpu_info:
                    print(f"   GPU {gpu_id}: ❌ Error - {gpu_info['error']}")
                else:
                    status_emoji = "🟢" if gpu_info['status'] == 'idle' else "🟡" if gpu_info['status'] == 'busy' else "🔴"
                    print(f"   GPU {gpu_id}: {status_emoji} {gpu_info['status']}")
                    print(f"      Memory: {gpu_info['memory_allocated_gb']:.1f}GB / {gpu_info['memory_total_gb']:.1f}GB")
                    if 'temperature_celsius' in gpu_info:
                        print(f"      Temp: {gpu_info['temperature_celsius']:.1f}°C, Power: {gpu_info.get('power_draw_watts', 0):.1f}W")
            
            return health
        else:
            print(f"❌ Failed to get GPU health: {response.status_code}")
            return None
    except Exception as e:
        print(f"❌ Error getting GPU health: {e}")
        return None

def test_single_generation(gpu_id: int, prompt: str, seed: int = 42):
    """Test single generation on specific GPU"""
    print(f"🎯 Testing single generation on GPU {gpu_id}...")
    try:
        data = {
            'prompt': prompt,
            'seed': seed,
            'gpu_id': gpu_id,
            'return_compressed': True
        }
        
        start_time = time.time()
        response = requests.post(f"{SERVER_URL}/generate/", data=data)
        generation_time = time.time() - start_time
        
        if response.status_code == 200:
            print(f"✅ Generation completed on GPU {gpu_id} in {generation_time:.2f}s")
            print(f"   Response size: {len(response.content):,} bytes")
            print(f"   Headers: {dict(response.headers)}")
            return True
        else:
            print(f"❌ Generation failed on GPU {gpu_id}: {response.status_code}")
            print(f"   Error: {response.text}")
            return False
    except Exception as e:
        print(f"❌ Error testing generation on GPU {gpu_id}: {e}")
        return False

def test_parallel_generation(prompt: str, seeds: List[int], max_parallel: int = 8):
    """Test parallel generation across multiple GPUs"""
    print(f"🚀 Testing parallel generation of {len(seeds)} models...")
    try:
        data = {
            'prompt': prompt,
            'seeds': ','.join(map(str, seeds)),
            'return_compressed': True,
            'max_parallel': max_parallel
        }
        
        start_time = time.time()
        response = requests.post(f"{SERVER_URL}/generate_parallel/", data=data)
        total_time = time.time() - start_time
        
        if response.status_code == 200:
            result = response.json()
            print(f"✅ Parallel generation completed in {total_time:.2f}s")
            print(f"   Total models: {result['total_models']}")
            print(f"   Successful: {result['successful']}")
            print(f"   Failed: {result['failed']}")
            print(f"   Success rate: {result['success_rate']:.1f}%")
            
            if 'jobs' in result:
                for job in result['jobs']:
                    status_emoji = "✅" if job['status'] == 'completed' else "❌"
                    gpu_info = f"GPU {job.get('gpu_id', 'N/A')}"
                    error_info = f" - {job.get('error', '')}" if job.get('error') else ""
                    print(f"   {status_emoji} Job {job['id']}: {job['status']} on {gpu_info}{error_info}")
            
            return result
        else:
            print(f"❌ Parallel generation failed: {response.status_code}")
            print(f"   Error: {response.text}")
            return None
    except Exception as e:
        print(f"❌ Error testing parallel generation: {e}")
        return None

def test_parallel_lora_generation(lora_style: str, prompt: str, seeds: List[int], max_parallel: int = 8):
    """Test parallel generation with specific LoRA style"""
    print(f"🎨 Testing parallel {lora_style} LoRA generation...")
    try:
        data = {
            'prompt': prompt,
            'seeds': ','.join(map(str, seeds)),
            'return_compressed': True,
            'max_parallel': max_parallel
        }
        
        start_time = time.time()
        response = requests.post(f"{SERVER_URL}/generate_parallel_lora/{lora_style}/", data=data)
        total_time = time.time() - start_time
        
        if response.status_code == 200:
            result = response.json()
            print(f"✅ {lora_style} LoRA generation completed in {total_time:.2f}s")
            print(f"   Total models: {result['total_models']}")
            print(f"   Successful: {result['successful']}")
            print(f"   Failed: {result['failed']}")
            print(f"   Success rate: {result['success_rate']:.1f}%")
            
            return result
        else:
            print(f"❌ {lora_style} LoRA generation failed: {response.status_code}")
            print(f"   Error: {response.text}")
            return None
    except Exception as e:
        print(f"❌ Error testing {lora_style} LoRA generation: {e}")
        return None

def test_server_status():
    """Test server status endpoint"""
    print("🔍 Testing Server Status...")
    try:
        response = requests.get(f"{SERVER_URL}/status/")
        if response.status_code == 200:
            status = response.json()
            print(f"✅ Server Status:")
            print(f"   Status: {status.get('status', 'N/A')}")
            print(f"   Ready: {status.get('ready', False)}")
            
            if 'multi_gpu' in status:
                multi_gpu = status['multi_gpu']
                print(f"   Multi-GPU Enabled: {multi_gpu.get('enabled', False)}")
                print(f"   Number of GPUs: {multi_gpu.get('num_gpus', 0)}")
                print(f"   Parallel Capacity: {multi_gpu.get('parallel_capacity', 0)}")
                print(f"   Active Generators: {multi_gpu.get('active_generators', 0)}")
            
            return status
        else:
            print(f"❌ Failed to get server status: {response.status_code}")
            return None
    except Exception as e:
        print(f"❌ Error getting server status: {e}")
        return None

def main():
    """Main test function"""
    print("🧪 Multi-GPU FLUX + TRELLIS Generation Server Test")
    print("=" * 60)
    
    # Test 1: Server status
    server_status = test_server_status()
    if not server_status:
        print("❌ Server not responding, exiting...")
        return
    
    print()
    
    # Test 2: GPU status
    gpu_status = test_gpu_status()
    if not gpu_status:
        print("❌ Cannot get GPU status, exiting...")
        return
    
    print()
    
    # Test 3: GPU health check
    gpu_health = test_gpu_health()
    if not gpu_health:
        print("⚠️ Cannot get GPU health, continuing...")
    
    print()
    
    # Test 4: Single generation on GPU 0
    test_single_generation(0, "a beautiful ceramic vase with intricate patterns", 42)
    
    print()
    
    # Test 5: Parallel generation across all GPUs
    seeds = [42, 43, 44, 45, 46, 47, 48, 49]
    test_parallel_generation("a futuristic robot with glowing blue eyes", seeds, 8)
    
    print()
    
    # Test 6: Parallel LoRA generation
    lora_seeds = [50, 51, 52, 53, 54, 55, 56, 57]
    test_parallel_lora_generation("isometric_3d", "a modern office building", lora_seeds, 8)
    
    print()
    
    # Test 7: Check parallel jobs status
    print("📊 Checking parallel jobs status...")
    try:
        response = requests.get(f"{SERVER_URL}/parallel_jobs/")
        if response.status_code == 200:
            jobs_status = response.json()
            print(f"   Active jobs: {jobs_status['active_jobs']}")
            print(f"   Completed jobs: {jobs_status['completed_jobs']}")
            print(f"   Failed jobs: {jobs_status['failed_jobs']}")
        else:
            print(f"   Failed to get jobs status: {response.status_code}")
    except Exception as e:
        print(f"   Error getting jobs status: {e}")
    
    print()
    print("✅ Multi-GPU testing completed!")

if __name__ == "__main__":
    main()

