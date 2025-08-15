#!/usr/bin/env python3
"""
Test GPU Wrapper - Subnet 17 (404-GEN)
Purpose: Test the GPU server wrapper functionality

This script demonstrates how to use the GPU wrapper to:
1. Start TRELLIS servers on all GPUs
2. Prime them in parallel
3. Test validation across all GPUs
4. Get comprehensive status reports
"""

import asyncio
import time
import json
from pathlib import Path
from gpu_server_wrapper import GPUServerManager

async def test_gpu_wrapper():
    """Test the GPU wrapper functionality"""
    print("🧪 Testing GPU Server Wrapper")
    print("=" * 50)
    
    # Create GPU server manager
    manager = GPUServerManager(
        num_gpus=8,  # Use all 8 GPUs
        base_port=8096,  # Start from port 8096
        server_script="trellis_subnit_server_mix_lora_flash.py",
        output_dir="./test_gpu_outputs"
    )
    
    try:
        # Step 1: Start all GPU servers
        print("\n🚀 STEP 1: Starting GPU servers...")
        if not manager.start_all_servers():
            print("❌ Failed to start GPU servers")
            return
        
        print("✅ All GPU servers started successfully")
        
        # Step 2: Prime all GPUs in parallel
        print("\n🎨 STEP 2: Priming all GPUs in parallel...")
        priming_results = manager.prime_all_gpus_parallel()
        
        # Analyze priming results
        successful_primes = [r for r in priming_results if r.get('success', False)]
        failed_primes = [r for r in priming_results if not r.get('success', False)]
        
        print(f"✅ Priming complete: {len(successful_primes)}/{len(priming_results)} successful")
        
        if successful_primes:
            avg_time = sum(r.get('generation_time', 0) for r in successful_primes) / len(successful_primes)
            print(f"   Average generation time: {avg_time:.2f}s")
        
        # Step 3: Test validation across all GPUs in parallel
        print("\n📊 STEP 3: Testing validation across all GPUs in parallel...")
        validation_results = manager.test_validation_parallel()
        
        # Analyze validation results
        successful_validations = [r for r in validation_results if r.get('success', False)]
        failed_validations = [r for r in validation_results if not r.get('success', False)]
        
        print(f"✅ Validation testing complete: {len(successful_validations)}/{len(validation_results)} successful")
        
        # Step 4: Get comprehensive status
        print("\n📊 STEP 4: Getting comprehensive status...")
        status_data = manager.get_comprehensive_status()
        
        # Print summary
        print(f"🏥 GPU Health: {status_data['health_summary']['healthy_servers']}/{status_data['health_summary']['total_servers']} healthy")
        print(f"🎨 Total Generations: {status_data['overall_stats']['total_generations']}")
        print(f"📊 Total Validations: {status_data['overall_stats']['total_validations']}")
        
        # Save results
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        results_file = manager.output_dir / f"test_results_{timestamp}.json"
        
        with open(results_file, 'w') as f:
            json.dump({
                'priming_results': priming_results,
                'validation_results': validation_results,
                'status_data': status_data,
                'timestamp': timestamp
            }, f, indent=2)
        
        print(f"💾 Test results saved to {results_file}")
        
        # Print detailed GPU status
        print("\n📊 GPU Status Details:")
        for gpu_id, gpu_status in status_data['gpu_servers'].items():
            status_icon = "✅" if gpu_status['status'] == 'healthy' else "❌"
            print(f"  GPU {gpu_id} (port {gpu_status['port']}): {status_icon} {gpu_status['status']}")
            print(f"    Generations: {gpu_status['generation_count']}, Validations: {gpu_status['validation_count']}")
            print(f"    Errors: {gpu_status['error_count']}")
        
        print("\n🎉 GPU wrapper test completed successfully!")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # Keep servers running for manual testing
        print("\n🔄 GPU servers are still running for manual testing...")
        print("   You can now test individual endpoints:")
        print("   curl -d 'prompt=pink bicycle' -X POST http://127.0.0.1:8096/generate/")
        print("   curl -d 'prompt=blue vase' -X POST http://127.0.0.1:8097/generate/")
        print("   # ... and so on for ports 8096-8103")
        print("\n   Press Ctrl+C to stop all servers")

def test_individual_endpoints():
    """Test individual GPU endpoints manually"""
    print("\n🧪 Testing individual GPU endpoints...")
    
    import requests
    
    test_prompts = [
        "a pink bicycle with chrome wheels",
        "a blue ceramic vase with red trim",
        "a wooden table with four chairs",
        "a silver laptop on a desk",
        "a red sports car in a garage",
        "a green plant in a pot",
        "a black coffee mug on a saucer",
        "a white cloud in a blue sky"
    ]
    
    for gpu_id in range(8):
        port = 8096 + gpu_id
        prompt = test_prompts[gpu_id]
        
        try:
            print(f"\n🎨 Testing GPU {gpu_id} (port {port}) with prompt: '{prompt[:30]}...'")
            
            response = requests.post(
                f"http://127.0.0.1:{port}/generate/",
                data={
                    'prompt': prompt,
                    'seed': 42,
                    'return_compressed': True
                },
                timeout=300
            )
            
            if response.status_code == 200:
                print(f"   ✅ GPU {gpu_id} successful: {len(response.content):,} bytes")
                print(f"   📊 Compression: {response.headers.get('X-Compression-Ratio', 'unknown')}")
            else:
                print(f"   ❌ GPU {gpu_id} failed: HTTP {response.status_code}")
                
        except Exception as e:
            print(f"   ❌ GPU {gpu_id} exception: {e}")

if __name__ == "__main__":
    print("🚀 GPU Server Wrapper Test")
    print("=" * 50)
    print("This script will:")
    print("1. Start TRELLIS servers on all 8 GPUs")
    print("2. Prime them in parallel")
    print("3. Test validation across all GPUs")
    print("4. Provide comprehensive status reports")
    print("\nMake sure you have:")
    print("- 8 GPUs available")
    print("- The trellis_subnit_server_mix_lora_flash.py script in the current directory")
    print("- Sufficient GPU memory for model loading")
    print("\nStarting test in 5 seconds...")
    
    try:
        time.sleep(5)
        asyncio.run(test_gpu_wrapper())
        
        # After the main test, offer to test individual endpoints
        print("\n" + "=" * 50)
        response = input("Would you like to test individual GPU endpoints? (y/n): ")
        if response.lower() in ['y', 'yes']:
            test_individual_endpoints()
        
    except KeyboardInterrupt:
        print("\n🛑 Test interrupted by user")
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
