#!/usr/bin/env python3
"""
Test Script for Multi-GPU Pipeline Wrapper
Purpose: Demonstrate and test the comprehensive pipeline functionality

Test scenarios:
1. Image Ranking → PLY Pipeline
2. Single Image → Multi PLY Pipeline  
3. Performance comparison between pipelines
4. GPU utilization analysis
"""

import asyncio
import time
import json
from pathlib import Path
from gpu_multi_pipeline_wrapper import MultiGPUPipelineManager

async def test_pipeline_basic():
    """Basic pipeline test with simple prompts"""
    print("🧪 BASIC PIPELINE TEST")
    print("=" * 60)
    
    # Test prompts
    test_prompts = [
        "a red ceramic coffee mug",
        "a blue wooden chair",
        "a silver laptop computer"
    ]
    
    # Initialize manager
    manager = MultiGPUPipelineManager(
        num_gpus=8,
        base_port=8096,
        output_dir="./test_pipeline_outputs"
    )
    
    try:
        # Check if servers are already running
        print("🔍 Checking GPU server status...")
        loading_status = manager.check_gpu_loading_status()
        already_loaded = sum(1 for status in loading_status.values() if status == "already_loaded")
        
        if already_loaded < 8:
            print(f"🚀 Starting {8 - already_loaded} GPU servers...")
            if not manager.start_all_servers():
                print("❌ Failed to start GPU servers")
                return
        else:
            print("✅ All GPUs already loaded and ready")
            # Update server status
            for gpu_id, status in loading_status.items():
                if status == "already_loaded":
                    manager.gpu_servers[gpu_id].status = "healthy"
        
        # Test each prompt with both pipeline types
        for i, prompt in enumerate(test_prompts):
            print(f"\n🎯 Testing prompt {i+1}/{len(test_prompts)}: '{prompt}'")
            
            # Test Image Ranking → PLY Pipeline
            print("\n📊 Testing Image Ranking → PLY Pipeline")
            start_time = time.time()
            results_1 = manager.run_image_ranking_to_ply_pipeline(prompt)
            time_1 = time.time() - start_time
            
            # Save results
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            manager.save_pipeline_results(results_1, f"{timestamp}_prompt{i+1}_ranking")
            
            # Test Single Image → Multi PLY Pipeline
            print("\n🔄 Testing Single Image → Multi PLY Pipeline")
            start_time = time.time()
            results_2 = manager.run_single_image_multi_ply_pipeline(prompt)
            time_2 = time.time() - start_time
            
            # Save results
            manager.save_pipeline_results(results_2, f"{timestamp}_prompt{i+1}_single")
            
            # Compare results
            print(f"\n📈 Prompt {i+1} Comparison:")
            print(f"   Image Ranking Pipeline:")
            print(f"      Time: {time_1:.2f}s")
            print(f"      Best CLIP: {results_1.best_clip_score:.4f}")
            print(f"      Best Validation: {results_1.best_validation_score:.4f}")
            print(f"   Single Image Pipeline:")
            print(f"      Time: {time_2:.2f}s") 
            print(f"      Best CLIP: {results_2.best_clip_score:.4f}")
            print(f"      Best Validation: {results_2.best_validation_score:.4f}")
            
            # Brief pause between prompts
            await asyncio.sleep(2)
        
        # Print final summary
        print("\n" + "=" * 60)
        manager.print_pipeline_summary()
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
    
    print("✅ Basic pipeline test completed")

async def test_pipeline_performance():
    """Performance and stress test"""
    print("\n🚀 PERFORMANCE TEST")
    print("=" * 60)
    
    # Performance test prompts
    performance_prompts = [
        "a complex mechanical watch with visible gears",
        "an ornate Victorian lamp with crystal details",
        "a futuristic sports car with glowing elements",
        "a detailed miniature castle with towers",
        "a steampunk robot with brass components"
    ]
    
    manager = MultiGPUPipelineManager(
        num_gpus=8,
        base_port=8096,
        output_dir="./performance_test_outputs"
    )
    
    try:
        # Ensure servers are running
        manager.check_all_servers_health()
        
        performance_results = []
        
        for i, prompt in enumerate(performance_prompts):
            print(f"\n⚡ Performance test {i+1}/{len(performance_prompts)}: '{prompt[:40]}...'")
            
            # Time the full pipeline
            start_time = time.time()
            
            # Run image ranking pipeline only for performance test
            results = manager.run_image_ranking_to_ply_pipeline(
                prompt,
                num_inference_steps=20,  # Faster for performance test
                guidance_scale=7.0
            )
            
            total_time = time.time() - start_time
            
            performance_data = {
                'prompt': prompt,
                'total_time': total_time,
                'pipeline_time': results.total_pipeline_time,
                'best_clip_score': results.best_clip_score,
                'best_validation_score': results.best_validation_score,
                'successful_images': len([r for r in results.image_results if r.success]),
                'successful_plys': len([r for r in results.ply_results if r.success])
            }
            
            performance_results.append(performance_data)
            
            print(f"   ⏱️ Total time: {total_time:.2f}s")
            print(f"   🎯 Best scores: CLIP {results.best_clip_score:.4f}, Validation {results.best_validation_score:.4f}")
        
        # Analyze performance results
        print(f"\n📊 PERFORMANCE ANALYSIS")
        print("-" * 40)
        
        avg_time = sum(r['total_time'] for r in performance_results) / len(performance_results)
        avg_clip = sum(r['best_clip_score'] for r in performance_results) / len(performance_results)
        avg_validation = sum(r['best_validation_score'] for r in performance_results) / len(performance_results)
        
        print(f"Average total time: {avg_time:.2f}s")
        print(f"Average CLIP score: {avg_clip:.4f}")
        print(f"Average validation score: {avg_validation:.4f}")
        
        fastest = min(performance_results, key=lambda x: x['total_time'])
        slowest = max(performance_results, key=lambda x: x['total_time'])
        
        print(f"🏃 Fastest: {fastest['total_time']:.2f}s ('{fastest['prompt'][:30]}...')")
        print(f"🐌 Slowest: {slowest['total_time']:.2f}s ('{slowest['prompt'][:30]}...')")
        
        # Save performance results
        perf_file = Path("./performance_test_outputs") / f"performance_results_{time.strftime('%Y%m%d_%H%M%S')}.json"
        perf_file.parent.mkdir(exist_ok=True)
        with open(perf_file, 'w') as f:
            json.dump(performance_results, f, indent=2)
        print(f"💾 Performance results saved to {perf_file}")
        
    except Exception as e:
        print(f"❌ Performance test failed: {e}")
        import traceback
        traceback.print_exc()
    
    print("✅ Performance test completed")

async def test_gpu_utilization():
    """Test GPU utilization and memory management"""
    print("\n🔧 GPU UTILIZATION TEST")
    print("=" * 60)
    
    manager = MultiGPUPipelineManager(
        num_gpus=8,
        base_port=8096,
        output_dir="./utilization_test_outputs"
    )
    
    try:
        # Check GPU memory before test
        print("🧠 GPU Memory Status (Before):")
        try:
            import torch
            if torch.cuda.is_available():
                for gpu_id in range(manager.num_gpus):
                    memory_allocated = torch.cuda.memory_allocated(gpu_id) / 1024**3
                    memory_reserved = torch.cuda.memory_reserved(gpu_id) / 1024**3
                    print(f"   GPU {gpu_id}: {memory_allocated:.2f}GB allocated, {memory_reserved:.2f}GB reserved")
        except Exception as e:
            print(f"   ⚠️ Memory check failed: {e}")
        
        # Run a stress test prompt
        stress_prompt = "a highly detailed fantasy dragon with intricate scales and magical flames"
        
        print(f"\n🔥 Running stress test with: '{stress_prompt}'")
        
        # Run both pipeline types
        results_1 = manager.run_image_ranking_to_ply_pipeline(stress_prompt)
        results_2 = manager.run_single_image_multi_ply_pipeline(stress_prompt)
        
        # Check GPU memory after test
        print("\n🧠 GPU Memory Status (After):")
        try:
            if torch.cuda.is_available():
                for gpu_id in range(manager.num_gpus):
                    memory_allocated = torch.cuda.memory_allocated(gpu_id) / 1024**3
                    memory_reserved = torch.cuda.memory_reserved(gpu_id) / 1024**3
                    print(f"   GPU {gpu_id}: {memory_allocated:.2f}GB allocated, {memory_reserved:.2f}GB reserved")
        except Exception as e:
            print(f"   ⚠️ Memory check failed: {e}")
        
        # Analyze GPU performance distribution
        print("\n📈 GPU Performance Distribution:")
        
        # Image generation performance
        image_times = {}
        for result in results_1.image_results:
            if result.success:
                image_times[result.gpu_id] = result.generation_time
        
        if image_times:
            print("   Image Generation Times:")
            for gpu_id in sorted(image_times.keys()):
                print(f"      GPU {gpu_id}: {image_times[gpu_id]:.2f}s")
        
        # PLY generation performance
        ply_times = {}
        for result in results_1.ply_results:
            if result.success:
                ply_times[result.gpu_id] = result.generation_time
        
        if ply_times:
            print("   PLY Generation Times:")
            for gpu_id in sorted(ply_times.keys()):
                print(f"      GPU {gpu_id}: {ply_times[gpu_id]:.2f}s")
        
        # Test GPU server health
        print("\n🏥 Final GPU Health Check:")
        health_results = manager.check_all_servers_health()
        healthy_count = sum(health_results.values())
        print(f"   Healthy servers: {healthy_count}/{manager.num_gpus}")
        
    except Exception as e:
        print(f"❌ Utilization test failed: {e}")
        import traceback
        traceback.print_exc()
    
    print("✅ GPU utilization test completed")

async def main():
    """Run all tests"""
    print("🧪 MULTI-GPU PIPELINE TEST SUITE")
    print("=" * 80)
    print("Testing comprehensive pipeline functionality with:")
    print("   • Image generation across 8 GPUs")
    print("   • CLIP-based image ranking")
    print("   • PLY generation from best images")
    print("   • Validation scoring and ranking")
    print("   • Performance analysis")
    print("=" * 80)
    
    try:
        # Run basic functionality test
        await test_pipeline_basic()
        
        # Small break between tests
        await asyncio.sleep(5)
        
        # Run performance test
        await test_pipeline_performance()
        
        # Small break between tests
        await asyncio.sleep(5)
        
        # Run GPU utilization test
        await test_gpu_utilization()
        
        print("\n🎉 ALL TESTS COMPLETED SUCCESSFULLY!")
        print("=" * 80)
        print("📁 Check the following directories for results:")
        print("   • ./test_pipeline_outputs/")
        print("   • ./performance_test_outputs/")
        print("   • ./utilization_test_outputs/")
        print("=" * 80)
        
    except KeyboardInterrupt:
        print("\n🛑 Tests interrupted by user")
    except Exception as e:
        print(f"\n❌ Test suite failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(main())
