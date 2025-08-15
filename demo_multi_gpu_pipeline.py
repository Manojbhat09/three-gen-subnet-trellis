#!/usr/bin/env python3
"""
Demo Script for Multi-GPU Pipeline Wrapper
Purpose: Simple demonstration of the pipeline functionality with a single prompt

This script shows how to:
1. Initialize the multi-GPU pipeline manager
2. Run both pipeline types on the same prompt
3. Compare results and performance
4. Save detailed analysis
"""

import asyncio
import time
import json
from pathlib import Path
from gpu_multi_pipeline_wrapper import MultiGPUPipelineManager

async def demo_pipeline():
    """Run a simple demo of both pipeline types"""
    
    # Demo prompt - choose something that should work well
    demo_prompt = "a blue ceramic coffee mug with a simple handle"
    
    print("🎯 MULTI-GPU PIPELINE DEMO")
    print("=" * 60)
    print(f"Demo prompt: '{demo_prompt}'")
    print("=" * 60)
    
    # Initialize the pipeline manager
    print("🚀 Initializing Multi-GPU Pipeline Manager...")
    manager = MultiGPUPipelineManager(
        num_gpus=8,
        base_port=8096,
        server_script="trellis_subnit_server_mix_lora_flash.py",
        output_dir="./demo_outputs"
    )
    
    try:
        # Check GPU server status
        print("\n🔍 Checking GPU server status...")
        loading_status = manager.check_gpu_loading_status()
        already_loaded = sum(1 for status in loading_status.values() if status == "already_loaded")
        
        print(f"   📊 GPU Status: {already_loaded}/8 already loaded")
        
        if already_loaded < 8:
            print(f"🚀 Starting {8 - already_loaded} GPU servers...")
            if not manager.start_all_servers():
                print("❌ Failed to start GPU servers - aborting demo")
                return
            print("✅ All GPU servers started successfully")
        else:
            print("✅ All GPUs already loaded and ready")
            # Update server status for already loaded GPUs
            for gpu_id, status in loading_status.items():
                if status == "already_loaded":
                    manager.gpu_servers[gpu_id].status = "healthy"
        
        # Demo 1: Image Ranking → PLY Pipeline
        print("\n" + "="*60)
        print("📊 DEMO 1: Image Ranking → PLY Pipeline")
        print("="*60)
        print("This pipeline:")
        print("  1. Generates 8 images across all GPUs")
        print("  2. Ranks images by CLIP text-image similarity")
        print("  3. Uses best images to generate PLY files")
        print("  4. Validates and ranks PLY files")
        print()
        
        start_time = time.time()
        results_1 = manager.run_image_ranking_to_ply_pipeline(
            prompt=demo_prompt,
            num_inference_steps=25,
            guidance_scale=7.5
        )
        time_1 = time.time() - start_time
        
        print(f"✅ Pipeline 1 completed in {time_1:.2f}s")
        print(f"   🥇 Best image: GPU {results_1.best_image_gpu} (CLIP: {results_1.best_clip_score:.4f})")
        if results_1.best_ply_gpu is not None:
            print(f"   🏆 Best PLY: GPU {results_1.best_ply_gpu} (Score: {results_1.best_validation_score:.4f})")
        
        # Demo 2: Single Image → Multi PLY Pipeline  
        print("\n" + "="*60)
        print("🔄 DEMO 2: Single Image → Multi PLY Pipeline")
        print("="*60)
        print("This pipeline:")
        print("  1. Generates 8 images across all GPUs")
        print("  2. Selects the single best image by CLIP score")
        print("  3. Generates 8 PLY variations from that image")
        print("  4. Validates and ranks PLY variations")
        print()
        
        start_time = time.time()
        results_2 = manager.run_single_image_multi_ply_pipeline(
            prompt=demo_prompt,
            num_inference_steps=25,
            guidance_scale=7.5
        )
        time_2 = time.time() - start_time
        
        print(f"✅ Pipeline 2 completed in {time_2:.2f}s")
        print(f"   🥇 Best image: GPU {results_2.best_image_gpu} (CLIP: {results_2.best_clip_score:.4f})")
        if results_2.best_ply_gpu is not None:
            print(f"   🏆 Best PLY: GPU {results_2.best_ply_gpu} (Score: {results_2.best_validation_score:.4f})")
        
        # Comparison Analysis
        print("\n" + "="*60)
        print("📈 PIPELINE COMPARISON ANALYSIS")
        print("="*60)
        
        print(f"⏱️  Execution Times:")
        print(f"   Image Ranking Pipeline: {time_1:.2f}s")
        print(f"   Single Image Pipeline:  {time_2:.2f}s")
        print(f"   Difference: {abs(time_1 - time_2):.2f}s ({'Pipeline 1' if time_1 < time_2 else 'Pipeline 2'} faster)")
        
        print(f"\n🎯 CLIP Scores (Image Quality):")
        print(f"   Image Ranking Pipeline: {results_1.best_clip_score:.4f}")
        print(f"   Single Image Pipeline:  {results_2.best_clip_score:.4f}")
        print(f"   Difference: {abs(results_1.best_clip_score - results_2.best_clip_score):.4f}")
        
        print(f"\n🏆 Validation Scores (PLY Quality):")
        print(f"   Image Ranking Pipeline: {results_1.best_validation_score:.4f}")
        print(f"   Single Image Pipeline:  {results_2.best_validation_score:.4f}")
        print(f"   Difference: {abs(results_1.best_validation_score - results_2.best_validation_score):.4f}")
        
        # Success Rate Analysis
        successful_images_1 = len([r for r in results_1.image_results if r.success])
        successful_images_2 = len([r for r in results_2.image_results if r.success])
        successful_plys_1 = len([r for r in results_1.ply_results if r.success])
        successful_plys_2 = len([r for r in results_2.ply_results if r.success])
        
        print(f"\n✅ Success Rates:")
        print(f"   Image Generation:")
        print(f"      Pipeline 1: {successful_images_1}/8 ({successful_images_1/8*100:.1f}%)")
        print(f"      Pipeline 2: {successful_images_2}/8 ({successful_images_2/8*100:.1f}%)")
        print(f"   PLY Generation:")
        print(f"      Pipeline 1: {successful_plys_1}/8 ({successful_plys_1/8*100:.1f}%)")
        print(f"      Pipeline 2: {successful_plys_2}/8 ({successful_plys_2/8*100:.1f}%)")
        
        # GPU Performance Analysis
        print(f"\n🔧 GPU Performance Analysis:")
        
        # Find fastest and slowest GPUs for images
        successful_image_results_1 = [r for r in results_1.image_results if r.success]
        if successful_image_results_1:
            fastest_img = min(successful_image_results_1, key=lambda x: x.generation_time)
            slowest_img = max(successful_image_results_1, key=lambda x: x.generation_time)
            
            print(f"   Image Generation:")
            print(f"      🏃 Fastest: GPU {fastest_img.gpu_id} ({fastest_img.generation_time:.2f}s)")
            print(f"      🐌 Slowest: GPU {slowest_img.gpu_id} ({slowest_img.generation_time:.2f}s)")
            print(f"      📊 Spread: {slowest_img.generation_time - fastest_img.generation_time:.2f}s")
        
        # Find fastest and slowest GPUs for PLY
        successful_ply_results_1 = [r for r in results_1.ply_results if r.success]
        if successful_ply_results_1:
            fastest_ply = min(successful_ply_results_1, key=lambda x: x.generation_time)
            slowest_ply = max(successful_ply_results_1, key=lambda x: x.generation_time)
            
            print(f"   PLY Generation:")
            print(f"      🏃 Fastest: GPU {fastest_ply.gpu_id} ({fastest_ply.generation_time:.2f}s)")
            print(f"      🐌 Slowest: GPU {slowest_ply.gpu_id} ({slowest_ply.generation_time:.2f}s)")
            print(f"      📊 Spread: {slowest_ply.generation_time - fastest_ply.generation_time:.2f}s")
        
        # Save demo results
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        manager.save_pipeline_results(results_1, f"{timestamp}_demo_image_ranking")
        manager.save_pipeline_results(results_2, f"{timestamp}_demo_single_image")
        
        # Create comparison summary
        comparison_data = {
            'demo_prompt': demo_prompt,
            'timestamp': timestamp,
            'execution_times': {
                'image_ranking_pipeline': time_1,
                'single_image_pipeline': time_2,
                'faster_pipeline': 'image_ranking' if time_1 < time_2 else 'single_image'
            },
            'clip_scores': {
                'image_ranking_pipeline': results_1.best_clip_score,
                'single_image_pipeline': results_2.best_clip_score,
                'better_pipeline': 'image_ranking' if results_1.best_clip_score > results_2.best_clip_score else 'single_image'
            },
            'validation_scores': {
                'image_ranking_pipeline': results_1.best_validation_score,
                'single_image_pipeline': results_2.best_validation_score,
                'better_pipeline': 'image_ranking' if results_1.best_validation_score > results_2.best_validation_score else 'single_image'
            },
            'success_rates': {
                'image_generation': {
                    'image_ranking_pipeline': successful_images_1 / 8,
                    'single_image_pipeline': successful_images_2 / 8
                },
                'ply_generation': {
                    'image_ranking_pipeline': successful_plys_1 / 8,
                    'single_image_pipeline': successful_plys_2 / 8
                }
            }
        }
        
        comparison_file = Path("./demo_outputs") / f"demo_comparison_{timestamp}.json"
        comparison_file.parent.mkdir(exist_ok=True)
        with open(comparison_file, 'w') as f:
            json.dump(comparison_data, f, indent=2)
        
        # Final Summary
        print("\n" + "="*60)
        print("🎉 DEMO COMPLETE!")
        print("="*60)
        print("📁 Results saved to:")
        print(f"   • ./demo_outputs/pipeline_results_image_ranking_to_ply_{timestamp}_demo_image_ranking.json")
        print(f"   • ./demo_outputs/pipeline_results_single_image_multi_ply_{timestamp}_demo_single_image.json")
        print(f"   • ./demo_outputs/demo_comparison_{timestamp}.json")
        print()
        print("📊 Overall Statistics:")
        manager.print_pipeline_summary()
        print()
        print("🚀 Next Steps:")
        print("   • Try different prompts with --prompt 'your prompt here'")
        print("   • Run the full test suite: python test_multi_gpu_pipeline.py")
        print("   • Check the example scripts: ./run_pipeline_example.sh")
        print("=" * 60)
        
    except KeyboardInterrupt:
        print("\n🛑 Demo interrupted by user")
    except Exception as e:
        print(f"\n❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Don't cleanup servers - they might be used by other processes
        print("\n📝 Note: GPU servers left running for potential additional use")
        print("   Use --skip-startup flag for subsequent runs")

if __name__ == "__main__":
    asyncio.run(demo_pipeline())
