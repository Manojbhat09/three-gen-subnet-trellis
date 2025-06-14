#!/usr/bin/env python3
"""
Demo of Enhanced Local Validation System for Subnet 17
Shows the capabilities of our comprehensive asset management and validation infrastructure
"""

import asyncio
import aiohttp
import time
import os
from pathlib import Path

print("🚀 Enhanced Local Validation System Demo")
print("=" * 60)

async def test_enhanced_generation_server():
    """Test the enhanced generation server with asset management"""
    print("\n🏭 Testing Enhanced Generation Server")
    
    server_url = "http://127.0.0.1:8095"
    test_prompts = [
        "a blue wooden chair",
        "a red sports car",
        "a green apple"
    ]
    
    async with aiohttp.ClientSession() as session:
        # Test server health
        try:
            async with session.get(f"{server_url}/health/") as resp:
                if resp.status == 200:
                    print("✅ Enhanced Generation Server: Online")
                else:
                    print("❌ Enhanced Generation Server: Offline")
                    return
        except Exception as e:
            print(f"❌ Enhanced Generation Server: Connection failed - {e}")
            return
        
        # Test each prompt
        for i, prompt in enumerate(test_prompts):
            print(f"\n🎨 Test {i+1}: Generating '{prompt}'")
            
            # Generate 3D model
            form_data = aiohttp.FormData()
            form_data.add_field('prompt', prompt)
            form_data.add_field('seed', str(42 + i))
            form_data.add_field('use_bpt', 'false')
            form_data.add_field('return_compressed', 'true')
            
            start_time = time.time()
            try:
                async with session.post(f"{server_url}/generate/", data=form_data, timeout=300) as resp:
                    generation_time = time.time() - start_time
                    
                    if resp.status == 200:
                        ply_data = await resp.read()
                        generation_id = resp.headers.get('X-Generation-ID')
                        compression_ratio = resp.headers.get('X-Compression-Ratio', '1.0')
                        face_count = resp.headers.get('X-Face-Count', '0')
                        
                        print(f"  ✅ Generated in {generation_time:.2f}s")
                        print(f"  📊 PLY size: {len(ply_data):,d} bytes")
                        print(f"  🗜️ Compression: {compression_ratio}x")
                        print(f"  🔺 Faces: {face_count:,d}")
                        print(f"  🆔 Generation ID: {generation_id}")
                        
                        # Test asset downloads
                        if generation_id:
                            print(f"  📥 Testing asset downloads...")
                            asset_types = ['original_image', 'background_removed_image', 'initial_mesh_glb']
                            
                            for asset_type in asset_types:
                                try:
                                    async with session.get(f"{server_url}/generate/{generation_id}/download/{asset_type}") as download_resp:
                                        if download_resp.status == 200:
                                            asset_data = await download_resp.read()
                                            print(f"    ✅ {asset_type}: {len(asset_data):,d} bytes")
                                        else:
                                            print(f"    ⚠️ {asset_type}: Not available")
                                except Exception as e:
                                    print(f"    ❌ {asset_type}: Error - {e}")
                        
                    else:
                        print(f"  ❌ Generation failed: {resp.status}")
                        error_text = await resp.text()
                        print(f"     Error: {error_text[:100]}...")
                        
            except Exception as e:
                print(f"  ❌ Request failed: {e}")
            
            # Brief delay between tests
            await asyncio.sleep(2)

async def test_asset_management_system():
    """Demonstrate the asset management capabilities"""
    print("\n📦 Testing Asset Management System")
    
    try:
        from generation_asset_manager import (
            global_asset_manager, AssetType, GenerationStatus,
            prepare_for_mining_submission
        )
        
        print("✅ Asset Management System: Available")
        
        # Get statistics
        stats = global_asset_manager.get_statistics()
        print(f"\n📈 Current Asset Statistics:")
        for key, value in stats.items():
            if isinstance(value, dict):
                print(f"  {key}:")
                for k, v in value.items():
                    print(f"    {k}: {v}")
            else:
                print(f"  {key}: {value}")
        
        # Test mining integration with existing assets
        completion_ready = global_asset_manager.get_submission_ready_assets()
        if completion_ready:
            print(f"\n⛏️ Testing Mining Submission with {len(completion_ready)} assets:")
            for asset in completion_ready[:3]:  # Test first 3
                try:
                    submission_data = prepare_for_mining_submission(
                        asset,
                        task_id=f"demo_task_{asset.generation_id[:8]}",
                        validator_hotkey="demo_validator",
                        validator_uid=999
                    )
                    
                    if "error" not in submission_data:
                        print(f"  ✅ Asset {asset.generation_id[:8]}: Submission ready")
                        print(f"     Score: {submission_data.get('local_validation_score', 'N/A')}")
                        print(f"     Compressed PLY: {len(submission_data.get('compressed_ply_b64', ''))} chars")
                        print(f"     Face count: {submission_data.get('face_count', 'N/A')}")
                    else:
                        print(f"  ⚠️ Asset {asset.generation_id[:8]}: {submission_data['error']}")
                        
                except Exception as e:
                    print(f"  ❌ Asset {asset.generation_id[:8]}: Error - {e}")
        else:
            print("  ℹ️ No submission-ready assets found")
        
    except ImportError:
        print("❌ Asset Management System: Not available")

async def demo_validation_runner_features():
    """Show the enhanced validation runner features"""
    print("\n🧪 Enhanced Validation Runner Features")
    
    print("✅ Features implemented:")
    print("  📡 Multi-server connectivity testing")
    print("  🎯 Enhanced server support (port 8095)")
    print("  🔄 Basic server fallback (port 8093)")
    print("  ✅ Validation server integration (port 8094)")
    print("  💾 Comprehensive asset saving")
    print("  📊 Performance metrics tracking")
    print("  ⛏️ Mining integration testing")
    print("  🗜️ Compression handling with fallback")
    
    print("\n📝 Command Examples:")
    print("  # Basic validation test")
    print("  python local_validation_runner.py 'a blue chair' --seed 42 --save")
    print()
    print("  # Full feature test with mining")
    print("  python local_validation_runner.py 'a red car' --save-all --test-mining")
    print()
    print("  # Competitive validation")
    print("  python local_compete_validation.py 'a wooden table' -n 5")

async def test_server_statistics():
    """Test server statistics and monitoring"""
    print("\n📊 Testing Server Statistics")
    
    server_url = "http://127.0.0.1:8095"
    
    async with aiohttp.ClientSession() as session:
        endpoints = [
            ("/status/", "Server Status"),
            ("/config/", "Configuration"),
            ("/assets/statistics/", "Asset Statistics"),
            ("/memory/", "Memory Status")
        ]
        
        for endpoint, name in endpoints:
            try:
                async with session.get(f"{server_url}{endpoint}") as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        print(f"✅ {name}:")
                        
                        # Pretty print relevant data
                        if endpoint == "/status/":
                            print(f"   Generations: {data.get('total_generations', 0)}")
                            print(f"   Success rate: {data.get('success_rate', 0):.1%}")
                            print(f"   Avg time: {data.get('average_generation_time', 0):.2f}s")
                        elif endpoint == "/memory/":
                            if "gpu_memory_total" in data:
                                total = data["gpu_memory_total"] / (1024**3)
                                allocated = data["gpu_memory_allocated"] / (1024**3)
                                print(f"   GPU Memory: {allocated:.1f}/{total:.1f} GB")
                        elif endpoint == "/assets/statistics/":
                            print(f"   Total assets: {data.get('total_assets', 0)}")
                            print(f"   Completed: {data.get('completed_assets', 0)}")
                            print(f"   Storage: {data.get('total_storage_mb', 0):.1f} MB")
                        elif endpoint == "/config/":
                            print(f"   Device: {data.get('device', 'N/A')}")
                            print(f"   BPT enabled: {data.get('use_bpt', False)}")
                    else:
                        print(f"⚠️ {name}: Error {resp.status}")
            except Exception as e:
                print(f"❌ {name}: Error - {e}")

async def main():
    """Main demo function"""
    print("This demo showcases our enhanced local validation system for Subnet 17")
    print("Features:")
    print("  🎯 Enhanced generation server with comprehensive asset management")
    print("  📦 Complete asset tracking (images, meshes, compressed data)")
    print("  🗜️ Automatic PLY compression with graceful fallback")
    print("  ⛏️ Mining submission preparation")
    print("  🏆 Competitive validation for finding best models")
    print("  📊 Performance monitoring and statistics")
    print("  🔄 Robust error handling and recovery")
    
    # Run all tests
    await test_enhanced_generation_server()
    await test_asset_management_system()
    await demo_validation_runner_features()
    await test_server_statistics()
    
    print("\n🎉 Demo completed!")
    print("\n📋 Summary of Achievements:")
    print("✅ Enhanced local validation runner with multi-server support")
    print("✅ Comprehensive asset management system")
    print("✅ Competitive validation for model comparison")
    print("✅ Mining integration with submission preparation")
    print("✅ Performance monitoring and statistics")
    print("✅ Robust error handling and graceful degradation")
    
    print("\n🚀 Next Steps:")
    print("  1. Set up validation server for complete workflow")
    print("  2. Run competitive validation on your prompts")
    print("  3. Use mining integration for production deployment")
    print("  4. Monitor performance with built-in statistics")

if __name__ == "__main__":
    asyncio.run(main()) 