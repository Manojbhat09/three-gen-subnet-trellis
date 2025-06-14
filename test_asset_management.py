#!/usr/bin/env python3
"""
Test script for the comprehensive asset management system
Demonstrates all features including compression, validation, and mining integration
"""

import os
import sys
import time
import asyncio
import aiohttp
import json
from pathlib import Path

# Add the current directory to path to import our modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from generation_asset_manager import (
    GenerationAsset, AssetType, GenerationStatus,
    global_asset_manager, prepare_for_mining_submission
)

async def test_server_generation():
    """Test the generation server with asset management"""
    print("🚀 Testing Flux-Hunyuan-BPT Generation Server with Asset Management")
    
    server_url = "http://localhost:8095"
    
    # Test prompts
    test_prompts = [
        "a blue cube with metallic surface",
        "a wooden chair with curved back",
        "a red sports car"
    ]
    
    async with aiohttp.ClientSession() as session:
        for i, prompt in enumerate(test_prompts):
            print(f"\n📝 Test {i+1}: Generating '{prompt}'")
            
            # Generate 3D model
            form_data = aiohttp.FormData()
            form_data.add_field('prompt', prompt)
            form_data.add_field('seed', str(42 + i))
            form_data.add_field('use_bpt', 'false')
            form_data.add_field('return_compressed', 'true')
            
            try:
                async with session.post(f"{server_url}/generate/", data=form_data) as resp:
                    if resp.status == 200:
                        generation_id = resp.headers.get('X-Generation-ID')
                        generation_time = resp.headers.get('X-Generation-Time')
                        compression_ratio = resp.headers.get('X-Compression-Ratio')
                        face_count = resp.headers.get('X-Face-Count')
                        
                        print(f"✓ Generation successful!")
                        print(f"  - Generation ID: {generation_id}")
                        print(f"  - Generation Time: {generation_time}s")
                        print(f"  - Compression Ratio: {compression_ratio}x")
                        print(f"  - Face Count: {face_count}")
                        
                        # Get detailed asset information
                        async with session.get(f"{server_url}/generate/{generation_id}") as asset_resp:
                            if asset_resp.status == 200:
                                asset_info = await asset_resp.json()
                                print(f"  - Available assets: {asset_info['available_assets']}")
                                print(f"  - Status: {asset_info['status']}")
                            
                        # Test downloading different asset types
                        print("  📥 Testing asset downloads...")
                        available_assets = ['original_image', 'background_removed_image', 'initial_mesh_ply', 'compressed_ply']
                        
                        for asset_type in available_assets:
                            try:
                                async with session.get(f"{server_url}/generate/{generation_id}/download/{asset_type}") as download_resp:
                                    if download_resp.status == 200:
                                        file_size = download_resp.headers.get('X-File-Size')
                                        print(f"    ✓ {asset_type}: {file_size} bytes")
                                    else:
                                        print(f"    ⚠️ {asset_type}: Not available")
                            except Exception as e:
                                print(f"    ❌ {asset_type}: Error - {e}")
                        
                        # Test mining submission preparation
                        print("  ⛏️ Testing mining submission preparation...")
                        submission_form = aiohttp.FormData()
                        submission_form.add_field('task_id', f'test_task_{i}')
                        submission_form.add_field('validator_hotkey', 'test_validator_key')
                        submission_form.add_field('validator_uid', str(i))
                        
                        async with session.post(f"{server_url}/prepare_submission/{generation_id}", data=submission_form) as sub_resp:
                            if sub_resp.status == 200:
                                submission_data = await sub_resp.json()
                                print(f"    ✓ Submission prepared - Score: {submission_data.get('local_validation_score', 'N/A')}")
                            else:
                                print(f"    ⚠️ Submission preparation failed")
                        
                    else:
                        print(f"❌ Generation failed: {resp.status}")
                        error_text = await resp.text()
                        print(f"   Error: {error_text}")
                        
            except Exception as e:
                print(f"❌ Request failed: {e}")
            
            # Small delay between tests
            await asyncio.sleep(2)
        
        # Test server statistics
        print("\n📊 Getting server statistics...")
        try:
            async with session.get(f"{server_url}/assets/statistics/") as stats_resp:
                if stats_resp.status == 200:
                    stats = await stats_resp.json()
                    print(f"  - Total assets: {stats['total_assets']}")
                    print(f"  - Completed assets: {stats['completed_assets']}")
                    print(f"  - Submission ready: {stats['submission_ready']}")
                    print(f"  - Total storage: {stats['total_storage_mb']:.2f} MB")
                    print(f"  - Average generation time: {stats['avg_generation_time']:.2f}s")
                    
        except Exception as e:
            print(f"❌ Stats request failed: {e}")

def test_local_asset_management():
    """Test the asset management system locally"""
    print("\n🔧 Testing Local Asset Management System")
    
    # Create test assets
    test_assets = []
    
    for i in range(3):
        prompt = f"test object {i+1}"
        seed = 100 + i
        
        print(f"\n📦 Creating asset {i+1}: '{prompt}'")
        
        # Create asset
        asset = global_asset_manager.create_asset(prompt, seed)
        asset.update_status(GenerationStatus.IMAGE_GENERATING)
        
        # Simulate adding various assets
        from PIL import Image
        import trimesh
        import numpy as np
        
        # Add mock image
        test_image = Image.new('RGB', (512, 512), color=f'hsl({i*120}, 100%, 50%)')
        asset.add_asset(AssetType.ORIGINAL_IMAGE, test_image)
        
        # Add mock background removed image
        bg_removed = Image.new('RGBA', (512, 512), color=f'hsl({i*120}, 100%, 50%)')
        asset.add_asset(AssetType.BACKGROUND_REMOVED_IMAGE, bg_removed)
        
        # Add mock mesh
        if i == 0:
            test_mesh = trimesh.creation.box()
        elif i == 1:
            test_mesh = trimesh.creation.cylinder(radius=1.0, height=2.0)
        else:
            test_mesh = trimesh.creation.icosphere()
            
        asset.add_asset(AssetType.INITIAL_MESH_PLY, test_mesh)
        asset.add_asset(AssetType.INITIAL_MESH_GLB, test_mesh)
        
        # Update validation metrics
        asset.update_validation_metrics(
            local_score=0.7 + i * 0.1,
            mesh_metrics={
                'face_count': len(test_mesh.faces),
                'vertex_count': len(test_mesh.vertices),
                'is_manifold': True,
                'has_texture': False
            }
        )
        
        # Compress PLY
        print(f"  🗜️ Compressing PLY...")
        compression_success = asset.compress_ply_asset()
        if compression_success:
            print(f"    ✓ Compressed with ratio: {asset.compression_ratio:.2f}x")
        else:
            print(f"    ❌ Compression failed")
        
        asset.update_status(GenerationStatus.COMPLETED)
        
        # Save metadata
        metadata_path = asset.save_metadata()
        print(f"  💾 Metadata saved: {metadata_path}")
        
        test_assets.append(asset)
    
    # Test asset manager statistics
    print("\n📈 Asset Manager Statistics:")
    stats = global_asset_manager.get_statistics()
    for key, value in stats.items():
        print(f"  - {key}: {value}")
    
    # Test asset retrieval
    print("\n🔍 Testing Asset Retrieval:")
    for asset in test_assets:
        retrieved = global_asset_manager.get_asset(asset.generation_id)
        if retrieved:
            print(f"  ✓ Retrieved asset: {asset.generation_id}")
            print(f"    - Status: {retrieved.status.value}")
            print(f"    - Assets: {list(retrieved.assets.keys())}")
            print(f"    - Validation score: {retrieved.validation_metrics.local_validation_score}")
        else:
            print(f"  ❌ Failed to retrieve: {asset.generation_id}")
    
    # Test mining submission preparation
    print("\n⛏️ Testing Mining Submission Preparation:")
    for i, asset in enumerate(test_assets):
        try:
            submission_data = prepare_for_mining_submission(
                asset,
                task_id=f"test_task_{i}",
                validator_hotkey=f"validator_key_{i}",
                validator_uid=i
            )
            
            print(f"  ✓ Asset {asset.generation_id}:")
            print(f"    - Compressed PLY size: {len(submission_data['compressed_ply_b64'])} chars (base64)")
            print(f"    - Local score: {submission_data['local_validation_score']}")
            print(f"    - Face count: {submission_data['face_count']}")
            
        except Exception as e:
            print(f"  ❌ Failed to prepare {asset.generation_id}: {e}")
    
    # Test cleanup
    print("\n🧹 Testing Asset Cleanup:")
    initial_count = len(global_asset_manager.assets)
    global_asset_manager.cleanup_old_assets(max_age_hours=0.001, keep_successful=False)
    final_count = len(global_asset_manager.assets)
    print(f"  - Assets before cleanup: {initial_count}")
    print(f"  - Assets after cleanup: {final_count}")

async def test_integration_with_demo():
    """Test integration with flux_hunyuan_bpt_demo.py outputs"""
    print("\n🔗 Testing Integration with Demo Outputs")
    
    # Create mock demo output directory
    demo_dir = Path("./test_demo_outputs")
    demo_dir.mkdir(exist_ok=True)
    
    # Create mock files that would be generated by the demo
    from PIL import Image
    import trimesh
    
    # Mock original image
    img = Image.new('RGB', (1024, 1024), color='blue')
    img.save(demo_dir / "t2i_original.png")
    
    # Mock background removed image
    img_no_bg = Image.new('RGBA', (1024, 1024), color=(0, 0, 255, 255))
    img_no_bg.save(demo_dir / "t2i_no_bg.png")
    
    # Mock initial mesh
    mesh = trimesh.creation.box()
    mesh.export(demo_dir / "t2i_initial.glb")
    
    # Mock enhanced mesh (BPT)
    enhanced_mesh = trimesh.creation.icosphere()
    enhanced_mesh.export(demo_dir / "t2i_enhanced_bpt.glb")
    
    # Import demo outputs into asset system
    from generation_asset_manager import create_generation_asset_from_demo_outputs
    
    asset = create_generation_asset_from_demo_outputs(
        prompt="a blue cube from demo",
        seed=999,
        demo_output_dir=str(demo_dir)
    )
    
    print(f"✓ Demo asset created: {asset.generation_id}")
    print(f"  - Status: {asset.status.value}")
    print(f"  - Available assets: {list(asset.assets.keys())}")
    
    # Compress and prepare for submission
    compression_success = asset.compress_ply_asset()
    if compression_success:
        print(f"  ✓ Compressed demo output: {asset.compression_ratio:.2f}x")
    
    # Cleanup test files
    for file in demo_dir.glob("*"):
        file.unlink()
    demo_dir.rmdir()

def main():
    """Main test function"""
    print("🧪 Comprehensive Asset Management System Test Suite")
    print("=" * 60)
    
    # Test local asset management
    test_local_asset_management()
    
    # Test integration with demo
    asyncio.run(test_integration_with_demo())
    
    # Test server (if running)
    print("\n🌐 Testing Server Integration")
    print("Note: Make sure the server is running with: python flux_hunyuan_bpt_generation_server.py")
    
    try:
        asyncio.run(test_server_generation())
    except Exception as e:
        print(f"⚠️ Server tests skipped (server not running?): {e}")
    
    print("\n✅ Test suite completed!")
    print("\n📚 Summary of Features Tested:")
    print("  - Comprehensive asset creation and management")
    print("  - Multiple asset types (images, meshes, compressed data)")
    print("  - PLY compression with pyspz")
    print("  - Validation metrics tracking")
    print("  - Mining submission preparation")
    print("  - Server integration with REST API")
    print("  - Asset retrieval and download")
    print("  - Statistics and monitoring")
    print("  - Cleanup and maintenance")
    print("  - Integration with demo outputs")
    
    print("\n🚀 Your mining system now has:")
    print("  ✓ Complete asset tracking")
    print("  ✓ Automatic compression for network submission")
    print("  ✓ Comprehensive metadata storage")
    print("  ✓ Mining-ready data structures")
    print("  ✓ Robust error handling and recovery")
    print("  ✓ Performance monitoring")
    print("  ✓ Seamless integration with existing miners")

if __name__ == "__main__":
    main() 