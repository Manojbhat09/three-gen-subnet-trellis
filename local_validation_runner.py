#!/usr/bin/env python3
# Subnet 17 (404-GEN) - Local Validation Runner
# Purpose: A simple tool to test the local generation and validation servers.

import asyncio
import aiohttp
import argparse
import time
import os
import sys
from pathlib import Path

# Add current directory to path for asset manager import
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from generation_asset_manager import (
        global_asset_manager, AssetType, GenerationStatus,
        prepare_for_mining_submission
    )
    ASSET_MANAGER_AVAILABLE = True
except ImportError:
    print("⚠️ Asset manager not available - running in basic mode")
    ASSET_MANAGER_AVAILABLE = False

# --- Configuration ---
GENERATION_SERVER_URL = "http://127.0.0.1:8093/generate/"
VALIDATION_SERVER_URL = "http://127.0.0.1:8094/validate_txt_to_3d_ply/"
OUTPUT_DIR = "locally_validated_assets"

# Enhanced server URLs (try new asset-aware server first)
ENHANCED_GENERATION_URL = "http://127.0.0.1:8095/generate/"

# --- Helper Functions ---
def print_header(title):
    print("=" * 60)
    print(f" {title}")
    print("=" * 60)

def print_status(message, success=True):
    symbol = "✓" if success else "✗"
    print(f"[{symbol}] {message}")

async def test_server_connectivity(session: aiohttp.ClientSession) -> dict:
    """Test connectivity to available servers"""
    print_status("Testing server connectivity...")
    
    servers = {
        "enhanced_generation": ENHANCED_GENERATION_URL.replace("/generate/", "/health/"),
        "generation": GENERATION_SERVER_URL.replace("/generate/", "/health/"),
        "validation": VALIDATION_SERVER_URL.replace("/validate_txt_to_3d_ply/", "/version/")
    }
    
    connectivity = {}
    
    for name, url in servers.items():
        try:
            async with session.get(url, timeout=5) as response:
                if response.status == 200:
                    connectivity[name] = True
                    print_status(f"{name.replace('_', ' ').title()} server: Online", success=True)
                else:
                    connectivity[name] = False
                    print_status(f"{name.replace('_', ' ').title()} server: Error {response.status}", success=False)
        except Exception as e:
            connectivity[name] = False
            print_status(f"{name.replace('_', ' ').title()} server: Offline ({e})", success=False)
    
    return connectivity

async def run_enhanced_generation(session: aiohttp.ClientSession, prompt: str, seed: int) -> tuple:
    """Try the enhanced generation server with asset management"""
    print_status(f"Requesting enhanced generation for prompt: '{prompt}' (seed: {seed})")
    
    start_time = time.time()
    try:
        # Use FormData for enhanced server
        form_data = aiohttp.FormData()
        form_data.add_field('prompt', prompt)
        form_data.add_field('seed', str(seed))
        form_data.add_field('use_bpt', 'false')
        form_data.add_field('return_compressed', 'true')
        
        async with session.post(ENHANCED_GENERATION_URL, data=form_data, timeout=300) as response:
            generation_time = time.time() - start_time
            
            if response.status == 200:
                ply_bytes = await response.read()
                generation_id = response.headers.get('X-Generation-ID')
                compression_ratio = response.headers.get('X-Compression-Ratio', '1.0')
                face_count = response.headers.get('X-Face-Count', '0')
                
                print_status(f"Enhanced generation completed in {generation_time:.2f}s")
                print_status(f"PLY size: {len(ply_bytes)} bytes, Compression: {compression_ratio}x, Faces: {face_count}")
                
                return ply_bytes, generation_time, response.status, {
                    'generation_id': generation_id,
                    'compression_ratio': float(compression_ratio),
                    'face_count': int(face_count),
                    'enhanced': True
                }
            else:
                error_text = await response.text()
                print_status(f"Enhanced generation failed: HTTP {response.status} - {error_text}", success=False)
                return None, generation_time, response.status, {}
                
    except Exception as e:
        generation_time = time.time() - start_time
        print_status(f"Enhanced generation exception: {e}", success=False)
        return None, generation_time, None, {}

async def run_generation(session: aiohttp.ClientSession, prompt: str, seed: int) -> tuple:
    """Calls the generation server and returns the PLY bytes and generation time."""
    print_status(f"Requesting generation for prompt: '{prompt}' (seed: {seed})")
    
    start_time = time.time()
    try:
        payload = {"prompt": prompt, "seed": seed}
        async with session.post(GENERATION_SERVER_URL, data=payload, timeout=300) as response:
            generation_time = time.time() - start_time
            
            if response.status == 200:
                ply_bytes = await response.read()
                print_status(f"Generation completed in {generation_time:.2f}s, PLY size: {len(ply_bytes)} bytes")
                return ply_bytes, generation_time, response.status, {'enhanced': False}
            else:
                error_text = await response.text()
                print_status(f"Generation failed: HTTP {response.status} - {error_text}", success=False)
                return None, generation_time, response.status, {}
                
    except Exception as e:
        generation_time = time.time() - start_time
        print_status(f"Generation exception: {e}", success=False)
        return None, generation_time, None, {}

async def run_validation(session: aiohttp.ClientSession, prompt: str, ply_bytes: bytes) -> tuple:
    """Calls the validation server and returns the validation score."""
    print_status("Running local validation...")
    
    if not ply_bytes:
        print_status("PLY bytes are empty, skipping validation.", success=False)
        return -1.0, 0.0, None, {}

    try:
        import base64
        
        # The validation server expects a specific JSON structure.
        # We will not use pyspz for the runner, as the server can handle raw PLY.
        
        base64_data = base64.b64encode(ply_bytes).decode('utf-8')
        
        payload = {
            "prompt": prompt,
            "data": base64_data,
            "compression": 0, # 0 for no compression
            "data_ver": 0,
            "generate_preview": True,
            "preview_score_threshold": 0.7 
        }
        
        start_time = time.time()
        async with session.post(VALIDATION_SERVER_URL, json=payload, timeout=120) as response:
            validation_time = time.time() - start_time
            
            if response.status == 200:
                result = await response.json()
                score = result.get("score", 0.0)
                print_status(f"Validation completed in {validation_time:.2f}s, Score: {score:.4f}")
                return score, validation_time, response.status, result
            else:
                error_text = await response.text()
                print_status(f"Validation failed: HTTP {response.status} - {error_text}", success=False)
                return -1.0, validation_time, response.status, {}
        
    except Exception as e:
        print_status(f"Validation exception: {e}", success=False)
        return -1.0, 0.0, None, {}

def save_ply_file(ply_bytes: bytes, prompt: str, score: float, metadata: dict = None) -> str:
    """Save PLY file to local directory with enhanced metadata."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Create safe filename
    safe_prompt = "".join(c for c in prompt if c.isalnum() or c in (' ', '-', '_')).rstrip()
    safe_prompt = safe_prompt.replace(' ', '_')[:50]  # Limit length
    
    timestamp = int(time.time())
    filename = f"{safe_prompt}_{score:.3f}_{timestamp}.ply"
    filepath = Path(OUTPUT_DIR) / filename
    
    with open(filepath, 'wb') as f:
        f.write(ply_bytes)
    
    # Save metadata if available
    if metadata:
        metadata_file = filepath.with_suffix('.json')
        import json
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        print_status(f"Metadata saved: {metadata_file}")
    
    print_status(f"PLY file saved: {filepath}")
    return str(filepath)

async def download_additional_assets(session: aiohttp.ClientSession, generation_id: str) -> dict:
    """Download additional assets from enhanced server"""
    if not generation_id:
        return {}
    
    print_status("Downloading additional assets...")
    
    assets = {}
    asset_types = ['original_image', 'background_removed_image', 'initial_mesh_glb']
    
    for asset_type in asset_types:
        try:
            url = f"{ENHANCED_GENERATION_URL.replace('/generate/', '')}/generate/{generation_id}/download/{asset_type}"
            async with session.get(url, timeout=30) as response:
                if response.status == 200:
                    data = await response.read()
                    assets[asset_type] = {
                        'data': data,
                        'size': len(data),
                        'type': asset_type
                    }
                    print_status(f"Downloaded {asset_type}: {len(data)} bytes")
                else:
                    print_status(f"Could not download {asset_type}: {response.status}", success=False)
        except Exception as e:
            print_status(f"Error downloading {asset_type}: {e}", success=False)
    
    return assets

def save_additional_assets(assets: dict, prompt: str, timestamp: int):
    """Save additional assets to files"""
    if not assets:
        return
    
    safe_prompt = "".join(c for c in prompt if c.isalnum() or c in (' ', '-', '_')).rstrip()
    safe_prompt = safe_prompt.replace(' ', '_')[:50]
    
    for asset_type, asset_data in assets.items():
        # Determine file extension
        if 'image' in asset_type:
            ext = '.png'
        elif 'glb' in asset_type:
            ext = '.glb'
        else:
            ext = '.bin'
        
        filename = f"{safe_prompt}_{asset_type}_{timestamp}{ext}"
        filepath = Path(OUTPUT_DIR) / filename
        
        with open(filepath, 'wb') as f:
            f.write(asset_data['data'])
        
        print_status(f"Saved {asset_type}: {filepath}")

async def test_mining_integration(generation_id: str, prompt: str, score: float):
    """Test mining submission preparation if asset manager is available"""
    if not ASSET_MANAGER_AVAILABLE or not generation_id:
        print_status("Skipping mining integration test (Asset manager not available or no generation ID)")
        return
    
    print_status("Testing mining integration...")
    
    # In a real scenario, we'd query the server for the asset.
    # For this local test, we recognize that the asset manager is on the server,
    # so we cannot directly access it. This test is a conceptual check.
    print_status("Note: Local runner cannot directly access server's asset manager.", success=True)
    print_status(f"  - Conceptual check for Generation ID: {generation_id}", success=True)
    print_status(f"  - This would be prepared for submission with score: {score:.4f}", success=True)

async def main():
    parser = argparse.ArgumentParser(description="Enhanced Local Validation Runner for Subnet 17")
    parser.add_argument("prompt", help="Text prompt for 3D generation")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for generation")
    parser.add_argument("--save", action="store_true", help="Save generated assets")
    parser.add_argument("--save-all", action="store_true", help="Save all available assets")
    parser.add_argument("--test-mining", action="store_true", help="Test mining integration")
    parser.add_argument("--server", choices=['enhanced', 'basic', 'auto'], default='auto',
                       help="Choose server type (auto tries enhanced first)")
    
    args = parser.parse_args()
    
    print_header("Enhanced Subnet 17 Local Validation Runner")
    print(f"Prompt: {args.prompt}")
    print(f"Seed: {args.seed}")
    print(f"Asset Manager: {'Available' if ASSET_MANAGER_AVAILABLE else 'Not Available'}")
    print()
    
    async with aiohttp.ClientSession() as session:
        # Test server connectivity
        connectivity = await test_server_connectivity(session)
        print()
        
        # Determine which server to use
        use_enhanced = False
        if args.server == 'enhanced' or (args.server == 'auto' and connectivity.get('enhanced_generation', False)):
            use_enhanced = True
        
        # Step 1: Generate 3D model
        print_status(f"Step 1: Generating 3D model ({'Enhanced' if use_enhanced else 'Basic'} server)")
        
        if use_enhanced:
            ply_bytes, gen_time, gen_status, gen_metadata = await run_enhanced_generation(session, args.prompt, args.seed)
        else:
            ply_bytes, gen_time, gen_status, gen_metadata = await run_generation(session, args.prompt, args.seed)
        
        if ply_bytes is None:
            print_status("Generation failed, cannot proceed to validation", success=False)
            return 1
        
        # Step 2: Validate model
        print_status("Step 2: Validating 3D model")
        score, val_time, val_status, val_result = await run_validation(session, args.prompt, ply_bytes)
        
        if score < 0:
            print_status("Validation failed", success=False)
            return 1
        
        # Step 3: Download additional assets if using enhanced server
        additional_assets = {}
        if use_enhanced and gen_metadata.get('generation_id'):
            additional_assets = await download_additional_assets(session, gen_metadata['generation_id'])
        
        # Step 4: Save files if requested
        timestamp = int(time.time())
        if args.save or args.save_all:
            print_status("Step 4: Saving assets")
            
            # Prepare metadata
            metadata = {
                'prompt': args.prompt,
                'seed': args.seed,
                'generation_time': gen_time,
                'validation_time': val_time,
                'validation_score': score,
                'generation_metadata': gen_metadata,
                'validation_result': val_result,
                'timestamp': timestamp
            }
            
            # Save PLY file
            filepath = save_ply_file(ply_bytes, args.prompt, score, metadata)
            
            # Save additional assets if requested
            if args.save_all and additional_assets:
                save_additional_assets(additional_assets, args.prompt, timestamp)
        
        # Step 5: Test mining integration if requested
        if args.test_mining and gen_metadata.get('generation_id'):
            await test_mining_integration(gen_metadata['generation_id'], args.prompt, score)
        
        # Final results
        print()
        print_header("RESULTS")
        print(f"Generation Time: {gen_time:.2f}s")
        print(f"Validation Time: {val_time:.2f}s")
        print(f"Total Time: {gen_time + val_time:.2f}s")
        print(f"Validation Score: {score:.4f}")
        print(f"PLY File Size: {len(ply_bytes):,} bytes")
        
        if gen_metadata.get('compression_ratio'):
            print(f"Compression Ratio: {gen_metadata['compression_ratio']:.2f}x")
        if gen_metadata.get('face_count'):
            print(f"Face Count: {gen_metadata['face_count']:,}")
        
        # Quality assessment
        print()
        if score >= 0.8:
            print_status("EXCELLENT quality score! 🎉", success=True)
        elif score >= 0.7:
            print_status("GOOD quality score ✨", success=True)
        elif score >= 0.6:
            print_status("ACCEPTABLE quality score ⭐", success=True)
        else:
            print_status("LOW quality score - consider adjusting generation parameters ⚠️", success=False)
        
        # Performance assessment
        total_time = gen_time + val_time
        if total_time < 10:
            print_status("FAST generation time! 🚀", success=True)
        elif total_time < 30:
            print_status("Good generation time ⏱️", success=True)
        else:
            print_status("Slow generation time - check system performance ⏳", success=False)
        
        return 0

if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code) 