#!/usr/bin/env python3
# Subnet 17 (404-GEN) - Local Validation Runner
# Purpose: A simple tool to test the local generation and validation servers.

import asyncio
import aiohttp
import argparse
import time
import os
from pathlib import Path

# --- Configuration ---
GENERATION_SERVER_URL = "http://127.0.0.1:8093/generate/"
VALIDATION_SERVER_URL = "http://127.0.0.1:8094/validate_txt_to_3d_ply/"
OUTPUT_DIR = "locally_validated_assets"

# --- Helper Functions ---
def print_header(title):
    print("=" * 60)
    print(f" {title}")
    print("=" * 60)

def print_status(message, success=True):
    symbol = "✓" if success else "✗"
    print(f"[{symbol}] {message}")

async def run_generation(session: aiohttp.ClientSession, prompt: str, seed: int) -> tuple:
    """Calls the generation server and returns the PLY bytes and generation time."""
    print_status(f"Requesting generation for prompt: '{prompt}' (seed: {seed})")
    
    start_time = time.time()
    try:
    payload = {"prompt": prompt}
        async with session.post(GENERATION_SERVER_URL, data=payload, timeout=300) as response:
                if response.status == 200:
                ply_bytes = await response.read()
                generation_time = time.time() - start_time
                print_status(f"Generation completed in {generation_time:.2f}s, PLY size: {len(ply_bytes)} bytes")
                return ply_bytes, generation_time, response.status
                else:
                    error_text = await response.text()
                print_status(f"Generation failed: HTTP {response.status} - {error_text}", success=False)
                return None, time.time() - start_time, response.status
    except Exception as e:
        generation_time = time.time() - start_time
        print_status(f"Generation exception: {e}", success=False)
        return None, generation_time, None

async def run_validation(session: aiohttp.ClientSession, prompt: str, ply_bytes: bytes) -> tuple:
    """Calls the validation server and returns the validation score."""
    print_status("Running local validation...")
    
    try:
        import pyspz
        import base64
        
        # Compress the PLY data
        compressed_data = pyspz.compress(ply_bytes)
        base64_data = base64.b64encode(compressed_data).decode('utf-8')
        
        payload = {
            "prompt": prompt,
            "data": base64_data,
            "compression": 2,
            "data_ver": 0
        }
        
        start_time = time.time()
        async with session.post(VALIDATION_SERVER_URL, json=payload, timeout=60) as response:
            validation_time = time.time() - start_time
                if response.status == 200:
                result = await response.json()
                score = result.get("score", 0.0)
                print_status(f"Validation completed in {validation_time:.2f}s, Score: {score:.4f}")
                return score, validation_time, response.status
                else:
                    error_text = await response.text()
                print_status(f"Validation failed: HTTP {response.status} - {error_text}", success=False)
                return -1.0, validation_time, response.status
        
    except Exception as e:
        print_status(f"Validation exception: {e}", success=False)
        return -1.0, 0.0, None

def save_ply_file(ply_bytes: bytes, prompt: str, score: float) -> str:
    """Save PLY file to local directory."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Create safe filename
    safe_prompt = "".join(c for c in prompt if c.isalnum() or c in (' ', '-', '_')).rstrip()
    safe_prompt = safe_prompt.replace(' ', '_')[:50]  # Limit length
    
    timestamp = int(time.time())
    filename = f"{safe_prompt}_{score:.3f}_{timestamp}.ply"
    filepath = Path(OUTPUT_DIR) / filename
    
    with open(filepath, 'wb') as f:
        f.write(ply_bytes)
    
    print_status(f"PLY file saved: {filepath}")
    return str(filepath)

async def main():
    parser = argparse.ArgumentParser(description="Local Validation Runner for Subnet 17")
    parser.add_argument("prompt", help="Text prompt for 3D generation")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for generation")
    parser.add_argument("--save", action="store_true", help="Save generated PLY file")
    
    args = parser.parse_args()
    
    print_header("Subnet 17 Local Validation Runner")
    print(f"Prompt: {args.prompt}")
    print(f"Seed: {args.seed}")
    print()
    
    async with aiohttp.ClientSession() as session:
        # Step 1: Generate 3D model
        print_status("Step 1: Generating 3D model")
        ply_bytes, gen_time, gen_status = await run_generation(session, args.prompt, args.seed)
        
        if ply_bytes is None:
            print_status("Generation failed, cannot proceed to validation", success=False)
            return 1
        
        # Step 2: Validate model
        print_status("Step 2: Validating 3D model")
        score, val_time, val_status = await run_validation(session, args.prompt, ply_bytes)
        
        if score < 0:
            print_status("Validation failed", success=False)
            return 1
        
        # Step 3: Save file if requested
        if args.save:
            print_status("Step 3: Saving PLY file")
            filepath = save_ply_file(ply_bytes, args.prompt, score)
        
        # Final results
        print()
        print_header("RESULTS")
        print(f"Generation Time: {gen_time:.2f}s")
        print(f"Validation Time: {val_time:.2f}s")
        print(f"Total Time: {gen_time + val_time:.2f}s")
        print(f"Validation Score: {score:.4f}")
        print(f"PLY File Size: {len(ply_bytes):,} bytes")
        
        if score >= 0.8:
            print_status("EXCELLENT quality score!", success=True)
        elif score >= 0.7:
            print_status("GOOD quality score", success=True)
        elif score >= 0.6:
            print_status("ACCEPTABLE quality score", success=True)
        else:
            print_status("LOW quality score - consider adjusting generation parameters", success=False)
        
        return 0

if __name__ == "__main__":
    exit_code = asyncio.run(main()) 