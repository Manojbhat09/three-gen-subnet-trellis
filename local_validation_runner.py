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

async def run_generation(session: aiohttp.ClientSession, prompt: str, seed: int) -> tuple[bytes | None, float, int | None]:
    """Calls the generation server and returns the PLY bytes and generation time."""
    print_status(f"Requesting generation for prompt: '{prompt}' (Seed: {seed})", success=True)
    payload = aiohttp.FormData()
    payload.add_field('prompt', prompt)
    payload.add_field('seed', str(seed))
    
    start_time = time.time()
    try:
        async with session.post(GENERATION_SERVER_URL, data=payload, timeout=300) as response:
            gen_time = time.time() - start_time
                if response.status == 200:
                ply_bytes = await response.read()
                response_seed = int(response.headers.get("X-Seed", seed))
                print_status(f"Generation successful in {gen_time:.2f}s. Received {len(ply_bytes)} bytes.", success=True)
                return ply_bytes, gen_time, response_seed
                else:
                    error_text = await response.text()
                print_status(f"Generation failed. Status: {response.status}, Error: {error_text}", success=False)
                return None, gen_time, None
    except asyncio.TimeoutError:
        gen_time = time.time() - start_time
        print_status(f"Generation timed out after {gen_time:.2f}s", success=False)
        return None, gen_time, None
    except Exception as e:
        gen_time = time.time() - start_time
        print_status(f"An error occurred during generation: {e}", success=False)
        return None, gen_time, None

async def run_validation(session: aiohttp.ClientSession, prompt: str, ply_bytes: bytes) -> dict | None:
    """Calls the validation server and returns the validation result."""
    import base64
    print_status("Requesting local validation...", success=True)
    
    # We send uncompressed base64 data
        payload = {
            "prompt": prompt,
        "data": base64.b64encode(ply_bytes).decode('utf-8'),
        "compression": 0, # 0 = None
        "generate_preview": True # Request preview images
    }
    
    start_time = time.time()
    try:
        async with session.post(VALIDATION_SERVER_URL, json=payload, timeout=120) as response:
            val_time = time.time() - start_time
                if response.status == 200:
                result = await response.json()
                print_status(f"Validation successful in {val_time:.2f}s. Score: {result.get('score', 0.0):.4f}", success=True)
                return result
                else:
                    error_text = await response.text()
                print_status(f"Validation failed. Status: {response.status}, Error: {error_text}", success=False)
        return None
    except Exception as e:
        print_status(f"An error occurred during validation: {e}", success=False)
        return None

def save_asset(prompt: str, ply_bytes: bytes, validation_result: dict, seed: int) -> str:
    """Saves the generated PLY file and validation results."""
    output_path = Path(OUTPUT_DIR)
    output_path.mkdir(exist_ok=True)
    
    score = validation_result.get('score', 0.0)
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    
    # Sanitize prompt for filename
    sanitized_prompt = "".join([c if c.isalnum() else "_" for c in prompt])[:50]
    
    # Save PLY file
    ply_filename = f"{timestamp}_{sanitized_prompt}_seed{seed}_score{score:.2f}.ply"
    ply_filepath = output_path / ply_filename
    with open(ply_filepath, "wb") as f:
        f.write(ply_bytes)
    print_status(f"Saved generated model to: {ply_filepath}", success=True)
    
    # Save validation report
    report_filename = f"{timestamp}_{sanitized_prompt}_seed{seed}_report.json"
    report_filepath = output_path / report_filename
    import json
    with open(report_filepath, "w") as f:
        json.dump(validation_result, f, indent=4)
    print_status(f"Saved validation report to: {report_filepath}", success=True)

    # Save preview images
    if validation_result.get('preview_images'):
        import base64
        image_dir = output_path / f"{timestamp}_{sanitized_prompt}_seed{seed}_previews"
        image_dir.mkdir(exist_ok=True)
        for i, img_b64 in enumerate(validation_result['preview_images']):
            img_bytes = base64.b64decode(img_b64)
            img_filepath = image_dir / f"view_{i+1}.png"
            with open(img_filepath, "wb") as f:
                f.write(img_bytes)
        print_status(f"Saved {len(validation_result['preview_images'])} preview images to: {image_dir}", success=True)
        
    return str(ply_filepath)

async def main():
    parser = argparse.ArgumentParser(description="Run local generation and validation for a single prompt.")
    parser.add_argument("prompt", type=str, help="The text prompt to generate a 3D model from.")
    parser.add_argument("--seed", type=int, default=None, help="A specific seed to use for generation. Random if not set.")
    parser.add_argument("--no-save", action="store_true", help="If set, the generated asset will not be saved to disk.")
    
    args = parser.parse_args()

    seed = args.seed if args.seed is not None else int(time.time())

    print_header("Local Validation Runner")
    print(f"Prompt: {args.prompt}")
    print(f"Seed: {seed}")
    print("-" * 60)

    async with aiohttp.ClientSession() as session:
        # Step 1: Generation
        print_header("Step 1: Generation")
        ply_bytes, gen_time, final_seed = await run_generation(session, args.prompt, seed)
        
        if ply_bytes is None:
            print_status("Aborting due to generation failure.", success=False)
            return

        # Step 2: Validation
        print_header("Step 2: Validation")
        validation_result = await run_validation(session, args.prompt, ply_bytes)

        if validation_result is None:
            print_status("Aborting due to validation failure.", success=False)
            return
            
        # Step 3: Save results
        if not args.no_save:
            print_header("Step 3: Saving Assets")
            save_asset(args.prompt, ply_bytes, validation_result, final_seed)
        
        print_header("Run Complete")

if __name__ == "__main__":
    asyncio.run(main()) 