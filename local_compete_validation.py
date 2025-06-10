#!/usr/bin/env python3
# Subnet 17 (404-GEN) - Local Competition Runner
# Purpose: A tool to generate multiple models for the same prompt with different
#          seeds and compare their local validation scores to find the best one.

import asyncio
import aiohttp
import argparse
import time
import os
import base64
import json
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import random

# --- Configuration ---
GENERATION_SERVER_URL = "http://127.0.0.1:8093/generate/"
VALIDATION_SERVER_URL = "http://127.0.0.1:8094/validate_txt_to_3d_ply/"
OUTPUT_DIR = "competition_results"

# --- Data Structures ---
@dataclass
class CompetitionEntry:
    seed: int
    ply_bytes: bytes
    generation_time: float
    validation_score: float = -1.0
    validation_details: Optional[Dict[str, Any]] = None
    filepath: Optional[str] = None

# --- Helper Functions ---
def print_header(title):
    print("\n" + "=" * 70)
    print(f" {title}")
    print("=" * 70)

def print_status(message, success=True):
    symbol = "✓" if success else "✗"
    print(f"[{symbol}] {message}")

async def run_single_generation(
    session: aiohttp.ClientSession, 
    prompt: str, 
    seed: int,
    run_index: int,
    total_runs: int
) -> Optional[CompetitionEntry]:
    """Runs one generation-validation cycle."""
    print_header(f"Run {run_index}/{total_runs} | Seed: {seed}")
    
    # 1. Generation
    payload = aiohttp.FormData()
    payload.add_field('prompt', prompt)
    payload.add_field('seed', str(seed))
    
    gen_start_time = time.time()
    try:
        async with session.post(GENERATION_SERVER_URL, data=payload, timeout=300) as response:
            gen_time = time.time() - gen_start_time
            if response.status != 200:
                error_text = await response.text()
                print_status(f"Generation failed. Status: {response.status}, Error: {error_text}", success=False)
                return None
            
            ply_bytes = await response.read()
            print_status(f"Generation successful in {gen_time:.2f}s.", success=True)
            
            entry = CompetitionEntry(seed=seed, ply_bytes=ply_bytes, generation_time=gen_time)
            
    except Exception as e:
        print_status(f"An error occurred during generation: {e}", success=False)
        return None

    # 2. Validation
    val_payload = {
        "prompt": prompt,
        "data": base64.b64encode(ply_bytes).decode('utf-8'),
        "compression": 0
    }
    try:
        async with session.post(VALIDATION_SERVER_URL, json=val_payload, timeout=120) as response:
            if response.status == 200:
                result = await response.json()
                entry.validation_score = result.get('score', 0.0)
                entry.validation_details = result.get('details', {})
                print_status(f"Validation successful. Score: {entry.validation_score:.4f}", success=True)
            else:
                error_text = await response.text()
                print_status(f"Validation failed. Status: {response.status}, Error: {error_text}", success=False)
    except Exception as e:
        print_status(f"An error occurred during validation: {e}", success=False)

    return entry

def save_best_asset(prompt: str, entry: CompetitionEntry, run_dir: Path):
    """Saves the winning asset and its report."""
    if not entry.ply_bytes:
        return
        
    sanitized_prompt = "".join([c if c.isalnum() else "_" for c in prompt])[:50]
    
    # Save PLY file
    ply_filename = f"BEST_{sanitized_prompt}_seed{entry.seed}_score{entry.validation_score:.2f}.ply"
    ply_filepath = run_dir / ply_filename
    with open(ply_filepath, "wb") as f:
        f.write(entry.ply_bytes)
    entry.filepath = str(ply_filepath)
    print_status(f"Saved best model to: {ply_filepath}", success=True)
    
    # Save validation report
    report_filename = f"BEST_{sanitized_prompt}_seed{entry.seed}_report.json"
    report_filepath = run_dir / report_filename
    with open(report_filepath, "w") as f:
        json.dump(entry.validation_details, f, indent=4)
    print_status(f"Saved best model's report to: {report_filepath}", success=True)


async def main():
    parser = argparse.ArgumentParser(description="Run a competition for a single prompt with multiple seeds.")
    parser.add_argument("prompt", type=str, help="The text prompt to generate a 3D model from.")
    parser.add_argument("-n", "--num_runs", type=int, default=5, help="Number of different seeds to try.")
    parser.add_argument("--save-all", action="store_true", help="Save all generated assets, not just the best one.")
    
    args = parser.parse_args()

    # Create a unique directory for this competition run
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    sanitized_prompt = "".join([c if c.isalnum() else "_" for c in args.prompt])[:50]
    run_dir = Path(OUTPUT_DIR) / f"{timestamp}_{sanitized_prompt}"
    run_dir.mkdir(exist_ok=True, parents=True)

    print_header("Local Competition Runner")
    print(f"Prompt: '{args.prompt}'")
    print(f"Number of Runs: {args.num_runs}")
    print(f"Results will be saved in: {run_dir}")
    print("-" * 70)

    # Generate a list of unique random seeds
    seeds = [random.randint(0, 2**32 - 1) for _ in range(args.num_runs)]
    
    results: List[CompetitionEntry] = []
    
    async with aiohttp.ClientSession() as session:
        for i, seed in enumerate(seeds):
            entry = await run_single_generation(session, args.prompt, seed, i + 1, args.num_runs)
            if entry:
                results.append(entry)

    if not results:
        print_header("Competition Finished - No Successful Runs")
        print("Could not generate or validate any models.")
        return

    # Sort results by validation score (highest first)
    results.sort(key=lambda x: x.validation_score, reverse=True)

    # --- Print Summary ---
    print_header("Competition Results Summary")
    print(f"{'Rank':<5} | {'Seed':<12} | {'Score':<10} | {'Gen Time (s)':<15}")
    print("-" * 70)
    for i, res in enumerate(results):
        rank = i + 1
        print(f"{rank:<5} | {res.seed:<12} | {res.validation_score:<10.4f} | {res.generation_time:<15.2f}")

    # --- Save Assets ---
    print_header("Saving Assets")
    best_entry = results[0]
    
    if best_entry.validation_score > 0:
        print(f"Best score was {best_entry.validation_score:.4f} from seed {best_entry.seed}.")
        save_best_asset(args.prompt, best_entry, run_dir)
    else:
        print("No entries achieved a positive score. Not saving a 'best' asset.")

    if args.save_all:
        print("\nSaving all other assets as requested...")
        for i, entry in enumerate(results):
            if entry.seed != best_entry.seed and entry.ply_bytes:
                 ply_filename = f"rank{i+1}_{sanitized_prompt}_seed{entry.seed}_score{entry.validation_score:.2f}.ply"
                 ply_filepath = run_dir / ply_filename
                 with open(ply_filepath, "wb") as f:
                     f.write(entry.ply_bytes)
                 print_status(f"Saved asset to: {ply_filepath}", success=True)
    
    print_header("Competition Complete")


if __name__ == "__main__":
    asyncio.run(main()) 