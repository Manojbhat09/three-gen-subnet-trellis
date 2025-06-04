#!/usr/bin/env python3
# Subnet 17 (404-GEN) - Local Compete Validation Script
# Purpose: Validates multiple pre-generated (and pre-compressed) PLY files
#          for a single prompt using a local validation endpoint to compare scores.

import asyncio
import base64
import json
import os
import sys
import traceback
from datetime import datetime
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass
from pathlib import Path

import aiohttp
import bittensor as bt

# --- Configuration Constants ---
LOCAL_VALIDATION_ENDPOINT_URL: str = "http://127.0.0.1:8094/validate_txt_to_3d_ply/"
MAX_RETRIES: int = 3
RETRY_DELAY: float = 2.0
VALIDATION_TIMEOUT: float = 60.0
MIN_FILE_SIZE: int = 100  # Minimum expected file size in bytes
OUTPUT_DIR: str = "./validation_results"
LOG_DIR: str = "./logs"

# --- Data Structures ---
@dataclass
class ValidationResult:
    label: str
    score: float
    file_path: str
    error: Optional[str] = None
    retry_count: int = 0

# --- Helper Functions ---
def setup_logging():
    """Configure logging with file and console handlers."""
    os.makedirs(LOG_DIR, exist_ok=True)
    log_file = os.path.join(LOG_DIR, f"compete_validation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    bt.logging.config(
        logging_dir=LOG_DIR,
        logging_file=log_file,
        debug=True,
        trace=False
    )

def validate_file_path(file_path: str) -> Tuple[bool, Optional[str]]:
    """Validates if a file exists and has valid size."""
    try:
        path = Path(file_path)
        if not path.exists():
            return False, f"File does not exist: {file_path}"
        if not path.is_file():
            return False, f"Path is not a file: {file_path}"
        if path.stat().st_size < MIN_FILE_SIZE:
            return False, f"File too small ({path.stat().st_size} bytes): {file_path}"
        return True, None
    except Exception as e:
        return False, f"Error validating file: {str(e)}"

async def validate_single_file(
    session: aiohttp.ClientSession,
    prompt: str,
    file_path: str,
    miner_label: str,
    retry_count: int = 0
) -> ValidationResult:
    """Validates a single compressed PLY file against a prompt with retries."""
    bt.logging.info(f"Validating '{miner_label}' ({file_path}) for prompt: '{prompt[:70]}...'")
    
    try:
        # Validate file before processing
        is_valid, error_msg = validate_file_path(file_path)
        if not is_valid:
            return ValidationResult(
                label=miner_label,
                score=-2.0,
                file_path=file_path,
                error=error_msg
            )

        with open(file_path, "rb") as f:
            compressed_ply_bytes = f.read()
        
        base64_compressed_data = base64.b64encode(compressed_ply_bytes).decode('utf-8')
        
        payload = {
            "prompt": prompt,
            "data": base64_compressed_data,
            "compression": 2,  # 2 for SPZ
            "data_ver": 0,
            "generate_preview": False 
        }
        
        async with session.post(LOCAL_VALIDATION_ENDPOINT_URL, json=payload, timeout=VALIDATION_TIMEOUT) as response:
            if response.status == 200:
                validation_result = await response.json()
                score = validation_result.get("score", 0.0)
                
                # Validate score
                if not 0 <= score <= 1:
                    raise ValueError(f"Invalid score received: {score}")
                
                bt.logging.success(f"Score for '{miner_label}': {score:.4f}")
                return ValidationResult(
                    label=miner_label,
                    score=score,
                    file_path=file_path
                )
            else:
                error_text = await response.text()
                error_msg = f"Validation HTTP error: Status {response.status}, {error_text}"
                bt.logging.error(f"Error for '{miner_label}': {error_msg}")
                
                if retry_count < MAX_RETRIES:
                    await asyncio.sleep(RETRY_DELAY * (retry_count + 1))
                    return await validate_single_file(
                        session, prompt, file_path, miner_label, retry_count + 1
                    )
                
                return ValidationResult(
                    label=miner_label,
                    score=-1.0,
                    file_path=file_path,
                    error=error_msg,
                    retry_count=retry_count
                )
                
    except FileNotFoundError:
        error_msg = f"File not found: {file_path}"
        bt.logging.error(f"Error for '{miner_label}': {error_msg}")
        return ValidationResult(
            label=miner_label,
            score=-2.0,
            file_path=file_path,
            error=error_msg
        )
    except Exception as e:
        error_msg = f"Validation exception: {str(e)}"
        bt.logging.error(f"Error for '{miner_label}': {error_msg}")
        
        if retry_count < MAX_RETRIES:
            await asyncio.sleep(RETRY_DELAY * (retry_count + 1))
            return await validate_single_file(
                session, prompt, file_path, miner_label, retry_count + 1
            )
        
        return ValidationResult(
            label=miner_label,
            score=-1.0,
            file_path=file_path,
            error=error_msg,
            retry_count=retry_count
        )

def save_results(prompt: str, results: List[ValidationResult]):
    """Saves validation results to a JSON file."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = os.path.join(OUTPUT_DIR, f"validation_results_{timestamp}.json")
    
    output_data = {
        "prompt": prompt,
        "timestamp": timestamp,
        "results": [
            {
                "label": r.label,
                "score": r.score,
                "file_path": r.file_path,
                "error": r.error,
                "retry_count": r.retry_count
            }
            for r in results
        ]
    }
    
    with open(output_file, "w") as f:
        json.dump(output_data, f, indent=2)
    
    bt.logging.info(f"Results saved to: {output_file}")

def print_results(prompt: str, results: List[ValidationResult]):
    """Prints validation results in a formatted way."""
    bt.logging.info("\n" + "="*50)
    bt.logging.info("Validation Results")
    bt.logging.info("="*50)
    bt.logging.info(f"Prompt: {prompt}")
    bt.logging.info("-"*50)
    
    # Sort results by score (descending)
    sorted_results = sorted(results, key=lambda x: x.score, reverse=True)
    
    for result in sorted_results:
        if result.score < 0:
            status = "ERROR" if result.score == -1.0 else "FILE ERROR"
            bt.logging.error(f"{result.label}: {status}")
            if result.error:
                bt.logging.error(f"  Error: {result.error}")
            if result.retry_count > 0:
                bt.logging.error(f"  Retries: {result.retry_count}")
        else:
            bt.logging.info(f"{result.label}: Score = {result.score:.4f}")
    
    bt.logging.info("="*50)

async def main_compete_validation():
    """Main function for running the compete validation."""
    setup_logging()
    bt.logging.info("--- Subnet 17 Local Compete Validation Script ---")

    # Get prompt
    target_prompt = input("\nEnter the prompt for which the models were generated: ").strip()
    if not target_prompt:
        bt.logging.error("Prompt cannot be empty.")
        return

    # Get model submissions
    miner_submissions: List[Tuple[str, str]] = []
    bt.logging.info("\nEnter model submissions (type 'done' when finished):")
    
    while True:
        label = input(f"\nEnter a label for a miner/model (or type 'done' to finish): ").strip()
        if label.lower() == 'done':
            break
            
        file_path = input(f"Enter the full path to the compressed .ply.spz file for '{label}': ").strip()
        
        is_valid, error_msg = validate_file_path(file_path)
        if not is_valid:
            bt.logging.warning(f"Invalid file: {error_msg}")
            continue
            
        miner_submissions.append((label, file_path))

    if not miner_submissions:
        bt.logging.info("No submissions provided. Exiting.")
        return

    bt.logging.info(f"\nStarting validation for {len(miner_submissions)} submissions against prompt: '{target_prompt}'")
    
    # Run validations
    results: List[ValidationResult] = []
    async with aiohttp.ClientSession() as session:
        tasks = [
            validate_single_file(session, target_prompt, fp, lbl)
            for lbl, fp in miner_submissions
        ]
        validation_results = await asyncio.gather(*tasks)
        results.extend(validation_results)

    # Save and print results
    save_results(target_prompt, results)
    print_results(target_prompt, results)

if __name__ == "__main__":
    try:
        asyncio.run(main_compete_validation())
    except KeyboardInterrupt:
        bt.logging.info("\nLocal compete validation stopped by user.")
    except Exception as e:
        bt.logging.critical(f"Critical error: {e}")
        traceback.print_exc()
        sys.exit(1) 