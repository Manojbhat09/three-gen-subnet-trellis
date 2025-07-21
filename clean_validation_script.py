#!/usr/bin/env python3

import requests
import torch
import sys
import os
from loguru import logger
import io
from typing import Any
import pyspz

# --- Setup Paths ---
validation_path = os.path.abspath('validation')
if validation_path not in sys.path:
    sys.path.insert(0, validation_path)

from validation.engine.validation_engine import ValidationEngine
from validation.engine.io.ply import PlyLoader
from validation.engine.rendering.renderer import Renderer

# --- Configuration ---
GENERATION_URL = "http://127.0.0.1:8096/generate/"
PROMPT = "a_motorcycle"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def generate_model(prompt):
    logger.info(f"Requesting model for prompt: '{prompt}'...")
    try:
        response = requests.post(
            GENERATION_URL,
            data={"prompt": prompt, "return_compressed": True}
        )
        if response.status_code == 200:
            logger.success(f"Successfully received {len(response.content)} bytes from server")
            return response.content
        else:
            logger.error(f"Failed to generate model. Status code: {response.status_code}")
            return None
    except Exception as e:
        logger.error(f"Error requesting model: {e}")
        return None

def decode_and_load(compressed_data: bytes) -> Any:
    logger.info("Decompressing SPZ data...")
    try:
        decompressed_data = pyspz.decompress(compressed_data)
        logger.info(f"Decompressed to {len(decompressed_data)} bytes")

        logger.info("Loading PLY data...")
        ply_loader = PlyLoader()
        gs_data = ply_loader.from_buffer(io.BytesIO(decompressed_data))
        logger.success("PLY data loaded successfully")
        return gs_data
    except Exception as e:
        logger.error(f"Failed to decompress or load model: {e}")
        return None

def main():
    logger.info("Starting local validation script")
    
    # 1. Generate Model
    logger.info("Step 1: Generating model from server...")
    compressed_model = generate_model(PROMPT)
    if not compressed_model:
        logger.error("Failed to get model from server")
        return

    # 2. Decode and Load PLY
    logger.info("Step 2: Decoding and loading PLY data...")
    gs_data = decode_and_load(compressed_model)
    if gs_data is None:
        logger.error("Failed to decode model")
        return

    # 3. Render Views
    logger.info("Step 3: Rendering views (this may take a while)...")
    try:
        renderer = Renderer()
        rendered_images = renderer.render_gs(gs_data, views_number=16)
        logger.success(f"Successfully rendered {len(rendered_images)} views")
    except Exception as e:
        logger.error(f"Rendering failed: {e}")
        return

    # 4. Validate using ValidationEngine
    logger.info("Step 4: Running validation...")
    try:
        validator = ValidationEngine()
        validator.load_pipelines()
        
        validation_results = validator.validate_text_to_gs(PROMPT, rendered_images)
        
        validator.unload_pipelines()
        
        scores = {
            'final_score': validation_results.final_score,
            'combined_quality_score': validation_results.combined_quality_score,
            'alignment_score': validation_results.alignment_score,
            'ssim_score': validation_results.ssim_score,
            'lpips_score': validation_results.lpips_score
        }
        logger.success("Validation completed successfully!")
        
        print("\n" + "="*50)
        print("VALIDATION RESULTS")
        print("="*50)
        for metric, value in scores.items():
            print(f"{metric:25}: {value:.4f}")
        print("="*50)
        
    except Exception as e:
        logger.error(f"Validation failed: {e}")
        return

if __name__ == "__main__":
    main() 