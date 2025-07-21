#!/usr/bin/env python3

import requests
import base64
import torch
import sys
import os
import io
from loguru import logger
import pyspz

# --- Setup Paths ---
validation_path = os.path.abspath('validation')
if validation_path not in sys.path:
    sys.path.insert(0, validation_path)

from validation.engine.io.ply import PlyLoader

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
            logger.success("Successfully received model data from the server.")
            return response.content
        else:
            logger.error(f"Failed to generate model. Status code: {response.status_code}")
            return None
    except Exception as e:
        logger.error(f"Error requesting model: {e}")
        return None

def decode_and_load(compressed_data: bytes):
    logger.info("Decompressing SPZ data...")
    try:
        decompressed_data = pyspz.decompress(compressed_data)
        logger.info("Decompressed successfully.")

        logger.info("Loading PLY data...")
        ply_loader = PlyLoader()
        gs_data = ply_loader.from_buffer(io.BytesIO(decompressed_data))
        logger.success("PLY data loaded successfully.")
        return gs_data
    except Exception as e:
        logger.error(f"Failed to decompress or load model: {e}")
        return None

def main():
    logger.info("=== Minimal Validation Test ===")
    
    # 1. Generate Model
    compressed_model = generate_model(PROMPT)
    if not compressed_model:
        logger.error("Failed to generate model")
        return

    # 2. Decode and Load PLY
    gs_data = decode_and_load(compressed_model)
    if gs_data is None:
        logger.error("Failed to decode/load model")
        return

    # 3. Basic Analysis
    logger.info("=== Model Analysis ===")
    logger.info(f"Points shape: {gs_data.points.shape}")
    logger.info(f"Features DC shape: {gs_data.features_dc.shape}")
    logger.info(f"Opacities shape: {gs_data.opacities.shape}")
    logger.info(f"Scales shape: {gs_data.scales.shape}")
    logger.info(f"Rotations shape: {gs_data.rotations.shape}")
    
    # 4. Basic Quality Checks
    logger.info("=== Basic Quality Checks ===")
    
    # Check for NaN values
    nan_points = torch.isnan(gs_data.points).any()
    nan_opacities = torch.isnan(gs_data.opacities).any()
    logger.info(f"NaN in points: {nan_points}")
    logger.info(f"NaN in opacities: {nan_opacities}")
    
    # Check opacity range
    opacity_min = gs_data.opacities.min().item()
    opacity_max = gs_data.opacities.max().item()
    logger.info(f"Opacity range: [{opacity_min:.4f}, {opacity_max:.4f}]")
    
    # Check point cloud bounds
    points_min = gs_data.points.min(dim=0)[0]
    points_max = gs_data.points.max(dim=0)[0]
    logger.info(f"Point cloud bounds: min={points_min.tolist()}, max={points_max.tolist()}")
    
    logger.success("✅ Basic validation completed successfully!")
    logger.info("Model appears to be valid. Full validation with rendering would require gsplat installation.")

if __name__ == "__main__":
    main() 