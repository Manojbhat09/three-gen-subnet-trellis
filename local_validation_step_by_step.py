import requests
import base64
import torch
import sys
import os
import zstandard
from loguru import logger
import io
from typing import Any
import pyspz  # Import pyspz for SPZ decompression

# --- Setup Paths ---
validation_path = os.path.abspath('validation')
if validation_path not in sys.path:
    sys.path.insert(0, validation_path)

# Add validation directory to Python path
validation_path = os.path.join(os.path.dirname(__file__), 'validation')
sys.path.insert(0, validation_path)

from validation.engine.validation_engine import ValidationEngine
from validation.engine.data_structures import RequestData, ResponseData
from validation.engine.io.ply import PlyLoader
from validation.engine.rendering.renderer import Renderer
import zstandard


# # --- Import validation components ---
# from engine.data_structures import RequestData, ResponseData
# from engine.io.ply import PlyLoader
# from engine.rendering.renderer import Renderer
# from engine.metrics.alignment_scorer import AlignmentScorer
# from engine.models.quality_model import QualityModel
# from engine.models.aethtetic_model import AestheticModel



# --- Configuration ---
GENERATION_URL = "http://127.0.0.1:8096/generate/"
PROMPT = "a_motorcycle"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def generate_model(prompt):
    logger.info(f"Requesting model for prompt: '{prompt}'...")
    try:
        response = requests.post(
            GENERATION_URL,
            data={"prompt": prompt, "return_compressed": True}  # Explicitly request compressed
        )
        if response.status_code == 200:
            logger.success("Successfully received model data from the server.")
            return response.content  # Raw SPZ-compressed bytes
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
    # 1. Generate Model
    compressed_model = generate_model(PROMPT)
    if not compressed_model:
        return

    # 2. Decode and Load PLY
    gs_data = decode_and_load(compressed_model)
    if gs_data is None:
        return

    # 3. Render Views
    logger.info("Initializing renderer and rendering views... (This is memory intensive)")
    try:
        renderer = Renderer()
        rendered_images = renderer.render_gs(gs_data, views_number=16)
        logger.success(f"Successfully rendered {len(rendered_images)} views.")
    except Exception as e:
        logger.error(f"Rendering failed: {e}")
        return

    # 4. Validate using ValidationEngine
    try:
        logger.info("Initializing ValidationEngine...")
        validator = ValidationEngine()
        validator.load_pipelines()
        
        logger.info("Computing validation scores...")
        validation_results = validator.validate_text_to_gs(PROMPT, rendered_images)
        
        validator.unload_pipelines()
        
        scores = {
            'final_score': validation_results.final_score,
            'combined_quality_score': validation_results.combined_quality_score,
            'alignment_score': validation_results.alignment_score,
            'ssim_score': validation_results.ssim_score,
            'lpips_score': validation_results.lpips_score
        }
        logger.success("Validation completed.")
    except Exception as e:
        logger.error(f"Validation failed: {e}")
        return

    logger.info("\n--- Final Scores ---")
    for metric, value in scores.items():
        print(f"  - {metric}: {value}")

if __name__ == "__main__":
    main() 