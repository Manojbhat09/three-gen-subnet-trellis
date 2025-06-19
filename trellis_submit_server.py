#!/usr/bin/env python3
"""
Subnet 17 (404-GEN) - FLUX + TRELLIS Generation Server
Purpose: HTTP server for high-quality 3D model generation using Flux and TRELLIS
Produces validation-compatible Gaussian Splatting PLY files with SPZ compression

Text Prompt → FLUX Image → TRELLIS 3D → Gaussian Splatting PLY + SPZ Compression

# Generate 3D model
curl -X POST "http://localhost:8096/generate/" \
  -F "prompt=a blue ceramic vase with red trim" \
  -F "seed=42"

# Get asset information
curl "http://localhost:8096/assets/"

# Download compressed PLY file
curl "http://localhost:8096/assets/gaussian_splatting_ply" -o model.ply.spz
"""

import os
import time
import torch
import traceback
import threading
import gc
import numpy as np
from pathlib import Path
from typing import Optional, Dict, Any
from dataclasses import dataclass
import trimesh
from PIL import Image
import random
import yaml
import logging
import tempfile
import argparse
import asyncio
import base64
import io
import json
import math
import requests
import signal
import subprocess
import queue
import multiprocessing
import imageio
from io import BytesIO

from fastapi import FastAPI, Form, HTTPException
from fastapi.responses import Response, JSONResponse
import uvicorn

# Initialize logger
logger = logging.getLogger(__name__)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('trellis_submit_server.log')
    ]
)

seed = 42

torch.manual_seed(seed)
torch.cuda.manual_seed(seed)  # If using GPU
np.random.seed(seed)
random.seed(seed)

torch.backends.cudnn.deterministic = True    # For reproducibility with cuDNN
torch.backends.cudnn.benchmark = False       # Disable for reproducibility

# Set environment variables
os.environ['SPCONV_ALGO'] = 'native'
os.environ['ATTN_BACKEND'] = 'xformers'
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

# Add TRELLIS to Python path
import sys
TRELLIS_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "TRELLIS-TextoImagen3D")
sys.path.append(TRELLIS_PATH)

# Add Hunyuan3D path for background removal
HUNYUAN3D_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Hunyuan3D-2")
sys.path.append(HUNYUAN3D_PATH)

# Import TRELLIS components
from diffusers import FluxTransformer2DModel, FluxPipeline, BitsAndBytesConfig, GGUFQuantizationConfig
from transformers import T5EncoderModel, BitsAndBytesConfig as BitsAndBytesConfigTF
from trellis.pipelines import TrellisImageTo3DPipeline
from trellis.utils import render_utils, postprocessing_utils

# Import background removal
from hy3dgen.rembg import BackgroundRemover

# Import OpenCV for object centering
import cv2

# Constants from TRELLIS
NUM_INFERENCE_STEPS = 9
MAX_SEED = np.iinfo(np.int32).max

# Configuration
GENERATION_CONFIG = {
    'output_dir': './trellis_submit_outputs',
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    'num_inference_steps_t2i': 8,
    'flux_model_url': "https://huggingface.co/gokaygokay/flux-game/blob/main/hyperflux_00001_.q8_0.gguf",
    'flux_base_model': "camenduru/FLUX.1-dev-diffusers",
    'trellis_model_path': 'cavargas10/TRELLIS',
    'save_intermediate_outputs': True,
    'save_preview': False,
    'auto_compress_ply': True,
    # TRELLIS specific settings
    # 'guidance_scale': 3.0,#3.5,
    # 'ss_guidance_strength': 9.0,#7.5, #, 8.5,
    # 'ss_sampling_steps': 12, #13,
    # 'slat_guidance_strength': 4.0, #3.0, #4.0,
    # 'slat_sampling_steps': 12, #13,
    'guidance_scale': 3.5,
    'ss_guidance_strength': 8.5, #, 8.5,
    'ss_sampling_steps': 13,
    'slat_guidance_strength': 9.0, #4.0,
    'slat_sampling_steps': 14,
    # Memory management
    'enable_memory_efficient_attention': True,
    'enable_cpu_offload': True,
    'max_memory_usage_gb': 20,
    'validation_server_url': 'http://127.0.0.1:10006',
    'auto_validate_generations': True,
    'validation_timeout': 120,
    # Object centering settings
    'enable_object_centering': True,
    'centering_white_threshold': 240,
    'centering_padding': 30,
}

@dataclass
class GenerationMetrics:
    total_generations: int = 0
    successful_generations: int = 0
    failed_generations: int = 0
    average_generation_time: float = 0.0
    last_generation_time: float = 0.0
    validation_submissions: int = 0
    successful_validations: int = 0
    average_validation_score: float = 0.0
    best_validation_score: float = 0.0
    last_validation_score: float = 0.0

# Asset management
from enum import Enum

class AssetType(Enum):
    """Asset types for the generation pipeline"""
    FLUX_IMAGE = "flux_image"
    GAUSSIAN_SPLATTING_PLY = "gaussian_splatting_ply"
    PREVIEW_VIDEO = "preview_video"
    COMPRESSED_PLY = "compressed_ply"

@dataclass
class GenerationAsset:
    """Container for generation assets"""
    generation_id: str
    prompt: str
    seed: int
    asset_directory: Path
    assets: Dict[AssetType, Any]
    metadata: Dict[str, Any]
    timestamp: float
    
    def __post_init__(self):
        self.asset_directory.mkdir(parents=True, exist_ok=True)
    
    def add_asset(self, asset_type: AssetType, data: Any):
        """Add an asset to the generation"""
        self.assets[asset_type] = data
        
        # Save to file if appropriate
        if asset_type == AssetType.FLUX_IMAGE:
            file_path = self.asset_directory / "flux_image.png"
            data.save(file_path)
        elif asset_type == AssetType.GAUSSIAN_SPLATTING_PLY:
            file_path = self.asset_directory / "gaussian_splatting.ply"
            with open(file_path, 'wb') as f:
                f.write(data)
        elif asset_type == AssetType.PREVIEW_VIDEO:
            file_path = self.asset_directory / "preview.mp4"
            imageio.mimsave(file_path, data, fps=15)
        elif asset_type == AssetType.COMPRESSED_PLY:
            file_path = self.asset_directory / "compressed.ply.spz"
            with open(file_path, 'wb') as f:
                f.write(data)
    
    def save_metadata(self) -> Path:
        """Save metadata to JSON file"""
        metadata_path = self.asset_directory / "metadata.json"
        
        # Prepare metadata with all relevant information
        full_metadata = {
            "generation_id": self.generation_id,
            "prompt": self.prompt,
            "seed": self.seed,
            "timestamp": self.timestamp,
            "assets": [asset_type.value for asset_type in self.assets.keys()],
            **self.metadata  # Include any additional metadata
        }
        
        # Write to JSON file
        with open(metadata_path, 'w') as f:
            json.dump(full_metadata, f, indent=2)
        
        return metadata_path

class AssetManager:
    """Manages generation assets"""
    
    def __init__(self, base_output_dir: str):
        self.base_output_dir = Path(base_output_dir)
        self.base_output_dir.mkdir(exist_ok=True)
        self.current_asset: Optional[GenerationAsset] = None
        
    def create_asset(self, prompt: str, seed: int) -> GenerationAsset:
        """Create a new generation asset"""
        generation_id = f"{int(time.time())}_{seed}"
        asset_dir = self.base_output_dir / generation_id
        
        self.current_asset = GenerationAsset(
            generation_id=generation_id,
            prompt=prompt,
            seed=seed,
            asset_directory=asset_dir,
            assets={},
            metadata={
                "prompt": prompt,
                "seed": seed,
                "timestamp": time.time(),
                "pipeline": "flux_trellis_v1.0"
            },
            timestamp=time.time()
        )
        
        return self.current_asset
    
    def get_asset(self, asset_type: AssetType) -> Optional[Any]:
        """Get an asset from current generation"""
        if self.current_asset and asset_type in self.current_asset.assets:
            return self.current_asset.assets[asset_type]
        return None

# Global job status tracking
generation_job_status = {
    "current_job_id": None,
    "status": "idle",
    "prompt": None,
    "seed": None,
    "start_time": None,
    "end_time": None,
    "ply_path": None,
    "error": None
}

class TrellisGenerator:
    def __init__(self):
        # Initialize model instance variables
        self.flux_pipeline = None
        self.flux_transformer = None
        self.flux_text_encoder_2 = None
        self.trellis_pipeline = None
        self.background_remover = None
        
        self.metrics = GenerationMetrics()
        self.generation_lock = threading.Lock()
        self.recent_generations = {}  # Store recent generations for retrieval
        
        # Initialize asset manager
        self.asset_manager = AssetManager(GENERATION_CONFIG['output_dir'])
        
        # Get HuggingFace token
        try:
            from huggingface_hub import HfFolder
            token = HfFolder.get_token()
            if token:
                os.environ["HUGGINGFACE_TOKEN"] = token
                print("✓ HuggingFace token loaded from cache")
            else:
                print("⚠️ No HuggingFace token found in cache")
        except Exception as e:
            print(f"⚠️ Error getting token from cache: {e}")
        
        Path(GENERATION_CONFIG['output_dir']).mkdir(exist_ok=True)
        print("🔧 TRELLIS Generator initialized")
        self.ready = True

    def _clear_gpu_memory(self):
        """Clear GPU memory cache aggressively"""
        gc.collect()
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
            
            try:
                torch.cuda.ipc_collect()
            except:
                pass
            
            if hasattr(torch.cuda, 'reset_accumulated_memory_stats'):
                torch.cuda.reset_accumulated_memory_stats()
        
        gc.collect()
        
        if torch.cuda.is_available():
            gpu_memory_free, gpu_memory_total = torch.cuda.mem_get_info()
            memory_used = gpu_memory_total - gpu_memory_free
            print(f"🧠 GPU Memory: {memory_used / 1e9:.1f}GB used, {gpu_memory_free / 1e9:.1f}GB free")
            return gpu_memory_free / 1e9
        
        return 0

    def _load_flux_models(self):
        """Load FLUX models"""
        if self.flux_pipeline is not None:
            print("✓ FLUX models already loaded")
            return
            
        print("🔧 Loading FLUX models...")
        
        try:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            huggingface_token = os.getenv("HUGGINGFACE_TOKEN")
            dtype = torch.bfloat16
            
            file_url = GENERATION_CONFIG['flux_model_url']
            file_url = file_url.replace("/resolve/main/", "/blob/main/").replace("?download=true", "")
            single_file_base_model = GENERATION_CONFIG['flux_base_model']
            
            # Load text encoder with 8-bit quantization
            print("Loading FLUX text encoder with 8-bit quantization...")
            quantization_config_tf = BitsAndBytesConfigTF(
                load_in_8bit=True,
                bnb_8bit_compute_dtype=torch.bfloat16
            )
            self.flux_text_encoder_2 = T5EncoderModel.from_pretrained(
                # single_file_base_model,
                "manbeast3b/flux.1-schnell-full1",
                revision = "cb1b599b0d712b9aab2c4df3ad27b050a27ec146",  
                subfolder="text_encoder_2", 
                torch_dtype=dtype, 
                quantization_config=quantization_config_tf, 
                token=huggingface_token
            )
            
            # Load transformer with GGUF configuration
            print("Loading FLUX transformer with GGUF quantization...")
            self.flux_transformer = FluxTransformer2DModel.from_single_file(
                file_url, 
                subfolder="transformer", 
                quantization_config=GGUFQuantizationConfig(compute_dtype=dtype), 
                torch_dtype=dtype, 
                config=single_file_base_model
            )
            
            # Initialize pipeline
            print("Initializing FLUX pipeline...")
            self.flux_pipeline = FluxPipeline.from_pretrained(
                single_file_base_model, 
                transformer=self.flux_transformer, 
                text_encoder_2=self.flux_text_encoder_2, 
                torch_dtype=dtype, 
                token=huggingface_token
            )
            self.flux_pipeline.to("cuda")

            from flux_caching import apply_cache_on_pipe
            apply_cache_on_pipe(self.flux_pipeline)
            self.flux_pipeline.to(memory_format=torch.channels_last)
            self.flux_pipeline.vae = torch.compile(self.flux_pipeline.vae, mode="max-autotune")

            # from torchao.quantization import quantize_, float8_dynamic_activation_float8_weight
            # quantize_(self.flux_pipeline.vae, float8_dynamic_activation_float8_weight())
            
            print("✅ FLUX models loaded successfully")
            
        except Exception as e:
            print(f"❌ FLUX model loading failed: {e}")
            traceback.print_exc()
            self._unload_flux_models()
    
    def _unload_flux_models(self):
        """Unload FLUX models to free GPU memory"""
        print("🧹 Unloading FLUX models...")
        
        models_unloaded = []
        
        if self.flux_pipeline is not None:
            del self.flux_pipeline
            self.flux_pipeline = None
            models_unloaded.append("flux_pipeline")
        
        if self.flux_transformer is not None:
            del self.flux_transformer
            self.flux_transformer = None
            models_unloaded.append("flux_transformer")
        
        if self.flux_text_encoder_2 is not None:
            del self.flux_text_encoder_2
            self.flux_text_encoder_2 = None
            models_unloaded.append("flux_text_encoder_2")
        
        if models_unloaded:
            self._clear_gpu_memory()
            print(f"✅ FLUX models unloaded: {', '.join(models_unloaded)}")

    def _load_trellis_pipeline(self):
        """Load TRELLIS pipeline"""
        if self.trellis_pipeline is not None:
            print("✓ TRELLIS pipeline already loaded")
            return
            
        print("🔧 Loading TRELLIS pipeline...")
        
        try:
            self.trellis_pipeline = TrellisImageTo3DPipeline.from_pretrained(
                GENERATION_CONFIG['trellis_model_path']
            )
            self.trellis_pipeline.cuda()
            
            # Warm up the pipeline
            try:
                self.trellis_pipeline.preprocess_image(
                    Image.fromarray(np.zeros((512, 512, 3), dtype=np.uint8))
                )
            except:
                pass
            
            print("✅ TRELLIS pipeline loaded successfully")
            
        except Exception as e:
            print(f"❌ TRELLIS pipeline loading failed: {e}")
            traceback.print_exc()
            self.trellis_pipeline = None

    def _unload_trellis_pipeline(self):
        """Unload TRELLIS pipeline to free GPU memory"""
        if self.trellis_pipeline is not None:
            print("🧹 Unloading TRELLIS pipeline...")
            del self.trellis_pipeline
            self.trellis_pipeline = None
            self._clear_gpu_memory()
            print("✅ TRELLIS pipeline unloaded")

    def _load_background_remover(self):
        """Load background removal model"""
        if self.background_remover is not None:
            print("✓ Background remover already loaded")
            return
            
        print("🔧 Loading background remover...")
        
        try:
            self.background_remover = BackgroundRemover()
            print("✅ Background remover loaded successfully")
            
        except Exception as e:
            print(f"❌ Background remover loading failed: {e}")
            traceback.print_exc()
            self.background_remover = None

    def _unload_background_remover(self):
        """Unload background remover to free GPU memory"""
        if self.background_remover is not None:
            print("🧹 Unloading background remover...")
            del self.background_remover
            self.background_remover = None
            self._clear_gpu_memory()
            print("✅ Background remover unloaded")

    def center_object_in_image(self, image: Image.Image, white_threshold: int = 240, padding: int = 20) -> Image.Image:
        """
        Center the main object in the image by detecting content and repositioning it
        
        Args:
            image: PIL Image with white background
            white_threshold: Pixel values above this are considered white/background
            padding: Extra padding around the detected object (in pixels)
            
        Returns:
            PIL Image with centered object
        """
        try:
            # Convert PIL to numpy array
            image_array = np.array(image)
            original_height, original_width = image_array.shape[:2]
            
            # Convert to grayscale for content detection
            if len(image_array.shape) == 3:
                if image_array.shape[2] == 4:  # RGBA
                    alpha = image_array[:, :, 3]
                    gray = cv2.cvtColor(image_array[:, :, :3], cv2.COLOR_RGB2GRAY)
                    gray = np.where(alpha > 0, gray, 255)
                else:  # RGB
                    gray = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)
            else:  # Already grayscale
                gray = image_array
            
            # Create mask of non-white pixels
            content_mask = gray < white_threshold
            
            # Find contours of content
            contours, _ = cv2.findContours(
                content_mask.astype(np.uint8), 
                cv2.RETR_EXTERNAL, 
                cv2.CHAIN_APPROX_SIMPLE
            )
            
            if not contours:
                print("⚠️ No content detected for centering, returning original image")
                return image
            
            # Find the bounding box that encompasses all content
            all_points = np.vstack(contours)
            x_min = np.min(all_points[:, :, 0])
            y_min = np.min(all_points[:, :, 1])
            x_max = np.max(all_points[:, :, 0])
            y_max = np.max(all_points[:, :, 1])
            
            # Add padding
            x_min = max(0, x_min - padding)
            y_min = max(0, y_min - padding)
            x_max = min(original_width, x_max + padding)
            y_max = min(original_height, y_max + padding)
            
            content_width = x_max - x_min
            content_height = y_max - y_min
            
            print(f"📦 Content bounding box: ({x_min}, {y_min}) to ({x_max}, {y_max})")
            print(f"📏 Content size: {content_width} x {content_height}")
            
            # Extract the content region
            content = image_array[y_min:y_max, x_min:x_max]
            
            # Create new image with white background
            if len(image_array.shape) == 3:
                if image_array.shape[2] == 4:  # RGBA
                    centered_image = np.full((original_height, original_width, 4), [255, 255, 255, 255], dtype=np.uint8)
                else:  # RGB
                    centered_image = np.full((original_height, original_width, 3), [255, 255, 255], dtype=np.uint8)
            else:  # Grayscale
                centered_image = np.full((original_height, original_width), 255, dtype=np.uint8)
            
            # Calculate position to center the content
            center_x = original_width // 2
            center_y = original_height // 2
            
            # Calculate top-left corner for centering
            paste_x = center_x - content_width // 2
            paste_y = center_y - content_height // 2
            
            # Ensure the content fits within the image
            paste_x = max(0, min(paste_x, original_width - content_width))
            paste_y = max(0, min(paste_y, original_height - content_height))
            
            # Paste the content into the centered position
            end_x = paste_x + content_width
            end_y = paste_y + content_height
            
            # Handle potential size mismatches
            if end_x > original_width:
                content = content[:, :original_width - paste_x]
                end_x = original_width
            if end_y > original_height:
                content = content[:original_height - paste_y]
                end_y = original_height
            
            centered_image[paste_y:end_y, paste_x:end_x] = content
            
            print(f"✅ Content centered at position ({paste_x}, {paste_y})")
            
            # Convert back to PIL Image
            centered_pil = Image.fromarray(centered_image)
            return centered_pil
            
        except Exception as e:
            print(f"⚠️ Object centering failed: {e}")
            print("   Continuing with original image...")
            return image

    def generate_3d_model(self, prompt: str, seed: int = None, dynamic_config: Dict[str, Any] = None) -> Optional[bytes]:
        """Generate 3D model from text prompt using FLUX + TRELLIS pipeline"""
        
        if dynamic_config is None:
            dynamic_config = {}
        
        # Handle seed
        if seed is None:
            seed = random.randint(0, MAX_SEED)
        
        # Update metrics
        self.metrics.total_generations += 1
        self.metrics.last_generation_time = time.time()
        
        generation_start = time.time()
        logger.info(f"🎨 Starting generation for prompt: '{prompt}' (seed: {seed})")
        
        # Create generation asset
        self.current_asset = self.asset_manager.create_asset(prompt, seed)
        
        try:
            # Step 1: Generate image with FLUX
            logger.info("🖼️ Step 1: Generating image with FLUX...")
            
            # Load FLUX models if not already loaded
            if self.flux_pipeline is None:
                self._load_flux_models()
            
            # Enhanced prompt for FLUX
            enhanced_prompt = f"{prompt}, game asset, 3d model, isometric view, white background, high quality, detailed"
            
            generator = torch.Generator("cuda").manual_seed(seed)
            
            with torch.inference_mode():
                image = self.flux_pipeline(
                    prompt=enhanced_prompt,
                    guidance_scale=dynamic_config.get('guidance_scale', GENERATION_CONFIG['guidance_scale']),
                    num_inference_steps=NUM_INFERENCE_STEPS,
                    width=1024,
                    height=1024,
                    generator=generator,
                ).images[0]
            
            # Apply centering if enabled
            if GENERATION_CONFIG['enable_object_centering']:
                logger.info("📐 Centering object in image...")
                image = self.center_object_in_image(
                    image,
                    white_threshold=GENERATION_CONFIG['centering_white_threshold'],
                    padding=GENERATION_CONFIG['centering_padding']
                )
            
            # Save Flux image if configured
            if GENERATION_CONFIG['save_intermediate_outputs']:
                self.current_asset.add_asset(AssetType.FLUX_IMAGE, image)
                logger.info(f"✅ Flux image saved to asset directory")
            
            # Step 2: Remove background
            logger.info("🎭 Step 2: Removing background...")
            
            # if self.background_remover is None:
            #     self._load_background_remover()
            
            # image_no_bg = self.background_remover(image)
            image_no_bg = image
            # Free Flux memory
            self._unload_flux_models()
            self._clear_gpu_memory()
            
            # Step 3: Generate 3D with TRELLIS
            logger.info("🏗️ Step 3: Generating 3D model with TRELLIS...")
            
            if self.trellis_pipeline is None:
                self._load_trellis_pipeline()
            
            # Use provided parameters or defaults
            outputs = self.trellis_pipeline.run(
                image_no_bg,
                seed=seed,
                preprocess_image=False,
                sparse_structure_sampler_params={
                    "steps": dynamic_config.get('ss_sampling_steps', GENERATION_CONFIG['ss_sampling_steps']),
                    "cfg_strength": dynamic_config.get('ss_guidance_strength', GENERATION_CONFIG['ss_guidance_strength']),
                },
                slat_sampler_params={
                    "steps": dynamic_config.get('slat_sampling_steps', GENERATION_CONFIG['slat_sampling_steps']),
                    "cfg_strength": dynamic_config.get('slat_guidance_strength', GENERATION_CONFIG['slat_guidance_strength']),
                },
            )
            
            # Get Gaussian splatting representation
            gs = outputs['gaussian'][0]
            
            # Save PLY file to memory
            ply_buffer = BytesIO()
            gs.save_ply(ply_buffer)
            ply_data = ply_buffer.getvalue()
            
            # Store uncompressed PLY
            if GENERATION_CONFIG['save_intermediate_outputs']:
                self.current_asset.add_asset(AssetType.GAUSSIAN_SPLATTING_PLY, ply_data)
            
            # Compress using pyspz
            compressed_data = None
            if GENERATION_CONFIG['auto_compress_ply']:
                try:
                    import pyspz
                    compressed_data = pyspz.compress(ply_data, workers=-1)
                    compression_ratio = len(compressed_data) / len(ply_data)
                    logger.info(f"🗜️ Compressed PLY: {len(ply_data):,} → {len(compressed_data):,} bytes ({compression_ratio:.1%})")
                    
                    # Store compressed version
                    if GENERATION_CONFIG['save_intermediate_outputs']:
                        self.current_asset.add_asset(AssetType.COMPRESSED_PLY, compressed_data)
                except Exception as e:
                    logger.error(f"❌ SPZ compression failed: {e}")
                    compressed_data = ply_data  # Fall back to uncompressed
            else:
                compressed_data = ply_data
            
            # Generate preview if enabled
            if GENERATION_CONFIG['save_preview']:
                logger.info("🎬 Generating preview video...")
                try:
                    video_frames = render_utils.render_video(
                        outputs['gaussian'][0], 
                        output_size=256
                    )['frames']
                    self.current_asset.add_asset(AssetType.PREVIEW_VIDEO, video_frames)
                except Exception as e:
                    logger.error(f"❌ Preview generation failed: {e}")
            
            # Free TRELLIS memory
            self._unload_trellis_pipeline()
            self._clear_gpu_memory()
            
            generation_time = time.time() - generation_start
            
            # Update metrics
            self.metrics.successful_generations += 1
            self.metrics.average_generation_time = (
                (self.metrics.average_generation_time * (self.metrics.successful_generations - 1) + generation_time) /
                self.metrics.successful_generations
            )
            
            # Store generation metadata
            self.current_asset.metadata = {
                'generation_time': generation_time,
                'seed': seed,
                'original_ply_size': len(ply_data),
                'compressed_size': len(compressed_data) if compressed_data else 0,
                'compression_ratio': len(compressed_data) / len(ply_data) if compressed_data else 1.0,
                'dynamic_params_used': bool(dynamic_config)
            }
            
            logger.info(f"✅ Generation completed in {generation_time:.2f}s")
            
            # Save metadata to JSON file
            metadata_path = self.current_asset.save_metadata()
            logger.info(f"💾 Metadata saved: {metadata_path}")
            
            # Store in recent generations
            self.recent_generations[seed] = {
                'prompt': prompt,
                'timestamp': time.time(),
                'ply_data': compressed_data,
                'asset': self.current_asset
            }
            
            # Return compressed data
            return compressed_data
            
        except Exception as e:
            logger.error(f"❌ Generation failed: {e}")
            traceback.print_exc()
            
            # Update failure metrics
            self.metrics.failed_generations += 1
            
            # Clear models on error
            self._clear_gpu_memory()
            
            return None

    def get_status(self) -> Dict[str, Any]:
        """Get server status and metrics"""
        return {
            "status": "running",
            "models_loaded": {
                "flux_pipeline": self.flux_pipeline is not None,
                "flux_transformer": self.flux_transformer is not None,  
                "flux_text_encoder": self.flux_text_encoder_2 is not None,
                "trellis_pipeline": self.trellis_pipeline is not None,
                "background_remover": self.background_remover is not None,
            },
            "metrics": {
                "total_generations": self.metrics.total_generations,
                "successful_generations": self.metrics.successful_generations,
                "failed_generations": self.metrics.failed_generations,
                "success_rate": (
                    self.metrics.successful_generations / max(1, self.metrics.total_generations) * 100
                ),
                "average_generation_time": self.metrics.average_generation_time,
                "last_generation_time": self.metrics.last_generation_time,
                "validation_submissions": self.metrics.validation_submissions,
                "successful_validations": self.metrics.successful_validations,
                "validation_success_rate": (
                    self.metrics.successful_validations / max(1, self.metrics.validation_submissions) * 100
                ),
                "average_validation_score": self.metrics.average_validation_score,
                "best_validation_score": self.metrics.best_validation_score,
                "last_validation_score": self.metrics.last_validation_score,
            },
            "config": GENERATION_CONFIG,
            "gpu_memory": self._clear_gpu_memory() if torch.cuda.is_available() else 0,
            "ready":self.ready
        }

    def submit_for_validation(self, prompt: str, ply_data: bytes) -> Dict[str, Any]:
        """Submit PLY for validation"""
        try:
            validation_url = GENERATION_CONFIG['validation_server_url']
            
            print("📊 Submitting generation for validation...")
            
            # Prepare validation request
            encoded_data = base64.b64encode(ply_data).decode('utf-8')
            
            request_data = {
                "prompt": prompt,
                "data": encoded_data,
                "compression": 0,
                "generate_preview": False,
                "preview_score_threshold": 0.8
            }
            
            response = requests.post(
                f"{validation_url}/validate_txt_to_3d_ply/",
                json=request_data,
                timeout=GENERATION_CONFIG['validation_timeout']
            )
            
            if response.status_code == 200:
                result = response.json()
                validation_score = result.get("score", 0.0)
                
                print(f"✅ Validation completed! Score: {validation_score:.4f}")
                
                # Update metrics
                self.metrics.validation_submissions += 1
                self.metrics.successful_validations += 1
                self.metrics.last_validation_score = validation_score
                self.metrics.average_validation_score = (
                    (self.metrics.average_validation_score * (self.metrics.successful_validations - 1) + validation_score)
                    / self.metrics.successful_validations
                )
                if validation_score > self.metrics.best_validation_score:
                    self.metrics.best_validation_score = validation_score
                
                return {
                    "status": "success",
                    "validation_score": validation_score,
                    "quality_metrics": {
                        "iqa": result.get("iqa", 0.0),
                        "alignment": result.get("alignment_score", 0.0),
                        "ssim": result.get("ssim", 0.0),
                        "lpips": result.get("lpips", 0.0)
                    }
                }
            else:
                print(f"⚠️ Validation request failed: {response.status_code}")
                self.metrics.validation_submissions += 1
                return {
                    "status": "error", 
                    "error": f"HTTP {response.status_code}",
                    "validation_score": 0.0
                }
                
        except Exception as e:
            print(f"⚠️ Validation submission failed: {e}")
            self.metrics.validation_submissions += 1
            return {
                "status": "exception",
                "error": str(e),
                "validation_score": 0.0
            }

# Initialize FastAPI app
app = FastAPI(title="FLUX + TRELLIS Generation Server", version="1.0.0")

# Initialize global generator
generator = TrellisGenerator()

@app.get("/job/status/")
async def get_job_status():
    """Get current job processing status"""
    return {
        "job_id": generation_job_status["current_job_id"],
        "status": generation_job_status["status"],
        "prompt": generation_job_status["prompt"],
        "seed": generation_job_status["seed"],
        "start_time": generation_job_status["start_time"],
        "end_time": generation_job_status["end_time"],
        "processing_time": (generation_job_status["end_time"] - generation_job_status["start_time"]) if generation_job_status["end_time"] and generation_job_status["start_time"] else None,
        "error": generation_job_status["error"]
    }

@app.post("/job/reset/")
async def reset_job_status():
    """Reset job status to idle"""
    global generation_job_status
    generation_job_status = {
        "current_job_id": None,
        "status": "idle",
        "prompt": None,
        "seed": None,
        "start_time": None,
        "end_time": None,
        "ply_path": None,
        "error": None
    }
    return {"status": "reset"}

@app.post("/generate/")
async def generate_3d_model_endpoint(
    prompt: str = Form(...), 
    seed: Optional[int] = Form(None),
    return_compressed: Optional[bool] = Form(True),
    # Dynamic TRELLIS parameters
    guidance_scale: Optional[float] = Form(None),
    ss_guidance_strength: Optional[float] = Form(None),
    ss_sampling_steps: Optional[int] = Form(None),
    slat_guidance_strength: Optional[float] = Form(None),
    slat_sampling_steps: Optional[int] = Form(None)
):
    """
    Generate a 3D model from a text prompt.
    
    Args:
        prompt: Text description of the 3D model
        seed: Random seed (optional, will be generated if not provided)
        return_compressed: Whether to return SPZ compressed data
        guidance_scale: TRELLIS guidance scale (optional)
        ss_guidance_strength: Sparse structure guidance strength (optional)
        ss_sampling_steps: Sparse structure sampling steps (optional)
        slat_guidance_strength: Structured latent guidance strength (optional)
        slat_sampling_steps: Structured latent sampling steps (optional)
    """
    try:
        # Log dynamic parameters if provided
        if any([guidance_scale, ss_guidance_strength, ss_sampling_steps, 
                slat_guidance_strength, slat_sampling_steps]):
            logger.info(f"🎯 Using dynamic parameters for generation:")
            if guidance_scale: logger.info(f"   guidance_scale: {guidance_scale}")
            if ss_guidance_strength: logger.info(f"   ss_guidance_strength: {ss_guidance_strength}")
            if ss_sampling_steps: logger.info(f"   ss_sampling_steps: {ss_sampling_steps}")
            if slat_guidance_strength: logger.info(f"   slat_guidance_strength: {slat_guidance_strength}")
            if slat_sampling_steps: logger.info(f"   slat_sampling_steps: {slat_sampling_steps}")
        
        # Override default parameters with dynamic ones if provided
        dynamic_config = {}
        if guidance_scale is not None:
            dynamic_config['guidance_scale'] = guidance_scale
        if ss_guidance_strength is not None:
            dynamic_config['ss_guidance_strength'] = ss_guidance_strength
        if ss_sampling_steps is not None:
            dynamic_config['ss_sampling_steps'] = ss_sampling_steps
        if slat_guidance_strength is not None:
            dynamic_config['slat_guidance_strength'] = slat_guidance_strength
        if slat_sampling_steps is not None:
            dynamic_config['slat_sampling_steps'] = slat_sampling_steps
        
        # Generate model with dynamic configuration
        ply_data = generator.generate_3d_model(prompt, seed, dynamic_config=dynamic_config)
        
        if ply_data is None:
            raise HTTPException(status_code=500, detail="Generation failed")
        
        # Return the appropriate data based on compression preference
        if return_compressed:
            # Data is already compressed
            headers = {
                "Content-Type": "application/octet-stream",
                "Content-Disposition": f"attachment; filename=model_{seed}.ply.spz",
                "X-Compression-Ratio": f"{len(ply_data) / 1000000:.2f}MB"  # Approximate ratio
            }
            return Response(content=ply_data, headers=headers)
        else:
            # Decompress if requested
            try:
                import pyspz
                decompressed_data = pyspz.decompress(ply_data)
                headers = {
                    "Content-Type": "application/octet-stream",
                    "Content-Disposition": f"attachment; filename=model_{seed}.ply"
                }
                return Response(content=decompressed_data, headers=headers)
            except:
                # If decompression fails, return compressed
                headers = {
                    "Content-Type": "application/octet-stream",
                    "Content-Disposition": f"attachment; filename=model_{seed}.ply.spz",
                    "X-Compression-Format": "spz"
                }
                return Response(content=ply_data, headers=headers)
        
    except Exception as e:
        logger.error(f"❌ Generation endpoint error: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/validate/")
async def validate_generation(
    prompt: str = Form(...),
    use_last_generation: Optional[bool] = Form(True)
):
    """Validate the last generation or submit for validation"""
    try:
        if use_last_generation:
            # Get the last generated PLY
            ply_data = generator.asset_manager.get_asset(AssetType.GAUSSIAN_SPLATTING_PLY)
            if ply_data is None:
                raise HTTPException(status_code=400, detail="No PLY data available for validation")
        else:
            raise HTTPException(status_code=400, detail="Manual PLY upload not implemented")
        
        # Submit for validation
        validation_results = generator.submit_for_validation(prompt, ply_data)
        
        return JSONResponse(content={
            "status": "success",
            "validation_results": validation_results
        })
        
    except Exception as e:
        return JSONResponse(content={
            "status": "error",
            "message": str(e)
        }, status_code=500)

@app.get("/status/")
async def get_server_status():
    """Get server status and metrics"""
    return generator.get_status()

@app.get("/health/")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "timestamp": time.time()}

@app.get("/assets/")
async def get_assets():
    """Get information about the last generation's assets"""
    try:
        if generator.asset_manager.current_asset:
            asset = generator.asset_manager.current_asset
            return JSONResponse(content={
                "status": "success",
                "generation_id": asset.generation_id,
                "prompt": asset.prompt,
                "seed": asset.seed,
                "timestamp": asset.timestamp,
                "assets": list(asset.assets.keys()),
                "asset_directory": str(asset.asset_directory)
            })
        else:
            return JSONResponse(content={
                "status": "no_generation",
                "message": "No generation has been completed yet"
            })
    except Exception as e:
        return JSONResponse(content={
            "status": "error",
            "message": str(e)
        }, status_code=500)

@app.get("/assets/{asset_type}")
async def get_asset_file(asset_type: str):
    """Get a specific asset file from the last generation"""
    try:
        # Convert string to AssetType enum
        try:
            asset_type_enum = AssetType(asset_type)
        except ValueError:
            return JSONResponse(content={
                "status": "error",
                "message": f"Invalid asset type: {asset_type}. Valid types: {[e.value for e in AssetType]}"
            }, status_code=400)
        
        # Get the asset
        asset_data = generator.asset_manager.get_asset(asset_type_enum)
        if asset_data is None:
            return JSONResponse(content={
                "status": "not_found",
                "message": f"Asset {asset_type} not found"
            }, status_code=404)
        
        # Return asset data
        content_type = "application/octet-stream"
        filename = f"trellis_{asset_type}"
        
        if asset_type_enum == AssetType.FLUX_IMAGE:
            content_type = "image/png"
            filename += ".png"
            # Convert PIL Image to bytes
            img_buffer = io.BytesIO()
            asset_data.save(img_buffer, format='PNG')
            asset_data = img_buffer.getvalue()
        elif asset_type_enum == AssetType.GAUSSIAN_SPLATTING_PLY:
            content_type = "application/ply"
            filename += ".ply"
        elif asset_type_enum == AssetType.COMPRESSED_PLY:
            content_type = "application/octet-stream"
            filename += ".ply.spz"
        
        return Response(
            content=asset_data,
            media_type=content_type,
            headers={
                "Content-Disposition": f"attachment; filename={filename}",
                "X-Asset-Type": asset_type,
                "X-Asset-Size": str(len(asset_data)),
                "X-Pipeline": "flux_trellis"
            }
        )
        
    except Exception as e:
        return JSONResponse(content={
            "status": "error",
            "message": str(e)
        }, status_code=500)

@app.post("/clear_cache/")
async def clear_cache():
    """Clear GPU memory cache"""
    try:
        available_memory = generator._clear_gpu_memory()
        return {
            "status": "cache_cleared", 
            "available_memory_gb": available_memory
        }
    except Exception as e:
        return {"error": f"Failed to clear cache: {str(e)}"}

@app.get("/config/")
async def get_config():
    """Get current configuration"""
    return GENERATION_CONFIG

@app.post("/config/centering/")
async def update_centering_config(
    enabled: bool = Form(True),
    white_threshold: int = Form(240),
    padding: int = Form(30)
):
    """Update object centering configuration"""
    try:
        GENERATION_CONFIG['enable_object_centering'] = enabled
        GENERATION_CONFIG['centering_white_threshold'] = white_threshold
        GENERATION_CONFIG['centering_padding'] = padding
        
        return {
            "status": "success",
            "message": "Object centering configuration updated",
            "config": {
                "enable_object_centering": enabled,
                "centering_white_threshold": white_threshold,
                "centering_padding": padding
            }
        }
    except Exception as e:
        return JSONResponse(content={
            "status": "error",
            "message": str(e)
        }, status_code=500)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FLUX + TRELLIS Generation Server")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8096, help="Port to bind to")
    parser.add_argument("--workers", type=int, default=1, help="Number of worker processes")
    
    args = parser.parse_args()
    
    print(f"Starting FLUX + TRELLIS Generation Server on {args.host}:{args.port}")
    print("=" * 80)
    print("Pipeline: Text → FLUX → Image → TRELLIS → Gaussian Splatting PLY")
    print("Features:")
    print("  • FLUX text-to-image generation with quantization")
    print("  • TRELLIS image-to-3D Gaussian Splatting generation")
    print("  • SPZ compression for efficient storage/transmission")
    print("  • Optional validation integration")
    print("  • Memory-optimized for RTX 4090 (24GB)")
    print("=" * 80)
    
    uvicorn.run(
        app, 
        host=args.host, 
        port=args.port,
        workers=args.workers,
        log_level="info"
    ) 