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

ython test_reproducibility_quality.py "stone-etched armor with leafy pattern" --port 8099 --endpoint "generate/" --min-similarity 0.3 --log-count 7 --ss_steps 12 --slat_steps 12 --slat_guidance  3.5 --ss_guidance 7.5

python trellis_subnit_server_mix_lora_flash_unload.py  --port 8096
python trellis_subnit_server_mix_lora_flash_grid.py  --port 8096
"""

import os
import time
import torch
import traceback
import threading
import gc
import numpy as np
from pathlib import Path
from typing import Optional, Dict, Any, Tuple
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
import socket
import json
from multiprocessing.connection import Client

from fastapi import FastAPI, Form, HTTPException, UploadFile, File
from fastapi.responses import Response, JSONResponse
import uvicorn
import torch
import torch._dynamo
torch._dynamo.config.suppress_errors = True
seed = 42
torch.manual_seed(seed)
# torch.use_deterministic_algorithms(True)

torch.manual_seed(seed)
torch.cuda.manual_seed(seed)  # If using GPU
np.random.seed(seed)
random.seed(seed)

torch.backends.cudnn.deterministic = True    # For reproducibility with cuDNN
torch.backends.cudnn.benchmark = False       # Disable for reproducibility
torch.backends.cuda.matmul.allow_tf32 = True
# Set environment variables
os.environ['SPCONV_ALGO'] = 'native'
os.environ['ATTN_BACKEND'] = 'xformers'
# export ATTN_BACKEND=xformers
# export SPARSE_ATTN_BACKEND=xformers
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

# Add TRELLIS to Python path
import sys
TRELLIS_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "TRELLIS-TextoImagen3D")
sys.path.append(TRELLIS_PATH)

# Add Hunyuan3D path for background removal
HUNYUAN3D_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Hunyuan3D-2")
sys.path.append(HUNYUAN3D_PATH)

# Import TRELLIS components
from diffusers import FluxTransformer2DModel, FluxPipeline, BitsAndBytesConfig, GGUFQuantizationConfig, StableDiffusionPipeline, AutoencoderTiny, DiffusionPipeline
from transformers import T5EncoderModel, BitsAndBytesConfig as BitsAndBytesConfigTF
from trellis.pipelines import TrellisImageTo3DPipeline
from trellis.utils import render_utils, postprocessing_utils
from huggingface_hub.constants import HF_HUB_CACHE

# Import background removal
from rembg import new_session, remove
from hy3dgen.rembg import BackgroundRemover

# Import HunyuanDiT
from hy3dgen.text2image import HunyuanDiTPipeline

# Import OpenCV for object centering
import cv2

# Import patcher for LoRA fixes
from patcher import patch_final_layer_adaLN

# Constants from TRELLIS
# NUM_INFERENCE_STEPS = 8
NUM_INFERENCE_STEPS = 7
MAX_SEED = np.iinfo(np.int32).max

# # Configuration
# GENERATION_CONFIG = {
#     'output_dir': './trellis_submit_outputs',
#     'device': 'cuda' if torch.cuda.is_available() else 'cpu',
#     # 'num_inference_steps_t2i': 8,
#     'num_inference_steps_t2i': 7,
#     'flux_model_url': "https://huggingface.co/gokaygokay/flux-game/blob/main/hyperflux_00001_.q8_0.gguf",
#     # 'flux_model_url': "black-forest-labs/FLUX.1-dev",
#     'flux_base_model': "camenduru/FLUX.1-dev-diffusers",
#     # 'flux_base_model': "black-forest-labs/FLUX.1-dev",
#     'sdxl_model_path': "stabilityai/stable-diffusion-xl-base-1.0",
#     'sd15_model_path': "runwayml/stable-diffusion-v1-5",
#     'hunyuan_model_path': "Tencent-Hunyuan/HunyuanDiT-v1.1-Diffusers-Distilled",
#     'trellis_model_path': 'cavargas10/TRELLIS',
#     'save_intermediate_outputs': True,
#     'save_preview': False,
#     'auto_compress_ply': True,
#     # Model selection
#     'current_model': 'flux',  # 'flux', 'sdxl', or 'sd15'
#     # TRELLIS specific settings - OPTIMIZED FOR MAXIMUM QUALITY
#     # 'guidance_scale': 4.0,  # Increased from 3.5 for better quality
#     'guidance_scale': 3.5,  # Increased from 3.5 for better quality
#     'ss_guidance_strength': 9.5,  # Increased from 8.5 for stronger structure guidance
#     # 'ss_sampling_steps': 30,  # Increased from 23 for more refinement
#     'ss_sampling_steps': 30,  # Increased from 23 for more refinement
#     # 'slat_guidance_strength': 5.0,  # Increased from 4.0 for better detail preservation
#     'slat_guidance_strength': 4.0,  # Increased from 4.0 for better detail preservation
#     # 'slat_sampling_steps': 30,  # Increased from 24 for more refinement
#     'slat_sampling_steps': 36,  # Increased from 24 for more refinement
#     # Memory management
#     'enable_memory_efficient_attention': True,
#     'enable_cpu_offload': True,
#     'max_memory_usage_gb': 20,
#     'validation_server_url': 'http://127.0.0.1:10006',
#     'auto_validate_generations': True,
#     'validation_timeout': 120,
#     # Object centering settings
#     'enable_object_centering': True,
#     'centering_white_threshold': 240,
#     'centering_padding': 30,
#     # LoRA configuration
#     'current_lora': None,
#     'lora_scale': 1.0,
#     # HunyuanDiT specific settings
#     'hunyuan_num_inference_steps': 25,
#     'hunyuan_pag_scale': 1.3,
#     'hunyuan_width': 1024,
#     'hunyuan_height': 1024,
#     # TRELLIS precision (use half-precision to reduce memory and speed up)
#     'trellis_use_fp16': True,
#     # TRELLIS torch.compile acceleration
#     'trellis_compile': False,
#     'trellis_compile_mode': 'reduce-overhead',  # options: 'reduce-overhead', 'max-autotune' (if supported)
#     # 'trellis_compile_mode': 'max-autotune',
#     'trellis_compile_dynamic': False,
#     'trellis_compile_flow_models': False,
#     # FLUX schnell mode (4-step fast inference)
#     'flux_use_schnell_socket': True,
#     'flux_use_schnell': False,
#     'flux_schnell_steps': 4,
#     'flux_schnell_guidance': 0.0,
# }


# Configuration
GENERATION_CONFIG = {
    'output_dir': './trellis_submit_outputs',
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    # 'num_inference_steps_t2i': 8,
    'num_inference_steps_t2i': 7,
    'flux_model_url': "https://huggingface.co/gokaygokay/flux-game/blob/main/hyperflux_00001_.q8_0.gguf",
    # 'flux_model_url': "black-forest-labs/FLUX.1-dev",
    'flux_base_model': "camenduru/FLUX.1-dev-diffusers",
    # 'flux_base_model': "black-forest-labs/FLUX.1-dev",
    'sdxl_model_path': "stabilityai/stable-diffusion-xl-base-1.0",
    'sd15_model_path': "runwayml/stable-diffusion-v1-5",
    'hunyuan_model_path': "Tencent-Hunyuan/HunyuanDiT-v1.1-Diffusers-Distilled",
    'trellis_model_path': 'cavargas10/TRELLIS',
    'save_intermediate_outputs': True,
    'save_preview': False,
    'auto_compress_ply': True,
    # Model selection
    'current_model': 'flux',  # 'flux', 'sdxl', or 'sd15'
    # TRELLIS specific settings - OPTIMIZED FOR MAXIMUM QUALITY
    # 'guidance_scale': 4.0,  # Increased from 3.5 for better quality
    'guidance_scale': 3.5,  # Increased from 3.5 for better quality
    'ss_guidance_strength': 7.5,  # Increased from 8.5 for stronger structure guidance
    # 'ss_sampling_steps': 30,  # Increased from 23 for more refinement
    'ss_sampling_steps': 21,  # Increased from 23 for more refinement
    # 'slat_guidance_strength': 5.0,  # Increased from 4.0 for better detail preservation
    'slat_guidance_strength': 3.5,  # Increased from 4.0 for better detail preservation
    # 'slat_sampling_steps': 30,  # Increased from 24 for more refinement
    'slat_sampling_steps': 24,  # Increased from 24 for more refinement
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
    # LoRA configuration
    'current_lora': None,
    'lora_scale': 1.0,
    # HunyuanDiT specific settings
    'hunyuan_num_inference_steps': 25,
    'hunyuan_pag_scale': 1.3,
    'hunyuan_width': 1024,
    'hunyuan_height': 1024,
    # TRELLIS precision (use half-precision to reduce memory and speed up)
    'trellis_use_fp16': True,
    # TRELLIS torch.compile acceleration
    'trellis_compile': False,
    'trellis_compile_mode': 'reduce-overhead',  # options: 'reduce-overhead', 'max-autotune' (if supported)
    # 'trellis_compile_mode': 'max-autotune',
    'trellis_compile_dynamic': False,
    'trellis_compile_flow_models': False,
    'flux_use_schnell_socket': False,
    'flux_use_schnell': True,
    'flux_schnell_steps': 6,
    'flux_schnell_guidance': 0.0,
}

# LoRA definitions
FLUX_LORAS = {
    'isometric_3d': {
        'name': 'Flux Isometric 3D',
        'repo': 'strangerzonehf/Flux-Isometric-3D-LoRA',
        'trigger_prefix': 'Isometric 3D,',
        'scale': 1.0,
        'description': 'Isometric 3D style LoRA for FLUX'
    },
    # 'live_3d': {
    #     'name': 'FLUX Live 3D',
    #     'repo': 'Shakker-Labs/FLUX.1-dev-LoRA-live-3D',
    #     'weight_name': 'FLUX-dev-lora-live_3D.safetensors',
    #     'trigger_prefix': '',
    #     'scale': 1.1,
    #     'fuse': False,  # Disabled fusion due to tensor size mismatch
    #     'description': 'Live 3D style LoRA for FLUX'
    # },
    'live_3d': {
        'name': 'FLUX Live 3D',
        'weight_name': '/home/mbhat/three-gen-subnet-trellis/LORAS/lowPolyStyle_Sora.safetensors',
        'trigger_prefix': 'Low Poly Style',
        'scale': 1.1,
        'fuse': False,  # Disabled fusion due to tensor size mismatch
        'description': 'Live 3D style LoRA for FLUX'
    },
    'game_assets': {
        'name': '3D Game Assets',
        'path': '/home/mbhat/three-gen-subnet-trellis/LORAS/game-assets.safetensors',
        'trigger_prefix': 'Create 3D game asset, isometric view version,',
        'scale': 1.0,
        'description': '3D game assets style LoRA for FLUX'
    },
    # 'patched_realism': {
    #     'name': 'Patched Realism',
    #     'path': 'patched_realism_LoRA.safetensors',
    #     'trigger_prefix': '',
    #     'scale': 1.0,
    #     'description': 'Realism enhancement LoRA for FLUX'
    # },
    'patched_realism': {
        'name': 'Patched Realism',
        'path': '/home/mbhat/three-gen-subnet-trellis/LORAS/patched_ZKcZdffUM6qyMYiEE8ed0_adapter_model_comfy_converted.safetensors',
        'trigger_prefix': 'Convert this image to low poly version,',
        'scale': 1.0,
        'description': 'Realism enhancement LoRA for FLUX'
    },
    'tf2_style': {
        'name': 'Team Fortress 2 Style',
        'path': '/home/mbhat/three-gen-subnet-trellis/LORAS/Team_Fortress_2_Style_F1D.safetensors',
        'trigger_prefix': 'tf2style,',
        'scale': 1.0,
        'description': 'Team Fortress 2 style LoRA for FLUX'
    },
    'baolei': {
        'name': 'Baolei Style',
        'path': '/home/mbhat/three-gen-subnet-trellis/LORAS/baolei.safetensors',
        'trigger_prefix': 'Cartoon-style design,',
        'scale': 1.0,
        'description': 'Baolei cartoon style LoRA for FLUX'
    },
    'cartoon_3d': {
        'name': 'Cartoon 3D Render',
        'path': '/home/mbhat/three-gen-subnet-trellis/LORAS/Cartoon 3D Render Style.safetensors',
        'trigger_prefix': '',
        'scale': 1.0,
        'description': 'Cartoon 3D render style LoRA for FLUX'
    },
    'cinema': {
        'name': 'Cinema Style',
        'path': '/home/mbhat/three-gen-subnet-trellis/LORA/everyday_000002000.safetensors',
        'trigger_prefix': 'c1n3ma,',
        'scale': 1.0,
        'description': 'Cinema style LoRA for FLUX'
    },
    'necklace': {
        'name': 'Necklace Style',
        'path': '/home/mbhat/three-gen-subnet-trellis/LORAS/necklace.safetensors',
        'trigger_prefix': 'NSHOpalite ',
        'scale': 1.0,
        'description': 'Necklace style LoRA for FLUX'
    }
}

# SD1.5 LoRA definitions
SD15_LORAS = {
    'game_icon': {
        'name': 'Game Icon Institute',
        'path': '/home/mbhat/three-gen-subnet-trellis/LORAS/GameIconResearch_TOY_Lora.safetensors',
        'trigger_prefix': 'game icon institute,',
        'scale': 1.0,
        'description': 'Game icon style LoRA for SD1.5'
    },
    'necklace': {
        'name': 'Necklace Style',
        'path': '/home/mbhat/three-gen-subnet-trellis/LORAS/necklace.safetensors',
        'trigger_prefix': 'NSHOpalite ',
        'scale': 1.0,
        'description': 'Necklace style LoRA for SD1.5'
    }
}

# HunyuanDiT LoRA definitions
HUNYUAN_LORAS = {
    'isometric_3d': {
        'name': 'HunyuanDiT Isometric 3D',
        'description': 'Isometric 3D style for HunyuanDiT',
        'trigger_prefix': 'isometric 3d,',
        'scale': 1.0
    },
    'live_3d': {
        'name': 'HunyuanDiT Live 3D',
        'description': 'Live 3D style for HunyuanDiT',
        'trigger_prefix': 'live 3d,',
        'scale': 1.0
    },
    'game_assets': {
        'name': 'HunyuanDiT Game Assets',
        'description': '3D game assets style for HunyuanDiT',
        'trigger_prefix': 'game asset, 3d model,',
        'scale': 1.0
    },
    'patched_realism': {
        'name': 'HunyuanDiT Patched Realism',
        'description': 'Realism enhancement for HunyuanDiT',
        'trigger_prefix': 'realistic, detailed,',
        'scale': 1.0
    },
    'tf2_style': {
        'name': 'HunyuanDiT TF2 Style',
        'description': 'Team Fortress 2 style for HunyuanDiT',
        'trigger_prefix': 'tf2 style,',
        'scale': 1.0
    },
    'baolei': {
        'name': 'HunyuanDiT Baolei Style',
        'description': 'Baolei cartoon style for HunyuanDiT',
        'trigger_prefix': 'cartoon style,',
        'scale': 1.0
    },
    'cartoon_3d': {
        'name': 'HunyuanDiT Cartoon 3D',
        'description': 'Cartoon 3D render style for HunyuanDiT',
        'trigger_prefix': 'cartoon 3d,',
        'scale': 1.0
    },
    'cinema': {
        'name': 'HunyuanDiT Cinema Style',
        'description': 'Cinema style for HunyuanDiT',
        'trigger_prefix': 'cinema,',
        'scale': 1.0
    },
    'sd15_game_icon': {
        'name': 'HunyuanDiT Game Icon',
        'description': 'Game icon style for HunyuanDiT',
        'trigger_prefix': 'game icon,',
        'scale': 1.0
    },
    'necklace': {
        'name': 'HunyuanDiT Necklace Style',
        'description': 'Necklace style for HunyuanDiT',
        'trigger_prefix': 'NSHOpalite ',
        'scale': 1.0
    }
}

from PIL import Image
from rembg import remove, new_session

class BackgroundRemover():
    def __init__(self, session=None, putalpha=True):
        self.session = new_session(session)
        self.putalpha = putalpha

    def __call__(self, image: Image.Image):
        output = remove(image, session=self.session, bgcolor=[255, 255, 255, 0], putalpha=self.putalpha)
        return output


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
    HUNYUAN_IMAGE = "hunyuan_image"
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
        elif asset_type == AssetType.HUNYUAN_IMAGE:
            file_path = self.asset_directory / "hunyuan_image.png"
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

class FluxSocketClient:
    """Client for communicating with isolated FLUX inference server"""
    
    def __init__(self, socket_path="/home/mbhat/three-gen-subnet-trellis/newcomer20_accurate/inferences.sock"):
        self.socket_path = socket_path
        self.connection = None
        
    def _ensure_connection(self):
        """Ensure socket connection is established"""
        try:
            if self.connection is None or hasattr(self.connection, 'closed') and self.connection.closed:
                self.connection = Client(self.socket_path)
        except Exception as e:
            print(f"⚠️ Failed to connect to FLUX socket: {e}")
            return False
        return True
    
    def generate_image(self, prompt: str, seed: int, width: int = 1024, height: int = 1024, **kwargs) -> Optional[Image.Image]:
        """Generate image via socket server"""
        try:
            if not self._ensure_connection():
                return None
                
            # Prepare request - match TextToImageRequest format
            request = {
                "prompt": prompt,
                "seed": seed,
                "width": width,
                "height": height
                # Note: guidance_scale, num_inference_steps are fixed by the server
                # guidance_scale=0.0, num_inference_steps=4
            }
            
            # Send request
            self.connection.send_bytes(json.dumps(request).encode('utf-8'))
            
            # Receive image data
            image_data = self.connection.recv_bytes()
            
            # Convert to PIL Image
            image = Image.open(io.BytesIO(image_data))
            return image
            
        except Exception as e:
            print(f"❌ Socket-based FLUX generation failed: {e}")
            return None

class TrellisGenerator:
    def __init__(self):
        # Initialize model instance variables
        self.flux_pipeline = None
        self.flux_transformer = None
        self.flux_text_encoder_2 = None
        self.sdxl_pipeline = None
        self.sd15_pipeline = None
        self.hunyuan_pipeline = None
        self.trellis_pipeline = None
        self.background_remover = None
        self.load_rembg = True
        self.metrics = GenerationMetrics()
        self.generation_lock = threading.Lock()
        self.flux_use_schnell_socket = False
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
        if GENERATION_CONFIG.get('flux_use_schnell_socket', False):
            self._start_flux_server()        
        Path(GENERATION_CONFIG['output_dir']).mkdir(exist_ok=True)
        print("🔧 TRELLIS Generator initialized")
        self.ready = True
        

    def _start_flux_server(self):
        """Start FLUX server as subprocess"""
        try:
            print("🚀 Starting FLUX server...")
            flux_dir = os.path.join(os.path.dirname(__file__), "..", "test_schnell")
            self.flux_process = subprocess.Popen(
                ["uv", "run", "python", "main.py"],
                cwd=flux_dir,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )
            
            # Wait for server to be ready
            time.sleep(5)
            print("✅ FLUX server started")
            
        except Exception as e:
            print(f"❌ Failed to start FLUX server: {e}")
            self.flux_process = None

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
            single_file_base_model = GENERATION_CONFIG['flux_base_model']
            # If schnell mode is enabled, prefer the schnell checHF_HUB_CACHEkpoint for the base repo
            if GENERATION_CONFIG.get('flux_use_schnell', False) and not GENERATION_CONFIG.get('flux_use_schnell_socket', False):
                try:
                    # single_file_base_model = "black-forest-labs/FLUX.1-schnell"
                    single_file_base_model = "manbeast3b/flux.1-schnell-full1"
                    file_url = None
                    print("⚡ Using FLUX.1-schnell base model (schnell mode enabled)")
                except Exception:
                    pass
                    
            if GENERATION_CONFIG.get('flux_use_schnell_socket', False):
                # Try to use socket-based FLUX first
                try:
                    print("🔧 Initializing FLUX socket client...")
                    self.flux_socket_client = FluxSocketClient()
                    
                    # Test connection with a simple generation
                    test_image = self.flux_socket_client.generate_image(
                        prompt="test", 
                        seed=42, 
                        width=512, 
                        height=512
                    )
                    
                    if test_image is not None:
                        print("✅ FLUX socket client initialized successfully")
                        self.flux_use_schnell_socket = True
                        return
                    else:
                        print("⚠️ FLUX socket client failed, falling back to direct loading")
                        self.flux_use_schnell_socket = False
                        
                except Exception as e:
                    print(f"⚠️ FLUX socket client failed: {e}")
                    self.flux_use_schnell_socket = False
                

            
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
            
            # Load transformer
            # If a direct file is provided (e.g., .gguf/.safetensors/.ckpt or http URL), use from_single_file.
            # Otherwise, load from the base repo via from_pretrained.
            use_single_file = False
            if file_url is not None and 'gguf' in file_url:
                use_single_file = True
                file_url = file_url.replace("/resolve/main/", "/blob/main/").replace("?download=true", "")
            elif isinstance(file_url, str):
                    lower_url = file_url.lower()
                    if lower_url.startswith("http://") or lower_url.startswith("https://"):
                        use_single_file = True
                        # Ensure we use the raw file endpoint for Hugging Face links
                        if "huggingface.co" in lower_url and "/blob/" in lower_url:
                            file_url = file_url.replace("/blob/", "/resolve/")
                    elif lower_url.endswith((".gguf", ".safetensors", ".ckpt")):
                        use_single_file = True

            if use_single_file:
                print("Loading FLUX transformer from single file (GGUF/ckpt)...")
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

            else:
                # print("Loading FLUX transformer from repo (no single file provided)...")
                # self.flux_transformer = FluxTransformer2DModel.from_pretrained(
                #     single_file_base_model,
                #     subfolder="transformer",
                #     torch_dtype=dtype,
                #     token=huggingface_token
                # )
                # Initialize pipeline
                print("Initializing FLUX pipeline...")
                # self.flux_pipeline = FluxPipeline.from_pretrained(
                #     "manbeast3b/flux.1-schnell-full1",
                #     text_encoder_2=self.flux_text_encoder_2, 
                #     torch_dtype=dtype, 
                #     token=huggingface_token
                # )
                # vae=AutoencoderTiny.from_pretrained(
                #     "RobertML/FLUX.1-schnell-vae_e3m2",
                #     revision="da0d2cd7815792fb40d084dbd8ed32b63f153d8d",
                #     torch_dtype=torch.bfloat16
                # )
                # transformer_path = os.path.join(
                #     HF_HUB_CACHE,
                #     "models--RobertML--FLUX.1-schnell-int8wo/snapshots/307e0777d92df966a3c0f99f31a6ee8957a9857a"
                # )
                # transformer=FluxTransformer2DModel.from_pretrained(
                #     # transformer_path,
                #     "RobertML/FLUX.1-schnell-int8wo",  # model repo id
                #     revision="307e0777d92df966a3c0f99f31a6ee8957a9857a",
                #     torch_dtype=torch.bfloat16,
                #     use_safetensors=False, 
                #     # local_files_only=True
                # )
                self.flux_pipeline = DiffusionPipeline.from_pretrained(
                    "black-forest-labs/FLUX.1-schnell",
                    # vae=vae,
                    revision="741f7c3ce8b383c54771c7003378a50191e9efe9",
                    # transformer=transformer,
                    text_encoder_2=self.flux_text_encoder_2,
                    torch_dtype=torch.bfloat16,
                )
                        
            self.flux_pipeline.to("cuda")

            from flux_caching import apply_cache_on_pipe
            apply_cache_on_pipe(self.flux_pipeline)
            self.flux_pipeline.to(memory_format=torch.channels_last)
            self.flux_pipeline.vae = torch.compile(self.flux_pipeline.vae, mode="max-autotune")
            # if GENERATION_CONFIG.get('flux_use_schnell', False):
            # self.flux_pipeline.transformer = torch.compile(self.flux_pipeline.transformer, mode="max-autotune")
    
            # from torchao.quantization import quantize_, float8_dynamic_activation_float8_weight
            # quantize_(self.flux_pipeline.vae, float8_dynamic_activation_float8_weight())
            
            print("✅ FLUX models loaded successfully")

            # Optional warmup for schnell mode (2 quick passes)
            if GENERATION_CONFIG.get('flux_use_schnell', False):
                try:
                    print("🧪 Warming up FLUX.1-schnell (2x empty prompt, 4 steps)...")
                    for _ in range(2):
                        gc.collect()
                        with torch.no_grad():
                            _ = self.flux_pipeline(
                                prompt="",
                                width=1024,
                                height=1024,
                                guidance_scale=GENERATION_CONFIG.get('flux_schnell_guidance', 0.0),
                                num_inference_steps=GENERATION_CONFIG.get('flux_schnell_steps', 4)
                            )
                    print("✓ FLUX schnell warmup complete")
                except Exception as we:
                    print(f"⚠️ FLUX schnell warmup skipped: {we}")

            if GENERATION_CONFIG.get('flux_use_schnell_socket', False):
                try:
                    print("🔧 Warming up FLUX socket client...")
                    for _ in range(2):
                        gc.collect()
                        with torch.no_grad():
                            _ = self.flux_socket_client.generate_image(
                                prompt="",
                                width=1024,
                                height=1024,
                            )
                        print("✓ FLUX socket client warmup complete")
                except Exception as we:
                    print(f"⚠️ FLUX socket client warmup skipped: {we}")
            
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

    def _load_sdxl_pipeline(self):
        """Load SDXL pipeline"""
        if self.sdxl_pipeline is not None:
            print("✓ SDXL pipeline already loaded")
            return
            
        print("🔧 Loading SDXL pipeline...")
        
        try:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            huggingface_token = os.getenv("HUGGINGFACE_TOKEN")
            
            # Load SDXL with optimizations
            self.sdxl_pipeline = StableDiffusionPipeline.from_pretrained(
                GENERATION_CONFIG['sdxl_model_path'],
                torch_dtype=torch.float16,
                use_safetensors=True,
                variant="fp16",
                token=huggingface_token
            )
            
            # Move to GPU and optimize
            self.sdxl_pipeline.to(device)
            self.sdxl_pipeline.enable_attention_slicing()
            self.sdxl_pipeline.enable_vae_slicing()
            
            # Enable memory efficient attention if available
            if hasattr(self.sdxl_pipeline, 'enable_xformers_memory_efficient_attention'):
                self.sdxl_pipeline.enable_xformers_memory_efficient_attention()
            
            print("✅ SDXL pipeline loaded successfully")
            
        except Exception as e:
            print(f"❌ SDXL pipeline loading failed: {e}")
            traceback.print_exc()
            self._unload_sdxl_pipeline()

    def _unload_sdxl_pipeline(self):
        """Unload SDXL pipeline to free GPU memory"""
        if self.sdxl_pipeline is not None:
            print("🧹 Unloading SDXL pipeline...")
            del self.sdxl_pipeline
            self.sdxl_pipeline = None
            self._clear_gpu_memory()
            print("✅ SDXL pipeline unloaded")

    def _load_sd15_pipeline(self):
        """Load SD1.5 pipeline"""
        if self.sd15_pipeline is not None:
            print("✓ SD1.5 pipeline already loaded")
            return
            
        print("🔧 Loading SD1.5 pipeline...")
        
        try:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            huggingface_token = os.getenv("HUGGINGFACE_TOKEN")
            
            # Load SD1.5 with optimizations
            self.sd15_pipeline = StableDiffusionPipeline.from_pretrained(
                GENERATION_CONFIG['sd15_model_path'],
                torch_dtype=torch.float16,
                use_safetensors=True,
                token=huggingface_token
            )
            
            # Move to GPU and optimize
            self.sd15_pipeline.to(device)
            self.sd15_pipeline.enable_attention_slicing()
            self.sd15_pipeline.enable_vae_slicing()
            
            # Enable memory efficient attention if available
            if hasattr(self.sd15_pipeline, 'enable_xformers_memory_efficient_attention'):
                self.sd15_pipeline.enable_xformers_memory_efficient_attention()
            
            print("✅ SD1.5 pipeline loaded successfully")
            
        except Exception as e:
            print(f"❌ SD1.5 pipeline loading failed: {e}")
            traceback.print_exc()
            self._unload_sd15_pipeline()

    def _unload_sd15_pipeline(self):
        """Unload SD1.5 pipeline to free GPU memory"""
        if self.sd15_pipeline is not None:
            print("🧹 Unloading SD1.5 pipeline...")
            del self.sd15_pipeline
            self.sd15_pipeline = None
            self._clear_gpu_memory()
            print("✅ SD1.5 pipeline unloaded")

    def _load_hunyuan_pipeline(self):
        """Load HunyuanDiT pipeline"""
        if self.hunyuan_pipeline is not None:
            print("✓ HunyuanDiT pipeline already loaded")
            return
            
        print("🔧 Loading HunyuanDiT pipeline...")
        
        try:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            
            # Initialize HunyuanDiT pipeline
            self.hunyuan_pipeline = HunyuanDiTPipeline(
                model_path=GENERATION_CONFIG['hunyuan_model_path'],
                device=device
            )
            
            # Compile for better performance (only once)
            try:
                print("   Compiling HunyuanDiT for better performance...")
                self.hunyuan_pipeline.compile()
                print("   ✓ HunyuanDiT compiled successfully")
            except Exception as e:
                print(f"   ⚠️ HunyuanDiT compilation failed: {e}")
                print("   Continuing without compilation...")
            
            print("✅ HunyuanDiT pipeline loaded successfully")
            
        except Exception as e:
            print(f"❌ HunyuanDiT pipeline loading failed: {e}")
            traceback.print_exc()
            self._unload_hunyuan_pipeline()
    
    def _unload_hunyuan_pipeline(self):
        """Unload HunyuanDiT pipeline to free GPU memory"""
        if self.hunyuan_pipeline is not None:
            print("🧹 Unloading HunyuanDiT pipeline...")
            del self.hunyuan_pipeline
            self.hunyuan_pipeline = None
            self._clear_gpu_memory()
            print("✅ HunyuanDiT pipeline unloaded")

    def _load_trellis_pipeline(self):
        """Load TRELLIS pipeline"""
        if self.trellis_pipeline is not None:
            print("✓ TRELLIS pipeline already loaded")
            return
            
        print("🔧 Loading TRELLIS pipeline...")
        
        try:
            # Always load the pipeline first
            use_fp16 = GENERATION_CONFIG.get('trellis_use_fp16', True)
            # Try to load directly in fp16 when supported; fall back gracefully
            try:
                self.trellis_pipeline = TrellisImageTo3DPipeline.from_pretrained(
                    GENERATION_CONFIG['trellis_model_path'],
                    torch_dtype=torch.float16 if use_fp16 else None
                )
            except TypeError:
                # Some implementations may not accept torch_dtype
                self.trellis_pipeline = TrellisImageTo3DPipeline.from_pretrained(
                    GENERATION_CONFIG['trellis_model_path']
                )

            # Move to device / dtype
            if torch.cuda.is_available():
                if use_fp16:
                    try:
                        # Preferred path if pipeline supports dtype argument on to()
                        self.trellis_pipeline.to("cuda", dtype=torch.float16)
                    except Exception:
                        # Fallback: move then cast if supported
                        self.trellis_pipeline.cuda()
                        if hasattr(self.trellis_pipeline, 'half'):
                            try:
                                self.trellis_pipeline.half()
                            except Exception:
                                pass
                else:
                    self.trellis_pipeline.cuda()
            else:
                # CPU fallback; cast to fp16 if requested and supported (has limited effect on CPU)
                if use_fp16 and hasattr(self.trellis_pipeline, 'to'):
                    try:
                        self.trellis_pipeline.to(dtype=torch.float16)
                    except Exception:
                        pass

            # Optionally compile modules after the pipeline is loaded
            if GENERATION_CONFIG.get('trellis_compile', False) and hasattr(torch, 'compile'):
                try:
                    mode = GENERATION_CONFIG.get('trellis_compile_mode', 'reduce-overhead')
                    dynamic = GENERATION_CONFIG.get('trellis_compile_dynamic', False)

                    compile_keys = [
                        'sparse_structure_decoder',
                        'slat_decoder_mesh',
                        'slat_decoder_gs',
                        'slat_decoder_rf',
                    ]
                    if GENERATION_CONFIG.get('trellis_compile_flow_models', False):
                        compile_keys.extend(['sparse_structure_flow_model', 'slat_flow_model'])

                    compiled_ok = []
                    compiled_fail = []
                    models_dict = getattr(self.trellis_pipeline, 'models', {}) or {}
                    for key in compile_keys:
                        module = models_dict.get(key)
                        if module is None:
                            continue
                        try:
                            compiled = torch.compile(module, mode=mode, dynamic=dynamic)
                            self.trellis_pipeline.models[key] = compiled
                            compiled_ok.append(key)
                        except Exception as ce:
                            compiled_fail.append((key, str(ce)))

                    if compiled_ok:
                        print(f"✅ TRELLIS compiled modules: {', '.join(compiled_ok)} (mode={mode}, dynamic={dynamic})")
                    if compiled_fail:
                        print("⚠️ TRELLIS compile failures:")
                        for key, err in compiled_fail:
                            print(f"   {key}: {err}")
                except Exception as e:
                    print(f"⚠️ TRELLIS compile setup failed: {e}")
            
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
            # self.background_remover = BackgroundRemover(session="u2netp", putalpha=True)
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

    def _load_lora(self, lora_key: str):
        """Load a specific LoRA onto the current pipeline"""
        current_model = GENERATION_CONFIG.get('current_model', 'flux')
        
        # Ensure the correct pipeline is loaded
        if current_model == 'flux' and self.flux_pipeline is None:
            print("🔧 Loading FLUX pipeline for LoRA...")
            self._load_flux_models()
        elif current_model == 'sdxl' and self.sdxl_pipeline is None:
            print("🔧 Loading SDXL pipeline for LoRA...")
            self._load_sdxl_pipeline()
        elif current_model == 'sd15' and self.sd15_pipeline is None:
            print("🔧 Loading SD1.5 pipeline for LoRA...")
            self._load_sd15_pipeline()
        
        # Select pipeline and LoRA configs based on current model
        if current_model == 'flux':
            pipeline = self.flux_pipeline
            lora_configs = FLUX_LORAS
        elif current_model == 'sdxl':
            pipeline = self.sdxl_pipeline
            lora_configs = SDXL_LORAS
        elif current_model == 'sd15':
            pipeline = self.sd15_pipeline
            lora_configs = SD15_LORAS
        else:
            print(f"❌ Unknown model type: {current_model}")
            return False
        
        if pipeline is None:
            print(f"❌ {current_model.upper()} pipeline not loaded. Load {current_model.upper()} models first.")
            return False
            
        if lora_key not in lora_configs:
            print(f"❌ LoRA '{lora_key}' not found in available {current_model.upper()} LoRAs")
            return False
            
        lora_config = lora_configs[lora_key]
        
        # Check if the same LoRA is already loaded
        current_lora = GENERATION_CONFIG.get('current_lora')
        if current_lora == lora_key:
            print(f"✅ {current_model.upper()} LoRA '{lora_config['name']}' is already loaded, skipping reload")
            return True
        
        print(f"🔧 Loading {current_model.upper()} LoRA: {lora_config['name']}")
        
        try:
            # Unload any existing LoRA first
            if hasattr(pipeline, 'unload_lora_weights'):
                pipeline.unload_lora_weights()
            
            # Load the new LoRA
            if 'repo' in lora_config:
                # Load from HuggingFace repo
                pipeline.load_lora_weights(lora_config['repo'])
                if 'weight_name' in lora_config:
                    pipeline.load_lora_weights(
                        lora_config['repo'], 
                        weight_name=lora_config['weight_name']
                    )
            elif 'path' in lora_config:
                # Load from local path
                lora_path = lora_config['path']
                if not os.path.exists(lora_path):
                    print(f"❌ LoRA file not found: {lora_path}")
                    return False
                
                # Check if patching is needed (only for FLUX)
                try:
                    pipeline.load_lora_weights(lora_path)
                except Exception as e:
                    if current_model == 'flux' and ("final_layer" in str(e) or "adaLN" in str(e)):
                        print(f"⚠️ LoRA needs patching, applying adaLN fix...")
                        # Apply patcher fix
                        import safetensors.torch
                        from safetensors import safe_open
                        
                        # Load and patch the LoRA
                        state_dict = {}
                        with safe_open(lora_path, framework="pt", device="cpu") as f:
                            for k in f.keys():
                                state_dict[k] = f.get_tensor(k)
                        
                        # Apply patch
                        state_dict = patch_final_layer_adaLN(state_dict, verbose=False)
                        
                        # Save patched version temporarily
                        patched_path = lora_path.replace('.safetensors', '_patched.safetensors')
                        safetensors.torch.save_file(state_dict, patched_path)
                        
                        # Load patched version
                        pipeline.load_lora_weights(patched_path)
                        print(f"✅ LoRA patched and loaded successfully")
                    else:
                        raise e
            
            # Apply fusion if specified
            if lora_config.get('fuse', False):
                try:
                    pipeline.fuse_lora(lora_scale=lora_config.get('scale', 1.0))
                    print(f"   🔗 LoRA fused with scale {lora_config.get('scale', 1.0)}")
                except Exception as fusion_error:
                    print(f"   ⚠️ LoRA fusion failed: {fusion_error}")
                    print(f"   📝 Continuing without fusion...")
            
            # Update current LoRA configuration
            GENERATION_CONFIG['current_lora'] = lora_key
            GENERATION_CONFIG['lora_scale'] = lora_config.get('scale', 1.0)
            
            print(f"✅ {current_model.upper()} LoRA '{lora_config['name']}' loaded successfully")
            return True
            
        except Exception as e:
            print(f"❌ Failed to load {current_model.upper()} LoRA '{lora_key}': {e}")
            traceback.print_exc()
            return False

    def _unload_lora(self):
        """Unload current LoRA from current pipeline"""
        current_model = GENERATION_CONFIG.get('current_model', 'flux')
        
        # Determine which pipeline to use
        if current_model == 'flux':
            pipeline = self.flux_pipeline
        elif current_model == 'sdxl':
            pipeline = self.sdxl_pipeline
        elif current_model == 'sd15':
            pipeline = self.sd15_pipeline
        else:
            print(f"❌ Unknown model type: {current_model}")
            return
        
        if pipeline is None:
            return
            
        try:
            if hasattr(pipeline, 'unload_lora_weights'):
                pipeline.unload_lora_weights()
                print(f"✅ {current_model.upper()} LoRA unloaded successfully")
            
            GENERATION_CONFIG['current_lora'] = None
            GENERATION_CONFIG['lora_scale'] = 1.0
            
        except Exception as e:
            print(f"⚠️ Error unloading {current_model.upper()} LoRA: {e}")

    def get_available_loras(self) -> Dict[str, Any]:
        """Get list of available LoRAs for current model"""
        current_model = GENERATION_CONFIG.get('current_model', 'flux')
        
        if current_model == 'flux':
            lora_configs = FLUX_LORAS
        elif current_model == 'sdxl':
            lora_configs = SDXL_LORAS
        elif current_model == 'sd15':
            lora_configs = SD15_LORAS
        else:
            return {}
        
        return {
            key: {
                'name': config['name'],
                'description': config['description'],
                'trigger_prefix': config['trigger_prefix'],
                'scale': config.get('scale', 1.0),
                'loaded': GENERATION_CONFIG['current_lora'] == key
            }
            for key, config in lora_configs.items()
        }
    


    def _resolve_flux_inference_params(self, guidance_scale: float, steps: int) -> Tuple[float, int, Dict[str, Any]]:
        """Apply schnell overrides for FLUX if enabled.

        Returns:
            (effective_guidance_scale, effective_steps, extra_kwargs)
        """
        extra_kwargs: Dict[str, Any] = {}
        if GENERATION_CONFIG.get('flux_use_schnell', False):
            guidance_scale = GENERATION_CONFIG.get('flux_schnell_guidance', 0.0)
            steps = GENERATION_CONFIG.get('flux_schnell_steps', 4)
            extra_kwargs['max_sequence_length'] = 256
        return guidance_scale, steps, extra_kwargs

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

    def generate_hunyuan_image(self, prompt: str, seed: int = 42, lora_key: str = None) -> Optional[Image.Image]:
        """Generate image using HunyuanDiT with optional LoRA"""
        try:
            print(f"🎨 Generating HunyuanDiT image for: '{prompt}' (seed: {seed})")
            
            # Load HunyuanDiT if not loaded
            if self.hunyuan_pipeline is None:
                self._load_hunyuan_pipeline()
            
            # Apply LoRA trigger prefix if specified
            enhanced_prompt = prompt
            if lora_key and lora_key in HUNYUAN_LORAS:
                lora_config = HUNYUAN_LORAS[lora_key]
                trigger_prefix = lora_config.get('trigger_prefix', '')
                if trigger_prefix:
                    enhanced_prompt = f"{trigger_prefix} {prompt}"
                    print(f"🎨 Applied HunyuanDiT LoRA trigger prefix: '{trigger_prefix}'")
            
            # Generate image with HunyuanDiT
            with torch.no_grad():
                image = self.hunyuan_pipeline(
                    prompt=enhanced_prompt,
                    seed=seed
                )
            
            print("✅ HunyuanDiT image generated successfully")
            
            # Create asset and save the image
            generation_asset = self.asset_manager.create_asset(prompt, seed)
            generation_asset.add_asset(AssetType.HUNYUAN_IMAGE, image)
            print(f"💾 HunyuanDiT image saved to: {generation_asset.asset_directory}")
            
            return image
            
        except Exception as e:
            print(f"❌ HunyuanDiT image generation failed: {e}")
            traceback.print_exc()
            return None

    def generate_3d_model(self, prompt: str, seed: int = 42, num_inference_steps: Optional[int] = None, guidance_scale: Optional[float] = None, ss_sampling_steps: Optional[int] = None, slat_sampling_steps: Optional[int] = None, slat_guidance_strength: Optional[float] = None, ss_guidance_strength: Optional[float] = None) -> Optional[Tuple[bytes, Optional[bytes]]]:
        """Generate 3D model from text prompt using FLUX + TRELLIS pipeline"""
        
        job_id = f"gen_{int(time.time())}_{seed}"
        generation_job_status.update({
            "current_job_id": job_id,
            "status": "processing",
            "prompt": prompt,
            "seed": seed,
            "start_time": time.time(),
            "end_time": None,
            "ply_path": None,
            "error": None
        })

        with self.generation_lock:
            start_time = time.time()
            
            try:
                print(f"🎯 Starting TRELLIS generation for: '{prompt}' (seed: {seed})")
                
                # Initialize asset manager for this generation
                generation_asset = self.asset_manager.create_asset(prompt, seed)
                
                # Step 1: Generate image with selected model
                current_model = GENERATION_CONFIG.get('current_model', 'flux')
                print(f"Step 1: Generating image with {current_model.upper()}...")
                
                device = "cuda" if torch.cuda.is_available() else "cpu"
                
                # Enhanced prompt with LoRA trigger prefix if applicable
                enhanced_prompt = prompt
                current_lora = GENERATION_CONFIG.get('current_lora')
                
                if current_model == 'flux':
                    if self.flux_pipeline is None:
                        self._load_flux_models()
                    
                    if current_lora and current_lora in FLUX_LORAS:
                        lora_config = FLUX_LORAS[current_lora]
                        trigger_prefix = lora_config.get('trigger_prefix', '')
                        if trigger_prefix:
                            enhanced_prompt = f"{trigger_prefix} {prompt}"
                            print(f"🎨 Applied FLUX LoRA trigger prefix: '{trigger_prefix}'")
                    
                    generator = torch.Generator(device=device).manual_seed(seed)
                    with torch.no_grad():
                        effective_guidance_scale = guidance_scale if guidance_scale is not None else GENERATION_CONFIG['guidance_scale']
                        effective_steps = num_inference_steps if num_inference_steps is not None else NUM_INFERENCE_STEPS
                        effective_guidance_scale, effective_steps, extra_kwargs = self._resolve_flux_inference_params(
                            effective_guidance_scale,
                            effective_steps
                        )
                        if GENERATION_CONFIG.get('flux_use_schnell_socket', False):
                            image = self.flux_socket_client.generate_image(
                                prompt=enhanced_prompt,
                                seed=seed,
                                width=1024,
                                height=1024,
                            )
                        else:   
                            image = self.flux_pipeline(
                                prompt=enhanced_prompt,
                                guidance_scale=effective_guidance_scale,
                                num_inference_steps=effective_steps,
                                width=1024,
                                height=1024,
                                generator=generator,
                                **extra_kwargs,
                            ).images[0]
                
                elif current_model == 'sdxl':
                    if self.sdxl_pipeline is None:
                        self._load_sdxl_pipeline()
                    
                    if current_lora and current_lora in SDXL_LORAS:
                        lora_config = SDXL_LORAS[current_lora]
                        trigger_prefix = lora_config.get('trigger_prefix', '')
                        if trigger_prefix:
                            enhanced_prompt = f"{trigger_prefix} {prompt}"
                            print(f"🎨 Applied SDXL LoRA trigger prefix: '{trigger_prefix}'")
                    
                    generator = torch.Generator(device=device).manual_seed(seed)
                    with torch.no_grad():
                        effective_guidance_scale = guidance_scale if guidance_scale is not None else 7.5
                        effective_steps = num_inference_steps if num_inference_steps is not None else 25
                        image = self.sdxl_pipeline(
                            prompt=enhanced_prompt,
                            guidance_scale=effective_guidance_scale,
                            num_inference_steps=effective_steps,
                            width=1024,
                            height=1024,
                            generator=generator,
                        ).images[0]
                
                elif current_model == 'sd15':
                    if self.sd15_pipeline is None:
                        self._load_sd15_pipeline()
                    
                    if current_lora and current_lora in SD15_LORAS:
                        lora_config = SD15_LORAS[current_lora]
                        trigger_prefix = lora_config.get('trigger_prefix', '')
                        if trigger_prefix:
                            enhanced_prompt = f"{trigger_prefix} {prompt}"
                            print(f"🎨 Applied SD1.5 LoRA trigger prefix: '{trigger_prefix}'")
                    
                    generator = torch.Generator(device=device).manual_seed(seed)
                    with torch.no_grad():
                        effective_guidance_scale = guidance_scale if guidance_scale is not None else 7.5
                        effective_steps = num_inference_steps if num_inference_steps is not None else 25
                        image = self.sd15_pipeline(
                            prompt=enhanced_prompt,
                            guidance_scale=effective_guidance_scale,
                            num_inference_steps=effective_steps,
                            width=512,
                            height=512,
                            generator=generator,
                        ).images[0]
                
                else:
                    raise ValueError(f"Unknown model type: {current_model}")
                
                print(f"✓ {current_model.upper()} image generated successfully")
                generation_asset.add_asset(AssetType.FLUX_IMAGE, image)  # Keep same asset type for compatibility
                
                # Unload FLUX models
                # self._unload_flux_models()
                
                # # Step 1.3: Center object in image before background removal
                # if GENERATION_CONFIG.get('enable_object_centering', True):
                #     print("Step 1.3: Centering object in image...")
                #     try:
                #         centered_image = self.center_object_in_image(
                #             image, 
                #             white_threshold=GENERATION_CONFIG.get('centering_white_threshold', 240),
                #             padding=GENERATION_CONFIG.get('centering_padding', 40)
                #         )
                #         print("✓ Object centered successfully")
                #         image = centered_image  # Use the centered image for next steps
                #         generation_asset.add_asset(AssetType.FLUX_IMAGE, centered_image)  # Update asset with centered version
                #     except Exception as e:
                #         print(f"⚠️ Object centering failed: {e}")
                #         print("   Continuing with original image...")
                # else:
                #     print("Step 1.3: Object centering disabled, skipping...")
                
                # Step 1.5: Remove background from image
                print("Step 1.5: Removing background from image...")
                if self.background_remover is None:
                    self._load_background_remover()
                
                try:
                    image_no_bg = self.background_remover(image)
                    print("✓ Background removed successfully")
                    # Save the background-removed image as well
                    generation_asset.add_asset(AssetType.FLUX_IMAGE, image_no_bg)  # Replace original with cleaned version
                    image = image_no_bg  # Use the cleaned image for TRELLIS
                except Exception as e:
                    print(f"⚠️ Background removal failed: {e}")
                    print("   Continuing with original image...")
                
                # Unload background remover
                # self._unload_background_remover()
                
                # Step 2: Generate 3D model with TRELLIS
                print("Step 2: Generating 3D model with TRELLIS...")
                if self.trellis_pipeline is None:   
                    self._load_trellis_pipeline()
                    if self.trellis_pipeline is None:
                        raise RuntimeError("TRELLIS pipeline failed to load; cannot generate 3D model.")
                
                # Enhanced TRELLIS parameters for maximum quality
                # Resolve TRELLIS quality parameters with overrides
                effective_ss_steps = ss_sampling_steps if ss_sampling_steps is not None else GENERATION_CONFIG['ss_sampling_steps']
                effective_slat_steps = slat_sampling_steps if slat_sampling_steps is not None else GENERATION_CONFIG['slat_sampling_steps']
                effective_slat_guidance = slat_guidance_strength if slat_guidance_strength is not None else GENERATION_CONFIG['slat_guidance_strength']
                effective_ss_guidance = ss_guidance_strength if ss_guidance_strength is not None else GENERATION_CONFIG['ss_guidance_strength']

                # Use autocast to reduce activation memory and speed up compute on CUDA
                use_fp16 = GENERATION_CONFIG.get('trellis_use_fp16', True) and torch.cuda.is_available()
                if use_fp16:
                    try:
                        with torch.autocast(device_type="cuda", dtype=torch.float16):
                            outputs = self.trellis_pipeline.run(
                                image,
                                seed=seed,
                                formats=["gaussian"],
                                preprocess_image=False,
                                sparse_structure_sampler_params={
                                    "steps": effective_ss_steps,
                                    "cfg_strength": effective_ss_guidance,
                                    "cfg_interval": (0.3, 0.98),  # Enhanced guidance scheduling
                                    "rescale_t": 3.0,  # Temperature rescaling for better quality
                                },
                                slat_sampler_params={
                                    "steps": effective_slat_steps,
                                    "cfg_strength": effective_slat_guidance,
                                    "cfg_interval": (0.3, 0.98),  # Enhanced guidance scheduling
                                    "rescale_t": 3.0,  # Temperature rescaling for better quality
                                },
                            )
                    except RuntimeError as e:
                        # Some mesh decoding ops may not support fp16 (scatter/scatter_reduce dtype issues)
                        if "scatter()" in str(e) or "scatter_reduce" in str(e):
                            print("⚠️ FP16 mesh decode failed (scatter dtype mismatch). Retrying gaussian-only without autocast...")
                            with torch.autocast(device_type="cuda", enabled=False):
                                outputs = self.trellis_pipeline.run(
                                    image,
                                    seed=seed,
                                    formats=["gaussian"],  # Avoid mesh path in fp16
                                    preprocess_image=False,
                                    sparse_structure_sampler_params={
                                        "steps": effective_ss_steps,
                                        "cfg_strength": effective_ss_guidance,
                                        "cfg_interval": (0.3, 0.98),
                                        "rescale_t": 3.0,
                                    },
                                    slat_sampler_params={
                                        "steps": effective_slat_steps,
                                        "cfg_strength": effective_slat_guidance,
                                        "cfg_interval": (0.3, 0.98),
                                        "rescale_t": 3.0,
                                    },
                                )
                        else:
                            raise
                else:
                    outputs = self.trellis_pipeline.run(
                        image,
                        seed=seed,
                        formats=["gaussian", "mesh"],
                        preprocess_image=False,
                        sparse_structure_sampler_params={
                            "steps": effective_ss_steps,
                            "cfg_strength": effective_ss_guidance,
                            "cfg_interval": (0.3, 0.98),  # Enhanced guidance scheduling
                            "rescale_t": 3.0,  # Temperature rescaling for better quality
                        },
                        slat_sampler_params={
                            "steps": effective_slat_steps,
                            "cfg_strength": effective_slat_guidance,
                            "cfg_interval": (0.3, 0.98),  # Enhanced guidance scheduling
                            "rescale_t": 3.0,  # Temperature rescaling for better quality
                        },
                    )
                
                print("✓ 3D model generated successfully")
                
                # Step 3: Extract and enhance Gaussian Splatting PLY
                print("Step 3: Extracting and enhancing Gaussian Splatting PLY...")
                gaussian_output = outputs['gaussian'][0]
                
                # Quality enhancement: Filter low-quality splats
                # print("   Enhancing quality by filtering low-quality splats...")
                # try:
                #     # Get splat data
                #     points = gaussian_output.points
                #     opacities = gaussian_output.opacities
                #     scales = gaussian_output.scales
                    
                #     # Filter out low-opacity and very small splats
                #     opacity_threshold = 0.01
                #     scale_threshold = 0.001
                    
                #     # Create quality mask
                #     quality_mask = (opacities > opacity_threshold) & (torch.norm(scales, dim=1) > scale_threshold)
                    
                #     if quality_mask.sum() > 7000:  # Ensure minimum splat count
                #         # Apply filtering
                #         gaussian_output.points = points[quality_mask]
                #         gaussian_output.opacities = opacities[quality_mask]
                #         gaussian_output.scales = scales[quality_mask]
                #         gaussian_output.rotations = gaussian_output.rotations[quality_mask]
                #         gaussian_output.features_dc = gaussian_output.features_dc[quality_mask]
                #         gaussian_output.features_rest = gaussian_output.features_rest[quality_mask]
                #         gaussian_output.normals = gaussian_output.normals[quality_mask]
                        
                #         print(f"   Quality enhancement: Kept {quality_mask.sum().item():,} high-quality splats out of {len(points):,}")
                #     else:
                #         print(f"   Quality enhancement skipped: Too few splats would remain ({quality_mask.sum().item()})")
                        
                # except Exception as e:
                #     print(f"   Quality enhancement failed: {e}")
                #     print("   Continuing with original splats...")
                
                # Save as PLY file
                import io
                ply_buffer = io.BytesIO()
                gaussian_output.save_ply(ply_buffer)
                ply_data = ply_buffer.getvalue()
                
                print(f"✓ Gaussian Splatting PLY extracted ({len(ply_data):,} bytes)")
                generation_asset.add_asset(AssetType.GAUSSIAN_SPLATTING_PLY, ply_data)
                
                # Step 4: Generate preview video (optional)
                if GENERATION_CONFIG.get('save_intermediate_outputs', False) and GENERATION_CONFIG.get('save_preview', False):
                    print("Step 4: Generating preview video...")
                    try:
                        video = render_utils.render_video(outputs['gaussian'][0], num_frames=120)['color']
                        video_geo = render_utils.render_video(outputs['mesh'][0], num_frames=120)['normal']
                        combined_video = [np.concatenate([video[i], video_geo[i]], axis=1) for i in range(len(video))]
                        generation_asset.add_asset(AssetType.PREVIEW_VIDEO, combined_video)
                        print("✓ Preview video generated")
                    except Exception as e:
                        print(f"⚠️ Preview video generation failed: {e}")
                
                # Step 5: Compress PLY if enabled
                compressed_data = None
                if GENERATION_CONFIG.get('auto_compress_ply', True):
                    print("Step 5: Compressing PLY with SPZ...")
                    try:
                        import pyspz
                        compressed_data = pyspz.compress(ply_data, workers=-1)
                        print(f"🗜️ SPZ Compression successful:")
                        print(f"   Original: {len(ply_data):,} bytes ({len(ply_data)/1024/1024:.1f} MB)")
                        print(f"   Compressed: {len(compressed_data):,} bytes ({len(compressed_data)/1024/1024:.1f} MB)") 
                        print(f"   Ratio: {len(compressed_data)/len(ply_data)*100:.1f}%")
                        print(f"   Space saved: {(len(ply_data)-len(compressed_data))/1024/1024:.1f} MB")
                        
                        generation_asset.add_asset(AssetType.COMPRESSED_PLY, compressed_data)
                    except Exception as e:
                        print(f"⚠️ SPZ compression failed: {e}")
                        compressed_data = None
                
                # Unload TRELLIS pipeline
                # self._unload_trellis_pipeline()
                
                generation_time = time.time() - start_time
                
                # Update metrics
                self.metrics.total_generations += 1
                self.metrics.successful_generations += 1
                self.metrics.last_generation_time = generation_time
                self.metrics.average_generation_time = (
                    (self.metrics.average_generation_time * (self.metrics.successful_generations - 1) + generation_time) 
                    / self.metrics.successful_generations
                )
                
                print(f"🎉 TRELLIS generation completed in {generation_time:.2f}s")
                
                # Save metadata
                if GENERATION_CONFIG.get('save_intermediate_outputs', False):
                    metadata_path = generation_asset.asset_directory / "metadata.json"
                    with open(metadata_path, 'w') as f:
                        json.dump({
                            **generation_asset.metadata,
                            "generation_time": generation_time,
                            "ply_size_bytes": len(ply_data),
                            "compressed_size_bytes": len(compressed_data) if compressed_data else None,
                            "compression_ratio": len(compressed_data)/len(ply_data) if compressed_data else None,
                        }, f, indent=2)
                    print(f"💾 Metadata saved: {metadata_path}")

                generation_job_status.update({
                    "status": "completed",
                    "end_time": time.time(),
                    "ply_path": f"generated_model_{seed}.ply"
                })
                            
                return ply_data, compressed_data
                
            except Exception as e:
                self.metrics.total_generations += 1
                self.metrics.failed_generations += 1
                print(f"❌ TRELLIS generation failed: {e}")
                traceback.print_exc()
                
                # Cleanup on failure
                self._unload_flux_models()
                self._unload_trellis_pipeline()
                self._unload_background_remover()
                
                generation_job_status.update({
                    "status": "failed",
                    "end_time": time.time(),
                    "error": str(e)
                })
                
                return None

    def generate_3d_model_image(self, prompt: str, seed: int = 42, num_inference_steps: Optional[int] = None, guidance_scale: Optional[float] = None, ss_sampling_steps: Optional[int] = None, slat_sampling_steps: Optional[int] = None, slat_guidance_strength: Optional[float] = None, ss_guidance_strength: Optional[float] = None) -> Optional[Tuple[bytes, Optional[bytes]]]:
        """Generate 3D model from text prompt using FLUX + TRELLIS pipeline"""
        
        job_id = f"gen_{int(time.time())}_{seed}"
        generation_job_status.update({
            "current_job_id": job_id,
            "status": "processing",
            "prompt": prompt,
            "seed": seed,
            "start_time": time.time(),
            "end_time": None,
            "ply_path": None,
            "error": None
        })

        with self.generation_lock:
            start_time = time.time()
            
            try:
                print(f"🎯 Starting TRELLIS generation for: '{prompt}' (seed: {seed})")
                
                # Initialize asset manager for this generation
                generation_asset = self.asset_manager.create_asset(prompt, seed)
                
                # Step 1: Generate image with selected model
                current_model = GENERATION_CONFIG.get('current_model', 'flux')
                print(f"Step 1: Generating image with {current_model.upper()}...")
                
                device = "cuda" if torch.cuda.is_available() else "cpu"
                
                # Enhanced prompt with LoRA trigger prefix if applicable
                enhanced_prompt = prompt
                current_lora = GENERATION_CONFIG.get('current_lora')
                
                if current_model == 'flux':
                    if self.flux_pipeline is None:
                        self._load_flux_models()
                    
                    if current_lora and current_lora in FLUX_LORAS:
                        lora_config = FLUX_LORAS[current_lora]
                        trigger_prefix = lora_config.get('trigger_prefix', '')
                        if trigger_prefix:
                            enhanced_prompt = f"{trigger_prefix} {prompt}"
                            print(f"🎨 Applied FLUX LoRA trigger prefix: '{trigger_prefix}'")
                    
                    generator = torch.Generator(device=device).manual_seed(seed)
                    with torch.no_grad():
                        effective_guidance_scale = guidance_scale if guidance_scale is not None else GENERATION_CONFIG['guidance_scale']
                        effective_steps = num_inference_steps if num_inference_steps is not None else NUM_INFERENCE_STEPS
                        effective_guidance_scale, effective_steps, extra_kwargs = self._resolve_flux_inference_params(
                            effective_guidance_scale,
                            effective_steps
                        )
                        if GENERATION_CONFIG.get('flux_use_schnell_socket', False):
                            image = self.flux_socket_client.generate_image(
                                prompt=enhanced_prompt,
                                seed=seed,
                                width=1024,
                                height=1024,
                            )
                        else:   
                            image = self.flux_pipeline(
                                prompt=enhanced_prompt,
                                guidance_scale=effective_guidance_scale,
                                num_inference_steps=effective_steps,
                                width=1024,
                                height=1024,
                                generator=generator,
                                **extra_kwargs,
                            ).images[0]
                
                elif current_model == 'sdxl':
                    if self.sdxl_pipeline is None:
                        self._load_sdxl_pipeline()
                    
                    if current_lora and current_lora in SDXL_LORAS:
                        lora_config = SDXL_LORAS[current_lora]
                        trigger_prefix = lora_config.get('trigger_prefix', '')
                        if trigger_prefix:
                            enhanced_prompt = f"{trigger_prefix} {prompt}"
                            print(f"🎨 Applied SDXL LoRA trigger prefix: '{trigger_prefix}'")
                    
                    generator = torch.Generator(device=device).manual_seed(seed)
                    with torch.no_grad():
                        effective_guidance_scale = guidance_scale if guidance_scale is not None else 7.5
                        effective_steps = num_inference_steps if num_inference_steps is not None else 25
                        image = self.sdxl_pipeline(
                            prompt=enhanced_prompt,
                            guidance_scale=effective_guidance_scale,
                            num_inference_steps=effective_steps,
                            width=1024,
                            height=1024,
                            generator=generator,
                        ).images[0]
                
                elif current_model == 'sd15':
                    if self.sd15_pipeline is None:
                        self._load_sd15_pipeline()
                    
                    if current_lora and current_lora in SD15_LORAS:
                        lora_config = SD15_LORAS[current_lora]
                        trigger_prefix = lora_config.get('trigger_prefix', '')
                        if trigger_prefix:
                            enhanced_prompt = f"{trigger_prefix} {prompt}"
                            print(f"🎨 Applied SD1.5 LoRA trigger prefix: '{trigger_prefix}'")
                    
                    generator = torch.Generator(device=device).manual_seed(seed)
                    with torch.no_grad():
                        effective_guidance_scale = guidance_scale if guidance_scale is not None else 7.5
                        effective_steps = num_inference_steps if num_inference_steps is not None else 25
                        image = self.sd15_pipeline(
                            prompt=enhanced_prompt,
                            guidance_scale=effective_guidance_scale,
                            num_inference_steps=effective_steps,
                            width=512,
                            height=512,
                            generator=generator,
                        ).images[0]
                
                else:
                    raise ValueError(f"Unknown model type: {current_model}")
                
                print(f"✓ {current_model.upper()} image generated successfully")
                generation_asset.add_asset(AssetType.FLUX_IMAGE, image)  # Keep same asset type for compatibility
                
                # Unload FLUX models
                # self._unload_flux_models()
                
                # # Step 1.3: Center object in image before background removal
                # if GENERATION_CONFIG.get('enable_object_centering', True):
                #     print("Step 1.3: Centering object in image...")
                #     try:
                #         centered_image = self.center_object_in_image(
                #             image, 
                #             white_threshold=GENERATION_CONFIG.get('centering_white_threshold', 240),
                #             padding=GENERATION_CONFIG.get('centering_padding', 40)
                #         )
                #         print("✓ Object centered successfully")
                #         image = centered_image  # Use the centered image for next steps
                #         generation_asset.add_asset(AssetType.FLUX_IMAGE, centered_image)  # Update asset with centered version
                #     except Exception as e:
                #         print(f"⚠️ Object centering failed: {e}")
                #         print("   Continuing with original image...")
                # else:
                #     print("Step 1.3: Object centering disabled, skipping...")
                
                # Step 1.5: Remove background from image
                print("Step 1.5: Removing background from image...")
                if self.background_remover is None:
                    self._load_background_remover()
                
                try:
                    image_no_bg = self.background_remover(image)
                    print("✓ Background removed successfully")
                    # Save the background-removed image as well
                    generation_asset.add_asset(AssetType.FLUX_IMAGE, image_no_bg)  # Replace original with cleaned version
                    image = image_no_bg  # Use the cleaned image for TRELLIS
                except Exception as e:
                    print(f"⚠️ Background removal failed: {e}")
                    print("   Continuing with original image...")
                
                # Unload background remover
                # self._unload_background_remover()
                
                # Step 2: Generate 3D model with TRELLIS
                print("Step 2: Generating 3D model with TRELLIS...")
                if self.trellis_pipeline is None:   
                    self._load_trellis_pipeline()
                    if self.trellis_pipeline is None:
                        raise RuntimeError("TRELLIS pipeline failed to load; cannot generate 3D model.")
                
                # Enhanced TRELLIS parameters for maximum quality
                # Resolve TRELLIS quality parameters with overrides
                effective_ss_steps = ss_sampling_steps if ss_sampling_steps is not None else GENERATION_CONFIG['ss_sampling_steps']
                effective_slat_steps = slat_sampling_steps if slat_sampling_steps is not None else GENERATION_CONFIG['slat_sampling_steps']
                effective_slat_guidance = slat_guidance_strength if slat_guidance_strength is not None else GENERATION_CONFIG['slat_guidance_strength']
                effective_ss_guidance = ss_guidance_strength if ss_guidance_strength is not None else GENERATION_CONFIG['ss_guidance_strength']

                # Use autocast to reduce activation memory and speed up compute on CUDA
                use_fp16 = GENERATION_CONFIG.get('trellis_use_fp16', True) and torch.cuda.is_available()
                if use_fp16:
                    try:
                        with torch.autocast(device_type="cuda", dtype=torch.float16):
                            outputs = self.trellis_pipeline.run(
                                image,
                                seed=seed,
                                formats=["gaussian"],
                                preprocess_image=False,
                                sparse_structure_sampler_params={
                                    "steps": effective_ss_steps,
                                    "cfg_strength": effective_ss_guidance,
                                    "cfg_interval": (0.3, 0.98),  # Enhanced guidance scheduling
                                    "rescale_t": 3.0,  # Temperature rescaling for better quality
                                },
                                slat_sampler_params={
                                    "steps": effective_slat_steps,
                                    "cfg_strength": effective_slat_guidance,
                                    "cfg_interval": (0.3, 0.98),  # Enhanced guidance scheduling
                                    "rescale_t": 3.0,  # Temperature rescaling for better quality
                                },
                            )
                    except RuntimeError as e:
                        # Some mesh decoding ops may not support fp16 (scatter/scatter_reduce dtype issues)
                        if "scatter()" in str(e) or "scatter_reduce" in str(e):
                            print("⚠️ FP16 mesh decode failed (scatter dtype mismatch). Retrying gaussian-only without autocast...")
                            with torch.autocast(device_type="cuda", enabled=False):
                                outputs = self.trellis_pipeline.run(
                                    image,
                                    seed=seed,
                                    formats=["gaussian"],  # Avoid mesh path in fp16
                                    preprocess_image=False,
                                    sparse_structure_sampler_params={
                                        "steps": effective_ss_steps,
                                        "cfg_strength": effective_ss_guidance,
                                        "cfg_interval": (0.3, 0.98),
                                        "rescale_t": 3.0,
                                    },
                                    slat_sampler_params={
                                        "steps": effective_slat_steps,
                                        "cfg_strength": effective_slat_guidance,
                                        "cfg_interval": (0.3, 0.98),
                                        "rescale_t": 3.0,
                                    },
                                )
                        else:
                            raise
                else:
                    outputs = self.trellis_pipeline.run(
                        image,
                        seed=seed,
                        formats=["gaussian", "mesh"],
                        preprocess_image=False,
                        sparse_structure_sampler_params={
                            "steps": effective_ss_steps,
                            "cfg_strength": effective_ss_guidance,
                            "cfg_interval": (0.3, 0.98),  # Enhanced guidance scheduling
                            "rescale_t": 3.0,  # Temperature rescaling for better quality
                        },
                        slat_sampler_params={
                            "steps": effective_slat_steps,
                            "cfg_strength": effective_slat_guidance,
                            "cfg_interval": (0.3, 0.98),  # Enhanced guidance scheduling
                            "rescale_t": 3.0,  # Temperature rescaling for better quality
                        },
                    )
                
                print("✓ 3D model generated successfully")
                
                # Step 3: Extract and enhance Gaussian Splatting PLY
                print("Step 3: Extracting and enhancing Gaussian Splatting PLY...")
                gaussian_output = outputs['gaussian'][0]
                
                # Quality enhancement: Filter low-quality splats
                # print("   Enhancing quality by filtering low-quality splats...")
                # try:
                #     # Get splat data
                #     points = gaussian_output.points
                #     opacities = gaussian_output.opacities
                #     scales = gaussian_output.scales
                    
                #     # Filter out low-opacity and very small splats
                #     opacity_threshold = 0.01
                #     scale_threshold = 0.001
                    
                #     # Create quality mask
                #     quality_mask = (opacities > opacity_threshold) & (torch.norm(scales, dim=1) > scale_threshold)
                    
                #     if quality_mask.sum() > 7000:  # Ensure minimum splat count
                #         # Apply filtering
                #         gaussian_output.points = points[quality_mask]
                #         gaussian_output.opacities = opacities[quality_mask]
                #         gaussian_output.scales = scales[quality_mask]
                #         gaussian_output.rotations = gaussian_output.rotations[quality_mask]
                #         gaussian_output.features_dc = gaussian_output.features_dc[quality_mask]
                #         gaussian_output.features_rest = gaussian_output.features_rest[quality_mask]
                #         gaussian_output.normals = gaussian_output.normals[quality_mask]
                        
                #         print(f"   Quality enhancement: Kept {quality_mask.sum().item():,} high-quality splats out of {len(points):,}")
                #     else:
                #         print(f"   Quality enhancement skipped: Too few splats would remain ({quality_mask.sum().item()})")
                        
                # except Exception as e:
                #     print(f"   Quality enhancement failed: {e}")
                #     print("   Continuing with original splats...")
                
                # Save as PLY file
                import io
                ply_buffer = io.BytesIO()
                gaussian_output.save_ply(ply_buffer)
                ply_data = ply_buffer.getvalue()
                
                print(f"✓ Gaussian Splatting PLY extracted ({len(ply_data):,} bytes)")
                generation_asset.add_asset(AssetType.GAUSSIAN_SPLATTING_PLY, ply_data)
                
                # Step 4: Generate preview video (optional)
                if GENERATION_CONFIG.get('save_intermediate_outputs', False) and GENERATION_CONFIG.get('save_preview', False):
                    print("Step 4: Generating preview video...")
                    try:
                        video = render_utils.render_video(outputs['gaussian'][0], num_frames=120)['color']
                        video_geo = render_utils.render_video(outputs['mesh'][0], num_frames=120)['normal']
                        combined_video = [np.concatenate([video[i], video_geo[i]], axis=1) for i in range(len(video))]
                        generation_asset.add_asset(AssetType.PREVIEW_VIDEO, combined_video)
                        print("✓ Preview video generated")
                    except Exception as e:
                        print(f"⚠️ Preview video generation failed: {e}")
                
                # Step 5: Compress PLY if enabled
                compressed_data = None
                if GENERATION_CONFIG.get('auto_compress_ply', True):
                    print("Step 5: Compressing PLY with SPZ...")
                    try:
                        import pyspz
                        compressed_data = pyspz.compress(ply_data, workers=-1)
                        print(f"🗜️ SPZ Compression successful:")
                        print(f"   Original: {len(ply_data):,} bytes ({len(ply_data)/1024/1024:.1f} MB)")
                        print(f"   Compressed: {len(compressed_data):,} bytes ({len(compressed_data)/1024/1024:.1f} MB)") 
                        print(f"   Ratio: {len(compressed_data)/len(ply_data)*100:.1f}%")
                        print(f"   Space saved: {(len(ply_data)-len(compressed_data))/1024/1024:.1f} MB")
                        
                        generation_asset.add_asset(AssetType.COMPRESSED_PLY, compressed_data)
                    except Exception as e:
                        print(f"⚠️ SPZ compression failed: {e}")
                        compressed_data = None
                
                # Unload TRELLIS pipeline
                # self._unload_trellis_pipeline()
                
                generation_time = time.time() - start_time
                
                # Update metrics
                self.metrics.total_generations += 1
                self.metrics.successful_generations += 1
                self.metrics.last_generation_time = generation_time
                self.metrics.average_generation_time = (
                    (self.metrics.average_generation_time * (self.metrics.successful_generations - 1) + generation_time) 
                    / self.metrics.successful_generations
                )
                
                print(f"🎉 TRELLIS generation completed in {generation_time:.2f}s")
                
                # Save metadata
                if GENERATION_CONFIG.get('save_intermediate_outputs', False):
                    metadata_path = generation_asset.asset_directory / "metadata.json"
                    with open(metadata_path, 'w') as f:
                        json.dump({
                            **generation_asset.metadata,
                            "generation_time": generation_time,
                            "ply_size_bytes": len(ply_data),
                            "compressed_size_bytes": len(compressed_data) if compressed_data else None,
                            "compression_ratio": len(compressed_data)/len(ply_data) if compressed_data else None,
                        }, f, indent=2)
                    print(f"💾 Metadata saved: {metadata_path}")

                generation_job_status.update({
                    "status": "completed",
                    "end_time": time.time(),
                    "ply_path": f"generated_model_{seed}.ply"
                })
                            
                return ply_data, compressed_data, image
                
            except Exception as e:
                self.metrics.total_generations += 1
                self.metrics.failed_generations += 1
                print(f"❌ TRELLIS generation failed: {e}")
                traceback.print_exc()
                
                # Cleanup on failure
                self._unload_flux_models()
                self._unload_trellis_pipeline()
                self._unload_background_remover()
                
                generation_job_status.update({
                    "status": "failed",
                    "end_time": time.time(),
                    "error": str(e)
                })
                
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
            "ready": self.ready # Removed gpu_memory call
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
                print(f"❌ Validation request failed: {response.status_code}")
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
    num_inference_steps: Optional[int] = Form(NUM_INFERENCE_STEPS),
    guidance_scale: Optional[float] = Form(GENERATION_CONFIG['guidance_scale']),
    ss_sampling_steps: Optional[int] = Form(GENERATION_CONFIG['ss_sampling_steps']),
    slat_sampling_steps: Optional[int] = Form(GENERATION_CONFIG['slat_sampling_steps']),
    slat_guidance_strength: Optional[float] = Form(GENERATION_CONFIG['slat_guidance_strength']),
    ss_guidance_strength: Optional[float] = Form(GENERATION_CONFIG['ss_guidance_strength'])
):
    """Generate 3D model from text prompt using FLUX + TRELLIS pipeline."""
    
    # Handle seed
    if seed is None:
        #seed = random.randint(0, MAX_SEED)
        seed = 42

    generator._unload_lora()
    
    # Generate model
    result = generator.generate_3d_model(
        prompt,
        seed,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        ss_sampling_steps=ss_sampling_steps,
        slat_sampling_steps=slat_sampling_steps,
        slat_guidance_strength=slat_guidance_strength,
        ss_guidance_strength=ss_guidance_strength
    )
    
    if result is None:
        raise HTTPException(status_code=500, detail="Generation failed")
    
    ply_data, compressed_data = result
    
    # Apply SPZ compression if requested
    if return_compressed:
        try:
            if compressed_data is None:
                import pyspz
                compressed_data = pyspz.compress(ply_data, workers=-1)
                print(f"🗜️ SPZ Compression for response:")
                print(f"   Original: {len(ply_data):,} bytes ({len(ply_data)/1024/1024:.1f} MB)")
                print(f"   Compressed: {len(compressed_data):,} bytes ({len(compressed_data)/1024/1024:.1f} MB)") 
                print(f"   Ratio: {len(compressed_data)/len(ply_data)*100:.1f}%")
            
            return Response(
                content=compressed_data,
                media_type="application/octet-stream",
                headers={
                    "Content-Disposition": f"attachment; filename=trellis_model_{seed}.ply.spz",
                    "X-Generation-Seed": str(seed),
                    "X-Generation-Prompt": prompt,
                    "X-Model-Format": "gaussian_splatting_ply",
                    "X-Pipeline": "flux_trellis",
                    "X-Compression": "spz",
                    "X-Compression-Ratio": f"{len(compressed_data)/len(ply_data)*100:.1f}%"
                }
            )
        except Exception as e:
            print(f"⚠️ SPZ compression failed: {e}")
            # Fall back to uncompressed
    
    # Return uncompressed PLY data
    return Response(
        content=ply_data,
        media_type="application/octet-stream",
        headers={
            "Content-Disposition": f"attachment; filename=trellis_model_{seed}.ply",
            "X-Generation-Seed": str(seed),
            "X-Generation-Prompt": prompt,
            "X-Model-Format": "gaussian_splatting_ply",
            "X-Pipeline": "flux_trellis",
            "X-Compression": "none"
        }
    )

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
        
        # Submit for validation in a separate thread to avoid blocking the event loop
        validation_results = await asyncio.to_thread(
            generator.submit_for_validation,
            prompt,
            ply_data
        )
        
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

@app.post("/config/quality/")
async def update_quality_config(
    guidance_scale: float = Form(4.0),
    ss_guidance_strength: float = Form(9.5),
    ss_sampling_steps: int = Form(30),
    slat_guidance_strength: float = Form(5.0),
    slat_sampling_steps: int = Form(30)
):
    """Update TRELLIS quality configuration for maximum validation scores"""
    try:
        GENERATION_CONFIG['guidance_scale'] = guidance_scale
        GENERATION_CONFIG['ss_guidance_strength'] = ss_guidance_strength
        GENERATION_CONFIG['ss_sampling_steps'] = ss_sampling_steps
        GENERATION_CONFIG['slat_guidance_strength'] = slat_guidance_strength
        GENERATION_CONFIG['slat_sampling_steps'] = slat_sampling_steps
        
        return {
            "status": "success",
            "message": "Quality configuration updated for maximum validation scores",
            "config": {
                "guidance_scale": guidance_scale,
                "ss_guidance_strength": ss_guidance_strength,
                "ss_sampling_steps": ss_sampling_steps,
                "slat_guidance_strength": slat_guidance_strength,
                "slat_sampling_steps": slat_sampling_steps
            }
        }
    except Exception as e:
        return JSONResponse(content={
            "status": "error",
            "message": str(e)
        }, status_code=500)

@app.post("/config/model/")
async def switch_model(
    model: str = Form(...)  # 'flux', 'sdxl', or 'sd15'
):
    """Switch between different models (FLUX, SDXL, SD1.5)"""
    try:
        if model not in ['flux', 'sdxl', 'sd15']:
            return JSONResponse(content={
                "status": "error",
                "message": "Invalid model. Must be 'flux', 'sdxl', or 'sd15'"
            }, status_code=400)
        
        # Unload current LoRA if any
        if GENERATION_CONFIG.get('current_lora'):
            generator._unload_lora()
        
        # Switch model
        GENERATION_CONFIG['current_model'] = model
        
        return {
            "status": "success",
            "message": f"Switched to {model.upper()} model",
            "current_model": model,
            "current_lora": None
        }
    except Exception as e:
        return JSONResponse(content={
            "status": "error",
            "message": str(e)
        }, status_code=500)

@app.get("/config/model/")
async def get_current_model():
    """Get current model configuration"""
    try:
        return {
            "status": "success",
            "current_model": GENERATION_CONFIG.get('current_model', 'flux'),
            "available_models": ['flux', 'sdxl', 'sd15']
        }
    except Exception as e:
        return JSONResponse(content={
            "status": "error",
            "message": str(e)
        }, status_code=500)

@app.get("/loras/")
async def get_available_loras():
    """Get list of available LoRAs"""
    try:
        loras = generator.get_available_loras()
        return {
            "status": "success",
            "loras": loras,
            "current_lora": GENERATION_CONFIG.get('current_lora'),
            "lora_scale": GENERATION_CONFIG.get('lora_scale', 1.0)
        }
    except Exception as e:
        return JSONResponse(content={
            "status": "error",
            "message": str(e)
        }, status_code=500)

@app.post("/loras/load/{lora_key}")
async def load_lora(lora_key: str):
    """Load a specific LoRA"""
    try:
        success = generator._load_lora(lora_key)
        if success:
            return {
                "status": "success",
                "message": f"LoRA '{lora_key}' loaded successfully",
                "current_lora": GENERATION_CONFIG.get('current_lora'),
                "lora_scale": GENERATION_CONFIG.get('lora_scale', 1.0)
            }
        else:
            return JSONResponse(content={
                "status": "error",
                "message": f"Failed to load LoRA '{lora_key}'"
            }, status_code=500)
    except Exception as e:
        return JSONResponse(content={
            "status": "error",
            "message": str(e)
        }, status_code=500)

@app.post("/loras/unload/")
async def unload_lora():
    """Unload current LoRA"""
    try:
        generator._unload_lora()
        return {
            "status": "success",
            "message": "LoRA unloaded successfully",
            "current_lora": None,
            "lora_scale": 1.0
        }
    except Exception as e:
        return JSONResponse(content={
            "status": "error",
            "message": str(e)
        }, status_code=500)

# Individual LoRA endpoints
@app.post("/generate/isometric_3d/")
async def generate_with_isometric_3d_lora(
    prompt: str = Form(...), 
    seed: Optional[int] = Form(None),
    return_compressed: Optional[bool] = Form(True),
    num_inference_steps: Optional[int] = Form(NUM_INFERENCE_STEPS),
    guidance_scale: Optional[float] = Form(GENERATION_CONFIG['guidance_scale']),
    ss_sampling_steps: Optional[int] = Form(GENERATION_CONFIG['ss_sampling_steps']),
    slat_sampling_steps: Optional[int] = Form(GENERATION_CONFIG['slat_sampling_steps']),
    slat_guidance_strength: Optional[float] = Form(GENERATION_CONFIG['slat_guidance_strength']),
    ss_guidance_strength: Optional[float] = Form(GENERATION_CONFIG['ss_guidance_strength'])
):
    """Generate 3D model using Isometric 3D LoRA"""
    try:
        # Load the LoRA
        success = generator._load_lora('isometric_3d')
        if not success:
            raise HTTPException(status_code=500, detail="Failed to load Isometric 3D LoRA")
        
        # Apply LoRA trigger prefix
        lora_config = FLUX_LORAS['isometric_3d']
        trigger_prefix = lora_config.get('trigger_prefix', '')
        enhanced_prompt = prompt
        if trigger_prefix:
            enhanced_prompt = f"{trigger_prefix} {prompt}"
            print(f"🎨 Applied FLUX LoRA trigger prefix: '{trigger_prefix}'")
        
        # Generate with the LoRA
        result = generator.generate_3d_model(
            enhanced_prompt,
            seed or 42,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            ss_sampling_steps=ss_sampling_steps,
            slat_sampling_steps=slat_sampling_steps,
            slat_guidance_strength=slat_guidance_strength,
            ss_guidance_strength=ss_guidance_strength
        )
        if result is None:
            raise HTTPException(status_code=500, detail="Generation failed")
        
        ply_data, compressed_data = result
        
        # Return compressed data if requested
        if return_compressed and compressed_data:
            return Response(
                content=compressed_data,
                media_type="application/octet-stream",
                headers={
                    "Content-Disposition": f"attachment; filename=isometric_3d_{seed or 42}.ply.spz",
                    "X-Generation-Seed": str(seed or 42),
                    "X-Generation-Prompt": prompt,
                    "X-Model-Format": "gaussian_splatting_ply",
                    "X-Pipeline": "flux_trellis",
                    "X-LoRA": "isometric_3d",
                    "X-Compression": "spz"
                }
            )
        
        # Return uncompressed PLY data
        return Response(
            content=ply_data,
            media_type="application/octet-stream",
            headers={
                "Content-Disposition": f"attachment; filename=isometric_3d_{seed or 42}.ply",
                "X-Generation-Seed": str(seed or 42),
                "X-Generation-Prompt": prompt,
                "X-Model-Format": "gaussian_splatting_ply",
                "X-Pipeline": "flux_trellis",
                "X-LoRA": "isometric_3d",
                "X-Compression": "none"
            }
        )
    except Exception as e:
        return JSONResponse(content={
            "status": "error",
            "message": str(e)
        }, status_code=500)

@app.post("/generate/live_3d/")
async def generate_with_live_3d_lora(
    prompt: str = Form(...), 
    seed: Optional[int] = Form(None),
    return_compressed: Optional[bool] = Form(True),
    num_inference_steps: Optional[int] = Form(NUM_INFERENCE_STEPS),
    guidance_scale: Optional[float] = Form(GENERATION_CONFIG['guidance_scale']),
    ss_sampling_steps: Optional[int] = Form(GENERATION_CONFIG['ss_sampling_steps']),
    slat_sampling_steps: Optional[int] = Form(GENERATION_CONFIG['slat_sampling_steps']),
    slat_guidance_strength: Optional[float] = Form(GENERATION_CONFIG['slat_guidance_strength']),
    ss_guidance_strength: Optional[float] = Form(GENERATION_CONFIG['ss_guidance_strength'])
):
    """Generate 3D model using Live 3D LoRA"""
    try:
        success = generator._load_lora('live_3d')
        if not success:
            raise HTTPException(status_code=500, detail="Failed to load Live 3D LoRA")
        
        # Apply LoRA trigger prefix
        lora_config = FLUX_LORAS['live_3d']
        trigger_prefix = lora_config.get('trigger_prefix', '')
        enhanced_prompt = prompt
        if trigger_prefix:
            enhanced_prompt = f"{trigger_prefix} {prompt}"
            print(f"🎨 Applied FLUX LoRA trigger prefix: '{trigger_prefix}'")
        
        result = generator.generate_3d_model(
            enhanced_prompt,
            seed or 42,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            ss_sampling_steps=ss_sampling_steps,
            slat_sampling_steps=slat_sampling_steps,
            slat_guidance_strength=slat_guidance_strength,
            ss_guidance_strength=ss_guidance_strength
        )
        if result is None:
            raise HTTPException(status_code=500, detail="Generation failed")
        
        ply_data, compressed_data = result
        
        if return_compressed and compressed_data:
            return Response(
                content=compressed_data,
                media_type="application/octet-stream",
                headers={
                    "Content-Disposition": f"attachment; filename=live_3d_{seed or 42}.ply.spz",
                    "X-Generation-Seed": str(seed or 42),
                    "X-Generation-Prompt": prompt,
                    "X-Model-Format": "gaussian_splatting_ply",
                    "X-Pipeline": "flux_trellis",
                    "X-LoRA": "live_3d",
                    "X-Compression": "spz"
                }
            )
        
        return Response(
            content=ply_data,
            media_type="application/octet-stream",
            headers={
                "Content-Disposition": f"attachment; filename=live_3d_{seed or 42}.ply",
                "X-Generation-Seed": str(seed or 42),
                "X-Generation-Prompt": prompt,
                "X-Model-Format": "gaussian_splatting_ply",
                "X-Pipeline": "flux_trellis",
                "X-LoRA": "live_3d",
                "X-Compression": "none"
            }
        )
    except Exception as e:
        return JSONResponse(content={
            "status": "error",
            "message": str(e)
        }, status_code=500)

@app.post("/generate/game_assets/")
async def generate_with_game_assets_lora(
    prompt: str = Form(...), 
    seed: Optional[int] = Form(None),
    return_compressed: Optional[bool] = Form(True),
    num_inference_steps: Optional[int] = Form(NUM_INFERENCE_STEPS),
    guidance_scale: Optional[float] = Form(GENERATION_CONFIG['guidance_scale']),
    ss_sampling_steps: Optional[int] = Form(GENERATION_CONFIG['ss_sampling_steps']),
    slat_sampling_steps: Optional[int] = Form(GENERATION_CONFIG['slat_sampling_steps']),
    slat_guidance_strength: Optional[float] = Form(GENERATION_CONFIG['slat_guidance_strength']),
    ss_guidance_strength: Optional[float] = Form(GENERATION_CONFIG['ss_guidance_strength'])
):
    """Generate 3D model using Game Assets LoRA"""
    try:
        success = generator._load_lora('game_assets')
        if not success:
            raise HTTPException(status_code=500, detail="Failed to load Game Assets LoRA")
        
        # Apply LoRA trigger prefix
        lora_config = FLUX_LORAS['game_assets']
        trigger_prefix = lora_config.get('trigger_prefix', '')
        enhanced_prompt = prompt
        if trigger_prefix:
            enhanced_prompt = f"{trigger_prefix} {prompt}"
            print(f"🎨 Applied FLUX LoRA trigger prefix: '{trigger_prefix}'")
        
        result = generator.generate_3d_model(
            enhanced_prompt,
            seed or 42,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            ss_sampling_steps=ss_sampling_steps,
            slat_sampling_steps=slat_sampling_steps,
            slat_guidance_strength=slat_guidance_strength,
            ss_guidance_strength=ss_guidance_strength
        )
        if result is None:
            raise HTTPException(status_code=500, detail="Generation failed")
        
        ply_data, compressed_data = result
        
        if return_compressed and compressed_data:
            return Response(
                content=compressed_data,
                media_type="application/octet-stream",
                headers={
                    "Content-Disposition": f"attachment; filename=game_assets_{seed or 42}.ply.spz",
                    "X-Generation-Seed": str(seed or 42),
                    "X-Generation-Prompt": prompt,
                    "X-Model-Format": "gaussian_splatting_ply",
                    "X-Pipeline": "flux_trellis",
                    "X-LoRA": "game_assets",
                    "X-Compression": "spz"
                }
            )
        
        return Response(
            content=ply_data,
            media_type="application/octet-stream",
            headers={
                "Content-Disposition": f"attachment; filename=game_assets_{seed or 42}.ply",
                "X-Generation-Seed": str(seed or 42),
                "X-Generation-Prompt": prompt,
                "X-Model-Format": "gaussian_splatting_ply",
                "X-Pipeline": "flux_trellis",
                "X-LoRA": "game_assets",
                "X-Compression": "none"
            }
        )
    except Exception as e:
        return JSONResponse(content={
            "status": "error",
            "message": str(e)
        }, status_code=500)

@app.post("/generate/patched_realism/")
async def generate_with_patched_realism_lora(
    prompt: str = Form(...), 
    seed: Optional[int] = Form(None),
    return_compressed: Optional[bool] = Form(True),
    num_inference_steps: Optional[int] = Form(NUM_INFERENCE_STEPS),
    guidance_scale: Optional[float] = Form(GENERATION_CONFIG['guidance_scale']),
    ss_sampling_steps: Optional[int] = Form(GENERATION_CONFIG['ss_sampling_steps']),
    slat_sampling_steps: Optional[int] = Form(GENERATION_CONFIG['slat_sampling_steps']),
    slat_guidance_strength: Optional[float] = Form(GENERATION_CONFIG['slat_guidance_strength']),
    ss_guidance_strength: Optional[float] = Form(GENERATION_CONFIG['ss_guidance_strength'])
):
    """Generate 3D model using Patched Realism LoRA"""
    try:
        success = generator._load_lora('patched_realism')
        if not success:
            raise HTTPException(status_code=500, detail="Failed to load Patched Realism LoRA")
        
        # Apply LoRA trigger prefix
        lora_config = FLUX_LORAS['patched_realism']
        trigger_prefix = lora_config.get('trigger_prefix', '')
        enhanced_prompt = prompt
        if trigger_prefix:
            enhanced_prompt = f"{trigger_prefix} {prompt}"
            print(f"🎨 Applied FLUX LoRA trigger prefix: '{trigger_prefix}'")
        
        result = generator.generate_3d_model(
            enhanced_prompt,
            seed or 42,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            ss_sampling_steps=ss_sampling_steps,
            slat_sampling_steps=slat_sampling_steps,
            slat_guidance_strength=slat_guidance_strength,
            ss_guidance_strength=ss_guidance_strength
        )
        if result is None:
            raise HTTPException(status_code=500, detail="Generation failed")
        
        ply_data, compressed_data = result
        
        if return_compressed and compressed_data:
            return Response(
                content=compressed_data,
                media_type="application/octet-stream",
                headers={
                    "Content-Disposition": f"attachment; filename=patched_realism_{seed or 42}.ply.spz",
                    "X-Generation-Seed": str(seed or 42),
                    "X-Generation-Prompt": prompt,
                    "X-Model-Format": "gaussian_splatting_ply",
                    "X-Pipeline": "flux_trellis",
                    "X-LoRA": "patched_realism",
                    "X-Compression": "spz"
                }
            )
        
        return Response(
            content=ply_data,
            media_type="application/octet-stream",
            headers={
                "Content-Disposition": f"attachment; filename=patched_realism_{seed or 42}.ply",
                "X-Generation-Seed": str(seed or 42),
                "X-Generation-Prompt": prompt,
                "X-Model-Format": "gaussian_splatting_ply",
                "X-Pipeline": "flux_trellis",
                "X-LoRA": "patched_realism",
                "X-Compression": "none"
            }
        )
    except Exception as e:
        return JSONResponse(content={
            "status": "error",
            "message": str(e)
        }, status_code=500)

@app.post("/generate/tf2_style/")
async def generate_with_tf2_style_lora(
    prompt: str = Form(...), 
    seed: Optional[int] = Form(None),
    return_compressed: Optional[bool] = Form(True),
    num_inference_steps: Optional[int] = Form(NUM_INFERENCE_STEPS),
    guidance_scale: Optional[float] = Form(GENERATION_CONFIG['guidance_scale']),
    ss_sampling_steps: Optional[int] = Form(GENERATION_CONFIG['ss_sampling_steps']),
    slat_sampling_steps: Optional[int] = Form(GENERATION_CONFIG['slat_sampling_steps']),
    slat_guidance_strength: Optional[float] = Form(GENERATION_CONFIG['slat_guidance_strength']),
    ss_guidance_strength: Optional[float] = Form(GENERATION_CONFIG['ss_guidance_strength'])
):
    """Generate 3D model using TF2 Style LoRA"""
    try:
        success = generator._load_lora('tf2_style')
        if not success:
            raise HTTPException(status_code=500, detail="Failed to load TF2 Style LoRA")
        
        # Apply LoRA trigger prefix
        lora_config = FLUX_LORAS['tf2_style']
        trigger_prefix = lora_config.get('trigger_prefix', '')
        enhanced_prompt = prompt
        if trigger_prefix:
            enhanced_prompt = f"{trigger_prefix} {prompt}"
            print(f"🎨 Applied FLUX LoRA trigger prefix: '{trigger_prefix}'")
        
        result = generator.generate_3d_model(
            enhanced_prompt,
            seed or 42,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            ss_sampling_steps=ss_sampling_steps,
            slat_sampling_steps=slat_sampling_steps,
            slat_guidance_strength=slat_guidance_strength,
            ss_guidance_strength=ss_guidance_strength
        )
        if result is None:
            raise HTTPException(status_code=500, detail="Generation failed")
        
        ply_data, compressed_data = result
        
        if return_compressed and compressed_data:
            return Response(
                content=compressed_data,
                media_type="application/octet-stream",
                headers={
                    "Content-Disposition": f"attachment; filename=tf2_style_{seed or 42}.ply.spz",
                    "X-Generation-Seed": str(seed or 42),
                    "X-Generation-Prompt": prompt,
                    "X-Model-Format": "gaussian_splatting_ply",
                    "X-Pipeline": "flux_trellis",
                    "X-LoRA": "tf2_style",
                    "X-Compression": "spz"
                }
            )
        
        return Response(
            content=ply_data,
            media_type="application/octet-stream",
            headers={
                "Content-Disposition": f"attachment; filename=tf2_style_{seed or 42}.ply",
                "X-Generation-Seed": str(seed or 42),
                "X-Generation-Prompt": prompt,
                "X-Model-Format": "gaussian_splatting_ply",
                "X-Pipeline": "flux_trellis",
                "X-LoRA": "tf2_style",
                "X-Compression": "none"
            }
        )
    except Exception as e:
        return JSONResponse(content={
            "status": "error",
            "message": str(e)
        }, status_code=500)

@app.post("/generate/baolei/")
async def generate_with_baolei_lora(
    prompt: str = Form(...), 
    seed: Optional[int] = Form(None),
    return_compressed: Optional[bool] = Form(True),
    num_inference_steps: Optional[int] = Form(NUM_INFERENCE_STEPS),
    guidance_scale: Optional[float] = Form(GENERATION_CONFIG['guidance_scale']),
    ss_sampling_steps: Optional[int] = Form(GENERATION_CONFIG['ss_sampling_steps']),
    slat_sampling_steps: Optional[int] = Form(GENERATION_CONFIG['slat_sampling_steps']),
    slat_guidance_strength: Optional[float] = Form(GENERATION_CONFIG['slat_guidance_strength']),
    ss_guidance_strength: Optional[float] = Form(GENERATION_CONFIG['ss_guidance_strength'])
):
    """Generate 3D model using Baolei Style LoRA"""
    try:
        success = generator._load_lora('baolei')
        if not success:
            raise HTTPException(status_code=500, detail="Failed to load Baolei Style LoRA")
        
        # Apply LoRA trigger prefix
        lora_config = FLUX_LORAS['baolei']
        trigger_prefix = lora_config.get('trigger_prefix', '')
        enhanced_prompt = prompt
        if trigger_prefix:
            enhanced_prompt = f"{trigger_prefix} {prompt}"
            print(f"🎨 Applied FLUX LoRA trigger prefix: '{trigger_prefix}'")
        
        result = generator.generate_3d_model(
            enhanced_prompt,
            seed or 42,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            ss_sampling_steps=ss_sampling_steps,
            slat_sampling_steps=slat_sampling_steps,
            slat_guidance_strength=slat_guidance_strength,
            ss_guidance_strength=ss_guidance_strength
        )
        if result is None:
            raise HTTPException(status_code=500, detail="Generation failed")
        
        ply_data, compressed_data = result
        
        if return_compressed and compressed_data:
            return Response(
                content=compressed_data,
                media_type="application/octet-stream",
                headers={
                    "Content-Disposition": f"attachment; filename=baolei_{seed or 42}.ply.spz",
                    "X-Generation-Seed": str(seed or 42),
                    "X-Generation-Prompt": prompt,
                    "X-Model-Format": "gaussian_splatting_ply",
                    "X-Pipeline": "flux_trellis",
                    "X-LoRA": "baolei",
                    "X-Compression": "spz"
                }
            )
        
        return Response(
            content=ply_data,
            media_type="application/octet-stream",
            headers={
                "Content-Disposition": f"attachment; filename=baolei_{seed or 42}.ply",
                "X-Generation-Seed": str(seed or 42),
                "X-Generation-Prompt": prompt,
                "X-Model-Format": "gaussian_splatting_ply",
                "X-Pipeline": "flux_trellis",
                "X-LoRA": "baolei",
                "X-Compression": "none"
            }
        )
    except Exception as e:
        return JSONResponse(content={
            "status": "error",
            "message": str(e)
        }, status_code=500)

@app.post("/generate/cartoon_3d/")
async def generate_with_cartoon_3d_lora(
    prompt: str = Form(...), 
    seed: Optional[int] = Form(None),
    return_compressed: Optional[bool] = Form(True),
    num_inference_steps: Optional[int] = Form(NUM_INFERENCE_STEPS),
    guidance_scale: Optional[float] = Form(GENERATION_CONFIG['guidance_scale']),
    ss_sampling_steps: Optional[int] = Form(GENERATION_CONFIG['ss_sampling_steps']),
    slat_sampling_steps: Optional[int] = Form(GENERATION_CONFIG['slat_sampling_steps']),
    slat_guidance_strength: Optional[float] = Form(GENERATION_CONFIG['slat_guidance_strength']),
    ss_guidance_strength: Optional[float] = Form(GENERATION_CONFIG['ss_guidance_strength'])
):
    """Generate 3D model using Cartoon 3D Render LoRA"""
    try:
        success = generator._load_lora('cartoon_3d')
        if not success:
            raise HTTPException(status_code=500, detail="Failed to load Cartoon 3D Render LoRA")
        
        # Apply LoRA trigger prefix
        lora_config = FLUX_LORAS['cartoon_3d']
        trigger_prefix = lora_config.get('trigger_prefix', '')
        enhanced_prompt = prompt
        if trigger_prefix:
            enhanced_prompt = f"{trigger_prefix} {prompt}"
            print(f"🎨 Applied FLUX LoRA trigger prefix: '{trigger_prefix}'")
        
        result = generator.generate_3d_model(
            enhanced_prompt,
            seed or 42,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            ss_sampling_steps=ss_sampling_steps,
            slat_sampling_steps=slat_sampling_steps,
            slat_guidance_strength=slat_guidance_strength,
            ss_guidance_strength=ss_guidance_strength
        )
        if result is None:
            raise HTTPException(status_code=500, detail="Generation failed")
        
        ply_data, compressed_data = result
        
        if return_compressed and compressed_data:
            return Response(
                content=compressed_data,
                media_type="application/octet-stream",
                headers={
                    "Content-Disposition": f"attachment; filename=cartoon_3d_{seed or 42}.ply.spz",
                    "X-Generation-Seed": str(seed or 42),
                    "X-Generation-Prompt": prompt,
                    "X-Model-Format": "gaussian_splatting_ply",
                    "X-Pipeline": "flux_trellis",
                    "X-LoRA": "cartoon_3d",
                    "X-Compression": "spz"
                }
            )
        
        return Response(
            content=ply_data,
            media_type="application/octet-stream",
            headers={
                "Content-Disposition": f"attachment; filename=cartoon_3d_{seed or 42}.ply",
                "X-Generation-Seed": str(seed or 42),
                "X-Generation-Prompt": prompt,
                "X-Model-Format": "gaussian_splatting_ply",
                "X-Pipeline": "flux_trellis",
                "X-LoRA": "cartoon_3d",
                "X-Compression": "none"
            }
        )
    except Exception as e:
        return JSONResponse(content={
            "status": "error",
            "message": str(e)
        }, status_code=500)

@app.post("/generate/cinema/")
async def generate_with_cinema_lora(
    prompt: str = Form(...), 
    seed: Optional[int] = Form(None),
    return_compressed: Optional[bool] = Form(True),
    num_inference_steps: Optional[int] = Form(NUM_INFERENCE_STEPS),
    guidance_scale: Optional[float] = Form(GENERATION_CONFIG['guidance_scale']),
    ss_sampling_steps: Optional[int] = Form(GENERATION_CONFIG['ss_sampling_steps']),
    slat_sampling_steps: Optional[int] = Form(GENERATION_CONFIG['slat_sampling_steps']),
    slat_guidance_strength: Optional[float] = Form(GENERATION_CONFIG['slat_guidance_strength']),
    ss_guidance_strength: Optional[float] = Form(GENERATION_CONFIG['ss_guidance_strength'])
):
    """Generate 3D model using Cinema Style LoRA"""
    try:
        success = generator._load_lora('cinema')
        if not success:
            raise HTTPException(status_code=500, detail="Failed to load Cinema Style LoRA")
        
        # Apply LoRA trigger prefix
        lora_config = FLUX_LORAS['cinema']
        trigger_prefix = lora_config.get('trigger_prefix', '')
        enhanced_prompt = prompt
        if trigger_prefix:
            enhanced_prompt = f"{trigger_prefix} {prompt}"
            print(f"🎨 Applied FLUX LoRA trigger prefix: '{trigger_prefix}'")
        
        result = generator.generate_3d_model(
            enhanced_prompt,
            seed or 42,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            ss_sampling_steps=ss_sampling_steps,
            slat_sampling_steps=slat_sampling_steps,
            slat_guidance_strength=slat_guidance_strength,
            ss_guidance_strength=ss_guidance_strength
        )
        if result is None:
            raise HTTPException(status_code=500, detail="Generation failed")
        
        ply_data, compressed_data = result
        
        if return_compressed and compressed_data:
            return Response(
                content=compressed_data,
                media_type="application/octet-stream",
                headers={
                    "Content-Disposition": f"attachment; filename=cinema_{seed or 42}.ply.spz",
                    "X-Generation-Seed": str(seed or 42),
                    "X-Generation-Prompt": prompt,
                    "X-Model-Format": "gaussian_splatting_ply",
                    "X-Pipeline": "flux_trellis",
                    "X-LoRA": "cinema",
                    "X-Compression": "spz"
                }
            )
        
        return Response(
            content=ply_data,
            media_type="application/octet-stream",
            headers={
                "Content-Disposition": f"attachment; filename=cinema_{seed or 42}.ply",
                "X-Generation-Seed": str(seed or 42),
                "X-Generation-Prompt": prompt,
                "X-Model-Format": "gaussian_splatting_ply",
                "X-Pipeline": "flux_trellis",
                "X-LoRA": "cinema",
                "X-Compression": "none"
            }
        )
    except Exception as e:
        return JSONResponse(content={
            "status": "error",
            "message": str(e)
        }, status_code=500)

@app.post("/generate/sd15_game_icon/")
async def generate_with_sd15_game_icon_lora(
    prompt: str = Form(...), 
    seed: Optional[int] = Form(None),
    return_compressed: Optional[bool] = Form(True),
    num_inference_steps: Optional[int] = Form(NUM_INFERENCE_STEPS),
    guidance_scale: Optional[float] = Form(7.5),
    ss_sampling_steps: Optional[int] = Form(GENERATION_CONFIG['ss_sampling_steps']),
    slat_sampling_steps: Optional[int] = Form(GENERATION_CONFIG['slat_sampling_steps']),
    slat_guidance_strength: Optional[float] = Form(GENERATION_CONFIG['slat_guidance_strength']),
    ss_guidance_strength: Optional[float] = Form(GENERATION_CONFIG['ss_guidance_strength'])
):
    """Generate 3D model using SD1.5 Game Icon LoRA"""
    try:
        # Switch to SD1.5 model first
        GENERATION_CONFIG['current_model'] = 'sd15'
        
        success = generator._load_lora('game_icon')
        if not success:
            raise HTTPException(status_code=500, detail="Failed to load SD1.5 Game Icon LoRA")
        
        # Apply LoRA trigger prefix
        lora_config = SD15_LORAS['game_icon']
        trigger_prefix = lora_config.get('trigger_prefix', '')
        enhanced_prompt = prompt
        if trigger_prefix:
            enhanced_prompt = f"{trigger_prefix} {prompt}"
            print(f"🎨 Applied SD15 LoRA trigger prefix: '{trigger_prefix}'")
        
        result = generator.generate_3d_model(
            enhanced_prompt,
            seed or 42,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            ss_sampling_steps=ss_sampling_steps,
            slat_sampling_steps=slat_sampling_steps,
            slat_guidance_strength=slat_guidance_strength,
            ss_guidance_strength=ss_guidance_strength
        )
        if result is None:
            raise HTTPException(status_code=500, detail="Generation failed")
        
        ply_data, compressed_data = result
        
        if return_compressed and compressed_data:
            return Response(
                content=compressed_data,
                media_type="application/octet-stream",
                headers={
                    "Content-Disposition": f"attachment; filename=sd15_game_icon_{seed or 42}.ply.spz",
                    "X-Generation-Seed": str(seed or 42),
                    "X-Generation-Prompt": prompt,
                    "X-Model-Format": "gaussian_splatting_ply",
                    "X-Pipeline": "sd15_trellis",
                    "X-LoRA": "game_icon",
                    "X-Compression": "spz"
                }
            )
        
        return Response(
            content=ply_data,
            media_type="application/octet-stream",
            headers={
                "Content-Disposition": f"attachment; filename=sd15_game_icon_{seed or 42}.ply",
                "X-Generation-Seed": str(seed or 42),
                "X-Generation-Prompt": prompt,
                "X-Model-Format": "gaussian_splatting_ply",
                "X-Pipeline": "sd15_trellis",
                "X-LoRA": "game_icon",
                "X-Compression": "none"
            }
        )
    except Exception as e:
        return JSONResponse(content={
            "status": "error",
            "message": str(e)
        }, status_code=500)

@app.post("/generate/necklace/")
async def generate_with_necklace_lora(
    prompt: str = Form(...), 
    seed: Optional[int] = Form(None),
    return_compressed: Optional[bool] = Form(True),
    num_inference_steps: Optional[int] = Form(NUM_INFERENCE_STEPS),
    guidance_scale: Optional[float] = Form(GENERATION_CONFIG['guidance_scale']),
    ss_sampling_steps: Optional[int] = Form(GENERATION_CONFIG['ss_sampling_steps']),
    slat_sampling_steps: Optional[int] = Form(GENERATION_CONFIG['slat_sampling_steps']),
    slat_guidance_strength: Optional[float] = Form(GENERATION_CONFIG['slat_guidance_strength']),
    ss_guidance_strength: Optional[float] = Form(GENERATION_CONFIG['ss_guidance_strength'])
):
    """Generate 3D model using FLUX with Necklace LoRA"""
    try:
        # Switch to FLUX model first
        GENERATION_CONFIG['current_model'] = 'flux'
        
        success = generator._load_lora('necklace')
        if not success:
            raise HTTPException(status_code=500, detail="Failed to load Necklace LoRA")
        
        # Apply LoRA trigger prefix
        lora_config = FLUX_LORAS['necklace']
        trigger_prefix = lora_config.get('trigger_prefix', '')
        enhanced_prompt = prompt
        if trigger_prefix:
            enhanced_prompt = f"{trigger_prefix} {prompt}"
            print(f"🎨 Applied FLUX LoRA trigger prefix: '{trigger_prefix}'")
        
        result = generator.generate_3d_model(
            enhanced_prompt,
            seed or 42,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            ss_sampling_steps=ss_sampling_steps,
            slat_sampling_steps=slat_sampling_steps,
            slat_guidance_strength=slat_guidance_strength,
            ss_guidance_strength=ss_guidance_strength
        )
        if result is None:
            raise HTTPException(status_code=500, detail="Generation failed")
        
        ply_data, compressed_data = result
        
        if return_compressed and compressed_data:
            return Response(
                content=compressed_data,
                media_type="application/octet-stream",
                headers={
                    "Content-Disposition": f"attachment; filename=necklace_{seed or 42}.ply.spz",
                    "X-Generation-Seed": str(seed or 42),
                    "X-Generation-Prompt": prompt,
                    "X-Model-Format": "gaussian_splatting_ply",
                    "X-Pipeline": "flux_trellis",
                    "X-LoRA": "necklace",
                    "X-Compression": "spz"
                }
            )
        
        return Response(
            content=ply_data,
            media_type="application/octet-stream",
            headers={
                "Content-Disposition": f"attachment; filename=necklace_{seed or 42}.ply",
                "X-Generation-Seed": str(seed or 42),
                "X-Generation-Prompt": prompt,
                "X-Model-Format": "gaussian_splatting_ply",
                "X-Pipeline": "flux_trellis",
                "X-LoRA": "necklace",
                "X-Compression": "none"
            }
        )
    except Exception as e:
        return JSONResponse(content={
            "status": "error",
            "message": str(e)
        }, status_code=500)

# Image generation endpoints for all LoRAs
@app.post("/generate_image/isometric_3d/")
async def generate_image_with_isometric_3d_lora(
    prompt: str = Form(...), 
    seed: Optional[int] = Form(None),
    num_inference_steps: Optional[int] = Form(25),
    guidance_scale: Optional[float] = Form(7.5)
):
    """Generate image only using FLUX with Isometric 3D LoRA (without 3D generation)."""
    
    # Handle seed
    if seed is None:
        seed = random.randint(0, 2**31 - 1)
    
    try:
        # Switch to FLUX model
        GENERATION_CONFIG['current_model'] = 'flux'
        
        # Load FLUX if not loaded
        if generator.flux_pipeline is None:
            generator._load_flux_models()
        
        # Load the LoRA
        success = generator._load_lora('isometric_3d')
        if not success:
            raise HTTPException(status_code=500, detail="Failed to load Isometric 3D LoRA")
        
        # Apply LoRA trigger prefix
        lora_config = FLUX_LORAS['isometric_3d']
        trigger_prefix = lora_config.get('trigger_prefix', '')
        enhanced_prompt = prompt
        if trigger_prefix:
            enhanced_prompt = f"{trigger_prefix} {prompt}"
            print(f"🎨 Applied FLUX LoRA trigger prefix: '{trigger_prefix}'")
        
        # Generate image with FLUX + LoRA
        print(f"🎨 Generating image with Isometric 3D LoRA for: '{enhanced_prompt}' (seed: {seed})")
        seed_generator = torch.Generator(device=GENERATION_CONFIG['device']).manual_seed(seed)
        with torch.no_grad():
            eff_guidance, eff_steps, extra_kwargs = generator._resolve_flux_inference_params(
                guidance_scale, num_inference_steps
            )
            flux_output = generator.flux_pipeline(
                prompt=enhanced_prompt,
                generator=seed_generator,
                num_inference_steps=eff_steps,
                guidance_scale=eff_guidance,
                **extra_kwargs
            )
            image = flux_output.images[0]  # Extract the first image from the output
        
        # Convert PIL Image to bytes
        img_buffer = io.BytesIO()
        image.save(img_buffer, format='PNG')
        image_data = img_buffer.getvalue()
        
        # Encode as base64 for JSON response
        image_base64 = base64.b64encode(image_data).decode('utf-8')
        
        return JSONResponse(content={
            "status": "success",
            "prompt": prompt,
            "seed": seed,
            "image": image_base64,
            "image_size_bytes": len(image_data),
            "pipeline": "flux_only",
            "lora": "isometric_3d"
        })
        
    except Exception as e:
        print(f"❌ Image generation failed: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Image generation failed: {str(e)}")

@app.post("/generate_image/live_3d/")
async def generate_image_with_live_3d_lora(
    prompt: str = Form(...), 
    seed: Optional[int] = Form(None),
    num_inference_steps: Optional[int] = Form(25),
    guidance_scale: Optional[float] = Form(7.5)
):
    """Generate image only using FLUX with Live 3D LoRA (without 3D generation)."""
    
    # Handle seed
    if seed is None:
        seed = random.randint(0, 2**31 - 1)
    
    try:
        # Switch to FLUX model
        GENERATION_CONFIG['current_model'] = 'flux'
        
        # Load FLUX if not loaded
        if generator.flux_pipeline is None:
            generator._load_flux_models()
        
        # Load the LoRA
        success = generator._load_lora('live_3d')
        if not success:
            raise HTTPException(status_code=500, detail="Failed to load Live 3D LoRA")
        
        # Apply LoRA trigger prefix
        lora_config = FLUX_LORAS['live_3d']
        trigger_prefix = lora_config.get('trigger_prefix', '')
        enhanced_prompt = prompt
        if trigger_prefix:
            enhanced_prompt = f"{trigger_prefix} {prompt}"
            print(f"🎨 Applied FLUX LoRA trigger prefix: '{trigger_prefix}'")
        
        # Generate image with FLUX + LoRA
        print(f"🎨 Generating image with Live 3D LoRA for: '{enhanced_prompt}' (seed: {seed})")
        seed_generator = torch.Generator(device=GENERATION_CONFIG['device']).manual_seed(seed)
        with torch.no_grad():
            eff_guidance, eff_steps, extra_kwargs = generator._resolve_flux_inference_params(
                guidance_scale, num_inference_steps
            )
            flux_output = generator.flux_pipeline(
                prompt=enhanced_prompt,
                generator=seed_generator,
                num_inference_steps=eff_steps,
                guidance_scale=eff_guidance,
                **extra_kwargs
            )
            image = flux_output.images[0]  # Extract the first image from the output
        
        # Convert PIL Image to bytes
        img_buffer = io.BytesIO()
        image.save(img_buffer, format='PNG')
        image_data = img_buffer.getvalue()
        
        # Encode as base64 for JSON response
        image_base64 = base64.b64encode(image_data).decode('utf-8')
        
        return JSONResponse(content={
            "status": "success",
            "prompt": prompt,
            "seed": seed,
            "image": image_base64,
            "image_size_bytes": len(image_data),
            "pipeline": "flux_only",
            "lora": "live_3d"
        })
        
    except Exception as e:
        print(f"❌ Image generation failed: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Image generation failed: {str(e)}")

@app.post("/generate_image/game_assets/")
async def generate_image_with_game_assets_lora(
    prompt: str = Form(...), 
    seed: Optional[int] = Form(None),
    num_inference_steps: Optional[int] = Form(25),
    guidance_scale: Optional[float] = Form(7.5)
):
    """Generate image only using FLUX with Game Assets LoRA (without 3D generation)."""
    
    # Handle seed
    if seed is None:
        seed = random.randint(0, 2**31 - 1)
    
    try:
        # Switch to FLUX model
        GENERATION_CONFIG['current_model'] = 'flux'
        
        # Load FLUX if not loaded
        if generator.flux_pipeline is None:
            generator._load_flux_models()
        
        # Load the LoRA
        success = generator._load_lora('game_assets')
        if not success:
            raise HTTPException(status_code=500, detail="Failed to load Game Assets LoRA")
        
        # Apply LoRA trigger prefix
        lora_config = FLUX_LORAS['game_assets']
        trigger_prefix = lora_config.get('trigger_prefix', '')
        enhanced_prompt = prompt
        if trigger_prefix:
            enhanced_prompt = f"{trigger_prefix} {prompt}"
            print(f"🎨 Applied FLUX LoRA trigger prefix: '{trigger_prefix}'")
        
        # Generate image with FLUX + LoRA
        print(f"🎨 Generating image with Game Assets LoRA for: '{enhanced_prompt}' (seed: {seed})")
        seed_generator = torch.Generator(device=GENERATION_CONFIG['device']).manual_seed(seed)
        with torch.no_grad():
            eff_guidance, eff_steps, extra_kwargs = generator._resolve_flux_inference_params(
                guidance_scale, num_inference_steps
            )
            flux_output = generator.flux_pipeline(
                prompt=enhanced_prompt,
                generator=seed_generator,
                num_inference_steps=eff_steps,
                guidance_scale=eff_guidance,
                **extra_kwargs
            )
            image = flux_output.images[0]  # Extract the first image from the output
        
        # Convert PIL Image to bytes
        img_buffer = io.BytesIO()
        image.save(img_buffer, format='PNG')
        image_data = img_buffer.getvalue()
        
        # Encode as base64 for JSON response
        image_base64 = base64.b64encode(image_data).decode('utf-8')
        
        return JSONResponse(content={
            "status": "success",
            "prompt": prompt,
            "seed": seed,
            "image": image_base64,
            "image_size_bytes": len(image_data),
            "pipeline": "flux_only",
            "lora": "game_assets"
        })
        
    except Exception as e:
        print(f"❌ Image generation failed: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Image generation failed: {str(e)}")

@app.post("/generate_image/patched_realism/")
async def generate_image_with_patched_realism_lora(
    prompt: str = Form(...), 
    seed: Optional[int] = Form(None),
    num_inference_steps: Optional[int] = Form(25),
    guidance_scale: Optional[float] = Form(7.5)
):
    """Generate image only using FLUX with Patched Realism LoRA (without 3D generation)."""
    
    # Handle seed
    if seed is None:
        seed = random.randint(0, 2**31 - 1)
    
    try:
        # Switch to FLUX model
        GENERATION_CONFIG['current_model'] = 'flux'
        
        # Load FLUX if not loaded
        if generator.flux_pipeline is None:
            generator._load_flux_models()
        
                # Load the LoRA
        success = generator._load_lora('patched_realism')
        if not success:
            raise HTTPException(status_code=500, detail="Failed to load Patched Realism LoRA")
        
        # Apply LoRA trigger prefix
        lora_config = FLUX_LORAS['patched_realism']
        trigger_prefix = lora_config.get('trigger_prefix', '')
        enhanced_prompt = prompt
        if trigger_prefix:
            enhanced_prompt = f"{trigger_prefix} {prompt}"
            print(f"🎨 Applied FLUX LoRA trigger prefix: '{trigger_prefix}'")
        
        # Generate image with FLUX + LoRA
        print(f"🎨 Generating image with Patched Realism LoRA for: '{enhanced_prompt}' (seed: {seed})")
        seed_generator = torch.Generator(device=GENERATION_CONFIG['device']).manual_seed(seed)
        with torch.no_grad():
            eff_guidance, eff_steps, extra_kwargs = generator._resolve_flux_inference_params(
                guidance_scale, num_inference_steps
            )
            flux_output = generator.flux_pipeline(
                prompt=enhanced_prompt,
                generator=seed_generator,
                num_inference_steps=eff_steps,
                guidance_scale=eff_guidance,
                **extra_kwargs
            )
            image = flux_output.images[0]  # Extract the first image from the output
        
        # Convert PIL Image to bytes
        img_buffer = io.BytesIO()
        image.save(img_buffer, format='PNG')
        image_data = img_buffer.getvalue()
        
        # Encode as base64 for JSON response
        image_base64 = base64.b64encode(image_data).decode('utf-8')
        
        return JSONResponse(content={
            "status": "success",
            "prompt": prompt,
            "seed": seed,
            "image": image_base64,
            "image_size_bytes": len(image_data),
            "pipeline": "flux_only",
            "lora": "patched_realism"
        })
        
    except Exception as e:
        print(f"❌ Image generation failed: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Image generation failed: {str(e)}")

@app.post("/generate_image/tf2_style/")
async def generate_image_with_tf2_style_lora(
    prompt: str = Form(...), 
    seed: Optional[int] = Form(None),
    num_inference_steps: Optional[int] = Form(25),
    guidance_scale: Optional[float] = Form(7.5)
):
    """Generate image only using FLUX with TF2 Style LoRA (without 3D generation)."""
    
    # Handle seed
    if seed is None:
        seed = random.randint(0, 2**31 - 1)
    
    try:
        # Switch to FLUX model
        GENERATION_CONFIG['current_model'] = 'flux'
        
        # Load FLUX if not loaded
        if generator.flux_pipeline is None:
            generator._load_flux_models()
        
        # Load the LoRA
        success = generator._load_lora('tf2_style')
        if not success:
            raise HTTPException(status_code=500, detail="Failed to load TF2 Style LoRA")
        
        # Apply LoRA trigger prefix
        lora_config = FLUX_LORAS['tf2_style']
        trigger_prefix = lora_config.get('trigger_prefix', '')
        enhanced_prompt = prompt
        if trigger_prefix:
            enhanced_prompt = f"{trigger_prefix} {prompt}"
            print(f"🎨 Applied FLUX LoRA trigger prefix: '{trigger_prefix}'")
        
        # Generate image with FLUX + LoRA
        print(f"🎨 Generating image with TF2 Style LoRA for: '{enhanced_prompt}' (seed: {seed})")
        seed_generator = torch.Generator(device=GENERATION_CONFIG['device']).manual_seed(seed)
        with torch.no_grad():
            eff_guidance, eff_steps, extra_kwargs = generator._resolve_flux_inference_params(
                guidance_scale, num_inference_steps
            )
            flux_output = generator.flux_pipeline(
                prompt=enhanced_prompt,
                generator=seed_generator,
                num_inference_steps=eff_steps,
                guidance_scale=eff_guidance,
                **extra_kwargs
            )
            image = flux_output.images[0]  # Extract the first image from the output
        
        # Convert PIL Image to bytes
        img_buffer = io.BytesIO()
        image.save(img_buffer, format='PNG')
        image_data = img_buffer.getvalue()
        
        # Encode as base64 for JSON response
        image_base64 = base64.b64encode(image_data).decode('utf-8')
        
        return JSONResponse(content={
            "status": "success",
            "prompt": prompt,
            "seed": seed,
            "image": image_base64,
            "image_size_bytes": len(image_data),
            "pipeline": "flux_only",
            "lora": "tf2_style"
        })
        
    except Exception as e:
        print(f"❌ Image generation failed: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Image generation failed: {str(e)}")

@app.post("/generate_image/baolei/")
async def generate_image_with_baolei_lora(
    prompt: str = Form(...), 
    seed: Optional[int] = Form(None),
    num_inference_steps: Optional[int] = Form(25),
    guidance_scale: Optional[float] = Form(7.5)
):
    """Generate image only using FLUX with Baolei LoRA (without 3D generation)."""
    
    # Handle seed
    if seed is None:
        seed = random.randint(0, 2**31 - 1)
    
    try:
        # Switch to FLUX model
        GENERATION_CONFIG['current_model'] = 'flux'
        
        # Load FLUX if not loaded
        if generator.flux_pipeline is None:
            generator._load_flux_models()
        
        # Load the LoRA
        success = generator._load_lora('baolei')
        if not success:
            raise HTTPException(status_code=500, detail="Failed to load Baolei LoRA")
        
        # Apply LoRA trigger prefix
        lora_config = FLUX_LORAS['baolei']
        trigger_prefix = lora_config.get('trigger_prefix', '')
        enhanced_prompt = prompt
        if trigger_prefix:
            enhanced_prompt = f"{trigger_prefix} {prompt}"
            print(f"🎨 Applied FLUX LoRA trigger prefix: '{trigger_prefix}'")
        
        # Generate image with FLUX + LoRA
        print(f"🎨 Generating image with Baolei LoRA for: '{enhanced_prompt}' (seed: {seed})")
        seed_generator = torch.Generator(device=GENERATION_CONFIG['device']).manual_seed(seed)
        with torch.no_grad():
            eff_guidance, eff_steps, extra_kwargs = generator._resolve_flux_inference_params(
                guidance_scale, num_inference_steps
            )
            flux_output = generator.flux_pipeline(
                prompt=enhanced_prompt,
                generator=seed_generator,
                num_inference_steps=eff_steps,
                guidance_scale=eff_guidance,
                **extra_kwargs
            )
            image = flux_output.images[0]  # Extract the first image from the output
        
        # Convert PIL Image to bytes
        img_buffer = io.BytesIO()
        image.save(img_buffer, format='PNG')
        image_data = img_buffer.getvalue()
        
        # Encode as base64 for JSON response
        image_base64 = base64.b64encode(image_data).decode('utf-8')
        
        return JSONResponse(content={
            "status": "success",
            "prompt": prompt,
            "seed": seed,
            "image": image_base64,
            "image_size_bytes": len(image_data),
            "pipeline": "flux_only",
            "lora": "baolei"
        })
        
    except Exception as e:
        print(f"❌ Image generation failed: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Image generation failed: {str(e)}")

@app.post("/generate_image/cartoon_3d/")
async def generate_image_with_cartoon_3d_lora(
    prompt: str = Form(...), 
    seed: Optional[int] = Form(None),
    num_inference_steps: Optional[int] = Form(25),
    guidance_scale: Optional[float] = Form(7.5)
):
    """Generate image only using FLUX with Cartoon 3D LoRA (without 3D generation)."""
    
    # Handle seed
    if seed is None:
        seed = random.randint(0, 2**31 - 1)
    
    try:
        # Switch to FLUX model
        GENERATION_CONFIG['current_model'] = 'flux'
        
        # Load FLUX if not loaded
        if generator.flux_pipeline is None:
            generator._load_flux_models()
        
        # Load the LoRA
        success = generator._load_lora('cartoon_3d')
        if not success:
            raise HTTPException(status_code=500, detail="Failed to load Cartoon 3D LoRA")
        
        # Apply LoRA trigger prefix
        lora_config = FLUX_LORAS['cartoon_3d']
        trigger_prefix = lora_config.get('trigger_prefix', '')
        enhanced_prompt = prompt
        if trigger_prefix:
            enhanced_prompt = f"{trigger_prefix} {prompt}"
            print(f"🎨 Applied FLUX LoRA trigger prefix: '{trigger_prefix}'")
        
        # Generate image with FLUX + LoRA
        print(f"🎨 Generating image with Cartoon 3D LoRA for: '{enhanced_prompt}' (seed: {seed})")
        seed_generator = torch.Generator(device=GENERATION_CONFIG['device']).manual_seed(seed)
        with torch.no_grad():
            eff_guidance, eff_steps, extra_kwargs = generator._resolve_flux_inference_params(
                guidance_scale, num_inference_steps
            )
            flux_output = generator.flux_pipeline(
                prompt=enhanced_prompt,
                generator=seed_generator,
                num_inference_steps=eff_steps,
                guidance_scale=eff_guidance,
                **extra_kwargs
            )
            image = flux_output.images[0]  # Extract the first image from the output
        
        # Convert PIL Image to bytes
        img_buffer = io.BytesIO()
        image.save(img_buffer, format='PNG')
        image_data = img_buffer.getvalue()
        
        # Encode as base64 for JSON response
        image_base64 = base64.b64encode(image_data).decode('utf-8')
        
        return JSONResponse(content={
            "status": "success",
            "prompt": prompt,
            "seed": seed,
            "image": image_base64,
            "image_size_bytes": len(image_data),
            "pipeline": "flux_only",
            "lora": "cartoon_3d"
        })
        
    except Exception as e:
        print(f"❌ Image generation failed: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Image generation failed: {str(e)}")

@app.post("/generate_image/cinema/")
async def generate_image_with_cinema_lora(
    prompt: str = Form(...), 
    seed: Optional[int] = Form(None),
    num_inference_steps: Optional[int] = Form(25),
    guidance_scale: Optional[float] = Form(7.5)
):
    """Generate image only using FLUX with Cinema LoRA (without 3D generation)."""
    
    # Handle seed
    if seed is None:
        seed = random.randint(0, 2**31 - 1)
    
    try:
        # Switch to FLUX model
        GENERATION_CONFIG['current_model'] = 'flux'
        
        # Load FLUX if not loaded
        if generator.flux_pipeline is None:
            generator._load_flux_models()
        
        # Load the LoRA
        success = generator._load_lora('cinema')
        if not success:
            raise HTTPException(status_code=500, detail="Failed to load Cinema LoRA")
        
        # Apply LoRA trigger prefix
        lora_config = FLUX_LORAS['cinema']
        trigger_prefix = lora_config.get('trigger_prefix', '')
        enhanced_prompt = prompt
        if trigger_prefix:
            enhanced_prompt = f"{trigger_prefix} {prompt}"
            print(f"🎨 Applied FLUX LoRA trigger prefix: '{trigger_prefix}'")
        
        # Generate image with FLUX + LoRA
        print(f"🎨 Generating image with Cinema LoRA for: '{enhanced_prompt}' (seed: {seed})")
        seed_generator = torch.Generator(device=GENERATION_CONFIG['device']).manual_seed(seed)
        with torch.no_grad():
            eff_guidance, eff_steps, extra_kwargs = generator._resolve_flux_inference_params(
                guidance_scale, num_inference_steps
            )
            flux_output = generator.flux_pipeline(
                prompt=enhanced_prompt,
                generator=seed_generator,
                num_inference_steps=eff_steps,
                guidance_scale=eff_guidance,
                **extra_kwargs
            )
            image = flux_output.images[0]  # Extract the first image from the output
        
        # Convert PIL Image to bytes
        img_buffer = io.BytesIO()
        image.save(img_buffer, format='PNG')
        image_data = img_buffer.getvalue()
        
        # Encode as base64 for JSON response
        image_base64 = base64.b64encode(image_data).decode('utf-8')
        
        return JSONResponse(content={
            "status": "success",
            "prompt": prompt,
            "seed": seed,
            "image": image_base64,
            "image_size_bytes": len(image_data),
            "pipeline": "flux_only",
            "lora": "cinema"
        })
        
    except Exception as e:
        print(f"❌ Image generation failed: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Image generation failed: {str(e)}")

@app.post("/generate_image/sd15_game_icon/")
async def generate_image_with_sd15_game_icon_lora(
    prompt: str = Form(...), 
    seed: Optional[int] = Form(None),
    num_inference_steps: Optional[int] = Form(25),
    guidance_scale: Optional[float] = Form(7.5)
):
    """Generate image only using SD1.5 with Game Icon LoRA (without 3D generation)."""
    
    # Handle seed
    if seed is None:
        seed = random.randint(0, 2**31 - 1)
    
    try:
        # Switch to SD1.5 model
        GENERATION_CONFIG['current_model'] = 'sd15'
        
        # Load SD1.5 if not loaded
        if generator.sd15_pipeline is None:
            generator._load_sd15_pipeline()
        
        # Load the LoRA
        success = generator._load_lora('game_icon')
        if not success:
            raise HTTPException(status_code=500, detail="Failed to load SD1.5 Game Icon LoRA")
        
        # Apply LoRA trigger prefix
        lora_config = SD15_LORAS['game_icon']
        trigger_prefix = lora_config.get('trigger_prefix', '')
        enhanced_prompt = prompt
        if trigger_prefix:
            enhanced_prompt = f"{trigger_prefix} {prompt}"
            print(f"🎨 Applied SD1.5 LoRA trigger prefix: '{trigger_prefix}'")
        
        # Generate image with SD1.5 + LoRA
        print(f"🎨 Generating image with SD1.5 Game Icon LoRA for: '{enhanced_prompt}' (seed: {seed})")
        seed_generator = torch.Generator(device=GENERATION_CONFIG['device']).manual_seed(seed)
        with torch.no_grad():
            image = generator.sd15_pipeline(
                prompt=enhanced_prompt,
                generator=seed_generator,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale
            ).images[0]
        
        # Convert PIL Image to bytes
        img_buffer = io.BytesIO()
        image.save(img_buffer, format='PNG')
        image_data = img_buffer.getvalue()
        
        # Encode as base64 for JSON response
        image_base64 = base64.b64encode(image_data).decode('utf-8')
        
        return JSONResponse(content={
            "status": "success",
            "prompt": prompt,
            "seed": seed,
            "image": image_base64,
            "image_size_bytes": len(image_data),
            "pipeline": "sd15_only",
            "lora": "game_icon"
        })
        
    except Exception as e:
        print(f"❌ Image generation failed: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Image generation failed: {str(e)}")

@app.post("/generate_image/necklace/")
async def generate_image_with_necklace_lora(
    prompt: str = Form(...), 
    seed: Optional[int] = Form(None),
    num_inference_steps: Optional[int] = Form(25),
    guidance_scale: Optional[float] = Form(7.5)
):
    """Generate image only using FLUX with Necklace LoRA (without 3D generation)."""
    
    # Handle seed
    if seed is None:
        seed = random.randint(0, 2**31 - 1)
    
    try:
        # Switch to FLUX model
        GENERATION_CONFIG['current_model'] = 'flux'
        
        # Load FLUX if not loaded
        if generator.flux_pipeline is None:
            generator._load_flux_models()
        
        # Load the LoRA
        success = generator._load_lora('necklace')
        if not success:
            raise HTTPException(status_code=500, detail="Failed to load Necklace LoRA")
        
        # Apply LoRA trigger prefix
        lora_config = FLUX_LORAS['necklace']
        trigger_prefix = lora_config.get('trigger_prefix', '')
        enhanced_prompt = prompt
        if trigger_prefix:
            enhanced_prompt = f"{trigger_prefix} {prompt}"
            print(f"🎨 Applied FLUX LoRA trigger prefix: '{trigger_prefix}'")
        
        # Generate image with FLUX + LoRA
        print(f"🎨 Generating image with FLUX Necklace LoRA for: '{enhanced_prompt}' (seed: {seed})")
        seed_generator = torch.Generator(device=GENERATION_CONFIG['device']).manual_seed(seed)
        with torch.no_grad():
            eff_guidance, eff_steps, extra_kwargs = generator._resolve_flux_inference_params(
                guidance_scale, num_inference_steps
            )
            image = generator.flux_pipeline(
                prompt=enhanced_prompt,
                generator=seed_generator,
                num_inference_steps=eff_steps,
                guidance_scale=eff_guidance,
                **extra_kwargs
            ).images[0]
        
        # Convert PIL Image to bytes
        img_buffer = io.BytesIO()
        image.save(img_buffer, format='PNG')
        image_data = img_buffer.getvalue()
        
        # Encode as base64 for JSON response
        image_base64 = base64.b64encode(image_data).decode('utf-8')
        
        return JSONResponse(content={
            "status": "success",
            "prompt": prompt,
            "seed": seed,
            "image": image_base64,
            "image_size_bytes": len(image_data),
            "pipeline": "flux_only",
            "lora": "necklace"
        })
        
    except Exception as e:
        print(f"❌ Image generation failed: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Image generation failed: {str(e)}")

# HunyuanDiT image generation endpoints
@app.post("/generate_image_hunyuan/")
async def generate_hunyuan_image_endpoint(
    prompt: str = Form(...), 
    seed: Optional[int] = Form(None)
):
    """Generate image using HunyuanDiT (without LoRA)."""
    
    # Handle seed
    if seed is None:
        seed = random.randint(0, 2**31 - 1)
    
    try:
        # Generate image with HunyuanDiT
        image = generator.generate_hunyuan_image(prompt, seed)
        
        if image is None:
            raise HTTPException(status_code=500, detail="HunyuanDiT image generation failed")
        
        # Convert PIL Image to bytes
        img_buffer = io.BytesIO()
        image.save(img_buffer, format='PNG')
        image_data = img_buffer.getvalue()
        
        # Encode as base64 for JSON response
        image_base64 = base64.b64encode(image_data).decode('utf-8')
        
        return JSONResponse(content={
            "status": "success",
            "prompt": prompt,
            "seed": seed,
            "image": image_base64,
            "image_size_bytes": len(image_data),
            "pipeline": "hunyuan_dit_only"
        })
        
    except Exception as e:
        print(f"❌ HunyuanDiT image generation failed: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"HunyuanDiT image generation failed: {str(e)}")



# General image generation endpoint (without LoRA)
@app.post("/generate_image/")
async def generate_image_endpoint(
    prompt: str = Form(...), 
    seed: Optional[int] = Form(None),
    num_inference_steps: Optional[int] = Form(25),
    guidance_scale: Optional[float] = Form(7.5)
):
    """Generate image only using current model (without 3D generation)."""
    
    # Handle seed
    if seed is None:
        seed = random.randint(0, 2**31 - 1)
    
    try:
        current_model = GENERATION_CONFIG.get('current_model', 'flux')
        # Create seeded generator for reproducibility across pipelines
        seed_generator = torch.Generator(device=GENERATION_CONFIG['device']).manual_seed(seed)
        
        if current_model == 'flux':
            # Load FLUX if not loaded
            if generator.flux_pipeline is None:
                generator._load_flux_models()
            
            # Generate image with FLUX
            print(f"🎨 Generating image with FLUX for: '{prompt}' (seed: {seed})")
            
            generator._unload_lora()

            with torch.no_grad():
                # Apply schnell overrides if enabled
                eff_guidance = guidance_scale
                eff_steps = num_inference_steps
                eff_guidance, eff_steps, extra_kwargs = generator._resolve_flux_inference_params(
                    eff_guidance, eff_steps
                )
                flux_output = generator.flux_pipeline(
                    prompt=prompt,
                    generator=seed_generator,
                    num_inference_steps=eff_steps,
                    guidance_scale=eff_guidance,
                    **extra_kwargs
                )
                image = flux_output.images[0]  # Extract the first image from the output
                
        elif current_model == 'sdxl':
            # Load SDXL if not loaded
            if generator.sdxl_pipeline is None:
                generator._load_sdxl_pipeline()
            
            # Generate image with SDXL
            print(f"🎨 Generating image with SDXL for: '{prompt}' (seed: {seed})")
            
            with torch.no_grad():
                image = generator.sdxl_pipeline(
                    prompt=prompt,
                    generator=seed_generator,
                    num_inference_steps=num_inference_steps,
                    guidance_scale=guidance_scale
                ).images[0]
                
        elif current_model == 'sd15':
            # Load SD1.5 if not loaded
            if generator.sd15_pipeline is None:
                generator._load_sd15_pipeline()
            
            # Generate image with SD1.5
            print(f"🎨 Generating image with SD1.5 for: '{prompt}' (seed: {seed})")
            
            with torch.no_grad():
                image = generator.sd15_pipeline(
                    prompt=prompt,
                    generator=seed_generator,
                    num_inference_steps=num_inference_steps,
                    guidance_scale=guidance_scale
                ).images[0]
        else:
            raise HTTPException(status_code=400, detail=f"Unsupported model: {current_model}")
        
        # Convert PIL Image to bytes
        img_buffer = io.BytesIO()
        image.save(img_buffer, format='PNG')
        image_data = img_buffer.getvalue()
        
        # Encode as base64 for JSON response
        image_base64 = base64.b64encode(image_data).decode('utf-8')
        
        return JSONResponse(content={
            "status": "success",
            "prompt": prompt,
            "seed": seed,
            "image": image_base64,
            "image_size_bytes": len(image_data),
            "pipeline": f"{current_model}_only",
            "lora": None
        })
        
    except Exception as e:
        print(f"❌ Image generation failed: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Image generation failed: {str(e)}")

# Optimization endpoints
@app.post("/optimize_prompt/")
async def optimize_prompt_endpoint(
    prompt: str = Form(...),
    seed: Optional[int] = Form(None),
    find_optimal_lora: Optional[bool] = Form(True),
    target_score: Optional[float] = Form(0.8)
):
    """Optimize prompt for maximum CLIP alignment score using feedback loops"""
    
    # Handle seed
    if seed is None:
        seed = random.randint(0, 2**31 - 1)
    
    try:
        from prompt_optimization_engine import CLIPAlignmentOptimizer
        
        # Initialize optimizer
        optimizer = CLIPAlignmentOptimizer(
            hunyuan_server_url="http://localhost:8098"
        )
        optimizer.target_score = target_score
        
        # Run comprehensive optimization
        session = await optimizer.optimize_prompt_comprehensive(
            prompt=prompt,
            seed=seed,
            find_optimal_lora=find_optimal_lora
        )
        
        # Calculate validation metrics
        normalized_score = session.final_score / 0.35
        validation_status = "✅ EXCELLENT" if normalized_score >= 0.8 else \
                          "🟡 GOOD" if normalized_score >= 0.6 else \
                          "🟠 POOR" if normalized_score >= 0.3 else "❌ FAIL"
        
        task_fidelity = 1.0 if normalized_score >= 0.8 else \
                       0.75 if normalized_score >= 0.6 else 0.0
        
        return JSONResponse(content={
            "status": "success",
            "session_id": session.session_id,
            "original_prompt": session.original_prompt,
            "optimized_prompt": session.final_prompt,
            "original_score": session.original_score,
            "final_score": session.final_score,
            "normalized_score": normalized_score,
            "improvement": session.total_improvement,
            "validation_status": validation_status,
            "task_fidelity": task_fidelity,
            "optimization_time": session.total_time,
            "optimal_lora": session.iterations[0].lora_endpoint if session.iterations else "isometric_3d"
        })
        
    except Exception as e:
        print(f"❌ Prompt optimization failed: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Prompt optimization failed: {str(e)}")


@app.post("/optimize_and_generate/")
async def optimize_and_generate_endpoint(
    prompt: str = Form(...),
    seed: Optional[int] = Form(None),
    return_compressed: Optional[bool] = Form(True),
    target_score: Optional[float] = Form(0.8),
    num_inference_steps: Optional[int] = Form(NUM_INFERENCE_STEPS),
    guidance_scale: Optional[float] = Form(GENERATION_CONFIG['guidance_scale'])
):
    """Optimize prompt then generate 3D model with optimal settings"""
    
    # Handle seed
    if seed is None:
        seed = random.randint(0, 2**31 - 1)
    
    try:
        from prompt_optimization_engine import CLIPAlignmentOptimizer
        
        print(f"🚀 Starting optimize-and-generate for: '{prompt}'")
        start_time = time.time()
        
        # Step 1: Optimize prompt
        optimizer = CLIPAlignmentOptimizer(
            hunyuan_server_url="http://localhost:8098"
        )
        optimizer.target_score = target_score
        
        optimization_session = await optimizer.optimize_prompt_comprehensive(
            prompt=prompt,
            seed=seed,
            find_optimal_lora=True
        )
        
        optimized_prompt = optimization_session.final_prompt
        optimal_lora = optimization_session.iterations[0].lora_endpoint if optimization_session.iterations else "isometric_3d"
        
        print(f"✅ Prompt optimized: '{optimized_prompt}'")
        print(f"   Optimal LoRA: {optimal_lora}")
        print(f"   Score improvement: {optimization_session.total_improvement:+.4f}")
        
        # Step 2: Load optimal LoRA and generate 3D model
        enhanced_prompt = optimized_prompt
        if optimal_lora and optimal_lora != "none":
            generator._load_lora(optimal_lora)
            
            # Apply LoRA trigger prefix if available
            if optimal_lora in FLUX_LORAS:
                lora_config = FLUX_LORAS[optimal_lora]
                trigger_prefix = lora_config.get('trigger_prefix', '')
                if trigger_prefix:
                    enhanced_prompt = f"{trigger_prefix} {optimized_prompt}"
                    print(f"🎨 Applied FLUX LoRA trigger prefix: '{trigger_prefix}'")
            elif optimal_lora in SD15_LORAS:
                lora_config = SD15_LORAS[optimal_lora]
                trigger_prefix = lora_config.get('trigger_prefix', '')
                if trigger_prefix:
                    enhanced_prompt = f"{trigger_prefix} {optimized_prompt}"
                    print(f"🎨 Applied SD15 LoRA trigger prefix: '{trigger_prefix}'")
        
        result = generator.generate_3d_model(
            enhanced_prompt,
            seed,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale
        )
        
        if result is None:
            return JSONResponse(content={
                "status": "error",
                "error": "3D model generation failed after optimization",
                "optimization_session": {
                    "original_prompt": prompt,
                    "optimized_prompt": optimized_prompt,
                    "improvement": optimization_session.total_improvement
                }
            })
        
        ply_data, compressed_data = result
        
        # Update metrics
        metrics.total_generations += 1
        metrics.successful_generations += 1
        generation_time = time.time() - start_time
        metrics.last_generation_time = generation_time
        metrics.average_generation_time = (
            (metrics.average_generation_time * (metrics.successful_generations - 1) + generation_time) 
            / metrics.successful_generations
        )
        
        # Prepare response
        response_data = {
            "status": "success",
            "prompt": prompt,
            "optimized_prompt": optimized_prompt,
            "seed": seed,
            "optimal_lora": optimal_lora,
            "optimization_improvement": optimization_session.total_improvement,
            "optimization_normalized_score": optimization_session.final_score / 0.35,
            "generation_time": generation_time,
            "ply_size_bytes": len(ply_data),
            "model_format": "gaussian_splatting_ply"
        }
        
        if return_compressed and compressed_data:
            compressed_base64 = base64.b64encode(compressed_data).decode('utf-8')
            response_data.update({
                "compressed_ply": compressed_base64,
                "compressed_size_bytes": len(compressed_data),
                "compression_ratio": len(ply_data) / len(compressed_data)
            })
        else:
            ply_base64 = base64.b64encode(ply_data).decode('utf-8')
            response_data["ply_data"] = ply_base64
        
        return JSONResponse(content=response_data)
        
    except Exception as e:
        print(f"❌ Optimize-and-generate failed: {e}")
        traceback.print_exc()
        metrics.failed_generations += 1
        raise HTTPException(status_code=500, detail=f"Optimize-and-generate failed: {str(e)}")


@app.post("/interrogate_image/")
async def interrogate_image_endpoint(
    image: UploadFile = File(...),
    style_focus: Optional[str] = Form("clip_optimized")
):
    """Use image interrogator to generate optimized prompt from uploaded image"""
    
    try:
        from prompt_optimization_engine import ImageInterrogatorInterface
        
        # Read and process uploaded image
        image_data = await image.read()
        pil_image = Image.open(io.BytesIO(image_data))
        
        # Initialize interrogator
        interrogator = ImageInterrogatorInterface()
        
        # Interrogate image
        optimized_prompt = interrogator.interrogate_image(pil_image, style_focus)
        
        if optimized_prompt:
            return JSONResponse(content={
                "status": "success",
                "optimized_prompt": optimized_prompt,
                "style_focus": style_focus,
                "image_size": pil_image.size
            })
        else:
            return JSONResponse(content={
                "status": "error",
                "error": "Image interrogation failed"
            })
            
    except Exception as e:
        print(f"❌ Image interrogation failed: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Image interrogation failed: {str(e)}")


@app.post("/clip_feedback_loop/")
async def clip_feedback_loop_endpoint(
    prompt: str = Form(...),
    lora_endpoint: Optional[str] = Form("isometric_3d"),
    seed: Optional[int] = Form(None),
    max_iterations: Optional[int] = Form(3)
):
    """Run CLIP feedback optimization loop for specific LoRA endpoint"""
    
    # Handle seed
    if seed is None:
        seed = random.randint(0, 2**31 - 1)
    
    try:
        from prompt_optimization_engine import CLIPAlignmentOptimizer
        
        # Initialize optimizer
        optimizer = CLIPAlignmentOptimizer(
            hunyuan_server_url="http://localhost:8098"
        )
        optimizer.max_iterations = max_iterations
        
        # Run optimization for specific LoRA
        result = optimizer.optimize_for_lora_endpoint(prompt, lora_endpoint, seed)
        
        normalized_original = result.original_score / 0.35
        normalized_optimized = result.optimized_score / 0.35
        
        return JSONResponse(content={
            "status": "success",
            "original_prompt": result.original_prompt,
            "optimized_prompt": result.optimized_prompt,
            "lora_endpoint": result.lora_endpoint,
            "original_score": result.original_score,
            "optimized_score": result.optimized_score,
            "normalized_original": normalized_original,
            "normalized_optimized": normalized_optimized,
            "improvement": result.improvement,
            "strategy_used": result.strategy_used,
            "iterations": result.iteration
        })
        
    except Exception as e:
        print(f"❌ CLIP feedback loop failed: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"CLIP feedback loop failed: {str(e)}")



# New endpoint that returns both PLY and image data
@app.post("/generate_both/")
async def generate_both_ply_and_image(
    prompt: str = Form(...), 
    seed: Optional[int] = Form(None),
    num_inference_steps: Optional[int] = Form(NUM_INFERENCE_STEPS),
    guidance_scale: Optional[float] = Form(GENERATION_CONFIG['guidance_scale']),
    ss_sampling_steps: Optional[int] = Form(GENERATION_CONFIG['ss_sampling_steps']),
    slat_sampling_steps: Optional[int] = Form(GENERATION_CONFIG['slat_sampling_steps']),
    slat_guidance_strength: Optional[float] = Form(GENERATION_CONFIG['slat_guidance_strength']),
    ss_guidance_strength: Optional[float] = Form(GENERATION_CONFIG['ss_guidance_strength'])
):
    """Generate both 3D model (PLY) and image in a single request"""
    try:
        # Handle seed
        if seed is None:
            seed = 42
        
        # unloading lora
        if hasattr(generator, '_unload_lora'):
            generator._unload_lora()

        print(f"🎯 Generating both PLY and image for: '{prompt}' (seed: {seed})")
        
        # Use the generate_3d_model_image method which returns both
        result = generator.generate_3d_model_image(
            prompt,
            seed,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            ss_sampling_steps=ss_sampling_steps,
            slat_sampling_steps=slat_sampling_steps,
            slat_guidance_strength=slat_guidance_strength,
            ss_guidance_strength=ss_guidance_strength
        )
        
        if result is None:
            raise HTTPException(status_code=500, detail="Generation failed")
        
        ply_data, compressed_data, image = result
        
        # Convert PIL Image to base64 for JSON response
        img_buffer = io.BytesIO()
        image.save(img_buffer, format='PNG')
        image_data = img_buffer.getvalue()
        image_base64 = base64.b64encode(image_data).decode('utf-8')
        
        # Prepare response data
        response_data = {
            "status": "success",
            "prompt": prompt,
            "seed": seed,
            "image": image_base64,
            "image_size_bytes": len(image_data),
            "ply_size_bytes": len(ply_data),
            "model_format": "gaussian_splatting_ply",
            "pipeline": "flux_trellis"
        }
        
        # Always send compressed PLY when available (more efficient)
        if compressed_data:
            compressed_base64 = base64.b64encode(compressed_data).decode('utf-8')
            response_data.update({
                "compressed_ply": compressed_base64,
                "compressed_size_bytes": len(compressed_data),
                "compression_ratio": len(ply_data) / len(compressed_data)
            })
        else:
            # Fallback to uncompressed PLY only if compression failed
            ply_base64 = base64.b64encode(ply_data).decode('utf-8')
            response_data["ply_data"] = ply_base64
        
        return JSONResponse(content=response_data)
        
    except Exception as e:
        print(f"❌ Generate both failed: {e}")
        traceback.print_exc()
        return JSONResponse(content={
            "status": "error",
            "message": str(e)
        }, status_code=500)

# LoRA-specific generate_both endpoints
@app.post("/generate_both/cinema/")
async def generate_both_with_cinema_lora(
    prompt: str = Form(...), 
    seed: Optional[int] = Form(None),
    num_inference_steps: Optional[int] = Form(NUM_INFERENCE_STEPS),
    guidance_scale: Optional[float] = Form(GENERATION_CONFIG['guidance_scale']),
    ss_sampling_steps: Optional[int] = Form(GENERATION_CONFIG['ss_sampling_steps']),
    slat_sampling_steps: Optional[int] = Form(GENERATION_CONFIG['slat_sampling_steps']),
    slat_guidance_strength: Optional[float] = Form(GENERATION_CONFIG['slat_guidance_strength']),
    ss_guidance_strength: Optional[float] = Form(GENERATION_CONFIG['ss_guidance_strength'])
):
    """Generate both PLY and image using Cinema Style LoRA"""
    try:
        # Switch to FLUX model first
        GENERATION_CONFIG['current_model'] = 'flux'
        
        # Load the LoRA
        success = generator._load_lora('cinema')
        if not success:
            raise HTTPException(status_code=500, detail="Failed to load Cinema Style LoRA")
        
        # Apply LoRA trigger prefix
        lora_config = FLUX_LORAS['cinema']
        trigger_prefix = lora_config.get('trigger_prefix', '')
        enhanced_prompt = prompt
        if trigger_prefix:
            enhanced_prompt = f"{trigger_prefix} {prompt}"
            print(f"🎨 Applied FLUX LoRA trigger prefix: '{trigger_prefix}'")
        
        # Handle seed
        if seed is None:
            seed = 42
        
        print(f"🎯 Generating both PLY and image with Cinema LoRA for: '{enhanced_prompt}' (seed: {seed})")
        
        # Use the generate_3d_model_image method which returns both
        result = generator.generate_3d_model_image(
            enhanced_prompt,
            seed,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            ss_sampling_steps=ss_sampling_steps,
            slat_sampling_steps=slat_sampling_steps,
            slat_guidance_strength=slat_guidance_strength,
            ss_guidance_strength=ss_guidance_strength
        )
        
        if result is None:
            raise HTTPException(status_code=500, detail="Generation failed")
        
        ply_data, compressed_data, image = result
        
        # Convert PIL Image to base64 for JSON response
        img_buffer = io.BytesIO()
        image.save(img_buffer, format='PNG')
        image_data = img_buffer.getvalue()
        image_base64 = base64.b64encode(image_data).decode('utf-8')
        
        # Prepare response data
        response_data = {
            "status": "success",
            "prompt": prompt,
            "enhanced_prompt": enhanced_prompt,
            "seed": seed,
            "image": image_base64,
            "image_size_bytes": len(image_data),
            "ply_size_bytes": len(ply_data),
            "model_format": "gaussian_splatting_ply",
            "pipeline": "flux_trellis",
            "lora": "cinema"
        }
        
        # Always send compressed PLY when available (more efficient)
        if compressed_data:
            compressed_base64 = base64.b64encode(compressed_data).decode('utf-8')
            response_data.update({
                "compressed_ply": compressed_base64,
                "compressed_size_bytes": len(compressed_data),
                "compression_ratio": len(ply_data) / len(compressed_data)
            })
        else:
            # Fallback to uncompressed PLY only if compression failed
            ply_base64 = base64.b64encode(ply_data).decode('utf-8')
            response_data["ply_data"] = ply_base64
        
        return JSONResponse(content=response_data)
        
    except Exception as e:
        print(f"❌ Generate both with Cinema LoRA failed: {e}")
        traceback.print_exc()
        return JSONResponse(content={
            "status": "error",
            "message": str(e)
        }, status_code=500)


@app.post("/clip_score/")
async def compute_clip_score(
    prompt: str = Form(...),
    image_data: str = Form(...)  # Base64 encoded image
):
    """Compute CLIP alignment score between prompt and image"""
    try:
        # Get the preloaded CLIP analyzer
        clip_analyzer = generator.get_clip_analyzer()
        if clip_analyzer is None:
            raise HTTPException(status_code=500, detail="CLIP model not available")
        
        # Decode base64 image
        try:
            image_bytes = base64.b64decode(image_data)
            image = Image.open(io.BytesIO(image_bytes))
            
            # Ensure image is RGB for CLIP processing
            if image.mode != 'RGB':
                image = image.convert('RGB')
                
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Invalid image data: {str(e)}")
        
        # Compute CLIP score
        clip_score = clip_analyzer.compute_clip_alignment_score(prompt, image)
        
        return JSONResponse(content={
            "status": "success",
            "prompt": prompt,
            "image_size": image.size,
            "clip_score": clip_score,
            "normalized_score": clip_score / 0.35  # Normalize to 0-1 range
        })
        
    except Exception as e:
        print(f"❌ CLIP scoring failed: {e}")
        traceback.print_exc()
        return JSONResponse(content={
            "status": "error",
            "message": str(e)
        }, status_code=500)

# FLUX Socket Inference Endpoint
@app.post("/generate_flux_socket/")
async def generate_flux_socket_image(
    prompt: str = Form(...), 
    seed: Optional[int] = Form(None),
    width: Optional[int] = Form(1024),
    height: Optional[int] = Form(1024)
):
    """Generate image using FLUX socket server (newcomer20_accurate)"""
    
    # Handle seed
    if seed is None:
        seed = random.randint(0, 2**31 - 1)
    
    try:
        # Ensure FLUX socket client is available
        if not hasattr(generator, 'flux_socket_client') or generator.flux_socket_client is None:
            # Initialize socket client
            generator.flux_socket_client = FluxSocketClient()
            generator.flux_use_schnell_socket = True
        
        print(f"🎨 Generating FLUX socket image for: '{prompt}' (seed: {seed}, size: {width}x{height})")
        
        # Generate image via socket
        image = generator.flux_socket_client.generate_image(
            prompt=prompt,
            seed=seed,
            width=width,
            height=height
        )
        
        if image is None:
            raise HTTPException(status_code=500, detail="FLUX socket generation failed")
        
        # Convert PIL Image to bytes
        img_buffer = io.BytesIO()
        image.save(img_buffer, format='PNG')
        image_data = img_buffer.getvalue()
        
        # Encode as base64 for JSON response
        image_base64 = base64.b64encode(image_data).decode('utf-8')
        
        return JSONResponse(content={
            "status": "success",
            "prompt": prompt,
            "seed": seed,
            "width": width,
            "height": height,
            "image": image_base64,
            "image_size_bytes": len(image_data),
            "pipeline": "flux_socket",
            "server": "newcomer20_accurate"
        })
        
    except Exception as e:
        print(f"❌ FLUX socket generation failed: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"FLUX socket generation failed: {str(e)}")

# FLUX Socket + TRELLIS 3D Generation Endpoint
@app.post("/generate_flux_socket_3d/")
async def generate_flux_socket_3d_model(
    prompt: str = Form(...), 
    seed: Optional[int] = Form(None),
    return_compressed: Optional[bool] = Form(True),
    width: Optional[int] = Form(1024),
    height: Optional[int] = Form(1024),
    ss_sampling_steps: Optional[int] = Form(GENERATION_CONFIG['ss_sampling_steps']),
    slat_sampling_steps: Optional[int] = Form(GENERATION_CONFIG['slat_sampling_steps']),
    slat_guidance_strength: Optional[float] = Form(GENERATION_CONFIG['slat_guidance_strength']),
    ss_guidance_strength: Optional[float] = Form(GENERATION_CONFIG['ss_guidance_strength'])
):
    """Generate 3D model using FLUX socket + TRELLIS pipeline"""
    
    # Handle seed
    if seed is None:
        seed = 42
    
    try:
        print(f"🎯 Starting FLUX socket + TRELLIS generation for: '{prompt}' (seed: {seed})")
        
        # Ensure FLUX socket client is available
        if not hasattr(generator, 'flux_socket_client') or generator.flux_socket_client is None:
            # Initialize socket client
            generator.flux_socket_client = FluxSocketClient()
            generator.flux_use_schnell_socket = True
        
        # Initialize asset manager for this generation
        generation_asset = generator.asset_manager.create_asset(prompt, seed)
        
        # Step 1: Generate image with FLUX socket
        print("Step 1: Generating image with FLUX socket...")
        
        image = generator.flux_socket_client.generate_image(
            prompt=prompt,
            seed=seed,
            width=width,
            height=height
        )
        
        if image is None:
            raise HTTPException(status_code=500, detail="FLUX socket image generation failed")
        
        print("✓ FLUX socket image generated successfully")
        generation_asset.add_asset(AssetType.FLUX_IMAGE, image)
        
        # Step 2: Remove background from image
        print("Step 2: Removing background from image...")
        if generator.background_remover is None:
            generator._load_background_remover()
        
        try:
            image_no_bg = generator.background_remover(image)
            print("✓ Background removed successfully")
            generation_asset.add_asset(AssetType.FLUX_IMAGE, image_no_bg)
            image = image_no_bg  # Use the cleaned image for TRELLIS
        except Exception as e:
            print(f"⚠️ Background removal failed: {e}")
            print("   Continuing with original image...")
        
        # Step 3: Generate 3D model with TRELLIS
        print("Step 3: Generating 3D model with TRELLIS...")
        if generator.trellis_pipeline is None:   
            generator._load_trellis_pipeline()
            if generator.trellis_pipeline is None:
                raise RuntimeError("TRELLIS pipeline failed to load; cannot generate 3D model.")
        
        # Resolve TRELLIS quality parameters
        effective_ss_steps = ss_sampling_steps if ss_sampling_steps is not None else GENERATION_CONFIG['ss_sampling_steps']
        effective_slat_steps = slat_sampling_steps if slat_sampling_steps is not None else GENERATION_CONFIG['slat_sampling_steps']
        effective_slat_guidance = slat_guidance_strength if slat_guidance_strength is not None else GENERATION_CONFIG['slat_guidance_strength']
        effective_ss_guidance = ss_guidance_strength if ss_guidance_strength is not None else GENERATION_CONFIG['ss_guidance_strength']

        # Use autocast for fp16
        use_fp16 = GENERATION_CONFIG.get('trellis_use_fp16', True) and torch.cuda.is_available()
        if use_fp16:
            try:
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    outputs = generator.trellis_pipeline.run(
                        image,
                        seed=seed,
                        formats=["gaussian"],
                        preprocess_image=False,
                        sparse_structure_sampler_params={
                            "steps": effective_ss_steps,
                            "cfg_strength": effective_ss_guidance,
                            "cfg_interval": (0.3, 0.98),
                            "rescale_t": 3.0,
                        },
                        slat_sampler_params={
                            "steps": effective_slat_steps,
                            "cfg_strength": effective_slat_guidance,
                            "cfg_interval": (0.3, 0.98),
                            "rescale_t": 3.0,
                        },
                    )
            except RuntimeError as e:
                if "scatter()" in str(e) or "scatter_reduce" in str(e):
                    print("⚠️ FP16 mesh decode failed. Retrying without autocast...")
                    with torch.autocast(device_type="cuda", enabled=False):
                        outputs = generator.trellis_pipeline.run(
                            image,
                            seed=seed,
                            formats=["gaussian"],
                            preprocess_image=False,
                            sparse_structure_sampler_params={
                                "steps": effective_ss_steps,
                                "cfg_strength": effective_ss_guidance,
                                "cfg_interval": (0.3, 0.98),
                                "rescale_t": 3.0,
                            },
                            slat_sampler_params={
                                "steps": effective_slat_steps,
                                "cfg_strength": effective_slat_guidance,
                                "cfg_interval": (0.3, 0.98),
                                "rescale_t": 3.0,
                            },
                        )
                else:
                    raise
        else:
            outputs = generator.trellis_pipeline.run(
                image,
                seed=seed,
                formats=["gaussian", "mesh"],
                preprocess_image=False,
                sparse_structure_sampler_params={
                    "steps": effective_ss_steps,
                    "cfg_strength": effective_ss_guidance,
                    "cfg_interval": (0.3, 0.98),
                    "rescale_t": 3.0,
                },
                slat_sampler_params={
                    "steps": effective_slat_steps,
                    "cfg_strength": effective_slat_guidance,
                    "cfg_interval": (0.3, 0.98),
                    "rescale_t": 3.0,
                },
            )
        
        print("✓ 3D model generated successfully")
        
        # Step 4: Extract Gaussian Splatting PLY
        print("Step 4: Extracting Gaussian Splatting PLY...")
        gaussian_output = outputs['gaussian'][0]
        
        # Save as PLY file
        import io
        ply_buffer = io.BytesIO()
        gaussian_output.save_ply(ply_buffer)
        ply_data = ply_buffer.getvalue()
        
        print(f"✓ Gaussian Splatting PLY extracted ({len(ply_data):,} bytes)")
        generation_asset.add_asset(AssetType.GAUSSIAN_SPLATTING_PLY, ply_data)
        
        # Step 5: Compress PLY if enabled
        compressed_data = None
        if GENERATION_CONFIG.get('auto_compress_ply', True):
            print("Step 5: Compressing PLY with SPZ...")
            try:
                import pyspz
                compressed_data = pyspz.compress(ply_data, workers=-1)
                print(f"🗜️ SPZ Compression successful:")
                print(f"   Original: {len(ply_data):,} bytes ({len(ply_data)/1024/1024:.1f} MB)")
                print(f"   Compressed: {len(compressed_data):,} bytes ({len(compressed_data)/1024/1024:.1f} MB)") 
                print(f"   Ratio: {len(compressed_data)/len(ply_data)*100:.1f}%")
                
                generation_asset.add_asset(AssetType.COMPRESSED_PLY, compressed_data)
            except Exception as e:
                print(f"⚠️ SPZ compression failed: {e}")
                compressed_data = None
        
        # Prepare response data
        response_data = {
            "status": "success",
            "prompt": prompt,
            "seed": seed,
            "width": width,
            "height": height,
            "ply_size_bytes": len(ply_data),
            "model_format": "gaussian_splatting_ply",
            "pipeline": "flux_socket_trellis",
            "server": "newcomer20_accurate"
        }
        
        # Return compressed PLY if requested and available
        if return_compressed and compressed_data:
            compressed_base64 = base64.b64encode(compressed_data).decode('utf-8')
            response_data.update({
                "compressed_ply": compressed_base64,
                "compressed_size_bytes": len(compressed_data),
                "compression_ratio": len(ply_data) / len(compressed_data)
            })
        else:
            # Fallback to uncompressed PLY
            ply_base64 = base64.b64encode(ply_data).decode('utf-8')
            response_data["ply_data"] = ply_base64
        
        return JSONResponse(content=response_data)
        
    except Exception as e:
        print(f"❌ FLUX socket + TRELLIS generation failed: {e}")
        traceback.print_exc()
        return JSONResponse(content={
            "status": "error",
            "message": str(e)
        }, status_code=500)

# Start FLUX Socket Server
@app.post("/start_flux_server/")
async def start_flux_server():
    """Start the FLUX socket server (newcomer20_accurate)"""
    try:
        print("🚀 Starting FLUX socket server...")
        
        # Path to the newcomer20_accurate directory
        flux_dir = os.path.join(os.path.dirname(__file__), "newcomer20_accurate")
        
        if not os.path.exists(flux_dir):
            return JSONResponse(content={
                "status": "error",
                "message": f"FLUX directory not found: {flux_dir}"
            }, status_code=404)
        
        # Start the server process
        flux_process = subprocess.Popen(
            ["uv", "run", "python", "src/main.py"],
            cwd=flux_dir,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        
        # Wait for server to be ready
        time.sleep(5)
        
        # Test connection
        try:
            test_client = FluxSocketClient()
            test_image = test_client.generate_image(
                prompt="test", 
                seed=42, 
                width=512, 
                height=512
            )
            
            if test_image is not None:
                return JSONResponse(content={
                    "status": "success",
                    "message": "FLUX socket server started successfully",
                    "process_id": flux_process.pid,
                    "socket_path": test_client.socket_path,
                    "test_connection": "successful"
                })
            else:
                return JSONResponse(content={
                    "status": "warning",
                    "message": "FLUX server started but connection test failed",
                    "process_id": flux_process.pid,
                    "socket_path": test_client.socket_path
                })
                
        except Exception as e:
            return JSONResponse(content={
                "status": "warning",
                "message": f"FLUX server started but connection test failed: {str(e)}",
                "process_id": flux_process.pid,
                "socket_path": "/home/mbhat/three-gen-subnet-trellis/newcomer20_accurate/inferences.sock"
            })
            
    except Exception as e:
        print(f"❌ Failed to start FLUX server: {e}")
        traceback.print_exc()
        return JSONResponse(content={
            "status": "error",
            "message": f"Failed to start FLUX server: {str(e)}"
        }, status_code=500)

# Check FLUX Server Status
@app.get("/flux_server_status/")
async def get_flux_server_status():
    """Check if the FLUX socket server is running and accessible"""
    try:
        # Try to connect to the socket
        test_client = FluxSocketClient()
        
        # Test with a simple generation
        test_image = test_client.generate_image(
            prompt="test", 
            seed=42, 
            width=512, 
            height=512
        )
        
        if test_image is not None:
            return JSONResponse(content={
                "status": "running",
                "message": "FLUX socket server is running and accessible",
                "socket_path": test_client.socket_path,
                "test_connection": "successful",
                "image_size": test_image.size
            })
        else:
            return JSONResponse(content={
                "status": "error",
                "message": "FLUX socket server is not responding",
                "socket_path": test_client.socket_path,
                "test_connection": "failed"
            })
            
    except Exception as e:
        return JSONResponse(content={
            "status": "error",
            "message": f"FLUX socket server connection failed: {str(e)}",
            "socket_path": "/home/mbhat/three-gen-subnet-trellis/newcomer20_accurate/inferences.sock"
        })

# Test FLUX Socket Connection
@app.post("/test_flux_socket/")
async def test_flux_socket_connection(
    prompt: str = Form("test connection"),
    seed: Optional[int] = Form(42),
    width: Optional[int] = Form(512),
    height: Optional[int] = Form(512)
):
    """Test FLUX socket connection with a simple image generation"""
    try:
        print(f"🧪 Testing FLUX socket connection...")
        
        # Initialize socket client
        test_client = FluxSocketClient()
        
        # Test generation
        test_image = test_client.generate_image(
            prompt=prompt,
            seed=seed,
            width=width,
            height=height
        )
        
        if test_image is not None:
            # Convert to base64 for response
            img_buffer = io.BytesIO()
            test_image.save(img_buffer, format='PNG')
            image_data = img_buffer.getvalue()
            image_base64 = base64.b64encode(image_data).decode('utf-8')
            
            return JSONResponse(content={
                "status": "success",
                "message": "FLUX socket connection test successful",
                "prompt": prompt,
                "seed": seed,
                "width": width,
                "height": height,
                "image": image_base64,
                "image_size": test_image.size,
                "image_size_bytes": len(image_data),
                "socket_path": test_client.socket_path
            })
        else:
            return JSONResponse(content={
                "status": "error",
                "message": "FLUX socket generation failed during test",
                "socket_path": test_client.socket_path
            })
            
    except Exception as e:
        print(f"❌ FLUX socket test failed: {e}")
        traceback.print_exc()
        return JSONResponse(content={
            "status": "error",
            "message": f"FLUX socket test failed: {str(e)}"
        }, status_code=500)

# FLUX Socket + TRELLIS with LoRA Support
@app.post("/generate_flux_socket_3d_lora/{lora_key}")
async def generate_flux_socket_3d_model_with_lora(
    lora_key: str,
    prompt: str = Form(...), 
    seed: Optional[int] = Form(None),
    return_compressed: Optional[bool] = Form(True),
    width: Optional[int] = Form(1024),
    height: Optional[int] = Form(1024),
    ss_sampling_steps: Optional[int] = Form(GENERATION_CONFIG['ss_sampling_steps']),
    slat_sampling_steps: Optional[int] = Form(GENERATION_CONFIG['slat_sampling_steps']),
    slat_guidance_strength: Optional[float] = Form(GENERATION_CONFIG['slat_guidance_strength']),
    ss_guidance_strength: Optional[float] = Form(GENERATION_CONFIG['ss_guidance_strength'])
):
    """Generate 3D model using FLUX socket + TRELLIS with specific LoRA"""
    
    # Handle seed
    if seed is None:
        seed = 42
    
    try:
        # Validate LoRA key
        if lora_key not in FLUX_LORAS:
            raise HTTPException(status_code=400, detail=f"Invalid LoRA key: {lora_key}")
        
        lora_config = FLUX_LORAS[lora_key]
        print(f"🎨 Using FLUX LoRA: {lora_config['name']}")
        
        # Apply LoRA trigger prefix
        trigger_prefix = lora_config.get('trigger_prefix', '')
        enhanced_prompt = prompt
        if trigger_prefix:
            enhanced_prompt = f"{trigger_prefix} {prompt}"
            print(f"🎨 Applied FLUX LoRA trigger prefix: '{trigger_prefix}'")
        
        print(f"🎯 Starting FLUX socket + TRELLIS generation with {lora_key} LoRA for: '{enhanced_prompt}' (seed: {seed})")
        
        # Ensure FLUX socket client is available
        if not hasattr(generator, 'flux_socket_client') or generator.flux_socket_client is None:
            # Initialize socket client
            generator.flux_socket_client = FluxSocketClient()
            generator.flux_use_schnell_socket = True
        
        # Initialize asset manager for this generation
        generation_asset = generator.asset_manager.create_asset(enhanced_prompt, seed)
        
        # Step 1: Generate image with FLUX socket
        print("Step 1: Generating image with FLUX socket...")
        
        image = generator.flux_socket_client.generate_image(
            prompt=enhanced_prompt,
            seed=seed,
            width=width,
            height=height
        )
        
        if image is None:
            raise HTTPException(status_code=500, detail="FLUX socket image generation failed")
        
        print("✓ FLUX socket image generated successfully")
        generation_asset.add_asset(AssetType.FLUX_IMAGE, image)
        
        # Step 2: Remove background from image
        print("Step 2: Removing background from image...")
        if generator.background_remover is None:
            generator._load_background_remover()
        
        try:
            image_no_bg = generator.background_remover(image)
            print("✓ Background removed successfully")
            generation_asset.add_asset(AssetType.FLUX_IMAGE, image_no_bg)
            image = image_no_bg  # Use the cleaned image for TRELLIS
        except Exception as e:
            print(f"⚠️ Background removal failed: {e}")
            print("   Continuing with original image...")
        
        # Step 3: Generate 3D model with TRELLIS
        print("Step 3: Generating 3D model with TRELLIS...")
        if generator.trellis_pipeline is None:   
            generator._load_trellis_pipeline()
            if generator.trellis_pipeline is None:
                raise RuntimeError("TRELLIS pipeline failed to load; cannot generate 3D model.")
        
        # Resolve TRELLIS quality parameters
        effective_ss_steps = ss_sampling_steps if ss_sampling_steps is not None else GENERATION_CONFIG['ss_sampling_steps']
        effective_slat_steps = slat_sampling_steps if slat_sampling_steps is not None else GENERATION_CONFIG['slat_sampling_steps']
        effective_slat_guidance = slat_guidance_strength if slat_guidance_strength is not None else GENERATION_CONFIG['slat_guidance_strength']
        effective_ss_guidance = ss_guidance_strength if ss_guidance_strength is not None else GENERATION_CONFIG['ss_guidance_strength']

        # Use autocast for fp16
        use_fp16 = GENERATION_CONFIG.get('trellis_use_fp16', True) and torch.cuda.is_available()
        if use_fp16:
            try:
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    outputs = generator.trellis_pipeline.run(
                        image,
                        seed=seed,
                        formats=["gaussian"],
                        preprocess_image=False,
                        sparse_structure_sampler_params={
                            "steps": effective_ss_steps,
                            "cfg_strength": effective_ss_guidance,
                            "cfg_interval": (0.3, 0.98),
                            "rescale_t": 3.0,
                        },
                        slat_sampler_params={
                            "steps": effective_slat_steps,
                            "cfg_strength": effective_slat_guidance,
                            "cfg_interval": (0.3, 0.98),
                            "rescale_t": 3.0,
                        },
                    )
            except RuntimeError as e:
                if "scatter()" in str(e) or "scatter_reduce" in str(e):
                    print("⚠️ FP16 mesh decode failed. Retrying without autocast...")
                    with torch.autocast(device_type="cuda", enabled=False):
                        outputs = generator.trellis_pipeline.run(
                            image,
                            seed=seed,
                            formats=["gaussian"],
                            preprocess_image=False,
                            sparse_structure_sampler_params={
                                "steps": effective_ss_steps,
                                "cfg_strength": effective_ss_guidance,
                                "cfg_interval": (0.3, 0.98),
                                "rescale_t": 3.0,
                            },
                            slat_sampler_params={
                                "steps": effective_slat_steps,
                                "cfg_strength": effective_slat_guidance,
                                "cfg_interval": (0.3, 0.98),
                                "rescale_t": 3.0,
                            },
                        )
                else:
                    raise
        else:
            outputs = generator.trellis_pipeline.run(
                image,
                seed=seed,
                formats=["gaussian", "mesh"],
                preprocess_image=False,
                sparse_structure_sampler_params={
                    "steps": effective_ss_steps,
                    "cfg_strength": effective_ss_guidance,
                    "cfg_interval": (0.3, 0.98),
                    "rescale_t": 3.0,
                },
                slat_sampler_params={
                    "steps": effective_slat_steps,
                    "cfg_strength": effective_slat_guidance,
                    "cfg_interval": (0.3, 0.98),
                    "rescale_t": 3.0,
                },
            )
        
        print("✓ 3D model generated successfully")
        
        # Step 4: Extract Gaussian Splatting PLY
        print("Step 4: Extracting Gaussian Splatting PLY...")
        gaussian_output = outputs['gaussian'][0]
        
        # Save as PLY file
        import io
        ply_buffer = io.BytesIO()
        gaussian_output.save_ply(ply_buffer)
        ply_data = ply_buffer.getvalue()
        
        print(f"✓ Gaussian Splatting PLY extracted ({len(ply_data):,} bytes)")
        generation_asset.add_asset(AssetType.GAUSSIAN_SPLATTING_PLY, ply_data)
        
        # Step 5: Compress PLY if enabled
        compressed_data = None
        if GENERATION_CONFIG.get('auto_compress_ply', True):
            print("Step 5: Compressing PLY with SPZ...")
            try:
                import pyspz
                compressed_data = pyspz.compress(ply_data, workers=-1)
                print(f"🗜️ SPZ Compression successful:")
                print(f"   Original: {len(ply_data):,} bytes ({len(ply_data)/1024/1024:.1f} MB)")
                print(f"   Compressed: {len(compressed_data):,} bytes ({len(compressed_data)/1024/1024:.1f} MB)") 
                print(f"   Ratio: {len(compressed_data)/len(ply_data)*100:.1f}%")
                
                generation_asset.add_asset(AssetType.COMPRESSED_PLY, compressed_data)
            except Exception as e:
                print(f"⚠️ SPZ compression failed: {e}")
                compressed_data = None
        
        # Prepare response data
        response_data = {
            "status": "success",
            "prompt": prompt,
            "enhanced_prompt": enhanced_prompt,
            "seed": seed,
            "width": width,
            "height": height,
            "lora": lora_key,
            "lora_name": lora_config['name'],
            "ply_size_bytes": len(ply_data),
            "model_format": "gaussian_splatting_ply",
            "pipeline": "flux_socket_trellis_lora",
            "server": "newcomer20_accurate"
        }
        
        # Return compressed PLY if requested and available
        if return_compressed and compressed_data:
            compressed_base64 = base64.b64encode(compressed_data).decode('utf-8')
            response_data.update({
                "compressed_ply": compressed_base64,
                "compressed_size_bytes": len(compressed_data),
                "compression_ratio": len(ply_data) / len(compressed_data)
            })
        else:
            # Fallback to uncompressed PLY
            ply_base64 = base64.b64encode(ply_data).decode('utf-8')
            response_data["ply_data"] = ply_base64
        
        return JSONResponse(content=response_data)
        
    except Exception as e:
        print(f"❌ FLUX socket + TRELLIS generation with LoRA failed: {e}")
        traceback.print_exc()
        return JSONResponse(content={
            "status": "error",
            "message": str(e)
        }, status_code=500)


@app.post("/generate_3d_from_prompt_grid_flow/")
async def generate_3d_from_prompt_grid_flow_endpoint(
    base_prompt: str = Form(...),
    style: str = Form("standard", description="Style to use: standard, cinema, 3d"),
    seed: int = Form(42, description="Random seed for reproducibility"),
    num_inference_steps: int = Form(8, description="Number of inference steps for image generation"),
    guidance_scale: float = Form(3.5, description="Guidance scale for image generation"),
    width: int = Form(1024, description="Image width"),
    height: int = Form(1024, description="Image height"),
    upscale: bool = Form(False, description="Whether to upscale images using Real-ESRGAN (4x)"),
    remove_background: bool = Form(True, description="Whether to remove backgrounds from images"),
    ss_guidance_strength: float = Form(7.5, description="Sparse structure guidance strength"),
    ss_sampling_steps: int = Form(21, description="Sparse structure sampling steps"),
    slat_guidance_strength: float = Form(4.0, description="SLAT guidance strength"),
    slat_sampling_steps: int = Form(24, description="SLAT sampling steps"),
    return_compressed: bool = Form(True, description="Whether to return compressed PLY"),
    save_preview: bool = Form(False, description="Whether to save preview video"),
    save_intermediate: bool = Form(False, description="Whether to save intermediate outputs (grid, cropped images, background-removed images)"),
    filter_low_quality: bool = Form(True, description="Whether to filter low-quality Gaussians"),
    timing: bool = Form(False, description="Whether to enable detailed timing measurements"),
    image_endpoint: str = Form("standard", description="Image generation endpoint: standard, cinema, lora"),
    lora_model: Optional[str] = Form(None, description="LoRA model to use if endpoint is 'lora'"),
    use_short_prompt: bool = Form(True, description="Whether to use short prompt to avoid CLIP token limits")
):
    """
    Comprehensive endpoint that follows the exact flow from test_img2img_prompt.py:
    1. Generate grid image with multiple views
    2. Crop grid into individual images
    3. Optionally upscale images using Real-ESRGAN
    4. Optionally remove backgrounds
    5. Generate 3D model using TRELLIS multi-image pipeline
    """
    
    try:
        # Check if generator is available
        if generator is None:
            raise HTTPException(status_code=500, detail="Generator not initialized")
        
        # Input validation
        if not base_prompt or not base_prompt.strip():
            raise HTTPException(status_code=400, detail="base_prompt cannot be empty")
        
        if width <= 0 or height <= 0:
            raise HTTPException(status_code=400, detail="width and height must be positive")
        
        if width > 2048 or height > 2048:
            raise HTTPException(status_code=400, detail="width and height cannot exceed 2048")
        
        if num_inference_steps <= 0 or num_inference_steps > 50:
            raise HTTPException(status_code=400, detail="num_inference_steps must be between 1 and 50")
        
        if guidance_scale <= 0 or guidance_scale > 20:
            raise HTTPException(status_code=400, detail="guidance_scale must be between 0.1 and 20")
        
        if style not in ["standard", "cinema", "3d"]:
            raise HTTPException(status_code=400, detail="style must be one of: standard, cinema, 3d")
        
        print(f"🎯 Starting comprehensive grid flow 3D generation for: '{base_prompt}'")
        print(f"   Style: {style}, Seed: {seed}, Upscale: {upscale}, Background removal: {remove_background}")
        start_time = time.time()
        
        # Step 1: Generate grid image with multiple views
        print("\n🎨 Step 1: Generating grid image with multiple views...")
        grid_start_time = time.time() if timing else None
        
        # Create grid prompt based on style and short_prompt option
        if use_short_prompt:
            grid_prompt = create_grid_prompt_short(base_prompt, style)
            print(f"   Using short prompt ({len(grid_prompt)} chars): '{grid_prompt}'")
        else:
            grid_prompt = create_grid_prompt(base_prompt, style)
            print(f"   Using full prompt ({len(grid_prompt)} chars): '{grid_prompt[:100]}...'")
        
        # Generate grid image using FLUX
        try:
            # Check if FLUX pipeline is loaded
            if generator.flux_pipeline is None:
                print("   Loading FLUX pipeline...")
                generator._load_flux_models()
                if generator.flux_pipeline is None:
                    raise RuntimeError("Failed to load FLUX pipeline")
                print("   ✓ FLUX pipeline loaded successfully")
            
            # Validate pipeline is callable
            if not callable(generator.flux_pipeline):
                raise RuntimeError("FLUX pipeline is not callable")
            
            print("   Generating grid image with FLUX...")
            
            # Load appropriate LoRA based on image_endpoint
            if image_endpoint == "cinema":
                print("   🎬 Loading Cinema LoRA...")
                success = generator._load_lora('cinema')
                if not success:
                    print("   ⚠️ Failed to load Cinema LoRA, continuing without it...")
                else:
                    print("   ✅ Cinema LoRA loaded successfully")
            elif image_endpoint == "lora" and lora_model:
                print(f"   🎭 Loading custom LoRA: {lora_model}")
                success = generator._load_lora(lora_model)
                if not success:
                    print(f"   ⚠️ Failed to load {lora_model} LoRA, continuing without it...")
                else:
                    print(f"   ✅ {lora_model} LoRA loaded successfully")
            
            # Create proper seed generator for FLUX
            seed_generator = torch.Generator(device=GENERATION_CONFIG.get('device', 'cuda')).manual_seed(seed)
            
            # Use the correct FLUX pipeline parameters based on working examples
            with torch.no_grad():
                flux_output = generator.flux_pipeline(
                    prompt=grid_prompt,
                    generator=seed_generator,
                    num_inference_steps=num_inference_steps,
                    guidance_scale=guidance_scale,
                    width=width,
                    height=height
                )
            
            if flux_output is None or not hasattr(flux_output, 'images') or len(flux_output.images) == 0:
                raise RuntimeError("FLUX pipeline returned invalid output")
            
            grid_image = flux_output.images[0]
            
            if grid_image is None or grid_image.size[0] == 0 or grid_image.size[1] == 0:
                raise RuntimeError("Failed to generate valid grid image")
            
            print(f"   ✓ Grid image generated successfully ({grid_image.size[0]}x{grid_image.size[1]})")
            
            # Save grid image if requested
            if save_intermediate:
                try:
                    grid_filename = f"grid_{base_prompt[:30]}_{seed}.png"
                    grid_path = os.path.join(GENERATION_CONFIG['output_dir'], grid_filename)
                    grid_image.save(grid_path)
                    print(f"   💾 Grid image saved: {grid_path}")
                except Exception as save_e:
                    print(f"   ⚠️ Failed to save grid image: {save_e}")
            
        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=f"Failed to generate grid image: {str(e)}"
            )
        
        if timing:
            grid_time = time.time() - grid_start_time
            print(f"   Grid generation time: {grid_time:.2f}s")
        
        # Step 2: Crop grid into individual views
        print("\n✂️ Step 2: Cropping grid into individual views...")
        crop_start_time = time.time() if timing else None
        
        try:
            cropped_images = crop_grid_image(grid_image, grid_size=(2, 2))
            print(f"   ✓ Grid cropped into 4 views:")
            for view_name, image in cropped_images.items():
                print(f"     - {view_name}: {image.size[0]}x{image.size[1]}")
                
                # Save cropped images if requested
                if save_intermediate:
                    try:
                        cropped_filename = f"cropped_{view_name}_{base_prompt[:30]}_{seed}.png"
                        cropped_path = os.path.join(GENERATION_CONFIG['output_dir'], cropped_filename)
                        image.save(cropped_path)
                        print(f"       💾 Saved: {cropped_path}")
                    except Exception as save_e:
                        print(f"       ⚠️ Failed to save {view_name} image: {save_e}")
        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=f"Failed to crop grid image: {str(e)}"
            )
        
        if timing:
            crop_time = time.time() - crop_start_time
            print(f"   Cropping time: {crop_time:.2f}s")
        
        # Step 3: Optionally upscale images using Real-ESRGAN
        upscaled_images = {}
        upscale_time = 0  # Initialize timing variable
        if upscale:
            print("\n🚀 Step 3: Upscaling images with Real-ESRGAN (4x)...")
            
            # Check GPU memory before upscaling
            if torch.cuda.is_available():
                try:
                    gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3  # GB
                    allocated_memory = torch.cuda.memory_allocated() / 1024**3  # GB
                    free_memory = gpu_memory - allocated_memory
                    
                    if free_memory < 4.0:  # Need at least 4GB for upscaling
                        print(f"   ⚠️ Insufficient GPU memory for upscaling: {free_memory:.1f}GB free, need 4GB+")
                        print("   Skipping upscaling to avoid OOM errors...")
                        upscale = False
                    else:
                        print(f"   GPU Memory: {free_memory:.1f}GB free, proceeding with upscaling")
                except Exception as e:
                    print(f"   ⚠️ Could not check GPU memory: {e}")
                    print("   Proceeding with upscaling...")
            
            upscale_start_time = time.time() if timing else None
            
            try:
                # Check if Real-ESRGAN is available
                try:
                    from RealESRGAN import RealESRGAN
                    realesrgan_available = True
                except ImportError:
                    realesrgan_available = False
                    print("   ⚠️ Real-ESRGAN not available, skipping upscaling")
                
                if realesrgan_available:
                    # Initialize Real-ESRGAN
                    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                    model = RealESRGAN(device, scale=4)
                    
                    # Load weights with fallback
                    weights_path = 'weights/RealESRGAN_x4.pth'
                    try:
                        model.load_weights(weights_path, download=True)
                    except Exception as e:
                        print(f"   ⚠️ Failed to load Real-ESRGAN weights: {e}")
                        print("   Skipping upscaling...")
                        realesrgan_available = False
                    
                    if realesrgan_available:
                        # Upscale the three main views
                        for view_name in ["front", "side", "back"]:
                            if view_name in cropped_images:
                                print(f"   Upscaling {view_name} view...")
                                image = cropped_images[view_name]
                                
                                # Process with Real-ESRGAN
                                sr_image = model.predict(image, batch_size=4, patches_size=192)
                                
                                # Store upscaled image
                                upscaled_images[view_name] = sr_image
                                
                                # Save upscaled image if requested
                                if save_intermediate:
                                    try:
                                        upscaled_filename = f"upscaled_{view_name}_{base_prompt[:30]}_{seed}.png"
                                        upscaled_path = os.path.join(GENERATION_CONFIG['output_dir'], upscaled_filename)
                                        sr_image.save(upscaled_path)
                                        print(f"       💾 Saved: {upscaled_path}")
                                    except Exception as save_e:
                                        print(f"       ⚠️ Failed to save {view_name} upscaled image: {save_e}")
                                
                                # Print size info
                                scale_factor = sr_image.size[0] / image.size[0]
                                print(f"     ✓ {view_name}: {image.size[0]}x{image.size[1]} → {sr_image.size[0]}x{sr_image.size[1]} ({scale_factor}x)")
                        
                        print(f"   ✓ Upscaling completed for {len(upscaled_images)} views")
                    else:
                        print("   ⚠️ Real-ESRGAN weights not available, using original images")
                else:
                    print("   ⚠️ Real-ESRGAN not available, using original images")
                    
            except Exception as e:
                print(f"   ⚠️ Upscaling failed: {e}")
                print("   Continuing with original images...")
            
            if timing:
                upscale_time = time.time() - upscale_start_time
                print(f"   Upscaling time: {upscale_time:.2f}s")
        
        # Step 4: Prepare images for 3D generation (use upscaled if available, otherwise original)
        print("\n🖼️ Step 4: Preparing images for 3D generation...")
        prep_start_time = time.time() if timing else None
        
        # Use upscaled images if available, otherwise use original cropped images
        images_for_3d = []
        image_names = ["front", "side", "back"]
        
        for view_name in image_names:
            if view_name in upscaled_images:
                images_for_3d.append(upscaled_images[view_name])
                print(f"   Using upscaled {view_name} view: {upscaled_images[view_name].size[0]}x{upscaled_images[view_name].size[1]}")
            elif view_name in cropped_images:
                images_for_3d.append(cropped_images[view_name])
                print(f"   Using original {view_name} view: {cropped_images[view_name].size[0]}x{cropped_images[view_name].size[1]}")
            else:
                raise HTTPException(
                    status_code=500,
                    detail=f"Missing {view_name} view for 3D generation"
                )
        
        if timing:
            prep_time = time.time() - prep_start_time
            print(f"   Image preparation time: {prep_time:.2f}s")
        
        # Step 5: Optionally remove backgrounds
        bg_total_time = 0  # Initialize timing variable
        if remove_background:
            print("\n🧹 Step 5: Removing backgrounds from images...")
            bg_start_time = time.time() if timing else None
            
            try:
                if generator.background_remover is None:
                    print("   Loading background remover...")
                    generator._load_background_remover()
                    if generator.background_remover is None:
                        raise RuntimeError("Failed to load background remover")
                    print("   ✓ Background remover loaded successfully")
                
                # Validate background remover is callable
                if not callable(generator.background_remover):
                    raise RuntimeError("Background remover is not callable")
                
                cleaned_images = []
                for i, (image, name) in enumerate(zip(images_for_3d, image_names)):
                    print(f"   Removing background from {name} image...")
                    bg_single_start = time.time() if timing else None
                    
                    cleaned_image = generator.background_remover(image)
                    
                    if timing:
                        bg_single_time = time.time() - bg_single_start
                        print(f"     ✓ {name} background removed in {bg_single_time:.2f}s")
                    else:
                        print(f"     ✓ {name} background removed")
                    
                    # Save background-removed image if requested
                    if save_intermediate:
                        try:
                            bg_removed_filename = f"bg_removed_{name}_{base_prompt[:30]}_{seed}.png"
                            bg_removed_path = os.path.join(GENERATION_CONFIG['output_dir'], bg_removed_filename)
                            cleaned_image.save(bg_removed_path)
                            print(f"       💾 Saved: {bg_removed_path}")
                        except Exception as save_e:
                            print(f"       ⚠️ Failed to save {name} background-removed image: {save_e}")
                    
                    cleaned_images.append(cleaned_image)
                
                # Update images for 3D generation
                images_for_3d = cleaned_images
                print(f"   ✓ Background removal completed for all {len(cleaned_images)} images")
                
            except Exception as e:
                print(f"   ⚠️ Background removal failed: {e}")
                print("   Continuing with original images...")
            
            if timing:
                bg_total_time = time.time() - bg_start_time
                print(f"   Background removal time: {bg_total_time:.2f}s")
        
        # Step 6: Generate 3D model using TRELLIS multi-image pipeline
        print("\n🎨 Step 6: Generating 3D model using TRELLIS pipeline...")
        trellis_start_time = time.time() if timing else None
        
        try:
            # Check if TRELLIS pipeline is available
            if generator.trellis_pipeline is None:
                print("   Loading TRELLIS pipeline...")
                generator._load_trellis_pipeline()
                if generator.trellis_pipeline is None:
                    raise RuntimeError("Failed to load TRELLIS pipeline")
                print("   ✓ TRELLIS pipeline loaded successfully")
            
            # Validate pipeline is callable
            if not hasattr(generator.trellis_pipeline, 'run_multi_image'):
                raise RuntimeError("TRELLIS pipeline missing run_multi_image method")
            
            # Enhanced TRELLIS parameters for maximum quality (Gaussian only)
            use_fp16 = GENERATION_CONFIG.get('trellis_use_fp16', True) and torch.cuda.is_available()
            
            if use_fp16:
                try:
                    print("   Using FP16 optimization...")
                    fp16_start = time.time() if timing else None
                    
                    with torch.autocast(device_type="cuda", dtype=torch.float16):
                        outputs = generator.trellis_pipeline.run_multi_image(
                            images_for_3d,
                            seed=seed,
                            formats=["gaussian"],  # Only Gaussian splatting, no mesh
                            preprocess_image=True,
                            sparse_structure_sampler_params={
                                "steps": ss_sampling_steps,
                                "cfg_strength": ss_guidance_strength,
                                "cfg_interval": (0.3, 0.98),
                                "rescale_t": 3.0,
                            },
                            slat_sampler_params={
                                "steps": slat_sampling_steps,
                                "cfg_strength": slat_guidance_strength,
                                "cfg_interval": (0.3, 0.98),
                                "rescale_t": 3.0,
                            },
                            mode="stochastic",
                        )
                    
                    if timing:
                        fp16_time = time.time() - fp16_start
                        print(f"     ✓ FP16 generation completed in {fp16_time:.2f}s")
                    else:
                        print(f"     ✓ FP16 generation completed")
                    
                except RuntimeError as e:
                    print("   ⚠️ FP16 generation failed, retrying without autocast...")
                    fp16_fallback_start = time.time() if timing else None
                    
                    with torch.autocast(device_type="cuda", enabled=False):
                        outputs = generator.trellis_pipeline.run_multi_image(
                            images_for_3d,
                            seed=seed,
                            formats=["gaussian"],
                            preprocess_image=True,
                            sparse_structure_sampler_params={
                                "steps": ss_sampling_steps,
                                "cfg_strength": ss_guidance_strength,
                                "cfg_interval": (0.3, 0.98),
                                "rescale_t": 3.0,
                            },
                            slat_sampler_params={
                                "steps": slat_sampling_steps,
                                "cfg_strength": slat_guidance_strength,
                                "cfg_interval": (0.3, 0.98),
                                "rescale_t": 3.0,
                            },
                            mode="stochastic",
                        )
                    
                    if timing:
                        fp16_fallback_time = time.time() - fp16_fallback_start
                        print(f"     ✓ FP16 fallback generation completed in {fp16_fallback_time:.2f}s")
                    else:
                        print(f"     ✓ FP16 fallback generation completed")
            else:
                print("   Using FP32 generation...")
                fp32_start = time.time() if timing else None
                
                outputs = generator.trellis_pipeline.run_multi_image(
                    images_for_3d,
                    seed=seed,
                    formats=["gaussian"],
                    preprocess_image=True,
                    sparse_structure_sampler_params={
                        "steps": ss_sampling_steps,
                        "cfg_strength": ss_guidance_strength,
                        "cfg_interval": (0.3, 0.98),
                        "rescale_t": 3.0,
                    },
                    slat_sampler_params={
                        "steps": slat_sampling_steps,
                        "cfg_strength": slat_guidance_strength,
                        "cfg_interval": (0.3, 0.98),
                        "rescale_t": 3.0,
                    },
                    mode="stochastic",
                )
                
                if timing:
                    fp32_time = time.time() - fp32_start
                    print(f"     ✓ FP32 generation completed in {fp32_time:.2f}s")
                else:
                    print(f"     ✓ FP32 generation completed")
            
            print(f"   ✓ 3D model generated successfully using TRELLIS pipeline")
            
        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=f"Failed to generate 3D model with TRELLIS: {str(e)}"
            )
        
        if timing:
            trellis_time = time.time() - trellis_start_time
            print(f"   TRELLIS generation time: {trellis_time:.2f}s")
        
        # Step 7: Extract and enhance Gaussian Splatting PLY
        print("\n🔧 Step 7: Extracting and enhancing Gaussian Splatting PLY...")
        ply_start_time = time.time() if timing else None
        
        try:
            gaussian_output = outputs['gaussian'][0]
            
            # Quality enhancement: Filter low-quality splats
            if filter_low_quality:
                print("   Enhancing quality by filtering low-quality splats...")
                quality_start = time.time() if timing else None
                
                try:
                    points = gaussian_output.points
                    opacities = gaussian_output.opacities
                    scales = gaussian_output.scales
                    
                    # Filter out low-opacity and very small splats
                    opacity_threshold = 0.01
                    scale_threshold = 0.001
                    
                    # Create quality mask
                    quality_mask = (opacities > opacity_threshold) & (torch.norm(scales, dim=1) > scale_threshold)
                    
                    if quality_mask.sum() > 7000:  # Ensure minimum splat count
                        # Apply filtering
                        gaussian_output.points = points[quality_mask]
                        gaussian_output.opacities = opacities[quality_mask]
                        gaussian_output.scales = scales[quality_mask]
                        gaussian_output.rotations = gaussian_output.rotations[quality_mask]
                        gaussian_output.features_dc = gaussian_output.features_dc[quality_mask]
                        gaussian_output.features_rest = gaussian_output.features_rest[quality_mask]
                        gaussian_output.normals = gaussian_output.normals[quality_mask]
                        
                        if timing:
                            quality_time = time.time() - quality_start
                            print(f"     ✓ Quality enhancement: Kept {quality_mask.sum().item():,} high-quality splats out of {len(points):,} in {quality_time:.2f}s")
                        else:
                            print(f"     ✓ Quality enhancement: Kept {quality_mask.sum().item():,} high-quality splats out of {len(points):,}")
                    else:
                        print(f"     Quality enhancement skipped: Too few splats would remain ({quality_mask.sum().item()})")
                        
                except Exception as e:
                    print(f"     Quality enhancement failed: {e}")
                    print("     Continuing with original splats...")
            else:
                print("   Quality filtering disabled, keeping all splats")
            
            # Save as PLY file
            save_start = time.time() if timing else None
            ply_buffer = io.BytesIO()
            gaussian_output.save_ply(ply_buffer)
            ply_data = ply_buffer.getvalue()
            
            if timing:
                save_time = time.time() - save_start
                print(f"     ✓ Gaussian Splatting PLY extracted ({len(ply_data):,} bytes) in {save_time:.2f}s")
                
                ply_total_time = time.time() - ply_start_time
                print(f"   Total PLY processing time: {ply_total_time:.2f}s")
            else:
                print(f"     ✓ Gaussian Splatting PLY extracted ({len(ply_data):,} bytes)")
            
        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=f"Failed to extract PLY data: {str(e)}"
            )
        
        # Step 8: Generate preview video if requested
        preview_video = None
        video_time = 0  # Initialize timing variable
        if save_preview:
            try:
                print("\n🎬 Step 8: Generating preview video...")
                video_start = time.time()
                
                # Check if required dependencies are available
                try:
                    import imageio
                    imageio_available = True
                except ImportError:
                    imageio_available = False
                    print("   ⚠️ ImageIO not available, skipping preview video generation")
                
                if imageio_available:
                    video = render_utils.render_video(outputs['gaussian'][0], num_frames=120)['color']
                    
                    # Convert to MP4 bytes
                    video_buffer = io.BytesIO()
                    imageio.mimsave(video_buffer, video, fps=15, format='mp4')
                    preview_video = video_buffer.getvalue()
                    
                    video_time = time.time() - video_start
                    print(f"   ✓ Preview video generated ({len(preview_video):,} bytes) in {video_time:.2f}s")
                else:
                    print("   ⚠️ Preview video generation skipped due to missing dependencies")
                    
            except Exception as e:
                print(f"   ⚠️ Preview video generation failed: {e}")
                print("   Continuing without preview video...")
        
        # Step 9: Compress PLY if requested
        compressed_data = None
        compression_time = 0  # Initialize timing variable
        if return_compressed:
            try:
                print("\n🗜️ Step 9: Compressing PLY with SPZ...")
                compression_start = time.time()
                
                # Check if PySPZ is available
                try:
                    import pyspz
                    pyspz_available = True
                except ImportError:
                    pyspz_available = False
                    print("   ⚠️ PySPZ not available, skipping compression")
                
                if pyspz_available:
                    compressed_data = pyspz.compress(ply_data, workers=-1)
                    
                    compression_time = time.time() - compression_start
                    print(f"     ✓ SPZ Compression successful in {compression_time:.2f}s:")
                    print(f"       Original: {len(ply_data):,} bytes ({len(ply_data)/1024/1024:.1f} MB)")
                    print(f"       Compressed: {len(compressed_data):,} bytes ({len(compressed_data)/1024/1024:.1f} MB)")
                    print(f"       Ratio: {len(compressed_data)/len(ply_data)*100:.1f}%")
                else:
                    print("   ⚠️ Compression skipped due to missing PySPZ dependency")
                    compressed_data = None
                
            except Exception as e:
                print(f"     ⚠️ SPZ compression failed: {e}")
                print("   Continuing without compression...")
                compressed_data = None
        
        # Calculate total generation time
        generation_time = time.time() - start_time
        print(f"\n🎉 Comprehensive grid flow 3D generation completed in {generation_time:.2f}s")
        
        # Print timing breakdown if enabled
        if timing:
            print(f"\n📊 Timing breakdown:")
            print(f"   - Grid generation: {grid_time:.2f}s")
            print(f"   - Grid cropping: {crop_time:.2f}s")
            if upscale and upscale_time > 0:
                print(f"   - Image upscaling: {upscale_time:.2f}s")
            if remove_background and bg_total_time > 0:
                print(f"   - Background removal: {bg_total_time:.2f}s")
            print(f"   - TRELLIS pipeline: {trellis_time:.2f}s")
            print(f"   - PLY processing: {ply_total_time:.2f}s")
            if save_preview and preview_video and video_time > 0:
                print(f"   - Video generation: {video_time:.2f}s")
            if return_compressed and compressed_data and compression_time > 0:
                print(f"   - Compression: {compression_time:.2f}s")
        
        # Prepare response data
        response_data = {
            "status": "success",
            "base_prompt": base_prompt,
            "style": style,
            "seed": seed,
            "generation_time": generation_time,
            "pipeline": "comprehensive_grid_flow",
            "steps_completed": [
                "grid_image_generation",
                "grid_cropping",
                "optional_upscaling" if upscale else "no_upscaling",
                "optional_background_removal" if remove_background else "no_background_removal",
                "trellis_3d_generation"
            ],
            "image_sizes": [f"{img.size[0]}x{img.size[1]}" for img in images_for_3d],
            "upscaled": upscale,
            "background_removed": remove_background,
            "quality_filtering": filter_low_quality,
            "ply_size_bytes": len(ply_data),
            "model_format": "gaussian_splatting_ply"
        }
        
        if compressed_data:
            response_data.update({
                "compressed_size_bytes": len(compressed_data),
                "compression_ratio": len(compressed_data)/len(ply_data)*100
            })
        
        if preview_video:
            response_data["preview_video_size_bytes"] = len(preview_video)
        
        # Return compressed data if requested
        if return_compressed and compressed_data:
            return Response(
                content=compressed_data,
                media_type="application/octet-stream",
                headers={
                    "Content-Disposition": f"attachment; filename=grid_flow_{base_prompt[:30]}_{seed}.ply.spz",
                    "X-Generation-Seed": str(seed),
                    "X-Model-Format": "gaussian_splatting_ply",
                    "X-Pipeline": "comprehensive_grid_flow",
                    "X-Compression": "spz",
                    "X-Compression-Ratio": f"{len(compressed_data)/len(ply_data)*100:.1f}%",
                    "X-Generation-Time": f"{generation_time:.2f}s",
                    "X-Response-Data": json.dumps(response_data)
                }
            )
        
        # Return uncompressed PLY data
        return Response(
            content=ply_data,
            media_type="application/octet-stream",
            headers={
                "Content-Disposition": f"attachment; filename=grid_flow_{base_prompt[:30]}_{seed}.ply",
                "X-Generation-Seed": str(seed),
                "X-Model-Format": "gaussian_splatting_ply",
                "X-Pipeline": "comprehensive_grid_flow",
                "X-Compression": "none",
                "X-Generation-Time": f"{generation_time:.2f}s",
                "X-Response-Data": json.dumps(response_data)
            }
        )
        
    except Exception as e:
        print(f"❌ Comprehensive grid flow 3D generation failed: {e}")
        traceback.print_exc()
        raise HTTPException(
            status_code=500,
            detail=f"Comprehensive grid flow 3D generation failed: {str(e)}"
        )

# Helper functions for grid generation
def create_grid_prompt_short(base_prompt: str, style: str = "standard") -> str:
    """Create a short grid prompt that fits within CLIP token limits."""
    if style == "cinema":
        base_system = "cinematic, dramatic lighting, professional photography"
    elif style == "3d":
        base_system = "3D isometric object asset, white background"
    else:
        base_system = "high quality, centered composition"
    
    grid_prompt = f"""2x2 grid: {base_prompt}. Top-left: front view. Top-right: side view. Bottom-left: back view. Bottom-right: 45-degree view. Same object, style, lighting."""
    return grid_prompt

def create_grid_prompt(base_prompt: str, style: str = "standard") -> str:
    """Create a detailed grid prompt for multiple views."""
    if style == "cinema":
        base_system = "cinematic, dramatic lighting, professional photography, centered composition, clean edges, high detail"
    elif style == "3d":
        base_system = "3D isometric object asset, white background, centered composition, clean edges, high detail, professional 3D modeling style"
    else:
        base_system = "high quality, centered composition, clean edges, high detail, professional photography"
    
    grid_prompt = f"""A 2x2 grid composed of four visually distinct images of the same {base_prompt}:

Top-left: {base_system}, {base_prompt}, front view, facing camera directly, symmetrical composition, centered subject, white background.

Top-right: {base_system}, {base_prompt}, side view, 90 degree angle, profile view, clear silhouette, centered subject, white background.

Bottom-left: {base_system}, {base_prompt}, back view, opposite side from front, clear rear details, centered subject, white background.

Bottom-right: {base_system}, {base_prompt}, 45-degree angle view, three-quarter perspective, showing both front and side, centered subject, white background.

Each section should be visually distinct while maintaining the exact same {base_prompt} object, style, lighting, and quality. The grid should have clear borders between sections."""
    
    return grid_prompt

def crop_grid_image(grid_image, grid_size=(2, 2)):
    """Crop a grid image into individual sections."""
    width, height = grid_image.size
    cell_width = width // grid_size[1]
    cell_height = height // grid_size[0]
    
    cropped_images = {}
    positions = ["front", "side", "back", "three_quarter"]
    
    for i, position in enumerate(positions):
        row = i // grid_size[1]
        col = i % grid_size[1]
        
        left = col * cell_width
        top = row * cell_height
        right = left + cell_width
        bottom = top + cell_height
        
        cropped = grid_image.crop((left, top, right, bottom))
        cropped_images[position] = cropped
    
    return cropped_images



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

