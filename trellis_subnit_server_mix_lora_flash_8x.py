#!/usr/bin/env python3
"""
Subnet 17 (404-GEN) - FLUX + TRELLIS Generation Server with 8-GPU Parallel Processing
Purpose: HTTP server for high-quality 3D model generation using Flux and TRELLIS on 8 GPUs
Produces validation-compatible Gaussian Splatting PLY files with SPZ compression

Text Prompt → FLUX Image → TRELLIS 3D → Gaussian Splatting PLY + SPZ Compression

# Generate 3D model on specific GPU
curl -X POST "http://localhost:8096/generate/" \
  -F "prompt=a blue ceramic vase with red trim" \
  -F "seed=42" \
  -F "gpu_id=0"

# Generate 8 models in parallel
curl -X POST "http://localhost:8096/generate_parallel/" \
  -F "prompt=a blue ceramic vase with red trim" \
  -F "seeds=42,43,44,45,46,47,48,49"

# Get GPU status
curl "http://localhost:8096/gpu_status/"

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
from typing import Optional, Dict, Any, Tuple, List
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
from concurrent.futures import ThreadPoolExecutor, as_completed
import psutil

from fastapi import FastAPI, Form, HTTPException, UploadFile, File
from fastapi.responses import Response, JSONResponse
import uvicorn
import torch

# Multi-GPU Configuration
MULTI_GPU_CONFIG = {
    'num_gpus': 8,
    'gpu_memory_limit_gb': 20,  # Per GPU memory limit
    'parallel_generation_limit': 8,  # Max parallel generations
    'gpu_affinity': True,  # Bind processes to specific GPUs
    'load_balancing': 'round_robin',  # 'round_robin', 'least_loaded', 'random'
}

# Set environment variables for multi-GPU
os.environ['SPCONV_ALGO'] = 'native'
# os.environ['ATTN_BACKEND'] = 'xformers'
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

# Multi-GPU management
class MultiGPUManager:
    """Manages 8 GPUs for parallel processing"""
    
    def __init__(self):
        self.num_gpus = min(MULTI_GPU_CONFIG['num_gpus'], torch.cuda.device_count())
        self.gpu_states = {}
        self.gpu_locks = {}
        self.gpu_loads = {}
        self.gpu_memory_usage = {}
        
        # Initialize GPU states
        for gpu_id in range(self.num_gpus):
            self.gpu_states[gpu_id] = {
                'status': 'idle',  # 'idle', 'busy', 'error'
                'current_job': None,
                'memory_used_gb': 0.0,
                'memory_total_gb': 0.0,
                'temperature': 0.0,
                'utilization': 0.0,
                'last_update': time.time()
            }
            self.gpu_locks[gpu_id] = threading.Lock()
            self.gpu_loads[gpu_id] = 0
            self.gpu_memory_usage[gpu_id] = 0.0
        
        print(f"🚀 Multi-GPU Manager initialized with {self.num_gpus} GPUs")
        self._update_gpu_info()
    
    def _update_gpu_info(self):
        """Update GPU information and status"""
        try:
            for gpu_id in range(self.num_gpus):
                if torch.cuda.is_available():
                    torch.cuda.set_device(gpu_id)
                    
                    # Get memory info
                    memory_allocated = torch.cuda.memory_allocated(gpu_id)
                    memory_reserved = torch.cuda.memory_reserved(gpu_id)
                    memory_total = torch.cuda.get_device_properties(gpu_id).total_memory
                    
                    self.gpu_states[gpu_id]['memory_used_gb'] = memory_allocated / 1e9
                    self.gpu_states[gpu_id]['memory_total_gb'] = memory_total / 1e9
                    
                    # Get GPU utilization (if available)
                    try:
                        # Use nvidia-smi for detailed GPU info
                        result = subprocess.run(
                            ['nvidia-smi', '--query-gpu=utilization.gpu,temperature.gpu,memory.used,memory.total', 
                             '--format=csv,noheader,nounits', '-i', str(gpu_id)],
                            capture_output=True, text=True
                        )
                        if result.returncode == 0:
                            lines = result.stdout.strip().split('\n')
                            if lines and len(lines) > 0:
                                parts = lines[0].split(', ')
                                if len(parts) >= 4:
                                    self.gpu_states[gpu_id]['utilization'] = float(parts[0])
                                    self.gpu_states[gpu_id]['temperature'] = float(parts[1])
                    except:
                        pass
                    
                    self.gpu_states[gpu_id]['last_update'] = time.time()
                    
        except Exception as e:
            print(f"⚠️ Error updating GPU info: {e}")
    
    def get_available_gpu(self, strategy: str = None) -> Optional[int]:
        """Get available GPU based on strategy"""
        if strategy is None:
            strategy = MULTI_GPU_CONFIG['load_balancing']
        
        self._update_gpu_info()
        
        if strategy == 'round_robin':
            # Simple round-robin assignment
            for gpu_id in range(self.num_gpus):
                if self.gpu_states[gpu_id]['status'] == 'idle':
                    return gpu_id
            return None
            
        elif strategy == 'least_loaded':
            # Find GPU with least memory usage
            best_gpu = None
            best_memory = float('inf')
            
            for gpu_id in range(self.num_gpus):
                if self.gpu_states[gpu_id]['status'] == 'idle':
                    memory_used = self.gpu_states[gpu_id]['memory_used_gb']
                    if memory_used < best_memory:
                        best_memory = memory_used
                        best_gpu = gpu_id
            
            return best_gpu
            
        elif strategy == 'random':
            # Random assignment from available GPUs
            available_gpus = [gpu_id for gpu_id in range(self.num_gpus) 
                            if self.gpu_states[gpu_id]['status'] == 'idle']
            return random.choice(available_gpus) if available_gpus else None
        
        return None
    
    def reserve_gpu(self, gpu_id: int, job_id: str) -> bool:
        """Reserve a GPU for a specific job"""
        if gpu_id not in self.gpu_states:
            return False
        
        with self.gpu_locks[gpu_id]:
            if self.gpu_states[gpu_id]['status'] == 'idle':
                self.gpu_states[gpu_id]['status'] = 'busy'
                self.gpu_states[gpu_id]['current_job'] = job_id
                self.gpu_loads[gpu_id] += 1
                return True
            return False
    
    def release_gpu(self, gpu_id: int):
        """Release a GPU after job completion"""
        if gpu_id in self.gpu_states:
            with self.gpu_locks[gpu_id]:
                self.gpu_states[gpu_id]['status'] = 'idle'
                self.gpu_states[gpu_id]['current_job'] = None
                self.gpu_loads[gpu_id] = max(0, self.gpu_loads[gpu_id] - 1)
    
    def get_gpu_status(self) -> Dict[str, Any]:
        """Get comprehensive GPU status"""
        self._update_gpu_info()
        return {
            'num_gpus': self.num_gpus,
            'gpu_states': self.gpu_states,
            'gpu_loads': self.gpu_loads,
            'total_jobs': sum(self.gpu_loads.values()),
            'available_gpus': sum(1 for state in self.gpu_states.values() if state['status'] == 'idle'),
            'busy_gpus': sum(1 for state in self.gpu_states.values() if state['status'] == 'busy'),
            'error_gpus': sum(1 for state in self.gpu_states.values() if state['status'] == 'error')
        }
    
    def get_parallel_generation_capacity(self) -> int:
        """Get how many parallel generations can be handled"""
        available = sum(1 for state in self.gpu_states.values() if state['status'] == 'idle')
        return min(available, MULTI_GPU_CONFIG['parallel_generation_limit'])

# Global multi-GPU manager
gpu_manager = MultiGPUManager()

seed = 42
torch.manual_seed(seed)
# torch.use_deterministic_algorithms(True)

torch.manual_seed(seed)
torch.cuda.manual_seed(seed)  # If using GPU
np.random.seed(seed)
random.seed(seed)

torch.backends.cudnn.deterministic = True    # For reproducibility with cuDNN
torch.backends.cudnn.benchmark = False       # Disable for reproducibility

# Add TRELLIS to Python path
import sys
TRELLIS_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "TRELLIS-TextoImagen3D")
sys.path.append(TRELLIS_PATH)

# Add Hunyuan3D path for background removal
HUNYUAN3D_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Hunyuan3D-2")
sys.path.append(HUNYUAN3D_PATH)

# Import TRELLIS components
from diffusers import FluxTransformer2DModel, FluxPipeline, BitsAndBytesConfig, GGUFQuantizationConfig, StableDiffusionPipeline
from transformers import T5EncoderModel, BitsAndBytesConfig as BitsAndBytesConfigTF
from trellis.pipelines import TrellisImageTo3DPipeline
from trellis.utils import render_utils, postprocessing_utils

# Import background removal
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
    'guidance_scale': 4.0,  # Increased from 3.5 for better quality
    'ss_guidance_strength': 9.5,  # Increased from 8.5 for stronger structure guidance
    'ss_sampling_steps': 30,  # Increased from 23 for more refinement
    'slat_guidance_strength': 5.0,  # Increased from 4.0 for better detail preservation
    'slat_sampling_steps': 30,  # Increased from 24 for more refinement
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
    # Multi-GPU settings
    'multi_gpu_enabled': True,
    'gpu_id': 0,  # Default GPU ID
    'parallel_generation': False,
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
        'path': '/home/mbhat/three-gen-subnet-trellis/LORAS/ZKcZdffUM6qyMYiEE8ed0_adapter_model_comfy_converted.safetensors',
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
        'path': '/home/mbhat/three-gen-subnet-trellis/LORAS/everyday_000002000.safetensors',
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

class TrellisGenerator:
    def __init__(self, gpu_id: int = 0):
        # GPU assignment
        self.gpu_id = gpu_id
        self.device = f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu"
        
        # Set CUDA device for this instance
        if torch.cuda.is_available():
            try:
                torch.cuda.set_device(gpu_id)
                torch.cuda.empty_cache()
                
                # Pre-allocate some GPU memory to avoid fragmentation
                try:
                    dummy_tensor = torch.zeros(1, device=f"cuda:{gpu_id}")
                    del dummy_tensor
                    torch.cuda.empty_cache()
                except Exception as e:
                    print(f"⚠️ GPU memory pre-allocation failed on GPU {gpu_id}: {e}")
                
                print(f"🔧 TRELLIS Generator initialized on GPU {gpu_id}")
            except Exception as e:
                print(f"❌ Failed to initialize GPU {gpu_id}: {e}")
                raise e
        else:
            print(f"🔧 TRELLIS Generator initialized on CPU")
        
        # Initialize model instance variables
        self.flux_pipeline = None
        self.flux_transformer = None
        self.flux_text_encoder_2 = None
        self.sdxl_pipeline = None
        self.sd15_pipeline = None
        self.hunyuan_pipeline = None
        self.trellis_pipeline = None
        self.background_remover = None
        
        self.metrics = GenerationMetrics()
        self.generation_lock = threading.Lock()
        
        # Initialize asset manager
        self.asset_manager = AssetManager(GENERATION_CONFIG['output_dir'])
        
        # Get HuggingFace token
        try:
            from huggingface_hub import HfFolder
            token = HfFolder.get_token()
            if token:
                os.environ["HUGGINGFACE_TOKEN"] = token
                print(f"✓ HuggingFace token loaded from cache (GPU {gpu_id})")
            else:
                print(f"⚠️ No HuggingFace token found in cache (GPU {gpu_id})")
        except Exception as e:
            print(f"⚠️ Error getting token from cache (GPU {gpu_id}): {e}")
        
        Path(GENERATION_CONFIG['output_dir']).mkdir(exist_ok=True)
        self.ready = True

    def _clear_gpu_memory(self):
        """Clear GPU memory cache aggressively"""
        gc.collect()
        
        if torch.cuda.is_available():
            try:
                # Set device to this GPU's device
                torch.cuda.set_device(self.gpu_id)
                
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()
                torch.cuda.synchronize()
                
                try:
                    torch.cuda.ipc_collect()
                except:
                    pass
                
                if hasattr(torch.cuda, 'reset_accumulated_memory_stats'):
                    torch.cuda.reset_accumulated_memory_stats()
                
                # Get memory info for this specific GPU
                gpu_memory_free, gpu_memory_total = torch.cuda.mem_get_info(self.gpu_id)
                memory_used = gpu_memory_total - gpu_memory_free
                print(f"🧠 GPU {self.gpu_id} Memory: {memory_used / 1e9:.1f}GB used, {gpu_memory_free / 1e9:.1f}GB free")
                return gpu_memory_free / 1e9
                
            except Exception as e:
                print(f"⚠️ Error clearing GPU {self.gpu_id} memory: {e}")
                return 0
        
        return 0

    def _load_flux_models(self):
        """Load FLUX models"""
        if self.flux_pipeline is not None:
            print(f"✓ FLUX models already loaded on GPU {self.gpu_id}")
            return
            
        print(f"🔧 Loading FLUX models on GPU {self.gpu_id}...")
        
        try:
            # Ensure we're on the correct GPU
            if torch.cuda.is_available():
                torch.cuda.set_device(self.gpu_id)
                torch.cuda.empty_cache()
                print(f"   Set CUDA device to GPU {self.gpu_id}")
            
            device = f"cuda:{self.gpu_id}" if torch.cuda.is_available() else "cpu"
            huggingface_token = os.getenv("HUGGINGFACE_TOKEN")
            dtype = torch.bfloat16
            
            file_url = GENERATION_CONFIG['flux_model_url']
            file_url = file_url.replace("/resolve/main/", "/blob/main/").replace("?download=true", "")
            single_file_base_model = GENERATION_CONFIG['flux_base_model']
            
            # Load text encoder with 8-bit quantization
            print(f"   Loading FLUX text encoder with 8-bit quantization on GPU {self.gpu_id}...")
            quantization_config_tf = BitsAndBytesConfigTF(
                load_in_8bit=True
            )
            
            # Load text encoder directly to GPU (8-bit models handle device placement automatically)
            self.flux_text_encoder_2 = T5EncoderModel.from_pretrained(
                single_file_base_model,
                subfolder="text_encoder_2", 
                torch_dtype=dtype, 
                quantization_config=quantization_config_tf, 
                token=huggingface_token,
                device_map=f"cuda:{self.gpu_id}" if torch.cuda.is_available() else "cpu"
            )
            
            # Load transformer with GGUF configuration
            print(f"   Loading FLUX transformer with GGUF quantization on GPU {self.gpu_id}...")
            self.flux_transformer = FluxTransformer2DModel.from_single_file(
                file_url, 
                subfolder="transformer", 
                quantization_config=GGUFQuantizationConfig(compute_dtype=dtype), 
                torch_dtype=dtype, 
                config=single_file_base_model,
                device_map=f"cuda:{self.gpu_id}" if torch.cuda.is_available() else "cpu"
            )
            
            # Initialize pipeline
            print(f"   Initializing FLUX pipeline on GPU {self.gpu_id}...")
            self.flux_pipeline = FluxPipeline.from_pretrained(
                single_file_base_model, 
                transformer=self.flux_transformer, 
                text_encoder_2=self.flux_text_encoder_2, 
                torch_dtype=dtype, 
                token=huggingface_token
            )
            
            # Move to specific GPU after initialization
            if torch.cuda.is_available():
                try:
                    # Set device first
                    torch.cuda.set_device(self.gpu_id)
                    
                    # Move pipeline to GPU
                    self.flux_pipeline = self.flux_pipeline.to(f"cuda:{self.gpu_id}")
                    
                    # Apply optimizations
                    try:
                        self.flux_pipeline.to(memory_format=torch.channels_last)
                        
                        # Compile VAE for better performance
                        try:
                            self.flux_pipeline.vae = torch.compile(self.flux_pipeline.vae, mode="max-autotune")
                            print(f"   ✓ VAE compiled successfully on GPU {self.gpu_id}")
                        except Exception as compile_error:
                            print(f"   ⚠️ VAE compilation failed on GPU {self.gpu_id}: {compile_error}")
                            print(f"   Continuing without compilation...")
                    except Exception as opt_error:
                        print(f"   ⚠️ Pipeline optimization failed on GPU {self.gpu_id}: {opt_error}")
                        print(f"   Continuing without optimization...")
                        
                except Exception as gpu_error:
                    print(f"   ❌ Failed to move pipeline to GPU {self.gpu_id}: {gpu_error}")
                    raise gpu_error
            
            print(f"✅ FLUX models loaded successfully on GPU {self.gpu_id}")
            
            # Small delay to ensure GPU stability
            time.sleep(1)
            
        except Exception as e:
            print(f"❌ FLUX model loading failed on GPU {self.gpu_id}: {e}")
            traceback.print_exc()
            self._unload_flux_models()
    
    def _unload_flux_models(self):
        """Unload FLUX models to free GPU memory"""
        print(f"🧹 Unloading FLUX models from GPU {self.gpu_id}...")
        
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
            print(f"✅ FLUX models unloaded from GPU {self.gpu_id}: {', '.join(models_unloaded)}")

    def _load_sdxl_pipeline(self):
        """Load SDXL pipeline"""
        if self.sdxl_pipeline is not None:
            print(f"✓ SDXL pipeline already loaded on GPU {self.gpu_id}")
            return
            
        print(f"🔧 Loading SDXL pipeline on GPU {self.gpu_id}...")
        
        try:
            # Ensure we're on the correct GPU
            if torch.cuda.is_available():
                torch.cuda.set_device(self.gpu_id)
                torch.cuda.empty_cache()
            
            device = f"cuda:{self.gpu_id}" if torch.cuda.is_available() else "cpu"
            huggingface_token = os.getenv("HUGGINGFACE_TOKEN")
            
            # Load SDXL with optimizations
            self.sdxl_pipeline = StableDiffusionPipeline.from_pretrained(
                GENERATION_CONFIG['sdxl_model_path'],
                torch_dtype=torch.float16,
                use_safetensors=True,
                variant="fp16",
                token=huggingface_token
            )
            
            # Move to specific GPU after initialization
            if torch.cuda.is_available():
                try:
                    # Set device first
                    torch.cuda.set_device(self.gpu_id)
                    
                    # Move pipeline to GPU
                    self.sdxl_pipeline = self.sdxl_pipeline.to(f"cuda:{self.gpu_id}")
                    
                    # Apply optimizations
                    self.sdxl_pipeline.enable_attention_slicing()
                    self.sdxl_pipeline.enable_vae_slicing()
                    
                    # Enable memory efficient attention if available
                    if hasattr(self.sdxl_pipeline, 'enable_xformers_memory_efficient_attention'):
                        self.sdxl_pipeline.enable_xformers_memory_efficient_attention()
                        
                except Exception as gpu_error:
                    print(f"   ❌ Failed to move SDXL pipeline to GPU {self.gpu_id}: {gpu_error}")
                    raise gpu_error
            else:
                # Apply optimizations for CPU
                self.sdxl_pipeline.enable_attention_slicing()
                self.sdxl_pipeline.enable_vae_slicing()
            
            print(f"✅ SDXL pipeline loaded successfully on GPU {self.gpu_id}")
            
            # Small delay to ensure GPU stability
            time.sleep(1)
            
        except Exception as e:
            print(f"❌ SDXL pipeline loading failed on GPU {self.gpu_id}: {e}")
            traceback.print_exc()
            self._unload_sdxl_pipeline()

    def _unload_sdxl_pipeline(self):
        """Unload SDXL pipeline to free GPU memory"""
        if self.sdxl_pipeline is not None:
            print(f"🧹 Unloading SDXL pipeline from GPU {self.gpu_id}...")
            del self.sdxl_pipeline
            self.sdxl_pipeline = None
            self._clear_gpu_memory()
            print(f"✅ SDXL pipeline unloaded from GPU {self.gpu_id}")

    def _load_sd15_pipeline(self):
        """Load SD1.5 pipeline"""
        if self.sd15_pipeline is not None:
            print(f"✓ SD1.5 pipeline already loaded on GPU {self.gpu_id}")
            return
            
        print(f"🔧 Loading SD1.5 pipeline on GPU {self.gpu_id}...")
        
        try:
            # Ensure we're on the correct GPU
            if torch.cuda.is_available():
                torch.cuda.set_device(self.gpu_id)
                torch.cuda.empty_cache()
            
            device = f"cuda:{self.gpu_id}" if torch.cuda.is_available() else "cpu"
            huggingface_token = os.getenv("HUGGINGFACE_TOKEN")
            
            # Load SD1.5 with optimizations
            self.sd15_pipeline = StableDiffusionPipeline.from_pretrained(
                GENERATION_CONFIG['sd15_model_path'],
                torch_dtype=torch.float16,
                use_safetensors=True,
                token=huggingface_token
            )
            
            # Move to specific GPU after initialization
            if torch.cuda.is_available():
                try:
                    # Set device first
                    torch.cuda.set_device(self.gpu_id)
                    
                    # Move pipeline to GPU
                    self.sd15_pipeline = self.sd15_pipeline.to(f"cuda:{self.gpu_id}")
                    
                    # Apply optimizations
                    self.sd15_pipeline.enable_attention_slicing()
                    self.sd15_pipeline.enable_vae_slicing()
                    
                    # Enable memory efficient attention if available
                    if hasattr(self.sd15_pipeline, 'enable_xformers_memory_efficient_attention'):
                        self.sd15_pipeline.enable_xformers_memory_efficient_attention()
                        
                except Exception as gpu_error:
                    print(f"   ❌ Failed to move SD15 pipeline to GPU {self.gpu_id}: {gpu_error}")
                    raise gpu_error
            else:
                # Apply optimizations for CPU
                self.sd15_pipeline.enable_attention_slicing()
                self.sd15_pipeline.enable_vae_slicing()
            
            print(f"✅ SD1.5 pipeline loaded successfully on GPU {self.gpu_id}")
            
            # Small delay to ensure GPU stability
            time.sleep(1)
            
        except Exception as e:
            print(f"❌ SD1.5 pipeline loading failed on GPU {self.gpu_id}: {e}")
            traceback.print_exc()
            self._unload_sd15_pipeline()

    def _unload_sd15_pipeline(self):
        """Unload SD1.5 pipeline to free GPU memory"""
        if self.sd15_pipeline is not None:
            print(f"🧹 Unloading SD1.5 pipeline from GPU {self.gpu_id}...")
            del self.sd15_pipeline
            self.sd15_pipeline = None
            self._clear_gpu_memory()
            print(f"✅ SD1.5 pipeline unloaded from GPU {self.gpu_id}")

    def _load_hunyuan_pipeline(self):
        """Load HunyuanDiT pipeline"""
        if self.hunyuan_pipeline is not None:
            print(f"✓ HunyuanDiT pipeline already loaded on GPU {self.gpu_id}")
            return
            
        print(f"🔧 Loading HunyuanDiT pipeline on GPU {self.gpu_id}...")
        
        try:
            # Ensure we're on the correct GPU
            if torch.cuda.is_available():
                torch.cuda.set_device(self.gpu_id)
                torch.cuda.empty_cache()
            
            device = f"cuda:{self.gpu_id}" if torch.cuda.is_available() else "cpu"
            
            # Initialize HunyuanDiT pipeline
            self.hunyuan_pipeline = HunyuanDiTPipeline(
                model_path=GENERATION_CONFIG['hunyuan_model_path'],
                device=device
            )
            
            # Compile for better performance (only once)
            try:
                print(f"   Compiling HunyuanDiT for better performance on GPU {self.gpu_id}...")
                self.hunyuan_pipeline.compile()
                print(f"   ✓ HunyuanDiT compiled successfully on GPU {self.gpu_id}")
            except Exception as e:
                print(f"   ⚠️ HunyuanDiT compilation failed on GPU {self.gpu_id}: {e}")
                print("   Continuing without compilation...")
            
            print(f"✅ HunyuanDiT pipeline loaded successfully on GPU {self.gpu_id}")
            
            # Small delay to ensure GPU stability
            time.sleep(1)
            
        except Exception as e:
            print(f"❌ HunyuanDiT pipeline loading failed on GPU {self.gpu_id}: {e}")
            traceback.print_exc()
            self._unload_hunyuan_pipeline()
    
    def _unload_hunyuan_pipeline(self):
        """Unload HunyuanDiT pipeline to free GPU memory"""
        if self.hunyuan_pipeline is not None:
            print(f"🧹 Unloading HunyuanDiT pipeline from GPU {self.gpu_id}...")
            del self.hunyuan_pipeline
            self.hunyuan_pipeline = None
            self._clear_gpu_memory()
            print(f"✅ HunyuanDiT pipeline unloaded from GPU {self.gpu_id}")

    def _load_trellis_pipeline(self):
        """Load TRELLIS pipeline"""
        if self.trellis_pipeline is not None:
            print(f"✓ TRELLIS pipeline already loaded on GPU {self.gpu_id}")
            return
            
        print(f"🔧 Loading TRELLIS pipeline on GPU {self.gpu_id}...")
        
        try:
            # Ensure we're on the correct GPU
            if torch.cuda.is_available():
                torch.cuda.set_device(self.gpu_id)
                torch.cuda.empty_cache()
            
            self.trellis_pipeline = TrellisImageTo3DPipeline.from_pretrained(
                GENERATION_CONFIG['trellis_model_path']
            )
            
            # Move to specific GPU
            if torch.cuda.is_available():
                # Set device first, then move pipeline to GPU
                torch.cuda.set_device(self.gpu_id)
                self.trellis_pipeline = self.trellis_pipeline.cuda()
            
            # Warm up the pipeline
            try:
                self.trellis_pipeline.preprocess_image(
                    Image.fromarray(np.zeros((512, 512, 3), dtype=np.uint8))
                )
            except:
                pass
            
            print(f"✅ TRELLIS pipeline loaded successfully on GPU {self.gpu_id}")
            print(f"   Debug: Pipeline object type: {type(self.trellis_pipeline)}")
            print(f"   Debug: Pipeline object: {self.trellis_pipeline}")
            
            # Small delay to ensure GPU stability
            time.sleep(1)
            
            # Final verification that pipeline is set
            print(f"   Debug: After delay, pipeline is: {self.trellis_pipeline}")
            print(f"   Debug: Pipeline object ID: {id(self.trellis_pipeline)}")
            
        except Exception as e:
            print(f"❌ TRELLIS pipeline loading failed on GPU {self.gpu_id}: {e}")
            traceback.print_exc()
            self.trellis_pipeline = None

    def _unload_trellis_pipeline(self):
        """Unload TRELLIS pipeline to free GPU memory"""
        if self.trellis_pipeline is not None:
            print(f"🧹 Unloading TRELLIS pipeline from GPU {self.gpu_id}...")
            del self.trellis_pipeline
            self.trellis_pipeline = None
            self._clear_gpu_memory()
            print(f"✅ TRELLIS pipeline unloaded from GPU {self.gpu_id}")

    def _load_background_remover(self):
        """Load background removal model"""
        if self.background_remover is not None:
            print(f"✓ Background remover already loaded on GPU {self.gpu_id}")
            return
            
        print(f"🔧 Loading background remover on GPU {self.gpu_id}...")
        
        try:
            # Ensure we're on the correct GPU
            if torch.cuda.is_available():
                torch.cuda.set_device(self.gpu_id)
                torch.cuda.empty_cache()
            
            self.background_remover = BackgroundRemover()
            print(f"✅ Background remover loaded successfully on GPU {self.gpu_id}")
            
        except Exception as e:
            print(f"❌ Background remover loading failed on GPU {self.gpu_id}: {e}")
            traceback.print_exc()
            self.background_remover = None

    def _unload_background_remover(self):
        """Unload background remover to free GPU memory"""
        if self.background_remover is not None:
            print(f"🧹 Unloading background remover from GPU {self.gpu_id}...")
            del self.background_remover
            self.background_remover = None
            self._clear_gpu_memory()
            print(f"✅ Background remover unloaded from GPU {self.gpu_id}")

    def _load_lora(self, lora_key: str):
        """Load a specific LoRA onto the current pipeline"""
        current_model = GENERATION_CONFIG.get('current_model', 'flux')
        
        # Ensure the correct pipeline is loaded
        if current_model == 'flux' and self.flux_pipeline is None:
            print(f"🔧 Loading FLUX pipeline for LoRA on GPU {self.gpu_id}...")
            self._load_flux_models()
        elif current_model == 'sdxl' and self.sdxl_pipeline is None:
            print(f"🔧 Loading SDXL pipeline for LoRA on GPU {self.gpu_id}...")
            self._load_sdxl_pipeline()
        elif current_model == 'sd15' and self.sd15_pipeline is None:
            print(f"🔧 Loading SD1.5 pipeline for LoRA on GPU {self.gpu_id}...")
            self._load_sd15_pipeline()
        
        # Select pipeline and LoRA configs based on current model
        if current_model == 'flux':
            pipeline = self.flux_pipeline
            lora_configs = FLUX_LORAS
        elif current_model == 'sd15':
            pipeline = self.sd15_pipeline
            lora_configs = SD15_LORAS
        else:
            print(f"❌ Unknown model type: {current_model}")
            return False
        
        if pipeline is None:
            print(f"❌ {current_model.upper()} pipeline not loaded on GPU {self.gpu_id}. Load {current_model.upper()} models first.")
            return False
            
        if lora_key not in lora_configs:
            print(f"❌ LoRA '{lora_key}' not found in available {current_model.upper()} LoRAs")
            return False
            
        lora_config = lora_configs[lora_key]
        print(f"🔧 Loading {current_model.upper()} LoRA on GPU {self.gpu_id}: {lora_config['name']}")
        
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
                        print(f"⚠️ LoRA needs patching, applying adaLN fix on GPU {self.gpu_id}...")
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
                        print(f"✅ LoRA patched and loaded successfully on GPU {self.gpu_id}")
                    else:
                        raise e
            
            # Apply fusion if specified
            if lora_config.get('fuse', False):
                try:
                    pipeline.fuse_lora(lora_scale=lora_config.get('scale', 1.0))
                    print(f"   🔗 LoRA fused with scale {lora_config.get('scale', 1.0)} on GPU {self.gpu_id}")
                except Exception as fusion_error:
                    print(f"   ⚠️ LoRA fusion failed on GPU {self.gpu_id}: {fusion_error}")
                    print(f"   📝 Continuing without fusion...")
            
            # Update current LoRA configuration
            GENERATION_CONFIG['current_lora'] = lora_key
            GENERATION_CONFIG['lora_scale'] = lora_config.get('scale', 1.0)
            
            print(f"✅ {current_model.upper()} LoRA '{lora_config['name']}' loaded successfully on GPU {self.gpu_id}")
            return True
            
        except Exception as e:
            print(f"❌ Failed to load {current_model.upper()} LoRA '{lora_key}' on GPU {self.gpu_id}: {e}")
            traceback.print_exc()
            return False

    def _unload_lora(self):
        """Unload current LoRA from current pipeline"""
        current_model = GENERATION_CONFIG.get('current_model', 'flux')
        
        # Determine which pipeline to use
        if current_model == 'flux':
            pipeline = self.flux_pipeline
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
                print(f"✅ {current_model.upper()} LoRA unloaded successfully from GPU {self.gpu_id}")
            
            GENERATION_CONFIG['current_lora'] = None
            GENERATION_CONFIG['lora_scale'] = 1.0
            
        except Exception as e:
            print(f"⚠️ Error unloading {current_model.upper()} LoRA from GPU {self.gpu_id}: {e}")

    def get_available_loras(self) -> Dict[str, Any]:
        """Get list of available LoRAs for current model"""
        current_model = GENERATION_CONFIG.get('current_model', 'flux')
        
        if current_model == 'flux':
            lora_configs = FLUX_LORAS
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
                'loaded': GENERATION_CONFIG['current_model'] == current_model and GENERATION_CONFIG['current_lora'] == key
            }
            for key, config in lora_configs.items()
        }

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
                print("🔧 HunyuanDiT pipeline not loaded, loading now...")
                self._load_hunyuan_pipeline()
            
            # Verify pipeline is loaded
            if self.hunyuan_pipeline is None:
                raise RuntimeError("Failed to load HunyuanDiT pipeline")
            
            # Ensure we're on the correct GPU
            if torch.cuda.is_available():
                torch.cuda.set_device(self.gpu_id)
            
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

    def generate_3d_model(self, prompt: str, seed: int = 42) -> Optional[Tuple[bytes, Optional[bytes]]]:
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
                print(f"🎯 Starting TRELLIS generation for: '{prompt}' (seed: {seed}) on GPU {self.gpu_id}")
                
                # Ensure we're on the correct GPU
                if torch.cuda.is_available():
                    torch.cuda.set_device(self.gpu_id)
                    print(f"   Set CUDA device to GPU {self.gpu_id}")
                
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
                        print("🔧 FLUX pipeline not loaded, loading now...")
                        self._load_flux_models()
                    
                    # Verify pipeline is loaded
                    if self.flux_pipeline is None:
                        raise RuntimeError("Failed to load FLUX pipeline")
                    
                    if current_lora and current_lora in FLUX_LORAS:
                        lora_config = FLUX_LORAS[current_lora]
                        trigger_prefix = lora_config.get('trigger_prefix', '')
                        if trigger_prefix:
                            enhanced_prompt = f"{trigger_prefix} {prompt}"
                            print(f"🎨 Applied FLUX LoRA trigger prefix: '{trigger_prefix}'")
                    
                    # Ensure we're on the correct GPU
                    if torch.cuda.is_available():
                        torch.cuda.set_device(self.gpu_id)
                    
                    generator = torch.Generator(device=device).manual_seed(seed)
                    with torch.no_grad():
                        image = self.flux_pipeline(
                            prompt=enhanced_prompt,
                            guidance_scale=GENERATION_CONFIG['guidance_scale'],
                            num_inference_steps=NUM_INFERENCE_STEPS,
                            width=1024,
                            height=1024,
                            generator=generator,
                        ).images[0]
                
                elif current_model == 'sdxl':
                    if self.sdxl_pipeline is None:
                        print("🔧 SDXL pipeline not loaded, loading now...")
                        self._load_sdxl_pipeline()
                    
                    # Verify pipeline is loaded
                    if self.sdxl_pipeline is None:
                        raise RuntimeError("Failed to load SDXL pipeline")
                    
                    if current_lora and current_lora in SDXL_LORAS:
                        lora_config = SDXL_LORAS[current_lora]
                        trigger_prefix = lora_config.get('trigger_prefix', '')
                        if trigger_prefix:
                            enhanced_prompt = f"{trigger_prefix} {prompt}"
                            print(f"🎨 Applied SDXL LoRA trigger prefix: '{trigger_prefix}'")
                    
                    # Ensure we're on the correct GPU
                    if torch.cuda.is_available():
                        torch.cuda.set_device(self.gpu_id)
                    
                    generator = torch.Generator(device=device).manual_seed(seed)
                    with torch.no_grad():
                        image = self.sdxl_pipeline(
                            prompt=enhanced_prompt,
                            guidance_scale=7.5,
                            num_inference_steps=25,
                            width=1024,
                            height=1024,
                            generator=generator,
                        ).images[0]
                
                elif current_model == 'sd15':
                    if self.sd15_pipeline is None:
                        print("🔧 SD15 pipeline not loaded, loading now...")
                        self._load_sd15_pipeline()
                    
                    # Verify pipeline is loaded
                    if self.sd15_pipeline is None:
                        raise RuntimeError("Failed to load SD15 pipeline")
                    
                    if current_lora and current_lora in SD15_LORAS:
                        lora_config = SD15_LORAS[current_lora]
                        trigger_prefix = lora_config.get('trigger_prefix', '')
                        if trigger_prefix:
                            enhanced_prompt = f"{trigger_prefix} {prompt}"
                            print(f"🎨 Applied SD1.5 LoRA trigger prefix: '{trigger_prefix}'")
                    
                    # Ensure we're on the correct GPU
                    if torch.cuda.is_available():
                        torch.cuda.set_device(self.gpu_id)
                    
                    generator = torch.Generator(device=device).manual_seed(seed)
                    with torch.no_grad():
                        image = self.sd15_pipeline(
                            prompt=enhanced_prompt,
                            guidance_scale=7.5,
                            num_inference_steps=25,
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
                
                # Step 1.3: Center object in image before background removal
                if GENERATION_CONFIG.get('enable_object_centering', True):
                    print("Step 1.3: Centering object in image...")
                    try:
                        centered_image = self.center_object_in_image(
                            image, 
                            white_threshold=GENERATION_CONFIG.get('centering_white_threshold', 240),
                            padding=GENERATION_CONFIG.get('centering_padding', 40)
                        )
                        print("✓ Object centered successfully")
                        image = centered_image  # Use the centered image for next steps
                        generation_asset.add_asset(AssetType.FLUX_IMAGE, centered_image)  # Update asset with centered version
                    except Exception as e:
                        print(f"⚠️ Object centering failed: {e}")
                        print("   Continuing with original image...")
                else:
                    print("Step 1.3: Object centering disabled, skipping...")
                
                # Step 1.5: Remove background from image
                # print("Step 1.5: Removing background from image...")
                # self._load_background_remover()
                
                # try:
                #     image_no_bg = self.background_remover(image)
                #     print("✓ Background removed successfully")
                #     # Save the background-removed image as well
                #     generation_asset.add_asset(AssetType.FLUX_IMAGE, image_no_bg)  # Replace original with cleaned version
                #     image = image_no_bg  # Use the cleaned image for TRELLIS
                # except Exception as e:
                #     print(f"⚠️ Background removal failed: {e}")
                #     print("   Continuing with original image...")
                
                # # Unload background remover
                # self._unload_background_remover()
                
                # Step 2: Generate 3D model with TRELLIS
                print("Step 2: Generating 3D model with TRELLIS...")
                print(f"   Debug: Before loading, self.trellis_pipeline = {self.trellis_pipeline}")
                print(f"   Debug: Thread ID: {threading.current_thread().ident}")
                
                # Check memory without clearing it
                if torch.cuda.is_available():
                    gpu_memory_free, gpu_memory_total = torch.cuda.mem_get_info(self.gpu_id)
                    memory_used = gpu_memory_total - gpu_memory_free
                    print(f"   Debug: Memory before loading: {memory_used / 1e9:.1f}GB used, {gpu_memory_free / 1e9:.1f}GB free")
                
                if self.trellis_pipeline is None:   
                    print("🔧 TRELLIS pipeline not loaded, loading now...")
                    with self.generation_lock:  # Ensure thread safety during loading
                        if self.trellis_pipeline is None:  # Double-check after acquiring lock
                            self._load_trellis_pipeline()
                            print(f"   Debug: After loading, self.trellis_pipeline = {self.trellis_pipeline}")
                
                # Verify TRELLIS pipeline is loaded with debugging
                print(f"   Debug: Final check - self.trellis_pipeline = {self.trellis_pipeline}")
                print(f"   Debug: Pipeline type = {type(self.trellis_pipeline)}")
                
                # Check memory after loading without clearing it
                if torch.cuda.is_available():
                    gpu_memory_free, gpu_memory_total = torch.cuda.mem_get_info(self.gpu_id)
                    memory_used = gpu_memory_total - gpu_memory_free
                    print(f"   Debug: Memory after loading: {memory_used / 1e9:.1f}GB used, {gpu_memory_free / 1e9:.1f}GB free")
                
                if self.trellis_pipeline is None:
                    print(f"   ❌ TRELLIS pipeline is still None after loading attempt")
                    print(f"   Debug: This suggests the pipeline was loaded but then set to None")
                    raise RuntimeError("Failed to load TRELLIS pipeline")
                
                print(f"   ✅ TRELLIS pipeline verified and ready on GPU {self.gpu_id}")
                
                # Ensure we're on the correct GPU
                if torch.cuda.is_available():
                    torch.cuda.set_device(self.gpu_id)
                    print(f"   Set CUDA device to GPU {self.gpu_id}")
                
                # Enhanced TRELLIS parameters for maximum quality
                print(f"   Starting TRELLIS generation with image size: {image.size}")
                print(f"   Debug: About to call trellis_pipeline.run()")
                print(f"   Debug: Pipeline object before run: {self.trellis_pipeline}")
                
                # Double-check pipeline is still valid
                if self.trellis_pipeline is None:
                    print(f"   ❌ TRELLIS pipeline became None before run() call!")
                    print(f"   Debug: This suggests the pipeline was unloaded between loading and run()")
                    raise RuntimeError("TRELLIS pipeline was unloaded before generation")
                
                print(f"   Debug: Pipeline object ID before run: {id(self.trellis_pipeline)}")
                outputs = self.trellis_pipeline.run(
                    image,
                    seed=seed,
                    formats=["gaussian", "mesh"],
                    preprocess_image=False,
                    sparse_structure_sampler_params={
                        "steps": GENERATION_CONFIG['ss_sampling_steps'],
                        "cfg_strength": GENERATION_CONFIG['ss_guidance_strength'],
                        "cfg_interval": (0.3, 0.98),  # Enhanced guidance scheduling
                        "rescale_t": 3.0,  # Temperature rescaling for better quality
                    },
                    slat_sampler_params={
                        "steps": GENERATION_CONFIG['slat_sampling_steps'],
                        "cfg_strength": GENERATION_CONFIG['slat_guidance_strength'],
                        "cfg_interval": (0.3, 0.98),  # Enhanced guidance scheduling
                        "rescale_t": 3.0,  # Temperature rescaling for better quality
                    },
                )
                
                print("✓ 3D model generated successfully")
                print(f"   Debug: Pipeline object ID after run: {id(self.trellis_pipeline)}")
                
                # Check memory after generation without clearing it
                if torch.cuda.is_available():
                    gpu_memory_free, gpu_memory_total = torch.cuda.mem_get_info(self.gpu_id)
                    memory_used = gpu_memory_total - gpu_memory_free
                    print(f"   Debug: Memory after generation: {memory_used / 1e9:.1f}GB used, {gpu_memory_free / 1e9:.1f}GB free")
                
                # Step 3: Extract and enhance Gaussian Splatting PLY
                print("Step 3: Extracting and enhancing Gaussian Splatting PLY...")
                gaussian_output = outputs['gaussian'][0]
                
                # Quality enhancement: Filter low-quality splats
                print("   Enhancing quality by filtering low-quality splats...")
                try:
                    # Get splat data
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
                        
                        print(f"   Quality enhancement: Kept {quality_mask.sum().item():,} high-quality splats out of {len(points):,}")
                    else:
                        print(f"   Quality enhancement skipped: Too few splats would remain ({quality_mask.sum().item()})")
                        
                except Exception as e:
                    print(f"   Quality enhancement failed: {e}")
                    print("   Continuing with original splats...")
                
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

# Multi-GPU generator instances
gpu_generators = {}
generation_jobs = {}
gpu_initialization_lock = threading.Lock()  # Prevent simultaneous GPU initialization

def get_or_create_generator(gpu_id: int) -> TrellisGenerator:
    """Get or create a generator instance for a specific GPU"""
    if gpu_id not in gpu_generators:
        with gpu_initialization_lock:  # Ensure only one GPU initializes at a time
            if gpu_id not in gpu_generators:  # Double-check after acquiring lock
                try:
                    print(f"🔧 Creating new generator instance for GPU {gpu_id}")
                    gpu_generators[gpu_id] = TrellisGenerator(gpu_id)
                    
                    # Wait a moment for GPU to stabilize
                    time.sleep(2)
                    
                except Exception as e:
                    print(f"❌ Failed to create generator for GPU {gpu_id}: {e}")
                    # Remove failed generator
                    if gpu_id in gpu_generators:
                        del gpu_generators[gpu_id]
                    raise e
    
    return gpu_generators[gpu_id]

def cleanup_gpu_generators():
    """Clean up GPU generators to free memory"""
    for gpu_id, generator in gpu_generators.items():
        try:
            generator._clear_gpu_memory()
            del generator
        except:
            pass
    gpu_generators.clear()

# Initialize default generator on GPU 0
default_generator = get_or_create_generator(0)

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
    gpu_id: Optional[int] = Form(0)
):
    """Generate 3D model from text prompt using FLUX + TRELLIS pipeline on specified GPU."""
    
    # Validate GPU ID
    if gpu_id < 0 or gpu_id >= gpu_manager.num_gpus:
        raise HTTPException(status_code=400, detail=f"Invalid GPU ID. Must be 0-{gpu_manager.num_gpus-1}")
    
    # Check if GPU is available
    if not gpu_manager.reserve_gpu(gpu_id, f"gen_{int(time.time())}_{seed or 42}"):
        raise HTTPException(status_code=503, detail=f"GPU {gpu_id} is currently busy")
    
    try:
        # Handle seed
        if seed is None:
            seed = 42
        
        # Get or create generator for this GPU
        generator = get_or_create_generator(gpu_id)
        
        # Generate model
        result = generator.generate_3d_model(prompt, seed)
        
        if result is None:
            raise HTTPException(status_code=500, detail="Generation failed")
        
        ply_data, compressed_data = result
        
        # Apply SPZ compression if requested
        if return_compressed:
            try:
                if compressed_data is None:
                    import pyspz
                    compressed_data = pyspz.compress(ply_data, workers=-1)
                    print(f"🗜️ SPZ Compression for response (GPU {gpu_id}):")
                    print(f"   Original: {len(ply_data):,} bytes ({len(ply_data)/1024/1024:.1f} MB)")
                    print(f"   Compressed: {len(compressed_data):,} bytes ({len(compressed_data)/1024/1024:.1f} MB)") 
                    print(f"   Ratio: {len(compressed_data)/len(ply_data)*100:.1f}%")
                
                return Response(
                    content=compressed_data,
                    media_type="application/octet-stream",
                    headers={
                        "Content-Disposition": f"attachment; filename=trellis_model_gpu{gpu_id}_{seed}.ply.spz",
                        "X-Generation-Seed": str(seed),
                        "X-Generation-Prompt": prompt,
                        "X-Model-Format": "gaussian_splatting_ply",
                        "X-Pipeline": "flux_trellis",
                        "X-GPU-ID": str(gpu_id),
                        "X-Compression": "spz",
                        "X-Compression-Ratio": f"{len(compressed_data)/len(ply_data)*100:.1f}%"
                    }
                )
            except Exception as e:
                print(f"⚠️ SPZ compression failed (GPU {gpu_id}): {e}")
                # Fall back to uncompressed
        
        # Return uncompressed PLY data
        return Response(
            content=ply_data,
            media_type="application/octet-stream",
            headers={
                "Content-Disposition": f"attachment; filename=trellis_model_gpu{gpu_id}_{seed}.ply",
                "X-Generation-Seed": str(seed),
                "X-Generation-Prompt": prompt,
                "X-Model-Format": "gaussian_splatting_ply",
                "X-Pipeline": "flux_trellis",
                "X-GPU-ID": str(gpu_id),
                "X-Compression": "none"
            }
        )
    
    finally:
        # Always release the GPU
        gpu_manager.release_gpu(gpu_id)

@app.post("/generate_parallel/")
async def generate_parallel_3d_models(
    prompt: str = Form(...),
    seeds: str = Form(...),  # Comma-separated seeds
    return_compressed: Optional[bool] = Form(True),
    max_parallel: Optional[int] = Form(8)
):
    """Generate multiple 3D models in parallel across available GPUs."""
    
    try:
        # Parse seeds
        seed_list = [int(s.strip()) for s in seeds.split(',')]
        num_models = len(seed_list)
        
        # Limit parallel generations
        max_parallel = min(max_parallel, gpu_manager.get_parallel_generation_capacity())
        if num_models > max_parallel:
            raise HTTPException(
                status_code=400, 
                detail=f"Too many models requested ({num_models}). Max parallel: {max_parallel}"
            )
        
        print(f"🚀 Starting parallel generation of {num_models} models on {max_parallel} GPUs")
        
        # Create generation jobs
        jobs = []
        for i, seed in enumerate(seed_list):
            job = {
                'id': f"parallel_gen_{int(time.time())}_{i}_{seed}",
                'prompt': prompt,
                'seed': seed,
                'status': 'pending',
                'start_time': None,
                'end_time': None,
                'result': None,
                'error': None,
                'gpu_id': None
            }
            jobs.append(job)
            generation_jobs[job['id']] = job
        
        # Execute jobs in parallel using ThreadPoolExecutor
        results = []
        
        def execute_generation(job):
            try:
                # Find available GPU
                gpu_id = gpu_manager.get_available_gpu()
                if gpu_id is None:
                    job['status'] = 'failed'
                    job['error'] = 'No GPU available'
                    return job
                
                # Reserve GPU
                if not gpu_manager.reserve_gpu(gpu_id, job['id']):
                    job['status'] = 'failed'
                    job['error'] = f'Failed to reserve GPU {gpu_id}'
                    return job
                
                job['gpu_id'] = gpu_id
                job['status'] = 'running'
                job['start_time'] = time.time()
                
                print(f"🎯 Starting generation {job['id']} on GPU {gpu_id}")
                
                # Get generator for this GPU with retry logic
                max_retries = 3
                generator = None
                
                for attempt in range(max_retries):
                    try:
                        generator = get_or_create_generator(gpu_id)
                        break
                    except Exception as e:
                        print(f"⚠️ Attempt {attempt + 1} failed to create generator for GPU {gpu_id}: {e}")
                        if attempt < max_retries - 1:
                            time.sleep(2)  # Wait before retry
                            # Try to reset GPU
                            try:
                                gpu_manager.release_gpu(gpu_id)
                                gpu_manager.gpu_states[gpu_id]['status'] = 'idle'
                                time.sleep(1)
                            except:
                                pass
                        else:
                            raise e
                
                if generator is None:
                    job['status'] = 'failed'
                    job['error'] = f'Failed to create generator for GPU {gpu_id} after {max_retries} attempts'
                    return job
                
                # Generate model
                result = generator.generate_3d_model(job['prompt'], job['seed'])
                
                if result is None:
                    job['status'] = 'failed'
                    job['error'] = 'Generation failed'
                else:
                    job['status'] = 'completed'
                    job['result'] = result
                
                job['end_time'] = time.time()
                
            except Exception as e:
                job['status'] = 'failed'
                job['error'] = str(e)
                job['end_time'] = time.time()
                print(f"❌ Generation {job['id']} failed: {e}")
            finally:
                # Release GPU
                if job['gpu_id'] is not None:
                    gpu_manager.release_gpu(job['gpu_id'])
            
            return job
        
        # Execute jobs in parallel
        with ThreadPoolExecutor(max_workers=max_parallel) as executor:
            future_to_job = {executor.submit(execute_generation, job): job for job in jobs}
            
            for future in as_completed(future_to_job):
                job = future.result()
                results.append(job)
                
                if job['status'] == 'completed':
                    print(f"✅ Generation {job['id']} completed on GPU {job['gpu_id']}")
                else:
                    print(f"❌ Generation {job['id']} failed: {job['error']}")
        
        # Prepare response
        successful_generations = [j for j in results if j['status'] == 'completed']
        failed_generations = [j for j in results if j['status'] == 'failed']
        
        # Calculate total time safely
        total_time = 0.0
        if results:
            valid_times = [j['end_time'] - j['start_time'] for j in results if j['start_time'] and j['end_time']]
            if valid_times:
                total_time = max(valid_times)
        
        response_data = {
            "status": "completed",
            "total_models": num_models,
            "successful": len(successful_generations),
            "failed": len(failed_generations),
            "success_rate": len(successful_generations) / num_models * 100,
            "total_time": total_time,
            "jobs": results
        }
        
        # Add compressed data if requested
        if return_compressed and successful_generations:
            compressed_models = []
            for job in successful_generations:
                ply_data, compressed_data = job['result']
                if compressed_data:
                    compressed_models.append({
                        'seed': job['seed'],
                        'gpu_id': job['gpu_id'],
                        'compressed_data': base64.b64encode(compressed_data).decode('utf-8'),
                        'size_bytes': len(compressed_data)
                    })
            
            response_data['compressed_models'] = compressed_models
        
        return JSONResponse(content=response_data)
        
    except Exception as e:
        print(f"❌ Parallel generation failed: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Parallel generation failed: {str(e)}")

@app.get("/parallel_jobs/")
async def get_parallel_jobs():
    """Get status of parallel generation jobs"""
    return {
        "active_jobs": len([j for j in generation_jobs.values() if j['status'] in ['pending', 'running']]),
        "completed_jobs": len([j for j in generation_jobs.values() if j['status'] == 'completed']),
        "failed_jobs": len([j for j in generation_jobs.values() if j['status'] == 'failed']),
        "jobs": generation_jobs
    }

@app.get("/gpu_status/")
async def get_gpu_status():
    """Get comprehensive GPU status and utilization"""
    return gpu_manager.get_gpu_status()

@app.post("/gpu_reset/{gpu_id}")
async def reset_gpu(gpu_id: int):
    """Reset a specific GPU (useful for recovery)"""
    if gpu_id < 0 or gpu_id >= gpu_manager.num_gpus:
        raise HTTPException(status_code=400, detail=f"Invalid GPU ID. Must be 0-{gpu_manager.num_gpus-1}")
    
    try:
        # Release GPU if busy
        gpu_manager.release_gpu(gpu_id)
        
        # Clear GPU memory
        if gpu_id in gpu_generators:
            gpu_generators[gpu_id]._clear_gpu_memory()
        
        # Reset GPU state
        gpu_manager.gpu_states[gpu_id]['status'] = 'idle'
        gpu_manager.gpu_states[gpu_id]['current_job'] = None
        gpu_manager.gpu_loads[gpu_id] = 0
        
        return {
            "status": "success",
            "message": f"GPU {gpu_id} reset successfully",
            "gpu_id": gpu_id
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to reset GPU {gpu_id}: {str(e)}")

@app.post("/gpu_cleanup/")
async def cleanup_all_gpus():
    """Clean up all GPUs and free memory"""
    try:
        cleanup_gpu_generators()
        return {
            "status": "success",
            "message": "All GPUs cleaned up successfully"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to cleanup GPUs: {str(e)}")

@app.get("/gpu_health/")
async def get_gpu_health():
    """Get detailed GPU health information"""
    try:
        health_info = {}
        
        for gpu_id in range(gpu_manager.num_gpus):
            try:
                if torch.cuda.is_available():
                    torch.cuda.set_device(gpu_id)
                    
                    # Get memory info
                    memory_allocated = torch.cuda.memory_allocated(gpu_id)
                    memory_reserved = torch.cuda.memory_reserved(gpu_id)
                    memory_total = torch.cuda.get_device_properties(gpu_id).total_memory
                    
                    # Get GPU properties
                    props = torch.cuda.get_device_properties(gpu_id)
                    
                    health_info[gpu_id] = {
                        "status": gpu_manager.gpu_states[gpu_id]['status'],
                        "memory_allocated_gb": memory_allocated / 1e9,
                        "memory_reserved_gb": memory_reserved / 1e9,
                        "memory_total_gb": memory_total / 1e9,
                        "memory_free_gb": (memory_total - memory_allocated) / 1e9,
                        "gpu_name": props.name,
                        "compute_capability": f"{props.major}.{props.minor}",
                        "multiprocessor_count": props.multi_processor_count,
                        "current_job": gpu_manager.gpu_states[gpu_id]['current_job']
                    }
                    
                    # Try to get nvidia-smi info
                    try:
                        result = subprocess.run(
                            ['nvidia-smi', '--query-gpu=utilization.gpu,temperature.gpu,power.draw', 
                             '--format=csv,noheader,nounits', '-i', str(gpu_id)],
                            capture_output=True, text=True
                        )
                        if result.returncode == 0:
                            lines = result.stdout.strip().split('\n')
                            if lines and len(lines) > 0:
                                parts = lines[0].split(', ')
                                if len(parts) >= 3:
                                    health_info[gpu_id].update({
                                        "utilization_percent": float(parts[0]),
                                        "temperature_celsius": float(parts[1]),
                                        "power_draw_watts": float(parts[2])
                                    })
                    except:
                        pass
                        
            except Exception as e:
                health_info[gpu_id] = {
                    "status": "error",
                    "error": str(e)
                }
        
        return {
            "status": "success",
            "gpu_health": health_info,
            "timestamp": time.time()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get GPU health: {str(e)}")

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
    # Get status from default generator (GPU 0)
    status = default_generator.get_status()
    
    # Add multi-GPU information
    status.update({
        "multi_gpu": {
            "enabled": True,
            "num_gpus": gpu_manager.num_gpus,
            "gpu_status": gpu_manager.get_gpu_status(),
            "parallel_capacity": gpu_manager.get_parallel_generation_capacity(),
            "active_generators": len(gpu_generators)
        }
    })
    
    return status

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
    gpu_id: Optional[int] = Form(0)
):
    """Generate 3D model using Isometric 3D LoRA on specified GPU"""
    try:
        # Validate GPU ID
        if gpu_id < 0 or gpu_id >= gpu_manager.num_gpus:
            raise HTTPException(status_code=400, detail=f"Invalid GPU ID. Must be 0-{gpu_manager.num_gpus-1}")
        
        # Check if GPU is available
        if not gpu_manager.reserve_gpu(gpu_id, f"isometric_3d_{int(time.time())}_{seed or 42}"):
            raise HTTPException(status_code=503, detail=f"GPU {gpu_id} is currently busy")
        
        try:
            # Get or create generator for this GPU
            generator = get_or_create_generator(gpu_id)
            
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
                print(f"🎨 Applied FLUX LoRA trigger prefix: '{trigger_prefix}' (GPU {gpu_id})")
            
            # Generate with the LoRA
            result = generator.generate_3d_model(enhanced_prompt, seed or 42)
            if result is None:
                raise HTTPException(status_code=500, detail="Generation failed")
            
            ply_data, compressed_data = result
            
            # Return compressed data if requested
            if return_compressed and compressed_data:
                return Response(
                    content=compressed_data,
                    media_type="application/octet-stream",
                    headers={
                        "Content-Disposition": f"attachment; filename=isometric_3d_gpu{gpu_id}_{seed or 42}.ply.spz",
                        "X-Generation-Seed": str(seed or 42),
                        "X-Generation-Prompt": prompt,
                        "X-Model-Format": "gaussian_splatting_ply",
                        "X-Pipeline": "flux_trellis",
                        "X-LoRA": "isometric_3d",
                        "X-GPU-ID": str(gpu_id),
                        "X-Compression": "spz"
                    }
                )
            
            # Return uncompressed PLY data
            return Response(
                content=ply_data,
                media_type="application/octet-stream",
                headers={
                    "Content-Disposition": f"attachment; filename=isometric_3d_gpu{gpu_id}_{seed or 42}.ply",
                    "X-Generation-Seed": str(seed or 42),
                    "X-Generation-Prompt": prompt,
                    "X-Model-Format": "gaussian_splatting_ply",
                    "X-Pipeline": "flux_trellis",
                    "X-LoRA": "isometric_3d",
                    "X-GPU-ID": str(gpu_id),
                    "X-Compression": "none"
                }
            )
        finally:
            # Always release the GPU
            gpu_manager.release_gpu(gpu_id)
            
    except Exception as e:
        return JSONResponse(content={
            "status": "error",
            "message": str(e)
        }, status_code=500)

@app.post("/generate/live_3d/")
async def generate_with_live_3d_lora(
    prompt: str = Form(...), 
    seed: Optional[int] = Form(None),
    return_compressed: Optional[bool] = Form(True)
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
        
        result = generator.generate_3d_model(enhanced_prompt, seed or 42)
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
    return_compressed: Optional[bool] = Form(True)
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
        
        result = generator.generate_3d_model(enhanced_prompt, seed or 42)
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
    return_compressed: Optional[bool] = Form(True)
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
        
        result = generator.generate_3d_model(enhanced_prompt, seed or 42)
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
    return_compressed: Optional[bool] = Form(True)
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
        
        result = generator.generate_3d_model(enhanced_prompt, seed or 42)
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
    return_compressed: Optional[bool] = Form(True)
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
        
        result = generator.generate_3d_model(enhanced_prompt, seed or 42)
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
    return_compressed: Optional[bool] = Form(True)
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
        
        result = generator.generate_3d_model(enhanced_prompt, seed or 42)
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
    return_compressed: Optional[bool] = Form(True)
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
        
        result = generator.generate_3d_model(enhanced_prompt, seed or 42)
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
    return_compressed: Optional[bool] = Form(True)
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
        
        result = generator.generate_3d_model(enhanced_prompt, seed or 42)
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
    gpu_id: Optional[int] = Form(0)
):
    """Generate 3D model using FLUX with Necklace LoRA on specified GPU"""
    try:
        # Validate GPU ID
        if gpu_id < 0 or gpu_id >= gpu_manager.num_gpus:
            raise HTTPException(status_code=400, detail=f"Invalid GPU ID. Must be 0-{gpu_manager.num_gpus-1}")
        
        # Check if GPU is available
        if not gpu_manager.reserve_gpu(gpu_id, f"necklace_{int(time.time())}_{seed or 42}"):
            raise HTTPException(status_code=503, detail=f"GPU {gpu_id} is currently busy")
        
        try:
            # Switch to FLUX model first
            GENERATION_CONFIG['current_model'] = 'flux'
            
            # Get or create generator for this GPU
            generator = get_or_create_generator(gpu_id)
            
            success = generator._load_lora('necklace')
            if not success:
                raise HTTPException(status_code=500, detail="Failed to load Necklace LoRA")
            
            # Apply LoRA trigger prefix
            lora_config = FLUX_LORAS['necklace']
            trigger_prefix = lora_config.get('trigger_prefix', '')
            enhanced_prompt = prompt
            if trigger_prefix:
                enhanced_prompt = f"{trigger_prefix} {prompt}"
                print(f"🎨 Applied FLUX LoRA trigger prefix: '{trigger_prefix}' (GPU {gpu_id})")
            
            result = generator.generate_3d_model(enhanced_prompt, seed or 42)
            if result is None:
                raise HTTPException(status_code=500, detail="Generation failed")
            
            ply_data, compressed_data = result
            
            if return_compressed and compressed_data:
                return Response(
                    content=compressed_data,
                    media_type="application/octet-stream",
                    headers={
                        "Content-Disposition": f"attachment; filename=necklace_gpu{gpu_id}_{seed or 42}.ply.spz",
                        "X-Generation-Seed": str(seed or 42),
                        "X-Generation-Prompt": prompt,
                        "X-Model-Format": "gaussian_splatting_ply",
                        "X-Pipeline": "flux_trellis",
                        "X-LoRA": "necklace",
                        "X-GPU-ID": str(gpu_id),
                        "X-Compression": "spz"
                    }
                )
            
            return Response(
                content=ply_data,
                media_type="application/octet-stream",
                headers={
                    "Content-Disposition": f"attachment; filename=necklace_gpu{gpu_id}_{seed or 42}.ply",
                    "X-Generation-Seed": str(seed or 42),
                    "X-Generation-Prompt": prompt,
                    "X-Model-Format": "gaussian_splatting_ply",
                    "X-Pipeline": "flux_trellis",
                    "X-LoRA": "necklace",
                    "X-GPU-ID": str(gpu_id),
                    "X-Compression": "none"
                }
            )
        finally:
            # Always release the GPU
            gpu_manager.release_gpu(gpu_id)
            
    except Exception as e:
        return JSONResponse(content={
            "status": "error",
            "message": str(e)
        }, status_code=500)

@app.post("/generate_parallel_lora/{lora_style}/")
async def generate_parallel_with_lora(
    lora_style: str,
    prompt: str = Form(...),
    seeds: str = Form(...),  # Comma-separated seeds
    return_compressed: Optional[bool] = Form(True),
    max_parallel: Optional[int] = Form(8)
):
    """Generate multiple 3D models in parallel using a specific LoRA style across available GPUs."""
    
    try:
        # Validate LoRA style
        if lora_style not in FLUX_LORAS and lora_style not in SD15_LORAS:
            raise HTTPException(status_code=400, detail=f"Invalid LoRA style: {lora_style}")
        
        # Parse seeds
        seed_list = [int(s.strip()) for s in seeds.split(',')]
        num_models = len(seed_list)
        
        # Limit parallel generations
        max_parallel = min(max_parallel, gpu_manager.get_parallel_generation_capacity())
        if num_models > max_parallel:
            raise HTTPException(
                status_code=400, 
                detail=f"Too many models requested ({num_models}). Max parallel: {max_parallel}"
            )
        
        print(f"🚀 Starting parallel generation of {num_models} models with {lora_style} LoRA on {max_parallel} GPUs")
        
        # Create generation jobs
        jobs = []
        for i, seed in enumerate(seed_list):
            job = {
                'id': f"parallel_lora_{lora_style}_{int(time.time())}_{i}_{seed}",
                'prompt': prompt,
                'seed': seed,
                'lora_style': lora_style,
                'status': 'pending',
                'start_time': None,
                'end_time': None,
                'result': None,
                'error': None,
                'gpu_id': None
            }
            jobs.append(job)
            generation_jobs[job['id']] = job
        
        # Execute jobs in parallel using ThreadPoolExecutor
        results = []
        
        def execute_lora_generation(job):
            try:
                # Find available GPU
                gpu_id = gpu_manager.get_available_gpu()
                if gpu_id is None:
                    job['status'] = 'failed'
                    job['error'] = 'No GPU available'
                    return job
                
                # Reserve GPU
                if not gpu_manager.reserve_gpu(gpu_id, job['id']):
                    job['status'] = 'failed'
                    job['error'] = f'Failed to reserve GPU {gpu_id}'
                    return job
                
                job['gpu_id'] = gpu_id
                job['status'] = 'running'
                job['start_time'] = time.time()
                
                print(f"🎯 Starting {job['lora_style']} LoRA generation {job['id']} on GPU {gpu_id}")
                
                # Get generator for this GPU with retry logic
                max_retries = 3
                generator = None
                
                for attempt in range(max_retries):
                    try:
                        generator = get_or_create_generator(gpu_id)
                        break
                    except Exception as e:
                        print(f"⚠️ Attempt {attempt + 1} failed to create generator for GPU {gpu_id}: {e}")
                        if attempt < max_retries - 1:
                            time.sleep(2)  # Wait before retry
                            # Try to reset GPU
                            try:
                                gpu_manager.release_gpu(gpu_id)
                                gpu_manager.gpu_states[gpu_id]['status'] = 'idle'
                                time.sleep(1)
                            except:
                                pass
                        else:
                            raise e
                
                if generator is None:
                    job['status'] = 'failed'
                    job['error'] = f'Failed to create generator for GPU {gpu_id} after {max_retries} attempts'
                    return job
                
                # Load LoRA
                success = generator._load_lora(job['lora_style'])
                if not success:
                    job['status'] = 'failed'
                    job['error'] = f'Failed to load {job["lora_style"]} LoRA'
                    return job
                
                # Apply LoRA trigger prefix
                enhanced_prompt = job['prompt']
                if job['lora_style'] in FLUX_LORAS:
                    lora_config = FLUX_LORAS[job['lora_style']]
                    trigger_prefix = lora_config.get('trigger_prefix', '')
                    if trigger_prefix:
                        enhanced_prompt = f"{trigger_prefix} {job['prompt']}"
                        print(f"🎨 Applied FLUX LoRA trigger prefix: '{trigger_prefix}' (GPU {gpu_id})")
                elif job['lora_style'] in SD15_LORAS:
                    lora_config = SD15_LORAS[job['lora_style']]
                    trigger_prefix = lora_config.get('trigger_prefix', '')
                    if trigger_prefix:
                        enhanced_prompt = f"{trigger_prefix} {job['prompt']}"
                        print(f"🎨 Applied SD15 LoRA trigger prefix: '{trigger_prefix}' (GPU {gpu_id})")
                
                # Generate model
                result = generator.generate_3d_model(enhanced_prompt, job['seed'])
                
                if result is None:
                    job['status'] = 'failed'
                    job['error'] = 'Generation failed'
                else:
                    job['status'] = 'completed'
                    job['result'] = result
                
                job['end_time'] = time.time()
                
            except Exception as e:
                job['status'] = 'failed'
                job['error'] = str(e)
                job['end_time'] = time.time()
            finally:
                # Release GPU
                if job['gpu_id'] is not None:
                    gpu_manager.release_gpu(job['gpu_id'])
            
            return job
        
        # Execute jobs in parallel
        with ThreadPoolExecutor(max_workers=max_parallel) as executor:
            future_to_job = {executor.submit(execute_lora_generation, job): job for job in jobs}
            
            for future in as_completed(future_to_job):
                job = future.result()
                results.append(job)
                
                if job['status'] == 'completed':
                    print(f"✅ {job['lora_style']} LoRA generation {job['id']} completed on GPU {job['gpu_id']}")
                else:
                    print(f"❌ {job['lora_style']} LoRA generation {job['id']} failed: {job['error']}")
        
        # Prepare response
        successful_generations = [j for j in results if j['status'] == 'completed']
        failed_generations = [j for j in results if j['status'] == 'failed']
        
        # Calculate total time safely
        total_time = 0.0
        if results:
            valid_times = [j['end_time'] - j['start_time'] for j in results if j['start_time'] and j['end_time']]
            if valid_times:
                total_time = max(valid_times)
        
        response_data = {
            "status": "completed",
            "lora_style": lora_style,
            "total_models": num_models,
            "successful": len(successful_generations),
            "failed": len(failed_generations),
            "success_rate": len(successful_generations) / num_models * 100,
            "total_time": total_time,
            "jobs": results
        }
        
        # Add compressed data if requested
        if return_compressed and successful_generations:
            compressed_models = []
            for job in successful_generations:
                ply_data, compressed_data = job['result']
                if compressed_data:
                    compressed_models.append({
                        'seed': job['seed'],
                        'gpu_id': job['gpu_id'],
                        'lora_style': job['lora_style'],
                        'compressed_data': base64.b64encode(compressed_data).decode('utf-8'),
                        'size_bytes': len(compressed_data)
                    })
            
            response_data['compressed_models'] = compressed_models
        
        return JSONResponse(content=response_data)
        
    except Exception as e:
        print(f"❌ Parallel LoRA generation failed: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Parallel LoRA generation failed: {str(e)}")

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
            flux_output = generator.flux_pipeline(
                prompt=enhanced_prompt,
                generator=seed_generator,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale
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
            flux_output = generator.flux_pipeline(
                prompt=enhanced_prompt,
                generator=seed_generator,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale
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
            flux_output = generator.flux_pipeline(
                prompt=enhanced_prompt,
                generator=seed_generator,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale
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
            flux_output = generator.flux_pipeline(
                prompt=enhanced_prompt,
                generator=seed_generator,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale
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
            flux_output = generator.flux_pipeline(
                prompt=enhanced_prompt,
                generator=seed_generator,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale
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
            flux_output = generator.flux_pipeline(
                prompt=enhanced_prompt,
                generator=seed_generator,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale
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
            flux_output = generator.flux_pipeline(
                prompt=enhanced_prompt,
                generator=seed_generator,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale
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
            flux_output = generator.flux_pipeline(
                prompt=enhanced_prompt,
                generator=seed_generator,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale
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
            image = generator.flux_pipeline(
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
        
        if current_model == 'flux':
            # Load FLUX if not loaded
            if generator.flux_pipeline is None:
                generator._load_flux_models()
            
            # Generate image with FLUX
            print(f"🎨 Generating image with FLUX for: '{prompt}' (seed: {seed})")
            
            with torch.no_grad():
                flux_output = generator.flux_pipeline(
                    prompt=prompt,
                    seed=seed,
                    num_inference_steps=num_inference_steps,
                    guidance_scale=guidance_scale
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
                    seed=seed,
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
                    seed=seed,
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
    target_score: Optional[float] = Form(0.8)
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
        
        result = generator.generate_3d_model(enhanced_prompt, seed)
        
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

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FLUX + TRELLIS Generation Server with 8-GPU Parallel Processing")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8096, help="Port to bind to")
    parser.add_argument("--workers", type=int, default=1, help="Number of worker processes")
    parser.add_argument("--gpus", type=int, default=8, help="Number of GPUs to use")
    
    args = parser.parse_args()
    
    # Update GPU configuration
    MULTI_GPU_CONFIG['num_gpus'] = min(args.gpus, torch.cuda.device_count())
    
    print(f"🚀 Starting FLUX + TRELLIS Generation Server with {MULTI_GPU_CONFIG['num_gpus']}-GPU Parallel Processing")
    print(f"📍 Server: {args.host}:{args.port}")
    print("=" * 80)
    print("Pipeline: Text → FLUX → Image → TRELLIS → Gaussian Splatting PLY")
    print("Multi-GPU Features:")
    print(f"  • {MULTI_GPU_CONFIG['num_gpus']} GPUs for parallel processing")
    print("  • Intelligent GPU load balancing and scheduling")
    print("  • Parallel generation of up to 8 models simultaneously")
    print("  • GPU-specific model instances for optimal memory management")
    print("  • Real-time GPU status monitoring and management")
    print("Core Features:")
    print("  • FLUX text-to-image generation with quantization")
    print("  • TRELLIS image-to-3D Gaussian Splatting generation")
    print("  • SPZ compression for efficient storage/transmission")
    print("  • Multiple LoRA styles (Isometric 3D, Live 3D, Game Assets, etc.)")
    print("  • Optional validation integration")
    print("  • Memory-optimized per GPU (20GB limit)")
    print("=" * 80)
    print("API Endpoints:")
    print("  • POST /generate/ - Single generation on specific GPU")
    print("  • POST /generate_parallel/ - Parallel generation across GPUs")
    print("  • POST /generate_parallel_lora/{style}/ - Parallel LoRA generation")
    print("  • GET /gpu_status/ - Real-time GPU status")
    print("  • GET /parallel_jobs/ - Parallel job status")
    print("  • POST /gpu_reset/{id} - Reset specific GPU")
    print("  • POST /gpu_cleanup/ - Clean up all GPUs")
    print("=" * 80)
    
    # Initialize GPU manager
    print(f"🔧 Initializing Multi-GPU Manager...")
    gpu_manager._update_gpu_info()
    
    print(f"✅ Multi-GPU system ready with {gpu_manager.num_gpus} GPUs")
    print(f"📊 Parallel generation capacity: {gpu_manager.get_parallel_generation_capacity()} models")
    
    uvicorn.run(
        app, 
        host=args.host, 
        port=args.port,
        workers=args.workers,
        log_level="info"
    ) 
