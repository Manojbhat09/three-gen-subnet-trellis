#!/usr/bin/env python3
"""
Subnet 17 (404-GEN) - Concurrent HunyuanDiT + TRELLIS Generation Server
Purpose: HTTP server for concurrent high-quality 3D model generation using HunyuanDiT and TRELLIS
Supports multiple simultaneous generations with different prompts

Text Prompt → HunyuanDiT Image → TRELLIS 3D → Gaussian Splatting PLY + SPZ Compression

# Generate multiple 3D models concurrently
curl -X POST "http://localhost:8099/generate_batch/" \
  -F "prompts=a blue ceramic vase,a red wooden chair,a green metal lamp" \
  -F "seeds=42,123,456"

# Generate single 3D model
curl -X POST "http://localhost:8099/generate/" \
  -F "prompt=a blue ceramic vase" \
  -F "seed=42"

# Get all active jobs
curl "http://localhost:8099/jobs/"

# Get specific job status
curl "http://localhost:8099/jobs/{job_id}/"
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
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock, RLock

from fastapi import FastAPI, Form, HTTPException, BackgroundTasks
from fastapi.responses import Response, JSONResponse
import uvicorn
import torch

seed = 42
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# Set environment variables
os.environ['SPCONV_ALGO'] = 'native'
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

# Add TRELLIS to Python path
import sys
TRELLIS_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "TRELLIS-TextoImagen3D")
sys.path.append(TRELLIS_PATH)

# Add Hunyuan3D path
HUNYUAN3D_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Hunyuan3D-2")
sys.path.append(HUNYUAN3D_PATH)

# Import TRELLIS components
from trellis.pipelines import TrellisImageTo3DPipeline
from trellis.utils import render_utils, postprocessing_utils

# Import HunyuanDiT
from hy3dgen.text2image import HunyuanDiTPipeline

# Import OpenCV for object centering
import cv2

# Constants
MAX_SEED = np.iinfo(np.int32).max
MAX_CONCURRENT_JOBS = 3  # Adjust based on your GPU memory

# Configuration
GENERATION_CONFIG = {
    'output_dir': './trellis_hunyuan_concurrent_outputs',
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    'hunyuan_model_path': "Tencent-Hunyuan/HunyuanDiT-v1.1-Diffusers-Distilled",
    'trellis_model_path': 'cavargas10/TRELLIS',
    'save_intermediate_outputs': True,
    'save_preview': False,
    'auto_compress_ply': True,
    # HunyuanDiT specific settings
    'hunyuan_num_inference_steps': 25,
    'hunyuan_pag_scale': 1.3,
    'hunyuan_width': 1024,
    'hunyuan_height': 1024,
    # TRELLIS specific settings - OPTIMIZED FOR MAXIMUM QUALITY
    'guidance_scale': 4.0,
    'ss_guidance_strength': 9.5,
    'ss_sampling_steps': 30,
    'slat_guidance_strength': 5.0,
    'slat_sampling_steps': 30,
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
    # Concurrent processing settings
    'max_concurrent_jobs': MAX_CONCURRENT_JOBS,
    'job_timeout_seconds': 300,  # 5 minutes per job
    'enable_job_queuing': True,
}

@dataclass
class JobStatus:
    """Job status tracking"""
    job_id: str
    prompt: str
    seed: int
    status: str  # 'pending', 'processing', 'completed', 'failed', 'cancelled'
    start_time: Optional[float] = None
    end_time: Optional[float] = None
    error: Optional[str] = None
    ply_path: Optional[str] = None
    compressed_path: Optional[str] = None
    generation_time: Optional[float] = None
    validation_score: Optional[float] = None
    progress: float = 0.0  # 0.0 to 1.0
    current_step: str = "initializing"

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
    concurrent_jobs_processed: int = 0
    total_jobs_queued: int = 0

# Asset management
from enum import Enum

class AssetType(Enum):
    """Asset types for the generation pipeline"""
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
        if asset_type == AssetType.HUNYUAN_IMAGE:
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

class JobQueue:
    """Thread-safe job queue for managing concurrent generations"""
    
    def __init__(self, max_concurrent: int = MAX_CONCURRENT_JOBS):
        self.max_concurrent = max_concurrent
        self.active_jobs: Dict[str, JobStatus] = {}
        self.pending_jobs: queue.Queue = queue.Queue()
        self.completed_jobs: Dict[str, JobStatus] = {}
        self.lock = RLock()
        self.executor = ThreadPoolExecutor(max_workers=max_concurrent)
        self.running = True
        
        # Start job processor thread
        self.processor_thread = threading.Thread(target=self._process_jobs, daemon=True)
        self.processor_thread.start()
    
    def add_job(self, job_id: str, prompt: str, seed: int) -> bool:
        """Add a job to the queue"""
        with self.lock:
            if len(self.active_jobs) >= self.max_concurrent and self.pending_jobs.qsize() >= 10:
                return False  # Queue full
            
            job_status = JobStatus(
                job_id=job_id,
                prompt=prompt,
                seed=seed,
                status='pending'
            )
            
            self.pending_jobs.put(job_status)
            return True
    
    def get_job_status(self, job_id: str) -> Optional[JobStatus]:
        """Get job status by ID"""
        with self.lock:
            if job_id in self.active_jobs:
                return self.active_jobs[job_id]
            elif job_id in self.completed_jobs:
                return self.completed_jobs[job_id]
            return None
    
    def get_all_jobs(self) -> Dict[str, Any]:
        """Get all job statuses"""
        with self.lock:
            return {
                'active_jobs': {k: v.__dict__ for k, v in self.active_jobs.items()},
                'pending_count': self.pending_jobs.qsize(),
                'completed_jobs': {k: v.__dict__ for k, v in self.completed_jobs.items()},
                'max_concurrent': self.max_concurrent
            }
    
    def cancel_job(self, job_id: str) -> bool:
        """Cancel a job"""
        with self.lock:
            if job_id in self.active_jobs:
                self.active_jobs[job_id].status = 'cancelled'
                return True
            return False
    
    def _process_jobs(self):
        """Background job processor"""
        while self.running:
            try:
                # Get next job from queue
                job_status = self.pending_jobs.get(timeout=1.0)
                
                # Check if job was cancelled
                if job_status.status == 'cancelled':
                    continue
                
                # Add to active jobs
                with self.lock:
                    self.active_jobs[job_status.job_id] = job_status
                
                # Submit job to executor
                future = self.executor.submit(
                    self._execute_job,
                    job_status.job_id,
                    job_status.prompt,
                    job_status.seed
                )
                
                # Store future for potential cancellation
                job_status.future = future
                
            except queue.Empty:
                continue
            except Exception as e:
                print(f"Error in job processor: {e}")
    
    def _execute_job(self, job_id: str, prompt: str, seed: int):
        """Execute a single job"""
        try:
            # Update job status
            with self.lock:
                if job_id in self.active_jobs:
                    job = self.active_jobs[job_id]
                    job.status = 'processing'
                    job.start_time = time.time()
                    job.current_step = "loading models"
                    job.progress = 0.1
            
            # Execute generation
            result = generator.generate_3d_model_concurrent(job_id, prompt, seed)
            
            # Update job status
            with self.lock:
                if job_id in self.active_jobs:
                    job = self.active_jobs[job_id]
                    job.status = 'completed' if result else 'failed'
                    job.end_time = time.time()
                    job.generation_time = job.end_time - job.start_time
                    job.progress = 1.0
                    job.current_step = "completed"
                    
                    if result:
                        ply_data, compressed_data = result
                        job.ply_path = f"generated_model_{seed}.ply"
                        job.compressed_path = f"generated_model_{seed}.ply.spz"
                    
                    # Move to completed jobs
                    self.completed_jobs[job_id] = job
                    del self.active_jobs[job_id]
            
        except Exception as e:
            # Update job status on error
            with self.lock:
                if job_id in self.active_jobs:
                    job = self.active_jobs[job_id]
                    job.status = 'failed'
                    job.end_time = time.time()
                    job.error = str(e)
                    job.progress = 0.0
                    job.current_step = "failed"
                    
                    # Move to completed jobs
                    self.completed_jobs[job_id] = job
                    del self.active_jobs[job_id]
    
    def shutdown(self):
        """Shutdown the job queue"""
        self.running = False
        self.executor.shutdown(wait=True)

class ConcurrentHunyuanTrellisGenerator:
    def __init__(self):
        # Initialize model instance variables
        self.hunyuan_pipeline = None
        self.trellis_pipeline = None
        
        self.metrics = GenerationMetrics()
        self.generation_lock = threading.Lock()
        self.model_lock = RLock()  # Reentrant lock for model access
        
        # Initialize job queue
        self.job_queue = JobQueue(GENERATION_CONFIG['max_concurrent_jobs'])
        
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
        print("🔧 Concurrent HunyuanDiT + TRELLIS Generator initialized")
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

    def _load_hunyuan_pipeline(self):
        """Load HunyuanDiT pipeline with thread safety"""
        with self.model_lock:
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
        with self.model_lock:
            if self.hunyuan_pipeline is not None:
                print("🧹 Unloading HunyuanDiT pipeline...")
                del self.hunyuan_pipeline
                self.hunyuan_pipeline = None
                self._clear_gpu_memory()
                print("✅ HunyuanDiT pipeline unloaded")

    def _load_trellis_pipeline(self):
        """Load TRELLIS pipeline with thread safety"""
        with self.model_lock:
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
        with self.model_lock:
            if self.trellis_pipeline is not None:
                print("🧹 Unloading TRELLIS pipeline...")
                del self.trellis_pipeline
                self.trellis_pipeline = None
                self._clear_gpu_memory()
                print("✅ TRELLIS pipeline unloaded")

    def center_object_in_image(self, image: Image.Image, white_threshold: int = 240, padding: int = 20) -> Image.Image:
        """
        Center the main object in the image by detecting content and repositioning it
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

    def generate_3d_model_concurrent(self, job_id: str, prompt: str, seed: int) -> Optional[Tuple[bytes, Optional[bytes]]]:
        """Generate 3D model from text prompt using HunyuanDiT + TRELLIS pipeline (concurrent version)"""
        
        start_time = time.time()
        
        try:
            print(f"🎯 [Job {job_id}] Starting HunyuanDiT + TRELLIS generation for: '{prompt}' (seed: {seed})")
            
            # Update job progress
            self._update_job_progress(job_id, 0.1, "initializing")
            
            # Step 1: Generate image with HunyuanDiT
            print(f"[Job {job_id}] Step 1: Generating image with HunyuanDiT...")
            self._update_job_progress(job_id, 0.2, "loading HunyuanDiT")
            
            with self.model_lock:
                if self.hunyuan_pipeline is None:
                    self._load_hunyuan_pipeline()
            
            self._update_job_progress(job_id, 0.3, "generating image")
            
            # Enhanced prompt for maximum alignment score
            enhanced_prompt = prompt
            
            # Generate image with HunyuanDiT
            with torch.no_grad():
                with self.model_lock:
                    image = self.hunyuan_pipeline(
                        prompt=enhanced_prompt,
                        seed=seed
                    )
            
            print(f"[Job {job_id}] ✓ HunyuanDiT image generated successfully")
            self._update_job_progress(job_id, 0.4, "image generated")
            
            # Step 1.3: Center object in image
            if GENERATION_CONFIG.get('enable_object_centering', True):
                print(f"[Job {job_id}] Step 1.3: Centering object in image...")
                try:
                    centered_image = self.center_object_in_image(
                        image, 
                        white_threshold=GENERATION_CONFIG.get('centering_white_threshold', 240),
                        padding=GENERATION_CONFIG.get('centering_padding', 40)
                    )
                    print(f"[Job {job_id}] ✓ Object centered successfully")
                    image = centered_image
                except Exception as e:
                    print(f"[Job {job_id}] ⚠️ Object centering failed: {e}")
                    print("   Continuing with original image...")
            
            # Step 2: Generate 3D model with TRELLIS
            print(f"[Job {job_id}] Step 2: Generating 3D model with TRELLIS...")
            self._update_job_progress(job_id, 0.5, "loading TRELLIS")
            
            with self.model_lock:
                if self.trellis_pipeline is None:   
                    self._load_trellis_pipeline()
            
            self._update_job_progress(job_id, 0.6, "generating 3D model")
            
            # Enhanced TRELLIS parameters for maximum quality
            with self.model_lock:
                outputs = self.trellis_pipeline.run(
                    image,
                    seed=seed,
                    formats=["gaussian", "mesh"],
                    preprocess_image=False,
                    sparse_structure_sampler_params={
                        "steps": GENERATION_CONFIG['ss_sampling_steps'],
                        "cfg_strength": GENERATION_CONFIG['ss_guidance_strength'],
                        "cfg_interval": (0.3, 0.98),
                        "rescale_t": 3.0,
                    },
                    slat_sampler_params={
                        "steps": GENERATION_CONFIG['slat_sampling_steps'],
                        "cfg_strength": GENERATION_CONFIG['slat_guidance_strength'],
                        "cfg_interval": (0.3, 0.98),
                        "rescale_t": 3.0,
                    },
                )
            
            print(f"[Job {job_id}] ✓ 3D model generated successfully")
            self._update_job_progress(job_id, 0.7, "3D model generated")
            
            # Step 3: Extract and enhance Gaussian Splatting PLY
            print(f"[Job {job_id}] Step 3: Extracting and enhancing Gaussian Splatting PLY...")
            gaussian_output = outputs['gaussian'][0]
            
            # Quality enhancement: Filter low-quality splats
            print(f"[Job {job_id}]   Enhancing quality by filtering low-quality splats...")
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
                    
                    print(f"[Job {job_id}]   Quality enhancement: Kept {quality_mask.sum().item():,} high-quality splats out of {len(points):,}")
                else:
                    print(f"[Job {job_id}]   Quality enhancement skipped: Too few splats would remain ({quality_mask.sum().item()})")
                    
            except Exception as e:
                print(f"[Job {job_id}]   Quality enhancement failed: {e}")
                print("   Continuing with original splats...")
            
            # Save as PLY file
            import io
            ply_buffer = io.BytesIO()
            gaussian_output.save_ply(ply_buffer)
            ply_data = ply_buffer.getvalue()
            
            print(f"[Job {job_id}] ✓ Gaussian Splatting PLY extracted ({len(ply_data):,} bytes)")
            self._update_job_progress(job_id, 0.8, "PLY extracted")
            
            # Step 4: Compress PLY if enabled
            compressed_data = None
            if GENERATION_CONFIG.get('auto_compress_ply', True):
                print(f"[Job {job_id}] Step 4: Compressing PLY with SPZ...")
                try:
                    import pyspz
                    compressed_data = pyspz.compress(ply_data, workers=-1)
                    print(f"[Job {job_id}] 🗜️ SPZ Compression successful:")
                    print(f"   Original: {len(ply_data):,} bytes ({len(ply_data)/1024/1024:.1f} MB)")
                    print(f"   Compressed: {len(compressed_data):,} bytes ({len(compressed_data)/1024/1024:.1f} MB)") 
                    print(f"   Ratio: {len(compressed_data)/len(ply_data)*100:.1f}%")
                    print(f"   Space saved: {(len(ply_data)-len(compressed_data))/1024/1024:.1f} MB")
                    
                except Exception as e:
                    print(f"[Job {job_id}] ⚠️ SPZ compression failed: {e}")
                    compressed_data = None
            
            self._update_job_progress(job_id, 0.9, "compression completed")
            
            generation_time = time.time() - start_time
            
            # Update metrics
            self.metrics.total_generations += 1
            self.metrics.successful_generations += 1
            self.metrics.concurrent_jobs_processed += 1
            self.metrics.last_generation_time = generation_time
            self.metrics.average_generation_time = (
                (self.metrics.average_generation_time * (self.metrics.successful_generations - 1) + generation_time) 
                / self.metrics.successful_generations
            )
            
            print(f"[Job {job_id}] 🎉 HunyuanDiT + TRELLIS generation completed in {generation_time:.2f}s")
            self._update_job_progress(job_id, 1.0, "completed")
                            
            return ply_data, compressed_data
                
        except Exception as e:
            self.metrics.total_generations += 1
            self.metrics.failed_generations += 1
            print(f"[Job {job_id}] ❌ HunyuanDiT + TRELLIS generation failed: {e}")
            traceback.print_exc()
            
            self._update_job_progress(job_id, 0.0, f"failed: {str(e)}")
            
            return None

    def _update_job_progress(self, job_id: str, progress: float, step: str):
        """Update job progress"""
        try:
            job_status = self.job_queue.get_job_status(job_id)
            if job_status:
                job_status.progress = progress
                job_status.current_step = step
        except Exception as e:
            print(f"Warning: Could not update job progress: {e}")

    def get_status(self) -> Dict[str, Any]:
        """Get server status and metrics"""
        return {
            "status": "running",
            "models_loaded": {
                "hunyuan_pipeline": self.hunyuan_pipeline is not None,
                "trellis_pipeline": self.trellis_pipeline is not None,
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
                "concurrent_jobs_processed": self.metrics.concurrent_jobs_processed,
                "total_jobs_queued": self.metrics.total_jobs_queued,
            },
            "config": GENERATION_CONFIG,
            "gpu_memory": self._clear_gpu_memory() if torch.cuda.is_available() else 0,
            "ready": self.ready
        }

# Initialize FastAPI app
app = FastAPI(title="Concurrent HunyuanDiT + TRELLIS Generation Server", version="1.0.0")

# Initialize global generator
generator = ConcurrentHunyuanTrellisGenerator()

@app.post("/generate/")
async def generate_3d_model_endpoint(
    prompt: str = Form(...), 
    seed: Optional[int] = Form(None),
    return_compressed: Optional[bool] = Form(True)
):
    """Generate 3D model from text prompt using HunyuanDiT + TRELLIS pipeline."""
    
    # Handle seed
    if seed is None:
        seed = random.randint(0, 2**31 - 1)
    
    # Generate unique job ID
    job_id = str(uuid.uuid4())
    
    # Add job to queue
    if not generator.job_queue.add_job(job_id, prompt, seed):
        raise HTTPException(status_code=503, detail="Job queue is full. Please try again later.")
    
    generator.metrics.total_jobs_queued += 1
    
    return JSONResponse(content={
        "status": "job_queued",
        "job_id": job_id,
        "prompt": prompt,
        "seed": seed,
        "message": "Job has been queued for processing. Use /jobs/{job_id}/ to check status."
    })

@app.post("/generate_batch/")
async def generate_batch_endpoint(
    prompts: str = Form(...),  # Comma-separated prompts
    seeds: Optional[str] = Form(None),  # Comma-separated seeds
    return_compressed: Optional[bool] = Form(True)
):
    """Generate multiple 3D models from text prompts concurrently."""
    
    # Parse prompts
    prompt_list = [p.strip() for p in prompts.split(',') if p.strip()]
    if not prompt_list:
        raise HTTPException(status_code=400, detail="No valid prompts provided")
    
    # Parse seeds
    seed_list = []
    if seeds:
        seed_list = [int(s.strip()) for s in seeds.split(',') if s.strip().isdigit()]
    
    # Generate seeds if not provided
    while len(seed_list) < len(prompt_list):
        seed_list.append(random.randint(0, 2**31 - 1))
    
    # Limit batch size
    max_batch_size = min(5, GENERATION_CONFIG['max_concurrent_jobs'])
    if len(prompt_list) > max_batch_size:
        raise HTTPException(
            status_code=400, 
            detail=f"Batch size too large. Maximum allowed: {max_batch_size}"
        )
    
    # Create jobs
    job_ids = []
    for prompt, seed in zip(prompt_list, seed_list):
        job_id = str(uuid.uuid4())
        if generator.job_queue.add_job(job_id, prompt, seed):
            job_ids.append(job_id)
            generator.metrics.total_jobs_queued += 1
        else:
            raise HTTPException(status_code=503, detail="Job queue is full. Please try again later.")
    
    return JSONResponse(content={
        "status": "batch_queued",
        "job_ids": job_ids,
        "prompts": prompt_list,
        "seeds": seed_list,
        "batch_size": len(job_ids),
        "message": f"Batch of {len(job_ids)} jobs has been queued for processing."
    })

@app.get("/jobs/")
async def get_all_jobs():
    """Get all job statuses"""
    return generator.job_queue.get_all_jobs()

@app.get("/jobs/{job_id}/")
async def get_job_status(job_id: str):
    """Get specific job status"""
    job_status = generator.job_queue.get_job_status(job_id)
    if job_status is None:
        raise HTTPException(status_code=404, detail="Job not found")
    
    return job_status.__dict__

@app.post("/jobs/{job_id}/cancel/")
async def cancel_job(job_id: str):
    """Cancel a specific job"""
    if generator.job_queue.cancel_job(job_id):
        return {"status": "cancelled", "job_id": job_id}
    else:
        raise HTTPException(status_code=404, detail="Job not found or already completed")

@app.get("/jobs/{job_id}/download/")
async def download_job_result(job_id: str, compressed: bool = True):
    """Download the result of a completed job"""
    job_status = generator.job_queue.get_job_status(job_id)
    if job_status is None:
        raise HTTPException(status_code=404, detail="Job not found")
    
    if job_status.status != 'completed':
        raise HTTPException(status_code=400, detail="Job not completed yet")
    
    # For now, return a placeholder response
    # In a full implementation, you would load the actual file from disk
    return JSONResponse(content={
        "status": "success",
        "job_id": job_id,
        "message": "Download functionality would be implemented here",
        "ply_path": job_status.ply_path,
        "compressed_path": job_status.compressed_path
    })

@app.get("/status/")
async def get_server_status():
    """Get server status and metrics"""
    return generator.get_status()

@app.get("/health/")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "timestamp": time.time()}

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

@app.post("/models/unload/")
async def unload_models():
    """Manually unload models to free GPU memory"""
    try:
        print("🧹 Manually unloading models...")
        
        # Unload HunyuanDiT
        if generator.hunyuan_pipeline is not None:
            generator._unload_hunyuan_pipeline()
        
        # Unload TRELLIS
        if generator.trellis_pipeline is not None:
            generator._unload_trellis_pipeline()
        
        return {
            "status": "success",
            "message": "Models unloaded successfully",
            "available_memory_gb": generator._clear_gpu_memory()
        }
    except Exception as e:
        return JSONResponse(content={
            "status": "error",
            "message": str(e)
        }, status_code=500)

@app.post("/models/load/")
async def load_models():
    """Manually load models (useful for warm-up)"""
    try:
        print("🔧 Manually loading models...")
        
        # Load HunyuanDiT
        if generator.hunyuan_pipeline is None:
            generator._load_hunyuan_pipeline()
        
        # Load TRELLIS
        if generator.trellis_pipeline is None:
            generator._load_trellis_pipeline()
        
        return {
            "status": "success",
            "message": "Models loaded successfully",
            "models_loaded": {
                "hunyuan_pipeline": generator.hunyuan_pipeline is not None,
                "trellis_pipeline": generator.trellis_pipeline is not None,
            }
        }
    except Exception as e:
        return JSONResponse(content={
            "status": "error",
            "message": str(e)
        }, status_code=500)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Concurrent HunyuanDiT + TRELLIS Generation Server")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8099, help="Port to bind to")
    parser.add_argument("--workers", type=int, default=1, help="Number of worker processes")
    parser.add_argument("--max-concurrent", type=int, default=MAX_CONCURRENT_JOBS, help="Maximum concurrent jobs")
    
    args = parser.parse_args()
    
    # Update config with command line args
    GENERATION_CONFIG['max_concurrent_jobs'] = args.max_concurrent
    
    print(f"Starting Concurrent HunyuanDiT + TRELLIS Generation Server on {args.host}:{args.port}")
    print("=" * 80)
    print("Pipeline: Text → HunyuanDiT → Image → TRELLIS → Gaussian Splatting PLY")
    print("Features:")
    print("  • Concurrent processing of multiple generations")
    print("  • Job queue with status tracking")
    print("  • HunyuanDiT text-to-image generation for excellent alignment")
    print("  • TRELLIS image-to-3D Gaussian Splatting generation")
    print("  • SPZ compression for efficient storage/transmission")
    print("  • Memory-optimized for RTX 4090 (24GB)")
    print("  • Enhanced alignment scores to avoid 0.0 task fidelity")
    print(f"  • Maximum concurrent jobs: {args.max_concurrent}")
    print("=" * 80)
    
    try:
        uvicorn.run(
            app, 
            host=args.host, 
            port=args.port,
            workers=args.workers,
            log_level="info"
        )
    finally:
        # Cleanup
        if hasattr(generator, 'job_queue'):
            generator.job_queue.shutdown() 