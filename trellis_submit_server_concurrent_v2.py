#!/usr/bin/env python3
"""
Subnet 17 (404-GEN) - Concurrent HunyuanDiT + TRELLIS Generation Server V2
Purpose: HTTP server for truly concurrent high-quality 3D model generation
Supports multiple simultaneous generations with different prompt enhancements

# Generate multiple models with different enhancements
curl -X POST "http://localhost:8099/generate_batch_enhanced/" \
  -F "prompts=a blue ceramic vase,a red wooden chair" \
  -F "enhancements=professional 3D render,isometric view version" \
  -F "seeds=42,123"

# Generate with same prompt but different enhancements
curl -X POST "http://localhost:8099/generate_batch_enhanced/" \
  -F "prompts=a blue ceramic vase,a blue ceramic vase" \
  -F "enhancements=professional 3D render,isometric view version" \
  -F "seeds=42,123"
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
from functools import partial

from fastapi import FastAPI, Form, HTTPException, BackgroundTasks
from fastapi.responses import Response, JSONResponse
import uvicorn
import torch

# Set environment variables
os.environ['SPCONV_ALGO'] = 'native'
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

# Add paths
import sys
TRELLIS_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "TRELLIS-TextoImagen3D")
HUNYUAN3D_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Hunyuan3D-2")
sys.path.extend([TRELLIS_PATH, HUNYUAN3D_PATH])

# Import components
from trellis.pipelines import TrellisImageTo3DPipeline
from trellis.utils import render_utils, postprocessing_utils
from hy3dgen.text2image import HunyuanDiTPipeline
import cv2

# Configuration
GENERATION_CONFIG = {
    'output_dir': './trellis_hunyuan_concurrent_outputs',
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    'hunyuan_model_path': "Tencent-Hunyuan/HunyuanDiT-v1.1-Diffusers-Distilled",
    'trellis_model_path': 'cavargas10/TRELLIS',
    'max_concurrent_jobs': 2,  # Reduced for better GPU memory management
    'job_timeout_seconds': 300,
    'enable_object_centering': True,
    'centering_white_threshold': 240,
    'centering_padding': 30,
    'auto_compress_ply': True,
    'ss_guidance_strength': 9.5,
    'ss_sampling_steps': 30,
    'slat_guidance_strength': 5.0,
    'slat_sampling_steps': 30,
}

@dataclass
class GenerationResult:
    """Result of a single generation"""
    job_id: str
    prompt: str
    enhancement: str
    seed: int
    status: str
    ply_data: Optional[bytes] = None
    compressed_data: Optional[bytes] = None
    error: Optional[str] = None
    generation_time: float = 0.0
    validation_score: Optional[float] = None

class ModelManager:
    """Manages model instances for concurrent access"""
    
    def __init__(self):
        self.hunyuan_pipelines = {}
        self.trellis_pipelines = {}
        self.lock = Lock()
        
    def get_hunyuan_pipeline(self, job_id: str):
        """Get or create HunyuanDiT pipeline for a job"""
        with self.lock:
            if job_id not in self.hunyuan_pipelines:
                print(f"[Job {job_id}] Loading HunyuanDiT pipeline...")
                try:
                    pipeline = HunyuanDiTPipeline(
                        model_path=GENERATION_CONFIG['hunyuan_model_path'],
                        device=GENERATION_CONFIG['device']
                    )
                    self.hunyuan_pipelines[job_id] = pipeline
                    print(f"[Job {job_id}] ✓ HunyuanDiT pipeline loaded")
                except Exception as e:
                    print(f"[Job {job_id}] ❌ HunyuanDiT pipeline loading failed: {e}")
                    return None
            return self.hunyuan_pipelines[job_id]
    
    def get_trellis_pipeline(self, job_id: str):
        """Get or create TRELLIS pipeline for a job"""
        with self.lock:
            if job_id not in self.trellis_pipelines:
                print(f"[Job {job_id}] Loading TRELLIS pipeline...")
                try:
                    pipeline = TrellisImageTo3DPipeline.from_pretrained(
                        GENERATION_CONFIG['trellis_model_path']
                    )
                    pipeline.cuda()
                    self.trellis_pipelines[job_id] = pipeline
                    print(f"[Job {job_id}] ✓ TRELLIS pipeline loaded")
                except Exception as e:
                    print(f"[Job {job_id}] ❌ TRELLIS pipeline loading failed: {e}")
                    return None
            return self.trellis_pipelines[job_id]
    
    def cleanup_job(self, job_id: str):
        """Clean up models for a specific job"""
        with self.lock:
            if job_id in self.hunyuan_pipelines:
                del self.hunyuan_pipelines[job_id]
            if job_id in self.trellis_pipelines:
                del self.trellis_pipelines[job_id]

def center_object_in_image(image: Image.Image, white_threshold: int = 240, padding: int = 20) -> Image.Image:
    """Center the main object in the image"""
    try:
        image_array = np.array(image)
        original_height, original_width = image_array.shape[:2]
        
        if len(image_array.shape) == 3:
            if image_array.shape[2] == 4:
                alpha = image_array[:, :, 3]
                gray = cv2.cvtColor(image_array[:, :, :3], cv2.COLOR_RGB2GRAY)
                gray = np.where(alpha > 0, gray, 255)
            else:
                gray = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)
        else:
            gray = image_array
        
        content_mask = gray < white_threshold
        contours, _ = cv2.findContours(
            content_mask.astype(np.uint8), 
            cv2.RETR_EXTERNAL, 
            cv2.CHAIN_APPROX_SIMPLE
        )
        
        if not contours:
            return image
        
        all_points = np.vstack(contours)
        x_min = np.min(all_points[:, :, 0])
        y_min = np.min(all_points[:, :, 1])
        x_max = np.max(all_points[:, :, 0])
        y_max = np.max(all_points[:, :, 1])
        
        x_min = max(0, x_min - padding)
        y_min = max(0, y_min - padding)
        x_max = min(original_width, x_max + padding)
        y_max = min(original_height, y_max + padding)
        
        content_width = x_max - x_min
        content_height = y_max - y_min
        content = image_array[y_min:y_max, x_min:x_max]
        
        if len(image_array.shape) == 3:
            if image_array.shape[2] == 4:
                centered_image = np.full((original_height, original_width, 4), [255, 255, 255, 255], dtype=np.uint8)
            else:
                centered_image = np.full((original_height, original_width, 3), [255, 255, 255], dtype=np.uint8)
        else:
            centered_image = np.full((original_height, original_width), 255, dtype=np.uint8)
        
        center_x = original_width // 2
        center_y = original_height // 2
        paste_x = center_x - content_width // 2
        paste_y = center_y - content_height // 2
        
        paste_x = max(0, min(paste_x, original_width - content_width))
        paste_y = max(0, min(paste_y, original_height - content_height))
        
        end_x = paste_x + content_width
        end_y = paste_y + content_height
        
        if end_x > original_width:
            content = content[:, :original_width - paste_x]
            end_x = original_width
        if end_y > original_height:
            content = content[:original_height - paste_y]
            end_y = original_height
        
        centered_image[paste_y:end_y, paste_x:end_x] = content
        
        return Image.fromarray(centered_image)
        
    except Exception as e:
        print(f"⚠️ Object centering failed: {e}")
        return image

def generate_single_model(job_id: str, prompt: str, enhancement: str, seed: int, model_manager: ModelManager) -> GenerationResult:
    """Generate a single 3D model with specific enhancement"""
    start_time = time.time()
    result = GenerationResult(
        job_id=job_id,
        prompt=prompt,
        enhancement=enhancement,
        seed=seed,
        status="processing"
    )
    
    try:
        print(f"🎯 [Job {job_id}] Starting generation: '{prompt}' + '{enhancement}' (seed: {seed})")
        
        # Combine prompt with enhancement
        full_prompt = f"{enhancement}, {prompt}" if enhancement else prompt
        
        # Step 1: Generate image with HunyuanDiT
        print(f"[Job {job_id}] Step 1: Generating image with HunyuanDiT...")
        hunyuan_pipeline = model_manager.get_hunyuan_pipeline(job_id)
        if hunyuan_pipeline is None:
            raise Exception("Failed to load HunyuanDiT pipeline")
        
        with torch.no_grad():
            image = hunyuan_pipeline(prompt=full_prompt, seed=seed)
        
        print(f"[Job {job_id}] ✓ HunyuanDiT image generated")
        
        # Step 1.3: Center object
        if GENERATION_CONFIG.get('enable_object_centering', True):
            print(f"[Job {job_id}] Step 1.3: Centering object...")
            image = center_object_in_image(
                image, 
                white_threshold=GENERATION_CONFIG.get('centering_white_threshold', 240),
                padding=GENERATION_CONFIG.get('centering_padding', 30)
            )
            print(f"[Job {job_id}] ✓ Object centered")
        
        # Step 2: Generate 3D model with TRELLIS
        print(f"[Job {job_id}] Step 2: Generating 3D model with TRELLIS...")
        trellis_pipeline = model_manager.get_trellis_pipeline(job_id)
        if trellis_pipeline is None:
            raise Exception("Failed to load TRELLIS pipeline")
        
        outputs = trellis_pipeline.run(
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
        
        print(f"[Job {job_id}] ✓ 3D model generated")
        
        # Step 3: Extract PLY
        print(f"[Job {job_id}] Step 3: Extracting PLY...")
        gaussian_output = outputs['gaussian'][0]
        
        # Quality enhancement
        try:
            points = gaussian_output.points
            opacities = gaussian_output.opacities
            scales = gaussian_output.scales
            
            opacity_threshold = 0.01
            scale_threshold = 0.001
            quality_mask = (opacities > opacity_threshold) & (torch.norm(scales, dim=1) > scale_threshold)
            
            if quality_mask.sum() > 7000:
                gaussian_output.points = points[quality_mask]
                gaussian_output.opacities = opacities[quality_mask]
                gaussian_output.scales = scales[quality_mask]
                gaussian_output.rotations = gaussian_output.rotations[quality_mask]
                gaussian_output.features_dc = gaussian_output.features_dc[quality_mask]
                gaussian_output.features_rest = gaussian_output.features_rest[quality_mask]
                gaussian_output.normals = gaussian_output.normals[quality_mask]
                print(f"[Job {job_id}] Quality enhancement: Kept {quality_mask.sum().item():,} splats")
        except Exception as e:
            print(f"[Job {job_id}] Quality enhancement failed: {e}")
        
        # Save PLY
        ply_buffer = io.BytesIO()
        gaussian_output.save_ply(ply_buffer)
        ply_data = ply_buffer.getvalue()
        
        print(f"[Job {job_id}] ✓ PLY extracted ({len(ply_data):,} bytes)")
        
        # Step 4: Compress if enabled
        compressed_data = None
        if GENERATION_CONFIG.get('auto_compress_ply', True):
            print(f"[Job {job_id}] Step 4: Compressing PLY...")
            try:
                import pyspz
                compressed_data = pyspz.compress(ply_data, workers=-1)
                print(f"[Job {job_id}] ✓ SPZ compression: {len(compressed_data):,} bytes")
            except Exception as e:
                print(f"[Job {job_id}] ⚠️ SPZ compression failed: {e}")
        
        # Update result
        result.status = "completed"
        result.ply_data = ply_data
        result.compressed_data = compressed_data
        result.generation_time = time.time() - start_time
        
        print(f"[Job {job_id}] 🎉 Generation completed in {result.generation_time:.2f}s")
        
    except Exception as e:
        result.status = "failed"
        result.error = str(e)
        result.generation_time = time.time() - start_time
        print(f"[Job {job_id}] ❌ Generation failed: {e}")
        traceback.print_exc()
    
    finally:
        # Cleanup models for this job
        model_manager.cleanup_job(job_id)
    
    return result

# Initialize FastAPI app
app = FastAPI(title="Concurrent HunyuanDiT + TRELLIS Generation Server V2", version="2.0.0")

# Global model manager
model_manager = ModelManager()

@app.post("/generate_batch_enhanced/")
async def generate_batch_enhanced(
    prompts: str = Form(...),  # Comma-separated prompts
    enhancements: str = Form(...),  # Comma-separated enhancements
    seeds: Optional[str] = Form(None),  # Comma-separated seeds
    return_compressed: Optional[bool] = Form(True)
):
    """Generate multiple 3D models with different prompt enhancements concurrently"""
    
    # Parse inputs
    prompt_list = [p.strip() for p in prompts.split(',') if p.strip()]
    enhancement_list = [e.strip() for e in enhancements.split(',') if e.strip()]
    
    if not prompt_list:
        raise HTTPException(status_code=400, detail="No valid prompts provided")
    
    # Extend enhancements if fewer than prompts
    while len(enhancement_list) < len(prompt_list):
        enhancement_list.append(enhancement_list[-1] if enhancement_list else "")
    
    # Parse seeds
    seed_list = []
    if seeds:
        seed_list = [int(s.strip()) for s in seeds.split(',') if s.strip().isdigit()]
    
    while len(seed_list) < len(prompt_list):
        seed_list.append(random.randint(0, 2**31 - 1))
    
    # Limit batch size
    max_batch_size = GENERATION_CONFIG['max_concurrent_jobs']
    if len(prompt_list) > max_batch_size:
        raise HTTPException(
            status_code=400, 
            detail=f"Batch size too large. Maximum allowed: {max_batch_size}"
        )
    
    print(f"🚀 Starting batch generation of {len(prompt_list)} models:")
    for i, (prompt, enhancement, seed) in enumerate(zip(prompt_list, enhancement_list, seed_list)):
        print(f"  {i+1}. Prompt: '{prompt}' + Enhancement: '{enhancement}' (seed: {seed})")
    
    # Generate job IDs
    job_ids = [str(uuid.uuid4()) for _ in prompt_list]
    
    # Create generation tasks
    generation_tasks = []
    for job_id, prompt, enhancement, seed in zip(job_ids, prompt_list, enhancement_list, seed_list):
        task = partial(generate_single_model, job_id, prompt, enhancement, seed, model_manager)
        generation_tasks.append(task)
    
    # Execute all generations concurrently
    results = []
    with ThreadPoolExecutor(max_workers=max_batch_size) as executor:
        futures = [executor.submit(task) for task in generation_tasks]
        
        for future in as_completed(futures):
            try:
                result = future.result()
                results.append(result)
            except Exception as e:
                print(f"❌ Task failed: {e}")
    
    # Sort results by job_id to maintain order
    results.sort(key=lambda x: job_ids.index(x.job_id))
    
    # Prepare response
    response_data = {
        "status": "batch_completed",
        "total_jobs": len(prompt_list),
        "successful_jobs": len([r for r in results if r.status == "completed"]),
        "failed_jobs": len([r for r in results if r.status == "failed"]),
        "results": []
    }
    
    # Add individual results
    for result in results:
        result_data = {
            "job_id": result.job_id,
            "prompt": result.prompt,
            "enhancement": result.enhancement,
            "seed": result.seed,
            "status": result.status,
            "generation_time": result.generation_time,
            "ply_size_bytes": len(result.ply_data) if result.ply_data else 0,
            "compressed_size_bytes": len(result.compressed_data) if result.compressed_data else 0,
        }
        
        if result.error:
            result_data["error"] = result.error
        
        response_data["results"].append(result_data)
    
    # If all successful and return_compressed, create combined response
    if return_compressed and all(r.status == "completed" for r in results):
        try:
            # Create combined compressed data
            combined_data = {}
            for result in results:
                if result.compressed_data:
                    combined_data[result.job_id] = {
                        "prompt": result.prompt,
                        "enhancement": result.enhancement,
                        "seed": result.seed,
                        "data": base64.b64encode(result.compressed_data).decode('utf-8')
                    }
            
            # Return combined compressed data
            return Response(
                content=json.dumps(combined_data, indent=2),
                media_type="application/json",
                headers={
                    "Content-Disposition": f"attachment; filename=batch_results_{int(time.time())}.json",
                    "X-Batch-Size": str(len(results)),
                    "X-Successful-Jobs": str(len([r for r in results if r.status == "completed"])),
                    "X-Pipeline": "hunyuan_trellis_concurrent_v2"
                }
            )
        except Exception as e:
            print(f"⚠️ Failed to create combined response: {e}")
    
    return JSONResponse(content=response_data)

@app.get("/status/")
async def get_server_status():
    """Get server status"""
    return {
        "status": "running",
        "max_concurrent_jobs": GENERATION_CONFIG['max_concurrent_jobs'],
        "config": GENERATION_CONFIG,
        "gpu_memory": torch.cuda.mem_get_info()[0] / 1e9 if torch.cuda.is_available() else 0
    }

@app.get("/health/")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "timestamp": time.time()}

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Concurrent HunyuanDiT + TRELLIS Generation Server V2")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8099, help="Port to bind to")
    parser.add_argument("--max-concurrent", type=int, default=2, help="Maximum concurrent jobs")
    
    args = parser.parse_args()
    GENERATION_CONFIG['max_concurrent_jobs'] = args.max_concurrent
    
    print(f"Starting Concurrent HunyuanDiT + TRELLIS Generation Server V2 on {args.host}:{args.port}")
    print("=" * 80)
    print("Features:")
    print("  • True concurrent processing with separate model instances")
    print("  • Different prompt enhancements for each generation")
    print("  • Combined results in single response")
    print("  • Optimized for GPU memory management")
    print(f"  • Maximum concurrent jobs: {args.max_concurrent}")
    print("=" * 80)
    
    uvicorn.run(app, host=args.host, port=args.port, log_level="info") 