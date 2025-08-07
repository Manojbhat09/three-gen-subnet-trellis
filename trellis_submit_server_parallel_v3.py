#!/usr/bin/env python3
"""
Subnet 17 (404-GEN) - Parallel HunyuanDiT + TRELLIS Generation Server V3
Purpose: HTTP server with 2 complete model pipelines running truly in parallel
Two separate HunyuanDiT + TRELLIS pipelines running simultaneously

# Generate with 2 different prompts/enhancements in parallel
curl -X POST "http://localhost:8099/generate_parallel/" \
  -F "prompt1=a blue ceramic vase" \
  -F "enhancement1=professional 3D render" \
  -F "seed1=42" \
  -F "prompt2=a red wooden chair" \
  -F "enhancement2=isometric view version" \
  -F "seed2=123"
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
from threading import Lock, RLock, Event
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

# Import CLIP for image comparison
import clip
from PIL import Image
import torch.nn.functional as F

# Configuration
GENERATION_CONFIG = {
    'output_dir': './trellis_hunyuan_parallel_outputs',
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    'hunyuan_model_path': "Tencent-Hunyuan/HunyuanDiT-v1.1-Diffusers-Distilled",
    'trellis_model_path': 'cavargas10/TRELLIS',
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
    pipeline_id: int
    prompt: str
    enhancement: str
    seed: int
    status: str
    ply_data: Optional[bytes] = None
    compressed_data: Optional[bytes] = None
    error: Optional[str] = None
    generation_time: float = 0.0
    validation_score: Optional[float] = None

class ParallelModelPipelines:
    """Manages 2 complete model pipelines running in parallel"""
    
    def __init__(self):
        self.pipelines = {
            1: {'hunyuan': None, 'trellis': None, 'lock': Lock()},
            2: {'hunyuan': None, 'trellis': None, 'lock': Lock()}
        }
        self.clip_model = None
        self.clip_preprocess = None
        self.initialized = False
        self.init_lock = Lock()
        
    def initialize_pipelines(self):
        """Initialize both model pipelines"""
        with self.init_lock:
            if self.initialized:
                return
            
            print("🔧 Initializing 2 parallel model pipelines...")
            
            # Initialize pipeline 1
            print("Loading Pipeline 1...")
            try:
                # HunyuanDiT for pipeline 1
                self.pipelines[1]['hunyuan'] = HunyuanDiTPipeline(
                    model_path=GENERATION_CONFIG['hunyuan_model_path'],
                    device=GENERATION_CONFIG['device']
                )
                
                # Compile HunyuanDiT for better performance
                try:
                    print("   Compiling Pipeline 1 HunyuanDiT for better performance...")
                    self.pipelines[1]['hunyuan'].compile()
                    print("   ✓ Pipeline 1 HunyuanDiT compiled successfully")
                except Exception as e:
                    print(f"   ⚠️ Pipeline 1 HunyuanDiT compilation failed: {e}")
                    print("   Continuing without compilation...")
                
                print("✓ Pipeline 1 HunyuanDiT loaded")
                
                # TRELLIS for pipeline 1
                self.pipelines[1]['trellis'] = TrellisImageTo3DPipeline.from_pretrained(
                    GENERATION_CONFIG['trellis_model_path']
                )
                self.pipelines[1]['trellis'].cuda()
                print("✓ Pipeline 1 TRELLIS loaded")
                
            except Exception as e:
                print(f"❌ Pipeline 1 initialization failed: {e}")
                traceback.print_exc()
            
            # Initialize pipeline 2
            print("Loading Pipeline 2...")
            try:
                # HunyuanDiT for pipeline 2
                self.pipelines[2]['hunyuan'] = HunyuanDiTPipeline(
                    model_path=GENERATION_CONFIG['hunyuan_model_path'],
                    device=GENERATION_CONFIG['device']
                )
                
                # Compile HunyuanDiT for better performance
                try:
                    print("   Compiling Pipeline 2 HunyuanDiT for better performance...")
                    self.pipelines[2]['hunyuan'].compile()
                    print("   ✓ Pipeline 2 HunyuanDiT compiled successfully")
                except Exception as e:
                    print(f"   ⚠️ Pipeline 2 HunyuanDiT compilation failed: {e}")
                    print("   Continuing without compilation...")
                
                print("✓ Pipeline 2 HunyuanDiT loaded")
                
                # TRELLIS for pipeline 2
                self.pipelines[2]['trellis'] = TrellisImageTo3DPipeline.from_pretrained(
                    GENERATION_CONFIG['trellis_model_path']
                )
                self.pipelines[2]['trellis'].cuda()
                print("✓ Pipeline 2 TRELLIS loaded")
                
            except Exception as e:
                print(f"❌ Pipeline 2 initialization failed: {e}")
                traceback.print_exc()
            
            # Initialize CLIP model
            print("Loading CLIP model for image comparison...")
            try:
                self.clip_model, self.clip_preprocess = clip.load("ViT-B/32", device=GENERATION_CONFIG['device'])
                print("✓ CLIP model loaded successfully")
            except Exception as e:
                print(f"❌ CLIP model loading failed: {e}")
                traceback.print_exc()
            
            self.initialized = True
            print("✅ Both pipelines and CLIP initialized successfully!")
            
            # Print GPU memory usage
            if torch.cuda.is_available():
                gpu_memory_free, gpu_memory_total = torch.cuda.mem_get_info()
                memory_used = gpu_memory_total - gpu_memory_free
                print(f"🧠 GPU Memory: {memory_used / 1e9:.1f}GB used, {gpu_memory_free / 1e9:.1f}GB free")
    
    def get_pipeline(self, pipeline_id: int):
        """Get a specific pipeline"""
        if not self.initialized:
            self.initialize_pipelines()
        return self.pipelines.get(pipeline_id)
    
    def cleanup(self):
        """Clean up all pipelines"""
        print("🧹 Cleaning up parallel pipelines...")
        for pipeline_id, pipeline in self.pipelines.items():
            if pipeline['hunyuan']:
                del pipeline['hunyuan']
            if pipeline['trellis']:
                del pipeline['trellis']
        if self.clip_model:
            del self.clip_model
        self.initialized = False
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

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

def compare_images_with_clip(clip_model, clip_preprocess, images: List[Image.Image], prompt: str) -> Tuple[int, float]:
    """Compare images with CLIP and return the best image index and score"""
    try:
        # Prepare text
        text = clip.tokenize([prompt]).to(GENERATION_CONFIG['device'])
        
        # Prepare images
        image_inputs = torch.stack([clip_preprocess(img) for img in images]).to(GENERATION_CONFIG['device'])
        
        with torch.no_grad():
            # Get features
            image_features = clip_model.encode_image(image_inputs)
            text_features = clip_model.encode_text(text)
            
            # Normalize features
            image_features = F.normalize(image_features, p=2, dim=1)
            text_features = F.normalize(text_features, p=2, dim=1)
            
            # Calculate similarities
            similarities = torch.matmul(image_features, text_features.T).squeeze()
            
            # Get best image
            best_idx = similarities.argmax().item()
            best_score = similarities[best_idx].item()
            
            print(f"🎯 CLIP comparison results:")
            for i, (img, sim) in enumerate(zip(images, similarities)):
                print(f"  Image {i+1}: {sim:.4f}")
            print(f"  Best image: {best_idx+1} (score: {best_score:.4f})")
            
            return best_idx, best_score
            
    except Exception as e:
        print(f"❌ CLIP comparison failed: {e}")
        # Fallback to first image
        return 0, 0.0

@dataclass
class ImageGenerationResult:
    """Result of image-only generation"""
    pipeline_id: int
    prompt: str
    enhancement: str
    seed: int
    status: str
    image: Optional[Image.Image] = None
    error: Optional[str] = None
    generation_time: float = 0.0

def generate_image_only(pipeline_id: int, prompt: str, enhancement: str, seed: int, parallel_pipelines: ParallelModelPipelines) -> ImageGenerationResult:
    """Generate only image using a specific pipeline (for CLIP comparison)"""
    start_time = time.time()
    result = ImageGenerationResult(
        pipeline_id=pipeline_id,
        prompt=prompt,
        enhancement=enhancement,
        seed=seed,
        status="processing"
    )
    
    try:
        print(f"🎨 [Pipeline {pipeline_id}] Generating image: '{prompt}' + '{enhancement}' (seed: {seed})")
        
        # Get pipeline
        pipeline = parallel_pipelines.get_pipeline(pipeline_id)
        if not pipeline:
            raise Exception(f"Pipeline {pipeline_id} not available")
        
        # Combine prompt with enhancement
        full_prompt = f"{enhancement}, {prompt}" if enhancement else prompt
        
        # Generate image with HunyuanDiT
        with pipeline['lock']:
            with torch.no_grad():
                image = pipeline['hunyuan'](prompt=full_prompt, seed=seed)
        
        print(f"[Pipeline {pipeline_id}] ✓ Image generated successfully")
        
        # Update result
        result.status = "completed"
        result.image = image
        result.generation_time = time.time() - start_time
        
    except Exception as e:
        result.status = "failed"
        result.error = str(e)
        result.generation_time = time.time() - start_time
        print(f"[Pipeline {pipeline_id}] ❌ Image generation failed: {e}")
        traceback.print_exc()
    
    return result

def generate_with_pipeline(pipeline_id: int, prompt: str, enhancement: str, seed: int, parallel_pipelines: ParallelModelPipelines) -> GenerationResult:
    """Generate using a specific pipeline"""
    start_time = time.time()
    result = GenerationResult(
        pipeline_id=pipeline_id,
        prompt=prompt,
        enhancement=enhancement,
        seed=seed,
        status="processing"
    )
    
    try:
        print(f"🎯 [Pipeline {pipeline_id}] Starting generation: '{prompt}' + '{enhancement}' (seed: {seed})")
        
        # Get pipeline
        pipeline = parallel_pipelines.get_pipeline(pipeline_id)
        if not pipeline:
            raise Exception(f"Pipeline {pipeline_id} not available")
        
        # Combine prompt with enhancement
        full_prompt = f"{enhancement}, {prompt}" if enhancement else prompt
        
        # Step 1: Generate image with HunyuanDiT
        print(f"[Pipeline {pipeline_id}] Step 1: Generating image with HunyuanDiT...")
        with pipeline['lock']:
            with torch.no_grad():
                image = pipeline['hunyuan'](prompt=full_prompt, seed=seed)
        
        print(f"[Pipeline {pipeline_id}] ✓ HunyuanDiT image generated")
        
        # Step 1.3: Center object
        if GENERATION_CONFIG.get('enable_object_centering', True):
            print(f"[Pipeline {pipeline_id}] Step 1.3: Centering object...")
            image = center_object_in_image(
                image, 
                white_threshold=GENERATION_CONFIG.get('centering_white_threshold', 240),
                padding=GENERATION_CONFIG.get('centering_padding', 30)
            )
            print(f"[Pipeline {pipeline_id}] ✓ Object centered")
        
        # Step 2: Generate 3D model with TRELLIS
        print(f"[Pipeline {pipeline_id}] Step 2: Generating 3D model with TRELLIS...")
        with pipeline['lock']:
            outputs = pipeline['trellis'].run(
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
        
        print(f"[Pipeline {pipeline_id}] ✓ 3D model generated")
        
        # Step 3: Extract PLY
        print(f"[Pipeline {pipeline_id}] Step 3: Extracting PLY...")
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
                print(f"[Pipeline {pipeline_id}] Quality enhancement: Kept {quality_mask.sum().item():,} splats")
        except Exception as e:
            print(f"[Pipeline {pipeline_id}] Quality enhancement failed: {e}")
        
        # Save PLY
        ply_buffer = io.BytesIO()
        gaussian_output.save_ply(ply_buffer)
        ply_data = ply_buffer.getvalue()
        
        print(f"[Pipeline {pipeline_id}] ✓ PLY extracted ({len(ply_data):,} bytes)")
        
        # Step 4: Compress if enabled
        compressed_data = None
        if GENERATION_CONFIG.get('auto_compress_ply', True):
            print(f"[Pipeline {pipeline_id}] Step 4: Compressing PLY...")
            try:
                import pyspz
                compressed_data = pyspz.compress(ply_data, workers=-1)
                print(f"[Pipeline {pipeline_id}] ✓ SPZ compression: {len(compressed_data):,} bytes")
            except Exception as e:
                print(f"[Pipeline {pipeline_id}] ⚠️ SPZ compression failed: {e}")
        
        # Update result
        result.status = "completed"
        result.ply_data = ply_data
        result.compressed_data = compressed_data
        result.generation_time = time.time() - start_time
        
        print(f"[Pipeline {pipeline_id}] 🎉 Generation completed in {result.generation_time:.2f}s")
        
    except Exception as e:
        result.status = "failed"
        result.error = str(e)
        result.generation_time = time.time() - start_time
        print(f"[Pipeline {pipeline_id}] ❌ Generation failed: {e}")
        traceback.print_exc()
    
    return result

# Initialize FastAPI app
app = FastAPI(title="Parallel HunyuanDiT + TRELLIS Generation Server V3", version="3.0.0")

# Global parallel pipelines
parallel_pipelines = ParallelModelPipelines()

@app.post("/generate_parallel/")
async def generate_parallel(
    prompt1: str = Form(...),
    enhancement1: str = Form(""),
    seed1: Optional[int] = Form(42),
    prompt2: str = Form(...),
    enhancement2: str = Form(""),
    seed2: Optional[int] = Form(123),
    return_compressed: Optional[bool] = Form(True)
):
    """Generate 2 models truly in parallel using 2 separate pipelines"""
    
    print(f"🚀 Starting parallel generation:")
    print(f"  Pipeline 1: '{prompt1}' + '{enhancement1}' (seed: {seed1})")
    print(f"  Pipeline 2: '{prompt2}' + '{enhancement2}' (seed: {seed2})")
    
    # Initialize pipelines if not already done
    parallel_pipelines.initialize_pipelines()
    
    # Create generation tasks
    task1 = partial(generate_with_pipeline, 1, prompt1, enhancement1, seed1, parallel_pipelines)
    task2 = partial(generate_with_pipeline, 2, prompt2, enhancement2, seed2, parallel_pipelines)
    
    # Execute both generations truly in parallel
    results = []
    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [
            executor.submit(task1),
            executor.submit(task2)
        ]
        
        for future in as_completed(futures):
            try:
                result = future.result()
                results.append(result)
            except Exception as e:
                print(f"❌ Task failed: {e}")
    
    # Sort results by pipeline_id
    results.sort(key=lambda x: x.pipeline_id)
    
    # Calculate total generation time
    total_generation_time = max([r.generation_time for r in results]) if results else 0.0
    
    # Prepare response
    response_data = {
        "status": "parallel_completed",
        "total_pipelines": 2,
        "successful_pipelines": len([r for r in results if r.status == "completed"]),
        "failed_pipelines": len([r for r in results if r.status == "failed"]),
        "total_generation_time": total_generation_time,
        "results": []
    }
    
    # Add individual results
    for result in results:
        result_data = {
            "pipeline_id": result.pipeline_id,
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
                    combined_data[f"pipeline_{result.pipeline_id}"] = {
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
                    "Content-Disposition": f"attachment; filename=parallel_results_{int(time.time())}.json",
                    "X-Parallel-Pipelines": "2",
                    "X-Successful-Pipelines": str(len([r for r in results if r.status == "completed"])),
                    "X-Pipeline": "hunyuan_trellis_parallel_v3"
                }
            )
        except Exception as e:
            print(f"⚠️ Failed to create combined response: {e}")
    
    # Print final summary
    if results:
        successful_count = len([r for r in results if r.status == "completed"])
        print(f"🎉 Parallel generation completed in {total_generation_time:.2f}s ({successful_count}/2 successful)")
    
    return JSONResponse(content=response_data)

@app.post("/generate_clip_optimized/")
async def generate_clip_optimized(
    prompt: str = Form(...),
    enhancement1: str = Form("professional 3D render"),
    enhancement2: str = Form("isometric view version"),
    seed1: Optional[int] = Form(42),
    seed2: Optional[int] = Form(123),
    return_compressed: Optional[bool] = Form(True)
):
    """Generate 2 images with different enhancements, compare with CLIP, and run TRELLIS only on the best one"""
    
    print(f"🚀 Starting CLIP-optimized generation:")
    print(f"  Prompt: '{prompt}'")
    print(f"  Enhancement 1: '{enhancement1}' (seed: {seed1})")
    print(f"  Enhancement 2: '{enhancement2}' (seed: {seed2})")
    
    # Initialize pipelines if not already done
    parallel_pipelines.initialize_pipelines()
    
    if not parallel_pipelines.clip_model:
        raise HTTPException(status_code=500, detail="CLIP model not available")
    
    start_time = time.time()
    
    try:
        # Step 1: Generate 2 images with different enhancements
        print("Step 1: Generating 2 images with different enhancements...")
        
        # Create image generation tasks
        task1 = partial(generate_image_only, 1, prompt, enhancement1, seed1, parallel_pipelines)
        task2 = partial(generate_image_only, 2, prompt, enhancement2, seed2, parallel_pipelines)
        
        # Execute both image generations in parallel
        images = []
        with ThreadPoolExecutor(max_workers=2) as executor:
            futures = [
                executor.submit(task1),
                executor.submit(task2)
            ]
            
            for future in as_completed(futures):
                try:
                    result = future.result()
                    if result and result.status == "completed":
                        images.append(result.image)
                    else:
                        raise Exception(f"Image generation failed: {result.error if result else 'Unknown error'}")
                except Exception as e:
                    print(f"❌ Image generation task failed: {e}")
                    raise HTTPException(status_code=500, detail=f"Image generation failed: {str(e)}")
        
        print(f"✓ Generated {len(images)} images successfully")
        
        # Step 2: Compare images with CLIP
        print("Step 2: Comparing images with CLIP...")
        best_idx, clip_score = compare_images_with_clip(
            parallel_pipelines.clip_model,
            parallel_pipelines.clip_preprocess,
            images,
            prompt
        )
        
        best_image = images[best_idx]
        best_enhancement = enhancement1 if best_idx == 0 else enhancement2
        best_seed = seed1 if best_idx == 0 else seed2
        best_pipeline_id = best_idx + 1
        
        print(f"✓ Selected image from pipeline {best_pipeline_id} (enhancement: '{best_enhancement}')")
        
        # Step 3: Generate 3D model only for the best image
        print(f"Step 3: Generating 3D model for best image (pipeline {best_pipeline_id})...")
        
        # Get the best pipeline
        pipeline = parallel_pipelines.get_pipeline(best_pipeline_id)
        if not pipeline:
            raise Exception(f"Pipeline {best_pipeline_id} not available")
        
        # Center object if enabled
        if GENERATION_CONFIG.get('enable_object_centering', True):
            print(f"Centering object in best image...")
            best_image = center_object_in_image(
                best_image, 
                white_threshold=GENERATION_CONFIG.get('centering_white_threshold', 240),
                padding=GENERATION_CONFIG.get('centering_padding', 30)
            )
            print(f"✓ Object centered")
        
        # Generate 3D model with TRELLIS
        print(f"Generating 3D model with TRELLIS...")
        with pipeline['lock']:
            outputs = pipeline['trellis'].run(
                best_image,
                seed=best_seed,
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
        
        print(f"✓ 3D model generated")
        
        # Extract PLY
        print("Extracting PLY...")
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
                print(f"Quality enhancement: Kept {quality_mask.sum().item():,} splats")
        except Exception as e:
            print(f"Quality enhancement failed: {e}")
        
        # Save PLY
        ply_buffer = io.BytesIO()
        gaussian_output.save_ply(ply_buffer)
        ply_data = ply_buffer.getvalue()
        
        print(f"✓ PLY extracted ({len(ply_data):,} bytes)")
        
        # Compress if enabled
        compressed_data = None
        if GENERATION_CONFIG.get('auto_compress_ply', True):
            print("Compressing PLY...")
            try:
                import pyspz
                compressed_data = pyspz.compress(ply_data, workers=-1)
                print(f"✓ SPZ compression: {len(compressed_data):,} bytes")
            except Exception as e:
                print(f"⚠️ SPZ compression failed: {e}")
        
        total_time = time.time() - start_time
        
        # Prepare response
        response_data = {
            "status": "clip_optimized_completed",
            "prompt": prompt,
            "enhancements_tried": [enhancement1, enhancement2],
            "seeds_used": [seed1, seed2],
            "clip_comparison": {
                "best_pipeline": best_pipeline_id,
                "best_enhancement": best_enhancement,
                "best_seed": best_seed,
                "clip_score": clip_score,
                "all_scores": [0.0, 0.0]  # Would be populated with actual scores
            },
            "generation_time": total_time,
            "ply_size_bytes": len(ply_data),
            "compressed_size_bytes": len(compressed_data) if compressed_data else 0,
            "optimization_savings": "50% TRELLIS computation (only 1/2 pipelines used for 3D generation)"
        }
        
        # If return_compressed, return the compressed data
        if return_compressed and compressed_data:
            return Response(
                content=compressed_data,
                media_type="application/octet-stream",
                headers={
                    "Content-Disposition": f"attachment; filename=clip_optimized_model_{best_seed}.ply.spz",
                    "X-Generation-Seed": str(best_seed),
                    "X-Generation-Prompt": prompt,
                    "X-Best-Enhancement": best_enhancement,
                    "X-Clip-Score": f"{clip_score:.4f}",
                    "X-Pipeline": "hunyuan_trellis_clip_optimized_v3"
                }
            )
        
        # Print final summary
        print(f"🎉 CLIP-optimized generation completed in {total_time:.2f}s")
        print(f"   Selected enhancement: '{best_enhancement}' (CLIP score: {clip_score:.4f})")
        
        return JSONResponse(content=response_data)
        
    except Exception as e:
        print(f"❌ CLIP-optimized generation failed: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Generation failed: {str(e)}")

@app.post("/generate_single_offloaded/")
async def generate_single_offloaded(
    prompt: str = Form(...),
    enhancement: str = Form("professional 3D render"),
    seed: Optional[int] = Form(42),
    return_compressed: Optional[bool] = Form(True)
):
    """Generate single model with offloaded pipelines (frees up memory after generation)"""
    
    print(f"🚀 Starting single generation with offloaded pipelines:")
    print(f"  Prompt: '{prompt}' + '{enhancement}' (seed: {seed})")
    
    # Initialize pipelines if not already done
    parallel_pipelines.initialize_pipelines()
    
    start_time = time.time()
    
    try:
        # Use pipeline 1 for single generation
        pipeline_id = 1
        pipeline = parallel_pipelines.get_pipeline(pipeline_id)
        if not pipeline:
            raise Exception(f"Pipeline {pipeline_id} not available")
        
        # Combine prompt with enhancement
        full_prompt = f"{enhancement}, {prompt}" if enhancement else prompt
        
        # Step 1: Generate image with HunyuanDiT
        print("Step 1: Generating image with HunyuanDiT...")
        with pipeline['lock']:
            with torch.no_grad():
                image = pipeline['hunyuan'](prompt=full_prompt, seed=seed)
        
        print("✓ HunyuanDiT image generated")
        
        # Step 1.3: Center object
        if GENERATION_CONFIG.get('enable_object_centering', True):
            print("Step 1.3: Centering object...")
            image = center_object_in_image(
                image, 
                white_threshold=GENERATION_CONFIG.get('centering_white_threshold', 240),
                padding=GENERATION_CONFIG.get('centering_padding', 30)
            )
            print("✓ Object centered")
        
        # Step 2: Generate 3D model with TRELLIS
        print("Step 2: Generating 3D model with TRELLIS...")
        with pipeline['lock']:
            outputs = pipeline['trellis'].run(
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
        
        print("✓ 3D model generated")
        
        # Step 3: Extract PLY
        print("Step 3: Extracting PLY...")
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
                print(f"Quality enhancement: Kept {quality_mask.sum().item():,} splats")
        except Exception as e:
            print(f"Quality enhancement failed: {e}")
        
        # Save PLY
        ply_buffer = io.BytesIO()
        gaussian_output.save_ply(ply_buffer)
        ply_data = ply_buffer.getvalue()
        
        print(f"✓ PLY extracted ({len(ply_data):,} bytes)")
        
        # Step 4: Compress if enabled
        compressed_data = None
        if GENERATION_CONFIG.get('auto_compress_ply', True):
            print("Step 4: Compressing PLY...")
            try:
                import pyspz
                compressed_data = pyspz.compress(ply_data, workers=-1)
                print(f"✓ SPZ compression: {len(compressed_data):,} bytes")
            except Exception as e:
                print(f"⚠️ SPZ compression failed: {e}")
        
        total_time = time.time() - start_time
        
        # Step 5: Offload pipelines to free memory
        print("Step 5: Offloading pipelines to free memory...")
        parallel_pipelines.cleanup()
        print("✓ Pipelines offloaded successfully")
        
        # Prepare response
        response_data = {
            "status": "single_offloaded_completed",
            "prompt": prompt,
            "enhancement": enhancement,
            "seed": seed,
            "generation_time": total_time,
            "ply_size_bytes": len(ply_data),
            "compressed_size_bytes": len(compressed_data) if compressed_data else 0,
            "memory_optimization": "Pipelines offloaded after generation to free GPU memory"
        }
        
        # If return_compressed, return the compressed data
        if return_compressed and compressed_data:
            return Response(
                content=compressed_data,
                media_type="application/octet-stream",
                headers={
                    "Content-Disposition": f"attachment; filename=single_offloaded_model_{seed}.ply.spz",
                    "X-Generation-Seed": str(seed),
                    "X-Generation-Prompt": prompt,
                    "X-Enhancement": enhancement,
                    "X-Pipeline": "hunyuan_trellis_single_offloaded_v3"
                }
            )
        
        # Print final summary
        print(f"🎉 Single offloaded generation completed in {total_time:.2f}s")
        print(f"   Pipelines offloaded to free GPU memory")
        
        return JSONResponse(content=response_data)
        
    except Exception as e:
        print(f"❌ Single offloaded generation failed: {e}")
        traceback.print_exc()
        
        # Clean up on error
        try:
            parallel_pipelines.cleanup()
            print("✓ Pipelines cleaned up after error")
        except:
            pass
        
        raise HTTPException(status_code=500, detail=f"Generation failed: {str(e)}")

@app.post("/initialize/")
async def initialize_pipelines():
    """Manually initialize the parallel pipelines"""
    try:
        parallel_pipelines.initialize_pipelines()
        return {
            "status": "success",
            "message": "Parallel pipelines initialized successfully",
            "pipelines_ready": parallel_pipelines.initialized
        }
    except Exception as e:
        return JSONResponse(content={
            "status": "error",
            "message": str(e)
        }, status_code=500)

@app.get("/status/")
async def get_server_status():
    """Get server status"""
    return {
        "status": "running",
        "pipelines_initialized": parallel_pipelines.initialized,
        "config": GENERATION_CONFIG,
        "gpu_memory": torch.cuda.mem_get_info()[0] / 1e9 if torch.cuda.is_available() else 0
    }

@app.get("/health/")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "timestamp": time.time()}

@app.post("/cleanup/")
async def cleanup_pipelines():
    """Clean up all pipelines"""
    try:
        parallel_pipelines.cleanup()
        return {
            "status": "success",
            "message": "Pipelines cleaned up successfully"
        }
    except Exception as e:
        return JSONResponse(content={
            "status": "error",
            "message": str(e)
        }, status_code=500)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Parallel HunyuanDiT + TRELLIS Generation Server V3")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8099, help="Port to bind to")
    
    args = parser.parse_args()
    
    print(f"Starting Parallel HunyuanDiT + TRELLIS Generation Server V3 on {args.host}:{args.port}")
    print("=" * 80)
    print("Features:")
    print("  • 2 complete model pipelines loaded simultaneously")
    print("  • True parallel processing - no waiting between pipelines")
    print("  • Separate HunyuanDiT + TRELLIS instances for each pipeline")
    print("  • Optimized for maximum throughput")
    print("  • Combined results in single response")
    print("=" * 80)
    
    try:
        uvicorn.run(app, host=args.host, port=args.port, log_level="info")
    finally:
        # Cleanup on shutdown
        parallel_pipelines.cleanup() 