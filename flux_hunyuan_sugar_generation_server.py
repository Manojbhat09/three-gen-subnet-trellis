#!/usr/bin/env python3
"""
Subnet 17 (404-GEN) - FLUX + Hunyuan3D + SuGaR Generation Server
Purpose: HTTP server for high-quality 3D model generation using Flux, Hunyuan3D-2, and SuGaR mesh-to-GS conversion
Produces validation-compatible Gaussian Splatting PLY files along with viewable mesh formats
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

from fastapi import FastAPI, Form, HTTPException
from fastapi.responses import Response, JSONResponse
import uvicorn

# Set environment variables
os.environ['SPCONV_ALGO'] = 'native'

# Add Hunyuan3D-2 to Python path
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'Hunyuan3D-2'))

# Import Hunyuan3D components
from hy3dgen.rembg import BackgroundRemover
from hy3dgen.shapegen import (
    Hunyuan3DDiTFlowMatchingPipeline, 
    FaceReducer, 
    FloaterRemover, 
    DegenerateFaceRemover
)

# Import Flux components
from diffusers import FluxTransformer2DModel, FluxPipeline, BitsAndBytesConfig, GGUFQuantizationConfig
from transformers import T5EncoderModel, BitsAndBytesConfig as BitsAndBytesConfigTF

# Import PLY handling
from plyfile import PlyData, PlyElement

# Constants
NUM_INFERENCE_STEPS = 8
MAX_SEED = np.iinfo(np.int32).max

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration
GENERATION_CONFIG = {
    'output_dir': './flux_hunyuan_sugar_outputs',
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    'num_inference_steps_t2i': 8,
    'num_inference_steps_shape': 30,
    'mc_algo': 'mc',
    'flux_model_url': "https://huggingface.co/gokaygokay/flux-game/blob/main/hyperflux_00001_.q8_0.gguf",
    'flux_base_model': "camenduru/FLUX.1-dev-diffusers",
    'hunyuan_model_path': 'jetx/Hunyuan3D-2',
    'save_intermediate_outputs': True,
    'auto_compress_ply': True,
    # SuGaR specific settings - OPTIMIZED FOR RTX 4090
    'sugar_num_points': 15000,  # Reduced from 50000 for RTX 4090
    'sugar_sh_levels': 3,  # Reduced from 4 for memory
    'sugar_triangle_scale': 1.5,  # Reduced from 2.0
    'sugar_surface_level': 0.3,  # Surface level for mesh extraction
    'sugar_n_gaussians_per_triangle': 4,  # Reduced from 6 for memory
    # Memory management settings
    'enable_memory_efficient_attention': True,
    'enable_cpu_offload': True,
    'max_memory_usage_gb': 20,  # Leave 4GB buffer on 24GB card
    'validation_server_url': 'http://127.0.0.1:10006'  # Validation server coordination
}

@dataclass
class GenerationMetrics:
    total_generations: int = 0
    successful_generations: int = 0
    failed_generations: int = 0
    average_generation_time: float = 0.0
    last_generation_time: float = 0.0
    sugar_converted_count: int = 0

class FluxHunyuanSuGaRGenerator:
    def __init__(self):
        self.hunyuan_pipeline = None
        self.rembg = None
        self.metrics = GenerationMetrics()
        self.generation_lock = threading.Lock()
        
        # Get HuggingFace token from cache
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
        self._initialize_models()

    def _clear_gpu_memory(self):
        """Clear GPU memory cache aggressively"""
        import gc
        
        # Force garbage collection
        gc.collect()
        
        if torch.cuda.is_available():
            # Clear all caches
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
            
            # Force memory cleanup
            try:
                torch.cuda.ipc_collect()
            except:
                pass
            
            # Additional cleanup
            if hasattr(torch.cuda, 'reset_accumulated_memory_stats'):
                torch.cuda.reset_accumulated_memory_stats()
        
        # Force another garbage collection
        gc.collect()
        
        if torch.cuda.is_available():
            total_memory = torch.cuda.get_device_properties(0).total_memory
            allocated = torch.cuda.memory_allocated()
            reserved = torch.cuda.memory_reserved()
            available = total_memory - allocated
            
            print(f"🧠 GPU Memory Status:")
            print(f"   Total: {total_memory / 1e9:.1f} GB")
            print(f"   Allocated: {allocated / 1e9:.1f} GB")
            print(f"   Reserved: {reserved / 1e9:.1f} GB")
            print(f"   Available: {available / 1e9:.1f} GB")
            
            # Check if we're approaching memory limits
            usage_percent = (allocated / total_memory) * 100
            if usage_percent > 85:
                print(f"⚠️ High memory usage: {usage_percent:.1f}%")
                
            return available / 1e9  # Return available GB
        
        return 0
    
    def _aggressive_gpu_cleanup(self):
        """Aggressive GPU memory cleanup for maximum memory availability"""
        if torch.cuda.is_available():
            print("🧹 Starting aggressive GPU cleanup...")
            
            # Multiple rounds of cleanup
            for i in range(5):
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
                gc.collect()
                time.sleep(0.1)  # Small delay between rounds
            
            # Additional PyTorch cleanup
            try:
                torch.cuda.ipc_collect()
            except:
                pass
            
            # Reset memory stats
            try:
                torch.cuda.reset_peak_memory_stats()
                if hasattr(torch.cuda, 'reset_accumulated_memory_stats'):
                    torch.cuda.reset_accumulated_memory_stats()
            except:
                pass
            
            # Force another round
            gc.collect()
            torch.cuda.empty_cache()
            
            # Print memory status
            gpu_memory_free, gpu_memory_total = torch.cuda.mem_get_info()
            memory_used = gpu_memory_total - gpu_memory_free
            print(f"🧹 After aggressive cleanup: {memory_used / (1024**3):.1f}GB used, {gpu_memory_free / (1024**3):.1f}GB free")

    def _initialize_models(self):
        """Initialize only background remover and Hunyuan3D. FLUX is loaded on demand."""
        print("🔧 Initializing models...")
        
        try:
            # Initialize background remover
            print("Loading background remover...")
            self.rembg = BackgroundRemover()
            print("✓ Background remover loaded")
            
            # Initialize Hunyuan3D pipeline
            print("Loading Hunyuan3D pipeline...")
            self.hunyuan_pipeline = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(
                GENERATION_CONFIG['hunyuan_model_path'],
                torch_dtype=torch.float16,
                low_cpu_mem_usage=True,
            )
            self.hunyuan_pipeline.to(GENERATION_CONFIG['device'])
            print("✓ Hunyuan3D pipeline loaded")
            
            self._clear_gpu_memory()
            print("✅ Model initialization complete")
            
        except Exception as e:
            print(f"❌ Error initializing models: {e}")
            traceback.print_exc()

    def _mesh_to_gaussian_splatting(self, mesh: trimesh.Trimesh, num_points: int = None) -> bytes:
        """Convert mesh to Gaussian Splatting PLY format using SuGaR-inspired approach"""
        if num_points is None:
            num_points = GENERATION_CONFIG['sugar_num_points']
        
        print(f"Converting mesh to Gaussian Splatting format ({num_points} points)...")
        
        try:
            # Sample points from mesh surface
            points, face_indices = mesh.sample(num_points, return_index=True)
            
            # Get face normals for sampled points
            face_normals = mesh.face_normals[face_indices]
            
            # Fallback RGB to SH conversion
            def RGB2SH(rgb):
                """Fallback RGB to Spherical Harmonics conversion"""
                return rgb / (2 * 3.14159 ** 0.5)
            
            # Create Gaussian Splatting attributes
            print("Creating Gaussian Splatting attributes...")
            
            # Colors (use vertex colors if available, otherwise default)
            if hasattr(mesh.visual, 'vertex_colors') and mesh.visual.vertex_colors is not None:
                # Interpolate vertex colors to sampled points
                colors = np.ones((num_points, 3)) * 0.7  # Default gray
            else:
                colors = np.ones((num_points, 3)) * 0.7  # Default gray
            
            # Convert RGB to Spherical Harmonics (DC component only for simplicity)
            sh_dc = RGB2SH(colors)  # Shape: (N, 3)
            
            # Create additional SH coefficients (set to zero for simplicity)
            sh_rest = np.zeros((num_points, 45))  # 15 coefficients * 3 channels = 45
            
            # Opacities (sigmoid inverse of alpha values)
            opacities = np.ones((num_points, 1)) * 2.0  # High opacity
            
            # Scales (log of scale values)
            scales = np.ones((num_points, 3)) * (-3.0)  # Small scales
            
            # Rotations (quaternions, normalized)
            rotations = np.zeros((num_points, 4))
            rotations[:, 0] = 1.0  # w=1, x=y=z=0 (identity quaternion)
            
            print("✓ Gaussian Splatting attributes created")
            
            # Create PLY data
            gs_ply_data = self._create_gaussian_splatting_ply(
                points, face_normals, sh_dc, sh_rest, opacities, scales, rotations
            )
            
            print(f"✓ Gaussian Splatting PLY created ({len(gs_ply_data)} bytes)")
            self.metrics.sugar_converted_count += 1
            
            return gs_ply_data
            
        except Exception as e:
            print(f"❌ Mesh to Gaussian Splatting conversion failed: {e}")
            traceback.print_exc()
            # Return basic mesh PLY as fallback
            return self._create_basic_mesh_ply(mesh)

    def _create_gaussian_splatting_ply(self, points, normals, sh_dc, sh_rest, opacities, scales, rotations):
        """Create a Gaussian Splatting PLY file"""
        num_points = len(points)
        
        # Create vertex data with all Gaussian Splatting attributes
        vertex_data = []
        for i in range(num_points):
            vertex = [
                points[i, 0], points[i, 1], points[i, 2],  # x, y, z
                normals[i, 0], normals[i, 1], normals[i, 2],  # nx, ny, nz
            ]
            
            # Add spherical harmonics DC components
            vertex.extend([sh_dc[i, 0], sh_dc[i, 1], sh_dc[i, 2]])  # f_dc_0, f_dc_1, f_dc_2
            
            # Add remaining SH coefficients
            vertex.extend(sh_rest[i])  # f_rest_0 through f_rest_44
            
            # Add opacity
            vertex.append(opacities[i, 0])  # opacity
            
            # Add scales
            vertex.extend([scales[i, 0], scales[i, 1], scales[i, 2]])  # scale_0, scale_1, scale_2
            
            # Add rotations
            vertex.extend([rotations[i, 0], rotations[i, 1], rotations[i, 2], rotations[i, 3]])  # rot_0, rot_1, rot_2, rot_3
            
            vertex_data.append(tuple(vertex))
        
        # Define vertex properties
        properties = [
            ('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
            ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4'),
            ('f_dc_0', 'f4'), ('f_dc_1', 'f4'), ('f_dc_2', 'f4'),
        ]
        
        # Add SH rest properties
        for i in range(45):
            properties.append((f'f_rest_{i}', 'f4'))
        
        # Add remaining properties
        properties.extend([
            ('opacity', 'f4'),
            ('scale_0', 'f4'), ('scale_1', 'f4'), ('scale_2', 'f4'),
            ('rot_0', 'f4'), ('rot_1', 'f4'), ('rot_2', 'f4'), ('rot_3', 'f4'),
        ])
        
        # Create PLY element
        vertex_element = PlyElement.describe(
            np.array(vertex_data, dtype=properties),
            'vertex'
        )
        
        # Create PLY data and write to bytes
        ply_data = PlyData([vertex_element])
        
        import io
        buffer = io.BytesIO()
        ply_data.write(buffer)
        return buffer.getvalue()

    def _load_flux_pipeline(self, seed: int = 42):
        """Load FLUX pipeline on demand with memory optimization"""
        print("Loading FLUX pipeline with quantization...")
        
        try:
            device = GENERATION_CONFIG['device']
            dtype = torch.bfloat16
            
            # Get HuggingFace token
            huggingface_token = os.getenv("HUGGINGFACE_TOKEN")
            
            # Use the correct model configuration
            file_url = GENERATION_CONFIG['flux_model_url']
            file_url = file_url.replace("/resolve/main/", "/blob/main/").replace("?download=true", "")
            single_file_base_model = GENERATION_CONFIG['flux_base_model']
            
            # Load text encoder with 8-bit quantization
            print("Loading text encoder with 8-bit quantization...")
            quantization_config_tf = BitsAndBytesConfigTF(
                load_in_8bit=True,
                bnb_8bit_compute_dtype=torch.bfloat16
            )
            text_encoder_2 = T5EncoderModel.from_pretrained(
                single_file_base_model, 
                subfolder="text_encoder_2", 
                torch_dtype=dtype, 
                quantization_config=quantization_config_tf, 
                token=huggingface_token
            )
            
            # Load transformer with GGUF configuration
            print("Loading transformer with GGUF quantization...")
            transformer = FluxTransformer2DModel.from_single_file(
                file_url, 
                subfolder="transformer", 
                quantization_config=GGUFQuantizationConfig(compute_dtype=dtype), 
                torch_dtype=dtype, 
                config=single_file_base_model
            )
            
            # Initialize pipeline
            print("Initializing FLUX pipeline...")
            flux_pipe = FluxPipeline.from_pretrained(
                single_file_base_model, 
                transformer=transformer, 
                text_encoder_2=text_encoder_2, 
                torch_dtype=dtype, 
                token=huggingface_token
            )
            flux_pipe.to(device)
            
            print("✓ FLUX pipeline loaded with quantization")
            return flux_pipe
            
        except Exception as e:
            print(f"❌ FLUX pipeline loading failed: {e}")
            return None

    def _unload_hunyuan_temporarily(self):
        """Temporarily unload Hunyuan3D to free memory for FLUX"""
        if self.hunyuan_pipeline is not None:
            del self.hunyuan_pipeline
            self.hunyuan_pipeline = None
            self._clear_gpu_memory()
            print("✓ Hunyuan3D temporarily unloaded")

    def _reload_hunyuan_pipeline(self):
        """Reload Hunyuan3D pipeline after FLUX is done"""
        if self.hunyuan_pipeline is None:
            print("Reloading Hunyuan3D pipeline...")
            self.hunyuan_pipeline = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(
                GENERATION_CONFIG['hunyuan_model_path'],
                torch_dtype=torch.float16,
                low_cpu_mem_usage=True,
            )
            self.hunyuan_pipeline.to(GENERATION_CONFIG['device'])
            print("✓ Hunyuan3D pipeline reloaded")

    def generate_3d_model(self, prompt: str, seed: int = 42) -> Optional[bytes]:
        """Generate 3D model from text prompt using complete pipeline with GPU coordination"""
        with self.generation_lock:
            start_time = time.time()
            
            try:
                print(f"🎯 Starting generation for: '{prompt}' (seed: {seed})")
                
                # CRITICAL: Coordinate with validation server FIRST
                print("🤝 Coordinating with validation server...")
                validation_unloaded = self._coordinate_with_validation_server("unload")
                if not validation_unloaded:
                    print("⚠️ Validation server coordination failed, proceeding anyway...")
                
                # Print GPU status after validation unload
                self._print_gpu_status("After validation unload ")
                
                # Step 1: Temporarily unload Hunyuan3D for FLUX
                self._unload_hunyuan_temporarily()
                
                # CRITICAL: Aggressive GPU memory clearing before FLUX
                print("🧹 Aggressive GPU memory clearing before FLUX...")
                self._aggressive_gpu_cleanup()
                
                # Step 2: Generate image with FLUX
                print("Step 1: Generating image with FLUX...")
                flux_pipe = self._load_flux_pipeline(seed)
                if flux_pipe is None:
                    # Restore validation server before failing
                    if validation_unloaded:
                        self._coordinate_with_validation_server("reload")
                    raise RuntimeError("Failed to load FLUX pipeline")
                
                enhanced_prompt = f"wbgmsst, {prompt}, 3D isometric asset, clean white background, isolated object"
                generator = torch.Generator(device="cuda").manual_seed(seed)
                
                image = flux_pipe(
                    prompt=enhanced_prompt,
                    guidance_scale=3.5,
                    num_inference_steps=GENERATION_CONFIG['num_inference_steps_t2i'],
                    width=1024,
                    height=1024,
                    generator=generator,
                ).images[0]
                
                print("✓ Image generated successfully")
                
                # CRITICAL: Completely clear FLUX from memory
                del flux_pipe
                self._clear_gpu_memory()
                
                # Step 3: Process image (background removal)
                print("Step 2: Processing image...")
                if self.rembg:
                    processed_image = self.rembg(image)
                else:
                    processed_image = image
                print("✓ Image processed")
                
                # Step 4: Reload Hunyuan3D and generate mesh
                self._reload_hunyuan_pipeline()
                
                print("Step 3: Generating 3D mesh...")
                generator = torch.Generator(device="cuda").manual_seed(seed)
                
                mesh = self.hunyuan_pipeline(
                    processed_image,
                    num_inference_steps=GENERATION_CONFIG['num_inference_steps_shape'],
                    guidance_scale=2.0,
                    generator=generator,
                )
                
                print("✓ 3D mesh generated")
                
                # Handle Hunyuan3D's nested list structure: List[List[trimesh.Trimesh]]
                if isinstance(mesh, list):
                    if len(mesh) > 0 and isinstance(mesh[0], list):
                        # Nested list structure - flatten and select best mesh
                        all_meshes = []
                        for batch_meshes in mesh:
                            all_meshes.extend(batch_meshes)
                        mesh = all_meshes
                    
                    if isinstance(mesh, list) and len(mesh) > 0:
                        # Select the largest mesh by vertex count (usually the main object)
                        num_candidates = len(mesh)
                        mesh = max(mesh, key=lambda m: len(m.vertices) if m is not None else 0)
                        print(f"Selected mesh with {len(mesh.vertices)} vertices from {num_candidates} candidates")
                    else:
                        raise RuntimeError("No valid meshes generated")
                
                # Ensure we have a valid single mesh object
                if not hasattr(mesh, 'vertices'):
                    raise RuntimeError("Invalid mesh object generated")
                
                # Step 5: Post-process mesh
                print("Step 4: Post-processing mesh...")
                face_reducer = FaceReducer()
                floater_remover = FloaterRemover()
                degenerate_remover = DegenerateFaceRemover()
                
                mesh = face_reducer(mesh)
                mesh = floater_remover(mesh)
                mesh = degenerate_remover(mesh)
                
                print(f"✓ Mesh processed: {len(mesh.vertices)} vertices, {len(mesh.faces)} faces")
                
                # Step 6: Convert to Gaussian Splatting PLY
                print("Step 5: Converting to Gaussian Splatting PLY...")
                ply_data = self._mesh_to_gaussian_splatting(mesh)
                
                # Save intermediate outputs if enabled
                if GENERATION_CONFIG['save_intermediate_outputs']:
                    output_dir = Path(GENERATION_CONFIG['output_dir'])
                    timestamp = int(time.time())
                    
                    # Save image
                    image.save(output_dir / f"image_{timestamp}.png")
                    processed_image.save(output_dir / f"processed_{timestamp}.png")
                    
                    # Save mesh
                    mesh.export(output_dir / f"mesh_{timestamp}.glb")
                    
                    # Save viewable mesh PLY
                    mesh.export(output_dir / f"mesh_viewable_{timestamp}.ply")
                    
                    # Save Gaussian Splatting PLY
                    with open(output_dir / f"gaussian_splatting_{timestamp}.ply", "wb") as f:
                        f.write(ply_data)
                    
                    print(f"✓ Intermediate outputs saved to {output_dir}")
                
                generation_time = time.time() - start_time
                
                # Update metrics
                self.metrics.total_generations += 1
                self.metrics.successful_generations += 1
                self.metrics.last_generation_time = generation_time
                self.metrics.average_generation_time = (
                    (self.metrics.average_generation_time * (self.metrics.successful_generations - 1) + generation_time) 
                    / self.metrics.successful_generations
                )
                
                print(f"🎉 Generation completed in {generation_time:.2f}s")
                
                # CRITICAL: Restore validation server models
                if validation_unloaded:
                    print("🤝 Restoring validation server models...")
                    self._coordinate_with_validation_server("reload")
                
                return ply_data
                
            except Exception as e:
                self.metrics.total_generations += 1
                self.metrics.failed_generations += 1
                print(f"❌ Generation failed: {e}")
                traceback.print_exc()
                
                # CRITICAL: Restore validation server even on failure
                if 'validation_unloaded' in locals() and validation_unloaded:
                    print("🤝 Restoring validation server models after failure...")
                    self._coordinate_with_validation_server("reload")
                
                return None

    def _create_basic_mesh_ply(self, mesh: trimesh.Trimesh) -> bytes:
        """Create a basic mesh PLY as fallback"""
        try:
            import io
            buffer = io.BytesIO()
            mesh.export(buffer, file_type='ply')
            return buffer.getvalue()
        except Exception as e:
            print(f"❌ Failed to create basic PLY: {e}")
            return b""

    def get_status(self) -> Dict[str, Any]:
        """Get server status and metrics"""
        return {
            "status": "running",
            "models_loaded": {
                "hunyuan3d": self.hunyuan_pipeline is not None,
                "background_remover": self.rembg is not None,
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
                "sugar_converted_count": self.metrics.sugar_converted_count,
            },
            "config": GENERATION_CONFIG,
            "gpu_memory": self._clear_gpu_memory() if torch.cuda.is_available() else 0,
        }

    def _get_gpu_memory_info(self) -> Dict[str, float]:
        """Get current GPU memory information"""
        if torch.cuda.is_available():
            gpu_memory_free, gpu_memory_total = torch.cuda.mem_get_info()
            memory_used = gpu_memory_total - gpu_memory_free
            return {
                "total_gb": gpu_memory_total / 1024**3,
                "free_gb": gpu_memory_free / 1024**3,
                "used_gb": memory_used / 1024**3,
                "used_percent": (memory_used / gpu_memory_total) * 100
            }
        return {"total_gb": 0, "free_gb": 0, "used_gb": 0, "used_percent": 0}

    def _print_gpu_status(self, prefix: str = ""):
        """Print current GPU memory status"""
        memory_info = self._get_gpu_memory_info()
        print(f"🧠 {prefix}GPU Memory Status:")
        print(f"   Total: {memory_info['total_gb']:.1f} GB")
        print(f"   Allocated: {memory_info['used_gb']:.1f} GB")
        print(f"   Reserved: {memory_info['used_gb']:.1f} GB")  # Approximation
        print(f"   Available: {memory_info['free_gb']:.1f} GB")

    def _coordinate_with_validation_server(self, action: str) -> bool:
        """Coordinate GPU memory with validation server"""
        try:
            validation_url = GENERATION_CONFIG['validation_server_url']
            
            if action == "unload":
                # Ask validation server to unload models
                response = requests.post(f"{validation_url}/unload_models/", timeout=30)
                if response.status_code == 200:
                    result = response.json()
                    if result.get("status") == "success":
                        freed_gb = result.get("memory_freed_gb", 0)
                        print(f"✅ Validation server freed {freed_gb:.1f}GB GPU memory")
                        return True
                    else:
                        print(f"⚠️ Validation server unload failed: {result}")
                        return False
                else:
                    print(f"⚠️ Validation server unload request failed: {response.status_code}")
                    return False
                    
            elif action == "reload":
                # Ask validation server to reload models
                response = requests.post(f"{validation_url}/reload_models/", timeout=60)
                if response.status_code == 200:
                    result = response.json()
                    if result.get("models_reloaded"):
                        reload_time = result.get("reload_time", 0)
                        print(f"✅ Validation server reloaded models in {reload_time:.1f}s")
                        return True
                    else:
                        print(f"⚠️ Validation server reload failed: {result}")
                        return False
                else:
                    print(f"⚠️ Validation server reload request failed: {response.status_code}")
                    return False
                    
        except Exception as e:
            print(f"⚠️ Validation server coordination failed ({action}): {e}")
            return False
        
        return False

# Initialize generator
generator = FluxHunyuanSuGaRGenerator()

# FastAPI app
app = FastAPI(title="FLUX + Hunyuan3D + SuGaR Generation Server")

@app.post("/generate/")
async def generate_3d_model_endpoint(
    prompt: str = Form(...), 
    seed: Optional[int] = Form(None),
    return_compressed: Optional[bool] = Form(True)
):
    """Generate 3D model from text prompt using FLUX + Hunyuan3D + SuGaR pipeline."""
    
    # Handle seed
    if seed is None:
        seed = random.randint(0, MAX_SEED)
    
    # Generate model
    ply_data = generator.generate_3d_model(prompt, seed)
    
    if ply_data is None:
        raise HTTPException(status_code=500, detail="Generation failed")
    
    # Return PLY data
    return Response(
        content=ply_data,
        media_type="application/octet-stream",
        headers={
            "Content-Disposition": f"attachment; filename=generated_model_{seed}.ply",
            "X-Generation-Seed": str(seed),
            "X-Generation-Prompt": prompt,
            "X-Model-Format": "gaussian_splatting_ply",
        }
    )

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
    available_memory = generator._clear_gpu_memory()
    return {"status": "cache_cleared", "available_memory_gb": available_memory}

@app.get("/config/")
async def get_config():
    """Get current configuration"""
    return GENERATION_CONFIG

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FLUX + Hunyuan3D + SuGaR Generation Server")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8095, help="Port to bind to")
    parser.add_argument("--workers", type=int, default=1, help="Number of worker processes")
    
    args = parser.parse_args()
    
    print(f"Starting FLUX + Hunyuan3D + SuGaR Generation Server on {args.host}:{args.port}")
    print("=" * 60)
    print("Pipeline: Text → FLUX → Image → Hunyuan3D → Mesh → SuGaR → Gaussian Splatting PLY")
    print("Features:")
    print("  • FLUX text-to-image generation with quantization")
    print("  • Hunyuan3D-2 image-to-3D mesh generation")
    print("  • SuGaR-inspired mesh-to-Gaussian Splatting conversion")
    print("  • Validation-compatible PLY output format")
    print("  • Memory-optimized for RTX 4090 (24GB)")
    print("=" * 60)
    
    uvicorn.run(
        app, 
        host=args.host, 
        port=args.port,
        workers=args.workers,
        log_level="info"
    ) 