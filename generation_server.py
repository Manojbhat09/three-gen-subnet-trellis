#!/usr/bin/env python3
# Subnet 17 (404-GEN) - Generation Server
# Purpose: HTTP server for 3D model generation using Hunyuan3D-2 and Flux

import os
import time
import torch
import traceback
import threading
import gc
from pathlib import Path
from typing import Optional, Dict, Any
from dataclasses import dataclass

from fastapi import FastAPI, Form, HTTPException
from fastapi.responses import Response
import uvicorn
import trimesh
from PIL import Image
import numpy as np

# Import Hunyuan3D-2 components
from Hunyuan3D_2.hy3dgen.rembg import BackgroundRemover
from Hunyuan3D_2.hy3dgen.shapegen import (
    Hunyuan3DDiTFlowMatchingPipeline, 
    FaceReducer, 
    FloaterRemover, 
    DegenerateFaceRemover
)

# Import Flux components
from diffusers import FluxTransformer2DModel, FluxPipeline
from transformers import T5EncoderModel
from accelerate import init_empty_weights
from accelerate.utils import load_and_quantize_model
from diffusers.utils.hub_utils import _get_model_file

# Configuration
GENERATION_CONFIG = {
    't2i_model_id': "black-forest-labs/FLUX.1-schnell",
    'shape_model_path': 'jetx/Hunyuan3D-2',
    'output_dir': './generation_outputs',
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    'num_inference_steps_t2i': 8,
    'num_inference_steps_shape': 30,
    'mc_algo': 'mc',
}

@dataclass
class GenerationMetrics:
    total_generations: int = 0
    successful_generations: int = 0
    failed_generations: int = 0
    average_generation_time: float = 0.0
    last_generation_time: float = 0.0

class HunyuanFluxGenerator:
    def __init__(self):
        self.shape_pipeline = None
        self.rembg = None
        self.metrics = GenerationMetrics()
        self.generation_lock = threading.Lock()
        
        Path(GENERATION_CONFIG['output_dir']).mkdir(exist_ok=True)
        
        self._initialize_models()

    def _clear_gpu_memory(self):
        """Clear GPU memory cache."""
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()

    def _initialize_models(self):
        """Initialize all required models for 3D generation."""
        try:
            print("Initializing Hunyuan3D-2 models...")
            
            self.rembg = BackgroundRemover()
            print("✓ Background remover initialized")
            
            # Defer T2I model loading to generation time to save VRAM
            print("✓ T2I pipeline will be loaded on-demand")

            self.shape_pipeline = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(
                GENERATION_CONFIG['shape_model_path'], 
                torch_dtype=torch.float16,
                use_safetensors=True
            ).to(GENERATION_CONFIG['device'])
            print("✓ 3D shape generation pipeline initialized and moved to GPU")
            
            print("✓ Models pre-initialized successfully")
            
        except Exception as e:
            print(f"Error initializing models: {e}")
            traceback.print_exc()
            raise RuntimeError(f"Failed to initialize Hunyuan3D-2 models: {e}")

    def _load_flux_pipeline(self):
        """Load the FLUX pipeline for text-to-image generation."""
        try:
            print("Loading FLUX T2I pipeline...")
            pipe = FluxPipeline.from_pretrained(
                GENERATION_CONFIG['t2i_model_id'], 
                torch_dtype=torch.bfloat16
            )
            pipe.to(GENERATION_CONFIG['device'])
            print("✓ FLUX T2I pipeline loaded successfully")
            return pipe
        except Exception as e:
            print(f"Error loading FLUX pipeline: {e}")
            traceback.print_exc()
            return None
    
    def generate_3d_model(self, prompt: str, seed: int = 42) -> Optional[bytes]:
        """
        Generate a 3D model from text prompt.
        Returns: PLY file bytes or None if generation fails
        """
        with self.generation_lock:
            start_time = time.time()
            self.metrics.total_generations += 1
            
            try:
                print(f"Generating 3D model for prompt: '{prompt}'")
                
                # Step 1: Generate image from text using FLUX
                print("Step 1: Generating image from text with FLUX...")
                flux_pipe = self._load_flux_pipeline()
                if not flux_pipe:
                    raise RuntimeError("Failed to load FLUX pipeline.")
                
                t2i_prompt = "wbgmsst, " + prompt + ", 3D isometric, white background"
                generator = torch.Generator(device=GENERATION_CONFIG['device']).manual_seed(seed)
                
                image = flux_pipe(
                    prompt=t2i_prompt,
                    num_inference_steps=GENERATION_CONFIG['num_inference_steps_t2i'],
                    generator=generator,
                ).images[0]
                print("✓ Image generated successfully")
                
                # Unload FLUX pipeline to free VRAM
                del flux_pipe
                self._clear_gpu_memory()
                print("✓ FLUX pipeline unloaded from memory")

                # Step 2: Remove background
                print("Step 2: Removing background...")
                image_no_bg = self.rembg(image)
                print("✓ Background removed successfully")
                
                # Step 3: Generate 3D mesh
                print("Step 3: Generating 3D mesh...")
                mesh_generator = torch.manual_seed(seed)
                mesh = self.shape_pipeline(
                    image=image_no_bg, 
                    num_inference_steps=GENERATION_CONFIG['num_inference_steps_shape'],
                    mc_algo=GENERATION_CONFIG['mc_algo'],
                    generator=mesh_generator
                )[0]
                
                # Step 4: Post-process mesh
                print("Step 4: Post-processing mesh...")
                mesh = FloaterRemover()(mesh)
                mesh = DegenerateFaceRemover()(mesh)
                mesh = FaceReducer()(mesh)
                
                # Step 5: Export to PLY format in-memory
                print("Step 5: Converting to PLY format...")
                with trimesh.util.capture_as_stream(mesh, file_type='ply') as stream:
                    ply_bytes = stream.read()

                # Update metrics
                generation_time = time.time() - start_time
                self.metrics.successful_generations += 1
                self.metrics.last_generation_time = generation_time
                self.metrics.average_generation_time = (
                    (self.metrics.average_generation_time * (self.metrics.successful_generations - 1) + generation_time) /
                    self.metrics.successful_generations
                )
                
                print(f"✓ 3D model generated successfully in {generation_time:.2f}s")
                print(f"✓ PLY file size: {len(ply_bytes)} bytes")
                
                return ply_bytes
                
            except Exception as e:
                self.metrics.failed_generations += 1
                print(f"Error generating 3D model: {e}")
                traceback.print_exc()
                return None
            
            finally:
                # Final cleanup
                self._clear_gpu_memory()

    def get_status(self) -> Dict[str, Any]:
        """Get generator status and metrics."""
        status = {
            'device': GENERATION_CONFIG['device'],
            'total_generations': self.metrics.total_generations,
            'successful_generations': self.metrics.successful_generations,
            'failed_generations': self.metrics.failed_generations,
            'success_rate': (
                self.metrics.successful_generations / max(self.metrics.total_generations, 1) * 100
            ),
            'average_generation_time': self.metrics.average_generation_time,
            'last_generation_time': self.metrics.last_generation_time,
        }
        if torch.cuda.is_available():
            status['gpu_memory_allocated'] = torch.cuda.memory_allocated()
            status['gpu_memory_cached'] = torch.cuda.memory_reserved()
        return status

# Initialize generator
generator = HunyuanFluxGenerator()

# FastAPI app
app = FastAPI(title="Subnet 17 Generation Server", version="1.1.0")

@app.post("/generate/")
async def generate_3d_model_endpoint(prompt: str = Form(...), seed: Optional[int] = Form(None)):
    """Generate a 3D model from text prompt."""
    try:
        if not prompt or len(prompt.strip()) == 0:
            raise HTTPException(status_code=400, detail="Prompt cannot be empty")
        
        # Use provided seed or generate a random one
        generation_seed = seed if seed is not None else np.random.randint(0, 2**32 - 1)
        
        ply_bytes = generator.generate_3d_model(prompt.strip(), seed=generation_seed)
        
        if ply_bytes is None:
            raise HTTPException(status_code=500, detail="3D model generation failed")
        
        return Response(
            content=ply_bytes,
            media_type="application/octet-stream",
            headers={
                "Content-Disposition": f"attachment; filename=generated_model_{generation_seed}.ply",
                "X-Generation-Time": str(generator.metrics.last_generation_time),
                "X-Seed": str(generation_seed)
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"Error in generate endpoint: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@app.get("/status/")
async def get_status():
    """Get generation server status."""
    return generator.get_status()

@app.get("/health/")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy" if generator.shape_pipeline else "unhealthy",
        "models_initialized": generator.shape_pipeline is not None,
        "timestamp": time.time()
    }

@app.post("/clear_cache/")
async def clear_cache():
    """Clear GPU memory cache."""
    generator._clear_gpu_memory()
    return {"message": "GPU cache cleared"}

if __name__ == "__main__":
    print("Starting Subnet 17 Generation Server...")
    print(f"Device: {GENERATION_CONFIG['device']}")
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8093,
        log_level="info"
    ) 