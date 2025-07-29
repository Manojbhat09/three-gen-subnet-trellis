#!/usr/bin/env python3
"""
Subnet 17 (404-GEN) - Base TRELLIS Generation Server
Purpose: HTTP server for 3D model generation using base TRELLIS text-to-3D pipeline
Produces Gaussian Splatting PLY files with SPZ compression

Text Prompt → TRELLIS 3D → Gaussian Splatting PLY + SPZ Compression

# Generate 3D model
curl -X POST "http://localhost:8097/generate/" \
  -F "prompt=a blue ceramic vase with red trim" \
  -F "seed=42"

# Get asset information
curl "http://localhost:8097/assets/"

# Download compressed PLY file
curl "http://localhost:8097/assets/gaussian_splatting_ply" -o model.ply.spz
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
import random
import logging
import tempfile
import argparse
import base64
import io
import json
import math
import signal
import queue
import requests

from fastapi import FastAPI, Form, HTTPException
from fastapi.responses import Response, JSONResponse
import uvicorn

# Set environment variables for TRELLIS
os.environ['SPCONV_ALGO'] = 'native'
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

# Add TRELLIS to Python path
import sys
TRELLIS_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "TRELLIS")
sys.path.append(TRELLIS_PATH)

# Import TRELLIS components
from trellis.pipelines import TrellisTextTo3DPipeline
from trellis.utils import render_utils, postprocessing_utils

# Constants
MAX_SEED = np.iinfo(np.int32).max

# Configuration
GENERATION_CONFIG = {
    'output_dir': './trellis_base_outputs',
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    'model_path': 'microsoft/TRELLIS-text-xlarge',
    'save_intermediate_outputs': True,
    'save_preview': False,
    'auto_compress_ply': True,
    # TRELLIS default settings
    'ss_guidance_strength': 7.5,
    'ss_sampling_steps': 25,
    'slat_guidance_strength': 7.5,
    'slat_sampling_steps': 25,
    # Memory management
    'enable_memory_efficient_attention': True,
    'enable_cpu_offload': True,
    'max_memory_usage_gb': 20,
    'validation_server_url': 'http://127.0.0.1:10006',
    'auto_validate_generations': True,
    'validation_timeout': 120,
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
        if asset_type == AssetType.GAUSSIAN_SPLATTING_PLY:
            file_path = self.asset_directory / "gaussian_splatting.ply"
            with open(file_path, 'wb') as f:
                f.write(data)
        elif asset_type == AssetType.PREVIEW_VIDEO:
            file_path = self.asset_directory / "preview.mp4"
            import imageio
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
                "pipeline": "trellis_base_v1.0"
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

class TrellisBaseGenerator:
    def __init__(self):
        # Initialize model instance variables
        self.trellis_pipeline = None
        
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
                print("✓ HuggingFace token loaded from cache")
            else:
                print("⚠️ No HuggingFace token found in cache")
        except Exception as e:
            print(f"⚠️ Error getting token from cache: {e}")
        
        Path(GENERATION_CONFIG['output_dir']).mkdir(exist_ok=True)
        print("🔧 TRELLIS Base Generator initialized")
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

    def _load_trellis_pipeline(self):
        """Load TRELLIS text-to-3D pipeline"""
        if self.trellis_pipeline is not None:
            print("✓ TRELLIS pipeline already loaded")
            return
            
        print("🔧 Loading TRELLIS text-to-3D pipeline...")
        
        try:
            self.trellis_pipeline = TrellisTextTo3DPipeline.from_pretrained(
                GENERATION_CONFIG['model_path']
            )
            self.trellis_pipeline.cuda()
            
            # Warm up the pipeline
            try:
                print("🔥 Warming up TRELLIS pipeline...")
                self.trellis_pipeline.run(
                    "a simple chair",
                    seed=42,
                    formats=["gaussian"],
                    sparse_structure_sampler_params={
                        "steps": 2,
                        "cfg_strength": GENERATION_CONFIG['ss_guidance_strength'],
                    },
                    slat_sampler_params={
                        "steps": 2,
                        "cfg_strength": GENERATION_CONFIG['slat_guidance_strength'],
                    },
                )
                print("✅ TRELLIS pipeline warmed up successfully")
            except Exception as e:
                print(f"⚠️ Pipeline warmup failed: {e}")
            
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

    def generate_3d_model(self, prompt: str, seed: int = 42) -> Optional[Tuple[bytes, Optional[bytes]]]:
        """Generate 3D model from text prompt using base TRELLIS pipeline"""
        
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
                print(f"🎯 Starting TRELLIS base generation for: '{prompt}' (seed: {seed})")
                
                # Initialize asset manager for this generation
                generation_asset = self.asset_manager.create_asset(prompt, seed)
                
                # Step 1: Generate 3D model with TRELLIS
                print("Step 1: Generating 3D model with TRELLIS...")
                if self.trellis_pipeline is None:   
                    self._load_trellis_pipeline()
                
                outputs = self.trellis_pipeline.run(
                    prompt,
                    seed=seed,
                    formats=["gaussian", "mesh"],
                    sparse_structure_sampler_params={
                        "steps": GENERATION_CONFIG['ss_sampling_steps'],
                        "cfg_strength": GENERATION_CONFIG['ss_guidance_strength'],
                    },
                    slat_sampler_params={
                        "steps": GENERATION_CONFIG['slat_sampling_steps'],
                        "cfg_strength": GENERATION_CONFIG['slat_guidance_strength'],
                    },
                )
                
                print("✓ 3D model generated successfully")
                
                # Step 2: Extract Gaussian Splatting PLY
                print("Step 2: Extracting Gaussian Splatting PLY...")
                gaussian_output = outputs['gaussian'][0]
                
                # Save as PLY file
                import io
                ply_buffer = io.BytesIO()
                gaussian_output.save_ply(ply_buffer)
                ply_data = ply_buffer.getvalue()
                
                print(f"✓ Gaussian Splatting PLY extracted ({len(ply_data):,} bytes)")
                generation_asset.add_asset(AssetType.GAUSSIAN_SPLATTING_PLY, ply_data)
                
                # Step 3: Generate preview video (optional)
                if GENERATION_CONFIG.get('save_intermediate_outputs', False) and GENERATION_CONFIG.get('save_preview', False):
                    print("Step 3: Generating preview video...")
                    try:
                        video = render_utils.render_video(outputs['gaussian'][0], num_frames=120)['color']
                        video_geo = render_utils.render_video(outputs['mesh'][0], num_frames=120)['normal']
                        combined_video = [np.concatenate([video[i], video_geo[i]], axis=1) for i in range(len(video))]
                        generation_asset.add_asset(AssetType.PREVIEW_VIDEO, combined_video)
                        print("✓ Preview video generated")
                    except Exception as e:
                        print(f"⚠️ Preview video generation failed: {e}")
                
                # Step 4: Compress PLY if enabled
                compressed_data = None
                if GENERATION_CONFIG.get('auto_compress_ply', True):
                    print("Step 4: Compressing PLY with SPZ...")
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
                
                generation_time = time.time() - start_time
                
                # Update metrics
                self.metrics.total_generations += 1
                self.metrics.successful_generations += 1
                self.metrics.last_generation_time = generation_time
                self.metrics.average_generation_time = (
                    (self.metrics.average_generation_time * (self.metrics.successful_generations - 1) + generation_time) 
                    / self.metrics.successful_generations
                )
                
                print(f"🎉 TRELLIS base generation completed in {generation_time:.2f}s")
                
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
                            "ss_guidance_strength": GENERATION_CONFIG['ss_guidance_strength'],
                            "ss_sampling_steps": GENERATION_CONFIG['ss_sampling_steps'],
                            "slat_guidance_strength": GENERATION_CONFIG['slat_guidance_strength'],
                            "slat_sampling_steps": GENERATION_CONFIG['slat_sampling_steps'],
                        }, f, indent=2)
                    print(f"💾 Metadata saved: {metadata_path}")

                # Step 5: Auto-validation if enabled
                if GENERATION_CONFIG.get('auto_validate_generations', False):
                    print("Step 5: Auto-validating generation...")
                    try:
                        validation_results = self.submit_for_validation(prompt, ply_data)
                        if validation_results["status"] == "success":
                            print(f"✅ Auto-validation completed! Score: {validation_results['validation_score']:.4f}")
                        else:
                            print(f"⚠️ Auto-validation failed: {validation_results.get('error', 'Unknown error')}")
                    except Exception as e:
                        print(f"⚠️ Auto-validation failed: {e}")

                generation_job_status.update({
                    "status": "completed",
                    "end_time": time.time(),
                    "ply_path": f"generated_model_{seed}.ply"
                })
                            
                return ply_data, compressed_data
                
            except Exception as e:
                self.metrics.total_generations += 1
                self.metrics.failed_generations += 1
                print(f"❌ TRELLIS base generation failed: {e}")
                traceback.print_exc()
                
                # Cleanup on failure
                self._unload_trellis_pipeline()
                
                generation_job_status.update({
                    "status": "failed",
                    "end_time": time.time(),
                    "error": str(e)
                })
                
                return None

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

    def get_status(self) -> Dict[str, Any]:
        """Get server status and metrics"""
        return {
            "status": "running",
            "models_loaded": {
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
            },
            "config": GENERATION_CONFIG,
            "gpu_memory": self._clear_gpu_memory() if torch.cuda.is_available() else 0,
            "ready": self.ready
        }

# Initialize FastAPI app
app = FastAPI(title="Base TRELLIS Generation Server", version="1.0.0")

# Initialize global generator
generator = TrellisBaseGenerator()

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
    return_compressed: Optional[bool] = Form(True)
):
    """Generate 3D model from text prompt using base TRELLIS pipeline."""
    
    # Handle seed
    if seed is None:
        seed = random.randint(0, MAX_SEED)
    
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
                print(f"🗜️ SPZ Compression for response:")
                print(f"   Original: {len(ply_data):,} bytes ({len(ply_data)/1024/1024:.1f} MB)")
                print(f"   Compressed: {len(compressed_data):,} bytes ({len(compressed_data)/1024/1024:.1f} MB)") 
                print(f"   Ratio: {len(compressed_data)/len(ply_data)*100:.1f}%")
            
            return Response(
                content=compressed_data,
                media_type="application/octet-stream",
                headers={
                    "Content-Disposition": f"attachment; filename=trellis_base_model_{seed}.ply.spz",
                    "X-Generation-Seed": str(seed),
                    "X-Generation-Prompt": prompt,
                    "X-Model-Format": "gaussian_splatting_ply",
                    "X-Pipeline": "trellis_base",
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
            "Content-Disposition": f"attachment; filename=trellis_base_model_{seed}.ply",
            "X-Generation-Seed": str(seed),
            "X-Generation-Prompt": prompt,
            "X-Model-Format": "gaussian_splatting_ply",
            "X-Pipeline": "trellis_base",
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
        filename = f"trellis_base_{asset_type}"
        
        if asset_type_enum == AssetType.GAUSSIAN_SPLATTING_PLY:
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
                "X-Pipeline": "trellis_base"
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

@app.post("/config/update/")
async def update_config(
    ss_guidance_strength: Optional[float] = Form(None),
    ss_sampling_steps: Optional[int] = Form(None),
    slat_guidance_strength: Optional[float] = Form(None),
    slat_sampling_steps: Optional[int] = Form(None),
    auto_compress_ply: Optional[bool] = Form(None),
    auto_validate_generations: Optional[bool] = Form(None),
    validation_server_url: Optional[str] = Form(None)
):
    """Update generation configuration"""
    try:
        if ss_guidance_strength is not None:
            GENERATION_CONFIG['ss_guidance_strength'] = ss_guidance_strength
        if ss_sampling_steps is not None:
            GENERATION_CONFIG['ss_sampling_steps'] = ss_sampling_steps
        if slat_guidance_strength is not None:
            GENERATION_CONFIG['slat_guidance_strength'] = slat_guidance_strength
        if slat_sampling_steps is not None:
            GENERATION_CONFIG['slat_sampling_steps'] = slat_sampling_steps
        if auto_compress_ply is not None:
            GENERATION_CONFIG['auto_compress_ply'] = auto_compress_ply
        if auto_validate_generations is not None:
            GENERATION_CONFIG['auto_validate_generations'] = auto_validate_generations
        if validation_server_url is not None:
            GENERATION_CONFIG['validation_server_url'] = validation_server_url
        
        return {
            "status": "success",
            "message": "Configuration updated",
            "config": GENERATION_CONFIG
        }
    except Exception as e:
        return JSONResponse(content={
            "status": "error",
            "message": str(e)
        }, status_code=500)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Base TRELLIS Generation Server")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8097, help="Port to bind to")
    parser.add_argument("--workers", type=int, default=1, help="Number of worker processes")
    
    args = parser.parse_args()
    
    print(f"Starting Base TRELLIS Generation Server on {args.host}:{args.port}")
    print("=" * 80)
    print("Pipeline: Text → TRELLIS → Gaussian Splatting PLY")
    print("Features:")
    print("  • Base TRELLIS text-to-3D generation")
    print("  • Gaussian Splatting PLY output")
    print("  • SPZ compression for efficient storage/transmission")
    print("  • Optional validation integration")
    print("  • Memory-optimized for high-end GPUs")
    print("=" * 80)
    
    uvicorn.run(
        app, 
        host=args.host, 
        port=args.port,
        workers=args.workers,
        log_level="info"
    ) 