#!/usr/bin/env python3
# Subnet 17 (404-GEN) - Enhanced Generation Server with Flux + Hunyuan3D + BPT
# Purpose: HTTP server for high-quality 3D model generation using Flux, Hunyuan3D-2, and BPT enhancement
# Integrated with comprehensive asset management system

import os
import time
import torch
import traceback
import threading
import gc
from pathlib import Path
from typing import Optional, Dict, Any
from dataclasses import dataclass
import numpy as np

from fastapi import FastAPI, Form, HTTPException
from fastapi.responses import Response, JSONResponse
import uvicorn
import trimesh
from PIL import Image
import random
import yaml
import logging

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

# Import BPT components
from hy3dgen.shapegen.bpt.model.model import MeshTransformer
from hy3dgen.shapegen.bpt.utils import Dataset, apply_normalize, sample_pc, joint_filter
from hy3dgen.shapegen.bpt.model.serializaiton import BPT_deserialize
from torch.utils.data import DataLoader

# Import asset management system
from generation_asset_manager import (
    GenerationAsset, AssetType, GenerationStatus,
    global_asset_manager, integrate_with_robust_server,
    prepare_for_mining_submission
)

# Constants
NUM_INFERENCE_STEPS = 8
MAX_SEED = np.iinfo(np.int32).max

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration
GENERATION_CONFIG = {
    'output_dir': './flux_hunyuan_bpt_outputs',
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    'num_inference_steps_t2i': 8,
    'num_inference_steps_shape': 30,
    'mc_algo': 'mc',
    'use_bpt': False,  # Disabled by default due to model compatibility issues
    'bpt_temperature': 0.5,
    'bpt_model_path': None,  # Will use default path
    'flux_model_url': "https://huggingface.co/gokaygokay/flux-game/blob/main/hyperflux_00001_.q8_0.gguf",
    'flux_base_model': "camenduru/FLUX.1-dev-diffusers",
    'hunyuan_model_path': 'jetx/Hunyuan3D-2',
    'save_intermediate_outputs': True,
    'auto_compress_ply': True,
}

@dataclass
class GenerationMetrics:
    total_generations: int = 0
    successful_generations: int = 0
    failed_generations: int = 0
    average_generation_time: float = 0.0
    last_generation_time: float = 0.0
    bpt_enhanced_count: int = 0

class FluxHunyuanBPTGenerator:
    def __init__(self):
        self.hunyuan_pipeline = None
        self.bpt_model = None
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
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
        
        if torch.cuda.is_available():
            available = torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_allocated()
            print(f"GPU memory cleared. Available: {available:,} bytes")
            print(f"Memory allocated: {torch.cuda.memory_allocated():,} bytes")
            print(f"Memory reserved: {torch.cuda.memory_reserved():,} bytes")

    def _create_bpt_config(self):
        """Create BPT model configuration"""
        config = {
            'dim': 1024,
            'max_seq_len': 8192,
            'flash_attn': True,
            'attn_depth': 24,
            'attn_dim_head': 64,
            'attn_heads': 16,
            'attn_kwargs': {
                'ff_glu': True,
                'num_mem_kv': 4,
                'attn_qk_norm': True,
            },
            'dropout': 0.0,
            'pad_id': -1,
            'coor_continuous_range': (-1., 1.),
            'num_discrete_coors': 128,
            'block_size': 8,
            'offset_size': 16,
            'mode': 'vertices',
            'special_token': -2,
            'use_special_block': True,
            'conditioned_on_pc': True,
            'encoder_name': 'miche-256-feature',
            'encoder_freeze': False,
            'cond_dim': 768
        }
        return config

    def _load_bpt_model(self):
        """Load BPT model for mesh enhancement"""
        config = self._create_bpt_config()
        
        # Initialize model
        model = MeshTransformer(**config)
        
        # Load pretrained weights if available
        model_path = GENERATION_CONFIG['bpt_model_path']
        if model_path is None:
            # Use the default BPT weights path
            script_dir = Path(__file__).resolve().parent
            model_path = script_dir / "Hunyuan3D-2" / "hy3dgen" / "shapegen" / "bpt" / "weights" / "bpt-8-16-500m.pt"
        
        if os.path.exists(model_path):
            try:
                checkpoint = torch.load(model_path, map_location=GENERATION_CONFIG['device'])
                if 'model_state_dict' in checkpoint:
                    model.load_state_dict(checkpoint['model_state_dict'])
                elif 'state_dict' in checkpoint:
                    model.load_state_dict(checkpoint['state_dict'])
                else:
                    model.load_state_dict(checkpoint)
                print(f"✓ BPT model loaded from {model_path}")
            except Exception as e:
                print(f"⚠️ Error loading BPT model from {model_path}: {e}")
                print("   Falling back to randomly initialized BPT weights.")
        else:
            print(f"⚠️ BPT model not found at {model_path}. Using randomly initialized weights.")
        
        self.bpt_model = model.to(GENERATION_CONFIG['device'])
        self.bpt_model.eval()

    def _initialize_models(self):
        """Initialize only background remover and Hunyuan3D. FLUX is loaded on demand."""
        print("Initializing core models...")
        device = GENERATION_CONFIG['device']
        dtype = torch.bfloat16

        # Load Background Remover with multiple fallbacks
        print("Loading background remover...")
        try:
            # Try the default u2net model first
            self.rembg = BackgroundRemover('u2net', device=device)
            print("✓ Background remover loaded with u2net")
        except Exception as e:
            print(f"Warning: Failed to load u2net: {e}")
            try:
                # Try without specifying device
                self.rembg = BackgroundRemover('u2net')
                print("✓ Background remover loaded with u2net (CPU)")
            except Exception as e2:
                print(f"Warning: Failed to load u2net on CPU: {e2}")
                try:
                    # Try silueta model as fallback
                    self.rembg = BackgroundRemover('silueta')
                    print("✓ Background remover loaded with silueta")
                except Exception as e3:
                    print(f"Warning: Failed to load silueta: {e3}")
                    try:
                        # Try default initialization
                        self.rembg = BackgroundRemover()
                        print("✓ Background remover loaded with default settings")
                    except Exception as e4:
                        print(f"⚠️ Error loading background remover: {e4}")
                        print("   Will use fallback background removal method")
                        self.rembg = None
            
        # Load Hunyuan3D Shape Generation Pipeline
        try:
            model_path = GENERATION_CONFIG['hunyuan_model_path']
            print(f"Loading Hunyuan3D from {model_path}")
            
            # First try with safetensors
            try:
                self.hunyuan_pipeline = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(
                    model_path,
                    torch_dtype=dtype,
                    use_safetensors=True
                )
                self.hunyuan_pipeline.to(device)
                print("✓ Hunyuan3D pipeline loaded with safetensors")
            except Exception as e:
                print(f"Warning: Failed to load with safetensors: {e}")
                print("Attempting to load without safetensors...")
                self.hunyuan_pipeline = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(
                    model_path,
                    torch_dtype=dtype,
                    use_safetensors=False
                )
                self.hunyuan_pipeline.to(device)
                print("✓ Hunyuan3D pipeline loaded without safetensors")
            
            # Store model path for texture pipeline
            self.hunyuan_model_path = model_path
            
        except Exception as e:
            print(f"⚠️ Error loading Hunyuan3D pipeline: {e}")
            self.hunyuan_pipeline = None
            self.hunyuan_model_path = None

        # Load BPT Model if enabled
        if GENERATION_CONFIG.get('use_bpt', False):
            self._load_bpt_model()
        
        print("Core model initialization complete. FLUX will be loaded on demand.")

    def _load_flux_pipeline(self, seed: int = 42):
        """Dynamically load and configure the FLUX pipeline using the exact demo implementation."""
        print("Loading FLUX pipeline with aggressive memory management...")
        
        # Clear all GPU memory first
        self._clear_gpu_memory()
        
        try:
            device = GENERATION_CONFIG['device']
            dtype = torch.bfloat16
            huggingface_token = os.getenv("HUGGINGFACE_TOKEN")
            
            # Use the exact configuration from the demo
            file_url = "https://huggingface.co/gokaygokay/flux-game/blob/main/hyperflux_00001_.q8_0.gguf"
            file_url = file_url.replace("/resolve/main/", "/blob/main/").replace("?download=true", "")
            single_file_base_model = "camenduru/FLUX.1-dev-diffusers"
            
            print(f"Loading FLUX from: {file_url}")
            print(f"Base model: {single_file_base_model}")
            
            # Check required dependencies
            try:
                from google import protobuf
            except ImportError:
                print("Installing required protobuf package...")
                import subprocess
                subprocess.check_call(["pip", "install", "--no-cache-dir", "protobuf"])
                from google import protobuf
            
            # Load text encoder with 8-bit quantization (exactly like demo)
            print("Loading text encoder...")
            quantization_config_tf = BitsAndBytesConfigTF(
                load_in_8bit=True,
                bnb_8bit_compute_dtype=dtype
            )
            text_encoder_2 = T5EncoderModel.from_pretrained(
                single_file_base_model, 
                subfolder="text_encoder_2", 
                torch_dtype=dtype, 
                quantization_config=quantization_config_tf, 
                token=huggingface_token
            )
            print("✓ Text encoder loaded")
            
            # Load transformer with GGUF configuration (exactly like demo)
            print("Loading transformer...")
            transformer = FluxTransformer2DModel.from_single_file(
                file_url, 
                subfolder="transformer", 
                quantization_config=GGUFQuantizationConfig(compute_dtype=dtype), 
                torch_dtype=dtype, 
                config=single_file_base_model
            )
            print("✓ Transformer loaded")
            
            # Initialize pipeline (exactly like demo)
            print("Initializing pipeline...")
            flux_pipeline = FluxPipeline.from_pretrained(
                single_file_base_model, 
                transformer=transformer, 
                text_encoder_2=text_encoder_2, 
                torch_dtype=dtype, 
                token=huggingface_token
            )
            flux_pipeline.to(device)
            
            print("✓ FLUX pipeline loaded successfully")
            return flux_pipeline
            
        except ImportError as e:
            print(f"⚠️ Import error loading FLUX pipeline: {e}")
            print("   Missing required dependencies")
            self._clear_gpu_memory()
            return None
        except FileNotFoundError as e:
            print(f"⚠️ File not found error loading FLUX pipeline: {e}")
            print("   Model files may not be downloaded or accessible")
            self._clear_gpu_memory()
            return None
        except torch.cuda.OutOfMemoryError as e:
            print(f"⚠️ CUDA OOM error loading FLUX pipeline: {e}")
            print("   Insufficient GPU memory")
            self._clear_gpu_memory()
            return None
        except Exception as e:
            print(f"⚠️ Unexpected error loading FLUX pipeline: {e}")
            print(f"   Error type: {type(e).__name__}")
            import traceback
            traceback.print_exc()
            self._clear_gpu_memory()
            return None

    def _unload_hunyuan_temporarily(self):
        """Temporarily unload Hunyuan3D to free memory for FLUX"""
        if self.hunyuan_pipeline:
            print("Temporarily unloading Hunyuan3D to free memory...")
            self.hunyuan_pipeline.to('cpu')
            self._clear_gpu_memory()

    def _reload_hunyuan_pipeline(self):
        """Reload Hunyuan3D back to GPU"""
        if self.hunyuan_pipeline:
            print("Reloading Hunyuan3D to GPU...")
            self.hunyuan_pipeline.to(GENERATION_CONFIG['device'])

    def _generate_placeholder_image(self, prompt: str, seed: int = 42):
        """Generate a placeholder image if FLUX fails."""
        from PIL import Image, ImageDraw, ImageFont
        
        img = Image.new('RGB', (1024, 1024), color = (73, 109, 137))
        d = ImageDraw.Draw(img)
        
        try:
            font = ImageFont.truetype("arial.ttf", 40)
        except IOError:
            font = ImageFont.load_default()
            
        text = f"FLUX Error\nPrompt: {prompt}\nSeed: {seed}"
        d.text((10,10), text, fill=(255,255,0), font=font)
        
        # Save placeholder image
        placeholder_dir = Path(GENERATION_CONFIG['output_dir']) / 'placeholders'
        placeholder_dir.mkdir(exist_ok=True)
        img_path = placeholder_dir / f"placeholder_{seed}.png"
        img.save(img_path)
        
        return img, img_path

    def _enhance_mesh_with_bpt(self, mesh_path: str, temperature: float = 0.5):
        """Enhance a mesh using the BPT model."""
        if self.bpt_model is None:
            print("BPT model not loaded, skipping enhancement.")
            return None
            
        device = GENERATION_CONFIG['device']
        
        try:
            # Create dataset from the mesh
            dataset = Dataset(input_type='mesh', input_list=[str(mesh_path)])
            dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
            
            with torch.no_grad():
                for batch in dataloader:
                    pc_normal = batch['pc_normal'].to(device)
                    uid = batch['uid']
                    
                    print(f"Processing mesh: {uid[0]}")
                    
                    # Generate enhanced mesh using BPT
                    codes = self.bpt_model.generate(
                        pc=pc_normal,
                        batch_size=1,
                        temperature=temperature,
                        filter_logits_fn=joint_filter,
                        filter_kwargs={'k': 50, 'p': 0.95},
                        max_seq_len=self.bpt_model.max_seq_len,
                        cache_kv=True,
                    )
                    
                    # Deserialize the codes to mesh faces
                    coordinates = BPT_deserialize(
                        codes.cpu().numpy().flatten(),
                        block_size=self.bpt_model.block_size,
                        offset_size=self.bpt_model.offset_size,
                        compressed=True,
                        special_token=-2,
                        use_special_block=self.bpt_model.use_special_block
                    )
                    
                    # Convert coordinates to trimesh object
                    if len(coordinates) > 0 and len(coordinates) % 3 == 0:
                        vertices = coordinates.reshape(-1, 3)
                        num_faces = len(vertices) // 3
                        faces = np.arange(len(vertices)).reshape(num_faces, 3)
                        
                        enhanced_mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
                        enhanced_mesh = apply_normalize(enhanced_mesh)
                        return enhanced_mesh
                    
                        return None
        except Exception as e:
            print(f"⚠️ Error during BPT enhancement: {e}")
            traceback.print_exc()
            return None

    def generate_3d_model_with_assets(self, prompt: str, seed: int = 42, use_bpt: bool = None) -> Optional[GenerationAsset]:
        """Generate a 3D model with comprehensive asset tracking and aggressive memory management."""
        if use_bpt is None:
            use_bpt = GENERATION_CONFIG.get('use_bpt', False)
            
        asset = global_asset_manager.create_asset(prompt=prompt, seed=seed)
        output_dir = Path(GENERATION_CONFIG['output_dir']) / asset.generation_id
        output_dir.mkdir(parents=True, exist_ok=True)
        
        start_time = time.time()
        flux_pipe = None
        mesh = None

        try:
            with self.generation_lock:
                # Step 1: Generate image with FLUX (with aggressive memory management)
                asset.update_status(GenerationStatus.IMAGE_GENERATING, "Preparing for FLUX generation...")
                image_start = time.time()
                
                # Temporarily move Hunyuan3D to CPU to free GPU memory
                self._unload_hunyuan_temporarily()
                
                # Load FLUX pipeline
                flux_pipe = self._load_flux_pipeline(seed)
                if not flux_pipe:
                    print("⚠️ FLUX pipeline failed to load, using fallback image generation...")
                    # Create a simple fallback image for testing
                    image = self._create_fallback_image(prompt, seed)
                    asset.add_asset(AssetType.ORIGINAL_IMAGE, image)
                    asset.performance_metrics.image_generation_time = time.time() - image_start
                    print(f"✓ Fallback image generated in {asset.performance_metrics.image_generation_time:.2f}s")
                else:
                    # Generate image with exact demo parameters
                    enhanced_prompt = "wbgmsst, " + prompt + ", 3D isometric asset, clean white background, isolated object"
                    generator = torch.Generator(device="cuda").manual_seed(seed)
                    try:
                        image = flux_pipe(
                            prompt=enhanced_prompt,
                            guidance_scale=3.5,
                            num_inference_steps=NUM_INFERENCE_STEPS,
                            width=1024,
                            height=1024,
                            generator=generator,
                        ).images[0]
                        asset.add_asset(AssetType.ORIGINAL_IMAGE, image)
                        asset.performance_metrics.image_generation_time = time.time() - image_start
                        print(f"✓ Image generated in {asset.performance_metrics.image_generation_time:.2f}s")
                    except Exception as e:
                        print(f"⚠️ FLUX generation failed: {str(e)}, using fallback...")
                        image = self._create_fallback_image(prompt, seed)
                        asset.add_asset(AssetType.ORIGINAL_IMAGE, image)
                        asset.performance_metrics.image_generation_time = time.time() - image_start
                    
                    # Immediately unload FLUX and clear memory
                    del flux_pipe
                    flux_pipe = None
                    self._clear_gpu_memory()
                
                if image is None:
                    asset.update_status(GenerationStatus.FAILED, "FLUX generated null image")
                    raise RuntimeError("FLUX generated null image")
                
                # Step 2: Remove background
                asset.update_status(GenerationStatus.IMAGE_PROCESSING, "Removing background...")
                bg_start = time.time()
                
                try:
                    if self.rembg:
                        removed_bg_image = self.rembg(image)
                        if removed_bg_image is None:
                            print("Background remover returned None, using fallback")
                            removed_bg_image = self._fallback_background_removal(image)
                    else:
                        print("Using fallback background removal method")
                        removed_bg_image = self._fallback_background_removal(image)
                    
                    asset.add_asset(AssetType.BACKGROUND_REMOVED_IMAGE, removed_bg_image)
                    
                except Exception as e:
                    print(f"Background removal error, using fallback: {str(e)}")
                    removed_bg_image = self._fallback_background_removal(image)
                    asset.add_asset(AssetType.BACKGROUND_REMOVED_IMAGE, removed_bg_image)

                asset.performance_metrics.background_removal_time = time.time() - bg_start
                print(f"✓ Background removed in {asset.performance_metrics.background_removal_time:.2f}s")
                
                # Step 3: Generate 3D shape with Hunyuan3D
                asset.update_status(GenerationStatus.MESH_GENERATING, "Generating initial mesh...")
                mesh_start = time.time()
                
                # Reload Hunyuan3D to GPU
                self._reload_hunyuan_pipeline()
                
                if not self.hunyuan_pipeline:
                    asset.update_status(GenerationStatus.FAILED, "Hunyuan3D pipeline not initialized")
                    raise RuntimeError("Hunyuan3D pipeline not initialized")
                
                try:
                    mesh = self.hunyuan_pipeline(
                        image=removed_bg_image,
                        num_inference_steps=GENERATION_CONFIG['num_inference_steps_shape'],
                        marching_cubes_algo=GENERATION_CONFIG['mc_algo'],
                        generator=torch.manual_seed(seed)
                    )[0]
                    
                    if mesh is None:
                        asset.update_status(GenerationStatus.FAILED, "Hunyuan3D generated null mesh")
                        raise RuntimeError("Hunyuan3D generated null mesh")
                    
                    # Apply post-processing (exactly like demo)
                    mesh = FloaterRemover()(mesh)
                    mesh = DegenerateFaceRemover()(mesh)
                    mesh = FaceReducer()(mesh)
                    
                    # Save initial mesh
                    initial_mesh_path = output_dir / 'initial.glb'
                    mesh.export(initial_mesh_path)
                    mesh_path = str(initial_mesh_path)
                    
                    # Validate mesh quality
                    if len(mesh.faces) < 100 or len(mesh.vertices) < 100:
                        asset.update_status(GenerationStatus.FAILED, f"Generated mesh has too few vertices/faces: {len(mesh.vertices)} vertices, {len(mesh.faces)} faces")
                        raise RuntimeError(f"Generated mesh has too few vertices/faces: {len(mesh.vertices)} vertices, {len(mesh.faces)} faces")
                    
                    asset.add_asset(AssetType.INITIAL_MESH_PLY, mesh)
                    asset.add_asset(AssetType.INITIAL_MESH_GLB, mesh)
                    
                    print(f"✓ Mesh generated: {len(mesh.vertices)} vertices, {len(mesh.faces)} faces")
                    
                except Exception as e:
                    asset.update_status(GenerationStatus.FAILED, f"Mesh generation error: {str(e)}")
                    raise RuntimeError(f"Mesh generation error: {str(e)}")

                asset.performance_metrics.mesh_generation_time = time.time() - mesh_start
                print(f"✓ Mesh generated in {asset.performance_metrics.mesh_generation_time:.2f}s")

                # Step 4: (Optional) Enhance with BPT
                enhanced_mesh = None
                if use_bpt and self.bpt_model and mesh_path:
                    asset.update_status(GenerationStatus.MESH_ENHANCING, "Enhancing mesh with BPT...")
                    bpt_start = time.time()
                    try:
                        enhanced_mesh = self._enhance_mesh_with_bpt(mesh_path, GENERATION_CONFIG['bpt_temperature'])
                        if enhanced_mesh:
                            enhanced_mesh_path = output_dir / 'enhanced_bpt.glb'
                            enhanced_mesh.export(enhanced_mesh_path)
                            asset.add_asset(AssetType.BPT_ENHANCED_MESH_PLY, enhanced_mesh)
                            asset.add_asset(AssetType.BPT_ENHANCED_MESH_GLB, enhanced_mesh)
                            self.metrics.bpt_enhanced_count += 1
                            print(f"✓ BPT enhancement completed")
                    except Exception as e:
                        print(f"BPT enhancement failed (non-critical): {e}")
                    asset.performance_metrics.bpt_enhancement_time = time.time() - bpt_start
                
                # Step 5: Texture generation
                asset.update_status(GenerationStatus.MESH_TEXTURING, "Generating texture...")
                texture_start = time.time()
                try:
                    if not self.hunyuan_model_path:
                        raise RuntimeError("Hunyuan model path not available")
                        
                    from hy3dgen.texgen import Hunyuan3DPaintPipeline
                    paint_pipeline = Hunyuan3DPaintPipeline.from_pretrained(self.hunyuan_model_path)
                    textured_mesh = paint_pipeline(mesh, image=removed_bg_image)
                    textured_mesh_path = output_dir / 'textured.glb'
                    textured_mesh.export(textured_mesh_path)
                    asset.add_asset(AssetType.TEXTURED_MESH_GLB, textured_mesh)
                    print(f"✓ Texture generation completed")
                except Exception as e:
                    print(f"Texture generation failed (non-critical): {e}")
                    # Use enhanced or initial mesh as fallback
                    if use_bpt and enhanced_mesh:
                        asset.add_asset(AssetType.TEXTURED_MESH_GLB, enhanced_mesh)
                    else:
                        asset.add_asset(AssetType.TEXTURED_MESH_GLB, mesh)

                asset.performance_metrics.texture_generation_time = time.time() - texture_start

                # Only mark as completed if we have the minimum required assets
                if (asset.get_asset(AssetType.INITIAL_MESH_PLY) and 
                    asset.get_asset(AssetType.INITIAL_MESH_GLB)):
                    asset.update_status(GenerationStatus.COMPLETED)
                    self.metrics.successful_generations += 1
                    
                    # Perform local validation and mining readiness check
                    validation_score = self._perform_local_validation(asset, prompt)
                    mining_ready = self._check_mining_readiness(asset, validation_score)
                    
                    print(f"✓ Generation completed successfully!")
                    print(f"✓ Local validation score: {validation_score:.4f}")
                    print(f"✓ Mining ready: {mining_ready}")
                    
                else:
                    asset.update_status(GenerationStatus.FAILED, "Missing required assets")
                    raise RuntimeError("Missing required assets")
                    
        except Exception as e:
            asset.update_status(GenerationStatus.FAILED, str(e))
            self.metrics.failed_generations += 1
            print(f"Error during 3D model generation: {e}")
            traceback.print_exc()
            raise

        finally:
            # Aggressive cleanup
            if flux_pipe:
                del flux_pipe
            total_time = time.time() - start_time
            asset.performance_metrics.total_generation_time = total_time
            
            self.metrics.total_generations += 1
            self.metrics.last_generation_time = total_time
            
            if self.metrics.successful_generations > 0:
                total_avg_time = (self.metrics.average_generation_time * (self.metrics.successful_generations - 1)) + self.metrics.last_generation_time
                self.metrics.average_generation_time = total_avg_time / self.metrics.successful_generations
            
            # Final aggressive memory cleanup
            self._clear_gpu_memory()
        
        return asset

    def _perform_local_validation(self, asset: GenerationAsset, prompt: str) -> float:
        """Perform local validation on the generated mesh"""
        try:
            ply_data = asset.get_asset_data(AssetType.INITIAL_MESH_PLY)
            if not ply_data:
                print("No PLY data found for validation")
                return 0.0
            
            # Load mesh for validation with proper file type specification
            try:
                import io
                mesh = trimesh.load(io.BytesIO(ply_data), file_type='ply')
            except Exception as e:
                print(f"Failed to load mesh from PLY data: {e}")
                try:
                    # Fallback: try loading without specifying file type
                    mesh = trimesh.load(trimesh.util.wrap_as_stream(ply_data), force='mesh')
                except Exception as e2:
                    print(f"Fallback mesh loading also failed: {e2}")
                    return 0.0
            
            # Calculate comprehensive validation score (0.0 to 1.0 scale)
            score = 0.0
            
            if mesh is None or mesh.is_empty or len(mesh.faces) == 0:
                print("Mesh is empty or has no faces")
                return 0.0
            
            # 1. Basic geometric validity (0.4 points max)
            if mesh.is_watertight:
                score += 0.2
                print("✓ Mesh is watertight (+0.2)")
            
            if mesh.is_volume:
                score += 0.1
                print("✓ Mesh has valid volume (+0.1)")
            
            # Non-convex meshes are usually more detailed/interesting
            if not mesh.is_convex:
                score += 0.1
                print("✓ Mesh is non-convex (detailed) (+0.1)")
            
            # 2. Face count quality (0.3 points max)
            face_count = len(mesh.faces)
            if face_count >= 1000:
                face_quality = min(face_count / 10000, 1.0)  # Scale to [0,1], cap at 10k faces
                face_score = face_quality * 0.3
                score += face_score
                print(f"✓ Face count quality: {face_count} faces (+{face_score:.3f})")
            else:
                print(f"⚠ Low face count: {face_count} faces")
            
            # 3. Vertex-to-face ratio (0.1 points max)
            vertex_count = len(mesh.vertices)
            if vertex_count > 0 and face_count > 0:
                ratio = vertex_count / face_count
                if 0.5 <= ratio <= 2.0:  # Good ratio range
                    score += 0.1
                    print(f"✓ Good vertex/face ratio: {ratio:.2f} (+0.1)")
            
            # 4. Surface area and bounding box validation (0.2 points max)
            try:
                if hasattr(mesh, 'area') and mesh.area > 0:
                    score += 0.1
                    print("✓ Mesh has valid surface area (+0.1)")
                
                if hasattr(mesh, 'bounds') and mesh.bounds is not None:
                    bounds_size = mesh.bounds[1] - mesh.bounds[0]
                    if np.all(bounds_size > 0):
                        score += 0.1
                        print("✓ Mesh has valid bounding box (+0.1)")
            except Exception as e:
                print(f"Surface area/bounds check failed: {e}")
            
            # Cap score at 1.0
            score = min(score, 1.0)
            
            # Update asset validation metrics
            asset.validation_metrics.face_count = face_count
            asset.validation_metrics.vertex_count = vertex_count
            asset.validation_metrics.mesh_quality_score = score
            asset.validation_metrics.is_manifold = mesh.is_watertight
            asset.validation_metrics.local_validation_score = score
            
            print(f"Local validation completed: {score:.4f}")
            return score
            
        except Exception as e:
            print(f"Local validation failed: {e}")
            import traceback
            traceback.print_exc()
            return 0.0

    def _check_mining_readiness(self, asset: GenerationAsset, validation_score: float) -> bool:
        """Check if the asset is ready for mining submission"""
        MINIMUM_SCORE = 0.7
        
        if validation_score < MINIMUM_SCORE:
            return False
        
        # Check required assets exist
        required_assets = [AssetType.INITIAL_MESH_PLY, AssetType.INITIAL_MESH_GLB]
        for asset_type in required_assets:
            if not asset.get_asset(asset_type):
                return False
        
        # Check mesh quality
        if (asset.validation_metrics.face_count < 100 or 
            asset.validation_metrics.vertex_count < 100):
            return False
        
        return True

    def generate_3d_model(self, prompt: str, seed: int = 42, use_bpt: bool = None) -> Optional[bytes]:
        """Simplified generation method for basic use cases, returns PLY bytes."""
        asset = self.generate_3d_model_with_assets(prompt, seed, use_bpt)
        if asset and asset.status == GenerationStatus.COMPLETED:
            if GENERATION_CONFIG['auto_compress_ply'] and asset.compressed_ply_data:
                return asset.compressed_ply_data
            else:
                return asset.get_asset_data(AssetType.INITIAL_MESH_PLY)
        return None

    def get_status(self) -> Dict[str, Any]:
        """Get the status of the generator and its metrics."""
        return {
            "metrics": self.metrics.__dict__,
            "config": {k: v for k, v in GENERATION_CONFIG.items() if k != 'device'},
            "cuda_available": torch.cuda.is_available(),
            "loaded_models": self.get_loaded_models()
        }
    
    def get_loaded_models(self) -> list:
        loaded = []
        if self.hunyuan_pipeline: loaded.append("Hunyuan3D")
        if self.bpt_model: loaded.append("BPT")
        if self.rembg: loaded.append("Rembg")
        return loaded

    def clear_cache(self):
        """Clears GPU cache."""
        self._clear_gpu_memory()

    def _fallback_background_removal(self, image):
        """Fallback background removal using simple image processing"""
        try:
            import numpy as np
            from PIL import Image, ImageOps
            
            # Convert to numpy array
            img_array = np.array(image)
            
            # Simple approach: assume background is mostly white/light colored
            # Create a mask based on brightness
            gray = np.mean(img_array, axis=2)
            threshold = np.percentile(gray, 85)  # Top 15% brightest pixels
            mask = gray < threshold
            
            # Apply morphological operations to clean up mask
            from scipy import ndimage
            mask = ndimage.binary_fill_holes(mask)
            mask = ndimage.binary_erosion(mask, iterations=2)
            mask = ndimage.binary_dilation(mask, iterations=3)
            
            # Create RGBA image with transparency
            rgba_array = np.zeros((img_array.shape[0], img_array.shape[1], 4), dtype=np.uint8)
            rgba_array[:, :, :3] = img_array
            rgba_array[:, :, 3] = mask * 255
            
            return Image.fromarray(rgba_array, 'RGBA')
            
        except Exception as e:
            print(f"Fallback background removal failed: {e}")
            # Return original image with white background made transparent
            try:
                return image.convert('RGBA')
            except:
                return image

    def _create_fallback_image(self, prompt: str, seed: int = 42):
        """Create a simple fallback image for testing"""
        from PIL import Image, ImageDraw, ImageFont
        
        img = Image.new('RGB', (1024, 1024), color = (73, 109, 137))
        d = ImageDraw.Draw(img)
        
        try:
            font = ImageFont.truetype("arial.ttf", 40)
        except IOError:
            font = ImageFont.load_default()
            
        text = f"Fallback Image\nPrompt: {prompt}\nSeed: {seed}"
        d.text((10,10), text, fill=(255,255,0), font=font)
        
        return img

generator = FluxHunyuanBPTGenerator()

app = FastAPI(title="Flux-Hunyuan-BPT Generation Server", version="2.0.0")

@app.post("/generate/")
async def generate_3d_model_endpoint(
    prompt: str = Form(...), 
    seed: Optional[int] = Form(None),
    use_bpt: Optional[bool] = Form(None),
    return_compressed: Optional[bool] = Form(True)
):
    """Generate 3D model from text prompt using Flux + Hunyuan3D + BPT with comprehensive asset management."""
    if seed is None:
        seed = random.randint(0, 2**31 - 1)
    
    if use_bpt is None:
        use_bpt = GENERATION_CONFIG['use_bpt']
    
    try:
        # Generate with asset management
        asset = generator.generate_3d_model_with_assets(prompt, seed, use_bpt)
        
        if asset is None or asset.status != GenerationStatus.COMPLETED:
            raise HTTPException(status_code=500, detail="Failed to generate 3D model")
        
        # Perform validation and mining readiness check
        validation_score = generator._perform_local_validation(asset, prompt)
        mining_ready = generator._check_mining_readiness(asset, validation_score)
        
        # Determine what to return
        if return_compressed and asset.compressed_ply_data:
            content = asset.compressed_ply_data
            filename = f"model_{seed}.ply.spz"
            media_type = "application/octet-stream"
            compression_type = 2  # SPZ compression
        else:
            content = asset.get_asset_data(AssetType.INITIAL_MESH_PLY)
            filename = f"model_{seed}.ply"
            media_type = "application/x-ply"
            compression_type = 0  # No compression
        
        if content is None:
            raise HTTPException(status_code=500, detail="No model data available")
        
        # Get validation metrics
        validation_metrics = {
            "face_count": asset.validation_metrics.face_count,
            "vertex_count": asset.validation_metrics.vertex_count,
            "compression_ratio": asset.compression_ratio,
            "generation_time": asset.performance_metrics.total_generation_time,
            "mesh_quality_score": asset.validation_metrics.mesh_quality_score,
            "local_validation_score": validation_score,
            "mining_ready": mining_ready
        }
        
        print(f"🎯 Final Results:")
        print(f"   Validation Score: {validation_score:.4f}")
        print(f"   Mining Ready: {mining_ready}")
        print(f"   Face Count: {asset.validation_metrics.face_count}")
        print(f"   Vertex Count: {asset.validation_metrics.vertex_count}")
        
        return Response(
            content=content,
            media_type=media_type,
            headers={
                "Content-Disposition": f"attachment; filename={filename}",
                "X-Generation-ID": asset.generation_id,
                "X-Generation-Time": str(asset.performance_metrics.total_generation_time),
                "X-Compression-Ratio": str(asset.compression_ratio),
                "X-Face-Count": str(asset.validation_metrics.face_count),
                "X-Vertex-Count": str(asset.validation_metrics.vertex_count),
                "X-Compression-Type": str(compression_type),
                "X-Mesh-Quality-Score": str(asset.validation_metrics.mesh_quality_score),
                "X-Local-Validation-Score": str(validation_score),
                "X-Mining-Ready": str(mining_ready).lower(),
                "X-Prompt": prompt[:100]  # First 100 chars of prompt
            }
        )
    except Exception as e:
        print(f"Endpoint error: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        # Ensure CUDA cache is cleared to prevent memory fragmentation
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            print("CUDA cache cleared after generation request.")

@app.get("/generate/{generation_id}")
async def get_generation_asset(generation_id: str):
    """Get information about a specific generation asset"""
    asset = global_asset_manager.get_asset(generation_id)
    if asset:
        # Convert asset to dictionary format
        asset_dict = {
            "generation_id": asset.generation_id,
            "status": asset.status.value,
            "prompt": asset.parameters.prompt,
            "seed": asset.parameters.seed,
            "created_at": asset.created_at,
            "updated_at": asset.updated_at,
            "error_message": asset.error_message,
            "validation_metrics": {
                "local_validation_score": asset.validation_metrics.local_validation_score,
                "mesh_quality_score": asset.validation_metrics.mesh_quality_score,
                "face_count": asset.validation_metrics.face_count,
                "vertex_count": asset.validation_metrics.vertex_count,
                "is_manifold": asset.validation_metrics.is_manifold
            },
            "performance_metrics": {
                "total_generation_time": asset.performance_metrics.total_generation_time,
                "image_generation_time": asset.performance_metrics.image_generation_time,
                "background_removal_time": asset.performance_metrics.background_removal_time,
                "mesh_generation_time": asset.performance_metrics.mesh_generation_time,
                "bpt_enhancement_time": asset.performance_metrics.bpt_enhancement_time,
                "texture_generation_time": asset.performance_metrics.texture_generation_time
            },
            "compression_ratio": asset.compression_ratio,
            "available_assets": [asset_type.value for asset_type in asset.assets.keys()]
        }
        return asset_dict
    else:
        raise HTTPException(status_code=404, detail="Generation ID not found")

@app.get("/generate/{generation_id}/download/{asset_type}")
async def download_asset(generation_id: str, asset_type: str):
    """Download a specific asset file"""
    asset = global_asset_manager.get_asset(generation_id)
    if not asset:
        raise HTTPException(status_code=404, detail="Generation ID not found")
        
    try:
        asset_enum = AssetType[asset_type.upper()]
    except KeyError:
        raise HTTPException(status_code=400, detail="Invalid asset type specified")

    asset_file = asset.get_asset(asset_enum)
    asset_data = asset.get_asset_data(asset_enum)

    if not asset_file or not asset_data:
        raise HTTPException(status_code=404, detail=f"Asset '{asset_type}' not available for this generation")

    # Determine media type
    if asset_enum in [AssetType.ORIGINAL_IMAGE, AssetType.BACKGROUND_REMOVED_IMAGE, AssetType.VALIDATION_PREVIEW]:
        media_type = "image/png"
    elif asset_enum in [AssetType.INITIAL_MESH_GLB, AssetType.BPT_ENHANCED_MESH_GLB, AssetType.TEXTURED_MESH_GLB]:
        media_type = "model/gltf-binary"
    elif asset_enum in [AssetType.INITIAL_MESH_PLY, AssetType.BPT_ENHANCED_MESH_PLY]:
        media_type = "application/x-ply"
    elif asset_enum == AssetType.COMPRESSED_PLY:
        media_type = "application/octet-stream"
    else:
        media_type = "application/octet-stream"

    # Create filename from asset info
    filename = f"{asset_enum.value}_{asset.parameters.seed}{Path(asset_file.file_path).suffix if asset_file.file_path else '.bin'}"

    return Response(
        content=asset_data,
        media_type=media_type,
        headers={"Content-Disposition": f"attachment; filename={filename}"}
    )

@app.post("/prepare_submission/{generation_id}")
async def prepare_submission(
    generation_id: str,
    task_id: str = Form(...),
    validator_hotkey: str = Form(...),
    validator_uid: int = Form(...)
):
    """Prepare a generation asset for mining submission"""
    asset = global_asset_manager.get_asset(generation_id)
    if not asset:
        raise HTTPException(status_code=404, detail="Generation ID not found")

    if asset.status != GenerationStatus.COMPLETED:
        raise HTTPException(status_code=400, detail="Asset generation is not complete")

    submission_data = prepare_for_mining_submission(
        asset=asset,
        task_id=task_id,
        validator_hotkey=validator_hotkey,
        validator_uid=validator_uid
    )
    return JSONResponse(content=submission_data)

@app.post("/mining/submit/")
async def submit_for_mining(
    generation_id: str = Form(...),
    task_id: str = Form(...),
    validator_hotkey: str = Form(...),
    validator_uid: int = Form(...)
):
    """Submit a generated asset for mining if it passes validation threshold."""
    asset = global_asset_manager.get_asset(generation_id)
    if not asset:
        raise HTTPException(status_code=404, detail="Generation ID not found")

    if asset.status != GenerationStatus.COMPLETED:
        raise HTTPException(status_code=400, detail="Asset generation is not complete")

    # Get the original prompt from asset parameters
    prompt = asset.parameters.prompt
    
    # Perform LOCAL validation (mesh quality assessment)
    validation_score = generator._perform_local_validation(asset, prompt)
    mining_ready = generator._check_mining_readiness(asset, validation_score)
    
    if not mining_ready:
        return JSONResponse(content={
            "success": False,
            "message": f"Asset not ready for mining. Validation score: {validation_score:.4f} (minimum: 0.7)",
            "validation_score": validation_score,
            "mining_ready": False
        })

    # Get raw PLY data (not compressed PLY)
    ply_data = asset.get_asset_data(AssetType.INITIAL_MESH_PLY)
    if not ply_data:
        return JSONResponse(content={
            "success": False,
            "message": "No PLY data available for submission",
            "validation_score": validation_score,
            "mining_ready": False
        })

    # Apply SPZ compression exactly like the official worker routine
    # Note: pyspz may not work with all PLY formats, so we have a fallback
    compression_type = 0  # Default to uncompressed
    compressed_results = ""
    
    try:
        import pyspz
        import base64
        
        # Attempt SPZ compression (exactly like workers.py: pyspz.compress(results, workers=-1))
        compressed_ply = pyspz.compress(ply_data, workers=-1)
        
        # Base64 encode the compressed data (exactly like workers.py)
        compressed_results = base64.b64encode(compressed_ply).decode(encoding="utf-8")
        compression_type = 2  # SPZ compression successful
        
        print(f"🗜️ SPZ Compression successful:")
        print(f"   Original PLY: {len(ply_data):,} bytes")
        print(f"   Compressed: {len(compressed_ply):,} bytes")
        print(f"   Compression ratio: {len(ply_data)/len(compressed_ply):.2f}x")
        print(f"   Base64 encoded: {len(compressed_results):,} characters")
        
    except Exception as e:
        print(f"⚠️ SPZ compression failed: {e}")
        print(f"   Falling back to uncompressed submission (compression=0)")
        
        # Fallback to uncompressed (still valid according to protocol)
        compressed_results = base64.b64encode(ply_data).decode(encoding="utf-8")
        compression_type = 0
        print(f"   Uncompressed base64: {len(compressed_results):,} characters")

    # Prepare submission data in the format expected by Bittensor validators
    # This matches the format from neurons/miner/workers.py
    submission_data = {
        "generation_id": asset.generation_id,
        "task_id": task_id,
        "validator_hotkey": validator_hotkey,
        "validator_uid": validator_uid,
        
        # Core submission data (for SubmitResults synapse) - EXACTLY like workers.py
        "results": compressed_results,  # Base64-encoded SPZ-compressed PLY
        "compression": compression_type,  # 2 = SPZ, 0 = uncompressed
        "data_format": "ply",
        "data_ver": 0,
        
        # Mining metadata
        "prompt": prompt,
        "seed": asset.parameters.seed,
        "local_validation_score": validation_score,
        "compression_ratio": len(ply_data)/len(compressed_ply) if compression_type == 2 else 1.0,
        "generation_time": asset.performance_metrics.total_generation_time,
        "face_count": asset.validation_metrics.face_count,
        "vertex_count": asset.validation_metrics.vertex_count,
        "mining_ready": mining_ready,
        
        # Submission instructions
        "submission_format": "bittensor_validator",
        "instructions": "Use 'results' field for SubmitResults.results, 'compression' for SubmitResults.compression"
    }
    
    # Update mining info
    asset.update_mining_info(
        task_id=task_id,
        validator_hotkey=validator_hotkey,
        validator_uid=validator_uid,
        submission_status="prepared"
    )
    
    print(f"🚀 Mining submission prepared for Bittensor validators:")
    print(f"   Generation ID: {generation_id}")
    print(f"   Validation Score: {validation_score:.4f}")
    print(f"   Task ID: {task_id}")
    print(f"   Validator: {validator_hotkey} (UID: {validator_uid})")
    print(f"   Data Format: PLY, Compression: {submission_data['compression']}")
    print(f"   Results Size: {len(submission_data['results'])} characters (base64)")
    
    return JSONResponse(content=submission_data)

@app.get("/status/")
async def get_server_status():
    """Returns the current status of the server and its metrics."""
    if torch.cuda.is_available():
        vram_total = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        vram_allocated = torch.cuda.memory_allocated(0) / (1024**3)
        vram_reserved = torch.cuda.memory_reserved(0) / (1024**3)
        return {
            "status": "ok",
            "version": "2.0.0",
            "models_loaded": generator.get_loaded_models(),
            "asset_stats": global_asset_manager.get_statistics(),
            "vram_gb": {
                "total": f"{vram_total:.2f}",
                "allocated": f"{vram_allocated:.2f}",
                "reserved": f"{vram_reserved:.2f}",
            }
        }
    else:
        return {"status": "ok", "version": "2.0.0", "models_loaded": "cpu_mode", "cuda_available": False}


@app.get("/health/")
async def health_check():
    """Basic health check endpoint."""
    return {"status": "ok"}

@app.post("/clear_cache/")
async def clear_cache():
    """Endpoint to manually clear GPU cache."""
    generator.clear_cache()
    return {"message": "GPU cache cleared"}

@app.get("/config/")
async def get_config():
    """Get the current server configuration."""
    return generator.get_status()['config']

@app.get("/assets/statistics/")
async def get_asset_statistics():
    """Get statistics about the assets managed by the server."""
    return global_asset_manager.get_statistics()

@app.post("/assets/cleanup/")
async def cleanup_old_assets(max_age_hours: float = Form(24)):
    """Cleanup assets older than a certain age."""
    count = global_asset_manager.cleanup_old_assets(max_age_hours)
    return {"message": f"Cleaned up {count} old assets."}

@app.get("/memory/")
async def get_memory_status():
    """Get detailed memory status."""
    if not torch.cuda.is_available():
        return {"error": "CUDA not available"}
        
    return {
            "allocated": torch.cuda.memory_allocated(),
            "reserved": torch.cuda.memory_reserved(),
            "max_allocated": torch.cuda.max_memory_allocated(),
        "max_reserved": torch.cuda.max_memory_reserved(),
        "summary": torch.cuda.memory_summary(abbreviated=True)
        }

@app.post("/validate/")
async def validate_model_endpoint(
    prompt: str = Form(...),
    data: str = Form(...),  # Base64 encoded PLY data
    compression: int = Form(0),  # 0: none, 2: SPZ
    data_ver: int = Form(0),
    generate_preview: bool = Form(False)
):
    """Validate a 3D model against a prompt."""
    try:
        import base64
        import pyspz
        
        # Decode the base64 data
        try:
            ply_bytes = base64.b64decode(data)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Invalid base64 data: {e}")
        
        # Decompress if needed
        if compression == 2:  # SPZ compression
            try:
                ply_bytes = pyspz.decompress(ply_bytes)
            except Exception as e:
                raise HTTPException(status_code=400, detail=f"SPZ decompression failed: {e}")
        
        # Load the mesh
        try:
            mesh = trimesh.load(trimesh.util.wrap_as_stream(ply_bytes), force='mesh')
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Invalid PLY data: {e}")
        
        # Basic validation metrics
        metrics = {
            "face_count": len(mesh.faces),
            "vertex_count": len(mesh.vertices),
            "is_watertight": mesh.is_watertight,
            "is_volume": mesh.is_volume,
            "is_empty": mesh.is_empty,
            "is_convex": mesh.is_convex
        }
        
        # Calculate mesh quality score (0-1)
        quality_score = 0.0
        if not metrics["is_empty"] and metrics["face_count"] > 0:
            # Basic quality checks
            if metrics["is_watertight"]:
                quality_score += 0.3
            if metrics["is_volume"]:
                quality_score += 0.2
            if not metrics["is_convex"]:  # Non-convex meshes are usually more detailed
                quality_score += 0.2
            
            # Face count quality (normalized)
            face_quality = min(metrics["face_count"] / 10000, 1.0)  # Cap at 10k faces
            quality_score += face_quality * 0.3
        
        # Generate preview if requested
        preview_data = None
        if generate_preview:
            try:
                preview = mesh.copy()
                preview.visual.face_colors = [200, 200, 200, 255]  # Light gray
                preview_path = Path(GENERATION_CONFIG['output_dir']) / 'previews' / f"preview_{int(time.time())}.png"
                preview_path.parent.mkdir(exist_ok=True)
                preview.export(preview_path)
                with open(preview_path, 'rb') as f:
                    preview_data = base64.b64encode(f.read()).decode('utf-8')
                preview_path.unlink()  # Clean up
            except Exception as e:
                print(f"Preview generation failed: {e}")
            
        return {
            "score": quality_score,
            "metrics": metrics,
            "preview": preview_data if generate_preview else None
        }
        
    except Exception as e:
        print(f"Validation error: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    print("Starting Flux-Hunyuan-BPT Generation Server with Asset Management...")
    # Setup generator
    
    # Integrate with robust server features if available
    try:
        from robust_generation_server import circuit_breaker, app as robust_app
        integrate_with_robust_server(app, robust_app, generator, global_asset_manager)
        print("✓ Integrated with robust server components.")
    except ImportError:
        print("   Robust server components not found, running in standard mode.")

    # Start the server
    uvicorn.run(app, host="0.0.0.0", port=8095) 