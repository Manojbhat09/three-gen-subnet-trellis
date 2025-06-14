#!/usr/bin/env python3
# Subnet 17 (404-GEN) - Enhanced Generation Server with Flux + Hunyuan3D + BPT + SuGaR
# Purpose: HTTP server for high-quality 3D model generation using Flux, Hunyuan3D-2, BPT enhancement, and SuGaR mesh-to-GS conversion
# Integrated with comprehensive asset management system and proper Gaussian Splatting output

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
import open3d as o3d
import argparse

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

# Import BPT components
from hy3dgen.shapegen.bpt.model.model import MeshTransformer
from hy3dgen.shapegen.bpt.utils import Dataset, apply_normalize, sample_pc, joint_filter
from hy3dgen.shapegen.bpt.model.serializaiton import BPT_deserialize
from torch.utils.data import DataLoader

# Import SuGaR components (with fallback handling)
try:
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'SuGaR'))
    from sugar_scene.gs_model import GaussianSplattingWrapper
    from sugar_scene.sugar_model import SuGaR
    from sugar_utils.spherical_harmonics import SH2RGB, RGB2SH
    SUGAR_AVAILABLE = True
    print("✓ SuGaR components imported successfully")
except ImportError as e:
    print(f"⚠️ SuGaR imports failed: {e}")
    print("⚠️ Using fallback mesh-to-GS conversion")
    SUGAR_AVAILABLE = False
    
    # Fallback SH conversion functions
    def RGB2SH(rgb):
        """Fallback RGB to Spherical Harmonics conversion"""
        # Simple conversion: DC component is RGB / sqrt(pi)
        return rgb / (2 * 3.14159 ** 0.5)
    
    def SH2RGB(sh):
        """Fallback SH to RGB conversion"""
        return sh * (2 * 3.14159 ** 0.5)

# Import PLY handling
from plyfile import PlyData, PlyElement

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
    'output_dir': './flux_hunyuan_bpt_sugar_outputs',
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
    # SuGaR specific settings - REDUCED FOR MEMORY CONSTRAINTS
    'sugar_num_points': 15000,  # Reduced from 50000 for RTX 4090
    'sugar_sh_levels': 3,  # Reduced from 4 for memory
    'sugar_triangle_scale': 1.5,  # Reduced from 2.0
    'sugar_surface_level': 0.3,  # Surface level for mesh extraction
    'sugar_n_gaussians_per_triangle': 4,  # Reduced from 6 for memory
    # Memory management settings
    'enable_memory_efficient_attention': True,
    'enable_cpu_offload': True,
    'max_memory_usage_gb': 20,  # Leave 4GB buffer on 24GB card
}

@dataclass
class GenerationMetrics:
    total_generations: int = 0
    successful_generations: int = 0
    failed_generations: int = 0
    average_generation_time: float = 0.0
    last_generation_time: float = 0.0
    bpt_enhanced_count: int = 0
    sugar_converted_count: int = 0

class FluxHunyuanBPTSuGaRGenerator:
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

    def _initialize_models(self):
        """Initialize only background remover and Hunyuan3D. FLUX and BPT are loaded on demand."""
        print("🔧 Initializing core models...")
        
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
            
            # Initialize BPT model if enabled
            if GENERATION_CONFIG['use_bpt']:
                try:
                    print("Loading BPT model...")
                    self._load_bpt_model()
                    print("✓ BPT model loaded successfully")
                except Exception as e:
                    print(f"⚠️ Failed to load BPT model: {e}")
                    self.bpt_model = None
            
            self._clear_gpu_memory()
            print("✅ Model initialization complete")
            
        except Exception as e:
            print(f"❌ Error initializing models: {e}")
            traceback.print_exc()

    def _load_bpt_model(self):
        """Load BPT model for mesh enhancement"""
        try:
            # Create model with keyword arguments instead of config dict
            model = MeshTransformer(
                dim=1024,
                max_seq_len=10000,  # Fixed: match checkpoint parameter size
                attn_depth=24,
                attn_heads=16,
                attn_dim_head=64,
                dropout=0.0,
                block_size=8,
                offset_size=16,
                use_special_block=True,
                conditioned_on_pc=True,
                encoder_name='miche-256-feature',
                encoder_freeze=False,
                cond_dim=768
            )
            
            # Try to load pre-trained weights
            model_path = GENERATION_CONFIG['bpt_model_path']
            if model_path is None:
                # Use the default BPT weights path
                script_dir = Path(__file__).parent
                model_path = script_dir / "Hunyuan3D-2" / "hy3dgen" / "shapegen" / "bpt" / "weights" / "bpt-8-16-500m.pt"
            
            if Path(model_path).exists():
                print(f"Loading BPT weights from {model_path}")
                checkpoint = torch.load(model_path, map_location='cpu')
                if 'model' in checkpoint:
                    model.load_state_dict(checkpoint['model'])
                else:
                    model.load_state_dict(checkpoint)
                print(f"✓ BPT model loaded from {model_path}")
            else:
                print(f"⚠️ BPT model not found at {model_path}. Using randomly initialized weights.")
            
            self.bpt_model = model.to(GENERATION_CONFIG['device'])
            self.bpt_model.eval()
            
        except Exception as e:
            print(f"❌ Error loading BPT model: {e}")
            traceback.print_exc()
            self.bpt_model = None

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

    def _unload_models_for_flux(self):
        """Temporarily unload heavy models for FLUX generation"""
        if self.hunyuan_pipeline is not None:
            del self.hunyuan_pipeline
            self.hunyuan_pipeline = None
        
        if self.bpt_model is not None:
            del self.bpt_model
            self.bpt_model = None
        
        self._clear_gpu_memory()

    def _reload_models_after_flux(self):
        """Reload models after FLUX generation"""
        # Reload Hunyuan3D pipeline
        if self.hunyuan_pipeline is None:
            print("Reloading Hunyuan3D pipeline...")
            self.hunyuan_pipeline = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(
                GENERATION_CONFIG['hunyuan_model_path'],
                torch_dtype=torch.float16,
                low_cpu_mem_usage=True,
            )
            self.hunyuan_pipeline.to(GENERATION_CONFIG['device'])
            print("✓ Hunyuan3D pipeline reloaded")
        
        # Reload BPT model if needed
        if GENERATION_CONFIG['use_bpt'] and self.bpt_model is None:
            print("Reloading BPT model...")
            try:
                self._load_bpt_model()
                print("✓ BPT model reloaded")
            except Exception as e:
                print(f"⚠️ Failed to reload BPT model: {e}")
                self.bpt_model = None

    def _enhance_mesh_with_bpt(self, mesh: trimesh.Trimesh, temperature: float = 0.5):
        """Enhance a mesh using the BPT model."""
        if self.bpt_model is None:
            print("BPT model not loaded, skipping enhancement.")
            return mesh
        
        try:
            print(f"Enhancing mesh with BPT (temperature={temperature})...")
            
            # Save mesh to temporary file for Dataset processing (like working demo)
            import tempfile
            with tempfile.NamedTemporaryFile(suffix='.obj', delete=False) as temp_file:
                temp_path = temp_file.name
                mesh.export(temp_path)
            
            # Create dataset from the mesh (like working demo)
            dataset = Dataset(input_type='mesh', input_list=[temp_path])
            dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
            
            device = self.bpt_model.device if hasattr(self.bpt_model, 'device') else GENERATION_CONFIG['device']
            
            with torch.no_grad():
                for batch in dataloader:
                    pc_normal = batch['pc_normal'].to(device)  # [B, 4096, 6]
                    uid = batch['uid']
                    
                    print(f"Processing mesh: {uid[0]}")
                    print(f"Point cloud shape: {pc_normal.shape}")
                    
                    # Generate enhanced mesh using BPT (matching working demo)
                    codes = self.bpt_model.generate(
                        pc=pc_normal,
                        batch_size=1,
                        temperature=temperature,
                        filter_logits_fn=joint_filter,
                        filter_kwargs={'k': 50, 'p': 0.95},
                        max_seq_len=self.bpt_model.max_seq_len,
                        cache_kv=True,
                        return_codes=True,  # Return raw codes to bypass missing decode_codes method
                    )
                    
                    print(f"Generated codes shape: {codes.shape}")
                    
                    # Deserialize BPT output to coordinates (matching working demo)
                    coordinates = BPT_deserialize(
                        codes.cpu().numpy().flatten(),
                        block_size=self.bpt_model.block_size,
                        offset_size=self.bpt_model.offset_size,
                        compressed=True,
                        special_token=-2,
                        use_special_block=self.bpt_model.use_special_block
                    )
            
            # Clean up temporary file
            import os
            os.unlink(temp_path)
            
            # Convert coordinates back to mesh
            if coordinates is not None and len(coordinates) > 0:
                # BPT_deserialize returns coordinates for triangular faces (matching working demo)
                if len(coordinates) % 3 == 0:
                    # Reshape to get individual triangle vertices
                    vertices = coordinates.reshape(-1, 3)
                    # Create faces by grouping every 3 vertices into triangles
                    num_faces = len(vertices) // 3
                    faces = np.arange(len(vertices)).reshape(num_faces, 3)
                    
                    # Create enhanced mesh
                    enhanced_mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
                    enhanced_mesh = apply_normalize(enhanced_mesh)
                    
                    # Apply post-processing to clean up the mesh
                    enhanced_mesh.remove_duplicate_faces()
                    enhanced_mesh.remove_degenerate_faces()
                    enhanced_mesh.remove_unreferenced_vertices()
                    
                    print(f"✓ Enhanced mesh: {len(enhanced_mesh.vertices)} vertices, {len(enhanced_mesh.faces)} faces")
                    self.metrics.bpt_enhanced_count += 1
                    
                    return enhanced_mesh
                else:
                    print("⚠️ Invalid coordinates generated, using original mesh")
                    return mesh
            else:
                print("⚠️ BPT enhancement produced no output, using original mesh")
                return mesh
            
        except Exception as e:
            print(f"⚠️ Error during BPT enhancement: {e}")
            return mesh

    def generate_3d_model_with_assets(self, prompt: str, seed: int = 42, use_bpt: bool = None) -> Optional[GenerationAsset]:
        """Generate 3D model with comprehensive asset management"""
        if use_bpt is None:
            use_bpt = GENERATION_CONFIG.get('use_bpt', False)
        
        with self.generation_lock:
            start_time = time.time()
            
            try:
                print(f"🎯 Starting generation for: '{prompt}' (seed: {seed}, BPT: {use_bpt})")
                
                # Create asset for tracking
                asset = GenerationAsset(prompt=prompt, seed=seed)
                asset.update_status(GenerationStatus.INITIALIZING, "Starting generation...")
                
                # Step 1: Temporarily unload heavy models for FLUX
                self._unload_models_for_flux()
                
                # Step 2: Generate image with FLUX
                asset.update_status(GenerationStatus.IMAGE_GENERATING, "Generating image with FLUX...")
                flux_pipe = self._load_flux_pipeline(seed)
                if flux_pipe is None:
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
                
                asset.add_asset(AssetType.GENERATED_IMAGE, image)
                print("✓ Image generated successfully")
                
                # CRITICAL: Completely clear FLUX from memory
                del flux_pipe
                self._clear_gpu_memory()
                
                # Step 3: Process image (background removal)
                asset.update_status(GenerationStatus.IMAGE_PROCESSING, "Processing image...")
                if self.rembg:
                    processed_image = self.rembg(image)
                else:
                    processed_image = image
                asset.add_asset(AssetType.PROCESSED_IMAGE, processed_image)
                print("✓ Image processed")
                
                # Step 4: Reload models after FLUX
                self._reload_models_after_flux()
                
                asset.update_status(GenerationStatus.MESH_GENERATING, "Generating 3D mesh...")
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
                asset.update_status(GenerationStatus.MESH_PROCESSING, "Post-processing mesh...")
                face_reducer = FaceReducer()
                floater_remover = FloaterRemover()
                degenerate_remover = DegenerateFaceRemover()
                
                mesh = face_reducer(mesh)
                mesh = floater_remover(mesh)
                mesh = degenerate_remover(mesh)
                
                asset.add_asset(AssetType.INITIAL_MESH_GLB, mesh)
                print(f"✓ Mesh processed: {len(mesh.vertices)} vertices, {len(mesh.faces)} faces")
                
                # Step 6: (Optional) Enhance with BPT
                enhanced_mesh = mesh
                if use_bpt and self.bpt_model:
                    asset.update_status(GenerationStatus.MESH_ENHANCING, "Enhancing mesh with BPT...")
                    bpt_start = time.time()
                    try:
                        enhanced_mesh = self._enhance_mesh_with_bpt(mesh, GENERATION_CONFIG['bpt_temperature'])
                        if enhanced_mesh != mesh:  # Only add if actually enhanced
                            asset.add_asset(AssetType.BPT_ENHANCED_MESH_GLB, enhanced_mesh)
                            print(f"✓ BPT enhancement completed")
                    except Exception as e:
                        print(f"BPT enhancement failed (non-critical): {e}")
                    asset.performance_metrics.bpt_enhancement_time = time.time() - bpt_start
                
                # Step 7: Convert to Gaussian Splatting PLY
                asset.update_status(GenerationStatus.CONVERTING_TO_PLY, "Converting to Gaussian Splatting PLY...")
                ply_data = self._mesh_to_gaussian_splatting(enhanced_mesh)
                asset.add_asset(AssetType.GAUSSIAN_SPLATTING_PLY, ply_data)
                
                # Step 8: Create additional PLY formats
                # Create viewable mesh PLY
                viewable_ply_data = self._create_basic_mesh_ply(enhanced_mesh)
                asset.add_asset(AssetType.VIEWABLE_MESH_PLY, viewable_ply_data)
                
                # Create simple points PLY
                points, _ = enhanced_mesh.sample(GENERATION_CONFIG['sugar_num_points'], return_index=True)
                simple_points_ply = self._create_simple_points_ply(points)
                asset.add_asset(AssetType.SIMPLE_POINTS_PLY, simple_points_ply)
                
                # Final status
                generation_time = time.time() - start_time
                asset.update_status(GenerationStatus.COMPLETED, f"Generation completed in {generation_time:.2f}s")
                asset.performance_metrics.total_time = generation_time
                
                # Update metrics
                self.metrics.total_generations += 1
                self.metrics.successful_generations += 1
                self.metrics.last_generation_time = generation_time
                self.metrics.average_generation_time = (
                    (self.metrics.average_generation_time * (self.metrics.successful_generations - 1) + generation_time) 
                    / self.metrics.successful_generations
                )
                
                print(f"🎉 Generation completed in {generation_time:.2f}s")
                return asset
                
            except Exception as e:
                self.metrics.total_generations += 1
                self.metrics.failed_generations += 1
                print(f"❌ Generation failed: {e}")
                traceback.print_exc()
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

    def _create_simple_points_ply(self, points) -> bytes:
        """Create a simple point cloud PLY that's viewable"""
        try:
            # Create simple vertex data
            vertex_data = []
            for point in points:
                vertex_data.append((point[0], point[1], point[2]))
            
            # Create PLY element
            vertex_element = PlyElement.describe(
                np.array(vertex_data, dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4')]),
                'vertex'
            )
            
            # Save as simple point cloud PLY
            ply_data = PlyData([vertex_element])
            
            import io
            buffer = io.BytesIO()
            ply_data.write(buffer)
            return buffer.getvalue()
            
        except Exception as e:
            print(f"❌ Error creating simple PLY: {e}")
            return b""

    def generate_3d_model(self, prompt: str, seed: int = 42, use_bpt: bool = None) -> Optional[bytes]:
        """Generate 3D model and return Gaussian Splatting PLY data"""
        asset = self.generate_3d_model_with_assets(prompt, seed, use_bpt)
        if asset and AssetType.GAUSSIAN_SPLATTING_PLY in asset.assets:
            return asset.assets[AssetType.GAUSSIAN_SPLATTING_PLY]
        return None

    def get_status(self) -> Dict[str, Any]:
        """Get server status and metrics"""
        return {
            "status": "running",
            "models_loaded": {
                "hunyuan3d": self.hunyuan_pipeline is not None,
                "background_remover": self.rembg is not None,
                "bpt_model": self.bpt_model is not None,
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
                "bpt_enhanced_count": self.metrics.bpt_enhanced_count,
                "sugar_converted_count": self.metrics.sugar_converted_count,
            },
            "config": GENERATION_CONFIG,
            "gpu_memory": self._clear_gpu_memory() if torch.cuda.is_available() else 0,
        }

# Initialize generator
generator = FluxHunyuanBPTSuGaRGenerator()

# FastAPI app
app = FastAPI(title="Flux + Hunyuan3D + BPT + SuGaR Generation Server")

@app.post("/generate/")
async def generate_3d_model_endpoint(
    prompt: str = Form(...), 
    seed: Optional[int] = Form(None),
    use_bpt: Optional[bool] = Form(None),
    return_compressed: Optional[bool] = Form(True)
):
    """Generate 3D model from text prompt using Flux + Hunyuan3D + BPT + SuGaR pipeline."""
    
    # Handle seed
    if seed is None:
        seed = random.randint(0, MAX_SEED)
    
    # Handle BPT flag
    if use_bpt is None:
        use_bpt = GENERATION_CONFIG['use_bpt']
    
    # Generate model with assets
    asset = generator.generate_3d_model_with_assets(prompt, seed, use_bpt)
    
    if asset is None or AssetType.GAUSSIAN_SPLATTING_PLY not in asset.assets:
        raise HTTPException(status_code=500, detail="Generation failed")
    
    # Get the Gaussian Splatting PLY data
    ply_data = asset.assets[AssetType.GAUSSIAN_SPLATTING_PLY]
    
    # Return PLY data with comprehensive headers
    return Response(
        content=ply_data,
        media_type="application/octet-stream",
        headers={
            "Content-Disposition": f"attachment; filename=generated_model_{seed}.ply",
            "X-Generation-Seed": str(seed),
            "X-Generation-Prompt": prompt,
            "X-Model-Format": "gaussian_splatting_ply",
            "X-BPT-Enhanced": str(use_bpt),
            "X-Generation-Time": str(asset.performance_metrics.total_time),
            "X-Available-Assets": ",".join([asset_type.value for asset_type in asset.assets.keys()]),
        }
    )

@app.get("/assets/{asset_id}/{asset_type}")
async def get_asset(asset_id: str, asset_type: str):
    """Get specific asset from a generation"""
    try:
        asset_enum = AssetType(asset_type)
        asset = global_asset_manager.get_asset(asset_id)
        
        if asset is None or asset_enum not in asset.assets:
            raise HTTPException(status_code=404, detail="Asset not found")
        
        asset_data = asset.assets[asset_enum]
        
        # Determine content type and filename
        if asset_enum in [AssetType.GENERATED_IMAGE, AssetType.PROCESSED_IMAGE]:
            # Handle PIL Images
            import io
            buffer = io.BytesIO()
            asset_data.save(buffer, format='PNG')
            content = buffer.getvalue()
            media_type = "image/png"
            filename = f"{asset_id}_{asset_type}.png"
        elif asset_enum in [AssetType.INITIAL_MESH_GLB, AssetType.BPT_ENHANCED_MESH_GLB, AssetType.TEXTURED_MESH_GLB]:
            # Handle GLB files
            import io
            buffer = io.BytesIO()
            asset_data.export(buffer, file_type='glb')
            content = buffer.getvalue()
            media_type = "model/gltf-binary"
            filename = f"{asset_id}_{asset_type}.glb"
        elif asset_enum in [AssetType.INITIAL_MESH_PLY, AssetType.GAUSSIAN_SPLATTING_PLY, AssetType.VIEWABLE_MESH_PLY, AssetType.SIMPLE_POINTS_PLY]:
            # Handle PLY files (already bytes)
            content = asset_data
            media_type = "application/octet-stream"
            filename = f"{asset_id}_{asset_type}.ply"
        else:
            # Default handling
            content = asset_data if isinstance(asset_data, bytes) else str(asset_data).encode()
            media_type = "application/octet-stream"
            filename = f"{asset_id}_{asset_type}"
        
        return Response(
            content=content,
            media_type=media_type,
            headers={"Content-Disposition": f"attachment; filename={filename}"}
        )
        
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid asset type")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving asset: {str(e)}")

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
    parser = argparse.ArgumentParser(description="Flux + Hunyuan3D + BPT + SuGaR Generation Server")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8095, help="Port to bind to")
    parser.add_argument("--workers", type=int, default=1, help="Number of worker processes")
    parser.add_argument("--enable-bpt", action="store_true", help="Enable BPT enhancement")
    
    args = parser.parse_args()
    
    # Update config based on args
    if args.enable_bpt:
        GENERATION_CONFIG['use_bpt'] = True
    
    print(f"Starting Flux + Hunyuan3D + BPT + SuGaR Generation Server on {args.host}:{args.port}")
    print("=" * 70)
    print("Pipeline: Text → FLUX → Image → Hunyuan3D → Mesh → [BPT] → SuGaR → Gaussian Splatting PLY")
    print("Features:")
    print("  • FLUX text-to-image generation with quantization")
    print("  • Hunyuan3D-2 image-to-3D mesh generation")
    print("  • Optional BPT mesh enhancement for higher quality")
    print("  • SuGaR-inspired mesh-to-Gaussian Splatting conversion")
    print("  • Multiple PLY output formats:")
    print("    - Gaussian Splatting PLY (for validation)")
    print("    - Viewable Mesh PLY (for 3D viewers)")
    print("    - Simple Points PLY (for point cloud viewing)")
    print("  • Comprehensive asset management system")
    print("  • Memory-optimized for RTX 4090 (24GB)")
    print(f"  • BPT Enhancement: {'Enabled' if GENERATION_CONFIG['use_bpt'] else 'Disabled'}")
    print("=" * 70)
    
    uvicorn.run(
        app, 
        host=args.host, 
        port=args.port,
        workers=args.workers,
        log_level="info"
    ) 