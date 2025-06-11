#!/usr/bin/env python3
"""
Advanced Flux + TRELLIS + BPT pipeline with shape optimization and improved re-texturing

Pipeline:
1. Text → Flux → Image
2. Image → Background Removal  
3. Image → TRELLIS → 3D Assets (Gaussians, Raw Mesh)
4. Raw Mesh → BPT Enhancement → High-Detail Mesh (8k+ faces)
5. High-Detail Mesh → Shape Optimization → Optimized Mesh
6. Optimized Mesh → Advanced Texture Baking → Final Textured GLB

BPT Integration:
- Uses Blocked and Patchified Tokenization for mesh enhancement
- Generates 8k+ face meshes with fine details
- Compressive tokenization reduces sequence length by ~75%
- Memory-efficient GPU management throughout pipeline
"""

import os
import datetime
import torch
import numpy as np
import gc
import re
import trimesh
import yaml
from PIL import Image
from pathlib import Path
from typing import List, Dict, Tuple

# TRELLIS imports
os.environ['SPCONV_ALGO'] = 'native'
import imageio
from trellis.pipelines import TrellisImageTo3DPipeline
from trellis.utils import render_utils, postprocessing_utils

# Flux imports
from diffusers import FluxTransformer2DModel, FluxPipeline, BitsAndBytesConfig, GGUFQuantizationConfig
from transformers import T5EncoderModel, BitsAndBytesConfig as BitsAndBytesConfigTF

# Background removal
try:
    from rembg import remove, new_session
    REMBG_AVAILABLE = True
except ImportError:
    print("Warning: rembg not available. Background removal will be skipped.")
    REMBG_AVAILABLE = False

# Hunyuan3D shape optimization tools
try:
    from hy3dgen.shapegen import FaceReducer, FloaterRemover, DegenerateFaceRemover, MeshSimplifier
    SHAPEGEN_AVAILABLE = True
    print("✅ Hunyuan3D shape optimization tools loaded")
except ImportError:
    print("⚠️  Warning: Hunyuan3D shapegen not available. Shape optimization will be skipped.")
    SHAPEGEN_AVAILABLE = False

# Hunyuan3D texture painting (optional)
try:
    from hy3dgen.texgen import Hunyuan3DPaintPipeline
    HUNYUAN_TEXGEN_AVAILABLE = True
    print("✅ Hunyuan3D texture generation available")
except ImportError:
    print("⚠️  Warning: Hunyuan3D texture generation not available. Will use TRELLIS texture baking.")
    HUNYUAN_TEXGEN_AVAILABLE = False

# BPT imports for mesh enhancement
try:
    from hy3dgen.shapegen.bpt.model.model import MeshTransformer
    from hy3dgen.shapegen.bpt.utils import Dataset, apply_normalize, sample_pc, joint_filter
    from hy3dgen.shapegen.bpt.model.serializaiton import BPT_deserialize
    from torch.utils.data import DataLoader
    BPT_AVAILABLE = True
    print("✅ BPT mesh enhancement tools loaded")
except ImportError:
    print("⚠️  Warning: BPT not available. Mesh enhancement will be skipped.")
    BPT_AVAILABLE = False

# Constants
NUM_INFERENCE_STEPS = 8
MAX_SEED = np.iinfo(np.int32).max

def clear_gpu_memory():
    """Clear GPU memory cache"""
    gc.collect()
    torch.cuda.empty_cache()
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

def create_bpt_config():
    """Create BPT model configuration optimized for TRELLIS integration"""
    config = {
        'dim': 1024,
        'max_seq_len': 10000,
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
        'num_discrete_coors': 128,  # 2^7
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

def load_bpt_model(model_path=None, config=None, device="cuda"):
    """Load BPT model for mesh enhancement"""
    if config is None:
        config = create_bpt_config()
    
    # Initialize model
    model = MeshTransformer(**config)
    
    # Load pretrained weights if available
    if model_path is None:
        # Use the default BPT weights path
        script_dir = Path(__file__).resolve().parent
        model_path = script_dir.parent / "Hunyuan3D-2" / "hy3dgen" / "shapegen" / "bpt" / "weights" / "bpt-8-16-500m.pt"
    
    if os.path.exists(model_path):
        print(f"Loading BPT model from {model_path}")
        checkpoint = torch.load(model_path, map_location=device)
        
        # Handle different checkpoint formats
        if 'model' in checkpoint:
            # This is a training checkpoint with nested structure
            model_state_dict = checkpoint['model']
            print("Loading from training checkpoint format (nested 'model' key)")
        elif 'model_state_dict' in checkpoint:
            model_state_dict = checkpoint['model_state_dict']
            print("Loading from model_state_dict format")
        elif 'state_dict' in checkpoint:
            model_state_dict = checkpoint['state_dict']
            print("Loading from state_dict format")
        else:
            # Assume the checkpoint is the state dict itself
            model_state_dict = checkpoint
            print("Loading from direct state_dict format")
        
        # Load the state dict into the model
        try:
            model.load_state_dict(model_state_dict, strict=False)
            print("✅ BPT model weights loaded successfully")
        except Exception as e:
            print(f"⚠️  Warning: Could not load BPT weights perfectly: {e}")
            print("   Continuing with partially loaded or random weights...")
    else:
        print(f"Warning: BPT model weights not found at {model_path}, using randomly initialized weights")
    
    model.to(device)
    model.eval()
    return model

class BackgroundRemover:
    """Background remover using rembg with session management"""
    def __init__(self):
        if REMBG_AVAILABLE:
            self.session = new_session()
        else:
            self.session = None

    def __call__(self, image: Image.Image):
        if self.session is not None:
            output = remove(image, session=self.session, bgcolor=[255, 255, 255, 0])
            return output
        else:
            return image

class BPTMeshEnhancer:
    """BPT mesh enhancement using Blocked and Patchified Tokenization"""
    def __init__(self, device="cuda"):
        self.device = device
        self.model = None
        self.config = None
        if BPT_AVAILABLE:
            self.config = create_bpt_config()
            # Don't load model here - load it only when needed
            print("🔬 BPT mesh enhancer initialized (model will load when needed)")
        else:
            print("⚠️  BPT not available")
    
    def load_bpt_model(self):
        """Load BPT model when needed"""
        if self.model is None and BPT_AVAILABLE:
            print("📦 Loading BPT model...")
            self.model = load_bpt_model(config=self.config, device=self.device)
        return self.model
    
    def unload_bpt_model(self):
        """Unload BPT model to free GPU memory"""
        if self.model is not None:
            print("🧹 Unloading BPT model...")
            del self.model
            self.model = None
            clear_gpu_memory()
    
    def enhance_mesh_with_bpt(self, mesh_trimesh, temperature=0.5, batch_size=1, verbose=True):
        """Enhance mesh using BPT to generate high-detail version"""
        if not BPT_AVAILABLE:
            if verbose:
                print("⚠️  BPT enhancement not available, returning original mesh")
            return mesh_trimesh
        
        if verbose:
            print("🔬 Enhancing mesh with BPT...")
        
        try:
            # Load BPT model only when needed
            bpt_model = self.load_bpt_model()
            if bpt_model is None:
                if verbose:
                    print("⚠️  Could not load BPT model, returning original mesh")
                return mesh_trimesh
            
            # Save mesh temporarily for BPT processing
            temp_obj_path = "/tmp/temp_mesh_for_bpt.obj"
            mesh_trimesh.export(temp_obj_path)
            
            # Create dataset from the mesh
            dataset = Dataset(input_type='mesh', input_list=[temp_obj_path])
            dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
            
            enhanced_mesh = None
            
            with torch.no_grad():
                for batch in dataloader:
                    pc_normal = batch['pc_normal'].to(self.device)  # [B, 4096, 6]
                    uid = batch['uid']
                    
                    if verbose:
                        print(f"  - Processing mesh: {uid[0]}")
                        print(f"  - Point cloud shape: {pc_normal.shape}")
                    
                    # Generate enhanced mesh using BPT
                    try:
                        codes = bpt_model.generate(
                            pc=pc_normal,
                            batch_size=batch_size,
                            temperature=temperature,
                            filter_logits_fn=joint_filter,
                            filter_kwargs={'k': 50, 'p': 0.95},
                            max_seq_len=bpt_model.max_seq_len,
                            cache_kv=True,
                            return_codes=True,
                        )
                        
                        if verbose:
                            print(f"  - Generated codes shape: {codes.shape}")
                        
                        # Deserialize the codes to mesh coordinates
                        coordinates = BPT_deserialize(
                            codes.cpu().numpy().flatten(),
                            block_size=bpt_model.block_size,
                            offset_size=bpt_model.offset_size,
                            compressed=True,
                            special_token=-2,
                            use_special_block=bpt_model.use_special_block
                        )
                        
                        # Convert coordinates to trimesh object
                        if len(coordinates) > 0 and len(coordinates) % 3 == 0:
                            # BPT_deserialize returns coordinates for triangular faces
                            vertices = coordinates.reshape(-1, 3)
                            # Create faces by grouping every 3 vertices into triangles
                            num_faces = len(vertices) // 3
                            faces = np.arange(len(vertices)).reshape(num_faces, 3)
                            
                            # Create enhanced mesh
                            enhanced_mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
                            enhanced_mesh = apply_normalize(enhanced_mesh)
                            
                            if verbose:
                                print(f"  - Enhanced mesh: {len(vertices)} vertices, {len(faces)} faces")
                                print("  ✅ BPT enhancement successful!")
                        else:
                            if verbose:
                                print("  ❌ Invalid coordinates generated")
                            enhanced_mesh = mesh_trimesh
                        
                    except Exception as e:
                        if verbose:
                            print(f"  ❌ BPT generation failed: {e}")
                        enhanced_mesh = mesh_trimesh
            
            # Clean up temporary file
            if os.path.exists(temp_obj_path):
                os.remove(temp_obj_path)
            
            # Unload BPT model immediately to free GPU memory
            self.unload_bpt_model()
            
            return enhanced_mesh if enhanced_mesh is not None else mesh_trimesh
            
        except Exception as e:
            if verbose:
                print(f"❌ BPT mesh enhancement failed: {e}")
            # Make sure to unload model even if there was an error
            self.unload_bpt_model()
            return mesh_trimesh

class ShapeOptimizer:
    """Shape optimization using Hunyuan3D tools"""
    def __init__(self):
        if SHAPEGEN_AVAILABLE:
            self.face_reducer = FaceReducer()
            self.floater_remover = FloaterRemover()
            self.degenerate_face_remover = DegenerateFaceRemover()
            self.mesh_simplifier = MeshSimplifier()
        else:
            self.face_reducer = None
            self.floater_remover = None
            self.degenerate_face_remover = None
            self.mesh_simplifier = None
    
    def optimize_mesh(self, mesh, max_faces: int = 40000, verbose: bool = True):
        """Apply comprehensive mesh optimization"""
        if not SHAPEGEN_AVAILABLE:
            if verbose:
                print("⚠️  Shape optimization not available, returning original mesh")
            return mesh
        
        if verbose:
            print("🔧 Applying shape optimizations...")
        
        # Step 1: Remove floating disconnected components
        if verbose:
            print("  - Removing floaters...")
        mesh = self.floater_remover(mesh)
        
        # Step 2: Remove degenerate faces
        if verbose:
            print("  - Removing degenerate faces...")
        mesh = self.degenerate_face_remover(mesh)
        
        # Step 3: Reduce face count for optimization (conservative for BPT-enhanced meshes)
        current_faces = len(mesh.faces)
        if current_faces > max_faces:
            if verbose:
                print(f"  - Reducing faces from {current_faces} to max {max_faces}...")
            mesh = self.face_reducer(mesh, max_facenum=max_faces)
        else:
            if verbose:
                print(f"  - Mesh already within face limit ({current_faces} <= {max_faces})")
        
        # Step 4: Apply mesh simplification, if available
        if verbose:
            print("  - Applying mesh simplification...")
        
        if self.mesh_simplifier:
            simplifier_path = self.mesh_simplifier.executable
            if not (os.path.isfile(simplifier_path) and os.access(simplifier_path, os.X_OK)):
                if verbose:
                    print(f"    ⚠️  Mesh simplifier binary not found. Skipping.")
            else:
                try:
                    mesh = self.mesh_simplifier(mesh)
                except Exception as e:
                    if verbose:
                        print(f"    ⚠️  Mesh simplification failed: {e}")
        else:
            if verbose:
                print("    ⚠️  Mesh simplifier not initialized. Skipping.")

        if verbose:
            print("✅ Shape optimization complete!")
        
        return mesh

class IntelligentMeshSimplifier:
    """Intelligent mesh simplification using multiple algorithms"""
    def __init__(self):
        self.available_methods = []
        
        # Check for Open3D (best quality)
        try:
            import open3d as o3d
            self.available_methods.append('open3d')
            self.o3d = o3d
        except ImportError:
            self.o3d = None
        
        # Check for PyMeshLab
        try:
            import pymeshlab
            self.available_methods.append('pymeshlab')
            self.pymeshlab = pymeshlab
        except ImportError:
            self.pymeshlab = None
        
        # Trimesh is always available (basic but reliable)
        self.available_methods.append('trimesh')
        
        print(f"🔬 Mesh simplification methods available: {', '.join(self.available_methods)}")
    
    def simplify_with_open3d(self, mesh_trimesh, target_faces: int, verbose: bool = True):
        """Simplify using Open3D's quadric error metric decimation"""
        try:
            if verbose:
                print(f"    - Using Open3D QEM simplification...")
            
            # Convert to Open3D mesh
            o3d_mesh = self.o3d.geometry.TriangleMesh()
            o3d_mesh.vertices = self.o3d.utility.Vector3dVector(mesh_trimesh.vertices)
            o3d_mesh.triangles = self.o3d.utility.Vector3iVector(mesh_trimesh.faces)
            
            # Calculate reduction ratio
            current_faces = len(mesh_trimesh.faces)
            reduction_ratio = target_faces / current_faces
            
            if verbose:
                print(f"    - Reducing from {current_faces} to ~{target_faces} faces (ratio: {reduction_ratio:.3f})")
            
            # Apply quadric decimation
            simplified_mesh = o3d_mesh.simplify_quadric_decimation(target_number_of_triangles=target_faces)
            
            # Convert back to trimesh
            vertices = np.asarray(simplified_mesh.vertices)
            faces = np.asarray(simplified_mesh.triangles)
            
            result = trimesh.Trimesh(vertices=vertices, faces=faces)
            
            if verbose:
                print(f"    - Open3D result: {len(result.vertices)} vertices, {len(result.faces)} faces")
            
            return result
            
        except Exception as e:
            if verbose:
                print(f"    - Open3D simplification failed: {e}")
            return None
    
    def simplify_with_pymeshlab(self, mesh_trimesh, target_faces: int, verbose: bool = True):
        """Simplify using PyMeshLab's quadric edge collapse"""
        try:
            if verbose:
                print(f"    - Using PyMeshLab QEC simplification...")
            
            # Create MeshSet
            ms = self.pymeshlab.MeshSet()
            
            # Add mesh to MeshSet
            mesh = self.pymeshlab.Mesh(mesh_trimesh.vertices, mesh_trimesh.faces)
            ms.add_mesh(mesh)
            
            # Apply quadric edge collapse decimation
            ms.apply_filter('simplification_quadric_edge_collapse_decimation', 
                          targetfacenum=target_faces, preservenormal=True)
            
            # Get simplified mesh
            simplified = ms.current_mesh()
            vertices = simplified.vertex_matrix()
            faces = simplified.face_matrix()
            
            result = trimesh.Trimesh(vertices=vertices, faces=faces)
            
            if verbose:
                print(f"    - PyMeshLab result: {len(result.vertices)} vertices, {len(result.faces)} faces")
            
            return result
            
        except Exception as e:
            if verbose:
                print(f"    - PyMeshLab simplification failed: {e}")
            return None
    
    def simplify_with_trimesh(self, mesh_trimesh, target_faces: int, verbose: bool = True):
        """Simplify using Trimesh's built-in simplification"""
        try:
            if verbose:
                print(f"    - Using Trimesh simplification...")
            
            current_faces = len(mesh_trimesh.faces)
            reduction_ratio = target_faces / current_faces
            
            # Use Trimesh's simplify_quadric_decimation if available
            if hasattr(mesh_trimesh, 'simplify_quadric_decimation'):
                try:
                    simplified = mesh_trimesh.simplify_quadric_decimation(face_count=target_faces)
                    if verbose:
                        print(f"    - Trimesh QD result: {len(simplified.vertices)} vertices, {len(simplified.faces)} faces")
                    return simplified
                except:
                    pass
            
            # Fallback to basic vertex clustering
            voxel_size = mesh_trimesh.scale / (target_faces ** 0.5 * 2)
            simplified = mesh_trimesh.simplify_voxel_grid(voxel_size=voxel_size)
            
            if verbose:
                print(f"    - Trimesh voxel result: {len(simplified.vertices)} vertices, {len(simplified.faces)} faces")
            
            return simplified
            
        except Exception as e:
            if verbose:
                print(f"    - Trimesh simplification failed: {e}")
            return mesh_trimesh
    
    def intelligent_simplify(self, mesh_trimesh, target_faces: int = 2000, preferred_methods: List[str] = None, verbose: bool = True):
        """Apply intelligent simplification using the specified methods in order"""
        if verbose:
            print(f"🔬 Applying intelligent mesh simplification to {target_faces} faces...")
        
        # Use preferred methods if provided, otherwise use default order
        methods_to_try = preferred_methods if preferred_methods else ['open3d', 'pymeshlab', 'trimesh']
        
        if verbose:
            available_methods_str = ', '.join([m for m in methods_to_try if m in self.available_methods])
            print(f"    - Trying methods in order: {available_methods_str}")
        
        # Try methods in the specified order
        for method in methods_to_try:
            if method in self.available_methods:
                if verbose:
                    print(f"    - Attempting {method} simplification...")
                
                if method == 'open3d' and self.o3d:
                    result = self.simplify_with_open3d(mesh_trimesh, target_faces, verbose)
                elif method == 'pymeshlab' and self.pymeshlab:
                    result = self.simplify_with_pymeshlab(mesh_trimesh, target_faces, verbose)
                elif method == 'trimesh':
                    result = self.simplify_with_trimesh(mesh_trimesh, target_faces, verbose)
                else:
                    if verbose:
                        print(f"    - {method} not available, skipping...")
                    continue
                
                if result is not None and len(result.faces) > 0:
                    if verbose:
                        print(f"    ✅ Simplification successful using {method}")
                    return result
                else:
                    if verbose:
                        print(f"    ❌ {method} simplification failed, trying next method...")
            else:
                if verbose:
                    print(f"    - {method} not available, skipping...")
        
        if verbose:
            print(f"    ⚠️  All specified simplification methods failed, returning original mesh")
        return mesh_trimesh

class FluxTrellisBPTRetexturedOptimized:
    """
    Advanced Flux + TRELLIS + BPT pipeline with shape optimization and improved re-texturing
    
    Pipeline:
    1. Text → Flux → Image
    2. Image → Background Removal
    3. Image → TRELLIS → 3D Assets (Gaussians, Raw Mesh)
    4. Raw Mesh → BPT Enhancement → High-Detail Mesh (8k+ faces)
    5. High-Detail Mesh → Shape Optimization → Optimized Mesh
    6. Optimized Mesh → Advanced Texture Baking → Final Textured GLB
    
    BPT INTEGRATION:
    - Blocked and Patchified Tokenization for mesh enhancement
    - Generates 8k+ face meshes with fine details
    - Compressive tokenization reduces sequence length by ~75%
    - Memory-efficient processing with GPU management
    
    TEXTURE BAKING (3 methods with proper priority order):
    Method 1: TRELLIS UV Re-baking (FIRST PRIORITY - Most Accurate)
    Method 2: Hunyuan3D Paint Pipeline (SECOND PRIORITY - Good Fallback)
    Method 3: Simple Spherical Projection (LAST RESORT - Basic Fallback)
    """
    
    def __init__(self, trellis_model_path="microsoft/TRELLIS-image-large"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.dtype = torch.bfloat16
        self.trellis_model_path = trellis_model_path
        
        # Initialize components
        if REMBG_AVAILABLE:
            print("🎨 Initializing background remover...")
            self.bg_remover = BackgroundRemover()
        else:
            self.bg_remover = None
        
        # Initialize BPT mesh enhancer
        print("🔬 Initializing BPT mesh enhancer (lazy loading)...")
        self.bpt_enhancer = BPTMeshEnhancer(device=self.device)
        
        # Initialize shape optimizer
        print("🔧 Initializing shape optimizer...")
        self.shape_optimizer = ShapeOptimizer()
        
        # Initialize intelligent mesh simplifier
        print("🔬 Initializing intelligent mesh simplifier...")
        self.intelligent_simplifier = IntelligentMeshSimplifier()
        
        # Don't load TRELLIS pipeline here - load it only when needed
        self.trellis_pipeline = None
        
        # Quality presets (adjusted for BPT-enhanced meshes)
        self.quality_presets = {
            'draft': {
                'sparse_steps': 15, 'slat_steps': 15, 'cfg_strength': 6.0,
                'texture_size': 1024, 'max_faces': 30000, 'bpt_temperature': 0.3,
            },
            'standard': {
                'sparse_steps': 25, 'slat_steps': 25, 'cfg_strength': 7.5,
                'texture_size': 2048, 'max_faces': 60000, 'bpt_temperature': 0.5,
            },
            'high': {
                'sparse_steps': 40, 'slat_steps': 40, 'cfg_strength': 9.0,
                'texture_size': 4096, 'max_faces': 80000, 'bpt_temperature': 0.4,
            },
            'ultra': {
                'sparse_steps': 60, 'slat_steps': 60, 'cfg_strength': 10.5,
                'texture_size': 6144, 'max_faces': 100000, 'bpt_temperature': 0.3,
            }
        }
    
    def load_trellis_pipeline(self):
        if self.trellis_pipeline is None:
            print("📦 Loading TRELLIS Image-to-3D pipeline...")
            self.trellis_pipeline = TrellisImageTo3DPipeline.from_pretrained(self.trellis_model_path)
            self.trellis_pipeline.cuda()
        return self.trellis_pipeline
    
    def unload_trellis_pipeline(self):
        if self.trellis_pipeline is not None:
            print("🧹 Unloading TRELLIS pipeline...")
            del self.trellis_pipeline
            self.trellis_pipeline = None
            clear_gpu_memory()
    
    def load_flux_pipeline(self):
        print("🖼️  Loading Flux pipeline...")
        huggingface_token = os.getenv("HUGGINGFACE_TOKEN")
        file_url = "https://huggingface.co/gokaygokay/flux-game/blob/main/hyperflux_00001_.q8_0.gguf"
        file_url = file_url.replace("/resolve/main/", "/blob/main/").replace("?download=true", "")
        single_file_base_model = "camenduru/FLUX.1-dev-diffusers"
        
        try:
            quantization_config_tf = BitsAndBytesConfigTF(load_in_8bit=True, bnb_8bit_compute_dtype=torch.bfloat16)
            text_encoder_2 = T5EncoderModel.from_pretrained(single_file_base_model, subfolder="text_encoder_2", torch_dtype=self.dtype, quantization_config=quantization_config_tf, token=huggingface_token)
            transformer = FluxTransformer2DModel.from_single_file(file_url, subfolder="transformer", quantization_config=GGUFQuantizationConfig(compute_dtype=self.dtype), torch_dtype=self.dtype, config=single_file_base_model)
            flux_pipeline = FluxPipeline.from_pretrained(single_file_base_model, transformer=transformer, text_encoder_2=text_encoder_2, torch_dtype=self.dtype, token=huggingface_token)
            flux_pipeline.to(self.device)
            return flux_pipeline
        except Exception as e:
            print(f"❌ Error loading Flux pipeline: {e}")
            raise
    
    def generate_image_from_text(self, prompt: str, seed: int = 42, width: int = 1024, height: int = 1024) -> Image.Image:
        flux_pipeline = self.load_flux_pipeline()
        try:
            enhanced_prompt = f"wbgmsst, {prompt}, 3D isometric asset, clean white background, isolated object"
            print(f"🎯 Generating image from prompt: {enhanced_prompt}")
            generator = torch.Generator(device=self.device).manual_seed(seed)
            image = flux_pipeline(prompt=enhanced_prompt, guidance_scale=3.5, num_inference_steps=NUM_INFERENCE_STEPS, width=width, height=height, generator=generator).images[0]
            return image
        finally:
            del flux_pipeline
            clear_gpu_memory()

    def remove_background(self, image: Image.Image) -> Image.Image:
        if self.bg_remover is not None:
            print("🎨 Removing background...")
            return self.bg_remover(image)
        else:
            print("⚠️  Background removal not available, using original image")
            return image

    def generate_3d_from_image(self, image: Image.Image, quality: str = 'standard', seed: int = 42) -> Dict:
        if quality not in self.quality_presets:
            raise ValueError(f"Quality must be one of {list(self.quality_presets.keys())}")
        
        trellis_pipeline = self.load_trellis_pipeline()
        preset = self.quality_presets[quality]
        
        sparse_cfg, sparse_interval = self.create_adaptive_guidance_schedule(preset['sparse_steps'], preset['cfg_strength'])
        slat_cfg, slat_interval = self.create_adaptive_guidance_schedule(preset['slat_steps'], preset['cfg_strength'])
        
        print(f"🎯 Generating 3D assets with quality: {quality}")
        
        sparse_structure_params = {"steps": preset['sparse_steps'], "cfg_strength": sparse_cfg, "cfg_interval": sparse_interval, "rescale_t": 3.0}
        slat_params = {"steps": preset['slat_steps'], "cfg_strength": slat_cfg, "cfg_interval": slat_interval, "rescale_t": 3.0}
        
        outputs = trellis_pipeline.run(image, seed=seed, sparse_structure_sampler_params=sparse_structure_params, slat_sampler_params=slat_params)
        return outputs

    def create_adaptive_guidance_schedule(self, steps: int, base_cfg: float) -> Tuple[float, Tuple[float, float]]:
        if steps >= 50:
            return base_cfg * 1.1, (0.3, 0.98)
        elif steps >= 30:
            return base_cfg * 1.05, (0.4, 0.95)
        else:
            return base_cfg, (0.5, 0.9)

    def correct_texture_orientation(self, image: Image.Image, rotation_angle: int = 0) -> Image.Image:
        """Correct texture orientation by rotating the input image"""
        if rotation_angle == 0:
            return image
        elif rotation_angle == 90:
            return image.transpose(Image.ROTATE_270)
        elif rotation_angle == 180:
            return image.transpose(Image.ROTATE_180)
        elif rotation_angle == 270:
            return image.transpose(Image.ROTATE_90)
        else:
            print(f"⚠️  Warning: Unsupported rotation angle {rotation_angle}. Using no rotation.")
            return image

    def apply_hunyuan_texture(self, mesh_trimesh, input_image: Image.Image, verbose: bool = True):
        """Apply texture using Hunyuan3D's painting pipeline"""
        if not HUNYUAN_TEXGEN_AVAILABLE:
            if verbose:
                print("⚠️  Hunyuan3D texture painting not available")
            return None
        
        try:
            if verbose:
                print("🎨 Applying texture using Hunyuan3D Paint Pipeline...")
            
            # Apply texture orientation correction first  
            corrected_image = self.correct_texture_orientation(input_image, rotation_angle=90)
            
            # Initialize paint pipeline
            paint_pipeline = Hunyuan3DPaintPipeline.from_pretrained("jetx/Hunyuan3D-2")
            
            # Apply texture to the mesh using corrected image
            textured_mesh = paint_pipeline(mesh_trimesh, image=corrected_image)
            
            if verbose:
                print("✅ Hunyuan3D texture application successful!")
            
            # Clean up
            del paint_pipeline
            clear_gpu_memory()
            
            return textured_mesh
            
        except Exception as e:
            if verbose:
                print(f"❌ Hunyuan3D texture application failed: {e}")
            return None

    def apply_simple_texture_projection(self, mesh_trimesh, input_image: Image.Image, verbose: bool = True):
        """Simple texture projection using spherical mapping with coordinate fixes"""
        try:
            if verbose:
                print("🎨 Applying simple texture projection...")
            
            # Apply texture orientation correction first
            corrected_image = self.correct_texture_orientation(input_image, rotation_angle=90)
            
            # Get mesh vertices and normalize to unit sphere
            vertices = mesh_trimesh.vertices.copy()
            center = vertices.mean(axis=0)
            vertices_centered = vertices - center
            max_dist = np.linalg.norm(vertices_centered, axis=1).max()
            vertices_normalized = vertices_centered / max_dist
            
            # Corrected spherical coordinates for TRELLIS coordinate system
            x, y, z = vertices_normalized[:, 0], vertices_normalized[:, 1], vertices_normalized[:, 2]
            theta = np.arctan2(x, z)  # Use x,z for correct front-facing
            phi = np.arccos(np.clip(y, -1, 1))  # Use y for elevation
            
            # UV mapping with rotation correction
            u = (theta + np.pi) / (2 * np.pi)
            v = 1.0 - (phi / np.pi)  # Flip V for proper orientation
            
            # Apply UV rotation correction (270 degrees)
            u_corrected = (u + 0.75) % 1.0
            v_corrected = v
            
            uv_coords = np.column_stack([u_corrected, v_corrected])
            
            # Create material with corrected image
            material = trimesh.visual.material.PBRMaterial(
                roughnessFactor=0.8,
                baseColorTexture=corrected_image,
                baseColorFactor=np.array([255, 255, 255, 255], dtype=np.uint8)
            )
            
            textured_mesh = trimesh.Trimesh(
                vertices=vertices,
                faces=mesh_trimesh.faces,
                visual=trimesh.visual.TextureVisuals(uv=uv_coords, material=material)
            )
            
            if verbose:
                print("✅ Simple texture projection successful!")
            
            return textured_mesh
            
        except Exception as e:
            if verbose:
                print(f"❌ Simple texture projection failed: {e}")
            return None

    def create_bpt_enhanced_and_retextured_glb(self, outputs: Dict, output_prefix: str, quality: str, input_image: Image.Image = None, bpt_choice: Dict = None, shape_optimization_enabled: bool = True, intelligent_simplification_choice: Dict = None, verbose: bool = True):
        """Create BPT-enhanced and re-textured GLB with advanced texture baking"""
        print("✨ Creating enhanced and re-textured GLB...")
        preset = self.quality_presets[quality]

        try:
            # 1. Get raw mesh from TRELLIS output
            raw_vertices = outputs['mesh'][0].vertices.cpu().numpy()
            raw_faces = outputs['mesh'][0].faces.cpu().numpy()
            raw_mesh_trimesh = trimesh.Trimesh(vertices=raw_vertices, faces=raw_faces)
            
            if verbose:
                print(f"📊 Raw mesh: {len(raw_vertices)} vertices, {len(raw_faces)} faces")

            # 2. Apply BPT mesh enhancement (OPTIONAL based on user choice)
            enhanced_mesh_trimesh = raw_mesh_trimesh
            if bpt_choice and bpt_choice.get('use_bpt', False):
                print("🔬 Applying BPT mesh enhancement...")
                enhanced_mesh_trimesh = self.bpt_enhancer.enhance_mesh_with_bpt(
                    raw_mesh_trimesh, 
                    temperature=bpt_choice.get('bpt_temperature', preset['bpt_temperature']), 
                    verbose=verbose
                )
                
                if verbose:
                    print(f"📊 BPT-enhanced mesh: {len(enhanced_mesh_trimesh.vertices)} vertices, {len(enhanced_mesh_trimesh.faces)} faces")
            else:
                if verbose:
                    print("⏭️  Skipping BPT enhancement (user choice)")

            # 3. Apply shape optimization (OPTIONAL based on user choice)
            optimized_mesh_trimesh = enhanced_mesh_trimesh
            if shape_optimization_enabled and SHAPEGEN_AVAILABLE:
                print("🔧 Applying shape optimization...")
                max_faces = bpt_choice.get('max_faces', preset['max_faces']) if bpt_choice else preset['max_faces']
                optimized_mesh_trimesh = self.shape_optimizer.optimize_mesh(
                    enhanced_mesh_trimesh, 
                    max_faces=max_faces, 
                    verbose=verbose
                )
            elif not shape_optimization_enabled:
                if verbose:
                    print("⏭️  Skipping shape optimization (user choice)")
            else:
                if verbose:
                    print("⚠️  Shape optimization not available")
            
            # 4. Apply intelligent mesh simplification (OPTIONAL based on user choice)
            simplified_mesh_trimesh = optimized_mesh_trimesh
            if intelligent_simplification_choice and intelligent_simplification_choice.get('use_simplification', False):
                print("🧠 Applying intelligent mesh simplification...")
                target_faces = intelligent_simplification_choice.get('target_faces', 2000)
                preferred_methods = intelligent_simplification_choice.get('preferred_methods', [])
                simplified_mesh_trimesh = self.intelligent_simplifier.intelligent_simplify(
                    optimized_mesh_trimesh, 
                    target_faces=target_faces,
                    preferred_methods=preferred_methods,
                    verbose=verbose
                )
                if verbose:
                    print(f"📊 Simplified mesh: {len(simplified_mesh_trimesh.vertices)} vertices, {len(simplified_mesh_trimesh.faces)} faces")
            else:
                if verbose:
                    print("⏭️  Skipping intelligent simplification (user choice)")
            
            # Ensure vertices are float32
            simplified_mesh_trimesh.vertices = simplified_mesh_trimesh.vertices.astype(np.float32)
            
            if verbose:
                print(f"📊 Final processed mesh: {len(simplified_mesh_trimesh.vertices)} vertices, {len(simplified_mesh_trimesh.faces)} faces")

            # 5. ADVANCED TEXTURE BAKING WITH PROPER PRIORITY ORDER
            textured_mesh = None
            
            # METHOD 1: TRELLIS UV Re-baking (FIRST PRIORITY - Most Accurate)
            print("  - 🎨 Method 1: TRELLIS UV re-baking (preserves original texture quality)...")
            try:
                # Re-parameterize the final mesh
                print("    - Re-parameterizing (UV unwrapping) the final mesh...")
                opt_verts_np, opt_faces_np, opt_uvs_np = postprocessing_utils.parametrize_mesh(
                    simplified_mesh_trimesh.vertices, 
                    simplified_mesh_trimesh.faces
                )

                # Ensure all arrays are float32 to avoid dtype mismatch
                opt_verts_np = opt_verts_np.astype(np.float32)
                opt_faces_np = opt_faces_np.astype(np.int32)
                opt_uvs_np = opt_uvs_np.astype(np.float32)

                if verbose:
                    print(f"    - Array dtypes: vertices={opt_verts_np.dtype}, faces={opt_faces_np.dtype}, uvs={opt_uvs_np.dtype}")

                # Render multiview images from Gaussians
                print("    - Rendering multiview images from Gaussians...")
                observations, extrinsics, intrinsics = postprocessing_utils.render_multiview(
                    outputs['gaussian'][0], 
                    resolution=1024,
                    nviews=120
                )
                
                # Create masks and ensure consistent dtypes
                masks = [np.any(observation > 0, axis=-1) for observation in observations]
                extrinsics = [extrinsics[i].cpu().numpy().astype(np.float32) for i in range(len(extrinsics))]
                intrinsics = [intrinsics[i].cpu().numpy().astype(np.float32) for i in range(len(intrinsics))]
                observations = [obs.astype(np.float32) for obs in observations]

                if verbose:
                    print(f"    - Multiview data: {len(observations)} views, extrinsics dtype={extrinsics[0].dtype}")

                # Bake texture onto the new UV map
                print("    - Baking textures onto the new UV map...")
                texture_image = postprocessing_utils.bake_texture(
                    opt_verts_np, opt_faces_np, opt_uvs_np,
                    observations, masks, extrinsics, intrinsics,
                    texture_size=preset['texture_size'], 
                    mode='opt',
                    lambda_tv=0.005,
                    verbose=verbose
                )
                texture = Image.fromarray(texture_image)

                # Assemble final GLB with coordinate correction
                print("    - Assembling final GLB with corrected coordinates...")
                final_vertices = opt_verts_np @ np.array([[0, 0, 1], [0, 1, 0], [-1, 0, 0]], dtype=np.float32)
                material = trimesh.visual.material.PBRMaterial(
                    roughnessFactor=1.0, 
                    baseColorTexture=texture, 
                    baseColorFactor=np.array([255, 255, 255, 255], dtype=np.uint8)
                )
                textured_mesh = trimesh.Trimesh(
                    vertices=final_vertices, 
                    faces=opt_faces_np, 
                    visual=trimesh.visual.TextureVisuals(uv=opt_uvs_np, material=material)
                )
                
                print("    ✅ TRELLIS UV re-baking successful!")
                
            except Exception as e:
                print(f"    ❌ TRELLIS UV re-baking failed: {e}")
                textured_mesh = None
            
            # METHOD 2: Hunyuan3D Paint Pipeline (FALLBACK)
            if textured_mesh is None and input_image is not None and HUNYUAN_TEXGEN_AVAILABLE:
                print("  - 🎨 Method 2: Hunyuan3D texture painting (good fallback)...")
                textured_mesh = self.apply_hunyuan_texture(simplified_mesh_trimesh, input_image, verbose=verbose)
            
            # METHOD 3: Simple Spherical Projection (LAST RESORT)
            if textured_mesh is None and input_image is not None:
                print("  - 🎨 Method 3: Simple spherical projection (last resort)...")
                textured_mesh = self.apply_simple_texture_projection(simplified_mesh_trimesh, input_image, verbose=verbose)

            # Export final textured mesh
            if textured_mesh is not None:
                # Create descriptive filename based on settings
                enhancement_parts = []
                if bpt_choice and bpt_choice.get('use_bpt', False):
                    enhancement_parts.append("bpt_enhanced")
                        
                if shape_optimization_enabled:
                    enhancement_parts.append("shape_optimized")
                
                if intelligent_simplification_choice and intelligent_simplification_choice.get('use_simplification', False):
                    enhancement_parts.append("intelligent_simplified")
                    
                enhancement_suffix = "_" + "_".join(enhancement_parts) if enhancement_parts else ""
                glb_path = f"{output_prefix}{enhancement_suffix}_retextured.glb"
                textured_mesh.export(glb_path)
                print(f"✅ Successfully created enhanced and re-textured GLB: {glb_path}")
                return glb_path
            else:
                print("❌ All texture methods failed")
                return None
                
        except Exception as e:
            print(f"❌ Enhancement and re-texturing failed: {e}")
            return None

    def run_pipeline(self, prompt: str, quality: str = 'standard', output_prefix: str = "flux_trellis_bpt", seed: int = 42, bpt_choice: Dict = None, shape_optimization_enabled: bool = True, intelligent_simplification_choice: Dict = None) -> Dict:
        print("=" * 80)
        print("🚀 FLUX + TRELLIS + BPT + SHAPE OPTIMIZATION + RE-TEXTURING")
        print("=" * 80)
        
        processed_files = {}

        try:
            print("🧹 Clearing GPU memory before processing...")
            clear_gpu_memory()
            
            # Step 1: Generate image
            print("\n--- Step 1: Generating Image ---")
            image = self.generate_image_from_text(prompt, seed)
            original_image_path = f"{output_prefix}_original.png"
            image.save(original_image_path)
            processed_files['original_image'] = original_image_path
            print(f"💾 Saved original image: {original_image_path}")
            
            print("\n--- Step 2: Removing Background ---")
            processed_image = self.remove_background(image)
            processed_image_path = f"{output_prefix}_processed.png"
            processed_image.save(processed_image_path)
            processed_files['processed_image'] = processed_image_path
            print(f"💾 Saved processed image: {processed_image_path}")

            print("\n--- Step 3: Generating 3D Assets ---")
            clear_gpu_memory()
            outputs = self.generate_3d_from_image(processed_image, quality, seed)
            
            print("\n--- Step 4: Post-Processing and Exporting ---")
            
            # Create standard textured GLB (for comparison)
            preset = self.quality_presets[quality]
            print(f"📦 Creating standard textured GLB ({preset['texture_size']}^2)...")
            glb_textured = postprocessing_utils.to_glb(outputs['gaussian'][0], outputs['mesh'][0], simplify=0.98, texture_size=preset['texture_size'], fill_holes=True, verbose=True)
            textured_path = f"{output_prefix}_textured.glb"
            glb_textured.export(textured_path)
            processed_files['glb_textured'] = textured_path
            
            # Create BPT-enhanced and re-textured GLB (using pre-selected choice)
            bpt_optimized_path = self.create_bpt_enhanced_and_retextured_glb(outputs, output_prefix, quality, processed_image, bpt_choice, shape_optimization_enabled, intelligent_simplification_choice)
            if bpt_optimized_path:
                processed_files['glb_bpt_retextured_optimized'] = bpt_optimized_path
            
            # Render videos and save other assets
            print("🎬 Rendering videos...")
            video = render_utils.render_video(outputs['gaussian'][0], render_size=(1024, 1024), num_frames=120, ss_level=2)['color']
            video_path = f"{output_prefix}_gs.mp4"
            imageio.mimsave(video_path, video, fps=30)
            processed_files['gaussian_video'] = video_path

            simplified_gs = postprocessing_utils.simplify_gs(outputs['gaussian'][0], simplify=0.98, verbose=True)
            ply_path = f"{output_prefix}_optimized.ply"
            simplified_gs.save_ply(ply_path)
            processed_files['ply'] = ply_path
            
            print("\n" + "=" * 80)
            print("✅ BPT-ENHANCED GENERATION COMPLETE!")
            for file_type, path in processed_files.items():
                print(f"  - {file_type}: {path}")
            print("=" * 80)

            return {'files': processed_files, 'settings': {'quality': quality, 'prompt': prompt, 'seed': seed}}
        except Exception as e:
            print(f"❌ Error in pipeline: {e}")
            raise
        finally:
            self.unload_trellis_pipeline()
            clear_gpu_memory()

def get_user_shape_optimization_choice():
    """Ask user about shape optimization preference"""
    print("\n🔧 SHAPE OPTIMIZATION SELECTION")
    print("=" * 50)
    print("Apply shape optimizations (floater removal, degenerate face cleanup, etc.)?")
    print("Recommended: YES for cleaner final meshes")
    print("=" * 50)
    
    choice = input("Use shape optimizations? (y/n) [y]: ").strip().lower() or 'y'
    use_optimization = choice in ['y', 'yes', '1', 'true']
    
    status = "✅ ENABLED" if use_optimization else "❌ DISABLED"
    print(f"Shape Optimizations: {status}")
    
    return use_optimization

def get_user_simplification_method_choice():
    """Ask user about preferred mesh simplification methods"""
    print("\n🔬 MESH SIMPLIFICATION METHOD SELECTION")
    print("=" * 60)
    print("Choose which mesh simplification algorithm(s) to use:")
    print("1. auto      - Try all available methods in quality order (recommended)")
    print("2. open3d    - Use Open3D QEM only (best quality, requires open3d)")
    print("3. pymeshlab - Use PyMeshLab QEC only (high quality, requires pymeshlab)")
    print("4. trimesh   - Use Trimesh methods only (basic quality, always available)")
    print("5. custom    - Choose specific methods and order")
    print("=" * 60)
    print("💡 Auto mode tries: Open3D → PyMeshLab → Trimesh (best to worst quality)")
    
    # Check what's actually available
    available_methods = []
    try:
        import open3d as o3d
        available_methods.append('open3d')
    except ImportError:
        pass
    
    try:
        import pymeshlab
        available_methods.append('pymeshlab')
    except ImportError:
        pass
    
    available_methods.append('trimesh')  # Always available
    
    print(f"📋 Available methods: {', '.join(available_methods)}")
    
    method_map = {
        '1': {'methods': ['open3d', 'pymeshlab', 'trimesh'], 'name': 'AUTO'},
        '2': {'methods': ['open3d'], 'name': 'OPEN3D_ONLY'},
        '3': {'methods': ['pymeshlab'], 'name': 'PYMESHLAB_ONLY'},
        '4': {'methods': ['trimesh'], 'name': 'TRIMESH_ONLY'},
        '5': {'methods': [], 'name': 'CUSTOM'},  # Will be filled by user
    }
    
    choice = input("Select method preference (1-5) [1 for auto]: ").strip() or '1'
    selected = method_map.get(choice, method_map['1'])
    
    # Handle custom method selection
    if choice == '5':
        print("\n🛠️  Custom Method Selection:")
        print("Available methods:")
        for i, method in enumerate(available_methods, 1):
            quality_desc = {
                'open3d': '(best quality, QEM algorithm)',
                'pymeshlab': '(high quality, QEC algorithm)', 
                'trimesh': '(basic quality, various algorithms)'
            }.get(method, '')
            print(f"  {i}. {method} {quality_desc}")
        
        try:
            selections = input("Enter method numbers separated by spaces (e.g., '1 3' for open3d then trimesh): ").strip().split()
            custom_methods = []
            for sel in selections:
                idx = int(sel) - 1
                if 0 <= idx < len(available_methods):
                    method = available_methods[idx]
                    if method not in custom_methods:  # Avoid duplicates
                        custom_methods.append(method)
            
            if custom_methods:
                selected['methods'] = custom_methods
                selected['name'] = f"CUSTOM_{'_'.join(custom_methods).upper()}"
            else:
                print("⚠️  Invalid selection, using auto mode")
                selected = method_map['1']
        except (ValueError, IndexError):
            print("⚠️  Invalid input, using auto mode")
            selected = method_map['1']
    
    # Filter methods to only include available ones
    available_selected = [m for m in selected['methods'] if m in available_methods]
    if not available_selected:
        print(f"⚠️  Selected methods not available, falling back to available: {available_methods}")
        available_selected = available_methods
    
    selected['methods'] = available_selected
    
    print(f"✅ Selected: {selected['name']}")
    print(f"   - Methods to use: {' → '.join(selected['methods'])}")
    
    return selected

def get_user_intelligent_simplification_choice():
    """Ask user about intelligent mesh simplification preference"""
    print("\n🧠 INTELLIGENT MESH SIMPLIFICATION SELECTION")
    print("=" * 60)
    print("Apply intelligent mesh simplification to reduce face count while preserving details?")
    print("Uses advanced algorithms (Open3D QEM, PyMeshLab QEC, etc.) for high-quality reduction.")
    print("1. skip   - No simplification")
    print("2. light  - Reduce to ~1-2K faces (good for real-time)")
    print("3. medium - Reduce to ~3-5K faces (balanced)")
    print("4. custom - Specify target face count")
    print("=" * 60)
    print("💡 Can be applied to any mesh: raw TRELLIS, BPT-enhanced, or after shape optimization")
    
    simplify_map = {
        '1': {'use_simplification': False, 'target_faces': 0, 'name': 'SKIP'},
        '2': {'use_simplification': True, 'target_faces': 1500, 'name': 'LIGHT'},
        '3': {'use_simplification': True, 'target_faces': 4000, 'name': 'MEDIUM'},
        '4': {'use_simplification': True, 'target_faces': -1, 'name': 'CUSTOM'},  # -1 means ask user
    }
    
    choice = input("Select simplification level (1-4) [1 for skip]: ").strip() or '1'
    selected = simplify_map.get(choice, simplify_map['1'])
    
    # Handle custom target face count
    if selected['target_faces'] == -1:
        try:
            custom_faces = int(input("Enter target face count (e.g., 2000): ").strip())
            selected['target_faces'] = max(100, custom_faces)  # Minimum 100 faces
        except ValueError:
            print("⚠️  Invalid input, using default 2000 faces")
            selected['target_faces'] = 2000
    
    print(f"✅ Selected: {selected['name']}")
    if selected['use_simplification']:
        print(f"   - Target Face Count: {selected['target_faces']:,}")
        print(f"   - Will apply intelligent simplification at the end of processing")
    else:
        print(f"   - No mesh simplification will be applied")
    
    return selected

def get_user_bpt_choice():
    """Ask user about BPT mesh enhancement preference"""
    print("\n🔬 BPT MESH ENHANCEMENT SELECTION")
    print("=" * 60)
    print("BPT can generate ultra-high detail meshes (8k+ faces) but takes longer.")
    print("1. skip         - No BPT, use TRELLIS mesh only (~1-2k faces, fastest)")
    print("2. conservative - BPT with lower settings (~4k faces, moderate time)")
    print("3. standard     - BPT with standard settings (~8k faces, longer time)")
    print("4. aggressive   - BPT with high settings (10k+ faces, longest time)")
    print("=" * 60)
    print("⚠️  Higher face counts = better detail but much longer processing time!")
    print("💡 Use intelligent simplification (next option) to reduce face count afterward")
    
    bpt_map = {
        '1': {'use_bpt': False, 'bpt_temperature': 0.5, 'max_faces': 2000, 'name': 'SKIP'},
        '2': {'use_bpt': True, 'bpt_temperature': 0.2, 'max_faces': 20000, 'name': 'CONSERVATIVE'},
        '3': {'use_bpt': True, 'bpt_temperature': 0.5, 'max_faces': 40000, 'name': 'STANDARD'},
        '4': {'use_bpt': True, 'bpt_temperature': 0.8, 'max_faces': 60000, 'name': 'AGGRESSIVE'},
    }
    
    choice = input("Select BPT option (1-4) [3 for standard]: ").strip() or '3'
    selected = bpt_map.get(choice, bpt_map['3'])
    
    print(f"✅ Selected: {selected['name']}")
    if selected['use_bpt']:
        print(f"   - Temperature: {selected['bpt_temperature']} | Max Faces: {selected['max_faces']:,}")
    else:
        print(f"   - Using TRELLIS mesh only for fastest processing")
    
    return selected

def get_user_quality_choice():
    print("\n🎯 QUALITY SELECTION")
    print("=" * 50)
    print("1. draft   - Fast, low quality (1K textures, 30K faces)")
    print("2. standard - Recommended balance (2K textures, 60K faces)")
    print("3. high    - High quality (4K textures, 80K faces)")
    print("4. ultra   - Maximum quality (6K textures, 100K faces)")
    print("=" * 50)
    quality_map = {'1': 'draft', '2': 'standard', '3': 'high', '4': 'ultra'}
    choice = input("Select quality level (1-4) [2 for standard]: ").strip() or '2'
    selected = quality_map.get(choice, 'standard')
    print(f"✅ Selected: {selected.upper()}")
    return selected

if __name__ == "__main__":
    # Set environment variables for CUDA compilation
    os.environ['TORCH_CUDA_ARCH_LIST'] = "8.9"
    os.environ['NVCC_APPEND_FLAGS'] = "-allow-unsupported-compiler"
    
    pipeline = FluxTrellisBPTRetexturedOptimized()
    quality_level = get_user_quality_choice()
    
    # Ask for BPT choice ONCE before processing all prompts
    bpt_choice = get_user_bpt_choice()
    
    # Ask for shape optimization choice ONCE before processing all prompts
    shape_optimization_enabled = get_user_shape_optimization_choice()
    
    # Ask for intelligent simplification choice ONCE before processing all prompts
    intelligent_simplification_choice = get_user_intelligent_simplification_choice()
    
    # If intelligent simplification is enabled, ask for method preference
    simplification_method_choice = None
    if intelligent_simplification_choice['use_simplification']:
        simplification_method_choice = get_user_simplification_method_choice()
        # Merge method choice into simplification choice
        intelligent_simplification_choice.update(simplification_method_choice)
    
    test_prompts = [
        "charming red barn with weathered wood without any windows or base",
        "a blue monkey sitting on temple", 
        "football", 
        "purple durable robotic arm", 
        "flying dragon"
    ]
    
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"bpt_retextured_outputs_{timestamp}_{quality_level}"
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"\n🚀 Processing {len(test_prompts)} prompts with BPT enhancement...")
    print(f"📁 Output directory: {output_dir}")
    print(f"🎯 Quality Level: {quality_level.upper()}")
    print(f"🔬 BPT Mode: {bpt_choice['name']}")
    if bpt_choice.get('use_bpt', False):
        print(f"🔬 BPT Settings: Temperature={bpt_choice['bpt_temperature']}, Max Faces={bpt_choice['max_faces']:,}")
    print(f"🔧 Shape Optimizations: {'✅ ENABLED' if shape_optimization_enabled else '❌ DISABLED'}")
    print(f"🔬 BPT Available: {BPT_AVAILABLE}")
    print(f"🔧 Shape Optimization Available: {SHAPEGEN_AVAILABLE}")
    print(f"🎨 Texture Painting Available: {HUNYUAN_TEXGEN_AVAILABLE}")
    print(f"🧠 Intelligent Simplification: {'✅ ENABLED' if intelligent_simplification_choice['use_simplification'] else '❌ DISABLED'}")
    if intelligent_simplification_choice['use_simplification']:
        methods_str = ' → '.join(intelligent_simplification_choice.get('methods', ['auto']))
        print(f"🔬 Simplification Methods: {methods_str}")
        print(f"🎯 Target Face Count: {intelligent_simplification_choice['target_faces']:,}")
    
    for idx, prompt in enumerate(test_prompts):
        safe_prefix = f"prompt_{idx+1}_{re.sub(r'[^a-zA-Z0-9]+', '_', prompt)[:32]}"
        output_prefix = os.path.join(output_dir, safe_prefix)
        
        print(f"\n{'='*100}\n🎯 Processing prompt {idx+1}/{len(test_prompts)}: '{prompt}'\n{'='*100}")
        
        try:
            # Use the SAME bpt_choice for ALL prompts
            pipeline.run_pipeline(prompt=prompt, quality=quality_level, output_prefix=output_prefix, seed=42 + idx * 100, bpt_choice=bpt_choice, shape_optimization_enabled=shape_optimization_enabled, intelligent_simplification_choice=intelligent_simplification_choice)
            print(f"\n✅ Prompt {idx+1} complete. Files saved to: {output_dir}")
        except Exception as e:
            print(f"❌ Error processing prompt {idx+1}: {e}")
            continue
    
    print(f"\n🎉 All prompts processed! Results saved in: {output_dir}")
    print(f"\n📋 PIPELINE SUMMARY:")
    print(f"- 🖼️  Text → Flux → Image")
    print(f"- 🎨 Background Removal")
    print(f"- 📦 Image → TRELLIS → Raw 3D Assets")
    if bpt_choice.get('use_bpt', False):
        print(f"- 🔬 Raw Mesh → BPT Enhancement → High-Detail Mesh")
    else:
        print(f"- ⏭️  Raw Mesh → (BPT Skipped)")
    
    if shape_optimization_enabled:
        print(f"- 🔧 Mesh → Shape Optimization → Cleaned Mesh")
    else:
        print(f"- ⏭️  Mesh → (Shape Optimization Skipped)")
    
    if intelligent_simplification_choice['use_simplification']:
        target_faces = intelligent_simplification_choice['target_faces']
        print(f"- 🧠 Mesh → Intelligent Simplification → Reduced Mesh (~{target_faces:,} faces)")
    else:
        print(f"- ⏭️  Mesh → (Intelligent Simplification Skipped)")
    
    print(f"- 🎯 Final Mesh → Advanced Texture Baking → Final GLB")
    preset = pipeline.quality_presets[quality_level]
    print(f"\n🎨 SETTINGS ({quality_level.upper()}):")
    print(f"- Texture Resolution: {preset['texture_size']}^2")
    
    if intelligent_simplification_choice['use_simplification']:
        print(f"- Target Face Count: {intelligent_simplification_choice['target_faces']:,} (Intelligent Simplification)")
    else:
        print(f"- Max Faces: {bpt_choice.get('max_faces', preset['max_faces']):,}")
    
    if bpt_choice.get('use_bpt', False):
        print(f"- BPT Temperature: {bpt_choice['bpt_temperature']}")
    else:
        print(f"- BPT: DISABLED") 