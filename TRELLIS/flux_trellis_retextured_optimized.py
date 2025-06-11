'''
export TORCH_CUDA_ARCH_LIST="8.9" && export NVCC_APPEND_FLAGS="-allow-unsupported-compiler"
'''
import os
import datetime
import torch
import numpy as np
import gc
import re
from PIL import Image
from pathlib import Path
from typing import List, Dict, Tuple
import trimesh

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

# Constants
NUM_INFERENCE_STEPS = 8
MAX_SEED = np.iinfo(np.int32).max

def clear_gpu_memory():
    """Clear GPU memory cache"""
    gc.collect()
    torch.cuda.empty_cache()
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

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
    
    def preserve_uv_during_optimization(self, mesh, max_faces: int = 40000, verbose: bool = True):
        """
        UV-preserving shape optimization - tries to maintain texture coordinates
        
        This method applies shape optimization while attempting to preserve UV mapping
        by using more conservative decimation and avoiding aggressive topology changes.
        """
        if not SHAPEGEN_AVAILABLE:
            if verbose:
                print("⚠️  Shape optimization not available, returning original mesh")
            return mesh
        
        if verbose:
            print("🔧 Applying UV-preserving shape optimization...")
        
        # Step 1: Remove only clearly degenerate faces (minimal impact on UVs)
        if verbose:
            print("  - Removing degenerate faces (conservative)...")
        mesh = self.degenerate_face_remover(mesh)
        
        # Step 2: Remove floating components (doesn't affect main mesh UVs)
        if verbose:
            print("  - Removing floaters...")
        mesh = self.floater_remover(mesh)
        
        # Step 3: Conservative face reduction (preserve UV topology as much as possible)
        if verbose:
            print(f"  - Conservative face reduction to max {max_faces}...")
        
        current_faces = len(mesh.faces)
        if current_faces > max_faces:
            # Use a higher target to preserve more UV mapping
            conservative_target = int(max_faces * 1.2)  # 20% more faces than target
            mesh = self.face_reducer(mesh, max_facenum=min(conservative_target, current_faces))
        
        # Step 4: Skip aggressive mesh simplification to preserve UVs
        if verbose:
            print("  - Skipping aggressive simplification to preserve UV mapping...")
        
        if verbose:
            print("✅ UV-preserving shape optimization complete!")
        
        return mesh
    
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
        
        # Step 3: Reduce face count for optimization
        if verbose:
            print(f"  - Reducing faces to max {max_faces}...")
        mesh = self.face_reducer(mesh, max_facenum=max_faces)
        
        # Step 4: Apply mesh simplification, if available
        if verbose:
            print("  - Applying mesh simplification...")
        
        if self.mesh_simplifier:
            simplifier_path = self.mesh_simplifier.executable
            # Check if the binary exists and is executable
            if not (os.path.isfile(simplifier_path) and os.access(simplifier_path, os.X_OK)):
                if verbose:
                    print(f"    ⚠️  Mesh simplifier binary not found or not executable at '{simplifier_path}'. Skipping this step.")
            else:
                try:
                    mesh = self.mesh_simplifier(mesh)
                except Exception as e:
                    if verbose:
                        print(f"    ⚠️  Mesh simplification failed during execution: {e}")
        else:
            if verbose:
                print("    ⚠️  Mesh simplifier not initialized. Skipping this step.")

        if verbose:
            print("✅ Shape optimization complete!")
        
        return mesh

class FluxTrellisRetexturedOptimized:
    """
    Advanced Flux + TRELLIS pipeline with shape optimization and improved re-texturing
    
    Pipeline:
    1. Text → Flux → Image
    2. Image → Background Removal
    3. Image → TRELLIS → 3D Assets (Gaussians, Raw Mesh)
    4. Raw Mesh → Shape Optimization → Optimized Mesh
    5. Optimized Mesh → IMPROVED TEXTURE BAKING → Final Textured & Optimized GLB
    
    IMPROVED TEXTURE BAKING (3 methods with proper priority order):
    
    Method 1: TRELLIS UV Re-baking (FIRST PRIORITY - Most Accurate)
        - Re-parameterizes optimized mesh with new UV unwrapping
        - Renders 120 multiview images from original Gaussians at 1024px
        - Projects multiview textures onto new UV map using advanced baking
        - Preserves original texture quality and detail from Gaussians
        - Handles 90-degree rotation correction with proper coordinate transforms
        - Most accurate but requires successful UV unwrapping
    
    Method 2: Hunyuan3D Paint Pipeline (SECOND PRIORITY - Good Fallback)
        - Uses Hunyuan3D's proven texture painting system
        - Directly applies corrected input image to mesh surface
        - Good quality and reliability for most object types
        - Works when UV re-baking fails
    
    Method 3: Simple Spherical Projection (LAST RESORT - Basic Fallback)
        - Projects 2D image onto 3D mesh using corrected spherical UV mapping
        - Fast and reliable for basic texturing
        - Works with any mesh topology
        - Used when all other methods fail
        
    UV PRESERVATION TECHNIQUES:
        - Conservative shape optimization option to preserve UV topology
        - Higher resolution multiview rendering (1024px, 120 views)
        - Lower TV regularization for sharper texture baking
        - Coordinate system corrections for proper orientation
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
        
        # Initialize shape optimizer
        print("🔧 Initializing shape optimizer...")
        self.shape_optimizer = ShapeOptimizer()
        
        # Don't load TRELLIS pipeline here - load it only when needed
        self.trellis_pipeline = None
        
        # Quality presets
        self.quality_presets = {
            'draft': {
                'sparse_steps': 15, 'slat_steps': 15, 'cfg_strength': 6.0,
                'texture_size': 1024, 'max_faces': 20000,
            },
            'standard': {
                'sparse_steps': 25, 'slat_steps': 25, 'cfg_strength': 7.5,
                'texture_size': 2048, 'max_faces': 40000,
            },
            'high': {
                'sparse_steps': 40, 'slat_steps': 40, 'cfg_strength': 9.0,
                'texture_size': 4096, 'max_faces': 60000,
            },
            'ultra': {
                'sparse_steps': 60, 'slat_steps': 60, 'cfg_strength': 10.5,
                'texture_size': 6144, 'max_faces': 80000,
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
        """
        Correct texture orientation by rotating the input image
        
        Args:
            image: Input texture image
            rotation_angle: Rotation angle in degrees (0, 90, 180, 270)
                          0: No rotation
                          90: 90 degrees clockwise 
                          180: 180 degrees
                          270: 270 degrees clockwise (90 degrees anticlockwise)
        """
        if rotation_angle == 0:
            return image
        elif rotation_angle == 90:
            return image.transpose(Image.ROTATE_270)  # PIL uses opposite convention
        elif rotation_angle == 180:
            return image.transpose(Image.ROTATE_180)
        elif rotation_angle == 270:
            return image.transpose(Image.ROTATE_90)   # PIL uses opposite convention
        else:
            print(f"⚠️  Warning: Unsupported rotation angle {rotation_angle}. Using no rotation.")
            return image

    def apply_hunyuan_texture(self, mesh_trimesh, input_image: Image.Image, verbose: bool = True):
        """Apply texture using Hunyuan3D's painting pipeline - more reliable than UV re-baking"""
        if not HUNYUAN_TEXGEN_AVAILABLE:
            if verbose:
                print("⚠️  Hunyuan3D texture painting not available")
            return None
        
        try:
            if verbose:
                print("🎨 Applying texture using Hunyuan3D Paint Pipeline...")
            
            # FIXED: Apply texture orientation correction first  
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
        """Simple texture projection - project 2D image onto 3D mesh using spherical mapping"""
        try:
            if verbose:
                print("🎨 Applying simple texture projection...")
            
            # FIXED: Apply texture orientation correction first
            corrected_image = self.correct_texture_orientation(input_image, rotation_angle=90)
            
            # Get mesh vertices and normalize to unit sphere
            vertices = mesh_trimesh.vertices.copy()
            center = vertices.mean(axis=0)
            vertices_centered = vertices - center
            max_dist = np.linalg.norm(vertices_centered, axis=1).max()
            vertices_normalized = vertices_centered / max_dist
            
            # FIXED: Correct coordinate system alignment for TRELLIS
            # TRELLIS uses a different coordinate system, so we need to adjust
            # Apply the same transformation that TRELLIS uses internally
            x, y, z = vertices_normalized[:, 0], vertices_normalized[:, 1], vertices_normalized[:, 2]
            
            # FIXED: Corrected spherical coordinates to match TRELLIS orientation
            # Swap and adjust axes to match TRELLIS coordinate system
            theta = np.arctan2(x, z)  # FIXED: Use x,z instead of y,x for correct front-facing
            phi = np.arccos(np.clip(y, -1, 1))  # FIXED: Use y for elevation (up-down)
            
            # FIXED: Adjust UV mapping to correct the 90-degree rotation
            # Rotate UV coordinates by 90 degrees clockwise to counter the anticlockwise rotation
            u = (theta + np.pi) / (2 * np.pi)  # Azimuth: Map [-π, π] to [0, 1]
            v = 1.0 - (phi / np.pi)  # FIXED: Flip V to correct orientation, Map [0, π] to [1, 0]
            
            # FIXED: Apply UV rotation correction to fix the 90-degree Y-axis issue
            # Rotate UV coordinates by 90 degrees clockwise
            u_corrected = (u + 0.75) % 1.0  # Rotate by 270 degrees (3/4 of full rotation) 
            v_corrected = v
            
            # Create UV array with corrected coordinates
            uv_coords = np.column_stack([u_corrected, v_corrected])
            
            # Create material with the corrected image as texture
            material = trimesh.visual.material.PBRMaterial(
                roughnessFactor=0.8,
                baseColorTexture=corrected_image,
                baseColorFactor=np.array([255, 255, 255, 255], dtype=np.uint8)
            )
            
            # Create textured mesh
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

    def create_optimized_and_retextured_glb(self, outputs: Dict, output_prefix: str, quality: str, input_image: Image.Image = None, verbose: bool = True):
        if not SHAPEGEN_AVAILABLE:
            print("⚠️  Shape optimization not available. Skipping re-textured GLB.")
            return None

        print("✨ Creating fully optimized and re-textured GLB...")
        preset = self.quality_presets[quality]

        try:
            # 1. Get raw mesh from TRELLIS output
            raw_vertices = outputs['mesh'][0].vertices.cpu().numpy()
            raw_faces = outputs['mesh'][0].faces.cpu().numpy()
            raw_mesh_trimesh = trimesh.Trimesh(vertices=raw_vertices, faces=raw_faces)

            # 2. Apply Hunyuan3D shape optimization
            optimized_mesh_trimesh = self.shape_optimizer.optimize_mesh(raw_mesh_trimesh, max_faces=preset['max_faces'], verbose=True)

            # Fix for dtype mismatch: Ensure vertices are float32 before re-texturing
            optimized_mesh_trimesh.vertices = optimized_mesh_trimesh.vertices.astype(np.float32)

            # 3. TEXTURE BAKING WITH PROPER PRIORITY ORDER
            textured_mesh = None
            
            # METHOD 1: TRELLIS UV Re-baking (FIRST PRIORITY - Most Accurate)
            print("  - 🎨 Method 1: TRELLIS UV re-baking (preserves original texture quality)...")
            try:
                # Re-parameterize the optimized mesh (create new UV unwrapping)
                print("    - Re-parameterizing (UV unwrapping) the optimized mesh...")
                opt_verts_np, opt_faces_np, opt_uvs_np = postprocessing_utils.parametrize_mesh(
                    optimized_mesh_trimesh.vertices, 
                    optimized_mesh_trimesh.faces
                )

                # FIXED: Ensure all arrays are float32 to avoid dtype mismatch errors
                opt_verts_np = opt_verts_np.astype(np.float32)
                opt_faces_np = opt_faces_np.astype(np.int32)  # Faces should be int32
                opt_uvs_np = opt_uvs_np.astype(np.float32)

                if verbose:
                    print(f"    - Array dtypes: vertices={opt_verts_np.dtype}, faces={opt_faces_np.dtype}, uvs={opt_uvs_np.dtype}")

                # Render multiview images from Gaussians at higher resolution for better quality
                print("    - Rendering multiview images from Gaussians...")
                observations, extrinsics, intrinsics = postprocessing_utils.render_multiview(
                    outputs['gaussian'][0], 
                    resolution=1024,  # High resolution for better texture quality
                    nviews=120        # More views for better coverage
                )
                
                # Create masks and convert to numpy with consistent dtypes
                masks = [np.any(observation > 0, axis=-1) for observation in observations]
                
                # FIXED: Ensure extrinsics and intrinsics are float32
                extrinsics = [extrinsics[i].cpu().numpy().astype(np.float32) for i in range(len(extrinsics))]
                intrinsics = [intrinsics[i].cpu().numpy().astype(np.float32) for i in range(len(intrinsics))]
                
                # FIXED: Ensure observations are also float32
                observations = [obs.astype(np.float32) for obs in observations]

                if verbose:
                    print(f"    - Multiview data: {len(observations)} views, extrinsics dtype={extrinsics[0].dtype}, intrinsics dtype={intrinsics[0].dtype}")

                # Bake texture onto the new UV map with optimized parameters
                print("    - Baking textures onto the new UV map...")
                texture_image = postprocessing_utils.bake_texture(
                    opt_verts_np, opt_faces_np, opt_uvs_np,
                    observations, masks, extrinsics, intrinsics,
                    texture_size=preset['texture_size'], 
                    mode='opt',
                    lambda_tv=0.005,  # Lower TV regularization for sharper textures
                    verbose=verbose
                )
                texture = Image.fromarray(texture_image)

                # Assemble the final textured and optimized GLB with corrected coordinates
                print("    - Assembling final GLB with corrected coordinates...")
                # FIXED: Corrected coordinate transformation to fix 90-degree Y-axis rotation
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
            
            # METHOD 2: Hunyuan3D Paint Pipeline (SECOND PRIORITY - Good Fallback)
            if textured_mesh is None and input_image is not None and HUNYUAN_TEXGEN_AVAILABLE:
                print("  - 🎨 Method 2: Hunyuan3D texture painting (good fallback)...")
                textured_mesh = self.apply_hunyuan_texture(optimized_mesh_trimesh, input_image, verbose=verbose)
            
            # METHOD 3: Simple Spherical Projection (LAST RESORT - Basic Fallback)
            if textured_mesh is None and input_image is not None:
                print("  - 🎨 Method 3: Simple spherical projection (last resort)...")
                textured_mesh = self.apply_simple_texture_projection(optimized_mesh_trimesh, input_image, verbose=verbose)

            # Export the final textured mesh
            if textured_mesh is not None:
                glb_path = f"{output_prefix}_retextured_optimized.glb"
                textured_mesh.export(glb_path)
                print(f"✅ Successfully created fully optimized and re-textured GLB: {glb_path}")
                return glb_path
            else:
                print("❌ All texture methods failed")
                return None
                
        except Exception as e:
            print(f"❌ Re-texturing of optimized mesh failed: {e}")
            return None

    def run_pipeline(self, prompt: str, quality: str = 'standard', output_prefix: str = "flux_trellis", seed: int = 42) -> Dict:
        print("=" * 60)
        print("🚀 FLUX + TRELLIS + SHAPE OPTIMIZATION + RE-TEXTURING")
        print("=" * 60)
        
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
            
            # Create standard textured GLB
            preset = self.quality_presets[quality]
            print(f"📦 Creating standard textured GLB ({preset['texture_size']}^2)...")
            glb_textured = postprocessing_utils.to_glb(outputs['gaussian'][0], outputs['mesh'][0], simplify=0.98, texture_size=preset['texture_size'], fill_holes=True, verbose=True)
            textured_path = f"{output_prefix}_textured.glb"
            glb_textured.export(textured_path)
            processed_files['glb_textured'] = textured_path
            
            # Create shape-optimized and re-textured GLB
            optimized_path = self.create_optimized_and_retextured_glb(outputs, output_prefix, quality, processed_image)
            if optimized_path:
                processed_files['glb_retextured_optimized'] = optimized_path
            
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
            
            print("\n" + "=" * 60)
            print("✅ GENERATION COMPLETE!")
            for file_type, path in processed_files.items():
                print(f"  - {file_type}: {path}")
            print("=" * 60)

            return {'files': processed_files, 'settings': {'quality': quality, 'prompt': prompt, 'seed': seed}}
        except Exception as e:
            print(f"❌ Error in pipeline: {e}")
            raise
        finally:
            self.unload_trellis_pipeline()
            clear_gpu_memory()

def get_user_quality_choice():
    print("\n🎯 QUALITY SELECTION")
    print("=" * 40)
    print("1. draft   - Fast, low quality (1K textures, 20K faces)")
    print("2. standard - Recommended balance (2K textures, 40K faces)")
    print("3. high    - High quality (4K textures, 60K faces)")
    print("4. ultra   - Maximum quality (6K textures, 80K faces)")
    print("=" * 40)
    quality_map = {'1': 'draft', '2': 'standard', '3': 'high', '4': 'ultra'}
    choice = input("Select quality level (1-4) [2 for standard]: ").strip() or '2'
    selected = quality_map.get(choice, 'standard')
    print(f"✅ Selected: {selected.upper()}")
    return selected

if __name__ == "__main__":
    pipeline = FluxTrellisRetexturedOptimized()
    quality_level = get_user_quality_choice()
    
    test_prompts = [
        "charming red barn with weathered wood without any windows or base",
        "a blue monkey sitting on temple", "football", "purple durable robotic arm", 
        "flying dragon", "a knight", "a realistic darth vader", "a animated skeleton"
    ]
    
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"retextured_optimized_outputs_{timestamp}_{quality_level}"
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"\n🚀 Processing {len(test_prompts)} prompts...")
    print(f"📁 Output directory: {output_dir}")
    print(f"🎯 Quality Level: {quality_level.upper()}")
    
    for idx, prompt in enumerate(test_prompts):
        safe_prefix = f"prompt_{idx+1}_{re.sub(r'[^a-zA-Z0-9]+', '_', prompt)[:32]}"
        output_prefix = os.path.join(output_dir, safe_prefix)
        
        print(f"\n{'='*80}\n🎯 Processing prompt {idx+1}/{len(test_prompts)}: '{prompt}'\n{'='*80}")
        
        try:
            pipeline.run_pipeline(prompt=prompt, quality=quality_level, output_prefix=output_prefix, seed=42 + idx * 100)
            print(f"\n✅ Prompt {idx+1} complete. Files saved to: {output_dir}")
        except Exception as e:
            print(f"❌ Error processing prompt {idx+1}: {e}")
            continue
    
    print(f"\n🎉 All prompts processed! Results saved in: {output_dir}")
    print(f"\n📋 PIPELINE SUMMARY:")
    print(f"- 🖼️  Text → Flux → Image")
    print(f"- 🎨 Background Removal")
    print(f"- 📦 Image → TRELLIS → Raw 3D Assets")
    print(f"- 🔧 Shape Optimization + Re-Texturing")
    print(f"- 💾 Outputs: Textured GLB & Shape-Optimized+Re-Textured GLB")
    preset = pipeline.quality_presets[quality_level]
    print(f"\n🎨 SETTINGS ({quality_level.upper()}):")
    print(f"- Texture Resolution: {preset['texture_size']}^2 | Max Faces: {preset['max_faces']:,}") 