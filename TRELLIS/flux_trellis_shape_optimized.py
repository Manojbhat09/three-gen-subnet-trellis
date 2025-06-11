import os
import datetime
import torch
import numpy as np
import gc
import re
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
        
        # Step 4: Apply mesh simplification
        if verbose:
            print("  - Applying mesh simplification...")
        try:
            mesh = self.mesh_simplifier(mesh)
        except Exception as e:
            if verbose:
                print(f"    ⚠️  Mesh simplification failed: {e}")
        
        if verbose:
            print("✅ Shape optimization complete!")
        
        return mesh

class FluxTrellisShapeOptimized:
    """
    Advanced Flux + TRELLIS pipeline with Hunyuan3D shape optimization
    
    Pipeline:
    1. Text → Flux → Image
    2. Image → Background Removal
    3. Image → TRELLIS → 3D Assets
    4. 3D Assets → Shape Optimization
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
        
        # Quality presets - standard is now default
        self.quality_presets = {
            'draft': {
                'sparse_steps': 15,
                'slat_steps': 15,
                'cfg_strength': 6.0,
                'texture_size': 1024,
                'max_faces': 20000,
            },
            'standard': {  # NEW DEFAULT - best balance
                'sparse_steps': 25,
                'slat_steps': 25,
                'cfg_strength': 7.5,
                'texture_size': 2048,
                'max_faces': 40000,
            },
            'high': {
                'sparse_steps': 40,
                'slat_steps': 40,
                'cfg_strength': 9.0,
                'texture_size': 4096,
                'max_faces': 60000,
            },
            'ultra': {
                'sparse_steps': 60,
                'slat_steps': 60,
                'cfg_strength': 10.5,
                'texture_size': 6144,
                'max_faces': 80000,
            }
        }
    
    def load_trellis_pipeline(self):
        """Load TRELLIS pipeline for image-to-3D generation"""
        if self.trellis_pipeline is None:
            print("📦 Loading TRELLIS Image-to-3D pipeline...")
            self.trellis_pipeline = TrellisImageTo3DPipeline.from_pretrained(self.trellis_model_path)
            self.trellis_pipeline.cuda()
        return self.trellis_pipeline
    
    def unload_trellis_pipeline(self):
        """Unload TRELLIS pipeline to free memory"""
        if self.trellis_pipeline is not None:
            print("🧹 Unloading TRELLIS pipeline...")
            del self.trellis_pipeline
            self.trellis_pipeline = None
            clear_gpu_memory()
    
    def load_flux_pipeline(self):
        """Load Flux pipeline for text-to-image generation"""
        print("🖼️  Loading Flux pipeline...")
        
        huggingface_token = os.getenv("HUGGINGFACE_TOKEN")
        
        # Flux model configuration
        file_url = "https://huggingface.co/gokaygokay/flux-game/blob/main/hyperflux_00001_.q8_0.gguf"
        file_url = file_url.replace("/resolve/main/", "/blob/main/").replace("?download=true", "")
        single_file_base_model = "camenduru/FLUX.1-dev-diffusers"
        
        try:
            # Load text encoder with 8-bit quantization
            quantization_config_tf = BitsAndBytesConfigTF(
                load_in_8bit=True,
                bnb_8bit_compute_dtype=torch.bfloat16
            )
            text_encoder_2 = T5EncoderModel.from_pretrained(
                single_file_base_model, 
                subfolder="text_encoder_2", 
                torch_dtype=self.dtype, 
                quantization_config=quantization_config_tf, 
                token=huggingface_token
            )
            
            # Load transformer with GGUF configuration
            transformer = FluxTransformer2DModel.from_single_file(
                file_url, 
                subfolder="transformer", 
                quantization_config=GGUFQuantizationConfig(compute_dtype=self.dtype), 
                torch_dtype=self.dtype, 
                config=single_file_base_model
            )
            
            # Initialize pipeline
            flux_pipeline = FluxPipeline.from_pretrained(
                single_file_base_model, 
                transformer=transformer, 
                text_encoder_2=text_encoder_2, 
                torch_dtype=self.dtype, 
                token=huggingface_token
            )
            flux_pipeline.to(self.device)
            
            return flux_pipeline
            
        except Exception as e:
            print(f"❌ Error loading Flux pipeline: {e}")
            raise
    
    def generate_image_from_text(self, prompt: str, seed: int = 42, 
                                width: int = 1024, height: int = 1024) -> Image.Image:
        """Generate image from text using Flux"""
        flux_pipeline = self.load_flux_pipeline()
        
        try:
            # Enhance prompt for 3D-style generation
            enhanced_prompt = f"wbgmsst, {prompt}, 3D isometric asset, clean white background, isolated object"
            
            print(f"🎯 Generating image from prompt: {enhanced_prompt}")
            generator = torch.Generator(device=self.device).manual_seed(seed)
            
            image = flux_pipeline(
                prompt=enhanced_prompt,
                guidance_scale=3.5,
                num_inference_steps=NUM_INFERENCE_STEPS,
                width=width,
                height=height,
                generator=generator,
            ).images[0]
            
            return image
            
        finally:
            # Clear Flux pipeline from memory
            del flux_pipeline
            clear_gpu_memory()
    
    def remove_background(self, image: Image.Image) -> Image.Image:
        """Remove background from image if rembg is available"""
        if self.bg_remover is not None:
            print("🎨 Removing background...")
            return self.bg_remover(image)
        else:
            print("⚠️  Background removal not available, using original image")
            return image
    
    def create_adaptive_guidance_schedule(self, steps: int, base_cfg: float) -> Tuple[float, Tuple[float, float]]:
        """Create adaptive guidance scheduling for better quality"""
        if steps >= 50:
            cfg_strength = base_cfg * 1.1
            cfg_interval = (0.3, 0.98)
        elif steps >= 30:
            cfg_strength = base_cfg * 1.05
            cfg_interval = (0.4, 0.95)
        else:
            cfg_strength = base_cfg
            cfg_interval = (0.5, 0.9)
        
        return cfg_strength, cfg_interval
    
    def generate_3d_from_image(self, image: Image.Image, quality: str = 'standard', seed: int = 42) -> Dict:
        """Generate 3D assets from image using TRELLIS"""
        if quality not in self.quality_presets:
            raise ValueError(f"Quality must be one of {list(self.quality_presets.keys())}")
        
        # Load TRELLIS pipeline
        trellis_pipeline = self.load_trellis_pipeline()
        
        preset = self.quality_presets[quality]
        
        # Create adaptive guidance schedule
        sparse_cfg, sparse_interval = self.create_adaptive_guidance_schedule(
            preset['sparse_steps'], preset['cfg_strength']
        )
        slat_cfg, slat_interval = self.create_adaptive_guidance_schedule(
            preset['slat_steps'], preset['cfg_strength']
        )
        
        print(f"🎯 Generating 3D assets with quality: {quality}")
        
        # Enhanced sampling parameters
        sparse_structure_params = {
            "steps": preset['sparse_steps'],
            "cfg_strength": sparse_cfg,
            "cfg_interval": sparse_interval,
            "rescale_t": 3.0,
        }
        
        slat_params = {
            "steps": preset['slat_steps'],
            "cfg_strength": slat_cfg,
            "cfg_interval": slat_interval,
            "rescale_t": 3.0,
        }
        
        # Generate 3D assets
        outputs = trellis_pipeline.run(
            image,
            seed=seed,
            sparse_structure_sampler_params=sparse_structure_params,
            slat_sampler_params=slat_params,
        )
        
        return outputs
    
    def postprocess_with_shape_optimization(self, outputs: Dict, output_prefix: str, quality: str) -> Dict:
        """Apply enhanced post-processing with shape optimization"""
        processed_files = {}
        preset = self.quality_presets[quality]
        
        # Render videos
        if 'gaussian' in outputs:
            print("🎬 Rendering Gaussian Splatting...")
            video = render_utils.render_video(
                outputs['gaussian'][0], 
                render_size=(1024, 1024),
                num_frames=120,
                ss_level=2
            )['color']
            video_path = f"{output_prefix}_gs.mp4"
            imageio.mimsave(video_path, video, fps=30)
            processed_files['gaussian_video'] = video_path
        
        if 'mesh' in outputs:
            print("🎬 Rendering Mesh...")
            video = render_utils.render_video(
                outputs['mesh'][0],
                render_size=(1024, 1024),
                num_frames=120
            )['normal']
            video_path = f"{output_prefix}_mesh.mp4"
            imageio.mimsave(video_path, video, fps=30)
            processed_files['mesh_video'] = video_path
        
        # Create multiple GLB versions
        if 'gaussian' in outputs and 'mesh' in outputs:
            # 1. Standard textured GLB (no shape optimization)
            print(f"📦 Creating standard textured GLB with {preset['texture_size']}x{preset['texture_size']} textures...")
            glb_textured = postprocessing_utils.to_glb(
                outputs['gaussian'][0],
                outputs['mesh'][0],
                simplify=0.98,
                texture_size=preset['texture_size'],
                fill_holes=True,
                verbose=True
            )
            glb_textured_path = f"{output_prefix}_textured.glb"
            glb_textured.export(glb_textured_path)
            processed_files['glb_textured'] = glb_textured_path
            print(f"✅ Saved textured GLB: {glb_textured_path}")
            
            # 2. Shape-optimized version (with texture preservation attempt)
            if SHAPEGEN_AVAILABLE:
                print("🔧 Creating shape-optimized version with texture preservation...")
                
                # Try to preserve textures during optimization
                try:
                    # First create a copy for optimization
                    import copy
                    glb_for_optimization = copy.deepcopy(glb_textured)
                    
                    # Apply shape optimization
                    glb_optimized = self.shape_optimizer.optimize_mesh(
                        glb_for_optimization, 
                        max_faces=preset['max_faces'],
                        verbose=True
                    )
                    
                    # Export shape-optimized version
                    glb_optimized_path = f"{output_prefix}_shape_optimized.glb"
                    glb_optimized.export(glb_optimized_path)
                    processed_files['glb_shape_optimized'] = glb_optimized_path
                    print(f"✅ Saved shape-optimized GLB: {glb_optimized_path}")
                    
                except Exception as e:
                    print(f"⚠️  Shape optimization with texture preservation failed: {e}")
                    
                    # Fallback: Create shape-optimized version without textures but note it
                    print("🔧 Creating shape-optimized version (geometry only)...")
                    try:
                        # Apply shape optimization to raw mesh
                        raw_mesh = outputs['mesh'][0]
                        optimized_mesh = self.shape_optimizer.optimize_mesh(
                            raw_mesh, 
                            max_faces=preset['max_faces'],
                            verbose=True
                        )
                        
                        # Create GLB from optimized mesh (will have less texture quality)
                        glb_shape_only = postprocessing_utils.to_glb(
                            outputs['gaussian'][0],
                            optimized_mesh,
                            simplify=0.99,  # Less simplification since we already optimized
                            texture_size=preset['texture_size'],
                            fill_holes=False,  # Already done in shape optimization
                            verbose=True
                        )
                        
                        glb_shape_only_path = f"{output_prefix}_shape_optimized.glb"
                        glb_shape_only.export(glb_shape_only_path)
                        processed_files['glb_shape_optimized'] = glb_shape_only_path
                        print(f"✅ Saved shape-optimized GLB: {glb_shape_only_path}")
                        
                    except Exception as e2:
                        print(f"❌ Shape optimization failed completely: {e2}")
                        print("📦 Only textured version will be available")
            else:
                print("⚠️  Shape optimization not available - only textured version created")
        
        # Enhanced PLY export
        if 'gaussian' in outputs:
            print("💎 Optimizing and saving Gaussians...")
            simplified_gs = postprocessing_utils.simplify_gs(
                outputs['gaussian'][0],
                simplify=0.98,
                verbose=True
            )
            ply_path = f"{output_prefix}_optimized.ply"
            simplified_gs.save_ply(ply_path)
            processed_files['ply'] = ply_path
        
        return processed_files
    
    def run_text_to_3d_pipeline(self, prompt: str, quality: str = 'standard', 
                                output_prefix: str = "flux_trellis", seed: int = 42) -> Dict:
        """Run the complete text-to-image-to-3D pipeline with shape optimization"""
        print("=" * 60)
        print("🚀 FLUX + TRELLIS + SHAPE OPTIMIZATION")
        print("=" * 60)
        
        try:
            # Clear GPU memory before starting
            print("🧹 Clearing GPU memory before processing...")
            clear_gpu_memory()
            
            # Step 1: Generate image from text
            print("Step 1: Generating image from text...")
            image = self.generate_image_from_text(prompt, seed)
            
            # Save original image
            original_image_path = f"{output_prefix}_original.png"
            image.save(original_image_path)
            print(f"💾 Saved original image: {original_image_path}")
            
            # Clear memory after Flux generation
            print("🧹 Clearing GPU memory after image generation...")
            clear_gpu_memory()
            
            # Step 2: Remove background
            print("Step 2: Processing image...")
            processed_image = self.remove_background(image)
            
            # Save processed image
            processed_image_path = f"{output_prefix}_processed.png"
            processed_image.save(processed_image_path)
            print(f"💾 Saved processed image: {processed_image_path}")
            
            # Step 3: Generate 3D from image
            print("Step 3: Generating 3D assets from image...")
            outputs = self.generate_3d_from_image(processed_image, quality, seed)
            
            # Step 4: Post-processing with shape optimization
            print("Step 4: Applying enhanced post-processing with shape optimization...")
            processed_files = self.postprocess_with_shape_optimization(outputs, output_prefix, quality)
            
            # Combine results
            result = {
                'raw_outputs': outputs,
                'files': processed_files,
                'images': {
                    'original': original_image_path,
                    'processed': processed_image_path
                },
                'settings': {
                    'quality': quality,
                    'prompt': prompt,
                    'seed': seed
                }
            }
            
            print("\n" + "=" * 60)
            print("✅ GENERATION COMPLETE!")
            print("Generated files:")
            print(f"  📸 Original image: {original_image_path}")
            print(f"  🎨 Processed image: {processed_image_path}")
            for file_type, path in processed_files.items():
                if 'glb_textured' in file_type:
                    preset = self.quality_presets[quality]
                    print(f"  📦 {file_type}: {path} ({preset['texture_size']}x{preset['texture_size']} textures, full quality)")
                elif 'glb_shape_optimized' in file_type:
                    preset = self.quality_presets[quality]
                    print(f"  🔧 {file_type}: {path} (shape optimized, {preset['max_faces']:,} max faces)")
                elif 'glb' in file_type:
                    preset = self.quality_presets[quality]
                    print(f"  📦 {file_type}: {path} ({preset['texture_size']}x{preset['texture_size']} textures)")
                else:
                    print(f"  🎬 {file_type}: {path}")
            
            print(f"\n🎯 Quality Level: {quality.upper()}")
            preset = self.quality_presets[quality]
            print(f"   - Texture Resolution: {preset['texture_size']}x{preset['texture_size']}")
            print(f"   - Max Faces: {preset['max_faces']:,}")
            print(f"   - Sampling Steps: {preset['sparse_steps']}/{preset['slat_steps']}")
            if SHAPEGEN_AVAILABLE:
                print(f"   - Shape Optimization: ✅ Enabled")
                print(f"   - Output Versions: 🎨 Textured + 🔧 Shape-Optimized")
            else:
                print(f"   - Shape Optimization: ⚠️  Disabled")
                print(f"   - Output Versions: 🎨 Textured Only")
            print("=" * 60)
            
            return result
            
        except Exception as e:
            print(f"❌ Error in pipeline: {e}")
            raise
        finally:
            # Unload TRELLIS pipeline and clear memory after each prompt
            self.unload_trellis_pipeline()
            clear_gpu_memory()


def get_user_quality_choice():
    """Get quality selection from user once"""
    print("\n🎯 QUALITY SELECTION")
    print("=" * 40)
    print("Available quality levels:")
    print("1. draft   - Fast generation (1K textures, 20K faces)")
    print("2. standard - Best balance (2K textures, 40K faces) [RECOMMENDED]")
    print("3. high    - High quality (4K textures, 60K faces)")
    print("4. ultra   - Maximum quality (6K textures, 80K faces)")
    print("=" * 40)
    
    quality_map = {
        '1': 'draft',
        '2': 'standard', 
        '3': 'high',
        '4': 'ultra'
    }
    
    while True:
        choice = input("Select quality level (1-4) [2 for standard]: ").strip()
        
        if not choice:  # Default to standard
            choice = '2'
        
        if choice in quality_map:
            selected = quality_map[choice]
            print(f"✅ Selected: {selected.upper()}")
            return selected
        else:
            print("❌ Invalid choice. Please enter 1, 2, 3, or 4.")


# Example usage and batch processing
if __name__ == "__main__":
    # Initialize the combined pipeline
    pipeline = FluxTrellisShapeOptimized()
    
    # Get quality selection from user once
    quality_level = get_user_quality_choice()
    
    # Test prompts
    test_prompts = [
        "charming red barn with weathered wood without any windows or base",
        "a blue monkey sitting on temple",
        "football",
        "purple durable robotic arm", 
        "flying dragon", 
        "a knight", 
        "a realistic darth vader", 
        "a animated skeleton"
    ]
    
    # Create timestamped output folder
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"shape_optimized_outputs_{timestamp}_{quality_level}"
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"\n🚀 Processing {len(test_prompts)} prompts with Flux + TRELLIS + Shape Optimization...")
    print(f"📁 Output directory: {output_dir}")
    print(f"🕒 Timestamp: {timestamp}")
    print(f"🎯 Quality Level: {quality_level.upper()}")
    print(f"🔧 Shape Optimization: {'✅ Enabled' if SHAPEGEN_AVAILABLE else '⚠️  Disabled'}\n")
    
    for idx, prompt in enumerate(test_prompts):
        # Create safe filename prefix
        safe_prefix = f"prompt_{idx+1}_{re.sub(r'[^a-zA-Z0-9]+', '_', prompt)[:32]}"
        output_prefix = os.path.join(output_dir, safe_prefix)
        
        print(f"\n{'='*80}")
        print(f"🎯 Processing prompt {idx+1}/{len(test_prompts)}:")
        print(f"📝 Prompt: {prompt}")
        print(f"📁 Output prefix: {output_prefix}")
        print(f"{'='*80}")
        
        # Clear GPU memory before each prompt
        print("🧹 Clearing GPU memory before prompt processing...")
        clear_gpu_memory()
        
        try:
            result = pipeline.run_text_to_3d_pipeline(
                prompt=prompt,
                quality=quality_level,
                output_prefix=output_prefix,
                seed=42 + idx * 100
            )
            print(f"\n✅ Prompt {idx+1} complete. Files saved to: {output_dir}")
            
            # Additional cleanup after each prompt
            print("🧹 Final cleanup after prompt completion...")
            clear_gpu_memory()
            
        except Exception as e:
            print(f"❌ Error processing prompt {idx+1}: {e}")
            # Clear memory even on error
            pipeline.unload_trellis_pipeline()
            clear_gpu_memory()
            continue
    
    print(f"\n🎉 All prompts processed!")
    print(f"📁 Results saved in: {output_dir}")
    print(f"🕒 Timestamp: {timestamp}")
    print(f"\n📋 PIPELINE SUMMARY:")
    print(f"- 🖼️  Text → Flux → High-quality 1024x1024 images")
    print(f"- 🎨 Background removal with proper alpha channel")
    print(f"- 📦 Image → TRELLIS → 3D assets (mesh, gaussians, radiance fields)")
    print(f"- 🔧 Hunyuan3D shape optimization (floater removal, face reduction, degenerate cleanup)")
    print(f"- 🧹 Aggressive memory management between prompts")
    print(f"- 🎯 Quality Level: {quality_level.upper()}")
    
    preset = pipeline.quality_presets[quality_level]
    print(f"\n🎨 TEXTURE & MESH SETTINGS:")
    print(f"- Texture Resolution: {preset['texture_size']}x{preset['texture_size']}")
    print(f"- Maximum Faces: {preset['max_faces']:,}")
    print(f"- Sampling Steps: {preset['sparse_steps']}/{preset['slat_steps']}")
    print(f"- Shape Optimization: {'✅ Enabled' if SHAPEGEN_AVAILABLE else '⚠️  Disabled'}") 