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

# Background removal (optional - you may need to install rembg)
try:
    from rembg import remove, new_session
    REMBG_AVAILABLE = True
except ImportError:
    print("Warning: rembg not available. Background removal will be skipped.")
    REMBG_AVAILABLE = False

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

class FluxTrellisOptimized:
    """
    Combined Flux + TRELLIS pipeline for text-to-image-to-3D generation
    
    Pipeline:
    1. Text → Flux → Image
    2. Image → Background Removal (optional)
    3. Image → TRELLIS → 3D Assets
    """
    
    def __init__(self, trellis_model_path="microsoft/TRELLIS-image-large"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.dtype = torch.bfloat16
        self.trellis_model_path = trellis_model_path
        
        # Initialize background remover if available
        if REMBG_AVAILABLE:
            print("Initializing background remover...")
            self.bg_remover = BackgroundRemover()
        else:
            self.bg_remover = None
        
        # Don't load TRELLIS pipeline here - load it only when needed
        self.trellis_pipeline = None
        
        # Quality presets for TRELLIS
        self.quality_presets = {
            'draft': {
                'sparse_steps': 15,
                'slat_steps': 15,
                'cfg_strength': 6.0,
            },
            'good': {
                'sparse_steps': 25,
                'slat_steps': 25,
                'cfg_strength': 7.5,
            },
            'high': {
                'sparse_steps': 40,
                'slat_steps': 40,
                'cfg_strength': 9.0,
            },
            'ultra': {
                'sparse_steps': 60,
                'slat_steps': 60,
                'cfg_strength': 10.5,
            }
        }
    
    def load_trellis_pipeline(self):
        """Load TRELLIS pipeline for image-to-3D generation"""
        if self.trellis_pipeline is None:
            print("Loading TRELLIS Image-to-3D pipeline...")
            self.trellis_pipeline = TrellisImageTo3DPipeline.from_pretrained(self.trellis_model_path)
            self.trellis_pipeline.cuda()
        return self.trellis_pipeline
    
    def unload_trellis_pipeline(self):
        """Unload TRELLIS pipeline to free memory"""
        if self.trellis_pipeline is not None:
            print("Unloading TRELLIS pipeline...")
            del self.trellis_pipeline
            self.trellis_pipeline = None
            clear_gpu_memory()
    
    def load_flux_pipeline(self):
        """Load Flux pipeline for text-to-image generation"""
        print("Loading Flux pipeline...")
        
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
            print(f"Error loading Flux pipeline: {e}")
            raise
    
    def generate_image_from_text(self, prompt: str, seed: int = 42, 
                                width: int = 1024, height: int = 1024) -> Image.Image:
        """Generate image from text using Flux"""
        # Load Flux pipeline
        flux_pipeline = self.load_flux_pipeline()
        
        try:
            # Enhance prompt for 3D-style generation
            enhanced_prompt = f"wbgmsst, {prompt}, 3D isometric asset, clean white background, isolated object"
            
            print(f"Generating image from prompt: {enhanced_prompt}")
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
            print("Removing background...")
            return self.bg_remover(image)
        else:
            print("Background removal not available, using original image")
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
    
    def generate_3d_from_image(self, image: Image.Image, quality: str = 'good', seed: int = 42) -> Dict:
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
        
        print(f"Generating 3D assets with quality: {quality}")
        
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
    
    def enhanced_postprocessing(self, outputs: Dict, output_prefix: str = "optimized") -> Dict:
        """Apply enhanced post-processing for better quality outputs"""
        processed_files = {}
        
        # Render with higher quality settings
        if 'gaussian' in outputs:
            print("Rendering Gaussian Splatting...")
            video = render_utils.render_video(
                outputs['gaussian'][0], 
                render_size=(1024, 1024),
                num_frames=120,
                ss_level=2
            )['color']
            video_path = f"{output_prefix}_gs_hq.mp4"
            imageio.mimsave(video_path, video, fps=30)
            processed_files['gaussian_video'] = video_path
        
        if 'radiance_field' in outputs:
            print("Rendering Radiance Field...")
            video = render_utils.render_video(
                outputs['radiance_field'][0],
                render_size=(1024, 1024),
                num_frames=120,
                ss_level=2
            )['color']
            video_path = f"{output_prefix}_rf_hq.mp4"
            imageio.mimsave(video_path, video, fps=30)
            processed_files['radiance_field_video'] = video_path
        
        if 'mesh' in outputs:
            print("Rendering Mesh...")
            video = render_utils.render_video(
                outputs['mesh'][0],
                render_size=(1024, 1024),
                num_frames=120
            )['normal']
            video_path = f"{output_prefix}_mesh_hq.mp4"
            imageio.mimsave(video_path, video, fps=30)
            processed_files['mesh_video'] = video_path
        
        # Enhanced GLB export with ultra-high quality textures
        if 'gaussian' in outputs and 'mesh' in outputs:
            print("Creating enhanced GLB with high-quality textures...")
            
            # Standard quality GLB
            glb_standard = postprocessing_utils.to_glb(
                outputs['gaussian'][0],
                outputs['mesh'][0],
                simplify=0.98,           
                texture_size=2048,       
                fill_holes=True,         
                verbose=True
            )
            glb_standard_path = f"{output_prefix}_standard.glb"
            glb_standard.export(glb_standard_path)
            processed_files['glb_standard'] = glb_standard_path
            
            # Ultra-high quality GLB with 4K textures
            print("Creating ultra-high quality GLB with 4K textures...")
            glb_ultra = postprocessing_utils.to_glb(
                outputs['gaussian'][0],
                outputs['mesh'][0],
                simplify=0.99,           # Minimal mesh simplification
                texture_size=4096,       # Ultra-high texture resolution
                fill_holes=True,         
                verbose=True
            )
            glb_ultra_path = f"{output_prefix}_ultra_4k.glb"
            glb_ultra.export(glb_ultra_path)
            processed_files['glb_ultra'] = glb_ultra_path
            
            # Super quality GLB with 8K textures (if you have enough VRAM)
            try:
                print("Attempting super-quality GLB with 8K textures...")
                glb_super = postprocessing_utils.to_glb(
                    outputs['gaussian'][0],
                    outputs['mesh'][0],
                    simplify=0.995,          # Almost no mesh simplification
                    texture_size=8192,       # 8K texture resolution
                    fill_holes=True,         
                    verbose=True
                )
                glb_super_path = f"{output_prefix}_super_8k.glb"
                glb_super.export(glb_super_path)
                processed_files['glb_super'] = glb_super_path
                print("Successfully created 8K texture GLB!")
            except Exception as e:
                print(f"8K texture generation failed (likely VRAM limitation): {e}")
                print("Continuing with 4K texture version...")
        
        # Enhanced PLY export with optimized Gaussians
        if 'gaussian' in outputs:
            print("Optimizing and saving Gaussians...")
            
            # Standard simplified version
            simplified_gs = postprocessing_utils.simplify_gs(
                outputs['gaussian'][0],
                simplify=0.98,
                verbose=True
            )
            ply_path = f"{output_prefix}_optimized.ply"
            simplified_gs.save_ply(ply_path)
            processed_files['ply_optimized'] = ply_path
            
            # High-quality version with less simplification
            try:
                print("Creating high-quality Gaussian version...")
                hq_gs = postprocessing_utils.simplify_gs(
                    outputs['gaussian'][0],
                    simplify=0.99,  # Less simplification for higher quality
                    verbose=True
                )
                ply_hq_path = f"{output_prefix}_hq.ply"
                hq_gs.save_ply(ply_hq_path)
                processed_files['ply_hq'] = ply_hq_path
            except Exception as e:
                print(f"High-quality Gaussian generation failed: {e}")
        
        return processed_files
    
    def create_texture_enhanced_glb(self, outputs: Dict, output_prefix: str, 
                                   texture_size: int = 4096, simplify: float = 0.99) -> str:
        """Create a GLB with enhanced texture quality using advanced baking"""
        if 'gaussian' not in outputs or 'mesh' not in outputs:
            raise ValueError("Both gaussian and mesh outputs required for texture enhancement")
        
        print(f"Creating texture-enhanced GLB with {texture_size}x{texture_size} textures...")
        
        # Use custom parameters for better texture quality
        glb = postprocessing_utils.to_glb(
            outputs['gaussian'][0],
            outputs['mesh'][0],
            simplify=simplify,
            fill_holes=True,
            fill_holes_max_size=0.02,    # Smaller holes for better detail
            texture_size=texture_size,
            debug=False,
            verbose=True
        )
        
        glb_path = f"{output_prefix}_enhanced_tex_{texture_size}k.glb"
        glb.export(glb_path)
        
        return glb_path
    
    def run_text_to_3d_pipeline(self, prompt: str, quality: str = 'good', 
                                output_prefix: str = "flux_trellis", seed: int = 42, 
                                enhance_textures: bool = True) -> Dict:
        """Run the complete text-to-image-to-3D pipeline"""
        print("=" * 60)
        print("FLUX + TRELLIS TEXT-TO-3D GENERATION")
        print("=" * 60)
        
        try:
            # Clear GPU memory before starting
            print("Clearing GPU memory before processing...")
            clear_gpu_memory()
            
            # Step 1: Generate image from text
            print("Step 1: Generating image from text...")
            image = self.generate_image_from_text(prompt, seed)
            
            # Save original image
            original_image_path = f"{output_prefix}_original.png"
            image.save(original_image_path)
            print(f"Saved original image: {original_image_path}")
            
            # Clear memory after Flux generation
            print("Clearing GPU memory after image generation...")
            clear_gpu_memory()
            
            # Step 2: Remove background
            print("Step 2: Processing image...")
            processed_image = self.remove_background(image)
            
            # Save processed image
            processed_image_path = f"{output_prefix}_processed.png"
            processed_image.save(processed_image_path)
            print(f"Saved processed image: {processed_image_path}")
            
            # Step 3: Generate 3D from image
            print("Step 3: Generating 3D assets from image...")
            outputs = self.generate_3d_from_image(processed_image, quality, seed)
            
            # Step 4: Post-processing
            print("Step 4: Applying enhanced post-processing...")
            processed_files = self.enhanced_postprocessing(outputs, output_prefix)
            
            # Step 5: Additional texture enhancement (optional)
            if enhance_textures:
                print("Step 5: Creating additional texture-enhanced versions...")
                try:
                    # Create a custom 6K texture version
                    tex_6k_path = self.create_texture_enhanced_glb(
                        outputs, output_prefix, texture_size=6144, simplify=0.995
                    )
                    processed_files['glb_custom_6k'] = tex_6k_path
                except Exception as e:
                    print(f"6K texture enhancement failed: {e}")
            
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
                    'seed': seed,
                    'enhance_textures': enhance_textures
                }
            }
            
            print("\n" + "=" * 60)
            print("GENERATION COMPLETE!")
            print("Generated files:")
            print(f"  Original image: {original_image_path}")
            print(f"  Processed image: {processed_image_path}")
            for file_type, path in processed_files.items():
                if 'glb' in file_type:
                    if 'standard' in file_type:
                        print(f"  {file_type}: {path} (2K textures)")
                    elif 'ultra' in file_type:
                        print(f"  {file_type}: {path} (4K textures)")
                    elif 'super' in file_type:
                        print(f"  {file_type}: {path} (8K textures)")
                    elif 'custom' in file_type:
                        print(f"  {file_type}: {path} (6K textures)")
                    else:
                        print(f"  {file_type}: {path}")
                else:
                    print(f"  {file_type}: {path}")
            
            print("\n📋 TEXTURE QUALITY SUMMARY:")
            print("- Standard GLB: 2K textures (fast, good quality)")
            print("- Ultra GLB: 4K textures (balanced quality/performance)")
            if 'glb_super' in processed_files:
                print("- Super GLB: 8K textures (maximum quality)")
            if 'glb_custom_6k' in processed_files:
                print("- Custom GLB: 6K textures (high quality alternative)")
            print("=" * 60)
            
            return result
            
        except Exception as e:
            print(f"Error in pipeline: {e}")
            raise
        finally:
            # Unload TRELLIS pipeline and clear memory after each prompt
            self.unload_trellis_pipeline()
            clear_gpu_memory()


# Example usage and batch processing
if __name__ == "__main__":
    # Initialize the combined pipeline
    pipeline = FluxTrellisOptimized()
    
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
    output_dir = f"flux_trellis_outputs_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"\nProcessing {len(test_prompts)} prompts with Flux + TRELLIS...")
    print(f"Output directory: {output_dir}")
    print(f"Timestamp: {timestamp}\n")
    
    for idx, prompt in enumerate(test_prompts):
        # Create safe filename prefix
        safe_prefix = f"prompt_{idx+1}_{re.sub(r'[^a-zA-Z0-9]+', '_', prompt)[:32]}"
        output_prefix = os.path.join(output_dir, safe_prefix)
        
        print(f"\n{'='*80}")
        print(f"Processing prompt {idx+1}/{len(test_prompts)}:")
        print(f"Prompt: {prompt}")
        print(f"Output prefix: {output_prefix}")
        print(f"{'='*80}")
        
        # Clear GPU memory before each prompt
        print("Clearing GPU memory before prompt processing...")
        clear_gpu_memory()
        
        try:
            result = pipeline.run_text_to_3d_pipeline(
                prompt=prompt,
                quality='good',  # Options: 'draft', 'good', 'high', 'ultra'
                output_prefix=output_prefix,
                seed=42 + idx * 100,
                enhance_textures=True  # Enable enhanced texture generation
            )
            print(f"\nPrompt {idx+1} complete. Files saved to: {output_dir}")
            
            # Additional cleanup after each prompt
            print("Final cleanup after prompt completion...")
            clear_gpu_memory()
            
        except Exception as e:
            print(f"Error processing prompt {idx+1}: {e}")
            # Clear memory even on error
            pipeline.unload_trellis_pipeline()
            clear_gpu_memory()
            continue
    
    print(f"\nAll prompts processed!")
    print(f"Results saved in: {output_dir}")
    print(f"Timestamp: {timestamp}")
    print(f"\nPipeline summary:")
    print(f"- Text → Flux → High-quality 1024x1024 images")
    print(f"- Optional background removal with proper alpha channel")
    print(f"- Image → TRELLIS → 3D assets (mesh, gaussians, radiance fields)")
    print(f"- Enhanced post-processing and rendering")
    print(f"- Aggressive memory management between prompts")
    print(f"\n🎨 TEXTURE QUALITY IMPROVEMENTS:")
    print(f"- Standard GLB: 2048x2048 textures")
    print(f"- Ultra GLB: 4096x4096 textures (4x better resolution)")
    print(f"- Super GLB: 8192x8192 textures (16x better resolution)")
    print(f"- Custom GLB: 6144x6144 textures (9x better resolution)")
    print(f"- Advanced texture baking with optimization")
    print(f"- Minimal mesh simplification for texture quality")
    print(f"- Better hole filling and surface parametrization") 