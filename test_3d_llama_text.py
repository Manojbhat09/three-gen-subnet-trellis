import os
import sys
import torch
import trimesh
import numpy as np
from PIL import Image
import gc
from diffusers import AutoencoderTiny, StableDiffusionXLPipeline

# Add 3D-LLAMA to Python path
LLAMA_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "3D-LLAMA")
sys.path.append(LLAMA_PATH)

# Import after adding to path
from step1x3d_geometry.models.pipelines.pipeline import Step1X3DGeometryPipeline
from step1x3d_texture.pipelines.step1x_3d_texture_synthesis_pipeline import Step1X3DTexturePipeline
from step1x3d_geometry.models.pipelines.pipeline_utils import reduce_face, remove_degenerate_face

def clear_memory():
    """Clear CUDA cache and run garbage collection"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

def prepare_mesh_for_texturing(mesh):
    """Prepare mesh for texturing by ensuring proper color data structure"""
    # Create a new mesh with the same geometry but clean visual data
    new_mesh = trimesh.Trimesh(
        vertices=mesh.vertices,
        faces=mesh.faces,
        process=False
    )
    
    # Initialize empty vertex colors (white)
    vertex_colors = np.ones((len(mesh.vertices), 4), dtype=np.uint8) * 255
    new_mesh.visual.vertex_colors = vertex_colors
    
    # Ensure face colors are properly initialized
    face_colors = np.ones((len(mesh.faces), 4), dtype=np.uint8) * 255
    new_mesh.visual.face_colors = face_colors
    
    return new_mesh

def generate_image_from_text(prompt, height=1024, width=1024, steps=8, guidance_scale=3.5, seed=None):
    """Generate image from text using SDXL"""
    try:
        # Load SDXL pipeline
        pipe = StableDiffusionXLPipeline.from_pretrained(
            "stabilityai/stable-diffusion-xl-base-1.0",
            torch_dtype=torch.float16,
            use_safetensors=True,
            variant="fp16"
        ).to("cuda")
        
        # Generate image
        if seed is not None:
            generator = torch.Generator("cuda").manual_seed(seed)
        else:
            generator = None
            
        image = pipe(
            prompt=prompt,
            height=height,
            width=width,
            num_inference_steps=steps,
            guidance_scale=guidance_scale,
            generator=generator
        ).images[0]
        
        # Save the image
        os.makedirs("outputs", exist_ok=True)
        image_path = "outputs/generated_image.png"
        image.save(image_path)
        
        # Clean up
        del pipe
        clear_memory()
        
        return image_path
            
    except Exception as e:
        print(f"Error generating image from text: {e}")
        return None

def main():
    # Set environment variables for better memory management
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:32,expandable_segments:True'
    
    # Create output directory
    os.makedirs("outputs", exist_ok=True)
    
    # Set your text prompt
    prompt = "a blue monkey sitting on temple"
    print(f"Generating 3D model from prompt: {prompt}")
    
    # First, generate an image from the text prompt
    print("Generating image from text...")
    image_path = generate_image_from_text(
        prompt=prompt,
        height=1024,
        width=1024,
        steps=8,
        guidance_scale=3.5,
        seed=-1  # Random seed
    )
    
    if not image_path:
        raise RuntimeError("Failed to generate image from text")
    
    print(f"Generated image saved at: {image_path}")
    
    # Step 1: Generate geometry
    print("Loading geometry model...")
    geometry_model = Step1X3DGeometryPipeline.from_pretrained(
        "stepfun-ai/Step1X-3D", 
        subfolder="Step1X-3D-Geometry-Label-1300m"
    ).to("cuda")
    
    print("Generating 3D model...")
    geometry_output = geometry_model(
        image_path,
        label={"symmetry": "asymmetry", "edge_type": "sharp"},
        guidance_scale=7.5,
        octree_resolution=96,
        max_facenum=15000,
        num_inference_steps=10,
    )
    
    # Save the geometry mesh
    geometry_mesh = geometry_output.mesh[0]
    geometry_mesh = remove_degenerate_face(geometry_mesh)
    geometry_mesh = reduce_face(geometry_mesh)
    geometry_save_path = "outputs/blue_monkey_geometry.glb"
    geometry_mesh.export(geometry_save_path)
    
    # Clear memory after geometry generation
    print("Clearing memory after geometry generation...")
    del geometry_model
    del geometry_output
    clear_memory()
    
    # Step 2: Generate texture
    print("Loading texture model...")
    texture_model = Step1X3DTexturePipeline.from_pretrained(
        "stepfun-ai/Step1X-3D", 
        subfolder="Step1X-3D-Texture"
    )
    
    # Load TAESD
    print("Loading TAESD...")
    taesd = AutoencoderTiny.from_pretrained(
        "madebyollin/taesdxl",
        torch_dtype=torch.float16
    ).to("cuda")
    
    # Replace VAE in both the texture model and the pipeline
    print("Replacing VAE with TAESD...")
    if hasattr(texture_model, 'vae'):
        texture_model.vae = taesd
    
    # Replace VAE in the pipeline
    if hasattr(texture_model, 'ig2mv_pipe') and hasattr(texture_model.ig2mv_pipe, 'vae'):
        texture_model.ig2mv_pipe.vae = taesd
    
    # Add compatibility layer for TAESD
    if not hasattr(taesd, 'block_out_channels'):
        taesd.block_out_channels = taesd.decoder_block_out_channels
    
    # Move models to CPU when not in use
    print("Moving models to CPU when not in use...")
    if hasattr(texture_model.ig2mv_pipe, 'text_encoder'):
        texture_model.ig2mv_pipe.text_encoder.to("cpu")
    if hasattr(texture_model.ig2mv_pipe, 'text_encoder_2'):
        texture_model.ig2mv_pipe.text_encoder_2.to("cpu")
    if hasattr(texture_model.ig2mv_pipe, 'unet'):
        texture_model.ig2mv_pipe.unet.to("cpu")
    if hasattr(texture_model.ig2mv_pipe, 'cond_encoder'):
        texture_model.ig2mv_pipe.cond_encoder.to("cpu")
    
    print("Generating textured mesh...")
    try:
        # Prepare mesh for texturing
        prepared_mesh = prepare_mesh_for_texturing(geometry_mesh)
        
        # Move models to GPU only when needed
        if hasattr(texture_model.ig2mv_pipe, 'text_encoder'):
            texture_model.ig2mv_pipe.text_encoder.to("cuda")
        if hasattr(texture_model.ig2mv_pipe, 'text_encoder_2'):
            texture_model.ig2mv_pipe.text_encoder_2.to("cuda")
        if hasattr(texture_model.ig2mv_pipe, 'unet'):
            texture_model.ig2mv_pipe.unet.to("cuda")
        if hasattr(texture_model.ig2mv_pipe, 'cond_encoder'):
            texture_model.ig2mv_pipe.cond_encoder.to("cuda")
        
        # Generate textured mesh with memory optimizations
        with torch.no_grad():
            # Ensure mesh renderer is on GPU and using FP32
            if hasattr(texture_model, 'mesh_render'):
                texture_model.mesh_render.to("cuda")
                # Set dtype to float32 for the rasterizer
                texture_model.mesh_render.dtype = torch.float32
            textured_mesh = texture_model(image_path, prepared_mesh)
        
        textured_save_path = "outputs/blue_monkey_textured.glb"
        textured_mesh.export(textured_save_path)
        
    except Exception as e:
        print(f"Error during texturing: {str(e)}")
        print("Saving geometry-only mesh as fallback...")
        geometry_mesh.export("outputs/blue_monkey_geometry_fallback.glb")
        raise
    
    finally:
        # Clean up
        del texture_model
        del taesd
        clear_memory()
    
    print("\nGeneration complete! Files saved in outputs/:")
    print("- blue_monkey_geometry.glb (Geometry mesh)")
    print("- blue_monkey_textured.glb (Textured mesh)")

if __name__ == "__main__":
    main() 