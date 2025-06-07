import os
import sys

# Add TRELLIS to Python path
TRELLIS_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "TRELLIS")
sys.path.append(TRELLIS_PATH)

# Set spconv algorithm to native for single run and use xformers for attention
os.environ['SPCONV_ALGO'] = 'native'
os.environ['ATTN_BACKEND'] = 'xformers'  # Use xformers instead of flash-attn

import imageio
from trellis.pipelines import TrellisTextTo3DPipeline
from trellis.utils import render_utils, postprocessing_utils

def main():
    # Initialize the TRELLIS pipeline
    print("Loading TRELLIS pipeline...")
    # Use Hugging Face model path directly
    # pipeline = TrellisTextTo3DPipeline.from_pretrained("microsoft/TRELLIS-text-xlarge")#, use_auth_token=True)
    pipeline = TrellisTextTo3DPipeline.from_pretrained("Stable-X/trellis-normal-v0-1")#, use_auth_token=True)
    pipeline.cuda()  # Move to GPU
    
    # Set your text prompt
    prompt = "a blue monkey sitting on temple"
    print(f"Generating 3D model from prompt: {prompt}")
    
    # Create output directory
    os.makedirs("outputs", exist_ok=True)
    
    # Generate the 3D model
    print("Generating 3D model...")
    outputs = pipeline.run(
        prompt,
        seed=42,
        # Optional parameters for better quality
        sparse_structure_sampler_params={
            "steps": 12,  # Number of denoising steps for sparse structure
            "cfg_strength": 7.5,  # Guidance scale for sparse structure
        },
        slat_sampler_params={
            "steps": 12,  # Number of denoising steps for structured latent
            "cfg_strength": 7.5,  # Guidance scale for structured latent
        },
    )
    
    # Save the outputs in different formats
    print("Saving outputs...")
    
    # Save preview videos
    print("Rendering preview videos...")
    video = render_utils.render_video(outputs['gaussian'][0])['color']
    imageio.mimsave("outputs/preview_gaussian.mp4", video, fps=30)
    
    video = render_utils.render_video(outputs['radiance_field'][0])['color']
    imageio.mimsave("outputs/preview_rf.mp4", video, fps=30)
    
    video = render_utils.render_video(outputs['mesh'][0])['normal']
    imageio.mimsave("outputs/preview_mesh.mp4", video, fps=30)
    
    # Save as PLY file (3D Gaussians)
    outputs['gaussian'][0].save_ply("outputs/blue_monkey_gaussian.ply")
    
    # Save as GLB file (textured mesh)
    glb = postprocessing_utils.to_glb(
        outputs['gaussian'][0],
        outputs['mesh'][0],
        simplify=0.95,  # Simplify mesh to reduce file size
        texture_size=1024,  # Texture resolution
    )
    glb.export("outputs/blue_monkey.glb")
    
    print("\nGeneration complete! Files saved in outputs/:")
    print("- blue_monkey_gaussian.ply (3D Gaussians)")
    print("- blue_monkey.glb (Textured mesh)")
    print("- preview_gaussian.mp4 (Gaussian splatting preview)")
    print("- preview_rf.mp4 (Radiance field preview)")
    print("- preview_mesh.mp4 (Mesh preview)")

if __name__ == "__main__":
    main() 