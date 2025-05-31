import os
import imageio
from trellis.pipelines import TrellisTextTo3DPipeline
from trellis.utils import render_utils, postprocessing_utils

# Set spconv algorithm to native for single run
os.environ['SPCONV_ALGO'] = 'native'

def main():
    # Load the TRELLIS pipeline
    print("Loading TRELLIS pipeline...")
    pipeline = TrellisTextTo3DPipeline.from_pretrained("microsoft/TRELLIS-text-xlarge")
    pipeline.cuda()
    
    # Set your text prompt
    prompt = "a blue monkey sitting on temple"
    
    print(f"Generating 3D model from prompt: {prompt}")
    # Run the pipeline with default parameters
    outputs = pipeline.run(
        prompt,
        seed=42,
        formats=['gaussian', 'mesh', 'radiance_field'],
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
    
    # Create output directory
    os.makedirs("outputs", exist_ok=True)
    
    # Save the outputs in different formats
    print("Saving outputs...")
    
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
    
    # Generate and save preview videos
    print("Rendering preview videos...")
    
    # Gaussian splatting preview
    video_gs = render_utils.render_video(outputs['gaussian'][0])['color']
    imageio.mimsave("outputs/preview_gaussian.mp4", video_gs, fps=30)
    
    # Mesh preview (with normals visualization)
    video_mesh = render_utils.render_video(outputs['mesh'][0])['normal']
    imageio.mimsave("outputs/preview_mesh.mp4", video_mesh, fps=30)
    
    # Radiance field preview
    video_rf = render_utils.render_video(outputs['radiance_field'][0])['color']
    imageio.mimsave("outputs/preview_rf.mp4", video_rf, fps=30)
    
    print("Generation complete! Files saved in outputs/:")
    print("- blue_monkey_gaussian.ply (3D Gaussians)")
    print("- blue_monkey.glb (Textured mesh)")
    print("- preview_gaussian.mp4 (Gaussian splatting preview)")
    print("- preview_mesh.mp4 (Mesh preview)")
    print("- preview_rf.mp4 (Radiance field preview)")

if __name__ == "__main__":
    main() 