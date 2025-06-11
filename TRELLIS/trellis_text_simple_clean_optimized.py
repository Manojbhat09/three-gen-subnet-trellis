import os
# os.environ['ATTN_BACKEND'] = 'xformers'   # Can be 'flash-attn' or 'xformers', default is 'flash-attn'
os.environ['SPCONV_ALGO'] = 'native'        # Can be 'native' or 'auto', default is 'auto'.
                                            # 'auto' is faster but will do benchmarking at the beginning.
                                            # Recommended to set to 'native' if run only once.

import imageio
import torch
import numpy as np
from trellis.pipelines import TrellisTextTo3DPipeline
from trellis.utils import render_utils, postprocessing_utils

# Load a pipeline from a model folder or a Hugging Face model hub.
pipeline = TrellisTextTo3DPipeline.from_pretrained("microsoft/TRELLIS-text-xlarge")
pipeline.cuda()

# Enhanced prompt preprocessing
prompt = "charming red barn with weathered wood without any windows or base"
enhanced_prompt = f"wbgmsst, {prompt}, 3D isometric asset, clean white background, isolated object"

print(f"Generating with enhanced prompt: {enhanced_prompt}")

# Enhanced sampling parameters for better quality
sparse_structure_sampler_params = {
    "steps": 35,           # Increased from default 25
    "cfg_strength": 8.5,   # Increased from default 7.5
    "cfg_interval": [0.4, 0.95],  # Extended guidance interval
    "rescale_t": 3.5,      # Slightly higher for better quality
}

slat_sampler_params = {
    "steps": 35,           # Increased from default 25
    "cfg_strength": 8.5,   # Increased from default 7.5  
    "cfg_interval": [0.4, 0.95],  # Extended guidance interval
    "rescale_t": 3.5,      # Slightly higher for better quality
}

# Run the pipeline with enhanced parameters
outputs = pipeline.run(
    enhanced_prompt,
    seed=42,
    sparse_structure_sampler_params=sparse_structure_sampler_params,
    slat_sampler_params=slat_sampler_params,
)

# Render the outputs with higher quality
print("Rendering Gaussian Splatting...")
video = render_utils.render_video(outputs['gaussian'][0], render_size=(1024, 1024), num_frames=120)['color']
imageio.mimsave("new_barn_optimized_gs.mp4", video, fps=30)

print("Rendering Radiance Field...")
video = render_utils.render_video(outputs['radiance_field'][0], render_size=(1024, 1024), num_frames=120)['color']
imageio.mimsave("new_barn_optimized_rf.mp4", video, fps=30)

print("Rendering Mesh...")
video = render_utils.render_video(outputs['mesh'][0], render_size=(1024, 1024), num_frames=120)['normal']
imageio.mimsave("new_barn_optimized_mesh.mp4", video, fps=30)

# Enhanced GLB export with better parameters
print("Creating enhanced GLB...")
glb = postprocessing_utils.to_glb(
    outputs['gaussian'][0],
    outputs['mesh'][0],
    simplify=0.98,          # Higher quality (less simplification than default 0.95)
    texture_size=2048,      # Higher texture resolution (vs default 1024)
    fill_holes=True,        # Enable hole filling
    verbose=True
)
glb.export("new_barn_optimized_enhanced.glb")

# Enhanced PLY export
print("Optimizing and saving Gaussians...")
simplified_gs = postprocessing_utils.simplify_gs(
    outputs['gaussian'][0],
    simplify=0.98,  # High quality simplification
    verbose=True
)
simplified_gs.save_ply("new_barn_optimized.ply")

print("\nOptimization complete! Generated files:")
print("- new_barn_optimized_enhanced.glb (2K textures, 98% mesh quality)")
print("- new_barn_optimized.ply (optimized Gaussians)")
print("- new_barn_optimized_gs.mp4 (1024x1024 Gaussian video)")
print("- new_barn_optimized_rf.mp4 (1024x1024 Radiance Field video)")
print("- new_barn_optimized_mesh.mp4 (1024x1024 Mesh video)")

print("\nQuality improvements applied:")
print("- Enhanced text preprocessing with 3D-specific keywords")
print("- Increased sampling steps (25 → 35)")
print("- Higher CFG strength (7.5 → 8.5)")
print("- Extended guidance interval for better quality")
print("- Higher resolution rendering (1024x1024)")
print("- Enhanced GLB with 2K textures and minimal simplification")
print("- Advanced Gaussian optimization") 