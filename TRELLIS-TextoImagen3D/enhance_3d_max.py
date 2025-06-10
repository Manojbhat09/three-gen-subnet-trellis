'''
python enhance_3d_max.py outputs/wbgmsst,_a_blue_monkey_sitting_on_temple,_3d_isometric,_white_background.glb --output-dir enhanced_outputs

python enhance_3d_max.py outputs/wbgmsst,_a_blue_monkey_sitting_on_temple,_3d_isometric,_white_background.glb \
    --output-dir enhanced_outputs \
    --octree-depth 8 \
    --num-inference-steps 15 \
    --guidance-scale 5.0

python enhance_3d_max.py outputs/wbgmsst,_a_blue_monkey_sitting_on_temple,_3d_isometric,_white_background.glb \
    --output-dir enhanced_outputs \
    --octree-depth 9 \
    --num-inference-steps 50 \
    --guidance-scale 10.0
'''

import os
import sys
import uuid
import time
import argparse
from pathlib import Path

# Add TRELLIS to Python path
TRELLIS_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "trellis")
sys.path.append(TRELLIS_PATH)

# Add DetailGen3D to Python path
DETAILGEN3D_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "DetailGen3D")
sys.path.append(DETAILGEN3D_PATH)

# Set environment variables to use xformers instead of flash attention
os.environ['SPCONV_ALGO'] = 'native'
os.environ['ATTN_BACKEND'] = 'xformers'
os.environ['XFORMERS_DISABLE_FLASH_ATTN'] = '1'  # Disable flash attention

import imageio
import torch
import numpy as np
from PIL import Image
import trimesh
from skimage import measure
from trellis.utils import render_utils
from trellis.pipelines import TrellisTextTo3DPipeline
from detailgen3d.pipelines import DetailGen3DPipeline
from detailgen3d.inference_utils import generate_dense_grid_points

def load_mesh(mesh_path, num_pc=20480, dtype=torch.float16):
    """Load and normalize mesh for DetailGen3D."""
    mesh = trimesh.load(mesh_path, force="mesh")
    
    # Center and scale the mesh
    center = mesh.bounding_box.centroid
    mesh.apply_translation(-center)
    scale = max(mesh.bounding_box.extents)
    mesh.apply_scale(1.9 / scale)
    
    # Sample points from surface
    surface, face_indices = trimesh.sample.sample_surface(mesh, 1000000)
    normal = mesh.face_normals[face_indices]
    
    # Randomly select points
    rng = np.random.default_rng()
    ind = rng.choice(surface.shape[0], num_pc, replace=False)
    surface = torch.FloatTensor(surface[ind]).to(dtype=dtype)
    normal = torch.FloatTensor(normal[ind]).to(dtype=dtype)
    surface = torch.cat([surface, normal], dim=-1).unsqueeze(0).cuda()
    
    return surface

def enhance_3d_model(input_glb_path, output_dir="outputs", octree_depth=9, num_inference_steps=50, guidance_scale=10.0):
    """
    Enhance a 3D model using DetailGen3D.
    
    Args:
        input_glb_path (str): Path to input GLB file
        output_dir (str): Directory to save the outputs
        octree_depth (int): Depth of octree for point sampling
        num_inference_steps (int): Number of inference steps for DetailGen3D
        guidance_scale (float): Guidance scale for DetailGen3D
    """
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    # Generate a unique ID for this run
    run_id = str(uuid.uuid4())
    run_dir = output_path / run_id
    run_dir.mkdir(exist_ok=True)
    
    # Initialize DetailGen3D pipeline
    print("Loading DetailGen3D pipeline...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16
    detail_pipeline = DetailGen3DPipeline.from_pretrained(
        "/home/mbhat/three-gen-subnet-trellis/detailgen3d",
        trust_remote_code=True,
        local_files_only=True
    ).to(device, dtype=dtype)
    
    # Load and normalize the input mesh
    print(f"Loading and normalizing input model: {input_glb_path}")
    surface = load_mesh(input_glb_path, dtype=dtype)
    
    # Initialize TRELLIS pipeline for rendering
    print("Loading TRELLIS pipeline for rendering...")
    trellis_pipeline = TrellisTextTo3DPipeline.from_pretrained("microsoft/TRELLIS-text-xlarge")
    trellis_pipeline.cuda()
    
    # Generate a temporary model with TRELLIS for rendering
    print("Generating temporary model for rendering...")
    trellis_outputs = trellis_pipeline.run(
        "a blue monkey sitting on temple",  # Use the same prompt as the input model
        seed=42,
        sparse_structure_sampler_params={
            "steps": 12,
            "cfg_strength": 7.5,
        },
        slat_sampler_params={
            "steps": 12,
            "cfg_strength": 7.5,
        },
    )
    
    # Render an image of the mesh for DetailGen3D
    print("Rendering reference image...")
    resolution = 512
    yaw = 0.0  # Front view
    pitch = 0.25  # Slightly above
    r = 2.0  # Distance
    fov = 40.0  # Field of view
    
    # Create camera matrices (using float32 for TRELLIS renderer)
    extrinsics, _ = render_utils.yaw_pitch_r_fov_to_extrinsics_intrinsics(yaw, pitch, r, fov)
    
    # Create intrinsics matrix in OpenCV format (using float32 for TRELLIS renderer)
    f = resolution / (2 * np.tan(np.radians(fov) / 2))
    cx = resolution / 2
    cy = resolution / 2
    intrinsics = torch.tensor([
        [f, 0, cx],
        [0, f, cy],
        [0, 0, 1]
    ], dtype=torch.float32, device=device)  # Use float32 for TRELLIS renderer
    
    # Convert to lists as expected by render_frames
    extrinsics = [extrinsics]
    intrinsics = [intrinsics]
    
    # Render using TRELLIS's gaussian output
    rendered_frames = render_utils.render_frames(
        trellis_outputs['gaussian'][0],
        extrinsics,
        intrinsics,
        {
            'resolution': resolution,
            'bg_color': (1, 1, 1),  # White background
            'use_white_background': True,  # Enable white background
            'use_white_light': True,  # Use white light
            'use_ambient_light': True,  # Enable ambient light
            'ambient_light_strength': 0.5,  # Ambient light strength
            'use_directional_light': True,  # Enable directional light
            'directional_light_strength': 0.5,  # Directional light strength
            'directional_light_direction': (0, 0, -1),  # Light direction
        }
    )
    rendered_image = rendered_frames['color'][0]
    
    # Save the reference image
    image_path = run_dir / "reference_image.png"
    imageio.imwrite(str(image_path), rendered_image)
    
    # Convert rendered image to PIL Image for DetailGen3D
    pil_image = Image.fromarray(rendered_image)
    
    # Generate dense grid points for decoding
    print("\nGenerating grid points for decoding...")
    start_time = time.time()
    box_min = np.array([-1.005, -1.005, -1.005])
    box_max = np.array([1.005, 1.005, 1.005])
    sampled_points, grid_size, bbox_size = generate_dense_grid_points(
        bbox_min=box_min,
        bbox_max=box_max,
        octree_depth=octree_depth,
        indexing="ij"
    )
    print(f"Grid size: {grid_size}, Number of points: {sampled_points.shape[0]}")
    sampled_points = torch.FloatTensor(sampled_points).to(device, dtype=dtype)  # Use float16 for DetailGen3D
    sampled_points = sampled_points.unsqueeze(0)
    print(f"Grid generation time: {time.time() - start_time:.2f} seconds")
    
    # Run DetailGen3D
    print("\nRunning DetailGen3D pipeline...")
    start_time = time.time()
    sdf = detail_pipeline(
        image=pil_image,  # Provide the rendered image
        latents=detail_pipeline.vae.encode(surface).latent_dist.sample(),
        sampled_points=sampled_points,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        noise_aug_level=0,
        generator=torch.Generator(device=device).manual_seed(42)
    ).samples[0]
    print(f"Pipeline execution time: {time.time() - start_time:.2f} seconds")
    
    # Convert SDF to mesh using marching cubes
    print("\nConverting SDF to mesh...")
    start_time = time.time()
    grid_logits = sdf.view(grid_size).cpu().numpy()
    vertices, faces, normals, _ = measure.marching_cubes(
        grid_logits, 0, method="lewiner"
    )
    vertices = vertices / grid_size * bbox_size + box_min
    enhanced_mesh = trimesh.Trimesh(vertices.astype(np.float32), np.ascontiguousarray(faces))
    print(f"Mesh conversion time: {time.time() - start_time:.2f} seconds")
    print(f"Generated mesh: {len(vertices)} vertices, {len(faces)} faces")
    
    # Save enhanced PLY
    enhanced_ply_path = run_dir / "enhanced_model.ply"
    enhanced_mesh.export(str(enhanced_ply_path))
    
    print("\nEnhancement complete! Files saved in", run_dir)
    print("- enhanced_model.ply (DetailGen3D enhanced model)")
    print("- reference_image.png (Rendered reference image)")

def main():
    parser = argparse.ArgumentParser(description="Enhance 3D models using DetailGen3D")
    parser.add_argument("input_glb", type=str, help="Path to input GLB file")
    parser.add_argument("--output-dir", type=str, default="outputs", help="Directory to save the outputs")
    parser.add_argument("--octree-depth", type=int, default=7, help="Depth of octree for point sampling")
    parser.add_argument("--num-inference-steps", type=int, default=10, help="Number of inference steps")
    parser.add_argument("--guidance-scale", type=float, default=4.0, help="Guidance scale")
    
    args = parser.parse_args()
    
    enhance_3d_model(
        args.input_glb,
        args.output_dir,
        args.octree_depth,
        args.num_inference_steps,
        args.guidance_scale
    )

if __name__ == "__main__":
    main() 