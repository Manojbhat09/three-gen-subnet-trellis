import os
import sys
import uuid
import time
from pathlib import Path

# Add TRELLIS to Python path
TRELLIS_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "TRELLIS")
sys.path.append(TRELLIS_PATH)

# Add DetailGen3D to Python path
DETAILGEN3D_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "detailgen3d")
sys.path.append(os.path.dirname(DETAILGEN3D_PATH))  # Add parent directory to path

# Set spconv algorithm to native for single run and use xformers for attention
os.environ['SPCONV_ALGO'] = 'native'
os.environ['ATTN_BACKEND'] = 'xformers'  # Use xformers instead of flash-attn

import imageio
import torch
import numpy as np
from trellis.pipelines import TrellisTextTo3DPipeline
from trellis.utils import render_utils, postprocessing_utils
from DetailGen3D.detailgen3d.pipelines import DetailGen3DPipeline
from DetailGen3D.detailgen3d.inference_utils import generate_dense_grid_points
from PIL import Image
import trimesh
from skimage import measure

def main():
    # Create output directory
    output_dir = "outputs"
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate a unique ID for this run
    run_id = str(uuid.uuid4())
    run_dir = os.path.join(output_dir, run_id)
    os.makedirs(run_dir, exist_ok=True)
    
    # Initialize the TRELLIS pipeline
    print("Loading TRELLIS pipeline...")
    trellis_pipeline = TrellisTextTo3DPipeline.from_pretrained("microsoft/TRELLIS-text-xlarge")
    trellis_pipeline.cuda()
    
    # Initialize DetailGen3D pipeline
    print("Loading DetailGen3D pipeline...")
    detail_pipeline = DetailGen3DPipeline.from_pretrained(
        "/home/mbhat/three-gen-subnet-trellis/detailgen3d",
        trust_remote_code=True,
        local_files_only=True
    )
    
    # Move neural network components to CUDA
    device = torch.device("cuda")
    # Move VAE model
    detail_pipeline.vae = detail_pipeline.vae.to(device)
    # Move transformer model
    detail_pipeline.transformer = detail_pipeline.transformer.to(device)
    # Move image encoder
    detail_pipeline.image_encoder_1 = detail_pipeline.image_encoder_1.to(device)
    # Feature extractor is a preprocessing component, no need to move to CUDA
    
    # Set your text prompt
    prompt = "a blue monkey sitting on temple"
    print(f"Generating 3D model from prompt: {prompt}")
    
    # Generate the 3D model with TRELLIS
    print("Generating 3D model with TRELLIS...")
    trellis_outputs = trellis_pipeline.run(
        prompt,
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
    
    # Save TRELLIS outputs
    print("Saving TRELLIS outputs...")
    
    # Save preview videos
    video = render_utils.render_video(trellis_outputs['gaussian'][0])['color']
    imageio.mimsave(os.path.join(run_dir, "preview_gaussian.mp4"), video, fps=30)
    
    video = render_utils.render_video(trellis_outputs['radiance_field'][0])['color']
    imageio.mimsave(os.path.join(run_dir, "preview_rf.mp4"), video, fps=30)
    
    video = render_utils.render_video(trellis_outputs['mesh'][0])['normal']
    imageio.mimsave(os.path.join(run_dir, "preview_mesh.mp4"), video, fps=30)
    
    # Save initial PLY file
    initial_ply_path = os.path.join(run_dir, "initial_model.ply")
    trellis_outputs['gaussian'][0].save_ply(initial_ply_path)
    
    # Save initial GLB file
    initial_glb = postprocessing_utils.to_glb(
        trellis_outputs['gaussian'][0],
        trellis_outputs['mesh'][0],
        simplify=0.95,
        texture_size=1024,
    )
    initial_glb_path = os.path.join(run_dir, "initial_model.glb")
    initial_glb.export(initial_glb_path)
    
    # Process with DetailGen3D
    print("Enhancing details with DetailGen3D...")
    # Convert GLB to mesh for DetailGen3D
    scene = trimesh.load(initial_glb_path)
    
    # Extract the first mesh from the scene
    if isinstance(scene, trimesh.Scene):
        # Get the first mesh from the scene
        mesh = list(scene.geometry.values())[0]
    else:
        mesh = scene
    
    # Sample points from the mesh surface
    num_points = 2048  # Default number of points expected by DetailGen3D
    points, face_indices = trimesh.sample.sample_surface(mesh, num_points)
    
    # Get face normals for the sampled points
    normals = mesh.face_normals[face_indices]
    
    # Convert points and normals to tensor format expected by DetailGen3D
    # Shape: (1, N, 6) where N is number of points and 6 is xyz + normal xyz
    points = torch.tensor(points, dtype=torch.float32, device='cuda')
    normals = torch.tensor(normals, dtype=torch.float32, device='cuda')
    points_with_normals = torch.cat([points, normals], dim=-1)
    points_with_normals = points_with_normals.unsqueeze(0)  # Add batch dimension
    
    # Normalize points to [-1, 1] range
    points_with_normals = points_with_normals / torch.max(torch.abs(points_with_normals))
    
    # Encode the points through the VAE
    with torch.no_grad():
        latents = detail_pipeline.vae.encode(points_with_normals).latent_dist.sample()
    
    # Convert mesh to tensor format expected by DetailGen3D
    # Get faces
    faces = torch.tensor(mesh.faces, dtype=torch.long, device='cuda')
    
    # Create a dictionary with the mesh data
    mesh_data = {
        'vertices': points,
        'faces': faces
    }
    
    # Get a rendered image for DetailGen3D using render_frames
    # Render a single frame from a fixed viewpoint
    yaw = 0.0  # Front view
    pitch = 0.25  # Slightly above
    r = 2.0  # Distance
    fov = 40.0  # Field of view
    
    # Create camera matrices using TRELLIS's utility function
    extrinsics, _ = render_utils.yaw_pitch_r_fov_to_extrinsics_intrinsics(yaw, pitch, r, fov)
    
    # Create intrinsics matrix in OpenCV format
    resolution = 512
    f = resolution / (2 * np.tan(np.radians(fov) / 2))
    cx = resolution / 2
    cy = resolution / 2
    
    # Create 3x3 intrinsics matrix in OpenCV format
    intrinsics = torch.tensor([
        [f, 0, cx],
        [0, f, cy],
        [0, 0, 1]
    ], dtype=torch.float32, device='cuda')
    
    # Debug prints
    print("Extrinsics shape:", extrinsics.shape)
    print("Intrinsics shape:", intrinsics.shape)
    print("Intrinsics:", intrinsics)
    
    # Convert to lists as expected by render_frames
    extrinsics = [extrinsics]
    intrinsics = [intrinsics]
    
    rendered_frames = render_utils.render_frames(
        trellis_outputs['gaussian'][0],
        extrinsics,
        intrinsics,
        {'resolution': resolution, 'bg_color': (0, 0, 0)}
    )
    rendered_image = rendered_frames['color'][0]  # Get the first (and only) frame
    
    # Save the reference image
    image_path = os.path.join(run_dir, "reference_image.png")
    imageio.imwrite(image_path, rendered_image)
    
    # Convert rendered image to PIL Image for DetailGen3D
    pil_image = Image.fromarray(rendered_image)
    
    # Generate dense grid points for decoding with lower resolution
    print("\nGenerating grid points for decoding...")
    start_time = time.time()
    box_min = np.array([-1.005, -1.005, -1.005])
    box_max = np.array([1.005, 1.005, 1.005])
    sampled_points, grid_size, bbox_size = generate_dense_grid_points(
        bbox_min=box_min, bbox_max=box_max, octree_depth=7,  # Reduced from 9 to 7 for faster processing
        indexing="ij"
    )
    print(f"Grid size: {grid_size}, Number of points: {sampled_points.shape[0]}")
    sampled_points = torch.FloatTensor(sampled_points).to(device, dtype=torch.float32)
    sampled_points = sampled_points.unsqueeze(0)  # Add batch dimension
    print(f"Grid generation time: {time.time() - start_time:.2f} seconds")
    
    # Run DetailGen3D
    print("\nRunning DetailGen3D pipeline...")
    start_time = time.time()
    sdf = detail_pipeline(
        image=pil_image,  # Pass PIL Image instead of file path
        latents=latents,  # Pass encoded latents
        sampled_points=sampled_points,  # Pass dense grid points for decoding
        num_inference_steps=10,
        guidance_scale=4.0,
        noise_aug_level=0
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
    enhanced_ply_path = os.path.join(run_dir, "enhanced_model.ply")
    enhanced_mesh.export(enhanced_ply_path)
    
    print("\nGeneration complete! Files saved in", run_dir)
    print("- initial_model.ply (Initial TRELLIS output)")
    print("- initial_model.glb (Initial textured mesh)")
    print("- enhanced_model.ply (DetailGen3D enhanced model)")
    print("- preview_gaussian.mp4 (Gaussian splatting preview)")
    print("- preview_rf.mp4 (Radiance field preview)")
    print("- preview_mesh.mp4 (Mesh preview)")
    print("- reference_image.png (Reference image for detail enhancement)")

if __name__ == "__main__":
    main() 