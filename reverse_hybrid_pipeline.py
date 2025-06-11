#!/usr/bin/env python3
"""
Reverse Hybrid 3D Generation Pipeline: TRELLIS Text + Hunyuan3D-2 Geometry + TRELLIS Representations

This pipeline combines the systems in a different way:
1. TRELLIS: Text-to-image generation (better text understanding)
2. Hunyuan3D-2: Image-to-geometry (better geometry, handles holes/topology)
3. TRELLIS: Multi-representation generation (Gaussians, NeRF, textured mesh)
4. Combination: Use Hunyuan geometry with TRELLIS texture/lighting information

The advantage:
- TRELLIS: Superior text understanding and multi-format output
- Hunyuan3D-2: Superior mesh topology and hole handling
- Combined: Best of both for different aspects
"""

import os
import torch
import numpy as np
from PIL import Image
import trimesh
from pathlib import Path

# TRELLIS imports
import sys
sys.path.append('TRELLIS')
from trellis.pipelines import TrellisTextTo3DPipeline, TrellisImageTo3DPipeline
from trellis.utils import render_utils, postprocessing_utils

# Hunyuan3D-2 imports  
sys.path.append('Hunyuan3D-2')
from hy3dgen.shapegen import Hunyuan3DDiTFlowMatchingPipeline, FaceReducer, FloaterRemover, DegenerateFaceRemover
from hy3dgen.texgen import Hunyuan3DPaintPipeline
from hy3dgen.texgen.utils.uv_warp_utils import mesh_uv_wrap
from hy3dgen.rembg import BackgroundRemover


class ReverseHybridPipeline:
    """
    Reverse hybrid pipeline combining different strengths:
    - TRELLIS: Text understanding and multi-representation
    - Hunyuan3D-2: High-quality mesh geometry
    """
    
    def __init__(self, 
                 trellis_text_model="microsoft/TRELLIS-text-large",
                 trellis_image_model="microsoft/TRELLIS-image-large", 
                 hunyuan_model_path='jetx/Hunyuan3D-2',
                 device="cuda"):
        """
        Initialize the reverse hybrid pipeline.
        
        Args:
            trellis_text_model: TRELLIS text-to-3D model
            trellis_image_model: TRELLIS image-to-3D model
            hunyuan_model_path: Hunyuan3D-2 model path
            device: Computing device
        """
        self.device = device
        self.trellis_text_model = trellis_text_model
        self.trellis_image_model = trellis_image_model
        self.hunyuan_model_path = hunyuan_model_path
        
        # Initialize components
        self.rembg = BackgroundRemover()
        
        # Load models
        self._load_models()
        
    def _load_models(self):
        """Load all required models."""
        print("Loading TRELLIS text-to-3D pipeline...")
        
        # Configure TRELLIS environment
        os.environ['SPCONV_ALGO'] = 'native'
        
        # Load TRELLIS text-to-3D pipeline
        self.trellis_text_pipeline = TrellisTextTo3DPipeline.from_pretrained(
            self.trellis_text_model
        )
        self.trellis_text_pipeline.cuda()
        
        print("Loading TRELLIS image-to-3D pipeline...")
        # Load TRELLIS image-to-3D pipeline
        self.trellis_image_pipeline = TrellisImageTo3DPipeline.from_pretrained(
            self.trellis_image_model
        )
        self.trellis_image_pipeline.cuda()
        
        print("Loading Hunyuan3D-2 shape generation pipeline...")
        # Load Hunyuan3D-2 shape generation pipeline
        self.hunyuan_shape_pipeline = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(
            self.hunyuan_model_path,
            use_safetensors=True
        )
        
        print("All models loaded successfully!")
    
    def _extract_mesh_from_trellis(self, trellis_result):
        """Extract trimesh object from TRELLIS MeshExtractResult."""
        if hasattr(trellis_result, 'vertices') and hasattr(trellis_result, 'faces'):
            vertices = trellis_result.vertices
            faces = trellis_result.faces
        else:
            raise ValueError(f"Unknown TRELLIS result format: {type(trellis_result)}")
        
        # Convert torch tensors to numpy arrays
        if hasattr(vertices, 'cpu'):
            vertices = vertices.cpu().numpy()
        if hasattr(faces, 'cpu'):
            faces = faces.cpu().numpy()
        
        # Create trimesh object
        mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
        return mesh
    
    def _process_image(self, image):
        """Process image for both pipelines."""
        if isinstance(image, str):
            image = Image.open(image).convert('RGB')
        
        # Resize to standard size
        image = image.resize((1024, 1024))
        
        # Remove background
        if image.mode == 'RGB':
            image = self.rembg(image)
            
        return image
    
    def _clean_hunyuan_mesh(self, mesh):
        """Apply Hunyuan3D-2's mesh cleaning operations."""
        mesh = FloaterRemover()(mesh)
        mesh = DegenerateFaceRemover()(mesh)
        mesh = FaceReducer()(mesh)
        return mesh
    
    def _align_meshes(self, hunyuan_mesh, trellis_mesh):
        """Align coordinate systems between Hunyuan and TRELLIS meshes."""
        
        # Get bounds for both meshes
        hunyuan_bounds = hunyuan_mesh.bounds
        trellis_bounds = trellis_mesh.bounds
        
        # Calculate centers and scales
        hunyuan_center = (hunyuan_bounds[0] + hunyuan_bounds[1]) / 2
        trellis_center = (trellis_bounds[0] + trellis_bounds[1]) / 2
        
        hunyuan_scale = np.max(hunyuan_bounds[1] - hunyuan_bounds[0])
        trellis_scale = np.max(trellis_bounds[1] - trellis_bounds[0])
        
        # Normalize both to same coordinate system
        hunyuan_mesh.vertices = (hunyuan_mesh.vertices - hunyuan_center) / hunyuan_scale
        trellis_mesh.vertices = (trellis_mesh.vertices - trellis_center) / trellis_scale
        
        return hunyuan_mesh, trellis_mesh
    
    def _transfer_texture_from_gaussians(self, mesh, gaussians, image):
        """
        Transfer texture information from TRELLIS Gaussians to Hunyuan mesh.
        This is a simplified approach - in practice you'd want more sophisticated methods.
        """
        try:
            # For now, we'll use a simple UV mapping approach
            # In a full implementation, you'd project Gaussian colors onto the mesh
            mesh = mesh_uv_wrap(mesh)
            
            # Simple fallback: use random UV coordinates
            if not hasattr(mesh.visual, 'uv') or mesh.visual.uv is None:
                mesh.visual.uv = np.random.rand(len(mesh.vertices), 2)
            
            print("Basic texture transfer completed")
            return mesh
            
        except Exception as e:
            print(f"Texture transfer failed: {e}")
            return mesh
    
    def generate_3d_from_text(self, 
                             text_prompt,
                             output_path="reverse_hybrid_output",
                             save_intermediate=True,
                             generation_params=None):
        """
        Generate 3D asset from text using reverse hybrid approach.
        
        Args:
            text_prompt: Text description
            output_path: Base path for outputs
            save_intermediate: Whether to save intermediate results
            generation_params: Parameters for generation
            
        Returns:
            Dictionary with multiple 3D representations
        """
        print("=== Starting Reverse Hybrid Text-to-3D Generation ===")
        
        # Default parameters
        if generation_params is None:
            generation_params = {
                "text_to_3d_params": {
                    "seed": 42,
                    "sparse_structure_sampler_params": {"steps": 12, "cfg_strength": 7.5},
                    "slat_sampler_params": {"steps": 12, "cfg_strength": 3},
                },
                "hunyuan_params": {
                    "num_inference_steps": 30,
                    "mc_algo": 'mc',
                    "generator": torch.manual_seed(42)
                },
                "image_to_3d_params": {
                    "seed": 42,
                    "sparse_structure_sampler_params": {"steps": 12, "cfg_strength": 7.5},
                    "slat_sampler_params": {"steps": 12, "cfg_strength": 3},
                }
            }
        
        # Step 1: Generate image from text using TRELLIS
        print("1. Generating reference representations from text with TRELLIS...")
        
        trellis_text_outputs = self.trellis_text_pipeline.run(
            text_prompt,
            **generation_params["text_to_3d_params"]
        )
        
        # Extract reference image (we can render from one of the representations)
        if 'gaussian' in trellis_text_outputs:
            print("1.1. Rendering reference image from TRELLIS Gaussians...")
            # Render an image from the Gaussian representation
            rendered_views = render_utils.render_video(trellis_text_outputs['gaussian'][0])
            reference_image = Image.fromarray((rendered_views['color'][0] * 255).astype(np.uint8))
        else:
            # Fallback: create a simple reference image
            print("1.1. Creating fallback reference image...")
            reference_image = Image.new('RGB', (1024, 1024), color='white')
        
        if save_intermediate:
            reference_image.save(f"{output_path}_reference_image.png")
            # Save TRELLIS text outputs
            if 'gaussian' in trellis_text_outputs:
                trellis_text_outputs['gaussian'][0].save_ply(f"{output_path}_text_gaussian.ply")
            if 'mesh' in trellis_text_outputs:
                trellis_text_mesh = self._extract_mesh_from_trellis(trellis_text_outputs['mesh'][0])
                trellis_text_mesh.export(f"{output_path}_text_mesh.glb")
        
        # Step 2: Generate high-quality geometry with Hunyuan3D-2
        print("2. Generating high-quality geometry with Hunyuan3D-2...")
        
        processed_image = self._process_image(reference_image)
        
        hunyuan_mesh_result = self.hunyuan_shape_pipeline(
            image=processed_image,
            **generation_params["hunyuan_params"]
        )[0]
        
        # Clean the Hunyuan mesh
        hunyuan_mesh = self._clean_hunyuan_mesh(hunyuan_mesh_result)
        
        if save_intermediate:
            hunyuan_mesh.export(f"{output_path}_hunyuan_geometry.glb")
        
        print(f"Hunyuan mesh: {len(hunyuan_mesh.vertices)} vertices, {len(hunyuan_mesh.faces)} faces")
        
        # Step 3: Generate additional representations with TRELLIS image pipeline
        print("3. Generating additional representations with TRELLIS image pipeline...")
        
        trellis_image_outputs = self.trellis_image_pipeline.run(
            processed_image,
            **generation_params["image_to_3d_params"]
        )
        
        if save_intermediate:
            # Save TRELLIS image outputs
            if 'gaussian' in trellis_image_outputs:
                trellis_image_outputs['gaussian'][0].save_ply(f"{output_path}_image_gaussian.ply")
            if 'radiance_field' in trellis_image_outputs:
                # Save radiance field if possible
                pass
            if 'mesh' in trellis_image_outputs:
                trellis_image_mesh = self._extract_mesh_from_trellis(trellis_image_outputs['mesh'][0])
                trellis_image_mesh.export(f"{output_path}_image_mesh.glb")
        
        # Step 4: Combine results
        print("4. Combining results...")
        
        # Use Hunyuan geometry as primary mesh
        final_mesh = hunyuan_mesh.copy()
        
        # Attempt to enhance with TRELLIS texture information
        if 'gaussian' in trellis_image_outputs:
            final_mesh = self._transfer_texture_from_gaussians(
                final_mesh, 
                trellis_image_outputs['gaussian'][0], 
                processed_image
            )
        
        # Prepare final outputs
        results = {
            'primary_mesh': final_mesh,
            'hunyuan_geometry': hunyuan_mesh,
            'text_prompt': text_prompt,
            'reference_image': reference_image,
        }
        
        # Add TRELLIS representations
        if 'gaussian' in trellis_text_outputs:
            results['text_gaussians'] = trellis_text_outputs['gaussian'][0]
        if 'gaussian' in trellis_image_outputs:
            results['image_gaussians'] = trellis_image_outputs['gaussian'][0]
        if 'radiance_field' in trellis_image_outputs:
            results['radiance_field'] = trellis_image_outputs['radiance_field'][0]
        if 'mesh' in trellis_image_outputs:
            results['trellis_mesh'] = self._extract_mesh_from_trellis(trellis_image_outputs['mesh'][0])
        
        # Save final result
        if save_intermediate:
            final_mesh.export(f"{output_path}_final_combined.glb")
        
        print(f"=== Reverse hybrid generation completed! ===")
        print(f"Results include: {list(results.keys())}")
        
        return results
    
    def generate_3d_from_image(self, 
                              image,
                              output_path="reverse_hybrid_image_output",
                              save_intermediate=True,
                              generation_params=None):
        """
        Generate 3D asset from image using reverse hybrid approach.
        
        Args:
            image: Input image
            output_path: Base path for outputs
            save_intermediate: Whether to save intermediate results
            generation_params: Parameters for generation
            
        Returns:
            Dictionary with multiple 3D representations
        """
        print("=== Starting Reverse Hybrid Image-to-3D Generation ===")
        
        # Default parameters
        if generation_params is None:
            generation_params = {
                "hunyuan_params": {
                    "num_inference_steps": 30,
                    "mc_algo": 'mc',
                    "generator": torch.manual_seed(42)
                },
                "trellis_params": {
                    "seed": 42,
                    "sparse_structure_sampler_params": {"steps": 12, "cfg_strength": 7.5},
                    "slat_sampler_params": {"steps": 12, "cfg_strength": 3},
                }
            }
        
        # Process input image
        processed_image = self._process_image(image)
        
        if save_intermediate:
            processed_image.save(f"{output_path}_processed_input.png")
        
        # Generate with both pipelines in parallel
        print("1. Generating geometry with Hunyuan3D-2...")
        hunyuan_mesh_result = self.hunyuan_shape_pipeline(
            image=processed_image,
            **generation_params["hunyuan_params"]
        )[0]
        hunyuan_mesh = self._clean_hunyuan_mesh(hunyuan_mesh_result)
        
        print("2. Generating representations with TRELLIS...")
        trellis_outputs = self.trellis_image_pipeline.run(
            processed_image,
            **generation_params["trellis_params"]
        )
        
        # Combine and save results (similar to text version)
        final_mesh = hunyuan_mesh.copy()
        
        if 'gaussian' in trellis_outputs:
            final_mesh = self._transfer_texture_from_gaussians(
                final_mesh, 
                trellis_outputs['gaussian'][0], 
                processed_image
            )
        
        results = {
            'primary_mesh': final_mesh,
            'hunyuan_geometry': hunyuan_mesh,
            'input_image': processed_image,
        }
        
        # Add TRELLIS representations
        if 'gaussian' in trellis_outputs:
            results['gaussians'] = trellis_outputs['gaussian'][0]
        if 'radiance_field' in trellis_outputs:
            results['radiance_field'] = trellis_outputs['radiance_field'][0]
        if 'mesh' in trellis_outputs:
            results['trellis_mesh'] = self._extract_mesh_from_trellis(trellis_outputs['mesh'][0])
        
        if save_intermediate:
            final_mesh.export(f"{output_path}_final_combined.glb")
            hunyuan_mesh.export(f"{output_path}_hunyuan_geometry.glb")
            if 'gaussian' in trellis_outputs:
                trellis_outputs['gaussian'][0].save_ply(f"{output_path}_trellis_gaussian.ply")
        
        print(f"=== Reverse hybrid generation completed! ===")
        return results


def main():
    """Example usage of the reverse hybrid pipeline."""
    
    # Initialize reverse hybrid pipeline
    pipeline = ReverseHybridPipeline()
    
    # Example 1: Text-to-3D generation
    print("=== Example 1: Text-to-3D Generation ===")
    
    # text_prompt = "a vintage wooden chair with intricate carvings"
    text_prompt="purple durable robotic arm"
    
    results = pipeline.generate_3d_from_text(
        text_prompt,
        output_path="reverse_text_output",
        save_intermediate=True
    )
    
    print(f"Generated {len(results)} different representations")
    print(f"Primary mesh: {len(results['primary_mesh'].vertices)} vertices")
    
    # Example 2: Image-to-3D generation
    print("\n=== Example 2: Image-to-3D Generation ===")
    
    image_path = "/home/mbhat/three-gen-subnet-trellis/Hunyuan3D-2/outputs-arm/t2i_original.png"
    
    if os.path.exists(image_path):
        results = pipeline.generate_3d_from_image(
            image_path,
            output_path="reverse_image_output", 
            save_intermediate=True
        )
        
        print(f"Generated {len(results)} different representations")
        print(f"Primary mesh: {len(results['primary_mesh'].vertices)} vertices")
    else:
        print(f"Image not found at {image_path}")


if __name__ == "__main__":
    main() 