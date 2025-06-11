#!/usr/bin/env python3
"""
Hybrid 3D Generation Pipeline: TRELLIS SLAT + Hunyuan3D-2 Texture

This script combines the best of both worlds:
- TRELLIS: Advanced SLAT (Structured Latents) for geometry generation
- Hunyuan3D-2: Superior texture synthesis capabilities

The pipeline:
1. Use TRELLIS to generate high-quality mesh geometry from SLAT representations
2. Extract the mesh and prepare it for texture processing
3. Use Hunyuan3D-2's texture pipeline to add high-quality textures
4. Output the final textured mesh

Requirements:
- Both TRELLIS and Hunyuan3D-2 environments set up
- Compatible mesh format handling
- Proper coordinate system alignment
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
from trellis.pipelines import TrellisImageTo3DPipeline
from trellis.utils import render_utils, postprocessing_utils

# Hunyuan3D-2 imports  
sys.path.append('Hunyuan3D-2')
from hy3dgen.texgen import Hunyuan3DPaintPipeline
from hy3dgen.texgen.utils.uv_warp_utils import mesh_uv_wrap
from hy3dgen.rembg import BackgroundRemover


class HybridTrellisHunyuanPipeline:
    """
    Hybrid pipeline combining TRELLIS geometry generation with Hunyuan3D-2 texture synthesis.
    """
    
    def __init__(self, 
                 trellis_model_path="microsoft/TRELLIS-image-large",
                 hunyuan_model_path='jetx/Hunyuan3D-2', #"tencent/Hunyuan3D-2",
                 device="cuda"):
        """
        Initialize the hybrid pipeline.
        
        Args:
            trellis_model_path: Path to TRELLIS model
            hunyuan_model_path: Path to Hunyuan3D-2 model
            device: Computing device
        """
        self.device = device
        self.trellis_model_path = trellis_model_path
        self.hunyuan_model_path = hunyuan_model_path
        
        # Initialize background remover
        self.rembg = BackgroundRemover()
        
        # Load models
        self._load_models()
        
    def _load_models(self):
        """Load both TRELLIS and Hunyuan3D-2 models."""
        print("Loading TRELLIS pipeline...")
        
        # Configure TRELLIS environment
        os.environ['SPCONV_ALGO'] = 'native'
        
        # Load TRELLIS pipeline
        self.trellis_pipeline = TrellisImageTo3DPipeline.from_pretrained(
            self.trellis_model_path
        )
        self.trellis_pipeline.cuda()
        
        print("Loading Hunyuan3D-2 texture pipeline...")
        
        # Load Hunyuan3D-2 texture pipeline
        try:
            self.texture_pipeline = Hunyuan3DPaintPipeline.from_pretrained(
                self.hunyuan_model_path,
                subfolder="hunyuan3d-paint-v2-0",
                use_safetensors=True
            )
        except Exception as e:
            print(f"Warning: Failed to load with subfolder, trying direct load: {e}")
            self.texture_pipeline = Hunyuan3DPaintPipeline.from_pretrained(
                self.hunyuan_model_path
            )
        
        print("Models loaded successfully!")
    
    def _preprocess_image(self, image):
        """
        Preprocess input image for both pipelines.
        
        Args:
            image: PIL Image or path string
            
        Returns:
            Processed PIL Image
        """
        if isinstance(image, str):
            image = Image.open(image).convert('RGB')
        
        # Resize to consistent size
        image = image.resize((1024, 1024))
        
        # Remove background if needed
        if image.mode == 'RGB':
            image = self.rembg(image)
            
        return image
    
    def _align_mesh_coordinates(self, mesh):
        """
        Align coordinate systems between TRELLIS and Hunyuan3D-2.
        
        Args:
            mesh: Trimesh object from TRELLIS
            
        Returns:
            Aligned trimesh object
        """
        try:
            # TRELLIS and Hunyuan3D-2 might use different coordinate conventions
            # This function handles any necessary transformations
            
            # Basic mesh validation and cleanup
            if not mesh.is_watertight:
                print("Warning: Mesh is not watertight, attempting to fix...")
                mesh.fill_holes()
            
            # Remove degenerate faces
            mesh.remove_degenerate_faces()
            
            # Remove duplicate vertices
            mesh.remove_duplicate_faces()
            
            # Ensure proper scaling (normalize to [-0.5, 0.5] range)
            bounds = mesh.bounds
            center = (bounds[0] + bounds[1]) / 2
            scale = np.max(bounds[1] - bounds[0])
            
            mesh.vertices = (mesh.vertices - center) / scale * 0.9
            
            print(f"Mesh aligned: bounds {mesh.bounds}")
            return mesh
            
        except Exception as e:
            print(f"Error during mesh alignment: {e}")
            print("Continuing with original mesh...")
            return mesh
    
    def _prepare_mesh_for_texture(self, mesh):
        """
        Prepare mesh for Hunyuan3D-2 texture processing.
        
        Args:
            mesh: Trimesh object
            
        Returns:
            Mesh ready for texture processing
        """
        try:
            print("Preparing mesh for texture processing...")
            
            # Apply UV wrapping using Hunyuan3D-2's utility
            mesh = mesh_uv_wrap(mesh)
            print("UV wrapping completed")
            
            # Ensure the mesh has proper UV coordinates
            if not hasattr(mesh.visual, 'uv') or mesh.visual.uv is None:
                print("Warning: Mesh lacks UV coordinates, attempting to generate...")
                # Basic UV unwrapping fallback
                mesh.visual.uv = np.random.rand(len(mesh.vertices), 2)
            
            print(f"Mesh prepared with {len(mesh.vertices)} vertices and UV coordinates")
            return mesh
            
        except Exception as e:
            print(f"Error during mesh preparation: {e}")
            print("Attempting to continue with basic UV coordinates...")
            
            # Fallback: add basic UV coordinates
            if not hasattr(mesh.visual, 'uv') or mesh.visual.uv is None:
                mesh.visual.uv = np.random.rand(len(mesh.vertices), 2)
            
            return mesh
    
    def _extract_mesh_from_trellis(self, trellis_result):
        """
        Extract trimesh object from TRELLIS MeshExtractResult.
        
        Args:
            trellis_result: TRELLIS MeshExtractResult object
            
        Returns:
            trimesh.Trimesh object
        """
        # Handle different types of TRELLIS outputs
        if hasattr(trellis_result, 'vertices') and hasattr(trellis_result, 'faces'):
            # TRELLIS MeshExtractResult format (confirmed)
            vertices = trellis_result.vertices
            faces = trellis_result.faces
            print(f"Using TRELLIS format: vertices {vertices.shape}, faces {faces.shape}")
        elif hasattr(trellis_result, 'mesh_v') and hasattr(trellis_result, 'mesh_f'):
            # Alternative format (fallback)
            vertices = trellis_result.mesh_v
            faces = trellis_result.mesh_f
            print(f"Using alternative format: mesh_v {vertices.shape}, mesh_f {faces.shape}")
        else:
            # Debug information
            if hasattr(trellis_result, '__dict__'):
                print(f"Available attributes: {list(trellis_result.__dict__.keys())}")
            raise ValueError(f"Unknown TRELLIS result format: {type(trellis_result)}")
        
        # Convert torch tensors to numpy arrays
        if hasattr(vertices, 'cpu'):
            vertices = vertices.cpu().numpy()
        if hasattr(faces, 'cpu'):
            faces = faces.cpu().numpy()
        
        print(f"Converted to numpy: vertices {vertices.shape}, faces {faces.shape}")
        
        # Create trimesh object
        mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
        
        print(f"Created trimesh: {len(mesh.vertices)} vertices, {len(mesh.faces)} faces")
        return mesh

    def generate_3d(self, 
                   image,
                   output_path="hybrid_output",
                   save_intermediate=True,
                   trellis_params=None,
                   texture_params=None):
        """
        Generate 3D asset using hybrid pipeline.
        
        Args:
            image: Input image (PIL Image or path)
            output_path: Base path for outputs
            save_intermediate: Whether to save intermediate results
            trellis_params: Parameters for TRELLIS generation
            texture_params: Parameters for texture generation
            
        Returns:
            Final textured mesh (trimesh object)
        """
        print("=== Starting Hybrid 3D Generation ===")
        
        # Default parameters
        if trellis_params is None:
            trellis_params = {
                "seed": 42,
                "sparse_structure_sampler_params": {
                    "steps": 12,
                    "cfg_strength": 7.5,
                },
                "slat_sampler_params": {
                    "steps": 12,
                    "cfg_strength": 3,
                },
            }
        
        # Step 1: Preprocess image
        print("1. Preprocessing image...")
        processed_image = self._preprocess_image(image)
        
        if save_intermediate:
            processed_image.save(f"{output_path}_input_processed.png")
        
        # Step 2: Generate geometry with TRELLIS
        print("2. Generating geometry with TRELLIS SLAT...")
        
        trellis_outputs = self.trellis_pipeline.run(
            processed_image,
            **trellis_params
        )
        
        # Extract mesh from TRELLIS outputs - fix the mesh extraction
        print("2.1. Extracting mesh from TRELLIS result...")
        trellis_mesh_result = trellis_outputs['mesh'][0]
        print(f"TRELLIS result type: {type(trellis_mesh_result)}")
        
        # Convert to trimesh object
        trellis_mesh = self._extract_mesh_from_trellis(trellis_mesh_result)
        
        if save_intermediate:
            # Save intermediate geometry
            trellis_mesh.export(f"{output_path}_geometry_trellis.glb")
            
            # Also save other TRELLIS representations if desired
            if 'gaussian' in trellis_outputs:
                trellis_outputs['gaussian'][0].save_ply(f"{output_path}_gaussian.ply")
        
        print(f"Generated mesh with {len(trellis_mesh.vertices)} vertices, {len(trellis_mesh.faces)} faces")
        
        # Step 3: Align and prepare mesh
        print("3. Aligning coordinate systems and preparing mesh...")
        
        aligned_mesh = self._align_mesh_coordinates(trellis_mesh)
        prepared_mesh = self._prepare_mesh_for_texture(aligned_mesh)
        
        if save_intermediate:
            prepared_mesh.export(f"{output_path}_geometry_prepared.glb")
        
        # Step 4: Generate texture with Hunyuan3D-2
        print("4. Generating high-quality texture with Hunyuan3D-2...")
        
        try:
            textured_mesh = self.texture_pipeline(prepared_mesh, image=processed_image)
            
            # Save final result
            output_file = f"{output_path}_final_textured.glb"
            textured_mesh.export(output_file)
            
            print(f"=== Hybrid generation completed! ===")
            print(f"Final textured mesh saved to: {output_file}")
            
            return textured_mesh
            
        except Exception as e:
            print(f"Warning: Texture generation failed: {e}")
            print("Returning mesh without texture...")
            
            # Save untextured mesh as fallback
            fallback_file = f"{output_path}_geometry_only.glb"
            prepared_mesh.export(fallback_file)
            
            return prepared_mesh
    
    def batch_generate(self, image_list, output_dir="batch_outputs"):
        """
        Generate multiple 3D assets in batch.
        
        Args:
            image_list: List of image paths or PIL Images
            output_dir: Directory for batch outputs
            
        Returns:
            List of generated meshes
        """
        Path(output_dir).mkdir(exist_ok=True)
        results = []
        
        for i, image in enumerate(image_list):
            print(f"\n=== Processing image {i+1}/{len(image_list)} ===")
            
            output_path = os.path.join(output_dir, f"result_{i:03d}")
            
            try:
                mesh = self.generate_3d(
                    image, 
                    output_path=output_path,
                    save_intermediate=True
                )
                results.append(mesh)
                
            except Exception as e:
                print(f"Failed to process image {i}: {e}")
                results.append(None)
        
        return results


def main():
    """Example usage of the hybrid pipeline."""
    
    # Initialize hybrid pipeline
    pipeline = HybridTrellisHunyuanPipeline()
    
    # Example 1: Single image generation
    print("=== Example 1: Single Image Generation ===")
    
    # You can use either a file path or PIL Image
    # image_path = "assets/example_image/T.png"  # Adjust path as needed
    image_path = "/home/mbhat/three-gen-subnet-trellis/Hunyuan3D-2/outputs-arm/t2i_original.png"
    
    if os.path.exists(image_path):
        result_mesh = pipeline.generate_3d(
            image_path,
            output_path="example_hybrid_output2",
            save_intermediate=True
        )
        print(f"Generated mesh with {len(result_mesh.vertices)} vertices")
    else:
        print(f"Example image not found at {image_path}")
    
    # Example 2: Batch processing
    print("\n=== Example 2: Batch Processing ===")
    
    # Create some example images (you would replace with real images)
    example_images = []
    
    # # Add your image paths here
    # image_paths = [
    #     "path/to/image1.png",
    #     "path/to/image2.png", 
    #     # Add more images...
    # ]
    
    # # Filter existing images
    # existing_images = [img for img in image_paths if os.path.exists(img)]
    
    # if existing_images:
    #     batch_results = pipeline.batch_generate(
    #         existing_images,
    #         output_dir="batch_hybrid_outputs"
    #     )
    #     print(f"Processed {len([r for r in batch_results if r is not None])} images successfully")
    # else:
    #     print("No example images found for batch processing")


if __name__ == "__main__":
    main() 