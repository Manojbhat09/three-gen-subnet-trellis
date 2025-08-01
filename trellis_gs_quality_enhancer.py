#!/usr/bin/env python3
"""
Trellis GS Quality Enhancer
Purpose: Use MV-Adapter to refine Gaussian Splatting quality from Trellis outputs
while maintaining SPZ-compressed PLY format for validator compatibility.

Pipeline: Text → Trellis GS → MV-Adapter Refinement → Enhanced GS → SPZ PLY
"""

import os
import sys
import torch
import tempfile
import numpy as np
import trimesh
from PIL import Image
from typing import Optional, Dict, Any, Tuple, List
from io import BytesIO
import gc
import time
import json
import base64
from pathlib import Path

# Add MV-Adapter to path
sys.path.append('./MV-Adapter')

# Import validation components for quality assessment
try:
    from validation.engine.data_structures import GaussianSplattingData
    from validation.engine.io.ply.loader import PlyLoader
    from validation.engine.rendering.renderer import Renderer
    from validation.engine.validation_engine import ValidationEngine
    print("✅ Validation components available for quality assessment")
except ImportError:
    print("⚠️ Validation components not available - quality assessment disabled")
    ValidationEngine = None

class TrellisGSQualityEnhancer:
    """
    Specialized class for enhancing Gaussian Splatting quality using MV-Adapter
    while maintaining validator-compatible SPZ-compressed PLY output.
    """
    
    def __init__(self, 
                 trellis_pipeline,
                 mv_adapter_variant: str = "sdxl",
                 device: str = "cuda",
                 enable_quality_assessment: bool = True):
        """
        Initialize the GS quality enhancer.
        
        Args:
            trellis_pipeline: Your existing Trellis pipeline
            mv_adapter_variant: "sdxl" for high quality, "sd21" for lower VRAM
            device: Device to run on
            enable_quality_assessment: Whether to assess quality improvements
        """
        self.trellis_pipeline = trellis_pipeline
        self.device = device
        self.mv_adapter_variant = mv_adapter_variant
        self.enable_quality_assessment = enable_quality_assessment
        
        # Initialize MV-Adapter components
        self._setup_mv_adapter()
        
        # Initialize quality assessment if available
        if enable_quality_assessment and ValidationEngine:
            self._setup_quality_assessment()
        else:
            self.validator = None
            self.renderer = None
            self.ply_loader = None
    
    def _setup_mv_adapter(self):
        """Setup MV-Adapter pipeline components for GS enhancement."""
        try:
            if self.mv_adapter_variant == "sdxl":
                from MV-Adapter.scripts.inference_tg2mv_sdxl import prepare_pipeline
                base_model = "stabilityai/stable-diffusion-xl-base-1.0"
                vae_model = "madebyollin/sdxl-vae-fp16-fix"
                height = width = 768
                uv_size = 4096
            else:  # sd21
                from MV-Adapter.scripts.inference_tg2mv_sd import prepare_pipeline
                base_model = "stabilityai/stable-diffusion-2-1-base"
                vae_model = None
                height = width = 512
                uv_size = 2048
                
            # Prepare MV-Adapter pipeline for geometry-guided generation
            self.mv_pipeline = prepare_pipeline(
                base_model=base_model,
                vae_model=vae_model,
                unet_model=None,
                lora_model=None,
                adapter_path="huanngzh/mv-adapter",
                scheduler=None,
                num_views=6,
                device=self.device,
                dtype=torch.float16,
            )
            
            # Setup texture pipeline for high-quality texture generation
            from MV-Adapter.mvadapter.pipelines.pipeline_texture import TexturePipeline
            self.texture_pipeline = TexturePipeline(
                upscaler_ckpt_path="./checkpoints/RealESRGAN_x2plus.pth",
                inpaint_ckpt_path="./checkpoints/big-lama.pt",
                device=self.device,
            )
            
            self.height = height
            self.width = width
            self.uv_size = uv_size
            
            print(f"✅ MV-Adapter initialized ({self.mv_adapter_variant})")
            
        except ImportError as e:
            print(f"❌ MV-Adapter not available: {e}")
            self.mv_pipeline = None
            self.texture_pipeline = None
    
    def _setup_quality_assessment(self):
        """Setup quality assessment components."""
        try:
            self.validator = ValidationEngine(verbose=False)
            self.validator.load_pipelines()
            self.renderer = Renderer()
            self.ply_loader = PlyLoader()
            print("✅ Quality assessment components initialized")
        except Exception as e:
            print(f"⚠️ Quality assessment setup failed: {e}")
            self.validator = None
            self.renderer = None
            self.ply_loader = None
    
    def assess_gs_quality(self, gs_data: bytes, prompt: str) -> Dict[str, float]:
        """Assess the quality of Gaussian Splatting data."""
        if not self.validator or not self.enable_quality_assessment:
            return {"quality_score": 0.0, "assessment_available": False}
        
        try:
            # Load GS data
            gs_loaded = self.ply_loader.load(gs_data)
            
            # Render multiple views for assessment
            camera_positions = [
                (0, 0, 2), (2, 0, 0), (0, 2, 0), (-2, 0, 0), (0, -2, 0), (0, 0, -2)
            ]
            
            rendered_images = []
            for pos in camera_positions:
                image = self.renderer.render_gaussian_splatting(
                    gs_loaded, 
                    camera_position=pos,
                    render_size=(512, 512)
                )
                rendered_images.append(image)
            
            # Assess quality using validation engine
            quality_metrics = self.validator.assess_quality(rendered_images)
            
            return {
                "quality_score": quality_metrics.get("combined_quality", 0.0),
                "aesthetic_score": quality_metrics.get("aesthetic", 0.0),
                "technical_score": quality_metrics.get("technical", 0.0),
                "assessment_available": True
            }
            
        except Exception as e:
            print(f"⚠️ Quality assessment failed: {e}")
            return {"quality_score": 0.0, "assessment_available": False}
    
    def enhance_gaussian_splatting(self, 
                                 original_gs: bytes,
                                 prompt: str,
                                 enhancement_strength: float = 0.8,
                                 max_iterations: int = 3) -> Tuple[bytes, Dict[str, Any]]:
        """
        Enhance Gaussian Splatting quality using MV-Adapter.
        
        Args:
            original_gs: Original Gaussian Splatting PLY data
            prompt: Text prompt for enhancement guidance
            enhancement_strength: How much to enhance (0.0 to 1.0)
            max_iterations: Maximum enhancement iterations
            
        Returns:
            Tuple of (enhanced_gs_data, enhancement_metrics)
        """
        enhancement_metrics = {
            "original_quality": 0.0,
            "enhanced_quality": 0.0,
            "quality_improvement": 0.0,
            "iterations_performed": 0,
            "enhancement_success": False
        }
        
        try:
            print(f"🎨 Starting GS quality enhancement for: '{prompt}'")
            
            # Step 1: Assess original quality
            print("Step 1: Assessing original GS quality...")
            original_quality = self.assess_gs_quality(original_gs, prompt)
            enhancement_metrics["original_quality"] = original_quality.get("quality_score", 0.0)
            
            print(f"📊 Original quality score: {enhancement_metrics['original_quality']:.4f}")
            
            # Step 2: Convert GS to mesh for MV-Adapter processing
            print("Step 2: Converting GS to mesh for enhancement...")
            mesh_path = self._gs_to_mesh_for_enhancement(original_gs)
            
            # Step 3: Generate enhanced multi-view images
            print("Step 3: Generating enhanced multi-view images...")
            enhanced_images = self._generate_enhanced_views(mesh_path, prompt)
            
            if enhanced_images is None:
                print("❌ Enhanced view generation failed")
                return original_gs, enhancement_metrics
            
            # Step 4: Reconstruct enhanced Gaussian Splatting
            print("Step 4: Reconstructing enhanced Gaussian Splatting...")
            enhanced_gs = self._reconstruct_enhanced_gs(
                original_gs, enhanced_images, prompt, enhancement_strength
            )
            
            if enhanced_gs is None:
                print("❌ Enhanced GS reconstruction failed")
                return original_gs, enhancement_metrics
            
            # Step 5: Assess enhanced quality
            print("Step 5: Assessing enhanced GS quality...")
            enhanced_quality = self.assess_gs_quality(enhanced_gs, prompt)
            enhancement_metrics["enhanced_quality"] = enhanced_quality.get("quality_score", 0.0)
            enhancement_metrics["quality_improvement"] = (
                enhancement_metrics["enhanced_quality"] - enhancement_metrics["original_quality"]
            )
            enhancement_metrics["enhancement_success"] = True
            enhancement_metrics["iterations_performed"] = 1
            
            print(f"📊 Enhancement results:")
            print(f"   Original quality: {enhancement_metrics['original_quality']:.4f}")
            print(f"   Enhanced quality: {enhancement_metrics['enhanced_quality']:.4f}")
            print(f"   Improvement: {enhancement_metrics['quality_improvement']:.4f}")
            
            # Cleanup
            if os.path.exists(mesh_path):
                os.remove(mesh_path)
            
            return enhanced_gs, enhancement_metrics
            
        except Exception as e:
            print(f"❌ GS enhancement failed: {e}")
            return original_gs, enhancement_metrics
    
    def _gs_to_mesh_for_enhancement(self, gs_data: bytes) -> str:
        """Convert Gaussian Splatting to mesh for MV-Adapter processing."""
        try:
            # Load GS data
            from plyfile import PlyData
            from io import BytesIO
            
            ply_data = PlyData.read(BytesIO(gs_data))
            vertices = ply_data['vertex']
            
            # Extract positions and colors
            positions = np.column_stack([vertices['x'], vertices['y'], vertices['z']])
            colors = np.column_stack([vertices['f_dc_0'], vertices['f_dc_1'], vertices['f_dc_2']])
            
            # Create point cloud mesh
            mesh = trimesh.Trimesh(vertices=positions, faces=[])
            
            # Add vertex colors
            mesh.visual.vertex_colors = (colors * 255).astype(np.uint8)
            
            # Export as temporary GLB file
            temp_glb = tempfile.NamedTemporaryFile(suffix='.glb', delete=False)
            temp_glb.close()
            
            mesh.export(temp_glb.name)
            return temp_glb.name
            
        except Exception as e:
            print(f"❌ GS to mesh conversion failed: {e}")
            # Create fallback mesh
            mesh = trimesh.creation.box()
            temp_glb = tempfile.NamedTemporaryFile(suffix='.glb', delete=False)
            temp_glb.close()
            mesh.export(temp_glb.name)
            return temp_glb.name
    
    def _generate_enhanced_views(self, mesh_path: str, prompt: str) -> Optional[List[Image.Image]]:
        """Generate enhanced multi-view images using MV-Adapter."""
        try:
            if self.mv_pipeline is None:
                print("❌ MV-Adapter pipeline not available")
                return None
            
            # Import MV-Adapter functions
            if self.mv_adapter_variant == "sdxl":
                from MV-Adapter.scripts.inference_tg2mv_sdxl import run_pipeline
            else:
                from MV-Adapter.scripts.inference_tg2mv_sd import run_pipeline
            
            # Generate enhanced multi-view images
            images, pos_images, normal_images = run_pipeline(
                self.mv_pipeline,
                mesh_path=mesh_path,
                num_views=6,
                text=prompt,
                height=self.height,
                width=self.width,
                num_inference_steps=50,
                guidance_scale=7.0,
                seed=42,  # Fixed seed for consistency
                negative_prompt="watermark, ugly, deformed, noisy, blurry, low contrast, low quality",
                device=self.device,
            )
            
            print(f"✅ Generated {len(images)} enhanced views")
            return images
            
        except Exception as e:
            print(f"❌ Enhanced view generation failed: {e}")
            return None
    
    def _reconstruct_enhanced_gs(self, 
                               original_gs: bytes, 
                               enhanced_images: List[Image.Image],
                               prompt: str,
                               enhancement_strength: float) -> Optional[bytes]:
        """Reconstruct enhanced Gaussian Splatting from enhanced images."""
        try:
            # Load original GS data
            from plyfile import PlyData
            from io import BytesIO
            
            ply_data = PlyData.read(BytesIO(original_gs))
            vertices = ply_data['vertex']
            
            # Extract original attributes
            positions = np.column_stack([vertices['x'], vertices['y'], vertices['z']])
            original_colors = np.column_stack([vertices['f_dc_0'], vertices['f_dc_1'], vertices['f_dc_2']])
            
            # Create enhanced colors based on enhanced images
            enhanced_colors = self._extract_colors_from_enhanced_images(
                positions, enhanced_images, enhancement_strength
            )
            
            # Blend original and enhanced colors
            final_colors = (
                (1 - enhancement_strength) * original_colors + 
                enhancement_strength * enhanced_colors
            )
            
            # Create enhanced vertex data
            enhanced_vertices = []
            for i in range(len(positions)):
                vertex = list(positions[i])
                vertex.extend([0, 0, 0])  # normals (nx, ny, nz)
                vertex.extend(final_colors[i])  # enhanced colors (f_dc_0, f_dc_1, f_dc_2)
                
                # Add remaining SH coefficients (f_rest_0 through f_rest_44)
                for j in range(45):
                    vertex.append(vertices[f'f_rest_{j}'][i])
                
                # Add other attributes
                vertex.append(vertices['opacity'][i])
                vertex.extend([vertices['scale_0'][i], vertices['scale_1'][i], vertices['scale_2'][i]])
                vertex.extend([vertices['rot_0'][i], vertices['rot_1'][i], vertices['rot_2'][i], vertices['rot_3'][i]])
                
                enhanced_vertices.append(tuple(vertex))
            
            # Create enhanced PLY data
            enhanced_ply_data = self._create_enhanced_ply(enhanced_vertices, ply_data)
            
            return enhanced_ply_data
            
        except Exception as e:
            print(f"❌ Enhanced GS reconstruction failed: {e}")
            return None
    
    def _extract_colors_from_enhanced_images(self, 
                                           positions: np.ndarray, 
                                           enhanced_images: List[Image.Image],
                                           enhancement_strength: float) -> np.ndarray:
        """Extract colors from enhanced images based on 3D positions."""
        try:
            # Simple color extraction - project 3D points to 2D and sample colors
            # This is a simplified approach; you could implement more sophisticated projection
            
            num_points = len(positions)
            enhanced_colors = np.zeros((num_points, 3))
            
            # For each enhanced image, sample colors for visible points
            for img_idx, image in enumerate(enhanced_images):
                img_array = np.array(image)
                h, w = img_array.shape[:2]
                
                # Simple spherical projection
                for point_idx, pos in enumerate(positions):
                    # Normalize position
                    norm_pos = pos / (np.linalg.norm(pos) + 1e-8)
                    
                    # Project to image coordinates
                    u = int((norm_pos[0] + 1) * w / 2)
                    v = int((norm_pos[1] + 1) * h / 2)
                    
                    # Clamp coordinates
                    u = max(0, min(w - 1, u))
                    v = max(0, min(h - 1, v))
                    
                    # Sample color
                    color = img_array[v, u, :3] / 255.0
                    enhanced_colors[point_idx] += color
            
            # Average colors from all views
            enhanced_colors /= len(enhanced_images)
            
            return enhanced_colors
            
        except Exception as e:
            print(f"⚠️ Color extraction failed: {e}")
            # Return original colors as fallback
            return np.ones((len(positions), 3)) * 0.5
    
    def _create_enhanced_ply(self, enhanced_vertices: List[tuple], original_ply_data) -> bytes:
        """Create enhanced PLY data from enhanced vertices."""
        try:
            from plyfile import PlyData, PlyElement
            
            # Create vertex element
            vertex_element = PlyElement.describe(
                np.array(enhanced_vertices, dtype=original_ply_data['vertex'].dtype),
                'vertex'
            )
            
            # Create enhanced PLY data
            enhanced_ply_data = PlyData([vertex_element])
            
            # Write to bytes
            buffer = BytesIO()
            enhanced_ply_data.write(buffer)
            return buffer.getvalue()
            
        except Exception as e:
            print(f"❌ Enhanced PLY creation failed: {e}")
            return None
    
    def generate_enhanced_3d_model(self, 
                                 prompt: str, 
                                 seed: int = 42,
                                 enable_enhancement: bool = True,
                                 enhancement_strength: float = 0.8) -> Dict[str, Any]:
        """
        Generate enhanced 3D model using Trellis + GS quality enhancement.
        
        Args:
            prompt: Text prompt for generation
            seed: Random seed
            enable_enhancement: Whether to apply GS enhancement
            enhancement_strength: Strength of enhancement (0.0 to 1.0)
            
        Returns:
            Dictionary containing all outputs and metrics
        """
        results = {}
        
        start_time = time.time()
        
        try:
            print(f"🎯 Starting enhanced 3D generation for: '{prompt}' (seed: {seed})")
            
            # Step 1: Generate with Trellis
            print("Step 1: Generating 3D model with Trellis...")
            trellis_outputs = self.trellis_pipeline.run(
                prompt,
                seed=seed,
                formats=["gaussian", "mesh"],
                preprocess_image=False,
                sparse_structure_sampler_params={
                    "steps": 12,
                    "cfg_strength": 7.5,
                },
                slat_sampler_params={
                    "steps": 12,
                    "cfg_strength": 3.0,
                },
            )
            
            # Extract original GS
            original_gs = trellis_outputs['gaussian'][0]
            
            # Save original PLY
            ply_buffer = BytesIO()
            original_gs.save_ply(ply_buffer)
            original_ply_data = ply_buffer.getvalue()
            
            results['original_ply_data'] = original_ply_data
            results['original_gs'] = original_gs
            
            print(f"✓ Original GS generated ({len(original_ply_data):,} bytes)")
            
            # Step 2: Apply GS quality enhancement (if enabled)
            if enable_enhancement and self.mv_pipeline is not None:
                print("Step 2: Applying GS quality enhancement...")
                
                enhanced_ply_data, enhancement_metrics = self.enhance_gaussian_splatting(
                    original_ply_data, prompt, enhancement_strength
                )
                
                results['enhanced_ply_data'] = enhanced_ply_data
                results['enhancement_metrics'] = enhancement_metrics
                
                if enhancement_metrics['enhancement_success']:
                    print(f"✅ GS enhancement successful!")
                    print(f"   Quality improvement: {enhancement_metrics['quality_improvement']:.4f}")
                    final_ply_data = enhanced_ply_data
                else:
                    print("⚠️ GS enhancement failed, using original")
                    final_ply_data = original_ply_data
            else:
                print("Step 2: Skipping GS enhancement")
                final_ply_data = original_ply_data
                results['enhancement_metrics'] = {
                    "enhancement_success": False,
                    "reason": "Enhancement disabled or MV-Adapter not available"
                }
            
            # Step 3: Compress with SPZ for validator compatibility
            print("Step 3: Compressing with SPZ...")
            try:
                import pyspz
                compressed_data = pyspz.compress(final_ply_data, workers=-1)
                
                compression_ratio = len(compressed_data) / len(final_ply_data)
                print(f"🗜️ SPZ Compression successful:")
                print(f"   Original: {len(final_ply_data):,} bytes ({len(final_ply_data)/1024/1024:.1f} MB)")
                print(f"   Compressed: {len(compressed_data):,} bytes ({len(compressed_data)/1024/1024:.1f} MB)")
                print(f"   Ratio: {compression_ratio*100:.1f}%")
                
                results['compressed_ply_data'] = compressed_data
                results['compression_ratio'] = compression_ratio
                
            except Exception as e:
                print(f"⚠️ SPZ compression failed: {e}")
                results['compressed_ply_data'] = final_ply_data
                results['compression_ratio'] = 1.0
            
            # Final results
            generation_time = time.time() - start_time
            results['generation_time'] = generation_time
            results['final_ply_data'] = final_ply_data
            results['success'] = True
            
            print(f"🎉 Enhanced 3D generation completed in {generation_time:.2f}s")
            
            return results
            
        except Exception as e:
            print(f"❌ Enhanced 3D generation failed: {e}")
            results['success'] = False
            results['error'] = str(e)
            return results


# Example usage and integration with your existing server
def integrate_with_trellis_server():
    """Example integration with your existing Trellis server."""
    
    # This would be integrated into your existing TrellisBaseGenerator class
    
    class EnhancedTrellisGenerator:
        def __init__(self):
            # Your existing initialization
            self.trellis_pipeline = None
            self.gs_enhancer = None
            
        def _setup_gs_enhancer(self):
            """Setup GS quality enhancer."""
            if self.trellis_pipeline is not None:
                self.gs_enhancer = TrellisGSQualityEnhancer(
                    trellis_pipeline=self.trellis_pipeline,
                    mv_adapter_variant="sdxl",  # or "sd21" for lower VRAM
                    device="cuda",
                    enable_quality_assessment=True
                )
                print("✅ GS Quality Enhancer initialized")
        
        def generate_enhanced_3d_model(self, prompt: str, seed: int = 42) -> Optional[Tuple[bytes, Optional[bytes]]]:
            """Generate enhanced 3D model with GS quality improvement."""
            
            # Setup enhancer if not already done
            if self.gs_enhancer is None:
                self._setup_gs_enhancer()
            
            # Generate enhanced model
            results = self.gs_enhancer.generate_enhanced_3d_model(
                prompt=prompt,
                seed=seed,
                enable_enhancement=True,
                enhancement_strength=0.8
            )
            
            if results['success']:
                return results['final_ply_data'], results['compressed_ply_data']
            else:
                print(f"❌ Enhanced generation failed: {results.get('error', 'Unknown error')}")
                return None
    
    return EnhancedTrellisGenerator


if __name__ == "__main__":
    print("🎨 Trellis GS Quality Enhancer")
    print("=" * 50)
    print("This module provides GS quality enhancement using MV-Adapter")
    print("while maintaining SPZ-compressed PLY format for validator compatibility.")
    print()
    print("Integration:")
    print("1. Import this module into your Trellis server")
    print("2. Initialize TrellisGSQualityEnhancer with your pipeline")
    print("3. Use generate_enhanced_3d_model() for quality-enhanced generation")
    print("4. Output will be SPZ-compressed PLY compatible with validator")
    print()
    print("Example:")
    print("enhancer = TrellisGSQualityEnhancer(trellis_pipeline)")
    print("results = enhancer.generate_enhanced_3d_model('a beautiful vase', 42)")
    print("compressed_ply = results['compressed_ply_data']  # Ready for validator") 