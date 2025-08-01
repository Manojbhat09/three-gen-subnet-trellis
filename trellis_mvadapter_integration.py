import os
import sys
import torch
import tempfile
from PIL import Image
from typing import Optional, Dict, Any, Tuple
import trimesh
from io import BytesIO

# Add MV-Adapter to path
sys.path.append('./MV-Adapter')

class TrellisMVAdapterIntegration:
    """
    Integration class that combines Trellis Gaussian Splatting generation
    with MV-Adapter texture enhancement for improved 3D assets.
    """
    
    def __init__(self, 
                 trellis_pipeline,
                 mv_adapter_variant: str = "sdxl",
                 device: str = "cuda"):
        """
        Initialize the integrated pipeline.
        
        Args:
            trellis_pipeline: Your existing Trellis pipeline
            mv_adapter_variant: "sdxl" for high quality, "sd21" for lower VRAM
            device: Device to run on
        """
        self.trellis_pipeline = trellis_pipeline
        self.device = device
        self.mv_adapter_variant = mv_adapter_variant
        
        # Initialize MV-Adapter components
        self._setup_mv_adapter()
        
    def _setup_mv_adapter(self):
        """Setup MV-Adapter pipeline components."""
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
                
            # Prepare MV-Adapter pipeline
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
            
            # Setup texture pipeline
            from MV-Adapter.mvadapter.pipelines.pipeline_texture import TexturePipeline
            self.texture_pipeline = TexturePipeline(
                upscaler_ckpt_path="./checkpoints/RealESRGAN_x2plus.pth",
                inpaint_ckpt_path="./checkpoints/big-lama.pt",
                device=self.device,
            )
            
            self.height = height
            self.width = width
            self.uv_size = uv_size
            
        except ImportError as e:
            print(f"Warning: MV-Adapter not available: {e}")
            self.mv_pipeline = None
            self.texture_pipeline = None
    
    def generate_enhanced_3d(self, 
                           prompt: str, 
                           seed: int = 42,
                           enhance_texture: bool = True,
                           save_intermediate: bool = True) -> Dict[str, Any]:
        """
        Generate enhanced 3D model using Trellis + MV-Adapter.
        
        Args:
            prompt: Text prompt for generation
            seed: Random seed
            enhance_texture: Whether to apply MV-Adapter texture enhancement
            save_intermediate: Whether to save intermediate files
            
        Returns:
            Dictionary containing all outputs
        """
        results = {}
        
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
        
        # Extract outputs
        gaussian_output = trellis_outputs['gaussian'][0]
        mesh_output = trellis_outputs['mesh'][0]
        
        # Save original Trellis outputs
        results['trellis_gaussian'] = gaussian_output
        results['trellis_mesh'] = mesh_output
        
        # Save PLY file
        ply_buffer = BytesIO()
        gaussian_output.save_ply(ply_buffer)
        results['ply_data'] = ply_buffer.getvalue()
        
        if save_intermediate:
            gaussian_output.save_ply("trellis_original.ply")
        
        # Step 2: Texture Enhancement with MV-Adapter (if enabled)
        if enhance_texture and self.mv_pipeline is not None:
            print("Step 2: Enhancing texture with MV-Adapter...")
            
            # Convert mesh to GLB for MV-Adapter
            mesh_glb = self._convert_mesh_to_glb(mesh_output)
            
            # Generate enhanced texture
            enhanced_results = self._apply_mv_adapter_texture(
                mesh_glb, prompt, seed, save_intermediate
            )
            
            results.update(enhanced_results)
            
        else:
            print("Step 2: Skipping texture enhancement (MV-Adapter not available)")
        
        return results
    
    def _convert_mesh_to_glb(self, mesh_output) -> str:
        """Convert Trellis mesh output to GLB file for MV-Adapter."""
        # This is a simplified conversion - you may need to adapt based on your mesh format
        temp_glb = tempfile.NamedTemporaryFile(suffix='.glb', delete=False)
        temp_glb.close()
        
        # Convert mesh to trimesh and export as GLB
        # You'll need to implement this based on your specific mesh format
        try:
            # Example conversion (adapt to your mesh format)
            if hasattr(mesh_output, 'export'):
                mesh_output.export(temp_glb.name)
            else:
                # Create a basic mesh if conversion fails
                mesh = trimesh.creation.box()
                mesh.export(temp_glb.name)
                
        except Exception as e:
            print(f"Warning: Mesh conversion failed: {e}")
            # Create fallback mesh
            mesh = trimesh.creation.box()
            mesh.export(temp_glb.name)
        
        return temp_glb.name
    
    def _apply_mv_adapter_texture(self, 
                                 mesh_path: str, 
                                 prompt: str, 
                                 seed: int,
                                 save_intermediate: bool) -> Dict[str, Any]:
        """Apply MV-Adapter texture enhancement to the mesh."""
        results = {}
        
        try:
            # Import MV-Adapter functions
            if self.mv_adapter_variant == "sdxl":
                from MV-Adapter.scripts.inference_tg2mv_sdxl import run_pipeline
            else:
                from MV-Adapter.scripts.inference_tg2mv_sd import run_pipeline
            
            # Generate multi-view images
            images, pos_images, normal_images = run_pipeline(
                self.mv_pipeline,
                mesh_path=mesh_path,
                num_views=6,
                text=prompt,
                height=self.height,
                width=self.width,
                num_inference_steps=50,
                guidance_scale=7.0,
                seed=seed,
                negative_prompt="watermark, ugly, deformed, noisy, blurry, low contrast",
                device=self.device,
            )
            
            results['mv_images'] = images
            
            if save_intermediate:
                from MV-Adapter.mvadapter.utils import make_image_grid
                mv_grid = make_image_grid(images, rows=1)
                mv_grid.save("mv_adapter_views.png")
            
            # Generate textured model
            output_dir = "./enhanced_output"
            os.makedirs(output_dir, exist_ok=True)
            
            mv_path = os.path.join(output_dir, "mv_views.png")
            from MV-Adapter.mvadapter.utils import make_image_grid
            make_image_grid(images, rows=1).save(mv_path)
            
            # Apply texture to mesh
            from MV-Adapter.mvadapter.pipelines.pipeline_texture import ModProcessConfig
            
            texture_output = self.texture_pipeline(
                mesh_path=mesh_path,
                save_dir=output_dir,
                save_name="enhanced_model",
                uv_unwarp=True,
                preprocess_mesh=True,
                uv_size=self.uv_size,
                rgb_path=mv_path,
                rgb_process_config=ModProcessConfig(
                    view_upscale=True, 
                    inpaint_mode="view"
                ),
                camera_azimuth_deg=[x - 90 for x in [0, 90, 180, 270, 180, 180]],
            )
            
            results['textured_model_path'] = texture_output.shaded_model_save_path
            results['enhancement_success'] = True
            
            print(f"✓ Enhanced textured model saved to: {texture_output.shaded_model_save_path}")
            
        except Exception as e:
            print(f"❌ MV-Adapter texture enhancement failed: {e}")
            results['enhancement_success'] = False
            results['enhancement_error'] = str(e)
        
        return results
    
    def generate_from_image(self, 
                          image: Image.Image, 
                          prompt: str,
                          seed: int = 42,
                          enhance_texture: bool = True) -> Dict[str, Any]:
        """
        Generate enhanced 3D model from image using Trellis + MV-Adapter.
        
        Args:
            image: Input image
            prompt: Text description
            seed: Random seed
            enhance_texture: Whether to apply texture enhancement
            
        Returns:
            Dictionary containing all outputs
        """
        # This would be similar to generate_enhanced_3d but using image-to-3D pipeline
        # Implementation depends on your specific image-to-3D setup
        
        results = {}
        
        # Step 1: Generate with Trellis image pipeline
        print("Step 1: Generating 3D model from image with Trellis...")
        # Your image-to-3D pipeline call here
        
        # Step 2: Apply MV-Adapter texture enhancement
        if enhance_texture and self.mv_pipeline is not None:
            print("Step 2: Enhancing texture with MV-Adapter...")
            # Similar to _apply_mv_adapter_texture but with image conditioning
        
        return results


# Example usage
def main():
    """Example usage of the integrated pipeline."""
    
    # Initialize your Trellis pipeline
    # trellis_pipeline = your_trellis_pipeline_here
    
    # Create integrated pipeline
    # integrated_pipeline = TrellisMVAdapterIntegration(trellis_pipeline)
    
    # Generate enhanced 3D model
    # results = integrated_pipeline.generate_enhanced_3d(
    #     prompt="a beautiful red sports car",
    #     seed=42,
    #     enhance_texture=True
    # )
    
    # Access results
    # ply_data = results['ply_data']  # Original Gaussian Splatting PLY
    # textured_model = results['textured_model_path']  # Enhanced textured GLB
    # mv_images = results['mv_images']  # Multi-view images
    
    print("Integration script created. Modify main() function with your specific pipeline setup.")


if __name__ == "__main__":
    main() 