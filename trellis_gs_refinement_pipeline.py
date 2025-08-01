import os
import sys
import torch
import numpy as np
import trimesh
import io
from typing import List, Tuple, Optional, Dict, Any
from PIL import Image
import tempfile
import shutil

# Add MV-Adapter to path
sys.path.append('./MV-Adapter')

from mvadapter.pipelines.pipeline_mvadapter_t2mv_sdxl import MVAdapterT2MVSDXLPipeline
from mvadapter.utils import get_orthogonal_camera, make_image_grid, tensor_to_image
from mvadapter.utils.mesh_utils import NVDiffRastContextWrapper, load_mesh, render
from mvadapter.models.attention_processor import DecoupledMVRowColSelfAttnProcessor2_0
from mvadapter.schedulers.scheduling_shift_snr import ShiftSNRScheduler

# Add Trellis to path
sys.path.append('./TRELLIS')
from trellis.representations.gaussian.gaussian_model import Gaussian
from trellis.utils.postprocessing_utils import simplify_gs

# Add validation components
sys.path.append('./validation')
from validation.validation_lib.validation.validation_pipeline import ValidationEngine
from validation.engine.rendering.renderer import Renderer
from validation.engine.io.ply.loader import PlyLoader
from validation.engine.data_structures import GaussianSplattingData


class TrellisGSRefinementPipeline:
    """
    Correct implementation of GS refinement using MV-Adapter's multi-view images as targets.
    
    Pipeline: Text → Trellis (GS + Mesh) → MV-Adapter (High-Quality Target Images) → GS Refinement → SPZ Compression
    """
    
    def __init__(
        self,
        trellis_pipeline=None,
        mv_adapter_variant: str = "sdxl",
        device: str = "cuda",
        refinement_steps: int = 1000,
        learning_rate: float = 1e-3,
        num_views: int = 6,
        image_size: int = 512,
        enable_validation: bool = True
    ):
        self.trellis_pipeline = trellis_pipeline
        self.device = device
        self.refinement_steps = refinement_steps
        self.learning_rate = learning_rate
        self.num_views = num_views
        self.image_size = image_size
        self.enable_validation = enable_validation
        
        # Initialize MV-Adapter pipeline
        self.mv_pipeline = self._setup_mv_adapter_pipeline(mv_adapter_variant)
        
        # Initialize validator's rendering pipeline (BETTER than MV-Adapter's)
        self.validator_renderer = Renderer()
        
        # Initialize validation engine if enabled
        self.validation_engine = None
        if enable_validation:
            self.validation_engine = ValidationEngine()
    
    def _setup_mv_adapter_pipeline(self, variant: str) -> MVAdapterT2MVSDXLPipeline:
        """Setup MV-Adapter pipeline for geometry-guided multi-view generation"""
        print("Setting up MV-Adapter pipeline...")
        
        if variant == "sdxl":
            base_model = "stabilityai/stable-diffusion-xl-base-1.0"
            vae_model = "madebyollin/sdxl-vae-fp16-fix"
            height = width = 768
        elif variant == "sd21":
            base_model = "stabilityai/stable-diffusion-2-1-base"
            vae_model = None
            height = width = 512
        else:
            raise ValueError(f"Invalid variant: {variant}")
        
        # Import pipeline preparation function
        from mvadapter.pipelines.pipeline_mvadapter_t2mv_sdxl import MVAdapterT2MVSDXLPipeline
        from diffusers import AutoencoderKL, DDPMScheduler, UNet2DConditionModel
        from transformers import CLIPTextModel, CLIPTextModelWithProjection, CLIPTokenizer
        
        # Load pipeline
        pipe = MVAdapterT2MVSDXLPipeline.from_pretrained(base_model)
        
        # Load VAE if specified
        if vae_model is not None:
            pipe.vae = AutoencoderKL.from_pretrained(vae_model)
        
        # Setup scheduler
        pipe.scheduler = ShiftSNRScheduler.from_scheduler(
            pipe.scheduler,
            shift_mode="interpolated",
            shift_scale=8.0,
            scheduler_class=DDPMScheduler,
        )
        
        # Initialize custom adapter
        pipe.init_custom_adapter(
            num_views=self.num_views,
            self_attn_processor=DecoupledMVRowColSelfAttnProcessor2_0
        )
        
        # Load adapter weights
        adapter_path = "huanngzh/mv-adapter"
        pipe.load_custom_adapter(
            adapter_path, 
            weight_name="mvadapter_tg2mv_sdxl.safetensors"
        )
        
        pipe.to(device=self.device, dtype=torch.float16)
        pipe.cond_encoder.to(device=self.device, dtype=torch.float16)
        pipe.enable_vae_slicing()
        
        print("✅ MV-Adapter pipeline ready")
        return pipe
    
    def generate_refined_3d_model(
        self, 
        prompt: str, 
        seed: int = 42,
        enable_refinement: bool = True,
        refinement_strength: float = 1.0
    ) -> Dict[str, Any]:
        """
        Generate refined 3D model using the correct pipeline.
        
        Args:
            prompt: Text prompt for generation
            seed: Random seed for reproducibility
            enable_refinement: Whether to apply GS refinement
            refinement_strength: Strength of refinement (0.0 to 1.0)
            
        Returns:
            Dictionary containing refined GS, mesh, and metadata
        """
        
        # Step 1: Generate initial GS and mesh with Trellis
        print("Step 1: Generating initial GS and mesh with Trellis...")
        trellis_outputs = self._generate_with_trellis(prompt, seed)
        
        if not enable_refinement:
            return trellis_outputs
        
        # Step 2: Generate high-quality target images with MV-Adapter
        print("Step 2: Generating high-quality target images with MV-Adapter...")
        target_images = self._generate_target_images(
            trellis_outputs['mesh'], 
            prompt, 
            seed
        )
        
        # Step 3: Refine Gaussian Splatting using target images
        print("Step 3: Refining Gaussian Splatting...")
        refined_gs = self._refine_gaussian_splatting(
            trellis_outputs['gaussian'],
            target_images,
            prompt,
            refinement_strength
        )
        
        # Step 4: Create final output
        print("Step 4: Creating final output...")
        refined_outputs = {
            'gaussian': refined_gs,
            'mesh': trellis_outputs['mesh'],
            'target_images': target_images,
            'original_gs': trellis_outputs['gaussian'],
            'prompt': prompt,
            'seed': seed,
            'refinement_applied': True
        }
        
        return refined_outputs
    
    def _generate_with_trellis(self, prompt: str, seed: int) -> Dict[str, Any]:
        """Generate initial GS and mesh using Trellis"""
        if self.trellis_pipeline is None:
            raise ValueError("Trellis pipeline not initialized")
        
        # Generate with Trellis
        outputs = self.trellis_pipeline.generate_3d_model(prompt, seed)
        
        return {
            'gaussian': outputs['gaussian'],
            'mesh': outputs['mesh'],
            'prompt': prompt,
            'seed': seed
        }
    
    def _generate_target_images(
        self, 
        mesh_path: str, 
        prompt: str, 
        seed: int
    ) -> List[Image.Image]:
        """Generate high-quality multi-view images using MV-Adapter"""
        
        # Prepare cameras for multi-view rendering
        cameras = get_orthogonal_camera(
            elevation_deg=[0, 0, 0, 0, 89.99, -89.99],
            distance=[1.8] * self.num_views,
            left=-0.55,
            right=0.55,
            bottom=-0.55,
            top=0.55,
            azimuth_deg=[x - 90 for x in [0, 90, 180, 270, 180, 180]],
            device=self.device,
        )
        
        # Setup rendering context
        ctx = NVDiffRastContextWrapper(device=self.device)
        
        # Load and render mesh
        mesh = load_mesh(mesh_path, rescale=True, device=self.device)
        render_out = render(
            ctx,
            mesh,
            cameras,
            height=self.image_size,
            width=self.image_size,
            render_attr=False,
            normal_background=0.0,
        )
        
        # Create control images (position + normal)
        pos_images = tensor_to_image((render_out.pos + 0.5).clamp(0, 1), batched=True)
        normal_images = tensor_to_image(
            (render_out.normal / 2 + 0.5).clamp(0, 1), batched=True
        )
        control_images = (
            torch.cat(
                [
                    (render_out.pos + 0.5).clamp(0, 1),
                    (render_out.normal / 2 + 0.5).clamp(0, 1),
                ],
                dim=-1,
            )
            .permute(0, 3, 1, 2)
            .to(self.device)
        )
        
        # Generate multi-view images with MV-Adapter
        pipe_kwargs = {}
        if seed != -1:
            pipe_kwargs["generator"] = torch.Generator(device=self.device).manual_seed(seed)
        
        images = self.mv_pipeline(
            prompt,
            height=self.image_size,
            width=self.image_size,
            num_inference_steps=50,
            guidance_scale=7.0,
            num_images_per_prompt=self.num_views,
            control_image=control_images,
            control_conditioning_scale=1.0,
            negative_prompt="watermark, ugly, deformed, noisy, blurry, low contrast",
            **pipe_kwargs,
        ).images
        
        return images
    
    def _refine_gaussian_splatting(
        self,
        original_gs: Gaussian,
        target_images: List[Image.Image],
        prompt: str,
        refinement_strength: float
    ) -> Gaussian:
        """
        Refine Gaussian Splatting attributes to match target images.
        
        This is the core refinement step that optimizes GS attributes
        to minimize rendering difference with MV-Adapter target images.
        """
        
        print(f"Refining GS with {len(target_images)} target images...")
        
        # Create a copy of the original GS for refinement
        refined_gs = self._clone_gaussian(original_gs)
        
        # Convert GS to validator's format
        gs_data = self._convert_gs_to_validator_format(refined_gs)
        
        # Convert target images to tensors
        target_tensors = self._images_to_tensors(target_images)
        
        # Setup optimization
        optimizer = torch.optim.Adam([
            {'params': [refined_gs._features_dc], 'lr': self.learning_rate},
            {'params': [refined_gs._features_rest], 'lr': self.learning_rate * 0.5},
            {'params': [refined_gs._opacity], 'lr': self.learning_rate * 0.1},
        ])
        
        # Refinement loop
        for step in range(self.refinement_steps):
            optimizer.zero_grad()
            
            total_loss = 0.0
            
            # Render GS using validator's renderer (BETTER than MV-Adapter's)
            rendered_images = self.validator_renderer.render_gs(
                gs_data=gs_data,
                views_number=min(self.num_views, len(target_tensors)),
                img_width=self.image_size,
                img_height=self.image_size,
                theta_angles=[0, 90, 180, 270, 180, 180],  # Match MV-Adapter views
                phi_angles=[0, 0, 0, 0, 89.99, -89.99],   # Match MV-Adapter views
                cam_rad=1.8,  # Match MV-Adapter distance
                cam_fov=49.1,
                cam_znear=0.01,
                cam_zfar=100.0,
                bg_color=torch.tensor([1, 1, 1], dtype=torch.float32).to(self.device)
            )
            
            # Compute loss against targets
            for view_idx, (rendered_image, target_image) in enumerate(zip(rendered_images, target_tensors)):
                # Convert rendered image to same format as target
                rendered_tensor = rendered_image.float() / 255.0  # Convert from uint8 to float
                rendered_tensor = rendered_tensor.permute(2, 0, 1).unsqueeze(0)  # CHW -> BCHW
                
                # Compute loss
                loss = self._compute_rendering_loss(rendered_tensor, target_image)
                total_loss += loss
            
            # Average loss across views
            total_loss = total_loss / len(rendered_images)
            
            # Apply refinement strength
            total_loss = total_loss * refinement_strength
            
            # Backward pass
            total_loss.backward()
            optimizer.step()
            
            # Update GS data for next iteration
            gs_data = self._convert_gs_to_validator_format(refined_gs)
            
            # Progress reporting
            if step % 100 == 0:
                print(f"Refinement step {step}/{self.refinement_steps}, Loss: {total_loss.item():.6f}")
        
        print("✅ GS refinement completed")
        return refined_gs
    
    def _convert_gs_to_validator_format(self, gs: Gaussian) -> GaussianSplattingData:
        """Convert Trellis Gaussian to validator's GaussianSplattingData format"""
        # Extract GS attributes
        points = gs.get_xyz.detach().cpu().numpy()
        rotations = gs.get_rotation.detach().cpu().numpy()
        scales = gs.get_scaling.detach().cpu().numpy()
        opacities = gs.get_opacity.detach().cpu().numpy()
        features_dc = gs._features_dc.detach().cpu().numpy()
        
        # Create validator's data structure
        gs_data = GaussianSplattingData(
            points=torch.from_numpy(points).to(self.device),
            rotations=torch.from_numpy(rotations).to(self.device),
            scales=torch.from_numpy(scales).to(self.device),
            opacities=torch.from_numpy(opacities).to(self.device),
            features_dc=torch.from_numpy(features_dc).to(self.device),
            features_rest=torch.from_numpy(gs._features_rest.detach().cpu().numpy()).to(self.device) if gs._features_rest is not None else None
        )
        
        return gs_data
    
    def _clone_gaussian(self, gs: Gaussian) -> Gaussian:
        """Create a deep copy of Gaussian Splatting for refinement"""
        # Create new Gaussian with same parameters
        cloned_gs = Gaussian(
            aabb=gs.init_params['aabb'],
            sh_degree=gs.sh_degree,
            mininum_kernel_size=gs.mininum_kernel_size,
            scaling_bias=gs.scaling_bias,
            opacity_bias=gs.opacity_bias,
            scaling_activation=gs.scaling_activation_type,
            device=gs.device
        )
        
        # Copy all attributes
        cloned_gs._xyz = gs._xyz.clone().detach().requires_grad_(True)
        cloned_gs._features_dc = gs._features_dc.clone().detach().requires_grad_(True)
        if gs._features_rest is not None:
            cloned_gs._features_rest = gs._features_rest.clone().detach().requires_grad_(True)
        cloned_gs._opacity = gs._opacity.clone().detach().requires_grad_(True)
        cloned_gs._scaling = gs._scaling.clone().detach()  # Keep fixed
        cloned_gs._rotation = gs._rotation.clone().detach()  # Keep fixed
        
        return cloned_gs
    
    def _images_to_tensors(self, images: List[Image.Image]) -> List[torch.Tensor]:
        """Convert PIL images to tensors"""
        tensors = []
        for img in images:
            # Convert PIL to tensor
            img_tensor = torch.from_numpy(np.array(img)).float() / 255.0
            img_tensor = img_tensor.permute(2, 0, 1).unsqueeze(0)  # BCHW format
            tensors.append(img_tensor.to(self.device))
        return tensors
    
    def _compute_rendering_loss(self, rendered: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute loss between rendered and target images"""
        # L1 loss for color
        l1_loss = torch.nn.functional.l1_loss(rendered, target)
        
        # SSIM loss for structural similarity
        ssim_loss = 1.0 - self._ssim(rendered, target)
        
        # Combined loss
        total_loss = l1_loss + 0.1 * ssim_loss
        
        return total_loss
    
    def _ssim(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Simplified SSIM computation"""
        # This is a simplified SSIM - in practice you'd use a proper SSIM implementation
        mu_x = torch.mean(x)
        mu_y = torch.mean(y)
        sigma_x = torch.var(x)
        sigma_y = torch.var(y)
        sigma_xy = torch.mean((x - mu_x) * (y - mu_y))
        
        c1 = 0.01 ** 2
        c2 = 0.03 ** 2
        
        ssim = ((2 * mu_x * mu_y + c1) * (2 * sigma_xy + c2)) / \
               ((mu_x ** 2 + mu_y ** 2 + c1) * (sigma_x + sigma_y + c2))
        
        return ssim
    
    def assess_gs_quality(self, gs_data: bytes, prompt: str) -> Dict[str, float]:
        """Assess the quality of generated GS using validation engine"""
        if self.validation_engine is None:
            return {"error": "Validation engine not initialized"}
        
        try:
            # Use validation engine to assess quality
            result = self.validation_engine.validate_ply(gs_data, prompt)
            return {
                "final_score": result.final_score,
                "quality_score": result.quality_score,
                "alignment_score": result.alignment_score,
                "ssim_score": result.ssim_score,
                "lpips_score": result.lpips_score
            }
        except Exception as e:
            return {"error": f"Validation failed: {str(e)}"}
    
    def save_refined_gs(self, refined_outputs: Dict[str, Any], output_path: str) -> str:
        """Save refined Gaussian Splatting to PLY file"""
        gs = refined_outputs['gaussian']
        
        # Save as PLY
        ply_buffer = io.BytesIO()
        gs.save_ply(ply_buffer)
        ply_data = ply_buffer.getvalue()
        
        # Write to file
        with open(output_path, 'wb') as f:
            f.write(ply_data)
        
        print(f"✅ Refined GS saved to {output_path}")
        return output_path


def create_refinement_server():
    """Create a FastAPI server that integrates the refinement pipeline"""
    from fastapi import FastAPI, HTTPException
    from pydantic import BaseModel
    import uvicorn
    
    app = FastAPI(title="Trellis GS Refinement Server")
    
    # Initialize refinement pipeline
    refinement_pipeline = TrellisGSRefinementPipeline(
        trellis_pipeline=None,  # Will be set by user
        mv_adapter_variant="sdxl",
        device="cuda",
        refinement_steps=1000,
        learning_rate=1e-3,
        enable_validation=True
    )
    
    class GenerationRequest(BaseModel):
        prompt: str
        seed: int = 42
        enable_refinement: bool = True
        refinement_strength: float = 1.0
    
    class GenerationResponse(BaseModel):
        success: bool
        message: str
        ply_data: Optional[bytes] = None
        quality_scores: Optional[Dict[str, float]] = None
    
    @app.post("/generate_refined/", response_model=GenerationResponse)
    async def generate_refined_model(request: GenerationRequest):
        try:
            # Generate refined model
            refined_outputs = refinement_pipeline.generate_refined_3d_model(
                prompt=request.prompt,
                seed=request.seed,
                enable_refinement=request.enable_refinement,
                refinement_strength=request.refinement_strength
            )
            
            # Save to temporary file
            temp_path = f"/tmp/refined_gs_{request.seed}.ply"
            ply_path = refinement_pipeline.save_refined_gs(refined_outputs, temp_path)
            
            # Read PLY data
            with open(ply_path, 'rb') as f:
                ply_data = f.read()
            
            # Assess quality
            quality_scores = refinement_pipeline.assess_gs_quality(ply_data, request.prompt)
            
            # Cleanup
            os.remove(temp_path)
            
            return GenerationResponse(
                success=True,
                message="Refined GS generated successfully",
                ply_data=ply_data,
                quality_scores=quality_scores
            )
            
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Generation failed: {str(e)}")
    
    @app.get("/health/")
    async def health_check():
        return {"status": "healthy", "pipeline_ready": refinement_pipeline.mv_pipeline is not None}
    
    return app


if __name__ == "__main__":
    # Example usage
    pipeline = TrellisGSRefinementPipeline(
        trellis_pipeline=None,  # Set your Trellis pipeline here
        mv_adapter_variant="sdxl",
        device="cuda",
        refinement_steps=500,  # Reduced for testing
        learning_rate=1e-3
    )
    
    # Test generation
    prompt = "A beautiful red sports car with chrome wheels"
    refined_outputs = pipeline.generate_refined_3d_model(
        prompt=prompt,
        seed=42,
        enable_refinement=True,
        refinement_strength=0.8
    )
    
    print("✅ Refinement pipeline test completed") 