"""
Integration script for Trellis GS Refinement Pipeline

This script shows how to integrate the refinement pipeline into the existing Trellis server
to enhance Gaussian Splatting quality using MV-Adapter's multi-view images as targets.
"""

import os
import sys
import torch
import io
from typing import Optional, Dict, Any

# Add paths
sys.path.append('./TRELLIS')
sys.path.append('./validation')

from trellis_gs_refinement_pipeline import TrellisGSRefinementPipeline


class TrellisRefinementGenerator:
    """
    Enhanced Trellis generator that incorporates GS refinement using MV-Adapter.
    
    This class wraps the existing Trellis pipeline and adds refinement capabilities
    to improve the quality of generated Gaussian Splatting.
    """
    
    def __init__(
        self,
        trellis_pipeline,
        mv_adapter_variant: str = "sdxl",
        device: str = "cuda",
        refinement_steps: int = 1000,
        learning_rate: float = 1e-3,
        enable_refinement: bool = True,
        enable_validation: bool = True
    ):
        self.trellis_pipeline = trellis_pipeline
        self.enable_refinement = enable_refinement
        
        # Initialize refinement pipeline
        self.refinement_pipeline = TrellisGSRefinementPipeline(
            trellis_pipeline=trellis_pipeline,
            mv_adapter_variant=mv_adapter_variant,
            device=device,
            refinement_steps=refinement_steps,
            learning_rate=learning_rate,
            enable_validation=enable_validation
        )
        
        print("✅ Trellis Refinement Generator initialized")
    
    def generate_3d_model(
        self, 
        prompt: str, 
        seed: int = 42,
        enable_refinement: Optional[bool] = None,
        refinement_strength: float = 1.0
    ) -> Dict[str, Any]:
        """
        Generate 3D model with optional GS refinement.
        
        Args:
            prompt: Text prompt for generation
            seed: Random seed for reproducibility
            enable_refinement: Whether to apply refinement (overrides instance setting)
            refinement_strength: Strength of refinement (0.0 to 1.0)
            
        Returns:
            Dictionary containing GS, mesh, and metadata
        """
        
        # Determine if refinement should be applied
        should_refine = enable_refinement if enable_refinement is not None else self.enable_refinement
        
        if not should_refine:
            # Use original Trellis generation
            print("Using original Trellis generation (no refinement)")
            return self.trellis_pipeline.generate_3d_model(prompt, seed)
        
        # Use refinement pipeline
        print("Using enhanced generation with GS refinement")
        return self.refinement_pipeline.generate_refined_3d_model(
            prompt=prompt,
            seed=seed,
            enable_refinement=True,
            refinement_strength=refinement_strength
        )
    
    def assess_quality(self, gs_data: bytes, prompt: str) -> Dict[str, float]:
        """Assess the quality of generated GS"""
        return self.refinement_pipeline.assess_gs_quality(gs_data, prompt)


def integrate_with_trellis_server():
    """
    Example of how to integrate the refinement pipeline with existing Trellis server.
    
    This shows the modifications needed to trellis_base_server.py
    """
    
    # Example integration code for trellis_base_server.py
    integration_code = '''
# Add this import at the top of trellis_base_server.py
from trellis_refinement_integration import TrellisRefinementGenerator

class TrellisBaseGenerator:
    def __init__(self):
        # ... existing initialization code ...
        
        # Add refinement generator
        self.refinement_generator = None
        self.enable_refinement = True  # Can be controlled via config
        
    def _setup_refinement_generator(self):
        """Setup the refinement generator if not already initialized"""
        if self.trellis_pipeline is not None and self.refinement_generator is None:
            try:
                self.refinement_generator = TrellisRefinementGenerator(
                    trellis_pipeline=self.trellis_pipeline,
                    mv_adapter_variant="sdxl",  # or "sd21" for lower VRAM
                    device="cuda",
                    refinement_steps=1000,
                    learning_rate=1e-3,
                    enable_refinement=self.enable_refinement,
                    enable_validation=True
                )
                print("✅ Refinement generator initialized")
            except Exception as e:
                print(f"⚠️ Refinement generator setup failed: {e}")
                self.refinement_generator = None
    
    def generate_3d_model(self, prompt: str, seed: int = 42, enable_refinement: bool = None):
        # ... existing code until after Trellis generation ...
        
        # NEW: Enhanced generation with refinement
        if self.refinement_generator is not None and (enable_refinement or self.enable_refinement):
            print("Step 1.5: Applying GS refinement with MV-Adapter...")
            
            # Generate with refinement
            refined_outputs = self.refinement_generator.generate_3d_model(
                prompt=prompt,
                seed=seed,
                enable_refinement=True,
                refinement_strength=1.0
            )
            
            # Use refined outputs
            gaussian_output = refined_outputs['gaussian']
            mesh_output = refined_outputs['mesh']
            
            print("✅ GS refinement completed")
        else:
            # Use original Trellis outputs
            gaussian_output = outputs['gaussian'][0]
            mesh_output = outputs['mesh'][0]
        
        # ... rest of existing code (PLY extraction, GLB, video, SPZ compression) ...
    '''
    
    return integration_code


def create_enhanced_server():
    """
    Create an enhanced FastAPI server with refinement capabilities.
    """
    from fastapi import FastAPI, HTTPException
    from pydantic import BaseModel
    import uvicorn
    
    app = FastAPI(title="Enhanced Trellis Server with GS Refinement")
    
    # Initialize enhanced generator
    # Note: You need to provide your actual Trellis pipeline here
    trellis_pipeline = None  # Replace with your Trellis pipeline
    
    enhanced_generator = TrellisRefinementGenerator(
        trellis_pipeline=trellis_pipeline,
        mv_adapter_variant="sdxl",
        device="cuda",
        refinement_steps=1000,
        learning_rate=1e-3,
        enable_refinement=True,
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
        glb_data: Optional[bytes] = None
        quality_scores: Optional[Dict[str, float]] = None
    
    @app.post("/generate/", response_model=GenerationResponse)
    async def generate_3d_model(request: GenerationRequest):
        try:
            # Generate enhanced 3D model
            outputs = enhanced_generator.generate_3d_model(
                prompt=request.prompt,
                seed=request.seed,
                enable_refinement=request.enable_refinement,
                refinement_strength=request.refinement_strength
            )
            
            # Extract PLY data
            gaussian_output = outputs['gaussian']
            ply_buffer = io.BytesIO()
            gaussian_output.save_ply(ply_buffer)
            ply_data = ply_buffer.getvalue()
            
            # Assess quality
            quality_scores = enhanced_generator.assess_quality(ply_data, request.prompt)
            
            # Extract GLB data if available
            glb_data = None
            if 'mesh' in outputs and outputs['mesh'] is not None:
                # Convert mesh to GLB (you'd implement this based on your mesh format)
                pass
            
            return GenerationResponse(
                success=True,
                message="Enhanced 3D model generated successfully",
                ply_data=ply_data,
                glb_data=glb_data,
                quality_scores=quality_scores
            )
            
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Generation failed: {str(e)}")
    
    @app.get("/health/")
    async def health_check():
        return {
            "status": "healthy",
            "refinement_ready": enhanced_generator.refinement_pipeline.mv_pipeline is not None,
            "trellis_ready": enhanced_generator.trellis_pipeline is not None
        }
    
    return app


def test_refinement_pipeline():
    """
    Test the refinement pipeline with a sample prompt.
    """
    print("Testing Trellis GS Refinement Pipeline...")
    
    # Note: This requires a working Trellis pipeline
    # You would replace this with your actual Trellis pipeline
    trellis_pipeline = None  # Replace with your Trellis pipeline
    
    if trellis_pipeline is None:
        print("⚠️ No Trellis pipeline provided. Skipping test.")
        return
    
    # Initialize enhanced generator
    enhanced_generator = TrellisRefinementGenerator(
        trellis_pipeline=trellis_pipeline,
        mv_adapter_variant="sdxl",
        device="cuda",
        refinement_steps=500,  # Reduced for testing
        learning_rate=1e-3,
        enable_refinement=True,
        enable_validation=True
    )
    
    # Test generation
    prompt = "A beautiful red sports car with chrome wheels"
    
    print(f"Testing with prompt: '{prompt}'")
    
    # Generate with refinement
    outputs = enhanced_generator.generate_3d_model(
        prompt=prompt,
        seed=42,
        enable_refinement=True,
        refinement_strength=0.8
    )
    
    # Assess quality
    gaussian_output = outputs['gaussian']
    ply_buffer = io.BytesIO()
    gaussian_output.save_ply(ply_buffer)
    ply_data = ply_buffer.getvalue()
    
    quality_scores = enhanced_generator.assess_quality(ply_data, prompt)
    
    print("✅ Test completed successfully")
    print(f"Quality scores: {quality_scores}")
    
    return outputs, quality_scores


if __name__ == "__main__":
    # Show integration code
    print("=== Trellis GS Refinement Integration ===")
    print()
    print("Integration code for trellis_base_server.py:")
    print(integrate_with_trellis_server())
    print()
    print("To use this integration:")
    print("1. Copy the integration code to your trellis_base_server.py")
    print("2. Initialize TrellisRefinementGenerator in __init__")
    print("3. Call _setup_refinement_generator() after Trellis pipeline setup")
    print("4. Modify generate_3d_model() to use refinement when enabled")
    print()
    print("The refinement pipeline will:")
    print("- Generate initial GS and mesh with Trellis")
    print("- Use MV-Adapter to create high-quality target images")
    print("- Refine GS attributes to match target images")
    print("- Preserve 3D structure while improving visual quality")
    print()
    print("This approach is NOT redundant because:")
    print("- MV-Adapter provides high-quality appearance targets")
    print("- GS refinement preserves 3D structure from Trellis")
    print("- No image rendering/reconstruction cycle")
    print("- Direct attribute optimization for better quality scores") 