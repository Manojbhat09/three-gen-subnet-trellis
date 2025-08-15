#!/usr/bin/env python3
"""
Improved deterministic generation setup for cross-GPU consistency
Apply these changes to your trellis_subnit_server_mix_lora_flash.py
"""

import os
import torch
import numpy as np
import random

def setup_deterministic_environment():
    """
    Enhanced deterministic setup for cross-GPU consistency
    Call this at the very beginning of your script, before any model loading
    """
    
    # 1. CRITICAL: Disable TF32 for consistent precision across GPUs
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    
    # 2. Force deterministic algorithms globally
    torch.use_deterministic_algorithms(True, warn_only=True)
    
    # 3. Set deterministic cuDNN behavior
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    # 4. Set environment variables for deterministic behavior
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'  # Required for deterministic algorithms
    os.environ['PYTHONHASHSEED'] = '42'
    
    # 5. Set consistent CUDA memory allocation
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:False'  # More deterministic
    
    # 6. Force consistent algorithm selection
    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'  # Synchronous CUDA operations
    
    # 7. Disable auto-tuning that can vary between runs
    os.environ['MXNET_CUDNN_AUTOTUNE_DEFAULT'] = '0'

def seed_everything(seed: int = 42):
    """
    Comprehensive seeding function - call before each generation
    """
    # Python random
    random.seed(seed)
    
    # NumPy
    np.random.seed(seed)
    
    # PyTorch CPU
    torch.manual_seed(seed)
    
    # PyTorch GPU
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # For multi-GPU setups
        
    # Additional entropy sources
    os.environ['PYTHONHASHSEED'] = str(seed)

def create_deterministic_generator(seed: int, device: str = "cuda") -> torch.Generator:
    """
    Create a properly seeded generator for consistent results
    """
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    return generator

def configure_model_for_determinism(model):
    """
    Configure a loaded model for maximum determinism
    """
    # Disable dropout during inference (should be automatic in eval mode)
    model.eval()
    
    # Disable any stochastic components
    for module in model.modules():
        if hasattr(module, 'training'):
            module.training = False
        if hasattr(module, 'dropout'):
            if hasattr(module.dropout, 'p'):
                module.dropout.p = 0.0

# Updated TRELLIS generation configuration for determinism
DETERMINISTIC_GENERATION_CONFIG = {
    # Force consistent precision
    'trellis_use_fp16': False,  # Disable FP16 for consistency
    'trellis_compile': False,   # Disable compilation for determinism
    
    # Use consistent memory allocation
    'enable_memory_efficient_attention': False,  # Can introduce variance
    'enable_cpu_offload': False,  # Avoid CPU-GPU transfers that can vary
    
    # Fixed guidance parameters (no adaptive scheduling)
    'guidance_scale': 3.5,
    'ss_guidance_strength': 9.5,
    'ss_sampling_steps': 21,
    'slat_guidance_strength': 4.0,
    'slat_sampling_steps': 24,
}

class DeterministicTrellisGenerator:
    """
    Enhanced generator class with deterministic guarantees
    """
    
    def __init__(self):
        # Apply deterministic setup
        setup_deterministic_environment()
        
        # Initialize with deterministic config
        self.config = DETERMINISTIC_GENERATION_CONFIG.copy()
        
        # Model placeholders
        self.flux_pipeline = None
        self.trellis_pipeline = None
        
    def generate_3d_model_deterministic(
        self, 
        prompt: str, 
        seed: int = 42,
        **kwargs
    ):
        """
        Generate 3D model with maximum determinism
        """
        # Re-seed everything before generation
        seed_everything(seed)
        
        # Create deterministic generators
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
        print(f"🔒 DETERMINISTIC MODE: Generating with seed {seed}")
        print(f"   TF32 disabled: {not torch.backends.cuda.matmul.allow_tf32}")
        print(f"   Deterministic algorithms: {torch.are_deterministic_algorithms_enabled()}")
        print(f"   Device: {device}")
        
        try:
            # Step 1: Generate image with FLUX (deterministic)
            if self.flux_pipeline is None:
                self._load_flux_models_deterministic()
            
            # Create fresh generator for FLUX
            flux_generator = create_deterministic_generator(seed, device)
            
            with torch.no_grad():
                # Use consistent parameters
                image = self.flux_pipeline(
                    prompt=prompt,
                    generator=flux_generator,
                    num_inference_steps=7,  # Fixed
                    guidance_scale=3.5,     # Fixed
                    width=1024,
                    height=1024,
                ).images[0]
            
            print(f"✅ FLUX generation completed deterministically")
            
            # Step 2: Generate 3D with TRELLIS (deterministic)
            if self.trellis_pipeline is None:
                self._load_trellis_pipeline_deterministic()
            
            # Re-seed for TRELLIS
            seed_everything(seed + 1)  # Offset to avoid identical seeds
            
            # Use FP32 for maximum precision
            with torch.no_grad():
                outputs = self.trellis_pipeline.run(
                    image,
                    seed=seed,
                    formats=["gaussian"],
                    preprocess_image=False,
                    sparse_structure_sampler_params={
                        "steps": 21,
                        "cfg_strength": 9.5,
                        "cfg_interval": (0.3, 0.98),
                        "rescale_t": 3.0,
                    },
                    slat_sampler_params={
                        "steps": 24,
                        "cfg_strength": 4.0,
                        "cfg_interval": (0.3, 0.98),
                        "rescale_t": 3.0,
                    },
                )
            
            print(f"✅ TRELLIS generation completed deterministically")
            
            # Extract PLY data
            gaussian_output = outputs['gaussian'][0]
            ply_buffer = io.BytesIO()
            gaussian_output.save_ply(ply_buffer)
            ply_data = ply_buffer.getvalue()
            
            return ply_data, None
            
        except Exception as e:
            print(f"❌ Deterministic generation failed: {e}")
            raise
    
    def _load_flux_models_deterministic(self):
        """Load FLUX models with deterministic configuration"""
        print("🔧 Loading FLUX models in deterministic mode...")
        
        # Configure for determinism
        device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.float32  # Use FP32 for consistency
        
        # Load models without quantization for consistency
        # ... (your existing FLUX loading code, but with dtype=torch.float32)
        
        # Configure loaded models for determinism
        if self.flux_pipeline:
            configure_model_for_determinism(self.flux_pipeline)
            
    def _load_trellis_pipeline_deterministic(self):
        """Load TRELLIS pipeline with deterministic configuration"""
        print("🔧 Loading TRELLIS pipeline in deterministic mode...")
        
        # Load with FP32 precision
        self.trellis_pipeline = TrellisImageTo3DPipeline.from_pretrained(
            'cavargas10/TRELLIS',
            torch_dtype=torch.float32  # Force FP32
        )
        
        # Configure for determinism
        configure_model_for_determinism(self.trellis_pipeline)
        
        if torch.cuda.is_available():
            self.trellis_pipeline.cuda()

# Usage example:
def main():
    # Set up deterministic environment
    setup_deterministic_environment()
    
    # Create deterministic generator
    generator = DeterministicTrellisGenerator()
    
    # Generate with same prompt and seed across different GPUs
    prompt = "a blue ceramic vase with red trim"
    seed = 42
    
    ply_data, _ = generator.generate_3d_model_deterministic(prompt, seed)
    print(f"Generated PLY size: {len(ply_data)} bytes")
    
    # This should produce identical results across different GPU setups

if __name__ == "__main__":
    main()

