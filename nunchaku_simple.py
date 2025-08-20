#!/usr/bin/env python3
"""
Simple Nunchaku integration for trellis_subnit_server_mix_lora_flash_nun.py
This provides a direct function call interface instead of socket communication.
"""

import os
import gc
import torch
from PIL import Image
from typing import Optional

# Set environment variables for Nunchaku
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = "expandable_segments:True"
os.environ["TOKENIZERS_PARALLELISM"] = "True"
torch._dynamo.config.suppress_errors = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.enabled = True

class NunchakuSimpleGenerator:
    """Simple Nunchaku generator that can be used directly in the main server"""
    
    def __init__(self):
        self.pipeline = None
        self.transformer = None
        self.initialized = False
        
    def initialize(self):
        """Initialize the Nunchaku pipeline"""
        if self.initialized:
            return True
            
        try:
            print("🔧 Initializing Nunchaku pipeline...")
            
            # Import Nunchaku components
            from nunchaku.models.transformers.transformer_flux import NunchakuFluxTransformer2dModel
            from diffusers import FluxPipeline
            
            # Clear GPU memory
            gc.collect()
            torch.cuda.empty_cache()
            
            # Load transformer
            self.transformer = NunchakuFluxTransformer2dModel.from_pretrained(
                "mit-han-lab/svdq-int4-flux.1-schnell",
                torch_dtype=torch.bfloat16
            )
            
            # Load pipeline
            self.pipeline = FluxPipeline.from_pretrained(
                "manbeast3b/flux.1-schnell-full1",
                revision="cb1b599b0d712b9aab2c4df3ad27b050a27ec146",
                transformer=self.transformer,
                torch_dtype=torch.bfloat16
            )
            
            # Move to GPU
            self.pipeline.to("cuda", memory_format=torch.channels_last)
            
            # Warmup call
            print("   Warming up Nunchaku pipeline...")
            _ = self.pipeline(
                prompt="A cat holding a sign that says hello world",
                width=1024,
                height=1024,
                guidance_scale=0.0,
                num_inference_steps=4,
                max_sequence_length=256,
                output_type="pil"
            )
            
            self.initialized = True
            print("✅ Nunchaku pipeline initialized successfully")
            return True
            
        except Exception as e:
            print(f"❌ Failed to initialize Nunchaku pipeline: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def generate_image(self, prompt: str, seed: int = 42, width: int = 1024, height: int = 1024) -> Optional[Image.Image]:
        """Generate image using Nunchaku"""
        if not self.initialized:
            if not self.initialize():
                return None
        
        try:
            print(f"🎨 Generating Nunchaku image: '{prompt}' (seed: {seed})")
            
            # Clear GPU memory before generation
            gc.collect()
            torch.cuda.empty_cache()
            
            # Generate image
            with torch.no_grad():
                image = self.pipeline(
                    prompt=prompt,
                    width=width,
                    height=height,
                    guidance_scale=0.0,
                    num_inference_steps=4,
                    max_sequence_length=256,
                    output_type="pil"
                ).images[0]
            
            print("✅ Nunchaku image generated successfully")
            return image
            
        except Exception as e:
            print(f"❌ Nunchaku image generation failed: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def cleanup(self):
        """Clean up GPU memory"""
        if self.pipeline is not None:
            del self.pipeline
            self.pipeline = None
        
        if self.transformer is not None:
            del self.transformer
            self.transformer = None
        
        gc.collect()
        torch.cuda.empty_cache()
        self.initialized = False
        print("🧹 Nunchaku pipeline cleaned up")

# Global instance
nunchaku_generator = NunchakuSimpleGenerator()

def generate_nunchaku_image(prompt: str, seed: int = 42, width: int = 1024, height: int = 1024) -> Optional[Image.Image]:
    """Simple function to generate Nunchaku images"""
    return nunchaku_generator.generate_image(prompt, seed, width, height)

def cleanup_nunchaku():
    """Clean up Nunchaku resources"""
    nunchaku_generator.cleanup()

if __name__ == "__main__":
    # Test the integration
    print("🧪 Testing Nunchaku integration...")
    
    # Test image generation
    image = generate_nunchaku_image("A futuristic robot in a cyberpunk city", seed=42)
    
    if image:
        print("✅ Test successful! Saving test image...")
        image.save("nunchaku_test.png")
        print("💾 Test image saved as 'nunchaku_test.png'")
    else:
        print("❌ Test failed!")
    
    # Cleanup
    cleanup_nunchaku()
