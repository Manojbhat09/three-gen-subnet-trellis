#!/usr/bin/env python3
"""
HunyuanDiT Image Generation Server
Standalone server for image generation only
Used by CLIP optimization scripts
"""

import os
import time
import torch
import random
import base64
import io
from typing import Optional
from fastapi import FastAPI, Form, HTTPException
from fastapi.responses import JSONResponse
import uvicorn
import argparse

# Add Hunyuan3D path
HUNYUAN3D_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Hunyuan3D-2")
import sys
sys.path.append(HUNYUAN3D_PATH)

# Import HunyuanDiT
from hy3dgen.text2image import HunyuanDiTPipeline

# Configuration
CONFIG = {
    'hunyuan_model_path': "Tencent-Hunyuan/HunyuanDiT-v1.1-Diffusers-Distilled",
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    'default_num_inference_steps': 25,
    'default_guidance_scale': 7.5,
    'default_width': 1024,
    'default_height': 1024,
}

class HunyuanImageGenerator:
    def __init__(self):
        self.pipeline = None
        self.ready = False
        
        # Get HuggingFace token
        try:
            from huggingface_hub import HfFolder
            token = HfFolder.get_token()
            if token:
                os.environ["HUGGINGFACE_TOKEN"] = token
                print("✓ HuggingFace token loaded from cache")
            else:
                print("⚠️ No HuggingFace token found in cache")
        except Exception as e:
            print(f"⚠️ Error getting token from cache: {e}")
        
        print("🔧 HunyuanDiT Image Generator initialized")
    
    def load_pipeline(self):
        """Load HunyuanDiT pipeline"""
        if self.pipeline is not None:
            print("✓ HunyuanDiT pipeline already loaded")
            return
        
        print("🔧 Loading HunyuanDiT pipeline...")
        
        try:
            # Initialize HunyuanDiT pipeline
            self.pipeline = HunyuanDiTPipeline(
                model_path=CONFIG['hunyuan_model_path'],
                device=CONFIG['device']
            )
            
            # Compile for better performance
            try:
                print("   Compiling HunyuanDiT for better performance...")
                self.pipeline.compile()
                print("   ✓ HunyuanDiT compiled successfully")
            except Exception as e:
                print(f"   ⚠️ HunyuanDiT compilation failed: {e}")
                print("   Continuing without compilation...")
            
            self.ready = True
            print("✅ HunyuanDiT pipeline loaded successfully")
            
        except Exception as e:
            print(f"❌ HunyuanDiT pipeline loading failed: {e}")
            self.pipeline = None
            self.ready = False
    
    def unload_pipeline(self):
        """Unload pipeline to free GPU memory"""
        if self.pipeline is not None:
            print("🧹 Unloading HunyuanDiT pipeline...")
            del self.pipeline
            self.pipeline = None
            self.ready = False
            
            # Clear GPU memory
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            print("✅ HunyuanDiT pipeline unloaded")
    
    def generate_image(self, prompt: str, seed: Optional[int] = None, 
                      num_inference_steps: Optional[int] = None,
                      guidance_scale: Optional[float] = None) -> str:
        """Generate image and return base64 encoded string"""
        
        if not self.ready:
            self.load_pipeline()
        
        if seed is None:
            seed = random.randint(0, 2**31 - 1)
        
        if num_inference_steps is None:
            num_inference_steps = CONFIG['default_num_inference_steps']
        
        if guidance_scale is None:
            guidance_scale = CONFIG['default_guidance_scale']
        
        try:
            print(f"🎨 Generating image for: '{prompt}' (seed: {seed})")
            
            with torch.no_grad():
                image = self.pipeline(
                    prompt=prompt,
                    seed=seed,
                    num_inference_steps=num_inference_steps,
                    guidance_scale=guidance_scale
                )
            
            # Convert PIL Image to base64
            img_buffer = io.BytesIO()
            image.save(img_buffer, format='PNG')
            image_data = img_buffer.getvalue()
            image_base64 = base64.b64encode(image_data).decode('utf-8')
            
            print(f"✅ Image generated successfully ({len(image_data):,} bytes)")
            return image_base64
            
        except Exception as e:
            print(f"❌ Image generation failed: {e}")
            raise e

# Initialize FastAPI app
app = FastAPI(title="HunyuanDiT Image Generation Server", version="1.0.0")

# Initialize generator
generator = HunyuanImageGenerator()

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "HunyuanDiT Image Generation Server",
        "status": "running",
        "pipeline": "hunyuan_dit_image_only",
        "ready": generator.ready
    }

@app.post("/generate_image/")
async def generate_image_endpoint(
    prompt: str = Form(...), 
    seed: Optional[int] = Form(None),
    num_inference_steps: Optional[int] = Form(None),
    guidance_scale: Optional[float] = Form(None)
):
    """Generate image using HunyuanDiT."""
    
    try:
        image_base64 = generator.generate_image(
            prompt=prompt,
            seed=seed,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale
        )
        
        return JSONResponse(content={
            "status": "success",
            "prompt": prompt,
            "seed": seed or random.randint(0, 2**31 - 1),
            "image": image_base64,
            "pipeline": "hunyuan_dit_only"
        })
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Image generation failed: {str(e)}")

@app.get("/health/")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy", 
        "timestamp": time.time(),
        "ready": generator.ready,
        "pipeline_loaded": generator.pipeline is not None
    }

@app.post("/models/load/")
async def load_models():
    """Manually load models"""
    try:
        generator.load_pipeline()
        return {
            "status": "success",
            "message": "Models loaded successfully",
            "ready": generator.ready
        }
    except Exception as e:
        return JSONResponse(content={
            "status": "error",
            "message": str(e)
        }, status_code=500)

@app.post("/models/unload/")
async def unload_models():
    """Manually unload models to free GPU memory"""
    try:
        generator.unload_pipeline()
        return {
            "status": "success",
            "message": "Models unloaded successfully"
        }
    except Exception as e:
        return JSONResponse(content={
            "status": "error",
            "message": str(e)
        }, status_code=500)

@app.get("/config/")
async def get_config():
    """Get current configuration"""
    return CONFIG

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="HunyuanDiT Image Generation Server")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind to")
    parser.add_argument("--auto-load", action="store_true", help="Auto-load models on startup")
    
    args = parser.parse_args()
    
    print(f"Starting HunyuanDiT Image Generation Server on {args.host}:{args.port}")
    print("=" * 60)
    print("Pipeline: Text → HunyuanDiT → Image")
    print("Features:")
    print("  • HunyuanDiT text-to-image generation")
    print("  • Base64 encoded image responses")
    print("  • Optimized for CLIP scoring")
    print("=" * 60)
    
    if args.auto_load:
        print("🔧 Auto-loading models...")
        generator.load_pipeline()
    
    uvicorn.run(
        app, 
        host=args.host, 
        port=args.port,
        log_level="info"
    ) 