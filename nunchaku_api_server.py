#!/usr/bin/env python3
"""
Simple Nunchaku API server - run this in the 'nun' environment
"""

from flask import Flask, request, jsonify
import os
import gc
import torch
from PIL import Image
import base64
import io
import json

app = Flask(__name__)

# Global pipeline
pipeline = None
transformer = None

def load_nunchaku():
    """Load Nunchaku pipeline"""
    global pipeline, transformer
    
    if pipeline is not None:
        return True
        
    try:
        print("🔧 Loading Nunchaku pipeline...")
        
        # Set environment variables
        os.environ['PYTORCH_CUDA_ALLOC_CONF'] = "expandable_segments:True"
        os.environ["TOKENIZERS_PARALLELISM"] = "True"
        torch._dynamo.config.suppress_errors = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.enabled = True
        
        # Clear GPU memory
        gc.collect()
        torch.cuda.empty_cache()
        
        # Import and load
        from nunchaku.models.transformers.transformer_flux import NunchakuFluxTransformer2dModel
        from diffusers import FluxPipeline
        
        transformer = NunchakuFluxTransformer2dModel.from_pretrained(
            "mit-han-lab/svdq-int4-flux.1-schnell",
            torch_dtype=torch.bfloat16
        )
        
        pipeline = FluxPipeline.from_pretrained(
            "manbeast3b/flux.1-schnell-full1",
            revision="cb1b599b0d712b9aab2c4df3ad27b050a27ec146",
            transformer=transformer,
            torch_dtype=torch.bfloat16
        )
        
        pipeline.to("cuda", memory_format=torch.channels_last)
        
        # Warmup
        print("   Warming up...")
        _ = pipeline(
            prompt="test",
            width=512,
            height=512,
            guidance_scale=0.0,
            num_inference_steps=4,
            max_sequence_length=256,
            output_type="pil"
        )
        
        print("✅ Nunchaku pipeline loaded successfully")
        return True
        
    except Exception as e:
        print(f"❌ Failed to load Nunchaku: {e}")
        return False

@app.route('/health', methods=['GET'])
def health():
    """Health check"""
    return jsonify({"status": "healthy", "pipeline_loaded": pipeline is not None})

@app.route('/generate', methods=['POST'])
def generate_image():
    """Generate image with Nunchaku"""
    try:
        # Debug logging
        print(f"🔍 Request content-type: {request.content_type}")
        print(f"🔍 Request headers: {dict(request.headers)}")
        print(f"🔍 Request form data: {dict(request.form)}")
        print(f"🔍 Request JSON: {request.get_json(silent=True)}")
        
        # Handle both JSON and form data
        if request.is_json:
            data = request.get_json()
            prompt = data.get('prompt', '')
            seed = data.get('seed', 42)
            width = data.get('width', 1024)
            height = data.get('height', 1024)
            print(f"🔍 Using JSON data: prompt='{prompt}', seed={seed}")
        else:
            # Handle form data
            prompt = request.form.get('prompt', '')
            seed = int(request.form.get('seed', 42))
            width = int(request.form.get('width', 1024))
            height = int(request.form.get('height', 1024))
            print(f"🔍 Using form data: prompt='{prompt}', seed={seed}")
        
        if not prompt:
            return jsonify({"error": "No prompt provided"}), 400
        
        # Ensure pipeline is loaded
        if not load_nunchaku():
            return jsonify({"error": "Failed to load Nunchaku pipeline"}), 500
        
        print(f"🎨 Generating: '{prompt}' (seed: {seed})")
        
        # Generate image
        gc.collect()
        torch.cuda.empty_cache()
        
        with torch.no_grad():
            image = pipeline(
                prompt=prompt,
                width=width,
                height=height,
                guidance_scale=0.0,
                num_inference_steps=4,
                max_sequence_length=256,
                output_type="pil"
            ).images[0]
        
        # Convert to base64
        img_buffer = io.BytesIO()
        image.save(img_buffer, format='PNG')
        img_data = img_buffer.getvalue()
        img_base64 = base64.b64encode(img_data).decode('utf-8')
        
        return jsonify({
            "status": "success",
            "prompt": prompt,
            "seed": seed,
            "width": width,
            "height": height,
            "image_base64": img_base64
        })
        
    except Exception as e:
        print(f"❌ Generation failed: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    print("🚀 Starting Nunchaku API server...")
    print("   This server must run in the 'nun' conda environment")
    
    # Load pipeline on startup
    if load_nunchaku():
        print("✅ Server ready!")
        app.run(host='0.0.0.0', port=8200, debug=False)
    else:
        print("❌ Failed to load Nunchaku pipeline")
        exit(1)
