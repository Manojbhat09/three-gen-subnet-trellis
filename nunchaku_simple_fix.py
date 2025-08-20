#!/usr/bin/env python3
"""
Simple Nunchaku fix - just run the existing working script as subprocess
"""

import subprocess
import tempfile
import os
from PIL import Image
from typing import Optional

def generate_nunchaku_image_simple(prompt: str, seed: int = 42, width: int = 1024, height: int = 1024) -> Optional[Image.Image]:
    """Generate image using existing working test_nunchaku2.py"""
    try:
        print(f"🎨 Generating Nunchaku image via subprocess: '{prompt}' (seed: {seed})")
        
        # Create a simple script that just calls the existing working code
        script_content = f'''
import os
import gc
import torch
from nunchaku.models.transformers.transformer_flux import NunchakuFluxTransformer2dModel
from diffusers import FluxPipeline

os.environ['PYTORCH_CUDA_ALLOC_CONF'] = "expandable_segments:True"
os.environ["TOKENIZERS_PARALLELISM"] = "True"
torch._dynamo.config.suppress_errors = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.enabled = True

gc.collect()
torch.cuda.empty_cache()

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

# Generate image
with torch.no_grad():
    image = pipeline(
        prompt="{prompt}",
        width={width},
        height={height},
        guidance_scale=0.0,
        num_inference_steps=4,
        max_sequence_length=256,
        output_type="pil"
    ).images[0]

# Save to temp file
temp_path = "/tmp/nunchaku_temp.png"
image.save(temp_path)
print(f"SAVED:{temp_path}")
'''
        
        # Write script to temporary file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(script_content)
            script_path = f.name
        
        try:
            # Run in nun environment
            cmd = f"conda run -n nun python {script_path}"
            result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=60)
            
            if result.returncode == 0:
                # Look for the SAVED: line
                for line in result.stdout.split('\n'):
                    if line.startswith('SAVED:'):
                        temp_path = line.split('SAVED:')[1].strip()
                        if os.path.exists(temp_path):
                            # Load the image
                            image = Image.open(temp_path)
                            # Clean up
                            os.unlink(temp_path)
                            print("✅ Nunchaku image generated successfully")
                            return image
                
                print("❌ No image file found in output")
                return None
            else:
                print(f"❌ Subprocess failed: {result.stderr}")
                return None
                
        finally:
            # Clean up script
            os.unlink(script_path)
            
    except Exception as e:
        print(f"❌ Nunchaku generation failed: {e}")
        return None

if __name__ == "__main__":
    # Test
    image = generate_nunchaku_image_simple("A red sports car", seed=42)
    if image:
        image.save("test_output.png")
        print("✅ Test successful!")
    else:
        print("❌ Test failed!")
