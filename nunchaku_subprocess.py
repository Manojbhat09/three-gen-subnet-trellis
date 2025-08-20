#!/usr/bin/env python3
"""
Subprocess-based Nunchaku integration for trellis_subnit_server_mix_lora_flash_nun.py
This runs Nunchaku in its own conda environment and returns the result.
"""

import subprocess
import tempfile
import os
import json
from PIL import Image
from typing import Optional
import base64
import io

def generate_nunchaku_image_subprocess(prompt: str, seed: int = 42, width: int = 1024, height: int = 1024) -> Optional[Image.Image]:
    """
    Generate image using Nunchaku via subprocess in the 'nun' conda environment
    """
    try:
        print(f"🎨 Generating Nunchaku image via subprocess: '{prompt}' (seed: {seed})")
        
        # Create a temporary Python script to run Nunchaku
        script_content = f'''
import os
import gc
import torch
from PIL import Image
import base64
import io
import json

# Set environment variables for Nunchaku
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = "expandable_segments:True"
os.environ["TOKENIZERS_PARALLELISM"] = "True"
torch._dynamo.config.suppress_errors = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.enabled = True

try:
    from nunchaku.models.transformers.transformer_flux import NunchakuFluxTransformer2dModel
    from diffusers import FluxPipeline
    
    # Clear GPU memory
    gc.collect()
    torch.cuda.empty_cache()
    
    # Load transformer
    transformer = NunchakuFluxTransformer2dModel.from_pretrained(
        "mit-han-lab/svdq-int4-flux.1-schnell",
        torch_dtype=torch.bfloat16
    )
    
    # Load pipeline
    pipeline = FluxPipeline.from_pretrained(
        "manbeast3b/flux.1-schnell-full1",
        revision="cb1b599b0d712b9aab2c4df3ad27b050a27ec146",
        transformer=transformer,
        torch_dtype=torch.bfloat16
    )
    
    # Move to GPU
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
    
    # Convert to base64
    img_buffer = io.BytesIO()
    image.save(img_buffer, format='PNG')
    img_data = img_buffer.getvalue()
    img_base64 = base64.b64encode(img_data).decode('utf-8')
    
    # Return result as JSON
    result = {{
        "status": "success",
        "image_base64": img_base64,
        "width": {width},
        "height": {height}
    }}
    
    print(json.dumps(result))
    
except Exception as e:
    result = {{
        "status": "error",
        "error": str(e)
    }}
    print(json.dumps(result))
'''
        
        # Write script to temporary file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(script_content)
            script_path = f.name
        
        try:
            # Run the script in the 'nun' conda environment
            cmd = f"conda run -n nun python {script_path}"
            result = subprocess.run(
                cmd,
                shell=True,
                capture_output=True,
                text=True,
                timeout=120  # 2 minute timeout
            )
            
            if result.returncode == 0:
                # Parse the JSON output
                try:
                    output_lines = result.stdout.strip().split('\n')
                    json_line = output_lines[-1]  # Get the last line which should be JSON
                    
                    if json_line.startswith('{'):
                        data = json.loads(json_line)
                        
                        if data['status'] == 'success':
                            # Decode base64 image
                            img_data = base64.b64decode(data['image_base64'])
                            image = Image.open(io.BytesIO(img_data))
                            print("✅ Nunchaku image generated successfully via subprocess")
                            return image
                        else:
                            print(f"❌ Nunchaku subprocess failed: {data.get('error', 'Unknown error')}")
                            return None
                    else:
                        print(f"❌ Unexpected output format: {json_line}")
                        return None
                        
                except json.JSONDecodeError as e:
                    print(f"❌ Failed to parse JSON output: {e}")
                    print(f"   Raw output: {result.stdout}")
                    return None
            else:
                print(f"❌ Nunchaku subprocess failed with return code {result.returncode}")
                print(f"   stdout: {result.stdout}")
                print(f"   stderr: {result.stderr}")
                return None
                
        finally:
            # Clean up temporary file
            os.unlink(script_path)
            
    except Exception as e:
        print(f"❌ Nunchaku subprocess execution failed: {e}")
        return None

if __name__ == "__main__":
    # Test the subprocess integration
    print("🧪 Testing Nunchaku subprocess integration...")
    
    image = generate_nunchaku_image_subprocess("A beautiful sunset over mountains", seed=42)
    
    if image:
        print("✅ Test successful! Saving test image...")
        image.save("nunchaku_subprocess_test.png")
        print("💾 Test image saved as 'nunchaku_subprocess_test.png'")
    else:
        print("❌ Test failed!")
