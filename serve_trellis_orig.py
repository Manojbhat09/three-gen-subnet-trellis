import os
import sys
import base64
import io
import imageio
import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# # Add TRELLIS to Python path
# TRELLIS_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "TRELLIS")
# sys.path.append(TRELLIS_PATH)
# Add TRELLIS to Python path
TRELLIS_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "TRELLIS")
sys.path.append(TRELLIS_PATH)


from trellis.pipelines import TrellisTextTo3DPipeline
from trellis.utils import render_utils, postprocessing_utils

# Set spconv algorithm to native for single run
os.environ['SPCONV_ALGO'] = 'native'
os.environ['ATTN_BACKEND'] = 'xformers'  # Use xformers instead of flash-attn

app = FastAPI()

# Initialize the TRELLIS pipeline
print("Loading TRELLIS pipeline...")
pipeline = TrellisTextTo3DPipeline.from_pretrained("microsoft/TRELLIS-text-xlarge")
pipeline.cuda()

class GenerationRequest(BaseModel):
    prompt: str

def encode_ply_to_base64(gaussian_output) -> str:
    """Convert Gaussian output to base64-encoded PLY string."""
    # Save to a bytes buffer
    buffer = io.BytesIO()
    gaussian_output.save_ply(buffer)
    buffer.seek(0)
    # Convert to base64
    return base64.b64encode(buffer.read()).decode('utf-8')

@app.post("/generate/")
async def generate(request: GenerationRequest):
    try:
        # Generate 3D model
        outputs = pipeline.run(
            request.prompt,
            seed=42,
            sparse_structure_sampler_params={
                "steps": 12,
                "cfg_strength": 7.5,
            },
            slat_sampler_params={
                "steps": 12,
                "cfg_strength": 7.5,
            },
        )
        
        # Convert Gaussian output to base64-encoded PLY
        ply_base64 = encode_ply_to_base64(outputs['gaussian'][0])
        
        return {
            "status": "success",
            "data": ply_base64,
            "format": "ply",
            "compression": 0  # No compression
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/generate_video/")
async def generate_video(request: GenerationRequest):
    try:
        # Generate 3D model
        outputs = pipeline.run(
            request.prompt,
            seed=42,
            sparse_structure_sampler_params={
                "steps": 12,
                "cfg_strength": 7.5,
            },
            slat_sampler_params={
                "steps": 12,
                "cfg_strength": 7.5,
            },
        )
        
        # Render video
        video = render_utils.render_video(outputs['gaussian'][0])['color']
        video_geo = render_utils.render_video(outputs['mesh'][0])['normal']
        video = [torch.cat([video[i], video_geo[i]], dim=1) for i in range(len(video))]
        
        # Save to bytes buffer
        buffer = io.BytesIO()
        imageio.mimsave(buffer, video, fps=30, format='mp4')
        buffer.seek(0)
        
        return buffer.read()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8093) 