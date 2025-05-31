import os
import sys
import uuid
from typing import Optional
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel
import uvicorn

# Add TRELLIS to Python path
TRELLIS_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "TRELLIS")
sys.path.append(TRELLIS_PATH)

# Set spconv algorithm to native for single run and use xformers for attention
os.environ['SPCONV_ALGO'] = 'native'
os.environ['ATTN_BACKEND'] = 'xformers'  # Use xformers instead of flash-attn

import imageio
from trellis.pipelines import TrellisTextTo3DPipeline
from trellis.utils import render_utils, postprocessing_utils

app = FastAPI(title="TRELLIS API Server")

# Initialize the TRELLIS pipeline at startup
pipeline = None

class GenerationRequest(BaseModel):
    prompt: str
    seed: Optional[int] = 42
    sparse_structure_steps: Optional[int] = 12
    sparse_structure_cfg: Optional[float] = 7.5
    slat_steps: Optional[int] = 12
    slat_cfg: Optional[float] = 7.5

@app.on_event("startup")
async def startup_event():
    global pipeline
    print("Loading TRELLIS pipeline...")
    pipeline = TrellisTextTo3DPipeline.from_pretrained("microsoft/TRELLIS-text-xlarge")
    pipeline.cuda()  # Move to GPU
    print("TRELLIS pipeline loaded and ready!")

@app.post("/generate")
async def generate_3d(request: GenerationRequest):
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")
    
    # Create a unique ID for this generation
    generation_id = str(uuid.uuid4())
    output_dir = os.path.join("outputs", generation_id)
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        # Generate the 3D model
        outputs = pipeline.run(
            request.prompt,
            seed=request.seed,
            sparse_structure_sampler_params={
                "steps": request.sparse_structure_steps,
                "cfg_strength": request.sparse_structure_cfg,
            },
            slat_sampler_params={
                "steps": request.slat_steps,
                "cfg_strength": request.slat_cfg,
            },
        )
        
        # Save preview videos
        video = render_utils.render_video(outputs['gaussian'][0])['color']
        gaussian_video_path = os.path.join(output_dir, "preview_gaussian.mp4")
        imageio.mimsave(gaussian_video_path, video, fps=30)
        
        video = render_utils.render_video(outputs['radiance_field'][0])['color']
        rf_video_path = os.path.join(output_dir, "preview_rf.mp4")
        imageio.mimsave(rf_video_path, video, fps=30)
        
        video = render_utils.render_video(outputs['mesh'][0])['normal']
        mesh_video_path = os.path.join(output_dir, "preview_mesh.mp4")
        imageio.mimsave(mesh_video_path, video, fps=30)
        
        # Save PLY file (3D Gaussians)
        ply_path = os.path.join(output_dir, "model_gaussian.ply")
        outputs['gaussian'][0].save_ply(ply_path)
        
        # Save GLB file (textured mesh)
        glb = postprocessing_utils.to_glb(
            outputs['gaussian'][0],
            outputs['mesh'][0],
            simplify=0.95,
            texture_size=1024,
        )
        glb_path = os.path.join(output_dir, "model.glb")
        glb.export(glb_path)
        
        return {
            "status": "success",
            "generation_id": generation_id,
            "files": {
                "gaussian_video": f"/download/{generation_id}/preview_gaussian.mp4",
                "rf_video": f"/download/{generation_id}/preview_rf.mp4",
                "mesh_video": f"/download/{generation_id}/preview_mesh.mp4",
                "ply": f"/download/{generation_id}/model_gaussian.ply",
                "glb": f"/download/{generation_id}/model.glb"
            }
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/download/{generation_id}/{filename}")
async def download_file(generation_id: str, filename: str):
    file_path = os.path.join("outputs", generation_id, filename)
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="File not found")
    return FileResponse(file_path)

@app.get("/health")
async def health_check():
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")
    return {"status": "healthy"}

def main():
    # Create outputs directory
    os.makedirs("outputs", exist_ok=True)
    
    # Run the server
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info"
    )

if __name__ == "__main__":
    main() 