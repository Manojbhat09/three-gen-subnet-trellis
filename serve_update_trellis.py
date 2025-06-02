import os
import sys
import uuid
import time
from typing import Optional
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel
import uvicorn
import torch
import aiohttp

# Add TRELLIS to Python path
TRELLIS_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "TRELLIS")
sys.path.append(TRELLIS_PATH)

# Set spconv algorithm to native for single run and use xformers for attention
os.environ['SPCONV_ALGO'] = 'native'
os.environ['ATTN_BACKEND'] = 'xformers'  # Use xformers instead of flash-attn

import imageio
from trellis.pipelines import TrellisTextTo3DPipeline
from trellis.utils import render_utils, postprocessing_utils

app = FastAPI(title="Optimized TRELLIS API Server")

# Initialize the TRELLIS pipeline at startup
pipeline = None

class GenerationRequest(BaseModel):
    prompt: str
    seed: Optional[int] = 42
    sparse_structure_steps: Optional[int] = 8  # Optimized from 12
    sparse_structure_cfg: Optional[float] = 6.5  # Optimized from 7.5
    slat_steps: Optional[int] = 8  # Optimized from 12
    slat_cfg: Optional[float] = 6.5  # Optimized from 7.5
    use_optimized_prompt: Optional[bool] = True

def clear_gpu_memory():
    """Clear GPU memory cache"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()

def assess_quality(outputs) -> float:
    """Basic quality assessment of the generated outputs"""
    try:
        # Check if we have all required outputs
        if not all(k in outputs for k in ['gaussian', 'radiance_field', 'mesh']):
            return 0.0
            
        # Check if any outputs are None or empty
        if any(not outputs[k] for k in ['gaussian', 'radiance_field', 'mesh']):
            return 0.0
            
        # Check mesh quality (basic checks)
        mesh = outputs['mesh'][0]
        if mesh.vertices.shape[0] < 100 or mesh.faces.shape[0] < 100:  # Too simple mesh
            return 0.3
            
        # Check gaussian quality
        gaussian = outputs['gaussian'][0]
        if gaussian.get_xyz.shape[0] < 1000:  # Too few gaussian points
            return 0.4
            
        return 1.0  # Passed all basic quality checks
        
    except Exception as e:
        print(f"Quality assessment failed: {str(e)}")
        return 0.0

async def get_optimized_prompt(original_prompt: str) -> str:
    """Get an optimized version of the prompt from the prompt service"""
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get("https://prompts.404.xyz/optimize", 
                                 params={"prompt": original_prompt}) as response:
                if response.status == 200:
                    result = await response.json()
                    if "optimized_prompt" in result:
                        return result["optimized_prompt"]
    except Exception as e:
        print(f"Failed to get optimized prompt: {str(e)}")
    return original_prompt

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
        # Clear GPU memory before generation
        clear_gpu_memory()
        
        # Get optimized prompt if requested
        prompt = await get_optimized_prompt(request.prompt) if request.use_optimized_prompt else request.prompt
        
        start_time = time.time()
        
        # Generate the 3D model
        outputs = pipeline.run(
            prompt,
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
        
        generation_time = time.time() - start_time
        
        # Check generation time
        if generation_time > 25:  # Warning threshold
            print(f"WARNING: Generation took {generation_time:.2f}s, approaching 30s limit")
            
        # Assess quality
        quality_score = assess_quality(outputs)
        if quality_score < 0.5:
            raise HTTPException(status_code=422, detail=f"Generated output failed quality check (score: {quality_score})")
        
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
        
        # Clear GPU memory after generation
        clear_gpu_memory()
        
        return {
            "status": "success",
            "generation_id": generation_id,
            "generation_time": generation_time,
            "quality_score": quality_score,
            "used_prompt": prompt,
            "files": {
                "gaussian_video": f"/download/{generation_id}/preview_gaussian.mp4",
                "rf_video": f"/download/{generation_id}/preview_rf.mp4",
                "mesh_video": f"/download/{generation_id}/preview_mesh.mp4",
                "ply": f"/download/{generation_id}/model_gaussian.ply",
                "glb": f"/download/{generation_id}/model.glb"
            }
        }
        
    except Exception as e:
        # Clean up failed generation directory
        if os.path.exists(output_dir):
            import shutil
            shutil.rmtree(output_dir)
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
    return {
        "status": "healthy",
        "gpu_available": torch.cuda.is_available(),
        "gpu_memory_allocated": f"{torch.cuda.memory_allocated() / 1024**3:.2f}GB" if torch.cuda.is_available() else "N/A"
    }

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