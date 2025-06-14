import argparse
import gc
import io
import time
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from typing import cast

import numpy as np
import pybase64
import pyspz
import torch
import uvicorn
import zstandard
from engine.data_structures import (
    GaussianSplattingData,
    RequestData,
    ResponseData,
    TimeStat,
    ValidationResult,
    ValidationResultData,
)
from engine.io.ply import PlyLoader
from engine.rendering.renderer import Renderer
from engine.utils.gs_data_checker_utils import is_input_data_valid
from engine.validation_engine import ValidationEngine
from fastapi import FastAPI
from loguru import logger
from PIL import Image


VERSION = "2.0.0"


def get_args() -> tuple[argparse.Namespace, list[str]]:
    """Function for handling input arguments related to running the server"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=10006)
    args, extras = parser.parse_known_args()
    return args, extras


app = FastAPI()
args, _ = get_args()


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator:
    """Function for initializing all pipelines"""
    # Startup logic
    app.state.validator = ValidationEngine()
    app.state.validator.load_pipelines()
    app.state.zstd_decompressor = zstandard.ZstdDecompressor()
    app.state.renderer = Renderer()
    app.state.ply_data_loader = PlyLoader()
    
    # Enhanced GPU memory management
    _enhanced_cleanup()
    yield
    
    # Cleanup on shutdown
    _enhanced_cleanup()


app.router.lifespan_context = lifespan


def _prepare_input_data(
    assets: bytes, renderer: Renderer, ply_data_loader: PlyLoader, validator: ValidationEngine
) -> tuple[GaussianSplattingData | None, list[torch.Tensor], TimeStat]:
    """Function for preparing input data for further processing"""

    time_stat = TimeStat()
    # Loading input data
    t1 = time.time()
    pcl_buffer = io.BytesIO(assets)
    gs_data: GaussianSplattingData = ply_data_loader.from_buffer(pcl_buffer)
    t2 = time.time()
    time_stat.loading_data_time = t2 - t1
    logger.info(f"Loading data took: {time_stat.loading_data_time} sec.")

    # Check required memory
    if not is_input_data_valid(gs_data):
        return None, [], time_stat

    # Render images for validation
    gs_data_gpu = gs_data.send_to_device(validator.device)
    images = renderer.render_gs(gs_data_gpu, 16, 224, 224)
    t3 = time.time()
    time_stat.image_rendering_time = t3 - t2
    logger.info(f"Image Rendering took: {time_stat.image_rendering_time} sec.")
    return gs_data_gpu, images, time_stat


def _render_preview_image(
    gs_data: GaussianSplattingData, validation_score: float, preview_score_threshold: float, renderer: Renderer
) -> str | None:
    """Function for rendering preview image of the input gs data"""
    if validation_score > preview_score_threshold:
        buffered = io.BytesIO()
        rendered_image = renderer.render_gs(gs_data, 1, 512, 512, [25.0], [-10.0])[0]
        preview_image = Image.fromarray(rendered_image.detach().cpu().numpy())
        preview_image.save(buffered, format="PNG")
        encoded_preview = pybase64.b64encode(buffered.getvalue()).decode("utf-8")
    else:
        encoded_preview = None
    return encoded_preview


def _validate_text_vs_image(
    prompt: str,
    images: list[torch.Tensor],
    validator: ValidationEngine,
) -> ValidationResult:
    """Function for validation of the data that was generated using provided prompt"""

    # Validate input GS data by assessing rendered images
    t1 = time.time()
    val_res: ValidationResult = validator.validate_text_to_gs(prompt, images)
    logger.info(f" Score: {val_res.final_score}. Prompt: {prompt}")
    val_res.validation_time = time.time() - t1
    logger.info(f"Validation took: {val_res.validation_time} sec.")
    return val_res


def _validate_image_vs_image(
    prompt_image: torch.Tensor,
    images: list[torch.Tensor],
) -> ValidationResult:
    """Function for validation of the data that was generated using prompt-image"""
    t1 = time.time()
    val_res: ValidationResult = app.state.validator.validate_image_to_gs(prompt_image, images)
    logger.info(f" Score: {val_res.final_score}. Prompt: provided image.")
    logger.info(f" Validation took: {time.time() - t1} sec.")
    return val_res


def _finalize_results(
    validation_results: ValidationResult,
    gs_data: GaussianSplattingData,
    generate_preview: bool,
    preview_score_threshold: float,
    renderer: Renderer,
) -> ResponseData:
    """Function that finalize results"""
    if generate_preview:
        encoded_preview = _render_preview_image(
            gs_data, validation_results.final_score, preview_score_threshold, renderer
        )
    else:
        encoded_preview = None

    return ResponseData(
        score=validation_results.final_score,
        iqa=validation_results.combined_quality_score,
        alignment_score=validation_results.alignment_score,
        ssim=validation_results.ssim_score,
        lpips=validation_results.lpips_score,
        preview=encoded_preview,
    )


def _enhanced_cleanup() -> None:
    """Enhanced function for aggressive GPU memory cleanup"""
    t1 = time.time()
    
    # Clear all GPU tensors and caches
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        
        # Force garbage collection
        gc.collect()
        
        # Additional cache clearing
        torch.cuda.ipc_collect()
        
        # Get memory info
        gpu_memory_free, gpu_memory_total = torch.cuda.mem_get_info()
        memory_used = gpu_memory_total - gpu_memory_free
        memory_percent = (memory_used / gpu_memory_total) * 100
        
        logger.info(f"Enhanced cache purge took: {time.time() - t1:.2f} sec.")
        logger.info(f"VRAM Memory: {gpu_memory_free / 1024**3:.1f}GB free / {gpu_memory_total / 1024**3:.1f}GB total")
        logger.info(f"VRAM Usage: {memory_used / 1024**3:.1f}GB ({memory_percent:.1f}%)")


def _cleanup() -> None:
    """Function for cleaning up the memory"""
    _enhanced_cleanup()


def decode_assets(request: RequestData, zstd_decomp: zstandard.ZstdDecompressor) -> bytes:
    t1 = time.time()
    assets = pybase64.b64decode(request.data, validate=True)
    t2 = time.time()
    logger.info(
        f"Assets decoded. Size: {len(request.data)} -> {len(assets)}. "
        f"Time taken: {t2 - t1:.2f} sec. Prompt: {request.prompt}."
    )

    if request.compression == 1:  # Experimental. Zstd compression.
        compressed_size = len(assets)
        assets = zstd_decomp.decompress(assets)
        logger.info(
            f"Decompressed. Size: {compressed_size} -> {len(assets)}. "
            f"Time taken: {time.time() - t2:.2f} sec. Prompt: {request.prompt}."
        )
    elif request.compression == 2:  # Experimental. SPZ compression.
        compressed_size = len(assets)
        assets = pyspz.decompress(assets, include_normals=False)
        logger.info(
            f"Decompressed. Size: {compressed_size} -> {len(assets)}. "
            f"Time taken: {time.time() - t2:.2f} sec. Prompt: {request.prompt}."
        )

    return assets


def decode_and_validate_txt(
    request: RequestData,
    ply_data_loader: PlyLoader,
    renderer: Renderer,
    zstd_decompressor: zstandard.ZstdDecompressor,
    validator: ValidationEngine,
    include_time_stat: bool = False,
) -> ValidationResultData:
    t1 = time.time()
    assets = decode_assets(request, zstd_decomp=zstd_decompressor)
    gs_data, gs_rendered_images, time_stat = _prepare_input_data(assets, renderer, ply_data_loader, validator)
    if gs_data and request.prompt is not None:
        validation_result = _validate_text_vs_image(request.prompt, gs_rendered_images, validator)
        time_stat.validation_time = cast(float, validation_result.validation_time)
        response = _finalize_results(
            validation_result,
            gs_data,
            request.generate_preview,
            request.preview_score_threshold,
            renderer,
        )
        time_stat.total_time = time.time() - t1
    else:
        response = ResponseData(score=0.0)
    return ValidationResultData(
        response_data=response,
        time_stat=time_stat if include_time_stat else None,
    )


@app.get("/version/", response_model=str)
async def version() -> str:
    """
    Returns current endpoint version.
    """
    return str(VERSION)


@app.get("/gpu_status/")
async def gpu_status() -> dict:
    """
    Returns current GPU memory status for coordination with generation server.
    """
    if torch.cuda.is_available():
        gpu_memory_free, gpu_memory_total = torch.cuda.mem_get_info()
        memory_used = gpu_memory_total - gpu_memory_free
        memory_percent = (memory_used / gpu_memory_total) * 100
        
        return {
            "gpu_available": True,
            "memory_total_gb": gpu_memory_total / 1024**3,
            "memory_free_gb": gpu_memory_free / 1024**3,
            "memory_used_gb": memory_used / 1024**3,
            "memory_used_percent": memory_percent
        }
    else:
        return {"gpu_available": False}


@app.post("/cleanup_gpu/")
async def cleanup_gpu() -> dict:
    """
    Force aggressive GPU memory cleanup - called by generation server before loading models.
    """
    t1 = time.time()
    _enhanced_cleanup()
    cleanup_time = time.time() - t1
    
    # Get post-cleanup status
    status = await gpu_status()
    status["cleanup_time"] = cleanup_time
    
    return status


@app.post("/unload_models/")
async def unload_models() -> dict:
    """
    AGGRESSIVELY unload validation models to free maximum GPU memory for generation.
    This completely removes models from GPU and stores state for reloading.
    """
    t1 = time.time()
    
    # Get memory before unloading
    gpu_memory_free_before, gpu_memory_total = torch.cuda.mem_get_info()
    memory_before = (gpu_memory_total - gpu_memory_free_before) / 1024**3
    
    try:
        # Store model states for reloading
        app.state.model_backup = {}
        
        # AGGRESSIVE MODEL UNLOADING
        if hasattr(app.state, 'validator') and app.state.validator:
            validator = app.state.validator
            
            # 1. Quality Model - Move to CPU and clear GPU references
            if hasattr(validator, 'quality_model') and validator.quality_model:
                logger.info("Unloading quality model...")
                validator.quality_model.to('cpu')
                app.state.model_backup['quality_model'] = validator.quality_model
                validator.quality_model = None  # Clear GPU reference
            
            # 2. Aesthetic Model - Move to CPU and clear GPU references  
            if hasattr(validator, 'aesthetic_model') and validator.aesthetic_model:
                logger.info("Unloading aesthetic model...")
                validator.aesthetic_model.to('cpu')
                app.state.model_backup['aesthetic_model'] = validator.aesthetic_model
                validator.aesthetic_model = None  # Clear GPU reference
            
            # 3. Alignment Scorer - More complex, has multiple components
            if hasattr(validator, 'alignment_scorer') and validator.alignment_scorer:
                logger.info("Unloading alignment scorer...")
                alignment_scorer = validator.alignment_scorer
                app.state.model_backup['alignment_scorer'] = {}
                
                if hasattr(alignment_scorer, 'model') and alignment_scorer.model:
                    alignment_scorer.model.to('cpu')
                    app.state.model_backup['alignment_scorer']['model'] = alignment_scorer.model
                    alignment_scorer.model = None
                
                if hasattr(alignment_scorer, 'processor'):
                    app.state.model_backup['alignment_scorer']['processor'] = alignment_scorer.processor
                    alignment_scorer.processor = None
                
                # Clear the entire alignment scorer
                validator.alignment_scorer = None
        
        # 4. Renderer - Clear any GPU tensors
        if hasattr(app.state, 'renderer') and app.state.renderer:
            logger.info("Clearing renderer GPU state...")
            renderer = app.state.renderer
            # Clear any cached GPU tensors in renderer
            if hasattr(renderer, 'clear_gpu_cache'):
                renderer.clear_gpu_cache()
            app.state.model_backup['renderer'] = renderer
            app.state.renderer = None
        
        # 5. NUCLEAR OPTION - Clear the entire validator temporarily
        if hasattr(app.state, 'validator'):
            app.state.model_backup['validator_device'] = app.state.validator.device if app.state.validator else 'cuda'
            app.state.validator = None
        
        # 6. AGGRESSIVE CLEANUP - Multiple passes
        for i in range(3):  # Multiple cleanup passes
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
                torch.cuda.ipc_collect()
                gc.collect()
                time.sleep(0.1)  # Small delay between passes
        
        # Get memory after unloading
        gpu_memory_free_after, _ = torch.cuda.mem_get_info()
        memory_after = (gpu_memory_total - gpu_memory_free_after) / 1024**3
        memory_freed = memory_before - memory_after
        
        unload_time = time.time() - t1
        
        logger.info(f"AGGRESSIVE UNLOAD: freed {memory_freed:.2f}GB in {unload_time:.2f}s")
        logger.info(f"GPU Memory: {memory_after:.1f}GB used, {gpu_memory_free_after/1024**3:.1f}GB free")
        
        return {
            "status": "success",
            "memory_before_gb": memory_before,
            "memory_after_gb": memory_after,
            "memory_freed_gb": memory_freed,
            "total_memory_gb": gpu_memory_total / 1024**3,
            "free_memory_gb": gpu_memory_free_after / 1024**3,
            "unload_time": unload_time,
            "models_unloaded": True,
            "aggressive_unload": True
        }
        
    except Exception as e:
        logger.error(f"Error in aggressive model unloading: {e}")
        import traceback
        traceback.print_exc()
        return {
            "status": "error",
            "error": str(e),
            "models_unloaded": False
        }


@app.post("/reload_models/")
async def reload_models() -> dict:
    """
    Reload validation models back to GPU after generation is complete.
    This restores the models from the backup state.
    """
    t1 = time.time()
    
    try:
        # Check if we have model backups
        if not hasattr(app.state, 'model_backup') or not app.state.model_backup:
            logger.warning("No model backup found - reinitializing from scratch")
            # Reinitialize everything from scratch
            app.state.validator = ValidationEngine()
            app.state.validator.load_pipelines()
            app.state.renderer = Renderer()
            
            reload_time = time.time() - t1
            status = await gpu_status()
            status["reload_time"] = reload_time
            status["models_reloaded"] = True
            status["reinitialized"] = True
            return status
        
        # Restore models from backup
        backup = app.state.model_backup
        device = backup.get('validator_device', 'cuda')
        
        # 1. Restore Validator
        if not hasattr(app.state, 'validator') or app.state.validator is None:
            logger.info("Reinitializing validator...")
            app.state.validator = ValidationEngine()
            # Don't load pipelines yet - we'll restore from backup
        
        validator = app.state.validator
        
        # 2. Restore Quality Model
        if 'quality_model' in backup and backup['quality_model']:
            logger.info("Reloading quality model...")
            validator.quality_model = backup['quality_model']
            validator.quality_model.to(device)
        
        # 3. Restore Aesthetic Model
        if 'aesthetic_model' in backup and backup['aesthetic_model']:
            logger.info("Reloading aesthetic model...")
            validator.aesthetic_model = backup['aesthetic_model']
            validator.aesthetic_model.to(device)
        
        # 4. Restore Alignment Scorer
        if 'alignment_scorer' in backup and backup['alignment_scorer']:
            logger.info("Reloading alignment scorer...")
            # This is more complex - need to restore the alignment scorer object
            from engine.metrics.alignment_scorer import AlignmentScorer
            validator.alignment_scorer = AlignmentScorer(device=device)
            
            if 'model' in backup['alignment_scorer'] and backup['alignment_scorer']['model']:
                validator.alignment_scorer.model = backup['alignment_scorer']['model']
                validator.alignment_scorer.model.to(device)
            
            if 'processor' in backup['alignment_scorer'] and backup['alignment_scorer']['processor']:
                validator.alignment_scorer.processor = backup['alignment_scorer']['processor']
        
        # 5. Restore Renderer
        if 'renderer' in backup and backup['renderer']:
            logger.info("Reloading renderer...")
            app.state.renderer = backup['renderer']
        elif not hasattr(app.state, 'renderer') or app.state.renderer is None:
            app.state.renderer = Renderer()
        
        # 6. If models are missing, load them properly
        if not hasattr(validator, 'quality_model') or validator.quality_model is None:
            logger.info("Quality model missing - loading fresh...")
            validator.load_pipelines()
        
        # Clear backup
        app.state.model_backup = {}
        
        # Final cleanup
        _enhanced_cleanup()
        
        reload_time = time.time() - t1
        
        logger.info(f"Models reloaded to GPU in {reload_time:.2f}s")
        
        # Get final status
        status = await gpu_status()
        status["reload_time"] = reload_time
        status["models_reloaded"] = True
        status["restored_from_backup"] = True
        
        return status
        
    except Exception as e:
        logger.error(f"Error reloading models: {e}")
        import traceback
        traceback.print_exc()
        
        # Fallback - reinitialize everything
        try:
            logger.info("Fallback: reinitializing all models...")
            app.state.validator = ValidationEngine()
            app.state.validator.load_pipelines()
            app.state.renderer = Renderer()
            app.state.model_backup = {}
            
            status = await gpu_status()
            status["models_reloaded"] = True
            status["fallback_reinitialized"] = True
            return status
            
        except Exception as e2:
            logger.error(f"Fallback reinitialization also failed: {e2}")
            return {
                "status": "error",
                "error": f"Original: {str(e)}, Fallback: {str(e2)}",
                "models_reloaded": False
            }


@app.post("/validate_txt_to_3d_ply/", response_model=ResponseData)
async def validate_txt_to_3d_ply(request: RequestData) -> ResponseData:
    """
    Validates the input prompt and PLY data to produce scores.

    Parameters:
    - request (RequestData): An instance of RequestData containing the input prompt and data.

    Returns:
    - ResponseData: An instance of ResponseData containing the scores generated from the validation_lib process.

    """
    try:
        validation_result = decode_and_validate_txt(
            request=request,
            ply_data_loader=app.state.ply_data_loader,
            renderer=app.state.renderer,
            zstd_decompressor=app.state.zstd_decompressor,
            validator=app.state.validator,
        )
        response = validation_result.response_data
    except Exception as e:
        logger.exception(e)
        response = ResponseData(score=0.0)
    finally:
        _enhanced_cleanup()  # Enhanced cleanup after each validation

    return response


@app.post("/validate_img_to_3d_ply/", response_model=ResponseData)
async def validate_img_to_3d_ply(request: RequestData) -> ResponseData:
    """
    Validates the input prompt and PLY data to produce scores.

    Parameters:
    - request (RequestData): An instance of RequestData containing the input prompt and data.

    Returns:
    - ResponseData: An instance of ResponseData containing the scores generated from the validation_lib process.

    """
    try:
        assets = decode_assets(request, zstd_decomp=app.state.zstd_decompressor)
        gs_data, gs_rendered_images, _ = _prepare_input_data(
            assets, app.state.renderer, app.state.ply_data_loader, app.state.validator
        )
        if gs_data and request.prompt_image:
            image_data = pybase64.b64decode(request.prompt_image)
            prompt_image = Image.open(io.BytesIO(image_data))
            torch_prompt_image = torch.tensor(np.asarray(prompt_image))
            validation_results = _validate_image_vs_image(torch_prompt_image, gs_rendered_images)
            response = _finalize_results(
                validation_results,
                gs_data,
                request.generate_preview,
                request.preview_score_threshold,
                app.state.renderer,
            )
        else:
            response = ResponseData(score=0.0)
    except Exception as e:
        logger.exception(e)
        response = ResponseData(score=0.0)
    finally:
        _enhanced_cleanup()  # Enhanced cleanup after each validation
    return response


if __name__ == "__main__":
    uvicorn.run(app, host=args.host, port=args.port, backlog=256)
