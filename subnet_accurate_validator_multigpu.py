#!/usr/bin/env python3
"""
Subnet-Accurate Local Validator v2.0
Purpose: Use the exact decode_and_validate_txt function from benchmark validation 
to match production validation logic exactly, resolving validation discrepancies.
"""
import subprocess
import sys
import os

# Fix CUDA deterministic behavior before any CUDA operations
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'

import contextlib
import base64
import time
import gc
from io import StringIO
import io
from pathlib import Path
import json
import hashlib
import tempfile
import os

# Fetch defaults from running server via HTTP (no heavy imports)
def _fetch_generation_defaults():
    try:
        import requests
        server_url = os.environ.get('GEN_SERVER_URL', 'http://127.0.0.1:8096')
        r = requests.get(f"{server_url}/config/", timeout=3)
        if r.status_code == 200:
            return r.json()
    except Exception:
        pass
    return {}

_SERVER_DEFAULTS = _fetch_generation_defaults()

# Unified defaults (prefer server responses; fall back to constants)
NUM_INFERENCE_STEPS = int(_SERVER_DEFAULTS.get('num_inference_steps_t2i', 7))
GUIDANCE_SCALE = float(_SERVER_DEFAULTS.get('guidance_scale', 3.5))
# Defaults for TRELLIS quality params
SS_SAMPLING_STEPS = int(_SERVER_DEFAULTS.get('ss_sampling_steps', 21))
SLAT_SAMPLING_STEPS = int(_SERVER_DEFAULTS.get('slat_sampling_steps', 24))
SLAT_GUIDANCE_STRENGTH = float(_SERVER_DEFAULTS.get('slat_guidance_strength', 4.0))
SS_GUIDANCE_STRENGTH = float(_SERVER_DEFAULTS.get('ss_guidance_strength', 9.5))
# Use package imports directly; no sys.path modifications needed

# Test pyspz availability
try:
    import pyspz
    print("✅ pyspz library available")
except ImportError:
    print("❌ pyspz library not available")
    sys.exit(1)

# Import production validation components
try:
    from validation.engine.data_structures import RequestData, ValidationResultData
    from validation.engine.io.ply import PlyLoader
    from validation.engine.rendering.renderer import Renderer
    from validation.engine.validation_engine import ValidationEngine
    from serve import decode_and_validate_txt
    import zstandard
    import torch
    print("✅ Production validation components available")
except ImportError as e:
    print(f"❌ Production validation components not available: {e}")
    sys.exit(1)

# CLIP utilities for image-endpoint validation
import open_clip
import numpy as np
from PIL import Image
from torchvision import transforms
import torch.nn.functional as F

def load_validator_clip(device):
    """Load the validator CLIP model (convnext_large_d/laion2b_s26b_b102k_augreg)."""
    model, _, _ = open_clip.create_model_and_transforms(
        "convnext_large_d", pretrained="laion2b_s26b_b102k_augreg", device=device
    )
    tokenizer = open_clip.get_tokenizer("convnext_large_d")
    model.eval()
    mean = torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(1, 3, 1, 1) * 3
    std = torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(1, 3, 1, 1) * 3
    normalize = transforms.Normalize(mean, std)
    return model, tokenizer, normalize

def encode_text(model, tokenizer, device, text: str):
    tokens = tokenizer(text).to(device)
    with torch.no_grad(), torch.amp.autocast(device.type):
        feats = model.encode_text(tokens)
        feats = feats / feats.norm(dim=-1, keepdim=True)
    return feats

def encode_image(model, normalize, device, img: Image.Image, res: int = 224):
    t = torch.tensor(np.array(img)).float() / 255.0
    if t.ndim == 3:
        t = t.permute(2, 0, 1)
    t = t.unsqueeze(0).to(device)
    t = F.interpolate(t, size=(res, res), mode="bicubic", align_corners=False)
    t = normalize(t)
    with torch.no_grad(), torch.amp.autocast(device.type):
        feats = model.encode_image(t)
        feats = feats / feats.norm(dim=-1, keepdim=True)
    return feats

def clip_text_text(model, tokenizer, device, a: str, b: str) -> float:
    fa = encode_text(model, tokenizer, device, a)
    fb = encode_text(model, tokenizer, device, b)
    sim = (fa @ fb.T).float().item()
    return float(np.clip(sim, 0, 1))

def clip_text_image(model, tokenizer, normalize, device, text: str, img: Image.Image) -> float:
    tf = encode_text(model, tokenizer, device, text)
    vf = encode_image(model, normalize, device, img)
    sim = (vf @ tf.T).float().cpu().numpy()[0][0]
    return float(np.clip(sim, 0, 1))

@contextlib.contextmanager
def suppress_stdout():
    """Context manager to suppress stdout and stderr temporarily"""
    original_stdout = sys.stdout
    original_stderr = sys.stderr
    sys.stdout = StringIO()
    sys.stderr = StringIO()
    try:
        yield
    finally:
        sys.stdout = original_stdout
        sys.stderr = original_stderr

def generate_and_get_ply_data(
    prompt: str,
    endpoint: str,
    num_inference_steps: int = NUM_INFERENCE_STEPS,
    guidance_scale: float = GUIDANCE_SCALE,
    ss_sampling_steps: int = SS_SAMPLING_STEPS,
    slat_sampling_steps: int = SLAT_SAMPLING_STEPS,
    slat_guidance_strength: float = SLAT_GUIDANCE_STRENGTH,
    ss_guidance_strength: float = SS_GUIDANCE_STRENGTH,
    port: int = 8096,
    save_to_file: str = None,
) -> bytes:
    """
    Generate 3D model using TRELLIS and return compressed PLY data
    Optionally save to file for later validation to avoid GPU memory conflicts
    """
    url = f"http://127.0.0.1:{port}/{endpoint}"
    
    import requests
    print(f"🎨 Generating 3D model for: '{prompt}'")

    try:
        # Always include tuning params for both image and 3D generation endpoints
        payload = {
            "prompt": prompt,
            "num_inference_steps": num_inference_steps,
            "guidance_scale": guidance_scale,
            "ss_sampling_steps": ss_sampling_steps,
            "slat_sampling_steps": slat_sampling_steps,
            "slat_guidance_strength": slat_guidance_strength,
            "ss_guidance_strength": ss_guidance_strength,
        }
        with requests.post(url, data=payload, timeout=300, stream=False) as response:
            response.raise_for_status()
            
            compression = response.headers.get('x-compression', 'none')
            content_length = len(response.content)
            
            print(f"📦 Response received: {content_length:,} bytes (compression: {compression})")
            
            # Optionally save to file for later validation
            if save_to_file:
                with open(save_to_file, 'wb') as f:
                    f.write(response.content)
                print(f"💾 PLY data saved to: {save_to_file}")
            
            # Return the raw response content (compressed or uncompressed)
            return response.content
            
    except requests.exceptions.RequestException as e:
        print(f"❌ Generation request failed: {e}")
        raise
    except Exception as e:
        print(f"❌ Generation failed: {e}")
        raise

def generate_ply_only(
    prompt: str,
    endpoint: str = "generate/",
    num_inference_steps: int = NUM_INFERENCE_STEPS,
    guidance_scale: float = GUIDANCE_SCALE,
    ss_sampling_steps: int = SS_SAMPLING_STEPS,
    slat_sampling_steps: int = SLAT_SAMPLING_STEPS,
    slat_guidance_strength: float = SLAT_GUIDANCE_STRENGTH,
    ss_guidance_strength: float = SS_GUIDANCE_STRENGTH,
    port: int = 8096,
    output_file: str = None,
) -> str:
    """
    Generate PLY data only and save to file. This allows generation models to be unloaded
    before loading validation models, avoiding GPU memory conflicts.
    
    Returns:
        str: Path to the saved PLY file
    """
    if output_file is None:
        import tempfile
        import os
        output_file = os.path.join(tempfile.gettempdir(), f"generated_ply_{int(time.time())}.ply")
    
    print(f"🎨 Phase 1: Generating PLY data only...")
    print(f"   Prompt: '{prompt}'")
    print(f"   Output file: {output_file}")
    
    # Generate and save PLY data
    ply_data = generate_and_get_ply_data(
        prompt=prompt,
        endpoint=endpoint,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        ss_sampling_steps=ss_sampling_steps,
        slat_sampling_steps=slat_sampling_steps,
        slat_guidance_strength=slat_guidance_strength,
        ss_guidance_strength=ss_guidance_strength,
        port=port,
        save_to_file=output_file
    )
    
    print(f"✅ PLY generation complete: {len(ply_data):,} bytes")
    print(f"💾 Saved to: {output_file}")
    
    return output_file

def validate_ply_data(
    ply_data: str,
    prompt: str,
    use_cached_models: bool = False,  # Changed default to False
    force_cpu: bool = False,
) -> dict:
    """
    Validate PLY data directly in memory using production validation logic.
    This loads validation models after generation is complete.
    
    Args:
        ply_data: PLY data string to validate
        prompt: Original prompt for validation scoring
        use_cached_models: Whether to use globally cached validation models
        force_cpu: Force CPU loading for validation models (3D rendering still requires CUDA)
    
    Returns:
        dict: Validation results with scores
    """
    print(f"🔍 Validating PLY data directly in memory: {len(ply_data):,} bytes")
    print(f"   Validation prompt: '{prompt}'")
    print(f"   Use cached models: {use_cached_models}")
    print(f"   Force CPU: {force_cpu}")
    
    # Use the same validation logic as the file-based function
    return _validate_ply_data_internal(ply_data, prompt, use_cached_models, force_cpu)

def _validate_ply_data_internal(
    ply_data: bytes,
    prompt: str,
    use_cached_models: bool = False,
    force_cpu: bool = False,
) -> dict:
    """
    Internal validation function that handles both cached and non-cached model scenarios.
    """
    # Use cached models if available and requested
    if use_cached_models:
        try:
            _ensure_models_loaded(force_cpu=force_cpu)
            print(f"✅ Using cached validation models")
            
            # Create RequestData
            encoded_data = base64.b64encode(ply_data).decode('utf-8')
            request_data = RequestData(
                prompt=prompt,
                data=encoded_data,
                compression=2,  # SPZ compression
                generate_preview=False,
                preview_score_threshold=0.8
            )
            
            # Run validation with cached components
            validation_result = decode_and_validate_txt(
                request=request_data,
                ply_data_loader=_global_ply_data_loader,
                renderer=_global_renderer,
                validator=_global_validator
            )
            
            # Handle the result - it should be a tuple of (response, time_stats)
            if isinstance(validation_result, tuple):
                if len(validation_result) == 2:
                    response, time_stats = validation_result
                else:
                    print(f"   ⚠️ Unexpected tuple length: {len(validation_result)}")
                    response = validation_result[0]
                    time_stats = validation_result[1] if len(validation_result) > 1 else None
            else:
                print(f"   ⚠️ Unexpected result type: {type(validation_result)}")
                response = validation_result
                time_stats = None
            
            # Calculate demo fidelity score
            demo_fidelity_score = calculate_demo_fidelity_score(response.score)
            task_fidelity_score = calculate_task_fidelity_score(response.score)
            
            print(f"✅ Validation complete with cached models")
            print(f"   Validation score: {response.score:.4f}")
            print(f"   Alignment score: {response.alignment_score:.4f}")
            print(f"   Quality score: {response.iqa:.4f}")
            print(f"   SSIM score: {response.ssim:.4f}")
            print(f"   LPIPS score: {response.lpips:.4f}")
            print(f"   Demo fidelity: {demo_fidelity_score:.4f}")
            print(f"   Task fidelity: {task_fidelity_score:.4f}")
            print(f"   Time stats: {time_stats}")
            print(f"=" * 60)
            
            return {
                'validation_engine_score': response.score,
                'alignment_score': response.alignment_score,
                'quality_score': response.iqa,
                'ssim_score': response.ssim,
                'lpips_score': response.lpips,
                'demo_fidelity_score': demo_fidelity_score,
                'task_fidelity_score': task_fidelity_score,
                'validation_passed': response.score >= 0.8,
                'time_stats': time_stats
            }
            
        except Exception as e:
            print(f"❌ Cached validation failed: {e}")
            return {
                'validation_engine_score': 0.0,
                'alignment_score': 0.0,
                'quality_score': 0.0,
                'ssim_score': 0.0,
                'lpips_score': 0.0,
                'demo_fidelity_score': 0.0,
                'task_fidelity_score': 0.0,
                'validation_passed': False,
                'error': str(e)
            }
    
    # Non-cached validation (load models fresh each time)
    try:
        print(f"🔄 Loading validation models fresh (no caching)")
        
        # Load validation models
        device = "cpu" if force_cpu else None  # None will use CUDA if available
        validator = ValidationEngine(device=device)
        validator.load_pipelines()
        
        # Create RequestData
        encoded_data = base64.b64encode(ply_data).decode('utf-8')
        request_data = RequestData(
            prompt=prompt,
            data=encoded_data,
            compression=2,  # SPZ compression
            generate_preview=False,
            preview_score_threshold=0.8
        )
        
        # Create the required components for validation
        ply_data_loader = PlyLoader()
        renderer = Renderer()
        
        # Run validation
        validation_result = decode_and_validate_txt(
            request=request_data,
            ply_data_loader=ply_data_loader,
            renderer=renderer,
            validator=validator
        )
        
        # Handle the result - it should be a tuple of (response, time_stats)
        if isinstance(validation_result, tuple):
            if len(validation_result) == 2:
                response, time_stats = validation_result
            else:
                print(f"   ⚠️ Unexpected tuple length: {len(validation_result)}")
                response = validation_result[0]
                time_stats = validation_result[1] if len(validation_result) > 1 else None
        else:
            print(f"   ⚠️ Unexpected result type: {type(validation_result)}")
            response = validation_result
            time_stats = None
        
        # Calculate demo fidelity score
        demo_fidelity_score = calculate_demo_fidelity_score(response.score)
        task_fidelity_score = calculate_task_fidelity_score(response.score)
        
        print(f"✅ Validation complete with fresh models")
        print(f"   Validation score: {response.score:.4f}")
        print(f"   Alignment score: {response.alignment_score:.4f}")
        print(f"   Quality score: {response.iqa:.4f}")
        print(f"   SSIM score: {response.ssim:.4f}")
        print(f"   LPIPS score: {response.lpips:.4f}")
        print(f"   Demo fidelity: {demo_fidelity_score:.4f}")
        print(f"   Task fidelity: {task_fidelity_score:.4f}")
        print(f"   Time stats: {time_stats}")
        print(f"=" * 60)
        
        return {
            'validation_engine_score': response.score,
            'alignment_score': response.alignment_score,
            'quality_score': response.iqa,
            'ssim_score': response.ssim,
            'lpips_score': response.lpips,
            'demo_fidelity_score': demo_fidelity_score,
            'task_fidelity_score': task_fidelity_score,
            'validation_passed': response.score >= 0.8,
            'time_stats': time_stats
        }
        
    except Exception as e:
        print(f"❌ Production validation failed: {e}")
        return {
            'validation_engine_score': 0.0,
            'alignment_score': 0.0,
            'quality_score': 0.0,
            'ssim_score': 0.0,
            'lpips_score': 0.0,
            'demo_fidelity_score': 0.0,
            'task_fidelity_score': 0.0,
            'validation_passed': False,
            'error': str(e)
        }
    finally:
        with suppress_stdout():
            validator.unload_pipelines()
        # Clean up components
        if 'ply_data_loader' in locals():
            del ply_data_loader
        if 'renderer' in locals():
            del renderer
        gc.collect()
        torch.cuda.empty_cache()

def validate_ply_file(
    ply_file_path: str,
    prompt: str,
    use_cached_models: bool = False,  # Changed default to False
    force_cpu: bool = False,
) -> dict:
    """
    Validate a PLY file using production validation logic.
    This loads validation models after generation is complete.
    
    Args:
        ply_file_path: Path to the PLY file to validate
        prompt: Original prompt for validation scoring
        use_cached_models: Whether to use globally cached validation models
        force_cpu: Whether to force CPU loading (3D rendering still requires CUDA)
    
    Returns:
        dict: Validation results
    """
    print(f"🔍 Phase 2: Validating PLY file...")
    print(f"   PLY file: {ply_file_path}")
    print(f"   Validation prompt: '{prompt}'")
    print(f"   Use cached models: {use_cached_models}")
    print(f"   Force CPU: {force_cpu}")
    
    # Read PLY data from file
    try:
        with open(ply_file_path, 'rb') as f:
            ply_data = f.read()
        print(f"📦 Loaded PLY data: {len(ply_data):,} bytes")
    except Exception as e:
        print(f"❌ Failed to read PLY file: {e}")
        return {
            'validation_engine_score': 0.0,
            'alignment_score': 0.0,
            'quality_score': 0.0,
            'ssim_score': 0.0,
            'lpips_score': 0.0,
            'demo_fidelity_score': 0.0,
            'task_fidelity_score': 0.0,
            'validation_passed': False,
            'error': f"Failed to read PLY file: {e}"
        }
    
    # Use the same validation logic as the direct data function
    return _validate_ply_data_internal(ply_data, prompt, use_cached_models, force_cpu)

def validate_with_production_logic(ply_data: bytes, prompt: str) -> dict:
    """
    Validate using the exact production decode_and_validate_txt function
    This ensures 100% accuracy with production validation results
    """
    
    print(f"🔍 Step 1: Initializing production validation components")
    
    # Initialize all components exactly as in production
    validator = ValidationEngine(verbose=True)
    with suppress_stdout():
        validator.load_pipelines()
    
    zstd_decompressor = zstandard.ZstdDecompressor()
    renderer = Renderer()
    ply_data_loader = PlyLoader()
    
    # Clear GPU memory
    gc.collect()
    torch.cuda.empty_cache()
    
    print(f"✅ Production validation components initialized")
    
    try:
        print(f"🔬 Step 2: Preparing RequestData (SPZ compression=2)")
        
        # Encode PLY data as base64 (as done in production)
        encoded_data = base64.b64encode(ply_data).decode('utf-8')
        
        # Create RequestData matching production format
        request_data = RequestData(
            prompt=prompt,
            data=encoded_data,
            compression=2,  # SPZ compression (production standard)
            generate_preview=False,
            preview_score_threshold=0.8
        )
        
        print(f"📊 RequestData prepared:")
        print(f"   Validation Prompt: '{prompt}' (original prompt for scoring)")
        print(f"   Data size: {len(encoded_data):,} characters (base64)")
        print(f"   Compression: 2 (SPZ)")
        
        print(f"🚀 Step 3: Running production decode_and_validate_txt")
        print(f"   Computing CLIP scores against: '{prompt}'")
        
        # Run the exact production validation function
        try:
            print(f"   Calling decode_and_validate_txt with:")
            print(f"   - request: {type(request_data)} ")
            print(f"   - ply_data_loader: {type(ply_data_loader)}")
            print(f"   - renderer: {type(renderer)}")
            print(f"   - validator: {type(validator)}")
            
            validation_result = decode_and_validate_txt(
                request=request_data,
                ply_data_loader=ply_data_loader,
                renderer=renderer,
                validator=validator
            )
            
            # Handle the result - it should be a tuple of (response, time_stats)
            if isinstance(validation_result, tuple):
                if len(validation_result) == 2:
                    response, time_stats = validation_result
                else:
                    print(f"   ⚠️ Unexpected tuple length: {len(validation_result)}")
                    response = validation_result[0]
                    time_stats = validation_result[1] if len(validation_result) > 1 else None
            else:
                print(f"   ⚠️ Unexpected result type: {type(validation_result)}")
                response = validation_result
                time_stats = None
        except Exception as inner_e:
            print(f"   Inner error details: {inner_e}")
            import traceback
            traceback.print_exc()
            raise inner_e
        
        print(f"✅ Production validation completed")
        
        print(f"📊 PRODUCTION VALIDATION RESULTS:")
        print(f"   🎯 Final Score: {response.score:.4f}")
        print(f"   🤝 Alignment Score: {response.alignment_score:.4f}")
        print(f"   🏆 Quality Score (IQA): {response.iqa:.4f}")
        print(f"   📐 SSIM: {response.ssim:.4f}")
        print(f"   👁️ LPIPS: {response.lpips:.4f}")
        
        if time_stats:
            print(f"⏱️ Performance Stats:")
            print(f"   Loading: {time_stats.loading_data_time:.3f}s")
            print(f"   Rendering: {time_stats.image_rendering_time:.3f}s")
            print(f"   Validation: {time_stats.validation_time:.3f}s")
            print(f"   Total: {time_stats.total_time:.3f}s")
        
        # Apply subnet-specific logic (demo fidelity scoring)
        demo_fidelity_score = calculate_demo_fidelity_score(response.score)
        
        print(f"🎭 Demo Fidelity Score: {demo_fidelity_score}")
        
        print(f"=" * 60)
        print(f"🏁 PRODUCTION-ACCURATE VALIDATION COMPLETE")
        print(f"=" * 60)
        
        return {
            'validation_engine_score': response.score,
            'alignment_score': response.alignment_score,
            'quality_score': response.iqa,
            'ssim_score': response.ssim,
            'lpips_score': response.lpips,
            'demo_fidelity_score': demo_fidelity_score,
            'task_fidelity_score': response.score,  # Production final score
            'validation_passed': response.score > 0.0,
            'quality_threshold': 0.6,
            'alignment_threshold_passed': response.alignment_score >= 0.3,
            'production_logic_applied': True,
            'time_stats': {
                'loading_time': time_stats.loading_data_time if time_stats else 0.0,
                'rendering_time': time_stats.image_rendering_time if time_stats else 0.0,
                'validation_time': time_stats.validation_time if time_stats else 0.0,
                'total_time': time_stats.total_time if time_stats else 0.0,
            } if time_stats else None
        }
        
    except Exception as e:
        print(f"❌ Production validation failed: {e}")
        return {
            'validation_engine_score': 0.0,
            'alignment_score': 0.0,
            'quality_score': 0.0,
            'ssim_score': 0.0,
            'lpips_score': 0.0,
            'demo_fidelity_score': 0.0,
            'task_fidelity_score': 0.0,
            'validation_passed': False,
            'quality_threshold': 0.6,
            'alignment_threshold_passed': False,
            'production_logic_applied': True,
            'error': str(e)
        }
    finally:
        # Cleanup
        with suppress_stdout():
            validator.unload_pipelines()
        gc.collect()
        torch.cuda.empty_cache()

# New: raw PLY validation (compression=0)
def validate_with_production_logic_raw(ply_raw_data: bytes, prompt: str) -> dict:
    """
    Run production validation assuming data is already decompressed raw PLY bytes.
    Sends compression=0 to decode_and_validate_txt.
    """
    validator = ValidationEngine(verbose=True)
    with suppress_stdout():
        validator.load_pipelines()
    zstd_decompressor = zstandard.ZstdDecompressor()
    renderer = Renderer()
    ply_data_loader = PlyLoader()
    gc.collect()
    torch.cuda.empty_cache()

    try:
        encoded_data = base64.b64encode(ply_raw_data).decode('utf-8')
        request_data = RequestData(
            prompt=prompt,
            data=encoded_data,
            compression=0,  # raw PLY
            generate_preview=False,
            preview_score_threshold=0.8
        )
        validation_result = decode_and_validate_txt(
            request=request_data,
            ply_data_loader=ply_data_loader,
            renderer=renderer,
            validator=validator
        )
        
        # Handle the result - it should be a tuple of (response, time_stats)
        if isinstance(validation_result, tuple):
            if len(validation_result) == 2:
                response, time_stats = validation_result
            else:
                print(f"   ⚠️ Unexpected tuple length: {len(validation_result)}")
                response = validation_result[0]
                time_stats = validation_result[1] if len(validation_result) > 1 else None
        else:
            print(f"   ⚠️ Unexpected result type: {type(validation_result)}")
            response = validation_result
            time_stats = None
        
        return {
            'validation_engine_score': response.score,
            'alignment_score': response.alignment_score,
            'quality_score': response.iqa,
            'ssim_score': response.ssim,
            'lpips_score': response.lpips,
            'time_stats': {
                'loading_time': time_stats.loading_data_time if time_stats else 0.0,
                'rendering_time': time_stats.image_rendering_time if time_stats else 0.0,
                'validation_time': time_stats.validation_time if time_stats else 0.0,
                'total_time': time_stats.total_time if time_stats else 0.0,
            } if time_stats else None
        }
    except Exception as e:
        return {
            'validation_engine_score': 0.0,
            'alignment_score': 0.0,
            'quality_score': 0.0,
            'ssim_score': 0.0,
            'lpips_score': 0.0,
            'error': str(e)
        }
    finally:
        with suppress_stdout():
            validator.unload_pipelines()
        gc.collect()
        torch.cuda.empty_cache()

def calculate_demo_fidelity_score(validation_score: float) -> float:
    """
    Calculate demo fidelity score based on demo.ipynb logic:
    - 1.0 if validation_score >= 0.8
    - 0.75 if validation_score >= 0.6 and < 0.8
    - 0 if validation_score < 0.6
    """
    if validation_score >= 0.8:
        return 1.0
    elif validation_score >= 0.6:
        return 0.75
    else:
        return 0.0

def calculate_task_fidelity_score(validation_score: float) -> float:
    """
    Calculate task fidelity score - same as validation score for now.
    """
    return validation_score

# PLY caching system to avoid re-generation
_PLY_CACHE_DIR = None
_PLY_CACHE = {}

def _get_ply_cache_dir():
    """Get or create the PLY cache directory"""
    global _PLY_CACHE_DIR
    if _PLY_CACHE_DIR is None:
        _PLY_CACHE_DIR = os.path.join(tempfile.gettempdir(), "trellis_ply_cache")
        os.makedirs(_PLY_CACHE_DIR, exist_ok=True)
    return _PLY_CACHE_DIR

def _get_ply_cache_key(prompt: str, endpoint: str, **kwargs) -> str:
    """Generate a cache key for PLY data based on prompt and parameters"""
    # Create a hash of all parameters that affect generation
    params = {
        'prompt': prompt,
        'endpoint': endpoint,
        'num_inference_steps': kwargs.get('num_inference_steps', NUM_INFERENCE_STEPS),
        'guidance_scale': kwargs.get('guidance_scale', GUIDANCE_SCALE),
        'ss_sampling_steps': kwargs.get('ss_sampling_steps', SS_SAMPLING_STEPS),
        'slat_sampling_steps': kwargs.get('slat_sampling_steps', SLAT_SAMPLING_STEPS),
        'slat_guidance_strength': kwargs.get('slat_guidance_strength', SLAT_GUIDANCE_STRENGTH),
        'ss_guidance_strength': kwargs.get('ss_guidance_strength', SS_GUIDANCE_STRENGTH),
    }
    
    # Create a deterministic hash
    param_str = json.dumps(params, sort_keys=True)
    return hashlib.md5(param_str.encode()).hexdigest()

def _get_cached_ply_file(cache_key: str) -> str:
    """Get the cached PLY file path if it exists"""
    cache_dir = _get_ply_cache_dir()
    return os.path.join(cache_dir, f"{cache_key}.ply")

def _is_ply_cached(cache_key: str) -> bool:
    """Check if PLY data is cached"""
    cached_file = _get_cached_ply_file(cache_key)
    return os.path.exists(cached_file)

def _cache_ply_data(cache_key: str, ply_data: bytes) -> str:
    """Cache PLY data and return the file path"""
    cached_file = _get_cached_ply_file(cache_key)
    with open(cached_file, 'wb') as f:
        f.write(ply_data)
    print(f"💾 PLY data cached: {cached_file}")
    return cached_file

def _get_cached_ply_data(cache_key: str) -> bytes:
    """Get cached PLY data"""
    cached_file = _get_cached_ply_file(cache_key)
    with open(cached_file, 'rb') as f:
        return f.read()

def generate_ply_with_cache(
    prompt: str,
    endpoint: str = "generate/",
    num_inference_steps: int = NUM_INFERENCE_STEPS,
    guidance_scale: float = GUIDANCE_SCALE,
    ss_sampling_steps: int = SS_SAMPLING_STEPS,
    slat_sampling_steps: int = SLAT_SAMPLING_STEPS,
    slat_guidance_strength: float = SLAT_GUIDANCE_STRENGTH,
    ss_guidance_strength: float = SS_GUIDANCE_STRENGTH,
    port: int = 8096,
    use_cache: bool = True,
    output_file: str = None,
) -> str:
    """
    Generate PLY data with caching support to avoid re-generation.
    Returns the path to the PLY file (either cached or newly generated).
    """
    # Generate cache key
    cache_key = _get_ply_cache_key(
        prompt=prompt,
        endpoint=endpoint,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        ss_sampling_steps=ss_sampling_steps,
        slat_sampling_steps=slat_sampling_steps,
        slat_guidance_strength=slat_guidance_strength,
        ss_guidance_strength=ss_guidance_strength,
    )
    
    # Check cache first
    if use_cache and _is_ply_cached(cache_key):
        cached_file = _get_cached_ply_file(cache_key)
        print(f"📁 Using cached PLY data: {cached_file}")
        return cached_file
    
    # Generate new PLY data
    print(f"🎨 Generating new PLY data for: '{prompt[:50]}...'")
    
    if output_file is None:
        output_file = os.path.join(tempfile.gettempdir(), f"generated_ply_{int(time.time())}.ply")
    
    # Generate PLY data
    ply_data = generate_and_get_ply_data(
        prompt=prompt,
        endpoint=endpoint,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        ss_sampling_steps=ss_sampling_steps,
        slat_sampling_steps=slat_sampling_steps,
        slat_guidance_strength=slat_guidance_strength,
        ss_guidance_strength=ss_guidance_strength,
        port=port,
        save_to_file=output_file
    )
    
    # Cache the data if requested
    if use_cache:
        _cache_ply_data(cache_key, ply_data)
    
    print(f"✅ PLY generation complete: {len(ply_data):,} bytes")
    return output_file

def clear_ply_cache():
    """Clear all cached PLY data"""
    cache_dir = _get_ply_cache_dir()
    if os.path.exists(cache_dir):
        import shutil
        shutil.rmtree(cache_dir)
        print(f"🗑️ PLY cache cleared: {cache_dir}")
    else:
        print(f"📁 No PLY cache to clear")

def get_ply_cache_info():
    """Get information about the PLY cache"""
    cache_dir = _get_ply_cache_dir()
    if not os.path.exists(cache_dir):
        return {"cache_dir": cache_dir, "cached_files": 0, "total_size": 0}
    
    cached_files = [f for f in os.listdir(cache_dir) if f.endswith('.ply')]
    total_size = sum(os.path.getsize(os.path.join(cache_dir, f)) for f in cached_files)
    
    return {
        "cache_dir": cache_dir,
        "cached_files": len(cached_files),
        "total_size": total_size,
        "total_size_mb": total_size / (1024 * 1024)
    }

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Subnet-Accurate Local Validator v2.0", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("original_prompt", type=str, help="Prompt to compute validation against")
    parser.add_argument("optimized_prompt", type=str, nargs="?", default=None, help="Prompt to use for generation (defaults to original)")
    parser.add_argument("--endpoint", type=str, nargs="?", default="generate/", help="Endpoint path, e.g. generate/ or generate/isometric_3d/")
    parser.add_argument("--num_inference_steps", type=int, nargs="?", default=NUM_INFERENCE_STEPS, help=f"Sampler steps for image model (default from server: {NUM_INFERENCE_STEPS})")
    parser.add_argument("--guidance_scale", type=float, nargs="?", default=GUIDANCE_SCALE, help=f"Guidance scale for image model (default from server: {GUIDANCE_SCALE})")
    # Optional named overrides
    parser.add_argument("--ss_steps", dest="ss_sampling_steps", type=int, default=SS_SAMPLING_STEPS, help=f"Sparse-structure sampler steps (default from server: {SS_SAMPLING_STEPS})")
    parser.add_argument("--slat_steps", dest="slat_sampling_steps", type=int, default=SLAT_SAMPLING_STEPS, help=f"SLAT sampler steps (default from server: {SLAT_SAMPLING_STEPS})")
    parser.add_argument("--slat_guidance", dest="slat_guidance_strength", type=float, default=SLAT_GUIDANCE_STRENGTH, help=f"SLAT guidance strength (default from server: {SLAT_GUIDANCE_STRENGTH})")
    parser.add_argument("--ss_guidance", dest="ss_guidance_strength", type=float, default=SS_GUIDANCE_STRENGTH, help=f"Sparse-structure guidance strength (default from server: {SS_GUIDANCE_STRENGTH})")
    parser.add_argument("--port", type=int, nargs="?", default=8096, help="Port to use for generation (default from server: 8096)")
    parser.add_argument("--pre-generated-ply", type=str, help="Path to pre-generated PLY file (skips generation)")
    
    # New options for separated generation/validation
    parser.add_argument("--separate-phases", action="store_true", help="Separate generation and validation phases to avoid GPU memory conflicts")
    parser.add_argument("--generate-only", action="store_true", help="Only generate PLY data and save to file (skip validation)")
    parser.add_argument("--validate-only", type=str, help="Only validate a pre-generated PLY file (skip generation)")
    parser.add_argument("--output-file", type=str, help="Output file path for generated PLY data")
    parser.add_argument("--use-cached-models", action="store_true", default=False, help="Use globally cached validation models (default: False)")
    parser.add_argument("--force-cpu", action="store_true", help="Force CPU loading for validation models (3D rendering still requires CUDA)")
    
    # PLY caching options
    parser.add_argument("--use-ply-cache", action="store_true", default=True, help="Use PLY caching to avoid re-generation (default: True)")
    parser.add_argument("--no-ply-cache", action="store_true", help="Disable PLY caching")
    parser.add_argument("--clear-ply-cache", action="store_true", help="Clear PLY cache and exit")
    parser.add_argument("--ply-cache-info", action="store_true", help="Show PLY cache information and exit")
    
    args = parser.parse_args()

    # Handle cache management options
    if args.clear_ply_cache:
        clear_ply_cache()
        return
    
    if args.ply_cache_info:
        cache_info = get_ply_cache_info()
        print(f"📁 PLY Cache Information:")
        print(f"   Cache directory: {cache_info['cache_dir']}")
        print(f"   Cached files: {cache_info['cached_files']}")
        print(f"   Total size: {cache_info['total_size_mb']:.2f} MB")
        return

    original_prompt = args.original_prompt
    optimized_prompt = args.optimized_prompt if args.optimized_prompt is not None else args.original_prompt
    endpoint = args.endpoint
    num_inference_steps = args.num_inference_steps
    guidance_scale = args.guidance_scale
    ss_sampling_steps = args.ss_sampling_steps
    slat_sampling_steps = args.slat_sampling_steps
    slat_guidance_strength = args.slat_guidance_strength
    ss_guidance_strength = args.ss_guidance_strength
    
    # Determine cache usage
    use_ply_cache = args.use_ply_cache and not args.no_ply_cache
    
    print(f"🚀 PRODUCTION-ACCURATE VALIDATION v2.0")
    print(f"=" * 60)
    print(f"📝 Original Prompt: '{original_prompt}'")
    if optimized_prompt != original_prompt:
        print(f"🔧 Optimized Prompt: '{optimized_prompt}'")
        print(f"   (Using optimized prompt for generation, original prompt for validation)")
    else:
        print(f"🔧 Using same prompt for generation and validation")
    print(f"🔧 Using endpoint: {endpoint}")
    print(f"🔧 Using: decode_and_validate_txt (production function)")
    print(f"🎯 CLIP Model: convnext_large_d (production-accurate)")
    print(f"🗜️ Compression: SPZ (production standard)")
    
    # Handle new separated phases options
    if args.generate_only:
        print(f"🎨 MODE: Generate PLY only (skip validation)")
        print(f"=" * 60)
    elif args.validate_only:
        print(f"🔍 MODE: Validate PLY file only (skip generation)")
        print(f"📁 PLY file: {args.validate_only}")
        print(f"=" * 60)
    elif args.separate_phases:
        print(f"🔄 MODE: Separated generation and validation phases")
        print(f"   (Avoids GPU memory conflicts)")
        print(f"=" * 60)
    else:
        print(f"🔄 MODE: Traditional combined generation and validation")
        print(f"=" * 60)
    
    try:
        # Handle different execution modes
        if args.generate_only:
            # Generate PLY only mode
            print(f"🎨 Generating PLY data only...")
            output_file = args.output_file
            ply_file_path = generate_ply_only(
                prompt=optimized_prompt,
                endpoint=endpoint,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                ss_sampling_steps=ss_sampling_steps,
                slat_sampling_steps=slat_sampling_steps,
                slat_guidance_strength=slat_guidance_strength,
                ss_guidance_strength=ss_guidance_strength,
                port=args.port,
                output_file=output_file
            )
            print(f"✅ PLY generation complete!")
            print(f"📁 Output file: {ply_file_path}")
            return
            
        elif args.validate_only:
            # Validate PLY file only mode
            print(f"🔍 Validating PLY file...")
            results = validate_ply_file(
                ply_file_path=args.validate_only,
                prompt=original_prompt,
                use_cached_models=args.use_cached_models,
                force_cpu=args.force_cpu
            )
            
            # Display results
            print(f"🎯 VALIDATION RESULTS:")
            print(f"=" * 60)
            print(f"📁 PLY file: {args.validate_only}")
            print(f"📝 Validation prompt: '{original_prompt}'")
            print(f"🏆 Validation Engine Score: {results['validation_engine_score']:.4f}")
            print(f"🤝 Alignment Score: {results['alignment_score']:.4f}")
            print(f"💎 Quality Score: {results['quality_score']:.4f}")
            print(f"🎭 Demo Fidelity Score: {results['demo_fidelity_score']:.4f}")
            print(f"✅ Validation Passed: {results['validation_passed']}")
            
            # Save results
            output_file = f"validation_results_{args.port}.json"
            results['ply_file'] = args.validate_only
            results['validation_prompt'] = original_prompt
            with open(output_file, "w") as f:
                json.dump(results, f, indent=2)
            print(f"💾 Results saved to {output_file}")
            return
            
        elif args.separate_phases:
            # Separated phases mode - generate first, then validate
            print(f"🎨 Phase 1: Generating PLY data...")
            ply_file_path = generate_ply_with_cache(
                prompt=optimized_prompt,
                endpoint=endpoint,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                ss_sampling_steps=ss_sampling_steps,
                slat_sampling_steps=slat_sampling_steps,
                slat_guidance_strength=slat_guidance_strength,
                ss_guidance_strength=ss_guidance_strength,
                port=args.port,
                use_cache=use_ply_cache,
                output_file=args.output_file
            )
            
            print(f"🔍 Phase 2: Validating PLY file...")
            results = validate_ply_file(
                ply_file_path=ply_file_path,
                prompt=original_prompt,
                use_cached_models=args.use_cached_models,
                force_cpu=args.force_cpu
            )
            
            # Display results
            print(f"🎯 FINAL RESULTS (Separated Phases):")
            print(f"=" * 60)
            print(f"📝 Original Prompt: '{original_prompt}'")
            if optimized_prompt != original_prompt:
                print(f"🔧 Optimized Prompt: '{optimized_prompt}'")
            print(f"📁 PLY file: {ply_file_path}")
            print(f"🏆 Validation Engine Score: {results['validation_engine_score']:.4f}")
            print(f"🤝 Alignment Score: {results['alignment_score']:.4f}")
            print(f"💎 Quality Score: {results['quality_score']:.4f}")
            print(f"🎭 Demo Fidelity Score: {results['demo_fidelity_score']:.4f}")
            print(f"✅ Validation Passed: {results['validation_passed']}")
            
            # Save results
            output_file = f"subnet_validation_results_{args.port}.json"
            results['original_prompt'] = original_prompt
            results['optimized_prompt'] = optimized_prompt
            results['ply_file'] = ply_file_path
            results['separated_phases'] = True
            with open(output_file, "w") as f:
                json.dump(results, f, indent=2)
            print(f"💾 Results saved to {output_file}")
            return
        
        # Traditional combined mode (existing logic)
        print(f"🎨 Phase 1: Generating with TRELLIS")
        print(f"   Using prompt for generation: '{optimized_prompt}'")
        if 'image' in endpoint:
            # Check if we have pre-generated PLY data
            if args.pre_generated_ply and Path(args.pre_generated_ply).exists():
                print(f"📁 Using pre-generated PLY file: {args.pre_generated_ply}")
                with open(args.pre_generated_ply, 'rb') as f:
                    resp_bytes = f.read()
                print(f"📦 Loaded PLY data: {len(resp_bytes):,} bytes")
            else:
                # Generate PLY data normally
                resp_bytes = generate_and_get_ply_data(
                    optimized_prompt,
                    endpoint,
                    num_inference_steps=num_inference_steps,
                    guidance_scale=guidance_scale,
                    ss_sampling_steps=ss_sampling_steps,
                    slat_sampling_steps=slat_sampling_steps,
                    slat_guidance_strength=slat_guidance_strength,
                    ss_guidance_strength=ss_guidance_strength,
                    port=args.port,
                )
            # Step 2: CLIP scoring for image endpoint
            print(f"🔍 Phase 2: Computing CLIP alignment (text–text and text–image)")
            try:
                payload = json.loads(resp_bytes.decode('utf-8'))
            except Exception:
                print("❌ Could not parse image-generation response as JSON")
                raise
            b64img = payload.get('image') or payload.get('image_base64')
            if not b64img:
                print("❌ No 'image' field found in response JSON")
                raise RuntimeError("image field missing")
            img = Image.open(io.BytesIO(base64.b64decode(b64img))).convert('RGB')
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            model, tokenizer, normalize = load_validator_clip(device)
            tt_clip = clip_text_text(model, tokenizer, device, original_prompt, optimized_prompt)
            ti_clip = clip_text_image(model, tokenizer, normalize, device, original_prompt, img)
            # Step 3: Report results
            print(f"🎯 FINAL IMAGE-ENDPOINT RESULTS")
            print("=" * 60)
            print(f"📝 Original Prompt: '{original_prompt}'")
            if optimized_prompt != original_prompt:
                print(f"🔧 Optimized Prompt: '{optimized_prompt}'")
            print(f"🖼️ Image Size: {img.width}x{img.height}")
            print(f"🧮 tt_clip (text–text): {tt_clip:.4f}")
            print(f"🧮 ti_clip (text–image): {ti_clip:.4f}")
            print("=" * 60)

            # Save minimal results JSON
            output_file = f"subnet_validation_results_image_{args.port}.json"
            results_with_prompts = {
                'original_prompt': original_prompt,
                'optimized_prompt': optimized_prompt,
                'prompt_optimized': optimized_prompt != original_prompt,
                'tt_clip': tt_clip,
                'ti_clip': ti_clip,
                'endpoint_type': 'image'
            }
            with open(output_file, "w") as f:
                json.dump(results_with_prompts, f, indent=2)
            print(f"💾 Results saved to {output_file}")
        else:
            # Check if we have pre-generated PLY data
            if args.pre_generated_ply and Path(args.pre_generated_ply).exists():
                print(f"📁 Using pre-generated PLY file: {args.pre_generated_ply}")
                with open(args.pre_generated_ply, 'rb') as f:
                    ply_data = f.read()
                print(f"📦 Loaded PLY data: {len(ply_data):,} bytes")
            else:
                # Generate PLY data normally
                ply_data = generate_and_get_ply_data(
                    optimized_prompt,
                    endpoint,
                    num_inference_steps=num_inference_steps,
                    guidance_scale=guidance_scale,
                    ss_sampling_steps=ss_sampling_steps,
                    slat_sampling_steps=slat_sampling_steps,
                    slat_guidance_strength=slat_guidance_strength,
                    ss_guidance_strength=ss_guidance_strength,
                    port=args.port,
                )
            # Step 2: Validate using production logic against original prompt
            print(f"🔍 Phase 2: Running production-accurate validation")
            print(f"   Computing scores against original prompt: '{original_prompt}'")
            results = validate_with_production_logic(ply_data, original_prompt)

            # Step 3: Final results
            print(f"🎯 FINAL PRODUCTION-ACCURATE RESULTS")
            print(f"=" * 60)
            print(f"📝 Original Prompt: '{original_prompt}'")
            if optimized_prompt != original_prompt:
                print(f"🔧 Optimized Prompt: '{optimized_prompt}'")
            print(f"📊 PLY Size: {len(ply_data):,} bytes")
            results['ply_size'] = len(ply_data)
            print(f"🏆 Validation Engine Score: {results['validation_engine_score']:.4f}")
            print(f"🤝 Alignment Score: {results['alignment_score']:.4f}")
            print(f"💎 Quality Score: {results['quality_score']:.4f}")
            print(f"🎭 Demo Fidelity Score: {results['demo_fidelity_score']:.4f}")
            print(f"🎯 Task Fidelity Score: {results['task_fidelity_score']:.4f}")
            print(f"✅ Validation Passed: {results['validation_passed']}")
            print(f"🚧 Quality Threshold: {results['quality_threshold']}")
            print(f"📊 Alignment Threshold (0.3): {'✅' if results['alignment_threshold_passed'] else '❌'}")
            print(f"=" * 60)

            # Interpretation
            if results['demo_fidelity_score'] == 0.0:
                print("❌ SUBNET RESULT: ZERO TASK FIDELITY")
                print(f"   Reason: Validation score {results['validation_engine_score']:.4f} < 0.6")
            elif results['demo_fidelity_score'] == 0.75:
                print("🟡 SUBNET RESULT: MEDIUM FIDELITY (0.75)")
                print(f"   Validation score in range [0.6, 0.8): {results['validation_engine_score']:.4f}")
            elif results['demo_fidelity_score'] == 1.0:
                print("🟢 SUBNET RESULT: PERFECT FIDELITY (1.0)")
                print(f"   Validation score ≥ 0.8: {results['validation_engine_score']:.4f}")
            else:
                print(f"🔵 SUBNET RESULT: PARTIAL FIDELITY ({results['demo_fidelity_score']:.4f})")

            # Save results with prompt information
            output_file = f"subnet_validation_results_{args.port}.json"
            results['port'] = args.port
            results_with_prompts = {
                'original_prompt': original_prompt,
                'optimized_prompt': optimized_prompt,
                'prompt_optimized': optimized_prompt != original_prompt,
                **results
            }
            with open(output_file, "w") as f:
                json.dump(results_with_prompts, f, indent=2)
            print(f"💾 Results saved to {output_file}")
        
    except Exception as e:
        print(f"❌ Production-accurate validation workflow failed: {e}")
        sys.exit(1)

# Global variables to keep models loaded on GPU
_global_clip_model = None
_global_clip_tokenizer = None
_global_clip_normalize = None
_global_clip_device = None
_global_validator = None
_global_renderer = None
_global_ply_data_loader = None
_global_zstd_decompressor = None

def _ensure_models_loaded(force_cpu: bool = False):
    """Ensure all models are loaded and cached globally"""
    global _global_clip_model, _global_clip_tokenizer, _global_clip_normalize, _global_clip_device
    global _global_validator, _global_renderer, _global_ply_data_loader, _global_zstd_decompressor

    # Determine device
    if force_cpu:
        device = "cpu"
    else:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load CLIP model if not already loaded
    if _global_clip_model is None:
        _global_clip_device = torch.device(device)
        _global_clip_model, _global_clip_tokenizer, _global_clip_normalize = load_validator_clip(_global_clip_device)
        print(f"✅ CLIP model loaded on {_global_clip_device}")

    # Load validation components if not already loaded
    if _global_validator is None:
        _global_validator = ValidationEngine(verbose=False, device=device)  # Pass device parameter
        with suppress_stdout():
            _global_validator.load_pipelines()

        _global_zstd_decompressor = zstandard.ZstdDecompressor()
        _global_renderer = Renderer()
        _global_ply_data_loader = PlyLoader()

        print("✅ Production validation components loaded and cached")

def validate_prompt_direct(
    original_prompt: str,
    optimized_prompt: str = None,
    endpoint: str = "generate/",
    num_inference_steps: int = NUM_INFERENCE_STEPS,
    guidance_scale: float = GUIDANCE_SCALE,
    ss_sampling_steps: int = SS_SAMPLING_STEPS,
    slat_sampling_steps: int = SLAT_SAMPLING_STEPS,
    slat_guidance_strength: float = SLAT_GUIDANCE_STRENGTH,
    ss_guidance_strength: float = SS_GUIDANCE_STRENGTH,
    port: int = 8096,
    pre_generated_ply: bytes = None
) -> dict:
    """
    Direct validation function that keeps models loaded on GPU for speed.
    Returns the same format as the subprocess-based validation.

    Args:
        original_prompt: Prompt to compute validation scores against
        optimized_prompt: Prompt to use for generation (defaults to original_prompt)
        endpoint: Generation endpoint ("generate/" or "generate/image/")
        num_inference_steps: Inference steps for image generation
        guidance_scale: Guidance scale for image generation
        ss_sampling_steps: Sparse structure sampling steps
        slat_sampling_steps: SLAT sampling steps
        slat_guidance_strength: SLAT guidance strength
        ss_guidance_strength: SS guidance strength
        port: Server port
        pre_generated_ply: Pre-generated PLY data bytes (optional)

    Returns:
        dict: Validation results with all scores and metadata
    """
    try:
        # Ensure models are loaded
        _ensure_models_loaded()

        # Use optimized prompt for generation if provided, otherwise use original
        generation_prompt = optimized_prompt if optimized_prompt else original_prompt

        print(f"🎨 Direct validation: generating with '{generation_prompt}'")
        print(f"🎯 Computing scores against: '{original_prompt}'")

        # Step 1: Generate or use pre-generated PLY data
        if pre_generated_ply is not None:
            print("📁 Using pre-generated PLY data")
            ply_data = pre_generated_ply
        else:
            ply_data = generate_and_get_ply_data(
                generation_prompt,
                endpoint,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                ss_sampling_steps=ss_sampling_steps,
                slat_sampling_steps=slat_sampling_steps,
                slat_guidance_strength=slat_guidance_strength,
                ss_guidance_strength=ss_guidance_strength,
                port=port,
            )

        # Handle image endpoint vs 3D endpoint
        if 'image' in endpoint:
            # Parse image response
            try:
                payload = json.loads(ply_data.decode('utf-8'))
            except Exception:
                raise RuntimeError("Could not parse image-generation response as JSON")

            b64img = payload.get('image') or payload.get('image_base64')
            if not b64img:
                raise RuntimeError("No 'image' field found in response JSON")

            img = Image.open(io.BytesIO(base64.b64decode(b64img))).convert('RGB')

            # Compute CLIP scores using cached models
            tt_clip = clip_text_text(_global_clip_model, _global_clip_tokenizer, _global_clip_device,
                                   original_prompt, generation_prompt)
            ti_clip = clip_text_image(_global_clip_model, _global_clip_tokenizer, _global_clip_normalize,
                                    _global_clip_device, original_prompt, img)

            return {
                'original_prompt': original_prompt,
                'optimized_prompt': optimized_prompt,
                'prompt_optimized': optimized_prompt != original_prompt,
                'tt_clip': tt_clip,
                'ti_clip': ti_clip,
                'endpoint_type': 'image',
                'validation_method': 'direct',
                'image_size': f"{img.width}x{img.height}"
            }
        else:
            # 3D validation using cached production components
            print("🔬 Running production validation with cached components")

            # Create RequestData
            encoded_data = base64.b64encode(ply_data).decode('utf-8')
            request_data = RequestData(
                prompt=original_prompt,
                data=encoded_data,
                compression=2,  # SPZ compression
                generate_preview=False,
                preview_score_threshold=0.8
            )

            # Run validation with cached components
            validation_result = decode_and_validate_txt(
                request=request_data,
                ply_data_loader=_global_ply_data_loader,
                renderer=_global_renderer,
                validator=_global_validator
            )
            
            # Handle the result - it should be a tuple of (response, time_stats)
            if isinstance(validation_result, tuple):
                if len(validation_result) == 2:
                    response, time_stats = validation_result
                else:
                    print(f"   ⚠️ Unexpected tuple length: {len(validation_result)}")
                    response = validation_result[0]
                    time_stats = validation_result[1] if len(validation_result) > 1 else None
            else:
                print(f"   ⚠️ Unexpected result type: {type(validation_result)}")
                response = validation_result
                time_stats = None

            # Calculate demo fidelity score
            demo_fidelity_score = calculate_demo_fidelity_score(response.score)

            result = {
                'original_prompt': original_prompt,
                'optimized_prompt': optimized_prompt,
                'prompt_optimized': optimized_prompt != original_prompt,
                'validation_engine_score': response.score,
                'alignment_score': response.alignment_score,
                'quality_score': response.iqa,
                'ssim_score': response.ssim,
                'lpips_score': response.lpips,
                'demo_fidelity_score': demo_fidelity_score,
                'task_fidelity_score': response.score,
                'validation_passed': response.score > 0.0,
                'quality_threshold': 0.6,
                'alignment_threshold_passed': response.alignment_score >= 0.3,
                'production_logic_applied': True,
                'endpoint_type': '3d',
                'validation_method': 'direct',
                'ply_size': len(ply_data),
                'time_stats': {
                    'loading_time': time_stats.loading_data_time if time_stats else 0.0,
                    'rendering_time': time_stats.image_rendering_time if time_stats else 0.0,
                    'validation_time': time_stats.validation_time if time_stats else 0.0,
                    'total_time': time_stats.total_time if time_stats else 0.0,
                } if time_stats else None
            }

            print(f"🏆 Validation Engine Score: {response.score:.4f}")
            print(f"🤝 Alignment Score: {response.alignment_score:.4f}")
            return result

    except Exception as e:
        print(f"❌ Direct validation failed: {e}")
        import traceback
        traceback.print_exc()

        return {
            'original_prompt': original_prompt,
            'optimized_prompt': optimized_prompt,
            'prompt_optimized': optimized_prompt != original_prompt if optimized_prompt else False,
            'validation_engine_score': 0.0,
            'alignment_score': 0.0,
            'quality_score': 0.0,
            'ssim_score': 0.0,
            'lpips_score': 0.0,
            'demo_fidelity_score': 0.0,
            'task_fidelity_score': 0.0,
            'validation_passed': False,
            'quality_threshold': 0.6,
            'alignment_threshold_passed': False,
            'production_logic_applied': True,
            'endpoint_type': '3d' if 'image' not in endpoint else 'image',
            'validation_method': 'direct',
            'error': str(e)
        }

def unload_cached_models():
    """Unload cached models to free GPU memory"""
    global _global_clip_model, _global_clip_tokenizer, _global_clip_normalize, _global_clip_device
    global _global_validator, _global_renderer, _global_ply_data_loader, _global_zstd_decompressor

    try:
        # Clear CLIP models
        if _global_clip_model is not None:
            del _global_clip_model
            _global_clip_model = None
        if _global_clip_tokenizer is not None:
            del _global_clip_tokenizer
            _global_clip_tokenizer = None
        if _global_clip_normalize is not None:
            del _global_clip_normalize
            _global_clip_normalize = None
        _global_clip_device = None

        # Clear validation components
        if _global_validator is not None:
            with suppress_stdout():
                _global_validator.unload_pipelines()
            del _global_validator
            _global_validator = None

        if _global_renderer is not None:
            del _global_renderer
            _global_renderer = None

        if _global_ply_data_loader is not None:
            del _global_ply_data_loader
            _global_ply_data_loader = None

        if _global_zstd_decompressor is not None:
            del _global_zstd_decompressor
            _global_zstd_decompressor = None

        # Clear GPU cache
        gc.collect()
        torch.cuda.empty_cache()
        print("✅ Cached models unloaded and GPU memory cleared")

    except Exception as e:
        print(f"⚠️ Error unloading cached models: {e}")

def demo_direct_validation():
    """Demo function showing how to use validate_prompt_direct"""
    print("🚀 Demo: Direct Validation Function")
    print("=" * 50)

    try:
        # Example prompts
        original_prompt = "A beautiful red rose with dew drops"
        optimized_prompt = "A photorealistic red rose with morning dew, highly detailed petals, 8k"

        print(f"📝 Original: '{original_prompt}'")
        print(f"🔧 Optimized: '{optimized_prompt}'")
        print()

        # Validate using direct function (models stay loaded)
        result = validate_prompt_direct(
            original_prompt=original_prompt,
            optimized_prompt=optimized_prompt,
            endpoint="generate/",
            port=8096
        )

        print("\n📊 DIRECT VALIDATION RESULTS:")
        print("=" * 50)

        if result.get('endpoint_type') == 'image':
            print(f"🖼️ Image Endpoint Results:")
            print(f"   tt_clip: {result['tt_clip']:.4f}")
            print(f"   ti_clip: {result['ti_clip']:.4f}")
        else:
            print(f"🏆 Validation Engine Score: {result['validation_engine_score']:.4f}")
            print(f"🤝 Alignment Score: {result['alignment_score']:.4f}")
            print(f"💎 Quality Score: {result['quality_score']:.4f}")
            print(f"🎭 Demo Fidelity Score: {result['demo_fidelity_score']:.4f}")

        print(f"✅ Validation Passed: {result['validation_passed']}")
        print(f"🚀 Method: {result['validation_method']}")

        return result

    except Exception as e:
        print(f"❌ Demo failed: {e}")
        return None
    finally:
        # Optionally unload models when done
        # unload_cached_models()
        pass

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "--demo":
        demo_direct_validation()
    else:
        main() 