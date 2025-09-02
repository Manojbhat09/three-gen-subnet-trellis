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
    from validation.engine.data_structures import RequestData, ValidationResultData, ValidationRequest
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
) -> bytes:
    """Generate 3D model using TRELLIS and return compressed PLY data"""
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
            
            # Return the raw response content (compressed or uncompressed)
            return response.content
            
    except requests.exceptions.RequestException as e:
        print(f"❌ Generation request failed: {e}")
        raise
    except Exception as e:
        print(f"❌ Generation failed: {e}")
        raise


def generate_and_get_ply_data_3view(
    prompt: str,
    endpoint: str,
    num_inference_steps: int = NUM_INFERENCE_STEPS,
    guidance_scale: float = GUIDANCE_SCALE,
    ss_sampling_steps: int = SS_SAMPLING_STEPS,
    slat_sampling_steps: int = SLAT_SAMPLING_STEPS,
    slat_guidance_strength: float = SLAT_GUIDANCE_STRENGTH,
    ss_guidance_strength: float = SS_GUIDANCE_STRENGTH,
    port: int = 8096,
    width: int = 1024,
    height: int = 1024,
    upscale: bool = False,
    remove_background: bool = True,
    filter_low_quality: bool = True,
    use_short_prompt: bool = True,
    style: str = "standard",
    image_endpoint: str = "standard",
    lora_model: str = None,
    return_compressed: bool = True,
    save_preview: bool = False,
    save_intermediate: bool = False,
    seed: int = 42
) -> bytes:
    """Generate 3D model using TRELLIS and return compressed PLY data"""
    url = f"http://127.0.0.1:{port}/{endpoint}"
    
    import requests
    print(f"🎨 Generating 3D model for: '{prompt}'")

    try:
        # Always include tuning params for both image and 3D generation endpoints
        payload = {
            "base_prompt": prompt,
            "num_inference_steps": num_inference_steps,
            "guidance_scale": guidance_scale,
            "ss_sampling_steps": ss_sampling_steps,
            "slat_sampling_steps": slat_sampling_steps,
            "slat_guidance_strength": slat_guidance_strength,
            "ss_guidance_strength": ss_guidance_strength,
            "width": width,
            "height": height,
            "upscale": upscale,
            "remove_background": remove_background,
            "filter_low_quality": filter_low_quality,
            "use_short_prompt": use_short_prompt,
            "style": style,
            "image_endpoint": image_endpoint,
            "return_compressed": return_compressed,
            "save_preview": save_preview,
            "save_intermediate": save_intermediate,
            "seed": seed
        }
        
        # Add LoRA model if specified
        if lora_model:
            payload["lora_model"] = lora_model
        with requests.post(url, data=payload, timeout=300, stream=False) as response:
            response.raise_for_status()
            
            compression = response.headers.get('x-compression', 'none')
            content_length = len(response.content)
            
            print(f"📦 Response received: {content_length:,} bytes (compression: {compression})")
            
            # Return the raw response content (compressed or uncompressed)
            return response.content
            
    except requests.exceptions.RequestException as e:
        print(f"❌ Generation request failed: {e}")
        raise
    except Exception as e:
        print(f"❌ Generation failed: {e}")
        raise

def validate_with_production_logic(ply_data: bytes, prompt: str, compression: int = 2) -> dict:
    """
    Validate using the exact production decode_and_validate_txt function
    This ensures 100% accuracy with production validation results
    
    Args:
        ply_data: PLY data bytes
        prompt: Text prompt for validation
        compression: Compression type (0=raw PLY, 2=SPZ compressed)
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
            compression=compression,  # SPZ compression (production standard)
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
            
            validation_result: ValidationResultData = decode_and_validate_txt(
                request=request_data,
                ply_data_loader=ply_data_loader,
                renderer=renderer,
                zstd_decompressor=zstd_decompressor,
                validator=validator,
                include_time_stat=True
            )
        except Exception as inner_e:
            print(f"   Inner error details: {inner_e}")
            import traceback
            traceback.print_exc()
            raise inner_e
        
        print(f"✅ Production validation completed")
        
        # Extract results from production format
        response = validation_result.response_data
        time_stats = validation_result.time_stat
        
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
        validation_result: ValidationResultData = decode_and_validate_txt(
            request=request_data,
            ply_data_loader=ply_data_loader,
            renderer=renderer,
            zstd_decompressor=zstd_decompressor,
            validator=validator,
            include_time_stat=True
        )
        response = validation_result.response_data
        time_stats = validation_result.time_stat
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
    parser.add_argument("--ply-file", type=str, help="Path to existing PLY file for validation (skips generation and uses file directly)")
    parser.add_argument("--raw", action="store_true", help="Force treat PLY file as raw (uncompressed) data")
    parser.add_argument("--test-reference", action="store_true", help="Test validation with reference SPZ file")
    parser.add_argument("--compare", type=str, help="Compare user PLY file with reference SPZ file")
    parser.add_argument("--validate", action="store_true", help="Run validation with full config parameters")
    parser.add_argument("--style", type=str, choices=["standard", "cinema", "3d"], default="standard", help="Style to use for generation")
    parser.add_argument("--width", type=int, default=1024, help="Image width for generation")
    parser.add_argument("--height", type=int, default=1024, help="Image height for generation")
    parser.add_argument("--upscale", action="store_true", help="Whether to upscale images using Real-ESRGAN")
    parser.add_argument("--remove_background", action="store_true", default=True, help="Whether to remove backgrounds from images")
    parser.add_argument("--filter_low_quality", action="store_true", default=True, help="Whether to filter low-quality Gaussians")
    parser.add_argument("--use_short_prompt", action="store_true", default=True, help="Whether to use short prompt to avoid CLIP token limits")
    parser.add_argument("--image_endpoint", type=str, choices=["standard", "cinema", "lora"], default="standard", help="Image generation endpoint")
    parser.add_argument("--lora_model", type=str, help="LoRA model to use if endpoint is 'lora'")
    parser.add_argument("--return_compressed", action="store_true", default=True, help="Whether to return compressed PLY")
    parser.add_argument("--save_preview", action="store_true", default=False, help="Whether to save preview video")
    parser.add_argument("--save_intermediate", action="store_true", default=False, help="Whether to save intermediate outputs")
    parser.add_argument("--seed", type=int, default=42, help="Seed to use for generation")
    args = parser.parse_args()

    original_prompt = args.original_prompt
    optimized_prompt = args.optimized_prompt if args.optimized_prompt is not None else args.original_prompt
    endpoint = args.endpoint
    num_inference_steps = args.num_inference_steps
    guidance_scale = args.guidance_scale
    ss_sampling_steps = args.ss_sampling_steps
    slat_sampling_steps = args.slat_sampling_steps
    slat_guidance_strength = args.slat_guidance_strength
    ss_guidance_strength = args.ss_guidance_strength
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
    print(f"=" * 60)
    
    try:
        # Check if we're testing the reference SPZ file
        if args.test_reference:
            print(f"🧪 REFERENCE SPZ TEST MODE")
            print(f"   Testing validation with known working SPZ file")
            print(f"   This will help diagnose validation pipeline issues")
            
            # Test reference SPZ file
            results = test_reference_spz_file()
            
            if results:
                print(f"\n🎯 REFERENCE TEST COMPLETED")
                print(f"   If this works but your PLY file doesn't, the issue is with your file")
                print(f"   If this also fails, the issue is with the validation pipeline")
            else:
                print(f"\n❌ REFERENCE TEST FAILED")
                print(f"   This indicates a fundamental issue with the validation pipeline")
            
            return

        # Check if we're comparing PLY files
        if args.compare:
            print(f"🔍 PLY FILE COMPARISON MODE")
            print(f"   User PLY: {args.compare}")
            print(f"   Reference SPZ: /home/mbhat/three-gen-subnet-trellis/trellis_submit_outputs/1756667456_42/compressed.ply.spz")
            
            # Compare the files
            compare_ply_files(
                reference_spz_path="/home/mbhat/three-gen-subnet-trellis/trellis_submit_outputs/1756667456_42/compressed.ply.spz",
                user_ply_path=args.compare,
                prompt="orange hut"  # Use the same prompt for fair comparison
            )
            
            return

        # Check if we're doing full validation with config
        if args.validate:
            print(f"🚀 FULL VALIDATION MODE")
            print(f"   Using all config parameters from test_grid_flow_endpoint.py")
            print(f"   Original Prompt: '{original_prompt}'")
            if optimized_prompt != original_prompt:
                print(f"   Optimized Prompt: '{optimized_prompt}'")
            args.endpoint = "/generate_3d_from_prompt_grid_flow/"
            print(f"   CHANGED: Using endpoint: {args.endpoint}")
            # Run full validation with config
            results = run_full_validation_with_config(
                original_prompt=original_prompt,
                optimized_prompt=optimized_prompt,
                style=args.style,
                seed=args.seed,  # Default seed
                num_inference_steps=args.num_inference_steps,
                guidance_scale=args.guidance_scale,
                width=args.width,
                height=args.height,
                upscale=args.upscale,
                remove_background=args.remove_background,
                ss_guidance_strength=args.ss_guidance_strength,
                ss_sampling_steps=args.ss_sampling_steps,
                slat_guidance_strength=args.slat_guidance_strength,
                slat_sampling_steps=args.slat_sampling_steps,
                filter_low_quality=args.filter_low_quality,
                use_short_prompt=args.use_short_prompt,
                image_endpoint=args.image_endpoint,
                lora_model=args.lora_model,
                port=args.port,
                endpoint=args.endpoint,
                return_compressed=args.return_compressed,
                save_preview=args.save_preview,
                save_intermediate=args.save_intermediate
            )
            
            # Display results
            if 'error' not in results:
                print(f"\n🎯 FULL VALIDATION RESULTS")
                print(f"=" * 60)
                print(f"📝 Original Prompt: '{original_prompt}'")
                if optimized_prompt != original_prompt:
                    print(f"🔧 Optimized Prompt: '{optimized_prompt}'")
                print(f"📊 PLY Size: {results['ply_size']:,} bytes")
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

                # Save results
                output_file = f"full_validation_results_{args.port}.json"
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
            else:
                print(f"❌ Full validation failed: {results['error']}")
            
            return

        # Check if we're doing PLY file validation
        if args.ply_file:
            print(f"📁 PLY FILE VALIDATION MODE")
            print(f"   File: {args.ply_file}")
            print(f"   Original Prompt: '{original_prompt}'")
            if optimized_prompt != original_prompt:
                print(f"   Optimized Prompt: '{optimized_prompt}'")
            print(f"   Skipping generation - using existing PLY file")
            if args.raw:
                print(f"   Raw mode: Forcing raw PLY handling (compression=0)")
            else:
                print(f"   Auto-detection: Will auto-detect compression type")
            
            # Run PLY file validation
            results = validate_ply_file_direct(
                ply_file_path=args.ply_file,
                original_prompt=original_prompt,
                optimized_prompt=optimized_prompt,
                compression=0 if args.raw else None  # Force raw if --raw flag is set
            )
            
            # Display results
            print(f"🎯 FINAL PLY FILE VALIDATION RESULTS")
            print(f"=" * 60)
            print(f"📁 PLY File: {args.ply_file}")
            print(f"📝 Original Prompt: '{original_prompt}'")
            if optimized_prompt != original_prompt:
                print(f"🔧 Optimized Prompt: '{optimized_prompt}'")
            print(f"📊 PLY Size: {results['ply_size']:,} bytes")
            print(f"🗜️ Compression Used: {results['compression_used']} ({'SPZ' if results['compression_used'] == 2 else 'Raw'})")
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

            # Save results
            output_file = f"subnet_validation_results_ply_file_{args.port}.json"
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
            
            return

        # Step 1: Generation (existing logic for non-PLY file mode)
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

def _ensure_models_loaded():
    """Ensure all models are loaded and cached globally"""
    global _global_clip_model, _global_clip_tokenizer, _global_clip_normalize, _global_clip_device
    global _global_validator, _global_renderer, _global_ply_data_loader, _global_zstd_decompressor

    # Load CLIP model if not already loaded
    if _global_clip_model is None:
        _global_clip_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        _global_clip_model, _global_clip_tokenizer, _global_clip_normalize = load_validator_clip(_global_clip_device)
        print(f"✅ CLIP model loaded on {_global_clip_device}")

    # Load validation components if not already loaded
    if _global_validator is None:
        _global_validator = ValidationEngine(verbose=False)  # Less verbose for direct calls
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
            validation_result: ValidationResultData = decode_and_validate_txt(
                request=request_data,
                ply_data_loader=_global_ply_data_loader,
                renderer=_global_renderer,
                zstd_decompressor=_global_zstd_decompressor,
                validator=_global_validator,
                include_time_stat=True
            )

            response = validation_result.response_data
            time_stats = validation_result.time_stat

            # Debug: Check what we got back
            print(f"🔍 DEBUG: Validation response details:")
            print(f"   Response type: {type(response)}")
            print(f"   Score: {response.score}")
            print(f"   Alignment score: {response.alignment_score}")
            print(f"   IQA: {response.iqa}")
            print(f"   SSIM: {response.ssim}")
            print(f"   LPIPS: {response.lpips}")
            
            # Check if any scores are NaN or inf
            import math
            if math.isnan(response.score) or math.isinf(response.score):
                print(f"⚠️ Warning: Score is NaN or inf: {response.score}")
            if math.isnan(response.alignment_score) or math.isinf(response.alignment_score):
                print(f"⚠️ Warning: Alignment score is NaN or inf: {response.alignment_score}")

            # If all scores are zero, try with different compression setting as fallback
            if response.score == 0.0 and response.alignment_score == 0.0 and response.iqa == 0.0:
                print("⚠️ All scores are zero - trying fallback compression setting...")
                
                # Try the opposite compression setting
                fallback_compression = 2 if compression == 0 else 0
                print(f"🔄 Trying fallback compression: {fallback_compression}")
                
                try:
                    fallback_request_data = RequestData(
                        prompt=original_prompt,
                        data=encoded_data,
                        compression=fallback_compression,
                        generate_preview=False,
                        preview_score_threshold=0.8
                    )
                    
                    print(f"🔄 Retrying with compression={fallback_compression}...")
                    fallback_result: ValidationResultData = decode_and_validate_txt(
                        request=fallback_request_data,
                        ply_data_loader=_global_ply_data_loader,
                        renderer=_global_renderer,
                        zstd_decompressor=_global_zstd_decompressor,
                        validator=_global_validator,
                        include_time_stat=True
                    )
                    
                    fallback_response = fallback_result.response_data
                    print(f"🔄 Fallback results:")
                    print(f"   Score: {fallback_response.score}")
                    print(f"   Alignment score: {fallback_response.alignment_score}")
                    print(f"   IQA: {fallback_response.iqa}")
                    
                    # Use fallback results if they're better
                    if fallback_response.score > 0.0 or fallback_response.alignment_score > 0.0:
                        print(f"✅ Fallback compression {fallback_compression} worked better!")
                        response = fallback_response
                        compression = fallback_compression
                    else:
                        print(f"❌ Fallback compression {fallback_compression} also failed")
                        
                except Exception as fallback_e:
                    print(f"⚠️ Fallback attempt failed: {fallback_e}")

            # Calculate demo fidelity score
            demo_fidelity_score = calculate_demo_fidelity_score(response.score)

            result = {
                'original_prompt': original_prompt,
                'optimized_prompt': optimized_prompt,
                'prompt_optimized': optimized_prompt != original_prompt if optimized_prompt else False,
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

def validate_ply_file_direct(ply_file_path: str, original_prompt: str, optimized_prompt: str = None, compression: int = None) -> dict:
    """Load a PLY file from disk and run validation without generation."""
    
    if optimized_prompt is None:
        optimized_prompt = original_prompt
    
    print(f"🔍 Validating PLY file: {ply_file_path}")
    print(f"📝 Original prompt: {original_prompt}")
    print(f"📝 Optimized prompt: {optimized_prompt}")
    print(f"🗜️ Compression: {compression}")
    
    # Load PLY file
    if not os.path.exists(ply_file_path):
        print(f"❌ PLY file not found: {ply_file_path}")
        return None
    
    # Read PLY file
    with open(ply_file_path, 'rb') as f:
        ply_data = f.read()
    
    print(f"📁 PLY file size: {len(ply_data)} bytes")
    
    # Preview PLY header
    header_preview = ply_data[:200].decode('utf-8', errors='ignore')
    print(f"📋 PLY header preview: {header_preview[:100]}...")
    
    # Auto-detect compression if not specified
    if compression is None:
        if ply_data.startswith(b'ply'):
            compression = 0  # Raw PLY
            print(f"🔍 Auto-detected: Raw PLY (compression=0)")
        else:
            compression = 2  # SPZ
            print(f"🔍 Auto-detected: SPZ compressed (compression=2)")
    
    # Handle raw PLY compression (server behavior)
    if compression == 0:
        print(f"📁 Raw PLY mode: No compression needed for production validation")
        # Don't compress - production validation expects uncompressed PLY data
        # The compression=0 flag tells the validation engine to treat it as raw PLY
    
    # Create validation request
    from validation.engine.data_structures import ValidationRequest
    
    # Encode binary data as base64 string
    import base64
    encoded_data = base64.b64encode(ply_data).decode('utf-8')
    
    request_data = ValidationRequest(
        prompt=original_prompt,
        data=encoded_data,  # Base64 encoded string
        compression=compression
    )
    
    print(f"🔍 DEBUG: RequestData structure:")
    print(f"   Prompt: {request_data.prompt}")
    print(f"   Data size: {len(request_data.data)} characters (base64)")
    print(f"   Compression: {request_data.compression}")
    
    try:
        # Run validation using production logic
        print(f"🚀 Running production validation...")
        
        # Use the original ply_data (before compression) for validation
        # The production logic will handle compression internally
        original_ply_data = ply_data
        # No need to re-read the file - ply_data is already the correct data
        
        results = validate_with_production_logic(original_ply_data, original_prompt, compression)
        print(f"✅ Production validation completed")
        
        # Add compression info to results
        results['compression_used'] = compression
        
        return results
            
    except Exception as e:
        print(f"❌ Validation failed: {e}")
        import traceback
        traceback.print_exc()
        return {
            'score': 0.0,
            'compression_used': compression,
            'validation_passed': False,
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

def demo_ply_file_validation():
    """Demo function showing how to use validate_ply_file_direct"""
    print("🚀 Demo: PLY File Direct Validation")
    print("=" * 50)

    try:
        # Example PLY file path and prompts
        ply_file_path = "/home/mbhat/three-gen-subnet-trellis/test_outputs/optimal_save_short_prompt_42/grid_flow_orange hut_42.ply"
        original_prompt = "A beautiful orange hut in a tropical setting"
        optimized_prompt = "A photorealistic orange tropical hut with palm trees, 8k detail"

        print(f"📁 PLY File: {ply_file_path}")
        print(f"📝 Original: '{original_prompt}'")
        print(f"🔧 Optimized: '{optimized_prompt}'")
        print()

        # Check if file exists
        if not os.path.exists(ply_file_path):
            print(f"⚠️ PLY file not found: {ply_file_path}")
            print("   Please update the path in the demo function to point to an existing PLY file")
            return None

        # Validate using the new PLY file function
        result = validate_ply_file_direct(
            ply_file_path=ply_file_path,
            original_prompt=original_prompt,
            optimized_prompt=optimized_prompt,
            compression=0  # Raw PLY file
        )

        print("\n📊 PLY FILE VALIDATION RESULTS:")
        print("=" * 50)
        print(f"📁 File: {result['ply_file_path']}")
        print(f"🏆 Validation Engine Score: {result['validation_engine_score']:.4f}")
        print(f"🤝 Alignment Score: {result['alignment_score']:.4f}")
        print(f"💎 Quality Score: {result['quality_score']:.4f}")
        print(f"🎭 Demo Fidelity Score: {result['demo_fidelity_score']:.4f}")
        print(f"✅ Validation Passed: {result['validation_passed']}")
        print(f"🚀 Method: {result['validation_method']}")
        print(f"🗜️ Compression: {result['compression_used']}")

        return result

    except Exception as e:
        print(f"❌ Demo failed: {e}")
        return None

def test_reference_spz_file():
    """Test validation with the reference SPZ file to compare results"""
    print("🔍 Testing Reference SPZ File")
    print("=" * 50)
    
    reference_spz_path = "/home/mbhat/three-gen-subnet-trellis/trellis_submit_outputs/1756667456_42/compressed.ply.spz"
    
    if not os.path.exists(reference_spz_path):
        print(f"❌ Reference SPZ file not found: {reference_spz_path}")
        return None
    
    print(f"📁 Reference SPZ: {reference_spz_path}")
    
    # Test with a simple prompt
    test_prompt = "test object"
    
    try:
        # Load and validate the reference SPZ file
        results = validate_ply_file_direct(
            ply_file_path=reference_spz_path,
            original_prompt=test_prompt,
            optimized_prompt=test_prompt,
            compression=2  # Force SPZ compression
        )
        
        print(f"\n📊 REFERENCE SPZ VALIDATION RESULTS:")
        print(f"=" * 50)
        print(f"📁 File: {reference_spz_path}")
        print(f"📝 Prompt: '{test_prompt}'")
        print(f"📊 File Size: {results['ply_size']:,} bytes")
        print(f"🗜️ Compression: {results['compression_used']} (SPZ)")
        print(f"🏆 Validation Engine Score: {results['validation_engine_score']:.4f}")
        print(f"🤝 Alignment Score: {results['alignment_score']:.4f}")
        print(f"💎 Quality Score: {results['quality_score']:.4f}")
        print(f"🎭 Demo Fidelity Score: {results['demo_fidelity_score']:.4f}")
        print(f"✅ Validation Passed: {results['validation_passed']}")
        
        return results
        
    except Exception as e:
        print(f"❌ Reference SPZ validation failed: {e}")
        import traceback
        traceback.print_exc()
        return None

def compare_ply_files(reference_spz_path: str, user_ply_path: str, prompt: str):
    """Compare validation results between reference SPZ and user PLY file"""
    print("🔍 COMPARING PLY FILES")
    print("=" * 60)
    
    print(f"📁 Reference SPZ: {reference_spz_path}")
    print(f"📁 User PLY: {user_ply_path}")
    print(f"📝 Prompt: '{prompt}'")
    print("=" * 60)
    
    try:
        # Test reference SPZ
        print("\n🧪 Testing Reference SPZ...")
        reference_results = validate_ply_file_direct(
            ply_file_path=reference_spz_path,
            original_prompt=prompt,
            optimized_prompt=prompt,
            compression=2  # SPZ
        )
        
        print("\n🧪 Testing User PLY...")
        user_results = validate_ply_file_direct(
            ply_file_path=user_ply_path,
            original_prompt=prompt,
            optimized_prompt=prompt,
            compression=0  # Raw PLY
        )
        
        # Compare results
        print("\n📊 COMPARISON RESULTS")
        print("=" * 60)
        print(f"{'Metric':<25} {'Reference SPZ':<15} {'User PLY':<15} {'Difference':<15}")
        print("-" * 60)
        
        metrics = [
            ('Validation Score', 'validation_engine_score'),
            ('Alignment Score', 'alignment_score'),
            ('Quality Score', 'quality_score'),
            ('Demo Fidelity', 'demo_fidelity_score'),
            ('File Size (MB)', 'ply_size', lambda x: x/1024/1024)
        ]
        
        for metric_name, key, *transform in metrics:
            ref_val = reference_results.get(key, 0)
            user_val = user_results.get(key, 0)
            
            if transform:
                ref_val = transform[0](ref_val)
                user_val = transform[0](user_val)
            
            diff = user_val - ref_val
            diff_str = f"{diff:+.4f}" if isinstance(diff, float) else f"{diff:+d}"
            
            print(f"{metric_name:<25} {ref_val:<15.4f} {user_val:<15.4f} {diff_str:<15}")
        
        print("=" * 60)
        
        # Analysis
        print("\n🔍 ANALYSIS:")
        if user_results['validation_engine_score'] > 0:
            print("✅ User PLY validation is working!")
            if user_results['validation_engine_score'] < reference_results['validation_engine_score']:
                print("⚠️  But scores are lower than reference - quality difference")
            else:
                print("🎉 User PLY scores are better than reference!")
        else:
            print("❌ User PLY validation is failing completely")
            print("   This suggests a fundamental issue with the file or format")
        
        return reference_results, user_results
        
    except Exception as e:
        print(f"❌ Comparison failed: {e}")
        import traceback
        traceback.print_exc()
        return None, None

def run_full_validation_with_config(
    original_prompt: str,
    optimized_prompt: str,
    style: str = "standard",
    seed: int = 42,
    num_inference_steps: int = 8,
    guidance_scale: float = 3.5,
    width: int = 1024,
    height: int = 1024,
    upscale: bool = False,
    remove_background: bool = True,
    ss_guidance_strength: float = 7.5,
    ss_sampling_steps: int = 21,
    slat_guidance_strength: float = 4.0,
    slat_sampling_steps: int = 24,
    filter_low_quality: bool = True,
    use_short_prompt: bool = True,
    image_endpoint: str = "standard",
    lora_model: str = None,
    port: int = 8096,
    endpoint: str = "generate/",
    timing: bool = False,
    return_compressed: bool = True,
    save_preview: bool = False,
    save_intermediate: bool = False
) -> dict:
    """
    Run full validation with all config parameters from test_grid_flow_endpoint.py
    """
    print("🚀 FULL VALIDATION WITH CONFIG PARAMETERS")
    print("=" * 60)
    print(f"📝 Original Prompt: '{original_prompt}'")
    print(f"🔧 Optimized Prompt: '{optimized_prompt}'")
    print(f"🎨 Style: {style}")
    print(f"🌱 Seed: {seed}")
    print(f"⚙️ Config Parameters:")
    print(f"   - Inference Steps: {num_inference_steps}")
    print(f"   - Guidance Scale: {guidance_scale}")
    print(f"   - Image Size: {width}x{height}")
    print(f"   - Upscale: {upscale}")
    print(f"   - Remove Background: {remove_background}")
    print(f"   - SS Guidance: {ss_guidance_strength} (steps: {ss_sampling_steps})")
    print(f"   - SLAT Guidance: {slat_guidance_strength} (steps: {slat_sampling_steps})")
    print(f"   - Filter Low Quality: {filter_low_quality}")
    print(f"   - Use Short Prompt: {use_short_prompt}")
    print(f"   - Image Endpoint: {image_endpoint}")
    if lora_model:
        print(f"   - LoRA Model: {lora_model}")
    print(f"🔌 Server: Port {port}, Endpoint: {endpoint}")
    print("=" * 60)
    
    try:
        # Generate 3D model using the full config
        print("\n🎨 Phase 1: Generating 3D model with full config...")
        ply_data = generate_and_get_ply_data_3view(
            prompt=optimized_prompt,
            endpoint=endpoint,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            ss_sampling_steps=ss_sampling_steps,
            slat_sampling_steps=slat_sampling_steps,
            slat_guidance_strength=slat_guidance_strength,
            ss_guidance_strength=ss_guidance_strength,
            port=port,
            width=width,
            height=height,
            upscale=upscale,
            remove_background=remove_background,
            filter_low_quality=filter_low_quality,
            use_short_prompt=use_short_prompt,
            style=style,
            image_endpoint=image_endpoint,
            lora_model=lora_model,
            return_compressed=return_compressed,
            save_preview=save_preview,
            save_intermediate=save_intermediate,
            seed=seed
        )
        
        print(f"✅ 3D model generated successfully ({len(ply_data):,} bytes)")
        
        # Phase 2: Validate using production logic
        print(f"\n🔍 Phase 2: Running production-accurate validation...")
        print(f"   Computing scores against original prompt: '{original_prompt}'")
        if return_compressed:
            compression = 2
        else:
            compression = 0
        results = validate_with_production_logic(ply_data, original_prompt, compression)
        
        # Add config metadata to results
        results.update({
            'config_parameters': {
                'style': style,
                'seed': seed,
                'num_inference_steps': num_inference_steps,
                'guidance_scale': guidance_scale,
                'width': width,
                'height': height,
                'upscale': upscale,
                'remove_background': remove_background,
                'ss_guidance_strength': ss_guidance_strength,
                'ss_sampling_steps': ss_sampling_steps,
                'slat_guidance_strength': slat_guidance_strength,
                'slat_sampling_steps': slat_sampling_steps,
                'filter_low_quality': filter_low_quality,
                'use_short_prompt': use_short_prompt,
                'image_endpoint': image_endpoint,
                'lora_model': lora_model,
                'port': port,
                'endpoint': endpoint,
                'return_compressed': return_compressed,
                'save_preview': save_preview,
                'save_intermediate': save_intermediate
            },
            'validation_mode': 'full_config',
            'ply_size': len(ply_data)
        })
        
        return results
        
    except Exception as e:
        print(f"❌ Full validation failed: {e}")
        import traceback
        traceback.print_exc()
        
        return {
            'error': str(e),
            'validation_mode': 'full_config',
            'config_parameters': {
                'style': style,
                'seed': seed,
                'num_inference_steps': num_inference_steps,
                'guidance_scale': guidance_scale,
                'width': width,
                'height': height,
                'upscale': upscale,
                'remove_background': remove_background,
                'ss_guidance_strength': ss_guidance_strength,
                'ss_sampling_steps': ss_sampling_steps,
                'slat_guidance_strength': slat_guidance_strength,
                'slat_sampling_steps': slat_sampling_steps,
                'filter_low_quality': filter_low_quality,
                'use_short_prompt': use_short_prompt,
                'image_endpoint': image_endpoint,
                'lora_model': lora_model,
                'port': port,
                'endpoint': endpoint,
                'return_compressed': return_compressed,
                'save_preview': save_preview,
                'save_intermediate': save_intermediate
            }
        }

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "--demo":
        demo_direct_validation()
    elif len(sys.argv) > 1 and sys.argv[1] == "--demo-ply":
        demo_ply_file_validation()
    else:
        main() 
