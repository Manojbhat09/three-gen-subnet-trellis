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
            return response.content, compression
            
    except requests.exceptions.RequestException as e:
        print(f"❌ Generation request failed: {e}")
        raise
    except Exception as e:
        print(f"❌ Generation failed: {e}")
        raise

def validate_with_production_logic(ply_data: bytes, prompt: str, compression: int) -> dict:
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
        # demo_fidelity_score = calculate_demo_fidelity_score(response.score)
        
        print(f"🎭 Demo Fidelity Score: {demo_fidelity_score}")
        
        print(f"=" * 60)
        print(f"🏁 PRODUCTION-ACCURATE VALIDATION COMPLETE")
        print(f"=" * 60)
        
        return {
            'ply_data': ply_data,
            'compression': compression,
            'validation_engine_score': response.score,
            'alignment_score': response.alignment_score,
            'quality_score': response.iqa,
            'ssim_score': response.ssim,
            'lpips_score': response.lpips,
            # 'demo_fidelity_score': demo_fidelity_score,
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
            'ply_data': ply_data,
            'compression': compression,
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
def validate_with_production_logic_raw(ply_raw_data: bytes, prompt: str, compression: int) -> dict:
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
        # Step 1: Generation
        print(f"�� Phase 1: Generating with TRELLIS")
        print(f"   Using prompt for generation: '{optimized_prompt}'")
        if 'image' in endpoint:
            # Check if we have pre-generated PLY data
            if args.pre_generated_ply and Path(args.pre_generated_ply).exists():
                print(f"�� Using pre-generated PLY file: {args.pre_generated_ply}")
                with open(args.pre_generated_ply, 'rb') as f:
                    resp_bytes = f.read()
                print(f"📦 Loaded PLY data: {len(resp_bytes):,} bytes")
            else:
                # Generate PLY data normally
                resp_bytes, compression = generate_and_get_ply_data(
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
                ply_data, compression = generate_and_get_ply_data(
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
            results = validate_with_production_logic(ply_data, original_prompt, compression)

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

if __name__ == "__main__":
    main() 
