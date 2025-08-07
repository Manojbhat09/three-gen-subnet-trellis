#!/usr/bin/env python3
"""
Subnet-Accurate Local Validator v2.0
Purpose: Use the exact decode_and_validate_txt function from benchmark validation 
to match production validation logic exactly, resolving validation discrepancies.
"""
import subprocess
import sys
import os
import contextlib
import base64
import time
import gc
from io import StringIO
from pathlib import Path
import json

# Add validation directory to path
validation_path = Path(__file__).parent / "validation"
sys.path.insert(0, str(validation_path))

# Test pyspz availability
try:
    import pyspz
    print("✅ pyspz library available")
except ImportError:
    print("❌ pyspz library not available")
    sys.exit(1)

# Import production validation components  
try:
    from engine.data_structures import RequestData, ValidationResultData
    from engine.io.ply.loader import PlyLoader
    from engine.rendering.renderer import Renderer
    from engine.validation_engine import ValidationEngine
    from serve import decode_and_validate_txt
    import zstandard
    import torch
    print("✅ Production validation components available")
except ImportError as e:
    print(f"❌ Production validation components not available: {e}")
    sys.exit(1)

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

def generate_and_get_ply_data(prompt: str, endpoint: str) -> bytes:
    """Generate 3D model using TRELLIS and return compressed PLY data"""
    url = f"http://127.0.0.1:8096/{endpoint}"
    
    import requests
    print(f"🎨 Generating 3D model for: '{prompt}'")
    
    try:
        with requests.post(url, data={'prompt': prompt}, timeout=300, stream=False) as response:
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
        validation_result: ValidationResultData = decode_and_validate_txt(
            request=request_data,
            ply_data_loader=ply_data_loader,
            renderer=renderer,
            zstd_decompressor=zstd_decompressor,
            validator=validator,
            include_time_stat=True
        )
        
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
    if len(sys.argv) < 2:
        print("Usage: python subnet_accurate_validator.py \"<original_prompt>\" [\"<optimized_prompt>\"]")
        print("  - original_prompt: The prompt to compute validation scores against")
        print("  - optimized_prompt (optional): The prompt to use for generation")
        print("    If not provided, original_prompt will be used for both generation and validation")
        sys.exit(1)
    
    original_prompt = sys.argv[1]
    optimized_prompt = sys.argv[2] if len(sys.argv) > 2 else original_prompt
    endpoint = sys.argv[3] if len(sys.argv) > 3 else "generate/"
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
        # Step 1: Generate and get PLY data using optimized prompt (or original if no optimization)
        print(f"🎨 Phase 1: Generating model with TRELLIS")
        print(f"   Using prompt for generation: '{optimized_prompt}'")
        ply_data = generate_and_get_ply_data(optimized_prompt, endpoint)
        
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
        output_file = f"subnet_validation_results.json"
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