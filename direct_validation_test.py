#!/usr/bin/env python3
"""
Direct Validation Test
Purpose: Test validation directly on an existing PLY file
"""

import json
import subprocess
import os
from pathlib import Path

def test_direct_validation():
    """Test validation on an existing PLY file"""
    print("🔍 Direct Validation Test")
    print("=" * 50)
    
    # Check if we have a PLY file to validate
    validation_outputs = Path("./validation_outputs")
    if not validation_outputs.exists():
        print("❌ No validation_outputs directory found")
        return
    
    files = list(validation_outputs.glob("*.ply.spz"))
    if not files:
        print("❌ No PLY files found in validation_outputs")
        return
    
    # Use the first file
    ply_file = files[0]
    print(f"📁 Using PLY file: {ply_file}")
    
    # Copy the file to the current directory for validation
    import shutil
    shutil.copy(ply_file, "./test_validation.ply.spz")
    print("📋 Copied file to ./test_validation.ply.spz")
    
    # Test validation with just the original prompt (no generation)
    original_prompt = "greek amphora scene detail"
    
    print(f"\n🔍 Running validation on existing file...")
    print(f"Original prompt: '{original_prompt}'")
    
    # We need to modify the validation script to skip generation
    # For now, let's create a simple validation test
    print("\n📊 Testing validation components...")
    
    try:
        # Test if we can import validation components
        import sys
        from pathlib import Path
        
        # Add validation directory to path (append so project root takes precedence)
        validation_path = Path(__file__).parent / "validation"
        sys.path.append(str(validation_path))
        
        try:
            import pyspz
            print("✅ pyspz library available")
        except ImportError:
            print("❌ pyspz library not available")
            return
        
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
            return
        
        # Read the PLY file
        with open("./test_validation.ply.spz", 'rb') as f:
            ply_data = f.read()
        
        print(f"📦 PLY file size: {len(ply_data):,} bytes")
        
        # Initialize validation components
        print("🔧 Initializing validation components...")
        validator = ValidationEngine(verbose=True)
        validator.load_pipelines()
        
        zstd_decompressor = zstandard.ZstdDecompressor()
        renderer = Renderer()
        ply_data_loader = PlyLoader()
        
        # Clear GPU memory
        import gc
        gc.collect()
        torch.cuda.empty_cache()
        
        print("✅ Validation components initialized")
        
        # Prepare request data
        import base64
        encoded_data = base64.b64encode(ply_data).decode('utf-8')
        
        request_data = RequestData(
            prompt=original_prompt,
            data=encoded_data,
            compression=2,  # SPZ compression
            generate_preview=False,
            preview_score_threshold=0.8
        )
        
        print(f"📊 RequestData prepared:")
        print(f"   Validation Prompt: '{original_prompt}'")
        print(f"   Data size: {len(encoded_data):,} characters (base64)")
        print(f"   Compression: 2 (SPZ)")
        
        # Run validation
        print("🚀 Running production validation...")
        validation_result = decode_and_validate_txt(
            request=request_data,
            ply_data_loader=ply_data_loader,
            renderer=renderer,
            zstd_decompressor=zstd_decompressor,
            validator=validator,
            include_time_stat=True
        )
        
        print("✅ Production validation completed")
        
        # Extract results
        response = validation_result.response_data
        time_stats = validation_result.time_stat
        
        print(f"📊 VALIDATION RESULTS:")
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
        
        # Save results
        results = {
            'validation_engine_score': response.score,
            'alignment_score': response.alignment_score,
            'quality_score': response.iqa,
            'ssim_score': response.ssim,
            'lpips_score': response.lpips,
            'validation_passed': response.score > 0.0,
            'time_stats': {
                'loading_time': time_stats.loading_data_time if time_stats else 0.0,
                'rendering_time': time_stats.image_rendering_time if time_stats else 0.0,
                'validation_time': time_stats.validation_time if time_stats else 0.0,
                'total_time': time_stats.total_time if time_stats else 0.0,
            } if time_stats else None
        }
        
        with open("direct_validation_results.json", "w") as f:
            json.dump(results, f, indent=2)
        
        print(f"💾 Results saved to: direct_validation_results.json")
        
        # Cleanup
        validator.unload_pipelines()
        gc.collect()
        torch.cuda.empty_cache()
        
    except Exception as e:
        print(f"❌ Validation failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_direct_validation() 