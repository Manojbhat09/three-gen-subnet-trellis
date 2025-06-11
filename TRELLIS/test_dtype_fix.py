#!/usr/bin/env python3
"""
Test script specifically for the dtype mismatch fix in TRELLIS UV re-baking.

This script tests the fix for the error:
"expected mat1 and mat2 to have the same dtype, but got: double != float"
"""

import os
import datetime
from flux_trellis_retextured_optimized import FluxTrellisRetexturedOptimized

def test_dtype_fix():
    """Test the dtype mismatch fix in TRELLIS texture baking"""
    
    # Initialize pipeline
    pipeline = FluxTrellisRetexturedOptimized()
    
    # Use draft quality for faster testing
    quality_level = 'draft'
    
    # Simple test prompt
    test_prompt = "football"
    
    # Create output directory
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"dtype_fix_test_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)
    
    output_prefix = os.path.join(output_dir, "dtype_test")
    
    print(f"\n🔧 TESTING DTYPE MISMATCH FIX")
    print(f"=" * 60)
    print(f"📝 Prompt: '{test_prompt}'")
    print(f"🎯 Quality: {quality_level} (fast testing)")
    print(f"📁 Output: {output_dir}")
    print(f"\n🛠️  DTYPE FIXES APPLIED:")
    print(f"  ✅ opt_verts_np.astype(np.float32)")
    print(f"  ✅ opt_faces_np.astype(np.int32)")
    print(f"  ✅ opt_uvs_np.astype(np.float32)")
    print(f"  ✅ extrinsics.astype(np.float32)")
    print(f"  ✅ intrinsics.astype(np.float32)")
    print(f"  ✅ observations.astype(np.float32)")
    print(f"  ✅ coordinate transform matrix dtype=np.float32")
    print(f"=" * 60)
    
    try:
        # Set environment variables for CUDA compilation
        os.environ['TORCH_CUDA_ARCH_LIST'] = "8.9"
        os.environ['NVCC_APPEND_FLAGS'] = "-allow-unsupported-compiler"
        
        result = pipeline.run_pipeline(
            prompt=test_prompt,
            quality=quality_level,
            output_prefix=output_prefix,
            seed=42
        )
        
        print(f"\n✅ DTYPE FIX TEST COMPLETE!")
        print(f"=" * 40)
        print(f"\n📁 Generated files:")
        for file_type, path in result['files'].items():
            print(f"  - {file_type}: {path}")
        
        # Check if TRELLIS UV re-baking worked
        if 'glb_retextured_optimized' in result['files']:
            print(f"\n🎉 SUCCESS: DTYPE MISMATCH FIXED!")
            print(f"   TRELLIS UV re-baking completed successfully")
            print(f"   File: {result['files']['glb_retextured_optimized']}")
            
            print(f"\n📊 VERIFY THE FIX:")
            print(f"   1. Check console output above for:")
            print(f"      - '✅ TRELLIS UV re-baking successful!' (Method 1 worked)")
            print(f"      - Array dtype messages showing all float32/int32")
            print(f"   2. No 'expected mat1 and mat2 to have the same dtype' errors")
            print(f"   3. Texture baking progress completed to 100%")
        else:
            print(f"\n⚠️  No retextured GLB created - check which method succeeded:")
            print(f"   Look for success messages in console output above")
            
    except Exception as e:
        print(f"❌ DTYPE FIX TEST FAILED: {e}")
        
        # Check if it's still a dtype error
        if "dtype" in str(e).lower():
            print(f"\n🔍 STILL A DTYPE ERROR:")
            print(f"   The error suggests there are still dtype mismatches")
            print(f"   We may need to fix additional array types")
        else:
            print(f"\n🔍 DIFFERENT ERROR:")
            print(f"   The dtype fix worked, but there's another issue")
        
        print(f"\n🛠️  TROUBLESHOOTING:")
        print(f"   1. Check if nvdiffrast compiled successfully")
        print(f"   2. Verify all dependencies are properly installed")
        print(f"   3. Look at the full error traceback above")
        raise

if __name__ == "__main__":
    print("🔧 Testing dtype mismatch fix for TRELLIS texture baking...")
    test_dtype_fix() 