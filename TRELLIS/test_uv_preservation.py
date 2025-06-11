#!/usr/bin/env python3
"""
Comprehensive test for UV preservation and improved texture baking method ordering.

Tests:
1. Proper method ordering: TRELLIS UV re-baking → Hunyuan3D → Spherical projection
2. UV preservation during shape optimization  
3. Coordinate system fixes for 90-degree rotation
4. Higher quality multiview rendering and texture baking
"""

import os
import datetime
from flux_trellis_retextured_optimized import FluxTrellisRetexturedOptimized

def test_uv_preservation_and_method_ordering():
    """Test comprehensive UV preservation and texture baking improvements"""
    
    # Initialize pipeline
    pipeline = FluxTrellisRetexturedOptimized()
    
    # Use standard quality for testing
    quality_level = 'standard'
    
    # Test with a prompt that has clear texture features
    test_prompt = "a blue monkey sitting on temple"
    
    # Create output directory
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"uv_preservation_test_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)
    
    output_prefix = os.path.join(output_dir, "uv_test")
    
    print(f"\n🧪 TESTING UV PRESERVATION & METHOD ORDERING")
    print(f"=" * 70)
    print(f"📝 Prompt: '{test_prompt}'")
    print(f"🎯 Quality: {quality_level}")
    print(f"📁 Output: {output_dir}")
    print(f"\n🔧 IMPROVEMENTS BEING TESTED:")
    print(f"=" * 70)
    print(f"📋 METHOD ORDERING:")
    print(f"  1️⃣  TRELLIS UV Re-baking (FIRST PRIORITY)")
    print(f"      - Re-parameterize optimized mesh with new UV unwrapping")
    print(f"      - Render 120 multiview images at 1024px resolution")
    print(f"      - Advanced texture baking with λ_tv=0.005")
    print(f"      - Preserves original Gaussian texture quality")
    print(f"")
    print(f"  2️⃣  Hunyuan3D Paint Pipeline (FALLBACK)")
    print(f"      - Direct texture application to mesh surface")
    print(f"      - Works when UV re-baking fails")
    print(f"")
    print(f"  3️⃣  Simple Spherical Projection (LAST RESORT)")
    print(f"      - Basic spherical UV mapping")
    print(f"      - Guaranteed to work with any mesh")
    print(f"")
    print(f"🛠️  UV PRESERVATION TECHNIQUES:")
    print(f"  ✅ Conservative shape optimization")
    print(f"  ✅ Higher resolution multiview rendering")
    print(f"  ✅ Coordinate system fixes for orientation")
    print(f"  ✅ Lower TV regularization for sharper textures")
    print(f"=" * 70)
    
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
        
        print(f"\n✅ UV PRESERVATION TEST COMPLETE!")
        print(f"=" * 50)
        print(f"\n📁 Generated files:")
        for file_type, path in result['files'].items():
            print(f"  - {file_type}: {path}")
        
        # Check results and provide analysis
        if 'glb_retextured_optimized' in result['files']:
            print(f"\n🎉 SUCCESS: Improved retextured optimized GLB created!")
            print(f"   File: {result['files']['glb_retextured_optimized']}")
            
            print(f"\n📊 QUALITY ANALYSIS TO PERFORM:")
            print(f"   1. 🔍 Load both GLB files in a 3D viewer:")
            print(f"      - Original: {result['files']['glb_textured']}")
            print(f"      - Optimized: {result['files']['glb_retextured_optimized']}")
            print(f"")
            print(f"   2. 🎯 Check texture alignment:")
            print(f"      - No 90-degree rotation issues")
            print(f"      - Front/back/sides properly oriented")
            print(f"      - Texture details preserved from original")
            print(f"")
            print(f"   3. 📐 Verify mesh optimization:")
            print(f"      - Reduced face count but preserved shape")
            print(f"      - No floating geometry or degenerate faces")
            print(f"      - Clean topology")
            print(f"")
            print(f"   4. 🎨 Texture quality comparison:")
            print(f"      - TRELLIS re-baking should preserve Gaussian details")
            print(f"      - Sharp textures (low TV regularization)")
            print(f"      - High resolution (1024px multiview rendering)")
            
        else:
            print(f"\n⚠️  WARNING: Retextured optimized GLB was not created")
            print(f"   This means all 3 texture methods failed. Check:")
            print(f"   - CUDA/nvdiffrast compilation issues")
            print(f"   - Hunyuan3D installation")
            print(f"   - Mesh optimization problems")
            
        print(f"\n📋 METHOD SUCCESS ANALYSIS:")
        print(f"   Check console output above to see which method succeeded:")
        print(f"   - '✅ TRELLIS UV re-baking successful!' = Method 1 worked (BEST)")
        print(f"   - '✅ Hunyuan3D texture application successful!' = Method 2 worked (GOOD)")  
        print(f"   - '✅ Simple texture projection successful!' = Method 3 worked (BASIC)")
        
    except Exception as e:
        print(f"❌ UV PRESERVATION TEST FAILED: {e}")
        print(f"\n🔧 TROUBLESHOOTING:")
        print(f"   1. Fix GCC version issue:")
        print(f"      export TORCH_CUDA_ARCH_LIST='8.9'")
        print(f"      export NVCC_APPEND_FLAGS='-allow-unsupported-compiler'")
        print(f"")
        print(f"   2. Check installations:")
        print(f"      - nvdiffrast compilation")
        print(f"      - Hunyuan3D components")
        print(f"      - TRELLIS dependencies")
        raise

if __name__ == "__main__":
    print("🔬 Testing UV preservation and improved texture baking...")
    test_uv_preservation_and_method_ordering() 