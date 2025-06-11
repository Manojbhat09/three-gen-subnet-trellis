#!/usr/bin/env python3
"""
Test script for orientation/coordinate fixes in TRELLIS retexturing.

This script tests the fixes for the 90-degree Y-axis rotation issue:
1. Corrected spherical UV mapping coordinates  
2. Fixed coordinate transformation matrix
3. Texture orientation correction
"""

import os
import datetime
from flux_trellis_retextured_optimized import FluxTrellisRetexturedOptimized

def test_orientation_fixes():
    """Test the orientation fixes with a simple prompt"""
    
    # Initialize pipeline
    pipeline = FluxTrellisRetexturedOptimized()
    
    # Use standard quality for testing
    quality_level = 'standard'
    
    # Test with a prompt that has clear directional features
    test_prompt = "a blue monkey sitting on temple"
    
    # Create output directory
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"orientation_fix_test_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)
    
    output_prefix = os.path.join(output_dir, "orientation_test")
    
    print(f"\n🧪 TESTING ORIENTATION FIXES")
    print(f"=" * 60)
    print(f"📝 Prompt: '{test_prompt}'")
    print(f"🎯 Quality: {quality_level}")
    print(f"📁 Output: {output_dir}")
    print(f"\n🔧 APPLIED FIXES:")
    print(f"  ✅ Corrected spherical coordinates (x,z for azimuth, y for elevation)")
    print(f"  ✅ Fixed coordinate transformation matrix") 
    print(f"  ✅ Added 90-degree texture rotation correction")
    print(f"  ✅ UV coordinate rotation (270 degrees)")
    print(f"  ✅ V-coordinate flipping for proper orientation")
    print(f"=" * 60)
    
    try:
        result = pipeline.run_pipeline(
            prompt=test_prompt,
            quality=quality_level,
            output_prefix=output_prefix,
            seed=42
        )
        
        print(f"\n✅ ORIENTATION TEST COMPLETE!")
        print(f"\n📁 Generated files:")
        for file_type, path in result['files'].items():
            print(f"  - {file_type}: {path}")
        
        # Check if retextured version was created
        if 'glb_retextured_optimized' in result['files']:
            print(f"\n🎉 SUCCESS: Fixed retextured optimized GLB created!")
            print(f"   File: {result['files']['glb_retextured_optimized']}")
            print(f"\n📋 WHAT TO CHECK:")
            print(f"   1. Load the GLB files in a 3D viewer (e.g., Blender, glTF Viewer)")
            print(f"   2. Compare '_textured.glb' vs '_retextured_optimized.glb'")
            print(f"   3. Verify the texture is properly aligned (no 90-degree rotation)")
            print(f"   4. Check that front/back/sides match the expected orientation")
        else:
            print(f"\n⚠️  WARNING: Retextured optimized GLB was not created")
            print(f"   Check console output for specific error messages")
            
    except Exception as e:
        print(f"❌ ORIENTATION TEST FAILED: {e}")
        print(f"\nDEBUG INFO:")
        print(f"   - Make sure GCC compiler issue is fixed first")
        print(f"   - Check if Hunyuan3D components are properly installed")
        print(f"   - Verify CUDA and nvdiffrast compilation works")
        raise

if __name__ == "__main__":
    print("🔧 Testing orientation fixes for 90-degree Y-axis rotation issue...")
    test_orientation_fixes() 