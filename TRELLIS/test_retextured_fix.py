#!/usr/bin/env python3
"""
Test script for the improved texture baking in TRELLIS with Hunyuan-style texture application.

This script demonstrates the three texture baking methods:
1. Hunyuan3D Paint Pipeline (most reliable) 
2. Simple spherical texture projection (fallback)
3. TRELLIS UV re-baking (last resort)
"""

import os
import datetime
from flux_trellis_retextured_optimized import FluxTrellisRetexturedOptimized

def test_single_prompt():
    """Test with a single prompt to verify texture baking works"""
    
    # Initialize pipeline
    pipeline = FluxTrellisRetexturedOptimized()
    
    # Choose quality level
    quality_level = 'standard'  # Use standard for faster testing
    
    # Test prompt
    test_prompt = "a blue monkey sitting on temple"
    
    # Create output directory
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"texture_fix_test_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)
    
    output_prefix = os.path.join(output_dir, "test_monkey")
    
    print(f"\n🧪 TESTING IMPROVED TEXTURE BAKING")
    print(f"=" * 60)
    print(f"📝 Prompt: '{test_prompt}'")
    print(f"🎯 Quality: {quality_level}")
    print(f"📁 Output: {output_dir}")
    print(f"=" * 60)
    
    try:
        result = pipeline.run_pipeline(
            prompt=test_prompt,
            quality=quality_level,
            output_prefix=output_prefix,
            seed=42
        )
        
        print(f"\n✅ TEST COMPLETE!")
        print(f"\n📁 Generated files:")
        for file_type, path in result['files'].items():
            print(f"  - {file_type}: {path}")
        
        # Check if retextured version was created
        if 'glb_retextured_optimized' in result['files']:
            print(f"\n🎉 SUCCESS: Retextured optimized GLB created!")
            print(f"   File: {result['files']['glb_retextured_optimized']}")
        else:
            print(f"\n⚠️  WARNING: Retextured optimized GLB was not created")
            
    except Exception as e:
        print(f"❌ TEST FAILED: {e}")
        raise

if __name__ == "__main__":
    test_single_prompt() 