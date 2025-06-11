#!/usr/bin/env python3
"""
Quick test to verify BPT integration works correctly in the TRELLIS pipeline.
"""

import os
import datetime
from flux_trellis_bpt_retextured_optimized import FluxTrellisBPTRetexturedOptimized

def test_bpt_integration():
    """Test that BPT integration works correctly"""
    
    # Set environment variables first
    os.environ['TORCH_CUDA_ARCH_LIST'] = "8.9"
    os.environ['NVCC_APPEND_FLAGS'] = "-allow-unsupported-compiler"
    
    # Initialize pipeline
    pipeline = FluxTrellisBPTRetexturedOptimized()
    
    # Use draft quality for fastest testing
    quality_level = 'draft'
    test_prompt = "football"
    
    # Create output directory
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"bpt_integration_test_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)
    
    output_prefix = os.path.join(output_dir, "bpt_test")
    
    print(f"\n🔬 TESTING BPT INTEGRATION")
    print(f"=" * 60)
    print(f"📝 Prompt: '{test_prompt}'")
    print(f"🎯 Quality: {quality_level}")
    print(f"📁 Output: {output_dir}")
    print(f"\n🔬 BPT Available: {pipeline.bpt_enhancer.model is not None}")
    print(f"🔧 Shape Optimization Available: {pipeline.shape_optimizer.face_reducer is not None}")
    print(f"🎨 Texture Painting Available: {hasattr(pipeline, 'apply_hunyuan_texture')}")
    print(f"=" * 60)
    
    try:
        result = pipeline.run_pipeline(
            prompt=test_prompt,
            quality=quality_level,
            output_prefix=output_prefix,
            seed=42
        )
        
        print(f"\n✅ BPT INTEGRATION TEST COMPLETE!")
        print(f"📁 Generated files:")
        for file_type, path in result['files'].items():
            print(f"  - {file_type}: {path}")
        
        # Check specifically for BPT-enhanced output
        if 'glb_bpt_retextured_optimized' in result['files']:
            print(f"\n🎉 SUCCESS: BPT enhancement pipeline working!")
            print(f"   Created BPT-enhanced and re-textured GLB")
        else:
            print(f"\n⚠️  BPT-enhanced GLB not created")
            print(f"   Check which methods succeeded in console output above")
        
        # Check for standard output for comparison
        if 'glb_textured' in result['files']:
            print(f"✅ Standard textured GLB also created for comparison")
            
    except Exception as e:
        print(f"❌ BPT INTEGRATION TEST FAILED: {e}")
        print(f"   Check dependencies and BPT model weights")
        
        # Still useful to know what components are available
        print(f"\n📊 COMPONENT STATUS:")
        print(f"- BPT Model Loaded: {pipeline.bpt_enhancer.model is not None}")
        print(f"- Shape Optimizer: {pipeline.shape_optimizer.face_reducer is not None}")
        print(f"- Background Remover: {pipeline.bg_remover is not None}")

if __name__ == "__main__":
    print("🔬 Testing BPT integration in TRELLIS pipeline...")
    test_bpt_integration() 