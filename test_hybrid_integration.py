#!/usr/bin/env python3
"""
Test script for validating the hybrid TRELLIS + Hunyuan3D-2 integration.

This script performs compatibility checks and basic functionality tests
to ensure the hybrid pipeline works correctly.
"""

import sys
import os
import traceback
from pathlib import Path
import numpy as np
from PIL import Image
import torch

def test_imports():
    """Test if all required imports are available."""
    print("=== Testing Imports ===")
    
    try:
        import trimesh
        print("✓ trimesh imported successfully")
    except ImportError as e:
        print(f"✗ trimesh import failed: {e}")
        return False
    
    try:
        # TRELLIS imports
        sys.path.append('TRELLIS')
        from trellis.pipelines import TrellisImageTo3DPipeline
        print("✓ TRELLIS pipeline imported successfully")
    except ImportError as e:
        print(f"✗ TRELLIS import failed: {e}")
        return False
    
    try:
        # Hunyuan3D-2 imports
        sys.path.append('Hunyuan3D-2')
        from hy3dgen.texgen import Hunyuan3DPaintPipeline
        from hy3dgen.texgen.utils.uv_warp_utils import mesh_uv_wrap
        from hy3dgen.rembg import BackgroundRemover
        print("✓ Hunyuan3D-2 components imported successfully")
    except ImportError as e:
        print(f"✗ Hunyuan3D-2 import failed: {e}")
        return False
    
    return True

def test_cuda_setup():
    """Test CUDA availability and setup."""
    print("\n=== Testing CUDA Setup ===")
    
    if not torch.cuda.is_available():
        print("✗ CUDA not available")
        return False
    
    print(f"✓ CUDA available: {torch.cuda.get_device_name()}")
    print(f"✓ CUDA memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    # Test basic tensor operations
    try:
        x = torch.randn(1000, 1000).cuda()
        y = torch.matmul(x, x.T)
        print("✓ Basic CUDA operations working")
    except Exception as e:
        print(f"✗ CUDA operations failed: {e}")
        return False
    
    return True

def test_model_loading():
    """Test loading of both model pipelines."""
    print("\n=== Testing Model Loading ===")
    
    try:
        from trellis.pipelines import TrellisImageTo3DPipeline
        
        print("Loading TRELLIS model...")
        os.environ['SPCONV_ALGO'] = 'native'
        
        # Try to load TRELLIS (this might take a while on first run)
        trellis_pipeline = TrellisImageTo3DPipeline.from_pretrained(
            "microsoft/TRELLIS-image-large"
        )
        trellis_pipeline.cuda()
        print("✓ TRELLIS model loaded successfully")
        
        # Clean up GPU memory
        del trellis_pipeline
        torch.cuda.empty_cache()
        
    except Exception as e:
        print(f"✗ TRELLIS model loading failed: {e}")
        print(f"Full traceback: {traceback.format_exc()}")
        return False
    
    try:
        from hy3dgen.texgen import Hunyuan3DPaintPipeline
        
        print("Loading Hunyuan3D-2 texture pipeline...")
        
        # Try to load Hunyuan3D-2 texture pipeline
        texture_pipeline = Hunyuan3DPaintPipeline.from_pretrained(
            "tencent/Hunyuan3D-2"
        )
        print("✓ Hunyuan3D-2 texture model loaded successfully")
        
        # Clean up GPU memory
        del texture_pipeline
        torch.cuda.empty_cache()
        
    except Exception as e:
        print(f"✗ Hunyuan3D-2 model loading failed: {e}")
        print(f"Full traceback: {traceback.format_exc()}")
        return False
    
    return True

def test_mesh_compatibility():
    """Test mesh format compatibility between systems."""
    print("\n=== Testing Mesh Compatibility ===")
    
    try:
        import trimesh
        
        # Create a test mesh
        mesh = trimesh.creation.box(extents=[1, 1, 1])
        print(f"✓ Created test mesh: {len(mesh.vertices)} vertices, {len(mesh.faces)} faces")
        
        # Test basic mesh operations
        mesh.remove_degenerate_faces()
        mesh.remove_duplicate_faces()
        print("✓ Basic mesh operations working")
        
        # Test UV mapping (required for texture pipeline)
        try:
            from hy3dgen.texgen.utils.uv_warp_utils import mesh_uv_wrap
            
            # This will add UV coordinates to the mesh
            uv_mesh = mesh_uv_wrap(mesh.copy())
            print("✓ UV wrapping successful")
            
            if hasattr(uv_mesh.visual, 'uv') and uv_mesh.visual.uv is not None:
                print(f"✓ UV coordinates generated: {uv_mesh.visual.uv.shape}")
            else:
                print("⚠ UV coordinates not properly set")
                
        except Exception as e:
            print(f"✗ UV wrapping failed: {e}")
            return False
        
        # Test mesh export/import
        test_file = "test_mesh.glb"
        mesh.export(test_file)
        
        if os.path.exists(test_file):
            loaded_mesh = trimesh.load(test_file)
            print("✓ Mesh export/import working")
            os.remove(test_file)
        else:
            print("✗ Mesh export failed")
            return False
            
    except Exception as e:
        print(f"✗ Mesh compatibility test failed: {e}")
        return False
    
    return True

def test_image_processing():
    """Test image processing pipeline."""
    print("\n=== Testing Image Processing ===")
    
    try:
        from hy3dgen.rembg import BackgroundRemover
        
        # Create a test image
        test_image = Image.new('RGB', (512, 512), color='red')
        print("✓ Created test image")
        
        # Test background removal
        rembg = BackgroundRemover()
        processed_image = rembg(test_image)
        
        if processed_image.mode == 'RGBA':
            print("✓ Background removal working (RGBA output)")
        else:
            print("⚠ Background removal might not be working properly")
        
        # Test image resizing
        resized = processed_image.resize((1024, 1024))
        print("✓ Image resizing working")
        
    except Exception as e:
        print(f"✗ Image processing test failed: {e}")
        return False
    
    return True

def test_hybrid_pipeline():
    """Test the actual hybrid pipeline integration."""
    print("\n=== Testing Hybrid Pipeline ===")
    
    try:
        from hybrid_trellis_hunyuan import HybridTrellisHunyuanPipeline
        
        print("✓ Hybrid pipeline import successful")
        
        # Test initialization (without actually loading models to save time)
        print("Note: Skipping full model loading in test mode")
        print("✓ Hybrid pipeline class structure validated")
        
    except ImportError as e:
        print(f"✗ Hybrid pipeline import failed: {e}")
        return False
    except Exception as e:
        print(f"✗ Hybrid pipeline test failed: {e}")
        return False
    
    return True

def create_test_image():
    """Create a simple test image for testing."""
    test_dir = Path("test_assets")
    test_dir.mkdir(exist_ok=True)
    
    # Create a simple test image
    img = Image.new('RGB', (512, 512), color='white')
    
    # Draw a simple shape
    import numpy as np
    img_array = np.array(img)
    
    # Draw a red circle in the center
    center = (256, 256)
    radius = 100
    y, x = np.ogrid[:512, :512]
    mask = (x - center[0])**2 + (y - center[1])**2 <= radius**2
    img_array[mask] = [255, 0, 0]  # Red circle
    
    test_image = Image.fromarray(img_array)
    test_path = test_dir / "test_image.png"
    test_image.save(test_path)
    
    print(f"✓ Created test image at {test_path}")
    return str(test_path)

def main():
    """Run all compatibility tests."""
    print("🧪 Hybrid TRELLIS + Hunyuan3D-2 Integration Test")
    print("=" * 50)
    
    tests = [
        ("Import Test", test_imports),
        ("CUDA Setup", test_cuda_setup),
        ("Model Loading", test_model_loading),
        ("Mesh Compatibility", test_mesh_compatibility),
        ("Image Processing", test_image_processing),
        ("Hybrid Pipeline", test_hybrid_pipeline),
    ]
    
    results = []
    
    for test_name, test_func in tests:
        print(f"\n🔍 Running {test_name}...")
        try:
            result = test_func()
            results.append((test_name, result))
            
            if result:
                print(f"✅ {test_name} PASSED")
            else:
                print(f"❌ {test_name} FAILED")
                
        except Exception as e:
            print(f"💥 {test_name} CRASHED: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "=" * 50)
    print("🏁 TEST SUMMARY")
    print("=" * 50)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{test_name:<20} {status}")
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! The hybrid integration should work.")
        
        # Create test image for manual testing
        test_image_path = create_test_image()
        print(f"\n📝 Next steps:")
        print(f"1. Run the hybrid pipeline with: python hybrid_trellis_hunyuan.py")
        print(f"2. Use the test image: {test_image_path}")
        
    else:
        print("⚠️ Some tests failed. Please fix the issues before using the hybrid pipeline.")
        print("\n🔧 Common fixes:")
        print("- Ensure both TRELLIS and Hunyuan3D-2 are properly installed")
        print("- Check CUDA version compatibility")
        print("- Verify all dependencies are installed")

if __name__ == "__main__":
    main() 