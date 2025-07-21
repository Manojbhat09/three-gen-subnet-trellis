#!/usr/bin/env python3
"""
Debug script to test the data flow between generation server and validation engine
to identify where the synchronization issue occurs.
"""
import requests
import base64
import sys
from pathlib import Path

# Add validation directory to path
validation_path = Path(__file__).parent / "validation"
sys.path.insert(0, str(validation_path))

# Try importing components
try:
    import pyspz
    print("✅ pyspz available")
except ImportError:
    print("❌ pyspz not available")
    sys.exit(1)

try:
    from engine.validation_engine import ValidationEngine
    from engine.rendering.renderer import Renderer
    from engine.io.ply import PlyLoader
    print("✅ Validation engine components available")
except ImportError as e:
    print(f"❌ Validation engine components not available: {e}")
    sys.exit(1)

def test_generation_server():
    """Test what the generation server actually returns"""
    print("\n" + "="*50)
    print("Testing Generation Server Response")
    print("="*50)
    
    url = "http://127.0.0.1:8096/generate/"
    prompt = "a simple red cube"
    
    try:
        print(f"🎨 Sending prompt: '{prompt}'")
        response = requests.post(url, data={'prompt': prompt}, timeout=300)
        response.raise_for_status()
        
        print(f"✅ Generation successful")
        print(f"📊 Response size: {len(response.content):,} bytes")
        print(f"📋 Content-Type: {response.headers.get('Content-Type', 'unknown')}")
        print(f"📋 Content-Disposition: {response.headers.get('Content-Disposition', 'unknown')}")
        print(f"📋 X-Compression: {response.headers.get('X-Compression', 'unknown')}")
        print(f"📋 X-Compression-Ratio: {response.headers.get('X-Compression-Ratio', 'unknown')}")
        
        # Check if it's binary data
        first_bytes = response.content[:50]
        print(f"📋 First 50 bytes: {first_bytes}")
        print(f"📋 Is likely SPZ compressed: {first_bytes.startswith(b'SPZ')}")
        
        return response.content
        
    except Exception as e:
        print(f"❌ Generation failed: {e}")
        return None

def test_decompression(compressed_data):
    """Test SPZ decompression"""
    print("\n" + "="*50)
    print("Testing SPZ Decompression")
    print("="*50)
    
    try:
        print(f"🔧 Attempting SPZ decompression of {len(compressed_data):,} bytes...")
        decompressed_data = pyspz.decompress(compressed_data)
        print(f"✅ Decompression successful")
        print(f"📊 Decompressed size: {len(decompressed_data):,} bytes")
        
        # Check PLY header
        header = decompressed_data[:100].decode('ascii', errors='ignore')
        print(f"📋 PLY header: {header[:50]}...")
        print(f"📋 Is valid PLY: {header.startswith('ply')}")
        
        return decompressed_data
        
    except Exception as e:
        print(f"❌ Decompression failed: {e}")
        return None

def test_ply_loading(ply_data):
    """Test PLY loading with validation engine"""
    print("\n" + "="*50)
    print("Testing PLY Loading")
    print("="*50)
    
    try:
        print("🔧 Initializing PLY loader...")
        ply_loader = PlyLoader()
        
        print(f"🔧 Loading PLY data ({len(ply_data):,} bytes)...")
        import io
        pcl_buffer = io.BytesIO(ply_data)
        gs_data = ply_loader.from_buffer(pcl_buffer)
        
        print(f"✅ PLY loading successful")
        print(f"📊 GS data type: {type(gs_data)}")
        print(f"📊 Has positions: {hasattr(gs_data, 'positions')}")
        if hasattr(gs_data, 'positions'):
            print(f"📊 Positions shape: {gs_data.positions.shape if hasattr(gs_data.positions, 'shape') else 'unknown'}")
        
        return gs_data
        
    except Exception as e:
        print(f"❌ PLY loading failed: {e}")
        import traceback
        traceback.print_exc()
        return None

def test_rendering(gs_data):
    """Test rendering with validation engine"""
    print("\n" + "="*50)
    print("Testing Rendering")
    print("="*50)
    
    try:
        print("🔧 Initializing renderer...")
        renderer = Renderer()
        
        print("🔧 Sending to device...")
        import torch
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        gs_data_gpu = gs_data.send_to_device(device)
        
        print(f"🔧 Rendering images on {device}...")
        images = renderer.render_gs(gs_data_gpu, 16, 224, 224)
        
        print(f"✅ Rendering successful")
        print(f"📊 Images type: {type(images)}")
        print(f"📊 Images shape: {images.shape if hasattr(images, 'shape') else 'unknown'}")
        
        return images
        
    except Exception as e:
        print(f"❌ Rendering failed: {e}")
        import traceback
        traceback.print_exc()
        return None

def test_validation(prompt, images):
    """Test validation with validation engine"""
    print("\n" + "="*50)
    print("Testing Validation")
    print("="*50)
    
    try:
        print("🔧 Initializing validation engine...")
        validator = ValidationEngine()
        validator.load_pipelines()
        
        print(f"🔧 Running validation for prompt: '{prompt}'...")
        validation_result = validator.validate_text_to_gs(prompt, images)
        
        print(f"✅ Validation successful")
        print(f"📊 Final score: {validation_result.final_score:.4f}")
        print(f"📊 LPIPS score: {validation_result.lpips_score:.4f}")
        print(f"📊 SSIM score: {validation_result.ssim_score:.4f}")
        print(f"📊 Quality score: {validation_result.combined_quality_score:.4f}")
        print(f"📊 Alignment score: {validation_result.alignment_score:.4f}")
        
        return validation_result
        
    except Exception as e:
        print(f"❌ Validation failed: {e}")
        import traceback
        traceback.print_exc()
        return None

def main():
    print("🚀 Starting Debug Test for Generation -> Validation Pipeline")
    
    # Step 1: Test generation server
    compressed_data = test_generation_server()
    if not compressed_data:
        print("❌ Cannot continue without generated data")
        return
    
    # Step 2: Test decompression
    ply_data = test_decompression(compressed_data)
    if not ply_data:
        print("❌ Cannot continue without decompressed data")
        return
    
    # Step 3: Test PLY loading
    gs_data = test_ply_loading(ply_data)
    if not gs_data:
        print("❌ Cannot continue without loaded GS data")
        return
    
    # Step 4: Test rendering
    images = test_rendering(gs_data)
    if images is None:
        print("❌ Cannot continue without rendered images")
        return
    
    # Step 5: Test validation
    prompt = "a simple red cube"
    validation_result = test_validation(prompt, images)
    if not validation_result:
        print("❌ Validation failed")
        return
    
    print("\n" + "="*50)
    print("🎉 ALL TESTS PASSED!")
    print("="*50)
    print(f"Final validation score: {validation_result.final_score:.4f}")

if __name__ == "__main__":
    main() 