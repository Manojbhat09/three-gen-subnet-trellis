#!/usr/bin/env python3
"""
Simple Local Validator for TRELLIS
Purpose: Test generation server and local validation engine workflow
"""
import requests
import sys
import os
import contextlib
from io import StringIO
from pathlib import Path

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

# Import validation components
try:
    from engine.validation_engine import ValidationEngine
    from engine.io.ply import PlyLoader
    from engine.rendering.renderer import Renderer
    print("✅ Validation engine components available")
except ImportError as e:
    print(f"❌ Validation engine components not available: {e}")
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

def decompress_spz_data(compressed_data: bytes) -> bytes:
    """Decompress SPZ-compressed data"""
    try:
        print(f"📦 Decompressing {len(compressed_data):,} bytes...")
        
        # Perform decompression with stdout suppressed to avoid binary output
        with suppress_stdout():
            decompressed_data = pyspz.decompress(compressed_data, False)  # False = don't include normals
        
        print(f"✅ Decompressed to {len(decompressed_data):,} bytes")
        compression_ratio = len(compressed_data) / len(decompressed_data)
        print(f"📊 Compression ratio: {compression_ratio:.2f}")
        
        # Verify PLY format
        if decompressed_data.startswith(b'ply\n'):
            print("✅ Valid PLY format detected")
            # Show first few lines of PLY header
            lines = decompressed_data.decode('utf-8', errors='ignore').split('\n')[:5]
            for i, line in enumerate(lines):
                print(f"   {i+1}: {line}")
        else:
            print("⚠️ PLY format not detected")
        
        return decompressed_data
        
    except Exception as e:
        print(f"❌ Decompression failed: {e}")
        raise

def generate_and_decompress(prompt: str) -> bytes:
    """Generate model and handle decompression if needed"""
    try:
        print(f"🎨 Generating model for prompt: '{prompt}'")
        
        url = "http://127.0.0.1:8096/generate/"
        
        with requests.post(url, data={'prompt': prompt}, timeout=120, stream=False) as response:
            response.raise_for_status()
            
            print(f"✅ Generation completed")
            print(f"📊 Response size: {len(response.content):,} bytes")
            print(f"📋 Content type: {response.headers.get('content-type', 'unknown')}")
            
            # Check compression header
            compression = response.headers.get('x-compression', 'none')
            print(f"📦 Compression: {compression}")
            
            # Store response content safely (no printing)
            response_data = response.content
        
        # Handle decompression if needed
        if compression == 'spz':
            return decompress_spz_data(response_data)
        else:
            print("📦 Data is not compressed")
            return response_data
            
    except Exception as e:
        print(f"❌ Generation failed: {e}")
        raise

def validate_with_local_engine(ply_data: bytes, prompt: str) -> float:
    """Validate using the local validation engine"""
    try:
        print(f"🔍 Initializing local validation engine...")
        
        # Initialize components
        validator = ValidationEngine()
        validator.load_pipelines()
        
        ply_loader = PlyLoader()
        renderer = Renderer()
        
        print(f"🔍 Loading PLY data...")
        # Load PLY data from bytes using BytesIO buffer
        import io
        ply_buffer = io.BytesIO(ply_data)
        gs_data = ply_loader.from_buffer(ply_buffer)
        
        print(f"📊 GS Data loaded:")
        print(f"   - Points shape: {gs_data.points.shape}")
        print(f"   - Features_dc shape: {gs_data.features_dc.shape}")
        print(f"   - Opacities shape: {gs_data.opacities.shape}")
        print(f"   - Scales shape: {gs_data.scales.shape}")
        print(f"   - Rotations shape: {gs_data.rotations.shape}")
        
        print(f"🖼️ Rendering views...")
        # Move GS data to GPU device for rendering
        gs_data_gpu = gs_data.send_to_device(validator.device)
        print(f"📊 GS Data moved to device: {validator.device}")
        
        # Render multiple views (using standard validation resolution 224x224)
        try:
            rendered_images = renderer.render_gs(gs_data_gpu, views_number=16, img_width=224, img_height=224)
            print(f"✅ Successfully rendered {len(rendered_images)} views")
        except Exception as render_error:
            print(f"❌ Rendering failed with error: {render_error}")
            print(f"   Error type: {type(render_error).__name__}")
            import traceback
            traceback.print_exc()
            raise
        
        print(f"🧮 Computing validation scores...")
        # Validate using the engine
        try:
            validation_results = validator.validate_text_to_gs(prompt, rendered_images)
        except Exception as validation_error:
            print(f"❌ Validation computation failed with error: {validation_error}")
            print(f"   Error type: {type(validation_error).__name__}")
            import traceback
            traceback.print_exc()
            raise
        
        # Cleanup
        validator.unload_pipelines()
        
        # Extract scores
        final_score = validation_results.final_score
        alignment_score = validation_results.alignment_score
        quality_score = validation_results.combined_quality_score
        
        print(f"✅ Local validation completed")
        print(f"📊 Final Score: {final_score:.4f}")
        print(f"📊 Alignment Score: {alignment_score:.4f}")  
        print(f"📊 Quality Score: {quality_score:.4f}")
        
        return final_score
        
    except Exception as e:
        print(f"❌ Local validation failed: {e}")
        import traceback
        traceback.print_exc()
        raise

def main():
    if len(sys.argv) < 2:
        print("Usage: python simple_local_validator.py \"<prompt>\" [--save-compressed]")
        sys.exit(1)
    
    prompt = sys.argv[1]
    save_compressed = "--save-compressed" in sys.argv
    
    print(f"🚀 Testing prompt: '{prompt}'")
    print("=" * 60)
    
    try:
        # Special mode to save compressed data for debugging
        if save_compressed:
            print(f"💾 Saving compressed data for prompt: '{prompt}'")
            url = "http://127.0.0.1:8096/generate/"
            with requests.post(url, data={'prompt': prompt}, timeout=120, stream=False) as response:
                response.raise_for_status()
                compression = response.headers.get('x-compression', 'none')
                if compression == 'spz':
                    with open("compressed_output.spz", "wb") as f:
                        f.write(response.content)
                    print(f"✅ Saved {len(response.content):,} bytes to compressed_output.spz")
                else:
                    print(f"⚠️ No SPZ compression detected ({compression})")
            return
        
        # Step 1: Generate and decompress
        print(f"🎨 Step 1: Generating and decompressing model")
        ply_data = generate_and_decompress(prompt)
        
        # Step 2: Validate locally using validation engine
        print(f"🔍 Step 2: Running local validation")
        score = validate_with_local_engine(ply_data, prompt)
        
        # Step 3: Results summary
        print("=" * 60)
        print("🎯 VALIDATION COMPLETE")
        print("=" * 60)
        print(f"📝 Prompt: '{prompt}'")
        print(f"📊 PLY Size: {len(ply_data):,} bytes") 
        print(f"🏆 Final Score: {score:.4f}")
        print("=" * 60)
        
        if score > 0.7:
            print("🌟 EXCELLENT - High quality model!")
        elif score > 0.5:
            print("✅ GOOD - Acceptable quality model")
        elif score > 0.3:
            print("⚠️ FAIR - Low quality model")
        else:
            print("❌ POOR - Very low quality model")
            
    except Exception as e:
        print(f"❌ Validation workflow failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 