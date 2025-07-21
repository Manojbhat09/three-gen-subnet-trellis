#!/usr/bin/env python3
"""
Local Validator Client for TRELLIS
Purpose: Provides an interactive command-line tool to test prompts against the
         local generation server and validate directly using the validation engine,
         allowing for rapid prompt engineering and optimization.
"""
import requests
import argparse
import time
import base64
import logging
import sys
import os
import io
import gc
from pathlib import Path

# Add validation directory to path
validation_path = Path(__file__).parent / "validation"
sys.path.insert(0, str(validation_path))

# Attempt to import pyspz, provide guidance if it fails
try:
    import pyspz
    SPZ_AVAILABLE = True
    print("✅ pyspz library available")
except ImportError:
    SPZ_AVAILABLE = False
    print("❌ pyspz library not available - decompression will fail")
    sys.exit(1)

# Try importing validation components
try:
    from engine.validation_engine import ValidationEngine
    from engine.rendering.renderer import Renderer
    from engine.io.ply import PlyLoader
    VALIDATION_AVAILABLE = True
    print("✅ Validation engine components available")
except ImportError as e:
    VALIDATION_AVAILABLE = False
    print(f"❌ Validation components not available: {e}")
    print("Will attempt to use validation server instead")

# --- Configuration ---
GENERATION_SERVER_URL = "http://127.0.0.1:8096"
VALIDATION_SERVER_URL = "http://127.0.0.1:10006"
GENERATION_ENDPOINT = f"{GENERATION_SERVER_URL}/generate/"
CLEAR_CACHE_ENDPOINT = f"{GENERATION_SERVER_URL}/clear_cache/"
VALIDATION_ENDPOINT = f"{VALIDATION_SERVER_URL}/validate_txt_to_3d_ply/"

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('local_validation_client.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def validate_with_local_engine(ply_data: bytes, prompt: str) -> float:
    """Validate using the local validation engine"""
    if not VALIDATION_AVAILABLE:
        raise RuntimeError("Validation engine not available")
    
    try:
        # Create validation engine
        engine = ValidationEngine()
        
        # Load PLY data
        loader = PlyLoader()
        model = loader.load_from_bytes(ply_data)
        
        # Render and validate
        renderer = Renderer()
        images = renderer.render_multiple_views(model, num_views=16)
        
        # Calculate validation score
        score = engine.compute_text_to_image_similarity(prompt, images)
        
        return score
        
    except Exception as e:
        logger.error(f"Local validation failed: {e}")
        raise

def validate_with_server(ply_data: bytes, prompt: str) -> float:
    """Validate using the validation server"""
    try:
        files = {'ply_file': ('model.ply', ply_data, 'application/octet-stream')}
        data = {'prompt': prompt}
        
        response = requests.post(VALIDATION_ENDPOINT, files=files, data=data, timeout=120)
        response.raise_for_status()
        
        result = response.json()
        return result.get('score', 0.0)
        
    except Exception as e:
        logger.error(f"Server validation failed: {e}")
        raise

def decompress_spz_data(compressed_data: bytes) -> bytes:
    """Decompress SPZ-compressed data"""
    if not SPZ_AVAILABLE:
        raise RuntimeError("pyspz not available for decompression")
    
    try:
        logger.info(f"🔄 Attempting to decompress {len(compressed_data):,} bytes of SPZ data...")
        decompressed_data = pyspz.decompress(compressed_data)
        logger.info(f"✅ SPZ decompression successful! Decompressed to {len(decompressed_data):,} bytes")
        
        # Verify it looks like PLY data
        if decompressed_data.startswith(b'ply\n') or decompressed_data.startswith(b'ply\r\n'):
            logger.info("✅ Decompressed data is valid PLY format")
        else:
            logger.warning("⚠️ Decompressed data may not be valid PLY format")
            logger.info(f"First 100 bytes: {decompressed_data[:100]}")
        
        return decompressed_data
        
    except Exception as e:
        logger.error(f"❌ SPZ decompression failed: {e}")
        raise

def generate_model(prompt: str) -> bytes:
    """Generate a 3D model using the TRELLIS generation server"""
    try:
        logger.info(f"🎨 Generating model for prompt: '{prompt}'")
        
        # Send generation request
        response = requests.post(GENERATION_ENDPOINT, data={'prompt': prompt}, timeout=120)
        response.raise_for_status()
        
        logger.info(f"✅ Generation completed successfully")
        logger.info(f"📊 Response size: {len(response.content):,} bytes")
        
        # Check compression headers
        compression = response.headers.get('x-compression', 'none')
        if compression == 'spz':
            logger.info("📦 Received SPZ-compressed data")
            return decompress_spz_data(response.content)
        else:
            logger.info(f"📦 Received uncompressed data (compression: {compression})")
            return response.content
            
    except Exception as e:
        logger.error(f"❌ Generation failed: {e}")
        raise

def clear_generation_cache():
    """Clear the generation server's GPU memory cache"""
    try:
        response = requests.post(CLEAR_CACHE_ENDPOINT, timeout=30)
        if response.status_code == 200:
            logger.info("🧹 Generation server cache cleared")
        else:
            logger.warning(f"⚠️ Cache clear returned status {response.status_code}")
    except Exception as e:
        logger.warning(f"⚠️ Failed to clear cache: {e}")

def validate_prompt(prompt: str) -> dict:
    """Complete validation workflow for a single prompt"""
    start_time = time.time()
    
    try:
        # Step 1: Generate model
        ply_data = generate_model(prompt)
        generation_time = time.time() - start_time
        
        # Step 2: Clear cache to free GPU memory
        clear_generation_cache()
        
        # Step 3: Validate
        validation_start = time.time()
        
        score = None
        validation_method = None
        
        # Try local validation first (if available)
        if VALIDATION_AVAILABLE:
            try:
                score = validate_with_local_engine(ply_data, prompt)
                validation_method = "local_engine"
            except Exception as e:
                logger.warning(f"Local validation failed: {e}")
                logger.info("Falling back to validation server...")
        
        # Fall back to server validation
        if score is None:
            try:
                score = validate_with_server(ply_data, prompt)
                validation_method = "validation_server"
            except Exception as e:
                logger.error(f"Server validation also failed: {e}")
                # Try CPU-only validation as last resort
                try:
                    os.environ['CUDA_VISIBLE_DEVICES'] = ''
                    score = validate_with_local_engine(ply_data, prompt)
                    validation_method = "local_engine_cpu"
                except Exception as cpu_e:
                    logger.error(f"CPU validation failed: {cpu_e}")
                    raise RuntimeError("All validation methods failed")
        
        validation_time = time.time() - validation_start
        total_time = time.time() - start_time
        
        # Log results
        result = {
            'prompt': prompt,
            'score': score,
            'generation_time': generation_time,
            'validation_time': validation_time,
            'total_time': total_time,
            'validation_method': validation_method,
            'ply_size': len(ply_data)
        }
        
        logger.info(f"📊 Validation Results:")
        logger.info(f"   Prompt: {prompt}")
        logger.info(f"   Score: {score:.4f}")
        logger.info(f"   Generation Time: {generation_time:.2f}s")
        logger.info(f"   Validation Time: {validation_time:.2f}s")
        logger.info(f"   Total Time: {total_time:.2f}s")
        logger.info(f"   Method: {validation_method}")
        logger.info(f"   PLY Size: {len(ply_data):,} bytes")
        
        return result
        
    except Exception as e:
        logger.error(f"❌ Validation workflow failed: {e}")
        return {
            'prompt': prompt,
            'error': str(e),
            'total_time': time.time() - start_time
        }

def interactive_mode():
    """Interactive prompt testing mode"""
    print("\n🎯 TRELLIS Local Validator - Interactive Mode")
    print("=" * 50)
    print("Commands:")
    print("  Enter a prompt to test")
    print("  'quit' or 'q' to exit")
    print("  'help' or 'h' for this help")
    print("=" * 50)
    
    session_results = []
    
    while True:
        try:
            prompt = input("\n💬 Enter prompt: ").strip()
            
            if prompt.lower() in ['quit', 'q', 'exit']:
                break
            elif prompt.lower() in ['help', 'h']:
                print("\nCommands:")
                print("  Enter a prompt to test")
                print("  'quit' or 'q' to exit")
                print("  'help' or 'h' for this help")
                continue
            elif not prompt:
                print("⚠️ Empty prompt, please try again")
                continue
            
            print(f"\n🚀 Testing prompt: '{prompt}'")
            result = validate_prompt(prompt)
            session_results.append(result)
            
            if 'error' in result:
                print(f"❌ Error: {result['error']}")
            else:
                score = result['score']
                print(f"\n📊 Score: {score:.4f}")
                
                if score >= 0.6:
                    print("✅ PASS (score >= 0.6)")
                else:
                    print("❌ FAIL (score < 0.6)")
                    print("💡 Consider:")
                    print("   - Simpler, more concrete descriptions")
                    print("   - Avoid subjective terms (beautiful, amazing)")
                    print("   - Use common materials (wood, metal, plastic)")
                    print("   - Be specific about shape and function")
            
        except KeyboardInterrupt:
            print("\n\n👋 Interrupted by user")
            break
        except Exception as e:
            logger.error(f"Interactive mode error: {e}")
            print(f"❌ Error: {e}")
    
    # Print session summary
    if session_results:
        print(f"\n📋 Session Summary ({len(session_results)} prompts tested):")
        passed = sum(1 for r in session_results if 'score' in r and r['score'] >= 0.6)
        failed = len(session_results) - passed
        print(f"   ✅ Passed: {passed}")
        print(f"   ❌ Failed: {failed}")
        if passed > 0:
            avg_score = sum(r['score'] for r in session_results if 'score' in r) / len([r for r in session_results if 'score' in r])
            print(f"   📊 Average Score: {avg_score:.4f}")

def main():
    parser = argparse.ArgumentParser(description="TRELLIS Local Validator Client")
    parser.add_argument("prompt", nargs='?', help="Prompt to test (if not provided, enters interactive mode)")
    parser.add_argument("--interactive", "-i", action="store_true", help="Force interactive mode")
    
    args = parser.parse_args()
    
    if args.interactive or not args.prompt:
        interactive_mode()
    else:
        result = validate_prompt(args.prompt)
        if 'error' in result:
            print(f"❌ Error: {result['error']}")
            sys.exit(1)
        else:
            score = result['score']
            print(f"Score: {score:.4f}")
            if score >= 0.6:
                print("✅ PASS")
                sys.exit(0)
            else:
                print("❌ FAIL")
                sys.exit(1)

if __name__ == "__main__":
    main() 