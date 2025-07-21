#!/usr/bin/env python3
"""
Simple debug script to test generation server response
"""
import requests
import sys
import base64

# Try to import pyspz
try:
    import pyspz
    SPZ_AVAILABLE = True
    print("✅ pyspz library available")
except ImportError:
    SPZ_AVAILABLE = False
    print("❌ pyspz library not available")

def detect_format(data):
    """Detect the format of the data"""
    if len(data) == 0:
        return "empty"
    
    # Check for SPZ magic bytes
    # SPZ format typically starts with specific bytes
    if data[:4] == b'(\xb5/\xfd':
        return "spz_zstd"
    elif data[:2] == b'\x1f\x8b':
        return "gzip"
    elif data[:4] == b'PK\x03\x04':
        return "zip"
    elif data.startswith(b'ply\n') or data.startswith(b'ply\r\n'):
        return "ply_text"
    elif b'ply' in data[:100].lower():
        return "ply_binary"
    elif data[:4] == b'\x89PNG':
        return "png"
    elif data[:2] == b'\xff\xd8':
        return "jpeg"
    else:
        # Check if it looks like base64
        try:
            base64.b64decode(data[:100])
            return "base64"
        except:
            pass
        
        # Check headers for more clues
        hex_start = data[:20].hex()
        print(f"🔍 First 20 bytes hex: {hex_start}")
        
        # Look for common patterns
        if b'format binary' in data[:200]:
            return "ply_binary_header"
        
        return "unknown_binary"

def test_spz_decompression(data):
    """Test SPZ decompression"""
    if not SPZ_AVAILABLE:
        print("❌ Cannot test SPZ decompression - pyspz not available")
        return None
    
    try:
        print("🔄 Attempting SPZ decompression...")
        decompressed = pyspz.decompress(data)
        print(f"✅ SPZ decompression successful!")
        print(f"📊 Decompressed size: {len(decompressed):,} bytes")
        print(f"📋 Compression ratio: {len(data) / len(decompressed) * 100:.1f}%")
        
        # Check what the decompressed data looks like
        if decompressed.startswith(b'ply\n') or decompressed.startswith(b'ply\r\n'):
            print("✅ Decompressed data is PLY format")
            # Show first few lines
            lines = decompressed.decode('utf-8', errors='ignore').split('\n')[:10]
            print("📋 First 10 lines of PLY:")
            for i, line in enumerate(lines):
                print(f"   {i+1}: {line}")
        else:
            print("❓ Decompressed data format unknown")
            print(f"📋 First 100 bytes: {decompressed[:100]}")
        
        return decompressed
        
    except Exception as e:
        print(f"❌ SPZ decompression failed: {e}")
        return None

def test_generation_server():
    url = "http://127.0.0.1:8096/generate/"
    prompt = "a simple red cube"
    
    try:
        print(f"🎨 Sending prompt: '{prompt}'")
        response = requests.post(url, data={'prompt': prompt}, timeout=60)
        response.raise_for_status()
        
        print(f"✅ Generation successful")
        print(f"📊 Response size: {len(response.content):,} bytes")
        print(f"📋 Status code: {response.status_code}")
        print(f"📋 Headers:")
        for key, value in response.headers.items():
            print(f"   {key}: {value}")
        
        # Detect format
        data_format = detect_format(response.content)
        print(f"📋 Detected format: {data_format}")
        
        # If it's SPZ, try to decompress
        if data_format == "spz_zstd":
            decompressed = test_spz_decompression(response.content)
            return response.content, decompressed
        else:
            print(f"❓ Unexpected format: {data_format}")
            return response.content, None
            
    except requests.exceptions.RequestException as e:
        print(f"❌ Generation failed: {e}")
        return None, None
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return None, None

if __name__ == "__main__":
    test_generation_server() 