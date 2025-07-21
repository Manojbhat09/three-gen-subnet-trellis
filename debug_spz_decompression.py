#!/usr/bin/env python3
"""
Debug SPZ Decompression
Purpose: Isolate pyspz library behavior to identify source of binary output
"""
import sys
import os
import contextlib
from io import StringIO

try:
    import pyspz
    print("✅ pyspz library loaded successfully")
except ImportError as e:
    print(f"❌ Failed to import pyspz: {e}")
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

def test_spz_decompression():
    """Test SPZ decompression on saved data"""
    
    # Check if compressed file exists
    if not os.path.exists("compressed_output.spz"):
        print("❌ compressed_output.spz not found")
        print("Run: python3 simple_local_validator.py \"a cat\" --save-compressed")
        sys.exit(1)
    
    # Load compressed data
    with open("compressed_output.spz", "rb") as f:
        compressed_data = f.read()
    
    print(f"📁 Loaded {len(compressed_data):,} bytes of compressed data")
    
    # Test decompression with detailed monitoring
    print("🔄 Starting pyspz.decompress() with stdout suppression...")
    print("=" * 50)
    
    try:
        # This is where we'll suppress the binary output from pyspz
        with suppress_stdout():
            decompressed_data = pyspz.decompress(compressed_data, False)  # False = don't include normals
        
        print("=" * 50)
        print(f"✅ Decompression completed successfully")
        print(f"📊 Original size: {len(compressed_data):,} bytes")
        print(f"📊 Decompressed size: {len(decompressed_data):,} bytes")
        print(f"📦 Compression ratio: {len(compressed_data)/len(decompressed_data)*100:.1f}%")
        
        # Validate PLY format
        if decompressed_data.startswith(b'ply\n') or decompressed_data.startswith(b'ply\r\n'):
            print("✅ Decompressed data is valid PLY format")
            
            # Show header safely
            try:
                header_text = decompressed_data.decode('utf-8', errors='ignore')
                header_lines = header_text.split('\n')[:5]
                print("📋 PLY Header:")
                for i, line in enumerate(header_lines, 1):
                    if line.strip():
                        print(f"   {i}: {line}")
            except Exception as e:
                print(f"⚠️ Header parsing failed: {e}")
        else:
            print("❌ Decompressed data is not valid PLY format")
            # Show first 100 bytes as hex for debugging
            hex_preview = decompressed_data[:100].hex()
            print(f"📋 First 100 bytes (hex): {hex_preview}")
            
        return True
        
    except Exception as e:
        print("=" * 50)
        print(f"❌ Decompression failed: {e}")
        return False

if __name__ == "__main__":
    print("🧪 Testing pyspz decompression in isolation with stdout suppression")
    print("=" * 60)
    
    success = test_spz_decompression()
    
    if success:
        print("\n✅ SPZ decompression test completed successfully")
    else:
        print("\n❌ SPZ decompression test failed")
        sys.exit(1) 