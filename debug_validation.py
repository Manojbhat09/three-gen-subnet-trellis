#!/usr/bin/env python3

import sys
import os
sys.path.insert(0, 'validation')

def test_imports():
    print("Testing imports...")
    try:
        import requests
        print("✓ requests imported")
        
        import pyspz
        print("✓ pyspz imported")
        
        import torch
        print("✓ torch imported")
        
        from validation.engine.validation_engine import ValidationEngine
        print("✓ ValidationEngine imported")
        
        from validation.engine.io.ply import PlyLoader
        print("✓ PlyLoader imported")
        
        from validation.engine.rendering.renderer import Renderer
        print("✓ Renderer imported")
        
        return True
    except Exception as e:
        print(f"✗ Import failed: {e}")
        return False

def test_server_connection():
    print("\nTesting server connection...")
    try:
        import requests
        response = requests.post(
            "http://127.0.0.1:8096/generate/",
            data={"prompt": "test", "return_compressed": True},
            timeout=5
        )
        print(f"✓ Server responded with status: {response.status_code}")
        print(f"✓ Response size: {len(response.content)} bytes")
        return response.content
    except Exception as e:
        print(f"✗ Server connection failed: {e}")
        return None

def test_decompression(data):
    print("\nTesting decompression...")
    try:
        import pyspz
        import io
        from validation.engine.io.ply import PlyLoader
        
        decompressed = pyspz.decompress(data)
        print(f"✓ Decompressed size: {len(decompressed)} bytes")
        
        ply_loader = PlyLoader()
        gs_data = ply_loader.from_buffer(io.BytesIO(decompressed))
        print("✓ PLY data loaded successfully")
        return gs_data
    except Exception as e:
        print(f"✗ Decompression failed: {e}")
        return None

def main():
    print("=== Validation Debug Script ===\n")
    
    # Test 1: Imports
    if not test_imports():
        return
    
    # Test 2: Server connection
    data = test_server_connection()
    if data is None:
        return
    
    # Test 3: Decompression
    gs_data = test_decompression(data)
    if gs_data is None:
        return
    
    print("\n✓ All basic tests passed!")
    print("The validation pipeline should work. Try running the full script now.")

if __name__ == "__main__":
    main() 