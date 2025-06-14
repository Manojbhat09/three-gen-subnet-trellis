#!/usr/bin/env python3
"""Test pyspz compression with different PLY files and parameters"""

import pyspz
import trimesh

def test_pyspz_compression():
    """Test pyspz compression with different approaches"""
    
    print("Testing pyspz compression...")
    
    # Test 1: Create a simple PLY with trimesh
    print("\n1. Testing with trimesh-generated PLY:")
    try:
        mesh = trimesh.creation.box()
        ply_data = trimesh.exchange.ply.export_ply(mesh)
        print(f"   Created PLY: {len(ply_data)} bytes")
        
        # Try compression with different parameters
        try:
            compressed = pyspz.compress(ply_data, 1, 1)  # level, workers
            print(f"   ✓ Compressed with (1,1): {len(compressed)} bytes")
            ratio = len(ply_data) / len(compressed)
            print(f"   ✓ Compression ratio: {ratio:.2f}x")
        except Exception as e:
            print(f"   ❌ Compression failed with (1,1): {e}")
            
        try:
            compressed = pyspz.compress(ply_data, 0, 1)  # level, workers
            print(f"   ✓ Compressed with (0,1): {len(compressed)} bytes")
        except Exception as e:
            print(f"   ❌ Compression failed with (0,1): {e}")
            
        try:
            compressed = pyspz.compress(ply_data)  # default params
            print(f"   ✓ Compressed with defaults: {len(compressed)} bytes")
        except Exception as e:
            print(f"   ❌ Compression failed with defaults: {e}")
            
    except Exception as e:
        print(f"   ❌ PLY creation failed: {e}")
    
    # Test 2: Test with existing PLY file
    print("\n2. Testing with existing PLY file:")
    try:
        ply_path = './validation/tests/resources/hamburger.ply'
        with open(ply_path, 'rb') as f:
            ply_data = f.read()
        print(f"   Loaded PLY: {len(ply_data)} bytes")
        
        try:
            compressed = pyspz.compress(ply_data, 1, 1)
            print(f"   ✓ Compressed existing PLY: {len(compressed)} bytes")
            ratio = len(ply_data) / len(compressed)
            print(f"   ✓ Compression ratio: {ratio:.2f}x")
        except Exception as e:
            print(f"   ❌ Compression failed: {e}")
            
    except Exception as e:
        print(f"   ❌ Could not load existing PLY: {e}")
    
    # Test 3: Check pyspz version and documentation
    print("\n3. Checking pyspz info:")
    try:
        print(f"   pyspz version: {pyspz.__version__ if hasattr(pyspz, '__version__') else 'unknown'}")
        print(f"   Available functions: {[f for f in dir(pyspz) if not f.startswith('_')]}")
    except Exception as e:
        print(f"   ❌ Error getting pyspz info: {e}")

if __name__ == "__main__":
    test_pyspz_compression() 