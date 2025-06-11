#!/usr/bin/env python3
"""
Quick test to debug TRELLIS mesh format
"""

import os
import sys
import torch
from PIL import Image

# Add TRELLIS to path
sys.path.append('TRELLIS')

def test_trellis_mesh_format():
    """Test TRELLIS mesh output format."""
    
    os.environ['SPCONV_ALGO'] = 'native'
    
    from trellis.pipelines import TrellisImageTo3DPipeline
    
    print("Loading TRELLIS pipeline...")
    pipeline = TrellisImageTo3DPipeline.from_pretrained("microsoft/TRELLIS-image-large")
    pipeline.cuda()
    
    # Create a simple test image
    test_image = Image.new('RGB', (512, 512), color='red')
    print("Created test image")
    
    # Run TRELLIS
    print("Running TRELLIS...")
    outputs = pipeline.run(
        test_image,
        seed=42,
        sparse_structure_sampler_params={"steps": 4},  # Faster for testing
        slat_sampler_params={"steps": 4},
    )
    
    print(f"TRELLIS outputs keys: {outputs.keys()}")
    
    # Examine mesh output
    mesh_result = outputs['mesh'][0]
    print(f"Mesh result type: {type(mesh_result)}")
    print(f"Mesh result attributes: {dir(mesh_result)}")
    
    if hasattr(mesh_result, '__dict__'):
        print(f"Mesh result dict: {mesh_result.__dict__.keys()}")
    
    # Try different attribute access patterns
    attrs_to_check = ['vertices', 'faces', 'mesh_v', 'mesh_f', 'v', 'f']
    for attr in attrs_to_check:
        if hasattr(mesh_result, attr):
            value = getattr(mesh_result, attr)
            print(f"✓ {attr}: {type(value)} shape: {getattr(value, 'shape', 'no shape')}")
        else:
            print(f"✗ {attr}: not found")

if __name__ == "__main__":
    test_trellis_mesh_format() 