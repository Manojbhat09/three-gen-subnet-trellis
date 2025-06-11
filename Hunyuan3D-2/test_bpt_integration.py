#!/usr/bin/env python3
"""Test script to validate BPT integration"""

import os
import torch
import numpy as np
import trimesh
from pathlib import Path

# Test BPT imports
try:
    from hy3dgen.shapegen.bpt.model.model import MeshTransformer
    from hy3dgen.shapegen.bpt.utils import Dataset, apply_normalize, sample_pc, joint_filter
    from hy3dgen.shapegen.bpt.model.serializaiton import BPT_deserialize, BPT_serialize
    print("✓ BPT imports successful")
except ImportError as e:
    print(f"✗ BPT import failed: {e}")
    exit(1)

def create_bpt_config():
    """Create BPT model configuration"""
    config = {
        'dim': 1024,
        'max_seq_len': 8192,
        'flash_attn': True,
        'attn_depth': 24,
        'attn_dim_head': 64,
        'attn_heads': 16,
        'attn_kwargs': {
            'ff_glu': True,
            'num_mem_kv': 4,
            'attn_qk_norm': True,
        },
        'dropout': 0.0,
        'pad_id': -1,
        'coor_continuous_range': (-1., 1.),
        'num_discrete_coors': 128,  # 2^7
        'block_size': 8,
        'offset_size': 16,
        'mode': 'vertices',
        'special_token': -2,
        'use_special_block': True,
        'conditioned_on_pc': True,
        'encoder_name': 'miche-256-feature',
        'encoder_freeze': False,
        'cond_dim': 768
    }
    return config

def test_bpt_model_loading():
    """Test BPT model initialization"""
    print("Testing BPT model loading...")
    
    try:
        config = create_bpt_config()
        model = MeshTransformer(**config)
        print(f"✓ BPT model initialized with {sum(p.numel() for p in model.parameters())} parameters")
        
        # Test moving to device
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model.to(device)
        print(f"✓ Model moved to {device}")
        
        return model
    except Exception as e:
        print(f"✗ BPT model loading failed: {e}")
        return None

def create_test_mesh():
    """Create a simple test mesh"""
    # Create a simple cube mesh for testing
    vertices = np.array([
        [-1, -1, -1], [1, -1, -1], [1, 1, -1], [-1, 1, -1],  # bottom face
        [-1, -1, 1], [1, -1, 1], [1, 1, 1], [-1, 1, 1]   # top face
    ]) * 0.5
    
    faces = np.array([
        [0, 1, 2], [0, 2, 3],  # bottom
        [4, 7, 6], [4, 6, 5],  # top
        [0, 4, 5], [0, 5, 1],  # front
        [2, 6, 7], [2, 7, 3],  # back
        [0, 3, 7], [0, 7, 4],  # left
        [1, 5, 6], [1, 6, 2]   # right
    ])
    
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
    return mesh

def test_bpt_serialization():
    """Test BPT serialization and deserialization"""
    print("Testing BPT serialization...")
    
    try:
        # Create test mesh
        mesh = create_test_mesh()
        mesh = apply_normalize(mesh)
        print(f"✓ Test mesh created with {len(mesh.vertices)} vertices, {len(mesh.faces)} faces")
        
        # Test serialization
        codes = BPT_serialize(mesh)
        print(f"✓ Mesh serialized to {len(codes)} codes")
        
        # Test deserialization
        coordinates = BPT_deserialize(
            codes,
            block_size=8,
            offset_size=16,
            compressed=True,
            special_token=-2,
            use_special_block=True
        )
        print(f"✓ Mesh deserialized to {len(coordinates)} coordinates")
        
        # Create mesh from coordinates
        if len(coordinates) > 0 and len(coordinates) % 3 == 0:
            vertices = coordinates.reshape(-1, 3)
            num_faces = len(vertices) // 3
            faces = np.arange(len(vertices)).reshape(num_faces, 3)
            reconstructed_mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
            print(f"✓ Reconstructed mesh created with {len(vertices)} vertices, {len(faces)} faces")
            return True
        else:
            print(f"✗ Invalid coordinates: {len(coordinates)} (should be multiple of 3)")
            return False
            
    except Exception as e:
        print(f"✗ BPT serialization test failed: {e}")
        return False

def test_dataset_creation():
    """Test BPT dataset creation"""
    print("Testing BPT dataset creation...")
    
    try:
        # Create and save test mesh
        mesh = create_test_mesh()
        mesh = apply_normalize(mesh)
        
        # Save as temporary file
        temp_path = "test_mesh.obj"
        mesh.export(temp_path)
        
        # Create dataset
        dataset = Dataset(input_type='mesh', input_list=[temp_path])
        print(f"✓ Dataset created with {len(dataset)} samples")
        
        # Test data loading
        sample = dataset[0]
        pc_normal = sample['pc_normal']
        print(f"✓ Point cloud shape: {pc_normal.shape}")
        
        # Cleanup
        os.remove(temp_path)
        
        return True
    except Exception as e:
        print(f"✗ Dataset creation test failed: {e}")
        if os.path.exists("test_mesh.obj"):
            os.remove("test_mesh.obj")
        return False

def main():
    """Run all tests"""
    print("=" * 50)
    print("BPT Integration Test")
    print("=" * 50)
    
    # Test 1: Model loading
    model = test_bpt_model_loading()
    if model is None:
        return False
    
    # Test 2: Serialization
    if not test_bpt_serialization():
        return False
    
    # Test 3: Dataset creation
    if not test_dataset_creation():
        return False
    
    print("=" * 50)
    print("✓ All BPT integration tests passed!")
    print("BPT is ready for use in the full pipeline.")
    print("=" * 50)
    return True

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1) 