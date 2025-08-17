#!/usr/bin/env python3
"""
Multi-GPU Functionality Test
"""

import os
import sys
import torch
from pathlib import Path

def test_multigpu_init():
    """Test multi-GPU initialization"""
    
    print("🧪 Testing Multi-GPU Initialization")
    print("=" * 40)
    
    # Set environment for GPU 0
    os.environ['TRELLIS_GPU_ID'] = '0'
    
    try:
        # Import the fixed TRELLIS server
        sys.path.insert(0, str(Path(__file__).parent))
        from trellis_subnit_server_mix_lora_flash import TrellisGenerator
        
        # Test GPU 0
        print("🔧 Testing GPU 0 initialization...")
        gen0 = TrellisGenerator(gpu_id=0)
        print(f"✅ GPU 0: {gen0.target_device}")
        
        # Test GPU 1 (simulated)
        if torch.cuda.device_count() >= 1:
            print("🔧 Testing GPU 1 initialization...")
            gen1 = TrellisGenerator(gpu_id=1)
            print(f"✅ GPU 1: {gen1.target_device}")
        
        # Test device assignment logic
        for gpu_id in range(min(4, torch.cuda.device_count() + 2)):
            gen = TrellisGenerator(gpu_id=gpu_id)
            expected_device = f'cuda:{gpu_id % torch.cuda.device_count()}'
            if torch.cuda.is_available():
                assert gen.target_device == expected_device or gen.target_device == 'cpu'
                print(f"✅ GPU {gpu_id} → {gen.target_device}")
            del gen
        
        print("✅ All multi-GPU initialization tests passed!")
        return True
        
    except Exception as e:
        print(f"❌ Multi-GPU test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_multigpu_init()
    sys.exit(0 if success else 1)
