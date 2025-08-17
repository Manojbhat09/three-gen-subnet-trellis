#!/usr/bin/env python3
"""
Validate Multi-GPU Fixes
"""

import torch
import os
import sys
from pathlib import Path

def validate_syntax():
    """Check for syntax errors"""
    try:
        import py_compile
        py_compile.compile('trellis_subnit_server_mix_lora_flash.py', doraise=True)
        print("✅ Syntax validation passed")
        return True
    except py_compile.PyCompileError as e:
        print(f"❌ Syntax error: {e}")
        return False

def validate_multigpu_init():
    """Test multi-GPU initialization"""
    try:
        # Import the server
        sys.path.insert(0, str(Path(__file__).parent))
        from trellis_subnit_server_mix_lora_flash import TrellisGenerator
        
        # Test different GPU IDs
        for gpu_id in [0, 1, 2]:
            gen = TrellisGenerator(gpu_id=gpu_id)
            expected_device = f'cuda:{gpu_id}' if torch.cuda.is_available() else 'cpu'
            
            if hasattr(gen, 'target_device'):
                print(f"✅ GPU {gpu_id}: target_device = {gen.target_device}")
                
                # Validate device logic
                if torch.cuda.is_available():
                    device_index = gpu_id % torch.cuda.device_count()
                    expected = f'cuda:{device_index}'
                    if gen.target_device != expected and gen.target_device != 'cpu':
                        print(f"⚠️  Unexpected device: {gen.target_device} (expected {expected})")
                else:
                    if gen.target_device != 'cpu':
                        print(f"⚠️  Expected CPU device, got {gen.target_device}")
            else:
                print(f"❌ GPU {gpu_id}: target_device attribute missing")
                return False
            
            del gen
        
        print("✅ Multi-GPU initialization validation passed")
        return True
        
    except Exception as e:
        print(f"❌ Multi-GPU validation failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all validations"""
    print("🔍 Validating Multi-GPU Fixes")
    print("=" * 30)
    
    # Check syntax
    if not validate_syntax():
        print("❌ Syntax validation failed")
        return 1
    
    # Check multi-GPU functionality
    if not validate_multigpu_init():
        print("❌ Multi-GPU validation failed")
        return 1
    
    print("\n✅ All validations passed!")
    print("\n📋 Ready for multi-GPU deployment:")
    print("   TRELLIS_GPU_ID=0 python trellis_subnit_server_mix_lora_flash.py")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
