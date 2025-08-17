#!/usr/bin/env python3
"""
Multi-GPU TRELLIS Server Wrapper
==============================

Wrapper script to run TRELLIS server with proper multi-GPU configuration.
"""

import os
import sys
import torch
import argparse
from pathlib import Path

# Set GPU memory allocation strategy
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

def setup_multigpu_environment(num_gpus: int = None):
    """Setup environment for multi-GPU operation"""
    
    if num_gpus is None:
        num_gpus = torch.cuda.device_count()
    
    print(f"🔧 Setting up multi-GPU environment for {num_gpus} GPUs")
    
    # Validate GPU availability
    available_gpus = torch.cuda.device_count()
    if available_gpus == 0:
        print("❌ No CUDA GPUs available")
        return False
    
    if num_gpus > available_gpus:
        print(f"⚠️  Requested {num_gpus} GPUs, but only {available_gpus} available")
        num_gpus = available_gpus
    
    # Set up GPU memory
    for gpu_id in range(num_gpus):
        with torch.cuda.device(gpu_id):
            # Pre-allocate some memory to ensure GPU is accessible
            _ = torch.zeros(1, device=f'cuda:{gpu_id}')
            torch.cuda.empty_cache()
    
    print(f"✅ Multi-GPU environment ready with {num_gpus} GPUs")
    return True

def main():
    parser = argparse.ArgumentParser(description='Multi-GPU TRELLIS Server')
    parser.add_argument('--gpu-id', type=int, default=0, 
                       help='GPU ID for this server instance')
    parser.add_argument('--port', type=int, default=8096,
                       help='Port number for this server instance')
    parser.add_argument('--host', type=str, default='127.0.0.1',
                       help='Host address')
    parser.add_argument('--num-gpus', type=int, default=None,
                       help='Total number of GPUs to use')
    
    args = parser.parse_args()
    
    # Setup multi-GPU environment
    if not setup_multigpu_environment(args.num_gpus):
        sys.exit(1)
    
    # Import and modify the original server
    sys.path.insert(0, str(Path(__file__).parent))
    
    # Set GPU device before importing to ensure proper device selection
    if torch.cuda.is_available():
        torch.cuda.set_device(args.gpu_id)
        print(f"🎯 Set default CUDA device to GPU {args.gpu_id}")
    
    # Import the original server module
    from trellis_subnit_server_mix_lora_flash import TrellisGenerator, app
    
    # Create generator with specific GPU
    print(f"🚀 Starting TRELLIS server on GPU {args.gpu_id}, port {args.port}")
    
    # The GPU selection is handled by CUDA_VISIBLE_DEVICES and torch.cuda.set_device()
    # No need to patch the constructor since it doesn't accept gpu_id parameter
    
    # Start the server
    import uvicorn
    uvicorn.run(app, host=args.host, port=args.port)

if __name__ == "__main__":
    main()
