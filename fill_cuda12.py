#!/usr/bin/env python3
import torch
import time
import sys
import os

def aggressive_gpu_fill(device_id=1, duration_seconds=30):
    """Fill GPU memory aggressively and hold it for specified duration"""
    try:
        device = torch.device(f'cuda:{device_id}')
        print(f"Targeting GPU {device_id}: {torch.cuda.get_device_name(device)}")
        
        # Get GPU info
        total_memory = torch.cuda.get_device_properties(device).total_memory
        print(f"Total GPU memory: {total_memory / 1024**3:.2f} GB")
        
        # Fill memory aggressively
        tensors = []
        chunk_size = 512 * 1024**2  # 512MB chunks
        
        print("Filling GPU memory aggressively...")
        while True:
            try:
                tensor = torch.randn(chunk_size // 4, device=device)
                tensors.append(tensor)
                
                current_memory = torch.cuda.memory_allocated(device) / 1024**3
                print(f"Allocated: {current_memory:.2f} GB", end='\r')
                
            except torch.cuda.OutOfMemoryError:
                print(f"\nGPU memory full at {torch.cuda.memory_allocated(device) / 1024**3:.2f} GB")
                break
        
        # Hold memory for specified duration
        print(f"Holding memory for {duration_seconds} seconds...")
        time.sleep(duration_seconds)
        
        # Release memory
        print("Releasing memory...")
        del tensors
        torch.cuda.empty_cache()
        
        print("Memory released successfully!")
        
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    duration = int(sys.argv[1]) if len(sys.argv) > 1 else 30
    aggressive_gpu_fill(device_id=1, duration_seconds=duration)
