#!/usr/bin/env python3
import torch
import time
import sys

def fill_gpu_memory(device_id=1, target_memory_gb=40):
    """Fill GPU memory and then release it"""
    try:
        # Set device
        device = torch.device(f'cuda:{device_id}')
        print(f"Targeting GPU {device_id}: {torch.cuda.get_device_name(device)}")
        
        # Get initial memory info
        initial_memory = torch.cuda.memory_allocated(device) / 1024**3
        total_memory = torch.cuda.get_device_properties(device).total_memory / 1024**3
        print(f"Initial memory: {initial_memory:.2f} GB / {total_memory:.2f} GB")
        
        # Fill memory gradually
        tensors = []
        chunk_size = 1024**3  # 1GB chunks
        target_bytes = int(target_memory_gb * 1024**3)
        
        print(f"Filling GPU memory to {target_memory_gb} GB...")
        current_memory = 0
        
        while current_memory < target_bytes:
            try:
                # Create tensor on GPU
                tensor = torch.randn(chunk_size // 4, device=device)  # 4 bytes per float32
                tensors.append(tensor)
                current_memory += chunk_size
                
                # Show progress
                current_gb = current_memory / 1024**3
                print(f"Allocated: {current_gb:.2f} GB")
                
                # Small delay to see progress
                time.sleep(0.1)
                
            except torch.cuda.OutOfMemoryError:
                print("GPU memory full!")
                break
        
        # Show final memory usage
        final_memory = torch.cuda.memory_allocated(device) / 1024**3
        print(f"Final memory usage: {final_memory:.2f} GB")
        
        # Hold memory for a few seconds
        print("Holding memory for 5 seconds...")
        time.sleep(5)
        
        # Release all memory
        print("Releasing memory...")
        del tensors
        torch.cuda.empty_cache()
        
        # Verify memory is released
        final_memory = torch.cuda.memory_allocated(device) / 1024**3
        print(f"Memory after release: {final_memory:.2f} GB")
        
        print("Script completed successfully!")
        
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    # Get target memory from command line or use default
    target_gb = float(sys.argv[1]) if len(sys.argv) > 1 else 40
    fill_gpu_memory(device_id=1, target_memory_gb=target_gb)
