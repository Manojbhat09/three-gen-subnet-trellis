#!/usr/bin/env python3
"""
Multi-GPU Configuration Helper for TRELLIS Server
================================================

Configuration utilities for proper multi-GPU setup.
"""

import torch
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

@dataclass
class MultiGPUConfig:
    """Multi-GPU configuration settings"""
    num_gpus: int = 1
    device_ids: List[int] = None
    memory_fraction: float = 0.9
    enable_peer_access: bool = True
    load_balancing: str = "round_robin"  # "round_robin", "memory_based", "custom"
    
    def __post_init__(self):
        if self.device_ids is None:
            self.device_ids = list(range(self.num_gpus))

class MultiGPUManager:
    """Manager for multi-GPU operations"""
    
    def __init__(self, config: MultiGPUConfig):
        self.config = config
        self.device_count = torch.cuda.device_count()
        self.current_device_idx = 0
        
        if config.num_gpus > self.device_count:
            print(f"⚠️  Requested {config.num_gpus} GPUs, but only {self.device_count} available")
            self.config.num_gpus = self.device_count
    
    def get_device(self, gpu_id: Optional[int] = None) -> torch.device:
        """Get appropriate device for model loading"""
        if gpu_id is not None:
            device_id = gpu_id % self.device_count
        else:
            # Load balancing
            device_id = self.current_device_idx % self.device_count
            self.current_device_idx += 1
        
        return torch.device(f'cuda:{device_id}')
    
    def load_model_to_device(self, model: torch.nn.Module, gpu_id: Optional[int] = None) -> torch.nn.Module:
        """Load model to appropriate device with proper error handling"""
        device = self.get_device(gpu_id)
        
        try:
            # Check available memory
            memory_free = torch.cuda.get_device_properties(device.index).total_memory
            memory_allocated = torch.cuda.memory_allocated(device.index)
            memory_available = (memory_free - memory_allocated) / 1024**3
            
            print(f"Loading model to {device}, available memory: {memory_available:.1f}GB")
            
            model = model.to(device)
            return model
            
        except RuntimeError as e:
            if "out of memory" in str(e):
                print(f"❌ Out of memory on {device}, trying CPU fallback")
                return model.to('cpu')
            else:
                raise e
    
    def clear_device_memory(self, device_id: Optional[int] = None):
        """Clear memory on specific device or all devices"""
        if device_id is not None:
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
        else:
            for i in range(self.device_count):
                with torch.cuda.device(i):
                    torch.cuda.empty_cache()
                    torch.cuda.reset_peak_memory_stats()
    
    def get_memory_stats(self) -> Dict[int, Dict[str, float]]:
        """Get memory statistics for all GPUs"""
        stats = {}
        for i in range(self.device_count):
            allocated = torch.cuda.memory_allocated(i) / 1024**3
            reserved = torch.cuda.memory_reserved(i) / 1024**3
            total = torch.cuda.get_device_properties(i).total_memory / 1024**3
            
            stats[i] = {
                'allocated_gb': allocated,
                'reserved_gb': reserved,
                'total_gb': total,
                'free_gb': total - allocated,
                'utilization': allocated / total * 100
            }
        
        return stats

# Example usage for TRELLIS server integration:
def create_multigpu_trellis_config(num_gpus: int = None) -> MultiGPUConfig:
    """Create optimized config for TRELLIS multi-GPU setup"""
    if num_gpus is None:
        num_gpus = torch.cuda.device_count()
    
    return MultiGPUConfig(
        num_gpus=num_gpus,
        memory_fraction=0.85,  # Leave some memory for other processes
        enable_peer_access=True,
        load_balancing="memory_based"
    )

# Global manager instance
gpu_manager = None

def init_multigpu_manager(config: MultiGPUConfig = None) -> MultiGPUManager:
    """Initialize global multi-GPU manager"""
    global gpu_manager
    if config is None:
        config = create_multigpu_trellis_config()
    
    gpu_manager = MultiGPUManager(config)
    return gpu_manager

def get_gpu_manager() -> MultiGPUManager:
    """Get the global GPU manager"""
    global gpu_manager
    if gpu_manager is None:
        gpu_manager = init_multigpu_manager()
    return gpu_manager
