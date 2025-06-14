#!/usr/bin/env python3
"""
Memory-Optimized Generation Script
Purpose: Generate 3D models with aggressive memory management to avoid CUDA OOM
"""

import torch
import gc
import time
import requests
import json

def clear_gpu_memory():
    """Aggressively clear GPU memory"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        gc.collect()

def generate_with_memory_management(prompt: str):
    """Generate 3D model with memory management"""
    print(f"🎯 Generating: {prompt}")
    
    # Clear memory before generation
    clear_gpu_memory()
    
    try:
        # Make generation request
        response = requests.post(
            "http://127.0.0.1:8095/generate/",
            json={"prompt": prompt},
            timeout=300
        )
        
        if response.status_code == 200:
            result = response.json()
            print("✅ Generation successful!")
            return result
        else:
            print(f"❌ Generation failed: {response.status_code}")
            return None
            
    except Exception as e:
        print(f"❌ Generation error: {e}")
        return None
    
    finally:
        # Clear memory after generation
        clear_gpu_memory()

if __name__ == "__main__":
    # Test generation
    result = generate_with_memory_management("a red cube")
    if result:
        print("🎉 Memory-optimized generation successful!")
    else:
        print("❌ Generation failed")
