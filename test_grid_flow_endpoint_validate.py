#!/usr/bin/env python3
"""
Enhanced Test script for the new comprehensive grid flow endpoint:
/generate_3d_from_prompt_grid_flow/

This endpoint follows the exact flow from test_img2img_prompt.py:
1. Generate grid image with multiple views
2. Crop grid into individual images
3. Optionally upscale images using Real-ESRGAN
4. Optionally remove backgrounds
5. Generate 3D model using TRELLIS multi-image pipeline

NEW: Saves all intermediate outputs for inspection!
NEW: Includes prompt optimization using vLLM!
"""

import requests
import json
import time
import os
import sys
from pathlib import Path
import subprocess

# Add the orchestrator directory to the path to import optimization functions
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Don't import orchestrator at module level - import only when needed
OPTIMIZATION_AVAILABLE = None  # Will be set when first needed

def get_optimization_availability():
    """Check if optimization is available without importing the full orchestrator"""
    global OPTIMIZATION_AVAILABLE
    
    if OPTIMIZATION_AVAILABLE is not None:
        return OPTIMIZATION_AVAILABLE
    
    try:
        # Try to import just the specific functions we need
        import importlib.util
        
        # Load the orchestrator module dynamically
        spec = importlib.util.spec_from_file_location(
            "orchestrator", 
            "continuous_trellis_orchestrator_working_a6000.py"
        )
        if spec and spec.loader:
            orchestrator_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(orchestrator_module)
            
            # Try to get the classes we need
            TaskRecord = getattr(orchestrator_module, 'TaskRecord', None)
            ContinuousTrellisOrchestrator = getattr(orchestrator_module, 'ContinuousTrellisOrchestrator', None)
            
            if TaskRecord and ContinuousTrellisOrchestrator:
                OPTIMIZATION_AVAILABLE = True
                print("✅ Prompt optimization available")
            else:
                OPTIMIZATION_AVAILABLE = False
                print("⚠️ Required classes not found in orchestrator")
        else:
            OPTIMIZATION_AVAILABLE = False
            print("⚠️ Could not load orchestrator module")
            
    except ImportError as e:
        OPTIMIZATION_AVAILABLE = False
        print(f"⚠️ Prompt optimization not available: {e}")
        print("   Continuing without optimization capabilities")
    except Exception as e:
        OPTIMIZATION_AVAILABLE = False
        print(f"⚠️ Prompt optimization not available: {e}")
        print("   Continuing without optimization capabilities")
    
    return OPTIMIZATION_AVAILABLE

# Server configuration
SERVER_URL = "http://localhost:8096"
ENDPOINT = "/generate_3d_from_prompt_grid_flow/"

# Test configurations - ENHANCED with intermediate saving and compression options
test_configs = [
    # {
    #     "name": "Fastest old",
    #     "params": {
    #         "base_prompt": "silver circlet on head",
    #         "style": "cinema",
    #         "seed": 42,
    #         "num_inference_steps": 4,
    #         "guidance_scale": 3.5,
    #         "width": 256,
    #         "height": 256,
    #         "upscale": False,
    #         "remove_background": True,
    #         "return_compressed": True,  # Get uncompressed PLY
    #         "save_preview": False,        # Generate preview video
    #         "save_intermediate": False,   # Save all intermediate outputs
    #         "filter_low_quality": False,
    #         "timing": False,
    #         "use_short_prompt": True,
    #         "ss_guidance_strength": 3.0,
    #         "ss_sampling_steps": 15,
    #         "slat_guidance_strength": 5.0,
    #         "slat_sampling_steps": 15,
    #         "image_endpoint": "cinema"
    #     }
    # },
    # {
    #     "name": "Fastest new",
    #     "params": {
    #         "base_prompt": "silver circlet on head",
    #         "style": "cinema",
    #         "seed": 42,
    #         "num_inference_steps": 7,
    #         "guidance_scale": 3.5,
    #         "width": 512,
    #         "height": 512,
    #         "upscale": False,
    #         "remove_background": True,
    #         "return_compressed": True,  # Get uncompressed PLY
    #         "save_preview": False,        # Generate preview video
    #         "save_intermediate": False,   # Save all intermediate outputs
    #         "filter_low_quality": False,
    #         "timing": False,
    #         "use_short_prompt": True,
    #         "ss_guidance_strength": 4.0,
    #         "ss_sampling_steps": 21,
    #         "slat_guidance_strength": 7.5,
    #         "slat_sampling_steps": 24,
    #         "image_endpoint": "cinema"
    #     }
    # },
    {
        "name": "validation maximum",
        "params": {
            "base_prompt": "silver circlet on head",
            "style": "3d",                      # Best validation performance
            "seed": 42,
            "num_inference_steps": 16,          # High quality generation
            "guidance_scale": 5.0,              # Strong guidance
            "width": 1024,                      # Maximum resolution
            "height": 1024,                     # Maximum resolution
            "upscale": False,                   # Never upscale (proven harmful)
            "remove_background": True,          # Essential for validation
            "return_compressed": False,         # Uncompressed for quality
            "save_preview": True,               # Enable preview
            "save_intermediate": True,          # Save all intermediates
            "filter_low_quality": True,         # Strict quality filtering
            "timing": True,                     # Enable timing
            "use_short_prompt": False,          # Full prompts for quality
            "ss_guidance_strength": 8.0,        # Maximum TRELLIS guidance
            "ss_sampling_steps": 30,            # Maximum TRELLIS steps
            "slat_guidance_strength": 5.0,     # Maximum TRELLIS guidance
            "slat_sampling_steps": 30,          # Maximum TRELLIS steps
            "image_endpoint": "3d",              # 3D endpoint for reconstruction
        }
    },
    {
        "name": "SMallest",
        "params": {
            "base_prompt": "silver circlet on head",
            "style": "cinema",
            "seed": 42,
            "num_inference_steps": 7,
            "guidance_scale": 3.5,
            "width": 256,
            "height": 256,
            "upscale": False,
            "remove_background": True,
            "return_compressed": True,  # Get uncompressed PLY
            "save_preview": True,        # Generate preview video
            "save_intermediate": True,   # Save all intermediate outputs
            "filter_low_quality": True,
            "timing": True,
            "use_short_prompt": True,
            "ss_guidance_strength": 7.5,
            "ss_sampling_steps": 21,
            "slat_guidance_strength": 4.0,
            "slat_sampling_steps": 24,
            "image_endpoint": "standard"
        }
    },
    {
        "name": "SMallest Cinema",
        "params": {
            "base_prompt": "silver circlet on head",
            "style": "cinema",
            "seed": 42,
            "num_inference_steps": 7,
            "guidance_scale": 3.5,
            "width": 256,
            "height": 256,
            "upscale": False,
            "remove_background": True,
            "return_compressed": True,  # Get uncompressed PLY
            "save_preview": True,        # Generate preview video
            "save_intermediate": True,   # Save all intermediate outputs
            "filter_low_quality": True,
            "timing": True,
            "use_short_prompt": True,
            "ss_guidance_strength": 7.5,
            "ss_sampling_steps": 21,
            "slat_guidance_strength": 4.0,
            "slat_sampling_steps": 24,
            "image_endpoint": "cinema"
        }
    },
    {
        "name": "SMallest Upscaled",
        "params": {
            "base_prompt": "silver circlet on head",
            "style": "cinema",
            "seed": 44,
            "num_inference_steps": 7,
            "guidance_scale": 3.5,
            "width": 256,
            "height": 256,
            "upscale": True,
            "remove_background": True,
            "return_compressed": True,  # Get uncompressed PLY
            "save_preview": True,        # Generate preview video
            "save_intermediate": True,   # Save all intermediate outputs
            "filter_low_quality": True,
            "timing": True,
            "use_short_prompt": True,
            "ss_guidance_strength": 7.5,
            "ss_sampling_steps": 21,
            "slat_guidance_strength": 4.0,
            "slat_sampling_steps": 24,
            "image_endpoint": "standard"
        }
    },
    {
        "name": "SMallest Upscaled Cinema",
        "params": {
            "base_prompt": "silver circlet on head",
            "style": "cinema",
            "seed": 44,
            "num_inference_steps": 7,
            "guidance_scale": 3.5,
            "width": 256,
            "height": 256,
            "upscale": True,
            "remove_background": True,
            "return_compressed": True,  # Get uncompressed PLY
            "save_preview": True,        # Generate preview video
            "save_intermediate": True,   # Save all intermediate outputs
            "filter_low_quality": True,
            "timing": True,
            "use_short_prompt": True,
            "ss_guidance_strength": 7.5,
            "ss_sampling_steps": 21,
            "slat_guidance_strength": 4.0,
            "slat_sampling_steps": 24,
            "image_endpoint": "cinema"
        }
    },
    {
        "name": "SMallest Upscaled long",
        "params": {
            "base_prompt": "silver circlet on head",
            "style": "cinema",
            "seed": 44,
            "num_inference_steps": 7,
            "guidance_scale": 3.5,
            "width": 256,
            "height": 256,
            "upscale": True,
            "remove_background": True,
            "return_compressed": True,  # Get uncompressed PLY
            "save_preview": True,        # Generate preview video
            "save_intermediate": True,   # Save all intermediate outputs
            "filter_low_quality": True,
            "timing": True,
            "use_short_prompt": False,
            "ss_guidance_strength": 7.5,
            "ss_sampling_steps": 21,
            "slat_guidance_strength": 4.0,
            "slat_sampling_steps": 24,
            "image_endpoint": "standard"
        }
    },
    {
        "name": "SMallest Upscaled long Cinema",
        "params": {
            "base_prompt": "silver circlet on head",
            "style": "cinema",
            "seed": 44,
            "num_inference_steps": 7,
            "guidance_scale": 3.5,
            "width": 256,
            "height": 256,
            "upscale": True,
            "remove_background": True,
            "return_compressed": True,  # Get uncompressed PLY
            "save_preview": True,        # Generate preview video
            "save_intermediate": True,   # Save all intermediate outputs
            "filter_low_quality": True,
            "timing": True,
            "use_short_prompt": False,
            "ss_guidance_strength": 7.5,
            "ss_sampling_steps": 21,
            "slat_guidance_strength": 4.0,
            "slat_sampling_steps": 24,
            "image_endpoint": "cinema"
        }
    },
    {
        "name": "GOOD",
        "params": {
            "base_prompt": "silver circlet on head",
            "style": "cinema",
            "seed": 45,
            "num_inference_steps": 7,
            "guidance_scale": 3.5,
            "width": 512,
            "height": 512,
            "upscale": False,
            "remove_background": True,
            "return_compressed": True,  # Get uncompressed PLY
            "save_preview": True,        # Generate preview video
            "save_intermediate": True,   # Save all intermediate outputs
            "filter_low_quality": True,
            "timing": True,
            "use_short_prompt": False,
            "ss_guidance_strength": 7.5,
            "ss_sampling_steps": 21,
            "slat_guidance_strength": 4.0,
            "slat_sampling_steps": 24,
            "image_endpoint": "standard"
        }
    },
    {
        "name": "GOOD Cinema",
        "params": {
            "base_prompt": "silver circlet on head",
            "style": "cinema",
            "seed": 45,
            "num_inference_steps": 7,
            "guidance_scale": 3.5,
            "width": 512,
            "height": 512,
            "upscale": False,
            "remove_background": True,
            "return_compressed": True,  # Get uncompressed PLY
            "save_preview": True,        # Generate preview video
            "save_intermediate": True,   # Save all intermediate outputs
            "filter_low_quality": True,
            "timing": True,
            "use_short_prompt": False,
            "ss_guidance_strength": 7.5,
            "ss_sampling_steps": 21,
            "slat_guidance_strength": 4.0,
            "slat_sampling_steps": 24,
            "image_endpoint": "cinema"
        }
    },
    {
        "name": "GOOD short",
        "params": {
            "base_prompt": "silver circlet on head",
            "style": "cinema",
            "seed": 46,
            "num_inference_steps": 7,
            "guidance_scale": 3.5,
            "width": 512,
            "height": 512,
            "upscale": False,
            "remove_background": True,
            "return_compressed": True,  # Get uncompressed PLY
            "save_preview": True,        # Generate preview video
            "save_intermediate": True,   # Save all intermediate outputs
            "filter_low_quality": True,
            "timing": True,
            "use_short_prompt": True,
            "ss_guidance_strength": 7.5,
            "ss_sampling_steps": 21,
            "slat_guidance_strength": 4.0,
            "slat_sampling_steps": 24,
            "image_endpoint": "standard"
        }
    },
    {
        "name": "GOOD short Cinema",
        "params": {
            "base_prompt": "silver circlet on head",
            "style": "cinema",
            "seed": 46,
            "num_inference_steps": 7,
            "guidance_scale": 3.5,
            "width": 512,
            "height": 512,
            "upscale": False,
            "remove_background": True,
            "return_compressed": True,  # Get uncompressed PLY
            "save_preview": True,        # Generate preview video
            "save_intermediate": True,   # Save all intermediate outputs
            "filter_low_quality": True,
            "timing": True,
            "use_short_prompt": True,
            "ss_guidance_strength": 7.5,
            "ss_sampling_steps": 21,
            "slat_guidance_strength": 4.0,
            "slat_sampling_steps": 24,
            "image_endpoint": "cinema"
        }
    },
    {
        "name": "Optimal save",
        "params": {
            "base_prompt": "silver circlet on head",
            "style": "cinema",
            "seed": 42,
            "num_inference_steps": 7,
            "guidance_scale": 3.5,
            "width": 512,
            "height": 512,
            "upscale": False,
            "remove_background": True,
            "return_compressed": True,  # Get uncompressed PLY
            "save_preview": True,        # Generate preview video
            "save_intermediate": True,   # Save all intermediate outputs
            "filter_low_quality": True,
            "timing": True,
            "use_short_prompt": False,
            "ss_guidance_strength": 7.5,
            "ss_sampling_steps": 21,
            "slat_guidance_strength": 4.0,
            "slat_sampling_steps": 24,
            "image_endpoint": "standard"
        }
    },
    {
        "name": "Optimal save 1024",
        "params": {
            "base_prompt": "silver circlet on head",
            "style": "cinema",
            "seed": 42,
            "num_inference_steps": 7,
            "guidance_scale": 3.5,
            "width": 1024,
            "height": 1024,
            "upscale": False,
            "remove_background": True,
            "return_compressed": True,  # Get uncompressed PLY
            "save_preview": True,        # Generate preview video
            "save_intermediate": True,   # Save all intermediate outputs
            "filter_low_quality": True,
            "timing": True,
            "use_short_prompt": False,
            "ss_guidance_strength": 7.5,
            "ss_sampling_steps": 21,
            "slat_guidance_strength": 4.0,
            "slat_sampling_steps": 24,
            "image_endpoint": "standard"
        }
    },
    {
        "name": "Optimal save upscale",
        "params": {
            "base_prompt": "silver circlet on head",
            "style": "cinema",
            "seed": 42,
            "num_inference_steps": 7,
            "guidance_scale": 3.5,
            "width": 512,
            "height": 512,
            "upscale": True,
            "remove_background": True,
            "return_compressed": False,  # Get uncompressed PLY
            "save_preview": True,        # Generate preview video
            "save_intermediate": True,   # Save all intermediate outputs
            "filter_low_quality": True,
            "timing": True,
            "use_short_prompt": False,
            "ss_guidance_strength": 7.5,
            "ss_sampling_steps": 21,
            "slat_guidance_strength": 4.0,
            "slat_sampling_steps": 24,
            "image_endpoint": "standard"
        }
    },
    {
        "name": "Optimal save short prompt",
        "params": {
            "base_prompt": "silver circlet on head",
            "style": "cinema",
            "seed": 42,
            "num_inference_steps": 7,
            "guidance_scale": 3.5,
            "width": 512,
            "height": 512,
            "upscale": False,
            "remove_background": True,
            "return_compressed": True,  # Get uncompressed PLY
            "save_preview": True,        # Generate preview video
            "save_intermediate": True,   # Save all intermediate outputs
            "filter_low_quality": True,
            "timing": True,
            "use_short_prompt": True,
            "ss_guidance_strength": 7.5,
            "ss_sampling_steps": 21,
            "slat_guidance_strength": 4.0,
            "slat_sampling_steps": 24,
            "image_endpoint": "standard"
        }
    },
    {
        "name": "Basic Standard Style save",
        "params": {
            "base_prompt": "silver circlet on head",
            "style": "standard",
            "seed": 42,
            "num_inference_steps": 8,
            "guidance_scale": 3.5,
            "width": 1024,
            "height": 1024,
            "upscale": False,
            "remove_background": True,
            "return_compressed": True,  # Get uncompressed PLY
            "save_preview": True,        # Generate preview video
            "save_intermediate": True,   # Save all intermediate outputs
            "filter_low_quality": True,
            "timing": True,
            "use_short_prompt": True,
            "ss_guidance_strength": 7.5,
            "ss_sampling_steps": 21,
            "slat_guidance_strength": 4.0,
            "slat_sampling_steps": 24,
            "image_endpoint": "standard"
        }
    },
    {
        "name": "Cinema Style with Upscaling save",
        "params": {
            "base_prompt": "silver circlet on head",
            "style": "cinema",
            "seed": 123,
            "num_inference_steps": 12,
            "guidance_scale": 4.0,
            "width": 1024,
            "height": 1024,
            "upscale": True,
            "remove_background": True,
            "return_compressed": True,  # Get uncompressed PLY
            "save_preview": True,        # Generate preview video
            "save_intermediate": True,   # Save all intermediate outputs
            "filter_low_quality": True,
            "timing": True,
            "use_short_prompt": True,
            "ss_guidance_strength": 7.5,
            "ss_sampling_steps": 21,
            "slat_guidance_strength": 4.0,
            "slat_sampling_steps": 24,
            "image_endpoint": "standard"
        }
    },
    {
        "name": "3D Style High Quality save",
        "params": {
            "base_prompt": "animal-plant hybrid-like object",
            "style": "3d",
            "seed": 456,
            "num_inference_steps": 16,
            "guidance_scale": 5.0,
            "width": 1024,
            "height": 1024,
            "upscale": True,
            "remove_background": True,
            "ss_guidance_strength": 8.0,
            "ss_sampling_steps": 25,
            "slat_guidance_strength": 5.0,
            "slat_sampling_steps": 30,
            "return_compressed": True,  # Get uncompressed PLY
            "save_preview": True,        # Generate preview video
            "save_intermediate": True,   # Save all intermediate outputs
            "filter_low_quality": True,
            "timing": True,
            "use_short_prompt": True
        }
    },
    {
        "name": "Fast Generation (512x512) save",
        "params": {
            "base_prompt": "silver circlet on head",
            "style": "standard",
            "seed": 789,
            "num_inference_steps": 4,
            "guidance_scale": 3.0,
            "width": 512,
            "height": 512,
            "upscale": False,
            "remove_background": False,
            "return_compressed": True,  # Get uncompressed PLY
            "save_preview": True,        # Generate preview video
            "save_intermediate": True,   # Save all intermediate outputs
            "filter_low_quality": True,
            "timing": True,
            "use_short_prompt": True,
            "ss_guidance_strength": 7.5,
            "ss_sampling_steps": 21,
            "slat_guidance_strength": 4.0,
            "slat_sampling_steps": 24,
            "image_endpoint": "standard"
        }
    },
    # {
    #     "name": "Basic Standard Style",
    #     "params": {
    #         "base_prompt": "silver circlet on head",
    #         "style": "standard",
    #         "seed": 42,
    #         "num_inference_steps": 8,
    #         "guidance_scale": 3.5,
    #         "width": 1024,
    #         "height": 1024,
    #         "upscale": False,
    #         "remove_background": True,
    #         "return_compressed": True,  # Get uncompressed PLY
    #         "save_preview": True,        # Generate preview video
    #         "save_intermediate": True,   # Save all intermediate outputs
    #         "filter_low_quality": True,
    #         "timing": True,
    #         "ss_guidance_strength": 7.5,
    #         "ss_sampling_steps": 21,
    #         "slat_guidance_strength": 4.0,
    #         "slat_sampling_steps": 24,
    #         "image_endpoint": "standard"
    #     }
    # },
    # {
    #     "name": "Cinema Style with Upscaling",
    #     "params": {
    #         "base_prompt": "car",
    #         "style": "cinema",
    #         "seed": 123,
    #         "num_inference_steps": 12,
    #         "guidance_scale": 4.0,
    #         "width": 1024,
    #         "height": 1024,
    #         "upscale": True,
    #         "remove_background": True,
    #         "return_compressed": True,  # Get uncompressed PLY
    #         "save_preview": True,        # Generate preview video
    #         "save_intermediate": True,   # Save all intermediate outputs
    #         "filter_low_quality": True,
    #         "timing": True,
    #         "ss_guidance_strength": 7.5,
    #         "ss_sampling_steps": 21,
    #         "slat_guidance_strength": 4.0,
    #         "slat_sampling_steps": 24,
    #         "image_endpoint": "standard"
    #     }
    # },
    # {
    #     "name": "3D Style High Quality",
    #     "params": {
    #         "base_prompt": "spaceship",
    #         "style": "3d",
    #         "seed": 456,
    #         "num_inference_steps": 16,
    #         "guidance_scale": 5.0,
    #         "width": 1024,
    #         "height": 1024,
    #         "upscale": True,
    #         "remove_background": True,
    #         "return_compressed": True,  # Get uncompressed PLY
    #         "save_preview": True,        # Generate preview video
    #         "save_intermediate": True,   # Save all intermediate outputs
    #         "filter_low_quality": True,
    #         "ss_guidance_strength": 8.0,
    #         "ss_sampling_steps": 25,
    #         "slat_guidance_strength": 5.0,
    #         "slat_sampling_steps": 30,
    #         "timing": True,
    #         "image_endpoint": "standard"
    #     }
    # },
    # {
    #     "name": "Fast Generation (512x512)",
    #     "params": {
    #         "base_prompt": "cat",
    #         "style": "standard",
    #         "seed": 789,
    #         "num_inference_steps": 4,
    #         "guidance_scale": 3.0,
    #         "width": 512,
    #         "height": 512,
    #         "upscale": False,
    #         "remove_background": False,
    #         "return_compressed": True,  # Get uncompressed PLY
    #         "save_preview": True,        # Generate preview video
    #         "save_intermediate": True,   # Save all intermediate outputs
    #         "filter_low_quality": True,
    #         "timing": False,
    #         "use_short_prompt": True,
    #         "ss_guidance_strength": 7.5,
    #         "ss_sampling_steps": 21,
    #         "slat_guidance_strength": 4.0,
    #         "slat_sampling_steps": 24,
    #         "image_endpoint": "standard"
    #     }
    # }
]

def ensure_output_dir():
    """Ensure test_outputs directory exists"""
    Path("test_outputs").mkdir(exist_ok=True)
    return "test_outputs"

def optimize_prompt_with_vllm(prompt: str) -> str:
    """Optimize prompt using vLLM optimization from the orchestrator"""
    if not get_optimization_availability():
        print("⚠️ Prompt optimization not available - using fallback enhancement")
        return enhance_prompt_fallback(prompt)
    
    try:
        print(f"🔄 Initializing prompt optimization...")
        
        # Create a minimal task record for optimization
        import importlib.util
        
        # Load the orchestrator module dynamically
        spec = importlib.util.spec_from_file_location(
            "orchestrator", 
            "continuous_trellis_orchestrator_working_a6000.py"
        )
        if spec and spec.loader:
            orchestrator_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(orchestrator_module)
            
            # Try to get the classes we need
            TaskRecord = getattr(orchestrator_module, 'TaskRecord', None)
            ContinuousTrellisOrchestrator = getattr(orchestrator_module, 'ContinuousTrellisOrchestrator', None)
            
            if TaskRecord and ContinuousTrellisOrchestrator:
                task = TaskRecord(
                    task_id="test_optimization",
                    prompt=prompt,
                    prompt_hash="test_hash",
                    validator_uid=1,
                    validator_hotkey="test_key",
                    validator_stake=1.0,
                    validation_threshold=0.5,
                    pulled_at=time.time()
                )
                
                # Create orchestrator instance with minimal config for optimization
                config = {
                    'enable_prompt_optimization': True,
                    'use_vllm_optim': True,
                    'vllm_optimization_priority': 'system_chat',
                    'vllm_port': 11300,  # Default vLLM port
                    'vllm_host': 'localhost'
                }
                
                print(f"🔧 Creating orchestrator with optimization config...")
                orchestrator = ContinuousTrellisOrchestrator(config)
                
                # Test vLLM connection first
                print(f"🔌 Testing vLLM connection...")
                if hasattr(orchestrator, 'test_vllm_connection'):
                    vllm_available = orchestrator.test_vllm_connection()
                    if not vllm_available:
                        print("⚠️ vLLM server not available - using fallback enhancement")
                        return enhance_prompt_fallback(prompt)
                    print("✅ vLLM connection successful")
                
                # Run optimization
                print(f"🚀 Running prompt optimization...")
                optimization_result = orchestrator.optimize_prompt_for_generation(task)
                
                if optimization_result and 'optimized_prompt' in optimization_result:
                    optimized_prompt = optimization_result['optimized_prompt']
                    method = optimization_result.get('method', 'unknown')
                    print(f"🚀 Prompt optimization successful using {method}")
                    print(f"   Original: {prompt}")
                    print(f"   Optimized: {optimized_prompt}")
                    return optimized_prompt
                else:
                    print("⚠️ Prompt optimization failed - using fallback enhancement")
                    return enhance_prompt_fallback(prompt)
            else:
                print("⚠️ Required classes not found in orchestrator")
                return enhance_prompt_fallback(prompt)
        else:
            print("⚠️ Could not load orchestrator module")
            return enhance_prompt_fallback(prompt)
            
    except Exception as e:
        print(f"❌ Prompt optimization error: {e}")
        print(f"   Error type: {type(e).__name__}")
        print("   Using fallback enhancement")
        return enhance_prompt_fallback(prompt)

def enhance_prompt_fallback(prompt: str) -> str:
    """Fallback prompt enhancement when vLLM optimization is not available"""
    print(f"🔄 Using fallback prompt enhancement...")
    
    # Simple prompt enhancement rules
    enhanced = prompt.strip()
    
    # Add quality descriptors if not present
    quality_terms = ["high quality", "detailed", "professional", "realistic"]
    has_quality = any(term in enhanced.lower() for term in quality_terms)
    
    if not has_quality:
        enhanced = f"high quality, detailed {enhanced}"
    
    # Add 3D-specific terms if not present
    if "3d" not in enhanced.lower() and "three dimensional" not in enhanced.lower():
        enhanced = f"{enhanced}, 3D model"
    
    # Add lighting/rendering terms if not present
    lighting_terms = ["lighting", "shading", "texture", "material"]
    has_lighting = any(term in enhanced.lower() for term in lighting_terms)
    
    if not has_lighting:
        enhanced = f"{enhanced}, with realistic lighting and materials"
    
    print(f"   Original: {prompt}")
    print(f"   Enhanced: {enhanced}")
    return enhanced

def run_validation_with_config(config_name: str, config_params: dict, compress: bool = False, optimize_prompt: bool = False, optimized_prompt: str = None):
    """Run validation using subnet_accurate_validator_multigpu_ply_inline.py with config parameters"""
    print(f"\n🔍 Running Validation for Config: {config_name}")
    print("=" * 60)
    
    # Build validation command
    validation_script = "subnet_accurate_validator_multigpu_ply_inline.py"
    if not os.path.exists(validation_script):
        print(f"❌ Validation script not found: {validation_script}")
        return None
    
    # Extract parameters for validation
    base_prompt = config_params["base_prompt"]
    style = config_params["style"]
    seed = config_params["seed"]
    num_inference_steps = config_params["num_inference_steps"]
    guidance_scale = config_params["guidance_scale"]
    width = config_params["width"]
    height = config_params["height"]
    upscale = config_params["upscale"]
    remove_background = config_params["remove_background"]
    filter_low_quality = config_params["filter_low_quality"]
    use_short_prompt = config_params["use_short_prompt"]
    
    # Extract additional parameters that might be in config
    ss_guidance_strength = config_params.get("ss_guidance_strength", 7.5)
    ss_sampling_steps = config_params.get("ss_sampling_steps", 21)
    slat_guidance_strength = config_params.get("slat_guidance_strength", 4.0)
    slat_sampling_steps = config_params.get("slat_sampling_steps", 24)
    image_endpoint = config_params.get("image_endpoint", "standard")
    lora_model = config_params.get("lora_model", None)
    timing = config_params.get("timing", False)
    save_preview = config_params.get("save_preview", False)
    save_intermediate = config_params.get("save_intermediate", False)
    return_compressed = config_params.get("return_compressed", True)
    
    # Apply prompt optimization if requested
    original_prompt = base_prompt
    if optimize_prompt:
        print(f"🔄 Optimizing prompt: {base_prompt}")
        optimized_prompt = optimize_prompt_with_vllm(base_prompt)
        if optimized_prompt != base_prompt:
            base_prompt = optimized_prompt
            print(f"✅ Using optimized prompt for validation")
        else:
            print(f"ℹ️ No optimization applied, using original prompt")
    else:
        if optimized_prompt:
            base_prompt = optimized_prompt
            print(f"✅ Using optimized prompt for validation: {base_prompt}")
        else:
            print(f"ℹ️ No optimization applied, using original prompt")


    # Build command
    cmd = [
        "python", validation_script,
        f'"{original_prompt}"',  # Original prompt
        f'"{base_prompt}"',      # Optimized prompt (or same if no optimization)
        "--validate",  # Use full validation mode
        "--style", style,
        "--num_inference_steps", str(num_inference_steps),
        "--guidance_scale", str(guidance_scale),
        "--width", str(width),
        "--height", str(height),
        "--ss_guidance", str(ss_guidance_strength),
        "--ss_steps", str(ss_sampling_steps),
        "--slat_guidance", str(slat_guidance_strength),
        "--slat_steps", str(slat_sampling_steps),
        "--image_endpoint", image_endpoint,
        "--seed", str(seed),
        "--port", "8096"
    ]
    
    # Add optional flags
    if upscale:
        cmd.append("--upscale")
    if remove_background:
        cmd.append("--remove_background")
    if filter_low_quality:
        cmd.append("--filter_low_quality")
    if use_short_prompt:
        cmd.append("--use_short_prompt")
    
    # Add LoRA model if specified
    if lora_model:
        cmd.extend(["--lora_model", lora_model])
    
    # Add compression flag if requested
    if compress:
        print(f"🗜️ Validation with compression enabled")
        # Note: Compression is handled internally by the validator
        # The --return_compressed flag doesn't exist in the validator script
    
    print(f"🚀 Running validation command:")
    print(f"   {' '.join(cmd)}")
    
    try:
        # Run validation
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=True,
            timeout=600,  # 10 minute timeout
            cwd=os.getcwd()
        )
        
        print(f"✅ Validation completed successfully!")
        
        # Look for results file
        results_file = "full_validation_results_8096.json"
        if os.path.exists(results_file):
            with open(results_file, 'r') as f:
                validation_results = json.load(f)
            
            print(f"\n📊 VALIDATION RESULTS for {config_name}:")
            print(f"=" * 60)
            
            if 'validation_engine_score' in validation_results:
                print(f"🏆 Validation Engine Score: {validation_results['validation_engine_score']:.4f}")
                print(f"🤝 Alignment Score: {validation_results['alignment_score']:.4f}")
                print(f"💎 Quality Score: {validation_results['quality_score']:.4f}")
                print(f"🎭 Demo Fidelity Score: {validation_results['demo_fidelity_score']:.4f}")
                print(f"✅ Validation Passed: {validation_results['validation_passed']}")
                
                # Interpretation
                if validation_results['demo_fidelity_score'] == 0.0:
                    print("❌ SUBNET RESULT: ZERO TASK FIDELITY")
                elif validation_results['demo_fidelity_score'] == 0.75:
                    print("🟡 SUBNET RESULT: MEDIUM FIDELITY (0.75)")
                elif validation_results['demo_fidelity_score'] == 1.0:
                    print("🟢 SUBNET RESULT: PERFECT FIDELITY (1.0)")
                else:
                    print(f"🔵 SUBNET RESULT: PARTIAL FIDELITY ({validation_results['demo_fidelity_score']:.4f})")
            
            return validation_results
        else:
            print(f"❌ Results file not found: {results_file}")
            return None
            
    except subprocess.TimeoutExpired:
        print(f"❌ Validation timed out after 10 minutes")
        return None
    except subprocess.CalledProcessError as e:
        print(f"❌ Validation failed with exit code {e.returncode}")
        print(f"   stdout: {e.stdout}")
        print(f"   stderr: {e.stderr}")
        return None
    except Exception as e:
        print(f"❌ Validation error: {e}")
        return None

def test_grid_flow_endpoint():
    """Test the comprehensive grid flow endpoint with various configurations."""
    
    print("🎯 Testing Comprehensive Grid Flow Endpoint")
    print("=" * 60)
    print(f"Server: {SERVER_URL}")
    print(f"Endpoint: {ENDPOINT}")
    print("=" * 60)
    print("💾 All intermediate outputs will be saved!")
    print("=" * 60)
    
    for i, config in enumerate(test_configs, 1):
        print(f"\n🧪 Test {i}: {config['name']}")
        print("-" * 40)
        
        # Create unique output directory for this test
        test_name = config['name'].lower().replace(' ', '_').replace('(', '').replace(')', '')
        test_output_dir = f"test_outputs/{test_name}_{config['params']['seed']}"
        Path(test_output_dir).mkdir(parents=True, exist_ok=True)
        
        print(f"📁 Output directory: {test_output_dir}")
        
        # Prepare form data
        form_data = config['params'].copy()
        
        # Convert boolean values to strings for form data
        for key, value in form_data.items():
            if isinstance(value, bool):
                form_data[key] = str(value).lower()
        
        print(f"Parameters:")
        for key, value in form_data.items():
            print(f"  {key}: {value}")
        
        try:
            # Make request to endpoint
            print(f"\n🚀 Sending request...")
            start_time = time.time()
            
            response = requests.post(
                f"{SERVER_URL}{ENDPOINT}",
                data=form_data,
                timeout=1800  # 30 minutes timeout
            )
            
            request_time = time.time() - start_time
            print(f"   Request completed in {request_time:.2f}s")
            print(f"   Status code: {response.status_code}")
            
            if response.status_code == 200:
                # Success - check response headers
                print(f"   ✅ Success!")
                
                # Extract response data from headers
                response_data = {}
                try:
                    response_data = json.loads(response.headers.get('X-Response-Data', '{}'))
                except:
                    pass
                
                # Print response info
                if response_data:
                    print(f"   📊 Response Data:")
                    print(f"     - Status: {response_data.get('status', 'unknown')}")
                    print(f"     - Pipeline: {response_data.get('pipeline', 'unknown')}")
                    print(f"     - Generation time: {response_data.get('generation_time', 0):.2f}s")
                    print(f"     - PLY size: {response_data.get('ply_size_bytes', 0):,} bytes")
                    print(f"     - Steps completed: {', '.join(response_data.get('steps_completed', []))}")
                
                # Check compression info
                compression = response.headers.get('X-Compression', 'none')
                if compression == 'spz':
                    compression_ratio = response.headers.get('X-Compression-Ratio', '0%')
                    print(f"   🗜️ Compression: SPZ ({compression_ratio})")
                else:
                    print(f"   📁 Compression: None")
                
                # Save the PLY file to test-specific directory
                filename = response.headers.get('Content-Disposition', '').split('filename=')[-1].strip('"')
                if not filename:
                    filename = f"grid_flow_{config['params']['base_prompt']}_{config['params']['seed']}.ply"
                    if compression == 'spz':
                        filename += '.spz'
                
                output_path = Path(f"{test_output_dir}/{filename}")
                
                with open(output_path, 'wb') as f:
                    f.write(response.content)
                
                print(f"   💾 File saved: {output_path}")
                print(f"   📏 File size: {len(response.content):,} bytes ({len(response.content)/1024/1024:.1f} MB)")
                
                # Save metadata
                if response_data:
                    metadata_file = Path(f"{test_output_dir}/metadata.json")
                    with open(metadata_file, 'w') as f:
                        json.dump(response_data, f, indent=2)
                    print(f"   💾 Metadata saved: {metadata_file}")
                
                print(f"   📁 Check {test_output_dir} for all intermediate outputs!")
                
            else:
                # Error response
                print(f"   ❌ Error: {response.status_code}")
                try:
                    error_detail = response.json()
                    print(f"   Error details: {error_detail}")
                except:
                    print(f"   Error text: {response.text[:200]}...")
        
        except requests.exceptions.Timeout:
            print(f"   ⏰ Request timed out after 30 minutes")
        except requests.exceptions.ConnectionError:
            print(f"   🔌 Connection error - is the server running?")
        except Exception as e:
            print(f"   💥 Unexpected error: {e}")
        
        print(f"   {'='*40}")
    
    print(f"\n🎉 All tests completed!")
    print(f"📁 Check the 'test_outputs/' directory for generated files.")
    print(f"💾 Each test has its own subdirectory with all intermediate outputs!")

def test_single_config():
    """Test a single configuration with detailed output."""
    
    print("🎯 Testing Single Configuration")
    print("=" * 40)
    
    # Single test configuration - ENHANCED with intermediate saving
    params = {
        "base_prompt": "robot",
        "style": "standard",
        "seed": 42,
        "num_inference_steps": 8,
        "guidance_scale": 3.5,
        "width": 1024,
        "height": 1024,
        "upscale": True,
        "remove_background": True,
        "ss_guidance_strength": 7.5,
        "ss_sampling_steps": 21,
        "slat_guidance_strength": 4.0,
        "slat_sampling_steps": 24,
        "return_compressed": False,  # Get uncompressed PLY
        "save_preview": True,        # Generate preview video
        "save_intermediate": True,   # Save all intermediate outputs
        "filter_low_quality": True,
        "timing": True,
        "use_short_prompt": True
    }
    
    # Create unique output directory for this test
    test_output_dir = f"test_outputs/single_test_{params['seed']}"
    Path(test_output_dir).mkdir(parents=True, exist_ok=True)
    
    print(f"📁 Output directory: {test_output_dir}")
    print(f"Parameters:")
    for key, value in params.items():
        print(f"  {key}: {value}")
    
    # Convert boolean values to strings for form data
    form_data = {k: str(v).lower() if isinstance(v, bool) else v for k, v in params.items()}
    
    try:
        print(f"\n🚀 Sending request...")
        start_time = time.time()
        
        response = requests.post(
            f"{SERVER_URL}{ENDPOINT}",
            data=form_data,
            timeout=1800
        )
        
        request_time = time.time() - start_time
        print(f"Request completed in {request_time:.2f}s")
        print(f"Status code: {response.status_code}")
        
        if response.status_code == 200:
            print(f"✅ Success!")
            
            # Extract and display response data
            response_data = {}
            try:
                response_data = json.loads(response.headers.get('X-Response-Data', '{}'))
            except:
                pass
            
            if response_data:
                print(f"\n📊 Response Data:")
                for key, value in response_data.items():
                    print(f"  {key}: {value}")
            
            # Save file to test-specific directory
            filename = response.headers.get('Content-Disposition', '').split('filename=')[-1].strip('"')
            if not filename:
                filename = f"grid_flow_{params['base_prompt']}_{params['seed']}.ply"
                if params['return_compressed']:
                    filename += '.spz'
            
            output_path = Path(f"{test_output_dir}/{filename}")
            
            with open(output_path, 'wb') as f:
                f.write(response.content)
            
            print(f"\n💾 File saved: {output_path}")
            print(f"📏 File size: {len(response.content):,} bytes ({len(response.content)/1024/1024:.1f} MB)")
            
            # Save metadata
            if response_data:
                metadata_file = Path(f"{test_output_dir}/metadata.json")
                with open(metadata_file, 'w') as f:
                    json.dump(response_data, f, indent=2)
                print(f"💾 Metadata saved: {metadata_file}")
            
            print(f"📁 Check {test_output_dir} for all intermediate outputs!")
            
        else:
            print(f"❌ Error: {response.status_code}")
            try:
                error_detail = response.json()
                print(f"Error details: {error_detail}")
            except:
                print(f"Error text: {response.text[:200]}...")
    
    except Exception as e:
        print(f"💥 Error: {e}")

if __name__ == "__main__":
    # Parse command line arguments first (before any imports)
    import argparse
    parser = argparse.ArgumentParser(description="Enhanced FLUX + TRELLIS Grid Flow Endpoint Tester")
    parser.add_argument("--validate", action="store_true", help="Run validation instead of generation")
    parser.add_argument("--compress", action="store_true", help="Enable compression for validation")
    parser.add_argument("--config", type=str, help="Specific config name to test (e.g., 'SMallest', 'GOOD')")
    parser.add_argument("--base_prompt", type=str, help="Base prompt to use for generation")
    parser.add_argument("--optimize-prompt", action="store_true", help="Enable prompt optimization using vLLM (requires orchestrator)")
    parser.add_argument("--optimized-prompt", default=None, type=str, help="Optimized prompt to use for generation")
    # Check for help first (argparse automatically provides --help)
    if "--help" in sys.argv or "-h" in sys.argv:
        parser.print_help()
        exit(0)
    
    args = parser.parse_args()
    
    print("🚀 Enhanced FLUX + TRELLIS Grid Flow Endpoint Tester")
    print("=" * 60)
    print("💾 Saves ALL intermediate outputs for inspection!")
    print("📁 Creates organized test directories")
    print("🎬 Generates preview videos")
    print("🗜️ Provides uncompressed PLY files")
    print("🔍 Includes validation capabilities")
    print("🚀 Includes prompt optimization using vLLM")
    print("=" * 60)
    print("\n💡 Usage Examples:")
    print("  python test_grid_flow_endpoint_validate.py --validate --optimize-prompt")
    print("  python test_grid_flow_endpoint_validate.py --validate --config SMallest --optimize-prompt")
    print("  python test_grid_flow_endpoint_validate.py --validate --base_prompt 'silver circlet' --optimize-prompt")
    print("=" * 60)
    
    # Ensure output directory exists
    ensure_output_dir()
    
    # Check if server is running
    try:
        health_check = requests.get(f"{SERVER_URL}/", timeout=5)
        print(f"✅ Server is running at {SERVER_URL}")
    except:
        print(f"❌ Server not accessible at {SERVER_URL}")
        print(f"   Please ensure the server is running on port 8096")
        exit(1)
    
    if args.validate:
        print(f"\n🔍 VALIDATION MODE")
        if args.compress:
            print(f"🗜️ Compression enabled")
        else:
            print(f"📁 No compression (uncompressed PLY)")
        if args.base_prompt:
            print(f"🔍 Base prompt: {args.base_prompt}")
        if args.optimize_prompt:
            print("🔄 Prompt optimization enabled")
        if args.config:
            # Find specific config
            config_found = None
            for config in test_configs:
                if config["name"] == args.config:
                    config_found = config
                    break
            
            if config_found:
                print(f"🎯 Running validation for config: {args.config}")
                if args.base_prompt:
                    config_found["params"]["base_prompt"] = args.base_prompt
                run_validation_with_config(args.config, config_found["params"], 
                args.compress, args.optimize_prompt, optimized_prompt=args.optimized_prompt)
            else:
                print(f"❌ Config '{args.config}' not found")
                print(f"   Available configs: {[c['name'] for c in test_configs]}")
        else:
            if args.base_prompt:
                print(f"🔍 Base prompt: {args.base_prompt}")
            # Run validation for all configs
            print(f"🎯 Running validation for all configs")
            for config in test_configs:
                print(f"🎯 Running validation for config: {config['name']}")
                if args.base_prompt:
                    config["params"]["base_prompt"] = args.base_prompt
                run_validation_with_config(config["name"], config["params"], 
                    args.compress, args.optimize_prompt, optimized_prompt=args.optimized_prompt)
                print(f"\n{'='*60}")
    else:
        print("\nChoose test mode:")
        print("1. Run all test configurations (saves all outputs)")
        print("2. Run single detailed test (saves all outputs)")
        print("3. Run validation for all configs")
        print("4. Run validation for specific config")
        
        choice = input("\nEnter choice (1, 2, 3, or 4): ").strip()
        
        if choice == "1":
            test_grid_flow_endpoint()
        elif choice == "2":
            test_single_config()
        elif choice == "3":
            print(f"🎯 Running validation for all configs")
            for config in test_configs:
                if args.base_prompt:
                    config["params"]["base_prompt"] = args.base_prompt
                run_validation_with_config(config["name"], config["params"], 
                    False, args.optimize_prompt, optimized_prompt=args.optimized_prompt)
                print(f"\n{'='*60}")
        elif choice == "4":
            print(f"Available configs: {[c['name'] for c in test_configs]}")
            config_name = input("Enter config name: ").strip()
            config_found = None
            for config in test_configs:
                if config["name"] == config_name:
                    config_found = config
                    break
            
            if config_found:
                compress = input("Enable compression? (y/n): ").strip().lower() == 'y'
                optimize = input("Enable prompt optimization? (y/n): ").strip().lower() == 'y'
                if args.base_prompt:
                    config_found["params"]["base_prompt"] = args.base_prompt
                # Use command line flag if provided, otherwise use interactive choice
                final_optimize = args.optimize_prompt if args.optimize_prompt else optimize
                run_validation_with_config(config_name, config_found["params"], 
                    compress, final_optimize, optimized_prompt=args.optimized_prompt)
            else:
                print(f"❌ Config '{config_name}' not found")
        else:
            print("Invalid choice. Running all tests...")
            test_grid_flow_endpoint()
