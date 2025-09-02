#!/usr/bin/env python3
"""
Continuous TRELLIS Orchestrator - Subnet 17 (404-GEN)
Purpose: Continuous mining with intelligent task deduplication and idle validation

Features:
- Continuous task harvesting with prompt deduplication
- Real-time feedback processing and score tracking
- Automatic validation during idle periods
- Comprehensive statistics and JSON logging
- Always-on generation server integration
- PRIORITY-BASED server coordination for time-critical tasks

python continuous_trellis_orchestrator_lora_working.py --activate-learning --only-log-learning 18
 --disable-task-tracking  --no-skip-duplicates --vllm-optim --system-prompt --vllm-priority system_chat --vllm-optim-port 11300 --vllm-url "http://localhost:11300" --lora "cinema"

ython continuous_trellis_orchestrator_lora_working.py --activate-learning --only-log-learning 18
 --disable-task-tracking  --no-skip-duplicates --vllm-optim --system-prompt --vllm-priority system_chat --vllm-optim-port 11300 --vllm-url "http://localhost:11300" --lora "cinema"

python continuous_trellis_orchestrator_lora_working.py --activate-learning --only-log-learning 18 --disable-task-tracking  --no-skip-duplicates --vllm-optim --system-prompt --vllm-priority system_chat --vllm-optim-port 11300 --vllm-url "http://localhost:11300" --lora "cinema"   --vllm --no-fallback

python continuous_trellis_orchestrator_lora_working.py --disable-task-tracking  --no-skip-duplicates  --system-prompt --vllm-priority system_chat  --no-fallback --lora "" 


python continuous_trellis_orchestrator_lora_working.py --activate-learning --only-log-learning 18 --disable-task-tracking  --no-skip-duplicates --vllm-optim --system-prompt --vllm-priority system_chat --vllm-optim-port 11300 --vllm-url "http://localhost:11300" --lora "cinema"   --vllm --no-fallback --production-mv-gen

multi endpoint, vllm true
l4:
python continuous_trellis_orchestrator_working_a6000_simulate.py --promptfile episodic_test_prompts.py --simulate --fastest-mv-gen --no-skip-duplicates --disable-task-tracking --lora  "cinema" --vllm --vllm-url "http://localhost:11300"  --vllm-optim --vllm-optim-port 11300 --system-prompt --vllm-priority "system_chat" --no-fallback

multi endpoint,
l3:
python continuous_trellis_orchestrator_working_a6000_simulate.py --promptfile episodic_test_prompts.py --simulate --fastest-mv-gen --no-skip-duplicates --disable-task-tracking --lora  "cinema" --vllm --vllm-url "http://localhost:11300"  --vllm-optim --vllm-optim-port 11300 --system-prompt --vllm-priority "system_chat" --no-fallback --no-optimize

single best cinema, vllm true
l2:
python continuous_trellis_orchestrator_working_a6000_simulate.py --promptfile episodic_test_prompts.py --simulate --fastest-mv-gen --no-skip-duplicates --disable-task-tracking --lora  "cinema" --vllm --vllm-url "http://localhost:11300"  --vllm-optim --vllm-optim-port 11300 --system-prompt --vllm-priority "system_chat" --no-fallback

single best cinema, no optimize 
l1:
python continuous_trellis_orchestrator_working_a6000_simulate.py --promptfile episodic_test_prompts.py --simulate --fastest-mv-gen --no-skip-duplicates --disable-task-tracking --lora  "cinema" --vllm --vllm-url "http://localhost:11300"  --vllm-optim --vllm-optim-port 11300 --system-prompt --vllm-priority "system_chat" --no-optimize --no-submit --no-fallback

python trellis_subnit_server_mix_lora_flash_unload.py  --port 8096 --unload-flux
vllm serve manbeast3b/dpo-full_03-step20-three-gen-1   --served-model-name llama-3-2-3b-it   --generation-config auto   --port 11300   --max-model-len 1000   --gpu-memory-utilization 0.14   --dtype=bfloat16   --kv-cache-dtype=auto   --swap-space 4   --cpu-offload-gb 2
"""

# Set CUDA deterministic behavior environment variable BEFORE any imports
import os
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
from tqdm import tqdm
import asyncio
import json
import time
import random
import argparse
import requests
import base64
import logging
import traceback
import hashlib
import sqlite3
import subprocess
from pathlib import Path
from typing import List, Dict, Optional, Any, Set
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
import re
import signal
import atexit
import uuid
import socket
import os
import open_clip
from open_clip import CLIP
from open_clip.tokenizer import HFTokenizer

# Import the prompt optimizer
try:
    # from smart_prompt_optimizer_fixed import OptimizedPromptOptimizer
    # from llm_prompt_optimizer_v7_f1 import LLMPromptOptimizer
    from llm_prompt_optimizer_v12_f1_lora import LLMPromptOptimizer
    OPTIMIZED_PROMPT_OPTIMIZER_AVAILABLE = False
    print("✅ Using new performance-optimized prompt optimizer")
except ImportError:
    from prompt_optimizer import TrellisPromptOptimizer
    OPTIMIZED_PROMPT_OPTIMIZER_AVAILABLE = False
    print("⚠️ Falling back to original prompt optimizer")

# Import the organic LoRA router
try:
    from final_organic_router import FinalOrganicRouter
    ORGANIC_LORA_ROUTER_AVAILABLE = False
    print("✅ Using organic LoRA router with 100% pattern learning accuracy")
except ImportError:
    ORGANIC_LORA_ROUTER_AVAILABLE = False
    print("⚠️ Organic LoRA router not available - using default model")

# Import the reproducibility system
try:
    from llm_close_prompt_reproducibility_test import LLMClosePromptReproducibility
    REPRODUCIBILITY_SYSTEM_AVAILABLE = True
    print("✅ Using reproducibility system for pre-optimization")
except ImportError:
    REPRODUCIBILITY_SYSTEM_AVAILABLE = False
    print("⚠️ Reproducibility system not available")

import torch
seed = 42
torch.manual_seed(seed)
torch.use_deterministic_algorithms(True)

# CLIP-related imports for fallback mechanism
try:
    import torch.nn.functional as F
    import open_clip
    from open_clip import CLIP
    from open_clip.tokenizer import HFTokenizer
    from torchvision import transforms
    from PIL import Image
    import io
    import numpy as np
    CLIP_AVAILABLE = True
    print("✅ CLIP dependencies available for fallback mechanism")
except ImportError as e:
    CLIP_AVAILABLE = False
    print(f"⚠️ CLIP dependencies not available for fallback mechanism: {e}")
    print("   Fallback mechanism will be disabled")

# Make bittensor optional for environments without it
try:
    import bittensor as bt
    BITTENSOR_AVAILABLE = True
except ImportError:
    print("⚠️ Bittensor not available - harvest and submit features disabled")
    BITTENSOR_AVAILABLE = False
    bt = None

# Setup logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('continuous_trellis.log'),
        logging.StreamHandler()
    ]
)

# Cooldown constants - conservative values to avoid violations
NETWORK_DELAY_TIME_BUFFER = 60
FAILED_VALIDATOR_DELAY = 300
GENERATION_ERROR_DELAY = 180
MIN_TASK_INTERVAL = 35  # Minimum time between tasks to respect throttle period
SYNTHETIC_TRAFFIC_COOLDOWN = 301
ORGANIC_TRAFFIC_COOLDOWN = 121 
MAX_COOLDOWN_DURATION = 300
THROTTLE_PERIOD = MIN_TASK_INTERVAL
COOLDOWN_VIOLATION_THRESHOLD = 5
COOLDOWN_VIOLATION_PENALTY = 10
COOLDOWN_PENALTY = 600
EMERGENCY_COOLDOWN_BUFFER = 5
CRITICAL_VIOLATION_THRESHOLD = 100
CRITICAL_VIOLATION_COOLDOWN = 300
BASE_BLACKLIST_DURATION = 31
VIOLATION_INCREASE_DELTA = 2

def optimized_system_prompt(original_prompt: str) -> str:
    """
    Method 2: Guides the LLM by providing few-shot examples.

    Args:
        original_prompt: The user's prompt.

    Returns:
        The optimized prompt.
    """

    #v4v2hunyuan
    system_prompt = """You are 'Aetheria,' a world-class prompt artist for a 3D generative AI. Your sole purpose is to transform simple user prompts into evocative, high-performance masterpieces that consistently score above 0.9.

### The Winning Formula
Your analysis of thousands of prompts has revealed a core formula for success. You must structure your optimized prompts around this pattern:
**[Adjective/Quality] + [Color] + [Specific Object] + "with" + [Key Feature/Detail]**

### Strategic Nuance: Adapt to the Object's Category
Your strategy must adapt based on the 3D model's known strengths:
-   **High-Reliability Categories (Tools, Robots, Instruments, Weapons, Creatures):** You can be more ambitious. Add a simple, elegant, and **thematically relevant** context that enhances the object's story.
-   **Low-Reliability Categories (Gems, Jewelry, Food):** Be precise and **object-focused**. Pour all detail into the object's material, texture, and light interaction. Keep the context minimal (e.g., `on a velvet cushion`, `surrounded by a soft glow`).

### The Prompting Toolkit
-   **Keywords:** Leverage proven high-scoring words like `sleek`, `intricate`, `classic`, `glowing`, `radiant`, `delicate`.
-   **Colors:** Prioritize reliable colors like `blue`, `green`, and `black` unless specified otherwise.
-   **Brevity:** High-scoring prompts are dense with keywords but are typically **5-12 words long**. Avoid unnecessary conversational language.

### Case Studies: Your Thought Process

---
**Case Study 1: High-Reliability Object (Tool)**

* **Original Prompt:** `drill bit`
* **Thought Process:** "This is a 'tool,' a high-reliability category. I will follow the winning formula. The object is 'drill bit'. I'll add a color, `yellow`, a quality, `slender`, and a key feature, `pointed tip`."
* **High-Scoring Optimized Prompt:** `drill bit yellow slender pointed tip`

---
**Case Study 2: Low-Reliability Object (Gem)**

* **Original Prompt:** `glowing staff`
* **Thought Process:** "This is a 'gem,' a low-reliability category. I must be object-focused. The core is the `glowing staff`. I'll specify the gem (`radiant sapphire stone`) and imply craftsmanship (`topped with`). The context will be minimal to avoid failure."
* **High-Scoring Optimized Prompt:** `glowing staff topped with radiant sapphire stone`

### ANTI-PATTERNS: What to Strictly Avoid
-   **Vague Combinations:** Do not combine materials and shapes that are ambiguous for a 3D model (e.g., "triangular wooden knife").
-   **Abstract Concepts:** Do not use abstract words like "scene detail." Focus on tangible, visual qualities.
-   **Multiple Objects:** The model fails when rendering multiple distinct items. The prompt must describe a **single, unified object**.
-   **Humans and Poses:** The model cannot render people or complex actions. If a prompt includes a person, **remove them** and focus only on their associated object.

### Final Instruction
Your entire response must be **only** the final, optimized prompt.
-   **End with:** `, front view, white background`
-   **No explanations.**

Process the following `ORIGINAL PROMPT` according to these instructions.
"""
    print("\n--- System Prompt (Method 2) ---")
    print("NOTE: The example-based prompt is very long and is not fully displayed here.")
    print("---------------------------------")
    full_prompt = f"{system_prompt}\n\nORIGINAL PROMPT:\n`{original_prompt}`\n\nOPTIMIZED PROMPT:"
    return full_prompt
logger = logging.getLogger(__name__)

from trellis_subnit_server_mix_lora_flash import GENERATION_CONFIG

async def run_validator_async(original_prompt: str, optimized_prompt: str, endpoint: str, port: int, 
                              num_inference_steps: int, guidance_scale: float, 
                              ss_sampling_steps: int, slat_sampling_steps: int,
                              slat_guidance_strength: float, ss_guidance_strength: float) -> Dict[str, Any]:
    """Run the subnet validator asynchronously and return results."""
    
    print(f"🔍 Running async validation for port {port}:")
    print(f"   Original: '{original_prompt[:60]}...'")
    print(f"   Optimized: '{optimized_prompt[:60]}...'")
    
    # Build the validator command
    cmd = [
        "python", "subnet_accurate_validator_multigpu.py",
        f'"{original_prompt}"',
        f'"{optimized_prompt}"',
        "--endpoint", endpoint,
        "--port", str(port),
        "--num_inference_steps", str(num_inference_steps),
        "--guidance_scale", str(guidance_scale),
        "--ss_steps", str(ss_sampling_steps),
        "--slat_steps", str(slat_sampling_steps),
        "--slat_guidance", str(slat_guidance_strength),
        "--ss_guidance", str(ss_guidance_strength)
    ]
    
    print(f"🚀 Running async command: {' '.join(cmd)}")
    
    try:
        # Run the validator in a thread pool to avoid blocking
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None, 
            lambda: subprocess.run(cmd, capture_output=True, text=True, check=True, timeout=600)
        )
        
        # Look for the results file
        results_file = f"subnet_validation_results_{port}.json"
        if os.path.exists(results_file):
            with open(results_file, 'r') as f:
                results = json.load(f)
            print(f"✅ Async validation completed successfully for port {port}")
            return results
        else:
            print(f"❌ Results file not found: {results_file}")
            return {"error": "Results file not found"}
            
    except subprocess.TimeoutExpired:
        print(f"❌ Async validation timed out after 10 minutes for port {port}")
        return {"error": "Validation timed out"}
    except subprocess.CalledProcessError as e:
        print(f"❌ Async validation failed with exit code {e.returncode} for port {port}")
        print(f"   stdout: {e.stdout}")
        print(f"   stderr: {e.stderr}")
        return {"error": f"Validation failed: {e}"}
    except Exception as e:
        print(f"❌ Async validation error for port {port}: {e}")
        return {"error": f"Validation error: {e}"}

def clean_vllm_response(full_response: str) -> str:
    """Clean vLLM response to extract just the optimized prompt."""
    # Extract just the optimized prompt part (remove explanatory text)
    optimized_prompt = full_response
    
    # Remove common explanatory prefixes
    prefixes_to_remove = [
        "Here's an optimized prompt for 3D generation:",
        "Here's an optimized version of the prompt for 3D generation:",
        "To optimize the prompt for 3D generation, I would suggest the following:",
        "Here's the optimized prompt:",
        "Optimized prompt:",
        "Here's an enhanced version:",
        "Enhanced prompt:",
        "Here's an optimized prompt for 3D generation of a golden statue:",
        "Here's an optimized prompt for 3D generation of a",
        "Here's an optimized prompt for 3D generation:",
        "Here's the optimized prompt:",
        "Optimized prompt:",
        "Enhanced prompt:",
        "Here's an enhanced version:"
    ]
    
    for prefix in prefixes_to_remove:
        if optimized_prompt.startswith(prefix):
            optimized_prompt = optimized_prompt[len(prefix):].strip()
            break
    
    # Clean up the response (same cleaning as original script)
    optimized_prompt = optimized_prompt.replace('\n', ' ').replace('\r', ' ').replace('\t', ' ')
    optimized_prompt = ''.join(char for char in optimized_prompt if ord(char) >= 32 or char == ' ')
    optimized_prompt = ' '.join(optimized_prompt.split())
    
    return optimized_prompt

def run_validator(original_prompt: str, optimized_prompt: str, endpoint: str, port: int, 
                  num_inference_steps: int, guidance_scale: float, 
                  ss_sampling_steps: int, slat_sampling_steps: int,
                  slat_guidance_strength: float, ss_guidance_strength: float) -> Dict[str, Any]:
    """Run the subnet validator synchronously and return results."""
    
    print(f"🔍 Running sync validation for:")
    print(f"   Original: '{original_prompt[:60]}...'")
    print(f"   Optimized: '{optimized_prompt[:60]}...'")
    
    # Build the validator command
    cmd = [
        "python", "subnet_accurate_validator_multigpu.py",
        f'"{original_prompt}"',
        f'"{optimized_prompt}"',
        "--endpoint", endpoint,
        "--port", str(port),
        "--num_inference_steps", str(num_inference_steps),
        "--guidance_scale", str(guidance_scale),
        "--ss_steps", str(ss_sampling_steps),
        "--slat_steps", str(slat_sampling_steps),
        "--slat_guidance", str(slat_guidance_strength),
        "--ss_guidance", str(ss_guidance_strength)
    ]
    
    print(f"🚀 Running sync command: {' '.join(cmd)}")
    
    try:
        # Run the validator
        result = subprocess.run(cmd, capture_output=True, text=True, check=True, timeout=600)
        
        # Look for the results file
        results_file = f"subnet_validation_results_{port}.json"
        if os.path.exists(results_file):
            with open(results_file, 'r') as f:
                results = json.load(f)
            print(f"✅ Sync validation completed successfully")
            return results
        else:
            print(f"❌ Results file not found: {results_file}")
            return {"error": "Results file not found"}
            
    except subprocess.TimeoutExpired:
        print(f"❌ Sync validation timed out after 10 minutes")
        return {"error": "Validation timed out"}
    except subprocess.CalledProcessError as e:
        print(f"❌ Sync validation failed with exit code {e.returncode}")
        print(f"   stdout: {e.stdout}")
        print(f"   stderr: {e.stderr}")
        return {"error": f"Validation failed: {e}"}
    except Exception as e:
        print(f"❌ Sync validation error: {e}")
        return {"error": f"Validation error: {e}"}

'''
# vLLM Optimization Methods
def test_vllm_connection(vllm_port: int = 11300) -> bool:
    """Test connection to vLLM server for optimization."""
    try:
        vllm_url = f"http://localhost:{vllm_port}/v1/models"
        
        print(f"🔍 Testing vLLM connection on port {vllm_port}")
        
        response = requests.get(vllm_url, timeout=10)
        
        if response.status_code == 200:
            print(f"✅ vLLM connection test successful")
            return True
        else:
            print(f"❌ vLLM connection test failed with status {response.status_code}")
            return False
            
    except Exception as e:
        print(f"❌ vLLM connection test error: {e}")
        return False


def query_vllm_no_system_prompt(prompt: str, vllm_port: int = 11300) -> Optional[str]:
    """
    Query vLLM WITHOUT system prompt - just send raw prompt to completions endpoint.
    This mimics the 'no system prompt' behavior from compare_system_vs_no_system.py
    """
    try:
        vllm_url = f"http://localhost:{vllm_port}/v1/completions"
        
        # No system prompt - just send the raw prompt directly
        payload = {
            "model": "llama-3-2-3b-it",
            "prompt": f"Please optimize this prompt for 3D generation: {prompt}",
            "max_tokens": 200,
            "temperature": 0.7,
            "top_p": 0.9,
            "stream": False
        }
        
        print("📝 No system prompt - using raw prompt directly")
        
        response = requests.post(
            vllm_url,
            json=payload,
            headers={"Content-Type": "application/json"},
            timeout=30
        )
        
        if response.status_code == 200:
            result = response.json()
            full_response = result['choices'][0]['text'].strip()
            
            # Clean the response to extract just the optimized prompt
            optimized_prompt = clean_vllm_response(full_response)
            
            print(f"✅ vLLM optimization successful (no system prompt):")
            print(f"   Original: '{prompt}'")
            print(f"   Optimized: '{optimized_prompt}'")
            
            return optimized_prompt
        else:
            print(f"❌ vLLM optimization failed with status {response.status_code}: {response.text}")
            return None
            
    except Exception as e:
        print(f"❌ vLLM optimization error: {e}")
        return None


def query_vllm_with_system_prompt_chat(prompt: str, vllm_port: int = 11300) -> Optional[str]:
    """
    Query vLLM WITH system prompt using chat completions endpoint.
    Uses structured chat format with system/user/assistant messages.
    """
    try:
        vllm_url = f"http://localhost:{vllm_port}/v1/chat/completions"
        
        # System prompt that explains the task
        system_prompt = "You are a prompt optimization expert. Your task is to take a simple prompt and enhance it with detailed, descriptive language that would be perfect for 3D generation. Make the description vivid, specific, and complete. Focus on materials, textures, lighting, perspective, and artistic style."
        
        # Format with system prompt (chat format)
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Please optimize this prompt for 3D generation: {prompt}"}
        ]
        
        payload = {
            "model": "llama-3-2-3b-it",
            "messages": messages,
            "max_tokens": 200,
            "temperature": 0.7,
            "top_p": 0.9,
            "stream": False
        }
        
        print("📝 Using system prompt with chat completions")
        
        response = requests.post(
            vllm_url,
            json=payload,
            headers={"Content-Type": "application/json"},
            timeout=30
        )
        
        if response.status_code == 200:
            result = response.json()
            full_response = result['choices'][0]['message']['content'].strip()
            
            # Clean the response to extract just the optimized prompt
            optimized_prompt = clean_vllm_response(full_response)
            
            print(f"✅ vLLM optimization successful (system prompt + chat):")
            print(f"   Original: '{prompt}'")
            print(f"   Optimized: '{optimized_prompt}'")
            
            return optimized_prompt
        else:
            print(f"❌ vLLM optimization failed with status {response.status_code}: {response.text}")
            return None
            
    except Exception as e:
        print(f"❌ vLLM optimization error: {e}")
        return None


def query_vllm_with_system_prompt_completions(prompt: str, vllm_port: int = 11300) -> Optional[str]:
    """
    Query vLLM WITH system prompt using completions endpoint.
    Uses the same format as compare_system_vs_no_system.py with system prompt.
    """
    try:
        vllm_url = f"http://localhost:{vllm_port}/v1/completions"
        
        # System prompt that explains the task (same as original script)
        system_prompt = "You are a prompt optimization expert. Your task is to take a simple prompt and enhance it with detailed, descriptive language that would be perfect for 3D generation. Make the description vivid, specific, and complete. Focus on materials, textures, lighting, perspective, and artistic style."
        
        # Format with system prompt (same format as original script)
        formatted_prompt = f"<|im_start|>system\n{system_prompt}<|im_end|>\n<|im_start|>user\nPlease optimize this prompt for 3D generation: {prompt}<|im_start|>assistant\n"
        
        payload = {
            "model": "llama-3-2-3b-it",
            "prompt": formatted_prompt,
            "max_tokens": 200,
            "temperature": 0.7,
            "top_p": 0.9,
            "stream": False
        }
        
        print("📝 Using system prompt with completions (like original script)")
        
        response = requests.post(
            vllm_url,
            json=payload,
            headers={"Content-Type": "application/json"},
            timeout=30
        )
        
        if response.status_code == 200:
            result = response.json()
            full_response = result['choices'][0]['text'].strip()
            
            print(f"✅ vLLM optimization successful (system prompt + completions):")
            print(f"   Original: '{prompt}'")
            print(f"   Optimized: '{full_response}'")
            
            return full_response
        else:
            print(f"❌ vLLM optimization failed with status {response.status_code}: {response.text}")
            return None
            
    except Exception as e:
        print(f"❌ vLLM optimization error: {e}")
        return None





def optimize_prompt_with_vllm(prompt: str, vllm_port: int = 11300, priority: str = 'system_chat') -> Optional[str]:
    """
    Optimize prompt using vLLM with the configured priority method.
    Returns optimized prompt or None if failed.
    """
    try:
        # Test vLLM connection first
        if not test_vllm_connection(vllm_port):
            print(f"❌ vLLM connection test failed, skipping vLLM optimization")
            return None
        
        print(f"🚀 Using vLLM optimization with priority: {priority}")
        
        # Try the priority method first
        optimized_prompt = None
        
        if priority == 'system_chat':
            optimized_prompt = query_vllm_with_system_prompt_chat(prompt, vllm_port)
        elif priority == 'system_completions':
            optimized_prompt = query_vllm_with_system_prompt_completions(prompt, vllm_port)
        elif priority == 'no_system':
            optimized_prompt = query_vllm_no_system_prompt(prompt, vllm_port)
        
        # If priority method failed, try fallback methods
        if not optimized_prompt:
            print(f"⚠️ Priority method {priority} failed, trying fallbacks...")
            
            if priority != 'system_chat':
                optimized_prompt = query_vllm_with_system_prompt_chat(prompt, vllm_port)
            
            if not optimized_prompt and priority != 'system_completions':
                optimized_prompt = query_vllm_with_system_prompt_completions(prompt, vllm_port)
            
            if not optimized_prompt and priority != 'no_system':
                optimized_prompt = query_vllm_no_system_prompt(prompt, vllm_port)
        
        if optimized_prompt:
            print(f"✅ vLLM optimization successful:")
            print(f"   Method: {priority}")
            print(f"   Original: '{prompt[:50]}...'")
            print(f"   Optimized: '{optimized_prompt[:50]}...'")
            
            return optimized_prompt
        else:
            # All vLLM methods failed
            print(f"❌ All vLLM optimization methods failed")
            return None
            
    except Exception as e:
        print(f"❌ vLLM optimization failed: {e}")
        return None
'''

def fast_quality_check(gs_data, verbose=True) -> tuple[bool, str]:
    """Ultra-fast quality check that takes <1 second"""
    
    # Quick checks that don't require full validation
    issues = []
    
    # Check splat count (fast)
    if gs_data.points.shape[0] < 7000:
        issues.append(f"Insufficient splats: {gs_data.points.shape[0]} < 7000")
    
    # Check opacity distribution (fast)
    zero_opacity = torch.sum(gs_data.opacities < 1e-3).item()
    opacity_pct = 100 * zero_opacity / len(gs_data.opacities)
    if opacity_pct > 80:
        issues.append(f"Too many zero opacity: {opacity_pct:.1f}%")
    
    # Check scale distribution (fast)
    zero_scales = torch.sum(torch.all(gs_data.scales < 0.001, dim=1)).item()
    scale_pct = 100 * zero_scales / len(gs_data.scales)
    if scale_pct > 80:
        issues.append(f"Too many zero scales: {scale_pct:.1f}%")
    
    is_valid = len(issues) == 0
    return is_valid, "; ".join(issues) if issues else "All checks passed"

class CLIPTextSimilarityServer:
    """CLIP text-to-text similarity server that keeps model in memory"""
    
    def __init__(self, verbose: bool = False):  
        self.verbose = verbose
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_name = "convnext_large_d"
        self.pretrained = "laion2b_s26b_b102k_augreg"
        
        # Model components
        self._model: CLIP = None
        self._tokenizer: HFTokenizer = None
        
        logger.info(f"🔧 CLIPTextSimilarityServer initialized")
        logger.info(f"   CLIP Model: {self.model_name}/{self.pretrained}")
        logger.info(f"   Device: {self.device}")
    
    def load_model_on_device(self, device_str):
        """Load CLIP model on specified device and measure loading time"""
        # Create proper torch device object
        device = torch.device(device_str)
        print(f"🔧 Loading CLIP model on {device}...")
        
        start_time = time.time()
        try:
            self._model, _, _ = open_clip.create_model_and_transforms(
                self.model_name, 
                pretrained=self.pretrained,
                device=device
            )
            self._tokenizer = open_clip.get_tokenizer(self.model_name)
            end_time = time.time()
            loading_time = (end_time - start_time) * 1000  # Convert to milliseconds
            
            print(f"✅ CLIP model loaded successfully on {device}")
            print(f"   Loading time: {loading_time:.2f} ms")
            return self._model, self._tokenizer, loading_time, device
        except Exception as e:
            print(f"❌ Failed to load CLIP model on {device}: {e}")
            return None, None, 0, device

    def load_clip_model(self, device="cpu"):
        """Load CLIP model and tokenizer (keeps in memory)"""
        logger.info(f"🔧 Loading CLIP model: {self.model_name}/{self.pretrained}")
        if device == "cpu":
            self.device = torch.device("cpu")
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        try:
            self._model, _, _ = open_clip.create_model_and_transforms(
                self.model_name, 
                pretrained=self.pretrained,
                device=self.device
            )
            self._tokenizer = open_clip.get_tokenizer(self.model_name)
            logger.info(f"✅ CLIP model loaded successfully and kept in memory")
        except Exception as e:
            logger.error(f"❌ Failed to load CLIP model: {e}")
            raise

    
    
    def compute_cosine_similarity(self, text1: str, text2: str) -> dict:
        """Compute cosine similarity between two texts using CLIP"""
        if self._model is None or self._tokenizer is None:
            raise RuntimeError("CLIP model not loaded. Call load_clip_model() first.")
        
        try:
            # Tokenize both texts
            tokenized_text1 = self._tokenizer(text1).to(self.device)
            tokenized_text2 = self._tokenizer(text2).to(self.device)
            
            with torch.no_grad(), torch.amp.autocast(self.device.type):
                # Encode both texts
                text1_features = self._model.encode_text(tokenized_text1)
                text2_features = self._model.encode_text(tokenized_text2)
                
                # Normalize features
                text1_features /= text1_features.norm(dim=-1, keepdim=True)
                text2_features /= text2_features.norm(dim=-1, keepdim=True)
                
                # Compute cosine similarity
                similarity = (text1_features @ text2_features.T).cpu().numpy()[0][0]
                
                # Clip to [0, 1] range
                similarity = max(0.0, min(1.0, similarity))
                
                # Determine similarity level
                if similarity >= 0.9:
                    level = "Very High"
                    description = "Texts are nearly identical"
                elif similarity >= 0.8:
                    level = "High"
                    description = "Texts are very similar"
                elif similarity >= 0.7:
                    level = "Good"
                    description = "Texts maintain strong semantic similarity"
                elif similarity >= 0.6:
                    level = "Moderate"
                    description = "Texts have good semantic overlap"
                elif similarity >= 0.5:
                    level = "Fair"
                    description = "Texts have some semantic overlap"
                elif similarity >= 0.4:
                    level = "Low"
                    description = "Texts have limited semantic overlap"
                else:
                    level = "Very Low"
                    description = "Texts may be semantically different"
                
                return {
                    "success": True,
                    "text1": text1,
                    "text2": text2,
                    "cosine_similarity": float(similarity),
                    "similarity_level": level,
                    "description": description
                }
                
        except Exception as e:
            logger.error(f"❌ Cosine similarity computation failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "text1": text1,
                "text2": text2
            }

    def compute_similarity_device(self, device, text1: str, text2: str, num_runs=10, warmup_runs=3, timer=False):
        """Compute similarity multiple times and measure performance with warmup option"""
        # Ensure device is a torch.device object
        if isinstance(device, str):
            device = torch.device(device)
            
        print(f"📊 Running {num_runs} similarity computations on {device} with {warmup_runs} warmup runs...")
        print(f"   Device type: {device.type}, Device: {device}")
        
        # Check if model is loaded
        if self._model is None or self._tokenizer is None:
            print(f"❌ CLIP model not loaded. Loading model on {device}...")
            try:
                self._model, _, _ = open_clip.create_model_and_transforms(
                    self.model_name, 
                    pretrained=self.pretrained,
                    device=device
                )
                self._tokenizer = open_clip.get_tokenizer(self.model_name)
                print(f"✅ CLIP model loaded successfully on {device}")
            except Exception as e:
                print(f"❌ Failed to load CLIP model on {device}: {e}")
                return {
                    "success": False,
                    "error": f"Failed to load CLIP model: {e}",
                    "text1": text1,
                    "text2": text2
                }
        else:
            print(f"✅ Using existing CLIP model and tokenizer")
        
        times = []
        similarities = []
        
        # Warmup runs (not counted in final results)
        if warmup_runs > 0:
            print(f"🔥 Running {warmup_runs} warmup runs...")
            for i in range(warmup_runs):
                try:
                    # Tokenize both texts
                    tokenized_text1 = self._tokenizer(text1).to(device)
                    tokenized_text2 = self._tokenizer(text2).to(device)
                    
                    # Use proper autocast for the device
                    if device.type == 'cuda':
                        with torch.no_grad(), torch.amp.autocast(device_type='cuda'):
                            # Encode both texts
                            text1_features = self._model.encode_text(tokenized_text1)
                            text2_features = self._model.encode_text(tokenized_text2)
                            
                            # Normalize features
                            text1_features /= text1_features.norm(dim=-1, keepdim=True)
                            text2_features /= text2_features.norm(dim=-1, keepdim=True)
                            
                            # Compute cosine similarity
                            similarity = (text1_features @ text2_features.T).cpu().numpy()[0][0]
                    else:
                        with torch.no_grad():
                            # Encode both texts
                            text1_features = self._model.encode_text(tokenized_text1)
                            text2_features = self._model.encode_text(tokenized_text2)
                            
                            # Normalize features
                            text1_features /= text1_features.norm(dim=-1, keepdim=True)
                            text2_features /= text2_features.norm(dim=-1, keepdim=True)
                            
                            # Compute cosine similarity
                            similarity = (text1_features @ text2_features.T).cpu().numpy()[0][0]
                    
                    print(f"   Warmup {i + 1}/{warmup_runs} completed with similarity: {similarity:.4f}")
                        
                except Exception as e:
                    print(f"❌ Error in warmup run {i + 1}: {e}")
                    continue
        
        # Actual timed runs
        print(f"⏱️ Starting {num_runs} timed runs...")
        if num_runs == 0:
            print("   Skipping timed runs (num_runs=0)")
            return {
                "success": True,
                "text1": text1,
                "text2": text2,
                "cosine_similarity": 0.0,
                "similarity_level": "Unknown",
                "description": "No timed runs performed",
                "performance_metrics": {
                    "num_runs": 0,
                    "warmup_runs": warmup_runs,
                    "avg_time_ms": 0,
                    "min_time_ms": 0,
                    "max_time_ms": 0,
                    "device": str(device),
                    "all_times_ms": [],
                    "all_similarities": []
                }
            }
            
        for i in range(num_runs):
            if timer:
                start_time = time.time()
            
            try:
                # Tokenize both texts
                tokenized_text1 = self._tokenizer(text1).to(device)
                tokenized_text2 = self._tokenizer(text2).to(device)
                
                # Use proper autocast for the device
                if device.type == 'cuda':
                    with torch.no_grad(), torch.amp.autocast(device_type='cuda'):
                        # Encode both texts
                        text1_features = self._model.encode_text(tokenized_text1)
                        text2_features = self._model.encode_text(tokenized_text2)
                        
                        # Normalize features
                        text1_features /= text1_features.norm(dim=-1, keepdim=True)
                        text2_features /= text2_features.norm(dim=-1, keepdim=True)
                        
                        # Compute cosine similarity
                        similarity = (text1_features @ text2_features.T).cpu().numpy()[0][0]
                        similarity = max(0.0, min(1.0, similarity))
                        
                        similarities.append(similarity)
                else:
                    with torch.no_grad():
                        # Encode both texts
                        text1_features = self._model.encode_text(tokenized_text1)
                        text2_features = self._model.encode_text(tokenized_text2)
                        
                        # Normalize features
                        text1_features /= text1_features.norm(dim=-1, keepdim=True)
                        text2_features /= text2_features.norm(dim=-1, keepdim=True)
                        
                        # Compute cosine similarity
                        similarity = (text1_features @ text2_features.T).cpu().numpy()[0][0]
                        similarity = max(0.0, min(1.0, similarity))
                        
                        similarities.append(similarity)
                
                if timer:
                    end_time = time.time()
                    run_time = (end_time - start_time) * 1000  # Convert to milliseconds
                    times.append(run_time)
                
                if (i + 1) % 5 == 0:
                    print(f"   Completed {i + 1}/{num_runs} runs...")
                
            except Exception as e:
                print(f"❌ Error in run {i + 1}: {e}")
                continue
        
        # Calculate performance statistics
        if similarities:
            if timer:
                avg_time = sum(times) / len(times)
                min_time = min(times)
                max_time = max(times)
            else:
                avg_time = 0
                min_time = 0
                max_time = 0
            avg_similarity = sum(similarities) / len(similarities)
            
            # Determine similarity level based on average
            if avg_similarity >= 0.9:
                level = "Very High"
                description = "Texts are nearly identical"
            elif avg_similarity >= 0.8:
                level = "High"
                description = "Texts are very similar"
            elif avg_similarity >= 0.7:
                level = "Good"
                description = "Texts maintain strong semantic similarity"
            elif avg_similarity >= 0.6:
                level = "Moderate"
                description = "Texts have good semantic overlap"
            elif avg_similarity >= 0.5:
                level = "Fair"
                description = "Texts have some semantic overlap"
            elif avg_similarity >= 0.4:
                level = "Low"
                description = "Texts have limited semantic overlap"
            else:
                level = "Very Low"
                description = "Texts may be semantically different"
            
            return {
                "success": True,
                "text1": text1,
                "text2": text2,
                "cosine_similarity": float(avg_similarity),
                "similarity_level": level,
                "description": description,
                "performance_metrics": {
                    "num_runs": len(times),
                    "warmup_runs": warmup_runs,
                    "avg_time_ms": round(avg_time, 2),
                    "min_time_ms": round(min_time, 2),
                    "max_time_ms": round(max_time, 2),
                    "device": str(device),
                    "all_times_ms": [round(t, 2) for t in times],
                    "all_similarities": [float(s) for s in similarities]
                }
            }
        else:
            return {
                "success": False,
                "error": "No successful runs completed",
                "text1": text1,
                "text2": text2,
                "performance_metrics": {
                    "num_runs": 0,
                    "warmup_runs": warmup_runs,
                    "device": str(device)
                }
            }


class PriorityServerCoordinator:
    """
    Priority-based server coordinator that gives the orchestrator HIGH PRIORITY access.
    This allows time-critical subnet tasks to bypass or interrupt other processes.
    """
    
    def __init__(self, server_url: str = "http://localhost:8096", 
                 max_wait_time_seconds: int = 60,
                 status_check_interval: int = 1,
                 priority_timeout: int = 30,
                 on_interruption_callback=None):
        """
        Initialize the priority server coordinator.
        
        Args:
            server_url: Base URL of the GPU server
            max_wait_time_seconds: Maximum time to wait for server availability
            status_check_interval: Interval between status checks (faster for priority)
            priority_timeout: Timeout for priority access attempts
        """
        self.server_url = server_url.rstrip('/')
        self.max_wait_time_seconds = max_wait_time_seconds
        self.status_check_interval = status_check_interval
        self.priority_timeout = priority_timeout
        self.on_interruption_callback = on_interruption_callback
        self.logger = logging.getLogger(__name__)
        
    def check_server_status(self) -> Dict[str, Any]:
        """
        Check the current status of the GPU server.
        
        Returns:
            Dictionary containing server status information
        """
        try:
            # First check health endpoint
            health_url = f"{self.server_url}/health/"
            health_resp = requests.get(health_url, timeout=3)  # Faster timeout for priority
            if health_resp.status_code != 200:
                return {
                    "available": False,
                    "status": "unhealthy",
                    "error": f"Health check failed: HTTP {health_resp.status_code}"
                }
            
            # Check job status
            job_status_url = f"{self.server_url}/job/status/"
            job_resp = requests.get(job_status_url, timeout=3)
            if job_resp.status_code != 200:
                return {
                    "available": False,
                    "status": "unknown",
                    "error": f"Job status check failed: HTTP {job_resp.status_code}"
                }
            
            job_data = job_resp.json()
            job_status = job_data.get('status', 'unknown')
            
            # For priority coordinator, we consider server available if it's not in critical busy states
            # We can interrupt non-critical operations
            if job_status in ('processing', 'generating', 'validating'):
                # Check if this is a priority operation (ours) or low priority (optimizer)
                job_id = job_data.get('job_id', '')
                prompt = job_data.get('prompt', '')
                
                # If it's our job, we can use the server
                if self._is_our_job(job_id, prompt):
                    return {
                        "available": True,
                        "status": job_status,
                        "job_id": job_id,
                        "our_job": True
                    }
                else:
                    # It's someone else's job - we can interrupt for priority
                    return {
                        "available": True,  # Available for priority access
                        "status": f"interruptible_{job_status}",
                        "job_id": job_id,
                        "prompt": prompt,
                        "interruptible": True
                    }
            
            # Server is available
            return {
                "available": True,
                "status": job_status,
                "job_id": job_data.get('job_id')
            }
            
        except requests.exceptions.Timeout:
            return {
                "available": False,
                "status": "timeout",
                "error": "Server status check timed out"
            }
        except requests.exceptions.ConnectionError:
            return {
                "available": False,
                "status": "connection_error",
                "error": "Cannot connect to server"
            }
        except Exception as e:
            return {
                "available": False,
                "status": "error",
                "error": str(e)
            }
    
    def _is_our_job(self, job_id: str, prompt: str) -> bool:
        """
        Determine if the current job is ours (orchestrator) or someone else's (optimizer).
        
        Args:
            job_id: Current job ID
            prompt: Current prompt being processed
            
        Returns:
            True if this is our job, False if it's someone else's
        """
        # Check if job_id contains our identifiers
        if job_id and any(identifier in job_id.lower() for identifier in ['orchestrator', 'subnet', 'miner', 'task']):
            return True
        
        # Check if prompt matches our patterns (subnet tasks are usually shorter and specific)
        if prompt and len(prompt) < 100:  # Subnet tasks are typically shorter
            return True
        
        # Default: assume it's not our job (optimizer jobs are usually longer prompts)
        return False
    
    def wait_for_priority_access(self, task_id: str = None) -> bool:
        """
        Wait for priority access to the server, with ability to interrupt other processes.
        
        Args:
            task_id: Our task ID for identification
            
        Returns:
            True if priority access granted, False if timeout reached
        """
        start_wait_time = time.time()
        
        while time.time() - start_wait_time < self.max_wait_time_seconds:
            status = self.check_server_status()
            
            if status["available"]:
                if status.get("interruptible"):
                    self.logger.warning(f"🚨 PRIORITY INTERRUPTION: Interrupting job {status.get('job_id', 'unknown')} for subnet task {task_id}")
                    # Force clear the server to interrupt the current job
                    self._force_clear_server()
                    time.sleep(2)  # Brief pause for server to reset
                    # Track this interruption
                    if self.on_interruption_callback:
                        self.on_interruption_callback()
                    return True
                else:
                    self.logger.info(f"✅ Priority access granted (status: {status['status']})")
                    return True
            
            # Log the current status
            error = status.get("error", "unknown error")
            self.logger.info(f"⏳ Waiting for priority access: {status['status']} - {error}")
            
            # Wait before next check (faster for priority)
            time.sleep(self.status_check_interval)
        
        self.logger.error(f"⏰ Priority access timeout ({self.max_wait_time_seconds}s) - subnet task may be missed!")
        return False
    
    def _force_clear_server(self):
        """
        Force clear the server to interrupt current operations.
        This is used for priority access when subnet tasks are at risk.
        """
        try:
            # Try to clear cache
            clear_url = f"{self.server_url}/clear_cache/"
            resp = requests.post(clear_url, timeout=5)
            if resp.status_code == 200:
                self.logger.info("🧹 Server cache cleared for priority access")
            else:
                self.logger.warning(f"⚠️ Failed to clear server cache: HTTP {resp.status_code}")
            
            # Try to reset job status
            reset_url = f"{self.server_url}/job/reset/"
            resp = requests.post(reset_url, timeout=5)
            if resp.status_code == 200:
                self.logger.info("🔄 Server job status reset for priority access")
            else:
                self.logger.warning(f"⚠️ Failed to reset job status: HTTP {resp.status_code}")
                
        except Exception as e:
            self.logger.warning(f"⚠️ Exception during force clear: {e}")
    
    def clear_server_cache(self) -> bool:
        """
        Clear the GPU cache on the server.
        
        Returns:
            True if successful, False otherwise
        """
        try:
            clear_url = f"{self.server_url}/clear_cache/"
            resp = requests.post(clear_url, timeout=5)
            if resp.status_code == 200:
                self.logger.info("🧹 GPU cache cleared successfully")
                return True
            else:
                self.logger.warning(f"⚠️ Failed to clear GPU cache: HTTP {resp.status_code}")
                return False
        except Exception as e:
            self.logger.warning(f"⚠️ Exception clearing GPU cache: {e}")
            return False
    
    def mark_priority_job_start(self, task_id: str, prompt: str):
        """
        Mark the start of a priority job to help with identification.
        
        Args:
            task_id: Our task ID
            prompt: The prompt being processed
        """
        self.logger.info(f"🚀 Starting PRIORITY job: {task_id} - '{prompt[:50]}...'")
    
    def mark_priority_job_end(self, task_id: str):
        """
        Mark the end of a priority job.
        
        Args:
            task_id: Our task ID
        """
        self.logger.info(f"✅ Completed PRIORITY job: {task_id}")



def optimized_system_prompt(original_prompt: str) -> str:
    """
    Method 2: Guides the LLM by providing few-shot examples.

    Args:
        original_prompt: The user's prompt.

    Returns:
        The optimized prompt.
    """

    #v4v2hunyuan
    system_prompt = """You are 'Aetheria,' a world-class prompt artist for a 3D generative AI. Your sole purpose is to transform simple user prompts into evocative, high-performance masterpieces that consistently score above 0.9.

### The Winning Formula
Your analysis of thousands of prompts has revealed a core formula for success. You must structure your optimized prompts around this pattern:
**[Adjective/Quality] + [Color] + [Specific Object] + "with" + [Key Feature/Detail]**

### Strategic Nuance: Adapt to the Object's Category
Your strategy must adapt based on the 3D model's known strengths:
-   **High-Reliability Categories (Tools, Robots, Instruments, Weapons, Creatures):** You can be more ambitious. Add a simple, elegant, and **thematically relevant** context that enhances the object's story.
-   **Low-Reliability Categories (Gems, Jewelry, Food):** Be precise and **object-focused**. Pour all detail into the object's material, texture, and light interaction. Keep the context minimal (e.g., `on a velvet cushion`, `surrounded by a soft glow`).

### The Prompting Toolkit
-   **Keywords:** Leverage proven high-scoring words like `sleek`, `intricate`, `classic`, `glowing`, `radiant`, `delicate`.
-   **Colors:** Prioritize reliable colors like `blue`, `green`, and `black` unless specified otherwise.
-   **Brevity:** High-scoring prompts are dense with keywords but are typically **5-12 words long**. Avoid unnecessary conversational language.

### Case Studies: Your Thought Process

---
**Case Study 1: High-Reliability Object (Tool)**

* **Original Prompt:** `drill bit`
* **Thought Process:** "This is a 'tool,' a high-reliability category. I will follow the winning formula. The object is 'drill bit'. I'll add a color, `yellow`, a quality, `slender`, and a key feature, `pointed tip`."
* **High-Scoring Optimized Prompt:** `drill bit yellow slender pointed tip`

---
**Case Study 2: Low-Reliability Object (Gem)**

* **Original Prompt:** `glowing staff`
* **Thought Process:** "This is a 'gem,' a low-reliability category. I must be object-focused. The core is the `glowing staff`. I'll specify the gem (`radiant sapphire stone`) and imply craftsmanship (`topped with`). The context will be minimal to avoid failure."
* **High-Scoring Optimized Prompt:** `glowing staff topped with radiant sapphire stone`

### ANTI-PATTERNS: What to Strictly Avoid
-   **Vague Combinations:** Do not combine materials and shapes that are ambiguous for a 3D model (e.g., "triangular wooden knife").
-   **Abstract Concepts:** Do not use abstract words like "scene detail." Focus on tangible, visual qualities.
-   **Multiple Objects:** The model fails when rendering multiple distinct items. The prompt must describe a **single, unified object**.
-   **Humans and Poses:** The model cannot render people or complex actions. If a prompt includes a person, **remove them** and focus only on their associated object.

### Final Instruction
Your entire response must be **only** the final, optimized prompt.
-   **End with:** `, front view, white background`
-   **No explanations.**

Process the following `ORIGINAL PROMPT` according to these instructions.
"""
    print("\n--- System Prompt (Method 2) ---")
    print("NOTE: The example-based prompt is very long and is not fully displayed here.")
    print("---------------------------------")
    full_prompt = f"{system_prompt}\n\nORIGINAL PROMPT:\n`{original_prompt}`\n\nOPTIMIZED PROMPT:"
    return full_prompt

@dataclass
class TaskRecord:
    """Record of a task with full metadata"""
    task_id: str
    prompt: str
    prompt_hash: str
    validator_uid: int
    validator_hotkey: str
    validator_stake: float
    validation_threshold: float
    pulled_at: float
    processed_at: Optional[float] = None
    submitted_at: Optional[float] = None
    generation_time: Optional[float] = None
    validation_time: Optional[float] = None
    total_processing_time: Optional[float] = None
    local_validation_score: Optional[float] = None
    submission_success: bool = False
    feedback_received: bool = False
    # Feedback scores
    task_fidelity_score: Optional[float] = None
    average_fidelity_score: Optional[float] = None
    current_miner_reward: Optional[float] = None
    validation_failed: Optional[bool] = None
    generations_in_window: Optional[int] = None
    # File paths
    ply_file_path: Optional[str] = None
    compressed_file_path: Optional[str] = None
    
    # Priority access tracking
    priority_access_timeout: bool = False

@dataclass 
class ValidatorState:
    """State tracking for each validator"""
    uid: int
    hotkey: str
    stake: float
    trust: float
    consensus: float
    last_task_pull: Optional[float] = None
    last_task_received: Optional[float] = None
    # cooldown_until: Optional[float] = None  # DEPRECATED: Replaced by validator_enforced_cooldown_until and miner_cooldown_until
    total_tasks_pulled: int = 0
    total_tasks_received: int = 0
    total_tasks_submitted: int = 0
    total_successful_submissions: int = 0
    average_score: float = 0.0
    recent_prompts: Set[str] = None
    is_active: bool = True
    
    # Enhanced cooldown and validation tracking
    throttle_period: int = 0
    cooldown_violations: int = 0
    # DEPRECATED: Validation lock removed - now using MIN_TASK_INTERVAL constant for rate limiting
    # validation_locked_until: Optional[float] = None
    last_submit_time: Optional[float] = None
    
    # Emergency cooldown management
    emergency_blacklist_until: Optional[float] = None
    last_violation_check: Optional[float] = None
    
    # FIXED: Separate miner and validator cooldown tracking (subnet compliance)
    validator_enforced_cooldown_until: Optional[float] = None  # Validator's exact cooldown
    miner_cooldown_until: Optional[float] = None              # Miner's own cooldown logic
    validator_reported_violations: int = 0                    # Violations reported by validator
    pending_cooldown_task_id: Optional[str] = None            # Task ID that has pending cooldown

    def __post_init__(self):
        if self.recent_prompts is None:
            self.recent_prompts = set()

class ValidatorStatePersistence:
    """
    Handles saving and loading validator states to maintain cooldowns, violations, 
    and blacklists across script restarts.
    """
    
    def __init__(self, state_file: str = "validator_states.json"):
        self.state_file = Path(state_file)
        self.backup_file = Path(f"{state_file}.backup")
        self.logger = logging.getLogger(__name__)
        
    def save_validator_states(self, validators: Dict[int, 'ValidatorState']) -> bool:
        """
        Save validator states to disk.
        
        Args:
            validators: Dictionary of validator states to save
            
        Returns:
            True if save was successful, False otherwise
        """
        try:
            current_time = time.time()
            state_data = {
                'saved_at': current_time,
                'saved_at_readable': datetime.fromtimestamp(current_time).isoformat(),
                'version': '2.0',
                'validators': {}
            }
            
            for uid, validator in validators.items():
                # Only save essential state information
                validator_state = {
                    'uid': validator.uid,
                    'stake': validator.stake,
                    'is_active': validator.is_active,
                    
                    # Cooldown and violation tracking (CRITICAL)
                    # FIX 2: Use only one field per type
                    # 'cooldown_until': validator.cooldown_until,
                    # 'cooldown_violations': validator.cooldown_violations,
                    'throttle_period': validator.throttle_period,

                    # FIXED: Add subnet-compliant cooldown fields
                    'validator_enforced_cooldown_until': validator.validator_enforced_cooldown_until,
                    'miner_cooldown_until': validator.miner_cooldown_until,
                    'validator_reported_violations': validator.validator_reported_violations,
                    'pending_cooldown_task_id': validator.pending_cooldown_task_id,
                    
                    # Validation and emergency state (CRITICAL)
                    # DEPRECATED: Validation lock removed - now using MIN_TASK_INTERVAL constant for rate limiting
                    # 'validation_locked_until': validator.validation_locked_until,
                    'emergency_blacklist_until': validator.emergency_blacklist_until,
                    'last_submit_time': validator.last_submit_time,
                    'last_violation_check': validator.last_violation_check,
                    
                    # Performance tracking
                    'total_tasks_received': validator.total_tasks_received,
                    'total_tasks_submitted': validator.total_tasks_submitted,
                    'total_successful_submissions': validator.total_successful_submissions,
                    'average_score': validator.average_score,
                    
                    # History for learning (limited to prevent bloat)
                    'violation_history': getattr(validator, 'violation_history', [])[-5:],  # Last 5 only
                    'buffer_history': getattr(validator, 'buffer_history', [])[-3:],  # Last 3 only
                }
                
                state_data['validators'][str(uid)] = validator_state
            
            # Create backup of existing file
            if self.state_file.exists():
                import shutil
                shutil.copy2(self.state_file, self.backup_file)
            
            # Save new state
            with open(self.state_file, 'w') as f:
                json.dump(state_data, f, indent=2)
            
            self.logger.info(f"💾 Saved {len(validators)} validator states to {self.state_file}")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Failed to save validator states: {e}")
            self.logger.error(f"   Traceback: {traceback.format_exc()}")
            return False
    
    def load_validator_states(self) -> Dict[int, Dict[str, Any]]:
        """
        Load validator states from disk.
        
        Returns:
            Dictionary of validator states keyed by UID
        """
        try:
            if not self.state_file.exists():
                self.logger.info(f"📁 No existing state file found at {self.state_file}")
                return {}
            
            with open(self.state_file, 'r') as f:
                state_data = json.load(f)
            
            # Validate file format
            if 'validators' not in state_data:
                self.logger.warning(f"⚠️ Invalid state file format - missing 'validators' key")
                return {}
            
            saved_at = state_data.get('saved_at', 0)
            saved_at_readable = state_data.get('saved_at_readable', 'unknown')
            current_time = time.time()
            age_hours = (current_time - saved_at) / 3600
            
            self.logger.info(f"📂 Loading validator states from {self.state_file}")
            self.logger.info(f"   File saved: {saved_at_readable}")
            self.logger.info(f"   File age: {age_hours:.1f} hours")
            
            # Convert string UIDs back to integers
            validator_states = {}
            loaded_count = 0
            expired_cooldowns = 0
            active_violations = 0
            
            for uid_str, validator_data in state_data['validators'].items():
                try:
                    uid = int(uid_str)
                    # Check if cooldowns have expired
                            # DEPRECATED: cooldown_until field handling - now using validator_enforced_cooldown_until and miner_cooldown_until
                        # if validator_data.get('cooldown_until'):
                        #     if current_time >= validator_data['cooldown_until']:
                        #         validator_data['cooldown_until'] = None
                    expired_cooldowns += 1
                    
                    # DEPRECATED: Validation lock database cleanup removed - now using MIN_TASK_INTERVAL constant for rate limiting
                    # if validator_data.get('validation_locked_until'):
                    #     if current_time >= validator_data['validation_locked_until']:
                    #         validator_data['validation_locked_until'] = None
                    
                    if validator_data.get('emergency_blacklist_until'):
                        if current_time >= validator_data['emergency_blacklist_until']:
                            validator_data['emergency_blacklist_until'] = None
                        else:
                            # Still blacklisted
                            validator_data['is_active'] = False
                    
                    # Count active violations
                    if validator_data.get('cooldown_violations', 0) > 0:
                        active_violations += 1
                    
                    validator_states[uid] = validator_data
                    loaded_count += 1
                    
                except (ValueError, KeyError) as e:
                    self.logger.warning(f"⚠️ Skipping invalid validator data for UID {uid_str}: {e}")
            
            self.logger.info(f"✅ Loaded {loaded_count} validator states")
            self.logger.info(f"   Expired cooldowns cleaned: {expired_cooldowns}")
            self.logger.info(f"   Validators with active violations: {active_violations}")
            
            # If file is very old (>24 hours), suggest caution
            if age_hours > 24:
                self.logger.warning(f"⚠️ State file is {age_hours:.1f} hours old - some data may be stale")
            
            return validator_states
            
        except Exception as e:
            self.logger.error(f"❌ Failed to load validator states: {e}")
            self.logger.error(f"   Attempting to load backup...")
            
            # Try backup file
            try:
                if self.backup_file.exists():
                    with open(self.backup_file, 'r') as f:
                        backup_data = json.load(f)
                    self.logger.warning(f"🔄 Loaded backup state file")
                    return self.load_validator_states_from_data(backup_data)
            except Exception as backup_error:
                self.logger.error(f"❌ Backup file also failed: {backup_error}")
            
            return {}
    
    def load_validator_states_from_data(self, state_data: Dict) -> Dict[int, Dict[str, Any]]:
        """Helper method to load states from parsed JSON data"""
        if 'validators' not in state_data:
            return {}
        
        validator_states = {}
        for uid_str, validator_data in state_data['validators'].items():
            try:
                uid = int(uid_str)
                validator_states[uid] = validator_data
            except ValueError:
                continue
        
        return validator_states
    
    def cleanup_old_states(self, max_age_hours: float = 168):  # 7 days default
        """
        Clean up very old state files.
        
        Args:
            max_age_hours: Maximum age in hours before cleanup
        """
        try:
            if self.state_file.exists():
                file_age = time.time() - self.state_file.stat().st_mtime
                age_hours = file_age / 3600
                
                if age_hours > max_age_hours:
                    backup_name = f"validator_states_old_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
                    self.state_file.rename(backup_name)
                    self.logger.info(f"🧹 Archived old state file: {backup_name}")
                    
        except Exception as e:
            self.logger.warning(f"⚠️ Failed to cleanup old states: {e}")

class TaskDatabase:
    """SQLite database for task tracking and deduplication"""
    
    def __init__(self, db_path: str = "continuous_trellis_tasks.db"):
        self.db_path = db_path
        self.init_database()
    
    def init_database(self):
        """Initialize the database schema"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Tasks table with comprehensive tracking
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS tasks (
                task_id TEXT PRIMARY KEY,
                prompt TEXT NOT NULL,
                prompt_hash TEXT NOT NULL,
                validator_uid INTEGER NOT NULL,
                validator_hotkey TEXT NOT NULL,
                validator_stake REAL NOT NULL,
                validation_threshold REAL NOT NULL,
                pulled_at REAL NOT NULL,
                processed_at REAL,
                submitted_at REAL,
                generation_time REAL,
                validation_time REAL,
                total_processing_time REAL,
                local_validation_score REAL,
                submission_success BOOLEAN DEFAULT FALSE,
                feedback_received BOOLEAN DEFAULT FALSE,
                task_fidelity_score REAL,
                average_fidelity_score REAL,
                current_miner_reward REAL,
                validation_failed BOOLEAN,
                generations_in_window INTEGER,
                ply_file_path TEXT,
                compressed_file_path TEXT,
                created_at REAL DEFAULT (strftime('%s', 'now'))
            )
        ''')
        
        # SHARED TASK TRACKING TABLE - Prevents duplicate task processing across instances
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS shared_task_tracking (
                task_id TEXT PRIMARY KEY,
                validator_uid INTEGER NOT NULL,
                status TEXT DEFAULT 'in_progress',
                instance_id TEXT NOT NULL,
                started_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                timeout_at TIMESTAMP,
                completed_at TIMESTAMP,
                instance_hostname TEXT,
                instance_pid INTEGER
            )
        ''')
        
        # Validators table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS validators (
                uid INTEGER PRIMARY KEY,
                hotkey TEXT NOT NULL,
                stake REAL NOT NULL,
                trust REAL NOT NULL,
                consensus REAL NOT NULL,
                last_task_pull REAL,
                last_task_received REAL,
                # cooldown_until REAL,  # DEPRECATED: Replaced by validator_enforced_cooldown_until and miner_cooldown_until
                total_tasks_pulled INTEGER DEFAULT 0,
                total_tasks_received INTEGER DEFAULT 0,
                total_tasks_submitted INTEGER DEFAULT 0,
                total_successful_submissions INTEGER DEFAULT 0,
                average_score REAL DEFAULT 0.0,
                is_active BOOLEAN DEFAULT TRUE,
                updated_at REAL DEFAULT (strftime('%s', 'now'))
            )
        ''')
        
        # Recent prompts table for deduplication
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS recent_prompts (
                prompt_hash TEXT NOT NULL,
                validator_uid INTEGER NOT NULL,
                prompt TEXT NOT NULL,
                pulled_at REAL NOT NULL,
                PRIMARY KEY (prompt_hash, validator_uid)
            )
        ''')
        
        # Statistics table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS statistics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp REAL NOT NULL,
                total_tasks_pulled INTEGER DEFAULT 0,
                total_tasks_processed INTEGER DEFAULT 0,
                total_successful_generations INTEGER DEFAULT 0,
                total_successful_validations INTEGER DEFAULT 0,
                total_successful_submissions INTEGER DEFAULT 0,
                average_generation_time REAL DEFAULT 0.0,
                average_validation_time REAL DEFAULT 0.0,
                average_local_score REAL DEFAULT 0.0,
                average_feedback_score REAL DEFAULT 0.0,
                total_rewards REAL DEFAULT 0.0,
                uptime_hours REAL DEFAULT 0.0
            )
        ''')
        
        # Create indexes
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_prompt_hash ON tasks(prompt_hash)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_validator_uid ON tasks(validator_uid)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_pulled_at ON tasks(pulled_at)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_recent_prompts_time ON recent_prompts(pulled_at)')
        
        # Create indexes for shared task tracking
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_shared_task_status ON shared_task_tracking(status)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_shared_task_validator ON shared_task_tracking(validator_uid, status)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_shared_task_timeout ON shared_task_tracking(timeout_at)')
        
        conn.commit()
        conn.close()
    
    def is_duplicate_prompt(self, prompt: str, validator_uid: int, hours_window: int = 24) -> bool:
        """Check if this prompt was recently processed successfully from this validator"""
        prompt_hash = hashlib.sha256(prompt.encode()).hexdigest()
        cutoff_time = time.time() - (hours_window * 3600)
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Check if we have a successful submission for this prompt from this validator recently
        cursor.execute('''
            SELECT COUNT(*) FROM tasks 
            WHERE prompt_hash = ? AND validator_uid = ? AND pulled_at > ? 
            AND submission_success = 1 AND feedback_received = 1
        ''', (prompt_hash, validator_uid, cutoff_time))
        
        successful_submissions = cursor.fetchone()[0]
        
        # Also check for any recent attempts (successful or not) but with shorter window
        recent_cutoff = time.time() - (1 * 3600)  # 1 hour for failed attempts (more forgiving)
        cursor.execute('''
            SELECT COUNT(*) FROM tasks 
            WHERE prompt_hash = ? AND validator_uid = ? AND pulled_at > ?
        ''', (prompt_hash, validator_uid, recent_cutoff))
        
        recent_attempts = cursor.fetchone()[0]
        
        conn.close()
        
        # Don't duplicate if we successfully submitted recently (24 hour window)
        if successful_submissions > 0:
            return True
        
        # Allow retry after 1 hour if previous attempts failed (more aggressive retry)
        if recent_attempts > 0:
            return True
            
        return False
    
    def add_recent_prompt(self, prompt: str, validator_uid: int):
        """Add prompt to recent prompts tracking"""
        prompt_hash = hashlib.sha256(prompt.encode()).hexdigest()
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT OR REPLACE INTO recent_prompts 
            (prompt_hash, validator_uid, prompt, pulled_at)
            VALUES (?, ?, ?, ?)
        ''', (prompt_hash, validator_uid, prompt, time.time()))
        
        conn.commit()
        conn.close()
    
    def save_task(self, task: TaskRecord):
        """Save task record to database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT OR REPLACE INTO tasks 
            (task_id, prompt, prompt_hash, validator_uid, validator_hotkey, validator_stake,
             validation_threshold, pulled_at, processed_at, submitted_at, generation_time,
             validation_time, total_processing_time, local_validation_score, submission_success, feedback_received,
             task_fidelity_score, average_fidelity_score, current_miner_reward,
             validation_failed, generations_in_window, ply_file_path, compressed_file_path)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            task.task_id, task.prompt, task.prompt_hash, task.validator_uid,
            task.validator_hotkey, task.validator_stake, task.validation_threshold,
            task.pulled_at, task.processed_at, task.submitted_at, task.generation_time,
            task.validation_time, task.total_processing_time, task.local_validation_score, task.submission_success,
            task.feedback_received, task.task_fidelity_score, task.average_fidelity_score,
            task.current_miner_reward, task.validation_failed, task.generations_in_window,
            task.ply_file_path, task.compressed_file_path
        ))
        
        conn.commit()
        conn.close()
    
    def get_recent_unvalidated_tasks(self, hours: int = 2) -> List[TaskRecord]:
        """Get recent tasks that haven't been locally validated"""
        cutoff_time = time.time() - (hours * 3600)
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT * FROM tasks 
            WHERE processed_at > ? AND local_validation_score IS NULL
            ORDER BY processed_at DESC
        ''', (cutoff_time,))
        
        rows = cursor.fetchall()
        conn.close()
        
        tasks = []
        for row in rows:
            task = TaskRecord(
                task_id=row[0], prompt=row[1], prompt_hash=row[2],
                validator_uid=row[3], validator_hotkey=row[4], validator_stake=row[5],
                validation_threshold=row[6], pulled_at=row[7], processed_at=row[8],
                submitted_at=row[9], generation_time=row[10], validation_time=row[11],
                total_processing_time=row[12], local_validation_score=row[13], submission_success=bool(row[14]),
                feedback_received=bool(row[15]), task_fidelity_score=row[16],
                average_fidelity_score=row[17], current_miner_reward=row[18],
                validation_failed=bool(row[19]) if row[19] is not None else None,
                generations_in_window=row[20], ply_file_path=row[21],
                compressed_file_path=row[22]
            )
            tasks.append(task)
        
        return tasks
    
    def get_unfinished_tasks(self, hours: int = 24) -> List[TaskRecord]:
        """Get tasks that were pulled but never completed successfully"""
        cutoff_time = time.time() - (hours * 3600)
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT * FROM tasks 
            WHERE pulled_at > ? AND (
                submission_success = 0 OR 
                feedback_received = 0 OR 
                processed_at IS NULL
            )
            ORDER BY pulled_at DESC
        ''', (cutoff_time,))
        
        rows = cursor.fetchall()
        conn.close()
        
        tasks = []
        for row in rows:
            task = TaskRecord(
                task_id=row[0], prompt=row[1], prompt_hash=row[2],
                validator_uid=row[3], validator_hotkey=row[4], validator_stake=row[5],
                validation_threshold=row[6], pulled_at=row[7], processed_at=row[8],
                submitted_at=row[9], generation_time=row[10], validation_time=row[11],
                total_processing_time=row[12], local_validation_score=row[13], submission_success=bool(row[14]),
                feedback_received=bool(row[15]), task_fidelity_score=row[16],
                average_fidelity_score=row[17], current_miner_reward=row[18],
                validation_failed=bool(row[19]) if row[19] is not None else None,
                generations_in_window=row[20], ply_file_path=row[21],
                compressed_file_path=row[22]
            )
            tasks.append(task)
        
        return tasks
    
    def get_duplicate_analysis(self, validator_uid: int, hours: int = 24) -> Dict[str, Any]:
        """Analyze duplicate checking for a specific validator"""
        cutoff_time = time.time() - (hours * 3600)
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Get all tasks from this validator in the time window
        cursor.execute('''
            SELECT prompt, prompt_hash, pulled_at, processed_at, submission_success, 
                   feedback_received, task_fidelity_score 
            FROM tasks 
            WHERE validator_uid = ? AND pulled_at > ?
            ORDER BY pulled_at DESC
        ''', (validator_uid, cutoff_time))
        
        tasks = cursor.fetchall()
        
        # Get recent prompts tracking
        cursor.execute('''
            SELECT prompt_hash, pulled_at FROM recent_prompts 
            WHERE validator_uid = ? AND pulled_at > ?
            ORDER BY pulled_at DESC
        ''', (validator_uid, cutoff_time))
        
        recent_prompts = cursor.fetchall()
        
        conn.close()
        
        analysis = {
            'validator_uid': validator_uid,
            'total_tasks_pulled': len(tasks),
            'successful_tasks': len([t for t in tasks if t[4] and t[5]]),  # submission_success and feedback_received
            'failed_tasks': len([t for t in tasks if not t[4] or not t[5]]),
            'unprocessed_tasks': len([t for t in tasks if t[3] is None]),  # processed_at is None
            'recent_prompts_tracked': len(recent_prompts),
            'unique_prompts': len(set(t[1] for t in tasks)),  # unique prompt_hashes
            'tasks': [
                {
                    'prompt': t[0][:50] + '...' if len(t[0]) > 50 else t[0],
                    'prompt_hash': t[1][:12],
                    'pulled_at': t[2],
                    'processed': t[3] is not None,
                    'submitted': t[4],
                    'feedback': t[5],
                    'score': t[6]
                }
                for t in tasks[-10:]  # Last 10 tasks
            ]
        }
        
        return analysis
    
    def cleanup_old_prompts(self, days: int = 7):
        """Clean up old prompt records and failed tasks"""
        cutoff_time = time.time() - (days * 24 * 3600)
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Clean up old recent_prompts
        cursor.execute('DELETE FROM recent_prompts WHERE pulled_at < ?', (cutoff_time,))
        deleted_prompts = cursor.rowcount
        
        # Clean up old failed tasks (keep successful ones longer)
        cursor.execute('''
            DELETE FROM tasks WHERE pulled_at < ? AND (
                submission_success = 0 OR 
                feedback_received = 0 OR 
                processed_at IS NULL
            )
        ''', (cutoff_time,))
        deleted_tasks = cursor.rowcount
        
        conn.commit()
        conn.close()
        
        logger.info(f"🧹 Cleaned up {deleted_prompts} old prompt records and {deleted_tasks} failed tasks")
    
    # ===== SHARED TASK TRACKING METHODS =====
    # These methods prevent duplicate task processing across multiple mining instances
    
    def acquire_task_lock(self, task_id: str, validator_uid: int, instance_id: str, timeout_minutes: int = 2) -> bool:
        """
        Try to acquire a lock on a task to prevent other instances from processing it.
        Returns True if lock was acquired, False if task is already being processed.
        """
        import socket
        import os
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            # Check if task is already being processed
            cursor.execute('''
                SELECT status, instance_id, started_at, timeout_at 
                FROM shared_task_tracking 
                WHERE task_id = ?
            ''', (task_id,))
            
            existing = cursor.fetchone()
            
            if existing:
                status, existing_instance_id, started_at, timeout_at = existing
                
                # If task is completed, allow reprocessing
                if status == 'completed':
                    cursor.execute('DELETE FROM shared_task_tracking WHERE task_id = ?', (task_id,))
                    conn.commit()
                # If task is in progress but timed out, allow takeover
                elif status == 'in_progress' and timeout_at and time.time() > time.mktime(time.strptime(timeout_at, '%Y-%m-%d %H:%M:%S')):
                    cursor.execute('DELETE FROM shared_task_tracking WHERE task_id = ?', (task_id,))
                    conn.commit()
                # If task is actively being processed by another instance, deny lock
                elif status == 'in_progress':
                    conn.close()
                    return False
            
            # Calculate timeout time
            timeout_time = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time() + (timeout_minutes * 60)))
            
            # Acquire lock
            cursor.execute('''
                INSERT OR REPLACE INTO shared_task_tracking 
                (task_id, validator_uid, status, instance_id, started_at, timeout_at, instance_hostname, instance_pid)
                VALUES (?, ?, 'in_progress', ?, CURRENT_TIMESTAMP, ?, ?, ?)
            ''', (task_id, validator_uid, instance_id, timeout_time, socket.gethostname(), os.getpid()))
            
            conn.commit()
            conn.close()
            return True
            
        except Exception as e:
            conn.close()
            print(f"Error acquiring task lock: {e}")
            return False
    
    def release_task_lock(self, task_id: str, instance_id: str, status: str = 'completed'):
        """
        Release the lock on a task and mark it as completed or failed.
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            cursor.execute('''
                UPDATE shared_task_tracking 
                SET status = ?, completed_at = CURRENT_TIMESTAMP
                WHERE task_id = ? AND instance_id = ?
            ''', (status, task_id, instance_id))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            conn.close()
            print(f"Error releasing task lock: {e}")
    
    def is_validator_busy(self, validator_uid: int, exclude_instance_id: str = None) -> bool:
        """
        Check if a validator is currently busy processing tasks.
        Returns True if validator has active tasks, False if available.
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            # Count active tasks for this validator
            if exclude_instance_id:
                cursor.execute('''
                    SELECT COUNT(*) FROM shared_task_tracking 
                    WHERE validator_uid = ? AND status = 'in_progress' AND instance_id != ?
                ''', (validator_uid, exclude_instance_id))
            else:
                cursor.execute('''
                    SELECT COUNT(*) FROM shared_task_tracking 
                    WHERE validator_uid = ? AND status = 'in_progress'
                ''', (validator_uid,))
            
            active_tasks = cursor.fetchone()[0]
            conn.close()
            
            return active_tasks > 0
            
        except Exception as e:
            conn.close()
            print(f"Error checking validator busy status: {e}")
            return False
    
    def cleanup_expired_locks(self, timeout_minutes: int = 2):
        """
        Clean up expired task locks that are older than the timeout period.
        This allows other instances to take over stalled tasks.
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            # Find expired locks
            timeout_time = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time() - (timeout_minutes * 60)))
            
            cursor.execute('''
                SELECT task_id, instance_id, started_at 
                FROM shared_task_tracking 
                WHERE status = 'in_progress' AND started_at < ?
            ''', (timeout_time,))
            
            expired_locks = cursor.fetchall()
            
            if expired_locks:
                print(f"🧹 Cleaning up {len(expired_locks)} expired task locks...")
                
                for task_id, instance_id, started_at in expired_locks:
                    print(f"   Expired: {task_id} (instance: {instance_id}, started: {started_at})")
                
                # Remove expired locks
                cursor.execute('''
                    DELETE FROM shared_task_tracking 
                    WHERE status = 'in_progress' AND started_at < ?
                ''', (timeout_time,))
                
                conn.commit()
                print(f"✅ Cleaned up {len(expired_locks)} expired locks")
            
            conn.close()
            
        except Exception as e:
            conn.close()
            print(f"Error cleaning up expired locks: {e}")
    
    def get_available_validators(self, exclude_instance_id: str = None) -> List[int]:
        """
        Get list of validator UIDs that are not currently busy processing tasks.
        This helps distribute work across validators and instances.
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            # Get all validators that don't have active tasks
            if exclude_instance_id:
                cursor.execute('''
                    SELECT DISTINCT v.uid 
                    FROM validators v
                    LEFT JOIN shared_task_tracking st ON v.uid = st.validator_uid AND st.status = 'in_progress'
                    WHERE v.is_active = 1 
                    AND (st.validator_uid IS NULL OR st.instance_id = ?)
                    ORDER BY v.stake DESC
                ''', (exclude_instance_id,))
            else:
                cursor.execute('''
                    SELECT DISTINCT v.uid 
                    FROM validators v
                    LEFT JOIN shared_task_tracking st ON v.uid = st.validator_uid AND st.status = 'in_progress'
                    WHERE v.is_active = 1 
                    AND st.validator_uid IS NULL
                    ORDER BY v.stake DESC
                ''')
            
            available_uids = [row[0] for row in cursor.fetchall()]
            conn.close()
            
            return available_uids
            
        except Exception as e:
            conn.close()
            print(f"Error getting available validators: {e}")
            return []
    
    def get_task_processing_stats(self) -> dict:
        """
        Get statistics about task processing across all instances.
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            # Count tasks by status
            cursor.execute('''
                SELECT status, COUNT(*) as count
                FROM shared_task_tracking
                GROUP BY status
            ''')
            
            status_counts = dict(cursor.fetchall())
            
            # Count tasks by instance
            cursor.execute('''
                SELECT instance_id, COUNT(*) as count
                FROM shared_task_tracking
                WHERE status = 'in_progress'
                GROUP BY instance_id
            ''')
            
            instance_counts = dict(cursor.fetchall())
            
            # Count tasks by validator
            cursor.execute('''
                SELECT validator_uid, COUNT(*) as count
                FROM shared_task_tracking
                WHERE status = 'in_progress'
                GROUP BY validator_uid
            ''')
            
            validator_counts = dict(cursor.fetchall())
            
            conn.close()
            
            return {
                'status_counts': status_counts,
                'instance_counts': instance_counts,
                'validator_counts': validator_counts,
                'total_tasks': sum(status_counts.values()),
                'active_tasks': status_counts.get('in_progress', 0),
                'completed_tasks': status_counts.get('completed', 0)
            }
            
        except Exception as e:
            conn.close()
            print(f"Error getting task processing stats: {e}")
            return {}

class FidelityScoreTracker:
    """
    Tracks task fidelity scores and automatically switches generation endpoints
    when consecutive 0.0 scores are detected to prevent further failures.
    """
    
    def __init__(self, history_size: int = 5, zero_threshold: int = 2, 
                 fallback_endpoint: str = "/generate_3d_from_prompt_grid_flow/"):
        """
        Initialize the fidelity score tracker.
        
        Args:
            history_size: Number of recent tasks to keep in history queue
            zero_threshold: Number of consecutive 0.0 scores to trigger endpoint switch
            fallback_endpoint: Endpoint to use when switching due to 0.0 scores
        """
        self.history_size = history_size
        self.zero_threshold = zero_threshold
        self.fallback_endpoint = fallback_endpoint
        
        # Per-validator tracking
        self.validator_histories: Dict[int, List[Dict[str, Any]]] = {}
        self.validator_endpoint_states: Dict[int, Dict[str, Any]] = {}
        
        # Global tracking
        self.global_history: List[Dict[str, Any]] = []
        self.global_endpoint_state = {
            'current_endpoint': None,
            'original_endpoint': None,
            'switched_at': None,
            'switch_reason': None,
            'consecutive_zeros': 0,
            'total_zeros': 0,
            'switches_made': 0
        }
        
        self.logger = logging.getLogger(__name__)
        self.logger.info(f"🎯 FidelityScoreTracker initialized")
        self.logger.info(f"   History size: {history_size}")
        self.logger.info(f"   Zero threshold: {zero_threshold}")
        self.logger.info(f"   Fallback endpoint: {fallback_endpoint}")
    
    def record_task_result(self, task_id: str, validator_uid: int, 
                          task_fidelity_score: float, endpoint: str, 
                          prompt: str = None) -> Dict[str, Any]:
        """
        Record a task result and determine if endpoint switching is needed.
        
        Args:
            task_id: Unique task identifier
            validator_uid: Validator UID
            task_fidelity_score: Fidelity score from the task
            endpoint: Endpoint used for generation
            prompt: Task prompt (optional, for logging)
            
        Returns:
            Dict with tracking information and endpoint switching decision
        """
        current_time = time.time()
        
        # Initialize validator tracking if needed
        if validator_uid not in self.validator_histories:
            self.validator_histories[validator_uid] = []
            self.validator_endpoint_states[validator_uid] = {
                'current_endpoint': endpoint,
                'original_endpoint': endpoint,
                'switched_at': None,
                'switch_reason': None,
                'consecutive_zeros': 0,
                'total_zeros': 0,
                'switches_made': 0,
                'last_switch_time': None
            }
        
        # Initialize global tracking if needed
        if not self.global_endpoint_state['current_endpoint']:
            self.global_endpoint_state['current_endpoint'] = endpoint
            self.global_endpoint_state['original_endpoint'] = endpoint
        
        # Create task record
        task_record = {
            'task_id': task_id,
            'validator_uid': validator_uid,
            'fidelity_score': task_fidelity_score,
            'endpoint': endpoint,
            'prompt': prompt,
            'timestamp': current_time,
            'is_zero_score': task_fidelity_score == 0.0
        }
        
        # Update validator history
        validator_history = self.validator_histories[validator_uid]
        validator_history.append(task_record)
        
        # Keep only recent history
        if len(validator_history) > self.history_size:
            validator_history.pop(0)
        
        # Update global history
        self.global_history.append(task_record)
        if len(self.global_history) > self.history_size:
            self.global_history.pop(0)
        
        # Update validator statistics
        validator_state = self.validator_endpoint_states[validator_uid]
        if task_fidelity_score == 0.0:
            validator_state['consecutive_zeros'] += 1
            validator_state['total_zeros'] += 1
        else:
            validator_state['consecutive_zeros'] = 0
        
        # Update global statistics
        if task_fidelity_score == 0.0:
            self.global_endpoint_state['consecutive_zeros'] += 1
            self.global_endpoint_state['total_zeros'] += 1
        else:
            self.global_endpoint_state['consecutive_zeros'] = 0
        
        # Check if endpoint switching is needed
        switching_decision = self._evaluate_endpoint_switching(
            validator_uid, endpoint, task_fidelity_score
        )
        
        # Log the tracking information
        self._log_tracking_info(validator_uid, task_record, switching_decision)
        
        return {
            'task_record': task_record,
            'switching_decision': switching_decision,
            'validator_stats': validator_state.copy(),
            'global_stats': self.global_endpoint_state.copy()
        }
    
    def _evaluate_endpoint_switching(self, validator_uid: int, current_endpoint: str, 
                                   fidelity_score: float) -> Dict[str, Any]:
        """
        Evaluate whether endpoint switching is needed based on recent performance.
        
        Args:
            validator_uid: Validator UID
            current_endpoint: Current endpoint being used
            fidelity_score: Current task fidelity score
            
        Returns:
            Dict with switching decision and reasoning
        """
        validator_state = self.validator_endpoint_states[validator_uid]
        global_state = self.global_endpoint_state
        
        # Check if we're already using the fallback endpoint
        using_fallback = (current_endpoint == self.fallback_endpoint)
        
        # Determine if switching is needed
        should_switch_to_fallback = False
        should_switch_back = False
        switch_reason = None
        
        # Check validator-specific switching
        if validator_state['consecutive_zeros'] >= self.zero_threshold:
            if not using_fallback:
                should_switch_to_fallback = True
                switch_reason = f"Validator {validator_uid} had {validator_state['consecutive_zeros']} consecutive 0.0 scores"
        
        # Check global switching (if any validator has issues)
        if global_state['consecutive_zeros'] >= self.zero_threshold:
            if not using_fallback:
                should_switch_to_fallback = True
                switch_reason = f"Global threshold reached: {global_state['consecutive_zeros']} consecutive 0.0 scores"
        
        # Check if we should switch back to original endpoint
        if using_fallback:
            # Switch back if we've had good scores recently
            recent_scores = [r['fidelity_score'] for r in self.global_history[-3:]]  # Last 3 scores
            if recent_scores and all(score > 0.0 for score in recent_scores):
                should_switch_back = True
                switch_reason = "Recent good scores - switching back to original endpoint"
        
        # Make the switching decision
        if should_switch_to_fallback:
            return {
                'should_switch': True,
                'new_endpoint': self.fallback_endpoint,
                'reason': switch_reason,
                'type': 'to_fallback',
                'trigger': 'consecutive_zeros'
            }
        elif should_switch_back:
            # Determine which original endpoint to use
            original_endpoint = validator_state.get('original_endpoint', current_endpoint)
            if original_endpoint == self.fallback_endpoint:
                # If the original was already fallback, use a default
                original_endpoint = '/generate/'
            
            return {
                'should_switch': True,
                'new_endpoint': original_endpoint,
                'reason': switch_reason,
                'type': 'to_original',
                'trigger': 'good_scores'
            }
        else:
            return {
                'should_switch': False,
                'new_endpoint': current_endpoint,
                'reason': "No switching needed",
                'type': 'none',
                'trigger': 'none'
            }
    
    def get_recommended_endpoint(self, validator_uid: int, 
                               requested_endpoint: str) -> str:
        """
        Get the recommended endpoint for a validator, considering recent performance.
        
        Args:
            validator_uid: Validator UID
            requested_endpoint: Endpoint that was originally requested
        
        Returns:
            Recommended endpoint to use
        """
        if validator_uid not in self.validator_endpoint_states:
            return requested_endpoint
        
        validator_state = self.validator_endpoint_states[validator_uid]
        
        # If we're currently using fallback endpoint, continue using it
        if validator_state['current_endpoint'] == self.fallback_endpoint:
            return self.fallback_endpoint
        
        # If we have recent 0.0 scores, recommend fallback
        if validator_state['consecutive_zeros'] >= self.zero_threshold:
            return self.fallback_endpoint
        
        # Otherwise, use the requested endpoint
        return requested_endpoint
    
    def apply_endpoint_switch(self, validator_uid: int, new_endpoint: str, 
                            reason: str) -> bool:
        """
        Apply an endpoint switch for a validator.
        
        Args:
            validator_uid: Validator UID
            new_endpoint: New endpoint to use
            reason: Reason for the switch
        
        Returns:
            True if switch was applied, False otherwise
        """
        if validator_uid not in self.validator_endpoint_states:
            return False
        
        validator_state = self.validator_endpoint_states[validator_uid]
        old_endpoint = validator_state['current_endpoint']
        
        # Update validator state
        validator_state['current_endpoint'] = new_endpoint
        validator_state['switched_at'] = time.time()
        validator_state['switch_reason'] = reason
        validator_state['switches_made'] += 1
        validator_state['last_switch_time'] = time.time()
        
        # Update global state if this is a significant switch
        if new_endpoint == self.fallback_endpoint:
            self.global_endpoint_state['current_endpoint'] = new_endpoint
            self.global_endpoint_state['switched_at'] = time.time()
            self.global_endpoint_state['switch_reason'] = reason
            self.global_endpoint_state['switches_made'] += 1
        
        # Log the switch
        self.logger.info(f"🔄 Endpoint switch for validator {validator_uid}:")
        self.logger.info(f"   Old: {old_endpoint}")
        self.logger.info(f"   New: {new_endpoint}")
        self.logger.info(f"   Reason: {reason}")
        
        return True
    
    def get_tracking_summary(self, validator_uid: int = None) -> Dict[str, Any]:
        """
        Get a summary of tracking information.
        
        Args:
            validator_uid: Specific validator UID, or None for global summary
        
        Returns:
            Summary of tracking information
        """
        if validator_uid:
            if validator_uid not in self.validator_endpoint_states:
                return {}
            
            validator_state = self.validator_endpoint_states[validator_uid]
            validator_history = self.validator_histories.get(validator_uid, [])
            
            return {
                'validator_uid': validator_uid,
                'current_endpoint': validator_state['current_endpoint'],
                'original_endpoint': validator_state['original_endpoint'],
                'consecutive_zeros': validator_state['consecutive_zeros'],
                'total_zeros': validator_state['total_zeros'],
                'switches_made': validator_state['switches_made'],
                'last_switch_time': validator_state['last_switch_time'],
                'recent_scores': [r['fidelity_score'] for r in validator_history[-5:]],
                'history_size': len(validator_history)
            }
        else:
            # Global summary
            return {
                'global_endpoint_state': self.global_endpoint_state.copy(),
                'total_validators_tracked': len(self.validator_histories),
                'total_tasks_tracked': len(self.global_history),
                'recent_global_scores': [r['fidelity_score'] for r in self.global_history[-5:]]
            }
    
    def _log_tracking_info(self, validator_uid: int, task_record: Dict[str, Any], 
                          switching_decision: Dict[str, Any]):
        """Log tracking information for debugging and monitoring."""
        fidelity_score = task_record['fidelity_score']
        endpoint = task_record['endpoint']
        
        # Log basic task result
        if fidelity_score == 0.0:
            self.logger.warning(f"⚠️ Zero fidelity score detected:")
            self.logger.warning(f"   Validator: {validator_uid}")
            self.logger.warning(f"   Task: {task_record['task_id']}")
            self.logger.warning(f"   Endpoint: {endpoint}")
            if task_record.get('prompt'):
                self.logger.warning(f"   Prompt: '{task_record['prompt'][:50]}...'")
        else:
            self.logger.debug(f"✅ Good fidelity score: {fidelity_score:.4f} for validator {validator_uid}")
        
        # Log switching decision
        if switching_decision['should_switch']:
            self.logger.info(f"🔄 Endpoint switching recommended:")
            self.logger.info(f"   Validator: {validator_uid}")
            self.logger.info(f"   New endpoint: {switching_decision['new_endpoint']}")
            self.logger.info(f"   Reason: {switching_decision['reason']}")
            self.logger.info(f"   Type: {switching_decision['type']}")
    
    def reset_validator_tracking(self, validator_uid: int):
        """Reset tracking for a specific validator."""
        if validator_uid in self.validator_histories:
            del self.validator_histories[validator_uid]
        if validator_uid in self.validator_endpoint_states:
            del self.validator_endpoint_states[validator_uid]
        
        self.logger.info(f"🔄 Reset tracking for validator {validator_uid}")
    
    def cleanup_old_history(self, max_age_hours: float = 24):
        """Clean up old history entries."""
        current_time = time.time()
        cutoff_time = current_time - (max_age_hours * 3600)
        
        # Clean global history
        original_size = len(self.global_history)
        self.global_history = [r for r in self.global_history if r['timestamp'] > cutoff_time]
        cleaned_global = original_size - len(self.global_history)
        
        # Clean validator histories
        total_cleaned_validator = 0
        for validator_uid, history in self.validator_histories.items():
            original_size = len(history)
            self.validator_histories[validator_uid] = [r for r in history if r['timestamp'] > cutoff_time]
            cleaned = original_size - len(self.validator_histories[validator_uid])
            total_cleaned_validator += cleaned
        
        if cleaned_global > 0 or total_cleaned_validator > 0:
            self.logger.info(f"🧹 Cleaned up old history:")
            self.logger.info(f"   Global: {cleaned_global} entries")
            self.logger.info(f"   Validators: {total_cleaned_validator} entries")
            self.logger.info(f"   Cutoff: {max_age_hours:.1f} hours ago")

class ContinuousTrellisOrchestrator:
    """Continuous TRELLIS orchestrator with intelligent features"""
    
    def __init__(self, config: Dict[str, Any]):
        # Merge with default config
        self.config = self._get_default_config()
        self.config.update(config)
        
        self.logger = logger
        
        # Setup output directory
        self.output_dir = Path(self.config['output_dir'])
        self.output_dir.mkdir(exist_ok=True)
        
        # Initialize database
        self.db = TaskDatabase()
        
        # Generate unique instance ID for shared task tracking
        self.instance_id = f"{socket.gethostname()}_{os.getpid()}_{uuid.uuid4().hex[:8]}"
        self.logger.info(f"🆔 Instance ID: {self.instance_id}")
        
        # Bittensor components
        self.wallet = None
        self.subtensor = None
        self.dendrite = None
        self.metagraph = None
        
        # State management
        self.validators: Dict[int, ValidatorState] = {}
        self.running = False
        self.start_time = time.time()
        
        # Initialize organic LoRA router
        if ORGANIC_LORA_ROUTER_AVAILABLE:
            self.lora_router = FinalOrganicRouter()
            self.logger.info("🧠 Initialized organic LoRA router with pattern learning (100% core accuracy)")
        else:
            self.lora_router = None
            self.logger.info("⚠️ Organic LoRA router not available - using default model")
        
        # Initialize prompt optimizer
        if OPTIMIZED_PROMPT_OPTIMIZER_AVAILABLE:
            # self.prompt_optimizer = OptimizedPromptOptimizer("rl_checkpoints_v3/prompt_score_log.csv")
            self.prompt_optimizer = LLMPromptOptimizer(
                ollama_url=self.config.get('ollama_url', 'http://localhost:11434'),
                model="llama3.2:3b",
                use_vllm=self.config.get('use_vllm', False),
                vllm_url=self.config.get('vllm_url', 'http://localhost:9000'),
                vllm_model=self.config.get('vllm_model', 'llama-3-2-3b-it')
            )
            self.logger.info("🚀 Initialized performance-optimized prompt optimizer")
        else:
            try:
                self.prompt_optimizer = TrellisPromptOptimizer()
                self.logger.info("🔧 Initialized standard prompt optimizer")
            except Exception as e:
                self.logger.info(f"❌ Failed to initialize standard prompt optimizer: {e}")
                self.prompt_optimizer = None
        
        # Initialize reproducibility system
        if REPRODUCIBILITY_SYSTEM_AVAILABLE:
            try:
                gold_standard_results = self.get_gold_prompts_from_orchestrator(log_count=15)
                # import pdb; pdb.set_trace()
                if not gold_standard_results:
                    print(f"❌ No gold prompts found from logs")
                    # return {"error": "No gold prompts found from logs"}
                
                print(f"✅ Gold prompts loaded: {len(gold_standard_results)} prompts")
                
            except Exception as e:
                print(f"❌ Failed to get gold prompts: {e}")
                # return {"error": f"Gold prompt loading failed: {e}"}

            self.reproducibility_system = LLMClosePromptReproducibility(
                episodic_memory_file="episodic_logs_usa/episodic_memory.json",
                use_vllm=self.config.get('use_vllm', False),
                vllm_url=self.config.get('vllm_url', 'http://localhost:9000'),
                vllm_model=self.config.get('vllm_model', 'llama-3-2-3b-it'),
                ollama_url=self.config.get('ollama_url', 'http://localhost:11434')
            )
            self.logger.info("🔄 Initialized reproducibility system for pre-optimization")
            self.reproducibility_system.gold_standard_results = self.reproducibility_system._load_episodic_memory()
            # self.reproducibility_system.gold_standard_results = gold_standard_results
        
            print(f"✅ Reproducibility system initialized with {len(gold_standard_results)} gold prompts")
            print(f"   Using EXACT same gold prompts as orchestrator")
            
            # Track when we last reloaded gold prompts
            self.last_gold_prompts_reload = time.time()
            self.gold_prompts_reload_interval = self.config.get('gold_prompts_reload_interval', 3600)  # 1 hour default
            self.logger.info(f"   📚 Gold prompts will reload every {self.gold_prompts_reload_interval/3600:.1f} hours")
            self.logger.debug(f"   🔧 Raw reload interval value: {self.gold_prompts_reload_interval} seconds")
        else:
            self.reproducibility_system = None
            self.logger.info("⚠️ Reproducibility system not available")
        
        # Priority server coordinator
        self.priority_coordinator = PriorityServerCoordinator(
            server_url=self.config.get('generation_server_url', 'http://localhost:8096'),
            max_wait_time_seconds=self.config.get('priority_access_max_wait', 60),
            status_check_interval=self.config.get('priority_access_check_interval', 1),
            priority_timeout=self.config.get('priority_access_timeout', 30),
            on_interruption_callback=self._on_priority_interruption
        )
        
        # Initialize fidelity score tracker for endpoint switching
        self.fidelity_tracker = FidelityScoreTracker(
            history_size=self.config.get('fidelity_tracker_history_size', 5),
            zero_threshold=self.config.get('fidelity_tracker_zero_threshold', 2),
            fallback_endpoint=self.config.get('fidelity_tracker_fallback_endpoint', "/generate_3d_from_prompt_grid_flow/")
        )
        self.logger.info("🎯 Initialized fidelity score tracker for automatic endpoint switching")
        
        # Statistics
        self.stats = {
            'session_start': time.time(),
            'tasks_pulled': 0,
            'tasks_processed': 0,
            'successful_generations': 0,
            'successful_validations': 0,
            'successful_submissions': 0,
            'total_generation_time': 0.0,
            'total_validation_time': 0.0,
            'total_processing_time': 0.0,
            'total_rewards': 0.0,
            'idle_validations': 0,
            'prompts_optimized': 0,
            'reproducibility_optimizations': 0,
            'traditional_optimizations': 0,
            'optimization_improvements': 0,
            'prompts_cleaned': 0,  # Track how many prompts were cleaned of artifacts
            'priority_access_timeouts': 0,  # Track priority access timeouts
            'priority_interruptions': 0,    # Track when we interrupt other jobs
            'server_unavailable_skips': 0,  # Track when we skip task pulls due to server unavailability
            'server_status_check_errors': 0, # Track server status check errors
            'lora_routing_decisions': 0,    # Track LoRA routing decisions
            'lora_routing_accuracy': 0.0,   # Track LoRA routing accuracy
            'blacklisted_validators_skipped': 0, # Track blacklisted validator skips
            'gold_prompts_reloaded': 0, # Track how many times gold prompts were reloaded
            'gold_prompts_available': 0, # Track current number of gold prompts available
            
            # Enhanced cooldown system statistics
            'cooldown_violations_total': 0,  # Total cooldown violations across all validators
            # DEPRECATED: Validation lock stats removed - now using MIN_TASK_INTERVAL constant for rate limiting
            # 'validation_locks_applied': 0,   # Total validation locks applied
            'enhanced_cooldown_penalties': 0, # Total enhanced cooldown penalties applied
            
            # Emergency cooldown management statistics
            'emergency_cooldowns_applied': 0,  # Total emergency cooldowns applied
            'critical_violations_handled': 0,  # Total critical violations handled
            'critical_violations_detected': 0,  # Total critical violations detected in real-time
            'validators_temporarily_blacklisted': 0,  # Total validators temporarily blacklisted
            'validators_reset_from_emergency': 0,  # Total validators reset from emergency restrictions
            'dynamic_cooldown_scaling': 0,  # Total times dynamic cooldown scaling was applied
            'dynamic_buffer_applied': 0,  # Total times dynamic buffer was applied
            'emergency_state_recoveries': 0,
            # State persistence statistics
            'validators_restored_from_disk': 0,  # Total validators restored from disk
            'violations_restored_from_disk': 0,  # Total validators with violations restored
            'blacklists_restored_from_disk': 0,  # Total blacklisted validators restored
            'validator_states_saved': 0,  # Total times states were saved to disk
            'validator_state_save_failures': 0,  # Total state save failures
            # New statistics for real-time learning
            'log_parsed_prompts': 0,  # Track prompts parsed from logs
            'enhanced_gold_prompts_available': 0,  # Track enhanced gold prompts (memory + logs)
            'enhanced_gold_prompts_reloaded': 0,  # Track enhanced reloads
            'total_gold_prompts_available': 0,  # Track total available gold prompts
            'memory_prompts': 0,  # Track prompts from episodic memory
            'log_prompts': 0,  # Track prompts from recent logs
            
            # NEW: vLLM optimization statistics
            'vllm_optimizations': 0,
            'vllm_system_chat_success': 0,
            'vllm_system_completions_success': 0,
            'vllm_no_system_success': 0,
            'vllm_failures': 0,
            'vllm_connection_tests': 0,
            'vllm_connection_success': 0,
            
            # Fidelity score tracking statistics
            'fidelity_tracker_endpoint_switches': 0,  # Total endpoint switches made by fidelity tracker
            'fidelity_tracker_zero_scores_detected': 0,  # Total 0.0 scores detected
            'fidelity_tracker_fallback_endpoint_usage': 0,  # Times fallback endpoint was used
            'fidelity_tracker_original_endpoint_recovery': 0,  # Times switched back to original endpoint
        }
        
        # Dynamic system management attributes
        self.current_task_pull_strategy = "AGGRESSIVE"  # Default strategy
        self.current_max_concurrent_tasks = self.config.get('max_concurrent_tasks', 5)  # Default max tasks
        
        # State persistence system
        self.state_persistence = ValidatorStatePersistence(
            state_file=self.config.get('validator_state_file', 'validator_states.json')
        )
        
        # Register shutdown handlers for state persistence
        self._register_shutdown_handlers()
        
        '''
        # Preload CLIP model for faster inference
        self.clip_analyzer = None
        try:
            from clip_alignment_with_generation import CLIPAlignmentWithGeneration
            self.logger.info("🔧 Preloading CLIP model for faster inference...")
            self.clip_analyzer = CLIPAlignmentWithGeneration()
            self.clip_analyzer.load_clip_model()
            self.logger.info("✅ CLIP model preloaded successfully")
        except Exception as e:
            self.logger.warning(f"⚠️ CLIP model preloading failed: {e}")
            self.clip_analyzer = None
        '''
        self.logger.info("🎯 Continuous TRELLIS Orchestrator initialized")
        self.logger.info(f"   Output directory: {self.output_dir}")
        self.logger.info(f"   Generation server: {self.config['generation_server_url']}")
        self.logger.info(f"   Validation server: {self.config['validation_server_url']}")
        
        # Log vLLM optimization configuration
        if self.config.get('use_vllm_optim', False):
            self.logger.info(f"🚀 vLLM Optimization: ENABLED on port {self.config.get('vllm_optim_port', 11300)}")
            if self.config.get('use_system_prompt', False):
                self.logger.info(f"📝 System Prompts: ENABLED")
            else:
                self.logger.info(f"📝 System Prompts: DISABLED")
            priority = self.config.get('vllm_optimization_priority', 'system_chat')
            self.logger.info(f"🎯 vLLM Optimization Priority: {priority}")
        else:
            self.logger.info(f"🔧 vLLM Optimization: DISABLED (using original prompts)")
        
        # Log LoRA configuration
        if self.config.get('enable_lora_routing', True):
            self.logger.info(f"🎯 LoRA Routing: ENABLED (confidence threshold: {self.config.get('lora_routing_confidence_threshold', 0.5)})")
        else:
            self.logger.info(f"🎯 LoRA Routing: DISABLED")
        
        default_lora = self.config.get('default_lora', 'cinema')
        self.logger.info(f"🎨 Default LoRA: {default_lora}")
        
        # Initialize gold prompts count and setup real-time learning if enabled
        if REPRODUCIBILITY_SYSTEM_AVAILABLE and self.reproducibility_system:
            self.stats['gold_prompts_available'] = len(self.reproducibility_system.gold_standard_results)
            self.logger.info(f"📊 Initial gold prompts loaded: {self.stats['gold_prompts_available']}")
            
                    # Setup real-time learning if enabled
        if self.config.get('activate_learning', False):
            if self.config.get('only_log_learning', False):
                log_count = self.config.get('log_learning_count', 6)
                if log_count == -1:
                    log_info = "all available logs"
                else:
                    log_info = f"most recent {log_count} logs"
                
                self.logger.info(f"🚀 ONLY-LOG-LEARNING ENABLED - using {log_info}")
                self.logger.info("   📖 Will parse recent episode logs for fresh learning")
                self.logger.info("   📁 Episodic memory: BYPASSED")
                self.logger.info(f"   🔄 Will use {log_info} exclusively for optimization")
            else:
                self.logger.info("🚀 Real-time learning ENABLED - setting up enhanced gold prompts system")
                self.logger.info("   📖 Will parse recent episode logs for fresh learning")
                self.logger.info("   📁 Will monitor episodic memory for live updates")
                self.logger.info("   🔄 Will combine memory + log data for comprehensive coverage")
            
            # Setup live monitoring (only if not in only-log-learning mode)
            if not self.config.get('only_log_learning', False):
                self.setup_live_episodic_memory_monitoring()
            
            # Initial enhanced reload (after stats are initialized)
            self.enhanced_reload_gold_prompts()
        else:
            self.logger.info("📚 Real-time learning DISABLED - using standard episodic memory only")
        
        # Log task tracking status
        if self.config.get('enable_task_tracking', True) and not self.config.get('disable_task_tracking', False):
            self.logger.info(f"🔄 Shared Task Tracking: ENABLED (Instance ID: {self.instance_id})")
        else:
            self.logger.info(f"🔄 Shared Task Tracking: DISABLED")
        
        # Log duplicate checking status
        if self.config.get('enable_duplicate_checking', True):
            self.logger.info(f"🔄 Duplicate Checking: ENABLED (will skip previously processed prompts)")
        else:
            self.logger.info(f"🔄 Duplicate Checking: DISABLED (will process all prompts including duplicates)")
        
        # Print LLM provider information prominently
        print("\n" + "="*60)
        print("🤖 CONTINUOUS TRELLIS ORCHESTRATOR - LLM PROVIDER CONFIGURATION")
        print("="*60)
        if self.config.get('use_vllm', False):
            print(f"✅ Using vLLM: {self.config.get('vllm_url', 'http://localhost:9000')}")
            print(f"   Model: {self.config.get('vllm_model', 'llama-3-2-3b-it')}")
            print(f"   Status: ACTIVE for prompt optimization")
        else:
            print(f"✅ Using Ollama: {self.config.get('ollama_url', 'http://localhost:11434')}")
            print(f"   Status: ACTIVE for prompt optimization")
        print("="*60)
        
        if self.config.get('use_vllm', False):
            self.logger.info(f"   Using vLLM: {self.config.get('vllm_url', 'http://localhost:9000')} with model {self.config.get('vllm_model', 'llama-3-2-3b-it')}")
        else:
            self.logger.info(f"   Using Ollama: {self.config.get('ollama_url', 'http://localhost:11434')}")
        
        # Log LoRA routing settings
        if ORGANIC_LORA_ROUTER_AVAILABLE:
            self.logger.info(f"🧠 Organic LoRA routing: ENABLED (100% pattern learning accuracy)")
        else:
            self.logger.info(f"🧠 Organic LoRA routing: DISABLED (using default model)")
        
        # Log optimization settings
        if self.config.get('enable_prompt_optimization', True):
            mode = "aggressive" if self.config.get('optimization_aggressive_mode', False) else "standard"
            detail = "minimal" if not self.config.get('log_optimization_details', True) else "detailed"
            cleaning = "ENABLED" if self.config.get('enable_prompt_cleaning', True) else "DISABLED"
            self.logger.info(f"🔧 Prompt optimization: ENABLED ({mode} mode, {detail} logging, cleaning: {cleaning})")
            
            # Log LLM provider for optimization
            if self.config.get('use_vllm', False):
                self.logger.info(f"   🤖 LLM Provider: vLLM ({self.config.get('vllm_model', 'llama-3-2-3b-it')})")
            else:
                self.logger.info(f"   🤖 LLM Provider: Ollama ({self.config.get('ollama_url', 'http://localhost:11434')})")
            
            # Log reproducibility settings
            if self.config.get('enable_reproducibility_optimization', True):
                min_sim = self.config.get('reproducibility_min_similarity', 0.3)
                self.logger.info(f"🔄 Reproducibility optimization: ENABLED (min similarity: {min_sim})")
            else:
                self.logger.info(f"🔄 Reproducibility optimization: DISABLED")
        else:
            self.logger.info(f"🔧 Prompt optimization: DISABLED")
        self.trellis_server_url: str = "http://localhost:8096"

        self.similarity_server = CLIPTextSimilarityServer(verbose=True)
        self.similarity_server.load_clip_model(device="cpu")
        print(f"✅ Instance CLIP model loaded: {self.similarity_server._model is not None}, Tokenizer: {self.similarity_server._tokenizer is not None}")
        cpu_model, cpu_tokenizer, cpu_loading_time, cpu_device = self.similarity_server.load_model_on_device("cpu")
        print(f"🔥 Priming CPU with 5 warm-up runs...")
        result = self.similarity_server.compute_similarity_device(cpu_device, "a photo of a cat", "a photo of a dog", warmup_runs=5, num_runs=0, timer=True)
        print(result)
        self.similarity_device = cpu_device

    def get_clip_analyzer(self):
        """Get the preloaded CLIP analyzer"""
        if self.clip_analyzer is None:
            try:
                from clip_alignment_with_generation import CLIPAlignmentWithGeneration
                self.logger.info("🔧 Loading CLIP model on demand...")
                self.clip_analyzer = CLIPAlignmentWithGeneration()
                self.clip_analyzer.load_clip_model()
                self.logger.info("✅ CLIP model loaded successfully")
            except Exception as e:
                self.logger.error(f"❌ Failed to load CLIP model: {e}")
                return None
        return self.clip_analyzer

    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration"""
        return {
            # Bittensor settings
            'wallet_name': 'test2m3b2',
            'hotkey_name': 't2m3b2',
            'netuid': 17,
            'min_validator_stake': 1000.0,  # Minimum stake required for a validator to be considered
            'min_validator_trust': 0.0,     # Minimum trust score
            'max_validators': 50,           # Maximum number of validators to track
            
            # Validator blacklisting
            'validator_blacklist': [180],   # UIDs to blacklist (e.g., 180 is a WC)
            'enable_validator_blacklisting': True,
            
            # Server settings
            'generation_server_url': 'http://localhost:8096',
            'validation_server_url': 'http://localhost:10006',
            
            # Operation settings
            'harvest_tasks': True,
            'validate_generations': True,
            'submit_results': True,
            'output_dir': './continuous_trellis_outputs',
            'save_intermediate_results': True,
            
            # Timing settings
            # DEPRECATED: Hardcoded task_pull_interval - now using MIN_TASK_INTERVAL constant
            # 'task_pull_interval': 20,  # seconds between validator scans
            'task_pull_interval': MIN_TASK_INTERVAL,  # seconds between validator scans (following _pull_task logic)
            'idle_validation_interval': 300,  # 5 minutes
            'stats_report_interval': 600,  # 10 minutes
            'cleanup_interval': 3600,  # 1 hour
            'duplicate_check_hours': 24,
            
            # Quality settings
            'min_local_score': 0.3,
            'generation_timeout': 300,
            'validation_timeout': 120,
            'submission_timeout': 16,
            
            # Determinism settings
            'use_fixed_seed': True,  # True = always seed 42, False = prompt-hash based seed
            
            # Prompt optimization settings
            'enable_prompt_optimization': True,
            'optimization_aggressive_mode': False,
            'log_optimization_details': True,
            'enable_prompt_cleaning': True,  # Enable automatic prompt cleaning to remove artifacts
            
            # Reproducibility optimization settings
            'enable_reproducibility_optimization': True,
            'reproducibility_min_similarity': 0.3,

            # LoRA routing settings
            'enable_lora_routing': True,  # Enable intelligent LoRA routing
            'lora_routing_confidence_threshold': 0.5,  # Minimum confidence for LoRA routing
            
            # LoRA selection setting
            'default_lora': 'cinema',  # Default LoRA when router is not available

            # Priority access settings
            'priority_access_max_wait': 60, # Max seconds to wait for priority access
            'priority_access_check_interval': 1, # Seconds between status checks
            'priority_access_timeout': 30, # Max seconds to wait for priority access
            
            # Gold prompts reload settings
            'gold_prompts_reload_interval': 3600,  # Reload gold prompts every hour (3600 seconds)
            
            # Duplicate checking settings
            'enable_duplicate_checking': True,  # Enable duplicate prompt checking
            
            # Real-time learning settings
            'activate_learning': False,  # Enable real-time learning from logs and live monitoring
            'log_learning_count': 6,     # Number of recent logs to use for learning (default: 6, -1 for all)
            
            # vLLM settings
            'use_vllm': False,  # Use vLLM instead of Ollama
            'vllm_url': 'http://localhost:11300',  # vLLM server URL
            'vllm_model': 'llama-3-2-3b-it',  # vLLM model name
            
            # NEW: vLLM optimization settings
            'use_vllm_optim': True,        # Use vLLM for prompt optimization
            'vllm_optim_port': 11300,       # vLLM port for prompt optimization
            'use_system_prompt': True,     # Use system prompts during inference
            'vllm_optimization_priority': 'system_chat',  # Priority: 'system_chat', 'system_completions', 'no_system'
            
            # Shared task tracking settings
            'enable_task_tracking': True,  # Enable shared task tracking to prevent duplicate processing
            'task_tracking_timeout_minutes': 2,  # Timeout for task locks (minutes)
            
            # Cooldown settings
            'network_error_cooldown': NETWORK_DELAY_TIME_BUFFER,  # Seconds to wait after network errors
            'submission_failure_cooldown': 60,  # Seconds to wait after submission failures
            'validator_error_cooldown': 45,  # Seconds to wait after validator errors
            'max_cooldown_duration': 300,  # Maximum cooldown duration (5 minutes)
            'enable_cooldown_logging': True,  # Enable detailed cooldown logging
            
            # Traffic-specific cooldown settings (subnet compliance)
            'synthetic_traffic_cooldown': SYNTHETIC_TRAFFIC_COOLDOWN,  # 300s cooldown for synthetic traffic
            'organic_traffic_cooldown': ORGANIC_TRAFFIC_COOLDOWN,   # 120s cooldown for organic traffic
            
            # Fallback mechanism settings
            'enable_fallback_mechanism': False,  # Enable CLIP-based fallback for low-fidelity tasks
            'fallback_ratio_threshold': 0.8,   # Ratio threshold for triggering fallback (original_vs_optimized / optimized_vs_optimized)
            'fallback_max_retries': 1,         # Maximum number of prompt re-optimization attempts
            
            # Enhanced cooldown system settings
            'cooldown_violation_threshold': 5,  # Number of violations before applying penalty
            'cooldown_violation_penalty': 15,  # Additional penalty cooldown in seconds
            # DEPRECATED: Validation lock duration removed - now using MIN_TASK_INTERVAL constant
        # 'validation_lock_duration': 31,  # Default validation lock duration in seconds

            # Validator-compliant generation settings (for miner behavior as orchestrator)
            'generation.throttle_period': THROTTLE_PERIOD,  # Minimum throttle period for task completion (seconds)
            'generation.task_cooldown': THROTTLE_PERIOD + EMERGENCY_COOLDOWN_BUFFER,  # Cooldown between tasks from same validator (seconds)
            'generation.cooldown_violation_penalty': 102,  # Penalty for cooldown violations (seconds)
            'generation.cooldown_violations_threshold': 100,  # Threshold for malicious behavior
            'generation.cooldown_penalty': 600,  # Penalty for low quality submissions (seconds)
            'generation.quality_threshold': 0.6,  # Minimum score threshold for acceptance
            
            # Emergency cooldown management settings
            'emergency_cooldown_buffer': 5,  # Buffer seconds added to validator cooldowns
            'critical_violation_threshold': 100,  # Violation count that triggers emergency measures
            
            # Reactive cooldown system settings
            'base_backoff_duration': 31,  # Base backoff duration in seconds
            'max_backoff_duration': 301,  # Maximum backoff duration (5 minutes)
            'base_adaptive_backoff': 61,  # Base adaptive backoff duration in seconds
            'max_adaptive_backoff': 601,  # Maximum adaptive backoff duration (10 minutes)
            'critical_violation_cooldown': CRITICAL_VIOLATION_COOLDOWN,  # Emergency cooldown duration for critical violations
            'base_blacklist_duration': 1800,  # Base duration for temporary blacklisting
            
            # Cooldown system control
            'disable_all_cooldowns': False,  # Global cooldown disable flag
            
            # Fidelity score tracker settings
            'fidelity_tracker_history_size': 5,  # Number of recent tasks to keep in history queue
            'fidelity_tracker_zero_threshold': 2,  # Number of consecutive 0.0 scores to trigger endpoint switch
            'fidelity_tracker_fallback_endpoint': "/generate_3d_from_prompt_grid_flow/",  # Fallback endpoint for 0.0 scores
            'log_fidelity_tracking': True,  # Enable detailed logging of fidelity tracking and endpoint switching
        }
    
    def _setup_bittensor(self) -> bool:
        """Setup Bittensor components"""
        if not BITTENSOR_AVAILABLE:
            self.logger.error("❌ Bittensor not available")
            return False
        
        try:
            if self.wallet is None:
                self.wallet = bt.wallet(
                    name=self.config['wallet_name'],
                    hotkey=self.config['hotkey_name']
                )
                self.logger.info(f"✅ Wallet loaded: {self.wallet.hotkey.ss58_address}")
            
            # self.subtensor = bt.subtensor(network="test") #TODO
            if self.subtensor is None:
                self.subtensor = bt.subtensor(network="finney")
                self.logger.info("✅ Subtensor connected")
            
            if self.dendrite is None:
                self.dendrite = bt.dendrite(wallet=self.wallet)
                self.logger.info("✅ Dendrite initialized")
            
            if self.metagraph is None:
                self.metagraph = self.subtensor.metagraph(self.config['netuid'])
                self.logger.info(f"✅ Metagraph loaded (netuid: {self.config['netuid']})")
            
            return True
        except Exception as e:
            self.logger.error(f"❌ Bittensor setup failed: {e}")
            return False
    
    def refresh_validators(self):
        """Refresh validator information from metagraph - discover all active validators"""
        if not self._setup_bittensor():
            return
        
        try:
            # Refresh metagraph
            self.metagraph = self.subtensor.metagraph(self.config['netuid'])
            
            # Clear existing validators that are no longer valid
            valid_uids = set()
            
            # Discover all validators on the subnet
            eligible_validators = []
            
            for uid, neuron in enumerate(self.metagraph.neurons):
                # Check if this is a valid validator
                if not neuron.validator_permit:
                    continue
                
                stake = float(neuron.stake)
                trust = float(neuron.trust)
                consensus = float(neuron.consensus)
                
                # Apply filtering criteria
                if stake < self.config['min_validator_stake']:
                    continue
                
                if trust < self.config['min_validator_trust']:
                    continue
                
                # Check if validator is responsive (has recent activity)
                # This could be enhanced with ping checks in the future
                
                eligible_validators.append({
                    'uid': uid,
                    'stake': stake,
                    'trust': trust,
                    'consensus': consensus,
                    'hotkey': neuron.hotkey,
                    'score': stake * trust * consensus  # Simple scoring for prioritization
                })
            
            # Sort by score (stake * trust * consensus) and take top validators
            eligible_validators.sort(key=lambda x: x['score'], reverse=True)
            eligible_validators = eligible_validators[:self.config['max_validators']]
            
            # Update validator states
            for validator_info in eligible_validators:
                uid = validator_info['uid']
                valid_uids.add(uid)
                
                if uid not in self.validators:
                    # Create new validator state
                    self.validators[uid] = ValidatorState(
                        uid=uid,
                        hotkey=validator_info['hotkey'],
                        stake=validator_info['stake'],
                        trust=validator_info['trust'],
                        consensus=validator_info['consensus']
                    )
                    self.logger.info(f"➕ Added new validator UID {uid} (stake: {validator_info['stake']:.1f}, trust: {validator_info['trust']:.3f})")
                else:
                    # Update existing validator
                    validator = self.validators[uid]
                    validator.stake = validator_info['stake']
                    validator.trust = validator_info['trust']
                    validator.consensus = validator_info['consensus']
                    validator.hotkey = validator_info['hotkey']
                    validator.is_active = True
            
            # Mark validators not in the current list as inactive
            inactive_count = 0
            for uid in list(self.validators.keys()):
                if uid not in valid_uids:
                    if self.validators[uid].is_active:
                        self.logger.info(f"➖ Validator UID {uid} is no longer active")
                        self.validators[uid].is_active = False
                        inactive_count += 1
            
            active_validators = len([v for v in self.validators.values() if v.is_active])
            blacklisted_validators = len([v for v in self.validators.values() if v.is_active and self.is_validator_blacklisted(v.uid)])
            
            self.logger.info(f"✅ Validator refresh complete:")
            self.logger.info(f"   Active validators: {active_validators}")
            self.logger.info(f"   Blacklisted validators: {blacklisted_validators}")
            self.logger.info(f"   Inactive validators: {inactive_count}")
            self.logger.info(f"   Total eligible validators found: {len(eligible_validators)}")
            
            # Log blacklisted validators if any
            if blacklisted_validators > 0:
                blacklisted_uids = [v.uid for v in self.validators.values() if v.is_active and self.is_validator_blacklisted(v.uid)]
                self.logger.info(f"   🚫 Blacklisted UIDs: {blacklisted_uids}")
            else:
                self.logger.info(f"   ✅ No blacklisted validators found")
            
            # Log blacklisting configuration
            blacklist_config = self.config.get('validator_blacklist', [])
            blacklist_enabled = self.config.get('enable_validator_blacklisting', True)
            self.logger.info(f"   🔧 Blacklisting config: {'ENABLED' if blacklist_enabled else 'DISABLED'}")
            self.logger.info(f"   📋 Blacklist UIDs: {blacklist_config}")
            
            # Check each active validator for blacklisting status
            self.logger.info(f"   🔍 Checking blacklist status for each active validator:")
            for validator in sorted([v for v in self.validators.values() if v.is_active], key=lambda x: x.stake, reverse=True):
                blacklist_status = "🚫 BLACKLISTED" if self.is_validator_blacklisted(validator.uid) else "✅ ALLOWED"
                self.logger.info(f"     UID {validator.uid}: {blacklist_status} (stake: {validator.stake:.1f} TAO)")
            
            # Log top validators by stake
            top_validators = sorted(
                [v for v in self.validators.values() if v.is_active], 
                key=lambda x: x.stake, 
                reverse=True
            )[:5]
            
            self.logger.info("   Top validators by stake:")
            for validator in top_validators:
                self.logger.info(f"     UID {validator.uid}: {validator.stake:.1f} TAO (trust: {validator.trust:.3f})")
            
            # CRITICAL: Restore validator states from disk after discovery
            self.restore_validator_states_from_disk()
            
        except Exception as e:
            self.logger.error(f"❌ Validator refresh failed: {e}")
            traceback.print_exc()
    
    def is_validator_blacklisted(self, validator_uid: int) -> bool:
        """Check if a validator is blacklisted"""
        if not self.config.get('enable_validator_blacklisting', True):
            self.logger.info(f"🔓 Blacklisting DISABLED - UID {validator_uid} allowed")
            return False
        
        blacklist = self.config.get('validator_blacklist', [])
        is_blacklisted = validator_uid in blacklist
        
        if is_blacklisted:
            # self.logger.info(f"🚫 Validator UID {validator_uid} is BLACKLISTED - skipping")
            self.stats['blacklisted_validators_skipped'] += 1
        else:
            self.logger.debug(f"✅ Validator UID {validator_uid} is NOT blacklisted - allowing")
        
        return is_blacklisted
    
    def _is_validator_on_cooldown(self, validator: ValidatorState) -> tuple[bool, str, float]:
        """
        Check if validator is on any type of cooldown.
        
        Returns:
            Tuple of (is_on_cooldown, cooldown_type, remaining_seconds)
        """
        # Check if cooldowns are globally disabled
        if self.config.get('disable_all_cooldowns', False):
            return False, "none", 0.0
        
        current_time = time.time()
        
        # Check validator-enforced cooldown first (highest priority)
        if validator.validator_enforced_cooldown_until and current_time < validator.validator_enforced_cooldown_until:
            remaining = validator.validator_enforced_cooldown_until - current_time
            return True, "validator", remaining
        
        # Check miner cooldown (only if no validator cooldown is active)
        if validator.miner_cooldown_until and current_time < validator.miner_cooldown_until:
            remaining = validator.miner_cooldown_until - current_time
            return True, "miner", remaining
        
        return False, "none", 0.0
    
    def _safe_set_cooldown(self, validator: ValidatorState, new_cooldown: float):
        """
        Safely set cooldown for a validator, handling None cases.
        
        Args:
            validator: The validator to set cooldown for
            new_cooldown: The new cooldown time
        """
        if validator.validator_enforced_cooldown_until is None:
            validator.validator_enforced_cooldown_until = new_cooldown
        else:
            validator.validator_enforced_cooldown_until = max(validator.validator_enforced_cooldown_until, new_cooldown)
    
    def is_validator_available(self, validator: ValidatorState) -> bool:
        """Check if validator is available for task pulling"""
        current_time = time.time()

        # FIXED: Emergency state recovery check for 100% compliance
        self._recover_from_emergency_state(validator)
        
        # Check if validator is active
        if not validator.is_active:
            self.logger.debug(f"🔴 Validator UID {validator.uid} not available: INACTIVE")
            return False
        
        # Check if validator is blacklisted
        if self.is_validator_blacklisted(validator.uid):
            # self.logger.info(f"🚫 Validator UID {validator.uid} not available: BLACKLISTED")
            return False
        
        # Check emergency blacklist (new critical feature)
        if validator.emergency_blacklist_until and current_time < validator.emergency_blacklist_until:
            remaining = validator.emergency_blacklist_until - current_time
            self.logger.warning(f"🚨 Validator UID {validator.uid} not available: EMERGENCY BLACKLIST ({remaining:.1f}s remaining)")
            return False
        
        # Enhanced cooldown checking using helper method (subnet compliant)
        is_cooldown, cooldown_type, remaining = self._is_validator_on_cooldown(validator)
        if is_cooldown:
            if cooldown_type == "validator":
                self.logger.debug(f"⏳ Validator UID {validator.uid} not available: VALIDATOR COOLDOWN ({remaining:.1f}s remaining)")
            else:
                self.logger.debug(f"⏳ Validator UID {validator.uid} not available: MINER COOLDOWN ({remaining:.1f}s remaining)")
            return False
        
        # FIXED: Check if we have a pending cooldown that should be enforced now
        # Only enforce if we've actually completed the pending task
        if validator.validator_enforced_cooldown_until:
            current_time = time.time()
            if current_time < validator.validator_enforced_cooldown_until:
                # Check if we have a pending task - if so, allow one more pull to complete it
                if validator.pending_cooldown_task_id:
                    # We have a pending task - allow processing to complete it
                    remaining_cooldown = validator.validator_enforced_cooldown_until - current_time
                    self.logger.debug(f"⏳ Pending cooldown for UID {validator.uid}: {remaining_cooldown:.1f}s - allowing completion of task {validator.pending_cooldown_task_id}")
                else:
                    # No pending task - enforce the cooldown
                    remaining_cooldown = validator.validator_enforced_cooldown_until - current_time
                    self.logger.info(f"⏳ Enforcing validator cooldown for UID {validator.uid}: {remaining_cooldown:.1f}s remaining")
                    return False
            else:
                # Pending cooldown has expired - clear it
                validator.validator_enforced_cooldown_until = None
                validator.pending_cooldown_task_id = None
                self.logger.info(f"✅ Validator cooldown for UID {validator.uid} has expired")
        
        # DEPRECATED: Validation lock check removed - now using MIN_TASK_INTERVAL constant for rate limiting
        # Check validation lock (new enhanced feature)
        # if validator.validation_locked_until and current_time < validator.validation_locked_until:
        #     remaining = validator.validation_locked_until - current_time
        #     self.logger.debug(f"🔒 Validator UID {validator.uid} validation locked for {remaining:.1f}s")
        #     return False
        
        # Check if we pulled recently (respect MIN_TASK_INTERVAL - following _pull_task logic from trellis_miner.py)
        if validator.last_task_pull:
            time_since_pull = current_time - validator.last_task_pull
            # DEPRECATED: Hardcoded task_pull_interval - now using MIN_TASK_INTERVAL constant
            # if time_since_pull < self.config['task_pull_interval']:
            #     time_until_available = self.config['task_pull_interval'] - time_since_pull
            #     self.logger.debug(f"⏰ Validator UID {validator.uid} not available: PULL INTERVAL ({time_until_available:.1f}s until available)")
            #     return False
            if time_since_pull < MIN_TASK_INTERVAL:
                time_until_available = MIN_TASK_INTERVAL - time_since_pull
                self.logger.debug(f"⏰ Validator UID {validator.uid} not available: MIN_TASK_INTERVAL ({time_until_available:.1f}s until available) - following _pull_task logic")
                return False
        
        self.logger.debug(f"✅ Validator UID {validator.uid} is AVAILABLE for task pulling")
        return True
    
    def set_validator_cooldown(self, validator: ValidatorState, cooldown_seconds: int, reason: str, task_id: str = None, prompt: str = None, cooldown_type: str = "miner"):
        """
        Set a cooldown period for a validator with proper logging and duration limits.
        Now supports traffic-specific cooldowns and throttle period reduction (validator-compliant).

        Args:
            validator: The validator to set cooldown for
            cooldown_seconds: Cooldown duration in seconds (can be overridden by traffic type)
            reason: Reason for the cooldown (for logging)
            task_id: Task ID for traffic type detection
            prompt: Task prompt for traffic type detection
            cooldown_type: Type of cooldown ("validator" or "miner")
        """
        traffic_type = "Unknown"
        # Detect traffic type if task information is provided
        if task_id:
            traffic_type = self.detect_traffic_type(task_id, prompt)
            traffic_specific_cooldown = self.get_traffic_specific_cooldown(traffic_type)
            
            # ENFORCE traffic-specific cooldown minimums (subnet compliance)
            if traffic_type == "Synthetic":
                # DEPRECATED: Hardcoded 300s - now using FAILED_VALIDATOR_DELAY constant
                # cooldown_seconds = max(cooldown_seconds, 300)  # Enforce 300s minimum for synthetic
                cooldown_seconds = max(cooldown_seconds, SYNTHETIC_TRAFFIC_COOLDOWN)  # Enforce FAILED_VALIDATOR_DELAY minimum for synthetic
                self.logger.info(f"🔒 SYNTHETIC traffic detected - enforcing minimum 300s cooldown")
            elif traffic_type == "Organic":
                # DEPRECATED: Hardcoded 120s - now using NETWORK_DELAY_TIME_BUFFER constant
                # cooldown_seconds = max(cooldown_seconds, 120)  # Enforce 120s minimum for organic
                cooldown_seconds = max(cooldown_seconds, ORGANIC_TRAFFIC_COOLDOWN)  # Enforce NETWORK_DELAY_TIME_BUFFER minimum for organic
                self.logger.info(f"🍃 ORGANIC traffic detected - enforcing minimum 120s cooldown")
            
            # Use traffic-specific cooldown if it's different from the provided one
            if traffic_specific_cooldown != cooldown_seconds:
                self.logger.info(f"🔄 Overriding cooldown from {cooldown_seconds}s to {traffic_specific_cooldown}s based on traffic type: {traffic_type}")
                cooldown_seconds = traffic_specific_cooldown
        
        # Limit cooldown duration to prevent excessive waiting
        # DEPRECATED: Hardcoded 300s - now using FAILED_VALIDATOR_DELAY constant
        # max_cooldown = self.config.get('max_cooldown_duration', 300)
        max_cooldown = self.config.get('max_cooldown_duration', FAILED_VALIDATOR_DELAY)
        cooldown_seconds = min(cooldown_seconds, max_cooldown)
        
        # Apply throttle period reduction (validator-compliant logic)
        # This mirrors the validator's reset_task method logic
        throttle_period = self.config.get('generation.throttle_period', THROTTLE_PERIOD)
        # DEPRECATED: Hardcoded 300s - now using FAILED_VALIDATOR_DELAY constant
        # task_cooldown = self.config.get('generation.task_cooldown', 300)
        task_cooldown = self.config.get('generation.task_cooldown', THROTTLE_PERIOD + EMERGENCY_COOLDOWN_BUFFER)

        # Calculate effective cooldown using throttle period reduction
        # Similar to: max(time.time() + cooldown_seconds - throttle_period, assignment_time + cooldown_seconds)
        # Since we don't have assignment_time for validators, use a simplified version
        effective_cooldown = cooldown_seconds - throttle_period

        # Ensure minimum cooldown duration
        effective_cooldown = max(effective_cooldown, throttle_period)
        effective_cooldown = max(effective_cooldown, 0)  # Prevent negative cooldowns

        # Set cooldown based on type (subnet compliant)
        current_time = time.time()
        if cooldown_type == "validator":
            validator.validator_enforced_cooldown_until = current_time + effective_cooldown
            self.logger.info(f"🔒 Validator-enforced cooldown set for UID {validator.uid}: {effective_cooldown}s (original: {cooldown_seconds}s, throttle reduction: {throttle_period}s) ({reason})")
        else:
            validator.miner_cooldown_until = max(validator.miner_cooldown_until, current_time + effective_cooldown)
            # FIX 2: Use only one cooldown field per cooldown type
            # validator.cooldown_until = current_time + effective_cooldown
            self.logger.info(f"⏳ Miner cooldown set for UID {validator.uid}: {effective_cooldown}s (original: {cooldown_seconds}s, throttle reduction: {throttle_period}s) ({reason})")
        
        # Log cooldown with human-readable duration and traffic type info
        if self.config.get('enable_cooldown_logging', True):
            if effective_cooldown < 60:
                duration_str = f"{effective_cooldown}s"
            elif effective_cooldown < 3600:
                duration_str = f"{effective_cooldown//60}m {effective_cooldown%60}s"
            else:
                hours = effective_cooldown // 3600
                minutes = (effective_cooldown % 3600) // 60
                duration_str = f"{hours}h {minutes}m"

            traffic_info = f" (traffic: {traffic_type})" if task_id and 'traffic_type' in locals() else ""
            self.logger.info(f"⏳ Cooldown set for UID {validator.uid}: {duration_str}{traffic_info} ({reason})")
            # DEPRECATED: cooldown_until field - now using validator_enforced_cooldown_until and miner_cooldown_until
            # self.logger.info(f"   Next available: {time.strftime('%H:%M:%S', time.localtime(validator.cooldown_until))}")
            if validator.validator_enforced_cooldown_until:
                self.logger.info(f"   Next available (validator): {time.strftime('%H:%M:%S', time.localtime(validator.validator_enforced_cooldown_until))}")
            elif validator.miner_cooldown_until:
                self.logger.info(f"   Next available (miner): {time.strftime('%H:%M:%S', time.localtime(validator.miner_cooldown_until))}")
        else:
            self.logger.debug(f"⏳ Cooldown set for UID {validator.uid}: {effective_cooldown}s ({reason})")
    
    # DEPRECATED: Validation lock method removed - now using MIN_TASK_INTERVAL constant for rate limiting
    # def set_validator_validation_lock(self, validator: ValidatorState, lock_duration_seconds: int, reason: str):
    #     """
    #     Set a validation lock period for a validator.
    #         
    #     Args:
    #         validator: The validator to set validation lock for
    #         lock_duration_seconds: Lock duration in seconds
    #         reason: Reason for the validation lock (for logging)
    #     """
    #     # Set validation lock
    #     validator.validation_locked_until = time.time() + lock_duration_seconds
    #     self.stats['validation_locks_applied'] += 1
    #         
    #     # Log validation lock with human-readable duration
    #     if self.config.get('enable_cooldown_logging', True):
    #         if lock_duration_seconds < 60:
    #         duration_str = f"{lock_duration_seconds}s"
    #         elif lock_duration_seconds < 3600:
    #         duration_str = f"{lock_duration_seconds//60}m {lock_duration_seconds%60}s"
    #         else:
    #         hours = lock_duration_seconds // 3600
    #         minutes = (lock_duration_seconds % 3600) // 60
    #         duration_str = f"{hours}h {minutes}m"
    #             
    #         self.logger.info(f"🔒 Validation lock set for UID {validator.uid}: {duration_str} ({reason})")
    #         self.logger.info(f"   Next available: {time.strftime('%H:%M:%S', time.localtime(validator.validation_locked_until))}")
    #     else:
    #         self.logger.debug(f"🔒 Validation lock set for UID {validator.uid}: {lock_duration_seconds}s ({reason})")
    
    def increment_cooldown_violations(self, validator: ValidatorState, reason: str):
        """
        Increment cooldown violations counter for a validator.

        Args:
            validator: The validator to increment violations for
            reason: Reason for the violation (for logging)
        """
        # FIX 2: Use only one violation field per violation type
        # validator.cooldown_violations += 1
        validator.validator_reported_violations += 1
        self.stats['cooldown_violations_total'] += 1
        self.logger.warning(f"⚠️ Cooldown violation #{validator.validator_reported_violations} for UID {validator.uid}: {reason}")

        # Check if we should apply additional penalties
        violation_threshold = self.config.get('cooldown_violation_threshold', 5)
        if validator.validator_reported_violations >= violation_threshold:
            # DEPRECATED: Hardcoded 60s - now using NETWORK_DELAY_TIME_BUFFER constant
            # penalty_seconds = self.config.get('cooldown_violation_penalty', 60)
            penalty_seconds = self.config.get('cooldown_violation_penalty', FAILED_VALIDATOR_DELAY)
            self.logger.warning(f"🚨 Cooldown violation threshold reached for UID {validator.uid} - applying {penalty_seconds}s penalty")
            self.set_validator_cooldown(validator, penalty_seconds, f"Violation penalty (violation #{validator.validator_reported_violations})")
            self.stats['enhanced_cooldown_penalties'] += 1

    def _check_rapid_submission(self, validator: ValidatorState, task_id: str, prompt: str) -> bool:
        """
        Check if validator is submitting too rapidly (validator-compliant throttle logic).

        This mirrors the validator's logic:
        miner.last_submit_time + task_cooldown - throttle_period > time.time()

        Args:
            validator: The validator to check
            task_id: Current task ID
            prompt: Current task prompt

        Returns:
            True if submission is too rapid, False otherwise
        """
        if not validator.last_submit_time:
            return False

        # DEPRECATED: Hardcoded 300s - now using FAILED_VALIDATOR_DELAY constant
        # task_cooldown = self.config.get('generation.task_cooldown', 300)
        task_cooldown = self.config.get('generation.task_cooldown', THROTTLE_PERIOD + EMERGENCY_COOLDOWN_BUFFER)
        throttle_period = self.config.get('generation.throttle_period', THROTTLE_PERIOD)

        # Calculate the minimum time that should have passed
        min_time_required = validator.last_submit_time + task_cooldown - throttle_period

        if min_time_required > time.time():
            time_since_last_submit = time.time() - validator.last_submit_time
            self.logger.warning(
                f"[{validator.uid}] submitted too quickly: {time_since_last_submit:.1f}s "
                f"after last submit. Task: {task_id} | Prompt: {prompt[:100]}"
            )
            return True

        return False

    def _check_rapid_submission_timing_only(self, validator: ValidatorState) -> bool:
        """
        Check rapid submission timing without logging warnings (for pre-pull validation).

        Args:
            validator: The validator to check

        Returns:
            True if submission would be too rapid, False otherwise
        """
        if not validator.last_submit_time:
            return False

        # DEPRECATED: Hardcoded 300s - now using FAILED_VALIDATOR_DELAY constant
        # task_cooldown = self.config.get('generation.task_cooldown', 300)
        task_cooldown = self.config.get('generation.task_cooldown', THROTTLE_PERIOD + EMERGENCY_COOLDOWN_BUFFER)
        throttle_period = self.config.get('generation.throttle_period', THROTTLE_PERIOD)

        # Calculate the minimum time that should have passed
        min_time_required = validator.last_submit_time + task_cooldown - throttle_period

        return min_time_required > time.time()

    def _handle_task_failure(self, validator: ValidatorState, task: TaskRecord, score: float = 0.0, reason: str = "Task failure"):
        """
        Handle task failure with quality-based penalties (validator-compliant logic).

        This mirrors the validator's _process_task_failure method.

        Args:
            validator: The validator that failed
            task: The failed task
            score: The validation score (if available)
            reason: Reason for the failure
        """
        quality_threshold = self.config.get('generation.quality_threshold', 0.6)

        # Check if this is a quality-based failure
        if score > 0 and score < quality_threshold:
            # Apply quality penalty (validator logic)
            # DEPRECATED: Hardcoded 600s - now using GENERATION_ERROR_DELAY * 2 constant
            # cooldown_penalty = self.config.get('generation.cooldown_penalty', 600)
            cooldown_penalty = self.config.get('generation.cooldown_penalty', GENERATION_ERROR_DELAY * 2)
            self.logger.warning(
                f"[{validator.uid}] Quality failure - score {score:.3f} < threshold {quality_threshold}. "
                f"Applying {cooldown_penalty}s penalty. Task: {task.task_id}"
            )

            # Set cooldown with quality penalty
            self.set_validator_cooldown(
                validator,
                cooldown_penalty,
                f"Quality penalty: score {score:.3f} < {quality_threshold}",
                task.task_id,
                task.prompt
            )

            # Update stats
            self.stats['quality_penalties'] = self.stats.get('quality_penalties', 0) + 1
        else:
            # Regular failure - use submission failure cooldown
            # DEPRECATED: Hardcoded 60s - now using NETWORK_DELAY_TIME_BUFFER constant
            # submission_cooldown = self.config.get('submission_failure_cooldown', 60)
            submission_cooldown = self.config.get('submission_failure_cooldown', NETWORK_DELAY_TIME_BUFFER)
            self.set_validator_cooldown(
                validator,
                submission_cooldown,
                reason,
                task.task_id,
                task.prompt
            )

    def _set_emergency_cooldown(self, validator: ValidatorState, cooldown_until: int, reason: str):
        """
        Set DYNAMIC emergency cooldown to prevent further violations.
        Automatically adjusts buffer based on validator history.
        
        Args:
            validator: The validator to set emergency cooldown for
            cooldown_until: Timestamp when cooldown expires
            reason: Reason for emergency cooldown
        """
        current_time = time.time()
        if cooldown_until > current_time:
            # DYNAMIC: Calculate buffer time based on validator history
            # DEPRECATED: Hardcoded 30s - now using MIN_TASK_INTERVAL constant
            # base_buffer = self.config.get('emergency_cooldown_buffer', 30)
            base_buffer = self.config.get('emergency_cooldown_buffer', MIN_TASK_INTERVAL)
            
            if hasattr(validator, 'violation_history') and validator.violation_history:
                # Analyze recent violations to determine buffer size
                recent_violations = [v['violations'] for v in validator.violation_history[-3:]]
                avg_recent_violations = sum(recent_violations) / len(recent_violations) if recent_violations else 0
                
                if avg_recent_violations > 1000:  # Extreme violations
                    buffer_multiplier = 3.0  # 3x buffer for extreme cases
                    self.logger.warning(f"   EXTREME violation history - applying 3x buffer multiplier")
                elif avg_recent_violations > 500:  # High violations
                    buffer_multiplier = 2.0  # 2x buffer for high cases
                    self.logger.warning(f"   HIGH violation history - applying 2x buffer multiplier")
                elif avg_recent_violations > 200:  # Moderate violations
                    buffer_multiplier = 1.5  # 1.5x buffer for moderate cases
                    self.logger.warning(f"   MODERATE violation history - applying 1.5x buffer multiplier")
                else:
                    buffer_multiplier = 1.0  # 1x buffer for standard cases
                    self.logger.info(f"   STANDARD violation history - applying 1x buffer multiplier")
                
                dynamic_buffer = int(base_buffer * buffer_multiplier)
            else:
                # No history - use base buffer
                dynamic_buffer = base_buffer
                buffer_multiplier = 1.0
            
            emergency_cooldown_until = cooldown_until + dynamic_buffer
            
            # CRITICAL FIX: Prevent infinite cooldown escalation
            # if (validator.cooldown_until and 
            #     validator.cooldown_until > emergency_cooldown_until):
            if (validator.validator_enforced_cooldown_until and 
                validator.validator_enforced_cooldown_until > emergency_cooldown_until):
                self.logger.warning(f"⚠️ Emergency cooldown already set for UID {validator.uid} - not escalating")
                return
            
            # FIX 2: Use only one cooldown field per cooldown type
            # validator.cooldown_until = emergency_cooldown_until
            self.logger.warning(f"🚨 DYNAMIC emergency cooldown set for UID {validator.uid}: {reason}")
            self.logger.warning(f"   Original cooldown: {cooldown_until}, Emergency cooldown: {emergency_cooldown_until}")
            self.logger.warning(f"   DYNAMIC buffer: {dynamic_buffer}s (base: {base_buffer}s, multiplier: {buffer_multiplier:.1f}x)")
            
            # Track emergency cooldowns with dynamic info
            self.stats['emergency_cooldowns_applied'] = self.stats.get('emergency_cooldowns_applied', 0) + 1
            self.stats['dynamic_buffer_applied'] = self.stats.get('dynamic_buffer_applied', 0) + 1
            
            # Store buffer history for learning
            if not hasattr(validator, 'buffer_history'):
                validator.buffer_history = []
            validator.buffer_history.append({
                'timestamp': time.time(),
                'base_buffer': base_buffer,
                'dynamic_buffer': dynamic_buffer,
                'multiplier': buffer_multiplier,
                'reason': reason
            })
            
            # Keep only last 5 buffer adjustments for memory management
            if len(validator.buffer_history) > 5:
                validator.buffer_history = validator.buffer_history[-5:]
        else:
            self.logger.debug(f"ℹ️ Emergency cooldown not needed for UID {validator.uid} - cooldown already expired")
    
    def _handle_critical_violations(self, validator: ValidatorState, violation_count: int):
        """
        Handle critical violation situations with DYNAMIC emergency measures.
        Automatically adjusts cooldown duration based on violation severity.
        
        Args:
            validator: The validator with critical violations
            violation_count: Current violation count
        """
        # CRITICAL FIX: Prevent multiple emergency measures
        if validator.emergency_blacklist_until and time.time() < validator.emergency_blacklist_until:
            self.logger.warning(f"⚠️ Emergency measures already active for UID {validator.uid} - skipping duplicate")
            return
        
        self.logger.error(f"🚨 CRITICAL: Implementing DYNAMIC emergency measures for UID {validator.uid}")
        self.logger.error(f"   Violation count: {violation_count} (threshold: 100)")
        
        # DYNAMIC: Calculate emergency duration based on violation severity
        base_duration = self.config.get('critical_violation_cooldown', 3600)  # 1 hour base
        
        # Scale duration based on violation count (exponential backoff)
        if violation_count > 1000:
            scale_factor = 4.0  # 4x for extreme violations
            self.logger.error(f"   EXTREME violations detected - applying 4x multiplier")
        elif violation_count > 500:
            scale_factor = 2.5  # 2.5x for high violations
            self.logger.error(f"   HIGH violations detected - applying 2.5x multiplier")
        elif violation_count > 200:
            scale_factor = 1.5  # 1.5x for moderate violations
            self.logger.error(f"   MODERATE violations detected - applying 1.5x multiplier")
        else:
            scale_factor = 1.0  # 1x for standard violations
            self.logger.error(f"   STANDARD violations detected - applying 1x multiplier")
        
        emergency_duration = int(base_duration * scale_factor)
        emergency_cooldown_until = time.time() + emergency_duration
        
        # FIX 2: Use only one cooldown field per cooldown type
        # validator.cooldown_until = emergency_cooldown_until
        self.logger.error(f"   DYNAMIC emergency cooldown: {emergency_duration}s (base: {base_duration}s, scale: {scale_factor:.1f}x)")
        self.logger.error(f"   Cooldown until: {emergency_cooldown_until}")
        
        # Mark validator as temporarily blacklisted
        validator.is_active = False
        validator.emergency_blacklist_until = emergency_cooldown_until
        
        self.logger.error(f"   Validator UID {validator.uid} temporarily blacklisted until cooldown expires")
        
        # Track critical violations with dynamic scaling info
        self.stats['critical_violations_handled'] = self.stats.get('critical_violations_handled', 0) + 1
        self.stats['dynamic_cooldown_scaling'] = self.stats.get('dynamic_cooldown_scaling', 0) + 1
        
        # CRITICAL: Save state immediately after handling critical violations
        self.save_validator_states_to_disk()
        
        # Store violation history for adaptive learning
        if not hasattr(validator, 'violation_history'):
            validator.violation_history = []
        validator.violation_history.append({
            'timestamp': time.time(),
            'violations': violation_count,
            'cooldown_duration': emergency_duration,
            'scale_factor': scale_factor
        })
        
        # Keep only last 10 violations for memory management
        if len(validator.violation_history) > 10:
            validator.violation_history = validator.violation_history[-10:]
    
    def _blacklist_validator_temporarily(self, validator: ValidatorState, violation_count: int):
        """
        Temporarily blacklist a validator due to excessive violations.
        
        Args:
            validator: The validator to blacklist
            violation_count: Current violation count
        """
        # CRITICAL FIX: Prevent multiple blacklistings
        if validator.emergency_blacklist_until and time.time() < validator.emergency_blacklist_until:
            self.logger.warning(f"⚠️ Validator UID {validator.uid} already blacklisted - skipping duplicate")
            return
        
        self.logger.error(f"🚨 BLACKLISTING: Validator UID {validator.uid} due to {violation_count} violations")
        
        # Calculate blacklist duration based on violation count
        base_duration = self.config.get('base_blacklist_duration', 900)  # 15 minutes
        violation_multiplier = min(violation_count / 50, 10)  # Cap at 10x
        blacklist_duration = int(base_duration * violation_multiplier)
        
        blacklist_until = time.time() + blacklist_duration
        
        # Set blacklist
        validator.is_active = False
        validator.emergency_blacklist_until = blacklist_until
        # FIX 2: Use only one cooldown field per cooldown type
        # validator.cooldown_until = blacklist_until
        
        self.logger.error(f"   Blacklist duration: {blacklist_duration}s (until {blacklist_until})")
        self.logger.error(f"   Violation multiplier: {violation_multiplier:.1f}x")
        
        # Track blacklists
        self.stats['validators_temporarily_blacklisted'] = self.stats.get('validators_temporarily_blacklisted', 0) + 1
        
        # CRITICAL: Save state immediately after blacklisting
        self.save_validator_states_to_disk()
    
    def _check_and_clear_expired_emergency_blacklists(self):
        """
        Check and clear expired emergency blacklists and cooldowns.
        This should be called periodically to restore validators.
        """
        current_time = time.time()
        cleared_count = 0
        
        for validator in self.validators.values():
            # Check emergency blacklist
            if (validator.emergency_blacklist_until and 
                current_time >= validator.emergency_blacklist_until):
                
                self.logger.info(f"✅ Emergency blacklist expired for UID {validator.uid}")
                self._safe_reset_validator(validator, "emergency blacklist")
                cleared_count += 1
            
            # Check if cooldown has expired but emergency blacklist is still active
                    # DEPRECATED: cooldown_until field - now using validator_enforced_cooldown_until and miner_cooldown_until
        # if (validator.cooldown_until and
        #     current_time >= validator.cooldown_until and
        #     validator.emergency_blacklist_until and
        #     current_time >= validator.emergency_blacklist_until):
        if (validator.emergency_blacklist_until and
            current_time >= validator.emergency_blacklist_until):
                
                self.logger.info(f"✅ Cooldown and emergency blacklist expired for UID {validator.uid}")
                self._safe_reset_validator(validator, "cooldown and emergency blacklist")
                cleared_count += 1
        
        if cleared_count > 0:
            self.logger.info(f"🔄 Restored {cleared_count} validators from expired emergency restrictions")
        
        return cleared_count
    
    def _safe_reset_validator(self, validator: ValidatorState, reason: str):
        """
        Safely reset a validator's emergency restrictions.
        
        Args:
            validator: The validator to reset
            reason: Reason for the reset
        """
        self.logger.info(f"🔄 Safely resetting UID {validator.uid}: {reason}")
        
        # Clear emergency restrictions
        validator.emergency_blacklist_until = None
        # FIX 2: Use only one cooldown field per cooldown type
        # validator.cooldown_until = None
        validator.is_active = True
        
        # DYNAMIC: Reset violation counter based on validator history and behavior
        if hasattr(validator, 'violation_history') and validator.violation_history:
            # Analyze recent violation patterns
            recent_violations = [v['violations'] for v in validator.violation_history[-3:]]  # Last 3 violations
            avg_recent_violations = sum(recent_violations) / len(recent_violations) if recent_violations else 0
            
            if avg_recent_violations > 1000:  # Extreme violations
                reduction_factor = 0.1  # Reduce to 10% (very aggressive)
                self.logger.warning(f"   EXTREME violation history - aggressive reduction to 10%")
            elif avg_recent_violations > 500:  # High violations
                reduction_factor = 0.2  # Reduce to 20%
                self.logger.warning(f"   HIGH violation history - aggressive reduction to 20%")
            elif avg_recent_violations > 200:  # Moderate violations
                reduction_factor = 0.3  # Reduce to 30%
                self.logger.warning(f"   MODERATE violation history - moderate reduction to 30%")
            else:
                reduction_factor = 0.5  # Reduce to 50% (standard)
                self.logger.info(f"   STANDARD violation history - standard reduction to 50%")
            
            new_violation_count = max(1, int(validator.validator_reported_violations * reduction_factor))
            old_count = validator.validator_reported_violations
            validator.validator_reported_violations = new_violation_count
            
            self.logger.info(f"   DYNAMIC violation reduction: {old_count} → {new_violation_count} (factor: {reduction_factor:.1f})")
        else:
            # No history - use standard reduction
            if validator.validator_reported_violations > 10:
                validator.validator_reported_violations = max(5, validator.validator_reported_violations // 2)
                self.logger.info(f"   Standard violation reduction: {validator.validator_reported_violations * 2} → {validator.validator_reported_violations}")
        
        # Log the reset
        self.logger.info(f"   UID {validator.uid} is now available for task pulling")
        
        # Track resets
        self.stats['validators_reset_from_emergency'] = self.stats.get('validators_reset_from_emergency', 0) + 1
        
        # CRITICAL: Save state after validator reset
        self.save_validator_states_to_disk()
        
        # DYNAMIC: Check if validator needs extended monitoring based on history
        if hasattr(validator, 'violation_history') and validator.violation_history:
            recent_trend = self._analyze_violation_trend(validator)
            if recent_trend == 'increasing':
                self.logger.warning(f"⚠️ UID {validator.uid} shows INCREASING violation trend - extended monitoring")
                # Set a shorter cooldown for problematic validators
                # DEPRECATED: Hardcoded 300s - now using FAILED_VALIDATOR_DELAY constant
                # extended_monitoring_cooldown = time.time() + 300  # 5 minutes
                extended_monitoring_cooldown = time.time() + FAILED_VALIDATOR_DELAY  # 5 minutes
                # FIX 2: Use only one cooldown field per cooldown type
                # validator.cooldown_until = extended_monitoring_cooldown
                self.logger.warning(f"   Extended monitoring cooldown set: 5 minutes")
            elif recent_trend == 'stable_high':
                self.logger.warning(f"⚠️ UID {validator.uid} shows STABLE HIGH violations - close monitoring")
            else:
                self.logger.info(f"✅ UID {validator.uid} shows IMPROVING trend - standard monitoring")
        else:
            # No history - standard check
            if validator.validator_reported_violations > 50:
                self.logger.warning(f"⚠️ UID {validator.uid} still has high violations ({validator.validator_reported_violations}) after reset")
                self.logger.warning(f"   Monitoring closely - may need extended cooldown if violations persist")
    
    def _check_validator_cooldown_state(self, validator: ValidatorState) -> Dict[str, Any]:
        """
        Pre-emptive cooldown state checking before attempting task pull.
        Returns comprehensive cooldown status and recommendations.
        
        Args:
            validator: The validator to check
            
        Returns:
            Dict with cooldown status, remaining time, and recommendations
        """
        current_time = time.time()
        cooldown_status = {
            'available': True,
            'reason': None,
            'remaining_time': 0,
            'cooldown_type': None,
            'recommendation': None
        }
        
        # Check local cooldown state (FIX 2: Use the right cooldown field)
        # if validator.cooldown_until and current_time < validator.cooldown_until:
        if validator.validator_enforced_cooldown_until and current_time < validator.validator_enforced_cooldown_until:
            remaining = validator.validator_enforced_cooldown_until - current_time
            cooldown_status.update({
                'available': False,
                'reason': 'Local cooldown active',
                'remaining_time': remaining,
                'cooldown_type': 'local',
                'recommendation': f'Wait {remaining:.1f}s for local cooldown to expire'
            })
            return cooldown_status
        
        # Check emergency blacklist
        if validator.emergency_blacklist_until and current_time < validator.emergency_blacklist_until:
            remaining = validator.emergency_blacklist_until - current_time
            cooldown_status.update({
                'available': False,
                'reason': 'Emergency blacklist active',
                'remaining_time': remaining,
                'cooldown_type': 'emergency',
                'recommendation': f'Wait {remaining:.1f}s for emergency blacklist to expire'
            })
            return cooldown_status
        
        # Check validation lock
        if validator.validation_locked_until and current_time < validator.validation_locked_until:
            remaining = validator.validation_locked_until - current_time
            cooldown_status.update({
                'available': False,
                'reason': 'Validation lock active',
                'remaining_time': remaining,
                'cooldown_type': 'validation_lock',
                'recommendation': f'Wait {remaining:.1f}s for validation lock to expire'
            })
            return cooldown_status
        
        # Check pull interval (following _pull_task logic from trellis_miner.py)
        if validator.last_task_pull:
            time_since_pull = current_time - validator.last_task_pull
            # DEPRECATED: Hardcoded task_pull_interval - now using MIN_TASK_INTERVAL constant
            # if time_since_pull < self.config['task_pull_interval']:
            #     time_until_available = self.config['task_pull_interval'] - time_since_pull
            if time_since_pull < MIN_TASK_INTERVAL:
                time_until_available = MIN_TASK_INTERVAL - time_since_pull
                cooldown_status.update({
                    'available': False,
                    'reason': 'MIN_TASK_INTERVAL not met',
                    'remaining_time': time_until_available,
                    'cooldown_type': 'min_task_interval',
                    'recommendation': f'Wait {time_until_available:.1f}s for MIN_TASK_INTERVAL (following _pull_task logic)'
                })
                return cooldown_status

        if validator.last_submit_time:
            time_since_submit = current_time - validator.last_submit_time
            if time_since_submit < THROTTLE_PERIOD:
                time_until_available = THROTTLE_PERIOD - time_since_submit
                cooldown_status.update({
                    'available': False,
                    'reason': 'THROTTLE_PERIOD not met',
                    'remaining_time': time_until_available,
                    'cooldown_type': 'throttle_period',
                    'recommendation': f'Wait {time_until_available:.1f}s for THROTTLE_PERIOD'
                })
                return cooldown_status
        
        # Check if validator is active and not blacklisted
        if not validator.is_active:
            cooldown_status.update({
                'available': False,
                'reason': 'Validator inactive',
                'cooldown_type': 'inactive',
                'recommendation': 'Validator marked as inactive'
            })
            return cooldown_status
        
        if self.is_validator_blacklisted(validator.uid):
            cooldown_status.update({
                'available': False,
                'reason': 'Validator blacklisted',
                'cooldown_type': 'blacklisted',
                'recommendation': 'Validator is in blacklist'
            })
            return cooldown_status
        
        # Validator is available
        cooldown_status.update({
            'available': True,
            'reason': 'Validator available',
            'cooldown_type': 'available',
            'recommendation': 'Proceed with task pull'
        })
        
        return cooldown_status
    
    def _synchronize_validator_state(self, validator: ValidatorState, response_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Synchronize local validator state with validator-reported state.
        Implements graceful degradation and backoff strategies.
        
        Args:
            validator: The validator to synchronize
            response_data: Response data from validator
            
        Returns:
            Dict with synchronization results and actions taken
        """
        sync_results = {
            'cooldown_updated': False,
            'violations_updated': False,
            'throttle_updated': False,
            'emergency_actions': [],
            'backoff_strategy': None
        }
        
        current_time = time.time()
        
        # DEPRECATED: Legacy cooldown_until field synchronization - now using validator_enforced_cooldown_until
        # Synchronize cooldown state
        if 'cooldown_until' in response_data and response_data['cooldown_until']:
            # old_cooldown = validator.cooldown_until  # DEPRECATED: cooldown_until field
            new_cooldown = response_data['cooldown_until']
            
            # Only update if the new cooldown is more restrictive
            # if not old_cooldown or new_cooldown > old_cooldown:  # DEPRECATED: cooldown_until field
            # DEPRECATED: Legacy field assignment
            # validator.cooldown_until = new_cooldown
            # FIX 2: Use only one cooldown field per cooldown type
            # validator.cooldown_until = new_cooldown +2 # Backward compatibility
                        # DEPRECATED: Hardcoded 2s - now using NETWORK_DELAY_TIME_BUFFER constant
            # validator.validator_enforced_cooldown_until = new_cooldown +2  # New subnet field
            validator.validator_enforced_cooldown_until = new_cooldown + 5  # New subnet field
            sync_results['cooldown_updated'] = True
            
            if new_cooldown > current_time:
                remaining = new_cooldown - current_time
                self.logger.warning(f"🔄 Synchronized VALIDATOR-ENFORCED cooldown for UID {validator.uid}: {remaining:.1f}s remaining")
                
                # Implement graceful degradation with backoff
                backoff_duration = self._calculate_backoff_duration(validator, remaining)
                sync_results['backoff_strategy'] = f'Backoff for {backoff_duration:.1f}s'
                
                # FIX 1: Only set emergency cooldowns for actual violations, not normal sync
                # self._set_emergency_cooldown_with_backoff(validator, new_cooldown, backoff_duration, "Validator state sync")
                # sync_results['emergency_actions'].append('emergency_cooldown_with_backoff')
        
        if 'validator_enforced_cooldown_until' in response_data and response_data['validator_enforced_cooldown_until']:
            old_cooldown = validator.validator_enforced_cooldown_until
            new_cooldown = response_data['validator_enforced_cooldown_until']
            
            if new_cooldown != old_cooldown:
                validator.validator_enforced_cooldown_until = new_cooldown
                sync_results['cooldown_updated'] = True
                if new_cooldown > current_time:
                    remaining = new_cooldown - current_time
                    self.logger.warning(f"🔄 Synchronized DUPLICATE VALIDATOR-ENFORCED cooldown for UID {validator.uid}: {remaining:.1f}s remaining")
                    
        # Synchronize violation counts
        if 'cooldown_violations' in response_data and response_data['cooldown_violations'] is not None:
            old_violations = getattr(validator, 'cooldown_violations', 0)
            new_violations = response_data['cooldown_violations']
            
            if new_violations != old_violations:
                # validator.cooldown_violations = new_violations
                
                # FIX 2: Use only one violation field per violation type
                # validator.cooldown_violations = new_violations  # Backward compatibility
                validator.validator_reported_violations = new_violations  # New subnet field
                sync_results['violations_updated'] = True
                
                if new_violations > old_violations:
                    violation_increase = new_violations - old_violations
                    self.logger.error(f"🔄 Synchronized violations for UID {validator.uid}: {old_violations} → {new_violations} (+{violation_increase})")
                    self.logger.error(f"🔄 Synchronized VALIDATOR-REPORTED violations for UID {validator.uid}: {old_violations} → {new_violations} (+{violation_increase})")

                    # Implement adaptive backoff based on violation increase
                    if violation_increase > VIOLATION_INCREASE_DELTA:
                        adaptive_backoff = self._calculate_adaptive_backoff(validator, violation_increase)
                        sync_results['backoff_strategy'] = f'Adaptive backoff for {adaptive_backoff:.1f}s'
                        sync_results['emergency_actions'].append('adaptive_backoff')
                        
                        # Set adaptive emergency cooldown
                        self._set_adaptive_emergency_cooldown(validator, adaptive_backoff, violation_increase)
        
        # Synchronize throttle period
        if 'throttle_period' in response_data and response_data['throttle_period']:
            old_throttle = getattr(validator, 'throttle_period', None)
            new_throttle = response_data['throttle_period']
            
            if new_throttle != old_throttle:
                validator.throttle_period = new_throttle
                sync_results['throttle_updated'] = True
                self.logger.debug(f"🔄 Synchronized throttle for UID {validator.uid}: {new_throttle}s")
        
        # FIXED: Add missing synchronization fields for 100% compliance
        # Synchronize violation history for adaptive backoff
        if 'violation_history' in response_data:
            validator.violation_history = response_data['violation_history']
            sync_results['violation_history_updated'] = True
            self.logger.debug(f"🔄 Synchronized violation history for UID {validator.uid}")
        
        # Synchronize buffer history for emergency cooldown management
        if 'buffer_history' in response_data:
            validator.buffer_history = response_data['buffer_history']
            sync_results['buffer_history_updated'] = True
            self.logger.debug(f"🔄 Synchronized buffer history for UID {validator.uid}")
        
        # Synchronize emergency blacklist state
        if 'emergency_blacklist_until' in response_data and response_data['emergency_blacklist_until']:
            old_blacklist = validator.emergency_blacklist_until
            new_blacklist = response_data['emergency_blacklist_until']
            
            if new_blacklist != old_blacklist:
                validator.emergency_blacklist_until = new_blacklist
                sync_results['emergency_blacklist_updated'] = True
                self.logger.warning(f"🔄 Synchronized emergency blacklist for UID {validator.uid}: {new_blacklist}")
        
        return sync_results
    
    def _calculate_backoff_duration(self, validator: ValidatorState, cooldown_remaining: float) -> float:
        """
        Calculate intelligent backoff duration based on validator history and cooldown.
        
        Args:
            validator: The validator
            cooldown_remaining: Remaining cooldown time
            
        Returns:
            Backoff duration in seconds
        """
        base_backoff = self.config.get('base_backoff_duration', 30)
        
        # Factor in violation history
        violation_multiplier = 1.0
        if hasattr(validator, 'validator_reported_violations') and validator.validator_reported_violations:
            if validator.validator_reported_violations > 1000:
                violation_multiplier = 3.0  # Extreme violations
            elif validator.validator_reported_violations > 500:
                violation_multiplier = 2.0  # High violations
            elif validator.validator_reported_violations > 200:
                violation_multiplier = 1.5  # Moderate violations
            else:
                violation_multiplier = 1.0  # Standard violations
        
        # Factor in cooldown remaining
        # DEPRECATED: Hardcoded 60s - now using NETWORK_DELAY_TIME_BUFFER constant
        # cooldown_factor = min(cooldown_remaining / 60, 2.0)  # Cap at 2x
        cooldown_factor = min(cooldown_remaining / NETWORK_DELAY_TIME_BUFFER, 2.0)  # Cap at 2x
        
        # Calculate final backoff
        backoff_duration = base_backoff * violation_multiplier * cooldown_factor
        
        # Cap maximum backoff
        max_backoff = self.config.get('max_backoff_duration', 300)
        backoff_duration = min(backoff_duration, max_backoff)
        
        return backoff_duration
    
    def _calculate_adaptive_backoff(self, validator: ValidatorState, violation_increase: int) -> float:
        """
        Calculate adaptive backoff based on violation increase rate.
        
        Args:
            validator: The validator
            violation_increase: Increase in violations since last check
            
        Returns:
            Adaptive backoff duration in seconds
        """
        base_adaptive_backoff = self.config.get('base_adaptive_backoff', 60)
        
        # Exponential backoff based on violation increase
        if violation_increase > 100:
            multiplier = 4.0  # Extreme increase
        elif violation_increase > 50:
            multiplier = 3.0  # High increase
        elif violation_increase > 25:
            multiplier = 2.0  # Moderate increase
        else:
            multiplier = 1.5  # Low increase
        
        # Factor in validator's historical performance
        if hasattr(validator, 'total_tasks_pulled') and validator.total_tasks_pulled > 0:
            success_rate = validator.total_tasks_pulled / (validator.total_tasks_pulled + getattr(validator, 'validator_reported_violations', 0))
            if success_rate < 0.5:
                multiplier *= 1.5  # Poor performance
            elif success_rate < 0.8:
                multiplier *= 1.2  # Below average performance
        
        adaptive_backoff = base_adaptive_backoff * multiplier
        
        # Cap maximum adaptive backoff
        max_adaptive_backoff = self.config.get('max_adaptive_backoff', 600)
        adaptive_backoff = min(adaptive_backoff, max_adaptive_backoff)
        
        return adaptive_backoff
    
    def _set_emergency_cooldown_with_backoff(self, validator: ValidatorState, cooldown_until: int, backoff_duration: float, reason: str):
        """
        Set emergency cooldown with intelligent backoff to prevent further violations.
        
        Args:
            validator: The validator
            cooldown_until: When the cooldown expires
            backoff_duration: Additional backoff duration
            reason: Reason for the emergency cooldown
        """
        current_time = time.time()
        
        # Calculate emergency cooldown with backoff
        emergency_cooldown_until = max(cooldown_until, current_time + backoff_duration)
        
        # Prevent infinite escalation (FIX 2: Use only one cooldown field per cooldown type)
        # if (validator.cooldown_until and 
        #     validator.cooldown_until > emergency_cooldown_until):
        if (validator.validator_enforced_cooldown_until and 
            validator.validator_enforced_cooldown_until > emergency_cooldown_until):
            self.logger.warning(f"⚠️ Emergency cooldown already set for UID {validator.uid} - not escalating")
            return
        
        # FIX 2: Use only one cooldown field per cooldown type
        # validator.cooldown_until = emergency_cooldown_until
        
        self.logger.warning(f"🚨 Emergency cooldown with backoff set for UID {validator.uid}: {reason}")
        self.logger.warning(f"   Original cooldown: {cooldown_until}, Emergency cooldown: {emergency_cooldown_until}")
        self.logger.warning(f"   Backoff duration: {backoff_duration:.1f}s")
        
        # Track emergency cooldowns with backoff
        self.stats['emergency_cooldowns_with_backoff'] = self.stats.get('emergency_cooldowns_with_backoff', 0) + 1
        
        # Store backoff history for learning
        if not hasattr(validator, 'backoff_history'):
            validator.backoff_history = []
        validator.backoff_history.append({
            'timestamp': time.time(),
            'backoff_duration': backoff_duration,
            'reason': reason,
            'cooldown_until': cooldown_until,
            'emergency_cooldown_until': emergency_cooldown_until
        })
        
        # Keep only last 5 backoff adjustments
        if len(validator.backoff_history) > 5:
            validator.backoff_history = validator.backoff_history[-5:]
    
    def _set_adaptive_emergency_cooldown(self, validator: ValidatorState, backoff_duration: float, violation_increase: int):
        """
        Set adaptive emergency cooldown based on violation increase rate.
        
        Args:
            validator: The validator
            backoff_duration: Calculated backoff duration
            violation_increase: Increase in violations
        """
        current_time = time.time()
        emergency_cooldown_until = current_time + backoff_duration
        
        # Prevent infinite escalation (FIX 2: Use only one cooldown field per cooldown type)
        # if (validator.cooldown_until and 
        #     validator.cooldown_until > emergency_cooldown_until):
        if (validator.validator_enforced_cooldown_until and 
            validator.validator_enforced_cooldown_until > emergency_cooldown_until):
            self.logger.warning(f"⚠️ Adaptive emergency cooldown already set for UID {validator.uid} - not escalating")
            return
        
        # FIX 2: Use only one cooldown field per cooldown type
        # validator.cooldown_until = emergency_cooldown_until
        validator.validator_enforced_cooldown_until = emergency_cooldown_until
        
        self.logger.error(f"🚨 Adaptive emergency cooldown set for UID {validator.uid}")
        self.logger.error(f"   Violation increase: +{violation_increase}, Backoff: {backoff_duration:.1f}s")
        self.logger.error(f"   Emergency cooldown until: {emergency_cooldown_until}")
        
        # Track adaptive emergency cooldowns
        self.stats['adaptive_emergency_cooldowns'] = self.stats.get('adaptive_emergency_cooldowns', 0) + 1
        
        # Store adaptive cooldown history
        if not hasattr(validator, 'adaptive_cooldown_history'):
            validator.adaptive_cooldown_history = []
        validator.adaptive_cooldown_history.append({
            'timestamp': time.time(),
            'violation_increase': violation_increase,
            'backoff_duration': backoff_duration,
            'emergency_cooldown_until': emergency_cooldown_until
        })
        
        # Keep only last 5 adaptive cooldown adjustments
        if len(validator.adaptive_cooldown_history) > 5:
            validator.adaptive_cooldown_history = validator.adaptive_cooldown_history[-5:]
    
    def detect_traffic_type(self, task_id: str, prompt: str = "") -> str:
        """
        Detect traffic type based on task ID pattern and prompt analysis.
        Returns: 'synthetic', 'organic', or 'unknown' (defaults to synthetic for safety)
        """
        try:
            # Method 1: Task ID pattern analysis (most reliable)
            task_id_lower = task_id.lower()
            
            # Synthetic task patterns
            synthetic_patterns = [
                'syn_', 'synthetic', 'test_', 'benchmark_', 'validation_',
                'duel_', 'challenge_', 'competition_', 'evaluation_'
            ]
            
            # Organic task patterns
            organic_patterns = [
                'org_', 'organic', 'gateway_', 'legacy_', 'user_', 'real_',
                'production_', 'live_', 'api_', 'external_'
            ]
            
            # Check for synthetic patterns
            for pattern in synthetic_patterns:
                if pattern in task_id_lower:
                    self.logger.debug(f"🔍 Task {task_id} identified as SYNTHETIC (pattern: {pattern})")
                    return 'synthetic'
            
            # Check for organic patterns
            for pattern in organic_patterns:
                if pattern in task_id_lower:
                    self.logger.debug(f"🔍 Task {task_id} identified as ORGANIC (pattern: {pattern})")
                    return 'organic'
            
            # Method 2: Prompt content analysis (fallback)
            if prompt:
                prompt_lower = prompt.lower()
                
                # Synthetic prompts often contain test-like content
                synthetic_keywords = ['test', 'benchmark', 'validation', 'duel', 'challenge']
                organic_keywords = ['user', 'real', 'production', 'live', 'actual']
                
                synthetic_count = sum(1 for keyword in synthetic_keywords if keyword in prompt_lower)
                organic_count = sum(1 for keyword in organic_keywords if keyword in prompt_lower)
                
                if synthetic_count > organic_count:
                    self.logger.debug(f"🔍 Task {task_id} identified as SYNTHETIC (prompt analysis)")
                    return 'synthetic'
                elif organic_count > synthetic_count:
                    self.logger.debug(f"🔍 Task {task_id} identified as ORGANIC (prompt analysis)")
                    return 'organic'
            
            # Method 3: Default fallback (safety first)
            self.logger.debug(f"🔍 Task {task_id} type UNKNOWN - defaulting to SYNTHETIC (300s cooldown)")
            return 'synthetic'  # Default to synthetic for safety
            
        except Exception as e:
            self.logger.warning(f"⚠️ Traffic type detection failed for task {task_id}: {e}")
            return 'synthetic'  # Default to synthetic for safety
    
    def get_traffic_specific_cooldown(self, traffic_type: str) -> int:
        """
        Get the appropriate cooldown duration based on traffic type.
        Returns cooldown duration in seconds.
        """
        if traffic_type == 'organic':
            # DEPRECATED: Hardcoded 120s - now using NETWORK_DELAY_TIME_BUFFER * 2 constant
            # cooldown = self.config.get('organic_traffic_cooldown', 120)  # 120s for organic
            cooldown = self.config.get('organic_traffic_cooldown', ORGANIC_TRAFFIC_COOLDOWN)  # 120s for organic
            self.logger.debug(f"🌱 Organic traffic detected - using {cooldown}s cooldown")
        else:
            # DEPRECATED: Hardcoded 300s - now using FAILED_VALIDATOR_DELAY constant
            # cooldown = self.config.get('synthetic_traffic_cooldown', 300)  # 300s for synthetic
            cooldown = self.config.get('synthetic_traffic_cooldown', SYNTHETIC_TRAFFIC_COOLDOWN)  # 300s for synthetic
            self.logger.debug(f"🧪 Synthetic traffic detected - using {cooldown}s cooldown")
        
        return cooldown
    
    def get_cooldown_status(self, validator: ValidatorState) -> str:
        """
        Get human-readable cooldown status for a validator.
        
        Args:
            validator: The validator to check
            
        Returns:
            Human-readable cooldown status string
        """
        status_parts = []
        
        # Check cooldown status (FIX 2: Use the right cooldown field)
        # if validator.cooldown_until:
        if validator.validator_enforced_cooldown_until:
            current_time = time.time()
            if current_time >= validator.validator_enforced_cooldown_until:
                status_parts.append("Cooldown expired")
            else:
                remaining = int(validator.validator_enforced_cooldown_until - current_time)
                if remaining < 60:
                    status_parts.append(f"Cooldown: {remaining}s remaining")
                elif remaining < 3600:
                    status_parts.append(f"Cooldown: {remaining//60}m {remaining%60}s remaining")
                else:
                    hours = remaining // 3600
                    minutes = (remaining % 3600) // 60
                    status_parts.append(f"Cooldown: {hours}h {minutes}m remaining")
        else:
            status_parts.append("No cooldown")
        
        # DEPRECATED: Validation lock status check removed - now using MIN_TASK_INTERVAL constant for rate limiting
        # Check validation lock status
        # if validator.validation_locked_until:
        #     current_time = time.time()
        #     if current_time >= validator.validation_locked_until:
        #         status_parts.append("Validation lock expired")
        #     else:
        #         remaining = int(validator.validation_locked_until - current_time)
        #         if remaining < 60:
        #         status_parts.append(f"Validation locked: {remaining}s remaining")
        #         elif remaining < 3600:
        #         status_parts.append(f"Validation locked: {remaining//60}m {remaining%60}s remaining")
        #     else:
        #         hours = remaining // 3600
        #         minutes = (remaining % 3600) // 60
        #         status_parts.append(f"Validation locked: {hours}h {minutes}m remaining")
        
        # Add violation count if any (FIX 2: Use only one violation field per violation type)
        # if validator.cooldown_violations > 0:
        if validator.validator_reported_violations > 0:
            status_parts.append(f"Violations: {validator.validator_reported_violations}")
        
        # Check emergency blacklist status (CRITICAL FIX)
        if validator.emergency_blacklist_until:
            current_time = time.time()
            if current_time >= validator.emergency_blacklist_until:
                status_parts.append("Emergency blacklist expired")
            else:
                remaining = int(validator.emergency_blacklist_until - current_time)
                if remaining < 60:
                    status_parts.append(f"EMERGENCY BLACKLIST: {remaining}s remaining")
                elif remaining < 3600:
                    status_parts.append(f"EMERGENCY BLACKLIST: {remaining//60}m {remaining%60}s remaining")
                else:
                    hours = remaining // 3600
                    minutes = (remaining % 3600) // 60
                    status_parts.append(f"EMERGENCY BLACKLIST: {hours}h {minutes}m remaining")
        
        return " | ".join(status_parts) if status_parts else "Available"

    def get_detailed_cooldown_report(self, validator: ValidatorState) -> Dict[str, Any]:
        """Get detailed cooldown status report for debugging cooldown issues"""
        current_time = time.time()

        report = {
            "uid": validator.uid,
            "is_available": self.is_validator_available(validator),
            "last_submit_time": validator.last_submit_time,
            "time_since_last_submit": current_time - validator.last_submit_time if validator.last_submit_time else None,
            "cooldowns": {},
            "violations": validator.validator_reported_violations,
            "recently_processed_tasks": len(getattr(self, '_recently_processed_tasks', {}))
        }

        # Check all cooldown types
        cooldown_types = [
            ("emergency_blacklist_until", "Emergency Blacklist"),
            ("validator_enforced_cooldown_until", "Validator Enforced"),
            ("miner_cooldown_until", "Miner Local"),
            ("cooldown_until", "Legacy Cooldown")
            # DEPRECATED: Validation lock removed - now using MIN_TASK_INTERVAL constant for rate limiting
            # ("validation_locked_until", "Validation Lock")
        ]

        for attr_name, display_name in cooldown_types:
            cooldown_time = getattr(validator, attr_name, None)
            if cooldown_time and current_time < cooldown_time:
                remaining = cooldown_time - current_time
                report["cooldowns"][display_name] = {
                    "remaining_seconds": remaining,
                    "expires_at": cooldown_time,
                    "human_readable": f"{remaining:.1f}s"
                }

        # Check rapid submission status
        if validator.last_submit_time:
            # DEPRECATED: Hardcoded 300s - now using FAILED_VALIDATOR_DELAY constant
            # task_cooldown = self.config.get('generation.task_cooldown', 300)
            task_cooldown = self.config.get('generation.task_cooldown', FAILED_VALIDATOR_DELAY)
            throttle_period = self.config.get('generation.throttle_period', THROTTLE_PERIOD)
            min_time_required = validator.last_submit_time + task_cooldown - throttle_period

            report["rapid_submission_check"] = {
                "task_cooldown": task_cooldown,
                "throttle_period": throttle_period,
                "min_time_required": min_time_required,
                "current_time": current_time,
                "would_trigger_rapid_submission": min_time_required > current_time,
                "time_until_ok": max(0, min_time_required - current_time),
                "last_submit_age": current_time - validator.last_submit_time
            }

        return report
    
    def _check_validators_needing_monitoring(self):
        """
        Check for validators that need extended monitoring due to persistent issues.
        DYNAMICALLY adjusts monitoring thresholds based on system health.
        """
        current_time = time.time()
        monitoring_count = 0
        
        # DYNAMIC: Calculate monitoring thresholds based on overall system health
        total_validators = len(self.validators)
        active_validators_count = len([v for v in self.validators.values() if v.is_active])
        system_health_ratio = active_validators_count / total_validators if total_validators > 0 else 0
        
        # Adjust thresholds based on system health
        if system_health_ratio < 0.3:  # Less than 30% validators active
            violation_threshold = 50  # Lower threshold for critical system state
            monitoring_threshold = 25
            self.logger.warning(f"🚨 CRITICAL SYSTEM STATE: Only {system_health_ratio:.1%} validators active")
            self.logger.warning(f"   Lowering monitoring thresholds: violations > {violation_threshold}, monitoring > {monitoring_threshold}")
        elif system_health_ratio < 0.6:  # Less than 60% validators active
            violation_threshold = 75  # Medium threshold for degraded system state
            monitoring_threshold = 40
            self.logger.warning(f"⚠️ DEGRADED SYSTEM STATE: Only {system_health_ratio:.1%} validators active")
            self.logger.warning(f"   Adjusting monitoring thresholds: violations > {violation_threshold}, monitoring > {monitoring_threshold}")
        else:  # Healthy system state
            violation_threshold = 100  # Standard threshold for healthy system
            monitoring_threshold = 50
            self.logger.info(f"✅ HEALTHY SYSTEM STATE: {system_health_ratio:.1%} validators active")
            self.logger.info(f"   Using standard monitoring thresholds: violations > {violation_threshold}, monitoring > {monitoring_threshold}")
        
        for validator in self.validators.values():
            # Check for validators with persistently high violations (DYNAMIC threshold)
            if (validator.validator_reported_violations > violation_threshold and 
                not validator.emergency_blacklist_until):
                
                self.logger.warning(f"⚠️ UID {validator.uid} needs monitoring: {validator.validator_reported_violations} violations")
                self.logger.warning(f"   DYNAMIC threshold: {violation_threshold} (system health: {system_health_ratio:.1%})")
                
                # DYNAMIC: Apply immediate cooldown for critical validators in degraded system
                if system_health_ratio < 0.6 and validator.validator_reported_violations > violation_threshold * 1.5:
                    # DEPRECATED: Hardcoded 600s - now using GENERATION_ERROR_DELAY * 2 constant
                    # immediate_cooldown = time.time() + 600  # 10 minutes
                    immediate_cooldown = time.time() + (GENERATION_ERROR_DELAY * 2)  # 10 minutes
                    # FIX 2: Use only one cooldown field per cooldown type
                    # validator.cooldown_until = immediate_cooldown
                    # FIX: Use safe cooldown setting method
                    self._safe_set_cooldown(validator, immediate_cooldown)
                    self.logger.error(f"🚨 IMMEDIATE cooldown applied to UID {validator.uid}: 10 minutes")
                    self.logger.error(f"   Critical validator in degraded system - protecting remaining validators")
                
                monitoring_count += 1
            
            # Check for validators that were recently reset but still have issues (DYNAMIC threshold)
            if (validator.validator_reported_violations > monitoring_threshold and 
                validator.last_violation_check and
                # DEPRECATED: Hardcoded 300s - now using FAILED_VALIDATOR_DELAY constant
                # current_time - validator.last_violation_check > 300):  # 5 minutes
                current_time - validator.last_violation_check > FAILED_VALIDATOR_DELAY):  # 5 minutes
                
                self.logger.warning(f"⚠️ UID {validator.uid} showing persistent issues after reset")
                self.logger.warning(f"   Violations: {validator.validator_reported_violations}, DYNAMIC threshold: {monitoring_threshold}")
                validator.last_violation_check = current_time
                monitoring_count += 1
        
        if monitoring_count > 0:
            self.logger.warning(f"⚠️ {monitoring_count} validators need extended monitoring (DYNAMIC thresholds)")
            self.logger.warning(f"   System health: {system_health_ratio:.1%}, Active: {active_validators_count}/{total_validators}")
        
        return monitoring_count
    
    def _analyze_violation_trend(self, validator: ValidatorState) -> str:
        """
        Analyze the trend of violations for a validator.
        
        Args:
            validator: The validator to analyze
            
        Returns:
            Trend analysis: 'increasing', 'decreasing', 'stable_high', 'stable_low', 'unknown'
        """
        if not hasattr(validator, 'violation_history') or len(validator.violation_history) < 3:
            return 'unknown'
        
        # Get last 3 violations
        recent_violations = [v['violations'] for v in validator.violation_history[-3:]]
        
        # Calculate trend
        if len(recent_violations) >= 3:
            # Simple trend analysis
            if recent_violations[-1] > recent_violations[-2] > recent_violations[-3]:
                return 'increasing'
            elif recent_violations[-1] < recent_violations[-2] < recent_violations[-3]:
                return 'decreasing'
            elif all(v > 500 for v in recent_violations):
                return 'stable_high'
            elif all(v < 100 for v in recent_violations):
                return 'stable_low'
            else:
                return 'stable_high'  # Default to stable_high for mixed patterns
        
        return 'unknown'
    
    def restore_validator_states_from_disk(self):
        """
        Restore validator states from disk if available.
        This method should be called after validators are discovered but before mining starts.
        """
        self.logger.info("🔄 Attempting to restore validator states from disk...")
        
        try:
            saved_states = self.state_persistence.load_validator_states()
            
            if not saved_states:
                self.logger.info("📁 No saved states found - starting with fresh validator states")
                return
            
            restored_count = 0
            violation_count = 0
            blacklisted_count = 0
            cooldown_count = 0
            
            for uid, saved_state in saved_states.items():
                if uid in self.validators:
                    validator = self.validators[uid]
                    
                    # Restore critical state information (FIX 2: Use only one field per type)
                    # validator.cooldown_until = saved_state.get('cooldown_until')
                    # validator.cooldown_violations = saved_state.get('cooldown_violations', 0)
                    validator.throttle_period = saved_state.get('throttle_period', 0)
                    # DEPRECATED: Validation lock removed - now using MIN_TASK_INTERVAL constant for rate limiting
                    # validator.validation_locked_until = saved_state.get('validation_locked_until')
                    validator.emergency_blacklist_until = saved_state.get('emergency_blacklist_until')
                    validator.last_submit_time = saved_state.get('last_submit_time')
                    validator.last_violation_check = saved_state.get('last_violation_check')
                    
                    # FIXED: Restore subnet-compliant cooldown fields
                    validator.validator_enforced_cooldown_until = saved_state.get('validator_enforced_cooldown_until')
                    validator.miner_cooldown_until = saved_state.get('miner_cooldown_until')
                    validator.validator_reported_violations = saved_state.get('validator_reported_violations', 0)
                    validator.pending_cooldown_task_id = saved_state.get('pending_cooldown_task_id')
                    
                    # Restore performance tracking
                    validator.total_tasks_received = saved_state.get('total_tasks_received', 0)
                    validator.total_tasks_submitted = saved_state.get('total_tasks_submitted', 0)
                    validator.total_successful_submissions = saved_state.get('total_successful_submissions', 0)
                    validator.average_score = saved_state.get('average_score', 0.0)
                    
                    # Restore activity state
                    validator.is_active = saved_state.get('is_active', True)
                    
                    # Restore learning history
                    if 'violation_history' in saved_state:
                        validator.violation_history = saved_state['violation_history']
                    if 'buffer_history' in saved_state:
                        validator.buffer_history = saved_state['buffer_history']
                    
                    restored_count += 1
                    
                    # Count different types of restored states
                    if validator.validator_reported_violations > 0:
                        violation_count += 1
                    if validator.emergency_blacklist_until and time.time() < validator.emergency_blacklist_until:
                        blacklisted_count += 1
                        validator.is_active = False  # Ensure blacklisted validators are inactive
                    # if validator.cooldown_until and time.time() < validator.cooldown_until:
                    # FIXED: Count cooldowns using new subnet-compliant fields
                    current_time = time.time()
                    if (validator.validator_enforced_cooldown_until and current_time < validator.validator_enforced_cooldown_until) or \
                       (validator.miner_cooldown_until and current_time < validator.miner_cooldown_until):
                        cooldown_count += 1
                    
                    # Log restoration for validators with critical states
                    if (validator.validator_reported_violations > 50 or 
                        validator.emergency_blacklist_until or 
                        validator.validator_enforced_cooldown_until):
                        
                        status = self.get_cooldown_status(validator)
                        self.logger.warning(f"🔄 Restored UID {uid}: {status}")
                        
                        if validator.validator_reported_violations > 100:
                            remaining_time = ""
                            if validator.emergency_blacklist_until:
                                remaining_seconds = validator.emergency_blacklist_until - time.time()
                                if remaining_seconds > 0:
                                    remaining_time = f" (blacklisted for {remaining_seconds/3600:.1f}h more)"
                            
                            self.logger.error(f"🚨 CRITICAL: UID {uid} has {validator.validator_reported_violations} violations{remaining_time}")
                else:
                    self.logger.debug(f"⚠️ Saved state for UID {uid} found but validator not in current set")
            
            # Summary logging
            self.logger.info(f"✅ Restored {restored_count} validator states from disk")
            if violation_count > 0:
                self.logger.warning(f"⚠️ {violation_count} validators restored with violations")
            if blacklisted_count > 0:
                self.logger.error(f"🚨 {blacklisted_count} validators restored as EMERGENCY BLACKLISTED")
            if cooldown_count > 0:
                self.logger.warning(f"⏳ {cooldown_count} validators restored with active cooldowns")
            
            # Update statistics
            self.stats['validators_restored_from_disk'] = restored_count
            self.stats['violations_restored_from_disk'] = violation_count
            self.stats['blacklists_restored_from_disk'] = blacklisted_count
            
        except Exception as e:
            self.logger.error(f"❌ Failed to restore validator states: {e}")
            self.logger.error(f"   Traceback: {traceback.format_exc()}")
    
    def _check_existing_critical_violations(self):
        """
        🚨 CRITICAL: Check for existing violations immediately after startup
        This method detects and alerts on violations that were restored from disk
        """
        self.logger.info("🚨 CRITICAL: Checking for existing violations after startup...")
        
        critical_violations_found = 0
        total_violations = 0
        
        for uid, validator in self.validators.items():
            if hasattr(validator, 'validator_reported_violations') and validator.validator_reported_violations > 0:
                total_violations += validator.validator_reported_violations
                
                # CRITICAL: Alert on high violation counts
                if validator.validator_reported_violations > 100:
                    self.logger.error(f"🚨 CRITICAL: UID {uid} has {validator.validator_reported_violations} violations!")
                    self.logger.error(f"   Stake: {validator.stake:.1f} TAO")
                    self.logger.error(f"   Trust: {getattr(validator, 'trust', 'N/A')}")
                    self.logger.error(f"   Status: {self.get_cooldown_status(validator)}")
                    critical_violations_found += 1
                    
                    # EMERGENCY: Set immediate cooldown for critical validators
                    if validator.validator_reported_violations > 200:
                        # DEPRECATED: Hardcoded 3600s - now using FAILED_VALIDATOR_DELAY * 4 constant
                        # emergency_cooldown = time.time() + 3600  # 1 hour
                        emergency_cooldown = time.time() + (FAILED_VALIDATOR_DELAY * 4)  # 1 hour
                        # FIX 2: Use only one cooldown field per cooldown type
                        # validator.cooldown_until = emergency_cooldown
                        self.logger.error(f"🚨 EMERGENCY: Set 1-hour cooldown for UID {uid} due to {validator.validator_reported_violations} violations!")
                        
                        # EMERGENCY: Implement blacklist for extreme violations
                        if validator.validator_reported_violations > 300:
                            # DEPRECATED: Hardcoded 7200s - now using FAILED_VALIDATOR_DELAY * 6 constant
                            # blacklist_duration = 7200  # 2 hours
                            blacklist_duration = FAILED_VALIDATOR_DELAY * 6  # 2 hours
                            validator.emergency_blacklist_until = time.time() + blacklist_duration
                            validator.is_active = False
                            self.logger.error(f"🚨 EMERGENCY BLACKLIST: UID {uid} blacklisted for {blacklist_duration/3600:.1f}h due to {validator.validator_reported_violations} violations!")
                
                # WARNING: Alert on moderate violation counts
                elif validator.validator_reported_violations > 50:
                    self.logger.warning(f"⚠️ WARNING: UID {uid} has {validator.validator_reported_violations} violations")
                    self.logger.warning(f"   Status: {self.get_cooldown_status(validator)}")
        
        # SUMMARY: Report total violations found
        if total_violations > 0:
            self.logger.error(f"🚨 CRITICAL SUMMARY: Found {total_violations} total violations across {critical_violations_found} critical validators!")
            self.logger.error(f"   This indicates a serious security issue that requires immediate attention!")
            
            # EMERGENCY: Update statistics
            self.stats['cooldown_violations_total'] = total_violations
            self.stats['critical_violations_detected'] = critical_violations_found
            
            # EMERGENCY: Save states immediately
            self.save_validator_states_to_disk()
        else:
            self.logger.info("✅ No existing violations found - system is clean")
    
    def _check_runtime_critical_violations(self):
        """
        🚨 CRITICAL: Check for violations during runtime
        This method is called periodically to monitor for new violations
        """
        critical_violations_found = 0
        total_violations = 0
        
        for uid, validator in self.validators.items():
            if hasattr(validator, 'validator_reported_violations') and validator.validator_reported_violations > 0:
                total_violations += validator.validator_reported_violations
                
                # CRITICAL: Alert on high violation counts
                if validator.validator_reported_violations > 100:
                    critical_violations_found += 1
                    
                    # EMERGENCY: Set immediate cooldown for critical validators
                    # DEPRECATED: cooldown_until field check - now using validator_enforced_cooldown_until
                    # if validator.validator_reported_violations > 200 and not validator.cooldown_until:
                    if validator.validator_reported_violations > 200 and not validator.validator_enforced_cooldown_until:
                        # DEPRECATED: Hardcoded 1800s - now using FAILED_VALIDATOR_DELAY * 2 constant
                        # emergency_cooldown = time.time() + 1800  # 30 minutes
                        emergency_cooldown = time.time() + (FAILED_VALIDATOR_DELAY * 2)  # 30 minutes
                        # FIX 2: Use only one cooldown field per cooldown type
                        # validator.cooldown_until = emergency_cooldown
                        # FIX: Use safe cooldown setting method
                        self._safe_set_cooldown(validator, emergency_cooldown)
                        self.logger.error(f"🚨 EMERGENCY: Set 30-minute cooldown for UID {uid} due to {validator.validator_reported_violations} violations!")
                        
                        # EMERGENCY: Implement blacklist for extreme violations
                        if validator.validator_reported_violations > 300 and not validator.emergency_blacklist_until:
                            # DEPRECATED: Hardcoded 3600s - now using FAILED_VALIDATOR_DELAY * 4 constant
                            # blacklist_duration = 3600  # 1 hour
                            blacklist_duration = FAILED_VALIDATOR_DELAY * 4  # 1 hour
                            validator.emergency_blacklist_until = time.time() + blacklist_duration
                            validator.is_active = False
                            self.logger.error(f"🚨 EMERGENCY BLACKLIST: UID {uid} blacklisted for {blacklist_duration/3600:.1f}h due to {validator.validator_reported_violations} violations!")
        
        # Update statistics if violations found
        if total_violations > 0:
            self.stats['cooldown_violations_total'] = total_violations
            self.stats['critical_violations_detected'] = critical_violations_found
            
            # Log summary every 5 minutes
            self.logger.warning(f"🚨 RUNTIME VIOLATION CHECK: {total_violations} total violations across {critical_violations_found} critical validators")
    
    def save_validator_states_to_disk(self):
        """
        Save current validator states to disk.
        """
        try:
            success = self.state_persistence.save_validator_states(self.validators)
            if success:
                self.stats['validator_states_saved'] = self.stats.get('validator_states_saved', 0) + 1
            else:
                self.stats['validator_state_save_failures'] = self.stats.get('validator_state_save_failures', 0) + 1
        except Exception as e:
            self.logger.error(f"❌ Failed to save validator states: {e}")
            self.stats['validator_state_save_failures'] = self.stats.get('validator_state_save_failures', 0) + 1
    
    def _register_shutdown_handlers(self):
        """
        Register handlers to save state on graceful shutdown.
        """
        def shutdown_handler(signum, frame):
            self.logger.info(f"🛑 Received shutdown signal {signum} - saving validator states...")
            try:
                self.save_validator_states_to_disk()
                self.logger.info("💾 Validator states saved successfully on shutdown")
            except Exception as e:
                self.logger.error(f"❌ Failed to save states on shutdown: {e}")
            
            # Exit gracefully
            import sys
            sys.exit(0)
        
        def atexit_handler():
            self.logger.info("🛑 Script exiting - saving validator states...")
            try:
                self.save_validator_states_to_disk()
                self.logger.info("💾 Validator states saved successfully on exit")
            except Exception as e:
                self.logger.error(f"❌ Failed to save states on exit: {e}")
        
        # Register signal handlers
        signal.signal(signal.SIGTERM, shutdown_handler)
        signal.signal(signal.SIGINT, shutdown_handler)
        
        # Register atexit handler
        atexit.register(atexit_handler)
        
        self.logger.info("🔒 Registered shutdown handlers for state persistence")
    
    def _recover_from_emergency_state(self, validator: ValidatorState):
        """
        FIXED: Emergency state recovery logic for 100% compliance.
        Gracefully recovers validators from emergency states.
        
        Args:
            validator: The validator to recover
        """
        current_time = time.time()
        recovered = False
        
        # Recover from emergency blacklist
        if validator.emergency_blacklist_until and current_time >= validator.emergency_blacklist_until:
            validator.emergency_blacklist_until = None
            validator.is_active = True
            self.logger.info(f"✅ UID {validator.uid} recovered from emergency blacklist")
            recovered = True
        
        # DEPRECATED: Validation lock recovery removed - now using MIN_TASK_INTERVAL constant for rate limiting
        # Recover from validation lock
        # if validator.validation_locked_until and current_time >= validator.validation_locked_until:
        #     validator.validation_locked_until = None
        #     self.logger.info(f"✅ UID {validator.uid} recovered from validation lock")
        #     recovered = True
        
        # Recover from expired cooldowns
        if validator.validator_enforced_cooldown_until and current_time >= validator.validator_enforced_cooldown_until:
            validator.validator_enforced_cooldown_until = None
            validator.pending_cooldown_task_id = None
            self.logger.info(f"✅ UID {validator.uid} recovered from validator-enforced cooldown")
            recovered = True
        
        if validator.miner_cooldown_until and current_time >= validator.miner_cooldown_until:
            validator.miner_cooldown_until = None
            self.logger.info(f"✅ UID {validator.uid} recovered from miner cooldown")
            recovered = True
        
        # Track recovery statistics
        if recovered:
            self.stats['emergency_state_recoveries'] = self.stats.get('emergency_state_recoveries', 0) + 1
            self.logger.info(f"🔄 UID {validator.uid} emergency state recovery completed")
        
        return recovered

    # async def pull_task_from_validator(self, validator: ValidatorState) -> Optional[TaskRecord]:
    #     """Pull task from a specific validator with deduplication"""
    #     try:
    #         # Check if TRELLIS server is available for priority access
    #         # CRITICAL: Don't pull tasks if server is unavailable - we can't process them!
    #         try:
    #             server_status = self.priority_coordinator.check_server_status()
    #             if not server_status.get("available", False):
    #                 status = server_status.get('status', 'unknown')
    #                 error = server_status.get('error', 'unknown error')
    #                 self.logger.warning(f"⏳ TRELLIS server unavailable (status: {status}, error: {error}) - SKIPPING task pull")
    #                 self.stats['server_unavailable_skips'] = self.stats.get('server_unavailable_skips', 0) + 1
    #                 return None  # Don't pull tasks when server is unavailable
    #             else:
    #                 self.logger.debug(f"✅ TRELLIS server available (status: {server_status.get('status', 'unknown')})")
    #         except Exception as e:
    #             self.logger.warning(f"⚠️ Exception checking TRELLIS server status: {e} - SKIPPING task pull")
    #             self.stats['server_status_check_errors'] = self.stats.get('server_status_check_errors', 0) + 1
    #             return None  # Don't pull tasks when we can't check server status
    #         if not self.is_validator_available(validator):
    #             return None
            
    #         self.logger.debug(f"📡 Pulling from UID {validator.uid} ({validator.stake:.1f} TAO)")
            
    #         # Import protocol
    #         from neurons.common.protocol import PullTask
            
    #         # Create task pull request
    #         synapse = PullTask()
    #         synapse.timeout = self.config['submission_timeout']
            
    #         # Get neuron info
    #         if validator.uid >= len(self.metagraph.neurons):
    #             return None
            
    #         neuron = self.metagraph.neurons[validator.uid]
            
    #         start_time = time.time()
            
    #         # Query the validator
    #         response = await self.dendrite.forward(
    #             axons=[neuron.axon_info],
    #             synapse=synapse,
    #             timeout=self.config['submission_timeout']
    #         )
            
    #         query_time = time.time() - start_time
    #         validator.last_task_pull = time.time()
            
    #         task = None
    #         if response and len(response) > 0:
    #             resp = response[0]

    #             if hasattr(resp, 'task') and resp.task and resp.task.prompt:
    #                 # SHARED TASK TRACKING: Check if this task is already being processed by another instance
    #                 if self.config.get('enable_task_tracking', True) and not self.config.get('disable_task_tracking', False):
    #                     if not self.db.acquire_task_lock(resp.task.id, validator.uid, self.instance_id, timeout_minutes=2):
    #                         self.logger.info(f"⏭️ Task {resp.task.id} already being processed by another instance - skipping UID {validator.uid}")
    #                         return None
                    
    #                 # Check for duplicates with detailed analysis (only if enabled)
    #                 if self.config.get('enable_duplicate_checking', True):
    #                     if self.db.is_duplicate_prompt(resp.task.prompt, validator.uid, self.config['duplicate_check_hours']):
    #                         # Release the task lock since we're not processing this task
    #                         if self.config.get('enable_task_tracking', True) and not self.config.get('disable_task_tracking', False):
    #                             self.db.release_task_lock(resp.task.id, self.instance_id, status='skipped_duplicate')
                            
    #                         # Get analysis for this validator to understand why it's being skipped
    #                         analysis = self.db.get_duplicate_analysis(validator.uid, 6)  # Last 6 hours
    #                         self.logger.info(f"⏭️ Skipping duplicate from UID {validator.uid}: '{resp.task.prompt[:50]}...'")
    #                         self.logger.info(f"   Analysis: {analysis['successful_tasks']}/{analysis['total_tasks_pulled']} successful, {analysis['failed_tasks']} failed, {analysis['unprocessed_tasks']} unprocessed")
    #                         return None
    #                 else:
    #                     self.logger.debug(f"🔄 Duplicate checking disabled - processing prompt from UID {validator.uid}")
                    
    #                 # Update validator state
    #                 validator.total_tasks_pulled += 1
    #                 validator.last_task_received = time.time()
                    
    #                 # Create task record with response time tracking
    #                 prompt_hash = hashlib.sha256(resp.task.prompt.encode()).hexdigest()
    #                 response_received_time = time.time()
                    
    #                 task = TaskRecord(
    #                     task_id=resp.task.id,
    #                     prompt=resp.task.prompt,
    #                     prompt_hash=prompt_hash,
    #                     validator_uid=validator.uid,
    #                     validator_hotkey=validator.hotkey,
    #                     validator_stake=validator.stake,
    #                     validation_threshold=getattr(resp, 'validation_threshold', 0.6),
    #                     pulled_at=response_received_time
    #                 )
                    
    #                 # Add to recent prompts tracking
    #                 self.db.add_recent_prompt(resp.task.prompt, validator.uid)
                    
    #                 self.logger.info(f"✅ New task from UID {validator.uid}: '{task.prompt[:50]}...'")
    #                 self.logger.info(f"   Threshold: {task.validation_threshold}, Query time: {query_time:.2f}s")
    #                 self.logger.info(f"   🔒 Task lock acquired: {resp.task.id}")
                    
    #                 self.stats['tasks_pulled'] += 1

    #             else:
    #                 self.logger.debug(f"⚠️ No task from UID {validator.uid}")
    #                 return None
                
    #             # Enhanced cooldown and validation tracking - CRITICAL FIX
    #             if hasattr(resp, 'cooldown_until') and resp.cooldown_until:
    #                 # CRITICAL: Update validator cooldown from response and respect it
    #                 old_cooldown = validator.cooldown_until
    #                 validator.cooldown_until = resp.cooldown_until
                    
    #                 # Calculate remaining cooldown time
    #                 current_time = time.time()
    #                 if resp.cooldown_until > current_time:
    #                     remaining_cooldown = resp.cooldown_until - current_time
    #                     self.logger.warning(f"🚨 CRITICAL: Validator UID {validator.uid} enforced cooldown: {remaining_cooldown:.1f}s remaining")
    #                     self.logger.warning(f"   Previous cooldown: {old_cooldown}, New cooldown: {resp.cooldown_until}")
                        
    #                     # Set emergency cooldown to prevent further violations
    #                     self._set_emergency_cooldown(validator, resp.cooldown_until, "Validator enforced cooldown")
    #                 else:
    #                     self.logger.info(f"✅ Validator UID {validator.uid} cooldown cleared: {resp.cooldown_until}")
                
    #             if hasattr(resp, 'cooldown_violations') and resp.cooldown_violations:
    #                 # Track cooldown violations from validator - CRITICAL
    #                 old_violations = validator.cooldown_violations
    #                 validator.cooldown_violations = resp.cooldown_violations
                    
    #                 if resp.cooldown_violations > 0:
    #                     self.logger.error(f"🚨 CRITICAL: Validator UID {validator.uid} reported {resp.cooldown_violations} cooldown violations!")
                        
    #                     # Check if violations increased significantly
    #                     if resp.cooldown_violations > old_violations + 10:
    #                         self.logger.error(f"🚨 Violations increased by {resp.cooldown_violations - old_violations} - implementing emergency measures")
    #                         self._handle_critical_violations(validator, resp.cooldown_violations)
                        
    #                     # Check if we're over the threshold
    #                     violation_threshold = self.config.get('critical_violation_threshold', 100)
    #                     if resp.cooldown_violations > violation_threshold:
    #                         self.logger.error(f"🚨 UID {validator.uid} exceeds violation threshold ({violation_threshold}) - implementing blacklist")
    #                         self._blacklist_validator_temporarily(validator, resp.cooldown_violations)
                
    #             if hasattr(resp, 'throttle_period') and resp.throttle_period:
    #                 # Update throttle period from validator
    #                 validator.throttle_period = resp.throttle_period
    #                 self.logger.debug(f"⏱️ Validator UID {validator.uid} throttle period: {resp.throttle_period}s")

    #             return task if isinstance(task, TaskRecord) else None
                
    #         else:
    #             self.logger.debug(f"❌ No response from UID {validator.uid}")
                
    #             # Set cooldown for no response (validator might be overloaded)
    #             validator_cooldown = self.config.get('validator_error_cooldown', 45)
    #             self.set_validator_cooldown(validator, validator_cooldown, "No response received")
                
    #             return None
        
    #     except Exception as e:
    #         self.logger.error(f"❌ Error pulling from UID {validator.uid}: {e}")
            
    #         # Set cooldown for network/validator errors
    #         network_cooldown = self.config.get('network_error_cooldown', 30)
    #         self.set_validator_cooldown(validator, network_cooldown, f"Network error: {str(e)[:50]}")
            
    #         return None

    async def pull_task_from_validator(self, validator: ValidatorState) -> Optional[TaskRecord]:
        """Pull task from a specific validator with enhanced reactive cooldown system"""
        try:
            # Check if TRELLIS server is available for priority access
            # CRITICAL: Don't pull tasks if server is unavailable - we can't process them!
            try:
                server_status = self.priority_coordinator.check_server_status()
                if not server_status.get("available", False):
                    status = server_status.get('status', 'unknown')
                    error = server_status.get('error', 'unknown error')
                    self.logger.warning(f"⏳ TRELLIS server unavailable (status: {status}, error: {error}) - SKIPPING task pull")
                    self.stats['server_unavailable_skips'] = self.stats.get('server_unavailable_skips', 0) + 1
                    return None  # Don't pull tasks when server is unavailable
                else:
                    self.logger.debug(f"✅ TRELLIS server available (status: {server_status.get('status', 'unknown')})")
            except Exception as e:
                self.logger.warning(f"⚠️ Exception checking TRELLIS server status: {e} - SKIPPING task pull")
                self.stats['server_status_check_errors'] = self.stats.get('server_status_check_errors', 0) + 1
                return None  # Don't pull tasks when we can't check server status
            
            
            # CRITICAL: Enhanced cooldown checking before task pull - prevents validator from sending tasks when miner is on cooldown
            cooldown_status = self._check_validator_cooldown_state(validator)
            if not cooldown_status['available']:
                self.logger.debug(f"⏳ Validator UID {validator.uid} not available: {cooldown_status['reason']}")
                self.logger.debug(f"   Recommendation: {cooldown_status['recommendation']}")
                # CRITICAL: Don't even attempt to pull tasks when on cooldown - prevents validator from wasting resources
                return None

            # SECONDARY: Check if validator is available using existing method (this should be redundant but kept for safety)
            if not self.is_validator_available(validator):
                return None

            # ADDITIONAL: Check rapid submission timing before pulling to prevent validator from sending tasks we can't process
            if self._check_rapid_submission_timing_only(validator):
                self.logger.debug(f"⏳ Validator UID {validator.uid} would trigger rapid submission, enforcing buffer time - skipping task pull")
                print(f"⏳ Validator UID {validator.uid} would trigger rapid submission, enforcing buffer time - skipping task pull")
                return None
            


            self.logger.debug(f"📡 Pulling from UID {validator.uid} ({validator.stake:.1f} TAO)")
            
            # Import protocol
            from neurons.common.protocol import PullTask
            
            # Create task pull request
            synapse = PullTask()
            # synapse.timeout = self.config['submission_timeout']
            
            # Get neuron info
            if validator.uid >= len(self.metagraph.neurons):
                return None
            
            neuron = self.metagraph.neurons[validator.uid]
            
            start_time = time.time()
            
            # Query the validator
            # response = await self.dendrite.forward(
            #     axons=[neuron.axon_info],
            #     synapse=synapse,
            #     timeout=self.config['submission_timeout']
            # )
            
            response = typing.cast(
                PullTask,
                await self.dendrite.call(
                    target_axon=neuron.axon_info,
                    synapse=synapse,
                    deserialize=False,
                    timeout=self.config['submission_timeout']
                )
            )

            if response.dendrite.status_code != 200:
                self.logger.error(f"❌ Failed to get task from [{metagraph.hotkeys[validator_uid]}]. Reason: {response.dendrite.status_message}.")
                validator.validator_enforced_cooldown_until = int(time.time()) + FAILED_VALIDATOR_DELAY
                return None 
            query_time = time.time() - start_time
            validator.last_task_pull = time.time()
            

            if response and len(response) > 0:
                # resp = response[0]
                resp = response

                # ENHANCED: Post-task state synchronization with graceful degradation
                response_data = {}

                if hasattr(resp, 'throttle_period'):
                    response_data['throttle_period'] = resp.throttle_period
                
                # 🚨 CRITICAL: IMMEDIATE VIOLATION DETECTION AND EMERGENCY RESPONSE
                if hasattr(resp, 'cooldown_violations') and resp.cooldown_violations:
                    
                    response_data['cooldown_violations'] = resp.cooldown_violations
                    # ALERT: If validator reports violations but cooldown is 0, there's a problem
                    if resp.cooldown_violations > 0 and original_cooldown == 0:
                        self.logger.warning(f"🚨 VALIDATOR COOLDOWN MISMATCH: {resp.cooldown_violations} violations but cooldown_until=0 for UID {validator.uid}")

                    old_violations = getattr(validator, 'cooldown_violations', 0)
                    new_violations = resp.cooldown_violations
                    self.logger.info(f"🔍 DEBUG: cooldown_violations found: {resp.cooldown_violations}")
                    
                    # IMMEDIATE: Log critical violation detection
                    self.logger.error(f"🚨 CRITICAL: Validator UID {validator.uid} reported {new_violations} cooldown violations!")
                    
                    if new_violations > old_violations:
                        violation_increase = new_violations - old_violations
                        self.logger.error(f"🚨 VIOLATIONS INCREASED by {violation_increase} for UID {validator.uid}: {old_violations} → {new_violations}")
                        
                        # EMERGENCY: Check for critical violation thresholds
                        critical_threshold = self.config.get('critical_violation_threshold', VIOLATION_INCREASE_DELTA)
                        if new_violations > critical_threshold:
                            self.logger.error(f"🚨 EMERGENCY: UID {validator.uid} exceeds critical threshold ({critical_threshold}) - implementing immediate blacklist!")
                            self._blacklist_validator_temporarily(validator, new_violations)
                        
                        # EMERGENCY: Check for rapid violation increase
                        if violation_increase > 20:
                            self.logger.error(f"🚨 EMERGENCY: UID {validator.uid} violations increased by {violation_increase} - implementing emergency measures!")
                            self._handle_critical_violations(validator, new_violations)
                        
                        # EMERGENCY: Set immediate cooldown for high violations
                        if new_violations > 50:
                            # DEPRECATED: Hardcoded 1800s - now using FAILED_VALIDATOR_DELAY * 2 constant
                            # emergency_cooldown = time.time() + 1800  # 30 minutes
                            emergency_cooldown = time.time() + (FAILED_VALIDATOR_DELAY * 2)  # 30 minutes
                            # validator.cooldown_until = emergency_cooldown
                            # FIX: Use safe cooldown setting method
                            self._safe_set_cooldown(validator, emergency_cooldown)
                            self.logger.error(f"🚨 EMERGENCY: Set 30-minute cooldown for UID {validator.uid} due to {new_violations} violations!")
                    
                    # UPDATE: Immediately update local violation count for real-time tracking
                    # DEPRECATED: cooldown_violations field - now using validator_reported_violations
                    # validator.cooldown_violations = new_violations
                    validator.validator_reported_violations = new_violations
                    
                    # STATS: Update violation statistics
                    self.stats['cooldown_violations_total'] = max(self.stats.get('cooldown_violations_total', 0), new_violations)
                    self.stats['critical_violations_detected'] = self.stats.get('critical_violations_detected', 0) + 1
                    
                    # ALERT: Log detailed violation analysis
                    self.logger.error(f"🚨 VIOLATION ANALYSIS for UID {validator.uid}:")
                    self.logger.error(f"   Current violations: {new_violations}")
                    self.logger.error(f"   Previous violations: {old_violations}")
                    self.logger.error(f"   Increase: {new_violations - old_violations}")
                    self.logger.error(f"   Stake: {validator.stake:.1f} TAO")
                    self.logger.error(f"   Trust: {getattr(validator, 'trust', 'N/A')}")
                
                else:
                    self.logger.info(f"🔍 DEBUG: no cooldown_violations found")

                # ENHANCED: Check for validator-enforced cooldowns
                if hasattr(resp, 'cooldown_until') and resp.cooldown_until:
                    original_cooldown = resp.cooldown_until
                    # FIX 3: Use the original cooldown, don't rely on traffic because they are used by the validator
                    # if original_cooldown > 0:
                    #     # DEPRECATED: Hardcoded 10s - now using NETWORK_DELAY_TIME_BUFFER constant
                    #     # response_data['cooldown_until'] = original_cooldown + 10  # Add 1 second to ensure we pass the cooldown
                    #     response_data['cooldown_until'] = original_cooldown + NETWORK_DELAY_TIME_BUFFER  # Add NETWORK_DELAY_TIME_BUFFER seconds to ensure we pass the cooldown
                    #     self.logger.debug(f"🛡️ Added 1s buffer to cooldown: {original_cooldown} → {response_data['cooldown_until']}")
                    # else:
                    response_data['cooldown_until'] = original_cooldown 
                    # FIX: Use safe cooldown setting method
                    # DEPRECATED: Hardcoded 3s - now using NETWORK_DELAY_TIME_BUFFER constant
                    # self._safe_set_cooldown(validator, original_cooldown + 3)
                    self._safe_set_cooldown(validator, original_cooldown + 5)
                    self.logger.debug(f"🛡️ Using original cooldown from validator: {original_cooldown}")

                    current_time = time.time()
                    if resp.cooldown_until > current_time:
                        remaining_cooldown = resp.cooldown_until - current_time
                        self.logger.warning(f"🚨 CRITICAL: PULLTASK Validator UID {validator.uid} enforced cooldown: {remaining_cooldown:.1f}s remaining")
                        # FIX: Use safe cooldown setting method
                        # FIX 1: Only set emergency cooldowns for actual violations, not normal sync
                        # self._set_emergency_cooldown(validator, resp.cooldown_until, "Validator enforced cooldown")
                    else:
                        self.logger.info(f"✅ Validator UID {validator.uid} cooldown cleared: {resp.cooldown_until}")
                
                if response_data:
                    sync_results = self._synchronize_validator_state(validator, response_data)
                    
                    # Log synchronization results
                    if sync_results['cooldown_updated'] or sync_results['violations_updated']:
                        self.logger.info(f"🔄 State synchronized for UID {validator.uid}")
                        if sync_results['backoff_strategy']:
                            self.logger.info(f"   Backoff strategy: {sync_results['backoff_strategy']}")
                        if sync_results['emergency_actions']:
                            self.logger.info(f"   Emergency actions: {', '.join(sync_results['emergency_actions'])}")
                
                if hasattr(resp, 'task') and resp.task and resp.task.prompt:
                    # Update validator state
                    validator.total_tasks_pulled += 1
                    validator.last_task_received = time.time()
                    
                    # Create task record with response time tracking
                    prompt_hash = hashlib.sha256(resp.task.prompt.encode()).hexdigest()
                    response_received_time = time.time()
                    
                    task = TaskRecord(
                        task_id=resp.task.id,
                        prompt=resp.task.prompt,
                        prompt_hash=prompt_hash,
                        validator_uid=validator.uid,
                        validator_hotkey=validator.hotkey,
                        validator_stake=validator.stake,
                        validation_threshold=getattr(resp, 'validation_threshold', 0.6),
                        pulled_at=response_received_time
                    )
                    
                    # Add to recent prompts tracking
                    self.db.add_recent_prompt(resp.task.prompt, validator.uid)
                    
                    self.logger.info(f"✅ New task from UID {validator.uid}: '{task.prompt[:50]}...'")
                    self.logger.info(f"   Threshold: {task.validation_threshold}, Query time: {query_time:.2f}s")
                    
                    self.stats['tasks_pulled'] += 1

                    self.logger.info(f"🔍 DEBUG: Validator response attributes: {dir(resp)}")
                    self.logger.info(f"🔍 DEBUG: Validator response data: {resp.__dict__}")
                    
                    return task
                else:
                    self.logger.debug(f"⚠️ No task from UID {validator.uid}")
                    return None
            else:
                self.logger.debug(f"❌ No response from UID {validator.uid}")
                return None
        
        except Exception as e:
            self.logger.error(f"❌ Error pulling from UID {validator.uid}: {e}")
            return None

    def get_gold_prompts_from_orchestrator(self, log_count: int = 7) -> Dict[str, Any]:
        """
        Use the EXACT same functions from the continuous orchestrator to get gold prompts.
        This ensures we're measuring the exact same performance.
        
        Args:
            log_count: Number of recent logs to parse (default: 7)
            
        Returns:
            Dictionary of gold prompts in the exact same format as the orchestrator
        """
        print(f"📚 Using EXACT orchestrator functions to get gold prompts from last {log_count} logs...")
        
        # Create a minimal orchestrator instance just for the gold prompt functions
        # We only need the config for the gold prompt parsing functions
        minimal_config = {
            'activate_learning': True,
            'only_log_learning': log_count,
            'log_learning_count': log_count,
            'max_logs_to_parse': log_count,
            'use_vllm': True,
            'vllm_url': 'http://localhost:9002',
            'vllm_model': 'llama-3-2-3b-it'
        }
        
        try:
            # Create orchestrator instance
            
            # Use the EXACT same function the orchestrator uses
            print(f"🔄 Calling orchestrator.parse_current_episode_logs() with {log_count} logs...")
            log_prompts = self.parse_current_episode_logs()
            
            print(f"📊 Parsed {len(log_prompts)} prompts from logs")
            
            # Convert to the format expected by the reproducibility system
            gold_standard_results = {}
            
            for prompt, data in log_prompts.items():
                if 'best_score' in data and data['best_score'] > 0:
                    # Create the method_2_hybrid_example structure that reproducibility system expects
                    gold_standard_results[prompt] = {
                        "method_2_hybrid_example": {
                            "optimized_prompt": data.get('optimized_prompt', prompt),
                            "validation_results": {
                                "validation_engine_score": data['best_score']
                            }
                        }
                    }
            
            print(f"✅ Converted {len(gold_standard_results)} prompts to gold standard format")
            
            # Show top scoring prompts
            if gold_standard_results:
                top_prompts = sorted(
                    gold_standard_results.items(),
                    key=lambda x: x[1]['method_2_hybrid_example']['validation_results']['validation_engine_score'],
                    reverse=True
                )[:5]
                
                print(f"🏆 Top scoring prompts from logs:")
                for i, (prompt, data) in enumerate(top_prompts, 1):
                    score = data['method_2_hybrid_example']['validation_results']['validation_engine_score']
                    print(f"   {i}. Score {score:.4f}: '{prompt[:60]}...'")
            
            return gold_standard_results
            
        except Exception as e:
            print(f"❌ Failed to get gold prompts from orchestrator: {e}")
            import traceback
            traceback.print_exc()
            return {}

    def _clean_vllm_response(self, full_response: str) -> str:
        """Clean vLLM response to extract just the optimized prompt."""
        # Extract just the optimized prompt part (remove explanatory text)
        optimized_prompt = full_response
        
        # Remove common explanatory prefixes
        prefixes_to_remove = [
            "Here's an optimized prompt for 3D generation:",
            "Here's an optimized version of the prompt for 3D generation:",
            "To optimize the prompt for 3D generation, I would suggest the following:",
            "Here's the optimized prompt:",
            "Optimized prompt:",
            "Here's an enhanced version:",
            "Enhanced prompt:",
            "Here's an optimized prompt for 3D generation of a golden statue:",
            "Here's an optimized prompt for 3D generation of a",
            "Here's an optimized prompt for 3D generation:",
            "Here's the optimized prompt:",
            "Optimized prompt:",
            "Enhanced prompt:",
            "Here's an enhanced version:", 
            "Heres an optimized version of the prompt ",
        ]
        
        for prefix in prefixes_to_remove:
            if optimized_prompt.startswith(prefix):
                optimized_prompt = optimized_prompt[len(prefix):].strip()
                break
        
        # Remove ALL quotes from the prompt (both single and double quotes)
        # This prevents shell argument parsing issues
        # optimized_prompt = optimized_prompt.replace('"', '').replace("'", '')
        
        # Additional shell-safe cleanup to prevent bash syntax errors
        self.logger.debug(f"🔧 Cleaning prompt for shell safety...")
        self.logger.debug(f"   Original length: {len(optimized_prompt)} chars")
        
        # Check for quotes safely
        has_quotes = '"' in optimized_prompt or "'" in optimized_prompt
        self.logger.debug(f"   Contains quotes: {has_quotes}")
        
        optimized_prompt = self._clean_prompt_for_shell(optimized_prompt)
        
        self.logger.debug(f"   Cleaned length: {len(optimized_prompt)} chars")
        self.logger.debug(f"   Final prompt: '{optimized_prompt[:100]}...'")
        
        # Clean up the response (same cleaning as original script)
        optimized_prompt = optimized_prompt.replace('\n', ' ').replace('\r', ' ').replace('\t', ' ')
        optimized_prompt = ''.join(char for char in optimized_prompt if ord(char) >= 32 or char == ' ')
        optimized_prompt = ' '.join(optimized_prompt.split())
        
        # Final validation: ensure the prompt is safe for shell execution
        if not self._is_shell_safe(optimized_prompt):
            self.logger.warning(f"⚠️ Prompt still contains shell-unsafe characters, applying aggressive cleaning")
            # Remove all non-alphanumeric characters except spaces, dots, and hyphens
            import re
            optimized_prompt = re.sub(r'[^\w\s\.\-]', '', optimized_prompt)
            optimized_prompt = ' '.join(optimized_prompt.split())
        
        return optimized_prompt

    def _clean_prompt_for_shell(self, prompt: str) -> str:
        """Clean prompt for safe use in shell commands by removing problematic characters."""
        # Remove or replace characters that can cause bash syntax errors
        cleaned_prompt = prompt
        
        # CRITICAL: Fix unmatched quotes first - this is the main cause of bash syntax errors
        # Count quotes and ensure they're balanced
        single_quotes = cleaned_prompt.count("'")
        double_quotes = cleaned_prompt.count('"')
        
        # If we have unmatched quotes, remove them completely to prevent bash errors
        if single_quotes % 2 != 0:  # Odd number of single quotes
            self.logger.warning(f"⚠️ Unmatched single quotes detected ({single_quotes}), removing all single quotes")
            cleaned_prompt = cleaned_prompt.replace("'", "")
        
        if double_quotes % 2 != 0:  # Odd number of double quotes
            self.logger.warning(f"⚠️ Unmatched double quotes detected ({double_quotes}), removing all double quotes")
            cleaned_prompt = cleaned_prompt.replace('"', "")
        
        # Remove parentheses and brackets that can cause bash syntax issues
        cleaned_prompt = cleaned_prompt.replace('(', '').replace(')', '')
        cleaned_prompt = cleaned_prompt.replace('[', '').replace(']', '')
        cleaned_prompt = cleaned_prompt.replace('{', '').replace('}', '')
        
        # Remove or replace other problematic characters
        cleaned_prompt = cleaned_prompt.replace('`', "'")  # Replace backticks with single quotes
        cleaned_prompt = cleaned_prompt.replace('$', 'dollar')  # Replace $ with text
        cleaned_prompt = cleaned_prompt.replace('&', 'and')  # Replace & with text
        cleaned_prompt = cleaned_prompt.replace('|', 'or')  # Replace | with text
        cleaned_prompt = cleaned_prompt.replace(';', '.')  # Replace ; with .
        cleaned_prompt = cleaned_prompt.replace('\\', '/')  # Replace backslashes with forward slashes
        
        # Remove any remaining special shell characters that could cause issues
        import re
        cleaned_prompt = re.sub(r'[^\w\s\.\,\-\_\#]', '', cleaned_prompt)
        
        # Clean up extra whitespace
        cleaned_prompt = ' '.join(cleaned_prompt.split())
        
        # Final safety check: ensure no quotes remain
        if "'" in cleaned_prompt or '"' in cleaned_prompt:
            self.logger.warning(f"⚠️ Quotes still present after cleaning, removing all quotes")
            cleaned_prompt = cleaned_prompt.replace("'", "").replace('"', "")
        
        return cleaned_prompt

    def _is_shell_safe(self, prompt: str) -> bool:
        """Check if a prompt is safe for shell execution."""
        # Check for dangerous shell characters
        dangerous_chars = ['"', "'", '`', '$', '&', '|', ';', '\\', '(', ')', '[', ']', '{', '}']
        
        for char in dangerous_chars:
            if char in prompt:
                return False
        
        # Check for unbalanced quotes (shouldn't happen after cleaning, but double-check)
        if prompt.count('"') % 2 != 0 or prompt.count("'") % 2 != 0:
            return False
        
        return True

    def query_vllm_no_system_prompt(self, prompt: str) -> Optional[str]:
        """
        Query vLLM WITHOUT system prompt - just send raw prompt to completions endpoint.
        This mimics the 'no system prompt' behavior from compare_system_vs_no_system.py
        """
        try:
            vllm_url = f"http://localhost:{self.config.get('vllm_optim_port', 11300)}/v1/completions"
            
            # No system prompt - just send the raw prompt directly
            payload = {
                "model": "llama-3-2-3b-it",
                "prompt": f"Please optimize this prompt for 3D generation: {prompt}",
                "max_tokens": 200,
                "temperature": 0.7,
                "top_p": 0.9,
                "stream": False
            }
            
            self.logger.info("📝 No system prompt - using raw prompt directly")
            
            response = requests.post(
                vllm_url,
                json=payload,
                headers={"Content-Type": "application/json"},
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                full_response = result['choices'][0]['text'].strip()
                
                # Clean the response to extract just the optimized prompt
                optimized_prompt = self._clean_vllm_response(full_response)
                
                self.logger.info(f"✅ vLLM optimization successful (no system prompt):")
                self.logger.info(f"   Original: '{prompt}'")
                self.logger.info(f"   Optimized: '{optimized_prompt}'")
                
                # Track success
                self.stats['vllm_no_system_success'] += 1
                
                return optimized_prompt
            else:
                self.logger.error(f"❌ vLLM optimization failed with status {response.status_code}: {response.text}")
                self.stats['vllm_failures'] += 1
                return None
                
        except Exception as e:
            self.logger.error(f"❌ vLLM optimization error: {e}")
            self.stats['vllm_failures'] += 1
            return None

    def query_vllm_with_system_prompt_chat(self, prompt: str) -> Optional[str]:
        """
        Query vLLM WITH system prompt using chat completions endpoint.
        Uses structured chat format with system/user/assistant messages.
        """
        try:
            vllm_url = f"http://localhost:{self.config.get('vllm_optim_port', 11300)}/v1/chat/completions"
            
            # System prompt that explains the task
            system_prompt = "You are a prompt optimization expert. Your task is to take a simple prompt and enhance it with detailed, descriptive language that would be perfect for 3D generation. Make the description vivid, specific, and complete. Focus on materials, textures, lighting, perspective, and artistic style."
            
            # Format with system prompt (chat format)
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Please optimize this prompt for 3D generation: {prompt}"}
            ]
            
            payload = {
                "model": "llama-3-2-3b-it",
                "messages": messages,
                "max_tokens": 200,
                "temperature": 0.7,
                "top_p": 0.9,
                "stream": False
            }
            
            self.logger.info("📝 Using system prompt with chat completions")
            
            response = requests.post(
                vllm_url,
                json=payload,
                headers={"Content-Type": "application/json"},
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                full_response = result['choices'][0]['message']['content'].strip()
                
                # Clean the response to extract just the optimized prompt
                optimized_prompt = self._clean_vllm_response(full_response)
                
                self.logger.info(f"✅ vLLM optimization successful (system prompt + chat):")
                self.logger.info(f"   Original: '{prompt}'")
                self.logger.info(f"   Optimized: '{optimized_prompt}'")
                
                # Track success
                self.stats['vllm_system_chat_success'] += 1
                
                return optimized_prompt
            else:
                self.logger.error(f"❌ vLLM optimization failed with status {response.status_code}: {response.text}")
                self.stats['vllm_failures'] += 1
                return None
                
        except Exception as e:
            self.logger.error(f"❌ vLLM optimization error: {e}")
            self.stats['vllm_failures'] += 1
            return None

    def query_vllm_with_system_prompt_completions(self, prompt: str) -> Optional[str]:
        """
        Query vLLM WITH system prompt using completions endpoint.
        Uses the same format as compare_system_vs_no_system.py with system prompt.
        """
        try:
            vllm_url = f"http://localhost:{self.config.get('vllm_optim_port', 11300)}/v1/completions"
            
            # System prompt that explains the task (same as original script)
            system_prompt = "You are a prompt optimization expert. Your task is to take a simple prompt and enhance it with detailed, descriptive language that would be perfect for 3D generation. Make the description vivid, specific, and complete. Focus on materials, textures, lighting, perspective, and artistic style."
            
            # Format with system prompt (same format as original script)
            formatted_prompt = f"<|im_start|>system\n{system_prompt}<|im_end|>\n<|im_start|>user\nPlease optimize this prompt for 3D generation: {prompt}<|im_end|>\n<|im_start|>assistant\n"
            
            payload = {
                "model": "llama-3-2-3b-it",
                "prompt": formatted_prompt,
                "max_tokens": 200,
                "temperature": 0.7,
                "top_p": 0.9,
                "stream": False
            }
            
            self.logger.info("📝 Using system prompt with completions (like original script)")
            
            response = requests.post(
                vllm_url,
                json=payload,
                headers={"Content-Type": "application/json"},
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                full_response = result['choices'][0]['text'].strip()
                
                # Clean the response to extract just the optimized prompt
                optimized_prompt = self._clean_vllm_response(full_response)
                
                self.logger.info(f"✅ vLLM optimization successful (system prompt + completions):")
                self.logger.info(f"   Original: '{prompt}'")
                self.logger.info(f"   Optimized: '{optimized_prompt}'")
                
                # Track success
                self.stats['vllm_system_completions_success'] += 1
                
                return optimized_prompt
            else:
                self.logger.error(f"❌ vLLM optimization failed with status {response.status_code}: {response.text}")
                self.stats['vllm_failures'] += 1
                return None
                
        except Exception as e:
            self.logger.error(f"❌ vLLM optimization error: {e}")
            self.stats['vllm_failures'] += 1
            return None

    def test_vllm_connection(self, vllm_port: int = 11300) -> bool:
        """Test connection to vLLM server for optimization."""
        try:
            vllm_url = f"http://localhost:{vllm_port}/v1/models"
            
            print(f"🔍 Testing vLLM connection on port {vllm_port}")
            
            response = requests.get(vllm_url, timeout=10)
            
            if response.status_code == 200:
                print(f"✅ vLLM connection test successful")
                return True
            else:
                print(f"❌ vLLM connection test failed with status {response.status_code}")
                return False
                
        except Exception as e:
            print(f"❌ vLLM connection test error: {e}")
            return False

        
    def optimize_prompt_with_vllm(self, task: TaskRecord) -> Optional[str]:
        """
        Optimize prompt using vLLM with the configured priority method.
        Returns optimized prompt or None if failed.
        """
        try:
            if not self.config.get('use_vllm_optim', False):
                return None
            
            # Test vLLM connection first
            if not self.test_vllm_connection(vllm_port=self.config.get('vllm_optim_port', 11300)):
                self.logger.error(f"❌ vLLM connection test failed, skipping vLLM optimization")
                return None
            
            priority = self.config.get('vllm_optimization_priority', 'system_chat')
            original_prompt = task.prompt
            
            self.logger.info(f"🚀 Using vLLM optimization with priority: {priority}")
            
            # Try the priority method first
            optimized_prompt = None
            method_used = None
            
            if priority == 'system_chat':
                optimized_prompt = self.query_vllm_with_system_prompt_chat(original_prompt)
                method_used = 'system_chat'
            elif priority == 'system_completions':
                optimized_prompt = self.query_vllm_with_system_prompt_completions(original_prompt)
                method_used = 'system_completions'
            elif priority == 'no_system':
                optimized_prompt = self.query_vllm_no_system_prompt(original_prompt)
                method_used = 'no_system'
            
            # If priority method failed, try fallback methods
            if not optimized_prompt:
                self.logger.warning(f"⚠️ Priority method {priority} failed, trying fallbacks...")
                
                if priority != 'system_chat':
                    optimized_prompt = self.query_vllm_with_system_prompt_chat(original_prompt)
                    method_used = 'system_chat_fallback'
                
                if not optimized_prompt and priority != 'system_completions':
                    optimized_prompt = self.query_vllm_with_system_prompt_completions(original_prompt)
                    method_used = 'system_completions_fallback'
                
                if not optimized_prompt and priority != 'no_system':
                    optimized_prompt = self.query_vllm_no_system_prompt(original_prompt)
                    method_used = 'no_system_fallback'
            
            if optimized_prompt:
                # Update task record with vLLM optimization details
                task.vllm_optimization_method = method_used
                task.vllm_optimized_prompt = optimized_prompt
                task.vllm_optimization_success = True
                
                # Track successful vLLM optimization
                self.stats['vllm_optimizations'] += 1
                
                self.logger.info(f"✅ vLLM optimization successful:")
                self.logger.info(f"   Method: {method_used}")
                self.logger.info(f"   Original: '{original_prompt[:50]}...'")
                self.logger.info(f"   Optimized: '{optimized_prompt[:50]}...'")
                
                return optimized_prompt
            else:
                # All vLLM methods failed
                task.vllm_optimization_success = False
                self.logger.error(f"❌ All vLLM optimization methods failed")
                return None
                
        except Exception as e:
            self.logger.error(f"❌ vLLM optimization failed: {e}")
            task.vllm_optimization_success = False
            return None
            
    def get_deterministic_seed(self, task: TaskRecord) -> int:
        """Generate deterministic seed based on prompt for consistent results with variety"""
        if self.config.get('use_fixed_seed', True):
            return self.config.get('fixed_seed_value', 42)  # Use configured fixed seed
        else:
            # Generate deterministic seed from prompt hash for variety but determinism
            import hashlib
            hash_obj = hashlib.sha256(task.prompt.encode())
            seed = int(hash_obj.hexdigest()[:8], 16) % (2**31)  # Convert to 32-bit int
            return seed
    
    def route_prompt_to_optimal_lora(self, task: TaskRecord) -> Dict[str, Any]:
        """
        Route prompt to optimal LoRA using intelligent analysis.
        Returns dict with lora_name, endpoint, and reasoning.
        """
        # Check if LoRA routing is enabled
        if not self.config.get('enable_lora_routing', True):
            return {
                'lora_name': 'Patched Realism',
                'endpoint': '/generate/',
                'reasoning': 'LoRA routing disabled in config',
                'confidence': 'Low'
            }
        
        if not ORGANIC_LORA_ROUTER_AVAILABLE or not self.lora_router:
            # Fallback to default model
            return {
                'lora_name': 'Patched Realism',
                'endpoint': '/generate/',
                'reasoning': 'Default model (LoRA router not available)',
                'confidence': 'Low'
            }
        
        try:
            # Use organic router to select optimal LoRA through pattern learning
            router_result = self.lora_router.route_final(task.prompt, "edge_case")
            
            # Map LoRA names to endpoints
            lora_endpoints = {
                'Patched Realism': '/generate/',
                'Team Fortress 2 Style': '/generate/tf2_style/',
                'Cartoon 3D Render': '/generate/cartoon_3d/',
                '3D Game Assets': '/generate/game_assets/',
                'Game Icon Institute': '/generate/sd15_game_icon/',
                'Cinema Style': '/generate/cinema/',
                'Flux Isometric 3D': '/generate/isometric_3d/',
                'Baolei Style': '/generate/baolei_style/'
            }
            
            endpoint = lora_endpoints.get(router_result.recommended_lora, '/generate/')
            
            # Track routing decision
            self.stats['lora_routing_decisions'] += 1
            
            routing_info = {
                'lora_name': router_result.recommended_lora,
                'endpoint': endpoint,
                'reasoning': router_result.reasoning,
                'confidence': router_result.confidence
            }
            
            self.logger.info(f"🧠 LoRA Routing Decision:")
            self.logger.info(f"   Prompt: '{task.prompt[:50]}...'")
            self.logger.info(f"   Selected LoRA: {routing_info['lora_name']}")
            self.logger.info(f"   Endpoint: {routing_info['endpoint']}")
            self.logger.info(f"   Reasoning: {routing_info['reasoning']}")
            self.logger.info(f"   Confidence: {routing_info['confidence']}")
            
            return routing_info
            
        except Exception as e:
            self.logger.error(f"❌ LoRA routing failed: {e}")
            # Fallback to default
            return {
                'lora_name': 'Patched Realism',
                'endpoint': '/generate/',
                'reasoning': f'Routing failed: {str(e)}',
                'confidence': 'Low'
            }
    
    def [optimize_prompt_for_generation](self, task: TaskRecord) -> Dict[str, Any]:
        """
        Optimize prompt and route to optimal LoRA.
        Returns dict with optimized_prompt, lora_info, and endpoint.
        """
        try:
            # Step 1: Route to optimal LoRA first
            # lora_info = self.route_prompt_to_optimal_lora(task)
            default_lora = self.config.get('default_lora', 'cinema')
            lora_info = {
                'lora_name': default_lora,
                'endpoint': f'/generate/{default_lora.lower().replace(" ", "_")}/',
                'reasoning': f'Default model: {default_lora} (LoRA router not available)',
                'confidence': 'High'
            }
            # Step 2: Optimize prompt based on selected LoRA
            optimized_prompt = task.prompt  # Default to original
            
            if self.config.get('enable_prompt_optimization', True):
                # Step 1: Try vLLM optimization first if enabled
                if self.config.get('use_vllm_optim', True):
                    vllm_optimized_prompt = self.optimize_prompt_with_vllm(
                        task
                    )
                    if vllm_optimized_prompt:
                        self.logger.info(f"🚀 vLLM optimization successful, using vLLM result")
                        optimized_prompt = vllm_optimized_prompt
                        self.stats['vllm_optimizations'] = self.stats.get('vllm_optimizations', 0) + 1
                        return {
                            'optimized_prompt': optimized_prompt,
                            'lora_info': lora_info,
                            'endpoint': lora_info['endpoint'],
                            'original_prompt': task.prompt,
                            'method': f"vllm_{self.config.get('vllm_optimization_priority', 'system_chat')}"
                        }
                    else:
                        self.logger.warning(f"⚠️ vLLM optimization failed, falling back to other methods")
                
                # Step 2: Check if reproducibility system is available and enabled
                if (REPRODUCIBILITY_SYSTEM_AVAILABLE and 
                    self.reproducibility_system and 
                    self.config.get('enable_reproducibility_optimization', True)):
                        
                        min_similarity = self.config.get('reproducibility_min_similarity', 0.3)
                        
                        # Use enhanced gold prompts (memory + logs) if real-time learning is enabled
                        if self.config.get('activate_learning', False):
                            enhanced_gold_prompts = self.get_fresh_gold_prompts()
                            gold_prompts_count = len(enhanced_gold_prompts)
                            self.logger.info(f"🚀 Using ENHANCED gold prompts: {gold_prompts_count} total (memory + logs)")
                            
                            # CRITICAL: Update the reproducibility system with enhanced gold prompts
                            # This ensures it uses the optimized versions instead of just original prompts
                            self.reproducibility_system.update_gold_standard_results(enhanced_gold_prompts)
                            self.logger.info(f"🔄 Updated reproducibility system with {len(enhanced_gold_prompts)} enhanced gold prompts")
                            
                            # Now use the standard reproducibility optimization with updated data
                            repro_result = self.reproducibility_system.optimize_prompt_with_reproducibility(
                                task.prompt, min_similarity, run_validation=False
                            )
                        else:
                            # Use standard episodic memory gold prompts
                            gold_prompts_count = len(self.reproducibility_system.gold_standard_results)
                            self.logger.debug(f"📚 Using {gold_prompts_count} gold prompts from episodic memory")
                            
                            # Log the similarity threshold being used
                            if self.config.get('log_optimization_details', True):
                                self.logger.info(f"🔍 Searching for gold prompts with similarity ≥ {min_similarity}")
                            
                            repro_result = self.reproducibility_system.optimize_prompt_with_reproducibility(
                                task.prompt, min_similarity, run_validation=False
                            )
                        
                        if repro_result:
                            optimized_prompt = repro_result['optimized_prompt']
                            similarity = repro_result['similarity']
                            gold_score = repro_result['gold_score']
                            
                            if self.config.get('log_optimization_details', True):
                                self.logger.info(f"🔄 Reproducibility optimization SUCCESS:")
                                self.logger.info(f"   Original: '{task.prompt}'")
                                self.logger.info(f"   Optimized: '{optimized_prompt}'")
                                self.logger.info(f"   Similarity: {similarity:.3f}")
                                self.logger.info(f"   Gold score: {gold_score:.4f}")
                                self.logger.info(f"   📚 Gold prompts available: {gold_prompts_count}")
                            else:
                                self.logger.info(f"🔄 Reproducibility optimized (sim: {similarity:.2f}, gold: {gold_score:.3f})")
                            
                            self.stats['prompts_optimized'] += 1
                            self.stats['reproducibility_optimizations'] = self.stats.get('reproducibility_optimizations', 0) + 1
                        else:
                            # Log when reproducibility optimization fails
                            if self.config.get('log_optimization_details', True):
                                self.logger.info(f"⚠️ Reproducibility optimization FAILED:")
                                self.logger.info(f"   Original: '{task.prompt}'")
                                self.logger.info(f"   Reason: No close gold prompt found (threshold: {min_similarity})")
                                self.logger.info(f"   📚 Gold prompts available: {gold_prompts_count}")
                                self.logger.info(f"   → Falling back to traditional optimization...")
                            else:
                                self.logger.info(f"⚠️ Reproducibility failed, using traditional optimization")
                            
                            # Try traditional optimization as fallback
                            if OPTIMIZED_PROMPT_OPTIMIZER_AVAILABLE:
                                if self.config.get('log_optimization_details', True):
                                    self.logger.info(f"🚀 Traditional optimization FALLBACK:")
                                    self.logger.info(f"   Original: '{task.prompt}'")
                                
                                result = self.prompt_optimizer.optimize_with_examples(task.prompt)
                                if 'error' in result.lower():
                                    self.logger.error(f"❌ Traditional optimization failed: {result['error']}")
                                    return {
                                        'optimized_prompt': task.prompt,
                                        'lora_info': lora_info,
                                        'endpoint': lora_info['endpoint'],
                                        'original_prompt': task.prompt
                                    }
                                optimized_prompt = result
                                confidence = 0.8
                                
                                if self.config.get('log_optimization_details', True):
                                    self.logger.info(f"   Optimized: '{optimized_prompt}'")
                                    self.logger.info(f"   Confidence: {confidence:.1%}")
                                    self.logger.info(f"   Method: Fast examples-based optimization")
                                
                                self.stats['prompts_optimized'] += 1
                                self.stats['traditional_optimizations'] = self.stats.get('traditional_optimizations', 0) + 1
                            else:
                                # Use original prompt if no optimizer available
                                optimized_prompt = task.prompt
                                if self.config.get('log_optimization_details', True):
                                    self.logger.info(f"ℹ️ No optimizer available - using original prompt")
                        
                
                
            else:
                # Fallback to original optimizer
                if self.config.get('log_optimization_details', True):
                    self.logger.info(f"🔍 Original optimizer FALLBACK:")
                    self.logger.info(f"   Original: '{task.prompt}'")
                
                optimization_result = self.prompt_optimizer.optimize_prompt(
                    task.prompt, 
                    aggressive=self.config.get('optimization_aggressive_mode', False)
                )
                analysis = optimization_result['analysis']
                
                # Log the analysis if enabled
                if self.config.get('log_optimization_details', True):
                    self.logger.info(f"   Risk Level: {analysis['risk_level']}")
                    
                    if analysis['risk_factors']:
                        self.logger.info(f"   Risk Factors:")
                        for factor in analysis['risk_factors']:
                            self.logger.info(f"     • {factor}")
                
                self.stats['prompts_optimized'] += 1
                self.stats['traditional_optimizations'] = self.stats.get('traditional_optimizations', 0) + 1
            
            # Return comprehensive optimization result
            return {
                'optimized_prompt': optimized_prompt,
                'lora_info': lora_info,
                'endpoint': lora_info['endpoint'],
                'original_prompt': task.prompt
            }
                
        except Exception as e:
            self.logger.error(f"❌ Prompt optimization failed: {e}")
            # Return fallback result
            return {
                'optimized_prompt': task.prompt,
                'lora_info': {
                    'lora_name': 'Patched Realism',
                    'endpoint': '/generate/',
                    'reasoning': f'Optimization failed: {str(e)}',
                    'confidence': 'Low'
                },
                'endpoint': '/generate/',
                'original_prompt': task.prompt
            }

    def _clear_trellis_gpu_cache(self):
        """Send a request to the TRELLIS server to clear GPU cache."""
        try:
            url = f"{self.trellis_server_url}/clear_cache/"
            resp = requests.post(url, timeout=10)
            if resp.status_code == 200:
                self.logger.info(f"[TRELLIS] GPU cache cleared: {resp.json()}")
            else:
                self.logger.warning(f"[TRELLIS] Failed to clear GPU cache: HTTP {resp.status_code}")
        except Exception as e:
            self.logger.warning(f"[TRELLIS] Exception clearing GPU cache: {e}")


    async def generate_3d_model(self, task: TaskRecord) -> Optional[Dict[str, Any]]:
        """Generate 3D model using TRELLIS server with prompt optimization"""
        self.logger.info(f"🎨 Generating 3D model: '{task.prompt}' (task: {task.task_id})")
        
        # Check if fallback mechanism is enabled
        if self.config.get('enable_fallback_mechanism', False) and CLIP_AVAILABLE:
            self.logger.info(f"🔄 Using fallback mechanism for generation")
            return await self.generate_3d_model_with_fallback(task)
        else:
            self.logger.info(f"🔄 Using standard generation without fallback")
            return await self._generate_3d_model_standard(task)
            
    def _get_generation_params_old(self) -> Dict[str, Any]:
        """
        Get TRELLIS generation parameters based on selected quality preset.
        Based on README_GRID_FLOW_EXPERIMENTS.md findings.
        """
        if self.config.get('fastest_mv_gen'):
            # 🚀 FASTEST Configuration (Speed Priority)
            # Expected: ~35-45s generation time, 4-6/10 quality, 5-10 MB PLY
            return {
                'ss_sampling_steps': 15,           # Minimal TRELLIS steps
                'slat_sampling_steps': 15,         # Minimal TRELLIS steps
                'slat_guidance_strength': 5.0,     # Reduced guidance for speed
                'ss_guidance_strength': 3.0,       # Reduced guidance for speed
                'width': 256,                      # Smallest resolution
                'height': 256,                     # Smallest resolution
                'num_inference_steps': 4,          # Minimal FLUX steps
                'guidance_scale': 2.5,             # Lower guidance for speed
                'upscale': False,                  # Never upscale (proven harmful)
                'remove_background': True,#False,        # Skip for speed
                'use_short_prompt': True,          # Short prompts for speed
                'filter_low_quality': False,       # Skip quality filtering
                'save_preview': False,             # Skip preview generation
                'save_intermediate': False         # Skip intermediate saves
            }
        elif self.config.get('long_fast_mv_gen'):
            # ⚡ FAST but GOOD QUALITY Configuration (Balanced)
            # Expected: ~55-65s generation time, 7-8/10 quality, 15-25 MB PLY
            return {
                'ss_sampling_steps': 18,           # Reduced TRELLIS steps
                'slat_sampling_steps': 18,         # Reduced TRELLIS steps
                'slat_guidance_strength': 6.0,     # Moderate guidance
                'ss_guidance_strength': 4.0,       # Moderate guidance
                'width': 512,                      # Sweet spot resolution
                'height': 512,                     # Sweet spot resolution
                'num_inference_steps': 7,          # Optimal FLUX steps
                'guidance_scale': 3.5,             # Balanced guidance
                'upscale': False,                  # Never upscale (proven harmful)
                'remove_background': True,         # Enable for quality
                'use_short_prompt': False,         # Full prompts for quality
                'filter_low_quality': True,        # Enable quality filtering
                'save_preview': True,              # Enable preview
                'save_intermediate': True          # Save intermediates for inspection
            }
        elif self.config.get('production_mv_gen'):
            # 🎯 PRODUCTION QUALITY Configuration (Recommended)
            # Expected: ~60-70s generation time, 7-8/10 quality, 15-25 MB PLY
            return {
                'ss_sampling_steps': 21,           # Optimal TRELLIS steps
                'slat_sampling_steps': 24,         # Optimal TRELLIS steps
                'slat_guidance_strength': 8.0,     # High guidance for quality
                'ss_guidance_strength': 4.5,       # High guidance for quality
                'width': 512,                      # Sweet spot resolution
                'height': 512,                     # Sweet spot resolution
                'num_inference_steps': 7,          # Optimal FLUX steps
                'guidance_scale': 3.5,             # Optimal guidance
                'upscale': False,                  # Never upscale (proven harmful)
                'remove_background': True,         # Essential for quality
                'use_short_prompt': False,         # Detailed prompts for quality
                'filter_low_quality': True,        # Strict quality filtering
                'save_preview': True,              # Enable preview
                'save_intermediate': True          # Save all intermediates
            }
        elif self.config.get('quality_mv_gen'):
            # 🎨 HIGHEST QUALITY Configuration (Quality Priority)
            # Expected: ~90-120s generation time, 9-10/10 quality, 40-60 MB PLY
            return {
                'ss_sampling_steps': 30,           # Maximum TRELLIS steps
                'slat_sampling_steps': 30,         # Maximum TRELLIS steps
                'slat_guidance_strength': 8.0,     # Maximum guidance for quality
                'ss_guidance_strength': 5.0,       # Maximum guidance for quality
                'width': 1024,                     # Maximum native resolution
                'height': 1024,                    # Maximum native resolution
                'num_inference_steps': 12,         # High-quality FLUX generation
                'guidance_scale': 4.5,             # High guidance for quality
                'upscale': False,                  # Never upscale (proven harmful)
                'remove_background': True,         # Essential for quality
                'use_short_prompt': False,         # Detailed prompts for quality
                'filter_low_quality': True,        # Strict quality filtering
                'save_preview': True,              # Enable preview
                'save_intermediate': True          # Save all intermediates
            }
        else:
            # ⚙️ DEFAULT Configuration (Current settings)
            return {
                'ss_sampling_steps': 20,           # Default TRELLIS steps
                'slat_sampling_steps': 24,         # Default TRELLIS steps
                'slat_guidance_strength': 8.0,     # Default guidance
                'ss_guidance_strength': 4.5,       # Default guidance
                'width': 512,                      # Default resolution
                'height': 512,                     # Default resolution
                'num_inference_steps': 7,          # Default FLUX steps
                'guidance_scale': 3.5,             # Default guidance
                'upscale': False,                  # Never upscale (proven harmful)
                'remove_background': True,         # Default background removal
                'use_short_prompt': False,         # Default prompt length
                'filter_low_quality': True,        # Default quality filtering
                'save_preview': True,              # Default preview
                'save_intermediate': True          # Default intermediate saves
            }

    def _get_generation_params(self) -> Dict[str, Any]:
        """
        Get TRELLIS generation parameters based on selected quality preset.
        UPDATED BASED ON NEW VALIDATION DATA - All presets now achieve Perfect Fidelity!
        """
        if self.config.get('fastest_mv_gen'):
            # 🚀 FASTEST Configuration (Speed Priority) - REVOLUTIONARY UPDATE
            # Expected: ~64-75s generation time, Perfect Fidelity (1.0), 7-11 MB PLY
            # Key Discovery: 512×512 is actually FASTER than 256×256 with perfect quality!
            # return {
            #     'ss_sampling_steps': 21,           # Optimal TRELLIS steps (from validation data)
            #     'slat_sampling_steps': 24,         # Optimal TRELLIS steps (from validation data)
            #     'slat_guidance_strength': 7.5,     # Optimal guidance (from validation data)
            #     'ss_guidance_strength': 4.0,       # Optimal guidance (from validation data)
            #     'width': 512,                      # Sweet spot resolution (proven fastest with quality)
            #     'height': 512,                     # Sweet spot resolution
            #     'num_inference_steps': 7,          # Optimal FLUX steps (from validation data)
            #     'guidance_scale': 3.5,             # Optimal guidance (from validation data)
            #     'upscale': False,                  # Never upscale (proven harmful in validation)
            #     'remove_background': True,         # Enable (proven no time impact in validation)
            #     'use_short_prompt': True,          # Short prompts (proven faster in validation)
            #     'filter_low_quality': True,        # Enable (proven no time impact in validation)
            #     'save_preview': True,              # Enable (proven no time impact in validation)
            #     'save_intermediate': True          # Enable (proven no time impact in validation)
            # }
            return {
                "num_inference_steps": 16,
                "guidance_scale": 3.5,
                "width": 1024,
                "height": 1024,
                "upscale": True,
                "remove_background": True,
                "ss_guidance_strength": 8.0,
                "ss_sampling_steps": 25,
                "slat_guidance_strength": 5.0,
                "slat_sampling_steps": 30,
                "save_preview": False,        # Generate preview video
                "save_intermediate": False,   # Save all intermediate outputs
                "filter_low_quality": False,
                "use_short_prompt": True, 
                # "image_endpoint": "cinema"
            }
        elif self.config.get('long_fast_mv_gen'):
            # ⚡ FAST but GOOD QUALITY Configuration (Balanced) - UPDATED
            # Expected: ~71-78s generation time, Perfect Fidelity (1.0), 7-11 MB PLY
            return {
                'ss_sampling_steps': 21,           # Optimal TRELLIS steps (from validation data)
                'slat_sampling_steps': 24,         # Optimal TRELLIS steps (from validation data)
                'slat_guidance_strength': 7.5,     # Optimal guidance (from validation data)
                'ss_guidance_strength': 4.0,       # Optimal guidance (from validation data)
                'width': 512,                      # Sweet spot resolution (proven optimal)
                'height': 512,                     # Sweet spot resolution
                'num_inference_steps': 7,          # Optimal FLUX steps (from validation data)
                'guidance_scale': 3.5,             # Optimal guidance (from validation data)
                'upscale': False,                  # Never upscale (proven harmful in validation)
                'remove_background': True,         # Essential for quality
                'use_short_prompt': False,         # Long prompts for quality (proven better in validation)
                'filter_low_quality': True,        # Essential for quality
                'save_preview': True,              # Enable preview
                'save_intermediate': True          # Save all intermediates
            }
        elif self.config.get('production_mv_gen'):
            # 🎯 PRODUCTION QUALITY Configuration (Recommended) - UPDATED
            # Expected: ~71-78s generation time, Perfect Fidelity (1.0), 7-11 MB PLY
            return {
                'ss_sampling_steps': 21,           # Optimal TRELLIS steps (from validation data)
                'slat_sampling_steps': 24,         # Optimal TRELLIS steps (from validation data)
                'slat_guidance_strength': 7.5,     # Optimal guidance (from validation data)
                'ss_guidance_strength': 4.0,       # Optimal guidance (from validation data)
                'width': 512,                      # Sweet spot resolution (proven optimal)
                'height': 512,                     # Sweet spot resolution
                'num_inference_steps': 7,          # Optimal FLUX steps (from validation data)
                'guidance_scale': 3.5,             # Optimal guidance (from validation data)
                'upscale': False,                  # Never upscale (proven harmful in validation)
                'remove_background': True,         # Essential for quality
                'use_short_prompt': False,         # Long prompts for quality (proven better in validation)
                'filter_low_quality': True,        # Essential for quality
                'save_preview': True,              # Enable preview
                'save_intermediate': True          # Save all intermediates
            }
        elif self.config.get('quality_mv_gen'):
            # 🎨 HIGHEST QUALITY Configuration (Quality Priority) - UPDATED
            # Expected: ~83-138s generation time, Perfect Fidelity (1.0), 23-58 MB PLY
            return {
                'ss_sampling_steps': 25,           # High TRELLIS steps (from validation data)
                'slat_sampling_steps': 30,         # High TRELLIS steps (from validation data)
                'slat_guidance_strength': 8.0,     # High guidance (from validation data)
                'ss_guidance_strength': 5.0,       # High guidance (from validation data)
                'width': 1024,                     # Maximum native resolution (proven quality)
                'height': 1024,                    # Maximum native resolution
                'num_inference_steps': 16,         # High-quality FLUX generation (from validation data)
                'guidance_scale': 5.0,             # High guidance for quality (from validation data)
                'upscale': False,                  # Never upscale (proven harmful in validation)
                'remove_background': True,         # Essential for quality
                'use_short_prompt': False,         # Detailed prompts for quality
                'filter_low_quality': True,        # Strict quality filtering
                'save_preview': True,              # Enable preview
                'save_intermediate': True          # Save all intermediates
            }
        else:
            # ⚙️ DEFAULT Configuration (Current settings) - UPDATED BASED ON VALIDATION DATA
            # Expected: ~71-78s generation time, Perfect Fidelity (1.0), 7-11 MB PLY
            return {
                'ss_sampling_steps': 21,           # Optimal TRELLIS steps (from validation data)
                'slat_sampling_steps': 24,         # Optimal TRELLIS steps (from validation data)
                'slat_guidance_strength': 7.5,     # Optimal guidance (from validation data)
                'ss_guidance_strength': 4.0,       # Optimal guidance (from validation data)
                'width': 512,                      # Sweet spot resolution (proven optimal)
                'height': 512,                     # Sweet spot resolution
                'num_inference_steps': 7,          # Optimal FLUX steps (from validation data)
                'guidance_scale': 3.5,             # Optimal guidance (from validation data)
                'upscale': False,                  # Never upscale (proven harmful in validation)
                'remove_background': True,         # Essential for quality
                'use_short_prompt': False,         # Long prompts for quality (proven better in validation)
                'filter_low_quality': True,        # Essential for quality
                'save_preview': True,              # Enable preview
                'save_intermediate': True          # Save all intermediates
            }

    async def _generate_3d_model_standard(self, task: TaskRecord) -> Optional[Dict[str, Any]]:
        """Generate 3D model using TRELLIS server with prompt optimization (standard method)"""
        self.logger.info(f"🎨 Generating 3D model (standard): '{task.prompt}' (task: {task.task_id})")
        
        try:
            # CRITICAL: Wait for priority access to the server
            # This is where we ensure subnet tasks get priority over optimizer tasks
            if not self.priority_coordinator.wait_for_priority_access(task.task_id):
                self.logger.error(f"❌ PRIORITY ACCESS TIMEOUT for task {task.task_id} - subnet task will be missed!")
                task.priority_access_timeout = True  # Mark this task as having priority access timeout
                return None
            
            # Mark the start of our priority job
            self.priority_coordinator.mark_priority_job_start(task.task_id, task.prompt)
            
            ### ONCE
            # Step 1: Optimize prompt and route to optimal LoRA
            # optimization_result = self.optimize_prompt_for_generation(task)
            # optimized_prompt = optimization_result['optimized_prompt']
            # lora_info = optimization_result['lora_info']
            # endpoint = optimization_result['endpoint']

            # # optimized_prompt = task.prompt
            # # lora_info = {
            # #     'lora_name': 'Cinema',
            # #     'endpoint': '/generate/cinema/',
            # #     'reasoning': 'Default model: Cinema',
            # #     'confidence': 'High'
            # # }
            # # # endpoint = '/generate/cinema/'
            # # endpoint = '/generate/'


            # # Step 1.5: Clean the optimized prompt to remove artifacts
            # cleaned_prompt = self.clean_optimized_prompt_wbgmsst(optimized_prompt)
            # # Only add "white background" if it's not already present
            # # cleaned_prompt = optimized_prompt
            # if "white background" not in cleaned_prompt.lower():
            #     cleaned_prompt = cleaned_prompt + " front view, white background"
            
            # optimize_generation = True
            # if optimize_generation:
            #     ### lOOP:
            #     max_iterations = 3
            #     idx = 0
            #     while idx < max_iterations:
            #         # Step 2: Generate with optimized prompt
            #         self.logger.info(f"🔄 Step 2: Generating with optimized prompt")
            #         optimization_result = self.optimize_prompt_for_generation(task)
            #         optimized_prompt = optimization_result['optimized_prompt']
            #         lora_info = optimization_result['lora_info']
            #         endpoint = optimization_result['endpoint']
                    
            #         # Clean the optimized prompt
            #         cleaned_prompt = self.clean_optimized_prompt_wbgmsst(optimized_prompt)
            #         # if "white background" not in cleaned_prompt.lower():
            #         #     cleaned_prompt = cleaned_prompt + " front view, white background"
                    
            #         # Compute cosine similarity between original and optimized prompts
            #         original_optimized_similarity = self.similarity_server.compute_similarity_device(self.similarity_device, task.prompt, optimized_prompt, warmup_runs=0, num_runs=1, timer=False)
                    
            #         # Check similarity threshold
            #         if original_optimized_similarity['cosine_similarity'] > 0.65:
            #             break
                        
            #         self.logger.warning(f"⚠️ Original and optimized prompts are very different, retrying")
            #         idx += 1
                        
            #     optimization_failed=False
            #     if original_optimized_similarity['cosine_similarity'] < 0.65:
            #         self.logger.error(f"❌ Original and optimized prompts are still very different using original prompt")
            #         cleaned_prompt = task.prompt + " front view, accurate, complete, white background"
            #         optimized_prompt = task.prompt
            #         optimization_failed=True
            #         optimization_result['optimized_prompt'] = cleaned_prompt
                
            #     original_endpoint = optimization_result['endpoint']
            # else:
            #     cleaned_prompt = task.prompt + " front view, accurate, complete, white background"
            #     optimized_prompt = task.prompt + " front view, accurate, complete, white background"
            #     lora_info = {
            #         'lora_name': 'Cinema',
            #         'endpoint': '/generate/cinema/',
            #         'reasoning': 'Default model: Cinema',
            #         'confidence': 'High'
            #     }
            #     original_endpoint = '/generate/cinema/'
            #     optimization_failed=False
                
            # # Check fidelity tracker for endpoint recommendations
            # recommended_endpoint = self.fidelity_tracker.get_recommended_endpoint(
            #     validator_uid=task.validator_uid,
            #     requested_endpoint=original_endpoint
            # )
            
            # # Use the recommended endpoint (may be different from original due to 0.0 score tracking)
            # # endpoint = recommended_endpoint
            # endpoint = original_endpoint
            
            # # Store the endpoint used for tracking purposes
            # task.endpoint_used = endpoint
            
            # # Log endpoint selection decision
            # if endpoint != original_endpoint:
            #     self.logger.info(f"🔄 Endpoint overridden by fidelity tracker:")
            #     self.logger.info(f"   Original: {original_endpoint}")
            #     self.logger.info(f"   Recommended: {endpoint}")
            #     self.logger.info(f"   Reason: Recent 0.0 scores detected")
            # else:
            #     self.logger.info(f"✅ Using original endpoint: {endpoint}")

            # # Log the final optimization result
            # if self.config.get('log_optimization_details', True):
            #     if optimized_prompt != task.prompt:
            #         self.logger.info(f"🎯 FINAL OPTIMIZATION RESULT:")
            #         self.logger.info(f"   Original: '{task.prompt}'")
            #         self.logger.info(f"   Optimized: '{optimized_prompt}'")
            #         self.logger.info(f"   Cleaned: '{cleaned_prompt}'")
            #         self.logger.info(f"   LoRA: {lora_info['lora_name']} via {endpoint}")
            #         if optimize_generation: 
            #             self.logger.info(f"   Similarity: {original_optimized_similarity['cosine_similarity']:.4f}")
            #             self.logger.info(f"   Similarity level: {original_optimized_similarity['similarity_level']}")
            #             self.logger.info(f"   Description: {original_optimized_similarity['description']}")
            #         else:
            #             self.logger.info(f"   Similarity: N/A")
            #             self.logger.info(f"   Similarity level: N/A")
            #             self.logger.info(f"   Description: N/A")
                    
            #     else:
            #         self.logger.info(f"ℹ️ No optimization applied - using original prompt")
            #         self.logger.info(f"   Prompt: '{task.prompt}'")
            #         self.logger.info(f"   LoRA: {lora_info['lora_name']} via {endpoint}")
            
            # # Clear cache on the server using priority coordinator
            # self.priority_coordinator.clear_server_cache()

            # # Step 2: Get deterministic seed
            # deterministic_seed = self.get_deterministic_seed(task)
            # self.logger.info(f"   🎲 Using deterministic seed: {deterministic_seed}")
            # self.logger.info(f"   �� Using LoRA: {lora_info['lora_name']} via {endpoint}")
            
            # generation_start = time.time()
            
            
            
            # # generation_params = {
            # #     'ss_sampling_steps': 21,
            # #     'slat_sampling_steps': 24,
            # #     'slat_guidance_strength': 7.5,
            # #     'ss_guidance_strength': 4.0,
            # #     'width': 512,
            # #     'height': 512,
            # #     'num_inference_steps': 7,
            # #     'guidance_scale': 3.5,
            # #     'upscale': False,
            # #     'remove_background': True,
            # #     'use_short_prompt': False,
            # #     'filter_low_quality': True,
            # #     'save_preview': True,
            # #     'save_intermediate': True
            # # }
            # generation_params = {}
            # if endpoint == "/generate_3d_from_prompt_grid_flow/":
            #     # Get generation parameters based on selected preset
            #     generation_params = self._get_generation_params()
                
            #     # Log the generation preset being used
            #     preset_name = "DEFAULT"
            #     if self.config.get('fastest_mv_gen'):
            #         preset_name = "🚀 FASTEST"
            #     elif self.config.get('long_fast_mv_gen'):
            #         preset_name = "⚡ FAST + GOOD QUALITY"
            #     elif self.config.get('production_mv_gen'):
            #         preset_name = "🎯 PRODUCTION QUALITY"
            #     elif self.config.get('quality_mv_gen'):
            #         preset_name = "🎨 HIGHEST QUALITY"
                
            #     self.logger.info(f"   🎛️ Using generation preset: {preset_name}")
            #     self.logger.info(f"   📏 Resolution: {generation_params.get('width', 'N/A')}×{generation_params.get('height', 'N/A')}")
            #     self.logger.info(f"   🔄 TRELLIS steps: SS={generation_params.get('ss_sampling_steps', 'N/A')}, SLAT={generation_params.get('slat_sampling_steps', 'N/A')}")
            #     self.logger.info(f"   🎯 FLUX steps: {generation_params.get('num_inference_steps', 'N/A')}")

            #     self.logger.info(f"   🔄 Final endpoint selection: {endpoint}")
            #     # Call TRELLIS generation server with cleaned prompt, deterministic seed, and LoRA-specific endpoint
            #     full_url = f"{self.config['generation_server_url']}{endpoint}"
                
            #     response = requests.post(
            #         full_url,
            #         data={
            #             'base_prompt': cleaned_prompt,  # Use cleaned prompt (artifacts removed)
            #             'seed': deterministic_seed,  # Use deterministic seed
            #             'return_compressed': True, 
            #             # 'ss_sampling_steps': 20,
            #             # 'slat_sampling_steps': 24,
            #             # 'slat_guidance_strength': 8.0,
            #             # 'ss_guidance_strength': 4.5
            #             **generation_params  # Use preset-based parameters/ default
            #         },
            #         timeout=self.config['generation_timeout']
            #     )
            # else:
                
            #     self.logger.info(f"   🔄 Final endpoint selection: {endpoint}")
            #     # Call TRELLIS generation server with cleaned prompt, deterministic seed, and LoRA-specific endpoint
            #     full_url = f"{self.config['generation_server_url']}{endpoint}"
                
            #     response = requests.post(
            #         full_url,
            #         data={
            #             'prompt': cleaned_prompt,  # Use cleaned prompt (artifacts removed)
            #             'seed': deterministic_seed,  # Use deterministic seed
            #             'return_compressed': True, 
            #             # 'ss_sampling_steps': 20,
            #             # 'slat_sampling_steps': 24,
            #             # 'slat_guidance_strength': 8.0,
            #             # 'ss_guidance_strength': 4.5
            #             **generation_params  # Use preset-based parameters/ default
            #         },
            #         timeout=self.config['generation_timeout']
            #     )

            
            optimize_generation = True
            if optimize_generation:
                ### lOOP:
                max_iterations = 3
                idx = 0
                best_similarity = 0.0
                best_optimization_result = None
                best_cleaned_prompt = None
                best_optimized_prompt = None
                best_lora_info = None
                best_endpoint = None
                
                while idx < max_iterations:
                    # Step 2: Generate with optimized prompt
                    self.logger.info(f"🔄 Step 2: Generating with optimized prompt (iteration {idx + 1})")
                    optimization_result = self.optimize_prompt_for_generation(task)
                    optimized_prompt = optimization_result['optimized_prompt']
                    lora_info = optimization_result['lora_info']
                    endpoint = optimization_result['endpoint']
                    
                    # Clean the optimized prompt
                    cleaned_prompt = self.clean_optimized_prompt_wbgmsst(optimized_prompt)
                    # if "white background" not in cleaned_prompt.lower():
                    #     cleaned_prompt = cleaned_prompt + " front view, white background"
                    
                    # Compute cosine similarity between original and optimized prompts
                    original_optimized_similarity = self.similarity_server.compute_similarity_device(self.similarity_device, task.prompt, optimized_prompt, warmup_runs=0, num_runs=1, timer=False)
                    current_similarity = original_optimized_similarity['cosine_similarity']
                    
                    # Track the best result so far
                    if current_similarity > best_similarity:
                        best_similarity = current_similarity
                        best_optimization_result = optimization_result
                        best_cleaned_prompt = cleaned_prompt
                        best_optimized_prompt = optimized_prompt
                        best_lora_info = lora_info
                        best_endpoint = endpoint
                        self.logger.info(f"📈 New best similarity: {best_similarity:.3f}")
                    
                    # Check similarity threshold
                    if current_similarity > 0.65:
                        self.logger.info(f"✅ Found prompt with similarity > 0.65: {current_similarity:.3f}")
                        break
                        
                    self.logger.warning(f"⚠️ Original and optimized prompts are very different (similarity: {current_similarity:.3f}), retrying")
                    idx += 1
                        
                optimization_failed=False
                # if original_optimized_similarity['cosine_similarity'] < 0.65:
                #     self.logger.error(f"❌ Original and optimized prompts are still very different using original prompt")
                #     cleaned_prompt = task.prompt + " front view, accurate, complete, white background"
                #     optimized_prompt = task.prompt
                if best_similarity < 0.65:
                    self.logger.error(f"❌ No prompt achieved >0.65 similarity. Using best result with similarity: {best_similarity:.3f}")
                    # Use the best result we found
                    optimization_result = best_optimization_result
                    cleaned_prompt = best_cleaned_prompt
                    optimized_prompt = best_optimized_prompt
                    lora_info = best_lora_info
                    endpoint = best_endpoint
                    optimization_failed=True
                else:
                    # Use the current result (which achieved >0.65)
                    optimization_result = optimization_result
                    cleaned_prompt = cleaned_prompt
                    optimized_prompt = optimized_prompt
                    lora_info = lora_info
                    endpoint = endpoint
                
                original_endpoint = optimization_result['endpoint']
            else:
                cleaned_prompt = task.prompt + " front view, accurate, complete, white background"
                optimized_prompt = task.prompt + " front view, accurate, complete, white background"
                lora_info = {
                    'lora_name': 'Cinema',
                    'endpoint': '/generate/cinema/',
                    'reasoning': 'Default model: Cinema',
                    'confidence': 'High'
                }
                original_endpoint = '/generate/cinema/'
                optimization_failed=False
                
            # Check fidelity tracker for endpoint recommendations
            # recommended_endpoint = self.fidelity_tracker.get_recommended_endpoint(
            #     validator_uid=task.validator_uid,
            #     requested_endpoint=original_endpoint
            # )
            
            # Use the recommended endpoint (may be different from original due to 0.0 score tracking)
            # endpoint = recommended_endpoint
            endpoint = original_endpoint
            # endpoint = "/generate_3d_from_prompt_grid_flow/"
            # Store the endpoint used for tracking purposes
            task.endpoint_used = endpoint
            
            # Log endpoint selection decision
            if endpoint != original_endpoint:
                self.logger.info(f"🔄 Endpoint overridden by fidelity tracker:")
                self.logger.info(f"   Original: {original_endpoint}")
                self.logger.info(f"   Recommended: {endpoint}")
                self.logger.info(f"   Reason: Recent 0.0 scores detected")
            else:
                self.logger.info(f"✅ Using original endpoint: {endpoint}")

            # Log the final optimization result
            if self.config.get('log_optimization_details', True):
                if optimized_prompt != task.prompt:
                    self.logger.info(f"🎯 FINAL OPTIMIZATION RESULT:")
                    self.logger.info(f"   Original: '{task.prompt}'")
                    self.logger.info(f"   Optimized: '{optimized_prompt}'")
                    self.logger.info(f"   Cleaned: '{cleaned_prompt}'")
                    self.logger.info(f"   LoRA: {lora_info['lora_name']} via {endpoint}")
                    if optimize_generation:     
                        self.logger.info(f"   Similarity: {original_optimized_similarity['cosine_similarity']:.4f}")
                        self.logger.info(f"   Similarity level: {original_optimized_similarity['similarity_level']}")
                        self.logger.info(f"   Description: {original_optimized_similarity['description']}")
                    else:
                        self.logger.info(f"   Similarity: N/A")
                        self.logger.info(f"   Similarity level: N/A")
                        self.logger.info(f"   Description: N/A")
                    
                else:
                    self.logger.info(f"ℹ️ No optimization applied - using original prompt")
                    self.logger.info(f"   Prompt: '{task.prompt}'")
                    self.logger.info(f"   LoRA: {lora_info['lora_name']} via {endpoint}")
            
            # Clear cache on the server using priority coordinator
            self.priority_coordinator.clear_server_cache()

            # Step 2: Get deterministic seed
            deterministic_seed = self.get_deterministic_seed(task)
            self.logger.info(f"   🎲 Using deterministic seed: {deterministic_seed}")
            self.logger.info(f"   �� Using LoRA: {lora_info['lora_name']} via {endpoint}")
            
            generation_start = time.time()
            
            
            
            # generation_params = {
            #     'ss_sampling_steps': 21,
            #     'slat_sampling_steps': 24,
            #     'slat_guidance_strength': 7.5,
            #     'ss_guidance_strength': 4.0,
            #     'width': 512,
            #     'height': 512,
            #     'num_inference_steps': 7,
            #     'guidance_scale': 3.5,
            #     'upscale': False,
            #     'remove_background': True,
            #     'use_short_prompt': False,
            #     'filter_low_quality': True,
            #     'save_preview': True,
            #     'save_intermediate': True
            # }
            generation_params = {}
            if endpoint == "/generate_3d_from_prompt_grid_flow/":
                # Get generation parameters based on selected preset
                generation_params = self._get_generation_params()
                
                # Log the generation preset being used
                preset_name = "DEFAULT"
                if self.config.get('fastest_mv_gen'):
                    preset_name = "🚀 FASTEST"
                elif self.config.get('long_fast_mv_gen'):
                    preset_name = "⚡ FAST + GOOD QUALITY"
                elif self.config.get('production_mv_gen'):
                    preset_name = "🎯 PRODUCTION QUALITY"
                elif self.config.get('quality_mv_gen'):
                    preset_name = "🎨 HIGHEST QUALITY"
                
                self.logger.info(f"   🎛️ Using generation preset: {preset_name}")
                self.logger.info(f"   📏 Resolution: {generation_params.get('width', 'N/A')}×{generation_params.get('height', 'N/A')}")
                self.logger.info(f"   🔄 TRELLIS steps: SS={generation_params.get('ss_sampling_steps', 'N/A')}, SLAT={generation_params.get('slat_sampling_steps', 'N/A')}")
                self.logger.info(f"   🎯 FLUX steps: {generation_params.get('num_inference_steps', 'N/A')}")

                self.logger.info(f"   🔄 Final endpoint selection: {endpoint}")
                # Call TRELLIS generation server with cleaned prompt, deterministic seed, and LoRA-specific endpoint
                full_url = f"{self.config['generation_server_url']}{endpoint}"

                response = requests.post(
                    full_url,
                    data={
                        'base_prompt': cleaned_prompt,  # Use cleaned prompt (artifacts removed)
                        'seed': deterministic_seed,  # Use deterministic seed
                        'return_compressed': True, 
                        # 'ss_sampling_steps': 20,
                        # 'slat_sampling_steps': 24,
                        # 'slat_guidance_strength': 8.0,
                        # 'ss_guidance_strength': 4.5
                        **generation_params  # Use preset-based parameters/ default
                    },
                    timeout=self.config['generation_timeout']
                )
            else:
                
                self.logger.info(f"   🔄 Final endpoint selection: {endpoint}")
                # Call TRELLIS generation server with cleaned prompt, deterministic seed, and LoRA-specific endpoint
                full_url = f"{self.config['generation_server_url']}{endpoint}"
                
                response = requests.post(
                    full_url,
                    data={
                        'prompt': cleaned_prompt,  # Use cleaned prompt (artifacts removed)
                        'seed': deterministic_seed,  # Use deterministic seed
                        'return_compressed': True, 
                        # 'ss_sampling_steps': 20,
                        # 'slat_sampling_steps': 24,
                        # 'slat_guidance_strength': 8.0,
                        # 'ss_guidance_strength': 4.5
                        **generation_params  # Use preset-based parameters/ default
                    },
                    timeout=self.config['generation_timeout']
                )
            
            generation_time = time.time() - generation_start
            task.generation_time = generation_time
            
            if response.status_code == 200:
                ply_data = response.content
                # is_valid, issues = fast_quality_check(ply_data, verbose=True)

                # if not is_valid:
                #     self.logger.warning(f"⚠️ Generation failed quality check: {issues}")
                #     import random
                #     deterministic_seed = random.randint(0, 1000)
                #     response = requests.post(
                #         full_url,
                #         data={
                #             'prompt': cleaned_prompt,  # Use cleaned prompt (artifacts removed)
                #             'seed': deterministic_seed,  # Use deterministic seed
                #             'return_compressed': True
                #         },
                #         timeout=self.config['generation_timeout']
                #     )
                #     if response.status_code == 200:
                #         ply_data = response.content
                #         is_valid, issues = fast_quality_check(ply_data, verbose=False)
                #         if not is_valid:
                #             self.logger.warning(f"⚠️ Generation failed quality check: {issues}")
                
                # Get metadata from headers to check compression status
                compression_ratio = response.headers.get('X-Compression-Ratio', 'unknown')

                # Save PLY file
                if self.config['save_intermediate_results']:
                    timestamp = int(time.time())
                    ply_file = self.output_dir / f"task_{task.task_id}_{timestamp}.ply.spz"
                    with open(ply_file, 'wb') as f:
                        f.write(ply_data)
                    task.compressed_file_path = str(ply_file)
                
                self.logger.info(f"✅ Generation successful in {generation_time:.2f}s ({len(ply_data):,} bytes)")
                
                self.stats['successful_generations'] += 1
                self.stats['total_generation_time'] += generation_time
                
                # Mark the completion of our priority job
                self.priority_coordinator.mark_priority_job_end(task.task_id)
                
                return {'ply_data': ply_data, 'compression_ratio': compression_ratio}
            else:
                self.logger.error(f"❌ Generation failed: HTTP {response.status_code}")
                # Mark the completion of our priority job even on failure
                self.priority_coordinator.mark_priority_job_end(task.task_id)
                return None
        
        except Exception as e:
            self.logger.error(f"❌ Generation exception: {e}")
            # Mark the completion of our priority job even on exception
            self.priority_coordinator.mark_priority_job_end(task.task_id)
            return None

    async def generate_3d_model_with_fallback(self, task: TaskRecord) -> Optional[Dict[str, Any]]:
        """
        Generate 3D model with fallback mechanism for low-fidelity tasks.
        Generates both original and optimized prompts, compares CLIP scores,
        and implements fallback if the ratio is below 0.8.
        """
        self.logger.info(f"🎨 Generating 3D model with fallback: '{task.prompt}' (task: {task.task_id})")
        
        try:
            # CRITICAL: Wait for priority access to the server
            if not self.priority_coordinator.wait_for_priority_access(task.task_id):
                self.logger.error(f"❌ PRIORITY ACCESS TIMEOUT for task {task.task_id} - subnet task will be missed!")
                task.priority_access_timeout = True
                return None
            
            # Mark the start of our priority job
            self.priority_coordinator.mark_priority_job_start(task.task_id, task.prompt)
            
            # Step 1: Generate with original prompt first
            self.logger.info(f"🔄 Step 1: Generating with original prompt")
            original_result = await self._generate_single_3d_model(task, task.prompt, "original")
            if not original_result:
                self.logger.error(f"❌ Failed to generate with original prompt")
                self.priority_coordinator.mark_priority_job_end(task.task_id)
                return None
            
            # Step 2: Generate with optimized prompt
            self.logger.info(f"🔄 Step 2: Generating with optimized prompt")
            optimization_result = self.optimize_prompt_for_generation(task)
            optimized_prompt = optimization_result['optimized_prompt']
            lora_info = optimization_result['lora_info']
            endpoint = optimization_result['endpoint']
            
            # Clean the optimized prompt
            cleaned_prompt = self.clean_optimized_prompt_wbgmsst(optimized_prompt)
            if "white background" not in cleaned_prompt.lower():
                cleaned_prompt = cleaned_prompt + " front view, white background"
                # cleaned_prompt = cleaned_prompt + " 3D isometric game asset, white background, object"
            optimized_result = await self._generate_single_3d_model(task, cleaned_prompt, "optimized", endpoint)
            if not optimized_result:
                self.logger.error(f"❌ Failed to generate with optimized prompt")
                # Fallback to original result
                self.logger.info(f"🔄 Falling back to original generation result")
                self.priority_coordinator.mark_priority_job_end(task.task_id)
                return original_result
            
            # Step 3: Generate images for CLIP score comparison
            self.logger.info(f"🔄 Step 3: Generating images for CLIP score comparison")
            original_image = await self._generate_image_for_clip(task, task.prompt, endpoint)
            optimized_image = await self._generate_image_for_clip(task, cleaned_prompt, endpoint)
            
            if not original_image or not optimized_image:
                self.logger.warning(f"⚠️ Failed to generate images for CLIP comparison, using original result")
                self.priority_coordinator.mark_priority_job_end(task.task_id)
                return original_result
            
            # Step 4: Calculate CLIP scores
            self.logger.info(f"🔄 Step 4: Calculating CLIP scores")
            clip_scores = await self._calculate_clip_scores(
                task.prompt, cleaned_prompt, original_image, optimized_image
            )
            
            if not clip_scores:
                self.logger.warning(f"⚠️ Failed to calculate CLIP scores, using original result")
                self.priority_coordinator.mark_priority_job_end(task.task_id)
                return original_result
            
            # Step 5: Analyze CLIP scores and decide on fallback
            original_vs_original = clip_scores['original_prompt_vs_original_image']
            original_vs_optimized = clip_scores['original_prompt_vs_optimized_image']
            optimized_vs_original = clip_scores['optimized_prompt_vs_original_image']
            optimized_vs_optimized = clip_scores['optimized_prompt_vs_optimized_image']
            
            # Calculate the critical ratio for fallback decision
            fallback_ratio = original_vs_optimized / optimized_vs_optimized
            fallback_threshold = self.config.get('fallback_ratio_threshold', 0.8)
            self.logger.info(f"📊 CLIP Score Analysis:")
            self.logger.info(f"   Original prompt vs Original image: {original_vs_original:.4f}")
            self.logger.info(f"   Original prompt vs Optimized image: {original_vs_optimized:.4f}")
            self.logger.info(f"   Optimized prompt vs Original image: {optimized_vs_original:.4f}")
            self.logger.info(f"   Optimized prompt vs Optimized image: {optimized_vs_optimized:.4f}")
            self.logger.info(f"   Fallback ratio (original_vs_optimized / optimized_vs_optimized): {fallback_ratio:.4f}")
            self.logger.info(f"   Fallback threshold: {fallback_threshold:.4f}")
            
            # Step 6: Implement fallback mechanism
            if fallback_ratio < fallback_threshold:
                self.logger.warning(f"⚠️ Fallback ratio {fallback_ratio:.4f} < 0.8, attempting prompt re-optimization")
                
                # Try to get a new optimized prompt using the LLM optimizer with optimized_system_prompt
                new_optimized_prompt = await self._get_new_optimized_prompt(task.prompt)
                if new_optimized_prompt and new_optimized_prompt != cleaned_prompt:
                    self.logger.info(f"🔄 Step 6a: Trying new optimized prompt")
                    
                    # Generate with new prompt
                    new_result = await self._generate_single_3d_model(task, new_optimized_prompt, "new_optimized", endpoint)
                    if new_result:
                        new_image = await self._generate_image_for_clip(task, new_optimized_prompt, endpoint)
                        if new_image:
                            # Calculate new CLIP scores
                            new_clip_scores = await self._calculate_clip_scores(
                                task.prompt, new_optimized_prompt, original_image, new_image
                            )
                            
                            if new_clip_scores:
                                new_fallback_ratio = new_clip_scores['original_prompt_vs_optimized_image'] / new_clip_scores['optimized_prompt_vs_optimized_image']
                                self.logger.info(f"📊 New fallback ratio: {new_fallback_ratio:.4f}")
                                
                                # Use the better result
                                if new_fallback_ratio > fallback_ratio:
                                    self.logger.info(f"✅ New prompt improved fallback ratio, using new result")
                                    self.priority_coordinator.mark_priority_job_end(task.task_id)
                                    return new_result
                                else:
                                    self.logger.info(f"ℹ️ New prompt didn't improve ratio, using original result")
                                    self.priority_coordinator.mark_priority_job_end(task.task_id)
                                    return original_result
                
                # If re-optimization failed or didn't help, use original result
                self.logger.info(f"🔄 Fallback: Using original generation result due to low CLIP alignment")
                self.priority_coordinator.mark_priority_job_end(task.task_id)
                return original_result
            else:
                self.logger.info(f"✅ Fallback ratio {fallback_ratio:.4f} >= 0.8, using optimized result")
                self.priority_coordinator.mark_priority_job_end(task.task_id)
                return optimized_result
        
        except Exception as e:
            self.logger.error(f"❌ Generation with fallback exception: {e}")
            self.priority_coordinator.mark_priority_job_end(task.task_id)
            return None

    async def _generate_single_3d_model(self, task: TaskRecord, prompt: str, prompt_type: str, endpoint: str = None) -> Optional[Dict[str, Any]]:
        """Generate a single 3D model with the given prompt"""
        try:
            if not endpoint:
                # Use default endpoint if none specified
                endpoint = "/generate/"
            
            # Clear cache on the server
            self.priority_coordinator.clear_server_cache()
            
            # Get deterministic seed
            deterministic_seed = self.get_deterministic_seed(task)
            self.logger.info(f"   🎲 Using deterministic seed: {deterministic_seed} for {prompt_type} prompt")
            
            generation_start = time.time()
            
            # Call TRELLIS generation server
            full_url = f"{self.config['generation_server_url']}{endpoint}"
            response = requests.post(
                full_url,
                data={
                    'prompt': prompt,
                    'seed': deterministic_seed,
                    'return_compressed': True
                },
                timeout=self.config['generation_timeout']
            )
            
            generation_time = time.time() - generation_start
            
            if response.status_code == 200:
                ply_data = response.content
                compression_ratio = response.headers.get('X-Compression-Ratio', 'unknown')
                
                self.logger.info(f"✅ {prompt_type.capitalize()} generation successful in {generation_time:.2f}s ({len(ply_data):,} bytes)")
                
                return {'ply_data': ply_data, 'compression_ratio': compression_ratio, 'prompt_type': prompt_type}
            else:
                self.logger.error(f"❌ {prompt_type.capitalize()} generation failed: HTTP {response.status_code}")
                return None
        
        except Exception as e:
            self.logger.error(f"❌ {prompt_type.capitalize()} generation exception: {e}")
            return None

    async def _generate_image_for_clip(self, task: TaskRecord, prompt: str, endpoint: str) -> Optional[bytes]:
        """Generate an image for CLIP score calculation"""
        try:
            # Convert 3D generation endpoint to image generation endpoint
            image_endpoint = endpoint.replace("/generate/", "/generate_image/")
            if not image_endpoint.endswith("/"):
                image_endpoint += "/"
            
            # Get deterministic seed
            deterministic_seed = self.get_deterministic_seed(task)
            
            # Call image generation endpoint
            full_url = f"{self.config['generation_server_url']}{image_endpoint}"
            response = requests.post(
                full_url,
                data={
                    'prompt': prompt,
                    'seed': deterministic_seed
                },
                timeout=self.config['generation_timeout']
            )
            
            if response.status_code == 200:
                image_data = response.content
                self.logger.info(f"✅ Image generated for CLIP analysis ({len(image_data):,} bytes)")
                return image_data
            else:
                self.logger.error(f"❌ Image generation failed: HTTP {response.status_code}")
                return None
        
        except Exception as e:
            self.logger.error(f"❌ Image generation exception: {e}")
            return None

    async def _calculate_clip_scores(self, original_prompt: str, optimized_prompt: str, 
                                   original_image: bytes, optimized_image: bytes) -> Optional[Dict[str, float]]:
        """Calculate CLIP scores between prompts and images using the same logic as clip_alignment_with_generation.py"""
        if not CLIP_AVAILABLE:
            self.logger.error(f"❌ CLIP dependencies not available - cannot calculate scores")
            return None
            
        try:
            
            # Initialize CLIP model if not already loaded
            if not hasattr(self, '_clip_model') or self._clip_model is None:
                self.logger.info(f"🔧 Loading CLIP model for score calculation")
                
                # CLIP model settings (same as production validation)
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                model_name = "convnext_large_d"
                pretrained = "laion2b_s26b_b102k_augreg"
                
                # Load CLIP model
                self._clip_model, _, _ = open_clip.create_model_and_transforms(
                    model_name, 
                    pretrained=pretrained, 
                    device=device
                )
                self._clip_tokenizer = open_clip.get_tokenizer(model_name)
                self._clip_model.eval()
                self._clip_device = device
                
                # Normalization transform (same as production)
                mean = torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(1, 3, 1, 1) * 3
                std = torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(1, 3, 1, 1) * 3
                self._clip_normalize_transform = transforms.Normalize(mean, std)
                
                self.logger.info(f"✅ CLIP model loaded successfully")
            
            # Convert bytes to PIL images
            original_pil = Image.open(io.BytesIO(original_image))
            optimized_pil = Image.open(io.BytesIO(optimized_image))
            
            # Preprocess images for CLIP (same as clip_alignment_with_generation.py)
            def preprocess_image_for_clip(image: Image.Image, image_res: int = 224) -> torch.Tensor:
                # Convert PIL to tensor
                image_tensor = torch.tensor(np.array(image)).float()
                
                # Normalize to [0, 1]
                image_tensor = image_tensor / 255.0
                
                # Convert to channels-first format
                if len(image_tensor.shape) == 3:
                    image_tensor = image_tensor.permute(2, 0, 1)
                
                # Add batch dimension
                image_tensor = image_tensor.unsqueeze(0)
                
                # Resize to CLIP input size
                image_tensor = F.interpolate(image_tensor, size=(image_res, image_res), mode="bicubic", align_corners=False)
                
                # Apply CLIP normalization
                image_tensor = self._clip_normalize_transform(image_tensor)
                
                return image_tensor.to(self._clip_device)
            
            # Preprocess both images
            original_tensor = preprocess_image_for_clip(original_pil)
            optimized_tensor = preprocess_image_for_clip(optimized_pil)
            
            # Tokenize prompts
            tokenized_original = self._clip_tokenizer(original_prompt).to(self._clip_device)
            tokenized_optimized = self._clip_tokenizer(optimized_prompt).to(self._clip_device)
            
            # Compute CLIP scores
            with torch.no_grad(), torch.amp.autocast(self._clip_device.type):
                # Encode images and texts
                original_image_features = self._clip_model.encode_image(original_tensor)
                optimized_image_features = self._clip_model.encode_image(optimized_tensor)
                original_text_features = self._clip_model.encode_text(tokenized_original)
                optimized_text_features = self._clip_model.encode_text(tokenized_optimized)
                
                # Normalize features
                original_image_features /= original_image_features.norm(dim=-1, keepdim=True)
                optimized_image_features /= optimized_image_features.norm(dim=-1, keepdim=True)
                original_text_features /= original_text_features.norm(dim=-1, keepdim=True)
                optimized_text_features /= optimized_text_features.norm(dim=-1, keepdim=True)
                
                # Compute all four alignment scores
                # 1. Original prompt vs Original image
                original_vs_original = (original_text_features @ original_image_features.T).cpu().numpy()[0][0]
                
                # 2. Original prompt vs Optimized image
                original_vs_optimized = (original_text_features @ optimized_image_features.T).cpu().numpy()[0][0]
                
                # 3. Optimized prompt vs Original image
                optimized_vs_original = (optimized_text_features @ original_image_features.T).cpu().numpy()[0][0]
                
                # 4. Optimized prompt vs Optimized image
                optimized_vs_optimized = (optimized_text_features @ optimized_image_features.T).cpu().numpy()[0][0]
                
                # Clip to [0, 1] range
                original_vs_original = np.clip(original_vs_original, 0, 1)
                original_vs_optimized = np.clip(original_vs_optimized, 0, 1)
                optimized_vs_original = np.clip(optimized_vs_original, 0, 1)
                optimized_vs_optimized = np.clip(optimized_vs_optimized, 0, 1)
                
                clip_scores = {
                    'original_prompt_vs_original_image': float(original_vs_original),
                    'original_prompt_vs_optimized_image': float(original_vs_optimized),
                    'optimized_prompt_vs_optimized_image': float(optimized_vs_optimized),
                    'optimized_prompt_vs_original_image': float(optimized_vs_original)
                }
                
                self.logger.info(f"📊 CLIP scores calculated successfully")
                return clip_scores
        
        except Exception as e:
            self.logger.error(f"❌ CLIP score calculation exception: {e}")
            return None

    async def _get_new_optimized_prompt(self, original_prompt: str) -> Optional[str]:
        """Get a new optimized prompt using the LLM optimizer with optimized_system_prompt"""
        try:
            # Import the LLM optimizer
            from llm_prompt_optimizer_v12_f1_lora import LLMPromptOptimizer
            
            # Initialize optimizer with vLLM
            optimizer = LLMPromptOptimizer(
                use_vllm=True,
                vllm_url=self.config.get('vllm_url', 'http://localhost:9000'),
                vllm_model=self.config.get('vllm_model', 'llama-3-2-3b-it')
            )
            
            # Get the optimized system prompt using the function from the file
            system_prompt = optimized_system_prompt(original_prompt)
            
            # Query vLLM directly with the system prompt
            new_prompt = optimizer._query_vllm(system_prompt, original_prompt)
            
            if new_prompt and new_prompt != original_prompt:
                # Clean the response
                new_prompt = optimizer._clean_response(new_prompt)
                self.logger.info(f"🔄 New optimized prompt generated: '{new_prompt[:100]}...'")
                return new_prompt
            else:
                self.logger.warning(f"⚠️ Failed to generate new optimized prompt")
                return None
        
        except Exception as e:
            self.logger.error(f"❌ New prompt optimization exception: {e}")
            return None

    '''
    async def generate_3d_model_clip(self, task: TaskRecord) -> Optional[Dict[str, Any]]:
        """Generate 3D model using TRELLIS server with prompt optimization and CLIP comparison"""
        self.logger.info(f"🎨 Generating 3D model with CLIP comparison: '{task.prompt}' (task: {task.task_id})")
        
        try:
            # CRITICAL: Wait for priority access to the server
            # This is where we ensure subnet tasks get priority over optimizer tasks
            if not self.priority_coordinator.wait_for_priority_access(task.task_id):
                self.logger.error(f"❌ PRIORITY ACCESS TIMEOUT for task {task.task_id} - subnet task will be missed!")
                task.priority_access_timeout = True  # Mark this task as having priority access timeout
                return None
            
            # Mark the start of our priority job
            self.priority_coordinator.mark_priority_job_start(task.task_id, task.prompt)
            
            # Step 1: Optimize prompt and route to optimal LoRA
            optimization_result = self.optimize_prompt_for_generation(task)
            optimized_prompt = optimization_result['optimized_prompt']
            lora_info = optimization_result['lora_info']
            endpoint = optimization_result['endpoint']
            
            # Step 1.5: Clean the optimized prompt to remove artifacts
            cleaned_prompt = self.clean_optimized_prompt_wbgmsst(optimized_prompt)
            # Only add "white background" if it's not already present
            # cleaned_prompt = optimized_prompt
            if "white background" not in cleaned_prompt.lower():
                cleaned_prompt = cleaned_prompt + " white background"
            # Log the final optimization result
            if self.config.get('log_optimization_details', True):
                if optimized_prompt != task.prompt:
                    self.logger.info(f"🎯 FINAL OPTIMIZATION RESULT:")
                    self.logger.info(f"   Original: '{task.prompt}'")
                    self.logger.info(f"   Optimized: '{optimized_prompt}'")
                    self.logger.info(f"   Cleaned: '{cleaned_prompt}'")
                    self.logger.info(f"   LoRA: {lora_info['lora_name']} via {endpoint}")
                else:
                    self.logger.info(f"ℹ️ No optimization applied - using original prompt")
                    self.logger.info(f"   Prompt: '{task.prompt}'")
                    self.logger.info(f"   LoRA: {lora_info['lora_name']} via {endpoint}")
            
            # Clear cache on the server using priority coordinator
            self.priority_coordinator.clear_server_cache()

            # Step 2: Get deterministic seed
            deterministic_seed = self.get_deterministic_seed(task)
            self.logger.info(f"   🎲 Using deterministic seed: {deterministic_seed}")
            self.logger.info(f"   🎨 Using LoRA: {lora_info['lora_name']} via {endpoint}")
            
            generation_start = time.time()
            
            # Step 3: Generate both prompts in parallel using asyncio
            self.logger.info(f"🚀 Starting parallel generation for both prompts")
            
            # Use the preloaded CLIP analyzer
            clip_analyzer = self.get_clip_analyzer()
            if clip_analyzer is None:
                self.logger.error("❌ CLIP analyzer not available")
                self.priority_coordinator.mark_priority_job_end(task.task_id)
                return None
            
            # Generate both prompts in parallel using the new /generate_both/ endpoint
            async def generate_single_prompt(prompt: str, is_optimized: bool = False, port: int = None):
                """Generate a single prompt and return results using the server endpoint"""
                try:
                    import aiohttp
                    import base64
                    from PIL import Image
                    import io
                    
                    # Prepare the request data
                    request_data = {
                        'prompt': prompt,
                        'seed': deterministic_seed,
                        'num_inference_steps': self.config.get('num_inference_steps', 7),
                        'guidance_scale': self.config.get('guidance_scale', 3.5),
                        'ss_sampling_steps': self.config.get('ss_sampling_steps', 21),
                        'slat_sampling_steps': self.config.get('slat_sampling_steps', 24),
                        'slat_guidance_strength': self.config.get('slat_guidance_strength', 4.0),
                        'ss_guidance_strength': self.config.get('ss_guidance_strength', 9.5)
                    }
                    
                    # Convert endpoint to /generate_both/ format
                    # Handle cases: /generate, /generate/, /generate/cinema, generate/cinema, etc.
                    if endpoint.startswith('/generate'):
                        # Remove leading slash if present, then split
                        clean_endpoint = endpoint.lstrip('/')
                        if clean_endpoint.startswith('generate/'):
                            # /generate/cinema -> /generate_both/cinema
                            both_endpoint = '/' + clean_endpoint.replace('generate/', 'generate_both/', 1)
                        elif clean_endpoint == 'generate':
                            # /generate -> /generate_both/
                            both_endpoint = '/generate_both/'
                        else:
                            # Fallback
                            both_endpoint = '/generate_both/'
                    elif endpoint.startswith('generate/'):
                        # generate/cinema -> /generate_both/cinema
                        both_endpoint = '/' + endpoint.replace('generate/', 'generate_both/', 1)
                    elif endpoint == 'generate':
                        # generate -> /generate_both/
                        both_endpoint = '/generate_both/'
                    else:
                        # Default fallback
                        both_endpoint = '/generate_both/'
                    
                    # Determine server URL with port
                    if port:
                        # Extract base URL and use specified port
                        try:
                            original_url = self.config['generation_server_url']
                            if '://' in original_url:
                                # Handle http://localhost:8097 -> http://localhost:8096
                                protocol_and_host = original_url.split(':')[0] + ':' + original_url.split(':')[1]
                                server_url = f"{protocol_and_host}:{port}"
                            else:
                                # Handle localhost:8097 -> http://localhost:8096
                                host = original_url.split(':')[0]
                                server_url = f"http://{host}:{port}"
                        except Exception as e:
                            self.logger.error(f"❌ URL construction failed: {e}")
                            server_url = self.config['generation_server_url']
                    else:
                        server_url = self.config['generation_server_url']
                    
                    self.logger.debug(f"🌐 Sending request to: {server_url}{both_endpoint}")
                    
                    # Send request to the server
                    async with aiohttp.ClientSession() as session:
                        async with session.post(
                            f"{server_url}{both_endpoint}",
                            data=request_data,
                            timeout=aiohttp.ClientTimeout(total=300)  # 5 minutes timeout
                        ) as response:
                            if response.status == 200:
                                result = await response.json()
                                
                                # Decode the base64 image
                                image_data = base64.b64decode(result['image'])
                                image = Image.open(io.BytesIO(image_data))
                                
                                # Get PLY data (always compressed when available)
                                if 'compressed_ply' in result:
                                    ply_data = base64.b64decode(result['compressed_ply'])
                                    compressed_data = ply_data  # Already compressed
                                elif 'ply_data' in result:
                                    # Fallback to uncompressed PLY if compression failed
                                    ply_data = base64.b64decode(result['ply_data'])
                                    compressed_data = None
                                else:
                                    self.logger.error("❌ No PLY data received from server")
                                    return None
                                
                                return {
                                    'ply_data': ply_data,
                                    'compressed_data': compressed_data,
                                    'image': image,
                                    'is_optimized': is_optimized,
                                    'prompt': prompt
                                }
                            else:
                                self.logger.error(f"❌ Server request failed: {response.status}")
                                return None
                    
                except Exception as e:
                    self.logger.error(f"❌ Generation failed for prompt '{prompt[:50]}...': {e}")
                    return None

            # Generate both prompts in parallel using the new /generate_both/ endpoint
            async def generate_single_prompt_endpoint_specific(prompt: str, is_optimized: bool = False, port: int = None, specific_endpoint: str = None):
                """Generate a single prompt and return results using the server endpoint, both for image+ply"""
                try:
                    import aiohttp
                    import base64
                    from PIL import Image
                    import io
                    
                    # Prepare the request data
                    request_data = {
                        'prompt': prompt,
                        'seed': deterministic_seed,
                        'num_inference_steps': self.config.get('num_inference_steps', 7),
                        'guidance_scale': self.config.get('guidance_scale', 3.5),
                        'ss_sampling_steps': self.config.get('ss_sampling_steps', 21),
                        'slat_sampling_steps': self.config.get('slat_sampling_steps', 24),
                        'slat_guidance_strength': self.config.get('slat_guidance_strength', 4.0),
                        'ss_guidance_strength': self.config.get('ss_guidance_strength', 9.5)
                    }
                    
                    
                    both_endpoint  = specific_endpoint
                    # Determine server URL with port
                    if port:
                        # Extract base URL and use specified port
                        try:
                            original_url = self.config['generation_server_url']
                            if '://' in original_url:
                                # Handle http://localhost:8097 -> http://localhost:8096
                                protocol_and_host = original_url.split(':')[0] + ':' + original_url.split(':')[1]
                                server_url = f"{protocol_and_host}:{port}"
                            else:
                                # Handle localhost:8097 -> http://localhost:8096
                                host = original_url.split(':')[0]
                                server_url = f"http://{host}:{port}"
                        except Exception as e:
                            self.logger.error(f"❌ URL construction failed: {e}")
                            server_url = self.config['generation_server_url']
                    else:
                        server_url = self.config['generation_server_url']
                    
                    self.logger.debug(f"🌐 Sending request to: {server_url}{both_endpoint}")
                    
                    # Send request to the server
                    async with aiohttp.ClientSession() as session:
                        async with session.post(
                            f"{server_url}{both_endpoint}",
                            data=request_data,
                            timeout=aiohttp.ClientTimeout(total=300)  # 5 minutes timeout
                        ) as response:
                            if response.status == 200:
                                result = await response.json()
                                
                                # Decode the base64 image
                                image_data = base64.b64decode(result['image'])
                                image = Image.open(io.BytesIO(image_data))
                                
                                # Get PLY data (always compressed when available)
                                if 'compressed_ply' in result:
                                    ply_data = base64.b64decode(result['compressed_ply'])
                                    compressed_data = ply_data  # Already compressed
                                elif 'ply_data' in result:
                                    # Fallback to uncompressed PLY if compression failed
                                    ply_data = base64.b64decode(result['ply_data'])
                                    compressed_data = None
                                else:
                                    self.logger.error("❌ No PLY data received from server")
                                    return None
                                
                                return {
                                    'ply_data': ply_data,
                                    'compressed_data': compressed_data,
                                    'image': image,
                                    'is_optimized': is_optimized,
                                    'prompt': prompt
                                }
                            else:
                                self.logger.error(f"❌ Server request failed: {response.status}")
                                return None
                    
                except Exception as e:
                    self.logger.error(f"❌ Generation failed for prompt '{prompt[:50]}...': {e}")
                    return None
            
            # Run both generations in parallel on different servers
            # Use different ports to avoid CUDA conflicts and run truly in parallel
            # self.logger.info(f"🔗 Original prompt → Server :8097")
            # self.logger.info(f"🔗 Optimized prompt → Server :8099")
            # original_task = generate_single_prompt(task.prompt, is_optimized=False, port=8097)
            # optimized_task = generate_single_prompt(cleaned_prompt, is_optimized=True, port=8099)

            # for flux schnell mode 
            self.logger.info(f"🔗 Original prompt /generate_both/ → Server :8097")
            self.logger.info(f"🔗 Optimized prompt /generate_both/cinema → Server :8099")
            original_task_flux = generate_single_prompt_endpoint_specific(task.prompt, is_optimized=False, port=8097, specific_endpoint="/generate_both/")
            optimized_task_flux = generate_single_prompt_endpoint_specific(cleaned_prompt, is_optimized=True, port=8099, specific_endpoint="/generate_both/cinema")
            
            # Wait for both to complete in parallel
            original_result, optimized_result = await asyncio.gather(original_task_flux, optimized_task_flux)
            
            if original_result is None or optimized_result is None:
                self.logger.error(f"❌ One or both generations failed")
                self.priority_coordinator.mark_priority_job_end(task.task_id)
                return None
            
            # Step 4: Compute CLIP scores for comparison
            self.logger.info(f"🎯 Computing CLIP alignment scores for comparison")
            
            try:
                # Get images from results
                original_image = original_result['image']
                optimized_image = optimized_result['image']
                
                # Ensure images are RGB for CLIP processing
                if original_image.mode != 'RGB':
                    original_image = original_image.convert('RGB')
                if optimized_image.mode != 'RGB':
                    optimized_image = optimized_image.convert('RGB')
                
                # Compute CLIP scores: original prompt vs both images
                original_vs_original = clip_analyzer.compute_clip_alignment_score(task.prompt, original_image)
                original_vs_optimized = clip_analyzer.compute_clip_alignment_score(task.prompt, optimized_image)
                
                self.logger.info(f"✅ CLIP scores computed:")
                self.logger.info(f"   Original prompt + Original image: {original_vs_original:.4f}")
                self.logger.info(f"   Original prompt + Optimized image: {original_vs_optimized:.4f}")
                
                # Step 5: Select the better result based on CLIP score
                if original_vs_original >= original_vs_optimized:
                    self.logger.info(f"✅ Using result from ORIGINAL prompt (CLIP: {original_vs_original:.4f} vs {original_vs_optimized:.4f})")
                    selected_result = original_result
                    selected_prompt = task.prompt
                else:
                    self.logger.info(f"✅ Using result from OPTIMIZED prompt (CLIP: {original_vs_optimized:.4f} vs {original_vs_original:.4f})")
                    selected_result = optimized_result
                    selected_prompt = cleaned_prompt
                
                # Extract the selected PLY data
                ply_data = selected_result['ply_data']
                compressed_data = selected_result['compressed_data']
                
                # Calculate compression ratio
                if compressed_data:
                    compression_ratio = len(ply_data) / len(compressed_data)
                else:
                    compression_ratio = 1.0
                
                generation_time = time.time() - generation_start
                task.generation_time = generation_time
                
                # Save PLY file if configured
                if self.config.get('save_intermediate_results', False):
                    timestamp = int(time.time())
                    ply_file = self.output_dir / f"task_{task.task_id}_{timestamp}.ply.spz"
                    with open(ply_file, 'wb') as f:
                        f.write(ply_data)
                    task.compressed_file_path = str(ply_file)
                
                self.logger.info(f"✅ CLIP-guided generation successful in {generation_time:.2f}s")
                self.logger.info(f"   Selected prompt: '{selected_prompt[:60]}...'")
                self.logger.info(f"   PLY size: {len(ply_data):,} bytes")
                self.logger.info(f"   Compression ratio: {compression_ratio:.2f}")
                
                self.stats['successful_generations'] += 1
                self.stats['total_generation_time'] += generation_time
                
                # Mark the completion of our priority job
                self.priority_coordinator.mark_priority_job_end(task.task_id)
                
                return {
                    'ply_data': ply_data, 
                    'compression_ratio': compression_ratio,
                    'selected_prompt': selected_prompt,
                    'clip_scores': {
                        'original_vs_original': original_vs_original,
                        'original_vs_optimized': original_vs_optimized
                    }
                }
                
            except Exception as e:
                self.logger.error(f"❌ CLIP scoring failed: {e}")
                # Fallback to original prompt result
                self.logger.info(f"🔄 Falling back to original prompt result")
                ply_data = original_result['ply_data']
                compressed_data = original_result['compressed_data']
                
                generation_time = time.time() - generation_start
                task.generation_time = generation_time
                
                self.priority_coordinator.mark_priority_job_end(task.task_id)
                
                return {
                    'ply_data': ply_data, 
                    'compression_ratio': 1.0,
                    'selected_prompt': task.prompt,
                    'clip_scores': {'error': str(e)}
                }
        
        except Exception as e:
            self.logger.error(f"❌ Generation exception: {e}")
            import traceback
            traceback.print_exc()
            
            # Mark the completion of our priority job even on exception
            try:
                self.priority_coordinator.mark_priority_job_end(task.task_id)
            except:
                pass
            
            return None
    '''

    # async def generate_3d_model(self, task: TaskRecord) -> Optional[Dict[str, Any]]:
    #     """Generate 3D model using TRELLIS server with prompt optimization"""
    #     self.logger.info(f"🎨 Generating 3D model: '{task.prompt}' (task: {task.task_id})")
        
    #     try:
    #         # CRITICAL: Wait for priority access to the server
    #         # This is where we ensure subnet tasks get priority over optimizer tasks
    #         if not self.priority_coordinator.wait_for_priority_access(task.task_id):
    #             self.logger.error(f"❌ PRIORITY ACCESS TIMEOUT for task {task.task_id} - subnet task will be missed!")
    #             task.priority_access_timeout = True  # Mark this task as having priority access timeout
    #             return None
            
    #         # Mark the start of our priority job
    #         self.priority_coordinator.mark_priority_job_start(task.task_id, task.prompt)
            
    #         # Step 1: Optimize prompt and route to optimal LoRA
    #         optimization_result = self.optimize_prompt_for_generation(task)
    #         optimized_prompt = optimization_result['optimized_prompt']
    #         lora_info = optimization_result['lora_info']
    #         endpoint = optimization_result['endpoint']
            
    #         # Step 1.5: Clean the optimized prompt to remove artifacts
    #         cleaned_prompt = self.clean_optimized_prompt_wbgmsst(optimized_prompt)
    #         # Only add "white background" if it's not already present
    #         # cleaned_prompt = optimized_prompt
    #         if "white background" not in cleaned_prompt.lower():
    #             cleaned_prompt = cleaned_prompt + " white background"
    #         # Log the final optimization result
    #         if self.config.get('log_optimization_details', True):
    #             if optimized_prompt != task.prompt:
    #                 self.logger.info(f"🎯 FINAL OPTIMIZATION RESULT:")
    #                 self.logger.info(f"   Original: '{task.prompt}'")
    #                 self.logger.info(f"   Optimized: '{optimized_prompt}'")
    #                 self.logger.info(f"   Cleaned: '{cleaned_prompt}'")
    #                 self.logger.info(f"   LoRA: {lora_info['lora_name']} via {endpoint}")
    #             else:
    #                 self.logger.info(f"ℹ️ No optimization applied - using original prompt")
    #                 self.logger.info(f"   Prompt: '{task.prompt}'")
    #                 self.logger.info(f"   LoRA: {lora_info['lora_name']} via {endpoint}")
            
    #         # Clear cache on the server using priority coordinator
    #         self.priority_coordinator.clear_server_cache()

    #         # Step 2: Get deterministic seed
    #         deterministic_seed = self.get_deterministic_seed(task)
    #         self.logger.info(f"   🎲 Using deterministic seed: {deterministic_seed}")
    #         self.logger.info(f"   �� Using LoRA: {lora_info['lora_name']} via {endpoint}")
            
    #         generation_start = time.time()
            
    #         # Call TRELLIS generation server with cleaned prompt, deterministic seed, and LoRA-specific endpoint
    #         full_url = f"{self.config['generation_server_url']}{endpoint}"

    #         port1 = 8099
    #         port2 = 8097
    #         num_inference_steps = GENERATION_CONFIG['num_inference_steps_t2i']
    #         guidance_scale = GENERATION_CONFIG['guidance_scale']
    #         ss_sampling_steps = GENERATION_CONFIG['ss_sampling_steps']
    #         slat_sampling_steps = GENERATION_CONFIG['slat_sampling_steps']
    #         slat_guidance_strength = GENERATION_CONFIG['slat_guidance_strength']
    #         ss_guidance_strength = GENERATION_CONFIG['ss_guidance_strength']
            
    #         # Run both validators in parallel using asyncio.gather
    #         self.logger.info(f"🚀 Starting parallel validation on ports {port1} and {port2}")
            
    #         # Create tasks for both validators
    #         task1 = run_validator_async(
    #             task.prompt, task.prompt, endpoint, port1, num_inference_steps, guidance_scale,
    #             ss_sampling_steps, slat_sampling_steps, slat_guidance_strength, ss_guidance_strength
    #         )
            
    #         task2 = run_validator_async(
    #             task.prompt, cleaned_prompt, endpoint, port2, num_inference_steps, guidance_scale,
    #             ss_sampling_steps, slat_sampling_steps, slat_guidance_strength, ss_guidance_strength
    #         )

    #         generator = TrellisGenerator()
    #         task1 = generator.generate_3d_model(task.prompt, num_inference_steps, guidance_scale, ss_sampling_steps, slat_sampling_steps, slat_guidance_strength, ss_guidance_strength)
    #         task2 = generator.generate_3d_model(cleaned_prompt, num_inference_steps, guidance_scale, ss_sampling_steps, slat_sampling_steps, slat_guidance_strength, ss_guidance_strength)
    #         imgae1 = task1.image
    #         image2 = task2.image
    #         # Wait for both validators to complete
    #         original_results1, original_results2 = await asyncio.gather(task1, task2)
            
    #         self.logger.info(f"✅ Both validators completed in parallel")

    #         ply_data = None
    #         compression_ratio = None

    #         if original_results1.get('validation_engine_score', None) > original_results2.get('validation_engine_score', None):
    #             self.logger.info(f"✅ Using result from original prompt: {task.prompt}")
    #             ply_data = original_results1.get('ply_data', None)
    #             compression_ratio = original_results1.get('compression', 'unknown')
    #         else:
    #             self.logger.info(f"✅ Using result from cleaned prompt: {cleaned_prompt}")
    #             ply_data = original_results2.get('ply_data', None)
    #             compression_ratio = original_results2.get('compression', 'unknown')
            
    #         if ply_data:
    #             generation_time = time.time() - generation_start
    #             task.generation_time = generation_time

    #             # Save PLY file
    #             if self.config['save_intermediate_results']:
    #                 timestamp = int(time.time())
    #                 ply_file = self.output_dir / f"task_{task.task_id}_{timestamp}.ply.spz"
    #                 with open(ply_file, 'wb') as f:
    #                     f.write(ply_data)
    #                 task.compressed_file_path = str(ply_file)
                
    #             self.logger.info(f"✅ Generation successful in {generation_time:.2f}s ({len(ply_data):,} bytes)")
                
    #             self.stats['successful_generations'] += 1
    #             self.stats['total_generation_time'] += generation_time
                
    #             # Mark the completion of our priority job
    #             self.priority_coordinator.mark_priority_job_end(task.task_id)
                
    #             return {'ply_data': ply_data, 'compression_ratio': compression_ratio}
    #         else:
    #             self.logger.error(f"❌ Generation failed: HTTP {response.status_code}")
    #             # Mark the completion of our priority job even on failure
    #             self.priority_coordinator.mark_priority_job_end(task.task_id)
    #             return None
        
    #     except Exception as e:
    #         self.logger.error(f"❌ Generation exception: {e}")
    #         # Mark the completion of our priority job even on exception
    #         self.priority_coordinator.mark_priority_job_end(task.task_id)
    #         return None
    
    async def validate_model(self, task: TaskRecord, ply_data: bytes) -> Optional[float]:
        """Validate generated model and update task record"""
        if not self.config['validate_generations']:
            return None
        
        self.logger.info(f"📊 Validating model: '{task.prompt[:50]}...'")
        
        try:
            validation_start = time.time()
            
            # Decompress PLY data for validation
            try:
                import pyspz
                decompressed_data = pyspz.decompress(ply_data)
            except ImportError:
                self.logger.error("❌ pyspz not available")
                return None
            except Exception as e:
                self.logger.error(f"❌ Decompression failed: {e}")
                return None
            
            # Convert to base64
            encoded_data = base64.b64encode(decompressed_data).decode('utf-8')
            
            request_data = {
                "prompt": task.prompt,
                "data": encoded_data,
                "compression": 0,
                "generate_preview": False,
                "preview_score_threshold": 0.8
            }
            
            response = requests.post(
                f"{self.config['validation_server_url']}/validate_txt_to_3d_ply/",
                json=request_data,
                timeout=self.config['validation_timeout']
            )
            
            validation_time = time.time() - validation_start
            task.validation_time = validation_time
            
            if response.status_code == 200:
                result = response.json()
                score = result.get("score", 0.0)
                task.local_validation_score = score
                
                self.logger.info(f"✅ Validation completed in {validation_time:.2f}s")
                self.logger.info(f"   Score: {score:.4f}, IQA: {result.get('iqa', 0):.3f}")
                self.logger.info(f"   Alignment: {result.get('alignment_score', 0):.3f}")
                
                self.stats['successful_validations'] += 1
                self.stats['total_validation_time'] += validation_time
                
                return score
            else:
                self.logger.error(f"❌ Validation failed: HTTP {response.status_code}")
                return None
        
        except Exception as e:
            self.logger.error(f"❌ Validation exception: {e}")
            return None
    
    async def submit_result(self, task: TaskRecord, generation_result: Dict[str, Any], validator: ValidatorState) -> bool:
        """Submit result to validator and process feedback"""
        if not self.config['submit_results']:
            return True
        
        self.logger.info(f"📤 Submitting result: {task.task_id}")
        
        try:
            if not self._setup_bittensor():
                return False
            
            # Import protocol
            from neurons.common.protocol import SubmitResults, Task
            
            # Get validator info
            if task.validator_uid >= len(self.metagraph.neurons):
                self.logger.error(f"❌ Validator UID {task.validator_uid} not found")
                return False
            
            neuron = self.metagraph.neurons[task.validator_uid]
            
            # Create task object
            task_obj = Task(id=task.task_id, prompt=task.prompt)
            
            # Get data from TRELLIS server - these are SPZ-compressed bytes
            ply_data = generation_result['ply_data']
            
            # The 'results' field in SubmitResults synapse requires a base64-encoded STRING.
            # The TRELLIS server already provides SPZ-compressed bytes, so we just need to base64 encode them.
            self.logger.info(f"   📦 Using SPZ-compressed data from server ({len(ply_data):,} bytes)")
            encoded_data = base64.b64encode(ply_data).decode('utf-8')

            # Create submission
            submit_time = time.time_ns()
            
            try:
                from neurons.common.miner_license_consent_declaration import MINER_LICENSE_CONSENT_DECLARATION
            except ImportError:
                # MINER_LICENSE_CONSENT_DECLARATION = "I, as a miner on SN17, have obtained all licenses, rights and consents required to use, reproduce, modify, display, distribute and make available my submitted results to this subnet and its end users"
                MINER_LICENSE_CONSENT_DECLARATION = "I, as a miner on SN17, have obtained all licenses, rights and consents required to use, reproduce, modify, display, distribute and make available my submitted results to this subnet and its end users"

            message = f"{MINER_LICENSE_CONSENT_DECLARATION}{submit_time}{task.prompt}{neuron.hotkey}{self.wallet.hotkey.ss58_address}"
            signature = base64.b64encode(self.dendrite.keypair.sign(message)).decode('utf-8')
            
            synapse = SubmitResults(
                task=task_obj,
                results=encoded_data,
                compression=2,  # spz compression
                submit_time=submit_time,
                signature=signature
            )
            
            synapse.timeout = self.config['submission_timeout']
            
            start_time = time.time()
            
            # Submit to validator using the correct API call
            response = await self.dendrite.call(
                target_axon=neuron.axon_info,
                synapse=synapse,
                deserialize=False,
                timeout=self.config['submission_timeout']
            )
            
            # 🔍 DEBUG: Log the complete validator response to see what's being sent
            self.logger.info(f"🔍 DEBUG: Validator submission response type: {type(response)}")
            self.logger.info(f"🔍 DEBUG: Validator submission response attributes: {dir(response)}")
            self.logger.info(f"🔍 DEBUG: Validator submission response data: {response.__dict__}")
            
            # 🔍 DEBUG: Check specifically for cooldown and violation fields
            if hasattr(response, 'cooldown_until'):
                self.logger.info(f"🔍 DEBUG: cooldown_until found: {response.cooldown_until}")
            else:
                self.logger.info(f"🔍 DEBUG: cooldown_until NOT found in response")
                
            if hasattr(response, 'cooldown_violations'):
                self.logger.info(f"🔍 DEBUG: cooldown_violations found: {response.cooldown_violations}")
            else:
                self.logger.info(f"🔍 DEBUG: cooldown_violations NOT found in response")
            

            submit_time_elapsed = time.time() - start_time
            print("time elapsed: ", submit_time_elapsed)
            task.submitted_at = time.time()
            
            # Calculate total processing time from validator response to submission
            if task.pulled_at:
                task.total_processing_time = task.submitted_at - task.pulled_at
                self.logger.info(f"⏱️ Total processing time: {task.total_processing_time:.2f}s (from validator response to submission)")
            
            response_data = {}
            # 🚨 CRITICAL: Extract cooldown and violation data from validator response
            # This is where the validator sends the actual cooldown and violation information

            if hasattr(response, 'throttle_period'):
                response_data['throttle_period'] = response.throttle_period

            if hasattr(response, 'cooldown_until'):
                current_time = time.time()
                original_cooldown = response.cooldown_until

                if original_cooldown > current_time:
                    remaining_cooldown = original_cooldown - current_time
                    self.logger.warning(f"✅  CRITICAL: Validator UID {validator.uid} enforced cooldown: {remaining_cooldown:.1f}s remaining")
                    
                    # FIX 3: Use the original cooldown, don't rely on traffic because they are used by the validator
                    # response_data['cooldown_until'] = original_cooldown + 10  # Add 1 second to ensure we pass the cooldown
                    response_data['cooldown_until'] = original_cooldown
                    # FIX: Use safe cooldown setting method
                    self._safe_set_cooldown(validator, original_cooldown + 5)
                    self.logger.debug(f"🛡️ Using original cooldown from validator: {original_cooldown} {original_cooldown + 5}")


                    # FIX 1: Only set emergency cooldowns for actual violations, not normal sync
                    # self._set_emergency_cooldown(validator, response.cooldown_until, "Validator enforced cooldown")
                else:
                    self.logger.info(f"✅✅  Validator UID {validator.uid} cooldown cleared: {response.cooldown_until}")
            
            
            if hasattr(response, 'cooldown_violations'):
                response_data['cooldown_violations'] = response.cooldown_violations

                old_violations = getattr(validator, 'cooldown_violations', 0)
                new_violations = response.cooldown_violations
                
                if new_violations != old_violations:
                    validator.cooldown_violations = new_violations
                    if new_violations > old_violations:
                        violation_increase = new_violations - old_violations
                        self.logger.error(f"🚨 CRITICAL: Validator UID {validator.uid} violations increased: {old_violations} → {new_violations} (+{violation_increase})")
                        
                        # EMERGENCY: Check for critical violation thresholds
                        critical_threshold = self.config.get('critical_violation_threshold', VIOLATION_INCREASE_DELTA)
                        if new_violations > critical_threshold:
                            self.logger.error(f"🚨 EMERGENCY: UID {validator.uid} exceeds critical threshold ({critical_threshold}) - implementing immediate blacklist!")
                            self._blacklist_validator_temporarily(validator, new_violations)
                        
                        # EMERGENCY: Check for rapid violation increase
                        if violation_increase > 20:
                            self.logger.error(f"🚨 EMERGENCY: UID {validator.uid} violations increased by {violation_increase} - implementing emergency measures!")
                            self._handle_critical_violations(validator, new_violations)
                        
                        # EMERGENCY: Set immediate cooldown for high violations
                        if new_violations > 50:
                            # DEPRECATED: Hardcoded 1800s - now using FAILED_VALIDATOR_DELAY * 2 constant
                            # emergency_cooldown = time.time() + 1800  # 30 minutes
                            emergency_cooldown = time.time() + (FAILED_VALIDATOR_DELAY * 2)  # 30 minutes
                            # validator.cooldown_until = emergency_cooldown
                            # FIX: Use safe cooldown setting method
                            self._safe_set_cooldown(validator, emergency_cooldown)
                            self.logger.error(f"🚨 EMERGENCY: Set 30-minute cooldown for UID {validator.uid} due to {new_violations} violations!")
                    
                    # UPDATE: Immediately update local violation count for real-time tracking
                    # DEPRECATED: cooldown_violations field - now using validator_reported_violations
                    # validator.cooldown_violations = new_violations
                    validator.validator_reported_violations = new_violations
                    # STATS: Update violation statistics
                    self.stats['cooldown_violations_total'] = max(self.stats.get('cooldown_violations_total', 0), new_violations)
                    self.stats['critical_violations_detected'] = self.stats.get('critical_violations_detected', 0) + 1
                    
                    # ALERT: Log detailed violation analysis
                    self.logger.error(f"🚨 VIOLATION ANALYSIS for UID {validator.uid}:")
                    self.logger.error(f"   Current violations: {new_violations}")
                    self.logger.error(f"   Previous violations: {old_violations}")
                    self.logger.error(f"   Increase: {new_violations - old_violations}")
                    self.logger.error(f"   Stake: {validator.stake:.1f} TAO")
                    self.logger.error(f"   Trust: {getattr(validator, 'trust', 'N/A')}")
            
            # Process feedback scores if available
            if response and hasattr(response, 'feedback') and response.feedback:
                feedback = response.feedback
                task.feedback_received = True
                task.submission_success = True
                task.task_fidelity_score = feedback.task_fidelity_score
                task.average_fidelity_score = feedback.average_fidelity_score
                task.current_miner_reward = feedback.current_miner_reward
                task.validation_failed = feedback.validation_failed
                task.generations_in_window = feedback.generations_within_the_window
                
                # Update validator statistics
                validator = self.validators[task.validator_uid]
                validator.total_tasks_submitted += 1
                validator.last_submit_time = time.time()
                
                if task.submission_success and task.task_fidelity_score is not None:
                    validator.total_successful_submissions += 1
                    # Update average score with exponential moving average
                    if validator.average_score == 0:
                        validator.average_score = task.task_fidelity_score
                    else:
                        validator.average_score = validator.average_score * 0.9 + task.task_fidelity_score * 0.1
                    
                    # DEPRECATED: Validation lock removed - now using MIN_TASK_INTERVAL constant for rate limiting
                    # Check if we should set validation lock (successful submission)
                    # validation_lock_duration = self.config.get('validation_lock_duration', 30)
                    # if validation_lock_duration > 0:
                    #     self.set_validator_validation_lock(validator, validation_lock_duration, "Successful submission")
                    #     self.logger.debug(f"🔒 Validation lock set for UID {validator.uid} after successful submission")
                else:
                    # Failed submission - increment violations
                    self.increment_cooldown_violations(validator, "Failed submission")
                
                # Update session stats
                self.stats['successful_submissions'] += 1
                if task.current_miner_reward:
                    self.stats['total_rewards'] += task.current_miner_reward
                if task.total_processing_time:
                    self.stats['total_processing_time'] += task.total_processing_time
                
                self.logger.info(f"✅ Submission successful to UID {task.validator_uid} ({submit_time_elapsed:.2f}s)")
                self.logger.info(f"   Task fidelity: {task.task_fidelity_score:.4f}")
                self.logger.info(f"   Average fidelity: {task.average_fidelity_score:.4f}")
                self.logger.info(f"   Miner reward: {task.current_miner_reward:.6f}")
                self.logger.info(f"   Validation failed: {task.validation_failed}")
                self.logger.info(f"   Generations in window: {task.generations_in_window}")
                
                # Log optimization impact if zero fidelity was avoided
                if (self.config.get('enable_prompt_optimization', True) and
                    task.task_fidelity_score > 0.0 and
                    self.stats['optimization_improvements'] > 0):
                    self.logger.info(f"   🎯 Zero fidelity avoided (optimization working!)")

                # Update validator's last submit time (validator-compliant throttle logic)
                validator.last_submit_time = time.time()

                # Track fidelity score and check for endpoint switching
                tracking_result = self.fidelity_tracker.record_task_result(
                    task_id=task.task_id,
                    validator_uid=task.validator_uid,
                    task_fidelity_score=task.task_fidelity_score,
                    endpoint=getattr(task, 'endpoint_used', '/generate/'),  # Track which endpoint was used
                    prompt=task.prompt
                )
                
                # Check if endpoint switching is recommended
                if tracking_result['switching_decision']['should_switch']:
                    self.logger.info(f"🔄 Endpoint switching recommended by fidelity tracker:")
                    self.logger.info(f"   Reason: {tracking_result['switching_decision']['reason']}")
                    self.logger.info(f"   New endpoint: {tracking_result['switching_decision']['new_endpoint']}")
                    
                    # Apply the endpoint switch
                    self.fidelity_tracker.apply_endpoint_switch(
                        validator_uid=task.validator_uid,
                        new_endpoint=tracking_result['switching_decision']['new_endpoint'],
                        reason=tracking_result['switching_decision']['reason']
                    )
                    
                    # Update statistics
                    self.stats['fidelity_tracker_endpoint_switches'] += 1
                    
                    # Track specific types of switches
                    if tracking_result['switching_decision']['type'] == 'to_fallback':
                        self.stats['fidelity_tracker_fallback_endpoint_usage'] += 1
                    elif tracking_result['switching_decision']['type'] == 'to_original':
                        self.stats['fidelity_tracker_original_endpoint_recovery'] += 1
                    
                    # Log the switch for future reference
                    self.logger.info(f"✅ Endpoint switch applied for validator {task.validator_uid}")
                
                # Track zero scores
                if task.task_fidelity_score == 0.0:
                    self.stats['fidelity_tracker_zero_scores_detected'] += 1
                
                # Log fidelity tracking summary
                if self.config.get('log_fidelity_tracking', True):
                    validator_summary = self.fidelity_tracker.get_tracking_summary(task.validator_uid)
                    if validator_summary:
                        self.logger.info(f"📊 Fidelity tracking summary for validator {task.validator_uid}:")
                        self.logger.info(f"   Current endpoint: {validator_summary.get('current_endpoint', 'N/A')}")
                        self.logger.info(f"   Consecutive zeros: {validator_summary.get('consecutive_zeros', 0)}")
                        self.logger.info(f"   Total zeros: {validator_summary.get('total_zeros', 0)}")
                        self.logger.info(f"   Endpoint switches: {validator_summary.get('switches_made', 0)}")

                sync_results = self._synchronize_validator_state(validator, response_data)
                
                # Log synchronization results
                if sync_results['cooldown_updated'] or sync_results['violations_updated']:
                    self.logger.info(f"🔄 State synchronized for UID {validator.uid}")
                    if sync_results['backoff_strategy']:
                        self.logger.info(f"   Backoff strategy: {sync_results['backoff_strategy']}")
                    if sync_results['emergency_actions']:
                        self.logger.info(f"   Emergency actions: {', '.join(sync_results['emergency_actions'])}")
            
                return True
            else:
                self.logger.error(f"❌ No feedback received from UID {task.validator_uid}")
                task.submission_success = False
                
                # Set cooldown for submission failures
                if task.validator_uid in self.validators:
                    validator = self.validators[task.validator_uid]
                    # DEPRECATED: Hardcoded 60s - now using NETWORK_DELAY_TIME_BUFFER constant
                    # submission_cooldown = self.config.get('submission_failure_cooldown', 60)
                    submission_cooldown = self.config.get('submission_failure_cooldown', NETWORK_DELAY_TIME_BUFFER)
                    self.set_validator_cooldown(validator, submission_cooldown, "No feedback received", task.task_id, task.prompt)
                
                return False
        
        except Exception as e:
            self.logger.error(f"❌ Submission failed: {e}")
            traceback.print_exc()
            task.submission_success = False

            # Handle task failure with quality penalties (validator-compliant)
            if task.validator_uid in self.validators:
                validator = self.validators[task.validator_uid]
                self._handle_task_failure(validator, task, 0.0, f"Submission exception: {str(e)[:50]}")

            return False
    
    async def process_task(self, task: TaskRecord, validator: ValidatorState) -> bool:
        """Process a single task end-to-end with priority access"""
        self.logger.info(f"�� Processing task {task.task_id}: '{task.prompt}'")

        task.processed_at = time.time()
        self.stats['tasks_processed'] += 1

        # PREVENT DUPLICATE PROCESSING: Check if this task was already processed recently
        if hasattr(self, '_recently_processed_tasks'):
            if task.task_id in self._recently_processed_tasks:
                last_processed = self._recently_processed_tasks[task.task_id]
                # DEPRECATED: Hardcoded 300s - now using FAILED_VALIDATOR_DELAY constant
                # if time.time() - last_processed < 300:  # 5 minutes window
                if time.time() - last_processed < FAILED_VALIDATOR_DELAY:  # 5 minutes window
                    self.logger.warning(f"🚫 Duplicate task detected - already processed {task.task_id} recently")
                    task.submission_success = False
                    self.db.save_task(task)
                    return False
        else:
            self._recently_processed_tasks = {}

    
        try:
            # Step 1: Generate 3D model with priority access
            generation_result = await self.generate_3d_model(task)
            # generation_result = await self.generate_3d_model_clip(task)
            if not generation_result:
                # Check if this was due to priority access timeout
                if hasattr(task, 'priority_access_timeout') and task.priority_access_timeout:
                    self.logger.error(f"❌ PRIORITY ACCESS TIMEOUT for task {task.task_id} - subnet task missed!")
                    self.stats['priority_access_timeouts'] = self.stats.get('priority_access_timeouts', 0) + 1
                else:
                    self.logger.error(f"❌ Generation failed for task {task.task_id}")
                self.db.save_task(task)
                return False
            
            # ply_data = generation_result['ply_data']

            # # Step 2: Validate locally
            # local_score = await self.validate_model(task, ply_data)
            # if local_score is not None and local_score < self.config['min_local_score']:
            #     self.logger.warning(f"⚠️ Local score too low ({local_score:.3f}), skipping submission")
            #     self.db.save_task(task)
            #     return False
            
            # Calculate time elapsed since task was pulled
            time_after_generation = time.time()
            elapsed_time_since_pull = time_after_generation - task.pulled_at

            # If elapsed time is less than 17 seconds, wait until 18 seconds have passed
            # DEPRECATED: Hardcoded 15.0s and 16.0s - now using MIN_TASK_INTERVAL constant
            # the minimum time delay between pull and submit is 15seconds
            if elapsed_time_since_pull < 15.0:
                wait_duration = 16.0 - elapsed_time_since_pull
                self.logger.info(f"⏳ Elapsed time since pull ({elapsed_time_since_pull:.2f}s) is < 17s. Waiting for {wait_duration:.2f}s to reach 18s before submission.")
                await asyncio.sleep(wait_duration)
                
            # Step 3: Submit results, passing the full generation result dictionary
            success = await self.submit_result(task, generation_result, validator)
            
            # FIXED: Complete pending task cooldown logic
            if success and task.validator_uid in self.validators:
                validator = self.validators[task.validator_uid]
                if validator.pending_cooldown_task_id == task.task_id:
                    validator.pending_cooldown_task_id = None  # Clear pending task
                    self.logger.info(f"✅ Pending cooldown task {task.task_id} completed for UID {validator.uid}")
                    # Now enforce the cooldown if it's still active
                    if validator.validator_enforced_cooldown_until and time.time() < validator.validator_enforced_cooldown_until:
                        remaining = validator.validator_enforced_cooldown_until - time.time()
                        self.logger.info(f"⏳ Enforcing cooldown for UID {validator.uid}: {remaining:.1f}s remaining")
            
            # Save task record
            self.db.save_task(task)
            
            # SHARED TASK TRACKING: Release the task lock
            if self.config.get('enable_task_tracking', True) and not self.config.get('disable_task_tracking', False):
                if success:
                    self.db.release_task_lock(task.task_id, self.instance_id, status='completed')
                    self.logger.info(f"✅ Task {task.task_id} completed successfully")
                    self.logger.info(f"   🔓 Task lock released: {task.task_id}")
                else:
                    self.db.release_task_lock(task.task_id, self.instance_id, status='failed')
                    self.logger.error(f"❌ Task {task.task_id} submission failed")
                    self.logger.info(f"   🔓 Task lock released: {task.task_id}")

            # TRACK PROCESSED TASKS: Add to recently processed list to prevent duplicates
            if success:
                if not hasattr(self, '_recently_processed_tasks'):
                    self._recently_processed_tasks = {}
                self._recently_processed_tasks[task.task_id] = time.time()

                # CLEANUP: Remove old entries from recently processed tasks (older than 10 minutes)
                current_time = time.time()
                old_tasks = [task_id for task_id, processed_time in self._recently_processed_tasks.items()
                           if current_time - processed_time > 600]  # 10 minutes
                for old_task in old_tasks:
                    del self._recently_processed_tasks[old_task]

                if old_tasks:
                    self.logger.debug(f"🧹 Cleaned up {len(old_tasks)} old task entries from recently processed cache")

            return success
        
        except Exception as e:
            self.logger.error(f"❌ Task processing failed: {e}")
            traceback.print_exc()
            self.db.save_task(task)
            
            # SHARED TASK TRACKING: Release the task lock on exception
            if self.config.get('enable_task_tracking', True) and not self.config.get('disable_task_tracking', False):
                self.db.release_task_lock(task.task_id, self.instance_id, status='failed_exception')
                self.logger.info(f"   🔓 Task lock released on exception: {task.task_id}")
            
            return False
    
    async def idle_validation_cycle(self):
        """Perform validation on recent unvalidated generations during idle time"""
        self.logger.info("🔍 Running idle validation cycle...")
        
        try:
            # Get recent unvalidated tasks
            unvalidated_tasks = self.db.get_recent_unvalidated_tasks(hours=2)
            
            if not unvalidated_tasks:
                self.logger.info("   No unvalidated tasks found")
                return
            
            self.logger.info(f"   Found {len(unvalidated_tasks)} unvalidated tasks")
            
            for task in unvalidated_tasks:
                if not self.running:
                    break
                
                # Check if PLY file exists
                if not task.compressed_file_path or not Path(task.compressed_file_path).exists():
                    continue
                
                try:
                    # Load PLY data
                    with open(task.compressed_file_path, 'rb') as f:
                        ply_data = f.read()
                    
                    # Validate
                    score = await self.validate_model(task, ply_data)
                    if score is not None:
                        self.logger.info(f"   Validated task {task.task_id}: score {score:.4f}")
                        self.stats['idle_validations'] += 1
                        
                        # Update task in database
                        self.db.save_task(task)
                
                except Exception as e:
                    self.logger.error(f"   Failed to validate task {task.task_id}: {e}")
        
        except Exception as e:
            self.logger.error(f"❌ Idle validation cycle failed: {e}")
    
    def save_statistics(self):
        """Save comprehensive statistics to JSON file"""
        try:
            uptime_hours = (time.time() - self.start_time) / 3600
            
            # Validator statistics
            validator_stats = {}
            for uid, validator in self.validators.items():
                validator_stats[uid] = {
                    'hotkey': validator.hotkey,
                    'stake': validator.stake,
                    'trust': validator.trust,
                    'consensus': validator.consensus,
                    'total_tasks_pulled': validator.total_tasks_pulled,
                    'total_tasks_received': validator.total_tasks_received,
                    'total_tasks_submitted': validator.total_tasks_submitted,
                    'total_successful_submissions': validator.total_successful_submissions,
                    'average_score': validator.average_score,
                    'success_rate': validator.total_successful_submissions / max(1, validator.total_tasks_submitted),
                    'last_task_received': validator.last_task_received,
                    'is_active': validator.is_active
                }
            
            # Comprehensive statistics
            stats = {
                'timestamp': datetime.now().isoformat(),
                'uptime_hours': uptime_hours,
                'session_stats': self.stats,
                'validator_stats': validator_stats,
                'performance': {
                    'tasks_per_hour': self.stats['tasks_processed'] / max(0.1, uptime_hours),
                    'success_rate': self.stats['successful_submissions'] / max(1, self.stats['tasks_processed']),
                    'avg_generation_time': self.stats['total_generation_time'] / max(1, self.stats['successful_generations']),
                    'avg_validation_time': self.stats['total_validation_time'] / max(1, self.stats['successful_validations']),
                    'avg_total_processing_time': self.stats['total_processing_time'] / max(1, self.stats['successful_submissions']),
                    'total_rewards': self.stats['total_rewards'],
                    'rewards_per_hour': self.stats['total_rewards'] / max(0.1, uptime_hours),
                    'optimization_rate': (self.stats['optimization_improvements'] / max(1, self.stats['prompts_optimized'])) * 100,
                    'prompts_optimized': self.stats['prompts_optimized'],
                    'optimization_improvements': self.stats['optimization_improvements']
                }
            }
            
            # Save to file
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            stats_file = self.output_dir / f"continuous_stats_{timestamp}.json"
            
            with open(stats_file, 'w') as f:
                json.dump(stats, f, indent=2)
            
            self.logger.info(f"📊 Statistics saved to {stats_file}")
            
        except Exception as e:
            self.logger.error(f"❌ Failed to save statistics: {e}")
    
    def print_status(self):
        """Print current status"""
        uptime_hours = (time.time() - self.start_time) / 3600
        
        self.logger.info("📊 CONTINUOUS ORCHESTRATOR STATUS")
        self.logger.info("="*60)
        self.logger.info(f"Uptime: {uptime_hours:.2f} hours")
        self.logger.info(f"Tasks pulled: {self.stats['tasks_pulled']}")
        self.logger.info(f"Tasks processed: {self.stats['tasks_processed']}")
        self.logger.info(f"Successful generations: {self.stats['successful_generations']}")
        self.logger.info(f"Successful validations: {self.stats['successful_validations']}")
        self.logger.info(f"Successful submissions: {self.stats['successful_submissions']}")
        self.logger.info(f"Total rewards: {self.stats['total_rewards']:.6f} TAO")
        self.logger.info(f"Idle validations: {self.stats['idle_validations']}")
        self.logger.info(f"Prompts optimized: {self.stats['prompts_optimized']}")
        self.logger.info(f"Prompts cleaned: {self.stats.get('prompts_cleaned', 0)}")
        self.logger.info(f"Reproducibility optimizations: {self.stats.get('reproducibility_optimizations', 0)}")
        self.logger.info(f"Traditional optimizations: {self.stats.get('traditional_optimizations', 0)}")
        self.logger.info(f"Optimization improvements: {self.stats['optimization_improvements']}")
        
        # Gold prompts statistics
        if REPRODUCIBILITY_SYSTEM_AVAILABLE and self.reproducibility_system:
            self.logger.info(f"Gold prompts available: {self.stats.get('gold_prompts_available', 0)}")
            self.logger.info(f"Gold prompts reloaded: {self.stats.get('gold_prompts_reloaded', 0)}")
            if hasattr(self, 'last_gold_prompts_reload'):
                time_since_reload = time.time() - self.last_gold_prompts_reload
                self.logger.info(f"Time since last gold prompts reload: {time_since_reload/3600:.1f} hours")
            
            # Real-time learning statistics
            if self.config.get('activate_learning', False):
                if self.config.get('only_log_learning', False):
                    log_count = self.config.get('log_learning_count', 6)
                    log_info = "all available logs" if log_count == -1 else f"most recent {log_count} logs"
                    
                    self.logger.info(f"🚀 ONLY-LOG-LEARNING STATISTICS:")
                    self.logger.info(f"   Enhanced gold prompts available: {self.stats.get('enhanced_gold_prompts_available', 0)}")
                    self.logger.info(f"   Enhanced reloads performed: {self.stats.get('enhanced_gold_prompts_reloaded', 0)}")
                    self.logger.info(f"   Total gold prompts (logs only): {self.stats.get('total_gold_prompts_available', 0)}")
                    self.logger.info(f"   From episodic memory: BYPASSED")
                    self.logger.info(f"   From recent logs: {self.stats.get('log_prompts', 0)} ({log_info})")
                    self.logger.info(f"   Live monitoring: DISABLED (logs only)")
                else:
                    self.logger.info(f"🚀 REAL-TIME LEARNING STATISTICS:")
                    self.logger.info(f"   Enhanced gold prompts available: {self.stats.get('enhanced_gold_prompts_available', 0)}")
                    self.logger.info(f"   Enhanced reloads performed: {self.stats.get('enhanced_gold_prompts_reloaded', 0)}")
                    self.logger.info(f"   Total gold prompts (memory + logs): {self.stats.get('total_gold_prompts_available', 0)}")
                    self.logger.info(f"   From episodic memory: {self.stats.get('memory_prompts', 0)}")
                    self.logger.info(f"   From recent logs: {self.stats.get('log_prompts', 0)}")
                    self.logger.info(f"   Live monitoring: ACTIVE")
            else:
                self.logger.info(f"📚 Real-time learning: DISABLED")
        
        # LLM Provider information
        if self.config.get('use_vllm', False):
            self.logger.info(f"🤖 LLM Provider: vLLM ({self.config.get('vllm_url', 'http://localhost:9000')})")
            self.logger.info(f"🤖 vLLM Model: {self.config.get('vllm_model', 'llama-3-2-3b-it')}")
        else:
            self.logger.info(f"🤖 LLM Provider: Ollama ({self.config.get('ollama_url', 'http://localhost:11434')}")
        
        # vLLM optimization statistics
        if self.config.get('use_vllm_optim', False):
            self.logger.info(f"🚀 vLLM Optimization: ENABLED on port {self.config.get('vllm_optim_port', 11300)}")
            if self.config.get('use_system_prompt', False):
                self.logger.info(f"📝 System Prompts: ENABLED")
            else:
                self.logger.info(f"📝 System Prompts: DISABLED")
            priority = self.config.get('vllm_optimization_priority', 'system_chat')
            self.logger.info(f"🎯 vLLM Optimization Priority: {priority}")
            
            # vLLM performance statistics
            self.logger.info(f"📊 vLLM Performance:")
            self.logger.info(f"   Total optimizations: {self.stats.get('vllm_optimizations', 0)}")
            self.logger.info(f"   System chat success: {self.stats.get('vllm_system_chat_success', 0)}")
            self.logger.info(f"   System completions success: {self.stats.get('vllm_system_completions_success', 0)}")
            self.logger.info(f"   No system success: {self.stats.get('vllm_no_system_success', 0)}")
            self.logger.info(f"   Failures: {self.stats.get('vllm_failures', 0)}")
            self.logger.info(f"   Connection tests: {self.stats.get('vllm_connection_tests', 0)}")
            self.logger.info(f"   Connection success: {self.stats.get('vllm_connection_success', 0)}")
        else:
            self.logger.info(f"🔧 vLLM Optimization: DISABLED (using original prompts)")
        
        # LoRA routing statistics
        self.logger.info(f"LoRA routing decisions: {self.stats.get('lora_routing_decisions', 0)}")
        self.logger.info(f"LoRA routing accuracy: {self.stats.get('lora_routing_accuracy', 0.0):.1f}%")
        
        # Priority access statistics
        self.logger.info(f"Priority access timeouts: {self.stats.get('priority_access_timeouts', 0)}")
        self.logger.info(f"Priority interruptions: {self.stats.get('priority_interruptions', 0)}")
        self.logger.info(f"Server unavailable skips: {self.stats.get('server_unavailable_skips', 0)}")
        self.logger.info(f"Server status check errors: {self.stats.get('server_status_check_errors', 0)}")
        
        # Validator blacklisting statistics
        self.logger.info(f"Blacklisted validators skipped: {self.stats.get('blacklisted_validators_skipped', 0)}")
        
        # Enhanced cooldown system statistics
        self.logger.info(f"Enhanced cooldown system:")
        self.logger.info(f"   Total cooldown violations: {self.stats.get('cooldown_violations_total', 0)}")
        # DEPRECATED: Validation lock stats logging removed - now using MIN_TASK_INTERVAL constant for rate limiting
        # self.logger.info(f"   Validation locks applied: {self.stats.get('validation_locks_applied', 0)}")
        self.logger.info(f"   Enhanced cooldown penalties: {self.stats.get('enhanced_cooldown_penalties', 0)}")
        
        # Fidelity score tracking statistics
        if hasattr(self, 'fidelity_tracker'):
            global_summary = self.fidelity_tracker.get_tracking_summary()
            self.logger.info(f"🎯 Fidelity Score Tracking:")
            self.logger.info(f"   Total validators tracked: {global_summary.get('total_validators_tracked', 0)}")
            self.logger.info(f"   Total tasks tracked: {global_summary.get('total_tasks_tracked', 0)}")
            self.logger.info(f"   Global consecutive zeros: {global_summary.get('global_endpoint_state', {}).get('consecutive_zeros', 0)}")
            self.logger.info(f"   Global total zeros: {global_summary.get('global_endpoint_state', {}).get('total_zeros', 0)}")
            self.logger.info(f"   Global endpoint switches: {global_summary.get('global_endpoint_state', {}).get('switches_made', 0)}")
            
            # Show current global endpoint state
            current_endpoint = global_summary.get('global_endpoint_state', {}).get('current_endpoint', 'N/A')
            original_endpoint = global_summary.get('global_endpoint_state', {}).get('original_endpoint', 'N/A')
            if current_endpoint != original_endpoint:
                self.logger.info(f"   🔄 Global endpoint switched from {original_endpoint} to {current_endpoint}")
            else:
                self.logger.info(f"   ✅ Using original global endpoint: {current_endpoint}")
            
            # Show session statistics
            self.logger.info(f"   Session endpoint switches: {self.stats.get('fidelity_tracker_endpoint_switches', 0)}")
            self.logger.info(f"   Session zero scores detected: {self.stats.get('fidelity_tracker_zero_scores_detected', 0)}")
            self.logger.info(f"   Session fallback endpoint usage: {self.stats.get('fidelity_tracker_fallback_endpoint_usage', 0)}")
            self.logger.info(f"   Session original endpoint recovery: {self.stats.get('fidelity_tracker_original_endpoint_recovery', 0)}")
        else:
            self.logger.info(f"🎯 Fidelity Score Tracking: NOT INITIALIZED")
        
        # Emergency cooldown management statistics
        self.logger.info(f"Emergency cooldown management:")
        self.logger.info(f"   Emergency cooldowns applied: {self.stats.get('emergency_cooldowns_applied', 0)}")
        self.logger.info(f"   Critical violations handled: {self.stats.get('critical_violations_handled', 0)}")
        self.logger.info(f"   Critical violations detected: {self.stats.get('critical_violations_detected', 0)}")
        self.logger.info(f"   Validators temporarily blacklisted: {self.stats.get('validators_temporarily_blacklisted', 0)}")
        self.logger.info(f"   Validators reset from emergency: {self.stats.get('validators_reset_from_emergency', 0)}")
        self.logger.info(f"   Dynamic cooldown scaling applied: {self.stats.get('dynamic_cooldown_scaling', 0)}")
        self.logger.info(f"   Dynamic buffer applied: {self.stats.get('dynamic_buffer_applied', 0)}")
        
        # State persistence statistics
        self.logger.info(f"State persistence:")
        self.logger.info(f"   States saved to disk: {self.stats.get('validator_states_saved', 0)}")
        self.logger.info(f"   Save failures: {self.stats.get('validator_state_save_failures', 0)}")
        self.logger.info(f"   Validators restored from disk: {self.stats.get('validators_restored_from_disk', 0)}")
        self.logger.info(f"   Violations restored from disk: {self.stats.get('violations_restored_from_disk', 0)}")
        self.logger.info(f"   Blacklists restored from disk: {self.stats.get('blacklists_restored_from_disk', 0)}")
        
        # Enhanced cooldown statistics with DYNAMIC system health analysis
        active_validators = [v for v in self.validators.values() if v.is_active]
        validators_on_cooldown = [v for v in active_validators if v.cooldown_until and time.time() < v.cooldown_until]
        # DEPRECATED: Validation lock check removed - now using MIN_TASK_INTERVAL constant for rate limiting
        # validators_validation_locked = [v for v in active_validators if v.validation_locked_until and time.time() < v.validation_locked_until]
        validators_with_violations = [v for v in active_validators if v.cooldown_violations > 0]
        validators_emergency_blacklisted = [v for v in self.validators.values() if v.emergency_blacklist_until and time.time() < v.emergency_blacklist_until]
        
        # DYNAMIC: Calculate system health and adjust task pulling strategy
        total_validators = len(self.validators)
        system_health_ratio = len(active_validators) / total_validators if total_validators > 0 else 0
        
        # Adjust task pulling strategy based on system health
        if system_health_ratio < 0.3:  # Critical system state
            task_pull_strategy = "CONSERVATIVE"
            max_concurrent_tasks = max(1, int(self.config.get('max_concurrent_tasks', 5) * 0.3))
            self.logger.error(f"🚨 CRITICAL SYSTEM STATE: Task pulling strategy set to CONSERVATIVE")
            self.logger.error(f"   Max concurrent tasks reduced to {max_concurrent_tasks} (from {self.config.get('max_concurrent_tasks', 5)})")
        elif system_health_ratio < 0.6:  # Degraded system state
            task_pull_strategy = "MODERATE"
            max_concurrent_tasks = max(2, int(self.config.get('max_concurrent_tasks', 5) * 0.6))
            self.logger.warning(f"⚠️ DEGRADED SYSTEM STATE: Task pulling strategy set to MODERATE")
            self.logger.warning(f"   Max concurrent tasks reduced to {max_concurrent_tasks} (from {self.config.get('max_concurrent_tasks', 5)})")
        else:  # Healthy system state
            task_pull_strategy = "AGGRESSIVE"
            max_concurrent_tasks = self.config.get('max_concurrent_tasks', 5)
            self.logger.info(f"✅ HEALTHY SYSTEM STATE: Task pulling strategy set to AGGRESSIVE")
            self.logger.info(f"   Max concurrent tasks: {max_concurrent_tasks}")
        
        # Store dynamic strategy for use in task pulling
        self.current_task_pull_strategy = task_pull_strategy
        self.current_max_concurrent_tasks = max_concurrent_tasks
        
        if validators_on_cooldown or validators_validation_locked or validators_emergency_blacklisted:
            total_restricted = len(validators_on_cooldown) + len(validators_validation_locked) + len(validators_emergency_blacklisted)
            self.logger.info(f"⏳ Validators with restrictions: {len(validators_on_cooldown)} cooldown, {len(validators_validation_locked)} validation locked, {len(validators_emergency_blacklisted)} emergency blacklisted")
            
            # Show emergency blacklisted validators first (most critical)
            if validators_emergency_blacklisted:
                self.logger.warning(f"🚨 EMERGENCY BLACKLISTED VALIDATORS:")
                for validator in validators_emergency_blacklisted[:3]:  # Show first 3
                    cooldown_status = self.get_cooldown_status(validator)
                    self.logger.warning(f"   UID {validator.uid}: {cooldown_status}")
            
            # Show other restricted validators
            other_restricted = validators_on_cooldown + validators_validation_locked
            if other_restricted:
                self.logger.info(f"⏳ Other restricted validators:")
                for validator in other_restricted[:5]:  # Show first 5
                    cooldown_status = self.get_cooldown_status(validator)
                    self.logger.info(f"   UID {validator.uid}: {cooldown_status}")
        else:
            self.logger.info(f"✅ No validators currently restricted")
        
        if validators_with_violations:
            self.logger.info(f"⚠️ Validators with cooldown violations: {len(validators_with_violations)}")
            for validator in validators_with_violations[:3]:  # Show first 3
                self.logger.info(f"   UID {validator.uid}: {validator.cooldown_violations} violations")
        else:
            self.logger.info(f"✅ No validators with cooldown violations")
        
        # DYNAMIC: Log system health summary
        self.logger.info(f"📊 SYSTEM HEALTH SUMMARY:")
        self.logger.info(f"   Total validators: {total_validators}")
        self.logger.info(f"   Active validators: {len(active_validators)} ({system_health_ratio:.1%})")
        self.logger.info(f"   Task pull strategy: {task_pull_strategy}")
        self.logger.info(f"   Max concurrent tasks: {max_concurrent_tasks}")
        blacklist = self.config.get('validator_blacklist', [])
        blacklist_enabled = self.config.get('enable_validator_blacklisting', True)
        if blacklist:
            self.logger.info(f"Current blacklist: {blacklist}")
            self.logger.info(f"Blacklisting: {'ENABLED' if blacklist_enabled else 'DISABLED'}")
            
            # Show which blacklisted UIDs are currently active
            active_blacklisted = [uid for uid in blacklist if uid in self.validators and self.validators[uid].is_active]
            if active_blacklisted:
                self.logger.info(f"🚫 Active blacklisted UIDs: {active_blacklisted}")
            else:
                self.logger.info(f"✅ No blacklisted UIDs are currently active on the subnet")
        else:
            self.logger.info(f"No validators in blacklist")
        
        if uptime_hours > 0:
            self.logger.info(f"Tasks/hour: {self.stats['tasks_processed'] / uptime_hours:.1f}")
            self.logger.info(f"Rewards/hour: {self.stats['total_rewards'] / uptime_hours:.6f} TAO")
            
        # Processing time statistics
        if self.stats['successful_submissions'] > 0:
            avg_processing_time = self.stats['total_processing_time'] / self.stats['successful_submissions']
            self.logger.info(f"Average total processing time: {avg_processing_time:.2f}s")
            
        # Optimization statistics
        if self.stats['prompts_optimized'] > 0:
            optimization_rate = (self.stats['optimization_improvements'] / self.stats['prompts_optimized']) * 100
            self.logger.info(f"Optimization rate: {optimization_rate:.1f}% of prompts improved")
        
        # Active validators
        active_validators = [v for v in self.validators.values() if v.is_active]
        self.logger.info(f"Active validators: {len(active_validators)}")
        
        for validator in sorted(active_validators, key=lambda v: v.stake, reverse=True)[:3]:
            cooldown_status = self.get_cooldown_status(validator)
            self.logger.info(f"  UID {validator.uid}: {validator.total_tasks_received} tasks, avg score: {validator.average_score:.3f}, cooldown: {cooldown_status}")
        
        # SHARED TASK TRACKING: Show task distribution across instances
        if self.config.get('enable_task_tracking', True) and not self.config.get('disable_task_tracking', False):
            task_stats = self.db.get_task_processing_stats()
            if task_stats:
                self.logger.info(f"🔄 Shared Task Tracking:")
                self.logger.info(f"  Total tracked tasks: {task_stats.get('total_tasks', 0)}")
                self.logger.info(f"  Active tasks: {task_stats.get('active_tasks', 0)}")
                self.logger.info(f"  Completed tasks: {task_stats.get('completed_tasks', 0)}")
                
                # Show instance distribution
                instance_counts = task_stats.get('instance_counts', {})
                if instance_counts:
                    self.logger.info(f"  Task distribution by instance:")
                    for instance_id, count in instance_counts.items():
                        if instance_id == self.instance_id:
                            self.logger.info(f"    {instance_id[:20]}...: {count} tasks (this instance)")
                        else:
                            self.logger.info(f"    {instance_id[:20]}...: {count} tasks")
                
                # Show validator distribution
                validator_counts = task_stats.get('validator_counts', {})
                if validator_counts:
                    self.logger.info(f"  Active tasks by validator:")
                    for uid, count in sorted(validator_counts.items()):
                        self.logger.info(f"    UID {uid}: {count} active tasks")
        else:
            self.logger.info(f"🔄 Shared Task Tracking: DISABLED")
        
        # Duplicate checking status
        if self.config.get('enable_duplicate_checking', True):
            self.logger.info(f"🔄 Duplicate Checking: ENABLED")
        else:
            self.logger.info(f"🔄 Duplicate Checking: DISABLED")
        
        # Check for unfinished tasks
        unfinished_tasks = self.db.get_unfinished_tasks(6)  # Last 6 hours
        if unfinished_tasks:
            self.logger.warning(f"⚠️ Found {len(unfinished_tasks)} unfinished tasks in last 6 hours:")
            for task in unfinished_tasks[-5:]:  # Show last 5
                status = "not_processed" if task.processed_at is None else ("no_submission" if not task.submission_success else "no_feedback")
                self.logger.warning(f"   UID {task.validator_uid}: '{task.prompt[:30]}...' - {status}")
        
        self.logger.info("="*60)
    
    def get_fidelity_tracking_details(self, validator_uid: int = None) -> Dict[str, Any]:
        """
        Get detailed fidelity tracking information for debugging and monitoring.
        
        Args:
            validator_uid: Specific validator UID, or None for global details
            
        Returns:
            Detailed tracking information
        """
        if not hasattr(self, 'fidelity_tracker'):
            return {"error": "Fidelity tracker not initialized"}
        
        if validator_uid:
            # Get validator-specific details
            validator_summary = self.fidelity_tracker.get_tracking_summary(validator_uid)
            if not validator_summary:
                return {"error": f"Validator {validator_uid} not found in tracking"}
            
            return {
                "type": "validator_details",
                "validator_uid": validator_uid,
                "summary": validator_summary,
                "recommendations": {
                    "should_switch": validator_summary.get('consecutive_zeros', 0) >= self.config.get('fidelity_tracker_zero_threshold', 2),
                    "recommended_endpoint": self.fidelity_tracker.get_recommended_endpoint(validator_uid, validator_summary.get('original_endpoint', '/generate/')),
                    "reason": f"Consecutive zeros: {validator_summary.get('consecutive_zeros', 0)} (threshold: {self.config.get('fidelity_tracker_zero_threshold', 2)})"
                }
            }
        else:
            # Get global details
            global_summary = self.fidelity_tracker.get_tracking_summary()
            return {
                "type": "global_details",
                "summary": global_summary,
                "configuration": {
                    "history_size": self.config.get('fidelity_tracker_history_size', 5),
                    "zero_threshold": self.config.get('fidelity_tracker_zero_threshold', 2),
                    "fallback_endpoint": self.config.get('fidelity_tracker_fallback_endpoint', "/generate_3d_from_prompt_grid_flow/"),
                    "log_fidelity_tracking": self.config.get('log_fidelity_tracking', True)
                }
            }
    
    def reset_fidelity_tracking(self, validator_uid: int = None):
        """
        Reset fidelity tracking for a specific validator or globally.
        
        Args:
            validator_uid: Specific validator UID, or None to reset all tracking
        """
        if not hasattr(self, 'fidelity_tracker'):
            self.logger.warning("❌ Fidelity tracker not initialized")
            return
        
        if validator_uid:
            self.fidelity_tracker.reset_validator_tracking(validator_uid)
            self.logger.info(f"🔄 Reset fidelity tracking for validator {validator_uid}")
        else:
            # Reset all tracking
            for uid in list(self.fidelity_tracker.validator_histories.keys()):
                self.fidelity_tracker.reset_validator_tracking(uid)
            self.logger.info("🔄 Reset all fidelity tracking")
    
    def cleanup_fidelity_tracking(self, max_age_hours: float = 24):
        """
        Clean up old fidelity tracking history.
        
        Args:
            max_age_hours: Maximum age in hours before cleanup
        """
        if not hasattr(self, 'fidelity_tracker'):
            self.logger.warning("❌ Fidelity tracker not initialized")
            return
        
        self.fidelity_tracker.cleanup_old_history(max_age_hours)
        self.logger.info(f"🧹 Cleaned up fidelity tracking history older than {max_age_hours:.1f} hours")
    
    async def continuous_mining_loop(self):
        """Main continuous mining loop"""
        self.logger.info("🚀 Starting continuous TRELLIS mining...")
        
        # 🚨 CRITICAL: Check for existing violations before starting
        self._check_existing_critical_violations()
        
        # Setup Bittensor
        if not self._setup_bittensor():
            self.logger.error("❌ Failed to setup Bittensor")
            return
        
        # Initial validator refresh
        self.refresh_validators()
        
        if not self.validators:
            self.logger.error("❌ No active validators found")
            return
        
        self.running = True
        self.start_time = time.time()
        
        # Initialize timing
        last_stats_report = 0
        last_cleanup = 0
        last_idle_validation = 0
        last_validator_refresh = 0

        # Explain cooldown system at startup
        self.logger.info("💡 COOLDOWN SYSTEM EXPLANATION:")
        self.logger.info("   • LOCAL COOLDOWN: Prevents pulling tasks from validator (longer, penalty-based)")
        self.logger.info("   • RAPID SUBMISSION: Checks if processing would be too soon (shorter, timing-based)")
        self.logger.info("   • These are DIFFERENT mechanisms serving different purposes!")
        
        try:
            while self.running:
                current_time = time.time()
                
                # Periodic validator refresh (every 10 minutes to catch changes)
                if current_time - last_validator_refresh > 600:
                    self.refresh_validators()
                    last_validator_refresh = current_time
                
                # 🚨 CRITICAL: Periodic violation monitoring (every 5 minutes)
                if current_time - last_cleanup > 300:  # Check every 5 minutes
                    self._check_runtime_critical_violations()
                
                # Periodic cleanup
                if current_time - last_cleanup > self.config['cleanup_interval']:
                    self.db.cleanup_old_prompts()
                    
                    # SHARED TASK TRACKING: Clean up expired task locks
                    if self.config.get('enable_task_tracking', True) and not self.config.get('disable_task_tracking', False):
                        self.db.cleanup_expired_locks(timeout_minutes=2)
                    
                    # Check and clear expired emergency blacklists
                    self._check_and_clear_expired_emergency_blacklists()
                    
                    # Check for validators that need extended monitoring
                    self._check_validators_needing_monitoring()
                    
                    # PERIODIC: Save validator states to disk
                    self.save_validator_states_to_disk()
                    
                    last_cleanup = current_time
                
                # SHARED TASK TRACKING: Get available validators (not busy with other instances)
                if self.config.get('enable_task_tracking', True) and not self.config.get('disable_task_tracking', False):
                    available_validators = self.db.get_available_validators(exclude_instance_id=self.instance_id)
                    if available_validators:
                        self.logger.debug(f"📊 Available validators: {available_validators}")
                
                # DEBUG: Periodic cooldown status reporting (every 30 seconds)
                if current_time - last_stats_report > 30:
                    # Report cooldown status for all validators with timing information
                    for uid, validator in self.validators.items():
                        if validator.last_submit_time:
                            cooldown_report = self.get_detailed_cooldown_report(validator)

                            # Show both cooldown types for clarity
                            # local_cooldown_remaining = None
                            # for cooldown_name, cooldown_info in cooldown_report["cooldowns"].items():
                            #     if cooldown_info["remaining_seconds"] > 0:
                            #         local_cooldown_remaining = cooldown_info["remaining_seconds"]
                            #         break

                            # rapid_submission_time = cooldown_report["rapid_submission_check"]["time_until_ok"]

                            # if local_cooldown_remaining:
                            #     self.logger.debug(f"🔍 COOLDOWN DEBUG UID {uid}: Local={local_cooldown_remaining:.1f}s, Rapid={rapid_submission_time:.1f}s")
                            #     # Only show explanation occasionally to avoid spam
                            #     if uid == 212 and int(current_time) % 300 < 30:  # Every 5 minutes for UID 212 as example
                            #         self.logger.info(f"💡 COOLDOWN EXPLANATION: Local cooldown prevents task PULLING, Rapid checks processing TIMING")
                            # elif rapid_submission_time > 0:
                            #     self.logger.debug(f"🔍 COOLDOWN DEBUG UID {uid}: Rapid submission would trigger in {rapid_submission_time:.1f}s")
                            # else:
                            #     self.logger.debug(f"🔍 COOLDOWN DEBUG UID {uid}: Available for processing")
                            if cooldown_report["is_available"]:
                                self.logger.debug(f"�� COOLDOWN DEBUG UID {uid}: AVAILABLE")
                            else:
                                # Show why validator is unavailable
                                reason = "Unknown"
                                # DEPRECATED: cooldown_until field - now using validator_enforced_cooldown_until and miner_cooldown_until
                                # if validator.cooldown_until:
                                #     reason = f"Local cooldown: {cooldown_report['cooldowns'].get('Miner Local', {}).get('remaining_seconds', 0):.1f}s"
                                if validator.miner_cooldown_until:
                                    reason = f"Local cooldown: {cooldown_report['cooldowns'].get('Miner Local', {}).get('remaining_seconds', 0):.1f}s"
                                elif validator.emergency_blacklist_until:
                                    reason = f"Emergency blacklist: {validator.emergency_blacklist_until - time.time():.1f}s"
                                
                                self.logger.debug(f"�� COOLDOWN DEBUG UID {uid}: UNAVAILABLE - {reason}")

                # Pull tasks from all available validators
                new_task_found = False

                for validator in self.validators.values():
                    if not self.running:
                        break
                
                    ## preliminary check to see if the validator is available
                    cooldown_status = self._check_validator_cooldown_state(validator)
                    if not cooldown_status['available']:
                        self.logger.debug(f"⏳ Validator UID {validator.uid} not available: {cooldown_status['reason']}")
                        # Show recommendation in a clean, single-line format
                        print(f"\r⏳ Validator {validator.uid}: {cooldown_status['recommendation']}", end='', flush=True)
                        continue
                    
                    # SHARED TASK TRACKING: Skip validators that are busy with other instances
                    if self.config.get('enable_task_tracking', True) and not self.config.get('disable_task_tracking', False):
                        if self.db.is_validator_busy(validator.uid, exclude_instance_id=self.instance_id):
                            self.logger.debug(f"⏳ Validator UID {validator.uid} busy with other instance - skipping")
                            continue
                    
                    self.logger.debug(f"📡 Attempting to pull task from UID {validator.uid}")
                    task = await self.pull_task_from_validator(validator)
                    if task:
                        new_task_found = True
                        # Process task immediately
                        await self.process_task(task, validator)
                
                # If no new tasks, do idle validation
                # if not new_task_found and current_time - last_idle_validation > self.config['idle_validation_interval']:
                #     await self.idle_validation_cycle()
                #     last_idle_validation = current_time
                
                # Periodic statistics report
                if current_time - last_stats_report > self.config['stats_report_interval']:
                    self.print_status()
                    self.save_statistics()
                    last_stats_report = current_time
                
                # Periodic gold prompts reload
                if (REPRODUCIBILITY_SYSTEM_AVAILABLE and 
                    self.reproducibility_system and 
                    current_time - self.last_gold_prompts_reload > self.gold_prompts_reload_interval):
                    
                    if self.config.get('activate_learning', False):
                        # Use enhanced reload with real-time learning
                        self.enhanced_reload_gold_prompts()
                    else:
                        # Use standard reload
                        self.reload_gold_prompts()
                
                # Wait before next cycle
                await asyncio.sleep(2)  # Short sleep between cycles
        
        except KeyboardInterrupt:
            self.logger.info("🛑 Mining interrupted by user")
        except Exception as e:
            self.logger.error(f"❌ Mining loop error: {e}")
            traceback.print_exc()
        finally:
            self.running = False
            
            # Stop live monitoring if enabled
            if self.config.get('activate_learning', False):
                self.stop_live_monitoring()
            
            self.print_status()
            self.save_statistics()
            self.logger.info("🏁 Continuous mining stopped")
    
    def _on_priority_interruption(self):
        """Callback when priority interruption occurs"""
        self.stats['priority_interruptions'] = self.stats.get('priority_interruptions', 0) + 1
        self.logger.info(f"📊 Priority interruption tracked: {self.stats['priority_interruptions']} total")
    
    def reload_gold_prompts(self):
        """Reload gold prompts from episodic memory to get fresh data"""
        if not REPRODUCIBILITY_SYSTEM_AVAILABLE or not self.reproducibility_system:
            return
        
        try:
            self.logger.info("📚 Reloading gold prompts from episodic memory...")
            
            # Reload the episodic memory
            old_count = len(self.reproducibility_system.gold_standard_results)
            self.reproducibility_system.gold_standard_results = self.reproducibility_system._load_episodic_memory()
            new_count = len(self.reproducibility_system.gold_standard_results)
            
            # Update timestamp and statistics
            self.last_gold_prompts_reload = time.time()
            self.stats['gold_prompts_reloaded'] += 1
            self.stats['gold_prompts_available'] = new_count
            
            # Log the results
            if new_count > old_count:
                self.logger.info(f"✅ Gold prompts updated: {old_count} → {new_count} (+{new_count - old_count})")
            elif new_count < old_count:
                self.logger.info(f"⚠️ Gold prompts updated: {old_count} → {new_count} (-{old_count - new_count})")
            else:
                self.logger.info(f"🔄 Gold prompts reloaded: {new_count} prompts (no change in count)")
            
            # Log some sample prompts for verification
            if new_count > 0:
                sample_prompts = list(self.reproducibility_system.gold_standard_results.keys())[:3]
                self.logger.info(f"   📝 Sample gold prompts:")
                for i, prompt in enumerate(sample_prompts, 1):
                    self.logger.info(f"     {i}. '{prompt[:60]}...'")
            
        except Exception as e:
            self.logger.error(f"❌ Failed to reload gold prompts: {e}")
            traceback.print_exc()

    def clean_optimized_prompt_wbgmsst(self, prompt: str) -> str:
        """
        Clean up common artifacts and formatting issues from optimized prompts.
        Removes common prefixes, suffixes, and formatting artifacts that shouldn't be sent to generation.
        
        Args:
            prompt: The raw optimized prompt that may contain artifacts
            
        Returns:
            Cleaned prompt ready for generation
        """
        # Check if prompt cleaning is enabled
        if not self.config.get('enable_prompt_cleaning', True):
            return prompt
            
        if not prompt:
            return prompt
        
        # Common artifacts to remove (case insensitive)
        artifacts_to_remove = [
            "wbgmsst", "wbgmsst,", "wbgmsst, ",  # Common prefix artifact
            "wbgsst", "wbgsst,", "wbgsst, ",     # Variant of above
            "wbgms", "wbgms,", "wbgms, ",        # Shorter variant
            "wbgs", "wbgs,", "wbgs, ",           # Even shorter variant
        ]
        
        cleaned_prompt = prompt
        
        # Remove artifacts from the beginning of the prompt
        for artifact in artifacts_to_remove:
            if cleaned_prompt.lower().startswith(artifact.lower()):
                cleaned_prompt = cleaned_prompt[len(artifact):].lstrip()
                self.logger.debug(f"🧹 Removed artifact '{artifact}' from prompt start")
                break
        
        # Remove artifacts that might appear elsewhere (with context)
        for artifact in artifacts_to_remove:
            # Remove standalone artifacts (with proper word boundaries)
            import re
            pattern = r'\b' + re.escape(artifact) + r'\b'
            if re.search(pattern, cleaned_prompt, re.IGNORECASE):
                cleaned_prompt = re.sub(pattern, '', cleaned_prompt, flags=re.IGNORECASE)
                self.logger.debug(f"🧹 Removed artifact '{artifact}' from prompt body")
        
        # Clean up extra whitespace and punctuation
        cleaned_prompt = cleaned_prompt.strip()
        
        # Remove leading commas and extra punctuation
        while cleaned_prompt.startswith(','):
            cleaned_prompt = cleaned_prompt[1:].lstrip()
        
        # Log if cleaning was performed
        if cleaned_prompt != prompt:
            self.logger.info(f"🧹 Prompt cleaned:")
            self.logger.info(f"   Before: '{prompt}'")
            self.logger.info(f"   After:  '{cleaned_prompt}'")
            # Track cleaning statistics
            self.stats['prompts_cleaned'] = self.stats.get('prompts_cleaned', 0) + 1
        else:
            self.logger.debug(f"🧹 No cleaning needed for prompt: '{prompt[:50]}...'")
        
        return cleaned_prompt

    # ===== REAL-TIME LEARNING INTEGRATION FUNCTIONS =====
    
    def parse_current_episode_logs(self) -> Dict[str, Any]:
        """
        Parse current episode logs to get real-time learning improvements.
        This extracts gold prompts and optimization results from the most recent logs.
        
        Returns:
            Dictionary of current gold prompts with their optimization data
        """
        current_gold_prompts = {}
        
        try:
            # Find the most recent episode log
            log_dir = Path("episodic_logs_first")
            if not log_dir.exists():
                self.logger.debug("📁 Log directory not found, skipping log parsing")
                return current_gold_prompts
            
            recent_logs = sorted(log_dir.glob("episodic_run_*.log"), key=lambda x: x.stat().st_mtime)
            
            if not recent_logs:
                self.logger.debug("📁 No episode logs found, skipping log parsing")
                return current_gold_prompts
            
            # Determine how many logs to parse based on configuration
            if self.config.get('only_log_learning', False):
                # Use log_learning_count for only-log-learning mode
                log_count = self.config.get('log_learning_count', 6)
                if log_count == -1:
                    latest_logs = recent_logs  # Use all logs
                    self.logger.debug(f"📖 ONLY-LOG-LEARNING: Parsing all {len(recent_logs)} available logs")
                else:
                    latest_logs = recent_logs[-log_count:]  # Use most recent N logs
                    self.logger.debug(f"📖 ONLY-LOG-LEARNING: Parsing most recent {len(latest_logs)} logs (limited by --only-log-learning={log_count})")
            else:
                # Use max_logs_to_parse for standard mode
                max_logs_to_parse = self.config.get('max_logs_to_parse', 10)
                if isinstance(max_logs_to_parse, int) and max_logs_to_parse == -1:
                    latest_logs = recent_logs
                    self.logger.debug(f"📖 Standard mode: Parsing all {len(recent_logs)} available logs")
                else:
                    latest_logs = recent_logs[-int(max_logs_to_parse):]
                    self.logger.debug(f"📖 Standard mode: Parsing most recent {len(latest_logs)} logs (limited by max_logs_to_parse={max_logs_to_parse})")
            
            for log_file in latest_logs:
                self.logger.debug(f"📖 Parsing log file: {log_file.name}")
                
                try:
                    with open(log_file, 'r') as f:
                        content = f.read()
                        
                    # Extract optimization results from the log
                    log_prompts = self._extract_optimization_results_from_log(content)
                    
                    # Merge with current results (newer logs take precedence)
                    for prompt, data in log_prompts.items():
                        if prompt not in current_gold_prompts or data.get('timestamp', 0) > current_gold_prompts[prompt].get('timestamp', 0):
                            current_gold_prompts[prompt] = data
                            
                except Exception as e:
                    self.logger.warning(f"⚠️ Failed to parse log {log_file.name}: {e}")
                    continue
            
            if current_gold_prompts:
                log_source = "ONLY-LOG-LEARNING" if self.config.get('only_log_learning', False) else "standard mode"
                self.logger.info(f"📚 Parsed {len(current_gold_prompts)} prompts from {len(latest_logs)} logs ({log_source})")
                self.stats['log_parsed_prompts'] = len(current_gold_prompts)
            else:
                self.logger.debug("📚 No prompts found in recent logs")
                
        except Exception as e:
            self.logger.error(f"❌ Error parsing episode logs: {e}")
            
        return current_gold_prompts
    
    def _extract_optimization_results_from_log(self, log_content: str) -> Dict[str, Any]:
        """
        Extract optimization results from a single log file.
        This captures ALL prompts being optimized with their scores and optimized versions.
        
        Args:
            log_content: Content of the log file
            
        Returns:
            Dictionary of prompts with their optimization data
        """
        extracted_prompts = {}
        
        try:
            import re
            
            # Split content into lines for better parsing
            lines = log_content.split('\n')
            
            # Process each line to find optimization data
            current_prompt = None
            current_score = None
            current_optimized = None
            current_round = 0
            
            def _normalize_prompt_text(text: str) -> str:
                if not isinstance(text, str):
                    return text
                s = text.strip()
                # Remove paired double/single quotes around the entire string
                if (s.startswith("''") and s.endswith("''")) or (s.startswith('""') and s.endswith('""')):
                    s = s[2:-2].strip()
                if (s.startswith("'") and s.endswith("'")) or (s.startswith('"') and s.endswith('"')):
                    s = s[1:-1].strip()
                return s
            
            for i, line in enumerate(lines):
                # Find "Original:" lines to get the prompt
                if 'Original:' in line:
                    current_prompt = _normalize_prompt_text(line.split('Original:')[1].strip())
                    current_score = None
                    current_optimized = None
                    current_round = 0
                
                # Find "Optimized:" lines to get the optimized version (old format)
                elif 'Optimized:' in line and 'wbgmsst,' in line:
                    if current_prompt:
                        # Extract the optimized prompt (remove "wbgmsst," prefix)
                        optimized_text = line.split('Optimized:')[1].strip()
                        if optimized_text.startswith('wbgmsst,'):
                            current_optimized = optimized_text[8:].strip()  # Remove "wbgmsst," prefix
                        else:
                            current_optimized = optimized_text

                elif 'Optimized:' in line:
                    if current_prompt:
                        # Extract the optimized prompt (remove "wbgmsst," prefix)
                        optimized_text = line.split('Optimized:')[1].strip()
                        current_optimized = optimized_text
                
                # Find "Using optimized prompt for generation:" lines (new format from logs)
                elif '📝 Using optimized prompt for generation:' in line:
                    if current_prompt:
                        # Extract the optimized prompt from quotes
                        optimized_match = re.search(r"'([^']+)'", line)
                        if optimized_match:
                            current_optimized = optimized_match.group(1).strip()
                            # Clean up common artifacts
                            if current_optimized.endswith('...'):
                                current_optimized = current_optimized[:-3].strip()
                            if current_optimized.endswith('front view, white background'):
                                current_optimized = current_optimized[:-28].strip()
                            if current_optimized.endswith('white background'):
                                current_optimized = current_optimized[:-16].strip()
                
                # Find "Validation score:" lines to get the score
                elif '📊 Validation score:' in line:
                    score_match = re.search(r'📊 Validation score: ([\d.]+)', line)
                    if score_match and current_prompt:
                        current_score = float(score_match.group(1))
                        
                        # Create or update prompt data
                        if current_prompt not in extracted_prompts:
                            extracted_prompts[current_prompt] = {
                                'original_prompt': current_prompt,
                                'optimized_prompt': current_optimized or current_prompt,
                                'best_score': current_score,
                                'current_round': current_round,
                                'is_gold': current_score > 0.75,
                                'source': 'log_parsing',
                                'method': 'comprehensive_extraction',
                                'status': 'completed' if current_score > 0 else 'optimizing'
                            }
                        else:
                            # Update with better score if found
                            existing_data = extracted_prompts[current_prompt]
                            if current_score > existing_data.get('best_score', 0.0):
                                existing_data['best_score'] = current_score
                                existing_data['is_gold'] = current_score > 0.75
                                if current_optimized:
                                    existing_data['optimized_prompt'] = current_optimized
                
                # Find round information: "🔄 RL Round X/20"
                elif '🔄 RL Round' in line:
                    round_match = re.search(r'🔄 RL Round (\d+)/20', line)
                    if round_match and current_prompt:
                        current_round = int(round_match.group(1))
                        if current_prompt in extracted_prompts:
                            extracted_prompts[current_prompt]['current_round'] = current_round
                
                # Find episode and prompt numbers
                elif '--- Episode' in line and 'Prompt' in line:
                    episode_match = re.search(r'--- Episode (\d+), Prompt (\d+) \(Total: (\d+)\) ---', line)
                    if episode_match and current_prompt:
                        episode, prompt_num, total = episode_match.groups()
                        if current_prompt in extracted_prompts:
                            extracted_prompts[current_prompt].update({
                                'episode': int(episode),
                                'prompt_number': int(prompt_num),
                                'total_prompts': int(total)
                            })
                
                # Find timestamps
                elif re.match(r'\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}', line):
                    timestamp_match = re.search(r'(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})', line)
                    if timestamp_match and current_prompt:
                        timestamp = timestamp_match.group(1)
                        if current_prompt in extracted_prompts:
                            extracted_prompts[current_prompt]['timestamp'] = timestamp
            
            # Log summary
            gold_count = sum(1 for p in extracted_prompts.values() if p.get('is_gold', False))
            total_count = len(extracted_prompts)
            
            if total_count > 0:
                self.logger.debug(f"📊 Log parsing summary: {total_count} prompts, {gold_count} gold prompts")
                
        except Exception as e:
            self.logger.error(f"❌ Error extracting optimization results: {e}")
            
        return extracted_prompts
    
    def get_fresh_gold_prompts(self) -> Dict[str, Any]:
        """
        Get gold prompts from both episodic memory and current logs for comprehensive coverage.
        This combines stable episodic memory data with real-time log data, prioritizing highest scores.
        
        Returns:
            Combined dictionary of gold prompts with real-time updates
        """
        combined_prompts = {}
        
        try:
            # Check if we should bypass episodic memory (only-log-learning mode)
            if self.config.get('only_log_learning', False):
                self.logger.info("📖 ONLY-LOG-LEARNING mode: Bypassing episodic memory, using only log data")
                memory_prompts = {}
                self.logger.debug("📚 Skipping episodic memory due to --only-log-learning flag")
            else:
                # Get from episodic memory (stable, complete)
                if self.reproducibility_system:
                    memory_prompts = self.reproducibility_system.gold_standard_results
                    self.logger.debug(f"📚 Loaded {len(memory_prompts)} prompts from episodic memory")
                    
                    # Convert episodic memory format to our standard format
                    for prompt, data in memory_prompts.items():
                        if 'method_2_hybrid_example' in data:
                            # Extract the optimized prompt and score
                            method_data = data['method_2_hybrid_example']
                            optimized_prompt = method_data.get('optimized_prompt', prompt)
                            score = method_data.get('validation_results', {}).get('validation_engine_score', 0.0)
                            
                            combined_prompts[prompt] = {
                                'original_prompt': prompt,
                                'optimized_prompt': optimized_prompt,
                                'best_score': score,
                                'source': 'episodic_memory',
                                'method': 'episodic_memory',
                                'status': 'completed',
                                'is_gold': score > 0.75
                            }
                        else:
                            # Fallback for other formats
                            combined_prompts[prompt] = {
                                'original_prompt': prompt,
                                'optimized_prompt': prompt,
                                'best_score': 0.0,
                                'source': 'episodic_memory',
                                'method': 'episodic_memory',
                                'status': 'unknown',
                                'is_gold': False
                            }
                else:
                    self.logger.debug("📚 Reproducibility system not available, skipping episodic memory")
                    memory_prompts = {}
            
            # Get from current logs (real-time, partial)
            if self.config.get('activate_learning', False):
                log_prompts = self.parse_current_episode_logs()
                self.logger.debug(f"📖 Loaded {len(log_prompts)} prompts from recent logs")
                
                # Sort log prompts by score (highest first) to prioritize best ones
                sorted_log_prompts = sorted(
                    log_prompts.items(), 
                    key=lambda x: x[1].get('best_score', 0.0), 
                    reverse=True
                )
                
                # In only-log-learning mode, all prompts come from logs
                if self.config.get('only_log_learning', False):
                    self.logger.info(f"📖 ONLY-LOG-LEARNING: Using {len(log_prompts)} prompts exclusively from logs")
                    for prompt, data in sorted_log_prompts:
                        combined_prompts[prompt] = data
                else:
                    # Merge them intelligently (logs take precedence for duplicates and high scores)
                    for prompt, data in sorted_log_prompts:
                        if prompt in combined_prompts:
                            # Check if log data has better score
                            existing_data = combined_prompts[prompt]
                            existing_score = existing_data.get('best_score', 0.0)
                            log_score = data.get('best_score', 0.0)
                            
                            # Prefer log data if it has higher score
                            if log_score > existing_score:
                                self.logger.debug(f"🔄 Updating prompt '{prompt[:30]}...' with better log data (score: {existing_score:.4f} → {log_score:.4f})")
                                # Merge data intelligently, keeping best of both
                                merged_data = existing_data.copy()
                                merged_data.update(data)
                                # Ensure we keep the best score
                                merged_data['best_score'] = log_score
                                combined_prompts[prompt] = merged_data
                            elif log_score == existing_score and data.get('timestamp') and existing_data.get('timestamp'):
                                # If scores are equal, prefer newer data and merge
                                if data['timestamp'] > existing_data['timestamp']:
                                    self.logger.debug(f"🔄 Updating prompt '{prompt[:30]}...' with newer log data")
                                    merged_data = existing_data.copy()
                                    merged_data.update(data)
                                    combined_prompts[prompt] = merged_data
                            elif log_score == existing_score:
                                # If scores are equal but no timestamp, merge to get complete data
                                self.logger.debug(f"🔄 Merging data for prompt '{prompt[:30]}...' with equal scores")
                                merged_data = existing_data.copy()
                                merged_data.update(data)
                                combined_prompts[prompt] = merged_data
                        else:
                            # New prompt from logs - add it
                            combined_prompts[prompt] = data
                        
                # Log the merge results
                memory_count = len(memory_prompts) if 'memory_prompts' in locals() else 0
                log_count = len(log_prompts)
                total_count = len(combined_prompts)
                
                if self.config.get('only_log_learning', False):
                    self.logger.info(f"📖 ONLY-LOG-LEARNING results:")
                    self.logger.info(f"   📖 From recent logs: {log_count}")
                    self.logger.info(f"   🔄 Total available: {total_count}")
                    self.logger.info(f"   📚 Episodic memory: BYPASSED")
                else:
                    self.logger.info(f"🔄 Enhanced merge results:")
                    self.logger.info(f"   📚 From episodic memory: {memory_count}")
                    self.logger.info(f"   📖 From recent logs: {log_count}")
                    self.logger.info(f"   🔄 Total combined: {total_count}")
                    
                    # Verify we didn't lose any prompts (only when not in only-log-learning mode)
                    if total_count < memory_count:
                        self.logger.warning(f"⚠️ WARNING: Lost {memory_count - total_count} prompts during merge!")
                        self.logger.warning(f"   Expected: {memory_count + log_count}, Got: {total_count}")
                        
                        # Debug: show what we have
                        memory_prompts_set = set(memory_prompts.keys())
                        log_prompts_set = set(log_prompts.keys())
                        combined_prompts_set = set(combined_prompts.keys())
                        
                        self.logger.debug(f"   Memory prompts: {len(memory_prompts_set)}")
                        self.logger.debug(f"   Log prompts: {len(log_prompts_set)}")
                        self.logger.debug(f"   Combined prompts: {len(combined_prompts_set)}")
                        
                        # Show what's missing
                        missing_from_memory = memory_prompts_set - combined_prompts_set
                        if missing_from_memory:
                            self.logger.warning(f"   Missing from memory: {len(missing_from_memory)} prompts")
                            for missing in list(missing_from_memory)[:3]:
                                self.logger.warning(f"     - '{missing[:50]}...'")
                
                # Show top scoring prompts from logs
                if log_prompts:
                    top_log_prompts = sorted(
                        log_prompts.items(), 
                        key=lambda x: x[1].get('best_score', 0.0), 
                        reverse=True
                    )[:5]  # Top 5
                    
                    self.logger.info(f"   🏆 Top scoring prompts from logs:")
                    for i, (prompt, data) in enumerate(top_log_prompts, 1):
                        score = data.get('best_score', 0.0)
                        round_info = f" (round {data.get('current_round', 0)})" if data.get('current_round', 0) > 0 else ""
                        self.logger.info(f"     {i}. Score {score:.4f}{round_info}: '{prompt[:50]}...'")
                
                # Show comprehensive top scoring prompts from combined data
                if combined_prompts:
                    top_combined_prompts = sorted(
                        combined_prompts.items(), 
                        key=lambda x: x[1].get('best_score', 0.0), 
                        reverse=True
                    )[:10]  # Top 10
                    
                    # if self.config.get('only_log_learning', False):
                    #     self.logger.info(f"   🏆 Top 10 scoring prompts (logs only):")
                    # else:
                    #     self.logger.info(f"   🏆 Top 10 scoring prompts (combined data):")
                        
                    # for i, (prompt, data) in enumerate(top_combined_prompts, 1):
                    #     score = data.get('best_score', 0.0)
                    #     source = data.get('source', 'unknown')
                    #     optimized = data.get('optimized_prompt', 'N/A')
                    #     self.logger.info(f"     {i:2d}. Score {score:.4f} ({source}): '{prompt[:50]}...'")
                    #     if optimized != prompt and len(optimized) > 50:
                    #         self.logger.info(f"         Optimized: '{optimized[:80]}...'")
                    #     elif optimized != prompt:
                    #         self.logger.info(f"         Optimized: '{optimized}'")
                
                # Show gold prompt count
                gold_count = sum(1 for p in combined_prompts.values() if p.get('best_score', 0.0) > 0.75)
                if self.config.get('only_log_learning', False):
                    self.logger.info(f"   🏆 Total gold prompts available (logs only): {gold_count}")
                else:
                    self.logger.info(f"   🏆 Total gold prompts available: {gold_count}")
                
                # Show comprehensive scoring summary
                if combined_prompts:
                    scores_list = [p.get('best_score', 0.0) for p in combined_prompts.values()]
                    avg_score = sum(scores_list) / len(scores_list)
                    max_score = max(scores_list)
                    min_score = min(scores_list)
                    
                    self.logger.info(f"   📊 Scoring Summary:")
                    self.logger.info(f"      Average score: {avg_score:.4f}")
                    self.logger.info(f"      Highest score: {max_score:.4f}")
                    self.logger.info(f"      Lowest score: {min_score:.4f}")
                    
                    # Score distribution
                    excellent = len([s for s in scores_list if s >= 0.9])
                    good = len([s for s in scores_list if 0.7 <= s < 0.9])
                    fair = len([s for s in scores_list if 0.5 <= s < 0.7])
                    poor = len([s for s in scores_list if s < 0.5])
                    
                    self.logger.info(f"      Score distribution: {excellent} excellent (≥0.9), {good} good (0.7-0.9), {fair} fair (0.5-0.7), {poor} poor (<0.5)")
                
            else:
                self.logger.debug("📖 Real-time learning disabled, using only episodic memory")
            
            # Update statistics
            self.stats['total_gold_prompts_available'] = len(combined_prompts)
            if self.config.get('only_log_learning', False):
                self.stats['memory_prompts'] = 0  # Bypassed in only-log-learning mode
                self.stats['log_prompts'] = len(log_prompts) if 'log_prompts' in locals() else 0
            else:
                self.stats['memory_prompts'] = len(memory_prompts) if 'memory_prompts' in locals() else 0
                self.stats['log_prompts'] = len(log_prompts) if 'log_prompts' in locals() else 0
            
        except Exception as e:
            self.logger.error(f"❌ Error getting fresh gold prompts: {e}")
            traceback.print_exc()
            # Fallback to just episodic memory
            if self.reproducibility_system:
                combined_prompts = self.reproducibility_system.gold_standard_results
        
        return combined_prompts
    
    def setup_live_episodic_memory_monitoring(self):
        """
        Setup live monitoring of episodic memory file for automatic updates.
        This watches for changes and automatically reloads gold prompts.
        """
        if not self.config.get('activate_learning', False):
            self.logger.debug("📁 Live monitoring disabled (activate_learning=False)")
            return
            
        try:
            # Try to import watchdog for file monitoring
            try:
                from watchdog.observers import Observer
                from watchdog.events import FileSystemEventHandler
                WATCHDOG_AVAILABLE = True
            except ImportError:
                WATCHDOG_AVAILABLE = False
                self.logger.warning("⚠️ watchdog not available - live monitoring disabled")
                return
            
            if not WATCHDOG_AVAILABLE:
                return
            
            class EpisodicMemoryWatcher(FileSystemEventHandler):
                def __init__(self, orchestrator):
                    self.orchestrator = orchestrator
                    self.last_modified = 0
                    self.logger = orchestrator.logger
                
                def on_modified(self, event):
                    if event.src_path.endswith('episodic_memory.json'):
                        current_time = time.time()
                        # Debounce to avoid multiple rapid updates
                        if current_time - self.last_modified > 5:
                            self.last_modified = current_time
                            self.logger.info("🔄 Episodic memory file modified - triggering gold prompts reload!")
                            
                            # Reload gold prompts
                            try:
                                self.orchestrator.reload_gold_prompts()
                                self.logger.info("✅ Gold prompts reloaded from episodic memory")
                            except Exception as e:
                                self.logger.error(f"❌ Failed to reload gold prompts: {e}")
            
            # Setup the file watcher
            self.episodic_memory_observer = Observer()
            self.episodic_memory_watcher = EpisodicMemoryWatcher(self)
            
            # Watch the episodic_logs_first directory
            watch_path = Path("episodic_logs_first")
            if watch_path.exists():
                self.episodic_memory_observer.schedule(
                    self.episodic_memory_watcher, 
                    str(watch_path), 
                    recursive=False
                )
                self.episodic_memory_observer.start()
                self.logger.info("📁 Live episodic memory monitoring ENABLED")
                self.logger.info(f"   Watching directory: {watch_path}")
                self.logger.info("   Gold prompts will auto-reload on memory updates")
            else:
                self.logger.warning("⚠️ Episodic logs directory not found, live monitoring disabled")
                
        except Exception as e:
            self.logger.error(f"❌ Failed to setup live monitoring: {e}")
    
    def stop_live_monitoring(self):
        """Stop the live episodic memory monitoring"""
        if hasattr(self, 'episodic_memory_observer'):
            try:
                self.episodic_memory_observer.stop()
                self.episodic_memory_observer.join()
                self.logger.info("📁 Live episodic memory monitoring stopped")
            except Exception as e:
                self.logger.error(f"❌ Error stopping live monitoring: {e}")
    
    def enhanced_reload_gold_prompts(self):
        """
        Enhanced reload that combines episodic memory with real-time log data.
        This is the main method called when --activate-learning is enabled.
        """
        if not REPRODUCIBILITY_SYSTEM_AVAILABLE or not self.reproducibility_system:
            return
        
        try:
            self.logger.info("🚀 Enhanced gold prompts reload with real-time learning...")
            
            # Get fresh gold prompts (memory + logs)
            fresh_prompts = self.get_fresh_gold_prompts()
            
            # Update the reproducibility system with fresh data
            old_count = len(self.reproducibility_system.gold_standard_results)
            
            # Create a temporary update to the reproducibility system
            # Note: This is a workaround since we can't directly modify the system's data structure
            # In a real implementation, you'd want to modify the reproducibility system to accept updates
            
            # For now, we'll update our local tracking
            self.stats['enhanced_gold_prompts_available'] = len(fresh_prompts)
            self.stats['enhanced_gold_prompts_reloaded'] += 1
            
            # Update timestamp
            self.last_gold_prompts_reload = time.time()
            
            # Log the results
            if len(fresh_prompts) > old_count:
                self.logger.info(f"✅ Enhanced reload: {old_count} → {len(fresh_prompts)} (+{len(fresh_prompts) - old_count})")
            elif len(fresh_prompts) < old_count:
                self.logger.info(f"⚠️ Enhanced reload: {old_count} → {len(fresh_prompts)} (-{old_count - len(fresh_prompts)})")
            else:
                self.logger.info(f"🔄 Enhanced reload: {len(fresh_prompts)} prompts (no change in count)")
            
            # Log source breakdown
            memory_count = self.stats.get('memory_prompts', 0)
            log_count = self.stats.get('log_prompts', 0)
            self.logger.info(f"   📚 From episodic memory: {memory_count}")
            self.logger.info(f"   📖 From recent logs: {log_count}")
            self.logger.info(f"   🔄 Total combined: {len(fresh_prompts)}")
            
            # Log some sample prompts for verification
            if fresh_prompts:
                sample_prompts = list(fresh_prompts.keys())[:3]
                self.logger.info(f"   📝 Sample gold prompts:")
                for i, prompt in enumerate(sample_prompts, 1):
                    prompt_data = fresh_prompts[prompt]
                    source = prompt_data.get('source', 'unknown')
                    score = prompt_data.get('final_score', prompt_data.get('score', 'unknown'))
                    self.logger.info(f"     {i}. '{prompt[:60]}...' (score: {score}, source: {source})")
            
        except Exception as e:
            self.logger.error(f"❌ Enhanced gold prompts reload failed: {e}")
            traceback.print_exc()

    def _enhanced_reproducibility_optimization(self, prompt: str, min_similarity: float, enhanced_gold_prompts: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Enhanced reproducibility optimization using our enhanced gold prompts (memory + logs).
        This method implements the similarity search and optimization logic directly.
        
        Args:
            prompt: The prompt to optimize
            min_similarity: Minimum similarity threshold
            enhanced_gold_prompts: Dictionary of enhanced gold prompts from memory + logs
            
        Returns:
            Optimization result or None if no close match found
        """
        try:
            self.logger.info(f"🔍 Enhanced reproducibility search for: '{prompt}'")
            self.logger.info(f"   Searching through {len(enhanced_gold_prompts)} enhanced gold prompts")
            self.logger.info(f"   Minimum similarity threshold: {min_similarity}")
            
            best_match = None
            best_similarity = 0.0
            candidates = []
            
            # Calculate similarity for all gold prompts
            for gold_prompt, gold_data in enhanced_gold_prompts.items():
                # Calculate similarity using enhanced similarity calculation
                similarity = self._calculate_simple_similarity(prompt, gold_prompt)
                
                # Get the score for this gold prompt
                if 'validation_results' in gold_data and 'validation_engine_score' in gold_data['validation_results']:
                    # Episodic memory format
                    gold_score = gold_data['validation_results']['validation_engine_score']
                    source = 'episodic_memory'
                else:
                    # Log format
                    gold_score = gold_data.get('best_score', 0.0)
                    source = gold_data.get('source', 'recent_logs')
                
                # Store candidate for analysis
                candidates.append({
                    'prompt': gold_prompt,
                    'similarity': similarity,
                    'score': gold_score,
                    'source': source,
                    'data': gold_data
                })
                
                if similarity > best_similarity and similarity >= min_similarity:
                    best_similarity = similarity
                    best_match = {
                        'gold_prompt': gold_prompt,
                        'gold_data': gold_data,
                        'similarity': similarity,
                        'gold_score': gold_score,
                        'source': source
                    }
            
            # Sort all candidates by similarity for comprehensive analysis
            sorted_candidates = sorted(candidates, key=lambda x: x['similarity'], reverse=True)
            
            # Log comprehensive similarity analysis
            self.logger.info(f"🔍 Similarity analysis results:")
            self.logger.info(f"   Total candidates analyzed: {len(sorted_candidates)}")
            self.logger.info(f"   Candidates above threshold ({min_similarity}): {len([c for c in sorted_candidates if c['similarity'] >= min_similarity])}")
            self.logger.info(f"   Candidates below threshold: {len([c for c in sorted_candidates if c['similarity'] < min_similarity])}")
            
            # Show top 10 candidates with their similarities
            self.logger.info(f"   Top 3 similarity candidates:")
            for i, candidate in enumerate(sorted_candidates[:3], 1):
                status = "✅ ABOVE THRESHOLD" if candidate['similarity'] >= min_similarity else "❌ Below threshold"
                self.logger.info(f"     {i:2d}. Sim: {candidate['similarity']:.3f} | Score: {candidate['score']:.4f} | {status}")
                self.logger.info(f"         Source: {candidate['source']} | Prompt: '{candidate['prompt'][:60]}...'")
            
            # Show threshold analysis
            if sorted_candidates:
                max_similarity = sorted_candidates[0]['similarity']
                min_similarity_found = sorted_candidates[-1]['similarity']
                avg_similarity = sum(c['similarity'] for c in sorted_candidates) / len(sorted_candidates)
                
                self.logger.info(f"   Similarity statistics:")
                self.logger.info(f"     Maximum similarity: {max_similarity:.3f}")
                self.logger.info(f"     Minimum similarity: {min_similarity_found:.3f}")
                self.logger.info(f"     Average similarity: {avg_similarity:.3f}")
                self.logger.info(f"     Threshold: {min_similarity:.3f}")
                
                if max_similarity < min_similarity:
                    self.logger.warning(f"   ⚠️ WARNING: No prompt meets the similarity threshold!")
                    self.logger.warning(f"      Consider lowering the threshold from {min_similarity:.3f} to {max_similarity:.3f} or lower")
            
            if best_match:
                self.logger.info(f"🏆 Found close gold prompt (similarity: {best_similarity:.3f})")
                self.logger.info(f"   Gold prompt: '{best_match['gold_prompt'][:50]}...'")
                self.logger.info(f"   Gold score: {best_match['gold_score']:.4f}")
                self.logger.info(f"   Source: {best_match['source']}")
                
                # Extract the optimized prompt from gold data
                if 'validation_results' in best_match['gold_data'] and 'method_2_hybrid_example' in best_match['gold_data']:
                    # Episodic memory format
                    optimized_prompt = best_match['gold_data']['method_2_hybrid_example']['optimized_prompt']
                else:
                    # Log format - use original for now (could be enhanced to extract actual optimized version)
                    optimized_prompt = best_match['gold_prompt']
                
                return {
                    'optimized_prompt': optimized_prompt,
                    'similarity': best_similarity,
                    'gold_score': best_match['gold_score'],
                    'gold_prompt': best_match['gold_prompt'],
                    'source': best_match['source'],
                    'method': 'enhanced_reproducibility'
                }
            else:
                self.logger.warning(f"❌ No gold prompt found with similarity ≥ {min_similarity}")
                
                # Suggest potential threshold adjustments
                if sorted_candidates:
                    top_similarities = [c['similarity'] for c in sorted_candidates[:5]]
                    suggested_threshold = max(top_similarities) - 0.05  # 0.05 below the highest
                    self.logger.info(f"   💡 Suggestion: Try lowering threshold to {suggested_threshold:.3f} to include top candidates")
                    self.logger.info(f"      Top similarities found: {[f'{s:.3f}' for s in top_similarities]}")
                
                return None
                
        except Exception as e:
            self.logger.error(f"❌ Enhanced reproducibility optimization failed: {e}")
            traceback.print_exc()
            return None
    

async def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Continuous TRELLIS Orchestrator")
    parser.add_argument("--no-harvest", action="store_true", help="Disable task harvesting")
    parser.add_argument("--no-validate", action="store_true", help="Disable validation")
    parser.add_argument("--no-submit", action="store_true", help="Disable result submission")
    parser.add_argument("--generation-server", default="http://localhost:8096", help="TRELLIS generation server URL")
    parser.add_argument("--validation-server", default="http://localhost:10006", help="Validation server URL")
    parser.add_argument("--output-dir", default="./continuous_trellis_outputs", help="Output directory")
    parser.add_argument("--min-score", type=float, default=0.3, help="Minimum local validation score")
    
    # Prompt optimization arguments
    parser.add_argument("--no-optimize", action="store_true", help="Disable prompt optimization")
    parser.add_argument("--aggressive-optimize", action="store_true", help="Enable aggressive optimization mode")
    parser.add_argument("--quiet-optimize", action="store_true", help="Reduce optimization logging detail")
    parser.add_argument("--no-prompt-cleaning", action="store_true", help="Disable automatic prompt cleaning (removes artifacts like 'wbgmsst')")
    
    # Ollama configuration
    parser.add_argument("--ollama-url", default="http://localhost:11434", help="URL for the Ollama API server")
    
    # vLLM configuration
    parser.add_argument("--vllm", action="store_true", help="Use vLLM instead of Ollama")
    parser.add_argument("--vllm-url", default="http://localhost:9000", help="URL for the vLLM server")
    parser.add_argument("--vllm-model", default="llama-3-2-3b-it", help="vLLM model name")
    
    # NEW: vLLM optimization arguments
    parser.add_argument("--vllm-optim", action="store_true", help="Use vLLM for prompt optimization (bypasses local optimizer)")
    parser.add_argument("--vllm-optim-port", type=int, default=11300, help="vLLM port for prompt optimization (default: 11300)")
    parser.add_argument("--system-prompt", action="store_true", help="Use system prompts during inference (activates trained behavior)")
    parser.add_argument("--vllm-priority", type=str, default="system_chat", choices=["system_chat", "system_completions", "no_system"], help="vLLM optimization priority method (default: system_chat)")
    
    # Reproducibility optimization arguments
    parser.add_argument("--no-reproducibility", action="store_true", help="Disable reproducibility optimization")
    parser.add_argument("--reproducibility-similarity", type=float, default=0.6, help="Minimum similarity threshold for reproducibility (default: 0.3)")
    
    # LoRA routing arguments
    parser.add_argument("--no-lora-routing", action="store_true", help="Disable intelligent LoRA routing")
    parser.add_argument("--lora-confidence-threshold", type=float, default=0.5, help="Minimum confidence threshold for LoRA routing (default: 0.5)")
    
    # LoRA selection argument
    parser.add_argument("--lora", type=str, default="cinema", help="Default LoRA to use when router is not available (default: Cinema Style)")
    
    # Determinism arguments
    parser.add_argument("--variable-seeds", action="store_true", help="Use prompt-hash based seeds (default: fixed seed 42)")
    parser.add_argument("--seed", type=int, default=42, help="Fixed seed to use when not using variable seeds")
    
    # Validator blacklisting arguments  
    parser.add_argument("--blacklist", type=int, nargs="*", default=[180, 253], help="Validator UIDs to blacklist (default: [180])")
    parser.add_argument("--no-blacklist", action="store_true", help="Disable validator blacklisting")
    
    # Gold prompts reload arguments
    parser.add_argument("--gold-prompts-reload-interval", type=int, default=120, help="Reload gold prompts every N seconds (default: 3600 = 1 hour)")
    
    # Real-time learning arguments
    parser.add_argument("--activate-learning", action="store_true", help="Enable real-time learning from episode logs and live episodic memory monitoring")
    parser.add_argument("--only-log-learning", nargs='?', const=6, type=int, metavar='N', 
                       help="Use only N most recent logs for learning, bypass episodic memory (default: 6, use -1 for all logs, requires --activate-learning)")
    
    # Shared task tracking arguments
    parser.add_argument("--enable-task-tracking", action="store_true", help="Enable shared task tracking to prevent duplicate processing across instances")
    parser.add_argument("--disable-task-tracking", action="store_true", help="Disable shared task tracking (default: enabled)")
    
    # Duplicate checking arguments
    parser.add_argument("--no-skip-duplicates", action="store_true", help="Disable duplicate prompt checking (default: enabled)")
    
    # Cooldown arguments
    parser.add_argument("--network-error-cooldown", type=int, default=30, help="Cooldown duration after network errors (seconds, default: 30)")
    parser.add_argument("--submission-failure-cooldown", type=int, default=60, help="Cooldown duration after submission failures (seconds, default: 60)")
    parser.add_argument("--validator-error-cooldown", type=int, default=45, help="Cooldown duration after validator errors (seconds, default: 45)")
    parser.add_argument("--max-cooldown-duration", type=int, default=301, help="Maximum cooldown duration (seconds, default: 300)")
    parser.add_argument("--synthetic-traffic-cooldown", type=int, default=301, help="Cooldown for synthetic traffic (seconds, default: 300)")
    parser.add_argument("--organic-traffic-cooldown", type=int, default=121, help="Cooldown for organic traffic (seconds, default: 120)")
    parser.add_argument("--no-cooldown-logging", action="store_true", help="Disable detailed cooldown logging")
    
    # Enhanced cooldown system arguments
    parser.add_argument("--cooldown-violation-threshold", type=int, default=5, help="Number of violations before applying penalty (default: 5)")
    parser.add_argument("--cooldown-violation-penalty", type=int, default=60, help="Additional penalty cooldown in seconds (default: 60)")
    # DEPRECATED: Validation lock duration argument removed - now using MIN_TASK_INTERVAL constant
    # parser.add_argument("--validation-lock-duration", type=int, default=31, help="Default validation lock duration in seconds (default: 30)")

    # Validator-compliant generation arguments (for miner behavior as orchestrator)
    parser.add_argument("--generation-throttle-period", type=int, default=34, help="Minimum throttle period for task completion (seconds, default: 30)")
    parser.add_argument("--generation-task-cooldown", type=int, default=305, help="Cooldown between tasks from same validator (seconds, default: 300)")
    parser.add_argument("--generation-cooldown-violation-penalty", type=int, default=102, help="Penalty for cooldown violations (seconds, default: 10)")
    parser.add_argument("--generation-cooldown-violations-threshold", type=int, default=100, help="Threshold for malicious behavior (default: 100)")
    parser.add_argument("--generation-cooldown-penalty", type=int, default=600, help="Penalty for low quality submissions (seconds, default: 600)")
    parser.add_argument("--generation-quality-threshold", type=float, default=0.6, help="Minimum score threshold for acceptance (default: 0.6)")
    
    # Emergency cooldown management arguments
    parser.add_argument("--emergency-cooldown-buffer", type=int, default=30, help="Buffer seconds added to validator cooldowns (default: 30)")
    parser.add_argument("--critical-violation-threshold", type=int, default=100, help="Violation count that triggers emergency measures (default: 100)")
    parser.add_argument("--critical-violation-cooldown", type=int, default=3600, help="Emergency cooldown duration for critical violations (default: 3600)")
    parser.add_argument("--base-blacklist-duration", type=int, default=1800, help="Base duration for temporary blacklisting (default: 1800)")
    
    # Fallback mechanism arguments
    parser.add_argument("--no-fallback", action="store_true", help="Disable CLIP-based fallback mechanism for low-fidelity tasks")
    parser.add_argument("--fallback-ratio-threshold", type=float, default=0.8, help="Ratio threshold for triggering fallback (default: 0.8)")
    parser.add_argument("--fallback-max-retries", type=int, default=1, help="Maximum number of prompt re-optimization attempts (default: 1)")
    
    # 🚀 TRELLIS Generation Quality Presets (Based on README_GRID_FLOW_EXPERIMENTS.md)
    parser.add_argument("--fastest-mv-gen", action="store_true", help="Use fastest generation preset: 256×256, minimal steps, short prompts")
    parser.add_argument("--long-fast-mv-gen", action="store_true", help="Use fast but good quality preset: 512×512, balanced steps, detailed prompts")
    parser.add_argument("--production-mv-gen", action="store_true", help="Use production quality preset: 512×512, optimal steps, cinema style")
    parser.add_argument("--quality-mv-gen", action="store_true", help="Use highest quality preset: 1024×1024, maximum steps, 3D style")
    
    args = parser.parse_args()
    
    # Build config
    config = {}
    
    if args.no_harvest:
        config['harvest_tasks'] = False
    if args.no_validate:
        config['validate_generations'] = False
    if args.no_submit:
        config['submit_results'] = False
    
    config['generation_server_url'] = args.generation_server
    config['validation_server_url'] = args.validation_server
    config['output_dir'] = args.output_dir
    config['min_local_score'] = args.min_score
    
    # Prompt optimization configuration
    if args.no_optimize:
        config['enable_prompt_optimization'] = False
    if args.aggressive_optimize:
        config['optimization_aggressive_mode'] = True
    if args.quiet_optimize:
        config['log_optimization_details'] = False
    if args.no_prompt_cleaning:
        config['enable_prompt_cleaning'] = False
    
    # Ollama configuration[]
    config['ollama_url'] = args.ollama_url
    
    # vLLM configuration
    config['use_vllm'] = args.vllm
    config['vllm_url'] = args.vllm_url
    config['vllm_model'] = args.vllm_model
    
    # NEW: vLLM optimization configuration
    config['use_vllm_optim'] = args.vllm_optim
    config['vllm_optim_port'] = args.vllm_optim_port
    config['use_system_prompt'] = args.system_prompt
    config['vllm_optimization_priority'] = args.vllm_priority
    
    # Reproducibility optimization configuration
    if args.no_reproducibility:
        config['enable_reproducibility_optimization'] = False
    config['reproducibility_min_similarity'] = args.reproducibility_similarity
    
    # LoRA routing configuration
    if args.no_lora_routing:
        config['enable_lora_routing'] = False
    config['lora_routing_confidence_threshold'] = args.lora_confidence_threshold
    
    # LoRA selection configuration
    config['default_lora'] = args.lora
    
    # Determinism configuration
    if args.variable_seeds:
        config['use_fixed_seed'] = False
    config['fixed_seed_value'] = args.seed
    
    # Validator blacklisting configuration
    if args.no_blacklist:
        config['enable_validator_blacklisting'] = False
    if args.blacklist is not None:
        config['validator_blacklist'] = args.blacklist
    
    # Gold prompts reload configuration
    config['gold_prompts_reload_interval'] = args.gold_prompts_reload_interval
    
    # Real-time learning configuration
    config['activate_learning'] = args.activate_learning
    config['only_log_learning'] = args.only_log_learning
    
    # Validate only-log-learning requires activate-learning
    if args.only_log_learning is not None and not args.activate_learning:
        print("❌ Error: --only-log-learning requires --activate-learning to be enabled")
        exit(1)
    
    # Set default log learning count if not specified
    if args.only_log_learning is None:
        config['log_learning_count'] = 6  # Default to 6 logs
    else:
        config['log_learning_count'] = args.only_log_learning
    
    # Shared task tracking configuration
    if args.enable_task_tracking:
        config['enable_task_tracking'] = True
        config['disable_task_tracking'] = False
    elif args.disable_task_tracking:
        config['enable_task_tracking'] = False
        config['disable_task_tracking'] = True
    else:
        # Default: enable task tracking
        config['enable_task_tracking'] = True
        config['disable_task_tracking'] = False
    
    # Duplicate checking configuration
    if args.no_skip_duplicates:
        config['enable_duplicate_checking'] = False
        print("⚠️ Duplicate checking DISABLED - will process all prompts including duplicates")
    else:
        config['enable_duplicate_checking'] = True
        print("✅ Duplicate checking ENABLED - will skip previously processed prompts")
    
    # Cooldown configuration
    config['network_error_cooldown'] = args.network_error_cooldown
    config['submission_failure_cooldown'] = args.submission_failure_cooldown
    config['validator_error_cooldown'] = args.validator_error_cooldown
    config['max_cooldown_duration'] = args.max_cooldown_duration
    config['enable_cooldown_logging'] = not args.no_cooldown_logging
    
    # Enhanced cooldown system configuration
    config['cooldown_violation_threshold'] = args.cooldown_violation_threshold
    config['cooldown_violation_penalty'] = args.cooldown_violation_penalty
            # DEPRECATED: Validation lock duration config assignment removed
        # config['validation_lock_duration'] = args.validation_lock_duration

    # Validator-compliant generation settings (for miner behavior as orchestrator)
    config['generation.throttle_period'] = args.generation_throttle_period
    config['generation.task_cooldown'] = args.generation_task_cooldown
    config['generation.cooldown_violation_penalty'] = args.generation_cooldown_violation_penalty
    config['generation.cooldown_violations_threshold'] = args.generation_cooldown_violations_threshold
    config['generation.cooldown_penalty'] = args.generation_cooldown_penalty
    config['generation.quality_threshold'] = args.generation_quality_threshold
    
    # Emergency cooldown management configuration
    config['emergency_cooldown_buffer'] = args.emergency_cooldown_buffer
    config['critical_violation_threshold'] = args.critical_violation_threshold
    config['critical_violation_cooldown'] = args.critical_violation_cooldown
    config['base_blacklist_duration'] = args.base_blacklist_duration
    
    print(f"⏳ Cooldown settings: Network errors: {args.network_error_cooldown}s, Submission failures: {args.submission_failure_cooldown}s, Validator errors: {args.validator_error_cooldown}s, Max: {args.max_cooldown_duration}s")
    print(f"📝 Cooldown logging: {'ENABLED' if not args.no_cooldown_logging else 'DISABLED'}")
    print(f"🚨 Enhanced cooldown: Violation threshold: {args.cooldown_violation_threshold}, Penalty: {args.cooldown_violation_penalty}s")
    print(f"🚨 Validator-compliant generation: Throttle: {args.generation_throttle_period}s, Task cooldown: {args.generation_task_cooldown}s, Quality threshold: {args.generation_quality_threshold}")
    print(f"🚨 Generation penalties: Violation penalty: {args.generation_cooldown_violation_penalty}s, Quality penalty: {args.generation_cooldown_penalty}s, Violation threshold: {args.generation_cooldown_violations_threshold}")
    print(f"🚨 Emergency cooldown: Buffer: {args.emergency_cooldown_buffer}s, Critical threshold: {args.critical_violation_threshold}, Blacklist base: {args.base_blacklist_duration}s")
    
    # Fallback mechanism configuration
    if args.no_fallback:
        config['enable_fallback_mechanism'] = False
        print("⚠️ Fallback mechanism DISABLED - will use standard generation without CLIP-based fallback")
    else:
        config['enable_fallback_mechanism'] = True
        print("✅ Fallback mechanism ENABLED - will use CLIP-based fallback for low-fidelity tasks")
    
    config['fallback_ratio_threshold'] = args.fallback_ratio_threshold
    config['fallback_max_retries'] = args.fallback_max_retries
    print(f"🔄 Fallback settings: Ratio threshold: {args.fallback_ratio_threshold}, Max retries: {args.fallback_max_retries}")
    
    # 🚀 TRELLIS Generation Quality Presets Configuration
    config['fastest_mv_gen'] = args.fastest_mv_gen
    config['long_fast_mv_gen'] = args.long_fast_mv_gen
    config['production_mv_gen'] = args.production_mv_gen
    config['quality_mv_gen'] = args.quality_mv_gen
    
    # Log the selected generation preset
    if args.fastest_mv_gen:
        print("🚀 Using FASTEST generation preset: 256×256, minimal steps, short prompts")
    elif args.long_fast_mv_gen:
        print("⚡ Using FAST + GOOD QUALITY preset: 512×512, balanced steps, detailed prompts")
    elif args.production_mv_gen:
        print("🎯 Using PRODUCTION QUALITY preset: 512×512, optimal steps, cinema style")
    elif args.quality_mv_gen:
        print("🎨 Using HIGHEST QUALITY preset: 1024×1024, maximum steps, 3D style")
    else:
        print("⚙️ Using DEFAULT generation settings (no preset selected)")
    
    # Create and run orchestrator
    orchestrator = ContinuousTrellisOrchestrator(config)
    
    # 🚨 CRITICAL: Check for existing violations before starting
    orchestrator._check_existing_critical_violations()
    
    try:
        await orchestrator.continuous_mining_loop()
    except Exception as e:
        logger.error(f"❌ Orchestrator failed: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(main()) 
