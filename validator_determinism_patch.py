#!/usr/bin/env python3
"""
Validator determinism patches for subnet_accurate_validator2.py
Apply these changes to ensure consistent validation across different GPUs
"""

import os
import torch

def setup_validator_determinism():
    """
    Setup deterministic environment specifically for validation
    """
    # Critical: Set CUDA workspace config before importing validation modules
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    
    # Disable TF32 for validation consistency
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    
    # Force deterministic operations
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    # Set consistent memory allocation
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:False'
    
    print("🔒 Validator determinism mode enabled")

# Patches to apply to your subnet_accurate_validator2.py

VALIDATOR_DETERMINISM_PATCHES = """
# Add this at the very beginning of subnet_accurate_validator2.py, after imports but before any CUDA operations:

# PATCH 1: Add determinism setup at the top of the file
def setup_validator_determinism():
    '''Setup deterministic environment for validation'''
    import os
    import torch
    
    # Critical: Must be set before any CUDA operations
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    
    # Disable TF32 for consistent precision
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    
    # Force deterministic algorithms
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    print("🔒 Validator determinism enabled")

# Call this before any CUDA operations
setup_validator_determinism()

# PATCH 2: Modify load_validator_clip function to use FP32
def load_validator_clip(device):
    '''Load the validator CLIP model with deterministic settings'''
    model, _, _ = open_clip.create_model_and_transforms(
        "convnext_large_d", 
        pretrained="laion2b_s26b_b102k_augreg", 
        device=device
    )
    tokenizer = open_clip.get_tokenizer("convnext_large_d")
    
    # Force FP32 for consistency
    model = model.float()
    model.eval()
    
    # Configure deterministic normalization
    mean = torch.tensor([0.48145466, 0.4578275, 0.40821073], dtype=torch.float32).view(1, 3, 1, 1) * 3
    std = torch.tensor([0.26862954, 0.26130258, 0.27577711], dtype=torch.float32).view(1, 3, 1, 1) * 3
    normalize = transforms.Normalize(mean, std)
    
    return model, tokenizer, normalize

# PATCH 3: Modify encode functions to use consistent precision
def encode_text(model, tokenizer, device, text: str):
    tokens = tokenizer(text).to(device)
    with torch.no_grad():
        # Use FP32 instead of autocast for consistency
        feats = model.encode_text(tokens).float()
        feats = feats / feats.norm(dim=-1, keepdim=True)
    return feats

def encode_image(model, normalize, device, img: Image.Image, res: int = 224):
    t = torch.tensor(np.array(img), dtype=torch.float32) / 255.0
    if t.ndim == 3:
        t = t.permute(2, 0, 1)
    t = t.unsqueeze(0).to(device)
    t = F.interpolate(t, size=(res, res), mode="bicubic", align_corners=False)
    t = normalize(t)
    with torch.no_grad():
        # Use FP32 instead of autocast
        feats = model.encode_image(t).float()
        feats = feats / feats.norm(dim=-1, keepdim=True)
    return feats

# PATCH 4: Add deterministic validation wrapper
def validate_with_production_logic_deterministic(ply_data: bytes, prompt: str) -> dict:
    '''
    Deterministic wrapper for production validation
    '''
    # Ensure deterministic state before validation
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)
        torch.cuda.manual_seed_all(42)
    
    # Clear GPU state
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    
    print(f"🔒 Running deterministic validation")
    print(f"   TF32 disabled: {not torch.backends.cuda.matmul.allow_tf32}")
    print(f"   Deterministic algorithms: {torch.are_deterministic_algorithms_enabled()}")
    
    # Call original validation function
    return validate_with_production_logic(ply_data, prompt)
"""

# Configuration for deterministic validation
DETERMINISTIC_VALIDATION_CONFIG = {
    # Disable mixed precision for validation
    'use_amp': False,
    'use_fp16': False,
    
    # Use consistent batch processing
    'batch_size': 1,
    
    # Disable optimizations that can introduce variance
    'enable_flash_attention': False,
    'enable_memory_efficient_attention': False,
}

def patch_validation_engine():
    """
    Runtime patches for the validation engine to ensure determinism
    """
    # These would need to be applied after importing the validation modules
    # but before running validation
    
    try:
        import torch
        
        # Ensure consistent floating point behavior
        torch.set_float32_matmul_precision('highest')  # Most precise
        
        # Disable optimizations that can introduce variance
        torch._C._set_print_sparse_tensors(False)
        
        print("✅ Validation engine patched for determinism")
        
    except Exception as e:
        print(f"⚠️ Could not apply all validation patches: {e}")

def create_deterministic_request(ply_data: bytes, prompt: str, compression: int = 2):
    """
    Create a RequestData object with deterministic settings
    """
    import base64
    from validation.engine.data_structures import RequestData
    
    # Ensure consistent encoding
    encoded_data = base64.b64encode(ply_data).decode('ascii')  # Force ASCII
    
    request_data = RequestData(
        prompt=prompt.strip(),  # Normalize whitespace
        data=encoded_data,
        compression=compression,
        generate_preview=False,  # Disable for consistency
        preview_score_threshold=0.8
    )
    
    return request_data

# Usage instructions for subnet_accurate_validator2.py:
INTEGRATION_INSTRUCTIONS = """
To integrate these patches into your subnet_accurate_validator2.py:

1. Add the determinism setup at the very beginning:
   ```python
   # Add after line 12 (after CUDA workspace config)
   setup_validator_determinism()
   ```

2. Replace the validation function call in main():
   ```python
   # Replace line 489:
   # results = validate_with_production_logic(ply_data, original_prompt)
   results = validate_with_production_logic_deterministic(ply_data, original_prompt)
   ```

3. Modify the CLIP loading to use the patched version

4. Add GPU info logging for debugging:
   ```python
   def log_gpu_info():
       if torch.cuda.is_available():
           print(f"🔧 GPU: {torch.cuda.get_device_name()}")
           print(f"   Compute Capability: {torch.cuda.get_device_capability()}")
           print(f"   TF32 Support: {torch.cuda.get_device_capability()[0] >= 8}")
           print(f"   TF32 Enabled: {torch.backends.cuda.matmul.allow_tf32}")
   ```

This should significantly reduce the variance between different GPU setups.
"""

if __name__ == "__main__":
    print("Validator Determinism Patches")
    print("=" * 50)
    print(INTEGRATION_INSTRUCTIONS)

