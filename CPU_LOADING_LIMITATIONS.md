# CPU Loading Limitations

## Overview

The `--cpu-loading` option allows loading some models on CPU instead of GPU, but has important limitations due to the 3D rendering pipeline requirements.

## What Works with CPU Loading ✅

### 1. CLIP Scoring (`--clip-score`)
- **Custom CLIP model** (ViT-B-32 + OpenAI weights) runs on CPU
- **Image-text similarity** calculation works on CPU
- **No 3D rendering required**

```bash
python test_rl_standalone.py "a simple red car" --clip-score --cpu-loading
```

### 2. Alignment Scoring (`--alignment-score`)
- **Validation system's CLIP model** (ConvNeXt Large D) runs on CPU
- **Text-image alignment** calculation works on CPU
- **No 3D rendering required**

```bash
python test_rl_standalone.py "a simple red car" --alignment-score --cpu-loading
```

### 3. Both Scores (`--both-scores`)
- **Both CLIP and alignment models** run on CPU
- **No 3D rendering required**

```bash
python test_rl_standalone.py "a simple red car" --both-scores --cpu-loading
```

## What Doesn't Work with CPU Loading ❌

### 1. RL Optimization (`--rl-alignment`, `--enhanced`)
- **Requires 3D rendering** (Gaussian Splatting)
- **Gaussian Splatting requires CUDA** and cannot run on CPU
- **Will fail with error**: `quats.value() must be a CUDA tensor`

```bash
# This will FAIL
python test_rl_standalone.py "a simple red car" --rl-alignment --cpu-loading
```

### 2. Any Operation Requiring 3D Rendering
- **Full validation** (quality + alignment + rendering)
- **RL optimization** (requires validation)
- **Enhanced validation** (includes 3D rendering)

## Technical Details

### Why 3D Rendering Requires CUDA

1. **Gaussian Splatting**: The 3D rendering pipeline uses Gaussian Splatting, which is implemented in CUDA
2. **CUDA Kernels**: The rendering operations use CUDA kernels that cannot run on CPU
3. **Tensor Requirements**: All tensors must be CUDA tensors for the rendering pipeline

### Error Message
```
RuntimeError: quats.value() must be a CUDA tensor
```

This error occurs because:
- The validation system loads models on CPU when `--cpu-loading` is used
- But the 3D rendering pipeline expects CUDA tensors
- The Gaussian Splatting renderer cannot handle CPU tensors

## Recommendations

### Use CPU Loading For:
- **Quick CLIP scoring** without 3D generation
- **Alignment scoring** without 3D generation  
- **Development/testing** on systems with limited GPU memory
- **Batch processing** of image-text similarity

### Avoid CPU Loading For:
- **RL optimization** (requires 3D rendering)
- **Full validation** (requires 3D rendering)
- **Any operation** that needs 3D model generation

## Alternative Approaches

### 1. Use Score-Only Operations
```bash
# Instead of RL optimization, use direct scoring
python test_rl_standalone.py "prompt" --alignment-score --cpu-loading
```

### 2. Use GPU with Memory Management
```bash
# Use GPU but with memory-efficient operations
python test_rl_standalone.py "prompt" --alignment-score  # No --cpu-loading
```

### 3. Batch Processing
```bash
# Process multiple prompts with CPU loading for scoring only
for prompt in prompts; do
    python test_rl_standalone.py "$prompt" --clip-score --cpu-loading
done
```

## Summary

- ✅ **CPU Loading works** for CLIP and alignment scoring
- ❌ **CPU Loading fails** for 3D rendering operations
- 💡 **Use CPU Loading** for score-only operations
- 🚫 **Avoid CPU Loading** for RL optimization and validation


