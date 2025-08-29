# CPU-based CLIP Episodic Optimizer

This document explains how to use the modified CLIP episodic optimizer that runs on CPU with vLLM integration.

## Overview

The CPU-based CLIP optimizer has been modified to:
- ✅ Run CLIP models on CPU instead of GPU
- ✅ Use vLLM for prompt optimization instead of OpenRouter/Ollama
- ✅ Use only the Cinema Style endpoint for image generation
- ✅ Work alongside the existing miner without conflicts
- ✅ Maintain episodic memory and learning capabilities

## Prerequisites

1. **vLLM Server**: Running on port 11300
   ```bash
   # Make sure vLLM is running on localhost:11300
   ```

2. **Image Generation Server**: Running on port 8096
   ```bash
   # The miner/orchestrator should be running this
   ```

3. **Required Python Packages**:
   ```bash
   pip install torch open_clip_torch requests
   ```

## Usage

### Basic Usage

```python
from clip_episodic_optimizer import MultiGeneratorCLIPOptimizer

# Create CPU-based optimizer
optimizer = MultiGeneratorCLIPOptimizer(
    num_episodes=3,
    target_score=0.8,
    max_rounds_per_episode=20,
    enable_router=False,  # CPU mode
    use_cpu=True,
    vllm_url="http://localhost:11300"
)

# Test prompts
test_prompts = [
    "small wooden hammer",
    "red sports car",
    "crystal vase"
]

# Run optimization
results = optimizer.run_all_episodes(test_prompts)
```

### Command Line Usage

```bash
cd /home/mbhat/three-gen-subnet-trellis
python clip_episodic_optimizer.py
```

### Test Script

```bash
cd /home/mbhat/three-gen-subnet-trellis
python test_cpu_optimizer.py
```

## Key Changes Made

### 1. CPU-based CLIP Processing
- CLIP models now load on CPU instead of GPU
- Memory usage is optimized for CPU processing
- Automatic fallback if CUDA is not available

### 2. vLLM Integration
- Replaced OpenRouter/Ollama with vLLM for prompt optimization
- Direct HTTP API calls to vLLM server
- Fallback handling for connection issues

### 3. Simplified Generator Selection
- Removed router dependency
- Always uses Cinema Style endpoint: `http://localhost:8096/generate_image/cinema/`
- Commented out multi-generator tie-breaking logic

### 4. Memory and Logging
- Separate memory file for CPU mode: `cpu_episodic_clip_memory.json`
- Separate log directory: `cpu_episodic_clip_logs`
- vLLM connection testing on startup

## Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   CLIP Optimizer │────│      vLLM        │────│   Image Server   │
│     (CPU)        │    │   (localhost:    │    │  (localhost:    │
│                  │    │     11300)      │    │      8096)      │
└─────────────────┘    └──────────────────┘    └─────────────────┘
        │                        │                        │
        └─Episodic Memory────────┼─Prompt Optimization─────┘
                                 │
                    ┌──────────────────┐
                    │   Miner/         │
                    │ Orchestrator     │
                    │   (Independent)  │
                    └──────────────────┘
```

## Configuration Options

| Parameter | Default | Description |
|-----------|---------|-------------|
| `use_cpu` | `True` | Force CPU usage for CLIP models |
| `vllm_url` | `"http://localhost:11300"` | vLLM server endpoint |
| `enable_router` | `False` | Disabled for CPU-only mode |
| `num_episodes` | `3` | Number of optimization episodes |
| `target_score` | `0.8` | Target CLIP score to achieve |
| `max_rounds_per_episode` | `20` | Max optimization rounds per episode |

## Concurrent Operation

The CPU-based optimizer is designed to work alongside the existing miner:

1. **Independent Processes**: Optimizer and miner run as separate processes
2. **Shared Resources**: Both use the same image generation server (port 8096)
3. **Priority Handling**: Server handles requests based on priority
4. **Resource Management**: CPU optimizer uses minimal GPU resources

## Troubleshooting

### vLLM Connection Issues
```bash
# Test vLLM connection
curl -X POST http://localhost:11300/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "default", "messages": [{"role": "user", "content": "Hello"}], "max_tokens": 10}'
```

### Image Generation Issues
```bash
# Test image generation endpoint
curl -X POST http://localhost:8096/generate_image/cinema/ \
  -H "Content-Type: application/json" \
  -d '{"prompt": "test prompt", "seed": 42, "num_inference_steps": 7}'
```

### Memory Issues
- The CPU optimizer uses episodic memory to avoid duplicate work
- Memory files are saved automatically on exit
- Check memory file: `cpu_episodic_clip_memory.json`

## Performance Notes

- **CPU Usage**: CLIP model inference runs on CPU
- **Memory**: ~2-4GB RAM usage for CLIP models
- **Speed**: Slower than GPU but works without CUDA
- **Optimization**: RL agent learns from previous sessions
- **Persistence**: All learning is saved between runs

## Files Modified

1. `clip_episodic_optimizer.py` - Main optimizer with CPU support
2. `test_cpu_optimizer.py` - Test script for validation
3. `README_CPU_OPTIMIZER.md` - This documentation

## Next Steps

1. Start the CPU optimizer when the miner is idle
2. Monitor performance and adjust parameters as needed
3. The optimizer will continuously improve prompts using learned patterns
4. Results are saved and can be analyzed for insights

