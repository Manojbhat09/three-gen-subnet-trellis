# vLLM Server Setup Guide

This guide explains how to install and set up vLLM on your server for fast LLM inference, which can be used as an alternative to Ollama for the prompt optimization scripts.

## What is vLLM?

vLLM is a high-performance inference engine for large language models that provides:
- **Faster inference** compared to Ollama
- **Better GPU memory management**
- **Higher throughput** for multiple requests
- **Optimized CUDA kernels** for various model architectures

## Prerequisites

- NVIDIA GPU with CUDA support
- CUDA 11.8 or 12.1 installed
- Python 3.11 (recommended)
- Conda or Miniconda installed

## Installation Steps

### 1. Create and Activate Conda Environment

```bash
# Create a new conda environment with Python 3.11
conda create -n vllm python=3.11

# Activate the environment
conda activate vllm
```

### 2. Install vLLM

```bash
# Install vLLM with pip
pip install vllm
```

**Note**: The first installation may take several minutes as it compiles CUDA kernels.

### 3. Verify Installation

```bash
# Check if vLLM is installed correctly
python -c "import vllm; print('vLLM installation successful!')"

# Check CUDA availability
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

## Starting the vLLM Server

### Basic Server Start

```bash
# Start vLLM server with a specific model
CUDA_VISIBLE_DEVICES=3 vllm serve NousResearch/Hermes-3-Llama-3.2-3B \
  --served-model-name llama-3-2-3b-it \
  --generation-config auto \
  --port 9000
```

### Command Breakdown

- `CUDA_VISIBLE_DEVICES=3`: Use GPU device 3 (adjust based on your setup)
- `vllm serve`: Start the vLLM server
- `NousResearch/Hermes-3-Llama-3.2-3B`: Model to load from Hugging Face
- `--served-model-name llama-3-2-3b-it`: Name for the model in the API
- `--generation-config auto`: Use automatic generation configuration
- `--port 9000`: Port to serve the API on

### Alternative Models

You can use different models by changing the model path:

```bash
# Meta's Llama 3.2 3B
CUDA_VISIBLE_DEVICES=3 vllm serve meta-llama/Llama-3.2-3B-Instruct \
  --served-model-name llama-3-2-3b-it \
  --generation-config auto \
  --port 9000

# Microsoft's Phi-3
CUDA_VISIBLE_DEVICES=3 vllm serve microsoft/Phi-3-mini-4k-instruct \
  --served-model-name phi-3-mini \
  --generation-config auto \
  --port 9000

# Mistral 7B
CUDA_VISIBLE_DEVICES=3 vllm serve mistralai/Mistral-7B-Instruct-v0.2 \
  --served-model-name mistral-7b \
  --generation-config auto \
  --port 9000
```

## Server Configuration Options

### Memory and Performance

```bash
# Limit GPU memory usage
CUDA_VISIBLE_DEVICES=3 vllm serve NousResearch/Hermes-3-Llama-3.2-3B \
  --served-model-name llama-3-2-3b-it \
  --generation-config auto \
  --port 9000 \
  --max-model-len 4096 \
  --gpu-memory-utilization 0.8

# Enable tensor parallelism for larger models
CUDA_VISIBLE_DEVICES=3 vllm serve NousResearch/Hermes-3-Llama-3.2-3B \
  --served-model-name llama-3-2-3b-it \
  --generation-config auto \
  --port 9000 \
  --tensor-parallel-size 2
```

### Advanced Options

```bash
# Enable continuous batching for better throughput
CUDA_VISIBLE_DEVICES=3 vllm serve NousResearch/Hermes-3-Llama-3.2-3B \
  --served-model-name llama-3-2-3b-it \
  --generation-config auto \
  --port 9000 \
  --enable-chunked-prefill

# Set custom generation parameters
CUDA_VISIBLE_DEVICES=3 vllm serve NousResearch/Hermes-3-Llama-3.2-3B \
  --served-model-name llama-3-2-3b-it \
  --generation-config auto \
  --port 9000 \
  --max-num-batched-tokens 4096
```

## Testing the Server

### Health Check

```bash
# Check if server is running
curl http://localhost:9000/health
```

### Test API Endpoint

```bash
# Test the chat completions endpoint
curl http://localhost:9000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "llama-3-2-3b-it",
    "messages": [{"role": "user", "content": "Hello, how are you?"}],
    "stream": false,
    "temperature": 0.7,
    "max_tokens": 100,
    "top_p": 0.9
  }'
```

### Python Test Script

Create a test script to verify the server:

```python
import requests

def test_vllm_server():
    url = "http://localhost:9000/v1/chat/completions"
    data = {
        "model": "llama-3-2-3b-it",
        "messages": [{"role": "user", "content": "Write a short poem about AI"}],
        "stream": False,
        "temperature": 0.7,
        "max_tokens": 150,
        "top_p": 0.9
    }
    
    try:
        response = requests.post(url, json=data, timeout=30)
        response.raise_for_status()
        result = response.json()
        print("✅ vLLM server test successful!")
        print(f"Response: {result['choices'][0]['message']['content']}")
    except Exception as e:
        print(f"❌ vLLM server test failed: {e}")

if __name__ == "__main__":
    test_vllm_server()
```

## Using vLLM with Prompt Optimization Scripts

### 1. Smart Prompt Optimizer

```bash
# Use vLLM instead of Ollama
python smart_prompt_optimizer_v5_rl_loop_lora.py "your prompt" --vllm

# Custom vLLM configuration
python smart_prompt_optimizer_v5_rl_loop_lora.py "your prompt" \
  --vllm --vllm-url http://localhost:9000 --vllm-model llama-3-2-3b-it
```

### 2. Episodic Prompt Optimizer

```bash
# Run episodic optimization with vLLM
python run_episodic_optimization.py --episodes 10 --vllm

# Full configuration
CUDA_VISIBLE_DEVICES=2 python run_episodic_optimization.py \
  --episodes 15 --target 0.95 --max-rounds 2 \
  --log-dir episodic_logs_vllm --endpoint "generate/cinema/" \
  --port 8097 --vllm --vllm-url http://localhost:9000 \
  --vllm-model llama-3-2-3b-it
```

### 3. Standalone LLM Optimizer

```bash
# Use vLLM for prompt optimization
python llm_prompt_optimizer_v12_f1_lora.py "your prompt" --method 2 --vllm

# Custom configuration
python llm_prompt_optimizer_v12_f1_lora.py "your prompt" --method 1 \
  --vllm --vllm-url http://localhost:9000 --vllm-model llama-3-2-3b-it
```

## Using vLLM with TRELLIS Mining Scripts

### 4. Unified Mining Runner

The `run_trellis_mining.sh` script provides a unified interface for running different mining modes with vLLM support:

```bash
# Run continuous mining with vLLM
./run_trellis_mining.sh --continuous --start-server --vllm

# Run continuous mining with custom vLLM settings
./run_trellis_mining.sh --continuous --start-server \
  --vllm --vllm-url http://localhost:9001 --vllm-model llama-3-2-8b-it

# Run simulation mode with vLLM
./run_trellis_mining.sh --simulate --promptfile episodic_test_prompts.py \
  --start-server --vllm --vllm-url http://localhost:9000 --vllm-model llama-3-2-3b-it

# Run one-shot mining with vLLM
./run_trellis_mining.sh --harvest --submit --max-tasks 5 --vllm
```

### 5. Continuous TRELLIS Orchestrator

The `continuous_trellis_orchestrator_lora_working.py` script can be run directly with vLLM:

```bash
# Run continuous orchestrator with vLLM
python3 continuous_trellis_orchestrator_lora_working.py --vllm

# Custom vLLM configuration
python3 continuous_trellis_orchestrator_lora_working.py \
  --vllm --vllm-url http://localhost:9000 --vllm-model llama-3-2-3b-it

# Mixed configuration (vLLM for optimization, other settings)
python3 continuous_trellis_orchestrator_lora_working.py \
  --vllm --vllm-url http://localhost:9000 --vllm-model llama-3-2-3b-it \
  --no-lora-routing --blacklist 180 --no-validate

# Full production configuration with vLLM
python3 continuous_trellis_orchestrator_lora_working.py \
  --vllm --vllm-url http://localhost:9000 --vllm-model llama-3-2-3b-it \
  --aggressive-optimize --variable-seeds --reproducibility-similarity 0.6
```

### 6. Mining Script vLLM Arguments

All mining scripts support the following vLLM arguments:

| Argument | Description | Default |
|----------|-------------|---------|
| `--vllm` | Enable vLLM mode instead of Ollama | `False` |
| `--vllm-url` | vLLM server URL | `http://localhost:9000` |
| `--vllm-model` | vLLM model name | `llama-3-2-3b-it` |

### 7. Complete Mining Workflow with vLLM

Here's a complete example of setting up and running a vLLM-powered mining operation:

```bash
# Terminal 1: Start vLLM server
conda activate vllm
CUDA_VISIBLE_DEVICES=3 vllm serve NousResearch/Hermes-3-Llama-3.2-3B \
  --served-model-name llama-3-2-3b-it \
  --generation-config auto \
  --port 9000

# Terminal 2: Run continuous mining with vLLM
./run_trellis_mining.sh --continuous --start-server \
  --vllm --vllm-url http://localhost:9000 --vllm-model llama-3-2-3b-it

# Or run the orchestrator directly
python3 continuous_trellis_orchestrator_lora_working.py \
  --vllm --vllm-url http://localhost:9000 --vllm-model llama-3-2-3b-it
```

### 8. Mining Script Benefits with vLLM

When using vLLM with the mining scripts, you get:

- **Faster Prompt Optimization**: vLLM's optimized inference speeds up prompt optimization cycles
- **Better Resource Utilization**: More efficient GPU memory usage during mining operations
- **Higher Throughput**: Process more mining tasks simultaneously
- **Reduced Latency**: Faster response times for prompt optimization requests
- **Production Ready**: Stable inference server for continuous mining operations

## Performance Comparison

| Metric | Ollama | vLLM | Improvement |
|--------|--------|------|-------------|
| **Inference Speed** | ~10-15 tokens/sec | ~50-100 tokens/sec | **3-7x faster** |
| **Memory Efficiency** | Moderate | High | **2-3x better** |
| **Concurrent Requests** | 1-2 | 10+ | **5-10x more** |
| **GPU Utilization** | 60-80% | 90-95% | **20-30% better** |

## Troubleshooting

### Common Issues

#### 1. CUDA Out of Memory
```bash
# Reduce GPU memory usage
CUDA_VISIBLE_DEVICES=3 vllm serve NousResearch/Hermes-3-Llama-3.2-3B \
  --served-model-name llama-3-2-3b-it \
  --generation-config auto \
  --port 9000 \
  --gpu-memory-utilization 0.7
```

#### 2. Port Already in Use
```bash
# Use a different port
CUDA_VISIBLE_DEVICES=3 vllm serve NousResearch/Hermes-3-Llama-3.2-3B \
  --served-model-name llama-3-2-3b-it \
  --generation-config auto \
  --port 9001
```

#### 3. Model Download Issues
```bash
# Set Hugging Face token for private models
export HF_TOKEN="your_token_here"
CUDA_VISIBLE_DEVICES=3 vllm serve NousResearch/Hermes-3-Llama-3.2-3B \
  --served-model-name llama-3-2-3b-it \
  --generation-config auto \
  --port 9000
```

### Debug Mode

```bash
# Enable debug logging
CUDA_VISIBLE_DEVICES=3 vllm serve NousResearch/Hermes-3-Llama-3.2-3B \
  --served-model-name llama-3-2-3b-it \
  --generation-config auto \
  --port 9000 \
  --log-level DEBUG
```

## Monitoring and Maintenance

### Check Server Status

```bash
# Health check
curl http://localhost:9000/health

# Model info
curl http://localhost:9000/v1/models
```

### Resource Monitoring

```bash
# Monitor GPU usage
nvidia-smi

# Monitor memory usage
watch -n 1 'nvidia-smi --query-gpu=memory.used,memory.total --format=csv'
```

### Restart Server

```bash
# Stop the server (Ctrl+C)
# Then restart with the same command
CUDA_VISIBLE_DEVICES=3 vllm serve NousResearch/Hermes-3-Llama-3.2-3B \
  --served-model-name llama-3-2-3b-it \
  --generation-config auto \
  --port 9000
```

## Best Practices

1. **GPU Selection**: Use `CUDA_VISIBLE_DEVICES` to specify which GPU to use
2. **Memory Management**: Monitor GPU memory usage and adjust `--gpu-memory-utilization` accordingly
3. **Model Caching**: vLLM automatically caches models in GPU memory for faster subsequent loads
4. **Batch Processing**: Enable continuous batching for better throughput with multiple requests
5. **Regular Updates**: Keep vLLM updated for the latest performance improvements

## Conclusion

vLLM provides significant performance improvements over Ollama for LLM inference, making it an excellent choice for production prompt optimization workflows. The setup is straightforward and the performance gains are substantial, especially for batch processing and concurrent requests.

For more information, visit the [vLLM documentation](https://docs.vllm.ai/) or check the [GitHub repository](https://github.com/vllm-project/vllm).
