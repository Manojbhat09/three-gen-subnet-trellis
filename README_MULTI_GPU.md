# Multi-GPU FLUX + TRELLIS Generation Server

## Overview

This is a high-performance 3D model generation server that utilizes **8 GPUs in parallel** to generate multiple 3D models simultaneously. The server combines FLUX (text-to-image) and TRELLIS (image-to-3D) pipelines with intelligent GPU management and load balancing.

## 🚀 Key Features

- **8-GPU Parallel Processing**: Generate up to 8 models simultaneously
- **Intelligent GPU Management**: Automatic load balancing and scheduling
- **GPU-Specific Model Instances**: Each GPU has its own model instance for optimal memory management
- **Real-time GPU Monitoring**: Live status updates and utilization tracking
- **Multiple LoRA Styles**: Support for various artistic styles (Isometric 3D, Live 3D, Game Assets, etc.)
- **SPZ Compression**: Efficient storage and transmission of generated models
- **RESTful API**: Easy integration with web applications and automation scripts

## 🏗️ Architecture

```
Text Prompt → FLUX (GPU 0-7) → Image → TRELLIS → Gaussian Splatting PLY → SPZ Compression
```

Each GPU runs an independent instance of the FLUX + TRELLIS pipeline, allowing true parallel processing.

## 📋 Requirements

- **Hardware**: 8 NVIDIA GPUs (RTX 4090, A100, H100, etc.)
- **Memory**: Minimum 20GB VRAM per GPU
- **Software**: CUDA 11.8+, PyTorch 2.0+, Python 3.8+
- **Dependencies**: See requirements.txt

## 🚀 Quick Start

### 1. Start the Server

```bash
# Start with default 8 GPUs
python trellis_subnit_server_mix_lora_flash_8x.py --host 0.0.0.0 --port 8096

# Start with custom number of GPUs
python trellis_subnit_server_mix_lora_flash_8x.py --gpus 4 --port 8096
```

### 2. Check GPU Status

```bash
curl "http://localhost:8096/gpu_status/"
```

### 3. Generate Single Model on Specific GPU

```bash
curl -X POST "http://localhost:8096/generate/" \
  -F "prompt=a beautiful ceramic vase with intricate patterns" \
  -F "seed=42" \
  -F "gpu_id=0" \
  -F "return_compressed=true"
```

### 4. Generate 8 Models in Parallel

```bash
curl -X POST "http://localhost:8096/generate_parallel/" \
  -F "prompt=a futuristic robot with glowing blue eyes" \
  -F "seeds=42,43,44,45,46,47,48,49" \
  -F "return_compressed=true" \
  -F "max_parallel=8"
```

### 5. Generate with Specific LoRA Style

```bash
curl -X POST "http://localhost:8096/generate_parallel_lora/isometric_3d/" \
  -F "prompt=a modern office building" \
  -F "seeds=50,51,52,53,54,55,56,57" \
  -F "return_compressed=true" \
  -F "max_parallel=8"
```

## 🔌 API Endpoints

### Core Generation

- `POST /generate/` - Single generation on specific GPU
- `POST /generate_parallel/` - Parallel generation across GPUs
- `POST /generate_parallel_lora/{style}/` - Parallel LoRA generation

### GPU Management

- `GET /gpu_status/` - Real-time GPU status and utilization
- `GET /parallel_jobs/` - Parallel job status and progress
- `POST /gpu_reset/{id}` - Reset specific GPU
- `POST /gpu_cleanup/` - Clean up all GPUs

### Status & Monitoring

- `GET /status/` - Server status with multi-GPU info
- `GET /health/` - Health check
- `GET /assets/` - Generated assets information

### LoRA-Specific Endpoints

- `POST /generate/{lora_style}/` - Generate with specific LoRA
- `POST /generate_image/{lora_style}/` - Generate image only with LoRA

## 🎯 Usage Examples

### Python Client Example

```python
import requests

# Generate 8 models in parallel
def generate_parallel_models(prompt, seeds):
    data = {
        'prompt': prompt,
        'seeds': ','.join(map(str, seeds)),
        'return_compressed': True,
        'max_parallel': 8
    }
    
    response = requests.post(
        "http://localhost:8096/generate_parallel/", 
        data=data
    )
    
    if response.status_code == 200:
        result = response.json()
        print(f"Generated {result['successful']}/{result['total_models']} models")
        return result
    else:
        print(f"Generation failed: {response.text}")
        return None

# Usage
seeds = list(range(42, 50))  # 8 seeds
result = generate_parallel_models(
    "a beautiful ceramic vase with intricate patterns", 
    seeds
)
```

### Batch Processing Script

```bash
#!/bin/bash

# Generate 24 models using 3 batches of 8
for batch in {1..3}; do
    echo "Starting batch $batch..."
    
    # Calculate seeds for this batch
    start_seed=$((40 + batch * 8))
    seeds=$(seq -s, $start_seed $((start_seed + 7)))
    
    # Generate models
    curl -X POST "http://localhost:8096/generate_parallel/" \
        -F "prompt=a futuristic robot with glowing blue eyes" \
        -F "seeds=$seeds" \
        -F "return_compressed=true" \
        -F "max_parallel=8"
    
    echo "Batch $batch completed"
    sleep 10  # Wait between batches
done
```

## 🔧 Configuration

### GPU Settings

```python
MULTI_GPU_CONFIG = {
    'num_gpus': 8,                    # Number of GPUs to use
    'gpu_memory_limit_gb': 20,        # Memory limit per GPU
    'parallel_generation_limit': 8,    # Max parallel generations
    'gpu_affinity': True,             # Bind processes to specific GPUs
    'load_balancing': 'round_robin',  # Load balancing strategy
}
```

### Load Balancing Strategies

- **`round_robin`**: Simple sequential GPU assignment
- **`least_loaded`**: Assign to GPU with lowest memory usage
- **`random`**: Random assignment from available GPUs

### Memory Management

- Each GPU instance has a 20GB memory limit
- Automatic memory cleanup after generation
- GPU-specific model instances for optimal memory usage

## 📊 Performance Monitoring

### GPU Status Response

```json
{
  "num_gpus": 8,
  "gpu_states": {
    "0": {
      "status": "idle",
      "current_job": null,
      "memory_used_gb": 2.1,
      "memory_total_gb": 24.0,
      "temperature": 65.0,
      "utilization": 0.0
    }
  },
  "total_jobs": 0,
  "available_gpus": 8,
  "busy_gpus": 0
}
```

### Parallel Generation Response

```json
{
  "status": "completed",
  "total_models": 8,
  "successful": 8,
  "failed": 0,
  "success_rate": 100.0,
  "total_time": 45.2,
  "jobs": [...],
  "compressed_models": [...]
}
```

## 🚨 Troubleshooting

### Common Issues

1. **GPU Memory Errors**
   ```bash
   # Reset specific GPU
   curl -X POST "http://localhost:8096/gpu_reset/0"
   
   # Clean up all GPUs
   curl -X POST "http://localhost:8096/gpu_cleanup/"
   ```

2. **GPU Not Available**
   ```bash
   # Check GPU status
   curl "http://localhost:8096/gpu_status/"
   
   # Wait for GPU to become available or use different GPU
   ```

3. **Model Loading Failures**
   ```bash
   # Check server status
   curl "http://localhost:8096/status/"
   
   # Restart server if models are corrupted
   ```

### Performance Optimization

1. **Memory Management**
   - Monitor GPU memory usage
   - Use `gpu_cleanup/` periodically
   - Adjust `gpu_memory_limit_gb` based on your hardware

2. **Load Balancing**
   - Use `least_loaded` strategy for better distribution
   - Monitor GPU utilization patterns
   - Adjust `parallel_generation_limit` based on workload

3. **Batch Processing**
   - Group similar prompts together
   - Use consistent seeds for reproducible results
   - Implement retry logic for failed generations

## 🔬 Testing

Run the test script to verify multi-GPU functionality:

```bash
python test_multi_gpu.py
```

This will test:
- Server connectivity
- GPU status monitoring
- Single generation on specific GPU
- Parallel generation across all GPUs
- LoRA-specific parallel generation
- Job status tracking

## 📈 Scaling Considerations

### Horizontal Scaling
- Run multiple server instances on different machines
- Use load balancer to distribute requests
- Implement Redis for job queue management

### Vertical Scaling
- Increase GPU memory limits
- Optimize model quantization
- Use mixed precision training

### Monitoring & Alerting
- GPU temperature monitoring
- Memory usage alerts
- Generation success rate tracking
- Performance metrics collection

## 🤝 Contributing

1. Fork the repository
2. Create feature branch
3. Implement multi-GPU improvements
4. Add tests for new functionality
5. Submit pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🆘 Support

For issues and questions:
1. Check the troubleshooting section
2. Review GPU status and logs
3. Open an issue on GitHub
4. Contact the development team

---

**Happy Multi-GPU Generation! 🚀🎨**


