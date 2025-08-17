# Multi-GPU Pipeline Wrapper - Subnet 17 (404-GEN)

A comprehensive pipeline system for multi-GPU image generation, CLIP scoring, PLY generation, and validation ranking.

## 🚀 Overview

This system implements two powerful pipelines that leverage all 8 GPUs for optimal performance:

### Pipeline 1: Image Ranking → PLY Generation
```
Text Prompt → [8x Image Generation] → CLIP Ranking → [Best Images → 8x PLY] → Validation Ranking
```
- Generates images on all 8 GPUs in parallel
- Ranks images using CLIP text-image similarity scores
- Distributes top-performing images across GPUs for PLY generation
- Validates and ranks PLY files by quality scores

### Pipeline 2: Single Image → Multi PLY Variations
```
Text Prompt → [8x Image Generation] → Best Image → [8x PLY Variations] → Validation Ranking
```
- Generates images on all 8 GPUs in parallel
- Selects the single best image based on CLIP score
- Generates 8 different PLY variations from the same best image
- Validates and ranks PLY variations by quality scores

## 🎯 Key Features

- **Multi-GPU Parallelization**: Utilizes all 8 GPUs simultaneously for maximum throughput
- **CLIP-Based Ranking**: Uses production-accurate CLIP scoring for image quality assessment
- **Production Validation**: Integrates with `subnet_accurate_validator_multigpu.py` for exact validation scoring
- **Performance Analysis**: Comprehensive GPU performance tracking and comparison
- **Automatic Load Balancing**: Intelligently distributes work across available GPUs
- **Robust Error Handling**: Graceful handling of GPU failures and recovery
- **Detailed Logging**: Complete audit trail of all operations and results

## 📋 Prerequisites

### Hardware Requirements
- 8 NVIDIA GPUs with sufficient VRAM (20GB+ recommended)
- CUDA-compatible compute capability

### Software Requirements
```bash
# Python dependencies
pip install torch torchvision open-clip-torch
pip install requests pillow numpy
pip install fastapi uvicorn

# System requirements
conda activate trellis_new  # Your TRELLIS environment
```

### Server Setup
Ensure TRELLIS servers can be started on ports 8096-8103:
```bash
# Your TRELLIS server script should be available
ls trellis_subnit_server_mix_lora_flash.py
```

## 🚀 Quick Start

### 1. Check GPU Status
```bash
python gpu_multi_pipeline_wrapper.py --check-status-only
```

### 2. Run Image Ranking Pipeline
```bash
python gpu_multi_pipeline_wrapper.py \
    --prompt "a vintage red bicycle with chrome details" \
    --pipeline image_ranking \
    --num-inference-steps 25 \
    --guidance-scale 7.5
```

### 3. Run Single Image Multi-PLY Pipeline
```bash
python gpu_multi_pipeline_wrapper.py \
    --prompt "a ceramic coffee mug with intricate patterns" \
    --pipeline single_image \
    --num-inference-steps 30 \
    --guidance-scale 8.0
```

### 4. Run Both Pipelines for Comparison
```bash
python gpu_multi_pipeline_wrapper.py \
    --prompt "a wooden chess piece with detailed carving" \
    --pipeline both \
    --num-inference-steps 25 \
    --guidance-scale 7.5
```

### 5. Run Comprehensive Test Suite
```bash
python test_multi_gpu_pipeline.py
```

## 📖 Detailed Usage

### Command Line Options

```bash
python gpu_multi_pipeline_wrapper.py [OPTIONS]

Required:
  --prompt TEXT                 Text prompt for generation

Pipeline Options:
  --pipeline {image_ranking,single_image,both}  Pipeline type to run (default: both)
  --num-inference-steps INT     Number of inference steps (default: 25)
  --guidance-scale FLOAT        Guidance scale (default: 7.5)

GPU Management:
  --gpus INT                    Number of GPUs to use (default: 8)
  --base-port INT               Base port number (default: 8096)
  --server-script TEXT          TRELLIS server script path
  --output-dir TEXT             Output directory (default: ./gpu_pipeline_outputs)

Server Control:
  --skip-startup                Skip server startup (assume already running)
  --check-status-only           Only check GPU loading status and exit
```

### Pipeline Types Explained

#### Image Ranking Pipeline
**Best for**: Finding the optimal image-to-3D conversion pipeline
```python
results = manager.run_image_ranking_to_ply_pipeline(
    prompt="a detailed mechanical watch",
    num_inference_steps=25,
    guidance_scale=7.5
)
```

**Process**:
1. Generate 8 images across GPUs with different seeds
2. Compute CLIP scores for each image
3. Rank images by CLIP similarity to prompt
4. Distribute top 4 images across all 8 GPUs for PLY generation
5. Validate and rank PLY files by quality scores

**Advantages**: 
- Maximizes image diversity
- Higher chance of finding exceptional results
- Better for complex prompts

#### Single Image Multi-PLY Pipeline
**Best for**: Exploring PLY generation variations from optimal images
```python
results = manager.run_single_image_multi_ply_pipeline(
    prompt="a ceramic vase with blue glaze",
    num_inference_steps=30,
    guidance_scale=8.0
)
```

**Process**:
1. Generate 8 images across GPUs with different seeds
2. Select the single best image by CLIP score
3. Use that image to generate 8 PLY variations (different seeds)
4. Validate and rank PLY files by quality scores

**Advantages**:
- Consistent source material
- Focuses on PLY generation quality
- Better for simple, well-defined objects

## 📊 Results and Analysis

### Output Files
Results are saved in structured JSON files:

```json
{
  "prompt": "a red ceramic coffee mug",
  "pipeline_type": "image_ranking_to_ply",
  "total_pipeline_time": 45.2,
  "best_image_gpu": 3,
  "best_ply_gpu": 7,
  "best_clip_score": 0.8247,
  "best_validation_score": 0.7891,
  "image_results": [...],
  "ply_results": [...]
}
```

### Performance Metrics
The system tracks comprehensive metrics:

- **Image Generation**: Success rates, CLIP scores, generation times
- **PLY Generation**: Validation scores, file sizes, compression ratios
- **GPU Utilization**: Per-GPU performance, memory usage, error rates
- **Pipeline Efficiency**: Total times, parallelization effectiveness

### Ranking Systems

#### CLIP Score Ranking (Images)
- Uses `convnext_large_d` model with `laion2b_s26b_b102k_augreg` weights
- Text-image similarity scores from 0.0 to 1.0
- Higher scores indicate better alignment with text prompt

#### Validation Score Ranking (PLY Files)
- Uses production `subnet_accurate_validator_multigpu.py`
- Comprehensive scoring including alignment, quality, and fidelity
- Scores from 0.0 to 1.0 with multiple sub-metrics

## 🧪 Testing and Validation

### Basic Functionality Test
```bash
# Test basic pipeline functionality
python test_multi_gpu_pipeline.py
```

This runs:
- Multiple prompt tests
- Both pipeline comparisons
- Error handling validation
- Performance analysis

### Performance Benchmarking
The test suite includes:
- **Throughput Tests**: Multiple prompts processed sequentially
- **Stress Tests**: Complex prompts with high resource usage  
- **Memory Tests**: GPU memory utilization tracking
- **Failure Recovery**: GPU error simulation and recovery

### GPU Health Monitoring
Real-time monitoring includes:
- Server responsiveness checks
- Memory allocation tracking
- Error rate monitoring
- Performance ranking updates

## 🔧 Advanced Configuration

### Custom CLIP Models
```python
# Modify CLIPScorer class to use different models
scorer = CLIPScorer(device="cuda")
scorer.model, _, _ = open_clip.create_model_and_transforms(
    "ViT-L-14", pretrained="openai", device=scorer.device
)
```

### Custom Validation Logic
```python
# Override validation method for custom scoring
def _validate_ply_data(self, ply_data: bytes, prompt: str, gpu_id: int):
    # Custom validation implementation
    return custom_validation_score
```

### Pipeline Customization
```python
# Create custom pipeline combinations
class CustomPipelineManager(MultiGPUPipelineManager):
    def run_hybrid_pipeline(self, prompt: str):
        # Custom pipeline logic
        pass
```

## 📈 Performance Optimization

### GPU Memory Management
- Automatic memory cleanup between generations
- Smart caching of frequently used models
- Memory usage monitoring and alerts

### Load Balancing Strategies
- Round-robin GPU assignment
- Least-loaded GPU selection
- Performance-based GPU ranking

### Network Optimization
- Parallel HTTP requests to GPU servers
- Request timeout management
- Automatic retry logic

## 🛠️ Troubleshooting

### Common Issues

#### GPU Servers Not Starting
```bash
# Check GPU availability
nvidia-smi

# Check port availability
netstat -tlnp | grep 809

# Restart servers manually
python gpu_server_wrapper.py --skip-priming --skip-validation
```

#### Memory Issues
```bash
# Clear GPU memory
python -c "import torch; torch.cuda.empty_cache()"

# Check memory usage
python gpu_multi_pipeline_wrapper.py --check-status-only
```

#### CLIP Scoring Disabled
```bash
# Install missing dependencies
pip install open-clip-torch
pip install torch torchvision
```

#### Validation Failures
```bash
# Check validator availability
python subnet_accurate_validator_multigpu.py "test prompt" "test prompt"

# Verify environment
conda activate trellis_new
```

### Debug Mode
Enable detailed logging:
```python
import logging
logging.getLogger().setLevel(logging.DEBUG)
```

## 📚 API Reference

### MultiGPUPipelineManager

#### Core Methods
- `generate_images_parallel(prompt, seeds=None, ...)` - Generate images on all GPUs
- `generate_plys_parallel(prompt, source_images=None, ...)` - Generate PLY files on all GPUs
- `run_image_ranking_to_ply_pipeline(prompt, ...)` - Run complete image ranking pipeline
- `run_single_image_multi_ply_pipeline(prompt, ...)` - Run single image multi-PLY pipeline

#### Analysis Methods
- `print_pipeline_summary()` - Print comprehensive statistics
- `save_pipeline_results(results, timestamp)` - Save results to JSON
- `check_all_servers_health()` - Health check all GPU servers

### CLIPScorer

#### Scoring Methods
- `compute_text_image_similarity(text, image)` - Compute CLIP similarity score
- `encode_text(text)` - Encode text to CLIP features
- `encode_image(image)` - Encode image to CLIP features

### Data Structures

#### ImageGenerationResult
- `gpu_id` - Source GPU ID
- `success` - Generation success status
- `clip_score` - Text-image CLIP similarity score
- `generation_time` - Time taken for generation
- `pil_image` - PIL Image object

#### PLYGenerationResult  
- `gpu_id` - Source GPU ID
- `success` - Generation success status
- `validation_score` - Production validation score
- `ply_size` - Size of generated PLY file
- `generation_time` - Time taken for generation

#### PipelineResults
- `pipeline_type` - Type of pipeline executed
- `image_results` - List of image generation results
- `ply_results` - List of PLY generation results
- `best_image_gpu` - GPU ID of best image
- `best_ply_gpu` - GPU ID of best PLY
- `total_pipeline_time` - Total execution time

## 🔄 Integration with Existing Systems

### With GPU Server Wrapper
```python
# Extends existing GPUServerManager
from gpu_server_wrapper import GPUServerManager
class MultiGPUPipelineManager(GPUServerManager):
    # Additional pipeline functionality
```

### With Subnet Validator
```python
# Uses existing validation logic
result = subprocess.run([
    sys.executable,
    "subnet_accurate_validator_multigpu.py",
    prompt, prompt, "--port", str(port)
])
```

### With TRELLIS Servers
```python
# Compatible with existing server endpoints
response = requests.post(f"http://127.0.0.1:{port}/generate_image/", ...)
response = requests.post(f"http://127.0.0.1:{port}/generate/", ...)
```

## 📝 Examples and Use Cases

### Research and Development
- Compare different generation strategies
- Analyze performance across GPU configurations
- Optimize pipeline parameters for specific use cases

### Production Deployment
- High-throughput 3D model generation
- Quality assurance and ranking systems
- Load balancing across GPU clusters

### Evaluation and Benchmarking
- Model performance comparison
- GPU utilization optimization
- Pipeline efficiency analysis

## 🤝 Contributing

To extend the pipeline system:

1. **Add new pipeline types**: Inherit from `MultiGPUPipelineManager`
2. **Custom scoring methods**: Override CLIP or validation logic
3. **Additional analysis**: Extend result structures and metrics
4. **Performance improvements**: Optimize parallelization strategies

## 📄 License

This project is part of Subnet 17 (404-GEN) and follows the same licensing terms.

---

**🎉 Happy Pipeline Processing!**

For questions and support, refer to the test scripts and example usage patterns provided.
