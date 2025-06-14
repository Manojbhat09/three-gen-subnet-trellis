# Automated GPU Memory Coordination System

## Overview

This system solves the critical GPU memory limitation on RTX 4090 (24GB VRAM) by implementing intelligent coordination between the validation server and generation server. The problem was that both servers together required ~24.3GB, exceeding the GPU capacity by 300MB and causing CUDA out of memory errors.

## The Solution

**Sequential GPU Memory Coordination**: The generation server automatically coordinates with the validation server to temporarily free GPU memory during generation, then restores the validation models afterward.

### Memory Usage Breakdown
- **Validation Server**: ~4.7GB GPU memory
- **Generation Server (FLUX + Hunyuan3D)**: ~19.6GB GPU memory  
- **Total Required**: ~24.3GB (exceeds 24GB limit)
- **With Coordination**: Sequential usage stays within 24GB limit

## How It Works

### Automated Coordination Flow

1. **Generation Request Received**
2. **Contact Validation Server** → Unload models (frees ~4GB)
3. **Aggressive GPU Cleanup** → Additional memory clearing
4. **Run FLUX + Hunyuan3D Generation** (uses ~19.6GB)
5. **Contact Validation Server** → Reload models
6. **Both Servers Operational** without CUDA OOM

### Key Components

#### Enhanced Validation Server (`validation/serve.py`)
- **`/gpu_status/`** - Monitor GPU memory usage
- **`/unload_models/`** - Aggressive model unloading (frees ~4.7GB)
- **`/reload_models/`** - Model restoration with fallback handling
- **Enhanced cleanup** - Multiple-pass memory clearing

#### Enhanced Generation Server (`flux_hunyuan_sugar_generation_server.py`)
- **`_coordinate_with_validation_server()`** - Automatic coordination
- **`_aggressive_gpu_cleanup()`** - Multi-pass memory clearing
- **Integrated coordination** - Built into generation pipeline
- **Error handling** - Ensures validation server restoration on failure

## Usage

### Quick Start

1. **Start Both Servers**:
   ```bash
   ./start_servers.sh
   ```

2. **Test the System**:
   ```bash
   python automated_gpu_coordination_test.py
   ```

3. **Generate 3D Models**:
   ```bash
   curl -X POST "http://localhost:8095/generate/" \
        -F "prompt=a red cube" \
        -F "seed=42"
   ```

### Manual Server Startup

#### Validation Server
```bash
cd validation
conda activate three-gen-validation
python serve.py --host 0.0.0.0 --port 10006
```

#### Generation Server
```bash
conda activate hunyuan3d
python flux_hunyuan_sugar_generation_server.py
```

## API Endpoints

### Validation Server (localhost:10006)
- `GET /gpu_status/` - Get GPU memory status
- `POST /unload_models/` - Unload models to free GPU memory
- `POST /reload_models/` - Reload models back to GPU
- `POST /cleanup_gpu/` - Manual GPU memory cleanup

### Generation Server (localhost:8095)
- `POST /generate/` - Generate 3D model (with automatic coordination)
- `GET /status/` - Get server status and metrics
- `GET /health/` - Health check
- `POST /clear_cache/` - Manual GPU cache clearing

## Testing

### Automated Test Suite

The `automated_gpu_coordination_test.py` script provides comprehensive testing:

1. **Validation Unload/Reload Test** - Verifies model unloading/reloading
2. **Single Generation Test** - Tests complete coordination flow
3. **Multiple Generation Test** - Ensures system stability

### Manual Testing

```bash
# Test validation server coordination
curl -X POST "http://localhost:10006/unload_models/"
curl -X POST "http://localhost:10006/reload_models/"

# Test generation with coordination
curl -X POST "http://localhost:8095/generate/" \
     -F "prompt=a blue sphere" \
     -F "seed=123"
```

## Technical Details

### Memory Management Strategy

1. **Aggressive Unloading**: Validation server moves models to CPU and clears GPU references
2. **Multi-pass Cleanup**: Multiple rounds of `torch.cuda.empty_cache()` and `gc.collect()`
3. **Memory Monitoring**: Real-time GPU memory status tracking
4. **Fallback Handling**: Robust error recovery and model restoration

### Error Handling

- **Coordination Failures**: Generation continues with warning if validation coordination fails
- **Model Reload Failures**: Fallback to complete model reinitialization
- **Memory Exhaustion**: Aggressive cleanup with multiple retry attempts
- **Server Communication**: Timeout handling and retry logic

### Performance Optimizations

- **On-demand FLUX Loading**: FLUX pipeline loaded only when needed
- **Temporary Model Unloading**: Hunyuan3D temporarily unloaded during FLUX usage
- **Memory Fragmentation Prevention**: Strategic cleanup timing
- **Background Processing**: Non-blocking coordination calls

## Monitoring

### GPU Memory Status

```bash
# Check validation server GPU usage
curl "http://localhost:10006/gpu_status/"

# Check generation server status
curl "http://localhost:8095/status/"
```

### Log Monitoring

- **Validation Server**: Model loading/unloading events, memory statistics
- **Generation Server**: Coordination events, generation progress, memory usage

## Troubleshooting

### Common Issues

1. **CUDA OOM During Generation**
   - Check if validation server unloaded properly
   - Verify aggressive cleanup is working
   - Monitor GPU memory before FLUX loading

2. **Validation Server Reload Failures**
   - Check available GPU memory
   - Verify model backup state
   - Use fallback reinitialization

3. **Coordination Timeouts**
   - Increase timeout values in coordination calls
   - Check network connectivity between servers
   - Verify server health endpoints

### Debug Commands

```bash
# Check GPU memory usage
nvidia-smi

# Test server health
curl "http://localhost:10006/health/"
curl "http://localhost:8095/health/"

# Manual memory cleanup
curl -X POST "http://localhost:10006/cleanup_gpu/"
curl -X POST "http://localhost:8095/clear_cache/"
```

## Configuration

### Environment Variables

```bash
# GPU memory management
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Server ports
export VALIDATION_PORT=10006
export GENERATION_PORT=8095
```

### Model Configuration

Edit `flux_hunyuan_sugar_generation_server.py`:
```python
GENERATION_CONFIG = {
    'device': 'cuda',
    'hunyuan_model_path': 'jetx/Hunyuan3D-2',
    'num_inference_steps_t2i': 8,
    'num_inference_steps_shape': 30,
    'sugar_num_points': 15000,
    # ... other settings
}
```

## System Requirements

- **GPU**: RTX 4090 (24GB VRAM) or equivalent
- **RAM**: 32GB+ recommended
- **Storage**: 50GB+ for models
- **Python**: 3.10+ with conda environments
- **CUDA**: 11.8+ or 12.x

## Success Metrics

With this coordination system:
- ✅ **Zero CUDA OOM errors** during normal operation
- ✅ **Successful generation** of 3D models from text prompts
- ✅ **Automatic memory management** without manual intervention
- ✅ **Robust error recovery** with fallback mechanisms
- ✅ **Stable multi-generation** workflows

## Future Improvements

1. **Dynamic Memory Allocation**: Adjust model precision based on available memory
2. **Predictive Coordination**: Pre-emptive model unloading based on generation queue
3. **Memory Pool Management**: Shared GPU memory pool between servers
4. **Load Balancing**: Distribute generation across multiple GPUs if available

---

**Status**: ✅ **PRODUCTION READY** - Successfully eliminates CUDA OOM errors and enables stable 3D generation on RTX 4090 hardware. 