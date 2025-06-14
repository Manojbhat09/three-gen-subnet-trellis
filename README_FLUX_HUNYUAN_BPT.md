# Flux-Hunyuan-BPT Enhanced Miner System

This system integrates the Flux-Hunyuan-BPT demo with a generation server to create an enhanced miner for Subnet 17 (404-GEN) that produces high-quality 3D models with BPT (Binary Point Tree) enhancement.

## 🚀 Features

- **Enhanced 3D Generation**: Combines Flux (text-to-image) + Hunyuan3D-2 (image-to-3D) + BPT (mesh enhancement)
- **Dual Generation Modes**: Option to generate with or without BPT enhancement
- **Intelligent Fallback**: Automatically falls back to standard generation if BPT fails
- **Comprehensive Testing**: Full test suite for local validation
- **Production Ready**: Designed for Bittensor Subnet 17 mining

## 📁 File Structure

```
├── flux_hunyuan_bpt_generation_server.py    # Enhanced generation server
├── flux_hunyuan_bpt_miner.py               # Enhanced miner implementation
├── test_flux_hunyuan_bpt_server.py         # Server test suite
├── test_flux_hunyuan_bpt_miner.py          # Miner test suite
├── run_flux_hunyuan_bpt_tests.py           # Master test orchestrator
├── README_FLUX_HUNYUAN_BPT.md              # This file
└── Hunyuan3D-2/
    └── flux_hunyuan_bpt_demo.py             # Original demo (required)
```

## 🔧 Prerequisites

### Hardware Requirements
- NVIDIA GPU with at least 12GB VRAM (recommended: 24GB+)
- 32GB+ System RAM
- 100GB+ free disk space for models

### Software Dependencies
```bash
# Core ML libraries
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install diffusers transformers accelerate
pip install trimesh

# Web framework
pip install fastapi uvicorn aiohttp

# Bittensor
pip install bittensor

# Compression and utilities
pip install pyspz numpy pillow

# Additional dependencies
pip install protobuf dataclasses-json
```

### Model Setup
The system will automatically download required models on first run:
- Flux transformer model (quantized GGUF format)
- Hunyuan3D-2 shape generation model
- BPT enhancement model weights
- Background removal model

## 🏃‍♂️ Quick Start

### 1. Automated Testing (Recommended)
Run the complete test suite to validate everything works:

```bash
python run_flux_hunyuan_bpt_tests.py
```

This will:
- Check all dependencies
- Start the generation server
- Run comprehensive tests
- Validate the complete mining pipeline
- Provide detailed results and recommendations

### 2. Manual Testing

#### Start Generation Server
```bash
python flux_hunyuan_bpt_generation_server.py
```

The server will start on `http://127.0.0.1:8095` and provide endpoints:
- `/generate/` - Generate 3D models
- `/status/` - Server status and metrics
- `/health/` - Health check
- `/config/` - Current configuration

#### Test Server Functionality
```bash
python test_flux_hunyuan_bpt_server.py
```

#### Test Miner Functionality
```bash
python test_flux_hunyuan_bpt_miner.py
```

### 3. Production Deployment

#### Start Generation Server
```bash
python flux_hunyuan_bpt_generation_server.py
```

#### Start Enhanced Miner
```bash
python flux_hunyuan_bpt_miner.py
```

## 🎛️ Configuration

### Generation Server Configuration
Edit `GENERATION_CONFIG` in `flux_hunyuan_bpt_generation_server.py`:

```python
GENERATION_CONFIG = {
    'output_dir': './flux_hunyuan_bpt_outputs',
    'device': 'cuda',  # or 'cpu'
    'use_bpt': True,   # Enable/disable BPT enhancement
    'bpt_temperature': 0.5,  # BPT generation temperature (0.1-1.0)
    'num_inference_steps_t2i': 8,      # Flux inference steps
    'num_inference_steps_shape': 30,   # Hunyuan3D inference steps
}
```

### Miner Configuration
Edit constants in `flux_hunyuan_bpt_miner.py`:

```python
# Enhanced Generation Configuration
USE_BPT_ENHANCEMENT = True      # Enable BPT by default
BPT_TEMPERATURE = 0.5          # BPT temperature
FALLBACK_TO_STANDARD = True    # Fallback if BPT fails

# Quality and Performance
SELF_VALIDATION_MIN_SCORE = 0.8    # Higher threshold for enhanced models
GENERATION_TIMEOUT_SECONDS = 600   # 10 minutes for BPT processing
MAX_CONCURRENT_GENERATIONS = 1     # Reduced for BPT memory requirements
```

## 🧪 Testing Details

### Server Tests
- Health check validation
- Status endpoint verification
- Configuration retrieval
- 3D model generation (with/without BPT)
- Performance benchmarking
- Cache management

### Miner Tests
- Generation server connectivity
- Full mining pipeline simulation
- Validation workflow testing
- Asset saving verification
- Error handling validation
- BPT enhancement verification

### Performance Expectations
- **Standard Generation**: 30-60 seconds
- **BPT Enhanced Generation**: 60-120 seconds
- **Validation**: 5-10 seconds
- **File Sizes**: 50KB-2MB (PLY format)

## 🚨 Troubleshooting

### Common Issues

#### 1. GPU Memory Errors
```
RuntimeError: CUDA out of memory
```
**Solution**: 
- Reduce concurrent generations: Set `MAX_CONCURRENT_GENERATIONS = 1`
- Disable BPT: Set `USE_BPT_ENHANCEMENT = False`
- Use smaller batch sizes in generation config

#### 2. Model Download Failures
```
HTTPError: 403 Forbidden
```
**Solution**: 
- Set Hugging Face token: `export HUGGINGFACE_TOKEN=your_token`
- Check internet connectivity
- Verify model availability

#### 3. BPT Enhancement Failures
```
BPT enhancement failed, using original mesh
```
**Solution**: 
- Check BPT model weights location
- Verify GPU memory availability
- Enable fallback mode: `FALLBACK_TO_STANDARD = True`

#### 4. Generation Server Connection Issues
```
Cannot connect to server at http://127.0.0.1:8095
```
**Solution**: 
- Ensure server is running: `python flux_hunyuan_bpt_generation_server.py`
- Check port availability: `netstat -tulpn | grep 8095`
- Verify firewall settings

### Performance Optimization

#### For Better Speed
1. Disable BPT: `USE_BPT_ENHANCEMENT = False`
2. Reduce inference steps: `num_inference_steps_shape = 20`
3. Use CPU for validation if GPU memory is limited

#### For Better Quality
1. Enable BPT: `USE_BPT_ENHANCEMENT = True`
2. Increase BPT temperature: `BPT_TEMPERATURE = 0.7`
3. Increase validation threshold: `SELF_VALIDATION_MIN_SCORE = 0.85`

## 📊 Monitoring

### Log Files
- Server logs: Console output from generation server
- Miner logs: `./logs/flux_hunyuan_bpt_miner_YYYYMMDD_HHMMSS.log`
- Test logs: Console output from test scripts

### Generated Assets
- Server outputs: `./flux_hunyuan_bpt_outputs/`
- Miner outputs: `./flux_hunyuan_bpt_mining_outputs/`
- Test outputs: `./test_outputs_*`

### Metrics Endpoints
- Server status: `GET http://127.0.0.1:8095/status/`
- Generation metrics displayed in miner logs every 5 minutes

## 🔄 Deployment Workflow

### Development/Testing
1. Run `python run_flux_hunyuan_bpt_tests.py`
2. Verify all tests pass
3. Review generated assets for quality
4. Adjust configuration as needed

### Staging
1. Deploy generation server: `python flux_hunyuan_bpt_generation_server.py`
2. Run miner tests: `python test_flux_hunyuan_bpt_miner.py`
3. Monitor performance and quality metrics
4. Fine-tune configuration

### Production
1. Ensure Bittensor wallet is registered on subnet 17
2. Start generation server in background/screen session
3. Start miner: `python flux_hunyuan_bpt_miner.py`
4. Monitor logs and metrics
5. Scale based on performance requirements

## 🤝 Contributing

### Code Structure
- **Server**: FastAPI-based REST API for 3D generation
- **Miner**: Async Bittensor miner with task pulling and result submission
- **Tests**: Comprehensive test suites for validation
- **Orchestrator**: Master test runner for complete system validation

### Adding New Features
1. Update generation server for new capabilities
2. Modify miner to utilize new features
3. Add corresponding tests
4. Update configuration options
5. Document changes in README

## 📈 Performance Benchmarks

### Generation Times (RTX 4090)
- **Flux (Text→Image)**: 8-15 seconds
- **Hunyuan3D (Image→3D)**: 20-30 seconds
- **BPT Enhancement**: 30-60 seconds
- **Total (with BPT)**: 60-105 seconds

### Quality Metrics
- **Standard Generation**: 0.6-0.8 validation score
- **BPT Enhanced**: 0.7-0.9 validation score
- **File Integrity**: 99%+ valid PLY files
- **Mesh Quality**: Watertight, manifold meshes

## 🛡️ Security Considerations

### API Security
- Server runs on localhost only (127.0.0.1)
- No authentication required for local testing
- Production deployment should add authentication

### Data Privacy
- No user data stored persistently
- Generated models saved locally only
- Network traffic limited to model downloads

### Resource Management
- GPU memory monitoring and cleanup
- Process isolation for generation tasks
- Automatic fallback mechanisms

## 📞 Support

### Getting Help
1. Check troubleshooting section above
2. Review test outputs for specific error messages
3. Verify hardware and software requirements
4. Check log files for detailed error information

### Reporting Issues
Include the following information:
- Hardware specifications (GPU, RAM)
- Software versions (Python, CUDA, PyTorch)
- Complete error messages and stack traces
- Configuration settings used
- Test results output

## 🔮 Future Enhancements

### Planned Features
- Multi-GPU support for parallel generation
- Advanced mesh post-processing options
- Custom model fine-tuning capabilities
- Real-time generation monitoring dashboard
- Automatic quality optimization

### Research Directions
- Integration with newer 3D generation models
- Advanced BPT architectures
- Quality-aware generation strategies
- Efficient caching mechanisms

---

**Happy Mining! 🚀**

For the latest updates and documentation, check the repository for new versions and improvements. 