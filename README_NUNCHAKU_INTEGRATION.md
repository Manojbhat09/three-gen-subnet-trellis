# Nunchaku Integration with TRELLIS Server

This project integrates the **Nunchaku** image generation model with the main **TRELLIS** server, enabling high-quality image generation through a clean HTTP API architecture.

## 🏗️ Architecture Overview

The system uses a **two-server architecture** to handle different conda environments:

```
┌─────────────────────────────────────────────────────────────┐
│                    Main TRELLIS Server                     │
│                    (trellis_new env)                       │
│  ┌─────────────────────────────────────────────────────┐   │
│  │  • FLUX Pipeline                                   │   │
│  │  • SDXL Pipeline                                   │   │
│  │  • SD1.5 Pipeline                                  │   │
│  │  • Nunchaku HTTP Client                            │   │
│  └─────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
                              │
                              │ HTTP Requests
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                 Nunchaku API Server                        │
│                    (nun env)                              │
│  ┌─────────────────────────────────────────────────────┐   │
│  │  • Nunchaku Pipeline                               │   │
│  │  • Flux Transformer                                │   │
│  │  • Image Generation                                │   │
│  │  • Base64 Encoding                                 │   │
│  └─────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
```

## 🚀 Quick Start

### 1. Environment Setup

Run the setup script to create the Nunchaku environment:

```bash
chmod +x setup_nunchaku_env.sh
./setup_nunchaku_env.sh
```

### 2. Start the Nunchaku API Server

```bash
conda activate nun
python nunchaku_api_server.py
```

The server will start on port 8200 and load the Nunchaku pipeline.

### 3. Start the Main TRELLIS Server

```bash
conda activate trellis_new
python trellis_subnit_server_mix_lora_flash_nun.py --port 8096
```

### 4. Test the Integration

```bash
# Switch to Nunchaku model
curl -X POST "http://localhost:8096/config/model/" -F "model=nunchaku_flux"

# Generate an image
curl -X POST "http://localhost:8096/generate_image/" -F "prompt=A beautiful sunset" -F "seed=42"
```

## 📁 File Structure

```
├── nunchaku_api_server.py          # Nunchaku API server (runs in nun env)
├── nunchaku_client.py              # HTTP client for main server
├── setup_nunchaku_env.sh           # Environment setup script
├── test_nunchaku_api.py            # API testing script
├── trellis_subnit_server_mix_lora_flash_nun.py  # Main server with Nunchaku integration
└── README_NUNCHAKU_INTEGRATION.md  # This file
```

## 🔧 Technical Details

### Nunchaku API Server (`nunchaku_api_server.py`)

- **Port**: 8200
- **Framework**: Flask
- **Endpoints**:
  - `GET /health` - Health check
  - `POST /generate` - Image generation
- **Data Formats**: Accepts both JSON and form data
- **Output**: Base64-encoded PNG images

### Main Server Integration

- **Model Type**: `nunchaku_flux`
- **Communication**: HTTP client to Nunchaku API
- **Fallback**: Graceful degradation if Nunchaku unavailable
- **Error Handling**: Comprehensive error reporting

### Image Generation Parameters

- **Default Size**: 1024x1024
- **Guidance Scale**: 0.0 (Nunchaku default)
- **Inference Steps**: 4 (optimized for speed)
- **Max Sequence Length**: 256

## 🐍 Conda Environment Requirements

### Main Server (`trellis_new`)
- Python 3.11+
- PyTorch with CUDA support
- FastAPI, PIL, requests
- TRELLIS dependencies

### Nunchaku Server (`nun`)
- Python 3.11
- PyTorch 2.7+ with CUDA 12.8
- Nunchaku 0.3.1
- Diffusers, Transformers, Accelerate

## 🔍 Troubleshooting

### Common Issues

1. **Port Already in Use**
   ```bash
   # Check what's using the port
   lsof -i :8200
   lsof -i :8096
   ```

2. **Nunchaku Import Errors**
   ```bash
   # Ensure you're in the right environment
   conda activate nun
   python -c "import nunchaku; print('✅ Nunchaku imported successfully')"
   ```

3. **GPU Memory Issues**
   ```bash
   # Clear GPU cache
   python -c "import torch; torch.cuda.empty_cache()"
   ```

4. **Connection Refused**
   ```bash
   # Check if Nunchaku server is running
   curl http://localhost:8200/health
   ```

### Debug Mode

Enable debug logging in the Nunchaku API server by setting:
```python
app.run(host='0.0.0.0', port=8200, debug=True)
```

## 📊 Performance

- **Model Loading**: ~30 seconds (first time)
- **Image Generation**: ~2-5 seconds per image
- **Memory Usage**: ~8-12GB GPU VRAM
- **Concurrent Requests**: Single-threaded (Flask default)

## 🔄 API Endpoints

### Main Server (Port 8096)

```
POST /config/model/          # Switch between models
POST /generate_image/        # Generate image with current model
POST /generate_3d_model/     # Generate 3D model from image
```

### Nunchaku Server (Port 8200)

```
GET  /health                 # Health check
POST /generate               # Generate image with Nunchaku
```

## 🧪 Testing

### Test Nunchaku API Directly

```bash
python test_nunchaku_api.py
```

### Test Main Server Integration

```bash
# Test health endpoint
curl http://localhost:8096/health

# Test model switching
curl -X POST "http://localhost:8096/config/model/" -F "model=nunchaku_flux"

# Test image generation
curl -X POST "http://localhost:8096/generate_image/" -F "prompt=Test prompt" -F "seed=42"
```

## 🚨 Important Notes

1. **Environment Isolation**: Never mix conda environments
2. **GPU Memory**: Ensure sufficient VRAM (8GB+ recommended)
3. **Port Conflicts**: Check for port conflicts before starting servers
4. **Model Loading**: First startup takes longer due to model loading
5. **Error Handling**: Always check server logs for detailed error information

## 🔮 Future Enhancements

- [ ] Multi-GPU support
- [ ] Batch processing
- [ ] Model caching
- [ ] Load balancing
- [ ] Authentication
- [ ] Rate limiting
- [ ] WebSocket support for real-time updates

## 📞 Support

For issues or questions:
1. Check the troubleshooting section
2. Review server logs for error details
3. Ensure both environments are properly set up
4. Verify GPU compatibility and memory availability

---

**Last Updated**: August 2024  
**Version**: 1.0.0  
**Compatibility**: CUDA 12.8+, PyTorch 2.7+, Python 3.11
