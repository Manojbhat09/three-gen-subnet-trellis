# FLUX Socket Endpoints Documentation

This document describes the new FLUX socket-based inference endpoints that integrate with the `newcomer20_accurate` FLUX server.

## Overview

The FLUX socket endpoints provide fast, efficient image generation using the `newcomer20_accurate` FLUX server, which communicates via Unix socket instead of loading models directly into the main server process.

## Server Setup

### 1. Start FLUX Socket Server

Before using the endpoints, you need to start the FLUX socket server:

```bash
# Option 1: Use the provided script
./start_flux_server.sh

# Option 2: Manual start
cd newcomer20_accurate
uv run python src/main.py

# Option 3: Use the API endpoint
curl -X POST "http://localhost:8096/start_flux_server/"
```

The server will create a Unix socket at `newcomer20_accurate/inferences.sock`.

### 2. Check Server Status

```bash
curl "http://localhost:8096/flux_server_status/"
```

## Available Endpoints

### 1. Image Generation Only

**Endpoint:** `POST /generate_flux_socket/`

Generate images using FLUX socket server without 3D generation.

**Parameters:**
- `prompt` (required): Text prompt for image generation
- `seed` (optional): Random seed for reproducibility (default: random)
- `width` (optional): Image width (default: 1024)
- `height` (optional): Image height (default: 1024)

**Example:**
```bash
curl -X POST "http://localhost:8096/generate_flux_socket/" \
  -F "prompt=a beautiful sunset over mountains" \
  -F "seed=42" \
  -F "width=1024" \
  -F "height=1024"
```

**Response:**
```json
{
  "status": "success",
  "prompt": "a beautiful sunset over mountains",
  "seed": 42,
  "width": 1024,
  "height": 1024,
  "image": "base64_encoded_image_data",
  "image_size_bytes": 123456,
  "pipeline": "flux_socket",
  "server": "newcomer20_accurate"
}
```

### 2. 3D Model Generation

**Endpoint:** `POST /generate_flux_socket_3d/`

Generate 3D models using FLUX socket + TRELLIS pipeline.

**Parameters:**
- `prompt` (required): Text prompt for generation
- `seed` (optional): Random seed (default: 42)
- `return_compressed` (optional): Return compressed PLY (default: true)
- `width` (optional): Image width (default: 1024)
- `height` (optional): Image height (default: 1024)
- `ss_sampling_steps` (optional): Structure sampling steps
- `slat_sampling_steps` (optional): SLAT sampling steps
- `slat_guidance_strength` (optional): SLAT guidance strength
- `ss_guidance_strength` (optional): Structure guidance strength

**Example:**
```bash
curl -X POST "http://localhost:8096/generate_flux_socket_3d/" \
  -F "prompt=a detailed 3D model of a medieval castle" \
  -F "seed=42" \
  -F "return_compressed=true"
```

### 3. 3D Model Generation with LoRA

**Endpoint:** `POST /generate_flux_socket_3d_lora/{lora_key}`

Generate 3D models using FLUX socket + TRELLIS with specific LoRA styles.

**Available LoRA Keys:**
- `isometric_3d`: Isometric 3D style
- `live_3d`: Live 3D style
- `game_assets`: 3D game assets style
- `patched_realism`: Realism enhancement
- `tf2_style`: Team Fortress 2 style
- `baolei`: Baolei cartoon style
- `cartoon_3d`: Cartoon 3D render style
- `cinema`: Cinema style
- `necklace`: Necklace style

**Parameters:** Same as 3D generation endpoint

**Example:**
```bash
curl -X POST "http://localhost:8096/generate_flux_socket_3d_lora/isometric_3d/" \
  -F "prompt=a futuristic spaceship" \
  -F "seed=42"
```

### 4. Server Management

**Start Server:** `POST /start_flux_server/`
**Check Status:** `GET /flux_server_status/`
**Test Connection:** `POST /test_flux_socket/`

## Configuration

The FLUX socket integration is enabled by default in the configuration:

```python
GENERATION_CONFIG = {
    # ... other config ...
    'flux_use_schnell_socket': True,  # Enable socket-based FLUX
    'flux_use_schnell': True,         # Enable schnell mode
    'flux_schnell_steps': 4,          # 4-step inference
    'flux_schnell_guidance': 0.0,     # No guidance for schnell
}
```

## Socket Path

The default socket path is:
```
/home/mbhat/three-gen-subnet-trellis/newcomer20_accurate/inferences.sock
```

## Error Handling

The endpoints include comprehensive error handling:

- **Connection failures**: Automatic retry and fallback
- **Generation failures**: Detailed error messages
- **Server status**: Health checks and status reporting

## Performance Benefits

- **Memory efficiency**: FLUX models run in separate process
- **Fast inference**: 4-step schnell mode
- **Scalability**: Socket-based communication allows multiple clients
- **Stability**: Isolated process prevents crashes from affecting main server

## Troubleshooting

### Common Issues

1. **Socket not found**: Start the FLUX server first
2. **Connection refused**: Check if server is running
3. **Permission denied**: Ensure socket file has correct permissions

### Debug Commands

```bash
# Check if socket exists
ls -la newcomer20_accurate/inferences.sock

# Check server process
ps aux | grep "python src/main.py"

# Test socket connection
curl -X POST "http://localhost:8096/test_flux_socket/"
```

## Integration with Existing Pipeline

The FLUX socket endpoints integrate seamlessly with the existing TRELLIS pipeline:

1. **Image Generation**: FLUX socket server generates high-quality images
2. **Background Removal**: Automatic background removal for better 3D generation
3. **3D Generation**: TRELLIS converts images to Gaussian Splatting models
4. **Compression**: Optional SPZ compression for efficient storage

This provides a complete pipeline from text prompt to compressed 3D model using the fastest available FLUX inference.
