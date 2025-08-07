# LoRA Integration for 3D Generation Servers

This document describes the LoRA (Low-Rank Adaptation) integration for the FLUX and SDXL 3D generation servers.

## Overview

The LoRA system allows you to apply different artistic styles and enhancements to your 3D generations by loading pre-trained LoRA weights onto the base models. Each LoRA has specific trigger words that activate the style when included in the prompt.

## Available LoRAs

### FLUX LoRAs

| LoRA Key | Name | Trigger Prefix | Description |
|----------|------|----------------|-------------|
| `isometric_3d` | Flux Isometric 3D | `Isometric 3D,` | Isometric 3D style LoRA for FLUX |
| `live_3d` | FLUX Live 3D | (none) | Live 3D style LoRA for FLUX |
| `game_assets` | 3D Game Assets | `Create 3D game asset, isometric view version,` | 3D game assets style LoRA for FLUX |
| `patched_realism` | Patched Realism | (none) | Realism enhancement LoRA for FLUX |
| `tf2_style` | Team Fortress 2 Style | `tf2style,` | Team Fortress 2 style LoRA for FLUX |
| `baolei` | Baolei Style | `Cartoon-style design,` | Baolei cartoon style LoRA for FLUX |
| `cartoon_3d` | Cartoon 3D Render | (none) | Cartoon 3D render style LoRA for FLUX |
| `cinema` | Cinema Style | `c1n3ma,` | Cinema style LoRA for FLUX |

### SDXL LoRAs

| LoRA Key | Name | Trigger Prefix | Description |
|----------|------|----------------|-------------|
| `game_icon` | Game Icon Institute | `game icon institute,` | Game icon style LoRA for SDXL |

## Server Endpoints

### FLUX Server (Port 8096)

#### LoRA Management
- `GET /loras/` - Get list of available LoRAs
- `POST /loras/load/{lora_key}` - Load a specific LoRA
- `POST /loras/unload/` - Unload current LoRA

#### LoRA-Specific Generation Endpoints
- `POST /generate/isometric_3d/` - Generate with Isometric 3D LoRA
- `POST /generate/live_3d/` - Generate with Live 3D LoRA
- `POST /generate/game_assets/` - Generate with Game Assets LoRA
- `POST /generate/patched_realism/` - Generate with Patched Realism LoRA
- `POST /generate/tf2_style/` - Generate with TF2 Style LoRA
- `POST /generate/baolei/` - Generate with Baolei Style LoRA
- `POST /generate/cartoon_3d/` - Generate with Cartoon 3D Render LoRA
- `POST /generate/cinema/` - Generate with Cinema Style LoRA

### SDXL Server (Port 8097)

#### LoRA Management
- `GET /loras/` - Get list of available LoRAs
- `POST /loras/load/{lora_key}` - Load a specific LoRA
- `POST /loras/unload/` - Unload current LoRA

#### LoRA-Specific Generation Endpoints
- `POST /generate/game_icon/` - Generate with Game Icon LoRA

## Usage Examples

### Using LoRA-Specific Endpoints

```bash
# Generate with Isometric 3D LoRA
curl -X POST "http://localhost:8096/generate/isometric_3d/" \
  -F "prompt=a blue ceramic vase with red trim" \
  -F "seed=42" \
  -F "return_compressed=true" \
  -o isometric_3d_vase.ply.spz

# Generate with Game Icon LoRA (SDXL)
curl -X POST "http://localhost:8097/generate/game_icon/" \
  -F "prompt=a blue ceramic vase with red trim" \
  -F "seed=42" \
  -F "return_compressed=true" \
  -o game_icon_vase.ply.spz
```

### Using LoRA Management

```bash
# Get available LoRAs
curl "http://localhost:8096/loras/"

# Load a specific LoRA
curl -X POST "http://localhost:8096/loras/load/isometric_3d"

# Generate with loaded LoRA (uses standard endpoint)
curl -X POST "http://localhost:8096/generate/" \
  -F "prompt=Isometric 3D, a blue ceramic vase with red trim" \
  -F "seed=42" \
  -F "return_compressed=true" \
  -o vase_with_lora.ply.spz

# Unload LoRA
curl -X POST "http://localhost:8096/loras/unload/"
```

## Testing and Validation

### Running the Test Suite

The `lora_test_wrapper.py` script provides comprehensive testing of all LoRAs against validation prompts:

```bash
# Test all LoRAs
python lora_test_wrapper.py

# Test specific LoRAs
python lora_test_wrapper.py --loras isometric_3d live_3d game_icon

# Test only FLUX LoRAs
python lora_test_wrapper.py --test-flux --no-test-sdxl

# Test only SDXL LoRAs
python lora_test_wrapper.py --test-sdxl --no-test-flux
```

### Demo Script

The `lora_demo.py` script demonstrates basic usage:

```bash
python lora_demo.py
```

## Test Prompts

The system includes these test prompts for validation:

1. "greek amphora scene detail"
2. "plastic straw of drink"
3. "small yellow triangular wooden kitchen knife"
4. "enormous black robot with round body"
5. "rose gold locket necklace with floral"

## LoRA File Locations

### FLUX LoRAs
- Local LoRAs are expected in `/home/mbhat/three-gen-subnet-trellis/LORAS/`
- HuggingFace LoRAs are downloaded automatically

### SDXL LoRAs
- Local LoRAs are expected in `/home/mbhat/three-gen-subnet-trellis/LORAS/`

## LoRA Patching

Some LoRAs may require patching for compatibility. The system automatically applies the `patcher.py` script when needed to fix adaLN layer issues.

## Configuration

LoRA settings can be configured in the server configuration:

```python
GENERATION_CONFIG = {
    # ... other settings ...
    'current_lora': None,  # Currently loaded LoRA
    'lora_scale': 1.0,     # LoRA strength scale
}
```

## Output Files

Generated files are saved with descriptive names including the LoRA type:

- `isometric_3d_greek_amphora_scene_detail_42.ply.spz`
- `game_icon_plastic_straw_of_drink_42.ply.spz`
- `tf2style_enormous_black_robot_with_round_body_42.ply.spz`

## Troubleshooting

### Common Issues

1. **LoRA loading fails**: Check if the LoRA file exists in the expected location
2. **Generation fails**: Ensure the server is running and models are loaded
3. **Validation fails**: Check if the validation server is running on port 10006

### Error Messages

- `LoRA file not found`: The LoRA file doesn't exist in the expected path
- `Failed to load LoRA`: The LoRA file is corrupted or incompatible
- `Generation failed`: The generation pipeline encountered an error

### Debugging

Enable verbose logging by checking the server logs for detailed error messages. The LoRA loading process includes automatic patching for compatibility issues.

## Performance Notes

- LoRA loading takes a few seconds but only needs to be done once per session
- Generation time may vary depending on the LoRA complexity
- Memory usage increases slightly when LoRAs are loaded
- Unloading LoRAs frees up memory for other operations

## Future Enhancements

- Support for LoRA mixing and blending
- Dynamic LoRA strength adjustment
- Batch processing with multiple LoRAs
- LoRA performance metrics and optimization 