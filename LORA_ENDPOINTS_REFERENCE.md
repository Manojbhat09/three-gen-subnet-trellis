# LoRA Endpoints Reference

This document lists all available LoRA endpoints in the TRELLIS server for use with the CLIP alignment analysis script.

## Available LoRA Endpoints

Based on the `trellis_subnit_server_mix_lora_flash.py` server, the following LoRA endpoints are available:

### 1. **isometric_3d**
- **Purpose**: Isometric 3D style generation
- **Best for**: Technical diagrams, architectural visualizations, game assets with isometric perspective
- **Usage**: `--lora1 "isometric_3d"` or `--lora2 "isometric_3d"`

### 2. **live_3d**
- **Purpose**: Live-action 3D style generation
- **Best for**: Realistic 3D objects, photorealistic renders
- **Usage**: `--lora1 "live_3d"` or `--lora2 "live_3d"`

### 3. **game_assets**
- **Purpose**: Game asset style generation
- **Best for**: Video game objects, stylized 3D models
- **Usage**: `--lora1 "game_assets"` or `--lora2 "game_assets"`

### 4. **patched_realism**
- **Purpose**: Patched realism style generation
- **Best for**: Realistic objects with enhanced details
- **Usage**: `--lora1 "patched_realism"` or `--lora2 "patched_realism"`

### 5. **tf2_style**
- **Purpose**: Team Fortress 2 style generation
- **Best for**: TF2-inspired objects, cartoon-style 3D models
- **Usage**: `--lora1 "tf2_style"` or `--lora2 "tf2_style"`

### 6. **baolei**
- **Purpose**: Baolei style generation
- **Best for**: Specific artistic style objects
- **Usage**: `--lora1 "baolei"` or `--lora2 "baolei"`

### 7. **cartoon_3d**
- **Purpose**: Cartoon 3D style generation
- **Best for**: Animated/cartoon-style 3D objects
- **Usage**: `--lora1 "cartoon_3d"` or `--lora2 "cartoon_3d"`

### 8. **cinema**
- **Purpose**: Cinematic style generation
- **Best for**: Movie-quality 3D objects, cinematic renders
- **Usage**: `--lora1 "cinema"` or `--lora2 "cinema"`

### 9. **sd15_game_icon**
- **Purpose**: SD1.5 game icon style generation
- **Best for**: Game icons, small detailed objects
- **Usage**: `--lora1 "sd15_game_icon"` or `--lora2 "sd15_game_icon"`

## Usage Examples

### Single LoRA Analysis
```bash
# Analyze with isometric 3D style
python clip_alignment_with_generation.py --lora1 "isometric_3d" "a blue vase"

# Analyze with live 3D style
python clip_alignment_with_generation.py --lora1 "live_3d" "a blue vase"

# Analyze with game assets style
python clip_alignment_with_generation.py --lora1 "game_assets" "a blue vase"
```

### Prefix Analysis with LoRA
```bash
# Add prefix with isometric 3D style
python clip_alignment_with_generation.py --prefix "professional 3D render, " --lora1 "isometric_3d" "a blue vase"

# Add prefix with live 3D style
python clip_alignment_with_generation.py --prefix "highly detailed, " --lora1 "live_3d" "a blue vase"
```

### Suffix Analysis with LoRA
```bash
# Add suffix with game assets style
python clip_alignment_with_generation.py --suffix ", stylized game asset" --lora1 "game_assets" "a blue vase"

# Add suffix with cartoon 3D style
python clip_alignment_with_generation.py --suffix ", cartoon style" --lora1 "cartoon_3d" "a blue vase"
```

### Optimization Analysis with LoRA
```bash
# Compare original vs optimized with live 3D style
python clip_alignment_with_generation.py --optimized "a blue ceramic vase with red trim" --lora1 "live_3d" "a blue vase"

# Compare original vs optimized with cinema style
python clip_alignment_with_generation.py --optimized "a blue ceramic vase with red trim" --lora1 "cinema" "a blue vase"
```

## Style Recommendations

### For Technical/Architectural Objects
- **isometric_3d**: Best for technical diagrams and architectural visualizations
- **patched_realism**: Good for realistic technical objects

### For Game Development
- **game_assets**: Ideal for general game objects
- **tf2_style**: Perfect for Team Fortress 2 inspired content
- **cartoon_3d**: Great for animated/cartoon games

### For Realistic Objects
- **live_3d**: Best for photorealistic renders
- **cinema**: Excellent for movie-quality objects
- **patched_realism**: Good for enhanced realistic details

### For Artistic/Stylized Objects
- **baolei**: Specific artistic style
- **cartoon_3d**: Animated/cartoon style
- **tf2_style**: Stylized game art

## Notes

- **LoRA Priority**: If both `--lora1` and `--lora2` are specified, `--lora1` takes precedence
- **Default Endpoint**: If no LoRA is specified, the script uses `/generate_image/isometric_3d/` as default
- **Server Compatibility**: Make sure the TRELLIS server is running with LoRA support enabled
- **Image Quality**: Different LoRA styles may produce different image sizes and quality levels 