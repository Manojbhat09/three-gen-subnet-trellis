# Flux + Hunyuan3D + BPT Integration

This enhanced script combines three powerful models to generate high-quality 3D meshes from text prompts:

1. **Flux** - Advanced text-to-image generation
2. **Hunyuan3D** - Image-to-3D mesh generation  
3. **BPT (Blocked and Patchified Tokenization)** - Mesh enhancement for high-detail output

## Features

- **High-Detail Meshes**: BPT can generate meshes with 8k+ faces, significantly more detailed than standard generation
- **Compressive Tokenization**: BPT reduces mesh sequence length by ~75% while maintaining quality
- **Flexible Control**: Works with point clouds and images as conditioning
- **Memory Efficient**: GPU memory management throughout the pipeline
- **Texture Support**: Optional texture generation with Hunyuan3D Paint

## New Files

- `flux_hunyuan_bpt_demo.py` - Main script with BPT integration
- `test_bpt_integration.py` - Test script to validate BPT functionality
- `README_BPT_Integration.md` - This documentation

## Requirements

Make sure you're in the correct conda environment:

```bash
source /home/mbhat/miniconda/bin/activate
conda activate hunyuan3d
```

Additional dependencies may be needed:
```bash
pip install dataclasses-json  # For texture generation
pip install protobuf         # Auto-installed by script
```

## Usage

### Basic Usage

```python
from flux_hunyuan_bpt_demo import text_to_3d_with_bpt

# Generate with BPT enhancement
text_to_3d_with_bpt(
    prompt="a red sports car", 
    use_bpt=True, 
    bpt_temperature=0.5
)
```

### Configuration Options

- **`use_bpt`** (bool): Enable/disable BPT enhancement
- **`bpt_temperature`** (float): Controls randomness in BPT generation (0.1-1.0)
  - Lower values (0.1-0.4): More conservative, consistent results
  - Higher values (0.5-1.0): More creative, varied results
- **`output_dir`** (str): Output directory for generated files
- **`seed`** (int): Random seed for reproducible results

### Example Commands

```python
# High-detail architectural model
text_to_3d_with_bpt(
    "charming red barn with weathered wood", 
    use_bpt=True, 
    bpt_temperature=0.4
)

# Creative character generation
text_to_3d_with_bpt(
    "3 eyed mystical creature with red horns", 
    use_bpt=True, 
    bpt_temperature=0.6, 
    seed=121
)

# Mechanical object with precision
text_to_3d_with_bpt(
    "purple durable robotic arm", 
    use_bpt=True, 
    bpt_temperature=0.3
)
```

## Output Files

The script generates several files in the output directory:

1. **`t2i_original.png`** - Original Flux-generated image
2. **`t2i_no_bg.png`** - Background-removed image
3. **`t2i_initial.glb`** - Initial Hunyuan3D mesh
4. **`t2i_enhanced_bpt.glb`** - BPT-enhanced high-detail mesh (if successful)
5. **`t2i_textured.glb`** - Textured version (if texture generation succeeds)

## BPT Enhancement Process

1. **Initial Generation**: Hunyuan3D creates a base mesh from the image
2. **Point Cloud Sampling**: Extract 4096 points with normals from the mesh
3. **BPT Generation**: Generate enhanced mesh using compressed tokenization
4. **Post-processing**: Apply normalization and quality improvements

## Performance Notes

- **GPU Memory**: ~12GB VRAM recommended for full pipeline
- **Generation Time**: 
  - Flux: ~30 seconds
  - Hunyuan3D: ~1-2 minutes
  - BPT Enhancement: ~2-3 minutes
- **Total Time**: ~5-6 minutes per generation

## Troubleshooting

### Test BPT Integration

Run the test script first to ensure BPT is working:

```bash
cd Hunyuan3D-2
python test_bpt_integration.py
```

### Common Issues

1. **Missing BPT weights**: The script will use randomly initialized weights if pre-trained weights aren't found
2. **CUDA out of memory**: Reduce batch size or use CPU for BPT if needed
3. **Import errors**: Ensure you're in the correct conda environment

### BPT Model Weights

The script looks for BPT weights at:
```
Hunyuan3D-2/hy3dgen/shapegen/bpt/weights/bpt-8-16-500m.pt
```

If weights are missing, the script will continue with randomly initialized weights (results may be lower quality).

## Comparison with Original

| Feature | Original Script | BPT Enhanced |
|---------|----------------|--------------|
| Face Count | ~1-2k faces | 8k+ faces |
| Detail Level | Standard | High |
| Generation Time | ~3 minutes | ~6 minutes |
| Memory Usage | ~8GB | ~12GB |
| Mesh Quality | Good | Excellent |

## Architecture Overview

```
Text Prompt
    ↓
[Flux] → Image (1024x1024)
    ↓
[Background Removal] → Clean Image
    ↓
[Hunyuan3D] → Base Mesh (~1k faces)
    ↓
[BPT Enhancement] → High-Detail Mesh (8k+ faces)
    ↓
[Optional Texturing] → Final Textured Mesh
```

## Advanced Configuration

### Custom BPT Config

```python
config = {
    'dim': 1024,           # Model dimension
    'max_seq_len': 8192,   # Maximum sequence length
    'attn_depth': 24,      # Number of attention layers
    'block_size': 8,       # Block size for compression
    'offset_size': 16,     # Offset size for compression
    'temperature': 0.5     # Generation temperature
}
```

### Memory Optimization

For lower memory usage:
- Set `use_bpt=False` to skip BPT enhancement
- Reduce `max_seq_len` in BPT config
- Use lower resolution images (512x512)

## Credits

- **Flux**: Advanced text-to-image generation
- **Hunyuan3D**: Tencent's image-to-3D pipeline
- **BPT**: Blocked and Patchified Tokenization for high-detail mesh generation
- **Integration**: Custom pipeline combining all three models

## License

This integration follows the licenses of the individual components:
- Flux: According to diffusers license
- Hunyuan3D: Apache License Version 2.0
- BPT: Original research implementation license 