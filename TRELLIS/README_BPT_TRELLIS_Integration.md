# Flux + TRELLIS + BPT Integration

This advanced script combines four powerful models to generate ultra-high-quality 3D meshes from text prompts:

1. **Flux** - Advanced text-to-image generation
2. **TRELLIS** - Image-to-3D mesh generation with superior texture quality
3. **BPT (Blocked and Patchified Tokenization)** - Mesh enhancement for high-detail output  
4. **Hunyuan3D Tools** - Shape optimization and texture painting

## 🚀 Complete Pipeline

```
Text Prompt
    ↓
[Flux] → Image (1024x1024)
    ↓
[Background Removal] → Clean Image
    ↓
[TRELLIS] → Raw 3D Assets (Gaussians + Mesh ~1-2k faces)
    ↓
[BPT Enhancement] → High-Detail Mesh (8k+ faces)
    ↓
[Shape Optimization] → Optimized Mesh
    ↓
[Advanced Texture Baking] → Final Textured GLB
```

## 🆕 New Features

### BPT Mesh Enhancement
- **8k+ Face Meshes**: Generates significantly more detailed meshes than standard pipelines
- **Compressive Tokenization**: Reduces sequence length by ~75% while maintaining quality
- **Point Cloud Conditioning**: Uses 4096 point cloud samples with normals for enhanced generation
- **Temperature Control**: Configurable creativity vs consistency (0.1-1.0)

### Advanced Texture Baking (3 Methods)
1. **TRELLIS UV Re-baking** (First Priority) - Preserves original Gaussian texture quality
2. **Hunyuan3D Paint Pipeline** (Fallback) - Direct texture painting to mesh surface
3. **Simple Spherical Projection** (Last Resort) - Basic UV mapping with coordinate fixes

### Quality Presets (Optimized for BPT)
- **Draft**: 1K textures, 30K faces, BPT temp 0.3
- **Standard**: 2K textures, 60K faces, BPT temp 0.5  
- **High**: 4K textures, 80K faces, BPT temp 0.4
- **Ultra**: 6K textures, 100K faces, BPT temp 0.3

## 📁 Files

- `flux_trellis_bpt_retextured_optimized.py` - Main integration script
- `test_bpt_integration.py` - Test script to validate BPT functionality
- `README_BPT_TRELLIS_Integration.md` - This documentation

## 🔧 Requirements

Make sure you're in the correct conda environment:

```bash
source /home/mbhat/miniconda/bin/activate
conda activate hunyuan3d
```

### Dependencies
```bash
# Core requirements (should already be installed)
pip install torch torchvision torchaudio
pip install diffusers transformers
pip install trimesh rembg
pip install imageio

# For texture generation (optional)
pip install dataclasses-json

# For BPT (from Hunyuan3D-2 installation)
# These should be available if Hunyuan3D-2 is properly installed
```

### Environment Variables
The script automatically sets these for CUDA compilation:
```bash
export TORCH_CUDA_ARCH_LIST="8.9"
export NVCC_APPEND_FLAGS="-allow-unsupported-compiler"
```

## 🚀 Usage

### Quick Test
```bash
cd TRELLIS
python test_bpt_integration.py
```

### Interactive Usage
```bash
cd TRELLIS  
python flux_trellis_bpt_retextured_optimized.py
```

### Programmatic Usage
```python
from flux_trellis_bpt_retextured_optimized import FluxTrellisBPTRetexturedOptimized

pipeline = FluxTrellisBPTRetexturedOptimized()

result = pipeline.run_pipeline(
    prompt="purple durable robotic arm",
    quality="standard",  # draft, standard, high, ultra
    output_prefix="my_output",
    seed=42
)

print("Generated files:", result['files'])
```

## 📊 Output Files

The pipeline generates multiple files for comparison and analysis:

1. **`*_original.png`** - Original Flux-generated image
2. **`*_processed.png`** - Background-removed image  
3. **`*_textured.glb`** - Standard TRELLIS textured mesh
4. **`*_bpt_retextured_optimized.glb`** - BPT-enhanced and re-textured mesh
5. **`*_gs.mp4`** - Gaussian splatting rotation video
6. **`*_optimized.ply`** - Optimized Gaussian point cloud

## 🔬 BPT Enhancement Details

### Configuration
```python
bpt_config = {
    'dim': 1024,              # Model dimension
    'max_seq_len': 8192,      # Maximum sequence length  
    'attn_depth': 24,         # Number of attention layers
    'block_size': 8,          # Block size for compression
    'offset_size': 16,        # Offset size for compression
    'temperature': 0.5        # Generation temperature
}
```

### Process
1. **Point Cloud Sampling**: Extract 4096 points with normals from TRELLIS mesh
2. **BPT Generation**: Generate enhanced mesh using compressed tokenization
3. **Deserialization**: Convert BPT codes back to 3D coordinates
4. **Mesh Creation**: Assemble coordinates into high-detail triangular mesh
5. **Normalization**: Apply standard mesh normalization

## 🎯 Performance Comparison

| Feature | TRELLIS Only | TRELLIS + BPT |
|---------|-------------|---------------|
| Face Count | ~1-2k faces | 8k+ faces |
| Detail Level | Good | Excellent |
| Generation Time | ~3 minutes | ~6 minutes |
| Memory Usage | ~8GB | ~14GB |
| Texture Quality | Excellent | Excellent (preserved) |

## 🔧 Troubleshooting

### BPT Model Weights
The script looks for BPT weights at:
```
../Hunyuan3D-2/hy3dgen/shapegen/bpt/weights/bpt-8-16-500m.pt
```

If not found, it uses randomly initialized weights (lower quality results).

### Common Issues

1. **CUDA Out of Memory**: Reduce quality level or use CPU for BPT
2. **Import Errors**: Ensure Hunyuan3D-2 is properly installed
3. **nvdiffrast Compilation**: Environment variables should fix GCC issues
4. **Dtype Mismatches**: Script includes comprehensive dtype fixes

### Memory Optimization
- Use `quality='draft'` for testing
- BPT enhancement is optional (graceful fallback)
- GPU memory is cleared between pipeline steps

## 📈 Quality Settings Guide

### Draft (Fast Testing)
- **Use Case**: Quick iterations, testing
- **Settings**: 1K textures, 30K faces, conservative BPT
- **Time**: ~4 minutes
- **Quality**: Good for prototyping

### Standard (Recommended)
- **Use Case**: Production use, balanced quality/speed
- **Settings**: 2K textures, 60K faces, moderate BPT
- **Time**: ~6 minutes  
- **Quality**: Excellent for most applications

### High (Premium Quality)
- **Use Case**: High-quality assets, detailed models
- **Settings**: 4K textures, 80K faces, creative BPT
- **Time**: ~8 minutes
- **Quality**: Professional-grade results

### Ultra (Maximum Quality)
- **Use Case**: Hero assets, showcasing
- **Settings**: 6K textures, 100K faces, precise BPT  
- **Time**: ~12 minutes
- **Quality**: Showcase-quality results

## 🎨 Texture Baking Methods

### Method 1: TRELLIS UV Re-baking (Primary)
- **Accuracy**: Highest - preserves original Gaussian texture
- **Process**: Re-parameterize mesh → render 120 multiview images → bake onto new UV map
- **Requirements**: Successful UV unwrapping
- **Quality**: Excellent texture preservation

### Method 2: Hunyuan3D Paint Pipeline (Fallback)  
- **Accuracy**: Good - reliable direct painting
- **Process**: Apply texture directly to mesh surface
- **Requirements**: Hunyuan3D texture generation available
- **Quality**: Good for most object types

### Method 3: Spherical Projection (Last Resort)
- **Accuracy**: Basic - simple UV mapping
- **Process**: Project 2D image onto 3D mesh using spherical coordinates
- **Requirements**: Always available
- **Quality**: Acceptable for simple objects

## 🔄 Coordinate System Fixes

The integration includes comprehensive coordinate system corrections:

- **90-degree Y-axis rotation fix**: `[[0, 0, 1], [0, 1, 0], [-1, 0, 0]]`
- **Spherical UV mapping**: Corrected `arctan2(x, z)` for proper front-facing
- **Texture rotation**: 90-degree correction for input images
- **UV coordinate rotation**: 270-degree correction with V-flipping

## 🏗️ Architecture Benefits

### Vs. Hunyuan3D Only
- **Better Textures**: TRELLIS produces superior texture quality
- **Higher Detail**: BPT enhancement adds 4-8x more faces
- **Better Integration**: Seamless pipeline vs separate tools

### Vs. TRELLIS Only  
- **Much Higher Detail**: 8k+ faces vs 1-2k faces
- **Enhanced Geometry**: BPT adds fine details and features
- **Maintained Quality**: All texture baking methods preserve quality

## 📋 Example Commands

```bash
# Quick test with BPT
python test_bpt_integration.py

# Single prompt with high quality
python -c "
from flux_trellis_bpt_retextured_optimized import FluxTrellisBPTRetexturedOptimized
pipeline = FluxTrellisBPTRetexturedOptimized()
pipeline.run_pipeline('flying dragon', 'high', 'dragon_output', 42)
"

# Batch processing with different qualities
python flux_trellis_bpt_retextured_optimized.py
```

## 🎉 Success Indicators

Look for these messages in the output:

```
✅ BPT mesh enhancement tools loaded
✅ BPT enhancement successful!
✅ TRELLIS UV re-baking successful!
✅ Successfully created BPT-enhanced and re-textured GLB
```

## 📝 Credits

- **Flux**: Advanced text-to-image generation
- **TRELLIS**: Microsoft's superior image-to-3D pipeline  
- **BPT**: Blocked and Patchified Tokenization for mesh enhancement
- **Hunyuan3D**: Tencent's shape optimization and texture painting tools
- **Integration**: Custom pipeline combining all four technologies

## 📄 License

This integration follows the licenses of the individual components:
- Flux: According to diffusers license
- TRELLIS: Microsoft Research license
- Hunyuan3D: Apache License Version 2.0  
- BPT: Original research implementation license 