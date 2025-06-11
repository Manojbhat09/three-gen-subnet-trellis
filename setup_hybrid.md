# Hybrid TRELLIS + Hunyuan3D-2 Setup Guide

This guide explains how to set up and use the hybrid pipeline that combines TRELLIS's advanced SLAT-based geometry generation with Hunyuan3D-2's superior texture synthesis.

## 🎯 What This Hybrid Achieves

- **Best Geometry**: TRELLIS's structured latent approach for high-quality, flexible 3D shape generation
- **Best Textures**: Hunyuan3D-2's specialized texture synthesis for photorealistic surface details
- **Multiple Formats**: Access to Gaussians, NeRF, and Meshes from TRELLIS + textured meshes from Hunyuan3D-2
- **Enhanced Quality**: Combines the strengths of both systems while mitigating their individual weaknesses

## 📋 Prerequisites

- Linux system (recommended)
- NVIDIA GPU with 16GB+ VRAM
- CUDA 11.8+ or 12.x
- Python 3.8+
- Git LFS for model downloads

## 🛠️ Installation Steps

### 1. Clone Both Repositories

```bash
# Clone TRELLIS
git clone --recurse-submodules https://github.com/microsoft/TRELLIS.git

# Clone Hunyuan3D-2  
git clone https://github.com/Tencent/Hunyuan3D-2.git

# Clone or create hybrid pipeline directory
mkdir hybrid_pipeline
cd hybrid_pipeline
```

### 2. Set Up TRELLIS Environment

```bash
cd TRELLIS

# Create conda environment
conda create -n hybrid_3d python=3.10 -y
conda activate hybrid_3d

# Install TRELLIS dependencies
# Note: Check your CUDA version first
nvidia-smi

# For CUDA 11.8
./setup.sh --new-env --basic --xformers --flash-attn --diffoctreerast --spconv --mipgaussian --kaolin --nvdiffrast

# Alternative: Manual installation if script fails
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt  # TRELLIS requirements
```

### 3. Set Up Hunyuan3D-2 Components

```bash
cd ../Hunyuan3D-2

# Install Hunyuan3D-2 dependencies
pip install -r requirements.txt

# Install texture processing components
cd hy3dgen/texgen/custom_rasterizer
python3 setup.py install

cd ../differentiable_renderer  
bash compile_mesh_painter.sh

# Install additional required packages
pip install xatlas dataclasses-json
```

### 4. Install Hybrid Pipeline Dependencies

```bash
cd ../../hybrid_pipeline

# Copy the hybrid script and requirements
# (These should be in your current directory)

# Install any remaining dependencies
pip install -r hybrid_requirements.txt
```

### 5. Download Models

```bash
# TRELLIS models will be downloaded automatically on first use
# Hunyuan3D-2 models will also be downloaded automatically

# Alternatively, pre-download for faster startup:
python -c "
from trellis.pipelines import TrellisImageTo3DPipeline
from hy3dgen.texgen import Hunyuan3DPaintPipeline

# This will download the models
TrellisImageTo3DPipeline.from_pretrained('microsoft/TRELLIS-image-large')
Hunyuan3DPaintPipeline.from_pretrained('tencent/Hunyuan3D-2')
"
```

## 🚀 Usage Examples

### Basic Usage

```python
from hybrid_trellis_hunyuan import HybridTrellisHunyuanPipeline

# Initialize the hybrid pipeline
pipeline = HybridTrellisHunyuanPipeline()

# Generate 3D asset from image
result = pipeline.generate_3d(
    image="path/to/your/image.png",
    output_path="my_3d_asset",
    save_intermediate=True
)

print(f"Generated mesh with {len(result.vertices)} vertices")
```

### Advanced Configuration

```python
# Custom parameters for each stage
trellis_params = {
    "seed": 42,
    "sparse_structure_sampler_params": {
        "steps": 15,  # Higher quality
        "cfg_strength": 8.0,
    },
    "slat_sampler_params": {
        "steps": 15,
        "cfg_strength": 4.0,
    },
}

result = pipeline.generate_3d(
    image="input.jpg",
    output_path="high_quality_output",
    trellis_params=trellis_params,
    save_intermediate=True
)
```

### Batch Processing

```python
# Process multiple images
image_list = ["img1.png", "img2.jpg", "img3.png"]

results = pipeline.batch_generate(
    image_list,
    output_dir="batch_results"
)
```

## 📁 Output Structure

The hybrid pipeline generates several outputs:

```
output_name_input_processed.png      # Preprocessed input image
output_name_geometry_trellis.glb     # Raw TRELLIS geometry
output_name_gaussian.ply             # 3D Gaussian representation
output_name_geometry_prepared.glb    # Prepared mesh for texturing
output_name_final_textured.glb       # Final textured mesh
```

## 🔧 Troubleshooting

### Common Issues

1. **CUDA Version Mismatch**
   ```bash
   # Check CUDA version
   nvidia-smi
   nvcc --version
   
   # Install correct PyTorch version
   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
   ```

2. **Memory Issues**
   ```python
   # Reduce batch size or use smaller models
   os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'
   ```

3. **Mesh Compatibility Issues**
   ```python
   # The hybrid pipeline includes automatic mesh alignment
   # If issues persist, check mesh validity:
   print(f"Mesh watertight: {mesh.is_watertight}")
   print(f"Mesh volume: {mesh.volume}")
   ```

### Performance Optimization

1. **Use FP16 for Lower VRAM**
   ```python
   pipeline = HybridTrellisHunyuanPipeline(device="cuda")
   # Models automatically use FP16 when available
   ```

2. **Enable Compilation (PyTorch 2.0+)**
   ```python
   # This is handled automatically in the pipeline
   torch.compile(pipeline.trellis_pipeline.model)
   ```

3. **Adjust Quality vs Speed**
   ```python
   # Fast mode
   trellis_params = {
       "sparse_structure_sampler_params": {"steps": 8},
       "slat_sampler_params": {"steps": 8},
   }
   
   # High quality mode  
   trellis_params = {
       "sparse_structure_sampler_params": {"steps": 20},
       "slat_sampler_params": {"steps": 20},
   }
   ```

## 📊 Expected Performance

- **Geometry Generation (TRELLIS)**: 2-5 minutes on RTX 4090
- **Texture Synthesis (Hunyuan3D-2)**: 3-8 minutes on RTX 4090
- **Total Pipeline**: 5-15 minutes per image
- **Memory Usage**: 12-20GB VRAM depending on settings

## 🎨 Quality Comparison

| Aspect | TRELLIS Only | Hunyuan3D-2 Only | Hybrid |
|--------|-------------|------------------|---------|
| Geometry Quality | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| Texture Quality | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| Output Diversity | ⭐⭐⭐⭐⭐ | ⭐⭐ | ⭐⭐⭐⭐ |
| Processing Speed | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐ |

## 🔄 Alternative Configurations

### TRELLIS Geometry + Custom Texture Pipeline
```python
# Use only TRELLIS for geometry
trellis_mesh = pipeline.trellis_pipeline.run(image)['mesh'][0]

# Apply your own texture processing
custom_textured_mesh = your_texture_function(trellis_mesh, image)
```

### Mixed Output Formats
```python
# Get all TRELLIS representations plus Hunyuan texture
outputs = pipeline.trellis_pipeline.run(image)
textured_mesh = pipeline.texture_pipeline(outputs['mesh'][0], image)

# Now you have:
# - outputs['gaussian'][0] (3D Gaussians)
# - outputs['radiance_field'][0] (NeRF)  
# - textured_mesh (High-quality textured mesh)
```

This hybrid approach gives you the flexibility of TRELLIS with the texture quality of Hunyuan3D-2! 