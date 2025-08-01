#!/bin/bash

# MV-Adapter GS Enhancement Setup Script
# Purpose: Install and configure MV-Adapter for Gaussian Splatting quality enhancement

set -e

echo "🎨 MV-Adapter GS Enhancement Setup"
echo "=================================="

# Check if MV-Adapter directory exists
if [ ! -d "MV-Adapter" ]; then
    echo "📥 Cloning MV-Adapter repository..."
    git clone https://github.com/huanngzh/MV-Adapter.git
    cd MV-Adapter
else
    echo "✅ MV-Adapter directory found"
    cd MV-Adapter
fi

# Check Python version
python_version=$(python3 --version 2>&1 | awk '{print $2}' | cut -d. -f1,2)
echo "🐍 Python version: $python_version"

# Create virtual environment if not exists
if [ ! -d "venv" ]; then
    echo "🔧 Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
echo "🔧 Activating virtual environment..."
source venv/bin/activate

# Install PyTorch (adjust CUDA version as needed)
echo "🔥 Installing PyTorch..."
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# Install MV-Adapter requirements
echo "📦 Installing MV-Adapter requirements..."
pip install -r requirements.txt

# Install additional dependencies for GS enhancement
echo "📦 Installing additional dependencies..."
pip install trimesh plyfile numpy pillow

# Create checkpoints directory
echo "📁 Creating checkpoints directory..."
mkdir -p checkpoints

# Download required model weights
echo "📥 Downloading model weights..."

# RealESRGAN for upscaling
if [ ! -f "checkpoints/RealESRGAN_x2plus.pth" ]; then
    echo "📥 Downloading RealESRGAN upscaler..."
    wget https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth -O checkpoints/RealESRGAN_x2plus.pth
else
    echo "✅ RealESRGAN upscaler already exists"
fi

# LaMa for inpainting
if [ ! -f "checkpoints/big-lama.pt" ]; then
    echo "📥 Downloading LaMa inpainter..."
    wget https://github.com/Sanster/models/releases/download/add_big_lama/big-lama.pt -O checkpoints/big-lama.pt
else
    echo "✅ LaMa inpainter already exists"
fi

# Test MV-Adapter installation
echo "🧪 Testing MV-Adapter installation..."
python3 -c "
import sys
sys.path.append('.')
try:
    from mvadapter.pipelines.pipeline_texture import TexturePipeline
    print('✅ MV-Adapter texture pipeline imported successfully')
except ImportError as e:
    print(f'❌ MV-Adapter import failed: {e}')
    sys.exit(1)
"

echo ""
echo "🎉 MV-Adapter GS Enhancement Setup Complete!"
echo "============================================"
echo ""
echo "Next steps:"
echo "1. Copy the trellis_gs_quality_enhancer.py to your project"
echo "2. Integrate it with your Trellis server"
echo "3. Test with a simple prompt"
echo ""
echo "Example integration:"
echo "from trellis_gs_quality_enhancer import TrellisGSQualityEnhancer"
echo "enhancer = TrellisGSQualityEnhancer(trellis_pipeline)"
echo "results = enhancer.generate_enhanced_3d_model('a beautiful vase', 42)"
echo "compressed_ply = results['compressed_ply_data']  # Ready for validator"
echo ""
echo "Configuration options:"
echo "- mv_adapter_variant: 'sdxl' (high quality) or 'sd21' (lower VRAM)"
echo "- enhancement_strength: 0.0 to 1.0 (how much to enhance)"
echo "- enable_quality_assessment: True/False (assess quality improvements)"
echo ""

# Deactivate virtual environment
deactivate

echo "✅ Setup completed successfully!" 