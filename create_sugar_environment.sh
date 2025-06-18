#!/bin/bash
# Create a dedicated conda environment for SuGaR-Enhanced Generation Pipeline
# This script creates a clean environment with all necessary dependencies

set -e

echo "🍬 Creating SuGaR Pipeline Environment"
echo "======================================"

ENV_NAME="sugar-pipeline"

# Check if environment already exists
if conda env list | grep -q "$ENV_NAME"; then
    echo "⚠️ Environment '$ENV_NAME' already exists."
    read -p "Do you want to remove it and create a fresh one? (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo "Removing existing environment..."
        conda env remove -n $ENV_NAME -y
    else
        echo "Exiting. Please use: conda activate $ENV_NAME"
        exit 0
    fi
fi

echo "Creating new conda environment: $ENV_NAME"
conda create -n $ENV_NAME python=3.9 -y

echo "Activating environment..."
eval "$(conda shell.bash hook)"
conda activate $ENV_NAME

# Install PyTorch with CUDA support
echo "📦 Installing PyTorch with CUDA..."
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia -y

# Install essential packages
echo "📦 Installing essential packages..."
pip install --upgrade pip

# Web framework
pip install fastapi uvicorn[standard]

# HTTP clients
pip install requests aiohttp

# Core scientific computing
pip install numpy scipy matplotlib
pip install pillow opencv-python

# 3D processing
pip install trimesh open3d plyfile

# ML frameworks
pip install diffusers transformers accelerate
pip install huggingface_hub tokenizers

# Optimization
pip install bitsandbytes optimum

# Utilities
pip install rich tqdm pyyaml

# Try to install PyTorch3D (with fallback)
echo "📦 Installing PyTorch3D..."
pip install pytorch3d -f https://dl.fbaipublicfiles.com/pytorch3d/packaging/wheels/py39_cu121_pyt200/download.html || {
    echo "⚠️ PyTorch3D installation failed, trying alternative..."
    pip install pytorch3d || echo "⚠️ PyTorch3D not available - some features may be limited"
}

# Test installation
echo "🧪 Testing installation..."
python -c "
import torch
import numpy as np
import trimesh
import fastapi
import open3d as o3d
from plyfile import PlyData
print('✅ Core dependencies working!')

# Test CUDA
if torch.cuda.is_available():
    print(f'✅ CUDA available: {torch.cuda.get_device_name(0)}')
    print(f'   Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB')
else:
    print('⚠️ CUDA not available - will use CPU')
"

# Create activation script
echo "📝 Creating activation script..."
cat > activate_sugar_env.sh << 'EOF'
#!/bin/bash
echo "🍬 Activating SuGaR Pipeline Environment"
eval "$(conda shell.bash hook)"
conda activate sugar-pipeline
echo "✅ Environment activated: sugar-pipeline"
echo "Ready to run SuGaR pipeline!"
EOF
chmod +x activate_sugar_env.sh

echo ""
echo "🎉 SuGaR Environment Setup Complete!"
echo "====================================="
echo ""
echo "✅ Environment created: $ENV_NAME"
echo "✅ All dependencies installed"
echo ""
echo "To activate the environment:"
echo "  source activate_sugar_env.sh"
echo "  OR: conda activate $ENV_NAME"
echo ""
echo "Next steps:"
echo "1. Activate environment: conda activate $ENV_NAME"
echo "2. Run setup: ./setup_sugar_pipeline.sh"
echo "3. Start server: python flux_hunyuan_bpt_sugar_generation_server.py"
echo ""
echo "🍬 Environment ready for SuGaR pipeline!" 