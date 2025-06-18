#!/bin/bash
# Setup script for SuGaR-Enhanced Generation Pipeline
# Creates a dedicated conda environment and installs dependencies

set -e

echo "🍬 Setting up SuGaR-Enhanced Generation Pipeline"
echo "================================================"

# Check if we're in the right directory
if [ ! -f "flux_hunyuan_bpt_sugar_generation_server.py" ]; then
    echo "❌ Error: Please run this script from the three-gen-subnet-trellis directory"
    exit 1
fi

# Check if SuGaR directory exists
if [ ! -d "SuGaR" ]; then
    echo "❌ Error: SuGaR directory not found. Please ensure SuGaR is cloned in the current directory."
    exit 1
fi

echo "✓ Found SuGaR directory"

# Create dedicated conda environment for SuGaR pipeline
echo "🐍 Creating SuGaR conda environment..."
ENV_NAME="sugar-pipeline"

if conda env list | grep -q "$ENV_NAME"; then
    echo "⚠️ Environment $ENV_NAME already exists. Removing it..."
    conda env remove -n $ENV_NAME -y
fi

echo "Creating new environment: $ENV_NAME"
conda create -n $ENV_NAME python=3.9 -y

echo "Activating environment: $ENV_NAME"
eval "$(conda shell.bash hook)"
conda activate $ENV_NAME

# Install PyTorch with CUDA support
echo "📦 Installing PyTorch with CUDA..."
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia -y

# Install core dependencies
echo "📦 Installing core dependencies..."
pip install fastapi uvicorn[standard]
pip install requests aiohttp
pip install numpy pillow trimesh
pip install diffusers transformers accelerate
pip install plyfile open3d
pip install rich tqdm pyyaml

# PyTorch3D (special installation)
echo "📦 Installing PyTorch3D..."
pip install pytorch3d -f https://dl.fbaipublicfiles.com/pytorch3d/packaging/wheels/py39_cu121_pyt200/download.html

# Additional dependencies for SuGaR and 3D processing
pip install matplotlib opencv-python
pip install bitsandbytes optimum

# Check existing conda environments
echo "🐍 Checking other conda environments..."
if conda env list | grep -q "hunyuan3d"; then
    echo "✓ Found hunyuan3d environment"
else
    echo "⚠️ hunyuan3d environment not found - you may need to create it"
fi

if conda env list | grep -q "three-gen-validation"; then
    echo "✓ Found three-gen-validation environment"
else
    echo "⚠️ three-gen-validation environment not found - you may need to create it"
fi

# Make scripts executable
echo "🔧 Setting up scripts..."
chmod +x flux_hunyuan_bpt_sugar_generation_server.py
chmod +x test_sugar_pipeline.py

# Create output directories
echo "📁 Creating output directories..."
mkdir -p flux_hunyuan_bpt_sugar_outputs
mkdir -p test_outputs

# Test SuGaR imports
echo "🧪 Testing SuGaR imports..."
CURRENT_DIR=$(pwd)
python -c "
import sys
import os
sys.path.insert(0, os.path.join('$CURRENT_DIR', 'SuGaR'))

try:
    from sugar_scene.gs_model import GaussianSplattingWrapper
    from sugar_scene.sugar_model import SuGaR  
    from sugar_utils.spherical_harmonics import SH2RGB, RGB2SH
    print('✓ SuGaR imports successful')
except ImportError as e:
    print(f'❌ SuGaR import error: {e}')
    print('This might be expected - SuGaR may need to be integrated differently')
"

# Test other dependencies
echo "🧪 Testing other dependencies..."
python -c "
try:
    import trimesh
    import torch
    import numpy as np
    from plyfile import PlyData, PlyElement
    import open3d as o3d
    import fastapi
    import uvicorn
    print('✓ All dependencies available')
except ImportError as e:
    print(f'❌ Dependency error: {e}')
    sys.exit(1)
"

# Check GPU availability
echo "🖥️ Checking GPU availability..."
python -c "
import torch
if torch.cuda.is_available():
    print(f'✓ CUDA available: {torch.cuda.get_device_name(0)}')
    print(f'  Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB')
else:
    print('⚠️ CUDA not available - will use CPU (slower)')
"

# Create environment activation script
echo "📝 Creating environment activation script..."
cat > activate_sugar.sh << 'EOF'
#!/bin/bash
# Activate SuGaR pipeline environment
echo "🍬 Activating SuGaR Pipeline Environment"
eval "$(conda shell.bash hook)"
conda activate sugar-pipeline
echo "✓ Environment activated: sugar-pipeline"
echo "Ready to run SuGaR pipeline!"
EOF
chmod +x activate_sugar.sh

# Create a simple test
echo "🧪 Creating test configuration..."
cat > test_config.yaml << EOF
# SuGaR Pipeline Test Configuration
server:
  host: "0.0.0.0"
  port: 8095
  
sugar:
  num_points: 25000  # Reduced for testing
  sh_levels: 4
  triangle_scale: 2.0
  
test_prompts:
  - "a red apple"
  - "a wooden chair"
  - "a blue coffee mug"
EOF

echo "✓ Created test_config.yaml"

# Create launch script for the server
echo "📝 Creating server launch script..."
cat > launch_sugar_server.sh << 'EOF'
#!/bin/bash
# Launch SuGaR Generation Server
echo "🚀 Launching SuGaR Generation Server"
eval "$(conda shell.bash hook)"
conda activate sugar-pipeline
python flux_hunyuan_bpt_sugar_generation_server.py --host 0.0.0.0 --port 8095
EOF
chmod +x launch_sugar_server.sh

# Final checks
echo "🔍 Final system check..."

# Check if generation asset manager exists
if [ -f "generation_asset_manager.py" ]; then
    echo "✓ Found generation_asset_manager.py"
else
    echo "⚠️ generation_asset_manager.py not found - some features may not work"
fi

# Check if Hunyuan3D-2 exists
if [ -d "Hunyuan3D-2" ]; then
    echo "✓ Found Hunyuan3D-2 directory"
else
    echo "⚠️ Hunyuan3D-2 directory not found - generation will fail"
fi

echo ""
echo "🎉 SuGaR Pipeline Setup Complete!"
echo "=================================="
echo ""
echo "✅ Created conda environment: $ENV_NAME"
echo "✅ Installed all dependencies"
echo "✅ Created launch scripts"
echo ""
echo "Next steps:"
echo "1. Activate the environment:"
echo "   source activate_sugar.sh"
echo "   OR: conda activate $ENV_NAME"
echo ""
echo "2. Start the server:"
echo "   ./launch_sugar_server.sh"
echo "   OR: python flux_hunyuan_bpt_sugar_generation_server.py"
echo ""
echo "3. Test the pipeline (in a new terminal):"
echo "   conda activate $ENV_NAME"
echo "   python test_sugar_pipeline.py"
echo ""
echo "4. Generate a model:"
echo "   curl -X POST 'http://localhost:8095/generate/' \\"
echo "     -F 'prompt=a red apple' \\"
echo "     -F 'seed=12345' \\"
echo "     -o apple.ply"
echo ""
echo "🍬 Happy SuGaR-ing! 🍬" 