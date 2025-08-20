#!/bin/bash

# Nunchaku Environment Setup Script
# This script creates and configures the 'nun' conda environment for Nunchaku

set -e  # Exit on any error

echo "🚀 Setting up Nunchaku environment..."

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if conda is available
if ! command -v conda &> /dev/null; then
    print_error "Conda is not installed or not in PATH"
    print_error "Please install Anaconda or Miniconda first"
    exit 1
fi

print_status "Conda found: $(conda --version)"

# Check if CUDA is available
if ! command -v nvidia-smi &> /dev/null; then
    print_warning "NVIDIA GPU not detected. Nunchaku requires CUDA support."
    print_warning "Continuing with setup, but GPU acceleration may not work."
else
    print_success "NVIDIA GPU detected: $(nvidia-smi --query-gpu=name --format=csv,noheader,nounits | head -1)"
fi

# Check if environment already exists
if conda env list | grep -q "nun"; then
    print_warning "Environment 'nun' already exists"
    read -p "Do you want to remove and recreate it? (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        print_status "Removing existing 'nun' environment..."
        conda env remove -n nun
        print_success "Environment removed"
    else
        print_status "Using existing environment"
        conda activate nun
        exit 0
    fi
fi

# Create new environment
print_status "Creating new conda environment 'nun' with Python 3.11..."
conda create -n nun python=3.11 -y
print_success "Environment created"

# Activate environment
print_status "Activating 'nun' environment..."
source $(conda info --base)/etc/profile.d/conda.sh
conda activate nun

# Verify Python version
PYTHON_VERSION=$(python --version)
print_success "Python version: $PYTHON_VERSION"

# Install PyTorch with CUDA support
print_status "Installing PyTorch with CUDA 12.8 support..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128

# Verify PyTorch installation
print_status "Verifying PyTorch installation..."
python -c "
import torch
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA version: {torch.version.cuda}')
    print(f'GPU count: {torch.cuda.device_count()}')
    print(f'Current GPU: {torch.cuda.get_device_name(0)}')
"

# Install Nunchaku
print_status "Installing Nunchaku 0.3.1..."
pip install https://github.com/nunchaku-tech/nunchaku/releases/download/v0.3.1/nunchaku-0.3.1+torch2.7-cp311-cp311-linux_x86_64.whl

# Install additional dependencies
print_status "Installing additional dependencies..."
pip install ninja wheel diffusers transformers accelerate sentencepiece protobuf huggingface_hub

# Install Flask for the API server
print_status "Installing Flask for API server..."
pip install flask requests pillow

# Verify Nunchaku installation
print_status "Verifying Nunchaku installation..."
python -c "
try:
    import nunchaku
    print('✅ Nunchaku imported successfully')
    print(f'Version: {nunchaku.__version__}')
except ImportError as e:
    print(f'❌ Failed to import Nunchaku: {e}')
    exit(1)
"

# Create a test script
print_status "Creating test script..."
cat > test_nunchaku_import.py << 'EOF'
#!/usr/bin/env python3
"""
Test script to verify Nunchaku installation
"""

import os
import gc
import torch

print("🧪 Testing Nunchaku installation...")

try:
    # Set environment variables
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = "expandable_segments:True"
    os.environ["TOKENIZERS_PARALLELISM"] = "True"
    torch._dynamo.config.suppress_errors = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.enabled = True
    
    # Clear GPU memory
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    # Import Nunchaku
    from nunchaku.models.transformers.transformer_flux import NunchakuFluxTransformer2dModel
    from diffusers import FluxPipeline
    
    print("✅ Nunchaku imports successful")
    
    # Test model loading (this will download models)
    print("🔧 Testing model loading...")
    transformer = NunchakuFluxTransformer2dModel.from_pretrained(
        "mit-han-lab/svdq-int4-flux.1-schnell",
        torch_dtype=torch.bfloat16
    )
    print("✅ Transformer loaded successfully")
    
    print("🎉 All tests passed! Nunchaku is ready to use.")
    
except Exception as e:
    print(f"❌ Test failed: {e}")
    exit(1)
EOF

print_success "Test script created: test_nunchaku_import.py"

# Print next steps
echo
print_success "🎉 Nunchaku environment setup complete!"
echo
echo "📋 Next steps:"
echo "1. Activate the environment: conda activate nun"
echo "2. Test the installation: python test_nunchaku_import.py"
echo "3. Start the API server: python nunchaku_api_server.py"
echo "4. In another terminal, start the main server: conda activate trellis_new && python trellis_subnit_server_mix_lora_flash_nun.py --port 8096"
echo
echo "🔧 Useful commands:"
echo "  - Check environment: conda env list"
echo "  - Activate: conda activate nun"
echo "  - Deactivate: conda deactivate"
echo "  - Remove environment: conda env remove -n nun"
echo
echo "📚 For more information, see: README_NUNCHAKU_INTEGRATION.md"
echo

# Test the installation
read -p "Do you want to run the test now? (Y/n): " -n 1 -r
echo
if [[ $REPLY =~ ^[Nn]$ ]]; then
    print_status "Skipping test. You can run it later with: python test_nunchaku_import.py"
else
    print_status "Running Nunchaku import test..."
    python test_nunchaku_import.py
fi

print_success "Setup complete! 🚀"
