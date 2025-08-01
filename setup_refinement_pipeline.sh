#!/bin/bash

# Setup script for Trellis GS Refinement Pipeline
# This script installs all necessary dependencies for the refinement pipeline

set -e

echo "🚀 Setting up Trellis GS Refinement Pipeline..."

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

# Check if we're in the right directory
if [ ! -d "TRELLIS" ] || [ ! -d "MV-Adapter" ]; then
    print_error "Please run this script from the root directory containing TRELLIS and MV-Adapter folders"
    exit 1
fi

# Create virtual environment
print_status "Creating Python virtual environment..."
python3 -m venv refinement_env
source refinement_env/bin/activate

# Upgrade pip
print_status "Upgrading pip..."
pip install --upgrade pip

# Install PyTorch (CUDA version)
print_status "Installing PyTorch with CUDA support..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install MV-Adapter dependencies
print_status "Installing MV-Adapter dependencies..."
cd MV-Adapter
pip install -r requirements.txt
cd ..

# Install additional dependencies for refinement pipeline
print_status "Installing refinement pipeline dependencies..."
pip install \
    trimesh \
    plyfile \
    numpy \
    pillow \
    fastapi \
    uvicorn \
    pydantic \
    requests \
    tqdm \
    matplotlib \
    plotly \
    kaleido

# Install validation dependencies
print_status "Installing validation dependencies..."
cd validation
pip install -r requirements.txt
cd ..

# Download MV-Adapter checkpoints
print_status "Setting up MV-Adapter checkpoints..."
mkdir -p checkpoints
cd checkpoints

# Download RealESRGAN upscaler
if [ ! -f "RealESRGAN_x2plus.pth" ]; then
    print_status "Downloading RealESRGAN upscaler..."
    wget https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x2plus.pth
fi

# Download Big-LAMA inpainting model
if [ ! -f "big-lama.pt" ]; then
    print_status "Downloading Big-LAMA inpainting model..."
    wget https://disk.yandex.ru/d/ouP6l8VJ0HpMZg
    mv ouP6l8VJ0HpMZg big-lama.pt
fi

cd ..

# Install SPZ compression library
print_status "Installing SPZ compression library..."
pip install pyspz

# Install additional GS rendering dependencies
print_status "Installing GS rendering dependencies..."
pip install \
    gsplat \
    ninja \
    pytorch3d

# Test imports
print_status "Testing imports..."
python3 -c "
import sys
sys.path.append('./MV-Adapter')
sys.path.append('./TRELLIS')
sys.path.append('./validation')

try:
    from mvadapter.pipelines.pipeline_mvadapter_t2mv_sdxl import MVAdapterT2MVSDXLPipeline
    print('✅ MV-Adapter imports successful')
except ImportError as e:
    print(f'❌ MV-Adapter import failed: {e}')

try:
    from trellis.representations.gaussian.gaussian_model import Gaussian
    print('✅ Trellis imports successful')
except ImportError as e:
    print(f'❌ Trellis import failed: {e}')

try:
    from validation.validation_lib.validation.validation_pipeline import ValidationEngine
    print('✅ Validation imports successful')
except ImportError as e:
    print(f'❌ Validation import failed: {e}')

try:
    import trimesh
    import plyfile
    import pyspz
    print('✅ Utility imports successful')
except ImportError as e:
    print(f'❌ Utility import failed: {e}')
"

# Create test script
print_status "Creating test script..."
cat > test_refinement_setup.py << 'EOF'
#!/usr/bin/env python3
"""
Test script to verify the refinement pipeline setup
"""

import sys
import os

def test_imports():
    """Test all necessary imports"""
    print("Testing imports...")
    
    try:
        import torch
        print(f"✅ PyTorch {torch.__version__}")
    except ImportError as e:
        print(f"❌ PyTorch import failed: {e}")
        return False
    
    try:
        import trimesh
        print(f"✅ Trimesh {trimesh.__version__}")
    except ImportError as e:
        print(f"❌ Trimesh import failed: {e}")
        return False
    
    try:
        import plyfile
        print("✅ Plyfile")
    except ImportError as e:
        print(f"❌ Plyfile import failed: {e}")
        return False
    
    try:
        import pyspz
        print("✅ Pyspz")
    except ImportError as e:
        print(f"❌ Pyspz import failed: {e}")
        return False
    
    # Test MV-Adapter
    sys.path.append('./MV-Adapter')
    try:
        from mvadapter.pipelines.pipeline_mvadapter_t2mv_sdxl import MVAdapterT2MVSDXLPipeline
        print("✅ MV-Adapter pipeline")
    except ImportError as e:
        print(f"❌ MV-Adapter import failed: {e}")
        return False
    
    # Test Trellis
    sys.path.append('./TRELLIS')
    try:
        from trellis.representations.gaussian.gaussian_model import Gaussian
        print("✅ Trellis Gaussian model")
    except ImportError as e:
        print(f"❌ Trellis import failed: {e}")
        return False
    
    # Test validation
    sys.path.append('./validation')
    try:
        from validation.validation_lib.validation.validation_pipeline import ValidationEngine
        print("✅ Validation engine")
    except ImportError as e:
        print(f"❌ Validation import failed: {e}")
        return False
    
    return True

def test_cuda():
    """Test CUDA availability"""
    print("\nTesting CUDA...")
    
    try:
        import torch
        if torch.cuda.is_available():
            print(f"✅ CUDA available: {torch.cuda.get_device_name(0)}")
            print(f"   CUDA version: {torch.version.cuda}")
            print(f"   GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
            return True
        else:
            print("⚠️ CUDA not available - will use CPU (slower)")
            return False
    except Exception as e:
        print(f"❌ CUDA test failed: {e}")
        return False

def test_checkpoints():
    """Test checkpoint availability"""
    print("\nTesting checkpoints...")
    
    checkpoints = [
        "checkpoints/RealESRGAN_x2plus.pth",
        "checkpoints/big-lama.pt"
    ]
    
    all_present = True
    for checkpoint in checkpoints:
        if os.path.exists(checkpoint):
            size_mb = os.path.getsize(checkpoint) / (1024 * 1024)
            print(f"✅ {checkpoint} ({size_mb:.1f} MB)")
        else:
            print(f"❌ {checkpoint} missing")
            all_present = False
    
    return all_present

def main():
    """Run all tests"""
    print("🧪 Testing Trellis GS Refinement Pipeline Setup")
    print("=" * 50)
    
    # Test imports
    imports_ok = test_imports()
    
    # Test CUDA
    cuda_ok = test_cuda()
    
    # Test checkpoints
    checkpoints_ok = test_checkpoints()
    
    # Summary
    print("\n" + "=" * 50)
    print("📊 Setup Test Summary:")
    print(f"   Imports: {'✅' if imports_ok else '❌'}")
    print(f"   CUDA: {'✅' if cuda_ok else '⚠️'}")
    print(f"   Checkpoints: {'✅' if checkpoints_ok else '❌'}")
    
    if imports_ok and checkpoints_ok:
        print("\n🎉 Setup appears successful!")
        print("You can now use the refinement pipeline.")
        if not cuda_ok:
            print("Note: CUDA not available - performance will be slower.")
    else:
        print("\n⚠️ Setup has issues. Please check the errors above.")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
EOF

chmod +x test_refinement_setup.py

# Create activation script
print_status "Creating activation script..."
cat > activate_refinement.sh << 'EOF'
#!/bin/bash
# Activate the refinement environment
source refinement_env/bin/activate
echo "✅ Refinement environment activated"
echo "Run 'python test_refinement_setup.py' to test the setup"
EOF

chmod +x activate_refinement.sh

# Create usage instructions
print_status "Creating usage instructions..."
cat > REFINEMENT_USAGE.md << 'EOF'
# Trellis GS Refinement Pipeline Usage

## Setup

1. **Activate the environment:**
   ```bash
   source activate_refinement.sh
   ```

2. **Test the setup:**
   ```bash
   python test_refinement_setup.py
   ```

## Usage

### Basic Usage

```python
from trellis_refinement_integration import TrellisRefinementGenerator

# Initialize with your Trellis pipeline
enhanced_generator = TrellisRefinementGenerator(
    trellis_pipeline=your_trellis_pipeline,
    mv_adapter_variant="sdxl",
    device="cuda",
    refinement_steps=1000,
    learning_rate=1e-3
)

# Generate with refinement
outputs = enhanced_generator.generate_3d_model(
    prompt="A beautiful red sports car",
    seed=42,
    enable_refinement=True,
    refinement_strength=0.8
)
```

### Integration with Existing Server

1. **Modify your trellis_base_server.py:**
   ```python
   from trellis_refinement_integration import TrellisRefinementGenerator
   
   class TrellisBaseGenerator:
       def __init__(self):
           # ... existing code ...
           self.refinement_generator = None
           self.enable_refinement = True
       
       def _setup_refinement_generator(self):
           if self.trellis_pipeline is not None:
               self.refinement_generator = TrellisRefinementGenerator(
                   trellis_pipeline=self.trellis_pipeline,
                   mv_adapter_variant="sdxl",
                   device="cuda"
               )
       
       def generate_3d_model(self, prompt: str, seed: int = 42):
           # ... existing Trellis generation ...
           
           # Apply refinement if enabled
           if self.refinement_generator is not None and self.enable_refinement:
               refined_outputs = self.refinement_generator.generate_3d_model(
                   prompt=prompt, seed=seed, enable_refinement=True
               )
               gaussian_output = refined_outputs['gaussian']
           else:
               gaussian_output = outputs['gaussian'][0]
           
           # ... rest of existing code ...
   ```

## Pipeline Flow

1. **Trellis Generation:** Generate initial GS and mesh
2. **MV-Adapter Target Generation:** Create high-quality multi-view images
3. **GS Refinement:** Optimize GS attributes to match target images
4. **Output:** Enhanced GS with improved visual quality

## Benefits

- ✅ **Non-redundant:** Uses MV-Adapter for appearance targets, not geometry
- ✅ **Structure-preserving:** Maintains 3D structure from Trellis
- ✅ **Quality-enhancing:** Improves visual quality for better validator scores
- ✅ **Direct optimization:** Optimizes GS attributes directly

## Configuration

- `refinement_steps`: Number of optimization steps (default: 1000)
- `learning_rate`: Learning rate for GS attribute optimization (default: 1e-3)
- `refinement_strength`: Strength of refinement (0.0 to 1.0)
- `mv_adapter_variant`: "sdxl" (better quality) or "sd21" (lower VRAM)

## Troubleshooting

1. **CUDA out of memory:** Reduce `refinement_steps` or use "sd21" variant
2. **Import errors:** Run `python test_refinement_setup.py` to diagnose
3. **Checkpoint missing:** Re-run setup script to download missing files
EOF

print_success "Setup completed successfully!"
print_status "Next steps:"
echo "1. Activate the environment: source activate_refinement.sh"
echo "2. Test the setup: python test_refinement_setup.py"
echo "3. Read usage instructions: cat REFINEMENT_USAGE.md"
echo ""
print_status "The refinement pipeline is now ready to use!" 