#!/bin/bash
# Subnet 17 (404-GEN) Mining Environment Setup Script
# This script sets up the complete mining environment for Subnet 17

set -e  # Exit on any error

echo "=========================================="
echo "Subnet 17 Mining Environment Setup"
echo "=========================================="

# Color codes for output
RED='\\033[0;31m'
GREEN='\\033[0;32m'
YELLOW='\\033[1;33m'
NC='\\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if running as root
if [[ $EUID -eq 0 ]]; then
   print_warning "Running as root is not recommended. Continuing, but consider using a non-root user."
fi

# Check system requirements
print_status "Checking system requirements..."

# Check CUDA
if ! command -v nvidia-smi &> /dev/null; then
    print_error "NVIDIA GPU drivers not found. Please install NVIDIA drivers and CUDA toolkit."
    exit 1
fi

# Check Python
if ! command -v python3 &> /dev/null; then
    print_error "Python 3 not found. Please install Python 3.8 or newer."
    exit 1
fi

# Check Python version
PYTHON_VERSION=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
REQUIRED_VERSION="3.10"
if ! python3 -c "import sys; exit(0 if sys.version_info >= (3, 10) else 1)"; then
    print_error "Python 3.10 or newer is required. Found: $PYTHON_VERSION"
    exit 1
fi

print_status "System requirements check passed ✓"

# Setup directories
print_status "Creating project directories..."
mkdir -p logs
mkdir -p mining_outputs
mkdir -p generation_outputs
mkdir -p validation_temp
mkdir -p locally_validated_assets
mkdir -p validation_results

# Check for conda
if ! command -v conda &> /dev/null; then
    print_error "Conda is not installed or not in your PATH. Please install Miniconda/Anaconda."
    exit 1
fi

# Setup conda environment
CONDA_ENV_NAME="hunyuan3d"
print_status "Setting up conda environment '$CONDA_ENV_NAME'..."

if ! conda env list | grep -q "$CONDA_ENV_NAME"; then
    print_status "Creating conda environment '$CONDA_ENV_NAME' with Python 3.10..."
    conda create -n "$CONDA_ENV_NAME" python=3.10 -y
    print_status "Created '$CONDA_ENV_NAME' conda environment"
else
    print_warning "'$CONDA_ENV_NAME' environment already exists. Will install/update packages in it."
fi

# Activate environment and install packages
print_status "Activating environment and installing packages..."
eval "$(conda shell.bash hook)"
conda activate "$CONDA_ENV_NAME"
export CONDA_PREFIX="$CONDA_HOME/envs/$CONDA_ENV_NAME"

# Install Python dependencies
print_status "Installing Python dependencies via pip..."

pip install --upgrade pip
pip install wheel setuptools

# Core mining dependencies
print_status "Installing core mining dependencies..."
pip install bittensor
pip install fastapi uvicorn aiohttp "pydantic<2"
pip install numpy trimesh pyspz Pillow psutil GPUtil
pip install protobuf

# PyTorch installation
print_status "Installing PyTorch..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# T2I and Shape generation dependencies
print_status "Installing 3D generation dependencies..."
pip install "diffusers>=0.29.0.dev0" transformers accelerate bitsandbytes safetensors
pip install "trimesh[easy]"
pip install dataclasses-json

print_status "Core dependencies installed ✓"

# Setup Hunyuan3D-2
print_status "Setting up Hunyuan3D-2..."

if [ ! -d "Hunyuan3D-2" ]; then
    print_error "Hunyuan3D-2 directory not found. Please clone it into the current directory."
    exit 1
fi

cd Hunyuan3D-2

# Install Hunyuan3D-2 dependencies
if [ -f "requirements.txt" ]; then
    print_status "Installing Hunyuan3D-2 requirements..."
    # Install dependencies, excluding torch and diffusers if they conflict
    pip install -r requirements.txt --no-deps
fi

# Install the package
if [ -f "setup.py" ]; then
    print_status "Installing Hunyuan3D-2 package in editable mode..."
    pip install -e .
fi

cd ..

print_status "Hunyuan3D-2 setup complete ✓"

# Download models (if needed)
print_status "Checking model availability..."
print_warning "Note: Models will be downloaded automatically on first run."
print_status "To pre-download, run the servers once."

# Setup configuration files
print_status "Setting up configuration and startup scripts..."

# Create PM2 ecosystem file
PM2_INTERPRETER="conda run -n $CONDA_ENV_NAME --no-capture-output python"
cat > ecosystem.config.js << EOF
module.exports = {
  apps: [
    {
      name: 'generation-server',
      script: 'generation_server.py',
      interpreter: '$PM2_INTERPRETER',
      cwd: process.cwd(),
      env: {
        PYTHONPATH: \`\${process.cwd()}:\${process.cwd()}/Hunyuan3D-2\`
      },
      log_file: './logs/generation-server.log',
      error_file: './logs/generation-server-error.log',
      out_file: './logs/generation-server-out.log',
      time: true
    },
    {
      name: 'validation-server',
      script: 'validation_server.py',
      interpreter: '$PM2_INTERPRETER',
      cwd: process.cwd(),
      env: {
        PYTHONPATH: \`\${process.cwd()}:\${process.cwd()}/Hunyuan3D-2\`
      },
      log_file: './logs/validation-server.log',
      error_file: './logs/validation-server-error.log',
      out_file: './logs/validation-server-out.log',
      time: true
    },
    {
      name: 'subnet17-miner',
      script: 'subnet17_miner.py',
      interpreter: '$PM2_INTERPRETER',
      cwd: process.cwd(),
      env: {
        PYTHONPATH: \`\${process.cwd()}:\${process.cwd()}/Hunyuan3D-2\`
      },
      log_file: './logs/miner.log',
      error_file: './logs/miner-error.log',
      out_file: './logs/miner-out.log',
      time: true
    }
  ]
};
EOF

print_status "PM2 ecosystem file created ✓"

# Create start script
cat > start_mining.sh << EOF
#!/bin/bash
# Start script for Subnet 17 mining environment

echo "Starting Subnet 17 Mining Environment..."

# Activate conda environment
if ! command -v conda &> /dev/null; then
    echo "Conda not found. Cannot start."
    exit 1
fi
eval "\\\$(conda shell.bash hook)"
conda activate "$CONDA_ENV_NAME"
echo "Activated conda environment: $CONDA_ENV_NAME"

# Set Python path
export PYTHONPATH="\\\$PWD:\\\$PWD/Hunyuan3D-2:\\\$PYTHONPATH"
echo "PYTHONPATH set"

# Check if PM2 is available
if ! command -v pm2 &> /dev/null; then
    echo "PM2 not found. Please install it with: npm install -g pm2"
    echo "Then run this script again."
    exit 1
fi

echo "Starting services with PM2..."

# Start servers first
pm2 start ecosystem.config.js --only generation-server,validation-server
echo "Waiting for servers to initialize (this may take a few minutes for model downloads)..."
sleep 15

# Test endpoints
MAX_RETRIES=10
RETRY_COUNT=0
while [ \$RETRY_COUNT -lt \$MAX_RETRIES ]; do
    if curl -s -f http://localhost:8093/health/ > /dev/null && curl -s -f http://localhost:8094/health/ > /dev/null; then
        echo "✓ Generation and Validation servers are healthy"
        break
    fi
    RETRY_COUNT=\$((RETRY_COUNT+1))
    echo "Servers not ready yet. Retrying in 15 seconds... (\$RETRY_COUNT/\$MAX_RETRIES)"
    sleep 15
done

if [ \$RETRY_COUNT -eq \$MAX_RETRIES ]; then
    echo "✗ Servers did not become healthy in time. Please check logs with 'pm2 logs'."
    exit 1
fi

# Start miner
echo "Starting miner..."
pm2 start ecosystem.config.js --only subnet17-miner

echo "All services started!"
echo "Use 'pm2 logs' to view logs"
echo "Use 'pm2 status' to check status"
echo "Use 'pm2 stop all' to stop all services"
EOF

chmod +x start_mining.sh

# Create stop script
cat > stop_mining.sh << EOF
#!/bin/bash
# Stop script for Subnet 17 mining environment

echo "Stopping Subnet 17 Mining Environment..."

if command -v pm2 &> /dev/null; then
    pm2 stop all
    pm2 delete all
    echo "All services stopped and deleted from PM2."
else
    echo "PM2 not found. Please stop services manually."
fi
EOF

chmod +x stop_mining.sh

# Create requirements.txt
cat > requirements.txt << EOF
# Subnet 17 Mining Environment Requirements
# Installed via setup script using pip

# Core
bittensor
fastapi
uvicorn
aiohttp
pydantic<2
numpy
trimesh[easy]
pyspz
Pillow
psutil
GPUtil
protobuf
dataclasses-json

# PyTorch (for CUDA 12.1)
# torch
# torchvision
# torchaudio

# 3D Generation
diffusers>=0.29.0.dev0
transformers
accelerate
bitsandbytes
safetensors
EOF

print_status "Requirements file created ✓"

# Final setup steps
print_status "Finalizing setup..."
chmod +x *.py 2>/dev/null || true

echo ""
print_status "=========================================="
print_status "Subnet 17 Mining Environment Setup Complete!"
print_status "=========================================="
echo ""
echo "Next steps:"
echo "1. Configure your Bittensor wallet (if not done already):"
echo "   btcli wallet new"
echo "   btcli subnet register --netuid 17"
echo ""
echo "2. Update your wallet name in 'subnet17_miner.py':"
echo "   - Edit WALLET_NAME and HOTKEY_NAME variables."
echo ""
echo "3. Start the entire mining stack:"
echo "   ./start_mining.sh"
echo ""
echo "4. Monitor your miner:"
echo "   pm2 logs subnet17-miner --lines 100"
echo "   pm2 status"
echo ""
echo "Useful commands:"
echo "  ./start_mining.sh  - Start all services"
echo "  ./stop_mining.sh   - Stop all services"
echo "  pm2 logs           - View all logs"
echo "  pm2 restart all    - Restart all services"
echo ""
print_warning "Ensure you have registered your hotkey on Netuid 17 before starting!" 