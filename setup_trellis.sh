#!/bin/bash

# Exit on error and print commands for debugging
set -e
set -x

# Define the workspace directory explicitly
WORKSPACE_DIR="/home/mbhat"

echo "=== Starting Trellis Setup ==="

# Step 1: Installation and Setup
echo "Creating workspace directory: $WORKSPACE_DIR"
mkdir -p "$WORKSPACE_DIR"
cd "$WORKSPACE_DIR"

echo "Updating and upgrading system packages..."
apt update --yes
apt upgrade --yes

echo "Installing Miniconda..."
wget https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh -O "$WORKSPACE_DIR/miniconda.sh"
bash "$WORKSPACE_DIR/miniconda.sh" -b -p "$WORKSPACE_DIR/miniconda"
rm "$WORKSPACE_DIR/miniconda.sh"

echo "Initializing Conda and cloning the repository..."
source "$WORKSPACE_DIR/miniconda/bin/activate"
git clone https://github.com/Manojbhat09/three-gen-subnet-trellis --recursive
cd three-gen-subnet-trellis
git checkout -b trellis

echo "Installing system dependencies..."
apt-get install -y \
    libgl1-mesa-dev \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    libglew-dev \
    libglfw3-dev \
    libglvnd-dev \
    pkg-config \
    xvfb

echo "Creating and activating Conda environment..."
conda create -n trellis python=3.10 --yes
conda activate trellis

echo "Installing GCC-11 and setting environment variables..."
apt-get install gcc-11 g++-11 --yes
export TORCH_CUDA_ARCH_LIST="8.6"
export CC=/usr/bin/gcc-11
export CXX=/usr/bin/g++-11
export CUDAHOSTCXX=/usr/bin/g++-11
export NVCC_FLAGS="--allow-unsupported-compiler"

echo "Setting CUDA paths..."
export PATH=/usr/local/cuda-12.1/bin${PATH:+:${PATH}}
export LD_LIBRARY_PATH=/usr/local/cuda-12.1/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}

echo "Creating Conda environment from YAML..."
conda env create -f environment_trellis.yml
conda activate trellis_new

echo "Cloning and setting up TRELLIS..."
git clone --recurse-submodules https://github.com/microsoft/TRELLIS.git
cd "$WORKSPACE_DIR/three-gen-subnet-trellis/TRELLIS"
./setup.sh --basic --xformers --flash-attn --diffoctreerast --spconv --mipgaussian --kaolin --nvdiffrast
cd ..

echo "Installing requirements for TRELLIS-TextoImagen3D..."
cd "$WORKSPACE_DIR/three-gen-subnet-trellis/TRELLIS-TextoImagen3D"
pip install -r requirements.txt
cd ..

# Step 2: Run the server and mining script in tmux
echo "Setting up tmux session for server and mining script..."
tmux new-session -d -s trellis_session

# Pane 0: Run the server
echo "Starting server in tmux pane 0..."
tmux send-keys -t trellis_session.0 "source $WORKSPACE_DIR/miniconda/bin/activate && conda activate trellis_new && python $WORKSPACE_DIR/three-gen-subnet-trellis/trellis_submit_server.py" C-m

# Pane 1: Run the mining script
echo "Starting mining script in tmux pane 1..."
tmux split-window -v -t trellis_session
tmux send-keys -t trellis_session.1 "source $WORKSPACE_DIR/miniconda/bin/activate && conda activate trellis_new && chmod +x $WORKSPACE_DIR/three-gen-subnet-trellis/run_trellis_mining.sh && bash $WORKSPACE_DIR/three-gen-subnet-trellis/run_trellis_mining.sh" C-m

echo "Attaching to tmux session..."
tmux attach-session -t trellis_session

echo "=== Setup Complete ==="
