<div align="center">

# **THREE GEN | SUBNET 17**

[![Discord Chat](https://img.shields.io/discord/308323056592486420.svg)](https://discord.gg/bittensor)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT) 

</div>

3D generation subnet provides a platform to democratize 3D content creation, ultimately allowing anyone to create virtual worlds, games and AR/VR/XR experiences. This subnet leverages the existing fragmented and diverse landscape of Open Source 3D generative models ranging from Gaussian Splatting, Neural Radiance Fields, 3D Diffusion Models and Point-Cloud approaches to facilitate innovation - ideal for decentralized incentive-based networks via Bittensor. This subnet aims to kickstart the next revolution in gaming around AI native games, ultimately leveraging the broader Bittensor ecosystem to facilitate experiences in which assets, voice and sound are all generated at runtime. This would effectively allow a creative individual without any coding or game-dev experience to simply describe the game they want to create and have it manifested before them in real time.

---
## Table of Content
1. [Project Structure](#project-structure)
2. [Hardware Requirements](#hardware-requirements)
3. [OS Requirements](#os-requirements)
4. [Setup Guidelines for Miners and Validators](#setup-guidelines-for-miners-and-validators)
   1. [Environment Management With Conda](#environment-management-with-conda)
   2. [Process Supervision With PM2](#process-supervision-with-pm2)
5. [Running the Miner](#running-the-miner)
   1. [Generation Endpoint](#generation-endpoints)
   2. [Miner Neuron](#miner-neuron)
6. [Running the Validator](#running-the-validator)
   1. [Validation Endpoint](#validation-endpoint)
   2. [Validation Neuron](#validator-neuron)
7. [Prompt Generation](#prompt-generation)
   1. [Prompt Generators](#prompt-generators)
   2. [Prompt Collector](#prompt-collector)

---
## Project Structure

The project is divided into three key modules, each designed to perform specific tasks within our 3D content generation and validation framework:

- Generation Module(`generation`): Central to 3D content creation, compatible with miner neurons but can also be used independently for development and testing.

- Neurons Module (`neurons`): This module contains the neuron entrypoints for miners and validators. Miners call the RPC endpoints in the `mining` module to generate images. Validators retrieve and validate generated images. This module handles running the Bittensor subnet protocols and workflows.

- Validation Module (`validation`): Dedicated to ensuring the quality and integrity of 3D content. Like the mining module, it is designed for tandem operation with validator neurons or standalone use for thorough testing and quality checks.

## Hardware Requirements

Our recommended setup aligns with RunPod 1 x RTX 4090 specs:
- GPU: NVIDIA 1 x RTX 4090
- CPU: 16 vCPU
- RAM: 62 GB
- Storage: 50GB SSD

Minimal setup aligns with RunPod 1 x L4 specs:
- GPU: NVIDIA 1 x L4
- CPU: 12 vCPU
- RAM: 62 GB

## OS Requirements

Our code is compatible across various operating systems, yet it has undergone most of its testing on Debian 11, Ubuntu 20 and Arch Linux. 
The most rigorous testing environment used is the Deep Learning VM Image, which includes pre-installed ML frameworks and tools essential for development.

**NOTE**: the linux image should come with **pytorch 2.1+** and **CUDA 12.1.1** otherwise you might have problems with running miner or validator pipelines.

## Setup Guidelines for Miners and Validators

### Environment Management With Conda

For optimal environment setup:
- Prefer [Conda](https://docs.conda.io/en/latest/) for handling dependencies and isolating environments. It's straightforward and efficient for project setup.
- If Conda isn't viable, fallback to manual installations guided by `conda_env_*.yml` files for package details, and use `requirements.txt`. Utilizing a virtual environment is highly advised for dependency management.

### Process Supervision With PM2

To manage application processes:
- Adopt [PM2](https://pm2.io) for benefits like auto-restarts, load balancing, and detailed monitoring. Setup scripts provide PM2 configuration templates for initial use. Modify these templates according to your setup needs before starting your processes.
- If PM2 is incompatible with your setup, but you're using [Conda](https://docs.conda.io/en/latest/), remember to activate the Conda environment first or specify the correct Python interpreter before executing any scripts.

## Running the Miner

By running a miner on this subnet you agree that you have obtained all licenses, rights and consents required to use, reproduce, modify, display, distribute and make available your submitted results to this subnet and its end users.

To operate the miner, the miner neuron and generation endpoints must be initiated. While currently supporting a single generation endpoint, future updates are intended to allow a miner to utilize multiple generation endpoints simultaneously.

### Generation Endpoints

Set up the environment by navigating to the directory and running the setup script:
```commandline
cd three-gen-subnet/generation
./setup_env.sh
```
This script creates a Conda environment `three-gen-mining`, installs dependencies, and sets up a PM2 configuration file (`generation.config.js`).

After optional modifications to `generation.config.js`, initiate it using [PM2](https://pm2.io):
```commandline
pm2 start generation.config.js
```

To verify the endpoint's functionality generate a test video:
```commandline
curl -d "prompt=pink bicycle" -X POST http://127.0.0.1:8093/generate_video/ > video.mp4
```

### Miner Neuron

#### Prerequisites

Ensure wallet registration as per the [official bittensor guide](https://docs.bittensor.com/subnets/register-validate-mine).

#### Setup
Prepare the neuron by executing the setup script in the `neurons` directory:
```commandline
cd three-gen-subnet/neurons
./setup_env.sh
```
This script generates a Conda environment `three-gen-neurons`, installs required dependencies, and prepares `miner.config.js` for PM2 configuration.

#### Running
Update `miner.config.js` with wallet information and ports, then execute with [PM2](https://pm2.io):
```commandline
pm2 start miner.config.js
```

## Running the Validator

Key Aspects of Operating a Validator:
1. A validator requires the operation of a validation endpoint. This endpoint functions as an independent local web server, which operates concurrently with the neuron process.
2. The validator must serve an axon, enabling miners to retrieve tasks and submit their results.

### Validation Endpoint

Set up the environment by navigating to the directory and running the setup script:
```commandline
cd three-gen-subnet/validation
./setup_env.sh
```
This script creates a Conda environment `three-gen-validation`, installs dependencies, and sets up a PM2 configuration file (`validation.config.js`).

After optional modifications to `validation.config.js`, initiate it using [PM2](https://pm2.io):
```commandline
pm2 start validation.config.js
```

**Security considerations:** Run validation endpoint behind the firewall (close the validation endpoint port).

### Validator Neuron

Ensure wallet registration as per the [official bittensor guide](https://docs.bittensor.com/subnets/register-validate-mine).

Prepare the neuron by executing the setup script in the `neurons` directory:
```commandline
cd three-gen-subnet/neurons
./setup_env.sh
```
This script generates a Conda environment `three-gen-neurons`, installs required dependencies, and prepares `validator.config.js` for PM2 configuration.

Update `validator.config.js` with wallet information and ports, then execute with [PM2](https://pm2.io):
```commandline
pm2 start validator.config.js
```

#### Important
Validator must serve the axon and the port must be opened. You can check the port using `nc`. 
```commandline
nc -vz [Your Validator IP] [Port]
```
You can also test the validator using the mock script. Navigate to the `mocks` folder and run
```commandline
PYTHONPATH=$PWD/.. python mock_miner.py --subtensor.network finney --netuid 17 --wallet.name default --wallet.hotkey default --logging.trace
```

## Prompt Generation

Our subnet supports prompt generation from two main sources: organic traffic via Public API 
and continuously updated datasets. By default, it regularly fetches new batches of prompts from our service. 
For real-time prompt generation, we currently utilize two different LLM models: 
`llama3-8b-instruct` and `gemma-1.1-7b-instruct`.

To ensure suitability for 3D generation, our system employs a carefully tailored input 
[prompt-instruction](https://github.com/404-Repo/text-prompt-generator/blob/LLM1_online_prompt_generator/launching_config.yml). 
This instruction forces the LLM to select objects for prompt generation from one of the 13 object categories identified 
based on our industry knowledge and research of gaming asset store trends. 
These selections can be updated in the future to better align with more specific datasets or marketplace curation.

To achieve true decentralization, you can switch to running the prompt generation locally and change the 
`--dataset.prompter.endpoint` parameter. 

Our prompter solution consists of two services: the generator and the collector.

### Prompt Generators

Multiple instances of prompt generators continuously produce small batches of prompts and send them to the 
collector service. You can and should launch multiple generator services to maintain a robust and dynamic system.

To set up the prompt generators:
- Generate an API key for the collector service.
- Configure the prompt generators to send batches of prompts to the collector using this API key.

For more details and to get started with prompt generators, visit the following URL:
- [Prompt Generators Repository](https://github.com/404-Repo/text-prompt-generator)

### Prompt Collector

The prompt collector accumulates prompts from multiple generators and serves fresh large batches of prompts to 
validators upon request. Validators fetch these batches every hour by default, but this interval can be customized.

To set up the prompt collector:
- Use the same API key generated for the prompt generators.
- Configure firewall rules to secure the collector service.

For more details and to get started with the prompt collector, visit the following URL:
- [Prompt Collector Repository](https://github.com/404-Repo/get-prompts)

# Text-to-3D Generation with TRELLIS

This repository contains scripts for generating 3D models from text descriptions using TRELLIS.

## Setup Instructions

### Prerequisites

1. **Hardware Requirements**:
   - NVIDIA GPU with CUDA support
   - Minimum 16GB GPU RAM recommended
   - Minimum 32GB System RAM recommended

2. **CUDA Requirements**:
   - CUDA 11.8 or CUDA 12.1
   - NVIDIA drivers version >= 525.60.13

3. **System Dependencies** (Ubuntu/Debian):
```bash
sudo apt-get update
sudo apt-get install -y \
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
```

There are three ways to set up the environment:

### 1: Using Environment YML Files (Recommended)

We provide three separate conda environments for different components:

1. **TRELLIS Environment** (for text-to-3D generation):
```bash
conda env create -f environment_trellis.yml
conda activate trellis_new
```

2. **Mining Environment** (for running generation endpoints):
```bash
conda env create -f environment_mining.yml
conda activate three-gen-mining
```

3. **Neurons Environment** (for running validator/miner nodes):
```bash
conda env create -f environment_neurons.yml
conda activate three-gen-neurons
```

### 2: Using Requirements.txt

1. Create and activate conda environment:
```bash
conda create -n trellis python=3.10
conda activate trellis
```

2. Install required conda packages:
```bash
# Install PyTorch with CUDA support
conda install -c pytorch pytorch==2.4.0 torchvision==0.19.0 pytorch-cuda=11.8 -c pytorch -c nvidia

# Install required compilers
conda install -c conda-forge gcc=12 gxx=12

# Install Kaolin
git clone --recursive https://github.com/NVIDIAGameWorks/kaolin
cd kaolin
git checkout v0.17.0  # Match your desired version
pip install -r tools/requirements.txt
python setup.py develop
cd ..

# Install nvdiffrast
git clone https://github.com/NVlabs/nvdiffrast
cd nvdiffrast
pip install ninja
pip install .
```
or alternatively for nvdiffrast:
```bash
pip install git+https://github.com/NVlabs/nvdiffrast.git
```

3. Set compiler environment variables:
```bash
export CC=$(which gcc)
export CXX=$(which g++)
```

4. Install Python packages:
```bash
pip install -r requirements.txt
```

5. Install special dependencies:
```bash
# Install mip-splatting and diff-gaussian-rasterization
git clone https://github.com/autonomousvision/mip-splatting --recursive /tmp/extensions/mip-splatting/
pip install /tmp/extensions/mip-splatting/submodules/diff-gaussian-rasterization/
```

### Next steps:

1. Activate the environment:
```bash
conda activate trellis_new
```

2. Clone TRELLIS repository:
```bash
git clone --recurse-submodules https://github.com/microsoft/TRELLIS.git
cd TRELLIS
```

3. Run the setup script with required flags:
```bash
./setup.sh --basic --xformers --flash-attn --diffoctreerast --spconv --mipgaussian --kaolin --nvdiffrast
```

## Environment Variables

Required environment variables:
```bash
export SPCONV_ALGO=native
export ATTN_BACKEND=xformers
export CC=$(which gcc)
export CXX=$(which g++)
```
```
export PATH=/usr/local/cuda/bin${PATH:+:${PATH}}
export LD_LIBRARY_PATH=/usr/local/cuda/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}
```

Optional environment variables:
```bash
export CUDA_VISIBLE_DEVICES=0  # Specify GPU device
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512  # Help with OOM errors
```

## Usage

Run the generation script:
```bash
python test_trellis.py
```

The script will generate a 3D model from the prompt "a blue monkey sitting on temple" and save the outputs in the following formats:
- PLY file (3D Gaussians): `outputs/blue_monkey_gaussian.ply`
- GLB file (Textured mesh): `outputs/blue_monkey.glb`
- Preview videos:
  - `outputs/preview_gaussian.mp4` (Gaussian splatting preview)
  - `outputs/preview_rf.mp4` (Radiance field preview)
  - `outputs/preview_mesh.mp4` (Mesh preview)

## Troubleshooting

### Common Issues

1. **CUDA Version Mismatch**
   ```
   RuntimeError: CUDA error: no kernel image is available for execution on the device
   ```
   Solution: Make sure your NVIDIA drivers support your CUDA version. For CUDA 11.8:
   ```bash
   # Check NVIDIA driver version
   nvidia-smi
   # If needed, install correct driver
   sudo apt install nvidia-driver-525
   ```

2. **OpenGL Issues**
   ```
   ImportError: libGL.so.1: cannot open shared object file
   ```
   Solution: Reinstall OpenGL libraries:
   ```bash
   sudo apt-get install --reinstall libgl1-mesa-glx
   ```

3. **Kaolin Installation Issues**
   If Kaolin fails to install, try:
   ```bash
   conda install -c conda-forge pytorch-cuda=11.8
   pip uninstall kaolin
   pip install kaolin==0.13.0
   ```

4. **Flash Attention Issues**
   If you encounter flash attention errors, set this environment variable:
   ```bash
   export ATTN_BACKEND=xformers
   ```

### Version Compatibility Matrix

| Component | Version | Compatible With |
|-----------|---------|-----------------|
| PyTorch   | 2.4.0   | CUDA 11.8/12.1 |
| Kaolin    | 0.13.0  | PyTorch 2.4.0  |
| nvdiffrast| latest  | CUDA 11.8/12.1 |
| gcc/g++   | 12.x    | All            |

## Monitoring

Monitor GPU usage during generation:
```bash
watch -n 1 nvidia-smi
```

Monitor CPU and memory:
```bash
htop
```

## Files to Save

When backing up or moving to a new instance, save these files:
1. Custom scripts:
   - `test_trellis.py`
   - `serve_trellis.py`
2. Environment files:
   - `environment_trellis.yml`
   - `environment_mining.yml`
   - `environment_neurons.yml`
   - `requirements.txt`
   - `setup_requirements.txt`
3. Any modified configuration files

The conda environments themselves should be recreated rather than backed up.



## Setup 3D topia XL


```
# install dependencies
conda create -n primx python=3.9
```



```
wget https://developer.download.nvidia.com/compute/cuda/11.8.0/local_installers/cuda_11.8.0_520.61.05_linux.run
sh cuda_11.8.0_520.61.05_linux.run --override --silent --toolkit 
export PATH=/usr/local/cuda-11.8/bin${PATH:+:${PATH}}
export LD_LIBRARY_PATH=/usr/local/cuda-11.8/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}
source /home/mbhat/miniconda/bin/activate
conda activate primx
nvcc --version
```

```
conda install pytorch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 pytorch-cuda=11.8 -c pytorch -c nvidia
# requires xformer for efficient attention
conda install xformers::xformers
cd 3DTopia-XL
# install other dependencies
pip install -r requirements.txt
```

```
pip install git+https://github.com/NVlabs/nvdiffrast
pip install "numpy<2"
pip install onnxruntime
```

```
export CUDA_LAUNCH_BLOCKING=1  # Synchronize CUDA errors
export TORCH_USE_CUDA_DSA=1    # Enable device-side assertions
apt-get install libeigen3-dev  # For Ubuntu/Debian
git clone https://github.com/ashawkey/cubvh
```
If needed:
Then modify cubvh/setup.py to use system Eigen headers instead of the submodule:
```
# kill -9 35563 && rm /var/lib/dpkg/lock-frontend /var/lib/apt/lists/lock /var/cache/apt/archives/lock /var/lib/dpkg/lock
# In setup.py, change:
include_dirs=[
    os.path.join(_src_path, 'include'),
    os.path.join(_src_path, 'third_party', 'eigen'),  # ← Remove this line
    '/usr/include/eigen3',  # ← Add this line
],
```

Then:
```
conda install -c fvcore -c iopath -c conda-forge fvcore iopath
conda install ninja  # Critical for compiling extensions
```

Then:
```
apt-get update && apt-get install -y libeigen3-dev
cd cubvh
git submodule update --init --recursive
rm -rf third_party/eigen
git clone https://gitlab.com/libeigen/eigen.git third_party/eigen
cd third_party/eigen
git checkout 3.4.0  # Use a stable version
cd ../..

# Copy eigen headers to torch include directory
mkdir -p /home/mbhat/miniconda/envs/3dtopia/lib/python3.9/site-packages/torch/include/eigen
cp -r third_party/eigen/Eigen /home/mbhat/miniconda/envs/3dtopia/lib/python3.9/site-packages/torch/include/eigen/

# Install with correct CUDA architecture flags
# For RTX 4090 (8.9), RTX 3090 (8.6), or A6000 (8.6)
export TORCH_CUDA_ARCH_LIST="8.6"  # Use 8.6 for most recent NVIDIA GPUs
export FORCE_CUDA=1
pip install . --verbose
```

If you encounter compilation errors, try these troubleshooting steps:

1. Check your GPU architecture:
```bash
nvidia-smi
```

2. Set the appropriate architecture flag based on your GPU:
- RTX 4090: "8.9"
- RTX 3090/A6000: "8.6"
- RTX 2080 Ti: "7.5"
- RTX 1080 Ti: "6.1"

3. If the installation still fails, you can use pysdf as an alternative:
```bash
pip install pysdf
```

4. For Eigen-related errors, try this alternative approach:
```bash
# Install Eigen from conda
conda install -c conda-forge eigen

# Then try installing cubvh again with specific flags
export EIGEN_INCLUDE_DIR=/home/mbhat/miniconda/envs/3dtopia/include/eigen3
export FORCE_CUDA=1
pip install . --verbose
```

then the main:
```
cd 3DTopia-XL
bash install.sh
```

download important files:
```
mkdir pretrained && cd pretrained
# download DiT
wget https://huggingface.co/FrozenBurning/3DTopia-XL/resolve/main/model_sview_dit_fp16.pt
# download VAE
wget https://huggingface.co/FrozenBurning/3DTopia-XL/resolve/main/model_vae_fp16.pt
mkdir pretrained
mv model_* pretrained/
cd ..
```

install inference files:
```
export CUB_HOME=/home/mbhat/miniconda/envs/primx/include/cub
conda install pytorch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 pytorch-cuda=12.1 -c pytorch -c nvidia
conda install -c iopath iopath
conda install -c bottler nvidiacub
conda install jupyter
pip install scikit-image matplotlib imageio plotly opencv-python

git clone --recursive https://github.com/facebookresearch/pytorch3d

#Normal:
cd pytorch3d
python setup.py clean
python setup.py install  # Use --user if not in a virtual env

#For A6000:
CUB_HOME=/home/mbhat/miniconda/envs/primx/include/cub FORCE_CUDA=1 TORCH_CUDA_ARCH_LIST="8.6" pip install -e .

#or for all together:
cd /home/mbhat/three-gen-subnet-trellis/3DTopia-XL/pytorch3d && CUDA_HOME=/usr/local/cuda-11.8 CUB_HOME=/home/mbhat/miniconda/envs/primx/include/cub FORCE_CUDA=1 TORCH_CUDA_ARCH_LIST="8.6" pip install -e .
```

Run inference:
```
python inference.py ./configs/inference_dit.yml
```

## Setup with DetailGen

```
git clone https://github.com/VAST-AI-Research/DetailGen3D --recursive
conda create -n trellis_detail python=3.10 numpy=1.24.3 scipy=1.10.1 -y
conda activate trellis_detail && pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118 && pip install -r requirements.txt && pip install -r DetailGen3D/requirements.txt
wget https://developer.download.nvidia.com/compute/cuda/11.8.0/local_installers/cuda_11.8.0_520.61.05_linux.run
sh cuda_11.8.0_520.61.05_linux.run --override --silent  --toolkit 
export CC=/usr/bin/gcc-9
export CXX=/usr/bin/g++-9
export CUDA_HOME=/usr/local/cuda-11.8
export TORCH_CUDA_ARCH_LIST="8.9;9.0;8.6"
export PATH=/usr/local/cuda-11.8/bin${PATH:+:${PATH}}
export LD_LIBRARY_PATH=/usr/local/cuda-11.8/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}
pip install numpy==1.24
conda install pytorch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1  pytorch-cuda=11.8 -c pytorch -c nvidia
git clone --recursive https://github.com/NVIDIAGameWorks/kaolin
cd kaolin
pip install Cython>=0.29.37
pip install -r tools/requirements.txt
pip install scipy==1.15.3
python setup.py develop

pip install ninja
 pip install git+https://github.com/NVlabs/nvdiffrast
pip install .
cd .
pip install /tmp/extensions/mip-splatting/submodules/diff-gaussian-rasterization/
pip uninstall -y torch-cluster
nvcc --version
pip uninstall torch-cluster
FORCE_CUDA=1 pip install --no-cache-dir --verbose torch-cluster
cd TRELLIS/
./setup.sh --basic --xformers --flash-attn --diffoctreerast --spconv --mipgaussian
cd ..
pip install huggingface-hub
huggingface-cli login
export SPCONV_ALGO=native
export ATTN_BACKEND=xformers
export CC=$(which gcc)
export CXX=$(which g++)
pip install accelerate
huggingface-cli download VAST-AI/DetailGen3D --local-dir detailgen3d --local-dir-use-symlinks False
cp DetailGen3D/detailgen3d detailgen3d/ -r
python test_trellis_detail.py 

pip uninstall -y numpy scipy trimesh scikit-image && pip install numpy==1.24.3 && pip install scipy==1.10.1 && pip install trimesh scikit-image
```


## Setup for 3Dtopia:



1. Test the first stage of repo:
```bash
conda env create -f requirements.yml
conda activate 3dtopia-first
python -u sample_stage1.py --text "a robot" --samples 1 --sampler ddim --steps 200 --cfg_scale 7.5 --seed 0
```


2. Run the second stage:
```
 source /home/mbhat/miniconda/bin/activate
 conda create -n 3dtopia-second python=3.9 -y
 
 wget https://developer.download.nvidia.com/compute/cuda/11.8.0/local_installers/cuda_11.8.0_520.61.05_linux.run
sh cuda_11.8.0_520.61.05_linux.run --override --silent  --toolkit 


export CC=/usr/bin/gcc-9
export CXX=/usr/bin/g++-9
export CUDA_HOME=/usr/local/cuda-11.8
export TORCH_CUDA_ARCH_LIST="8.9;9.0;8.6"
export PATH=/usr/local/cuda-11.8/bin${PATH:+:${PATH}}
export LD_LIBRARY_PATH=/usr/local/cuda-11.8/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}
export TORCH_CUDA_ARCH_LIST="8.6"  # Use 8.6 for most recent NVIDIA GPUs
export FORCE_CUDA=1

mkdir -p /home/mbhat/miniconda/envs/3dtopia-second/bin && ln -s /usr/local/cuda-11.3/bin/nvcc /home/mbhat/miniconda/envs/3dtopia-second/bin/nvcc
 conda install -c conda-forge cudatoolkit=11.8
 conda install -c conda-forge eigen
 pip install pysdf
 conda activate 3dtopia-second
conda install pytorch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1 pytorch-cuda=11.8 -c pytorch -c nvidia

git clone https://github.com/3DTopia/threefiner
cd threefiner
export CUDA_HOME=/usr/local/cuda-11.8 && export PATH=/usr/local/cuda-11.8/bin:$PATH && export LD_LIBRARY_PATH=/usr/local/cuda-11.8/lib64:$LD_LIBRARY_PATH && cd /home/mbhat/three-gen-subnet-trellis/3DTopia/threefiner && pip install .

 pip install git+https://github.com/NVlabs/nvdiffrast
 
pip install --no-cache-dir --verbose git+https://github.com/NVlabs/tiny-cuda-nn/#subdirectory=bindings/torch

apt-get update && apt-get install -y libegl1-mesa-dev

apt-get update && apt-get install -y xvfb && Xvfb :1 -screen 0 1024x768x24 & export DISPLAY=:1 && nvidia-smi -pm 1

threefiner sd --mesh results/default/stage1/a_robot_0_0.ply --prompt "a robot" --text_dir --front_dir='-y' --outdir results/default/stage2/ --save a_robot_0_0_sd.glb --force_cuda_rast

huggingface-cli login
or export HF_TOKEN= 

threefiner if2 --mesh results/default/stage2/a_robot_0_0_sd.glb --prompt "a robot" --outdir results/default/stage2/ --save a_robot_0_0_if2.glb --force_cuda_rast
```

Install trellis-text

```
conda create -n trellis_text python=3.10
conda activate trellis_text                                                                               
conda install pytorch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 pytorch-cuda=12.1 -c pytorch -c nvidia   

export CUDA_HOME=/usr/local/cuda-12.1
export TORCH_CUDA_ARCH_LIST="8.9;9.0;8.6"
export PATH=/usr/local/cuda-12.1/bin${PATH:+:${PATH}}
export LD_LIBRARY_PATH=/usr/local/cuda-11.8/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}
export FORCE_CUDA=1

pip install -r requirements.txt

git clone --recurse-submodules https://github.com/microsoft/TRELLIS.git
cd TRELLIS
cp trellis ../TRELLIS-TextImagen3D
./setup.sh --basic --kaolin
export HF_TOKEN=
huggingface-cli login
```


<!-- 
# # Then try installing cubvh again with specific flags
# export EIGEN_INCLUDE_DIR=/home/mbhat/miniconda/envs/3dtopia/include/eigen3
# export FORCE_CUDA=1
# pip install . --verbose
# ```

# 4. Install the main 3DTopia package:
# ```bash
# cd ..
# cd 3DTopia
# pip install -r requirement.txt
# ```


# 6. Test the second stage:
# ```bash
# # Set CUB home directory
# export CUB_HOME=/home/mbhat/miniconda/envs/3dtopia-second/include/cub

# # Install PyTorch with CUDA 12.1
# conda install pytorch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 pytorch-cuda=12.1 -c pytorch -c nvidia

# # Install additional dependencies
# conda install -c iopath iopath
# conda install -c bottler nvidiacub
# conda install jupyter
# pip install scikit-image matplotlib imageio plotly opencv-python
# ```

# 7. Install PyTorch3D:
# ```bash
# # For A6000 GPUs:
# CUB_HOME=/home/mbhat/miniconda/envs/3dtopia-second/include/cub FORCE_CUDA=1 TORCH_CUDA_ARCH_LIST="8.6" pip install -e .

# # Or for all GPUs:
# cd pytorch3d && CUDA_HOME=/usr/local/cuda-11.3 CUB_HOME=/home/mbhat/miniconda/envs/3dtopia-second/include/cub FORCE_CUDA=1 TORCH_CUDA_ARCH_LIST="8.6" pip install -e .
# ```

# 8. Run inference:
# ```bash
# python inference.py ./configs/inference_dit.yml
# ```
 -->
Note: Make sure all environment variables are properly set and the CUDA toolkit is correctly installed before running inference.

## License

TRELLIS is licensed under [Microsoft Research License](https://github.com/microsoft/TRELLIS/blob/main/LICENSE.txt).

