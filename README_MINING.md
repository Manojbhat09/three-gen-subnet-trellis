# Subnet 17 (404-GEN) Mining Environment

A complete and robust mining setup for Subnet 17, using a high-performance, memory-efficient pipeline (Hunyuan3D-2 with Flux) for 3D model generation.

## Overview

This mining environment provides:

- **High-quality 3D model generation** using a Flux-based text-to-image pipeline and Hunyuan3D-2 for shape generation.
- **Memory-efficient pipeline** that loads models on-demand to minimize VRAM usage.
- **Local validation server** to ensure model quality before mainnet submission.
- **Asynchronous, high-throughput mining client** with robust error handling.
- **Comprehensive local testing tools** to run, validate, and compare generations.
- **Automated environment setup** and process management with PM2.

## Architecture

The environment consists of three core services, designed to run concurrently:

```
┌──────────────────┐      ┌─────────────────────┐      ┌──────────────────┐
│ Generation Server│◄───┐ │    Miner Neuron     │ ┌───►│    Validators    │
│ (generates 3D)   │    │ │  (subnet17_miner)   │ │    │   (Bittensor)    │
└──────────────────┘    └─┤ (pulls tasks,       ├─┘    └──────────────────┘
                          │  submits results)   │
┌──────────────────┐    ┌─►└─────────────────────┘◄───┐
│ Validation Server│    │                               │
│ (validates 3D)   │────┘                               │
└──────────────────┘                               (self-validates)
```

## Components

### 1. Generation Server (`generation_server.py`)
- **FastAPI server** using the memory-efficient **Flux** text-to-image pipeline.
- Loads models on-demand to keep idle VRAM usage low.
- Endpoints: `POST /generate/`, `GET /status/`, `GET /health/`.

### 2. Validation Server (`validation_server.py`)
- **Validates PLY files** with a comprehensive quality scoring system.
- Features multi-view rendering, mesh analysis, and preview generation.
- Endpoints: `POST /validate_txt_to_3d_ply/`, `GET /status/`, `GET /health/`.

### 3. Miner Neuron (`subnet17_miner.py`)
- **Connects to Bittensor Subnet 17**, pulls tasks, and submits results.
- **Fully asynchronous** with configurable workers for high throughput.
- **Self-validates** generations locally before submitting to the network.

### 4. Local Testing & Validation Tools
- **`local_validation_runner.py`**: A CLI tool to test the full pipeline. Give it a prompt, and it will generate the model, validate it, and save the results. **Perfect for a quick sanity check.**
- **`local_compete_validation.py`**: A CLI tool to find the best possible model for a prompt. It runs multiple generations with different seeds and returns the one with the highest validation score. **Use this to analyze model quality.**

### 5. Setup & Management Scripts
- `setup_mining_environment.sh`: A comprehensive setup script that creates the conda environment and installs all dependencies.
- `start_mining.sh` / `stop_mining.sh`: Easy-to-use scripts to start and stop all services with PM2.
- `ecosystem.config.js`: PM2 configuration for managing the three core services.

## Quick Start Guide

### Prerequisites
- **NVIDIA GPU** with CUDA 12.1+ drivers (RTX 30-series or newer recommended).
- **Linux OS** (Ubuntu 20.04+ recommended).
- **Miniconda/Anaconda** installed.
- **Node.js and npm** installed (for PM2).
- **Git** installed.

### Installation Steps

1.  **Clone the Repository and `Hunyuan3D-2`**:
    ```bash
    git clone <your-repo-url>
    cd <your-repo-url>
    git clone https://github.com/Tencent/Hunyuan3D-2.git
    ```

2.  **Run the Setup Script**:
    This script will create a `hunyuan3d` conda environment and install all necessary dependencies.
    ```bash
    chmod +x setup_mining_environment.sh
    ./setup_mining_environment.sh
    ```
    *(This may take a while, especially when installing PyTorch).*

3.  **Install PM2**:
    PM2 is a process manager that will keep your miner and servers running.
    ```bash
    npm install -g pm2
    ```

4.  **Configure Your Bittensor Wallet**:
    ```bash
    # If using the new environment, activate it first
    conda activate hunyuan3d
    
    # Create a wallet if you don't have one
    btcli wallet new
    
    # Register your wallet to Subnet 17 (requires TAO)
    btcli subnet register --netuid 17
    ```

5.  **Update Miner Configuration**:
    Open `subnet17_miner.py` and set your wallet details:
    ```python
    # Wallet Configuration
    WALLET_NAME: str = "your_wallet_name"
    HOTKEY_NAME: str = "your_hotkey_name"
    ```

### Running the Miner

1.  **Start all services**:
    ```bash
    ./start_mining.sh
    ```
    This will use PM2 to start the generation server, validation server, and the miner in the background. The first run may be slow as models are downloaded.

2.  **Monitor the services**:
    ```bash
    # View the status of all services
    pm2 status
    
    # View the logs for the miner
    pm2 logs subnet17-miner
    
    # View all logs
    pm2 logs
    ```

3.  **Stop all services**:
    ```bash
    ./stop_mining.sh
    ```

## Using the Local Test Tools

Make sure the services are running (`./start_mining.sh`) before using these tools.

### Quick Sanity Check (`local_validation_runner.py`)
Test a single prompt to see if everything is working correctly.
```bash
# Activate conda environment first
conda activate hunyuan3d

# Run with a prompt
python local_validation_runner.py "a cute cat wearing a wizard hat"

# Run with a specific seed
python local_validation_runner.py "a rusty robot" --seed 12345
```
This will generate, validate, and save the asset in the `locally_validated_assets` directory.

### Finding the Best Model (`local_compete_validation.py`)
Run a competition to find the best seed for a given prompt.
```bash
# Activate conda environment first
conda activate hunyuan3d

# Run a competition with 5 different seeds (default)
python local_compete_validation.py "a detailed fantasy sword"

# Run a competition with 10 seeds
python local_compete_validation.py "an ancient tree with glowing runes" -n 10
```
This will create a new directory in `competition_results` containing the best model and a summary of all runs.

## Troubleshooting

- **Conda environment issues**: Make sure `conda activate hunyuan3d` is active in your terminal before running any python scripts manually.
- **`pm2` not found**: Ensure you have installed PM2 globally with `npm install -g pm2`.
- **CUDA Out of Memory**: The on-demand pipeline should prevent this, but if it occurs, try restarting the services (`pm2 restart all`). If it persists, your GPU may not have enough VRAM for the models.
- **Servers don't start**: Check the logs with `pm2 logs generation-server` and `pm2 logs validation-server`. The most common issue is a failure during model download or a missing dependency.

## Configuration

### Miner Configuration (`subnet17_miner.py`)

```python
# Wallet Configuration
WALLET_NAME: str = "default"        # Your wallet name
HOTKEY_NAME: str = "default"        # Your hotkey name

# Performance Tuning
MAX_CONCURRENT_GENERATIONS: int = 2    # Concurrent generation tasks
SELF_VALIDATION_MIN_SCORE: float = 0.7 # Minimum score for submission
TARGET_GENERATION_TIME: float = 5.0    # Target generation time (seconds)

# Resource Management
NUM_GENERATION_WORKERS: int = 2        # Number of generation workers
NUM_SUBMISSION_WORKERS: int = 3        # Number of submission workers
```

### Generation Server Configuration

```python
# Model Configuration
GENERATION_CONFIG = {
    'model_path': 'jetx/Hunyuan3D-2',
    'num_inference_steps': 30,
    'mc_algo': 'mc',
    'device': 'cuda'
}
```

## Performance Optimization

### Target Metrics
- **Generation Time**: < 5 seconds
- **Quality Score**: > 0.8 average
- **Acceptance Rate**: > 90%
- **Uptime**: > 99%

### Optimization Tips

1. **GPU Memory Management**:
   - Use `torch.cuda.empty_cache()` after each generation
   - Monitor VRAM usage with `/status/` endpoints
   - Consider reducing batch sizes if OOM occurs

2. **Generation Speed**:
   - Use fewer inference steps (20-30) for speed
   - Optimize prompt preprocessing
   - Cache commonly used models

3. **Quality Improvement**:
   - Implement local validation thresholds
   - Use higher inference steps for better quality
   - Fine-tune generation parameters

4. **Resource Usage**:
   - Use multiple generation endpoints for scaling
   - Balance worker counts based on hardware
   - Monitor queue sizes and adjust accordingly

## Monitoring and Maintenance

### Service Management

```bash
# Check status
pm2 status

# View logs
pm2 logs                    # All logs
pm2 logs subnet17-miner     # Miner logs only
pm2 logs generation-server  # Generation server logs

# Restart services
pm2 restart all            # Restart all
pm2 restart subnet17-miner # Restart miner only

# Stop services
./stop_mining.sh
```

### Health Checks

```bash
# Check generation server
curl http://localhost:8093/health/
curl http://localhost:8093/status/

# Check validation server
curl http://localhost:8094/health/
curl http://localhost:8094/status/
```

### Performance Monitoring

The miner automatically logs performance metrics every 5 minutes:
- Total tasks processed
- Generation success rate
- Average generation time
- Average local validation score
- Last reward received

## Troubleshooting

### Common Issues

1. **CUDA Out of Memory**:
   ```bash
   # Reduce concurrent generations
   # Edit subnet17_miner.py: MAX_CONCURRENT_GENERATIONS = 1
   ```

2. **Generation Server Not Starting**:
   ```bash
   # Check CUDA availability
   python -c "import torch; print(torch.cuda.is_available())"
   
   # Check Hunyuan3D-2 installation
   python -c "from Hunyuan3D_2.hy3dgen import *"
   ```

3. **Low Validation Scores**:
   ```bash
   # Test local validation
   python local_validation_runner.py
   
   # Compare with other models
   python local_compete_validation.py
   ```

4. **Bittensor Connection Issues**:
   ```bash
   # Check wallet registration
   btcli wallet overview --netuid 17
   
   # Check network connectivity
   btcli network list
   ```

### Debug Mode

Enable debug logging by editing the scripts:
```python
bt.logging(debug=True, trace=True)
```

## Hardware Requirements

### Minimum Setup
- **GPU**: NVIDIA L4 (24GB VRAM)
- **CPU**: 12 vCPU cores
- **RAM**: 32GB
- **Storage**: 100GB SSD
- **Network**: Stable internet with low latency

### Recommended Setup
- **GPU**: NVIDIA RTX 4090 (24GB VRAM)
- **CPU**: 16+ vCPU cores
- **RAM**: 64GB
- **Storage**: 500GB NVMe SSD
- **Network**: High-speed internet (100+ Mbps)

### Optimal Setup
- **GPU**: 2x NVIDIA RTX 4090
- **CPU**: AMD Ryzen 9 7950X or Intel i9-13900K
- **RAM**: 128GB DDR5
- **Storage**: 1TB NVMe SSD
- **Network**: Dedicated server with 1Gbps+ connection



## 📋 **Comprehensive Robustness Improvements Summary**

I've created a significantly enhanced and robust mining system for Subnet 17. Here are the key improvements you should be aware of:

### **🛡️ Critical Robustness Features Added:**

#### **1. Enhanced Generation Server (`robust_generation_server.py`)**
- **Circuit Breaker Pattern**: Automatically stops accepting requests after consecutive failures and recovers gradually
- **Memory Pressure Management**: Monitors VRAM usage and performs cleanup at 90% threshold
- **Health Monitoring**: Comprehensive health endpoints with detailed system metrics
- **Auto-Recovery**: Automatically restarts pipeline after OOM errors
- **Graceful Degradation**: Continues operating in degraded mode rather than failing completely

#### **2. Performance Optimization (`performance_optimizer.py`)**
- **Adaptive Configuration**: Automatically adjusts generation parameters based on performance
- **Competitive Analysis**: Analyzes market conditions and optimizes strategy accordingly
- **Performance Profiles**: Pre-defined settings for different optimization goals (speed vs quality)
- **Benchmarking**: Regular performance testing with automatic profile switching

#### **3. Enhanced Mining Client (`robust_subnet17_miner.py`)**
- **Intelligent Task Prioritization**: Prioritizes validators based on stake, success rate, and response time
- **Competitive Scoring**: Analyzes competitor performance to optimize strategy
- **Advanced Error Recovery**: Multiple retry strategies with exponential backoff
- **Validator Performance Tracking**: Maintains detailed metrics on all validators

#### **4. Process Orchestration (`mining_orchestrator.py`)**
- **Centralized Process Management**: Manages all components from a single controller
- **Health Monitoring**: Continuous health checks with automatic restarts
- **Graceful Shutdown**: Proper cleanup and state preservation
- **Resource Monitoring**: Tracks CPU, memory, and disk usage
- **Configuration Management**: Persistent configuration with hot-reloading

### **⚠️ Things You Need to Take Care Of:**

#### **1. System Resource Management**
```bash
# Monitor these continuously:
- VRAM usage (should stay below 90%)
- Disk space (keep 10GB+ free for generations)
- CPU load (high CPU can slow generation)
- Network stability (critical for Bittensor communication)
```

#### **2. Model Updates and Compatibility**
```bash
# Ensure Hunyuan3D-2 models are updated:
- Check for model updates weekly
- Test compatibility after updates
- Backup working model versions
```

#### **3. Competitive Monitoring**
```bash
# Monitor competitor performance:
- Track validation scores vs market average
- Adjust quality thresholds based on competition
- Monitor generation time vs competitors
```

#### **4. Network and Validator Health**
```bash
# Critical validator interactions:
- Monitor validator response times
- Track validator stake changes
- Avoid validators with poor success rates
- Implement validator blacklisting for persistent failures
```

### **🚀 How to Run the Enhanced System:**

#### **1. Complete Setup**
```bash
# Run the enhanced setup
chmod +x setup_mining_environment.sh
./setup_mining_environment.sh

# Activate the correct environment
source /home/mbhat/miniconda/bin/activate
conda activate hunyuan3d
```

#### **2. Start the Mining Environment**
```bash
# Start everything with orchestrator
python mining_orchestrator.py run

# Or start individual components
python mining_orchestrator.py start --process validation_server
python mining_orchestrator.py start --process generation_server
python mining_orchestrator.py start --process subnet17_miner
```

#### **3. Monitor Performance**
```bash
# Check system status
python mining_orchestrator.py status

# Monitor logs
tail -f mining_orchestrator.log

# Check performance optimization
python performance_optimizer.py
```

#### **4. Testing and Validation**
```bash
# Test local generation and validation
python local_validation_runner.py "A red sports car"

# Run competitive analysis
python local_compete_validation.py "A wooden chair" --num_variants 5
```

### **🔧 Key Configuration Points:**

#### **1. Performance Tuning**
- Adjust `max_concurrent_requests` in generation server based on VRAM
- Set `memory_cleanup_threshold` appropriately for your GPU
- Configure `competitive_score_threshold` based on network performance

#### **2. Error Handling**
- Set appropriate `max_restarts` limits for each process
- Configure `circuit_breaker_failure_threshold` for stability
- Adjust timeouts based on your system performance

#### **3. Quality vs Speed Trade-offs**
- Monitor `market_competitiveness` and adjust accordingly
- Set quality thresholds that balance acceptance rate vs scores
- Use performance profiles to optimize for current conditions

This enhanced system provides enterprise-grade robustness with automatic recovery, intelligent optimization, and comprehensive monitoring. The system can now handle production workloads with minimal manual intervention while maintaining competitive performance in the Subnet 17 environment.


## Security Considerations

1. **Wallet Security**:
   - Store wallet files securely
   - Use strong passwords
   - Consider hardware wallets for large stakes

2. **Network Security**:
   - Use firewall rules to limit access
   - Monitor for suspicious connections
   - Keep software updated

3. **Resource Protection**:
   - Monitor resource usage
   - Set up alerts for anomalies
   - Regular backup of configurations

## Support and Resources

- **Discord**: [404 Discord Server](https://discord.gg/404gen)
- **Documentation**: [Official Docs](https://doc.404.xyz/)
- **Miner Dashboard**: [dashboard.404.xyz](https://dashboard.404.xyz/dashboards)
- **GitHub Issues**: Report bugs and feature requests

## License

This mining environment is provided under the same license terms as the Subnet 17 project. Please ensure compliance with the miner license consent declaration when submitting results.

---

**Happy Mining! 🚀** 