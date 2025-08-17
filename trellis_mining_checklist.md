# 🚀 TRELLIS Mining Pre-Launch Checklist

## 📋 System & Environment Setup

### 🔑 API Keys & Authentication
- [ ] **HuggingFace Token**: `export HF_TOKEN=TOKEN`
- [ ] **Weights & Biases API Key**: `export WANDB_API_KEY=KEY'
- [ ] **Bittensor Wallet**: Ensure wallet is properly configured and has sufficient TAO
- [ ] **Network Connection**: Verify stable internet connection for subnet communication

### 🐍 Python Environment
- [ ] **Python Version**: Confirm Python 3.8+ is installed and active
- [ ] **Virtual Environment**: Activate appropriate virtual environment if using one
- [ ] **Dependencies**: Verify all required packages are installed:
  - [ ] `torch` (with CUDA support if using GPU)
  - [ ] `transformers`
  - [ ] `diffusers`
  - [ ] `accelerate`
  - [ ] `bittensor`
  - [ ] `wandb`
  - [ ] `requests`
  - [ ] `pillow`
  - [ ] `numpy`
  - [ ] `pandas`

### 💾 Storage & Cache
- [ ] **Cache Directory**: Ensure `/home/mbhat/.cache_god` exists and has sufficient space
- [ ] **Checkpoints Directory**: Verify `/home/mbhat/.checkpoints_god` exists and has required models
- [ ] **Output Directory**: Create `./trellis_mining_outputs_test` if it doesn't exist
- [ ] **Database File**: Confirm `continuous_trellis_tasks_test.db` is accessible

## 🖥️ Hardware & Performance

### 🎮 GPU Configuration
- [ ] **CUDA Installation**: Verify CUDA is properly installed and accessible
- [ ] **GPU Memory**: Check available GPU memory (recommended: 8GB+)
- [ ] **GPU Driver**: Ensure latest NVIDIA drivers are installed
- [ ] **CUDA Determinism**: Set `export CUBLAS_WORKSPACE_CONFIG=:4096:8` for reproducible results

### 💻 System Resources
- [ ] **RAM**: Ensure sufficient system memory (recommended: 16GB+)
- [ ] **Disk Space**: Verify adequate storage for logs, outputs, and temporary files
- [ ] **CPU**: Check CPU cores available for parallel processing
- [ ] **Network Bandwidth**: Confirm stable network connection

## 🔧 TRELLIS Server & Components

### 🌐 Server Status
- [ ] **TRELLIS Server**: Verify server is running on port 8096
  ```bash
  curl -s "http://localhost:8096/status/" | grep '"ready":true'
  ```
- [ ] **Server Health**: Check server logs for any errors or warnings
- [ ] **Port Availability**: Ensure port 8096 is not blocked by firewall

### 📁 Core Files
- [ ] **Main Orchestrator**: `continuous_trellis_orchestrator_lora_test_mod.py`
- [ ] **TRELLIS Server**: `trellis_submit_server.py`
- [ ] **Episodic Optimizer**: `episodic_trellis_optimizer.py`
- [ ] **Test Runner**: `run_trellis_mining_test.sh`
- [ ] **Database**: `continuous_trellis_tasks_test.db`

### 🔄 Dependencies & Imports
- [ ] **Import Tests**: Verify all Python imports work without errors
- [ ] **Module Paths**: Ensure all custom modules are in Python path
- [ ] **File Permissions**: Check read/write permissions for all directories

## 🎯 Model & LoRA Configuration

### 🎨 LoRA Models
- [ ] **Default LoRA**: Confirm `baolei` LoRA is available
- [ ] **Alternative LoRAs**: Verify other LoRAs are accessible if needed:
  - [ ] `patched_realism`
  - [ ] `tf2_style`
  - [ ] `cartoon_3d`
  - [ ] `game_assets`
  - [ ] `sd15_game_icon`
  - [ ] `cinema`
  - [ ] `isometric_3d`
  - [ ] `live_3d`
  - [ ] `necklace`

### ⚙️ Generation Parameters
- [ ] **Inference Steps**: Set appropriate number (default: 20-50)
- [ ] **Guidance Scale**: Configure guidance strength (default: 7.5)
- [ ] **SS Sampling Steps**: Set for TRELLIS generation
- [ ] **SLAT Parameters**: Configure SLAT sampling and guidance

## 📊 Mining Configuration

### 🎲 Mining Mode
- [ ] **Mode Selection**: Choose appropriate mining mode:
  - [ ] `--continuous` (default)
  - [ ] `--harvest` / `--no-harvest`
  - [ ] `--submit` / `--no-submit`
  - [ ] `--validate` / `--no-validate`

### 🔍 Optimization Settings
- [ ] **Prompt Optimization**: Enable/disable with `--no-optimize`
- [ ] **Reproducibility**: Configure with `--no-reproducibility`
- [ ] **Similarity Threshold**: Set reproducibility similarity (default: 0.3)
- [ ] **Logging Level**: Configure with `--quiet-optimize` if needed

### 🎯 Validation & Quality
- [ ] **Dual Validation**: Enable with `--dual-validation` for production comparison
- [ ] **Local Validation**: Ensure validation pipeline is working
- [ ] **Score Thresholds**: Verify quality and alignment score thresholds

## 🧪 Pre-Launch Testing

### 🔬 Component Tests
- [ ] **TRELLIS Server Test**: Verify server responds correctly
- [ ] **Database Connection**: Test database read/write operations
- [ ] **Model Loading**: Confirm models load without errors
- [ ] **Image Generation**: Test basic image generation pipeline
- [ ] **Optimization Loop**: Verify episodic optimizer functionality

### 📝 Logging & Monitoring
- [ ] **Log Directories**: Ensure log directories are writable
- [ ] **Log Rotation**: Configure log file rotation if needed
- [ ] **Monitoring**: Set up monitoring for system resources
- [ ] **Alerting**: Configure alerts for critical failures

## 🚨 Emergency Procedures

### ⚡ Quick Stop Commands
- [ ] **Graceful Shutdown**: `Ctrl+C` to stop mining gracefully
- [ ] **Force Stop**: `pkill -f trellis` if needed
- [ ] **Server Restart**: Restart TRELLIS server if it becomes unresponsive
- [ ] **Process Cleanup**: Clean up any zombie processes

### 🔍 Troubleshooting
- [ ] **Common Errors**: Review common error patterns and solutions
- [ ] **Debug Mode**: Enable debug logging if issues arise
- [ ] **Fallback Options**: Have backup configurations ready
- [ ] **Support Resources**: Know where to get help if needed

## 📋 Final Launch Checklist

### ✅ Pre-Launch Verification
- [ ] **All checkboxes above are completed**
- [ ] **System resources are adequate**
- [ ] **No critical errors in logs**
- [ ] **All components are responding**
- [ ] **Backup configurations are ready**

### 🚀 Launch Commands
```bash
# Set environment variables
export HF_TOKEN=TOKEN
export WANDB_API_KEY=KEY
export CUBLAS_WORKSPACE_CONFIG=:4096:8

# Start mining (adjust parameters as needed)
./run_trellis_mining_test.sh --start-server --dual-validation --lora baolei

# Or run episodic optimization directly
python3 episodic_trellis_optimizer.py
```

### 📊 Post-Launch Monitoring
- [ ] **Server Status**: Monitor TRELLIS server health
- [ ] **Resource Usage**: Watch GPU, CPU, and memory usage
- [ ] **Log Output**: Monitor logs for errors or warnings
- [ ] **Performance Metrics**: Track mining performance and scores
- [ ] **Network Activity**: Monitor subnet communication

---

## 🆘 Emergency Contacts & Resources

- **Documentation**: Check codebase README and documentation
- **Logs**: Review log files in `./trellis_mining_logs/`
- **Community**: Reach out to TRELLIS community for support
- **Backup**: Have backup configurations and fallback options ready

---

*Last Updated: $(date)*
*Status: ✅ Ready for Launch*
