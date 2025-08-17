# GPU Server Wrapper for TRELLIS - Subnet 17 (404-GEN)

This wrapper system allows you to run TRELLIS servers on multiple GPUs simultaneously, with parallel testing and validation capabilities.

## 🚀 Features

- **Smart GPU Detection**: Automatically detects which GPUs are already loaded and serving
- **Multi-GPU Support**: Start TRELLIS servers on all 8 GPUs with unique ports
- **Parallel Operations**: Prime all GPUs simultaneously and test validation in parallel
- **Performance Ranking**: 🏆 Rank GPUs by speed (fastest to slowest) with medals and detailed metrics
- **Health Monitoring**: Comprehensive GPU server health checking and status reporting
- **Result Analysis**: Detailed analysis of generation and validation results
- **Easy Testing**: Simple scripts for testing individual endpoints and bulk operations

## 📁 Files

- `gpu_server_wrapper.py` - Main wrapper script for managing GPU servers
- `test_gpu_wrapper.py` - Test script demonstrating the wrapper functionality
- `quick_test.sh` - Bash script for quick testing of individual GPU endpoints
- `README_GPU_Wrapper.md` - This documentation file

## 🛠️ Prerequisites

1. **Hardware**: 8 GPUs with sufficient VRAM for TRELLIS models
2. **Software**: Python 3.8+, TRELLIS server script (`trellis_subnit_server_mix_lora_flash.py`)
3. **Dependencies**: 
   ```bash
   pip install requests asyncio
   ```

## 🚀 Quick Start

### 1. Check GPU Loading Status

```bash
# Check which GPUs are already loaded
python gpu_server_wrapper.py --check-status-only
```

### 2. Start All GPU Servers (if needed)

```bash
python gpu_server_wrapper.py
```

This will:
- Check which GPUs are already loaded
- Start TRELLIS servers only on GPUs that need loading
- Use ports 8096-8103 respectively
- Wait for servers to initialize
- Perform health checks

### 2. Test Individual Endpoints

```bash
# Test GPU 0 (port 8096)
curl -d "prompt=pink bicycle" -X POST http://127.0.0.1:8096/generate/

# Test GPU 1 (port 8097)
curl -d "prompt=blue vase" -X POST http://127.0.0.1:8097/generate/

# ... and so on for all 8 GPUs
```

### 3. Quick Test All GPUs

```bash
chmod +x quick_test.sh
./quick_test.sh
```

### 4. Run Full Test Suite

```bash
python test_gpu_wrapper.py
```

## 📊 Port Mapping

| GPU ID | Port | URL |
|--------|------|-----|
| 0 | 8096 | http://127.0.0.1:8096 |
| 1 | 8097 | http://127.0.0.1:8097 |
| 2 | 8098 | http://127.0.0.1:8098 |
| 3 | 8099 | http://127.0.0.1:8099 |
| 4 | 8100 | http://127.0.0.1:8100 |
| 5 | 8101 | http://127.0.0.1:8101 |
| 6 | 8102 | http://127.0.0.1:8102 |
| 7 | 8103 | http://127.0.0.1:8103 |

## 🔧 Configuration Options

### Command Line Arguments

```bash
python gpu_server_wrapper.py [OPTIONS]

Options:
  --gpus INT              Number of GPUs to use (default: 8)
  --base-port INT         Base port number (default: 8096)
  --server-script PATH    TRELLIS server script path
  --output-dir PATH       Output directory for logs and results
  --skip-startup          Skip server startup (assume already running)
  --skip-priming          Skip GPU priming
  --skip-validation       Skip validation testing
  --check-status-only     Only check GPU loading status and exit
  --show-ranking          Show current performance ranking and exit
  --run-additional-tests Run additional tests (consistency, memory, latency)
  --show-validation-results Show validation results table from JSON files
```

### Example Configurations

```bash
# Use only 4 GPUs starting from port 9000
python gpu_server_wrapper.py --gpus 4 --base-port 9000

# Skip startup (servers already running)
python gpu_server_wrapper.py --skip-startup

# Only test priming, skip validation
python gpu_server_wrapper.py --skip-validation

# Just check which GPUs are already loaded
python gpu_server_wrapper.py --check-status-only

# Show current performance ranking
python gpu_server_wrapper.py --show-ranking

# Run additional tests (consistency, memory, latency)
python gpu_server_wrapper.py --run-additional-tests

# Show validation results from previous runs
python gpu_server_wrapper.py --show-validation-results
```

## 📈 Testing Workflow

### Phase 1: Smart GPU Detection & Startup
1. Check which GPUs are already loaded and serving requests
2. Start TRELLIS servers only on GPUs that need loading
3. Wait for model loading (30-45 seconds) for newly started servers
4. Perform health checks on all GPUs
5. Verify all servers are responsive

### Phase 2: Parallel Priming
1. Send generation requests to all GPUs simultaneously
2. Use different test prompts for each GPU
3. Measure generation times and success rates
4. **🏆 Rank GPUs by performance (fastest to slowest)**
5. Save results for analysis

### Phase 3: Parallel Validation
1. Generate test models on each GPU
2. Test validation endpoints in parallel
3. Measure validation performance
4. **🏆 Rank GPUs by validation speed**
5. Collect comprehensive metrics

### Phase 4: Status Reporting
1. Generate comprehensive status reports
2. Save results to JSON files
3. Display performance summaries
4. Keep servers running for manual testing

## 📊 Output Files

The wrapper generates several output files:

- `gpu_server_wrapper.log` - Main log file
- `gpu_server_outputs/` - Output directory containing:
  - `priming_results_YYYYMMDD_HHMMSS.json` - Priming test results
  - `validation_results_YYYYMMDD_HHMMSS.json` - Validation test results
  - `gpu_status_report_YYYYMMDD_HHMMSS.json` - Comprehensive status reports

## 🧪 Testing Individual GPUs

### Manual Testing

```bash
# Test generation on specific GPU
curl -d "prompt=test prompt" -X POST http://127.0.0.1:8096/generate/

# Check server status
curl http://127.0.0.1:8096/status/

# Check health
curl http://127.0.0.1:8096/health/
```

### Automated Testing

```python
from gpu_server_wrapper import GPUServerManager

# Create manager for specific GPU
manager = GPUServerManager(num_gpus=1, base_port=8096)

# Test specific GPU
result = manager.prime_single_gpu(0, "test prompt")
print(f"GPU 0 result: {result}")
```

## 🔍 Monitoring and Debugging

### Health Checks

```python
# Check health of all servers
health_results = manager.check_all_servers_health()

# Get detailed status
status_data = manager.get_comprehensive_status()
```

### Log Analysis

```bash
# Monitor logs in real-time
tail -f gpu_server_wrapper.log

# Search for errors
grep "ERROR" gpu_server_wrapper.log

# Check GPU-specific logs
grep "GPU 0" gpu_server_wrapper.log
```

## 🚨 Troubleshooting

### Common Issues

1. **Port Already in Use**
   ```bash
   # Check what's using the port
   lsof -i :8096
   
   # Kill the process
   kill -9 <PID>
   ```

2. **GPU Memory Issues**
   ```bash
   # Check GPU memory usage
   nvidia-smi
   
   # Clear GPU cache
   curl -X POST http://127.0.0.1:8096/clear_cache/
   ```

3. **Server Not Responding**
   ```bash
   # Check if process is running
   ps aux | grep trellis_subnit_server
   
   # Restart specific GPU server
   # (Use the wrapper's restart functionality)
   ```

### Performance Optimization

1. **Model Loading**: Ensure sufficient GPU memory for all models
2. **Parallel Operations**: Use the wrapper's parallel testing capabilities
3. **Resource Monitoring**: Monitor GPU utilization during testing
4. **Batch Processing**: Group similar operations for efficiency
5. **Smart Loading**: Use `--check-status-only` to verify GPU readiness before testing

### Smart Loading Issues

1. **GPU Not Detected as Loaded**
   ```bash
   # Check if the port is actually responding
   curl http://127.0.0.1:8096/health/
   
   # Verify the server is running
   ps aux | grep trellis_subnit_server
   ```

2. **False Positive Detection**
   ```bash
   # Force a fresh status check
   python gpu_server_wrapper.py --check-status-only
   ```

## 🔄 Advanced Usage

### Smart GPU Loading Detection

The wrapper automatically detects which GPUs are already loaded and serving requests:

```python
# Check loading status without starting servers
loading_status = manager.check_gpu_loading_status()

# This returns a dict like:
# {0: "already_loaded", 1: "needs_loading", 2: "already_loaded", ...}

# Check if specific GPU is ready
if loading_status[0] == "already_loaded":
    print("GPU 0 is ready for requests")
```

### Performance Ranking

The wrapper automatically ranks GPUs by performance after each test:

```python
# Get current performance ranking
ranking_data = manager.get_performance_ranking()

# Access ranking information
fastest_gpu = ranking_data['performance_summary']['fastest_gpu']
slowest_gpu = ranking_data['performance_summary']['slowest_gpu']
performance_spread = ranking_data['performance_summary']['performance_spread']

print(f"Fastest GPU: {fastest_gpu}, Slowest: {slowest_gpu}")
print(f"Performance spread: {performance_spread:.2f}s")
```

### Performance Comparison Table

The wrapper generates a comprehensive performance comparison table:

```python
# Print detailed performance table
manager.print_performance_comparison_table()
```

This shows:
- GPU ranking by total time (generation + validation)
- Individual generation and validation times
- PLY file sizes and compression ratios
- Performance spread analysis

### Additional Tests

Run comprehensive tests beyond basic functionality:

```python
# Run additional tests
manager.run_additional_tests()
```

Tests include:
- **Consistency Test**: Same prompt across all GPUs
- **Memory Check**: GPU memory usage monitoring
- **Latency Test**: Network response times
- **Validation Results Collection**: From JSON files

### Enhanced Validation System

The wrapper now uses local production-accurate validation instead of broken server endpoints:

```python
# Direct local validation using subnet_accurate_validator.py
manager._test_single_gpu_validation(gpu_id)
```

Features:
- **Local Validation**: Uses `subnet_accurate_validator.py` directly
- **Production Accuracy**: Same validation logic as subnet validators
- **Detailed Metrics**: Validation scores, alignment, quality, SSIM, LPIPS
- **JSON Results**: Saves detailed results for analysis
- **Comprehensive Tables**: Visual comparison of all GPU validation results

### Validation Results Table

View detailed validation results from all GPUs:

```bash
python gpu_server_wrapper.py --show-validation-results
```

Shows:
- 🏆 **Rankings**: Best to worst validation scores
- 📊 **Detailed Metrics**: Score, alignment, quality, demo fidelity
- ✅ **Pass/Fail Status**: Based on production thresholds
- 📈 **Summary Stats**: Average scores, pass rates

### Custom Test Prompts

```python
# Modify test prompts in GPUServerManager
manager.test_prompts = [
    "your custom prompt 1",
    "your custom prompt 2",
    # ... more prompts
]
```

### Integration with Orchestrator

```python
# Import orchestrator components
from continuous_trellis_orchestrator_lora import TaskRecord, ValidatorState

# Use with GPU wrapper for subnet operations
# (See the wrapper code for integration examples)
```

### Custom Validation Testing

```python
# Extend validation testing
def custom_validation_test(gpu_id, prompt):
    # Your custom validation logic
    pass

# Integrate with manager
manager.custom_validation = custom_validation_test
```

## 📚 API Reference

### GPUServerManager Class

```python
class GPUServerManager:
    def __init__(self, num_gpus=8, base_port=8096, 
                 server_script="trellis_subnit_server_mix_lora_flash.py",
                 output_dir="./gpu_server_outputs")
    
    def start_all_servers(self) -> bool
    def check_all_servers_health(self) -> Dict[int, bool]
    def prime_all_gpus_parallel(self) -> List[Dict[str, Any]]
    def test_validation_parallel(self) -> List[Dict[str, Any]]
    def get_comprehensive_status(self) -> Dict[str, Any]
    def cleanup(self)
```

### GPUServer Class

```python
@dataclass
class GPUServer:
    gpu_id: int
    port: int
    process: Optional[subprocess.Popen]
    status: str
    generation_count: int
    validation_count: int
    error_count: int
```

## 🤝 Contributing

To extend the wrapper system:

1. **Add New Test Types**: Extend the manager with new testing methods
2. **Improve Monitoring**: Add more detailed health checks and metrics
3. **Performance Optimization**: Optimize parallel operations and resource usage
4. **Error Handling**: Improve error recovery and debugging capabilities

## 📄 License

This wrapper is part of the Subnet 17 (404-GEN) project and follows the same licensing terms.

## 🆘 Support

For issues and questions:

1. Check the logs for detailed error information
2. Verify GPU availability and memory
3. Ensure all dependencies are installed
4. Check port availability and firewall settings

---

**Happy GPU Testing! 🎨🚀**
