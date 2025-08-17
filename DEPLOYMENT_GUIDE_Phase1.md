# Phase 1 Deployment Guide - Distributed RL System

## 🎯 Overview

This guide provides step-by-step instructions for deploying the Phase 1 Distributed RL System, which implements intelligent job distribution across 8 GPUs with load balancing and episodic memory management.

## 📋 Prerequisites

### Hardware Requirements
- **8x NVIDIA GPUs** (recommended: A6000 or RTX 3090/4090 with 24GB+ VRAM)
- **CPU**: 16+ cores (Intel Xeon or AMD EPYC recommended)
- **RAM**: 64GB+ system memory
- **Storage**: 1TB+ NVMe SSD for fast I/O
- **Network**: Gigabit Ethernet for multi-node deployments

### Software Requirements
- **OS**: Ubuntu 20.04+ / CentOS 8+ / RHEL 8+
- **Python**: 3.9+
- **CUDA**: 11.8+ or 12.0+
- **Docker**: 20.10+ (optional, for containerized deployment)
- **Redis**: 6.0+ (for distributed memory)

### GPU Driver Requirements
```bash
# Check NVIDIA driver version
nvidia-smi

# Required: Driver version 470+ for CUDA 11.8+
# If needed, install/update drivers:
sudo apt update
sudo apt install nvidia-driver-525
```

## 🚀 Quick Start (Single Machine)

### 1. Environment Setup

```bash
# Clone the repository
git clone <repository-url>
cd distributed-rl-system

# Create Python environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Setup project structure
python setup_project.py
```

### 2. Configuration

```bash
# Copy environment configuration
cp .env.example .env

# Edit configuration (adjust for your setup)
nano .env
```

Key configuration options:
```bash
# System Configuration
NUM_GPUS=8                    # Number of GPUs available
BASE_GPU_PORT=8096           # Starting port for GPU agents
COORDINATOR_PORT=8090        # Coordinator API port

# Redis Configuration
REDIS_HOST=localhost
REDIS_PORT=6379

# Performance Configuration
GPU_MEMORY_LIMIT_GB=20.0     # Per-GPU memory limit
GPU_TEMPERATURE_LIMIT=85.0   # Thermal limit
MAX_CONCURRENT_JOBS=3        # Maximum parallel jobs

# Logging
LOG_LEVEL=INFO
LOG_DIR=./logs
```

### 3. Start Redis (if not already running)

```bash
# Install Redis
sudo apt install redis-server

# Start Redis service
sudo systemctl start redis-server
sudo systemctl enable redis-server

# Verify Redis is running
redis-cli ping
# Should return: PONG
```

### 4. Initialize System

```bash
# Setup system configuration
python main.py setup

# Verify GPU accessibility
python -c "
import torch
print(f'GPUs available: {torch.cuda.device_count()}')
for i in range(torch.cuda.device_count()):
    print(f'GPU {i}: {torch.cuda.get_device_name(i)}')
"
```

### 5. Start System Components

```bash
# Option A: Use startup script (recommended)
python scripts/phase1/start_system.py

# Option B: Start components manually
# Terminal 1 - Coordinator
python main.py coordinator --port 8090 --num-gpus 8

# Terminal 2-9 - GPU Agents
python main.py gpu-agent 0
python main.py gpu-agent 1
# ... repeat for GPUs 2-7

# Terminal 10 - Monitor
python main.py monitor
```

### 6. Verify System Health

```bash
# Run system tests
python scripts/phase1/test_system.py

# Check system status
curl http://localhost:8090/api/system/status

# View logs
tail -f logs/coordinator_*.log
```

## 🔧 Advanced Deployment

### Multi-Node Deployment

For distributed deployment across multiple machines:

#### Master Node (Coordinator + Redis)
```bash
# Start Redis with network binding
redis-server --bind 0.0.0.0 --port 6379

# Start coordinator
python main.py coordinator \
  --port 8090 \
  --redis-host 0.0.0.0 \
  --num-gpus 16  # Total across all nodes
```

#### Worker Nodes (GPU Agents)
```bash
# Set coordinator URL
export COORDINATOR_URL=http://master-node-ip:8090
export REDIS_HOST=master-node-ip

# Start GPU agents (adjust GPU IDs per node)
python main.py gpu-agent 0 --coordinator-url $COORDINATOR_URL
python main.py gpu-agent 1 --coordinator-url $COORDINATOR_URL
# ... etc
```

### Docker Deployment

#### Build Images
```bash
# Build coordinator image
docker build -f docker/Dockerfile.coordinator -t distributed-rl-coordinator .

# Build GPU agent image
docker build -f docker/Dockerfile.gpu-agent -t distributed-rl-gpu-agent .
```

#### Docker Compose Deployment
```yaml
# docker-compose.yml
version: '3.8'

services:
  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
    volumes:
      - redis_data:/data

  coordinator:
    image: distributed-rl-coordinator
    ports:
      - "8090:8090"
    environment:
      - REDIS_HOST=redis
      - NUM_GPUS=8
    depends_on:
      - redis

  gpu-agent-0:
    image: distributed-rl-gpu-agent
    environment:
      - GPU_ID=0
      - COORDINATOR_URL=http://coordinator:8090
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              device_ids: ['0']
              capabilities: [gpu]

volumes:
  redis_data:
```

Start with Docker Compose:
```bash
docker-compose up -d
```

### Kubernetes Deployment

```yaml
# k8s/distributed-rl-deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: distributed-rl-coordinator
spec:
  replicas: 1
  selector:
    matchLabels:
      app: coordinator
  template:
    metadata:
      labels:
        app: coordinator
    spec:
      containers:
      - name: coordinator
        image: distributed-rl-coordinator:latest
        ports:
        - containerPort: 8090
        env:
        - name: REDIS_HOST
          value: "redis-service"
        - name: NUM_GPUS
          value: "8"

---
apiVersion: apps/v1
kind: DaemonSet
metadata:
  name: gpu-agents
spec:
  selector:
    matchLabels:
      app: gpu-agent
  template:
    metadata:
      labels:
        app: gpu-agent
    spec:
      containers:
      - name: gpu-agent
        image: distributed-rl-gpu-agent:latest
        env:
        - name: COORDINATOR_URL
          value: "http://coordinator-service:8090"
        resources:
          limits:
            nvidia.com/gpu: 1
```

## 📊 Monitoring & Management

### System Monitoring

#### Built-in Monitor
```bash
# Start system monitor
python main.py monitor --update-interval 5

# View real-time system status
watch -n 2 "curl -s http://localhost:8090/api/system/status | jq"
```

#### Prometheus Metrics (Optional)
```bash
# Install Prometheus Python client
pip install prometheus-client

# Enable Prometheus metrics in .env
ENABLE_PROMETHEUS=true
PROMETHEUS_PORT=9090

# Metrics available at http://localhost:9090/metrics
```

### Log Management

#### Log Locations
- **Coordinator**: `logs/coordinator_*.log`
- **GPU Agents**: `logs/gpu_agent_*_*.log`
- **System Monitor**: `logs/monitor_*.log`
- **Errors**: `logs/*_errors_*.log`

#### Log Rotation
```bash
# Setup logrotate for automated log management
sudo tee /etc/logrotate.d/distributed-rl << EOF
/path/to/distributed-rl-system/logs/*.log {
    daily
    rotate 30
    compress
    delaycompress
    missingok
    notifempty
    copytruncate
}
EOF
```

### Performance Tuning

#### GPU Memory Optimization
```bash
# Monitor GPU memory usage
nvidia-smi -l 5

# Adjust memory limits in .env
GPU_MEMORY_LIMIT_GB=18.0    # Leave 6GB buffer for system

# Enable memory growth (for TensorFlow/PyTorch)
export TF_FORCE_GPU_ALLOW_GROWTH=true
```

#### Redis Optimization
```bash
# Optimize Redis for performance
redis-cli CONFIG SET save ""                    # Disable persistence for speed
redis-cli CONFIG SET maxmemory 8gb             # Set memory limit
redis-cli CONFIG SET maxmemory-policy allkeys-lru  # LRU eviction
```

#### Network Optimization
```bash
# Increase network buffer sizes
echo 'net.core.rmem_max = 134217728' >> /etc/sysctl.conf
echo 'net.core.wmem_max = 134217728' >> /etc/sysctl.conf
sysctl -p
```

## 🧪 Testing & Validation

### Automated Testing
```bash
# Run full test suite
python scripts/phase1/test_system.py --output test_results.json

# Run specific test categories
python -m pytest tests/unit/ -v
python -m pytest tests/integration/ -v
python -m pytest tests/performance/ -v

# Load testing
python scripts/phase1/test_system.py --load-test --jobs 10
```

### Manual Testing

#### Submit Test Job
```python
import requests

# Submit a test job
response = requests.post("http://localhost:8090/api/jobs/submit", json={
    "prompts": [
        "a red sports car",
        "a blue mountain landscape", 
        "a green forest scene"
    ],
    "target_score": 0.85,
    "max_episodes": 5,
    "job_name": "Test Job"
})

job_id = response.json()["job_id"]
print(f"Job submitted: {job_id}")

# Check job status
status = requests.get(f"http://localhost:8090/api/jobs/{job_id}/status")
print(status.json())
```

#### Performance Benchmarking
```bash
# Benchmark prompt processing
python -c "
import time
from src.coordinator.batch_splitter.analyzer import PromptAnalyzer

analyzer = PromptAnalyzer()
prompts = ['a red car'] * 100

start = time.time()
for prompt in prompts:
    analysis = analyzer.analyze_prompt(prompt)
end = time.time()

print(f'Processed {len(prompts)} prompts in {end-start:.2f}s')
print(f'Rate: {len(prompts)/(end-start):.1f} prompts/second')
"
```

## 🛠️ Troubleshooting

### Common Issues

#### GPU Not Detected
```bash
# Check NVIDIA drivers
nvidia-smi

# Check CUDA installation
nvcc --version

# Verify PyTorch GPU access
python -c "import torch; print(torch.cuda.is_available())"

# If GPU not available, reinstall CUDA toolkit:
wget https://developer.download.nvidia.com/compute/cuda/12.0.0/local_installers/cuda_12.0.0_525.60.13_linux.run
sudo sh cuda_12.0.0_525.60.13_linux.run
```

#### Redis Connection Issues
```bash
# Check Redis status
sudo systemctl status redis-server

# Test Redis connectivity
redis-cli ping

# Check Redis configuration
redis-cli CONFIG GET "*"

# Restart Redis if needed
sudo systemctl restart redis-server
```

#### Port Conflicts
```bash
# Check if ports are in use
sudo netstat -tlnp | grep :8090
sudo netstat -tlnp | grep :8096

# Kill processes using ports
sudo fuser -k 8090/tcp
sudo fuser -k 8096/tcp
```

#### Memory Issues
```bash
# Check system memory
free -h

# Check GPU memory
nvidia-smi

# Monitor memory usage
watch -n 1 'free -h; echo ""; nvidia-smi --query-gpu=memory.used,memory.total --format=csv'
```

### Performance Issues

#### Slow Job Processing
1. **Check GPU utilization**: `nvidia-smi -l 1`
2. **Verify batch distribution**: Check coordinator logs
3. **Monitor memory usage**: Ensure no memory leaks
4. **Adjust batch sizes**: Reduce `max_batch_size` in GPU config

#### High Memory Usage
1. **Reduce batch sizes**: Lower `GPU_MEMORY_LIMIT_GB`
2. **Enable memory growth**: Set PyTorch/TensorFlow memory options
3. **Clear Redis cache**: `redis-cli FLUSHDB`

#### Network Bottlenecks
1. **Monitor network I/O**: `iftop` or `netstat -i`
2. **Increase buffer sizes**: Tune kernel network parameters
3. **Use faster storage**: NVMe SSD for logs and temp files

## 📈 Scaling Guidelines

### Horizontal Scaling

#### Adding More GPUs
1. Update `NUM_GPUS` in configuration
2. Start additional GPU agents
3. Verify load balancing distribution

#### Multi-Node Scaling
1. Deploy coordinator on master node
2. Start GPU agents on worker nodes
3. Configure shared Redis instance
4. Monitor cross-node communication

### Vertical Scaling

#### Resource Optimization
- **CPU**: Scale coordinator cores with job volume
- **Memory**: 8GB base + 1GB per concurrent job
- **Storage**: Scale with episodic memory growth
- **Network**: Scale bandwidth with multi-node deployments

## 🔐 Security Considerations

### Network Security
```bash
# Firewall configuration
sudo ufw allow 8090    # Coordinator API
sudo ufw allow 6379    # Redis (restrict to cluster IPs)
sudo ufw allow 8096:8103  # GPU agent ports

# For production, use TLS:
# - Generate SSL certificates
# - Configure HTTPS endpoints
# - Enable Redis AUTH
```

### Access Control
```bash
# Redis authentication
redis-cli CONFIG SET requirepass "your-secure-password"

# Update .env file
echo "REDIS_PASSWORD=your-secure-password" >> .env
```

### Data Security
- **Episodic Memory**: Encrypt sensitive prompt data
- **Job Results**: Secure API endpoints with authentication
- **Logs**: Rotate and encrypt log files

## 📊 Production Checklist

### Pre-Production
- [ ] All tests passing (>95% success rate)
- [ ] Performance benchmarks meet requirements
- [ ] Security configurations applied
- [ ] Monitoring and alerting configured
- [ ] Backup procedures established
- [ ] Disaster recovery plan documented

### Go-Live
- [ ] System health validated
- [ ] All GPUs reporting healthy
- [ ] Redis connectivity confirmed
- [ ] Initial job submitted successfully
- [ ] Monitoring dashboards functional
- [ ] On-call procedures activated

### Post-Production
- [ ] Performance monitoring active
- [ ] Regular health checks scheduled
- [ ] Maintenance windows planned
- [ ] Capacity planning initiated
- [ ] Documentation updated

## 🎯 Next Steps

After successful Phase 1 deployment:

1. **Phase 2**: Parallel RL Execution with Cross-GPU Communication
2. **Phase 3**: Results Aggregation and Global Memory Synchronization  
3. **Phase 4**: Real-time Dashboard and Advanced Monitoring
4. **Phase 5**: Production Optimization and Auto-scaling

This completes the Phase 1 deployment foundation for the Distributed RL System. The system is now ready to handle intelligent job distribution across multiple GPUs with comprehensive monitoring and management capabilities.




