# Simple Working Distributed RL System

## 🎯 Overview

This is a **simplified but fully functional** implementation of the distributed RL system that actually works end-to-end. It includes all the core components but with simplified implementations that can run independently and together.

## 🏗️ Architecture

```
┌─────────────────────────────────────────────┐
│              Simple Coordinator             │
│  - Job queue management                     │
│  - Cross-GPU insight sharing               │
│  - Results aggregation                     │
│  Port: 8090                                │
└─────────────────┬───────────────────────────┘
                  │
    ┌─────────────┼─────────────┐
    │             │             │
┌───▼───┐    ┌───▼───┐    ┌───▼───┐
│GPU 0  │    │GPU 1  │    │GPU N  │
│Agent  │    │Agent  │    │Agent  │
│8096   │    │8097   │    │809N   │
└───────┘    └───────┘    └───────┘
```

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install fastapi uvicorn aiohttp redis rich
```

### 2. Start the System

```bash
# Start the complete system (coordinator + 8 GPU agents)
python scripts/start_simple_system.py

# Or start with specific GPUs only
python scripts/start_simple_system.py --gpus "0,1,2"

# Run with system test
python scripts/start_simple_system.py --test
```

### 3. Test the System

```bash
# Run comprehensive tests
python test_simple_system.py
```

## 🧪 Usage Examples

### Test Single GPU Agent (Standalone)

```bash
# Test single prompt processing on GPU 0
curl -X POST http://localhost:8096/test_prompt \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "a red sports car",
    "target_score": 0.85,
    "max_episodes": 3
  }'
```

### Submit Job to Coordinator (Distributed)

```bash
# Submit multi-prompt job for distributed processing
curl -X POST http://localhost:8090/api/jobs/submit \
  -H "Content-Type: application/json" \
  -d '{
    "prompts": [
      "a blue house with a garden",
      "a mountain landscape at sunset",
      "a futuristic car"
    ],
    "target_score": 0.85,
    "max_episodes": 5
  }'
```

### Check System Status

```bash
# Get overall system status
curl http://localhost:8090/api/system/status

# Get specific GPU status
curl http://localhost:8096/status

# Get cross-GPU learning insights
curl http://localhost:8090/api/insights
```

### Check Job Status

```bash
# Check job status (replace JOB_ID with actual job ID)
curl http://localhost:8090/api/jobs/JOB_ID
```

## 📊 Key Features

### ✅ Working Features

1. **Job Queue Management**: Submit multiple jobs, automatic queuing and processing
2. **Cross-GPU Processing**: Distribute prompts across multiple GPU agents
3. **Strategy Sharing**: GPUs share successful optimization strategies
4. **Episodic Memory**: Local memory caching for prompt optimization history
5. **Performance Tracking**: Real-time statistics and performance monitoring
6. **Independent Operation**: GPU agents can work standalone or with coordinator
7. **Graceful Shutdown**: Proper cleanup of all processes

### ✅ API Endpoints

#### Coordinator (Port 8090)
- `POST /api/jobs/submit` - Submit new RL job
- `GET /api/jobs/{job_id}` - Get job status
- `GET /api/system/status` - Get system status
- `GET /api/insights` - Get cross-GPU insights
- `GET /health` - Health check

#### GPU Agent (Ports 8096-8103)
- `POST /process_batch` - Process prompt batch (coordinator use)
- `POST /test_prompt` - Test single prompt (standalone use)
- `POST /receive_insight` - Receive cross-GPU insight
- `GET /status` - Get agent status
- `GET /health` - Health check

## 🔧 System Components

### 1. Simple Coordinator (`src/coordinator/simple_coordinator.py`)

**Features:**
- Job queue with priority handling
- Simple but effective batch distribution
- Cross-GPU insight collection and broadcasting
- Strategy performance tracking
- Redis integration (optional, falls back to memory)

**Key Methods:**
- `submit_job()`: Accept and queue new jobs
- `_try_start_next_job()`: Distribute jobs to available GPUs
- `_send_batch_to_gpu()`: Send work to specific GPU
- `receive_gpu_insight()`: Collect insights from GPUs

### 2. Simple GPU Agent (`src/gpu_agent/simple_gpu_agent.py`)

**Features:**
- Simulated RL optimization with realistic behavior
- Multiple optimization strategies
- Local episodic memory caching
- Cross-GPU insight sharing
- Standalone testing capability

**Key Methods:**
- `process_batch()`: Process batch from coordinator
- `_process_single_prompt()`: Run RL optimization on single prompt
- `_run_episode_optimization()`: Simulate episodic RL
- `test_single_prompt()`: Standalone prompt testing

### 3. System Manager (`scripts/start_simple_system.py`)

**Features:**
- Automated startup of all components
- Health monitoring and verification
- Graceful shutdown handling
- System status display

## ⚙️ Configuration

Edit `config/settings.py` to customize:

```python
# System configuration
num_gpus: int = 8                    # Number of GPU agents
base_gpu_port: int = 8096           # Base port for GPU agents
coordinator_port: int = 8090        # Coordinator port

# Job defaults
default_target_score: float = 0.85  # Default optimization target
default_max_episodes: int = 5       # Default episodes per prompt
max_concurrent_jobs: int = 3        # Max concurrent jobs
```

## 🧠 How It Works

### Job Processing Flow

1. **Job Submission**: User submits job with multiple prompts to coordinator
2. **Batch Distribution**: Coordinator splits prompts across available GPU agents
3. **Parallel Processing**: Each GPU agent runs RL optimization on assigned prompts
4. **Strategy Sharing**: GPUs share successful strategies in real-time
5. **Results Aggregation**: Coordinator collects and combines results
6. **Completion**: Final results returned to user

### RL Optimization Process (Per GPU)

1. **Memory Loading**: Load episodic memory for prompt
2. **Strategy Selection**: Choose optimization strategy based on insights
3. **Episodes**: Run multiple optimization episodes
4. **Rounds**: Multiple optimization rounds per episode
5. **Validation**: Simulate TRELLIS validation (scoring)
6. **Memory Update**: Update local memory with results
7. **Insight Sharing**: Share successful strategies with other GPUs

## 📈 Performance

### Expected Performance

- **Processing**: ~0.1-0.5 seconds per prompt per episode (simulated)
- **Throughput**: 8x speedup with 8 GPUs vs single GPU
- **Learning**: Cross-GPU strategy sharing improves scores over time
- **Reliability**: Automatic retry and error handling

### Monitoring

- Real-time job progress tracking
- GPU utilization monitoring
- Strategy effectiveness analysis
- Cross-GPU learning insights

## 🐛 Troubleshooting

### Common Issues

1. **Port conflicts**: Ensure ports 8090-8103 are available
2. **Redis connection**: System works without Redis (falls back to memory)
3. **GPU startup**: Check individual GPU agent logs in `logs/` directory
4. **Job stuck**: Check coordinator and GPU agent health endpoints

### Debug Commands

```bash
# Check if coordinator is running
curl http://localhost:8090/health

# Check specific GPU agent
curl http://localhost:8096/health

# View system logs
tail -f logs/simple_coordinator_*.log
tail -f logs/simple_gpu_agent_0_*.log
```

### Manual Component Testing

```bash
# Start coordinator only
python -m src.coordinator.simple_coordinator

# Start single GPU agent
python -m src.gpu_agent.simple_gpu_agent --gpu-id 0

# Test single GPU agent
python test_simple_system.py
```

## 📚 Extension Points

This simple system provides a foundation for adding:

1. **Real TRELLIS Integration**: Replace simulated validation with actual TRELLIS calls
2. **Advanced RL Algorithms**: Implement sophisticated RL optimization
3. **Memory Persistence**: Add database storage for episodic memory
4. **Load Balancing**: Implement intelligent GPU assignment
5. **Dashboard**: Add web-based monitoring interface
6. **Scaling**: Add support for multiple machines

## ✅ Verification

Run the complete test suite to verify everything works:

```bash
# Start system and run all tests
python scripts/start_simple_system.py --test

# Or run tests against running system
python test_simple_system.py
```

**Expected Results:**
- All health checks pass
- Single prompt processing works
- Job submission and completion works
- Cross-GPU insights are generated
- Strategy performance is tracked

This simple system demonstrates all the core concepts of the distributed RL architecture while being practical and actually working!




