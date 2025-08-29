# 🚀 Validation Simulation Server

A comprehensive simulation server that mimics **subnet 17 validator behavior** for stress testing miners and orchestrators. This server replicates the exact cooldown logic, validation behavior, and response patterns of the real subnet.

## 🎯 Purpose

- **Stress test** your miner/orchestrator implementations
- **Find edge cases** and failure modes before production
- **Validate cooldown compliance** with subnet requirements
- **Test rate limiting** and network failure handling
- **Benchmark performance** under realistic conditions

## ✨ Features

### 🔄 **Realistic Cooldown System**
- **Synthetic Traffic**: 300 seconds cooldown (matching real subnet)
- **Organic Traffic**: 120 seconds cooldown (matching real subnet)
- **Automatic traffic detection** from prompt patterns
- **Cooldown violation tracking** with penalties
- **Throttle period management** (faster completion = longer cooldown)

### 🎲 **Random Prompt Generation**
- **Synthetic prompts**: test, benchmark, validation, duel, challenge
- **Organic prompts**: realistic user requests with materials, styles, environments
- **Traffic type detection** matching your orchestrator's logic

### 🧪 **Simulation Features**
- **Random validation failures** (5% rate, configurable)
- **Network issues simulation** (2% rate, configurable)
- **Realistic response delays** (0.1s to 2s)
- **Rate limiting** (100 requests/minute, 50 concurrent)
- **Stress testing** with configurable parameters

### �� **Comprehensive Monitoring**
- **Real-time statistics** via `/stats` endpoint
- **Miner state tracking** with cooldown history
- **Performance metrics** and error analysis
- **Detailed logging** for debugging

## 🚀 Quick Start

### 1. **Install Dependencies**
```bash
pip install fastapi uvicorn aiohttp
```

### 2. **Start the Simulation Server**
```bash
python validation_simulation_server.py
```

The server will start on `http://localhost:8094`

### 3. **Test Basic Functionality**
```bash
python test_simulation_server.py
```

### 4. **Run Stress Tests**
```bash
python stress_test_client.py
```

## 📋 API Endpoints

### **GET /** - Server Information
```bash
curl http://localhost:8094/
```
Returns server description and available endpoints.

### **GET /health** - Health Check
```bash
curl http://localhost:8094/health
```
Returns server status and uptime.

### **GET /stats** - Statistics
```bash
curl http://localhost:8094/stats
```
Returns comprehensive simulation statistics.

### **POST /pull_task** - Pull Task (Simulate Validator)
```bash
curl -X POST http://localhost:8094/pull_task \
  -H "Content-Type: application/json" \
  -d '{"hotkey": "test_miner_001"}'
```

**Response:**
```json
{
  "task": {
    "id": "uuid-123",
    "prompt": "modern wooden chair for indoor use",
    "traffic_type": "organic",
    "expected_cooldown": 120
  },
  "cooldown_until": 0,
  "cooldown_violations": 0,
  "throttle_period": 30,
  "validation_threshold": 0.6
}
```

### **POST /submit_results** - Submit Results (Simulate Validator)
```bash
curl -X POST http://localhost:8094/submit_results \
  -H "Content-Type: application/json" \
  -d '{
    "hotkey": "test_miner_001",
    "task_id": "uuid-123",
    "prompt": "modern wooden chair for indoor use",
    "results": "base64_encoded_data",
    "submit_time": 1703123456789,
    "signature": "miner_signature"
  }'
```

**Response:**
```json
{
  "feedback": {
    "validation_failed": false,
    "task_fidelity_score": 0.85,
    "average_fidelity_score": 0.78,
    "generations_within_the_window": 15,
    "current_duel_rating": 0.92,
    "current_miner_reward": 0.92
  },
  "cooldown_until": 1703123576
}
```

### **POST /validate_txt_to_3d_ply** - Validate Results (Simulate Validation Service)
```bash
curl -X POST http://localhost:8094/validate_txt_to_3d_ply \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "modern wooden chair",
    "data": "base64_encoded_ply_data",
    "compression": 2,
    "generate_preview": true,
    "preview_score_threshold": 0.5
  }'
```

**Response:**
```json
{
  "score": 0.8542,
  "iqa": 0.7234,
  "alignment_score": 0.8123,
  "ssim": 0.6543,
  "lpips": 0.2345,
  "preview": "data:image/png;base64,..."
}
```

## ⚙️ Configuration

### **Environment Variables**
```bash
# Server settings
export VALIDATION_SIM_SERVER_PORT=8094
export VALIDATION_SIM_HOST="0.0.0.0"

# Cooldown settings
export VALIDATION_SIM_SYNTHETIC_TRAFFIC_COOLDOWN=300
export VALIDATION_SIM_ORGANIC_TRAFFIC_COOLDOWN=120

# Validation settings
export VALIDATION_SIM_QUALITY_THRESHOLD=0.6
export VALIDATION_SIM_THROTTLE_PERIOD=30

# Simulation settings
export VALIDATION_SIM_ENABLE_RANDOM_FAILURES=true
export VALIDATION_SIM_FAILURE_RATE=0.05
export VALIDATION_SIM_ENABLE_NETWORK_ISSUES=true
export VALIDATION_SIM_NETWORK_ISSUE_RATE=0.02

# Stress testing
export VALIDATION_SIM_MAX_CONCURRENT_REQUESTS=50
export VALIDATION_SIM_RATE_LIMIT_PER_MINUTE=100
```

### **Default Configuration**
```python
DEFAULT_CONFIG = {
    'server_port': 8094,
    'host': '0.0.0.0',
    
    # Cooldown settings (matching real subnet)
    'synthetic_traffic_cooldown': 300,  # 300s for synthetic traffic
    'organic_traffic_cooldown': 120,    # 120s for organic traffic
    
    # Validation settings
    'quality_threshold': 0.6,
    'throttle_period': 30,
    'cooldown_violation_penalty': 10,
    'cooldown_violations_threshold': 100,
    'cooldown_penalty': 600,  # 10 minutes for poor quality
    
    # Simulation settings
    'enable_random_failures': True,
    'failure_rate': 0.05,  # 5% random failures
    'response_delay_range': (0.1, 2.0),  # 0.1s to 2s delay
    'enable_network_issues': True,
    'network_issue_rate': 0.02,  # 2% network issues
    
    # Stress testing
    'enable_stress_testing': True,
    'max_concurrent_requests': 50,
    'rate_limit_per_minute': 100,
}
```

## 🧪 Stress Testing

### **Basic Stress Test**
```bash
python stress_test_client.py
```

### **Custom Stress Test Configuration**
```python
config = StressTestConfig(
    server_url="http://localhost:8094",
    num_miners=50,           # 50 concurrent miners
    requests_per_miner=100,  # 100 requests per miner
    concurrent_requests=25,  # 25 concurrent requests
    delay_between_requests=0.01,  # Very fast requests
    enable_cooldown_violations=True,
    enable_rate_limit_testing=True,
    enable_network_failures=True,
    test_duration_seconds=600  # 10 minutes
)
```

### **Stress Test Results**
The stress test generates a comprehensive JSON report:
```json
{
  "test_summary": {
    "test_duration_seconds": 180.5,
    "num_miners": 20,
    "total_requests": 600,
    "total_submissions": 580,
    "total_successful": 520,
    "total_errors": 20,
    "total_cooldown_violations": 45,
    "success_rate": 0.896
  },
  "performance_metrics": {
    "avg_response_time": 0.234,
    "min_response_time": 0.123,
    "max_response_time": 1.876,
    "total_response_time": 140.4
  },
  "error_analysis": {
    "pull_errors": 8,
    "submit_errors": 12,
    "cycle_errors": 0,
    "rate_limit_responses": 15
  }
}
```

## 🔍 Edge Case Testing

### **1. Cooldown Violations**
- **Purpose**: Test how your system handles repeated cooldown violations
- **Method**: Send requests while on cooldown
- **Expected**: Violation count increases, penalties applied

### **2. Rate Limiting**
- **Purpose**: Test rate limit handling
- **Method**: Send 200+ requests rapidly
- **Expected**: HTTP 429 responses for excess requests

### **3. Network Failures**
- **Purpose**: Test network error handling
- **Method**: Server simulates 2% network issues
- **Expected**: Proper error handling and retry logic

### **4. Validation Failures**
- **Purpose**: Test low-quality result handling
- **Method**: Server simulates 5% validation failures
- **Expected**: Proper cooldown penalties and error handling

### **5. Concurrent Requests**
- **Purpose**: Test system under load
- **Method**: Multiple miners making simultaneous requests
- **Expected**: Stable performance under concurrent load

## 📊 Monitoring and Debugging

### **Real-time Statistics**
```bash
# Get current statistics
curl http://localhost:8094/stats | jq

# Monitor specific metrics
watch -n 1 'curl -s http://localhost:8094/stats | jq ".active_miners, .miners_on_cooldown, .total_violations"'
```

### **Log Analysis**
```bash
# Monitor server logs
tail -f validation_simulation.log

# Search for specific patterns
grep "cooldown violation" validation_simulation.log
grep "rate limit" validation_simulation.log
grep "network issue" validation_simulation.log
```

### **Performance Monitoring**
```bash
# Monitor response times
grep "response_time" validation_simulation.log | awk '{print $NF}' | sort -n

# Monitor cooldown periods
grep "cooldown.*set" validation_simulation.log | tail -20
```

## 🚨 Troubleshooting

### **Server Won't Start**
```bash
# Check dependencies
pip list | grep fastapi
pip list | grep uvicorn

# Check port availability
netstat -tulpn | grep 8094

# Check logs
cat validation_simulation.log
```

### **High Error Rates**
```bash
# Check server statistics
curl http://localhost:8094/stats

# Reduce load
export VALIDATION_SIM_MAX_CONCURRENT_REQUESTS=10
export VALIDATION_SIM_RATE_LIMIT_PER_MINUTE=50
```

### **Cooldown Issues**
```bash
# Check cooldown configuration
grep "cooldown" validation_simulation.log | tail -10

# Verify traffic detection
grep "traffic.*detected" validation_simulation.log
```

## 🔧 Integration with Your Orchestrator

### **Update Configuration**
```python
# In your orchestrator config
'validation_endpoints': ['http://localhost:8094'],  # Use simulation server
'generation_server_url': 'http://localhost:8094',   # For validation calls
```

### **Test Cooldown Compliance**
```python
# The simulation server will enforce the same cooldown rules
# as the real subnet, allowing you to test compliance
```

### **Performance Benchmarking**
```python
# Use stress test results to optimize your orchestrator
# Monitor response times, error rates, and cooldown handling
```

## 📈 Best Practices

### **1. Start Small**
- Begin with 5-10 miners
- Gradually increase load
- Monitor system stability

### **2. Test Different Scenarios**
- Normal operation
- High load conditions
- Network failures
- Cooldown violations

### **3. Monitor Key Metrics**
- Response times
- Error rates
- Cooldown compliance
- Rate limit handling

### **4. Iterate and Improve**
- Identify bottlenecks
- Optimize error handling
- Improve cooldown logic
- Test edge cases

## 🎯 Use Cases

### **Development Testing**
- Test new features before production
- Validate cooldown logic
- Debug edge cases

### **Performance Testing**
- Benchmark system performance
- Identify bottlenecks
- Test scalability

### **Compliance Testing**
- Verify subnet cooldown compliance
- Test rate limiting
- Validate error handling

### **Load Testing**
- Test under high load
- Validate concurrent request handling
- Test system stability

## 🚀 Next Steps

1. **Start the simulation server**
2. **Run basic tests** to verify functionality
3. **Configure your orchestrator** to use the simulation server
4. **Run stress tests** to find edge cases
5. **Monitor and optimize** based on test results
6. **Test with real subnet** once confident

The validation simulation server provides a **safe, controlled environment** to thoroughly test your miner/orchestrator implementation before deploying to the real subnet. Use it to find and fix issues early, ensuring robust production performance! 🎉





