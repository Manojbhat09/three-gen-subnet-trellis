# Server Coordination System

## Overview

The Episodic Prompt Optimizer now includes a robust server coordination system that prevents race conditions and ensures proper sequencing when using the GPU server. This system is designed to work with the `trellis_submit_server_highscore_A6000_flash.py` server script.

## Features

### 🛡️ Race Condition Prevention
- **Buffer Times**: Configurable buffer time between server uses to prevent conflicts
- **Status Checking**: Real-time monitoring of server job status
- **Health Monitoring**: Continuous health checks to ensure server availability
- **Graceful Waiting**: Intelligent waiting for server availability with timeout protection

### 🔄 Proper Request Sequencing
- **Health First**: Always check `/health/` endpoint before proceeding
- **Job Status**: Monitor `/job/status/` to understand current server state
- **Cache Management**: Automatic GPU cache clearing when needed
- **Error Recovery**: Robust error handling and retry mechanisms

### 📊 Server State Management
- **Busy Detection**: Identifies when server is processing, generating, or validating
- **Buffer Tracking**: Maintains timing to enforce buffer periods
- **Connection Monitoring**: Handles network issues and timeouts gracefully

## Server Endpoints Used

The coordination system uses these endpoints from the GPU server:

### `/health/`
- **Purpose**: Basic health check
- **Method**: GET
- **Expected Response**: `{"status": "healthy", "timestamp": 1234567890}`

### `/job/status/`
- **Purpose**: Check current job processing status
- **Method**: GET
- **Expected Response**: 
```json
{
  "job_id": "job_123",
  "status": "idle|processing|generating|validating|completed",
  "prompt": "current prompt being processed",
  "start_time": 1234567890,
  "end_time": 1234567890,
  "processing_time": 30.5,
  "error": null
}
```

### `/clear_cache/`
- **Purpose**: Clear GPU memory cache
- **Method**: POST
- **Expected Response**: `{"status": "cache_cleared"}`

## Configuration

### ServerCoordinator Parameters

```python
coordinator = ServerCoordinator(
    server_url="http://localhost:8096",        # Base URL of GPU server
    buffer_time_seconds=30,                    # Buffer time between uses
    max_wait_time_seconds=300,                 # Max time to wait for availability
    status_check_interval=5                    # Interval between status checks
)
```

### EpisodicPromptOptimizer Parameters

```python
optimizer = EpisodicPromptOptimizer(
    num_episodes=30,
    target_score=0.85,
    max_rounds_per_prompt=5,
    server_url="http://localhost:8096",        # GPU server URL
    server_buffer_time=30                      # Buffer time in seconds
)
```

## Usage Flow

### 1. Episode Start
```python
# Check server health at episode start
server_status = coordinator.check_server_status()
if not server_status.get("available", False):
    # Wait for server to become healthy
    coordinator.wait_for_server_availability()
```

### 2. Before Each Prompt Optimization
```python
# Wait for server availability before starting optimization
if not self._wait_for_server_availability():
    # Skip this prompt if server is not available
    continue
```

### 3. During Optimization
```python
# If validation fails with 0.0 score (likely CUDA OOM)
if final_score == 0.0:
    # Clear server GPU cache
    coordinator.clear_server_cache()
    # Retry optimization
```

### 4. Cleanup
```python
# Always perform cleanup on exit
optimizer.cleanup()
```

## Server Status States

### Available States
- `idle`: Server is ready to accept new jobs
- `completed`: Previous job finished successfully

### Busy States
- `processing`: Server is processing a job
- `generating`: Server is generating 3D models
- `validating`: Server is validating results

### Error States
- `unhealthy`: Health check failed
- `timeout`: Request timed out
- `connection_error`: Cannot connect to server
- `buffer_time`: Still within buffer period after last use

## Testing

Use the provided test script to verify server coordination:

```bash
python test_server_coordination.py
```

This script will:
1. Test all server endpoints
2. Verify buffer time functionality
3. Check server availability waiting
4. Test cache clearing
5. Validate error handling

## Best Practices

### 1. Always Check Server Health
```python
# Check health before starting any operations
status = coordinator.check_server_status()
if status.get("status") == "unhealthy":
    # Handle unhealthy server
    pass
```

### 2. Use Appropriate Buffer Times
- **Short operations**: 10-30 seconds
- **Long operations**: 60-120 seconds
- **Heavy GPU operations**: 120+ seconds

### 3. Handle Timeouts Gracefully
```python
# Set reasonable timeouts
coordinator = ServerCoordinator(
    max_wait_time_seconds=300,  # 5 minutes max wait
    status_check_interval=5     # Check every 5 seconds
)
```

### 4. Implement Proper Cleanup
```python
try:
    # Your optimization code
    pass
finally:
    # Always cleanup
    optimizer.cleanup()
```

### 5. Monitor Server State
```python
# Log server state changes
if status.get("status") in ("processing", "generating", "validating"):
    logger.info(f"Server busy: {status['status']} - {status.get('prompt', 'unknown')}")
```

## Error Handling

### Common Errors and Solutions

#### Connection Errors
```
Error: Cannot connect to server
Solution: Check if server is running on correct port
```

#### Timeout Errors
```
Error: Server status check timed out
Solution: Increase timeout or check server load
```

#### Health Check Failures
```
Error: Health check failed: HTTP 500
Solution: Restart the GPU server
```

#### Buffer Time Violations
```
Error: Server still in buffer time
Solution: Wait for buffer period to complete
```

## Integration with Other Processes

The coordination system is designed to work alongside other processes that use the GPU server:

### Continuous Trellis Orchestrator
- Both systems check server status before using it
- Buffer times prevent conflicts
- Health monitoring ensures server stability

### Manual Server Usage
- Buffer times allow for manual server access
- Status checking prevents interrupting ongoing operations
- Graceful waiting for server availability

## Monitoring and Logging

The system provides comprehensive logging:

```
🔍 Checking server availability before optimization...
⏳ Server busy: processing (job: job_123, prompt: red car...)
✅ Server is available (status: idle)
📝 Marked server as used at 1234567890.123
🧹 GPU cache cleared successfully
```

## Troubleshooting

### Server Not Responding
1. Check if server is running: `curl http://localhost:8096/health/`
2. Check server logs for errors
3. Restart server if necessary

### Buffer Time Issues
1. Check buffer time configuration
2. Verify server status endpoint responses
3. Adjust buffer time if needed

### Cache Clearing Failures
1. Check server cache endpoint
2. Verify server has GPU memory management
3. Check server logs for GPU errors

## Performance Considerations

### Buffer Time Impact
- Longer buffer times = fewer conflicts but slower throughput
- Shorter buffer times = faster throughput but potential conflicts
- Balance based on server capacity and usage patterns

### Status Check Frequency
- More frequent checks = faster response but more server load
- Less frequent checks = lower server load but slower response
- Recommended: 5-10 second intervals

### Timeout Settings
- Too short = premature failures
- Too long = slow error recovery
- Recommended: 300 seconds max wait time 