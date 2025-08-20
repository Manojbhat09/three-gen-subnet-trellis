# Ollama Coordination System for Multiple RL Runners

## Overview

This system provides priority-based queuing and coordination for multiple RL (Reinforcement Learning) runners that share a single Ollama server. It prevents timeouts, race conditions, and ensures efficient resource utilization.

## Problem Solved

When running multiple instances of `episodic_prompt_optimizer.py` on different GPUs but sharing one Ollama server, you can encounter:

- **Timeouts**: Multiple processes trying to use Ollama simultaneously
- **Race conditions**: Conflicts between concurrent requests
- **Resource waste**: Inefficient queuing and processing
- **Process failures**: Uncoordinated access leading to crashes

## Solution: OllamaCoordinator

The `OllamaCoordinator` class provides:

1. **Priority-based queuing**: HIGH (1), MEDIUM (2), LOW (3) priority levels
2. **Concurrent request management**: Up to 2 concurrent requests allowed
3. **Automatic timeout handling**: Prevents hanging requests
4. **Thread-safe operations**: Safe for multi-process environments
5. **Real-time monitoring**: Queue status and request tracking
6. **Granular resource management**: Coordinate access at the round level, not just episode level

## Key Features

### Priority Levels
- **HIGH (1)**: Critical operations, can interrupt lower priority tasks
- **MEDIUM (2)**: Normal operations, wait for server to be free
- **LOW (3)**: Background operations, lowest priority

### Efficiency Improvements
- **Granular Control**: Ollama access is managed at the round level, not episode level
- **Resource Sharing**: Multiple RL runners can share Ollama efficiently during validation phases
- **Minimal Idle Time**: Ollama is only occupied during actual prompt generation (1-5 seconds)
- **Validation Overlap**: Long validation runs happen while Ollama is free for other processes

### Request Lifecycle
1. **Request Access**: Submit request with priority and description
2. **Wait for Access**: Block until access is granted
3. **Use Ollama**: Perform RL optimization work (typically 1-5 seconds)
4. **Release Access**: Free up the slot for other requests
5. **Validation Phase**: Long-running validation happens while Ollama is free for other processes

### Automatic Management
- Background queue processor
- Automatic cleanup of timed-out requests
- Health monitoring of Ollama server
- Graceful shutdown handling

## Usage Examples

### Basic Usage in EpisodicPromptOptimizer

The system is automatically integrated into `episodic_prompt_optimizer.py`:

```python
# Initialize with Ollama coordination
optimizer = EpisodicPromptOptimizer(
    num_episodes=30,
    target_score=0.85,
    max_rounds_per_prompt=5,
    server_url="http://localhost:8096",
    ollama_url="http://localhost:11434"  # Ollama server URL
)

# The system automatically:
# 1. Requests Ollama access before each optimization ROUND (not episode)
# 2. Waits for access to be granted
# 3. Uses Ollama for prompt generation (1-5 seconds)
# 4. Releases access immediately after generation
# 5. Runs validation while Ollama is free for other processes
# 6. Handles timeouts gracefully with fallback prompts
```

### Manual Usage

```python
from episodic_prompt_optimizer import OllamaCoordinator

# Initialize coordinator
coordinator = OllamaCoordinator(
    ollama_url="http://localhost:11434",
    max_wait_time_seconds=300,
    status_check_interval=2,
    priority_timeout_seconds=60
)

# Request access
request_id = coordinator.request_access(
    priority=2,  # MEDIUM priority
    description="RL optimization for prompt X"
)

# Wait for access
if coordinator.wait_for_access(request_id):
    try:
        # Use Ollama for RL optimization
        result = perform_rl_optimization()
    finally:
        # Always release access
        coordinator.release_access(request_id)
else:
    print("Failed to get Ollama access")
```

### Multiple RL Runners

```python
import threading
from episodic_prompt_optimizer import OllamaCoordinator

def rl_runner(runner_id, coordinator):
    for i in range(5):
        # Request access
        request_id = coordinator.request_access(
            priority=2,
            description=f"Runner {runner_id} - Request {i}"
        )
        
        if coordinator.wait_for_access(request_id):
            try:
                # Do RL work
                time.sleep(random.uniform(2, 8))
            finally:
                coordinator.release_access(request_id)

# Start multiple runners
coordinator = OllamaCoordinator()
runners = []
for i in range(3):
    runner = threading.Thread(target=rl_runner, args=(i, coordinator))
    runners.append(runner)
    runner.start()

# Wait for completion
for runner in runners:
    runner.join()
```

## Configuration Options

### OllamaCoordinator Parameters

```python
coordinator = OllamaCoordinator(
    ollama_url="http://localhost:11434",      # Ollama server URL
    max_wait_time_seconds=300,                # Max wait for access
    status_check_interval=2,                  # Status check frequency
    priority_timeout_seconds=60               # Request timeout
)
```

### Priority Guidelines

- **HIGH (1)**: Critical validation, emergency operations
- **MEDIUM (2)**: Normal RL optimization, prompt generation
- **LOW (3)**: Background analysis, logging, cleanup

## Monitoring and Debugging

### Queue Status

```python
status = coordinator.get_queue_status()
print(f"Queue: {status['queue_length']} pending, {status['active_requests']} active")

for req in status['queue']:
    print(f"Pending: {req['description']} (Priority: {req['priority']})")

for req in status['active']:
    print(f"Active: {req['description']} (Priority: {req['priority']})")
```

### Health Monitoring

```python
health = coordinator.check_ollama_status()
if health['available']:
    print(f"Ollama healthy: {health['status']}")
else:
    print(f"Ollama issue: {health['error']}")
```

## Integration with Existing Code

### Automatic Integration

The system automatically integrates with your existing `episodic_prompt_optimizer.py`:

1. **No code changes required** - just use the existing interface
2. **Automatic Ollama coordination** - happens behind the scenes at the round level
3. **Graceful fallback** - if Ollama is unavailable, fallback prompts are used
4. **Resource cleanup** - automatic cleanup after each round
5. **Efficient sharing** - multiple RL runners can share Ollama during validation phases

### New Workflow

**Before (Episode-level coordination):**
```
Episode 1: Get Ollama → Round 1 → Round 2 → Round 3 → Validate → Release Ollama
Episode 2: Get Ollama → Round 1 → Round 2 → Round 3 → Validate → Release Ollama
```

**After (Round-level coordination):**
```
Episode 1: Round 1: Get Ollama → Generate → Release Ollama → Validate (long)
Episode 2: Round 1: Get Ollama → Generate → Release Ollama → Validate (long)
Episode 1: Round 2: Get Ollama → Generate → Release Ollama → Validate (long)
Episode 2: Round 2: Get Ollama → Generate → Release Ollama → Validate (long)
```

This allows **efficient resource sharing** where Ollama is only occupied during the brief generation phase.

### Manual Integration

If you want to use the coordinator in other scripts:

```python
# Import the coordinator
from episodic_prompt_optimizer import OllamaCoordinator

# Use it in your custom RL loops
coordinator = OllamaCoordinator()
# ... your custom logic
```

## Best Practices

### 1. Always Release Access

```python
request_id = coordinator.request_access(priority=2, description="My work")
try:
    if coordinator.wait_for_access(request_id):
        # Do your work
        pass
finally:
    # Always release, even on error
    coordinator.release_access(request_id)
```

### 2. Use Appropriate Priorities

```python
# High priority for critical operations
coordinator.request_access(priority=1, description="Critical validation")

# Medium priority for normal work
coordinator.request_access(priority=2, description="RL optimization")

# Low priority for background tasks
coordinator.request_access(priority=3, description="Logging and cleanup")
```

### 3. Handle Timeouts Gracefully

```python
if coordinator.wait_for_access(request_id):
    # Do work
    pass
else:
    # Handle timeout - maybe retry later or skip
    print("Ollama access timeout - will retry later")
```

### 4. Monitor Queue Health

```python
# Check queue status periodically
status = coordinator.get_queue_status()
if status['queue_length'] > 10:
    print("Warning: Long queue detected")
```

## Troubleshooting

### Common Issues

1. **Timeout errors**: Increase `max_wait_time_seconds`
2. **Queue not processing**: Check if Ollama server is responding
3. **Memory leaks**: Ensure `release_access()` is always called
4. **High latency**: Reduce `status_check_interval`

### Debug Mode

Enable detailed logging:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

### Health Checks

```python
# Check Ollama server health
health = coordinator.check_ollama_status()
print(f"Health: {health}")

# Check queue status
queue_status = coordinator.get_queue_status()
print(f"Queue: {queue_status}")
```

## Performance Considerations

### Concurrent Requests

- **Default**: 2 concurrent requests allowed
- **Adjustable**: Modify the `_process_queue` method if needed
- **Optimal**: 2-3 concurrent requests for most Ollama setups

### Timeout Values

- **Request timeout**: 60 seconds (adjustable per request)
- **Wait timeout**: 300 seconds (adjustable globally)
- **Status check**: 2 seconds (adjustable for responsiveness)

### Memory Usage

- **Minimal overhead**: ~1-2 MB per coordinator instance
- **Queue memory**: Scales with number of pending requests
- **Cleanup**: Automatic cleanup prevents memory leaks

## Example Scenarios

### Scenario 1: Two RL Runners on Different GPUs

```bash
# Terminal 1 - GPU 0
CUDA_VISIBLE_DEVICES=0 python episodic_prompt_optimizer.py

# Terminal 2 - GPU 1  
CUDA_VISIBLE_DEVICES=1 python episodic_prompt_optimizer.py
```

Both will automatically coordinate Ollama access without conflicts.

### Scenario 2: High-Priority Validation

```python
# Emergency validation gets priority
request_id = coordinator.request_access(
    priority=1,
    description="Emergency validation - critical prompt"
)
```

### Scenario 3: Background Analysis

```python
# Background work gets lowest priority
request_id = coordinator.request_access(
    priority=3,
    description="Background prompt analysis"
)
```

## Conclusion

The Ollama coordination system provides robust, efficient management of multiple RL runners sharing a single Ollama server. It prevents timeouts, ensures fair resource allocation, and maintains system stability under high load.

Key benefits:
- ✅ **No more timeouts** - coordinated access prevents conflicts
- ✅ **Efficient queuing** - priority-based processing
- ✅ **Automatic cleanup** - no resource leaks
- ✅ **Easy integration** - works with existing code
- ✅ **Real-time monitoring** - visibility into system state

For questions or issues, check the logs for detailed information about queue status and request processing.
