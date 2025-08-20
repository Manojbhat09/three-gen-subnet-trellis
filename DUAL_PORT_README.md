# Dual-Port Continuous TRELLIS Orchestrator

This modification allows the continuous orchestrator to use two different ports for generation, comparing results and selecting the best one for submission.

## Overview

The dual-port system works as follows:

1. **Port 1 (default: 8097)**: Generates 3D models using optimized prompts
2. **Port 2 (default: 8098)**: Generates 3D models using original prompts
3. **Validation**: Both models are validated using the subnet validator
4. **Selection**: The model with the higher validation score is selected
5. **Submission**: The better model is submitted to the subnet

## Features

- **Dual Generation**: **PARALLEL** generation on both ports (not sequential)
- **Intelligent Comparison**: Uses subnet validation to compare quality
- **Automatic Selection**: Automatically chooses the better result
- **Comprehensive Logging**: Detailed logs for both generation and validation
- **Fallback Support**: Falls back to single-port if dual-port fails
- **Async HTTP Support**: Uses aiohttp for true parallel HTTP requests with requests fallback

## Configuration

### Command Line Arguments

```bash
python continuous_trellis_orchestrator_lora_working_multi.py \
    --port1 8097 \
    --port2 8098 \
    --generation-server http://localhost:8097 \
    --validation-server http://localhost:10006
```

### Configuration Options

- `--port1`: Port for optimized prompt generation (default: 8097)
- `--port2`: Port for original prompt generation (default: 8098)
- `--generation-server`: Base generation server URL
- `--validation-server`: Validation server URL

## Architecture

### New Methods Added

1. **`generate_3d_model_dual_port()`**: Generates on both ports **simultaneously** using async/await
2. **`validate_model_dual_port()`**: Validates both models and compares scores
3. **`_validate_with_subnet_validator()`**: Helper for subnet validation

### Parallel Generation Implementation

The dual-port system now uses **true parallel generation** instead of sequential:

- **Before**: Port 1 → wait → Port 2 → wait → Total time = sum of both
- **After**: Port 1 + Port 2 simultaneously → Total time = max of both

This is achieved using:
- `asyncio.gather()` to run both generation tasks concurrently
- `aiohttp` for async HTTP requests (with `requests` fallback)
- Proper async/await patterns throughout the generation pipeline

### Modified Methods

1. **`process_task()`**: Now uses dual-port approach by default
2. **`generate_3d_model()`**: Kept as legacy single-port method

## Usage Examples

### Basic Dual-Port Mining

```bash
# Start with default ports (8097, 8098)
python continuous_trellis_orchestrator_lora_working_multi.py

# Custom ports
python continuous_trellis_orchestrator_lora_working_multi.py \
    --port1 8095 \
    --port2 8096
```

### Testing the System

```bash
# Run the test script
python test_dual_port.py
```

## Prerequisites

1. **TRELLIS Generation Servers**: Must be running on both ports
2. **Subnet Validator**: `subnet_accurate_validator_multigpu.py` must be available
3. **Validation Server**: HTTP validation server (optional, for fallback)
4. **Ollama/vLLM**: For prompt optimization

## Server Setup

### Port 1 (Optimized Prompts)
```bash
# Start TRELLIS server on port 8097
python trellis_server.py --port 8097
```

### Port 2 (Original Prompts)
```bash
# Start TRELLIS server on port 8098
python trellis_server.py --port 8098
```

### Validation Server
```bash
# Start validation server on port 10006
python validation_server.py --port 10006
```

## Workflow

1. **Task Pull**: Orchestrator pulls task from validator
2. **Prompt Optimization**: Original prompt is optimized
3. **Dual Generation**: 
   - Port 1: Generate with optimized prompt
   - Port 2: Generate with original prompt
4. **Validation**: Both models validated using subnet validator
5. **Score Comparison**: Scores compared to select winner
6. **Submission**: Better model submitted to subnet

## Benefits

- **Quality Improvement**: Always submits the better result
- **Redundancy**: Two generation attempts per task
- **Optimization Testing**: Validates if prompt optimization helps
- **Performance Monitoring**: Tracks success rates for each port

## Monitoring

The system provides comprehensive logging:

- Generation success/failure for each port
- Validation scores for both models
- Winner selection with score comparison
- Performance metrics and timing

## Troubleshooting

### Common Issues

1. **Port Unavailable**: Ensure both ports are free and servers are running
2. **Validation Failures**: Check subnet validator availability
3. **Generation Timeouts**: Adjust timeout settings in config

### Debug Mode

Enable detailed logging:
```bash
python continuous_trellis_orchestrator_lora_working_multi.py \
    --port1 8097 \
    --port2 8098 \
    --log-level DEBUG
```

## Performance Considerations

- **Memory Usage**: Higher due to dual generation
- **Processing Time**: Slightly longer due to validation comparison
- **Network**: Requires communication with two generation servers
- **Storage**: Saves both models temporarily for comparison

## Future Enhancements

- **Weighted Scoring**: Consider generation time in selection
- **Adaptive Ports**: Automatically adjust port usage based on performance
- **Batch Processing**: Process multiple tasks simultaneously
- **Advanced Metrics**: Track long-term performance trends
