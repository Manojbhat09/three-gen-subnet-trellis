# vLLM Integration for Continuous TRELLIS Orchestrator

This document explains how to use the new vLLM optimization features added to `continuous_trellis_orchestrator_lora_working.py`.

## Overview

The orchestrator now supports vLLM-based prompt optimization, which can provide higher quality prompt enhancements compared to traditional methods. vLLM optimization is designed to work alongside existing optimization systems and can be configured to use different approaches.

## New Command Line Arguments

### vLLM Optimization Arguments

- `--vllm-optim`: Enable vLLM for prompt optimization (bypasses local optimizer)
- `--vllm-optim-port`: vLLM port for prompt optimization (default: 11300)
- `--system-prompt`: Use system prompts during inference (activates trained behavior)
- `--vllm-priority`: vLLM optimization priority method (choices: `system_chat`, `system_completions`, `no_system`, default: `system_chat`)

### LoRA Selection Arguments

- `--lora`: Default LoRA to use when router is not available (default: "Cinema Style")

## vLLM Optimization Methods

The orchestrator supports three vLLM optimization approaches:

### 1. System Chat (`system_chat`)
- Uses structured chat format with system/user/assistant messages
- Most reliable method with clear role separation
- Recommended for production use

### 2. System Completions (`system_completions`)
- Uses completions endpoint with system prompt formatting
- Compatible with older vLLM versions
- Good fallback option

### 3. No System (`no_system`)
- Direct prompt-to-completion without system instructions
- Fastest method but may be less consistent
- Useful for simple optimizations

## Example Commands

### Basic vLLM Optimization
```bash
python continuous_trellis_orchestrator_lora_working.py \
  --vllm-optim \
  --vllm-optim-port 11300 \
  --vllm-priority system_chat
```

### vLLM with System Prompts
```bash
python continuous_trellis_orchestrator_lora_working.py \
  --vllm-optim \
  --system-prompt \
  --vllm-priority system_chat \
  --vllm-optim-port 11300
```

### vLLM with Custom LoRA
```bash
python continuous_trellis_orchestrator_lora_working.py \
  --vllm-optim \
  --vllm-priority system_chat \
  --vllm-optim-port 11300 \
  --lora "Cinema Style"
```

### vLLM with No LoRA Routing
```bash
python continuous_trellis_orchestrator_lora_working.py \
  --vllm-optim \
  --no-lora-routing \
  --vllm-priority system_chat \
  --vllm-optim-port 11300 \
  --lora "cinema"
```

## Configuration

### Default Settings
- vLLM optimization: **DISABLED** by default
- vLLM port: **11300**
- Priority method: **system_chat**
- System prompts: **DISABLED** by default
- Default LoRA: **Cinema Style**

### Environment Requirements
- vLLM server running on the specified port
- Model: `llama-3-2-3b-it` (configurable)
- Network access to vLLM server

## How It Works

1. **Connection Test**: Orchestrator tests vLLM connection before optimization
2. **Priority Method**: Tries the configured priority method first
3. **Fallback Methods**: If priority fails, tries other methods automatically
4. **Integration**: vLLM results are integrated into the existing optimization pipeline
5. **Statistics**: Comprehensive tracking of vLLM performance and success rates

## Statistics and Monitoring

The orchestrator tracks detailed vLLM statistics:

- Total optimizations attempted
- Success rates for each method
- Connection test results
- Failure counts and reasons

View statistics in the status output or logs.

## Fallback Behavior

If vLLM optimization fails:
1. Falls back to reproducibility optimization (if enabled)
2. Falls back to traditional optimization (if available)
3. Uses original prompt as last resort

## Testing

Use the provided test script to verify vLLM integration:

```bash
python test_vllm_integration.py
```

This will test:
- Connection to vLLM server
- All three optimization methods
- Main optimization function

## Troubleshooting

### Common Issues

1. **Connection Failed**
   - Verify vLLM server is running
   - Check port number (default: 11300)
   - Ensure network connectivity

2. **Optimization Failed**
   - Check vLLM server logs
   - Verify model availability
   - Check prompt format compatibility

3. **Performance Issues**
   - Monitor vLLM server resources
   - Adjust timeout settings if needed
   - Consider using faster optimization methods

### Debug Information

Enable detailed logging to see:
- Connection test results
- Optimization method selection
- Fallback behavior
- Performance metrics

## Performance Considerations

- vLLM optimization adds network latency
- System prompts may be slower but more reliable
- No-system method is fastest but less consistent
- Consider caching optimized prompts for repeated use

## Integration with Existing Features

vLLM optimization works alongside:
- LoRA routing
- Reproducibility optimization
- Traditional optimization
- Prompt cleaning
- All existing orchestrator features

## Future Enhancements

Potential improvements:
- Batch optimization for multiple prompts
- Caching of vLLM results
- Adaptive method selection based on performance
- Support for additional vLLM models
- Custom system prompt templates

