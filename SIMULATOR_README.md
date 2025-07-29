# TRELLIS Orchestrator Simulator

This simulator allows you to test the TRELLIS 3D generation system using prompts from a file, similar to how the continuous orchestrator works but without requiring Bittensor network access.

## Features

- **Prompt File Processing**: Load prompts from a Python file containing an `EPISODIC_TEST_PROMPTS` list
- **3D Model Generation**: Generate 3D models using the local TRELLIS server
- **Local Validation**: Optionally validate generated models using the validation server
- **Prompt Optimization**: Support for both reproducibility and traditional prompt optimization
- **Task Deduplication**: SQLite database to track processed prompts and avoid duplicates
- **Comprehensive Logging**: Detailed logs of generation and validation processes
- **Statistics Tracking**: Track performance metrics and optimization effectiveness

## Quick Start

### 1. Basic Simulation

```bash
# Run simulation with prompts from episodic_test_prompts.py
./run_trellis_mining.sh --simulate --promptfile episodic_test_prompts.py --start-server
```

### 2. Direct Python Usage

```bash
# Run simulator directly with Python
python3 continuous_trellis_orchestrator_simulator.py --promptfile episodic_test_prompts.py
```

### 3. Test Run

```bash
# Run a quick test with simple prompts
python3 test_simulator.py
```

## Command Line Options

### Required Arguments
- `--promptfile FILE`: Path to Python file containing `EPISODIC_TEST_PROMPTS` list

### Optional Arguments
- `--no-validate`: Disable local validation of generated models
- `--generation-server URL`: TRELLIS generation server URL (default: http://localhost:8096)
- `--validation-server URL`: Validation server URL (default: http://localhost:10006)
- `--output-dir DIR`: Output directory for logs and models (default: ./trellis_simulation_outputs)

### Optimization Options
- `--no-optimize`: Disable prompt optimization
- `--aggressive-optimize`: Enable aggressive optimization mode
- `--quiet-optimize`: Reduce optimization logging detail
- `--no-reproducibility`: Disable reproducibility optimization
- `--reproducibility-similarity FLOAT`: Minimum similarity threshold for reproducibility (default: 0.3)
- `--variable-seeds`: Use prompt-hash based seeds (default: fixed seed 42)
- `--seed INT`: Fixed seed to use when not using variable seeds (default: 42)

## Prompt File Format

Create a Python file with the following format:

```python
# my_prompts.py
EPISODIC_TEST_PROMPTS = [
    "simple red cube",
    "blue sphere with white stripes", 
    "green cylinder",
    "wooden table with four legs",
    # Add more prompts here...
]
```

## Examples

### Basic Simulation
```bash
# Run with default settings
./run_trellis_mining.sh --simulate --promptfile episodic_test_prompts.py --start-server
```

### Simulation with Custom Settings
```bash
# Run with aggressive optimization and variable seeds
python3 continuous_trellis_orchestrator_simulator.py \
    --promptfile episodic_test_prompts.py \
    --aggressive-optimize \
    --variable-seeds \
    --no-validate
```

### Simulation with Reproducibility Optimization
```bash
# Run with reproducibility optimization enabled
python3 continuous_trellis_orchestrator_simulator.py \
    --promptfile episodic_test_prompts.py \
    --reproducibility-similarity 0.5 \
    --output-dir ./my_simulation_outputs
```

## Output

The simulator creates the following outputs:

1. **Generated Models**: SPZ-compressed PLY files in the output directory
2. **Logs**: Detailed logs in `continuous_trellis_simulator.log`
3. **Database**: SQLite database for task tracking (`trellis_simulator_tasks.db`)
4. **Statistics**: Console output with performance metrics

## Database Schema

The simulator uses a SQLite database to track:
- Processed prompts (to avoid duplicates)
- Generation and validation times
- Local validation scores
- Optimization statistics

## Differences from Continuous Orchestrator

The simulator is similar to the continuous orchestrator but with these key differences:

1. **No Bittensor Integration**: Doesn't connect to the Bittensor network
2. **File-based Prompts**: Uses prompts from a file instead of harvesting from validators
3. **No Submission**: Doesn't submit results to validators
4. **Simplified Task Management**: Focused on generation and validation only
5. **Simulation-specific Logging**: Logs tailored for simulation scenarios

## Troubleshooting

### Common Issues

1. **TRELLIS Server Not Running**
   ```bash
   # Start the server first
   python trellis_submit_server.py --port 8096
   ```

2. **Validation Server Not Available**
   ```bash
   # Run without validation
   python3 continuous_trellis_orchestrator_simulator.py --promptfile my_prompts.py --no-validate
   ```

3. **Prompt Optimization Dependencies Missing**
   ```bash
   # Run without optimization
   python3 continuous_trellis_orchestrator_simulator.py --promptfile my_prompts.py --no-optimize
   ```

4. **Database Lock Issues**
   ```bash
   # Remove the database file to start fresh
   rm trellis_simulator_tasks.db
   ```

### Performance Tips

1. **Skip Validation for Speed**: Use `--no-validate` for faster testing
2. **Disable Optimization**: Use `--no-optimize` to focus on generation speed
3. **Use Fixed Seeds**: Use `--seed 42` for consistent, reproducible results
4. **Limit Prompt Count**: Edit your prompt file to include fewer prompts for quick testing

## Integration with Shell Script

The simulator integrates with the main `run_trellis_mining.sh` script:

```bash
# Basic simulation mode
./run_trellis_mining.sh --simulate --promptfile episodic_test_prompts.py

# With server auto-start
./run_trellis_mining.sh --simulate --promptfile episodic_test_prompts.py --start-server

# With validation disabled
./run_trellis_mining.sh --simulate --promptfile episodic_test_prompts.py --no-validate
```

This provides a unified interface for all TRELLIS mining operations while maintaining the flexibility of direct Python execution for advanced use cases. 