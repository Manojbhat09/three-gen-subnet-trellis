# Episodic Prompt Optimization System

## Overview

The Episodic Prompt Optimization System is a wrapper around the V4.1 RL Loop optimizer that enables multi-episode learning across a fixed set of prompts. The agent learns principles and strategies that transfer across different prompt types and episodes, demonstrating true reinforcement learning behavior.

## Key Features

- **Multi-Episode Learning**: Agent improves across episodes by learning from previous optimization sessions
- **Cross-Prompt Principle Extraction**: Learns generalizable principles that apply to different prompt types
- **Progressive Strategy Refinement**: Strategy performance tracking with epsilon-greedy exploration/exploitation
- **Comprehensive Logging**: Detailed episode statistics, learning trends, and principle extraction
- **Persistent Memory**: Knowledge persists across sessions and episodes
- **Convergence Analysis**: Tracks improvement patterns and learning efficiency over time

## System Architecture

```
EpisodicPromptOptimizer
├── RLLoopAgent (V4.1 RL Loop)
│   ├── Strategy Selection (Epsilon-Greedy)
│   ├── Multi-Round Optimization
│   ├── Score-Based Learning
│   └── Principle Extraction
├── Episode Management
│   ├── Prompt Cycling
│   ├── Session Tracking
│   └── Cross-Episode Analysis
└── Logging & Analytics
    ├── Episode Statistics
    ├── Learning Trends
    └── Principle Accumulation
```

## Files

1. **`episodic_prompt_optimizer.py`** - Main episodic optimization system
2. **`run_episodic_optimization.py`** - Configurable launcher script
3. **`demo_episodic_optimization.py`** - Quick 3-episode demo
4. **`test_episodic_system.py`** - System validation script

## Test Prompts

The system cycles through 13 diverse prompts each episode:

```python
test_prompts = [
    "sapphire-studded sharp spear",
    "emerald pendant",
    "bottle of red wine with cork in it",
    "crystal staff with swirling light",
    "harp adorned with pearl inlays and gilded frame",
    "necklace with heart-shaped pendant made of silver and turquoise stones",
    "bottle of red wine with cork in it",
    "cupcake with chocolate icing on top",
    "matte black candle holder two interlocking pieces",
    "greek kylix cup black-figure technique mythological scenes",
    "small round blue creature with long nose and pointed ears",
    "tall glass of layered lemonade",
    "cylindrical glass of bubbly lemonade"
]
```

## Usage

### Quick Test
```bash
# Verify system works
python test_episodic_system.py

# Run quick test with one prompt
python test_episodic_system.py --run
```

### Demo (3 Episodes)
```bash
python demo_episodic_optimization.py
```

### Full Run (30 Episodes - Default)
```bash
# Default: 30 episodes, target 0.85, max 5 rounds per prompt
python run_episodic_optimization.py

# Custom configuration
python run_episodic_optimization.py --episodes 10 --target 0.90 --max-rounds 3
```

### Direct Python Usage
```python
from episodic_prompt_optimizer import EpisodicPromptOptimizer

# Create optimizer
optimizer = EpisodicPromptOptimizer(
    num_episodes=30,
    target_score=0.85,
    max_rounds_per_prompt=5,
    log_dir="my_episodic_logs"
)

# Run all episodes
results = optimizer.run_all_episodes()

# Access results
learning_analysis = results['learning_analysis']
episode_results = results['episode_results']
```

## Configuration Options

| Parameter | Default | Description |
|-----------|---------|-------------|
| `num_episodes` | 30 | Number of episodes to run |
| `target_score` | 0.85 | Target validation score for each prompt |
| `max_rounds_per_prompt` | 5 | Maximum optimization rounds per prompt |
| `log_dir` | "episodic_logs" | Directory for storing logs and results |

## Output Structure

### Episode Results
```json
{
  "episode": 1,
  "start_time": "2024-01-01T12:00:00",
  "prompt_results": [
    {
      "prompt": "emerald pendant",
      "rounds_used": 2,
      "initial_score": 0.756,
      "final_score": 0.872,
      "score_improvement": 0.116,
      "converged": true,
      "optimized_prompt": "detailed emerald pendant..."
    }
  ],
  "episode_summary": {
    "success_rate": 0.85,
    "avg_rounds_per_prompt": 2.3,
    "avg_score_improvement": 0.073,
    "principles_learned": ["material_focus", "artistic_elaboration"]
  }
}
```

### Learning Analysis
```json
{
  "success_rate_trend": {
    "first_5_episodes": 0.72,
    "last_5_episodes": 0.89,
    "overall_average": 0.83
  },
  "efficiency_trend": {
    "first_5_episodes_avg_rounds": 3.2,
    "last_5_episodes_avg_rounds": 2.1
  },
  "total_principles_learned": 45,
  "unique_principles": 12
}
```

## Learning Behavior

### Episode-to-Episode Learning
- **Strategy Performance Tracking**: Each strategy's success rate influences future selection
- **Principle Accumulation**: Successful optimization patterns become reusable principles
- **Exploration vs Exploitation**: Epsilon-greedy balances trying new strategies vs using proven ones

### Expected Learning Patterns
1. **Early Episodes**: High exploration, varied performance, many new principles
2. **Middle Episodes**: Balanced exploration/exploitation, stabilizing performance
3. **Late Episodes**: Efficient exploitation of learned strategies, consistent high performance

### Convergence Indicators
- **Increasing Success Rate**: More prompts reach target score
- **Decreasing Average Rounds**: Faster optimization due to better strategy selection
- **Stable Score Improvements**: Consistent optimization quality
- **Principle Reuse**: Fewer new principles, more application of existing ones

## Integration with Existing Systems

The episodic optimizer is designed to work seamlessly with existing infrastructure:

- **Validator Integration**: Uses the same `subnet_accurate_validator.py` for scoring
- **Model Compatibility**: Works with any Ollama-compatible model
- **Memory Persistence**: Builds on V4.1 RL Loop's memory system
- **Logging Format**: Compatible with existing log analysis tools

## Performance Expectations

Based on V4.1 RL Loop testing:

- **Individual Optimization**: 0.04-2 seconds per round (depending on prompt complexity)
- **Episode Duration**: ~15-45 seconds per episode (13 prompts × 1-3 rounds average)
- **30 Episodes**: ~15-25 minutes total runtime
- **Memory Usage**: <100MB for agent memory, logs scale with episodes
- **Learning Convergence**: Significant improvement typically visible by episode 10-15

## Troubleshooting

### Common Issues

1. **Import Errors**: Ensure `smart_prompt_optimizer_v4_1_rl_loop.py` is in the same directory
2. **Validator Errors**: Check that conda environment and validator script are accessible
3. **Memory Issues**: For very long runs, monitor disk space for logs
4. **Ollama Connection**: Ensure Ollama is running on localhost:11434

### Debug Commands
```bash
# Test system configuration
python test_episodic_system.py

# Check validator works
source /home/mbhat/miniconda/bin/activate && conda activate trellis_new && python subnet_accurate_validator.py "test prompt"

# Run minimal demo
python demo_episodic_optimization.py
```

## Expected Results

A successful 30-episode run should demonstrate:

- **Success Rate**: 70-95% of prompts reaching target score by final episodes
- **Efficiency**: Average rounds decreasing from ~3.5 to ~2.0
- **Learning**: 40-60 total principles learned, 10-15 unique principles
- **Consistency**: Lower variance in performance across episodes
- **Transfer**: Principles learned on one prompt type helping optimize others

## File Outputs

After running, check the log directory for:
- `episodic_run_YYYYMMDD_HHMMSS.log` - Detailed execution log
- `episodes_1_to_N_results.json` - Intermediate results after each episode
- `final_episodic_results_YYYYMMDD_HHMMSS.json` - Complete final results
- `episodic_memory.json` - Persistent agent memory

This system demonstrates true reinforcement learning where the agent genuinely improves its optimization capabilities through experience across multiple episodes and prompt types. 