# Validator Prompt Simulation System

This system simulates the complete workflow of a 3D generation subnet, from prompt generation to validation, allowing you to test and analyze the performance of different components and scenarios.

## 🏗️ Architecture Overview

The simulation system consists of three main components:

1. **Text Prompt Generator** (`text-prompt-generator/`) - Generates synthetic prompts using vLLM
2. **Get-Prompts Service** (`get-prompts/`) - Central service for distributing prompts to validators
3. **Validation Engine** (`validation/`) - Validates 3D models against prompts
4. **Simulation Scripts** - Mock the entire workflow for testing and analysis

## 🚀 Quick Start

### 1. Basic Simulation

Run the basic simulation that demonstrates the core workflow:

```bash
# Make the startup script executable
chmod +x start_simulation.sh

# Run the complete simulation
./start_simulation.sh
```

This will:
- Start the get-prompts service on port 8000
- Create default prompts if none exist
- Run a 2-minute simulation with 3 validators
- Generate 25 prompts per cycle
- Save results to `simulation_results.json`

### 2. Advanced Simulation

For more detailed analysis and different scenarios:

```bash
# Balanced scenario (default)
python3 advanced_validator_simulation.py --scenario balanced --validators 5 --duration 600

# High load testing
python3 advanced_validator_simulation.py --scenario high_load --validators 8 --duration 300

# Stress testing
python3 advanced_validator_simulation.py --scenario stress_test --validators 10 --duration 180

# Learning scenario
python3 advanced_validator_simulation.py --scenario learning --validators 3 --duration 900
```

## 📊 Simulation Scenarios

### Balanced (Default)
- **Validators**: 5
- **Duration**: 10 minutes
- **Prompts per batch**: 100
- **Validation interval**: 45 seconds
- **Purpose**: General testing and baseline performance

### High Load
- **Validators**: 8
- **Duration**: 5 minutes
- **Prompts per batch**: 150
- **Validation interval**: 20 seconds
- **Purpose**: Test system under moderate stress

### Stress Test
- **Validators**: 10
- **Duration**: 3 minutes
- **Prompts per batch**: 200
- **Validation interval**: 15 seconds
- **Purpose**: Maximum stress testing

### Learning
- **Validators**: 3
- **Duration**: 15 minutes
- **Prompts per batch**: 75
- **Validation interval**: 60 seconds
- **Purpose**: Observe learning and adaptation patterns

## 🔧 Configuration

### Basic Simulation Configuration

Edit `validator_prompt_simulation.py` to modify:

```python
@dataclass
class SimulationConfig:
    num_validators: int = 3
    prompts_per_batch: int = 50
    simulation_duration: int = 300  # 5 minutes
    validation_interval: int = 30   # seconds between cycles
    generation_success_rate: float = 0.8
    validation_success_rate: float = 0.9
```

### Advanced Simulation Configuration

Edit `advanced_validator_simulation.py` to modify:

```python
@dataclass
class AdvancedSimulationConfig:
    scenario: str = "balanced"
    num_validators: int = 5
    simulation_duration: int = 600
    prompts_per_batch: int = 100
    
    # Prompt complexity distribution
    prompt_complexity_distribution: Dict[str, float] = field(default_factory=lambda: {
        "simple": 0.3,      # 3-5 words
        "medium": 0.5,      # 6-8 words  
        "complex": 0.2      # 9+ words
    })
    
    # Generation success rates by complexity
    generation_success_rates: Dict[str, float] = field(default_factory=lambda: {
        "simple": 0.95,
        "medium": 0.85,
        "complex": 0.65
    })
```

## 📈 Understanding the Results

### Basic Simulation Output

The basic simulation provides:
- Real-time statistics during execution
- Final summary with total counts and averages
- Results saved to `simulation_results.json`

### Advanced Simulation Output

The advanced simulation provides:
- Detailed complexity analysis
- Quality trend tracking
- Validator performance metrics
- Learning and adaptation patterns
- Results saved to `advanced_simulation_results.json`

### Key Metrics

1. **Generation Success Rate**: Percentage of successful 3D model generations
2. **Validation Success Rate**: Percentage of successful model validations
3. **Quality Scores**: Alignment, quality, SSIM, and LPIPS scores
4. **Complexity Distribution**: Breakdown of prompt complexity levels
5. **Performance Trends**: How validators improve over time

## 🎯 Use Cases

### 1. System Testing
- Test the get-prompts service under different loads
- Verify prompt distribution and retrieval
- Validate the complete workflow

### 2. Performance Analysis
- Measure throughput (prompts/models per second)
- Analyze success rates under different conditions
- Identify bottlenecks and optimization opportunities

### 3. Load Testing
- Test system behavior under high load
- Determine maximum capacity
- Stress test individual components

### 4. Learning Analysis
- Observe how validators adapt over time
- Analyze performance improvement patterns
- Test different learning algorithms

### 5. Quality Assessment
- Analyze validation score distributions
- Compare different prompt complexity levels
- Assess the impact of generation quality on validation

## 🔍 Monitoring and Debugging

### Real-time Monitoring

During simulation, you'll see:
- Cycle-by-cycle progress updates
- Validator performance statistics
- Prompt generation and submission status
- Model generation and validation results

### Log Analysis

Check the console output for:
- Error messages and warnings
- Performance metrics
- System status updates

### Result Files

After completion, examine:
- `simulation_results.json` - Basic simulation results
- `advanced_simulation_results.json` - Detailed analysis
- Console output for real-time insights

## 🛠️ Customization

### Adding New Prompt Categories

Edit the `AdvancedPromptGenerator` class in `advanced_validator_simulation.py`:

```python
self.prompt_categories = {
    "your_category": {
        "templates": [
            "your template with {placeholder}",
            "another template with {feature}"
        ],
        "placeholder": ["value1", "value2", "value3"],
        "feature": ["feature1", "feature2", "feature3"]
    }
}
```

### Modifying Validation Logic

Edit the `QualityScoreGenerator` class to change how validation scores are calculated:

```python
def generate_scores(self, prompt: str, complexity: str, generation_quality: float) -> Dict[str, float]:
    # Customize your scoring logic here
    pass
```

### Adding New Scenarios

Extend the `_setup_scenario` method in `AdvancedSimulationOrchestrator`:

```python
def _setup_scenario(self):
    if self.config.scenario == "your_scenario":
        self.config.validation_interval = 30
        self.config.prompts_per_batch = 125
```

## 🚨 Troubleshooting

### Common Issues

1. **Port Already in Use**
   ```bash
   # Check what's using port 8000
   lsof -i :8000
   
   # Kill the process if needed
   kill -9 <PID>
   ```

2. **Missing Dependencies**
   ```bash
   # Install required packages
   pip3 install aiohttp pybase64
   ```

3. **Service Not Starting**
   - Check if the get-prompts directory exists
   - Verify Python 3.7+ is installed
   - Check console output for error messages

### Debug Mode

Enable verbose logging by modifying the logging level:

```python
logging.basicConfig(
    level=logging.DEBUG,  # Change from INFO to DEBUG
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
```

## 📚 Integration with Real Components

### Using Real Text Prompt Generator

To integrate with the actual text-prompt-generator:

1. Configure the generator in `text-prompt-generator/configs/`
2. Modify the simulation to call the real generator
3. Use actual vLLM models for prompt generation

### Using Real Validation Engine

To integrate with the actual validation engine:

1. Import the validation components from `validation/`
2. Replace mock validation with real validation calls
3. Use actual 3D models and validation metrics

### Using Real Get-Prompts Service

The simulation already uses the real get-prompts service, so you can:
- Test with real prompt data
- Verify API endpoints and authentication
- Test scalability and performance

## 🔄 Continuous Integration

### Automated Testing

Set up automated testing by:

1. Creating test scenarios
2. Running simulations in CI/CD pipelines
3. Comparing results against baselines
4. Alerting on performance regressions

### Performance Monitoring

Track performance over time by:

1. Running regular simulations
2. Storing historical results
3. Analyzing trends and patterns
4. Setting performance benchmarks

## 📖 Further Reading

- [Text Prompt Generator Documentation](text-prompt-generator/README.md)
- [Get-Prompts Service Documentation](get-prompts/README.md)
- [Validation Engine Documentation](validation/README.md)
- [Three-Gen Subnet Documentation](three-gen-subnet/README.md)

## 🤝 Contributing

To contribute to the simulation system:

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Submit a pull request

## 📄 License

This simulation system is part of the Three-Gen Subnet project and follows the same licensing terms.
