# Miner-Integrated Validator Simulation

This advanced simulation system integrates directly with your actual `ContinuousTrellisOrchestrator` miner to provide realistic testing of the complete 3D generation subnet workflow.

## 🎯 **What Makes This Special**

Unlike generic simulations, this system:

- **Uses Your Actual Miner**: Integrates with `ContinuousTrellisOrchestrator` components
- **Real Task Processing**: Uses actual `TaskRecord`, `ValidatorState`, and `TaskDatabase` classes
- **Priority Access Simulation**: Tests your priority server coordination system
- **Real Validation Logic**: Uses your actual validation and submission pipelines
- **Configurable Scenarios**: Test different load patterns and configurations

## 🏗️ **Architecture Integration**

The simulation integrates with these key miner components:

```python
from continuous_trellis_orchestrator_lora_working import (
    ContinuousTrellisOrchestrator,  # Your main miner class
    TaskRecord,                      # Task tracking and metadata
    ValidatorState,                  # Validator state management
    TaskDatabase,                    # Task persistence and tracking
    ValidatorStatePersistence        # State persistence layer
)
```

## 🚀 **Quick Start**

### **1. Basic Miner Integration**

```bash
# Make the script executable
chmod +x start_miner_simulation.sh

# Run the complete simulation
./start_miner_simulation.sh
```

This will:
- ✅ Import your actual miner components
- ✅ Initialize `ContinuousTrellisOrchestrator` with simulation config
- ✅ Create realistic validator states
- ✅ Run 10-minute simulation with 5 validators
- ✅ Use your actual task processing pipeline
- ✅ Save results to `miner_integrated_simulation_results.json`

### **2. Custom Scenarios**

```bash
# High load testing
python3 miner_integrated_simulation.py --scenario high_load --validators 8 --duration 300

# Stress testing
python3 miner_integrated_simulation.py --scenario stress_test --validators 10 --duration 180

# Learning scenario
python3 miner_integrated_simulation.py --scenario learning --validators 3 --duration 900

# Realistic subnet behavior
python3 miner_integrated_simulation.py --scenario realistic --validators 6 --duration 600
```

## 📊 **Simulation Scenarios**

### **Balanced (Default)**
- **Validators**: 5
- **Duration**: 10 minutes
- **Task Pull Interval**: 30 seconds
- **Prompts per Cycle**: 50
- **Purpose**: General testing and baseline performance

### **High Load**
- **Validators**: 8
- **Duration**: 5 minutes
- **Task Pull Interval**: 15 seconds
- **Prompts per Cycle**: 100
- **Purpose**: Test system under moderate stress

### **Stress Test**
- **Validators**: 10
- **Duration**: 3 minutes
- **Task Pull Interval**: 10 seconds
- **Prompts per Cycle**: 150
- **Purpose**: Maximum stress testing

### **Learning**
- **Validators**: 3
- **Duration**: 15 minutes
- **Task Pull Interval**: 60 seconds
- **Prompts per Cycle**: 75
- **Purpose**: Observe learning and adaptation patterns

### **Realistic**
- **Validators**: 6
- **Duration**: 10 minutes
- **Task Pull Interval**: 45 seconds
- **Prompts per Cycle**: 80
- **Purpose**: Based on actual subnet behavior patterns

## 🔧 **Configuration Options**

### **Command Line Arguments**

```bash
python3 miner_integrated_simulation.py [OPTIONS]

Options:
  --scenario SCENARIO     Simulation scenario (balanced, high_load, stress_test, learning, realistic)
  --validators N          Number of validators to simulate
  --duration SECONDS      Simulation duration in seconds
  --miner-config FILE     Path to custom miner configuration file
  --enable-submission     Enable real subnet submission (use with caution!)
```

### **Configuration File**

Create a custom miner configuration file:

```json
{
  "output_dir": "custom_simulation_outputs",
  "enable_task_tracking": true,
  "min_local_score": 0.6,
  "submission_timeout": 45,
  "generation_server_url": "http://localhost:8097",
  "priority_access_max_wait": 90,
  "priority_access_timeout": 45
}
```

Then use it:

```bash
python3 miner_integrated_simulation.py --miner-config my_config.json --scenario realistic
```

## 🔍 **What the Simulation Tests**

### **1. Miner Component Integration**
- ✅ `ContinuousTrellisOrchestrator` initialization
- ✅ `TaskDatabase` operations
- ✅ `ValidatorState` management
- ✅ `TaskRecord` creation and processing

### **2. Task Processing Pipeline**
- ✅ Task creation and assignment
- ✅ 3D model generation simulation
- ✅ Local validation simulation
- ✅ Result submission simulation
- ✅ Performance tracking

### **3. Priority Access System**
- ✅ Server availability checking
- ✅ Priority access coordination
- ✅ Task interruption handling
- ✅ Resource management

### **4. Validator State Management**
- ✅ Cooldown system simulation
- ✅ Performance tracking
- ✅ State synchronization
- ✅ Learning and adaptation

## 📈 **Understanding the Results**

### **Real-Time Output**

During simulation, you'll see:
```
=== Simulation Cycle 1 ===
📝 Starting prompt generation cycle
✅ Prompt generation cycle completed: 50 prompts
🔄 Starting task processing cycles
   Created 5 tasks for processing
🔄 Processing task sim_task_1234567890_0: 'mechanical robot with steel plating...'
✅ Task sim_task_1234567890_0 completed successfully
✅ Task processing cycles completed: 5/5 successful
```

### **Final Results**

```
🎯 MINER-INTEGRATED SIMULATION COMPLETED
============================================================
Scenario: balanced
Total Runtime: 600.0 seconds
Tasks Generated: 500
Tasks Pulled: 100
Tasks Processed: 95
Tasks Submitted: 95
Overall Average Quality Score: 0.784

📈 Performance Metrics:
   Tasks per Second: 0.16
   Generation Success Rate: 95.00%
   Validation Success Rate: 95.00%
   Submission Success Rate: 100.00%
   Average Generation Time: 12.45s
   Average Validation Time: 4.23s
```

### **Result Files**

- `miner_integrated_simulation_results.json` - Complete simulation data
- `simulation_outputs/` - Generated files and logs
- Console output - Real-time statistics and progress

## 🎮 **Advanced Usage**

### **Custom Miner Integration**

Modify the simulation to use specific miner features:

```python
# In miner_integrated_simulation.py
def _initialize_miner(self) -> Optional[ContinuousTrellisOrchestrator]:
    miner_config = {
        'output_dir': 'my_simulation_outputs',
        'enable_task_tracking': True,
        'min_local_score': 0.7,
        'submission_timeout': 60,
        # Add your custom miner settings
        'custom_setting': 'custom_value'
    }
    
    miner = ContinuousTrellisOrchestrator(miner_config)
    return miner
```

### **Real Validation Integration**

To use your actual validation engine:

```python
# Replace simulation validation with real validation
async def _validate_with_real_engine(self, task: Any, generation_result: Dict[str, Any]):
    if self.miner and hasattr(self.miner, 'validate_model'):
        # Use your actual validation method
        validation_result = await self.miner.validate_model(task, generation_result['ply_data'])
        return validation_result
    else:
        # Fall back to simulation
        return await self._simulate_model_validation(task, generation_result)
```

### **Priority Server Testing**

Test your priority access system:

```bash
# Start your TRELLIS server first
python3 trellis_subnit_server_mix_lora_flash.py

# Then run simulation with priority access
python3 miner_integrated_simulation.py --scenario high_load --enable-submission
```

## 🔒 **Safety Features**

### **Simulation Mode (Default)**
- ✅ **No real subnet submission** by default
- ✅ **Mock 3D generation** (no GPU usage)
- ✅ **Simulated validation** (no real models)
- ✅ **Safe testing** of all components

### **Production Mode (Optional)**
- ⚠️ **Real subnet submission** when `--enable-submission` is used
- ⚠️ **Actual 3D generation** with GPU resources
- ⚠️ **Real validation** with actual models
- ⚠️ **Use with caution** in production environments

## 🚨 **Troubleshooting**

### **Common Issues**

1. **Import Errors**
   ```bash
   # Check if miner file exists
   ls -la continuous_trellis_orchestrator_lora_working.py
   
   # Test import manually
   python3 -c "from continuous_trellis_orchestrator_lora_working import ContinuousTrellisOrchestrator"
   ```

2. **Configuration Errors**
   ```bash
   # Check miner configuration
   python3 -c "import json; print(json.dumps({'test': 'config'}, indent=2))"
   ```

3. **Permission Issues**
   ```bash
   # Make scripts executable
   chmod +x start_miner_simulation.sh
   chmod +x start_simulation.sh
   ```

### **Debug Mode**

Enable verbose logging:

```python
# In miner_integrated_simulation.py
logging.basicConfig(
    level=logging.DEBUG,  # Change from INFO to DEBUG
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
```

### **Component Testing**

Test individual components:

```bash
# Test miner import
python3 -c "
from continuous_trellis_orchestrator_lora_working import ContinuousTrellisOrchestrator
print('✅ Miner import successful')
"

# Test simulation components
python3 -c "
from miner_integrated_simulation import MinerIntegratedSimulation, MinerSimulationConfig
print('✅ Simulation components import successful')
"
```

## 🔄 **Continuous Integration**

### **Automated Testing**

Set up automated testing in CI/CD:

```yaml
# .github/workflows/simulation-test.yml
name: Miner Simulation Tests
on: [push, pull_request]

jobs:
  test-simulation:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - name: Run Simulation Tests
        run: |
          python3 miner_integrated_simulation.py --scenario balanced --duration 60
          python3 miner_integrated_simulation.py --scenario high_load --duration 30
```

### **Performance Monitoring**

Track performance over time:

```bash
# Run regular simulations
0 */6 * * * /path/to/start_miner_simulation.sh  # Every 6 hours

# Compare results
python3 -c "
import json
with open('miner_integrated_simulation_results.json') as f:
    data = json.load(f)
    print(f'Performance: {data[\"simulation_stats\"][\"total_tasks_processed\"]} tasks')
"
```

## 📚 **Integration Examples**

### **With Your Existing Workflow**

1. **Pre-deployment Testing**
   ```bash
   # Test before deploying to production
   ./start_miner_simulation.sh
   # Review results and performance metrics
   ```

2. **Load Testing**
   ```bash
   # Test system capacity
   python3 miner_integrated_simulation.py --scenario stress_test --validators 15
   ```

3. **Performance Optimization**
   ```bash
   # Test different configurations
   python3 miner_integrated_simulation.py --miner-config config1.json
   python3 miner_integrated_simulation.py --miner-config config2.json
   # Compare results
   ```

### **With Other Components**

- **Text Prompt Generator**: Test prompt distribution
- **Get-Prompts Service**: Test prompt retrieval
- **Validation Engine**: Test model validation
- **Priority Server**: Test access coordination

## 🎯 **Next Steps**

1. **Run Basic Simulation**: `./start_miner_simulation.sh`
2. **Test Different Scenarios**: Try various `--scenario` options
3. **Customize Configuration**: Modify miner settings
4. **Integrate Real Components**: Enable actual validation/generation
5. **Set Up Monitoring**: Regular performance tracking
6. **Optimize Performance**: Use results to improve your miner

## 🤝 **Support and Contributing**

- **Issues**: Report problems with the simulation
- **Enhancements**: Suggest new features or scenarios
- **Integration**: Help improve miner component integration
- **Documentation**: Improve guides and examples

## 📄 **License**

This simulation system is part of your Three-Gen Subnet project and follows the same licensing terms.

---

**Ready to test your miner?** 🚀

```bash
./start_miner_simulation.sh
```

This will give you a comprehensive test of your `ContinuousTrellisOrchestrator` in action!
