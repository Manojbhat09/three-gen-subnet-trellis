# Test Scripts for Fallback Mechanism

This directory contains test scripts to verify that the `generate_3d_model_with_fallback` function works correctly.

## 📁 Files

### 1. `test_fallback_mechanism.py`
- **Purpose**: Basic mock testing with simulated components
- **Usage**: `python test_fallback_mechanism.py`
- **Features**: 
  - Tests individual components in isolation
  - Uses mock objects for all dependencies
  - Good for basic functionality verification

### 2. `test_fallback_mechanism_real.py`
- **Purpose**: Tests the actual function from the orchestrator
- **Usage**: `python test_fallback_mechanism_real.py`
- **Features**:
  - Imports the real orchestrator
  - Mocks external dependencies (CLIP, torch, requests, etc.)
  - Tests actual function execution
  - Performance benchmarking
  - Comprehensive error handling

## 🚀 Quick Start

### Prerequisites
```bash
# Ensure you're in the project directory
cd /path/to/three-gen-subnet-trellis

# Make sure the orchestrator file exists
ls continuous_trellis_orchestrator_lora_working.py
```

### Run Basic Tests
```bash
# Test with mock components
python test_fallback_mechanism.py
```

### Run Comprehensive Tests
```bash
# Test the actual function with mocked dependencies
python test_fallback_mechanism_real.py
```

## 📊 What the Tests Verify

### 1. **Function Availability**
- ✅ Import success
- ✅ Function existence
- ✅ Required dependencies

### 2. **Component Functionality**
- ✅ Mock task creation
- ✅ Mock priority coordination
- ✅ Mock configuration
- ✅ Mock prompt optimization

### 3. **Core Function Execution**
- ✅ Function runs without errors
- ✅ Proper timing measurements
- ✅ Result validation
- ✅ Error handling

### 4. **Performance Metrics**
- ✅ Execution time measurement
- ✅ Performance thresholds
- ✅ Statistical analysis
- ✅ Benchmarking

## 🔧 Test Configuration

### Mock Settings
- **Generation Server**: `http://localhost:8096` (mocked)
- **Validation Server**: `http://localhost:10006` (mocked)
- **vLLM Server**: `http://localhost:9000` (mocked)
- **CLIP Model**: Mocked with simulated scores
- **Requests**: Mocked responses

### Test Prompts
- `"a simple wooden chair"`
- `"test prompt for validation"`
- `"complex mechanical device"`
- `"delicate glass ornament"`

## 📈 Performance Thresholds

| Category | Time | Status |
|----------|------|---------|
| Excellent | < 5s | ✅ |
| Good | < 10s | ✅ |
| Acceptable | < 20s | ⚠️ |
| Slow | > 20s | ❌ |

## 📝 Log Files

### Console Output
- Real-time test progress
- Immediate feedback
- Error details

### Log Files
- `test_fallback.log` - Basic test logs
- `test_fallback_real.log` - Comprehensive test logs

## 🐛 Troubleshooting

### Common Issues

#### 1. Import Errors
```bash
❌ Failed to import orchestrator
```
**Solution**: Ensure you're in the correct directory and the orchestrator file exists.

#### 2. Mock Dependencies
```bash
❌ Mock module not found
```
**Solution**: The test scripts handle this automatically by mocking external dependencies.

#### 3. Performance Issues
```bash
⚠️ Test took longer than expected
```
**Solution**: This is expected for the first run due to initialization. Subsequent runs should be faster.

### Debug Mode
To see more detailed output, modify the logging level in the test scripts:
```python
logging.basicConfig(level=logging.DEBUG)
```

## 🎯 Expected Results

### Successful Test Run
```
🚀 Starting Real Fallback Mechanism Test Suite
============================================================
📦 Import Status: ✅ SUCCESS
✅ All required functions are available

🔧 Testing individual functions...
🧪 Testing optimized_system_prompt function...
✅ optimized_system_prompt function: PASSED

🧪 Testing main fallback function...
🧪 Testing generate_3d_model_with_fallback function...
⏱️ Function execution time: 2.34 seconds
✅ Function execution: PASSED

⏱️ Testing performance benchmarks...
📊 Performance Statistics:
   Average execution time: 2.45s
   Minimum execution time: 2.12s
   Maximum execution time: 2.89s
   Total test time: 9.80s
✅ Performance: EXCELLENT (< 5s average)

============================================================
🎯 TEST SUMMARY
============================================================
📦 Import Status: ✅ PASSED
🔧 System Prompt: ✅ PASSED
🧪 Fallback Function: ✅ PASSED
⏱️ Performance: ✅ PASSED

🎉 ALL TESTS PASSED! The fallback mechanism is working correctly.
```

## 🔄 Running Tests Multiple Times

### Performance Consistency
```bash
# Run multiple times to check consistency
for i in {1..5}; do
    echo "=== Run $i ==="
    python test_fallback_mechanism_real.py
    echo "=== End Run $i ==="
    sleep 2
done
```

### Stress Testing
```bash
# Run with different prompts
python test_fallback_mechanism_real.py --stress-test
```

## 📋 Test Coverage

| Component | Tested | Status |
|-----------|--------|---------|
| Import System | ✅ | Full |
| Function Execution | ✅ | Full |
| Error Handling | ✅ | Full |
| Performance | ✅ | Full |
| Mock Dependencies | ✅ | Full |
| Logging | ✅ | Full |
| Configuration | ✅ | Full |

## 🎉 Success Criteria

A test run is considered successful when:
1. ✅ All imports succeed
2. ✅ All individual tests pass
3. ✅ Main function executes without errors
4. ✅ Performance meets thresholds
5. ✅ Logs are generated correctly
6. ✅ No critical errors occur

## 🚨 Failure Modes

### Critical Failures
- Import failures
- Function execution crashes
- Memory errors
- Infinite loops

### Warning Conditions
- Slow performance
- Missing optional dependencies
- Mock failures
- Logging issues

## 📞 Support

If tests fail consistently:
1. Check the log files for detailed error messages
2. Verify the orchestrator file exists and is accessible
3. Ensure Python environment has required packages
4. Check system resources (memory, disk space)
5. Review the error output for specific failure reasons

## 🔮 Future Enhancements

- Integration with CI/CD pipelines
- Automated performance regression testing
- Real server testing (optional)
- Extended prompt testing
- Memory usage monitoring
- Network latency simulation








