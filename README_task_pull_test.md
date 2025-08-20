# Task Pull Test Scripts

These scripts test if you can successfully pull tasks from validators using your Bittensor wallet and hotkey.

## Files

- `test_task_pull.py` - Simple test script
- `test_task_pull_enhanced.py` - Enhanced version with config file support
- `test_config.json` - Configuration file
- `README_task_pull_test.md` - This file

## Prerequisites

1. **Bittensor installed**: Make sure you have `bittensor` installed
2. **Wallet setup**: You need a Bittensor wallet with a hotkey
3. **Neurons module**: The scripts need access to the `neurons.common.protocol` module
4. **Network access**: You need internet access to connect to the Bittensor network

## Quick Start

### 1. Update Configuration

Edit `test_config.json` to use your wallet and hotkey names:

```json
{
    "wallet_name": "YOUR_WALLET_NAME",
    "hotkey_name": "YOUR_HOTKEY_NAME",
    "netuid": 17,
    "min_stake": 1000.0,
    "min_trust": 0.0,
    "max_validators_to_test": 3,
    "query_timeout": 60,
    "delay_between_pulls": 2
}
```

### 2. Run the Test

```bash
# Run the enhanced version (recommended)
python test_task_pull_enhanced.py

# Or run the simple version
python test_task_pull.py
```

## What the Scripts Do

1. **Setup Bittensor**: Load your wallet, connect to subtensor, initialize dendrite
2. **Find Validators**: Scan for eligible validators based on stake and trust criteria
3. **Test Task Pulling**: Attempt to pull tasks from each validator
4. **Report Results**: Show success/failure rates and save results to JSON

## Expected Output

If successful, you should see:

```
🧪 ENHANCED TASK PULL TEST
==================================================
🔧 Initializing Enhanced Task Puller
   Wallet: YOUR_WALLET_NAME
   Hotkey: YOUR_HOTKEY_NAME
   NetUID: 17
   Min stake: 1000.0 TAO
   Min trust: 0.0
   Max validators to test: 3

🔧 Setting up Bittensor components...
   Loading wallet...
   ✅ Wallet loaded: 5F... (your hotkey address)
   Connecting to subtensor...
   ✅ Subtensor connected
   Initializing dendrite...
   ✅ Dendrite initialized
   Loading metagraph...
   ✅ Metagraph loaded (netuid: 17)

🔍 Scanning for eligible validators...
   Min stake: 1000.0 TAO
   Min trust: 0.0
   Max validators: 3
✅ Found 3 eligible validators
    1. UID  45:  5000.0 TAO, trust: 0.850
    2. UID  67:  3000.0 TAO, trust: 0.720
    3. UID  89:  2000.0 TAO, trust: 0.680

🚀 Starting task pulling test...
   Testing up to 3 validators

--- Testing Validator 1/3 ---
📡 Attempting to pull task from UID 45...
   Sending request to validator...
✅ SUCCESS! Got task from UID 45
   Task ID: task_12345
   Prompt: 'A red apple on a white background'
   Query time: 1.23s
✅ Task pull SUCCESSFUL from UID 45
   Waiting 2 seconds before next validator...

--- Testing Validator 2/3 ---
📡 Attempting to pull task from UID 67...
   Sending request to validator...
✅ SUCCESS! Got task from UID 67
   Task ID: task_67890
   Prompt: 'A blue crystal gemstone'
   Query time: 0.98s
✅ Task pull SUCCESSFUL from UID 67
   Waiting 2 seconds before next validator...

--- Testing Validator 3/3 ---
📡 Attempting to pull task from UID 89...
   Sending request to validator...
⚠️ No task received from UID 89
❌ Task pull FAILED from UID 89

📊 TEST SUMMARY
==================================================
Total validators tested: 3
Successful task pulls: 2
Failed task pulls: 1
Success rate: 66.7%
✅ Task pulling is WORKING! You can successfully pull tasks.

📝 Successfully pulled tasks:
   1. UID 45: 'A red apple on a white background'
   2. UID 67: 'A blue crystal gemstone'

🏁 Test completed!
💾 Results saved to task_pull_results_1234567890.json
```

## Troubleshooting

### Common Issues

1. **Import Error**: `ModuleNotFoundError: No module named 'neurons.common.protocol'`
   - Make sure you're running from the correct directory
   - The `neurons` module should be in your Python path

2. **Wallet Error**: `Wallet not found`
   - Check your wallet name and hotkey name in the config
   - Make sure the wallet exists and is properly set up

3. **Network Error**: `Connection failed`
   - Check your internet connection
   - Try using a different network (testnet vs mainnet)

4. **No Validators Found**: 
   - Lower the `min_stake` value in config
   - Check if you're using the correct `netuid`

### Configuration Options

- `wallet_name`: Your Bittensor wallet name
- `hotkey_name`: Your Bittensor hotkey name  
- `netuid`: Subnet ID (17 for 404-GEN)
- `min_stake`: Minimum stake required for validators (in TAO)
- `min_trust`: Minimum trust score required
- `max_validators_to_test`: Maximum number of validators to test
- `query_timeout`: Timeout for each validator query (seconds)
- `delay_between_pulls`: Delay between testing validators (seconds)

## Success Criteria

The test is considered successful if:
- ✅ Bittensor components setup successfully
- ✅ At least one validator is found
- ✅ At least one task pull succeeds
- ✅ Success rate > 0%

## Next Steps

If the test is successful, you can:
1. Use the full orchestrator: `python continuous_trellis_orchestrator_lora_working.py`
2. Customize the configuration for your needs
3. Monitor the results and adjust parameters

If the test fails, check:
1. Wallet and hotkey configuration
2. Network connectivity
3. Bittensor installation
4. Module imports
