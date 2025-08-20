#!/bin/bash

# Miner-Integrated Simulation Startup Script
# This script runs the simulation that integrates with your actual ContinuousTrellisOrchestrator

set -e

echo "🚀 Starting Miner-Integrated Validator Simulation..."

# Check if required files exist
if [ ! -f "miner_integrated_simulation.py" ]; then
    echo "❌ Error: miner_integrated_simulation.py not found"
    exit 1
fi

if [ ! -f "continuous_trellis_orchestrator_lora_working.py" ]; then
    echo "❌ Error: continuous_trellis_orchestrator_lora_working.py not found"
    echo "   This is required for the miner-integrated simulation"
    exit 1
fi

# Function to cleanup background processes
cleanup() {
    echo "🧹 Cleaning up background processes..."
    if [ ! -z "$SIMULATION_PID" ]; then
        kill $SIMULATION_PID 2>/dev/null || true
    fi
    exit 0
}

# Set trap for cleanup
trap cleanup SIGINT SIGTERM

# Check if Python dependencies are installed
echo "🔍 Checking Python dependencies..."
python3 -c "import asyncio, json, time, random, logging, argparse, traceback, pathlib, typing, dataclasses, datetime, statistics, collections" 2>/dev/null || {
    echo "📦 Installing required Python packages..."
    pip3 install asyncio json time random logging argparse traceback pathlib typing dataclasses datetime statistics collections
}

# Create simulation outputs directory
mkdir -p simulation_outputs
echo "📁 Created simulation_outputs directory"

# Check if we can import the miner components
echo "🔍 Testing miner component imports..."
python3 -c "
try:
    from continuous_trellis_orchestrator_lora_working import ContinuousTrellisOrchestrator, TaskRecord, ValidatorState
    print('✅ Miner components import successfully')
except ImportError as e:
    print(f'⚠️ Warning: Some miner components not available: {e}')
    print('   Simulation will run in mock mode')
"

# Start the simulation
echo "🎮 Starting miner-integrated simulation..."
echo ""
echo "📋 Simulation Configuration:"
echo "   - Scenario: balanced (default)"
echo "   - Validators: 5"
echo "   - Duration: 10 minutes"
echo "   - Real Miner Integration: Enabled"
echo "   - Subnet Submission: Disabled (simulation mode)"
echo ""

# Run the simulation
python3 miner_integrated_simulation.py \
    --scenario balanced \
    --validators 5 \
    --duration 600 &
SIMULATION_PID=$!

echo "✅ Simulation started with PID: $SIMULATION_PID"
echo ""
echo "🔄 Simulation is running... Press Ctrl+C to stop"
echo "📊 Check the output above for real-time statistics"
echo "💾 Results will be saved to: miner_integrated_simulation_results.json"
echo ""
echo "🎯 Available Scenarios:"
echo "   - balanced: General testing and baseline performance"
echo "   - high_load: Test system under moderate stress"
echo "   - stress_test: Maximum stress testing"
echo "   - learning: Observe learning and adaptation patterns"
echo "   - realistic: Based on actual subnet behavior"
echo ""
echo "🔧 Customization Options:"
echo "   --scenario SCENARIO: Choose simulation scenario"
echo "   --validators N: Number of validators to simulate"
echo "   --duration SECONDS: Simulation duration"
echo "   --miner-config FILE: Custom miner configuration"
echo "   --enable-submission: Enable real subnet submission (use with caution!)"

# Wait for simulation to complete
wait $SIMULATION_PID

echo ""
echo "🎯 Simulation completed!"
echo "📊 Results saved to: miner_integrated_simulation_results.json"

# Cleanup
cleanup
