#!/bin/bash

# Validator Prompt Simulation Startup Script
# This script starts the get-prompts service and then runs the simulation

set -e

echo "🚀 Starting Validator Prompt Simulation..."

# Check if required directories exist
if [ ! -d "get-prompts" ]; then
    echo "❌ Error: get-prompts directory not found"
    exit 1
fi

if [ ! -d "text-prompt-generator" ]; then
    echo "❌ Error: text-prompt-generator directory not found"
    exit 1
fi

# Function to cleanup background processes
cleanup() {
    echo "🧹 Cleaning up background processes..."
    if [ ! -z "$GET_PROMPTS_PID" ]; then
        kill $GET_PROMPTS_PID 2>/dev/null || true
    fi
    if [ ! -z "$SIMULATION_PID" ]; then
        kill $SIMULATION_PID 2>/dev/null || true
    fi
    exit 0
}

# Set trap for cleanup
trap cleanup SIGINT SIGTERM

# Check if Python dependencies are installed
echo "🔍 Checking Python dependencies..."
python3 -c "import aiohttp, pybase64" 2>/dev/null || {
    echo "📦 Installing required Python packages..."
    pip3 install aiohttp pybase64
}

# Create default prompts file if it doesn't exist
DEFAULT_PROMPTS_FILE="get-prompts/resources/default_prompts.txt"
if [ ! -f "$DEFAULT_PROMPTS_FILE" ]; then
    echo "📝 Creating default prompts file..."
    mkdir -p "get-prompts/resources"
    cat > "$DEFAULT_PROMPTS_FILE" << 'EOF'
mechanical robot with steel plating
majestic lion in natural pose
ethereal elf with magical aura
detailed dragon with intricate design
chrome android with glowing eyes
peaceful deer in forest setting
mystical wizard with enchanted staff
complex spaceship with elaborate details
steel warrior with battle armor
magical unicorn with rainbow mane
EOF
    echo "✅ Created default prompts file with $DEFAULT_PROMPTS_FILE"
fi

# Start the get-prompts service
echo "🌐 Starting get-prompts service..."
cd get-prompts

# Check if the service is already running
if lsof -Pi :8000 -sTCP:LISTEN -t >/dev/null ; then
    echo "⚠️  get-prompts service already running on port 8000"
    GET_PROMPTS_PID=""
else
    # Start the service in background
    python3 serve.py --port 8000 --api-key "test_api_key_123" &
    GET_PROMPTS_PID=$!
    echo "✅ get-prompts service started with PID: $GET_PROMPTS_PID"
    
    # Wait for service to be ready
    echo "⏳ Waiting for get-prompts service to be ready..."
    for i in {1..30}; do
        if curl -s http://localhost:8000/version >/dev/null 2>&1; then
            echo "✅ get-prompts service is ready!"
            break
        fi
        if [ $i -eq 30 ]; then
            echo "❌ Timeout waiting for get-prompts service"
            cleanup
        fi
        sleep 1
    done
fi

cd ..

# Wait a moment for service to fully initialize
sleep 2

# Start the simulation
echo "🎮 Starting validator simulation..."
python3 validator_prompt_simulation.py \
    --simulate-validators 3 \
    --duration 120 \
    --prompts-per-batch 25 &
SIMULATION_PID=$!

echo "✅ Simulation started with PID: $SIMULATION_PID"
echo ""
echo "📋 Simulation Configuration:"
echo "   - Validators: 3"
echo "   - Duration: 120 seconds (2 minutes)"
echo "   - Prompts per batch: 25"
echo "   - Get-prompts service: http://localhost:8000"
echo ""
echo "🔄 Simulation is running... Press Ctrl+C to stop"
echo "📊 Check the output above for real-time statistics"
echo "💾 Results will be saved to simulation_results.json"

# Wait for simulation to complete
wait $SIMULATION_PID

echo ""
echo "🎯 Simulation completed!"
echo "📊 Results saved to: simulation_results.json"

# Cleanup
cleanup
