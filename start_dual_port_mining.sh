#!/bin/bash

# Dual-Port Continuous TRELLIS Mining Startup Script

echo "🚀 Starting Dual-Port Continuous TRELLIS Mining System"
echo "=================================================="

# Configuration
PORT1=${1:-8097}
PORT2=${2:-8098}
VALIDATION_PORT=${3:-10006}
OUTPUT_DIR="./dual_port_mining_outputs"

echo "Configuration:"
echo "  Port 1 (optimized): $PORT1"
echo "  Port 2 (original): $PORT2"
echo "  Validation port: $VALIDATION_PORT"
echo "  Output directory: $OUTPUT_DIR"
echo ""

# Check if ports are available
check_port() {
    local port=$1
    if lsof -Pi :$port -sTCP:LISTEN -t >/dev/null 2>&1; then
        echo "✅ Port $port is available"
        return 0
    else
        echo "❌ Port $port is not available"
        return 1
    fi
}

echo "🔍 Checking port availability..."
check_port $PORT1
check_port $PORT2
check_port $VALIDATION_PORT

echo ""

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Start the dual-port orchestrator
echo "🎯 Starting dual-port orchestrator..."
python continuous_trellis_orchestrator_lora_working_multi.py \
    --port1 $PORT1 \
    --port2 $PORT2 \
    --generation-server "http://localhost:$PORT1" \
    --validation-server "http://localhost:$VALIDATION_PORT" \
    --output-dir "$OUTPUT_DIR" \
    --enable-prompt-optimization \
    --enable-reproducibility-optimization \
    --enable-prompt-cleaning \
    --ollama-url "http://localhost:11434"

echo "🏁 Dual-port mining stopped"
