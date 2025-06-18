#!/bin/bash

# GPU Coordination Server Startup Script
# This script starts both validation and generation servers with proper GPU coordination

set -e

echo "🚀 Starting GPU-Coordinated 3D Generation Servers"
echo "=================================================="

# Function to cleanup on exit
cleanup() {
    echo "🛑 Shutting down servers..."
    if [ ! -z "$VALIDATION_PID" ]; then
        kill $VALIDATION_PID 2>/dev/null || true
    fi
    if [ ! -z "$GENERATION_PID" ]; then
        kill $GENERATION_PID 2>/dev/null || true
    fi
    echo "✅ Cleanup complete"
}

# Set trap for cleanup
trap cleanup EXIT INT TERM

# Check if conda is available
if ! command -v conda &> /dev/null; then
    echo "❌ Conda not found. Please install conda first."
    exit 1
fi

# Source conda
source /home/mbhat/miniconda/etc/profile.d/conda.sh

echo "🔧 Starting Validation Server..."
cd validation
conda activate three-gen-validation
python serve.py --host 0.0.0.0 --port 10006 &
VALIDATION_PID=$!
cd ..

echo "⏳ Waiting for validation server to start..."
sleep 10

# Check if validation server is running
if ! curl -s http://localhost:10006/health/ > /dev/null; then
    echo "❌ Validation server failed to start"
    exit 1
fi
echo "✅ Validation server started (PID: $VALIDATION_PID)"

echo "🔧 Starting Generation Server..."
conda activate hunyuan3d
python flux_hunyuan_sugar_generation_server.py &
GENERATION_PID=$!

echo "⏳ Waiting for generation server to start..."
sleep 15

# Check if generation server is running
if ! curl -s http://localhost:8095/health/ > /dev/null; then
    echo "❌ Generation server failed to start"
    exit 1
fi
echo "✅ Generation server started (PID: $GENERATION_PID)"

echo ""
echo "🎉 Both servers are running with GPU coordination!"
echo "=================================================="
echo "📊 Server Status:"
echo "   Validation Server: http://localhost:10006"
echo "   Generation Server: http://localhost:8095"
echo ""
echo "🧪 To test the system, run:"
echo "   python automated_gpu_coordination_test.py"
echo ""
echo "🛑 Press Ctrl+C to stop both servers"
echo ""

# Wait for user interrupt
wait 