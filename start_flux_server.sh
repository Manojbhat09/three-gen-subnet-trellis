#!/bin/bash

# Start FLUX Socket Server (newcomer20_accurate)
# This script starts the FLUX inference server that communicates via Unix socket

echo "🚀 Starting FLUX Socket Server (newcomer20_accurate)..."

# Change to the newcomer20_accurate directory
cd "$(dirname "$0")/newcomer20_accurate"

# Check if the directory exists
if [ ! -d "src" ]; then
    echo "❌ Error: newcomer20_accurate/src directory not found!"
    echo "   Make sure you're running this from the correct location."
    exit 1
fi

# Check if uv is available
if ! command -v uv &> /dev/null; then
    echo "❌ Error: 'uv' command not found!"
    echo "   Please install uv: https://docs.astral.sh/uv/getting-started/installation/"
    exit 1
fi

# Remove existing socket if it exists
if [ -S "inferences.sock" ]; then
    echo "🧹 Removing existing socket file..."
    rm -f inferences.sock
fi

echo "📁 Working directory: $(pwd)"
echo "🔌 Socket will be created at: $(pwd)/inferences.sock"
echo ""

# Start the server
echo "🚀 Starting server with: uv run python src/main.py"
echo "   Press Ctrl+C to stop the server"
echo ""

# Start the server
uv run python src/main.py

echo ""
echo "🛑 FLUX server stopped."
