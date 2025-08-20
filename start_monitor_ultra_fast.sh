#!/bin/bash

echo "🚀 Starting Continuous Trellis Log Monitor (ULTRA-FAST)..."
echo "📁 Looking for log file: continuous_trellis.log"
echo "⚡ Using 0.01s interval for maximum responsiveness"
echo "💡 This will use more CPU but provide instant updates"
echo ""

# Check if log file exists
if [ ! -f "continuous_trellis.log" ]; then
    echo "⚠️  Warning: continuous_trellis.log not found in current directory"
    echo "   The monitor will wait for the file to be created..."
    echo ""
fi

# Start the monitor with ultra-fast interval
python3 log_monitor.py --log-file continuous_trellis.log --interval 0.01 --show-recent 3
