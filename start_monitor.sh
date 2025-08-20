#!/bin/bash

echo "🚀 Starting Continuous Trellis Log Monitor..."
echo "📁 Looking for log file: continuous_trellis.log"
echo "⚡ Using 0.1s interval for near real-time monitoring"
echo "💡 For ultra-fast monitoring, edit start_monitor.sh and change --interval 0.01"
echo ""

# Check if log file exists
if [ ! -f "continuous_trellis.log" ]; then
    echo "⚠️  Warning: continuous_trellis.log not found in current directory"
    echo "   The monitor will wait for the file to be created..."
    echo ""
fi

# Start the monitor
python3 log_monitor.py --log-file continuous_trellis.log --interval 0.1 --show-recent 3
