#!/bin/bash
# Installation script for TRELLIS Output Cleanup Service

set -e

echo "🧹 Installing TRELLIS Output Cleanup Service..."

# Check if running as root
if [[ $EUID -eq 0 ]]; then
   echo "❌ This script should not be run as root"
   exit 1
fi

# Get current user and directory
CURRENT_USER=$(whoami)
CURRENT_DIR=$(pwd)

echo "   User: $CURRENT_USER"
echo "   Directory: $CURRENT_DIR"

# Check if required files exist
if [[ ! -f "trellis_output_cleanup.py" ]]; then
    echo "❌ trellis_output_cleanup.py not found in current directory"
    exit 1
fi

if [[ ! -f "trellis-cleanup.service" ]]; then
    echo "❌ trellis-cleanup.service not found in current directory"
    exit 1
fi

# Make cleanup script executable
chmod +x trellis_output_cleanup.py

# Update service file with correct paths
sed -i "s|User=mbhat|User=$CURRENT_USER|g" trellis-cleanup.service
sed -i "s|Group=mbhat|Group=$CURRENT_USER|g" trellis-cleanup.service
sed -i "s|WorkingDirectory=/home/mbhat/three-gen-subnet-trellis|WorkingDirectory=$CURRENT_DIR|g" trellis-cleanup.service
sed -i "s|ExecStart=/usr/bin/python3 /home/mbhat/three-gen-subnet-trellis/trellis_output_cleanup.py|ExecStart=/usr/bin/python3 $CURRENT_DIR/trellis_output_cleanup.py|g" trellis-cleanup.service
sed -i "s|ReadWritePaths=/home/mbhat/three-gen-subnet-trellis|ReadWritePaths=$CURRENT_DIR|g" trellis-cleanup.service

# Copy service file to systemd directory
echo "📋 Copying service file to systemd..."
sudo cp trellis-cleanup.service /etc/systemd/system/

# Reload systemd daemon
echo "🔄 Reloading systemd daemon..."
sudo systemctl daemon-reload

# Enable service to start on boot
echo "✅ Enabling service to start on boot..."
sudo systemctl enable trellis-cleanup.service

echo ""
echo "🎉 Installation completed successfully!"
echo ""
echo "Service commands:"
echo "   Start service:     sudo systemctl start trellis-cleanup"
echo "   Stop service:      sudo systemctl stop trellis-cleanup"
echo "   Restart service:   sudo systemctl restart trellis-cleanup"
echo "   Check status:      sudo systemctl status trellis-cleanup"
echo "   View logs:         sudo journalctl -u trellis-cleanup -f"
echo ""
echo "The service will automatically:"
echo "   - Start when the system boots"
echo "   - Clean up trellis_submit_outputs/ every 15 minutes"
echo "   - Only clean when the server is not processing requests"
echo "   - Restart automatically if it crashes"
echo ""
echo "To start the service now, run: sudo systemctl start trellis-cleanup" 