#!/bin/bash

# Quick SSH Tunnel Setup - Non-interactive
# Usage: ./quick_tunnel.sh [REMOTE_IP] [USERNAME] [PRIVATE_KEY_PATH] [SSH_PORT]
# bash quick.sh 64.247.196.29 root "/c/Users/manoj/.ssh/id_vastai1" 42497

REMOTE_HOST="${1:-localhost}"
REMOTE_USER="${2:-$USER}"
SSH_PORT="${4:-42497}"  # SSH port (default: 42497)

echo "🚀 Quick SSH Tunnel Setup"
echo "Remote: $REMOTE_USER@$REMOTE_HOST:$SSH_PORT"

# Get private key path
if [ -z "$3" ]; then
    echo -e "${YELLOW}Enter the path to your private key (e.g., ~/.ssh/id_vastai1):${NC}"
    read -p "Private key path: " PRIVATE_KEY_PATH
else
    PRIVATE_KEY_PATH="$3"
fi

# Validate private key path
if [ -z "$PRIVATE_KEY_PATH" ]; then
    echo -e "${RED}❌ Private key path is required${NC}"
    exit 1
fi

# Expand tilde and check if file exists
PRIVATE_KEY_PATH=$(eval echo "$PRIVATE_KEY_PATH")
if [ ! -f "$PRIVATE_KEY_PATH" ]; then
    echo -e "${RED}❌ Private key file not found: $PRIVATE_KEY_PATH${NC}"
    exit 1
fi

echo -e "${GREEN}Using private key: $PRIVATE_KEY_PATH${NC}"
echo -e "${GREEN}Using SSH port: $SSH_PORT${NC}"

# Check SSH authentication first
echo "🔐 Checking SSH authentication..."
if ! ssh -i "$PRIVATE_KEY_PATH" -p "$SSH_PORT" -o ConnectTimeout=10 -o BatchMode=yes $REMOTE_USER@$REMOTE_HOST "echo 'SSH key authentication successful'" 2>/dev/null; then
    echo "❌ SSH key authentication failed!"
    echo ""
    echo "You need to setup SSH keys first:"
    echo "1. Generate SSH key: ssh-keygen -t ed25519 -f $PRIVATE_KEY_PATH"
    echo "2. Copy to remote: ssh-copy-id -i $PRIVATE_KEY_PATH.pub -p $SSH_PORT $REMOTE_USER@$REMOTE_HOST"
    echo ""
    echo "Or run the interactive setup: ./ssh_tunnel_setup.sh $REMOTE_HOST $REMOTE_USER $PRIVATE_KEY_PATH $SSH_PORT"
    exit 1
fi

echo "✅ SSH authentication working"

# Kill existing tunnels
pkill -f "ssh.*-L 18[0-9][0-9][0-9]:$REMOTE_HOST" 2>/dev/null

# Create tunnels
echo "Creating tunnels..."

# Coordinator API
ssh -i "$PRIVATE_KEY_PATH" -p "$SSH_PORT" -f -N -L 18090:$REMOTE_HOST:8090 $REMOTE_USER@$REMOTE_HOST && echo "✅ Coordinator tunnel (18090)" || echo "❌ Coordinator tunnel failed"

# Dashboard
ssh -i "$PRIVATE_KEY_PATH" -p "$SSH_PORT" -f -N -L 18100:$REMOTE_HOST:8100 $REMOTE_USER@$REMOTE_HOST && echo "✅ Dashboard tunnel (18100)" || echo "❌ Dashboard tunnel failed"

# GPU Agents
for i in {0..7}; do
    local_port=$((18101 + i))
    remote_port=$((8096 + i))
    ssh -i "$PRIVATE_KEY_PATH" -p "$SSH_PORT" -f -N -L $local_port:$REMOTE_HOST:$remote_port $REMOTE_USER@$REMOTE_HOST && echo "✅ GPU $i tunnel ($local_port)" || echo "❌ GPU $i tunnel failed"
done

echo ""
echo "🌐 Access URLs:"
echo "  Coordinator: http://localhost:18090"
echo "  Dashboard:   http://localhost:18100"
echo "  GPU Agents:  http://localhost:18101-18108"
echo ""
echo "To test: python test_tunnel_connectivity.py"
echo "To kill: pkill -f 'ssh.*-L 18[0-9][0-9][0-9]:'"
