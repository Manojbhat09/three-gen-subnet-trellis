#!/bin/bash

# SSH Tunnel Setup for Distributed RL System Dashboard
# This script helps you tunnel from remote server to local machine

echo "🚀 SSH Tunnel Setup for Distributed RL System Dashboard"
echo "======================================================"

# Configuration
REMOTE_HOST="${1:-localhost}"  # Remote server IP/hostname
REMOTE_USER="${2:-$USER}"      # Remote username
SSH_PORT="${4:-42497}"         # SSH port (default: 42497)
COORDINATOR_PORT=8090          # Coordinator API port
DASHBOARD_PORT=8100            # Dashboard frontend port
GPU_BASE_PORT=8096             # Base GPU agent port
NUM_GPUS=8                     # Number of GPU agents

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
    echo -e "${YELLOW}Do you want to generate a new key? (y/N):${NC}"
    read -r response
    if [[ "$response" =~ ^([yY][eE][sS]|[yY])$ ]]; then
        echo -e "${YELLOW}Generating new key at: $PRIVATE_KEY_PATH${NC}"
        ssh-keygen -t ed25519 -f "$PRIVATE_KEY_PATH" -C "$(whoami)@$(hostname)"
        if [ $? -eq 0 ]; then
            echo -e "${GREEN}✅ New key generated successfully${NC}"
        else
            echo -e "${RED}❌ Failed to generate key${NC}"
            exit 1
        fi
    else
        echo -e "${RED}❌ Please provide a valid private key path${NC}"
        exit 1
    fi
fi

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}Remote Server:${NC} $REMOTE_USER@$REMOTE_HOST"
echo -e "${BLUE}SSH Port:${NC} $SSH_PORT"
echo -e "${BLUE}Private Key:${NC} $PRIVATE_KEY_PATH"
echo -e "${BLUE}Coordinator Port:${NC} $COORDINATOR_PORT"
echo -e "${BLUE}Dashboard Port:${NC} $DASHBOARD_PORT"
echo -e "${BLUE}GPU Agents:${NC} $NUM_GPUS agents (ports $GPU_BASE_PORT-$((GPU_BASE_PORT + NUM_GPUS - 1)))"
echo ""

# Function to check SSH key setup
check_ssh_setup() {
    echo -e "${BLUE}🔐 Checking SSH Authentication Setup...${NC}"
    echo "=============================================="
    
    # Check if private key exists and has correct permissions
    if [ -f "$PRIVATE_KEY_PATH" ]; then
        echo -e "${GREEN}✅ Private key found: $PRIVATE_KEY_PATH${NC}"
        
        # Check permissions
        local perms=$(stat -c "%a" "$PRIVATE_KEY_PATH")
        if [ "$perms" != "600" ]; then
            echo -e "${YELLOW}⚠️  Fixing private key permissions...${NC}"
            chmod 600 "$PRIVATE_KEY_PATH"
            echo -e "${GREEN}✅ Permissions fixed${NC}"
        fi
    else
        echo -e "${RED}❌ Private key not found: $PRIVATE_KEY_PATH${NC}"
        return 1
    fi
    
    # Check if public key is on remote server
    echo -e "${YELLOW}Testing SSH connection to $REMOTE_USER@$REMOTE_HOST:$SSH_PORT...${NC}"
    if ssh -i "$PRIVATE_KEY_PATH" -p "$SSH_PORT" -o ConnectTimeout=10 -o BatchMode=yes $REMOTE_USER@$REMOTE_HOST "echo 'SSH key authentication successful'" 2>/dev/null; then
        echo -e "${GREEN}✅ SSH key authentication working${NC}"
        return 0
    else
        echo -e "${RED}❌ SSH key authentication failed${NC}"
        echo -e "${YELLOW}You need to copy your public key to the remote server:${NC}"
        echo "  ssh-copy-id -i $PRIVATE_KEY_PATH.pub -p $SSH_PORT $REMOTE_USER@$REMOTE_HOST"
        echo ""
        echo -e "${YELLOW}Or manually copy your public key:${NC}"
        echo "  cat $PRIVATE_KEY_PATH.pub | ssh -i $PRIVATE_KEY_PATH -p $SSH_PORT $REMOTE_USER@$REMOTE_HOST 'mkdir -p ~/.ssh && cat >> ~/.ssh/authorized_keys'"
        echo ""
        return 1
    fi
}

# Function to setup SSH keys
setup_ssh_keys() {
    echo -e "${BLUE}🔑 Setting up SSH Keys...${NC}"
    echo "=============================="
    
    # Check if key exists
    if [ -f "$PRIVATE_KEY_PATH" ]; then
        echo -e "${YELLOW}SSH key already exists at: $PRIVATE_KEY_PATH${NC}"
        echo -e "${YELLOW}Do you want to generate a new one? (y/N):${NC}"
        read -r response
        if [[ "$response" =~ ^([yY][eE][sS]|[yY])$ ]]; then
            echo -e "${YELLOW}Backing up existing keys...${NC}"
            mkdir -p ~/.ssh/backup
            cp "$PRIVATE_KEY_PATH"* ~/.ssh/backup/ 2>/dev/null || true
        else
            echo -e "${GREEN}Using existing SSH key${NC}"
            return 0
        fi
    fi
    
    # Generate new key
    echo -e "${YELLOW}Generating new SSH key pair at: $PRIVATE_KEY_PATH${NC}"
    
    ssh-keygen -t ed25519 -f "$PRIVATE_KEY_PATH" -C "$(whoami)@$(hostname)"
    
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}✅ SSH key generated successfully${NC}"
        
        # Set correct permissions
        chmod 600 "$PRIVATE_KEY_PATH"
        chmod 644 "$PRIVATE_KEY_PATH.pub"
        
        # Copy to remote server
        echo -e "${YELLOW}Copying public key to remote server...${NC}"
        if ssh-copy-id -i "$PRIVATE_KEY_PATH.pub" -p "$SSH_PORT" $REMOTE_USER@$REMOTE_HOST; then
            echo -e "${GREEN}✅ Public key copied to remote server${NC}"
            return 0
        else
            echo -e "${RED}❌ Failed to copy public key${NC}"
            echo -e "${YELLOW}You can copy it manually:${NC}"
            echo "  cat $PRIVATE_KEY_PATH.pub | ssh -i $PRIVATE_KEY_PATH -p $SSH_PORT $REMOTE_USER@$REMOTE_HOST 'mkdir -p ~/.ssh && cat >> ~/.ssh/authorized_keys'"
            return 1
        fi
    else
        echo -e "${RED}❌ Failed to generate SSH key${NC}"
        return 1
    fi
}

# Function to create tunnel
create_tunnel() {
    local local_port=$1
    local remote_port=$2
    local service_name=$3
    
    echo -e "${YELLOW}Creating tunnel for $service_name...${NC}"
    echo -e "Local: ${GREEN}localhost:$local_port${NC} -> Remote: ${GREEN}$REMOTE_HOST:$remote_port${NC}"
    
    # Kill existing tunnel if it exists
    pkill -f "ssh.*-L $local_port:$REMOTE_HOST:$remote_port" 2>/dev/null
    
    # Create new tunnel with private key and custom port
    ssh -i "$PRIVATE_KEY_PATH" -p "$SSH_PORT" -f -N -L $local_port:$REMOTE_HOST:$remote_port $REMOTE_USER@$REMOTE_HOST
    
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}✅ Tunnel created successfully!${NC}"
        echo -e "   Access $service_name at: ${GREEN}http://localhost:$local_port${NC}"
    else
        echo -e "${RED}❌ Failed to create tunnel for $service_name${NC}"
    fi
    echo ""
}

# Function to create all tunnels
create_all_tunnels() {
    echo -e "${BLUE}Creating all tunnels...${NC}"
    echo ""
    
    # Coordinator API tunnel
    create_tunnel 18090 $COORDINATOR_PORT "Coordinator API"
    
    # Dashboard tunnel
    create_tunnel 18100 $DASHBOARD_PORT "Dashboard Frontend"
    
    # GPU agent tunnels (optional - for direct GPU monitoring)
    echo -e "${YELLOW}Creating GPU agent tunnels (optional)...${NC}"
    for i in $(seq 0 $((NUM_GPUS-1))); do
        local gpu_port=$((GPU_BASE_PORT + i))
        local local_gpu_port=$((18100 + i + 1))
        create_tunnel $local_gpu_port $gpu_port "GPU Agent $i"
    done
    
    echo -e "${GREEN}🎉 All tunnels created!${NC}"
    echo ""
    echo -e "${BLUE}Access URLs:${NC}"
    echo -e "  Coordinator API: ${GREEN}http://localhost:18090${NC}"
    echo -e "  Dashboard:       ${GREEN}http://localhost:18100${NC}"
    echo -e "  GPU Agents:      ${GREEN}http://localhost:18101-$((18100 + NUM_GPUS))${NC}"
}

# Function to list active tunnels
list_tunnels() {
    echo -e "${BLUE}Active SSH Tunnels:${NC}"
    echo "======================"
    
    local tunnels=$(netstat -tlnp 2>/dev/null | grep "127.0.0.1:18" | grep LISTEN)
    
    if [ -z "$tunnels" ]; then
        echo -e "${YELLOW}No active tunnels found.${NC}"
    else
        echo "$tunnels" | while read line; do
            local port=$(echo "$line" | awk '{print $4}' | cut -d: -f2)
            local service=""
            
            case $port in
                18090) service="Coordinator API" ;;
                18100) service="Dashboard Frontend" ;;
                1810[1-9]) service="GPU Agent $((port - 18101))" ;;
                *) service="Unknown" ;;
            esac
            
            echo -e "${GREEN}Port $port${NC} -> $service"
        done
    fi
}

# Function to kill all tunnels
kill_all_tunnels() {
    echo -e "${YELLOW}Killing all SSH tunnels...${NC}"
    
    # Kill tunnels by port pattern
    pkill -f "ssh.*-L 18[0-9][0-9][0-9]:$REMOTE_HOST" 2>/dev/null
    
    echo -e "${GREEN}✅ All tunnels killed!${NC}"
}

# Function to test connectivity
test_connectivity() {
    echo -e "${BLUE}Testing tunnel connectivity...${NC}"
    echo "========================"
    
    local services=(
        "18090:Coordinator API"
        "18100:Dashboard Frontend"
    )
    
    for service in "${services[@]}"; do
        local port=$(echo "$service" | cut -d: -f1)
        local name=$(echo "$service" | cut -d: -f2)
        
        echo -n "Testing $name (port $port)... "
        
        if curl -s "http://localhost:$port" >/dev/null 2>&1; then
            echo -e "${GREEN}✅ OK${NC}"
        else
            echo -e "${RED}❌ FAILED${NC}"
        fi
    done
}

# Main menu
show_menu() {
    echo ""
    echo -e "${BLUE}SSH Tunnel Management Menu:${NC}"
    echo "================================"
    echo "1) Setup SSH keys (if needed)"
    echo "2) Create all tunnels"
    echo "3) Create coordinator tunnel only"
    echo "4) Create dashboard tunnel only"
    echo "5) List active tunnels"
    echo "6) Test tunnel connectivity"
    echo "7) Kill all tunnels"
    echo "8) Exit"
    echo ""
    read -p "Choose an option (1-8): " choice
    
    case $choice in
        1) setup_ssh_keys ;;
        2) create_all_tunnels ;;
        3) create_tunnel 18090 $COORDINATOR_PORT "Coordinator API" ;;
        4) create_tunnel 18100 $DASHBOARD_PORT "Dashboard Frontend" ;;
        5) list_tunnels ;;
        6) test_connectivity ;;
        7) kill_all_tunnels ;;
        8) echo "Goodbye!"; exit 0 ;;
        *) echo -e "${RED}Invalid option. Please try again.${NC}"; show_menu ;;
    esac
}

# Check if running interactively
if [ -t 0 ]; then
    # Interactive mode - check SSH setup first
    if check_ssh_setup; then
        echo -e "${GREEN}✅ SSH authentication ready!${NC}"
        show_menu
    else
        echo -e "${YELLOW}⚠️  SSH setup incomplete. Choose option 1 to setup SSH keys.${NC}"
        show_menu
    fi
else
    # Non-interactive mode - check SSH setup first
    if check_ssh_setup; then
        create_all_tunnels
    else
        echo -e "${RED}❌ SSH setup incomplete. Run interactively to setup SSH keys.${NC}"
        exit 1
    fi
fi
