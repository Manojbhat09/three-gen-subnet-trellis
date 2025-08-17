#!/bin/bash

# SSH Key Setup Script for Distributed RL System Tunneling
# This script helps you generate and configure SSH keys for secure tunneling

echo "🔑 SSH Key Setup for Distributed RL System Tunneling"
echo "=================================================="

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Get remote server details
read -p "Enter remote server IP/hostname: " REMOTE_HOST
read -p "Enter remote username: " REMOTE_USER
read -p "Enter SSH port (default: 42497): " SSH_PORT

# Set default SSH port if not provided
if [ -z "$SSH_PORT" ]; then
    SSH_PORT="42497"
fi

if [ -z "$REMOTE_HOST" ] || [ -z "$REMOTE_USER" ]; then
    echo -e "${RED}❌ Remote host and username are required${NC}"
    exit 1
fi

# Get private key path
read -p "Enter private key path (e.g., ~/.ssh/id_vastai1): " PRIVATE_KEY_PATH

if [ -z "$PRIVATE_KEY_PATH" ]; then
    echo -e "${RED}❌ Private key path is required${NC}"
    exit 1
fi

# Expand tilde
PRIVATE_KEY_PATH=$(eval echo "$PRIVATE_KEY_PATH")

echo ""
echo -e "${BLUE}Remote Server:${NC} $REMOTE_USER@$REMOTE_HOST:$SSH_PORT"
echo -e "${BLUE}Private Key:${NC} $PRIVATE_KEY_PATH"
echo ""

# Function to check existing keys
check_existing_keys() {
    echo -e "${BLUE}🔍 Checking for existing SSH keys...${NC}"
    
    if [ -f "$PRIVATE_KEY_PATH" ]; then
        echo -e "${GREEN}✅ Existing SSH key found at: $PRIVATE_KEY_PATH${NC}"
        ls -la "$PRIVATE_KEY_PATH"*
        echo ""
        
        read -p "Do you want to use existing key? (Y/n): " use_existing
        if [[ "$use_existing" =~ ^([nN][oO]|[nN])$ ]]; then
            echo -e "${YELLOW}Backing up existing keys...${NC}"
            mkdir -p ~/.ssh/backup
            cp "$PRIVATE_KEY_PATH"* ~/.ssh/backup/ 2>/dev/null || true
            echo -e "${GREEN}✅ Keys backed up to ~/.ssh/backup/${NC}"
            return 1
        else
            return 0
        fi
    else
        echo -e "${YELLOW}No existing SSH key found at: $PRIVATE_KEY_PATH${NC}"
        return 1
    fi
}

# Function to generate new key
generate_new_key() {
    echo -e "${BLUE}🔑 Generating new SSH key pair...${NC}"
    echo ""
    
    # Choose key type
    echo "Select key type:"
    echo "1) ed25519 (recommended - faster, more secure)"
    echo "2) RSA 4096-bit (wider compatibility)"
    read -p "Choose option (1-2): " key_type
    
    case $key_type in
        1)
            key_type="ed25519"
            echo -e "${GREEN}Using ed25519 key type${NC}"
            ;;
        2)
            key_type="rsa"
            echo -e "${GREEN}Using RSA 4096-bit key type${NC}"
            ;;
        *)
            key_type="ed25519"
            echo -e "${YELLOW}Invalid choice, using ed25519 (default)${NC}"
            ;;
    esac
    
    # Generate key at specified path
    if [ "$key_type" = "ed25519" ]; then
        ssh-keygen -t ed25519 -f "$PRIVATE_KEY_PATH" -C "$(whoami)@$(hostname)"
    else
        ssh-keygen -t rsa -b 4096 -f "$PRIVATE_KEY_PATH" -C "$(whoami)@$(hostname)"
    fi
    
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}✅ SSH key generated successfully at: $PRIVATE_KEY_PATH${NC}"
        
        # Set correct permissions
        chmod 600 "$PRIVATE_KEY_PATH"
        chmod 644 "$PRIVATE_KEY_PATH.pub"
        
        return 0
    else
        echo -e "${RED}❌ Failed to generate SSH key${NC}"
        return 1
    fi
}

# Function to copy key to remote server
copy_key_to_remote() {
    echo -e "${BLUE}📤 Copying public key to remote server...${NC}"
    echo ""
    
    # Try ssh-copy-id first
    if command -v ssh-copy-id >/dev/null 2>&1; then
        echo -e "${YELLOW}Trying ssh-copy-id...${NC}"
        if ssh-copy-id -i "$PRIVATE_KEY_PATH.pub" -p "$SSH_PORT" $REMOTE_USER@$REMOTE_HOST; then
            echo -e "${GREEN}✅ Public key copied successfully using ssh-copy-id${NC}"
            return 0
        else
            echo -e "${YELLOW}ssh-copy-id failed, trying manual method...${NC}"
        fi
    fi
    
    # Manual copy method
    echo -e "${YELLOW}Manual copy method...${NC}"
    echo "You'll need to enter your password for the remote server."
    echo ""
    
    echo -e "${YELLOW}Copying $PRIVATE_KEY_PATH.pub to remote server...${NC}"
    
    # Create .ssh directory and copy key
    ssh -p "$SSH_PORT" $REMOTE_USER@$REMOTE_HOST "mkdir -p ~/.ssh && chmod 700 ~/.ssh"
    cat "$PRIVATE_KEY_PATH.pub" | ssh -p "$SSH_PORT" $REMOTE_USER@$REMOTE_HOST "cat >> ~/.ssh/authorized_keys && chmod 600 ~/.ssh/authorized_keys"
    
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}✅ Public key copied successfully${NC}"
        return 0
    else
        echo -e "${RED}❌ Failed to copy public key${NC}"
        return 1
    fi
}

# Function to test SSH connection
test_ssh_connection() {
    echo -e "${BLUE}🧪 Testing SSH connection...${NC}"
    echo ""
    
    echo -e "${YELLOW}Testing key-based authentication to $REMOTE_USER@$REMOTE_HOST:$SSH_PORT...${NC}"
    
    if ssh -i "$PRIVATE_KEY_PATH" -p "$SSH_PORT" -o ConnectTimeout=10 -o BatchMode=yes $REMOTE_USER@$REMOTE_HOST "echo 'SSH key authentication successful'" 2>/dev/null; then
        echo -e "${GREEN}✅ SSH key authentication working!${NC}"
        echo ""
        echo -e "${GREEN}🎉 SSH key setup complete!${NC}"
        echo ""
        echo -e "${BLUE}Next steps:${NC}"
        echo "1. Run tunnel setup: ./ssh_tunnel_setup.sh $REMOTE_HOST $REMOTE_USER $PRIVATE_KEY_PATH $SSH_PORT"
        echo "2. Or quick setup: ./quick_tunnel.sh $REMOTE_HOST $REMOTE_USER $PRIVATE_KEY_PATH $SSH_PORT"
        echo "3. Test connectivity: python test_tunnel_connectivity.py"
        return 0
    else
        echo -e "${RED}❌ SSH key authentication failed${NC}"
        echo ""
        echo -e "${YELLOW}Troubleshooting steps:${NC}"
        echo "1. Check if remote server allows key authentication"
        echo "2. Verify username and hostname are correct"
        echo "3. Check remote server SSH configuration"
        echo "4. Try manual connection: ssh -i $PRIVATE_KEY_PATH -p $SSH_PORT $REMOTE_USER@$REMOTE_HOST"
        return 1
    fi
}

# Function to show manual setup instructions
show_manual_instructions() {
    echo -e "${BLUE}📋 Manual SSH Key Setup Instructions${NC}"
    echo "=========================================="
    echo ""
    echo "If automatic setup fails, follow these steps manually:"
    echo ""
    echo "1. Generate SSH key:"
    echo "   ssh-keygen -t ed25519 -f $PRIVATE_KEY_PATH -C 'your_email@example.com'"
    echo ""
    echo "2. Copy public key to remote server:"
    echo "   ssh-copy-id -i $PRIVATE_KEY_PATH.pub -p $SSH_PORT $REMOTE_USER@$REMOTE_HOST"
    echo ""
    echo "3. Or copy manually:"
    echo "   cat $PRIVATE_KEY_PATH.pub | ssh -p $SSH_PORT $REMOTE_USER@$REMOTE_HOST 'mkdir -p ~/.ssh && cat >> ~/.ssh/authorized_keys'"
    echo ""
    echo "4. Test connection:"
    echo "   ssh -i $PRIVATE_KEY_PATH -p $SSH_PORT $REMOTE_USER@$REMOTE_HOST"
    echo ""
    echo "5. Set proper permissions:"
    echo "   chmod 600 $PRIVATE_KEY_PATH"
    echo "   chmod 644 $PRIVATE_KEY_PATH.pub"
}

# Main execution
main() {
    # Check if .ssh directory exists
    if [ ! -d ~/.ssh ]; then
        echo -e "${YELLOW}Creating .ssh directory...${NC}"
        mkdir -p ~/.ssh
        chmod 700 ~/.ssh
    fi
    
    # Check existing keys
    if ! check_existing_keys; then
        # Generate new key
        if ! generate_new_key; then
            echo -e "${RED}❌ Failed to generate SSH key${NC}"
            show_manual_instructions
            exit 1
        fi
    fi
    
    # Copy key to remote server
    if ! copy_key_to_remote; then
        echo -e "${RED}❌ Failed to copy key to remote server${NC}"
        show_manual_instructions
        exit 1
    fi
    
    # Test connection
    if ! test_ssh_connection; then
        echo -e "${RED}❌ SSH connection test failed${NC}"
        show_manual_instructions
        exit 1
    fi
}

# Run main function
main
