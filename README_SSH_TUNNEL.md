# SSH Tunnel Setup for Distributed RL System Dashboard

This setup allows you to access your distributed RL system dashboard from your local machine through SSH tunneling.

## 🔐 **SSH Key Authentication Required**

**IMPORTANT**: You need SSH key authentication set up between your local machine and the remote server. Password authentication won't work for tunneling.

## 🚀 **Complete Setup Process**

### **Step 1: Setup SSH Keys (One-time setup)**
```bash
# Option A: Use the automated script (recommended)
./setup_ssh_keys.sh

# Option B: Manual setup
ssh-keygen -t ed25519 -f ~/.ssh/id_vastai1 -C "your_email@example.com"
ssh-copy-id -i ~/.ssh/id_vastai1.pub -p 42497 username@remote_server_ip
```

### **Step 2: Create SSH Tunnels**
```bash
# Option A: Interactive setup (recommended for first time)
./ssh_tunnel_setup.sh YOUR_REMOTE_IP YOUR_USERNAME ~/.ssh/id_vastai1 42497

# Option B: Quick setup
./quick_tunnel.sh YOUR_REMOTE_IP YOUR_USERNAME ~/.ssh/id_vastai1 42497
```

### **Step 3: Test & Access**
```bash
# Test connectivity
python test_tunnel_connectivity.py

# Access dashboard in your browser
open http://localhost:18100
```

## 🔑 **Custom Private Key Paths & SSH Ports**

The scripts now support **custom private key paths** like `~/.ssh/id_vastai1` and **custom SSH ports** like `42497`:

### **Usage Examples:**
```bash
# Using custom key path and SSH port
./ssh_tunnel_setup.sh 192.168.1.100 myuser ~/.ssh/id_vastai1 42497

# Using custom key path and SSH port with quick setup
./quick_tunnel.sh 192.168.1.100 myuser ~/.ssh/id_vastai1 42497

# Interactive input (script will prompt for all details)
./ssh_tunnel_setup.sh 192.168.1.100 myuser
# Script will prompt: 
# - "Enter the path to your private key (e.g., ~/.ssh/id_vastai1):"
# - SSH port defaults to 42497
```

### **Parameter Order:**
```bash
./script.sh [REMOTE_IP] [USERNAME] [PRIVATE_KEY_PATH] [SSH_PORT]

# Examples:
./ssh_tunnel_setup.sh 192.168.1.100 myuser ~/.ssh/id_vastai1 42497
./quick_tunnel.sh 192.168.1.100 myuser ~/.ssh/id_vastai1 42497
```

### **Supported Key Paths:**
- **Default keys**: `~/.ssh/id_ed25519`, `~/.ssh/id_rsa`
- **Custom keys**: `~/.ssh/id_vastai1`, `~/.ssh/my_custom_key`, etc.
- **Absolute paths**: `/home/user/.ssh/my_key`

### **SSH Port Support:**
- **Default port**: 42497 (configurable)
- **Standard SSH**: 22 (if needed)
- **Custom ports**: Any port your server uses

## 📋 **What Gets Tunneled**

| Service | Remote Port | Local Port | Purpose |
|---------|-------------|------------|---------|
| **Coordinator API** | 8090 | 18090 | System management & job submission |
| **Dashboard Frontend** | 8100 | 18100 | Web dashboard interface |
| **GPU Agent 0** | 8096 | 18101 | Direct GPU monitoring |
| **GPU Agent 1** | 8097 | 18102 | Direct GPU monitoring |
| **GPU Agent 2** | 8098 | 18103 | Direct GPU monitoring |
| **GPU Agent 3** | 8099 | 18104 | Direct GPU monitoring |
| **GPU Agent 4** | 8100 | 18105 | Direct GPU monitoring |
| **GPU Agent 5** | 8101 | 18106 | Direct GPU monitoring |
| **GPU Agent 6** | 8102 | 18107 | Direct GPU monitoring |
| **GPU Agent 7** | 8103 | 18108 | Direct GPU monitoring |

## 🔧 **Manual SSH Commands**

If you prefer to set up tunnels manually:

### Create all tunnels at once:
```bash
# Coordinator API
ssh -i ~/.ssh/id_vastai1 -p 42497 -f -N -L 18090:localhost:8090 username@remote_server_ip

# Dashboard
ssh -i ~/.ssh/id_vastai1 -p 42497 -f -N -L 18100:localhost:8100 username@remote_server_ip

# GPU Agents (optional)
for i in {0..7}; do
    local_port=$((18101 + i))
    remote_port=$((8096 + i))
    ssh -i ~/.ssh/id_vastai1 -p 42497 -f -N -L ${local_port}:localhost:${remote_port} username@remote_server_ip
done
```

### Kill all tunnels:
```bash
pkill -f "ssh.*-L 18[0-9][0-9][0-9]:localhost"
```

## 🧪 **Testing the Setup**

### 1. Install Python dependencies
```bash
pip install requests
```

### 2. Run the connectivity test
```bash
python test_tunnel_connectivity.py
```

This will test all tunneled services and provide a detailed report.

## 🌐 **Access URLs**

Once tunnels are active, access your services at:

- **Coordinator API**: http://localhost:18090
- **Dashboard**: http://localhost:18100
- **GPU Agent 0**: http://localhost:18101
- **GPU Agent 1**: http://localhost:18102
- **GPU Agent 2**: http://localhost:18103
- **GPU Agent 3**: http://localhost:18104
- **GPU Agent 4**: http://localhost:18105
- **GPU Agent 5**: http://localhost:18106
- **GPU Agent 6**: http://localhost:18107
- **GPU Agent 7**: http://localhost:18108

## 📱 **Dashboard Features**

Through the tunneled dashboard, you can:

- **Monitor System Status**: View all GPU agents and their health
- **Submit Jobs**: Send RL optimization tasks to the system
- **Track Progress**: Monitor job execution and results
- **View Insights**: See cross-GPU learning patterns
- **GPU Monitoring**: Check individual GPU agent status

## 🔍 **Troubleshooting**

### **SSH Key Issues:**

1. **"Permission denied (publickey)"**
   ```bash
   # Check if key exists and has correct permissions
   ls -la ~/.ssh/id_vastai1
   
   # Fix permissions (should be 600)
   chmod 600 ~/.ssh/id_vastai1
   chmod 700 ~/.ssh/
   
   # Verify public key is on remote server
   ssh -i ~/.ssh/id_vastai1 -p 42497 username@remote_server_ip 'cat ~/.ssh/authorized_keys'
   ```

2. **"No such file or directory"**
   ```bash
   # Create .ssh directory if it doesn't exist
   mkdir -p ~/.ssh
   chmod 700 ~/.ssh
   
   # Generate new key at custom path
   ssh-keygen -t ed25519 -f ~/.ssh/id_vastai1 -C "your_email@example.com"
   ```

3. **"Agent admitted failure to sign"**
   ```bash
   # Start SSH agent
   eval $(ssh-agent)
   ssh-add ~/.ssh/id_vastai1
   ```

### **SSH Port Issues:**

1. **"Connection refused" on SSH port**
   ```bash
   # Verify SSH port is correct
   telnet remote_server_ip 42497
   
   # Check if server is listening on that port
   nmap -p 42497 remote_server_ip
   ```

2. **"Connection timed out"**
   ```bash
   # Check firewall settings
   # Verify SSH service is running on custom port
   # Test with standard SSH port 22 first
   ```

### **Tunnel Issues:**

1. **"Connection refused" errors**
   - Verify SSH tunnels are active: `./ssh_tunnel_setup.sh` → Option 5
   - Check if remote services are running
   - Ensure firewall allows connections

2. **"Port already in use"**
   - Kill existing tunnels: `./ssh_tunnel_setup.sh` → Option 7
   - Check for other services using the same ports

3. **"Slow response times"**
   - This is normal for SSH tunneling
   - Consider using a more direct connection for production

### **Debug Commands:**
```bash
# Check active tunnels
netstat -tlnp | grep "127.0.0.1:18"

# Check SSH processes
ps aux | grep "ssh.*-L"

# Test specific port
curl -v http://localhost:18090/api/system/status

# Check SSH connection with custom key and port
ssh -i ~/.ssh/id_vastai1 -p 42497 -v username@remote_server_ip

# View SSH logs
tail -f /var/log/auth.log  # On Linux
tail -f /var/log/secure    # On CentOS/RHEL
```

## 🚀 **Advanced Usage**

### **Persistent Tunnels (AutoSSH):**
```bash
# Install autossh
sudo apt-get install autossh

# Create persistent tunnel with custom key and port
autossh -M 20000 -f -N -i ~/.ssh/id_vastai1 -p 42497 -L 18100:localhost:8100 username@remote_server_ip
```

### **Multiple Remote Servers:**
```bash
# Create tunnels to different servers with different keys and ports
./ssh_tunnel_setup.sh server1_ip username1 ~/.ssh/id_vastai1 42497
./ssh_tunnel_setup.sh server2_ip username2 ~/.ssh/id_server2 22
```

### **Custom Port Mappings:**
Edit the script to change local port mappings:
```bash
# In ssh_tunnel_setup.sh, modify these lines:
COORDINATOR_PORT=8090          # Remote port
DASHBOARD_PORT=8100            # Remote port
# Change local ports as needed
```

### **SSH Config File:**
Create `~/.ssh/config` for easier management:
```bash
Host remote-rl-server
    HostName your_server_ip
    User your_username
    Port 42497
    IdentityFile ~/.ssh/id_vastai1
    ServerAliveInterval 60
    ServerAliveCountMax 3

# Then use: ssh remote-rl-server
```

## 📊 **Monitoring**

### Check tunnel health:
```bash
./ssh_tunnel_setup.sh
# Choose option 6: Test tunnel connectivity
```

### View detailed results:
```bash
cat tunnel_test_results.json
```

## 🔐 **Security Best Practices**

- **Use SSH keys** instead of passwords
- **Set key passphrases** for additional security
- **Restrict SSH access** on remote server
- **Use non-standard ports** for SSH (like 42497)
- **Monitor SSH logs** for unauthorized access
- **Regular key rotation** (every 6-12 months)

### **SSH Server Security (on remote server):**
```bash
# Edit /etc/ssh/sshd_config
PasswordAuthentication no
PubkeyAuthentication yes
PermitRootLogin no
AllowUsers your_username
Port 42497  # Custom SSH port

# Restart SSH service
sudo systemctl restart sshd
```

## 📞 **Support**

If you encounter issues:

1. **Check SSH key setup** first
2. **Verify remote server SSH configuration**
3. **Check SSH port accessibility**
4. **Run the connectivity test**: `python test_tunnel_connectivity.py`
5. **Check SSH logs** on both machines
6. **Verify firewall settings** on remote server

### **Common Error Messages:**
- `"Permission denied (publickey)"` → SSH key not set up correctly
- `"Connection timed out"` → Network/firewall issue or wrong SSH port
- `"Connection refused"` → Service not running or port blocked
- `"No route to host"` → Network connectivity issue

---

**Happy tunneling! 🚀** Your distributed RL system dashboard is now accessible from anywhere through secure SSH connections.
