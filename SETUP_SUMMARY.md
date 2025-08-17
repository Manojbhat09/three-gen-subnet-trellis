# 🚀 Complete SSH Tunneling Setup for Distributed RL System

## 📋 **What You Get**

This setup provides **complete remote access** to your distributed RL system dashboard through secure SSH tunneling. You can monitor your 8-GPU system, submit jobs, and view insights from your local machine.

## 🔐 **SSH Key Authentication - REQUIRED**

**Yes, you absolutely need SSH keys!** Password authentication won't work for tunneling. Here's why and how:

### **Why SSH Keys?**
- **Security**: More secure than passwords
- **Automation**: Required for non-interactive tunneling
- **Reliability**: No password prompts during tunnel creation
- **Best Practice**: Industry standard for server access

### **Custom Private Key Support**
The scripts now support **custom private key paths** like `~/.ssh/id_vastai1` and **custom SSH ports** like `42497`, automatically using the `-i` and `-p` flags for SSH connections.

## 🎯 **Complete Setup Process (3 Steps)**

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

## 📁 **Files Created**

| File | Purpose | Usage |
|------|---------|-------|
| `setup_ssh_keys.sh` | 🔑 SSH key generation & setup | First-time setup |
| `ssh_tunnel_setup.sh` | 🚀 Interactive tunnel management | Full-featured setup |
| `quick_tunnel.sh` | ⚡ Quick tunnel creation | Fast setup |
| `test_tunnel_connectivity.py` | 🧪 Connectivity testing | Verify setup |
| `README_SSH_TUNNEL.md` | 📖 Complete documentation | Reference guide |

## 🌐 **Port Mapping**

| Service | Remote Port | Local Port | Access URL |
|---------|-------------|------------|------------|
| **Dashboard** | 8100 | 18100 | http://localhost:18100 |
| **Coordinator API** | 8090 | 18090 | http://localhost:18090 |
| **GPU Agent 0** | 8096 | 18101 | http://localhost:18101 |
| **GPU Agent 1** | 8097 | 18102 | http://localhost:18102 |
| **GPU Agent 2** | 8098 | 18103 | http://localhost:18103 |
| **GPU Agent 3** | 8099 | 18104 | http://localhost:18104 |
| **GPU Agent 4** | 8100 | 18105 | http://localhost:18105 |
| **GPU Agent 5** | 8101 | 18106 | http://localhost:18106 |
| **GPU Agent 6** | 8102 | 18107 | http://localhost:18107 |
| **GPU Agent 7** | 8103 | 18108 | http://localhost:18108 |

## 🚀 **Quick Start Commands**

### **From Your Local Machine:**

```bash
# 1. Clone/copy these files to your local machine
cd /path/to/your/local/directory

# 2. Make scripts executable
chmod +x *.sh

# 3. Setup SSH keys (one-time)
./setup_ssh_keys.sh

# 4. Create tunnels with custom key and port
./quick_tunnel.sh YOUR_REMOTE_IP YOUR_USERNAME ~/.ssh/id_vastai1 42497

# 5. Test connectivity
python test_tunnel_connectivity.py

# 6. Access dashboard
open http://localhost:18100
```

### **Replace with Your Details:**
- `YOUR_REMOTE_IP`: Your remote server's IP address
- `YOUR_USERNAME`: Your username on the remote server
- `~/.ssh/id_vastai1`: Path to your private key file
- `42497`: SSH port (or your custom port)

## 🔍 **What Gets Tested**

The Python testing script validates:
- ✅ **SSH tunnel connectivity** to all services
- ✅ **Coordinator API** endpoints (`/api/system/status`, `/api/jobs`, `/api/insights`)
- ✅ **Dashboard frontend** accessibility
- ✅ **All 8 GPU agents** individual status
- ✅ **Job submission** functionality
- ✅ **Cross-GPU communication** verification

## 🛠️ **Troubleshooting**

### **Common Issues & Solutions:**

1. **"Permission denied (publickey)"**
   ```bash
   # Check SSH key setup
   ./setup_ssh_keys.sh
   
   # Or verify manually
   ls -la ~/.ssh/id_vastai1
   chmod 600 ~/.ssh/id_vastai1
   ```

2. **"Connection refused" on SSH port**
   ```bash
   # Verify SSH port is correct
   telnet remote_server_ip 42497
   
   # Check if server is listening on that port
   nmap -p 42497 remote_server_ip
   ```

3. **"Port already in use"**
   ```bash
   # Kill existing tunnels
   ./ssh_tunnel_setup.sh → Option 7
   ```

4. **Slow response times**
   - Normal for SSH tunneling
   - Consider direct connection for production

### **Debug Commands:**
```bash
# Check active tunnels
netstat -tlnp | grep "127.0.0.1:18"

# Check SSH processes
ps aux | grep "ssh.*-L"

# Test specific service
curl -v http://localhost:18090/api/system/status

# Check SSH connection with custom key and port
ssh -i ~/.ssh/id_vastai1 -p 42497 -v username@remote_server_ip
```

## 🔐 **Security Features**

- **Encrypted tunnels**: All traffic encrypted via SSH
- **Key-based auth**: No passwords transmitted
- **Custom key support**: Use any private key path
- **Custom port support**: Use any SSH port (like 42497)
- **Local access only**: Tunnels only accessible from localhost
- **Automatic cleanup**: Scripts can kill tunnels when done

## 📊 **Monitoring & Management**

### **Check Tunnel Status:**
```bash
./ssh_tunnel_setup.sh
# Choose option 5: List active tunnels
```

### **Test Connectivity:**
```bash
./ssh_tunnel_setup.sh
# Choose option 6: Test tunnel connectivity
```

### **Kill All Tunnels:**
```bash
./ssh_tunnel_setup.sh
# Choose option 7: Kill all tunnels
```

## 🎯 **Use Cases**

### **Perfect For:**
- **Remote monitoring** of your distributed RL system
- **Development** from your local machine
- **Debugging** system issues remotely
- **Demo presentations** from anywhere
- **Team collaboration** without VPN setup

### **Not Recommended For:**
- **Production deployment** (use direct connections)
- **High-traffic scenarios** (SSH overhead)
- **Real-time gaming** (latency issues)

## 🚀 **Advanced Features**

### **Persistent Tunnels:**
```bash
# Install autossh for auto-reconnection
sudo apt-get install autossh

# Create persistent tunnel with custom key and port
autossh -M 20000 -f -N -i ~/.ssh/id_vastai1 -p 42497 -L 18100:localhost:8100 username@remote_server_ip
```

### **SSH Config File:**
```bash
# Create ~/.ssh/config for easier management
Host rl-server
    HostName your_server_ip
    User your_username
    Port 42497
    IdentityFile ~/.ssh/id_vastai1
    ServerAliveInterval 60
    ServerAliveCountMax 3

# Then use: ssh rl-server
```

### **Multiple Remote Servers:**
```bash
# Create tunnels to different servers with different keys and ports
./ssh_tunnel_setup.sh server1_ip username1 ~/.ssh/id_vastai1 42497
./ssh_tunnel_setup.sh server2_ip username2 ~/.ssh/id_server2 22
```

## 📞 **Support & Next Steps**

### **If You Need Help:**
1. **Check troubleshooting section** above
2. **Verify SSH port accessibility**
3. **Run the connectivity test**: `python test_tunnel_connectivity.py`
4. **Check SSH configuration** on both machines
5. **Check remote service logs** for errors

### **Next Steps After Setup:**
1. **Access your dashboard**: http://localhost:18100
2. **Monitor system status**: http://localhost:18090/api/system/status
3. **Submit test jobs** through the dashboard
4. **Explore GPU monitoring** at individual agent URLs

---

## 🎉 **You're All Set!**

With this setup, you now have:
- ✅ **Secure remote access** to your distributed RL system
- ✅ **Complete dashboard functionality** from your local machine
- ✅ **Real-time monitoring** of all 8 GPUs
- ✅ **Job submission and tracking** capabilities
- ✅ **Cross-GPU insights** and learning patterns
- ✅ **Custom private key support** for flexible authentication
- ✅ **Custom SSH port support** for non-standard configurations

**Happy tunneling! 🚀** Your distributed RL system is now accessible from anywhere through secure SSH connections.
