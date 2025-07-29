# TRELLIS Output Directory Cleanup System

This system automatically cleans up the `trellis_submit_outputs` directory every 15 minutes, but only when the TRELLIS server is not processing any requests. This prevents generation failures while saving disk space.

## How It Works

The cleanup script:
1. **Monitors server status** via the `/job/status/` endpoint
2. **Checks processing state** - only cleans when server status is "idle", "completed", or "failed"
3. **Skips during processing** - never cleans when server status is "processing"
4. **Logs all activities** - tracks cleanup attempts, successes, and failures
5. **Provides statistics** - shows space saved and cleanup frequency

## Files

- `trellis_output_cleanup.py` - Main cleanup script
- `trellis-cleanup.service` - Systemd service file
- `install_cleanup_service.sh` - Installation script
- `README_cleanup.md` - This documentation

## Quick Start

### 1. Install as System Service (Recommended)

```bash
# Run the installation script
./install_cleanup_service.sh

# Start the service
sudo systemctl start trellis-cleanup

# Check status
sudo systemctl status trellis-cleanup
```

### 2. Manual Usage

```bash
# Test with dry-run (see what would be cleaned without actually deleting)
python3 trellis_output_cleanup.py --dry-run

# Run cleanup once and exit
python3 trellis_output_cleanup.py --once

# Run continuous cleanup with custom interval (e.g., 10 minutes)
python3 trellis_output_cleanup.py --interval 600

# Run with custom server URL
python3 trellis_output_cleanup.py --server-url http://localhost:8096
```

## Service Management

```bash
# Start the service
sudo systemctl start trellis-cleanup

# Stop the service
sudo systemctl stop trellis-cleanup

# Restart the service
sudo systemctl restart trellis-cleanup

# Check service status
sudo systemctl status trellis-cleanup

# View live logs
sudo journalctl -u trellis-cleanup -f

# View recent logs
sudo journalctl -u trellis-cleanup --since "1 hour ago"

# Enable/disable auto-start on boot
sudo systemctl enable trellis-cleanup
sudo systemctl disable trellis-cleanup
```

## Configuration

### Default Settings
- **Cleanup interval**: 15 minutes (900 seconds)
- **Server URL**: http://localhost:8096
- **Output directory**: ./trellis_submit_outputs
- **Log file**: trellis_cleanup.log

### Customization

You can modify the service file to change settings:

```bash
# Edit the service file
sudo nano /etc/systemd/system/trellis-cleanup.service

# Reload systemd after changes
sudo systemctl daemon-reload
sudo systemctl restart trellis-cleanup
```

## Safety Features

### Server Status Monitoring
The script checks the server's job status before cleaning:
- ✅ **Safe to clean**: "idle", "completed", "failed"
- ⏭️ **Skip cleaning**: "processing" (server is generating)
- ⚠️ **Conservative**: Unknown status (assume processing)

### Logging and Statistics
- All cleanup activities are logged with timestamps
- Statistics track successful cleanups, skipped cleanups, and space freed
- Dry-run mode available for testing

### Error Handling
- Connection failures don't break the service
- Invalid server responses are handled gracefully
- Service automatically restarts if it crashes

## Monitoring

### Check Cleanup Statistics
```bash
# View service logs for statistics
sudo journalctl -u trellis-cleanup | grep "CLEANUP STATISTICS" -A 15
```

### Monitor Disk Usage
```bash
# Check current disk usage
du -sh trellis_submit_outputs/

# Monitor disk usage over time
watch -n 60 'du -sh trellis_submit_outputs/'
```

### Check Server Status
```bash
# Check if server is processing
curl http://localhost:8096/job/status/
```

## Troubleshooting

### Service Won't Start
```bash
# Check service status
sudo systemctl status trellis-cleanup

# View detailed logs
sudo journalctl -u trellis-cleanup -n 50

# Check if Python script is executable
ls -la trellis_output_cleanup.py
```

### Cleanup Not Working
```bash
# Test server connectivity
curl http://localhost:8096/health/

# Test job status endpoint
curl http://localhost:8096/job/status/

# Run manual cleanup test
python3 trellis_output_cleanup.py --once --dry-run
```

### Permission Issues
```bash
# Check file permissions
ls -la trellis_output_cleanup.py

# Fix permissions if needed
chmod +x trellis_output_cleanup.py
```

## Uninstallation

```bash
# Stop and disable the service
sudo systemctl stop trellis-cleanup
sudo systemctl disable trellis-cleanup

# Remove service file
sudo rm /etc/systemd/system/trellis-cleanup.service

# Reload systemd
sudo systemctl daemon-reload
```

## Log Files

- **Service logs**: `sudo journalctl -u trellis-cleanup`
- **Script logs**: `trellis_cleanup.log` (in working directory)

## Performance Impact

The cleanup script is designed to be lightweight:
- Minimal CPU usage (only runs every 15 minutes)
- Small memory footprint
- Non-blocking HTTP requests to server
- Efficient file system operations

The script will not interfere with TRELLIS generation performance as it only runs when the server is idle. 