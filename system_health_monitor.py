#!/usr/bin/env python3
"""
Lightweight System Health Monitor with Rich Display
Monitors CPU, GPU, Memory, and Disk usage with minimal resource consumption
"""

import psutil
import time
import datetime
import subprocess
import json
import os
import signal
import sys
from pathlib import Path
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.layout import Layout
from rich.live import Live
from rich.text import Text
from rich.progress import Progress, BarColumn, TextColumn
from rich.columns import Columns
from rich.align import Align

class SystemHealthMonitor:
    def __init__(self, log_file=None, update_interval=5):
        """
        Initialize the system health monitor
        
        Args:
            log_file: Path to log file (None for stdout only)
            update_interval: Seconds between updates (default: 5)
        """
        self.log_file = log_file
        self.update_interval = update_interval
        self.running = True
        self.console = Console()
        
        # Create log directory if needed
        if self.log_file:
            Path(self.log_file).parent.mkdir(parents=True, exist_ok=True)
        
        # Set up signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals gracefully"""
        self.running = False
        print("\n\nShutting down monitor...")
        sys.exit(0)
    
    def get_cpu_usage(self):
        """Get CPU usage percentage"""
        return psutil.cpu_percent(interval=1, percpu=True)
    
    def get_memory_usage(self):
        """Get memory usage statistics"""
        mem = psutil.virtual_memory()
        return {
            'total_gb': round(mem.total / (1024**3), 2),
            'used_gb': round(mem.used / (1024**3), 2),
            'available_gb': round(mem.available / (1024**3), 2),
            'percent': mem.percent
        }
    
    def get_disk_usage(self):
        """Get disk usage for main partitions"""
        disk_info = []
        seen_devices = set()
        
        for partition in psutil.disk_partitions():
            # Skip duplicate devices and special mounts
            if partition.device in seen_devices or partition.mountpoint.startswith('/usr/') or partition.mountpoint.startswith('/var/lib/'):
                continue
                
            try:
                usage = psutil.disk_usage(partition.mountpoint)
                # Only include partitions with significant size
                if usage.total > 1024**3:  # > 1GB
                    disk_info.append({
                        'device': partition.device,
                        'mountpoint': partition.mountpoint,
                        'total_gb': round(usage.total / (1024**3), 2),
                        'used_gb': round(usage.used / (1024**3), 2),
                        'free_gb': round(usage.free / (1024**3), 2),
                        'percent': usage.percent
                    })
                    seen_devices.add(partition.device)
            except PermissionError:
                continue
                
        # Sort by mountpoint and limit to top 5
        return sorted(disk_info, key=lambda x: x['mountpoint'])[:5]
    
    def get_gpu_usage(self):
        """Get GPU usage using nvidia-smi"""
        try:
            cmd = [
                'nvidia-smi',
                '--query-gpu=index,name,temperature.gpu,utilization.gpu,memory.used,memory.total',
                '--format=csv,noheader,nounits'
            ]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=5)
            
            if result.returncode != 0:
                return None
            
            gpu_info = []
            for line in result.stdout.strip().split('\n'):
                parts = line.split(', ')
                if len(parts) >= 6:
                    gpu_info.append({
                        'index': int(parts[0]),
                        'name': parts[1],
                        'temperature': float(parts[2]),
                        'utilization': float(parts[3]),
                        'memory_used_mb': float(parts[4]),
                        'memory_total_mb': float(parts[5]),
                        'memory_percent': round((float(parts[4]) / float(parts[5])) * 100, 2)
                    })
            return gpu_info
        except Exception:
            return None
    
    def get_network_io(self):
        """Get network I/O statistics"""
        net_io = psutil.net_io_counters()
        return {
            'bytes_sent_gb': round(net_io.bytes_sent / (1024**3), 2),
            'bytes_recv_gb': round(net_io.bytes_recv / (1024**3), 2)
        }
    
    def get_processes_summary(self, top_n=5):
        """Get top N processes by CPU and memory usage"""
        processes = []
        for proc in psutil.process_iter(['pid', 'name', 'cpu_percent', 'memory_percent']):
            try:
                pinfo = proc.info
                if pinfo['cpu_percent'] > 0 or pinfo['memory_percent'] > 0:
                    processes.append({
                        'pid': pinfo['pid'],
                        'name': pinfo['name'][:20],
                        'cpu_percent': round(pinfo['cpu_percent'], 2),
                        'memory_percent': round(pinfo['memory_percent'], 2)
                    })
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass
        
        # Sort by CPU usage and get top N
        top_cpu = sorted(processes, key=lambda x: x['cpu_percent'], reverse=True)[:top_n]
        
        # Sort by memory usage and get top N
        top_memory = sorted(processes, key=lambda x: x['memory_percent'], reverse=True)[:top_n]
        
        return {
            'top_cpu': top_cpu,
            'top_memory': top_memory
        }
    
    def collect_metrics(self):
        """Collect all system metrics"""
        timestamp = datetime.datetime.now()
        
        metrics = {
            'timestamp': timestamp.isoformat(),
            'timestamp_obj': timestamp,
            'cpu': {
                'usage_per_core': self.get_cpu_usage(),
                'average': round(psutil.cpu_percent(interval=0.1), 2)
            },
            'memory': self.get_memory_usage(),
            'disk': self.get_disk_usage(),
            'network': self.get_network_io(),
            'processes': self.get_processes_summary()
        }
        
        # Add GPU info if available
        gpu_info = self.get_gpu_usage()
        if gpu_info:
            metrics['gpu'] = gpu_info
        
        return metrics
    
    def create_cpu_panel(self, metrics):
        """Create CPU usage panel"""
        cpu_data = metrics['cpu']
        
        # Create CPU table
        cpu_table = Table(show_header=False, padding=0, box=None)
        cpu_table.add_column("Label", style="cyan")
        cpu_table.add_column("Value", style="green")
        
        # Overall CPU
        cpu_bar = self._create_progress_bar(cpu_data['average'], "yellow")
        cpu_table.add_row("Average", cpu_bar)
        
        # Per-core usage (show first 8 cores in compact format)
        cores = cpu_data['usage_per_core'][:8]
        core_bars = []
        for i, usage in enumerate(cores):
            color = "green" if usage < 50 else "yellow" if usage < 80 else "red"
            core_bars.append(f"[{color}]C{i}: {usage:>4.1f}%[/]")
        
        if len(cores) > 4:
            cpu_table.add_row("Cores 0-3", "  ".join(core_bars[:4]))
            cpu_table.add_row("Cores 4-7", "  ".join(core_bars[4:8]))
        else:
            cpu_table.add_row("Cores", "  ".join(core_bars))
        
        return Panel(cpu_table, title="🖥️  CPU Usage", border_style="blue")
    
    def create_memory_panel(self, metrics):
        """Create memory usage panel"""
        mem = metrics['memory']
        
        mem_table = Table(show_header=False, padding=0, box=None)
        mem_table.add_column("Label", style="cyan")
        mem_table.add_column("Value", style="green")
        
        mem_bar = self._create_progress_bar(mem['percent'], "magenta")
        mem_table.add_row("Usage", mem_bar)
        mem_table.add_row("Used", f"{mem['used_gb']} / {mem['total_gb']} GB")
        mem_table.add_row("Available", f"{mem['available_gb']} GB")
        
        return Panel(mem_table, title="💾 Memory", border_style="magenta")
    
    def create_gpu_panel(self, metrics):
        """Create GPU usage panel"""
        if 'gpu' not in metrics or not metrics['gpu']:
            return Panel("No NVIDIA GPU detected", title="🎮 GPU", border_style="yellow")
        
        gpu_table = Table(show_header=False, padding=0, box=None)
        
        for gpu in metrics['gpu'][:2]:  # Show max 2 GPUs
            gpu_table.add_column(f"GPU {gpu['index']}", style="green")
        
        # Add rows for each metric
        row_data = {
            'Name': [],
            'Temp': [],
            'Util': [],
            'Memory': []
        }
        
        for gpu in metrics['gpu'][:2]:
            row_data['Name'].append(gpu['name'][:20])
            row_data['Temp'].append(f"{gpu['temperature']:.0f}°C")
            
            util_bar = self._create_progress_bar(gpu['utilization'], "cyan", width=15)
            row_data['Util'].append(util_bar)
            
            mem_bar = self._create_progress_bar(gpu['memory_percent'], "yellow", width=15)
            row_data['Memory'].append(mem_bar)
        
        for label, values in row_data.items():
            gpu_table.add_row(*values)
        
        return Panel(gpu_table, title="🎮 GPU", border_style="yellow")
    
    def create_disk_panel(self, metrics):
        """Create disk usage panel"""
        disk_table = Table(show_header=False, padding=0, box=None)
        disk_table.add_column("Mount", style="cyan", width=15)
        disk_table.add_column("Usage", style="green")
        
        for disk in metrics['disk'][:3]:  # Show top 3 disks
            mount = disk['mountpoint']
            if len(mount) > 15:
                mount = "..." + mount[-12:]
            
            bar = self._create_progress_bar(disk['percent'], "blue", width=20)
            info = f"{disk['used_gb']}/{disk['total_gb']}GB"
            disk_table.add_row(mount, f"{bar} {info}")
        
        return Panel(disk_table, title="💿 Disk Usage", border_style="green")
    
    def create_network_panel(self, metrics):
        """Create network I/O panel"""
        net = metrics['network']
        
        net_table = Table(show_header=False, padding=0, box=None)
        net_table.add_column("Direction", style="cyan")
        net_table.add_column("Amount", style="green")
        
        net_table.add_row("⬆️  Sent", f"{net['bytes_sent_gb']} GB")
        net_table.add_row("⬇️  Received", f"{net['bytes_recv_gb']} GB")
        
        return Panel(net_table, title="🌐 Network I/O", border_style="cyan")
    
    def create_process_panel(self, metrics):
        """Create top processes panel"""
        proc_table = Table(padding=0, box=None)
        proc_table.add_column("Process", style="cyan", width=20)
        proc_table.add_column("PID", style="white", width=8)
        proc_table.add_column("CPU%", style="yellow", width=8)
        proc_table.add_column("MEM%", style="magenta", width=8)
        
        # Combine top CPU and memory processes, remove duplicates
        all_procs = {}
        for proc in metrics['processes']['top_cpu'] + metrics['processes']['top_memory']:
            pid = proc['pid']
            if pid not in all_procs or proc['cpu_percent'] > all_procs[pid]['cpu_percent']:
                all_procs[pid] = proc
        
        # Sort by CPU usage and show top 5
        sorted_procs = sorted(all_procs.values(), key=lambda x: x['cpu_percent'], reverse=True)[:5]
        
        for proc in sorted_procs:
            proc_table.add_row(
                proc['name'][:20],
                str(proc['pid']),
                f"{proc['cpu_percent']:>5.1f}%",
                f"{proc['memory_percent']:>5.1f}%"
            )
        
        return Panel(proc_table, title="📊 Top Processes", border_style="red")
    
    def _create_progress_bar(self, percent, color, width=25):
        """Create a simple progress bar"""
        filled = int(percent / 100 * width)
        bar = "█" * filled + "░" * (width - filled)
        return f"[{color}]{bar}[/] {percent:>5.1f}%"
    
    def create_display(self, metrics):
        """Create the main display layout"""
        layout = Layout()
        
        # Create header
        header_text = Text(f"System Health Monitor - {metrics['timestamp_obj'].strftime('%Y-%m-%d %H:%M:%S')}", 
                          style="bold white", justify="center")
        header = Panel(header_text, style="bold blue", height=3)
        
        # Create main layout structure
        layout.split_column(
            Layout(header, size=3),
            Layout(name="main", ratio=1)
        )
        
        # Split main area into grid
        layout["main"].split_row(
            Layout(name="left", ratio=1),
            Layout(name="middle", ratio=1),
            Layout(name="right", ratio=1)
        )
        
        # Left column
        layout["left"].split_column(
            Layout(self.create_cpu_panel(metrics), ratio=2),
            Layout(self.create_memory_panel(metrics), ratio=1)
        )
        
        # Middle column
        if 'gpu' in metrics and metrics['gpu']:
            layout["middle"].split_column(
                Layout(self.create_gpu_panel(metrics), ratio=1),
                Layout(self.create_disk_panel(metrics), ratio=1)
            )
        else:
            layout["middle"] = self.create_disk_panel(metrics)
        
        # Right column
        layout["right"].split_column(
            Layout(self.create_network_panel(metrics), ratio=1),
            Layout(self.create_process_panel(metrics), ratio=2)
        )
        
        return layout
    
    def log_metrics(self, metrics):
        """Log metrics to file if specified"""
        if self.log_file:
            # Remove timestamp_obj before saving
            metrics_copy = metrics.copy()
            metrics_copy.pop('timestamp_obj', None)
            
            with open(self.log_file, 'a') as f:
                f.write(json.dumps(metrics_copy) + '\n')
    
    def run(self):
        """Main monitoring loop"""
        self.console.clear()
        
        print(f"Starting System Health Monitor...")
        print(f"Update interval: {self.update_interval} seconds")
        if self.log_file:
            print(f"Logging to: {self.log_file}")
        print(f"Press Ctrl+C to stop\n")
        
        time.sleep(2)
        self.console.clear()
        
        with Live(console=self.console, refresh_per_second=1, screen=True) as live:
            while self.running:
                try:
                    # Collect metrics
                    metrics = self.collect_metrics()
                    
                    # Update display
                    display = self.create_display(metrics)
                    live.update(display)
                    
                    # Log metrics
                    self.log_metrics(metrics)
                    
                    # Sleep until next update
                    time.sleep(self.update_interval)
                    
                except KeyboardInterrupt:
                    break
                except Exception as e:
                    self.console.print(f"[red]Error collecting metrics: {e}[/]")
                    time.sleep(self.update_interval)


def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Lightweight System Health Monitor with Rich Display')
    parser.add_argument('--log-file', '-l', help='Path to log file (JSON format)')
    parser.add_argument('--interval', '-i', type=int, default=5, 
                       help='Update interval in seconds (default: 5)')
    parser.add_argument('--minimal', '-m', action='store_true',
                       help='Minimal mode - longer interval (30s) and no process tracking')
    
    args = parser.parse_args()
    
    # Adjust settings for minimal mode
    if args.minimal:
        args.interval = max(args.interval, 30)
        print("Running in minimal mode (30s interval, no process tracking)")
    
    # Create and run monitor
    monitor = SystemHealthMonitor(
        log_file=args.log_file,
        update_interval=args.interval
    )
    
    # Override process tracking in minimal mode
    if args.minimal:
        monitor.get_processes_summary = lambda top_n=5: {'top_cpu': [], 'top_memory': []}
    
    try:
        monitor.run()
    except KeyboardInterrupt:
        print("\nMonitor stopped.")
        sys.exit(0)


if __name__ == '__main__':
    main() 