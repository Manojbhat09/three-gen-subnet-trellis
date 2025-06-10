#!/usr/bin/env python3
# Mining Orchestrator for Subnet 17
# Centralized process management, health monitoring, and automated recovery

import asyncio
import json
import os
import signal
import subprocess
import sys
import time
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
from enum import Enum
import psutil
import aiohttp
from collections import deque

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('mining_orchestrator.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class ProcessState(Enum):
    STOPPED = "stopped"
    STARTING = "starting"
    RUNNING = "running"
    UNHEALTHY = "unhealthy"
    RESTARTING = "restarting"
    FAILED = "failed"

@dataclass
class ProcessConfig:
    name: str
    script_path: str
    port: Optional[int] = None
    health_endpoint: Optional[str] = None
    startup_time: int = 30  # seconds
    restart_delay: int = 10  # seconds
    max_restarts: int = 5
    restart_window: int = 300  # 5 minutes
    environment: Dict[str, str] = None
    working_directory: Optional[str] = None
    
    def __post_init__(self):
        if self.environment is None:
            self.environment = {}

@dataclass
class ProcessStatus:
    config: ProcessConfig
    state: ProcessState = ProcessState.STOPPED
    pid: Optional[int] = None
    start_time: Optional[float] = None
    restart_count: int = 0
    restart_history: deque = None
    last_health_check: Optional[float] = None
    health_status: Optional[Dict] = None
    error_log: deque = None
    
    def __post_init__(self):
        if self.restart_history is None:
            self.restart_history = deque(maxlen=20)
        if self.error_log is None:
            self.error_log = deque(maxlen=100)

class MiningOrchestrator:
    def __init__(self, config_file: str = "orchestrator_config.json"):
        self.config_file = Path(config_file)
        self.processes: Dict[str, ProcessStatus] = {}
        self.shutdown_requested = False
        
        # Define default process configurations
        self.default_configs = {
            "generation_server": ProcessConfig(
                name="generation_server",
                script_path="robust_generation_server.py",
                port=8093,
                health_endpoint="http://127.0.0.1:8093/health",
                startup_time=60,  # Generation server takes longer to load models
                restart_delay=30,
                max_restarts=3,
                working_directory="."
            ),
            "validation_server": ProcessConfig(
                name="validation_server",
                script_path="validation_server.py",
                port=8094,
                health_endpoint="http://127.0.0.1:8094/health",
                startup_time=30,
                restart_delay=15,
                max_restarts=5,
                working_directory="."
            ),
            "subnet17_miner": ProcessConfig(
                name="subnet17_miner",
                script_path="robust_subnet17_miner.py",
                startup_time=20,
                restart_delay=10,
                max_restarts=10,
                working_directory="."
            ),
            "performance_optimizer": ProcessConfig(
                name="performance_optimizer",
                script_path="performance_optimizer.py",
                startup_time=10,
                restart_delay=5,
                max_restarts=3,
                working_directory="."
            )
        }
        
        # Initialize process statuses
        for name, config in self.default_configs.items():
            self.processes[name] = ProcessStatus(config=config)
            
        self.load_config()
        
    def load_config(self):
        """Load orchestrator configuration from file"""
        if self.config_file.exists():
            try:
                with open(self.config_file, 'r') as f:
                    config_data = json.load(f)
                    
                # Update process configurations
                for process_name, process_config in config_data.get("processes", {}).items():
                    if process_name in self.processes:
                        # Update existing config
                        for key, value in process_config.items():
                            if hasattr(self.processes[process_name].config, key):
                                setattr(self.processes[process_name].config, key, value)
                                
                logger.info("Configuration loaded successfully")
                
            except Exception as e:
                logger.error(f"Failed to load configuration: {e}")
                
    def save_config(self):
        """Save current configuration to file"""
        config_data = {
            "processes": {},
            "last_updated": time.time()
        }
        
        for name, status in self.processes.items():
            config_data["processes"][name] = asdict(status.config)
            
        try:
            with open(self.config_file, 'w') as f:
                json.dump(config_data, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save configuration: {e}")
            
    async def start_process(self, process_name: str) -> bool:
        """Start a specific process"""
        if process_name not in self.processes:
            logger.error(f"Unknown process: {process_name}")
            return False
            
        status = self.processes[process_name]
        config = status.config
        
        if status.state in [ProcessState.RUNNING, ProcessState.STARTING]:
            logger.warning(f"Process {process_name} is already running or starting")
            return True
            
        logger.info(f"Starting process: {process_name}")
        status.state = ProcessState.STARTING
        
        try:
            # Prepare environment
            env = os.environ.copy()
            env.update(config.environment)
            
            # Start process
            cmd = [sys.executable, config.script_path]
            
            process = subprocess.Popen(
                cmd,
                cwd=config.working_directory or ".",
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                start_new_session=True
            )
            
            status.pid = process.pid
            status.start_time = time.time()
            status.state = ProcessState.RUNNING
            
            logger.info(f"Process {process_name} started with PID {process.pid}")
            
            # Wait for startup
            await asyncio.sleep(min(config.startup_time, 10))
            
            # Verify process is still running
            if not self.is_process_running(process_name):
                logger.error(f"Process {process_name} failed to start properly")
                status.state = ProcessState.FAILED
                return False
                
            return True
            
        except Exception as e:
            logger.error(f"Failed to start process {process_name}: {e}")
            status.state = ProcessState.FAILED
            status.error_log.append(f"{time.time()}: Start failed: {e}")
            return False
            
    async def stop_process(self, process_name: str, force: bool = False) -> bool:
        """Stop a specific process"""
        if process_name not in self.processes:
            logger.error(f"Unknown process: {process_name}")
            return False
            
        status = self.processes[process_name]
        
        if status.state == ProcessState.STOPPED:
            logger.info(f"Process {process_name} is already stopped")
            return True
            
        if not status.pid:
            logger.warning(f"No PID recorded for process {process_name}")
            status.state = ProcessState.STOPPED
            return True
            
        logger.info(f"Stopping process: {process_name} (PID: {status.pid})")
        
        try:
            # Try graceful shutdown first
            if not force:
                os.kill(status.pid, signal.SIGTERM)
                
                # Wait for graceful shutdown
                for _ in range(10):
                    if not self.is_process_running(process_name):
                        break
                    await asyncio.sleep(1)
                    
            # Force kill if still running
            if self.is_process_running(process_name):
                logger.warning(f"Force killing process {process_name}")
                os.kill(status.pid, signal.SIGKILL)
                await asyncio.sleep(2)
                
            status.state = ProcessState.STOPPED
            status.pid = None
            status.start_time = None
            
            logger.info(f"Process {process_name} stopped successfully")
            return True
            
        except ProcessLookupError:
            # Process already dead
            status.state = ProcessState.STOPPED
            status.pid = None
            status.start_time = None
            return True
            
        except Exception as e:
            logger.error(f"Failed to stop process {process_name}: {e}")
            return False
            
    def is_process_running(self, process_name: str) -> bool:
        """Check if a process is running"""
        status = self.processes.get(process_name)
        if not status or not status.pid:
            return False
            
        try:
            # Check if PID exists and is the correct process
            process = psutil.Process(status.pid)
            return process.is_running() and process.status() != psutil.STATUS_ZOMBIE
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            return False
            
    async def check_process_health(self, process_name: str) -> bool:
        """Check health of a specific process"""
        status = self.processes.get(process_name)
        if not status:
            return False
            
        config = status.config
        
        # Basic process existence check
        if not self.is_process_running(process_name):
            status.state = ProcessState.STOPPED
            return False
            
        # Health endpoint check
        if config.health_endpoint:
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.get(config.health_endpoint, timeout=10) as response:
                        if response.status == 200:
                            health_data = await response.json()
                            status.health_status = health_data
                            status.last_health_check = time.time()
                            
                            # Check if service reports itself as healthy
                            service_healthy = health_data.get("can_accept_requests", True)
                            if not service_healthy:
                                status.state = ProcessState.UNHEALTHY
                                return False
                                
                            status.state = ProcessState.RUNNING
                            return True
                        else:
                            status.state = ProcessState.UNHEALTHY
                            return False
                            
            except Exception as e:
                logger.warning(f"Health check failed for {process_name}: {e}")
                status.state = ProcessState.UNHEALTHY
                status.error_log.append(f"{time.time()}: Health check failed: {e}")
                return False
        
        # If no health endpoint, assume healthy if running
        status.state = ProcessState.RUNNING
        return True
        
    async def restart_process(self, process_name: str) -> bool:
        """Restart a specific process"""
        if process_name not in self.processes:
            logger.error(f"Unknown process: {process_name}")
            return False
            
        status = self.processes[process_name]
        config = status.config
        
        # Check restart limits
        current_time = time.time()
        recent_restarts = [
            t for t in status.restart_history 
            if current_time - t < config.restart_window
        ]
        
        if len(recent_restarts) >= config.max_restarts:
            logger.error(f"Process {process_name} has exceeded restart limit ({config.max_restarts} in {config.restart_window}s)")
            status.state = ProcessState.FAILED
            return False
            
        logger.info(f"Restarting process: {process_name}")
        status.state = ProcessState.RESTARTING
        
        # Stop the process
        await self.stop_process(process_name)
        
        # Wait for restart delay
        await asyncio.sleep(config.restart_delay)
        
        # Start the process
        success = await self.start_process(process_name)
        
        if success:
            status.restart_count += 1
            status.restart_history.append(current_time)
            logger.info(f"Process {process_name} restarted successfully")
        else:
            logger.error(f"Failed to restart process {process_name}")
            
        return success
        
    async def monitor_processes(self):
        """Monitor all processes and restart if needed"""
        logger.info("Process monitoring started")
        
        while not self.shutdown_requested:
            try:
                for process_name, status in self.processes.items():
                    if status.state == ProcessState.FAILED:
                        continue  # Don't monitor failed processes
                        
                    # Check if process should be running
                    if status.state in [ProcessState.RUNNING, ProcessState.UNHEALTHY]:
                        healthy = await self.check_process_health(process_name)
                        
                        if not healthy:
                            logger.warning(f"Process {process_name} is unhealthy, restarting...")
                            await self.restart_process(process_name)
                            
                    elif status.state == ProcessState.STOPPED:
                        # Auto-restart stopped processes
                        logger.info(f"Auto-starting stopped process: {process_name}")
                        await self.start_process(process_name)
                        
                await asyncio.sleep(30)  # Check every 30 seconds
                
            except Exception as e:
                logger.error(f"Error in process monitoring: {e}")
                await asyncio.sleep(10)
                
    async def start_all_processes(self):
        """Start all configured processes"""
        logger.info("Starting all processes...")
        
        # Start in order: validation_server, generation_server, optimizer, miner
        start_order = [
            "validation_server",
            "generation_server", 
            "performance_optimizer",
            "subnet17_miner"
        ]
        
        for process_name in start_order:
            if process_name in self.processes:
                await self.start_process(process_name)
                await asyncio.sleep(5)  # Stagger starts
                
        logger.info("All processes started")
        
    async def stop_all_processes(self):
        """Stop all processes"""
        logger.info("Stopping all processes...")
        
        # Stop in reverse order
        stop_order = [
            "subnet17_miner",
            "performance_optimizer", 
            "generation_server",
            "validation_server"
        ]
        
        for process_name in stop_order:
            if process_name in self.processes:
                await self.stop_process(process_name)
                
        logger.info("All processes stopped")
        
    async def status_reporter(self):
        """Report system status periodically"""
        while not self.shutdown_requested:
            try:
                status_report = {
                    "timestamp": datetime.now().isoformat(),
                    "processes": {}
                }
                
                for name, status in self.processes.items():
                    status_report["processes"][name] = {
                        "state": status.state.value,
                        "pid": status.pid,
                        "uptime": time.time() - status.start_time if status.start_time else 0,
                        "restart_count": status.restart_count,
                        "last_health_check": status.last_health_check,
                        "health_status": status.health_status
                    }
                    
                logger.info(f"Status Report: {json.dumps(status_report, indent=2)}")
                
                # Save status to file
                status_file = Path("mining_status.json")
                with open(status_file, 'w') as f:
                    json.dump(status_report, f, indent=2)
                    
            except Exception as e:
                logger.error(f"Error in status reporting: {e}")
                
            await asyncio.sleep(300)  # Report every 5 minutes
            
    async def handle_shutdown(self):
        """Handle graceful shutdown"""
        logger.info("Shutdown signal received")
        self.shutdown_requested = True
        
        await self.stop_all_processes()
        self.save_config()
        
        logger.info("Mining orchestrator shutdown complete")
        
    async def run(self):
        """Main orchestrator loop"""
        logger.info("Mining Orchestrator starting...")
        
        # Set up signal handlers
        loop = asyncio.get_event_loop()
        for sig in [signal.SIGTERM, signal.SIGINT]:
            loop.add_signal_handler(sig, lambda: asyncio.create_task(self.handle_shutdown()))
            
        try:
            # Start all background tasks
            tasks = [
                asyncio.create_task(self.monitor_processes()),
                asyncio.create_task(self.status_reporter()),
            ]
            
            # Start all processes
            await self.start_all_processes()
            
            # Wait for shutdown or task completion
            await asyncio.gather(*tasks, return_exceptions=True)
            
        except Exception as e:
            logger.error(f"Error in orchestrator main loop: {e}")
            
        finally:
            await self.handle_shutdown()

# CLI Interface
async def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Mining Orchestrator for Subnet 17")
    parser.add_argument("command", choices=["start", "stop", "restart", "status", "run"], 
                       help="Command to execute")
    parser.add_argument("--process", help="Specific process name (for start/stop/restart)")
    parser.add_argument("--config", default="orchestrator_config.json", 
                       help="Configuration file path")
    
    args = parser.parse_args()
    
    orchestrator = MiningOrchestrator(args.config)
    
    if args.command == "run":
        await orchestrator.run()
    elif args.command == "start":
        if args.process:
            success = await orchestrator.start_process(args.process)
            sys.exit(0 if success else 1)
        else:
            await orchestrator.start_all_processes()
    elif args.command == "stop":
        if args.process:
            success = await orchestrator.stop_process(args.process)
            sys.exit(0 if success else 1)
        else:
            await orchestrator.stop_all_processes()
    elif args.command == "restart":
        if args.process:
            success = await orchestrator.restart_process(args.process)
            sys.exit(0 if success else 1)
        else:
            await orchestrator.stop_all_processes()
            await asyncio.sleep(5)
            await orchestrator.start_all_processes()
    elif args.command == "status":
        for name, status in orchestrator.processes.items():
            health = "HEALTHY" if status.state == ProcessState.RUNNING else status.state.value.upper()
            uptime = time.time() - status.start_time if status.start_time else 0
            print(f"{name:20} | {health:10} | PID: {status.pid or 'N/A':8} | Uptime: {uptime:6.0f}s | Restarts: {status.restart_count}")

if __name__ == "__main__":
    asyncio.run(main()) 