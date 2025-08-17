#!/usr/bin/env python3
"""
Project setup script for Distributed RL System
Creates the complete project structure for Phase 1 implementation
"""

import os
from pathlib import Path
import sys

def create_directory_structure():
    """Create the complete project directory structure"""
    
    directories = [
        # Core application directories
        "src/coordinator",
        "src/gpu_agent", 
        "src/memory",
        "src/monitoring",
        "src/utils",
        "src/config",
        
        # Phase 1 specific directories
        "src/coordinator/job_queue",
        "src/coordinator/batch_splitter", 
        "src/coordinator/load_balancer",
        "src/coordinator/assignment_engine",
        
        # Testing directories
        "tests/unit",
        "tests/integration",
        "tests/performance",
        "tests/fixtures",
        
        # Configuration and data
        "config",
        "data/episodic_memory",
        "data/performance_history",
        "data/job_results",
        
        # Logging and monitoring
        "logs",
        "metrics",
        
        # Scripts and utilities
        "scripts/phase1",
        "scripts/deployment",
        "scripts/monitoring",
        
        # Documentation
        "docs/api",
        "docs/architecture", 
        "docs/deployment",
        "docs/troubleshooting",
        
        # Dashboard (for later phases)
        "dashboard/src",
        "dashboard/public",
        
        # Deployment files
        "deployment/local",
        "deployment/production"
    ]
    
    print("🏗️  Creating project directory structure...")
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        
        # Create __init__.py files for Python packages
        if directory.startswith("src/") or directory.startswith("tests/"):
            init_file = Path(directory) / "__init__.py"
            if not init_file.exists():
                init_file.touch()
    
    print(f"✅ Created {len(directories)} directories")

def create_core_files():
    """Create essential configuration and setup files"""
    
    files_to_create = {
        # Python environment
        "requirements.txt": """# Core dependencies for Phase 1
# Web framework and async support
fastapi>=0.104.0
uvicorn>=0.24.0
aiohttp>=3.9.0
asyncio-mqtt>=0.14.0

# Data handling
pydantic>=2.4.0
pydantic-settings>=2.0.0
redis>=5.0.0
pandas>=2.0.0
numpy>=1.24.0

# GPU and system monitoring
pynvml>=11.5.0
psutil>=5.9.0
torch>=2.0.0

# Logging and utilities
loguru>=0.7.0
python-dotenv>=1.0.0
typer>=0.9.0
rich>=13.0.0

# Testing
pytest>=7.4.0
pytest-asyncio>=0.21.0
pytest-cov>=4.1.0
httpx>=0.25.0

# Development tools
black>=23.0.0
flake8>=6.0.0
mypy>=1.5.0
""",

        # Environment configuration
        ".env.example": """# Distributed RL System Configuration

# System Configuration
NUM_GPUS=8
BASE_GPU_PORT=8096
COORDINATOR_PORT=8090

# Redis Configuration
REDIS_HOST=localhost
REDIS_PORT=6379
REDIS_DB=0
REDIS_PASSWORD=

# Job Configuration
MAX_CONCURRENT_JOBS=3
DEFAULT_TARGET_SCORE=0.85
DEFAULT_MAX_EPISODES=10
DEFAULT_MAX_ROUNDS=12

# Performance Configuration
BATCH_DISTRIBUTION_STRATEGY=performance_based
MEMORY_SYNC_INTERVAL=300
HEALTH_CHECK_INTERVAL=10

# GPU Configuration  
GPU_MEMORY_LIMIT_GB=20.0
GPU_TEMPERATURE_LIMIT=85.0
GPU_UTILIZATION_TARGET=90.0

# Logging Configuration
LOG_LEVEL=INFO
LOG_DIR=./logs
ENABLE_FILE_LOGGING=true
ENABLE_PERFORMANCE_LOGGING=true

# Monitoring Configuration
METRICS_RETENTION_HOURS=24
ENABLE_PROMETHEUS=false
PROMETHEUS_PORT=9090
""",

        # Git configuration
        ".gitignore": """# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
share/python-wheels/
*.egg-info/
.installed.cfg
*.egg
MANIFEST

# Environment
.env
.venv
env/
venv/
ENV/
env.bak/
venv.bak/

# IDE
.vscode/
.idea/
*.swp
*.swo
*~

# Project specific
logs/
data/
metrics/
checkpoints/
*.log
*.pid

# Temporary files
tmp/
temp/
.cache/
.pytest_cache/

# OS
.DS_Store
Thumbs.db
""",

        # Main entry point
        "main.py": """#!/usr/bin/env python3
\"\"\"
Main entry point for the Distributed RL System
Supports different modes: coordinator, gpu-agent, monitor
\"\"\"

import asyncio
import typer
from rich.console import Console
from pathlib import Path
import sys

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

from config.settings import settings
from utils.logging_config import setup_logging

app = typer.Typer(help="Distributed RL System - Phase 1")
console = Console()

@app.command()
def coordinator(
    port: int = typer.Option(8090, help="Port for coordinator server"),
    redis_host: str = typer.Option("localhost", help="Redis host"),
    num_gpus: int = typer.Option(8, help="Number of GPUs to manage")
):
    \"\"\"Start the job distribution coordinator\"\"\"
    console.print(f"🚀 Starting Coordinator on port {port}", style="bold green")
    console.print(f"Managing {num_gpus} GPUs with Redis at {redis_host}")
    
    # Import here to avoid circular imports
    from coordinator.main import CoordinatorMain
    
    coordinator_main = CoordinatorMain(
        port=port,
        redis_host=redis_host, 
        num_gpus=num_gpus
    )
    
    asyncio.run(coordinator_main.start())

@app.command()
def gpu_agent(
    gpu_id: int = typer.Argument(..., help="GPU ID (0-7)"),
    coordinator_url: str = typer.Option("http://localhost:8090", help="Coordinator URL"),
    port: int = typer.Option(None, help="Override default port (8096 + gpu_id)")
):
    \"\"\"Start a GPU RL agent\"\"\"
    if port is None:
        port = 8096 + gpu_id
        
    console.print(f"🔥 Starting GPU Agent {gpu_id} on port {port}", style="bold blue")
    console.print(f"Connecting to coordinator: {coordinator_url}")
    
    # Import here to avoid circular imports
    from gpu_agent.main import GPUAgentMain
    
    agent_main = GPUAgentMain(
        gpu_id=gpu_id,
        port=port,
        coordinator_url=coordinator_url
    )
    
    asyncio.run(agent_main.start())

@app.command()
def monitor(
    coordinator_url: str = typer.Option("http://localhost:8090", help="Coordinator URL"),
    update_interval: int = typer.Option(5, help="Update interval in seconds")
):
    \"\"\"Start the system monitor\"\"\"
    console.print("📊 Starting System Monitor", style="bold yellow")
    
    # Import here to avoid circular imports  
    from monitoring.main import MonitorMain
    
    monitor_main = MonitorMain(
        coordinator_url=coordinator_url,
        update_interval=update_interval
    )
    
    asyncio.run(monitor_main.start())

@app.command()
def setup():
    \"\"\"Setup the system for first time use\"\"\"
    console.print("🛠️  Setting up Distributed RL System...", style="bold cyan")
    
    # Check if .env exists
    env_file = Path(".env")
    if not env_file.exists():
        console.print("Creating .env file from template...")
        example_env = Path(".env.example")
        if example_env.exists():
            env_file.write_text(example_env.read_text())
            console.print("✅ Created .env file")
        else:
            console.print("❌ .env.example not found", style="red")
    
    # Setup logging directory
    setup_logging()
    
    console.print("✅ Setup complete!", style="bold green")
    console.print("\\nNext steps:")
    console.print("1. Edit .env file with your configuration")
    console.print("2. Start coordinator: python main.py coordinator")
    console.print("3. Start GPU agents: python main.py gpu-agent <gpu_id>")

if __name__ == "__main__":
    app()
""",

        # Project README
        "README.md": """# Distributed RL System - Phase 1

A distributed reinforcement learning system for prompt optimization across multiple GPUs with intelligent job distribution and load balancing.

## 🎯 Phase 1 Features

- **Intelligent Job Distribution**: Automatically split and distribute prompt batches across 8 GPUs
- **Dynamic Load Balancing**: Real-time GPU assessment and optimal workload assignment  
- **Episodic Memory Integration**: Load relevant historical context for each GPU
- **Failure Recovery**: Automatic detection and redistribution of failed GPU work
- **Real-time Monitoring**: Track system performance and GPU health

## 🚀 Quick Start

1. **Setup Environment**:
```bash
python setup_project.py  # Create directory structure
python main.py setup     # Initialize configuration
```

2. **Start System**:
```bash
# Terminal 1: Start coordinator
python main.py coordinator

# Terminal 2-9: Start GPU agents  
python main.py gpu-agent 0
python main.py gpu-agent 1
# ... repeat for GPUs 2-7

# Terminal 10: Monitor system
python main.py monitor
```

3. **Submit a Job**:
```python
import requests

response = requests.post("http://localhost:8090/api/jobs/submit", json={
    "prompts": ["a red sports car", "a blue house", "a green tree"],
    "target_score": 0.85,
    "max_episodes": 5
})

job_id = response.json()["job_id"]
print(f"Job submitted: {job_id}")
```

## 📁 Project Structure

```
src/
├── coordinator/          # Job distribution coordinator
│   ├── job_queue/       # Job queuing and prioritization
│   ├── batch_splitter/  # Intelligent prompt batch creation
│   ├── load_balancer/   # GPU performance assessment
│   └── assignment_engine/ # Optimal batch assignment
├── gpu_agent/           # Individual GPU RL agents
├── memory/              # Episodic memory management
├── monitoring/          # System health and performance tracking
└── utils/               # Shared utilities and configuration

tests/                   # Comprehensive test suite
config/                  # Configuration files
scripts/                 # Deployment and utility scripts
docs/                    # Documentation
```

## 🔧 Configuration

Edit `.env` file to customize:
- Number of GPUs and port configuration
- Redis settings for global memory
- Performance thresholds and optimization parameters
- Logging and monitoring options

## 📊 Monitoring

The system provides real-time monitoring of:
- GPU utilization and health metrics
- Job queue status and processing rates
- Performance statistics and optimization effectiveness
- Error rates and recovery metrics

## 🧪 Testing

```bash
# Run all tests
pytest

# Run specific test categories
pytest tests/unit/          # Unit tests
pytest tests/integration/   # Integration tests  
pytest tests/performance/   # Performance benchmarks
```

## 📖 Documentation

- [Architecture Overview](docs/architecture/)
- [API Reference](docs/api/)
- [Deployment Guide](docs/deployment/)
- [Troubleshooting](docs/troubleshooting/)

## 🛠️ Development

This is Phase 1 of a multi-phase implementation:
- **Phase 1**: Job Distribution (Current)
- **Phase 2**: Parallel RL Execution
- **Phase 3**: Results Aggregation
- **Phase 4**: Dashboard & Monitoring
- **Phase 5**: Production Optimization
"""
    }
    
    print("📄 Creating core project files...")
    
    for filename, content in files_to_create.items():
        file_path = Path(filename)
        file_path.write_text(content)
        print(f"✅ Created {filename}")

def create_phase1_scripts():
    """Create the main Phase 1 implementation scripts"""
    
    # Configuration management
    config_files = {
        "src/config/__init__.py": "",
        
        "src/config/settings.py": """\"\"\"
Centralized configuration management using Pydantic settings
\"\"\"

from pydantic_settings import BaseSettings
from typing import Optional, List, Dict, Any
from pathlib import Path

class Settings(BaseSettings):
    # System Configuration
    num_gpus: int = 8
    base_gpu_port: int = 8096
    coordinator_port: int = 8090
    
    # Redis Configuration for Global Memory
    redis_host: str = "localhost"
    redis_port: int = 6379
    redis_db: int = 0
    redis_password: Optional[str] = None
    
    # Job Configuration
    max_concurrent_jobs: int = 3
    default_target_score: float = 0.85
    default_max_episodes: int = 10
    default_max_rounds: int = 12
    default_improvement_threshold: float = 0.03
    
    # Performance Configuration
    batch_distribution_strategy: str = "performance_based"  # or "equal"
    memory_sync_interval: int = 300  # seconds
    health_check_interval: int = 10  # seconds
    
    # GPU Configuration
    gpu_memory_limit_gb: float = 20.0
    gpu_temperature_limit: float = 85.0
    gpu_utilization_target: float = 90.0
    
    # Failure Recovery
    max_gpu_failures: int = 3
    failure_recovery_timeout: int = 60  # seconds
    
    # Monitoring
    metrics_retention_hours: int = 24
    
    # Logging
    log_level: str = "INFO"
    log_dir: str = "./logs"
    enable_file_logging: bool = True
    enable_performance_logging: bool = True
    
    # Paths
    data_dir: str = "./data"
    checkpoint_dir: str = "./checkpoints"
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"

# Global settings instance
settings = Settings()
""",

        "src/config/gpu_config.py": """\"\"\"
GPU-specific configuration and capability definitions
\"\"\"

from dataclasses import dataclass
from typing import Dict, List, Optional
from .settings import settings

@dataclass
class GPUCapability:
    \"\"\"Defines the capabilities and specialization of a specific GPU\"\"\"
    gpu_id: int
    max_batch_size: int
    memory_gb: float
    compute_capability: str  # e.g., "8.6" for RTX 3090
    preferred_complexity: str  # simple, medium, complex
    specializations: List[str]  # e.g., ["vehicles", "architecture", "characters"]
    base_performance_score: float = 1.0
    reliability_score: float = 1.0

# Default GPU configurations for A6000 GPUs
DEFAULT_GPU_CONFIGS = {
    0: GPUCapability(0, 20, 48.0, "8.6", "complex", ["vehicles", "architecture"], 1.0, 1.0),
    1: GPUCapability(1, 20, 48.0, "8.6", "complex", ["characters", "animals"], 1.0, 1.0),
    2: GPUCapability(2, 20, 48.0, "8.6", "medium", ["objects", "furniture"], 1.0, 1.0),
    3: GPUCapability(3, 20, 48.0, "8.6", "medium", ["nature", "landscapes"], 1.0, 1.0),
    4: GPUCapability(4, 20, 48.0, "8.6", "simple", ["abstract", "patterns"], 1.0, 1.0),
    5: GPUCapability(5, 20, 48.0, "8.6", "simple", ["textures", "materials"], 1.0, 1.0),
    6: GPUCapability(6, 20, 48.0, "8.6", "complex", ["artistic", "creative"], 1.0, 1.0),
    7: GPUCapability(7, 20, 48.0, "8.6", "medium", ["technical", "mechanical"], 1.0, 1.0),
}

def get_gpu_config(gpu_id: int) -> GPUCapability:
    \"\"\"Get configuration for a specific GPU\"\"\"
    return DEFAULT_GPU_CONFIGS.get(gpu_id, GPUCapability(
        gpu_id=gpu_id,
        max_batch_size=15,
        memory_gb=24.0,
        compute_capability="8.0",
        preferred_complexity="medium",
        specializations=["general"],
        base_performance_score=1.0,
        reliability_score=1.0
    ))

def get_all_gpu_configs() -> Dict[int, GPUCapability]:
    \"\"\"Get configurations for all GPUs\"\"\"
    return {
        gpu_id: get_gpu_config(gpu_id) 
        for gpu_id in range(settings.num_gpus)
    }
"""
    }
    
    # Utility functions
    util_files = {
        "src/utils/__init__.py": "",
        
        "src/utils/logging_config.py": """\"\"\"
Centralized logging configuration using loguru
\"\"\"

import sys
from pathlib import Path
from datetime import datetime
from loguru import logger
from .settings import settings

def setup_logging(component_name: str = "system") -> None:
    \"\"\"Setup logging configuration for the system\"\"\"
    
    # Create logs directory
    log_dir = Path(settings.log_dir)
    log_dir.mkdir(exist_ok=True)
    
    # Remove default handler
    logger.remove()
    
    # Console handler with colors
    logger.add(
        sys.stdout,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{extra[component]}</cyan> | <level>{message}</level>",
        level=settings.log_level,
        colorize=True,
        filter=lambda record: record["extra"].get("component", "system") == component_name
    )
    
    if settings.enable_file_logging:
        # Main log file
        log_file = log_dir / f"{component_name}_{datetime.now():%Y%m%d_%H%M%S}.log"
        logger.add(
            log_file,
            format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {extra[component]} | {message}",
            level=settings.log_level,
            rotation="100 MB",
            retention="7 days",
            compression="zip",
            filter=lambda record: record["extra"].get("component", "system") == component_name
        )
        
        # Error-only log file
        error_file = log_dir / f"{component_name}_errors_{datetime.now():%Y%m%d}.log"
        logger.add(
            error_file,
            format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {extra[component]} | {message}\\n{exception}",
            level="ERROR",
            rotation="50 MB",
            retention="30 days",
            backtrace=True,
            diagnose=True,
            filter=lambda record: record["extra"].get("component", "system") == component_name
        )

def get_logger(component_name: str):
    \"\"\"Get a logger instance for a specific component\"\"\"
    return logger.bind(component=component_name)
""",

        "src/utils/gpu_utils.py": """\"\"\"
GPU utility functions for monitoring and management
\"\"\"

import subprocess
import json
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime

try:
    import pynvml
    PYNVML_AVAILABLE = True
except ImportError:
    PYNVML_AVAILABLE = False

@dataclass
class GPUMetrics:
    \"\"\"GPU hardware and performance metrics\"\"\"
    gpu_id: int
    memory_used_gb: float
    memory_total_gb: float
    memory_free_gb: float
    utilization_gpu: int  # 0-100
    utilization_memory: int  # 0-100
    temperature_celsius: int
    power_watts: float
    process_count: int
    timestamp: datetime

class GPUMonitor:
    \"\"\"Monitor GPU hardware metrics\"\"\"
    
    def __init__(self):
        if PYNVML_AVAILABLE:
            try:
                pynvml.nvmlInit()
                self.pynvml_available = True
            except Exception:
                self.pynvml_available = False
        else:
            self.pynvml_available = False
    
    def get_gpu_metrics(self, gpu_id: int) -> Optional[GPUMetrics]:
        \"\"\"Get comprehensive metrics for a specific GPU\"\"\"
        
        if self.pynvml_available:
            return self._get_metrics_pynvml(gpu_id)
        else:
            return self._get_metrics_nvidia_smi(gpu_id)
    
    def _get_metrics_pynvml(self, gpu_id: int) -> Optional[GPUMetrics]:
        \"\"\"Get metrics using pynvml (preferred method)\"\"\"
        try:
            handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_id)
            
            # Memory info
            mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            memory_used_gb = mem_info.used / (1024**3)
            memory_total_gb = mem_info.total / (1024**3)
            memory_free_gb = mem_info.free / (1024**3)
            
            # Utilization
            util = pynvml.nvmlDeviceGetUtilizationRates(handle)
            
            # Temperature
            temperature = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
            
            # Power
            power_watts = pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0
            
            # Running processes
            processes = pynvml.nvmlDeviceGetComputeRunningProcesses(handle)
            process_count = len(processes)
            
            return GPUMetrics(
                gpu_id=gpu_id,
                memory_used_gb=memory_used_gb,
                memory_total_gb=memory_total_gb,
                memory_free_gb=memory_free_gb,
                utilization_gpu=util.gpu,
                utilization_memory=util.memory,
                temperature_celsius=temperature,
                power_watts=power_watts,
                process_count=process_count,
                timestamp=datetime.now()
            )
            
        except Exception as e:
            print(f"Error getting metrics for GPU {gpu_id}: {e}")
            return None
    
    def _get_metrics_nvidia_smi(self, gpu_id: int) -> Optional[GPUMetrics]:
        \"\"\"Fallback method using nvidia-smi command\"\"\"
        try:
            cmd = [
                "nvidia-smi", 
                "--query-gpu=index,memory.used,memory.total,memory.free,utilization.gpu,utilization.memory,temperature.gpu,power.draw",
                "--format=csv,noheader,nounits",
                f"--id={gpu_id}"
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
            
            if result.returncode != 0:
                return None
            
            values = result.stdout.strip().split(", ")
            
            return GPUMetrics(
                gpu_id=int(values[0]),
                memory_used_gb=float(values[1]) / 1024,
                memory_total_gb=float(values[2]) / 1024, 
                memory_free_gb=float(values[3]) / 1024,
                utilization_gpu=int(values[4]),
                utilization_memory=int(values[5]),
                temperature_celsius=int(values[6]),
                power_watts=float(values[7]),
                process_count=0,  # Not available via nvidia-smi
                timestamp=datetime.now()
            )
            
        except Exception as e:
            print(f"Error getting metrics for GPU {gpu_id} via nvidia-smi: {e}")
            return None
    
    def get_all_gpu_metrics(self, num_gpus: int = 8) -> Dict[int, GPUMetrics]:
        \"\"\"Get metrics for all GPUs\"\"\"
        metrics = {}
        
        for gpu_id in range(num_gpus):
            gpu_metrics = self.get_gpu_metrics(gpu_id)
            if gpu_metrics:
                metrics[gpu_id] = gpu_metrics
        
        return metrics
    
    def check_gpu_health(self, gpu_metrics: GPUMetrics) -> Tuple[bool, List[str]]:
        \"\"\"Check if GPU is healthy based on metrics\"\"\"
        issues = []
        
        # Temperature check
        if gpu_metrics.temperature_celsius > 85:
            issues.append(f"High temperature: {gpu_metrics.temperature_celsius}°C")
        
        # Memory usage check
        memory_usage_pct = (gpu_metrics.memory_used_gb / gpu_metrics.memory_total_gb) * 100
        if memory_usage_pct > 95:
            issues.append(f"High memory usage: {memory_usage_pct:.1f}%")
        
        # Power usage check (assuming max 300W for A6000)
        if gpu_metrics.power_watts > 280:
            issues.append(f"High power usage: {gpu_metrics.power_watts:.1f}W")
        
        return len(issues) == 0, issues

# Global GPU monitor instance
gpu_monitor = GPUMonitor()
"""
    }
    
    print("🔧 Creating Phase 1 implementation scripts...")
    
    # Create all files
    all_files = {**config_files, **util_files}
    
    for filename, content in all_files.items():
        file_path = Path(filename)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_text(content)
        print(f"✅ Created {filename}")

def main():
    """Main setup function"""
    print("🚀 Setting up Distributed RL System - Phase 1")
    print("=" * 60)
    
    try:
        # Create directory structure
        create_directory_structure()
        print()
        
        # Create core files
        create_core_files() 
        print()
        
        # Create Phase 1 scripts
        create_phase1_scripts()
        print()
        
        print("✅ Project setup complete!")
        print("\n🎯 Next Steps:")
        print("1. Install dependencies: pip install -r requirements.txt")
        print("2. Copy .env.example to .env and configure")
        print("3. Start development: python main.py setup")
        print("\n📖 See README.md for detailed instructions")
        
    except Exception as e:
        print(f"❌ Setup failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()




