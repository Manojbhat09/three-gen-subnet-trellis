#!/usr/bin/env python3
"""
TRELLIS Mining Setup Validator
==============================
Automatically checks many items from the pre-launch checklist
"""

import os
import sys
import json
import subprocess
import requests
import importlib
from pathlib import Path
from typing import Dict, List, Tuple

class TrellisSetupValidator:
    def __init__(self):
        self.results = {}
        self.errors = []
        self.warnings = []
        
    def print_header(self, title: str):
        print(f"\n{'='*60}")
        print(f"🔍 {title}")
        print(f"{'='*60}")
    
    def print_result(self, item: str, status: bool, details: str = ""):
        icon = "✅" if status else "❌"
        print(f"{icon} {item}")
        if details:
            print(f"   {details}")
        self.results[item] = status
    
    def print_warning(self, item: str, details: str):
        print(f"⚠️  {item}: {details}")
        self.warnings.append(f"{item}: {details}")
    
    def print_error(self, item: str, details: str):
        print(f"❌ {item}: {details}")
        self.errors.append(f"{item}: {details}")
    
    def check_environment_variables(self):
        """Check required environment variables"""
        self.print_header("Environment Variables")
        
        # Check HF Token
        hf_token = os.getenv('HF_TOKEN')
        if hf_token:
            self.print_result("HuggingFace Token", True, f"Set (length: {len(hf_token)})")
        else:
            self.print_result("HuggingFace Token", False, "Not set")
        
        # Check WANDB API Key
        wandb_key = os.getenv('WANDB_API_KEY')
        if wandb_key:
            self.print_result("Weights & Biases API Key", True, f"Set (length: {len(wandb_key)})")
        else:
            self.print_result("Weights & Biases API Key", False, "Not set")
        
        # Check CUDA config
        cuda_config = os.getenv('CUBLAS_WORKSPACE_CONFIG')
        if cuda_config:
            self.print_result("CUDA Determinism Config", True, f"Set to: {cuda_config}")
        else:
            self.print_result("CUDA Determinism Config", False, "Not set")
    
    def check_python_environment(self):
        """Check Python environment and dependencies"""
        self.print_header("Python Environment")
        
        # Python version
        python_version = sys.version_info
        if python_version.major >= 3 and python_version.minor >= 8:
            self.print_result("Python Version", True, f"Python {python_version.major}.{python_version.minor}.{python_version.micro}")
        else:
            self.print_result("Python Version", False, f"Python {python_version.major}.{python_version.minor}.{python_version.micro} (need 3.8+)")
        
        # Check required packages
        required_packages = [
            'torch', 'transformers', 'diffusers', 'accelerate',
            'bittensor', 'wandb', 'requests', 'PIL', 'numpy', 'pandas'
        ]
        
        for package in required_packages:
            try:
                if package == 'PIL':
                    importlib.import_module('PIL')
                    self.print_result(f"Package: {package}", True)
                else:
                    importlib.import_module(package)
                    self.print_result(f"Package: {package}", True)
            except ImportError:
                self.print_result(f"Package: {package}", False, "Not installed")
    
    def check_storage_and_cache(self):
        """Check storage directories and cache"""
        self.print_header("Storage & Cache")
        
        # Check cache directory
        cache_dir = Path("/home/mbhat/.cache_god")
        if cache_dir.exists():
            try:
                cache_size = sum(f.stat().st_size for f in cache_dir.rglob('*') if f.is_file())
                cache_size_gb = cache_size / (1024**3)
                self.print_result("Cache Directory", True, f"Exists ({cache_size_gb:.2f} GB)")
            except Exception as e:
                self.print_result("Cache Directory", True, f"Exists (size check failed: {e})")
        else:
            self.print_result("Cache Directory", False, "Does not exist")
        
        # Check checkpoints directory
        checkpoints_dir = Path("/home/mbhat/.checkpoints_god")
        if checkpoints_dir.exists():
            try:
                checkpoint_count = len(list(checkpoints_dir.rglob('*.safetensors'))) + len(list(checkpoints_dir.rglob('*.ckpt')))
                self.print_result("Checkpoints Directory", True, f"Exists ({checkpoint_count} model files)")
            except Exception as e:
                self.print_result("Checkpoints Directory", True, f"Exists (file count failed: {e})")
        else:
            self.print_result("Checkpoints Directory", False, "Does not exist")
        
        # Check output directory
        output_dir = Path("./trellis_mining_outputs_test")
        if output_dir.exists():
            self.print_result("Output Directory", True, "Exists")
        else:
            try:
                output_dir.mkdir(parents=True, exist_ok=True)
                self.print_result("Output Directory", True, "Created")
            except Exception as e:
                self.print_result("Output Directory", False, f"Failed to create: {e}")
        
        # Check database file
        db_file = Path("continuous_trellis_tasks_test.db")
        if db_file.exists():
            try:
                db_size = db_file.stat().st_size
                db_size_mb = db_size / (1024**2)
                self.print_result("Database File", True, f"Exists ({db_size_mb:.2f} MB)")
            except Exception as e:
                self.print_result("Database File", True, f"Exists (size check failed: {e})")
        else:
            self.print_result("Database File", False, "Does not exist")
    
    def check_hardware(self):
        """Check hardware configuration"""
        self.print_header("Hardware Configuration")
        
        # Check GPU
        try:
            import torch
            if torch.cuda.is_available():
                gpu_count = torch.cuda.device_count()
                gpu_name = torch.cuda.get_device_name(0)
                gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
                self.print_result("CUDA GPU", True, f"{gpu_count} GPU(s), {gpu_name}, {gpu_memory:.1f} GB")
            else:
                self.print_result("CUDA GPU", False, "No CUDA GPU available")
        except Exception as e:
            self.print_result("CUDA GPU", False, f"Check failed: {e}")
        
        # Check system memory
        try:
            import psutil
            memory = psutil.virtual_memory()
            memory_gb = memory.total / (1024**3)
            if memory_gb >= 16:
                self.print_result("System RAM", True, f"{memory_gb:.1f} GB")
            else:
                self.print_result("System RAM", False, f"{memory_gb:.1f} GB (recommended: 16GB+)")
        except ImportError:
            self.print_result("System RAM", True, "psutil not available, skipping check")
        except Exception as e:
            self.print_result("System RAM", False, f"Check failed: {e}")
        
        # Check disk space
        try:
            disk = psutil.disk_usage('.')
            disk_gb = disk.free / (1024**3)
            if disk_gb >= 10:
                self.print_result("Disk Space", True, f"{disk_gb:.1f} GB free")
            else:
                self.print_result("Disk Space", False, f"{disk_gb:.1f} GB free (recommended: 10GB+)")
        except Exception as e:
            self.print_result("Disk Space", False, f"Check failed: {e}")
    
    def check_trellis_server(self):
        """Check TRELLIS server status"""
        self.print_header("TRELLIS Server")
        
        try:
            response = requests.get("http://localhost:8096/status/", timeout=5)
            if response.status_code == 200:
                data = response.json()
                if data.get('ready', False):
                    self.print_result("Server Status", True, "Running and ready")
                else:
                    self.print_result("Server Status", False, "Running but not ready")
            else:
                self.print_result("Server Status", False, f"HTTP {response.status_code}")
        except requests.exceptions.ConnectionError:
            self.print_result("Server Status", False, "Connection refused - server not running")
        except Exception as e:
            self.print_result("Server Status", False, f"Check failed: {e}")
        
        # Check if server process is running
        try:
            result = subprocess.run(['pgrep', '-f', 'trellis_submit_server'], 
                                  capture_output=True, text=True)
            if result.returncode == 0:
                pids = result.stdout.strip().split('\n')
                self.print_result("Server Process", True, f"Running (PIDs: {', '.join(pids)})")
            else:
                self.print_result("Server Process", False, "No server process found")
        except Exception as e:
            self.print_result("Server Process", False, f"Check failed: {e}")
    
    def check_core_files(self):
        """Check if core files exist"""
        self.print_header("Core Files")
        
        core_files = [
            "continuous_trellis_orchestrator_lora_test_mod.py",
            "trellis_submit_server.py",
            "episodic_trellis_optimizer.py",
            "run_trellis_mining_test.sh"
        ]
        
        for file_path in core_files:
            if Path(file_path).exists():
                self.print_result(f"File: {file_path}", True)
            else:
                self.print_result(f"File: {file_path}", False, "Missing")
    
    def check_imports(self):
        """Test importing core modules"""
        self.print_header("Module Imports")
        
        try:
            # Test basic imports
            import episodic_trellis_optimizer
            self.print_result("episodic_trellis_optimizer", True)
        except Exception as e:
            self.print_result("episodic_trellis_optimizer", False, f"Import failed: {e}")
        
        # Check if we can find the orchestrator
        orchestrator_files = [
            "continuous_trellis_orchestrator_lora_test_mod.py",
            "continuous_trellis_orchestrator_lora.py",
            "continuous_trellis_orchestrator.py"
        ]
        
        found_orchestrator = False
        for file_path in orchestrator_files:
            if Path(file_path).exists():
                found_orchestrator = True
                self.print_result(f"Orchestrator: {file_path}", True)
                break
        
        if not found_orchestrator:
            self.print_result("Orchestrator", False, "No orchestrator file found")
    
    def run_basic_tests(self):
        """Run basic functionality tests"""
        self.print_header("Basic Functionality Tests")
        
        # Test episodic optimizer initialization
        try:
            from episodic_trellis_optimizer import EpisodicTrellisOptimizer
            optimizer = EpisodicTrellisOptimizer(num_episodes=1, target_score=0.8)
            self.print_result("Episodic Optimizer Init", True)
        except Exception as e:
            self.print_result("Episodic Optimizer Init", False, f"Failed: {e}")
        
        # Test database connection (if exists)
        db_file = Path("continuous_trellis_tasks_test.db")
        if db_file.exists():
            try:
                import sqlite3
                conn = sqlite3.connect(db_file)
                cursor = conn.cursor()
                cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
                tables = cursor.fetchall()
                conn.close()
                self.print_result("Database Connection", True, f"Connected, {len(tables)} tables")
            except Exception as e:
                self.print_result("Database Connection", False, f"Failed: {e}")
        else:
            self.print_result("Database Connection", True, "No database file to test")
    
    def generate_summary(self):
        """Generate summary report"""
        self.print_header("Summary Report")
        
        total_checks = len(self.results)
        passed_checks = sum(self.results.values())
        failed_checks = total_checks - passed_checks
        
        print(f"📊 Total Checks: {total_checks}")
        print(f"✅ Passed: {passed_checks}")
        print(f"❌ Failed: {failed_checks}")
        print(f"⚠️  Warnings: {len(self.warnings)}")
        print(f"🚨 Errors: {len(self.errors)}")
        
        if failed_checks == 0:
            print("\n🎉 All checks passed! Your TRELLIS mining setup is ready.")
        else:
            print(f"\n⚠️  {failed_checks} check(s) failed. Please review and fix before starting mining.")
        
        if self.warnings:
            print(f"\n⚠️  Warnings:")
            for warning in self.warnings:
                print(f"   - {warning}")
        
        if self.errors:
            print(f"\n❌ Errors:")
            for error in self.errors:
                print(f"   - {error}")
        
        # Save results to file
        try:
            results_file = Path("trellis_setup_check_results.json")
            with open(results_file, 'w') as f:
                json.dump({
                    'timestamp': str(Path().cwd()),
                    'results': self.results,
                    'warnings': self.warnings,
                    'errors': self.errors,
                    'summary': {
                        'total': total_checks,
                        'passed': passed_checks,
                        'failed': failed_checks
                    }
                }, f, indent=2)
            print(f"\n💾 Results saved to: {results_file}")
        except Exception as e:
            print(f"\n⚠️  Failed to save results: {e}")
    
    def run_all_checks(self):
        """Run all validation checks"""
        print("🚀 TRELLIS Mining Setup Validator")
        print("=" * 60)
        
        self.check_environment_variables()
        self.check_python_environment()
        self.check_storage_and_cache()
        self.check_hardware()
        self.check_trellis_server()
        self.check_core_files()
        self.check_imports()
        self.run_basic_tests()
        
        self.generate_summary()

def main():
    """Main function"""
    validator = TrellisSetupValidator()
    
    try:
        validator.run_all_checks()
    except KeyboardInterrupt:
        print("\n⚠️  Validation interrupted by user")
    except Exception as e:
        print(f"\n❌ Validation failed with error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
