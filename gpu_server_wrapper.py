#!/usr/bin/env python3
"""
GPU Server Wrapper - Subnet 17 (404-GEN)
Purpose: Start TRELLIS servers on all 8 GPUs and test parallel generation/validation

Features:
- Starts TRELLIS server on each GPU with unique port
- Primes all GPUs simultaneously with parallel generation
- Tests validation across all GPUs in parallel
- Comprehensive response analysis and summarization
- GPU health monitoring and status reporting
"""

import os
import sys
import time
import json
import asyncio
import requests
import subprocess
import threading
import signal
import argparse
import logging
from pathlib import Path
from typing import List, Dict, Optional, Any, Tuple
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, as_completed
import base64
import traceback

# Import components from the orchestrator
try:
    from continuous_trellis_orchestrator_lora import TaskRecord, ValidatorState
    ORCHESTRATOR_IMPORTS_AVAILABLE = True
    print("✅ Successfully imported orchestrator components")
except ImportError as e:
    print(f"⚠️ Warning: Could not import orchestrator components: {e}")
    print("   Some features may be limited")
    ORCHESTRATOR_IMPORTS_AVAILABLE = False

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('gpu_server_wrapper.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

@dataclass
class GPUServer:
    """Represents a GPU server instance"""
    gpu_id: int
    port: int
    process: Optional[subprocess.Popen] = None
    status: str = "stopped"
    last_health_check: Optional[float] = None
    generation_count: int = 0
    validation_count: int = 0
    last_response_time: Optional[float] = None
    error_count: int = 0
    last_generation_time: float = 0.0
    last_validation_time: float = 0.0
    last_ply_size: int = 0
    last_compression: float = 0.0
    
    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.port}"
    
    @property
    def health_url(self) -> str:
        return f"{self.url}/health/"
    
    @property
    def generate_url(self) -> str:
        return f"{self.url}/generate/"
    
    @property
    def validate_url(self) -> str:
        return f"{self.url}/validate/"
    
    @property
    def status_url(self) -> str:
        return f"{self.url}/status/"

class GPUServerManager:
    """Manages multiple GPU servers"""
    
    def __init__(self, num_gpus: int = 8, base_port: int = 8096, 
                 server_script: str = "trellis_subnit_server_mix_lora_flash.py",
                 output_dir: str = "./gpu_server_outputs"):
        self.num_gpus = num_gpus
        self.base_port = base_port
        self.server_script = server_script
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Initialize GPU servers
        self.gpu_servers: Dict[int, GPUServer] = {}
        for gpu_id in range(num_gpus):
            port = base_port + gpu_id
            self.gpu_servers[gpu_id] = GPUServer(gpu_id=gpu_id, port=port)
        
        # Server management
        self.running = False
        self.start_time = time.time()
        
        # Test configuration
        self.test_prompts = [
            "a pink bicycle with chrome wheels",
            "a blue ceramic vase with red trim",
            "a wooden table with four chairs",
            "a silver laptop on a desk",
            "a red sports car in a garage",
            "a green plant in a pot",
            "a black coffee mug on a saucer",
            "a white cloud in a blue sky"
        ]
        
        # Statistics
        self.stats = {
            'servers_started': 0,
            'servers_already_loaded': 0,  # New counter for already loaded GPUs
            'servers_failed': 0,
            'total_generations': 0,
            'successful_generations': 0,
            'failed_generations': 0,
            'total_validations': 0,
            'successful_validations': 0,
            'failed_validations': 0,
            'parallel_generation_time': 0.0,
            'parallel_validation_time': 0.0,
            'gpu_health_checks': 0,
            'gpu_errors': 0
        }
        
        logger.info(f"🎯 GPU Server Manager initialized for {num_gpus} GPUs")
        logger.info(f"   Base port: {base_port}")
        logger.info(f"   Server script: {server_script}")
        logger.info(f"   Output directory: {output_dir}")
    
    def check_gpu_already_loaded(self, gpu_id: int) -> bool:
        """Check if GPU is already loaded and serving requests"""
        gpu_server = self.gpu_servers[gpu_id]
        
        try:
            # Try to connect to the health endpoint
            response = requests.get(gpu_server.health_url, timeout=5)
            if response.status_code == 200:
                logger.info(f"✅ GPU {gpu_id} (port {gpu_server.port}) is already loaded and serving")
                gpu_server.status = "already_loaded"
                gpu_server.last_health_check = time.time()
                return True
            else:
                logger.debug(f"GPU {gpu_id} health check returned HTTP {response.status_code}")
                return False
        except requests.exceptions.ConnectionError:
            logger.debug(f"GPU {gpu_id} (port {gpu_server.port}) is not responding - needs loading")
            return False
        except Exception as e:
            logger.debug(f"GPU {gpu_id} health check exception: {e}")
            return False
    
    def start_server_on_gpu(self, gpu_id: int) -> bool:
        """Start TRELLIS server on a specific GPU"""
        gpu_server = self.gpu_servers[gpu_id]
        
        # First check if GPU is already loaded
        if self.check_gpu_already_loaded(gpu_id):
            logger.info(f"⏭️ GPU {gpu_id} (port {gpu_server.port}) already loaded, skipping startup")
            self.stats['servers_already_loaded'] = self.stats.get('servers_already_loaded', 0) + 1
            return True
        
        try:
            logger.info(f"🚀 Starting server on GPU {gpu_id} (port {gpu_server.port})")
            
            # Set environment variables
            env = os.environ.copy()
            env['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
            
            # Start the server process
            cmd = [
                sys.executable,  # Use current Python interpreter
                self.server_script,
                '--port', str(gpu_server.port),
                '--host', '127.0.0.1'
            ]
            
            # Start process
            process = subprocess.Popen(
                cmd,
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1,
                universal_newlines=True
            )
            
            gpu_server.process = process
            gpu_server.status = "starting"
            
            logger.info(f"   ✅ Server process started on GPU {gpu_id} (PID: {process.pid})")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to start server on GPU {gpu_id}: {e}")
            gpu_server.status = "failed"
            gpu_server.error_count += 1
            self.stats['gpu_errors'] += 1
            return False
    
    def start_all_servers(self) -> bool:
        """Start TRELLIS servers on all GPUs"""
        logger.info("🚀 Starting TRELLIS servers on all GPUs...")
        
        success_count = 0
        failed_count = 0
        already_loaded_count = 0
        
        # Start servers in parallel
        with ThreadPoolExecutor(max_workers=self.num_gpus) as executor:
            future_to_gpu = {
                executor.submit(self.start_server_on_gpu, gpu_id): gpu_id 
                for gpu_id in range(self.num_gpus)
            }
            
            for future in as_completed(future_to_gpu):
                gpu_id = future_to_gpu[future]
                try:
                    success = future.result()
                    if success:
                        success_count += 1
                        # Check if this was already loaded or newly started
                        if self.gpu_servers[gpu_id].status == "already_loaded":
                            already_loaded_count += 1
                        else:
                            self.stats['servers_started'] += 1
                    else:
                        failed_count += 1
                        self.stats['servers_failed'] += 1
                except Exception as e:
                    logger.error(f"❌ Exception starting server on GPU {gpu_id}: {e}")
                    failed_count += 1
                    self.stats['servers_failed'] += 1
        
        logger.info(f"✅ Server startup complete: {success_count} total, {already_loaded_count} already loaded, {self.stats['servers_started']} newly started, {failed_count} failed")
        
        if success_count > 0:
            # If we have newly started servers, wait for them to initialize
            if self.stats['servers_started'] > 0:
                logger.info("⏳ Waiting for newly started servers to initialize...")
                time.sleep(30)  # Give servers time to load models
            
            # Check server health
            self.check_all_servers_health()
            
            return True
        else:
            logger.error("❌ No servers available successfully")
            return False
    
    def check_server_health(self, gpu_id: int) -> bool:
        """Check health of a specific GPU server"""
        gpu_server = self.gpu_servers[gpu_id]
        
        try:
            response = requests.get(gpu_server.health_url, timeout=5)
            if response.status_code == 200:
                gpu_server.status = "healthy"
                gpu_server.last_health_check = time.time()
                self.stats['gpu_health_checks'] += 1
                return True
            else:
                gpu_server.status = "unhealthy"
                gpu_server.error_count += 1
                return False
        except Exception as e:
            gpu_server.status = "unreachable"
            gpu_server.error_count += 1
            self.stats['gpu_errors'] += 1
            return False
    
    def check_gpu_loading_status(self) -> Dict[int, str]:
        """Check loading status of all GPUs without starting servers"""
        logger.info("🔍 Checking GPU loading status...")
        
        loading_status = {}
        
        with ThreadPoolExecutor(max_workers=self.num_gpus) as executor:
            future_to_gpu = {
                executor.submit(self.check_gpu_already_loaded, gpu_id): gpu_id 
                for gpu_id in range(self.num_gpus)
            }
            
            for future in as_completed(future_to_gpu):
                gpu_id = future_to_gpu[future]
                try:
                    already_loaded = future.result()
                    if already_loaded:
                        loading_status[gpu_id] = "already_loaded"
                        self.gpu_servers[gpu_id].status = "already_loaded"
                        logger.info(f"   GPU {gpu_id} (port {self.gpu_servers[gpu_id].port}): 🔄 Already Loaded")
                    else:
                        loading_status[gpu_id] = "needs_loading"
                        logger.info(f"   GPU {gpu_id} (port {self.gpu_servers[gpu_id].port}): ⏳ Needs Loading")
                except Exception as e:
                    logger.error(f"❌ Exception checking loading status of GPU {gpu_id}: {e}")
                    loading_status[gpu_id] = "error"
        
        # Count already loaded GPUs
        already_loaded_count = sum(1 for status in loading_status.values() if status == "already_loaded")
        needs_loading_count = sum(1 for status in loading_status.values() if status == "needs_loading")
        
        logger.info(f"🔍 Loading status check complete: {already_loaded_count} already loaded, {needs_loading_count} need loading")
        
        return loading_status
    
    def check_all_servers_health(self) -> Dict[int, bool]:
        """Check health of all GPU servers"""
        logger.info("🏥 Checking health of all GPU servers...")
        
        health_results = {}
        
        with ThreadPoolExecutor(max_workers=self.num_gpus) as executor:
            future_to_gpu = {
                executor.submit(self.check_server_health, gpu_id): gpu_id 
                for gpu_id in range(self.num_gpus)
            }
            
            for future in as_completed(future_to_gpu):
                gpu_id = future_to_gpu[future]
                try:
                    healthy = future.result()
                    health_results[gpu_id] = healthy
                    
                    status = "✅ Healthy" if healthy else "❌ Unhealthy"
                    logger.info(f"   GPU {gpu_id} (port {self.gpu_servers[gpu_id].port}): {status}")
                    
                except Exception as e:
                    logger.error(f"❌ Exception checking health of GPU {gpu_id}: {e}")
                    health_results[gpu_id] = False
        
        # Count healthy servers
        healthy_count = sum(health_results.values())
        logger.info(f"🏥 Health check complete: {healthy_count}/{self.num_gpus} servers healthy")
        
        return health_results
    
    def get_server_status(self, gpu_id: int) -> Optional[Dict[str, Any]]:
        """Get detailed status of a specific GPU server"""
        gpu_server = self.gpu_servers[gpu_id]
        
        try:
            response = requests.get(gpu_server.status_url, timeout=10)
            if response.status_code == 200:
                return response.json()
            else:
                logger.warning(f"⚠️ Failed to get status from GPU {gpu_id}: HTTP {response.status_code}")
                return None
        except Exception as e:
            logger.warning(f"⚠️ Exception getting status from GPU {gpu_id}: {e}")
            return None
    
    def prime_single_gpu(self, gpu_id: int, prompt: str = None) -> Dict[str, Any]:
        """Prime a single GPU with a generation request"""
        gpu_server = self.gpu_servers[gpu_id]
        
        if prompt is None:
            prompt = self.test_prompts[gpu_id % len(self.test_prompts)]
        
        try:
            logger.info(f"🎨 Priming GPU {gpu_id} with prompt: '{prompt[:50]}...'")
            
            start_time = time.time()
            
            # Send generation request
            response = requests.post(
                gpu_server.generate_url,
                data={
                    'prompt': prompt,
                    'seed': 42,
                    'return_compressed': True
                },
                timeout=300  # 5 minutes timeout for generation
            )
            
            generation_time = time.time() - start_time
            
            if response.status_code == 200:
                # Success
                ply_data = response.content
                compression_ratio = response.headers.get('X-Compression-Ratio', 'unknown')
                
                gpu_server.generation_count += 1
                gpu_server.last_response_time = generation_time
                gpu_server.last_generation_time = generation_time
                gpu_server.last_ply_size = len(ply_data)
                # Handle compression ratio - remove % if present
                if compression_ratio != 'unknown' and compression_ratio:
                    try:
                        # Remove % sign if present and convert to float
                        compression_str = str(compression_ratio).replace('%', '')
                        gpu_server.last_compression = float(compression_str)
                    except (ValueError, TypeError):
                        gpu_server.last_compression = 0.0
                else:
                    gpu_server.last_compression = 0.0
                self.stats['successful_generations'] += 1
                self.stats['total_generations'] += 1
                
                result = {
                    'success': True,
                    'gpu_id': gpu_id,
                    'prompt': prompt,
                    'generation_time': generation_time,
                    'ply_size_bytes': len(ply_data),
                    'compression_ratio': compression_ratio,
                    'response_headers': dict(response.headers)
                }
                
                logger.info(f"   ✅ GPU {gpu_id} primed successfully in {generation_time:.2f}s")
                logger.info(f"      PLY size: {len(ply_data):,} bytes, Compression: {compression_ratio}")
                
                return result
            else:
                # Failure
                gpu_server.error_count += 1
                self.stats['failed_generations'] += 1
                self.stats['total_generations'] += 1
                
                result = {
                    'success': False,
                    'gpu_id': gpu_id,
                    'prompt': prompt,
                    'error': f"HTTP {response.status_code}",
                    'generation_time': generation_time
                }
                
                logger.error(f"   ❌ GPU {gpu_id} priming failed: HTTP {response.status_code}")
                return result
                
        except Exception as e:
            gpu_server.error_count += 1
            self.stats['failed_generations'] += 1
            self.stats['total_generations'] += 1
            
            result = {
                'success': False,
                'gpu_id': gpu_id,
                'prompt': prompt,
                'error': str(e),
                'generation_time': 0.0
            }
            
            logger.error(f"   ❌ GPU {gpu_id} priming exception: {e}")
            return result
    
    def prime_all_gpus_parallel(self) -> List[Dict[str, Any]]:
        """Prime all GPUs simultaneously with parallel generation"""
        logger.info("🚀 Priming all GPUs in parallel...")
        
        start_time = time.time()
        results = []
        
        # Prime all GPUs simultaneously
        with ThreadPoolExecutor(max_workers=self.num_gpus) as executor:
            future_to_gpu = {
                executor.submit(self.prime_single_gpu, gpu_id): gpu_id 
                for gpu_id in range(self.num_gpus)
            }
            
            for future in as_completed(future_to_gpu):
                gpu_id = future_to_gpu[future]
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    logger.error(f"❌ Exception priming GPU {gpu_id}: {e}")
                    results.append({
                        'success': False,
                        'gpu_id': gpu_id,
                        'error': str(e)
                    })
        
        parallel_time = time.time() - start_time
        self.stats['parallel_generation_time'] = parallel_time
        
        # Analyze results
        successful = [r for r in results if r.get('success', False)]
        failed = [r for r in results if not r.get('success', False)]
        
        logger.info(f"✅ Parallel priming complete in {parallel_time:.2f}s")
        logger.info(f"   Successful: {len(successful)}/{len(results)}")
        logger.info(f"   Failed: {len(failed)}/{len(results)}")
        
        if successful:
            avg_time = sum(r.get('generation_time', 0) for r in successful) / len(successful)
            logger.info(f"   Average generation time: {avg_time:.2f}s")
            
            # Rank GPUs by performance (fastest first)
            ranked_results = sorted(successful, key=lambda x: x.get('generation_time', float('inf')))
            
            logger.info("🏆 GPU Performance Ranking (Fastest to Slowest):")
            for i, result in enumerate(ranked_results):
                gpu_id = result['gpu_id']
                generation_time = result['generation_time']
                ply_size = result.get('ply_size_bytes', 0)
                compression = result.get('compression_ratio', 'unknown')
                
                medal = "🥇" if i == 0 else "🥈" if i == 1 else "🥉" if i == 2 else f"#{i+1}"
                logger.info(f"   {medal} GPU {gpu_id} (port {self.gpu_servers[gpu_id].port}): {generation_time:.2f}s")
                logger.info(f"      PLY: {ply_size:,} bytes, Compression: {compression}")
            
            # Store ranking in stats
            self.stats['fastest_gpu'] = ranked_results[0]['gpu_id']
            self.stats['slowest_gpu'] = ranked_results[-1]['gpu_id']
            self.stats['fastest_time'] = ranked_results[0]['generation_time']
            self.stats['slowest_time'] = ranked_results[-1]['generation_time']
            self.stats['performance_spread'] = ranked_results[-1]['generation_time'] - ranked_results[0]['generation_time']
        
        return results
    
    def test_validation_parallel(self) -> List[Dict[str, Any]]:
        """Test validation across all GPUs in parallel"""
        logger.info("📊 Testing validation across all GPUs in parallel...")
        
        start_time = time.time()
        results = []
        
        # Test validation on all GPUs simultaneously
        with ThreadPoolExecutor(max_workers=self.num_gpus) as executor:
            future_to_gpu = {
                executor.submit(self._test_single_gpu_validation, gpu_id): gpu_id 
                for gpu_id in range(self.num_gpus)
            }
            
            for future in as_completed(future_to_gpu):
                gpu_id = future_to_gpu[future]
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    logger.error(f"❌ Exception testing validation on GPU {gpu_id}: {e}")
                    results.append({
                        'success': False,
                        'gpu_id': gpu_id,
                        'error': str(e)
                    })
        
        parallel_time = time.time() - start_time
        self.stats['parallel_validation_time'] = parallel_time
        
        # Analyze results
        successful = [r for r in results if r.get('success', False)]
        failed = [r for r in results if not r.get('success', False)]
        
        logger.info(f"✅ Parallel validation testing complete in {parallel_time:.2f}s")
        logger.info(f"   Successful: {len(successful)}/{len(results)}")
        logger.info(f"   Failed: {len(failed)}/{len(results)}")
        
        if successful:
            # Rank GPUs by validation performance (fastest first)
            ranked_results = sorted(successful, key=lambda x: x.get('validation_time', float('inf')))
            
            logger.info("🏆 GPU Validation Performance Ranking (Fastest to Slowest):")
            for i, result in enumerate(ranked_results):
                gpu_id = result['gpu_id']
                validation_time = result['validation_time']
                total_time = result.get('total_time', 0)
                generation_time = result.get('generation_result', {}).get('generation_time', 0)
                validation_score = result.get('validation_score', 0.0)
                
                medal = "🥇" if i == 0 else "🥈" if i == 1 else "🥉" if i == 2 else f"#{i+1}"
                logger.info(f"   {medal} GPU {gpu_id} (port {self.gpu_servers[gpu_id].port}): {validation_time:.2f}s")
                logger.info(f"      Generation: {generation_time:.2f}s, Total: {total_time:.2f}s, Score: {validation_score:.4f}")
            
            # Store validation ranking in stats
            self.stats['fastest_validation_gpu'] = ranked_results[0]['gpu_id']
            self.stats['slowest_validation_gpu'] = ranked_results[-1]['gpu_id']
            self.stats['fastest_validation_time'] = ranked_results[0]['validation_time']
            self.stats['slowest_validation_time'] = ranked_results[-1]['validation_time']
            self.stats['fastest_total_time'] = ranked_results[0].get('total_time', 0)
            self.stats['slowest_total_time'] = ranked_results[-1].get('total_time', 0)
        
        return results
    
    def _test_single_gpu_validation(self, gpu_id: int) -> Dict[str, Any]:
        """Test validation on a single GPU using local subnet_accurate_validator.py"""
        gpu_server = self.gpu_servers[gpu_id]
        
        try:
            logger.info(f"📊 Testing validation on GPU {gpu_id}")
            
            # Track total request time from start to finish
            total_start_time = time.time()
            
            # First, generate a model to validate
            generation_result = self.prime_single_gpu(gpu_id, "a test object for validation")
            
            if not generation_result.get('success', False):
                return {
                    'success': False,
                    'gpu_id': gpu_id,
                    'error': 'Generation failed, cannot test validation',
                    'generation_result': generation_result
                }
            
            # Now run local validation using subnet_accurate_validator.py
            validation_start_time = time.time()
            
            # Run local validation via subprocess
            try:
                cmd = [
                    'bash', '-c',
                    f'source /home/mbhat/miniconda/bin/activate && conda activate trellis_new && CUDA_VISIBLE_DEVICES={gpu_id} python subnet_accurate_validator.py "{generation_result["prompt"]}" "{generation_result["prompt"]}" --endpoint generate/ --port {gpu_server.port}'
                ]
                
                # Debug: Log the exact command being executed
                logger.debug(f"   Executing validation command: {cmd[2]}")
                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    timeout=120,
                    cwd=Path(__file__).parent
                )
                
                validation_time = time.time() - validation_start_time
                total_time = time.time() - total_start_time
                
                if result.returncode == 0:
                    # Successfully ran validation, now read the results
                    results_file = f"subnet_validation_results_{gpu_server.port}.json"
                    
                    if Path(results_file).exists():
                        with open(results_file, 'r') as f:
                            validation_data = json.load(f)
                    else:
                        validation_data = {
                            'validation_engine_score': 0.0,
                            'note': 'Results file not found but validation completed'
                        }
                else:
                    # Log the detailed error output for debugging
                    logger.error(f"   Validation subprocess failed (returncode: {result.returncode})")
                    if result.stdout:
                        logger.error(f"   STDOUT: {result.stdout}")
                    if result.stderr:
                        logger.error(f"   STDERR: {result.stderr}")
                    
                    validation_data = {
                        'validation_engine_score': 0.0,
                        'error': f"Validation failed: {result.stderr}",
                        'stdout': result.stdout,
                        'returncode': result.returncode
                    }
                
            except subprocess.TimeoutExpired:
                validation_time = time.time() - validation_start_time
                total_time = time.time() - total_start_time
                validation_data = {
                    'validation_engine_score': 0.0,
                    'error': 'Validation timeout after 120s'
                }
            except Exception as e:
                validation_time = time.time() - validation_start_time
                total_time = time.time() - total_start_time
                validation_data = {
                    'validation_engine_score': 0.0,
                    'error': f'Local validation failed: {str(e)}'
                }
            
            # Process validation results
            validation_score = validation_data.get('validation_engine_score', 0.0)
            has_error = 'error' in validation_data
            
            if validation_score > 0.0 or not has_error:
                gpu_server.validation_count += 1
                gpu_server.last_validation_time = validation_time
                self.stats['successful_validations'] += 1
                self.stats['total_validations'] += 1
                
                result = {
                    'success': True,
                    'gpu_id': gpu_id,
                    'validation_time': validation_time,
                    'total_time': total_time,
                    'validation_data': validation_data,
                    'generation_result': generation_result,
                    'validation_score': validation_score
                }
                
                logger.info(f"   ✅ GPU {gpu_id} validation successful in {validation_time:.2f}s (score: {validation_score:.4f})")
                return result
            else:
                gpu_server.error_count += 1
                self.stats['failed_validations'] += 1
                self.stats['total_validations'] += 1
                
                result = {
                    'success': False,
                    'gpu_id': gpu_id,
                    'error': validation_data.get('error', 'Unknown validation error'),
                    'validation_time': validation_time,
                    'total_time': total_time,
                    'generation_result': generation_result,
                    'validation_score': validation_score
                }
                
                logger.error(f"   ❌ GPU {gpu_id} validation failed: {validation_data.get('error', 'Unknown error')}")
                return result
                
        except Exception as e:
            gpu_server.error_count += 1
            self.stats['failed_validations'] += 1
            self.stats['total_validations'] += 1
            
            result = {
                'success': False,
                'gpu_id': gpu_id,
                'error': str(e),
                'validation_time': 0.0,
                'total_time': 0.0,
                'validation_score': 0.0
            }
            
            logger.error(f"   ❌ GPU {gpu_id} validation exception: {e}")
            return result
    
    def get_performance_ranking(self) -> Dict[str, Any]:
        """Get current performance ranking of GPUs based on last test results"""
        ranking_data = {
            'generation_ranking': [],
            'validation_ranking': [],
            'performance_summary': {}
        }
        
        if 'fastest_gpu' in self.stats:
            ranking_data['performance_summary'] = {
                'fastest_gpu': self.stats['fastest_gpu'],
                'slowest_gpu': self.stats['slowest_gpu'],
                'fastest_time': self.stats['fastest_time'],
                'slowest_time': self.stats['slowest_time'],
                'performance_spread': self.stats['performance_spread'],
                'average_time': self.stats.get('total_generation_time', 0) / max(1, self.stats.get('successful_generations', 1))
            }
        
        if 'fastest_validation_gpu' in self.stats:
            ranking_data['performance_summary'].update({
                'fastest_validation_gpu': self.stats['fastest_validation_gpu'],
                'slowest_validation_gpu': self.stats['slowest_validation_gpu'],
                'fastest_validation_time': self.stats['fastest_validation_time'],
                'slowest_validation_time': self.stats['slowest_validation_time']
            })
        
        return ranking_data
    
    def print_performance_comparison_table(self):
        """Print a comprehensive performance comparison table for all GPUs"""
        logger.info("📊 COMPREHENSIVE GPU PERFORMANCE COMPARISON TABLE")
        logger.info("=" * 100)
        
        # Table header
        logger.info(f"{'GPU':<4} {'Port':<6} {'Status':<8} {'Gen Time':<10} {'Val Time':<10} {'Total Time':<12} {'PLY Size':<12} {'Compression':<12}")
        logger.info("-" * 100)
        
        # Sort GPUs by total performance (generation + validation)
        gpu_performance = []
        for gpu_id, gpu_server in self.gpu_servers.items():
            if gpu_server.generation_count > 0:
                generation_time = gpu_server.last_generation_time or 0
                validation_time = gpu_server.last_validation_time or 0
                total_time = generation_time + validation_time
                
                gpu_performance.append({
                    'gpu_id': gpu_id,
                    'port': gpu_server.port,
                    'status': gpu_server.status,
                    'generation_time': generation_time,
                    'validation_time': validation_time,
                    'total_time': total_time,
                    'ply_size': gpu_server.last_ply_size or 0,
                    'compression': gpu_server.last_compression or 0
                })
        
        # Sort by total time (fastest first)
        gpu_performance.sort(key=lambda x: x['total_time'])
        
        # Print table rows
        for i, perf in enumerate(gpu_performance):
            medal = "🥇" if i == 0 else "🥈" if i == 1 else "🥉" if i == 2 else "  "
            status_icon = "✅" if perf['status'] == "healthy" else "⚠️" if perf['status'] == "already_loaded" else "❌"
            
            logger.info(f"{medal} {perf['gpu_id']:<3} {perf['port']:<6} {status_icon} {perf['status']:<6} "
                       f"{perf['generation_time']:<9.2f}s {perf['validation_time']:<9.2f}s "
                       f"{perf['total_time']:<11.2f}s {perf['ply_size']:<11,} {perf['compression']:<12}")
        
        logger.info("-" * 100)
        
        # Summary statistics
        if gpu_performance:
            fastest_total = gpu_performance[0]['total_time']
            slowest_total = gpu_performance[-1]['total_time']
            avg_total = sum(p['total_time'] for p in gpu_performance) / len(gpu_performance)
            
            logger.info(f"🏆 PERFORMANCE SUMMARY:")
            logger.info(f"   🥇 Fastest Total: GPU {gpu_performance[0]['gpu_id']} ({fastest_total:.2f}s)")
            logger.info(f"   🐌 Slowest Total: GPU {gpu_performance[-1]['gpu_id']} ({slowest_total:.2f}s)")
            logger.info(f"   📊 Average Total: {avg_total:.2f}s")
            logger.info(f"   📈 Performance Spread: {slowest_total - fastest_total:.2f}s")
        
        logger.info("=" * 100)
    
    def collect_validation_results_from_files(self) -> Dict[str, Any]:
        """Collect validation results from JSON files created by subnet_accurate_validator.py"""
        logger.info("📁 Collecting validation results from JSON files...")
        
        results = {}
        base_dir = Path(".")
        
        # Look for validation result files
        for file_path in base_dir.glob("subnet_validation_results_*.json"):
            try:
                with open(file_path, 'r') as f:
                    data = json.load(f)
                    
                port = data.get('port', 'unknown')
                gpu_id = None
                
                # Find which GPU this port corresponds to
                for gpu_id, gpu_server in self.gpu_servers.items():
                    if gpu_server.port == port:
                        break
                
                if gpu_id is not None:
                    results[gpu_id] = {
                        'port': port,
                        'file': file_path.name,
                        'original_prompt': data.get('original_prompt', ''),
                        'optimized_prompt': data.get('optimized_prompt', ''),
                        'prompt_optimized': data.get('prompt_optimized', False),
                        'validation_engine_score': data.get('validation_engine_score', 0.0),
                        'alignment_score': data.get('alignment_score', 0.0),
                        'quality_score': data.get('quality_score', 0.0),
                        'ssim_score': data.get('ssim_score', 0.0),
                        'lpips_score': data.get('lpips_score', 0.0),
                        'demo_fidelity_score': data.get('demo_fidelity_score', 0.0),
                        'task_fidelity_score': data.get('task_fidelity_score', 0.0),
                        'validation_passed': data.get('validation_passed', False),
                        'time_stats': data.get('time_stats', {})
                    }
                    logger.info(f"   ✅ GPU {gpu_id} (port {port}): {file_path.name}")
                else:
                    logger.warning(f"   ⚠️ Port {port} not found in GPU servers")
                    
            except Exception as e:
                logger.error(f"   ❌ Error reading {file_path}: {e}")
        
        logger.info(f"📊 Collected results for {len(results)} GPUs")
        return results
    
    def run_additional_tests(self):
        """Run additional tests and checks"""
        logger.info("🧪 Running additional tests and checks...")
        
        # Test 1: Check if all GPUs can handle the same prompt consistently
        test_prompt = "a simple red cube on a white surface"
        logger.info(f"🎯 Test 1: Consistency test with prompt: '{test_prompt}'")
        
        consistency_results = {}
        for gpu_id in range(self.num_gpus):
            try:
                result = self.prime_single_gpu(gpu_id, test_prompt)
                if result.get('success', False):
                    consistency_results[gpu_id] = {
                        'generation_time': result.get('generation_time', 0),
                        'ply_size': result.get('ply_size_bytes', 0),
                        'compression': result.get('compression_ratio', 0)
                    }
                    logger.info(f"   ✅ GPU {gpu_id}: {result.get('generation_time', 0):.2f}s, "
                               f"{result.get('ply_size_bytes', 0):,} bytes")
                else:
                    logger.warning(f"   ⚠️ GPU {gpu_id}: Failed")
            except Exception as e:
                logger.error(f"   ❌ GPU {gpu_id}: Error - {e}")
        
        # Test 2: Check memory usage across GPUs
        logger.info("🧠 Test 2: GPU memory usage check")
        try:
            import torch
            if torch.cuda.is_available():
                for gpu_id in range(self.num_gpus):
                    memory_allocated = torch.cuda.memory_allocated(gpu_id) / 1024**3  # GB
                    memory_reserved = torch.cuda.memory_reserved(gpu_id) / 1024**3  # GB
                    logger.info(f"   GPU {gpu_id}: {memory_allocated:.2f}GB allocated, {memory_reserved:.2f}GB reserved")
            else:
                logger.info("   ⚠️ CUDA not available for memory check")
        except Exception as e:
            logger.warning(f"   ⚠️ Memory check failed: {e}")
        
        # Test 3: Check network latency to each GPU server
        logger.info("🌐 Test 3: Network latency check")
        for gpu_id, gpu_server in self.gpu_servers.items():
            try:
                import time
                start_time = time.time()
                response = requests.get(gpu_server.health_url, timeout=5)
                latency = (time.time() - start_time) * 1000  # ms
                
                if response.status_code == 200:
                    logger.info(f"   GPU {gpu_id} (port {gpu_server.port}): {latency:.1f}ms")
                else:
                    logger.warning(f"   GPU {gpu_id} (port {gpu_server.port}): HTTP {response.status_code}")
            except Exception as e:
                logger.error(f"   GPU {gpu_id} (port {gpu_server.port}): Connection failed - {e}")
        
        logger.info("✅ Additional tests completed")
        return consistency_results
    
    def print_validation_results_table(self):
        """Print a comprehensive table of validation results from JSON files"""
        logger.info("📊 VALIDATION RESULTS COMPARISON TABLE")
        logger.info("=" * 120)
        
        validation_results = self.collect_validation_results_from_files()
        
        if not validation_results:
            logger.info("⚠️ No validation results found. Run validation tests first.")
            logger.info("=" * 120)
            return
        
        # Table header
        logger.info(f"{'GPU':<4} {'Port':<6} {'Prompt':<30} {'Score':<8} {'Alignment':<10} {'Quality':<8} {'Demo Fidelity':<12} {'Status':<10}")
        logger.info("-" * 120)
        
        # Sort by validation score (highest first)
        sorted_results = sorted(
            validation_results.items(), 
            key=lambda x: x[1].get('validation_engine_score', 0.0), 
            reverse=True
        )
        
        # Print table rows
        for i, (gpu_id, result) in enumerate(sorted_results):
            medal = "🥇" if i == 0 else "🥈" if i == 1 else "🥉" if i == 2 else "  "
            
            prompt = result.get('original_prompt', 'Unknown')[:28] + "..." if len(result.get('original_prompt', '')) > 30 else result.get('original_prompt', 'Unknown')
            score = result.get('validation_engine_score', 0.0)
            alignment = result.get('alignment_score', 0.0)
            quality = result.get('quality_score', 0.0)
            demo_fidelity = result.get('demo_fidelity_score', 0.0)
            status = "✅ PASS" if result.get('validation_passed', False) else "❌ FAIL"
            
            logger.info(f"{medal} {gpu_id:<3} {result.get('port', 'N/A'):<6} {prompt:<30} "
                       f"{score:<8.4f} {alignment:<10.4f} {quality:<8.4f} {demo_fidelity:<12.2f} {status:<10}")
        
        logger.info("-" * 120)
        
        # Summary statistics
        scores = [r.get('validation_engine_score', 0.0) for r in validation_results.values()]
        alignments = [r.get('alignment_score', 0.0) for r in validation_results.values()]
        qualities = [r.get('quality_score', 0.0) for r in validation_results.values()]
        
        if scores:
            logger.info(f"📊 VALIDATION SUMMARY:")
            logger.info(f"   🏆 Best Score: {max(scores):.4f} (GPU {sorted_results[0][0]})")
            logger.info(f"   📉 Worst Score: {min(scores):.4f}")
            logger.info(f"   📈 Average Score: {sum(scores)/len(scores):.4f}")
            logger.info(f"   🎯 Average Alignment: {sum(alignments)/len(alignments):.4f}")
            logger.info(f"   💎 Average Quality: {sum(qualities)/len(qualities):.4f}")
            logger.info(f"   ✅ Pass Rate: {sum(1 for r in validation_results.values() if r.get('validation_passed', False))}/{len(validation_results)} "
                       f"({100*sum(1 for r in validation_results.values() if r.get('validation_passed', False))/len(validation_results):.1f}%)")
        
        logger.info("=" * 120)
    
    def get_comprehensive_status(self) -> Dict[str, Any]:
        """Get comprehensive status of all GPU servers"""
        logger.info("📊 Getting comprehensive status of all GPU servers...")
        
        status_data = {
            'timestamp': time.time(),
            'uptime_seconds': time.time() - self.start_time,
            'gpu_servers': {},
            'overall_stats': self.stats.copy(),
            'health_summary': {},
            'performance_ranking': self.get_performance_ranking()
        }
        
        # Get status from each GPU server
        for gpu_id, gpu_server in self.gpu_servers.items():
            gpu_status = {
                'gpu_id': gpu_id,
                'port': gpu_server.port,
                'status': gpu_server.status,
                'process_pid': gpu_server.process.pid if gpu_server.process else None,
                'generation_count': gpu_server.generation_count,
                'validation_count': gpu_server.validation_count,
                'error_count': gpu_server.error_count,
                'last_response_time': gpu_server.last_response_time,
                'last_health_check': gpu_server.last_health_check
            }
            
            # Get detailed server status if available
            try:
                server_status = self.get_server_status(gpu_id)
                if server_status:
                    gpu_status['server_status'] = server_status
            except Exception as e:
                gpu_status['server_status_error'] = str(e)
            
            status_data['gpu_servers'][gpu_id] = gpu_status
        
        # Health summary
        healthy_count = sum(1 for s in status_data['gpu_servers'].values() if s['status'] == 'healthy')
        total_count = len(status_data['gpu_servers'])
        
        status_data['health_summary'] = {
            'total_servers': total_count,
            'healthy_servers': healthy_count,
            'unhealthy_servers': total_count - healthy_count,
            'health_percentage': (healthy_count / total_count * 100) if total_count > 0 else 0
        }
        
        return status_data
    
    def save_status_report(self, status_data: Dict[str, Any]):
        """Save status report to file"""
        try:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            status_file = self.output_dir / f"gpu_status_report_{timestamp}.json"
            
            with open(status_file, 'w') as f:
                json.dump(status_data, f, indent=2)
            
            logger.info(f"💾 Status report saved to {status_file}")
            
        except Exception as e:
            logger.error(f"❌ Failed to save status report: {e}")
    
    def print_status_summary(self):
        """Print a summary of current status"""
        logger.info("📊 GPU SERVER STATUS SUMMARY")
        logger.info("=" * 60)
        
        uptime_hours = (time.time() - self.start_time) / 3600
        logger.info(f"Uptime: {uptime_hours:.2f} hours")
        
        # Server status
        healthy_count = sum(1 for s in self.gpu_servers.values() if s.status == 'healthy')
        already_loaded_count = sum(1 for s in self.gpu_servers.values() if s.status == 'already_loaded')
        logger.info(f"GPU Servers: {healthy_count}/{self.num_gpus} healthy")
        logger.info(f"Already Loaded: {already_loaded_count}, Newly Started: {self.stats['servers_started']}")
        
        # Generation statistics
        logger.info(f"Total Generations: {self.stats['total_generations']}")
        logger.info(f"Successful: {self.stats['successful_generations']}")
        logger.info(f"Failed: {self.stats['failed_generations']}")
        
        if self.stats['successful_generations'] > 0:
            success_rate = (self.stats['successful_generations'] / self.stats['total_generations']) * 100
            logger.info(f"Success Rate: {success_rate:.1f}%")
        
        # Validation statistics
        logger.info(f"Total Validations: {self.stats['total_validations']}")
        logger.info(f"Successful: {self.stats['successful_validations']}")
        logger.info(f"Failed: {self.stats['failed_validations']}")
        
        # Performance metrics
        if self.stats['parallel_generation_time'] > 0:
            logger.info(f"Parallel Generation Time: {self.stats['parallel_generation_time']:.2f}s")
        
        if self.stats['parallel_validation_time'] > 0:
            logger.info(f"Parallel Validation Time: {self.stats['parallel_validation_time']:.2f}s")
        
        # Performance rankings
        if 'fastest_gpu' in self.stats:
            logger.info(f"🏆 Fastest GPU: {self.stats['fastest_gpu']} ({self.stats['fastest_time']:.2f}s)")
            logger.info(f"🐌 Slowest GPU: {self.stats['slowest_gpu']} ({self.stats['slowest_time']:.2f}s)")
            logger.info(f"📊 Performance Spread: {self.stats['performance_spread']:.2f}s")
        
        if 'fastest_validation_gpu' in self.stats:
            logger.info(f"🏆 Fastest Validation: GPU {self.stats['fastest_validation_gpu']} ({self.stats['fastest_validation_time']:.2f}s)")
            logger.info(f"🐌 Slowest Validation: GPU {self.stats['slowest_validation_gpu']} ({self.stats['slowest_validation_time']:.2f}s)")
        
        # Performance comparison table
        if any(gpu.generation_count > 0 for gpu in self.gpu_servers.values()):
            logger.info("")
            self.print_performance_comparison_table()
        
        # Validation results table
        logger.info("")
        self.print_validation_results_table()
        
        # GPU-specific information
        logger.info("\nGPU Details:")
        for gpu_id, gpu_server in self.gpu_servers.items():
            if gpu_server.status == 'healthy':
                status_icon = "✅"
            elif gpu_server.status == 'already_loaded':
                status_icon = "🔄"
            else:
                status_icon = "❌"
            logger.info(f"  GPU {gpu_id} (port {gpu_server.port}): {status_icon} {gpu_server.status}")
            logger.info(f"    Generations: {gpu_server.generation_count}, Validations: {gpu_server.validation_count}")
            logger.info(f"    Errors: {gpu_server.error_count}")
        
        logger.info("=" * 60)
    
    def cleanup(self):
        """Clean up all GPU servers"""
        logger.info("🧹 Cleaning up GPU servers...")
        
        for gpu_id, gpu_server in self.gpu_servers.items():
            if gpu_server.process:
                try:
                    logger.info(f"   Stopping GPU {gpu_id} server (PID: {gpu_server.process.pid})")
                    gpu_server.process.terminate()
                    gpu_server.process.wait(timeout=10)
                    gpu_server.status = "stopped"
                    logger.info(f"   ✅ GPU {gpu_id} server stopped")
                except subprocess.TimeoutExpired:
                    logger.warning(f"   ⚠️ GPU {gpu_id} server didn't stop gracefully, killing...")
                    gpu_server.process.kill()
                    gpu_server.status = "killed"
                except Exception as e:
                    logger.error(f"   ❌ Error stopping GPU {gpu_id} server: {e}")
        
        logger.info("✅ Cleanup complete")

def signal_handler(signum, frame):
    """Handle shutdown signals"""
    logger.info(f"🛑 Received signal {signum}, shutting down...")
    if hasattr(signal_handler, 'manager'):
        signal_handler.manager.cleanup()
    sys.exit(0)

async def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="GPU Server Wrapper for TRELLIS")
    parser.add_argument("--gpus", type=int, default=8, help="Number of GPUs to use")
    parser.add_argument("--base-port", type=int, default=8096, help="Base port number")
    parser.add_argument("--server-script", default="trellis_subnit_server_mix_lora_flash.py", 
                       help="TRELLIS server script path")
    parser.add_argument("--output-dir", default="./gpu_server_outputs", help="Output directory")
    parser.add_argument("--skip-startup", action="store_true", help="Skip server startup (assume already running)")
    parser.add_argument("--skip-priming", action="store_true", help="Skip GPU priming")
    parser.add_argument("--skip-validation", action="store_true", help="Skip validation testing")
    parser.add_argument("--check-status-only", action="store_true", help="Only check GPU loading status and exit")
    parser.add_argument("--show-ranking", action="store_true", help="Show current performance ranking and exit")
    parser.add_argument("--run-additional-tests", action="store_true", help="Run additional tests (consistency, memory, latency)")
    parser.add_argument("--show-validation-results", action="store_true", help="Show validation results table from JSON files")
    
    args = parser.parse_args()
    
    # Setup signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Create GPU server manager
    manager = GPUServerManager(
        num_gpus=args.gpus,
        base_port=args.base_port,
        server_script=args.server_script,
        output_dir=args.output_dir
    )
    
    # Store manager reference for signal handler
    signal_handler.manager = manager
    
    try:
        # Check if we only want to check status
        if args.check_status_only:
            logger.info("🔍 Checking GPU loading status only...")
            loading_status = manager.check_gpu_loading_status()
            
            # Print summary and exit
            already_loaded_count = sum(1 for status in loading_status.values() if status == "already_loaded")
            needs_loading_count = sum(1 for status in loading_status.values() if status == "needs_loading")
            
            logger.info("📊 GPU Loading Status Summary:")
            logger.info(f"   Already Loaded: {already_loaded_count}/{args.gpus}")
            logger.info(f"   Needs Loading: {needs_loading_count}/{args.gpus}")
            
            if already_loaded_count == args.gpus:
                logger.info("🎉 All GPUs are ready for use!")
            elif already_loaded_count > 0:
                logger.info(f"🔄 {already_loaded_count} GPUs ready, {needs_loading_count} need loading")
            else:
                logger.info("⏳ All GPUs need loading")
            
            return
        
        # Check if we only want to show ranking
        if args.show_ranking:
            logger.info("🏆 Showing current performance ranking...")
            ranking_data = manager.get_performance_ranking()
            
            if ranking_data['performance_summary']:
                logger.info("📊 Performance Ranking Summary:")
                if 'fastest_gpu' in ranking_data['performance_summary']:
                    logger.info(f"   🥇 Fastest GPU: {ranking_data['performance_summary']['fastest_gpu']} ({ranking_data['performance_summary']['fastest_time']:.2f}s)")
                    logger.info(f"   🐌 Slowest GPU: {ranking_data['performance_summary']['slowest_gpu']} ({ranking_data['performance_summary']['slowest_time']:.2f}s)")
                    logger.info(f"   📊 Performance Spread: {ranking_data['performance_summary']['performance_spread']:.2f}s")
                    logger.info(f"   📈 Average Time: {ranking_data['performance_summary']['average_time']:.2f}s")
                
                if 'fastest_validation_gpu' in ranking_data['performance_summary']:
                    logger.info(f"   🏆 Fastest Validation: GPU {ranking_data['performance_summary']['fastest_validation_gpu']} ({ranking_data['performance_summary']['fastest_validation_time']:.2f}s)")
                    logger.info(f"   🐌 Slowest Validation: GPU {ranking_data['performance_summary']['slowest_validation_gpu']} ({ranking_data['performance_summary']['slowest_validation_time']:.2f}s)")
            else:
                logger.info("⚠️ No performance data available. Run tests first to generate rankings.")
            
            return
        
        # Check if we only want to run additional tests
        if args.run_additional_tests:
            logger.info("🧪 Running additional tests only...")
            manager.run_additional_tests()
            return
        
        # Check if we only want to show validation results
        if args.show_validation_results:
            logger.info("📊 Showing validation results table...")
            manager.print_validation_results_table()
            return
        
        # Step 1: Check GPU loading status and start servers if needed
        if not args.skip_startup:
            logger.info("🔍 STEP 1: Checking GPU loading status...")
            
            # First check which GPUs are already loaded
            loading_status = manager.check_gpu_loading_status()
            already_loaded_count = sum(1 for status in loading_status.values() if status == "already_loaded")
            
            if already_loaded_count == args.gpus:
                logger.info("✅ All GPUs are already loaded and ready!")
                # Update status to healthy for already loaded GPUs
                for gpu_id, status in loading_status.items():
                    if status == "already_loaded":
                        manager.gpu_servers[gpu_id].status = "healthy"
            else:
                logger.info(f"🚀 Starting servers on {args.gpus - already_loaded_count} GPUs that need loading...")
                if not manager.start_all_servers():
                    logger.error("❌ Failed to start GPU servers")
                    return
                
                # Wait a bit more for servers to fully initialize
                logger.info("⏳ Waiting for servers to fully initialize...")
                time.sleep(15)
            
            # Final health check
            health_results = manager.check_all_servers_health()
            healthy_count = sum(health_results.values())
            
            if healthy_count == 0:
                logger.error("❌ No healthy servers found after startup")
                return
            
            logger.info(f"✅ {healthy_count}/{args.gpus} servers are healthy and ready")
        else:
            logger.info("⏭️ Skipping server startup (assume already running)")
            manager.check_all_servers_health()
        
        # Step 2: Prime all GPUs in parallel
        if not args.skip_priming:
            logger.info("🎨 STEP 2: Priming all GPUs in parallel...")
            priming_results = manager.prime_all_gpus_parallel()
            
            # Save priming results
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            priming_file = manager.output_dir / f"priming_results_{timestamp}.json"
            with open(priming_file, 'w') as f:
                json.dump(priming_results, f, indent=2)
            logger.info(f"💾 Priming results saved to {priming_file}")
        else:
            logger.info("⏭️ Skipping GPU priming")
        
        # Step 3: Test validation across all GPUs in parallel
        if not args.skip_validation:
            logger.info("📊 STEP 3: Testing validation across all GPUs in parallel...")
            validation_results = manager.test_validation_parallel()
            
            # Save validation results
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            validation_file = manager.output_dir / f"validation_results_{timestamp}.json"
            with open(validation_file, 'w') as f:
                json.dump(validation_results, f, indent=2)
            logger.info(f"💾 Validation results saved to {validation_file}")
        else:
            logger.info("⏭️ Skipping validation testing")
        
        # Final status report
        logger.info("📊 Generating final status report...")
        status_data = manager.get_comprehensive_status()
        manager.save_status_report(status_data)
        manager.print_status_summary()
        
        logger.info("🎉 GPU server testing complete!")
        
        # Keep servers running for manual testing
        logger.info("🔄 Servers will continue running for manual testing...")
        logger.info("   Use Ctrl+C to stop all servers")
        
        # Keep alive
        while True:
            await asyncio.sleep(60)  # Check every minute
            # Periodic health check
            manager.check_all_servers_health()
            
    except KeyboardInterrupt:
        logger.info("🛑 Interrupted by user")
    except Exception as e:
        logger.error(f"❌ Main execution failed: {e}")
        traceback.print_exc()
    finally:
        if not args.skip_startup:
            logger.info("🧹 Cleaning up...")
            manager.cleanup()

if __name__ == "__main__":
    asyncio.run(main())
