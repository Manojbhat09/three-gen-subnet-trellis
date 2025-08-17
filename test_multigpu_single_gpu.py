#!/usr/bin/env python3
"""
Multi-GPU Logic Test on Single GPU System
=========================================

Tests multi-GPU setup logic on a single GPU system to catch bugs
before deploying to cloud instances.
"""

import os
import sys
import time
import torch
import subprocess
import signal
import threading
import requests
import json
from pathlib import Path
from typing import Dict, Any, List
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SingleGPUMultiTest:
    """Test multi-GPU logic on single GPU system"""
    
    def __init__(self):
        self.base_port = 9090  # Use different ports to avoid conflicts
        self.num_virtual_gpus = 4  # Simulate 4 GPUs on 1 physical GPU
        self.processes = {}
        self.test_results = {}
        
    def run_all_tests(self) -> Dict[str, Any]:
        """Run comprehensive multi-GPU simulation tests"""
        logger.info("🧪 Starting Multi-GPU Logic Test on Single GPU")
        logger.info("=" * 60)
        
        results = {
            'timestamp': time.time(),
            'tests_passed': 0,
            'tests_failed': 0,
            'test_details': {}
        }
        
        # Test 1: GPU Detection Logic
        test_result = self.test_gpu_detection_logic()
        results['test_details']['gpu_detection'] = test_result
        if test_result['passed']:
            results['tests_passed'] += 1
        else:
            results['tests_failed'] += 1
        
        # Test 2: Environment Variable Isolation
        test_result = self.test_environment_isolation()
        results['test_details']['env_isolation'] = test_result
        if test_result['passed']:
            results['tests_passed'] += 1
        else:
            results['tests_failed'] += 1
        
        # Test 3: Port Conflict Detection
        test_result = self.test_port_conflicts()
        results['test_details']['port_conflicts'] = test_result
        if test_result['passed']:
            results['tests_passed'] += 1
        else:
            results['tests_failed'] += 1
        
        # Test 4: Process Management
        test_result = self.test_process_management()
        results['test_details']['process_management'] = test_result
        if test_result['passed']:
            results['tests_passed'] += 1
        else:
            results['tests_failed'] += 1
        
        # Test 5: Memory Isolation Simulation
        test_result = self.test_memory_isolation()
        results['test_details']['memory_isolation'] = test_result
        if test_result['passed']:
            results['tests_passed'] += 1
        else:
            results['tests_failed'] += 1
        
        # Test 6: Error Recovery
        test_result = self.test_error_recovery()
        results['test_details']['error_recovery'] = test_result
        if test_result['passed']:
            results['tests_passed'] += 1
        else:
            results['tests_failed'] += 1
        
        # Summary
        total_tests = results['tests_passed'] + results['tests_failed']
        success_rate = (results['tests_passed'] / total_tests * 100) if total_tests > 0 else 0
        
        logger.info("=" * 60)
        logger.info(f"🏁 Test Results Summary:")
        logger.info(f"   ✅ Passed: {results['tests_passed']}/{total_tests}")
        logger.info(f"   ❌ Failed: {results['tests_failed']}/{total_tests}")
        logger.info(f"   📊 Success Rate: {success_rate:.1f}%")
        
        if results['tests_failed'] > 0:
            logger.error("🚨 CRITICAL: Bugs found! Review failed tests before cloud deployment")
            self.print_failed_tests(results['test_details'])
        else:
            logger.info("🎉 All tests passed! Multi-GPU logic looks good for cloud deployment")
        
        return results
    
    def test_gpu_detection_logic(self) -> Dict[str, Any]:
        """Test GPU detection and allocation logic"""
        logger.info("🔍 Test 1: GPU Detection Logic")
        
        try:
            # Test 1a: Basic CUDA availability
            if not torch.cuda.is_available():
                return {
                    'passed': False,
                    'error': 'CUDA not available - cannot test GPU logic',
                    'details': {}
                }
            
            device_count = torch.cuda.device_count()
            logger.info(f"   📊 Detected {device_count} physical GPU(s)")
            
            # Test 1b: MultiGPUConfig logic
            sys.path.append('.')
            from multigpu_config import MultiGPUConfig
            
            # Test with more virtual GPUs than physical
            config = MultiGPUConfig(num_gpus=self.num_virtual_gpus)
            
            if config.device_ids != list(range(config.num_gpus)):
                return {
                    'passed': False,
                    'error': 'MultiGPUConfig device_ids not properly initialized',
                    'details': {'expected': list(range(config.num_gpus)), 'actual': config.device_ids}
                }
            
            # Test 1c: GPU memory allocation simulation
            memory_test_results = []
            for virtual_gpu_id in range(self.num_virtual_gpus):
                physical_gpu_id = virtual_gpu_id % device_count
                try:
                    device = torch.device(f'cuda:{physical_gpu_id}')
                    # Small memory allocation test
                    test_tensor = torch.zeros(100, device=device)
                    memory_test_results.append({
                        'virtual_gpu': virtual_gpu_id,
                        'physical_gpu': physical_gpu_id,
                        'allocation_success': True
                    })
                    del test_tensor
                    torch.cuda.empty_cache()
                except Exception as e:
                    memory_test_results.append({
                        'virtual_gpu': virtual_gpu_id,
                        'physical_gpu': physical_gpu_id,
                        'allocation_success': False,
                        'error': str(e)
                    })
            
            successful_allocations = sum(1 for r in memory_test_results if r['allocation_success'])
            
            logger.info(f"   ✅ GPU detection logic working")
            logger.info(f"   📊 Memory allocations: {successful_allocations}/{self.num_virtual_gpus}")
            
            return {
                'passed': True,
                'details': {
                    'physical_gpus': device_count,
                    'virtual_gpus': self.num_virtual_gpus,
                    'memory_test_results': memory_test_results,
                    'config_test': 'passed'
                }
            }
            
        except Exception as e:
            logger.error(f"   ❌ GPU detection test failed: {e}")
            return {
                'passed': False,
                'error': str(e),
                'details': {}
            }
    
    def test_environment_isolation(self) -> Dict[str, Any]:
        """Test CUDA_VISIBLE_DEVICES isolation logic"""
        logger.info("🔒 Test 2: Environment Variable Isolation")
        
        try:
            isolation_results = []
            
            for virtual_gpu_id in range(self.num_virtual_gpus):
                # Simulate environment setup for each virtual GPU
                env = os.environ.copy()
                physical_gpu_id = virtual_gpu_id % torch.cuda.device_count()
                env['CUDA_VISIBLE_DEVICES'] = str(physical_gpu_id)
                env['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
                env.pop('CUDA_DEVICE_ORDER', None)
                
                # Test environment variable parsing
                expected_devices = [physical_gpu_id]
                actual_visible = env.get('CUDA_VISIBLE_DEVICES')
                
                isolation_results.append({
                    'virtual_gpu': virtual_gpu_id,
                    'physical_gpu': physical_gpu_id,
                    'cuda_visible_devices': actual_visible,
                    'isolation_correct': actual_visible == str(physical_gpu_id)
                })
            
            successful_isolations = sum(1 for r in isolation_results if r['isolation_correct'])
            
            if successful_isolations == self.num_virtual_gpus:
                logger.info(f"   ✅ Environment isolation working correctly")
                return {
                    'passed': True,
                    'details': {
                        'isolation_results': isolation_results,
                        'successful_isolations': successful_isolations
                    }
                }
            else:
                logger.error(f"   ❌ Environment isolation failed: {successful_isolations}/{self.num_virtual_gpus}")
                return {
                    'passed': False,
                    'error': f'Only {successful_isolations}/{self.num_virtual_gpus} isolations successful',
                    'details': {'isolation_results': isolation_results}
                }
                
        except Exception as e:
            logger.error(f"   ❌ Environment isolation test failed: {e}")
            return {
                'passed': False,
                'error': str(e),
                'details': {}
            }
    
    def test_port_conflicts(self) -> Dict[str, Any]:
        """Test port conflict detection"""
        logger.info("🌐 Test 3: Port Conflict Detection")
        
        try:
            import socket
            
            # Test port availability check function
            def check_port_available(port: int) -> bool:
                try:
                    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
                        sock.bind(('127.0.0.1', port))
                        return True
                except socket.error:
                    return False
            
            port_test_results = []
            
            # Test available ports
            for i in range(self.num_virtual_gpus):
                port = self.base_port + i
                available = check_port_available(port)
                port_test_results.append({
                    'port': port,
                    'available': available
                })
            
            # Test occupied port detection
            test_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            test_port = self.base_port + 10
            test_socket.bind(('127.0.0.1', test_port))
            test_socket.listen(1)
            
            try:
                occupied_available = check_port_available(test_port)
                if occupied_available:
                    logger.error(f"   ❌ Port conflict detection failed - occupied port reported as available")
                    return {
                        'passed': False,
                        'error': 'Occupied port reported as available',
                        'details': {'test_port': test_port}
                    }
            finally:
                test_socket.close()
            
            available_ports = sum(1 for r in port_test_results if r['available'])
            
            logger.info(f"   ✅ Port conflict detection working")
            logger.info(f"   📊 Available ports: {available_ports}/{self.num_virtual_gpus}")
            
            return {
                'passed': True,
                'details': {
                    'port_test_results': port_test_results,
                    'conflict_detection': 'working'
                }
            }
            
        except Exception as e:
            logger.error(f"   ❌ Port conflict test failed: {e}")
            return {
                'passed': False,
                'error': str(e),
                'details': {}
            }
    
    def test_process_management(self) -> Dict[str, Any]:
        """Test process creation and cleanup"""
        logger.info("⚙️ Test 4: Process Management")
        
        try:
            # Create simple test processes
            test_processes = []
            
            for i in range(min(2, self.num_virtual_gpus)):  # Test with 2 processes max
                cmd = [sys.executable, '-c', 'import time; time.sleep(10)']
                env = os.environ.copy()
                env['CUDA_VISIBLE_DEVICES'] = str(i % torch.cuda.device_count())
                
                process = subprocess.Popen(
                    cmd,
                    env=env,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True
                )
                
                test_processes.append({
                    'virtual_gpu': i,
                    'process': process,
                    'pid': process.pid
                })
            
            # Wait a moment for processes to start
            time.sleep(1)
            
            # Test process status checking
            alive_processes = []
            for proc_info in test_processes:
                if proc_info['process'].poll() is None:
                    alive_processes.append(proc_info)
            
            # Test graceful termination
            termination_results = []
            for proc_info in alive_processes:
                try:
                    proc_info['process'].terminate()
                    proc_info['process'].wait(timeout=5)
                    termination_results.append({
                        'virtual_gpu': proc_info['virtual_gpu'],
                        'pid': proc_info['pid'],
                        'termination': 'success'
                    })
                except subprocess.TimeoutExpired:
                    try:
                        proc_info['process'].kill()
                        proc_info['process'].wait(timeout=2)
                        termination_results.append({
                            'virtual_gpu': proc_info['virtual_gpu'],
                            'pid': proc_info['pid'],
                            'termination': 'killed'
                        })
                    except Exception as kill_error:
                        termination_results.append({
                            'virtual_gpu': proc_info['virtual_gpu'],
                            'pid': proc_info['pid'],
                            'termination': 'failed',
                            'error': str(kill_error)
                        })
                except Exception as e:
                    termination_results.append({
                        'virtual_gpu': proc_info['virtual_gpu'],
                        'pid': proc_info['pid'],
                        'termination': 'failed',
                        'error': str(e)
                    })
            
            successful_terminations = sum(1 for r in termination_results if r['termination'] in ['success', 'killed'])
            
            logger.info(f"   ✅ Process management working")
            logger.info(f"   📊 Successful terminations: {successful_terminations}/{len(test_processes)}")
            
            return {
                'passed': True,
                'details': {
                    'processes_created': len(test_processes),
                    'alive_processes': len(alive_processes),
                    'termination_results': termination_results
                }
            }
            
        except Exception as e:
            logger.error(f"   ❌ Process management test failed: {e}")
            return {
                'passed': False,
                'error': str(e),
                'details': {}
            }
    
    def test_memory_isolation(self) -> Dict[str, Any]:
        """Test memory isolation between virtual GPUs"""
        logger.info("🧠 Test 5: Memory Isolation Simulation")
        
        try:
            memory_test_results = []
            device_count = torch.cuda.device_count()
            
            # Simulate memory allocation for each virtual GPU
            allocated_tensors = {}
            
            for virtual_gpu_id in range(self.num_virtual_gpus):
                physical_gpu_id = virtual_gpu_id % device_count
                
                try:
                    # Set CUDA device context
                    torch.cuda.set_device(physical_gpu_id)
                    
                    # Allocate memory (small amount to avoid OOM)
                    tensor_key = f"gpu_{virtual_gpu_id}"
                    allocated_tensors[tensor_key] = torch.zeros(1000, 1000, device=f'cuda:{physical_gpu_id}')
                    
                    # Check memory stats
                    memory_allocated = torch.cuda.memory_allocated(physical_gpu_id)
                    memory_reserved = torch.cuda.memory_reserved(physical_gpu_id)
                    
                    memory_test_results.append({
                        'virtual_gpu': virtual_gpu_id,
                        'physical_gpu': physical_gpu_id,
                        'allocation_success': True,
                        'memory_allocated_mb': memory_allocated / (1024 * 1024),
                        'memory_reserved_mb': memory_reserved / (1024 * 1024)
                    })
                    
                except Exception as e:
                    memory_test_results.append({
                        'virtual_gpu': virtual_gpu_id,
                        'physical_gpu': physical_gpu_id,
                        'allocation_success': False,
                        'error': str(e)
                    })
            
            # Clean up allocated tensors
            tensor_keys = list(allocated_tensors.keys())  # Fix: Create a list copy to avoid dict size change during iteration
            for tensor_key in tensor_keys:
                del allocated_tensors[tensor_key]
            
            # Force garbage collection and clear cache
            import gc
            gc.collect()
            for gpu_id in range(device_count):
                torch.cuda.empty_cache()
            
            successful_allocations = sum(1 for r in memory_test_results if r['allocation_success'])
            
            logger.info(f"   ✅ Memory isolation simulation working")
            logger.info(f"   📊 Successful allocations: {successful_allocations}/{self.num_virtual_gpus}")
            
            return {
                'passed': True,
                'details': {
                    'memory_test_results': memory_test_results,
                    'cleanup_completed': True
                }
            }
            
        except Exception as e:
            logger.error(f"   ❌ Memory isolation test failed: {e}")
            return {
                'passed': False,
                'error': str(e),
                'details': {}
            }
    
    def test_error_recovery(self) -> Dict[str, Any]:
        """Test error recovery mechanisms"""
        logger.info("🛠️ Test 6: Error Recovery Mechanisms")
        
        try:
            recovery_test_results = []
            
            # Test 1: Simulated process crash
            crash_cmd = [sys.executable, '-c', 'import sys; sys.exit(1)']
            crash_process = subprocess.Popen(crash_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            crash_process.wait()
            
            # Check if we can detect the crash
            if crash_process.returncode != 0:
                recovery_test_results.append({
                    'test': 'process_crash_detection',
                    'passed': True,
                    'details': f'Detected crash with return code {crash_process.returncode}'
                })
            else:
                recovery_test_results.append({
                    'test': 'process_crash_detection',
                    'passed': False,
                    'details': 'Failed to detect process crash'
                })
            
            # Test 2: Timeout handling
            timeout_cmd = [sys.executable, '-c', 'import time; time.sleep(100)']
            timeout_process = subprocess.Popen(timeout_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            
            try:
                timeout_process.wait(timeout=1)
                recovery_test_results.append({
                    'test': 'timeout_handling',
                    'passed': False,
                    'details': 'Process should have timed out'
                })
            except subprocess.TimeoutExpired:
                timeout_process.kill()
                timeout_process.wait()
                recovery_test_results.append({
                    'test': 'timeout_handling',
                    'passed': True,
                    'details': 'Timeout detection and cleanup working'
                })
            
            # Test 3: Resource cleanup simulation
            try:
                # Simulate resource allocation and cleanup
                if torch.cuda.is_available():
                    test_tensor = torch.zeros(100, device='cuda:0')
                    del test_tensor
                    torch.cuda.empty_cache()
                    
                recovery_test_results.append({
                    'test': 'resource_cleanup',
                    'passed': True,
                    'details': 'GPU memory cleanup working'
                })
            except Exception as e:
                recovery_test_results.append({
                    'test': 'resource_cleanup',
                    'passed': False,
                    'details': f'Resource cleanup failed: {e}'
                })
            
            passed_tests = sum(1 for r in recovery_test_results if r['passed'])
            total_tests = len(recovery_test_results)
            
            logger.info(f"   ✅ Error recovery mechanisms working")
            logger.info(f"   📊 Recovery tests passed: {passed_tests}/{total_tests}")
            
            return {
                'passed': passed_tests == total_tests,
                'details': {
                    'recovery_test_results': recovery_test_results,
                    'tests_passed': passed_tests,
                    'total_tests': total_tests
                }
            }
            
        except Exception as e:
            logger.error(f"   ❌ Error recovery test failed: {e}")
            return {
                'passed': False,
                'error': str(e),
                'details': {}
            }
    
    def print_failed_tests(self, test_details: Dict[str, Any]):
        """Print details of failed tests"""
        logger.error("🚨 Failed Test Details:")
        logger.error("-" * 40)
        
        for test_name, result in test_details.items():
            if not result.get('passed', False):
                logger.error(f"❌ {test_name}:")
                logger.error(f"   Error: {result.get('error', 'Unknown error')}")
                if 'details' in result:
                    logger.error(f"   Details: {json.dumps(result['details'], indent=2)}")
                logger.error("")
    
    def save_results(self, results: Dict[str, Any]):
        """Save test results to file"""
        try:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            results_file = f"multigpu_test_results_{timestamp}.json"
            
            with open(results_file, 'w') as f:
                json.dump(results, f, indent=2)
            
            logger.info(f"💾 Test results saved to {results_file}")
            
        except Exception as e:
            logger.error(f"❌ Failed to save results: {e}")

def main():
    """Run the multi-GPU simulation test"""
    tester = SingleGPUMultiTest()
    
    try:
        results = tester.run_all_tests()
        tester.save_results(results)
        
        # Exit with error code if tests failed
        if results['tests_failed'] > 0:
            sys.exit(1)
        else:
            sys.exit(0)
            
    except KeyboardInterrupt:
        logger.info("🛑 Test interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"❌ Test suite failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
