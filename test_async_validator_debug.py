#!/usr/bin/env python3
"""
Debug test script for the async validator function with better timing and error handling
"""

import asyncio
import time
import logging
import signal
import os

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

async def run_validator_with_timeout(prompt: str, port: int, timeout: int = 30) -> dict:
    """Run validator with timeout and better error handling"""
    
    logger.info(f"🚀 Starting validator on port {port}")
    start_time = time.time()
    
    try:
        # Build the validator command
        cmd = [
            "python", "subnet_accurate_validator_multigpu_ply.py",
            f'"{prompt}"',
            f'"{prompt}"',  # Same prompt for both
            "--endpoint", "/generate",
            "--port", str(port),
            "--num_inference_steps", "20",
            "--guidance_scale", "7.5",
            "--ss_steps", "10",
            "--slat_steps", "10",
            "--slat_guidance", "0.8",
            "--ss_guidance", "0.8"
        ]
        
        logger.info(f"   Command: {' '.join(cmd)}")
        
        # Run with timeout
        process = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        
        try:
            stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=timeout)
            end_time = time.time()
            execution_time = end_time - start_time
            
            logger.info(f"✅ Validator on port {port} completed in {execution_time:.2f}s")
            logger.info(f"   Exit code: {process.returncode}")
            
            if stdout:
                logger.info(f"   Stdout: {stdout.decode()[:200]}...")
            if stderr:
                logger.info(f"   Stderr: {stderr.decode()[:200]}...")
            
            # Check for results file
            results_file = f"subnet_validation_results_{port}.json"
            if os.path.exists(results_file):
                logger.info(f"   📁 Results file found: {results_file}")
                try:
                    import json
                    with open(results_file, 'r') as f:
                        results = json.load(f)
                    return {
                        'port': port,
                        'execution_time': execution_time,
                        'exit_code': process.returncode,
                        'results': results,
                        'status': 'success'
                    }
                except Exception as e:
                    logger.error(f"   ❌ Error reading results file: {e}")
                    return {
                        'port': port,
                        'execution_time': execution_time,
                        'exit_code': process.returncode,
                        'error': f"Results file read error: {e}",
                        'status': 'error'
                    }
            else:
                logger.warning(f"   ⚠️  Results file not found: {results_file}")
                return {
                    'port': port,
                    'execution_time': execution_time,
                    'exit_code': process.returncode,
                    'error': "Results file not found",
                    'status': 'error'
                }
                
        except asyncio.TimeoutError:
            logger.error(f"❌ Validator on port {port} timed out after {timeout}s")
            process.terminate()
            try:
                await asyncio.wait_for(process.wait(), timeout=5)
            except asyncio.TimeoutError:
                process.kill()
                await process.wait()
            
            return {
                'port': port,
                'execution_time': timeout,
                'error': "Timeout",
                'status': 'timeout'
            }
            
    except Exception as e:
        end_time = time.time()
        execution_time = end_time - start_time
        logger.error(f"❌ Validator on port {port} failed: {e}")
        return {
            'port': port,
            'execution_time': execution_time,
            'error': str(e),
            'status': 'error'
        }

async def test_parallel_validation():
    """Test parallel validation with detailed timing"""
    
    logger.info("🧪 Testing parallel validation with debug info")
    
    # Test parameters
    prompt = "A beautiful red rose in a glass vase"
    port1 = 8099
    port2 = 8097
    timeout = 30  # 30 seconds timeout
    
    logger.info(f"📝 Prompt: '{prompt}'")
    logger.info(f"🚀 Ports: {port1}, {port2}")
    logger.info(f"⏱️  Timeout: {timeout}s")
    
    # Start timing
    total_start = time.time()
    
    # Create both validator tasks
    logger.info(f"\n🚀 Starting both validators in parallel...")
    task1 = run_validator_with_timeout(prompt, port1, timeout)
    task2 = run_validator_with_timeout(prompt, port2, timeout)
    
    # Wait for both to complete
    logger.info("⏳ Waiting for both validators to complete...")
    parallel_start = time.time()
    
    try:
        results = await asyncio.gather(task1, task2, return_exceptions=True)
        parallel_end = time.time()
        
        total_end = time.time()
        
        # Calculate timing
        total_time = total_end - total_start
        parallel_time = parallel_end - parallel_start
        
        logger.info(f"\n{'='*60}")
        logger.info(f"⏱️  TIMING RESULTS")
        logger.info(f"{'='*60}")
        logger.info(f"   Total execution time: {total_time:.2f}s")
        logger.info(f"   Parallel execution time: {parallel_time:.2f}s")
        logger.info(f"   Setup overhead: {total_time - parallel_time:.2f}s")
        
        # Show detailed results
        logger.info(f"\n{'='*60}")
        logger.info(f"📊 VALIDATION RESULTS")
        logger.info(f"{'='*60}")
        
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"   ❌ Validator {i+1} raised exception: {result}")
                continue
                
            port = result['port']
            status = result['status']
            execution_time = result['execution_time']
            
            logger.info(f"\n🔍 Port {port} ({status}):")
            logger.info(f"   ⏱️  Execution time: {execution_time:.2f}s")
            
            if status == 'success':
                logger.info(f"   ✅ Exit code: {result['exit_code']}")
                if 'results' in result:
                    logger.info(f"   📁 Results loaded successfully")
                    # Show key results
                    results_data = result['results']
                    for key, value in results_data.items():
                        if key == 'ply_data' and isinstance(value, bytes):
                            logger.info(f"   📦 {key}: {len(value):,} bytes")
                        elif key == 'compression':
                            logger.info(f"   📦 {key}: {value}")
                        elif key == 'validation_engine_score':
                            logger.info(f"   🎯 {key}: {value}")
                        else:
                            logger.info(f"   📋 {key}: {value}")
            else:
                logger.error(f"   ❌ Error: {result.get('error', 'Unknown error')}")
                if 'exit_code' in result:
                    logger.info(f"   📊 Exit code: {result['exit_code']}")
        
        # Performance analysis
        logger.info(f"\n{'='*60}")
        logger.info(f"🏆 PERFORMANCE ANALYSIS")
        logger.info(f"{'='*60}")
        
        successful_results = [r for r in results if isinstance(r, dict) and r.get('status') == 'success']
        if len(successful_results) == 2:
            # Both succeeded
            max_time = max(r['execution_time'] for r in successful_results)
            min_time = min(r['execution_time'] for r in successful_results)
            
            logger.info(f"✅ Both validators completed successfully")
            logger.info(f"   Fastest: {min_time:.2f}s")
            logger.info(f"   Slowest: {max_time:.2f}s")
            logger.info(f"   Parallel time: {parallel_time:.2f}s")
            
            if parallel_time < max_time:
                logger.info(f"   ⚡ Parallel execution faster than slowest validator!")
            else:
                logger.info(f"   ⚠️  Parallel execution slower than expected")
                
        elif len(successful_results) == 1:
            logger.warning(f"⚠️  Only one validator completed successfully")
        else:
            logger.error(f"❌ No validators completed successfully")
        
        return results
        
    except Exception as e:
        logger.error(f"❌ Parallel execution failed: {e}")
        import traceback
        traceback.print_exc()
        return None

async def main():
    """Main test function"""
    logger.info("🧪 Starting async validator debug test")
    
    try:
        results = await test_parallel_validation()
        if results:
            logger.info("🎉 Test completed!")
        else:
            logger.error("❌ Test failed")
    except Exception as e:
        logger.error(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    # Set up signal handler for graceful shutdown
    def signal_handler(signum, frame):
        logger.info("🛑 Received interrupt signal, shutting down...")
        exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Run the test
    asyncio.run(main())
