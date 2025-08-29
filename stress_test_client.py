#!/usr/bin/env python3
"""
Stress Test Client for Validation Simulation Server
Purpose: Test the simulation server to find edge cases and failure modes

Features:
- Concurrent miner simulation
- Cooldown violation testing
- Rate limit testing
- Network failure simulation
- Performance benchmarking
- Edge case discovery
"""

import asyncio
import aiohttp
import time
import random
import json
import logging
from typing import List, Dict, Optional, Any
from dataclasses import dataclass
from collections import defaultdict
import statistics

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('stress_test.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

@dataclass
class StressTestConfig:
    """Configuration for stress testing"""
    server_url: str = "http://localhost:8094"
    num_miners: int = 10
    requests_per_miner: int = 50
    concurrent_requests: int = 20
    delay_between_requests: float = 0.1
    enable_cooldown_violations: bool = True
    enable_rate_limit_testing: bool = True
    enable_network_failures: bool = True
    test_duration_seconds: int = 300  # 5 minutes

class SimulatedMiner:
    """Simulates a single miner for stress testing"""
    
    def __init__(self, miner_id: int, hotkey: str, config: StressTestConfig):
        self.miner_id = miner_id
        self.hotkey = hotkey
        self.config = config
        
        # State tracking
        self.tasks_pulled = 0
        self.tasks_submitted = 0
        self.successful_submissions = 0
        self.cooldown_violations = 0
        self.current_task = None
        self.last_cooldown_until = 0
        
        # Performance tracking
        self.response_times = []
        self.errors = []
        self.cooldown_periods = []
        
        logger.info(f"🆕 Miner {miner_id} ({hotkey}) initialized")
    
    async def pull_task(self, session: aiohttp.ClientSession) -> Dict:
        """Pull a task from the simulation server"""
        try:
            start_time = time.time()
            
            payload = {
                "hotkey": self.hotkey
            }
            
            async with session.post(
                f"{self.config.server_url}/pull_task",
                json=payload,
                timeout=30
            ) as response:
                response_time = time.time() - start_time
                self.response_times.append(response_time)
                
                if response.status == 200:
                    result = await response.json()
                    
                    if result.get('task'):
                        self.current_task = result['task']
                        self.tasks_pulled += 1
                        logger.debug(f"Miner {self.miner_id}: Task pulled - '{result['task']['prompt'][:30]}...'")
                        
                        # Track cooldown info
                        if result.get('cooldown_until'):
                            self.last_cooldown_until = result['cooldown_until']
                        
                        if result.get('cooldown_violations'):
                            self.cooldown_violations = result['cooldown_violations']
                        
                        return result
                    else:
                        # No task (probably on cooldown)
                        cooldown_until = result.get('cooldown_until', 0)
                        if cooldown_until > 0:
                            remaining = max(0, cooldown_until - time.time())
                            logger.debug(f"Miner {self.miner_id}: On cooldown for {remaining:.1f}s")
                        
                        return result
                else:
                    error_text = await response.text()
                    error_msg = f"HTTP {response.status}: {error_text}"
                    self.errors.append(error_msg)
                    logger.warning(f"Miner {self.miner_id}: Pull task failed - {error_msg}")
                    return {"error": error_msg}
                    
        except Exception as e:
            error_msg = f"Exception: {str(e)}"
            self.errors.append(error_msg)
            logger.error(f"Miner {self.miner_id}: Pull task exception - {error_msg}")
            return {"error": error_msg}
    
    async def submit_results(self, session: aiohttp.ClientSession) -> Dict:
        """Submit results for current task"""
        if not self.current_task:
            logger.debug(f"Miner {self.miner_id}: No task to submit")
            return {"error": "No task assigned"}
        
        try:
            start_time = time.time()
            
            # Simulate 3D model data (base64 encoded)
            simulated_data = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPhfDwAChwGA60e6kgAAAABJRU5ErkJggg=="
            
            payload = {
                "hotkey": self.hotkey,
                "task_id": self.current_task['id'],
                "prompt": self.current_task['prompt'],
                "results": simulated_data,
                "submit_time": int(time.time_ns()),
                "signature": "test_signature_" + str(random.randint(1000, 9999))
            }
            
            async with session.post(
                f"{self.config.server_url}/submit_results",
                json=payload,
                timeout=30
            ) as response:
                response_time = time.time() - start_time
                self.response_times.append(response_time)
                
                if response.status == 200:
                    result = await response.json()
                    self.tasks_submitted += 1
                    
                    # Check if submission was successful
                    feedback = result.get('feedback', {})
                    if not feedback.get('validation_failed', True):
                        self.successful_submissions += 1
                        logger.debug(f"Miner {self.miner_id}: Results submitted successfully")
                    else:
                        logger.debug(f"Miner {self.miner_id}: Results failed validation")
                    
                    # Track cooldown
                    if result.get('cooldown_until'):
                        cooldown_duration = result['cooldown_until'] - time.time()
                        if cooldown_duration > 0:
                            self.cooldown_periods.append(cooldown_duration)
                    
                    return result
                else:
                    error_text = await response.text()
                    error_msg = f"HTTP {response.status}: {error_text}"
                    self.errors.append(error_msg)
                    logger.warning(f"Miner {self.miner_id}: Submit results failed - {error_msg}")
                    return {"error": error_msg}
                    
        except Exception as e:
            error_msg = f"Exception: {str(e)}"
            self.errors.append(error_msg)
            logger.error(f"Miner {self.miner_id}: Submit results exception - {error_msg}")
            return {"error": error_msg}
    
    def get_statistics(self) -> Dict:
        """Get miner statistics"""
        return {
            "miner_id": self.miner_id,
            "hotkey": self.hotkey,
            "tasks_pulled": self.tasks_pulled,
            "tasks_submitted": self.tasks_submitted,
            "successful_submissions": self.successful_submissions,
            "cooldown_violations": self.cooldown_violations,
            "success_rate": self.successful_submissions / max(self.tasks_submitted, 1),
            "avg_response_time": statistics.mean(self.response_times) if self.response_times else 0,
            "total_errors": len(self.errors),
            "avg_cooldown_duration": statistics.mean(self.cooldown_periods) if self.cooldown_periods else 0
        }

class StressTestOrchestrator:
    """Orchestrates stress testing across multiple miners"""
    
    def __init__(self, config: StressTestConfig):
        self.config = config
        self.miners: List[SimulatedMiner] = []
        self.test_start_time = None
        self.test_results = defaultdict(list)
        
        # Generate miner hotkeys
        for i in range(config.num_miners):
            hotkey = f"test_miner_{i:03d}_" + "x" * 48  # Simulate real hotkey format
            miner = SimulatedMiner(i, hotkey, config)
            self.miners.append(miner)
        
        logger.info(f"🚀 Stress test orchestrator initialized with {config.num_miners} miners")
    
    async def run_miner_cycle(self, miner: SimulatedMiner, session: aiohttp.ClientSession):
        """Run a complete cycle for a single miner"""
        try:
            # Pull task
            pull_result = await miner.pull_task(session)
            
            if pull_result.get('error'):
                self.test_results['pull_errors'].append({
                    'miner_id': miner.miner_id,
                    'error': pull_result['error'],
                    'timestamp': time.time()
                })
                return
            
            # If we got a task, submit results
            if pull_result.get('task'):
                # Simulate processing time
                await asyncio.sleep(random.uniform(0.1, 2.0))
                
                # Submit results
                submit_result = await miner.submit_results(session)
                
                if submit_result.get('error'):
                    self.test_results['submit_errors'].append({
                        'miner_id': miner.miner_id,
                        'error': submit_result['error'],
                        'timestamp': time.time()
                    })
                else:
                    self.test_results['successful_submissions'].append({
                        'miner_id': miner.miner_id,
                        'timestamp': time.time(),
                        'feedback': submit_result.get('feedback', {})
                    })
            
            # Track cooldown violations if enabled
            if self.config.enable_cooldown_violations and random.random() < 0.1:  # 10% chance
                # Simulate cooldown violation by pulling again immediately
                logger.debug(f"Miner {miner.miner_id}: Simulating cooldown violation")
                await miner.pull_task(session)
                
        except Exception as e:
            logger.error(f"Miner {miner.miner_id}: Cycle error - {e}")
            self.test_results['cycle_errors'].append({
                'miner_id': miner.miner_id,
                'error': str(e),
                'timestamp': time.time()
            })
    
    async def run_concurrent_test(self, session: aiohttp.ClientSession):
        """Run concurrent stress test"""
        logger.info("🚀 Starting concurrent stress test")
        
        # Create tasks for all miners
        tasks = []
        for miner in self.miners:
            for _ in range(self.config.requests_per_miner):
                task = self.run_miner_cycle(miner, session)
                tasks.append(task)
        
        # Run with concurrency limit
        semaphore = asyncio.Semaphore(self.config.concurrent_requests)
        
        async def limited_task(task):
            async with semaphore:
                await task
        
        # Execute all tasks
        await asyncio.gather(*[limited_task(task) for task in tasks])
        
        logger.info("✅ Concurrent stress test completed")
    
    async def run_rate_limit_test(self, session: aiohttp.ClientSession):
        """Test rate limiting by sending many requests quickly"""
        logger.info("🚀 Starting rate limit test")
        
        # Send requests as fast as possible
        tasks = []
        for i in range(200):  # Send 200 requests
            task = self.miners[i % len(self.miners)].pull_task(session)
            tasks.append(task)
        
        # Execute all tasks concurrently
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Count rate limit responses
        rate_limited = sum(1 for r in results if isinstance(r, dict) and r.get('error', '').startswith('HTTP 429'))
        logger.info(f"✅ Rate limit test completed: {rate_limited}/{len(results)} requests rate limited")
        
        self.test_results['rate_limit_test'] = {
            'total_requests': len(results),
            'rate_limited': rate_limited,
            'timestamp': time.time()
        }
    
    async def run_network_failure_test(self, session: aiohttp.ClientSession):
        """Test network failure handling"""
        logger.info("🚀 Starting network failure test")
        
        # Simulate network issues by using invalid URLs
        invalid_url = "http://invalid-host:9999"
        invalid_session = aiohttp.ClientSession()
        
        try:
            # Try to connect to invalid host
            async with invalid_session.post(f"{invalid_url}/pull_task", json={"hotkey": "test"}) as response:
                pass
        except Exception as e:
            logger.info(f"✅ Network failure test: Expected error - {type(e).__name__}")
            self.test_results['network_failure_test'] = {
                'expected_error': str(e),
                'timestamp': time.time()
            }
        finally:
            await invalid_session.close()
    
    async def run_stress_test(self):
        """Run complete stress test suite"""
        self.test_start_time = time.time()
        logger.info("🚀 Starting comprehensive stress test suite")
        
        async with aiohttp.ClientSession() as session:
            # Run different types of tests
            await self.run_concurrent_test(session)
            
            if self.config.enable_rate_limit_testing:
                await self.run_rate_limit_test(session)
            
            if self.config.enable_network_failures:
                await self.run_network_failure_test(session)
        
        # Calculate test duration
        test_duration = time.time() - self.test_start_time
        logger.info(f"✅ Stress test suite completed in {test_duration:.2f}s")
        
        # Generate comprehensive report
        self.generate_test_report()
    
    def generate_test_report(self):
        """Generate comprehensive test report"""
        logger.info("📊 Generating stress test report...")
        
        # Calculate overall statistics
        total_requests = sum(m.tasks_pulled for m in self.miners)
        total_submissions = sum(m.tasks_submitted for m in self.miners)
        total_successful = sum(m.successful_submissions for m in self.miners)
        total_errors = sum(len(m.errors) for m in self.miners)
        total_cooldown_violations = sum(m.cooldown_violations for m in self.miners)
        
        # Calculate response time statistics
        all_response_times = []
        for miner in self.miners:
            all_response_times.extend(miner.response_times)
        
        avg_response_time = statistics.mean(all_response_times) if all_response_times else 0
        max_response_time = max(all_response_times) if all_response_times else 0
        min_response_time = min(all_response_times) if all_response_times else 0
        
        # Generate report
        report = {
            "test_summary": {
                "test_duration_seconds": time.time() - self.test_start_time,
                "num_miners": len(self.miners),
                "total_requests": total_requests,
                "total_submissions": total_submissions,
                "total_successful": total_successful,
                "total_errors": total_errors,
                "total_cooldown_violations": total_cooldown_violations,
                "success_rate": total_successful / max(total_submissions, 1)
            },
            "performance_metrics": {
                "avg_response_time": avg_response_time,
                "min_response_time": min_response_time,
                "max_response_time": max_response_time,
                "total_response_time": sum(all_response_times)
            },
            "error_analysis": {
                "pull_errors": len(self.test_results['pull_errors']),
                "submit_errors": len(self.test_results['submit_errors']),
                "cycle_errors": len(self.test_results['cycle_errors']),
                "rate_limit_responses": self.test_results.get('rate_limit_test', {}).get('rate_limited', 0)
            },
            "miner_details": [miner.get_statistics() for miner in self.miners],
            "test_results": dict(self.test_results)
        }
        
        # Save report to file
        report_file = f"stress_test_report_{int(time.time())}.json"
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"📊 Test report saved to {report_file}")
        
        # Print summary
        print("\n" + "="*60)
        print("🚀 STRESS TEST REPORT SUMMARY")
        print("="*60)
        print(f"Test Duration: {report['test_summary']['test_duration_seconds']:.2f}s")
        print(f"Total Requests: {report['test_summary']['total_requests']}")
        print(f"Success Rate: {report['test_summary']['success_rate']:.2%}")
        print(f"Avg Response Time: {report['performance_metrics']['avg_response_time']:.3f}s")
        print(f"Total Errors: {report['test_summary']['total_errors']}")
        print(f"Cooldown Violations: {report['test_summary']['total_cooldown_violations']}")
        print(f"Rate Limited: {report['error_analysis']['rate_limit_responses']}")
        print("="*60)

async def main():
    """Main entry point"""
    # Configuration
    config = StressTestConfig(
        server_url="http://localhost:8094",
        num_miners=20,
        requests_per_miner=30,
        concurrent_requests=15,
        delay_between_requests=0.05,
        enable_cooldown_violations=True,
        enable_rate_limit_testing=True,
        enable_network_failures=True,
        test_duration_seconds=180  # 3 minutes
    )
    
    # Create orchestrator
    orchestrator = StressTestOrchestrator(config)
    
    # Run stress test
    await orchestrator.run_stress_test()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("🛑 Stress test stopped by user")
    except Exception as e:
        logger.error(f"❌ Stress test error: {e}")
        import traceback
        traceback.print_exc()





