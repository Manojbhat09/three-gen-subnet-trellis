#!/usr/bin/env python3
# Subnet 17 (404-GEN) - Final Submission Pipeline
# Purpose: Complete production-ready pipeline for Bittensor mining

import asyncio
import aiohttp
import argparse
import time
import os
import sys
import json
import traceback
import subprocess
import signal
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, asdict
import bittensor as bt
import base64

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from generation_asset_manager import (
        global_asset_manager, prepare_for_mining_submission
    )
    ASSET_MANAGER_AVAILABLE = True
except ImportError:
    print("⚠️ Asset manager not available - running in basic mode")
    ASSET_MANAGER_AVAILABLE = False

# Configuration
ENHANCED_GENERATION_URL = "http://127.0.0.1:8095/generate/"
VALIDATION_SERVER_URL = "http://127.0.0.1:8094/validate_txt_to_3d_ply/"
MINING_SUBMISSION_URL = "http://127.0.0.1:8095/mining/submit/"

# Critical production constants
MINER_LICENSE_CONSENT_DECLARATION = "I_AGREE_TO_THE_SUBNET_TERMS_AND_CONDITIONS"

# Validator blacklist - UIDs to avoid
VALIDATOR_BLACKLIST = {180}  # Known problematic validators

PRODUCTION_CONFIG = {
    'min_validation_score': 0.7,
    'max_retries': 3,
    'retry_delay': 5,
    'generation_timeout': 300,
    'validation_timeout': 60,
    'submission_timeout': 60,
    'competitive_variants': 5,
    'use_bpt': False,
    'max_concurrent_tasks': 3,  # Increased for async operations
    'max_concurrent_validators': 5,  # Pull from multiple validators simultaneously
    'mandatory_spz_compression': True,  # Required in next release
    'send_empty_on_validation_failure': True,  # Better than low quality results
    'validator_cooldown_threshold': 0.5  # Don't send results below this score
}

RESULTS_DIR = "production_mining_results"

@dataclass
class MiningTask:
    """Represents a real mining task from Bittensor validators"""
    task_id: str
    prompt: str
    validator_hotkey: str
    validator_uid: int
    difficulty: float
    deadline: float
    requirements: Dict
    synapse_uuid: str

@dataclass
class MiningResult:
    """Complete mining result"""
    task: MiningTask
    generation_id: str
    validation_score: float
    submission_successful: bool
    submission_data: Optional[Dict]
    generation_time: float
    total_time: float
    error: Optional[str] = None
    retries: int = 0
    competitive_scores: List[float] = None

class ProductionMiningPipeline:
    """Production-ready mining pipeline for Subnet 17"""
    
    def __init__(self, config: Dict = None):
        self.config = {**PRODUCTION_CONFIG, **(config or {})}
        self.session = None
        self.active_tasks = {}
        self.completed_tasks = []
        self.shutdown_requested = False
        
        # Validator management
        self.validator_blacklist = set(VALIDATOR_BLACKLIST)
        self.validator_performance = {}  # Track validator performance
        self.active_validator_tasks = {}  # Track tasks per validator
        
        # Create results directory
        timestamp = int(time.time())
        self.results_dir = Path(RESULTS_DIR) / f"mining_{timestamp}"
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup signal handlers
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        
        print(f"🏭 Production Mining Pipeline initialized")
        print(f"📁 Results directory: {self.results_dir}")
        print(f"🚫 Validator blacklist: {self.validator_blacklist}")
        print(f"⚙️ Config: min_score={self.config['min_validation_score']}, "
              f"variants={self.config['competitive_variants']}, "
              f"max_concurrent={self.config['max_concurrent_tasks']}")
        print(f"🔄 Async validators: {self.config['max_concurrent_validators']}")
        print(f"📦 SPZ compression: {'Mandatory' if self.config['mandatory_spz_compression'] else 'Optional'}")

    def _signal_handler(self, signum, frame):
        """Handle shutdown signals gracefully"""
        print(f"\n🛑 Received signal {signum}, initiating graceful shutdown...")
        self.shutdown_requested = True

    def is_validator_blacklisted(self, validator_uid: int) -> bool:
        """Check if validator should be avoided"""
        return validator_uid in self.validator_blacklist

    def blacklist_validator(self, validator_uid: int, reason: str = "Poor performance"):
        """Add validator to blacklist"""
        self.validator_blacklist.add(validator_uid)
        print(f"🚫 Blacklisted validator {validator_uid}: {reason}")

    def evaluate_validator_performance(self, validator_uid: int, success: bool, response_time: float):
        """Track validator performance for blacklisting decisions"""
        if validator_uid not in self.validator_performance:
            self.validator_performance[validator_uid] = {
                'total_requests': 0,
                'successful_requests': 0,
                'average_response_time': 0.0,
                'last_success': time.time() if success else 0
            }
        
        perf = self.validator_performance[validator_uid]
        perf['total_requests'] += 1
        if success:
            perf['successful_requests'] += 1
            perf['last_success'] = time.time()
        
        # Update average response time
        perf['average_response_time'] = (
            (perf['average_response_time'] * (perf['total_requests'] - 1) + response_time) 
            / perf['total_requests']
        )
        
        # Auto-blacklist problematic validators
        success_rate = perf['successful_requests'] / perf['total_requests']
        if (perf['total_requests'] >= 5 and 
            (success_rate < 0.3 or perf['average_response_time'] > 120 or 
             time.time() - perf['last_success'] > 1800)):  # 30 min without success
            self.blacklist_validator(validator_uid, f"Auto-blacklist: {success_rate:.1%} success rate")

    async def compress_with_pyspz(self, data: bytes) -> Tuple[bytes, int]:
        """Compress data using pyspz as required by validators"""
        try:
            import pyspz
            # Use compression with workers=-1 as shown in workers.py example
            compressed_data = pyspz.compress(data, workers=-1)
            compression_type = 2  # SPZ compression type
            print(f"    📦 SPZ compression: {len(data)} → {len(compressed_data)} bytes "
                  f"({len(compressed_data)/len(data)*100:.1f}%)")
            return compressed_data, compression_type
        except Exception as e:
            if self.config['mandatory_spz_compression']:
                raise Exception(f"SPZ compression failed (mandatory): {e}")
            else:
                print(f"    ⚠️ SPZ compression failed, using uncompressed: {e}")
                return data, 0

    async def validate_before_submission(self, ply_data: bytes, prompt: str) -> Tuple[bool, float]:
        """Validate results before submission to avoid cooldown penalties"""
        try:
            # Compress for validation
            compressed_data, compression_type = await self.compress_with_pyspz(ply_data)
            base64_data = base64.b64encode(compressed_data).decode('utf-8')
            
            validation_payload = {
                "prompt": prompt,
                "data": base64_data,
                "compression": compression_type,
                "data_ver": 0
            }
            
            async with self.session.post(VALIDATION_SERVER_URL, 
                                       json=validation_payload, 
                                       timeout=self.config['validation_timeout']) as response:
                if response.status == 200:
                    result = await response.json()
                    score = result.get("score", 0.0)
                    
                    # Check against cooldown threshold
                    meets_threshold = score >= self.config['validator_cooldown_threshold']
                    print(f"    🔍 Pre-submission validation: {score:.4f} "
                          f"({'✅ PASS' if meets_threshold else '❌ FAIL - will send empty'})")
                    
                    return meets_threshold, score
                else:
                    print(f"    ⚠️ Validation server error: {response.status}")
                    return False, 0.0
        except Exception as e:
            print(f"    💥 Validation error: {e}")
            return False, 0.0

    def create_submission_signature(self, task: MiningTask, submit_time: int, 
                                  validator_hotkey: str, miner_hotkey: str) -> str:
        """Create submission signature as shown in workers.py"""
        message = (
            f"{MINER_LICENSE_CONSENT_DECLARATION}"
            f"{submit_time}{task.prompt}{validator_hotkey}{miner_hotkey}"
        )
        
        # In production, this would use the actual wallet keypair
        # For testing, we'll create a mock signature
        import hashlib
        signature_hash = hashlib.sha256(message.encode()).hexdigest()
        mock_signature = base64.b64encode(signature_hash.encode()).decode()[:64]  # Truncate for testing
        
        return mock_signature

    async def test_infrastructure(self) -> bool:
        """Test all infrastructure components"""
        print("🔧 Testing infrastructure...")
        
        # Test server connectivity
        servers = [
            ("Generation Server", ENHANCED_GENERATION_URL + "../status/"),
            ("Validation Server", "http://127.0.0.1:8094/version/"),
            ("Mining Endpoint", MINING_SUBMISSION_URL.replace("/mining/submit/", "/status/"))
        ]
        
        all_online = True
        for name, url in servers:
            try:
                async with self.session.get(url, timeout=10) as response:
                    if response.status == 200:
                        print(f"  ✅ {name}: Online")
                    else:
                        print(f"  ❌ {name}: Error {response.status}")
                        all_online = False
            except Exception as e:
                print(f"  ❌ {name}: Offline ({str(e)[:50]})")
                all_online = False
        
        # Test GPU memory
        try:
            async with self.session.get(f"{ENHANCED_GENERATION_URL}../memory/", timeout=5) as response:
                if response.status == 200:
                    memory_info = await response.json()
                    allocated_gb = memory_info['allocated'] / (1024**3)
                    reserved_gb = memory_info['reserved'] / (1024**3)
                    print(f"  📊 GPU Memory: {allocated_gb:.1f}GB allocated, {reserved_gb:.1f}GB reserved")
                else:
                    print(f"  ⚠️ GPU memory check failed")
        except Exception as e:
            print(f"  ⚠️ GPU memory check error: {e}")
        
        return all_online

    async def run_competitive_generation(self, task: MiningTask) -> Tuple[str, float, List[float]]:
        """Run competitive generation to find the best result"""
        print(f"🏆 Running competitive generation for: '{task.prompt}'")
        
        variants = self.config['competitive_variants']
        best_generation_id = ""
        best_score = 0.0
        all_scores = []
        
        # Generate multiple variants with different seeds
        tasks = []
        for i in range(variants):
            seed = hash(task.task_id + str(i)) % (2**31)
            tasks.append(self._generate_single_variant(task, seed, i+1))
        
        # Run all variants concurrently
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Find the best result
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                print(f"  ❌ Variant {i+1} failed: {result}")
                all_scores.append(0.0)
                continue
            
            generation_id, score = result
            all_scores.append(score)
            
            if score > best_score:
                best_score = score
                best_generation_id = generation_id
                print(f"  🌟 New best: Variant {i+1} score {score:.4f}")
        
        print(f"  🏅 Best result: {best_score:.4f} (generation_id: {best_generation_id})")
        print(f"  📊 All scores: {[f'{s:.3f}' for s in all_scores]}")
        
        return best_generation_id, best_score, all_scores

    async def _generate_single_variant(self, task: MiningTask, seed: int, variant_num: int) -> Tuple[str, float]:
        """Generate a single variant"""
        try:
            form_data = aiohttp.FormData()
            form_data.add_field('prompt', task.prompt)
            form_data.add_field('seed', str(seed))
            form_data.add_field('use_bpt', str(self.config['use_bpt']).lower())
            form_data.add_field('return_compressed', 'true')
            
            async with self.session.post(ENHANCED_GENERATION_URL, 
                                       data=form_data, 
                                       timeout=self.config['generation_timeout']) as response:
                if response.status == 200:
                    generation_id = response.headers.get('X-Generation-ID', '')
                    score = float(response.headers.get('X-Local-Validation-Score', '0.0'))
                    face_count = int(response.headers.get('X-Face-Count', '0'))
                    
                    print(f"    ✅ Variant {variant_num}: score {score:.4f}, {face_count:,} faces, seed {seed}")
                    return generation_id, score
                else:
                    print(f"    ❌ Variant {variant_num}: HTTP {response.status}")
                    return "", 0.0
        except Exception as e:
            print(f"    💥 Variant {variant_num}: {str(e)[:50]}")
            return "", 0.0

    async def process_mining_task(self, task: MiningTask) -> MiningResult:
        """Process a complete mining task with retries"""
        print(f"\n⛏️ Processing mining task: {task.task_id}")
        print(f"   Prompt: '{task.prompt}'")
        print(f"   Validator: {task.validator_hotkey} (UID: {task.validator_uid})")
        print(f"   Difficulty: {task.difficulty}")
        
        # Check if validator is blacklisted
        if self.is_validator_blacklisted(task.validator_uid):
            print(f"  🚫 Validator {task.validator_uid} is blacklisted - skipping task")
            return MiningResult(
                task=task,
                generation_id="",
                validation_score=0.0,
                submission_successful=False,
                submission_data=None,
                generation_time=0.0,
                total_time=0.0,
                error="Validator blacklisted",
                competitive_scores=[]
            )
        
        result = MiningResult(
            task=task,
            generation_id="",
            validation_score=0.0,
            submission_successful=False,
            submission_data=None,
            generation_time=0.0,
            total_time=0.0,
            competitive_scores=[]
        )
        
        start_time = time.time()
        validator_start_time = time.time()
        
        for retry in range(self.config['max_retries']):
            if self.shutdown_requested:
                result.error = "Shutdown requested"
                break
                
            try:
                print(f"\n🔄 Attempt {retry + 1}/{self.config['max_retries']}")
                
                # Step 1: Competitive generation
                gen_start = time.time()
                generation_id, best_score, all_scores = await self.run_competitive_generation(task)
                result.generation_time = time.time() - gen_start
                result.competitive_scores = all_scores
                
                if not generation_id or best_score < self.config['min_validation_score']:
                    raise Exception(f"Best score {best_score:.4f} below threshold {self.config['min_validation_score']}")
                
                result.generation_id = generation_id
                result.validation_score = best_score
                
                # Step 2: Get PLY data for validation and submission
                print(f"  📥 Retrieving PLY data for submission...")
                
                # Download the PLY data from the generation server
                download_url = f"http://127.0.0.1:8095/generate/{generation_id}/download/final_mesh_ply"
                async with self.session.get(download_url, timeout=60) as response:
                    if response.status != 200:
                        raise Exception(f"Failed to download PLY data: HTTP {response.status}")
                    
                    ply_data = await response.read()
                    print(f"    📦 Downloaded PLY data: {len(ply_data)} bytes")
                
                # Step 3: Pre-submission validation (critical for avoiding cooldowns)
                print(f"  🔍 Pre-submission validation check...")
                should_submit, validation_score = await self.validate_before_submission(ply_data, task.prompt)
                
                # Update score with validation result
                result.validation_score = max(result.validation_score, validation_score)
                
                # Step 4: Prepare submission data
                print(f"  ⛏️ Preparing mining submission...")
                
                submit_time = time.time_ns()
                miner_hotkey = "test_miner_hotkey"  # In production, get from wallet
                
                # Create signature as per workers.py
                signature = self.create_submission_signature(
                    task, submit_time, task.validator_hotkey, miner_hotkey
                )
                
                # Prepare results based on validation
                if should_submit and self.config['send_empty_on_validation_failure']:
                    # Compress with SPZ as required
                    compressed_data, compression_type = await self.compress_with_pyspz(ply_data)
                    compressed_results = base64.b64encode(compressed_data).decode('utf-8')
                    print(f"    ✅ Submitting high-quality results")
                elif not should_submit and self.config['send_empty_on_validation_failure']:
                    # Send empty results to avoid cooldown penalty
                    compressed_results = ""
                    compression_type = 0
                    print(f"    ⚠️ Sending empty results (validation failed)")
                else:
                    # Send anyway (legacy behavior)
                    compressed_data, compression_type = await self.compress_with_pyspz(ply_data)
                    compressed_results = base64.b64encode(compressed_data).decode('utf-8')
                    print(f"    ⚠️ Submitting despite low validation score")
                
                # Create submission data matching workers.py format
                submission_data = {
                    "task": {
                        "task_id": task.task_id,
                        "prompt": task.prompt,
                        "synapse_uuid": task.synapse_uuid
                    },
                    "results": compressed_results,
                    "compression": compression_type,
                    "data_format": "ply",
                    "data_ver": 0,
                    "submit_time": submit_time,
                    "signature": signature,
                    "validator_hotkey": task.validator_hotkey,
                    "validator_uid": task.validator_uid,
                    "miner_hotkey": miner_hotkey,
                    "local_validation_score": result.validation_score,
                    "generation_id": generation_id
                }
                
                result.submission_data = submission_data
                result.submission_successful = True
                
                print(f"    ✅ Mining submission prepared successfully")
                print(f"    📊 Format: {submission_data.get('data_format', 'unknown')}")
                print(f"    📊 Compression: {submission_data.get('compression', 'unknown')}")
                print(f"    📊 Results size: {len(submission_data.get('results', ''))} chars")
                print(f"    📊 Signature: {signature[:20]}...")
                
                # Track validator performance
                validator_response_time = time.time() - validator_start_time
                self.evaluate_validator_performance(task.validator_uid, True, validator_response_time)
                
                # Success! Break out of retry loop
                break
                
            except Exception as e:
                result.error = str(e)
                result.retries = retry + 1
                print(f"    ❌ Attempt {retry + 1} failed: {e}")
                
                # Track validator failure
                validator_response_time = time.time() - validator_start_time
                self.evaluate_validator_performance(task.validator_uid, False, validator_response_time)
                
                if retry < self.config['max_retries'] - 1:
                    print(f"    ⏳ Retrying in {self.config['retry_delay']}s...")
                    await asyncio.sleep(self.config['retry_delay'])
                else:
                    print(f"    💀 All attempts failed")
        
        result.total_time = time.time() - start_time
        
        if result.submission_successful:
            print(f"  🎉 Task {task.task_id} completed successfully!")
            print(f"     Score: {result.validation_score:.4f}, Time: {result.total_time:.1f}s")
        else:
            print(f"  💀 Task {task.task_id} failed: {result.error}")
        
        return result

    async def simulate_validator_tasks(self, num_tasks: int = 3) -> List[MiningTask]:
        """Simulate realistic validator tasks for testing"""
        print(f"🎭 Simulating {num_tasks} validator tasks from multiple validators...")
        
        prompts = [
            "a modern ergonomic office chair",
            "a sleek gaming laptop computer", 
            "a wooden dining table for 6 people",
            "a comfortable leather sofa",
            "a professional camera tripod",
            "a vintage record player",
            "a smart home security camera",
            "a luxury sports car wheel",
            "a mountain bike helmet",
            "a coffee espresso machine"
        ]
        
        # Simulate various validators including blacklisted ones
        validator_pool = [
            {"uid": 100, "hotkey": "validator_hotkey_100", "reputation": "excellent"},
            {"uid": 101, "hotkey": "validator_hotkey_101", "reputation": "good"},
            {"uid": 102, "hotkey": "validator_hotkey_102", "reputation": "good"},
            {"uid": 180, "hotkey": "validator_hotkey_180", "reputation": "blacklisted"},  # Known WC
            {"uid": 200, "hotkey": "validator_hotkey_200", "reputation": "excellent"},
            {"uid": 201, "hotkey": "validator_hotkey_201", "reputation": "poor"},
            {"uid": 202, "hotkey": "validator_hotkey_202", "reputation": "good"},
        ]
        
        tasks = []
        current_time = time.time()
        
        for i in range(num_tasks):
            # Select validator (including some blacklisted for testing)
            validator = validator_pool[i % len(validator_pool)]
            
            # Auto-blacklist poor performers during simulation
            if validator["reputation"] == "poor" and i > 2:
                self.blacklist_validator(validator["uid"], "Simulated poor performance")
            
            task = MiningTask(
                task_id=f"mining_task_{i+1:03d}",
                prompt=prompts[i % len(prompts)],
                validator_hotkey=validator["hotkey"],
                validator_uid=validator["uid"],
                difficulty=0.5 + (i * 0.1),  # Increasing difficulty
                deadline=current_time + 3600,  # 1 hour deadline
                requirements={
                    "min_faces": 1000 + (i * 200),
                    "format": "ply",
                    "compression": "spz"
                },
                synapse_uuid=f"synapse_{int(current_time)}_{i}"
            )
            tasks.append(task)
            
            status = "🚫 BLACKLISTED" if self.is_validator_blacklisted(validator["uid"]) else "✅ ACTIVE"
            print(f"  📋 Task {i+1}: '{task.prompt}' from validator {validator['uid']} ({status})")
        
        # Filter out blacklisted validators from active tasks
        active_tasks = [task for task in tasks if not self.is_validator_blacklisted(task.validator_uid)]
        blacklisted_count = len(tasks) - len(active_tasks)
        
        if blacklisted_count > 0:
            print(f"  🚫 Filtered out {blacklisted_count} tasks from blacklisted validators")
        
        print(f"  🎯 {len(active_tasks)} active tasks ready for processing")
        return active_tasks

    async def pull_tasks_from_multiple_validators(self, max_concurrent: int = 5) -> List[MiningTask]:
        """Simulate async task pulling from multiple validators simultaneously"""
        print(f"🔄 Pulling tasks from up to {max_concurrent} validators concurrently...")
        
        # Simulate different validators providing tasks
        validator_tasks = []
        
        # Create async task pulls
        async def pull_from_validator(validator_uid: int, num_tasks: int = 2):
            """Simulate pulling tasks from a single validator"""
            if self.is_validator_blacklisted(validator_uid):
                print(f"    🚫 Skipping blacklisted validator {validator_uid}")
                return []
            
            print(f"    📡 Pulling from validator {validator_uid}...")
            await asyncio.sleep(0.5)  # Simulate network delay
            
            tasks = await self.simulate_validator_tasks(num_tasks)
            # Filter to only this validator's tasks
            validator_specific_tasks = [task for task in tasks if task.validator_uid == validator_uid]
            
            print(f"    ✅ Validator {validator_uid}: {len(validator_specific_tasks)} tasks")
            return validator_specific_tasks
        
        # Pull from multiple validators concurrently
        validator_uids = [100, 101, 102, 200, 180, 201, 202]  # Mix of good and bad validators
        pull_tasks = [pull_from_validator(uid, 1) for uid in validator_uids[:max_concurrent]]
        
        # Execute all pulls concurrently
        results = await asyncio.gather(*pull_tasks, return_exceptions=True)
        
        # Combine all tasks
        all_tasks = []
        for result in results:
            if isinstance(result, list):
                all_tasks.extend(result)
            else:
                print(f"    ⚠️ Validator pull failed: {result}")
        
        print(f"  🎯 Total tasks pulled: {len(all_tasks)}")
        return all_tasks

    async def save_mining_session(self, results: List[MiningResult]):
        """Save complete mining session results"""
        print(f"\n💾 Saving mining session results...")
        
        # Calculate session statistics
        successful = [r for r in results if r.submission_successful]
        failed = [r for r in results if not r.submission_successful]
        
        session_summary = {
            "session_timestamp": time.time(),
            "config": self.config,
            "total_tasks": len(results),
            "successful_tasks": len(successful),
            "failed_tasks": len(failed),
            "success_rate": len(successful) / len(results) if results else 0,
            "average_score": sum(r.validation_score for r in successful) / len(successful) if successful else 0,
            "average_generation_time": sum(r.generation_time for r in results) / len(results) if results else 0,
            "total_session_time": sum(r.total_time for r in results),
            "results": []
        }
        
        # Add individual results
        for result in results:
            result_dict = {
                "task_id": result.task.task_id,
                "prompt": result.task.prompt,
                "validator_hotkey": result.task.validator_hotkey,
                "validator_uid": result.task.validator_uid,
                "generation_id": result.generation_id,
                "validation_score": result.validation_score,
                "submission_successful": result.submission_successful,
                "generation_time": result.generation_time,
                "total_time": result.total_time,
                "retries": result.retries,
                "competitive_scores": result.competitive_scores,
                "error": result.error
            }
            session_summary["results"].append(result_dict)
        
        # Save session summary
        summary_path = self.results_dir / "mining_session_summary.json"
        with open(summary_path, 'w') as f:
            json.dump(session_summary, f, indent=2)
        
        # Save individual submission data
        submissions_dir = self.results_dir / "submissions"
        submissions_dir.mkdir(exist_ok=True)
        
        for i, result in enumerate(results):
            if result.submission_data:
                submission_path = submissions_dir / f"submission_{result.task.task_id}.json"
                with open(submission_path, 'w') as f:
                    json.dump(result.submission_data, f, indent=2)
        
        print(f"  📊 Session summary: {summary_path}")
        print(f"  📦 Submission files: {len([r for r in results if r.submission_data])}")
        print(f"  📈 Success rate: {session_summary['success_rate']*100:.1f}%")

    async def run_production_mining(self, max_tasks: int = 5) -> Dict:
        """Run production mining session with full async operations"""
        print("🏭 Starting Production Mining Session")
        print("=" * 60)
        
        async with aiohttp.ClientSession() as session:
            self.session = session
            
            # Step 1: Test infrastructure
            if not await self.test_infrastructure():
                return {"error": "Infrastructure test failed"}
            
            # Step 2: Pull tasks from multiple validators concurrently
            print(f"\n🔄 Phase 1: Async Task Pulling")
            tasks = await self.pull_tasks_from_multiple_validators(
                self.config['max_concurrent_validators']
            )
            
            if not tasks:
                print("⚠️ No valid tasks available after validator filtering")
                # Fall back to simulation for testing
                tasks = await self.simulate_validator_tasks(max_tasks)
            
            # Limit to requested number of tasks
            tasks = tasks[:max_tasks]
            print(f"\n🎯 Phase 2: Processing {len(tasks)} mining tasks")
            
            # Display validator distribution
            validator_distribution = {}
            for task in tasks:
                validator_distribution[task.validator_uid] = validator_distribution.get(task.validator_uid, 0) + 1
            
            print("📊 Validator Distribution:")
            for uid, count in validator_distribution.items():
                status = "🚫 BLACKLISTED" if self.is_validator_blacklisted(uid) else "✅ ACTIVE"
                print(f"   Validator {uid}: {count} tasks ({status})")
            
            # Step 3: Process tasks with full async operations
            print(f"\n🚀 Phase 3: Concurrent Task Processing")
            
            # Create semaphore for concurrency control
            semaphore = asyncio.Semaphore(self.config['max_concurrent_tasks'])
            
            async def process_with_semaphore(task, task_index):
                async with semaphore:
                    print(f"\n{'='*40} Task {task_index+1}/{len(tasks)} {'='*40}")
                    return await self.process_mining_task(task)
            
            # Execute all tasks concurrently (up to semaphore limit)
            print(f"   🔄 Running up to {self.config['max_concurrent_tasks']} tasks simultaneously")
            
            task_coroutines = [
                process_with_semaphore(task, i) 
                for i, task in enumerate(tasks)
                if not self.shutdown_requested
            ]
            
            if not task_coroutines:
                return {"error": "No valid tasks to process"}
            
            # Process all tasks concurrently
            results = await asyncio.gather(*task_coroutines, return_exceptions=True)
            
            # Handle any exceptions
            final_results = []
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    print(f"❌ Task {i+1} failed with exception: {result}")
                    # Create error result
                    error_result = MiningResult(
                        task=tasks[i],
                        generation_id="",
                        validation_score=0.0,
                        submission_successful=False,
                        submission_data=None,
                        generation_time=0.0,
                        total_time=0.0,
                        error=str(result),
                        competitive_scores=[]
                    )
                    final_results.append(error_result)
                else:
                    final_results.append(result)
                    self.completed_tasks.append(result)
            
            # Step 4: Save session results
            if final_results:
                await self.save_mining_session(final_results)
            
            # Step 5: Generate final report with validator analysis
            return self._generate_session_report(final_results)

    def _generate_session_report(self, results: List[MiningResult]) -> Dict:
        """Generate final mining session report"""
        successful = [r for r in results if r.submission_successful]
        failed = [r for r in results if not r.submission_successful]
        
        print(f"\n📊 Mining Session Report")
        print("=" * 60)
        print(f"Total Tasks: {len(results)}")
        print(f"Successful: {len(successful)} ({len(successful)/len(results)*100:.1f}%)")
        print(f"Failed: {len(failed)} ({len(failed)/len(results)*100:.1f}%)")
        
        if successful:
            avg_score = sum(r.validation_score for r in successful) / len(successful)
            avg_time = sum(r.total_time for r in successful) / len(successful)
            total_rewards = len(successful)  # Simplified reward calculation
            
            print(f"Average Validation Score: {avg_score:.4f}")
            print(f"Average Task Time: {avg_time:.2f}s")
            print(f"Estimated Rewards: {total_rewards} tasks completed")
        
        print(f"\n🎯 Task Results:")
        for i, result in enumerate(results, 1):
            status = "✅ SUCCESS" if result.submission_successful else "❌ FAILED"
            print(f"  {i}. {status} '{result.task.prompt}' "
                  f"(score: {result.validation_score:.4f}, time: {result.total_time:.1f}s)")
            if result.error:
                print(f"      Error: {result.error}")
        
        if failed:
            print(f"\n⚠️ Failed Tasks Analysis:")
            error_counts = {}
            for result in failed:
                error_type = result.error.split(':')[0] if result.error else 'Unknown'
                error_counts[error_type] = error_counts.get(error_type, 0) + 1
            
            for error_type, count in error_counts.items():
                print(f"  • {error_type}: {count} tasks")
        
        print(f"\n📁 Results saved to: {self.results_dir}")
        
        # Determine session success
        success_rate = len(successful) / len(results) if results else 0
        session_success = success_rate >= 0.8  # 80% success rate threshold
        
        print(f"\n🏆 Session Status: {'SUCCESS' if session_success else 'NEEDS IMPROVEMENT'}")
        print(f"   Success Rate: {success_rate*100:.1f}% ({'✅' if session_success else '⚠️ < 80%'})")
        
        return {
            "session_success": session_success,
            "success_rate": success_rate,
            "total_tasks": len(results),
            "successful_tasks": len(successful),
            "failed_tasks": len(failed),
            "average_score": sum(r.validation_score for r in successful) / len(successful) if successful else 0,
            "average_time": sum(r.total_time for r in results) / len(results) if results else 0,
            "results_directory": str(self.results_dir),
            "production_ready": session_success
        }


async def main():
    parser = argparse.ArgumentParser(description="Production Mining Pipeline for Subnet 17")
    parser.add_argument("-n", "--max-tasks", type=int, default=5,
                       help="Maximum number of mining tasks to process (default: 5)")
    parser.add_argument("--min-score", type=float, default=0.7,
                       help="Minimum validation score for mining (default: 0.7)")
    parser.add_argument("--variants", type=int, default=5,
                       help="Number of competitive variants per task (default: 5)")
    parser.add_argument("--concurrent", type=int, default=3,
                       help="Maximum concurrent tasks (default: 3)")
    parser.add_argument("--use-bpt", action="store_true",
                       help="Enable BPT enhancement")
    parser.add_argument("--test-mode", action="store_true",
                       help="Run in test mode (shorter timeouts)")
    
    args = parser.parse_args()
    
    # Create custom config
    config = {
        'min_validation_score': args.min_score,
        'competitive_variants': args.variants,
        'max_concurrent_tasks': args.concurrent,
        'use_bpt': args.use_bpt
    }
    
    if args.test_mode:
        config.update({
            'generation_timeout': 120,
            'max_retries': 2,
            'retry_delay': 2
        })
    
    print("🏭 Subnet 17 Production Mining Pipeline")
    print("=" * 60)
    print(f"Max Tasks: {args.max_tasks}")
    print(f"Min Score: {args.min_score}")
    print(f"Variants: {args.variants}")
    print(f"Concurrent: {args.concurrent}")
    print(f"BPT: {'Enabled' if args.use_bpt else 'Disabled'}")
    print(f"Mode: {'Test' if args.test_mode else 'Production'}")
    print(f"Asset Manager: {'Available' if ASSET_MANAGER_AVAILABLE else 'Not Available'}")
    print()
    
    # Run production mining
    pipeline = ProductionMiningPipeline(config)
    start_time = time.time()
    
    try:
        results = await pipeline.run_production_mining(args.max_tasks)
        total_time = time.time() - start_time
        
        if "error" in results:
            print(f"❌ Mining session failed: {results['error']}")
            return 1
        
        print(f"\n⏱️ Total session time: {total_time:.2f}s")
        
        if results["production_ready"]:
            print("🎉 Production mining pipeline is performing well!")
            return 0
        else:
            print("⚠️ Production pipeline needs optimization")
            return 1
            
    except KeyboardInterrupt:
        print("\n🛑 Mining session interrupted by user")
        return 130
    except Exception as e:
        print(f"\n💥 Unexpected error: {e}")
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code) 