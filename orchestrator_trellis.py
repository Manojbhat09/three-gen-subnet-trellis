#!/usr/bin/env python3
"""
TRELLIS Orchestrator - Subnet 17 (404-GEN)
Purpose: Orchestrate TRELLIS 3D generation pipeline with options for harvest, validate, and submit

Features:
- Harvest tasks from validators via Bittensor
- Generate 3D models using FLUX + TRELLIS pipeline
- Optional validation of generated models
- Submit results back to validators
- SPZ compression and decompression support
- Configurable operation modes
"""

import asyncio
import json
import time
import random
import argparse
import requests
import base64
import logging
import traceback
import os
import sys
from pathlib import Path
from typing import List, Dict, Optional, Any
from datetime import datetime
from dataclasses import dataclass, asdict
import sqlite3
import hashlib

# Make bittensor optional for environments without it
try:
    import bittensor as bt
    BITTENSOR_AVAILABLE = True
except ImportError:
    print("⚠️ Bittensor not available - harvest and submit features disabled")
    BITTENSOR_AVAILABLE = False
    bt = None

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('trellis_orchestrator.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Configuration
ORCHESTRATOR_CONFIG = {
    # Bittensor settings
    'wallet_name': 'test2m3b2',
    'hotkey_name': 't2m3b21',
    'netuid': 17,
    'target_validators': [128, 212, 142, 27, 124, 173, 49, 253, 163, 81],  # High-stake validators
    'max_tasks': 10,
    
    # Generation server settings
    'generation_server_url': 'http://localhost:8096',
    'validation_server_url': 'http://localhost:10006',
    
    # Output settings
    'output_dir': './trellis_orchestrator_outputs',
    'save_intermediate_results': True,
    'auto_compress_results': True,
    
    # Operation modes
    'harvest_tasks': True,
    'validate_generations': True,
    'submit_results': True,
    'use_harvested_tasks': True,  # Use harvested tasks or generate random prompts
    
    # Generation settings
    'default_prompts': [
        "a blue ceramic vase with intricate patterns",
        "a wooden chair with carved details",
        "a red sports car with sleek design",
        "a glass bottle with cork stopper",
        "a metal lantern with stained glass",
        "a stone statue of a lion",
        "a leather boot with laces",
        "a ceramic coffee mug with handle",
        "a wooden toy train with wheels",
        "a crystal chandelier with multiple tiers"
    ],
    'generation_timeout': 300,  # 5 minutes
    'validation_timeout': 120,  # 2 minutes
    'submission_timeout': 60,   # 1 minute
}

# Data classes and Database from continuous_trellis_orchestrator.py for unified state
@dataclass
class TaskRecord:
    """Record of a task with full metadata"""
    task_id: str
    prompt: str
    prompt_hash: str
    validator_uid: int
    validator_hotkey: str
    validator_stake: float
    validation_threshold: float
    pulled_at: float
    processed_at: Optional[float] = None
    submitted_at: Optional[float] = None
    generation_time: Optional[float] = None
    validation_time: Optional[float] = None
    local_validation_score: Optional[float] = None
    submission_success: bool = False
    feedback_received: bool = False
    task_fidelity_score: Optional[float] = None
    average_fidelity_score: Optional[float] = None
    current_miner_reward: Optional[float] = None
    validation_failed: Optional[bool] = None
    generations_in_window: Optional[int] = None
    ply_file_path: Optional[str] = None
    compressed_file_path: Optional[str] = None

class TaskDatabase:
    """SQLite database for task tracking and deduplication - SHARED"""
    
    def __init__(self, db_path: str = "trellis_mining_tasks.db"):
        self.db_path = db_path
        self.init_database()
    
    def init_database(self):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS tasks (
                task_id TEXT PRIMARY KEY,
                prompt TEXT NOT NULL,
                prompt_hash TEXT NOT NULL,
                validator_uid INTEGER NOT NULL,
                validator_hotkey TEXT NOT NULL,
                validator_stake REAL NOT NULL,
                validation_threshold REAL NOT NULL,
                pulled_at REAL NOT NULL,
                processed_at REAL,
                submitted_at REAL,
                generation_time REAL,
                validation_time REAL,
                local_validation_score REAL,
                submission_success BOOLEAN DEFAULT FALSE,
                feedback_received BOOLEAN DEFAULT FALSE,
                task_fidelity_score REAL,
                average_fidelity_score REAL,
                current_miner_reward REAL,
                validation_failed BOOLEAN,
                generations_in_window INTEGER,
                ply_file_path TEXT,
                compressed_file_path TEXT,
                created_at REAL DEFAULT (strftime('%s', 'now'))
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS recent_prompts (
                prompt_hash TEXT NOT NULL,
                validator_uid INTEGER NOT NULL,
                prompt TEXT NOT NULL,
                pulled_at REAL NOT NULL,
                PRIMARY KEY (prompt_hash, validator_uid)
            )
        ''')
        conn.commit()
        conn.close()

    def is_duplicate_prompt(self, prompt: str, validator_uid: int, hours_window: int = 24) -> bool:
        prompt_hash = hashlib.sha256(prompt.encode()).hexdigest()
        cutoff_time = time.time() - (hours_window * 3600)
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT COUNT(*) FROM recent_prompts 
            WHERE prompt_hash = ? AND validator_uid = ? AND pulled_at > ?
        ''', (prompt_hash, validator_uid, cutoff_time))
        
        count = cursor.fetchone()[0]
        conn.close()
        return count > 0
    
    def add_recent_prompt(self, prompt: str, validator_uid: int):
        prompt_hash = hashlib.sha256(prompt.encode()).hexdigest()
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('''
            INSERT OR REPLACE INTO recent_prompts 
            (prompt_hash, validator_uid, prompt, pulled_at)
            VALUES (?, ?, ?, ?)
        ''', (prompt_hash, validator_uid, prompt, time.time()))
        conn.commit()
        conn.close()
    
    def save_task(self, task: TaskRecord):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('''
            INSERT OR REPLACE INTO tasks 
            (task_id, prompt, prompt_hash, validator_uid, validator_hotkey, validator_stake,
             validation_threshold, pulled_at, processed_at, submitted_at, generation_time,
             validation_time, local_validation_score, submission_success, feedback_received,
             task_fidelity_score, average_fidelity_score, current_miner_reward,
             validation_failed, generations_in_window, ply_file_path, compressed_file_path)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', asdict(task).values())
        conn.commit()
        conn.close()

class TrellisOrchestrator:
    def __init__(self, config: Dict[str, Any]):
        # Merge with default config to ensure all keys are present
        self.config = ORCHESTRATOR_CONFIG.copy()
        self.config.update(config)
        self.logger = logger
        
        # Setup output directory and unified database
        self.output_dir = Path(self.config['output_dir'])
        self.output_dir.mkdir(exist_ok=True)
        self.db = TaskDatabase()
        
        # Bittensor components (initialized when needed)
        self.wallet = None
        self.subtensor = None
        self.dendrite = None
        self.metagraph = None
        
        # Task management
        self.harvested_tasks = []
        self.completed_tasks = []
        self.failed_tasks = []
        
        # Statistics
        self.stats = {
            'tasks_harvested': 0,
            'generations_attempted': 0,
            'generations_successful': 0,
            'validations_attempted': 0,
            'validations_successful': 0,
            'submissions_attempted': 0,
            'submissions_successful': 0,
            'total_generation_time': 0.0,
            'total_validation_time': 0.0,
            'best_validation_score': 0.0,
            'average_validation_score': 0.0
        }
        
        self.logger.info("🎯 TRELLIS Orchestrator initialized")
        self.logger.info(f"   Output directory: {self.output_dir}")
        self.logger.info(f"   Generation server: {self.config['generation_server_url']}")
        self.logger.info(f"   Validation server: {self.config['validation_server_url']}")
        self.logger.info(f"   Unified Database: {self.db.db_path}")

    def generate_3d_model(self, prompt: str, seed: int = None) -> Optional[Dict[str, Any]]:
        """Generate 3D model using TRELLIS server"""
        if seed is None:
            seed = random.randint(0, 2**32 - 1)
        
        self.logger.info(f"🎨 Generating 3D model for: '{prompt}' (seed: {seed})")
        
        try:
            generation_start = time.time()
            
            # Call TRELLIS generation server
            response = requests.post(
                f"{self.config['generation_server_url']}/generate/",
                data={
                    'prompt': prompt,
                    'seed': seed,
                    'return_compressed': True  # Get SPZ compressed data
                },
                timeout=self.config['generation_timeout']
            )
            
            generation_time = time.time() - generation_start
            
            if response.status_code == 200:
                # Get compressed PLY data
                compressed_ply_data = response.content
                
                # Get metadata from headers
                generation_seed = response.headers.get('X-Generation-Seed', str(seed))
                compression_ratio = response.headers.get('X-Compression-Ratio', 'unknown')
                
                self.logger.info(f"✅ Generation successful in {generation_time:.2f}s")
                self.logger.info(f"   Compressed size: {len(compressed_ply_data):,} bytes")
                self.logger.info(f"   Compression ratio: {compression_ratio}")
                
                # Save compressed PLY
                if self.config['save_intermediate_results']:
                    timestamp = int(time.time())
                    output_file = self.output_dir / f"trellis_model_{timestamp}_{seed}.ply.spz"
                    with open(output_file, 'wb') as f:
                        f.write(compressed_ply_data)
                    self.logger.info(f"💾 Saved compressed model: {output_file}")
                
                # Update stats
                self.stats['generations_attempted'] += 1
                self.stats['generations_successful'] += 1
                self.stats['total_generation_time'] += generation_time
                
                return {
                    'status': 'success',
                    'prompt': prompt,
                    'seed': int(generation_seed),
                    'ply_data': compressed_ply_data,
                    'generation_time': generation_time,
                    'compression_ratio': compression_ratio,
                    'timestamp': time.time()
                }
            else:
                self.logger.error(f"❌ Generation failed: HTTP {response.status_code}")
                self.logger.error(f"   Response: {response.text[:200]}...")
                
                self.stats['generations_attempted'] += 1
                return None
                
        except Exception as e:
            self.logger.error(f"❌ Generation exception: {e}")
            traceback.print_exc()
            self.stats['generations_attempted'] += 1
            return None

    def validate_model(self, prompt: str, compressed_ply_data: bytes) -> Optional[Dict[str, Any]]:
        """Validate generated model"""
        if not self.config['validate_generations']:
            self.logger.info("⏭️ Validation disabled")
            return {'status': 'skipped', 'score': 0.0}
        
        self.logger.info(f"📊 Validating model for: '{prompt[:50]}...'")
        
        try:
            validation_start = time.time()
            
            # Decompress PLY data for validation
            try:
                import pyspz
                ply_data = pyspz.decompress(compressed_ply_data)
                self.logger.info(f"   📤 Decompressed: {len(ply_data):,} bytes")
            except ImportError:
                self.logger.error("❌ pyspz not available - cannot decompress PLY data")
                return None
            except Exception as e:
                self.logger.error(f"❌ Decompression failed: {e}")
                return None
            
            # Convert to base64 as expected by validation server
            encoded_data = base64.b64encode(ply_data).decode('utf-8')
            self.logger.info(f"   📤 Base64 encoded size: {len(encoded_data):,} chars")
            
            request_data = {
                "prompt": prompt,
                "data": encoded_data,
                "compression": 0,  # 0=none (already decompressed)
                "generate_preview": False,
                "preview_score_threshold": 0.8
            }
            
            response = requests.post(
                f"{self.config['validation_server_url']}/validate_txt_to_3d_ply/",
                json=request_data,
                timeout=self.config['validation_timeout']
            )
            
            validation_time = time.time() - validation_start
            
            if response.status_code == 200:
                result = response.json()
                validation_score = result.get("score", 0.0)
                quality_metrics = {
                    "iqa": result.get("iqa", 0.0),
                    "alignment": result.get("alignment_score", 0.0),
                    "ssim": result.get("ssim", 0.0),
                    "lpips": result.get("lpips", 0.0)
                }
                
                self.logger.info(f"✅ Validation completed in {validation_time:.2f}s")
                self.logger.info(f"   Score: {validation_score:.4f}")
                self.logger.info(f"   Quality metrics: {quality_metrics}")
                
                # Update stats
                self.stats['validations_attempted'] += 1
                self.stats['validations_successful'] += 1
                self.stats['total_validation_time'] += validation_time
                
                if validation_score > self.stats['best_validation_score']:
                    self.stats['best_validation_score'] = validation_score
                
                # Update average (use safe division)
                if self.stats['validations_successful'] > 1:
                    self.stats['average_validation_score'] = (
                        (self.stats['average_validation_score'] * (self.stats['validations_successful'] - 1) + validation_score)
                        / self.stats['validations_successful']
                    )
                else:
                    self.stats['average_validation_score'] = validation_score
                
                return {
                    'status': 'success',
                    'validation_score': validation_score,
                    'quality_metrics': quality_metrics,
                    'validation_time': validation_time,
                    'timestamp': time.time()
                }
            else:
                self.logger.error(f"❌ Validation failed: HTTP {response.status_code}")
                self.logger.error(f"   Response: {response.text[:200]}...")
                
                self.stats['validations_attempted'] += 1
                return None
                
        except Exception as e:
            self.logger.error(f"❌ Validation exception: {e}")
            traceback.print_exc()
            self.stats['validations_attempted'] += 1
            return None

    def _setup_bittensor(self):
        """Setup Bittensor components for task harvesting and submission"""
        if not BITTENSOR_AVAILABLE:
            self.logger.error("❌ Bittensor not available")
            return False
            
        try:
            if self.wallet is None:
                self.wallet = bt.wallet(
                    name=self.config['wallet_name'],
                    hotkey=self.config['hotkey_name']
                )
                self.logger.info(f"✅ Wallet loaded: {self.wallet.hotkey.ss58_address}")
            
            if self.subtensor is None:
                self.subtensor = bt.subtensor(network="finney")
                self.logger.info("✅ Subtensor connected")
            
            if self.dendrite is None:
                self.dendrite = bt.dendrite(wallet=self.wallet)
                self.logger.info("✅ Dendrite initialized")
            
            if self.metagraph is None:
                self.metagraph = self.subtensor.metagraph(self.config['netuid'])
                self.logger.info(f"✅ Metagraph loaded (netuid: {self.config['netuid']})")
            
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Bittensor setup failed: {e}")
            return False

    async def harvest_tasks(self) -> List[Dict[str, Any]]:
        """Harvest tasks from validators"""
        if not self.config['harvest_tasks']:
            self.logger.info("⏭️ Task harvesting disabled")
            return []
        
        self.logger.info("🎯 Starting task harvesting...")
        
        if not self._setup_bittensor():
            self.logger.error("❌ Bittensor setup failed, cannot harvest tasks")
            return []
        
        try:
            # Import protocol
            from neurons.common.protocol import PullTask
        except ImportError:
            self.logger.error("❌ Could not import PullTask protocol")
            return []
        
        tasks = []
        target_validators = self.config['target_validators']
        max_tasks = self.config['max_tasks']
        
        for uid in target_validators:
            if uid >= len(self.metagraph.neurons):
                continue
                
            neuron = self.metagraph.neurons[uid]
            if not neuron.validator_permit or neuron.stake <= 0:
                continue
            
            try:
                self.logger.info(f"📡 Pulling from UID {uid} ({float(neuron.stake):.1f} TAO)...")
                
                synapse = PullTask()
                # Use call instead of forward for compatibility with workers.py
                response = await self.dendrite.call(
                    target_axon=neuron.axon_info,
                    synapse=synapse,
                    deserialize=False,
                    timeout=30.0
                )
                
                if response and hasattr(response, 'task') and response.task and response.task.prompt:
                    # Check for duplicates before adding
                    if self.db.is_duplicate_prompt(response.task.prompt, uid):
                        self.logger.info(f"   ⏭️ Skipping duplicate prompt from UID {uid}")
                        continue

                    response_obj = response
                    task_info = {
                        'task_id': response_obj.task.id,
                        'prompt': response_obj.task.prompt,
                        'validator_uid': uid,
                        'validator_hotkey': neuron.hotkey,
                        'validator_stake': float(neuron.stake),
                        'validation_threshold': getattr(response_obj, 'validation_threshold', 0.6),
                        'pulled_at': time.time(),
                        'pulled_datetime': datetime.now().isoformat()
                    }
                    
                    tasks.append(task_info)
                    self.logger.info(f"   ✅ Task: '{task_info['prompt'][:50]}...'")
                    self.logger.info(f"   Threshold: {task_info['validation_threshold']}")
                    self.db.add_recent_prompt(task_info['prompt'], uid)
                    
                    if len(tasks) >= max_tasks:
                        break
                else:
                    self.logger.info(f"   ⚠️ No task available")
                    
            except Exception as e:
                self.logger.error(f"   ❌ Error pulling from UID {uid}: {e}")
        
        # Save harvested tasks (for logging, not for state)
        if tasks:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            tasks_file = self.output_dir / f"harvested_tasks_log_{timestamp}.json"
            
            with open(tasks_file, 'w') as f:
                json.dump(tasks, f, indent=2)
            
            self.logger.info(f"💾 Saved {len(tasks)} tasks to {tasks_file}")
            self.stats['tasks_harvested'] = len(tasks)
        
        self.harvested_tasks = tasks
        return tasks

    async def submit_result(self, task_info: Dict[str, Any], generation_result: Dict[str, Any], 
                          validation_result: Dict[str, Any] = None) -> bool:
        """Submit result back to validator with proper miner declaration"""
        if not self.config['submit_results']:
            self.logger.info("⏭️ Result submission disabled")
            return True, None
        
        self.logger.info(f"📤 Submitting result for task: {task_info['task_id']}")
        
        try:
            if not self._setup_bittensor():
                self.logger.error("❌ Bittensor setup failed, cannot submit result")
                return False, None
            
            # Import protocol
            try:
                from neurons.common.protocol import SubmitResults, Task
            except ImportError:
                self.logger.error("❌ Could not import SubmitResults protocol")
                return False, None
            
            # Get validator info
            validator_uid = task_info['validator_uid']
            if validator_uid is None:
                self.logger.error("❌ Cannot submit result for default task (no validator UID)")
                return False, None
            if validator_uid >= len(self.metagraph.neurons):
                self.logger.error(f"❌ Validator UID {validator_uid} not found in metagraph")
                return False, None
                
            neuron = self.metagraph.neurons[validator_uid]
            
            # Create task object
            task_obj = Task(
                id=task_info['task_id'],
                prompt=task_info['prompt']
            )
            
            # Get data from TRELLIS server
            ply_data = generation_result['ply_data'] # These are SPZ-compressed bytes
            
            # The 'results' field in SubmitResults synapse requires a base64-encoded STRING.
            # The TRELLIS server already provides SPZ-compressed bytes, so we just need to base64 encode them.
            
            compression_ratio = generation_result.get('compression_ratio', 'unknown')
            if compression_ratio == 'unknown':
                 self.logger.warning("   ⚠️ Server did not return compression ratio. Assuming data is SPZ-compressed.")

            self.logger.info(f"   📦 Using SPZ-compressed data from server ({len(ply_data):,} bytes)")
            encoded_data = base64.b64encode(ply_data).decode('utf-8')
            
            # Create submission with proper miner declaration
            submit_time = time.time_ns()
            
            # Import miner license declaration
            try:
                from neurons.common.miner_license_consent_declaration import MINER_LICENSE_CONSENT_DECLARATION
            except ImportError:
                MINER_LICENSE_CONSENT_DECLARATION = "I, as a miner on SN17, have obtained all licenses, rights and consents required to use, reproduce, modify, display, distribute and make available my submitted results to this subnet and its end users"

            
            # Create signature (miner declaration) - exactly like workers.py
            message = f"{MINER_LICENSE_CONSENT_DECLARATION}{submit_time}{task_info['prompt']}{neuron.hotkey}{self.wallet.hotkey.ss58_address}"
            signature = base64.b64encode(self.dendrite.keypair.sign(message)).decode('utf-8')
            
            
            synapse = SubmitResults(
                task=task_obj,
                results=encoded_data,
                # data_format="ply",
                data_ver=0,
                compression=2,  # spz compression
                submit_time=submit_time,
                signature=signature
            )
            
            synapse.timeout = self.config['submission_timeout']
            
            start_time = time.time()
            
            # Submit to validator - use call like workers.py
            response = await self.dendrite.call(
                target_axon=neuron.axon_info,
                synapse=synapse,
                deserialize=False,
                timeout=self.config['submission_timeout']
            )
            
            submit_time_elapsed = time.time() - start_time
            
            if response and hasattr(response, 'feedback') and response.feedback:
                response_obj = response
                self.logger.info(f"✅ Result submitted successfully to UID {validator_uid} ({submit_time_elapsed:.2f}s)")
                self.logger.info(f"   Validation failed: {response_obj.feedback.validation_failed}")
                self.logger.info(f"   Task fidelity score: {response_obj.feedback.task_fidelity_score}")
                self.logger.info(f"   Average fidelity score: {response_obj.feedback.average_fidelity_score}")
                self.logger.info(f"   Current miner reward: {response_obj.feedback.current_miner_reward}")
                
                self.stats['submissions_attempted'] += 1
                self.stats['submissions_successful'] += 1
                return True, response_obj.feedback
            else:
                self.logger.error(f"❌ Result submission failed to UID {validator_uid}")
                self.stats['submissions_attempted'] += 1
                return False, None
                
        except Exception as e:
            self.logger.error(f"❌ Submission exception: {e}")
            traceback.print_exc()
            self.stats['submissions_attempted'] += 1
            return False, None

    def get_tasks_to_process(self) -> List[Dict[str, Any]]:
        """Get tasks to process (harvested or default)"""
        if self.config['harvest_tasks'] and self.harvested_tasks:
            self.logger.info(f"📋 Using {len(self.harvested_tasks)} harvested tasks")
            return self.harvested_tasks
        elif self.config['harvest_tasks'] and not self.harvested_tasks:
            # Harvesting was enabled but no tasks were found
            self.logger.warning("⚠️ Harvesting enabled but no tasks were harvested from validators")
            self.logger.info("   This is normal - validators may not have tasks available")
            self.logger.info("   Falling back to default prompts for testing")
            
            # Generate tasks from default prompts
            default_prompts = self.config['default_prompts'][:self.config['max_tasks']]
            tasks = []
            
            for i, prompt in enumerate(default_prompts):
                task = {
                    'task_id': f"default_{int(time.time())}_{i}",
                    'prompt': prompt,
                    'validator_uid': None,
                    'validator_hotkey': None,
                    'validator_stake': 0.0,
                    'validation_threshold': 0.6,
                    'pulled_at': time.time(),
                    'pulled_datetime': datetime.now().isoformat(),
                    'is_default': True
                }
                tasks.append(task)
            
            self.logger.info(f"📋 Using {len(tasks)} default tasks (fallback)")
            return tasks
        else:
            # Generate tasks from default prompts
            default_prompts = self.config['default_prompts'][:self.config['max_tasks']]
            tasks = []
            
            for i, prompt in enumerate(default_prompts):
                task = {
                    'task_id': f"default_{int(time.time())}_{i}",
                    'prompt': prompt,
                    'validator_uid': None,
                    'validator_hotkey': None,
                    'validator_stake': 0.0,
                    'validation_threshold': 0.6,
                    'pulled_at': time.time(),
                    'pulled_datetime': datetime.now().isoformat(),
                    'is_default': True
                }
                tasks.append(task)
            
            self.logger.info(f"📋 Using {len(tasks)} default tasks")
            return tasks

    async def process_task(self, task_info: Dict[str, Any]) -> Dict[str, Any]:
        """Process a single task (generate, validate, submit)"""
        self.logger.info(f"🔄 Processing task: {task_info['task_id']}")
        self.logger.info(f"   Prompt: '{task_info['prompt']}'")
        
        task_start = time.time()
        
        # Step 1: Generate 3D model
        generation_result = self.generate_3d_model(task_info['prompt'])
        if generation_result is None:
            self.logger.error(f"❌ Generation failed for task: {task_info['task_id']}")
            self.failed_tasks.append(task_info)
            return {'status': 'failed', 'step': 'generation'}
        
        # Create TaskRecord for unified logging
        task_record = TaskRecord(
            task_id=task_info['task_id'],
            prompt=task_info['prompt'],
            prompt_hash=hashlib.sha256(task_info['prompt'].encode()).hexdigest(),
            validator_uid=task_info['validator_uid'],
            validator_hotkey=task_info['validator_hotkey'],
            validator_stake=task_info['validator_stake'],
            validation_threshold=task_info['validation_threshold'],
            pulled_at=task_info['pulled_at'],
            processed_at=time.time(),
            generation_time=generation_result['generation_time'],
        )

        # Step 2: Validate model (optional)
        validation_result = None
        if self.config['validate_generations']:
            validation_result = self.validate_model(
                task_info['prompt'], 
                generation_result['ply_data']
            )
            if validation_result:
                task_record.validation_time = validation_result.get('validation_time')
                task_record.local_validation_score = validation_result.get('validation_score')

        # Step 3: Submit result (optional)
        submission_success = True
        if self.config['submit_results'] and not task_info.get('is_default', False):
            submission_success, feedback = await self.submit_result(task_info, generation_result, validation_result)
            task_record.submitted_at = time.time()
            task_record.submission_success = submission_success
            if feedback:
                task_record.feedback_received = True
                task_record.task_fidelity_score = feedback.task_fidelity_score
                task_record.average_fidelity_score = feedback.average_fidelity_score
                task_record.current_miner_reward = feedback.current_miner_reward
                task_record.validation_failed = feedback.validation_failed
                task_record.generations_in_window = feedback.generations_within_the_window

        # Save to unified database
        self.db.save_task(task_record)
        
        task_time = time.time() - task_start
        
        # Compile task result
        task_result = {
            'task_info': task_info,
            'generation_result': generation_result,
            'validation_result': validation_result,
            'submission_success': submission_success,
            'total_time': task_time,
            'timestamp': time.time(),
            'status': 'completed'
        }
        
        self.completed_tasks.append(task_result)
        
        self.logger.info(f"✅ Task completed in {task_time:.2f}s: {task_info['task_id']}")
        if validation_result and validation_result.get('validation_score'):
            self.logger.info(f"   Validation score: {validation_result['validation_score']:.4f}")
        
        return task_result

    async def run_orchestration(self):
        """Run the complete orchestration pipeline"""
        self.logger.info("🚀 Starting TRELLIS orchestration...")
        
        orchestration_start = time.time()
        
        # Step 1: Harvest tasks (if enabled)
        if self.config['harvest_tasks']:
            await self.harvest_tasks()
        
        # Step 2: Get tasks to process
        tasks_to_process = self.get_tasks_to_process()
        
        if not tasks_to_process:
            self.logger.warning("⚠️ No tasks to process")
            return
        
        self.logger.info(f"📋 Processing {len(tasks_to_process)} tasks...")
        
        # Step 3: Process tasks
        for i, task_info in enumerate(tasks_to_process, 1):
            self.logger.info(f"🔄 Processing task {i}/{len(tasks_to_process)}")
            
            try:
                await self.process_task(task_info)
            except Exception as e:
                self.logger.error(f"❌ Task processing failed: {e}")
                traceback.print_exc()
                self.failed_tasks.append(task_info)
        
        orchestration_time = time.time() - orchestration_start
        
        # Final summary is now based on DB, not in-memory lists
        self.logger.info(f"🎉 Orchestration completed in {orchestration_time:.2f}s")
        self.print_summary()

    async def save_results(self, orchestration_time: float):
        """Save orchestration results and statistics - DEPRECATED in favor of DB"""
        self.logger.info("💾 Results are now saved continuously to the SQLite database.")

    def print_summary(self):
        """Print orchestration summary"""
        self.logger.info("📊 ORCHESTRATION SUMMARY")
        self.logger.info("="*60)
        self.logger.info(f"Tasks harvested: {self.stats['tasks_harvested']}")
        self.logger.info(f"Generations attempted: {self.stats['generations_attempted']}")
        self.logger.info(f"Generations successful: {self.stats['generations_successful']}")
        self.logger.info(f"Validations attempted: {self.stats['validations_attempted']}")
        self.logger.info(f"Validations successful: {self.stats['validations_successful']}")
        self.logger.info(f"Submissions attempted: {self.stats['submissions_attempted']}")
        self.logger.info(f"Submissions successful: {self.stats['submissions_successful']}")
        
        if self.stats['generations_successful'] > 0:
            avg_gen_time = self.stats['total_generation_time'] / self.stats['generations_successful']
            self.logger.info(f"Average generation time: {avg_gen_time:.2f}s")
        
        if self.stats['validations_successful'] > 0:
            avg_val_time = self.stats['total_validation_time'] / self.stats['validations_successful']
            self.logger.info(f"Average validation time: {avg_val_time:.2f}s")
            self.logger.info(f"Best validation score: {self.stats['best_validation_score']:.4f}")
            self.logger.info(f"Average validation score: {self.stats['average_validation_score']:.4f}")
        
        self.logger.info(f"Completed tasks: {len(self.completed_tasks)}")
        self.logger.info(f"Failed tasks: {len(self.failed_tasks)}")
        
        success_rate = (len(self.completed_tasks) / max(1, len(self.completed_tasks) + len(self.failed_tasks))) * 100
        self.logger.info(f"Success rate: {success_rate:.1f}%")

async def main():
    parser = argparse.ArgumentParser(description="TRELLIS Orchestrator for Subnet 17")
    parser.add_argument("--harvest", action="store_true", help="Enable task harvesting")
    parser.add_argument("--no-harvest", action="store_true", help="Disable task harvesting")
    parser.add_argument("--validate", action="store_true", help="Enable validation")
    parser.add_argument("--no-validate", action="store_true", help="Disable validation")
    parser.add_argument("--submit", action="store_true", help="Enable result submission")
    parser.add_argument("--no-submit", action="store_true", help="Disable result submission")
    parser.add_argument("--use-defaults", action="store_true", help="Use default prompts instead of harvested tasks")
    parser.add_argument("--generation-server", default="http://localhost:8096", help="TRELLIS generation server URL")
    parser.add_argument("--validation-server", default="http://localhost:10006", help="Validation server URL")
    parser.add_argument("--output-dir", default="./trellis_orchestrator_outputs", help="Output directory")
    parser.add_argument("--max-tasks", type=int, default=10, help="Maximum tasks to process")
    
    args = parser.parse_args()
    
    # Update config based on arguments
    config = ORCHESTRATOR_CONFIG.copy()
    
    if args.harvest:
        config['harvest_tasks'] = True
    elif args.no_harvest:
        config['harvest_tasks'] = False
    
    if args.validate:
        config['validate_generations'] = True
    elif args.no_validate:
        config['validate_generations'] = False
    
    if args.submit:
        config['submit_results'] = True
    elif args.no_submit:
        config['submit_results'] = False
    
    if args.use_defaults:
        config['use_harvested_tasks'] = False
    
    config['generation_server_url'] = args.generation_server
    config['validation_server_url'] = args.validation_server
    config['output_dir'] = args.output_dir
    config['max_tasks'] = args.max_tasks
    
    # Create and run orchestrator
    orchestrator = TrellisOrchestrator(config)
    
    try:
        await orchestrator.run_orchestration()
    except KeyboardInterrupt:
        logger.info("🛑 Orchestration interrupted by user")
    except Exception as e:
        logger.error(f"❌ Orchestration failed: {e}")
        traceback.print_exc()
    finally:
        # No need to call save_results, as it's done continuously
        logger.info("🏁 TRELLIS Orchestrator finished")

if __name__ == "__main__":
    asyncio.run(main())
