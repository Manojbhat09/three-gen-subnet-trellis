#!/usr/bin/env python3
"""
Continuous TRELLIS Orchestrator - Subnet 17 (404-GEN)
Purpose: Continuous mining with intelligent task deduplication and idle validation

Features:
- Continuous task harvesting with prompt deduplication
- Real-time feedback processing and score tracking
- Automatic validation during idle periods
- Comprehensive statistics and JSON logging
- Always-on generation server integration
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
import hashlib
import sqlite3
from pathlib import Path
from typing import List, Dict, Optional, Any, Set, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict

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
        logging.FileHandler('continuous_trellis.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

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
    # Feedback scores
    task_fidelity_score: Optional[float] = None
    average_fidelity_score: Optional[float] = None
    current_miner_reward: Optional[float] = None
    validation_failed: Optional[bool] = None
    generations_in_window: Optional[int] = None
    # File paths
    ply_file_path: Optional[str] = None
    compressed_file_path: Optional[str] = None

@dataclass 
class ValidatorState:
    """State tracking for each validator"""
    uid: int
    hotkey: str
    stake: float
    trust: float
    consensus: float
    last_task_pull: Optional[float] = None
    last_task_received: Optional[float] = None
    cooldown_until: Optional[float] = None
    total_tasks_pulled: int = 0
    total_tasks_received: int = 0
    total_tasks_submitted: int = 0
    total_successful_submissions: int = 0
    average_score: float = 0.0
    recent_prompts: Set[str] = None
    is_active: bool = True

    def __post_init__(self):
        if self.recent_prompts is None:
            self.recent_prompts = set()

class TaskDatabase:
    """SQLite database for task tracking and deduplication"""
    
    def __init__(self, db_path: str = "continuous_trellis_tasks.db"):
        self.db_path = db_path
        self.init_database()
    
    def init_database(self):
        """Initialize the database schema"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Tasks table with comprehensive tracking
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
        
        # Validators table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS validators (
                uid INTEGER PRIMARY KEY,
                hotkey TEXT NOT NULL,
                stake REAL NOT NULL,
                trust REAL NOT NULL,
                consensus REAL NOT NULL,
                last_task_pull REAL,
                last_task_received REAL,
                cooldown_until REAL,
                total_tasks_pulled INTEGER DEFAULT 0,
                total_tasks_received INTEGER DEFAULT 0,
                total_tasks_submitted INTEGER DEFAULT 0,
                total_successful_submissions INTEGER DEFAULT 0,
                average_score REAL DEFAULT 0.0,
                is_active BOOLEAN DEFAULT TRUE,
                updated_at REAL DEFAULT (strftime('%s', 'now'))
            )
        ''')
        
        # Recent prompts table for deduplication
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS recent_prompts (
                prompt_hash TEXT NOT NULL,
                validator_uid INTEGER NOT NULL,
                prompt TEXT NOT NULL,
                pulled_at REAL NOT NULL,
                PRIMARY KEY (prompt_hash, validator_uid)
            )
        ''')
        
        # Statistics table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS statistics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp REAL NOT NULL,
                total_tasks_pulled INTEGER DEFAULT 0,
                total_tasks_processed INTEGER DEFAULT 0,
                total_successful_generations INTEGER DEFAULT 0,
                total_successful_validations INTEGER DEFAULT 0,
                total_successful_submissions INTEGER DEFAULT 0,
                average_generation_time REAL DEFAULT 0.0,
                average_validation_time REAL DEFAULT 0.0,
                average_local_score REAL DEFAULT 0.0,
                average_feedback_score REAL DEFAULT 0.0,
                total_rewards REAL DEFAULT 0.0,
                uptime_hours REAL DEFAULT 0.0
            )
        ''')
        
        # Create indexes
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_prompt_hash ON tasks(prompt_hash)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_validator_uid ON tasks(validator_uid)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_pulled_at ON tasks(pulled_at)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_recent_prompts_time ON recent_prompts(pulled_at)')
        
        conn.commit()
        conn.close()
    
    def is_duplicate_prompt(self, prompt: str, validator_uid: int, hours_window: int = 24) -> bool:
        """Check if this prompt was recently processed successfully from this validator"""
        prompt_hash = hashlib.sha256(prompt.encode()).hexdigest()
        cutoff_time = time.time() - (hours_window * 3600)
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Check if we have a successful submission for this prompt from this validator recently
        cursor.execute('''
            SELECT COUNT(*) FROM tasks 
            WHERE prompt_hash = ? AND validator_uid = ? AND pulled_at > ? 
            AND submission_success = 1 AND feedback_received = 1
        ''', (prompt_hash, validator_uid, cutoff_time))
        
        successful_submissions = cursor.fetchone()[0]
        
        # Check for very recent failed attempts (to avoid immediate retry loops)
        # Only check last 15 minutes for failed attempts to allow quicker retries
        recent_failed_cutoff = time.time() - (0.25 * 3600)  # 15 minutes for failed attempts
        cursor.execute('''
            SELECT COUNT(*) FROM tasks 
            WHERE prompt_hash = ? AND validator_uid = ? AND pulled_at > ?
            AND (submission_success = 0 OR feedback_received = 0 OR processed_at IS NULL)
        ''', (prompt_hash, validator_uid, recent_failed_cutoff))
        
        recent_failed_attempts = cursor.fetchone()[0]
        
        conn.close()
        
        # Don't duplicate if we successfully submitted recently (24 hour window)
        if successful_submissions > 0:
            return True
        
        # Only skip if there was a very recent failed attempt (15 minute cooldown)
        if recent_failed_attempts > 0:
            return True
            
        return False
    
    def add_recent_prompt(self, prompt: str, validator_uid: int):
        """Add prompt to recent prompts tracking"""
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
        """Save task record to database"""
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
        ''', (
            task.task_id, task.prompt, task.prompt_hash, task.validator_uid,
            task.validator_hotkey, task.validator_stake, task.validation_threshold,
            task.pulled_at, task.processed_at, task.submitted_at, task.generation_time,
            task.validation_time, task.local_validation_score, task.submission_success,
            task.feedback_received, task.task_fidelity_score, task.average_fidelity_score,
            task.current_miner_reward, task.validation_failed, task.generations_in_window,
            task.ply_file_path, task.compressed_file_path
        ))
        
        conn.commit()
        conn.close()
    
    def get_recent_unvalidated_tasks(self, hours: int = 2) -> List[TaskRecord]:
        """Get recent tasks that haven't been locally validated"""
        cutoff_time = time.time() - (hours * 3600)
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT * FROM tasks 
            WHERE processed_at > ? AND local_validation_score IS NULL
            ORDER BY processed_at DESC
        ''', (cutoff_time,))
        
        rows = cursor.fetchall()
        conn.close()
        
        tasks = []
        for row in rows:
            task = TaskRecord(
                task_id=row[0], prompt=row[1], prompt_hash=row[2],
                validator_uid=row[3], validator_hotkey=row[4], validator_stake=row[5],
                validation_threshold=row[6], pulled_at=row[7], processed_at=row[8],
                submitted_at=row[9], generation_time=row[10], validation_time=row[11],
                local_validation_score=row[12], submission_success=bool(row[13]),
                feedback_received=bool(row[14]), task_fidelity_score=row[15],
                average_fidelity_score=row[16], current_miner_reward=row[17],
                validation_failed=bool(row[18]) if row[18] is not None else None,
                generations_in_window=row[19], ply_file_path=row[20],
                compressed_file_path=row[21]
            )
            tasks.append(task)
        
        return tasks
    
    def get_unfinished_tasks(self, hours: int = 24) -> List[TaskRecord]:
        """Get tasks that were pulled but never completed successfully"""
        cutoff_time = time.time() - (hours * 3600)
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT * FROM tasks 
            WHERE pulled_at > ? AND (
                submission_success = 0 OR 
                feedback_received = 0 OR 
                processed_at IS NULL
            )
            ORDER BY pulled_at DESC
        ''', (cutoff_time,))
        
        rows = cursor.fetchall()
        conn.close()
        
        tasks = []
        for row in rows:
            task = TaskRecord(
                task_id=row[0], prompt=row[1], prompt_hash=row[2],
                validator_uid=row[3], validator_hotkey=row[4], validator_stake=row[5],
                validation_threshold=row[6], pulled_at=row[7], processed_at=row[8],
                submitted_at=row[9], generation_time=row[10], validation_time=row[11],
                local_validation_score=row[12], submission_success=bool(row[13]),
                feedback_received=bool(row[14]), task_fidelity_score=row[15],
                average_fidelity_score=row[16], current_miner_reward=row[17],
                validation_failed=bool(row[18]) if row[18] is not None else None,
                generations_in_window=row[19], ply_file_path=row[20],
                compressed_file_path=row[21]
            )
            tasks.append(task)
        
        return tasks
    
    def get_duplicate_analysis(self, validator_uid: int, hours: int = 24) -> Dict[str, Any]:
        """Analyze duplicate checking for a specific validator"""
        cutoff_time = time.time() - (hours * 3600)
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Get all tasks from this validator in the time window
        cursor.execute('''
            SELECT prompt, prompt_hash, pulled_at, processed_at, submission_success, 
                   feedback_received, task_fidelity_score 
            FROM tasks 
            WHERE validator_uid = ? AND pulled_at > ?
            ORDER BY pulled_at DESC
        ''', (validator_uid, cutoff_time))
        
        tasks = cursor.fetchall()
        
        # Get recent prompts tracking
        cursor.execute('''
            SELECT prompt_hash, pulled_at FROM recent_prompts 
            WHERE validator_uid = ? AND pulled_at > ?
            ORDER BY pulled_at DESC
        ''', (validator_uid, cutoff_time))
        
        recent_prompts = cursor.fetchall()
        
        conn.close()
        
        analysis = {
            'validator_uid': validator_uid,
            'total_tasks_pulled': len(tasks),
            'successful_tasks': len([t for t in tasks if t[4] and t[5]]),  # submission_success and feedback_received
            'failed_tasks': len([t for t in tasks if not t[4] or not t[5]]),
            'unprocessed_tasks': len([t for t in tasks if t[3] is None]),  # processed_at is None
            'recent_prompts_tracked': len(recent_prompts),
            'unique_prompts': len(set(t[1] for t in tasks)),  # unique prompt_hashes
            'tasks': [
                {
                    'prompt': t[0][:50] + '...' if len(t[0]) > 50 else t[0],
                    'prompt_hash': t[1][:12],
                    'pulled_at': t[2],
                    'processed': t[3] is not None,
                    'submitted': t[4],
                    'feedback': t[5],
                    'score': t[6]
                }
                for t in tasks[-10:]  # Last 10 tasks
            ]
        }
        
        return analysis
    
    def cleanup_old_prompts(self, days: int = 7):
        """Clean up old prompt records and failed tasks"""
        cutoff_time = time.time() - (days * 24 * 3600)
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Clean up old recent_prompts
        cursor.execute('DELETE FROM recent_prompts WHERE pulled_at < ?', (cutoff_time,))
        deleted_prompts = cursor.rowcount
        
        # Clean up old failed tasks (keep successful ones longer)
        cursor.execute('''
            DELETE FROM tasks WHERE pulled_at < ? AND (
                submission_success = 0 OR 
                feedback_received = 0 OR 
                processed_at IS NULL
            )
        ''', (cutoff_time,))
        deleted_tasks = cursor.rowcount
        
        conn.commit()
        conn.close()
        
        logger.info(f"🧹 Cleaned up {deleted_prompts} old prompt records and {deleted_tasks} failed tasks")

class ContinuousTrellisOrchestrator:
    """Continuous TRELLIS orchestrator with intelligent features"""
    
    def __init__(self, config: Dict[str, Any]):
        # Merge with default config
        self.config = self._get_default_config()
        self.config.update(config)
        
        self.logger = logger
        
        # Setup output directory
        self.output_dir = Path(self.config['output_dir'])
        self.output_dir.mkdir(exist_ok=True)
        
        # Initialize database
        self.db = TaskDatabase()
        
        # Bittensor components
        self.wallet = None
        self.subtensor = None
        self.dendrite = None
        self.metagraph = None
        
        # State management
        self.validators: Dict[int, ValidatorState] = {}
        self.running = False
        self.start_time = time.time()
        
        # Initialize prompt optimizer
        try:
            # Try to use V2 optimizer first
            from prompt_optimizer_v2 import TrellisPromptOptimizerV2
            self.prompt_optimizer = TrellisPromptOptimizerV2()
            self.logger.info("✅ Using TrellisPromptOptimizerV2 (advanced optimization)")
            
            # Enable CLIP optimization if configured
            if self.config.get('enable_clip_optimization', True):
                try:
                    self.prompt_optimizer.enable_clip_optimization()
                    self.logger.info("🚀 CLIP optimization enabled for maximum alignment scores")
                except Exception as e:
                    self.logger.warning(f"⚠️ Could not enable CLIP optimization: {e}")
                    
        except ImportError:
            # Fall back to V1 optimizer
            from prompt_optimizer import TrellisPromptOptimizer
            self.prompt_optimizer = TrellisPromptOptimizer()
            self.logger.info("✅ Using TrellisPromptOptimizer (standard optimization)")
        
        # Initialize session statistics
        self.stats = {
            'session_start': time.time(),
            'tasks_pulled': 0,
            'tasks_processed': 0,
            'successful_generations': 0,
            'successful_validations': 0,
            'successful_submissions': 0,
            'total_rewards': 0.0,
            'total_generation_time': 0.0,
            'total_validation_time': 0.0,
            'idle_validations': 0,
            'prompts_optimized': 0,
            'optimization_improvements': 0
        }
        
        self.logger.info("🎯 Continuous TRELLIS Orchestrator initialized")
        self.logger.info(f"   Output directory: {self.output_dir}")
        self.logger.info(f"   Generation server: {self.config['generation_server_url']}")
        self.logger.info(f"   Validation server: {self.config['validation_server_url']}")
        
        # Log optimization settings
        if self.config.get('enable_prompt_optimization', True):
            mode = "aggressive" if self.config.get('optimization_aggressive_mode', False) else "standard"
            detail = "minimal" if not self.config.get('log_optimization_details', True) else "detailed"
            self.logger.info(f"🔧 Prompt optimization: ENABLED ({mode} mode, {detail} logging)")
        else:
            self.logger.info(f"🔧 Prompt optimization: DISABLED")
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration"""
        return {
            # Bittensor settings
            'wallet_name': 'test2m3b2',
            'hotkey_name': 't2m3b21',
            'netuid': 17,
            'min_validator_stake': 1000.0,  # Minimum stake required for a validator to be considered
            'min_validator_trust': 0.0,     # Minimum trust score
            'max_validators': 50,           # Maximum number of validators to track
            
            # Server settings
            'generation_server_url': 'http://localhost:8096',
            'validation_server_url': 'http://localhost:10006',
            
            # Operation settings
            'harvest_tasks': True,
            'validate_generations': True,
            'submit_results': True,
            'output_dir': './continuous_trellis_outputs',
            'save_intermediate_results': True,
            
            # Timing settings
            'task_pull_interval': 45,  # seconds between validator scans
            'idle_validation_interval': 300,  # 5 minutes
            'stats_report_interval': 600,  # 10 minutes
            'cleanup_interval': 3600,  # 1 hour
            'duplicate_check_hours': 24,
            
            # Quality settings
            'min_local_score': 0.3,
            'generation_timeout': 300,
            'validation_timeout': 120,
            'submission_timeout': 60,
            
            # Prompt optimization settings
            'enable_prompt_optimization': True,
            'optimization_aggressive_mode': True,
            'log_optimization_details': True,
            'enable_clip_optimization': True,  # Enable CLIP-based optimization
        }
    
    def _setup_bittensor(self) -> bool:
        """Setup Bittensor components"""
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
    
    def refresh_validators(self):
        """Refresh validator information from metagraph - discover all active validators"""
        if not self._setup_bittensor():
            return
        
        try:
            # Refresh metagraph
            self.metagraph = self.subtensor.metagraph(self.config['netuid'])
            
            # Clear existing validators that are no longer valid
            valid_uids = set()
            
            # Discover all validators on the subnet
            eligible_validators = []
            
            for uid, neuron in enumerate(self.metagraph.neurons):
                # Check if this is a valid validator
                if not neuron.validator_permit:
                    continue
                
                stake = float(neuron.stake)
                trust = float(neuron.trust)
                consensus = float(neuron.consensus)
                
                # Apply filtering criteria
                if stake < self.config['min_validator_stake']:
                    continue
                
                if trust < self.config['min_validator_trust']:
                    continue
                
                # Check if validator is responsive (has recent activity)
                # This could be enhanced with ping checks in the future
                
                eligible_validators.append({
                    'uid': uid,
                    'stake': stake,
                    'trust': trust,
                    'consensus': consensus,
                    'hotkey': neuron.hotkey,
                    'score': stake * trust * consensus  # Simple scoring for prioritization
                })
            
            # Sort by score (stake * trust * consensus) and take top validators
            eligible_validators.sort(key=lambda x: x['score'], reverse=True)
            eligible_validators = eligible_validators[:self.config['max_validators']]
            
            # Update validator states
            for validator_info in eligible_validators:
                uid = validator_info['uid']
                valid_uids.add(uid)
                
                if uid not in self.validators:
                    # Create new validator state
                    self.validators[uid] = ValidatorState(
                        uid=uid,
                        hotkey=validator_info['hotkey'],
                        stake=validator_info['stake'],
                        trust=validator_info['trust'],
                        consensus=validator_info['consensus']
                    )
                    self.logger.info(f"➕ Added new validator UID {uid} (stake: {validator_info['stake']:.1f}, trust: {validator_info['trust']:.3f})")
                else:
                    # Update existing validator
                    validator = self.validators[uid]
                    validator.stake = validator_info['stake']
                    validator.trust = validator_info['trust']
                    validator.consensus = validator_info['consensus']
                    validator.hotkey = validator_info['hotkey']
                    validator.is_active = True
            
            # Mark validators not in the current list as inactive
            inactive_count = 0
            for uid in list(self.validators.keys()):
                if uid not in valid_uids:
                    if self.validators[uid].is_active:
                        self.logger.info(f"➖ Validator UID {uid} is no longer active")
                        self.validators[uid].is_active = False
                        inactive_count += 1
            
            active_validators = len([v for v in self.validators.values() if v.is_active])
            
            self.logger.info(f"✅ Validator refresh complete:")
            self.logger.info(f"   Active validators: {active_validators}")
            self.logger.info(f"   Inactive validators: {inactive_count}")
            self.logger.info(f"   Total eligible validators found: {len(eligible_validators)}")
            
            # Log top validators by stake
            top_validators = sorted(
                [v for v in self.validators.values() if v.is_active], 
                key=lambda x: x.stake, 
                reverse=True
            )[:5]
            
            self.logger.info("   Top validators by stake:")
            for validator in top_validators:
                self.logger.info(f"     UID {validator.uid}: {validator.stake:.1f} TAO (trust: {validator.trust:.3f})")
            
        except Exception as e:
            self.logger.error(f"❌ Validator refresh failed: {e}")
            traceback.print_exc()
    
    def is_validator_available(self, validator: ValidatorState) -> bool:
        """Check if validator is available for task pulling"""
        current_time = time.time()
        
        # Check if validator is active
        if not validator.is_active:
            return False
        
        # Check cooldown
        if validator.cooldown_until and current_time < validator.cooldown_until:
            return False
        
        # Check if we pulled recently (respect pull interval)
        if validator.last_task_pull:
            time_since_pull = current_time - validator.last_task_pull
            if time_since_pull < self.config['task_pull_interval']:
                return False
        
        return True
    
    async def pull_task_from_validator(self, validator: ValidatorState) -> Optional[TaskRecord]:
        """Pull task from a specific validator with deduplication"""
        try:
            if not self.is_validator_available(validator):
                return None
            
            self.logger.debug(f"📡 Pulling from UID {validator.uid} ({validator.stake:.1f} TAO)")
            
            # Import protocol
            from neurons.common.protocol import PullTask
            
            # Create task pull request
            synapse = PullTask()
            synapse.timeout = self.config['submission_timeout']
            
            # Get neuron info
            if validator.uid >= len(self.metagraph.neurons):
                return None
            
            neuron = self.metagraph.neurons[validator.uid]
            
            start_time = time.time()
            
            # Query the validator
            response = await self.dendrite.forward(
                axons=[neuron.axon_info],
                synapse=synapse,
                timeout=self.config['submission_timeout']
            )
            
            query_time = time.time() - start_time
            validator.last_task_pull = time.time()
            
            if response and len(response) > 0:
                resp = response[0]
                
                if hasattr(resp, 'task') and resp.task and resp.task.prompt:
                    # Check for duplicates with detailed analysis
                    if self.db.is_duplicate_prompt(resp.task.prompt, validator.uid, self.config['duplicate_check_hours']):
                        # Get analysis for this validator to understand why it's being skipped
                        analysis = self.db.get_duplicate_analysis(validator.uid, 6)  # Last 6 hours
                        self.logger.info(f"⏭️ Skipping duplicate from UID {validator.uid}: '{resp.task.prompt[:50]}...'")
                        self.logger.info(f"   Analysis: {analysis['successful_tasks']}/{analysis['total_tasks_pulled']} successful, {analysis['failed_tasks']} failed, {analysis['unprocessed_tasks']} unprocessed")
                        return None
                    
                    # Update validator state
                    validator.total_tasks_pulled += 1
                    validator.last_task_received = time.time()
                    
                    # Update cooldown if provided
                    if hasattr(resp, 'cooldown_until'):
                        validator.cooldown_until = resp.cooldown_until
                    
                    # Create task record
                    prompt_hash = hashlib.sha256(resp.task.prompt.encode()).hexdigest()
                    
                    task = TaskRecord(
                        task_id=resp.task.id,
                        prompt=resp.task.prompt,
                        prompt_hash=prompt_hash,
                        validator_uid=validator.uid,
                        validator_hotkey=validator.hotkey,
                        validator_stake=validator.stake,
                        validation_threshold=getattr(resp, 'validation_threshold', 0.6),
                        pulled_at=time.time()
                    )
                    
                    # Add to recent prompts tracking
                    self.db.add_recent_prompt(resp.task.prompt, validator.uid)
                    
                    self.logger.info(f"✅ New task from UID {validator.uid}: '{task.prompt[:50]}...'")
                    self.logger.info(f"   Threshold: {task.validation_threshold}, Query time: {query_time:.2f}s")
                    
                    self.stats['tasks_pulled'] += 1
                    return task
                else:
                    self.logger.debug(f"⚠️ No task from UID {validator.uid}")
                    return None
            else:
                self.logger.debug(f"❌ No response from UID {validator.uid}")
                return None
        
        except Exception as e:
            self.logger.error(f"❌ Error pulling from UID {validator.uid}: {e}")
            return None
    
    def optimize_prompt_for_generation(self, task: TaskRecord) -> Tuple[str, Dict[str, any]]:
        """Optimize prompt to reduce zero fidelity risk and return parameter adjustments"""
        try:
            # Check if optimization is enabled
            if not self.config.get('enable_prompt_optimization', True):
                return task.prompt, {}
            
            # Load CLIP model if needed (for V2 optimizer)
            if hasattr(self.prompt_optimizer, 'load_clip_model'):
                self.prompt_optimizer.load_clip_model()
            
            try:
                # Analyze and optimize the prompt
                optimization_result = self.prompt_optimizer.optimize_prompt(
                    task.prompt, 
                    aggressive=self.config.get('optimization_aggressive_mode', True)  # Default to aggressive
                )
                analysis = optimization_result['analysis']
                
                # Get parameter adjustments if available (from V2 optimizer)
                param_adjustments = {}
                if hasattr(self.prompt_optimizer, '_optimize_prompt_v2'):
                    # V2 optimizer provides parameter adjustments
                    v2_result = self.prompt_optimizer._optimize_prompt_v2(task.prompt, aggressive=True)
                    param_adjustments = v2_result.parameter_adjustments
                
                # Log the analysis if enabled
                if self.config.get('log_optimization_details', True):
                    self.logger.info(f"🔍 Prompt Analysis for '{task.prompt[:50]}...':")
                    self.logger.info(f"   Risk Level: {analysis['risk_level']}")
                    
                    if analysis['risk_factors']:
                        self.logger.info(f"   Risk Factors:")
                        for factor in analysis['risk_factors']:
                            self.logger.info(f"     • {factor}")
                    
                    if param_adjustments:
                        self.logger.info(f"   📊 Parameter Adjustments:")
                        for param, value in param_adjustments.items():
                            self.logger.info(f"     • {param}: {value}")
                
                # Check if optimization is needed
                if optimization_result['improvement_expected']:
                    optimized_prompt = optimization_result['optimized_prompt']
                    applied_strategies = optimization_result['applied_strategies']
                    
                    if self.config.get('log_optimization_details', True):
                        self.logger.info(f"🔧 Prompt Optimization Applied:")
                        self.logger.info(f"   Original: {task.prompt}")
                        self.logger.info(f"   Optimized: {optimized_prompt}")
                        self.logger.info(f"   Strategies: {', '.join(applied_strategies)}")
                    else:
                        # Minimal logging when details are disabled
                        self.logger.info(f"🔧 Optimized prompt (risk: {analysis['risk_level']}): '{task.prompt[:30]}...' → '{optimized_prompt[:50]}...'")
                    
                    # Update statistics
                    self.stats['prompts_optimized'] += 1
                    self.stats['optimization_improvements'] += 1
                    
                    return optimized_prompt, param_adjustments
                else:
                    if self.config.get('log_optimization_details', True):
                        self.logger.info(f"✅ Prompt is low risk - minimal optimization applied")
                    
                    self.stats['prompts_optimized'] += 1
                    # Still add basic optimization for low-risk prompts
                    optimized = f"{task.prompt}, 3D model, detailed object, high quality"
                    return optimized, param_adjustments
            
            finally:
                # Always unload CLIP model after optimization to free GPU memory
                if hasattr(self.prompt_optimizer, 'unload_clip_model'):
                    self.prompt_optimizer.unload_clip_model()
                    
        except Exception as e:
            self.logger.error(f"❌ Prompt optimization failed: {e}")
            # Ensure CLIP model is unloaded even on error
            if hasattr(self.prompt_optimizer, 'unload_clip_model'):
                self.prompt_optimizer.unload_clip_model()
            return task.prompt, {}

    async def generate_3d_model(self, task: TaskRecord) -> Optional[Dict[str, Any]]:
        """Generate 3D model using TRELLIS server with prompt optimization and dynamic parameters"""
        self.logger.info(f"🎨 Generating 3D model: '{task.prompt}' (task: {task.task_id})")
        
        try:
            # Step 1: Optimize prompt and get parameter adjustments
            optimized_prompt, param_adjustments = self.optimize_prompt_for_generation(task)
            
            generation_start = time.time()
            
            # Prepare generation parameters
            generation_params = {
                'prompt': optimized_prompt,
                'seed': random.randint(0, 2**32 - 1),
                'return_compressed': True
            }
            
            # Add dynamic parameter adjustments if available
            if param_adjustments:
                # Map optimizer parameters to server parameters
                if 'guidance_scale' in param_adjustments:
                    generation_params['guidance_scale'] = param_adjustments['guidance_scale']
                if 'ss_guidance_strength' in param_adjustments:
                    generation_params['ss_guidance_strength'] = param_adjustments['ss_guidance_strength']
                if 'ss_sampling_steps' in param_adjustments:
                    generation_params['ss_sampling_steps'] = param_adjustments['ss_sampling_steps']
                if 'slat_guidance_strength' in param_adjustments:
                    generation_params['slat_guidance_strength'] = param_adjustments['slat_guidance_strength']
                if 'slat_sampling_steps' in param_adjustments:
                    generation_params['slat_sampling_steps'] = param_adjustments['slat_sampling_steps']
                
                self.logger.info(f"   🎯 Using custom parameters for high-risk prompt")
            
            # Call TRELLIS generation server with optimized prompt and parameters
            response = requests.post(
                f"{self.config['generation_server_url']}/generate/",
                data=generation_params,
                timeout=self.config['generation_timeout']
            )
            
            generation_time = time.time() - generation_start
            task.generation_time = generation_time
            
            if response.status_code == 200:
                ply_data = response.content
                
                # Get metadata from headers to check compression status
                compression_ratio = response.headers.get('X-Compression-Ratio', 'unknown')

                # Save PLY file
                if self.config['save_intermediate_results']:
                    timestamp = int(time.time())
                    ply_file = self.output_dir / f"task_{task.task_id}_{timestamp}.ply.spz"
                    with open(ply_file, 'wb') as f:
                        f.write(ply_data)
                    task.compressed_file_path = str(ply_file)
                
                self.logger.info(f"✅ Generation successful in {generation_time:.2f}s ({len(ply_data):,} bytes)")
                
                self.stats['successful_generations'] += 1
                self.stats['total_generation_time'] += generation_time
                
                return {'ply_data': ply_data, 'compression_ratio': compression_ratio}
            else:
                self.logger.error(f"❌ Generation failed: HTTP {response.status_code}")
                return None
        
        except Exception as e:
            self.logger.error(f"❌ Generation exception: {e}")
            return None
    
    async def validate_model(self, task: TaskRecord, ply_data: bytes) -> Optional[float]:
        """Validate generated model and update task record"""
        if not self.config['validate_generations']:
            return None
        
        self.logger.info(f"📊 Validating model: '{task.prompt[:50]}...'")
        
        try:
            validation_start = time.time()
            
            # Decompress PLY data for validation
            try:
                import pyspz
                decompressed_data = pyspz.decompress(ply_data)
            except ImportError:
                self.logger.error("❌ pyspz not available")
                return None
            except Exception as e:
                self.logger.error(f"❌ Decompression failed: {e}")
                return None
            
            # Convert to base64
            encoded_data = base64.b64encode(decompressed_data).decode('utf-8')
            
            request_data = {
                "prompt": task.prompt,
                "data": encoded_data,
                "compression": 0,
                "generate_preview": False,
                "preview_score_threshold": 0.8
            }
            
            response = requests.post(
                f"{self.config['validation_server_url']}/validate_txt_to_3d_ply/",
                json=request_data,
                timeout=self.config['validation_timeout']
            )
            
            validation_time = time.time() - validation_start
            task.validation_time = validation_time
            
            if response.status_code == 200:
                result = response.json()
                score = result.get("score", 0.0)
                task.local_validation_score = score
                
                self.logger.info(f"✅ Validation completed in {validation_time:.2f}s")
                self.logger.info(f"   Score: {score:.4f}, IQA: {result.get('iqa', 0):.3f}")
                self.logger.info(f"   Alignment: {result.get('alignment_score', 0):.3f}")
                
                self.stats['successful_validations'] += 1
                self.stats['total_validation_time'] += validation_time
                
                return score
            else:
                self.logger.error(f"❌ Validation failed: HTTP {response.status_code}")
                return None
        
        except Exception as e:
            self.logger.error(f"❌ Validation exception: {e}")
            return None
    
    async def submit_result(self, task: TaskRecord, generation_result: Dict[str, Any]) -> bool:
        """Submit result to validator and process feedback"""
        if not self.config['submit_results']:
            return True
        
        self.logger.info(f"📤 Submitting result: {task.task_id}")
        
        try:
            if not self._setup_bittensor():
                return False
            
            # Import protocol
            from neurons.common.protocol import SubmitResults, Task
            
            # Get validator info
            if task.validator_uid >= len(self.metagraph.neurons):
                self.logger.error(f"❌ Validator UID {task.validator_uid} not found")
                return False
            
            neuron = self.metagraph.neurons[task.validator_uid]
            
            # Create task object
            task_obj = Task(id=task.task_id, prompt=task.prompt)
            
            # Get data from TRELLIS server - these are SPZ-compressed bytes
            ply_data = generation_result['ply_data']
            
            # The 'results' field in SubmitResults synapse requires a base64-encoded STRING.
            # The TRELLIS server already provides SPZ-compressed bytes, so we just need to base64 encode them.
            self.logger.info(f"   📦 Using SPZ-compressed data from server ({len(ply_data):,} bytes)")
            encoded_data = base64.b64encode(ply_data).decode('utf-8')

            # Create submission
            submit_time = time.time_ns()
            
            try:
                from neurons.common.miner_license_consent_declaration import MINER_LICENSE_CONSENT_DECLARATION
            except ImportError:
                MINER_LICENSE_CONSENT_DECLARATION = "I, as a miner on SN17, have obtained all licenses, rights and consents required to use, reproduce, modify, display, distribute and make available my submitted results to this subnet and its end users"

            message = f"{MINER_LICENSE_CONSENT_DECLARATION}{submit_time}{task.prompt}{neuron.hotkey}{self.wallet.hotkey.ss58_address}"
            signature = base64.b64encode(self.dendrite.keypair.sign(message)).decode('utf-8')
            
            synapse = SubmitResults(
                task=task_obj,
                results=encoded_data,
                compression=2,  # spz compression
                submit_time=submit_time,
                signature=signature
            )
            
            synapse.timeout = self.config['submission_timeout']
            
            start_time = time.time()
            
            # Submit to validator using the correct API call
            response = await self.dendrite.call(
                target_axon=neuron.axon_info,
                synapse=synapse,
                deserialize=False,
                timeout=self.config['submission_timeout']
            )
            
            submit_time_elapsed = time.time() - start_time
            # print("time elapsed: ", submit_time_elapsed)
            task.submitted_at = time.time()
            
            if response and hasattr(response, 'feedback') and response.feedback:
                feedback = response.feedback
                
                # Process feedback scores
                task.feedback_received = True
                task.submission_success = True
                task.task_fidelity_score = feedback.task_fidelity_score
                task.average_fidelity_score = feedback.average_fidelity_score
                task.current_miner_reward = feedback.current_miner_reward
                task.validation_failed = feedback.validation_failed
                task.generations_in_window = feedback.generations_within_the_window
                
                # Update validator statistics
                validator = self.validators[task.validator_uid]
                validator.total_tasks_submitted += 1
                if task.submission_success and task.task_fidelity_score is not None:
                    validator.total_successful_submissions += 1
                    # Update average score with exponential moving average
                    if validator.average_score == 0:
                        validator.average_score = task.task_fidelity_score
                    else:
                        validator.average_score = validator.average_score * 0.9 + task.task_fidelity_score * 0.1
                
                # Update session stats
                self.stats['successful_submissions'] += 1
                if task.current_miner_reward:
                    self.stats['total_rewards'] += task.current_miner_reward
                
                self.logger.info(f"✅ Submission successful to UID {task.validator_uid} ({submit_time_elapsed:.2f}s)")
                self.logger.info(f"   Task fidelity: {task.task_fidelity_score:.4f}")
                self.logger.info(f"   Average fidelity: {task.average_fidelity_score:.4f}")
                self.logger.info(f"   Miner reward: {task.current_miner_reward:.6f}")
                self.logger.info(f"   Validation failed: {task.validation_failed}")
                self.logger.info(f"   Generations in window: {task.generations_in_window}")
                
                # Log optimization impact if zero fidelity was avoided
                if (self.config.get('enable_prompt_optimization', True) and 
                    task.task_fidelity_score > 0.0 and 
                    self.stats['optimization_improvements'] > 0):
                    self.logger.info(f"   🎯 Zero fidelity avoided (optimization working!)")
                
                return True
            else:
                self.logger.error(f"❌ No feedback received from UID {task.validator_uid}")
                task.submission_success = False
                return False
        
        except Exception as e:
            self.logger.error(f"❌ Submission failed: {e}")
            traceback.print_exc()
            task.submission_success = False
            return False
    
    async def process_task(self, task: TaskRecord) -> bool:
        """Process a single task end-to-end"""
        self.logger.info(f"🔄 Processing task {task.task_id}: '{task.prompt}'")
        
        task.processed_at = time.time()
        self.stats['tasks_processed'] += 1
        
        try:
            # Step 1: Generate 3D model
            generation_result = await self.generate_3d_model(task)
            if not generation_result:
                self.logger.error(f"❌ Generation failed for task {task.task_id}")
                self.db.save_task(task)
                return False
            
            # ply_data = generation_result['ply_data']

            # # Step 2: Validate locally
            # local_score = await self.validate_model(task, ply_data)
            # if local_score is not None and local_score < self.config['min_local_score']:
            #     self.logger.warning(f"⚠️ Local score too low ({local_score:.3f}), skipping submission")
            #     self.db.save_task(task)
            #     return False
            
            # Step 3: Submit results, passing the full generation result dictionary
            success = await self.submit_result(task, generation_result)
            
            # Save task record
            self.db.save_task(task)
            
            if success:
                self.logger.info(f"✅ Task {task.task_id} completed successfully")
            else:
                self.logger.error(f"❌ Task {task.task_id} submission failed")
            
            return success
        
        except Exception as e:
            self.logger.error(f"❌ Task processing failed: {e}")
            traceback.print_exc()
            self.db.save_task(task)
            return False
    
    async def idle_validation_cycle(self):
        """Perform validation on recent unvalidated generations during idle time"""
        self.logger.info("🔍 Running idle validation cycle...")
        
        try:
            # Get recent unvalidated tasks
            unvalidated_tasks = self.db.get_recent_unvalidated_tasks(hours=2)
            
            if not unvalidated_tasks:
                self.logger.info("   No unvalidated tasks found")
                return
            
            self.logger.info(f"   Found {len(unvalidated_tasks)} unvalidated tasks")
            
            for task in unvalidated_tasks:
                if not self.running:
                    break
                
                # Check if PLY file exists
                if not task.compressed_file_path or not Path(task.compressed_file_path).exists():
                    continue
                
                try:
                    # Load PLY data
                    with open(task.compressed_file_path, 'rb') as f:
                        ply_data = f.read()
                    
                    # Validate
                    score = await self.validate_model(task, ply_data)
                    if score is not None:
                        self.logger.info(f"   Validated task {task.task_id}: score {score:.4f}")
                        self.stats['idle_validations'] += 1
                        
                        # Update task in database
                        self.db.save_task(task)
                
                except Exception as e:
                    self.logger.error(f"   Failed to validate task {task.task_id}: {e}")
        
        except Exception as e:
            self.logger.error(f"❌ Idle validation cycle failed: {e}")
    
    def save_statistics(self):
        """Save comprehensive statistics to JSON file"""
        try:
            uptime_hours = (time.time() - self.start_time) / 3600
            
            # Validator statistics
            validator_stats = {}
            for uid, validator in self.validators.items():
                validator_stats[uid] = {
                    'hotkey': validator.hotkey,
                    'stake': validator.stake,
                    'trust': validator.trust,
                    'consensus': validator.consensus,
                    'total_tasks_pulled': validator.total_tasks_pulled,
                    'total_tasks_received': validator.total_tasks_received,
                    'total_tasks_submitted': validator.total_tasks_submitted,
                    'total_successful_submissions': validator.total_successful_submissions,
                    'average_score': validator.average_score,
                    'success_rate': validator.total_successful_submissions / max(1, validator.total_tasks_submitted),
                    'last_task_received': validator.last_task_received,
                    'is_active': validator.is_active
                }
            
            # Comprehensive statistics
            stats = {
                'timestamp': datetime.now().isoformat(),
                'uptime_hours': uptime_hours,
                'session_stats': self.stats,
                'validator_stats': validator_stats,
                'performance': {
                    'tasks_per_hour': self.stats['tasks_processed'] / max(0.1, uptime_hours),
                    'success_rate': self.stats['successful_submissions'] / max(1, self.stats['tasks_processed']),
                    'avg_generation_time': self.stats['total_generation_time'] / max(1, self.stats['successful_generations']),
                    'avg_validation_time': self.stats['total_validation_time'] / max(1, self.stats['successful_validations']),
                    'total_rewards': self.stats['total_rewards'],
                    'rewards_per_hour': self.stats['total_rewards'] / max(0.1, uptime_hours),
                    'optimization_rate': (self.stats['optimization_improvements'] / max(1, self.stats['prompts_optimized'])) * 100,
                    'prompts_optimized': self.stats['prompts_optimized'],
                    'optimization_improvements': self.stats['optimization_improvements']
                }
            }
            
            # Save to file
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            stats_file = self.output_dir / f"continuous_stats_{timestamp}.json"
            
            with open(stats_file, 'w') as f:
                json.dump(stats, f, indent=2)
            
            self.logger.info(f"📊 Statistics saved to {stats_file}")
            
        except Exception as e:
            self.logger.error(f"❌ Failed to save statistics: {e}")
    
    def print_status(self):
        """Print current status"""
        uptime_hours = (time.time() - self.start_time) / 3600
        
        self.logger.info("📊 CONTINUOUS ORCHESTRATOR STATUS")
        self.logger.info("="*60)
        self.logger.info(f"Uptime: {uptime_hours:.2f} hours")
        self.logger.info(f"Tasks pulled: {self.stats['tasks_pulled']}")
        self.logger.info(f"Tasks processed: {self.stats['tasks_processed']}")
        self.logger.info(f"Successful generations: {self.stats['successful_generations']}")
        self.logger.info(f"Successful validations: {self.stats['successful_validations']}")
        self.logger.info(f"Successful submissions: {self.stats['successful_submissions']}")
        self.logger.info(f"Total rewards: {self.stats['total_rewards']:.6f} TAO")
        self.logger.info(f"Idle validations: {self.stats['idle_validations']}")
        self.logger.info(f"Prompts optimized: {self.stats['prompts_optimized']}")
        self.logger.info(f"Optimization improvements: {self.stats['optimization_improvements']}")
        
        if uptime_hours > 0:
            self.logger.info(f"Tasks/hour: {self.stats['tasks_processed'] / uptime_hours:.1f}")
            self.logger.info(f"Rewards/hour: {self.stats['total_rewards'] / uptime_hours:.6f} TAO")
            
        # Optimization statistics
        if self.stats['prompts_optimized'] > 0:
            optimization_rate = (self.stats['optimization_improvements'] / self.stats['prompts_optimized']) * 100
            self.logger.info(f"Optimization rate: {optimization_rate:.1f}% of prompts improved")
        
        # Active validators
        active_validators = [v for v in self.validators.values() if v.is_active]
        self.logger.info(f"Active validators: {len(active_validators)}")
        
        for validator in sorted(active_validators, key=lambda v: v.stake, reverse=True)[:3]:
            self.logger.info(f"  UID {validator.uid}: {validator.total_tasks_received} tasks, avg score: {validator.average_score:.3f}")
        
        # Check for unfinished tasks
        unfinished_tasks = self.db.get_unfinished_tasks(6)  # Last 6 hours
        if unfinished_tasks:
            self.logger.warning(f"⚠️ Found {len(unfinished_tasks)} unfinished tasks in last 6 hours:")
            for task in unfinished_tasks[-5:]:  # Show last 5
                status = "not_processed" if task.processed_at is None else ("no_submission" if not task.submission_success else "no_feedback")
                self.logger.warning(f"   UID {task.validator_uid}: '{task.prompt[:30]}...' - {status}")
        
        self.logger.info("="*60)
    
    async def retry_unfinished_tasks(self):
        """Retry tasks that failed to submit in recent hours"""
        self.logger.info("🔄 Checking for unfinished tasks to retry...")
        
        try:
            # Get unfinished tasks from the last 2 hours (older than 15 minutes to avoid immediate retry)
            unfinished_tasks = self.db.get_unfinished_tasks(hours=2)
            
            if not unfinished_tasks:
                return
            
            # Filter tasks older than 15 minutes
            current_time = time.time()
            retry_cutoff = current_time - (0.25 * 3600)  # 15 minutes
            
            tasks_to_retry = []
            for task in unfinished_tasks:
                if task.pulled_at < retry_cutoff and not task.submission_success:
                    tasks_to_retry.append(task)
            
            if not tasks_to_retry:
                return
            
            self.logger.info(f"🔁 Found {len(tasks_to_retry)} unfinished tasks to retry")
            
            # Retry up to 3 tasks per cycle to avoid overwhelming
            for task in tasks_to_retry[:3]:
                if not self.running:
                    break
                
                self.logger.info(f"🔁 Retrying task {task.task_id} from UID {task.validator_uid}: '{task.prompt[:50]}...'")
                
                # Process the task again
                success = await self.process_task(task)
                
                if success:
                    self.logger.info(f"✅ Successfully retried task {task.task_id}")
                else:
                    self.logger.warning(f"⚠️ Retry failed for task {task.task_id}")
                
                # Small delay between retries
                await asyncio.sleep(2)
        
        except Exception as e:
            self.logger.error(f"❌ Failed to retry unfinished tasks: {e}")
            traceback.print_exc()
    
    async def continuous_mining_loop(self):
        """Main continuous mining loop"""
        self.logger.info("🚀 Starting continuous TRELLIS mining...")
        
        # Setup Bittensor
        if not self._setup_bittensor():
            self.logger.error("❌ Failed to setup Bittensor")
            return
        
        # Initial validator refresh
        self.refresh_validators()
        
        if not self.validators:
            self.logger.error("❌ No active validators found")
            return
        
        self.running = True
        self.start_time = time.time()
        
        # Initialize timing
        last_stats_report = 0
        last_cleanup = 0
        last_idle_validation = 0
        last_validator_refresh = 0
        last_retry_check = 0  # Track last retry check
        
        try:
            while self.running:
                current_time = time.time()
                
                # Periodic validator refresh (every 10 minutes to catch changes)
                if current_time - last_validator_refresh > 600:
                    self.refresh_validators()
                    last_validator_refresh = current_time
                
                # Check for unfinished tasks to retry (every 5 minutes)
                if current_time - last_retry_check > 300:
                    await self.retry_unfinished_tasks()
                    last_retry_check = current_time
                
                # Pull tasks from all available validators
                new_task_found = False
                
                for validator in self.validators.values():
                    if not self.running:
                        break
                    
                    task = await self.pull_task_from_validator(validator)
                    if task:
                        new_task_found = True
                        # Process task immediately
                        await self.process_task(task)
                
                # If no new tasks, do idle validation
                # if not new_task_found and current_time - last_idle_validation > self.config['idle_validation_interval']:
                #     await self.idle_validation_cycle()
                #     last_idle_validation = current_time
                
                # Periodic statistics report
                if current_time - last_stats_report > self.config['stats_report_interval']:
                    self.print_status()
                    self.save_statistics()
                    last_stats_report = current_time
                
                # Periodic cleanup
                if current_time - last_cleanup > self.config['cleanup_interval']:
                    self.db.cleanup_old_prompts()
                    last_cleanup = current_time
                
                # Wait before next cycle
                await asyncio.sleep(2)  # Short sleep between cycles
        
        except KeyboardInterrupt:
            self.logger.info("🛑 Mining interrupted by user")
        except Exception as e:
            self.logger.error(f"❌ Mining loop error: {e}")
            traceback.print_exc()
        finally:
            self.running = False
            self.print_status()
            self.save_statistics()
            self.logger.info("🏁 Continuous mining stopped")

async def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Continuous TRELLIS Orchestrator")
    parser.add_argument("--no-harvest", action="store_true", help="Disable task harvesting")
    parser.add_argument("--no-validate", action="store_true", help="Disable validation")
    parser.add_argument("--no-submit", action="store_true", help="Disable result submission")
    parser.add_argument("--generation-server", default="http://localhost:8096", help="TRELLIS generation server URL")
    parser.add_argument("--validation-server", default="http://localhost:10006", help="Validation server URL")
    parser.add_argument("--output-dir", default="./continuous_trellis_outputs", help="Output directory")
    parser.add_argument("--min-score", type=float, default=0.3, help="Minimum local validation score")
    
    # Prompt optimization arguments
    parser.add_argument("--no-optimize", action="store_true", help="Disable prompt optimization")
    parser.add_argument("--aggressive-optimize", action="store_true", help="Enable aggressive optimization mode")
    parser.add_argument("--quiet-optimize", action="store_true", help="Reduce optimization logging detail")
    parser.add_argument("--no-clip", action="store_true", help="Disable CLIP-based prompt optimization")
    
    args = parser.parse_args()
    
    # Build config
    config = {}
    
    if args.no_harvest:
        config['harvest_tasks'] = False
    if args.no_validate:
        config['validate_generations'] = False
    if args.no_submit:
        config['submit_results'] = False
    
    config['generation_server_url'] = args.generation_server
    config['validation_server_url'] = args.validation_server
    config['output_dir'] = args.output_dir
    config['min_local_score'] = args.min_score
    
    # Prompt optimization configuration
    if args.no_optimize:
        config['enable_prompt_optimization'] = False
    if args.aggressive_optimize:
        config['optimization_aggressive_mode'] = True
    if args.quiet_optimize:
        config['log_optimization_details'] = False
    if args.no_clip:
        config['enable_clip_optimization'] = False
    
    # Create and run orchestrator
    orchestrator = ContinuousTrellisOrchestrator(config)
    
    try:
        await orchestrator.continuous_mining_loop()
    except Exception as e:
        logger.error(f"❌ Orchestrator failed: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(main()) 