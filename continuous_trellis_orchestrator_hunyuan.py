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
- PRIORITY-BASED server coordination for time-critical tasks
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
import importlib.util
import subprocess
from pathlib import Path
from typing import List, Dict, Optional, Any, Set
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict

# Import torch for CUDA cache management
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("⚠️ PyTorch not available - CUDA cache management disabled")

# Import the prompt optimizer
try:
    # from smart_prompt_optimizer_fixed import OptimizedPromptOptimizer
    # from llm_prompt_optimizer_v7_f1 import LLMPromptOptimizer
    from llm_prompt_optimizer_v12_f1 import LLMPromptOptimizer
    OPTIMIZED_PROMPT_OPTIMIZER_AVAILABLE = True
    print("✅ Using new performance-optimized prompt optimizer")
except ImportError:
    from prompt_optimizer import TrellisPromptOptimizer
    OPTIMIZED_PROMPT_OPTIMIZER_AVAILABLE = False
    print("⚠️ Falling back to original prompt optimizer")

# Import the reproducibility system
try:
    from llm_close_prompt_reproducibility_test import LLMClosePromptReproducibility
    REPRODUCIBILITY_SYSTEM_AVAILABLE = False
    print("✅ Using reproducibility system for pre-optimization")
except ImportError:
    REPRODUCIBILITY_SYSTEM_AVAILABLE = False
    print("⚠️ Reproducibility system not available")

# Import the CLIP-based prompt optimizer
# try:
#     from get_max_clip_score import CLIPScoreMaximizer
#     # from clip_episodic_optimizer import ClipEpisodicOptimizer
#     CLIP_OPTIMIZER_AVAILABLE = True
#     print("✅ Using CLIP-based episodic prompt optimizer")
# except ImportError:
#     CLIP_OPTIMIZER_AVAILABLE = False
#     print("⚠️ CLIP-based prompt optimizer not available")

CLIP_OPTIMIZER_AVAILABLE = True

import torch
seed = 42
torch.manual_seed(seed)
torch.use_deterministic_algorithms(True)

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
        logging.FileHandler('continuous_trellis_hunyuan.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class PriorityServerCoordinator:
    """
    Priority-based server coordinator that gives the orchestrator HIGH PRIORITY access.
    This allows time-critical subnet tasks to bypass or interrupt other processes.
    """
    
    def __init__(self, server_url: str = "http://localhost:8096", 
                 max_wait_time_seconds: int = 60,
                 status_check_interval: int = 1,
                 priority_timeout: int = 30,
                 on_interruption_callback=None):
        """
        Initialize the priority server coordinator.
        
        Args:
            server_url: Base URL of the GPU server
            max_wait_time_seconds: Maximum time to wait for server availability
            status_check_interval: Interval between status checks (faster for priority)
            priority_timeout: Timeout for priority access attempts
        """
        self.server_url = server_url.rstrip('/')
        self.max_wait_time_seconds = max_wait_time_seconds
        self.status_check_interval = status_check_interval
        self.priority_timeout = priority_timeout
        self.on_interruption_callback = on_interruption_callback
        self.logger = logging.getLogger(__name__)
        
    def check_server_status(self) -> Dict[str, Any]:
        """
        Check the current status of the GPU server.
        
        Returns:
            Dictionary containing server status information
        """
        try:
            # First check health endpoint
            health_url = f"{self.server_url}/health/"
            health_resp = requests.get(health_url, timeout=3)  # Faster timeout for priority
            if health_resp.status_code != 200:
                return {
                    "available": False,
                    "status": "unhealthy",
                    "error": f"Health check failed: HTTP {health_resp.status_code}"
                }
            
            # Check job status
            job_status_url = f"{self.server_url}/job/status/"
            job_resp = requests.get(job_status_url, timeout=3)
            if job_resp.status_code != 200:
                return {
                    "available": False,
                    "status": "unknown",
                    "error": f"Job status check failed: HTTP {job_resp.status_code}"
                }
            
            job_data = job_resp.json()
            job_status = job_data.get('status', 'unknown')
            
            # For priority coordinator, we consider server available if it's not in critical busy states
            # We can interrupt non-critical operations
            if job_status in ('processing', 'generating', 'validating'):
                # Check if this is a priority operation (ours) or low priority (optimizer)
                job_id = job_data.get('job_id', '')
                prompt = job_data.get('prompt', '')
                
                # If it's our job, we can use the server
                if self._is_our_job(job_id, prompt):
                    return {
                        "available": True,
                        "status": job_status,
                        "job_id": job_id,
                        "our_job": True
                    }
                else:
                    # It's someone else's job - we can interrupt for priority
                    return {
                        "available": True,  # Available for priority access
                        "status": f"interruptible_{job_status}",
                        "job_id": job_id,
                        "prompt": prompt,
                        "interruptible": True
                    }
            
            # Server is available
            return {
                "available": True,
                "status": job_status,
                "job_id": job_data.get('job_id')
            }
            
        except requests.exceptions.Timeout:
            return {
                "available": False,
                "status": "timeout",
                "error": "Server status check timed out"
            }
        except requests.exceptions.ConnectionError:
            return {
                "available": False,
                "status": "connection_error",
                "error": "Cannot connect to server"
            }
        except Exception as e:
            return {
                "available": False,
                "status": "error",
                "error": str(e)
            }
    
    def _is_our_job(self, job_id: str, prompt: str) -> bool:
        """
        Determine if the current job is ours (orchestrator) or someone else's (optimizer).
        
        Args:
            job_id: Current job ID
            prompt: Current prompt being processed
            
        Returns:
            True if this is our job, False if it's someone else's
        """
        # Check if job_id contains our identifiers
        if job_id and any(identifier in job_id.lower() for identifier in ['orchestrator', 'subnet', 'miner', 'task']):
            return True
        
        # Check if prompt matches our patterns (subnet tasks are usually shorter and specific)
        if prompt and len(prompt) < 100:  # Subnet tasks are typically shorter
            return True
        
        # Default: assume it's not our job (optimizer jobs are usually longer prompts)
        return False
    
    def wait_for_priority_access(self, task_id: str = None) -> bool:
        """
        Wait for priority access to the server, with ability to interrupt other processes.
        
        Args:
            task_id: Our task ID for identification
            
        Returns:
            True if priority access granted, False if timeout reached
        """
        start_wait_time = time.time()
        
        while time.time() - start_wait_time < self.max_wait_time_seconds:
            status = self.check_server_status()
            
            if status["available"]:
                if status.get("interruptible"):
                    self.logger.warning(f"🚨 PRIORITY INTERRUPTION: Interrupting job {status.get('job_id', 'unknown')} for subnet task {task_id}")
                    # Force clear the server to interrupt the current job
                    self._force_clear_server()
                    time.sleep(2)  # Brief pause for server to reset
                    # Track this interruption
                    if self.on_interruption_callback:
                        self.on_interruption_callback()
                    return True
                else:
                    self.logger.info(f"✅ Priority access granted (status: {status['status']})")
                    return True
            
            # Log the current status
            error = status.get("error", "unknown error")
            self.logger.info(f"⏳ Waiting for priority access: {status['status']} - {error}")
            
            # Wait before next check (faster for priority)
            time.sleep(self.status_check_interval)
        
        self.logger.error(f"⏰ Priority access timeout ({self.max_wait_time_seconds}s) - subnet task may be missed!")
        return False
    
    def _force_clear_server(self):
        """
        Force clear the server to interrupt current operations.
        This is used for priority access when subnet tasks are at risk.
        """
        try:
            # Try to clear cache
            clear_url = f"{self.server_url}/clear_cache/"
            resp = requests.post(clear_url, timeout=5)
            if resp.status_code == 200:
                self.logger.info("🧹 Server cache cleared for priority access")
            else:
                self.logger.warning(f"⚠️ Failed to clear server cache: HTTP {resp.status_code}")
            
            # Try to reset job status
            reset_url = f"{self.server_url}/job/reset/"
            resp = requests.post(reset_url, timeout=5)
            if resp.status_code == 200:
                self.logger.info("🔄 Server job status reset for priority access")
            else:
                self.logger.warning(f"⚠️ Failed to reset job status: HTTP {resp.status_code}")
                
        except Exception as e:
            self.logger.warning(f"⚠️ Exception during force clear: {e}")
    
    def clear_server_cache(self) -> bool:
        """
        Clear the GPU cache on the server.
        
        Returns:
            True if successful, False otherwise
        """
        try:
            clear_url = f"{self.server_url}/clear_cache/"
            resp = requests.post(clear_url, timeout=5)
            if resp.status_code == 200:
                self.logger.info("🧹 GPU cache cleared successfully")
                return True
            else:
                self.logger.warning(f"⚠️ Failed to clear GPU cache: HTTP {resp.status_code}")
                return False
        except Exception as e:
            self.logger.warning(f"⚠️ Exception clearing GPU cache: {e}")
            return False
    
    def mark_priority_job_start(self, task_id: str, prompt: str):
        """
        Mark the start of a priority job to help with identification.
        
        Args:
            task_id: Our task ID
            prompt: The prompt being processed
        """
        self.logger.info(f"🚀 Starting PRIORITY job: {task_id} - '{prompt[:50]}...'")
    
    def mark_priority_job_end(self, task_id: str):
        """
        Mark the end of a priority job.
        
        Args:
            task_id: Our task ID
        """
        self.logger.info(f"✅ Completed PRIORITY job: {task_id}")


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
    total_processing_time: Optional[float] = None
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
    
    # Priority access tracking
    priority_access_timeout: bool = False

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
    
    def __init__(self, db_path: str = "continuous_trellis_hunyuan_tasks.db"):
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
                total_processing_time REAL,
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
        
        # Also check for any recent attempts (successful or not) but with shorter window
        recent_cutoff = time.time() - (1 * 3600)  # 1 hour for failed attempts (more forgiving)
        cursor.execute('''
            SELECT COUNT(*) FROM tasks 
            WHERE prompt_hash = ? AND validator_uid = ? AND pulled_at > ?
        ''', (prompt_hash, validator_uid, recent_cutoff))
        
        recent_attempts = cursor.fetchone()[0]
        
        conn.close()
        
        # Don't duplicate if we successfully submitted recently (24 hour window)
        if successful_submissions > 0:
            return True
        
        # Allow retry after 1 hour if previous attempts failed (more aggressive retry)
        if recent_attempts > 0:
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
             validation_time, total_processing_time, local_validation_score, submission_success, feedback_received,
             task_fidelity_score, average_fidelity_score, current_miner_reward,
             validation_failed, generations_in_window, ply_file_path, compressed_file_path)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            task.task_id, task.prompt, task.prompt_hash, task.validator_uid,
            task.validator_hotkey, task.validator_stake, task.validation_threshold,
            task.pulled_at, task.processed_at, task.submitted_at, task.generation_time,
            task.validation_time, task.total_processing_time, task.local_validation_score, task.submission_success,
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
                total_processing_time=row[12], local_validation_score=row[13], submission_success=bool(row[14]),
                feedback_received=bool(row[15]), task_fidelity_score=row[16],
                average_fidelity_score=row[17], current_miner_reward=row[18],
                validation_failed=bool(row[19]) if row[19] is not None else None,
                generations_in_window=row[20], ply_file_path=row[21],
                compressed_file_path=row[22]
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
                total_processing_time=row[12], local_validation_score=row[13], submission_success=bool(row[14]),
                feedback_received=bool(row[15]), task_fidelity_score=row[16],
                average_fidelity_score=row[17], current_miner_reward=row[18],
                validation_failed=bool(row[19]) if row[19] is not None else None,
                generations_in_window=row[20], ply_file_path=row[21],
                compressed_file_path=row[22]
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
    
    def has_processed_prompt(self, prompt_hash: str) -> bool:
        """Check if this prompt has already been processed (for simulation mode)."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('SELECT COUNT(*) FROM tasks WHERE prompt_hash = ? AND processed_at IS NOT NULL', (prompt_hash,))
        count = cursor.fetchone()[0]
        conn.close()
        return count > 0
    
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
        if OPTIMIZED_PROMPT_OPTIMIZER_AVAILABLE:
            # self.prompt_optimizer = OptimizedPromptOptimizer("rl_checkpoints_v3/prompt_score_log.csv")
            self.prompt_optimizer = LLMPromptOptimizer(model="llama3.2:3b")
            self.logger.info("🚀 Initialized performance-optimized prompt optimizer")
        # else:
        #     self.prompt_optimizer = TrellisPromptOptimizer()
        #     self.logger.info("🔧 Initialized standard prompt optimizer")
        
        # Initialize reproducibility system
        if REPRODUCIBILITY_SYSTEM_AVAILABLE:
            self.reproducibility_system = LLMClosePromptReproducibility()
            self.logger.info("🔄 Initialized reproducibility system for pre-optimization")
        else:
            self.reproducibility_system = None
            self.logger.info("⚠️ Reproducibility system not available")
        
        # Initialize CLIP episodic memory for similarity-based optimization
        self.clip_memory = None
        if CLIP_OPTIMIZER_AVAILABLE:
            try:
                # Load the episodic memory from the CLIP optimizer
                memory_file = "clip_optimizer_logs/clip_optimizer_memory.json"
                if Path(memory_file).exists():
                    with open(memory_file, 'r') as f:
                        self.clip_memory = json.load(f)
                    
                    # Count high-scoring prompts in memory
                    high_scoring_prompts = [
                        prompt_data for prompt_data in self.clip_memory.get("prompt_optimizations", {}).values()
                        if prompt_data.get("best_score", 0) > 0.3
                    ]
                    
                    self.logger.info("🎯 Loaded CLIP episodic memory for similarity-based optimization")
                    self.logger.info(f"   Memory file: {memory_file}")
                    self.logger.info(f"   Total prompts in memory: {len(self.clip_memory.get('prompt_optimizations', {}))}")
                    self.logger.info(f"   High-scoring prompts (>0.3): {len(high_scoring_prompts)}")
                    self.logger.info(f"   Similarity threshold: {self.config.get('clip_memory_similarity_threshold', 0.7)}")
                else:
                    self.logger.warning(f"⚠️ CLIP memory file not found: {memory_file}")
                    self.clip_memory = None
            except Exception as e:
                self.logger.warning(f"⚠️ Failed to load CLIP memory: {e}")
                self.clip_memory = None
        else:
            self.logger.info("⚠️ CLIP-based prompt optimizer not available")
        
        # Priority server coordinator
        self.priority_coordinator = PriorityServerCoordinator(
            server_url=self.config.get('generation_server_url', 'http://localhost:8096'),
            max_wait_time_seconds=self.config.get('priority_access_max_wait', 60),
            status_check_interval=self.config.get('priority_access_check_interval', 1),
            priority_timeout=self.config.get('priority_access_timeout', 30),
            on_interruption_callback=self._on_priority_interruption
        )
        
        # Statistics
        self.stats = {
            'session_start': time.time(),
            'tasks_pulled': 0,
            'tasks_processed': 0,
            'tasks_skipped': 0,  # Track skipped tasks in simulation mode
            'successful_generations': 0,
            'successful_validations': 0,
            'successful_submissions': 0,
            'total_generation_time': 0.0,
            'total_validation_time': 0.0,
            'total_processing_time': 0.0,
            'total_rewards': 0.0,
            'idle_validations': 0,
            'prompts_optimized': 0,
            'reproducibility_optimizations': 0,
            'traditional_optimizations': 0,
            'clip_optimizations': 0,  # Track CLIP-based optimizations
            'clip_optimization_improvements': 0,  # Track improvements from CLIP optimization
            'clip_memory_hits': 0,  # Track when we find similar prompts in memory
            'clip_target_achievements': 0,  # Track when CLIP optimization achieves target
            'optimization_improvements': 0,
            'priority_access_timeouts': 0,  # Track priority access timeouts
            'priority_interruptions': 0,    # Track when we interrupt other jobs
            'server_unavailable_skips': 0,  # Track when we skip task pulls due to server unavailability
            'server_status_check_errors': 0, # Track server status check errors
        }
        
        self.logger.info("🎯 Continuous TRELLIS Orchestrator initialized")
        self.logger.info(f"   Output directory: {self.output_dir}")
        self.logger.info(f"   Generation server: {self.config['generation_server_url']}")
        self.logger.info(f"   Validation server: {self.config['validation_server_url']}")
        
        # Check if running in simulation mode
        if self.config.get('promptfile'):
            self.logger.info(f"🎮 SIMULATION MODE: Using prompts from {self.config['promptfile']}")
        else:
            self.logger.info("🌐 MINING MODE: Pulling tasks from validators")
        
        # Log optimization settings
        if self.config.get('enable_prompt_optimization', True):
            mode = "aggressive" if self.config.get('optimization_aggressive_mode', False) else "standard"
            detail = "minimal" if not self.config.get('log_optimization_details', True) else "detailed"
            self.logger.info(f"🔧 Prompt optimization: ENABLED ({mode} mode, {detail} logging)")
            
            # Log reproducibility settings
            if self.config.get('enable_reproducibility_optimization', True):
                min_sim = self.config.get('reproducibility_min_similarity', 0.3)
                self.logger.info(f"🔄 Reproducibility optimization: ENABLED (min similarity: {min_sim})")
            else:
                self.logger.info(f"🔄 Reproducibility optimization: DISABLED")
            
            # Log CLIP memory optimization settings
            if self.config.get('enable_clip_optimization', True):
                sim_threshold = self.config.get('clip_memory_similarity_threshold', 0.7)
                self.logger.info(f"🎯 CLIP memory optimization: ENABLED (similarity threshold: {sim_threshold})")
            else:
                self.logger.info(f"🎯 CLIP memory optimization: DISABLED")
        else:
            self.logger.info(f"🔧 Prompt optimization: DISABLED")
        self.trellis_server_url: str = "http://localhost:8096"


    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration"""
        return {
            # Bittensor settings
            'wallet_name': 'manbeast3b',
            'hotkey_name': 'm3b',
            'netuid': 89,
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
            'output_dir': './continuous_trellis_hunyuan_outputs',
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
            
            # Determinism settings
            'use_fixed_seed': True,  # True = always seed 42, False = prompt-hash based seed
            
            # Prompt optimization settings
            'enable_prompt_optimization': True,
            'optimization_aggressive_mode': False,
            'log_optimization_details': True,
            
            # Reproducibility optimization settings
            'enable_reproducibility_optimization': True,
            'reproducibility_min_similarity': 0.3,

            # CLIP-based optimization settings
            'enable_clip_optimization': True,
            'clip_target_score': 0.3,  # Target alignment score for CLIP optimization
            'clip_max_iterations': 3,  # Max iterations per CLIP optimization attempt
            'clip_min_improvement': 0.01,  # Minimum improvement threshold
            'clip_mutation_chance': 0.2,  # Probability of creative mutations
            'clip_memory_similarity_threshold': 0.7,  # Similarity threshold for memory hits

            # Priority access settings
            'priority_access_max_wait': 60, # Max seconds to wait for priority access
            'priority_access_check_interval': 1, # Seconds between status checks
            'priority_access_timeout': 30, # Max seconds to wait for priority access
            
            # Simulation settings
            'promptfile': None,  # Path to Python file with EPISODIC_TEST_PROMPTS list
            'simulation_mode': False,  # Whether running in simulation mode
        }
    
    def load_prompts_from_file(self, filepath: str) -> List[str]:
        """Dynamically load the EPISODIC_TEST_PROMPTS list from a Python file."""
        try:
            path = Path(filepath)
            spec = importlib.util.spec_from_file_location(path.stem, path.resolve())
            prompt_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(prompt_module)
            
            prompts = getattr(prompt_module, 'EPISODIC_TEST_PROMPTS', None)
            
            if prompts is None or not isinstance(prompts, list):
                self.logger.error(f"❌ Could not find a list named 'EPISODIC_TEST_PROMPTS' in {filepath}")
                return []
            
            self.logger.info(f"✅ Successfully loaded {len(prompts)} prompts from {filepath}")
            return prompts
        except Exception as e:
            self.logger.error(f"❌ Failed to load prompts from {filepath}: {e}")
            return []
    
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
            
            self.subtensor = bt.subtensor(network="test") #TODO
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
            # Check if TRELLIS server is available for priority access
            # CRITICAL: Don't pull tasks if server is unavailable - we can't process them!
            try:
                server_status = self.priority_coordinator.check_server_status()
                if not server_status.get("available", False):
                    status = server_status.get('status', 'unknown')
                    error = server_status.get('error', 'unknown error')
                    self.logger.warning(f"⏳ TRELLIS server unavailable (status: {status}, error: {error}) - SKIPPING task pull")
                    self.stats['server_unavailable_skips'] = self.stats.get('server_unavailable_skips', 0) + 1
                    return None  # Don't pull tasks when server is unavailable
                else:
                    self.logger.debug(f"✅ TRELLIS server available (status: {server_status.get('status', 'unknown')})")
            except Exception as e:
                self.logger.warning(f"⚠️ Exception checking TRELLIS server status: {e} - SKIPPING task pull")
                self.stats['server_status_check_errors'] = self.stats.get('server_status_check_errors', 0) + 1
                return None  # Don't pull tasks when we can't check server status
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
                    
                    # Create task record with response time tracking
                    prompt_hash = hashlib.sha256(resp.task.prompt.encode()).hexdigest()
                    response_received_time = time.time()
                    
                    task = TaskRecord(
                        task_id=resp.task.id,
                        prompt=resp.task.prompt,
                        prompt_hash=prompt_hash,
                        validator_uid=validator.uid,
                        validator_hotkey=validator.hotkey,
                        validator_stake=validator.stake,
                        validation_threshold=getattr(resp, 'validation_threshold', 0.6),
                        pulled_at=response_received_time
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
    
    def get_deterministic_seed(self, task: TaskRecord) -> int:
        """Generate deterministic seed based on prompt for consistent results with variety"""
        if self.config.get('use_fixed_seed', True):
            return self.config.get('fixed_seed_value', 42)  # Use configured fixed seed
        else:
            # Generate deterministic seed from prompt hash for variety but determinism
            import hashlib
            hash_obj = hashlib.sha256(task.prompt.encode())
            seed = int(hash_obj.hexdigest()[:8], 16) % (2**31)  # Convert to 32-bit int
            return seed
    
    def optimize_prompt_for_generation(self, task: TaskRecord) -> str:
        """Optimize prompt to reduce zero fidelity risk using reproducibility system first"""
        try:
            # Check if optimization is enabled
            if not self.config.get('enable_prompt_optimization', True):
                return task.prompt
            
            # Step 1: Try reproducibility system first (if available)
            if (REPRODUCIBILITY_SYSTEM_AVAILABLE and 
                self.reproducibility_system and 
                self.config.get('enable_reproducibility_optimization', True)):
                
                min_similarity = self.config.get('reproducibility_min_similarity', 0.3)
                repro_result = self.reproducibility_system.optimize_prompt_with_reproducibility(
                    task.prompt, min_similarity, run_validation=False
                )
                
                if repro_result:
                    optimized_prompt = repro_result['optimized_prompt']
                    similarity = repro_result['similarity']
                    gold_score = repro_result['gold_score']
                    
                    if self.config.get('log_optimization_details', True):
                        self.logger.info(f"🔄 Reproducibility optimization applied:")
                        self.logger.info(f"   Original: {task.prompt}")
                        self.logger.info(f"   Optimized: {optimized_prompt}")
                        self.logger.info(f"   Similarity: {similarity:.3f}")
                        self.logger.info(f"   Gold score: {gold_score:.4f}")
                    else:
                        self.logger.info(f"🔄 Reproducibility optimized (sim: {similarity:.2f}, gold: {gold_score:.3f}): '{task.prompt[:30]}...'")
                    
                    self.stats['prompts_optimized'] += 1
                    self.stats['reproducibility_optimizations'] = self.stats.get('reproducibility_optimizations', 0) + 1
                    
                    return optimized_prompt
            
            # Step 1.5: Try CLIP memory-based optimization (if available)
            if (self.clip_memory and 
                self.config.get('enable_clip_optimization', True)):
                
                clip_optimized_prompt, score = self._optimize_prompt_with_clip_memory(task.prompt)
                if clip_optimized_prompt and clip_optimized_prompt != task.prompt:
                    if self.config.get('log_optimization_details', True):
                        self.logger.info(f"🎯 CLIP memory-based optimization applied:")
                        self.logger.info(f"   Original: {task.prompt}")
                        self.logger.info(f"   Optimized: {clip_optimized_prompt}")
                        self.logger.info(f"   Score: {score:.3f}")
                    else:
                        self.logger.info(f"🎯 CLIP memory optimized: '{task.prompt[:30]}...'")
                    
                    self.stats['prompts_optimized'] += 1
                    self.stats['clip_optimizations'] = self.stats.get('clip_optimizations', 0) + 1
                    
                    return clip_optimized_prompt
            
            # Step 2: Fall back to traditional optimization if reproducibility didn't work
            if OPTIMIZED_PROMPT_OPTIMIZER_AVAILABLE:
                # Use the new fast optimizer
                # result = self.prompt_optimizer.optimize(task.prompt, use_validation=False)
                result = self.prompt_optimizer.optimize_with_examples(task.prompt)
                # result = self.prompt_optimizer.optimize_with_strategies(task.prompt)
                optimized_prompt = result #result['optimized_prompt']
                confidence = 0.8 #result['confidence']
                
                if self.config.get('log_optimization_details', True):
                    self.logger.info(f"🚀 Traditional optimization applied:")
                    self.logger.info(f"   Original: {task.prompt}")
                    self.logger.info(f"   Optimized: {optimized_prompt}")
                    self.logger.info(f"   Confidence: {confidence:.1%}")
                    # self.logger.info(f"   Similar examples: {result['similar_examples_count']}")
                    # self.logger.info(f"   Processing time: {result['processing_time']:.3f}s")
                # else:
                    # self.logger.info(f"🚀 Fast optimized (conf: {confidence:.0%}, {result['processing_time']:.2f}s): '{task.prompt[:30]}...'")
                
                self.stats['prompts_optimized'] += 1
                self.stats['traditional_optimizations'] = self.stats.get('traditional_optimizations', 0) + 1
                # if result['similar_examples_count'] > 0:
                #     self.stats['optimization_improvements'] += 1
                
                return optimized_prompt
            
            else:
                # Fallback to original optimizer
                optimization_result = self.prompt_optimizer.optimize_prompt(
                    task.prompt, 
                    aggressive=self.config.get('optimization_aggressive_mode', False)
                )
                analysis = optimization_result['analysis']
                
                # Log the analysis if enabled
                if self.config.get('log_optimization_details', True):
                    self.logger.info(f"🔍 Prompt Analysis for '{task.prompt[:50]}...':")
                    self.logger.info(f"   Risk Level: {analysis['risk_level']}")
                    
                    if analysis['risk_factors']:
                        self.logger.info(f"   Risk Factors:")
                        for factor in analysis['risk_factors']:
                            self.logger.info(f"     • {factor}")
                
                self.stats['prompts_optimized'] += 1
                self.stats['traditional_optimizations'] = self.stats.get('traditional_optimizations', 0) + 1
                return task.prompt
                
        except Exception as e:
            self.logger.error(f"❌ Prompt optimization failed: {e}")
            return task.prompt

    def _optimize_prompt_with_clip_memory(self, prompt: str) -> Optional[str]:
        """
        Optimize prompt using CLIP episodic memory by finding similar high-scoring prompts
        and using them as inspiration for LLM-based optimization.
        
        Args:
            prompt: The original prompt to optimize
            
        Returns:
            Optimized prompt or None if no optimization applied
        """
        try:
            if not self.clip_memory:
                return None
            
            # Get all high-scoring prompts from memory and sort by score
            high_scoring_prompts = []
            for original_prompt, prompt_data in self.clip_memory.get("prompt_optimizations", {}).items():
                best_score = prompt_data.get("best_score", 0)
                if best_score > 0.3:  # Only consider high-scoring prompts
                    high_scoring_prompts.append({
                        'original': original_prompt,
                        'optimized': prompt_data.get("best_prompt", original_prompt),
                        'score': best_score,
                        'keywords': prompt_data.get("keyword_scores", {}),
                        'adjectives': prompt_data.get("dynamic_adjectives", [])
                    })
            
            # Sort by alignment score in descending order (highest scores first)
            high_scoring_prompts.sort(key=lambda x: x['score'], reverse=True)
            
            if not high_scoring_prompts:
                return None
            
            # Calculate cosine similarity between current prompt and high-scoring prompts
            similarities = []
            for prompt_data in high_scoring_prompts:
                similarity = self._calculate_cosine_similarity(prompt, prompt_data['original'])
                similarities.append({
                    'prompt_data': prompt_data,
                    'similarity': similarity
                })
            
            # Sort by similarity and get top matches
            similarities.sort(key=lambda x: x['similarity'], reverse=True)
            threshold = self.config.get('clip_memory_similarity_threshold', 0.7)
            
            # Get top similar prompts that meet the threshold
            similar_prompts = [
                sim for sim in similarities 
                if sim['similarity'] >= threshold
            ][:3]  # Top 3 most similar
            
            if not similar_prompts:
                return None
            
            # Track memory hit
            self.stats['clip_memory_hits'] = self.stats.get('clip_memory_hits', 0) + 1
            
            # Use LLM to optimize based on similar high-scoring prompts
            # optimized_prompt = self._llm_optimize_with_similar_prompts(prompt, similar_prompts)
            optimized_prompt = similar_prompts[0]["prompt_data"]["optimized"]
            score = similar_prompts[0]["prompt_data"]["score"]
            if optimized_prompt and optimized_prompt != prompt:
                self.stats['clip_optimization_improvements'] = self.stats.get('clip_optimization_improvements', 0) + 1
            
            return optimized_prompt, score
            
        except Exception as e:
            self.logger.error(f"❌ CLIP memory optimization failed: {e}")
            return None
    
    def _calculate_cosine_similarity(self, prompt1: str, prompt2: str) -> float:
        """
        Calculate cosine similarity between two prompts using simple word overlap.
        This is a lightweight approach for mining - could be enhanced with embeddings.
        """
        try:
            # Simple word-based similarity
            words1 = set(prompt1.lower().split())
            words2 = set(prompt2.lower().split())
            
            if not words1 or not words2:
                return 0.0
            
            intersection = words1.intersection(words2)
            union = words1.union(words2)
            
            if not union:
                return 0.0
            
            return len(intersection) / len(union)
            
        except Exception as e:
            self.logger.debug(f"Cosine similarity calculation failed: {e}")
            return 0.0
    
    def _llm_optimize_with_similar_prompts(self, original_prompt: str, similar_prompts: List[Dict]) -> Optional[str]:
        """
        Use LLM to optimize prompt based on similar high-scoring prompts from CLIP memory.
        """
        try:
            # Prepare examples for the LLM
            examples_text = ""
            for i, sim_data in enumerate(similar_prompts[:2]):  # Use top 2 examples
                prompt_data = sim_data['prompt_data']
                similarity = sim_data['similarity']
                examples_text += f"""
Example {i+1} (Similarity: {similarity:.2f}):
Original: "{prompt_data['original']}"
Optimized: "{prompt_data['optimized']}"
Score: {prompt_data['score']:.3f}
Keywords: {list(prompt_data['keywords'].keys())[:5] if prompt_data['keywords'] else 'None'}
Adjectives: {prompt_data['adjectives'][:3] if prompt_data['adjectives'] else 'None'}

"""
            
            # Create LLM prompt
            system_prompt = """You are a Technical Director for 3D modeling. Your task is to optimize prompts for 3D asset generation by learning from successful examples.

Study the provided examples of high-scoring prompts and identify the patterns that made them successful. Then apply similar optimization techniques to the new prompt.

Focus on:
1. Adding relevant descriptive adjectives
2. Specifying geometric clarity and 3D-friendly details
3. Using technical, clean language suitable for 3D modeling
4. Avoiding artistic, photographic, or complex background descriptions

Return ONLY the optimized prompt, no explanation or quotes."""

            user_prompt = f"""Here are successful prompt optimization examples:

{examples_text}

Now optimize this prompt using similar techniques:
Original: "{original_prompt}"

Optimized prompt:"""

            # Query LLM using the same method as the CLIP optimizer
            import subprocess
            import json
            
            # Use ollama for consistency with CLIP optimizer
            # Combine system and user prompts into a single prompt
            full_prompt = f"{system_prompt}\n\n{user_prompt}"
            
            cmd = [
                "ollama", "run", "llama3.2:3b",
                full_prompt
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            
            if result.returncode == 0:
                response = result.stdout.strip()
                
                # Clean the response
                response = response.replace('"', '').replace("'", "").strip()
                
                # Basic validation
                if response and len(response) > 5 and response != original_prompt:
                    return response
                else:
                    return None
            else:
                self.logger.debug(f"LLM command failed: {result.stderr}")
                return None
                
        except Exception as e:
            self.logger.debug(f"LLM optimization with similar prompts failed: {e}")
            return None

    def _wait_for_trellis_server_ready(self, max_wait_time: int = 300) -> bool:
        """Wait for TRELLIS server to become ready, with timeout."""
        self.logger.info(f"[TRELLIS] Waiting for server to be ready (max {max_wait_time}s)...")
        start_time = time.time()
        
        while time.time() - start_time < max_wait_time:
            try:
                # Check server status
                status_url = f"{self.config['generation_server_url']}/status/"
                resp = requests.get(status_url, timeout=5)
                if resp.status_code == 200:
                    status_data = resp.json()
                    if status_data.get('ready', False):
                        elapsed = time.time() - start_time
                        self.logger.info(f"[TRELLIS] Server ready after {elapsed:.1f}s ✅")
                        return True
                    else:
                        self.logger.info(f"[TRELLIS] Server not ready yet: {status_data.get('status', 'unknown')}")
                else:
                    self.logger.info(f"[TRELLIS] Server status check failed: HTTP {resp.status_code}")
            except Exception as e:
                self.logger.info(f"[TRELLIS] Server check failed: {e}")
            
            # Wait before next check
            time.sleep(5)
        
        self.logger.error(f"[TRELLIS] Server did not become ready within {max_wait_time}s")
        return False

    def _check_trellis_server_health(self) -> bool:
        """Check if TRELLIS server is healthy and ready for validation."""
        try:
            # Check server status
            status_url = f"{self.config['generation_server_url']}/status/"
            resp = requests.get(status_url, timeout=5)
            if resp.status_code == 200:
                status_data = resp.json()
                if status_data.get('ready', False):
                    return True
                else:
                    return False
            else:
                return False
        except Exception as e:
            return False

    def _clear_trellis_gpu_cache(self):
        """Send a request to the TRELLIS server to clear GPU cache."""
        try:
            # First check server health
            if not self._check_trellis_server_health():
                self.logger.warning(f"[TRELLIS] Skipping GPU cache clear - server not healthy")
                return False
                
            url = f"{self.config['generation_server_url']}/clear_cache/"
            resp = requests.post(url, timeout=15)  # Increased timeout
            if resp.status_code == 200:
                self.logger.info(f"[TRELLIS] GPU cache cleared: {resp.json()}")
                return True
            else:
                self.logger.warning(f"[TRELLIS] Failed to clear GPU cache: HTTP {resp.status_code}")
                return False
        except Exception as e:
            self.logger.warning(f"[TRELLIS] Exception clearing GPU cache: {e}")
            return False


    async def generate_3d_model(self, task: TaskRecord) -> Optional[Dict[str, Any]]:
        """Generate 3D model using TRELLIS server with prompt optimization"""
        self.logger.info(f"🎨 Generating 3D model: '{task.prompt}' (task: {task.task_id})")
        
        try:
            # CRITICAL: Wait for priority access to the server
            # This is where we ensure subnet tasks get priority over optimizer tasks
            if not self.priority_coordinator.wait_for_priority_access(task.task_id):
                self.logger.error(f"❌ PRIORITY ACCESS TIMEOUT for task {task.task_id} - subnet task will be missed!")
                task.priority_access_timeout = True  # Mark this task as having priority access timeout
                return None
            
            # Mark the start of our priority job
            self.priority_coordinator.mark_priority_job_start(task.task_id, task.prompt)
            
            # Step 1: Optimize prompt to reduce zero fidelity risk
            optimized_prompt = self.optimize_prompt_for_generation(task)
            
            # Clear cache on the server using priority coordinator
            self.priority_coordinator.clear_server_cache()

            # Step 2: Get deterministic seed
            deterministic_seed = self.get_deterministic_seed(task)
            self.logger.info(f"   🎲 Using deterministic seed: {deterministic_seed}")
            
            generation_start = time.time()
            
            # Call TRELLIS generation server with optimized prompt and deterministic seed
            response = requests.post(
                f"{self.config['generation_server_url']}/generate/",
                data={
                    'prompt': optimized_prompt,  # Use optimized prompt
                    'seed': deterministic_seed,  # Use deterministic seed
                    'return_compressed': True
                },
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
                
                # Mark the completion of our priority job
                self.priority_coordinator.mark_priority_job_end(task.task_id)
                
                return {'ply_data': ply_data, 'compression_ratio': compression_ratio}
            else:
                self.logger.error(f"❌ Generation failed: HTTP {response.status_code}")
                # Mark the completion of our priority job even on failure
                self.priority_coordinator.mark_priority_job_end(task.task_id)
                return None
        
        except Exception as e:
            self.logger.error(f"❌ Generation exception: {e}")
            # Mark the completion of our priority job even on exception
            self.priority_coordinator.mark_priority_job_end(task.task_id)
            return None
    
    def _validate_prompt(self, original_prompt: str, optimized_prompt: str = None) -> float:
        """Run validation with conda environment (same as simulator)."""
        try:
            # Wait for server to be ready before validation
            if not self._wait_for_trellis_server_ready():
                self.logger.warning(f"      ⚠️ TRELLIS server did not become ready, skipping validation")
                return 0.0
            
            self.logger.info("      🔍 Validating...")
            
            # Use optimized prompt for generation if provided, otherwise use original
            if optimized_prompt and optimized_prompt != original_prompt:
                self.logger.info(f"      📝 Using optimized prompt for generation: '{optimized_prompt[:50]}...'")
                self.logger.info(f"      🎯 Computing scores against original prompt: '{original_prompt[:50]}...'")
                cmd = [
                    "bash", "-c",
                    f"source /home/mbhat/miniconda/bin/activate && conda activate trellis_new && python subnet_accurate_validator.py \"{original_prompt}\" \"{optimized_prompt}\""
                ]
            else:
                self.logger.info(f"      📝 Using same prompt for generation and validation: '{original_prompt[:50]}...'")
                cmd = [
                    "bash", "-c",
                    f"source /home/mbhat/miniconda/bin/activate && conda activate trellis_new && python subnet_accurate_validator.py \"{original_prompt}\""
                ]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
            
            # Log the validation output for debugging
            if result.stderr:
                self.logger.warning(f"      ⚠️ Validation stderr: {result.stderr.strip()}")
            
            if result.returncode != 0:
                self.logger.warning(f"   ❌ Validation failed (return code {result.returncode})")
                if "CUDA" in result.stderr or "out of memory" in result.stderr.lower():
                    self.logger.warning(f"   🔥 CUDA OOM detected in validation - clearing cache")
                    if TORCH_AVAILABLE and torch.cuda.is_available():
                        torch.cuda.empty_cache()
                        time.sleep(2)  # Brief pause for memory cleanup
                return 0.0
            
            # Read and log the validation results
            try:
                with open("subnet_validation_results.json", 'r') as f:
                    data = json.load(f)
                    score = data.get("validation_engine_score", 0.0)
                    
                    # Log detailed validation results
                    self.logger.info(f"      📊 Validation Results:")
                    self.logger.info(f"         🏆 Validation Engine Score: {score:.4f}")
                    self.logger.info(f"         🤝 Alignment Score: {data.get('alignment_score', 0):.4f}")
                    self.logger.info(f"         💎 Quality Score: {data.get('quality_score', 0):.4f}")
                    self.logger.info(f"         🎭 Demo Fidelity Score: {data.get('demo_fidelity_score', 0):.4f}")
                    self.logger.info(f"         🎯 Task Fidelity Score: {data.get('task_fidelity_score', 0):.4f}")
                    self.logger.info(f"         ✅ Validation Passed: {data.get('validation_passed', False)}")
                    
                    # Check for zero score and provide more context
                    if score == 0.0:
                        self.logger.warning(f"   🔧 Zero validation score detected!")
                        self.logger.warning(f"      Alignment: {data.get('alignment_score', 0):.4f}")
                        self.logger.warning(f"      Quality: {data.get('quality_score', 0):.4f}")
                        self.logger.warning(f"      Demo Fidelity: {data.get('demo_fidelity_score', 0):.4f}")
                        
                        if TORCH_AVAILABLE and torch.cuda.is_available():
                            self.logger.warning(f"   🔧 Clearing CUDA cache due to zero score")
                            torch.cuda.empty_cache()
                        
            except FileNotFoundError:
                self.logger.error(f"      ❌ Validation results file not found")
                return 0.0
            except json.JSONDecodeError as e:
                self.logger.error(f"      ❌ Invalid JSON in validation results: {e}")
                return 0.0
                
        except Exception as e:
            self.logger.error(f"      ❌ Validation error: {e}")
            return 0.0
        return score

    async def validate_model(self, task: TaskRecord, ply_data: bytes) -> Optional[float]:
        """Validate generated model using the subnet validator (same as simulator)"""
        if not self.config['validate_generations']:
            return None
        
        self.logger.info(f"📊 Validating model: '{task.prompt[:50]}...'")
        
        try:
            validation_start = time.time()
            
            # Clear cache again before validation to ensure clean state
            self.logger.info(f"      🧹 Clearing cache before validation...")
            cache_cleared = self._clear_trellis_gpu_cache()
            if not cache_cleared:
                self.logger.warning(f"      ⚠️ Failed to clear cache before validation, proceeding anyway")
            
            # Use the subnet validator function (same as simulator)
            # Check if we have an optimized prompt for this task
            optimized_prompt = getattr(task, 'optimized_prompt', None)
            score = self._validate_prompt(task.prompt, optimized_prompt)
            
            validation_time = time.time() - validation_start
            task.validation_time = validation_time
            task.local_validation_score = score
            
            self.logger.info(f"✅ Validation completed in {validation_time:.2f}s")
            self.logger.info(f"   Score: {score:.4f}")
            
            self.stats['successful_validations'] += 1
            self.stats['total_validation_time'] += validation_time
            
            return score
        
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
            
            # Calculate total processing time from validator response to submission
            if task.pulled_at:
                task.total_processing_time = task.submitted_at - task.pulled_at
                self.logger.info(f"⏱️ Total processing time: {task.total_processing_time:.2f}s (from validator response to submission)")
            
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
                if task.total_processing_time:
                    self.stats['total_processing_time'] += task.total_processing_time
                
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
        """Process a single task end-to-end with priority access"""
        self.logger.info(f"🔄 Processing task {task.task_id}: '{task.prompt}'")
        
        task.processed_at = time.time()
        self.stats['tasks_processed'] += 1
        
        try:
            # Step 1: Generate 3D model with priority access
            generation_result = await self.generate_3d_model(task)
            if not generation_result:
                # Check if this was due to priority access timeout
                if hasattr(task, 'priority_access_timeout') and task.priority_access_timeout:
                    self.logger.error(f"❌ PRIORITY ACCESS TIMEOUT for task {task.task_id} - subnet task missed!")
                    self.stats['priority_access_timeouts'] = self.stats.get('priority_access_timeouts', 0) + 1
                else:
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
                    'avg_total_processing_time': self.stats['total_processing_time'] / max(1, self.stats['successful_submissions']),
                    'total_rewards': self.stats['total_rewards'],
                    'rewards_per_hour': self.stats['total_rewards'] / max(0.1, uptime_hours),
                    'optimization_rate': (self.stats['optimization_improvements'] / max(1, self.stats['prompts_optimized'])) * 100,
                    'prompts_optimized': self.stats['prompts_optimized'],
                    'optimization_improvements': self.stats['optimization_improvements']
                }
            }
            
            # Save to file
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            stats_file = self.output_dir / f"continuous_stats_hunyuan_{timestamp}.json"
            
            with open(stats_file, 'w') as f:
                json.dump(stats, f, indent=2)
            
            self.logger.info(f"📊 Statistics saved to {stats_file}")
            
        except Exception as e:
            self.logger.error(f"❌ Failed to save statistics: {e}")
    
    def print_status(self):
        """Print current status"""
        uptime_hours = (time.time() - self.start_time) / 3600
        
        # Check if running in simulation mode
        if self.config.get('promptfile'):
            self.logger.info("📊 CONTINUOUS ORCHESTRATOR STATUS (SIMULATION MODE)")
        else:
            self.logger.info("📊 CONTINUOUS ORCHESTRATOR STATUS (MINING MODE)")
        self.logger.info("="*60)
        self.logger.info(f"Uptime: {uptime_hours:.2f} hours")
        self.logger.info(f"Tasks pulled: {self.stats['tasks_pulled']}")
        self.logger.info(f"Tasks processed: {self.stats['tasks_processed']}")
        if self.config.get('promptfile'):
            self.logger.info(f"Tasks skipped: {self.stats.get('tasks_skipped', 0)}")
        self.logger.info(f"Successful generations: {self.stats['successful_generations']}")
        self.logger.info(f"Successful validations: {self.stats['successful_validations']}")
        self.logger.info(f"Successful submissions: {self.stats['successful_submissions']}")
        self.logger.info(f"Total rewards: {self.stats['total_rewards']:.6f} TAO")
        self.logger.info(f"Idle validations: {self.stats['idle_validations']}")
        self.logger.info(f"Prompts optimized: {self.stats['prompts_optimized']}")
        self.logger.info(f"Reproducibility optimizations: {self.stats.get('reproducibility_optimizations', 0)}")
        self.logger.info(f"Traditional optimizations: {self.stats.get('traditional_optimizations', 0)}")
        self.logger.info(f"CLIP memory optimizations: {self.stats.get('clip_optimizations', 0)}")
        self.logger.info(f"CLIP memory hits: {self.stats.get('clip_memory_hits', 0)}")
        self.logger.info(f"CLIP optimization improvements: {self.stats.get('clip_optimization_improvements', 0)}")
        self.logger.info(f"Optimization improvements: {self.stats['optimization_improvements']}")
        
        # Priority access statistics
        self.logger.info(f"Priority access timeouts: {self.stats.get('priority_access_timeouts', 0)}")
        self.logger.info(f"Priority interruptions: {self.stats.get('priority_interruptions', 0)}")
        self.logger.info(f"Server unavailable skips: {self.stats.get('server_unavailable_skips', 0)}")
        self.logger.info(f"Server status check errors: {self.stats.get('server_status_check_errors', 0)}")
        
        if uptime_hours > 0:
            self.logger.info(f"Tasks/hour: {self.stats['tasks_processed'] / uptime_hours:.1f}")
            self.logger.info(f"Rewards/hour: {self.stats['total_rewards'] / uptime_hours:.6f} TAO")
            
        # Processing time statistics
        if self.stats['successful_submissions'] > 0:
            avg_processing_time = self.stats['total_processing_time'] / self.stats['successful_submissions']
            self.logger.info(f"Average total processing time: {avg_processing_time:.2f}s")
            
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
    
    async def continuous_mining_loop(self):
        """Main continuous mining loop"""
        self.logger.info("🚀 Starting continuous TRELLIS mining...")
        
        # Check if running in simulation mode
        if self.config.get('promptfile'):
            await self._run_simulation_mode()
            return
        
        # Setup Bittensor for mining mode
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
        
        try:
            while self.running:
                current_time = time.time()
                
                # Periodic validator refresh (every 10 minutes to catch changes)
                if current_time - last_validator_refresh > 600:
                    self.refresh_validators()
                    last_validator_refresh = current_time
                
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
    
    async def _run_simulation_mode(self):
        """Run in simulation mode using prompts from file"""
        self.logger.info("🎮 Starting TRELLIS simulation mode...")
        
        # Load prompts from file
        prompts = self.load_prompts_from_file(self.config['promptfile'])
        if not prompts:
            self.logger.error("❌ No prompts loaded from file")
            return
        
        # Create tasks from prompts
        tasks_to_process = []
        for i, prompt_text in enumerate(prompts):
            prompt_hash = hashlib.sha256(prompt_text.encode()).hexdigest()
            if self.db.has_processed_prompt(prompt_hash):
                self.logger.info(f"⏭️ Skipping already processed prompt: '{prompt_text[:50]}...'")
                self.stats['tasks_skipped'] = self.stats.get('tasks_skipped', 0) + 1
                continue
                
            task = TaskRecord(
                task_id=f"sim_{i+1}",
                prompt=prompt_text,
                prompt_hash=prompt_hash,
                validator_uid=-1,  # Dummy validator for simulation
                validator_hotkey="simulator",
                validator_stake=0.0,
                validation_threshold=0.0,
                pulled_at=time.time()
            )
            tasks_to_process.append(task)
        
        self.logger.info(f"📋 Found {len(tasks_to_process)} new prompts to process")
        
        self.running = True
        self.start_time = time.time()
        
        try:
            for task in tasks_to_process:
                if not self.running:
                    self.logger.info("🛑 Simulation interrupted")
                    break
                
                # Process task (generate and validate, but don't submit)
                success = await self._process_simulation_task(task)
                if success:
                    self.stats['tasks_processed'] += 1
                    
        except KeyboardInterrupt:
            self.logger.info("🛑 Simulation interrupted by user")
        except Exception as e:
            self.logger.error(f"❌ Simulation error: {e}")
            traceback.print_exc()
        finally:
            self.running = False
            self.print_status()
            self.save_statistics()
            self.logger.info("🏁 Simulation stopped")
    
    async def _generate_3d_model_simulation(self, task: TaskRecord) -> Optional[Dict[str, Any]]:
        """Generate 3D model for simulation mode (without priority coordination)"""
        self.logger.info(f"🎨 Generating 3D model: '{task.prompt}' (task: {task.task_id})")
        
        try:
            # Step 1: Wait for server to be ready before generation
            if not self._wait_for_trellis_server_ready():
                self.logger.error(f"❌ TRELLIS server did not become ready, skipping generation")
                return None
            
            # Step 2: Optimize prompt to reduce zero fidelity risk
            optimized_prompt = self.optimize_prompt_for_generation(task)
            
            # Step 3: Clear cache on the server
            cache_cleared = self._clear_trellis_gpu_cache()
            if not cache_cleared:
                self.logger.warning(f"⚠️ Failed to clear GPU cache, proceeding anyway")

            # Step 4: Get deterministic seed
            deterministic_seed = self.get_deterministic_seed(task)
            self.logger.info(f"   🎲 Using deterministic seed: {deterministic_seed}")
            
            generation_start = time.time()
            
            # Step 5: Call TRELLIS generation server with optimized prompt and deterministic seed
            response = requests.post(
                f"{self.config['generation_server_url']}/generate/",
                data={
                    'prompt': optimized_prompt,  # Use optimized prompt
                    'seed': deterministic_seed,  # Use deterministic seed
                    'return_compressed': True
                },
                timeout=self.config['generation_timeout']
            )
            
            generation_time = time.time() - generation_start
            task.generation_time = generation_time
            
            if response.status_code == 200:
                ply_data = response.content
                
                # Get metadata from headers to check compression status
                compression_ratio = response.headers.get('X-Compression-Ratio', 'unknown')

                # Save PLY file
                if self.config.get('save_intermediate_results', True):
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

    async def _process_simulation_task(self, task: TaskRecord) -> bool:
        """Process a single simulation task (generate and validate only)"""
        self.logger.info(f"🔄 Processing simulation task {task.task_id}: '{task.prompt}'")
        
        task.processed_at = time.time()
        
        try:
            # Step 1: Generate 3D model (using simulation-specific method)
            generation_result = await self._generate_3d_model_simulation(task)
            if not generation_result:
                self.logger.error(f"❌ Generation failed for task {task.task_id}")
                self.db.save_task(task)
                return False
            
            # Step 2: Validate locally (if enabled)
            if self.config.get('validate_generations', True):
                await self.validate_model(task, generation_result['ply_data'])
            
            # Step 3: Record total time and save
            if task.pulled_at:
                task.total_processing_time = time.time() - task.pulled_at
            
            self.db.save_task(task)
            self.logger.info(f"✅ Simulation task {task.task_id} finished processing")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Simulation task processing failed: {e}")
            traceback.print_exc()
            self.db.save_task(task)
            return False
    
    def _on_priority_interruption(self):
        """Callback when priority interruption occurs"""
        self.stats['priority_interruptions'] = self.stats.get('priority_interruptions', 0) + 1
        self.logger.info(f"📊 Priority interruption tracked: {self.stats['priority_interruptions']} total")

async def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Continuous TRELLIS Orchestrator")
    parser.add_argument("--promptfile", help="Path to Python file with EPISODIC_TEST_PROMPTS list (simulation mode)")
    parser.add_argument("--no-harvest", action="store_true", help="Disable task harvesting")
    parser.add_argument("--no-validate", action="store_true", help="Disable validation")
    parser.add_argument("--no-submit", action="store_true", help="Disable result submission")
    parser.add_argument("--generation-server", default="http://localhost:8096", help="TRELLIS generation server URL")
    parser.add_argument("--validation-server", default="http://localhost:10006", help="Validation server URL")
    parser.add_argument("--output-dir", default="./continuous_trellis_hunyuan_outputs", help="Output directory")
    parser.add_argument("--min-score", type=float, default=0.3, help="Minimum local validation score")
    
    # Prompt optimization arguments
    parser.add_argument("--no-optimize", action="store_true", help="Disable prompt optimization")
    parser.add_argument("--aggressive-optimize", action="store_true", help="Enable aggressive optimization mode")
    parser.add_argument("--quiet-optimize", action="store_true", help="Reduce optimization logging detail")
    
    # Reproducibility optimization arguments
    parser.add_argument("--no-reproducibility", action="store_true", help="Disable reproducibility optimization")
    parser.add_argument("--reproducibility-similarity", type=float, default=0.51, help="Minimum similarity threshold for reproducibility (default: 0.3)")
    
    # CLIP memory optimization arguments
    parser.add_argument("--no-clip-optimization", action="store_true", help="Disable CLIP memory-based optimization")
    parser.add_argument("--clip-similarity-threshold", type=float, default=0.5, help="Similarity threshold for CLIP memory hits (default: 0.)")
    
    # Determinism arguments
    parser.add_argument("--variable-seeds", action="store_true", help="Use prompt-hash based seeds (default: fixed seed 42)")
    parser.add_argument("--seed", type=int, default=42, help="Fixed seed to use when not using variable seeds")
    
    args = parser.parse_args()
    
    # Build config
    config = {}
    
    # Simulation mode configuration
    if args.promptfile:
        config['promptfile'] = args.promptfile
        config['simulation_mode'] = True
        # In simulation mode, disable submission and harvesting
        config['submit_results'] = False
        config['harvest_tasks'] = False
    else:
        # Mining mode configuration
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
    
    # Reproducibility optimization configuration
    if args.no_reproducibility:
        config['enable_reproducibility_optimization'] = False
    config['reproducibility_min_similarity'] = args.reproducibility_similarity
    
    # CLIP memory optimization configuration
    if args.no_clip_optimization:
        config['enable_clip_optimization'] = False
    config['clip_memory_similarity_threshold'] = args.clip_similarity_threshold
    
    # Determinism configuration
    if args.variable_seeds:
        config['use_fixed_seed'] = False
    config['fixed_seed_value'] = args.seed
    
    # Create and run orchestrator
    orchestrator = ContinuousTrellisOrchestrator(config)
    
    try:
        await orchestrator.continuous_mining_loop()
    except Exception as e:
        logger.error(f"❌ Orchestrator failed: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(main()) 