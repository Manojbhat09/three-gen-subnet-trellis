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
from pathlib import Path
from typing import List, Dict, Optional, Any, Set
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
import re
import signal
import atexit
import uuid
import socket
import os

# Import the prompt optimizer
try:
    # from smart_prompt_optimizer_fixed import OptimizedPromptOptimizer
    # from llm_prompt_optimizer_v7_f1 import LLMPromptOptimizer
    from llm_prompt_optimizer_v12_f1_lora import LLMPromptOptimizer
    OPTIMIZED_PROMPT_OPTIMIZER_AVAILABLE = True
    print("✅ Using new performance-optimized prompt optimizer")
except ImportError:
    from prompt_optimizer import TrellisPromptOptimizer
    OPTIMIZED_PROMPT_OPTIMIZER_AVAILABLE = False
    print("⚠️ Falling back to original prompt optimizer")

# Import the organic LoRA router
try:
    from final_organic_router import FinalOrganicRouter
    ORGANIC_LORA_ROUTER_AVAILABLE = False
    print("✅ Using organic LoRA router with 100% pattern learning accuracy")
except ImportError:
    ORGANIC_LORA_ROUTER_AVAILABLE = False
    print("⚠️ Organic LoRA router not available - using default model")

# Import the reproducibility system
try:
    from llm_close_prompt_reproducibility_test import LLMClosePromptReproducibility
    REPRODUCIBILITY_SYSTEM_AVAILABLE = True
    print("✅ Using reproducibility system for pre-optimization")
except ImportError:
    REPRODUCIBILITY_SYSTEM_AVAILABLE = False
    print("⚠️ Reproducibility system not available")

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
        logging.FileHandler('continuous_trellis.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class PriorityServerCoordinator:
    """
    Priority-based server coordinator that gives the orchestrator HIGH PRIORITY access.
    This allows time-critical subnet tasks to bypass or interrupt other processes.
    """
    
    def __init__(self, server_url: str = "http://localhost:8097", 
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
    
    # Enhanced cooldown and validation tracking
    throttle_period: int = 0
    cooldown_violations: int = 0
    validation_locked_until: Optional[float] = None
    last_submit_time: Optional[float] = None
    
    # Emergency cooldown management
    emergency_blacklist_until: Optional[float] = None
    last_violation_check: Optional[float] = None

    def __post_init__(self):
        if self.recent_prompts is None:
            self.recent_prompts = set()

class ValidatorStatePersistence:
    """
    Handles saving and loading validator states to maintain cooldowns, violations, 
    and blacklists across script restarts.
    """
    
    def __init__(self, state_file: str = "validator_states.json"):
        self.state_file = Path(state_file)
        self.backup_file = Path(f"{state_file}.backup")
        self.logger = logging.getLogger(__name__)
        
    def save_validator_states(self, validators: Dict[int, 'ValidatorState']) -> bool:
        """
        Save validator states to disk.
        
        Args:
            validators: Dictionary of validator states to save
            
        Returns:
            True if save was successful, False otherwise
        """
        try:
            current_time = time.time()
            state_data = {
                'saved_at': current_time,
                'saved_at_readable': datetime.fromtimestamp(current_time).isoformat(),
                'version': '1.0',
                'validators': {}
            }
            
            for uid, validator in validators.items():
                # Only save essential state information
                validator_state = {
                    'uid': validator.uid,
                    'stake': validator.stake,
                    'is_active': validator.is_active,
                    
                    # Cooldown and violation tracking (CRITICAL)
                    'cooldown_until': validator.cooldown_until,
                    'cooldown_violations': validator.cooldown_violations,
                    'throttle_period': validator.throttle_period,
                    
                    # Validation and emergency state (CRITICAL)
                    'validation_locked_until': validator.validation_locked_until,
                    'emergency_blacklist_until': validator.emergency_blacklist_until,
                    'last_submit_time': validator.last_submit_time,
                    'last_violation_check': validator.last_violation_check,
                    
                    # Performance tracking
                    'total_tasks_received': validator.total_tasks_received,
                    'total_tasks_submitted': validator.total_tasks_submitted,
                    'total_successful_submissions': validator.total_successful_submissions,
                    'average_score': validator.average_score,
                    
                    # History for learning (limited to prevent bloat)
                    'violation_history': getattr(validator, 'violation_history', [])[-5:],  # Last 5 only
                    'buffer_history': getattr(validator, 'buffer_history', [])[-3:],  # Last 3 only
                }
                
                state_data['validators'][str(uid)] = validator_state
            
            # Create backup of existing file
            if self.state_file.exists():
                import shutil
                shutil.copy2(self.state_file, self.backup_file)
            
            # Save new state
            with open(self.state_file, 'w') as f:
                json.dump(state_data, f, indent=2)
            
            self.logger.info(f"💾 Saved {len(validators)} validator states to {self.state_file}")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Failed to save validator states: {e}")
            self.logger.error(f"   Traceback: {traceback.format_exc()}")
            return False
    
    def load_validator_states(self) -> Dict[int, Dict[str, Any]]:
        """
        Load validator states from disk.
        
        Returns:
            Dictionary of validator states keyed by UID
        """
        try:
            if not self.state_file.exists():
                self.logger.info(f"📁 No existing state file found at {self.state_file}")
                return {}
            
            with open(self.state_file, 'r') as f:
                state_data = json.load(f)
            
            # Validate file format
            if 'validators' not in state_data:
                self.logger.warning(f"⚠️ Invalid state file format - missing 'validators' key")
                return {}
            
            saved_at = state_data.get('saved_at', 0)
            saved_at_readable = state_data.get('saved_at_readable', 'unknown')
            current_time = time.time()
            age_hours = (current_time - saved_at) / 3600
            
            self.logger.info(f"📂 Loading validator states from {self.state_file}")
            self.logger.info(f"   File saved: {saved_at_readable}")
            self.logger.info(f"   File age: {age_hours:.1f} hours")
            
            # Convert string UIDs back to integers
            validator_states = {}
            loaded_count = 0
            expired_cooldowns = 0
            active_violations = 0
            
            for uid_str, validator_data in state_data['validators'].items():
                try:
                    uid = int(uid_str)
                    
                    # Check if cooldowns have expired
                    if validator_data.get('cooldown_until'):
                        if current_time >= validator_data['cooldown_until']:
                            validator_data['cooldown_until'] = None
                            expired_cooldowns += 1
                    
                    if validator_data.get('validation_locked_until'):
                        if current_time >= validator_data['validation_locked_until']:
                            validator_data['validation_locked_until'] = None
                    
                    if validator_data.get('emergency_blacklist_until'):
                        if current_time >= validator_data['emergency_blacklist_until']:
                            validator_data['emergency_blacklist_until'] = None
                        else:
                            # Still blacklisted
                            validator_data['is_active'] = False
                    
                    # Count active violations
                    if validator_data.get('cooldown_violations', 0) > 0:
                        active_violations += 1
                    
                    validator_states[uid] = validator_data
                    loaded_count += 1
                    
                except (ValueError, KeyError) as e:
                    self.logger.warning(f"⚠️ Skipping invalid validator data for UID {uid_str}: {e}")
            
            self.logger.info(f"✅ Loaded {loaded_count} validator states")
            self.logger.info(f"   Expired cooldowns cleaned: {expired_cooldowns}")
            self.logger.info(f"   Validators with active violations: {active_violations}")
            
            # If file is very old (>24 hours), suggest caution
            if age_hours > 24:
                self.logger.warning(f"⚠️ State file is {age_hours:.1f} hours old - some data may be stale")
            
            return validator_states
            
        except Exception as e:
            self.logger.error(f"❌ Failed to load validator states: {e}")
            self.logger.error(f"   Attempting to load backup...")
            
            # Try backup file
            try:
                if self.backup_file.exists():
                    with open(self.backup_file, 'r') as f:
                        backup_data = json.load(f)
                    self.logger.warning(f"🔄 Loaded backup state file")
                    return self.load_validator_states_from_data(backup_data)
            except Exception as backup_error:
                self.logger.error(f"❌ Backup file also failed: {backup_error}")
            
            return {}
    
    def load_validator_states_from_data(self, state_data: Dict) -> Dict[int, Dict[str, Any]]:
        """Helper method to load states from parsed JSON data"""
        if 'validators' not in state_data:
            return {}
        
        validator_states = {}
        for uid_str, validator_data in state_data['validators'].items():
            try:
                uid = int(uid_str)
                validator_states[uid] = validator_data
            except ValueError:
                continue
        
        return validator_states
    
    def cleanup_old_states(self, max_age_hours: float = 168):  # 7 days default
        """
        Clean up very old state files.
        
        Args:
            max_age_hours: Maximum age in hours before cleanup
        """
        try:
            if self.state_file.exists():
                file_age = time.time() - self.state_file.stat().st_mtime
                age_hours = file_age / 3600
                
                if age_hours > max_age_hours:
                    backup_name = f"validator_states_old_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
                    self.state_file.rename(backup_name)
                    self.logger.info(f"🧹 Archived old state file: {backup_name}")
                    
        except Exception as e:
            self.logger.warning(f"⚠️ Failed to cleanup old states: {e}")

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
        
        # SHARED TASK TRACKING TABLE - Prevents duplicate task processing across instances
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS shared_task_tracking (
                task_id TEXT PRIMARY KEY,
                validator_uid INTEGER NOT NULL,
                status TEXT DEFAULT 'in_progress',
                instance_id TEXT NOT NULL,
                started_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                timeout_at TIMESTAMP,
                completed_at TIMESTAMP,
                instance_hostname TEXT,
                instance_pid INTEGER
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
        
        # Create indexes for shared task tracking
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_shared_task_status ON shared_task_tracking(status)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_shared_task_validator ON shared_task_tracking(validator_uid, status)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_shared_task_timeout ON shared_task_tracking(timeout_at)')
        
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
    
    # ===== SHARED TASK TRACKING METHODS =====
    # These methods prevent duplicate task processing across multiple mining instances
    
    def acquire_task_lock(self, task_id: str, validator_uid: int, instance_id: str, timeout_minutes: int = 2) -> bool:
        """
        Try to acquire a lock on a task to prevent other instances from processing it.
        Returns True if lock was acquired, False if task is already being processed.
        """
        import socket
        import os
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            # Check if task is already being processed
            cursor.execute('''
                SELECT status, instance_id, started_at, timeout_at 
                FROM shared_task_tracking 
                WHERE task_id = ?
            ''', (task_id,))
            
            existing = cursor.fetchone()
            
            if existing:
                status, existing_instance_id, started_at, timeout_at = existing
                
                # If task is completed, allow reprocessing
                if status == 'completed':
                    cursor.execute('DELETE FROM shared_task_tracking WHERE task_id = ?', (task_id,))
                    conn.commit()
                # If task is in progress but timed out, allow takeover
                elif status == 'in_progress' and timeout_at and time.time() > time.mktime(time.strptime(timeout_at, '%Y-%m-%d %H:%M:%S')):
                    cursor.execute('DELETE FROM shared_task_tracking WHERE task_id = ?', (task_id,))
                    conn.commit()
                # If task is actively being processed by another instance, deny lock
                elif status == 'in_progress':
                    conn.close()
                    return False
            
            # Calculate timeout time
            timeout_time = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time() + (timeout_minutes * 60)))
            
            # Acquire lock
            cursor.execute('''
                INSERT OR REPLACE INTO shared_task_tracking 
                (task_id, validator_uid, status, instance_id, started_at, timeout_at, instance_hostname, instance_pid)
                VALUES (?, ?, 'in_progress', ?, CURRENT_TIMESTAMP, ?, ?, ?)
            ''', (task_id, validator_uid, instance_id, timeout_time, socket.gethostname(), os.getpid()))
            
            conn.commit()
            conn.close()
            return True
            
        except Exception as e:
            conn.close()
            print(f"Error acquiring task lock: {e}")
            return False
    
    def release_task_lock(self, task_id: str, instance_id: str, status: str = 'completed'):
        """
        Release the lock on a task and mark it as completed or failed.
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            cursor.execute('''
                UPDATE shared_task_tracking 
                SET status = ?, completed_at = CURRENT_TIMESTAMP
                WHERE task_id = ? AND instance_id = ?
            ''', (status, task_id, instance_id))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            conn.close()
            print(f"Error releasing task lock: {e}")
    
    def is_validator_busy(self, validator_uid: int, exclude_instance_id: str = None) -> bool:
        """
        Check if a validator is currently busy processing tasks.
        Returns True if validator has active tasks, False if available.
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            # Count active tasks for this validator
            if exclude_instance_id:
                cursor.execute('''
                    SELECT COUNT(*) FROM shared_task_tracking 
                    WHERE validator_uid = ? AND status = 'in_progress' AND instance_id != ?
                ''', (validator_uid, exclude_instance_id))
            else:
                cursor.execute('''
                    SELECT COUNT(*) FROM shared_task_tracking 
                    WHERE validator_uid = ? AND status = 'in_progress'
                ''', (validator_uid,))
            
            active_tasks = cursor.fetchone()[0]
            conn.close()
            
            return active_tasks > 0
            
        except Exception as e:
            conn.close()
            print(f"Error checking validator busy status: {e}")
            return False
    
    def cleanup_expired_locks(self, timeout_minutes: int = 2):
        """
        Clean up expired task locks that are older than the timeout period.
        This allows other instances to take over stalled tasks.
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            # Find expired locks
            timeout_time = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time() - (timeout_minutes * 60)))
            
            cursor.execute('''
                SELECT task_id, instance_id, started_at 
                FROM shared_task_tracking 
                WHERE status = 'in_progress' AND started_at < ?
            ''', (timeout_time,))
            
            expired_locks = cursor.fetchall()
            
            if expired_locks:
                print(f"🧹 Cleaning up {len(expired_locks)} expired task locks...")
                
                for task_id, instance_id, started_at in expired_locks:
                    print(f"   Expired: {task_id} (instance: {instance_id}, started: {started_at})")
                
                # Remove expired locks
                cursor.execute('''
                    DELETE FROM shared_task_tracking 
                    WHERE status = 'in_progress' AND started_at < ?
                ''', (timeout_time,))
                
                conn.commit()
                print(f"✅ Cleaned up {len(expired_locks)} expired locks")
            
            conn.close()
            
        except Exception as e:
            conn.close()
            print(f"Error cleaning up expired locks: {e}")
    
    def get_available_validators(self, exclude_instance_id: str = None) -> List[int]:
        """
        Get list of validator UIDs that are not currently busy processing tasks.
        This helps distribute work across validators and instances.
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            # Get all validators that don't have active tasks
            if exclude_instance_id:
                cursor.execute('''
                    SELECT DISTINCT v.uid 
                    FROM validators v
                    LEFT JOIN shared_task_tracking st ON v.uid = st.validator_uid AND st.status = 'in_progress'
                    WHERE v.is_active = 1 
                    AND (st.validator_uid IS NULL OR st.instance_id = ?)
                    ORDER BY v.stake DESC
                ''', (exclude_instance_id,))
            else:
                cursor.execute('''
                    SELECT DISTINCT v.uid 
                    FROM validators v
                    LEFT JOIN shared_task_tracking st ON v.uid = st.validator_uid AND st.status = 'in_progress'
                    WHERE v.is_active = 1 
                    AND st.validator_uid IS NULL
                    ORDER BY v.stake DESC
                ''')
            
            available_uids = [row[0] for row in cursor.fetchall()]
            conn.close()
            
            return available_uids
            
        except Exception as e:
            conn.close()
            print(f"Error getting available validators: {e}")
            return []
    
    def get_task_processing_stats(self) -> dict:
        """
        Get statistics about task processing across all instances.
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            # Count tasks by status
            cursor.execute('''
                SELECT status, COUNT(*) as count
                FROM shared_task_tracking
                GROUP BY status
            ''')
            
            status_counts = dict(cursor.fetchall())
            
            # Count tasks by instance
            cursor.execute('''
                SELECT instance_id, COUNT(*) as count
                FROM shared_task_tracking
                WHERE status = 'in_progress'
                GROUP BY instance_id
            ''')
            
            instance_counts = dict(cursor.fetchall())
            
            # Count tasks by validator
            cursor.execute('''
                SELECT validator_uid, COUNT(*) as count
                FROM shared_task_tracking
                WHERE status = 'in_progress'
                GROUP BY validator_uid
            ''')
            
            validator_counts = dict(cursor.fetchall())
            
            conn.close()
            
            return {
                'status_counts': status_counts,
                'instance_counts': instance_counts,
                'validator_counts': validator_counts,
                'total_tasks': sum(status_counts.values()),
                'active_tasks': status_counts.get('in_progress', 0),
                'completed_tasks': status_counts.get('completed', 0)
            }
            
        except Exception as e:
            conn.close()
            print(f"Error getting task processing stats: {e}")
            return {}

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
        
        # Generate unique instance ID for shared task tracking
        self.instance_id = f"{socket.gethostname()}_{os.getpid()}_{uuid.uuid4().hex[:8]}"
        self.logger.info(f"🆔 Instance ID: {self.instance_id}")
        
        # Bittensor components
        self.wallet = None
        self.subtensor = None
        self.dendrite = None
        self.metagraph = None
        
        # State management
        self.validators: Dict[int, ValidatorState] = {}
        self.running = False
        self.start_time = time.time()
        
        # Initialize organic LoRA router
        if ORGANIC_LORA_ROUTER_AVAILABLE:
            self.lora_router = FinalOrganicRouter()
            self.logger.info("🧠 Initialized organic LoRA router with pattern learning (100% core accuracy)")
        else:
            self.lora_router = None
            self.logger.info("⚠️ Organic LoRA router not available - using default model")
        
        # Initialize prompt optimizer
        if OPTIMIZED_PROMPT_OPTIMIZER_AVAILABLE:
            # self.prompt_optimizer = OptimizedPromptOptimizer("rl_checkpoints_v3/prompt_score_log.csv")
            self.prompt_optimizer = LLMPromptOptimizer(
                ollama_url=self.config.get('ollama_url', 'http://localhost:11434'),
                model="llama3.2:3b",
                use_vllm=self.config.get('use_vllm', False),
                vllm_url=self.config.get('vllm_url', 'http://localhost:9000'),
                vllm_model=self.config.get('vllm_model', 'llama-3-2-3b-it')
            )
            self.logger.info("🚀 Initialized performance-optimized prompt optimizer")
        else:
            self.prompt_optimizer = TrellisPromptOptimizer()
            self.logger.info("�� Initialized standard prompt optimizer")
        
        # Initialize reproducibility system
        if REPRODUCIBILITY_SYSTEM_AVAILABLE:
            self.reproducibility_system = LLMClosePromptReproducibility(
                episodic_memory_file="episodic_logs_first/episodic_memory.json",
                use_vllm=self.config.get('use_vllm', False),
                vllm_url=self.config.get('vllm_url', 'http://localhost:9000'),
                vllm_model=self.config.get('vllm_model', 'llama-3-2-3b-it'),
                ollama_url=self.config.get('ollama_url', 'http://localhost:11434')
            )
            self.logger.info("🔄 Initialized reproducibility system for pre-optimization")
            
            # Track when we last reloaded gold prompts
            self.last_gold_prompts_reload = time.time()
            self.gold_prompts_reload_interval = self.config.get('gold_prompts_reload_interval', 3600)  # 1 hour default
            self.logger.info(f"   📚 Gold prompts will reload every {self.gold_prompts_reload_interval/3600:.1f} hours")
            self.logger.debug(f"   �� Raw reload interval value: {self.gold_prompts_reload_interval} seconds")
        else:
            self.reproducibility_system = None
            self.logger.info("⚠️ Reproducibility system not available")
        
        # Priority server coordinator
        self.priority_coordinator = PriorityServerCoordinator(
            server_url=self.config.get('generation_server_url', 'http://localhost:8097'),
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
            'optimization_improvements': 0,
            'prompts_cleaned': 0,  # Track how many prompts were cleaned of artifacts
            'priority_access_timeouts': 0,  # Track priority access timeouts
            'priority_interruptions': 0,    # Track when we interrupt other jobs
            'server_unavailable_skips': 0,  # Track when we skip task pulls due to server unavailability
            'server_status_check_errors': 0, # Track server status check errors
            'lora_routing_decisions': 0,    # Track LoRA routing decisions
            'lora_routing_accuracy': 0.0,   # Track LoRA routing accuracy
            'blacklisted_validators_skipped': 0, # Track blacklisted validator skips
            'gold_prompts_reloaded': 0, # Track how many times gold prompts were reloaded
            'gold_prompts_available': 0, # Track current number of gold prompts available
            
            # Enhanced cooldown system statistics
            'cooldown_violations_total': 0,  # Total cooldown violations across all validators
            'validation_locks_applied': 0,   # Total validation locks applied
            'enhanced_cooldown_penalties': 0, # Total enhanced cooldown penalties applied
            
            # Emergency cooldown management statistics
            'emergency_cooldowns_applied': 0,  # Total emergency cooldowns applied
            'critical_violations_handled': 0,  # Total critical violations handled
            'validators_temporarily_blacklisted': 0,  # Total validators temporarily blacklisted
            'validators_reset_from_emergency': 0,  # Total validators reset from emergency restrictions
            'dynamic_cooldown_scaling': 0,  # Total times dynamic cooldown scaling was applied
            'dynamic_buffer_applied': 0,  # Total times dynamic buffer was applied
            # State persistence statistics
            'validators_restored_from_disk': 0,  # Total validators restored from disk
            'violations_restored_from_disk': 0,  # Total validators with violations restored
            'blacklists_restored_from_disk': 0,  # Total blacklisted validators restored
            'validator_states_saved': 0,  # Total times states were saved to disk
            'validator_state_save_failures': 0,  # Total state save failures
            # New statistics for real-time learning
            'log_parsed_prompts': 0,  # Track prompts parsed from logs
            'enhanced_gold_prompts_available': 0,  # Track enhanced gold prompts (memory + logs)
            'enhanced_gold_prompts_reloaded': 0,  # Track enhanced reloads
            'total_gold_prompts_available': 0,  # Track total available gold prompts
            'memory_prompts': 0,  # Track prompts from episodic memory
            'log_prompts': 0,  # Track prompts from recent logs
        }
        
        # Dynamic system management attributes
        self.current_task_pull_strategy = "AGGRESSIVE"  # Default strategy
        self.current_max_concurrent_tasks = self.config.get('max_concurrent_tasks', 5)  # Default max tasks
        
        # State persistence system
        self.state_persistence = ValidatorStatePersistence(
            state_file=self.config.get('validator_state_file', 'validator_states.json')
        )
        
        # Register shutdown handlers for state persistence
        self._register_shutdown_handlers()
        
        self.logger.info("�� Continuous TRELLIS Orchestrator initialized")
        self.logger.info(f"   Output directory: {self.output_dir}")
        self.logger.info(f"   Generation server: {self.config['generation_server_url']}")
        self.logger.info(f"   Validation server: {self.config['validation_server_url']}")
        
        # Initialize gold prompts count and setup real-time learning if enabled
        if REPRODUCIBILITY_SYSTEM_AVAILABLE and self.reproducibility_system:
            self.stats['gold_prompts_available'] = len(self.reproducibility_system.gold_standard_results)
            self.logger.info(f"�� Initial gold prompts loaded: {self.stats['gold_prompts_available']}")
            
                    # Setup real-time learning if enabled
        if self.config.get('activate_learning', False):
            if self.config.get('only_log_learning', False):
                log_count = self.config.get('log_learning_count', 6)
                if log_count == -1:
                    log_info = "all available logs"
                else:
                    log_info = f"most recent {log_count} logs"
                
                self.logger.info(f"🚀 ONLY-LOG-LEARNING ENABLED - using {log_info}")
                self.logger.info("   📖 Will parse recent episode logs for fresh learning")
                self.logger.info("   📁 Episodic memory: BYPASSED")
                self.logger.info(f"   🔄 Will use {log_info} exclusively for optimization")
            else:
                self.logger.info("🚀 Real-time learning ENABLED - setting up enhanced gold prompts system")
                self.logger.info("   📖 Will parse recent episode logs for fresh learning")
                self.logger.info("   📁 Will monitor episodic memory for live updates")
                self.logger.info("   🔄 Will combine memory + log data for comprehensive coverage")
            
            # Setup live monitoring (only if not in only-log-learning mode)
            if not self.config.get('only_log_learning', False):
                self.setup_live_episodic_memory_monitoring()
            
            # Initial enhanced reload (after stats are initialized)
            self.enhanced_reload_gold_prompts()
        else:
            self.logger.info("📚 Real-time learning DISABLED - using standard episodic memory only")
        
        # Log task tracking status
        if self.config.get('enable_task_tracking', True) and not self.config.get('disable_task_tracking', False):
            self.logger.info(f"🔄 Shared Task Tracking: ENABLED (Instance ID: {self.instance_id})")
        else:
            self.logger.info(f"🔄 Shared Task Tracking: DISABLED")
        
        # Log duplicate checking status
        if self.config.get('enable_duplicate_checking', True):
            self.logger.info(f"🔄 Duplicate Checking: ENABLED (will skip previously processed prompts)")
        else:
            self.logger.info(f"🔄 Duplicate Checking: DISABLED (will process all prompts including duplicates)")
        
        # Print LLM provider information prominently
        print("\n" + "="*60)
        print("🤖 CONTINUOUS TRELLIS ORCHESTRATOR - LLM PROVIDER CONFIGURATION")
        print("="*60)
        if self.config.get('use_vllm', False):
            print(f"✅ Using vLLM: {self.config.get('vllm_url', 'http://localhost:9000')}")
            print(f"   Model: {self.config.get('vllm_model', 'llama-3-2-3b-it')}")
            print(f"   Status: ACTIVE for prompt optimization")
        else:
            print(f"✅ Using Ollama: {self.config.get('ollama_url', 'http://localhost:11434')}")
            print(f"   Status: ACTIVE for prompt optimization")
        print("="*60)
        
        if self.config.get('use_vllm', False):
            self.logger.info(f"   Using vLLM: {self.config.get('vllm_url', 'http://localhost:9000')} with model {self.config.get('vllm_model', 'llama-3-2-3b-it')}")
        else:
            self.logger.info(f"   Using Ollama: {self.config.get('ollama_url', 'http://localhost:11434')}")
        
        # Log LoRA routing settings
        if ORGANIC_LORA_ROUTER_AVAILABLE:
            self.logger.info(f"🧠 Organic LoRA routing: ENABLED (100% pattern learning accuracy)")
        else:
            self.logger.info(f"🧠 Organic LoRA routing: DISABLED (using default model)")
        
        # Log optimization settings
        if self.config.get('enable_prompt_optimization', True):
            mode = "aggressive" if self.config.get('optimization_aggressive_mode', False) else "standard"
            detail = "minimal" if not self.config.get('log_optimization_details', True) else "detailed"
            cleaning = "ENABLED" if self.config.get('enable_prompt_cleaning', True) else "DISABLED"
            self.logger.info(f"🔧 Prompt optimization: ENABLED ({mode} mode, {detail} logging, cleaning: {cleaning})")
            
            # Log LLM provider for optimization
            if self.config.get('use_vllm', False):
                self.logger.info(f"   🤖 LLM Provider: vLLM ({self.config.get('vllm_model', 'llama-3-2-3b-it')})")
            else:
                self.logger.info(f"   🤖 LLM Provider: Ollama ({self.config.get('ollama_url', 'http://localhost:11434')})")
            
            # Log reproducibility settings
            if self.config.get('enable_reproducibility_optimization', True):
                min_sim = self.config.get('reproducibility_min_similarity', 0.3)
                self.logger.info(f"🔄 Reproducibility optimization: ENABLED (min similarity: {min_sim})")
            else:
                self.logger.info(f"🔄 Reproducibility optimization: DISABLED")
        else:
            self.logger.info(f"🔧 Prompt optimization: DISABLED")
        self.trellis_server_url: str = "http://localhost:8097"


    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration"""
        return {
            # Bittensor settings
            'wallet_name': 'test2m3b2',
            'hotkey_name': 't2m3b2',
            'netuid': 17,
            'min_validator_stake': 1000.0,  # Minimum stake required for a validator to be considered
            'min_validator_trust': 0.0,     # Minimum trust score
            'max_validators': 50,           # Maximum number of validators to track
            
            # Validator blacklisting
            'validator_blacklist': [180],   # UIDs to blacklist (e.g., 180 is a WC)
            'enable_validator_blacklisting': True,
            
            # Server settings
            'generation_server_url': 'http://localhost:8097',
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
            
            # Determinism settings
            'use_fixed_seed': True,  # True = always seed 42, False = prompt-hash based seed
            
            # Prompt optimization settings
            'enable_prompt_optimization': True,
            'optimization_aggressive_mode': False,
            'log_optimization_details': True,
            'enable_prompt_cleaning': True,  # Enable automatic prompt cleaning to remove artifacts
            
            # Reproducibility optimization settings
            'enable_reproducibility_optimization': True,
            'reproducibility_min_similarity': 0.3,

            # LoRA routing settings
            'enable_lora_routing': True,  # Enable intelligent LoRA routing
            'lora_routing_confidence_threshold': 0.5,  # Minimum confidence for LoRA routing

            # Priority access settings
            'priority_access_max_wait': 60, # Max seconds to wait for priority access
            'priority_access_check_interval': 1, # Seconds between status checks
            'priority_access_timeout': 30, # Max seconds to wait for priority access
            
            # Gold prompts reload settings
            'gold_prompts_reload_interval': 3600,  # Reload gold prompts every hour (3600 seconds)
            
            # Duplicate checking settings
            'enable_duplicate_checking': True,  # Enable duplicate prompt checking
            
            # Real-time learning settings
            'activate_learning': False,  # Enable real-time learning from logs and live monitoring
            'log_learning_count': 6,     # Number of recent logs to use for learning (default: 6, -1 for all)
            
            # vLLM settings
            'use_vllm': False,  # Use vLLM instead of Ollama
            'vllm_url': 'http://localhost:9000',  # vLLM server URL
            'vllm_model': 'llama-3-2-3b-it',  # vLLM model name
            
            # Shared task tracking settings
            'enable_task_tracking': True,  # Enable shared task tracking to prevent duplicate processing
            'task_tracking_timeout_minutes': 2,  # Timeout for task locks (minutes)
            
            # Cooldown settings
            'network_error_cooldown': 30,  # Seconds to wait after network errors
            'submission_failure_cooldown': 60,  # Seconds to wait after submission failures
            'validator_error_cooldown': 45,  # Seconds to wait after validator errors
            'max_cooldown_duration': 300,  # Maximum cooldown duration (5 minutes)
            'enable_cooldown_logging': True,  # Enable detailed cooldown logging
            
            # Enhanced cooldown system settings
            'cooldown_violation_threshold': 5,  # Number of violations before applying penalty
            'cooldown_violation_penalty': 60,  # Additional penalty cooldown in seconds
            'validation_lock_duration': 30,  # Default validation lock duration in seconds
            
            # Emergency cooldown management settings
            'emergency_cooldown_buffer': 30,  # Buffer seconds added to validator cooldowns
            'critical_violation_threshold': 100,  # Violation count that triggers emergency measures
            'critical_violation_cooldown': 3600,  # Emergency cooldown duration for critical violations
            'base_blacklist_duration': 1800,  # Base duration for temporary blacklisting
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
            
            # self.subtensor = bt.subtensor(network="test") #TODO
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
            blacklisted_validators = len([v for v in self.validators.values() if v.is_active and self.is_validator_blacklisted(v.uid)])
            
            self.logger.info(f"✅ Validator refresh complete:")
            self.logger.info(f"   Active validators: {active_validators}")
            self.logger.info(f"   Blacklisted validators: {blacklisted_validators}")
            self.logger.info(f"   Inactive validators: {inactive_count}")
            self.logger.info(f"   Total eligible validators found: {len(eligible_validators)}")
            
            # Log blacklisted validators if any
            if blacklisted_validators > 0:
                blacklisted_uids = [v.uid for v in self.validators.values() if v.is_active and self.is_validator_blacklisted(v.uid)]
                self.logger.info(f"   🚫 Blacklisted UIDs: {blacklisted_uids}")
            else:
                self.logger.info(f"   ✅ No blacklisted validators found")
            
            # Log blacklisting configuration
            blacklist_config = self.config.get('validator_blacklist', [])
            blacklist_enabled = self.config.get('enable_validator_blacklisting', True)
            self.logger.info(f"   🔧 Blacklisting config: {'ENABLED' if blacklist_enabled else 'DISABLED'}")
            self.logger.info(f"   📋 Blacklist UIDs: {blacklist_config}")
            
            # Check each active validator for blacklisting status
            self.logger.info(f"   🔍 Checking blacklist status for each active validator:")
            for validator in sorted([v for v in self.validators.values() if v.is_active], key=lambda x: x.stake, reverse=True):
                blacklist_status = "🚫 BLACKLISTED" if self.is_validator_blacklisted(validator.uid) else "✅ ALLOWED"
                self.logger.info(f"     UID {validator.uid}: {blacklist_status} (stake: {validator.stake:.1f} TAO)")
            
            # Log top validators by stake
            top_validators = sorted(
                [v for v in self.validators.values() if v.is_active], 
                key=lambda x: x.stake, 
                reverse=True
            )[:5]
            
            self.logger.info("   Top validators by stake:")
            for validator in top_validators:
                self.logger.info(f"     UID {validator.uid}: {validator.stake:.1f} TAO (trust: {validator.trust:.3f})")
            
            # CRITICAL: Restore validator states from disk after discovery
            self.restore_validator_states_from_disk()
            
        except Exception as e:
            self.logger.error(f"❌ Validator refresh failed: {e}")
            traceback.print_exc()
    
    def is_validator_blacklisted(self, validator_uid: int) -> bool:
        """Check if a validator is blacklisted"""
        if not self.config.get('enable_validator_blacklisting', True):
            self.logger.info(f"🔓 Blacklisting DISABLED - UID {validator_uid} allowed")
            return False
        
        blacklist = self.config.get('validator_blacklist', [])
        is_blacklisted = validator_uid in blacklist
        
        if is_blacklisted:
            # self.logger.info(f"🚫 Validator UID {validator_uid} is BLACKLISTED - skipping")
            self.stats['blacklisted_validators_skipped'] += 1
        else:
            self.logger.debug(f"✅ Validator UID {validator_uid} is NOT blacklisted - allowing")
        
        return is_blacklisted
    
    def is_validator_available(self, validator: ValidatorState) -> bool:
        """Check if validator is available for task pulling"""
        current_time = time.time()
        
        # Check if validator is active
        if not validator.is_active:
            self.logger.debug(f"🔴 Validator UID {validator.uid} not available: INACTIVE")
            return False
        
        # Check if validator is blacklisted
        if self.is_validator_blacklisted(validator.uid):
            # self.logger.info(f"🚫 Validator UID {validator.uid} not available: BLACKLISTED")
            return False
        
        # Check emergency blacklist (new critical feature)
        if validator.emergency_blacklist_until and current_time < validator.emergency_blacklist_until:
            remaining = validator.emergency_blacklist_until - current_time
            self.logger.warning(f"🚨 Validator UID {validator.uid} not available: EMERGENCY BLACKLIST ({remaining:.1f}s remaining)")
            return False
        
        # Enhanced cooldown checking (maintains existing protocol compliance)
        if validator.cooldown_until and current_time < validator.cooldown_until:
            cooldown_remaining = validator.cooldown_until - current_time
            cooldown_status = self.get_cooldown_status(validator)
            self.logger.debug(f"⏳ Validator UID {validator.uid} not available: COOLDOWN ({cooldown_status})")
            return False
        
        # Check validation lock (new enhanced feature)
        if validator.validation_locked_until and current_time < validator.validation_locked_until:
            remaining = validator.validation_locked_until - current_time
            self.logger.debug(f"🔒 Validator UID {validator.uid} validation locked for {remaining:.1f}s")
            return False
        
        # Check if we pulled recently (respect pull interval)
        if validator.last_task_pull:
            time_since_pull = current_time - validator.last_task_pull
            if time_since_pull < self.config['task_pull_interval']:
                time_until_available = self.config['task_pull_interval'] - time_since_pull
                self.logger.debug(f"⏰ Validator UID {validator.uid} not available: PULL INTERVAL ({time_until_available:.1f}s until available)")
                return False
        
        self.logger.debug(f"✅ Validator UID {validator.uid} is AVAILABLE for task pulling")
        return True
    
    def set_validator_cooldown(self, validator: ValidatorState, cooldown_seconds: int, reason: str):
        """
        Set a cooldown period for a validator with proper logging and duration limits.
        
        Args:
            validator: The validator to set cooldown for
            cooldown_seconds: Cooldown duration in seconds
            reason: Reason for the cooldown (for logging)
        """
        # Limit cooldown duration to prevent excessive waiting
        max_cooldown = self.config.get('max_cooldown_duration', 300)
        cooldown_seconds = min(cooldown_seconds, max_cooldown)
        
        # Set cooldown
        validator.cooldown_until = time.time() + cooldown_seconds
        
        # Log cooldown with human-readable duration
        if self.config.get('enable_cooldown_logging', True):
            if cooldown_seconds < 60:
                duration_str = f"{cooldown_seconds}s"
            elif cooldown_seconds < 3600:
                duration_str = f"{cooldown_seconds//60}m {cooldown_seconds%60}s"
            else:
                hours = cooldown_seconds // 3600
                minutes = (cooldown_seconds % 3600) // 60
                duration_str = f"{hours}h {minutes}m"
            
            self.logger.info(f"⏳ Cooldown set for UID {validator.uid}: {duration_str} ({reason})")
            self.logger.info(f"   Next available: {time.strftime('%H:%M:%S', time.localtime(validator.cooldown_until))}")
        else:
            self.logger.debug(f"⏳ Cooldown set for UID {validator.uid}: {cooldown_seconds}s ({reason})")
    
    def set_validator_validation_lock(self, validator: ValidatorState, lock_duration_seconds: int, reason: str):
        """
        Set a validation lock period for a validator.
        
        Args:
            validator: The validator to set validation lock for
            lock_duration_seconds: Lock duration in seconds
            reason: Reason for the validation lock (for logging)
        """
        # Set validation lock
        validator.validation_locked_until = time.time() + lock_duration_seconds
        self.stats['validation_locks_applied'] += 1
        
        # Log validation lock with human-readable duration
        if self.config.get('enable_cooldown_logging', True):
            if lock_duration_seconds < 60:
                duration_str = f"{lock_duration_seconds}s"
            elif lock_duration_seconds < 3600:
                duration_str = f"{lock_duration_seconds//60}m {lock_duration_seconds%60}s"
            else:
                hours = lock_duration_seconds // 3600
                minutes = (lock_duration_seconds % 3600) // 60
                duration_str = f"{hours}h {minutes}m"
            
            self.logger.info(f"🔒 Validation lock set for UID {validator.uid}: {duration_str} ({reason})")
            self.logger.info(f"   Next available: {time.strftime('%H:%M:%S', time.localtime(validator.validation_locked_until))}")
        else:
            self.logger.debug(f"🔒 Validation lock set for UID {validator.uid}: {lock_duration_seconds}s ({reason})")
    
    def increment_cooldown_violations(self, validator: ValidatorState, reason: str):
        """
        Increment cooldown violations counter for a validator.
        
        Args:
            validator: The validator to increment violations for
            reason: Reason for the violation (for logging)
        """
        validator.cooldown_violations += 1
        self.stats['cooldown_violations_total'] += 1
        self.logger.warning(f"⚠️ Cooldown violation #{validator.cooldown_violations} for UID {validator.uid}: {reason}")
        
        # Check if we should apply additional penalties
        violation_threshold = self.config.get('cooldown_violation_threshold', 5)
        if validator.cooldown_violations >= violation_threshold:
            penalty_seconds = self.config.get('cooldown_violation_penalty', 60)
            self.logger.warning(f"🚨 Cooldown violation threshold reached for UID {validator.uid} - applying {penalty_seconds}s penalty")
            self.set_validator_cooldown(validator, penalty_seconds, f"Violation penalty (violation #{validator.cooldown_violations})")
            self.stats['enhanced_cooldown_penalties'] += 1
    
    def _set_emergency_cooldown(self, validator: ValidatorState, cooldown_until: int, reason: str):
        """
        Set DYNAMIC emergency cooldown to prevent further violations.
        Automatically adjusts buffer based on validator history.
        
        Args:
            validator: The validator to set emergency cooldown for
            cooldown_until: Timestamp when cooldown expires
            reason: Reason for emergency cooldown
        """
        current_time = time.time()
        if cooldown_until > current_time:
            # DYNAMIC: Calculate buffer time based on validator history
            base_buffer = self.config.get('emergency_cooldown_buffer', 30)
            
            if hasattr(validator, 'violation_history') and validator.violation_history:
                # Analyze recent violations to determine buffer size
                recent_violations = [v['violations'] for v in validator.violation_history[-3:]]
                avg_recent_violations = sum(recent_violations) / len(recent_violations) if recent_violations else 0
                
                if avg_recent_violations > 1000:  # Extreme violations
                    buffer_multiplier = 3.0  # 3x buffer for extreme cases
                    self.logger.warning(f"   EXTREME violation history - applying 3x buffer multiplier")
                elif avg_recent_violations > 500:  # High violations
                    buffer_multiplier = 2.0  # 2x buffer for high cases
                    self.logger.warning(f"   HIGH violation history - applying 2x buffer multiplier")
                elif avg_recent_violations > 200:  # Moderate violations
                    buffer_multiplier = 1.5  # 1.5x buffer for moderate cases
                    self.logger.warning(f"   MODERATE violation history - applying 1.5x buffer multiplier")
                else:
                    buffer_multiplier = 1.0  # 1x buffer for standard cases
                    self.logger.info(f"   STANDARD violation history - applying 1x buffer multiplier")
                
                dynamic_buffer = int(base_buffer * buffer_multiplier)
            else:
                # No history - use base buffer
                dynamic_buffer = base_buffer
                buffer_multiplier = 1.0
            
            emergency_cooldown_until = cooldown_until + dynamic_buffer
            
            # CRITICAL FIX: Prevent infinite cooldown escalation
            if (validator.cooldown_until and 
                validator.cooldown_until > emergency_cooldown_until):
                self.logger.warning(f"⚠️ Emergency cooldown already set for UID {validator.uid} - not escalating")
                return
            
            validator.cooldown_until = emergency_cooldown_until
            self.logger.warning(f"🚨 DYNAMIC emergency cooldown set for UID {validator.uid}: {reason}")
            self.logger.warning(f"   Original cooldown: {cooldown_until}, Emergency cooldown: {emergency_cooldown_until}")
            self.logger.warning(f"   DYNAMIC buffer: {dynamic_buffer}s (base: {base_buffer}s, multiplier: {buffer_multiplier:.1f}x)")
            
            # Track emergency cooldowns with dynamic info
            self.stats['emergency_cooldowns_applied'] = self.stats.get('emergency_cooldowns_applied', 0) + 1
            self.stats['dynamic_buffer_applied'] = self.stats.get('dynamic_buffer_applied', 0) + 1
            
            # Store buffer history for learning
            if not hasattr(validator, 'buffer_history'):
                validator.buffer_history = []
            validator.buffer_history.append({
                'timestamp': time.time(),
                'base_buffer': base_buffer,
                'dynamic_buffer': dynamic_buffer,
                'multiplier': buffer_multiplier,
                'reason': reason
            })
            
            # Keep only last 5 buffer adjustments for memory management
            if len(validator.buffer_history) > 5:
                validator.buffer_history = validator.buffer_history[-5:]
        else:
            self.logger.debug(f"ℹ️ Emergency cooldown not needed for UID {validator.uid} - cooldown already expired")
    
    def _handle_critical_violations(self, validator: ValidatorState, violation_count: int):
        """
        Handle critical violation situations with DYNAMIC emergency measures.
        Automatically adjusts cooldown duration based on violation severity.
        
        Args:
            validator: The validator with critical violations
            violation_count: Current violation count
        """
        # CRITICAL FIX: Prevent multiple emergency measures
        if validator.emergency_blacklist_until and time.time() < validator.emergency_blacklist_until:
            self.logger.warning(f"⚠️ Emergency measures already active for UID {validator.uid} - skipping duplicate")
            return
        
        self.logger.error(f"🚨 CRITICAL: Implementing DYNAMIC emergency measures for UID {validator.uid}")
        self.logger.error(f"   Violation count: {violation_count} (threshold: 100)")
        
        # DYNAMIC: Calculate emergency duration based on violation severity
        base_duration = self.config.get('critical_violation_cooldown', 3600)  # 1 hour base
        
        # Scale duration based on violation count (exponential backoff)
        if violation_count > 1000:
            scale_factor = 4.0  # 4x for extreme violations
            self.logger.error(f"   EXTREME violations detected - applying 4x multiplier")
        elif violation_count > 500:
            scale_factor = 2.5  # 2.5x for high violations
            self.logger.error(f"   HIGH violations detected - applying 2.5x multiplier")
        elif violation_count > 200:
            scale_factor = 1.5  # 1.5x for moderate violations
            self.logger.error(f"   MODERATE violations detected - applying 1.5x multiplier")
        else:
            scale_factor = 1.0  # 1x for standard violations
            self.logger.error(f"   STANDARD violations detected - applying 1x multiplier")
        
        emergency_duration = int(base_duration * scale_factor)
        emergency_cooldown_until = time.time() + emergency_duration
        
        validator.cooldown_until = emergency_cooldown_until
        self.logger.error(f"   DYNAMIC emergency cooldown: {emergency_duration}s (base: {base_duration}s, scale: {scale_factor:.1f}x)")
        self.logger.error(f"   Cooldown until: {emergency_cooldown_until}")
        
        # Mark validator as temporarily blacklisted
        validator.is_active = False
        validator.emergency_blacklist_until = emergency_cooldown_until
        
        self.logger.error(f"   Validator UID {validator.uid} temporarily blacklisted until cooldown expires")
        
        # Track critical violations with dynamic scaling info
        self.stats['critical_violations_handled'] = self.stats.get('critical_violations_handled', 0) + 1
        self.stats['dynamic_cooldown_scaling'] = self.stats.get('dynamic_cooldown_scaling', 0) + 1
        
        # CRITICAL: Save state immediately after handling critical violations
        self.save_validator_states_to_disk()
        
        # Store violation history for adaptive learning
        if not hasattr(validator, 'violation_history'):
            validator.violation_history = []
        validator.violation_history.append({
            'timestamp': time.time(),
            'violations': violation_count,
            'cooldown_duration': emergency_duration,
            'scale_factor': scale_factor
        })
        
        # Keep only last 10 violations for memory management
        if len(validator.violation_history) > 10:
            validator.violation_history = validator.violation_history[-10:]
    
    def _blacklist_validator_temporarily(self, validator: ValidatorState, violation_count: int):
        """
        Temporarily blacklist a validator due to excessive violations.
        
        Args:
            validator: The validator to blacklist
            violation_count: Current violation count
        """
        # CRITICAL FIX: Prevent multiple blacklistings
        if validator.emergency_blacklist_until and time.time() < validator.emergency_blacklist_until:
            self.logger.warning(f"⚠️ Validator UID {validator.uid} already blacklisted - skipping duplicate")
            return
        
        self.logger.error(f"🚨 BLACKLISTING: Validator UID {validator.uid} due to {violation_count} violations")
        
        # Calculate blacklist duration based on violation count
        base_duration = self.config.get('base_blacklist_duration', 1800)  # 30 minutes
        violation_multiplier = min(violation_count / 100, 10)  # Cap at 10x
        blacklist_duration = int(base_duration * violation_multiplier)
        
        blacklist_until = time.time() + blacklist_duration
        
        # Set blacklist
        validator.is_active = False
        validator.emergency_blacklist_until = blacklist_until
        validator.cooldown_until = blacklist_until
        
        self.logger.error(f"   Blacklist duration: {blacklist_duration}s (until {blacklist_until})")
        self.logger.error(f"   Violation multiplier: {violation_multiplier:.1f}x")
        
        # Track blacklists
        self.stats['validators_temporarily_blacklisted'] = self.stats.get('validators_temporarily_blacklisted', 0) + 1
        
        # CRITICAL: Save state immediately after blacklisting
        self.save_validator_states_to_disk()
    
    def _check_and_clear_expired_emergency_blacklists(self):
        """
        Check and clear expired emergency blacklists and cooldowns.
        This should be called periodically to restore validators.
        """
        current_time = time.time()
        cleared_count = 0
        
        for validator in self.validators.values():
            # Check emergency blacklist
            if (validator.emergency_blacklist_until and 
                current_time >= validator.emergency_blacklist_until):
                
                self.logger.info(f"✅ Emergency blacklist expired for UID {validator.uid}")
                self._safe_reset_validator(validator, "emergency blacklist")
                cleared_count += 1
            
            # Check if cooldown has expired but emergency blacklist is still active
            if (validator.cooldown_until and 
                current_time >= validator.cooldown_until and
                validator.emergency_blacklist_until and
                current_time >= validator.emergency_blacklist_until):
                
                self.logger.info(f"✅ Cooldown and emergency blacklist expired for UID {validator.uid}")
                self._safe_reset_validator(validator, "cooldown and emergency blacklist")
                cleared_count += 1
        
        if cleared_count > 0:
            self.logger.info(f"🔄 Restored {cleared_count} validators from expired emergency restrictions")
        
        return cleared_count
    
    def _safe_reset_validator(self, validator: ValidatorState, reason: str):
        """
        Safely reset a validator's emergency restrictions.
        
        Args:
            validator: The validator to reset
            reason: Reason for the reset
        """
        self.logger.info(f"🔄 Safely resetting UID {validator.uid}: {reason}")
        
        # Clear emergency restrictions
        validator.emergency_blacklist_until = None
        validator.cooldown_until = None
        validator.is_active = True
        
        # DYNAMIC: Reset violation counter based on validator history and behavior
        if hasattr(validator, 'violation_history') and validator.violation_history:
            # Analyze recent violation patterns
            recent_violations = [v['violations'] for v in validator.violation_history[-3:]]  # Last 3 violations
            avg_recent_violations = sum(recent_violations) / len(recent_violations) if recent_violations else 0
            
            if avg_recent_violations > 1000:  # Extreme violations
                reduction_factor = 0.1  # Reduce to 10% (very aggressive)
                self.logger.warning(f"   EXTREME violation history - aggressive reduction to 10%")
            elif avg_recent_violations > 500:  # High violations
                reduction_factor = 0.2  # Reduce to 20%
                self.logger.warning(f"   HIGH violation history - aggressive reduction to 20%")
            elif avg_recent_violations > 200:  # Moderate violations
                reduction_factor = 0.3  # Reduce to 30%
                self.logger.warning(f"   MODERATE violation history - moderate reduction to 30%")
            else:
                reduction_factor = 0.5  # Reduce to 50% (standard)
                self.logger.info(f"   STANDARD violation history - standard reduction to 50%")
            
            new_violation_count = max(1, int(validator.cooldown_violations * reduction_factor))
            old_count = validator.cooldown_violations
            validator.cooldown_violations = new_violation_count
            
            self.logger.info(f"   DYNAMIC violation reduction: {old_count} → {new_violation_count} (factor: {reduction_factor:.1f})")
        else:
            # No history - use standard reduction
            if validator.cooldown_violations > 10:
                validator.cooldown_violations = max(5, validator.cooldown_violations // 2)
                self.logger.info(f"   Standard violation reduction: {validator.cooldown_violations * 2} → {validator.cooldown_violations}")
        
        # Log the reset
        self.logger.info(f"   UID {validator.uid} is now available for task pulling")
        
        # Track resets
        self.stats['validators_reset_from_emergency'] = self.stats.get('validators_reset_from_emergency', 0) + 1
        
        # CRITICAL: Save state after validator reset
        self.save_validator_states_to_disk()
        
        # DYNAMIC: Check if validator needs extended monitoring based on history
        if hasattr(validator, 'violation_history') and validator.violation_history:
            recent_trend = self._analyze_violation_trend(validator)
            if recent_trend == 'increasing':
                self.logger.warning(f"⚠️ UID {validator.uid} shows INCREASING violation trend - extended monitoring")
                # Set a shorter cooldown for problematic validators
                extended_monitoring_cooldown = time.time() + 300  # 5 minutes
                validator.cooldown_until = extended_monitoring_cooldown
                self.logger.warning(f"   Extended monitoring cooldown set: 5 minutes")
            elif recent_trend == 'stable_high':
                self.logger.warning(f"⚠️ UID {validator.uid} shows STABLE HIGH violations - close monitoring")
            else:
                self.logger.info(f"✅ UID {validator.uid} shows IMPROVING trend - standard monitoring")
        else:
            # No history - standard check
            if validator.cooldown_violations > 50:
                self.logger.warning(f"⚠️ UID {validator.uid} still has high violations ({validator.cooldown_violations}) after reset")
                self.logger.warning(f"   Monitoring closely - may need extended cooldown if violations persist")
    
    def get_cooldown_status(self, validator: ValidatorState) -> str:
        """
        Get human-readable cooldown status for a validator.
        
        Args:
            validator: The validator to check
            
        Returns:
            Human-readable cooldown status string
        """
        status_parts = []
        
        # Check cooldown status
        if validator.cooldown_until:
            current_time = time.time()
            if current_time >= validator.cooldown_until:
                status_parts.append("Cooldown expired")
            else:
                remaining = int(validator.cooldown_until - current_time)
            if remaining < 60:
                        status_parts.append(f"Cooldown: {remaining}s remaining")
            elif remaining < 3600:
                        status_parts.append(f"Cooldown: {remaining//60}m {remaining%60}s remaining")
            else:
                hours = remaining // 3600
                minutes = (remaining % 3600) // 60
                status_parts.append(f"Cooldown: {hours}h {minutes}m remaining")
        else:
            status_parts.append("No cooldown")
        
        # Check validation lock status
        if validator.validation_locked_until:
            current_time = time.time()
            if current_time >= validator.validation_locked_until:
                status_parts.append("Validation lock expired")
            else:
                remaining = int(validator.validation_locked_until - current_time)
                if remaining < 60:
                    status_parts.append(f"Validation locked: {remaining}s remaining")
                elif remaining < 3600:
                    status_parts.append(f"Validation locked: {remaining//60}m {remaining%60}s remaining")
                else:
                    hours = remaining // 3600
                    minutes = (remaining % 3600) // 60
                    status_parts.append(f"Validation locked: {hours}h {minutes}m remaining")
        
        # Add violation count if any
        if validator.cooldown_violations > 0:
            status_parts.append(f"Violations: {validator.cooldown_violations}")
        
        # Check emergency blacklist status (CRITICAL FIX)
        if validator.emergency_blacklist_until:
            current_time = time.time()
            if current_time >= validator.emergency_blacklist_until:
                status_parts.append("Emergency blacklist expired")
            else:
                remaining = int(validator.emergency_blacklist_until - current_time)
                if remaining < 60:
                    status_parts.append(f"EMERGENCY BLACKLIST: {remaining}s remaining")
                elif remaining < 3600:
                    status_parts.append(f"EMERGENCY BLACKLIST: {remaining//60}m {remaining%60}s remaining")
                else:
                    hours = remaining // 3600
                    minutes = (remaining % 3600) // 60
                    status_parts.append(f"EMERGENCY BLACKLIST: {hours}h {minutes}m remaining")
        
        return " | ".join(status_parts) if status_parts else "Available"
    
    def _check_validators_needing_monitoring(self):
        """
        Check for validators that need extended monitoring due to persistent issues.
        DYNAMICALLY adjusts monitoring thresholds based on system health.
        """
        current_time = time.time()
        monitoring_count = 0
        
        # DYNAMIC: Calculate monitoring thresholds based on overall system health
        total_validators = len(self.validators)
        active_validators_count = len([v for v in self.validators.values() if v.is_active])
        system_health_ratio = active_validators_count / total_validators if total_validators > 0 else 0
        
        # Adjust thresholds based on system health
        if system_health_ratio < 0.3:  # Less than 30% validators active
            violation_threshold = 50  # Lower threshold for critical system state
            monitoring_threshold = 25
            self.logger.warning(f"🚨 CRITICAL SYSTEM STATE: Only {system_health_ratio:.1%} validators active")
            self.logger.warning(f"   Lowering monitoring thresholds: violations > {violation_threshold}, monitoring > {monitoring_threshold}")
        elif system_health_ratio < 0.6:  # Less than 60% validators active
            violation_threshold = 75  # Medium threshold for degraded system state
            monitoring_threshold = 40
            self.logger.warning(f"⚠️ DEGRADED SYSTEM STATE: Only {system_health_ratio:.1%} validators active")
            self.logger.warning(f"   Adjusting monitoring thresholds: violations > {violation_threshold}, monitoring > {monitoring_threshold}")
        else:  # Healthy system state
            violation_threshold = 100  # Standard threshold for healthy system
            monitoring_threshold = 50
            self.logger.info(f"✅ HEALTHY SYSTEM STATE: {system_health_ratio:.1%} validators active")
            self.logger.info(f"   Using standard monitoring thresholds: violations > {violation_threshold}, monitoring > {monitoring_threshold}")
        
        for validator in self.validators.values():
            # Check for validators with persistently high violations (DYNAMIC threshold)
            if (validator.cooldown_violations > violation_threshold and 
                not validator.emergency_blacklist_until):
                
                self.logger.warning(f"⚠️ UID {validator.uid} needs monitoring: {validator.cooldown_violations} violations")
                self.logger.warning(f"   DYNAMIC threshold: {violation_threshold} (system health: {system_health_ratio:.1%})")
                
                # DYNAMIC: Apply immediate cooldown for critical validators in degraded system
                if system_health_ratio < 0.6 and validator.cooldown_violations > violation_threshold * 1.5:
                    immediate_cooldown = time.time() + 600  # 10 minutes
                    validator.cooldown_until = immediate_cooldown
                    self.logger.error(f"�� IMMEDIATE cooldown applied to UID {validator.uid}: 10 minutes")
                    self.logger.error(f"   Critical validator in degraded system - protecting remaining validators")
                
                monitoring_count += 1
            
            # Check for validators that were recently reset but still have issues (DYNAMIC threshold)
            if (validator.cooldown_violations > monitoring_threshold and 
                validator.last_violation_check and
                current_time - validator.last_violation_check > 300):  # 5 minutes
                
                self.logger.warning(f"⚠️ UID {validator.uid} showing persistent issues after reset")
                self.logger.warning(f"   Violations: {validator.cooldown_violations}, DYNAMIC threshold: {monitoring_threshold}")
                validator.last_violation_check = current_time
                monitoring_count += 1
        
        if monitoring_count > 0:
            self.logger.warning(f"⚠️ {monitoring_count} validators need extended monitoring (DYNAMIC thresholds)")
            self.logger.warning(f"   System health: {system_health_ratio:.1%}, Active: {active_validators_count}/{total_validators}")
        
        return monitoring_count
    
    def _analyze_violation_trend(self, validator: ValidatorState) -> str:
        """
        Analyze the trend of violations for a validator.
        
        Args:
            validator: The validator to analyze
            
        Returns:
            Trend analysis: 'increasing', 'decreasing', 'stable_high', 'stable_low', 'unknown'
        """
        if not hasattr(validator, 'violation_history') or len(validator.violation_history) < 3:
            return 'unknown'
        
        # Get last 3 violations
        recent_violations = [v['violations'] for v in validator.violation_history[-3:]]
        
        # Calculate trend
        if len(recent_violations) >= 3:
            # Simple trend analysis
            if recent_violations[-1] > recent_violations[-2] > recent_violations[-3]:
                return 'increasing'
            elif recent_violations[-1] < recent_violations[-2] < recent_violations[-3]:
                return 'decreasing'
            elif all(v > 500 for v in recent_violations):
                return 'stable_high'
            elif all(v < 100 for v in recent_violations):
                return 'stable_low'
            else:
                return 'stable_high'  # Default to stable_high for mixed patterns
        
        return 'unknown'
    
    def restore_validator_states_from_disk(self):
        """
        Restore validator states from disk if available.
        This method should be called after validators are discovered but before mining starts.
        """
        self.logger.info("🔄 Attempting to restore validator states from disk...")
        
        try:
            saved_states = self.state_persistence.load_validator_states()
            
            if not saved_states:
                self.logger.info("📁 No saved states found - starting with fresh validator states")
                return
            
            restored_count = 0
            violation_count = 0
            blacklisted_count = 0
            cooldown_count = 0
            
            for uid, saved_state in saved_states.items():
                if uid in self.validators:
                    validator = self.validators[uid]
                    
                    # Restore critical state information
                    validator.cooldown_until = saved_state.get('cooldown_until')
                    validator.cooldown_violations = saved_state.get('cooldown_violations', 0)
                    validator.throttle_period = saved_state.get('throttle_period', 0)
                    validator.validation_locked_until = saved_state.get('validation_locked_until')
                    validator.emergency_blacklist_until = saved_state.get('emergency_blacklist_until')
                    validator.last_submit_time = saved_state.get('last_submit_time')
                    validator.last_violation_check = saved_state.get('last_violation_check')
                    
                    # Restore performance tracking
                    validator.total_tasks_received = saved_state.get('total_tasks_received', 0)
                    validator.total_tasks_submitted = saved_state.get('total_tasks_submitted', 0)
                    validator.total_successful_submissions = saved_state.get('total_successful_submissions', 0)
                    validator.average_score = saved_state.get('average_score', 0.0)
                    
                    # Restore activity state
                    validator.is_active = saved_state.get('is_active', True)
                    
                    # Restore learning history
                    if 'violation_history' in saved_state:
                        validator.violation_history = saved_state['violation_history']
                    if 'buffer_history' in saved_state:
                        validator.buffer_history = saved_state['buffer_history']
                    
                    restored_count += 1
                    
                    # Count different types of restored states
                    if validator.cooldown_violations > 0:
                        violation_count += 1
                    if validator.emergency_blacklist_until and time.time() < validator.emergency_blacklist_until:
                        blacklisted_count += 1
                        validator.is_active = False  # Ensure blacklisted validators are inactive
                    if validator.cooldown_until and time.time() < validator.cooldown_until:
                        cooldown_count += 1
                    
                    # Log restoration for validators with critical states
                    if (validator.cooldown_violations > 50 or 
                        validator.emergency_blacklist_until or 
                        validator.cooldown_until):
                        
                        status = self.get_cooldown_status(validator)
                        self.logger.warning(f"�� Restored UID {uid}: {status}")
                        
                        if validator.cooldown_violations > 100:
                            remaining_time = ""
                            if validator.emergency_blacklist_until:
                                remaining_seconds = validator.emergency_blacklist_until - time.time()
                                if remaining_seconds > 0:
                                    remaining_time = f" (blacklisted for {remaining_seconds/3600:.1f}h more)"
                            
                            self.logger.error(f"🚨 CRITICAL: UID {uid} has {validator.cooldown_violations} violations{remaining_time}")
                else:
                    self.logger.debug(f"⚠️ Saved state for UID {uid} found but validator not in current set")
            
            # Summary logging
            self.logger.info(f"✅ Restored {restored_count} validator states from disk")
            if violation_count > 0:
                self.logger.warning(f"⚠️ {violation_count} validators restored with violations")
            if blacklisted_count > 0:
                self.logger.error(f"🚨 {blacklisted_count} validators restored as EMERGENCY BLACKLISTED")
            if cooldown_count > 0:
                self.logger.warning(f"⏳ {cooldown_count} validators restored with active cooldowns")
            
            # Update statistics
            self.stats['validators_restored_from_disk'] = restored_count
            self.stats['violations_restored_from_disk'] = violation_count
            self.stats['blacklists_restored_from_disk'] = blacklisted_count
            
        except Exception as e:
            self.logger.error(f"❌ Failed to restore validator states: {e}")
            self.logger.error(f"   Traceback: {traceback.format_exc()}")
    
    def save_validator_states_to_disk(self):
        """
        Save current validator states to disk.
        """
        try:
            success = self.state_persistence.save_validator_states(self.validators)
            if success:
                self.stats['validator_states_saved'] = self.stats.get('validator_states_saved', 0) + 1
            else:
                self.stats['validator_state_save_failures'] = self.stats.get('validator_state_save_failures', 0) + 1
        except Exception as e:
            self.logger.error(f"❌ Failed to save validator states: {e}")
            self.stats['validator_state_save_failures'] = self.stats.get('validator_state_save_failures', 0) + 1
    
    def _register_shutdown_handlers(self):
        """
        Register handlers to save state on graceful shutdown.
        """
        def shutdown_handler(signum, frame):
            self.logger.info(f"🛑 Received shutdown signal {signum} - saving validator states...")
            try:
                self.save_validator_states_to_disk()
                self.logger.info("💾 Validator states saved successfully on shutdown")
            except Exception as e:
                self.logger.error(f"❌ Failed to save states on shutdown: {e}")
            
            # Exit gracefully
            import sys
            sys.exit(0)
        
        def atexit_handler():
            self.logger.info("🛑 Script exiting - saving validator states...")
            try:
                self.save_validator_states_to_disk()
                self.logger.info("💾 Validator states saved successfully on exit")
            except Exception as e:
                self.logger.error(f"❌ Failed to save states on exit: {e}")
        
        # Register signal handlers
        signal.signal(signal.SIGTERM, shutdown_handler)
        signal.signal(signal.SIGINT, shutdown_handler)
        
        # Register atexit handler
        atexit.register(atexit_handler)
        
        self.logger.info("🔒 Registered shutdown handlers for state persistence")
    
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
            
            task = None
            if response and len(response) > 0:
                resp = response[0]

                if hasattr(resp, 'task') and resp.task and resp.task.prompt:
                    # SHARED TASK TRACKING: Check if this task is already being processed by another instance
                    if self.config.get('enable_task_tracking', True) and not self.config.get('disable_task_tracking', False):
                        if not self.db.acquire_task_lock(resp.task.id, validator.uid, self.instance_id, timeout_minutes=2):
                            self.logger.info(f"⏭️ Task {resp.task.id} already being processed by another instance - skipping UID {validator.uid}")
                            return None
                    
                    # Check for duplicates with detailed analysis (only if enabled)
                    if self.config.get('enable_duplicate_checking', True):
                        if self.db.is_duplicate_prompt(resp.task.prompt, validator.uid, self.config['duplicate_check_hours']):
                            # Release the task lock since we're not processing this task
                            if self.config.get('enable_task_tracking', True) and not self.config.get('disable_task_tracking', False):
                                self.db.release_task_lock(resp.task.id, self.instance_id, status='skipped_duplicate')
                            
                            # Get analysis for this validator to understand why it's being skipped
                            analysis = self.db.get_duplicate_analysis(validator.uid, 6)  # Last 6 hours
                            self.logger.info(f"⏭️ Skipping duplicate from UID {validator.uid}: '{resp.task.prompt[:50]}...'")
                            self.logger.info(f"   Analysis: {analysis['successful_tasks']}/{analysis['total_tasks_pulled']} successful, {analysis['failed_tasks']} failed, {analysis['unprocessed_tasks']} unprocessed")
                            return None
                    else:
                        self.logger.debug(f"🔄 Duplicate checking disabled - processing prompt from UID {validator.uid}")
                    
                    # Update validator state
                    validator.total_tasks_pulled += 1
                    validator.last_task_received = time.time()
                    
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
                    self.logger.info(f"   🔒 Task lock acquired: {resp.task.id}")
                    
                    self.stats['tasks_pulled'] += 1

                else:
                    self.logger.debug(f"⚠️ No task from UID {validator.uid}")
                    return None
                
                # Enhanced cooldown and validation tracking - CRITICAL FIX
                if hasattr(resp, 'cooldown_until') and resp.cooldown_until:
                    # CRITICAL: Update validator cooldown from response and respect it
                    old_cooldown = validator.cooldown_until
                    validator.cooldown_until = resp.cooldown_until
                    
                    # Calculate remaining cooldown time
                    current_time = time.time()
                    if resp.cooldown_until > current_time:
                        remaining_cooldown = resp.cooldown_until - current_time
                        self.logger.warning(f"🚨 CRITICAL: Validator UID {validator.uid} enforced cooldown: {remaining_cooldown:.1f}s remaining")
                        self.logger.warning(f"   Previous cooldown: {old_cooldown}, New cooldown: {resp.cooldown_until}")
                        
                        # Set emergency cooldown to prevent further violations
                        self._set_emergency_cooldown(validator, resp.cooldown_until, "Validator enforced cooldown")
                    else:
                        self.logger.info(f"✅ Validator UID {validator.uid} cooldown cleared: {resp.cooldown_until}")
                
                if hasattr(resp, 'cooldown_violations') and resp.cooldown_violations:
                    # Track cooldown violations from validator - CRITICAL
                    old_violations = validator.cooldown_violations
                    validator.cooldown_violations = resp.cooldown_violations
                    
                    if resp.cooldown_violations > 0:
                        self.logger.error(f"🚨 CRITICAL: Validator UID {validator.uid} reported {resp.cooldown_violations} cooldown violations!")
                        
                        # Check if violations increased significantly
                        if resp.cooldown_violations > old_violations + 10:
                            self.logger.error(f"🚨 Violations increased by {resp.cooldown_violations - old_violations} - implementing emergency measures")
                            self._handle_critical_violations(validator, resp.cooldown_violations)
                        
                        # Check if we're over the threshold
                        violation_threshold = self.config.get('critical_violation_threshold', 100)
                        if resp.cooldown_violations > violation_threshold:
                            self.logger.error(f"🚨 UID {validator.uid} exceeds violation threshold ({violation_threshold}) - implementing blacklist")
                            self._blacklist_validator_temporarily(validator, resp.cooldown_violations)
                
                if hasattr(resp, 'throttle_period') and resp.throttle_period:
                    # Update throttle period from validator
                    validator.throttle_period = resp.throttle_period
                    self.logger.debug(f"⏱️ Validator UID {validator.uid} throttle period: {resp.throttle_period}s")

                return task if isinstance(task, TaskRecord) else None
                
            else:
                self.logger.debug(f"❌ No response from UID {validator.uid}")
                
                # Set cooldown for no response (validator might be overloaded)
                validator_cooldown = self.config.get('validator_error_cooldown', 45)
                self.set_validator_cooldown(validator, validator_cooldown, "No response received")
                
                return None
        
        except Exception as e:
            self.logger.error(f"❌ Error pulling from UID {validator.uid}: {e}")
            
            # Set cooldown for network/validator errors
            network_cooldown = self.config.get('network_error_cooldown', 30)
            self.set_validator_cooldown(validator, network_cooldown, f"Network error: {str(e)[:50]}")
            
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
    
    def route_prompt_to_optimal_lora(self, task: TaskRecord) -> Dict[str, Any]:
        """
        Route prompt to optimal LoRA using intelligent analysis.
        Returns dict with lora_name, endpoint, and reasoning.
        """
        # Check if LoRA routing is enabled
        if not self.config.get('enable_lora_routing', True):
            return {
                'lora_name': 'Patched Realism',
                'endpoint': '/generate/',
                'reasoning': 'LoRA routing disabled in config',
                'confidence': 'Low'
            }
        
        if not ORGANIC_LORA_ROUTER_AVAILABLE or not self.lora_router:
            # Fallback to default model
            return {
                'lora_name': 'Patched Realism',
                'endpoint': '/generate/',
                'reasoning': 'Default model (LoRA router not available)',
                'confidence': 'Low'
            }
        
        try:
            # Use organic router to select optimal LoRA through pattern learning
            router_result = self.lora_router.route_final(task.prompt, "edge_case")
            
            # Map LoRA names to endpoints
            lora_endpoints = {
                'Patched Realism': '/generate/',
                'Team Fortress 2 Style': '/generate/tf2_style/',
                'Cartoon 3D Render': '/generate/cartoon_3d/',
                '3D Game Assets': '/generate/game_assets/',
                'Game Icon Institute': '/generate/sd15_game_icon/',
                'Cinema Style': '/generate/cinema/',
                'Flux Isometric 3D': '/generate/isometric_3d/',
                'Baolei Style': '/generate/baolei_style/'
            }
            
            endpoint = lora_endpoints.get(router_result.recommended_lora, '/generate/')
            
            # Track routing decision
            self.stats['lora_routing_decisions'] += 1
            
            routing_info = {
                'lora_name': router_result.recommended_lora,
                'endpoint': endpoint,
                'reasoning': router_result.reasoning,
                'confidence': router_result.confidence
            }
            
            self.logger.info(f"🧠 LoRA Routing Decision:")
            self.logger.info(f"   Prompt: '{task.prompt[:50]}...'")
            self.logger.info(f"   Selected LoRA: {routing_info['lora_name']}")
            self.logger.info(f"   Endpoint: {routing_info['endpoint']}")
            self.logger.info(f"   Reasoning: {routing_info['reasoning']}")
            self.logger.info(f"   Confidence: {routing_info['confidence']}")
            
            return routing_info
            
        except Exception as e:
            self.logger.error(f"❌ LoRA routing failed: {e}")
            # Fallback to default
            return {
                'lora_name': 'Patched Realism',
                'endpoint': '/generate/',
                'reasoning': f'Routing failed: {str(e)}',
                'confidence': 'Low'
            }
    
    def optimize_prompt_for_generation(self, task: TaskRecord) -> Dict[str, Any]:
        """
        Optimize prompt and route to optimal LoRA.
        Returns dict with optimized_prompt, lora_info, and endpoint.
        """
        try:
            # Step 1: Route to optimal LoRA first
            # lora_info = self.route_prompt_to_optimal_lora(task)
            lora_info = {
                'lora_name': 'cinema',
                'endpoint': '/generate/cinema/',
                'reasoning': 'Default model (LoRA router not available)',
                'confidence': 'High'
            }
            # Step 2: Optimize prompt based on selected LoRA
            optimized_prompt = task.prompt  # Default to original
            
            if self.config.get('enable_prompt_optimization', True):
                # Check if reproducibility system is available and enabled
                if (REPRODUCIBILITY_SYSTEM_AVAILABLE and 
                    self.reproducibility_system and 
                    self.config.get('enable_reproducibility_optimization', True)):
                        
                        min_similarity = self.config.get('reproducibility_min_similarity', 0.3)
                        
                        # Use enhanced gold prompts (memory + logs) if real-time learning is enabled
                        if self.config.get('activate_learning', False):
                            enhanced_gold_prompts = self.get_fresh_gold_prompts()
                            gold_prompts_count = len(enhanced_gold_prompts)
                            self.logger.info(f"🚀 Using ENHANCED gold prompts: {gold_prompts_count} total (memory + logs)")
                            
                            # CRITICAL: Update the reproducibility system with enhanced gold prompts
                            # This ensures it uses the optimized versions instead of just original prompts
                            self.reproducibility_system.update_gold_standard_results(enhanced_gold_prompts)
                            self.logger.info(f"🔄 Updated reproducibility system with {len(enhanced_gold_prompts)} enhanced gold prompts")
                            
                            # Now use the standard reproducibility optimization with updated data
                            repro_result = self.reproducibility_system.optimize_prompt_with_reproducibility(
                                task.prompt, min_similarity, run_validation=False
                            )
                        else:
                            # Use standard episodic memory gold prompts
                            gold_prompts_count = len(self.reproducibility_system.gold_standard_results)
                            self.logger.debug(f"📚 Using {gold_prompts_count} gold prompts from episodic memory")
                            
                            # Log the similarity threshold being used
                            if self.config.get('log_optimization_details', True):
                                self.logger.info(f"🔍 Searching for gold prompts with similarity ≥ {min_similarity}")
                            
                            repro_result = self.reproducibility_system.optimize_prompt_with_reproducibility(
                                task.prompt, min_similarity, run_validation=False
                            )
                        
                        if repro_result:
                            optimized_prompt = repro_result['optimized_prompt']
                            similarity = repro_result['similarity']
                            gold_score = repro_result['gold_score']
                            
                            if self.config.get('log_optimization_details', True):
                                self.logger.info(f"🔄 Reproducibility optimization SUCCESS:")
                                self.logger.info(f"   Original: '{task.prompt}'")
                                self.logger.info(f"   Optimized: '{optimized_prompt}'")
                                self.logger.info(f"   Similarity: {similarity:.3f}")
                                self.logger.info(f"   Gold score: {gold_score:.4f}")
                                self.logger.info(f"   📚 Gold prompts available: {gold_prompts_count}")
                            else:
                                self.logger.info(f"🔄 Reproducibility optimized (sim: {similarity:.2f}, gold: {gold_score:.3f})")
                            
                            self.stats['prompts_optimized'] += 1
                            self.stats['reproducibility_optimizations'] = self.stats.get('reproducibility_optimizations', 0) + 1
                        else:
                            # Log when reproducibility optimization fails
                            if self.config.get('log_optimization_details', True):
                                self.logger.info(f"⚠️ Reproducibility optimization FAILED:")
                                self.logger.info(f"   Original: '{task.prompt}'")
                                self.logger.info(f"   Reason: No close gold prompt found (threshold: {min_similarity})")
                                self.logger.info(f"   📚 Gold prompts available: {gold_prompts_count}")
                                self.logger.info(f"   → Falling back to traditional optimization...")
                            else:
                                self.logger.info(f"⚠️ Reproducibility failed, using traditional optimization")
                            
                            # Try traditional optimization as fallback
                            if OPTIMIZED_PROMPT_OPTIMIZER_AVAILABLE:
                                if self.config.get('log_optimization_details', True):
                                    self.logger.info(f"�� Traditional optimization FALLBACK:")
                                    self.logger.info(f"   Original: '{task.prompt}'")
                                
                                result = self.prompt_optimizer.optimize_with_examples(task.prompt)
                                optimized_prompt = result
                                confidence = 0.8
                                
                                if self.config.get('log_optimization_details', True):
                                    self.logger.info(f"   Optimized: '{optimized_prompt}'")
                                    self.logger.info(f"   Confidence: {confidence:.1%}")
                                    self.logger.info(f"   Method: Fast examples-based optimization")
                                
                                self.stats['prompts_optimized'] += 1
                                self.stats['traditional_optimizations'] = self.stats.get('traditional_optimizations', 0) + 1
                            else:
                                # Use original prompt if no optimizer available
                                optimized_prompt = task.prompt
                                if self.config.get('log_optimization_details', True):
                                    self.logger.info(f"ℹ️ No optimizer available - using original prompt")
                        
                
                
            else:
                # Fallback to original optimizer
                if self.config.get('log_optimization_details', True):
                    self.logger.info(f"🔍 Original optimizer FALLBACK:")
                    self.logger.info(f"   Original: '{task.prompt}'")
                
                optimization_result = self.prompt_optimizer.optimize_prompt(
                    task.prompt, 
                    aggressive=self.config.get('optimization_aggressive_mode', False)
                )
                analysis = optimization_result['analysis']
                
                # Log the analysis if enabled
                if self.config.get('log_optimization_details', True):
                    self.logger.info(f"   Risk Level: {analysis['risk_level']}")
                    
                    if analysis['risk_factors']:
                        self.logger.info(f"   Risk Factors:")
                        for factor in analysis['risk_factors']:
                            self.logger.info(f"     • {factor}")
                
                self.stats['prompts_optimized'] += 1
                self.stats['traditional_optimizations'] = self.stats.get('traditional_optimizations', 0) + 1
            
            # Return comprehensive optimization result
            return {
                'optimized_prompt': optimized_prompt,
                'lora_info': lora_info,
                'endpoint': lora_info['endpoint'],
                'original_prompt': task.prompt
            }
                
        except Exception as e:
            self.logger.error(f"❌ Prompt optimization failed: {e}")
            # Return fallback result
            return {
                'optimized_prompt': task.prompt,
                'lora_info': {
                    'lora_name': 'Patched Realism',
                    'endpoint': '/generate/',
                    'reasoning': f'Optimization failed: {str(e)}',
                    'confidence': 'Low'
                },
                'endpoint': '/generate/',
                'original_prompt': task.prompt
            }

    def _clear_trellis_gpu_cache(self):
        """Send a request to the TRELLIS server to clear GPU cache."""
        try:
            url = f"{self.trellis_server_url}/clear_cache/"
            resp = requests.post(url, timeout=10)
            if resp.status_code == 200:
                self.logger.info(f"[TRELLIS] GPU cache cleared: {resp.json()}")
            else:
                self.logger.warning(f"[TRELLIS] Failed to clear GPU cache: HTTP {resp.status_code}")
        except Exception as e:
            self.logger.warning(f"[TRELLIS] Exception clearing GPU cache: {e}")


    async def generate_3d_model(self, task: TaskRecord) -> Optional[Dict[str, Any]]:
        """Generate 3D model using TRELLIS server with prompt optimization"""
        self.logger.info(f"�� Generating 3D model: '{task.prompt}' (task: {task.task_id})")
        
        try:
            # CRITICAL: Wait for priority access to the server
            # This is where we ensure subnet tasks get priority over optimizer tasks
            if not self.priority_coordinator.wait_for_priority_access(task.task_id):
                self.logger.error(f"❌ PRIORITY ACCESS TIMEOUT for task {task.task_id} - subnet task will be missed!")
                task.priority_access_timeout = True  # Mark this task as having priority access timeout
                return None
            
            # Mark the start of our priority job
            self.priority_coordinator.mark_priority_job_start(task.task_id, task.prompt)
            
            # Step 1: Optimize prompt and route to optimal LoRA
            optimization_result = self.optimize_prompt_for_generation(task)
            optimized_prompt = optimization_result['optimized_prompt']
            lora_info = optimization_result['lora_info']
            endpoint = optimization_result['endpoint']
            
            # Step 1.5: Clean the optimized prompt to remove artifacts
            cleaned_prompt = self.clean_optimized_prompt(optimized_prompt)
            # Only add "white background" if it's not already present
            # cleaned_prompt = optimized_prompt
            if "white background" not in cleaned_prompt.lower():
                cleaned_prompt = cleaned_prompt + " white background"
            # Log the final optimization result
            if self.config.get('log_optimization_details', True):
                if optimized_prompt != task.prompt:
                    self.logger.info(f"🎯 FINAL OPTIMIZATION RESULT:")
                    self.logger.info(f"   Original: '{task.prompt}'")
                    self.logger.info(f"   Optimized: '{optimized_prompt}'")
                    self.logger.info(f"   Cleaned: '{cleaned_prompt}'")
                    self.logger.info(f"   LoRA: {lora_info['lora_name']} via {endpoint}")
                else:
                    self.logger.info(f"ℹ️ No optimization applied - using original prompt")
                    self.logger.info(f"   Prompt: '{task.prompt}'")
                    self.logger.info(f"   LoRA: {lora_info['lora_name']} via {endpoint}")
            
            # Clear cache on the server using priority coordinator
            self.priority_coordinator.clear_server_cache()

            # Step 2: Get deterministic seed
            deterministic_seed = self.get_deterministic_seed(task)
            self.logger.info(f"   🎲 Using deterministic seed: {deterministic_seed}")
            self.logger.info(f"   �� Using LoRA: {lora_info['lora_name']} via {endpoint}")
            
            generation_start = time.time()
            
            # Call TRELLIS generation server with cleaned prompt, deterministic seed, and LoRA-specific endpoint
            full_url = f"{self.config['generation_server_url']}{endpoint}"
            response = requests.post(
                full_url,
                data={
                    'prompt': cleaned_prompt,  # Use cleaned prompt (artifacts removed)
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
                validator.last_submit_time = time.time()
                
                if task.submission_success and task.task_fidelity_score is not None:
                    validator.total_successful_submissions += 1
                    # Update average score with exponential moving average
                    if validator.average_score == 0:
                        validator.average_score = task.task_fidelity_score
                    else:
                        validator.average_score = validator.average_score * 0.9 + task.task_fidelity_score * 0.1
                    
                    # Check if we should set validation lock (successful submission)
                    validation_lock_duration = self.config.get('validation_lock_duration', 30)
                    if validation_lock_duration > 0:
                        self.set_validator_validation_lock(validator, validation_lock_duration, "Successful submission")
                        self.logger.debug(f"🔒 Validation lock set for UID {validator.uid} after successful submission")
                else:
                    # Failed submission - increment violations
                    self.increment_cooldown_violations(validator, "Failed submission")
                
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
                
                # Set cooldown for submission failures
                if task.validator_uid in self.validators:
                    validator = self.validators[task.validator_uid]
                    submission_cooldown = self.config.get('submission_failure_cooldown', 60)
                    self.set_validator_cooldown(validator, submission_cooldown, "No feedback received")
                
                return False
        
        except Exception as e:
            self.logger.error(f"❌ Submission failed: {e}")
            traceback.print_exc()
            task.submission_success = False
            
            # Set cooldown for submission exceptions
            if task.validator_uid in self.validators:
                validator = self.validators[task.validator_uid]
                submission_cooldown = self.config.get('submission_failure_cooldown', 60)
                self.set_validator_cooldown(validator, submission_cooldown, f"Submission exception: {str(e)[:50]}")
            
            return False
    
    async def process_task(self, task: TaskRecord) -> bool:
        """Process a single task end-to-end with priority access"""
        self.logger.info(f"�� Processing task {task.task_id}: '{task.prompt}'")
        
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
            
            # Calculate time elapsed since task was pulled
            time_after_generation = time.time()
            elapsed_time_since_pull = time_after_generation - task.pulled_at

            # If elapsed time is less than 17 seconds, wait until 18 seconds have passed
            if elapsed_time_since_pull < 17.0:
                wait_duration = 18.0 - elapsed_time_since_pull
                self.logger.info(f"⏳ Elapsed time since pull ({elapsed_time_since_pull:.2f}s) is < 17s. Waiting for {wait_duration:.2f}s to reach 18s before submission.")
                await asyncio.sleep(wait_duration)
                
            # Step 3: Submit results, passing the full generation result dictionary
            success = await self.submit_result(task, generation_result)
            
            # Save task record
            self.db.save_task(task)
            
            # SHARED TASK TRACKING: Release the task lock
            if self.config.get('enable_task_tracking', True) and not self.config.get('disable_task_tracking', False):
                if success:
                    self.db.release_task_lock(task.task_id, self.instance_id, status='completed')
                    self.logger.info(f"✅ Task {task.task_id} completed successfully")
                    self.logger.info(f"   🔓 Task lock released: {task.task_id}")
                else:
                    self.db.release_task_lock(task.task_id, self.instance_id, status='failed')
                    self.logger.error(f"❌ Task {task.task_id} submission failed")
                    self.logger.info(f"   🔓 Task lock released: {task.task_id}")
            
            return success
        
        except Exception as e:
            self.logger.error(f"❌ Task processing failed: {e}")
            traceback.print_exc()
            self.db.save_task(task)
            
            # SHARED TASK TRACKING: Release the task lock on exception
            if self.config.get('enable_task_tracking', True) and not self.config.get('disable_task_tracking', False):
                self.db.release_task_lock(task.task_id, self.instance_id, status='failed_exception')
                self.logger.info(f"   🔓 Task lock released on exception: {task.task_id}")
            
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
        self.logger.info(f"Prompts cleaned: {self.stats.get('prompts_cleaned', 0)}")
        self.logger.info(f"Reproducibility optimizations: {self.stats.get('reproducibility_optimizations', 0)}")
        self.logger.info(f"Traditional optimizations: {self.stats.get('traditional_optimizations', 0)}")
        self.logger.info(f"Optimization improvements: {self.stats['optimization_improvements']}")
        
        # Gold prompts statistics
        if REPRODUCIBILITY_SYSTEM_AVAILABLE and self.reproducibility_system:
            self.logger.info(f"Gold prompts available: {self.stats.get('gold_prompts_available', 0)}")
            self.logger.info(f"Gold prompts reloaded: {self.stats.get('gold_prompts_reloaded', 0)}")
            if hasattr(self, 'last_gold_prompts_reload'):
                time_since_reload = time.time() - self.last_gold_prompts_reload
                self.logger.info(f"Time since last gold prompts reload: {time_since_reload/3600:.1f} hours")
            
            # Real-time learning statistics
            if self.config.get('activate_learning', False):
                if self.config.get('only_log_learning', False):
                    log_count = self.config.get('log_learning_count', 6)
                    log_info = "all available logs" if log_count == -1 else f"most recent {log_count} logs"
                    
                    self.logger.info(f"🚀 ONLY-LOG-LEARNING STATISTICS:")
                    self.logger.info(f"   Enhanced gold prompts available: {self.stats.get('enhanced_gold_prompts_available', 0)}")
                    self.logger.info(f"   Enhanced reloads performed: {self.stats.get('enhanced_gold_prompts_reloaded', 0)}")
                    self.logger.info(f"   Total gold prompts (logs only): {self.stats.get('total_gold_prompts_available', 0)}")
                    self.logger.info(f"   From episodic memory: BYPASSED")
                    self.logger.info(f"   From recent logs: {self.stats.get('log_prompts', 0)} ({log_info})")
                    self.logger.info(f"   Live monitoring: DISABLED (logs only)")
                else:
                    self.logger.info(f"🚀 REAL-TIME LEARNING STATISTICS:")
                    self.logger.info(f"   Enhanced gold prompts available: {self.stats.get('enhanced_gold_prompts_available', 0)}")
                    self.logger.info(f"   Enhanced reloads performed: {self.stats.get('enhanced_gold_prompts_reloaded', 0)}")
                    self.logger.info(f"   Total gold prompts (memory + logs): {self.stats.get('total_gold_prompts_available', 0)}")
                    self.logger.info(f"   From episodic memory: {self.stats.get('memory_prompts', 0)}")
                    self.logger.info(f"   From recent logs: {self.stats.get('log_prompts', 0)}")
                    self.logger.info(f"   Live monitoring: ACTIVE")
            else:
                self.logger.info(f"📚 Real-time learning: DISABLED")
        
        # LLM Provider information
        if self.config.get('use_vllm', False):
            self.logger.info(f"🤖 LLM Provider: vLLM ({self.config.get('vllm_url', 'http://localhost:9000')})")
            self.logger.info(f"🤖 vLLM Model: {self.config.get('vllm_model', 'llama-3-2-3b-it')}")
        else:
            self.logger.info(f"🤖 LLM Provider: Ollama ({self.config.get('ollama_url', 'http://localhost:11434')}")
        
        # LoRA routing statistics
        self.logger.info(f"LoRA routing decisions: {self.stats.get('lora_routing_decisions', 0)}")
        self.logger.info(f"LoRA routing accuracy: {self.stats.get('lora_routing_accuracy', 0.0):.1f}%")
        
        # Priority access statistics
        self.logger.info(f"Priority access timeouts: {self.stats.get('priority_access_timeouts', 0)}")
        self.logger.info(f"Priority interruptions: {self.stats.get('priority_interruptions', 0)}")
        self.logger.info(f"Server unavailable skips: {self.stats.get('server_unavailable_skips', 0)}")
        self.logger.info(f"Server status check errors: {self.stats.get('server_status_check_errors', 0)}")
        
        # Validator blacklisting statistics
        self.logger.info(f"Blacklisted validators skipped: {self.stats.get('blacklisted_validators_skipped', 0)}")
        
        # Enhanced cooldown system statistics
        self.logger.info(f"Enhanced cooldown system:")
        self.logger.info(f"   Total cooldown violations: {self.stats.get('cooldown_violations_total', 0)}")
        self.logger.info(f"   Validation locks applied: {self.stats.get('validation_locks_applied', 0)}")
        self.logger.info(f"   Enhanced cooldown penalties: {self.stats.get('enhanced_cooldown_penalties', 0)}")
        
        # Emergency cooldown management statistics
        self.logger.info(f"Emergency cooldown management:")
        self.logger.info(f"   Emergency cooldowns applied: {self.stats.get('emergency_cooldowns_applied', 0)}")
        self.logger.info(f"   Critical violations handled: {self.stats.get('critical_violations_handled', 0)}")
        self.logger.info(f"   Validators temporarily blacklisted: {self.stats.get('validators_temporarily_blacklisted', 0)}")
        self.logger.info(f"   Validators reset from emergency: {self.stats.get('validators_reset_from_emergency', 0)}")
        self.logger.info(f"   Dynamic cooldown scaling applied: {self.stats.get('dynamic_cooldown_scaling', 0)}")
        self.logger.info(f"   Dynamic buffer applied: {self.stats.get('dynamic_buffer_applied', 0)}")
        
        # State persistence statistics
        self.logger.info(f"State persistence:")
        self.logger.info(f"   States saved to disk: {self.stats.get('validator_states_saved', 0)}")
        self.logger.info(f"   Save failures: {self.stats.get('validator_state_save_failures', 0)}")
        self.logger.info(f"   Validators restored from disk: {self.stats.get('validators_restored_from_disk', 0)}")
        self.logger.info(f"   Violations restored from disk: {self.stats.get('violations_restored_from_disk', 0)}")
        self.logger.info(f"   Blacklists restored from disk: {self.stats.get('blacklists_restored_from_disk', 0)}")
        
        # Enhanced cooldown statistics with DYNAMIC system health analysis
        active_validators = [v for v in self.validators.values() if v.is_active]
        validators_on_cooldown = [v for v in active_validators if v.cooldown_until and time.time() < v.cooldown_until]
        validators_validation_locked = [v for v in active_validators if v.validation_locked_until and time.time() < v.validation_locked_until]
        validators_with_violations = [v for v in active_validators if v.cooldown_violations > 0]
        validators_emergency_blacklisted = [v for v in self.validators.values() if v.emergency_blacklist_until and time.time() < v.emergency_blacklist_until]
        
        # DYNAMIC: Calculate system health and adjust task pulling strategy
        total_validators = len(self.validators)
        system_health_ratio = len(active_validators) / total_validators if total_validators > 0 else 0
        
        # Adjust task pulling strategy based on system health
        if system_health_ratio < 0.3:  # Critical system state
            task_pull_strategy = "CONSERVATIVE"
            max_concurrent_tasks = max(1, int(self.config.get('max_concurrent_tasks', 5) * 0.3))
            self.logger.error(f"🚨 CRITICAL SYSTEM STATE: Task pulling strategy set to CONSERVATIVE")
            self.logger.error(f"   Max concurrent tasks reduced to {max_concurrent_tasks} (from {self.config.get('max_concurrent_tasks', 5)})")
        elif system_health_ratio < 0.6:  # Degraded system state
            task_pull_strategy = "MODERATE"
            max_concurrent_tasks = max(2, int(self.config.get('max_concurrent_tasks', 5) * 0.6))
            self.logger.warning(f"⚠️ DEGRADED SYSTEM STATE: Task pulling strategy set to MODERATE")
            self.logger.warning(f"   Max concurrent tasks reduced to {max_concurrent_tasks} (from {self.config.get('max_concurrent_tasks', 5)})")
        else:  # Healthy system state
            task_pull_strategy = "AGGRESSIVE"
            max_concurrent_tasks = self.config.get('max_concurrent_tasks', 5)
            self.logger.info(f"✅ HEALTHY SYSTEM STATE: Task pulling strategy set to AGGRESSIVE")
            self.logger.info(f"   Max concurrent tasks: {max_concurrent_tasks}")
        
        # Store dynamic strategy for use in task pulling
        self.current_task_pull_strategy = task_pull_strategy
        self.current_max_concurrent_tasks = max_concurrent_tasks
        
        if validators_on_cooldown or validators_validation_locked or validators_emergency_blacklisted:
            total_restricted = len(validators_on_cooldown) + len(validators_validation_locked) + len(validators_emergency_blacklisted)
            self.logger.info(f"⏳ Validators with restrictions: {len(validators_on_cooldown)} cooldown, {len(validators_validation_locked)} validation locked, {len(validators_emergency_blacklisted)} emergency blacklisted")
            
            # Show emergency blacklisted validators first (most critical)
            if validators_emergency_blacklisted:
                self.logger.warning(f"🚨 EMERGENCY BLACKLISTED VALIDATORS:")
                for validator in validators_emergency_blacklisted[:3]:  # Show first 3
                    cooldown_status = self.get_cooldown_status(validator)
                    self.logger.warning(f"   UID {validator.uid}: {cooldown_status}")
            
            # Show other restricted validators
            other_restricted = validators_on_cooldown + validators_validation_locked
            if other_restricted:
                self.logger.info(f"⏳ Other restricted validators:")
                for validator in other_restricted[:5]:  # Show first 5
                    cooldown_status = self.get_cooldown_status(validator)
                    self.logger.info(f"   UID {validator.uid}: {cooldown_status}")
        else:
            self.logger.info(f"✅ No validators currently restricted")
        
        if validators_with_violations:
            self.logger.info(f"⚠️ Validators with cooldown violations: {len(validators_with_violations)}")
            for validator in validators_with_violations[:3]:  # Show first 3
                self.logger.info(f"   UID {validator.uid}: {validator.cooldown_violations} violations")
        else:
            self.logger.info(f"✅ No validators with cooldown violations")
        
        # DYNAMIC: Log system health summary
        self.logger.info(f"📊 SYSTEM HEALTH SUMMARY:")
        self.logger.info(f"   Total validators: {total_validators}")
        self.logger.info(f"   Active validators: {len(active_validators)} ({system_health_ratio:.1%})")
        self.logger.info(f"   Task pull strategy: {task_pull_strategy}")
        self.logger.info(f"   Max concurrent tasks: {max_concurrent_tasks}")
        blacklist = self.config.get('validator_blacklist', [])
        blacklist_enabled = self.config.get('enable_validator_blacklisting', True)
        if blacklist:
            self.logger.info(f"Current blacklist: {blacklist}")
            self.logger.info(f"Blacklisting: {'ENABLED' if blacklist_enabled else 'DISABLED'}")
            
            # Show which blacklisted UIDs are currently active
            active_blacklisted = [uid for uid in blacklist if uid in self.validators and self.validators[uid].is_active]
            if active_blacklisted:
                self.logger.info(f"🚫 Active blacklisted UIDs: {active_blacklisted}")
            else:
                self.logger.info(f"✅ No blacklisted UIDs are currently active on the subnet")
        else:
            self.logger.info(f"No validators in blacklist")
        
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
            cooldown_status = self.get_cooldown_status(validator)
            self.logger.info(f"  UID {validator.uid}: {validator.total_tasks_received} tasks, avg score: {validator.average_score:.3f}, cooldown: {cooldown_status}")
        
        # SHARED TASK TRACKING: Show task distribution across instances
        if self.config.get('enable_task_tracking', True) and not self.config.get('disable_task_tracking', False):
            task_stats = self.db.get_task_processing_stats()
            if task_stats:
                self.logger.info(f"🔄 Shared Task Tracking:")
                self.logger.info(f"  Total tracked tasks: {task_stats.get('total_tasks', 0)}")
                self.logger.info(f"  Active tasks: {task_stats.get('active_tasks', 0)}")
                self.logger.info(f"  Completed tasks: {task_stats.get('completed_tasks', 0)}")
                
                # Show instance distribution
                instance_counts = task_stats.get('instance_counts', {})
                if instance_counts:
                    self.logger.info(f"  Task distribution by instance:")
                    for instance_id, count in instance_counts.items():
                        if instance_id == self.instance_id:
                            self.logger.info(f"    {instance_id[:20]}...: {count} tasks (this instance)")
                        else:
                            self.logger.info(f"    {instance_id[:20]}...: {count} tasks")
                
                # Show validator distribution
                validator_counts = task_stats.get('validator_counts', {})
                if validator_counts:
                    self.logger.info(f"  Active tasks by validator:")
                    for uid, count in sorted(validator_counts.items()):
                        self.logger.info(f"    UID {uid}: {count} active tasks")
        else:
            self.logger.info(f"🔄 Shared Task Tracking: DISABLED")
        
        # Duplicate checking status
        if self.config.get('enable_duplicate_checking', True):
            self.logger.info(f"🔄 Duplicate Checking: ENABLED")
        else:
            self.logger.info(f"🔄 Duplicate Checking: DISABLED")
        
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
        
        try:
            while self.running:
                current_time = time.time()
                
                # Periodic validator refresh (every 10 minutes to catch changes)
                if current_time - last_validator_refresh > 600:
                    self.refresh_validators()
                    last_validator_refresh = current_time
                
                # Periodic cleanup
                if current_time - last_cleanup > self.config['cleanup_interval']:
                    self.db.cleanup_old_prompts()
                    
                    # SHARED TASK TRACKING: Clean up expired task locks
                    if self.config.get('enable_task_tracking', True) and not self.config.get('disable_task_tracking', False):
                        self.db.cleanup_expired_locks(timeout_minutes=2)
                    
                    # Check and clear expired emergency blacklists
                    self._check_and_clear_expired_emergency_blacklists()
                    
                    # Check for validators that need extended monitoring
                    self._check_validators_needing_monitoring()
                    
                    # PERIODIC: Save validator states to disk
                    self.save_validator_states_to_disk()
                    
                    last_cleanup = current_time
                
                # SHARED TASK TRACKING: Get available validators (not busy with other instances)
                if self.config.get('enable_task_tracking', True) and not self.config.get('disable_task_tracking', False):
                    available_validators = self.db.get_available_validators(exclude_instance_id=self.instance_id)
                    if available_validators:
                        self.logger.debug(f"📊 Available validators: {available_validators}")
                
                # Pull tasks from all available validators
                new_task_found = False
                
                for validator in self.validators.values():
                    if not self.running:
                        break
                    
                    # Log validator availability check
                    if not self.is_validator_available(validator):
                        continue  # Skip unavailable validators
                    
                    # SHARED TASK TRACKING: Skip validators that are busy with other instances
                    if self.config.get('enable_task_tracking', True) and not self.config.get('disable_task_tracking', False):
                        if self.db.is_validator_busy(validator.uid, exclude_instance_id=self.instance_id):
                            self.logger.debug(f"⏳ Validator UID {validator.uid} busy with other instance - skipping")
                            continue
                    
                    self.logger.debug(f"📡 Attempting to pull task from UID {validator.uid}")
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
                
                # Periodic gold prompts reload
                if (REPRODUCIBILITY_SYSTEM_AVAILABLE and 
                    self.reproducibility_system and 
                    current_time - self.last_gold_prompts_reload > self.gold_prompts_reload_interval):
                    
                    if self.config.get('activate_learning', False):
                        # Use enhanced reload with real-time learning
                        self.enhanced_reload_gold_prompts()
                    else:
                        # Use standard reload
                        self.reload_gold_prompts()
                
                # Wait before next cycle
                await asyncio.sleep(2)  # Short sleep between cycles
        
        except KeyboardInterrupt:
            self.logger.info("🛑 Mining interrupted by user")
        except Exception as e:
            self.logger.error(f"❌ Mining loop error: {e}")
            traceback.print_exc()
        finally:
            self.running = False
            
            # Stop live monitoring if enabled
            if self.config.get('activate_learning', False):
                self.stop_live_monitoring()
            
            self.print_status()
            self.save_statistics()
            self.logger.info("🏁 Continuous mining stopped")
    
    def _on_priority_interruption(self):
        """Callback when priority interruption occurs"""
        self.stats['priority_interruptions'] = self.stats.get('priority_interruptions', 0) + 1
        self.logger.info(f"📊 Priority interruption tracked: {self.stats['priority_interruptions']} total")
    
    def reload_gold_prompts(self):
        """Reload gold prompts from episodic memory to get fresh data"""
        if not REPRODUCIBILITY_SYSTEM_AVAILABLE or not self.reproducibility_system:
            return
        
        try:
            self.logger.info("📚 Reloading gold prompts from episodic memory...")
            
            # Reload the episodic memory
            old_count = len(self.reproducibility_system.gold_standard_results)
            self.reproducibility_system.gold_standard_results = self.reproducibility_system._load_episodic_memory()
            new_count = len(self.reproducibility_system.gold_standard_results)
            
            # Update timestamp and statistics
            self.last_gold_prompts_reload = time.time()
            self.stats['gold_prompts_reloaded'] += 1
            self.stats['gold_prompts_available'] = new_count
            
            # Log the results
            if new_count > old_count:
                self.logger.info(f"✅ Gold prompts updated: {old_count} → {new_count} (+{new_count - old_count})")
            elif new_count < old_count:
                self.logger.info(f"⚠️ Gold prompts updated: {old_count} → {new_count} (-{old_count - new_count})")
            else:
                self.logger.info(f"🔄 Gold prompts reloaded: {new_count} prompts (no change in count)")
            
            # Log some sample prompts for verification
            if new_count > 0:
                sample_prompts = list(self.reproducibility_system.gold_standard_results.keys())[:3]
                self.logger.info(f"   📝 Sample gold prompts:")
                for i, prompt in enumerate(sample_prompts, 1):
                    self.logger.info(f"     {i}. '{prompt[:60]}...'")
            
        except Exception as e:
            self.logger.error(f"❌ Failed to reload gold prompts: {e}")
            traceback.print_exc()

    def clean_optimized_prompt(self, prompt: str) -> str:
        """
        Clean up common artifacts and formatting issues from optimized prompts.
        Removes common prefixes, suffixes, and formatting artifacts that shouldn't be sent to generation.
        
        Args:
            prompt: The raw optimized prompt that may contain artifacts
            
        Returns:
            Cleaned prompt ready for generation
        """
        # Check if prompt cleaning is enabled
        if not self.config.get('enable_prompt_cleaning', True):
            return prompt
            
        if not prompt:
            return prompt
        
        # Common artifacts to remove (case insensitive)
        artifacts_to_remove = [
            "wbgmsst", "wbgmsst,", "wbgmsst, ",  # Common prefix artifact
            "wbgsst", "wbgsst,", "wbgsst, ",     # Variant of above
            "wbgms", "wbgms,", "wbgms, ",        # Shorter variant
            "wbgs", "wbgs,", "wbgs, ",           # Even shorter variant
        ]
        
        cleaned_prompt = prompt
        
        # Remove artifacts from the beginning of the prompt
        for artifact in artifacts_to_remove:
            if cleaned_prompt.lower().startswith(artifact.lower()):
                cleaned_prompt = cleaned_prompt[len(artifact):].lstrip()
                self.logger.debug(f"🧹 Removed artifact '{artifact}' from prompt start")
                break
        
        # Remove artifacts that might appear elsewhere (with context)
        for artifact in artifacts_to_remove:
            # Remove standalone artifacts (with proper word boundaries)
            import re
            pattern = r'\b' + re.escape(artifact) + r'\b'
            if re.search(pattern, cleaned_prompt, re.IGNORECASE):
                cleaned_prompt = re.sub(pattern, '', cleaned_prompt, flags=re.IGNORECASE)
                self.logger.debug(f"🧹 Removed artifact '{artifact}' from prompt body")
        
        # Clean up extra whitespace and punctuation
        cleaned_prompt = cleaned_prompt.strip()
        
        # Remove leading commas and extra punctuation
        while cleaned_prompt.startswith(','):
            cleaned_prompt = cleaned_prompt[1:].lstrip()
        
        # Log if cleaning was performed
        if cleaned_prompt != prompt:
            self.logger.info(f"🧹 Prompt cleaned:")
            self.logger.info(f"   Before: '{prompt}'")
            self.logger.info(f"   After:  '{cleaned_prompt}'")
            # Track cleaning statistics
            self.stats['prompts_cleaned'] = self.stats.get('prompts_cleaned', 0) + 1
        else:
            self.logger.debug(f"🧹 No cleaning needed for prompt: '{prompt[:50]}...'")
        
        return cleaned_prompt

    # ===== REAL-TIME LEARNING INTEGRATION FUNCTIONS =====
    
    def parse_current_episode_logs(self) -> Dict[str, Any]:
        """
        Parse current episode logs to get real-time learning improvements.
        This extracts gold prompts and optimization results from the most recent logs.
        
        Returns:
            Dictionary of current gold prompts with their optimization data
        """
        current_gold_prompts = {}
        
        try:
            # Find the most recent episode log
            log_dir = Path("episodic_logs_first")
            if not log_dir.exists():
                self.logger.debug("📁 Log directory not found, skipping log parsing")
                return current_gold_prompts
            
            recent_logs = sorted(log_dir.glob("episodic_run_*.log"), key=lambda x: x.stat().st_mtime)
            
            if not recent_logs:
                self.logger.debug("📁 No episode logs found, skipping log parsing")
                return current_gold_prompts
            
            # Determine how many logs to parse based on configuration
            if self.config.get('only_log_learning', False):
                # Use log_learning_count for only-log-learning mode
                log_count = self.config.get('log_learning_count', 6)
                if log_count == -1:
                    latest_logs = recent_logs  # Use all logs
                    self.logger.debug(f"📖 ONLY-LOG-LEARNING: Parsing all {len(recent_logs)} available logs")
                else:
                    latest_logs = recent_logs[-log_count:]  # Use most recent N logs
                    self.logger.debug(f"📖 ONLY-LOG-LEARNING: Parsing most recent {len(latest_logs)} logs (limited by --only-log-learning={log_count})")
            else:
                # Use max_logs_to_parse for standard mode
                max_logs_to_parse = self.config.get('max_logs_to_parse', 10)
                if isinstance(max_logs_to_parse, int) and max_logs_to_parse == -1:
                    latest_logs = recent_logs
                    self.logger.debug(f"📖 Standard mode: Parsing all {len(recent_logs)} available logs")
                else:
                    latest_logs = recent_logs[-int(max_logs_to_parse):]
                    self.logger.debug(f"📖 Standard mode: Parsing most recent {len(latest_logs)} logs (limited by max_logs_to_parse={max_logs_to_parse})")
            
            for log_file in latest_logs:
                self.logger.debug(f"📖 Parsing log file: {log_file.name}")
                
                try:
                    with open(log_file, 'r') as f:
                        content = f.read()
                        
                    # Extract optimization results from the log
                    log_prompts = self._extract_optimization_results_from_log(content)
                    
                    # Merge with current results (newer logs take precedence)
                    for prompt, data in log_prompts.items():
                        if prompt not in current_gold_prompts or data.get('timestamp', 0) > current_gold_prompts[prompt].get('timestamp', 0):
                            current_gold_prompts[prompt] = data
                            
                except Exception as e:
                    self.logger.warning(f"⚠️ Failed to parse log {log_file.name}: {e}")
                    continue
            
            if current_gold_prompts:
                log_source = "ONLY-LOG-LEARNING" if self.config.get('only_log_learning', False) else "standard mode"
                self.logger.info(f"📚 Parsed {len(current_gold_prompts)} prompts from {len(latest_logs)} logs ({log_source})")
                self.stats['log_parsed_prompts'] = len(current_gold_prompts)
            else:
                self.logger.debug("📚 No prompts found in recent logs")
                
        except Exception as e:
            self.logger.error(f"❌ Error parsing episode logs: {e}")
            
        return current_gold_prompts
    
    def _extract_optimization_results_from_log(self, log_content: str) -> Dict[str, Any]:
        """
        Extract optimization results from a single log file.
        This captures ALL prompts being optimized with their scores and optimized versions.
        
        Args:
            log_content: Content of the log file
            
        Returns:
            Dictionary of prompts with their optimization data
        """
        extracted_prompts = {}
        
        try:
            import re
            
            # Split content into lines for better parsing
            lines = log_content.split('\n')
            
            # Process each line to find optimization data
            current_prompt = None
            current_score = None
            current_optimized = None
            current_round = 0
            
            def _normalize_prompt_text(text: str) -> str:
                if not isinstance(text, str):
                    return text
                s = text.strip()
                # Remove paired double/single quotes around the entire string
                if (s.startswith("''") and s.endswith("''")) or (s.startswith('""') and s.endswith('""')):
                    s = s[2:-2].strip()
                if (s.startswith("'") and s.endswith("'")) or (s.startswith('"') and s.endswith('"')):
                    s = s[1:-1].strip()
                return s
            
            for i, line in enumerate(lines):
                # Find "Original:" lines to get the prompt
                if 'Original:' in line:
                    current_prompt = _normalize_prompt_text(line.split('Original:')[1].strip())
                    current_score = None
                    current_optimized = None
                    current_round = 0
                
                # Find "Optimized:" lines to get the optimized version (old format)
                elif 'Optimized:' in line and 'wbgmsst,' in line:
                    if current_prompt:
                        # Extract the optimized prompt (remove "wbgmsst," prefix)
                        optimized_text = line.split('Optimized:')[1].strip()
                        if optimized_text.startswith('wbgmsst,'):
                            current_optimized = optimized_text[8:].strip()  # Remove "wbgmsst," prefix
                        else:
                            current_optimized = optimized_text
                
                # Find "Using optimized prompt for generation:" lines (new format from logs)
                elif '📝 Using optimized prompt for generation:' in line:
                    if current_prompt:
                        # Extract the optimized prompt from quotes
                        optimized_match = re.search(r"'([^']+)'", line)
                        if optimized_match:
                            current_optimized = optimized_match.group(1).strip()
                            # Clean up common artifacts
                            if current_optimized.endswith('...'):
                                current_optimized = current_optimized[:-3].strip()
                            if current_optimized.endswith('front view, white background'):
                                current_optimized = current_optimized[:-28].strip()
                            if current_optimized.endswith('white background'):
                                current_optimized = current_optimized[:-16].strip()
                
                # Find "Validation score:" lines to get the score
                elif '📊 Validation score:' in line:
                    score_match = re.search(r'📊 Validation score: ([\d.]+)', line)
                    if score_match and current_prompt:
                        current_score = float(score_match.group(1))
                        
                        # Create or update prompt data
                        if current_prompt not in extracted_prompts:
                            extracted_prompts[current_prompt] = {
                                'original_prompt': current_prompt,
                                'optimized_prompt': current_optimized or current_prompt,
                                'best_score': current_score,
                                'current_round': current_round,
                                'is_gold': current_score > 0.75,
                                'source': 'log_parsing',
                                'method': 'comprehensive_extraction',
                                'status': 'completed' if current_score > 0 else 'optimizing'
                            }
                        else:
                            # Update with better score if found
                            existing_data = extracted_prompts[current_prompt]
                            if current_score > existing_data.get('best_score', 0.0):
                                existing_data['best_score'] = current_score
                                existing_data['is_gold'] = current_score > 0.75
                                if current_optimized:
                                    existing_data['optimized_prompt'] = current_optimized
                
                # Find round information: "🔄 RL Round X/20"
                elif '🔄 RL Round' in line:
                    round_match = re.search(r'🔄 RL Round (\d+)/20', line)
                    if round_match and current_prompt:
                        current_round = int(round_match.group(1))
                        if current_prompt in extracted_prompts:
                            extracted_prompts[current_prompt]['current_round'] = current_round
                
                # Find episode and prompt numbers
                elif '--- Episode' in line and 'Prompt' in line:
                    episode_match = re.search(r'--- Episode (\d+), Prompt (\d+) \(Total: (\d+)\) ---', line)
                    if episode_match and current_prompt:
                        episode, prompt_num, total = episode_match.groups()
                        if current_prompt in extracted_prompts:
                            extracted_prompts[current_prompt].update({
                                'episode': int(episode),
                                'prompt_number': int(prompt_num),
                                'total_prompts': int(total)
                            })
                
                # Find timestamps
                elif re.match(r'\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}', line):
                    timestamp_match = re.search(r'(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})', line)
                    if timestamp_match and current_prompt:
                        timestamp = timestamp_match.group(1)
                        if current_prompt in extracted_prompts:
                            extracted_prompts[current_prompt]['timestamp'] = timestamp
            
            # Log summary
            gold_count = sum(1 for p in extracted_prompts.values() if p.get('is_gold', False))
            total_count = len(extracted_prompts)
            
            if total_count > 0:
                self.logger.debug(f"📊 Log parsing summary: {total_count} prompts, {gold_count} gold prompts")
                
        except Exception as e:
            self.logger.error(f"❌ Error extracting optimization results: {e}")
            
        return extracted_prompts
    
    def get_fresh_gold_prompts(self) -> Dict[str, Any]:
        """
        Get gold prompts from both episodic memory and current logs for comprehensive coverage.
        This combines stable episodic memory data with real-time log data, prioritizing highest scores.
        
        Returns:
            Combined dictionary of gold prompts with real-time updates
        """
        combined_prompts = {}
        
        try:
            # Check if we should bypass episodic memory (only-log-learning mode)
            if self.config.get('only_log_learning', False):
                self.logger.info("📖 ONLY-LOG-LEARNING mode: Bypassing episodic memory, using only log data")
                memory_prompts = {}
                self.logger.debug("📚 Skipping episodic memory due to --only-log-learning flag")
            else:
                # Get from episodic memory (stable, complete)
                if self.reproducibility_system:
                    memory_prompts = self.reproducibility_system.gold_standard_results
                    self.logger.debug(f"📚 Loaded {len(memory_prompts)} prompts from episodic memory")
                    
                    # Convert episodic memory format to our standard format
                    for prompt, data in memory_prompts.items():
                        if 'method_2_hybrid_example' in data:
                            # Extract the optimized prompt and score
                            method_data = data['method_2_hybrid_example']
                            optimized_prompt = method_data.get('optimized_prompt', prompt)
                            score = method_data.get('validation_results', {}).get('validation_engine_score', 0.0)
                            
                            combined_prompts[prompt] = {
                                'original_prompt': prompt,
                                'optimized_prompt': optimized_prompt,
                                'best_score': score,
                                'source': 'episodic_memory',
                                'method': 'episodic_memory',
                                'status': 'completed',
                                'is_gold': score > 0.75
                            }
                        else:
                            # Fallback for other formats
                            combined_prompts[prompt] = {
                                'original_prompt': prompt,
                                'optimized_prompt': prompt,
                                'best_score': 0.0,
                                'source': 'episodic_memory',
                                'method': 'episodic_memory',
                                'status': 'unknown',
                                'is_gold': False
                            }
                else:
                    self.logger.debug("📚 Reproducibility system not available, skipping episodic memory")
                    memory_prompts = {}
            
            # Get from current logs (real-time, partial)
            if self.config.get('activate_learning', False):
                log_prompts = self.parse_current_episode_logs()
                self.logger.debug(f"📖 Loaded {len(log_prompts)} prompts from recent logs")
                
                # Sort log prompts by score (highest first) to prioritize best ones
                sorted_log_prompts = sorted(
                    log_prompts.items(), 
                    key=lambda x: x[1].get('best_score', 0.0), 
                    reverse=True
                )
                
                # In only-log-learning mode, all prompts come from logs
                if self.config.get('only_log_learning', False):
                    self.logger.info(f"📖 ONLY-LOG-LEARNING: Using {len(log_prompts)} prompts exclusively from logs")
                    for prompt, data in sorted_log_prompts:
                        combined_prompts[prompt] = data
                else:
                    # Merge them intelligently (logs take precedence for duplicates and high scores)
                    for prompt, data in sorted_log_prompts:
                        if prompt in combined_prompts:
                            # Check if log data has better score
                            existing_data = combined_prompts[prompt]
                            existing_score = existing_data.get('best_score', 0.0)
                            log_score = data.get('best_score', 0.0)
                            
                            # Prefer log data if it has higher score
                            if log_score > existing_score:
                                self.logger.debug(f"🔄 Updating prompt '{prompt[:30]}...' with better log data (score: {existing_score:.4f} → {log_score:.4f})")
                                # Merge data intelligently, keeping best of both
                                merged_data = existing_data.copy()
                                merged_data.update(data)
                                # Ensure we keep the best score
                                merged_data['best_score'] = log_score
                                combined_prompts[prompt] = merged_data
                            elif log_score == existing_score and data.get('timestamp') and existing_data.get('timestamp'):
                                # If scores are equal, prefer newer data and merge
                                if data['timestamp'] > existing_data['timestamp']:
                                    self.logger.debug(f"🔄 Updating prompt '{prompt[:30]}...' with newer log data")
                                    merged_data = existing_data.copy()
                                    merged_data.update(data)
                                    combined_prompts[prompt] = merged_data
                            elif log_score == existing_score:
                                # If scores are equal but no timestamp, merge to get complete data
                                self.logger.debug(f"🔄 Merging data for prompt '{prompt[:30]}...' with equal scores")
                                merged_data = existing_data.copy()
                                merged_data.update(data)
                                combined_prompts[prompt] = merged_data
                        else:
                            # New prompt from logs - add it
                            combined_prompts[prompt] = data
                        
                # Log the merge results
                memory_count = len(memory_prompts) if 'memory_prompts' in locals() else 0
                log_count = len(log_prompts)
                total_count = len(combined_prompts)
                
                if self.config.get('only_log_learning', False):
                    self.logger.info(f"📖 ONLY-LOG-LEARNING results:")
                    self.logger.info(f"   📖 From recent logs: {log_count}")
                    self.logger.info(f"   🔄 Total available: {total_count}")
                    self.logger.info(f"   📚 Episodic memory: BYPASSED")
                else:
                    self.logger.info(f"🔄 Enhanced merge results:")
                    self.logger.info(f"   📚 From episodic memory: {memory_count}")
                    self.logger.info(f"   📖 From recent logs: {log_count}")
                    self.logger.info(f"   🔄 Total combined: {total_count}")
                    
                    # Verify we didn't lose any prompts (only when not in only-log-learning mode)
                    if total_count < memory_count:
                        self.logger.warning(f"⚠️ WARNING: Lost {memory_count - total_count} prompts during merge!")
                        self.logger.warning(f"   Expected: {memory_count + log_count}, Got: {total_count}")
                        
                        # Debug: show what we have
                        memory_prompts_set = set(memory_prompts.keys())
                        log_prompts_set = set(log_prompts.keys())
                        combined_prompts_set = set(combined_prompts.keys())
                        
                        self.logger.debug(f"   Memory prompts: {len(memory_prompts_set)}")
                        self.logger.debug(f"   Log prompts: {len(log_prompts_set)}")
                        self.logger.debug(f"   Combined prompts: {len(combined_prompts_set)}")
                        
                        # Show what's missing
                        missing_from_memory = memory_prompts_set - combined_prompts_set
                        if missing_from_memory:
                            self.logger.warning(f"   Missing from memory: {len(missing_from_memory)} prompts")
                            for missing in list(missing_from_memory)[:3]:
                                self.logger.warning(f"     - '{missing[:50]}...'")
                
                # Show top scoring prompts from logs
                if log_prompts:
                    top_log_prompts = sorted(
                        log_prompts.items(), 
                        key=lambda x: x[1].get('best_score', 0.0), 
                        reverse=True
                    )[:5]  # Top 5
                    
                    self.logger.info(f"   🏆 Top scoring prompts from logs:")
                    for i, (prompt, data) in enumerate(top_log_prompts, 1):
                        score = data.get('best_score', 0.0)
                        round_info = f" (round {data.get('current_round', 0)})" if data.get('current_round', 0) > 0 else ""
                        self.logger.info(f"     {i}. Score {score:.4f}{round_info}: '{prompt[:50]}...'")
                
                # Show comprehensive top scoring prompts from combined data
                if combined_prompts:
                    top_combined_prompts = sorted(
                        combined_prompts.items(), 
                        key=lambda x: x[1].get('best_score', 0.0), 
                        reverse=True
                    )[:10]  # Top 10
                    
                    # if self.config.get('only_log_learning', False):
                    #     self.logger.info(f"   🏆 Top 10 scoring prompts (logs only):")
                    # else:
                    #     self.logger.info(f"   🏆 Top 10 scoring prompts (combined data):")
                        
                    # for i, (prompt, data) in enumerate(top_combined_prompts, 1):
                    #     score = data.get('best_score', 0.0)
                    #     source = data.get('source', 'unknown')
                    #     optimized = data.get('optimized_prompt', 'N/A')
                    #     self.logger.info(f"     {i:2d}. Score {score:.4f} ({source}): '{prompt[:50]}...'")
                    #     if optimized != prompt and len(optimized) > 50:
                    #         self.logger.info(f"         Optimized: '{optimized[:80]}...'")
                    #     elif optimized != prompt:
                    #         self.logger.info(f"         Optimized: '{optimized}'")
                
                # Show gold prompt count
                gold_count = sum(1 for p in combined_prompts.values() if p.get('best_score', 0.0) > 0.75)
                if self.config.get('only_log_learning', False):
                    self.logger.info(f"   🏆 Total gold prompts available (logs only): {gold_count}")
                else:
                    self.logger.info(f"   🏆 Total gold prompts available: {gold_count}")
                
                # Show comprehensive scoring summary
                if combined_prompts:
                    scores_list = [p.get('best_score', 0.0) for p in combined_prompts.values()]
                    avg_score = sum(scores_list) / len(scores_list)
                    max_score = max(scores_list)
                    min_score = min(scores_list)
                    
                    self.logger.info(f"   📊 Scoring Summary:")
                    self.logger.info(f"      Average score: {avg_score:.4f}")
                    self.logger.info(f"      Highest score: {max_score:.4f}")
                    self.logger.info(f"      Lowest score: {min_score:.4f}")
                    
                    # Score distribution
                    excellent = len([s for s in scores_list if s >= 0.9])
                    good = len([s for s in scores_list if 0.7 <= s < 0.9])
                    fair = len([s for s in scores_list if 0.5 <= s < 0.7])
                    poor = len([s for s in scores_list if s < 0.5])
                    
                    self.logger.info(f"      Score distribution: {excellent} excellent (≥0.9), {good} good (0.7-0.9), {fair} fair (0.5-0.7), {poor} poor (<0.5)")
                
            else:
                self.logger.debug("📖 Real-time learning disabled, using only episodic memory")
            
            # Update statistics
            self.stats['total_gold_prompts_available'] = len(combined_prompts)
            if self.config.get('only_log_learning', False):
                self.stats['memory_prompts'] = 0  # Bypassed in only-log-learning mode
                self.stats['log_prompts'] = len(log_prompts) if 'log_prompts' in locals() else 0
            else:
                self.stats['memory_prompts'] = len(memory_prompts) if 'memory_prompts' in locals() else 0
                self.stats['log_prompts'] = len(log_prompts) if 'log_prompts' in locals() else 0
            
        except Exception as e:
            self.logger.error(f"❌ Error getting fresh gold prompts: {e}")
            traceback.print_exc()
            # Fallback to just episodic memory
            if self.reproducibility_system:
                combined_prompts = self.reproducibility_system.gold_standard_results
        
        return combined_prompts
    
    def setup_live_episodic_memory_monitoring(self):
        """
        Setup live monitoring of episodic memory file for automatic updates.
        This watches for changes and automatically reloads gold prompts.
        """
        if not self.config.get('activate_learning', False):
            self.logger.debug("📁 Live monitoring disabled (activate_learning=False)")
            return
            
        try:
            # Try to import watchdog for file monitoring
            try:
                from watchdog.observers import Observer
                from watchdog.events import FileSystemEventHandler
                WATCHDOG_AVAILABLE = True
            except ImportError:
                WATCHDOG_AVAILABLE = False
                self.logger.warning("⚠️ watchdog not available - live monitoring disabled")
                return
            
            if not WATCHDOG_AVAILABLE:
                return
            
            class EpisodicMemoryWatcher(FileSystemEventHandler):
                def __init__(self, orchestrator):
                    self.orchestrator = orchestrator
                    self.last_modified = 0
                    self.logger = orchestrator.logger
                
                def on_modified(self, event):
                    if event.src_path.endswith('episodic_memory.json'):
                        current_time = time.time()
                        # Debounce to avoid multiple rapid updates
                        if current_time - self.last_modified > 5:
                            self.last_modified = current_time
                            self.logger.info("🔄 Episodic memory file modified - triggering gold prompts reload!")
                            
                            # Reload gold prompts
                            try:
                                self.orchestrator.reload_gold_prompts()
                                self.logger.info("✅ Gold prompts reloaded from episodic memory")
                            except Exception as e:
                                self.logger.error(f"❌ Failed to reload gold prompts: {e}")
            
            # Setup the file watcher
            self.episodic_memory_observer = Observer()
            self.episodic_memory_watcher = EpisodicMemoryWatcher(self)
            
            # Watch the episodic_logs_first directory
            watch_path = Path("episodic_logs_first")
            if watch_path.exists():
                self.episodic_memory_observer.schedule(
                    self.episodic_memory_watcher, 
                    str(watch_path), 
                    recursive=False
                )
                self.episodic_memory_observer.start()
                self.logger.info("📁 Live episodic memory monitoring ENABLED")
                self.logger.info(f"   Watching directory: {watch_path}")
                self.logger.info("   Gold prompts will auto-reload on memory updates")
            else:
                self.logger.warning("⚠️ Episodic logs directory not found, live monitoring disabled")
                
        except Exception as e:
            self.logger.error(f"❌ Failed to setup live monitoring: {e}")
    
    def stop_live_monitoring(self):
        """Stop the live episodic memory monitoring"""
        if hasattr(self, 'episodic_memory_observer'):
            try:
                self.episodic_memory_observer.stop()
                self.episodic_memory_observer.join()
                self.logger.info("📁 Live episodic memory monitoring stopped")
            except Exception as e:
                self.logger.error(f"❌ Error stopping live monitoring: {e}")
    
    def enhanced_reload_gold_prompts(self):
        """
        Enhanced reload that combines episodic memory with real-time log data.
        This is the main method called when --activate-learning is enabled.
        """
        if not REPRODUCIBILITY_SYSTEM_AVAILABLE or not self.reproducibility_system:
            return
        
        try:
            self.logger.info("🚀 Enhanced gold prompts reload with real-time learning...")
            
            # Get fresh gold prompts (memory + logs)
            fresh_prompts = self.get_fresh_gold_prompts()
            
            # Update the reproducibility system with fresh data
            old_count = len(self.reproducibility_system.gold_standard_results)
            
            # Create a temporary update to the reproducibility system
            # Note: This is a workaround since we can't directly modify the system's data structure
            # In a real implementation, you'd want to modify the reproducibility system to accept updates
            
            # For now, we'll update our local tracking
            self.stats['enhanced_gold_prompts_available'] = len(fresh_prompts)
            self.stats['enhanced_gold_prompts_reloaded'] += 1
            
            # Update timestamp
            self.last_gold_prompts_reload = time.time()
            
            # Log the results
            if len(fresh_prompts) > old_count:
                self.logger.info(f"✅ Enhanced reload: {old_count} → {len(fresh_prompts)} (+{len(fresh_prompts) - old_count})")
            elif len(fresh_prompts) < old_count:
                self.logger.info(f"⚠️ Enhanced reload: {old_count} → {len(fresh_prompts)} (-{old_count - len(fresh_prompts)})")
            else:
                self.logger.info(f"🔄 Enhanced reload: {len(fresh_prompts)} prompts (no change in count)")
            
            # Log source breakdown
            memory_count = self.stats.get('memory_prompts', 0)
            log_count = self.stats.get('log_prompts', 0)
            self.logger.info(f"   📚 From episodic memory: {memory_count}")
            self.logger.info(f"   📖 From recent logs: {log_count}")
            self.logger.info(f"   🔄 Total combined: {len(fresh_prompts)}")
            
            # Log some sample prompts for verification
            if fresh_prompts:
                sample_prompts = list(fresh_prompts.keys())[:3]
                self.logger.info(f"   📝 Sample gold prompts:")
                for i, prompt in enumerate(sample_prompts, 1):
                    prompt_data = fresh_prompts[prompt]
                    source = prompt_data.get('source', 'unknown')
                    score = prompt_data.get('final_score', prompt_data.get('score', 'unknown'))
                    self.logger.info(f"     {i}. '{prompt[:60]}...' (score: {score}, source: {source})")
            
        except Exception as e:
            self.logger.error(f"❌ Enhanced gold prompts reload failed: {e}")
            traceback.print_exc()

    def _enhanced_reproducibility_optimization(self, prompt: str, min_similarity: float, enhanced_gold_prompts: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Enhanced reproducibility optimization using our enhanced gold prompts (memory + logs).
        This method implements the similarity search and optimization logic directly.
        
        Args:
            prompt: The prompt to optimize
            min_similarity: Minimum similarity threshold
            enhanced_gold_prompts: Dictionary of enhanced gold prompts from memory + logs
            
        Returns:
            Optimization result or None if no close match found
        """
        try:
            self.logger.info(f"🔍 Enhanced reproducibility search for: '{prompt}'")
            self.logger.info(f"   Searching through {len(enhanced_gold_prompts)} enhanced gold prompts")
            self.logger.info(f"   Minimum similarity threshold: {min_similarity}")
            
            best_match = None
            best_similarity = 0.0
            candidates = []
            
            # Calculate similarity for all gold prompts
            for gold_prompt, gold_data in enhanced_gold_prompts.items():
                # Calculate similarity using enhanced similarity calculation
                similarity = self._calculate_simple_similarity(prompt, gold_prompt)
                
                # Get the score for this gold prompt
                if 'validation_results' in gold_data and 'validation_engine_score' in gold_data['validation_results']:
                    # Episodic memory format
                    gold_score = gold_data['validation_results']['validation_engine_score']
                    source = 'episodic_memory'
                else:
                    # Log format
                    gold_score = gold_data.get('best_score', 0.0)
                    source = gold_data.get('source', 'recent_logs')
                
                # Store candidate for analysis
                candidates.append({
                    'prompt': gold_prompt,
                    'similarity': similarity,
                    'score': gold_score,
                    'source': source,
                    'data': gold_data
                })
                
                if similarity > best_similarity and similarity >= min_similarity:
                    best_similarity = similarity
                    best_match = {
                        'gold_prompt': gold_prompt,
                        'gold_data': gold_data,
                        'similarity': similarity,
                        'gold_score': gold_score,
                        'source': source
                    }
            
            # Sort all candidates by similarity for comprehensive analysis
            sorted_candidates = sorted(candidates, key=lambda x: x['similarity'], reverse=True)
            
            # Log comprehensive similarity analysis
            self.logger.info(f"🔍 Similarity analysis results:")
            self.logger.info(f"   Total candidates analyzed: {len(sorted_candidates)}")
            self.logger.info(f"   Candidates above threshold ({min_similarity}): {len([c for c in sorted_candidates if c['similarity'] >= min_similarity])}")
            self.logger.info(f"   Candidates below threshold: {len([c for c in sorted_candidates if c['similarity'] < min_similarity])}")
            
            # Show top 10 candidates with their similarities
            self.logger.info(f"   Top 3 similarity candidates:")
            for i, candidate in enumerate(sorted_candidates[:3], 1):
                status = "✅ ABOVE THRESHOLD" if candidate['similarity'] >= min_similarity else "❌ Below threshold"
                self.logger.info(f"     {i:2d}. Sim: {candidate['similarity']:.3f} | Score: {candidate['score']:.4f} | {status}")
                self.logger.info(f"         Source: {candidate['source']} | Prompt: '{candidate['prompt'][:60]}...'")
            
            # Show threshold analysis
            if sorted_candidates:
                max_similarity = sorted_candidates[0]['similarity']
                min_similarity_found = sorted_candidates[-1]['similarity']
                avg_similarity = sum(c['similarity'] for c in sorted_candidates) / len(sorted_candidates)
                
                self.logger.info(f"   Similarity statistics:")
                self.logger.info(f"     Maximum similarity: {max_similarity:.3f}")
                self.logger.info(f"     Minimum similarity: {min_similarity_found:.3f}")
                self.logger.info(f"     Average similarity: {avg_similarity:.3f}")
                self.logger.info(f"     Threshold: {min_similarity:.3f}")
                
                if max_similarity < min_similarity:
                    self.logger.warning(f"   ⚠️ WARNING: No prompt meets the similarity threshold!")
                    self.logger.warning(f"      Consider lowering the threshold from {min_similarity:.3f} to {max_similarity:.3f} or lower")
            
            if best_match:
                self.logger.info(f"🏆 Found close gold prompt (similarity: {best_similarity:.3f})")
                self.logger.info(f"   Gold prompt: '{best_match['gold_prompt'][:50]}...'")
                self.logger.info(f"   Gold score: {best_match['gold_score']:.4f}")
                self.logger.info(f"   Source: {best_match['source']}")
                
                # Extract the optimized prompt from gold data
                if 'validation_results' in best_match['gold_data'] and 'method_2_hybrid_example' in best_match['gold_data']:
                    # Episodic memory format
                    optimized_prompt = best_match['gold_data']['method_2_hybrid_example']['optimized_prompt']
                else:
                    # Log format - use original for now (could be enhanced to extract actual optimized version)
                    optimized_prompt = best_match['gold_prompt']
                
                return {
                    'optimized_prompt': optimized_prompt,
                    'similarity': best_similarity,
                    'gold_score': best_match['gold_score'],
                    'gold_prompt': best_match['gold_prompt'],
                    'source': best_match['source'],
                    'method': 'enhanced_reproducibility'
                }
            else:
                self.logger.warning(f"❌ No gold prompt found with similarity ≥ {min_similarity}")
                
                # Suggest potential threshold adjustments
                if sorted_candidates:
                    top_similarities = [c['similarity'] for c in sorted_candidates[:5]]
                    suggested_threshold = max(top_similarities) - 0.05  # 0.05 below the highest
                    self.logger.info(f"   💡 Suggestion: Try lowering threshold to {suggested_threshold:.3f} to include top candidates")
                    self.logger.info(f"      Top similarities found: {[f'{s:.3f}' for s in top_similarities]}")
                
                return None
                
        except Exception as e:
            self.logger.error(f"❌ Enhanced reproducibility optimization failed: {e}")
            traceback.print_exc()
            return None
    
    def _calculate_simple_similarity(self, prompt1: str, prompt2: str) -> float:
        """
        Calculate similarity between two prompts using multiple similarity metrics.
        This is an enhanced implementation that combines different similarity approaches.
        
        Args:
            prompt1: First prompt
            prompt2: Second prompt
            
        Returns:
            Similarity score between 0.0 and 1.0
        """
        try:
            # Convert to lowercase for comparison
            p1_lower = prompt1.lower()
            p2_lower = prompt2.lower()
            
            # Method 1: Jaccard similarity (word overlap)
            words1 = set(p1_lower.split())
            words2 = set(p2_lower.split())
            
            if not words1 or not words2:
                jaccard_sim = 0.0
            else:
                intersection = len(words1.intersection(words2))
                union = len(words1.union(words2))
                jaccard_sim = intersection / union if union > 0 else 0.0
            
            # Method 2: Sequence similarity (character-level)
            from difflib import SequenceMatcher
            sequence_sim = SequenceMatcher(None, p1_lower, p2_lower).ratio()
            
            # Method 3: Common word ratio
            common_words = words1.intersection(words2)
            total_unique_words = len(words1.union(words2))
            word_ratio = len(common_words) / total_unique_words if total_unique_words > 0 else 0.0
            
            # Method 4: Theme-based similarity boost
            theme_boost = 0.0
            
            # Food/fruit themes
            food_themes = ['fruit', 'pear', 'apple', 'banana', 'orange', 'grape', 'food', 'edible', 'fresh', 'juicy', 'ripe']
            if any(word in p1_lower for word in food_themes) and any(word in p2_lower for word in food_themes):
                theme_boost += 0.15
            
            # Object/material themes
            object_themes = ['crystal', 'gem', 'jewel', 'stone', 'metal', 'wood', 'glass', 'ceramic', 'fabric']
            if any(word in p1_lower for word in object_themes) and any(word in p2_lower for word in object_themes):
                theme_boost += 0.15
            
            # Size/shape themes
            size_themes = ['small', 'large', 'tiny', 'huge', 'round', 'square', 'oval', 'rectangular', 'spherical']
            if any(word in p1_lower for word in size_themes) and any(word in p2_lower for word in size_themes):
                theme_boost += 0.1
            
            # Color themes
            color_themes = ['red', 'blue', 'green', 'yellow', 'purple', 'orange', 'pink', 'brown', 'black', 'white', 'gold', 'silver']
            if any(word in p1_lower for word in color_themes) and any(word in p2_lower for word in color_themes):
                theme_boost += 0.1
            
            # Method 5: Length similarity (shorter prompts are more similar to each other)
            len1, len2 = len(p1_lower), len(p2_lower)
            max_len = max(len1, len2)
            min_len = min(len1, len2)
            length_sim = min_len / max_len if max_len > 0 else 0.0
            
            # Combine all similarity metrics with weights
            weighted_sim = (
                jaccard_sim * 0.4 +      # Word overlap (40%)
                sequence_sim * 0.3 +     # Character sequence (30%)
                word_ratio * 0.2 +       # Common word ratio (20%)
                length_sim * 0.1 +       # Length similarity (10%)
                theme_boost               # Theme boost (bonus)
            )
            
            # Ensure final similarity is between 0.0 and 1.0
            final_similarity = max(0.0, min(1.0, weighted_sim))
            
            # Debug logging for similarity calculation
            # if final_similarity > 0.1:  # Only log if there's some similarity
            #     print(f"🔍 Similarity calculation for '{prompt1[:30]}...' vs '{prompt2[:30]}...':")
            #     print(f"   Jaccard: {jaccard_sim:.3f}, Sequence: {sequence_sim:.3f}, Word ratio: {word_ratio:.3f}")
            #     print(f"   Length: {length_sim:.3f}, Theme boost: {theme_boost:.3f}")
            #     print(f"   Final: {final_similarity:.3f}")
            
            return final_similarity
            
        except Exception as e:
            print(f"❌ Error calculating similarity: {e}")
            return 0.0

async def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Continuous TRELLIS Orchestrator")
    parser.add_argument("--no-harvest", action="store_true", help="Disable task harvesting")
    parser.add_argument("--no-validate", action="store_true", help="Disable validation")
    parser.add_argument("--no-submit", action="store_true", help="Disable result submission")
    parser.add_argument("--generation-server", default="http://localhost:8097", help="TRELLIS generation server URL")
    parser.add_argument("--validation-server", default="http://localhost:10006", help="Validation server URL")
    parser.add_argument("--output-dir", default="./continuous_trellis_outputs", help="Output directory")
    parser.add_argument("--min-score", type=float, default=0.3, help="Minimum local validation score")
    
    # Prompt optimization arguments
    parser.add_argument("--no-optimize", action="store_true", help="Disable prompt optimization")
    parser.add_argument("--aggressive-optimize", action="store_true", help="Enable aggressive optimization mode")
    parser.add_argument("--quiet-optimize", action="store_true", help="Reduce optimization logging detail")
    parser.add_argument("--no-prompt-cleaning", action="store_true", help="Disable automatic prompt cleaning (removes artifacts like 'wbgmsst')")
    
    # Ollama configuration
    parser.add_argument("--ollama-url", default="http://localhost:11434", help="URL for the Ollama API server")
    
    # vLLM configuration
    parser.add_argument("--vllm", action="store_true", help="Use vLLM instead of Ollama")
    parser.add_argument("--vllm-url", default="http://localhost:9000", help="URL for the vLLM server")
    parser.add_argument("--vllm-model", default="llama-3-2-3b-it", help="vLLM model name")
    
    # Reproducibility optimization arguments
    parser.add_argument("--no-reproducibility", action="store_true", help="Disable reproducibility optimization")
    parser.add_argument("--reproducibility-similarity", type=float, default=0.6, help="Minimum similarity threshold for reproducibility (default: 0.3)")
    
    # LoRA routing arguments
    parser.add_argument("--no-lora-routing", action="store_true", help="Disable intelligent LoRA routing")
    parser.add_argument("--lora-confidence-threshold", type=float, default=0.5, help="Minimum confidence threshold for LoRA routing (default: 0.5)")
    
    # Determinism arguments
    parser.add_argument("--variable-seeds", action="store_true", help="Use prompt-hash based seeds (default: fixed seed 42)")
    parser.add_argument("--seed", type=int, default=42, help="Fixed seed to use when not using variable seeds")
    
    # Validator blacklisting arguments  
    parser.add_argument("--blacklist", type=int, nargs="*", default=[180, 253], help="Validator UIDs to blacklist (default: [180])")
    parser.add_argument("--no-blacklist", action="store_true", help="Disable validator blacklisting")
    
    # Gold prompts reload arguments
    parser.add_argument("--gold-prompts-reload-interval", type=int, default=120, help="Reload gold prompts every N seconds (default: 3600 = 1 hour)")
    
    # Real-time learning arguments
    parser.add_argument("--activate-learning", action="store_true", help="Enable real-time learning from episode logs and live episodic memory monitoring")
    parser.add_argument("--only-log-learning", nargs='?', const=6, type=int, metavar='N', 
                       help="Use only N most recent logs for learning, bypass episodic memory (default: 6, use -1 for all logs, requires --activate-learning)")
    
    # Shared task tracking arguments
    parser.add_argument("--enable-task-tracking", action="store_true", help="Enable shared task tracking to prevent duplicate processing across instances")
    parser.add_argument("--disable-task-tracking", action="store_true", help="Disable shared task tracking (default: enabled)")
    
    # Duplicate checking arguments
    parser.add_argument("--no-skip-duplicates", action="store_true", help="Disable duplicate prompt checking (default: enabled)")
    
    # Cooldown arguments
    parser.add_argument("--network-error-cooldown", type=int, default=30, help="Cooldown duration after network errors (seconds, default: 30)")
    parser.add_argument("--submission-failure-cooldown", type=int, default=60, help="Cooldown duration after submission failures (seconds, default: 60)")
    parser.add_argument("--validator-error-cooldown", type=int, default=45, help="Cooldown duration after validator errors (seconds, default: 45)")
    parser.add_argument("--max-cooldown-duration", type=int, default=300, help="Maximum cooldown duration (seconds, default: 300)")
    parser.add_argument("--no-cooldown-logging", action="store_true", help="Disable detailed cooldown logging")
    
    # Enhanced cooldown system arguments
    parser.add_argument("--cooldown-violation-threshold", type=int, default=5, help="Number of violations before applying penalty (default: 5)")
    parser.add_argument("--cooldown-violation-penalty", type=int, default=60, help="Additional penalty cooldown in seconds (default: 60)")
    parser.add_argument("--validation-lock-duration", type=int, default=30, help="Default validation lock duration in seconds (default: 30)")
    
    # Emergency cooldown management arguments
    parser.add_argument("--emergency-cooldown-buffer", type=int, default=30, help="Buffer seconds added to validator cooldowns (default: 30)")
    parser.add_argument("--critical-violation-threshold", type=int, default=100, help="Violation count that triggers emergency measures (default: 100)")
    parser.add_argument("--critical-violation-cooldown", type=int, default=3600, help="Emergency cooldown duration for critical violations (default: 3600)")
    parser.add_argument("--base-blacklist-duration", type=int, default=1800, help="Base duration for temporary blacklisting (default: 1800)")
    
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
    if args.no_prompt_cleaning:
        config['enable_prompt_cleaning'] = False
    
    # Ollama configuration[]
    config['ollama_url'] = args.ollama_url
    
    # vLLM configuration
    config['use_vllm'] = args.vllm
    config['vllm_url'] = args.vllm_url
    config['vllm_model'] = args.vllm_model
    
    # Reproducibility optimization configuration
    if args.no_reproducibility:
        config['enable_reproducibility_optimization'] = False
    config['reproducibility_min_similarity'] = args.reproducibility_similarity
    
    # LoRA routing configuration
    if args.no_lora_routing:
        config['enable_lora_routing'] = False
    config['lora_routing_confidence_threshold'] = args.lora_confidence_threshold
    
    # Determinism configuration
    if args.variable_seeds:
        config['use_fixed_seed'] = False
    config['fixed_seed_value'] = args.seed
    
    # Validator blacklisting configuration
    if args.no_blacklist:
        config['enable_validator_blacklisting'] = False
    if args.blacklist is not None:
        config['validator_blacklist'] = args.blacklist
    
    # Gold prompts reload configuration
    config['gold_prompts_reload_interval'] = args.gold_prompts_reload_interval
    
    # Real-time learning configuration
    config['activate_learning'] = args.activate_learning
    config['only_log_learning'] = args.only_log_learning
    
    # Validate only-log-learning requires activate-learning
    if args.only_log_learning is not None and not args.activate_learning:
        print("❌ Error: --only-log-learning requires --activate-learning to be enabled")
        exit(1)
    
    # Set default log learning count if not specified
    if args.only_log_learning is None:
        config['log_learning_count'] = 6  # Default to 6 logs
    else:
        config['log_learning_count'] = args.only_log_learning
    
    # Shared task tracking configuration
    if args.enable_task_tracking:
        config['enable_task_tracking'] = True
        config['disable_task_tracking'] = False
    elif args.disable_task_tracking:
        config['enable_task_tracking'] = False
        config['disable_task_tracking'] = True
    else:
        # Default: enable task tracking
        config['enable_task_tracking'] = True
        config['disable_task_tracking'] = False
    
    # Duplicate checking configuration
    if args.no_skip_duplicates:
        config['enable_duplicate_checking'] = False
        print("⚠️ Duplicate checking DISABLED - will process all prompts including duplicates")
    else:
        config['enable_duplicate_checking'] = True
        print("✅ Duplicate checking ENABLED - will skip previously processed prompts")
    
    # Cooldown configuration
    config['network_error_cooldown'] = args.network_error_cooldown
    config['submission_failure_cooldown'] = args.submission_failure_cooldown
    config['validator_error_cooldown'] = args.validator_error_cooldown
    config['max_cooldown_duration'] = args.max_cooldown_duration
    config['enable_cooldown_logging'] = not args.no_cooldown_logging
    
    # Enhanced cooldown system configuration
    config['cooldown_violation_threshold'] = args.cooldown_violation_threshold
    config['cooldown_violation_penalty'] = args.cooldown_violation_penalty
    config['validation_lock_duration'] = args.validation_lock_duration
    
    # Emergency cooldown management configuration
    config['emergency_cooldown_buffer'] = args.emergency_cooldown_buffer
    config['critical_violation_threshold'] = args.critical_violation_threshold
    config['critical_violation_cooldown'] = args.critical_violation_cooldown
    config['base_blacklist_duration'] = args.base_blacklist_duration
    
    print(f"⏳ Cooldown settings: Network errors: {args.network_error_cooldown}s, Submission failures: {args.submission_failure_cooldown}s, Validator errors: {args.validator_error_cooldown}s, Max: {args.max_cooldown_duration}s")
    print(f"📝 Cooldown logging: {'ENABLED' if not args.no_cooldown_logging else 'DISABLED'}")
    print(f"🚨 Enhanced cooldown: Violation threshold: {args.cooldown_violation_threshold}, Penalty: {args.cooldown_violation_penalty}s, Validation lock: {args.validation_lock_duration}s")
    print(f"🚨 Emergency cooldown: Buffer: {args.emergency_cooldown_buffer}s, Critical threshold: {args.critical_violation_threshold}, Blacklist base: {args.base_blacklist_duration}s")
    
    # Create and run orchestrator
    orchestrator = ContinuousTrellisOrchestrator(config)
    
    try:
        await orchestrator.continuous_mining_loop()
    except Exception as e:
        logger.error(f"❌ Orchestrator failed: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(main()) 
