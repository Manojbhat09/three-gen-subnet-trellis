#!/usr/bin/env python3
"""
Continuous TRELLIS Orchestrator - Subnet 17 (404-GEN) with CLIP Optimization
Purpose: Continuous mining with intelligent task deduplication, idle validation, and CLIP-based prompt optimization

Features:
- Continuous task harvesting with prompt deduplication
- Real-time feedback processing and score tracking
- Automatic validation during idle periods
- Comprehensive statistics and JSON logging
- Always-on generation server integration
- PRIORITY-BASED server coordination for time-critical tasks
- CLIP-based episodic memory optimization for prompts
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
    print("⚠️ Using legacy prompt optimizer")

# Import reproducibility system
try:
    # from smart_reproducibility_system_v2 import SmartReproducibilitySystem
    from smart_reproducibility_system_v7_f1 import SmartReproducibilitySystem
    REPRODUCIBILITY_SYSTEM_AVAILABLE = True
    print("✅ Smart reproducibility system available")
except ImportError:
    REPRODUCIBILITY_SYSTEM_AVAILABLE = False
    print("⚠️ Reproducibility system not available")

# Bittensor imports
try:
    import bittensor as bt
    BITTENSOR_AVAILABLE = True
    print("✅ Bittensor imported successfully")
except ImportError:
    BITTENSOR_AVAILABLE = False
    print("⚠️ Bittensor not available - running in development mode")

# Setup logging with both file and console handlers
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('continuous_trellis_hunyuan_clip.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("ContinuousTrellisOrchestrator")

class PriorityServerCoordinator:
    """
    Coordinates priority access to the TRELLIS server.
    Ensures subnet mining tasks get priority over ongoing optimization tasks.
    """
    
    def __init__(self, server_url: str = "http://localhost:8096", 
                 max_wait_time_seconds: int = 60,
                 status_check_interval: int = 1,
                 priority_timeout: int = 30,
                 on_interruption_callback=None):
        self.server_url = server_url
        self.max_wait_time_seconds = max_wait_time_seconds
        self.status_check_interval = status_check_interval
        self.priority_timeout = priority_timeout
        self.on_interruption_callback = on_interruption_callback
        
        # Track current priority job
        self.current_priority_job = None
        self.priority_job_start_time = None
        
        # Track server status
        self.last_server_check = 0
        self.server_status_cache = None
        self.cache_duration = 2  # Cache status for 2 seconds
        
        self.logger = logging.getLogger("PriorityCoordinator")
        
    def check_server_status(self) -> Dict[str, Any]:
        """Check server status with caching to avoid spam"""
        current_time = time.time()
        
        # Use cached status if recent
        if (self.server_status_cache and 
            current_time - self.last_server_check < self.cache_duration):
            return self.server_status_cache
        
        try:
            response = requests.get(f"{self.server_url}/job/status/", timeout=5)
            if response.status_code == 200:
                status = response.json()
                self.server_status_cache = status
                self.last_server_check = current_time
                return status
            else:
                self.logger.debug(f"Server status check failed: {response.status_code}")
                return {"is_busy": False, "current_job": None}
        except Exception as e:
            self.logger.debug(f"Server status check exception: {e}")
            return {"is_busy": False, "current_job": None}
    
    def _interrupt_current_job(self, current_job_info: Dict) -> bool:
        """Interrupt the current job if it's an optimizer job"""
        try:
            # Check if current job is from optimizer by examining the prompt pattern
            current_prompt = current_job_info.get("prompt", "")
            current_job_id = current_job_info.get("job_id", "")
            
            # Patterns that indicate optimizer/testing jobs vs real subnet tasks
            optimizer_patterns = [
                "cinematic",
                "dramatic lighting", 
                "hyperrealistic",
                "ultra-detailed",
                "optimization test",
                "test prompt",
                "8k resolution",
                "photorealistic",
                "studio lighting"
            ]
            
            # Check if this looks like an optimizer job
            prompt_lower = current_prompt.lower()
            is_optimizer_job = any(pattern in prompt_lower for pattern in optimizer_patterns)
            
            # Also check for very long prompts (typical of optimizers)
            is_long_prompt = len(current_prompt) > 200
            
            if is_optimizer_job or is_long_prompt:
                self.logger.info(f"🛑 Interrupting optimizer job: '{current_prompt[:50]}...'")
                
                # Reset the server by posting to reset endpoint
                reset_response = requests.post(f"{self.server_url}/job/reset/", timeout=10)
                if reset_response.status_code == 200:
                    self.logger.info(f"✅ Server reset successful, optimizer job interrupted")
                    
                    # Call interruption callback if provided
                    if self.on_interruption_callback:
                        try:
                            self.on_interruption_callback()
                        except Exception as e:
                            self.logger.warning(f"⚠️ Interruption callback failed: {e}")
                    
                    # Wait a moment for reset to take effect
                    time.sleep(2)
                    return True
                else:
                    self.logger.warning(f"⚠️ Server reset failed: {reset_response.status_code}")
                    return False
            else:
                self.logger.info(f"🤚 Current job appears to be a subnet task, will wait instead of interrupting")
                return False
                
        except Exception as e:
            self.logger.error(f"❌ Failed to interrupt current job: {e}")
            return False
    
    def _is_our_job(self, job_id: str, prompt: str) -> bool:
        """Check if the current job belongs to our priority session"""
        if not self.current_priority_job:
            return False
            
        # Simple check - could be enhanced with more sophisticated tracking
        our_job_id = self.current_priority_job.get("job_id")
        our_prompt = self.current_priority_job.get("prompt")
        
        return (job_id == our_job_id or 
                (prompt and our_prompt and prompt == our_prompt))
    
    def wait_for_priority_access(self, task_id: str = None) -> bool:
        """
        Wait for priority access to the server.
        Returns True if access granted, False if timeout.
        """
        wait_start = time.time()
        
        self.logger.info(f"🔄 Requesting priority access to TRELLIS server...")
        
        while time.time() - wait_start < self.max_wait_time_seconds:
            status = self.check_server_status()
            
            if not status.get("is_busy", False):
                self.logger.info(f"✅ Server available for priority access")
                return True
            
            # Server is busy - check if we should interrupt
            current_job = status.get("current_job", {})
            job_id = current_job.get("job_id", "")
            prompt = current_job.get("prompt", "")
            
            # If it's our job, we already have access
            if self._is_our_job(job_id, prompt):
                self.logger.info(f"✅ Server already running our priority job")
                return True
            
            # Check if we should interrupt the current job
            if self._interrupt_current_job(current_job):
                # Wait a moment after interruption, then check again
                time.sleep(3)
                continue
            
            # Wait before checking again
            self.logger.debug(f"⏳ Waiting for server (busy with: '{prompt[:30]}...')")
            time.sleep(self.status_check_interval)
        
        # Timeout reached
        elapsed = time.time() - wait_start
        self.logger.error(f"❌ PRIORITY ACCESS TIMEOUT after {elapsed:.1f}s - server remained busy")
        return False
    
    def _force_clear_server(self):
        """Force clear server cache and reset state"""
        try:
            self.logger.info(f"🧹 Force clearing server state...")
            
            # Try cache clear first
            cache_response = requests.post(f"{self.server_url}/clear_cache/", timeout=15)
            if cache_response.status_code == 200:
                self.logger.info(f"✅ Server cache cleared")
            else:
                self.logger.warning(f"⚠️ Cache clear failed: {cache_response.status_code}")
            
            # Then reset job status
            reset_response = requests.post(f"{self.server_url}/job/reset/", timeout=10)
            if reset_response.status_code == 200:
                self.logger.info(f"✅ Server job status reset")
            else:
                self.logger.warning(f"⚠️ Job reset failed: {reset_response.status_code}")
            
            # Wait for operations to complete
            time.sleep(3)
            
        except Exception as e:
            self.logger.error(f"❌ Force clear failed: {e}")
    
    def clear_server_cache(self) -> bool:
        """Clear server cache to ensure clean state"""
        try:
            self.logger.debug(f"🧹 Clearing server cache...")
            response = requests.post(f"{self.server_url}/clear_cache/", timeout=15)
            if response.status_code == 200:
                self.logger.debug(f"✅ Cache cleared successfully")
                return True
            else:
                self.logger.warning(f"⚠️ Cache clear failed: {response.status_code}")
                return False
        except Exception as e:
            self.logger.error(f"❌ Cache clear exception: {e}")
            return False
    
    def mark_priority_job_start(self, task_id: str, prompt: str):
        """Mark the start of a priority job"""
        self.current_priority_job = {
            "job_id": task_id,
            "prompt": prompt,
            "start_time": time.time()
        }
        self.priority_job_start_time = time.time()
        self.logger.debug(f"🎯 Priority job started: {task_id}")
    
    def mark_priority_job_end(self, task_id: str):
        """Mark the end of a priority job"""
        if self.current_priority_job and self.current_priority_job.get("job_id") == task_id:
            duration = time.time() - self.priority_job_start_time if self.priority_job_start_time else 0
            self.logger.debug(f"✅ Priority job completed: {task_id} (duration: {duration:.1f}s)")
            self.current_priority_job = None
            self.priority_job_start_time = None
        else:
            self.logger.debug(f"⚠️ Job end called for unknown task: {task_id}")

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
    
    # CLIP optimization tracking
    optimized_prompt: Optional[str] = None
    optimization_method: Optional[str] = None
    optimization_similarity_score: Optional[float] = None
    reproducibility_references_found: Optional[int] = None
    reproducibility_similarity_scores: Optional[List[float]] = None
    clip_optimization_method: Optional[str] = None  # "exact_match", "similar_pattern", "llm_examples"
    similar_prompt_found: Optional[str] = None
    optimization_similarity: Optional[float] = None
    pattern_analysis_used: Optional[bool] = None
    llm_optimization_attempts: Optional[int] = None

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
    """Database for storing task records and statistics"""
    
    def __init__(self, db_path: str = "continuous_trellis_hunyuan_clip_tasks.db"):
        self.db_path = db_path
        self.init_database()
    
    def init_database(self):
        """Initialize database tables"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Tasks table with all fields from TaskRecord
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
                    priority_access_timeout BOOLEAN DEFAULT FALSE,
                    optimized_prompt TEXT,
                    optimization_method TEXT,
                    optimization_similarity_score REAL,
                    reproducibility_references_found INTEGER,
                    reproducibility_similarity_scores TEXT,
                    clip_optimization_method TEXT,
                    similar_prompt_found TEXT,
                    optimization_similarity REAL,
                    pattern_analysis_used BOOLEAN,
                    llm_optimization_attempts INTEGER,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # Recent prompts table for deduplication
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS recent_prompts (
                    prompt_hash TEXT NOT NULL,
                    validator_uid INTEGER NOT NULL,
                    prompt TEXT NOT NULL,
                    added_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    PRIMARY KEY (prompt_hash, validator_uid)
                )
            ''')
            
            # Add indexes for better performance
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_tasks_validator ON tasks(validator_uid)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_tasks_pulled_at ON tasks(pulled_at)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_tasks_processed_at ON tasks(processed_at)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_tasks_prompt_hash ON tasks(prompt_hash)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_recent_prompts_added_at ON recent_prompts(added_at)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_tasks_fidelity_score ON tasks(task_fidelity_score)')
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.error(f"❌ Database initialization failed: {e}")
            raise

    def is_duplicate_prompt(self, prompt: str, validator_uid: int, hours_window: int = 24) -> bool:
        """Check if prompt is a duplicate within the time window"""
        try:
            prompt_hash = hashlib.sha256(prompt.encode()).hexdigest()
            
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cutoff_time = datetime.now() - timedelta(hours=hours_window)
            
            cursor.execute('''
                SELECT COUNT(*) FROM recent_prompts 
                WHERE prompt_hash = ? AND validator_uid = ? AND added_at > ?
            ''', (prompt_hash, validator_uid, cutoff_time))
            
            count = cursor.fetchone()[0]
            conn.close()
            
            return count > 0
            
        except Exception as e:
            logger.error(f"❌ Error checking duplicate prompt: {e}")
            return False

    def add_recent_prompt(self, prompt: str, validator_uid: int):
        """Add prompt to recent prompts tracking"""
        try:
            prompt_hash = hashlib.sha256(prompt.encode()).hexdigest()
            
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT OR REPLACE INTO recent_prompts (prompt_hash, validator_uid, prompt)
                VALUES (?, ?, ?)
            ''', (prompt_hash, validator_uid, prompt))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.error(f"❌ Error adding recent prompt: {e}")

    def save_task(self, task: TaskRecord):
        """Save task record to database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Convert lists to JSON strings for storage
            repro_scores_json = json.dumps(task.reproducibility_similarity_scores) if task.reproducibility_similarity_scores else None
            
            cursor.execute('''
                INSERT OR REPLACE INTO tasks (
                    task_id, prompt, prompt_hash, validator_uid, validator_hotkey,
                    validator_stake, validation_threshold, pulled_at, processed_at,
                    submitted_at, generation_time, validation_time, total_processing_time,
                    local_validation_score, submission_success, feedback_received,
                    task_fidelity_score, average_fidelity_score, current_miner_reward,
                    validation_failed, generations_in_window, ply_file_path,
                    compressed_file_path, priority_access_timeout, optimized_prompt,
                    optimization_method, optimization_similarity_score,
                    reproducibility_references_found, reproducibility_similarity_scores,
                    clip_optimization_method, similar_prompt_found, optimization_similarity,
                    pattern_analysis_used, llm_optimization_attempts
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                task.task_id, task.prompt, task.prompt_hash, task.validator_uid,
                task.validator_hotkey, task.validator_stake, task.validation_threshold,
                task.pulled_at, task.processed_at, task.submitted_at, task.generation_time,
                task.validation_time, task.total_processing_time, task.local_validation_score,
                task.submission_success, task.feedback_received, task.task_fidelity_score,
                task.average_fidelity_score, task.current_miner_reward, task.validation_failed,
                task.generations_in_window, task.ply_file_path, task.compressed_file_path,
                task.priority_access_timeout, task.optimized_prompt, task.optimization_method,
                task.optimization_similarity_score, task.reproducibility_references_found,
                repro_scores_json, task.clip_optimization_method, task.similar_prompt_found,
                task.optimization_similarity, task.pattern_analysis_used, task.llm_optimization_attempts
            ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.error(f"❌ Error saving task: {e}")

    def get_recent_unvalidated_tasks(self, hours: int = 2) -> List[TaskRecord]:
        """Get recent tasks that haven't been validated yet"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cutoff_time = time.time() - (hours * 3600)
            
            cursor.execute('''
                SELECT * FROM tasks 
                WHERE processed_at > ? AND local_validation_score IS NULL
                ORDER BY processed_at DESC
            ''', (cutoff_time,))
            
            tasks = []
            for row in cursor.fetchall():
                # Convert row to TaskRecord
                task_data = dict(zip([col[0] for col in cursor.description], row))
                
                # Convert JSON fields back to lists
                if task_data['reproducibility_similarity_scores']:
                    task_data['reproducibility_similarity_scores'] = json.loads(task_data['reproducibility_similarity_scores'])
                
                # Remove the created_at field as it's not in TaskRecord
                task_data.pop('created_at', None)
                
                task = TaskRecord(**task_data)
                tasks.append(task)
            
            conn.close()
            return tasks
            
        except Exception as e:
            logger.error(f"❌ Error getting unvalidated tasks: {e}")
            return []

    def get_unfinished_tasks(self, hours: int = 24) -> List[TaskRecord]:
        """Get tasks that were pulled but never completed"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cutoff_time = time.time() - (hours * 3600)
            
            cursor.execute('''
                SELECT * FROM tasks 
                WHERE pulled_at > ? AND (processed_at IS NULL OR submission_success = FALSE)
                ORDER BY pulled_at DESC
            ''', (cutoff_time,))
            
            tasks = []
            for row in cursor.fetchall():
                # Convert row to TaskRecord
                task_data = dict(zip([col[0] for col in cursor.description], row))
                
                # Convert JSON fields back to lists
                if task_data['reproducibility_similarity_scores']:
                    task_data['reproducibility_similarity_scores'] = json.loads(task_data['reproducibility_similarity_scores'])
                
                # Remove the created_at field
                task_data.pop('created_at', None)
                
                task = TaskRecord(**task_data)
                tasks.append(task)
            
            conn.close()
            return tasks
            
        except Exception as e:
            logger.error(f"❌ Error getting unfinished tasks: {e}")
            return []

    def get_duplicate_analysis(self, validator_uid: int, hours: int = 24) -> Dict[str, Any]:
        """Get duplicate prompt analysis for a validator"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cutoff_time = datetime.now() - timedelta(hours=hours)
            
            # Get total prompts and unique prompts
            cursor.execute('''
                SELECT COUNT(*) as total_prompts, COUNT(DISTINCT prompt_hash) as unique_prompts
                FROM recent_prompts 
                WHERE validator_uid = ? AND added_at > ?
            ''', (validator_uid, cutoff_time))
            
            result = cursor.fetchone()
            total_prompts = result[0] if result else 0
            unique_prompts = result[1] if result else 0
            
            # Get duplicate prompts with their counts
            cursor.execute('''
                SELECT prompt, COUNT(*) as count
                FROM recent_prompts 
                WHERE validator_uid = ? AND added_at > ?
                GROUP BY prompt_hash
                HAVING COUNT(*) > 1
                ORDER BY count DESC
                LIMIT 10
            ''', (validator_uid, cutoff_time))
            
            duplicates = cursor.fetchall()
            
            conn.close()
            
            return {
                'total_prompts': total_prompts,
                'unique_prompts': unique_prompts,
                'duplicate_count': total_prompts - unique_prompts,
                'duplicate_rate': (total_prompts - unique_prompts) / total_prompts if total_prompts > 0 else 0,
                'top_duplicates': [{'prompt': prompt, 'count': count} for prompt, count in duplicates]
            }
            
        except Exception as e:
            logger.error(f"❌ Error getting duplicate analysis: {e}")
            return {
                'total_prompts': 0,
                'unique_prompts': 0,
                'duplicate_count': 0,
                'duplicate_rate': 0,
                'top_duplicates': []
            }

    def has_processed_prompt(self, prompt_hash: str) -> bool:
        """Check if we've already processed this exact prompt hash"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT COUNT(*) FROM tasks 
                WHERE prompt_hash = ? AND processed_at IS NOT NULL
            ''', (prompt_hash,))
            
            count = cursor.fetchone()[0]
            conn.close()
            
            return count > 0
            
        except Exception as e:
            logger.error(f"❌ Error checking processed prompt: {e}")
            return False

    def cleanup_old_prompts(self, days: int = 7):
        """Clean up old prompt records"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cutoff_time = datetime.now() - timedelta(days=days)
            
            cursor.execute('''
                DELETE FROM recent_prompts WHERE added_at < ?
            ''', (cutoff_time,))
            
            deleted_count = cursor.rowcount
            
            conn.commit()
            conn.close()
            
            if deleted_count > 0:
                logger.info(f"🧹 Cleaned up {deleted_count} old prompt records")
                
        except Exception as e:
            logger.error(f"❌ Error cleaning up old prompts: {e}")

class ContinuousTrellisOrchestrator:
    """Main orchestrator for continuous TRELLIS mining with CLIP optimization"""
    
    def __init__(self, config: Dict[str, Any]):
        # Merge with default config
        self.config = {**self._get_default_config(), **config}
        
        # Set up logging
        self.logger = logger
        self.logger.setLevel(logging.DEBUG if self.config.get('debug_mode', False) else logging.INFO)
        
        # Create output directory
        self.output_dir = Path(self.config['output_dir'])
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize database
        self.db = TaskDatabase()
        
        # Initialize statistics
        self.stats = {
            'tasks_pulled': 0,
            'tasks_processed': 0,
            'duplicate_prompts_skipped': 0,
            'successful_generations': 0,
            'failed_generations': 0,
            'successful_validations': 0,
            'failed_validations': 0,
            'successful_submissions': 0,
            'failed_submissions': 0,
            'total_generation_time': 0.0,
            'total_validation_time': 0.0,
            'total_submission_time': 0.0,
            'prompts_optimized': 0,
            'zero_score_generations': 0,
            'mining_interruptions': 0,
            'priority_access_timeouts': 0,
            # CLIP optimization stats
            'clip_exact_matches': 0,
            'clip_similar_patterns': 0,
            'clip_llm_examples': 0,
            'reproducibility_references_found': 0,
            'clip_memory_hits': 0,
            'clip_optimization_improvements': 0,
            # Traditional optimization fallback stats
            'traditional_optimizations': 0,
            'reproducibility_optimizations': 0,
            'clip_optimizations': 0
        }
        
        # Initialize validator tracking
        self.validators: Dict[int, ValidatorState] = {}
        self.last_validator_refresh = 0
        
        # Track active mining state
        self.is_mining = False
        self.last_task_pull = 0
        self.consecutive_failures = 0
        self.last_statistics_save = 0
        self.last_status_print = 0
        
        # Initialize priority coordinator for server access
        self.priority_coordinator = PriorityServerCoordinator(
            server_url=self.config['generation_server_url'],
            max_wait_time_seconds=self.config.get('priority_max_wait_seconds', 60),
            priority_timeout=self.config.get('priority_timeout_seconds', 30),
            on_interruption_callback=self._on_priority_interruption
        )
        
        # Simulation mode
        self.simulation_mode = config.get('simulation_mode', False)
        self.simulation_prompts = []
        self.simulation_index = 0
        
        if self.simulation_mode:
            self.logger.info("🎯 Running in SIMULATION mode")
            if config.get('promptfile'):
                self.simulation_prompts = self.load_prompts_from_file(config['promptfile'])
        
        # Initialize Bittensor components if available
        self.wallet = None
        self.subtensor = None
        self.metagraph = None
        self.setup_success = False
        
        if not self.simulation_mode and BITTENSOR_AVAILABLE:
            self.setup_success = self._setup_bittensor()
        else:
            self.setup_success = True  # Assume success for simulation mode
        
        # Initialize prompt optimizer
        if OPTIMIZED_PROMPT_OPTIMIZER_AVAILABLE:
            self.prompt_optimizer = LLMPromptOptimizer()
        else:
            self.prompt_optimizer = TrellisPromptOptimizer()
        
        # Initialize reproducibility system (optional)
        self.reproducibility_system = None
        if REPRODUCIBILITY_SYSTEM_AVAILABLE and self.config.get('enable_reproducibility_optimization', True):
            try:
                self.reproducibility_system = SmartReproducibilitySystem()
                self.logger.info("✅ Reproducibility system initialized")
            except Exception as e:
                self.logger.warning(f"⚠️ Failed to initialize reproducibility system: {e}")
        
        # Initialize CLIP episodic memory
        self.episodic_memory = {}
        self.clip_memory = None
        self._load_episodic_memory()
        
        self.logger.info("✅ Continuous TRELLIS Orchestrator with CLIP optimization initialized")

    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration"""
        return {
            'generation_server_url': 'http://localhost:8096',
            'validation_server_url': 'http://localhost:10006',
            'output_dir': './continuous_trellis_hunyuan_clip_outputs',
            'harvest_tasks': True,
            'validate_generations': True,
            'submit_results': True,
            'task_pull_interval': 30,
            'validation_check_interval': 120,
            'status_print_interval': 300,
            'statistics_save_interval': 300,
            'generation_timeout': 300,
            'submission_timeout': 120,
            'max_consecutive_failures': 5,
            'validator_refresh_interval': 1800,
            'cooldown_duration': 300,
            'min_validator_stake': 50.0,
            'min_local_score': 0.3,
            'enable_prompt_optimization': True,
            'optimization_aggressive_mode': False,
            'log_optimization_details': True,
            'save_intermediate_results': True,
            'netuid': 17,
            'debug_mode': False,
            'enable_reproducibility_optimization': True,
            'reproducibility_min_similarity': 0.3,
            'enable_clip_optimization': True,
            'clip_memory_similarity_threshold': 0.7,
            'use_fixed_seed': True,
            'fixed_seed_value': 42,
            'priority_max_wait_seconds': 60,
            'priority_timeout_seconds': 30,
            # CLIP optimization specific configs
            'episodic_memory_file': 'episodic_clip_memory.json',
            'clip_similarity_threshold': 0.51,
            'reproducibility_min_fidelity_score': 0.85,
            'reproducibility_similarity_threshold': 0.51,
            'llm_optimization_attempts': 3,
            'enable_reproducibility_check': True
        }

    def load_prompts_from_file(self, filepath: str) -> List[str]:
        """Load prompts from a Python file containing EPISODIC_TEST_PROMPTS"""
        try:
            spec = importlib.util.spec_from_file_location("prompt_module", filepath)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            
            if hasattr(module, 'EPISODIC_TEST_PROMPTS'):
                prompts = module.EPISODIC_TEST_PROMPTS
                self.logger.info(f"📄 Loaded {len(prompts)} prompts from {filepath}")
                return prompts
            else:
                self.logger.error(f"❌ No EPISODIC_TEST_PROMPTS found in {filepath}")
                return []
                
        except Exception as e:
            self.logger.error(f"❌ Failed to load prompts from {filepath}: {e}")
            return []

    def _setup_bittensor(self) -> bool:
        """Initialize Bittensor components"""
        try:
            if not BITTENSOR_AVAILABLE:
                self.logger.warning("⚠️ Bittensor not available - mining operations will be simulated")
                return False
            
            # Initialize wallet
            self.wallet = bt.wallet()
            self.logger.info(f"💳 Wallet: {self.wallet.hotkey.ss58_address}")
            
            # Initialize subtensor
            self.subtensor = bt.subtensor()
            self.logger.info(f"🌐 Connected to subtensor: {self.subtensor.network}")
            
            # Initialize metagraph
            self.metagraph = bt.metagraph(netuid=self.config['netuid'])
            self.logger.info(f"📊 Metagraph loaded for netuid {self.config['netuid']}")
            
            # Sync metagraph
            self.metagraph.sync(subtensor=self.subtensor)
            self.logger.info(f"🔄 Metagraph synced - {len(self.metagraph.neurons)} neurons")
            
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Bittensor setup failed: {e}")
            return False

    def refresh_validators(self):
        """Refresh validator information from the metagraph"""
        if not self.metagraph:
            return
        
        try:
            self.logger.info("🔄 Refreshing validator information...")
            
            # Sync metagraph to get latest state
            self.metagraph.sync(subtensor=self.subtensor)
            
            # Update validator states
            for neuron in self.metagraph.neurons:
                # Check if this neuron is a validator (has high stake)
                if neuron.stake.tao >= self.config['min_validator_stake']:
                    uid = neuron.uid
                    
                    if uid not in self.validators:
                        # New validator
                        self.validators[uid] = ValidatorState(
                            uid=uid,
                            hotkey=neuron.hotkey,
                            stake=neuron.stake.tao,
                            trust=neuron.trust,
                            consensus=neuron.consensus
                        )
                        self.logger.info(f"➕ New validator {uid}: {neuron.stake.tao:.1f} TAO")
                    else:
                        # Update existing validator
                        validator = self.validators[uid]
                        validator.stake = neuron.stake.tao
                        validator.trust = neuron.trust
                        validator.consensus = neuron.consensus
                        validator.hotkey = neuron.hotkey
            
            # Mark inactive validators
            active_uids = {neuron.uid for neuron in self.metagraph.neurons 
                          if neuron.stake.tao >= self.config['min_validator_stake']}
            
            for uid in self.validators:
                self.validators[uid].is_active = uid in active_uids
            
            active_count = sum(1 for v in self.validators.values() if v.is_active)
            self.logger.info(f"✅ Validator refresh complete: {active_count} active validators")
            
            self.last_validator_refresh = time.time()
            
        except Exception as e:
            self.logger.error(f"❌ Validator refresh failed: {e}")

    def is_validator_available(self, validator: ValidatorState) -> bool:
        """Check if validator is available for task pulling"""
        if not validator.is_active:
            return False
        
        current_time = time.time()
        
        # Check cooldown
        if validator.cooldown_until and current_time < validator.cooldown_until:
            return False
        
        # Check if we've pulled from this validator recently
        min_interval = self.config['task_pull_interval']
        if (validator.last_task_pull and 
            current_time - validator.last_task_pull < min_interval):
            return False
        
        return True

    async def pull_task_from_validator(self, validator: ValidatorState) -> Optional[TaskRecord]:
        """Pull a task from a specific validator"""
        try:
            if not self.wallet:
                return None
            
            # Create task pull request
            request_data = {
                'miner_hotkey': self.wallet.hotkey.ss58_address,
                'miner_uid': getattr(self.metagraph.hotkeys.index(self.wallet.hotkey.ss58_address), 'uid', -1)
            }
            
            # For simulation, we don't actually make HTTP requests
            if self.simulation_mode:
                return None
            
            # In real mode, this would make HTTP request to validator
            # For now, we'll simulate with None return
            return None
            
        except Exception as e:
            self.logger.error(f"❌ Task pull failed from validator {validator.uid}: {e}")
            return None

    def get_deterministic_seed(self, task: TaskRecord) -> int:
        """Generate deterministic seed based on task and configuration"""
        if self.config.get('use_fixed_seed', True):
            return self.config.get('fixed_seed_value', 42)
        else:
            # Generate hash-based seed from prompt and task ID for determinism
            seed_string = f"{task.prompt}{task.task_id}"
            seed_hash = hashlib.sha256(seed_string.encode()).hexdigest()
            return int(seed_hash[:8], 16) % (2**31)

    def _load_episodic_memory(self):
        """Load episodic memory from file for CLIP optimization"""
        memory_file = Path(self.config.get('episodic_memory_file', 'episodic_clip_memory.json'))
        if memory_file.exists():
            try:
                with open(memory_file, 'r') as f:
                    data = json.load(f)
                
                for prompt, memory_data in data.items():
                    self.episodic_memory[prompt] = memory_data
                
                self.logger.info(f"📚 Loaded episodic memory: {len(self.episodic_memory)} prompts")
                
                # Also set clip_memory for compatibility
                self.clip_memory = data
                
            except Exception as e:
                self.logger.warning(f"⚠️ Failed to load episodic memory: {e}")
                self.episodic_memory = {}
        else:
            self.logger.info("📄 Starting fresh episodic memory")
    
    def calculate_cosine_similarity(self, prompt1: str, prompt2: str) -> float:
        """Calculate cosine similarity between two prompts using simple word overlap"""
        # Simple word-based similarity (can be enhanced with embeddings later)
        words1 = set(prompt1.lower().split())
        words2 = set(prompt2.lower().split())
        
        if not words1 or not words2:
            return 0.0
        
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        
        return len(intersection) / len(union) if union else 0.0
    
    def search_database_for_similar_prompts(self, prompt: str, min_fidelity_score: float = 0.85, min_similarity: float = 0.51) -> List[Dict[str, Any]]:
        """Search database for similar prompts with high fidelity scores"""
        try:
            conn = sqlite3.connect(self.db.db_path)
            cursor = conn.cursor()
            
            # Get all prompts with task_fidelity_score > threshold
            cursor.execute('''
                SELECT prompt, task_fidelity_score, local_validation_score, processed_at, task_id
                FROM tasks 
                WHERE task_fidelity_score > ? 
                AND processed_at IS NOT NULL
                ORDER BY task_fidelity_score DESC
            ''', (min_fidelity_score,))
            
            high_scoring_prompts = cursor.fetchall()
            conn.close()
            
            # Calculate similarity scores
            similar_prompts = []
            for db_prompt, fidelity_score, validation_score, processed_at, task_id in high_scoring_prompts:
                similarity = self.calculate_cosine_similarity(prompt, db_prompt)
                if similarity >= min_similarity:
                    similar_prompts.append({
                        'prompt': db_prompt,
                        'fidelity_score': fidelity_score,
                        'validation_score': validation_score,
                        'similarity': similarity,
                        'processed_at': processed_at,
                        'task_id': task_id
                    })
            
            # Sort by similarity score (highest first)
            similar_prompts.sort(key=lambda x: x['similarity'], reverse=True)
            return similar_prompts
            
        except Exception as e:
            self.logger.error(f"❌ Database search failed: {e}")
            return []
    
    def collect_reproducibility_references(self, similar_prompts: List[Dict[str, Any]]) -> str:
        """Format reproducibility references for LLM"""
        if not similar_prompts:
            return ""
        
        references = "REPRODUCIBILITY REFERENCES (High-scoring similar prompts from database):\n"
        references += "Use these as examples of successful optimizations:\n\n"
        
        for i, ref in enumerate(similar_prompts[:5], 1):  # Top 5 references
            references += f"{i}. Original: '{ref['prompt']}'\n"
            references += f"   Fidelity Score: {ref['fidelity_score']:.4f}\n"
            references += f"   Validation Score: {ref['validation_score']:.4f}\n"
            references += f"   Similarity: {ref['similarity']:.3f}\n\n"
        
        return references
    
    def find_exact_match_in_memory(self, prompt: str) -> Optional[Dict[str, Any]]:
        """Find exact match in episodic memory"""
        return self.episodic_memory.get(prompt)
    
    def find_similar_prompt_in_memory(self, prompt: str, min_similarity: float = 0.51) -> Optional[Dict[str, Any]]:
        """Find similar prompt in episodic memory"""
        best_match = None
        best_similarity = 0.0
        
        for memory_prompt, memory_data in self.episodic_memory.items():
            similarity = self.calculate_cosine_similarity(prompt, memory_prompt)
            if similarity >= min_similarity and similarity > best_similarity:
                best_similarity = similarity
                best_match = {
                    'prompt': memory_prompt,
                    'similarity': similarity,
                    'memory_data': memory_data
                }
        
        return best_match
    
    def format_pattern_analysis(self, top_3_prompts: List[Dict]) -> str:
        """Format pattern analysis for LLM"""
        if not top_3_prompts:
            return "No pattern analysis available."
        
        analysis = "PATTERN ANALYSIS:\n"
        for i, prompt_data in enumerate(top_3_prompts, 1):
            analysis += f"{i}. Score: {prompt_data.get('score', 0):.4f}\n"
            analysis += f"   Prompt: '{prompt_data.get('prompt', '')}'\n"
            analysis += f"   Strategy: {prompt_data.get('strategy', 'unknown')}\n\n"
        
        return analysis
    
    def call_llm_for_optimization(self, system_prompt: str, original_prompt: str) -> str:
        """Call LLM for prompt optimization"""
        try:
            if OPTIMIZED_PROMPT_OPTIMIZER_AVAILABLE:
                # Use the fast optimizer
                result = self.prompt_optimizer.optimize_with_examples(original_prompt)
                return result
            else:
                # Use the original optimizer
                optimization_result = self.prompt_optimizer.optimize_prompt(
                    original_prompt, 
                    aggressive=self.config.get('optimization_aggressive_mode', False)
                )
                return original_prompt  # Fallback to original
        except Exception as e:
            self.logger.error(f"❌ LLM optimization failed: {e}")
            return original_prompt
    
    def optimize_with_pattern_analysis_and_references(self, original_prompt: str, similar_prompt: str, top_3_prompts: List[Dict], reproducibility_references: str) -> Dict[str, Any]:
        """LLM optimization with both pattern analysis and reproducibility references"""
        
        system_prompt = f"""You are an expert prompt optimizer. Optimize the given prompt using:

{reproducibility_references}

PATTERN ANALYSIS FROM SIMILAR PROMPT:
Original similar prompt: "{similar_prompt}"

{self.format_pattern_analysis(top_3_prompts)}

TASK: Optimize the following prompt using the patterns above and reproducibility references:
Original prompt: "{original_prompt}"

Provide an optimized version that follows the successful patterns."""
        
        optimized_prompt = self.call_llm_for_optimization(system_prompt, original_prompt)
        
        return {
            'optimized_prompt': optimized_prompt,
            'method': 'similar_pattern',
            'similarity_score': 0.8,  # Placeholder
            'pattern_analysis': True
        }
    
    def optimize_with_examples_and_references_3_attempts(self, prompt: str, reproducibility_references: str) -> Dict[str, Any]:
        """LLM optimization with examples and reproducibility references (3 attempts)"""
        
        attempts = []
        for i in range(self.config.get('llm_optimization_attempts', 3)):
            system_prompt = f"""You are an expert prompt optimizer. Optimize the given prompt using:

{reproducibility_references}

TASK: Optimize the following prompt using the reproducibility references above:
Original prompt: "{prompt}"

Provide an optimized version that follows successful patterns from the references."""
            
            optimized_prompt = self.call_llm_for_optimization(system_prompt, prompt)
            similarity = self.calculate_cosine_similarity(prompt, optimized_prompt)
            
            attempts.append({
                'attempt': i + 1,
                'optimized_prompt': optimized_prompt,
                'similarity': similarity
            })
        
        # Choose the attempt with highest similarity
        best_attempt = max(attempts, key=lambda x: x['similarity'])
        
        return {
            'optimized_prompt': best_attempt['optimized_prompt'],
            'method': 'llm_examples',
            'similarity_score': best_attempt['similarity'],
            'all_attempts': attempts
        }
    
    def advanced_clip_optimization(self, original_prompt: str, reproducibility_references: str = "") -> Dict[str, Any]:
        """Advanced CLIP optimization with 3-tier system"""
        
        # Tier 1: Check for exact match
        exact_match = self.find_exact_match_in_memory(original_prompt)
        if exact_match:
            return {
                'optimized_prompt': exact_match.get('best_prompt', original_prompt),
                'method': 'exact_match',
                'similarity_score': 1.0
            }
        
        # Tier 2: Find similar prompt (similarity > threshold)
        similar_result = self.find_similar_prompt_in_memory(original_prompt, min_similarity=self.config.get('clip_similarity_threshold', 0.51))
        if similar_result:
            # Mock top 3 prompts for now (in real implementation, this would come from episodic memory)
            top_3_prompts = [
                {'prompt': similar_result['prompt'], 'score': 0.9, 'strategy': 'pattern_1'},
                {'prompt': similar_result['prompt'], 'score': 0.85, 'strategy': 'pattern_2'},
                {'prompt': similar_result['prompt'], 'score': 0.8, 'strategy': 'pattern_3'}
            ]
            return self.optimize_with_pattern_analysis_and_references(
                original_prompt, similar_result['prompt'], top_3_prompts, reproducibility_references
            )
        
        # Tier 3: No similar prompt - use LLM with examples
        return self.optimize_with_examples_and_references_3_attempts(original_prompt, reproducibility_references)

    def optimize_prompt_for_generation(self, task: TaskRecord) -> Dict[str, Any]:
        """Advanced prompt optimization with reproducibility + CLIP optimization"""
        
        optimization_result = {
            'optimized_prompt': task.prompt,
            'optimization_method': 'none',
            'similarity_score': 0.0,
            'pattern_analysis': False,
            'reproducibility_references': ""
        }
        
        try:
            # Step 1: Reproducibility Pre-Check
            reproducibility_references = ""
            if self.config.get('enable_reproducibility_check', True):
                similar_prompts = self.search_database_for_similar_prompts(
                    task.prompt,
                    min_fidelity_score=self.config.get('reproducibility_min_fidelity_score', 0.85),
                    min_similarity=self.config.get('reproducibility_similarity_threshold', 0.51)
                )
                
                if similar_prompts:
                    reproducibility_references = self.collect_reproducibility_references(similar_prompts)
                    optimization_result['reproducibility_references'] = reproducibility_references
                    optimization_result['reproducibility_references_found'] = len(similar_prompts)
                    optimization_result['reproducibility_similarity_scores'] = [p['similarity'] for p in similar_prompts]
                    
                    self.logger.info(f"🔄 Found {len(similar_prompts)} reproducibility references")
                    self.stats['reproducibility_references_found'] += len(similar_prompts)
                else:
                    self.logger.info(f"🔄 No reproducibility references found")
            
            # Step 2: CLIP Optimization
            if self.config.get('enable_clip_optimization', True):
                clip_result = self.advanced_clip_optimization(task.prompt, reproducibility_references)
                optimization_result.update(clip_result)
                self.logger.info(f"🧠 CLIP optimization method: {clip_result['method']}")
                
                # Update CLIP optimization statistics
                method = clip_result.get('method', 'none')
                if method == 'exact_match':
                    self.stats['clip_exact_matches'] += 1
                elif method == 'similar_pattern':
                    self.stats['clip_similar_patterns'] += 1
                elif method == 'llm_examples':
                    self.stats['clip_llm_examples'] += 1
                
                # Store optimization details in task record
                task.optimization_method = method
                task.optimization_similarity_score = clip_result.get('similarity_score', 0.0)
                task.optimized_prompt = clip_result.get('optimized_prompt', task.prompt)
                task.reproducibility_references_found = optimization_result.get('reproducibility_references_found')
                task.reproducibility_similarity_scores = optimization_result.get('reproducibility_similarity_scores')
                task.clip_optimization_method = method
                
                if clip_result.get('optimized_prompt') != task.prompt:
                    self.stats['prompts_optimized'] += 1
                    self.stats['clip_optimizations'] = self.stats.get('clip_optimizations', 0) + 1
            
            return optimization_result
            
        except Exception as e:
            self.logger.error(f"❌ Advanced optimization failed: {e}")
            return optimization_result

    def _wait_for_trellis_server_ready(self, max_wait_time: int = 300) -> bool:
        """Wait for TRELLIS server to be ready"""
        start_time = time.time()
        
        while time.time() - start_time < max_wait_time:
            try:
                response = requests.get(f"{self.config['generation_server_url']}/health/", timeout=5)
                if response.status_code == 200:
                    return True
            except:
                pass
            
            time.sleep(2)
        
        return False

    def _check_trellis_server_health(self) -> bool:
        """Check if TRELLIS server is healthy"""
        try:
            response = requests.get(f"{self.config['generation_server_url']}/health/", timeout=10)
            return response.status_code == 200
        except:
            return False

    def _clear_trellis_gpu_cache(self):
        """Clear TRELLIS server GPU cache"""
        try:
            response = requests.post(f"{self.config['generation_server_url']}/clear_cache/", timeout=15)
            if response.status_code == 200:
                self.logger.debug("🧹 Server cache cleared")
                return True
            else:
                self.logger.warning(f"⚠️ Cache clear failed: {response.status_code}")
                return False
        except Exception as e:
            self.logger.error(f"❌ Cache clear exception: {e}")
            return False

    async def generate_3d_model(self, task: TaskRecord) -> Optional[Dict[str, Any]]:
        """Generate 3D model using TRELLIS server with CLIP-optimized prompts"""
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
            
            # Step 1: Optimize prompt using CLIP optimization
            optimization_result = self.optimize_prompt_for_generation(task)
            optimized_prompt = optimization_result.get('optimized_prompt', task.prompt)
            
            # Log optimization details
            if optimized_prompt != task.prompt:
                self.logger.info(f"✨ Prompt optimized using {optimization_result.get('optimization_method', 'unknown')}")
                self.logger.info(f"   Original: {task.prompt}")
                self.logger.info(f"   Optimized: {optimized_prompt}")
            
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
        """Run validation with conda environment, supporting both original and optimized prompts."""
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
        """Validate generated model using the subnet validator with CLIP optimization support"""
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
            
            # Use the subnet validator function with optimized prompt support
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
        """Submit result to validator"""
        if not self.config['submit_results']:
            return True
        
        self.logger.info(f"📤 Submitting result: {task.task_id}")
        
        try:
            submission_start = time.time()
            
            # In simulation mode, always return success
            if self.simulation_mode:
                task.submission_success = True
                task.submitted_at = time.time()
                return True
            
            # Real submission would go here
            # For now, we'll simulate success
            task.submission_success = True
            task.submitted_at = time.time()
            
            submission_time = time.time() - submission_start
            self.stats['successful_submissions'] += 1
            
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Submission exception: {e}")
            self.stats['failed_submissions'] += 1
            return False

    async def process_task(self, task: TaskRecord) -> bool:
        """Process a single task end-to-end with priority access and CLIP optimization"""
        self.logger.info(f"🔄 Processing task {task.task_id}: '{task.prompt}'")
        
        task.processed_at = time.time()
        self.stats['tasks_processed'] += 1
        
        try:
            # Step 1: Generate 3D model with priority access and CLIP optimization
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
        """Perform validation on recent unvalidated tasks during idle periods"""
        if not self.config['validate_generations']:
            return
        
        try:
            # Get recent unvalidated tasks
            unvalidated_tasks = self.db.get_recent_unvalidated_tasks(hours=2)
            
            if not unvalidated_tasks:
                return
            
            self.logger.info(f"🔍 Found {len(unvalidated_tasks)} unvalidated tasks")
            
            for task in unvalidated_tasks[:3]:  # Validate up to 3 tasks per cycle
                self.logger.info(f"🔍 Idle validation: {task.task_id}")
                
                # Load PLY data if file exists
                if task.compressed_file_path and Path(task.compressed_file_path).exists():
                    with open(task.compressed_file_path, 'rb') as f:
                        ply_data = f.read()
                    
                    # Validate the model
                    score = await self.validate_model(task, ply_data)
                    
                    if score is not None:
                        # Update task record
                        self.db.save_task(task)
                        self.logger.info(f"✅ Idle validation completed: {score:.4f}")
                
        except Exception as e:
            self.logger.error(f"❌ Idle validation failed: {e}")

    def save_statistics(self):
        """Save statistics to JSON file"""
        try:
            stats_file = self.output_dir / "statistics.json"
            
            # Calculate additional statistics
            total_time = time.time() - getattr(self, 'start_time', time.time())
            
            enhanced_stats = {
                **self.stats,
                'total_runtime_seconds': total_time,
                'total_runtime_hours': total_time / 3600,
                'average_generation_time': (
                    self.stats['total_generation_time'] / self.stats['successful_generations']
                    if self.stats['successful_generations'] > 0 else 0
                ),
                'average_validation_time': (
                    self.stats['total_validation_time'] / self.stats['successful_validations']
                    if self.stats['successful_validations'] > 0 else 0
                ),
                'success_rate': (
                    self.stats['successful_submissions'] / self.stats['tasks_processed']
                    if self.stats['tasks_processed'] > 0 else 0
                ),
                'generation_success_rate': (
                    self.stats['successful_generations'] / self.stats['tasks_processed']
                    if self.stats['tasks_processed'] > 0 else 0
                ),
                'validation_success_rate': (
                    self.stats['successful_validations'] / self.stats['successful_generations']
                    if self.stats['successful_generations'] > 0 else 0
                ),
                'optimization_rate': (
                    self.stats['prompts_optimized'] / self.stats['tasks_processed']
                    if self.stats['tasks_processed'] > 0 else 0
                ),
                'clip_optimization_breakdown': {
                    'exact_matches': self.stats['clip_exact_matches'],
                    'similar_patterns': self.stats['clip_similar_patterns'],
                    'llm_examples': self.stats['clip_llm_examples']
                },
                'timestamp': time.time(),
                'human_timestamp': datetime.now().isoformat()
            }
            
            with open(stats_file, 'w') as f:
                json.dump(enhanced_stats, f, indent=2)
            
            self.last_statistics_save = time.time()
            
        except Exception as e:
            self.logger.error(f"❌ Failed to save statistics: {e}")

    def print_status(self):
        """Print current status"""
        try:
            total_time = time.time() - getattr(self, 'start_time', time.time())
            
            self.logger.info("="*60)
            self.logger.info(f"📊 TRELLIS ORCHESTRATOR STATUS (Runtime: {total_time/3600:.1f}h)")
            self.logger.info("="*60)
            
            # Task statistics
            self.logger.info(f"📋 Tasks: {self.stats['tasks_processed']} processed, {self.stats['tasks_pulled']} pulled")
            self.logger.info(f"   Duplicates skipped: {self.stats['duplicate_prompts_skipped']}")
            
            # Generation statistics
            success_rate = (self.stats['successful_generations'] / self.stats['tasks_processed'] * 100
                          if self.stats['tasks_processed'] > 0 else 0)
            self.logger.info(f"🎨 Generations: {self.stats['successful_generations']}/{self.stats['tasks_processed']} ({success_rate:.1f}%)")
            
            if self.stats['successful_generations'] > 0:
                avg_gen_time = self.stats['total_generation_time'] / self.stats['successful_generations']
                self.logger.info(f"   Average time: {avg_gen_time:.2f}s")
            
            # Validation statistics
            if self.config['validate_generations']:
                val_rate = (self.stats['successful_validations'] / self.stats['successful_generations'] * 100
                           if self.stats['successful_generations'] > 0 else 0)
                self.logger.info(f"📊 Validations: {self.stats['successful_validations']}/{self.stats['successful_generations']} ({val_rate:.1f}%)")
                
                if self.stats['successful_validations'] > 0:
                    avg_val_time = self.stats['total_validation_time'] / self.stats['successful_validations']
                    self.logger.info(f"   Average time: {avg_val_time:.2f}s")
            
            # CLIP Optimization statistics
            opt_rate = (self.stats['prompts_optimized'] / self.stats['tasks_processed'] * 100
                       if self.stats['tasks_processed'] > 0 else 0)
            self.logger.info(f"🧠 CLIP Optimizations: {self.stats['prompts_optimized']}/{self.stats['tasks_processed']} ({opt_rate:.1f}%)")
            self.logger.info(f"   Exact matches: {self.stats['clip_exact_matches']}")
            self.logger.info(f"   Similar patterns: {self.stats['clip_similar_patterns']}")
            self.logger.info(f"   LLM examples: {self.stats['clip_llm_examples']}")
            self.logger.info(f"   Reproducibility refs: {self.stats['reproducibility_references_found']}")
            
            # Submission statistics
            if self.config['submit_results']:
                sub_rate = (self.stats['successful_submissions'] / self.stats['tasks_processed'] * 100
                           if self.stats['tasks_processed'] > 0 else 0)
                self.logger.info(f"📤 Submissions: {self.stats['successful_submissions']}/{self.stats['tasks_processed']} ({sub_rate:.1f}%)")
            
            # Priority access statistics
            if self.stats.get('priority_access_timeouts', 0) > 0:
                self.logger.info(f"⏰ Priority timeouts: {self.stats['priority_access_timeouts']}")
            
            # Active validators
            if self.validators:
                active_validators = sum(1 for v in self.validators.values() if v.is_active)
                self.logger.info(f"👥 Active validators: {active_validators}/{len(self.validators)}")
            
            self.logger.info("="*60)
            
            self.last_status_print = time.time()
            
        except Exception as e:
            self.logger.error(f"❌ Failed to print status: {e}")

    async def continuous_mining_loop(self):
        """Main continuous mining loop with CLIP optimization"""
        self.logger.info("🚀 Starting continuous TRELLIS mining with CLIP optimization...")
        self.start_time = time.time()
        self.is_mining = True
        
        # Initial setup
        if not self.simulation_mode and self.metagraph:
            self.refresh_validators()
        
        try:
            while self.is_mining:
                loop_start = time.time()
                
                # Check if we should refresh validators
                if (not self.simulation_mode and 
                    time.time() - self.last_validator_refresh > self.config['validator_refresh_interval']):
                    self.refresh_validators()
                
                # Task harvesting
                if self.config['harvest_tasks'] and not self.simulation_mode:
                    # In real mode, try to pull tasks from validators
                    available_validators = [v for v in self.validators.values() if self.is_validator_available(v)]
                    
                    if available_validators:
                        # Select validator (could be randomized or based on stake)
                        validator = random.choice(available_validators)
                        
                        # Try to pull task
                        task = await self.pull_task_from_validator(validator)
                        if task:
                            self.stats['tasks_pulled'] += 1
                            
                            # Check for duplicates
                            if self.db.is_duplicate_prompt(task.prompt, validator.uid):
                                self.logger.info(f"⏭️ Duplicate prompt skipped: '{task.prompt[:50]}...'")
                                self.stats['duplicate_prompts_skipped'] += 1
                                continue
                            
                            # Process the task
                            await self.process_task(task)
                            
                            # Update validator state
                            validator.last_task_pull = time.time()
                            validator.total_tasks_pulled += 1
                            
                            # Add to recent prompts
                            self.db.add_recent_prompt(task.prompt, validator.uid)
                
                # Simulation mode - process prompts from file
                elif self.simulation_mode and self.simulation_prompts:
                    await self._run_simulation_mode()
                
                # Periodic maintenance
                current_time = time.time()
                
                # Idle validation
                if (current_time - getattr(self, 'last_validation_check', 0) > 
                    self.config['validation_check_interval']):
                    await self.idle_validation_cycle()
                    self.last_validation_check = current_time
                
                # Status printing
                if (current_time - self.last_status_print > self.config['status_print_interval']):
                    self.print_status()
                
                # Statistics saving
                if (current_time - self.last_statistics_save > self.config['statistics_save_interval']):
                    self.save_statistics()
                
                # Cleanup old prompts daily
                if (current_time - getattr(self, 'last_cleanup', 0) > 86400):  # 24 hours
                    self.db.cleanup_old_prompts()
                    self.last_cleanup = current_time
                
                # Sleep before next iteration
                loop_time = time.time() - loop_start
                sleep_time = max(1, self.config['task_pull_interval'] - loop_time)
                await asyncio.sleep(sleep_time)
                
        except KeyboardInterrupt:
            self.logger.info("🛑 Mining interrupted by user")
        except Exception as e:
            self.logger.error(f"❌ Mining loop failed: {e}")
            traceback.print_exc()
        finally:
            self.is_mining = False
            self.logger.info("🏁 Mining stopped")
            
            # Final statistics
            self.print_status()
            self.save_statistics()

    async def _run_simulation_mode(self):
        """Run simulation mode processing"""
        if self.simulation_index >= len(self.simulation_prompts):
            self.logger.info("🎯 All simulation prompts processed")
            self.is_mining = False
            return
        
        prompt = self.simulation_prompts[self.simulation_index]
        self.simulation_index += 1
        
        # Create simulated task
        task = TaskRecord(
            task_id=f"sim_{self.simulation_index}_{int(time.time())}",
            prompt=prompt,
            prompt_hash=hashlib.sha256(prompt.encode()).hexdigest(),
            validator_uid=-1,  # Simulation mode
            validator_hotkey="simulator",
            validator_stake=0.0,
            validation_threshold=0.0,
            pulled_at=time.time()
        )
        
        self.logger.info(f"🎯 Simulation task {self.simulation_index}/{len(self.simulation_prompts)}: '{prompt}'")
        
        # Process the simulated task
        await self.process_task(task)

    def _on_priority_interruption(self):
        """Callback for when we interrupt an optimizer job"""
        self.stats['mining_interruptions'] += 1
        self.logger.info("🛑 Optimizer job interrupted for subnet mining priority")

async def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Continuous TRELLIS Orchestrator with CLIP Optimization")
    parser.add_argument("--promptfile", help="Path to Python file with EPISODIC_TEST_PROMPTS list (simulation mode)")
    parser.add_argument("--no-harvest", action="store_true", help="Disable task harvesting")
    parser.add_argument("--no-validate", action="store_true", help="Disable validation")
    parser.add_argument("--no-submit", action="store_true", help="Disable result submission")
    parser.add_argument("--generation-server", default="http://localhost:8096", help="TRELLIS generation server URL")
    parser.add_argument("--validation-server", default="http://localhost:10006", help="Validation server URL")
    parser.add_argument("--output-dir", default="./continuous_trellis_hunyuan_clip_outputs", help="Output directory")
    parser.add_argument("--min-score", type=float, default=0.3, help="Minimum local validation score")
    
    # Prompt optimization arguments
    parser.add_argument("--no-optimize", action="store_true", help="Disable prompt optimization")
    parser.add_argument("--aggressive-optimize", action="store_true", help="Enable aggressive optimization mode")
    parser.add_argument("--quiet-optimize", action="store_true", help="Reduce optimization logging detail")
    
    # Reproducibility optimization arguments
    parser.add_argument("--no-reproducibility", action="store_true", help="Disable reproducibility optimization")
    parser.add_argument("--reproducibility-similarity", type=float, default=0.51, help="Minimum similarity threshold for reproducibility (default: 0.51)")
    
    # CLIP memory optimization arguments
    parser.add_argument("--no-clip-optimization", action="store_true", help="Disable CLIP memory-based optimization")
    parser.add_argument("--clip-similarity-threshold", type=float, default=0.51, help="Similarity threshold for CLIP memory hits (default: 0.51)")
    
    # CLIP optimization specific arguments
    parser.add_argument("--no-reproducibility-check", action="store_true", help="Disable reproducibility pre-check")
    parser.add_argument("--reproducibility-fidelity-threshold", type=float, default=0.85, help="Minimum fidelity score for reproducibility references")
    parser.add_argument("--reproducibility-similarity-threshold", type=float, default=0.51, help="Minimum similarity for reproducibility references")
    parser.add_argument("--episodic-memory-file", default="episodic_clip_memory.json", help="Path to episodic memory file")
    parser.add_argument("--episodic-run-log-file", default="episodic_clip_logs/multi_generator_results_run1.json", help="Path to episodic run log file for detailed results")
    parser.add_argument("--llm-optimization-attempts", type=int, default=3, help="Number of LLM optimization attempts")
    
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
    
    # CLIP optimization specific configuration
    if args.no_reproducibility_check:
        config['enable_reproducibility_check'] = False
    config['reproducibility_min_fidelity_score'] = args.reproducibility_fidelity_threshold
    config['reproducibility_similarity_threshold'] = args.reproducibility_similarity_threshold
    config['clip_similarity_threshold'] = args.clip_similarity_threshold
    config['episodic_memory_file'] = args.episodic_memory_file
    config['episodic_run_log_file'] = args.episodic_run_log_file
    config['llm_optimization_attempts'] = args.llm_optimization_attempts
    
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