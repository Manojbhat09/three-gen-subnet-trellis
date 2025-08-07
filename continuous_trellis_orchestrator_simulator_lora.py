#!/usr/bin/env python3
"""
Continuous TRELLIS Orchestrator SIMULATOR with LoRA Routing and Advanced CLIP Optimization
Purpose: Simulate mining by processing a list of prompts from a file against the
         local TRELLIS generation and validation servers with intelligent LoRA routing
         and advanced CLIP optimization including reproducibility pre-check.
"""

import asyncio
import json
import time
import argparse
import requests
import base64
import logging
import traceback
import hashlib
import sqlite3
import importlib.util
import subprocess
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional, Any, Tuple
from dataclasses import dataclass

# Import torch for CUDA cache management
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("⚠️ PyTorch not available - CUDA cache management disabled")

# Import the prompt optimizer
try:
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
    REPRODUCIBILITY_SYSTEM_AVAILABLE = True
    print("✅ Using reproducibility system for pre-optimization")
except ImportError:
    REPRODUCIBILITY_SYSTEM_AVAILABLE = False
    print("⚠️ Reproducibility system not available")

# Import the hybrid LoRA router
try:
    from hybrid_ultimate_router import HybridUltimateRouter, RouterResult
    # LORA_ROUTER_AVAILABLE = True
    LORA_ROUTER_AVAILABLE = False
    print("✅ Using hybrid ultimate LoRA router")
except ImportError:
    LORA_ROUTER_AVAILABLE = False
    print("⚠️ LoRA router not available")

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('continuous_trellis_simulator.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Simplified TaskRecord for simulation
@dataclass
class TaskRecord:
    """Record of a simulated task with essential metadata"""
    task_id: str
    prompt: str
    prompt_hash: str
    pulled_at: float
    # Optional fields from generation/validation
    processed_at: Optional[float] = None
    generation_time: Optional[float] = None
    validation_time: Optional[float] = None
    total_processing_time: Optional[float] = None
    local_validation_score: Optional[float] = None
    ply_file_path: Optional[str] = None
    compressed_file_path: Optional[str] = None
    # Add dummy fields to match the DB schema for simplicity
    validator_uid: int = -1
    validator_hotkey: str = "simulator"
    validator_stake: float = 0.0
    validation_threshold: float = 0.0
    submitted_at: Optional[float] = None
    submission_success: bool = False
    feedback_received: bool = False
    task_fidelity_score: Optional[float] = None
    average_fidelity_score: Optional[float] = None
    current_miner_reward: Optional[float] = None
    validation_failed: Optional[bool] = None
    generations_in_window: Optional[int] = None
    # NEW: LoRA routing and optimization fields
    selected_generator: Optional[str] = None
    router_decision: Optional[Dict[str, Any]] = None
    reproducibility_references_found: Optional[int] = None
    reproducibility_similarity_scores: Optional[List[float]] = None
    clip_optimization_method: Optional[str] = None  # "exact_match", "similar_pattern", "llm_examples"
    similar_prompt_found: Optional[str] = None
    optimization_similarity: Optional[float] = None
    pattern_analysis_used: Optional[bool] = None
    llm_optimization_attempts: Optional[int] = None
    final_optimized_prompt: Optional[str] = None


class TaskDatabase:
    """SQLite database for task tracking and deduplication"""
    
    def __init__(self, db_path: str = "continuous_trellis_simulator_lora_tasks.db"):
        self.db_path = db_path
        self.init_database()
    
    def init_database(self):
        """Initialize the database schema"""
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
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_prompt_hash ON tasks(prompt_hash)')
        conn.commit()
        conn.close()

    def has_processed_prompt(self, prompt_hash: str) -> bool:
        """Check if this prompt has already been processed."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('SELECT COUNT(*) FROM tasks WHERE prompt_hash = ? AND processed_at IS NOT NULL', (prompt_hash,))
        count = cursor.fetchone()[0]
        conn.close()
        return count > 0

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


class ContinuousTrellisSimulator:
    """Simulator for the TRELLIS orchestrator"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = self._get_default_config()
        self.config.update(config)
        self.logger = logger
        
        self.output_dir = Path(self.config['output_dir'])
        self.output_dir.mkdir(exist_ok=True)
        
        self.db = TaskDatabase(db_path=str(self.output_dir / "trellis_simulator_tasks.db"))
        
        self.running = False
        self.start_time = time.time()
        
        # Initialize prompt optimizer
        if OPTIMIZED_PROMPT_OPTIMIZER_AVAILABLE:
            self.prompt_optimizer = LLMPromptOptimizer(model="llama3.2:3b")
            self.logger.info("🚀 Initialized performance-optimized prompt optimizer")
        else:
            self.prompt_optimizer = TrellisPromptOptimizer()
            self.logger.info("🔧 Initialized standard prompt optimizer")
        
        # Initialize reproducibility system
        if REPRODUCIBILITY_SYSTEM_AVAILABLE:
            self.reproducibility_system = LLMClosePromptReproducibility()
            self.logger.info("🔄 Initialized reproducibility system for pre-optimization")
        else:
            self.reproducibility_system = None
            self.logger.info("⚠️ Reproducibility system not available")
        
        # Initialize LoRA router
        if LORA_ROUTER_AVAILABLE and self.config.get('enable_lora_routing', True):
            self.lora_router = HybridUltimateRouter()
            self.logger.info("🎯 Initialized hybrid ultimate LoRA router")
        else:
            self.lora_router = None
            self.logger.info("⚠️ LoRA router not available or disabled")
        
        # Generator endpoints mapping for LoRA routing
        self.generator_endpoints = {
            "Patched Realism": "http://localhost:8096/generate_image/patched_realism/",
            "Team Fortress 2 Style": "http://localhost:8096/generate_image/tf2_style/",
            "Cartoon 3D Render": "http://localhost:8096/generate_image/cartoon_3d/",
            "3D Game Assets": "http://localhost:8096/generate_image/game_assets/",
            "Game Icon Institute": "http://localhost:8096/generate_image/sd15_game_icon/",
            "Cinema Style": "http://localhost:8096/generate_image/cinema/",
            "Flux Isometric 3D": "http://localhost:8096/generate_image/isometric_3d/",
            "baolei": "http://localhost:8096/generate_image/baolei/",
            "Necklace Style": "http://localhost:8096/generate_image/necklace/"
        }
        
        # Load episodic memory for CLIP optimization
        self.episodic_memory = {}
        if self.config.get('enable_clip_optimization', True):
            self._load_episodic_memory()
        
        self.stats = {
            'session_start': time.time(),
            'prompts_loaded': 0,
            'tasks_to_process': 0,
            'tasks_processed': 0,
            'tasks_skipped': 0,
            'successful_generations': 0,
            'successful_validations': 0,
            'total_generation_time': 0.0,
            'total_validation_time': 0.0,
            'prompts_optimized': 0,
            'reproducibility_optimizations': 0,
            'traditional_optimizations': 0,
            'optimization_improvements': 0,
            # NEW: Advanced optimization statistics
            'lora_routing_decisions': 0,
            'reproducibility_references_found': 0,
            'clip_exact_matches': 0,
            'clip_similar_patterns': 0,
            'clip_llm_examples': 0,
            'generator_usage': {},
            'clip_feedback_optimizations': 0,
            'advanced_optimizations': 0,
            'basic_optimizations': 0,
        }
        
        self.logger.info("🎯 Continuous TRELLIS Simulator initialized")
        self.logger.info(f"   Output directory: {self.output_dir}")
        self.logger.info(f"   Generation server: {self.config['generation_server_url']}")
        self.logger.info(f"   Validation server: {self.config['validation_server_url']}")
        
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
        else:
            self.logger.info(f"🔧 Prompt optimization: DISABLED")

    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration"""
        return {
            'generation_server_url': 'http://localhost:8096',
            'validation_server_url': 'http://localhost:10006',
            'output_dir': './trellis_simulation_outputs',
            'validate_generations': True,
            'save_intermediate_results': True,
            'generation_timeout': 300,
            'validation_timeout': 120,
            'enable_prompt_optimization': True,
            'optimization_aggressive_mode': False,
            'log_optimization_details': True,
            'enable_reproducibility_optimization': True,
            'reproducibility_min_similarity': 0.3,
            'use_fixed_seed': True,
            'fixed_seed_value': 42,
            # NEW: LoRA routing and advanced optimization settings
            'enable_lora_routing': True,
            'enable_reproducibility_check': True,
            'enable_clip_optimization': True,
            'reproducibility_min_fidelity_score': 0.85,
            'reproducibility_similarity_threshold': 0.51,
            'clip_similarity_threshold': 0.51,
            'lora_confidence_threshold': 0.5,
            'episodic_memory_file': 'episodic_clip_memory.json',
            'pattern_analysis_enabled': True,
            'llm_optimization_attempts': 3,
            'enable_clip_feedback_optimization': True,
            'target_clip_score': 0.8,
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
            self.stats['prompts_loaded'] = len(prompts)
            return prompts
        except Exception as e:
            self.logger.error(f"❌ Failed to load prompts from {filepath}: {e}")
            return []

    def get_deterministic_seed(self, task: TaskRecord) -> int:
        """Generate deterministic seed based on prompt for consistent results with variety"""
        if self.config.get('use_fixed_seed', True):
            return self.config.get('fixed_seed_value', 42)  # Use configured fixed seed
        else:
            # Generate deterministic seed from prompt hash for variety but determinism
            hash_obj = hashlib.sha256(task.prompt.encode())
            seed = int(hash_obj.hexdigest()[:8], 16) % (2**31)  # Convert to 32-bit int
            return seed

    def optimize_prompt_for_generation(self, task: TaskRecord) -> Dict[str, Any]:
        """Enhanced prompt optimization with CLIP feedback loops and image interrogator"""
        try:
            # Check if optimization is enabled
            if not self.config.get('enable_prompt_optimization', True):
                return {
                    'optimized_prompt': task.prompt,
                    'method': 'disabled',
                    'improvement': 0.0,
                    'original_score': 0.0,
                    'final_score': 0.0
                }
            
            # Step 1: Try new CLIP feedback optimization system if available
            if self.config.get('enable_clip_feedback_optimization', True):
                try:
                    # Call the new optimization endpoint
                    response = requests.post(
                        f"{self.config['generation_server_url']}/optimize_prompt/",
                        data={
                            'prompt': task.prompt,
                            'seed': self.get_deterministic_seed(task),
                            'find_optimal_lora': True,
                            'target_score': self.config.get('target_clip_score', 0.8)
                        },
                        timeout=300  # 5 minutes for comprehensive optimization
                    )
                    
                    if response.status_code == 200:
                        result = response.json()
                        if result.get('status') == 'success':
                            optimized_prompt = result['optimized_prompt']
                            improvement = result['improvement']
                            
                            # Update task record with optimization details
                            task.clip_optimization_method = "clip_feedback_loop"
                            task.final_optimized_prompt = optimized_prompt
                            task.selected_generator = result.get('optimal_lora', 'isometric_3d')
                            
                            if self.config.get('log_optimization_details', True):
                                self.logger.info(f"🚀 CLIP feedback optimization applied:")
                                self.logger.info(f"   Original: {task.prompt}")
                                self.logger.info(f"   Optimized: {optimized_prompt}")
                                self.logger.info(f"   Improvement: {improvement:+.4f}")
                                self.logger.info(f"   Normalized score: {result['normalized_score']:.4f}")
                                self.logger.info(f"   Optimal LoRA: {result['optimal_lora']}")
                            else:
                                self.logger.info(f"🚀 CLIP optimized (Δ{improvement:+.3f}): '{task.prompt[:30]}...'")
                            
                            self.stats['prompts_optimized'] += 1
                            self.stats['clip_feedback_optimizations'] = self.stats.get('clip_feedback_optimizations', 0) + 1
                            
                            return {
                                'optimized_prompt': optimized_prompt,
                                'method': 'clip_feedback_loop',
                                'improvement': improvement,
                                'original_score': result['original_score'],
                                'final_score': result['final_score'],
                                'optimal_lora': result['optimal_lora'],
                                'optimization_time': result['optimization_time']
                            }
                
                except Exception as e:
                    self.logger.warning(f"CLIP feedback optimization failed, falling back: {e}")
            
            # Step 2: Try reproducibility system first (if available and enabled)
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
                    
                    # Update task record
                    task.clip_optimization_method = "reproducibility_system"
                    task.final_optimized_prompt = optimized_prompt
                    task.optimization_similarity = similarity
                    
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
                    
                    return {
                        'optimized_prompt': optimized_prompt,
                        'method': 'reproducibility_system',
                        'improvement': gold_score,
                        'similarity': similarity,
                        'gold_score': gold_score
                    }
            
            # Step 3: Fall back to traditional optimization if CLIP feedback didn't work
            if (OPTIMIZED_PROMPT_OPTIMIZER_AVAILABLE and 
                hasattr(self, 'prompt_optimizer') and 
                self.prompt_optimizer):
                
                # Advanced CLIP optimization with memory search
                optimization_result = self.advanced_clip_optimization(task.prompt)
                
                if optimization_result and optimization_result.get('optimized_prompt'):
                    optimized_prompt = optimization_result['optimized_prompt']
                    method = optimization_result.get('method', 'advanced_clip')
                    
                    # Update task record
                    task.clip_optimization_method = method
                    task.final_optimized_prompt = optimized_prompt
                    if 'similarity' in optimization_result:
                        task.optimization_similarity = optimization_result['similarity']
                    if 'similar_prompt_found' in optimization_result:
                        task.similar_prompt_found = optimization_result['similar_prompt_found']
                    if 'llm_attempts' in optimization_result:
                        task.llm_optimization_attempts = optimization_result['llm_attempts']
                    
                    if self.config.get('log_optimization_details', True):
                        self.logger.info(f"🧠 Advanced optimization applied:")
                        self.logger.info(f"   Original: {task.prompt}")
                        self.logger.info(f"   Optimized: {optimized_prompt}")
                        self.logger.info(f"   Method: {method}")
                    else:
                        self.logger.info(f"🧠 Advanced optimized ({method}): '{task.prompt[:30]}...'")
                    
                    self.stats['prompts_optimized'] += 1
                    self.stats['advanced_optimizations'] = self.stats.get('advanced_optimizations', 0) + 1
                    
                    return {
                        'optimized_prompt': optimized_prompt,
                        'method': method,
                        'improvement': optimization_result.get('confidence', 0.0)
                    }
            
            # Step 4: Final fallback - basic optimization
            if hasattr(self, 'prompt_optimizer') and self.prompt_optimizer:
                try:
                    optimized_prompt = self.prompt_optimizer.optimize_prompt(task.prompt)
                    if optimized_prompt and optimized_prompt != task.prompt:
                        task.clip_optimization_method = "basic_optimization"
                        task.final_optimized_prompt = optimized_prompt
                        
                        self.logger.info(f"🔧 Basic optimization applied: '{task.prompt[:30]}...' -> '{optimized_prompt[:30]}...'")
                        self.stats['prompts_optimized'] += 1
                        self.stats['basic_optimizations'] = self.stats.get('basic_optimizations', 0) + 1
                        
                        return {
                            'optimized_prompt': optimized_prompt,
                            'method': 'basic_optimization',
                            'improvement': 0.1  # Assume small improvement
                        }
                except Exception as e:
                    self.logger.warning(f"Basic optimization failed: {e}")
            
            # No optimization applied
            self.logger.info(f"ℹ️ No optimization applied to: '{task.prompt[:50]}...'")
            return {
                'optimized_prompt': task.prompt,
                'method': 'none',
                'improvement': 0.0
            }
            
        except Exception as e:
            self.logger.error(f"Prompt optimization failed: {e}")
            traceback.print_exc()
            return {
                'optimized_prompt': task.prompt,
                'method': 'error',
                'improvement': 0.0,
                'error': str(e)
            }

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
        """Generate 3D model using TRELLIS server with advanced optimization and LoRA routing"""
        self.logger.info(f"🎨 Generating 3D model: '{task.prompt}' (task: {task.task_id})")
        
        try:
            # Step 1: Wait for server to be ready before generation
            if not self._wait_for_trellis_server_ready():
                self.logger.error(f"❌ TRELLIS server did not become ready, skipping generation")
                return None
            
            # Step 2: Enhanced optimization with CLIP feedback loops, LoRA routing and reproducibility
            # optimization_result = self.optimize_prompt_for_generation(task)
            
            manual_optimized_prompt = f"{task.prompt}, front view, accurate, complete, white background"
            optimization_result = {'optimized_prompt': manual_optimized_prompt, 'method': 'none', 'improvement': 0.0}
            
            # Extract optimized prompt and optimization details
            optimized_prompt = optimization_result.get('optimized_prompt', task.prompt)
            optimization_method = optimization_result.get('method', 'none')
            optimization_improvement = optimization_result.get('improvement', 0.0)
            
            # Store optimization results in task
            task.final_optimized_prompt = optimized_prompt
            task.clip_optimization_method = optimization_method
            
            # Handle different optimization methods and their specific data
            if optimization_method == 'clip_feedback_loop':
                task.selected_generator = optimization_result.get('optimal_lora', 'isometric_3d')
                selected_generator = task.selected_generator
                generator_endpoint = f"{self.config['generation_server_url']}/generate/{selected_generator}/"
                
                self.logger.info(f"🚀 CLIP feedback optimization:")
                self.logger.info(f"   Method: {optimization_method}")
                self.logger.info(f"   Improvement: {optimization_improvement:+.4f}")
                self.logger.info(f"   Optimal LoRA: {selected_generator}")
                
            elif optimization_method in ['reproducibility_system', 'advanced_clip', 'basic_optimization']:
                # Use router for generator selection if available
                if self.lora_router:
                    router_result = self.route_prompt_to_optimal_lora(task)
                    task.selected_generator = router_result.get('selected_generator', 'Cinema Style')
                    task.router_decision = router_result
                    selected_generator = task.selected_generator
                    generator_endpoint = router_result.get('generator_endpoint', self.generator_endpoints['Cinema Style'])
                else:
                    selected_generator = 'Cinema Style'
                    generator_endpoint = self.generator_endpoints['Cinema Style']
                    task.selected_generator = selected_generator
                
                # Store method-specific data
                if 'similarity' in optimization_result:
                    task.optimization_similarity = optimization_result['similarity']
                if 'similar_prompt_found' in optimization_result:
                    task.similar_prompt_found = optimization_result['similar_prompt_found']
                if 'llm_attempts' in optimization_result:
                    task.llm_optimization_attempts = optimization_result['llm_attempts']
                    
            else:
                # No optimization or error case - use default routing
                if self.lora_router:
                    router_result = self.route_prompt_to_optimal_lora(task)
                    task.selected_generator = router_result.get('selected_generator', 'Cinema Style')
                    task.router_decision = router_result
                    selected_generator = task.selected_generator
                    generator_endpoint = router_result.get('generator_endpoint', self.generator_endpoints['Cinema Style'])
                else:
                    selected_generator = 'baolei'
                    generator_endpoint = self.generator_endpoints['baolei']
                    task.selected_generator = selected_generator
            
            self.logger.info(f"🎯 Using generator: {selected_generator}")
            self.logger.info(f"🎯 Generator endpoint: {generator_endpoint}")
            self.logger.info(f"📝 Optimized prompt: '{optimized_prompt}'")
            
            # Step 3: Clear cache on the server
            cache_cleared = self._clear_trellis_gpu_cache()
            if not cache_cleared:
                self.logger.warning(f"⚠️ Failed to clear GPU cache, proceeding anyway")

            # Step 4: Get deterministic seed
            deterministic_seed = self.get_deterministic_seed(task)
            self.logger.info(f"   🎲 Using deterministic seed: {deterministic_seed}")
            
            generation_start = time.time()
            
            # Step 5: Call TRELLIS generation server with selected generator endpoint
            response = requests.post(
                generator_endpoint,  # Use selected generator endpoint
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
                
                return {'ply_data': ply_data, 'compression_ratio': compression_ratio}
            else:
                self.logger.error(f"❌ Generation failed: HTTP {response.status_code}")
                return None
        
        except Exception as e:
            self.logger.error(f"❌ Generation exception: {e}")
            return None

    def _validate_prompt(self, original_prompt: str, optimized_prompt: str = None) -> float:
        """Run validation with conda environment."""
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
            # if result.stdout:
            #     self.logger.info(f"      📊 Validation stdout: {result.stdout.strip()}")
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
        """Validate generated model using the subnet validator"""
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
            
            # Use the subnet validator function
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

    async def process_task(self, task: TaskRecord) -> bool:
        """Process a single simulated task."""
        self.logger.info(f"🔄 Processing task {task.task_id}: '{task.prompt}'")
        task.processed_at = time.time()
        
        try:
            # Step 1: Generate 3D model
            generation_result = await self.generate_3d_model(task)
            if not generation_result:
                self.logger.error(f"❌ Generation failed for task {task.task_id}")
                self.db.save_task(task)
                return False
            
            # Step 2: Validate locally
            await self.validate_model(task, generation_result['ply_data'])
            
            # Step 3: Record total time and save
            if task.pulled_at:
                task.total_processing_time = time.time() - task.pulled_at
            
            self.db.save_task(task)
            self.logger.info(f"✅ Task {task.task_id} finished processing.")
            return True

        except Exception as e:
            self.logger.error(f"❌ Task processing failed: {e}")
            traceback.print_exc()
            self.db.save_task(task)
            return False

    def print_status(self):
        """Print final status summary."""
        uptime_hours = (time.time() - self.start_time) / 3600
        
        self.logger.info("\n" + "="*60)
        self.logger.info("📊 SIMULATION COMPLETE")
        self.logger.info(f"Total time: {uptime_hours * 60:.2f} minutes")
        self.logger.info(f"Prompts Loaded: {self.stats['prompts_loaded']}")
        self.logger.info(f"Prompts Skipped (already processed): {self.stats['tasks_skipped']}")
        self.logger.info(f"Tasks Processed: {self.stats['tasks_processed']} / {self.stats['tasks_to_process']}")
        self.logger.info(f"Successful Generations: {self.stats['successful_generations']}")
        
        if self.config['validate_generations']:
            self.logger.info(f"Successful Validations: {self.stats['successful_validations']}")
            
        if self.stats['successful_generations'] > 0:
            avg_gen_time = self.stats['total_generation_time'] / self.stats['successful_generations']
            self.logger.info(f"Average Generation Time: {avg_gen_time:.2f}s")
            
        if self.stats['successful_validations'] > 0:
            avg_val_time = self.stats['total_validation_time'] / self.stats['successful_validations']
            self.logger.info(f"Average Validation Time: {avg_val_time:.2f}s")
        
        # Optimization statistics
        if self.stats['prompts_optimized'] > 0:
            self.logger.info(f"Prompts Optimized: {self.stats['prompts_optimized']}")
            self.logger.info(f"Reproducibility Optimizations: {self.stats.get('reproducibility_optimizations', 0)}")
            self.logger.info(f"Traditional Optimizations: {self.stats.get('traditional_optimizations', 0)}")
            self.logger.info(f"CLIP Feedback Optimizations: {self.stats.get('clip_feedback_optimizations', 0)}")
            self.logger.info(f"Advanced Optimizations: {self.stats.get('advanced_optimizations', 0)}")
            self.logger.info(f"Basic Optimizations: {self.stats.get('basic_optimizations', 0)}")
        
        # Advanced optimization statistics
        if self.stats.get('lora_routing_decisions', 0) > 0:
            self.logger.info(f"LoRA Routing Decisions: {self.stats['lora_routing_decisions']}")
            self.logger.info(f"Reproducibility References Found: {self.stats.get('reproducibility_references_found', 0)}")
            self.logger.info(f"CLIP Exact Matches: {self.stats.get('clip_exact_matches', 0)}")
            self.logger.info(f"CLIP Similar Patterns: {self.stats.get('clip_similar_patterns', 0)}")
            self.logger.info(f"CLIP LLM Examples: {self.stats.get('clip_llm_examples', 0)}")
            
            # Generator usage statistics
            if self.stats.get('generator_usage'):
                self.logger.info(f"Generator Usage:")
                for generator, count in self.stats['generator_usage'].items():
                    self.logger.info(f"  {generator}: {count} times")
            
        self.logger.info(f"Outputs saved in: {self.output_dir}")
        self.logger.info("="*60)
    
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
    
    def route_prompt_to_optimal_lora(self, task: TaskRecord) -> Dict[str, Any]:
        """Route prompt to optimal LoRA using hybrid router"""
        if not self.lora_router:
            return {
                'generator': 'Cinema Style',
                'endpoint': self.generator_endpoints['Cinema Style'],
                'method': 'default_fallback'
            }
        
        try:
            router_result = self.lora_router.route_hybrid(task.prompt)
            
            return {
                'generator': router_result.recommended_lora,
                'endpoint': self.generator_endpoints.get(router_result.recommended_lora, self.generator_endpoints['Cinema Style']),
                'method': 'router_selection',
                'confidence': router_result.confidence,
                'reasoning': router_result.reasoning,
                'alternatives': router_result.alternatives
            }
        except Exception as e:
            self.logger.error(f"❌ LoRA routing failed: {e}")
            return {
                'generator': 'Cinema Style',
                'endpoint': self.generator_endpoints['Cinema Style'],
                'method': 'error_fallback'
            }
    
    def optimize_prompt_for_generation(self, task: TaskRecord) -> Dict[str, Any]:
        """Advanced prompt optimization with LoRA routing and reproducibility + CLIP optimization"""
        
        optimization_result = {
            'selected_generator': 'Cinema Style',
            'generator_endpoint': self.generator_endpoints['Cinema Style'],
            'optimized_prompt': task.prompt,
            'optimization_method': 'none',
            'similarity_score': 0.0,
            'pattern_analysis': False,
            'reproducibility_references': ""
        }
        
        try:
            # Step 1: LoRA Routing
            if self.config.get('enable_lora_routing', True):
                router_result = self.route_prompt_to_optimal_lora(task)
                optimization_result.update(router_result)
                self.logger.info(f"🎯 LoRA Router selected: {router_result['generator']} ({router_result['method']})")
                
                # Update statistics
                self.stats['lora_routing_decisions'] += 1
                generator = router_result['generator']
                self.stats['generator_usage'][generator] = self.stats['generator_usage'].get(generator, 0) + 1
            
            # Step 2: Reproducibility Pre-Check
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
            
            # Step 3: CLIP Optimization
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
            
            return optimization_result
            
        except Exception as e:
            self.logger.error(f"❌ Advanced optimization failed: {e}")
            return optimization_result

    async def run_simulation(self):
        """Main simulation loop."""
        self.logger.info("🚀 Starting TRELLIS simulation...")
        
        prompts = self.load_prompts_from_file(self.config['promptfile'])
        if not prompts:
            return
        
        tasks_to_process = []
        for i, prompt_text in enumerate(prompts):
            prompt_hash = hashlib.sha256(prompt_text.encode()).hexdigest()
            if self.db.has_processed_prompt(prompt_hash):
                self.logger.info(f"⏭️ Skipping already processed prompt: '{prompt_text[:50]}...'")
                self.stats['tasks_skipped'] += 1
                continue
                
            task = TaskRecord(
                task_id=f"sim_{i+1}",
                prompt=prompt_text,
                prompt_hash=prompt_hash,
                pulled_at=time.time()
            )
            tasks_to_process.append(task)
            
        self.stats['tasks_to_process'] = len(tasks_to_process)
        self.logger.info(f"Found {self.stats['tasks_to_process']} new prompts to process.")
        
        self.running = True
        self.start_time = time.time()
        
        try:
            for task in tasks_to_process:
                if not self.running:
                    self.logger.info("🛑 Simulation interrupted.")
                    break
                success = await self.process_task(task)
                if success:
                    self.stats['tasks_processed'] += 1
        except KeyboardInterrupt:
            self.logger.info("🛑 Simulation interrupted by user.")
        except Exception as e:
            self.logger.error(f"❌ Simulation loop error: {e}")
            traceback.print_exc()
        finally:
            self.running = False
            self.print_status()
            self.logger.info("🏁 Simulation stopped.")

async def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="TRELLIS Orchestrator Simulator")
    parser.add_argument("--promptfile", required=True, help="Path to Python file with EPISODIC_TEST_PROMPTS list.")
    parser.add_argument("--no-validate", action="store_true", help="Disable local validation of generated models.")
    parser.add_argument("--generation-server", default="http://localhost:8096", help="TRELLIS generation server URL.")
    parser.add_argument("--validation-server", default="http://localhost:10006", help="Validation server URL.")
    parser.add_argument("--output-dir", default="./trellis_simulation_outputs", help="Output directory for logs and models.")
    parser.add_argument("--no-optimize", action="store_true", help="Disable prompt optimization.")
    parser.add_argument("--aggressive-optimize", action="store_true", help="Enable aggressive optimization mode.")
    parser.add_argument("--quiet-optimize", action="store_true", help="Reduce optimization logging detail.")
    parser.add_argument("--no-reproducibility", action="store_true", help="Disable reproducibility optimization.")
    parser.add_argument("--reproducibility-similarity", type=float, default=0.3, help="Minimum similarity threshold for reproducibility.")
    parser.add_argument("--variable-seeds", action="store_true", help="Use prompt-hash based seeds (default: fixed seed 42).")
    parser.add_argument("--seed", type=int, default=42, help="Fixed seed to use when not using variable seeds.")
    # NEW: LoRA routing and advanced optimization arguments
    parser.add_argument("--no-lora-routing", action="store_true", help="Disable LoRA routing.")
    parser.add_argument("--no-reproducibility-check", action="store_true", help="Disable reproducibility pre-check.")
    parser.add_argument("--no-clip-optimization", action="store_true", help="Disable CLIP optimization.")
    parser.add_argument("--reproducibility-fidelity-threshold", type=float, default=0.85, help="Minimum fidelity score for reproducibility references.")
    parser.add_argument("--reproducibility-similarity-threshold", type=float, default=0.51, help="Minimum similarity for reproducibility references.")
    parser.add_argument("--clip-similarity-threshold", type=float, default=0.51, help="Minimum similarity for CLIP optimization.")
    parser.add_argument("--lora-confidence-threshold", type=float, default=0.5, help="Minimum confidence for LoRA router.")
    parser.add_argument("--episodic-memory-file", default="episodic_clip_memory.json", help="Path to episodic memory file.")
    parser.add_argument("--llm-optimization-attempts", type=int, default=3, help="Number of LLM optimization attempts.")
    parser.add_argument("--enable-clip-feedback-optimization", action="store_true", help="Enable new CLIP feedback optimization system.")
    parser.add_argument("--target-clip-score", type=float, default=0.8, help="Target CLIP score for feedback optimization.")
    
    args = parser.parse_args()
    
    config = {
        'promptfile': args.promptfile,
        'validate_generations': not args.no_validate,
        'generation_server_url': args.generation_server,
        'validation_server_url': args.validation_server,
        'output_dir': args.output_dir,
        'enable_prompt_optimization': not args.no_optimize,
        'optimization_aggressive_mode': args.aggressive_optimize,
        'log_optimization_details': not args.quiet_optimize,
        'enable_reproducibility_optimization': not args.no_reproducibility,
        'reproducibility_min_similarity': args.reproducibility_similarity,
        'use_fixed_seed': not args.variable_seeds,
        'fixed_seed_value': args.seed,
        # NEW: LoRA routing and advanced optimization settings
        'enable_lora_routing': not args.no_lora_routing,
        'enable_reproducibility_check': not args.no_reproducibility_check,
        'enable_clip_optimization': not args.no_clip_optimization,
        'reproducibility_min_fidelity_score': args.reproducibility_fidelity_threshold,
        'reproducibility_similarity_threshold': args.reproducibility_similarity_threshold,
        'clip_similarity_threshold': args.clip_similarity_threshold,
        'lora_confidence_threshold': args.lora_confidence_threshold,
        'episodic_memory_file': args.episodic_memory_file,
        'llm_optimization_attempts': args.llm_optimization_attempts,
        'enable_clip_feedback_optimization': args.enable_clip_feedback_optimization,
        'target_clip_score': args.target_clip_score,
    }
    
    simulator = ContinuousTrellisSimulator(config)
    await simulator.run_simulation()

if __name__ == "__main__":
    asyncio.run(main())
