#!/usr/bin/env python3
"""
Miner-Integrated Validator Simulation

This simulation integrates with the actual ContinuousTrellisOrchestrator miner
to provide realistic testing of the complete 3D generation subnet workflow.

Features:
- Uses actual miner components (TaskDatabase, ValidatorState, etc.)
- Realistic task processing with priority access
- Actual validation and submission logic
- Configurable simulation scenarios
- Performance monitoring and analysis

Usage:
    python3 miner_integrated_simulation.py [--scenario SCENARIO] [--validators N] [--duration SECONDS]
"""

import asyncio
import json
import time
import random
import logging
import argparse
import traceback
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import statistics
from collections import defaultdict, deque

# Import the actual miner components
try:
    from continuous_trellis_orchestrator_lora_working import (
        ContinuousTrellisOrchestrator,
        TaskRecord,
        ValidatorState,
        TaskDatabase,
        ValidatorStatePersistence
    )
    MINER_AVAILABLE = True
except ImportError as e:
    print(f"⚠️ Warning: Could not import miner components: {e}")
    print("   Running in mock mode - some features will be simulated")
    MINER_AVAILABLE = False

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class MinerSimulationConfig:
    """Configuration for the miner-integrated simulation"""
    
    # Simulation parameters
    scenario: str = "balanced"  # balanced, high_load, stress_test, learning, realistic
    num_validators: int = 5
    simulation_duration: int = 600  # 10 minutes
    task_pull_interval: int = 30   # seconds between task pulls
    
    # Miner integration settings
    use_real_miner: bool = MINER_AVAILABLE
    miner_config_file: Optional[str] = None
    enable_priority_access: bool = True
    enable_local_validation: bool = True
    enable_submission: bool = False  # Don't submit to real subnet during simulation
    
    # Task generation settings
    prompts_per_cycle: int = 50
    prompt_complexity_distribution: Dict[str, float] = field(default_factory=lambda: {
        "simple": 0.3,      # 3-5 words
        "medium": 0.5,      # 6-8 words  
        "complex": 0.2      # 9+ words
    })
    
    # Performance simulation
    generation_success_rate: float = 0.85
    validation_success_rate: float = 0.90
    submission_success_rate: float = 0.95
    
    # Timing simulation (seconds)
    generation_time_range: Tuple[float, float] = (8.0, 25.0)
    validation_time_range: Tuple[float, float] = (2.0, 8.0)
    submission_time_range: Tuple[float, float] = (1.0, 3.0)
    
    # Quality scoring
    quality_score_range: Tuple[float, float] = (0.6, 0.95)
    alignment_score_range: Tuple[float, float] = (0.5, 0.9)
    
    # Learning and adaptation
    enable_validator_learning: bool = True
    performance_memory_size: int = 100


class MockValidator:
    """Mock validator for simulation when real miner is not available"""
    
    def __init__(self, validator_id: int, config: MinerSimulationConfig):
        self.uid = validator_id
        self.hotkey = f"mock_validator_{validator_id}_hotkey"
        self.stake = random.uniform(100.0, 1000.0)
        self.trust = random.uniform(0.7, 1.0)
        self.consensus = random.uniform(0.6, 0.95)
        
        self.stats = {
            'tasks_pulled': 0,
            'tasks_processed': 0,
            'tasks_submitted': 0,
            'total_score': 0.0,
            'start_time': time.time(),
            'generation_success_rate': 0.8,
            'validation_success_rate': 0.9,
            'average_quality_score': 0.75
        }
        
        self.last_task_pull = None
        self.last_task_received = None
        self.cooldown_until = None
        self.throttle_period = 0
        self.cooldown_violations = 0
        self.is_active = True


class MockTaskRecord:
    """Mock task record for simulation"""
    
    def __init__(self, task_id: str, prompt: str, validator_uid: int, validator_hotkey: str, validator_stake: float):
        self.task_id = task_id
        self.prompt = prompt
        self.prompt_hash = self._generate_hash(prompt)
        self.validator_uid = validator_uid
        self.validator_hotkey = validator_hotkey
        self.validator_stake = validator_stake
        self.validation_threshold = 0.6
        self.pulled_at = time.time()
        self.processed_at = None
        self.submitted_at = None
        self.generation_time = None
        self.validation_time = None
        self.total_processing_time = None
        self.local_validation_score = None
        self.submission_success = False
        self.feedback_received = False
        self.task_fidelity_score = None
        self.average_fidelity_score = None
        self.current_miner_reward = None
        self.validation_failed = None
        self.generations_in_window = None
        self.ply_file_path = None
        self.compressed_file_path = None
        self.priority_access_timeout = False
    
    def _generate_hash(self, prompt: str) -> str:
        """Generate a mock hash for the prompt"""
        import hashlib
        return hashlib.sha256(prompt.encode()).hexdigest()


class MinerIntegratedSimulation:
    """Simulation that integrates with the actual miner system"""
    
    def __init__(self, config: MinerSimulationConfig):
        self.config = config
        self.logger = logger
        
        # Initialize miner if available
        if config.use_real_miner and MINER_AVAILABLE:
            self.miner = self._initialize_miner()
            self.logger.info("✅ Using real ContinuousTrellisOrchestrator")
        else:
            self.miner = None
            self.logger.info("⚠️ Using mock miner components")
        
        # Initialize components
        self.validators = self._initialize_validators()
        self.task_database = self._initialize_task_database()
        self.prompt_generator = self._initialize_prompt_generator()
        
        # Simulation state
        self.simulation_stats = {
            'total_tasks_generated': 0,
            'total_tasks_pulled': 0,
            'total_tasks_processed': 0,
            'total_tasks_submitted': 0,
            'total_generation_time': 0.0,
            'total_validation_time': 0.0,
            'total_submission_time': 0.0,
            'average_quality_score': 0.0,
            'start_time': time.time(),
            'complexity_distribution': defaultdict(int),
            'quality_trends': deque(maxlen=100),
            'validator_performance': defaultdict(list)
        }
        
        # Setup scenario-specific behavior
        self._setup_scenario()
        
        self.logger.info(f"🚀 Miner-Integrated Simulation initialized: {config.scenario} scenario")
    
    def _initialize_miner(self) -> Optional[ContinuousTrellisOrchestrator]:
        """Initialize the actual miner with simulation configuration"""
        try:
            # Create a minimal config for simulation
            miner_config = {
                'output_dir': 'simulation_outputs',
                'enable_task_tracking': True,
                'disable_task_tracking': False,
                'min_local_score': 0.5,
                'submission_timeout': 30,
                'generation_server_url': 'http://localhost:8097',
                'priority_access_max_wait': 60,
                'priority_access_check_interval': 1,
                'priority_access_timeout': 30,
                'gold_prompts_reload_interval': 3600,
                'use_vllm': False,
                'ollama_url': 'http://localhost:11434',
                'vllm_url': 'http://localhost:9000',
                'vllm_model': 'llama-3-2-3b-it'
            }
            
            # Update with custom config if provided
            if self.config.miner_config_file and Path(self.config.miner_config_file).exists():
                with open(self.config.miner_config_file, 'r') as f:
                    custom_config = json.load(f)
                    miner_config.update(custom_config)
            
            miner = ContinuousTrellisOrchestrator(miner_config)
            
            # Disable real subnet submission during simulation
            if not self.config.enable_submission:
                miner.config['enable_submission'] = False
                self.logger.info("🚫 Real subnet submission disabled for simulation")
            
            return miner
            
        except Exception as e:
            self.logger.error(f"❌ Failed to initialize miner: {e}")
            return None
    
    def _initialize_validators(self) -> Dict[int, Any]:
        """Initialize validators (real or mock)"""
        validators = {}
        
        for i in range(self.config.num_validators):
            if self.miner and MINER_AVAILABLE:
                # Create real validator state
                validator = ValidatorState(
                    uid=i,
                    hotkey=f"simulation_validator_{i}_hotkey",
                    stake=random.uniform(100.0, 1000.0),
                    trust=random.uniform(0.7, 1.0),
                    consensus=random.uniform(0.6, 0.95)
                )
                validators[i] = validator
            else:
                # Create mock validator
                validator = MockValidator(i, self.config)
                validators[i] = validator
        
        self.logger.info(f"✅ Initialized {len(validators)} validators")
        return validators
    
    def _initialize_task_database(self) -> Any:
        """Initialize task database (real or mock)"""
        if self.miner and MINER_AVAILABLE:
            return self.miner.db
        else:
            # Create mock database
            return MockTaskDatabase()
    
    def _initialize_prompt_generator(self) -> Any:
        """Initialize prompt generator"""
        return PromptGenerator(self.config)
    
    def _setup_scenario(self):
        """Configure simulation based on selected scenario"""
        if self.config.scenario == "high_load":
            self.config.task_pull_interval = 15
            self.config.prompts_per_cycle = 100
        elif self.config.scenario == "stress_test":
            self.config.task_pull_interval = 10
            self.config.prompts_per_cycle = 150
        elif self.config.scenario == "learning":
            self.config.task_pull_interval = 60
            self.config.prompts_per_cycle = 75
        elif self.config.scenario == "realistic":
            # Realistic scenario based on actual subnet behavior
            self.config.task_pull_interval = 45
            self.config.prompts_per_cycle = 80
            self.config.generation_success_rate = 0.75
            self.config.validation_success_rate = 0.85
    
    async def run_simulation(self):
        """Run the complete miner-integrated simulation"""
        self.logger.info(f"🎮 Starting Miner-Integrated Simulation")
        self.logger.info(f"   Scenario: {self.config.scenario}")
        self.logger.info(f"   Validators: {self.config.num_validators}")
        self.logger.info(f"   Duration: {self.config.simulation_duration} seconds")
        self.logger.info(f"   Real Miner: {'Yes' if self.miner else 'No'}")
        
        start_time = time.time()
        cycle_count = 0
        
        try:
            while time.time() - start_time < self.config.simulation_duration:
                cycle_count += 1
                self.logger.info(f"\n=== Simulation Cycle {cycle_count} ===")
                
                # Generate and distribute prompts
                await self._run_prompt_generation_cycle()
                
                # Wait for prompts to be available
                await asyncio.sleep(5)
                
                # Run task processing cycles
                await self._run_task_processing_cycles()
                
                # Wait before next cycle
                await asyncio.sleep(self.config.task_pull_interval)
                
                # Print detailed stats
                self._print_detailed_stats()
                
        except KeyboardInterrupt:
            self.logger.info("⏹️ Simulation interrupted by user")
        except Exception as e:
            self.logger.error(f"❌ Simulation error: {e}")
            traceback.print_exc()
        finally:
            # Print final results
            self._print_final_results()
            
            # Cleanup
            if self.miner:
                await self._cleanup_miner()
    
    async def _run_prompt_generation_cycle(self):
        """Run prompt generation and distribution cycle"""
        self.logger.info("📝 Starting prompt generation cycle")
        
        # Generate new prompts
        new_prompts = self.prompt_generator.generate_prompts(self.config.prompts_per_cycle)
        self.simulation_stats['total_tasks_generated'] += len(new_prompts)
        
        # Analyze complexity distribution
        for prompt in new_prompts:
            complexity = self._analyze_prompt_complexity(prompt)
            self.simulation_stats['complexity_distribution'][complexity] += 1
        
        # Store prompts in database for task pulling
        for prompt in new_prompts:
            self._store_prompt_for_pulling(prompt)
        
        self.logger.info(f"✅ Prompt generation cycle completed: {len(new_prompts)} prompts")
    
    async def _run_task_processing_cycles(self):
        """Run task processing cycles for all validators"""
        self.logger.info("🔄 Starting task processing cycles")
        
        # Create tasks for each validator
        tasks = []
        for validator in self.validators.values():
            if self._is_validator_available(validator):
                task = await self._create_task_for_validator(validator)
                if task:
                    tasks.append(task)
        
        if not tasks:
            self.logger.info("   No tasks created for processing")
            return
        
        self.logger.info(f"   Created {len(tasks)} tasks for processing")
        
        # Process tasks concurrently
        processing_tasks = [
            self._process_single_task(task) for task in tasks
        ]
        
        results = await asyncio.gather(*processing_tasks, return_exceptions=True)
        
        # Process results
        successful_tasks = 0
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                self.logger.error(f"Task {i} processing error: {result}")
            elif result:
                successful_tasks += 1
                self._update_simulation_stats_from_task(result)
        
        self.logger.info(f"✅ Task processing cycles completed: {successful_tasks}/{len(tasks)} successful")
    
    async def _create_task_for_validator(self, validator: Any) -> Optional[Any]:
        """Create a task for a specific validator"""
        try:
            # Get a prompt from the database
            prompt = self._get_available_prompt()
            if not prompt:
                return None
            
            # Create task record
            if self.miner and MINER_AVAILABLE:
                task = TaskRecord(
                    task_id=f"sim_task_{int(time.time())}_{validator.uid}",
                    prompt=prompt,
                    prompt_hash=self._generate_hash(prompt),
                    validator_uid=validator.uid,
                    validator_hotkey=validator.hotkey,
                    validator_stake=validator.stake,
                    validation_threshold=0.6,
                    pulled_at=time.time()
                )
            else:
                task = MockTaskRecord(
                    task_id=f"sim_task_{int(time.time())}_{validator.uid}",
                    prompt=prompt,
                    validator_uid=validator.uid,
                    validator_hotkey=validator.hotkey,
                    validator_stake=validator.stake
                )
            
            # Update validator stats
            validator.stats['tasks_pulled'] += 1
            validator.last_task_pull = time.time()
            validator.last_task_received = time.time()
            
            self.simulation_stats['total_tasks_pulled'] += 1
            
            return task
            
        except Exception as e:
            self.logger.error(f"Error creating task for validator {validator.uid}: {e}")
            return None
    
    async def _process_single_task(self, task: Any) -> Optional[Any]:
        """Process a single task through the complete pipeline"""
        try:
            self.logger.info(f"🔄 Processing task {task.task_id}: '{task.prompt[:50]}...'")
            
            start_time = time.time()
            
            # Step 1: Generate 3D model
            generation_result = await self._simulate_3d_generation(task)
            if not generation_result:
                self.logger.warning(f"❌ Generation failed for task {task.task_id}")
                return None
            
            generation_time = time.time() - start_time
            task.generation_time = generation_time
            self.simulation_stats['total_generation_time'] += generation_time
            
            # Step 2: Validate model
            validation_start = time.time()
            validation_result = await self._simulate_model_validation(task, generation_result)
            if not validation_result:
                self.logger.warning(f"❌ Validation failed for task {task.task_id}")
                return None
            
            validation_time = time.time() - validation_start
            task.validation_time = validation_time
            task.local_validation_score = validation_result['final_score']
            self.simulation_stats['total_validation_time'] += validation_time
            
            # Step 3: Submit result (if enabled)
            if self.config.enable_submission:
                submission_start = time.time()
                submission_success = await self._simulate_result_submission(task, generation_result)
                submission_time = time.time() - submission_start
                self.simulation_stats['total_submission_time'] += submission_time
                
                if submission_success:
                    task.submission_success = True
                    task.submitted_at = time.time()
                    self.simulation_stats['total_tasks_submitted'] += 1
                else:
                    self.logger.warning(f"❌ Submission failed for task {task.task_id}")
            else:
                # Simulate successful submission for statistics
                task.submission_success = True
                task.submitted_at = time.time()
                self.simulation_stats['total_tasks_submitted'] += 1
            
            # Calculate total processing time
            task.total_processing_time = time.time() - start_time
            task.processed_at = time.time()
            
            # Update validator stats
            validator = self.validators.get(task.validator_uid)
            if validator:
                validator.stats['tasks_processed'] += 1
                if task.submission_success:
                    validator.stats['tasks_submitted'] += 1
                if task.local_validation_score:
                    validator.stats['total_score'] += task.local_validation_score
            
            self.simulation_stats['total_tasks_processed'] += 1
            
            self.logger.info(f"✅ Task {task.task_id} completed successfully")
            return task
            
        except Exception as e:
            self.logger.error(f"❌ Task processing error: {e}")
            traceback.print_exc()
            return None
    
    async def _simulate_3d_generation(self, task: Any) -> Optional[Dict[str, Any]]:
        """Simulate 3D model generation"""
        try:
            # Check success rate
            if random.random() > self.config.generation_success_rate:
                return None
            
            # Simulate generation time
            generation_time = random.uniform(*self.config.generation_time_range)
            await asyncio.sleep(generation_time)
            
            # Generate mock PLY data
            ply_data = f"mock_ply_data_for_task_{task.task_id}".encode()
            
            # Create generation result
            result = {
                'ply_data': ply_data,
                'generation_time': generation_time,
                'status': 'success',
                'model_path': f"simulation_outputs/{task.task_id}.ply",
                'metadata': {
                    'prompt': task.prompt,
                    'validator_uid': task.validator_uid,
                    'timestamp': time.time()
                }
            }
            
            return result
            
        except Exception as e:
            self.logger.error(f"Generation simulation error: {e}")
            return None
    
    async def _simulate_model_validation(self, task: Any, generation_result: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Simulate model validation"""
        try:
            # Check success rate
            if random.random() > self.config.validation_success_rate:
                return None
            
            # Simulate validation time
            validation_time = random.uniform(*self.config.validation_time_range)
            await asyncio.sleep(validation_time)
            
            # Generate quality scores
            quality_score = random.uniform(*self.config.quality_score_range)
            alignment_score = random.uniform(*self.config.alignment_score_range)
            ssim_score = random.uniform(0.4, 0.85)
            lpips_score = random.uniform(0.3, 0.8)
            
            # Calculate final score
            final_score = (
                quality_score * 0.4 +
                alignment_score * 0.3 +
                ssim_score * 0.2 +
                lpips_score * 0.1
            )
            
            result = {
                'final_score': final_score,
                'quality_score': quality_score,
                'alignment_score': alignment_score,
                'ssim_score': ssim_score,
                'lpips_score': lpips_score,
                'validation_time': validation_time
            }
            
            return result
            
        except Exception as e:
            self.logger.error(f"Validation simulation error: {e}")
            return None
    
    async def _simulate_result_submission(self, task: Any, generation_result: Dict[str, Any]) -> bool:
        """Simulate result submission to subnet"""
        try:
            # Check success rate
            if random.random() > self.config.submission_success_rate:
                return False
            
            # Simulate submission time
            submission_time = random.uniform(*self.config.submission_time_range)
            await asyncio.sleep(submission_time)
            
            # Simulate feedback
            task.feedback_received = True
            task.task_fidelity_score = random.uniform(0.6, 0.95)
            task.average_fidelity_score = random.uniform(0.5, 0.9)
            task.current_miner_reward = random.uniform(0.1, 1.0)
            
            return True
            
        except Exception as e:
            self.logger.error(f"Submission simulation error: {e}")
            return False
    
    def _is_validator_available(self, validator: Any) -> bool:
        """Check if validator is available for task processing"""
        if not validator.is_active:
            return False
        
        if hasattr(validator, 'cooldown_until') and validator.cooldown_until:
            if time.time() < validator.cooldown_until:
                return False
        
        if hasattr(validator, 'throttle_period') and validator.throttle_period > 0:
            return False
        
        return True
    
    def _store_prompt_for_pulling(self, prompt: str):
        """Store prompt in database for task pulling"""
        # This would integrate with the actual task database
        pass
    
    def _get_available_prompt(self) -> Optional[str]:
        """Get an available prompt for task creation"""
        # This would integrate with the actual prompt system
        return f"simulated prompt {int(time.time())}"
    
    def _generate_hash(self, prompt: str) -> str:
        """Generate hash for prompt"""
        import hashlib
        return hashlib.sha256(prompt.encode()).hexdigest()
    
    def _analyze_prompt_complexity(self, prompt: str) -> str:
        """Analyze prompt complexity"""
        words = prompt.split()
        if len(words) <= 5:
            return "simple"
        elif len(words) <= 8:
            return "medium"
        else:
            return "complex"
    
    def _update_simulation_stats_from_task(self, task: Any):
        """Update simulation statistics from completed task"""
        if hasattr(task, 'local_validation_score') and task.local_validation_score:
            self.simulation_stats['quality_trends'].append({
                'timestamp': time.time(),
                'score': task.local_validation_score,
                'task_id': task.task_id
            })
            
            # Update average quality score
            scores = [t['score'] for t in self.simulation_stats['quality_trends']]
            if scores:
                self.simulation_stats['average_quality_score'] = statistics.mean(scores)
    
    def _print_detailed_stats(self):
        """Print detailed simulation statistics"""
        elapsed = time.time() - self.simulation_stats['start_time']
        
        print(f"\n📊 Miner-Integrated Simulation Statistics (Cycle {elapsed:.0f}s)")
        print(f"   Tasks Generated: {self.simulation_stats['total_tasks_generated']}")
        print(f"   Tasks Pulled: {self.simulation_stats['total_tasks_pulled']}")
        print(f"   Tasks Processed: {self.simulation_stats['total_tasks_processed']}")
        print(f"   Tasks Submitted: {self.simulation_stats['total_tasks_submitted']}")
        print(f"   Avg Quality Score: {self.simulation_stats['average_quality_score']:.3f}")
        
        # Complexity distribution
        print(f"   Complexity Distribution:")
        for complexity, count in self.simulation_stats['complexity_distribution'].items():
            print(f"     {complexity.capitalize()}: {count}")
        
        # Validator performance
        print(f"   Validator Performance:")
        for validator in self.validators.values():
            if hasattr(validator, 'stats'):
                stats = validator.stats
                print(f"     Validator {validator.uid}: "
                      f"Pulled: {stats['tasks_pulled']}, "
                      f"Processed: {stats['tasks_processed']}, "
                      f"Submitted: {stats['tasks_submitted']}")
    
    def _print_final_results(self):
        """Print comprehensive final results"""
        total_time = time.time() - self.simulation_stats['start_time']
        
        print(f"\n🎯 MINER-INTEGRATED SIMULATION COMPLETED")
        print(f"=" * 60)
        print(f"Scenario: {self.config.scenario}")
        print(f"Total Runtime: {total_time:.1f} seconds")
        print(f"Tasks Generated: {self.simulation_stats['total_tasks_generated']}")
        print(f"Tasks Pulled: {self.simulation_stats['total_tasks_pulled']}")
        print(f"Tasks Processed: {self.simulation_stats['total_tasks_processed']}")
        print(f"Tasks Submitted: {self.simulation_stats['total_tasks_submitted']}")
        print(f"Overall Average Quality Score: {self.simulation_stats['average_quality_score']:.3f}")
        
        # Performance metrics
        print(f"\n📈 Performance Metrics:")
        print(f"   Tasks per Second: {self.simulation_stats['total_tasks_processed']/total_time:.2f}")
        print(f"   Generation Success Rate: {self.simulation_stats['total_tasks_processed']/max(self.simulation_stats['total_tasks_pulled'], 1):.2%}")
        print(f"   Validation Success Rate: {self.simulation_stats['total_tasks_processed']/max(self.simulation_stats['total_tasks_pulled'], 1):.2%}")
        print(f"   Submission Success Rate: {self.simulation_stats['total_tasks_submitted']/max(self.simulation_stats['total_tasks_processed'], 1):.2%}")
        
        # Timing analysis
        if self.simulation_stats['total_tasks_processed'] > 0:
            avg_generation_time = self.simulation_stats['total_generation_time'] / self.simulation_stats['total_tasks_processed']
            avg_validation_time = self.simulation_stats['total_validation_time'] / self.simulation_stats['total_tasks_processed']
            print(f"   Average Generation Time: {avg_generation_time:.2f}s")
            print(f"   Average Validation Time: {avg_validation_time:.2f}s")
        
        # Save comprehensive results
        results = {
            'simulation_config': self.config.__dict__,
            'simulation_stats': dict(self.simulation_stats),
            'validator_stats': [v.stats if hasattr(v, 'stats') else {} for v in self.validators.values()],
            'complexity_distribution': dict(self.simulation_stats['complexity_distribution']),
            'quality_trends': list(self.simulation_stats['quality_trends']),
            'timestamp': time.time()
        }
        
        with open('miner_integrated_simulation_results.json', 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\nResults saved to: miner_integrated_simulation_results.json")
    
    async def _cleanup_miner(self):
        """Cleanup miner resources"""
        if self.miner:
            try:
                # Save validator states
                if hasattr(self.miner, 'save_validator_states_to_disk'):
                    self.miner.save_validator_states_to_disk()
                
                # Close database connections
                if hasattr(self.miner.db, 'close'):
                    self.miner.db.close()
                
                self.logger.info("✅ Miner cleanup completed")
            except Exception as e:
                self.logger.error(f"❌ Miner cleanup error: {e}")


class PromptGenerator:
    """Generates prompts for simulation"""
    
    def __init__(self, config: MinerSimulationConfig):
        self.config = config
        self.categories = {
            "robots": ["mechanical", "android", "cybernetic", "automated"],
            "animals": ["majestic", "wild", "domestic", "mythical"],
            "characters": ["fantasy", "sci-fi", "heroic", "mystical"],
            "objects": ["detailed", "intricate", "elaborate", "complex"]
        }
    
    def generate_prompts(self, count: int) -> List[str]:
        """Generate diverse prompts"""
        prompts = []
        
        for i in range(count):
            category = random.choice(list(self.categories.keys()))
            descriptors = self.categories[category]
            
            if category == "robots":
                prompt = f"{random.choice(descriptors)} {category} with {random.choice(['steel', 'chrome', 'copper'])} plating"
            elif category == "animals":
                prompt = f"{random.choice(descriptors)} {category} in {random.choice(['natural', 'dramatic', 'peaceful'])} pose"
            elif category == "characters":
                prompt = f"{random.choice(descriptors)} {category} character with {random.choice(['magical', 'mystical', 'enchanted'])} aura"
            else:
                prompt = f"{random.choice(descriptors)} {category} with {random.choice(['intricate', 'elaborate', 'complex'])} design"
            
            prompts.append(prompt)
        
        return prompts


class MockTaskDatabase:
    """Mock task database for simulation"""
    
    def __init__(self):
        self.tasks = []
        self.prompts = []
    
    def add_recent_prompt(self, prompt: str, validator_uid: int):
        """Add recent prompt"""
        self.prompts.append((prompt, validator_uid, time.time()))
    
    def save_task(self, task: Any):
        """Save task"""
        self.tasks.append(task)


async def main():
    """Main entry point for miner-integrated simulation"""
    parser = argparse.ArgumentParser(description="Miner-Integrated Validator Simulation")
    parser.add_argument("--scenario", choices=["balanced", "high_load", "stress_test", "learning", "realistic"], 
                       default="balanced", help="Simulation scenario")
    parser.add_argument("--validators", type=int, default=5, help="Number of validators")
    parser.add_argument("--duration", type=int, default=600, help="Simulation duration in seconds")
    parser.add_argument("--miner-config", help="Path to miner configuration file")
    parser.add_argument("--enable-submission", action="store_true", help="Enable real subnet submission")
    
    args = parser.parse_args()
    
    # Create configuration
    config = MinerSimulationConfig(
        scenario=args.scenario,
        num_validators=args.validators,
        simulation_duration=args.duration,
        miner_config_file=args.miner_config,
        enable_submission=args.enable_submission
    )
    
    # Create and run simulation
    simulation = MinerIntegratedSimulation(config)
    
    try:
        await simulation.run_simulation()
    except KeyboardInterrupt:
        logger.info("Miner-integrated simulation interrupted by user")
    except Exception as e:
        logger.error(f"Miner-integrated simulation error: {e}")
        traceback.print_exc()
        raise


if __name__ == "__main__":
    asyncio.run(main())
