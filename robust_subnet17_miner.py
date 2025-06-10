#!/usr/bin/env python3
# Enhanced Robust Subnet 17 Miner
# Advanced error recovery, task prioritization, and competitive analysis

import asyncio
import base64
import json
import time
import traceback
import random
import signal
import sys
import os
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List, Set, Tuple
from dataclasses import dataclass, field
from pathlib import Path
from collections import defaultdict, deque
import statistics

import bittensor as bt
import aiohttp
import pyspz

# Enhanced Configuration
NETUID: int = 17
WALLET_NAME: str = "default"
HOTKEY_NAME: str = "default"

# Adaptive Timing Configuration
class AdaptiveConfig:
    def __init__(self):
        self.generation_endpoint_urls = ["http://127.0.0.1:8093/generate/"]
        self.local_validation_url = "http://127.0.0.1:8094/validate_txt_to_3d_ply/"
        
        # Adaptive timeouts based on historical performance
        self.base_generation_timeout = 300.0
        self.base_validation_timeout = 60.0
        self.base_submission_timeout = 30.0
        
        # Quality and competition thresholds
        self.min_local_validation_score = 0.7
        self.competitive_score_threshold = 0.8  # Aim for top tier
        self.max_retries = 3
        
        # Resource management
        self.max_concurrent_generations = 2
        self.max_task_queue_size = 50
        self.memory_cleanup_interval = 300  # 5 minutes
        
        # Intelligent task selection
        self.validator_success_weight = 0.3
        self.validator_stake_weight = 0.4
        self.validator_response_time_weight = 0.3
        
    def get_generation_timeout(self, avg_time: float) -> float:
        """Get adaptive generation timeout based on historical performance"""
        return max(self.base_generation_timeout, avg_time * 2.5)
        
    def should_accept_task(self, validator_uid: int, validator_performance: Dict) -> bool:
        """Decide whether to accept a task from a validator based on historical performance"""
        if not validator_performance:
            return True
            
        success_rate = validator_performance.get('success_rate', 1.0)
        avg_score = validator_performance.get('avg_score', 0.5)
        
        # Accept if validator has good success rate and provides competitive scoring
        return success_rate > 0.7 and avg_score > 0.6

config = AdaptiveConfig()

@dataclass
class TaskPriority:
    """Priority scoring for tasks"""
    validator_uid: int
    validator_stake: float
    validator_success_rate: float
    validator_avg_response_time: float
    task_complexity_score: float  # Based on prompt analysis
    priority_score: float = 0.0
    
    def __post_init__(self):
        self.calculate_priority()
        
    def calculate_priority(self):
        """Calculate priority score for task selection"""
        # Normalize and weight factors
        stake_score = min(self.validator_stake / 10000.0, 1.0)  # Normalize to 10k stake
        success_score = self.validator_success_rate
        speed_score = max(0, 1.0 - (self.validator_avg_response_time / 60.0))  # Normalize to 1 minute
        
        self.priority_score = (
            stake_score * config.validator_stake_weight +
            success_score * config.validator_success_weight +
            speed_score * config.validator_response_time_weight
        )

@dataclass
class ValidatorPerformance:
    """Track validator performance metrics"""
    uid: int
    hotkey: str
    stake: float
    tasks_received: int = 0
    tasks_completed: int = 0
    tasks_failed: int = 0
    total_score: float = 0.0
    response_times: deque = field(default_factory=lambda: deque(maxlen=100))
    last_interaction: float = 0.0
    
    @property
    def success_rate(self) -> float:
        if self.tasks_received == 0:
            return 1.0
        return self.tasks_completed / self.tasks_received
    
    @property
    def avg_score(self) -> float:
        if self.tasks_completed == 0:
            return 0.0
        return self.total_score / self.tasks_completed
    
    @property
    def avg_response_time(self) -> float:
        if not self.response_times:
            return 30.0  # Default assumption
        return statistics.mean(self.response_times)

@dataclass
class EnhancedMiningTask:
    """Enhanced task with priority and performance tracking"""
    task_id: str
    prompt: str
    validator_hotkey: str
    validator_uid: int
    assignment_time: float
    validation_threshold: float
    throttle_period: float
    priority: TaskPriority
    retry_count: int = 0
    generation_attempts: List[float] = field(default_factory=list)
    validation_scores: List[float] = field(default_factory=list)

class CompetitiveAnalyzer:
    """Analyze competitive landscape and optimize strategy"""
    
    def __init__(self):
        self.score_history: Dict[int, List[float]] = defaultdict(list)  # validator_uid -> scores
        self.time_history: Dict[int, List[float]] = defaultdict(list)   # validator_uid -> times
        self.market_analysis = {
            'avg_score': 0.0,
            'top_10_percent_score': 0.0,
            'score_volatility': 0.0,
            'time_advantage_threshold': 0.0
        }
        
    def record_submission(self, validator_uid: int, score: float, generation_time: float):
        """Record a submission result for competitive analysis"""
        self.score_history[validator_uid].append(score)
        self.time_history[validator_uid].append(generation_time)
        
        # Keep only recent history (last 1000 submissions per validator)
        if len(self.score_history[validator_uid]) > 1000:
            self.score_history[validator_uid] = self.score_history[validator_uid][-1000:]
        if len(self.time_history[validator_uid]) > 1000:
            self.time_history[validator_uid] = self.time_history[validator_uid][-1000:]
            
        self.update_market_analysis()
        
    def update_market_analysis(self):
        """Update overall market analysis"""
        all_scores = []
        all_times = []
        
        for scores in self.score_history.values():
            all_scores.extend(scores[-100:])  # Recent scores only
        for times in self.time_history.values():
            all_times.extend(times[-100:])
            
        if all_scores:
            sorted_scores = sorted(all_scores, reverse=True)
            self.market_analysis['avg_score'] = statistics.mean(all_scores)
            
            # Top 10% threshold
            top_10_count = max(1, len(sorted_scores) // 10)
            self.market_analysis['top_10_percent_score'] = statistics.mean(sorted_scores[:top_10_count])
            
            # Score volatility
            if len(all_scores) > 1:
                self.market_analysis['score_volatility'] = statistics.stdev(all_scores)
                
        if all_times:
            self.market_analysis['time_advantage_threshold'] = statistics.median(all_times) * 0.8
            
    def should_prioritize_quality_over_speed(self) -> bool:
        """Determine if we should prioritize quality over speed based on market conditions"""
        # If market is highly competitive (low volatility, high scores), prioritize quality
        if (self.market_analysis['score_volatility'] < 0.1 and 
            self.market_analysis['avg_score'] > 0.75):
            return True
            
        # If we're consistently below market average, prioritize quality
        recent_scores = []
        for scores in self.score_history.values():
            recent_scores.extend(scores[-10:])  # Very recent scores
            
        if recent_scores:
            our_avg = statistics.mean(recent_scores)
            if our_avg < self.market_analysis['avg_score'] - 0.05:
                return True
                
        return False
        
    def get_competitive_recommendation(self) -> Dict[str, Any]:
        """Get competitive strategy recommendation"""
        return {
            'prioritize_quality': self.should_prioritize_quality_over_speed(),
            'target_score': max(0.8, self.market_analysis['top_10_percent_score'] - 0.02),
            'max_acceptable_time': self.market_analysis['time_advantage_threshold'],
            'market_competitiveness': 'high' if self.market_analysis['score_volatility'] < 0.1 else 'medium'
        }

# Global state with enhanced tracking
validator_performance: Dict[str, ValidatorPerformance] = {}
task_queue = asyncio.PriorityQueue()
submission_queue = asyncio.Queue()
competitive_analyzer = CompetitiveAnalyzer()
shutdown_event = asyncio.Event()

# Bittensor objects
subtensor: Optional[bt.subtensor] = None
dendrite: Optional[bt.dendrite] = None
wallet: Optional[bt.wallet] = None
metagraph: Optional[bt.metagraph] = None

# Enhanced synapse definitions
class PullTask(bt.Synapse):
    task: Optional[Dict[str, Any]] = None
    validation_threshold: float = 0.8
    throttle_period: float = 60.0
    cooldown_until: Optional[float] = None

class SubmitResults(bt.Synapse):
    task_id: str
    miner_hotkey: str = ""
    compressed_result: str = ""
    validator_hotkey: str = ""

async def enhanced_generate_3d_model(prompt: str, task: EnhancedMiningTask) -> Optional[bytes]:
    """Enhanced generation with adaptive timeouts and quality optimization"""
    
    # Get competitive recommendation
    competitive_rec = competitive_analyzer.get_competitive_recommendation()
    
    # Choose generation endpoint (could implement load balancing here)
    generation_endpoint = random.choice(config.generation_endpoint_urls)
    
    # Calculate adaptive timeout
    validator_perf = validator_performance.get(task.validator_hotkey, ValidatorPerformance(task.validator_uid, task.validator_hotkey, 0))
    timeout = config.get_generation_timeout(validator_perf.avg_response_time)
    
    bt.logging.info(f"Generating with competitive focus: {'QUALITY' if competitive_rec['prioritize_quality'] else 'SPEED'}")
    bt.logging.info(f"Target score: {competitive_rec['target_score']:.3f}, Max time: {competitive_rec.get('max_acceptable_time', 120):.1f}s")
    
    payload = {
        "prompt": prompt,
        "quality_mode": competitive_rec['prioritize_quality'],
        "target_score": competitive_rec['target_score']
    }
    
    start_time = time.time()
    
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(
                generation_endpoint,
                data=payload,
                timeout=timeout
            ) as response:
                if response.status == 200:
                    raw_ply_bytes = await response.read()
                    generation_time = time.time() - start_time
                    
                    # Record generation attempt
                    task.generation_attempts.append(generation_time)
                    
                    if len(raw_ply_bytes) < 100:
                        raise ValueError("Generated PLY data too small")
                        
                    bt.logging.success(f"Generated PLY in {generation_time:.2f}s. Size: {len(raw_ply_bytes)} bytes.")
                    return raw_ply_bytes
                    
                else:
                    error_text = await response.text()
                    bt.logging.error(f"Generation failed. Status: {response.status}. Error: {error_text}")
                    return None
                    
    except asyncio.TimeoutError:
        generation_time = time.time() - start_time
        task.generation_attempts.append(generation_time)
        bt.logging.error(f"Generation timeout after {generation_time:.2f}s for prompt: '{prompt}'")
        return None
        
    except Exception as e:
        generation_time = time.time() - start_time
        task.generation_attempts.append(generation_time)
        bt.logging.error(f"Generation exception after {generation_time:.2f}s: {e}")
        return None

async def enhanced_validate_locally(prompt: str, compressed_ply_bytes: bytes, task: EnhancedMiningTask) -> float:
    """Enhanced local validation with competitive analysis"""
    
    try:
        base64_compressed_data = base64.b64encode(compressed_ply_bytes).decode('utf-8')
        payload = {
            "prompt": prompt,
            "data": base64_compressed_data,
            "compression": 2,
            "data_ver": 0,
            "generate_preview": False,
            "detailed_metrics": True  # Request detailed quality metrics
        }
        
        async with aiohttp.ClientSession() as session:
            async with session.post(
                config.local_validation_url,
                json=payload,
                timeout=config.base_validation_timeout
            ) as response:
                if response.status == 200:
                    validation_result = await response.json()
                    score = validation_result.get("score", 0.0)
                    
                    # Record validation score
                    task.validation_scores.append(score)
                    
                    # Enhanced validation with detailed metrics
                    if "detailed_metrics" in validation_result:
                        metrics = validation_result["detailed_metrics"]
                        bt.logging.info(f"Detailed validation - Score: {score:.3f}, "
                                      f"Geometry: {metrics.get('geometry_score', 0):.3f}, "
                                      f"Texture: {metrics.get('texture_score', 0):.3f}, "
                                      f"Alignment: {metrics.get('alignment_score', 0):.3f}")
                    
                    if not 0 <= score <= 1:
                        raise ValueError(f"Invalid validation score: {score}")
                        
                    return score
                else:
                    error_text = await response.text()
                    bt.logging.warning(f"Local validation HTTP error: Status {response.status}, {error_text}")
                    return -1.0
                    
    except Exception as e:
        bt.logging.error(f"Error during local validation: {e}")
        return -1.0

async def smart_task_processor():
    """Enhanced task processor with priority queue and intelligent selection"""
    bt.logging.info("Smart task processor started.")
    
    while not shutdown_event.is_set():
        try:
            # Get highest priority task
            priority_score, task = await asyncio.wait_for(task_queue.get(), timeout=1.0)
            
            # Double-check if we should still process this task
            validator_perf = validator_performance.get(task.validator_hotkey)
            if validator_perf and not config.should_accept_task(task.validator_uid, {
                'success_rate': validator_perf.success_rate,
                'avg_score': validator_perf.avg_score
            }):
                bt.logging.warning(f"Skipping task from underperforming validator UID {task.validator_uid}")
                continue
            
            # Process the task
            await process_mining_task(task)
            
        except asyncio.TimeoutError:
            continue
        except Exception as e:
            bt.logging.error(f"Error in smart task processor: {e}\n{traceback.format_exc()}")
            await asyncio.sleep(1)

async def process_mining_task(task: EnhancedMiningTask):
    """Process a mining task with enhanced error handling and quality optimization"""
    
    start_time = time.time()
    bt.logging.info(f"Processing task {task.task_id} from validator UID {task.validator_uid}: '{task.prompt[:50]}...'")
    
    # Update validator performance tracking
    validator_perf = validator_performance.get(task.validator_hotkey)
    if validator_perf:
        validator_perf.tasks_received += 1
        validator_perf.last_interaction = time.time()
    
    try:
        # Generate 3D model
        raw_ply_bytes = await enhanced_generate_3d_model(task.prompt, task)
        if not raw_ply_bytes:
            raise ValueError("Failed to generate 3D model")
        
        # Compress PLY data
        compressed_ply_bytes = pyspz.compress(raw_ply_bytes)
        bt.logging.info(f"Compressed PLY from {len(raw_ply_bytes)} to {len(compressed_ply_bytes)} bytes")
        
        # Local validation
        local_score = await enhanced_validate_locally(task.prompt, compressed_ply_bytes, task)
        
        if local_score < 0:
            raise ValueError("Local validation failed")
        
        # Check if score meets competitive threshold
        competitive_rec = competitive_analyzer.get_competitive_recommendation()
        min_score = max(config.min_local_validation_score, competitive_rec.get('target_score', 0.8) - 0.05)
        
        if local_score < min_score:
            bt.logging.warning(f"Local score {local_score:.3f} below competitive threshold {min_score:.3f}")
            
            # Retry with higher quality settings if we have retries left
            if task.retry_count < config.max_retries:
                task.retry_count += 1
                bt.logging.info(f"Retrying task {task.task_id} with higher quality settings (attempt {task.retry_count})")
                
                # Re-queue with higher priority for quality retry
                priority = (-1000, task)  # Very high priority for retries
                await task_queue.put(priority)
                return
            else:
                bt.logging.warning(f"Submitting below-threshold score after {config.max_retries} retries")
        
        # Prepare submission
        base64_compressed_data = base64.b64encode(compressed_ply_bytes).decode('utf-8')
        
        submission_data = {
            'task': task,
            'compressed_data': base64_compressed_data,
            'local_score': local_score,
            'generation_time': time.time() - start_time
        }
        
        await submission_queue.put(submission_data)
        
        bt.logging.success(f"Task {task.task_id} queued for submission. Local score: {local_score:.3f}")
        
        # Record successful completion
        if validator_perf:
            validator_perf.tasks_completed += 1
            validator_perf.total_score += local_score
            validator_perf.response_times.append(time.time() - start_time)
        
    except Exception as e:
        bt.logging.error(f"Failed to process task {task.task_id}: {e}")
        
        # Record failure
        if validator_perf:
            validator_perf.tasks_failed += 1

async def enhanced_submission_worker():
    """Enhanced submission worker with retry logic and performance tracking"""
    bt.logging.info("Enhanced submission worker started.")
    
    while not shutdown_event.is_set():
        try:
            submission_data = await asyncio.wait_for(submission_queue.get(), timeout=1.0)
            
            task = submission_data['task']
            compressed_data = submission_data['compressed_data']
            local_score = submission_data['local_score']
            generation_time = submission_data['generation_time']
            
            # Submit to validator
            success = await submit_to_validator(task, compressed_data)
            
            if success:
                # Record successful submission for competitive analysis
                competitive_analyzer.record_submission(
                    task.validator_uid, 
                    local_score, 
                    generation_time
                )
                bt.logging.success(f"Successfully submitted task {task.task_id}")
            else:
                bt.logging.error(f"Failed to submit task {task.task_id}")
                
        except asyncio.TimeoutError:
            continue
        except Exception as e:
            bt.logging.error(f"Error in submission worker: {e}\n{traceback.format_exc()}")
            await asyncio.sleep(1)

async def submit_to_validator(task: EnhancedMiningTask, compressed_data: str) -> bool:
    """Submit result to validator with enhanced error handling"""
    
    try:
        # Get validator info
        validator_perf = validator_performance.get(task.validator_hotkey)
        if not validator_perf:
            bt.logging.error(f"No performance data for validator {task.validator_hotkey}")
            return False
        
        # Create target axon
        if not metagraph:
            bt.logging.error("Metagraph not available for submission")
            return False
            
        target_axon = None
        for neuron in metagraph.neurons:
            if neuron.hotkey == task.validator_hotkey:
                target_axon = neuron.axon
                break
                
        if not target_axon:
            bt.logging.error(f"Could not find axon for validator {task.validator_hotkey}")
            return False
        
        # Create submission synapse
        submit_synapse = SubmitResults(
            task_id=task.task_id,
            miner_hotkey=wallet.hotkey.ss58_address,
            compressed_result=compressed_data,
            validator_hotkey=task.validator_hotkey
        )
        
        # Submit with adaptive timeout
        timeout = config.base_submission_timeout
        response_synapse = await dendrite.call(
            target_axon=target_axon,
            synapse=submit_synapse,
            timeout=timeout,
            deserialize=True
        )
        
        if response_synapse.dendrite.status_code == 200:
            bt.logging.success(f"Submission successful for task {task.task_id}")
            return True
        else:
            bt.logging.error(f"Submission failed: {response_synapse.dendrite.status_code} - {response_synapse.dendrite.status_message}")
            return False
            
    except Exception as e:
        bt.logging.error(f"Exception during submission: {e}")
        return False

async def main():
    """Enhanced main function with competitive analysis and performance optimization"""
    global subtensor, dendrite, wallet, metagraph
    
    # Initialize Bittensor components
    bt.logging.info("Initializing Enhanced Subnet 17 Miner...")
    
    config_bt = bt.config()
    wallet = bt.wallet(name=WALLET_NAME, hotkey=HOTKEY_NAME)
    subtensor = bt.subtensor(config=config_bt)
    dendrite = bt.dendrite(wallet=wallet)
    metagraph = subtensor.metagraph(NETUID)
    
    bt.logging.info(f"Miner initialized with hotkey: {wallet.hotkey.ss58_address}")
    
    # Start enhanced background tasks
    tasks = [
        asyncio.create_task(enhanced_metagraph_syncer()),
        asyncio.create_task(intelligent_task_puller()),
        asyncio.create_task(smart_task_processor()),
        asyncio.create_task(enhanced_submission_worker()),
        asyncio.create_task(performance_monitor()),
        asyncio.create_task(competitive_analysis_reporter())
    ]
    
    # Wait for shutdown
    try:
        await asyncio.gather(*tasks)
    except KeyboardInterrupt:
        bt.logging.info("Shutting down Enhanced Subnet 17 Miner...")
        shutdown_event.set()
        
        # Cancel all tasks
        for task in tasks:
            task.cancel()
            
        await asyncio.gather(*tasks, return_exceptions=True)

async def enhanced_metagraph_syncer():
    """Enhanced metagraph syncer with validator performance tracking"""
    global metagraph, validator_performance
    
    while not shutdown_event.is_set():
        try:
            if subtensor and subtensor.is_connected():
                old_metagraph = metagraph
                metagraph = subtensor.metagraph(NETUID)
                
                # Update validator performance tracking
                for neuron in metagraph.neurons:
                    if neuron.stake >= 1000.0:  # Only track validators with significant stake
                        hotkey = neuron.hotkey
                        if hotkey not in validator_performance:
                            validator_performance[hotkey] = ValidatorPerformance(
                                uid=neuron.uid,
                                hotkey=hotkey,
                                stake=neuron.stake
                            )
                        else:
                            # Update stake information
                            validator_performance[hotkey].stake = neuron.stake
                
                bt.logging.info(f"Metagraph synced. Tracking {len(validator_performance)} validators.")
                
            else:
                bt.logging.warning("Subtensor not connected. Retrying...")
                
        except Exception as e:
            bt.logging.error(f"Error in enhanced metagraph syncer: {e}")
            
        await asyncio.sleep(300)  # 5 minutes

async def intelligent_task_puller():
    """Intelligent task puller that prioritizes validators based on performance"""
    bt.logging.info("Intelligent task puller started.")
    
    while not shutdown_event.is_set():
        try:
            if not validator_performance:
                await asyncio.sleep(10)
                continue
            
            # Sort validators by priority
            sorted_validators = sorted(
                validator_performance.items(),
                key=lambda x: TaskPriority(
                    validator_uid=x[1].uid,
                    validator_stake=x[1].stake,
                    validator_success_rate=x[1].success_rate,
                    validator_avg_response_time=x[1].avg_response_time,
                    task_complexity_score=0.5  # Will be updated based on actual prompts
                ).priority_score,
                reverse=True
            )
            
            # Pull from top validators
            for hotkey, validator_perf in sorted_validators[:10]:  # Top 10 validators
                if task_queue.qsize() >= config.max_task_queue_size:
                    break
                    
                try:
                    # Pull task logic here (similar to original but with priority)
                    task = await pull_task_from_validator(hotkey, validator_perf)
                    if task:
                        # Calculate priority and add to queue
                        priority_obj = TaskPriority(
                            validator_uid=validator_perf.uid,
                            validator_stake=validator_perf.stake,
                            validator_success_rate=validator_perf.success_rate,
                            validator_avg_response_time=validator_perf.avg_response_time,
                            task_complexity_score=analyze_prompt_complexity(task.prompt)
                        )
                        
                        enhanced_task = EnhancedMiningTask(
                            task_id=task.task_id,
                            prompt=task.prompt,
                            validator_hotkey=hotkey,
                            validator_uid=validator_perf.uid,
                            assignment_time=time.time(),
                            validation_threshold=0.8,  # Default
                            throttle_period=60.0,  # Default
                            priority=priority_obj
                        )
                        
                        # Add to priority queue (negative score for max priority queue)
                        priority = (-priority_obj.priority_score, enhanced_task)
                        await task_queue.put(priority)
                        
                except Exception as e:
                    bt.logging.error(f"Error pulling from validator {hotkey}: {e}")
                    
            await asyncio.sleep(5)  # Check every 5 seconds
            
        except Exception as e:
            bt.logging.error(f"Error in intelligent task puller: {e}")
            await asyncio.sleep(10)

async def pull_task_from_validator(hotkey: str, validator_perf: ValidatorPerformance) -> Optional[Any]:
    """Pull a task from a specific validator"""
    # Implementation similar to original task pulling logic
    # This would include the actual Bittensor communication
    pass

def analyze_prompt_complexity(prompt: str) -> float:
    """Analyze prompt complexity to help with task prioritization"""
    # Simple complexity analysis based on prompt characteristics
    complexity_score = 0.5  # Base score
    
    # Length factor
    length_factor = min(len(prompt) / 100.0, 1.0)
    complexity_score += length_factor * 0.2
    
    # Keyword complexity
    complex_keywords = ['detailed', 'intricate', 'complex', 'realistic', 'high-quality']
    simple_keywords = ['simple', 'basic', 'minimal']
    
    for keyword in complex_keywords:
        if keyword.lower() in prompt.lower():
            complexity_score += 0.1
            
    for keyword in simple_keywords:
        if keyword.lower() in prompt.lower():
            complexity_score -= 0.1
            
    return max(0.1, min(1.0, complexity_score))

async def performance_monitor():
    """Monitor and report system performance"""
    while not shutdown_event.is_set():
        try:
            # Performance reporting logic
            report = {
                'timestamp': time.time(),
                'validators_tracked': len(validator_performance),
                'tasks_in_queue': task_queue.qsize(),
                'submissions_in_queue': submission_queue.qsize(),
                'competitive_analysis': competitive_analyzer.get_competitive_recommendation()
            }
            
            bt.logging.info(f"Performance Report: {json.dumps(report, indent=2)}")
            
        except Exception as e:
            bt.logging.error(f"Error in performance monitor: {e}")
            
        await asyncio.sleep(300)  # Report every 5 minutes

async def competitive_analysis_reporter():
    """Report competitive analysis and recommendations"""
    while not shutdown_event.is_set():
        try:
            if competitive_analyzer.score_history:
                analysis = competitive_analyzer.get_competitive_recommendation()
                bt.logging.info(f"Competitive Analysis: {json.dumps(analysis, indent=2)}")
                
        except Exception as e:
            bt.logging.error(f"Error in competitive analysis reporter: {e}")
            
        await asyncio.sleep(600)  # Report every 10 minutes

if __name__ == "__main__":
    # Set up signal handlers
    signal.signal(signal.SIGINT, lambda s, f: shutdown_event.set())
    signal.signal(signal.SIGTERM, lambda s, f: shutdown_event.set())
    
    asyncio.run(main()) 