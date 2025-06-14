#!/usr/bin/env python3
# Subnet 17 (404-GEN) - Enhanced Miner for Flux-Hunyuan-BPT
# Purpose: Complete miner that uses the enhanced Flux-Hunyuan-BPT generation server

import asyncio
import base64
import json
import time
import traceback
import random
import signal
import sys
import os
from datetime import datetime
from typing import Dict, Any, Optional, List, Set, Tuple
from dataclasses import dataclass
from pathlib import Path

import bittensor as bt
import aiohttp
import pyspz

# --- Configuration Constants ---
NETUID: int = 17
WALLET_NAME: str = "default"
HOTKEY_NAME: str = "default"

# Enhanced Generation Server Endpoints
GENERATION_ENDPOINT_URLS: List[str] = ["http://127.0.0.1:8095/generate/"]  # New BPT-enhanced server
LOCAL_VALIDATION_ENDPOINT_URL: str = "http://127.0.0.1:8094/validate_txt_to_3d_ply/"

# Timing Configuration
METAGRAPH_SYNC_INTERVAL_SECONDS: int = 60 * 10  # 10 minutes
TASK_PULL_INTERVAL_SECONDS: float = 1.0
VALIDATOR_ERROR_RETRY_DELAY_SECONDS: float = 60.0
PULL_TASK_TIMEOUT_SECONDS: float = 15.0  # Increased for BPT processing
GENERATION_TIMEOUT_SECONDS: float = 600.0  # 10 minutes for BPT enhancement
VALIDATION_TIMEOUT_SECONDS: float = 60.0
SUBMISSION_TIMEOUT_SECONDS: float = 30.0
MAX_RETRIES: int = 3
RETRY_DELAY_BASE: float = 2.0

# Quality and Performance
SELF_VALIDATION_MIN_SCORE: float = 0.8  # Higher threshold for enhanced models
MINIMUM_THROTTLE_PERIOD_SECONDS: float = 25.0  # Slightly higher for better quality
MIN_VALIDATOR_STAKE: float = 1000.0
TARGET_GENERATION_TIME: float = 8.0  # Adjusted for BPT processing

# Resource Management
MAX_CONCURRENT_GENERATIONS: int = 1  # Reduced due to BPT memory requirements
MAX_TASK_QUEUE_SIZE: int = 50
MAX_SUBMISSION_QUEUE_SIZE: int = 100
NUM_GENERATION_WORKERS: int = 1  # Single worker for BPT processing
NUM_SUBMISSION_WORKERS: int = 3

# Enhanced Generation Configuration
USE_BPT_ENHANCEMENT: bool = True
BPT_TEMPERATURE: float = 0.5
FALLBACK_TO_STANDARD: bool = True  # Fallback if BPT fails

# Output Configuration
SAVE_GENERATED_ASSETS: bool = True
GENERATED_ASSETS_DIR: str = "flux_hunyuan_bpt_mining_outputs"
LOG_DIR: str = "./logs"
os.makedirs(GENERATED_ASSETS_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

# Miner License Declaration
MINER_LICENSE_CONSENT_DECLARATION: str = (
    "I, as a miner on SN17, have obtained all licenses, rights and consents required to use, reproduce, "
    "modify, display, distribute and make available my submitted results to this subnet and its end users"
)

# --- Data Structures ---
@dataclass
class ValidatorState:
    last_pulled: float = 0.0
    error_count: int = 0
    success_count: int = 0
    last_success: Optional[float] = None
    active_task: Optional[str] = None
    uid: Optional[int] = None
    axon_info: Optional[Any] = None
    cooldown_until: float = 0.0

@dataclass
class MiningTask:
    task_id: str
    prompt: str
    validator_hotkey: str
    validator_uid: int
    assignment_time: float
    validation_threshold: float = 0.7  # Higher for enhanced models
    throttle_period: float = 25.0

@dataclass
class GenerationResult:
    task: MiningTask
    raw_ply: Optional[bytes]
    compressed_ply: Optional[bytes]
    local_score: float
    generation_time: float
    used_bpt: bool = False
    error: Optional[str] = None

@dataclass
class MiningMetrics:
    total_tasks: int = 0
    successful_generations: int = 0
    failed_generations: int = 0
    bpt_enhanced_generations: int = 0
    successful_submissions: int = 0
    failed_submissions: int = 0
    average_generation_time: float = 0.0
    average_local_score: float = 0.0
    last_reward: float = 0.0

# --- Global State ---
validator_states: Dict[str, ValidatorState] = {}
active_generations: Set[str] = set()
running_tasks: Set[asyncio.Task] = set()
shutdown_event = asyncio.Event()
task_queue: Optional[asyncio.Queue] = None
submission_queue: Optional[asyncio.Queue] = None
mining_metrics = MiningMetrics()

# --- Bittensor Objects ---
subtensor: Optional[bt.subtensor] = None
dendrite: Optional[bt.dendrite] = None
wallet: Optional[bt.wallet] = None
metagraph: Optional[bt.metagraph] = None

# --- Synapse Definitions ---
class Task(bt.BaseModel):
    id: str = bt.Field(default_factory=lambda: str(random.randint(10000, 99999)))
    prompt: str = bt.Field(default="")

class Feedback(bt.BaseModel):
    validation_failed: bool = False
    task_fidelity_score: float = 0.0
    average_fidelity_score: float = 0.0
    generations_within_the_window: int = 0
    current_miner_reward: float = 0.0

class PullTask(bt.Synapse):
    task: Optional[Task] = None
    validation_threshold: float = 0.7
    throttle_period: int = 0
    cooldown_until: int = 0
    cooldown_violations: int = 0

    def deserialize(self) -> Optional[Task]:
        return self.task

class SubmitResults(bt.Synapse):
    task: Task
    results: str  # Base64 encoded compressed PLY data
    data_format: str = "ply"
    data_ver: int = 0
    compression: int = 2  # SPZ compression
    submit_time: int = 0
    signature: str = ""
    
    feedback: Optional[Feedback] = None
    cooldown_until: int = 0

# --- Helper Functions ---
def setup_logging():
    """Configure logging with file and console handlers."""
    log_file = os.path.join(LOG_DIR, f"flux_hunyuan_bpt_miner_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    bt.logging.config(
        logging_dir=LOG_DIR,
        logging_file=log_file,
        debug=True,
        trace=False
    )

def sign_submission_data(
    submit_time_ns: int,
    prompt_text: str,
    validator_hotkey_ss58: str,
    miner_wallet: bt.wallet
) -> str:
    """Sign the submission data with the miner's hotkey."""
    message_to_sign = (
        f"{MINER_LICENSE_CONSENT_DECLARATION}"
        f"{submit_time_ns}"
        f"{prompt_text}"
        f"{validator_hotkey_ss58}"
        f"{miner_wallet.hotkey.ss58_address}"
    )
    signature_bytes = miner_wallet.hotkey.sign(message_to_sign.encode('utf-8'))
    signature_b64 = base64.b64encode(signature_bytes).decode('utf-8')
    return signature_b64

async def check_generation_server_health() -> bool:
    """Check if the generation server is healthy and supports our features."""
    try:
        async with aiohttp.ClientSession() as session:
            # Check health endpoint
            async with session.get(f"{GENERATION_ENDPOINT_URLS[0].replace('/generate/', '/health/')}", timeout=5) as response:
                if response.status != 200:
                    return False
            
            # Check status endpoint for BPT support
            async with session.get(f"{GENERATION_ENDPOINT_URLS[0].replace('/generate/', '/status/')}", timeout=5) as response:
                if response.status == 200:
                    data = await response.json()
                    bpt_enabled = data.get('bpt_enabled', False)
                    models_loaded = data.get('models_loaded', {})
                    
                    bt.logging.info(f"Generation server status: BPT={bpt_enabled}, Models={models_loaded}")
                    return True
                return False
                
    except Exception as e:
        bt.logging.error(f"Generation server health check failed: {e}")
        return False

async def generate_3d_model_raw_ply(prompt: str, use_bpt: bool = None, retry_count: int = 0) -> Tuple[Optional[bytes], bool]:
    """Generate 3D model using enhanced generation server."""
    if use_bpt is None:
        use_bpt = USE_BPT_ENHANCEMENT
        
    generation_endpoint = random.choice(GENERATION_ENDPOINT_URLS)
    bt.logging.info(f"Requesting generation for prompt: '{prompt}' (BPT: {use_bpt}) (attempt {retry_count + 1}/{MAX_RETRIES})")
    
    payload = {
        "prompt": prompt,
        "use_bpt": use_bpt,
        "seed": random.randint(0, 2**31 - 1)
    }
    
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(
                generation_endpoint,
                data=payload,
                timeout=aiohttp.ClientTimeout(total=GENERATION_TIMEOUT_SECONDS)
            ) as response:
                if response.status == 200:
                    ply_data = await response.read()
                    bt.logging.success(f"Generated 3D model: {len(ply_data)} bytes (BPT: {use_bpt})")
                    return ply_data, use_bpt
                else:
                    error_text = await response.text()
                    bt.logging.error(f"Generation failed with status {response.status}: {error_text}")
                    
                    # Try fallback without BPT if enabled
                    if use_bpt and FALLBACK_TO_STANDARD and retry_count == 0:
                        bt.logging.info("Attempting fallback without BPT enhancement...")
                        return await generate_3d_model_raw_ply(prompt, use_bpt=False, retry_count=retry_count + 1)
                    
                    return None, False
                    
    except asyncio.TimeoutError:
        bt.logging.error(f"Generation request timed out after {GENERATION_TIMEOUT_SECONDS}s")
        
        # Try fallback without BPT if enabled
        if use_bpt and FALLBACK_TO_STANDARD and retry_count == 0:
            bt.logging.info("Attempting fallback without BPT due to timeout...")
            return await generate_3d_model_raw_ply(prompt, use_bpt=False, retry_count=retry_count + 1)
        
        return None, False
        
    except Exception as e:
        bt.logging.error(f"Generation request failed: {e}")
        
        # Try fallback without BPT if enabled
        if use_bpt and FALLBACK_TO_STANDARD and retry_count == 0:
            bt.logging.info("Attempting fallback without BPT due to error...")
            return await generate_3d_model_raw_ply(prompt, use_bpt=False, retry_count=retry_count + 1)
        
        return None, False

async def validate_locally(prompt: str, compressed_ply_bytes: bytes, retry_count: int = 0) -> float:
    """Validate the generated 3D model locally."""
    bt.logging.info(f"Validating locally for prompt: '{prompt}' (attempt {retry_count + 1}/{MAX_RETRIES})")
    
    try:
        async with aiohttp.ClientSession() as session:
            form_data = aiohttp.FormData()
            form_data.add_field('prompt', prompt)
            form_data.add_field('ply_file', compressed_ply_bytes, filename='model.ply.spz')
            
            async with session.post(
                LOCAL_VALIDATION_ENDPOINT_URL,
                data=form_data,
                timeout=aiohttp.ClientTimeout(total=VALIDATION_TIMEOUT_SECONDS)
            ) as response:
                if response.status == 200:
                    result = await response.json()
                    score = result.get('score', 0.0)
                    bt.logging.success(f"Local validation score: {score:.3f}")
                    return score
                else:
                    error_text = await response.text()
                    bt.logging.error(f"Local validation failed with status {response.status}: {error_text}")
                    return 0.0
                    
    except asyncio.TimeoutError:
        bt.logging.error(f"Local validation timed out after {VALIDATION_TIMEOUT_SECONDS}s")
        return 0.0
    except Exception as e:
        bt.logging.error(f"Local validation failed: {e}")
        return 0.0

def save_generated_asset(result: GenerationResult) -> Optional[str]:
    """Save the generated asset to disk."""
    if not SAVE_GENERATED_ASSETS or not result.raw_ply:
        return None
    
    try:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        bpt_suffix = "_bpt" if result.used_bpt else ""
        filename = f"{result.task.task_id}_{timestamp}{bpt_suffix}.ply"
        filepath = os.path.join(GENERATED_ASSETS_DIR, filename)
        
        with open(filepath, 'wb') as f:
            f.write(result.raw_ply)
        
        bt.logging.info(f"Saved generated asset: {filepath}")
        return filepath
    except Exception as e:
        bt.logging.error(f"Failed to save generated asset: {e}")
        return None

async def metagraph_syncer():
    """Sync metagraph periodically."""
    global metagraph
    
    while not shutdown_event.is_set():
        try:
            if metagraph:
                bt.logging.info("Syncing metagraph...")
                metagraph.sync(subtensor=subtensor)
                bt.logging.success(f"Metagraph synced - {len(metagraph.axons)} axons")
            
            await asyncio.sleep(METAGRAPH_SYNC_INTERVAL_SECONDS)
            
        except Exception as e:
            bt.logging.error(f"Error syncing metagraph: {e}")
            await asyncio.sleep(60)

async def task_puller(puller_id: int):
    """Pull tasks from validators."""
    bt.logging.info(f"Task puller {puller_id} started")
    
    while not shutdown_event.is_set():
        try:
            # Check generation server health periodically
            if not await check_generation_server_health():
                bt.logging.warning("Generation server not healthy, waiting...")
                await asyncio.sleep(30)
                continue
            
            # Get available validators
            if not metagraph:
                await asyncio.sleep(TASK_PULL_INTERVAL_SECONDS)
                continue
            
            available_validators = []
            current_time = time.time()
            
            for uid, axon in enumerate(metagraph.axons):
                if not axon.is_serving:
                    continue
                
                hotkey = metagraph.hotkeys[uid]
                stake = metagraph.stake[uid]
                
                if stake < MIN_VALIDATOR_STAKE:
                    continue
                
                state = validator_states.get(hotkey, ValidatorState())
                state.uid = uid
                state.axon_info = axon
                
                # Check cooldown
                if current_time < state.cooldown_until:
                    continue
                
                # Check if we can pull from this validator
                if (current_time - state.last_pulled) >= TASK_PULL_INTERVAL_SECONDS:
                    available_validators.append((uid, hotkey, axon, state))
            
            if not available_validators:
                await asyncio.sleep(TASK_PULL_INTERVAL_SECONDS)
                continue
            
            # Select validator (prioritize successful ones)
            validator_uid, validator_hotkey, validator_axon, validator_state = random.choice(available_validators)
            validator_state.last_pulled = current_time
            validator_states[validator_hotkey] = validator_state
            
            # Pull task
            try:
                bt.logging.info(f"Pulling task from validator {validator_uid} ({validator_hotkey[:8]}...)")
                
                synapse = PullTask()
                response = await dendrite.call(
                    target_axon=validator_axon,
                    synapse=synapse,
                    deserialize=False,
                    timeout=PULL_TASK_TIMEOUT_SECONDS
                )
                
                if response and response.task:
                    task = MiningTask(
                        task_id=response.task.id,
                        prompt=response.task.prompt,
                        validator_hotkey=validator_hotkey,
                        validator_uid=validator_uid,
                        assignment_time=current_time,
                        validation_threshold=response.validation_threshold,
                        throttle_period=response.throttle_period
                    )
                    
                    # Add to task queue
                    if task_queue.qsize() < MAX_TASK_QUEUE_SIZE:
                        await task_queue.put(task)
                        validator_state.active_task = task.task_id
                        validator_state.success_count += 1
                        mining_metrics.total_tasks += 1
                        
                        bt.logging.success(f"Received task: {task.task_id} - '{task.prompt[:50]}...'")
                    else:
                        bt.logging.warning("Task queue full, dropping task")
                        
                    # Handle cooldown
                    if response.cooldown_until > 0:
                        validator_state.cooldown_until = response.cooldown_until
                        bt.logging.info(f"Validator cooldown until: {response.cooldown_until}")
                        
                else:
                    bt.logging.debug(f"No task from validator {validator_uid}")
                    
            except Exception as e:
                validator_state.error_count += 1
                bt.logging.error(f"Error pulling task from validator {validator_uid}: {e}")
                
                if validator_state.error_count >= 3:
                    validator_state.cooldown_until = current_time + VALIDATOR_ERROR_RETRY_DELAY_SECONDS
                
        except Exception as e:
            bt.logging.error(f"Error in task puller {puller_id}: {e}")
            
        await asyncio.sleep(TASK_PULL_INTERVAL_SECONDS)
    
    bt.logging.info(f"Task puller {puller_id} stopped")

async def generation_processor(processor_id: int):
    """Process generation tasks."""
    bt.logging.info(f"Generation processor {processor_id} started")
    
    while not shutdown_event.is_set():
        try:
            # Get task from queue
            task = await asyncio.wait_for(task_queue.get(), timeout=1.0)
            
            if task.task_id in active_generations:
                bt.logging.warning(f"Task {task.task_id} already being processed")
                continue
            
            active_generations.add(task.task_id)
            
            try:
                bt.logging.info(f"Processing task {task.task_id}: '{task.prompt}'")
                generation_start = time.time()
                
                # Generate 3D model
                raw_ply, used_bpt = await generate_3d_model_raw_ply(task.prompt, USE_BPT_ENHANCEMENT)
                
                if raw_ply is None:
                    bt.logging.error(f"Failed to generate 3D model for task {task.task_id}")
                    mining_metrics.failed_generations += 1
                    continue
                
                generation_time = time.time() - generation_start
                
                # Compress the PLY data
                compressed_ply = pyspz.compress(raw_ply)
                
                # Validate locally
                local_score = await validate_locally(task.prompt, compressed_ply)
                
                # Create result
                result = GenerationResult(
                    task=task,
                    raw_ply=raw_ply,
                    compressed_ply=compressed_ply,
                    local_score=local_score,
                    generation_time=generation_time,
                    used_bpt=used_bpt
                )
                
                # Check if result meets quality threshold
                if local_score >= SELF_VALIDATION_MIN_SCORE:
                    # Save asset if enabled
                    save_generated_asset(result)
                    
                    # Add to submission queue
                    if submission_queue.qsize() < MAX_SUBMISSION_QUEUE_SIZE:
                        await submission_queue.put(result)
                        mining_metrics.successful_generations += 1
                        if used_bpt:
                            mining_metrics.bpt_enhanced_generations += 1
                        
                        bt.logging.success(f"Task {task.task_id} completed successfully (score: {local_score:.3f}, BPT: {used_bpt}, time: {generation_time:.2f}s)")
                    else:
                        bt.logging.warning("Submission queue full, dropping result")
                        mining_metrics.failed_generations += 1
                else:
                    bt.logging.warning(f"Task {task.task_id} failed local validation (score: {local_score:.3f} < {SELF_VALIDATION_MIN_SCORE})")
                    mining_metrics.failed_generations += 1
                
            finally:
                active_generations.discard(task.task_id)
                task_queue.task_done()
                
        except asyncio.TimeoutError:
            continue
        except Exception as e:
            bt.logging.error(f"Error in generation processor {processor_id}: {e}")
            traceback.print_exc()
            
    bt.logging.info(f"Generation processor {processor_id} stopped")

async def result_submitter(submitter_id: int):
    """Submit results to validators."""
    bt.logging.info(f"Result submitter {submitter_id} started")
    
    while not shutdown_event.is_set():
        try:
            # Get result from queue
            result = await asyncio.wait_for(submission_queue.get(), timeout=1.0)
            
            try:
                bt.logging.info(f"Submitting result for task {result.task.task_id}")
                
                # Prepare submission
                submit_time_ns = int(time.time_ns())
                signature = sign_submission_data(
                    submit_time_ns,
                    result.task.prompt,
                    result.task.validator_hotkey,
                    wallet
                )
                
                # Create submission synapse
                submission = SubmitResults(
                    task=Task(id=result.task.task_id, prompt=result.task.prompt),
                    results=base64.b64encode(result.compressed_ply).decode('utf-8'),
                    data_format="ply",
                    data_ver=0,
                    compression=2,
                    submit_time=submit_time_ns,
                    signature=signature
                )
                
                # Get validator info
                validator_state = validator_states.get(result.task.validator_hotkey)
                if not validator_state or not validator_state.axon_info:
                    bt.logging.error(f"No validator info for {result.task.validator_hotkey}")
                    continue
                
                # Submit to validator
                response = await dendrite.call(
                    target_axon=validator_state.axon_info,
                    synapse=submission,
                    deserialize=False,
                    timeout=SUBMISSION_TIMEOUT_SECONDS
                )
                
                if response and hasattr(response, 'feedback') and response.feedback:
                    feedback = response.feedback
                    mining_metrics.successful_submissions += 1
                    mining_metrics.last_reward = feedback.current_miner_reward
                    
                    bt.logging.success(f"Task {result.task.task_id} submitted successfully")
                    bt.logging.info(f"Feedback: fidelity={feedback.task_fidelity_score:.3f}, "
                                   f"avg_fidelity={feedback.average_fidelity_score:.3f}, "
                                   f"reward={feedback.current_miner_reward:.6f}")
                    
                    # Handle cooldown
                    if response.cooldown_until > 0:
                        validator_state.cooldown_until = response.cooldown_until
                        bt.logging.info(f"Validator cooldown until: {response.cooldown_until}")
                        
                else:
                    bt.logging.warning(f"Task {result.task.task_id} submission failed or no feedback received")
                    mining_metrics.failed_submissions += 1
                
            finally:
                submission_queue.task_done()
                
        except asyncio.TimeoutError:
            continue
        except Exception as e:
            bt.logging.error(f"Error in result submitter {submitter_id}: {e}")
            traceback.print_exc()
            
    bt.logging.info(f"Result submitter {submitter_id} stopped")

def handle_shutdown(signum, frame):
    """Handle shutdown signals."""
    bt.logging.info("Received shutdown signal, stopping miner...")
    shutdown_event.set()

async def metrics_reporter():
    """Report mining metrics periodically."""
    while not shutdown_event.is_set():
        try:
            await asyncio.sleep(300)  # Report every 5 minutes
            
            bt.logging.info("=== Mining Metrics ===")
            bt.logging.info(f"Total tasks: {mining_metrics.total_tasks}")
            bt.logging.info(f"Successful generations: {mining_metrics.successful_generations}")
            bt.logging.info(f"BPT enhanced generations: {mining_metrics.bpt_enhanced_generations}")
            bt.logging.info(f"Failed generations: {mining_metrics.failed_generations}")
            bt.logging.info(f"Successful submissions: {mining_metrics.successful_submissions}")
            bt.logging.info(f"Failed submissions: {mining_metrics.failed_submissions}")
            bt.logging.info(f"Last reward: {mining_metrics.last_reward:.6f}")
            
            if mining_metrics.successful_generations > 0:
                success_rate = (mining_metrics.successful_generations / 
                               max(1, mining_metrics.successful_generations + mining_metrics.failed_generations))
                bpt_rate = mining_metrics.bpt_enhanced_generations / mining_metrics.successful_generations
                bt.logging.info(f"Generation success rate: {success_rate:.2%}")
                bt.logging.info(f"BPT enhancement rate: {bpt_rate:.2%}")
            
            bt.logging.info("=== End Metrics ===")
            
        except Exception as e:
            bt.logging.error(f"Error in metrics reporter: {e}")

async def main():
    """Main miner function."""
    global subtensor, dendrite, wallet, metagraph, task_queue, submission_queue
    
    # Setup
    setup_logging()
    bt.logging.info("Starting Flux-Hunyuan-BPT Enhanced Miner...")
    
    # Initialize Bittensor objects
    try:
        bt.logging.info("Initializing Bittensor objects...")
        
        wallet = bt.wallet(name=WALLET_NAME, hotkey=HOTKEY_NAME)
        subtensor = bt.subtensor(network="finney")
        dendrite = bt.dendrite(wallet=wallet)
        metagraph = bt.metagraph(netuid=NETUID, network="finney")
        
        # Check registration
        if wallet.hotkey.ss58_address not in metagraph.hotkeys:
            bt.logging.error(f"Wallet {wallet.hotkey.ss58_address} not registered on subnet {NETUID}")
            return
        
        bt.logging.success("Bittensor objects initialized successfully")
        
    except Exception as e:
        bt.logging.error(f"Failed to initialize Bittensor objects: {e}")
        return
    
    # Check generation server
    if not await check_generation_server_health():
        bt.logging.error("Generation server not available. Please start the flux_hunyuan_bpt_generation_server.py first.")
        return
    
    bt.logging.success("Generation server is healthy and ready")
    
    # Initialize queues
    task_queue = asyncio.Queue(maxsize=MAX_TASK_QUEUE_SIZE)
    submission_queue = asyncio.Queue(maxsize=MAX_SUBMISSION_QUEUE_SIZE)
    
    # Setup signal handlers
    signal.signal(signal.SIGINT, handle_shutdown)
    signal.signal(signal.SIGTERM, handle_shutdown)
    
    # Start workers
    bt.logging.info("Starting worker tasks...")
    
    # Metagraph syncer
    running_tasks.add(asyncio.create_task(metagraph_syncer()))
    
    # Task pullers
    for i in range(1):  # Single task puller for now
        running_tasks.add(asyncio.create_task(task_puller(i)))
    
    # Generation processors
    for i in range(NUM_GENERATION_WORKERS):
        running_tasks.add(asyncio.create_task(generation_processor(i)))
    
    # Result submitters
    for i in range(NUM_SUBMISSION_WORKERS):
        running_tasks.add(asyncio.create_task(result_submitter(i)))
    
    # Metrics reporter
    running_tasks.add(asyncio.create_task(metrics_reporter()))
    
    bt.logging.success(f"All workers started - {len(running_tasks)} tasks running")
    
    # Wait for shutdown
    try:
        await shutdown_event.wait()
    except KeyboardInterrupt:
        bt.logging.info("Keyboard interrupt received")
    
    # Cleanup
    bt.logging.info("Stopping all tasks...")
    for task in running_tasks:
        task.cancel()
    
    await asyncio.gather(*running_tasks, return_exceptions=True)
    bt.logging.info("Flux-Hunyuan-BPT Enhanced Miner stopped")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        bt.logging.info("Miner interrupted by user")
    except Exception as e:
        bt.logging.error(f"Miner crashed: {e}")
        traceback.print_exc()
        sys.exit(1) 