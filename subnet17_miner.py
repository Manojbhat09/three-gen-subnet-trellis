#!/usr/bin/env python3
# Subnet 17 (404-GEN) - Main Miner Script
# Purpose: Complete miner for Subnet 17 that pulls tasks, generates 3D models, and submits results

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

# Local Endpoints
GENERATION_ENDPOINT_URLS: List[str] = ["http://127.0.0.1:8093/generate/"]
LOCAL_VALIDATION_ENDPOINT_URL: str = "http://127.0.0.1:8094/validate_txt_to_3d_ply/"

# Timing Configuration
METAGRAPH_SYNC_INTERVAL_SECONDS: int = 60 * 10  # 10 minutes
TASK_PULL_INTERVAL_SECONDS: float = 1.0
VALIDATOR_ERROR_RETRY_DELAY_SECONDS: float = 60.0
PULL_TASK_TIMEOUT_SECONDS: float = 12.0
GENERATION_TIMEOUT_SECONDS: float = 300.0  # 5 minutes
VALIDATION_TIMEOUT_SECONDS: float = 60.0
SUBMISSION_TIMEOUT_SECONDS: float = 30.0
MAX_RETRIES: int = 3
RETRY_DELAY_BASE: float = 2.0

# Quality and Performance
SELF_VALIDATION_MIN_SCORE: float = 0.7
MINIMUM_THROTTLE_PERIOD_SECONDS: float = 20.0
MIN_VALIDATOR_STAKE: float = 1000.0
TARGET_GENERATION_TIME: float = 5.0  # Target under 5 seconds

# Resource Management
MAX_CONCURRENT_GENERATIONS: int = 2
MAX_TASK_QUEUE_SIZE: int = 100
MAX_SUBMISSION_QUEUE_SIZE: int = 100
NUM_GENERATION_WORKERS: int = 2
NUM_SUBMISSION_WORKERS: int = 3

# Output Configuration
SAVE_GENERATED_ASSETS: bool = True
GENERATED_ASSETS_DIR: str = "mining_outputs"
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
    validation_threshold: float = 0.6
    throttle_period: float = 20.0

@dataclass
class GenerationResult:
    task: MiningTask
    raw_ply: Optional[bytes]
    compressed_ply: Optional[bytes]
    local_score: float
    generation_time: float
    error: Optional[str] = None

@dataclass
class MiningMetrics:
    total_tasks: int = 0
    successful_generations: int = 0
    failed_generations: int = 0
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
    validation_threshold: float = 0.6
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
    log_file = os.path.join(LOG_DIR, f"miner_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
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

async def generate_3d_model_raw_ply(prompt: str, retry_count: int = 0) -> Optional[bytes]:
    """Generate 3D model using local generation server."""
    generation_endpoint = random.choice(GENERATION_ENDPOINT_URLS)
    bt.logging.info(f"Requesting generation for prompt: '{prompt}' (attempt {retry_count + 1}/{MAX_RETRIES})")
    
    payload = {"prompt": prompt}
    
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(
                generation_endpoint,
                data=payload,
                timeout=GENERATION_TIMEOUT_SECONDS
            ) as response:
                if response.status == 200:
                    raw_ply_bytes = await response.read()
                    if len(raw_ply_bytes) < 100:
                        raise ValueError("Generated PLY data too small")
                    bt.logging.success(f"Generated raw PLY. Size: {len(raw_ply_bytes)} bytes.")
                    return raw_ply_bytes
                else:
                    error_text = await response.text()
                    bt.logging.error(f"Generation failed. Status: {response.status}. Error: {error_text}")
                    if retry_count < MAX_RETRIES - 1:
                        await asyncio.sleep(RETRY_DELAY_BASE * (retry_count + 1))
                        return await generate_3d_model_raw_ply(prompt, retry_count + 1)
                    return None
    except Exception as e:
        bt.logging.error(f"Exception during generation: {e}")
        if retry_count < MAX_RETRIES - 1:
            await asyncio.sleep(RETRY_DELAY_BASE * (retry_count + 1))
            return await generate_3d_model_raw_ply(prompt, retry_count + 1)
        return None

async def validate_locally(prompt: str, compressed_ply_bytes: bytes, retry_count: int = 0) -> float:
    """Validate the compressed PLY data locally."""
    try:
        base64_compressed_data = base64.b64encode(compressed_ply_bytes).decode('utf-8')
        payload = {
            "prompt": prompt,
            "data": base64_compressed_data,
            "compression": 2,  # SPZ compression
            "data_ver": 0,
            "generate_preview": False
        }
        
        async with aiohttp.ClientSession() as session:
            async with session.post(
                LOCAL_VALIDATION_ENDPOINT_URL,
                json=payload,
                timeout=VALIDATION_TIMEOUT_SECONDS
            ) as response:
                if response.status == 200:
                    validation_result = await response.json()
                    score = validation_result.get("score", 0.0)
                    if not 0 <= score <= 1:
                        raise ValueError(f"Invalid validation score: {score}")
                    return score
                else:
                    error_text = await response.text()
                    bt.logging.warning(f"Local validation HTTP error: Status {response.status}, {error_text}")
                    if retry_count < MAX_RETRIES - 1:
                        await asyncio.sleep(RETRY_DELAY_BASE * (retry_count + 1))
                        return await validate_locally(prompt, compressed_ply_bytes, retry_count + 1)
                    return -1.0
    except Exception as e:
        bt.logging.error(f"Error during local validation: {e}")
        if retry_count < MAX_RETRIES - 1:
            await asyncio.sleep(RETRY_DELAY_BASE * (retry_count + 1))
            return await validate_locally(prompt, compressed_ply_bytes, retry_count + 1)
        return -1.0

def save_generated_asset(result: GenerationResult) -> Optional[str]:
    """Save the generated asset to disk."""
    if not SAVE_GENERATED_ASSETS or not result.compressed_ply:
        return None
        
    try:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{result.task.prompt.replace(' ', '_')[:50]}_{result.task.task_id}_{result.local_score:.2f}_{timestamp}.ply.spz"
        filepath = os.path.join(GENERATED_ASSETS_DIR, filename)
        
        with open(filepath, "wb") as f:
            f.write(result.compressed_ply)
            
        bt.logging.info(f"Saved asset to {filepath}")
        return filepath
    except Exception as e:
        bt.logging.error(f"Failed to save asset: {e}")
        return None

# --- Main Async Components ---
async def metagraph_syncer():
    """Periodically sync metagraph and update validator states."""
    global metagraph, validator_states
    bt.logging.info("Metagraph syncer started.")
    
    while not shutdown_event.is_set():
        try:
            if subtensor and subtensor.is_connected():
                bt.logging.info("Syncing metagraph...")
                new_metagraph = subtensor.metagraph(netuid=NETUID)
                
                if not new_metagraph:
                    bt.logging.warning("Failed to get new metagraph. Retrying...")
                    await asyncio.sleep(30)
                    continue
                
                current_validator_hotkeys = set()
                for neuron_lite in new_metagraph.neurons:
                    if neuron_lite.stake > MIN_VALIDATOR_STAKE and neuron_lite.is_serving:
                        current_validator_hotkeys.add(neuron_lite.hotkey)
                        if neuron_lite.hotkey not in validator_states:
                            validator_states[neuron_lite.hotkey] = ValidatorState(
                                uid=neuron_lite.uid,
                                axon_info=neuron_lite.axon_info
                            )
                            bt.logging.debug(f"Added new validator: UID {neuron_lite.uid}")
                
                # Remove stale validators
                stale_validators = set(validator_states.keys()) - current_validator_hotkeys
                for hotkey_to_remove in stale_validators:
                    if hotkey_to_remove in validator_states:
                        del validator_states[hotkey_to_remove]
                        bt.logging.debug(f"Removed stale validator: {hotkey_to_remove}")
                
                metagraph = new_metagraph
                bt.logging.success(f"Metagraph synced. Found {len(validator_states)} active validators.")
            else:
                bt.logging.warning("Subtensor not connected. Retrying in 30 seconds...")
                await asyncio.sleep(30)
        except Exception as e:
            bt.logging.error(f"Error in metagraph_syncer: {e}\n{traceback.format_exc()}")
            await asyncio.sleep(30)
        
        await asyncio.sleep(METAGRAPH_SYNC_INTERVAL_SECONDS)

async def task_puller(puller_id: int):
    """Pull tasks from validators and add to task queue."""
    global validator_states, task_queue
    bt.logging.info(f"Task puller {puller_id} started.")
    
    while not shutdown_event.is_set():
        if not validator_states:
            bt.logging.debug("No validators available. Waiting...")
            await asyncio.sleep(10)
            continue

        if task_queue.full():
            await asyncio.sleep(1)
            continue

        validator_hotkeys = list(validator_states.keys())
        random.shuffle(validator_hotkeys)

        for hotkey in validator_hotkeys:
            if shutdown_event.is_set():
                break
                
            if hotkey not in validator_states:
                continue

            val_state = validator_states[hotkey]
            current_time = time.time()

            if current_time >= val_state.cooldown_until and val_state.active_task is None:
                try:
                    bt.logging.debug(f"Attempting to pull task from UID {val_state.uid}...")
                    
                    if not val_state.axon_info or not val_state.axon_info.ip or not val_state.axon_info.port:
                        bt.logging.warning(f"Axon info missing for validator UID {val_state.uid}. Skipping.")
                        val_state.cooldown_until = current_time + VALIDATOR_ERROR_RETRY_DELAY_SECONDS
                        continue
                        
                    target_axon = bt.axon(
                        ip=val_state.axon_info.ip,
                        port=val_state.axon_info.port,
                        hotkey=hotkey
                    )
                    pull_task_synapse = PullTask()
                    
                    response_synapse: PullTask = await dendrite.call(
                        target_axon=target_axon,
                        synapse=pull_task_synapse,
                        timeout=PULL_TASK_TIMEOUT_SECONDS,
                        deserialize=True
                    )

                    if response_synapse.dendrite.status_code == 200 and response_synapse.task and response_synapse.task.prompt:
                        prompt = response_synapse.task.prompt
                        task_id = response_synapse.task.id
                        
                        mining_task = MiningTask(
                            task_id=task_id,
                            prompt=prompt,
                            validator_hotkey=hotkey,
                            validator_uid=val_state.uid,
                            assignment_time=current_time,
                            validation_threshold=response_synapse.validation_threshold,
                            throttle_period=response_synapse.throttle_period
                        )
                        
                        val_state.active_task = prompt
                        val_state.cooldown_until = current_time + (SUBMISSION_TIMEOUT_SECONDS * 2)
                        
                        await task_queue.put(mining_task)
                        bt.logging.success(f"Task puller {puller_id}: Pulled task '{task_id}' from UID {val_state.uid}: '{prompt[:70]}...'")
                        
                        mining_metrics.total_tasks += 1
                        
                    else:
                        bt.logging.warning(f"Failed to pull task from UID {val_state.uid}. Status: {response_synapse.dendrite.status_code}")
                        val_state.error_count += 1
                        val_state.cooldown_until = current_time + VALIDATOR_ERROR_RETRY_DELAY_SECONDS
                
                except Exception as e:
                    bt.logging.error(f"Exception pulling task from UID {val_state.uid}: {e}\n{traceback.format_exc()}")
                    val_state.error_count += 1
                    val_state.cooldown_until = current_time + VALIDATOR_ERROR_RETRY_DELAY_SECONDS
                
                await asyncio.sleep(0.1)
            
            await asyncio.sleep(TASK_PULL_INTERVAL_SECONDS)

async def generation_processor(processor_id: int):
    """Process tasks from task queue and generate 3D models."""
    global task_queue, submission_queue, active_generations, mining_metrics
    bt.logging.info(f"Generation processor {processor_id} started.")
    
    while not shutdown_event.is_set():
        try:
            mining_task: MiningTask = await task_queue.get()
            
            if mining_task.prompt in active_generations:
                bt.logging.warning(f"Generation processor {processor_id}: Prompt already being processed. Skipping.")
                task_queue.task_done()
                continue
            
            active_generations.add(mining_task.prompt)
            bt.logging.info(f"Generation processor {processor_id}: Processing task '{mining_task.task_id}': '{mining_task.prompt[:50]}...'")
            
            start_time = time.time()
            
            # Generate 3D model
            raw_ply = await generate_3d_model_raw_ply(mining_task.prompt)
            
            result = GenerationResult(
                task=mining_task,
                raw_ply=raw_ply,
                compressed_ply=None,
                local_score=-1.0,
                generation_time=time.time() - start_time,
                error=None
            )
            
            if raw_ply:
                try:
                    # Compress the PLY
                    compressed_ply = pyspz.compress(raw_ply, workers=-1)
                    result.compressed_ply = compressed_ply
                    
                    bt.logging.info(f"Generation processor {processor_id}: Compressed PLY. Original: {len(raw_ply)}, Compressed: {len(compressed_ply)}")
                    
                    # Self-validate
                    local_score = await validate_locally(mining_task.prompt, compressed_ply)
                    result.local_score = local_score
                    
                    bt.logging.info(f"Generation processor {processor_id}: Local validation score: {local_score:.4f}")
                    
                    # Update metrics
                    mining_metrics.successful_generations += 1
                    mining_metrics.average_generation_time = (
                        (mining_metrics.average_generation_time * (mining_metrics.successful_generations - 1) + result.generation_time) /
                        mining_metrics.successful_generations
                    )
                    
                    if local_score >= 0:
                        mining_metrics.average_local_score = (
                            (mining_metrics.average_local_score * (mining_metrics.successful_generations - 1) + local_score) /
                            mining_metrics.successful_generations
                        )
                    
                except Exception as e:
                    result.error = f"Compression/validation failed: {str(e)}"
                    bt.logging.error(f"Generation processor {processor_id}: {result.error}")
                    mining_metrics.failed_generations += 1
            else:
                result.error = "Generation failed"
                mining_metrics.failed_generations += 1
            
            # Save asset if successful
            if result.compressed_ply and result.local_score >= 0:
                save_generated_asset(result)
            
            # Add to submission queue
            await submission_queue.put(result)
            
            active_generations.remove(mining_task.prompt)
            task_queue.task_done()
            
        except Exception as e:
            bt.logging.error(f"Error in generation_processor {processor_id}: {e}\n{traceback.format_exc()}")
            if 'mining_task' in locals() and mining_task.prompt in active_generations:
                active_generations.remove(mining_task.prompt)
            if task_queue and not task_queue.empty():
                try:
                    task_queue.task_done()
                except ValueError:
                    pass

async def result_submitter(submitter_id: int):
    """Submit generation results to validators."""
    global submission_queue, validator_states, mining_metrics
    bt.logging.info(f"Result submitter {submitter_id} started.")
    
    while not shutdown_event.is_set():
        try:
            result: GenerationResult = await submission_queue.get()
            current_time = time.time()
            
            validator_hotkey = result.task.validator_hotkey
            if validator_hotkey not in validator_states:
                bt.logging.warning(f"Result submitter {submitter_id}: Validator {validator_hotkey} no longer available. Skipping.")
                submission_queue.task_done()
                continue
            
            val_state = validator_states[validator_hotkey]
            
            # Enforce throttle period
            time_since_assignment = current_time - result.task.assignment_time
            throttle_delay = max(0, result.task.throttle_period - time_since_assignment)
            if throttle_delay > 0:
                bt.logging.debug(f"Result submitter {submitter_id}: Throttling for {throttle_delay:.2f}s")
                await asyncio.sleep(throttle_delay)
            
            # Prepare submission data
            if result.local_score < result.task.validation_threshold:
                results_b64 = ""  # Submit empty result for low quality
                bt.logging.info(f"Result submitter {submitter_id}: Submitting EMPTY result (score {result.local_score:.4f} < threshold {result.task.validation_threshold:.4f})")
            else:
                results_b64 = base64.b64encode(result.compressed_ply).decode('utf-8')
                bt.logging.info(f"Result submitter {submitter_id}: Submitting REAL result (score: {result.local_score:.4f})")
            
            # Create submission synapse
            submit_time_ns = time.time_ns()
            signature = sign_submission_data(
                submit_time_ns, result.task.prompt, validator_hotkey, wallet
            )
            
            submit_synapse = SubmitResults(
                task=Task(id=result.task.task_id, prompt=result.task.prompt),
                results=results_b64,
                compression=2,  # SPZ
                submit_time=submit_time_ns,
                signature=signature
            )
            
            # Submit to validator
            try:
                target_axon = bt.axon(
                    ip=val_state.axon_info.ip,
                    port=val_state.axon_info.port,
                    hotkey=validator_hotkey
                )
                
                response_synapse: SubmitResults = await dendrite.call(
                    target_axon=target_axon,
                    synapse=submit_synapse,
                    timeout=SUBMISSION_TIMEOUT_SECONDS,
                    deserialize=True
                )
                
                # Process response
                current_time_after_submit = time.time()
                if response_synapse.dendrite.status_code == 200 and response_synapse.feedback:
                    feedback = response_synapse.feedback
                    cooldown_until = response_synapse.cooldown_until
                    
                    bt.logging.success(f"Result submitter {submitter_id}: Submission successful. Score: {feedback.task_fidelity_score:.4f}, Reward: {feedback.current_miner_reward:.4f}")
                    
                    val_state.success_count += 1
                    val_state.last_success = current_time_after_submit
                    val_state.error_count = 0
                    val_state.cooldown_until = cooldown_until if cooldown_until > current_time_after_submit else current_time_after_submit + 60
                    
                    mining_metrics.successful_submissions += 1
                    mining_metrics.last_reward = feedback.current_miner_reward
                    
                else:
                    bt.logging.warning(f"Result submitter {submitter_id}: Submission failed. Status: {response_synapse.dendrite.status_code}")
                    val_state.error_count += 1
                    val_state.cooldown_until = current_time_after_submit + VALIDATOR_ERROR_RETRY_DELAY_SECONDS
                    mining_metrics.failed_submissions += 1
                
                val_state.active_task = None
                
            except Exception as e:
                bt.logging.error(f"Result submitter {submitter_id}: Exception during submission: {e}")
                val_state.error_count += 1
                val_state.cooldown_until = time.time() + VALIDATOR_ERROR_RETRY_DELAY_SECONDS
                val_state.active_task = None
                mining_metrics.failed_submissions += 1
            
            submission_queue.task_done()
            
        except Exception as e:
            bt.logging.error(f"Error in result_submitter {submitter_id}: {e}\n{traceback.format_exc()}")
            if submission_queue and not submission_queue.empty():
                try:
                    submission_queue.task_done()
                except ValueError:
                    pass

def handle_shutdown(signum, frame):
    """Handle graceful shutdown on SIGINT/SIGTERM."""
    bt.logging.info("Shutdown signal received. Initiating graceful shutdown...")
    shutdown_event.set()

async def metrics_reporter():
    """Periodically log mining metrics."""
    while not shutdown_event.is_set():
        bt.logging.info(f"Mining Metrics - Tasks: {mining_metrics.total_tasks}, "
                       f"Gen Success: {mining_metrics.successful_generations}, "
                       f"Sub Success: {mining_metrics.successful_submissions}, "
                       f"Avg Gen Time: {mining_metrics.average_generation_time:.2f}s, "
                       f"Avg Score: {mining_metrics.average_local_score:.4f}, "
                       f"Last Reward: {mining_metrics.last_reward:.4f}")
        await asyncio.sleep(300)  # Report every 5 minutes

async def main():
    """Main entry point for the Subnet 17 miner."""
    global subtensor, dendrite, wallet, metagraph, task_queue, submission_queue

    # Set up signal handlers
    signal.signal(signal.SIGINT, handle_shutdown)
    signal.signal(signal.SIGTERM, handle_shutdown)

    try:
        # Initialize Bittensor
        subtensor = bt.subtensor(network="finney")
        bt.logging.info(f"Connected to Subtensor: {subtensor.network}")
        
        wallet = bt.wallet(name=WALLET_NAME, hotkey=HOTKEY_NAME)
        bt.logging.info(f"Wallet loaded: {wallet.name}, Hotkey: {wallet.hotkey_str}")
        
        dendrite = bt.dendrite(wallet=wallet)
        bt.logging.info(f"Dendrite initialized for hotkey: {dendrite.keypair.ss58_address}")
        
        metagraph = subtensor.metagraph(netuid=NETUID)
        bt.logging.info(f"Initial metagraph sync complete for NETUID {NETUID}")
        
        # Verify registration
        my_hotkey_ss58 = wallet.hotkey.ss58_address
        if my_hotkey_ss58 not in metagraph.hotkeys:
            bt.logging.error(f"Hotkey {my_hotkey_ss58} is not registered on NETUID {NETUID}")
            return
        else:
            my_uid = metagraph.hotkeys.index(my_hotkey_ss58)
            bt.logging.success(f"Hotkey registered with UID {my_uid} on NETUID {NETUID}")
        
        # Initialize queues
        task_queue = asyncio.Queue(maxsize=MAX_TASK_QUEUE_SIZE)
        submission_queue = asyncio.Queue(maxsize=MAX_SUBMISSION_QUEUE_SIZE)
        
        # Start all components
        tasks = [
            asyncio.create_task(metagraph_syncer()),
            asyncio.create_task(metrics_reporter()),
        ]
        
        # Start task pullers
        for i in range(max(1, len(GENERATION_ENDPOINT_URLS))):
            tasks.append(asyncio.create_task(task_puller(puller_id=i)))
        
        # Start generation processors
        for i in range(NUM_GENERATION_WORKERS):
            tasks.append(asyncio.create_task(generation_processor(processor_id=i)))
        
        # Start result submitters
        for i in range(NUM_SUBMISSION_WORKERS):
            tasks.append(asyncio.create_task(result_submitter(submitter_id=i)))
        
        running_tasks.update(tasks)
        bt.logging.info("Subnet 17 Miner started successfully. All components running...")
        
        try:
            while not shutdown_event.is_set():
                await asyncio.sleep(60)
        except Exception as e:
            bt.logging.error(f"Unhandled exception in main loop: {e}\n{traceback.format_exc()}")
        finally:
            bt.logging.info("Initiating shutdown...")
            shutdown_event.set()
            
            # Cancel all running tasks
            for task in running_tasks:
                if not task.done():
                    task.cancel()
            
            # Wait for tasks to complete
            await asyncio.gather(*running_tasks, return_exceptions=True)
            bt.logging.info("Shutdown complete.")
            
    except Exception as e:
        bt.logging.error(f"Critical error in startup: {e}")
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    setup_logging()
    bt.logging.info("Starting Subnet 17 Miner...")
    asyncio.run(main()) 