#!/usr/bin/env python3
# Subnet 17 (404-GEN) - Local Validation Runner Script
# Purpose: Pulls tasks from mainnet validators, generates models locally,
#          and validates them using a local validation endpoint for testing.

import asyncio
import base64
import json
import time
import traceback
import random
import signal
import sys
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

# Endpoints
GENERATION_ENDPOINT_URLS: List[str] = ["http://127.0.0.1:8093/generate/"]  # Can add multiple endpoints
LOCAL_VALIDATION_ENDPOINT_URL: str = "http://127.0.0.1:8094/validate_txt_to_3d_ply/"

# Timing Configuration
METAGRAPH_SYNC_INTERVAL_SECONDS: int = 60 * 5  # 5 minutes
VALIDATOR_PULL_DELAY_SECONDS: float = 60.0
PULL_TASK_TIMEOUT_SECONDS: float = 12.0
GENERATION_TIMEOUT_SECONDS: float = 300.0  # 5 minutes
VALIDATION_TIMEOUT_SECONDS: float = 60.0
MAX_RETRIES: int = 3
RETRY_DELAY_BASE: float = 2.0

# Resource Management
MAX_CONCURRENT_GENERATIONS: int = 2
MAX_QUEUE_SIZE: int = 100
MIN_VALIDATOR_STAKE: float = 1000.0

# Output Configuration
SAVE_GENERATED_ASSETS: bool = True
GENERATED_ASSETS_DIR: str = "locally_validated_assets"
LOG_DIR: str = "./logs"
os.makedirs(GENERATED_ASSETS_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

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

@dataclass
class GenerationResult:
    prompt: str
    task_id: str
    validator_uid: int
    raw_ply: Optional[bytes]
    compressed_ply: Optional[bytes]
    local_score: float
    error: Optional[str] = None
    retry_count: int = 0

# --- Global State ---
validator_states: Dict[str, ValidatorState] = {}
active_generations: Set[str] = set()
running_tasks: Set[asyncio.Task] = set()
shutdown_event = asyncio.Event()

# --- Bittensor Objects ---
subtensor: Optional[bt.subtensor] = None
dendrite: Optional[bt.dendrite] = None
wallet: Optional[bt.wallet] = None
metagraph: Optional[bt.metagraph] = None

# --- Synapse Definitions ---
class Task(bt.BaseModel):
    id: str = bt.Field(default_factory=lambda: str(random.randint(10000, 99999)))
    prompt: str = bt.Field(default="")

class PullTask(bt.Synapse):
    task: Optional[Task] = None
    validation_threshold: float = 0.6
    throttle_period: int = 0
    cooldown_until: int = 0
    cooldown_violations: int = 0

    def deserialize(self) -> Optional[Task]:
        return self.task

# --- Helper Functions ---
def setup_logging():
    """Configure logging with file and console handlers."""
    log_file = os.path.join(LOG_DIR, f"validation_runner_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    bt.logging.config(
        logging_dir=LOG_DIR,
        logging_file=log_file,
        debug=True,
        trace=False
    )

async def generate_3d_model_raw_ply(
    prompt: str,
    generation_endpoint_url: str,
    retry_count: int = 0
) -> Optional[bytes]:
    """Calls the local generation service and returns raw PLY data with retries."""
    bt.logging.info(f"Requesting generation for prompt: '{prompt}' (attempt {retry_count + 1}/{MAX_RETRIES})")
    payload = {"prompt": prompt}
    
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(
                generation_endpoint_url,
                data=payload,
                timeout=GENERATION_TIMEOUT_SECONDS
            ) as response:
                if response.status == 200:
                    raw_ply_bytes = await response.read()
                    if len(raw_ply_bytes) < 100:  # Basic sanity check
                        raise ValueError("Generated PLY data too small")
                    bt.logging.success(f"Generated raw PLY. Size: {len(raw_ply_bytes)} bytes.")
                    return raw_ply_bytes
                else:
                    error_text = await response.text()
                    bt.logging.error(f"Generation failed. Status: {response.status}. Error: {error_text}")
                    if retry_count < MAX_RETRIES - 1:
                        await asyncio.sleep(RETRY_DELAY_BASE * (retry_count + 1))
                        return await generate_3d_model_raw_ply(
                            prompt, generation_endpoint_url, retry_count + 1
                        )
                    return None
    except asyncio.TimeoutError:
        bt.logging.error(f"Generation timeout for prompt: '{prompt}'")
        if retry_count < MAX_RETRIES - 1:
            await asyncio.sleep(RETRY_DELAY_BASE * (retry_count + 1))
            return await generate_3d_model_raw_ply(
                prompt, generation_endpoint_url, retry_count + 1
            )
        return None
    except Exception as e:
        bt.logging.error(f"Exception during generation: {e}")
        if retry_count < MAX_RETRIES - 1:
            await asyncio.sleep(RETRY_DELAY_BASE * (retry_count + 1))
            return await generate_3d_model_raw_ply(
                prompt, generation_endpoint_url, retry_count + 1
            )
        return None

async def validate_locally(
    prompt: str,
    compressed_ply_bytes: bytes,
    retry_count: int = 0
) -> float:
    """Validates the compressed PLY data locally with retries."""
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
    """Saves the generated and validated asset to disk."""
    if not SAVE_GENERATED_ASSETS or not result.compressed_ply:
        return None
        
    try:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{result.prompt.replace(' ', '_')[:50]}_{result.task_id}_{result.validator_uid}_{result.local_score:.2f}_{timestamp}.ply.spz"
        filepath = os.path.join(GENERATED_ASSETS_DIR, filename)
        
        with open(filepath, "wb") as f:
            f.write(result.compressed_ply)
            
        bt.logging.info(f"Saved asset to {filepath}")
        return filepath
    except Exception as e:
        bt.logging.error(f"Failed to save asset: {e}")
        return None

async def process_task(
    prompt: str,
    task_id: str,
    validator_hotkey: str,
    validator_uid: int
) -> GenerationResult:
    """Process a single task: generate, compress, and validate."""
    try:
        # Select generation endpoint
        generation_endpoint = random.choice(GENERATION_ENDPOINT_URLS)
        
        # Generate 3D model
        raw_ply = await generate_3d_model_raw_ply(prompt, generation_endpoint)
        if not raw_ply:
            return GenerationResult(
                prompt=prompt,
                task_id=task_id,
                validator_uid=validator_uid,
                raw_ply=None,
                compressed_ply=None,
                local_score=-1.0,
                error="Generation failed"
            )
            
        # Compress the model
        try:
            compressed_ply = pyspz.compress(raw_ply, workers=-1)
            bt.logging.info(f"Compressed PLY. Original: {len(raw_ply)}, Compressed: {len(compressed_ply)}")
        except Exception as e:
            return GenerationResult(
                prompt=prompt,
                task_id=task_id,
                validator_uid=validator_uid,
                raw_ply=raw_ply,
                compressed_ply=None,
                local_score=-1.0,
                error=f"Compression failed: {str(e)}"
            )
            
        # Validate locally
        local_score = await validate_locally(prompt, compressed_ply)
        
        return GenerationResult(
            prompt=prompt,
            task_id=task_id,
            validator_uid=validator_uid,
            raw_ply=raw_ply,
            compressed_ply=compressed_ply,
            local_score=local_score
        )
        
    except Exception as e:
        return GenerationResult(
            prompt=prompt,
            task_id=task_id,
            validator_uid=validator_uid,
            raw_ply=None,
            compressed_ply=None,
            local_score=-1.0,
            error=str(e)
        )

async def metagraph_syncer():
    """Periodically syncs with the metagraph to update validator information."""
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

async def task_processor():
    """Main task processing loop that pulls tasks and processes them."""
    global validator_states, active_generations
    bt.logging.info("Task processor started.")
    
    while not shutdown_event.is_set():
        if not validator_states:
            bt.logging.debug("No validators available. Waiting...")
            await asyncio.sleep(10)
            continue

        if len(active_generations) >= MAX_CONCURRENT_GENERATIONS:
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

            if current_time >= val_state.last_pulled and val_state.active_task is None:
                try:
                    bt.logging.debug(f"Attempting to pull task from UID {val_state.uid}...")
                    
                    if not val_state.axon_info or not val_state.axon_info.ip or not val_state.axon_info.port:
                        bt.logging.warning(f"Axon info missing for validator UID {val_state.uid}. Skipping.")
                        val_state.last_pulled = current_time + VALIDATOR_PULL_DELAY_SECONDS
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
                        
                        bt.logging.success(f"Pulled task '{task_id}' from UID {val_state.uid}: '{prompt[:70]}...'")
                        
                        if prompt in active_generations:
                            bt.logging.warning(f"Prompt already being processed. Skipping.")
                            continue
                        
                        active_generations.add(prompt)
                        val_state.active_task = prompt
                        
                        # Process the task
                        result = await process_task(prompt, task_id, hotkey, val_state.uid)
                        
                        # Update validator stats
                        if result.local_score >= 0:
                            val_state.success_count += 1
                            val_state.last_success = current_time
                            val_state.error_count = 0
                        else:
                            val_state.error_count += 1
                        
                        # Save the result
                        if result.local_score >= 0:
                            save_generated_asset(result)
                        
                        active_generations.remove(prompt)
                        val_state.active_task = None
                        
                        # Set cooldown
                        val_state.last_pulled = current_time + VALIDATOR_PULL_DELAY_SECONDS
                    else:
                        bt.logging.warning(f"Failed to pull task from UID {val_state.uid}. Status: {response_synapse.dendrite.status_code}")
                        val_state.error_count += 1
                        val_state.last_pulled = current_time + VALIDATOR_PULL_DELAY_SECONDS
                
                except Exception as e:
                    bt.logging.error(f"Exception processing task from UID {val_state.uid}: {e}\n{traceback.format_exc()}")
                    val_state.error_count += 1
                    val_state.last_pulled = current_time + VALIDATOR_PULL_DELAY_SECONDS
                
                await asyncio.sleep(0.1)
            
            await asyncio.sleep(1)

def handle_shutdown(signum, frame):
    """Handle graceful shutdown on SIGINT/SIGTERM."""
    bt.logging.info("Shutdown signal received. Initiating graceful shutdown...")
    shutdown_event.set()

async def main():
    """Main entry point for the local validation runner."""
    global subtensor, dendrite, wallet, metagraph

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
        
        # Start tasks
        tasks = [
            asyncio.create_task(metagraph_syncer()),
            asyncio.create_task(task_processor())
        ]
        
        running_tasks.update(tasks)
        bt.logging.info("Local validation runner started. Running...")
        
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
    asyncio.run(main()) 