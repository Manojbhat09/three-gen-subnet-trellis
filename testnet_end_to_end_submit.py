#!/usr/bin/env python3
"""
Minimal end-to-end: Pull a task on testnet (UID 79 preferred),
use local generation server at http://127.0.0.1:8096 to generate 3D (SPZ compressed),
then submit results back to the validator as a miner.

Usage:
  python testnet_end_to_end_submit.py [uid]

Notes:
- Expects a running generation server on port 8096 with /generate/ endpoint.
- Uses wallet name 'manbeast3b' and hotkey 'm3b' (adjust if needed).
- Network: testnet mirror of SN17 (netuid 89).
"""

import asyncio
import base64
import sys
import time
from typing import Optional

import requests
import bittensor as bt

from neurons.common.protocol import PullTask, SubmitResults
from neurons.common.miner_license_consent_declaration import MINER_LICENSE_CONSENT_DECLARATION

GEN_SERVER_URL = "http://127.0.0.1:8096"
DEFAULT_UID = 79  # known working validator on testnet
NETUID = 89       # testnet mirror of SN17


def format_time_remaining(cooldown_until: int) -> str:
    """Return a human-readable remaining time until the given unix timestamp."""
    now = int(time.time())
    remaining_seconds = max(0, int(cooldown_until) - now)
    hours, remainder = divmod(remaining_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    parts = []
    if hours:
        parts.append(f"{hours}h")
    if minutes:
        parts.append(f"{minutes}m")
    parts.append(f"{seconds}s")
    return " ".join(parts)


async def pull_task_from_validator(wallet: bt.wallet, validator_uid: int) -> Optional[PullTask]:
    subtensor = bt.subtensor(network="test")
    metagraph = subtensor.metagraph(NETUID)
    dendrite = bt.dendrite(wallet=wallet)

    if validator_uid >= len(metagraph.neurons):
        print(f"❌ UID {validator_uid} out of range ({len(metagraph.neurons)} neurons)")
        return None

    neuron = metagraph.neurons[validator_uid]
    if not neuron.validator_permit:
        print(f"❌ UID {validator_uid} is not a validator")
        return None

    axon = metagraph.axons[validator_uid]

    print(f"📡 Pulling task from UID {validator_uid} ({metagraph.hotkeys[validator_uid]})...")
    synapse = PullTask()
    synapse.timeout = 60

    responses = await dendrite.forward(axons=[axon], synapse=synapse, timeout=60)
    if not responses:
        print("❌ No response from validator")
        return None

    resp = responses[0]
    if hasattr(resp, "task") and resp.task and getattr(resp.task, "prompt", None):
        print("✅ Task received")
        print(f"   Task ID: {getattr(resp.task, 'id', '-')}")
        print(f"   Prompt: {resp.task.prompt}")
        print(f"   Validation threshold: {getattr(resp, 'validation_threshold', 0.6)}")
        return resp

    cooldown = getattr(resp, "cooldown_until", 0)
    print("⏳ No task available")
    if cooldown and cooldown > 0:
        now = int(time.time())
        remaining = max(0, int(cooldown) - now)
        human_remaining = format_time_remaining(int(cooldown))
        print(f"   cooldown_until: {cooldown} (now: {now}, remaining: {remaining}s ≈ {human_remaining})")
    return None


def generate_with_local_server(prompt: str, seed: int = 42, return_compressed: bool = True) -> Optional[bytes]:
    url = f"{GEN_SERVER_URL}/generate/"
    print(f"🎨 Generating 3D via {url}")
    try:
        # FastAPI Form expects form-encoded fields
        data = {
            "prompt": prompt,
            "seed": str(seed),
            "return_compressed": "true" if return_compressed else "false",
        }
        r = requests.post(url, data=data, timeout=600)
        if r.status_code != 200:
            print(f"❌ Generation failed: HTTP {r.status_code}")
            return None
        print(f"✅ Generation complete: {len(r.content):,} bytes")
        return r.content
    except Exception as e:
        print(f"❌ Generation error: {e}")
        return None


async def submit_results(wallet: bt.wallet, validator_uid: int, pull_response: PullTask, results_bytes: bytes) -> Optional[SubmitResults]:
    subtensor = bt.subtensor(network="test")
    metagraph = subtensor.metagraph(NETUID)
    dendrite = bt.dendrite(wallet=wallet)

    axon = metagraph.axons[validator_uid]
    validator_hotkey = metagraph.hotkeys[validator_uid]

    # Prepare submit synapse
    submit_time = time.time_ns()

    # Signature payload
    message = f"{MINER_LICENSE_CONSENT_DECLARATION}{submit_time}{pull_response.task.prompt}{validator_hotkey}{wallet.hotkey.ss58_address}"
    signature_bytes = wallet.hotkey.sign(message.encode("utf-8"))
    signature_b64 = base64.b64encode(signature_bytes).decode("utf-8")

    submit = SubmitResults(
        task=pull_response.task,
        results=base64.b64encode(results_bytes).decode("utf-8"),
        data_format="ply",
        data_ver=0,
        compression=2,  # SPZ
        submit_time=submit_time,
        signature=signature_b64,
    )

    print("📤 Submitting results to validator...")
    responses = await dendrite.forward(axons=[axon], synapse=submit, timeout=120)
    if not responses:
        print("❌ No response to submission")
        return None

    resp = responses[0]
    feedback = getattr(resp, "feedback", None)
    cooldown_until = getattr(resp, "cooldown_until", 0)

    print("✅ Submission acknowledged")
    if feedback is not None:
        print(f"   validation_failed: {getattr(feedback, 'validation_failed', False)}")
        print(f"   task_fidelity_score: {getattr(feedback, 'task_fidelity_score', 0.0):.4f}")
        print(f"   average_fidelity_score: {getattr(feedback, 'average_fidelity_score', 0.0):.4f}")
        print(f"   generations_within_the_window: {getattr(feedback, 'generations_within_the_window', 0)}")
        print(f"   current_miner_reward: {getattr(feedback, 'current_miner_reward', 0.0):.6f}")
    if cooldown_until:
        now = int(time.time())
        remaining = max(0, int(cooldown_until) - now)
        human_remaining = format_time_remaining(int(cooldown_until))
        print(f"   cooldown_until: {cooldown_until} (now: {now}, remaining: {remaining}s ≈ {human_remaining})")

    return resp


async def main():
    # Optional CLI arg for UID
    target_uid = DEFAULT_UID
    if len(sys.argv) > 1:
        try:
            target_uid = int(sys.argv[1])
        except Exception:
            pass

    # Wallet
    wallet = bt.wallet(name="manbeast3b", hotkey="m3bnew")
    print(f"🔑 Miner: {wallet.hotkey.ss58_address}")

    # Pull task
    pull_resp = await pull_task_from_validator(wallet, target_uid)
    if not pull_resp:
        print("❌ Could not obtain a task. Exiting.")
        return

    prompt = pull_resp.task.prompt

    # Generate locally
    result_bytes = generate_with_local_server(prompt, seed=42, return_compressed=True)
    if not result_bytes:
        print("❌ Generation failed. Exiting.")
        return

    # Submit results
    await submit_results(wallet, target_uid, pull_resp, result_bytes)


if __name__ == "__main__":
    asyncio.run(main()) 