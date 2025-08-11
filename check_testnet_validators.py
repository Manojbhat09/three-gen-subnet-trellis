#!/usr/bin/env python3

"""
Iterate over validators on Bittensor testnet (subnet 89) and attempt to pull tasks.
Usage: python check_testnet_validators.py [max_validators]
"""

import asyncio
import sys
import time
from datetime import datetime
from typing import List, Tuple

import bittensor as bt

try:
    from neurons.common.protocol import PullTask
except Exception as e:
    print(f"❌ Could not import PullTask from neurons.common.protocol: {e}")
    sys.exit(1)


def format_ts(ts: int) -> str:
    if not ts or ts <= 0:
        return "-"
    try:
        return datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M:%S")
    except Exception:
        return str(ts)


async def check_testnet_validators(max_validators: int = 20) -> None:
    # Configure wallet used for dendrite calls (adjust if needed)
    wallet = bt.wallet(name="manbeast3b", hotkey="m3b")

    # Connect to testnet
    subtensor = bt.subtensor(network="test")
    print("✅ Connected to testnet")

    # Load metagraph for testnet mirror of subnet 17 (UID 89)
    netuid = 89
    print(f"📊 Loading metagraph for subnet {netuid}...")
    metagraph = subtensor.metagraph(netuid)
    print(f"✅ Metagraph loaded: {len(metagraph.neurons)} neurons")

    # Initialize dendrite
    dendrite = bt.dendrite(wallet=wallet)

    # Collect active validators
    active: List[Tuple[int, float, object, str]] = []
    for uid, neuron in enumerate(metagraph.neurons):
        axon = metagraph.axons[uid]
        hotkey = metagraph.hotkeys[uid]
        stake = float(metagraph.stake[uid]) if hasattr(metagraph.stake[uid], '__float__') else float(metagraph.stake[uid])

        if neuron.validator_permit and axon.is_serving and axon.ip != '0.0.0.0':
            active.append((uid, stake, axon, hotkey))

    active.sort(key=lambda x: x[1], reverse=True)
    if not active:
        print("❌ No active validators found on testnet")
        return

    to_try = active[:max_validators]
    print(f"🔄 Trying {len(to_try)} validators (top by stake)...")

    tasks_found = 0
    for i, (uid, stake, axon, hotkey) in enumerate(to_try, 1):
        try:
            print(f"\n--- {i}/{len(to_try)} | UID {uid} | stake {stake:.1f} | {axon.ip}:{axon.port} ---")
            synapse = PullTask()
            synapse.timeout = 20

            start = time.time()
            response_list = await dendrite.forward(
                axons=[axon],
                synapse=synapse,
                timeout=20,
            )
            elapsed = time.time() - start

            if response_list and len(response_list) > 0:
                resp = response_list[0]
                if hasattr(resp, 'task') and resp.task and getattr(resp.task, 'prompt', None):
                    print("🎉 TASK AVAILABLE")
                    print(f"   Task ID: {getattr(resp.task, 'id', '-')}")
                    print(f"   Prompt: {resp.task.prompt}")
                    print(f"   Validation threshold: {getattr(resp, 'validation_threshold', '-')}")
                    print(f"   Took: {elapsed:.2f}s")
                    tasks_found += 1
                else:
                    cooldown = getattr(resp, 'cooldown_until', 0)
                    throttle = getattr(resp, 'throttle_period', 0)
                    print("⏳ No task. Cooldown/throttle info:")
                    print(f"   cooldown_until: {cooldown} ({format_ts(cooldown)})")
                    print(f"   throttle_period: {throttle}")
            else:
                print("❌ No response from validator")
        except Exception as e:
            print(f"💥 Error with UID {uid}: {str(e)[:200]}")

        await asyncio.sleep(0.5)

    print(f"\n📊 Summary: {tasks_found} validators returned a task out of {len(to_try)} tried")


if __name__ == "__main__":
    max_v = 20
    if len(sys.argv) > 1:
        try:
            max_v = int(sys.argv[1])
        except Exception:
            pass
    asyncio.run(check_testnet_validators(max_v)) 