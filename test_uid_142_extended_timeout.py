#!/usr/bin/env python3
"""
Test UID 142 with extended timeouts to see if we can make it work from this machine
"""

import bittensor as bt
import logging
import asyncio
import time
from neurons.common.protocol import PullTask

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

async def test_uid_142_extended_timeout():
    """Test UID 142 with extended timeout settings"""
    try:
        logger.info("🔍 TESTING UID 142 WITH EXTENDED TIMEOUTS")

        # Initialize with extended timeout
        subtensor = bt.subtensor(network='archive')
        metagraph = subtensor.metagraph(netuid=103)

        uid = 142
        if uid >= len(metagraph.neurons):
            logger.error(f"UID {uid} not found in metagraph")
            return

        neuron = metagraph.neurons[uid]
        axon_info = neuron.axon_info

        logger.info(f"📍 UID 142 CURRENT ENDPOINT: {axon_info.ip}:{axon_info.port}")
        logger.info(f"🔑 HOTKEY: {metagraph.hotkeys[uid]}")

        # Create dendrite with extended timeout
        wallet = bt.wallet()
        dendrite = bt.dendrite(wallet=wallet)

        # Test with different timeout values
        timeouts = [30, 60, 120]  # 30s, 1min, 2min

        for timeout in timeouts:
            logger.info(f"⏱️ TESTING WITH {timeout}s TIMEOUT")

            try:
                start_time = time.time()

                # Create synapse with extended timeout
                synapse = PullTask(timeout=timeout)

                # Try the call
                response = await dendrite.call(
                    target_axon=bt.axon(
                        info=axon_info,
                        wallet=wallet
                    ),
                    synapse=synapse
                )

                elapsed = time.time() - start_time
                logger.info(f"✅ RESPONSE RECEIVED in {elapsed:.2f}s")
                logger.info(f"📊 RESPONSE: {response}")
                logger.info(f"📊 RESPONSE ATTRIBUTES: {dir(response)}")

                if hasattr(response, 'task') and response.task:
                    logger.info("🎉 SUCCESS: Got a valid task!")
                    return True
                else:
                    logger.warning("⚠️ Got response but no valid task")

            except Exception as e:
                elapsed = time.time() - start_time
                logger.error(f"❌ FAILED after {elapsed:.2f}s with timeout {timeout}s: {e}")

                # If it's a timeout, continue to next timeout
                if "timeout" in str(e).lower():
                    logger.info(f"⏭️ Timeout occurred, trying next timeout value...")
                    continue
                else:
                    # Other error, might be permanent
                    logger.error(f"💥 Permanent error (not timeout): {e}")
                    break

        logger.error("❌ All timeout attempts failed")
        return False

    except Exception as e:
        logger.error(f"❌ Setup error: {e}")
        return False

def main():
    """Main function"""
    logger.info("🌐 TESTING UID 142 CONNECTIVITY WITH EXTENDED TIMEOUTS")
    logger.info("This will try different timeout values to see if UID 142 can work")

    result = asyncio.run(test_uid_142_extended_timeout())

    if result:
        logger.info("🎉 SUCCESS: UID 142 works with extended timeouts!")
    else:
        logger.info("❌ FAILED: UID 142 doesn't work even with extended timeouts")
        logger.info("💡 RECOMMENDATION: Consider skipping UID 142 or using different network")

if __name__ == "__main__":
    main()

