#!/usr/bin/env python3
"""
Test script to pull a task from UID 142 using axon info from archive network
"""

import bittensor as bt
import asyncio
import logging
import time

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def test_uid_142_pull_task():
    """Test pulling a task from UID 142 using archive network axon info"""

    uid = 142

    try:
        logger.info("🔍 Getting UID 142 axon info from ARCHIVE network...")

        # Get axon info from archive network
        archive_subtensor = bt.subtensor(network='archive')
        archive_metagraph = archive_subtensor.metagraph(netuid=103)

        if uid >= len(archive_metagraph.neurons):
            logger.error(f"UID {uid} not found in archive network")
            return False

        neuron = archive_metagraph.neurons[uid]
        axon_info = neuron.axon_info

        logger.info("📡 AXON INFORMATION FROM ARCHIVE:")
        logger.info("=" * 50)
        logger.info(f"IP: {getattr(axon_info, 'ip', 'None')}")
        logger.info(f"Port: {getattr(axon_info, 'port', 'None')}")
        logger.info(f"Version: {getattr(axon_info, 'version', 'None')}")
        logger.info(f"Hotkey: {archive_metagraph.hotkeys[uid]}")

        # Validate axon info
        if not axon_info or not hasattr(axon_info, 'ip') or not hasattr(axon_info, 'port'):
            logger.error("❌ Invalid axon info structure")
            return False

        if not axon_info.ip or not axon_info.port or axon_info.ip == '0.0.0.0' or axon_info.port == 0:
            logger.error("❌ Axon endpoint appears invalid")
            return False

        logger.info("✅ Axon endpoint appears valid")

        # Now test the actual pull task
        logger.info("\n🚀 TESTING TASK PULL FROM UID 142...")
        logger.info("=" * 50)

        # Create dendrite for testing
        wallet = bt.wallet()
        dendrite = bt.dendrite(wallet=wallet)

        # Create PullTask synapse
        synapse = bt.neurons.protocol.PullTask()

        start_time = time.time()

        try:
            logger.info(f"📡 Attempting to pull task from {axon_info.ip}:{axon_info.port}...")

            # Make the call with a reasonable timeout
            response = await dendrite.call(
                target_axon=axon_info,
                synapse=synapse,
                deserialize=False,
                timeout=30.0  # 30 second timeout
            )

            end_time = time.time()
            process_time = end_time - start_time

            logger.info("🎯 PULL TASK RESULT:")
            logger.info("=" * 50)

            if response and hasattr(response, 'is_success') and response.is_success:
                logger.info("✅ SUCCESS: Task pulled successfully!"                logger.info(f"Status Code: {getattr(response.dendrite, 'status_code', 'Unknown')}")
                logger.info(f"Process Time: {process_time:.2f}s")
                logger.info(f"Response Type: {type(response)}")

                # Check if we actually got a task
                if hasattr(response, 'task') and response.task:
                    logger.info("✅ Task data received!")
                    logger.info(f"Task ID: {getattr(response.task, 'id', 'Unknown')}")
                    logger.info(f"Task Prompt: {getattr(response.task, 'prompt', 'Unknown')}")
                else:
                    logger.warning("⚠️ No task data in response")

                return True

            else:
                logger.error("❌ FAILED: Pull task unsuccessful")
                logger.error(f"Status Code: {getattr(response.dendrite, 'status_code', 'Unknown') if response else 'No response'}")
                logger.error(f"Status Message: {getattr(response.dendrite, 'status_message', 'Unknown') if response else 'No response'}")
                logger.error(f"Process Time: {process_time:.2f}s")
                return False

        except Exception as e:
            end_time = time.time()
            process_time = end_time - start_time
            logger.error(f"❌ EXCEPTION during pull task: {e}")
            logger.error(f"Process Time: {process_time:.2f}s")
            return False

        finally:
            # Clean up
            await dendrite.close()

    except Exception as e:
        logger.error(f"❌ Error in test: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main function"""
    logger.info("🧪 TESTING UID 142 TASK PULL CAPABILITY")
    logger.info("=" * 60)

    # Run the async test
    success = asyncio.run(test_uid_142_pull_task())

    logger.info("\n" + "=" * 60)
    if success:
        logger.info("✅ CONCLUSION: UID 142 is reachable and can provide tasks!")
        logger.info("   The archive network axon info appears to be valid.")
        logger.info("   RECOMMENDATION: Switch orchestrator to use 'archive' network.")
    else:
        logger.info("❌ CONCLUSION: UID 142 pull task failed.")
        logger.info("   The axon info may be stale or the validator may be offline.")
        logger.info("   RECOMMENDATION: Keep using 'finney' network or investigate further.")
    logger.info("=" * 60)

if __name__ == "__main__":
    main()
