#!/usr/bin/env python3
"""
Quick script to check UID 142's current neuron information from metagraph
"""

import bittensor as bt
import logging
import sys

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def check_uid_info():
    """Check UID 142's current information"""
    try:
        # Initialize subtensor and metagraph
        subtensor = bt.subtensor(network='archive')
        metagraph = subtensor.metagraph(netuid=103)

        uid = 142

        if uid >= len(metagraph.neurons):
            logger.error(f"UID {uid} not found in metagraph")
            return

        neuron = metagraph.neurons[uid]
        axon_info = neuron.axon_info

        logger.info("🔍 UID 142 CURRENT INFORMATION:")
        logger.info("=" * 50)
        logger.info(f"UID: {uid}")
        logger.info(f"Hotkey: {metagraph.hotkeys[uid]}")
        logger.info(f"Stake: {metagraph.S[uid]:.4f} TAO")
        logger.info(f"Trust: {metagraph.trust[uid]:.4f}")
        logger.info(f"Consensus: {metagraph.consensus[uid]:.4f}")
        logger.info(f"Incentive: {metagraph.incentive[uid]:.4f}")

        logger.info("\n📡 AXON INFORMATION:")
        logger.info(f"IP: {getattr(axon_info, 'ip', 'None')}")
        logger.info(f"Port: {getattr(axon_info, 'port', 'None')}")
        logger.info(f"Version: {getattr(axon_info, 'version', 'None')}")

        # Check if axon info is valid
        if axon_info and hasattr(axon_info, 'ip') and hasattr(axon_info, 'port'):
            if axon_info.ip and axon_info.port and axon_info.ip != '0.0.0.0' and axon_info.port != 0:
                logger.info("✅ Axon endpoint appears valid")
            else:
                logger.warning("⚠️ Axon endpoint appears invalid")
        else:
            logger.warning("⚠️ No axon info available")

        return axon_info

    except Exception as e:
        logger.error(f"Error checking UID {uid}: {e}")
        return None

if __name__ == "__main__":
    check_uid_info()

