#!/usr/bin/env python3
"""
Dynamic script to check current UID 142 information and update the blacklist if needed
"""

import bittensor as bt
import logging
import sys

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def get_current_uid_142_info():
    """Get current UID 142 information from metagraph"""
    try:
        logger.info("🔍 Fetching current metagraph data...")
        subtensor = bt.subtensor(network='archive')
        metagraph = subtensor.metagraph(netuid=103)

        uid = 142

        if uid >= len(metagraph.neurons):
            logger.error(f"❌ UID {uid} not found in metagraph")
            return None

        neuron = metagraph.neurons[uid]
        axon_info = neuron.axon_info

        info = {
            'uid': uid,
            'hotkey': metagraph.hotkeys[uid],
            'stake': float(metagraph.S[uid]),
            'trust': float(metagraph.trust[uid]),
            'consensus': float(metagraph.consensus[uid]),
            'incentive': float(metagraph.incentive[uid]),
            'axon_ip': getattr(axon_info, 'ip', None),
            'axon_port': getattr(axon_info, 'port', None),
            'axon_version': getattr(axon_info, 'version', None)
        }

        logger.info("✅ UID 142 CURRENT INFORMATION:")
        logger.info("=" * 60)
        logger.info(f"UID: {info['uid']}")
        logger.info(f"Hotkey: {info['hotkey']}")
        logger.info(f"Stake: {info['stake']:.4f} TAO")
        logger.info(f"Trust: {info['trust']:.4f}")
        logger.info(f"Consensus: {info['consensus']:.4f}")
        logger.info(f"Incentive: {info['incentive']:.4f}")
        logger.info(f"Axon IP: {info['axon_ip']}")
        logger.info(f"Axon Port: {info['axon_port']}")
        logger.info(f"Axon Version: {info['axon_version']}")

        # Check if axon endpoint is valid
        if info['axon_ip'] and info['axon_port'] and info['axon_ip'] != '0.0.0.0' and info['axon_port'] != 0:
            logger.info("✅ Axon endpoint appears valid")
            return info
        else:
            logger.warning("⚠️ Axon endpoint appears invalid")
            return info

    except Exception as e:
        logger.error(f"❌ Error fetching metagraph data: {e}")
        return None

def update_blacklist_if_needed(current_info):
    """Update the blacklist configuration if IP has changed"""
    if not current_info or not current_info['axon_ip']:
        return

    current_ip = current_info['axon_ip']
    hardcoded_ip = '129.146.3.173'

    if current_ip != hardcoded_ip:
        logger.warning(f"🔄 IP CHANGED: Hardcoded IP ({hardcoded_ip}) != Current IP ({current_ip})")
        logger.info("💡 RECOMMENDATION: Update the hardcoded IP in the orchestrator")

        # Read the current orchestrator file
        try:
            with open('continuous_trellis_orchestrator_working_a6000.py', 'r') as f:
                content = f.read()

            # Replace the hardcoded IP
            old_line = f"            if validator.uid == 142 and axon_info.ip == '{hardcoded_ip}':"
            new_line = f"            if validator.uid == 142 and axon_info.ip == '{current_ip}':"

            if old_line in content:
                content = content.replace(old_line, new_line)
                with open('continuous_trellis_orchestrator_working_a6000.py', 'w') as f:
                    f.write(content)
                logger.info(f"✅ Updated hardcoded IP from {hardcoded_ip} to {current_ip}")
            else:
                logger.warning("⚠️ Could not find hardcoded IP line to update")

        except Exception as e:
            logger.error(f"❌ Error updating orchestrator: {e}")
    else:
        logger.info(f"✅ IP matches: Current IP ({current_ip}) matches hardcoded IP ({hardcoded_ip})")

def main():
    """Main function"""
    logger.info("🌐 CHECKING CURRENT UID 142 INFORMATION")
    logger.info("This will verify if the IP/port is still correct")

    current_info = get_current_uid_142_info()

    if current_info:
        update_blacklist_if_needed(current_info)
    else:
        logger.error("❌ Could not retrieve current UID 142 information")

if __name__ == "__main__":
    main()

