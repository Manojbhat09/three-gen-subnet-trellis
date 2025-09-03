#!/usr/bin/env python3
"""
Debug script to check current axon info for UID 142 from different networks
"""

import bittensor as bt
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def debug_uid_142():
    """Debug current axon info for UID 142"""

    uid = 142

    logger.info("🔍 DEBUGGING UID 142 CURRENT AXON INFO")
    logger.info("=" * 60)

    networks = ['archive', 'finney']
    results = {}

    for network in networks:
        try:
            logger.info(f"\n🌐 Checking {network.upper()} network...")

            subtensor = bt.subtensor(network=network)
            metagraph = subtensor.metagraph(netuid=17)  # Your actual netuid

            if uid >= len(metagraph.neurons):
                logger.error(f"❌ UID {uid} not found in {network} network")
                continue

            neuron = metagraph.neurons[uid]
            axon_info = neuron.axon_info

            results[network] = {
                'ip': getattr(axon_info, 'ip', None) if axon_info else None,
                'port': getattr(axon_info, 'port', None) if axon_info else None,
                'version': getattr(axon_info, 'version', None) if axon_info else None,
                'hotkey': metagraph.hotkeys[uid],
            }

            logger.info(f"Hotkey: {results[network]['hotkey']}")
            logger.info(f"Axon IP: {results[network]['ip']}")
            logger.info(f"Axon Port: {results[network]['port']}")
            logger.info(f"Axon Version: {results[network]['version']}")

            # Check validity
            ip = results[network]['ip']
            port = results[network]['port']

            if ip and port and ip != '0.0.0.0' and port != 0:
                logger.info("✅ Axon endpoint appears VALID")
                results[network]['valid'] = True
            else:
                logger.info("❌ Axon endpoint appears INVALID")
                results[network]['valid'] = False

        except Exception as e:
            logger.error(f"❌ Error checking {network}: {e}")
            results[network] = None

    # Analysis
    logger.info("\n" + "=" * 60)
    logger.info("📊 ANALYSIS")
    logger.info("=" * 60)

    if results.get('archive') and results.get('finney'):
        archive = results['archive']
        finney = results['finney']

        logger.info("ARCHIVE NETWORK:")
        logger.info(f"  Axon: {archive['ip']}:{archive['port']} {'✅' if archive['valid'] else '❌'}")

        logger.info("FINNEY NETWORK:")
        logger.info(f"  Axon: {finney['ip']}:{finney['port']} {'✅' if finney['valid'] else '❌'}")

        logger.info("\nLOG ISSUE:")
        logger.info("  Log shows: 129.146.3.173:8092")
        logger.info(f"  Archive shows: {archive['ip']}:{archive['port']}")
        logger.info(f"  Finney shows: {finney['ip']}:{finney['port']}")

        if archive['ip'] == '129.146.3.173' or finney['ip'] == '129.146.3.173':
            logger.warning("⚠️ FOUND THE PROBLEM: Networks are returning 129.146.3.173!")
            logger.warning("This explains why the log shows this IP - it's from the metagraph!")
        else:
            logger.info("✅ Neither network returns 129.146.3.173 - different issue")

    return results

if __name__ == "__main__":
    debug_uid_142()
