#!/usr/bin/env python3
"""
Compare UID 142 information between different Bittensor networks
"""

import bittensor as bt
import logging
import sys

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def compare_networks():
    """Compare UID 142 info between different networks"""
    uid = 142

    networks = ['archive', 'finney']
    results = {}

    for network in networks:
        try:
            logger.info(f"\n🔍 Checking {network.upper()} network...")

            # Initialize subtensor and metagraph
            subtensor = bt.subtensor(network=network)
            metagraph = subtensor.metagraph(netuid=103)

            if uid >= len(metagraph.neurons):
                logger.error(f"UID {uid} not found in {network} network")
                continue

            neuron = metagraph.neurons[uid]
            axon_info = neuron.axon_info

            results[network] = {
                'hotkey': metagraph.hotkeys[uid],
                'stake': float(metagraph.S[uid]),
                'trust': float(metagraph.trust[uid]),
                'consensus': float(metagraph.consensus[uid]),
                'incentive': float(metagraph.incentive[uid]),
                'axon_ip': getattr(axon_info, 'ip', None) if axon_info else None,
                'axon_port': getattr(axon_info, 'port', None) if axon_info else None,
                'axon_version': getattr(axon_info, 'version', None) if axon_info else None,
            }

            logger.info(f"Hotkey: {results[network]['hotkey']}")
            logger.info(f"Stake: {results[network]['stake']:.4f} TAO")
            logger.info(f"Trust: {results[network]['trust']:.4f}")
            logger.info(f"Axon IP: {results[network]['axon_ip']}")
            logger.info(f"Axon Port: {results[network]['axon_port']}")

        except Exception as e:
            logger.error(f"Error checking {network}: {e}")
            results[network] = None

    # Compare results
    logger.info("\n" + "="*60)
    logger.info("📊 NETWORK COMPARISON FOR UID 142")
    logger.info("="*60)

    if 'archive' in results and 'finney' in results and results['archive'] and results['finney']:
        archive = results['archive']
        finney = results['finney']

        logger.info("ARCHIVE NETWORK:")
        logger.info(f"  Axon: {archive['axon_ip']}:{archive['axon_port']}")

        logger.info("FINNEY NETWORK:")
        logger.info(f"  Axon: {finney['axon_ip']}:{finney['axon_port']}")

        if archive['axon_ip'] != finney['axon_ip'] or archive['axon_port'] != finney['axon_port']:
            logger.warning("⚠️ AXON INFORMATION DIFFERENT BETWEEN NETWORKS!")
            logger.warning("This explains why the orchestrator sees different axon info!")
        else:
            logger.info("✅ Axon information is consistent between networks")

    return results

if __name__ == "__main__":
    compare_networks()
