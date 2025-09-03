#!/usr/bin/env python3
"""
Simple synchronous test to check UID 142 axon info from different networks
"""

import bittensor as bt
import logging
import sys

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_network_axon_info():
    """Test axon info from different networks"""

    uid = 142

    logger.info("🧪 TESTING UID 142 AXON INFO FROM DIFFERENT NETWORKS")
    logger.info("=" * 60)

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
                results[network] = None
                continue

            neuron = metagraph.neurons[uid]
            axon_info = neuron.axon_info

            results[network] = {
                'ip': getattr(axon_info, 'ip', None) if axon_info else None,
                'port': getattr(axon_info, 'port', None) if axon_info else None,
                'version': getattr(axon_info, 'version', None) if axon_info else None,
                'hotkey': metagraph.hotkeys[uid],
                'stake': float(metagraph.S[uid]),
            }

            # Validate axon info
            ip = results[network]['ip']
            port = results[network]['port']

            logger.info(f"Hotkey: {results[network]['hotkey']}")
            logger.info(f"Stake: {results[network]['stake']:.4f} TAO")
            logger.info(f"Axon IP: {ip}")
            logger.info(f"Axon Port: {port}")

            # Check validity
            is_valid = (axon_info and
                       hasattr(axon_info, 'ip') and
                       hasattr(axon_info, 'port') and
                       ip and port and
                       ip != '0.0.0.0' and
                       port != 0)

            if is_valid:
                logger.info("✅ Axon endpoint appears VALID")
                results[network]['valid'] = True
            else:
                logger.info("❌ Axon endpoint appears INVALID")
                results[network]['valid'] = False

        except Exception as e:
            logger.error(f"❌ Error checking {network}: {e}")
            results[network] = None

    # Compare results
    logger.info("\n" + "=" * 60)
    logger.info("📊 COMPARISON RESULTS")
    logger.info("=" * 60)

    if results.get('archive') and results.get('finney'):
        archive = results['archive']
        finney = results['finney']

        logger.info("ARCHIVE NETWORK:")
        logger.info(f"  Axon: {archive['ip']}:{archive['port']} {'✅' if archive['valid'] else '❌'}")

        logger.info("FINNEY NETWORK:")
        logger.info(f"  Axon: {finney['ip']}:{finney['port']} {'✅' if finney['valid'] else '❌'}")

        if archive['ip'] != finney['ip'] or archive['port'] != finney['port']:
            logger.warning("⚠️ AXON INFORMATION DIFFERENT BETWEEN NETWORKS!")
            if archive['valid'] and not finney['valid']:
                logger.info("🎯 RECOMMENDATION: Use ARCHIVE network - it has valid axon info!")
            elif finney['valid'] and not archive['valid']:
                logger.info("🎯 RECOMMENDATION: Use FINNEY network - it has valid axon info!")
            else:
                logger.info("🎯 RECOMMENDATION: Both networks have different info - test both!")
        else:
            logger.info("✅ Axon information is consistent between networks")
            if archive['valid']:
                logger.info("🎯 Both networks show valid axon info - either should work!")
            else:
                logger.info("❌ Both networks show invalid axon info - UID 142 may be offline!")

    return results

if __name__ == "__main__":
    test_network_axon_info()