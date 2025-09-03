#!/usr/bin/env python3
"""
Quick debug script to investigate validator issues:
1. UID 142 - why is it completely broken (503/408 errors)
2. UID 81 - why no cooldown between pulls (violations increasing)
"""

import asyncio
import time
import bittensor as bt
from neurons.common.protocol import PullTask
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

async def debug_validator_issues():
    """Debug specific validator issues"""
    
    # Initialize bittensor components
    wallet = bt.wallet(name="default", hotkey="default")
    subtensor = bt.subtensor(network="finney")
    metagraph = subtensor.metagraph(netuid=26)
    dendrite = bt.dendrite(wallet=wallet)
    
    logger.info("🔍 Starting validator debug session...")
    
    # Test validators
    test_uids = [81, 142, 128]  # Problem validators + one working
    
    for uid in test_uids:
        logger.info(f"\n{'='*60}")
        logger.info(f"🔍 DEBUGGING UID {uid}")
        logger.info(f"{'='*60}")
        
        if uid >= len(metagraph.hotkeys):
            logger.error(f"❌ UID {uid} is out of range (max: {len(metagraph.hotkeys)-1})")
            continue
            
        # Get validator info
        hotkey = metagraph.hotkeys[uid]
        stake = metagraph.S[uid]
        trust = metagraph.T[uid]
        
        logger.info(f"📊 Validator Info:")
        logger.info(f"   Hotkey: {hotkey}")
        logger.info(f"   Stake: {stake:.1f} TAO")
        logger.info(f"   Trust: {trust:.3f}")
        
        # Get axon info
        neuron = metagraph.neurons[uid]
        axon_info = neuron.axon_info
        
        logger.info(f"🌐 Network Info:")
        logger.info(f"   IP: {axon_info.ip}")
        logger.info(f"   Port: {axon_info.port}")
        logger.info(f"   IP Type: {type(axon_info.ip)}")
        logger.info(f"   Port Type: {type(axon_info.port)}")
        
        # Check if axon info is valid
        if not axon_info or not axon_info.ip or not axon_info.port:
            logger.error(f"❌ UID {uid} has invalid axon info!")
            continue
            
        if axon_info.ip == "0.0.0.0" or axon_info.port == 0:
            logger.error(f"❌ UID {uid} has placeholder axon info: {axon_info.ip}:{axon_info.port}")
            continue
            
        # Test connection
        logger.info(f"🔌 Testing connection to {axon_info.ip}:{axon_info.port}")
        
        try:
            # Create PullTask synapse
            synapse = PullTask()
            
            # Test the connection
            start_time = time.time()
            response = await dendrite.call(
                target_axon=axon_info,
                synapse=synapse,
                deserialize=False,
                timeout=10  # Short timeout for testing
            )
            query_time = time.time() - start_time
            
            logger.info(f"✅ Connection successful!")
            logger.info(f"   Status Code: {response.dendrite.status_code}")
            logger.info(f"   Status Message: {response.dendrite.status_message}")
            logger.info(f"   Process Time: {response.dendrite.process_time}")
            logger.info(f"   Query Time: {query_time:.3f}s")
            
            # Check response fields
            if hasattr(response, 'cooldown_until'):
                logger.info(f"   Cooldown Until: {response.cooldown_until}")
            if hasattr(response, 'cooldown_violations'):
                logger.info(f"   Cooldown Violations: {response.cooldown_violations}")
            if hasattr(response, 'throttle_period'):
                logger.info(f"   Throttle Period: {response.throttle_period}")
            if hasattr(response, 'task') and response.task:
                logger.info(f"   Task ID: {response.task.id}")
                logger.info(f"   Task Prompt: {response.task.prompt[:50]}...")
            else:
                logger.info(f"   Task: None")
                
        except Exception as e:
            logger.error(f"❌ Connection failed: {e}")
            logger.error(f"   Exception type: {type(e)}")
            
        # Wait between tests
        await asyncio.sleep(2)
    
    logger.info(f"\n{'='*60}")
    logger.info("🔍 DEBUG SESSION COMPLETE")
    logger.info(f"{'='*60}")

if __name__ == "__main__":
    asyncio.run(debug_validator_issues())


