#!/usr/bin/env python3
"""
Simple script to test UID 142 connectivity
"""

import asyncio
import time
import bittensor as bt
from neurons.common.protocol import PullTask
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

async def test_uid_142():
    """Test UID 142 connectivity"""

    # Initialize bittensor components with error handling
    try:
        # Try to find existing wallet
        wallet = bt.wallet()
        if not wallet.hotkey_file.exists() or not wallet.coldkey_file.exists():
            logger.error("❌ No valid wallet found. Please create or configure a wallet first.")
            logger.info("💡 To create a wallet, run: btcli wallet new")
            logger.info("💡 To regenerate keys, run: btcli wallet regen-hotkey")
            return

        subtensor = bt.subtensor(network="finney")
        metagraph = subtensor.metagraph(netuid=26)
        dendrite = bt.dendrite(wallet=wallet)
    except Exception as e:
        logger.error(f"❌ Failed to initialize Bittensor components: {e}")
        logger.info("💡 Make sure you have a valid wallet configured")
        return
    
    uid = 142
    
    logger.info(f"🔍 Testing UID {uid} connectivity...")
    
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
    
    # Test connection with different timeouts
    timeouts = [5, 10, 20, 40]
    
    for timeout in timeouts:
        logger.info(f"\n🔌 Testing with {timeout}s timeout...")
        
        try:
            synapse = PullTask()
            start_time = time.time()
            
            response = await dendrite.call(
                target_axon=axon_info,
                synapse=synapse,
                deserialize=False,
                timeout=timeout
            )
            
            query_time = time.time() - start_time
            
            logger.info(f"✅ SUCCESS with {timeout}s timeout!")
            logger.info(f"   Status Code: {response.dendrite.status_code}")
            logger.info(f"   Status Message: {response.dendrite.status_message}")
            logger.info(f"   Process Time: {response.dendrite.process_time}")
            logger.info(f"   Query Time: {query_time:.3f}s")
            
            if hasattr(response, 'task') and response.task:
                logger.info(f"   Task ID: {response.task.id}")
                logger.info(f"   Task Prompt: {response.task.prompt[:50]}...")
            else:
                logger.info(f"   Task: None")
                
            break  # Success, no need to test longer timeouts
            
        except Exception as e:
            logger.error(f"❌ FAILED with {timeout}s timeout: {e}")
            if timeout < max(timeouts):
                logger.info(f"   Trying longer timeout...")
            else:
                logger.error(f"   All timeout tests failed!")
    
    logger.info(f"\n🔍 Test complete!")

if __name__ == "__main__":
    asyncio.run(test_uid_142())
