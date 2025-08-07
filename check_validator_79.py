#!/usr/bin/env python3
"""
Simple script to check if validator UID 79 exists and is available
"""

import asyncio
import time
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

try:
    import bittensor as bt
    BITTENSOR_AVAILABLE = True
except ImportError:
    print("❌ Bittensor not available")
    BITTENSOR_AVAILABLE = False
    bt = None

async def check_validator_79():
    """Check if validator UID 79 exists and is available"""
    
    if not BITTENSOR_AVAILABLE:
        logger.error("❌ Bittensor not available")
        return
    
    try:
        logger.info("🔧 Setting up Bittensor...")
        
        # Create wallet
        wallet = bt.wallet(name="manbeast3b", hotkey="m3b")
        logger.info(f"✅ Wallet: {wallet.hotkey.ss58_address}")
        
        # Connect to testnet
        subtensor = bt.subtensor(network="test")
        logger.info("✅ Connected to testnet")
        
        # Load metagraph for subnet 17
        logger.info("📊 Loading metagraph for subnet 17...")
        metagraph = subtensor.metagraph(17)
        logger.info(f"✅ Metagraph loaded with {len(metagraph.neurons)} neurons")
        
        # Check if UID 79 exists
        target_uid = 79
        logger.info(f"🔍 Checking UID {target_uid}...")
        
        if target_uid >= len(metagraph.neurons):
            logger.error(f"❌ UID {target_uid} does not exist!")
            logger.info(f"   Available UIDs: 0 to {len(metagraph.neurons)-1}")
            
            # Show some available validators
            logger.info("📋 Some available validators:")
            for uid in range(min(10, len(metagraph.neurons))):
                neuron = metagraph.neurons[uid]
                if neuron.validator_permit:
                    logger.info(f"   UID {uid}: {float(neuron.stake):.1f} TAO (trust: {float(neuron.trust):.3f})")
            return
        
        # Get neuron info
        neuron = metagraph.neurons[target_uid]
        logger.info(f"✅ UID {target_uid} exists!")
        
        # Check if it's a validator
        if not neuron.validator_permit:
            logger.error(f"❌ UID {target_uid} is not a validator!")
            logger.info(f"   validator_permit: {neuron.validator_permit}")
            return
        
        logger.info(f"✅ UID {target_uid} is a validator!")
        logger.info(f"   Hotkey: {neuron.hotkey}")
        logger.info(f"   Stake: {float(neuron.stake):.2f} TAO")
        logger.info(f"   Trust: {float(neuron.trust):.3f}")
        logger.info(f"   Consensus: {float(neuron.consensus):.3f}")
        logger.info(f"   Axon info: {neuron.axon_info}")
        
        # Check if axon is available
        if hasattr(neuron.axon_info, 'ip') and neuron.axon_info.ip:
            logger.info(f"   IP: {neuron.axon_info.ip}")
            logger.info(f"   Port: {neuron.axon_info.port}")
            logger.info("✅ Validator appears to be online")
        else:
            logger.warning("⚠️ Validator axon info incomplete")
        
        # Try to pull a task
        logger.info("📡 Attempting to pull a task...")
        
        dendrite = bt.dendrite(wallet=wallet)
        
        try:
            from neurons.common.protocol import PullTask
            synapse = PullTask()
            synapse.timeout = 30
            
            start_time = time.time()
            response = await dendrite.forward(
                axons=[neuron.axon_info],
                synapse=synapse,
                timeout=30
            )
            query_time = time.time() - start_time
            
            if response and len(response) > 0:
                resp = response[0]
                if hasattr(resp, 'task') and resp.task and resp.task.prompt:
                    logger.info("🎉 SUCCESS! Task received!")
                    logger.info(f"   Task ID: {resp.task.id}")
                    logger.info(f"   Prompt: '{resp.task.prompt}'")
                    logger.info(f"   Query time: {query_time:.2f}s")
                else:
                    logger.warning("⚠️ No task available")
                    logger.info(f"   Response: {resp}")
            else:
                logger.error("❌ No response from validator")
                
        except ImportError:
            logger.error("❌ Could not import PullTask protocol")
        except Exception as e:
            logger.error(f"❌ Error pulling task: {e}")
    
    except Exception as e:
        logger.error(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

async def main():
    logger.info("🚀 Checking validator UID 79 on testnet subnet 17")
    await check_validator_79()

if __name__ == "__main__":
    asyncio.run(main()) 