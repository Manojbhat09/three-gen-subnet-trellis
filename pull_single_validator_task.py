#!/usr/bin/env python3
"""
Simple script to pull a task from a single validator on testnet
Usage: python pull_single_validator_task.py
"""

import asyncio
import time
import logging
import traceback
from typing import Optional

# Setup logging with debug level
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Don't modify bittensor logging to avoid issues

# Make bittensor optional for environments without it
try:
    import bittensor as bt
    BITTENSOR_AVAILABLE = True
except ImportError:
    print("❌ Bittensor not available - please install bittensor")
    BITTENSOR_AVAILABLE = False
    bt = None

async def pull_task_from_validator_79():
    """Pull a task from validator UID 79 on testnet"""
    
    logger.debug("🔍 Starting pull_task_from_validator_79 function")
    
    if not BITTENSOR_AVAILABLE:
        logger.error("❌ Bittensor not available")
        return None
    
    try:
        # Setup Bittensor components
        logger.info("🔧 Setting up Bittensor components...")
        logger.debug("   Creating wallet...")
        
        # Create wallet (you'll need to have registered first)
        logger.debug("   Creating wallet with name='manbeast3b', hotkey='m3b'")
        wallet = bt.wallet(name="manbeast3b", hotkey="m3b")
        logger.info(f"✅ Wallet loaded: {wallet.hotkey.ss58_address}")
        logger.debug(f"   Wallet hotkey address: {wallet.hotkey.ss58_address}")
        
        # Connect to testnet
        logger.debug("   Connecting to testnet subtensor...")
        subtensor = bt.subtensor(network="test")
        logger.info("✅ Connected to testnet")
        logger.debug(f"   Subtensor chain endpoint: {subtensor.chain_endpoint}")
        
        # Initialize dendrite
        logger.debug("   Initializing dendrite...")
        dendrite = bt.dendrite(wallet=wallet)
        logger.info("✅ Dendrite initialized")
        logger.debug(f"   Dendrite wallet: {wallet.hotkey.ss58_address}")
        
        # Load metagraph for subnet 89
        netuid = 89
        logger.debug(f"   Loading metagraph for netuid {netuid}...")
        metagraph = subtensor.metagraph(netuid)
        logger.info(f"✅ Metagraph loaded (netuid: {netuid})")
        logger.debug(f"   Metagraph has {len(metagraph.neurons)} neurons")
        
        # Check if UID 79 exists and is a validator
        target_uid = 79
        logger.debug(f"   Checking if UID {target_uid} exists...")
        
        if target_uid >= len(metagraph.neurons):
            logger.error(f"❌ UID {target_uid} does not exist on subnet {netuid}")
            logger.debug(f"   Available UIDs: 0 to {len(metagraph.neurons)-1}")
            return None
        
        logger.debug(f"   UID {target_uid} exists, getting neuron info...")
        neuron = metagraph.neurons[target_uid]
        logger.debug(f"   Neuron validator_permit: {neuron.validator_permit}")
        
        if not neuron.validator_permit:
            logger.error(f"❌ UID {target_uid} is not a validator")
            return None
        
        logger.info(f"✅ Found validator UID {target_uid}")
        logger.info(f"   Hotkey: {neuron.hotkey}")
        logger.info(f"   Stake: {float(neuron.stake):.2f} TAO")
        logger.info(f"   Trust: {float(neuron.trust):.3f}")
        logger.info(f"   Consensus: {float(neuron.consensus):.3f}")
        logger.debug(f"   Neuron axon_info: {neuron.axon_info}")
        
        # Import protocol
        logger.debug("   Importing PullTask protocol...")
        try:
            from neurons.common.protocol import PullTask
            logger.debug("   ✅ PullTask protocol imported successfully")
        except ImportError as e:
            logger.error("❌ Could not import PullTask protocol - make sure you're in the correct directory")
            logger.debug(f"   Import error: {e}")
            return None
        
        # Create task pull request
        logger.debug("   Creating PullTask synapse...")
        synapse = PullTask()
        synapse.timeout = 60  # 60 second timeout
        logger.debug(f"   Synapse timeout set to {synapse.timeout}s")
        
        logger.info(f"📡 Pulling task from UID {target_uid}...")
        logger.debug(f"   Using axon_info: {neuron.axon_info}")
        
        start_time = time.time()
        
        # Query the validator
        logger.debug("   Sending dendrite.forward request...")
        response = await dendrite.forward(
            axons=[neuron.axon_info],
            synapse=synapse,
            timeout=60
        )
        logger.debug(f"   Dendrite response received: {response}")
        
        query_time = time.time() - start_time
        logger.debug(f"   Query completed in {query_time:.2f}s")
        
        if response and len(response) > 0:
            resp = response[0]
            logger.debug(f"   Response object type: {type(resp)}")
            logger.debug(f"   Response attributes: {dir(resp)}")
            
            if hasattr(resp, 'task') and resp.task and resp.task.prompt:
                logger.info(f"✅ Task received from UID {target_uid}!")
                logger.info(f"   Task ID: {resp.task.id}")
                logger.info(f"   Prompt: '{resp.task.prompt}'")
                logger.info(f"   Query time: {query_time:.2f}s")
                
                # Check for validation threshold
                if hasattr(resp, 'validation_threshold'):
                    logger.info(f"   Validation threshold: {resp.validation_threshold}")
                
                # Check for cooldown
                if hasattr(resp, 'cooldown_until') and resp.cooldown_until:
                    logger.info(f"   Cooldown until: {resp.cooldown_until}")
                
                return {
                    'task_id': resp.task.id,
                    'prompt': resp.task.prompt,
                    'validation_threshold': getattr(resp, 'validation_threshold', 0.6),
                    'cooldown_until': getattr(resp, 'cooldown_until', None),
                    'query_time': query_time
                }
            else:
                logger.warning(f"⚠️ No task received from UID {target_uid}")
                logger.debug(f"   Response has task: {hasattr(resp, 'task')}")
                if hasattr(resp, 'task'):
                    logger.debug(f"   Task object: {resp.task}")
                    if resp.task:
                        logger.debug(f"   Task has prompt: {hasattr(resp.task, 'prompt')}")
                        if hasattr(resp.task, 'prompt'):
                            logger.debug(f"   Task prompt: {resp.task.prompt}")
                logger.info(f"   Response: {resp}")
                return None
        else:
            logger.error(f"❌ No response from UID {target_uid}")
            logger.debug(f"   Response: {response}")
            logger.debug(f"   Response length: {len(response) if response else 0}")
            return None
    
    except Exception as e:
        logger.error(f"❌ Error pulling task: {e}")
        logger.debug(f"   Exception type: {type(e)}")
        logger.debug(f"   Exception args: {e.args}")
        traceback.print_exc()
        return None

async def main():
    """Main function"""
    logger.info("🚀 Starting single validator task pull...")
    logger.info("   Target: UID 79 on testnet subnet 89")
    logger.info("   Wallet: manbeast3b/m3b")
    logger.debug("   Debug logging enabled")
    
    try:
        result = await pull_task_from_validator_79()
        
        if result:
            logger.info("🎉 Task pull successful!")
            logger.info("="*50)
            logger.info(f"Task ID: {result['task_id']}")
            logger.info(f"Prompt: {result['prompt']}")
            logger.info(f"Validation threshold: {result['validation_threshold']}")
            logger.info(f"Query time: {result['query_time']:.2f}s")
            if result['cooldown_until']:
                logger.info(f"Cooldown until: {result['cooldown_until']}")
        else:
            logger.error("❌ Task pull failed")
    except Exception as e:
        logger.error(f"❌ Main function failed: {e}")
        logger.debug(f"   Main exception type: {type(e)}")
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(main()) 