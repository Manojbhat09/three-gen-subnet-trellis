#!/usr/bin/env python3
"""
Pull task from UID 4 (highest stake validator) on testnet subnet 17
"""

import asyncio
import time
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

async def pull_from_uid4():
    """Pull task from UID 4 on testnet"""
    
    try:
        import bittensor as bt
        
        # Setup
        wallet = bt.wallet(name="manbeast3b", hotkey="m3b")
        subtensor = bt.subtensor(network="test")
        metagraph = subtensor.metagraph(17)
        dendrite = bt.dendrite(wallet=wallet)
        
        logger.info(f"✅ Connected to testnet subnet 17")
        logger.info(f"✅ Metagraph has {len(metagraph.neurons)} neurons")
        
        # Check UID 4
        target_uid = 4
        if target_uid >= len(metagraph.neurons):
            logger.error(f"❌ UID {target_uid} does not exist!")
            return
        
        neuron = metagraph.neurons[target_uid]
        if not neuron.validator_permit:
            logger.error(f"❌ UID {target_uid} is not a validator!")
            return
        
        logger.info(f"✅ Found validator UID {target_uid}")
        logger.info(f"   Stake: {float(neuron.stake):.1f} TAO")
        logger.info(f"   Trust: {float(neuron.trust):.3f}")
        
        # Pull task
        logger.info("📡 Pulling task...")
        
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
                
                if hasattr(resp, 'validation_threshold'):
                    logger.info(f"   Validation threshold: {resp.validation_threshold}")
                
                return {
                    'task_id': resp.task.id,
                    'prompt': resp.task.prompt,
                    'validation_threshold': getattr(resp, 'validation_threshold', 0.6),
                    'query_time': query_time
                }
            else:
                logger.warning("⚠️ No task available")
                logger.info(f"   Response: {resp}")
        else:
            logger.error("❌ No response from validator")
            
    except Exception as e:
        logger.error(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

async def main():
    logger.info("🚀 Pulling task from UID 4 on testnet subnet 17")
    result = await pull_from_uid4()
    
    if result:
        logger.info("="*50)
        logger.info(f"Task ID: {result['task_id']}")
        logger.info(f"Prompt: {result['prompt']}")
        logger.info(f"Validation threshold: {result['validation_threshold']}")
        logger.info(f"Query time: {result['query_time']:.2f}s")

if __name__ == "__main__":
    asyncio.run(main()) 