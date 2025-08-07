#!/usr/bin/env python3
"""
Test script to pull a task from a single validator using the existing orchestrator code
Usage: python test_single_validator.py
"""

import asyncio
import time
import logging
from continuous_trellis_orchestrator import ContinuousTrellisOrchestrator

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

async def test_single_validator():
    """Test pulling a task from UID 79 on testnet"""
    
    # Create a minimal config for testing
    config = {
        'wallet_name': 'manbeast3b',
        'hotkey_name': 'm3b',
        'netuid': 17,
        'harvest_tasks': False,  # We'll manually pull
        'validate_generations': False,  # Skip validation for testing
        'submit_results': False,  # Skip submission for testing
        'generation_server_url': 'http://localhost:8096',
        'validation_server_url': 'http://localhost:10006',
        'output_dir': './test_outputs',
        'enable_prompt_optimization': False,  # Disable for testing
    }
    
    # Create orchestrator
    orchestrator = ContinuousTrellisOrchestrator(config)
    
    try:
        # Setup Bittensor
        if not orchestrator._setup_bittensor():
            logger.error("❌ Failed to setup Bittensor")
            return
        
        # Refresh validators to get current metagraph
        orchestrator.refresh_validators()
        
        # Find UID 79
        target_uid = 79
        
        if target_uid not in orchestrator.validators:
            logger.error(f"❌ UID {target_uid} not found in active validators")
            logger.info("Available validators:")
            for uid in sorted(orchestrator.validators.keys()):
                validator = orchestrator.validators[uid]
                logger.info(f"   UID {uid}: {validator.stake:.1f} TAO (trust: {validator.trust:.3f})")
            return
        
        validator = orchestrator.validators[target_uid]
        logger.info(f"✅ Found validator UID {target_uid}")
        logger.info(f"   Hotkey: {validator.hotkey}")
        logger.info(f"   Stake: {validator.stake:.2f} TAO")
        logger.info(f"   Trust: {validator.trust:.3f}")
        logger.info(f"   Consensus: {validator.consensus:.3f}")
        
        # Check if validator is available
        if not orchestrator.is_validator_available(validator):
            logger.warning(f"⚠️ Validator UID {target_uid} is not available")
            if validator.cooldown_until:
                cooldown_remaining = validator.cooldown_until - time.time()
                logger.info(f"   Cooldown remaining: {cooldown_remaining:.1f}s")
            return
        
        # Pull task
        logger.info(f"📡 Pulling task from UID {target_uid}...")
        task = await orchestrator.pull_task_from_validator(validator)
        
        if task:
            logger.info("🎉 Task pull successful!")
            logger.info("="*50)
            logger.info(f"Task ID: {task.task_id}")
            logger.info(f"Prompt: '{task.prompt}'")
            logger.info(f"Validation threshold: {task.validation_threshold}")
            logger.info(f"Pulled at: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(task.pulled_at))}")
        else:
            logger.warning("⚠️ No task received")
    
    except Exception as e:
        logger.error(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()

async def main():
    """Main function"""
    logger.info("🚀 Testing single validator task pull...")
    logger.info("   Target: UID 79 on testnet subnet 17")
    logger.info("   Wallet: manbeast3b/m3b")
    
    await test_single_validator()

if __name__ == "__main__":
    asyncio.run(main()) 