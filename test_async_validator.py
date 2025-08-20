#!/usr/bin/env python3
"""
Simple test script for the async validator function
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

async def test_async_validator():
    """Test the async validator function directly"""
    
    logger.info("🧪 Testing async validator function")
    
    try:
        # Import the async validator function
        from continuous_trellis_orchestrator_lora_working import run_validator_async
        
        # Test parameters
        original_prompt = "A beautiful red rose in a glass vase"
        optimized_prompt = "cleaned_A beautiful red rose in a glass vase"
        endpoint = "/generate"
        port1 = 8099
        port2 = 8097  # Updated to match user's change
        
        # Generation config
        num_inference_steps = 20
        guidance_scale = 7.5
        ss_sampling_steps = 10
        slat_sampling_steps = 10
        slat_guidance_strength = 0.8
        ss_guidance_strength = 0.8
        
        logger.info(f"🚀 Starting parallel validation test")
        logger.info(f"   Port 1: {port1}")
        logger.info(f"   Port 2: {port2}")
        logger.info(f"   Original prompt: '{original_prompt}'")
        logger.info(f"   Optimized prompt: '{optimized_prompt}'")
        
        # Time the individual validator calls
        individual_start = time.time()
        
        # Create both validator tasks
        task1 = run_validator_async(
            original_prompt, original_prompt, endpoint, port1,
            num_inference_steps, guidance_scale,
            ss_sampling_steps, slat_sampling_steps,
            slat_guidance_strength, ss_guidance_strength
        )
        
        task2 = run_validator_async(
            original_prompt, optimized_prompt, endpoint, port2,
            num_inference_steps, guidance_scale,
            ss_sampling_steps, slat_sampling_steps,
            slat_guidance_strength, ss_guidance_strength
        )
        
        # Wait for both to complete
        logger.info("⏳ Waiting for both validators to complete...")
        parallel_start = time.time()
        results = await asyncio.gather(task1, task2)
        parallel_end = time.time()
        
        individual_end = time.time()
        
        # Calculate timing
        total_time = individual_end - individual_start
        parallel_time = parallel_end - parallel_start
        
        logger.info(f"\n{'='*60}")
        logger.info(f"⏱️  TIMING RESULTS")
        logger.info(f"{'='*60}")
        logger.info(f"   Total execution time: {total_time:.2f}s")
        logger.info(f"   Parallel execution time: {parallel_time:.2f}s")
        logger.info(f"   Setup overhead: {total_time - parallel_time:.2f}s")
        
        # Show detailed results
        logger.info(f"\n{'='*60}")
        logger.info(f"📊 VALIDATION RESULTS")
        logger.info(f"{'='*60}")
        
        for i, result in enumerate(results):
            port = port1 if i == 0 else port2
            prompt_type = "Original" if i == 0 else "Optimized"
            logger.info(f"\n🔍 {prompt_type} Prompt (Port {port}):")
            
            if 'error' in result:
                logger.error(f"   ❌ Error: {result['error']}")
            else:
                # Show all available fields
                for key, value in result.items():
                    if key == 'ply_data' and isinstance(value, bytes):
                        logger.info(f"   📁 {key}: {len(value):,} bytes")
                    elif key == 'compression':
                        logger.info(f"   📦 {key}: {value}")
                    elif key == 'validation_engine_score':
                        logger.info(f"   🎯 {key}: {value}")
                    else:
                        logger.info(f"   📋 {key}: {value}")
        
        # Compare results
        logger.info(f"\n{'='*60}")
        logger.info(f"🏆 RESULT COMPARISON")
        logger.info(f"{'='*60}")
        
        if 'error' not in results[0] and 'error' not in results[1]:
            score1 = results[0].get('validation_engine_score', 0)
            score2 = results[1].get('validation_engine_score', 0)
            
            if score1 > score2:
                logger.info(f"✅ Original prompt wins with score: {score1:.4f} vs {score2:.4f}")
            elif score2 > score1:
                logger.info(f"✅ Optimized prompt wins with score: {score2:.4f} vs {score1:.4f}")
            else:
                logger.info(f"🤝 Both prompts tied with score: {score1:.4f}")
        else:
            logger.warning("⚠️  Cannot compare results due to errors")
        
        return results
        
    except Exception as e:
        logger.error(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return None

async def main():
    """Main test function"""
    logger.info("🧪 Starting async validator test")
    
    try:
        results = await test_async_validator()
        if results:
            logger.info("🎉 Test completed successfully!")
        else:
            logger.error("❌ Test failed")
    except Exception as e:
        logger.error(f"❌ Test failed with error: {e}")

if __name__ == "__main__":
    asyncio.run(main())
