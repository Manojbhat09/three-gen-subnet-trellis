#!/usr/bin/env python3
"""
Simple Test Script for Validation Simulation Server
Purpose: Verify the simulation server works correctly before running stress tests
"""

import asyncio
import aiohttp
import json
import time
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def test_simulation_server():
    """Test the simulation server endpoints"""
    server_url = "http://localhost:8094"
    
    async with aiohttp.ClientSession() as session:
        logger.info("🧪 Testing Validation Simulation Server...")
        
        # Test 1: Health check
        logger.info("1️⃣ Testing health check...")
        try:
            async with session.get(f"{server_url}/health") as response:
                if response.status == 200:
                    health_data = await response.json()
                    logger.info(f"✅ Health check passed: {health_data}")
                else:
                    logger.error(f"❌ Health check failed: HTTP {response.status}")
                    return False
        except Exception as e:
            logger.error(f"❌ Health check exception: {e}")
            return False
        
        # Test 2: Get statistics
        logger.info("2️⃣ Testing statistics endpoint...")
        try:
            async with session.get(f"{server_url}/stats") as response:
                if response.status == 200:
                    stats_data = await response.json()
                    logger.info(f"✅ Statistics retrieved: {json.dumps(stats_data, indent=2)}")
                else:
                    logger.error(f"❌ Statistics failed: HTTP {response.status}")
                    return False
        except Exception as e:
            logger.error(f"❌ Statistics exception: {e}")
            return False
        
        # Test 3: Pull task
        logger.info("3️⃣ Testing task pull...")
        try:
            payload = {"hotkey": "test_miner_001"}
            async with session.post(f"{server_url}/pull_task", json=payload) as response:
                if response.status == 200:
                    pull_data = await response.json()
                    logger.info(f"✅ Task pull successful: {json.dumps(pull_data, indent=2)}")
                    
                    # Check if we got a task
                    if pull_data.get('task'):
                        task = pull_data['task']
                        logger.info(f"   📋 Task ID: {task['id']}")
                        logger.info(f"   📝 Prompt: {task['prompt']}")
                        logger.info(f"   🚦 Traffic Type: {task.get('traffic_type', 'unknown')}")
                        logger.info(f"   ⏰ Expected Cooldown: {task.get('expected_cooldown', 'unknown')}s")
                        
                        # Test 4: Submit results
                        logger.info("4️⃣ Testing results submission...")
                        submit_payload = {
                            "hotkey": "test_miner_001",
                            "task_id": task['id'],
                            "prompt": task['prompt'],
                            "results": "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPhfDwAChwGA60e6kgAAAABJRU5ErkJggg==",
                            "submit_time": int(time.time_ns()),
                            "signature": "test_signature_1234"
                        }
                        
                        async with session.post(f"{server_url}/submit_results", json=submit_payload) as submit_response:
                            if submit_response.status == 200:
                                submit_data = await submit_response.json()
                                logger.info(f"✅ Results submission successful: {json.dumps(submit_data, indent=2)}")
                                
                                # Check feedback
                                feedback = submit_data.get('feedback', {})
                                if feedback:
                                    logger.info(f"   📊 Score: {feedback.get('task_fidelity_score', 'N/A')}")
                                    logger.info(f"   🎯 Validation Failed: {feedback.get('validation_failed', 'N/A')}")
                                    logger.info(f"   ⏰ Cooldown Until: {submit_data.get('cooldown_until', 'N/A')}")
                            else:
                                logger.error(f"❌ Results submission failed: HTTP {submit_response.status}")
                                return False
                    else:
                        logger.info("   ⏳ No task assigned (probably on cooldown)")
                        
                else:
                    logger.error(f"❌ Task pull failed: HTTP {response.status}")
                    return False
        except Exception as e:
            logger.error(f"❌ Task pull exception: {e}")
            return False
        
        # Test 5: Test cooldown violation
        logger.info("5️⃣ Testing cooldown violation...")
        try:
            # Try to pull again immediately (should trigger cooldown violation)
            payload = {"hotkey": "test_miner_001"}
            async with session.post(f"{server_url}/pull_task", json=payload) as response:
                if response.status == 200:
                    violation_data = await response.json()
                    logger.info(f"✅ Cooldown violation test: {json.dumps(violation_data, indent=2)}")
                    
                    if violation_data.get('cooldown_until'):
                        cooldown_until = violation_data['cooldown_until']
                        remaining = max(0, cooldown_until - time.time())
                        logger.info(f"   ⏰ Cooldown remaining: {remaining:.1f}s")
                    
                    if violation_data.get('cooldown_violations'):
                        violations = violation_data['cooldown_violations']
                        logger.info(f"   🚨 Cooldown violations: {violations}")
                else:
                    logger.error(f"❌ Cooldown violation test failed: HTTP {response.status}")
                    return False
        except Exception as e:
            logger.error(f"❌ Cooldown violation test exception: {e}")
            return False
        
        # Test 6: Test validation endpoint
        logger.info("6️⃣ Testing validation endpoint...")
        try:
            validation_payload = {
                "prompt": "test modern chair for validation",
                "data": "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPhfDwAChwGA60e6kgAAAABJRU5ErkJggg==",
                "compression": 2,
                "generate_preview": True,
                "preview_score_threshold": 0.5
            }
            
            async with session.post(f"{server_url}/validate_txt_to_3d_ply", json=validation_payload) as response:
                if response.status == 200:
                    validation_data = await response.json()
                    logger.info(f"✅ Validation successful: {json.dumps(validation_data, indent=2)}")
                    
                    # Check validation scores
                    if 'score' in validation_data:
                        logger.info(f"   📊 Validation Score: {validation_data['score']}")
                    if 'iqa' in validation_data:
                        logger.info(f"   🎨 IQA Score: {validation_data['iqa']}")
                    if 'preview' in validation_data:
                        logger.info(f"   🖼️ Preview Generated: {'Yes' if validation_data['preview'] else 'No'}")
                else:
                    logger.error(f"❌ Validation failed: HTTP {response.status}")
                    return False
        except Exception as e:
            logger.error(f"❌ Validation test exception: {e}")
            return False
        
        # Test 7: Test rate limiting
        logger.info("7️⃣ Testing rate limiting...")
        try:
            # Send many requests quickly to trigger rate limiting
            tasks = []
            for i in range(50):  # Send 50 requests
                payload = {"hotkey": f"rate_test_miner_{i:03d}"}
                task = session.post(f"{server_url}/pull_task", json=payload)
                tasks.append(task)
            
            # Execute all requests concurrently
            responses = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Count rate limited responses
            rate_limited = 0
            successful = 0
            errors = 0
            
            for response in responses:
                if isinstance(response, Exception):
                    errors += 1
                elif hasattr(response, 'status'):
                    if response.status == 429:  # Rate limited
                        rate_limited += 1
                    elif response.status == 200:
                        successful += 1
                    else:
                        errors += 1
            
            logger.info(f"✅ Rate limit test completed:")
            logger.info(f"   📊 Successful: {successful}")
            logger.info(f"   🚫 Rate Limited: {rate_limited}")
            logger.info(f"   ❌ Errors: {errors}")
            
        except Exception as e:
            logger.error(f"❌ Rate limit test exception: {e}")
            return False
        
        logger.info("�� All tests completed successfully!")
        return True

async def main():
    """Main entry point"""
    try:
        success = await test_simulation_server()
        if success:
            logger.info("✅ All tests passed! Simulation server is working correctly.")
        else:
            logger.error("❌ Some tests failed. Check the logs above.")
            
    except Exception as e:
        logger.error(f"❌ Test suite failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(main())





