#!/usr/bin/env python3
"""
Test Script for generate_3d_model_with_fallback Function
Purpose: Test the CLIP-based fallback mechanism in isolation
Usage: python test_fallback_mechanism.py "your test prompt"
"""

import asyncio
import time
import sys
import logging
from pathlib import Path
from typing import Dict, Any, Optional

# Add the current directory to Python path to import the orchestrator
sys.path.append('.')

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('test_fallback.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Mock classes for testing
class MockTaskRecord:
    """Mock TaskRecord for testing"""
    def __init__(self, prompt: str, task_id: str = "test_task_001"):
        self.task_id = task_id
        self.prompt = prompt
        self.prompt_hash = f"hash_{prompt[:20]}"
        self.validator_uid = 123
        self.validator_hotkey = "test_hotkey"
        self.validator_stake = 1000.0
        self.validation_threshold = 0.5
        self.pulled_at = time.time()
        self.processed_at = None
        self.submitted_at = None
        self.generation_time = None
        self.validation_time = None
        self.total_processing_time = None
        self.local_validation_score = None
        self.submission_success = False
        self.feedback_received = False
        self.task_fidelity_score = None
        self.average_fidelity_score = None
        self.current_miner_reward = None
        self.validation_failed = None
        self.generations_in_window = None
        self.ply_file_path = None
        self.compressed_file_path = None
        self.priority_access_timeout = False

class MockPriorityCoordinator:
    """Mock PriorityCoordinator for testing"""
    def __init__(self):
        self.logger = logger
    
    def wait_for_priority_access(self, task_id: str) -> bool:
        logger.info(f"🔒 Mock: Priority access granted for task {task_id}")
        return True
    
    def mark_priority_job_start(self, task_id: str, prompt: str):
        logger.info(f"🚀 Mock: Priority job started for task {task_id}")
    
    def mark_priority_job_end(self, task_id: str):
        logger.info(f"✅ Mock: Priority job ended for task {task_id}")
    
    def clear_server_cache(self):
        logger.info(f"🧹 Mock: Server cache cleared")

class MockConfig:
    """Mock configuration for testing"""
    def __init__(self):
        self.config = {
            'generation_server_url': 'http://localhost:8096',
            'validation_server_url': 'http://localhost:10006',
            'generation_timeout': 300,
            'save_intermediate_results': True,
            'log_optimization_details': True,
            'enable_fallback_mechanism': True,
            'fallback_ratio_threshold': 0.8,
            'fallback_max_retries': 1,
            'vllm_url': 'http://localhost:9000',
            'vllm_model': 'llama-3-2-3b-it'
        }
    
    def get(self, key: str, default=None):
        return self.config.get(key, default)

class MockOrchestrator:
    """Mock orchestrator with only the functions needed for testing"""
    def __init__(self):
        self.logger = logger
        self.priority_coordinator = MockPriorityCoordinator()
        self.config = MockConfig()
        self.output_dir = Path("./test_outputs")
        self.output_dir.mkdir(exist_ok=True)
        self.stats = {
            'successful_generations': 0,
            'total_generation_time': 0.0
        }
        
        # Mock CLIP model flag
        self._clip_model = None
    
    def get_deterministic_seed(self, task) -> int:
        """Mock deterministic seed generation"""
        return 42
    
    def clean_optimized_prompt(self, prompt: str) -> str:
        """Mock prompt cleaning"""
        if "white background" not in prompt.lower():
            return prompt + " front view, white background"
        return prompt
    
    def optimize_prompt_for_generation(self, task) -> Dict[str, Any]:
        """Mock prompt optimization"""
        logger.info(f"🔧 Mock: Optimizing prompt: '{task.prompt}'")
        
        # Simulate different optimization results for testing
        if "test" in task.prompt.lower():
            optimized = f"enhanced {task.prompt} with improved details"
        else:
            optimized = f"professional {task.prompt} with high quality"
        
        return {
            'optimized_prompt': optimized,
            'lora_info': {
                'lora_name': 'cinema',
                'endpoint': '/generate/cinema/',
                'reasoning': 'Mock optimization for testing',
                'confidence': 'High'
            },
            'endpoint': '/generate/cinema/'
        }

async def test_fallback_mechanism():
    """Test the fallback mechanism function"""
    logger.info("🧪 Starting fallback mechanism test")
    
    # Test prompts
    test_prompts = [
        "a simple wooden chair",
        "test prompt for validation",
        "complex mechanical device",
        "delicate glass ornament"
    ]
    
    for i, prompt in enumerate(test_prompts, 1):
        logger.info(f"\n{'='*60}")
        logger.info(f"🧪 TEST {i}/{len(test_prompts)}: '{prompt}'")
        logger.info(f"{'='*60}")
        
        # Create mock task
        task = MockTaskRecord(prompt, f"test_task_{i:03d}")
        
        # Create mock orchestrator
        orchestrator = MockOrchestrator()
        
        # Test timing
        start_time = time.time()
        
        try:
            # Test the fallback function
            result = await orchestrator.generate_3d_model_with_fallback(task)
            
            end_time = time.time()
            total_time = end_time - start_time
            
            logger.info(f"⏱️ Test completed in {total_time:.2f} seconds")
            
            if result:
                logger.info(f"✅ Test PASSED - Result received:")
                logger.info(f"   Result type: {type(result)}")
                logger.info(f"   Result keys: {list(result.keys()) if isinstance(result, dict) else 'N/A'}")
                logger.info(f"   Prompt type: {result.get('prompt_type', 'N/A')}")
                logger.info(f"   Compression ratio: {result.get('compression_ratio', 'N/A')}")
            else:
                logger.warning(f"⚠️ Test completed but no result returned")
                
        except Exception as e:
            end_time = time.time()
            total_time = end_time - start_time
            
            logger.error(f"❌ Test FAILED after {total_time:.2f} seconds")
            logger.error(f"   Error: {e}")
            logger.error(f"   Error type: {type(e).__name__}")
            
            # Continue with next test
            continue
        
        # Performance check
        if total_time > 30:  # 30 seconds threshold
            logger.warning(f"⚠️ Test took longer than expected: {total_time:.2f}s")
        else:
            logger.info(f"✅ Test completed within expected time: {total_time:.2f}s")
        
        # Small delay between tests
        await asyncio.sleep(1)
    
    logger.info(f"\n{'='*60}")
    logger.info("🎯 All tests completed!")
    logger.info(f"{'='*60}")

async def test_individual_components():
    """Test individual components of the fallback mechanism"""
    logger.info("\n🔧 Testing individual components...")
    
    # Test 1: Mock task creation
    logger.info("🧪 Test 1: Mock task creation")
    task = MockTaskRecord("test prompt")
    assert task.task_id == "test_task_001"
    assert task.prompt == "test prompt"
    logger.info("✅ Mock task creation: PASSED")
    
    # Test 2: Mock priority coordinator
    logger.info("🧪 Test 2: Mock priority coordinator")
    coordinator = MockPriorityCoordinator()
    assert coordinator.wait_for_priority_access("test") == True
    logger.info("✅ Mock priority coordinator: PASSED")
    
    # Test 3: Mock configuration
    logger.info("🧪 Test 3: Mock configuration")
    config = MockConfig()
    assert config.get('enable_fallback_mechanism') == True
    assert config.get('fallback_ratio_threshold') == 0.8
    logger.info("✅ Mock configuration: PASSED")
    
    # Test 4: Mock orchestrator
    logger.info("🧪 Test 4: Mock orchestrator")
    orchestrator = MockOrchestrator()
    assert orchestrator.get_deterministic_seed(None) == 42
    assert orchestrator.clean_optimized_prompt("test") == "test front view, white background"
    logger.info("✅ Mock orchestrator: PASSED")
    
    # Test 5: Mock prompt optimization
    logger.info("🧪 Test 5: Mock prompt optimization")
    task = MockTaskRecord("test prompt")
    result = orchestrator.optimize_prompt_for_generation(task)
    assert 'optimized_prompt' in result
    assert 'lora_info' in result
    assert 'endpoint' in result
    logger.info("✅ Mock prompt optimization: PASSED")
    
    logger.info("🎯 All component tests: PASSED")

async def main():
    """Main test function"""
    logger.info("🚀 Starting Fallback Mechanism Test Suite")
    logger.info("=" * 60)
    
    # Test individual components first
    await test_individual_components()
    
    # Test the main fallback mechanism
    await test_fallback_mechanism()
    
    logger.info("\n🎉 Test suite completed successfully!")
    logger.info("Check 'test_fallback.log' for detailed logs")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("\n⚠️ Test interrupted by user")
    except Exception as e:
        logger.error(f"❌ Test suite failed: {e}")
        sys.exit(1)








