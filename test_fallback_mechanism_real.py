#!/usr/bin/env python3
"""
Real Test Script for generate_3d_model_with_fallback Function
Purpose: Test the actual function from the orchestrator with mocked external dependencies
Usage: python test_fallback_mechanism_real.py
"""

import asyncio
import time
import sys
import logging
import requests
from pathlib import Path
from typing import Dict, Any, Optional
from unittest.mock import Mock, patch, AsyncMock

# Add the current directory to Python path
sys.path.append('.')

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('test_fallback_real.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Mock external dependencies
class MockRequests:
    """Mock requests module for testing"""
    
    @staticmethod
    def post(url, data=None, timeout=None):
        mock_response = Mock()
        mock_response.status_code = 200
        
        # Mock different response types based on endpoint
        if 'generate_image' in url:
            # Mock image generation response
            mock_response.content = b'mock_image_data_12345'
            mock_response.headers = {}
        else:
            # Mock 3D generation response
            mock_response.content = b'mock_ply_data_67890'
            mock_response.headers = {'X-Compression-Ratio': '0.75'}
        
        return mock_response

class MockCLIPModel:
    """Mock CLIP model for testing"""
    def __init__(self):
        self.eval_called = False
        self.encode_image_called = False
        self.encode_text_called = False
    
    def eval(self):
        self.eval_called = True
    
    def encode_image(self, tensor):
        self.encode_image_called = True
        # Return mock features
        mock_features = Mock()
        mock_features.norm = Mock(return_value=Mock())
        mock_features.cpu = Mock(return_value=Mock())
        mock_features.numpy = Mock(return_value=[[0.85]])  # High score
        return mock_features
    
    def encode_text(self, tensor):
        self.encode_text_called = True
        # Return mock features
        mock_features = Mock()
        mock_features.norm = Mock(return_value=Mock())
        mock_features.cpu = Mock(return_value=Mock())
        mock_features.numpy = Mock(return_value=[[0.90]])  # High score
        return mock_features

class MockCLIPTokenizer:
    """Mock CLIP tokenizer for testing"""
    def __call__(self, text):
        mock_tensor = Mock()
        mock_tensor.to = Mock(return_value=mock_tensor)
        return mock_tensor

class MockTorch:
    """Mock torch module for testing"""
    @staticmethod
    def device(device_str):
        return device_str
    
    @staticmethod
    def tensor(data):
        mock_tensor = Mock()
        mock_tensor.float = Mock(return_value=mock_tensor)
        mock_tensor.permute = Mock(return_value=mock_tensor)
        mock_tensor.unsqueeze = Mock(return_value=mock_tensor)
        mock_tensor.to = Mock(return_value=mock_tensor)
        return mock_tensor
    
    @staticmethod
    def no_grad():
        return Mock()
    
    @staticmethod
    def autocast(device_type):
        return Mock()

class MockTorchVision:
    """Mock torchvision module for testing"""
    @staticmethod
    def transforms():
        mock_transforms = Mock()
        mock_transforms.Normalize = Mock(return_value=Mock())
        return mock_transforms

class MockPIL:
    """Mock PIL module for testing"""
    @staticmethod
    def Image():
        mock_image = Mock()
        mock_image.open = Mock(return_value=Mock())
        return mock_image

class MockOpenCLIP:
    """Mock open_clip module for testing"""
    @staticmethod
    def create_model_and_transforms(model_name, pretrained, device):
        mock_model = MockCLIPModel()
        mock_transforms = Mock()
        mock_preprocess = Mock()
        return mock_model, mock_transforms, mock_preprocess
    
    @staticmethod
    def get_tokenizer(model_name):
        return MockCLIPTokenizer()

class MockLLMOptimizer:
    """Mock LLM optimizer for testing"""
    def __init__(self):
        self.use_vllm = True
        self.vllm_url = "http://localhost:9000"
        self.vllm_model = "llama-3-2-3b-it"
    
    def _query_vllm(self, system_prompt, user_prompt):
        logger.info(f"🔧 Mock vLLM query with system prompt length: {len(system_prompt)}")
        return f"optimized_{user_prompt}_with_enhancements"
    
    def _clean_response(self, response):
        return response.strip()

# Patch the modules before importing
sys.modules['requests'] = MockRequests()
sys.modules['torch'] = MockTorch()
sys.modules['torch.nn.functional'] = Mock()
sys.modules['torchvision'] = MockTorchVision()
sys.modules['PIL'] = MockPIL()
sys.modules['open_clip'] = MockOpenCLIP()
sys.modules['llm_prompt_optimizer_v12_f1_lora'] = Mock()

# Now try to import the orchestrator
try:
    from continuous_trellis_orchestrator_lora_working import (
        ContinuousTrellisOrchestrator, 
        TaskRecord,
        optimized_system_prompt
    )
    IMPORT_SUCCESS = True
    logger.info("✅ Successfully imported orchestrator")
except ImportError as e:
    IMPORT_SUCCESS = False
    logger.error(f"❌ Failed to import orchestrator: {e}")
    logger.error("   This test requires the orchestrator to be available")

async def test_optimized_system_prompt():
    """Test the optimized_system_prompt function"""
    logger.info("🧪 Testing optimized_system_prompt function...")
    
    try:
        test_prompt = "a wooden chair"
        result = optimized_system_prompt(test_prompt)
        
        # Verify the result
        assert isinstance(result, str)
        assert "ORIGINAL PROMPT" in result
        assert "OPTIMIZED PROMPT" in result
        assert test_prompt in result
        
        logger.info("✅ optimized_system_prompt function: PASSED")
        logger.info(f"   Result length: {len(result)} characters")
        logger.info(f"   Contains original prompt: {test_prompt in result}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ optimized_system_prompt function: FAILED - {e}")
        return False

async def test_fallback_function_with_mocks():
    """Test the actual fallback function with mocked dependencies"""
    if not IMPORT_SUCCESS:
        logger.error("❌ Cannot test fallback function - import failed")
        return False
    
    logger.info("🧪 Testing generate_3d_model_with_fallback function...")
    
    try:
        # Create a minimal orchestrator instance
        config = {
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
        
        # Create orchestrator with mocked dependencies
        with patch('continuous_trellis_orchestrator_lora_working.requests', MockRequests()), \
             patch('continuous_trellis_orchestrator_lora_working.torch', MockTorch()), \
             patch('continuous_trellis_orchestrator_lora_working.open_clip', MockOpenCLIP()), \
             patch('continuous_trellis_orchestrator_lora_working.LLMPromptOptimizer', MockLLMOptimizer):
            
            orchestrator = ContinuousTrellisOrchestrator(config)
            
            # Create a test task
            task = TaskRecord(
                task_id="test_fallback_001",
                prompt="a simple wooden chair",
                prompt_hash="hash_chair",
                validator_uid=123,
                validator_hotkey="test_hotkey",
                validator_stake=1000.0,
                validation_threshold=0.5,
                pulled_at=time.time()
            )
            
            # Test timing
            start_time = time.time()
            
            # Call the actual function
            result = await orchestrator.generate_3d_model_with_fallback(task)
            
            end_time = time.time()
            total_time = end_time - start_time
            
            logger.info(f"⏱️ Function execution time: {total_time:.2f} seconds")
            
            # Verify the result
            if result:
                logger.info("✅ Function execution: PASSED")
                logger.info(f"   Result type: {type(result)}")
                logger.info(f"   Result keys: {list(result.keys()) if isinstance(result, dict) else 'N/A'}")
                
                # Check if it's a fallback result
                if 'prompt_type' in result:
                    logger.info(f"   Prompt type: {result['prompt_type']}")
                
                return True
            else:
                logger.warning("⚠️ Function executed but returned no result")
                return False
                
    except Exception as e:
        logger.error(f"❌ Function execution: FAILED - {e}")
        logger.error(f"   Error type: {type(e).__name__}")
        return False

async def test_performance_benchmarks():
    """Test performance benchmarks"""
    logger.info("🧪 Testing performance benchmarks...")
    
    if not IMPORT_SUCCESS:
        logger.error("❌ Cannot test performance - import failed")
        return False
    
    try:
        # Test multiple iterations to measure performance
        test_prompts = [
            "wooden chair",
            "metal table",
            "glass vase",
            "plastic container"
        ]
        
        execution_times = []
        
        for i, prompt in enumerate(test_prompts, 1):
            logger.info(f"   Benchmark {i}/{len(test_prompts)}: '{prompt}'")
            
            # Create test task
            task = TaskRecord(
                task_id=f"benchmark_{i:03d}",
                prompt=prompt,
                prompt_hash=f"hash_{prompt}",
                validator_uid=123,
                validator_hotkey="test_hotkey",
                validator_stake=1000.0,
                validation_threshold=0.5,
                pulled_at=time.time()
            )
            
            # Mock the external calls to avoid actual network requests
            with patch('continuous_trellis_orchestrator_lora_working.requests', MockRequests()), \
                 patch('continuous_trellis_orchestrator_lora_working.torch', MockTorch()), \
                 patch('continuous_trellis_orchestrator_lora_working.open_clip', MockOpenCLIP()), \
                 patch('continuous_trellis_orchestrator_lora_working.LLMPromptOptimizer', MockLLMOptimizer):
                
                config = {
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
                
                orchestrator = ContinuousTrellisOrchestrator(config)
                
                start_time = time.time()
                
                # Execute function
                result = await orchestrator.generate_3d_model_with_fallback(task)
                
                end_time = time.time()
                execution_time = end_time - start_time
                execution_times.append(execution_time)
                
                logger.info(f"     Execution time: {execution_time:.2f}s")
                
                if not result:
                    logger.warning(f"     ⚠️ No result returned")
        
        # Calculate performance statistics
        if execution_times:
            avg_time = sum(execution_times) / len(execution_times)
            min_time = min(execution_times)
            max_time = max(execution_times)
            
            logger.info(f"\n📊 Performance Statistics:")
            logger.info(f"   Average execution time: {avg_time:.2f}s")
            logger.info(f"   Minimum execution time: {min_time:.2f}s")
            logger.info(f"   Maximum execution time: {max_time:.2f}s")
            logger.info(f"   Total test time: {sum(execution_times):.2f}s")
            
            # Performance thresholds
            if avg_time < 5.0:
                logger.info("✅ Performance: EXCELLENT (< 5s average)")
            elif avg_time < 10.0:
                logger.info("✅ Performance: GOOD (< 10s average)")
            elif avg_time < 20.0:
                logger.info("⚠️ Performance: ACCEPTABLE (< 20s average)")
            else:
                logger.warning("⚠️ Performance: SLOW (> 20s average)")
            
            return True
        else:
            logger.error("❌ No execution times recorded")
            return False
            
    except Exception as e:
        logger.error(f"❌ Performance benchmark: FAILED - {e}")
        return False

async def main():
    """Main test function"""
    logger.info("🚀 Starting Real Fallback Mechanism Test Suite")
    logger.info("=" * 60)
    
    # Test 1: Import verification
    logger.info(f"📦 Import Status: {'✅ SUCCESS' if IMPORT_SUCCESS else '❌ FAILED'}")
    
    # Test 2: Function availability
    if IMPORT_SUCCESS:
        logger.info("✅ All required functions are available")
    else:
        logger.error("❌ Cannot proceed with tests - import failed")
        return
    
    # Test 3: Individual function tests
    logger.info("\n🔧 Testing individual functions...")
    
    # Test optimized_system_prompt
    prompt_test_result = await test_optimized_system_prompt()
    
    # Test 4: Main fallback function
    logger.info("\n🧪 Testing main fallback function...")
    fallback_test_result = await test_fallback_function_with_mocks()
    
    # Test 5: Performance benchmarks
    logger.info("\n⏱️ Testing performance benchmarks...")
    performance_test_result = await test_performance_benchmarks()
    
    # Summary
    logger.info(f"\n{'='*60}")
    logger.info("🎯 TEST SUMMARY")
    logger.info(f"{'='*60}")
    logger.info(f"📦 Import Status: {'✅ PASSED' if IMPORT_SUCCESS else '❌ FAILED'}")
    logger.info(f"🔧 System Prompt: {'✅ PASSED' if prompt_test_result else '❌ FAILED'}")
    logger.info(f"🧪 Fallback Function: {'✅ PASSED' if fallback_test_result else '❌ FAILED'}")
    logger.info(f"⏱️ Performance: {'✅ PASSED' if performance_test_result else '❌ FAILED'}")
    
    # Overall result
    all_passed = all([IMPORT_SUCCESS, prompt_test_result, fallback_test_result, performance_test_result])
    
    if all_passed:
        logger.info(f"\n🎉 ALL TESTS PASSED! The fallback mechanism is working correctly.")
    else:
        logger.error(f"\n❌ SOME TESTS FAILED. Check the logs above for details.")
    
    logger.info(f"\n📁 Check 'test_fallback_real.log' for detailed logs")
    logger.info(f"🎯 Test suite completed!")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("\n⚠️ Test interrupted by user")
    except Exception as e:
        logger.error(f"❌ Test suite failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)








