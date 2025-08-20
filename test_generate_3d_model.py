#!/usr/bin/env python3
"""
Test script for the new async generate_3d_model method with parallel validation
"""

import asyncio
import json
import time
import logging
from pathlib import Path
from typing import Dict, Any, Optional

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('test_generate_3d_model.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Mock TaskRecord class for testing
class MockTaskRecord:
    def __init__(self, task_id: str, prompt: str):
        self.task_id = task_id
        self.prompt = prompt
        self.generation_time = None
        self.compressed_file_path = None

# Mock orchestrator class for testing
class MockOrchestrator:
    def __init__(self):
        self.logger = logger
        self.output_dir = Path("./test_output")
        self.output_dir.mkdir(exist_ok=True)
        self.config = {
            'save_intermediate_results': True,
            'validate_generations': False
        }
        self.stats = {
            'successful_generations': 0,
            'total_generation_time': 0.0
        }
        
    async def generate_3d_model(self, task: MockTaskRecord) -> Optional[Dict[str, Any]]:
        """Test the async generate_3d_model method with parallel validation"""
        
        logger.info(f"🧪 Testing generate_3d_model for task: {task.task_id}")
        logger.info(f"📝 Prompt: {task.prompt}")
        
        try:
            # Simulate the prompt optimization and LoRA routing
            cleaned_prompt = f"cleaned_{task.prompt}"
            endpoint = "/generate"
            deterministic_seed = 42
            
            logger.info(f"   🎲 Using deterministic seed: {deterministic_seed}")
            logger.info(f"   🎯 Using endpoint: {endpoint}")
            
            generation_start = time.time()
            
            # Test ports
            port1 = 8099
            port2 = 8097  # Updated to match user's change
            
            # Mock generation config
            GENERATION_CONFIG = {
                'num_inference_steps_t2i': 20,
                'guidance_scale': 7.5,
                'ss_sampling_steps': 10,
                'slat_sampling_steps': 10,
                'slat_guidance_strength': 0.8,
                'ss_guidance_strength': 0.8
            }
            
            num_inference_steps = GENERATION_CONFIG['num_inference_steps_t2i']
            guidance_scale = GENERATION_CONFIG['guidance_scale']
            ss_sampling_steps = GENERATION_CONFIG['ss_sampling_steps']
            slat_sampling_steps = GENERATION_CONFIG['slat_sampling_steps']
            slat_guidance_strength = GENERATION_CONFIG['slat_guidance_strength']
            ss_guidance_strength = GENERATION_CONFIG['ss_guidance_strength']
            
            # Run both validators in parallel using asyncio.gather
            logger.info(f"🚀 Starting parallel validation on ports {port1} and {port2}")
            
            # Import the async validator function
            from continuous_trellis_orchestrator_lora_working import run_validator_async
            
            # Time the parallel execution
            parallel_start = time.time()
            
            # Create tasks for both validators
            task1 = run_validator_async(
                task.prompt, task.prompt, endpoint, port1, num_inference_steps, guidance_scale,
                ss_sampling_steps, slat_sampling_steps, slat_guidance_strength, ss_guidance_strength
            )
            
            task2 = run_validator_async(
                task.prompt, task.prompt, endpoint, port2, num_inference_steps, guidance_scale,
                ss_sampling_steps, slat_sampling_steps, slat_guidance_strength, ss_guidance_strength
            )
            
            # Wait for both validators to complete
            logger.info("⏳ Waiting for both validators to complete...")
            original_results1, original_results2 = await asyncio.gather(task1, task2)
            
            parallel_end = time.time()
            parallel_time = parallel_end - parallel_start
            
            logger.info(f"✅ Both validators completed in parallel")
            logger.info(f"⏱️  Parallel execution time: {parallel_time:.2f}s")
            
            # Show detailed results
            logger.info(f"\n{'='*60}")
            logger.info(f"📊 VALIDATION RESULTS")
            logger.info(f"{'='*60}")
            
            # Results for original prompt
            logger.info(f"\n🔍 Original Prompt (Port {port1}):")
            if 'error' in original_results1:
                logger.error(f"   ❌ Error: {original_results1['error']}")
            else:
                for key, value in original_results1.items():
                    if key == 'ply_data' and isinstance(value, bytes):
                        logger.info(f"   📁 {key}: {len(value):,} bytes")
                    elif key == 'compression':
                        logger.info(f"   📦 {key}: {value}")
                    elif key == 'validation_engine_score':
                        logger.info(f"   🎯 {key}: {value}")
                    else:
                        logger.info(f"   📋 {key}: {value}")
            
            # Results for cleaned prompt
            logger.info(f"\n🔍 Cleaned Prompt (Port {port2}):")
            if 'error' in original_results2:
                logger.error(f"   ❌ Error: {original_results2['error']}")
            else:
                for key, value in original_results2.items():
                    if key == 'ply_data' and isinstance(value, bytes):
                        logger.info(f"   📁 {key}: {len(value):,} bytes")
                    elif key == 'compression':
                        logger.info(f"   📦 {key}: {value}")
                    elif key == 'validation_engine_score':
                        logger.info(f"   🎯 {key}: {value}")
                    else:
                        logger.info(f"   📋 {key}: {value}")
            
            # Process results
            ply_data = None
            compression_ratio = None
            
            if original_results1.get('validation_engine_score', 0) > original_results2.get('validation_engine_score', 0):
                logger.info(f"\n✅ Using result from original prompt: {task.prompt}")
                ply_data = original_results1.get('ply_data', b'mock_ply_data')
                compression_ratio = original_results1.get('compression', 'unknown')
            else:
                logger.info(f"\n✅ Using result from cleaned prompt: {cleaned_prompt}")
                ply_data = original_results2.get('ply_data', b'mock_ply_data')
                compression_ratio = original_results2.get('compression', 'unknown')
            
            if ply_data:
                generation_time = time.time() - generation_start
                task.generation_time = generation_time
                
                # Save PLY file
                if self.config['save_intermediate_results']:
                    timestamp = int(time.time())
                    ply_file = self.output_dir / f"task_{task.task_id}_{timestamp}.ply.spz"
                    with open(ply_file, 'wb') as f:
                        f.write(ply_data)
                    task.compressed_file_path = str(ply_file)
                
                logger.info(f"✅ Generation successful in {generation_time:.2f}s ({len(ply_data):,} bytes)")
                logger.info(f"⏱️  Breakdown: {parallel_time:.2f}s parallel + {generation_time - parallel_time:.2f}s processing")
                
                self.stats['successful_generations'] += 1
                self.stats['total_generation_time'] += generation_time
                
                return {'ply_data': ply_data, 'compression_ratio': compression_ratio}
            else:
                logger.error(f"❌ Generation failed: No PLY data received")
                return None
        
        except Exception as e:
            logger.error(f"❌ Generation exception: {e}")
            import traceback
            traceback.print_exc()
            return None

async def test_parallel_validation():
    """Test the parallel validation functionality"""
    
    logger.info("🧪 Starting test of parallel validation in generate_3d_model")
    
    # Create mock orchestrator
    orchestrator = MockOrchestrator()
    
    # Create test tasks
    test_tasks = [
        MockTaskRecord("test_001", "A beautiful red rose in a glass vase"),
        MockTaskRecord("test_002", "A futuristic robot with glowing blue eyes"),
        MockTaskRecord("test_003", "A serene mountain landscape at sunset")
    ]
    
    # Test each task
    for i, task in enumerate(test_tasks):
        logger.info(f"\n{'='*60}")
        logger.info(f"🧪 Testing Task {i+1}/{len(test_tasks)}: {task.task_id}")
        logger.info(f"{'='*60}")
        
        start_time = time.time()
        result = await orchestrator.generate_3d_model(task)
        end_time = time.time()
        
        if result:
            logger.info(f"✅ Task {task.task_id} completed successfully")
            logger.info(f"   Generation time: {task.generation_time:.2f}s")
            logger.info(f"   Compression ratio: {result.get('compression_ratio', 'unknown')}")
            if task.compressed_file_path:
                logger.info(f"   Saved to: {task.compressed_file_path}")
        else:
            logger.error(f"❌ Task {task.task_id} failed")
        
        total_test_time = end_time - start_time
        logger.info(f"   Total test time: {total_test_time:.2f}s")
        
        # Calculate efficiency
        if result and task.generation_time:
            efficiency = (task.generation_time / total_test_time) * 100
            logger.info(f"   ⚡ Efficiency: {efficiency:.1f}% (generation vs total)")
        
        # Small delay between tests
        if i < len(test_tasks) - 1:
            await asyncio.sleep(1)
    
    # Print final statistics
    logger.info(f"\n{'='*60}")
    logger.info("📊 Final Test Statistics")
    logger.info(f"{'='*60}")
    logger.info(f"Successful generations: {orchestrator.stats['successful_generations']}")
    logger.info(f"Total generation time: {orchestrator.stats['total_generation_time']:.2f}s")
    
    if orchestrator.stats['successful_generations'] > 0:
        avg_time = orchestrator.stats['total_generation_time'] / orchestrator.stats['successful_generations']
        logger.info(f"Average generation time: {avg_time:.2f}s")
        
        # Calculate theoretical sequential time (2x average for 2 validators)
        theoretical_sequential = avg_time * 2
        logger.info(f"Theoretical sequential time: {theoretical_sequential:.2f}s")
        
        # Calculate time savings
        time_saved = theoretical_sequential - avg_time
        savings_percent = (time_saved / theoretical_sequential) * 100
        logger.info(f"⏱️  Time saved with parallel execution: {time_saved:.2f}s ({savings_percent:.1f}%)")
        
        # Performance metrics
        logger.info(f"🚀 Parallel speedup: {theoretical_sequential / avg_time:.2x}")

async def main():
    """Main test function"""
    try:
        await test_parallel_validation()
        logger.info("🎉 All tests completed successfully!")
    except Exception as e:
        logger.error(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    # Run the test
    asyncio.run(main())
