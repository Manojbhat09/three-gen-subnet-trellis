#!/usr/bin/env python3
"""
Demo script showing timing comparison between sequential and parallel validation
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

async def mock_validator(prompt: str, port: int, delay: float) -> dict:
    """Mock validator that simulates processing time"""
    logger.info(f"🚀 Starting mock validator on port {port} with prompt: '{prompt[:30]}...'")
    
    start_time = time.time()
    
    # Simulate processing time
    await asyncio.sleep(delay)
    
    end_time = time.time()
    processing_time = end_time - start_time
    
    logger.info(f"✅ Mock validator on port {port} completed in {processing_time:.2f}s")
    
    return {
        'port': port,
        'prompt': prompt,
        'validation_engine_score': 0.85 + (port % 10) * 0.01,  # Vary scores slightly
        'processing_time': processing_time,
        'ply_data': b'mock_ply_data_' + str(port).encode(),
        'compression': f'0.{port % 10}'
    }

async def sequential_validation():
    """Run validators sequentially"""
    logger.info(f"\n{'='*60}")
    logger.info("🐌 SEQUENTIAL VALIDATION")
    logger.info(f"{'='*60}")
    
    start_time = time.time()
    
    # Run validators one after another
    result1 = await mock_validator("A beautiful red rose", 8099, 2.0)
    result2 = await mock_validator("A futuristic robot", 8097, 2.5)
    
    end_time = time.time()
    total_time = end_time - start_time
    
    logger.info(f"⏱️  Sequential total time: {total_time:.2f}s")
    logger.info(f"   Validator 1: {result1['processing_time']:.2f}s")
    logger.info(f"   Validator 2: {result2['processing_time']:.2f}s")
    logger.info(f"   Overhead: {total_time - result1['processing_time'] - result2['processing_time']:.2f}s")
    
    return total_time, [result1, result2]

async def parallel_validation():
    """Run validators in parallel"""
    logger.info(f"\n{'='*60}")
    logger.info("🚀 PARALLEL VALIDATION")
    logger.info(f"{'='*60}")
    
    start_time = time.time()
    
    # Run validators simultaneously
    task1 = mock_validator("A beautiful red rose", 8099, 2.0)
    task2 = mock_validator("A futuristic robot", 8097, 2.5)
    
    results = await asyncio.gather(task1, task2)
    
    end_time = time.time()
    total_time = end_time - start_time
    
    logger.info(f"⏱️  Parallel total time: {total_time:.2f}s")
    logger.info(f"   Validator 1: {results[0]['processing_time']:.2f}s")
    logger.info(f"   Validator 2: {results[1]['processing_time']:.2f}s")
    logger.info(f"   Overhead: {total_time - max(r['processing_time'] for r in results):.2f}s")
    
    return total_time, results

async def main():
    """Main demo function"""
    logger.info("🧪 Starting timing comparison demo")
    
    try:
        # Run sequential validation
        seq_time, seq_results = await sequential_validation()
        
        # Run parallel validation
        par_time, par_results = await parallel_validation()
        
        # Compare results
        logger.info(f"\n{'='*60}")
        logger.info("🏆 PERFORMANCE COMPARISON")
        logger.info(f"{'='*60}")
        
        time_saved = seq_time - par_time
        savings_percent = (time_saved / seq_time) * 100
        speedup = seq_time / par_time
        
        logger.info(f"⏱️  Sequential time: {seq_time:.2f}s")
        logger.info(f"⏱️  Parallel time: {par_time:.2f}s")
        logger.info(f"💰 Time saved: {time_saved:.2f}s ({savings_percent:.1f}%)")
        logger.info(f"🚀 Speedup: {speedup:.2x}")
        
        # Show result comparison
        logger.info(f"\n{'='*60}")
        logger.info("📊 RESULT COMPARISON")
        logger.info(f"{'='*60}")
        
        logger.info(f"\n🔍 Sequential Results:")
        for result in seq_results:
            logger.info(f"   Port {result['port']}: Score {result['validation_engine_score']:.4f}, Time {result['processing_time']:.2f}s")
        
        logger.info(f"\n🔍 Parallel Results:")
        for result in par_results:
            logger.info(f"   Port {result['port']}: Score {result['validation_engine_score']:.4f}, Time {result['processing_time']:.2f}s")
        
        # Calculate efficiency
        max_individual_time = max(r['processing_time'] for r in par_results)
        efficiency = (max_individual_time / par_time) * 100
        logger.info(f"\n⚡ Parallel efficiency: {efficiency:.1f}% (best individual time vs total)")
        
        logger.info(f"\n🎉 Demo completed successfully!")
        
    except Exception as e:
        logger.error(f"❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(main())
