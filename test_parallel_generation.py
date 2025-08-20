#!/usr/bin/env python3
"""
Test script to demonstrate parallel generation functionality
"""
import asyncio
import time
from pathlib import Path

# Add the current directory to Python path
import sys
sys.path.append(str(Path(__file__).parent))

async def test_parallel_generation():
    """Test the parallel generation concept"""
    print("🚀 Testing Parallel Generation Concept")
    print("=" * 50)
    
    # Simulate two parallel generation tasks
    async def simulate_generation(port_num, prompt, delay):
        """Simulate generation on a specific port"""
        start_time = time.time()
        print(f"🎨 Starting generation on port {port_num} at {start_time:.2f}s")
        print(f"   Prompt: '{prompt[:30]}...'")
        
        # Simulate the generation delay
        await asyncio.sleep(delay)
        
        end_time = time.time()
        duration = end_time - start_time
        print(f"✅ Port {port_num} completed in {duration:.2f}s")
        return {'port': port_num, 'duration': duration, 'prompt': prompt}
    
    # Start both generations simultaneously
    print("🚀 Starting PARALLEL generation on both ports...")
    start_time = time.time()
    
    # Create tasks for both ports (they run in parallel)
    port1_task = simulate_generation(8097, "optimized prompt for better 3D generation", 3)
    port2_task = simulate_generation(8099, "original prompt as baseline", 2.5)
    
    # Wait for both to complete (they run in parallel)
    port1_result, port2_result = await asyncio.gather(port1_task, port2_task)
    
    total_time = time.time() - start_time
    
    print("\n📊 Results:")
    print(f"   Port 8097: {port1_result['duration']:.2f}s")
    print(f"   Port 8099: {port2_result['duration']:.2f}s")
    print(f"   Total time: {total_time:.2f}s")
    print(f"   Time saved: {max(port1_result['duration'], port2_result['duration']) - total_time:.2f}s")
    
    print("\n✅ Parallel generation test completed!")
    print("   Both generations ran simultaneously instead of sequentially")

if __name__ == "__main__":
    asyncio.run(test_parallel_generation())
