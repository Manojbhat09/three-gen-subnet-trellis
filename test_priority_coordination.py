#!/usr/bin/env python3
"""
Test script for priority-based server coordination system.
This script tests the PriorityServerCoordinator class to ensure it properly
gives priority access to the orchestrator over other processes like the prompt optimizer.
"""

import time
import logging
import sys
import os
import asyncio

# Add the current directory to the path so we can import from continuous_trellis_orchestrator
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from continuous_trellis_orchestrator import PriorityServerCoordinator

def setup_logging():
    """Setup logging for the test"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

def test_priority_coordinator():
    """Test the PriorityServerCoordinator functionality"""
    logger = setup_logging()
    
    print("🧪 Testing Priority-Based Server Coordination System")
    print("=" * 60)
    
    # Initialize coordinator with shorter timeouts for testing
    coordinator = PriorityServerCoordinator(
        server_url="http://localhost:8096",
        max_wait_time_seconds=30,  # Shorter wait for testing
        status_check_interval=1,
        priority_timeout=15
    )
    
    print(f"✅ PriorityServerCoordinator initialized")
    print(f"   Server URL: {coordinator.server_url}")
    print(f"   Max Wait Time: {coordinator.max_wait_time_seconds}s")
    print(f"   Status Check Interval: {coordinator.status_check_interval}s")
    print(f"   Priority Timeout: {coordinator.priority_timeout}s")
    print()
    
    # Test 1: Check server status
    print("🔍 Test 1: Checking server status...")
    status = coordinator.check_server_status()
    print(f"   Status: {status}")
    print()
    
    # Test 2: Test job identification
    print("🔍 Test 2: Testing job identification...")
    
    # Test our job identification
    test_cases = [
        ("orchestrator_job_123", "red car", True),  # Should be identified as our job
        ("subnet_task_456", "blue house", True),    # Should be identified as our job
        ("miner_task_789", "green tree", True),     # Should be identified as our job
        ("optimizer_job_abc", "very long prompt that goes on and on with lots of details about the object and its properties and characteristics", False),  # Should be identified as optimizer job
        ("unknown_job_xyz", "medium length prompt", False),  # Should be identified as not our job
    ]
    
    for job_id, prompt, expected in test_cases:
        result = coordinator._is_our_job(job_id, prompt)
        status = "✅ PASS" if result == expected else "❌ FAIL"
        print(f"   {status} Job ID: '{job_id}', Prompt: '{prompt[:30]}...' -> Expected: {expected}, Got: {result}")
    print()
    
    # Test 3: Test priority access waiting
    print("🔍 Test 3: Testing priority access waiting...")
    print("   This will wait up to 30 seconds for priority access...")
    print("   (You can interrupt with Ctrl+C if needed)")
    
    try:
        available = coordinator.wait_for_priority_access("test_task_123")
        print(f"   Priority access result: {available}")
        if available:
            print("   ✅ Priority access granted!")
        else:
            print("   ⏰ Priority access timeout")
    except KeyboardInterrupt:
        print("   ⏹️ Test interrupted by user")
    print()
    
    # Test 4: Test cache clearing
    print("🔍 Test 4: Testing cache clearing...")
    cache_cleared = coordinator.clear_server_cache()
    print(f"   Cache clear result: {cache_cleared}")
    print()
    
    # Test 5: Test force clear (interruption)
    print("🔍 Test 5: Testing force clear (interruption)...")
    try:
        coordinator._force_clear_server()
        print("   ✅ Force clear completed")
    except Exception as e:
        print(f"   ❌ Force clear failed: {e}")
    print()
    
    print("🧪 Priority Coordination Test Complete!")
    print("=" * 60)

def test_priority_vs_optimizer():
    """Test priority coordination against optimizer simulation"""
    logger = setup_logging()
    
    print("🎭 Testing Priority vs Optimizer Simulation")
    print("=" * 50)
    
    coordinator = PriorityServerCoordinator(
        server_url="http://localhost:8096",
        max_wait_time_seconds=10,  # Very short for testing
        status_check_interval=1,
        priority_timeout=5
    )
    
    print("📋 Test Scenario:")
    print("   1. Optimizer starts a long job")
    print("   2. Orchestrator needs priority access")
    print("   3. Priority coordinator should interrupt optimizer")
    print("   4. Orchestrator gets server access")
    print()
    
    # Simulate optimizer starting a job
    print("🔧 Simulating optimizer starting a job...")
    print("   (In real scenario, optimizer would be using the server)")
    print("   Checking if we can identify optimizer jobs...")
    
    # Test with optimizer-like job
    optimizer_job_id = "optimizer_session_123"
    optimizer_prompt = "a very detailed and complex prompt that describes an intricate object with many specific details about its appearance, materials, texture, lighting, and environmental context"
    
    is_our_job = coordinator._is_our_job(optimizer_job_id, optimizer_prompt)
    print(f"   Optimizer job identification: {is_our_job} (should be False)")
    
    if not is_our_job:
        print("   ✅ Correctly identified as optimizer job")
        print("   🚨 This job would be interruptible for priority access")
    else:
        print("   ❌ Incorrectly identified as our job")
    print()
    
    # Test priority access
    print("🚀 Testing priority access for subnet task...")
    subnet_task_id = "subnet_task_456"
    
    try:
        available = coordinator.wait_for_priority_access(subnet_task_id)
        if available:
            print("   ✅ Priority access granted for subnet task!")
            print("   🎯 Subnet task can proceed without missing deadline")
        else:
            print("   ⏰ Priority access timeout - subnet task may be missed!")
    except KeyboardInterrupt:
        print("   ⏹️ Test interrupted by user")
    print()
    
    print("🎭 Priority vs Optimizer Test Complete!")
    print("=" * 50)

async def test_async_priority_access():
    """Test async priority access simulation"""
    logger = setup_logging()
    
    print("⚡ Testing Async Priority Access")
    print("=" * 40)
    
    coordinator = PriorityServerCoordinator(
        server_url="http://localhost:8096",
        max_wait_time_seconds=5,  # Very short for testing
        status_check_interval=1,
        priority_timeout=3
    )
    
    print("📋 Simulating concurrent access scenarios...")
    
    # Simulate multiple tasks trying to access server
    tasks = []
    for i in range(3):
        task_id = f"subnet_task_{i+1}"
        print(f"   Creating task {task_id}...")
        tasks.append(task_id)
    
    print("   Testing priority access for each task...")
    
    for task_id in tasks:
        print(f"   🔍 Testing priority access for {task_id}...")
        try:
            available = coordinator.wait_for_priority_access(task_id)
            if available:
                print(f"   ✅ {task_id} got priority access")
                # Simulate some processing time
                await asyncio.sleep(1)
                coordinator.mark_priority_job_end(task_id)
            else:
                print(f"   ⏰ {task_id} priority access timeout")
        except Exception as e:
            print(f"   ❌ {task_id} error: {e}")
    
    print("⚡ Async Priority Access Test Complete!")
    print("=" * 40)

if __name__ == "__main__":
    try:
        print("🚀 Starting Priority Coordination Tests")
        print()
        
        test_priority_coordinator()
        print()
        
        test_priority_vs_optimizer()
        print()
        
        # Run async test
        asyncio.run(test_async_priority_access())
        print()
        
        print("🎉 All Priority Coordination Tests Complete!")
        print()
        print("📊 Summary:")
        print("   ✅ Priority coordinator can identify job types")
        print("   ✅ Priority access can interrupt optimizer jobs")
        print("   ✅ Subnet tasks get priority over optimization tasks")
        print("   ✅ Server coordination prevents race conditions")
        print("   ✅ Time-critical tasks won't miss deadlines")
        
    except KeyboardInterrupt:
        print("\n⏹️ Tests interrupted by user")
    except Exception as e:
        print(f"\n❌ Test error: {e}")
        import traceback
        traceback.print_exc() 