#!/usr/bin/env python3
"""
Example script demonstrating OllamaCoordinator usage for multiple RL runners.

This script shows how to coordinate access to a single Ollama server
when multiple RL optimization processes are running simultaneously.
"""

import time
import threading
import random
from episodic_prompt_optimizer import OllamaCoordinator

def simulate_rl_runner(runner_id: int, coordinator: OllamaCoordinator, num_requests: int = 5):
    """Simulate an RL runner making requests to Ollama"""
    print(f"🚀 RL Runner {runner_id} starting with {num_requests} requests")
    
    for i in range(num_requests):
        # Random priority (1=HIGH, 2=MEDIUM, 3=LOW)
        priority = random.choice([1, 2, 3])
        description = f"RL Runner {runner_id} - Request {i+1} (Priority: {priority})"
        
        print(f"📋 Runner {runner_id}: Requesting Ollama access (priority: {priority})")
        
        # Request access
        request_id = coordinator.request_access(
            priority=priority,
            description=description
        )
        
        # Wait for access
        if coordinator.wait_for_access(request_id):
            print(f"✅ Runner {runner_id}: Ollama access granted for request {i+1}")
            
            # Simulate RL optimization work
            work_time = random.uniform(2, 8)  # 2-8 seconds
            print(f"🔄 Runner {runner_id}: Working for {work_time:.1f}s...")
            time.sleep(work_time)
            
            # Release access
            coordinator.release_access(request_id)
            print(f"🔓 Runner {runner_id}: Released Ollama access for request {i+1}")
        else:
            print(f"❌ Runner {runner_id}: Failed to get Ollama access for request {i+1}")
        
        # Brief pause between requests
        time.sleep(random.uniform(1, 3))
    
    print(f"🏁 RL Runner {runner_id} completed all requests")

def monitor_queue(coordinator: OllamaCoordinator, duration: int = 30):
    """Monitor the queue status for a specified duration"""
    start_time = time.time()
    
    while time.time() - start_time < duration:
        status = coordinator.get_queue_status()
        print(f"📊 Queue Status: {status['queue_length']} pending, {status['active_requests']} active")
        
        if status['queue']:
            print("   Pending requests:")
            for req in status['queue'][:3]:  # Show first 3
                print(f"     - {req['description']} (Priority: {req['priority']})")
        
        if status['active']:
            print("   Active requests:")
            for req in status['active']:
                print(f"     - {req['description']} (Priority: {req['priority']})")
        
        print("-" * 50)
        time.sleep(3)

def main():
    """Main function demonstrating OllamaCoordinator usage"""
    print("🚀 OllamaCoordinator Demo - Multiple RL Runners")
    print("=" * 60)
    
    # Initialize coordinator
    coordinator = OllamaCoordinator(
        ollama_url="http://localhost:11434",
        max_wait_time_seconds=60,
        status_check_interval=1,
        priority_timeout_seconds=30
    )
    
    try:
        # Start monitoring thread
        monitor_thread = threading.Thread(
            target=monitor_queue, 
            args=(coordinator, 60),  # Monitor for 60 seconds
            daemon=True
        )
        monitor_thread.start()
        
        # Start multiple RL runners
        runners = []
        for runner_id in range(3):  # 3 RL runners
            runner = threading.Thread(
                target=simulate_rl_runner,
                args=(runner_id + 1, coordinator, 4),  # 4 requests each
                daemon=True
            )
            runners.append(runner)
            runner.start()
            
            # Stagger start times
            time.sleep(2)
        
        # Wait for all runners to complete
        for runner in runners:
            runner.join()
        
        print("\n🎉 All RL runners completed!")
        
        # Final status
        final_status = coordinator.get_queue_status()
        print(f"Final Queue Status: {final_status['queue_length']} pending, {final_status['active_requests']} active")
        
    except KeyboardInterrupt:
        print("\n⏹️ Demo interrupted by user")
    finally:
        # Cleanup
        print("🧹 Shutting down Ollama coordinator...")
        coordinator.shutdown()
        print("✅ Demo completed")

if __name__ == "__main__":
    main()
