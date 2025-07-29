#!/usr/bin/env python3
"""
Test script for server coordination system.
This script tests the ServerCoordinator class to ensure it properly
coordinates with the GPU server and prevents race conditions.
"""

import time
import logging
import sys
import os

# Add the current directory to the path so we can import from episodic_prompt_optimizer
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from episodic_prompt_optimizer import ServerCoordinator

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

def test_server_coordinator():
    """Test the ServerCoordinator functionality"""
    logger = setup_logging()
    
    print("🧪 Testing Server Coordination System")
    print("=" * 50)
    
    # Initialize coordinator with shorter timeouts for testing
    coordinator = ServerCoordinator(
        server_url="http://localhost:8096",
        buffer_time_seconds=10,  # Shorter buffer for testing
        max_wait_time_seconds=60,  # Shorter wait for testing
        status_check_interval=2
    )
    
    print(f"✅ ServerCoordinator initialized")
    print(f"   Server URL: {coordinator.server_url}")
    print(f"   Buffer Time: {coordinator.buffer_time_seconds}s")
    print(f"   Max Wait Time: {coordinator.max_wait_time_seconds}s")
    print(f"   Status Check Interval: {coordinator.status_check_interval}s")
    print()
    
    # Test 1: Check server status
    print("🔍 Test 1: Checking server status...")
    status = coordinator.check_server_status()
    print(f"   Status: {status}")
    print()
    
    # Test 2: Test health endpoint
    print("🔍 Test 2: Testing health endpoint...")
    try:
        import requests
        health_resp = requests.get(f"{coordinator.server_url}/health/", timeout=5)
        print(f"   Health endpoint status: {health_resp.status_code}")
        if health_resp.status_code == 200:
            print(f"   Health response: {health_resp.json()}")
        else:
            print(f"   Health check failed: {health_resp.text}")
    except Exception as e:
        print(f"   Health check error: {e}")
    print()
    
    # Test 3: Test job status endpoint
    print("🔍 Test 3: Testing job status endpoint...")
    try:
        job_resp = requests.get(f"{coordinator.server_url}/job/status/", timeout=5)
        print(f"   Job status endpoint status: {job_resp.status_code}")
        if job_resp.status_code == 200:
            job_data = job_resp.json()
            print(f"   Job status: {job_data.get('status', 'unknown')}")
            print(f"   Job ID: {job_data.get('job_id', 'none')}")
            print(f"   Prompt: {job_data.get('prompt', 'none')}")
        else:
            print(f"   Job status check failed: {job_resp.text}")
    except Exception as e:
        print(f"   Job status check error: {e}")
    print()
    
    # Test 4: Test cache clearing
    print("🔍 Test 4: Testing cache clearing...")
    cache_cleared = coordinator.clear_server_cache()
    print(f"   Cache clear result: {cache_cleared}")
    print()
    
    # Test 5: Test server availability waiting
    print("🔍 Test 5: Testing server availability waiting...")
    print("   This will wait up to 60 seconds for server to become available...")
    print("   (You can interrupt with Ctrl+C if needed)")
    
    try:
        available = coordinator.wait_for_server_availability()
        print(f"   Server availability result: {available}")
        if available:
            print("   ✅ Server is available!")
        else:
            print("   ⏰ Server availability timeout")
    except KeyboardInterrupt:
        print("   ⏹️ Test interrupted by user")
    print()
    
    # Test 6: Test buffer time functionality
    print("🔍 Test 6: Testing buffer time functionality...")
    print("   Marking server as used...")
    coordinator.mark_server_used()
    
    print("   Checking status immediately after marking as used...")
    status = coordinator.check_server_status()
    print(f"   Status after marking as used: {status}")
    
    if not status.get("available", False):
        print("   ✅ Buffer time is working - server marked as unavailable")
    else:
        print("   ⚠️ Buffer time may not be working correctly")
    print()
    
    print("🧪 Server Coordination Test Complete!")
    print("=" * 50)

def test_server_endpoints():
    """Test individual server endpoints"""
    logger = setup_logging()
    
    print("🔌 Testing Individual Server Endpoints")
    print("=" * 40)
    
    base_url = "http://localhost:8096"
    
    endpoints = [
        ("/health/", "Health Check"),
        ("/status/", "Server Status"),
        ("/job/status/", "Job Status"),
        ("/config/", "Configuration")
    ]
    
    for endpoint, name in endpoints:
        print(f"🔍 Testing {name} ({endpoint})...")
        try:
            import requests
            resp = requests.get(f"{base_url}{endpoint}", timeout=5)
            print(f"   Status: {resp.status_code}")
            if resp.status_code == 200:
                try:
                    data = resp.json()
                    print(f"   Response: {data}")
                except:
                    print(f"   Response: {resp.text[:100]}...")
            else:
                print(f"   Error: {resp.text}")
        except Exception as e:
            print(f"   Exception: {e}")
        print()

if __name__ == "__main__":
    try:
        test_server_endpoints()
        print()
        test_server_coordinator()
    except KeyboardInterrupt:
        print("\n⏹️ Test interrupted by user")
    except Exception as e:
        print(f"\n❌ Test error: {e}")
        import traceback
        traceback.print_exc() 