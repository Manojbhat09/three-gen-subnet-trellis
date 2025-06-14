#!/usr/bin/env python3
"""
Test GPU Memory Coordination
Purpose: Test the coordination between validation and generation servers for GPU memory management
"""

import asyncio
import time
import requests
import subprocess
import signal
import os
from typing import Dict, Optional

# Server configurations
VALIDATION_SERVER_URL = "http://127.0.0.1:10006"
GENERATION_SERVER_URL = "http://127.0.0.1:8095"

class GPUCoordinationTester:
    def __init__(self):
        self.validation_process = None
        self.generation_process = None
        
    def check_server_health(self, url: str, name: str) -> bool:
        """Check if a server is healthy"""
        try:
            # Use different endpoints for different servers
            if "10006" in url:  # Validation server
                endpoint = "/version/"
            else:  # Generation server
                endpoint = "/health/"
                
            response = requests.get(f"{url}{endpoint}", timeout=5)
            if response.status_code == 200:
                print(f"✅ {name} server is healthy")
                return True
            else:
                print(f"❌ {name} server returned {response.status_code}")
                return False
        except Exception as e:
            print(f"❌ {name} server is not responding: {e}")
            return False
    
    def get_gpu_status(self, server_url: str, server_name: str) -> Optional[Dict]:
        """Get GPU status from a server"""
        try:
            response = requests.get(f"{server_url}/gpu_status/", timeout=5)
            if response.status_code == 200:
                status = response.json()
                print(f"📊 {server_name} GPU Status:")
                print(f"   Total: {status.get('memory_total_gb', 0):.1f} GB")
                print(f"   Used: {status.get('memory_used_gb', 0):.1f} GB")
                print(f"   Free: {status.get('memory_free_gb', 0):.1f} GB")
                print(f"   Usage: {status.get('memory_used_percent', 0):.1f}%")
                return status
            else:
                print(f"❌ Failed to get GPU status from {server_name}: {response.status_code}")
                return None
        except Exception as e:
            print(f"❌ Error getting GPU status from {server_name}: {e}")
            return None
    
    def test_validation_unload(self) -> bool:
        """Test validation server model unloading"""
        print("\n🧪 Testing validation server model unloading...")
        
        # Get initial status
        initial_status = self.get_gpu_status(VALIDATION_SERVER_URL, "Validation")
        if not initial_status:
            return False
        
        initial_used = initial_status.get('memory_used_gb', 0)
        
        # Request model unload
        try:
            response = requests.post(f"{VALIDATION_SERVER_URL}/unload_models/", timeout=30)
            if response.status_code == 200:
                result = response.json()
                if result.get("status") == "success":
                    freed_gb = result.get("memory_freed_gb", 0)
                    print(f"✅ Validation server unloaded models, freed {freed_gb:.1f}GB")
                    
                    # Get post-unload status
                    post_status = self.get_gpu_status(VALIDATION_SERVER_URL, "Validation (post-unload)")
                    if post_status:
                        post_used = post_status.get('memory_used_gb', 0)
                        actual_freed = initial_used - post_used
                        print(f"📈 Actual memory freed: {actual_freed:.1f}GB")
                        return actual_freed > 3.0  # Should free at least 3GB
                    return True
                else:
                    print(f"❌ Validation server unload failed: {result}")
                    return False
            else:
                print(f"❌ Validation server unload request failed: {response.status_code}")
                return False
        except Exception as e:
            print(f"❌ Error testing validation unload: {e}")
            return False
    
    def test_validation_reload(self) -> bool:
        """Test validation server model reloading"""
        print("\n🧪 Testing validation server model reloading...")
        
        try:
            response = requests.post(f"{VALIDATION_SERVER_URL}/reload_models/", timeout=60)
            if response.status_code == 200:
                result = response.json()
                if result.get("models_reloaded"):
                    reload_time = result.get("reload_time", 0)
                    print(f"✅ Validation server reloaded models in {reload_time:.1f}s")
                    
                    # Get post-reload status
                    self.get_gpu_status(VALIDATION_SERVER_URL, "Validation (post-reload)")
                    return True
                else:
                    print(f"❌ Validation server reload failed: {result}")
                    return False
            else:
                print(f"❌ Validation server reload request failed: {response.status_code}")
                return False
        except Exception as e:
            print(f"❌ Error testing validation reload: {e}")
            return False
    
    def test_generation_with_coordination(self) -> bool:
        """Test generation with GPU coordination"""
        print("\n🧪 Testing generation with GPU coordination...")
        
        try:
            # Test a simple generation
            data = {
                "prompt": "a red cube",
                "seed": 42,
                "return_compressed": True
            }
            
            print("🎯 Starting coordinated generation...")
            response = requests.post(f"{GENERATION_SERVER_URL}/generate/", data=data, timeout=300)
            
            if response.status_code == 200:
                result = response.json()
                if result.get("success"):
                    print("✅ Coordinated generation successful!")
                    print(f"   Generation time: {result.get('generation_time', 0):.1f}s")
                    print(f"   PLY size: {result.get('ply_size_bytes', 0)} bytes")
                    return True
                else:
                    print(f"❌ Generation failed: {result.get('error', 'Unknown error')}")
                    return False
            else:
                print(f"❌ Generation request failed: {response.status_code}")
                try:
                    error_detail = response.json()
                    print(f"   Error details: {error_detail}")
                except:
                    print(f"   Response text: {response.text}")
                return False
                
        except Exception as e:
            print(f"❌ Error testing coordinated generation: {e}")
            return False
    
    def run_comprehensive_test(self) -> bool:
        """Run comprehensive GPU coordination test"""
        print("🚀 GPU Memory Coordination Test")
        print("=" * 50)
        
        # Check server health
        print("\n1️⃣ Checking server health...")
        validation_healthy = self.check_server_health(VALIDATION_SERVER_URL, "Validation")
        generation_healthy = self.check_server_health(GENERATION_SERVER_URL, "Generation")
        
        if not validation_healthy or not generation_healthy:
            print("❌ One or more servers are not healthy. Please start them first.")
            return False
        
        # Test validation server unload
        print("\n2️⃣ Testing validation server coordination...")
        unload_success = self.test_validation_unload()
        
        if not unload_success:
            print("❌ Validation server unload test failed")
            return False
        
        # Test generation with coordination
        print("\n3️⃣ Testing coordinated generation...")
        generation_success = self.test_generation_with_coordination()
        
        # Test validation server reload (should happen automatically, but test manually too)
        print("\n4️⃣ Testing validation server reload...")
        reload_success = self.test_validation_reload()
        
        # Final status
        print("\n5️⃣ Final GPU status check...")
        self.get_gpu_status(VALIDATION_SERVER_URL, "Validation (final)")
        self.get_gpu_status(GENERATION_SERVER_URL, "Generation (final)")
        
        # Summary
        print("\n" + "=" * 50)
        print("📊 TEST SUMMARY")
        print("=" * 50)
        print(f"Server health:           {'✅' if validation_healthy and generation_healthy else '❌'}")
        print(f"Validation unload:       {'✅' if unload_success else '❌'}")
        print(f"Coordinated generation:  {'✅' if generation_success else '❌'}")
        print(f"Validation reload:       {'✅' if reload_success else '❌'}")
        
        overall_success = all([validation_healthy, generation_healthy, unload_success, generation_success, reload_success])
        print(f"\nOverall result:          {'✅ SUCCESS' if overall_success else '❌ FAILED'}")
        
        return overall_success

def main():
    """Main test function"""
    tester = GPUCoordinationTester()
    
    try:
        success = tester.run_comprehensive_test()
        return 0 if success else 1
        
    except KeyboardInterrupt:
        print("\n🛑 Test interrupted by user")
        return 130
    except Exception as e:
        print(f"\n💥 Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    exit(main()) 