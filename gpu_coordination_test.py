#!/usr/bin/env python3
"""
GPU Memory Coordination Test
Test the coordination between validation and generation servers
"""

import asyncio
import aiohttp
import requests
import time
import subprocess
import psutil
from pathlib import Path


class GPUCoordinationTester:
    """Test GPU memory coordination between servers"""
    
    def __init__(self):
        self.validation_url = "http://localhost:10006"
        self.generation_url = "http://localhost:8095"
        
    async def check_validation_server(self) -> dict:
        """Check validation server status"""
        try:
            async with aiohttp.ClientSession() as session:
                # Check version
                async with session.get(f"{self.validation_url}/version/") as response:
                    if response.status == 200:
                        version = await response.text()
                        return {"status": "healthy", "version": version.strip('"')}
                    else:
                        return {"status": "error", "code": response.status}
        except Exception as e:
            return {"status": "offline", "error": str(e)}
    
    async def check_generation_server(self) -> dict:
        """Check generation server status"""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(f"{self.generation_url}/health/") as response:
                    if response.status == 200:
                        data = await response.json()
                        return {"status": "healthy", "data": data}
                    else:
                        return {"status": "error", "code": response.status}
        except Exception as e:
            return {"status": "offline", "error": str(e)}
    
    async def get_gpu_status(self) -> dict:
        """Get GPU status from validation server"""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(f"{self.validation_url}/gpu_status/") as response:
                    if response.status == 200:
                        return await response.json()
                    else:
                        return {"error": f"HTTP {response.status}"}
        except Exception as e:
            return {"error": str(e)}
    
    async def force_cleanup(self) -> dict:
        """Force GPU cleanup on validation server"""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(f"{self.validation_url}/cleanup_gpu/") as response:
                    if response.status == 200:
                        return await response.json()
                    else:
                        return {"error": f"HTTP {response.status}"}
        except Exception as e:
            return {"error": str(e)}
    
    def get_process_gpu_usage(self):
        """Get GPU memory usage by process"""
        try:
            # Use nvidia-smi to get process info
            result = subprocess.run([
                "nvidia-smi", "--query-compute-apps=pid,process_name,used_memory",
                "--format=csv,noheader,nounits"
            ], capture_output=True, text=True)
            
            if result.returncode == 0:
                processes = []
                for line in result.stdout.strip().split('\n'):
                    if line.strip():
                        parts = line.split(', ')
                        if len(parts) >= 3:
                            processes.append({
                                'pid': int(parts[0]),
                                'name': parts[1],
                                'memory_mb': int(parts[2])
                            })
                return processes
            else:
                return []
        except Exception as e:
            print(f"Error getting GPU processes: {e}")
            return []
    
    async def test_memory_coordination(self):
        """Test the full memory coordination workflow"""
        print("🧪 Testing GPU Memory Coordination")
        print("=" * 50)
        
        # Step 1: Check initial server status
        print("\n1️⃣ Checking server status...")
        val_status = await self.check_validation_server()
        gen_status = await self.check_generation_server()
        
        print(f"   Validation server: {val_status}")
        print(f"   Generation server: {gen_status}")
        
        if val_status["status"] != "healthy":
            print("❌ Validation server not healthy")
            return False
        
        if gen_status["status"] != "healthy":
            print("❌ Generation server not healthy")
            return False
        
        # Step 2: Check initial GPU status
        print("\n2️⃣ Checking initial GPU status...")
        gpu_status = await self.get_gpu_status()
        print(f"   GPU Status: {gpu_status}")
        
        # Step 3: Check GPU processes
        print("\n3️⃣ Checking GPU processes...")
        processes = self.get_process_gpu_usage()
        print(f"   Found {len(processes)} GPU processes:")
        for proc in processes:
            print(f"     PID {proc['pid']}: {proc['name']} - {proc['memory_mb']} MB")
        
        # Step 4: Force cleanup
        print("\n4️⃣ Testing GPU cleanup...")
        cleanup_result = await self.force_cleanup()
        print(f"   Cleanup result: {cleanup_result}")
        
        # Step 5: Check post-cleanup status
        print("\n5️⃣ Checking post-cleanup GPU status...")
        gpu_status_after = await self.get_gpu_status()
        print(f"   GPU Status after cleanup: {gpu_status_after}")
        
        # Step 6: Check processes after cleanup
        print("\n6️⃣ Checking GPU processes after cleanup...")
        processes_after = self.get_process_gpu_usage()
        print(f"   Found {len(processes_after)} GPU processes after cleanup:")
        for proc in processes_after:
            print(f"     PID {proc['pid']}: {proc['name']} - {proc['memory_mb']} MB")
        
        # Compare memory usage
        if 'memory_used_gb' in gpu_status and 'memory_used_gb' in gpu_status_after:
            memory_before = gpu_status['memory_used_gb']
            memory_after = gpu_status_after['memory_used_gb']
            memory_freed = memory_before - memory_after
            
            print(f"\n💾 Memory Analysis:")
            print(f"   Memory before cleanup: {memory_before:.2f} GB")
            print(f"   Memory after cleanup: {memory_after:.2f} GB")
            print(f"   Memory freed: {memory_freed:.2f} GB")
            
            if memory_freed > 0:
                print("✅ GPU memory cleanup successful!")
            else:
                print("⚠️  No significant memory freed")
        
        return True
    
    async def test_generation_with_coordination(self):
        """Test generation with memory coordination"""
        print("\n🎯 Testing Generation with Memory Coordination")
        print("=" * 50)
        
        # Step 1: Force cleanup before generation
        print("\n1️⃣ Forcing cleanup before generation...")
        cleanup_result = await self.force_cleanup()
        print(f"   Cleanup result: {cleanup_result}")
        
        # Step 2: Wait for cleanup to settle
        print("\n2️⃣ Waiting for cleanup to settle...")
        await asyncio.sleep(2)
        
        # Step 3: Test generation
        print("\n3️⃣ Testing generation...")
        try:
            data = {"prompt": "a simple wooden chair"}
            
            print("   Making generation request...")
            start_time = time.time()
            
            response = requests.post(
                f"{self.generation_url}/generate/",
                data=data,
                timeout=120
            )
            
            generation_time = time.time() - start_time
            
            if response.status_code == 200:
                ply_size = len(response.content)
                generation_id = response.headers.get('X-Generation-ID', 'unknown')
                local_score = response.headers.get('X-Local-Validation-Score', 'unknown')
                
                print(f"✅ Generation successful!")
                print(f"   Generation ID: {generation_id}")
                print(f"   PLY size: {ply_size:,} bytes")
                print(f"   Local score: {local_score}")
                print(f"   Generation time: {generation_time:.1f} seconds")
                
                return True
            else:
                print(f"❌ Generation failed with status {response.status_code}")
                print(f"   Response: {response.text}")
                return False
                
        except requests.exceptions.Timeout:
            print("❌ Generation request timed out")
            return False
        except Exception as e:
            print(f"❌ Generation failed: {e}")
            return False
    
    async def test_validation_integration(self):
        """Test validation integration"""
        print("\n✅ Testing Validation Integration")
        print("=" * 50)
        
        # Generate a test PLY first
        print("1️⃣ Generating test PLY...")
        try:
            data = {"prompt": "a red cube"}
            response = requests.post(f"{self.generation_url}/generate/", data=data, timeout=120)
            
            if response.status_code != 200:
                print("❌ Failed to generate test PLY")
                return False
            
            ply_data = response.content
            print(f"✓ Test PLY generated: {len(ply_data):,} bytes")
            
        except Exception as e:
            print(f"❌ Test PLY generation failed: {e}")
            return False
        
        # Test validation
        print("\n2️⃣ Testing local validation...")
        try:
            import pybase64
            
            encoded_data = pybase64.b64encode(ply_data).decode('utf-8')
            
            request_data = {
                "data": encoded_data,
                "prompt": "a red cube",
                "compression": 0,
                "generate_preview": False,
                "preview_score_threshold": 0.5
            }
            
            response = requests.post(
                f"{self.validation_url}/validate_txt_to_3d_ply/",
                json=request_data,
                timeout=60
            )
            
            if response.status_code == 200:
                result = response.json()
                score = result.get('score', 0.0)
                
                print(f"✅ Validation successful!")
                print(f"   Final score: {score:.3f}")
                print(f"   IQA score: {result.get('iqa', 0.0):.3f}")
                print(f"   Alignment score: {result.get('alignment_score', 0.0):.3f}")
                
                return True
            else:
                print(f"❌ Validation failed with status {response.status_code}")
                return False
                
        except Exception as e:
            print(f"❌ Validation failed: {e}")
            return False
    
    async def run_full_test(self):
        """Run all coordination tests"""
        print("🚀 GPU Memory Coordination Test Suite")
        print("=" * 60)
        
        tests = [
            ("Memory Coordination", self.test_memory_coordination()),
            ("Generation with Coordination", self.test_generation_with_coordination()),
            ("Validation Integration", self.test_validation_integration())
        ]
        
        results = []
        
        for test_name, test_coro in tests:
            print(f"\n🧪 Running: {test_name}")
            try:
                result = await test_coro
                results.append((test_name, result))
                if result:
                    print(f"✅ {test_name}: PASSED")
                else:
                    print(f"❌ {test_name}: FAILED")
            except Exception as e:
                print(f"❌ {test_name}: ERROR - {e}")
                results.append((test_name, False))
        
        # Summary
        print("\n" + "=" * 60)
        print("📊 TEST SUMMARY")
        print("=" * 60)
        
        passed = sum(1 for _, result in results if result)
        total = len(results)
        
        for test_name, result in results:
            status = "✅ PASS" if result else "❌ FAIL"
            print(f"{status} {test_name}")
        
        print(f"\nOverall: {passed}/{total} tests passed")
        
        if passed == total:
            print("🎉 All tests passed! GPU coordination is working properly.")
        else:
            print("⚠️  Some tests failed. Check the logs above for details.")
        
        return passed == total


async def main():
    """Main function"""
    tester = GPUCoordinationTester()
    success = await tester.run_full_test()
    
    if success:
        print("\n✅ Ready for full mining pipeline test!")
        print("Run: python complete_mining_pipeline_test2m3b2.py")
    else:
        print("\n❌ Fix coordination issues before running mining pipeline")


if __name__ == "__main__":
    asyncio.run(main()) 