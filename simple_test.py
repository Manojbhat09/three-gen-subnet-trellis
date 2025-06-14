#!/usr/bin/env python3
"""
Simple test for Flux-Hunyuan-BPT Generation Server
Purpose: Quick validation of core functionality
"""

import asyncio
import aiohttp
import time
import sys

SERVER_URL = "http://127.0.0.1:8095"

async def test_basic_functionality():
    """Test basic server functionality."""
    print("🧪 Simple Flux-Hunyuan-BPT Test")
    print("=" * 40)
    
    # Test 1: Health Check
    print("\n1. Testing health endpoint...")
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(f"{SERVER_URL}/health/", timeout=5) as response:
                if response.status == 200:
                    print("✅ Health check passed")
                else:
                    print(f"❌ Health check failed: {response.status}")
                    return False
    except Exception as e:
        print(f"❌ Health check failed: {e}")
        return False
    
    # Test 2: Status Check
    print("\n2. Testing status endpoint...")
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(f"{SERVER_URL}/status/", timeout=5) as response:
                if response.status == 200:
                    data = await response.json()
                    print(f"✅ Status check passed")
                    print(f"   Device: {data.get('device', 'unknown')}")
                    print(f"   BPT Enabled: {data.get('bpt_enabled', False)}")
                else:
                    print(f"❌ Status check failed: {response.status}")
                    return False
    except Exception as e:
        print(f"❌ Status check failed: {e}")
        return False
    
    # Test 3: Single Generation
    print("\n3. Testing single generation...")
    try:
        start_time = time.time()
        
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{SERVER_URL}/generate/",
                data={"prompt": "a simple red cube", "use_bpt": False, "seed": 42},
                timeout=120  # 2 minute timeout
            ) as response:
                if response.status == 200:
                    ply_data = await response.read()
                    generation_time = time.time() - start_time
                    
                    print(f"✅ Generation successful!")
                    print(f"   Time: {generation_time:.2f}s")
                    print(f"   File size: {len(ply_data):,} bytes")
                    
                    # Save the file
                    with open("simple_test_output.ply", "wb") as f:
                        f.write(ply_data)
                    print(f"   Saved: simple_test_output.ply")
                    
                    return True
                else:
                    error_text = await response.text()
                    print(f"❌ Generation failed: {response.status} - {error_text}")
                    return False
                    
    except asyncio.TimeoutError:
        print(f"❌ Generation timed out after 120s")
        return False
    except Exception as e:
        print(f"❌ Generation failed: {e}")
        return False

async def main():
    """Main test function."""
    success = await test_basic_functionality()
    
    print("\n" + "=" * 40)
    if success:
        print("🎉 All tests passed! System is working.")
        print("\n✅ Core functionality validated:")
        print("  - Server responds to health checks")
        print("  - Status endpoint provides information") 
        print("  - 3D model generation works")
        print("  - PLY file export successful")
        print("\n🚀 Ready for deployment!")
    else:
        print("❌ Some tests failed. Check server status.")
    
    return success

if __name__ == "__main__":
    print("Starting simple test...")
    success = asyncio.run(main())
    sys.exit(0 if success else 1) 