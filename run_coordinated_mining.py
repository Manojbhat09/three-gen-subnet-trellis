#!/usr/bin/env python3
"""
Coordinated Mining Pipeline Launcher
Runs the complete mining pipeline with GPU memory coordination
"""

import subprocess
import time
import requests
import sys
from pathlib import Path


def check_server_health(url: str, timeout: int = 5) -> bool:
    """Check if server is healthy"""
    try:
        response = requests.get(url, timeout=timeout)
        return response.status_code == 200
    except:
        return False


def force_gpu_cleanup():
    """Force GPU cleanup via validation server"""
    try:
        response = requests.post("http://localhost:10006/cleanup_gpu/", timeout=10)
        if response.status_code == 200:
            result = response.json()
            print(f"✅ GPU cleanup: {result['memory_used_gb']:.1f}GB -> {result['memory_free_gb']:.1f}GB free")
            return True
        else:
            print(f"⚠️  GPU cleanup returned status {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ GPU cleanup failed: {e}")
        return False


def main():
    print("🚀 Coordinated Mining Pipeline with GPU Memory Management")
    print("=" * 70)
    
    # Step 1: Check servers
    print("\n1️⃣ Checking server health...")
    val_health = check_server_health("http://localhost:10006/version/")
    gen_health = check_server_health("http://localhost:8095/health/")
    
    print(f"   Validation server: {'✅ Healthy' if val_health else '❌ Not responding'}")
    print(f"   Generation server: {'✅ Healthy' if gen_health else '❌ Not responding'}")
    
    if not val_health or not gen_health:
        print("\n❌ Servers not healthy. Please start them first:")
        print("   Validation: conda activate three-gen-validation && cd validation && python serve.py --host 0.0.0.0 --port 10006")
        print("   Generation: conda activate hunyuan3d && python flux_hunyuan_sugar_generation_server.py")
        return 1
    
    # Step 2: Force initial GPU cleanup
    print("\n2️⃣ Initial GPU memory coordination...")
    if not force_gpu_cleanup():
        print("⚠️  GPU cleanup failed, but continuing anyway...")
    
    # Step 3: Test coordination with a simple generation
    print("\n3️⃣ Testing generation coordination...")
    try:
        print("   Forcing GPU cleanup before test generation...")
        force_gpu_cleanup()
        time.sleep(2)
        
        print("   Testing simple generation...")
        response = requests.post(
            "http://localhost:8095/generate/",
            data={"prompt": "a red cube"},
            timeout=120
        )
        
        if response.status_code == 200:
            print(f"✅ Test generation successful! PLY size: {len(response.content):,} bytes")
        else:
            print(f"❌ Test generation failed: {response.status_code}")
            print("   This suggests memory coordination needs refinement")
            
            # Try one more aggressive cleanup
            print("   Attempting more aggressive cleanup...")
            subprocess.run(["pkill", "-f", "python"], check=False)
            time.sleep(5)
            print("   Please restart servers and try again")
            return 1
            
    except Exception as e:
        print(f"❌ Test generation failed: {e}")
        return 1
    
    # Step 4: Run full mining pipeline
    print("\n4️⃣ Running full mining pipeline...")
    print("   Using registered miner: test2m3b2 wallet, t2m3b21 hotkey")
    print("   Subnet: 17")
    print("   Max tasks: 3")
    
    try:
        # Run in hunyuan3d environment for bittensor compatibility
        result = subprocess.run([
            "conda", "run", "-n", "hunyuan3d",
            "python", "complete_mining_pipeline_test2m3b2.py"
        ], check=False)
        
        if result.returncode == 0:
            print("\n🎉 Mining pipeline completed successfully!")
        else:
            print(f"\n⚠️  Mining pipeline ended with code {result.returncode}")
    
    except KeyboardInterrupt:
        print("\n⚠️  Mining interrupted by user")
    except Exception as e:
        print(f"\n❌ Mining pipeline error: {e}")
    
    print("\n✅ Coordinated mining pipeline completed!")
    return 0


if __name__ == "__main__":
    sys.exit(main()) 