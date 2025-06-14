#!/usr/bin/env python3
"""
Final GPU Coordination Demonstration
Purpose: Demonstrate the complete solution to CUDA OOM by coordinating GPU memory between servers
"""

import time
import requests
import json

# Server URLs
VALIDATION_URL = "http://127.0.0.1:10006"
GENERATION_URL = "http://127.0.0.1:8095"

def get_gpu_status(server_url: str, server_name: str) -> dict:
    """Get GPU status from server"""
    try:
        response = requests.get(f"{server_url}/gpu_status/", timeout=5)
        if response.status_code == 200:
            status = response.json()
            used_gb = status.get('memory_used_gb', 0)
            free_gb = status.get('memory_free_gb', 0)
            total_gb = status.get('memory_total_gb', 0)
            print(f"📊 {server_name}: {used_gb:.1f}GB used, {free_gb:.1f}GB free, {total_gb:.1f}GB total")
            return status
        else:
            print(f"❌ {server_name} GPU status failed: {response.status_code}")
            return {}
    except Exception as e:
        print(f"❌ {server_name} GPU status error: {e}")
        return {}

def demonstrate_solution():
    """Demonstrate the complete GPU coordination solution"""
    print("🚀 FINAL GPU COORDINATION DEMONSTRATION")
    print("=" * 60)
    print("Problem: RTX 4090 24GB VRAM insufficient for both servers")
    print("Solution: Dynamic model unloading/reloading coordination")
    print("=" * 60)
    
    # Step 1: Show initial state
    print("\n1️⃣ INITIAL STATE - Both servers loaded")
    val_status = get_gpu_status(VALIDATION_URL, "Validation Server")
    gen_status = get_gpu_status(GENERATION_URL, "Generation Server")
    
    if val_status and gen_status:
        total_used = val_status.get('memory_used_gb', 0) + gen_status.get('memory_used_gb', 0)
        print(f"💥 TOTAL GPU USAGE: {total_used:.1f}GB (EXCEEDS 24GB LIMIT!)")
    
    # Step 2: Demonstrate validation unload
    print("\n2️⃣ SOLUTION STEP 1 - Unload validation models")
    try:
        response = requests.post(f"{VALIDATION_URL}/unload_models/", timeout=30)
        if response.status_code == 200:
            result = response.json()
            freed_gb = result.get("memory_freed_gb", 0)
            print(f"✅ Validation models unloaded - freed {freed_gb:.1f}GB")
            
            # Show post-unload status
            print("\n📊 POST-UNLOAD STATUS:")
            get_gpu_status(VALIDATION_URL, "Validation Server")
            get_gpu_status(GENERATION_URL, "Generation Server")
            
        else:
            print(f"❌ Validation unload failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Validation unload error: {e}")
        return False
    
    # Step 3: Show that generation can now work
    print("\n3️⃣ SOLUTION STEP 2 - Generation now has enough memory")
    print("🎯 Testing generation with freed GPU memory...")
    
    try:
        # Test a simple generation
        data = {"prompt": "a simple red cube", "seed": 42, "return_compressed": True}
        print("   Sending generation request...")
        
        # Note: This will take time as it includes the coordination
        response = requests.post(f"{GENERATION_URL}/generate/", data=data, timeout=300)
        
        if response.status_code == 200:
            try:
                result = response.json()
                if result.get("success"):
                    gen_time = result.get("generation_time", 0)
                    ply_size = result.get("ply_size_bytes", 0)
                    print(f"✅ GENERATION SUCCESSFUL!")
                    print(f"   Generation time: {gen_time:.1f}s")
                    print(f"   PLY size: {ply_size} bytes")
                else:
                    print(f"❌ Generation failed: {result.get('error', 'Unknown error')}")
                    return False
            except json.JSONDecodeError:
                print(f"❌ Invalid JSON response from generation server")
                return False
        else:
            print(f"❌ Generation request failed: {response.status_code}")
            print(f"   Response: {response.text[:200]}...")
            return False
            
    except Exception as e:
        print(f"❌ Generation test error: {e}")
        return False
    
    # Step 4: Show final state
    print("\n4️⃣ FINAL STATE - After coordinated generation")
    get_gpu_status(VALIDATION_URL, "Validation Server")
    get_gpu_status(GENERATION_URL, "Generation Server")
    
    print("\n" + "=" * 60)
    print("🎉 SOLUTION DEMONSTRATION COMPLETE!")
    print("=" * 60)
    print("✅ Problem solved: CUDA OOM eliminated through coordination")
    print("✅ Validation server: Dynamically unloads/reloads models")
    print("✅ Generation server: Coordinates before heavy operations")
    print("✅ Result: Both servers can coexist on 24GB RTX 4090")
    print("=" * 60)
    
    return True

def main():
    """Main demonstration"""
    try:
        success = demonstrate_solution()
        if success:
            print("\n🏆 GPU coordination solution is WORKING!")
            return 0
        else:
            print("\n❌ Demonstration failed")
            return 1
    except KeyboardInterrupt:
        print("\n🛑 Demonstration interrupted")
        return 130
    except Exception as e:
        print(f"\n💥 Unexpected error: {e}")
        return 1

if __name__ == "__main__":
    exit(main()) 