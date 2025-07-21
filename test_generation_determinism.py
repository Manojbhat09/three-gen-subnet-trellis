#!/usr/bin/env python3
"""
Test Generation Server Determinism
==================================
Purpose: Test if the generation server produces identical outputs for identical inputs
"""

import requests
import hashlib
import time

def test_generation_determinism():
    """Test if generation server is deterministic"""
    
    print("🧪 TESTING GENERATION SERVER DETERMINISM")
    print("=" * 60)
    
    # Test parameters
    test_prompt = "simple red cube"
    test_seed = 42
    num_tests = 3
    
    results = []
    
    for i in range(num_tests):
        print(f"\n🎯 Test {i+1}/{num_tests}: Generating with seed {test_seed}")
        
        try:
            response = requests.post(
                "http://localhost:8096/generate/",
                data={
                    'prompt': test_prompt,
                    'seed': test_seed,
                    'return_compressed': True
                },
                timeout=300
            )
            
            if response.status_code == 200:
                data = response.content
                data_hash = hashlib.sha256(data).hexdigest()
                size = len(data)
                
                print(f"   ✅ Generation successful: {size:,} bytes")
                print(f"   🔍 SHA256: {data_hash[:16]}...")
                
                results.append({
                    'test': i+1,
                    'size': size,
                    'hash': data_hash,
                    'data': data
                })
            else:
                print(f"   ❌ Generation failed: HTTP {response.status_code}")
                return
                
        except Exception as e:
            print(f"   ❌ Request failed: {e}")
            return
        
        # Small delay between tests
        time.sleep(2)
    
    # Analysis
    print(f"\n📊 DETERMINISM ANALYSIS")
    print("=" * 60)
    
    if len(results) < 2:
        print("❌ Not enough successful generations to compare")
        return
    
    # Compare all hashes
    first_hash = results[0]['hash']
    all_identical = all(result['hash'] == first_hash for result in results)
    
    if all_identical:
        print("✅ GENERATION IS DETERMINISTIC!")
        print(f"   All {num_tests} generations produced identical output")
        print(f"   Consistent hash: {first_hash[:16]}...")
        print(f"   Consistent size: {results[0]['size']:,} bytes")
    else:
        print("❌ GENERATION IS NOT DETERMINISTIC!")
        print("   Different outputs for same prompt+seed:")
        for result in results:
            print(f"   Test {result['test']}: {result['hash'][:16]}... ({result['size']:,} bytes)")
        
        # Find differences
        unique_hashes = set(result['hash'] for result in results)
        print(f"   Found {len(unique_hashes)} unique outputs from {num_tests} tests")
    
    print("\n" + "=" * 60)

if __name__ == "__main__":
    test_generation_determinism() 