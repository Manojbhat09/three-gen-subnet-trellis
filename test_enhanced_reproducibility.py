#!/usr/bin/env python3
"""
Test Enhanced Reproducibility System

This script tests the enhanced reproducibility system that now uses
optimized versions from logs instead of just original prompts.
"""

import json
import tempfile
import os
from llm_close_prompt_reproducibility_test import LLMClosePromptReproducibility

def create_test_enhanced_data():
    """Create test data that simulates the enhanced gold prompts from logs"""
    test_data = {
        "cylindrical glass of bubbly lemonade": {
            "original_prompt": "cylindrical glass of bubbly lemonade",
            "optimized_prompt": "a sleek, slender, clear glass cylinder made of high-quality, transparent glass, filled with a vibrant, effervescent lemonade, the bubbles gently rising to the top and dancing along the sides, creating a mesmerizing effect, front view, accurate, complete, white background",
            "best_score": 0.7196,
            "source": "log_parsing",
            "method": "comprehensive_extraction",
            "status": "completed",
            "is_gold": True,
            "current_round": 7
        },
        "intricate sandstone sculpture of cat lounging": {
            "original_prompt": "intricate sandstone sculpture of cat lounging",
            "optimized_prompt": "intricate sandstone sculpture of cat lounging, white background",
            "best_score": 0.7608,
            "source": "log_parsing",
            "method": "comprehensive_extraction",
            "status": "completed",
            "is_gold": True,
            "current_round": 1
        }
    }
    return test_data

def test_enhanced_reproducibility():
    """Test the enhanced reproducibility system with optimized versions"""
    print("🚀 Testing Enhanced Reproducibility System")
    print("=" * 70)
    
    # Create test data
    temp_dir = tempfile.mkdtemp()
    test_file = os.path.join(temp_dir, "test_memory.json")
    
    # Create minimal episodic memory (just for initialization)
    minimal_memory = {
        "test_prompt": {
            "method_2_hybrid_example": {
                "optimized_prompt": "test optimized prompt",
                "validation_results": {
                    "validation_engine_score": 0.8
                }
            }
        }
    }
    
    with open(test_file, 'w') as f:
        json.dump(minimal_memory, f)
    
    try:
        # Initialize reproducibility system
        print("🔧 Initializing reproducibility system...")
        repro_system = LLMClosePromptReproducibility(
            episodic_memory_file=test_file,
            use_vllm=True,
            vllm_url="http://localhost:9000",
            vllm_model="llama-3-2-3b-it"
        )
        
        print("✅ Reproducibility system initialized")
        print(f"   Initial gold prompts: {len(repro_system.gold_standard_results)}")
        
        # Test 1: Show initial state
        print(f"\n🔍 TEST 1: Initial State")
        print(f"   Gold prompts available: {len(repro_system.gold_standard_results)}")
        for prompt, data in repro_system.gold_standard_results.items():
            if 'method_2_hybrid_example' in data:
                optimized = data['method_2_hybrid_example']['optimized_prompt']
                score = data['method_2_hybrid_example']['validation_results']['validation_engine_score']
                print(f"     '{prompt[:50]}...' → '{optimized[:50]}...' (score: {score:.4f})")
        
        # Test 2: Update with enhanced gold prompts
        print(f"\n🔄 TEST 2: Updating with Enhanced Gold Prompts")
        enhanced_data = create_test_enhanced_data()
        print(f"   Enhanced data contains {len(enhanced_data)} prompts:")
        for prompt, data in enhanced_data.items():
            print(f"     '{prompt[:50]}...' → '{data['optimized_prompt'][:50]}...' (score: {data['best_score']:.4f})")
        
        # Update the reproducibility system
        repro_system.update_gold_standard_results(enhanced_data)
        
        # Test 3: Show updated state
        print(f"\n✅ TEST 3: Updated State")
        print(f"   Gold prompts available: {len(repro_system.gold_standard_results)}")
        for prompt, data in repro_system.gold_standard_results.items():
            if 'method_2_hybrid_example' in data:
                optimized = data['method_2_hybrid_example']['optimized_prompt']
                score = data['method_2_hybrid_example']['validation_results']['validation_engine_score']
                print(f"     '{prompt[:50]}...' → '{optimized[:50]}...' (score: {score:.4f})")
        
        # Test 4: Test reproducibility optimization with enhanced data
        print(f"\n🎯 TEST 4: Reproducibility Optimization with Enhanced Data")
        test_prompt = "intricate sandstone sculpture of cat lounging"
        print(f"   Testing prompt: '{test_prompt}'")
        
        result = repro_system.optimize_prompt_with_reproducibility(
            test_prompt, 
            min_similarity=0.3, 
            run_validation=False
        )
        
        if result:
            print(f"   ✅ Optimization SUCCESS:")
            print(f"      Original: '{result['original_prompt']}'")
            print(f"      Optimized: '{result['optimized_prompt']}'")
            print(f"      Gold prompt used: '{result['gold_prompt']}'")
            print(f"      Gold score: {result['gold_score']:.4f}")
            print(f"      Similarity: {result['similarity']:.3f}")
            
            # Check if we're using the optimized version
            if result['gold_prompt'] != "intricate sandstone sculpture of cat lounging":
                print(f"   🎉 SUCCESS: Using optimized version instead of original!")
            else:
                print(f"   ⚠️ Still using original prompt - check data structure")
        else:
            print(f"   ❌ Optimization failed")
        
        # Test 5: Test with the cylindrical glass prompt
        print(f"\n🎯 TEST 5: Testing Cylindrical Glass Optimization")
        test_prompt = "cylindrical glass of bubbly lemonade"
        print(f"   Testing prompt: '{test_prompt}'")
        
        result = repro_system.optimize_prompt_with_reproducibility(
            test_prompt, 
            min_similarity=0.3, 
            run_validation=False
        )
        
        if result:
            print(f"   ✅ Optimization SUCCESS:")
            print(f"      Original: '{result['original_prompt']}'")
            print(f"      Optimized: '{result['optimized_prompt']}'")
            print(f"      Gold prompt used: '{result['gold_prompt']}'")
            print(f"      Gold score: {result['gold_score']:.4f}")
            print(f"      Similarity: {result['similarity']:.3f}")
            
            # Check if we're using the optimized version
            if result['gold_prompt'] != "cylindrical glass of bubbly lemonade":
                print(f"   🎉 SUCCESS: Using optimized version instead of original!")
                print(f"   🎯 The optimized version should be much more detailed and high-scoring")
            else:
                print(f"   ⚠️ Still using original prompt - check data structure")
        else:
            print(f"   ❌ Optimization failed")
        
        # Test 6: Show enhanced gold prompts
        print(f"\n📊 TEST 6: Enhanced Gold Prompts Summary")
        enhanced_prompts = repro_system.get_enhanced_gold_prompts()
        print(f"   Enhanced prompts available: {len(enhanced_prompts)}")
        for prompt, data in enhanced_prompts.items():
            print(f"     '{prompt[:50]}...' → '{data['optimized_prompt'][:50]}...' (score: {data['best_score']:.4f})")
        
    except Exception as e:
        print(f"❌ Error during testing: {e}")
        import traceback
        traceback.print_exc()
        
    finally:
        import shutil
        shutil.rmtree(temp_dir)

if __name__ == "__main__":
    test_enhanced_reproducibility()
    
    print(f"\n" + "=" * 70)
    print("✅ Enhanced Reproducibility Testing Complete!")
    print("=" * 70)
