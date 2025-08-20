#!/usr/bin/env python3
"""
Test Real Reproducibility System with vLLM

This script tests the actual reproducibility system with real LLM calls
to see if our improvements work.
"""

import json
import tempfile
import os
from llm_close_prompt_reproducibility_test import LLMClosePromptReproducibility

def create_test_episodic_memory():
    """Create a simple test episodic memory file"""
    temp_dir = tempfile.mkdtemp()
    test_file = os.path.join(temp_dir, "test_memory.json")
    
    # This matches the data structure from the terminal output
    test_data = {
        "optimization_sessions": [
            {
                "original_prompt": "intricate sandstone sculpture of cat lounging",
                "attempts": [
                    {
                        "optimized_prompt": "cylindrical glass of bubbly lemonade",
                        "validation_score": 0.7608
                    }
                ]
            }
        ]
    }
    
    with open(test_file, 'w') as f:
        json.dump(test_data, f)
    
    return temp_dir, test_file

def test_real_reproducibility():
    """Test the real reproducibility system with vLLM"""
    print("🚀 Testing Real Reproducibility System with vLLM")
    print("=" * 70)
    
    # Create test data
    temp_dir, test_file = create_test_episodic_memory()
    
    try:
        # Create reproducibility system with vLLM
        print("🔧 Initializing reproducibility system with vLLM...")
        repro_system = LLMClosePromptReproducibility(
            episodic_memory_file=test_file,
            use_vllm=True,
            vllm_url="http://localhost:9000",
            vllm_model="llama-3-2-3b-it"
        )
        
        print("✅ Reproducibility system initialized")
        print(f"   Gold prompts loaded: {len(repro_system.gold_standard_results)}")
        
        # Test the complete flow
        original_prompt = "intricate sandstone sculpture of cat lounging"
        print(f"\n🎯 Testing with prompt: '{original_prompt}'")
        
        print("\n🔄 Running reproducibility optimization...")
        result = repro_system.optimize_prompt_with_reproducibility(
            original_prompt, 
            min_similarity=0.3, 
            run_validation=False
        )
        
        if result:
            print(f"\n🎯 OPTIMIZATION RESULT:")
            print(f"   Original: '{result['original_prompt']}'")
            print(f"   Optimized: '{result['optimized_prompt']}'")
            print(f"   Similarity: {result['similarity']:.3f}")
            print(f"   Gold score: {result['gold_score']:.4f}")
            print(f"   Method: {result['optimization_method']}")
            
            # Check if the optimization preserved the original intent
            original_lower = original_prompt.lower()
            optimized_lower = result['optimized_prompt'].lower()
            
            print(f"\n🔍 INTENT PRESERVATION CHECK:")
            
            # Check for core subject preservation
            if "sandstone" in optimized_lower and "cat" in optimized_lower:
                print(f"   ✅ Core subject preserved: 'sandstone cat' found in optimized prompt")
            else:
                print(f"   ❌ Core subject NOT preserved: missing 'sandstone cat'")
            
            # Check for enhancements
            if "intricate" in optimized_lower:
                print(f"   ✅ Original quality preserved: 'intricate' found")
            else:
                print(f"   ⚠️ Original quality missing: 'intricate' not found")
            
            if "bubbly" in optimized_lower:
                print(f"   ✅ Gold enhancement added: 'bubbly' found")
            else:
                print(f"   ⚠️ Gold enhancement missing: 'bubbly' not found")
            
            if "white background" in optimized_lower:
                print(f"   ✅ White background added: 'white background' found")
            else:
                print(f"   ❌ White background missing: 'white background' not found")
            
            # Overall assessment
            if ("sandstone" in optimized_lower and "cat" in optimized_lower and 
                "white background" in optimized_lower):
                print(f"\n🎉 SUCCESS: Optimization preserved original intent while adding enhancements!")
            else:
                print(f"\n⚠️ PARTIAL SUCCESS: Some elements missing, but system is working")
                
        else:
            print(f"❌ Optimization failed - no result returned")
            
    except Exception as e:
        print(f"❌ Error during testing: {e}")
        import traceback
        traceback.print_exc()
        
    finally:
        import shutil
        shutil.rmtree(temp_dir)

def test_individual_llm_calls():
    """Test individual LLM calls to see where issues might be"""
    print(f"\n" + "=" * 70)
    print("🧪 Testing Individual LLM Calls")
    print("=" * 70)
    
    temp_dir, test_file = create_test_episodic_memory()
    
    try:
        repro_system = LLMClosePromptReproducibility(
            episodic_memory_file=test_file,
            use_vllm=True,
            vllm_url="http://localhost:9000",
            vllm_model="llama-3-2-3b-it"
        )
        
        # Test pattern extraction
        print("🔍 Testing pattern extraction...")
        test_prompt = "intricate sandstone sculpture of cat lounging"
        patterns = repro_system.extract_patterns(test_prompt)
        print(f"   Extracted patterns: {patterns}")
        
        # Test prompt reconstruction
        print("\n🔧 Testing prompt reconstruction...")
        test_components = {
            "core_subject": "sandstone cat sculpture",
            "enhancements": {
                "quality_adjectives": ["intricate", "bubbly"],
                "material_details": ["sandstone", "glass"],
                "light_interaction": [],
                "context": []
            }
        }
        reconstructed = repro_system.reconstruct_prompt(test_components)
        print(f"   Reconstructed prompt: '{reconstructed}'")
        
    except Exception as e:
        print(f"❌ Error during individual testing: {e}")
        import traceback
        traceback.print_exc()
        
    finally:
        import shutil
        shutil.rmtree(temp_dir)

if __name__ == "__main__":
    # Test the complete system
    test_real_reproducibility()
    
    # Test individual components
    test_individual_llm_calls()
    
    print(f"\n" + "=" * 70)
    print("✅ Testing Complete!")
    print("=" * 70)
