#!/usr/bin/env python3
"""
Simple Test to Understand Reproducibility System Flow

This script demonstrates step-by-step how the reproducibility system works,
showing the "jump" from "found close gold prompt" to "reproducibility optimization SUCCESS".
"""

import json
import tempfile
import os
from unittest.mock import Mock, patch

# Import the reproducibility system
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

def demonstrate_step_by_step():
    """Demonstrate the complete flow step by step"""
    print("🎯 REPRODUCIBILITY SYSTEM STEP-BY-STEP DEMONSTRATION")
    print("=" * 70)
    
    # Create test data
    temp_dir, test_file = create_test_episodic_memory()
    
    try:
        # Mock the LLM optimizer to avoid actual API calls
        with patch('llm_close_prompt_reproducibility_test.LLMPromptOptimizer') as mock_optimizer_class:
            mock_optimizer = Mock()
            mock_optimizer._query_llm = Mock()
            mock_optimizer_class.return_value = mock_optimizer
            
            # Create reproducibility system with vLLM (since Ollama isn't running)
            repro_system = LLMClosePromptReproducibility(
                episodic_memory_file=test_file,
                use_vllm=True,
                vllm_url="http://localhost:9000",
                vllm_model="llama-3-2-3b-it"
            )
            
            print("📚 STEP 1: Loading episodic memory...")
            gold_count = len(repro_system.gold_standard_results)
            print(f"   ✅ Loaded {gold_count} gold prompts from episodic memory")
            
            # Show what's in the memory
            for prompt, data in repro_system.gold_standard_results.items():
                print(f"   📝 Gold prompt: '{prompt}'")
                if 'method_2_hybrid_example' in data:
                    optimized = data['method_2_hybrid_example']['optimized_prompt']
                    score = data['method_2_hybrid_example']['validation_results']['validation_engine_score']
                    print(f"      Optimized: '{optimized}'")
                    print(f"      Score: {score:.4f}")
            
            print(f"\n🔍 STEP 2: Finding close gold prompt...")
            original_prompt = "intricate sandstone sculpture of cat lounging"
            print(f"   🎯 Looking for prompt similar to: '{original_prompt}'")
            
            # Find the closest gold prompt
            closest_match = repro_system.find_closest_gold_prompt(original_prompt, min_similarity=0.3)
            
            if closest_match:
                gold_prompt, gold_score, similarity = closest_match
                print(f"   🏆 Found close gold prompt (similarity: {similarity:.3f})")
                print(f"      Gold prompt: '{gold_prompt}'")
                print(f"      Gold score: {gold_score:.4f}")
                print(f"      Source: episodic_memory")
                
                print(f"\n🔄 STEP 3: Running reproducibility optimization...")
                print(f"   This is where the 'jump' happens!")
                print(f"   The system will now:")
                print(f"      a) Extract patterns from original prompt using LLM")
                print(f"      b) Extract patterns from gold prompt using LLM")
                print(f"      c) Merge components intelligently")
                print(f"      d) Reconstruct optimized prompt using LLM")
                
                # Mock the LLM responses for the complete flow
                mock_optimizer._query_llm.side_effect = [
                    # Response for extracting patterns from original prompt
                    '''
                    {
                        "core_subject": "sandstone cat sculpture",
                        "enhancements": {
                            "quality_adjectives": ["intricate"],
                            "material_details": ["sandstone"],
                            "light_interaction": [],
                            "context": []
                        }
                    }
                    ''',
                    # Response for extracting patterns from gold prompt
                    '''
                    {
                        "core_subject": "cylindrical glass",
                        "enhancements": {
                            "quality_adjectives": ["bubbly"],
                            "material_details": ["glass"],
                            "light_interaction": [],
                            "context": []
                        }
                    }
                    ''',
                    # Response for reconstructing final prompt
                    "cylindrical glass of bubbly lemonade, white background"
                ]
                
                print(f"\n   🚀 Calling optimize_prompt_with_reproducibility()...")
                result = repro_system.optimize_prompt_with_reproducibility(
                    original_prompt, min_similarity=0.3, run_validation=False
                )
                
                if result:
                    print(f"\n🎯 REPRODUCIBILITY OPTIMIZATION SUCCESS:")
                    print(f"   Original: '{result['original_prompt']}'")
                    print(f"   Optimized: '{result['optimized_prompt']}'")
                    print(f"   Similarity: {result['similarity']:.3f}")
                    print(f"   Gold score: {result['gold_score']:.4f}")
                    
                    print(f"\n💡 EXPLANATION OF THE 'JUMP':")
                    print(f"   The 'jump' from 'found close gold prompt' to 'SUCCESS' happens")
                    print(f"   because optimize_prompt_with_reproducibility() does 4 steps internally:")
                    print(f"")
                    print(f"   1. ✅ Find closest gold prompt (already done)")
                    print(f"   2. 🔍 Extract patterns from original prompt using LLM")
                    print(f"   3. 🔍 Extract patterns from gold prompt using LLM") 
                    print(f"   4. 🔄 Merge components intelligently")
                    print(f"   5. 🔧 Reconstruct optimized prompt using LLM")
                    print(f"")
                    print(f"   Each step uses LLM calls to:")
                    print(f"   - Parse prompts into structured components")
                    print(f"   - Merge high-scoring elements from gold prompts")
                    print(f"   - Reconstruct natural language prompts")
                    print(f"")
                    print(f"   This provides fast, deterministic optimization without")
                    print(f"   needing to run full reinforcement learning loops!")
                    
                    # Show the LLM calls that were made
                    print(f"\n🤖 LLM CALLS MADE DURING OPTIMIZATION:")
                    for i, call in enumerate(mock_optimizer._query_llm.call_args_list, 1):
                        system_prompt = call[0][0][:100] + "..." if len(call[0][0]) > 100 else call[0][0]
                        user_prompt = call[0][1]
                        print(f"   Call {i}:")
                        print(f"      System: {system_prompt}")
                        print(f"      User: '{user_prompt}'")
                        print(f"")
                    
                else:
                    print(f"   ❌ Optimization failed")
            else:
                print(f"   ❌ No close gold prompt found")
                
    finally:
        import shutil
        shutil.rmtree(temp_dir)

def test_individual_functions():
    """Test each function individually to understand how they work"""
    print("\n" + "=" * 70)
    print("🧪 TESTING INDIVIDUAL FUNCTIONS")
    print("=" * 70)
    
    temp_dir, test_file = create_test_episodic_memory()
    
    try:
        with patch('llm_close_prompt_reproducibility_test.LLMPromptOptimizer') as mock_optimizer_class:
            mock_optimizer = Mock()
            mock_optimizer._query_llm = Mock()
            mock_optimizer_class.return_value = mock_optimizer
            
            repro_system = LLMClosePromptReproducibility(
                episodic_memory_file=test_file,
                use_vllm=True,
                vllm_url="http://localhost:9000",
                vllm_model="llama-3-2-3b-it"
            )
            
            print("🔍 TEST 1: find_closest_gold_prompt()")
            original = "intricate sandstone sculpture of cat lounging"
            result = repro_system.find_closest_gold_prompt(original, min_similarity=0.3)
            if result:
                gold_prompt, gold_score, similarity = result
                print(f"   ✅ Found: '{gold_prompt}' (score: {gold_score:.4f}, sim: {similarity:.3f})")
            else:
                print(f"   ❌ No match found")
            
            print(f"\n🔍 TEST 2: extract_patterns()")
            mock_optimizer._query_llm.return_value = '''
            {
                "core_subject": "test subject",
                "enhancements": {
                    "quality_adjectives": ["test"],
                    "material_details": ["test"],
                    "light_interaction": [],
                    "context": []
                }
            }
            '''
            patterns = repro_system.extract_patterns("test prompt")
            print(f"   ✅ Extracted patterns: {patterns['core_subject']}")
            
            print(f"\n🔍 TEST 3: merge_components_intelligently()")
            original_comp = {
                "core_subject": "stone cat",
                "enhancements": {"quality_adjectives": ["rough"], "material_details": ["stone"], "light_interaction": [], "context": []}
            }
            gold_comp = {
                "core_subject": "glass object",
                "enhancements": {"quality_adjectives": ["shiny"], "material_details": ["glass"], "light_interaction": [], "context": []}
            }
            merged = repro_system.merge_components_intelligently(original_comp, gold_comp, similarity=0.6)
            print(f"   ✅ Merged core subject: {merged['core_subject']}")
            
            print(f"\n🔍 TEST 4: reconstruct_prompt()")
            mock_optimizer._query_llm.return_value = "reconstructed prompt, white background"
            reconstructed = repro_system.reconstruct_prompt(merged)
            print(f"   ✅ Reconstructed: '{reconstructed}'")
            
    finally:
        import shutil
        shutil.rmtree(temp_dir)

if __name__ == "__main__":
    # Run the step-by-step demonstration
    demonstrate_step_by_step()
    
    # Test individual functions
    test_individual_functions()
    
    print("\n" + "=" * 70)
    print("✅ DEMONSTRATION COMPLETE!")
    print("=" * 70)
    print("\n💡 KEY INSIGHTS:")
    print("   • The reproducibility system is NOT just finding similar prompts")
    print("   • It's doing intelligent pattern extraction and reconstruction using LLMs")
    print("   • The 'jump' happens because the system combines:")
    print("     - Episodic memory (gold prompts)")
    print("     - LLM intelligence (pattern extraction)")
    print("     - Component merging (intelligent combination)")
    print("     - Prompt reconstruction (LLM generation)")
    print("   • This creates a fast, deterministic optimization pipeline")
    print("   • Much faster than running full RL optimization loops!")
