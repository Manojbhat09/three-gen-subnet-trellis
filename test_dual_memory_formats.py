#!/usr/bin/env python3
"""
Test script for dual memory format support in continuous_trellis_orchestrator_simulator_lora.py
Tests both RL memory format and episodic run log format loading and parsing.
"""

import sys
import os
sys.path.append('/home/mbhat/three-gen-subnet-trellis')

from continuous_trellis_orchestrator_simulator_lora import ContinuousTrellisSimulator
import json

def test_rl_memory_format():
    """Test RL memory format loading"""
    print("🧪 Testing RL Memory Format")
    print("=" * 50)
    
    # Test with RL memory file
    config = {
        'episodic_memory_file': 'episodic_clip_logs/rl_memory.json',
        'episodic_run_log_file': None,
        'enable_clip_optimization': True
    }
    
    simulator = ContinuousTrellisSimulator(config)
    
    print(f"📊 RL Memory loaded: {len(simulator.episodic_memory)} prompts")
    
    if simulator.episodic_memory:
        sample_prompts = list(simulator.episodic_memory.keys())[:3]
        print(f"📝 Sample prompts: {sample_prompts}")
        
        # Test exact match
        for prompt in sample_prompts[:1]:
            exact_match = simulator.find_exact_match_in_memory(prompt)
            if exact_match:
                print(f"\n✅ Exact match test for '{prompt[:30]}...':")
                print(f"   Best score: {exact_match['best_score']:.4f}")
                print(f"   Best prompt: '{exact_match['best_prompt'][:50]}...'")
                print(f"   Method: {exact_match['method']}")
    
    return simulator

def test_episodic_run_log_format():
    """Test episodic run log format loading"""
    print("\n🧪 Testing Episodic Run Log Format")
    print("=" * 50)
    
    # Test with run log file
    config = {
        'episodic_memory_file': 'episodic_clip_logs/rl_memory.json',
        'episodic_run_log_file': 'episodic_clip_logs/multi_generator_results_run1.json',
        'enable_clip_optimization': True
    }
    
    simulator = ContinuousTrellisSimulator(config)
    
    print(f"📊 Run Log Memory loaded: {len(simulator.episodic_memory)} prompts")
    
    if simulator.episodic_memory:
        sample_prompts = list(simulator.episodic_memory.keys())[:3]
        print(f"📝 Sample prompts: {sample_prompts}")
        
        # Test the specific ukulele prompt
        ukulele_prompt = "ukulele sporting vibrant sunflower yellow"
        if ukulele_prompt in simulator.episodic_memory:
            print(f"\n✅ Found ukulele prompt in run log memory!")
            
            exact_match = simulator.find_exact_match_in_memory(ukulele_prompt)
            if exact_match:
                print(f"   Best score: {exact_match['best_score']:.4f}")
                print(f"   Best prompt: '{exact_match['best_prompt'][:80]}...'")
                print(f"   Best generator: {exact_match['best_generator']}")
                print(f"   Episodes run: {exact_match['episodes_run']}")
            
            # Test pattern extraction
            memory_entry = simulator.episodic_memory[ukulele_prompt]
            if hasattr(memory_entry, 'attempt_history'):
                attempt_history = memory_entry.attempt_history
            else:
                attempt_history = memory_entry.get('attempt_history', [])
            
            if attempt_history:
                print(f"\n📊 Testing pattern extraction for ukulele prompt:")
                print(f"   Attempt history length: {len(attempt_history)}")
                
                top_patterns = simulator.extract_top_scoring_patterns(attempt_history)
                print(f"   Extracted {len(top_patterns)} top patterns:")
                
                for pattern in top_patterns:
                    print(f"     Rank {pattern['rank']}: Score {pattern['score']:.4f}")
                    print(f"       Strategy: {pattern['strategy']}")
                    print(f"       Generator: {pattern['generator']}")
                    print(f"       Episode: {pattern['attempt_number']}")
                    print(f"       Prompt: '{pattern['prompt'][:60]}...'")
                
                # Test pattern analysis formatting
                pattern_analysis = simulator.format_pattern_analysis(top_patterns)
                print(f"\n📝 Formatted pattern analysis:")
                print(pattern_analysis[:500] + "..." if len(pattern_analysis) > 500 else pattern_analysis)
        else:
            print(f"❌ Ukulele prompt not found in run log memory")
    
    return simulator

def test_clip_optimization():
    """Test CLIP optimization with run log data"""
    print("\n🧪 Testing CLIP Optimization with Run Log Data")
    print("=" * 50)
    
    # Test with run log file
    config = {
        'episodic_memory_file': 'episodic_clip_logs/rl_memory.json',
        'episodic_run_log_file': 'episodic_clip_logs/multi_generator_results_run1.json',
        'enable_clip_optimization': True,
        'clip_similarity_threshold': 0.51
    }
    
    simulator = ContinuousTrellisSimulator(config)
    
    # Test with the ukulele prompt
    test_prompt = "ukulele sporting vibrant sunflower yellow"
    
    print(f"🎯 Testing CLIP optimization for: '{test_prompt}'")
    
    # Test exact match (Tier 1)
    exact_match = simulator.find_exact_match_in_memory(test_prompt)
    if exact_match:
        print(f"✅ Tier 1 - Exact match found!")
        print(f"   Score: {exact_match['best_score']:.4f}")
        print(f"   Generator: {exact_match['best_generator']}")
    
    # Test similar prompt search (Tier 2)
    similar_match = simulator.find_similar_prompt_in_memory("yellow ukulele instrument", min_similarity=0.3)
    if similar_match:
        print(f"✅ Tier 2 - Similar prompt found!")
        print(f"   Similar prompt: '{similar_match['prompt']}'")
        print(f"   Similarity: {similar_match['similarity']:.3f}")
        print(f"   Score: {similar_match['best_score']:.4f}")
    
    # Test full advanced optimization
    print(f"\n🧠 Running full advanced CLIP optimization...")
    clip_result = simulator.advanced_clip_optimization(test_prompt, "")
    
    print(f"✅ CLIP optimization completed:")
    print(f"   Method: {clip_result.get('method', 'unknown')}")
    print(f"   Optimized prompt: '{clip_result.get('optimized_prompt', 'N/A')[:80]}...'")
    print(f"   Similarity score: {clip_result.get('similarity_score', 0.0):.3f}")
    if 'clip_best_score' in clip_result:
        print(f"   Historical CLIP score: {clip_result['clip_best_score']:.4f}")

def main():
    """Main test function"""
    print("🚀 Testing Dual Memory Format Support")
    print("=" * 70)
    
    try:
        # Test RL memory format
        rl_simulator = test_rl_memory_format()
        
        # Test episodic run log format
        run_log_simulator = test_episodic_run_log_format()
        
        # Test CLIP optimization
        test_clip_optimization()
        
        print("\n✅ All tests completed successfully!")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 