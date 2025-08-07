#!/usr/bin/env python3
"""
Test script to check CLIP reproducibility system
Tests if the "plastic straw of drink" prompt can be found and optimized correctly
"""

import json
import os
import sys
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Optional, Any

# Add the current directory to the path
sys.path.append('/home/mbhat/three-gen-subnet-trellis')

def test_rl_memory_file():
    """Test if RL memory file exists and contains the target prompt"""
    print("🔍 Testing RL Memory File")
    print("=" * 50)
    
    file_path = 'episodic_clip_logs/rl_memory.json'
    target_prompt = "plastic straw of drink"
    
    if not os.path.exists(file_path):
        print(f"❌ File not found: {file_path}")
        return False, None
    
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        size = os.path.getsize(file_path)
        print(f"📁 File size: {size / (1024*1024):.2f} MB")
        
        # Check if this is RL memory format
        if 'optimization_sessions' in data:
            sessions = data.get('optimization_sessions', [])
            print(f"📊 RL Memory Format - Optimization sessions: {len(sessions)}")
            
            # Look for target prompt in sessions
            found_session = None
            for i, session in enumerate(sessions):
                original_prompt = session.get('original_prompt', '')
                if target_prompt == original_prompt:
                    found_session = session
                    print(f"✅ Found exact match in session {i + 1}: '{target_prompt}'")
                    print(f"   Final best score: {session.get('final_best_score', 'N/A')}")
                    print(f"   Final best prompt: '{session.get('final_best_prompt', 'N/A')}'")
                    print(f"   Total rounds: {session.get('total_rounds', 'N/A')}")
                    print(f"   Convergence: {session.get('convergence_achieved', 'N/A')}")
                    return True, found_session
            
            # If exact match not found, look for similar
            similar_sessions = []
            for i, session in enumerate(sessions):
                original_prompt = session.get('original_prompt', '')
                if 'plastic' in original_prompt.lower() and 'straw' in original_prompt.lower():
                    similar_sessions.append((i, session))
            
            if similar_sessions:
                print(f"🔍 Found {len(similar_sessions)} similar sessions:")
                for i, (session_idx, session) in enumerate(similar_sessions[:3]):
                    print(f"   {i+1}. Session {session_idx + 1}: '{session.get('original_prompt', 'N/A')}'")
                    print(f"      Score: {session.get('final_best_score', 'N/A')}")
            else:
                print(f"❌ No similar sessions found for: '{target_prompt}'")
                # Show sample prompts
                sample_prompts = [s.get('original_prompt', 'N/A') for s in sessions[:5]]
                print(f"📝 Sample prompts: {sample_prompts}")
            
            return bool(similar_sessions), similar_sessions
        
        else:
            print(f"📊 Classic format - Total entries: {len(data)}")
            # Original logic for classic format
            if target_prompt in data:
                print(f"✅ Found exact match: '{target_prompt}'")
                entry = data[target_prompt]
                print(f"   Best score: {entry.get('best_score', 'N/A')}")
                print(f"   Episodes run: {entry.get('episodes_run', 'N/A')}")
                print(f"   Best prompt: '{entry.get('best_prompt', 'N/A')}'")
                print(f"   Best generator: {entry.get('best_generator', 'N/A')}")
                return True, entry
            else:
                print(f"❌ Exact match not found: '{target_prompt}'")
                return False, None
            
    except Exception as e:
        print(f"❌ Error loading file: {e}")
        return False, None

def test_clip_episodic_memory_loading():
    """Test if the CLIP episodic memory loading works"""
    print("\n🧠 Testing CLIP Episodic Memory Loading")
    print("=" * 50)
    
    try:
        from continuous_trellis_orchestrator_simulator_lora import ContinuousTrellisSimulator
        
        # Create minimal config
        config = {
            'output_dir': './test_outputs',
            'episodic_memory_file': 'episodic_clip_logs/rl_memory.json',
            'enable_clip_optimization': True
        }
        
        # Initialize simulator
        simulator = ContinuousTrellisSimulator(config)
        
        print(f"✅ Simulator initialized successfully")
        print(f"📚 Episodic memory entries loaded: {len(simulator.episodic_memory)}")
        
        # Test exact match
        target_prompt = "plastic straw of drink"
        exact_match = simulator.find_exact_match_in_memory(target_prompt)
        
        if exact_match:
            print(f"✅ Exact match found in memory:")
            print(f"   Best prompt: '{exact_match.get('best_prompt', 'N/A')}'")
            print(f"   Best score: {exact_match.get('best_score', 'N/A')}")
            print(f"   Best generator: {exact_match.get('best_generator', 'N/A')}")
        else:
            print(f"❌ Exact match not found in memory")
            
            # Test similar match
            similar_match = simulator.find_similar_prompt_in_memory(target_prompt, min_similarity=0.3)
            if similar_match:
                print(f"🔍 Similar match found:")
                print(f"   Similar prompt: '{similar_match.get('prompt', 'N/A')}'")
                print(f"   Similarity: {similar_match.get('similarity', 'N/A'):.3f}")
                print(f"   Best prompt: '{similar_match.get('best_prompt', 'N/A')}'")
                print(f"   Best score: {similar_match.get('best_score', 'N/A')}")
            else:
                print(f"❌ No similar match found either")
        
        return True, simulator
        
    except Exception as e:
        print(f"❌ Error testing memory loading: {e}")
        import traceback
        traceback.print_exc()
        return False, None

def test_advanced_clip_optimization():
    """Test the advanced CLIP optimization system"""
    print("\n🎯 Testing Advanced CLIP Optimization")
    print("=" * 50)
    
    try:
        from continuous_trellis_orchestrator_simulator_lora import ContinuousTrellisSimulator
        
        config = {
            'output_dir': './test_outputs',
            'episodic_memory_file': 'episodic_clip_logs/rl_memory.json',
            'enable_clip_optimization': True,
            'clip_similarity_threshold': 0.51
        }
        
        simulator = ContinuousTrellisSimulator(config)
        target_prompt = "plastic straw of drink"
        
        # Test the 3-tier optimization system
        result = simulator.advanced_clip_optimization(target_prompt, "")
        
        print(f"✅ CLIP optimization completed:")
        print(f"   Method used: {result.get('method', 'N/A')}")
        print(f"   Optimized prompt: '{result.get('optimized_prompt', 'N/A')}'")
        print(f"   Similarity score: {result.get('similarity_score', 'N/A')}")
        
        if 'clip_best_score' in result:
            print(f"   Historical CLIP score: {result.get('clip_best_score', 'N/A')}")
            print(f"   Historical generator: {result.get('clip_best_generator', 'N/A')}")
        
        return True, result
        
    except Exception as e:
        print(f"❌ Error testing CLIP optimization: {e}")
        import traceback
        traceback.print_exc()
        return False, None

def test_cosine_similarity():
    """Test the cosine similarity calculation"""
    print("\n📐 Testing Cosine Similarity")
    print("=" * 50)
    
    try:
        from continuous_trellis_orchestrator_simulator_lora import ContinuousTrellisSimulator
        
        config = {'output_dir': './test_outputs'}
        simulator = ContinuousTrellisSimulator(config)
        
        target_prompt = "plastic straw of drink"
        test_prompts = [
            "plastic straw of drink",  # Exact match
            "plastic drinking straw",  # Very similar
            "plastic straw for drinks", # Similar
            "plastic cup with straw",  # Somewhat similar
            "wooden spoon for cooking", # Different
        ]
        
        print(f"Target: '{target_prompt}'")
        print("Similarity scores:")
        
        for prompt in test_prompts:
            similarity = simulator.calculate_cosine_similarity(target_prompt, prompt)
            print(f"   '{prompt}' → {similarity:.3f}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error testing similarity: {e}")
        return False

def test_top_scoring_patterns():
    """Test the extraction of top scoring patterns from attempt history"""
    print("\n🏆 Testing Top Scoring Patterns Extraction")
    print("=" * 50)
    
    try:
        from continuous_trellis_orchestrator_simulator_lora import ContinuousTrellisSimulator
        
        config = {'output_dir': './test_outputs'}
        simulator = ContinuousTrellisSimulator(config)
        
        # Get the plastic straw prompt from episodic memory
        target_prompt = "plastic straw of drink"
        memory_entry = simulator.episodic_memory.get(target_prompt)
        
        if memory_entry:
            if hasattr(memory_entry, 'attempt_history'):
                attempt_history = memory_entry.attempt_history
            else:
                attempt_history = memory_entry.get('attempt_history', [])
            
            print(f"📊 Found {len(attempt_history)} attempts in memory")
            
            # Extract top patterns
            top_patterns = simulator.extract_top_scoring_patterns(attempt_history)
            
            if top_patterns:
                print(f"✅ Successfully extracted {len(top_patterns)} top patterns:")
                for pattern in top_patterns:
                    print(f"   {pattern['rank']}. Score: {pattern['score']:.4f} ({pattern['score_field']})")
                    print(f"      Strategy: {pattern['strategy']}")
                    print(f"      Prompt: '{pattern['prompt'][:80]}...'")
                    print(f"      Attempt #: {pattern['attempt_number']}")
                    print()
                return True, top_patterns
            else:
                print("❌ No top patterns extracted")
                return False, None
        else:
            print(f"❌ Target prompt not found in memory: '{target_prompt}'")
            return False, None
        
    except Exception as e:
        print(f"❌ Error testing pattern extraction: {e}")
        import traceback
        traceback.print_exc()
        return False, None

def main():
    """Main test function"""
    print("🧪 CLIP Reproducibility System Test")
    print("=" * 60)
    print()
    
    # Test 1: RL Memory File
    file_success, file_data = test_rl_memory_file()
    
    # Test 2: Memory Loading
    memory_success, simulator = test_clip_episodic_memory_loading()
    
    # Test 3: CLIP Optimization
    if memory_success and simulator:
        optimization_success, opt_result = test_advanced_clip_optimization()
    else:
        optimization_success = False
        opt_result = None
    
    # Test 4: Cosine Similarity
    similarity_success = test_cosine_similarity()
    
    # Test 5: Top Scoring Patterns
    patterns_success, patterns_result = test_top_scoring_patterns()
    
    # Summary
    print("\n📋 TEST SUMMARY")
    print("=" * 60)
    print(f"✅ RL Memory File: {'PASS' if file_success else 'FAIL'}")
    print(f"✅ Memory Loading: {'PASS' if memory_success else 'FAIL'}")
    print(f"✅ CLIP Optimization: {'PASS' if optimization_success else 'FAIL'}")
    print(f"✅ Cosine Similarity: {'PASS' if similarity_success else 'FAIL'}")
    print(f"✅ Top Scoring Patterns: {'PASS' if patterns_success else 'FAIL'}")
    
    if all([file_success, memory_success, optimization_success, similarity_success, patterns_success]):
        print("\n🎉 ALL TESTS PASSED!")
        return 0
    else:
        print("\n❌ SOME TESTS FAILED!")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code) 