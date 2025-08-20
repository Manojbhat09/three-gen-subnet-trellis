#!/usr/bin/env python3
"""
Gold Prompt Diagnostic Script

This script helps diagnose why no close gold prompts are being found in the episodic memory.
It checks the data structure, loads the memory, and tests similarity calculations.
"""

import json
import os
from difflib import SequenceMatcher
from typing import Dict, Any, List, Tuple, Optional

def calculate_similarity(prompt1: str, prompt2: str) -> float:
    """Calculate similarity between two prompts using sequence matching."""
    return SequenceMatcher(None, prompt1.lower(), prompt2.lower()).ratio()

def extract_true_prompt(prompt: str) -> str:
    """Extract the true prompt from potentially formatted text."""
    if not prompt:
        return ""
    
    # Remove common prefixes and suffixes
    prompt = prompt.strip()
    
    # Remove markdown code blocks
    if prompt.startswith('`') and prompt.endswith('`'):
        prompt = prompt[1:-1]
    
    # Remove "wbgmsst," prefix if present
    if prompt.lower().startswith('wbgmsst,'):
        prompt = prompt[8:].lstrip()
    
    # Remove ", white background" suffix if present
    if prompt.lower().endswith(', white background'):
        prompt = prompt[:-18].rstrip()
    
    return prompt.strip()

def load_episodic_memory(episodic_memory_file: str) -> Dict[str, Any]:
    """Load episodic memory and extract gold standard results."""
    if not os.path.exists(episodic_memory_file):
        print(f"❌ Error: Episodic memory file not found at '{episodic_memory_file}'.")
        return {}
    
    try:
        with open(episodic_memory_file, 'r') as f:
            episodic_data = json.load(f)
        
        print(f"📚 Loaded episodic memory file: {os.path.getsize(episodic_memory_file)} bytes")
        
        # Handle both old format (dict) and new format (list of sessions)
        if isinstance(episodic_data, dict):
            # Old format - single session
            print(f"📚 Detected old single-session memory format")
            optimization_sessions = episodic_data.get("optimization_sessions", [])
        elif isinstance(episodic_data, list):
            # New format - multiple sessions
            print(f"📚 Detected new multi-session memory format ({len(episodic_data)} sessions)")
            # Combine optimization sessions from all sessions
            optimization_sessions = []
            for session_data in episodic_data:
                if isinstance(session_data, dict):
                    session_sessions = session_data.get("optimization_sessions", [])
                    optimization_sessions.extend(session_sessions)
                    print(f"   📖 Session {session_data.get('session_id', 'unknown')}: {len(session_sessions)} optimization sessions")
        else:
            print(f"❌ Error: Unknown episodic memory format: {type(episodic_data)}")
            return {}
        
        print(f"📊 Total optimization sessions found: {len(optimization_sessions)}")
        
        # Convert to gold standard format
        gold_standard_results = {}
        
        for session in optimization_sessions:
            original_prompt = extract_true_prompt(session.get("original_prompt", ""))
            if not original_prompt:
                continue
            
            # Find the best attempt in this session
            best_attempt = None
            best_score = 0.0
            
            for attempt in session.get("attempts", []):
                validation_score = attempt.get("validation_score")
                if validation_score is not None and validation_score > best_score:
                    best_score = validation_score
                    best_attempt = attempt
            
            if best_attempt:
                # Store in format similar to experiment_results.json
                if original_prompt not in gold_standard_results:
                    gold_standard_results[original_prompt] = {}
                
                # Use method_2_hybrid_example format for consistency
                gold_standard_results[original_prompt]["method_2_hybrid_example"] = {
                    "optimized_prompt": best_attempt["optimized_prompt"],
                    "validation_results": {
                        "validation_engine_score": best_score
                    }
                }
        
        print(f"✅ Loaded {len(gold_standard_results)} gold standard prompts from episodic memory")
        return gold_standard_results
        
    except (json.JSONDecodeError, ValueError) as e:
        print(f"❌ Error parsing episodic memory file '{episodic_memory_file}': {e}")
        return {}

def find_closest_gold_prompt(gold_standard_results: Dict[str, Any], original_prompt: str, min_similarity: float = 0.51) -> Optional[Tuple[str, str, float, float]]:
    """
    Find the closest gold prompt to the original prompt.
    Returns (gold_original, gold_optimized, gold_score, similarity) or None if no close match found.
    """
    best_match = None
    best_similarity = 0.0
    
    print(f"🔍 Searching through {len(gold_standard_results)} gold prompts...")
    print(f"🎯 Target prompt: '{original_prompt}'")
    print(f"📏 Minimum similarity threshold: {min_similarity}")
    
    # Show some sample gold prompts for debugging
    sample_prompts = list(gold_standard_results.keys())[:5]
    print(f"📝 Sample gold prompts:")
    for i, prompt in enumerate(sample_prompts, 1):
        print(f"   {i}. '{prompt}'")
    
    print("\n🔍 Calculating similarities...")
    
    for gold_original, gold_data in gold_standard_results.items():
        # Calculate similarity between original prompts
        similarity = calculate_similarity(original_prompt, gold_original)
        
        if similarity > best_similarity:
            best_similarity = similarity
            best_run = gold_data.get("method_2_hybrid_example", {})
            if best_run and "optimized_prompt" in best_run:
                gold_prompt = best_run["optimized_prompt"]
                gold_score = best_run["validation_results"]["validation_engine_score"]
                best_match = (gold_original, gold_prompt, gold_score, similarity)
        
        # Show top 5 similarities for debugging
        if similarity > 0.3:  # Show any reasonable similarity
            print(f"   Similarity {similarity:.3f}: '{gold_original}'")
    
    if best_match and best_match[3] >= min_similarity:
        print(f"\n🏆 Best match found:")
        print(f"   Original: '{best_match[0]}'")
        print(f"   Optimized: '{best_match[1]}'")
        print(f"   Score: {best_match[2]:.4f}")
        print(f"   Similarity: {best_match[3]:.3f}")
        return best_match
    else:
        print(f"\n⚠️ No close match found above threshold {min_similarity}")
        if best_match:
            print(f"   Best similarity was {best_match[3]:.3f} for: '{best_match[0]}'")
        return None

def test_similarity_calculations():
    """Test the similarity calculation function with known examples."""
    print("\n🧪 Testing similarity calculations...")
    
    test_cases = [
        ("almond jar of honey", "honey jar with almonds"),
        ("large crafting tool with rectangular blade", "crafting tool with blade"),
        ("flexible putty knife in grey", "putty knife grey"),
        ("cupcake with chocolate icing", "chocolate cupcake with frosting"),
        ("necklace with heart pendant", "heart pendant necklace")
    ]
    
    for prompt1, prompt2 in test_cases:
        similarity = calculate_similarity(prompt1, prompt2)
        print(f"   '{prompt1}' vs '{prompt2}': {similarity:.3f}")

def main():
    """Main diagnostic function."""
    print("🔍 GOLD PROMPT DIAGNOSTIC SCRIPT")
    print("=" * 50)
    
    # Configuration
    episodic_memory_file = "/home/mbhat/three-gen-subnet-trellis/episodic_logs_first/episodic_memory.json"
    
    # Test similarity calculations first
    test_similarity_calculations()
    
    print("\n" + "=" * 50)
    print("📚 LOADING EPISODIC MEMORY")
    print("=" * 50)
    
    # Load the episodic memory
    gold_standard_results = load_episodic_memory(episodic_memory_file)
    
    if not gold_standard_results:
        print("❌ No gold standard results loaded. Exiting.")
        return
    
    print("\n" + "=" * 50)
    print("🔍 TESTING SIMILARITY SEARCH")
    print("=" * 50)
    
    # Test with the prompts from the logs
    test_prompts = [
        "almond jar of honey",
        "large crafting tool with rectangular blade", 
        "flexible putty knife in grey"
    ]
    
    for test_prompt in test_prompts:
        print(f"\n🎯 Testing prompt: '{test_prompt}'")
        print("-" * 40)
        
        result = find_closest_gold_prompt(gold_standard_results, test_prompt, min_similarity=0.51)
        
        if result:
            gold_original, gold_optimized, gold_score, similarity = result
            print(f"✅ SUCCESS: Found close prompt with similarity {similarity:.3f}")
        else:
            print(f"❌ FAILURE: No close prompt found above threshold 0.51")
    
    print("\n" + "=" * 50)
    print("📊 SUMMARY")
    print("=" * 50)
    print(f"Total gold prompts available: {len(gold_standard_results)}")
    print(f"Similarity threshold used: 0.51")
    print(f"Similarity calculation method: SequenceMatcher (difflib)")
    
    # Show some statistics about the gold prompts
    if gold_standard_results:
        prompt_lengths = [len(prompt) for prompt in gold_standard_results.keys()]
        avg_length = sum(prompt_lengths) / len(prompt_lengths)
        print(f"Average prompt length: {avg_length:.1f} characters")
        print(f"Shortest prompt: {min(prompt_lengths)} characters")
        print(f"Longest prompt: {max(prompt_lengths)} characters")

if __name__ == "__main__":
    main()
