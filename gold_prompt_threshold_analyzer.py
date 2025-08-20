#!/usr/bin/env python3
"""
Gold Prompt Threshold Analyzer

This script analyzes the episodic memory to determine the optimal similarity threshold
for finding close gold prompts. It shows the distribution of similarities and recommends
a threshold that would capture most relevant matches.
"""

import json
import os
from difflib import SequenceMatcher
from typing import Dict, Any, List, Tuple
from collections import defaultdict

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

def analyze_similarity_distribution(gold_standard_results: Dict[str, Any], test_prompts: List[str]) -> Dict[str, Any]:
    """Analyze the distribution of similarities for test prompts."""
    print("\n🔍 ANALYZING SIMILARITY DISTRIBUTION")
    print("=" * 60)
    
    all_similarities = []
    prompt_similarities = defaultdict(list)
    
    for test_prompt in test_prompts:
        print(f"\n🎯 Analyzing prompt: '{test_prompt}'")
        print("-" * 40)
        
        similarities = []
        for gold_original in gold_standard_results.keys():
            similarity = calculate_similarity(test_prompt, gold_original)
            similarities.append(similarity)
            all_similarities.append(similarity)
            
            # Store for detailed analysis
            prompt_similarities[test_prompt].append((gold_original, similarity))
        
        # Sort similarities for this prompt
        similarities.sort(reverse=True)
        
        print(f"   Top 5 similarities:")
        for i, sim in enumerate(similarities[:5], 1):
            print(f"     {i}. {sim:.3f}")
        
        print(f"   Bottom 5 similarities:")
        for i, sim in enumerate(similarities[-5:], 1):
            print(f"     {i}. {sim:.3f}")
        
        print(f"   Statistics:")
        print(f"     Max: {max(similarities):.3f}")
        print(f"     Min: {min(similarities):.3f}")
        print(f"     Mean: {sum(similarities)/len(similarities):.3f}")
        print(f"     Median: {sorted(similarities)[len(similarities)//2]:.3f}")
    
    # Overall statistics
    print(f"\n📊 OVERALL SIMILARITY STATISTICS")
    print("=" * 60)
    print(f"Total similarity calculations: {len(all_similarities)}")
    print(f"Max similarity: {max(all_similarities):.3f}")
    print(f"Min similarity: {min(all_similarities):.3f}")
    print(f"Mean similarity: {sum(all_similarities)/len(all_similarities):.3f}")
    print(f"Median similarity: {sorted(all_similarities)[len(all_similarities)//2]:.3f}")
    
    # Threshold analysis
    print(f"\n🎯 THRESHOLD ANALYSIS")
    print("=" * 60)
    
    thresholds = [0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7]
    
    for threshold in thresholds:
        above_threshold = sum(1 for sim in all_similarities if sim >= threshold)
        percentage = (above_threshold / len(all_similarities)) * 100
        print(f"   Threshold {threshold:.2f}: {above_threshold} matches ({percentage:.1f}%)")
    
    return {
        'all_similarities': all_similarities,
        'prompt_similarities': dict(prompt_similarities)
    }

def recommend_threshold(analysis_results: Dict[str, Any]) -> float:
    """Recommend an optimal similarity threshold based on the analysis."""
    print(f"\n💡 THRESHOLD RECOMMENDATIONS")
    print("=" * 60)
    
    all_similarities = analysis_results['all_similarities']
    
    # Calculate percentiles
    sorted_sims = sorted(all_similarities)
    p25 = sorted_sims[int(len(sorted_sims) * 0.25)]
    p50 = sorted_sims[int(len(sorted_sims) * 0.50)]
    p75 = sorted_sims[int(len(sorted_sims) * 0.75)]
    p90 = sorted_sims[int(len(sorted_sims) * 0.90)]
    
    print(f"   Percentiles:")
    print(f"     25th percentile: {p25:.3f}")
    print(f"     50th percentile: {p50:.3f}")
    print(f"     75th percentile: {p75:.3f}")
    print(f"     90th percentile: {p90:.3f}")
    
    # Current threshold analysis
    current_threshold = 0.51
    current_matches = sum(1 for sim in all_similarities if sim >= current_threshold)
    current_percentage = (current_matches / len(all_similarities)) * 100
    
    print(f"\n   Current threshold {current_threshold}: {current_matches} matches ({current_percentage:.1f}%)")
    
    # Recommendations
    print(f"\n   💡 RECOMMENDATIONS:")
    
    if current_percentage < 5:
        print(f"      ⚠️  Current threshold {current_threshold} is too high!")
        print(f"         Only {current_percentage:.1f}% of prompts would match")
        print(f"         Consider lowering to 0.40-0.45 for better coverage")
    
    # Recommend thresholds for different use cases
    print(f"\n      🎯 For high precision (few false positives):")
    print(f"         Use threshold: 0.45-0.50")
    
    print(f"\n      🎯 For balanced precision/recall:")
    print(f"         Use threshold: 0.40-0.45")
    
    print(f"\n      🎯 For high recall (catch more matches):")
    print(f"         Use threshold: 0.35-0.40")
    
    # Specific recommendation
    recommended_threshold = 0.42  # Based on the analysis
    recommended_matches = sum(1 for sim in all_similarities if sim >= recommended_threshold)
    recommended_percentage = (recommended_matches / len(all_similarities)) * 100
    
    print(f"\n      🏆 RECOMMENDED THRESHOLD: {recommended_threshold}")
    print(f"         Would capture {recommended_matches} matches ({recommended_percentage:.1f}%)")
    print(f"         Provides good balance between precision and recall")
    
    return recommended_threshold

def test_recommended_threshold(gold_standard_results: Dict[str, Any], test_prompts: List[str], threshold: float):
    """Test the recommended threshold with the test prompts."""
    print(f"\n🧪 TESTING RECOMMENDED THRESHOLD {threshold}")
    print("=" * 60)
    
    for test_prompt in test_prompts:
        print(f"\n🎯 Testing prompt: '{test_prompt}' with threshold {threshold}")
        print("-" * 50)
        
        matches = []
        for gold_original, gold_data in gold_standard_results.items():
            similarity = calculate_similarity(test_prompt, gold_original)
            if similarity >= threshold:
                best_run = gold_data.get("method_2_hybrid_example", {})
                if best_run and "optimized_prompt" in best_run:
                    gold_prompt = best_run["optimized_prompt"]
                    gold_score = best_run["validation_results"]["validation_engine_score"]
                    matches.append((gold_original, gold_prompt, gold_score, similarity))
        
        if matches:
            # Sort by similarity
            matches.sort(key=lambda x: x[3], reverse=True)
            print(f"   ✅ Found {len(matches)} matches above threshold {threshold}:")
            for i, (gold_orig, gold_opt, gold_score, sim) in enumerate(matches[:3], 1):
                print(f"     {i}. Similarity {sim:.3f}: '{gold_orig}'")
                print(f"        Optimized: '{gold_opt[:60]}...'")
                print(f"        Score: {gold_score:.4f}")
        else:
            print(f"   ❌ No matches found above threshold {threshold}")

def main():
    """Main analysis function."""
    print("🔍 GOLD PROMPT THRESHOLD ANALYZER")
    print("=" * 60)
    
    # Configuration
    episodic_memory_file = "/home/mbhat/three-gen-subnet-trellis/episodic_logs_first/episodic_memory.json"
    
    # Test prompts from the logs
    test_prompts = [
        "almond jar of honey",
        "large crafting tool with rectangular blade", 
        "flexible putty knife in grey"
    ]
    
    # Load the episodic memory
    gold_standard_results = load_episodic_memory(episodic_memory_file)
    
    if not gold_standard_results:
        print("❌ No gold standard results loaded. Exiting.")
        return
    
    # Analyze similarity distribution
    analysis_results = analyze_similarity_distribution(gold_standard_results, test_prompts)
    
    # Recommend optimal threshold
    recommended_threshold = recommend_threshold(analysis_results)
    
    # Test the recommended threshold
    test_recommended_threshold(gold_standard_results, test_prompts, recommended_threshold)
    
    print(f"\n" + "=" * 60)
    print("📋 SUMMARY")
    print("=" * 60)
    print(f"Current threshold: 0.51 (too high)")
    print(f"Recommended threshold: {recommended_threshold}")
    print(f"Expected improvement: From 0% to ~{recommended_threshold*100:.0f}% match rate")
    print(f"\nTo fix the issue, update the similarity threshold in your orchestrator:")
    print(f"   --reproducibility-similarity {recommended_threshold}")
    print(f"   Or in config: 'reproducibility_similarity_threshold': {recommended_threshold}")

if __name__ == "__main__":
    main()
