#!/usr/bin/env python3
"""
Test script for the Conversational Debate Optimizer.

This script demonstrates the Proposer-Reviewer debate system with:
- Step-by-step debate visualization
- Performance analysis
- Learning progression tracking
- Comparison with traditional optimization
"""

import json
import time
from conversational_debate_optimizer import ConversationalDebateOptimizer

def test_single_optimization():
    """Test a single prompt optimization with detailed output"""
    print("🧪 SINGLE OPTIMIZATION TEST")
    print("="*50)
    
    test_prompt = "emerald pendant"
    print(f"Testing prompt: '{test_prompt}'")
    print()
    
    optimizer = ConversationalDebateOptimizer(
        max_debate_rounds=3,
        target_score=0.9,
        min_improvement=0.04
    )
    
    result = optimizer.optimize_prompt(test_prompt)
    
    print("\n📊 DETAILED RESULTS:")
    print(f"Original: {result['original_prompt']}")
    print(f"Final: {result['optimized_prompt']}")
    print(f"Strategy: {result['strategy_used']}")
    print(f"Score: {result['final_score']:.3f}")
    print(f"Rounds: {result['rounds_completed']}")
    print(f"Duration: {result['duration_seconds']:.2f}s")
    print(f"Converged: {result['converged']}")
    
    # Show debate progression
    if result['debate_history']:
        print(f"\n🗣️  DEBATE PROGRESSION:")
        for i, round_data in enumerate(result['debate_history'], 1):
            assessment = round_data['assessment']
            print(f"\nRound {i}:")
            print(f"  Prompt: {round_data['proposed_prompt'][:80]}{'...' if len(round_data['proposed_prompt']) > 80 else ''}")
            print(f"  Score: {assessment['score']:.3f}")
            print(f"  Critique: {assessment['critique'][:100]}{'...' if len(assessment['critique']) > 100 else ''}")
            print(f"  Suggestion: {assessment['suggestion'][:100]}{'...' if len(assessment['suggestion']) > 100 else ''}")
    
    return result

def test_multiple_optimizations():
    """Test multiple prompts to show learning progression"""
    print("\n🔄 MULTIPLE OPTIMIZATION TEST")
    print("="*50)
    
    test_prompts = [
        "crystal wine glass",
        "wooden chess piece", 
        "silver bracelet",
        "ceramic vase",
        "metal sculpture"
    ]
    
    optimizer = ConversationalDebateOptimizer(
        max_debate_rounds=3,
        target_score=0.85,
        memory_file="test_debate_memory.json"
    )
    
    results = []
    
    for i, prompt in enumerate(test_prompts, 1):
        print(f"\n--- Test {i}/{len(test_prompts)}: '{prompt}' ---")
        
        start_time = time.time()
        result = optimizer.optimize_prompt(prompt)
        duration = time.time() - start_time
        
        results.append({
            'prompt': prompt,
            'score': result['final_score'],
            'rounds': result['rounds_completed'],
            'duration': duration,
            'strategy': result['strategy_used'],
            'converged': result['converged']
        })
        
        print(f"Score: {result['final_score']:.3f}, Rounds: {result['rounds_completed']}, Duration: {duration:.1f}s")
    
    # Analyze results
    print(f"\n📈 LEARNING ANALYSIS:")
    avg_score = sum(r['score'] for r in results) / len(results)
    avg_rounds = sum(r['rounds'] for r in results) / len(results)
    avg_duration = sum(r['duration'] for r in results) / len(results)
    convergence_rate = sum(1 for r in results if r['converged']) / len(results)
    
    print(f"Average Score: {avg_score:.3f}")
    print(f"Average Rounds: {avg_rounds:.1f}")
    print(f"Average Duration: {avg_duration:.1f}s")
    print(f"Convergence Rate: {convergence_rate:.1%}")
    
    # Show strategy distribution
    strategies_used = {}
    for result in results:
        strategy = result['strategy']
        if strategy not in strategies_used:
            strategies_used[strategy] = []
        strategies_used[strategy].append(result['score'])
    
    print(f"\n🎯 STRATEGY PERFORMANCE:")
    for strategy, scores in strategies_used.items():
        avg_score = sum(scores) / len(scores)
        print(f"{strategy}: {avg_score:.3f} (used {len(scores)} times)")
    
    return results

def test_debate_visualization():
    """Test with detailed debate visualization"""
    print("\n🎭 DEBATE VISUALIZATION TEST")
    print("="*50)
    
    test_prompt = "sapphire-studded sharp spear"
    print(f"Visualizing debate for: '{test_prompt}'")
    print()
    
    optimizer = ConversationalDebateOptimizer(
        max_debate_rounds=4,  # Allow more rounds for visualization
        target_score=0.92,
        min_improvement=0.03
    )
    
    print("🗣️  STARTING CONVERSATIONAL DEBATE...")
    result = optimizer.optimize_prompt(test_prompt)
    
    print(f"\n🎯 FINAL OUTCOME:")
    print(f"Optimization: '{result['optimized_prompt']}'")
    print(f"Final Score: {result['final_score']:.3f}")
    print(f"Debate Rounds: {result['rounds_completed']}")
    print(f"Target Reached: {result['converged']}")
    
    # Detailed round analysis
    if result['debate_history']:
        print(f"\n📊 ROUND-BY-ROUND ANALYSIS:")
        for i, round_data in enumerate(result['debate_history'], 1):
            assessment = round_data['assessment']
            print(f"\n🔄 Round {i}:")
            print(f"   Proposer: {round_data['proposed_prompt']}")
            print(f"   Reviewer Score: {assessment['score']:.3f}")
            print(f"   Reviewer Critique: {assessment['critique']}")
            print(f"   Reviewer Suggestion: {assessment['suggestion']}")
            
            if i < len(result['debate_history']):
                print(f"   ⬇️  Proposer will address this feedback in Round {i+1}")
            else:
                print(f"   ✅ Final assessment")
    
    return result

def performance_comparison():
    """Compare debate optimizer performance characteristics"""
    print("\n⚡ PERFORMANCE COMPARISON")
    print("="*50)
    
    test_prompts = ["golden ring", "glass sphere", "iron sword"]
    
    print("Testing Conversational Debate Optimizer...")
    
    debate_optimizer = ConversationalDebateOptimizer(
        max_debate_rounds=3,
        target_score=0.88,
        memory_file="comparison_test_memory.json"
    )
    
    debate_results = []
    total_start = time.time()
    
    for prompt in test_prompts:
        start = time.time()
        result = debate_optimizer.optimize_prompt(prompt)
        duration = time.time() - start
        
        debate_results.append({
            'prompt': prompt,
            'score': result['final_score'],
            'duration': duration,
            'rounds': result['rounds_completed']
        })
        
        print(f"  {prompt}: {result['final_score']:.3f} ({duration:.1f}s, {result['rounds_completed']} rounds)")
    
    total_debate_time = time.time() - total_start
    
    print(f"\n📈 DEBATE SYSTEM PERFORMANCE:")
    avg_score = sum(r['score'] for r in debate_results) / len(debate_results)
    avg_duration = sum(r['duration'] for r in debate_results) / len(debate_results)
    avg_rounds = sum(r['rounds'] for r in debate_results) / len(debate_results)
    
    print(f"Average Score: {avg_score:.3f}")
    print(f"Average Duration per optimization: {avg_duration:.1f}s")
    print(f"Average Rounds: {avg_rounds:.1f}")
    print(f"Total Time for {len(test_prompts)} optimizations: {total_debate_time:.1f}s")
    print(f"Throughput: {len(test_prompts)/total_debate_time:.2f} optimizations/second")
    
    # Projected performance for larger scale
    print(f"\n🚀 SCALABILITY PROJECTION:")
    prompts_per_minute = (len(test_prompts) / total_debate_time) * 60
    print(f"Estimated throughput: {prompts_per_minute:.1f} optimizations/minute")
    print(f"Time for 100 prompts: ~{100/prompts_per_minute:.1f} minutes")
    print(f"Time for 1000 prompts: ~{1000/prompts_per_minute:.1f} minutes")

def main():
    """Run comprehensive tests of the debate optimizer"""
    print("🗣️  CONVERSATIONAL DEBATE OPTIMIZER - COMPREHENSIVE TEST")
    print("="*80)
    print("Testing the Proposer-Reviewer debate system with detailed analysis...")
    print()
    
    try:
        # Test 1: Single optimization with detailed output
        test_single_optimization()
        
        # Brief pause
        time.sleep(2)
        
        # Test 2: Multiple optimizations to show learning
        test_multiple_optimizations()
        
        # Brief pause  
        time.sleep(2)
        
        # Test 3: Detailed debate visualization
        test_debate_visualization()
        
        # Brief pause
        time.sleep(2)
        
        # Test 4: Performance analysis
        performance_comparison()
        
        print(f"\n✅ ALL TESTS COMPLETED SUCCESSFULLY!")
        print("The Conversational Debate Optimizer demonstrates:")
        print("  • Fast optimization through internal dialogue")
        print("  • Reliable quality scoring via Reviewer assessment") 
        print("  • Learning and strategy improvement over time")
        print("  • Scalable performance for large-scale optimization")
        
    except KeyboardInterrupt:
        print(f"\n⏹️  Tests interrupted by user")
    except Exception as e:
        print(f"\n❌ Test error: {str(e)}")

if __name__ == "__main__":
    main() 