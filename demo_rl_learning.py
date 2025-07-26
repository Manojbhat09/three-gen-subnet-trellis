#!/usr/bin/env python3
"""
Demonstration of RL Agent Learning Process
==========================================
Shows how the LLM RL agent would learn and evolve through validation feedback
"""

import json
import time
import subprocess
import sys
from pathlib import Path

class ValidationRunner:
    """Runs validation and captures scores"""
    
    def validate_prompt(self, prompt: str) -> float:
        """Run validation and return score"""
        try:
            print(f"🔍 Validating: '{prompt[:50]}...'")
            
            # Run the validator
            cmd = [
                "bash", "-c", 
                f"source /home/mbhat/miniconda/bin/activate && conda activate trellis_new && python subnet_accurate_validator.py \"{prompt}\""
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
            
            if result.returncode == 0:
                # Parse the results file
                with open("subnet_validation_results.json", 'r') as f:
                    data = json.load(f)
                    score = data.get("validation_engine_score", 0.0)
                    print(f"   📊 Score: {score:.4f}")
                    return score
            else:
                print(f"   ❌ Validation failed: {result.stderr}")
                return 0.0
        except Exception as e:
            print(f"   ❌ Error: {e}")
            return 0.0

def demonstrate_rl_learning():
    """Demonstrate how the RL agent would learn through iterations"""
    
    print("🧠 RL AGENT LEARNING DEMONSTRATION")
    print("=" * 60)
    print("Simulating how the agent learns through validation feedback")
    print("=" * 60)
    
    validator = ValidationRunner()
    
    # Test prompts with different optimization strategies
    test_cases = [
        {
            "original": "golden ring",
            "optimizations": [
                ("baseline", "wbgmsst, golden ring, white background"),
                ("conservative", "wbgmsst, polished golden ring, white background"),
                ("material_focus", "wbgmsst, lustrous golden ring with reflective metallic surface, white background"),
                ("artistic", "wbgmsst, elegant golden ring with warm ambient lighting, white background"),
                ("technical", "wbgmsst, precision-crafted golden ring with mirror-finish surface, white background"),
                ("aggressive", "wbgmsst, exquisite handcrafted golden ring with intricate engravings and lustrous mirror-polish finish, set against pristine white background"),
            ]
        },
        {
            "original": "red sports car",
            "optimizations": [
                ("baseline", "wbgmsst, red sports car, white background"),
                ("conservative", "wbgmsst, sleek red sports car, white background"),
                ("material_focus", "wbgmsst, glossy red sports car with metallic paint finish, white background"),
                ("artistic", "wbgmsst, dynamic red sports car with dramatic lighting, white background"),
                ("technical", "wbgmsst, precision-engineered red sports car with aerodynamic design, white background"),
            ]
        }
    ]
    
    learning_results = []
    
    for test_case in test_cases:
        print(f"\n🎯 TESTING: '{test_case['original']}'")
        print("-" * 50)
        
        case_results = []
        
        for strategy, prompt in test_case["optimizations"]:
            print(f"\n📋 Strategy: {strategy}")
            score = validator.validate_prompt(prompt)
            
            case_results.append({
                "strategy": strategy,
                "prompt": prompt,
                "score": score,
                "success": score >= 0.8  # High score threshold
            })
            
            time.sleep(1)  # Brief pause between validations
        
        # Analyze results for this case
        print(f"\n📊 RESULTS ANALYSIS for '{test_case['original']}':")
        best_strategy = max(case_results, key=lambda x: x["score"])
        worst_strategy = min(case_results, key=lambda x: x["score"])
        
        print(f"   🏆 Best: {best_strategy['strategy']} (score: {best_strategy['score']:.4f})")
        print(f"   📉 Worst: {worst_strategy['strategy']} (score: {worst_strategy['score']:.4f})")
        
        # Show what the RL agent would learn
        successful_strategies = [r for r in case_results if r["success"]]
        if successful_strategies:
            print(f"   ✅ Successful strategies: {[s['strategy'] for s in successful_strategies]}")
        else:
            print(f"   ⚠️ No strategies achieved high scores (≥0.8)")
        
        learning_results.append({
            "original": test_case["original"],
            "results": case_results,
            "best_strategy": best_strategy,
            "learning": {
                "successful_strategies": [s["strategy"] for s in successful_strategies],
                "strategy_rankings": sorted(case_results, key=lambda x: x["score"], reverse=True)
            }
        })
    
    # Show overall learning insights
    print(f"\n🧠 OVERALL RL LEARNING INSIGHTS:")
    print("=" * 50)
    
    # Aggregate strategy performance
    strategy_performance = {}
    for result in learning_results:
        for case_result in result["results"]:
            strategy = case_result["strategy"]
            if strategy not in strategy_performance:
                strategy_performance[strategy] = {"scores": [], "successes": 0, "attempts": 0}
            
            strategy_performance[strategy]["scores"].append(case_result["score"])
            strategy_performance[strategy]["attempts"] += 1
            if case_result["success"]:
                strategy_performance[strategy]["successes"] += 1
    
    # Rank strategies by performance
    print("📈 STRATEGY PERFORMANCE RANKING:")
    strategy_rankings = []
    for strategy, perf in strategy_performance.items():
        avg_score = sum(perf["scores"]) / len(perf["scores"])
        success_rate = perf["successes"] / perf["attempts"]
        combined_score = (success_rate * 0.7) + (avg_score * 0.3)
        
        strategy_rankings.append({
            "strategy": strategy,
            "avg_score": avg_score,
            "success_rate": success_rate,
            "combined_score": combined_score,
            "attempts": perf["attempts"]
        })
    
    strategy_rankings.sort(key=lambda x: x["combined_score"], reverse=True)
    
    for i, rank in enumerate(strategy_rankings, 1):
        print(f"   {i}. {rank['strategy']:20} | Avg: {rank['avg_score']:.4f} | Success: {rank['success_rate']:.1%} | Combined: {rank['combined_score']:.4f}")
    
    # Show what the agent would learn
    best_strategy = strategy_rankings[0]
    worst_strategy = strategy_rankings[-1]
    
    print(f"\n🎓 RL AGENT LEARNING CONCLUSIONS:")
    print(f"   🏆 Exploit: '{best_strategy['strategy']}' (combined score: {best_strategy['combined_score']:.4f})")
    print(f"   🔍 Avoid: '{worst_strategy['strategy']}' (combined score: {worst_strategy['combined_score']:.4f})")
    print(f"   📊 Exploration vs Exploitation balance needed")
    
    # Show how epsilon would decay
    print(f"\n🔄 EPSILON DECAY SIMULATION:")
    epsilon = 0.3
    epsilon_decay = 0.99
    epsilon_min = 0.1
    
    for step in range(0, 20, 5):
        print(f"   Step {step:2d}: ε={epsilon:.3f} ({'explore' if epsilon > 0.2 else 'mostly exploit'})")
        for _ in range(5):
            epsilon = max(epsilon_min, epsilon * epsilon_decay)
    
    print(f"\n✅ DEMONSTRATION COMPLETE")
    print(f"This shows how the RL agent learns optimal strategies through validation feedback!")

if __name__ == "__main__":
    demonstrate_rl_learning() 
