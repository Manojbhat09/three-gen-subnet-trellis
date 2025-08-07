#!/usr/bin/env python3
"""
Test the Adaptive Learning Router against comprehensive new benchmark data.
"""

import json
from adaptive_learning_router import AdaptiveLearningRouter

def load_new_benchmark_data():
    """Load the new comprehensive benchmark data"""
    test_prompts = [
        "rose quartz heart pendant symbolizing love",
        "glossy blue glass candle holder elegant", 
        "orange electric sander with variable speed",
        "polished steel drums bright and tropical",
        "glimmering orange agate with wavy pattern",
        "heavy-duty green plasma rifle",
        "amethyst anklet with swirling vine-like patterns",
        "copper measuring tape retractable",
        "metal scissors with two sharp blades and curved shape",
        "red triangle with black circle on it",
        "smooth purple lacrosse stick",
        "dark steel knife serrated edge and pointed tip",
        "ornate bronze cannon with curved barrel",
        "red and blue monkey with long tail",
        "silver glowing mermaid"
    ]
    
    # Optimal choices based on actual performance
    optimal_choices = {
        "rose quartz heart pendant symbolizing love": "Baolei Style",  # 0.826
        "glossy blue glass candle holder elegant": "Cartoon 3D Render",  # 0.864
        "orange electric sander with variable speed": "Cinema Style",  # 0.867
        "polished steel drums bright and tropical": "3D Game Assets",  # 0.868
        "glimmering orange agate with wavy pattern": "Cinema Style",  # 0.870
        "heavy-duty green plasma rifle": "Flux Isometric 3D",  # 0.953
        "amethyst anklet with swirling vine-like patterns": "Flux Isometric 3D",  # 0.876
        "copper measuring tape retractable": "Team Fortress 2 Style",  # 0.853
        "metal scissors with two sharp blades and curved shape": "Cinema Style",  # 0.942
        "red triangle with black circle on it": "Cinema Style",  # 0.931
        "smooth purple lacrosse stick": "Team Fortress 2 Style",  # 0.964
        "dark steel knife serrated edge and pointed tip": "Patched Realism",  # 0.926
        "ornate bronze cannon with curved barrel": "Cinema Style",  # 0.898
        "red and blue monkey with long tail": "Cartoon 3D Render",  # 0.884
        "silver glowing mermaid": "Cartoon 3D Render"  # 0.927
    }
    
    return test_prompts, optimal_choices

def test_adaptive_router_full_benchmark():
    """Test the adaptive learning router against the full benchmark"""
    print("🧠 Testing Adaptive Learning Router Against Full Benchmark")
    print("=" * 70)
    
    # Load test data
    test_prompts, optimal_choices = load_new_benchmark_data()
    
    # Initialize router
    router = AdaptiveLearningRouter()
    
    correct_predictions = 0
    total_predictions = len(test_prompts)
    
    print(f"\n🎯 Testing {total_predictions} prompts...\n")
    
    results = []
    
    for i, prompt in enumerate(test_prompts, 1):
        # Get router's recommendation
        result = router.route(prompt)
        predicted_lora = result.recommended_lora
        optimal_lora = optimal_choices[prompt]
        
        # Check if prediction is correct
        is_correct = predicted_lora == optimal_lora
        if is_correct:
            correct_predictions += 1
            status = "✅ CORRECT"
        else:
            status = "❌ INCORRECT"
        
        results.append({
            'prompt': prompt,
            'predicted': predicted_lora,
            'optimal': optimal_lora,
            'correct': is_correct,
            'reasoning': result.reasoning,
            'confidence': result.confidence
        })
        
        print(f"{i:2d}. {status}")
        print(f"    Prompt: '{prompt[:60]}{'...' if len(prompt) > 60 else ''}'")
        print(f"    Predicted: {predicted_lora}")
        print(f"    Optimal:   {optimal_lora}")
        if not is_correct:
            print(f"    Reasoning: {result.reasoning[:100]}{'...' if len(result.reasoning) > 100 else ''}")
        print()
    
    # Calculate accuracy
    accuracy = (correct_predictions / total_predictions) * 100
    
    print("=" * 70)
    print(f"📊 FINAL RESULTS:")
    print(f"   Correct predictions: {correct_predictions}/{total_predictions}")
    print(f"   Accuracy: {accuracy:.1f}%")
    print("=" * 70)
    
    # Analyze failures if any
    failures = [r for r in results if not r['correct']]
    
    if failures:
        print(f"\n🔍 FAILURE ANALYSIS ({len(failures)} failures):")
        for failure in failures:
            print(f"\n❌ FAILED: '{failure['prompt'][:50]}{'...' if len(failure['prompt']) > 50 else ''}'")
            print(f"   Predicted: {failure['predicted']}")
            print(f"   Should be: {failure['optimal']}")
            print(f"   Reasoning: {failure['reasoning'][:120]}{'...' if len(failure['reasoning']) > 120 else ''}")
        
        print(f"\n📈 PATTERNS IN FAILURES:")
        
        # Analyze patterns in failures
        failed_by_predicted = {}
        failed_by_optimal = {}
        
        for failure in failures:
            pred = failure['predicted']
            opt = failure['optimal']
            
            if pred not in failed_by_predicted:
                failed_by_predicted[pred] = []
            failed_by_predicted[pred].append(failure['prompt'])
            
            if opt not in failed_by_optimal:
                failed_by_optimal[opt] = []
            failed_by_optimal[opt].append(failure['prompt'])
        
        print("\nOver-predicted LoRAs:")
        for lora, prompts in failed_by_predicted.items():
            print(f"   {lora}: {len(prompts)} times")
        
        print("\nUnder-predicted LoRAs:")
        for lora, prompts in failed_by_optimal.items():
            print(f"   {lora}: {len(prompts)} times")
    
    else:
        print("\n🎉 PERFECT SCORE! The adaptive router achieved 100% accuracy!")
        print("🏆 Mission accomplished - truly intelligent adaptive learning!")
    
    return accuracy, failures

def analyze_learning_effectiveness():
    """Analyze how well the router learns from the performance data"""
    print("\n🔬 LEARNING EFFECTIVENESS ANALYSIS:")
    print("=" * 50)
    
    test_prompts, optimal_choices = load_new_benchmark_data()
    
    # Count optimal LoRA usage
    lora_counts = {}
    for optimal in optimal_choices.values():
        lora_counts[optimal] = lora_counts.get(optimal, 0) + 1
    
    print("📊 Distribution of optimal LoRAs in benchmark:")
    for lora, count in sorted(lora_counts.items(), key=lambda x: x[1], reverse=True):
        percentage = (count / len(test_prompts)) * 100
        print(f"   {lora}: {count} prompts ({percentage:.1f}%)")
    
    print("\n💡 LEARNING INSIGHTS:")
    print("   • Cinema Style is optimal for 5/15 prompts (33.3%) - most versatile")
    print("   • Cartoon 3D Render is optimal for 3/15 prompts (20%) - creatures/elegant")
    print("   • Flux Isometric 3D is optimal for 2/15 prompts (13.3%) - weapons/jewelry")
    print("   • Team Fortress 2 Style is optimal for 2/15 prompts (13.3%) - sports/tools")
    print("   • The adaptive router must learn these patterns from data, not rules!")

if __name__ == "__main__":
    # Analyze learning effectiveness first
    analyze_learning_effectiveness()
    
    # Test the router
    accuracy, failures = test_adaptive_router_full_benchmark()
    
    if accuracy >= 90.0:
        print(f"\n🌟 EXCELLENT! Achieved {accuracy:.1f}% accuracy!")
        print("🚀 The adaptive learning approach is working!")
    elif accuracy >= 70.0:
        print(f"\n⚡ GOOD PROGRESS! Achieved {accuracy:.1f}% accuracy!")
        print("🔧 Getting close to optimal performance!")
    else:
        print(f"\n⚠️ Need improvement: {accuracy:.1f}% accuracy")
        print("🧠 Time to enhance the learning algorithm!") 