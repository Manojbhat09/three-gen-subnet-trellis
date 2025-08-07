#!/usr/bin/env python3
"""
Test the Optimized Intelligent Router against comprehensive new benchmark data.
This will help us evaluate and improve the router's performance.
"""

import json
from optimized_intelligent_router import OptimizedIntelligentRouter

def load_new_benchmark_data():
    """Load the new comprehensive benchmark data"""
    benchmark_data = {
        "timestamp": 1754206094.2856083,
        "test_prompts": [
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
        ],
        "summaries": {
            "flux_isometric_3d": {
                "lora_name": "Flux Isometric 3D",
                "avg_validation_score": 0.7731558521588643,
                "success_rate": 1.0
            },
            "flux_game_assets": {
                "lora_name": "3D Game Assets", 
                "avg_validation_score": 0.7481546183427175,
                "success_rate": 1.0
            },
            "flux_patched_realism": {
                "lora_name": "Patched Realism",
                "avg_validation_score": 0.7507900098959605,
                "success_rate": 1.0
            },
            "flux_tf2_style": {
                "lora_name": "Team Fortress 2 Style",
                "avg_validation_score": 0.705452275276184,
                "success_rate": 1.0
            },
            "flux_baolei": {
                "lora_name": "Baolei Style",
                "avg_validation_score": 0.8267956336339315,
                "success_rate": 1.0
            },
            "flux_cartoon_3d": {
                "lora_name": "Cartoon 3D Render",
                "avg_validation_score": 0.7964165488878886,
                "success_rate": 1.0
            },
            "flux_cinema": {
                "lora_name": "Cinema Style",
                "avg_validation_score": 0.8009371221065521,
                "success_rate": 1.0
            },
            "sd15_game_icon": {
                "lora_name": "Game Icon Institute",
                "avg_validation_score": 0.6979164222876231,
                "success_rate": 1.0
            }
        }
    }
    
    # Extract the optimal choice for each prompt based on detailed results
    detailed_results = {
        "rose quartz heart pendant symbolizing love": "Baolei Style",  # 0.825728714466095
        "glossy blue glass candle holder elegant": "Cartoon 3D Render",  # 0.8643853664398193
        "orange electric sander with variable speed": "Cinema Style",  # 0.867091715335835
        "polished steel drums bright and tropical": "3D Game Assets",  # 0.867581844329834
        "glimmering orange agate with wavy pattern": "Cinema Style",  # 0.8700268864631653
        "heavy-duty green plasma rifle": "Flux Isometric 3D",  # 0.9531517028808594
        "amethyst anklet with swirling vine-like patterns": "Flux Isometric 3D",  # 0.8758448958396912
        "copper measuring tape retractable": "Team Fortress 2 Style",  # 0.852660596370697
        "metal scissors with two sharp blades and curved shape": "Cinema Style",  # 0.9419572949409485
        "red triangle with black circle on it": "Cinema Style",  # 0.9305820465087891
        "smooth purple lacrosse stick": "Team Fortress 2 Style",  # 0.963606595993042
        "dark steel knife serrated edge and pointed tip": "Patched Realism",  # 0.9255967736244202
        "ornate bronze cannon with curved barrel": "Cinema Style",  # 0.8980454206466675
        "red and blue monkey with long tail": "Cartoon 3D Render",  # 0.8839678764343262
        "silver glowing mermaid": "Cartoon 3D Render"  # 0.9270880818367004
    }
    
    return benchmark_data["test_prompts"], detailed_results

def test_router_against_new_benchmark():
    """Test the optimized intelligent router against new benchmark data"""
    print("🧠 Testing Optimized Intelligent Router Against New Comprehensive Benchmark")
    print("=" * 80)
    
    # Load test data
    test_prompts, optimal_choices = load_new_benchmark_data()
    
    # Initialize router
    router = OptimizedIntelligentRouter()
    
    correct_predictions = 0
    total_predictions = len(test_prompts)
    
    print(f"\n🎯 Testing {total_predictions} prompts...\n")
    
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
        
        print(f"{i:2d}. {status}")
        print(f"    Prompt: '{prompt}'")
        print(f"    Predicted: {predicted_lora}")
        print(f"    Optimal:   {optimal_lora}")
        print(f"    Reasoning: {result.reasoning}")
        print(f"    Confidence: {result.confidence}")
        print()
    
    # Calculate accuracy
    accuracy = (correct_predictions / total_predictions) * 100
    
    print("=" * 80)
    print(f"📊 FINAL RESULTS:")
    print(f"   Correct predictions: {correct_predictions}/{total_predictions}")
    print(f"   Accuracy: {accuracy:.1f}%")
    print("=" * 80)
    
    # Analyze failures if any
    if correct_predictions < total_predictions:
        print("\n🔍 FAILURE ANALYSIS:")
        failures = []
        for prompt in test_prompts:
            result = router.route(prompt)
            if result.recommended_lora != optimal_choices[prompt]:
                failures.append({
                    'prompt': prompt,
                    'predicted': result.recommended_lora,
                    'optimal': optimal_choices[prompt],
                    'reasoning': result.reasoning
                })
        
        for failure in failures:
            print(f"\n❌ FAILED: '{failure['prompt']}'")
            print(f"   Predicted: {failure['predicted']}")
            print(f"   Should be: {failure['optimal']}")
            print(f"   Reasoning: {failure['reasoning']}")
        
        print(f"\n📈 IMPROVEMENT OPPORTUNITIES:")
        print(f"   - Need to refine decision rules for {len(failures)} cases")
        print(f"   - Focus on specific object categories that are failing")
        print(f"   - Review and enhance the intelligent decision framework")
    
    return accuracy, failures if correct_predictions < total_predictions else []

def analyze_benchmark_patterns():
    """Analyze patterns in the new benchmark data to inform improvements"""
    print("\n🔍 ANALYZING NEW BENCHMARK PATTERNS:")
    print("=" * 60)
    
    test_prompts, optimal_choices = load_new_benchmark_data()
    
    # Group by optimal LoRA
    lora_groups = {}
    for prompt, lora in optimal_choices.items():
        if lora not in lora_groups:
            lora_groups[lora] = []
        lora_groups[lora].append(prompt)
    
    print("📋 OPTIMAL LoRA DISTRIBUTION:")
    for lora, prompts in lora_groups.items():
        print(f"\n🎯 {lora} ({len(prompts)} prompts):")
        for prompt in prompts:
            print(f"   • {prompt}")
    
    print("\n💡 PATTERN INSIGHTS:")
    print("   • Cinema Style: Best for tools/weapons and geometric shapes")
    print("   • Flux Isometric 3D: Excellent for weapons and jewelry")
    print("   • Cartoon 3D Render: Great for creatures and elegant objects")
    print("   • Team Fortress 2 Style: Good for sports equipment and tools")
    print("   • Baolei Style: Excellent for precious materials/jewelry")
    print("   • 3D Game Assets: Good for musical instruments")
    print("   • Patched Realism: Good for sharp tools/weapons")

if __name__ == "__main__":
    # First analyze the patterns
    analyze_benchmark_patterns()
    
    # Then test the router
    accuracy, failures = test_router_against_new_benchmark()
    
    if accuracy == 100.0:
        print("\n🎉 PERFECT SCORE! The router achieved 100% accuracy!")
        print("🏆 Mission accomplished - truly intelligent organic routing!")
    else:
        print(f"\n⚡ Current accuracy: {accuracy:.1f}%")
        print("🔧 Time to optimize further and achieve 100% accuracy!")
        print("💪 Keep pushing for perfection!") 