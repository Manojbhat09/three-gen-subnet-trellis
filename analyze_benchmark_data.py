#!/usr/bin/env python3
"""
Comprehensive Analysis of New Benchmark Data
Extract patterns and performance insights for intelligent routing
"""

import json
from typing import Dict, List, Tuple
from dataclasses import dataclass
import statistics

@dataclass
class LoRAPerformance:
    name: str
    avg_score: float
    best_score: float
    worst_score: float
    consistency: float  # 1 - (range / avg)
    variance: float
    success_rate: float

def analyze_comprehensive_data():
    """Analyze the new comprehensive benchmark data"""
    
    # Parse the data you provided
    data = {
        "test_prompts": [
            "robot in sitting down position",
            "mystical orb pulsating with arcane energy", 
            "small winged fairy with golden wings",
            "parachute deployed mid-air high-speed descent",
            "metallic robot turning right",
            "colorful candy in clear glass bottle",
            "black knight armored in shadow",
            "magical lantern casting soft blue glow",
            "purple sapphire in necklace",
            "white pear delicate texture slightly translucent"
        ],
        "summaries": {
            "flux_isometric_3d": {"avg_validation_score": 0.7886, "best_score": 0.9044, "worst_score": 0.4869},
            "flux_game_assets": {"avg_validation_score": 0.8292, "best_score": 0.9007, "worst_score": 0.6745},
            "flux_patched_realism": {"avg_validation_score": 0.8302, "best_score": 0.9065, "worst_score": 0.6045},
            "flux_tf2_style": {"avg_validation_score": 0.8598, "best_score": 0.9317, "worst_score": 0.8092},
            "flux_baolei": {"avg_validation_score": 0.8475, "best_score": 0.8983, "worst_score": 0.7398},
            "flux_cartoon_3d": {"avg_validation_score": 0.8423, "best_score": 0.9516, "worst_score": 0.6729},
            "flux_cinema": {"avg_validation_score": 0.8102, "best_score": 0.8930, "worst_score": 0.5238},
            "sd15_game_icon": {"avg_validation_score": 0.6859, "best_score": 0.8684, "worst_score": -0.0094}
        }
    }

    # Individual scores for each prompt (extracted from detailed results)
    prompt_scores = {
        "robot in sitting down position": {
            "Flux Isometric 3D": 0.8426, "3D Game Assets": 0.8701, "Patched Realism": 0.8591,
            "Team Fortress 2 Style": 0.8163, "Baolei Style": 0.7398, "Cartoon 3D Render": 0.7937,
            "Cinema Style": 0.8499, "Game Icon Institute": 0.7200
        },
        "mystical orb pulsating with arcane energy": {
            "Flux Isometric 3D": 0.8109, "3D Game Assets": 0.7439, "Patched Realism": 0.6045,
            "Team Fortress 2 Style": 0.8943, "Baolei Style": 0.8534, "Cartoon 3D Render": 0.8737,
            "Cinema Style": 0.8930, "Game Icon Institute": 0.8172
        },
        "small winged fairy with golden wings": {
            "Flux Isometric 3D": 0.8352, "3D Game Assets": 0.8838, "Patched Realism": 0.8445,
            "Team Fortress 2 Style": 0.8530, "Baolei Style": 0.8690, "Cartoon 3D Render": 0.8794,
            "Cinema Style": 0.8919, "Game Icon Institute": 0.7714
        },
        "parachute deployed mid-air high-speed descent": {
            "Flux Isometric 3D": 0.6961, "3D Game Assets": 0.8931, "Patched Realism": 0.8677,
            "Team Fortress 2 Style": 0.8394, "Baolei Style": 0.8320, "Cartoon 3D Render": 0.8705,
            "Cinema Style": 0.6936, "Game Icon Institute": 0.8342
        },
        "metallic robot turning right": {
            "Flux Isometric 3D": 0.8715, "3D Game Assets": 0.8682, "Patched Realism": 0.8280,
            "Team Fortress 2 Style": 0.8464, "Baolei Style": 0.8271, "Cartoon 3D Render": 0.8575,
            "Cinema Style": 0.8640, "Game Icon Institute": 0.8119
        },
        "colorful candy in clear glass bottle": {
            "Flux Isometric 3D": 0.8406, "3D Game Assets": 0.7992, "Patched Realism": 0.8644,
            "Team Fortress 2 Style": 0.8465, "Baolei Style": 0.8844, "Cartoon 3D Render": 0.9282,
            "Cinema Style": 0.8813, "Game Icon Institute": -0.0094
        },
        "black knight armored in shadow": {
            "Flux Isometric 3D": 0.9044, "3D Game Assets": 0.9007, "Patched Realism": 0.8417,
            "Team Fortress 2 Style": 0.8851, "Baolei Style": 0.8550, "Cartoon 3D Render": 0.8659,
            "Cinema Style": 0.7723, "Game Icon Institute": 0.8147
        },
        "magical lantern casting soft blue glow": {
            "Flux Isometric 3D": 0.4869, "3D Game Assets": 0.6745, "Patched Realism": 0.9065,
            "Team Fortress 2 Style": 0.8761, "Baolei Style": 0.8482, "Cartoon 3D Render": 0.7300,
            "Cinema Style": 0.5238, "Game Icon Institute": 0.4072
        },
        "purple sapphire in necklace": {
            "Flux Isometric 3D": 0.8650, "3D Game Assets": 0.7861, "Patched Realism": 0.8117,
            "Team Fortress 2 Style": 0.8092, "Baolei Style": 0.8983, "Cartoon 3D Render": 0.6729,
            "Cinema Style": 0.8690, "Game Icon Institute": 0.8234
        },
        "white pear delicate texture slightly translucent": {
            "Flux Isometric 3D": 0.7324, "3D Game Assets": 0.8728, "Patched Realism": 0.8739,
            "Team Fortress 2 Style": 0.9317, "Baolei Style": 0.8679, "Cartoon 3D Render": 0.9516,
            "Cinema Style": 0.8632, "Game Icon Institute": 0.8684
        }
    }

    print("🔍 COMPREHENSIVE BENCHMARK ANALYSIS")
    print("=" * 60)

    # 1. Analyze overall performance
    print("\n📊 OVERALL PERFORMANCE RANKING:")
    lora_averages = {}
    for prompt, scores in prompt_scores.items():
        for lora, score in scores.items():
            if lora not in lora_averages:
                lora_averages[lora] = []
            lora_averages[lora].append(score)
    
    lora_stats = {}
    for lora, scores in lora_averages.items():
        avg = statistics.mean(scores)
        variance = statistics.variance(scores) if len(scores) > 1 else 0
        consistency = 1 - ((max(scores) - min(scores)) / avg) if avg > 0 else 0
        lora_stats[lora] = {
            'avg': avg, 'best': max(scores), 'worst': min(scores), 
            'variance': variance, 'consistency': consistency
        }
    
    sorted_loras = sorted(lora_stats.items(), key=lambda x: x[1]['avg'], reverse=True)
    for i, (lora, stats) in enumerate(sorted_loras, 1):
        consistency_pct = stats['consistency'] * 100
        print(f"{i}. {lora}: {stats['avg']:.3f} avg (range: {stats['worst']:.3f}-{stats['best']:.3f}, consistency: {consistency_pct:.1f}%)")

    # 2. Find best LoRA for each prompt
    print(f"\n🎯 OPTIMAL CHOICES PER PROMPT:")
    optimal_choices = {}
    for prompt, scores in prompt_scores.items():
        best_lora = max(scores.items(), key=lambda x: x[1])
        second_best = sorted(scores.items(), key=lambda x: x[1], reverse=True)[1]
        optimal_choices[prompt] = best_lora[0]
        margin = best_lora[1] - second_best[1]
        print(f"'{prompt}'")
        print(f"  🥇 {best_lora[0]}: {best_lora[1]:.3f}")
        print(f"  🥈 {second_best[0]}: {second_best[1]:.3f} (margin: {margin:.3f})")

    # 3. Identify clear patterns
    print(f"\n🧠 PATTERN ANALYSIS:")
    
    # Robot patterns
    robot_prompts = [p for p in prompt_scores.keys() if 'robot' in p.lower()]
    print(f"\n🤖 ROBOT PROMPTS ({len(robot_prompts)}):")
    for prompt in robot_prompts:
        best = max(prompt_scores[prompt].items(), key=lambda x: x[1])
        print(f"  '{prompt}' → {best[0]} ({best[1]:.3f})")
    
    # Small object patterns  
    small_object_prompts = [p for p in prompt_scores.keys() if any(word in p.lower() for word in ['small', 'candy', 'sapphire', 'pear'])]
    print(f"\n🔹 SMALL/DETAILED OBJECTS ({len(small_object_prompts)}):")
    for prompt in small_object_prompts:
        best = max(prompt_scores[prompt].items(), key=lambda x: x[1])
        print(f"  '{prompt}' → {best[0]} ({best[1]:.3f})")
    
    # Fantasy/magical patterns
    fantasy_prompts = [p for p in prompt_scores.keys() if any(word in p.lower() for word in ['mystical', 'magical', 'fairy', 'knight', 'orb'])]
    print(f"\n✨ FANTASY/MAGICAL OBJECTS ({len(fantasy_prompts)}):")
    for prompt in fantasy_prompts:
        best = max(prompt_scores[prompt].items(), key=lambda x: x[1])
        print(f"  '{prompt}' → {best[0]} ({best[1]:.3f})")

    # 4. Analyze failure cases
    print(f"\n⚠️  NOTABLE FAILURES:")
    for prompt, scores in prompt_scores.items():
        for lora, score in scores.items():
            if score < 0.5:
                print(f"  {lora} on '{prompt}': {score:.3f}")

    # 5. Generate decision rules
    print(f"\n📋 EXTRACTED DECISION RULES:")
    
    print(f"1. CONSISTENCY CHAMPIONS (low variance, reliable):")
    consistent_loras = [(lora, stats) for lora, stats in lora_stats.items() if stats['consistency'] > 0.7]
    for lora, stats in sorted(consistent_loras, key=lambda x: x[1]['avg'], reverse=True):
        print(f"   - {lora}: {stats['avg']:.3f} avg, {stats['consistency']*100:.1f}% consistency")
    
    print(f"\n2. HIGH-CEILING SPECIALISTS (high variance, can excel):")
    specialist_loras = [(lora, stats) for lora, stats in lora_stats.items() if stats['variance'] > 0.02 and stats['best'] > 0.9]
    for lora, stats in sorted(specialist_loras, key=lambda x: x[1]['best'], reverse=True):
        print(f"   - {lora}: best {stats['best']:.3f}, variance {stats['variance']:.3f}")

    print(f"\n3. OBJECT TYPE PREFERENCES:")
    # Analyze which LoRAs win most often for each category
    category_winners = {
        'robots': {},
        'small_objects': {},
        'fantasy': {}
    }
    
    for prompt in robot_prompts:
        winner = max(prompt_scores[prompt].items(), key=lambda x: x[1])[0]
        category_winners['robots'][winner] = category_winners['robots'].get(winner, 0) + 1
    
    for prompt in small_object_prompts:
        winner = max(prompt_scores[prompt].items(), key=lambda x: x[1])[0]
        category_winners['small_objects'][winner] = category_winners['small_objects'].get(winner, 0) + 1
        
    for prompt in fantasy_prompts:
        winner = max(prompt_scores[prompt].items(), key=lambda x: x[1])[0]
        category_winners['fantasy'][winner] = category_winners['fantasy'].get(winner, 0) + 1
    
    for category, winners in category_winners.items():
        if winners:
            top_winner = max(winners.items(), key=lambda x: x[1])
            print(f"   - {category.upper()}: {top_winner[0]} wins {top_winner[1]} times")

    return optimal_choices, lora_stats, prompt_scores

if __name__ == "__main__":
    optimal_choices, lora_stats, prompt_scores = analyze_comprehensive_data()
    
    print(f"\n💾 Analysis complete - patterns identified for intelligent routing!") 