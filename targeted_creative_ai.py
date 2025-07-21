#!/usr/bin/env python3
"""
Targeted Creative AI Generator
=============================
Focused approach that forces AI to generate simple, effective variations
based on proven patterns from our successful tests.
"""

import requests
import json
import subprocess
import sys
from typing import List, Tuple

def force_targeted_creative_variations(target_prompt: str) -> List[str]:
    """Force AI to create variations using proven successful patterns"""
    
    print(f"🎯 TARGETED CREATIVE GENERATION: '{target_prompt}'")
    
    # Ultra-targeted prompt based on our successful patterns
    creative_request = f"""Generate 5 enhanced prompts for 3D generation using proven high-scoring patterns.

TARGET OBJECT: {target_prompt}

PROVEN PATTERNS (use these exactly):
Pattern A: "wbgmsst, aerospace-grade precision-engineered [TARGET], ultra-high technical specification, white background"
Pattern B: "wbgmsst, defense-grade ultra-precision [TARGET], ultra-high technical specification, white background"  
Pattern C: "wbgmsst, military-spec ultra-detailed [TARGET], advanced engineering design, white background"
Pattern D: "wbgmsst, masterpiece-quality precision-crafted [TARGET], premium manufacturing excellence, white background"
Pattern E: "wbgmsst, laboratory-grade ultra-precision [TARGET], aerospace-engineering excellence, white background"

STRICT RULES:
1. Replace [TARGET] with exactly: {target_prompt}
2. Keep prompts under 130 characters
3. Use patterns exactly as shown
4. Do NOT add extra words or descriptions

RESPOND WITH EXACTLY 5 PROMPTS:

Prompt 1:
Prompt 2:
Prompt 3:
Prompt 4:
Prompt 5:"""

    data = {
        "model": "llama3.2:3b",
        "messages": [
            {
                "role": "system", 
                "content": "You follow instructions exactly. You replace [TARGET] with the specified object. You keep responses under 130 characters. You use the exact patterns provided."
            },
            {"role": "user", "content": creative_request}
        ],
        "stream": False,
        "options": {
            "temperature": 0.3,  # Lower temperature for better compliance
            "num_predict": 400
        }
    }
    
    try:
        print("   🤖 Generating targeted variations...")
        response = requests.post("http://localhost:11434/api/chat", json=data, timeout=30)
        response.raise_for_status()
        ai_response = response.json()["message"]["content"]
        
        print(f"   📝 AI Response: {ai_response}")
        
        # Extract prompts
        variations = extract_targeted_prompts(ai_response, target_prompt)
        
        # If AI failed, use manual patterns
        if len(variations) < 3:
            print("   🔧 AI didn't follow format, using manual patterns...")
            variations = generate_manual_patterns(target_prompt)
        
        print(f"   ✅ Generated {len(variations)} targeted variations:")
        for i, var in enumerate(variations, 1):
            print(f"      {i}. {var}")
        
        return variations
        
    except Exception as e:
        print(f"   ❌ Error: {e}")
        return generate_manual_patterns(target_prompt)

def extract_targeted_prompts(response: str, target: str) -> List[str]:
    """Extract prompts from AI response - improved to handle AI patterns"""
    
    variations = []
    lines = response.split('\n')
    
    for line in lines:
        line = line.strip()
        
        # Remove pattern prefixes that AI might add
        for prefix in ['Pattern A:', 'Pattern B:', 'Pattern C:', 'Pattern D:', 'Pattern E:', 
                      'Prompt 1:', 'Prompt 2:', 'Prompt 3:', 'Prompt 4:', 'Prompt 5:', 
                      '1.', '2.', '3.', '4.', '5.']:
            if line.startswith(prefix):
                line = line[len(prefix):].strip()
        
        # Remove quotes
        line = line.strip('"\'')
        
        # Check if valid - look for any line that starts with wbgmsst and contains target
        if (line.lower().startswith('wbgmsst,') and 
            target.lower() in line.lower() and
            ', white background' in line.lower() and
            50 <= len(line) <= 150):  # Increased length limit slightly
            variations.append(line)
    
    return list(set(variations))

def generate_manual_patterns(target_prompt: str) -> List[str]:
    """Generate variations using proven manual patterns"""
    
    print("   🔧 Using proven manual patterns...")
    
    patterns = [
        f"wbgmsst, aerospace-grade precision-engineered {target_prompt}, ultra-high technical specification, white background",
        f"wbgmsst, defense-grade ultra-precision {target_prompt}, ultra-high technical specification, white background",
        f"wbgmsst, military-spec ultra-detailed {target_prompt}, advanced engineering design, white background",
        f"wbgmsst, masterpiece-quality precision-crafted {target_prompt}, premium manufacturing excellence, white background",
        f"wbgmsst, laboratory-grade ultra-precision {target_prompt}, aerospace-engineering excellence, white background"
    ]
    
    return patterns

def test_variations_fast(variations: List[str]) -> List[Tuple[str, float]]:
    """Test variations with validation"""
    
    print(f"\n🔬 TESTING {len(variations)} TARGETED VARIATIONS")
    print("=" * 80)
    
    results = []
    
    for i, variation in enumerate(variations, 1):
        print(f"\n🔧 Testing {i}/{len(variations)}")
        print(f"   📝 {variation}")
        
        score = run_validation(variation)
        results.append((variation, score))
        
        print(f"   📊 Score: {score:.3f}")
        
        if score >= 0.96:
            print(f"   🎉 ULTRA SUCCESS!")
        elif score >= 0.8:
            print(f"   ✨ HIGH SUCCESS!")
        elif score >= 0.6:
            print(f"   📈 GOOD SUCCESS!")
    
    return results

def run_validation(prompt: str) -> float:
    """Run validation with proper environment"""
    try:
        cmd = [
            "bash", "-c", 
            f"source /home/mbhat/miniconda/bin/activate && conda activate trellis_new && python subnet_accurate_validator.py '{prompt}'"
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        
        if result.returncode != 0:
            return 0.0
        
        with open("subnet_validation_results.json", 'r') as f:
            data = json.load(f)
            return data.get("validation_engine_score", 0.0)
    
    except Exception:
        return 0.0

def analyze_targeted_results(results: List[Tuple[str, float]], target: str):
    """Analyze targeted results"""
    
    print(f"\n🎓 TARGETED ANALYSIS: '{target}'")
    print("=" * 80)
    
    if not results:
        print("❌ No results to analyze")
        return
    
    # Sort by score
    sorted_results = sorted(results, key=lambda x: x[1], reverse=True)
    
    ultra_count = sum(1 for _, score in results if score >= 0.96)
    high_count = sum(1 for _, score in results if 0.8 <= score < 0.96)
    good_count = sum(1 for _, score in results if 0.6 <= score < 0.8)
    avg_score = sum(score for _, score in results) / len(results)
    
    print(f"📊 TARGETED PERFORMANCE:")
    print(f"   🏆 Ultra (≥0.96): {ultra_count}")
    print(f"   ✨ High (≥0.80): {high_count}")
    print(f"   📈 Good (≥0.60): {good_count}")
    print(f"   📊 Average: {avg_score:.3f}")
    
    if sorted_results:
        best_prompt, best_score = sorted_results[0]
        print(f"\n🏆 BEST TARGETED RESULT:")
        print(f"   📊 Score: {best_score:.3f}")
        print(f"   📝 Prompt: {best_prompt}")
    
    print(f"\n📈 ALL TARGETED RESULTS:")
    for prompt, score in sorted_results:
        status = "🎉" if score >= 0.96 else "✨" if score >= 0.8 else "📈" if score >= 0.6 else "📊"
        print(f"   {status} {score:.3f}: {prompt}")
    
    # Find most effective pattern
    print(f"\n🔍 PATTERN EFFECTIVENESS ANALYSIS:")
    pattern_scores = {}
    
    for prompt, score in results:
        if "aerospace-grade precision-engineered" in prompt:
            pattern = "aerospace-grade precision-engineered"
        elif "defense-grade ultra-precision" in prompt:
            pattern = "defense-grade ultra-precision"
        elif "military-spec ultra-detailed" in prompt:
            pattern = "military-spec ultra-detailed"
        elif "masterpiece-quality precision-crafted" in prompt:
            pattern = "masterpiece-quality precision-crafted"
        elif "laboratory-grade ultra-precision" in prompt:
            pattern = "laboratory-grade ultra-precision"
        else:
            pattern = "other"
        
        if pattern not in pattern_scores:
            pattern_scores[pattern] = []
        pattern_scores[pattern].append(score)
    
    for pattern, scores in pattern_scores.items():
        avg_pattern_score = sum(scores) / len(scores)
        max_pattern_score = max(scores)
        print(f"   🎯 {pattern}: Avg {avg_pattern_score:.3f}, Max {max_pattern_score:.3f}")

def main():
    """Test targeted creative generation"""
    
    print("🚀 TARGETED CREATIVE AI GENERATOR")
    print("🎯 Mission: Use proven patterns for consistent high scores")
    print("⚡ Strategy: Force AI to use successful patterns exactly")
    print("=" * 80)
    
    test_prompts = [
        "hexagonal prism steel structure",
        "elegant silk fabric draping",
        "transparent glass sphere with reflections"
    ]
    
    all_results = []
    
    for prompt in test_prompts:
        print(f"\n{'='*20} TARGETED SESSION: {prompt} {'='*20}")
        
        # Generate targeted variations
        variations = force_targeted_creative_variations(prompt)
        
        # Test variations
        results = test_variations_fast(variations)
        all_results.extend(results)
        
        # Analyze results
        analyze_targeted_results(results, prompt)
        
        print(f"\n⏸️ Brief pause...")
        import time
        time.sleep(2)
    
    # Overall summary
    if all_results:
        print(f"\n🎓 OVERALL TARGETED PERFORMANCE")
        print("=" * 80)
        
        overall_avg = sum(score for _, score in all_results) / len(all_results)
        overall_best = max(score for _, score in all_results)
        ultra_total = sum(1 for _, score in all_results if score >= 0.96)
        
        print(f"📊 FINAL SUMMARY:")
        print(f"   Total Variations Tested: {len(all_results)}")
        print(f"   Overall Average Score: {overall_avg:.3f}")
        print(f"   Overall Best Score: {overall_best:.3f}")
        print(f"   Total Ultra Achievements: {ultra_total}")
        
        if overall_avg >= 0.7:
            print(f"   🏆 EXCELLENT: Targeted approach is highly effective!")
        elif overall_avg >= 0.6:
            print(f"   ✨ GOOD: Targeted approach shows strong results!")
        else:
            print(f"   📈 DEVELOPING: Targeted approach needs refinement!")

if __name__ == "__main__":
    main() 