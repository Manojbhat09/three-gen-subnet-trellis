#!/usr/bin/env python3
"""
Direct Creative Test - Manual + AI Generated Variations
======================================================
Combines the working pattern with AI creativity
"""

import subprocess
import sys
import json
from typing import List, Tuple

def create_manual_creative_variations(target_prompt: str) -> List[str]:
    """Create manual creative variations using proven patterns"""
    
    variations = [
        # Proven ultra pattern
        f"wbgmsst, defense-grade ultra-precision {target_prompt}, ultra-high technical specification, white background",
        
        # High-scoring patterns
        f"wbgmsst, aerospace-grade precision-engineered {target_prompt}, ultra-high technical specification, white background",
        f"wbgmsst, military-spec ultra-detailed {target_prompt}, advanced engineering design, white background",
        f"wbgmsst, masterpiece-quality precision-crafted {target_prompt}, premium manufacturing excellence, white background",
        
        # Creative combinations
        f"wbgmsst, precision-aerospace-grade {target_prompt}, ultra-precision specification, white background",
        f"wbgmsst, defense-aerospace ultra-precision {target_prompt}, military-spec excellence, white background",
        f"wbgmsst, ultra-military-spec precision-engineered {target_prompt}, aerospace-grade quality, white background",
        f"wbgmsst, laboratory-grade precision-forged {target_prompt}, scientific-specification excellence, white background",
        
        # Authority stacking variations
        f"wbgmsst, military-aerospace-grade {target_prompt}, defense-specification precision, white background",
        f"wbgmsst, ultra-defense-grade precision-crafted {target_prompt}, aerospace-engineering excellence, white background"
    ]
    
    return variations

def run_validation(prompt: str) -> float:
    """Run validation with proper environment"""
    try:
        cmd = [
            "bash", "-c", 
            f"source /home/mbhat/miniconda/bin/activate && conda activate trellis_new && python subnet_accurate_validator.py '{prompt}'"
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        
        if result.returncode != 0:
            print(f"      ⚠️ Validation failed")
            return 0.0
        
        with open("subnet_validation_results.json", 'r') as f:
            data = json.load(f)
            return data.get("validation_engine_score", 0.0)
    
    except Exception as e:
        print(f"      ❌ Error: {e}")
        return 0.0

def test_creative_variations(target_prompt: str):
    """Test creative variations for a target prompt"""
    
    print(f"🎨 DIRECT CREATIVE TEST: '{target_prompt}'")
    print("=" * 80)
    
    # Generate creative variations
    variations = create_manual_creative_variations(target_prompt)
    
    print(f"📋 Generated {len(variations)} creative variations")
    
    # Test each variation
    results = []
    
    for i, variation in enumerate(variations, 1):
        print(f"\n🔧 Testing variation {i}/{len(variations)}")
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
    
    # Analyze results
    print(f"\n🎓 CREATIVE ANALYSIS")
    print("=" * 80)
    
    sorted_results = sorted(results, key=lambda x: x[1], reverse=True)
    
    ultra_count = sum(1 for _, score in results if score >= 0.96)
    high_count = sum(1 for _, score in results if 0.8 <= score < 0.96)
    good_count = sum(1 for _, score in results if 0.6 <= score < 0.8)
    avg_score = sum(score for _, score in results) / len(results)
    
    print(f"📊 PERFORMANCE SUMMARY:")
    print(f"   🏆 Ultra (≥0.96): {ultra_count}")
    print(f"   ✨ High (≥0.80): {high_count}")
    print(f"   📈 Good (≥0.60): {good_count}")
    print(f"   📊 Average: {avg_score:.3f}")
    
    if sorted_results:
        best_prompt, best_score = sorted_results[0]
        print(f"\n🏆 BEST RESULT:")
        print(f"   📊 Score: {best_score:.3f}")
        print(f"   📝 Prompt: {best_prompt}")
    
    print(f"\n📈 TOP 5 RESULTS:")
    for prompt, score in sorted_results[:5]:
        status = "🎉" if score >= 0.96 else "✨" if score >= 0.8 else "📈" if score >= 0.6 else "📊"
        print(f"   {status} {score:.3f}: {prompt}")
    
    return sorted_results

def main():
    """Test creative variations on different prompts"""
    
    print("🚀 DIRECT CREATIVE VARIATION TESTING")
    print("🎯 Testing manual creative combinations + proven patterns")
    print("=" * 80)
    
    test_prompts = [
        "hexagonal prism steel structure",
        "elegant silk fabric draping"
    ]
    
    all_results = {}
    
    for prompt in test_prompts:
        results = test_creative_variations(prompt)
        all_results[prompt] = results
        print(f"\n⏸️ Brief pause...")
        import time
        time.sleep(2)
    
    # Final comparison
    print(f"\n🏆 FINAL COMPARISON")
    print("=" * 80)
    
    for prompt, results in all_results.items():
        best_score = max(score for _, score in results)
        status = "🎉" if best_score >= 0.96 else "✨" if best_score >= 0.8 else "📈"
        print(f"{status} {prompt}: Best score {best_score:.3f}")

if __name__ == "__main__":
    main()
