#!/usr/bin/env python3
"""
Simple Creative AI Generator - Improved Version
===============================================
Better prompt engineering and parsing for LLaMA 3.2:3b responses
"""

import requests
import json
import subprocess
import sys
from typing import List, Tuple
import re

def test_ai_connection():
    """Test if AI is responding"""
    try:
        response = requests.get("http://localhost:11434/api/tags", timeout=10)
        return response.status_code == 200
    except:
        return False

def force_creative_prompt(target_prompt: str) -> List[str]:
    """Force AI to create creative variations with improved prompting"""
    
    print(f"🎨 CREATIVE GENERATION FOR: '{target_prompt}'")
    
    # Improved creative prompt with clear format requirements
    creative_request = f"""Create 5 creative enhanced versions of this prompt for 3D generation.

TARGET: "{target_prompt}"

STRICT FORMAT REQUIREMENTS:
- Must start with EXACTLY: "wbgmsst, "
- Must end with EXACTLY: ", white background"
- Must include the target object: "{target_prompt}"
- Use premium descriptors: aerospace-grade, military-spec, defense-grade, ultra-precision, masterpiece-quality, precision-engineered
- Use technical specs: ultra-high technical specification, advanced engineering design, premium manufacturing excellence

RESPOND WITH EXACTLY 5 PROMPTS IN THIS FORMAT:

1. wbgmsst, [premium descriptor] [process term] {target_prompt}, [technical specification], white background
2. wbgmsst, [premium descriptor] [process term] {target_prompt}, [technical specification], white background  
3. wbgmsst, [premium descriptor] [process term] {target_prompt}, [technical specification], white background
4. wbgmsst, [premium descriptor] [process term] {target_prompt}, [technical specification], white background
5. wbgmsst, [premium descriptor] [process term] {target_prompt}, [technical specification], white background

EXAMPLES:
wbgmsst, aerospace-grade precision-engineered robot arm, ultra-high technical specification, white background
wbgmsst, military-spec ultra-precision steel beam, advanced engineering design, white background

Generate 5 prompts now - follow the format EXACTLY:"""

    data = {
        "model": "llama3.2:3b",
        "messages": [
            {
                "role": "system", 
                "content": "You are a precise AI that follows instructions exactly. You always provide exactly what is requested in the exact format specified. You never deviate from the format requirements."
            },
            {"role": "user", "content": creative_request}
        ],
        "stream": False,
        "options": {
            "temperature": 0.8,  # Slightly lower for better format compliance
            "num_predict": 500   # More tokens for complete responses
        }
    }
    
    try:
        print("   🤖 Querying AI...")
        response = requests.post("http://localhost:11434/api/chat", json=data, timeout=45)
        response.raise_for_status()
        ai_response = response.json()["message"]["content"]
        
        print(f"   📝 AI Response received ({len(ai_response)} chars)")
        print(f"   🔍 Response preview: {ai_response[:150]}...")
        
        print(f"\n   📋 FULL AI RESPONSE:")
        print("   " + "-" * 60)
        print("   " + ai_response.replace('\n', '\n   '))
        print("   " + "-" * 60)
        
        # Extract prompts from response with improved parsing
        variations = extract_prompts_improved(ai_response, target_prompt)
        
        print(f"   🔍 Parsing {len(ai_response.split(chr(10)))} lines...")
        for i, var in enumerate(variations, 1):
            print(f"      ✅ Found valid prompt: {var}")
        
        print(f"   📊 Total extracted: {len(variations)}")
        
        # If we got fewer than expected, try to fix some prompts
        if len(variations) < 3:
            print("   🔧 Attempting to fix invalid prompts...")
            fixed_variations = fix_invalid_prompts(ai_response, target_prompt)
            variations.extend(fixed_variations)
        
        print(f"   ✅ Final result: {len(variations)} variations")
        for i, var in enumerate(variations, 1):
            print(f"      {i}. {var}")
        
        return variations
        
    except Exception as e:
        print(f"   ❌ Error: {e}")
        return generate_manual_creative_variations(target_prompt)

def extract_prompts_improved(response: str, target: str) -> List[str]:
    """Improved prompt extraction with better parsing"""
    
    variations = []
    lines = response.split('\n')
    
    for line in lines:
        line = line.strip()
        
        # Remove numbering and common prefixes
        for prefix in ['1.', '2.', '3.', '4.', '5.', '-', '•', '*', '1)', '2)', '3)', '4)', '5)']:
            if line.startswith(prefix):
                line = line[len(prefix):].strip()
        
        # Remove quotes if present
        line = line.strip('"\'')
        
        # Check if it's a valid prompt with exact format
        if (line.lower().startswith('wbgmsst,') and 
            line.lower().endswith(', white background') and
            target.lower() in line.lower() and
            50 <= len(line) <= 200):
            variations.append(line)
    
    return list(set(variations))  # Remove duplicates

def fix_invalid_prompts(response: str, target: str) -> List[str]:
    """Try to fix prompts that don't end with ', white background'"""
    
    fixed_variations = []
    lines = response.split('\n')
    
    for line in lines:
        line = line.strip()
        
        # Remove numbering
        for prefix in ['1.', '2.', '3.', '4.', '5.', '-', '•', '*']:
            if line.startswith(prefix):
                line = line[len(prefix):].strip()
        
        # Remove quotes
        line = line.strip('"\'')
        
        # If it starts with wbgmsst and contains target but doesn't end correctly
        if (line.lower().startswith('wbgmsst,') and 
            target.lower() in line.lower() and
            not line.lower().endswith(', white background')):
            
            # Try to fix the ending
            # Remove any existing background reference
            if ', ' in line:
                parts = line.split(', ')
                # Remove last part if it looks like a background
                if any(bg in parts[-1].lower() for bg in ['background', 'backdrop', 'scene']):
                    line = ', '.join(parts[:-1])
            
            # Add correct ending
            fixed_line = line + ', white background'
            
            # Validate the fixed prompt
            if (50 <= len(fixed_line) <= 200 and 
                target.lower() in fixed_line.lower()):
                fixed_variations.append(fixed_line)
                print(f"      🔧 Fixed prompt: {fixed_line}")
    
    return fixed_variations

def generate_manual_creative_variations(target_prompt: str) -> List[str]:
    """Generate creative variations manually as fallback"""
    
    print("   🔧 Generating manual creative variations...")
    
    variations = [
        f"wbgmsst, aerospace-grade precision-engineered {target_prompt}, ultra-high technical specification, white background",
        f"wbgmsst, military-spec ultra-precision {target_prompt}, advanced engineering design, white background",
        f"wbgmsst, defense-grade ultra-detailed {target_prompt}, premium manufacturing excellence, white background",
        f"wbgmsst, masterpiece-quality precision-crafted {target_prompt}, aerospace-engineering excellence, white background",
        f"wbgmsst, laboratory-grade ultra-precision {target_prompt}, scientific-specification excellence, white background"
    ]
    
    return variations

def test_creative_variations(variations: List[str]) -> List[Tuple[str, float]]:
    """Test variations with validation"""
    
    print(f"\n🔬 TESTING {len(variations)} CREATIVE VARIATIONS")
    print("=" * 80)
    
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
            print(f"      ⚠️ Validation failed: {result.stderr[:100]}")
            return 0.0
        
        with open("subnet_validation_results.json", 'r') as f:
            data = json.load(f)
            return data.get("validation_engine_score", 0.0)
    
    except Exception as e:
        print(f"      ❌ Validation error: {e}")
        return 0.0

def analyze_results(results: List[Tuple[str, float]], target: str):
    """Analyze and display results"""
    
    print(f"\n🎓 CREATIVE ANALYSIS: '{target}'")
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
    
    print(f"📊 PERFORMANCE SUMMARY:")
    print(f"   Total tested: {len(results)}")
    print(f"   🏆 Ultra (≥0.96): {ultra_count}")
    print(f"   ✨ High (≥0.80): {high_count}")
    print(f"   📈 Good (≥0.60): {good_count}")
    print(f"   📊 Average: {avg_score:.3f}")
    
    if sorted_results:
        best_prompt, best_score = sorted_results[0]
        print(f"\n🏆 BEST CREATIVE RESULT:")
        print(f"   📊 Score: {best_score:.3f}")
        print(f"   📝 Prompt: {best_prompt}")
        
        if best_score >= 0.96:
            print(f"   🎉 ULTRA ACHIEVEMENT! This pattern should be stored and reused!")
    
    print(f"\n📈 ALL RESULTS (sorted by score):")
    for prompt, score in sorted_results:
        status = "🎉" if score >= 0.96 else "✨" if score >= 0.8 else "📈" if score >= 0.6 else "📊"
        print(f"   {status} {score:.3f}: {prompt}")
    
    # Analyze what works
    print(f"\n🔍 PATTERN ANALYSIS:")
    best_descriptors = []
    for prompt, score in sorted_results[:3]:  # Top 3
        if score >= 0.6:
            # Extract descriptors
            parts = prompt.split(',')
            if len(parts) >= 2:
                descriptor_part = parts[1].strip()
                print(f"   🎯 High-scoring pattern: {descriptor_part} → {score:.3f}")
                best_descriptors.append(descriptor_part)
    
    if best_descriptors:
        print(f"\n💡 SUCCESSFUL PATTERNS DISCOVERED:")
        for desc in best_descriptors:
            print(f"   ✨ {desc}")

def provide_ai_feedback(results: List[Tuple[str, float]]):
    """Provide feedback on AI performance"""
    
    print(f"\n🤖 AI PERFORMANCE FEEDBACK")
    print("=" * 80)
    
    if not results:
        print("❌ No results to provide feedback on")
        return
    
    avg_score = sum(score for _, score in results) / len(results)
    best_score = max(score for _, score in results)
    
    print(f"📊 AI CREATIVITY ASSESSMENT:")
    print(f"   Average Score: {avg_score:.3f}")
    print(f"   Best Score: {best_score:.3f}")
    print(f"   Variations Generated: {len(results)}")
    
    if best_score >= 0.9:
        print(f"   🏆 EXCELLENT: AI generated ultra-high scoring prompts!")
    elif best_score >= 0.8:
        print(f"   ✨ VERY GOOD: AI generated high-scoring prompts!")
    elif best_score >= 0.6:
        print(f"   📈 GOOD: AI generated decent prompts with improvement potential!")
    else:
        print(f"   📊 NEEDS IMPROVEMENT: AI prompts need better optimization!")
    
    # Pattern feedback
    successful_patterns = [prompt for prompt, score in results if score >= 0.7]
    if successful_patterns:
        print(f"\n✅ SUCCESSFUL AI PATTERNS:")
        for prompt in successful_patterns[:3]:
            print(f"   🎯 {prompt}")
    
    # Improvement suggestions
    print(f"\n💡 IMPROVEMENT SUGGESTIONS FOR AI:")
    if avg_score < 0.7:
        print(f"   📝 Focus on proven descriptors: aerospace-grade, defense-grade, ultra-precision")
        print(f"   🔧 Use technical specifications: ultra-high technical specification")
        print(f"   ⚡ Combine authority + process + specification for higher scores")

def main():
    """Main creative testing function with improved feedback"""
    
    print("🚀 IMPROVED CREATIVE AI GENERATOR")
    print("🎯 Mission: Force creative variations and provide AI feedback")
    print("⚡ Using: LLaMA 3.2:3b with improved prompting")
    print("=" * 80)
    
    # Test AI connection
    if not test_ai_connection():
        print("❌ AI connection failed - check if Ollama is running")
        return
    
    print("✅ AI connection confirmed")
    
    test_prompts = [
        "hexagonal prism steel structure",
        "elegant silk fabric draping",
        "transparent glass sphere with reflections"
    ]
    
    all_results = []
    
    for prompt in test_prompts:
        print(f"\n{'='*20} CREATIVE SESSION: {prompt} {'='*20}")
        
        # Generate creative variations
        variations = force_creative_prompt(prompt)
        
        if variations:
            # Test variations
            results = test_creative_variations(variations)
            all_results.extend(results)
            
            # Analyze results
            analyze_results(results, prompt)
        else:
            print("❌ No variations generated")
        
        print(f"\n⏸️ Brief pause...")
        import time
        time.sleep(2)
    
    # Overall AI feedback
    if all_results:
        provide_ai_feedback(all_results)

if __name__ == "__main__":
    main() 