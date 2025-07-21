#!/usr/bin/env python3
"""
Simple Prompt Generator with DeepSeek Variations
==============================================
Use DeepSeek to generate correctly formatted prompts with strict validation
"""

import subprocess
import sys
import json
import time
import random
import requests
import re

def run_validation(prompt: str):
    """Run validation and return score"""
    try:
        cmd = [sys.executable, "subnet_accurate_validator.py", prompt]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        
        if result.returncode != 0:
            print(f"   ⚠️ Validation failed: {result.stderr}")
            return 0.0
        
        with open("subnet_validation_results.json", 'r') as f:
            data = json.load(f)
            return data.get("validation_engine_score", 0.0)
    
    except Exception as e:
        print(f"   ❌ Validation error: {e}")
        return 0.0

def query_deepseek_for_variations(target_prompt: str, num_variations: int = 15):
    """Use DeepSeek to generate prompt variations with full context"""
    
    variations = []
    used_prompts = set()
    
    # Variation strategies to prevent repetition
    variation_strategies = [
        "Focus on aerospace engineering excellence",
        "Emphasize military-grade precision manufacturing", 
        "Highlight industrial-grade technical perfection",
        "Prioritize laboratory-grade scientific accuracy",
        "Stress defense-grade structural integrity",
        "Feature aviation-standard quality assurance",
        "Showcase precision-engineering mastery",
        "Demonstrate ultra-precision craftsmanship",
        "Display masterpiece-quality artisanship",
        "Present technical-perfection standards"
    ]
    
    for i in range(num_variations):
        print(f"🤖 Generating variation {i+1}/{num_variations}...")
        
        # Add variation strategy to prevent repetition
        strategy = variation_strategies[i % len(variation_strategies)]
        
        # Give full context about what we're doing
        request = f"""CONTEXT: We are optimizing prompts for a 3D model generation AI system. The system takes text prompts and generates 3D models. We need to achieve validation scores of 0.96+ (out of 1.0) by crafting the perfect prompt.

TARGET OBJECT: {target_prompt}
VARIATION STRATEGY: {strategy}

PROMPT FORMAT REQUIREMENTS (CRITICAL):
- Must start with EXACTLY: "wbgmsst, "
- Must end with EXACTLY: ", white background"
- Must include the target object: "{target_prompt}"
- Length should be 80-150 characters
- Use premium technical descriptors for higher scores

SCORING SYSTEM:
- Premium descriptors increase scores: aerospace-grade, ultra-precision, masterpiece-quality, precision-engineered
- Technical terms help: military-spec, defense-grade, aviation-standard, laboratory-grade
- Process terms boost scores: ultra-detailed, technical-perfection, precision-forged, ultra-refined
- Specification terms: ultra-high technical specification, advanced engineering design, premium manufacturing excellence

EXAMPLES OF HIGH-SCORING PATTERNS:
- "wbgmsst, aerospace-grade precision-engineered [object], ultra-high technical specification, white background"
- "wbgmsst, military-spec ultra-precision [object], masterpiece-quality rendering, white background"
- "wbgmsst, defense-grade ultra-detailed [object], advanced engineering design, white background"

VARIATION #{i+1} REQUIREMENTS:
- Create a UNIQUE prompt different from previous attempts
- Apply the variation strategy: {strategy}
- Use different descriptor combinations
- Be creative while maintaining technical excellence

YOUR TASK: Create a single optimized prompt for "{target_prompt}" that will score 0.96+. Focus on {strategy.lower()}.

RESPOND WITH ONLY THE PROMPT:"""

        data = {
            "model": "deepseek-r1:1.5b",
            "messages": [{"role": "user", "content": request}],
            "stream": False,
            "options": {
                "temperature": 0.9 + (i * 0.02),  # Increase temperature slightly for each variation
                "top_p": 0.9,
                "num_predict": 300,
                "repeat_penalty": 1.3 + (i * 0.02)  # Increase repeat penalty to avoid repetition
            }
        }
        
        try:
            response = requests.post("http://localhost:11434/api/chat", json=data, timeout=30)
            response.raise_for_status()
            content = response.json()["message"]["content"].strip()
            
            # Clean and validate the response
            cleaned_prompt = clean_deepseek_response(content, target_prompt)
            
            if validate_prompt_format(cleaned_prompt, target_prompt) and cleaned_prompt not in used_prompts:
                variations.append(cleaned_prompt)
                used_prompts.add(cleaned_prompt)
                print(f"   ✅ Valid: {cleaned_prompt}")
            else:
                # Fallback to manual generation for this variation
                fallback = generate_manual_fallback(target_prompt)
                # Ensure fallback is also unique
                attempt_count = 0
                while fallback in used_prompts and attempt_count < 10:
                    fallback = generate_manual_fallback(target_prompt)
                    attempt_count += 1
                
                variations.append(fallback)
                used_prompts.add(fallback)
                print(f"   🔄 Invalid/duplicate, using fallback: {fallback}")
                
        except Exception as e:
            print(f"   ❌ Error: {e}")
            # Fallback to manual generation
            fallback = generate_manual_fallback(target_prompt)
            # Ensure fallback is also unique
            attempt_count = 0
            while fallback in used_prompts and attempt_count < 10:
                fallback = generate_manual_fallback(target_prompt)
                attempt_count += 1
                
            variations.append(fallback)
            used_prompts.add(fallback)
            print(f"   🔄 Error fallback: {fallback}")
        
        time.sleep(0.3)  # Brief pause between requests
    
    return variations

def clean_deepseek_response(response: str, target_prompt: str) -> str:
    """Clean DeepSeek response to extract valid prompt"""
    
    # Remove thinking sections
    if '<think>' in response and '</think>' in response:
        response = re.sub(r'<think>.*?</think>', '', response, flags=re.DOTALL)
    
    # Remove common prefixes and formatting
    response = re.sub(r'^["\']|["\']$', '', response.strip())
    response = re.sub(r'^\*\*|\*\*$', '', response)
    response = re.sub(r'^(here\'s|here is|prompt:|result:)', '', response, flags=re.IGNORECASE)
    
    # Look for lines that start with wbgmsst
    lines = response.split('\n')
    for line in lines:
        line = line.strip()
        if line.lower().startswith('wbgmsst,'):
            return line
    
    # If no valid line found, try to fix the response
    if target_prompt.lower() in response.lower():
        # Try to construct a valid prompt from the response
        return f"wbgmsst, aerospace-grade precision-engineered {target_prompt}, ultra-high technical specification, white background"
    
    # Ultimate fallback
    return generate_manual_fallback(target_prompt)

def validate_prompt_format(prompt: str, target_prompt: str) -> bool:
    """Validate that the prompt has correct format"""
    return (
        prompt.startswith("wbgmsst, ") and 
        prompt.endswith(", white background") and
        target_prompt.lower() in prompt.lower() and
        len(prompt) >= 50 and
        len(prompt) <= 150
    )

def generate_manual_fallback(target_prompt: str) -> str:
    """Generate a manual fallback prompt when DeepSeek fails"""
    
    authority_terms = [
        "aerospace-grade", "military-spec", "industrial-grade", "defense-grade",
        "aviation-standard", "precision-manufacturing"
    ]
    
    process_terms = [
        "ultra-precision", "precision-engineered", "masterpiece-quality", 
        "ultra-detailed", "technical-perfection", "precision-forged"
    ]
    
    spec_terms = [
        "ultra-high technical specification", "advanced engineering design",
        "premium manufacturing excellence", "precision-crafted components"
    ]
    
    authority = random.choice(authority_terms)
    process = random.choice(process_terms)
    spec = random.choice(spec_terms)
    
    return f"wbgmsst, {authority} {process} {target_prompt}, {spec}, white background"

def generate_prompt_variations(target_prompt: str):
    """Generate prompt variations using DeepSeek with fallbacks"""
    
    print(f"🎯 Generating variations for: '{target_prompt}'")
    print("🤖 Using DeepSeek AI with manual fallbacks...")
    
    variations = query_deepseek_for_variations(target_prompt, 15)
    
    # Remove duplicates while preserving order
    unique_variations = []
    seen = set()
    for prompt in variations:
        if prompt not in seen:
            unique_variations.append(prompt)
            seen.add(prompt)
    
    print(f"✅ Generated {len(unique_variations)} unique variations")
    return unique_variations

def test_prompt_variations(target_prompt: str):
    """Test multiple prompt variations and find the best"""
    
    print(f"🎯 TESTING PROMPT VARIATIONS FOR: '{target_prompt}'")
    print("=" * 80)
    
    variations = generate_prompt_variations(target_prompt)
    results = []
    
    for i, prompt in enumerate(variations, 1):
        print(f"\n🔄 TESTING VARIATION {i}/{len(variations)}:")
        print(f"📝 Prompt: {prompt}")
        
        # Validate format
        format_valid = validate_prompt_format(prompt, target_prompt)
        
        if not format_valid:
            print(f"❌ FORMAT ERROR - Skipping")
            continue
        
        print("🔧 Validating...")
        score = run_validation(prompt)
        
        results.append({
            'variation': i,
            'prompt': prompt,
            'score': score,
            'length': len(prompt)
        })
        
        print(f"📊 Score: {score:.3f}")
        
        if score >= 0.96:
            print(f"🏆 ULTRA ACHIEVEMENT! Found perfect prompt!")
            break
        
        time.sleep(1)
    
    # Analyze results
    if results:
        best = max(results, key=lambda x: x['score'])
        avg_score = sum(r['score'] for r in results) / len(results)
        
        print(f"\n📊 RESULTS ANALYSIS:")
        print(f"   Total variations tested: {len(results)}")
        print(f"   Best score: {best['score']:.3f}")
        print(f"   Average score: {avg_score:.3f}")
        print(f"   Best prompt: '{best['prompt']}'")
        print(f"   Best prompt length: {best['length']} characters")
        
        # Show all results
        print(f"\n📈 ALL RESULTS:")
        for result in sorted(results, key=lambda x: x['score'], reverse=True):
            print(f"   {result['score']:.3f}: {result['prompt'][:80]}{'...' if len(result['prompt']) > 80 else ''}")
    
    return results

def main():
    """Test with different targets"""
    
    test_prompts = [
        "hexagonal prism steel structure",
        "elegant silk fabric draping",
        "transparent glass sphere with reflections",
        "ornate wooden sculpture"
    ]
    
    print("🚀 DEEPSEEK PROMPT GENERATOR - AI VARIATIONS")
    print("=" * 80)
    print("🎯 Goal: Find high-scoring prompts using DeepSeek AI")
    print("⚡ Strategy: AI generation with strict format validation")
    print("=" * 80)
    
    all_results = {}
    
    for i, prompt in enumerate(test_prompts, 1):
        print(f"\n🎯 TESTING TARGET {i}/{len(test_prompts)}: '{prompt}'")
        results = test_prompt_variations(prompt)
        all_results[prompt] = results
        
        if i < len(test_prompts):
            print(f"\n⏸️ Brief pause before next test...")
            time.sleep(3)
    
    # Final summary
    print(f"\n🎓 FINAL SUMMARY")
    print("=" * 80)
    
    for target, results in all_results.items():
        if results:
            best = max(results, key=lambda x: x['score'])
            print(f"📝 {target}: Best score {best['score']:.3f}")
            print(f"   🏆 {best['prompt']}")
        else:
            print(f"📝 {target}: No valid results")
    
    print("\n🚀 DeepSeek prompt generation complete!")

if __name__ == "__main__":
    main() 