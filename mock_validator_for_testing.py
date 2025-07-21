#!/usr/bin/env python3
"""
Mock Validator for Testing v6.5 Optimizer
Purpose: Simulates realistic validation scores for testing when pyspz library is not available
"""
import json
import random
import hashlib
import sys

def mock_validation(prompt: str) -> dict:
    """Generate realistic mock validation scores based on prompt quality"""
    
    # Create deterministic but varying scores based on prompt hash
    prompt_hash = int(hashlib.md5(prompt.encode()).hexdigest()[:8], 16)
    random.seed(prompt_hash)
    
    # Base score influenced by prompt characteristics
    base_score = 0.4 + random.random() * 0.4  # 0.4 to 0.8 base range
    
    # Quality indicators that boost scores
    quality_indicators = [
        "wbgmsst", "3d", "model", "detailed", "high quality", "professional",
        "ultra", "precision", "masterpiece", "perfect", "white background",
        "photorealistic", "technical", "engineering", "CAD", "accurate"
    ]
    
    quality_boost = 0.0
    for indicator in quality_indicators:
        if indicator.lower() in prompt.lower():
            quality_boost += 0.02  # 2% boost per quality indicator
    
    # Length bonus (longer, more detailed prompts tend to score better)
    length_bonus = min(0.1, len(prompt) / 500)  # Up to 10% bonus for detailed prompts
    
    # Specific pattern bonuses
    pattern_bonus = 0.0
    if "ultra-" in prompt.lower():
        pattern_bonus += 0.05
    if "precision" in prompt.lower():
        pattern_bonus += 0.03
    if "masterpiece" in prompt.lower():
        pattern_bonus += 0.04
    if "technical" in prompt.lower() and any(word in prompt.lower() for word in ["cad", "engineering", "blueprint"]):
        pattern_bonus += 0.06
    
    # Calculate final score
    final_score = base_score + quality_boost + length_bonus + pattern_bonus
    
    # Add some randomness but keep it reasonable
    final_score += (random.random() - 0.5) * 0.1  # ±5% randomness
    
    # Clamp to valid range
    final_score = max(0.0, min(1.0, final_score))
    
    # Ultra scores are rare but possible with high-quality prompts
    if quality_boost > 0.1 and pattern_bonus > 0.05 and random.random() < 0.3:
        final_score = max(final_score, 0.85 + random.random() * 0.15)  # Chance for 0.85-1.0
    
    # Demo fidelity (usually close to validation score)
    demo_fidelity = final_score + (random.random() - 0.5) * 0.1
    demo_fidelity = max(0.0, min(1.0, demo_fidelity))
    
    return {
        "validation_engine_score": final_score,
        "demo_fidelity_score": demo_fidelity,
        "mock_testing": True,
        "prompt_analyzed": prompt,
        "quality_indicators_found": [ind for ind in quality_indicators if ind.lower() in prompt.lower()],
        "quality_boost": quality_boost,
        "pattern_bonus": pattern_bonus
    }

def main():
    """Mock validator main function"""
    if len(sys.argv) < 2:
        print("Usage: python mock_validator_for_testing.py <prompt>")
        sys.exit(1)
    
    prompt = sys.argv[1]
    
    print(f"🧪 MOCK VALIDATOR - Testing Mode")
    print(f"📝 Prompt: '{prompt}'")
    
    # Generate mock results
    results = mock_validation(prompt)
    
    print(f"📊 Mock Score: {results['validation_engine_score']:.3f}")
    print(f"🎭 Demo Fidelity: {results['demo_fidelity_score']:.3f}")
    print(f"🏆 Quality Indicators: {len(results['quality_indicators_found'])}")
    
    # Save results in expected format
    with open("subnet_validation_results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"💾 Results saved to subnet_validation_results.json")

if __name__ == "__main__":
    main() 