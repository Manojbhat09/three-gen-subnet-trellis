#!/usr/bin/env python3
"""
CLIP Alignment Score Demonstration
Purpose: Show practical examples of how CLIP alignment scores work
"""

import torch
import numpy as np
from simple_clip_score_check import SimpleCLIPScoreChecker


def demonstrate_clip_alignment():
    """Demonstrate CLIP alignment score concepts"""
    
    print("🎯 CLIP ALIGNMENT SCORE DEMONSTRATION")
    print("=" * 60)
    
    # Initialize checker
    checker = SimpleCLIPScoreChecker(verbose=False)
    
    try:
        # Example 1: Basic Object Recognition
        print("\n📊 EXAMPLE 1: Basic Object Recognition")
        print("-" * 40)
        
        base_prompt = "a vase"
        variations = [
            "a vase",
            "a blue vase", 
            "a red vase",
            "a ceramic vase",
            "a glass vase",
            "a metal vase"
        ]
        
        print(f"Base prompt: '{base_prompt}'")
        print("Similarity scores:")
        
        for variation in variations:
            similarity = checker.compute_text_similarity(base_prompt, variation)
            print(f"   '{base_prompt}' ↔ '{variation}': {similarity:.4f}")
        
        # Example 2: Semantic Understanding
        print("\n📊 EXAMPLE 2: Semantic Understanding")
        print("-" * 40)
        
        target_prompt = "a blue ceramic vase with red trim"
        test_prompts = [
            "a blue ceramic vase with red trim",  # Exact match
            "a blue ceramic vase",                # Missing detail
            "a red ceramic vase with blue trim",  # Swapped colors
            "a blue glass vase with red trim",    # Wrong material
            "a blue ceramic cup with red trim",   # Wrong object
            "a wooden table"                      # Completely different
        ]
        
        print(f"Target prompt: '{target_prompt}'")
        print("Alignment scores:")
        
        for test_prompt in test_prompts:
            similarity = checker.compute_text_similarity(target_prompt, test_prompt)
            
            # Interpretation
            if similarity >= 0.9:
                interpretation = "Perfect Match"
            elif similarity >= 0.7:
                interpretation = "High Alignment"
            elif similarity >= 0.5:
                interpretation = "Moderate Alignment"
            elif similarity >= 0.3:
                interpretation = "Low Alignment"
            else:
                interpretation = "Poor Alignment"
            
            print(f"   '{test_prompt}': {similarity:.4f} ({interpretation})")
        
        # Example 3: Prompt Engineering Impact
        print("\n📊 EXAMPLE 3: Prompt Engineering Impact")
        print("-" * 40)
        
        base_concept = "a blue ceramic vase"
        engineering_levels = [
            base_concept,
            f"{base_concept} with red trim",
            f"{base_concept} with red trim, professional 3D render",
            f"{base_concept} with red trim, professional 3D render, highly detailed, photorealistic",
            f"professional 3D render, Create 3D game asset, isometric view, {base_concept} with red trim, highly detailed, photorealistic, studio lighting, clean white background"
        ]
        
        print("Prompt engineering progression:")
        for i, prompt in enumerate(engineering_levels):
            quality_score = checker.compute_prompt_quality_score(prompt)
            print(f"   Level {i+1}: {quality_score:.4f} - '{prompt}'")
        
        # Example 4: Cross-Modal Understanding
        print("\n📊 EXAMPLE 4: Cross-Modal Understanding")
        print("-" * 40)
        
        target = "a blue ceramic vase"
        synonyms = [
            "a blue ceramic urn",
            "a blue ceramic pot",
            "a blue ceramic container",
            "a blue ceramic vessel",
            "a blue ceramic jar"
        ]
        
        print(f"Target: '{target}'")
        print("Synonym similarity scores:")
        
        for synonym in synonyms:
            similarity = checker.compute_text_similarity(target, synonym)
            print(f"   '{target}' ↔ '{synonym}': {similarity:.4f}")
        
        # Example 5: Material and Attribute Understanding
        print("\n📊 EXAMPLE 5: Material and Attribute Understanding")
        print("-" * 40)
        
        base_object = "vase"
        materials = ["ceramic", "glass", "metal", "wooden", "plastic"]
        colors = ["blue", "red", "green", "transparent", "white"]
        
        print("Material understanding (with 'blue' attribute):")
        for material in materials:
            prompt = f"a blue {material} {base_object}"
            quality = checker.compute_prompt_quality_score(prompt)
            print(f"   '{prompt}': {quality:.4f}")
        
        print("\nColor understanding (with 'ceramic' material):")
        for color in colors:
            prompt = f"a {color} ceramic {base_object}"
            quality = checker.compute_prompt_quality_score(prompt)
            print(f"   '{prompt}': {quality:.4f}")
        
        # Example 6: Production Validation Simulation
        print("\n📊 EXAMPLE 6: Production Validation Simulation")
        print("-" * 40)
        
        # Simulate different generation qualities
        test_cases = [
            {
                "prompt": "a blue ceramic vase with red trim",
                "generated_prompts": [
                    "a blue ceramic vase with red trim",  # Perfect
                    "a blue ceramic vase",                # Good
                    "a red ceramic vase",                 # Poor
                    "a wooden table"                      # Fail
                ]
            }
        ]
        
        for case in test_cases:
            original_prompt = case["prompt"]
            print(f"Original prompt: '{original_prompt}'")
            print("Generation quality assessment:")
            
            for i, generated_prompt in enumerate(case["generated_prompts"]):
                alignment_score = checker.compute_text_similarity(original_prompt, generated_prompt)
                
                # Apply production normalization
                normalized_score = alignment_score / 0.35
                
                # Determine if it passes validation
                if normalized_score < 0.3:
                    status = "❌ FAIL"
                    task_fidelity = 0.0
                elif normalized_score >= 0.8:
                    status = "✅ EXCELLENT"
                    task_fidelity = 1.0
                elif normalized_score >= 0.6:
                    status = "🟡 GOOD"
                    task_fidelity = 0.75
                else:
                    status = "🟠 POOR"
                    task_fidelity = 0.0
                
                print(f"   Generation {i+1}: {normalized_score:.4f} {status} (Task Fidelity: {task_fidelity})")
                print(f"      Generated: '{generated_prompt}'")
        
        print("\n🎯 KEY INSIGHTS:")
        print("1. CLIP understands semantic relationships, not just exact matches")
        print("2. Prompt engineering can significantly impact quality scores")
        print("3. Material and attribute understanding is robust")
        print("4. Production validation uses normalized scores with thresholds")
        print("5. Task fidelity depends on alignment score thresholds")
        
    finally:
        checker.unload_model()


def demonstrate_threshold_analysis():
    """Demonstrate how different thresholds affect validation"""
    
    print("\n\n🔬 THRESHOLD ANALYSIS")
    print("=" * 60)
    
    checker = SimpleCLIPScoreChecker(verbose=False)
    
    try:
        # Test various prompt pairs
        test_pairs = [
            ("a blue vase", "a blue ceramic vase"),
            ("a blue vase", "a red vase"),
            ("a blue vase", "a blue cup"),
            ("a blue vase", "a wooden table"),
            ("a blue ceramic vase with red trim", "a blue ceramic vase"),
            ("a blue ceramic vase with red trim", "a red ceramic vase with blue trim"),
            ("a blue ceramic vase with red trim", "a blue glass vase with red trim"),
        ]
        
        print("Alignment Score Analysis:")
        print("Score Range | Interpretation | Validation Status")
        print("-" * 50)
        
        for prompt1, prompt2 in test_pairs:
            score = checker.compute_text_similarity(prompt1, prompt2)
            normalized_score = score / 0.35
            
            if normalized_score >= 0.9:
                interpretation = "Very High"
                status = "✅ Excellent"
            elif normalized_score >= 0.7:
                interpretation = "High"
                status = "✅ Good"
            elif normalized_score >= 0.5:
                interpretation = "Moderate"
                status = "🟡 Acceptable"
            elif normalized_score >= 0.3:
                interpretation = "Low"
                status = "🟠 Poor"
            else:
                interpretation = "Very Low"
                status = "❌ Fail"
            
            print(f"{normalized_score:.3f}      | {interpretation:12} | {status}")
            print(f"           | '{prompt1}' ↔ '{prompt2}'")
            print()
        
    finally:
        checker.unload_model()


if __name__ == "__main__":
    demonstrate_clip_alignment()
    demonstrate_threshold_analysis() 