#!/usr/bin/env python3
"""
Simple Example: CLIP Score Comparison Between Two 3D Model Generations

This script demonstrates the basic usage pattern:
1. Generate 3D models from original and cleaned prompts
2. Extract images from both generations
3. Compute CLIP alignment scores
4. Compare the results

Usage:
    python example_clip_comparison.py "original prompt" "cleaned prompt"
"""

import sys
from typing import Tuple, Optional
from PIL import Image

# Import the TrellisGenerator class
from trellis_subnit_server_mix_lora_flash import TrellisGenerator

# Import CLIP alignment utilities
from clip_alignment_with_generation import CLIPAlignmentWithGeneration

def generate_and_compare(original_prompt: str, cleaned_prompt: str, seed: int = 42) -> None:
    """Generate 3D models from both prompts and compare CLIP scores."""
    
    print(f"🧪 CLIP Score Comparison Example")
    print(f"=" * 50)
    print(f"📝 Original: '{original_prompt}'")
    print(f"📝 Cleaned:  '{cleaned_prompt}'")
    print(f"🎲 Seed: {seed}")
    print(f"=" * 50)
    
    try:
        # Step 1: Initialize TrellisGenerator
        print(f"\n🚀 Initializing TrellisGenerator...")
        generator = TrellisGenerator()
        print(f"✅ TrellisGenerator ready")
        
        # Step 2: Generate 3D models from both prompts
        print(f"\n🎨 Generating 3D models...")
        
        print(f"   Generating from original prompt...")
        result1 = generator.generate_3d_model_image(
            original_prompt, seed, 
            num_inference_steps=7,
            guidance_scale=3.5,
            ss_sampling_steps=21,
            slat_sampling_steps=24,
            slat_guidance_strength=4.0,
            ss_guidance_strength=9.5
        )
        
        if result1 is None:
            print(f"❌ Original prompt generation failed")
            return
        
        ply_data1, compressed_data1, image1 = result1
        print(f"✅ Original generation completed")
        print(f"   PLY size: {len(ply_data1):,} bytes")
        
        print(f"   Generating from cleaned prompt...")
        result2 = generator.generate_3d_model_image(
            cleaned_prompt, seed,
            num_inference_steps=7,
            guidance_scale=3.5,
            ss_sampling_steps=21,
            slat_sampling_steps=24,
            slat_guidance_strength=4.0,
            ss_guidance_strength=9.5
        )
        
        if result2 is None:
            print(f"❌ Cleaned prompt generation failed")
            return
        
        ply_data2, compressed_data2, image2 = result2
        print(f"✅ Cleaned generation completed")
        print(f"   PLY size: {len(ply_data2):,} bytes")
        
        # Step 3: Images are already PIL Image objects, no conversion needed
        print(f"\n🖼️ Images ready for CLIP analysis...")
        pil_image1 = image1
        pil_image2 = image2
        print(f"✅ Images ready for analysis")
        print(f"   Image 1 size: {pil_image1.size}")
        print(f"   Image 2 size: {pil_image2.size}")
        
        print(f"   Computing CLIP scores...")
        
        # Step 4: Initialize CLIP analyzer and compute scores
        print(f"\n🎯 Computing CLIP alignment scores...")
        clip_analyzer = CLIPAlignmentWithGeneration()
        
        # Load CLIP model before computing scores
        print(f"   Loading CLIP model...")
        clip_analyzer.load_clip_model()
        
        # Compute all possible CLIP score combinations
        score_original_original = clip_analyzer.compute_clip_alignment_score(original_prompt, pil_image1)
        score_cleaned_cleaned = clip_analyzer.compute_clip_alignment_score(cleaned_prompt, pil_image2)
        score_original_cleaned = clip_analyzer.compute_clip_alignment_score(original_prompt, pil_image2)
        score_cleaned_original = clip_analyzer.compute_clip_alignment_score(cleaned_prompt, pil_image1)
        
        print(f"✅ CLIP scores computed successfully")
        
        # Step 5: Display results
        print(f"\n📊 CLIP ALIGNMENT SCORE RESULTS")
        print(f"=" * 50)
        print(f"🎯 DIRECT MATCHES:")
        print(f"   Original prompt + Original image: {score_original_original:.4f}")
        print(f"   Cleaned prompt + Cleaned image:  {score_cleaned_cleaned:.4f}")
        print(f"")
        print(f"🔍 CROSS-MATCHES:")
        print(f"   Original prompt + Cleaned image: {score_original_cleaned:.4f}")
        print(f"   Cleaned prompt + Original image: {score_cleaned_original:.4f}")
        print(f"")
        print(f"📈 ANALYSIS:")
        
        # Compare direct matches
        direct_delta = score_cleaned_cleaned - score_original_original
        direct_improvement = ((score_cleaned_cleaned - score_original_original) / score_original_original * 100) if score_original_original > 0 else 0
        
        print(f"   Direct match improvement: {direct_delta:+.4f} ({direct_improvement:+.1f}%)")
        
        # Check if cleaned prompt is better
        if score_cleaned_cleaned > score_original_original:
            print(f"   ✅ Cleaned prompt produces better CLIP alignment")
        elif score_cleaned_cleaned < score_original_original:
            print(f"   ❌ Original prompt produces better CLIP alignment")
        else:
            print(f"   🟡 Both prompts produce similar CLIP alignment")
        
        # Check cross-compatibility
        print(f"")
        print(f"🔄 CROSS-COMPATIBILITY:")
        if score_original_cleaned > score_original_original * 0.9:
            print(f"   ✅ Cleaned image works well with original prompt")
        else:
            print(f"   ⚠️ Cleaned image doesn't align well with original prompt")
            
        if score_cleaned_original > score_cleaned_cleaned * 0.9:
            print(f"   ✅ Original image works well with cleaned prompt")
        else:
            print(f"   ⚠️ Original image doesn't align well with cleaned prompt")
        
        print(f"=" * 50)
        
        # Step 6: Save results summary
        results_summary = {
            "original_prompt": original_prompt,
            "cleaned_prompt": cleaned_prompt,
            "seed": seed,
            "clip_scores": {
                "original_prompt_original_image": score_original_original,
                "cleaned_prompt_cleaned_image": score_cleaned_cleaned,
                "original_prompt_cleaned_image": score_original_cleaned,
                "cleaned_prompt_original_image": score_cleaned_original
            },
            "analysis": {
                "direct_improvement": direct_improvement,
                "cleaned_prompt_better": score_cleaned_cleaned > score_original_original,
                "cross_compatibility_original": score_original_cleaned > score_original_original * 0.9,
                "cross_compatibility_cleaned": score_cleaned_original > score_cleaned_cleaned * 0.9
            }
        }
        
        # Save to file
        import json
        output_file = f"clip_comparison_example_{seed}.json"
        with open(output_file, 'w') as f:
            json.dump(results_summary, f, indent=2)
        print(f"💾 Results summary saved to {output_file}")
        
    except Exception as e:
        print(f"❌ Error during generation or comparison: {e}")
        import traceback
        traceback.print_exc()

def main():
    """Main entry point."""
    if len(sys.argv) < 3:
        print("Usage: python example_clip_comparison.py 'original prompt' 'cleaned prompt' [seed]")
        print("Example: python example_clip_comparison.py 'a red car' 'a red sports car on road' 42")
        sys.exit(1)
    
    original_prompt = sys.argv[1]
    cleaned_prompt = sys.argv[2]
    seed = int(sys.argv[3]) if len(sys.argv) > 3 else 42
    
    print(f"🚀 Starting CLIP comparison example...")
    print(f"   Original prompt: '{original_prompt}'")
    print(f"   Cleaned prompt: '{cleaned_prompt}'")
    print(f"   Seed: {seed}")
    
    try:
        generate_and_compare(original_prompt, cleaned_prompt, seed)
        print(f"\n✅ Example completed successfully!")
        
    except KeyboardInterrupt:
        print(f"\n⏹️ Example interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
