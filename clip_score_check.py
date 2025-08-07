#!/usr/bin/env python3
"""
Simple CLIP Score Check - Command Line Interface
Purpose: Compare CLIP scores between two prompts using the same model as subnet accurate validator
Uses: convnext_large_d model with laion2b_s26b_b102k_augreg weights (production standard)

Usage:
    python clip_score_check.py "prompt1" "prompt2"
    python clip_score_check.py --quality "prompt"
    python clip_score_check.py --compare "prompt1" "prompt2" "prompt3"
"""

import argparse
import sys
from simple_clip_score_check import SimpleCLIPScoreChecker


def main():
    parser = argparse.ArgumentParser(
        description="Simple CLIP Score Check using production validation model",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Compare two prompts
  python clip_score_check.py "a blue vase" "a red vase"
  
  # Check quality score of a single prompt
  python clip_score_check.py --quality "a blue ceramic vase with red trim"
  
  # Compare multiple prompts
  python clip_score_check.py --compare "vase" "blue vase" "red vase" "ceramic vase"
  
  # Compare with reference prompt
  python clip_score_check.py --compare "vase" "blue vase" "red vase" --reference "a blue ceramic vase"
        """
    )
    
    parser.add_argument("prompts", nargs="*", help="Prompts to compare")
    parser.add_argument("--quality", action="store_true", help="Compute quality score for single prompt")
    parser.add_argument("--compare", action="store_true", help="Compare multiple prompts")
    parser.add_argument("--reference", help="Reference prompt for comparison")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    
    args = parser.parse_args()
    
    if not args.prompts:
        parser.print_help()
        return 1
    
    # Initialize checker
    checker = SimpleCLIPScoreChecker(verbose=args.verbose)
    
    try:
        if args.quality:
            # Quality score mode
            if len(args.prompts) != 1:
                print("❌ Quality mode requires exactly one prompt")
                return 1
            
            prompt = args.prompts[0]
            print(f"🔍 Computing quality score for: '{prompt}'")
            
            quality_score = checker.compute_prompt_quality_score(prompt)
            print(f"\n📊 Quality Score: {quality_score:.4f}")
            
            # Interpretation
            if quality_score > 0.8:
                interpretation = "Excellent"
            elif quality_score > 0.6:
                interpretation = "Good"
            elif quality_score > 0.4:
                interpretation = "Fair"
            else:
                interpretation = "Poor"
            
            print(f"🎯 Interpretation: {interpretation}")
            
        elif args.compare:
            # Multiple comparison mode
            if len(args.prompts) < 2:
                print("❌ Compare mode requires at least 2 prompts")
                return 1
            
            print(f"🔍 Comparing {len(args.prompts)} prompts...")
            if args.reference:
                print(f"📋 Reference prompt: '{args.reference}'")
            
            results = checker.compare_multiple_prompts(
                prompts=args.prompts,
                reference_prompt=args.reference
            )
            
            # Summary
            print(f"\n📋 SUMMARY:")
            print(f"   Best quality: {max(results['quality_scores'].items(), key=lambda x: x[1])[0]}")
            
            if args.reference:
                most_similar = max(results['reference_similarities'].items(), key=lambda x: x[1])
                print(f"   Most similar to reference: '{most_similar[0]}' (score: {most_similar[1]:.4f})")
            
            # Find most similar pair
            if results['pairwise_similarities']:
                most_similar_pair = max(results['pairwise_similarities'].items(), key=lambda x: x[1])
                print(f"   Most similar pair: {most_similar_pair[0]} (score: {most_similar_pair[1]:.4f})")
            
        else:
            # Simple two-prompt comparison
            if len(args.prompts) != 2:
                print("❌ Simple comparison requires exactly 2 prompts")
                return 1
            
            prompt1, prompt2 = args.prompts
            print(f"🔍 Comparing prompts:")
            print(f"   Prompt 1: '{prompt1}'")
            print(f"   Prompt 2: '{prompt2}'")
            
            similarity = checker.compute_text_similarity(prompt1, prompt2)
            print(f"\n📊 Similarity Score: {similarity:.4f}")
            
            # Interpretation
            if similarity >= 0.9:
                interpretation = "Very High Similarity"
            elif similarity >= 0.7:
                interpretation = "High Similarity"
            elif similarity >= 0.5:
                interpretation = "Moderate Similarity"
            elif similarity >= 0.3:
                interpretation = "Low Similarity"
            else:
                interpretation = "Very Low Similarity"
            
            print(f"🎯 Interpretation: {interpretation}")
            
            # Also show quality scores
            print(f"\n📊 Quality Scores:")
            quality1 = checker.compute_prompt_quality_score(prompt1)
            quality2 = checker.compute_prompt_quality_score(prompt2)
            print(f"   Prompt 1: {quality1:.4f}")
            print(f"   Prompt 2: {quality2:.4f}")
        
        return 0
        
    except KeyboardInterrupt:
        print("\n⚠️ Interrupted by user")
        return 1
    except Exception as e:
        print(f"❌ Error: {e}")
        return 1
    finally:
        # Cleanup
        checker.unload_model()


if __name__ == "__main__":
    sys.exit(main()) 