#!/usr/bin/env python3
"""
CLIP Alignment and Validation Score Comparison Script with Reproducibility Integration

This script compares two 3D model generations by:
1. Using the reproducibility system to automatically generate optimized prompts
2. Generating 3D models from original and optimized prompts using TrellisGenerator
3. Computing CLIP alignment scores between prompts and generated images
4. Running validation on both PLY files to get validation scores
5. Comparing all scores to assess the effectiveness of prompt optimization

Usage:
    python test_clip_validation_comparison.py "original prompt" [options]

CUBLAS_WORKSPACE_CONFIG=:4096:8 CUDA_VISIBLE_DEVICES=4 python test_clip_validation_comparison.py "bronze machine gun mount" --log-count 15 --endpoint "generate/cinema" --port 8097 
"""

import argparse
import json
import os
import sys
import time
from typing import Dict, Any, Optional, Tuple
from pathlib import Path

# Import the TrellisGenerator class
from trellis_subnit_server_mix_lora_flash import TrellisGenerator

# Import CLIP alignment utilities
from clip_alignment_with_generation import CLIPAlignmentWithGeneration

# Import the reproducibility system
from llm_close_prompt_reproducibility_test import LLMClosePromptReproducibility

# Import the exact functions from the continuous orchestrator for gold prompts
# Note: Import moved inside function to avoid argument parser conflicts

def get_gold_prompts_from_orchestrator(log_count: int = 7) -> Dict[str, Any]:
    """
    Use the EXACT same functions from the continuous orchestrator to get gold prompts.
    This ensures we're measuring the exact same performance.
    
    Args:
        log_count: Number of recent logs to parse (default: 7)
        
    Returns:
        Dictionary of gold prompts in the exact same format as the orchestrator
    """
    print(f"📚 Using EXACT orchestrator functions to get gold prompts from last {log_count} logs...")
    
    # Import here to avoid argument parser conflicts
    from continuous_trellis_orchestrator_lora_working import ContinuousTrellisOrchestrator
    
    # Create a minimal orchestrator instance just for the gold prompt functions
    # We only need the config for the gold prompt parsing functions
    minimal_config = {
        'activate_learning': True,
        'only_log_learning': log_count,
        'log_learning_count': log_count,
        'max_logs_to_parse': log_count,
        'use_vllm': True,
        'vllm_url': 'http://localhost:9002',
        'vllm_model': 'llama-3-2-3b-it'
    }
    
    try:
        # Create orchestrator instance
        orchestrator = ContinuousTrellisOrchestrator(minimal_config)
        
        # Use the EXACT same function the orchestrator uses
        print(f"🔄 Calling orchestrator.parse_current_episode_logs() with {log_count} logs...")
        log_prompts = orchestrator.parse_current_episode_logs()
        
        print(f"📊 Parsed {len(log_prompts)} prompts from logs")
        
        # Convert to the format expected by the reproducibility system
        gold_standard_results = {}
        
        for prompt, data in log_prompts.items():
            if 'best_score' in data and data['best_score'] > 0:
                # Create the method_2_hybrid_example structure that reproducibility system expects
                gold_standard_results[prompt] = {
                    "method_2_hybrid_example": {
                        "optimized_prompt": data.get('optimized_prompt', prompt),
                        "validation_results": {
                            "validation_engine_score": data['best_score']
                        }
                    }
                }
        
        print(f"✅ Converted {len(gold_standard_results)} prompts to gold standard format")
        
        # Show top scoring prompts
        if gold_standard_results:
            top_prompts = sorted(
                gold_standard_results.items(),
                key=lambda x: x[1]['method_2_hybrid_example']['validation_results']['validation_engine_score'],
                reverse=True
            )[:5]
            
            print(f"🏆 Top scoring prompts from logs:")
            for i, (prompt, data) in enumerate(top_prompts, 1):
                score = data['method_2_hybrid_example']['validation_results']['validation_engine_score']
                print(f"   {i}. Score {score:.4f}: '{prompt[:60]}...'")
        
        return gold_standard_results
        
    except Exception as e:
        print(f"❌ Failed to get gold prompts from orchestrator: {e}")
        import traceback
        traceback.print_exc()
        return {}

def run_ply_validation(ply_data: bytes, prompt: str, endpoint: str = "generate/", port: int = 8099) -> Dict[str, Any]:
    """Run the PLY validator directly using imported functions."""
    
    print(f"🔍 Running PLY validation for prompt: '{prompt[:60]}...'")
    
    try:
        # Import validation functions directly
        from subnet_accurate_validator_multigpu import validate_with_production_logic_raw
        
        print(f"   Using production-accurate validation logic")
        
        # Run validation directly (no subprocess needed)
        results = validate_with_production_logic_raw(ply_data, prompt)
        
        print(f"✅ PLY validation completed successfully")
        return results
        
    except ImportError as e:
        print(f"❌ Failed to import validation functions: {e}")
        return {"error": f"Import failed: {e}"}
    except Exception as e:
        print(f"❌ PLY validation error: {e}")
        import traceback
        traceback.print_exc()
        return {"error": f"Validation error: {e}"}

def compute_clip_scores(original_prompt: str, optimized_prompt: str, 
                       image1: bytes, image2: bytes) -> Dict[str, float]:
    """Compute CLIP alignment scores for both prompts and images."""
    
    print(f"🎯 Computing CLIP alignment scores...")
    
    try:
        # Initialize CLIP analyzer
        clip_analyzer = CLIPAlignmentWithGeneration()
        
        # Load CLIP model before computing scores
        print(f"   Loading CLIP model...")
        clip_analyzer.load_clip_model()
        
        # Images are already PIL Image objects, no conversion needed
        pil_image1 = image1
        pil_image2 = image2
        
        print(f"   Images loaded successfully")
        print(f"   Image 1 size: {pil_image1.size}, mode: {pil_image1.mode}")
        print(f"   Image 2 size: {pil_image2.size}, mode: {pil_image2.mode}")
        
        # Ensure images are in RGB mode for CLIP processing
        if pil_image1.mode != 'RGB':
            print(f"   Converting Image 1 from {pil_image1.mode} to RGB")
            pil_image1 = pil_image1.convert('RGB')
        if pil_image2.mode != 'RGB':
            print(f"   Converting Image 2 from {pil_image2.mode} to RGB")
            pil_image2 = pil_image2.convert('RGB')
        
        # Compute CLIP scores
        print(f"   Computing CLIP scores...")
        score1 = clip_analyzer.compute_clip_alignment_score(original_prompt, pil_image1)
        score2 = clip_analyzer.compute_clip_alignment_score(optimized_prompt, pil_image2)
        
        # Also compute cross-scores for comparison
        cross_score1 = clip_analyzer.compute_clip_alignment_score(original_prompt, pil_image2)
        cross_score2 = clip_analyzer.compute_clip_alignment_score(optimized_prompt, pil_image1)
        
        print(f"✅ CLIP scores computed successfully")
        
        return {
            "original_prompt_original_image": score1,
            "optimized_prompt_optimized_image": score2,
            "original_prompt_optimized_image": cross_score1,
            "optimized_prompt_original_image": cross_score2
        }
        
    except Exception as e:
        print(f"❌ CLIP score computation failed: {e}")
        import traceback
        traceback.print_exc()
        return {
            "original_prompt_original_image": 0.0,
            "optimized_prompt_optimized_image": 0.0,
            "original_prompt_optimized_image": 0.0,
            "optimized_prompt_original_image": 0.0,
            "error": str(e)
        }

def test_clip_validation_comparison_with_reproducibility(original_prompt: str, 
                                                       log_count: int, min_similarity: float,
                                                       endpoint: str, port: int, num_inference_steps: int, 
                                                       guidance_scale: float, ss_sampling_steps: int, 
                                                       slat_sampling_steps: int, slat_guidance_strength: float,
                                                       ss_guidance_strength: float, seed: int = 42) -> Dict[str, Any]:
    """Main function to test CLIP alignment and validation score comparison with reproducibility optimization."""
    
    print(f"🧪 CLIP ALIGNMENT & VALIDATION SCORE COMPARISON WITH REPRODUCIBILITY")
    print(f"=" * 80)
    print(f"📝 Original Prompt: '{original_prompt}'")
    print(f"📚 Using last {log_count} episodic logs for gold prompts")
    print(f"🎯 Min similarity: {min_similarity}")
    print(f"🎲 Seed: {seed}")
    print(f"🔧 Endpoint: {endpoint}")
    print(f"🌐 Port: {port}")
    print(f"=" * 80)
    
    # Step 1: Get gold prompts using EXACT orchestrator functions
    print(f"\n📚 Step 1: Getting gold prompts using EXACT orchestrator functions...")
    try:
        gold_standard_results = get_gold_prompts_from_orchestrator(log_count)
        
        if not gold_standard_results:
            print(f"❌ No gold prompts found from logs")
            return {"error": "No gold prompts found from logs"}
        
        print(f"✅ Gold prompts loaded: {len(gold_standard_results)} prompts")
        
    except Exception as e:
        print(f"❌ Failed to get gold prompts: {e}")
        return {"error": f"Gold prompt loading failed: {e}"}
    
    # Step 2: Initialize reproducibility system with the gold prompts
    print(f"\n🔄 Step 2: Initializing reproducibility system with orchestrator gold prompts...")
    try:
        # Create reproducibility system instance
        repro_system = LLMClosePromptReproducibility(
            episodic_memory_file="dummy.json",  # We'll override this
            use_vllm=True,
            vllm_url="http://localhost:9002",
            vllm_model="llama-3-2-3b-it"
        )
        
        # CRITICAL: Override the gold standard results with the ones from orchestrator
        # This ensures we're using the EXACT same data
        repro_system.gold_standard_results = gold_standard_results
        
        print(f"✅ Reproducibility system initialized with {len(gold_standard_results)} gold prompts")
        print(f"   Using EXACT same gold prompts as orchestrator")
        
    except Exception as e:
        print(f"❌ Failed to initialize reproducibility system: {e}")
        return {"error": f"Initialization failed: {e}"}
    
    # Step 3: Run reproducibility optimization
    print(f"\n🔄 Step 3: Running reproducibility optimization...")
    try:
        optimization_result = repro_system.optimize_prompt_with_reproducibility(
            original_prompt, min_similarity, run_validation=False
        )
        
        if not optimization_result:
            print(f"❌ No suitable gold prompt found for reproducibility")
            return {"error": "No suitable gold prompt found"}
        
        print(f"✅ Reproducibility optimization completed")
        print(f"   Gold similarity: {optimization_result['similarity']:.3f}")
        print(f"   Gold score: {optimization_result['gold_score']:.4f}")
        print(f"   Optimized prompt: '{optimization_result['optimized_prompt']}'")
        
    except Exception as e:
        print(f"❌ Reproducibility optimization failed: {e}")
        return {"error": f"Optimization failed: {e}"}
    
    # Step 4: Initialize TrellisGenerator
    print(f"\n🚀 Step 4: Initializing TrellisGenerator...")
    try:
        generator = TrellisGenerator()
        print(f"✅ TrellisGenerator initialized successfully")
    except Exception as e:
        print(f"❌ Failed to initialize TrellisGenerator: {e}")
        return {"error": f"Generator initialization failed: {e}"}
    
    # Step 5: Generate 3D models from both prompts
    print(f"\n🎨 Step 5: Generating 3D models from both prompts...")
    
    print(f"   Generating from original prompt...")
    try:
        result1 = generator.generate_3d_model_image(
            original_prompt, seed, num_inference_steps, guidance_scale,
            ss_sampling_steps, slat_sampling_steps, slat_guidance_strength, ss_guidance_strength
        )
        
        if result1 is None:
            print(f"❌ Original prompt generation failed")
            return {"error": "Original prompt generation failed"}
        
        ply_data1, compressed_data1, image1 = result1
        print(f"✅ Original prompt generation completed")
        print(f"   PLY size: {len(ply_data1):,} bytes")
        print(f"   Image generated successfully")
        
    except Exception as e:
        print(f"❌ Original prompt generation failed: {e}")
        return {"error": f"Original generation failed: {e}"}
    
    print(f"   Generating from optimized prompt...")
    try:
        result2 = generator.generate_3d_model_image(
            optimization_result['optimized_prompt'], seed, num_inference_steps, guidance_scale,
            ss_sampling_steps, slat_sampling_steps, slat_guidance_strength, ss_guidance_strength
        )
        
        if result2 is None:
            print(f"❌ Optimized prompt generation failed")
            return {"error": "Optimized prompt generation failed"}
        
        ply_data2, compressed_data2, image2 = result2
        print(f"✅ Optimized prompt generation completed")
        print(f"   PLY size: {len(ply_data2):,} bytes")
        print(f"   Image generated successfully")
        
    except Exception as e:
        print(f"❌ Optimized prompt generation failed: {e}")
        return {"error": f"Optimized generation failed: {e}"}
    
    # Step 6: Compute CLIP alignment scores
    print(f"\n🎯 Step 6: Computing CLIP alignment scores...")
    try:
        clip_scores = compute_clip_scores(original_prompt, optimization_result['optimized_prompt'], image1, image2)
        
        if "error" in clip_scores:
            print(f"⚠️ CLIP score computation had errors: {clip_scores['error']}")
        
        print(f"✅ CLIP scores computed:")
        print(f"   Original prompt + Original image: {clip_scores['original_prompt_original_image']:.4f}")
        print(f"   Optimized prompt + Optimized image: {clip_scores['optimized_prompt_optimized_image']:.4f}")
        print(f"   Original prompt + Optimized image: {clip_scores['original_prompt_optimized_image']:.4f}")
        print(f"   Optimized prompt + Original image: {clip_scores['optimized_prompt_original_image']:.4f}")
        
    except Exception as e:
        print(f"❌ CLIP score computation failed: {e}")
        clip_scores = {"error": str(e)}
    
    # Step 7: Validate both PLY files
    print(f"\n🔍 Step 7: Validating both PLY files...")
    
    print(f"   Validating original prompt PLY...")
    original_validation = run_ply_validation(
        ply_data1, original_prompt, endpoint, port
    )
    
    if "error" in original_validation:
        print(f"❌ Original PLY validation failed: {original_validation['error']}")
        return {"error": f"Original validation failed: {original_validation['error']}"}
    
    print(f"✅ Original PLY validation completed")
    print(f"   Score: {original_validation.get('validation_engine_score', 'N/A')}")
    
    print(f"   Validating optimized prompt PLY...")
    optimized_validation = run_ply_validation(
        ply_data2, original_prompt, endpoint, port  # Note: validate against original prompt for fair comparison
    )
    
    if "error" in optimized_validation:
        print(f"❌ Optimized PLY validation failed: {optimized_validation['error']}")
        return {"error": f"Optimized validation failed: {optimized_validation['error']}"}
    
    print(f"✅ Optimized PLY validation completed")
    print(f"   Score: {optimized_validation.get('validation_engine_score', 'N/A')}")
    
    # Step 8: Analyze and compare results
    print(f"\n📊 Step 8: Analyzing and comparing results...")
    
    original_score = original_validation.get('validation_engine_score', 0.0)
    optimized_score = optimized_validation.get('validation_engine_score', 0.0)
    gold_score = optimization_result['gold_score']
    
    # Calculate improvements
    score_delta = optimized_score - original_score
    score_improvement_pct = ((optimized_score - original_score) / original_score * 100) if original_score > 0 else 0
    
    gold_delta = optimized_score - gold_score
    gold_improvement_pct = ((optimized_score - gold_score) / gold_score * 100) if gold_score > 0 else 0
    
    # CLIP score analysis
    original_clip = clip_scores.get('original_prompt_original_image', 0.0)
    optimized_clip = clip_scores.get('optimized_prompt_optimized_image', 0.0)
    clip_delta = optimized_clip - original_clip
    clip_improvement_pct = ((optimized_clip - original_clip) / original_clip * 100) if original_clip > 0 else 0
    
    # Determine overall assessment
    if optimized_score >= original_score * 1.1 and optimized_clip >= original_clip * 1.05:
        overall_assessment = "✅ EXCELLENT (Both scores improved significantly)"
    elif optimized_score >= original_score * 1.05 or optimized_clip >= original_clip * 1.1:
        overall_assessment = "🟡 GOOD (One score improved significantly)"
    elif optimized_score >= original_score * 0.95 and optimized_clip >= original_clip * 0.95:
        overall_assessment = "🟠 ACCEPTABLE (Scores maintained)"
    else:
        overall_assessment = "❌ POOR (Scores degraded)"
    
    # Determine reproducibility effectiveness
    if optimized_score >= gold_score * 0.95:
        reproducibility_assessment = "✅ EXCELLENT (≥95% of gold score)"
    elif optimized_score >= gold_score * 0.9:
        reproducibility_assessment = "🟡 GOOD (≥90% of gold score)"
    elif optimized_score >= gold_score * 0.8:
        reproducibility_assessment = "🟠 ACCEPTABLE (≥80% of gold score)"
    else:
        reproducibility_assessment = "❌ POOR (<80% of gold score)"
    
    # Compile final results
    final_results = {
        "test_info": {
            "original_prompt": original_prompt,
            "optimized_prompt": optimization_result['optimized_prompt'],
            "seed": seed,
            "endpoint": endpoint,
            "port": port,
            "log_count_used": log_count,
            "min_similarity": min_similarity
        },
        "reproducibility_results": optimization_result,
        "generation_results": {
            "original_ply_size_bytes": len(ply_data1),
            "optimized_ply_size_bytes": len(ply_data2),
            "original_compressed_size_bytes": len(compressed_data1) if compressed_data1 else None,
            "optimized_compressed_size_bytes": len(compressed_data2) if compressed_data2 else None
        },
        "clip_alignment_scores": clip_scores,
        "validation_scores": {
            "original_ply": original_validation,
            "optimized_ply": optimized_validation
        },
        "comparison_analysis": {
            "validation_score_delta": score_delta,
            "validation_score_improvement_pct": score_improvement_pct,
            "clip_score_delta": clip_delta,
            "clip_score_improvement_pct": clip_improvement_pct,
            "gold_delta": gold_delta,
            "gold_improvement_pct": gold_improvement_pct,
            "overall_assessment": overall_assessment,
            "reproducibility_assessment": reproducibility_assessment,
            "prompt_optimization_effectiveness": "positive" if score_delta > 0 and clip_delta > 0 else "mixed" if score_delta > 0 or clip_delta > 0 else "negative"
        }
    }
    
    # Print summary
    print(f"\n🏁 FINAL COMPARISON RESULTS WITH REPRODUCIBILITY")
    print(f"=" * 80)
    print(f"📝 PROMPTS:")
    print(f"   Original: '{original_prompt[:60]}...'")
    print(f"   Optimized: '{optimization_result['optimized_prompt'][:60]}...'")
    print(f"")
    print(f"🔄 REPRODUCIBILITY ANALYSIS:")
    print(f"   Gold similarity: {optimization_result['similarity']:.3f}")
    print(f"   Gold score: {gold_score:.4f}")
    print(f"   Reproducibility assessment: {reproducibility_assessment}")
    print(f"")
    print(f"📊 VALIDATION SCORES:")
    print(f"   Original PLY: {original_score:.4f}")
    print(f"   Optimized PLY: {optimized_score:.4f}")
    print(f"   Delta: {score_delta:+.4f} ({score_improvement_pct:+.1f}%)")
    print(f"")
    print(f"🎯 CLIP ALIGNMENT SCORES:")
    print(f"   Original prompt + Original image: {original_clip:.4f}")
    print(f"   Optimized prompt + Optimized image: {optimized_clip:.4f}")
    print(f"   Delta: {clip_delta:+.4f} ({clip_improvement_pct:+.1f}%)")
    print(f"")
    print(f"🔍 CROSS-COMPARISON:")
    print(f"   Original prompt + Optimized image: {clip_scores.get('original_prompt_optimized_image', 0.0):.4f}")
    print(f"   Optimized prompt + Original image: {clip_scores.get('optimized_prompt_original_image', 0.0):.4f}")
    print(f"")
    print(f"🎯 OVERALL ASSESSMENT: {overall_assessment}")
    print(f"📈 Prompt Optimization Effectiveness: {final_results['comparison_analysis']['prompt_optimization_effectiveness'].upper()}")
    print(f"🔬 Reproducibility Quality: {reproducibility_assessment}")
    print(f"=" * 80)
    
    # Save results
    output_file = f"clip_validation_reproducibility_comparison_{port}.json"
    with open(output_file, 'w') as f:
        json.dump(final_results, f, indent=2)
    print(f"💾 Results saved to {output_file}")
    
    return final_results

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Compare CLIP alignment and validation scores between original and reproducibility-optimized prompts",
        formatter_class=argparse.RawTextHelpFormatter
    )
    
    # Required arguments
    parser.add_argument("original_prompt", type=str, help="Original prompt to test and optimize")
    
    # Reproducibility options
    parser.add_argument(
        "--log-count", type=int, default=7,
        help="Number of recent episodic logs to use for gold prompts (default: 7)"
    )
    parser.add_argument(
        "--min-similarity", type=float, default=0.3,
        help="Minimum similarity threshold for finding close prompts (default: 0.3)"
    )
    
    # Generation options
    parser.add_argument("--seed", type=int, default=42, help="Random seed for generation (default: 42)")
    
    # Generation/validation options
    parser.add_argument("--endpoint", type=str, default="generate/", 
                       help="Endpoint path, e.g. generate/ or generate/isometric_3d/ (default: generate/)")
    parser.add_argument("--port", type=int, default=8099, 
                       help="Port to use for generation (default: 8099)")
    parser.add_argument("--num_inference_steps", type=int, default=7,
                       help="Sampler steps for image model (default: 7)")
    parser.add_argument("--guidance_scale", type=float, default=3.5,
                       help="Guidance scale for image model (default: 3.5)")
    parser.add_argument("--ss_steps", dest="ss_sampling_steps", type=int, default=21,
                       help="Sparse-structure sampler steps (default: 21)")
    parser.add_argument("--slat_steps", dest="slat_sampling_steps", type=int, default=24,
                       help="SLAT sampler steps (default: 24)")
    parser.add_argument("--slat_guidance", dest="slat_guidance_strength", type=float, default=4.0,
                       help="SLAT guidance strength (default: 4.0)")
    parser.add_argument("--ss_guidance", dest="ss_guidance_strength", type=float, default=9.5,
                       help="Sparse-structure guidance strength (default: 9.5)")
    
    args = parser.parse_args()
    
    try:
        # Run the CLIP alignment and validation comparison test with reproducibility
        results = test_clip_validation_comparison_with_reproducibility(
            original_prompt=args.original_prompt,
            log_count=args.log_count,
            min_similarity=args.min_similarity,
            endpoint=args.endpoint,
            port=args.port,
            num_inference_steps=args.num_inference_steps,
            guidance_scale=args.guidance_scale,
            ss_sampling_steps=args.ss_sampling_steps,
            slat_sampling_steps=args.slat_sampling_steps,
            slat_guidance_strength=args.slat_guidance_strength,
            ss_guidance_strength=args.ss_guidance_strength,
            seed=args.seed
        )
        
        if "error" in results:
            print(f"\n❌ Test failed: {results['error']}")
            sys.exit(1)
        else:
            print(f"\n✅ Test completed successfully!")
            print(f"🔧 Generated and compared 3D models from original and optimized prompts")
            print(f"🎯 Computed CLIP alignment scores for all combinations")
            print(f"🔍 Validated both PLY files for quality assessment")
            print(f"🔄 Used reproducibility system for automatic prompt optimization")
            sys.exit(0)
            
    except KeyboardInterrupt:
        print(f"\n⏹️ Test interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
