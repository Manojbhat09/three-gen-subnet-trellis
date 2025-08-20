#!/usr/bin/env python3
"""
Reproducibility Quality Test Script

This script tests how well the reproducibility system reconstructs prompts by:
1. Running the reproducibility system to find close gold prompts and merge components
2. Validating both the original and reconstructed prompts using subnet_accurate_validator_multigpu.py
3. Comparing scores to measure reconstruction quality

This version uses the EXACT same functions from continuous_trellis_orchestrator_lora_working.py
to ensure we measure the exact same performance as the orchestrator.

Usage:
    python test_reproducibility_quality.py "your prompt here" [options]
"""

import argparse
import json
import os
import subprocess
import sys
from typing import Dict, Any, Optional
from pathlib import Path

# Import the reproducibility system
from llm_close_prompt_reproducibility_test import LLMClosePromptReproducibility
# from llm_close_prompt_reproducibility_test_legacy_old import LLMClosePromptReproducibility

# Import the exact functions from the continuous orchestrator
from continuous_trellis_orchestrator_lora_working import ContinuousTrellisOrchestrator

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

def run_validator(original_prompt: str, optimized_prompt: str, endpoint: str, port: int, 
                  num_inference_steps: int, guidance_scale: float, 
                  ss_sampling_steps: int, slat_sampling_steps: int,
                  slat_guidance_strength: float, ss_guidance_strength: float) -> Dict[str, Any]:
    """Run the subnet validator and return results."""
    
    print(f"🔍 Running validation for:")
    print(f"   Original: '{original_prompt[:60]}...'")
    print(f"   Optimized: '{optimized_prompt[:60]}...'")
    
    # Build the validator command
    cmd = [
        "python", "subnet_accurate_validator_multigpu.py",
        f'"{original_prompt}"',
        f'"{optimized_prompt}"',
        "--endpoint", endpoint,
        "--port", str(port),
        "--num_inference_steps", str(num_inference_steps),
        "--guidance_scale", str(guidance_scale),
        "--ss_steps", str(ss_sampling_steps),
        "--slat_steps", str(slat_sampling_steps),
        "--slat_guidance", str(slat_guidance_strength),
        "--ss_guidance", str(ss_guidance_strength)
    ]
    
    print(f"🚀 Running command: {' '.join(cmd)}")
    
    try:
        # Run the validator
        result = subprocess.run(cmd, capture_output=True, text=True, check=True, timeout=600)
        
        # Look for the results file
        results_file = f"subnet_validation_results_{port}.json"
        if os.path.exists(results_file):
            with open(results_file, 'r') as f:
                results = json.load(f)
            print(f"✅ Validation completed successfully")
            return results
        else:
            print(f"❌ Results file not found: {results_file}")
            return {"error": "Results file not found"}
            
    except subprocess.TimeoutExpired:
        print(f"❌ Validation timed out after 10 minutes")
        return {"error": "Validation timed out"}
    except subprocess.CalledProcessError as e:
        print(f"❌ Validation failed with exit code {e.returncode}")
        print(f"   stdout: {e.stdout}")
        print(f"   stderr: {e.stderr}")
        return {"error": f"Validation failed: {e}"}
    except Exception as e:
        print(f"❌ Validation error: {e}")
        return {"error": f"Validation error: {e}"}

def test_reproducibility_quality(prompt: str, log_count: int, min_similarity: float,
                                endpoint: str, port: int, num_inference_steps: int, 
                                guidance_scale: float, ss_sampling_steps: int, 
                                slat_sampling_steps: int, slat_guidance_strength: float,
                                ss_guidance_strength: float) -> Dict[str, Any]:
    """Main function to test reproducibility quality using EXACT orchestrator functions."""
    
    print(f"🧪 REPRODUCIBILITY QUALITY TEST (USING EXACT ORCHESTRATOR FUNCTIONS)")
    print(f"=" * 60)
    print(f"📝 Testing prompt: '{prompt}'")
    print(f"📚 Using last {log_count} episodic logs (EXACT orchestrator method)")
    print(f"🎯 Min similarity: {min_similarity}")
    print(f"🔧 Endpoint: {endpoint}")
    print(f"🌐 Port: {port}")
    print(f"=" * 60)
    
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
            prompt, min_similarity, run_validation=False
        )
        
        if not optimization_result:
            print(f"❌ No suitable gold prompt found for reproducibility")
            return {"error": "No suitable gold prompt found"}
        
        print(f"✅ Reproducibility optimization completed")
        print(f"   Gold similarity: {optimization_result['similarity']:.3f}")
        print(f"   Gold score: {optimization_result['gold_score']:.4f}")
        print(f"   Reconstructed prompt: '{optimization_result['optimized_prompt']}'")
        
    except Exception as e:
        print(f"❌ Reproducibility optimization failed: {e}")
        return {"error": f"Optimization failed: {e}"}
    
    # Step 4: Validate original prompt
    print(f"\n🎯 Step 4: Validating original prompt...")
    original_results = run_validator(
        prompt, prompt, endpoint, port, num_inference_steps, guidance_scale,
        ss_sampling_steps, slat_sampling_steps, slat_guidance_strength, ss_guidance_strength
    )
    
    if "error" in original_results:
        print(f"❌ Original prompt validation failed: {original_results['error']}")
        return {"error": f"Original validation failed: {original_results['error']}"}
    
    print(f"✅ Original prompt validation completed")
    print(f"   Score: {original_results.get('validation_engine_score', 'N/A')}")
    
    # Step 5: Validate reconstructed prompt
    print(f"\n🎯 Step 5: Validating reconstructed prompt...")
    reconstructed_results = run_validator(
        prompt, optimization_result['optimized_prompt'], endpoint, port, 
        num_inference_steps, guidance_scale, ss_sampling_steps, slat_sampling_steps,
        slat_guidance_strength, ss_guidance_strength
    )
    
    if "error" in reconstructed_results:
        print(f"❌ Reconstructed prompt validation failed: {reconstructed_results['error']}")
        return {"error": f"Reconstructed validation failed: {reconstructed_results['error']}"}
    
    print(f"✅ Reconstructed prompt validation completed")
    print(f"   Score: {reconstructed_results.get('validation_engine_score', 'N/A')}")
    
    # Step 6: Analyze results
    print(f"\n📊 Step 6: Analyzing results...")
    
    original_score = original_results.get('validation_engine_score', 0.0)
    reconstructed_score = reconstructed_results.get('validation_engine_score', 0.0)
    gold_score = optimization_result['gold_score']
    
    # Calculate improvements
    score_delta = reconstructed_score - original_score
    score_improvement_pct = ((reconstructed_score - original_score) / original_score * 100) if original_score > 0 else 0
    
    gold_delta = reconstructed_score - gold_score
    gold_improvement_pct = ((reconstructed_score - gold_score) / gold_score * 100) if gold_score > 0 else 0
    
    # Determine quality assessment
    if reconstructed_score >= gold_score * 0.95:
        quality_assessment = "✅ EXCELLENT (≥95% of gold score)"
    elif reconstructed_score >= gold_score * 0.9:
        quality_assessment = "🟡 GOOD (≥90% of gold score)"
    elif reconstructed_score >= gold_score * 0.8:
        quality_assessment = "🟠 ACCEPTABLE (≥80% of gold score)"
    else:
        quality_assessment = "❌ POOR (<80% of gold score)"
    
    # Compile final results
    final_results = {
        "test_prompt": prompt,
        "reproducibility_results": optimization_result,
        "original_validation": original_results,
        "reconstructed_validation": reconstructed_results,
        "quality_analysis": {
            "original_score": original_score,
            "reconstructed_score": reconstructed_score,
            "gold_score": gold_score,
            "score_delta": score_delta,
            "score_improvement_pct": score_improvement_pct,
            "gold_delta": gold_delta,
            "gold_improvement_pct": gold_improvement_pct,
            "quality_assessment": quality_assessment,
            "reproducibility_similarity": optimization_result['similarity'],
            "log_count_used": log_count,
            "gold_prompts_available": len(gold_standard_results)
        }
    }
    
    # Print summary
    print(f"\n🏁 FINAL RESULTS SUMMARY")
    print(f"=" * 60)
    print(f"📝 Test Prompt: '{prompt[:60]}...'")
    print(f"🔧 Reconstructed: '{optimization_result['optimized_prompt'][:60]}...'")
    print(f"")
    print(f"📊 SCORES:")
    print(f"   Original: {original_score:.4f}")
    print(f"   Reconstructed: {reconstructed_score:.4f}")
    print(f"   Gold Standard: {gold_score:.4f}")
    print(f"")
    print(f"📈 IMPROVEMENTS:")
    print(f"   vs Original: {score_delta:+.4f} ({score_improvement_pct:+.1f}%)")
    print(f"   vs Gold: {gold_delta:+.4f} ({gold_improvement_pct:+.1f}%)")
    print(f"")
    print(f"🎯 QUALITY ASSESSMENT: {quality_assessment}")
    print(f"🔍 Reproducibility Similarity: {optimization_result['similarity']:.3f}")
    print(f"📚 Gold prompts available: {len(gold_standard_results)} (from last {log_count} logs)")
    print(f"🔧 Using EXACT orchestrator functions for consistency")
    print(f"=" * 60)
    
    # Save results
    output_file = f"reproducibility_quality_test_{port}.json"
    with open(output_file, 'w') as f:
        json.dump(final_results, f, indent=2)
    print(f"💾 Results saved to {output_file}")
    
    return final_results

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Test reproducibility quality using EXACT orchestrator functions",
        formatter_class=argparse.RawTextHelpFormatter
    )
    
    # Required arguments
    parser.add_argument("prompt", type=str, help="Prompt to test for reproducibility quality")
    
    # Gold prompt options
    parser.add_argument(
        "--log-count", type=int, default=7,
        help="Number of recent episodic logs to use for gold prompts (default: 7)"
    )
    parser.add_argument(
        "--min-similarity", type=float, default=0.3,
        help="Minimum similarity threshold for finding close prompts (default: 0.3)"
    )
    
    # Generation/validation options (matching subnet_accurate_validator_multigpu.py)
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
        # Run the reproducibility quality test using EXACT orchestrator functions
        results = test_reproducibility_quality(
            prompt=args.prompt,
            log_count=args.log_count,
            min_similarity=args.min_similarity,
            endpoint=args.endpoint,
            port=args.port,
            num_inference_steps=args.num_inference_steps,
            guidance_scale=args.guidance_scale,
            ss_sampling_steps=args.ss_sampling_steps,
            slat_sampling_steps=args.slat_sampling_steps,
            slat_guidance_strength=args.slat_guidance_strength,
            ss_guidance_strength=args.ss_guidance_strength
        )
        
        if "error" in results:
            print(f"\n❌ Test failed: {results['error']}")
            sys.exit(1)
        else:
            print(f"\n✅ Test completed successfully!")
            print(f"🔧 Used EXACT orchestrator functions for consistency")
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
