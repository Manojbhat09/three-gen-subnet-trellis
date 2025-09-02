#!/usr/bin/env python3
"""
Standalone test script for the lightweight RL optimizer
Tests the RL system with a specific prompt when vLLM/Ollama and generation servers are running

New functionality:
- Direct validation: Use --direct flag to import validation models directly instead of subprocess
- Model preloading: Validation models are loaded once and reused for better performance
- Model cleanup: Models are unloaded after testing to free GPU memory
"""

import logging
import sys
import argparse
import time
from pathlib import Path
from typing import Dict, Optional

# Add the current directory to the path so we can import our modules
sys.path.append(str(Path(__file__).parent))

from lightweight_rl_optimizer import LightweightRLOptimizer
from clip_scorer import get_clip_scorer, unload_global_clip_scorer

def preload_validation_models(force_cpu: bool = False):
    """
    Preload validation models to avoid loading time during optimization
    This should be called once at the start of the program
    Note: This loads the validation system's CLIP model (ConvNeXt Large D), not our custom CLIP scorer
    
    Args:
        force_cpu: If True, load models on CPU instead of GPU
                  ⚠️ WARNING: 3D rendering (Gaussian Splatting) requires CUDA and cannot run on CPU
    """
    device_type = "CPU" if force_cpu else "GPU"
    print(f"🔄 Preloading validation models on {device_type}...")
    
    if force_cpu:
        print("⚠️  WARNING: CPU loading enabled - 3D rendering will still require CUDA!")
        print("   📝 Only CLIP and alignment models will run on CPU")
        print("   🚫 Gaussian Splatting rendering requires CUDA and cannot run on CPU")
    
    preload_start = time.time()
    
    try:
        from subnet_accurate_validator_multigpu import _ensure_models_loaded
        _ensure_models_loaded(force_cpu=force_cpu)
        preload_time = time.time() - preload_start
        print(f"✅ Validation models preloaded in {preload_time:.2f}s")
        print(f"   📝 Note: This includes validation system's CLIP model (ConvNeXt Large D)")
        return True
    except Exception as e:
        preload_time = time.time() - preload_start
        print(f"❌ Failed to preload validation models: {e}")
        print(f"⏱️ Preload attempt time: {preload_time:.2f}s")
        return False

def unload_validation_models():
    """
    Unload validation models to free GPU memory
    This should be called when done with validation
    """
    print("🔄 Unloading validation models...")
    unload_start = time.time()
    
    try:
        from subnet_accurate_validator_multigpu import unload_cached_models
        unload_cached_models()
        unload_time = time.time() - unload_start
        print(f"✅ Validation models unloaded in {unload_time:.2f}s")
        return True
    except Exception as e:
        unload_time = time.time() - unload_start
        print(f"❌ Failed to unload validation models: {e}")
        print(f"⏱️ Unload attempt time: {unload_time:.2f}s")
        return False

def direct_validation_callback(original_prompt: str, optimized_prompt: str, endpoint: str) -> float:
    """
    Direct validation callback that imports and uses validation models directly
    This avoids subprocess overhead and keeps models loaded in memory
    """
    validation_start_time = time.time()
    try:
        print(f"      🔍 Direct validation: '{optimized_prompt[:50]}...'")
        
        # Import validation components directly
        try:
            from subnet_accurate_validator_multigpu import validate_prompt_direct
            print(f"      ✅ Validation modules imported successfully")
        except ImportError as e:
            print(f"      ❌ Could not import validation modules: {e}")
            return 0.0
        
        # Run direct validation
        cmd_start = time.time()
        result = validate_prompt_direct(
            original_prompt=original_prompt,
            optimized_prompt=optimized_prompt if optimized_prompt != original_prompt else None,
            endpoint=endpoint
        )
        cmd_time = time.time() - cmd_start
        print(f"      ⏱️ Direct validation: {cmd_time:.2f}s")
        
        # Extract score from result
        score = result.get("validation_engine_score", 0.0)
        alignment_score = result.get("alignment_score", 0.0)
        total_time = time.time() - validation_start_time
        print(f"      ✅ Direct validation score: {score:.4f}")
        print(f"      🤝 Alignment score: {alignment_score:.4f}")
        print(f"      ⏱️ Total direct validation: {total_time:.2f}s")
        return score
        
    except Exception as e:
        total_time = time.time() - validation_start_time
        print(f"      ❌ Direct validation error: {e}")
        print(f"      ⏱️ Total direct validation (failed): {total_time:.2f}s")
        return 0.0

def generate_both_from_trellis(prompt: str, trellis_url: str = "http://localhost:8096", 
                              seed: int = None, endpoint: str = "generate_both/") -> Optional[Dict[str, str]]:
    """
    Generate both PLY and image from TRELLIS server using the generate_both endpoint
    
    Args:
        prompt: Text prompt for generation
        trellis_url: TRELLIS server URL
        seed: Random seed for generation (optional)
        endpoint: TRELLIS endpoint to use (default: generate_both/)
        
    Returns:
        Dictionary with 'image' and 'ply' keys containing base64 data, or None if failed
    """
    try:
        import requests
        
        # Prepare request data
        data = {"prompt": prompt}
        if seed is not None:
            data["seed"] = seed
        
        # Make request to generate_both endpoint
        response = requests.post(
            f"{trellis_url}/{endpoint}",
            data=data,
            timeout=180  # Longer timeout for both PLY and image generation
        )
        response.raise_for_status()
        
        # Parse response
        result = response.json()
        
        if result.get("status") != "success":
            print(f"      ❌ Generation failed: {result.get('message', 'Unknown error')}")
            return None
        
        # Extract image and PLY data
        image_base64 = result.get("image")
        ply_data = result.get("compressed_ply") or result.get("ply_data")
        
        if not image_base64:
            print(f"      ❌ No image data in response")
            return None
            
        if not ply_data:
            print(f"      ❌ No PLY data in response")
            return None
        
        return {
            "image": image_base64,
            "ply": ply_data,
            "image_size": result.get("image_size_bytes", 0),
            "ply_size": result.get("ply_size_bytes", 0),
            "compression_ratio": result.get("compression_ratio", 1.0)
        }
        
    except Exception as e:
        print(f"      ❌ Generate both failed: {e}")
        return None

def enhanced_validation_callback(original_prompt: str, optimized_prompt: str, endpoint: str) -> Dict[str, float]:
    """
    Enhanced validation callback that includes both validation engine score and CLIP score
    Uses generate_both endpoint to efficiently get both PLY and image data
    Returns a dictionary with both scores for comprehensive evaluation
    """
    validation_start_time = time.time()
    try:
        print(f"      🔍 Enhanced validation: '{optimized_prompt[:50]}...'")
        
        # Import validation components directly
        try:
            from subnet_accurate_validator_multigpu import validate_prompt_direct
            print(f"      ✅ Validation modules imported successfully")
        except ImportError as e:
            print(f"      ❌ Could not import validation modules: {e}")
            return {"validation_engine_score": 0.0, "clip_score": 0.0}
        
        # First, generate both PLY and image using generate_both endpoint
        print(f"      🎨 Generating both PLY and image using generate_both endpoint...")
        generation_start = time.time()
        
        # Use the same endpoint as validation but with generate_both
        both_endpoint = endpoint.replace("generate/", "generate_both/")
        generation_result = generate_both_from_trellis(optimized_prompt, endpoint=both_endpoint)
        generation_time = time.time() - generation_start
        print(f"      ⏱️ Generation (both PLY+image): {generation_time:.2f}s")
        
        if not generation_result:
            print(f"      ❌ Generation failed, falling back to standard validation")
            # Fallback to standard validation without CLIP
            result = validate_prompt_direct(
                original_prompt=original_prompt,
                optimized_prompt=optimized_prompt if optimized_prompt != original_prompt else None,
                endpoint=endpoint
            )
            validation_score = result.get("validation_engine_score", 0.0)
            alignment_score = result.get("alignment_score", 0.0)
            return {
                "validation_engine_score": validation_score,
                "alignment_score": alignment_score,
                "clip_score": alignment_score  # Use alignment as proxy
            }
        
        # Extract PLY data for validation
        ply_data_base64 = generation_result.get("ply")
        image_base64 = generation_result.get("image")
        
        if not ply_data_base64:
            print(f"      ❌ No PLY data in generation result")
            return {"validation_engine_score": 0.0, "alignment_score": 0.0, "clip_score": 0.0}
        
        # Decode PLY data for validation
        import base64
        ply_data = base64.b64decode(ply_data_base64)
        
        # Run validation with pre-generated PLY data
        print(f"      🔍 Running validation with pre-generated PLY data...")
        validation_start = time.time()
        result = validate_prompt_direct(
            original_prompt=original_prompt,
            optimized_prompt=optimized_prompt if optimized_prompt != original_prompt else None,
            endpoint=endpoint,
            pre_generated_ply=ply_data
        )
        validation_time = time.time() - validation_start
        print(f"      ⏱️ Validation with pre-generated PLY: {validation_time:.2f}s")
        
        # Extract validation scores
        validation_score = result.get("validation_engine_score", 0.0)
        alignment_score = result.get("alignment_score", 0.0)
        
        # Calculate CLIP score using the generated image
        clip_score = 0.0
        if image_base64:
            try:
                # Get CLIP scorer
                clip_scorer = get_clip_scorer()
                
                # Compute CLIP score using the generated image
                clip_start = time.time()
                clip_score = clip_scorer.compute_clip_score(original_prompt, image_base64)
                clip_time = time.time() - clip_start
                print(f"      ⏱️ CLIP computation: {clip_time:.2f}s")
                print(f"      🖼️ CLIP score: {clip_score:.4f}")
                print(f"      📊 Image size: {generation_result.get('image_size', 0)} bytes")
                print(f"      📊 PLY size: {generation_result.get('ply_size', 0)} bytes")
                print(f"      📊 Compression ratio: {generation_result.get('compression_ratio', 1.0):.2f}x")
                
            except Exception as e:
                print(f"      ⚠️ CLIP scoring failed: {e}")
                clip_score = alignment_score  # Fallback to alignment score
        else:
            print(f"      ⚠️ No image data, using alignment score as proxy")
            clip_score = alignment_score  # Fallback to alignment score
        
        total_time = time.time() - validation_start_time
        print(f"      ✅ Validation engine score: {validation_score:.4f}")
        print(f"      🤝 Alignment score: {alignment_score:.4f}")
        print(f"      🖼️ CLIP score: {clip_score:.4f}")
        print(f"      ⏱️ Total enhanced validation: {total_time:.2f}s")
        
        return {
            "validation_engine_score": validation_score,
            "alignment_score": alignment_score,
            "clip_score": clip_score
        }
        
    except Exception as e:
        total_time = time.time() - validation_start_time
        print(f"      ❌ Enhanced validation error: {e}")
        print(f"      ⏱️ Total enhanced validation (failed): {total_time:.2f}s")
        return {"validation_engine_score": 0.0, "alignment_score": 0.0, "clip_score": 0.0}

def alignment_score_validation_callback(original_prompt: str, optimized_prompt: str, endpoint: str) -> float:
    """
    Validation callback that returns only the alignment score for RL optimization
    This allows RL to optimize specifically for alignment rather than overall validation score
    Does NOT load the custom CLIP scorer - only uses validation system's CLIP model
    """
    validation_start_time = time.time()
    try:
        print(f"      🔍 Alignment validation: '{optimized_prompt[:50]}...'")
        
        # Generate both PLY and image using generate_both endpoint
        print(f"      🎨 Generating PLY and image for alignment scoring...")
        generation_start = time.time()
        
        # Use the same endpoint as validation but with generate_both
        both_endpoint = endpoint.replace("generate/", "generate_both/")
        generation_result = generate_both_from_trellis(optimized_prompt, endpoint=both_endpoint)
        generation_time = time.time() - generation_start
        print(f"      ⏱️ Generation (both PLY+image): {generation_time:.2f}s")
        
        if not generation_result or not generation_result.get("ply"):
            print(f"      ❌ Generation failed, returning 0.0")
            return 0.0
        
        # Decode PLY data for validation
        import base64
        ply_data = base64.b64decode(generation_result["ply"])
        
        # Import validation components
        try:
            from subnet_accurate_validator_multigpu import validate_prompt_direct
        except ImportError as e:
            print(f"      ❌ Could not import validation modules: {e}")
            return 0.0
        
        # Run validation with pre-generated PLY data
        print(f"      🔍 Computing alignment score...")
        validation_start = time.time()
        result = validate_prompt_direct(
            original_prompt=original_prompt,
            optimized_prompt=optimized_prompt if optimized_prompt != original_prompt else None,
            endpoint=endpoint,
            pre_generated_ply=ply_data
        )
        validation_time = time.time() - validation_start
        print(f"      ⏱️ Alignment computation: {validation_time:.2f}s")
        
        # Extract only the alignment score
        alignment_score = result.get("alignment_score", 0.0)
        
        total_time = time.time() - validation_start_time
        print(f"      ✅ Alignment score: {alignment_score:.4f}")
        print(f"      ⏱️ Total alignment validation: {total_time:.2f}s")
        
        return alignment_score
        
    except Exception as e:
        total_time = time.time() - validation_start_time
        print(f"      ❌ Alignment validation error: {e}")
        print(f"      ⏱️ Total alignment validation (failed): {total_time:.2f}s")
        return 0.0

def calculate_alignment_score_only(prompt: str, endpoint: str = "generate_both/", force_cpu: bool = False) -> float:
    """
    Calculate only the alignment score using the validation system's CLIP model
    This is much faster than full validation as it skips quality metrics
    Does NOT load the custom CLIP scorer - only uses validation system's CLIP model
    """
    try:
        print(f"🔍 Calculating alignment score for: '{prompt[:50]}...'")
        
        # Generate both PLY and image using generate_both endpoint
        print(f"🎨 Generating PLY and image...")
        generation_start = time.time()
        generation_result = generate_both_from_trellis(prompt, endpoint=endpoint)
        generation_time = time.time() - generation_start
        print(f"⏱️ Generation: {generation_time:.2f}s")
        
        if not generation_result or not generation_result.get("ply"):
            print(f"❌ Generation failed")
            return 0.0
        
        # Decode PLY data
        import base64
        ply_data = base64.b64decode(generation_result["ply"])
        
        # Import validation components
        try:
            from subnet_accurate_validator_multigpu import validate_prompt_direct
        except ImportError as e:
            print(f"❌ Could not import validation modules: {e}")
            return 0.0
        
        # Run validation with pre-generated PLY data
        print(f"🔍 Computing alignment score...")
        validation_start = time.time()
        result = validate_prompt_direct(
            original_prompt=prompt,
            optimized_prompt=prompt,
            endpoint=endpoint.replace("generate_both/", "generate/"),
            pre_generated_ply=ply_data
        )
        validation_time = time.time() - validation_start
        print(f"⏱️ Alignment computation: {validation_time:.2f}s")
        
        alignment_score = result.get("alignment_score", 0.0)
        print(f"✅ Alignment score: {alignment_score:.4f}")
        
        return alignment_score
        
    except Exception as e:
        print(f"❌ Alignment score calculation failed: {e}")
        return 0.0

def calculate_clip_score_only(prompt: str, endpoint: str = "generate_both/", force_cpu: bool = False) -> float:
    """
    Calculate only the CLIP score using our custom CLIP scorer
    This is much faster than full validation as it only does image-text similarity
    Loads the custom CLIP scorer (ViT-B-32 + OpenAI weights)
    """
    try:
        print(f"🔍 Calculating CLIP score for: '{prompt[:50]}...'")
        
        # Generate image using generate_both endpoint
        print(f"🎨 Generating image...")
        generation_start = time.time()
        generation_result = generate_both_from_trellis(prompt, endpoint=endpoint)
        generation_time = time.time() - generation_start
        print(f"⏱️ Generation: {generation_time:.2f}s")
        
        if not generation_result or not generation_result.get("image"):
            print(f"❌ Image generation failed")
            return 0.0
        
        # Get CLIP scorer (this loads our custom CLIP model - ViT-B-32 + OpenAI)
        clip_scorer = get_clip_scorer(force_cpu=force_cpu)
        
        # Compute CLIP score
        print(f"🖼️ Computing CLIP score...")
        clip_start = time.time()
        clip_score = clip_scorer.compute_clip_score(prompt, generation_result["image"])
        clip_time = time.time() - clip_start
        print(f"⏱️ CLIP computation: {clip_time:.2f}s")
        
        print(f"✅ CLIP score: {clip_score:.4f}")
        print(f"📊 Image size: {generation_result.get('image_size', 0)} bytes")
        
        return clip_score
        
    except Exception as e:
        print(f"❌ CLIP score calculation failed: {e}")
        return 0.0

def calculate_both_scores(prompt: str, endpoint: str = "generate_both/", force_cpu: bool = False) -> Dict[str, float]:
    """
    Calculate both alignment score and CLIP score for comparison
    This is faster than full validation but gives both metrics
    """
    try:
        print(f"🔍 Calculating both alignment and CLIP scores for: '{prompt[:50]}...'")
        
        # Generate both PLY and image using generate_both endpoint
        print(f"🎨 Generating PLY and image...")
        generation_start = time.time()
        generation_result = generate_both_from_trellis(prompt, endpoint=endpoint)
        generation_time = time.time() - generation_start
        print(f"⏱️ Generation: {generation_time:.2f}s")
        
        if not generation_result:
            print(f"❌ Generation failed")
            return {"alignment_score": 0.0, "clip_score": 0.0}
        
        # Calculate alignment score
        alignment_score = 0.0
        if generation_result.get("ply"):
            try:
                import base64
                ply_data = base64.b64decode(generation_result["ply"])
                
                from subnet_accurate_validator_multigpu import validate_prompt_direct
                
                print(f"🔍 Computing alignment score...")
                alignment_start = time.time()
                result = validate_prompt_direct(
                    original_prompt=prompt,
                    optimized_prompt=prompt,
                    endpoint=endpoint.replace("generate_both/", "generate/"),
                    pre_generated_ply=ply_data
                )
                alignment_time = time.time() - alignment_start
                print(f"⏱️ Alignment computation: {alignment_time:.2f}s")
                
                alignment_score = result.get("alignment_score", 0.0)
                
            except Exception as e:
                print(f"⚠️ Alignment score calculation failed: {e}")
        
        # Calculate CLIP score
        clip_score = 0.0
        if generation_result.get("image"):
            try:
                # Only load CLIP scorer when we actually need it (ViT-B-32 + OpenAI)
                clip_scorer = get_clip_scorer(force_cpu=force_cpu)
                
                print(f"🖼️ Computing CLIP score...")
                clip_start = time.time()
                clip_score = clip_scorer.compute_clip_score(prompt, generation_result["image"])
                clip_time = time.time() - clip_start
                print(f"⏱️ CLIP computation: {clip_time:.2f}s")
                
            except Exception as e:
                print(f"⚠️ CLIP score calculation failed: {e}")
        
        print(f"✅ Alignment score: {alignment_score:.4f}")
        print(f"✅ CLIP score: {clip_score:.4f}")
        print(f"📊 Image size: {generation_result.get('image_size', 0)} bytes")
        print(f"📊 PLY size: {generation_result.get('ply_size', 0)} bytes")
        
        return {
            "alignment_score": alignment_score,
            "clip_score": clip_score
        }
        
    except Exception as e:
        print(f"❌ Score calculation failed: {e}")
        return {"alignment_score": 0.0, "clip_score": 0.0}

def real_validation_callback(original_prompt: str, optimized_prompt: str, endpoint: str) -> float:
    """
    Real validation callback that uses the actual subnet_accurate_validator.py system
    This is the same validation system used by the orchestrator
    """
    validation_start_time = time.time()
    try:
        import subprocess
        import json
        import os
        
        print(f"      🔍 Validating: '{optimized_prompt[:50]}...'")
        
        # Clear old validation results to prevent reading stale data
        clear_start = time.time()
        validation_file = "subnet_validation_results.json"
        try:
            if os.path.exists(validation_file):
                os.remove(validation_file)
                print(f"      🗑️ Cleared old validation results file")
        except Exception as e:
            print(f"      ⚠️ Could not clear old validation file: {e}")
        clear_time = time.time() - clear_start
        print(f"      ⏱️ File cleanup: {clear_time:.2f}s")
        
        # Run validation using the same command as the orchestrator
        cmd_start = time.time()
        if optimized_prompt and optimized_prompt != original_prompt:
            cmd = [
                "bash", "-c",
                f"source /home/mbhat/miniconda/bin/activate && conda activate trellis_new && python subnet_accurate_validator_multigpu.py \"{original_prompt}\" \"{optimized_prompt}\" --endpoint \"{endpoint}\""
            ]
        else:
            cmd = [
                "bash", "-c",
                f"source /home/mbhat/miniconda/bin/activate && conda activate trellis_new && python subnet_accurate_validator_multigpu.py \"{original_prompt}\" --endpoint \"{endpoint}\""
            ]
        
        print(f"      🚀 Running validation command...")
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        cmd_time = time.time() - cmd_start
        print(f"      ⏱️ Validation command: {cmd_time:.2f}s")
        
        if result.returncode != 0:
            print(f"      ❌ Validation failed (return code {result.returncode})")
            print(f"      Error: {result.stderr}")
            return 0.0
        
        # Wait for file writing to complete
        wait_start = time.time()
        time.sleep(2)
        wait_time = time.time() - wait_start
        print(f"      ⏱️ File write wait: {wait_time:.2f}s")
        
        # Read validation results
        read_start = time.time()
        # try:
        #     with open(validation_file, 'r') as f:
        #         data = json.load(f)
        #         score = data.get("validation_engine_score", 0.0)
        #         read_time = time.time() - read_start
        #         total_time = time.time() - validation_start_time
        #         print(f"      ✅ Validation score: {score:.4f}")
        #         print(f"      ⏱️ Result read: {read_time:.2f}s")
        #         print(f"      ⏱️ Total validation: {total_time:.2f}s")
        #         return score
        # except (FileNotFoundError, json.JSONDecodeError) as e:
        #     print(f"      ❌ Could not read validation results: {e}")
            # return 0.0
        validation_file = "subnet_validation_results_8096.json"
        try: 
            with open(validation_file, 'r') as f:
                data = json.load(f)
                score = data.get("validation_engine_score", 0.0)
                read_time = time.time() - read_start
                total_time = time.time() - validation_start_time
                print(f"      ✅ Validation score: {score:.4f}")
                print(f"      ⏱️ Result read: {read_time:.2f}s")
                print(f"      ⏱️ Total validation: {total_time:.2f}s")
                return score
        except (FileNotFoundError, json.JSONDecodeError) as e:
            print(f"      ❌ Could not read validation results: {e}")
            return 0.0
    except Exception as e:
        total_time = time.time() - validation_start_time
        print(f"      ❌ Validation error: {e}")
        print(f"      ⏱️ Total validation (failed): {total_time:.2f}s")
        return 0.0

def test_rl_optimization_clip_only(prompt: str, use_vllm: bool = True, vllm_url: str = "http://localhost:11300", 
                                  ollama_url: str = "http://localhost:11434", endpoint: str = "generate/",
                                  force_cpu: bool = False):
    """Test RL optimization using ONLY CLIP scoring (no validation system)"""
    
    total_start_time = time.time()
    
    print("🧪 RL Optimization with CLIP-Only Scoring")
    print("=" * 60)
    print(f"Prompt: '{prompt}'")
    print(f"LLM: {'vLLM' if use_vllm else 'Ollama'}")
    print(f"Endpoint: {endpoint}")
    print("🎯 Using CLIP SCORE for RL optimization grading")
    print("=" * 60)
    
    # No validation models needed - only CLIP scorer
    print("🔄 Loading CLIP scorer only...")
    clip_start = time.time()
    clip_scorer = get_clip_scorer(force_cpu=force_cpu)
    clip_time = time.time() - clip_start
    print(f"✅ CLIP scorer loaded in {clip_time:.2f}s")
    
    # Get initial CLIP score
    print(f"\n🔍 Step 1: Getting initial CLIP score...")
    initial_start = time.time()
    
    # Generate image for initial scoring
    both_endpoint = endpoint.replace("generate/", "generate_both/")
    generation_result = generate_both_from_trellis(prompt, endpoint=both_endpoint)
    
    if not generation_result or not generation_result.get("image"):
        print(f"❌ Initial image generation failed")
        return
    
    initial_clip_score = clip_scorer.compute_clip_score(prompt, generation_result["image"])
    initial_time = time.time() - initial_start
    print(f"✅ Initial CLIP score: {initial_clip_score:.4f}")
    print(f"⏱️ Initial scoring: {initial_time:.2f}s")
    
    # Check if RL should trigger
    trigger_threshold = 0.7
    should_trigger = initial_clip_score < trigger_threshold
    
    print(f"\n📊 Initial Results:")
    print(f"   Original prompt: '{prompt}'")
    print(f"   Initial CLIP score: {initial_clip_score:.4f}")
    print(f"   Trigger threshold: {trigger_threshold}")
    print(f"   🎯 Using CLIP SCORE for RL optimization")
    print(f"   Should trigger RL: {should_trigger}")
    
    if not should_trigger:
        print(f"\n✅ CLIP score already above threshold ({trigger_threshold}) - no RL needed!")
        print(f"🎯 Final CLIP Score: {initial_clip_score:.4f}")
        return
    
    # Run RL optimization with CLIP scoring
    print(f"\n🔄 Step 2: Running 3-round RL optimization (CLIP score)...")
    rl_start = time.time()
    
    # Create CLIP-only validation callback
    def clip_only_validation_callback(original_prompt: str, optimized_prompt: str, endpoint: str) -> dict:
        """Validation callback that only uses CLIP scoring"""
        try:
            # Generate image for the optimized prompt
            both_endpoint = endpoint.replace("generate/", "generate_both/")
            generation_result = generate_both_from_trellis(optimized_prompt, endpoint=both_endpoint)
            
            if not generation_result or not generation_result.get("image"):
                return {
                    "validation_engine_score": 0.0,
                    "clip_score": 0.0,
                    "alignment_score": 0.0
                }
            
            # Compute CLIP score
            clip_score = clip_scorer.compute_clip_score(original_prompt, generation_result["image"])
            return {
                "validation_engine_score": clip_score,  # Use CLIP score as the main score
                "clip_score": clip_score,
                "alignment_score": 0.0  # No alignment score in CLIP-only mode
            }
            
        except Exception as e:
            print(f"      ❌ CLIP validation failed: {e}")
            return {
                "validation_engine_score": 0.0,
                "clip_score": 0.0,
                "alignment_score": 0.0
            }
    
    # Initialize RL optimizer
    rl_optimizer = LightweightRLOptimizer(
        logger=logging.getLogger(__name__),
        use_vllm=use_vllm,
        vllm_url=vllm_url,
        ollama_url=ollama_url
    )
    
    # Run RL optimization
    result = rl_optimizer.optimize_with_3_rounds(
        original_prompt=prompt,
        initial_score=initial_clip_score,
        validation_callback=clip_only_validation_callback,
        endpoint=endpoint
    )
    
    rl_time = time.time() - rl_start
    print(f"⏱️ RL optimization: {rl_time:.2f}s")
    
    # Display results
    print(f"\n🎯 Final Results:")
    print(f"   Success: {result['success']}")
    print(f"   Final CLIP score: {result['final_score']:.4f}")
    print(f"   Improvement: {result['improvement']:+.4f}")
    print(f"   Rounds used: {result['rounds_used']}")
    print(f"   Final prompt: '{result['final_optimized_prompt']}'")
    
    print(f"\n📈 Optimization Attempts:")
    for i, attempt in enumerate(result['attempts'], 1):
        print(f"   Round {i}: {attempt['strategy']}")
        print(f"     Prompt: '{attempt['prompt']}'")
        print(f"     CLIP score: {attempt['clip_score']:.4f}")
        print(f"     Confidence: {attempt['confidence']:.2f}")
    
    # Cleanup
    unload_global_clip_scorer()
    
    total_time = time.time() - total_start_time
    print(f"\n⏱️ COMPLETE TEST RUNTIME: {total_time:.2f}s")

def test_rl_optimization_enhanced(prompt: str, use_vllm: bool = True, vllm_url: str = "http://localhost:11300", 
                                 ollama_url: str = "http://localhost:11434", endpoint: str = "generate/",
                                 use_alignment_score: bool = False, force_cpu: bool = False):
    """Test RL optimization with enhanced validation including CLIP scores"""
    
    total_start_time = time.time()
    
    print("🧪 Standalone RL Optimization Test (Enhanced with CLIP)")
    print("=" * 60)
    print(f"Prompt: '{prompt}'")
    print(f"LLM: {'vLLM' if use_vllm else 'Ollama'}")
    print(f"Endpoint: {endpoint}")
    print("=" * 60)
    
    # Setup logging
    setup_start = time.time()
    logging.basicConfig(level=logging.INFO, format='%(message)s')
    logger = logging.getLogger(__name__)
    setup_time = time.time() - setup_start
    print(f"⏱️ Setup time: {setup_time:.2f}s")
    
    # Preload validation models for better performance
    preload_start = time.time()
    models_loaded = preload_validation_models(force_cpu=force_cpu)
    preload_time = time.time() - preload_start
    print(f"⏱️ Model preloading: {preload_time:.2f}s")
    
    if not models_loaded:
        print("⚠️ Warning: Validation models not preloaded, performance may be slower")
    
    # Initialize RL optimizer
    init_start = time.time()
    rl_optimizer = LightweightRLOptimizer(
        logger=logger,
        trellis_server_url="http://localhost:8096",
        use_vllm=use_vllm,
        vllm_url=vllm_url,
        ollama_url=ollama_url
    )
    init_time = time.time() - init_start
    print(f"⏱️ RL optimizer initialization: {init_time:.2f}s")
    
    # Choose validation callback based on use_alignment_score flag
    if use_alignment_score:
        print(f"\n🔍 Step 1: Getting initial alignment score...")
        initial_validation_start = time.time()
        initial_score = alignment_score_validation_callback(prompt, prompt, endpoint)
        initial_validation_time = time.time() - initial_validation_start
        print(f"⏱️ Initial validation: {initial_validation_time:.2f}s")
        
        print(f"\n📊 Initial Results:")
        print(f"   Original prompt: '{prompt}'")
        print(f"   Initial alignment score: {initial_score:.4f}")
        print(f"   Trigger threshold: {rl_optimizer.trigger_threshold}")
        print(f"   🎯 Using ALIGNMENT SCORE for RL optimization")
        
        validation_callback = alignment_score_validation_callback
    else:
        print(f"\n🔍 Step 1: Getting initial validation scores (enhanced)...")
        initial_validation_start = time.time()
        initial_result = enhanced_validation_callback(prompt, prompt, endpoint)
        initial_validation_time = time.time() - initial_validation_start
        print(f"⏱️ Initial validation: {initial_validation_time:.2f}s")
        
        initial_score = initial_result.get("validation_engine_score", 0.0)
        initial_clip_score = initial_result.get("clip_score", 0.0)
        initial_alignment_score = initial_result.get("alignment_score", 0.0)
        
        print(f"\n📊 Initial Results:")
        print(f"   Original prompt: '{prompt}'")
        print(f"   Initial validation score: {initial_score:.4f}")
        print(f"   Initial CLIP score: {initial_clip_score:.4f}")
        print(f"   Initial alignment score: {initial_alignment_score:.4f}")
        print(f"   Trigger threshold: {rl_optimizer.trigger_threshold}")
        
        validation_callback = enhanced_validation_callback
    
    # Check if RL should trigger
    trigger_check_start = time.time()
    should_trigger = rl_optimizer.should_trigger_rl_optimization(initial_score)
    trigger_check_time = time.time() - trigger_check_start
    print(f"   Should trigger RL: {should_trigger}")
    print(f"⏱️ Trigger check: {trigger_check_time:.2f}s")
    
    if not should_trigger:
        total_time = time.time() - total_start_time
        print(f"\n✅ Initial score {initial_score:.4f} is already above threshold {rl_optimizer.trigger_threshold}")
        print("   No RL optimization needed!")
        print(f"⏱️ Total test time: {total_time:.2f}s")
        return
    
    # Run RL optimization with chosen validation callback
    optimization_mode = "alignment score" if use_alignment_score else "enhanced validation"
    print(f"\n🔄 Step 2: Running 3-round RL optimization ({optimization_mode})...")
    optimization_start = time.time()
    result = rl_optimizer.optimize_with_3_rounds(
        original_prompt=prompt,
        initial_score=initial_score,
        validation_callback=validation_callback,
        endpoint=endpoint
    )
    optimization_time = time.time() - optimization_start
    print(f"⏱️ RL optimization: {optimization_time:.2f}s")
    
    # Display results
    score_type = "alignment score" if use_alignment_score else "validation score"
    print(f"\n🎯 Final Results:")
    print(f"   Success: {result['success']}")
    print(f"   Final {score_type}: {result['final_score']:.4f}")
    print(f"   Improvement: {result['improvement']:+.4f}")
    print(f"   Rounds used: {result['rounds_used']}")
    print(f"   Final prompt: '{result['final_optimized_prompt']}'")
    
    print(f"\n📈 Optimization Attempts:")
    for i, attempt in enumerate(result['attempts'], 1):
        print(f"   Round {i}: {attempt['strategy']}")
        print(f"     Prompt: '{attempt['prompt']}'")
        if use_alignment_score:
            print(f"     Alignment score: {attempt['alignment_score']:.4f}")
        else:
            print(f"     Validation score: {attempt['validation_score']:.4f}")
            if attempt['clip_score'] is not None:
                print(f"     CLIP score: {attempt['clip_score']:.4f}")
            if attempt['alignment_score'] is not None:
                print(f"     Alignment score: {attempt['alignment_score']:.4f}")
        print(f"     Confidence: {attempt['confidence']:.2f}")
    
    # Strategy insights
    insights_start = time.time()
    print(f"\n🧠 Strategy Performance:")
    insights = rl_optimizer.get_strategy_insights()
    print(f"   Best strategy: {insights['best_strategy']}")
    print(f"   Average performance: {insights['average_performance']:.3f}")
    insights_time = time.time() - insights_start
    print(f"⏱️ Strategy insights: {insights_time:.2f}s")
    
    # Cleanup: unload validation models
    cleanup_start = time.time()
    unload_validation_models()
    unload_global_clip_scorer()
    cleanup_time = time.time() - cleanup_start
    print(f"⏱️ Model cleanup: {cleanup_time:.2f}s")
    
    # Final timing summary
    total_time = time.time() - total_start_time
    print(f"\n⏱️ TIMING SUMMARY (Enhanced Validation):")
    print(f"   Setup: {setup_time:.2f}s")
    print(f"   Model preloading: {preload_time:.2f}s")
    print(f"   RL optimizer init: {init_time:.2f}s")
    print(f"   Initial validation: {initial_validation_time:.2f}s")
    print(f"   Trigger check: {trigger_check_time:.2f}s")
    print(f"   RL optimization: {optimization_time:.2f}s")
    print(f"   Strategy insights: {insights_time:.2f}s")
    print(f"   Model cleanup: {cleanup_time:.2f}s")
    print(f"   TOTAL TIME: {total_time:.2f}s")
    
    if result['success']:
        print(f"\n🎉 RL optimization successful! Score improved from {initial_score:.4f} to {result['final_score']:.4f}")
    else:
        print(f"\n⚠️ RL optimization did not achieve target score. Final score: {result['final_score']:.4f}")

def test_rl_optimization_direct(prompt: str, use_vllm: bool = True, vllm_url: str = "http://localhost:11300", 
                               ollama_url: str = "http://localhost:11434", endpoint: str = "generate/"):
    """Test RL optimization with direct validation (no subprocess)"""
    
    total_start_time = time.time()
    
    print("🧪 Standalone RL Optimization Test (Direct Validation)")
    print("=" * 60)
    print(f"Prompt: '{prompt}'")
    print(f"LLM: {'vLLM' if use_vllm else 'Ollama'}")
    print(f"Endpoint: {endpoint}")
    print("=" * 60)
    
    # Setup logging
    setup_start = time.time()
    logging.basicConfig(level=logging.INFO, format='%(message)s')
    logger = logging.getLogger(__name__)
    setup_time = time.time() - setup_start
    print(f"⏱️ Setup time: {setup_time:.2f}s")
    
    # Preload validation models for better performance
    preload_start = time.time()
    models_loaded = preload_validation_models(force_cpu=force_cpu)
    preload_time = time.time() - preload_start
    print(f"⏱️ Model preloading: {preload_time:.2f}s")
    
    if not models_loaded:
        print("⚠️ Warning: Validation models not preloaded, performance may be slower")
    
    # Initialize RL optimizer
    init_start = time.time()
    rl_optimizer = LightweightRLOptimizer(
        logger=logger,
        trellis_server_url="http://localhost:8096",
        use_vllm=use_vllm,
        vllm_url=vllm_url,
        ollama_url=ollama_url
    )
    init_time = time.time() - init_start
    print(f"⏱️ RL optimizer initialization: {init_time:.2f}s")
    
    # Get initial score using direct validation
    print(f"\n🔍 Step 1: Getting initial validation score (direct)...")
    initial_validation_start = time.time()
    initial_score = direct_validation_callback(prompt, prompt, endpoint)
    initial_validation_time = time.time() - initial_validation_start
    print(f"⏱️ Initial validation: {initial_validation_time:.2f}s")
    
    print(f"\n📊 Initial Results:")
    print(f"   Original prompt: '{prompt}'")
    print(f"   Initial score: {initial_score:.4f}")
    print(f"   Trigger threshold: {rl_optimizer.trigger_threshold}")
    
    # Check if RL should trigger
    trigger_check_start = time.time()
    should_trigger = rl_optimizer.should_trigger_rl_optimization(initial_score)
    trigger_check_time = time.time() - trigger_check_start
    print(f"   Should trigger RL: {should_trigger}")
    print(f"⏱️ Trigger check: {trigger_check_time:.2f}s")
    
    if not should_trigger:
        total_time = time.time() - total_start_time
        print(f"\n✅ Initial score {initial_score:.4f} is already above threshold {rl_optimizer.trigger_threshold}")
        print("   No RL optimization needed!")
        print(f"⏱️ Total test time: {total_time:.2f}s")
        return
    
    # Run RL optimization with direct validation
    print(f"\n🔄 Step 2: Running 3-round RL optimization (direct validation)...")
    optimization_start = time.time()
    result = rl_optimizer.optimize_with_3_rounds(
        original_prompt=prompt,
        initial_score=initial_score,
        validation_callback=direct_validation_callback,
        endpoint=endpoint
    )
    optimization_time = time.time() - optimization_start
    print(f"⏱️ RL optimization: {optimization_time:.2f}s")
    
    # Display results
    print(f"\n🎯 Final Results:")
    print(f"   Success: {result['success']}")
    print(f"   Final score: {result['final_score']:.4f}")
    print(f"   Improvement: {result['improvement']:+.4f}")
    print(f"   Rounds used: {result['rounds_used']}")
    print(f"   Final prompt: '{result['final_optimized_prompt']}'")
    
    print(f"\n📈 Optimization Attempts:")
    for i, attempt in enumerate(result['attempts'], 1):
        print(f"   Round {i}: {attempt['strategy']}")
        print(f"     Prompt: '{attempt['prompt']}'")
        print(f"     Score: {attempt['score']:.4f}")
        print(f"     Confidence: {attempt['confidence']:.2f}")
    
    # Strategy insights
    insights_start = time.time()
    print(f"\n🧠 Strategy Performance:")
    insights = rl_optimizer.get_strategy_insights()
    print(f"   Best strategy: {insights['best_strategy']}")
    print(f"   Average performance: {insights['average_performance']:.3f}")
    insights_time = time.time() - insights_start
    print(f"⏱️ Strategy insights: {insights_time:.2f}s")
    
    # Cleanup: unload validation models
    cleanup_start = time.time()
    unload_validation_models()
    cleanup_time = time.time() - cleanup_start
    print(f"⏱️ Model cleanup: {cleanup_time:.2f}s")
    
    # Final timing summary
    total_time = time.time() - total_start_time
    print(f"\n⏱️ TIMING SUMMARY (Direct Validation):")
    print(f"   Setup: {setup_time:.2f}s")
    print(f"   Model preloading: {preload_time:.2f}s")
    print(f"   RL optimizer init: {init_time:.2f}s")
    print(f"   Initial validation: {initial_validation_time:.2f}s")
    print(f"   Trigger check: {trigger_check_time:.2f}s")
    print(f"   RL optimization: {optimization_time:.2f}s")
    print(f"   Strategy insights: {insights_time:.2f}s")
    print(f"   Model cleanup: {cleanup_time:.2f}s")
    print(f"   TOTAL TIME: {total_time:.2f}s")
    
    if result['success']:
        print(f"\n🎉 RL optimization successful! Score improved from {initial_score:.4f} to {result['final_score']:.4f}")
    else:
        print(f"\n⚠️ RL optimization did not achieve target score. Final score: {result['final_score']:.4f}")

def test_rl_optimization(prompt: str, use_vllm: bool = True, vllm_url: str = "http://localhost:11300", 
                        ollama_url: str = "http://localhost:11434", endpoint: str = "generate/"):
    """Test RL optimization with a specific prompt"""
    
    total_start_time = time.time()
    
    print("🧪 Standalone RL Optimization Test")
    print("=" * 60)
    print(f"Prompt: '{prompt}'")
    print(f"LLM: {'vLLM' if use_vllm else 'Ollama'}")
    print(f"Endpoint: {endpoint}")
    print("=" * 60)
    
    # Setup logging
    setup_start = time.time()
    logging.basicConfig(level=logging.INFO, format='%(message)s')
    logger = logging.getLogger(__name__)
    setup_time = time.time() - setup_start
    print(f"⏱️ Setup time: {setup_time:.2f}s")
    
    # Initialize RL optimizer
    init_start = time.time()
    rl_optimizer = LightweightRLOptimizer(
        logger=logger,
        trellis_server_url="http://localhost:8096",
        use_vllm=use_vllm,
        vllm_url=vllm_url,
        ollama_url=ollama_url
    )
    init_time = time.time() - init_start
    print(f"⏱️ RL optimizer initialization: {init_time:.2f}s")
    
    # Get initial score
    print(f"\n🔍 Step 1: Getting initial validation score...")
    initial_validation_start = time.time()
    initial_score = real_validation_callback(prompt, prompt, endpoint)
    initial_validation_time = time.time() - initial_validation_start
    print(f"⏱️ Initial validation: {initial_validation_time:.2f}s")
    
    print(f"\n📊 Initial Results:")
    print(f"   Original prompt: '{prompt}'")
    print(f"   Initial score: {initial_score:.4f}")
    print(f"   Trigger threshold: {rl_optimizer.trigger_threshold}")
    
    # Check if RL should trigger
    trigger_check_start = time.time()
    should_trigger = rl_optimizer.should_trigger_rl_optimization(initial_score)
    trigger_check_time = time.time() - trigger_check_start
    print(f"   Should trigger RL: {should_trigger}")
    print(f"⏱️ Trigger check: {trigger_check_time:.2f}s")
    
    if not should_trigger:
        total_time = time.time() - total_start_time
        print(f"\n✅ Initial score {initial_score:.4f} is already above threshold {rl_optimizer.trigger_threshold}")
        print("   No RL optimization needed!")
        print(f"⏱️ Total test time: {total_time:.2f}s")
        return
    
    # Run RL optimization
    print(f"\n🔄 Step 2: Running 3-round RL optimization...")
    optimization_start = time.time()
    result = rl_optimizer.optimize_with_3_rounds(
        original_prompt=prompt,
        initial_score=initial_score,
        validation_callback=real_validation_callback,
        endpoint=endpoint
    )
    optimization_time = time.time() - optimization_start
    print(f"⏱️ RL optimization: {optimization_time:.2f}s")
    
    # Display results
    print(f"\n🎯 Final Results:")
    print(f"   Success: {result['success']}")
    print(f"   Final score: {result['final_score']:.4f}")
    print(f"   Improvement: {result['improvement']:+.4f}")
    print(f"   Rounds used: {result['rounds_used']}")
    print(f"   Final prompt: '{result['final_optimized_prompt']}'")
    
    print(f"\n📈 Optimization Attempts:")
    for i, attempt in enumerate(result['attempts'], 1):
        print(f"   Round {i}: {attempt['strategy']}")
        print(f"     Prompt: '{attempt['prompt']}'")
        print(f"     Score: {attempt['score']:.4f}")
        print(f"     Confidence: {attempt['confidence']:.2f}")
    
    # Strategy insights
    insights_start = time.time()
    print(f"\n🧠 Strategy Performance:")
    insights = rl_optimizer.get_strategy_insights()
    print(f"   Best strategy: {insights['best_strategy']}")
    print(f"   Average performance: {insights['average_performance']:.3f}")
    insights_time = time.time() - insights_start
    print(f"⏱️ Strategy insights: {insights_time:.2f}s")
    
    # Final timing summary
    total_time = time.time() - total_start_time
    print(f"\n⏱️ TIMING SUMMARY:")
    print(f"   Setup: {setup_time:.2f}s")
    print(f"   RL optimizer init: {init_time:.2f}s")
    print(f"   Initial validation: {initial_validation_time:.2f}s")
    print(f"   Trigger check: {trigger_check_time:.2f}s")
    print(f"   RL optimization: {optimization_time:.2f}s")
    print(f"   Strategy insights: {insights_time:.2f}s")
    print(f"   TOTAL TIME: {total_time:.2f}s")
    
    if result['success']:
        print(f"\n🎉 RL optimization successful! Score improved from {initial_score:.4f} to {result['final_score']:.4f}")
    else:
        print(f"\n⚠️ RL optimization did not achieve target score. Final score: {result['final_score']:.4f}")

def main():
    main_start_time = time.time()
    
    parser = argparse.ArgumentParser(description='Test standalone RL optimization')
    parser.add_argument('prompt', help='Input prompt to optimize')
    parser.add_argument('--use-ollama', action='store_true', help='Use Ollama instead of vLLM')
    parser.add_argument('--vllm-url', default='http://localhost:11300', help='vLLM server URL')
    parser.add_argument('--ollama-url', default='http://localhost:11434', help='Ollama server URL')
    parser.add_argument('--endpoint', default='generate/cinema/', help='TRELLIS endpoint to use')
    parser.add_argument('--direct', action='store_true', help='Use direct validation (no subprocess)')
    parser.add_argument('--enhanced', action='store_true', help='Use enhanced validation with CLIP scores')
    parser.add_argument('--alignment-score', action='store_true', help='Calculate only alignment score (fast)')
    parser.add_argument('--clip-score', action='store_true', help='Calculate only CLIP score (fast)')
    parser.add_argument('--both-scores', action='store_true', help='Calculate both alignment and CLIP scores (fast)')
    parser.add_argument('--rl-alignment', action='store_true', help='Run RL optimization using alignment score for grading (faster than full validation)')
    parser.add_argument('--rl-clip', action='store_true', help='Run RL optimization using CLIP score only (no validation system, works with --cpu-loading)')
    parser.add_argument('--cpu-loading', action='store_true', help='Load CLIP/alignment models on CPU (3D rendering still requires CUDA)')
    
    args = parser.parse_args()
    
    # Debug: Print which flags are set
    print(f"🔧 Debug - Flags set:")
    print(f"   --alignment-score: {args.alignment_score}")
    print(f"   --rl-alignment: {args.rl_alignment}")
    print(f"   --rl-clip: {args.rl_clip}")
    print(f"   --clip-score: {args.clip_score}")
    print(f"   --both-scores: {args.both_scores}")
    print(f"   --enhanced: {args.enhanced}")
    print(f"   --direct: {args.direct}")
    print(f"   --cpu-loading: {args.cpu_loading}")
    print()
    
    try:
        # Check for CPU loading limitations
        if args.cpu_loading:
            print("⚠️  CPU Loading Limitations:")
            print("   📝 CLIP and alignment models will run on CPU")
            print("   🚫 3D rendering (Gaussian Splatting) still requires CUDA")
            print("   💡 Use --clip-score, --alignment-score, or --rl-clip for CPU-only operations")
            print()
        
        if args.rl_clip:
            # Run RL optimization using CLIP score only (no validation system)
            print("🧪 RL Optimization with CLIP-Only Scoring")
            print("=" * 60)
            print(f"Prompt: '{args.prompt}'")
            print(f"LLM: {'vLLM' if not args.use_ollama else 'Ollama'}")
            print(f"Endpoint: {args.endpoint}")
            print("🎯 Using CLIP SCORE for RL optimization grading")
            print("=" * 60)
            
            # Run CLIP-only RL optimization
            test_rl_optimization_clip_only(
                prompt=args.prompt,
                use_vllm=not args.use_ollama,
                vllm_url=args.vllm_url,
                ollama_url=args.ollama_url,
                endpoint=args.endpoint,
                force_cpu=args.cpu_loading
            )
            
        elif args.rl_alignment:
            # Run RL optimization using alignment score for grading
            print("🧪 RL Optimization with Alignment Score Grading")
            print("=" * 60)
            print(f"Prompt: '{args.prompt}'")
            print(f"LLM: {'vLLM' if not args.use_ollama else 'Ollama'}")
            print(f"Endpoint: {args.endpoint}")
            print("🎯 Using ALIGNMENT SCORE for RL optimization grading")
            print("=" * 60)
            
            # Preload validation models
            preload_validation_models(force_cpu=args.cpu_loading)
            
            # Run RL optimization with alignment score
            test_rl_optimization_enhanced(
                prompt=args.prompt,
                use_vllm=not args.use_ollama,
                vllm_url=args.vllm_url,
                ollama_url=args.ollama_url,
                endpoint=args.endpoint,
                use_alignment_score=True,
                force_cpu=args.cpu_loading
            )
            
            # Cleanup
            unload_validation_models()
            # No need to unload CLIP scorer since we didn't load it
            
        elif args.alignment_score:
            # Calculate only alignment score
            print("🧪 Alignment Score Calculator")
            print("=" * 50)
            print(f"Prompt: '{args.prompt}'")
            print(f"Endpoint: {args.endpoint}")
            print("=" * 50)
            
            # Preload validation models
            preload_validation_models(force_cpu=args.cpu_loading)
            
            # Calculate alignment score
            both_endpoint = args.endpoint.replace("generate/", "generate_both/")
            score = calculate_alignment_score_only(args.prompt, both_endpoint, force_cpu=args.cpu_loading)
            
            # Cleanup
            unload_validation_models()
            
            print(f"\n🎯 Final Alignment Score: {score:.4f}")
            
        elif args.clip_score:
            # Calculate only CLIP score
            print("🧪 CLIP Score Calculator")
            print("=" * 50)
            print(f"Prompt: '{args.prompt}'")
            print(f"Endpoint: {args.endpoint}")
            print("=" * 50)
            
            # Calculate CLIP score
            both_endpoint = args.endpoint.replace("generate/", "generate_both/")
            score = calculate_clip_score_only(args.prompt, both_endpoint, force_cpu=args.cpu_loading)
            
            # Cleanup
            unload_global_clip_scorer()
            
            print(f"\n🎯 Final CLIP Score: {score:.4f}")
            
        elif args.both_scores:
            # Calculate both alignment and CLIP scores
            print("🧪 Both Scores Calculator")
            print("=" * 50)
            print(f"Prompt: '{args.prompt}'")
            print(f"Endpoint: {args.endpoint}")
            print("=" * 50)
            
            # Preload validation models
            preload_validation_models(force_cpu=args.cpu_loading)
            
            # Calculate both scores
            both_endpoint = args.endpoint.replace("generate/", "generate_both/")
            scores = calculate_both_scores(args.prompt, both_endpoint, force_cpu=args.cpu_loading)
            
            # Cleanup
            unload_validation_models()
            unload_global_clip_scorer()
            
            print(f"\n🎯 Final Scores:")
            print(f"   Alignment Score: {scores['alignment_score']:.4f}")
            print(f"   CLIP Score: {scores['clip_score']:.4f}")
            print(f"   Difference: {scores['alignment_score'] - scores['clip_score']:+.4f}")
            
        elif args.enhanced:
            test_rl_optimization_enhanced(
                prompt=args.prompt,
                use_vllm=not args.use_ollama,
                vllm_url=args.vllm_url,
                ollama_url=args.ollama_url,
                endpoint=args.endpoint,
                use_alignment_score=args.alignment_score,
                force_cpu=args.cpu_loading
            )
        elif args.direct:
            test_rl_optimization_direct(
                prompt=args.prompt,
                use_vllm=not args.use_ollama,
                vllm_url=args.vllm_url,
                ollama_url=args.ollama_url,
                endpoint=args.endpoint
            )
        else:
            test_rl_optimization(
                prompt=args.prompt,
                use_vllm=not args.use_ollama,
                vllm_url=args.vllm_url,
                ollama_url=args.ollama_url,
                endpoint=args.endpoint
            )
    except KeyboardInterrupt:
        main_time = time.time() - main_start_time
        print(f"\n⚠️ Test interrupted by user")
        print(f"⏱️ Total runtime before interruption: {main_time:.2f}s")
    except Exception as e:
        main_time = time.time() - main_start_time
        print(f"\n❌ Test failed: {e}")
        print(f"⏱️ Total runtime before failure: {main_time:.2f}s")
        import traceback
        traceback.print_exc()
    else:
        main_time = time.time() - main_start_time
        print(f"\n⏱️ COMPLETE TEST RUNTIME: {main_time:.2f}s")

if __name__ == "__main__":
    main()
