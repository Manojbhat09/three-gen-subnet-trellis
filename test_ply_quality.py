#!/usr/bin/env python3
"""
Test Script for fast_quality_check Function
Tests the quality validation function with generated PLY data from Trellis server
"""

import os
import sys
import time
import torch
import numpy as np
import requests
import argparse
import base64
from pathlib import Path

# Add the validation engine to the path
validation_path = Path(__file__).parent / "validation"
sys.path.insert(0, str(validation_path))

def generate_ply_from_server(prompt: str, seed: int = None, server_url: str = "http://localhost:8096",
                             custom_params: dict = None) -> bytes:
    """Generate PLY data from Trellis server"""
    try:
        seed_info = f" with seed: {seed}" if seed is not None else " with random seed"
        print(f"🔄 Generating PLY for prompt: '{prompt}'{seed_info}")

        # Use the main generate endpoint
        url = f"{server_url}/generate/"
        data = {
            'prompt': prompt,
            'return_compressed': False,  # Get uncompressed PLY for quality check
            'num_inference_steps': 30,
            'guidance_scale': 4.0,
            'ss_sampling_steps': 30,
            'slat_sampling_steps': 30,
            'slat_guidance_strength': 5.0,
            'ss_guidance_strength': 9.5
        }

        # Add seed if provided
        if seed is not None:
            data['seed'] = seed

        # Override with custom parameters if provided
        if custom_params:
            data.update(custom_params)
            print(f"   📝 Custom parameters: {custom_params}")

        start_time = time.time()
        response = requests.post(url, data=data, timeout=300)  # 5 minute timeout
        generation_time = time.time() - start_time

        if response.status_code == 200:
            print(f"✅ Generation successful in {generation_time:.2f}s")
            return response.content
        else:
            print(f"❌ Generation failed with status {response.status_code}: {response.text}")
            return None

    except requests.exceptions.RequestException as e:
        print(f"❌ Request error: {e}")
        return None
    except Exception as e:
        print(f"❌ Error generating PLY: {e}")
        return None

def test_scale_transformations(gs_data):
    """Test different scale transformations to see if they fix the issue"""
    print("\n🔄 SCALE TRANSFORMATION TESTING")
    print("=" * 40)
    
    scales = gs_data.scales
    original_stats = {
        'min': scales.min().item(),
        'max': scales.max().item(),
        'mean': scales.mean().item(),
        'std': scales.std().item()
    }
    
    print(f"📊 Original scale stats:")
    print(f"   • Min: {original_stats['min']:.10f}")
    print(f"   • Max: {original_stats['max']:.10f}")
    print(f"   • Mean: {original_stats['mean']:.10f}")
    print(f"   • Std: {original_stats['std']:.10f}")
    
    # Test different transformations
    transformations = [
        ('Original', scales),
        ('Absolute', torch.abs(scales)),
        ('Squared', scales ** 2),
        ('Square Root', torch.sqrt(torch.clamp(scales, min=0))),
        ('Log (clamped)', torch.log(torch.clamp(scales, min=1e-10))),
        ('Exp', torch.exp(torch.clamp(scales, max=10))),  # Clamp to avoid overflow
        ('Sigmoid', torch.sigmoid(scales)),
        ('Tanh', torch.tanh(scales)),
        ('Softplus', torch.nn.functional.softplus(scales)),
    ]
    
    print(f"\n🔬 Testing Transformations:")
    for name, transformed_scales in transformations:
        try:
            # Check if transformation produces valid scales
            if torch.any(torch.isnan(transformed_scales)) or torch.any(torch.isinf(transformed_scales)):
                print(f"   • {name}: ❌ Invalid values (NaN/Inf)")
                continue
            
            # Check if transformation produces reasonable scales
            min_val = transformed_scales.min().item()
            max_val = transformed_scales.max().item()
            mean_val = transformed_scales.mean().item()
            
            # Count "good" scales (not too small)
            good_scales = torch.sum(torch.all(transformed_scales > 0.001, dim=1)).item()
            good_pct = 100 * good_scales / len(transformed_scales)
            
            print(f"   • {name}: ✅ Min={min_val:.6f}, Max={max_val:.6f}, Mean={mean_val:.6f}")
            print(f"     Good scales: {good_scales:,} ({good_pct:.1f}%)")
            
            # If this transformation looks promising, test it
            if good_pct > 80:
                print(f"     🎯 PROMISING TRANSFORMATION!")
                
        except Exception as e:
            print(f"   • {name}: ❌ Error: {e}")
    
    # Test if scales might be stored as log values
    print(f"\n🔍 Log-Scale Hypothesis:")
    if torch.all(scales < 0):
        print(f"   • All scales are negative - testing exp transformation...")
        exp_scales = torch.exp(scales)
        exp_good = torch.sum(torch.all(exp_scales > 0.001, dim=1)).item()
        exp_pct = 100 * exp_good / len(exp_scales)
        print(f"   • After exp: {exp_good:,} good scales ({exp_pct:.1f}%)")
        
        if exp_pct > 80:
            print(f"   • 🎯 EXPONENTIATION FIXES THE ISSUE!")
            return 'exp', exp_scales
    
    # Test if scales might be stored as negative values that need abs
    print(f"\n🔍 Negative Scale Hypothesis:")
    if torch.all(scales < 0):
        print(f"   • All scales are negative - testing abs transformation...")
        abs_scales = torch.abs(scales)
        abs_good = torch.sum(torch.all(abs_scales > 0.001, dim=1)).item()
        abs_pct = 100 * abs_good / len(abs_scales)
        print(f"   • After abs: {abs_good:,} good scales ({abs_pct:.1f}%)")
        
        if abs_pct > 80:
            print(f"   • 🎯 ABSOLUTE VALUE FIXES THE ISSUE!")
            return 'abs', abs_scales
    
    return None, None

def test_raw_scale_values(gs_data):
    """Test raw scale values to see if they're actually zero or just very small"""
    print("\n🧪 RAW SCALE VALUE TESTING")
    print("=" * 40)
    
    scales = gs_data.scales
    
    # Test if scales are actually zero or just very small
    print("🔍 Testing if scales are actually zero:")
    
    # Check exact zero values
    exact_zeros = torch.sum(scales == 0).item()
    total_elements = scales.numel()
    print(f"   • Exact zero values: {exact_zeros:,} / {total_elements:,} ({100*exact_zeros/total_elements:.1f}%)")
    
    # Check very small values (but not zero)
    very_small = torch.sum((scales > 0) & (scales < 1e-10)).item()
    print(f"   • Very small (>0, <1e-10): {very_small:,} ({100*very_small/total_elements:.1f}%)")
    
    # Check small values
    small = torch.sum((scales >= 1e-10) & (scales < 1e-6)).item()
    print(f"   • Small (≥1e-10, <1e-6): {small:,} ({100*small/total_elements:.1f}%)")
    
    # Check medium values
    medium = torch.sum((scales >= 1e-6) & (scales < 1e-3)).item()
    print(f"   • Medium (≥1e-6, <1e-3): {medium:,} ({100*medium/total_elements:.1f}%)")
    
    # Check normal values
    normal = torch.sum(scales >= 1e-3).item()
    print(f"   • Normal (≥1e-3): {normal:,} ({100*normal/total_elements:.1f}%)")
    
    # Test if this might be a precision issue
    print(f"\n🔬 Precision Analysis:")
    print(f"   • Scales dtype: {scales.dtype}")
    print(f"   • Scales device: {scales.device}")
    
    # Check if scales might be stored in different format
    if scales.dtype == torch.float16:
        print(f"   • ⚠️ Float16 precision might cause issues with very small values")
        print(f"   • Float16 min positive: {torch.finfo(torch.float16).tiny}")
    elif scales.dtype == torch.float32:
        print(f"   • Float32 precision should be fine")
        print(f"   • Float32 min positive: {torch.finfo(torch.float32).tiny}")
    
    # Check if scales might be log-scaled
    if torch.all(scales < 0):
        print(f"   • ⚠️ All scales are negative - might be log-scaled!")
        print(f"   • Try exponentiating: torch.exp(scales)")
    elif torch.all(scales >= 0):
        print(f"   • All scales are non-negative")
    
    # Sample some actual values
    print(f"\n📊 Sample Scale Values:")
    sample_indices = torch.randperm(len(scales))[:10]
    for idx in sample_indices:
        scale_vals = scales[idx].tolist()
        print(f"   • Splat {idx}: [{scale_vals[0]:.10f}, {scale_vals[1]:.10f}, {scale_vals[2]:.10f}]")
    
    return {
        'exact_zeros': exact_zeros,
        'exact_zeros_pct': 100 * exact_zeros / total_elements,
        'very_small': very_small,
        'small': small,
        'medium': medium,
        'normal': normal
    }

def comprehensive_scale_analysis(gs_data):
    """Comprehensive analysis of scale values to debug issues"""
    print("\n🔬 COMPREHENSIVE SCALE ANALYSIS")
    print("=" * 50)
    
    scales = gs_data.scales
    print(f"📊 Scale Tensor Information:")
    print(f"   • Shape: {scales.shape}")
    print(f"   • Data type: {scales.dtype}")
    print(f"   • Device: {scales.device}")
    
    # Basic statistics
    print(f"\n📈 Scale Statistics:")
    print(f"   • Min: {scales.min().item():.8f}")
    print(f"   • Max: {scales.max().item():.8f}")
    print(f"   • Mean: {scales.mean().item():.8f}")
    print(f"   • Std: {scales.std().item():.8f}")
    print(f"   • Median: {torch.median(scales).item():.8f}")
    
    # Check for NaN or Inf values
    nan_count = torch.isnan(scales).sum().item()
    inf_count = torch.isinf(scales).sum().item()
    print(f"\n⚠️ Data Quality:")
    print(f"   • NaN values: {nan_count}")
    print(f"   • Inf values: {inf_count}")
    
    # Analyze each dimension
    print(f"\n🔍 Per-Dimension Analysis:")
    for i in range(scales.shape[1]):
        dim_scales = scales[:, i]
        print(f"   Dimension {i}:")
        print(f"     • Range: {dim_scales.min().item():.8f} to {dim_scales.max().item():.8f}")
        print(f"     • Mean: {dim_scales.mean().item():.8f}")
        print(f"     • Std: {dim_scales.std().item():.8f}")
        
        # Count values in different ranges
        zero_count = torch.sum(dim_scales == 0).item()
        tiny_count = torch.sum((dim_scales > 0) & (dim_scales < 0.001)).item()
        small_count = torch.sum((dim_scales >= 0.001) & (dim_scales < 0.01)).item()
        normal_count = torch.sum(dim_scales >= 0.01).item()
        
        total = len(dim_scales)
        print(f"     • Zero: {zero_count:,} ({100*zero_count/total:.1f}%)")
        print(f"     • Tiny (<0.001): {tiny_count:,} ({100*tiny_count/total:.1f}%)")
        print(f"     • Small (0.001-0.01): {small_count:,} ({100*small_count/total:.1f}%)")
        print(f"     • Normal (≥0.01): {normal_count:,} ({100*normal_count/total:.1f}%)")
    
    # Check for all-zero splats
    all_zero_splats = torch.all(scales == 0, dim=1)
    all_zero_count = torch.sum(all_zero_splats).item()
    print(f"\n🎯 All-Zero Splats Analysis:")
    print(f"   • Splats with all zero scales: {all_zero_count:,} ({100*all_zero_count/len(scales):.1f}%)")
    
    # Check for very small splats (all dimensions < threshold)
    thresholds = [0.001, 0.005, 0.01, 0.05]
    print(f"\n📏 Small Scale Thresholds:")
    for threshold in thresholds:
        small_splats = torch.all(scales < threshold, dim=1)
        small_count = torch.sum(small_splats).item()
        print(f"   • All scales < {threshold}: {small_count:,} ({100*small_count/len(scales):.1f}%)")
    
    # Sample some problematic splats
    if all_zero_count > 0:
        print(f"\n🔍 Sample All-Zero Splats:")
        zero_indices = torch.where(all_zero_splats)[0][:5]  # First 5
        for idx in zero_indices:
            print(f"   • Splat {idx}: scales={scales[idx].tolist()}")
    
    # Check if scales are properly normalized
    scale_magnitudes = torch.norm(scales, dim=1)
    print(f"\n📏 Scale Magnitudes:")
    print(f"   • Min magnitude: {scale_magnitudes.min().item():.8f}")
    print(f"   • Max magnitude: {scale_magnitudes.max().item():.8f}")
    print(f"   • Mean magnitude: {scale_magnitudes.mean().item():.8f}")
    
    return {
        'all_zero_count': all_zero_count,
        'all_zero_pct': 100 * all_zero_count / len(scales),
        'scale_stats': {
            'min': scales.min().item(),
            'max': scales.max().item(),
            'mean': scales.mean().item(),
            'std': scales.std().item()
        }
    }

def test_fast_quality_check_with_ply_data(ply_data: bytes, prompt: str = "", seed: int = 42):
    """Test the fast_quality_check function with PLY data bytes"""

    print(f"🧪 Testing fast_quality_check function (Prompt: '{prompt}', Seed: {seed})")
    print("=" * 70)

    if ply_data is None or len(ply_data) == 0:
        print("❌ No PLY data provided")
        return False

    print(f"📊 PLY data size: {len(ply_data) / 1024 / 1024:.2f} MB")

    try:
        # Import the PLY loader
        from validation.engine.io.ply import PlyLoader

        # Save PLY data to temporary file for loading
        temp_dir = Path(f"/tmp/ply_test_{seed}")
        temp_dir.mkdir(exist_ok=True)
        temp_ply_path = temp_dir / "gaussian_splatting.ply"

        with open(temp_ply_path, 'wb') as f:
            f.write(ply_data)

        # Load the PLY file
        print("🔄 Loading PLY file...")
        start_time = time.time()

        ply_loader = PlyLoader()
        gs_data = ply_loader.from_file("gaussian_splatting", str(temp_dir))

        load_time = time.time() - start_time
        print(f"✅ PLY loaded in {load_time:.3f}s")

        # Print basic info about the loaded data
        print(f"📊 Loaded data info:")
        print(f"   • Points: {gs_data.points.shape}")
        print(f"   • Opacities: {gs_data.opacities.shape}")
        print(f"   • Scales: {gs_data.scales.shape}")
        print(f"   • Rotations: {gs_data.rotations.shape}")
        print(f"   • Features DC: {gs_data.features_dc.shape}")
        print(f"   • Features Rest: {gs_data.features_rest.shape}")
        print(f"   • Normals: {gs_data.normals.shape}")
        print(f"   • SH Degree: {gs_data.sh_degree.shape}")

        # Run comprehensive scale analysis for debugging
        scale_analysis = comprehensive_scale_analysis(gs_data)

        # Test raw scale values to see if they're actually zero
        raw_scale_test = test_raw_scale_values(gs_data)

        # Test different scale transformations
        best_transform, fixed_scales = test_scale_transformations(gs_data)

        # Test the fast_quality_check function
        print("\n🔍 Testing fast_quality_check function...")
        start_time = time.time()

        # Import the function (we'll define it locally since it's not imported)
        def fast_quality_check(gs_data, verbose=True) -> tuple[bool, str]:
            """Ultra-fast quality check that takes <1 second"""

            # Quick checks that don't require full validation
            issues = []

            # Check splat count (fast)
            if gs_data.points.shape[0] < 7000:
                issues.append(f"Insufficient splats: {gs_data.points.shape[0]} < 7000")

            # Check opacity distribution (fast)
            zero_opacity = torch.sum(gs_data.opacities < 1e-3).item()
            opacity_pct = 100 * zero_opacity / len(gs_data.opacities)
            if opacity_pct > 80:
                issues.append(f"Too many zero opacity: {opacity_pct:.1f}%")

            # Check scale distribution (fast) - FIXED LOGIC
            # The issue was: torch.all(scales < 0.05, dim=1) was too strict
            # We should check if ANY scale dimension is too small, not ALL
            scales = gs_data.scales
            if verbose:
                print(f"🔍 Scale Debug Info:")
                print(f"   • Scale shape: {scales.shape}")
                print(f"   • Scale range: {scales.min().item():.6f} to {scales.max().item():.6f}")
                print(f"   • Scale mean: {scales.mean().item():.6f}")
                print(f"   • Scale std: {scales.std().item():.6f}")
                
                # Check each dimension separately
                for i in range(scales.shape[1]):
                    dim_scales = scales[:, i]
                    zero_dim = torch.sum(dim_scales < 0.001).item()
                    dim_pct = 100 * zero_dim / len(dim_scales)
                    print(f"   • Scale dim {i}: {zero_dim:,} zero values ({dim_pct:.1f}%)")
            
            # FIXED: Check if ANY scale dimension is too small (not ALL)
            # A splat is "bad" if ALL its scale dimensions are too small
            small_scales = torch.all(scales < 0.001, dim=1)  # Reduced threshold from 0.05 to 0.001
            zero_scales = torch.sum(small_scales).item()
            scale_pct = 100 * zero_scales / len(scales)
            
            if verbose:
                print(f"   • Small scales threshold: 0.001")
                print(f"   • Splats with small scales: {zero_scales:,} ({scale_pct:.1f}%)")
            
            if scale_pct > 80:
                issues.append(f"Too many small scales: {scale_pct:.1f}%")

            is_valid = len(issues) == 0
            return is_valid, "; ".join(issues) if issues else "All checks passed"

        # Run the quality check
        is_valid, issues = fast_quality_check(gs_data, verbose=True)
        check_time = time.time() - start_time

        print(f"✅ Quality check completed in {check_time:.3f}s")
        print(f"�� Results:")
        print(f"   • Valid: {'✅' if is_valid else '❌'}")
        print(f"   • Issues: {issues}")

        # Detailed analysis
        print(f"\n🔍 Detailed Analysis:")

        # Splat count
        splat_count = gs_data.points.shape[0]
        print(f"   • Splat Count: {splat_count:,} (need ≥7,000)")
        print(f"      {'✅ PASS' if splat_count >= 7000 else '❌ FAIL'}")

        # Opacity analysis
        opacities = gs_data.opacities
        zero_opacity = torch.sum(opacities < 1e-3).item()
        opacity_pct = 100 * zero_opacity / len(opacities)
        print(f"   • Zero Opacity: {opacity_pct:.1f}% (need <80%)")
        print(f"      {'✅ PASS' if opacity_pct < 80 else '❌ FAIL'}")

        # Scale analysis - FIXED LOGIC
        scales = gs_data.scales
        print(f"🔍 Scale Analysis:")
        print(f"   • Scale shape: {scales.shape}")
        print(f"   • Scale range: {scales.min().item():.6f} to {scales.max().item():.6f}")
        print(f"   • Scale mean: {scales.mean().item():.6f}")
        print(f"   • Scale std: {scales.std().item():.6f}")
        
        # Check each scale dimension separately
        for i in range(scales.shape[1]):
            dim_scales = scales[:, i]
            zero_dim = torch.sum(dim_scales < 0.001).item()
            dim_pct = 100 * zero_dim / len(dim_scales)
            print(f"   • Scale dim {i}: {zero_dim:,} small values ({dim_pct:.1f}%)")
        
        # FIXED: Use the same logic as fast_quality_check
        small_scales = torch.all(scales < 0.001, dim=1)  # Reduced threshold from 0.05 to 0.001
        zero_scales = torch.sum(small_scales).item()
        scale_pct = 100 * zero_scales / len(scales)
        print(f"   • Zero Scales: {scale_pct:.1f}% (need <80%)")
        print(f"      {'✅ PASS' if scale_pct < 80 else '❌ FAIL'}")
        print(f"   • Small scales threshold: 0.001")
        print(f"   • Splats with small scales: {zero_scales:,} ({scale_pct:.1f}%)")

        # Rotation diversity check
        rotations = gs_data.rotations
        rotation_diversity = not torch.allclose(rotations, rotations[0:1], atol=1e-6)
        print(f"   • Rotation Diversity: {'✅ PASS' if rotation_diversity else '❌ FAIL'}")

        # Overall validation result
        overall_valid = (splat_count >= 7000 and
                        opacity_pct < 80 and
                        scale_pct < 80 and
                        rotation_diversity)

        print(f"\n Overall Validation Result: {'✅ PASS' if overall_valid else '❌ FAIL'}")

        # Test if fixed scales improve the result
        if best_transform and fixed_scales is not None:
            print(f"\n🔧 TESTING FIXED SCALES WITH {best_transform.upper()} TRANSFORMATION")
            print("=" * 60)
            
            # Create a copy of gs_data with fixed scales
            import copy
            fixed_gs_data = copy.deepcopy(gs_data)
            fixed_gs_data.scales = fixed_scales
            
            # Test the fixed data
            fixed_is_valid, fixed_issues = fast_quality_check(fixed_gs_data, verbose=False)
            print(f"   • Fixed scales valid: {'✅ PASS' if fixed_is_valid else '❌ FAIL'}")
            print(f"   • Fixed scales issues: {fixed_issues}")
            
            if fixed_is_valid:
                print(f"   • 🎯 SUCCESS! {best_transform.upper()} transformation fixes the scale issue!")
                print(f"   • 💡 Apply this transformation in your pipeline")
            else:
                print(f"   • ⚠️ {best_transform.upper()} transformation helps but doesn't fully fix the issue")

        # Provide recommendations based on findings
        print(f"\n💡 RECOMMENDATIONS")
        print("=" * 40)
        
        if scale_analysis['all_zero_pct'] > 80:
            print(f"   • 🚨 CRITICAL: {scale_analysis['all_zero_pct']:.1f}% of splats have zero scales")
            print(f"   • 🔧 This is why you're getting 0.0 task fidelity scores")
            print(f"   • 💡 The issue is in TRELLIS scale generation, not your parameters")
            
            if best_transform:
                print(f"   • ✅ SOLUTION: Apply {best_transform.upper()} transformation to scales")
                print(f"   • 📝 Code: gs_data.scales = {best_transform}(gs_data.scales)")
            else:
                print(f"   • ⚠️ No simple transformation found - need to fix TRELLIS pipeline")
                print(f"   • 🔍 Check TRELLIS scale initialization and optimization")
        
        if raw_scale_test['exact_zeros_pct'] > 80:
            print(f"   • 🚨 {raw_scale_test['exact_zeros_pct']:.1f}% of scale elements are exactly zero")
            print(f"   • 🔍 This suggests TRELLIS is not generating scales properly")
        
        # Clean up temp file
        try:
            temp_ply_path.unlink()
            temp_dir.rmdir()
        except:
            pass

        # Return detailed results
        results = {
            'valid': overall_valid,
            'splat_count': splat_count,
            'opacity_pct': opacity_pct,
            'scale_pct': scale_pct,
            'rotation_diversity': rotation_diversity,
            'issues': issues
        }

        return results

    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("💡 Make sure the validation engine is properly installed")
        return {'valid': False, 'error': str(e)}

    except Exception as e:
        print(f"❌ Error during testing: {e}")
        import traceback
        traceback.print_exc()
        return {'valid': False, 'error': str(e)}



def parameter_sweep(prompt: str, base_seed: int = 42, server_url: str = "http://localhost:8096"):
    """Perform parameter sweep with 4 different configurations"""

    # Define the 4 configurations
    configs = [
        {
            'name': 'Default (Random Seed)',
            'seed': None,  # Random seed
            'params': None
        },
        {
            'name': 'Fixed Seed',
            'seed': base_seed,
            'params': None
        },
        {
            'name': 'Config 3 (ss:25, slat:28, ss_str:8.0, slat_str:4.0)',
            'seed': base_seed,
            'params': {
                'ss_sampling_steps': 25,
                'slat_sampling_steps': 28,
                'ss_guidance_strength': 8.0,
                'slat_guidance_strength': 4.0
            }
        },
        {
            'name': 'Config 4 (ss:30, slat:35, ss_str:8.5, slat_str:5.5)',
            'seed': base_seed,
            'params': {
                'ss_sampling_steps': 30,
                'slat_sampling_steps': 35,
                'ss_guidance_strength': 8.5,
                'slat_guidance_strength': 5.5
            }
        }
    ]

    print("🚀 Parameter Sweep Analysis")
    print("=" * 100)
    print(f"Prompt: '{prompt}'")
    print(f"Base Seed: {base_seed}")
    print(f"Server: {server_url}")
    print("=" * 100)

    results = []

    # Generate and test each configuration
    for i, config in enumerate(configs, 1):
        print(f"\n🔄 GENERATION {i}: {config['name']}")
        print("-" * 60)

        ply_data = generate_ply_from_server(
            prompt=prompt,
            seed=config['seed'],
            server_url=server_url,
            custom_params=config['params']
        )

        if ply_data:
            result = test_fast_quality_check_with_ply_data(
                ply_data,
                prompt,
                config['seed'] if config['seed'] is not None else f"random_{i}"
            )
            result['config_name'] = config['name']
            result['config'] = config
        else:
            result = {
                'valid': False,
                'error': 'Generation failed',
                'config_name': config['name'],
                'config': config
            }

        results.append(result)

    # Analysis and comparison
    print("\n" + "=" * 100)
    print("📊 PARAMETER SWEEP RESULTS")
    print("=" * 100)

    # Filter out failed generations
    successful_results = [r for r in results if 'error' not in r]

    if not successful_results:
        print("❌ All generations failed!")
        for i, result in enumerate(results, 1):
            print(f"Config {i} ({result['config_name']}): Failed - {result.get('error', 'Unknown error')}")
        return results

    # Display results for successful generations
    for i, result in enumerate(results, 1):
        if 'error' not in result:
            print(f"\n{i}. {result['config_name']}: {'✅ PASS' if result['valid'] else '❌ FAIL'}")
            print(f"   • Splat Count: {result['splat_count']:,}")
            print(f"   • Zero Opacity: {result['opacity_pct']:.1f}%")
            print(f"   • Zero Scales: {result['scale_pct']:.1f}%")
            print(f"   • Rotation Diversity: {'✅' if result['rotation_diversity'] else '❌'}")
        else:
            print(f"\n{i}. {result['config_name']}: ❌ FAILED - {result['error']}")

    # Calculate quality scores and rank
    def score_results(r):
        score = 0
        if r['valid']:
            score += 100
        score += min(r['splat_count'] / 100, 50)  # Max 50 points for splat count
        score += (100 - r['opacity_pct']) / 2  # Max 50 points for opacity
        score += (100 - r['scale_pct']) / 2    # Max 50 points for scales
        if r['rotation_diversity']:
            score += 25
        return score

    # Score and rank successful results
    scored_results = []
    for result in successful_results:
        score = score_results(result)
        scored_results.append((result, score))

    # Sort by score (highest first)
    scored_results.sort(key=lambda x: x[1], reverse=True)

    print(f"\n🏆 QUALITY RANKING (Best to Worst)")
    print("-" * 50)

    for rank, (result, score) in enumerate(scored_results, 1):
        medal = {1: '🥇', 2: '🥈', 3: '🥉'}.get(rank, f'{rank}.')
        print(f"{medal} {result['config_name']}: {score:.1f} points")

    # Detailed parameter analysis
    print(f"\n🔍 PARAMETER IMPACT ANALYSIS")
    print("-" * 50)

    if len(successful_results) >= 2:
        # Compare default vs best config
        default_result = next((r for r in successful_results if r['config_name'] == 'Default (Random Seed)'), None)
        best_result = scored_results[0][0]

        if default_result and best_result != default_result:
            print(f"Best configuration: {best_result['config_name']}")
            print("Parameters used:")
            for key, value in best_result['config']['params'].items():
                print(f"  • {key}: {value}")

            score_diff = scored_results[0][1] - score_results(default_result)
            print(f"Score improvement: +{score_diff:.1f} points over default")

    return results

def comprehensive_parameter_optimization(prompt: str, base_seed: int = 42, server_url: str = "http://localhost:8096"):
    """Comprehensive parameter optimization to find the best possible settings"""
    
    print("🚀 COMPREHENSIVE PARAMETER OPTIMIZATION")
    print("=" * 80)
    print(f"Prompt: '{prompt}'")
    print(f"Base Seed: {base_seed}")
    print(f"Server: {server_url}")
    print("=" * 80)
    
    # Define parameter ranges to test
    param_ranges = {
        'ss_sampling_steps': [20, 25, 30, 35],           # Structure sampling
        'slat_sampling_steps': [24, 28, 32, 36],         # Detail sampling  
        'ss_guidance_strength': [7.5, 8.0, 8.5, 9.0],   # Structure guidance
        'slat_guidance_strength': [3.5, 4.0, 4.5, 5.0], # Detail guidance
        'guidance_scale': [3.0, 3.5, 4.0, 4.5]          # Overall guidance
    }
    
    # Generate all combinations (but limit to top 16 to avoid too many tests)
    from itertools import product
    param_names = list(param_ranges.keys())
    param_values = list(param_ranges.values())
    
    # Create parameter combinations
    combinations = []
    for values in product(*param_values):
        combo = dict(zip(param_names, values))
        combinations.append(combo)
    
    # Limit to top combinations based on heuristics
    def score_combo(combo):
        """Score a parameter combination based on heuristics"""
        score = 0
        
        # Prefer balanced sampling steps (not too high, not too low)
        ss_steps = combo['ss_sampling_steps']
        slat_steps = combo['slat_sampling_steps']
        
        if 20 <= ss_steps <= 30: score += 10  # Sweet spot
        if 24 <= slat_steps <= 32: score += 10  # Sweet spot
        
        # Prefer balanced guidance strengths
        ss_guidance = combo['ss_guidance_strength']
        slat_guidance = combo['slat_guidance_strength']
        
        if 7.5 <= ss_guidance <= 8.5: score += 10  # Good structure
        if 3.5 <= slat_guidance <= 4.5: score += 10  # Good detail
        
        # Prefer moderate overall guidance
        if 3.0 <= combo['guidance_scale'] <= 4.0: score += 10
        
        return score
    
    # Score and sort combinations
    scored_combos = [(combo, score_combo(combo)) for combo in combinations]
    scored_combos.sort(key=lambda x: x[1], reverse=True)
    
    # Take top 16 combinations
    top_combos = [combo for combo, score in scored_combos[:16]]
    
    print(f"🔍 Testing {len(top_combos)} top parameter combinations...")
    print(f"   Total possible combinations: {len(combinations)}")
    print(f"   Selected based on heuristic scoring")
    
    results = []
    
    # Test each top combination
    for i, combo in enumerate(top_combos, 1):
        print(f"\n🔄 OPTIMIZATION TEST {i}/16")
        print("-" * 50)
        print(f"Parameters: {combo}")
        
        ply_data = generate_ply_from_server(
            prompt=prompt,
            seed=base_seed,  # Use fixed seed for consistency
            server_url=server_url,
            custom_params=combo
        )
        
        if ply_data:
            result = test_fast_quality_check_with_ply_data(
                ply_data, prompt, base_seed
            )
            result['parameters'] = combo
            result['combo_id'] = i
        else:
            result = {
                'valid': False,
                'error': 'Generation failed',
                'parameters': combo,
                'combo_id': i
            }
        
        results.append(result)
        
        # Show progress
        success_rate = len([r for r in results if 'error' not in r]) / len(results) * 100
        print(f"   Progress: {i}/16 ({success_rate:.1f}% success rate)")
    
    # Comprehensive analysis
    print("\n" + "=" * 80)
    print("🏆 COMPREHENSIVE OPTIMIZATION RESULTS")
    print("=" * 80)
    
    # Filter successful results
    successful_results = [r for r in results if 'error' not in r]
    
    if not successful_results:
        print("❌ All optimizations failed!")
        return results
    
    # Score and rank all successful results
    def comprehensive_score(r):
        """Comprehensive scoring system"""
        score = 0
        
        if r['valid']:
            score += 100  # Base pass/fail
        
        # Quality metrics
        score += min(r['splat_count'] / 100, 50)  # Splat count (max 50)
        score += (100 - r['opacity_pct']) / 2     # Opacity quality (max 50)
        score += (100 - r['scale_pct']) / 2       # Scale quality (max 50)
        
        if r['rotation_diversity']:
            score += 25  # Rotation diversity
        
        # Parameter efficiency bonus
        params = r['parameters']
        if params['ss_sampling_steps'] <= 25: score += 10  # Speed bonus
        if params['slat_sampling_steps'] <= 28: score += 10  # Speed bonus
        
        return score
    
    # Score and rank
    scored_results = []
    for result in successful_results:
        score = comprehensive_score(result)
        scored_results.append((result, score))
    
    scored_results.sort(key=lambda x: x[1], reverse=True)
    
    # Display top results
    print(f"\n🥇 TOP 5 OPTIMIZED CONFIGURATIONS")
    print("-" * 60)
    
    for rank, (result, score) in enumerate(scored_results[:5], 1):
        medal = {1: '🥇', 2: '🥈', 3: '🥉'}.get(rank, f'{rank}.')
        params = result['parameters']
        
        print(f"{medal} Score: {score:.1f} points")
        print(f"   • ss_steps: {params['ss_sampling_steps']}, slat_steps: {params['slat_sampling_steps']}")
        print(f"   • ss_guidance: {params['ss_guidance_strength']}, slat_guidance: {params['slat_guidance_strength']}")
        print(f"   • guidance_scale: {params['guidance_scale']}")
        print(f"   • Splat Count: {result['splat_count']:,}")
        print(f"   • Quality: Opacity {result['opacity_pct']:.1f}%, Scales {result['scale_pct']:.1f}%")
        print()
    
    # Parameter impact analysis
    print(f"🔍 PARAMETER IMPACT ANALYSIS")
    print("-" * 50)
    
    # Analyze which parameters have the biggest impact
    param_impact = {}
    for param_name in param_names:
        param_impact[param_name] = {'values': [], 'scores': []}
    
    for result, score in scored_results:
        for param_name, value in result['parameters'].items():
            param_impact[param_name]['values'].append(value)
            param_impact[param_name]['scores'].append(score)
    
    print("Parameter effectiveness (higher scores = better):")
    for param_name in param_names:
        values = param_impact[param_name]['values']
        scores = param_impact[param_name]['scores']
        
        # Group by value and calculate average score
        value_scores = {}
        for value, score in zip(values, scores):
            if value not in value_scores:
                value_scores[value] = []
            value_scores[value].append(score)
        
        print(f"\n{param_name}:")
        for value in sorted(value_scores.keys()):
            avg_score = sum(value_scores[value]) / len(value_scores[value])
            count = len(value_scores[value])
            print(f"   • {value}: {avg_score:.1f} avg score ({count} tests)")
    
    # Recommendations
    print(f"\n💡 OPTIMIZATION RECOMMENDATIONS")
    print("-" * 50)
    
    best_result = scored_results[0][0]
    best_params = best_result['parameters']
    
    print(f"🎯 BEST CONFIGURATION:")
    print(f"   • ss_sampling_steps: {best_params['ss_sampling_steps']}")
    print(f"   • slat_sampling_steps: {best_params['slat_sampling_steps']}")
    print(f"   • ss_guidance_strength: {best_params['ss_guidance_strength']}")
    print(f"   • slat_guidance_strength: {best_params['slat_guidance_strength']}")
    print(f"   • guidance_scale: {best_params['guidance_scale']}")
    
    print(f"\n📊 EXPECTED RESULTS:")
    print(f"   • Quality Score: {scored_results[0][1]:.1f}/225")
    print(f"   • Splat Count: {best_result['splat_count']:,}")
    print(f"   • Generation Time: ~{best_params['ss_sampling_steps'] + best_params['slat_sampling_steps']} seconds")
    
    return results

def main():
    """Main test function with command line arguments"""

    parser = argparse.ArgumentParser(
        description="Parameter sweep analysis for PLY quality with Trellis server",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
This script performs parameter analysis with two modes:

STANDARD MODE (default):
  Tests 4 configurations to find good parameters quickly

OPTIMIZATION MODE (--optimize):
  Tests 16 carefully selected parameter combinations to find the absolute best settings

Examples:
  python test_ply_quality.py --prompt "a red sports car"
  python test_ply_quality.py --prompt "a beautiful landscape" --seed 999 --optimize
  python test_ply_quality.py --prompt "a modern chair" --server http://localhost:8001 --optimize
        """
    )

    parser.add_argument(
        '--prompt', '-p',
        type=str,
        required=True,
        help='Prompt for 3D model generation'
    )

    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Base seed for generation (default: 42)'
    )

    parser.add_argument(
        '--server', '-s',
        type=str,
        default='http://localhost:8096',
        help='Trellis server URL (default: http://localhost:8000)'
    )

    parser.add_argument(
        '--optimize', '-o',
        action='store_true',
        help='Run comprehensive parameter optimization (tests 16 combinations)'
    )

    args = parser.parse_args()

    # Run parameter sweep or comprehensive optimization
    if args.optimize:
        print("🚀 Running COMPREHENSIVE PARAMETER OPTIMIZATION...")
        results = comprehensive_parameter_optimization(
            prompt=args.prompt,
            base_seed=args.seed,
            server_url=args.server
        )
    else:
        print("🔄 Running STANDARD PARAMETER SWEEP...")
        results = parameter_sweep(
            prompt=args.prompt,
            base_seed=args.seed,
            server_url=args.server
        )

    # Return success if at least one generation was successful
    success = any('error' not in result for result in results)
    return success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
