#!/usr/bin/env python3
"""
Zero Score Analysis Tool
Purpose: Analyze the exact conditions that cause zero validation scores
         and test if our validation setup matches the subnet behavior
"""
import sys
import json
import time
from pathlib import Path

# Add validation directory to path
validation_path = Path(__file__).parent / "validation"
sys.path.insert(0, str(validation_path))

try:
    from engine.validation_engine import ValidationEngine
    from engine.io.ply import PlyLoader
    from engine.rendering.renderer import Renderer
    print("✅ Validation engine imports successful")
except ImportError as e:
    print(f"❌ Failed to import validation engine: {e}")
    sys.exit(1)

def analyze_validation_scoring_logic():
    """Analyze the exact scoring logic to understand zero-score conditions"""
    print("🔍 ANALYZING VALIDATION SCORING LOGIC")
    print("=" * 60)
    
    # Key insights from validation_engine.py analysis:
    print("📊 Zero-Score Conditions:")
    print(f"   1. Raw alignment score < 0.3 → Final score = 0.0")
    print(f"   2. Alignment normalization: score / 0.35")
    print(f"   3. Effective threshold: 0.3 × 0.35 = 0.105 (raw CLIP)")
    print()
    
    print("📊 Final Score Formula:")
    print(f"   final_score = 0.75 × quality + 0.2 × alignment + 0.025 × ssim + 0.025 × lpips")
    print(f"   BUT: if alignment < 0.3 → final_score = 0.0 (override)")
    print()
    
    print("📊 Quality Requirements:")
    print(f"   - Minimum 7000 Gaussian points")
    print(f"   - <80% zero opacity points") 
    print(f"   - <80% zero scale points")
    print()

def test_alignment_threshold(prompt: str, ply_data: bytes) -> dict:
    """Test if a specific prompt/model falls below the alignment threshold"""
    try:
        print(f"🧪 Testing alignment threshold for: '{prompt}'")
        
        # Initialize validation components
        validator = ValidationEngine()
        validator.load_pipelines()
        
        ply_loader = PlyLoader()
        renderer = Renderer()
        
        # Load PLY data
        import io
        ply_buffer = io.BytesIO(ply_data)
        gs_data = ply_loader.from_buffer(ply_buffer)
        
        # Move to GPU
        gs_data_gpu = gs_data.send_to_device(validator.device)
        
        # Render views
        rendered_images = renderer.render_gs(gs_data_gpu, views_number=16, img_width=224, img_height=224)
        
        # Get raw validation scores BEFORE final score calculation
        # We need to access the internal metrics directly
        alignment_score_raw = validator._text_vs_image_metric.score_text_alignment(
            rendered_images, prompt, mean_op="geometric_mean", use_filter_outliers=True
        )
        
        # Apply the normalization that happens in validate_text_to_gs
        alignment_score_normalized = alignment_score_raw / 0.35
        
        print(f"   📊 Raw alignment score: {alignment_score_raw:.6f}")
        print(f"   📊 Normalized alignment: {alignment_score_normalized:.6f}")
        print(f"   📊 Threshold check: {alignment_score_normalized:.6f} < 0.3 = {alignment_score_normalized < 0.3}")
        
        # Now get the full validation result
        validation_results = validator.validate_text_to_gs(prompt, rendered_images)
        
        print(f"   📊 Final score: {validation_results.final_score:.6f}")
        print(f"   📊 Quality score: {validation_results.combined_quality_score:.6f}")
        print(f"   📊 SSIM score: {validation_results.ssim_score:.6f}")
        print(f"   📊 LPIPS score: {validation_results.lpips_score:.6f}")
        
        # Determine if this would be zero-score
        is_zero_score = alignment_score_normalized < 0.3
        print(f"   🎯 Would produce ZERO score: {is_zero_score}")
        
        return {
            'prompt': prompt,
            'raw_alignment': float(alignment_score_raw),
            'normalized_alignment': float(alignment_score_normalized),
            'final_score': float(validation_results.final_score),
            'quality_score': float(validation_results.combined_quality_score),
            'ssim_score': float(validation_results.ssim_score), 
            'lpips_score': float(validation_results.lpips_score),
            'is_zero_score': is_zero_score,
            'gaussian_count': int(gs_data.points.shape[0])
        }
        
    except Exception as e:
        print(f"   ❌ Error: {e}")
        return {'error': str(e), 'prompt': prompt}

def test_problematic_prompts():
    """Test a selection of prompts that should trigger zero scores"""
    
    # These prompts are designed to have very low CLIP alignment
    low_alignment_prompts = [
        "glass jug filled juice",  # Grammatically incomplete
        "silver chalice with leafy vine pattern",  # Complex descriptive text
        "translucent crystalline formation",  # Abstract concepts
        "amorphous blob of metallic substance",  # Vague, hard to visualize
        "thing with parts and stuff",  # Extremely vague
        "geometric abstract conceptual entity",  # Abstract, non-concrete
        "transparent invisible object floating",  # Contradictory concepts
        "the essence of blueness manifested",  # Philosophical concepts
        "quantum mechanical probability cloud",  # Scientific abstractions
        "ineffable mystical energy construct"  # Mystical/abstract
    ]
    
    print("🧪 TESTING PROBLEMATIC PROMPTS FOR ZERO SCORES")
    print("=" * 60)
    
    results = []
    
    for i, prompt in enumerate(low_alignment_prompts, 1):
        print(f"\n[{i}/{len(low_alignment_prompts)}]")
        
        # Generate using simple_local_validator approach
        try:
            import subprocess
            result = subprocess.run([
                'python3', 'simple_local_validator.py', prompt
            ], capture_output=True, text=True, timeout=180)
            
            if result.returncode == 0:
                # Parse the score from output
                lines = result.stdout.strip().split('\n')
                score_line = [line for line in lines if 'Final Score:' in line]
                if score_line:
                    score = float(score_line[0].split('Final Score:')[1].strip())
                    print(f"✅ Score: {score:.6f}")
                    
                    results.append({
                        'prompt': prompt,
                        'score': score,
                        'is_zero_equivalent': score < 0.1,  # Practically zero
                        'success': True
                    })
                else:
                    print(f"❌ Could not parse score from output")
                    results.append({'prompt': prompt, 'success': False, 'error': 'parse_error'})
            else:
                print(f"❌ Generation failed: {result.stderr}")
                results.append({'prompt': prompt, 'success': False, 'error': result.stderr})
                
        except Exception as e:
            print(f"❌ Error: {e}")
            results.append({'prompt': prompt, 'success': False, 'error': str(e)})
        
        # Small delay between tests
        time.sleep(1)
    
    return results

def main():
    print("🔬 ZERO-FIDELITY ANALYSIS TOOL")
    print("=" * 60)
    
    # First, analyze the scoring logic
    analyze_validation_scoring_logic()
    
    # Test problematic prompts
    results = test_problematic_prompts()
    
    # Analyze results
    print("\n📊 ANALYSIS SUMMARY")
    print("=" * 60)
    
    successful_tests = [r for r in results if r.get('success', False)]
    zero_score_tests = [r for r in successful_tests if r.get('is_zero_equivalent', False)]
    
    print(f"Total tests: {len(results)}")
    print(f"Successful: {len(successful_tests)}")
    print(f"Zero/near-zero scores: {len(zero_score_tests)}")
    
    if zero_score_tests:
        print("\n🎯 ZERO-SCORE PROMPTS FOUND:")
        for test in zero_score_tests:
            print(f"   Score: {test['score']:.6f} | Prompt: '{test['prompt']}'")
    else:
        print("\n⚠️  NO ZERO SCORES DETECTED")
        print("This suggests our validation setup differs from the actual subnet!")
        
        avg_score = sum(r['score'] for r in successful_tests) / len(successful_tests) if successful_tests else 0
        print(f"Average score: {avg_score:.6f}")
        
        print("\n🔧 POTENTIAL ISSUES:")
        print("   1. Different CLIP model version")
        print("   2. Different normalization factor (not 0.35)")
        print("   3. Different quality models")
        print("   4. Missing subnet-specific configurations")
    
    # Save results
    with open('zero_score_analysis_results.json', 'w') as f:
        json.dump({
            'analysis_timestamp': time.time(),
            'test_results': results,
            'summary': {
                'total_tests': len(results),
                'successful_tests': len(successful_tests),
                'zero_score_tests': len(zero_score_tests)
            }
        }, f, indent=2)
    
    print(f"\n💾 Results saved to: zero_score_analysis_results.json")

if __name__ == "__main__":
    main() 