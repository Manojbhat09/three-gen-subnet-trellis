#!/usr/bin/env python3
"""
Test Subnet-Accurate Validation with Real Zero-Score Prompts
Purpose: Test our prompt optimization against the actual subnet validation logic
"""
import subprocess
import json
import time
from auto_prompt_optimizer import AutoPromptOptimizer

# Real prompts that got 0.0 task fidelity scores from the log
REAL_ZERO_SCORE_PROMPTS = [
    "samurai helmet kabuto detailed",
    "violet amulet with star emblem", 
    "crystal-clear domes reflect moonlight softly",
]

def test_prompt_with_subnet_validator(prompt: str, quality_threshold: float = 0.6) -> dict:
    """Test a prompt using subnet_accurate_validator.py"""
    try:
        result = subprocess.run([
            'python3', 'subnet_accurate_validator.py', prompt, 
            '--quality-threshold', str(quality_threshold)
        ], capture_output=True, text=True, timeout=300)
        
        if result.returncode == 0:
            # Load the saved results
            try:
                with open("subnet_validation_results.json", "r") as f:
                    results = json.load(f)
                return {"success": True, "results": results, "error": None}
            except FileNotFoundError:
                return {"success": False, "results": None, "error": "Results file not found"}
        else:
            return {"success": False, "results": None, "error": f"Return code: {result.returncode}, stderr: {result.stderr[:500]}"}
        
    except subprocess.TimeoutExpired:
        return {"success": False, "results": None, "error": "Subnet validation timeout (300s)"}
    except Exception as e:
        return {"success": False, "results": None, "error": str(e)}

def analyze_subnet_behavior():
    """Analyze the subnet behavior for all problematic prompts"""
    
    print("🔬 ANALYZING SUBNET-ACCURATE BEHAVIOR")
    print("=" * 80)
    print("Testing prompts that got 0.0 task fidelity in actual subnet runs")
    print()
    
    results = []
    
    for i, prompt in enumerate(REAL_ZERO_SCORE_PROMPTS, 1):
        print(f"[{i}/{len(REAL_ZERO_SCORE_PROMPTS)}] Testing: '{prompt}'")
        print("-" * 60)
        
        test_result = test_prompt_with_subnet_validator(prompt)
        
        if test_result["success"]:
            data = test_result["results"]
            
            print(f"✅ Subnet validation completed:")
            print(f"   🏆 Validation Engine Score: {data['validation_engine_score']:.4f}")
            print(f"   🤝 Alignment Score: {data['alignment_score']:.4f}")
            print(f"   💎 Quality Score: {data['quality_score']:.4f}")
            print(f"   🎯 Task Fidelity Score: {data['task_fidelity_score']:.4f}")
            print(f"   🎭 Demo Fidelity Score: {data['demo_fidelity_score']:.4f}")
            print(f"   ✅ Validation Passed: {data['validation_passed']}")
            print(f"   🚧 Quality Threshold: {data['quality_threshold']}")
            
            # Determine why it would get 0.0 in real subnet
            zero_reason = None
            if not data['validation_passed']:
                zero_reason = f"Failed quality threshold: {data['validation_engine_score']:.4f} < {data['quality_threshold']}"
            elif not data['alignment_threshold_passed']:
                zero_reason = f"Failed alignment threshold: {data['alignment_score']:.4f} < 0.3"
            elif data['demo_fidelity_score'] == 0.0:
                zero_reason = f"Demo fidelity zero: CLIP {data['alignment_score']:.4f} < 0.6"
            
            if zero_reason:
                print(f"   ❌ ZERO SCORE REASON: {zero_reason}")
            else:
                print(f"   ✅ WOULD NOT BE ZERO in subnet")
            
            results.append({
                'prompt': prompt,
                'subnet_data': data,
                'zero_reason': zero_reason,
                'would_be_zero': zero_reason is not None
            })
        else:
            print(f"   ❌ Subnet validation failed: {test_result['error']}")
            results.append({
                'prompt': prompt,
                'subnet_data': None,
                'zero_reason': f"Validation failed: {test_result['error']}",
                'would_be_zero': True
            })
        
        print()
        time.sleep(3)  # Brief pause between tests
    
    return results

def test_optimization_against_subnet():
    """Test our optimization framework against subnet-accurate validation"""
    
    print("🔧 TESTING OPTIMIZATION AGAINST SUBNET VALIDATION")
    print("=" * 80)
    
    optimizer = AutoPromptOptimizer({
        "risk_threshold": 0.6,
        "enable_validation": False,  # We'll validate manually with subnet validator
        "optimization_strategies": ["comprehensive", "aggressive"],
        "min_confidence": 0.3
    })
    
    optimization_results = []
    
    for i, prompt in enumerate(REAL_ZERO_SCORE_PROMPTS, 1):
        print(f"[{i}/{len(REAL_ZERO_SCORE_PROMPTS)}] Optimizing: '{prompt}'")
        print("-" * 60)
        
        # Test original with subnet validator
        print("📊 Testing ORIGINAL with subnet validation...")
        original_result = test_prompt_with_subnet_validator(prompt)
        
        # Get optimization
        print("🔧 Running optimization...")
        optimization = optimizer.optimize_for_mining(prompt)
        
        # Test optimized with subnet validator
        print("📈 Testing OPTIMIZED with subnet validation...")
        optimized_result = test_prompt_with_subnet_validator(optimization.final_prompt)
        
        # Compare results
        result = {
            'original_prompt': prompt,
            'final_prompt': optimization.final_prompt,
            'optimization_applied': optimization.optimization_applied,
            'optimization_successful': original_result["success"] and optimized_result["success"]
        }
        
        if result['optimization_successful']:
            original_data = original_result["results"]
            optimized_data = optimized_result["results"]
            
            original_score = original_data['task_fidelity_score'] if original_data['validation_passed'] else 0.0
            optimized_score = optimized_data['task_fidelity_score'] if optimized_data['validation_passed'] else 0.0
            
            result.update({
                'original_validation_score': original_data['validation_engine_score'],
                'original_alignment_score': original_data['alignment_score'],
                'original_task_fidelity': original_score,
                'original_would_be_zero': not original_data['validation_passed'] or original_data['demo_fidelity_score'] == 0.0,
                
                'optimized_validation_score': optimized_data['validation_engine_score'],
                'optimized_alignment_score': optimized_data['alignment_score'],
                'optimized_task_fidelity': optimized_score,
                'optimized_would_be_zero': not optimized_data['validation_passed'] or optimized_data['demo_fidelity_score'] == 0.0,
                
                'improvement': optimized_score - original_score,
                'crosses_subnet_threshold': optimized_data['validation_passed'] and optimized_score >= 0.6
            })
            
            print(f"📊 RESULTS:")
            print(f"   Original Task Fidelity: {original_score:.4f}")
            print(f"   Optimized Task Fidelity: {optimized_score:.4f}")
            print(f"   Improvement: {result['improvement']:+.4f}")
            print(f"   Original would be zero: {result['original_would_be_zero']}")
            print(f"   Optimized would be zero: {result['optimized_would_be_zero']}")
            print(f"   Crosses subnet threshold: {result['crosses_subnet_threshold']}")
            
            if optimization.optimization_applied:
                print(f"✨ Optimization: '{prompt}' → '{optimization.final_prompt}'")
            else:
                print(f"✅ No optimization needed")
        else:
            result.update({
                'original_error': original_result.get("error"),
                'optimized_error': optimized_result.get("error")
            })
            print(f"❌ Optimization test failed")
        
        optimization_results.append(result)
        print()
        time.sleep(3)
    
    return optimization_results

def main():
    """Run comprehensive subnet-accurate testing"""
    
    print("🚀 COMPREHENSIVE SUBNET-ACCURATE TESTING")
    print("=" * 80)
    print("Goal: Test prompt optimization against real subnet validation logic")
    print()
    
    # Phase 1: Analyze subnet behavior
    print("📋 PHASE 1: ANALYZE SUBNET BEHAVIOR")
    subnet_analysis = analyze_subnet_behavior()
    
    print("\n📋 PHASE 1 SUMMARY:")
    print("=" * 50)
    zero_count = sum(1 for r in subnet_analysis if r['would_be_zero'])
    print(f"Prompts that would get 0.0 in subnet: {zero_count}/{len(subnet_analysis)}")
    
    for result in subnet_analysis:
        status = "❌ ZERO" if result['would_be_zero'] else "✅ NON-ZERO"
        print(f"   {status}: '{result['prompt']}'")
        if result['zero_reason']:
            print(f"     Reason: {result['zero_reason']}")
    
    # Phase 2: Test optimization
    print(f"\n📋 PHASE 2: TEST OPTIMIZATION")
    optimization_analysis = test_optimization_against_subnet()
    
    print("\n📋 PHASE 2 SUMMARY:")
    print("=" * 50)
    successful_optimizations = [r for r in optimization_analysis if r.get('optimization_successful') and r.get('improvement', 0) > 0.1]
    threshold_crossers = [r for r in optimization_analysis if r.get('crosses_subnet_threshold')]
    
    print(f"Successful optimizations (>+0.1): {len(successful_optimizations)}")
    print(f"Cross subnet threshold: {len(threshold_crossers)}")
    
    if successful_optimizations:
        print(f"\n✅ SUCCESSFUL OPTIMIZATIONS:")
        for result in successful_optimizations:
            print(f"   {result['improvement']:+.4f} | '{result['original_prompt']}'")
            print(f"        → '{result['final_prompt']}'")
    
    if threshold_crossers:
        print(f"\n🎯 PROMPTS THAT CROSS SUBNET THRESHOLD:")
        for result in threshold_crossers:
            print(f"   {result['optimized_task_fidelity']:.4f} | '{result['final_prompt']}'")
    
    # Save comprehensive results
    final_results = {
        'subnet_analysis': subnet_analysis,
        'optimization_analysis': optimization_analysis,
        'summary': {
            'total_prompts': len(REAL_ZERO_SCORE_PROMPTS),
            'subnet_zeros': zero_count,
            'successful_optimizations': len(successful_optimizations),
            'threshold_crossers': len(threshold_crossers)
        }
    }
    
    with open("comprehensive_subnet_testing_results.json", "w") as f:
        json.dump(final_results, f, indent=2)
    
    print(f"\n💾 Comprehensive results saved to comprehensive_subnet_testing_results.json")
    
    # Final analysis
    print(f"\n🎯 FINAL ANALYSIS:")
    print("=" * 50)
    
    if zero_count == len(subnet_analysis):
        print("✅ All prompts correctly identified as problematic by subnet validation")
    else:
        print(f"⚠️ Only {zero_count}/{len(subnet_analysis)} prompts are actually problematic")
    
    if len(successful_optimizations) > 0:
        print(f"✅ Optimization framework successfully improves {len(successful_optimizations)} prompts")
    else:
        print("❌ Optimization framework needs improvement")
    
    if len(threshold_crossers) > 0:
        print(f"🎯 {len(threshold_crossers)} optimized prompts would succeed in actual subnet")
    else:
        print("❌ No optimized prompts reach subnet success threshold")

if __name__ == "__main__":
    main() 