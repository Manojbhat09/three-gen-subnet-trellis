#!/usr/bin/env python3
"""
Test the improved prompt optimizer with problematic prompts that were getting 0.0 fidelity
"""

import sys
sys.path.append('.')

from prompt_optimizer_v2 import TrellisPromptOptimizerV2

def test_optimizer():
    """Test the optimizer with known problematic prompts"""
    
    optimizer = TrellisPromptOptimizerV2()
    
    # Prompts that were getting 0.0 fidelity
    problematic_prompts = [
        # Transparent liquids
        "orange juice served in tall clear glass",
        
        # Contemporary/abstract
        "contemporary blue candle holder asymmetrical form",
        
        # Multi-object scenes
        "statue of saint with book",
        "glass side table holding vase of flowers",
        
        # Tiny objects
        "tiny yellow daisy plant in small pot",
        
        # Transparent stones
        "topaz stone in rectangular prism shape",
        
        # Control cases (should work well)
        "red robot with blue arms",
        "simple wooden chair",
        "metal sword with black handle"
    ]
    
    print("=" * 100)
    print("TESTING PROMPT OPTIMIZER V2")
    print("=" * 100)
    
    for i, prompt in enumerate(problematic_prompts, 1):
        print(f"\n[{i}] Testing: {prompt}")
        print("-" * 80)
        
        # Get optimization result
        result = optimizer.optimize_prompt(prompt, aggressive=True)
        
        print(f"Risk Level: {result['analysis']['risk_level']}")
        
        if result['analysis']['risk_factors']:
            print(f"Risk Factors:")
            for factor in result['analysis']['risk_factors']:
                print(f"  • {factor}")
        
        if result['improvement_expected']:
            print(f"\nOptimized Prompt:")
            print(f"  {result['optimized_prompt']}")
            
            print(f"\nApplied Strategies:")
            for strategy in result['applied_strategies']:
                print(f"  • {strategy}")
            
            # Check if we have parameter adjustments (from V2)
            if hasattr(optimizer, '_optimize_prompt_v2'):
                v2_result = optimizer._optimize_prompt_v2(prompt, aggressive=True)
                if v2_result.parameter_adjustments:
                    print(f"\nRecommended Parameters:")
                    for param, value in v2_result.parameter_adjustments.items():
                        print(f"  • {param}: {value}")
        else:
            print(f"✅ Low risk - minimal optimization needed")
    
    print("\n" + "=" * 100)
    print("SUMMARY")
    print("=" * 100)
    
    # Count high-risk prompts
    high_risk_count = 0
    for prompt in problematic_prompts[:6]:  # First 6 are known problematic
        result = optimizer.optimize_prompt(prompt)
        if result['analysis']['risk_level'] in ['CRITICAL', 'HIGH']:
            high_risk_count += 1
    
    print(f"Detected {high_risk_count}/6 known problematic prompts as high risk")
    print(f"Success rate: {(high_risk_count/6)*100:.1f}%")

if __name__ == "__main__":
    test_optimizer() 