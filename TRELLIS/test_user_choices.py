#!/usr/bin/env python3
"""
Test script for user choice functions
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from flux_trellis_bpt_retextured_optimized import get_user_quality_choice, get_user_bpt_choice, get_user_shape_optimization_choice, get_user_intelligent_simplification_choice, get_user_simplification_method_choice

if __name__ == "__main__":
    print("🧪 Testing User Choice Functions")
    print("=" * 50)
    
    # Test quality choice
    quality = get_user_quality_choice()
    print(f"Quality selected: {quality}")
    
    # Test BPT choice
    bpt_choice = get_user_bpt_choice()
    print(f"BPT choice: {bpt_choice}")
    
    # Test shape optimization choice
    shape_opt = get_user_shape_optimization_choice()
    print(f"Shape optimization: {shape_opt}")
    
    # Test intelligent simplification choice
    intelligent_simplify = get_user_intelligent_simplification_choice()
    print(f"Intelligent simplification: {intelligent_simplify}")
    
    # Test simplification method choice (only if simplification is enabled)
    simplification_methods = None
    if intelligent_simplify['use_simplification']:
        simplification_methods = get_user_simplification_method_choice()
        print(f"Simplification methods: {simplification_methods}")
    
    print("\n✅ All user choice functions working correctly!")
    print(f"Final settings:")
    print(f"- Quality: {quality}")
    print(f"- BPT: {bpt_choice['name']}")
    print(f"- Shape Optimization: {'Enabled' if shape_opt else 'Disabled'}")
    print(f"- Intelligent Simplification: {intelligent_simplify['name']}")
    if intelligent_simplify['use_simplification']:
        print(f"  ├─ Target Faces: {intelligent_simplify['target_faces']:,}")
        if simplification_methods:
            print(f"  └─ Methods: {' → '.join(simplification_methods['methods'])}") 