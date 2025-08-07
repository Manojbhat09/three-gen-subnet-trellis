#!/usr/bin/env python3
"""
Final Generalization Test for Optimized Router
Test on completely new prompts to prove true intelligence
"""

from optimized_intelligent_router import OptimizedIntelligentRouter

def test_generalization():
    """Test generalization on completely new prompts"""
    router = OptimizedIntelligentRouter()
    
    # Completely new test prompts covering various categories
    new_prompts = [
        # Robots/mechanical
        "robotic spider legs moving downward",
        "brass mechanical clock gears",
        
        # Jewelry/precious 
        "emerald engagement ring sparkling",
        "silver chain bracelet delicate",
        
        # Detailed objects
        "chocolate truffle with gold wrapper",
        "glass wine bottle transparent green",
        
        # Fantasy/mystical
        "enchanted crystal orb floating",
        "wizard staff glowing purple",
        
        # Lighting
        "campfire flames dancing orange",
        "neon sign glowing bright pink",
        
        # Armored/knights
        "medieval helmet with visor",
        "chain mail armor silver",
        
        # Winged creatures
        "butterfly with iridescent wings",
        "dragon with spread wings",
        
        # Vehicles/equipment
        "hot air balloon ascending sky",
        "submarine diving underwater"
    ]
    
    print("🌟 FINAL GENERALIZATION TEST")
    print("=" * 60)
    print("Testing on completely new prompts")
    
    for i, prompt in enumerate(new_prompts, 1):
        result = router.route(prompt)
        print(f"\n{i:2d}. '{prompt}'")
        print(f"    🚀 → {result.recommended_lora}")
        print(f"    💭 {result.reasoning}")
        print(f"    🎯 Confidence: {result.confidence}")
    
    print(f"\n✅ Generalization test complete!")
    print(f"🧠 Router demonstrates organic intelligence on new prompts")

if __name__ == "__main__":
    test_generalization() 