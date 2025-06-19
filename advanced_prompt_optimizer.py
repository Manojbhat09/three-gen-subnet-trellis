#!/usr/bin/env python3

"""
Advanced TRELLIS Prompt Optimizer v2.0
Integrates structured object enhancement for 0.96+ fidelity targeting
"""

import re
from typing import Dict, List, Tuple, Optional
from structured_object_enhancer import StructuredObjectEnhancer

class AdvancedTrellisPromptOptimizer:
    """
    Next-generation prompt optimizer combining zero-fidelity prevention 
    with structured object enhancement for maximum fidelity scores.
    """
    
    def __init__(self):
        # Initialize the structured object enhancer
        self.structured_enhancer = StructuredObjectEnhancer()
        
        # Original risk factors from zero-fidelity analysis
        self.risk_factors = {
            'transparent_materials': [
                'glass', 'crystal', 'diamond', 'sapphire', 'ruby', 'emerald', 
                'amethyst', 'transparent', 'clear', 'translucent'
            ],
            'complex_scene_words': [
                'beside', 'holding', 'with', 'and', 'on', 'in', 'around', 
                'near', 'next to', 'featuring', 'containing'
            ],
            'weapons': [
                'sword', 'katana', 'dagger', 'knife', 'revolver', 'gun', 
                'rifle', 'pistol', 'blade', 'spear'
            ],
            'tools': [
                'drill', 'hammer', 'wrench', 'scissors', 'screwdriver', 
                'saw', 'pliers', 'chisel'
            ],
            'reflective_effects': [
                'shiny', 'polished', 'glossy', 'reflective', 'metallic', 
                'chrome', 'mirror', 'sparkling'
            ]
        }
        
        # Enhanced optimization strategies
        self.optimization_strategies = {
            'transparent_objects': [
                'wbgmsst', 'solid detailed structure', 'well-defined edges',
                'opaque rendering style', 'matte surface finish'
            ],
            'complex_scenes': [
                'single object focus', 'isolated subject', 'clean background',
                'centered composition', 'minimal context'
            ],
            'weapons': [
                'detailed craftsmanship', 'professional product photography',
                'museum quality', 'historical accuracy', 'ceremonial design'
            ],
            'tools': [
                'professional grade', 'precision engineering', 'industrial design',
                'ergonomic construction', 'commercial quality'
            ],
            'structured_objects': [
                '3D isometric object', 'clean design', 'geometric precision',
                'technical illustration', 'product render'
            ],
            'general_quality': [
                'high resolution', 'detailed texture', 'perfect lighting',
                'studio photography', 'pristine condition'
            ]
        }
        
        # Fidelity boost modifiers for different score targets
        self.fidelity_boosters = {
            'ultra_high': [  # For 0.96+ targeting
                'museum exhibition quality', 'photorealistic detail',
                'precision crafted', 'flawless construction',
                'professional studio lighting'
            ],
            'high': [  # For 0.90+ targeting
                'high quality', 'detailed design', 'well crafted',
                'clean finish', 'professional grade'
            ],
            'standard': [  # For general improvement
                'good quality', 'neat design', 'solid construction'
            ]
        }

    def analyze_prompt_comprehensive(self, prompt: str) -> Dict:
        """Comprehensive analysis including both risk and enhancement opportunities."""
        analysis = {
            'original_prompt': prompt,
            'risk_score': 0,
            'risk_factors': [],
            'is_structured_object': False,
            'object_category': None,
            'enhancement_opportunities': [],
            'recommended_target': 'standard'
        }
        
        prompt_lower = prompt.lower()
        
        # 1. Risk factor analysis
        for factor_type, keywords in self.risk_factors.items():
            matches = [word for word in keywords if word in prompt_lower]
            if matches:
                analysis['risk_factors'].append({
                    'type': factor_type,
                    'matches': matches,
                    'weight': self._get_risk_weight(factor_type)
                })
                analysis['risk_score'] += len(matches) * self._get_risk_weight(factor_type)
        
        # 2. Structured object analysis
        category, main_object = self.structured_enhancer.identify_object_category(prompt)
        if category:
            analysis['is_structured_object'] = True
            analysis['object_category'] = category
            analysis['main_object'] = main_object
            analysis['recommended_target'] = 'ultra_high'  # Structured objects can achieve 0.96+
        
        # 3. Enhancement opportunities
        analysis['enhancement_opportunities'] = self._identify_enhancement_opportunities(prompt, analysis)
        
        return analysis

    def _get_risk_weight(self, factor_type: str) -> int:
        """Get risk weight for different factor types."""
        weights = {
            'transparent_materials': 3,
            'complex_scene_words': 2,
            'weapons': 2,
            'tools': 1,
            'reflective_effects': 2
        }
        return weights.get(factor_type, 1)

    def _identify_enhancement_opportunities(self, prompt: str, analysis: Dict) -> List[str]:
        """Identify specific enhancement opportunities."""
        opportunities = []
        prompt_lower = prompt.lower()
        
        # Check for missing material specification
        if analysis['is_structured_object']:
            cat_data = self.structured_enhancer.object_categories[analysis['object_category']]
            has_material = any(mat in prompt_lower for mat in cat_data['materials'])
            if not has_material:
                opportunities.append('material_specification')
        
        # Check for missing functional details
        if analysis['is_structured_object']:
            cat_data = self.structured_enhancer.object_categories[analysis['object_category']]
            has_functional = any(part in prompt_lower for part in cat_data['functional_parts'])
            if not has_functional:
                opportunities.append('functional_details')
        
        # Check for missing surface quality
        surface_words = ['smooth', 'rough', 'textured', 'polished', 'matte', 'glossy']
        if not any(word in prompt_lower for word in surface_words):
            opportunities.append('surface_quality')
        
        # Check for missing size/quality descriptors
        quality_words = ['professional', 'high-quality', 'premium', 'detailed', 'precision']
        if not any(word in prompt_lower for word in quality_words):
            opportunities.append('quality_descriptors')
        
        return opportunities

    def optimize_prompt_advanced(self, prompt: str, target_fidelity: str = 'auto', 
                                aggressive: bool = False) -> Tuple[str, Dict]:
        """
        Advanced prompt optimization with structured object enhancement.
        
        Args:
            prompt: Original prompt
            target_fidelity: 'ultra_high' (0.96+), 'high' (0.90+), 'standard', or 'auto'
            aggressive: Apply aggressive optimizations
            
        Returns:
            Tuple of (optimized_prompt, analysis_data)
        """
        # Comprehensive analysis
        analysis = self.analyze_prompt_comprehensive(prompt)
        
        # Determine target fidelity
        if target_fidelity == 'auto':
            if analysis['is_structured_object']:
                target_fidelity = 'ultra_high'
            elif analysis['risk_score'] > 5:
                target_fidelity = 'high'
            else:
                target_fidelity = 'standard'
        
        analysis['target_fidelity'] = target_fidelity
        
        # Start with original prompt
        optimized = prompt
        applied_strategies = []
        
        # 1. Apply structured object enhancement if applicable
        if analysis['is_structured_object']:
            if target_fidelity == 'ultra_high':
                optimized = self.structured_enhancer.enhance_structured_object(optimized, aggressive=True)
                applied_strategies.append('structured_object_ultra_enhancement')
            else:
                optimized = self.structured_enhancer.enhance_structured_object(optimized, aggressive=False)
                applied_strategies.append('structured_object_enhancement')
        
        # 2. Apply risk mitigation strategies
        for risk_factor in analysis['risk_factors']:
            factor_type = risk_factor['type']
            if factor_type in self.optimization_strategies:
                strategies = self.optimization_strategies[factor_type]
                if aggressive:
                    # Apply more strategies for aggressive mode
                    selected_strategies = strategies[:3]
                else:
                    selected_strategies = strategies[:2]
                
                optimized += f", {', '.join(selected_strategies)}"
                applied_strategies.extend(selected_strategies)
        
        # 3. Apply fidelity boosters based on target
        if target_fidelity in self.fidelity_boosters:
            boosters = self.fidelity_boosters[target_fidelity]
            if aggressive:
                selected_boosters = boosters[:3]
            else:
                selected_boosters = boosters[:2]
            
            optimized += f", {', '.join(selected_boosters)}"
            applied_strategies.extend(selected_boosters)
        
        # 4. Always add general quality improvements
        general_strategies = self.optimization_strategies['general_quality']
        if target_fidelity == 'ultra_high':
            optimized += f", {', '.join(general_strategies[:2])}"
            applied_strategies.extend(general_strategies[:2])
        
        # Update analysis with optimization results
        analysis['optimized_prompt'] = optimized
        analysis['applied_strategies'] = applied_strategies
        analysis['optimization_applied'] = len(applied_strategies) > 0
        analysis['estimated_improvement'] = self._estimate_improvement(analysis)
        
        return optimized, analysis

    def _estimate_improvement(self, analysis: Dict) -> str:
        """Estimate the expected fidelity improvement."""
        if analysis['is_structured_object'] and analysis['target_fidelity'] == 'ultra_high':
            return "Expected: 0.95-0.97+ (Ultra-high structured object)"
        elif analysis['risk_score'] > 5:
            return "Expected: 0.85-0.92 (High-risk mitigation)"
        elif analysis['is_structured_object']:
            return "Expected: 0.90-0.95 (Structured object enhancement)"
        else:
            return "Expected: 0.88-0.93 (General optimization)"

    def generate_multiple_variants(self, prompt: str) -> List[Tuple[str, str, Dict]]:
        """Generate multiple optimized variants for A/B testing."""
        variants = []
        
        # Standard optimization
        opt_standard, analysis_standard = self.optimize_prompt_advanced(
            prompt, target_fidelity='standard', aggressive=False
        )
        variants.append((opt_standard, "Standard optimization", analysis_standard))
        
        # High-fidelity optimization
        opt_high, analysis_high = self.optimize_prompt_advanced(
            prompt, target_fidelity='high', aggressive=False
        )
        variants.append((opt_high, "High-fidelity optimization", analysis_high))
        
        # Ultra-high aggressive (if structured object)
        analysis_base = self.analyze_prompt_comprehensive(prompt)
        if analysis_base['is_structured_object']:
            opt_ultra, analysis_ultra = self.optimize_prompt_advanced(
                prompt, target_fidelity='ultra_high', aggressive=True
            )
            variants.append((opt_ultra, "Ultra-high aggressive", analysis_ultra))
        
        return variants

def main():
    """Demo the advanced prompt optimizer."""
    optimizer = AdvancedTrellisPromptOptimizer()
    
    # Test prompts covering different scenarios
    test_prompts = [
        "hammer",  # Simple structured object
        "glass vase with flowers",  # Transparent + complex scene
        "metal sword with jeweled hilt",  # Weapon + complex
        "shiny chrome robot",  # Reflective + structured
        "wooden violin",  # High-performing category
        "crystal dragon figurine"  # Transparent + complex
    ]
    
    print("=== ADVANCED PROMPT OPTIMIZATION DEMO ===\n")
    
    for prompt in test_prompts:
        print(f"🔍 ANALYZING: '{prompt}'")
        print("-" * 50)
        
        # Generate variants
        variants = optimizer.generate_multiple_variants(prompt)
        
        for i, (optimized, variant_type, analysis) in enumerate(variants, 1):
            print(f"\n{i}. {variant_type}:")
            print(f"   Original: '{prompt}'")
            print(f"   Optimized: '{optimized}'")
            print(f"   {analysis['estimated_improvement']}")
            
            if analysis['risk_factors']:
                risk_types = [rf['type'] for rf in analysis['risk_factors']]
                print(f"   Risk factors: {', '.join(risk_types)}")
            
            if analysis['is_structured_object']:
                print(f"   Category: {analysis['object_category']} (structured object)")
        
        print("\n" + "="*70 + "\n")

if __name__ == "__main__":
    main() 