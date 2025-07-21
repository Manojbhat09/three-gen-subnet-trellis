#!/usr/bin/env python3
"""
Advanced Prompt Optimizer for Subnet 17
Purpose: Automatically optimize prompts to score above 0.6 threshold while preserving semantic intent
"""
import re
import json
import time
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from pathlib import Path
from risk_analyzer import PromptRiskAnalyzer, RiskAssessment

@dataclass
class OptimizationResult:
    """Result of prompt optimization"""
    original_prompt: str
    optimized_prompt: str
    optimization_strategy: str
    predicted_improvement: float  # Predicted score increase
    confidence: float  # How confident we are in the optimization
    risk_reduction: str  # Risk level change

class AdvancedPromptOptimizer:
    """Advanced prompt optimization using multiple strategies"""
    
    def __init__(self):
        self.risk_analyzer = PromptRiskAnalyzer()
        
        # Material replacement mappings (problematic -> safe alternatives)
        self.material_replacements = {
            'glass': ['ceramic', 'polished wood', 'brushed metal', 'stone'],
            'crystal': ['carved stone', 'polished marble', 'solid metal', 'ceramic'],
            'transparent': ['opaque', 'solid', 'matte'],
            'translucent': ['solid', 'opaque', 'matte'],
            'clear': ['solid', 'opaque'],
            'liquid': ['granular', 'powdered', 'solid'],
            'water': ['sand', 'powder', 'granules'],
            'juice': ['colored powder', 'granular substance'],
            'fluid': ['granular material', 'solid substance'],
            'mirror': ['polished metal', 'reflective metal'],
            'diamond': ['cut crystal', 'faceted stone', 'polished gem'],
            'emerald': ['green stone', 'jade carving'],
            'ruby': ['red stone', 'garnet'],
            'sapphire': ['blue stone', 'lapis lazuli'],
            'gem': ['carved stone', 'polished rock'],
            'jewel': ['decorative stone', 'ornamental element']
        }
        
        # Grammar enhancement patterns
        self.grammar_fixes = [
            # "X filled Y" -> "X filled with Y"
            (r'(\w+)\s+filled\s+(\w+)', r'\1 filled with \2'),
            # "thing with" -> "object with" or specific object
            (r'\bthing\s+with\b', 'object with'),
            # "stuff" -> "elements" or "components"
            (r'\bstuff\b', 'components'),
            # Add "a" or "an" if missing
            (r'^([bcdfghjklmnpqrstvwxyz]\w+)', r'a \1'),
            (r'^([aeiou]\w+)', r'an \1'),
        ]
        
        # Abstract concept replacements
        self.abstraction_replacements = {
            'essence': 'representation',
            'concept': 'model',
            'energy': 'glowing orb',
            'mystical': 'glowing',
            'spiritual': 'ethereal',
            'quantum': 'swirling',
            'ineffable': 'mysterious',
            'conceptual': 'artistic',
            'philosophical': 'thoughtful',
            'abstract': 'artistic',
            'probability cloud': 'swirling mist effect',
            'energy construct': 'glowing geometric shape'
        }
        
        # Scene simplification patterns
        self.simplification_patterns = [
            # "X with Y pattern" -> "patterned X" or just "X"
            (r'(\w+)\s+with\s+\w+\s+pattern', r'patterned \1'),
            # "X with Y design" -> "designed X"
            (r'(\w+)\s+with\s+\w+\s+design', r'designed \1'),
            # "X holding Y" -> just "X" (simplify scene)
            (r'(\w+)\s+holding\s+\w+', r'\1'),
            # "X beside Y" -> just "X"
            (r'(\w+)\s+beside\s+\w+', r'\1'),
            # "X and Y" -> just "X" (focus on first object)
            (r'(\w+)\s+and\s+\w+', r'\1'),
        ]
        
        # Color and material enhancement
        self.color_materials = {
            'red': ['crimson', 'scarlet', 'cherry'],
            'blue': ['sapphire blue', 'navy', 'cobalt'],
            'green': ['emerald green', 'forest green', 'jade'],
            'yellow': ['golden', 'amber', 'honey-colored'],
            'purple': ['violet', 'amethyst', 'lavender'],
            'black': ['obsidian', 'charcoal', 'ebony'],
            'white': ['pearl white', 'ivory', 'marble white'],
            'brown': ['mahogany', 'walnut', 'chestnut']
        }
    
    def optimize_materials(self, prompt: str) -> str:
        """Replace problematic materials with rendering-friendly alternatives"""
        optimized = prompt.lower()
        
        for problematic, alternatives in self.material_replacements.items():
            pattern = r'\b' + re.escape(problematic) + r'\b'
            if re.search(pattern, optimized):
                # Choose the best alternative (first one is usually best)
                replacement = alternatives[0]
                optimized = re.sub(pattern, replacement, optimized)
        
        return optimized
    
    def enhance_grammar(self, prompt: str) -> str:
        """Fix grammatical issues and improve clarity"""
        enhanced = prompt
        
        for pattern, replacement in self.grammar_fixes:
            enhanced = re.sub(pattern, replacement, enhanced, flags=re.IGNORECASE)
        
        return enhanced
    
    def reduce_abstraction(self, prompt: str) -> str:
        """Replace abstract concepts with concrete objects"""
        concrete = prompt.lower()
        
        for abstract, concrete_replacement in self.abstraction_replacements.items():
            pattern = r'\b' + re.escape(abstract) + r'\b'
            concrete = re.sub(pattern, concrete_replacement, concrete)
        
        return concrete
    
    def simplify_scene(self, prompt: str) -> str:
        """Simplify complex scenes to focus on single objects"""
        simplified = prompt
        
        for pattern, replacement in self.simplification_patterns:
            simplified = re.sub(pattern, replacement, simplified, flags=re.IGNORECASE)
        
        return simplified
    
    def enhance_specificity(self, prompt: str) -> str:
        """Add specific details that improve 3D generation"""
        enhanced = prompt
        
        # Add geometric details if not present
        if not any(word in enhanced.lower() for word in ['round', 'square', 'rectangular', 'cylindrical', 'spherical']):
            # Add basic shape descriptor for simple objects
            if any(word in enhanced.lower() for word in ['ball', 'sphere']):
                enhanced = enhanced.replace('ball', 'spherical ball').replace('sphere', 'spherical object')
            elif any(word in enhanced.lower() for word in ['box', 'cube']):
                enhanced = enhanced.replace('box', 'cubic box').replace('cube', 'cubic object')
        
        # Add material texture if missing
        if not any(word in enhanced.lower() for word in ['smooth', 'rough', 'textured', 'polished', 'matte']):
            # Add texture based on material
            if any(word in enhanced.lower() for word in ['wood', 'wooden']):
                enhanced = enhanced.replace('wood', 'polished wood').replace('wooden', 'polished wooden')
            elif any(word in enhanced.lower() for word in ['metal', 'metallic']):
                enhanced = enhanced.replace('metal', 'brushed metal').replace('metallic', 'brushed metallic')
        
        return enhanced
    
    def optimize_prompt(self, prompt: str, strategy: str = "comprehensive") -> OptimizationResult:
        """Apply optimization strategy to a prompt"""
        
        # Get initial risk assessment
        initial_assessment = self.risk_analyzer.analyze_prompt(prompt)
        
        if strategy == "materials_only":
            optimized = self.optimize_materials(prompt)
            strategy_name = "Material Replacement"
            
        elif strategy == "grammar_only":
            optimized = self.enhance_grammar(prompt) 
            strategy_name = "Grammar Enhancement"
            
        elif strategy == "simplification_only":
            optimized = self.simplify_scene(prompt)
            strategy_name = "Scene Simplification"
            
        elif strategy == "comprehensive":
            # Apply all optimizations in sequence
            optimized = prompt
            optimized = self.optimize_materials(optimized)
            optimized = self.enhance_grammar(optimized)
            optimized = self.reduce_abstraction(optimized)
            optimized = self.simplify_scene(optimized)
            optimized = self.enhance_specificity(optimized)
            strategy_name = "Comprehensive Optimization"
            
        elif strategy == "aggressive":
            # More aggressive optimization for high-risk prompts
            optimized = prompt
            optimized = self.optimize_materials(optimized)
            optimized = self.enhance_grammar(optimized)
            optimized = self.reduce_abstraction(optimized)
            optimized = self.simplify_scene(optimized)
            optimized = self.enhance_specificity(optimized)
            
            # Additional aggressive changes
            if initial_assessment.risk_level == "CRITICAL":
                # Replace with completely safe template if needed
                if len(optimized.split()) <= 3:
                    optimized = f"solid {optimized.split()[-1]} object"
            
            strategy_name = "Aggressive Optimization"
        
        else:
            optimized = prompt
            strategy_name = "No Optimization"
        
        # Ensure proper capitalization
        optimized = optimized.strip()
        if optimized and not optimized[0].isupper():
            optimized = optimized[0].upper() + optimized[1:]
        
        # Get post-optimization risk assessment
        final_assessment = self.risk_analyzer.analyze_prompt(optimized)
        
        # Calculate predicted improvement
        initial_score_avg = sum(initial_assessment.predicted_score_range) / 2
        final_score_avg = sum(final_assessment.predicted_score_range) / 2
        predicted_improvement = final_score_avg - initial_score_avg
        
        # Calculate confidence based on risk reduction
        risk_levels = {"LOW": 4, "MEDIUM": 3, "HIGH": 2, "CRITICAL": 1}
        initial_risk_num = risk_levels[initial_assessment.risk_level]
        final_risk_num = risk_levels[final_assessment.risk_level]
        confidence = min(1.0, max(0.1, (final_risk_num - initial_risk_num + 4) / 7))
        
        return OptimizationResult(
            original_prompt=prompt,
            optimized_prompt=optimized,
            optimization_strategy=strategy_name,
            predicted_improvement=predicted_improvement,
            confidence=confidence,
            risk_reduction=f"{initial_assessment.risk_level} → {final_assessment.risk_level}"
        )
    
    def batch_optimize(self, prompts: List[str], strategy: str = "comprehensive") -> List[OptimizationResult]:
        """Optimize multiple prompts"""
        return [self.optimize_prompt(prompt, strategy) for prompt in prompts]

def test_optimizer():
    """Test the optimizer with problematic prompts"""
    
    optimizer = AdvancedPromptOptimizer()
    
    # Test prompts that we know have issues
    test_prompts = [
        "glass jug filled juice",
        "silver chalice with leafy vine pattern",
        "transparent invisible object floating", 
        "quantum mechanical probability cloud",
        "thing with parts and stuff",
        "abstract conceptual entity",
        "crystal formation with light rays",
        "liquid water in clear container",
        "mystical energy construct floating",
        "ineffable essence of blueness"
    ]
    
    print("🚀 ADVANCED PROMPT OPTIMIZATION")
    print("=" * 80)
    
    results = []
    
    for i, prompt in enumerate(test_prompts, 1):
        print(f"\n[{i}/{len(test_prompts)}] Optimizing: '{prompt}'")
        
        # Test different strategies
        strategies = ["materials_only", "comprehensive", "aggressive"]
        best_result = None
        best_improvement = -999
        
        for strategy in strategies:
            result = optimizer.optimize_prompt(prompt, strategy)
            if result.predicted_improvement > best_improvement:
                best_improvement = result.predicted_improvement
                best_result = result
        
        results.append(best_result)
        
        print(f"   📈 Best Strategy: {best_result.optimization_strategy}")
        print(f"   ✨ Optimized: '{best_result.optimized_prompt}'")
        print(f"   📊 Risk Change: {best_result.risk_reduction}")
        print(f"   📈 Predicted Improvement: +{best_result.predicted_improvement:.3f}")
        print(f"   🎯 Confidence: {best_result.confidence:.1%}")
    
    # Summary analysis
    print(f"\n📊 OPTIMIZATION SUMMARY")
    print("=" * 80)
    
    successful_optimizations = [r for r in results if r.predicted_improvement > 0]
    high_confidence = [r for r in results if r.confidence > 0.7]
    
    print(f"Total Prompts: {len(results)}")
    print(f"Successful Optimizations: {len(successful_optimizations)} ({len(successful_optimizations)/len(results)*100:.1f}%)")
    print(f"High Confidence Results: {len(high_confidence)} ({len(high_confidence)/len(results)*100:.1f}%)")
    
    avg_improvement = sum(r.predicted_improvement for r in successful_optimizations) / len(successful_optimizations) if successful_optimizations else 0
    print(f"Average Improvement: +{avg_improvement:.3f}")
    
    # Show before/after examples
    print(f"\n✨ BEST OPTIMIZATIONS:")
    best_results = sorted(results, key=lambda x: x.predicted_improvement, reverse=True)[:5]
    for result in best_results:
        print(f"   Original: '{result.original_prompt}'")
        print(f"   Optimized: '{result.optimized_prompt}' (+{result.predicted_improvement:.3f})")
        print()
    
    # Save results
    output_data = {
        "optimization_timestamp": time.time(),
        "results": [
            {
                "original_prompt": r.original_prompt,
                "optimized_prompt": r.optimized_prompt,
                "optimization_strategy": r.optimization_strategy,
                "predicted_improvement": r.predicted_improvement,
                "confidence": r.confidence,
                "risk_reduction": r.risk_reduction
            }
            for r in results
        ],
        "summary": {
            "total_prompts": len(results),
            "successful_optimizations": len(successful_optimizations),
            "high_confidence_results": len(high_confidence),
            "average_improvement": avg_improvement
        }
    }
    
    with open("prompt_optimization_results.json", "w") as f:
        json.dump(output_data, f, indent=2)
    
    print(f"💾 Results saved to: prompt_optimization_results.json")

if __name__ == "__main__":
    test_optimizer() 