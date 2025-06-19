#!/usr/bin/env python3
"""
Real-Time TRELLIS Prompt Optimizer
Detects and optimizes prompts that are likely to get 0.0 fidelity scores.
Based on analysis of actual zero fidelity patterns from continuous mining.
"""

import re
import json
import argparse
from typing import Dict, List, Tuple

class TrellisPromptOptimizer:
    def __init__(self):
        """Initialize the optimizer with patterns from zero fidelity analysis"""
        
        # High-risk keywords that correlate with 0.0 fidelity
        self.transparent_keywords = [
            'glass', 'crystal', 'sapphire', 'diamond', 'ruby', 'emerald', 
            'transparent', 'translucent', 'clear', 'crystalline', 'quartz', 'amethyst'
        ]
        
        self.complex_scene_keywords = [
            'beside', 'holding', 'with', 'and', 'on', 'in', 'around', 'wrapped'
        ]
        
        self.reflective_keywords = [
            'shiny', 'glossy', 'reflective', 'glowing', 'sparkling', 'polished',
            'blue-tinted', 'deep red hue', 'violet hues'
        ]
        
        self.weapon_keywords = [
            'sword', 'katana', 'knife', 'dagger', 'axe', 'spear', 'revolver'
        ]
        
        self.tool_keywords = [
            'drill', 'scissors', 'wrench', 'hammer'
        ]
        
        self.furniture_keywords = [
            'door', 'table', 'lampstand', 'harp', 'chair'
        ]
        
        # Optimization strategies
        self.transparent_optimizations = [
            'wbgmsst',
            'solid detailed structure',
            'well-defined edges',
            'opaque rendering style',
            'detailed surface texture'
        ]
        
        self.complex_scene_optimizations = [
            'single object focus',
            'isolated object',
            'clean background'
        ]
        
        self.weapon_tool_optimizations = [
            'detailed craftsmanship',
            'professional product photography',
            'single object'
        ]
        
        self.general_optimizations = [
            'wbgmsst',
            'detailed 3D object',
            'professional rendering',
            '3D isometric view'
        ]
    
    def analyze_prompt(self, prompt: str) -> Dict:
        """Analyze a prompt for risk factors"""
        prompt_lower = prompt.lower()
        
        analysis = {
            'original_prompt': prompt,
            'risk_factors': [],
            'risk_level': 'LOW',
            'transparent_materials': [],
            'complex_scene_words': [],
            'reflective_effects': [],
            'weapons': [],
            'tools': [],
            'furniture': []
        }
        
        # Check for transparent materials
        for keyword in self.transparent_keywords:
            if keyword in prompt_lower:
                analysis['transparent_materials'].append(keyword)
        
        # Check for complex scene indicators
        for keyword in self.complex_scene_keywords:
            if keyword in prompt_lower:
                analysis['complex_scene_words'].append(keyword)
        
        # Check for reflective effects
        for keyword in self.reflective_keywords:
            if keyword in prompt_lower:
                analysis['reflective_effects'].append(keyword)
        
        # Check for weapons
        for keyword in self.weapon_keywords:
            if keyword in prompt_lower:
                analysis['weapons'].append(keyword)
        
        # Check for tools
        for keyword in self.tool_keywords:
            if keyword in prompt_lower:
                analysis['tools'].append(keyword)
        
        # Check for furniture
        for keyword in self.furniture_keywords:
            if keyword in prompt_lower:
                analysis['furniture'].append(keyword)
        
        # Determine risk factors and level
        if analysis['transparent_materials']:
            analysis['risk_factors'].append(f"Transparent materials: {', '.join(analysis['transparent_materials'])}")
        
        if len(analysis['complex_scene_words']) >= 2:
            analysis['risk_factors'].append(f"Complex scene: {', '.join(analysis['complex_scene_words'])}")
        
        if analysis['reflective_effects']:
            analysis['risk_factors'].append(f"Reflective effects: {', '.join(analysis['reflective_effects'])}")
        
        if analysis['weapons']:
            analysis['risk_factors'].append(f"Weapons: {', '.join(analysis['weapons'])}")
        
        if analysis['tools']:
            analysis['risk_factors'].append(f"Tools: {', '.join(analysis['tools'])}")
        
        if analysis['furniture']:
            analysis['risk_factors'].append(f"Furniture: {', '.join(analysis['furniture'])}")
        
        # Calculate risk level
        risk_score = 0
        risk_score += len(analysis['transparent_materials']) * 3  # High risk
        risk_score += max(0, len(analysis['complex_scene_words']) - 1) * 2  # Medium risk for 2+ words
        risk_score += len(analysis['reflective_effects']) * 2  # Medium risk
        risk_score += len(analysis['weapons']) * 2  # Medium risk
        risk_score += len(analysis['tools']) * 1  # Low risk
        risk_score += len(analysis['furniture']) * 1  # Low risk
        
        if risk_score >= 6:
            analysis['risk_level'] = 'HIGH'
        elif risk_score >= 3:
            analysis['risk_level'] = 'MEDIUM'
        else:
            analysis['risk_level'] = 'LOW'
        
        return analysis
    
    def optimize_prompt(self, prompt: str, aggressive: bool = False) -> Dict:
        """Optimize a prompt to reduce zero fidelity risk"""
        analysis = self.analyze_prompt(prompt)
        
        optimizations = []
        applied_strategies = []
        
        # Apply optimizations based on detected patterns
        if analysis['transparent_materials']:
            optimizations.extend(self.transparent_optimizations[:3])
            applied_strategies.append("Transparent object handling")
        
        if len(analysis['complex_scene_words']) >= 2:
            optimizations.extend(self.complex_scene_optimizations[:2])
            applied_strategies.append("Complex scene simplification")
        
        if analysis['weapons'] or analysis['tools']:
            optimizations.extend(self.weapon_tool_optimizations[:2])
            applied_strategies.append("Weapon/tool enhancement")
        
        # Add general optimizations if no specific ones or if aggressive mode
        if not optimizations or aggressive:
            optimizations.extend(self.general_optimizations[:2])
            applied_strategies.append("General quality enhancement")
        
        # Remove duplicates while preserving order
        seen = set()
        unique_optimizations = []
        for opt in optimizations:
            if opt not in seen:
                unique_optimizations.append(opt)
                seen.add(opt)
        
        # Limit to top 3-4 optimizations to avoid prompt bloat
        final_optimizations = unique_optimizations[:4]
        
        # Generate optimized prompt
        if final_optimizations:
            optimized_prompt = f"{', '.join(final_optimizations)}, {prompt}, 3D isometric object, clean design"
        else:
            optimized_prompt = f"wbgmsst, {prompt}, detailed 3D object, professional rendering"
        
        return {
            'analysis': analysis,
            'optimized_prompt': optimized_prompt,
            'applied_strategies': applied_strategies,
            'optimization_keywords': final_optimizations,
            'improvement_expected': analysis['risk_level'] != 'LOW'
        }
    
    def batch_optimize(self, prompts: List[str], aggressive: bool = False) -> List[Dict]:
        """Optimize a batch of prompts"""
        return [self.optimize_prompt(prompt, aggressive) for prompt in prompts]
    
    def get_optimization_stats(self, prompts: List[str]) -> Dict:
        """Get statistics on how many prompts need optimization"""
        results = self.batch_optimize(prompts)
        
        stats = {
            'total_prompts': len(prompts),
            'high_risk': sum(1 for r in results if r['analysis']['risk_level'] == 'HIGH'),
            'medium_risk': sum(1 for r in results if r['analysis']['risk_level'] == 'MEDIUM'),
            'low_risk': sum(1 for r in results if r['analysis']['risk_level'] == 'LOW'),
            'needs_optimization': sum(1 for r in results if r['improvement_expected']),
            'transparent_objects': sum(1 for r in results if r['analysis']['transparent_materials']),
            'complex_scenes': sum(1 for r in results if len(r['analysis']['complex_scene_words']) >= 2),
            'weapons_tools': sum(1 for r in results if r['analysis']['weapons'] or r['analysis']['tools'])
        }
        
        stats['risk_percentage'] = (stats['high_risk'] + stats['medium_risk']) / stats['total_prompts'] * 100
        
        return stats

def main():
    """Command line interface for the optimizer"""
    parser = argparse.ArgumentParser(description='TRELLIS Prompt Optimizer - Reduce zero fidelity risk')
    parser.add_argument('prompt', nargs='?', help='Prompt to optimize')
    parser.add_argument('--file', '-f', help='File containing prompts (one per line)')
    parser.add_argument('--aggressive', '-a', action='store_true', help='Apply aggressive optimizations')
    parser.add_argument('--analyze-only', action='store_true', help='Only analyze, don\'t optimize')
    parser.add_argument('--stats', action='store_true', help='Show statistics for batch processing')
    parser.add_argument('--examples', action='store_true', help='Show example optimizations')
    
    args = parser.parse_args()
    
    optimizer = TrellisPromptOptimizer()
    
    if args.examples:
        print("🔍 TRELLIS PROMPT OPTIMIZER - EXAMPLES")
        print("=" * 60)
        
        example_prompts = [
            "glass side table holding vase of flowers",
            "cool blue-tinted sapphire obelisk",
            "rusted ancient katana sword",
            "silver-plated revolver beside glass vase",
            "shiny obsidian rock black and pointed",
            "simple wooden chair",
            "red sports car",
            "metallic robot"
        ]
        
        for i, prompt in enumerate(example_prompts, 1):
            result = optimizer.optimize_prompt(prompt)
            analysis = result['analysis']
            
            print(f"\n{i}. Original: {prompt}")
            print(f"   Risk Level: {analysis['risk_level']}")
            if analysis['risk_factors']:
                print(f"   Risk Factors: {'; '.join(analysis['risk_factors'])}")
            print(f"   Optimized: {result['optimized_prompt']}")
            if result['applied_strategies']:
                print(f"   Strategies: {', '.join(result['applied_strategies'])}")
        
        return
    
    if args.file:
        # Process file
        try:
            with open(args.file, 'r') as f:
                prompts = [line.strip() for line in f if line.strip()]
        except FileNotFoundError:
            print(f"❌ File not found: {args.file}")
            return
        
        print(f"📂 Processing {len(prompts)} prompts from {args.file}")
        
        if args.stats:
            stats = optimizer.get_optimization_stats(prompts)
            print(f"\n📊 OPTIMIZATION STATISTICS")
            print("-" * 40)
            print(f"Total prompts: {stats['total_prompts']}")
            print(f"High risk: {stats['high_risk']} ({stats['high_risk']/stats['total_prompts']*100:.1f}%)")
            print(f"Medium risk: {stats['medium_risk']} ({stats['medium_risk']/stats['total_prompts']*100:.1f}%)")
            print(f"Low risk: {stats['low_risk']} ({stats['low_risk']/stats['total_prompts']*100:.1f}%)")
            print(f"Need optimization: {stats['needs_optimization']} ({stats['needs_optimization']/stats['total_prompts']*100:.1f}%)")
            print(f"\nRisk factors:")
            print(f"  Transparent objects: {stats['transparent_objects']}")
            print(f"  Complex scenes: {stats['complex_scenes']}")
            print(f"  Weapons/tools: {stats['weapons_tools']}")
            print(f"\nOverall risk percentage: {stats['risk_percentage']:.1f}%")
        else:
            results = optimizer.batch_optimize(prompts, args.aggressive)
            
            for i, result in enumerate(results, 1):
                analysis = result['analysis']
                
                if args.analyze_only:
                    print(f"\n{i}. {analysis['original_prompt']}")
                    print(f"   Risk: {analysis['risk_level']}")
                    if analysis['risk_factors']:
                        print(f"   Factors: {'; '.join(analysis['risk_factors'])}")
                else:
                    print(f"\n{i}. Original: {analysis['original_prompt']}")
                    print(f"   Risk: {analysis['risk_level']}")
                    if analysis['risk_factors']:
                        print(f"   Issues: {'; '.join(analysis['risk_factors'])}")
                    if result['improvement_expected']:
                        print(f"   Optimized: {result['optimized_prompt']}")
                        print(f"   Applied: {', '.join(result['applied_strategies'])}")
                    else:
                        print(f"   ✅ No optimization needed")
    
    elif args.prompt:
        # Process single prompt
        if args.analyze_only:
            analysis = optimizer.analyze_prompt(args.prompt)
            print(f"🔍 PROMPT ANALYSIS")
            print("-" * 30)
            print(f"Prompt: {analysis['original_prompt']}")
            print(f"Risk Level: {analysis['risk_level']}")
            if analysis['risk_factors']:
                print(f"Risk Factors:")
                for factor in analysis['risk_factors']:
                    print(f"  • {factor}")
            else:
                print("No significant risk factors detected")
        else:
            result = optimizer.optimize_prompt(args.prompt, args.aggressive)
            analysis = result['analysis']
            
            print(f"🔧 PROMPT OPTIMIZATION")
            print("-" * 30)
            print(f"Original: {analysis['original_prompt']}")
            print(f"Risk Level: {analysis['risk_level']}")
            
            if analysis['risk_factors']:
                print(f"Risk Factors:")
                for factor in analysis['risk_factors']:
                    print(f"  • {factor}")
            
            print(f"\nOptimized: {result['optimized_prompt']}")
            
            if result['applied_strategies']:
                print(f"Strategies Applied:")
                for strategy in result['applied_strategies']:
                    print(f"  • {strategy}")
            
            if result['improvement_expected']:
                print(f"\n✅ Expected improvement: Reduced zero fidelity risk")
            else:
                print(f"\nℹ️  Low risk prompt - minimal optimization applied")
    
    else:
        parser.print_help()

if __name__ == "__main__":
    main()