#!/usr/bin/env python3

"""
Architecturally-Aware TRELLIS Prompt Optimizer
Optimizes prompts by injecting hints that align with high-quality
generation pipeline configurations (e.g., high resolution, more steps).
"""

import re
from typing import Dict, List, Tuple

class ArchitecturalPromptOptimizer:
    """
    Enhances prompts to maximize fidelity scores by aligning them with
    the capabilities of the underlying generation architecture (TRELLIS/Flux).
    """
    
    def __init__(self, aggressive_mode: bool = False):
        self.aggressive = aggressive_mode
        
        # Architectural hints and quality specifiers
        # These are designed to signal the need for high-quality generation settings.
        self.architectural_hints = {
            'resolution': {
                'keywords': ['4k', '8k', 'high resolution', 'ultra high-resolution', 'photorealistic'],
                'prefix': 'res:1024'  # Hint for 1024x1024 T2I rendering
            },
            'detail_level': {
                'keywords': ['ultra-detailed', 'hyperrealistic', 'intricate details', 'fine-grained'],
                'prefix': 'steps:50+' # Hint for 50+ refinement steps
            },
            'guidance_scale': {
                'keywords': ['precision-crafted', 'geometrically perfect', 'exact specifications'],
                'prefix': 'cfg:7.5' # Hint for strong prompt adherence
            },
            'lighting_quality': {
                'keywords': ['professional studio lighting', 'cinematic lighting', 'dramatic lighting'],
                'prefix': 'light:pro'
            },
            'rendering_style': {
                'keywords': ['product render', 'architectural visualization', 'unreal engine 5'],
                'prefix': 'render:pro'
            }
        }
        
        # High-potential object categories identified from previous analysis
        self.high_potential_categories = [
            'tools', 'robots', 'wands', 'sports_equipment', 'instruments', 'furniture'
        ]
        
        self.category_keywords = {
            'tools': ['hammer', 'drill', 'wrench', 'screwdriver'],
            'robots': ['robot', 'mech', 'android'],
            'wands': ['wand', 'staff', 'scepter'],
            'sports_equipment': ['bat', 'stick', 'racket'],
            'instruments': ['violin', 'guitar', 'piano'],
            'furniture': ['chair', 'table', 'desk']
        }

    def _identify_category(self, prompt: str) -> str:
        """Identifies the category of the object in the prompt."""
        prompt_lower = prompt.lower()
        for category, keywords in self.category_keywords.items():
            if any(word in prompt_lower for word in keywords):
                return category
        return 'general'
    
    def optimize_prompt(self, prompt: str) -> Tuple[str, List[str]]:
        """
        Applies architectural enhancements to a prompt.

        Returns:
            A tuple containing (optimized_prompt, applied_strategies).
        """
        category = self._identify_category(prompt)
        is_high_potential = category in self.high_potential_categories
        
        optimized_parts = []
        applied_strategies = []
        
        # Start with the original prompt
        optimized_parts.append(prompt)
        
        # 1. Add Resolution Hints (most impactful for detail)
        # For high-potential objects, we can be more aggressive.
        if is_high_potential or self.aggressive:
            if not any(kw in prompt.lower() for kw in self.architectural_hints['resolution']['keywords']):
                optimized_parts.append('high resolution')
                applied_strategies.append("Targeted High Resolution")
        
        # 2. Add Detail Level Hints
        if is_high_potential or self.aggressive:
            if not any(kw in prompt.lower() for kw in self.architectural_hints['detail_level']['keywords']):
                optimized_parts.append('ultra-detailed')
                applied_strategies.append("Targeted Detail Level")

        # 3. Add Guidance/Precision Hints
        # Especially useful for tools, robots, and furniture.
        if category in ['tools', 'robots', 'furniture']:
            if not any(kw in prompt.lower() for kw in self.architectural_hints['guidance_scale']['keywords']):
                optimized_parts.append('precision-crafted')
                applied_strategies.append("Targeted Precision")
        
        # 4. Add Professional Lighting
        if is_high_potential or self.aggressive:
            if not any(kw in prompt.lower() for kw in self.architectural_hints['lighting_quality']['keywords']):
                optimized_parts.append('professional studio lighting')
                applied_strategies.append("Targeted Pro Lighting")
                
        # 5. Add Professional Rendering Style
        if is_high_potential:
            if not any(kw in prompt.lower() for kw in self.architectural_hints['rendering_style']['keywords']):
                optimized_parts.append('professional product render')
                applied_strategies.append("Targeted Pro Rendering Style")

        # 6. Add "wbgmsst" as a fallback for solid structures, unless transparency is requested
        if 'glass' not in prompt.lower() and 'crystal' not in prompt.lower():
            if 'wbgmsst' not in prompt.lower():
                optimized_parts.insert(0, 'wbgmsst') # Add to the front
                applied_strategies.append("Ensured Solid Structure (wbgmsst)")

        # Join the parts into the final optimized prompt
        optimized_prompt = ', '.join(part.strip() for part in optimized_parts if part)
        
        return optimized_prompt, applied_strategies

def main():
    """Demo the architectural prompt optimizer."""
    optimizer_standard = ArchitecturalPromptOptimizer(aggressive_mode=False)
    optimizer_aggressive = ArchitecturalPromptOptimizer(aggressive_mode=True)
    
    test_prompts = [
        # High potential
        "a metal hammer with a claw head",
        "a purple robot with intricate gears",
        "a polished wooden violin",
        
        # Lower potential / general
        "a simple red cube",
        "a crystal dragon",
        "a futuristic sword"
    ]
    
    print("=== ARCHITECTURALLY-AWARE PROMPT OPTIMIZER DEMO ===")
    
    for prompt in test_prompts:
        print(f"\n" + "="*50)
        print(f"Original Prompt: '{prompt}'")
        print("-" * 50)
        
        # Standard Mode
        optimized_std, strategies_std = optimizer_standard.optimize_prompt(prompt)
        print(f"STANDARD Optimized:")
        print(f"  -> '{optimized_std}'")
        print(f"  Strategies: {strategies_std}")

        # Aggressive Mode
        optimized_agg, strategies_agg = optimizer_aggressive.optimize_prompt(prompt)
        print(f"\nAGGRESSIVE Optimized:")
        print(f"  -> '{optimized_agg}'")
        print(f"  Strategies: {strategies_agg}")

if __name__ == "__main__":
    main() 