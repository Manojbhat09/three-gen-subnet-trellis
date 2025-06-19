#!/usr/bin/env python3
"""
TRELLIS Prompt Optimizer V2 - Advanced Pattern Detection and Optimization
Addresses specific failure patterns causing 0.0 fidelity scores
"""

import re
import json
import argparse
import random
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

@dataclass
class OptimizationResult:
    original_prompt: str
    optimized_prompt: str
    risk_level: str
    risk_score: float
    detected_issues: List[str]
    applied_strategies: List[str]
    parameter_adjustments: Dict[str, any]
    confidence: float

class TrellisPromptOptimizerV2:
    def __init__(self):
        """Initialize with comprehensive failure patterns from production data"""
        
        # Critical failure patterns (almost always get 0.0 fidelity)
        self.critical_patterns = {
            'transparent_liquids': {
                'keywords': ['juice', 'water', 'liquid', 'drink', 'beverage', 'wine', 'beer'],
                'modifiers': ['in glass', 'in cup', 'in bottle', 'served in', 'poured'],
                'risk_weight': 10.0
            },
            'glass_containers': {
                'keywords': ['glass', 'crystal', 'transparent', 'clear'],
                'modifiers': ['vase', 'bottle', 'container', 'bowl', 'cup', 'jar'],
                'risk_weight': 8.0
            },
            'bottles_containers': {
                'keywords': ['bottle', 'jar', 'flask', 'container', 'vessel'],
                'modifiers': ['plastic', 'glass', 'empty', 'transparent', 'with lid'],
                'risk_weight': 5.0
            },
            'multi_object_scenes': {
                'keywords': ['holding', 'with', 'beside', 'on top of', 'containing', 'and'],
                'object_count': 2,  # Scenes with 2+ distinct objects
                'risk_weight': 7.0
            },
            'abstract_descriptors': {
                'keywords': ['contemporary', 'modern', 'abstract', 'asymmetrical', 'artistic'],
                'risk_weight': 6.0
            },
            'tiny_objects': {
                'keywords': ['tiny', 'miniature', 'small', 'little', 'micro'],
                'risk_weight': 5.0
            },
            'generic_objects': {
                'keywords': ['object', 'thing', 'item', 'something', 'stuff'],
                'modifiers': ['small', 'simple', 'basic', 'generic'],
                'risk_weight': 3.0
            },
            'complex_footwear': {
                'keywords': ['shoe', 'boot', 'sneaker', 'sandal', 'heel', 'slipper'],
                'modifiers': ['spike', 'cleats', 'laces', 'buckles', 'straps', 'sole', 'multi-layer'],
                'risk_weight': 6.0
            },
            'complex_geometry': {
                'keywords': ['spike', 'thorn', 'needle', 'bristle', 'prong', 'teeth', 'serrated'],
                'modifiers': ['multiple', 'array', 'pattern', 'repeating', 'small'],
                'risk_weight': 7.0
            },
            'architectural_scenes': {
                'keywords': ['interior', 'building', 'warehouse', 'house', 'room', 'hall', 'structure'],
                'modifiers': ['exposed', 'complex', 'detailed', 'framework', 'scaffolding', 'beams'],
                'risk_weight': 6.5
            },
            'complex_structures': {
                'keywords': ['framework', 'scaffolding', 'lattice', 'grid', 'mesh', 'network'],
                'modifiers': ['steel', 'metal', 'overhead', 'suspended', 'intricate'],
                'risk_weight': 7.5
            }
        }
        
        # Material-specific patterns
        self.material_patterns = {
            'precious_stones': ['diamond', 'sapphire', 'ruby', 'emerald', 'topaz', 'opal', 'crystal'],
            'transparent_materials': ['glass', 'crystal', 'acrylic', 'transparent', 'clear', 'see-through'],
            'reflective_materials': ['mirror', 'chrome', 'polished', 'glossy', 'shiny', 'metallic'],
            'organic_materials': ['plant', 'flower', 'leaf', 'wood', 'bark', 'root']
        }
        
        # Object category patterns
        self.object_categories = {
            'weapons': ['sword', 'katana', 'knife', 'dagger', 'spear', 'axe', 'bow', 'crossbow', 'gun', 'pistol', 'revolver'],
            'tools': ['hammer', 'screwdriver', 'wrench', 'drill', 'pliers', 'saw', 'chisel'],
            'furniture': ['table', 'chair', 'desk', 'cabinet', 'shelf', 'door', 'lamp'],
            'jewelry': ['ring', 'necklace', 'bracelet', 'brooch', 'pendant', 'earring'],
            'containers': ['vase', 'bottle', 'jar', 'box', 'container', 'pot', 'bowl']
        }
        
        # Successful prompt patterns (from high fidelity scores)
        self.success_patterns = {
            'simple_objects': ['robot', 'creature', 'monster', 'character'],
            'solid_materials': ['metal', 'stone', 'concrete', 'plastic', 'ceramic'],
            'clear_descriptions': ['red', 'blue', 'green', 'yellow', 'purple', 'orange'],
            'single_focus': ['single', 'one', 'isolated', 'standalone']
        }
        
        # Optimization strategies
        self.optimization_strategies = {
            'material_substitution': {
                'description': 'Replace problematic materials with solid alternatives',
                'priority': 1
            },
            'scene_simplification': {
                'description': 'Convert multi-object scenes to single objects',
                'priority': 2
            },
            'concrete_description': {
                'description': 'Replace abstract terms with concrete descriptors',
                'priority': 3
            },
            'size_normalization': {
                'description': 'Remove tiny/miniature modifiers',
                'priority': 4
            },
            'detail_enhancement': {
                'description': 'Add specific 3D modeling cues',
                'priority': 5
            }
        }
        
        # Parameter adjustment recommendations
        self.parameter_adjustments = {
            'high_risk': {
                'guidance_scale': 4.5,
                'ss_guidance_strength': 9.5,
                'ss_sampling_steps': 16,
                'slat_guidance_strength': 4.0,
                'slat_sampling_steps': 16
            },
            'medium_risk': {
                'guidance_scale': 4.0,
                'ss_guidance_strength': 9.0,
                'ss_sampling_steps': 14,
                'slat_guidance_strength': 3.5,
                'slat_sampling_steps': 14
            },
            'low_risk': {
                # Keep defaults
            }
        }
        
        # CLIP alignment boosters - style cues that improve alignment scores
        self.clip_boosters = {
            'rendering_context': [
                '3D render',
                'CGI model', 
                'game asset',
                'digital sculpture',
                'volumetric render'
            ],
            'lighting': [
                'studio lighting',
                'dramatic lighting',
                'soft lighting',
                'rim lighting',
                'professional lighting'
            ],
            'camera': [
                'centered view',
                'product shot',
                'hero angle',
                'isometric view',
                'turntable view'
            ],
            'quality': [
                'high detail',
                'photorealistic',
                'professional quality',
                '8K resolution',
                'ultra detailed'
            ],
            'background': [
                'white background',
                'studio background',
                'gradient background',
                'clean background',
                'neutral background'
            ]
        }
        
        # CLIP optimization settings (optional)
        self.use_clip_optimization = False
        self.clip_optimizer = None
    
    def enable_clip_optimization(self, model_name: str = "ViT-B-32", pretrained: str = "openai"):
        """Enable CLIP-based prompt optimization for maximum alignment scores"""
        try:
            from clip_prompt_optimizer import CLIPPromptOptimizer
            self.clip_optimizer = CLIPPromptOptimizer(model_name, pretrained)
            self.use_clip_optimization = True
            print("✅ CLIP optimization enabled")
        except Exception as e:
            print(f"⚠️ Could not enable CLIP optimization: {e}")
            self.use_clip_optimization = False
    
    def load_clip_model(self):
        """Load CLIP model if optimization is enabled"""
        if self.use_clip_optimization and self.clip_optimizer:
            self.clip_optimizer.load_model()
    
    def unload_clip_model(self):
        """Unload CLIP model to free GPU memory"""
        if self.use_clip_optimization and self.clip_optimizer:
            self.clip_optimizer.unload_model()
    
    def detect_multi_object_scene(self, prompt: str) -> Tuple[bool, List[str]]:
        """Detect if prompt describes multiple distinct objects"""
        # Common object separators
        separators = [' with ', ' and ', ' beside ', ' next to ', ' holding ', ' on ', ' in ']
        
        # Split prompt by separators and analyze
        objects = []
        remaining = prompt.lower()
        
        for sep in separators:
            if sep in remaining:
                parts = remaining.split(sep, 1)
                if len(parts[0].split()) <= 4:  # Likely an object description
                    objects.append(parts[0].strip())
                remaining = parts[1] if len(parts) > 1 else ""
        
        # Add the last part
        if remaining and len(remaining.split()) <= 4:
            objects.append(remaining.strip())
        
        # Check for object count indicators
        has_multiple = len(objects) >= 2
        
        # Also check for explicit multi-object keywords
        multi_keywords = ['holding', 'containing', 'with', 'and', 'beside', 'on top of']
        for keyword in multi_keywords:
            if keyword in prompt.lower():
                has_multiple = True
                break
        
        return has_multiple, objects
    
    def analyze_prompt_risk(self, prompt: str) -> Dict:
        """Comprehensive risk analysis of prompt"""
        prompt_lower = prompt.lower()
        
        risk_score = 0.0
        detected_issues = []
        
        # Check critical patterns
        for pattern_name, pattern_data in self.critical_patterns.items():
            if pattern_name == 'multi_object_scenes':
                is_multi, objects = self.detect_multi_object_scene(prompt)
                if is_multi:
                    risk_score += pattern_data['risk_weight']
                    detected_issues.append(f"Multi-object scene detected: {', '.join(objects)}")
            else:
                for keyword in pattern_data.get('keywords', []):
                    if keyword in prompt_lower:
                        risk_score += pattern_data['risk_weight']
                        detected_issues.append(f"{pattern_name}: '{keyword}'")
                        
                        # Check for modifiers that increase risk
                        if 'modifiers' in pattern_data:
                            for modifier in pattern_data['modifiers']:
                                if modifier in prompt_lower:
                                    risk_score += 2.0  # Extra penalty for modifier combinations
                                    detected_issues.append(f"  + modifier: '{modifier}'")
                        break
        
        # Check material patterns
        for material_type, keywords in self.material_patterns.items():
            for keyword in keywords:
                if keyword in prompt_lower:
                    if material_type in ['transparent_materials', 'precious_stones']:
                        risk_score += 10.0
                        detected_issues.append(f"Problematic material: {keyword}")
                    elif material_type == 'reflective_materials':
                        risk_score += 3.0
                        detected_issues.append(f"Reflective material: {keyword}")
        
        # Determine risk level
        if risk_score >= 15:
            risk_level = 'CRITICAL'
        elif risk_score >= 10:
            risk_level = 'HIGH'
        elif risk_score >= 5:
            risk_level = 'MEDIUM'
        else:
            risk_level = 'LOW'
        
        return {
            'risk_score': risk_score,
            'risk_level': risk_level,
            'detected_issues': detected_issues
        }
    
    def apply_material_substitution(self, prompt: str) -> str:
        """Apply material substitutions for problematic materials"""
        # Existing substitutions
        substitutions = {
            # Liquids and transparent materials
            'water': 'clear blue solid material',
            'juice': 'colored solid material',
            'wine': 'dark red material',
            'liquid': 'solid material',
            'glass': 'ceramic',
            'crystal': 'polished stone',
            'transparent': 'opaque',
            'translucent': 'solid colored',
            'clear': 'light colored',
            
            # Precious materials that fail
            'opal': 'iridescent stone',
            'pearl': 'white sphere',
            'diamond': 'faceted stone',
            'emerald': 'green colored stone',
            'ruby': 'red colored stone',
            'sapphire': 'blue colored stone',
            'gemstone': 'colored stone',
            'jewel': 'decorative stone',
            
            # Complex footwear simplifications
            'spike shoes': 'athletic shoes',
            'track spikes': 'running shoes',
            'cleated': 'textured sole',
            'spiked': 'studded',
            'with spikes': 'with texture',
            'spike': 'textured'
        }
        
        modified = prompt
        for old, new in substitutions.items():
            if old in prompt.lower():
                # Case-insensitive replacement
                pattern = re.compile(re.escape(old), re.IGNORECASE)
                modified = pattern.sub(new, modified)
        
        return modified
    
    def simplify_scene(self, prompt: str) -> str:
        """Convert multi-object scenes to single object focus"""
        # Identify the primary object (usually the first noun phrase)
        
        # Remove secondary objects and relationships
        simplifications = [
            (r'\s+holding\s+.*', ''),
            (r'\s+with\s+.*\s+on\s+it', ''),
            (r'\s+beside\s+.*', ''),
            (r'\s+on\s+top\s+of\s+.*', ''),
            (r'\s+containing\s+.*', ''),
            (r'\s+and\s+.*', ''),
            (r'\s+in\s+.*\s+pot', ''),
            (r'\s+in\s+.*\s+glass', ''),
            (r'\s+with\s+.*\s+handles?', ''),
        ]
        
        modified = prompt
        for pattern, replacement in simplifications:
            modified = re.sub(pattern, replacement, modified, flags=re.IGNORECASE)
        
        # Add single object emphasis
        if modified != prompt:
            modified = f"single {modified.strip()}, isolated object"
        
        return modified
    
    def make_concrete(self, prompt: str) -> str:
        """Replace abstract terms with concrete descriptions"""
        concrete_replacements = {
            'contemporary': 'modern geometric',
            'abstract': 'geometric shaped',
            'asymmetrical': 'angular',
            'artistic': 'sculptural',
            'elegant': 'streamlined',
            'vintage': 'classic styled',
            'antique': 'traditional',
            'tiny': 'small scale',
            'miniature': 'scaled down',
            'massive': 'large scale'
        }
        
        modified = prompt
        for abstract, concrete in concrete_replacements.items():
            if abstract in prompt.lower():
                pattern = re.compile(re.escape(abstract), re.IGNORECASE)
                modified = pattern.sub(concrete, modified)
        
        return modified
    
    def add_3d_modeling_cues(self, prompt: str) -> str:
        """Add specific cues that help with 3D generation"""
        # Determine object category
        category = None
        for cat, keywords in self.object_categories.items():
            for keyword in keywords:
                if keyword in prompt.lower():
                    category = cat
                    break
            if category:
                break
        
        # Add category-specific cues
        cues = []
        
        if category == 'weapons':
            cues = ['detailed metalwork', 'game asset style', 'centered view']
        elif category == 'tools':
            cues = ['industrial design', 'functional form', 'product visualization']
        elif category == 'furniture':
            cues = ['architectural model', 'clean geometry', 'isometric view']
        elif category == 'jewelry':
            cues = ['product photography style', 'detailed craftsmanship', 'centered display']
        elif category == 'containers':
            cues = ['solid form', 'ceramic or metal material', 'single object']
        else:
            cues = ['3D model', 'game asset', 'centered composition']
        
        return f"{prompt}, {', '.join(cues)}"
    
    def add_clip_alignment_boosters(self, prompt: str, risk_level: str) -> str:
        """Add CLIP-friendly descriptors to maximize alignment score"""
        
        # Select appropriate boosters based on risk level
        boosters = []
        
        # Always add rendering context
        boosters.append(random.choice(self.clip_boosters['rendering_context']))
        
        # Add background descriptor if not already present
        if not any(bg in prompt.lower() for bg in ['background', 'backdrop']):
            boosters.append(random.choice(self.clip_boosters['background']))
        
        # Add quality descriptor for better CLIP recognition
        if risk_level in ['HIGH', 'CRITICAL']:
            boosters.append(random.choice(self.clip_boosters['quality']))
        
        # Add camera/view descriptor if not present
        if not any(view in prompt.lower() for view in ['view', 'angle', 'shot']):
            boosters.append(random.choice(self.clip_boosters['camera']))
        
        # Add lighting for complex objects
        if risk_level in ['MEDIUM', 'HIGH', 'CRITICAL']:
            boosters.append(random.choice(self.clip_boosters['lighting']))
        
        # Combine with original prompt
        return f"{prompt}, {', '.join(boosters)}"
    
    def enhance_object_description(self, prompt: str) -> str:
        """Add specific visual details to generic descriptions"""
        
        # Generic object enhancements
        enhancements = {
            'bottle': 'cylindrical bottle with label',
            'jar': 'rounded jar with lid',
            'box': 'rectangular box with edges',
            'chair': 'four-legged chair with backrest',
            'table': 'flat-top table with legs',
            'cup': 'handled cup with rim',
            'vase': 'decorative vase with narrow neck',
            'bowl': 'rounded bowl with wide opening'
        }
        
        enhanced = prompt
        for generic, specific in enhancements.items():
            if generic in prompt.lower() and specific not in prompt.lower():
                # Replace generic with specific
                pattern = re.compile(r'\b' + generic + r'\b', re.IGNORECASE)
                enhanced = pattern.sub(specific, enhanced)
        
        return enhanced
    
    def optimize_prompt(self, prompt: str, aggressive: bool = False) -> Dict:
        """Simplified optimization for compatibility with existing code"""
        result = self._optimize_prompt_v2(prompt, aggressive)
        
        # Convert to dict format expected by existing code
        return {
            'analysis': {
                'original_prompt': result.original_prompt,
                'risk_level': result.risk_level,
                'risk_factors': result.detected_issues
            },
            'optimized_prompt': result.optimized_prompt,
            'applied_strategies': result.applied_strategies,
            'optimization_keywords': [],  # For compatibility
            'improvement_expected': result.risk_level != 'LOW'
        }
    
    def apply_scene_simplification(self, prompt: str) -> str:
        """Simplify multi-object scenes to single focus"""
        return self.simplify_scene(prompt)
    
    def simplify_complex_geometry(self, prompt: str) -> str:
        """Simplify objects with complex repeating geometry"""
        prompt_lower = prompt.lower()
        
        # Simplify footwear descriptions
        footwear_simplifications = {
            'track spike shoes': 'athletic running shoes',
            'soccer cleats': 'athletic shoes with textured sole',
            'high heel shoes': 'dress shoes',
            'combat boots': 'military style boots',
            'hiking boots': 'outdoor boots'
        }
        
        # Simplify architectural descriptions
        architectural_simplifications = {
            'exposed interior': 'simple interior view',
            'overhead steel framework': 'ceiling structure',
            'warehouse with overhead': 'industrial building',
            'complex framework': 'simple structure',
            'scaffolding structure': 'support frame',
            'intricate lattice': 'grid pattern'
        }
        
        for complex_term, simple_term in footwear_simplifications.items():
            if complex_term in prompt_lower:
                prompt = prompt.lower().replace(complex_term, simple_term)
                # Preserve original casing for first letter
                prompt = prompt[0].upper() + prompt[1:] if prompt else prompt
        
        for complex_term, simple_term in architectural_simplifications.items():
            if complex_term in prompt_lower:
                prompt = prompt.lower().replace(complex_term, simple_term)
                # Preserve original casing for first letter
                prompt = prompt[0].upper() + prompt[1:] if prompt else prompt
                
        return prompt
    
    def apply_complexity_reduction(self, prompt: str) -> str:
        """Reduce complexity by making abstract terms concrete and simplifying geometry"""
        # First simplify complex geometry
        prompt = self.simplify_complex_geometry(prompt)
        
        # Then make abstract terms concrete
        concrete = self.make_concrete(prompt)
        enhanced = self.enhance_object_description(concrete)
        return enhanced
    
    def add_alignment_boosters(self, prompt: str) -> str:
        """Add CLIP alignment boosters"""
        # Determine risk level for appropriate boosters
        analysis = self.analyze_prompt_risk(prompt)
        return self.add_clip_alignment_boosters(prompt, analysis['risk_level'])
    
    def _optimize_prompt_v2(self, prompt: str, aggressive: bool = False) -> OptimizationResult:
        """Internal optimization with full functionality"""
        
        # Use comprehensive analysis
        analysis = self.analyze_prompt_risk(prompt)
        risk_level = analysis['risk_level']
        risk_score = analysis['risk_score']
        issues = analysis['detected_issues']
        
        # Apply optimizations
        optimized = prompt
        strategies = []
        
        if risk_level in ['CRITICAL', 'HIGH', 'MEDIUM'] or aggressive:
            # Apply material substitution
            original_optimized = optimized
            optimized = self.apply_material_substitution(optimized)
            if optimized != original_optimized:
                strategies.append('material_substitution')
            
            # Apply scene simplification
            original_optimized = optimized
            optimized = self.apply_scene_simplification(optimized)
            if optimized != original_optimized:
                strategies.append('scene_simplification')
            
            # Apply complexity reduction
            original_optimized = optimized
            optimized = self.apply_complexity_reduction(optimized)
            if optimized != original_optimized:
                strategies.append('complexity_reduction')
        
        # Always add alignment boosters
        original_optimized = optimized
        optimized = self.add_alignment_boosters(optimized)
        if optimized != original_optimized:
            strategies.append('alignment_boosters')
        
        # Apply CLIP optimization if enabled
        if self.use_clip_optimization and self.clip_optimizer:
            try:
                # Ensure CLIP model is loaded
                self.clip_optimizer.load_model()
                
                clip_result = self.clip_optimizer.optimize_prompt(optimized, num_iterations=20)
                if clip_result['improvement_percent'] > 5:  # Only use if significant improvement
                    optimized = clip_result['optimized_prompt']
                    strategies.append(f'clip_optimization (+{clip_result["improvement_percent"]:.1f}%)')
                    
                    # Add CLIP scores to analysis
                    analysis['clip_original_score'] = clip_result['original_score']
                    analysis['clip_optimized_score'] = clip_result['optimized_score']
                    analysis['clip_improvement'] = clip_result['improvement_percent']
            except Exception as e:
                print(f"⚠️ CLIP optimization failed: {e}")
        
        # Determine parameter adjustments
        param_adj = {}
        if risk_level == 'CRITICAL':
            param_adj = self.parameter_adjustments['high_risk'].copy()
            param_adj['guidance_scale'] = 5.0  # Even higher for critical
            param_adj['ss_guidance_strength'] = 10.0
        elif risk_level == 'HIGH':
            param_adj = self.parameter_adjustments['high_risk'].copy()
        elif risk_level == 'MEDIUM':
            param_adj = self.parameter_adjustments['medium_risk'].copy()
        else:
            param_adj = self.parameter_adjustments.get('low_risk', {})
        
        confidence = 0.9 if strategies and risk_level in ['CRITICAL', 'HIGH'] else 0.7 if strategies else 0.3
        
        return OptimizationResult(
            original_prompt=prompt,
            optimized_prompt=optimized,
            risk_level=risk_level,
            risk_score=risk_score,
            detected_issues=issues,
            applied_strategies=strategies,
            parameter_adjustments=param_adj,
            confidence=confidence
        )

def main():
    """Test the optimizer"""
    optimizer = TrellisPromptOptimizerV2()
    
    test_prompts = [
        "orange juice served in tall clear glass",
        "contemporary blue candle holder asymmetrical form",
        "statue of saint with book",
        "tiny yellow daisy plant in small pot",
        "topaz stone in rectangular prism shape",
        "simple wooden chair"
    ]
    
    print("Testing Prompt Optimizer V2")
    print("=" * 80)
    
    for prompt in test_prompts:
        result = optimizer.optimize_prompt(prompt)
        print(f"\nOriginal: {prompt}")
        print(f"Risk: {result['analysis']['risk_level']}")
        print(f"Optimized: {result['optimized_prompt']}")
        if result['applied_strategies']:
            print(f"Strategies: {', '.join(result['applied_strategies'])}")

if __name__ == "__main__":
    main() 