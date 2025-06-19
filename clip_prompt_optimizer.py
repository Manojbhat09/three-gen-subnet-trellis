#!/usr/bin/env python3
"""
CLIP-Based Prompt Optimizer
Pre-tests prompts with CLIP to maximize alignment scores before generation
"""

import torch
import open_clip
from typing import List, Dict, Tuple
import numpy as np
from itertools import combinations, product
import random

class CLIPPromptOptimizer:
    def __init__(self, model_name: str = "ViT-B-32", pretrained: str = "openai"):
        """Initialize CLIP optimizer without loading the model"""
        self.model_name = model_name
        self.pretrained = pretrained
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Model components (will be loaded on demand)
        self.model = None
        self.preprocess = None
        self.tokenizer = None
        self._model_loaded = False
        
        print(f"🔧 CLIP optimizer initialized (model will be loaded on demand)")
        
        # Optimization components
        self.style_variations = {
            'rendering': [
                '3D render',
                'CGI model',
                'game asset',
                'digital art',
                '3D model',
                'volumetric render',
                'ray traced render',
                'octane render',
                'unreal engine'
            ],
            'quality': [
                'high quality',
                'ultra detailed',
                'photorealistic',
                'highly detailed',
                '8K resolution',
                'professional',
                'masterpiece',
                'best quality',
                'award winning'
            ],
            'lighting': [
                'studio lighting',
                'dramatic lighting',
                'soft lighting',
                'rim lighting',
                'volumetric lighting',
                'global illumination',
                'ambient occlusion',
                'subsurface scattering'
            ],
            'camera': [
                'centered composition',
                'product photography',
                'hero shot',
                'isometric view',
                'three quarter view',
                'front view',
                'beauty shot',
                'turntable render'
            ],
            'background': [
                'white background',
                'studio background',
                'gradient background',
                'HDRI lighting',
                'neutral background',
                'clean background',
                'professional backdrop'
            ],
            'style': [
                'trending on artstation',
                'concept art',
                'production quality',
                'portfolio piece',
                'showcase render',
                'advertisement style',
                'product visualization'
            ]
        }
        
        # Object enhancement templates
        self.object_enhancers = {
            'material': ['metallic', 'matte', 'glossy', 'textured', 'polished', 'brushed'],
            'color': ['vibrant', 'saturated', 'rich', 'deep', 'bright'],
            'detail': ['intricate details', 'fine details', 'sharp edges', 'smooth surfaces'],
            'scale': ['detailed', 'high resolution', 'crisp', 'sharp']
        }
    
    def load_model(self):
        """Load CLIP model into memory"""
        if self._model_loaded:
            return
        
        print(f"📥 Loading CLIP model ({self.model_name}) on {self.device}...")
        
        # Load CLIP model
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            self.model_name, pretrained=self.pretrained, device=self.device
        )
        self.tokenizer = open_clip.get_tokenizer(self.model_name)
        self.model.eval()
        
        self._model_loaded = True
        print("✅ CLIP model loaded")
    
    def unload_model(self):
        """Unload CLIP model from memory and free GPU resources"""
        if not self._model_loaded:
            return
        
        print("📤 Unloading CLIP model...")
        
        # Delete model references
        if self.model is not None:
            del self.model
            self.model = None
        
        if self.preprocess is not None:
            del self.preprocess
            self.preprocess = None
        
        if self.tokenizer is not None:
            del self.tokenizer
            self.tokenizer = None
        
        # Clear GPU cache
        if self.device.type == 'cuda':
            torch.cuda.empty_cache()
        
        self._model_loaded = False
        print("✅ CLIP model unloaded, GPU memory freed")
    
    def _ensure_model_loaded(self):
        """Ensure model is loaded before using it"""
        if not self._model_loaded:
            self.load_model()
    
    def score_prompt(self, prompt: str) -> float:
        """Score a single prompt using CLIP text encoder"""
        self._ensure_model_loaded()
        
        with torch.no_grad():
            text_tokens = self.tokenizer([prompt]).to(self.device)
            text_features = self.model.encode_text(text_tokens)
            text_features /= text_features.norm(dim=-1, keepdim=True)
            
            # Self-similarity as a proxy for CLIP confidence
            # Higher magnitude features often correlate with better CLIP understanding
            score = float(text_features.norm().cpu())
            
        return score
    
    def score_prompt_similarity(self, prompt: str, reference_prompts: List[str]) -> float:
        """Score prompt based on similarity to high-quality reference prompts"""
        self._ensure_model_loaded()
        
        with torch.no_grad():
            # Encode prompt
            prompt_tokens = self.tokenizer([prompt]).to(self.device)
            prompt_features = self.model.encode_text(prompt_tokens)
            prompt_features /= prompt_features.norm(dim=-1, keepdim=True)
            
            # Encode references
            ref_tokens = self.tokenizer(reference_prompts).to(self.device)
            ref_features = self.model.encode_text(ref_tokens)
            ref_features /= ref_features.norm(dim=-1, keepdim=True)
            
            # Calculate similarities
            similarities = (prompt_features @ ref_features.T).cpu().numpy()
            
            # Return average similarity
            return float(np.mean(similarities))
    
    def generate_variations(self, base_prompt: str, num_variations: int = 20) -> List[str]:
        """Generate multiple variations of a prompt with different style modifiers"""
        variations = [base_prompt]  # Include original
        
        # Method 1: Add single category modifiers
        for category, options in self.style_variations.items():
            for modifier in random.sample(options, min(3, len(options))):
                variations.append(f"{base_prompt}, {modifier}")
        
        # Method 2: Combine multiple modifiers
        for _ in range(num_variations // 2):
            modifiers = []
            # Pick one from each category randomly
            for category, options in self.style_variations.items():
                if random.random() > 0.5:  # 50% chance to include each category
                    modifiers.append(random.choice(options))
            
            if modifiers:
                variations.append(f"{base_prompt}, {', '.join(modifiers)}")
        
        # Method 3: Rephrase with common CLIP-friendly patterns
        templates = [
            "a {quality} {render} of {prompt}",
            "{prompt}, {quality}, {render}, {background}",
            "{render} of {prompt}, {lighting}, {camera}",
            "{prompt} rendered in {quality} with {lighting}",
            "professional {render} of {prompt}, {background}, {quality}"
        ]
        
        for template in random.sample(templates, min(5, len(templates))):
            filled = template.format(
                prompt=base_prompt,
                quality=random.choice(self.style_variations['quality']),
                render=random.choice(self.style_variations['rendering']),
                lighting=random.choice(self.style_variations['lighting']),
                camera=random.choice(self.style_variations['camera']),
                background=random.choice(self.style_variations['background'])
            )
            variations.append(filled)
        
        # Remove duplicates while preserving order
        seen = set()
        unique_variations = []
        for v in variations:
            if v not in seen:
                seen.add(v)
                unique_variations.append(v)
        
        return unique_variations[:num_variations]
    
    def optimize_prompt(self, base_prompt: str, num_iterations: int = 50) -> Dict:
        """Find the best prompt variation using CLIP scoring"""
        print(f"\n🔍 Optimizing prompt: '{base_prompt}'")
        
        # Ensure model is loaded
        self._ensure_model_loaded()
        
        # Generate high-quality reference prompts (these typically score well)
        references = [
            "high quality 3D render on white background",
            "professional CGI model with studio lighting",
            "photorealistic game asset centered composition",
            "ultra detailed 3D model product photography",
            "masterpiece quality volumetric render"
        ]
        
        # Generate variations
        variations = self.generate_variations(base_prompt, num_iterations)
        
        # Score all variations
        results = []
        for i, variant in enumerate(variations):
            # Get self-score
            self_score = self.score_prompt(variant)
            
            # Get similarity to references
            ref_score = self.score_prompt_similarity(variant, references)
            
            # Combined score (weighted average)
            combined_score = 0.3 * self_score + 0.7 * ref_score
            
            results.append({
                'prompt': variant,
                'self_score': self_score,
                'ref_score': ref_score,
                'combined_score': combined_score,
                'added_tokens': len(variant.split()) - len(base_prompt.split())
            })
            
            if i % 10 == 0:
                print(f"   Tested {i+1}/{len(variations)} variations...")
        
        # Sort by combined score
        results.sort(key=lambda x: x['combined_score'], reverse=True)
        
        # Get best result
        best = results[0]
        baseline = next((r for r in results if r['prompt'] == base_prompt), results[-1])
        
        improvement = ((best['combined_score'] - baseline['combined_score']) / 
                      baseline['combined_score'] * 100)
        
        print(f"\n✅ Optimization complete!")
        print(f"   Original score: {baseline['combined_score']:.4f}")
        print(f"   Best score: {best['combined_score']:.4f} (+{improvement:.1f}%)")
        print(f"   Best prompt: '{best['prompt']}'")
        
        return {
            'original_prompt': base_prompt,
            'optimized_prompt': best['prompt'],
            'original_score': baseline['combined_score'],
            'optimized_score': best['combined_score'],
            'improvement_percent': improvement,
            'all_variations': results[:10]  # Top 10
        }
    
    def test_prompt_quality(self, prompt: str) -> Dict:
        """Test a prompt and provide detailed CLIP analysis"""
        self._ensure_model_loaded()
        
        with torch.no_grad():
            # Tokenize and encode
            tokens = self.tokenizer([prompt]).to(self.device)
            features = self.model.encode_text(tokens)
            features_norm = features / features.norm(dim=-1, keepdim=True)
            
            # Analyze different aspects
            feature_magnitude = float(features.norm().cpu())
            feature_variance = float(features.var().cpu())
            
            # Test against known good patterns
            good_patterns = [
                "3D render",
                "high quality",
                "white background",
                "centered",
                "detailed"
            ]
            
            pattern_scores = {}
            for pattern in good_patterns:
                test_prompt = f"{prompt}, {pattern}"
                test_tokens = self.tokenizer([test_prompt]).to(self.device)
                test_features = self.model.encode_text(test_tokens)
                test_features = test_features / test_features.norm(dim=-1, keepdim=True)
                
                similarity = float((features_norm @ test_features.T).cpu())
                pattern_scores[pattern] = similarity
            
        return {
            'prompt': prompt,
            'feature_magnitude': feature_magnitude,
            'feature_variance': feature_variance,
            'pattern_compatibility': pattern_scores,
            'overall_quality': feature_magnitude * 0.5 + np.mean(list(pattern_scores.values())) * 0.5
        }
    
    def __del__(self):
        """Cleanup when object is destroyed"""
        self.unload_model()

def main():
    """Test the CLIP optimizer"""
    optimizer = CLIPPromptOptimizer()
    
    # Test prompts
    test_prompts = [
        "short plastic bottle",
        "red ceramic vase",
        "metallic robot",
        "glass container with flowers"
    ]
    
    print("🚀 CLIP-Based Prompt Optimization Demo")
    print("=" * 60)
    
    for prompt in test_prompts:
        # Test original quality
        quality = optimizer.test_prompt_quality(prompt)
        print(f"\nOriginal: '{prompt}'")
        print(f"Quality Score: {quality['overall_quality']:.4f}")
        
        # Optimize
        result = optimizer.optimize_prompt(prompt, num_iterations=30)
        
        # Show top 3 variations
        print(f"\nTop 3 variations:")
        for i, var in enumerate(result['all_variations'][:3]):
            print(f"{i+1}. Score: {var['combined_score']:.4f} - '{var['prompt']}'")
        
        print("-" * 60)
    
    # Explicitly unload model at the end
    optimizer.unload_model()

if __name__ == "__main__":
    main() 