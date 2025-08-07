#!/usr/bin/env python3
"""
Simple CLIP Score Check
Purpose: Compare CLIP scores between two prompts using the same model as subnet accurate validator
Uses: convnext_large_d model with laion2b_s26b_b102k_augreg weights (production standard)
"""

import torch
import torch.nn.functional as F
import open_clip
from open_clip import CLIP
from open_clip.tokenizer import HFTokenizer
from torchvision import transforms
import numpy as np
from typing import Tuple, Optional
import gc


class SimpleCLIPScoreChecker:
    """Simple CLIP score checker using production validation model"""
    
    def __init__(self, verbose: bool = False):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.verbose = verbose
        
        # Production model settings (same as subnet accurate validator)
        self.model_name = "convnext_large_d"
        self.pretrained = "laion2b_s26b_b102k_augreg"
        
        # Model components
        self._model: Optional[CLIP] = None
        self._tokenizer: Optional[HFTokenizer] = None
        
        # Normalization transform (same as production)
        mean = torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(1, 3, 1, 1) * 3
        std = torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(1, 3, 1, 1) * 3
        self._normalize_transform = transforms.Normalize(mean, std)
        
        print(f"🔧 SimpleCLIPScoreChecker initialized on {self.device}")
        print(f"   Model: {self.model_name}")
        print(f"   Weights: {self.pretrained}")
    
    def load_model(self) -> None:
        """Load the CLIP model (same as production validation)"""
        if self._model is not None:
            print("✓ CLIP model already loaded")
            return
            
        print(f"🔧 Loading CLIP model: {self.model_name}/{self.pretrained}")
        
        try:
            self._model, _, _ = open_clip.create_model_and_transforms(
                self.model_name, 
                pretrained=self.pretrained, 
                device=self.device
            )
            self._tokenizer = open_clip.get_tokenizer(self.model_name)
            self._model.eval()
            
            print("✅ CLIP model loaded successfully")
            
        except Exception as e:
            print(f"❌ Failed to load CLIP model: {e}")
            raise
    
    def unload_model(self) -> None:
        """Unload the CLIP model to free memory"""
        if self._model is not None:
            print("🧹 Unloading CLIP model...")
            del self._model
            del self._tokenizer
            self._model = None
            self._tokenizer = None
            torch.cuda.empty_cache()
            gc.collect()
            print("✅ CLIP model unloaded")
    
    def get_text_features(self, prompt: str) -> torch.Tensor:
        """Get text features for a prompt"""
        if self._model is None or self._tokenizer is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        
        # Tokenize prompt and move to device
        tokenized_prompt = self._tokenizer(prompt).to(self.device)
        
        with torch.no_grad(), torch.amp.autocast(self.device.type):
            # Encode text
            text_features = self._model.encode_text(tokenized_prompt)
            # Normalize features
            text_features /= text_features.norm(dim=-1, keepdim=True)
        
        return text_features
    
    def compute_text_similarity(self, prompt1: str, prompt2: str) -> float:
        """
        Compute CLIP similarity between two text prompts
        Returns similarity score between 0 and 1
        """
        if self._model is None:
            self.load_model()
        
        try:
            # Get text features for both prompts
            text_features1 = self.get_text_features(prompt1)
            text_features2 = self.get_text_features(prompt2)
            
            # Compute cosine similarity
            with torch.no_grad():
                similarity = (text_features1 @ text_features2.T).cpu().numpy()[0][0]
                # Clip to [0, 1] range
                similarity = np.clip(similarity, 0, 1)
            
            if self.verbose:
                print(f"📊 Text similarity between:")
                print(f"   Prompt 1: '{prompt1}'")
                print(f"   Prompt 2: '{prompt2}'")
                print(f"   Similarity: {similarity:.4f}")
            
            return float(similarity)
            
        except Exception as e:
            print(f"❌ Text similarity computation failed: {e}")
            return 0.0
    
    def compute_prompt_quality_score(self, prompt: str) -> float:
        """
        Compute a quality score for a single prompt using CLIP text encoder
        Higher magnitude features often correlate with better CLIP understanding
        """
        if self._model is None:
            self.load_model()
        
        try:
            # Get text features
            text_features = self.get_text_features(prompt)
            
            # Self-similarity as a proxy for CLIP confidence
            # Higher magnitude features often correlate with better CLIP understanding
            score = float(text_features.norm().cpu())
            
            if self.verbose:
                print(f"📊 Prompt quality score for: '{prompt}'")
                print(f"   Quality score: {score:.4f}")
            
            return score
            
        except Exception as e:
            print(f"❌ Prompt quality computation failed: {e}")
            return 0.0
    
    def compare_multiple_prompts(self, prompts: list[str], reference_prompt: str = None) -> dict:
        """
        Compare multiple prompts against each other or a reference prompt
        """
        if self._model is None:
            self.load_model()
        
        results = {
            'prompts': prompts,
            'reference_prompt': reference_prompt,
            'pairwise_similarities': {},
            'quality_scores': {},
            'reference_similarities': {}
        }
        
        # Compute quality scores for all prompts
        print("📊 Computing quality scores...")
        for i, prompt in enumerate(prompts):
            quality_score = self.compute_prompt_quality_score(prompt)
            results['quality_scores'][prompt] = quality_score
            print(f"   Prompt {i+1}: {quality_score:.4f} - '{prompt}'")
        
        # Compute pairwise similarities
        print("\n📊 Computing pairwise similarities...")
        for i, prompt1 in enumerate(prompts):
            for j, prompt2 in enumerate(prompts):
                if i < j:  # Avoid duplicate comparisons
                    similarity = self.compute_text_similarity(prompt1, prompt2)
                    key = f"{prompt1} <-> {prompt2}"
                    results['pairwise_similarities'][key] = similarity
                    print(f"   {key}: {similarity:.4f}")
        
        # Compute similarities to reference prompt if provided
        if reference_prompt:
            print(f"\n📊 Computing similarities to reference: '{reference_prompt}'")
            for prompt in prompts:
                similarity = self.compute_text_similarity(prompt, reference_prompt)
                results['reference_similarities'][prompt] = similarity
                print(f"   '{prompt}' -> reference: {similarity:.4f}")
        
        return results


def main():
    """Example usage of SimpleCLIPScoreChecker"""
    
    # Initialize checker
    checker = SimpleCLIPScoreChecker(verbose=True)
    
    try:
        # Example 1: Simple text similarity
        print("\n" + "="*60)
        print("EXAMPLE 1: Simple Text Similarity")
        print("="*60)
        
        prompt1 = "a blue ceramic vase with red trim"
        prompt2 = "a blue ceramic vase with red trim, professional 3D render"
        
        similarity = checker.compute_text_similarity(prompt1, prompt2)
        print(f"\n🎯 Final similarity: {similarity:.4f}")
        
        # Example 2: Quality scores
        print("\n" + "="*60)
        print("EXAMPLE 2: Prompt Quality Scores")
        print("="*60)
        
        test_prompts = [
            "vase",
            "a blue ceramic vase with red trim",
            "a blue ceramic vase with red trim, professional 3D render, highly detailed, photorealistic",
            "professional 3D render, Create 3D game asset, isometric view version, highly detailed, photorealistic, studio lighting"
        ]
        
        for prompt in test_prompts:
            quality_score = checker.compute_prompt_quality_score(prompt)
            print(f"Quality: {quality_score:.4f} - '{prompt}'")
        
        # Example 3: Multiple prompt comparison
        print("\n" + "="*60)
        print("EXAMPLE 3: Multiple Prompt Comparison")
        print("="*60)
        
        comparison_prompts = [
            "a blue ceramic vase",
            "a blue ceramic vase with red trim",
            "a red ceramic vase with blue trim",
            "a wooden table"
        ]
        
        results = checker.compare_multiple_prompts(
            prompts=comparison_prompts,
            reference_prompt="a blue ceramic vase with red trim"
        )
        
        print(f"\n📋 Summary:")
        print(f"   Best quality prompt: {max(results['quality_scores'].items(), key=lambda x: x[1])[0]}")
        print(f"   Most similar to reference: {max(results['reference_similarities'].items(), key=lambda x: x[1])[0]}")
        
    finally:
        # Cleanup
        checker.unload_model()


if __name__ == "__main__":
    main() 