#!/usr/bin/env python3
"""
DiT + CLIP Optimizer Integration
Simple integration for existing DiT + TRELLIS pipelines
Adds prompt optimization using CLIP feedback before 3D generation
"""

import torch
import open_clip
import numpy as np
import time
import random
from typing import List, Dict, Optional, Tuple
from loguru import logger
import requests
from PIL import Image
import io
import base64

class DiTClipOptimizer:
    """Simple DiT + CLIP prompt optimizer for integration"""
    
    def __init__(self, 
                 dit_server_url: str = "http://localhost:8000",
                 max_iterations: int = 3,
                 target_score: float = 0.7):
        
        self.dit_server_url = dit_server_url
        self.max_iterations = max_iterations
        self.target_score = target_score
        
        # CLIP setup
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.tokenizer = None
        self.preprocess = None
        self._model_loaded = False
        
        # Quick optimization templates
        self.optimization_templates = [
            "{prompt}, high quality, ultra detailed",
            "{prompt}, 3D render, professional CGI",
            "{prompt}, studio lighting, white background",
            "{prompt}, masterpiece quality, photorealistic",
            "{prompt}, centered composition, product photography",
            "{prompt}, trending on artstation, concept art",
            "{prompt}, volumetric render, ray traced",
            "{prompt}, award winning, best quality"
        ]
        
        logger.info(f"🔧 DiT + CLIP Optimizer initialized (max_iterations={max_iterations})")
    
    def load_clip_model(self):
        """Load CLIP model"""
        if self._model_loaded:
            return
        
        logger.info("📥 Loading CLIP model...")
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            "ViT-B-32", pretrained="openai", device=self.device
        )
        self.tokenizer = open_clip.get_tokenizer("ViT-B-32")
        self.model.eval()
        self._model_loaded = True
    
    def generate_dit_image(self, prompt: str, seed: Optional[int] = None) -> Optional[str]:
        """Generate image with DiT server"""
        if seed is None:
            seed = random.randint(0, 2**31 - 1)
        
        try:
            payload = {
                "prompt": prompt,
                "seed": seed,
                "num_inference_steps": 20,
                "guidance_scale": 7.5
            }
            
            response = requests.post(
                f"{self.dit_server_url}/generate_image",
                json=payload,
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                return result.get('image')
            else:
                logger.warning(f"DiT server error: {response.status_code}")
                return None
                
        except Exception as e:
            logger.warning(f"DiT generation failed: {e}")
            return None
    
    def compute_clip_score(self, prompt: str, image_base64: str) -> float:
        """Compute CLIP score between prompt and image"""
        if not self._model_loaded:
            self.load_clip_model()
        
        try:
            # Decode and preprocess image
            image_data = base64.b64decode(image_base64)
            image = Image.open(io.BytesIO(image_data))
            image_tensor = self.preprocess(image).unsqueeze(0).to(self.device)
            
            # Tokenize prompt
            text_tokens = self.tokenizer([prompt]).to(self.device)
            
            with torch.no_grad():
                # Encode
                image_features = self.model.encode_image(image_tensor)
                text_features = self.model.encode_text(text_tokens)
                
                # Normalize and compute similarity
                image_features /= image_features.norm(dim=-1, keepdim=True)
                text_features /= text_features.norm(dim=-1, keepdim=True)
                similarity = (image_features @ text_features.T).cpu().numpy()[0][0]
                
                return float(np.clip(similarity, 0, 1))
                
        except Exception as e:
            logger.warning(f"CLIP scoring failed: {e}")
            return 0.0
    
    def optimize_prompt(self, original_prompt: str, seed: Optional[int] = None) -> Tuple[str, float, Dict]:
        """Optimize prompt using DiT + CLIP feedback"""
        start_time = time.time()
        
        logger.info(f"🔍 Optimizing prompt: '{original_prompt}'")
        
        best_prompt = original_prompt
        best_score = 0.0
        attempts = []
        
        # Test original prompt first
        original_image = self.generate_dit_image(original_prompt, seed)
        if original_image:
            original_score = self.compute_clip_score(original_prompt, original_image)
            best_score = original_score
            attempts.append({
                'prompt': original_prompt,
                'score': original_score,
                'iteration': 0
            })
            logger.info(f"   Original score: {original_score:.4f}")
        
        # Optimization loop
        for iteration in range(1, self.max_iterations + 1):
            logger.info(f"   Iteration {iteration}/{self.max_iterations}")
            
            # Generate variations
            variations = []
            for template in random.sample(self.optimization_templates, 4):
                variations.append(template.format(prompt=original_prompt))
            
            # Test variations
            for variation in variations:
                if variation == best_prompt:
                    continue
                
                image = self.generate_dit_image(variation, seed)
                if not image:
                    continue
                
                score = self.compute_clip_score(variation, image)
                attempts.append({
                    'prompt': variation,
                    'score': score,
                    'iteration': iteration
                })
                
                logger.info(f"     '{variation[:50]}...' -> {score:.4f}")
                
                if score > best_score:
                    best_score = score
                    best_prompt = variation
                    logger.info(f"     🏆 New best: {score:.4f}")
                
                # Early stopping if target reached
                if best_score >= self.target_score:
                    logger.info(f"     🎯 Target reached: {best_score:.4f}")
                    break
            
            # Early stopping if target reached
            if best_score >= self.target_score:
                break
        
        optimization_time = time.time() - start_time
        improvement = ((best_score - attempts[0]['score']) / attempts[0]['score'] * 100) if attempts else 0
        
        logger.info(f"✅ Optimization complete: {best_score:.4f} (+{improvement:.1f}%) in {optimization_time:.1f}s")
        
        return best_prompt, best_score, {
            'original_prompt': original_prompt,
            'optimized_prompt': best_prompt,
            'best_score': best_score,
            'improvement_percent': improvement,
            'iterations': iteration,
            'optimization_time': optimization_time,
            'attempts': attempts
        }
    
    def cleanup(self):
        """Cleanup resources"""
        if self._model_loaded:
            del self.model, self.tokenizer, self.preprocess
            self.model = self.tokenizer = self.preprocess = None
            self._model_loaded = False
            if self.device.type == 'cuda':
                torch.cuda.empty_cache()

# Integration example for existing pipelines
def integrate_with_trellis_pipeline():
    """Example of how to integrate with existing TRELLIS pipeline"""
    
    # Initialize optimizer
    optimizer = DiTClipOptimizer(
        dit_server_url="http://localhost:8000",  # Your DiT server
        max_iterations=3,
        target_score=0.7
    )
    
    def generate_3d_with_optimization(prompt: str, seed: Optional[int] = None):
        """Enhanced 3D generation with prompt optimization"""
        
        # Step 1: Optimize prompt using DiT + CLIP feedback
        optimized_prompt, clip_score, optimization_data = optimizer.optimize_prompt(prompt, seed)
        
        # Step 2: Generate 3D model with optimized prompt
        # (Your existing TRELLIS generation code here)
        logger.info(f"🎯 Generating 3D model with optimized prompt: '{optimized_prompt}'")
        
        # Example: Call your existing 3D generation function
        # ply_data = your_trellis_generation_function(optimized_prompt, seed)
        
        return {
            'original_prompt': prompt,
            'optimized_prompt': optimized_prompt,
            'clip_score': clip_score,
            'optimization_data': optimization_data,
            'ply_data': None  # Replace with actual PLY data
        }
    
    return generate_3d_with_optimization

# Simple usage example
if __name__ == "__main__":
    optimizer = DiTClipOptimizer()
    
    test_prompts = [
        "red ceramic vase",
        "metallic robot",
        "wooden chair"
    ]
    
    for prompt in test_prompts:
        optimized_prompt, score, data = optimizer.optimize_prompt(prompt)
        print(f"\nOriginal: '{prompt}'")
        print(f"Optimized: '{optimized_prompt}'")
        print(f"Score: {score:.4f} (+{data['improvement_percent']:.1f}%)")
    
    optimizer.cleanup() 