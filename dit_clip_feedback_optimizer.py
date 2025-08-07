#!/usr/bin/env python3
"""
DiT + CLIP Feedback Loop Optimizer
Optimizes prompts by using DiT-generated images and CLIP scores as feedback
Pipeline: Text → DiT (Image) → CLIP Score → Prompt Optimization → Repeat
"""

import torch
import open_clip
import numpy as np
import time
import json
import random
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from loguru import logger
import requests
from PIL import Image
import io
import base64

@dataclass
class OptimizationResult:
    original_prompt: str
    optimized_prompt: str
    best_clip_score: float
    improvement_percent: float
    iterations: int
    optimization_time: float
    all_attempts: List[Dict]
    final_image: Optional[str] = None  # base64 encoded

class DiTClipFeedbackOptimizer:
    """Optimizes prompts using DiT-generated images and CLIP scores as feedback"""
    
    def __init__(self, 
                 dit_server_url: str = "http://localhost:8000",
                 clip_model_name: str = "ViT-B-32", 
                 clip_pretrained: str = "openai",
                 max_iterations: int = 5,
                 target_score: float = 0.8,
                 min_improvement: float = 0.02):
        
        self.dit_server_url = dit_server_url
        self.max_iterations = max_iterations
        self.target_score = target_score
        self.min_improvement = min_improvement
        
        # CLIP setup
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.clip_model_name = clip_model_name
        self.clip_pretrained = clip_pretrained
        self.model = None
        self.tokenizer = None
        self.preprocess = None
        self._model_loaded = False
        
        # Optimization strategies
        self.optimization_strategies = {
            'quality_boosters': [
                'high quality', 'ultra detailed', 'photorealistic', 'masterpiece',
                'professional', 'award winning', 'best quality', '8K resolution'
            ],
            'rendering_enhancers': [
                '3D render', 'CGI model', 'game asset', 'digital art',
                'volumetric render', 'ray traced', 'octane render', 'unreal engine'
            ],
            'lighting_modifiers': [
                'studio lighting', 'dramatic lighting', 'soft lighting',
                'rim lighting', 'volumetric lighting', 'global illumination'
            ],
            'camera_angles': [
                'centered composition', 'product photography', 'hero shot',
                'isometric view', 'three quarter view', 'front view'
            ],
            'background_enhancers': [
                'white background', 'studio background', 'gradient background',
                'neutral background', 'clean background', 'professional backdrop'
            ],
            'style_enhancers': [
                'trending on artstation', 'concept art', 'production quality',
                'portfolio piece', 'showcase render', 'advertisement style'
            ]
        }
        
        # High-scoring reference prompts for similarity
        self.reference_prompts = [
            "high quality 3D render on white background",
            "professional CGI model with studio lighting",
            "photorealistic game asset centered composition",
            "ultra detailed 3D model product photography",
            "masterpiece quality volumetric render"
        ]
        
        logger.info(f"🔧 DiT + CLIP Feedback Optimizer initialized")
        logger.info(f"   DiT Server: {dit_server_url}")
        logger.info(f"   CLIP Model: {clip_model_name}")
        logger.info(f"   Max Iterations: {max_iterations}")
        logger.info(f"   Target Score: {target_score}")
    
    def load_clip_model(self):
        """Load CLIP model for scoring"""
        if self._model_loaded:
            return
        
        logger.info(f"📥 Loading CLIP model ({self.clip_model_name})...")
        
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            self.clip_model_name, pretrained=self.clip_pretrained, device=self.device
        )
        self.tokenizer = open_clip.get_tokenizer(self.clip_model_name)
        self.model.eval()
        
        self._model_loaded = True
        logger.info("✅ CLIP model loaded")
    
    def unload_clip_model(self):
        """Unload CLIP model to free GPU memory"""
        if not self._model_loaded:
            return
        
        logger.info("📤 Unloading CLIP model...")
        
        if self.model is not None:
            del self.model
            self.model = None
        
        if self.preprocess is not None:
            del self.preprocess
            self.preprocess = None
        
        if self.tokenizer is not None:
            del self.tokenizer
            self.tokenizer = None
        
        if self.device.type == 'cuda':
            torch.cuda.empty_cache()
        
        self._model_loaded = False
        logger.info("✅ CLIP model unloaded")
    
    def generate_image_with_dit(self, prompt: str, seed: Optional[int] = None) -> Tuple[Optional[str], float]:
        """Generate image using DiT server and return base64 image + generation time"""
        if seed is None:
            seed = random.randint(0, 2**31 - 1)
        
        try:
            start_time = time.time()
            
            # Prepare request to DiT server
            payload = {
                "prompt": prompt,
                "seed": seed,
                "num_inference_steps": 20,  # Adjust based on your DiT setup
                "guidance_scale": 7.5
            }
            
            response = requests.post(
                f"{self.dit_server_url}/generate_image",
                json=payload,
                timeout=60
            )
            
            if response.status_code == 200:
                result = response.json()
                image_base64 = result.get('image')
                generation_time = time.time() - start_time
                
                if image_base64:
                    logger.info(f"✅ DiT image generated in {generation_time:.2f}s")
                    return image_base64, generation_time
                else:
                    logger.error("❌ DiT returned empty image")
                    return None, generation_time
            else:
                logger.error(f"❌ DiT server error: {response.status_code}")
                return None, time.time() - start_time
                
        except Exception as e:
            logger.error(f"❌ DiT generation failed: {e}")
            return None, time.time() - start_time
    
    def compute_clip_score(self, prompt: str, image_base64: str) -> float:
        """Compute CLIP score between prompt and generated image"""
        if not self._model_loaded:
            self.load_clip_model()
        
        try:
            # Decode base64 image
            image_data = base64.b64decode(image_base64)
            image = Image.open(io.BytesIO(image_data))
            
            # Preprocess image for CLIP
            image_tensor = self.preprocess(image).unsqueeze(0).to(self.device)
            
            # Tokenize prompt
            text_tokens = self.tokenizer([prompt]).to(self.device)
            
            with torch.no_grad():
                # Encode image and text
                image_features = self.model.encode_image(image_tensor)
                text_features = self.model.encode_text(text_tokens)
                
                # Normalize features
                image_features /= image_features.norm(dim=-1, keepdim=True)
                text_features /= text_features.norm(dim=-1, keepdim=True)
                
                # Compute similarity
                similarity = (image_features @ text_features.T).cpu().numpy()[0][0]
                
                # Clip to [0, 1] range
                similarity = np.clip(similarity, 0, 1)
                
                return float(similarity)
                
        except Exception as e:
            logger.error(f"❌ CLIP scoring failed: {e}")
            return 0.0
    
    def compute_reference_similarity(self, prompt: str) -> float:
        """Compute similarity to high-quality reference prompts"""
        if not self._model_loaded:
            self.load_clip_model()
        
        try:
            # Tokenize prompt and references
            prompt_tokens = self.tokenizer([prompt]).to(self.device)
            ref_tokens = self.tokenizer(self.reference_prompts).to(self.device)
            
            with torch.no_grad():
                # Encode
                prompt_features = self.model.encode_text(prompt_tokens)
                ref_features = self.model.encode_text(ref_tokens)
                
                # Normalize
                prompt_features /= prompt_features.norm(dim=-1, keepdim=True)
                ref_features /= ref_features.norm(dim=-1, keepdim=True)
                
                # Compute similarities
                similarities = (prompt_features @ ref_features.T).cpu().numpy()
                
                return float(np.mean(similarities))
                
        except Exception as e:
            logger.error(f"❌ Reference similarity failed: {e}")
            return 0.0
    
    def generate_prompt_variations(self, base_prompt: str, strategy: str = "adaptive") -> List[str]:
        """Generate multiple prompt variations based on strategy"""
        variations = [base_prompt]  # Include original
        
        if strategy == "quality_focus":
            # Add quality boosters
            for booster in random.sample(self.optimization_strategies['quality_boosters'], 3):
                variations.append(f"{base_prompt}, {booster}")
        
        elif strategy == "rendering_focus":
            # Add rendering enhancers
            for enhancer in random.sample(self.optimization_strategies['rendering_enhancers'], 3):
                variations.append(f"{base_prompt}, {enhancer}")
        
        elif strategy == "comprehensive":
            # Combine multiple categories
            for _ in range(5):
                modifiers = []
                for category, options in self.optimization_strategies.items():
                    if random.random() > 0.5:  # 50% chance per category
                        modifiers.append(random.choice(options))
                
                if modifiers:
                    variations.append(f"{base_prompt}, {', '.join(modifiers)}")
        
        elif strategy == "adaptive":
            # Adaptive strategy based on prompt analysis
            prompt_lower = base_prompt.lower()
            
            # Check what's missing and add accordingly
            if not any(word in prompt_lower for word in ['quality', 'detailed', 'high']):
                variations.append(f"{base_prompt}, high quality, ultra detailed")
            
            if not any(word in prompt_lower for word in ['render', '3d', 'cgi']):
                variations.append(f"{base_prompt}, 3D render, professional CGI")
            
            if not any(word in prompt_lower for word in ['lighting', 'studio', 'dramatic']):
                variations.append(f"{base_prompt}, studio lighting, dramatic illumination")
            
            if not any(word in prompt_lower for word in ['background', 'white', 'clean']):
                variations.append(f"{base_prompt}, white background, clean studio setting")
        
        # Remove duplicates while preserving order
        seen = set()
        unique_variations = []
        for v in variations:
            if v not in seen:
                seen.add(v)
                unique_variations.append(v)
        
        return unique_variations[:8]  # Limit to 8 variations
    
    def optimize_prompt_with_feedback(self, original_prompt: str, seed: Optional[int] = None) -> OptimizationResult:
        """Main optimization loop using DiT + CLIP feedback"""
        start_time = time.time()
        
        logger.info(f"🚀 Starting DiT + CLIP feedback optimization for: '{original_prompt}'")
        
        best_prompt = original_prompt
        best_score = 0.0
        best_image = None
        all_attempts = []
        
        # Initial evaluation
        logger.info("📊 Evaluating original prompt...")
        initial_image, gen_time = self.generate_image_with_dit(original_prompt, seed)
        if initial_image:
            initial_score = self.compute_clip_score(original_prompt, initial_image)
            logger.info(f"   Original CLIP score: {initial_score:.4f}")
            best_score = initial_score
            best_image = initial_image
            
            all_attempts.append({
                'prompt': original_prompt,
                'clip_score': initial_score,
                'generation_time': gen_time,
                'image': initial_image,
                'iteration': 0
            })
        
        # Optimization loop
        for iteration in range(1, self.max_iterations + 1):
            logger.info(f"\n🔄 Iteration {iteration}/{self.max_iterations}")
            
            # Generate variations based on current best
            strategy = "adaptive" if iteration == 1 else "comprehensive"
            variations = self.generate_prompt_variations(best_prompt, strategy)
            
            logger.info(f"   Testing {len(variations)} variations...")
            
            iteration_best_score = best_score
            iteration_best_prompt = best_prompt
            
            # Test each variation
            for i, variation in enumerate(variations):
                if variation == best_prompt:  # Skip if same as current best
                    continue
                
                # Generate image with variation
                image, gen_time = self.generate_image_with_dit(variation, seed)
                if not image:
                    continue
                
                # Compute CLIP score
                clip_score = self.compute_clip_score(variation, image)
                
                # Also compute reference similarity for additional guidance
                ref_similarity = self.compute_reference_similarity(variation)
                
                # Combined score (weighted average)
                combined_score = 0.7 * clip_score + 0.3 * ref_similarity
                
                logger.info(f"     {i+1}. Score: {combined_score:.4f} (CLIP: {clip_score:.4f}, Ref: {ref_similarity:.4f})")
                logger.info(f"        Prompt: '{variation}'")
                
                all_attempts.append({
                    'prompt': variation,
                    'clip_score': clip_score,
                    'ref_similarity': ref_similarity,
                    'combined_score': combined_score,
                    'generation_time': gen_time,
                    'image': image,
                    'iteration': iteration
                })
                
                # Update best if improved
                if combined_score > iteration_best_score:
                    iteration_best_score = combined_score
                    iteration_best_prompt = variation
                    if combined_score > best_score:
                        best_score = combined_score
                        best_prompt = variation
                        best_image = image
                        logger.info(f"     🏆 New best score: {best_score:.4f}")
            
            # Check for convergence
            improvement = iteration_best_score - (best_score - (iteration_best_score - best_score))
            if improvement < self.min_improvement:
                logger.info(f"   ⏸️  Minimal improvement ({improvement:.4f}), stopping early")
                break
            
            # Check if target reached
            if best_score >= self.target_score:
                logger.info(f"   🎯 Target score reached: {best_score:.4f}")
                break
        
        optimization_time = time.time() - start_time
        improvement_percent = ((best_score - all_attempts[0]['clip_score']) / 
                              all_attempts[0]['clip_score'] * 100) if all_attempts else 0
        
        logger.info(f"\n✅ Optimization completed!")
        logger.info(f"   Original score: {all_attempts[0]['clip_score']:.4f}")
        logger.info(f"   Best score: {best_score:.4f} (+{improvement_percent:.1f}%)")
        logger.info(f"   Best prompt: '{best_prompt}'")
        logger.info(f"   Total time: {optimization_time:.2f}s")
        
        return OptimizationResult(
            original_prompt=original_prompt,
            optimized_prompt=best_prompt,
            best_clip_score=best_score,
            improvement_percent=improvement_percent,
            iterations=len(set(attempt['iteration'] for attempt in all_attempts)),
            optimization_time=optimization_time,
            all_attempts=all_attempts,
            final_image=best_image
        )
    
    def batch_optimize(self, prompts: List[str], seeds: Optional[List[int]] = None) -> List[OptimizationResult]:
        """Optimize multiple prompts in batch"""
        if seeds is None:
            seeds = [random.randint(0, 2**31 - 1) for _ in prompts]
        
        results = []
        for i, (prompt, seed) in enumerate(zip(prompts, seeds)):
            logger.info(f"\n{'='*60}")
            logger.info(f"Batch optimization {i+1}/{len(prompts)}")
            logger.info(f"{'='*60}")
            
            result = self.optimize_prompt_with_feedback(prompt, seed)
            results.append(result)
        
        return results
    
    def save_results(self, results: List[OptimizationResult], filename: str):
        """Save optimization results to JSON file"""
        data = []
        for result in results:
            data.append({
                'original_prompt': result.original_prompt,
                'optimized_prompt': result.optimized_prompt,
                'best_clip_score': result.best_clip_score,
                'improvement_percent': result.improvement_percent,
                'iterations': result.iterations,
                'optimization_time': result.optimization_time,
                'all_attempts': result.all_attempts,
                'final_image': result.final_image[:100] + "..." if result.final_image else None  # Truncate for JSON
            })
        
        with open(filename, 'w') as f:
            json.dump(data, f, indent=2)
        
        logger.info(f"💾 Results saved to {filename}")
    
    def __del__(self):
        """Cleanup when object is destroyed"""
        self.unload_clip_model()

def main():
    """Test the DiT + CLIP feedback optimizer"""
    
    # Test prompts
    test_prompts = [
        "red ceramic vase",
        "metallic robot",
        "glass container with flowers",
        "wooden chair",
        "plastic bottle"
    ]
    
    # Initialize optimizer
    optimizer = DiTClipFeedbackOptimizer(
        dit_server_url="http://localhost:8000",  # Adjust to your DiT server
        max_iterations=3,
        target_score=0.75,
        min_improvement=0.01
    )
    
    print("🚀 DiT + CLIP Feedback Optimization Demo")
    print("=" * 80)
    
    # Optimize each prompt
    results = optimizer.batch_optimize(test_prompts)
    
    # Save results
    optimizer.save_results(results, "dit_clip_optimization_results.json")
    
    # Summary
    print("\n📊 Optimization Summary:")
    print("=" * 80)
    total_improvement = 0
    for i, result in enumerate(results):
        print(f"{i+1}. '{result.original_prompt[:30]}...'")
        print(f"   Score: {result.best_clip_score:.4f} (+{result.improvement_percent:.1f}%)")
        print(f"   Optimized: '{result.optimized_prompt[:50]}...'")
        print()
        total_improvement += result.improvement_percent
    
    avg_improvement = total_improvement / len(results)
    print(f"Average improvement: {avg_improvement:.1f}%")
    
    # Cleanup
    optimizer.unload_clip_model()

if __name__ == "__main__":
    main() 