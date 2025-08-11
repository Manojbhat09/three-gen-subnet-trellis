#!/usr/bin/env python3
"""
Advanced Prompt Optimization Engine
Purpose: Maximize CLIP alignment scores through iterative optimization using
         image interrogator, CLIP feedback, and LoRA-aware generation
"""

import asyncio
import json
import time
import logging
import base64
import requests
import subprocess
import tempfile
import torch
import torch.nn.functional as F
import open_clip
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional, Any, Tuple
from dataclasses import dataclass
from PIL import Image
import io
from sentence_transformers import SentenceTransformer
import os

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class OptimizationResult:
    """Result of a single optimization iteration"""
    iteration: int
    original_prompt: str
    optimized_prompt: str
    original_score: float
    optimized_score: float
    improvement: float
    lora_endpoint: str
    strategy_used: str
    interrogator_output: Optional[str] = None
    convergence_reached: bool = False


@dataclass
class OptimizationSession:
    """Complete optimization session results"""
    session_id: str
    original_prompt: str
    final_prompt: str
    original_score: float
    final_score: float
    total_improvement: float
    iterations: List[OptimizationResult]
    best_iteration: int
    convergence_iteration: Optional[int] = None
    total_time: float = 0.0


class ImageInterrogatorInterface:
    """Interface to the existing image-interrogator framework"""
    
    def __init__(self, 
                 clip_model_name: str = "ViT-L-14/openai",
                 caption_model_name: str = "blip-large"):
        
        # Import the existing image interrogator
        import sys
        import os
        sys.path.append(os.path.join(os.path.dirname(__file__), "image-interrogator"))
        
        from clip_interrogator.clip_interrogator import Config, Interrogator
        
        self.config = Config()
        self.config.clip_model_name = clip_model_name
        self.config.caption_model_name = caption_model_name
        self.config.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.config.generate_features = True
        self.config.cache_path = './cache/embeddings'
        self.config.chunk_size = 2048
        self.config.flavor_intermediate_count = 2048
        # Enable cache for faster loading
        self.config.download_cache = True
        
        self.interrogator = None
        logger.info(f"🔍 Image Interrogator initialized with {clip_model_name}/{caption_model_name}")
        
    def _load_interrogator(self):
        """Lazy load the interrogator to avoid memory issues"""
        if self.interrogator is None:
            from clip_interrogator.clip_interrogator import Interrogator
            self.interrogator = Interrogator(self.config)
            logger.info("✅ Image Interrogator models loaded")
    
    def _unload_interrogator(self):
        """Unload interrogator to free memory"""
        if self.interrogator is not None:
            del self.interrogator
            self.interrogator = None
            torch.cuda.empty_cache()
            logger.info("🧹 Image Interrogator models unloaded")
        
    def interrogate_image(self, image: Image.Image, style_focus: str = "detailed") -> str:
        """Use the existing image interrogator to generate optimized prompt from image"""
        try:
            self._load_interrogator()
            
            # Convert PIL image to RGB if needed
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # Configure interrogation based on style focus
            question_prompt = self._get_interrogation_prompt(style_focus)
            
            if style_focus == "detailed":
                # Use full interrogation with all features
                # Some versions of clip_interrogator do not accept 'question_prompt' kwarg; call with image only
                result = self.interrogator.interrogate(image)
            elif style_focus == "3d_optimized":
                # Focus on 3D-specific features
                result = self.interrogator.interrogate_classic(image)
            elif style_focus == "clip_optimized":
                # Use fast interrogation optimized for CLIP
                result = self.interrogator.interrogate_fast(image)
            else:
                # Default to standard interrogation
                result = self.interrogator.interrogate(image)
            
            # Clean up the result
            cleaned_result = self._clean_interrogation_result(result)
            
            # NOTE: Keep interrogator loaded for subsequent strategy calls; caller will unload when finished
            # self._unload_interrogator()
            
            return cleaned_result
                
        except Exception as e:
            logger.error(f"Image interrogation error: {e}")
            # Do not unload here; caller may want to retry
            # self._unload_interrogator()
            return None
    
    def _get_interrogation_prompt(self, style_focus: str) -> str:
        """Get the appropriate interrogation prompt based on focus"""
        prompts = {
            "detailed": "Describe this image in detail for text-to-image generation",
            "3d_optimized": "Describe this object for 3D model generation", 
            "clip_optimized": "Create a descriptive prompt that captures this image",
            "default": ""
        }
        return prompts.get(style_focus, prompts["default"])
    
    def _clean_interrogation_result(self, result: str) -> str:
        """Clean and optimize the interrogation result"""
        if not result:
            return ""
        
        # Remove common prefixes/suffixes that might interfere
        prefixes_to_remove = [
            "a painting of ", "an image of ", "a photo of ",
            "a picture of ", "a drawing of ", "artwork of "
        ]
        
        cleaned = result.strip()
        
        # Remove problematic prefixes
        for prefix in prefixes_to_remove:
            if cleaned.lower().startswith(prefix):
                cleaned = cleaned[len(prefix):]
                break
        
        # Clean up multiple commas and spaces
        import re
        cleaned = re.sub(r',\s*,', ',', cleaned)  # Remove double commas
        cleaned = re.sub(r'\s+', ' ', cleaned)    # Normalize whitespace
        cleaned = cleaned.strip(' ,')             # Remove leading/trailing commas
        
        return cleaned


class CLIPAlignmentOptimizer:
    """Core CLIP alignment optimizer using feedback loops"""
    
    def __init__(self, 
                 hunyuan_server_url: str = "http://localhost:8098",
                 clip_model_name: str = "convnext_large_d",
                 clip_pretrained: str = "laion2b_s26b_b102k_augreg"):
        
        self.hunyuan_server_url = hunyuan_server_url
        self.clip_model_name = clip_model_name
        self.clip_pretrained = clip_pretrained
        
        # Initialize components
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Initialize image interrogator with same CLIP model as production
        # Format: convnext_large_d/laion2b_s26b_b102k_augreg
        interrogator_clip_model = f"{clip_model_name}/{clip_pretrained}"
        self.interrogator = ImageInterrogatorInterface(
            clip_model_name=interrogator_clip_model,  # Use same CLIP model as production
            caption_model_name="blip-large"
        )
        # Reduce flavor budget to limit stylistic drift
        try:
            self.interrogator.config.flavor_intermediate_count = 128
        except Exception:
            pass
        self.sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # CLIP model (lazy loaded)
        self._clip_model = None
        self._clip_tokenizer = None
        
        # Available LoRA endpoints
        self.lora_endpoints = [
            "isometric_3d", "live_3d", "game_assets", "patched_realism",
            "tf2_style", "baolei", "cartoon_3d", "cinema", "sd15_game_icon"
        ]
        
        # Optimization settings
        self.max_iterations = 5
        self.convergence_threshold = 0.01  # Stop if improvement < 1%
        self.target_score = 0.8  # Aim for excellent scores
        
        # Interrogator preferences
        self.use_interrogator = True
        self.interrogator_mode = "clip_optimized"  # fast mode
        self.max_new_terms = 2
        self.min_phrase_similarity = 0.55
        self.block_trending = True
        self.block_artists = True
        self.block_movements = True
        self.block_mediums = True
        self.allowed_modifiers = [
            "three dimensional", "3d", "well lit", "studio lighting", "high quality", "sharp", "clear",
            "front view", "full object", "single object", "centered composition",
            "clean product render", "isolated on white background", "plain white background",
        ]
        self.min_sentence_similarity_final = 0.84
        self.max_added_tokens = 6
        self.grid_search_max_terms = 2
        self.grid_search_max_candidates = 24
        # Require at least 50% relative improvement over original to consider convergence/target met
        self.min_improvement_ratio = 0.50
        self.min_convergence_ratio = 0.50
        # Organic suffix via LLM (optional)
        self.use_organic_suffix = True
        self.organic_llm_url = os.getenv("ORGANIC_LLM_URL", "http://localhost:11434/api/chat")
        self.organic_llm_model = os.getenv("ORGANIC_LLM_MODEL", "llama3.2:3b")
        # Disable slower strategies by default
        self.enable_grid_search = False
        self.use_semantic_enhancement = False

        # Known drift patterns (subset)
        self._trending_sites = set([
            'artstation', 'behance', 'cg society', 'cgsociety', 'deviantart', 'dribbble',
            'flickr', 'instagram', 'pexels', 'pinterest', 'pixabay', 'pixiv', 'polycount',
            'reddit', 'shutterstock', 'tumblr', 'unsplash', 'zbrush central'
        ])
        
        logger.info(f"🎯 CLIP Alignment Optimizer initialized")
        logger.info(f"   Server: {hunyuan_server_url}")
        logger.info(f"   CLIP Model: {clip_model_name}/{clip_pretrained}")
 
    def load_clip_model(self):
        """Load CLIP model for scoring"""
        if self._clip_model is not None:
            return
            
        logger.info(f"🔧 Loading CLIP model: {self.clip_model_name}")
        self._clip_model, _, _ = open_clip.create_model_and_transforms(
            self.clip_model_name, 
            pretrained=self.clip_pretrained, 
            device=self.device
        )
        self._clip_tokenizer = open_clip.get_tokenizer(self.clip_model_name)
        self._clip_model.eval()
        
        # Set up normalization (same as production)
        mean = torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(1, 3, 1, 1) * 3
        std = torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(1, 3, 1, 1) * 3
        from torchvision import transforms as _tv_transforms
        self._normalize_transform = _tv_transforms.Normalize(mean.squeeze(), std.squeeze())
        
        logger.info("✅ CLIP model loaded successfully")
 
    def unload_clip_model(self):
        """Unload CLIP model to free memory"""
        if self._clip_model is not None:
            del self._clip_model
            del self._clip_tokenizer
            self._clip_model = None
            self._clip_tokenizer = None
            torch.cuda.empty_cache()
            logger.info("🧹 CLIP model unloaded")
 
    def compute_clip_score(self, prompt: str, image: Image.Image) -> float:
        """Compute CLIP alignment score (same as production)"""
        if self._clip_model is None:
            raise RuntimeError("CLIP model not loaded")
        
        try:
            # Preprocess image
            image_tensor = torch.tensor(np.array(image)).float() / 255.0
            if len(image_tensor.shape) == 3:
                image_tensor = image_tensor.permute(2, 0, 1)
            image_tensor = image_tensor.unsqueeze(0)
            image_tensor = F.interpolate(image_tensor, size=(224, 224), mode="bicubic", align_corners=False)
            image_tensor = self._normalize_transform(image_tensor).to(self.device)
            
            # Tokenize prompt
            tokenized_prompt = self._clip_tokenizer(prompt).to(self.device)
            
            with torch.no_grad(), torch.amp.autocast(self.device.type):
                # Encode
                image_features = self._clip_model.encode_image(image_tensor)
                text_features = self._clip_model.encode_text(tokenized_prompt)
                
                # Normalize
                image_features /= image_features.norm(dim=-1, keepdim=True)
                text_features /= text_features.norm(dim=-1, keepdim=True)
                
                # Compute similarity
                similarity = (image_features @ text_features.T).cpu().numpy()[0][0]
                return float(np.clip(similarity, 0, 1))
                
        except Exception as e:
            logger.error(f"CLIP scoring failed: {e}")
            return 0.0

    def _tokenize_phrases(self, text: str) -> list[str]:
        parts = [p.strip() for p in text.split(',')]
        parts = [p for p in parts if p]
        return parts

    def _is_blocked_phrase(self, phrase_lower: str) -> bool:
        if self.block_artists and (phrase_lower.startswith("by ") or phrase_lower.startswith("inspired by ")):
            return True
        if self.block_trending and ("trending" in phrase_lower or any(site in phrase_lower for site in self._trending_sites)):
            return True
        # Heuristic: drop explicit movement/medium keywords that often induce style drift
        blocked_keywords = [
            "movement", "cubism", "impressionism", "baroque", "renaissance",
            "digital painting", "watercolor", "oil painting", "photorealism", "low poly",
        ]
        if self.block_movements or self.block_mediums:
            if any(k in phrase_lower for k in blocked_keywords):
                return True
        return False

    def _conservative_blend_interrogated(self, original: str, interrogated: str, lora_endpoint: str) -> str:
        """Blend interrogator output with original using conservative, alignment-friendly rules."""
        original_lower = original.lower()
        # Significant tokens from original (basic noun-lock heuristic)
        import re
        orig_tokens = [t for t in re.findall(r"[a-zA-Z]+", original_lower) if len(t) >= 3]
        orig_token_set = set(orig_tokens)
        phrases = self._tokenize_phrases(interrogated)
        # Drop blocked phrases and those already present in original
        candidates = []
        for p in phrases:
            pl = p.lower()
            if self._is_blocked_phrase(pl):
                continue
            if pl in original_lower:
                continue
            # avoid obvious new nouns: crude heuristic - keep short modifiers
            if len(pl.split()) > 4:
                continue
            # Only allow modifiers that map to our whitelist (substring match)
            allowed = None
            for mod in self.allowed_modifiers:
                if mod in pl:
                    allowed = mod
                    break
            if not allowed:
                continue
            candidates.append(allowed)
        # Rank by semantic similarity to original text
        try:
            orig_emb = self.sentence_model.encode([original], normalize_embeddings=True)
            cand_embs = self.sentence_model.encode(candidates, normalize_embeddings=True) if candidates else []
        except Exception:
            cand_embs = []
        if not candidates or len(cand_embs) == 0:
            # Fall back to minimal style + semantics only
            blended = self._align_with_lora_style(original, lora_endpoint)
            blended = self._enhance_semantically(blended, lora_endpoint)
            return blended
        import numpy as np
        sims = (cand_embs @ orig_emb.T).reshape(-1)
        # Select top-N above threshold
        ranked = sorted([(float(sims[i]), candidates[i]) for i in range(len(candidates))], reverse=True)
        selected = [c for s, c in ranked if s >= self.min_phrase_similarity][: self.max_new_terms]
        blended = original
        if selected:
            blended = f"{blended}, {', '.join(selected)}"
        blended = self._align_with_lora_style(blended, lora_endpoint)
        # Minimal semantic boost
        blended = self._enhance_semantically(blended, lora_endpoint)
        # Length cap
        new_tokens = re.findall(r"[\w-]+", blended)
        if len(new_tokens) > len(orig_tokens) + self.max_added_tokens:
            # truncate by removing last comma-separated segment(s)
            parts = self._tokenize_phrases(blended)
            while parts and len(re.findall(r"[\w-]+", ", ".join(parts))) > len(orig_tokens) + self.max_added_tokens:
                parts.pop()
            blended = ", ".join(parts) if parts else original
        # Noun-lock heuristic: ensure all significant original tokens remain
        lower_blended = blended.lower()
        if not all(tok in lower_blended for tok in orig_token_set):
            blended = original
        return blended

    def _grid_search_modifiers(self, original: str, lora_endpoint: str) -> list[str]:
        """Generate a small set of candidate prompts from allowed modifiers (1-2 terms)."""
        base = self._align_with_lora_style(original, lora_endpoint)
        # Build unique candidates
        candidates: list[str] = []
        # Single-term
        for mod in self.allowed_modifiers:
            cand = f"{base}, {mod}"
            candidates.append(cand)
            if len(candidates) >= self.grid_search_max_candidates:
                break
        # Two-terms (first few pairs)
        if len(candidates) < self.grid_search_max_candidates:
            count_added = 0
            for i in range(min(4, len(self.allowed_modifiers))):
                for j in range(i + 1, min(6, len(self.allowed_modifiers))):
                    cand = f"{base}, {self.allowed_modifiers[i]}, {self.allowed_modifiers[j]}"
                    candidates.append(cand)
                    count_added += 1
                    if len(candidates) >= self.grid_search_max_candidates:
                        break
                if len(candidates) >= self.grid_search_max_candidates:
                    break
        # Ensure de-dup
        seen = set()
        uniq = []
        for c in candidates:
            if c not in seen:
                seen.add(c)
                uniq.append(c)
        return uniq

    def _rewrite_with_organic_suffix(self, original: str) -> str:
        """Use an LLM to add an organic, natural suffix that keeps the object and adds clean white background/centering subtly."""
        if not self.use_organic_suffix:
            return original
        prompt_inst = (
            "Rewrite the following short object prompt to remain semantically identical, "
            "but add a natural, concise photographic rendering note that implies a clean product shot: "
            "plain white background, centered composition, front view. Do not add new nouns or styles. "
            "Keep it under 20 extra words.\n\n"
            f"Original: {original}\n"
            "Rewrite:"
        )
        try:
            # Try local Ollama chat API
            payload = {
                "model": self.organic_llm_model,
                "messages": [{"role": "user", "content": prompt_inst}],
                "stream": False,
                "options": {"temperature": 0.3, "num_predict": 64, "top_p": 0.9},
            }
            resp = requests.post(self.organic_llm_url, json=payload, timeout=5)
            if resp.ok:
                data = resp.json()
                rewritten = data.get("message", {}).get("content", "").strip()
                # Basic sanity: ensure original head terms remain
                if rewritten and len(rewritten) <= len(original) + 80:
                    return rewritten
        except Exception:
            pass
        # Fallback: static organic suffix
        fallback = f"{original}, plain white background, centered composition, front view"
        return fallback

    def generate_image(self, prompt: str, seed: int = 42, lora_endpoint: str = "isometric_3d") -> Optional[Image.Image]:
        """Generate image using specified LoRA endpoint"""
        try:
            endpoint = f"/generate_image/{lora_endpoint}/"
            response = requests.post(
                f"{self.hunyuan_server_url}{endpoint}",
                data={'prompt': prompt, 'seed': seed},
                timeout=300
            )
            
            if response.status_code == 200:
                result = response.json()
                if result.get('status') == 'success':
                    image_data = base64.b64decode(result['image'])
                    return Image.open(io.BytesIO(image_data))
            
            logger.error(f"Image generation failed: {response.status_code}")
            return None
            
        except Exception as e:
            logger.error(f"Image generation error: {e}")
            return None
    
    def optimize_for_lora_endpoint(self, 
                                   prompt: str, 
                                   lora_endpoint: str, 
                                   seed: int = 42) -> OptimizationResult:
        """Optimize prompt for a specific LoRA endpoint using feedback loop"""
        
        logger.info(f"🎯 Optimizing '{prompt[:50]}...' for {lora_endpoint}")
        
        # Generate initial image and score
        original_image = self.generate_image(prompt, seed, lora_endpoint)
        if original_image is None:
            return OptimizationResult(
                iteration=0, original_prompt=prompt, optimized_prompt=prompt,
                original_score=0.0, optimized_score=0.0, improvement=0.0,
                lora_endpoint=lora_endpoint, strategy_used="failed_generation"
            )
        
        original_score = self.compute_clip_score(prompt, original_image)
        logger.info(f"   Original score: {original_score:.4f}")
        
        best_prompt = prompt
        best_score = original_score
        best_image = original_image
        
        # Prefer conservative interrogator-based blending first (fast), then minimal style/semantics
        strategies = [
            ("interrogator_conservative", self.interrogator_mode),
            ("style_alignment", None),
            ("organic_suffix", None),
        ]
        
        try:
            # Preload interrogator once for all strategies
            self.interrogator._load_interrogator()
        except Exception:
            pass
        
        for iteration, (strategy, interrogator_style) in enumerate(strategies, 1):
            try:
                if strategy == "interrogator_conservative" and self.use_interrogator:
                    interrogated_prompt = self.interrogator.interrogate_image(best_image, interrogator_style or "clip_optimized")
                    if not interrogated_prompt:
                        continue
                    logger.info(f"   {strategy} → interrogated: '{interrogated_prompt}'")
                    optimized_prompt = self._conservative_blend_interrogated(prompt, interrogated_prompt, lora_endpoint)
                
                elif strategy == "semantic_enhancement":
                    optimized_prompt = self._enhance_semantically(best_prompt, lora_endpoint)
                    if not self.use_semantic_enhancement:
                        continue
                
                elif strategy == "style_alignment":
                    optimized_prompt = self._align_with_lora_style(best_prompt, lora_endpoint)

                elif strategy == "organic_suffix":
                    optimized_prompt = self._rewrite_with_organic_suffix(best_prompt)

                else:
                    continue
                
                logger.info(f"   {strategy} → candidate: '{optimized_prompt}'")
                
                # Apply text-similarity guard only to interrogator-based candidates
                if strategy == "interrogator_conservative":
                    try:
                        sim_text = float(self.sentence_model.similarity(
                            self.sentence_model.encode([optimized_prompt], normalize_embeddings=True),
                            self.sentence_model.encode([prompt], normalize_embeddings=True)
                        ))
                    except Exception:
                        sim_text = 1.0
                    if sim_text < self.min_sentence_similarity_final:
                        logger.info(f"   Skipping candidate due to low text similarity ({sim_text:.3f} < {self.min_sentence_similarity_final:.2f})")
                        continue
                
                # Test optimized prompt
                optimized_image = self.generate_image(optimized_prompt, seed, lora_endpoint)
                if optimized_image is None:
                    continue
                
                optimized_score = self.compute_clip_score(prompt, optimized_image)  # Score against original intent
                improvement = optimized_score - best_score
                
                logger.info(f"   Strategy {strategy}: {optimized_score:.4f} (Δ{improvement:+.4f})")
                
                # Keep if improved
                if improvement > 0:
                    best_prompt = optimized_prompt
                    best_score = optimized_score
                    best_image = optimized_image
                    
                    # Check for convergence only if we've reached sufficient relative improvement
                    if improvement < self.convergence_threshold:
                        if original_score > 0 and (best_score - original_score) >= self.min_convergence_ratio * original_score:
                            logger.info(f"   ✅ Converged after {iteration} iterations (≥{self.min_convergence_ratio*100:.0f}% relative improvement)")
                            break
                 
                    # Check if target reached
                    normalized_score = best_score / 0.35
                    if normalized_score >= self.target_score:
                        logger.info(f"   🎯 Target score reached: {normalized_score:.4f}")
                        break
                 
            except Exception as e:
                logger.error(f"   Strategy {strategy} failed: {e}")
                continue
 
        # If improvement is below the 50% target, try a small modifier grid search
        if self.enable_grid_search and ((original_score == 0 and best_score <= 0) or (original_score > 0 and (best_score - original_score) < self.min_improvement_ratio * original_score)):
            logger.info("   🔎 Grid search over allowed modifiers...")
            candidates = self._grid_search_modifiers(prompt, lora_endpoint)
            for cand in candidates:
                logger.info(f"   grid → candidate: '{cand}'")
                cand_img = self.generate_image(cand, seed, lora_endpoint)
                if cand_img is None:
                    continue
                cand_score = self.compute_clip_score(prompt, cand_img)
                improvement = cand_score - best_score
                logger.info(f"   grid result: {cand_score:.4f} (Δ{improvement:+.4f})")
                if improvement > 0:
                    best_prompt = cand
                    best_score = cand_score
                    best_image = cand_img
                    # If we reached target ratio improvement, stop early
                    if original_score > 0 and (best_score - original_score) >= self.min_improvement_ratio * original_score:
                        logger.info(f"   🎯 Target improvement reached (≥{self.min_improvement_ratio*100:.0f}% of original)")
                        break
        
        final_improvement = best_score - original_score
        return OptimizationResult(
            iteration=len(strategies),
            original_prompt=prompt,
            optimized_prompt=best_prompt,
            original_score=original_score,
            optimized_score=best_score,
            improvement=final_improvement,
            lora_endpoint=lora_endpoint,
            strategy_used="multi_strategy_feedback_loop"
        )
    
    def _blend_prompts(self, original: str, interrogated: str, lora_endpoint: str) -> str:
        """Intelligently blend original prompt with interrogated details"""
        # Simple blend strategy - can be enhanced with LLM
        original_words = set(original.lower().split())
        interrogated_words = interrogated.lower().split()
        
        # Keep original core concepts, add new descriptive elements
        new_elements = [word for word in interrogated_words 
                       if word not in original_words and len(word) > 3]
        
        # Add LoRA-specific style hints
        lora_hints = self._get_lora_style_hints(lora_endpoint)
        
        # Construct blended prompt
        blended = f"{original}"
        if new_elements:
            blended += f", {', '.join(new_elements[:5])}"  # Top 5 new elements
        if lora_hints:
            blended += f", {lora_hints}"
            
        return blended
    
    def _enhance_semantically(self, prompt: str, lora_endpoint: str) -> str:
        """Add semantic keywords that improve CLIP alignment"""
        enhancements = {
            "3d": ["three dimensional", "rendered", "detailed", "realistic"],
            "material": ["textured", "surface details", "material properties"],
            "lighting": ["well lit", "professional lighting", "clear visibility"],
            "quality": ["high quality", "sharp", "detailed", "clear"]
        }
        
        # Select relevant enhancements
        selected = []
        for category, keywords in enhancements.items():
            if not any(kw in prompt.lower() for kw in keywords):
                selected.append(keywords[0])  # Add one from each missing category
        
        if selected:
            return f"{prompt}, {', '.join(selected)}"
        return prompt
    
    def _align_with_lora_style(self, prompt: str, lora_endpoint: str) -> str:
        """Align prompt with LoRA-specific style characteristics"""
        style_mappings = {
            "isometric_3d": "isometric view, clean geometry",
            "live_3d": "realistic 3D, lifelike details",
            "game_assets": "game asset style, clean topology",
            "patched_realism": "realistic textures, detailed surfaces",
            "tf2_style": "Team Fortress 2 style, cartoon aesthetic",
            "baolei": "clean product render, isolated on white background, centered",
            "cartoon_3d": "cartoon style, vibrant colors",
            "cinema": "cinematic quality, professional rendering",
            "sd15_game_icon": "icon style, clear symbolism"
        }
        
        style_hint = style_mappings.get(lora_endpoint, "")
        if style_hint and style_hint not in prompt:
            return f"{prompt}, {style_hint}"
        return prompt
    
    def _get_lora_style_hints(self, lora_endpoint: str) -> str:
        """Get style hints for specific LoRA endpoint"""
        return self._align_with_lora_style("", lora_endpoint).strip(", ")
    
    def find_optimal_lora_for_prompt(self, prompt: str, seed: int = 42) -> Tuple[str, float]:
        """Find the LoRA endpoint that gives the best CLIP score for this prompt"""
        logger.info(f"🔍 Finding optimal LoRA for: '{prompt[:50]}...'")
        
        best_lora = "isometric_3d"
        best_score = 0.0
        
        for lora in self.lora_endpoints:
            try:
                image = self.generate_image(prompt, seed, lora)
                if image is None:
                    continue
                
                score = self.compute_clip_score(prompt, image)
                logger.info(f"   {lora}: {score:.4f}")
                
                if score > best_score:
                    best_score = score
                    best_lora = lora
                    
            except Exception as e:
                logger.error(f"   {lora}: Failed - {e}")
                continue
        
        logger.info(f"   🏆 Best LoRA: {best_lora} ({best_score:.4f})")
        return best_lora, best_score
    
    async def optimize_prompt_comprehensive(self, 
                                          prompt: str, 
                                          seed: int = 42,
                                          find_optimal_lora: bool = True) -> OptimizationSession:
        """Comprehensive prompt optimization across strategies and LoRA endpoints"""
        
        session_id = f"opt_{int(time.time())}_{hash(prompt) % 10000}"
        start_time = time.time()
        
        logger.info(f"🚀 Starting comprehensive optimization session {session_id}")
        logger.info(f"   Original prompt: '{prompt}'")
        
        # Load CLIP model
        self.load_clip_model()
        
        try:
            # Step 1: Find optimal LoRA endpoint if requested
            if find_optimal_lora:
                optimal_lora, baseline_score = self.find_optimal_lora_for_prompt(prompt, seed)
            else:
                optimal_lora = "isometric_3d"
                # Get baseline score
                image = self.generate_image(prompt, seed, optimal_lora)
                baseline_score = self.compute_clip_score(prompt, image) if image else 0.0
            
            # Step 2: Optimize for the best LoRA endpoint
            optimization_result = self.optimize_for_lora_endpoint(prompt, optimal_lora, seed)
            
            # Step 3: Create session result
            session = OptimizationSession(
                session_id=session_id,
                original_prompt=prompt,
                final_prompt=optimization_result.optimized_prompt,
                original_score=baseline_score,
                final_score=optimization_result.optimized_score,
                total_improvement=optimization_result.optimized_score - baseline_score,
                iterations=[optimization_result],
                best_iteration=0,
                total_time=time.time() - start_time
            )
            
            # Log results
            normalized_final = session.final_score / 0.35
            logger.info(f"✅ Optimization session {session_id} completed")
            logger.info(f"   Original score: {session.original_score:.4f}")
            logger.info(f"   Final score: {session.final_score:.4f} (normalized: {normalized_final:.4f})")
            logger.info(f"   Improvement: {session.total_improvement:+.4f}")
            logger.info(f"   Final prompt: '{session.final_prompt}'")
            logger.info(f"   Time: {session.total_time:.1f}s")
            
            return session
            
        finally:
            # Always unload CLIP model to free memory
            self.unload_clip_model()


# Example usage and testing
async def test_optimization():
    """Test the optimization system"""
    optimizer = CLIPAlignmentOptimizer()
    
    test_prompts = [
        "a blue ceramic vase",
        "red sports car",
        "wooden chair with metal legs",
        "glass sphere on marble table"
    ]
    
    for prompt in test_prompts:
        session = await optimizer.optimize_prompt_comprehensive(prompt)
        print(f"\nTest result for '{prompt}':")
        print(f"  Improvement: {session.total_improvement:+.4f}")
        print(f"  Final normalized score: {session.final_score/0.35:.4f}")


if __name__ == "__main__":
    asyncio.run(test_optimization()) 