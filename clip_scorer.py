#!/usr/bin/env python3
"""
CLIP Scorer for Image-Text Similarity
=====================================
A standalone module for calculating CLIP scores between generated images and text prompts.
Based on the existing CLIP implementations in the repository.
"""

import torch
import open_clip
import numpy as np
import time
import logging
from typing import Optional, Tuple, Dict, Any
from PIL import Image
import io
import base64
import requests

class CLIPScorer:
    """CLIP scorer for calculating image-text similarity scores"""
    
    def __init__(self, 
                 clip_model_name: str = "ViT-B-32", 
                 clip_pretrained: str = "openai",
                 device: Optional[str] = None):
        """
        Initialize CLIP scorer
        
        Args:
            clip_model_name: CLIP model variant to use
            clip_pretrained: Pretrained weights to use
            device: Device to run on (auto-detect if None)
        """
        self.clip_model_name = clip_model_name
        self.clip_pretrained = clip_pretrained
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        
        # Model components
        self.model = None
        self.preprocess = None
        self.tokenizer = None
        self._model_loaded = False
        
        self.logger = logging.getLogger(__name__)
        self.logger.info(f"🖼️ CLIP Scorer initialized (model: {clip_model_name}, device: {self.device})")
    
    def load_clip_model(self) -> bool:
        """Load CLIP model and preprocessing components"""
        if self._model_loaded:
            return True
            
        try:
            self.logger.info(f"🔄 Loading CLIP model: {self.clip_model_name} ({self.clip_pretrained})")
            load_start = time.time()
            
            # Load model and preprocessing
            self.model, _, self.preprocess = open_clip.create_model_and_transforms(
                self.clip_model_name, 
                pretrained=self.clip_pretrained,
                device=self.device
            )
            
            # Get tokenizer
            self.tokenizer = open_clip.get_tokenizer(self.clip_model_name)
            
            # Set to eval mode
            self.model.eval()
            
            load_time = time.time() - load_start
            self._model_loaded = True
            self.logger.info(f"✅ CLIP model loaded in {load_time:.2f}s")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Failed to load CLIP model: {e}")
            return False
    
    def compute_clip_score(self, prompt: str, image_base64: str) -> float:
        """
        Compute CLIP score between prompt and base64-encoded image
        
        Args:
            prompt: Text prompt to compare against
            image_base64: Base64-encoded image data
            
        Returns:
            CLIP similarity score between 0.0 and 1.0
        """
        if not self._model_loaded:
            if not self.load_clip_model():
                return 0.0
        
        try:
            # Decode base64 image
            image_data = base64.b64decode(image_base64)
            image = Image.open(io.BytesIO(image_data))
            
            # Convert to RGB if needed
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
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
            self.logger.error(f"❌ CLIP scoring failed: {e}")
            return 0.0
    
    def compute_clip_score_from_image(self, prompt: str, image: Image.Image) -> float:
        """
        Compute CLIP score between prompt and PIL Image object
        
        Args:
            prompt: Text prompt to compare against
            image: PIL Image object
            
        Returns:
            CLIP similarity score between 0.0 and 1.0
        """
        if not self._model_loaded:
            if not self.load_clip_model():
                return 0.0
        
        try:
            # Convert to RGB if needed
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
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
            self.logger.error(f"❌ CLIP scoring failed: {e}")
            return 0.0
    
    def compute_clip_score_from_url(self, prompt: str, image_url: str) -> float:
        """
        Compute CLIP score between prompt and image from URL
        
        Args:
            prompt: Text prompt to compare against
            image_url: URL to fetch image from
            
        Returns:
            CLIP similarity score between 0.0 and 1.0
        """
        try:
            # Fetch image from URL
            response = requests.get(image_url, timeout=30)
            response.raise_for_status()
            
            # Convert to PIL Image
            image = Image.open(io.BytesIO(response.content))
            
            return self.compute_clip_score_from_image(prompt, image)
            
        except Exception as e:
            self.logger.error(f"❌ Failed to fetch/process image from URL: {e}")
            return 0.0
    
    def compute_text_to_text_similarity(self, text1: str, text2: str) -> float:
        """
        Compute CLIP text-to-text similarity between two prompts
        
        Args:
            text1: First text prompt
            text2: Second text prompt
            
        Returns:
            CLIP text similarity score between 0.0 and 1.0
        """
        if not self._model_loaded:
            if not self.load_clip_model():
                return 0.0
        
        try:
            # Tokenize both texts
            text1_tokens = self.tokenizer([text1]).to(self.device)
            text2_tokens = self.tokenizer([text2]).to(self.device)
            
            with torch.no_grad():
                # Encode both texts
                text1_features = self.model.encode_text(text1_tokens)
                text2_features = self.model.encode_text(text2_tokens)
                
                # Normalize features
                text1_features /= text1_features.norm(dim=-1, keepdim=True)
                text2_features /= text2_features.norm(dim=-1, keepdim=True)
                
                # Compute similarity
                similarity = (text1_features @ text2_features.T).cpu().numpy()[0][0]
                
                # Clip to [0, 1] range
                similarity = np.clip(similarity, 0, 1)
                
                return float(similarity)
                
        except Exception as e:
            self.logger.error(f"❌ Text-to-text CLIP scoring failed: {e}")
            return 0.0
    
    def batch_compute_clip_scores(self, prompts: list, images_base64: list) -> list:
        """
        Compute CLIP scores for multiple prompt-image pairs
        
        Args:
            prompts: List of text prompts
            images_base64: List of base64-encoded images
            
        Returns:
            List of CLIP similarity scores
        """
        if len(prompts) != len(images_base64):
            raise ValueError("Number of prompts must match number of images")
        
        scores = []
        for prompt, image_base64 in zip(prompts, images_base64):
            score = self.compute_clip_score(prompt, image_base64)
            scores.append(score)
        
        return scores
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the loaded CLIP model"""
        return {
            "model_name": self.clip_model_name,
            "pretrained": self.clip_pretrained,
            "device": str(self.device),
            "model_loaded": self._model_loaded,
            "model_parameters": sum(p.numel() for p in self.model.parameters()) if self.model else 0
        }
    
    def unload_model(self):
        """Unload CLIP model to free memory"""
        if self._model_loaded:
            self.model = None
            self.preprocess = None
            self.tokenizer = None
            self._model_loaded = False
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
            self.logger.info("🗑️ CLIP model unloaded")


# Global CLIP scorer instance for reuse
_global_clip_scorer = None

def get_clip_scorer(clip_model_name: str = "ViT-B-32", 
                   clip_pretrained: str = "openai",
                   device: Optional[str] = None,
                   force_cpu: bool = False) -> CLIPScorer:
    """
    Get or create a global CLIP scorer instance
    
    Args:
        clip_model_name: CLIP model variant to use
        clip_pretrained: Pretrained weights to use
        device: Device to run on (auto-detect if None)
        force_cpu: Force CPU usage even if GPU is available
        
    Returns:
        CLIPScorer instance
    """
    global _global_clip_scorer
    
    # Determine device
    if force_cpu:
        final_device = "cpu"
    elif device is not None:
        final_device = device
    else:
        final_device = "cuda" if torch.cuda.is_available() else "cpu"
    
    if _global_clip_scorer is None:
        _global_clip_scorer = CLIPScorer(
            clip_model_name=clip_model_name,
            clip_pretrained=clip_pretrained,
            device=final_device
        )
    
    return _global_clip_scorer

def compute_clip_score_standalone(prompt: str, image_base64: str, 
                                clip_model_name: str = "ViT-B-32",
                                clip_pretrained: str = "openai") -> float:
    """
    Standalone function to compute CLIP score without managing model lifecycle
    
    Args:
        prompt: Text prompt to compare against
        image_base64: Base64-encoded image data
        clip_model_name: CLIP model variant to use
        clip_pretrained: Pretrained weights to use
        
    Returns:
        CLIP similarity score between 0.0 and 1.0
    """
    scorer = get_clip_scorer(clip_model_name, clip_pretrained)
    return scorer.compute_clip_score(prompt, image_base64)

def unload_global_clip_scorer():
    """Unload the global CLIP scorer to free memory"""
    global _global_clip_scorer
    if _global_clip_scorer is not None:
        _global_clip_scorer.unload_model()
        _global_clip_scorer = None
