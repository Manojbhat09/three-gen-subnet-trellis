#!/usr/bin/env python3
"""
CLIP Alignment Score with Real Image Generation
Purpose: Generate images using Hunyuan server and compute actual CLIP alignment scores
Uses: Hunyuan server for image generation + CLIP for alignment scoring

Usage:
    python clip_alignment_with_generation.py "a blue vase"
    python clip_alignment_with_generation.py --prefix "professional 3D render, " "a blue vase"
    python clip_alignment_with_generation.py --suffix ", highly detailed, photorealistic" "a blue vase"
    python clip_alignment_with_generation.py --optimized "a blue ceramic vase with red trim" "a blue vase"
"""

import argparse
import sys
import requests
import base64
import io
import time
import torch
import torch.nn.functional as F
import open_clip
from open_clip import CLIP
from open_clip.tokenizer import HFTokenizer
from torchvision import transforms
from PIL import Image
import numpy as np
from typing import Optional, Tuple


class CLIPAlignmentWithGeneration:
    """CLIP alignment scorer with real image generation capabilities"""
    
    # Available LoRA endpoints
    AVAILABLE_LORAS = [
        "isometric_3d",
        "live_3d", 
        "game_assets",
        "patched_realism",
        "tf2_style",
        "baolei",
        "cartoon_3d",
        "cinema",
        "sd15_game_icon"
    ]
    
    def __init__(self, hunyuan_server_url: str = "http://localhost:8096", verbose: bool = False, lora1_endpoint: str = None, lora2_endpoint: str = None):
        self.hunyuan_server_url = hunyuan_server_url
        self.verbose = verbose
        self.lora1_endpoint = lora1_endpoint
        self.lora2_endpoint = lora2_endpoint
        
        # CLIP model settings (same as production validation)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_name = "convnext_large_d"
        self.pretrained = "laion2b_s26b_b102k_augreg"
        
        # Model components
        self._model: Optional[CLIP] = None
        self._tokenizer: Optional[HFTokenizer] = None
        
        # Normalization transform (same as production)
        mean = torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(1, 3, 1, 1) * 3
        std = torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(1, 3, 1, 1) * 3
        self._normalize_transform = transforms.Normalize(mean, std)
        
        print(f"🔧 CLIPAlignmentWithGeneration initialized")
        print(f"   Hunyuan Server: {hunyuan_server_url}")
        print(f"   CLIP Model: {self.model_name}/{self.pretrained}")
        print(f"   Device: {self.device}")
        if lora1_endpoint:
            print(f"   LoRA 1 Endpoint: {lora1_endpoint}")
        if lora2_endpoint:
            print(f"   LoRA 2 Endpoint: {lora2_endpoint}")
    
    def select_lora_interactive(self, prompt: str = "Select LoRA endpoint") -> str:
        """Interactively select a LoRA endpoint"""
        print(f"\n🎨 {prompt}")
        print("Available LoRA endpoints:")
        print("  0. none (use default endpoint)")
        
        for i, lora in enumerate(self.AVAILABLE_LORAS, 1):
            print(f"  {i}. {lora}")
        
        while True:
            try:
                choice = input(f"\nEnter your choice (0-{len(self.AVAILABLE_LORAS)}): ").strip()
                choice_num = int(choice)
                
                if choice_num == 0:
                    return "none"
                elif 1 <= choice_num <= len(self.AVAILABLE_LORAS):
                    selected_lora = self.AVAILABLE_LORAS[choice_num - 1]
                    print(f"✅ Selected: {selected_lora}")
                    return selected_lora
                else:
                    print(f"❌ Invalid choice. Please enter a number between 0 and {len(self.AVAILABLE_LORAS)}")
            except ValueError:
                print("❌ Invalid input. Please enter a number.")
            except KeyboardInterrupt:
                print("\n⚠️ Selection cancelled. Using default endpoint.")
                return "none"
    
    def load_clip_model(self) -> None:
        """Load the CLIP model for alignment scoring"""
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
    
    def unload_clip_model(self) -> None:
        """Unload the CLIP model to free memory"""
        if self._model is not None:
            print("🧹 Unloading CLIP model...")
            del self._model
            del self._tokenizer
            self._model = None
            self._tokenizer = None
            torch.cuda.empty_cache()
            print("✅ CLIP model unloaded")
    
    def check_hunyuan_server(self) -> bool:
        """Check if Hunyuan server is running and accessible"""
        try:
            response = requests.get(f"{self.hunyuan_server_url}/health/", timeout=5)
            if response.status_code == 200:
                print("✅ Hunyuan server is running")
                return True
            else:
                print(f"⚠️ Hunyuan server responded with status {response.status_code}")
                return False
        except requests.exceptions.RequestException as e:
            print(f"❌ Cannot connect to Hunyuan server: {e}")
            return False
    
    def generate_image(self, prompt: str, seed: int = 42, use_lora1: bool = False, use_lora2: bool = False) -> Optional[Image.Image]:
        """Generate image using Hunyuan server with optional LoRA endpoints"""
        try:
            print(f"🎨 Generating image for: '{prompt}' (seed: {seed})")
            
            # Determine endpoint based on LoRA selection
            if use_lora1 and self.lora1_endpoint and self.lora1_endpoint != "none":
                endpoint = f"/generate_image/{self.lora1_endpoint}/"
                print(f"   Using LoRA 1 endpoint: {endpoint}")
            elif use_lora2 and self.lora2_endpoint and self.lora2_endpoint != "none":
                endpoint = f"/generate_image/{self.lora2_endpoint}/"
                print(f"   Using LoRA 2 endpoint: {endpoint}")
            else:
                endpoint = "/generate_image/isometric_3d/"
                print(f"   Using default endpoint: {endpoint}")
            
            # Make request to Hunyuan server
            response = requests.post(
                f"{self.hunyuan_server_url}{endpoint}",
                data={'prompt': prompt, 'seed': seed},
                timeout=300
            )
            
            if response.status_code == 200:
                result = response.json()
                if result.get('status') == 'success':
                    image_data = base64.b64decode(result['image'])
                    image = Image.open(io.BytesIO(image_data))
                    print(f"✅ Image generated successfully ({result['image_size_bytes']:,} bytes)")
                    return image
                else:
                    print(f"❌ Image generation failed: {result.get('error', 'Unknown error')}")
                    return None
            else:
                print(f"❌ Server request failed: {response.status_code}")
                return None
        except Exception as e:
            print(f"❌ Image generation failed: {e}")
            return None

    def generate_hunyuan_image(self, prompt: str, seed: int = 42) -> Optional[Image.Image]:
        """Generate image using HunyuanDiT (no LoRA)"""
        try:
            print(f"🎨 Generating HunyuanDiT image for: '{prompt}' (seed: {seed})")
            endpoint = "/generate_image_hunyuan/"
            response = requests.post(
                f"{self.hunyuan_server_url}{endpoint}",
                data={'prompt': prompt, 'seed': seed},
                timeout=300
            )
            if response.status_code == 200:
                result = response.json()
                if result.get('status') == 'success':
                    image_data = base64.b64decode(result['image'])
                    image = Image.open(io.BytesIO(image_data))
                    print(f"✅ HunyuanDiT image generated successfully ({result['image_size_bytes']:,} bytes)")
                    return image
                else:
                    print(f"❌ HunyuanDiT image generation failed: {result.get('error', 'Unknown error')}")
                    return None
            else:
                print(f"❌ HunyuanDiT server request failed: {response.status_code}")
                return None
        except Exception as e:
            print(f"❌ HunyuanDiT image generation failed: {e}")
            return None

    def analyze_all_loras(self, prompt: str, seed: int = 42) -> dict:
        """Analyze a prompt across all LoRA endpoints and HunyuanDiT and generate a table"""
        print(f"\n🔍 ANALYZING ACROSS ALL LoRA ENDPOINTS + HUNYUANDiT")
        print(f"=" * 60)
        print(f"Prompt: '{prompt}'")
        print(f"Seed: {seed}")
        print(f"Total LoRA endpoints: {len(self.AVAILABLE_LORAS)} + 1 (HunyuanDiT)")
        
        results = {}
        
        # Test with no LoRA (default endpoint) first
        print(f"\n🎨 Testing default endpoint (no LoRA)...")
        default_image = self.generate_image(prompt, seed, False, False)
        if default_image is not None:
            default_alignment = self.compute_clip_alignment_score(prompt, default_image)
            default_normalized = default_alignment / 0.35
            results["default"] = {
                "prompt": prompt,
                "alignment_score": default_alignment,
                "normalized_score": default_normalized,
                "status": self._get_status(default_normalized),
                "image_size": default_image.size
            }
            print(f"   ✅ Default: {default_alignment:.4f} ({default_normalized:.4f})")
        else:
            results["default"] = {"error": "Generation failed"}
            print(f"   ❌ Default: Generation failed")
        
        # Test each LoRA endpoint
        for i, lora in enumerate(self.AVAILABLE_LORAS, 1):
            print(f"\n🎨 Testing LoRA {i}/{len(self.AVAILABLE_LORAS)}: {lora}...")
            original_lora1 = self.lora1_endpoint
            self.lora1_endpoint = lora
            try:
                image = self.generate_image(prompt, seed, True, False)
                if image is not None:
                    alignment = self.compute_clip_alignment_score(prompt, image)
                    normalized = alignment / 0.35
                    results[lora] = {
                        "prompt": prompt,
                        "alignment_score": alignment,
                        "normalized_score": normalized,
                        "status": self._get_status(normalized),
                        "image_size": image.size
                    }
                    print(f"   ✅ {lora}: {alignment:.4f} ({normalized:.4f})")
                else:
                    results[lora] = {"error": "Generation failed"}
                    print(f"   ❌ {lora}: Generation failed")
            except Exception as e:
                results[lora] = {"error": str(e)}
                print(f"   ❌ {lora}: Error - {e}")
            finally:
                self.lora1_endpoint = original_lora1
        
        # Add HunyuanDiT as a separate row
        print(f"\n🎨 Testing HunyuanDiT (no LoRA)...")
        hunyuan_image = self.generate_hunyuan_image(prompt, seed)
        if hunyuan_image is not None:
            hunyuan_alignment = self.compute_clip_alignment_score(prompt, hunyuan_image)
            hunyuan_normalized = hunyuan_alignment / 0.35
            results["hunyuan"] = {
                "prompt": prompt,
                "alignment_score": hunyuan_alignment,
                "normalized_score": hunyuan_normalized,
                "status": self._get_status(hunyuan_normalized),
                "image_size": hunyuan_image.size
            }
            print(f"   ✅ HunyuanDiT: {hunyuan_alignment:.4f} ({hunyuan_normalized:.4f})")
        else:
            results["hunyuan"] = {"error": "Generation failed"}
            print(f"   ❌ HunyuanDiT: Generation failed")
        
        return results
    
    def preprocess_image_for_clip(self, image: Image.Image, image_res: int = 224) -> torch.Tensor:
        """Preprocess image for CLIP model (same as production validation)"""
        # Convert PIL to tensor
        image_tensor = torch.tensor(np.array(image)).float()
        
        # Normalize to [0, 1]
        image_tensor = image_tensor / 255.0
        
        # Convert to channels-first format
        if len(image_tensor.shape) == 3:
            image_tensor = image_tensor.permute(2, 0, 1)
        
        # Add batch dimension
        image_tensor = image_tensor.unsqueeze(0)
        
        # Resize to CLIP input size
        image_tensor = F.interpolate(image_tensor, size=(image_res, image_res), mode="bicubic", align_corners=False)
        
        # Apply CLIP normalization
        image_tensor = self._normalize_transform(image_tensor)
        
        return image_tensor.to(self.device)
    
    def compute_text_to_text_similarity(self, text1: str, text2: str) -> float:
        """Compute CLIP text-to-text similarity between two prompts"""
        if self._model is None or self._tokenizer is None:
            raise RuntimeError("CLIP model not loaded. Call load_clip_model() first.")
        
        try:
            # Tokenize both texts
            tokenized_text1 = self._tokenizer(text1).to(self.device)
            tokenized_text2 = self._tokenizer(text2).to(self.device)
            
            with torch.no_grad(), torch.amp.autocast(self.device.type):
                # Encode both texts
                text1_features = self._model.encode_text(tokenized_text1)
                text2_features = self._model.encode_text(tokenized_text2)
                
                # Normalize features
                text1_features /= text1_features.norm(dim=-1, keepdim=True)
                text2_features /= text2_features.norm(dim=-1, keepdim=True)
                
                # Compute similarity
                similarity = (text1_features @ text2_features.T).cpu().numpy()[0][0]
                
                # Clip to [0, 1] range
                similarity = np.clip(similarity, 0, 1)
                
                return float(similarity)
                
        except Exception as e:
            print(f"❌ CLIP text-to-text similarity computation failed: {e}")
            return 0.0

    def compute_clip_alignment_score(self, prompt: str, image: Image.Image) -> float:
        """Compute CLIP alignment score between prompt and image"""
        if self._model is None or self._tokenizer is None:
            raise RuntimeError("CLIP model not loaded. Call load_clip_model() first.")
        
        try:
            # Preprocess image
            image_tensor = self.preprocess_image_for_clip(image)
            
            # Tokenize prompt
            tokenized_prompt = self._tokenizer(prompt).to(self.device)
            
            with torch.no_grad(), torch.amp.autocast(self.device.type):
                # Encode image and text
                image_features = self._model.encode_image(image_tensor)
                text_features = self._model.encode_text(tokenized_prompt)
                
                # Normalize features
                image_features /= image_features.norm(dim=-1, keepdim=True)
                text_features /= text_features.norm(dim=-1, keepdim=True)
                
                # Compute similarity
                similarity = (image_features @ text_features.T).cpu().numpy()[0][0]
                
                # Clip to [0, 1] range
                similarity = np.clip(similarity, 0, 1)
                
                return float(similarity)
                
        except Exception as e:
            print(f"❌ CLIP alignment computation failed: {e}")
            return 0.0
    
    def analyze_single_prompt(self, prompt: str, seed: int = 42, use_lora1: bool = False, use_lora2: bool = False) -> dict:
        """Analyze a single prompt by generating image and computing alignment"""
        print(f"\n🔍 ANALYZING SINGLE PROMPT")
        print(f"=" * 50)
        print(f"Prompt: '{prompt}'")
        print(f"Seed: {seed}")
        if use_lora1:
            print(f"LoRA 1: {self.lora1_endpoint}")
        if use_lora2:
            print(f"LoRA 2: {self.lora2_endpoint}")
        
        # Generate image
        image = self.generate_image(prompt, seed, use_lora1, use_lora2)
        if image is None:
            return {"error": "Image generation failed"}
        
        # Compute alignment score
        alignment_score = self.compute_clip_alignment_score(prompt, image)
        
        # Apply production normalization
        normalized_score = alignment_score / 0.35
        
        # Determine validation status
        if normalized_score < 0.3:
            status = "❌ FAIL"
            task_fidelity = 0.0
        elif normalized_score >= 0.8:
            status = "✅ EXCELLENT"
            task_fidelity = 1.0
        elif normalized_score >= 0.6:
            status = "🟡 GOOD"
            task_fidelity = 0.75
        else:
            status = "🟠 POOR"
            task_fidelity = 0.0
        
        results = {
            "prompt": prompt,
            "seed": seed,
            "alignment_score": alignment_score,
            "normalized_score": normalized_score,
            "status": status,
            "task_fidelity": task_fidelity,
            "image_size": image.size
        }
        
        print(f"\n📊 RESULTS:")
        print(f"   Raw Alignment Score: {alignment_score:.4f}")
        print(f"   Normalized Score: {normalized_score:.4f}")
        print(f"   Status: {status}")
        print(f"   Task Fidelity: {task_fidelity}")
        print(f"   Image Size: {image.size}")
        
        return results
    
    def analyze_with_prefix(self, prompt: str, prefix: str, seed: int = 42, use_lora1: bool = False, use_lora2: bool = False) -> dict:
        """Analyze prompt with and without prefix"""
        print(f"\n🔍 ANALYZING WITH PREFIX")
        print(f"=" * 50)
        print(f"Original Prompt: '{prompt}'")
        print(f"Prefix: '{prefix}'")
        print(f"Prefixed Prompt: '{prefix + prompt}'")
        print(f"Seed: {seed}")
        if use_lora1:
            print(f"LoRA 1: {self.lora1_endpoint}")
        if use_lora2:
            print(f"LoRA 2: {self.lora2_endpoint}")
        
        # Generate image with original prompt
        print(f"\n🎨 Generating image with original prompt...")
        original_image = self.generate_image(prompt, seed, use_lora1, False)
        if original_image is None:
            return {"error": "Original image generation failed"}
        
        # Generate image with prefixed prompt
        print(f"\n🎨 Generating image with prefixed prompt...")
        prefixed_image = self.generate_image(prefix + prompt, seed, False, use_lora2)
        if prefixed_image is None:
            return {"error": "Prefixed image generation failed"}
        
        # Compute alignment scores
        original_alignment = self.compute_clip_alignment_score(prompt, original_image)
        prefixed_alignment = self.compute_clip_alignment_score(prefix + prompt, prefixed_image)
        
        # Apply production normalization
        original_normalized = original_alignment / 0.35
        prefixed_normalized = prefixed_alignment / 0.35
        
        # Determine statuses
        def get_status(score):
            if score < 0.3:
                return "❌ FAIL"
            elif score >= 0.8:
                return "✅ EXCELLENT"
            elif score >= 0.6:
                return "🟡 GOOD"
            else:
                return "🟠 POOR"
        
        results = {
            "original": {
                "prompt": prompt,
                "alignment_score": original_alignment,
                "normalized_score": original_normalized,
                "status": get_status(original_normalized)
            },
            "prefixed": {
                "prompt": prefix + prompt,
                "alignment_score": prefixed_alignment,
                "normalized_score": prefixed_normalized,
                "status": get_status(prefixed_normalized)
            },
            "improvement": prefixed_normalized - original_normalized
        }
        
        print(f"\n📊 RESULTS:")
        print(f"   Original Prompt: '{prompt}'")
        print(f"     Alignment Score: {original_alignment:.4f}")
        print(f"     Normalized Score: {original_normalized:.4f}")
        print(f"     Status: {results['original']['status']}")
        
        print(f"\n   Prefixed Prompt: '{prefix + prompt}'")
        print(f"     Alignment Score: {prefixed_alignment:.4f}")
        print(f"     Normalized Score: {prefixed_normalized:.4f}")
        print(f"     Status: {results['prefixed']['status']}")
        
        print(f"\n   Improvement: {results['improvement']:+.4f}")
        if results['improvement'] > 0:
            print(f"   ✅ Prefix improved alignment score")
        elif results['improvement'] < 0:
            print(f"   ❌ Prefix decreased alignment score")
        else:
            print(f"   ➖ Prefix had no effect")
        
        return results
    
    def analyze_with_suffix(self, prompt: str, suffix: str, seed: int = 42, use_lora1: bool = False, use_lora2: bool = False) -> dict:
        """Analyze prompt with and without suffix"""
        print(f"\n🔍 ANALYZING WITH SUFFIX")
        print(f"=" * 50)
        print(f"Original Prompt: '{prompt}'")
        print(f"Suffix: '{suffix}'")
        print(f"Suffixed Prompt: '{prompt + suffix}'")
        print(f"Seed: {seed}")
        if use_lora1:
            print(f"LoRA 1: {self.lora1_endpoint}")
        if use_lora2:
            print(f"LoRA 2: {self.lora2_endpoint}")
        
        # Generate image with original prompt
        print(f"\n🎨 Generating image with original prompt...")
        original_image = self.generate_image(prompt, seed, use_lora1, False)
        if original_image is None:
            return {"error": "Original image generation failed"}
        
        # Generate image with suffixed prompt
        print(f"\n🎨 Generating image with suffixed prompt...")
        suffixed_image = self.generate_image(prompt + suffix, seed, False, use_lora2)
        if suffixed_image is None:
            return {"error": "Suffixed image generation failed"}
        
        # Compute alignment scores
        original_alignment = self.compute_clip_alignment_score(prompt, original_image)
        suffixed_alignment = self.compute_clip_alignment_score(prompt + suffix, suffixed_image)
        
        # Apply production normalization
        original_normalized = original_alignment / 0.35
        suffixed_normalized = suffixed_alignment / 0.35
        
        # Determine statuses
        def get_status(score):
            if score < 0.3:
                return "❌ FAIL"
            elif score >= 0.8:
                return "✅ EXCELLENT"
            elif score >= 0.6:
                return "🟡 GOOD"
            else:
                return "🟠 POOR"
        
        results = {
            "original": {
                "prompt": prompt,
                "alignment_score": original_alignment,
                "normalized_score": original_normalized,
                "status": get_status(original_normalized)
            },
            "suffixed": {
                "prompt": prompt + suffix,
                "alignment_score": suffixed_alignment,
                "normalized_score": suffixed_normalized,
                "status": get_status(suffixed_normalized)
            },
            "improvement": suffixed_normalized - original_normalized
        }
        
        print(f"\n📊 RESULTS:")
        print(f"   Original Prompt: '{prompt}'")
        print(f"     Alignment Score: {original_alignment:.4f}")
        print(f"     Normalized Score: {original_normalized:.4f}")
        print(f"     Status: {results['original']['status']}")
        
        print(f"\n   Suffixed Prompt: '{prompt + suffix}'")
        print(f"     Alignment Score: {suffixed_alignment:.4f}")
        print(f"     Normalized Score: {suffixed_normalized:.4f}")
        print(f"     Status: {results['suffixed']['status']}")
        
        print(f"\n   Improvement: {results['improvement']:+.4f}")
        if results['improvement'] > 0:
            print(f"   ✅ Suffix improved alignment score")
        elif results['improvement'] < 0:
            print(f"   ❌ Suffix decreased alignment score")
        else:
            print(f"   ➖ Suffix had no effect")
        
        return results
    
    def analyze_optimization(self, original_prompt: str, optimized_prompt: str, seed: int = 42, use_lora1: bool = False, use_lora2: bool = False) -> dict:
        """Compare original prompt with optimized prompt"""
        print(f"\n🔍 ANALYZING PROMPT OPTIMIZATION")
        print(f"=" * 50)
        print(f"Original Prompt: '{original_prompt}'")
        print(f"Optimized Prompt: '{optimized_prompt}'")
        print(f"Seed: {seed}")
        if use_lora1:
            print(f"LoRA 1: {self.lora1_endpoint}")
        if use_lora2:
            print(f"LoRA 2: {self.lora2_endpoint}")
        
        # Generate image with original prompt
        print(f"\n🎨 Generating image with original prompt...")
        original_image = self.generate_image(original_prompt, seed, use_lora1, False)
        if original_image is None:
            return {"error": "Original image generation failed"}
        
        # Generate image with optimized prompt
        print(f"\n🎨 Generating image with optimized prompt...")
        optimized_image = self.generate_image(optimized_prompt, seed, False, use_lora2)
        if optimized_image is None:
            return {"error": "Optimized image generation failed"}
        
        # Compute alignment scores (using original prompt as reference for both)
        original_alignment = self.compute_clip_alignment_score(original_prompt, original_image)
        optimized_alignment = self.compute_clip_alignment_score(original_prompt, optimized_image)
        
        # Apply production normalization
        original_normalized = original_alignment / 0.35
        optimized_normalized = optimized_alignment / 0.35
        
        # Determine statuses
        def get_status(score):
            if score < 0.3:
                return "❌ FAIL"
            elif score >= 0.8:
                return "✅ EXCELLENT"
            elif score >= 0.6:
                return "🟡 GOOD"
            else:
                return "🟠 POOR"
        
        results = {
            "original": {
                "prompt": original_prompt,
                "alignment_score": original_alignment,
                "normalized_score": original_normalized,
                "status": get_status(original_normalized)
            },
            "optimized": {
                "prompt": optimized_prompt,
                "alignment_score": optimized_alignment,
                "normalized_score": optimized_normalized,
                "status": get_status(optimized_normalized)
            },
            "improvement": optimized_normalized - original_normalized
        }
        
        print(f"\n📊 RESULTS:")
        print(f"   Original Generation: '{original_prompt}'")
        print(f"     Alignment Score: {original_alignment:.4f}")
        print(f"     Normalized Score: {original_normalized:.4f}")
        print(f"     Status: {results['original']['status']}")
        
        print(f"\n   Optimized Generation: '{optimized_prompt}'")
        print(f"     Alignment Score: {optimized_alignment:.4f}")
        print(f"     Normalized Score: {optimized_normalized:.4f}")
        print(f"     Status: {results['optimized']['status']}")
        
        print(f"\n   Improvement: {results['improvement']:+.4f}")
        if results['improvement'] > 0:
            print(f"   ✅ Optimization improved alignment score")
        elif results['improvement'] < 0:
            print(f"   ❌ Optimization decreased alignment score")
        else:
            print(f"   ➖ Optimization had no effect")
        
        return results

    def analyze_confusion_matrix(self, prompt: str, optimized_prompt: str, seed: int = 42, use_lora1: bool = False, use_lora2: bool = False) -> dict:
        """Analyze all four alignment combinations between prompts and images"""
        print(f"\n🔍 ANALYZING CONFUSION MATRIX")
        print(f"=" * 50)
        print(f"Original Prompt: '{prompt}'")
        print(f"Optimized Prompt: '{optimized_prompt}'")
        print(f"Seed: {seed}")
        if use_lora1:
            print(f"LoRA 1: {self.lora1_endpoint}")
        if use_lora2:
            print(f"LoRA 2: {self.lora2_endpoint}")
        
        # Generate both images
        print(f"\n🎨 Generating image with original prompt...")
        original_image = self.generate_image(prompt, seed, use_lora1, False)
        if original_image is None:
            return {"error": "Original image generation failed"}
        
        print(f"\n🎨 Generating image with optimized prompt...")
        optimized_image = self.generate_image(optimized_prompt, seed, False, use_lora2)
        if optimized_image is None:
            return {"error": "Optimized image generation failed"}
        
        # Compute all four alignment scores
        print(f"\n📊 Computing alignment scores...")
        
        # 1. Original prompt vs Original image
        original_vs_original = self.compute_clip_alignment_score(prompt, original_image)
        original_vs_original_norm = original_vs_original / 0.35
        
        # 2. Original prompt vs Optimized image
        original_vs_optimized = self.compute_clip_alignment_score(prompt, optimized_image)
        original_vs_optimized_norm = original_vs_optimized / 0.35
        
        # 3. Optimized prompt vs Original image
        optimized_vs_original = self.compute_clip_alignment_score(optimized_prompt, original_image)
        optimized_vs_original_norm = optimized_vs_original / 0.35
        
        # 4. Optimized prompt vs Optimized image
        optimized_vs_optimized = self.compute_clip_alignment_score(optimized_prompt, optimized_image)
        optimized_vs_optimized_norm = optimized_vs_optimized / 0.35
        
        # Determine statuses
        def get_status(score):
            if score < 0.3:
                return "❌ FAIL"
            elif score >= 0.8:
                return "✅ EXCELLENT"
            elif score >= 0.6:
                return "🟡 GOOD"
            else:
                return "🟠 POOR"
        
        results = {
            "original_prompt_vs_original_image": {
                "prompt": prompt,
                "image": "original",
                "alignment_score": original_vs_original,
                "normalized_score": original_vs_original_norm,
                "status": get_status(original_vs_original_norm)
            },
            "original_prompt_vs_optimized_image": {
                "prompt": prompt,
                "image": "optimized",
                "alignment_score": original_vs_optimized,
                "normalized_score": original_vs_optimized_norm,
                "status": get_status(original_vs_optimized_norm)
            },
            "optimized_prompt_vs_original_image": {
                "prompt": optimized_prompt,
                "image": "original",
                "alignment_score": optimized_vs_original,
                "normalized_score": optimized_vs_original_norm,
                "status": get_status(optimized_vs_original_norm)
            },
            "optimized_prompt_vs_optimized_image": {
                "prompt": optimized_prompt,
                "image": "optimized",
                "alignment_score": optimized_vs_optimized,
                "normalized_score": optimized_vs_optimized_norm,
                "status": get_status(optimized_vs_optimized_norm)
            }
        }
        
        # Print confusion matrix
        print(f"\n📊 CONFUSION MATRIX RESULTS:")
        print(f"=" * 80)
        print(f"{'Prompt':<30} {'Image':<15} {'Raw Score':<12} {'Norm Score':<12} {'Status':<15}")
        print(f"{'-' * 30} {'-' * 15} {'-' * 12} {'-' * 12} {'-' * 15}")
        
        for key, result in results.items():
            prompt_text = result["prompt"][:27] + "..." if len(result["prompt"]) > 30 else result["prompt"]
            image_type = result["image"]
            raw_score = result["alignment_score"]
            norm_score = result["normalized_score"]
            status = result["status"]
            
            print(f"{prompt_text:<30} {image_type:<15} {raw_score:<12.4f} {norm_score:<12.4f} {status:<15}")
        
        # Analysis insights
        print(f"\n🔍 ANALYSIS INSIGHTS:")
        print(f"=" * 50)
        
        # Best and worst combinations
        all_scores = [(k, v["normalized_score"]) for k, v in results.items()]
        best_combination = max(all_scores, key=lambda x: x[1])
        worst_combination = min(all_scores, key=lambda x: x[1])
        
        print(f"🏆 Best combination: {best_combination[0]} (Score: {best_combination[1]:.4f})")
        print(f"❌ Worst combination: {worst_combination[0]} (Score: {worst_combination[1]:.4f})")
        
        # Prompt effectiveness
        original_prompt_avg = (original_vs_original_norm + original_vs_optimized_norm) / 2
        optimized_prompt_avg = (optimized_vs_original_norm + optimized_vs_optimized_norm) / 2
        
        print(f"\n📝 PROMPT EFFECTIVENESS:")
        print(f"   Original prompt average: {original_prompt_avg:.4f}")
        print(f"   Optimized prompt average: {optimized_prompt_avg:.4f}")
        print(f"   Prompt improvement: {optimized_prompt_avg - original_prompt_avg:+.4f}")
        
        # Image effectiveness
        original_image_avg = (original_vs_original_norm + optimized_vs_original_norm) / 2
        optimized_image_avg = (original_vs_optimized_norm + optimized_vs_optimized_norm) / 2
        
        print(f"\n🖼️ IMAGE EFFECTIVENESS:")
        print(f"   Original image average: {original_image_avg:.4f}")
        print(f"   Optimized image average: {optimized_image_avg:.4f}")
        print(f"   Image improvement: {optimized_image_avg - original_image_avg:+.4f}")
        
        # Overall optimization effectiveness
        diagonal_improvement = (optimized_vs_optimized_norm - original_vs_original_norm)
        print(f"\n🎯 OVERALL OPTIMIZATION:")
        print(f"   Diagonal improvement: {diagonal_improvement:+.4f}")
        if diagonal_improvement > 0:
            print(f"   ✅ Optimization improved overall alignment")
        elif diagonal_improvement < 0:
            print(f"   ❌ Optimization decreased overall alignment")
        else:
            print(f"   ➖ Optimization had no effect")
        
        return results

    def analyze_prompt_self_alignment(self, prompt: str, optimized_prompt: str) -> dict:
        """Compute prompt-to-prompt alignment (semantic similarity)"""
        print(f"\n🔗 PROMPT-TO-PROMPT ALIGNMENT (Semantic Similarity)")
        print(f"=" * 60)
        print(f"Original Prompt: '{prompt}'")
        print(f"Optimized Prompt: '{optimized_prompt}'")
        
        # Compute prompt-to-prompt alignment (semantic similarity)
        print(f"\n📊 Computing prompt-to-prompt alignment...")
        prompt_alignment = self.compute_text_to_text_similarity(prompt, optimized_prompt)
        prompt_alignment_norm = prompt_alignment / 0.35
        
        # Determine status
        def get_status(score):
            if score < 0.3:
                return "❌ FAIL"
            elif score >= 0.8:
                return "✅ EXCELLENT"
            elif score >= 0.6:
                return "🟡 GOOD"
            else:
                return "🟠 POOR"
        
        status = get_status(prompt_alignment_norm)
        
        # Print results
        print(f"\n📊 RESULTS:")
        print(f"   Raw Alignment Score: {prompt_alignment:.4f}")
        print(f"   Normalized Score: {prompt_alignment_norm:.4f}")
        print(f"   Status: {status}")
        
        # Semantic similarity analysis
        print(f"\n🔍 SEMANTIC SIMILARITY ANALYSIS:")
        if prompt_alignment_norm >= 0.8:
            print(f"   ✅ Excellent semantic alignment - prompts are very similar")
        elif prompt_alignment_norm >= 0.6:
            print(f"   🟡 Good semantic alignment - prompts maintain core meaning")
        elif prompt_alignment_norm >= 0.4:
            print(f"   🟠 Moderate semantic alignment - some meaning preserved")
        else:
            print(f"   ❌ Poor semantic alignment - prompts may be too different")
        
        results = {
            "prompt_to_prompt": {
                "type": "semantic_similarity",
                "original_prompt": prompt,
                "optimized_prompt": optimized_prompt,
                "alignment_score": prompt_alignment,
                "normalized_score": prompt_alignment_norm,
                "status": status
            }
        }
        
        return results

    def compare_prompts_across_all_loras(self, prompt1: str, prompt2: str, seed: int = 42) -> dict:
        """Compare two prompts across all LoRA endpoints and HunyuanDiT and generate a comparison table"""
        print(f"\n🔍 COMPARING TWO PROMPTS ACROSS ALL LoRA ENDPOINTS + HUNYUANDiT")
        print(f"=" * 70)
        print(f"Prompt 1: '{prompt1}'")
        print(f"Prompt 2: '{prompt2}'")
        print(f"Seed: {seed}")
        print(f"Total LoRA endpoints: {len(self.AVAILABLE_LORAS)} + 1 (HunyuanDiT)")
        
        results = {}
        
        # Test with no LoRA (default endpoint) first
        print(f"\n🎨 Testing default endpoint (no LoRA)...")
        default_image1 = self.generate_image(prompt1, seed, False, False)
        default_image2 = self.generate_image(prompt2, seed, False, False)
        
        if default_image1 is not None and default_image2 is not None:
            default_alignment1 = self.compute_clip_alignment_score(prompt1, default_image1)
            default_alignment2 = self.compute_clip_alignment_score(prompt1, default_image2)  # Use prompt1 as reference
            default_normalized1 = default_alignment1 / 0.35
            default_normalized2 = default_alignment2 / 0.35
            
            results["default"] = {
                "prompt1": {
                    "prompt": prompt1,
                    "alignment_score": default_alignment1,
                    "normalized_score": default_normalized1,
                    "status": self._get_status(default_normalized1)
                },
                "prompt2": {
                    "prompt": prompt2,
                    "alignment_score": default_alignment2,
                    "normalized_score": default_normalized2,
                    "status": self._get_status(default_normalized2)
                },
                "improvement": default_normalized2 - default_normalized1
            }
            print(f"   ✅ Default: P1={default_alignment1:.4f} P2={default_alignment2:.4f} (Δ={default_normalized2-default_normalized1:+.4f})")
        else:
            results["default"] = {"error": "Generation failed"}
            print(f"   ❌ Default: Generation failed")
        
        # Test each LoRA endpoint
        for i, lora in enumerate(self.AVAILABLE_LORAS, 1):
            print(f"\n🎨 Testing LoRA {i}/{len(self.AVAILABLE_LORAS)}: {lora}...")
            original_lora1 = self.lora1_endpoint
            self.lora1_endpoint = lora
            try:
                image1 = self.generate_image(prompt1, seed, True, False)
                image2 = self.generate_image(prompt2, seed, True, False)
                
                if image1 is not None and image2 is not None:
                    alignment1 = self.compute_clip_alignment_score(prompt1, image1)
                    alignment2 = self.compute_clip_alignment_score(prompt1, image2)  # Use prompt1 as reference
                    normalized1 = alignment1 / 0.35
                    normalized2 = alignment2 / 0.35
                    
                    results[lora] = {
                        "prompt1": {
                            "prompt": prompt1,
                            "alignment_score": alignment1,
                            "normalized_score": normalized1,
                            "status": self._get_status(normalized1)
                        },
                        "prompt2": {
                            "prompt": prompt2,
                            "alignment_score": alignment2,
                            "normalized_score": normalized2,
                            "status": self._get_status(normalized2)
                        },
                        "improvement": normalized2 - normalized1
                    }
                    print(f"   ✅ {lora}: P1={alignment1:.4f} P2={alignment2:.4f} (Δ={normalized2-normalized1:+.4f})")
                else:
                    results[lora] = {"error": "Generation failed"}
                    print(f"   ❌ {lora}: Generation failed")
            except Exception as e:
                results[lora] = {"error": str(e)}
                print(f"   ❌ {lora}: Error - {e}")
            finally:
                self.lora1_endpoint = original_lora1
        
        # Add HunyuanDiT as a separate row
        print(f"\n🎨 Testing HunyuanDiT (no LoRA)...")
        hunyuan_image1 = self.generate_hunyuan_image(prompt1, seed)
        hunyuan_image2 = self.generate_hunyuan_image(prompt2, seed)
        if hunyuan_image1 is not None and hunyuan_image2 is not None:
            hunyuan_alignment1 = self.compute_clip_alignment_score(prompt1, hunyuan_image1)
            hunyuan_alignment2 = self.compute_clip_alignment_score(prompt1, hunyuan_image2)
            hunyuan_normalized1 = hunyuan_alignment1 / 0.35
            hunyuan_normalized2 = hunyuan_alignment2 / 0.35
            results["hunyuan"] = {
                "prompt1": {
                    "prompt": prompt1,
                    "alignment_score": hunyuan_alignment1,
                    "normalized_score": hunyuan_normalized1,
                    "status": self._get_status(hunyuan_normalized1)
                },
                "prompt2": {
                    "prompt": prompt2,
                    "alignment_score": hunyuan_alignment2,
                    "normalized_score": hunyuan_normalized2,
                    "status": self._get_status(hunyuan_normalized2)
                },
                "improvement": hunyuan_normalized2 - hunyuan_normalized1
            }
            print(f"   ✅ HunyuanDiT: P1={hunyuan_alignment1:.4f} P2={hunyuan_alignment2:.4f} (Δ={hunyuan_normalized2-hunyuan_normalized1:+.4f})")
        else:
            results["hunyuan"] = {"error": "Generation failed"}
            print(f"   ❌ HunyuanDiT: Generation failed")
        
        return results

    def _get_status(self, score: float) -> str:
        """Get status string for a normalized score"""
        if score < 0.3:
            return "❌ FAIL"
        elif score >= 0.8:
            return "✅ EXCELLENT"
        elif score >= 0.6:
            return "🟡 GOOD"
        else:
            return "🟠 POOR"
    
    def print_lora_comparison_table(self, results: dict) -> None:
        """Print a formatted table comparing results across all LoRA endpoints"""
        print(f"\n📊 LoRA ENDPOINT COMPARISON TABLE")
        print(f"=" * 80)
        
        # Table header
        print(f"{'LoRA Endpoint':<20} {'Raw Score':<12} {'Normalized':<12} {'Status':<12} {'Image Size':<15}")
        print(f"{'-' * 20} {'-' * 12} {'-' * 12} {'-' * 12} {'-' * 15}")
        
        # Sort results by normalized score (descending)
        sorted_results = sorted(
            [(k, v) for k, v in results.items() if "error" not in v],
            key=lambda x: x[1]["normalized_score"],
            reverse=True
        )
        
        # Print successful results
        for lora, result in sorted_results:
            print(f"{lora:<20} {result['alignment_score']:<12.4f} {result['normalized_score']:<12.4f} {result['status']:<12} {str(result['image_size']):<15}")
        
        # Print failed results
        failed_results = [(k, v) for k, v in results.items() if "error" in v]
        if failed_results:
            print(f"\n❌ FAILED GENERATIONS:")
            for lora, result in failed_results:
                print(f"   {lora}: {result['error']}")
        
        # Summary statistics
        successful_results = [v for v in results.values() if "error" not in v]
        if successful_results:
            scores = [r["normalized_score"] for r in successful_results]
            print(f"\n📈 SUMMARY STATISTICS:")
            print(f"   Total endpoints tested: {len(results)}")
            print(f"   Successful generations: {len(successful_results)}")
            print(f"   Failed generations: {len(failed_results)}")
            print(f"   Best score: {max(scores):.4f}")
            print(f"   Worst score: {min(scores):.4f}")
            print(f"   Average score: {sum(scores)/len(scores):.4f}")
            
            # Top 3 performers
            top_3 = sorted_results[:3]
            print(f"\n🏆 TOP 3 PERFORMERS:")
            for i, (lora, result) in enumerate(top_3, 1):
                print(f"   {i}. {lora}: {result['normalized_score']:.4f} ({result['status']})")
    
    def print_prompt_comparison_table(self, results: dict, prompt1: str, prompt2: str) -> None:
        """Print a formatted table comparing two prompts across all LoRA endpoints"""
        print(f"\n📊 PROMPT COMPARISON TABLE")
        print(f"=" * 100)
        print(f"Prompt 1: '{prompt1}'")
        print(f"Prompt 2: '{prompt2}' (optimized version)")
        print(f"=" * 100)
        
        # Table header
        print(f"{'LoRA Endpoint':<20} {'P1 Score':<12} {'P2 Score':<12} {'Improvement':<12} {'Best':<8} {'Status':<15}")
        print(f"{'-' * 20} {'-' * 12} {'-' * 12} {'-' * 12} {'-' * 8} {'-' * 15}")
        
        # Sort results by improvement (descending)
        sorted_results = sorted(
            [(k, v) for k, v in results.items() if "error" not in v],
            key=lambda x: x[1]["improvement"],
            reverse=True
        )
        
        # Print successful results
        for lora, result in sorted_results:
            p1_score = result["prompt1"]["normalized_score"]
            p2_score = result["prompt2"]["normalized_score"]
            improvement = result["improvement"]
            best = "P2" if improvement > 0 else "P1" if improvement < 0 else "TIE"
            status = "✅ BETTER" if improvement > 0 else "❌ WORSE" if improvement < 0 else "➖ SAME"
            
            print(f"{lora:<20} {p1_score:<12.4f} {p2_score:<12.4f} {improvement:<+12.4f} {best:<8} {status:<15}")
        
        # Print failed results
        failed_results = [(k, v) for k, v in results.items() if "error" in v]
        if failed_results:
            print(f"\n❌ FAILED GENERATIONS:")
            for lora, result in failed_results:
                print(f"   {lora}: {result['error']}")
        
        # Summary statistics
        successful_results = [v for v in results.values() if "error" not in v]
        if successful_results:
            improvements = [r["improvement"] for r in successful_results]
            p1_scores = [r["prompt1"]["normalized_score"] for r in successful_results]
            p2_scores = [r["prompt2"]["normalized_score"] for r in successful_results]
            
            print(f"\n📈 SUMMARY STATISTICS:")
            print(f"   Total endpoints tested: {len(results)}")
            print(f"   Successful generations: {len(successful_results)}")
            print(f"   Failed generations: {len(failed_results)}")
            print(f"   Average P1 score: {sum(p1_scores)/len(p1_scores):.4f}")
            print(f"   Average P2 score: {sum(p2_scores)/len(p2_scores):.4f}")
            print(f"   Average improvement: {sum(improvements)/len(improvements):+.4f}")
            print(f"   Best improvement: {max(improvements):+.4f}")
            print(f"   Worst improvement: {min(improvements):+.4f}")
            
            # Count improvements
            better_count = sum(1 for imp in improvements if imp > 0)
            worse_count = sum(1 for imp in improvements if imp < 0)
            same_count = sum(1 for imp in improvements if imp == 0)
            
            print(f"\n🏆 IMPROVEMENT BREAKDOWN:")
            print(f"   Better with P2: {better_count}/{len(successful_results)} ({better_count/len(successful_results)*100:.1f}%)")
            print(f"   Worse with P2: {worse_count}/{len(successful_results)} ({worse_count/len(successful_results)*100:.1f}%)")
            print(f"   Same performance: {same_count}/{len(successful_results)} ({same_count/len(successful_results)*100:.1f}%)")
            
            # Top 3 improvements
            top_3 = sorted_results[:3]
            print(f"\n🏆 TOP 3 IMPROVEMENTS:")
            for i, (lora, result) in enumerate(top_3, 1):
                print(f"   {i}. {lora}: {result['improvement']:+.4f} (P1: {result['prompt1']['normalized_score']:.4f} → P2: {result['prompt2']['normalized_score']:.4f})")


def main():
    parser = argparse.ArgumentParser(
        description="CLIP Alignment Score with Real Image Generation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single prompt analysis
  python clip_alignment_with_generation.py "a blue vase"
  
  # Test across all LoRA endpoints
  python clip_alignment_with_generation.py --all-loras "a blue vase"
  
  # Compare two prompts across all LoRA endpoints
  python clip_alignment_with_generation.py --compare-all-loras "a blue vase" --optimized "a blue ceramic vase with red trim"
  
  # Prefix analysis
  python clip_alignment_with_generation.py --prefix "professional 3D render, " "a blue vase"
  
  # Suffix analysis  
  python clip_alignment_with_generation.py --suffix ", highly detailed, photorealistic" "a blue vase"
  
  # Optimization analysis
  python clip_alignment_with_generation.py --optimized "a blue ceramic vase with red trim" "a blue vase"
  
  # Confusion matrix analysis
  python clip_alignment_with_generation.py --confusion-mat --optimized "a blue ceramic vase with red trim" "a blue vase"
  
  # Prompt self-alignment analysis
  python clip_alignment_with_generation.py --prompt-self --optimized "a blue ceramic vase with red trim" "a blue vase"
  
  # Using LoRA endpoints
  python clip_alignment_with_generation.py --lora1 "live_3d" "a blue vase"
  python clip_alignment_with_generation.py --lora2 "game_assets" "a blue vase"
  python clip_alignment_with_generation.py --lora1 "isometric_3d" --lora2 "live_3d" "a blue vase"
  
  # Using LoRA numbers (0=none, 1=isometric_3d, 2=live_3d, etc.)
  python clip_alignment_with_generation.py -1 2 "a blue vase"
  python clip_alignment_with_generation.py -2 3 "a blue vase"
  python clip_alignment_with_generation.py -1 1 -2 2 "a blue vase"
        """
    )
    
    parser.add_argument("prompt", help="Base prompt for analysis")
    parser.add_argument("--prefix", help="Prefix to add to prompt")
    parser.add_argument("--suffix", help="Suffix to add to prompt")
    parser.add_argument("--optimized", help="Optimized prompt to compare against")
    parser.add_argument("--all-loras", action="store_true", help="Test prompt across all LoRA endpoints and generate comparison table")
    parser.add_argument("--compare-all-loras", action="store_true", help="Compare two prompts across all LoRA endpoints and generate comparison table")
    parser.add_argument("--confusion-mat", action="store_true", help="Analyze confusion matrix between original/optimized prompts and images")
    parser.add_argument("--prompt-self", action="store_true", help="Analyze prompt-to-prompt alignment and comprehensive image results")
    parser.add_argument("--lora1", help="LoRA 1 endpoint (e.g., isometric_3d, live_3d, game_assets, etc.)")
    parser.add_argument("--lora2", help="LoRA 2 endpoint (e.g., isometric_3d, live_3d, game_assets, etc.)")
    parser.add_argument("-1", "--lora1_num", type=int, help="LoRA 1 by number (0=none, 1=isometric_3d, 2=live_3d, etc.)")
    parser.add_argument("-2", "--lora2_num", type=int, help="LoRA 2 by number (0=none, 1=isometric_3d, 2=live_3d, etc.)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for generation")
    parser.add_argument("--server", default="http://localhost:8096", help="Hunyuan server URL")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    
    args = parser.parse_args()
    
    # Initialize analyzer
    analyzer = CLIPAlignmentWithGeneration(
        hunyuan_server_url=args.server,
        verbose=args.verbose,
        lora1_endpoint=args.lora1,
        lora2_endpoint=args.lora2
    )
    
    try:
        # Check server connectivity
        if not analyzer.check_hunyuan_server():
            print("❌ Cannot connect to Hunyuan server. Please ensure it's running.")
            return 1
        
        # Load CLIP model
        analyzer.load_clip_model()

        # Always compute prompt-to-prompt similarity if --prompt-self is given
        if args.prompt_self and args.optimized:
            print(f"\n🔗 COMPUTING PROMPT-TO-PROMPT SIMILARITY FIRST")
            print(f"=" * 60)
            prompt_similarity_results = analyzer.analyze_prompt_self_alignment(args.prompt, args.optimized)
            print(f"\n" + "=" * 60)
        
        # Handle LoRA selection (number-based or string-based)
        def get_lora_from_number(num: int) -> str:
            if num == 0:
                return "none"
            elif 1 <= num <= len(analyzer.AVAILABLE_LORAS):
                return analyzer.AVAILABLE_LORAS[num - 1]
            else:
                print(f"❌ Invalid LoRA number: {num}. Valid range: 0-{len(analyzer.AVAILABLE_LORAS)}")
                return "none"
        
        # Process LoRA arguments
        if args.lora1_num is not None:
            analyzer.lora1_endpoint = get_lora_from_number(args.lora1_num)
        if args.lora2_num is not None:
            analyzer.lora2_endpoint = get_lora_from_number(args.lora2_num)
        
        # Interactive LoRA selection if not provided (skip for --all-loras and --compare-all-loras)
        if not args.all_loras and not args.compare_all_loras and args.lora1 is None and args.lora2 is None and args.lora1_num is None and args.lora2_num is None:
            print("\n🎨 No LoRA endpoints specified. Please select one:")
            selected_lora = analyzer.select_lora_interactive("Select LoRA endpoint for analysis")
            if selected_lora != "none":
                analyzer.lora1_endpoint = selected_lora
                use_lora1 = True
                use_lora2 = False
            else:
                use_lora1 = False
                use_lora2 = False
                
            # For comparative analysis, ask about second LoRA
            if args.prefix or args.suffix or args.optimized:
                print(f"\n🎨 For the second generation, do you want to:")
                print("  1. Use the same LoRA endpoint")
                print("  2. Use a different LoRA endpoint")
                print("  3. Use no LoRA (default endpoint)")
                
                while True:
                    try:
                        choice = input("Enter your choice (1-3): ").strip()
                        choice_num = int(choice)
                        
                        if choice_num == 1:
                            # Use same LoRA
                            analyzer.lora2_endpoint = analyzer.lora1_endpoint
                            use_lora2 = use_lora1
                            print(f"✅ Using same LoRA: {analyzer.lora1_endpoint}")
                            break
                        elif choice_num == 2:
                            # Use different LoRA
                            selected_lora2 = analyzer.select_lora_interactive("Select LoRA endpoint for second generation")
                            if selected_lora2 != "none":
                                analyzer.lora2_endpoint = selected_lora2
                                use_lora2 = True
                                print(f"✅ Using different LoRA: {analyzer.lora2_endpoint}")
                            else:
                                use_lora2 = False
                                print("✅ Using default endpoint for second generation")
                            break
                        elif choice_num == 3:
                            # Use no LoRA for second generation
                            use_lora2 = False
                            print("✅ Using default endpoint for second generation")
                            break
                        else:
                            print("❌ Invalid choice. Please enter 1, 2, or 3.")
                    except ValueError:
                        print("❌ Invalid input. Please enter a number.")
                    except KeyboardInterrupt:
                        print("\n⚠️ Selection cancelled. Using same LoRA for both generations.")
                        analyzer.lora2_endpoint = analyzer.lora1_endpoint
                        use_lora2 = use_lora1
                        break
        else:
            # Use provided LoRA arguments
            use_lora1 = (args.lora1 is not None or args.lora1_num is not None) and analyzer.lora1_endpoint != "none"
            use_lora2 = (args.lora2 is not None or args.lora2_num is not None) and analyzer.lora2_endpoint != "none"
        
        
        
        # Perform analysis based on arguments
        if args.compare_all_loras:
            # Compare two prompts across all LoRA endpoints
            if not args.optimized:
                print("❌ Error: --compare-all-loras requires --optimized argument")
                return 1
            results = analyzer.compare_prompts_across_all_loras(args.prompt, args.optimized, args.seed)
            analyzer.print_prompt_comparison_table(results, args.prompt, args.optimized)
        elif args.all_loras:
            # Test across all LoRA endpoints
            results = analyzer.analyze_all_loras(args.prompt, args.seed)
            analyzer.print_lora_comparison_table(results)
        elif args.confusion_mat and args.optimized:
            # Confusion matrix analysis
            results = analyzer.analyze_confusion_matrix(args.prompt, args.optimized, args.seed, use_lora1, use_lora2)
        elif args.optimized:
            # Optimization analysis
            results = analyzer.analyze_optimization(args.prompt, args.optimized, args.seed, use_lora1, use_lora2)
        elif args.prefix:
            # Prefix analysis
            results = analyzer.analyze_with_prefix(args.prompt, args.prefix, args.seed, use_lora1, use_lora2)
        elif args.suffix:
            # Suffix analysis
            results = analyzer.analyze_with_suffix(args.prompt, args.suffix, args.seed, use_lora1, use_lora2)
        else:
            # Single prompt analysis
            results = analyzer.analyze_single_prompt(args.prompt, args.seed, use_lora1, use_lora2)
        
        if "error" in results:
            print(f"❌ Analysis failed: {results['error']}")
            return 1
        
        print(f"\n🎯 ANALYSIS COMPLETE")
        print(f"=" * 50)
        
        return 0
        
    except KeyboardInterrupt:
        print("\n⚠️ Interrupted by user")
        return 1
    except Exception as e:
        print(f"❌ Error: {e}")
        return 1
    finally:
        # Cleanup
        analyzer.unload_clip_model()


if __name__ == "__main__":
    sys.exit(main()) 