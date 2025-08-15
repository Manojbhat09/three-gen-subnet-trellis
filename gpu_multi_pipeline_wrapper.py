#!/usr/bin/env python3
"""
Multi-GPU Pipeline Wrapper - Subnet 17 (404-GEN)
Purpose: Complete pipeline for image generation, CLIP scoring, PLY generation, and validation ranking

Features:
1. Parallel image generation across 8 GPUs with CLIP scoring and ranking
2. Best images → PLY generation pipeline across all GPUs  
3. Single high-scoring image → multiple PLY variations across GPUs
4. Comprehensive validation scoring and ranking system
5. Performance analysis and GPU utilization optimization

Workflow:
Text Prompt → [8x Image Generation] → CLIP Ranking → [Best Images → 8x PLY] → Validation Ranking
Text Prompt → [Best Image → 8x PLY Variations] → Validation Ranking
"""

import os
import sys
import time
import json
import asyncio
import requests
import subprocess
import threading
import signal
import argparse
import logging
import base64
import io
import traceback
from pathlib import Path
from typing import List, Dict, Optional, Any, Tuple
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np
from PIL import Image

# Import GPU server management components
from gpu_server_wrapper import GPUServerManager, GPUServer

# Import CLIP utilities
try:
    import open_clip
    import torch
    import torch.nn.functional as F
    from torchvision import transforms
    CLIP_AVAILABLE = True
    print("✅ CLIP utilities available")
except ImportError as e:
    print(f"❌ CLIP utilities not available: {e}")
    CLIP_AVAILABLE = False

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('gpu_multi_pipeline_wrapper.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

@dataclass
class ImageGenerationResult:
    """Results from image generation on a single GPU"""
    gpu_id: int
    port: int
    prompt: str
    success: bool
    image_data: Optional[bytes] = None
    image_b64: Optional[str] = None
    generation_time: float = 0.0
    seed: Optional[int] = None
    error: Optional[str] = None
    clip_score: float = 0.0
    
    @property
    def pil_image(self) -> Optional[Image.Image]:
        """Convert image data to PIL Image"""
        if self.image_data:
            return Image.open(io.BytesIO(self.image_data)).convert('RGB')
        elif self.image_b64:
            return Image.open(io.BytesIO(base64.b64decode(self.image_b64))).convert('RGB')
        return None

@dataclass
class PLYGenerationResult:
    """Results from PLY generation on a single GPU"""
    gpu_id: int
    port: int
    prompt: str
    source_image_gpu: Optional[int]  # Which GPU generated the source image
    success: bool
    ply_data: Optional[bytes] = None
    generation_time: float = 0.0
    ply_size: int = 0
    compression_ratio: Optional[str] = None
    error: Optional[str] = None
    validation_score: float = 0.0
    alignment_score: float = 0.0
    quality_score: float = 0.0
    demo_fidelity_score: float = 0.0

@dataclass  
class PipelineResults:
    """Complete pipeline results"""
    prompt: str
    pipeline_type: str  # "image_ranking_to_ply" or "single_image_multi_ply"
    image_results: List[ImageGenerationResult] = field(default_factory=list)
    ply_results: List[PLYGenerationResult] = field(default_factory=list)
    best_image_gpu: Optional[int] = None
    best_ply_gpu: Optional[int] = None
    best_clip_score: float = 0.0
    best_validation_score: float = 0.0
    total_pipeline_time: float = 0.0
    
class CLIPScorer:
    """CLIP-based scoring utilities"""
    
    def __init__(self, device: str = "cuda"):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.model = None
        self.tokenizer = None
        self.normalize = None
        self._load_clip_model()
    
    def _load_clip_model(self):
        """Load the validator CLIP model"""
        if not CLIP_AVAILABLE:
            logger.error("❌ CLIP not available - scoring disabled")
            return
            
        try:
            logger.info(f"🔄 Loading CLIP model on {self.device}")
            self.model, _, _ = open_clip.create_model_and_transforms(
                "convnext_large_d", 
                pretrained="laion2b_s26b_b102k_augreg", 
                device=self.device
            )
            self.tokenizer = open_clip.get_tokenizer("convnext_large_d")
            self.model.eval()
            
            # Setup normalization (same as validator)
            mean = torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(1, 3, 1, 1) * 3
            std = torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(1, 3, 1, 1) * 3
            self.normalize = transforms.Normalize(mean, std)
            
            logger.info("✅ CLIP model loaded successfully")
            
        except Exception as e:
            logger.error(f"❌ Failed to load CLIP model: {e}")
            self.model = None
    
    def encode_text(self, text: str) -> torch.Tensor:
        """Encode text to CLIP features"""
        if self.model is None:
            return torch.zeros(1, 768).to(self.device)
            
        tokens = self.tokenizer(text).to(self.device)
        with torch.no_grad(), torch.amp.autocast(self.device.type):
            feats = self.model.encode_text(tokens)
            feats = feats / feats.norm(dim=-1, keepdim=True)
        return feats
    
    def encode_image(self, img: Image.Image, res: int = 224) -> torch.Tensor:
        """Encode image to CLIP features"""
        if self.model is None:
            return torch.zeros(1, 768).to(self.device)
            
        t = torch.tensor(np.array(img)).float() / 255.0
        if t.ndim == 3:
            t = t.permute(2, 0, 1)
        t = t.unsqueeze(0).to(self.device)
        t = F.interpolate(t, size=(res, res), mode="bicubic", align_corners=False)
        t = self.normalize(t)
        
        with torch.no_grad(), torch.amp.autocast(self.device.type):
            feats = self.model.encode_image(t)
            feats = feats / feats.norm(dim=-1, keepdim=True)
        return feats
    
    def compute_text_image_similarity(self, text: str, img: Image.Image) -> float:
        """Compute CLIP similarity between text and image"""
        if self.model is None:
            return 0.0
            
        try:
            tf = self.encode_text(text)
            vf = self.encode_image(img)
            sim = (vf @ tf.T).float().cpu().numpy()[0][0]
            return float(np.clip(sim, 0, 1))
        except Exception as e:
            logger.error(f"❌ CLIP scoring failed: {e}")
            return 0.0

class MultiGPUPipelineManager(GPUServerManager):
    """Extended GPU server manager with pipeline capabilities"""
    
    def __init__(self, num_gpus: int = 8, base_port: int = 8096, 
                 server_script: str = "trellis_subnit_server_mix_lora_flash.py",
                 output_dir: str = "./gpu_pipeline_outputs"):
        super().__init__(num_gpus, base_port, server_script, output_dir)
        
        # Initialize CLIP scorer
        self.clip_scorer = CLIPScorer()
        
        # Pipeline statistics
        self.pipeline_stats = {
            'image_generations': 0,
            'successful_image_generations': 0,
            'failed_image_generations': 0,
            'ply_generations': 0,
            'successful_ply_generations': 0,
            'failed_ply_generations': 0,
            'validations': 0,
            'successful_validations': 0,
            'failed_validations': 0,
            'total_pipeline_time': 0.0,
            'best_clip_scores': [],
            'best_validation_scores': []
        }
        
        logger.info("🚀 Multi-GPU Pipeline Manager initialized")
        logger.info(f"   CLIP Scoring: {'✅ Enabled' if CLIP_AVAILABLE else '❌ Disabled'}")
    
    def generate_image_on_gpu(self, gpu_id: int, prompt: str, seed: Optional[int] = None,
                             num_inference_steps: int = 25, guidance_scale: float = 7.5) -> ImageGenerationResult:
        """Generate image on a specific GPU"""
        gpu_server = self.gpu_servers[gpu_id]
        
        if seed is None:
            seed = np.random.randint(0, 2**31 - 1)
        
        try:
            logger.info(f"🎨 Generating image on GPU {gpu_id} for: '{prompt[:50]}...' (seed: {seed})")
            start_time = time.time()
            
            # Send image generation request
            response = requests.post(
                f"{gpu_server.url}/generate_image/",
                data={
                    'prompt': prompt,
                    'seed': seed,
                    'num_inference_steps': num_inference_steps,
                    'guidance_scale': guidance_scale
                },
                timeout=180  # 3 minutes timeout for image generation
            )
            
            generation_time = time.time() - start_time
            
            if response.status_code == 200:
                # Try to parse JSON response
                try:
                    response_data = response.json()
                    image_b64 = response_data.get('image') or response_data.get('image_base64')
                    
                    if image_b64:
                        # Decode image for CLIP scoring
                        image_data = base64.b64decode(image_b64)
                        pil_image = Image.open(io.BytesIO(image_data)).convert('RGB')
                        
                        # Compute CLIP score
                        clip_score = self.clip_scorer.compute_text_image_similarity(prompt, pil_image)
                        
                        result = ImageGenerationResult(
                            gpu_id=gpu_id,
                            port=gpu_server.port,
                            prompt=prompt,
                            success=True,
                            image_data=image_data,
                            image_b64=image_b64,
                            generation_time=generation_time,
                            seed=seed,
                            clip_score=clip_score
                        )
                        
                        logger.info(f"   ✅ GPU {gpu_id} image generated in {generation_time:.2f}s")
                        logger.info(f"      Image size: {len(image_data):,} bytes, CLIP score: {clip_score:.4f}")
                        
                        return result
                    else:
                        raise ValueError("No image data in response")
                        
                except Exception as e:
                    # Fallback: treat as raw image data
                    image_data = response.content
                    pil_image = Image.open(io.BytesIO(image_data)).convert('RGB')
                    clip_score = self.clip_scorer.compute_text_image_similarity(prompt, pil_image)
                    
                    result = ImageGenerationResult(
                        gpu_id=gpu_id,
                        port=gpu_server.port,
                        prompt=prompt,
                        success=True,
                        image_data=image_data,
                        generation_time=generation_time,
                        seed=seed,
                        clip_score=clip_score
                    )
                    
                    logger.info(f"   ✅ GPU {gpu_id} image generated in {generation_time:.2f}s (raw format)")
                    logger.info(f"      Image size: {len(image_data):,} bytes, CLIP score: {clip_score:.4f}")
                    
                    return result
            else:
                error_msg = f"HTTP {response.status_code}"
                logger.error(f"   ❌ GPU {gpu_id} image generation failed: {error_msg}")
                
                return ImageGenerationResult(
                    gpu_id=gpu_id,
                    port=gpu_server.port,
                    prompt=prompt,
                    success=False,
                    generation_time=generation_time,
                    seed=seed,
                    error=error_msg
                )
                
        except Exception as e:
            error_msg = str(e)
            logger.error(f"   ❌ GPU {gpu_id} image generation exception: {error_msg}")
            
            return ImageGenerationResult(
                gpu_id=gpu_id,
                port=gpu_server.port,
                prompt=prompt,
                success=False,
                generation_time=0.0,
                seed=seed,
                error=error_msg
            )
    
    def generate_images_parallel(self, prompt: str, seeds: Optional[List[int]] = None,
                                num_inference_steps: int = 25, guidance_scale: float = 7.5) -> List[ImageGenerationResult]:
        """Generate images on all GPUs in parallel"""
        logger.info(f"🎨 Generating images on all {self.num_gpus} GPUs for: '{prompt}'")
        
        if seeds is None:
            seeds = [np.random.randint(0, 2**31 - 1) for _ in range(self.num_gpus)]
        elif len(seeds) < self.num_gpus:
            # Extend seeds list if needed
            seeds.extend([np.random.randint(0, 2**31 - 1) for _ in range(self.num_gpus - len(seeds))])
        
        start_time = time.time()
        results = []
        
        # Generate images on all GPUs simultaneously
        with ThreadPoolExecutor(max_workers=self.num_gpus) as executor:
            future_to_gpu = {
                executor.submit(
                    self.generate_image_on_gpu, 
                    gpu_id, prompt, seeds[gpu_id], 
                    num_inference_steps, guidance_scale
                ): gpu_id 
                for gpu_id in range(self.num_gpus)
            }
            
            for future in as_completed(future_to_gpu):
                gpu_id = future_to_gpu[future]
                try:
                    result = future.result()
                    results.append(result)
                    
                    # Update statistics
                    self.pipeline_stats['image_generations'] += 1
                    if result.success:
                        self.pipeline_stats['successful_image_generations'] += 1
                    else:
                        self.pipeline_stats['failed_image_generations'] += 1
                        
                except Exception as e:
                    logger.error(f"❌ Exception in image generation on GPU {gpu_id}: {e}")
                    results.append(ImageGenerationResult(
                        gpu_id=gpu_id,
                        port=self.gpu_servers[gpu_id].port,
                        prompt=prompt,
                        success=False,
                        error=str(e)
                    ))
                    self.pipeline_stats['failed_image_generations'] += 1
        
        parallel_time = time.time() - start_time
        
        # Sort results by GPU ID for consistency
        results.sort(key=lambda x: x.gpu_id)
        
        # Analyze results
        successful_results = [r for r in results if r.success]
        failed_results = [r for r in results if not r.success]
        
        logger.info(f"✅ Parallel image generation complete in {parallel_time:.2f}s")
        logger.info(f"   Successful: {len(successful_results)}/{len(results)}")
        logger.info(f"   Failed: {len(failed_results)}/{len(results)}")
        
        if successful_results:
            # Rank by CLIP score
            ranked_results = sorted(successful_results, key=lambda x: x.clip_score, reverse=True)
            
            logger.info("🏆 Image CLIP Score Ranking (Best to Worst):")
            for i, result in enumerate(ranked_results):
                medal = "🥇" if i == 0 else "🥈" if i == 1 else "🥉" if i == 2 else f"#{i+1}"
                logger.info(f"   {medal} GPU {result.gpu_id} (port {result.port}): CLIP {result.clip_score:.4f}")
                logger.info(f"      Generation time: {result.generation_time:.2f}s, Seed: {result.seed}")
            
            # Store best scores
            self.pipeline_stats['best_clip_scores'].append(ranked_results[0].clip_score)
        
        return results
    
    def generate_ply_on_gpu(self, gpu_id: int, prompt: str, seed: Optional[int] = None,
                           source_image: Optional[Image.Image] = None,
                           source_image_gpu: Optional[int] = None) -> PLYGenerationResult:
        """Generate PLY on a specific GPU"""
        gpu_server = self.gpu_servers[gpu_id]
        
        if seed is None:
            seed = np.random.randint(0, 2**31 - 1)
        
        try:
            logger.info(f"🔮 Generating PLY on GPU {gpu_id} for: '{prompt[:50]}...' (seed: {seed})")
            start_time = time.time()
            
            # Prepare request data
            request_data = {
                'prompt': prompt,
                'seed': seed,
                'return_compressed': True
            }
            
            # Add image if provided (for image-to-3D generation)
            files = {}
            if source_image:
                img_buffer = io.BytesIO()
                source_image.save(img_buffer, format='PNG')
                img_buffer.seek(0)
                files['image'] = ('image.png', img_buffer, 'image/png')
                logger.info(f"   Using source image from GPU {source_image_gpu}")
            
            # Send PLY generation request
            if files:
                response = requests.post(
                    f"{gpu_server.url}/generate/",
                    data=request_data,
                    files=files,
                    timeout=300  # 5 minutes timeout for PLY generation
                )
            else:
                response = requests.post(
                    f"{gpu_server.url}/generate/",
                    data=request_data,
                    timeout=300
                )
            
            generation_time = time.time() - start_time
            
            if response.status_code == 200:
                ply_data = response.content
                compression_ratio = response.headers.get('X-Compression-Ratio', 'unknown')
                
                # Run validation on the PLY data
                validation_result = self._validate_ply_data(ply_data, prompt, gpu_id)
                
                result = PLYGenerationResult(
                    gpu_id=gpu_id,
                    port=gpu_server.port,
                    prompt=prompt,
                    source_image_gpu=source_image_gpu,
                    success=True,
                    ply_data=ply_data,
                    generation_time=generation_time,
                    ply_size=len(ply_data),
                    compression_ratio=compression_ratio,
                    validation_score=validation_result.get('validation_engine_score', 0.0),
                    alignment_score=validation_result.get('alignment_score', 0.0),
                    quality_score=validation_result.get('quality_score', 0.0),
                    demo_fidelity_score=validation_result.get('demo_fidelity_score', 0.0)
                )
                
                logger.info(f"   ✅ GPU {gpu_id} PLY generated in {generation_time:.2f}s")
                logger.info(f"      PLY size: {len(ply_data):,} bytes, Validation score: {result.validation_score:.4f}")
                
                return result
            else:
                error_msg = f"HTTP {response.status_code}"
                logger.error(f"   ❌ GPU {gpu_id} PLY generation failed: {error_msg}")
                
                return PLYGenerationResult(
                    gpu_id=gpu_id,
                    port=gpu_server.port,
                    prompt=prompt,
                    source_image_gpu=source_image_gpu,
                    success=False,
                    generation_time=generation_time,
                    error=error_msg
                )
                
        except Exception as e:
            error_msg = str(e)
            logger.error(f"   ❌ GPU {gpu_id} PLY generation exception: {error_msg}")
            
            return PLYGenerationResult(
                gpu_id=gpu_id,
                port=gpu_server.port,
                prompt=prompt,
                source_image_gpu=source_image_gpu,
                success=False,
                generation_time=0.0,
                error=error_msg
            )
    
    def _validate_ply_data(self, ply_data: bytes, prompt: str, gpu_id: int) -> Dict[str, Any]:
        """Validate PLY data using local validator"""
        try:
            logger.debug(f"🔍 Validating PLY data from GPU {gpu_id}")
            
            # Save PLY data temporarily
            temp_file = self.output_dir / f"temp_ply_gpu_{gpu_id}_{int(time.time())}.ply"
            with open(temp_file, 'wb') as f:
                f.write(ply_data)
            
            # Run validation using subnet_accurate_validator_multigpu.py
            cmd = [
                sys.executable,
                "subnet_accurate_validator_multigpu.py",
                prompt,
                prompt,
                "--endpoint", "generate/",
                "--port", str(self.gpu_servers[gpu_id].port)
            ]
            
            # Set GPU environment
            env = os.environ.copy()
            env['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
            
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=120,
                env=env,
                cwd=Path(__file__).parent
            )
            
            # Clean up temp file
            try:
                temp_file.unlink()
            except:
                pass
            
            if result.returncode == 0:
                # Read validation results
                results_file = f"subnet_validation_results_{self.gpu_servers[gpu_id].port}.json"
                if Path(results_file).exists():
                    with open(results_file, 'r') as f:
                        validation_data = json.load(f)
                    
                    self.pipeline_stats['validations'] += 1
                    self.pipeline_stats['successful_validations'] += 1
                    
                    return validation_data
                else:
                    logger.warning(f"   ⚠️ Validation results file not found for GPU {gpu_id}")
                    return {'validation_engine_score': 0.0}
            else:
                logger.error(f"   ❌ Validation failed for GPU {gpu_id}: {result.stderr}")
                self.pipeline_stats['validations'] += 1
                self.pipeline_stats['failed_validations'] += 1
                return {'validation_engine_score': 0.0, 'error': result.stderr}
                
        except Exception as e:
            logger.error(f"   ❌ Validation exception for GPU {gpu_id}: {e}")
            self.pipeline_stats['validations'] += 1
            self.pipeline_stats['failed_validations'] += 1
            return {'validation_engine_score': 0.0, 'error': str(e)}
    
    def generate_plys_parallel(self, prompt: str, source_images: Optional[List[Tuple[int, Image.Image]]] = None,
                              seeds: Optional[List[int]] = None) -> List[PLYGenerationResult]:
        """Generate PLY files on all GPUs in parallel"""
        logger.info(f"🔮 Generating PLY files on all {self.num_gpus} GPUs for: '{prompt}'")
        
        if seeds is None:
            seeds = [np.random.randint(0, 2**31 - 1) for _ in range(self.num_gpus)]
        elif len(seeds) < self.num_gpus:
            seeds.extend([np.random.randint(0, 2**31 - 1) for _ in range(self.num_gpus - len(seeds))])
        
        start_time = time.time()
        results = []
        
        # Generate PLY files on all GPUs simultaneously  
        with ThreadPoolExecutor(max_workers=self.num_gpus) as executor:
            future_to_gpu = {}
            
            for gpu_id in range(self.num_gpus):
                source_image = None
                source_image_gpu = None
                
                # Use source image if provided
                if source_images:
                    for img_gpu_id, img in source_images:
                        if img_gpu_id == gpu_id or len(source_images) == 1:
                            source_image = img
                            source_image_gpu = img_gpu_id
                            break
                
                future_to_gpu[executor.submit(
                    self.generate_ply_on_gpu,
                    gpu_id, prompt, seeds[gpu_id],
                    source_image, source_image_gpu
                )] = gpu_id
            
            for future in as_completed(future_to_gpu):
                gpu_id = future_to_gpu[future]
                try:
                    result = future.result()
                    results.append(result)
                    
                    # Update statistics
                    self.pipeline_stats['ply_generations'] += 1
                    if result.success:
                        self.pipeline_stats['successful_ply_generations'] += 1
                    else:
                        self.pipeline_stats['failed_ply_generations'] += 1
                        
                except Exception as e:
                    logger.error(f"❌ Exception in PLY generation on GPU {gpu_id}: {e}")
                    results.append(PLYGenerationResult(
                        gpu_id=gpu_id,
                        port=self.gpu_servers[gpu_id].port,
                        prompt=prompt,
                        success=False,
                        error=str(e)
                    ))
                    self.pipeline_stats['failed_ply_generations'] += 1
        
        parallel_time = time.time() - start_time
        
        # Sort results by GPU ID for consistency
        results.sort(key=lambda x: x.gpu_id)
        
        # Analyze results
        successful_results = [r for r in results if r.success]
        failed_results = [r for r in results if not r.success]
        
        logger.info(f"✅ Parallel PLY generation complete in {parallel_time:.2f}s")
        logger.info(f"   Successful: {len(successful_results)}/{len(results)}")
        logger.info(f"   Failed: {len(failed_results)}/{len(results)}")
        
        if successful_results:
            # Rank by validation score
            ranked_results = sorted(successful_results, key=lambda x: x.validation_score, reverse=True)
            
            logger.info("🏆 PLY Validation Score Ranking (Best to Worst):")
            for i, result in enumerate(ranked_results):
                medal = "🥇" if i == 0 else "🥈" if i == 1 else "🥉" if i == 2 else f"#{i+1}"
                logger.info(f"   {medal} GPU {result.gpu_id} (port {result.port}): Score {result.validation_score:.4f}")
                logger.info(f"      Generation time: {result.generation_time:.2f}s, PLY size: {result.ply_size:,}")
                if result.source_image_gpu is not None:
                    logger.info(f"      Source image from GPU {result.source_image_gpu}")
            
            # Store best scores
            self.pipeline_stats['best_validation_scores'].append(ranked_results[0].validation_score)
        
        return results
    
    def run_image_ranking_to_ply_pipeline(self, prompt: str, num_inference_steps: int = 25,
                                         guidance_scale: float = 7.5) -> PipelineResults:
        """
        Pipeline: Generate images on all GPUs → Rank by CLIP → Generate PLY from best images
        """
        logger.info("🚀 Starting Image Ranking → PLY Pipeline")
        logger.info(f"   Prompt: '{prompt}'")
        logger.info(f"   Strategy: 8x Image Generation → CLIP Ranking → Best Images → 8x PLY → Validation Ranking")
        
        pipeline_start_time = time.time()
        
        # Step 1: Generate images on all GPUs
        logger.info("📸 Phase 1: Parallel Image Generation")
        image_results = self.generate_images_parallel(
            prompt, num_inference_steps=num_inference_steps, guidance_scale=guidance_scale
        )
        
        # Step 2: Select best images based on CLIP scores
        successful_images = [r for r in image_results if r.success]
        if not successful_images:
            logger.error("❌ No successful image generations - aborting pipeline")
            return PipelineResults(
                prompt=prompt,
                pipeline_type="image_ranking_to_ply",
                image_results=image_results,
                total_pipeline_time=time.time() - pipeline_start_time
            )
        
        # Rank images by CLIP score and select top performers
        ranked_images = sorted(successful_images, key=lambda x: x.clip_score, reverse=True)
        best_image = ranked_images[0]
        
        # Select top 4 images for PLY generation (or all if less than 4)
        top_images = ranked_images[:min(4, len(ranked_images))]
        
        logger.info(f"📊 Phase 2: Selected {len(top_images)} best images for PLY generation")
        logger.info(f"   🥇 Best image: GPU {best_image.gpu_id} (CLIP: {best_image.clip_score:.4f})")
        
        # Step 3: Generate PLY files using best images
        logger.info("🔮 Phase 3: PLY Generation from Best Images")
        
        # Prepare source images: distribute top images across GPUs
        source_images = []
        for i, gpu_id in enumerate(range(self.num_gpus)):
            if i < len(top_images):
                source_images.append((top_images[i].gpu_id, top_images[i].pil_image))
            else:
                # Use best image for remaining GPUs
                source_images.append((best_image.gpu_id, best_image.pil_image))
        
        ply_results = self.generate_plys_parallel(prompt, source_images)
        
        # Step 4: Analyze final results
        successful_plys = [r for r in ply_results if r.success]
        best_ply = None
        if successful_plys:
            best_ply = max(successful_plys, key=lambda x: x.validation_score)
        
        total_time = time.time() - pipeline_start_time
        self.pipeline_stats['total_pipeline_time'] += total_time
        
        results = PipelineResults(
            prompt=prompt,
            pipeline_type="image_ranking_to_ply",
            image_results=image_results,
            ply_results=ply_results,
            best_image_gpu=best_image.gpu_id,
            best_ply_gpu=best_ply.gpu_id if best_ply else None,
            best_clip_score=best_image.clip_score,
            best_validation_score=best_ply.validation_score if best_ply else 0.0,
            total_pipeline_time=total_time
        )
        
        logger.info("🎉 Image Ranking → PLY Pipeline Complete!")
        logger.info(f"   Total time: {total_time:.2f}s")
        logger.info(f"   🥇 Best image: GPU {best_image.gpu_id} (CLIP: {best_image.clip_score:.4f})")
        if best_ply:
            logger.info(f"   🏆 Best PLY: GPU {best_ply.gpu_id} (Score: {best_ply.validation_score:.4f})")
        
        return results
    
    def run_single_image_multi_ply_pipeline(self, prompt: str, num_inference_steps: int = 25,
                                           guidance_scale: float = 7.5) -> PipelineResults:
        """
        Pipeline: Generate images → Select best → Generate multiple PLY variations
        """
        logger.info("🚀 Starting Single Image → Multi PLY Pipeline")
        logger.info(f"   Prompt: '{prompt}'")
        logger.info(f"   Strategy: 8x Image Generation → Best Image → 8x PLY Variations → Validation Ranking")
        
        pipeline_start_time = time.time()
        
        # Step 1: Generate images on all GPUs
        logger.info("📸 Phase 1: Parallel Image Generation")
        image_results = self.generate_images_parallel(
            prompt, num_inference_steps=num_inference_steps, guidance_scale=guidance_scale
        )
        
        # Step 2: Select single best image
        successful_images = [r for r in image_results if r.success]
        if not successful_images:
            logger.error("❌ No successful image generations - aborting pipeline")
            return PipelineResults(
                prompt=prompt,
                pipeline_type="single_image_multi_ply",
                image_results=image_results,
                total_pipeline_time=time.time() - pipeline_start_time
            )
        
        best_image = max(successful_images, key=lambda x: x.clip_score)
        
        logger.info(f"📊 Phase 2: Selected best image for PLY variations")
        logger.info(f"   🥇 Best image: GPU {best_image.gpu_id} (CLIP: {best_image.clip_score:.4f})")
        
        # Step 3: Generate PLY variations using the same best image on all GPUs
        logger.info("🔮 Phase 3: Multiple PLY Variations from Best Image")
        
        # Use the same best image on all GPUs with different seeds
        source_images = [(best_image.gpu_id, best_image.pil_image) for _ in range(self.num_gpus)]
        
        ply_results = self.generate_plys_parallel(prompt, source_images)
        
        # Step 4: Analyze final results
        successful_plys = [r for r in ply_results if r.success]
        best_ply = None
        if successful_plys:
            best_ply = max(successful_plys, key=lambda x: x.validation_score)
        
        total_time = time.time() - pipeline_start_time
        self.pipeline_stats['total_pipeline_time'] += total_time
        
        results = PipelineResults(
            prompt=prompt,
            pipeline_type="single_image_multi_ply",
            image_results=image_results,
            ply_results=ply_results,
            best_image_gpu=best_image.gpu_id,
            best_ply_gpu=best_ply.gpu_id if best_ply else None,
            best_clip_score=best_image.clip_score,
            best_validation_score=best_ply.validation_score if best_ply else 0.0,
            total_pipeline_time=total_time
        )
        
        logger.info("🎉 Single Image → Multi PLY Pipeline Complete!")
        logger.info(f"   Total time: {total_time:.2f}s")
        logger.info(f"   🥇 Best image: GPU {best_image.gpu_id} (CLIP: {best_image.clip_score:.4f})")
        if best_ply:
            logger.info(f"   🏆 Best PLY: GPU {best_ply.gpu_id} (Score: {best_ply.validation_score:.4f})")
        
        return results
    
    def save_pipeline_results(self, results: PipelineResults, timestamp: Optional[str] = None):
        """Save pipeline results to JSON file"""
        if timestamp is None:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
        
        filename = f"pipeline_results_{results.pipeline_type}_{timestamp}.json"
        filepath = self.output_dir / filename
        
        # Convert results to serializable format
        results_dict = {
            'prompt': results.prompt,
            'pipeline_type': results.pipeline_type,
            'timestamp': timestamp,
            'total_pipeline_time': results.total_pipeline_time,
            'best_image_gpu': results.best_image_gpu,
            'best_ply_gpu': results.best_ply_gpu,
            'best_clip_score': results.best_clip_score,
            'best_validation_score': results.best_validation_score,
            'image_results': [
                {
                    'gpu_id': r.gpu_id,
                    'port': r.port,
                    'success': r.success,
                    'generation_time': r.generation_time,
                    'clip_score': r.clip_score,
                    'seed': r.seed,
                    'error': r.error
                } for r in results.image_results
            ],
            'ply_results': [
                {
                    'gpu_id': r.gpu_id,
                    'port': r.port,
                    'success': r.success,
                    'generation_time': r.generation_time,
                    'ply_size': r.ply_size,
                    'validation_score': r.validation_score,
                    'alignment_score': r.alignment_score,
                    'quality_score': r.quality_score,
                    'demo_fidelity_score': r.demo_fidelity_score,
                    'source_image_gpu': r.source_image_gpu,
                    'error': r.error
                } for r in results.ply_results
            ]
        }
        
        with open(filepath, 'w') as f:
            json.dump(results_dict, f, indent=2)
        
        logger.info(f"💾 Pipeline results saved to {filepath}")
    
    def print_pipeline_summary(self):
        """Print comprehensive pipeline statistics"""
        logger.info("📊 MULTI-GPU PIPELINE SUMMARY")
        logger.info("=" * 80)
        
        logger.info(f"🎨 Image Generation:")
        logger.info(f"   Total: {self.pipeline_stats['image_generations']}")
        logger.info(f"   Successful: {self.pipeline_stats['successful_image_generations']}")
        logger.info(f"   Failed: {self.pipeline_stats['failed_image_generations']}")
        if self.pipeline_stats['image_generations'] > 0:
            success_rate = (self.pipeline_stats['successful_image_generations'] / 
                          self.pipeline_stats['image_generations']) * 100
            logger.info(f"   Success Rate: {success_rate:.1f}%")
        
        logger.info(f"🔮 PLY Generation:")
        logger.info(f"   Total: {self.pipeline_stats['ply_generations']}")
        logger.info(f"   Successful: {self.pipeline_stats['successful_ply_generations']}")
        logger.info(f"   Failed: {self.pipeline_stats['failed_ply_generations']}")
        if self.pipeline_stats['ply_generations'] > 0:
            success_rate = (self.pipeline_stats['successful_ply_generations'] / 
                          self.pipeline_stats['ply_generations']) * 100
            logger.info(f"   Success Rate: {success_rate:.1f}%")
        
        logger.info(f"🔍 Validation:")
        logger.info(f"   Total: {self.pipeline_stats['validations']}")
        logger.info(f"   Successful: {self.pipeline_stats['successful_validations']}")
        logger.info(f"   Failed: {self.pipeline_stats['failed_validations']}")
        
        if self.pipeline_stats['best_clip_scores']:
            logger.info(f"🏆 CLIP Scores:")
            logger.info(f"   Best: {max(self.pipeline_stats['best_clip_scores']):.4f}")
            logger.info(f"   Average: {np.mean(self.pipeline_stats['best_clip_scores']):.4f}")
        
        if self.pipeline_stats['best_validation_scores']:
            logger.info(f"🎯 Validation Scores:")
            logger.info(f"   Best: {max(self.pipeline_stats['best_validation_scores']):.4f}")
            logger.info(f"   Average: {np.mean(self.pipeline_stats['best_validation_scores']):.4f}")
        
        logger.info(f"⏱️ Total Pipeline Time: {self.pipeline_stats['total_pipeline_time']:.2f}s")
        
        logger.info("=" * 80)

def signal_handler(signum, frame):
    """Handle shutdown signals"""
    logger.info(f"🛑 Received signal {signum}, shutting down...")
    if hasattr(signal_handler, 'manager'):
        signal_handler.manager.cleanup()
    sys.exit(0)

async def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Multi-GPU Pipeline Wrapper for TRELLIS")
    parser.add_argument("--gpus", type=int, default=8, help="Number of GPUs to use")
    parser.add_argument("--base-port", type=int, default=8096, help="Base port number")
    parser.add_argument("--server-script", default="trellis_subnit_server_mix_lora_flash.py", 
                       help="TRELLIS server script path")
    parser.add_argument("--output-dir", default="./gpu_pipeline_outputs", help="Output directory")
    
    # Pipeline options
    parser.add_argument("--prompt", type=str, required=True, help="Text prompt for generation")
    parser.add_argument("--pipeline", choices=["image_ranking", "single_image", "both"], 
                       default="both", help="Pipeline type to run")
    parser.add_argument("--num-inference-steps", type=int, default=25, help="Number of inference steps")
    parser.add_argument("--guidance-scale", type=float, default=7.5, help="Guidance scale")
    
    # Server management options
    parser.add_argument("--skip-startup", action="store_true", help="Skip server startup (assume already running)")
    parser.add_argument("--check-status-only", action="store_true", help="Only check GPU loading status and exit")
    
    args = parser.parse_args()
    
    # Setup signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Create pipeline manager
    manager = MultiGPUPipelineManager(
        num_gpus=args.gpus,
        base_port=args.base_port,
        server_script=args.server_script,
        output_dir=args.output_dir
    )
    
    # Store manager reference for signal handler
    signal_handler.manager = manager
    
    try:
        # Check status only if requested
        if args.check_status_only:
            logger.info("🔍 Checking GPU loading status only...")
            loading_status = manager.check_gpu_loading_status()
            return
        
        # Start servers if needed
        if not args.skip_startup:
            logger.info("🚀 Starting GPU servers...")
            if not manager.start_all_servers():
                logger.error("❌ Failed to start GPU servers")
                return
        else:
            logger.info("⏭️ Skipping server startup (assume already running)")
            manager.check_all_servers_health()
        
        # Run pipeline(s)
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        
        if args.pipeline in ["image_ranking", "both"]:
            logger.info("🚀 Running Image Ranking → PLY Pipeline")
            results = manager.run_image_ranking_to_ply_pipeline(
                args.prompt, 
                num_inference_steps=args.num_inference_steps,
                guidance_scale=args.guidance_scale
            )
            manager.save_pipeline_results(results, timestamp + "_image_ranking")
        
        if args.pipeline in ["single_image", "both"]:
            logger.info("🚀 Running Single Image → Multi PLY Pipeline")
            results = manager.run_single_image_multi_ply_pipeline(
                args.prompt,
                num_inference_steps=args.num_inference_steps,
                guidance_scale=args.guidance_scale
            )
            manager.save_pipeline_results(results, timestamp + "_single_image")
        
        # Print final summary
        manager.print_pipeline_summary()
        
        logger.info("🎉 All pipelines completed successfully!")
        
    except KeyboardInterrupt:
        logger.info("🛑 Interrupted by user")
    except Exception as e:
        logger.error(f"❌ Pipeline execution failed: {e}")
        traceback.print_exc()
    finally:
        if not args.skip_startup:
            logger.info("🧹 Cleaning up...")
            manager.cleanup()

if __name__ == "__main__":
    asyncio.run(main())
