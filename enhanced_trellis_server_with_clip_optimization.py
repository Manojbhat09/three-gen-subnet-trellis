#!/usr/bin/env python3
"""
Enhanced TRELLIS Server with DiT + CLIP Optimization
Integrates prompt optimization using CLIP feedback before 3D generation
Pipeline: Text → DiT (Image) → CLIP Score → Prompt Optimization → DiT (Final Image) → TRELLIS 3D
"""

import asyncio
import time
import random
import base64
import io
from typing import Optional, Dict, Any
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn
import argparse
from loguru import logger

# Import your existing TRELLIS components
try:
    from trellis_submit_server_hunyuan_dit import HunyuanDiTTrellisGenerator
except ImportError:
    logger.warning("Could not import HunyuanDiTTrellisGenerator, using mock")
    class HunyuanDiTTrellisGenerator:
        def __init__(self):
            pass
        async def generate_3d_model(self, prompt: str, seed: int = None):
            return {"status": "mock", "prompt": prompt}

# Import CLIP optimizer
from dit_clip_optimizer_integration import DiTClipOptimizer

class GenerationRequest(BaseModel):
    prompt: str
    seed: Optional[int] = None
    enable_clip_optimization: bool = True
    max_optimization_iterations: int = 3
    target_clip_score: float = 0.7

class GenerationResponse(BaseModel):
    status: str
    original_prompt: str
    optimized_prompt: Optional[str] = None
    clip_score: Optional[float] = None
    optimization_data: Optional[Dict] = None
    ply_data: Optional[str] = None
    generation_time: float
    total_time: float

class EnhancedTrellisServer:
    """Enhanced TRELLIS server with DiT + CLIP optimization"""
    
    def __init__(self, 
                 dit_server_url: str = "http://localhost:8000",
                 enable_clip_optimization: bool = True):
        
        self.dit_server_url = dit_server_url
        self.enable_clip_optimization = enable_clip_optimization
        
        # Initialize components
        self.trellis_generator = HunyuanDiTTrellisGenerator()
        
        if self.enable_clip_optimization:
            self.clip_optimizer = DiTClipOptimizer(
                dit_server_url=dit_server_url,
                max_iterations=3,
                target_score=0.7
            )
            logger.info("✅ CLIP optimization enabled")
        else:
            self.clip_optimizer = None
            logger.info("⚠️ CLIP optimization disabled")
        
        # FastAPI app
        self.app = FastAPI(
            title="Enhanced TRELLIS Server with CLIP Optimization",
            version="2.0.0",
            description="TRELLIS 3D generation with DiT + CLIP prompt optimization"
        )
        
        self._setup_routes()
    
    def _setup_routes(self):
        """Setup FastAPI routes"""
        
        @self.app.get("/")
        async def root():
            return {
                "message": "Enhanced TRELLIS Server with CLIP Optimization",
                "pipeline": "Text → DiT (Image) → CLIP Score → Prompt Optimization → DiT (Final) → TRELLIS 3D",
                "features": [
                    "DiT image generation",
                    "CLIP-based prompt optimization", 
                    "TRELLIS 3D generation",
                    "Automatic prompt improvement"
                ]
            }
        
        @self.app.post("/generate_3d_optimized/", response_model=GenerationResponse)
        async def generate_3d_optimized(request: GenerationRequest):
            """Generate 3D model with optional CLIP optimization"""
            return await self._generate_with_optimization(request)
        
        @self.app.post("/generate_3d_direct/", response_model=GenerationResponse)
        async def generate_3d_direct(request: GenerationRequest):
            """Generate 3D model directly without optimization (for comparison)"""
            return await self._generate_direct(request)
        
        @self.app.get("/health/")
        async def health_check():
            return {"status": "healthy", "clip_optimization": self.enable_clip_optimization}
    
    async def _generate_with_optimization(self, request: GenerationRequest) -> GenerationResponse:
        """Generate 3D model with CLIP optimization"""
        total_start_time = time.time()
        
        try:
            original_prompt = request.prompt
            seed = request.seed or random.randint(0, 2**31 - 1)
            
            logger.info(f"🚀 Starting optimized generation for: '{original_prompt}'")
            
            # Step 1: CLIP Optimization (if enabled)
            optimization_data = None
            optimized_prompt = original_prompt
            clip_score = None
            
            if self.enable_clip_optimization and request.enable_clip_optimization:
                logger.info("🔍 Starting CLIP optimization...")
                optimization_start = time.time()
                
                optimized_prompt, clip_score, optimization_data = self.clip_optimizer.optimize_prompt(
                    original_prompt, 
                    seed=seed
                )
                
                optimization_time = time.time() - optimization_start
                logger.info(f"✅ CLIP optimization completed in {optimization_time:.2f}s")
                logger.info(f"   Original: '{original_prompt}'")
                logger.info(f"   Optimized: '{optimized_prompt}'")
                logger.info(f"   CLIP Score: {clip_score:.4f}")
            
            # Step 2: Generate 3D model with optimized prompt
            logger.info("🎯 Generating 3D model with optimized prompt...")
            generation_start = time.time()
            
            # Use the optimized prompt for 3D generation
            trellis_result = await self.trellis_generator.generate_3d_model(
                optimized_prompt, 
                seed=seed
            )
            
            generation_time = time.time() - generation_start
            total_time = time.time() - total_start_time
            
            logger.info(f"✅ 3D generation completed in {generation_time:.2f}s")
            logger.info(f"📊 Total time: {total_time:.2f}s")
            
            return GenerationResponse(
                status="success",
                original_prompt=original_prompt,
                optimized_prompt=optimized_prompt,
                clip_score=clip_score,
                optimization_data=optimization_data,
                ply_data=trellis_result.get('ply_data'),
                generation_time=generation_time,
                total_time=total_time
            )
            
        except Exception as e:
            logger.error(f"❌ Generation failed: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    async def _generate_direct(self, request: GenerationRequest) -> GenerationResponse:
        """Generate 3D model directly without optimization"""
        total_start_time = time.time()
        
        try:
            original_prompt = request.prompt
            seed = request.seed or random.randint(0, 2**31 - 1)
            
            logger.info(f"🚀 Starting direct generation for: '{original_prompt}'")
            
            # Generate 3D model directly
            generation_start = time.time()
            trellis_result = await self.trellis_generator.generate_3d_model(
                original_prompt, 
                seed=seed
            )
            
            generation_time = time.time() - generation_start
            total_time = time.time() - total_start_time
            
            logger.info(f"✅ Direct generation completed in {generation_time:.2f}s")
            
            return GenerationResponse(
                status="success",
                original_prompt=original_prompt,
                optimized_prompt=None,
                clip_score=None,
                optimization_data=None,
                ply_data=trellis_result.get('ply_data'),
                generation_time=generation_time,
                total_time=total_time
            )
            
        except Exception as e:
            logger.error(f"❌ Direct generation failed: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    def run(self, host: str = "0.0.0.0", port: int = 8001):
        """Run the enhanced server"""
        logger.info(f"🚀 Starting Enhanced TRELLIS Server on {host}:{port}")
        logger.info(f"   Pipeline: Text → DiT → CLIP Optimization → TRELLIS 3D")
        logger.info(f"   CLIP Optimization: {'✅ Enabled' if self.enable_clip_optimization else '❌ Disabled'}")
        
        uvicorn.run(self.app, host=host, port=port)

def compare_optimized_vs_direct():
    """Compare optimized vs direct generation"""
    import requests
    
    test_prompts = [
        "red ceramic vase",
        "metallic robot",
        "wooden chair",
        "glass container with flowers"
    ]
    
    server_url = "http://localhost:8001"
    
    print("🔍 Comparing Optimized vs Direct Generation")
    print("=" * 60)
    
    for prompt in test_prompts:
        print(f"\n📝 Testing: '{prompt}'")
        
        # Test optimized generation
        try:
            optimized_response = requests.post(
                f"{server_url}/generate_3d_optimized/",
                json={
                    "prompt": prompt,
                    "enable_clip_optimization": True,
                    "max_optimization_iterations": 3
                },
                timeout=120
            )
            
            if optimized_response.status_code == 200:
                optimized_data = optimized_response.json()
                print(f"   ✅ Optimized: {optimized_data['total_time']:.1f}s")
                print(f"      CLIP Score: {optimized_data.get('clip_score', 'N/A')}")
                print(f"      Optimized Prompt: '{optimized_data.get('optimized_prompt', 'N/A')}'")
            else:
                print(f"   ❌ Optimized failed: {optimized_response.status_code}")
        
        except Exception as e:
            print(f"   ❌ Optimized error: {e}")
        
        # Test direct generation
        try:
            direct_response = requests.post(
                f"{server_url}/generate_3d_direct/",
                json={"prompt": prompt},
                timeout=120
            )
            
            if direct_response.status_code == 200:
                direct_data = direct_response.json()
                print(f"   ✅ Direct: {direct_data['total_time']:.1f}s")
            else:
                print(f"   ❌ Direct failed: {direct_response.status_code}")
        
        except Exception as e:
            print(f"   ❌ Direct error: {e}")

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Enhanced TRELLIS Server with CLIP Optimization")
    parser.add_argument("--host", default="0.0.0.0", help="Server host")
    parser.add_argument("--port", type=int, default=8001, help="Server port")
    parser.add_argument("--dit-server", default="http://localhost:8000", help="DiT server URL")
    parser.add_argument("--disable-clip-optimization", action="store_true", help="Disable CLIP optimization")
    parser.add_argument("--compare", action="store_true", help="Run comparison test")
    
    args = parser.parse_args()
    
    if args.compare:
        # Run comparison test
        compare_optimized_vs_direct()
        return
    
    # Start server
    server = EnhancedTrellisServer(
        dit_server_url=args.dit_server,
        enable_clip_optimization=not args.disable_clip_optimization
    )
    
    try:
        server.run(host=args.host, port=args.port)
    except KeyboardInterrupt:
        logger.info("🛑 Server stopped by user")
    finally:
        if server.clip_optimizer:
            server.clip_optimizer.cleanup()

if __name__ == "__main__":
    main() 