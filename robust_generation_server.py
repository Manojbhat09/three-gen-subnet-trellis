#!/usr/bin/env python3
# Robust Generation Server for Subnet 17
# Enhanced with circuit breakers, health monitoring, and advanced error recovery

import os
import time
import torch
import traceback
import threading
import gc
import asyncio
import psutil
from pathlib import Path
from typing import Optional, Dict, Any, List
from dataclasses import dataclass, asdict
from contextlib import asynccontextmanager
from enum import Enum

from fastapi import FastAPI, Form, HTTPException, BackgroundTasks
from fastapi.responses import Response, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import trimesh
from PIL import Image
import numpy as np
import GPUtil

# Import Hunyuan3D-2 components
from Hunyuan3D_2.hy3dgen.rembg import BackgroundRemover
from Hunyuan3D_2.hy3dgen.shapegen import (
    Hunyuan3DDiTFlowMatchingPipeline, 
    FaceReducer, 
    FloaterRemover, 
    DegenerateFaceRemover
)

# Import Flux components
from diffusers import FluxTransformer2DModel, FluxPipeline
from transformers import T5EncoderModel
from accelerate import init_empty_weights
from accelerate.utils import load_checkpoint_and_dispatch

# Configuration
GENERATION_CONFIG = {
    'model_cache_dir': './models',
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    'output_dir': './generation_outputs',
    'max_concurrent_requests': 3,
    'memory_cleanup_threshold': 0.9,  # 90% VRAM usage triggers cleanup
    'circuit_breaker_failure_threshold': 5,
    'circuit_breaker_recovery_timeout': 300,  # 5 minutes
    'health_check_interval': 30,
    'max_generation_time': 300,
    'quality_check_enabled': True,
    'auto_restart_on_oom': True,
}

class ServerState(Enum):
    INITIALIZING = "initializing"
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    MAINTENANCE = "maintenance"

@dataclass
class SystemMetrics:
    cpu_percent: float
    memory_percent: float
    vram_used_gb: float
    vram_total_gb: float
    disk_free_gb: float
    active_requests: int
    total_generations: int
    failed_generations: int
    avg_generation_time: float
    last_error: Optional[str] = None
    uptime_seconds: float = 0

@dataclass
class CircuitBreakerState:
    failure_count: int = 0
    last_failure_time: float = 0
    state: str = "CLOSED"  # CLOSED, OPEN, HALF_OPEN
    
class CircuitBreaker:
    def __init__(self, failure_threshold: int = 5, recovery_timeout: float = 300):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.state = CircuitBreakerState()
        
    def record_success(self):
        self.state.failure_count = 0
        self.state.state = "CLOSED"
        
    def record_failure(self):
        self.state.failure_count += 1
        self.state.last_failure_time = time.time()
        if self.state.failure_count >= self.failure_threshold:
            self.state.state = "OPEN"
            
    def can_proceed(self) -> bool:
        if self.state.state == "CLOSED":
            return True
        elif self.state.state == "OPEN":
            if time.time() - self.state.last_failure_time > self.recovery_timeout:
                self.state.state = "HALF_OPEN"
                return True
            return False
        else:  # HALF_OPEN
            return True

class ResourceMonitor:
    def __init__(self):
        self.start_time = time.time()
        self.generation_times = []
        self.error_log = []
        self.peak_memory = 0
        
    def get_system_metrics(self) -> SystemMetrics:
        # CPU and RAM
        cpu_percent = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/')
        
        # GPU metrics
        vram_used, vram_total = 0, 0
        try:
            gpus = GPUtil.getGPUs()
            if gpus:
                gpu = gpus[0]
                vram_used = gpu.memoryUsed / 1024  # Convert to GB
                vram_total = gpu.memoryTotal / 1024
        except:
            pass
            
        # Generation metrics
        avg_gen_time = sum(self.generation_times[-100:]) / max(len(self.generation_times[-100:]), 1)
        
        return SystemMetrics(
            cpu_percent=cpu_percent,
            memory_percent=memory.percent,
            vram_used_gb=vram_used,
            vram_total_gb=vram_total,
            disk_free_gb=disk.free / (1024**3),
            active_requests=active_requests_count,
            total_generations=len(self.generation_times),
            failed_generations=len(self.error_log),
            avg_generation_time=avg_gen_time,
            last_error=self.error_log[-1] if self.error_log else None,
            uptime_seconds=time.time() - self.start_time
        )
        
    def record_generation_time(self, duration: float):
        self.generation_times.append(duration)
        # Keep only last 1000 records
        if len(self.generation_times) > 1000:
            self.generation_times = self.generation_times[-1000:]
            
    def record_error(self, error: str):
        self.error_log.append(f"{time.time()}: {error}")
        if len(self.error_log) > 100:
            self.error_log = self.error_log[-100:]

# Global state
server_state = ServerState.INITIALIZING
active_requests_count = 0
resource_monitor = ResourceMonitor()
circuit_breaker = CircuitBreaker()

# Pipeline components (loaded lazily)
flux_pipeline = None
hunyuan_pipeline = None
background_remover = None

async def load_models_if_needed():
    """Load models only when needed to save memory"""
    global flux_pipeline, hunyuan_pipeline, background_remover
    
    try:
        if flux_pipeline is None:
            print("Loading Flux pipeline...")
            flux_pipeline = FluxPipeline.from_pretrained(
                "black-forest-labs/FLUX.1-dev",
                torch_dtype=torch.bfloat16,
                cache_dir=GENERATION_CONFIG['model_cache_dir']
            ).to(GENERATION_CONFIG['device'])
            
        if hunyuan_pipeline is None:
            print("Loading Hunyuan3D pipeline...")
            hunyuan_pipeline = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(
                "Hunyuan3D-2/",
                torch_dtype=torch.float16,
                cache_dir=GENERATION_CONFIG['model_cache_dir']
            ).to(GENERATION_CONFIG['device'])
            
        if background_remover is None:
            background_remover = BackgroundRemover()
            
        return True
    except Exception as e:
        print(f"Failed to load models: {e}")
        traceback.print_exc()
        return False

async def cleanup_memory():
    """Aggressive memory cleanup"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()
    
async def check_memory_pressure() -> bool:
    """Check if we're running low on memory"""
    try:
        gpus = GPUtil.getGPUs()
        if gpus:
            gpu = gpus[0]
            memory_usage = gpu.memoryUsed / gpu.memoryTotal
            return memory_usage > GENERATION_CONFIG['memory_cleanup_threshold']
    except:
        pass
    return False

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    global server_state
    print("Starting Robust Generation Server...")
    
    # Create output directory
    os.makedirs(GENERATION_CONFIG['output_dir'], exist_ok=True)
    
    # Start background tasks
    asyncio.create_task(health_monitor())
    asyncio.create_task(memory_monitor())
    
    server_state = ServerState.HEALTHY
    print("Server started successfully!")
    
    yield
    
    # Shutdown
    print("Shutting down server...")
    await cleanup_memory()

app = FastAPI(
    title="Robust Generation Server",
    description="High-performance 3D generation with advanced error handling",
    version="2.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

async def health_monitor():
    """Background task to monitor system health"""
    global server_state
    
    while True:
        try:
            metrics = resource_monitor.get_system_metrics()
            
            # Determine server state based on metrics
            if metrics.vram_used_gb > metrics.vram_total_gb * 0.95:
                server_state = ServerState.UNHEALTHY
            elif metrics.memory_percent > 90 or metrics.disk_free_gb < 1:
                server_state = ServerState.DEGRADED
            elif circuit_breaker.state.state == "OPEN":
                server_state = ServerState.UNHEALTHY
            else:
                server_state = ServerState.HEALTHY
                
        except Exception as e:
            print(f"Health monitor error: {e}")
            
        await asyncio.sleep(GENERATION_CONFIG['health_check_interval'])

async def memory_monitor():
    """Background task to monitor and manage memory"""
    while True:
        try:
            if await check_memory_pressure():
                print("Memory pressure detected, performing cleanup...")
                await cleanup_memory()
                await asyncio.sleep(5)  # Give time for cleanup
                
        except Exception as e:
            print(f"Memory monitor error: {e}")
            
        await asyncio.sleep(10)

@app.get("/health")
async def health_check():
    """Comprehensive health check endpoint"""
    metrics = resource_monitor.get_system_metrics()
    
    health_status = {
        "status": server_state.value,
        "timestamp": time.time(),
        "circuit_breaker": asdict(circuit_breaker.state),
        "metrics": asdict(metrics),
        "can_accept_requests": (
            server_state in [ServerState.HEALTHY, ServerState.DEGRADED] and
            circuit_breaker.can_proceed() and
            active_requests_count < GENERATION_CONFIG['max_concurrent_requests']
        )
    }
    
    status_code = 200 if server_state != ServerState.UNHEALTHY else 503
    return JSONResponse(content=health_status, status_code=status_code)

@app.get("/metrics")
async def get_metrics():
    """Detailed metrics endpoint for monitoring"""
    return asdict(resource_monitor.get_system_metrics())

@app.post("/generate/")
async def generate_3d_model(
    prompt: str = Form(...),
    background_tasks: BackgroundTasks = None
):
    """Enhanced generation endpoint with robust error handling"""
    global active_requests_count
    
    # Pre-flight checks
    if not circuit_breaker.can_proceed():
        raise HTTPException(
            status_code=503, 
            detail="Service temporarily unavailable due to circuit breaker"
        )
        
    if active_requests_count >= GENERATION_CONFIG['max_concurrent_requests']:
        raise HTTPException(
            status_code=429, 
            detail="Too many active requests. Please try again later."
        )
        
    if server_state == ServerState.UNHEALTHY:
        raise HTTPException(
            status_code=503,
            detail="Server is unhealthy. Please try again later."
        )
    
    active_requests_count += 1
    start_time = time.time()
    
    try:
        # Load models if needed
        if not await load_models_if_needed():
            circuit_breaker.record_failure()
            raise HTTPException(status_code=500, detail="Failed to load models")
            
        # Memory pressure check
        if await check_memory_pressure():
            await cleanup_memory()
            
        # Generate image using Flux
        print(f"Generating image for prompt: {prompt}")
        with torch.inference_mode():
            image = flux_pipeline(
                prompt=prompt,
                num_inference_steps=20,
                guidance_scale=3.5,
                height=1024,
                width=1024,
                generator=torch.Generator().manual_seed(42)
            ).images[0]
            
        # Remove background
        image_np = np.array(image)
        rgba_image = background_remover.process(image_np)
        
        # Generate 3D model
        print("Generating 3D mesh...")
        with torch.inference_mode():
            mesh = hunyuan_pipeline(
                image=rgba_image,
                num_inference_steps=50,
                guidance_scale=7.5
            )
            
        # Post-process mesh
        mesh = FloaterRemover().process(mesh)
        mesh = DegenerateFaceRemover().process(mesh)
        mesh = FaceReducer().process(mesh)
        
        # Convert to PLY
        ply_data = mesh.export(file_type='ply')
        
        # Record success
        generation_time = time.time() - start_time
        resource_monitor.record_generation_time(generation_time)
        circuit_breaker.record_success()
        
        print(f"Generation completed in {generation_time:.2f}s")
        
        # Schedule cleanup
        if background_tasks:
            background_tasks.add_task(cleanup_memory)
            
        return Response(
            content=ply_data,
            media_type="application/octet-stream",
            headers={
                "Content-Disposition": "attachment; filename=model.ply",
                "X-Generation-Time": str(generation_time),
                "X-Model-Size": str(len(ply_data))
            }
        )
        
    except torch.cuda.OutOfMemoryError as e:
        error_msg = f"GPU Out of Memory: {str(e)}"
        resource_monitor.record_error(error_msg)
        circuit_breaker.record_failure()
        
        # Aggressive cleanup on OOM
        await cleanup_memory()
        
        if GENERATION_CONFIG['auto_restart_on_oom']:
            # Schedule server restart
            asyncio.create_task(restart_server_after_delay(60))
            
        raise HTTPException(status_code=507, detail="Insufficient GPU memory")
        
    except Exception as e:
        error_msg = f"Generation failed: {str(e)}"
        resource_monitor.record_error(error_msg)
        circuit_breaker.record_failure()
        
        print(f"Generation error: {e}")
        traceback.print_exc()
        
        raise HTTPException(status_code=500, detail=str(e))
        
    finally:
        active_requests_count -= 1

async def restart_server_after_delay(delay_seconds: int):
    """Restart server after a delay (for OOM recovery)"""
    global server_state
    
    await asyncio.sleep(delay_seconds)
    server_state = ServerState.MAINTENANCE
    
    # Reload all models
    global flux_pipeline, hunyuan_pipeline, background_remover
    flux_pipeline = None
    hunyuan_pipeline = None
    background_remover = None
    
    await cleanup_memory()
    await load_models_if_needed()
    
    server_state = ServerState.HEALTHY

if __name__ == "__main__":
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8093,
        workers=1,
        access_log=False,
        log_level="info"
    ) 