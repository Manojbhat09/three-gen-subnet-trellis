#!/usr/bin/env python3
"""
Validation Simulation Server - Subnet 17 Validator Simulator
Purpose: Stress test miners/orchestrators by simulating real validator behavior

Features:
- Random word prompt generation
- Realistic cooldown enforcement (300s synthetic, 120s organic)
- Cooldown violation tracking and penalties
- Quality threshold validation
- Throttle period management
- Emergency cooldown escalation
- Realistic response delays and failures
- Comprehensive logging and monitoring
"""

import asyncio
import json
import time
import random
import logging
import traceback
import hashlib
import sqlite3
from pathlib import Path
from typing import List, Dict, Optional, Any, Set
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
import re
import signal
import atexit
import uuid
import socket
import os
from collections import deque, defaultdict

# FastAPI for HTTP endpoints
try:
    from fastapi import FastAPI, HTTPException, Form, Request
    from fastapi.responses import JSONResponse, Response
    from fastapi.middleware.cors import CORSMiddleware
    import uvicorn
    from pydantic import BaseModel, Field
    FASTAPI_AVAILABLE = True
except ImportError:
    print("⚠️ FastAPI not available - install with: pip install fastapi uvicorn")
    FASTAPI_AVAILABLE = False

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('validation_simulation.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Configuration
DEFAULT_CONFIG = {
    'server_port': 8094,
    'host': '0.0.0.0',
    
    # Cooldown settings (matching real subnet)
    'synthetic_traffic_cooldown': 300,  # 300s for synthetic traffic
    'organic_traffic_cooldown': 120,    # 120s for organic traffic
    
    # Validation settings
    'quality_threshold': 0.6,
    'throttle_period': 30,
    'cooldown_violation_penalty': 10,
    'cooldown_violations_threshold': 100,
    'cooldown_penalty': 600,  # 10 minutes for poor quality
    
    # Simulation settings
    'enable_random_failures': True,
    'failure_rate': 0.05,  # 5% random failures
    'response_delay_range': (0.1, 2.0),  # 0.1s to 2s delay
    'enable_network_issues': True,
    'network_issue_rate': 0.02,  # 2% network issues
    
    # Prompt generation
    'min_prompt_length': 10,
    'max_prompt_length': 100,
    'enable_traffic_detection': True,
    
    # Stress testing
    'enable_stress_testing': True,
    'max_concurrent_requests': 50,
    'rate_limit_per_minute': 100,
}

@dataclass
class SimulatedMiner:
    """Simulates a miner's state for cooldown tracking"""
    hotkey: str
    uid: int
    
    # Task tracking
    assigned_task: Optional[Dict] = None
    assignment_time: Optional[float] = None
    
    # Cooldown management (matching real subnet)
    cooldown_until: int = 0
    cooldown_violations: int = 0
    validation_locked_until: int = 0
    last_submit_time: int = 0
    
    # Performance tracking
    total_tasks_pulled: int = 0
    total_tasks_submitted: int = 0
    total_successful_submissions: int = 0
    average_score: float = 1.0
    observations: deque = None
    
    # Emergency cooldown management
    emergency_blacklist_until: Optional[int] = None
    last_violation_check: Optional[float] = None
    
    def __post_init__(self):
        if self.observations is None:
            self.observations = deque(maxlen=100)  # Last 100 observations
    
    def is_on_cooldown(self) -> bool:
        """Check if miner is on cooldown (matching real subnet logic)"""
        if self.cooldown_until == 0:
            return False
        return time.time() < self.cooldown_until
    
    def cooldown_left(self) -> int:
        """Return remaining cooldown time"""
        return max(0, int(self.cooldown_until - time.time()))
    
    def reset_task(self, throttle_period: int, cooldown: int) -> None:
        """Reset task and set cooldown (matching real subnet logic)"""
        if self.assignment_time is None:
            self.cooldown_until = int(time.time()) + cooldown
        else:
            # Account for throttle period (faster completion = longer cooldown)
            self.cooldown_until = int(max(
                time.time() + cooldown - throttle_period,
                self.assignment_time + cooldown
            ))
        
        self.assigned_task = None
        self.assignment_time = None
    
    def assign_task(self, task: Dict) -> None:
        """Assign task to miner"""
        self.assigned_task = task
        self.assignment_time = time.time()
    
    def add_observation(self, task_finish_time: int, fidelity_score: float, moving_average_alpha: float = 0.05) -> None:
        """Add task observation and update average score"""
        self.observations.append(task_finish_time)
        
        # Update average score using exponential moving average
        prev_score = self.average_score
        self.average_score = prev_score * (1 - moving_average_alpha) + moving_average_alpha * fidelity_score
        
        logger.debug(
            f"[{self.uid}] score: {fidelity_score:.2f}. "
            f"Avg score: {prev_score:.2f} -> {self.average_score:.2f}. "
            f"Observations: {len(self.observations)}"
        )

class PromptGenerator:
    """Generates realistic prompts for testing"""
    
    def __init__(self):
        # Common 3D generation prompts
        self.common_objects = [
            "chair", "table", "lamp", "vase", "book", "cup", "bottle", "plant", "toy", "tool",
            "vehicle", "building", "animal", "human", "furniture", "decoration", "instrument"
        ]
        
        self.adjectives = [
            "modern", "vintage", "rustic", "elegant", "simple", "complex", "detailed", "minimalist",
            "colorful", "monochrome", "textured", "smooth", "rough", "shiny", "matte", "transparent"
        ]
        
        self.materials = [
            "wooden", "metal", "glass", "plastic", "ceramic", "stone", "fabric", "leather",
            "bamboo", "concrete", "marble", "granite", "copper", "brass", "steel", "aluminum"
        ]
        
        self.styles = [
            "art deco", "mid-century modern", "scandinavian", "industrial", "bohemian", "minimalist",
            "traditional", "contemporary", "futuristic", "classical", "gothic", "baroque"
        ]
        
        self.environments = [
            "indoor", "outdoor", "garden", "office", "kitchen", "bedroom", "living room",
            "studio", "workshop", "gallery", "museum", "park", "beach", "mountain"
        ]
    
    def generate_synthetic_prompt(self) -> str:
        """Generate synthetic traffic prompt (test/benchmark style)"""
        prompt_type = random.choice([
            "test", "benchmark", "validation", "duel", "challenge", "evaluation", "performance"
        ])
        
        object_type = random.choice(self.common_objects)
        adjective = random.choice(self.adjectives)
        
        return f"{prompt_type} {adjective} {object_type} for validation"
    
    def generate_organic_prompt(self) -> str:
        """Generate organic traffic prompt (user/real style)"""
        object_type = random.choice(self.common_objects)
        adjective = random.choice(self.adjectives)
        material = random.choice(self.materials)
        style = random.choice(self.styles)
        environment = random.choice(self.environments)
        
        return f"{material} {adjective} {style} {object_type} for {environment} use"
    
    def generate_random_prompt(self) -> str:
        """Generate random prompt with traffic type detection"""
        if random.random() < 0.3:  # 30% synthetic traffic
            return self.generate_synthetic_prompt()
        else:  # 70% organic traffic
            return self.generate_organic_prompt()
    
    def detect_traffic_type(self, prompt: str) -> str:
        """Detect traffic type from prompt (matching orchestrator logic)"""
        prompt_lower = prompt.lower()
        
        # Synthetic patterns
        synthetic_patterns = [
            'syn_', 'synthetic', 'test_', 'benchmark_', 'validation_',
            'duel_', 'challenge_', 'competition_', 'evaluation_'
        ]
        
        # Organic patterns
        organic_patterns = [
            'org_', 'organic', 'gateway_', 'legacy_', 'user_', 'real_',
            'production_', 'live_', 'api_', 'external_'
        ]
        
        # Check for synthetic patterns
        for pattern in synthetic_patterns:
            if pattern in prompt_lower:
                return 'synthetic'
        
        # Check for organic patterns
        for pattern in organic_patterns:
            if pattern in prompt_lower:
                return 'organic'
        
        # Default to synthetic for safety
        return 'synthetic'

class ValidationSimulator:
    """Simulates the validation service behavior"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.prompt_generator = PromptGenerator()
        
        # Miner tracking (simulates real validator behavior)
        self.miners: Dict[str, SimulatedMiner] = {}
        self.miner_counter = 0
        
        # Performance tracking
        self.total_requests = 0
        self.total_validations = 0
        self.total_cooldowns = 0
        self.total_violations = 0
        
        # Rate limiting
        self.request_times = deque(maxlen=1000)
        self.concurrent_requests = 0
        
        # Stress testing
        self.stress_test_mode = config.get('enable_stress_testing', True)
        self.failure_mode = config.get('enable_random_failures', True)
        
        logger.info("🚀 Validation Simulator initialized")
    
    def get_or_create_miner(self, hotkey: str) -> SimulatedMiner:
        """Get or create miner instance"""
        if hotkey not in self.miners:
            self.miner_counter += 1
            self.miners[hotkey] = SimulatedMiner(
                hotkey=hotkey,
                uid=self.miner_counter
            )
            logger.info(f"🆕 Created new miner: {hotkey} (UID: {self.miner_counter})")
        
        return self.miners[hotkey]
    
    def check_rate_limit(self) -> bool:
        """Check if request is within rate limits"""
        current_time = time.time()
        
        # Remove old requests (older than 1 minute)
        while self.request_times and current_time - self.request_times[0] > 60:
            self.request_times.popleft()
        
        # Check rate limit
        if len(self.request_times) >= self.config['rate_limit_per_minute']:
            return False
        
        # Check concurrent limit
        if self.concurrent_requests >= self.config['max_concurrent_requests']:
            return False
        
        return True
    
    def simulate_network_issues(self) -> bool:
        """Simulate random network issues"""
        if not self.config.get('enable_network_issues', True):
            return False
        
        return random.random() < self.config.get('network_issue_rate', 0.02)
    
    def simulate_validation_failure(self) -> bool:
        """Simulate random validation failures"""
        if not self.config.get('enable_random_failures', True):
            return False
        
        return random.random() < self.config.get('failure_rate', 0.05)
    
    def add_response_delay(self) -> float:
        """Add realistic response delay"""
        delay_range = self.config.get('response_delay_range', (0.1, 2.0))
        delay = random.uniform(*delay_range)
        return delay
    
    def get_traffic_specific_cooldown(self, traffic_type: str) -> int:
        """Get cooldown based on traffic type (matching real subnet)"""
        if traffic_type == 'organic':
            return self.config['organic_traffic_cooldown']
        else:
            return self.config['synthetic_traffic_cooldown']
    
    async def validate_txt_to_3d_ply(self, prompt: str, data: str, compression: int = 2, 
                                    generate_preview: bool = False, preview_score_threshold: float = 0.5) -> Dict:
        """Simulate validation endpoint (matching real subnet)"""
        try:
            # Simulate network issues
            if self.simulate_network_issues():
                raise HTTPException(status_code=503, detail="Network timeout")
            
            # Add realistic delay
            delay = self.add_response_delay()
            await asyncio.sleep(delay)
            
            # Simulate validation failure
            if self.simulate_validation_failure():
                raise HTTPException(status_code=500, detail="Validation service error")
            
            # Generate realistic validation scores
            base_score = random.uniform(0.3, 0.95)  # Realistic score range
            
            # Adjust score based on prompt quality
            if len(prompt) < 10:
                base_score *= 0.7  # Penalize short prompts
            elif len(prompt) > 80:
                base_score *= 0.9  # Slight penalty for very long prompts
            
            # Add some randomness
            final_score = max(0.0, min(1.0, base_score + random.uniform(-0.1, 0.1)))
            
            # Generate validation response
            validation_response = {
                "score": round(final_score, 4),
                "iqa": round(random.uniform(0.4, 0.9), 4),  # Aesthetic score
                "alignment_score": round(random.uniform(0.5, 0.95), 4),  # Prompt alignment
                "ssim": round(random.uniform(0.3, 0.8), 4),  # Structure similarity
                "lpips": round(random.uniform(0.1, 0.6), 4),  # Perceptual similarity
            }
            
            # Add preview if requested
            if generate_preview and final_score >= preview_score_threshold:
                # Simulate base64 preview image
                validation_response["preview"] = "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPhfDwAChwGA60e6kgAAAABJRU5ErkJggg=="
            
            self.total_validations += 1
            logger.info(f"✅ Validation completed: score={final_score:.4f}, prompt='{prompt[:50]}...'")
            
            return validation_response
            
        except Exception as e:
            logger.error(f"❌ Validation failed: {e}")
            raise HTTPException(status_code=500, detail=f"Validation error: {str(e)}")
    
    async def pull_task(self, hotkey: str) -> Dict:
        """Simulate task pull endpoint (matching real subnet)"""
        try:
            # Rate limiting
            if not self.check_rate_limit():
                raise HTTPException(status_code=429, detail="Rate limit exceeded")
            
            self.concurrent_requests += 1
            self.request_times.append(time.time())
            
            try:
                # Get or create miner
                miner = self.get_or_create_miner(hotkey)
                
                # Check if miner is on cooldown
                if miner.is_on_cooldown():
                    miner.cooldown_violations += 1
                    remaining_cooldown = miner.cooldown_left()
                    
                    logger.warning(
                        f"[{miner.uid}] asked for task while on cooldown "
                        f"({remaining_cooldown}s left). Total violations: {miner.cooldown_violations}"
                    )
                    
                    # Apply penalty if violations exceed threshold
                    if miner.cooldown_violations > self.config['cooldown_violations_threshold']:
                        penalty = self.config['cooldown_violation_penalty']
                        miner.cooldown_until += penalty
                        logger.error(f"[{miner.uid}] Cooldown penalty added: +{penalty}s")
                    
                    # Return cooldown info
                    return {
                        "task": None,
                        "cooldown_until": miner.cooldown_until,
                        "cooldown_violations": miner.cooldown_violations,
                        "throttle_period": self.config['throttle_period'],
                        "validation_threshold": self.config['quality_threshold']
                    }
                
                # Check if miner has assigned task
                if miner.assigned_task:
                    logger.warning(f"[{miner.uid}] asked for task while having assigned task")
                    return {
                        "task": miner.assigned_task,
                        "cooldown_until": 0,
                        "cooldown_violations": miner.cooldown_violations,
                        "throttle_period": self.config['throttle_period'],
                        "validation_threshold": self.config['quality_threshold']
                    }
                
                # Generate new task
                prompt = self.prompt_generator.generate_random_prompt()
                traffic_type = self.prompt_generator.detect_traffic_type(prompt)
                expected_cooldown = self.get_traffic_specific_cooldown(traffic_type)
                
                task = {
                    "id": str(uuid.uuid4()),
                    "prompt": prompt,
                    "traffic_type": traffic_type,
                    "expected_cooldown": expected_cooldown
                }
                
                # Assign task to miner
                miner.assign_task(task)
                miner.total_tasks_pulled += 1
                
                logger.info(f"📋 Task assigned to {hotkey}: '{prompt[:50]}...' (traffic: {traffic_type}, cooldown: {expected_cooldown}s)")
                
                return {
                    "task": task,
                    "cooldown_until": 0,
                    "cooldown_violations": miner.cooldown_violations,
                    "throttle_period": self.config['throttle_period'],
                    "validation_threshold": self.config['quality_threshold']
                }
                
            finally:
                self.concurrent_requests -= 1
                
        except Exception as e:
            logger.error(f"❌ Task pull failed: {e}")
            raise HTTPException(status_code=500, detail=f"Task pull error: {str(e)}")
    
    async def submit_results(self, hotkey: str, task_id: str, prompt: str, results: str, 
                           submit_time: int, signature: str) -> Dict:
        """Simulate results submission endpoint (matching real subnet)"""
        try:
            # Rate limiting
            if not self.check_rate_limit():
                raise HTTPException(status_code=429, detail="Rate limit exceeded")
            
            self.concurrent_requests += 1
            self.request_times.append(time.time())
            
            try:
                # Get or create miner
                miner = self.get_or_create_miner(hotkey)
                
                # Check signature (simplified)
                if not signature or len(signature) < 10:
                    logger.warning(f"[{miner.uid}] submitted results with invalid signature")
                    return await self._process_task_failure(miner, 0.0, self.config['cooldown_penalty'])
                
                # Check if miner has assigned task
                if not miner.assigned_task or miner.assigned_task['id'] != task_id:
                    logger.warning(f"[{miner.uid}] submitted results for wrong task")
                    return await self._process_task_failure(miner, 0.0, 0)
                
                # Check if results are empty
                if not results:
                    logger.debug(f"[{miner.uid}] submitted empty results")
                    return await self._process_task_failure(miner, 0.0, 0)
                
                # Simulate validation
                validation_res = await self.validate_txt_to_3d_ply(prompt, results)
                score = validation_res['score']
                
                logger.debug(f"[{miner.uid}] submitted results with score: {score:.4f}")
                
                # Check if score meets threshold
                if score < self.config['quality_threshold']:
                    return await self._process_task_failure(
                        miner, score, self.config['cooldown_penalty']
                    )
                
                # Process successful result
                return await self._process_valid_result(miner, score)
                
            finally:
                self.concurrent_requests -= 1
                
        except Exception as e:
            logger.error(f"❌ Results submission failed: {e}")
            raise HTTPException(status_code=500, detail=f"Submission error: {str(e)}")
    
    async def _process_task_failure(self, miner: SimulatedMiner, score: float, cooldown_penalty: int) -> Dict:
        """Process task failure (matching real subnet logic)"""
        delivery_time = int(time.time() - miner.assignment_time) if miner.assignment_time else 0
        
        # Reset task with cooldown penalty
        miner.reset_task(
            throttle_period=self.config['throttle_period'],
            cooldown=self.get_traffic_specific_cooldown('synthetic') + cooldown_penalty
        )
        
        # Update statistics
        miner.total_tasks_submitted += 1
        self.total_cooldowns += 1
        
        logger.warning(f"[{miner.uid}] Task failed: score={score:.4f}, cooldown={miner.cooldown_until}")
        
        return {
            "feedback": {
                "validation_failed": True,
                "task_fidelity_score": score,
                "average_fidelity_score": miner.average_score,
                "generations_within_the_window": len(miner.observations),
                "current_duel_rating": 0.0,
                "current_miner_reward": 0.0
            },
            "cooldown_until": miner.cooldown_until
        }
    
    async def _process_valid_result(self, miner: SimulatedMiner, score: float) -> Dict:
        """Process valid result (matching real subnet logic)"""
        current_time = int(time.time())
        fidelity_score = min(1.0, score)
        delivery_time = int(current_time - miner.assignment_time) if miner.assignment_time else 0
        
        # Reset task and set cooldown
        miner.reset_task(
            throttle_period=self.config['throttle_period'],
            cooldown=self.get_traffic_specific_cooldown('synthetic')
        )
        
        # Add observation
        miner.add_observation(
            task_finish_time=current_time,
            fidelity_score=fidelity_score
        )
        
        # Update statistics
        miner.last_submit_time = current_time
        miner.total_tasks_submitted += 1
        miner.total_successful_submissions += 1
        
        # Calculate reward (simplified)
        reward = fidelity_score * (1.0 + len(miner.observations) * 0.01)
        
        logger.info(f"[{miner.uid}] Task successful: score={score:.4f}, reward={reward:.4f}")
        
        return {
            "feedback": {
                "validation_failed": False,
                "task_fidelity_score": fidelity_score,
                "average_fidelity_score": miner.average_score,
                "generations_within_the_window": len(miner.observations),
                "current_duel_rating": reward,
                "current_miner_reward": reward
            },
            "cooldown_until": miner.cooldown_until
        }
    
    def get_statistics(self) -> Dict:
        """Get simulation statistics"""
        active_miners = len([m for m in self.miners.values() if m.is_active])
        miners_on_cooldown = len([m for m in self.miners.values() if m.is_on_cooldown()])
        miners_with_violations = len([m for m in self.miners.values() if m.cooldown_violations > 0])
        
        return {
            "total_requests": self.total_requests,
            "total_validations": self.total_validations,
            "total_cooldowns": self.total_cooldowns,
            "total_violations": self.total_violations,
            "active_miners": active_miners,
            "miners_on_cooldown": miners_on_cooldown,
            "miners_with_violations": miners_with_violations,
            "concurrent_requests": self.concurrent_requests,
            "rate_limit_status": {
                "requests_last_minute": len(self.request_times),
                "max_per_minute": self.config['rate_limit_per_minute'],
                "max_concurrent": self.config['max_concurrent_requests']
            }
        }

class ValidationSimulationServer:
    """Main simulation server class"""
    
    def __init__(self, config: Dict = None):
        self.config = config or DEFAULT_CONFIG.copy()
        self.simulator = ValidationSimulator(self.config)
        
        if FASTAPI_AVAILABLE:
            self.app = FastAPI(
                title="Validation Simulation Server",
                description="Simulates subnet 17 validator behavior for stress testing",
                version="1.0.0"
            )
            
            # Add CORS middleware
            self.app.add_middleware(
                CORSMiddleware,
                allow_origins=["*"],
                allow_credentials=True,
                allow_methods=["*"],
                allow_headers=["*"],
            )
            
            # Register routes
            self._register_routes()
            
            logger.info("🚀 Validation Simulation Server initialized")
        else:
            logger.error("❌ FastAPI not available - server cannot start")
            self.app = None
    
    def _register_routes(self):
        """Register API routes"""
        
        @self.app.get("/")
        async def root():
            return {
                "message": "Validation Simulation Server",
                "description": "Simulates subnet 17 validator behavior",
                "endpoints": {
                    "GET /": "This info",
                    "GET /health": "Health check",
                    "GET /stats": "Simulation statistics",
                    "POST /pull_task": "Pull task (simulate validator)",
                    "POST /submit_results": "Submit results (simulate validator)",
                    "POST /validate_txt_to_3d_ply": "Validate results (simulate validation service)"
                }
            }
        
        @self.app.get("/health")
        async def health_check():
            return {
                "status": "healthy",
                "timestamp": time.time(),
                "uptime": time.time() - self.start_time if hasattr(self, 'start_time') else 0
            }
        
        @self.app.get("/stats")
        async def get_statistics():
            return self.simulator.get_statistics()
        
        @self.app.post("/pull_task")
        async def pull_task(request: Request):
            try:
                data = await request.json()
                hotkey = data.get('hotkey', 'unknown_miner')
                
                result = await self.simulator.pull_task(hotkey)
                return result
                
            except Exception as e:
                logger.error(f"❌ Pull task error: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.post("/submit_results")
        async def submit_results(request: Request):
            try:
                data = await request.json()
                hotkey = data.get('hotkey', 'unknown_miner')
                task_id = data.get('task_id', '')
                prompt = data.get('prompt', '')
                results = data.get('results', '')
                submit_time = data.get('submit_time', int(time.time_ns()))
                signature = data.get('signature', '')
                
                result = await self.simulator.submit_results(
                    hotkey, task_id, prompt, results, submit_time, signature
                )
                return result
                
            except Exception as e:
                logger.error(f"❌ Submit results error: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.post("/validate_txt_to_3d_ply")
        async def validate_txt_to_3d_ply(request: Request):
            try:
                data = await request.json()
                prompt = data.get('prompt', '')
                data_content = data.get('data', '')
                compression = data.get('compression', 2)
                generate_preview = data.get('generate_preview', False)
                preview_score_threshold = data.get('preview_score_threshold', 0.5)
                
                result = await self.simulator.validate_txt_to_3d_ply(
                    prompt, data_content, compression, generate_preview, preview_score_threshold
                )
                return result
                
            except Exception as e:
                logger.error(f"❌ Validation error: {e}")
                raise HTTPException(status_code=500, detail=str(e))
    
    async def start(self):
        """Start the simulation server"""
        if not self.app:
            logger.error("❌ Cannot start server - FastAPI not available")
            return
        
        self.start_time = time.time()
        
        # Start uvicorn server
        config = uvicorn.Config(
            self.app,
            host=self.config['host'],
            port=self.config['server_port'],
            log_level="info"
        )
        
        server = uvicorn.Server(config)
        logger.info(f"🚀 Starting Validation Simulation Server on {self.config['host']}:{self.config['server_port']}")
        
        await server.serve()

async def main():
    """Main entry point"""
    # Load config from environment or use defaults
    config = DEFAULT_CONFIG.copy()
    
    # Override with environment variables
    for key in config:
        env_key = f"VALIDATION_SIM_{key.upper()}"
        if env_key in os.environ:
            if isinstance(config[key], bool):
                config[key] = os.environ[env_key].lower() == 'true'
            elif isinstance(config[key], int):
                config[key] = int(os.environ[env_key])
            elif isinstance(config[key], float):
                config[key] = float(os.environ[env_key])
            else:
                config[key] = os.environ[env_key]
    
    # Create and start server
    server = ValidationSimulationServer(config)
    await server.start()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("🛑 Server stopped by user")
    except Exception as e:
        logger.error(f"❌ Server error: {e}")
        traceback.print_exc()





