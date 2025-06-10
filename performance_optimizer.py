#!/usr/bin/env python3
# Performance Optimizer for Subnet 17 Mining
# Dynamically adjusts generation parameters based on system performance and competition

import asyncio
import json
import time
import statistics
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
import aiohttp
import numpy as np

@dataclass
class PerformanceProfile:
    """Performance profile for different system configurations"""
    name: str
    target_generation_time: float
    quality_threshold: float
    flux_inference_steps: int
    hunyuan_inference_steps: int
    guidance_scale: float
    image_resolution: int
    batch_size: int
    memory_optimization: str  # "speed", "memory", "balanced"

@dataclass
class CompetitionMetrics:
    """Metrics about competitor performance"""
    avg_score: float
    top_10_percent_score: float
    avg_generation_time: float
    score_distribution: List[float]
    time_distribution: List[float]
    
@dataclass
class SystemBenchmark:
    """System performance benchmark results"""
    avg_generation_time: float
    peak_memory_usage: float
    successful_generations: int
    failed_generations: int
    average_quality_score: float
    timestamp: float

class PerformanceOptimizer:
    def __init__(self, config_file: str = "optimization_config.json"):
        self.config_file = Path(config_file)
        self.performance_history: List[SystemBenchmark] = []
        self.competition_data: Optional[CompetitionMetrics] = None
        
        # Predefined performance profiles
        self.profiles = {
            "ultra_quality": PerformanceProfile(
                name="ultra_quality",
                target_generation_time=300.0,
                quality_threshold=0.9,
                flux_inference_steps=50,
                hunyuan_inference_steps=100,
                guidance_scale=7.5,
                image_resolution=1024,
                batch_size=1,
                memory_optimization="quality"
            ),
            "balanced": PerformanceProfile(
                name="balanced",
                target_generation_time=120.0,
                quality_threshold=0.8,
                flux_inference_steps=28,
                hunyuan_inference_steps=70,
                guidance_scale=5.0,
                image_resolution=1024,
                batch_size=1,
                memory_optimization="balanced"
            ),
            "speed": PerformanceProfile(
                name="speed",
                target_generation_time=60.0,
                quality_threshold=0.7,
                flux_inference_steps=20,
                hunyuan_inference_steps=50,
                guidance_scale=3.5,
                image_resolution=768,
                batch_size=1,
                memory_optimization="speed"
            ),
            "turbo": PerformanceProfile(
                name="turbo",
                target_generation_time=30.0,
                quality_threshold=0.65,
                flux_inference_steps=12,
                hunyuan_inference_steps=30,
                guidance_scale=2.0,
                image_resolution=512,
                batch_size=1,
                memory_optimization="speed"
            )
        }
        
        self.current_profile = self.profiles["balanced"]
        self.optimization_enabled = True
        
    def load_config(self):
        """Load optimization configuration from file"""
        if self.config_file.exists():
            try:
                with open(self.config_file, 'r') as f:
                    config = json.load(f)
                    
                # Update profiles with saved settings
                if "profiles" in config:
                    for name, profile_data in config["profiles"].items():
                        if name in self.profiles:
                            for key, value in profile_data.items():
                                setattr(self.profiles[name], key, value)
                                
                # Load current profile
                if "current_profile" in config:
                    profile_name = config["current_profile"]
                    if profile_name in self.profiles:
                        self.current_profile = self.profiles[profile_name]
                        
                self.optimization_enabled = config.get("optimization_enabled", True)
                
            except Exception as e:
                print(f"Failed to load optimization config: {e}")
                
    def save_config(self):
        """Save current optimization configuration"""
        config = {
            "profiles": {name: asdict(profile) for name, profile in self.profiles.items()},
            "current_profile": self.current_profile.name,
            "optimization_enabled": self.optimization_enabled,
            "last_updated": time.time()
        }
        
        try:
            with open(self.config_file, 'w') as f:
                json.dump(config, f, indent=2)
        except Exception as e:
            print(f"Failed to save optimization config: {e}")
            
    async def benchmark_current_setup(self, num_tests: int = 5) -> SystemBenchmark:
        """Run performance benchmark with current settings"""
        print(f"Running performance benchmark with {num_tests} tests...")
        
        generation_times = []
        quality_scores = []
        memory_usage = []
        successful = 0
        failed = 0
        
        test_prompts = [
            "A red sports car",
            "A wooden chair",
            "A medieval castle",
            "A cute robot",
            "A tropical flower"
        ]
        
        for i in range(num_tests):
            prompt = test_prompts[i % len(test_prompts)]
            
            try:
                # Generate model
                start_time = time.time()
                
                # Simulate generation call
                async with aiohttp.ClientSession() as session:
                    payload = {"prompt": prompt}
                    async with session.post(
                        "http://127.0.0.1:8093/generate/",
                        data=payload,
                        timeout=300
                    ) as response:
                        if response.status == 200:
                            ply_data = await response.read()
                            generation_time = time.time() - start_time
                            generation_times.append(generation_time)
                            
                            # Get memory usage from headers if available
                            memory_info = response.headers.get("X-Memory-Usage", "0")
                            if memory_info != "0":
                                memory_usage.append(float(memory_info))
                                
                            # Validate locally
                            score = await self.validate_locally(prompt, ply_data)
                            quality_scores.append(score)
                            successful += 1
                            
                            print(f"Test {i+1}/{num_tests}: {generation_time:.2f}s, Score: {score:.3f}")
                        else:
                            failed += 1
                            print(f"Test {i+1}/{num_tests}: Failed ({response.status})")
                            
            except Exception as e:
                failed += 1
                print(f"Test {i+1}/{num_tests}: Exception - {e}")
                
            # Small delay between tests
            await asyncio.sleep(2)
            
        # Calculate benchmark results
        avg_generation_time = statistics.mean(generation_times) if generation_times else 0
        peak_memory = max(memory_usage) if memory_usage else 0
        avg_quality = statistics.mean(quality_scores) if quality_scores else 0
        
        benchmark = SystemBenchmark(
            avg_generation_time=avg_generation_time,
            peak_memory_usage=peak_memory,
            successful_generations=successful,
            failed_generations=failed,
            average_quality_score=avg_quality,
            timestamp=time.time()
        )
        
        self.performance_history.append(benchmark)
        
        print(f"Benchmark Results:")
        print(f"  Average Generation Time: {avg_generation_time:.2f}s")
        print(f"  Average Quality Score: {avg_quality:.3f}")
        print(f"  Success Rate: {successful}/{num_tests}")
        
        return benchmark
        
    async def validate_locally(self, prompt: str, ply_data: bytes) -> float:
        """Validate PLY data locally and return quality score"""
        try:
            import pyspz
            
            # Compress PLY data
            compressed_data = pyspz.compress(ply_data)
            
            # Send to local validation
            import base64
            payload = {
                "prompt": prompt,
                "data": base64.b64encode(compressed_data).decode('utf-8'),
                "compression": 2,
                "data_ver": 0
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    "http://127.0.0.1:8094/validate_txt_to_3d_ply/",
                    json=payload,
                    timeout=60
                ) as response:
                    if response.status == 200:
                        result = await response.json()
                        return result.get("score", 0.0)
                        
        except Exception as e:
            print(f"Local validation failed: {e}")
            
        return 0.0
        
    def analyze_competition(self, competitor_scores: List[float], competitor_times: List[float]) -> CompetitionMetrics:
        """Analyze competitor performance data"""
        if not competitor_scores:
            return None
            
        sorted_scores = sorted(competitor_scores, reverse=True)
        top_10_percent_count = max(1, len(sorted_scores) // 10)
        
        metrics = CompetitionMetrics(
            avg_score=statistics.mean(competitor_scores),
            top_10_percent_score=statistics.mean(sorted_scores[:top_10_percent_count]),
            avg_generation_time=statistics.mean(competitor_times) if competitor_times else 0,
            score_distribution=competitor_scores,
            time_distribution=competitor_times
        )
        
        self.competition_data = metrics
        
        print(f"Competition Analysis:")
        print(f"  Average Score: {metrics.avg_score:.3f}")
        print(f"  Top 10% Score: {metrics.top_10_percent_score:.3f}")
        print(f"  Average Time: {metrics.avg_generation_time:.2f}s")
        
        return metrics
        
    def recommend_profile(self) -> PerformanceProfile:
        """Recommend optimal profile based on system performance and competition"""
        if not self.performance_history:
            return self.profiles["balanced"]
            
        latest_benchmark = self.performance_history[-1]
        
        # If we have competition data, use it for optimization
        if self.competition_data:
            # If our quality is significantly below top performers, prioritize quality
            if latest_benchmark.average_quality_score < self.competition_data.top_10_percent_score - 0.1:
                return self.profiles["ultra_quality"]
            
            # If our quality is competitive but we're slow, prioritize speed
            elif (latest_benchmark.average_quality_score >= self.competition_data.avg_score and 
                  latest_benchmark.avg_generation_time > self.competition_data.avg_generation_time * 1.5):
                return self.profiles["speed"]
        
        # System-based optimization
        success_rate = latest_benchmark.successful_generations / (
            latest_benchmark.successful_generations + latest_benchmark.failed_generations
        )
        
        # If failing frequently, use more conservative settings
        if success_rate < 0.8:
            return self.profiles["speed"]
        
        # If quality is below threshold, prioritize quality
        if latest_benchmark.average_quality_score < self.current_profile.quality_threshold:
            # Step up to higher quality profile
            if self.current_profile.name == "turbo":
                return self.profiles["speed"]
            elif self.current_profile.name == "speed":
                return self.profiles["balanced"]
            elif self.current_profile.name == "balanced":
                return self.profiles["ultra_quality"]
        
        # If generation time is too high, prioritize speed
        if latest_benchmark.avg_generation_time > self.current_profile.target_generation_time * 1.2:
            # Step down to faster profile
            if self.current_profile.name == "ultra_quality":
                return self.profiles["balanced"]
            elif self.current_profile.name == "balanced":
                return self.profiles["speed"]
            elif self.current_profile.name == "speed":
                return self.profiles["turbo"]
        
        # Current profile is fine
        return self.current_profile
        
    def apply_profile(self, profile: PerformanceProfile):
        """Apply a performance profile to the generation server"""
        self.current_profile = profile
        
        # Generate configuration for the generation server
        config = {
            "flux_inference_steps": profile.flux_inference_steps,
            "hunyuan_inference_steps": profile.hunyuan_inference_steps,
            "guidance_scale": profile.guidance_scale,
            "image_resolution": profile.image_resolution,
            "batch_size": profile.batch_size,
            "memory_optimization": profile.memory_optimization
        }
        
        # Save to generation server config file
        config_path = Path("generation_server_config.json")
        try:
            with open(config_path, 'w') as f:
                json.dump(config, f, indent=2)
            print(f"Applied profile: {profile.name}")
        except Exception as e:
            print(f"Failed to apply profile: {e}")
            
    async def auto_optimize(self):
        """Run automatic optimization cycle"""
        if not self.optimization_enabled:
            print("Auto-optimization disabled")
            return
            
        print("Starting auto-optimization cycle...")
        
        # Run benchmark with current settings
        benchmark = await self.benchmark_current_setup()
        
        # Get recommendation
        recommended_profile = self.recommend_profile()
        
        # Apply if different from current
        if recommended_profile.name != self.current_profile.name:
            print(f"Switching from {self.current_profile.name} to {recommended_profile.name}")
            self.apply_profile(recommended_profile)
            
            # Re-benchmark with new settings
            await asyncio.sleep(10)  # Wait for server to apply changes
            new_benchmark = await self.benchmark_current_setup(num_tests=3)
            
            # Compare results
            improvement = new_benchmark.average_quality_score - benchmark.average_quality_score
            print(f"Quality change: {improvement:+.3f}")
            
        else:
            print(f"Keeping current profile: {self.current_profile.name}")
            
        # Save current state
        self.save_config()

async def main():
    """Main function for performance optimization"""
    optimizer = PerformanceOptimizer()
    optimizer.load_config()
    
    print("Performance Optimizer for Subnet 17")
    print("===================================")
    
    while True:
        try:
            # Run optimization cycle
            await optimizer.auto_optimize()
            
            # Wait before next cycle (1 hour)
            print("Waiting 1 hour before next optimization cycle...")
            await asyncio.sleep(3600)
            
        except KeyboardInterrupt:
            print("Optimization stopped by user")
            break
        except Exception as e:
            print(f"Optimization error: {e}")
            await asyncio.sleep(300)  # Wait 5 minutes on error

if __name__ == "__main__":
    asyncio.run(main()) 