#!/usr/bin/env python3
"""
Validator Prompt Simulation - Complete Workflow Demo

This script simulates the complete workflow:
1. Text prompt generator creates prompts
2. Prompts are sent to get-prompts service
3. Validators fetch prompts from the service
4. 3D models are generated based on prompts
5. Models are validated using the validation engine

Usage:
    python validator_prompt_simulation.py [--config-file CONFIG_FILE] [--simulate-validators N]
"""

import asyncio
import json
import random
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import argparse
import logging
import aiohttp
import pybase64
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor
import threading

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class SimulationConfig:
    """Configuration for the simulation"""
    # Text prompt generator settings
    prompt_generator_config: str = "text-prompt-generator/configs/generator_config.yml"
    pipeline_config: str = "text-prompt-generator/configs/pipeline_config.yml"
    
    # Get-prompts service settings
    get_prompts_url: str = "http://localhost:8000"
    api_key: str = "test_api_key_123"
    
    # Simulation settings
    num_validators: int = 3
    prompts_per_batch: int = 50
    simulation_duration: int = 300  # 5 minutes
    validation_interval: int = 30   # seconds between validation cycles
    
    # 3D generation settings (mock)
    generation_success_rate: float = 0.8
    generation_time_range: Tuple[float, float] = (2.0, 8.0)
    
    # Validation settings
    validation_success_rate: float = 0.9
    validation_time_range: Tuple[float, float] = (1.0, 3.0)


class MockValidator:
    """Simulates a validator that fetches prompts and generates/validates 3D models"""
    
    def __init__(self, validator_id: int, config: SimulationConfig):
        self.validator_id = validator_id
        self.config = config
        self.hotkey = f"validator_{validator_id}_hotkey"
        self.stats = {
            'prompts_fetched': 0,
            'models_generated': 0,
            'models_validated': 0,
            'total_score': 0.0,
            'start_time': time.time()
        }
        self.current_prompts = []
        
    async def fetch_prompts(self, session: aiohttp.ClientSession) -> List[str]:
        """Fetch prompts from the get-prompts service"""
        try:
            nonce = int(time.time_ns())
            message = f"{nonce}{self.hotkey}"
            # Mock signature for simulation
            signature = pybase64.b64encode(f"mock_signature_{nonce}".encode()).decode()
            
            payload = {
                "hotkey": self.hotkey,
                "nonce": nonce,
                "signature": signature
            }
            
            async with session.get(f"{self.config.get_prompts_url}/get", json=payload) as response:
                if response.status == 200:
                    result = await response.json()
                    prompts = result.get("prompts", [])
                    self.current_prompts = prompts
                    self.stats['prompts_fetched'] += len(prompts)
                    logger.info(f"Validator {self.validator_id} fetched {len(prompts)} prompts")
                    return prompts
                else:
                    logger.warning(f"Validator {self.validator_id} failed to fetch prompts: {response.status}")
                    return []
                    
        except Exception as e:
            logger.error(f"Validator {self.validator_id} error fetching prompts: {e}")
            return []
    
    async def generate_3d_models(self) -> List[Dict]:
        """Simulate 3D model generation based on prompts"""
        if not self.current_prompts:
            return []
        
        models = []
        for prompt in self.current_prompts[:5]:  # Limit to 5 models per cycle
            if random.random() < self.config.generation_success_rate:
                generation_time = random.uniform(*self.config.generation_time_range)
                await asyncio.sleep(generation_time)  # Simulate generation time
                
                model = {
                    'id': f"model_{self.validator_id}_{int(time.time())}",
                    'prompt': prompt,
                    'validator_id': self.validator_id,
                    'generation_time': generation_time,
                    'status': 'generated',
                    'timestamp': time.time()
                }
                models.append(model)
                self.stats['models_generated'] += 1
                
                logger.info(f"Validator {self.validator_id} generated model for: '{prompt[:50]}...'")
            else:
                logger.warning(f"Validator {self.validator_id} failed to generate model for: '{prompt[:50]}...'")
        
        return models
    
    async def validate_models(self, models: List[Dict]) -> List[Dict]:
        """Simulate model validation"""
        validated_models = []
        
        for model in models:
            if random.random() < self.config.validation_success_rate:
                validation_time = random.uniform(*self.config.validation_time_range)
                await asyncio.sleep(validation_time)  # Simulate validation time
                
                # Generate mock validation scores
                alignment_score = random.uniform(0.6, 0.95)
                quality_score = random.uniform(0.5, 0.9)
                ssim_score = random.uniform(0.4, 0.85)
                lpips_score = random.uniform(0.3, 0.8)
                
                # Calculate final score (weighted average)
                final_score = (
                    alignment_score * 0.4 +
                    quality_score * 0.3 +
                    ssim_score * 0.2 +
                    lpips_score * 0.1
                )
                
                model.update({
                    'validation_scores': {
                        'final_score': final_score,
                        'alignment_score': alignment_score,
                        'quality_score': quality_score,
                        'ssim_score': ssim_score,
                        'lpips_score': lpips_score
                    },
                    'validation_time': validation_time,
                    'status': 'validated'
                })
                
                validated_models.append(model)
                self.stats['models_validated'] += 1
                self.stats['total_score'] += final_score
                
                logger.info(f"Validator {self.validator_id} validated model: {final_score:.3f}")
            else:
                model['status'] = 'validation_failed'
                logger.warning(f"Validator {self.validator_id} failed to validate model")
        
        return validated_models
    
    async def run_validation_cycle(self, session: aiohttp.ClientSession):
        """Run a complete validation cycle"""
        logger.info(f"Validator {self.validator_id} starting validation cycle")
        
        # Fetch prompts
        prompts = await self.fetch_prompts(session)
        if not prompts:
            logger.warning(f"Validator {self.validator_id} no prompts available")
            return
        
        # Generate 3D models
        models = await self.generate_3d_models()
        if not models:
            logger.warning(f"Validator {self.validator_id} no models generated")
            return
        
        # Validate models
        validated_models = await self.validate_models(models)
        
        # Log cycle completion
        logger.info(f"Validator {self.validator_id} completed cycle: {len(validated_models)} models validated")
        
        return validated_models


class PromptGeneratorSimulator:
    """Simulates the text prompt generator"""
    
    def __init__(self, config: SimulationConfig):
        self.config = config
        self.generated_prompts = []
        self.categories = [
            "robots", "animals", "fantasy characters", "science fiction characters",
            "monsters and mythical creatures", "statues", "accessories", "plants",
            "rocks and minerals", "vehicles", "weapons", "armor"
        ]
    
    def generate_prompts(self, count: int) -> List[str]:
        """Generate mock prompts for simulation"""
        prompts = []
        
        for i in range(count):
            category = random.choice(self.categories)
            # Generate diverse prompts based on category
            if category == "robots":
                prompt = f"mechanical {category} with {random.choice(['steel', 'chrome', 'copper'])} plating"
            elif category == "animals":
                prompt = f"majestic {category} in {random.choice(['natural', 'dramatic', 'peaceful'])} pose"
            elif category == "fantasy characters":
                prompt = f"ethereal {category} with {random.choice(['magical', 'mystical', 'enchanted'])} aura"
            else:
                prompt = f"detailed {category} with {random.choice(['intricate', 'elaborate', 'complex'])} design"
            
            prompts.append(prompt)
        
        self.generated_prompts.extend(prompts)
        logger.info(f"Generated {count} new prompts (total: {len(self.generated_prompts)})")
        return prompts
    
    async def submit_prompts_to_service(self, prompts: List[str], session: aiohttp.ClientSession):
        """Submit generated prompts to the get-prompts service"""
        try:
            payload = {"prompts": prompts}
            headers = {"X-Api-Key": self.config.api_key}
            
            async with session.post(
                f"{self.config.get_prompts_url}/submit",
                json=payload,
                headers=headers
            ) as response:
                if response.status == 200:
                    logger.info(f"Successfully submitted {len(prompts)} prompts to service")
                else:
                    logger.warning(f"Failed to submit prompts: {response.status}")
                    
        except Exception as e:
            logger.error(f"Error submitting prompts: {e}")


class SimulationOrchestrator:
    """Orchestrates the entire simulation"""
    
    def __init__(self, config: SimulationConfig):
        self.config = config
        self.prompt_generator = PromptGeneratorSimulator(config)
        self.validators = [
            MockValidator(i, config) for i in range(config.num_validators)
        ]
        self.simulation_stats = {
            'total_prompts_generated': 0,
            'total_models_generated': 0,
            'total_models_validated': 0,
            'average_validation_score': 0.0,
            'start_time': time.time()
        }
    
    async def run_prompt_generation_cycle(self, session: aiohttp.ClientSession):
        """Run prompt generation and submission cycle"""
        logger.info("Starting prompt generation cycle")
        
        # Generate new prompts
        new_prompts = self.prompt_generator.generate_prompts(self.config.prompts_per_batch)
        self.simulation_stats['total_prompts_generated'] += len(new_prompts)
        
        # Submit to get-prompts service
        await self.prompt_generator.submit_prompts_to_service(new_prompts, session)
        
        logger.info(f"Prompt generation cycle completed: {len(new_prompts)} prompts")
    
    async def run_validator_cycles(self, session: aiohttp.ClientSession):
        """Run validation cycles for all validators"""
        logger.info("Starting validator validation cycles")
        
        # Run validators concurrently
        tasks = [
            validator.run_validation_cycle(session)
            for validator in self.validators
        ]
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Process results and update stats
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"Validator {i} encountered error: {result}")
            elif result:
                self.simulation_stats['total_models_generated'] += len(result)
                self.simulation_stats['total_models_validated'] += len(result)
                
                # Calculate average score
                scores = [m['validation_scores']['final_score'] for m in result]
                if scores:
                    avg_score = sum(scores) / len(scores)
                    self.simulation_stats['average_validation_score'] = (
                        (self.simulation_stats['average_validation_score'] + avg_score) / 2
                    )
        
        logger.info("Validator cycles completed")
    
    async def run_simulation(self):
        """Run the complete simulation"""
        logger.info(f"Starting simulation with {self.config.num_validators} validators")
        logger.info(f"Simulation duration: {self.config.simulation_duration} seconds")
        
        start_time = time.time()
        cycle_count = 0
        
        async with aiohttp.ClientSession() as session:
            while time.time() - start_time < self.config.simulation_duration:
                cycle_count += 1
                logger.info(f"\n=== Simulation Cycle {cycle_count} ===")
                
                # Run prompt generation
                await self.run_prompt_generation_cycle(session)
                
                # Wait for prompts to be available
                await asyncio.sleep(5)
                
                # Run validator cycles
                await self.run_validator_cycles(session)
                
                # Wait before next cycle
                await asyncio.sleep(self.config.validation_interval)
                
                # Print current stats
                self.print_simulation_stats()
        
        # Print final results
        self.print_final_results()
    
    def print_simulation_stats(self):
        """Print current simulation statistics"""
        elapsed = time.time() - self.simulation_stats['start_time']
        
        print(f"\n📊 Simulation Statistics (Cycle {elapsed:.0f}s)")
        print(f"   Prompts Generated: {self.simulation_stats['total_prompts_generated']}")
        print(f"   Models Generated: {self.simulation_stats['total_models_generated']}")
        print(f"   Models Validated: {self.simulation_stats['total_models_validated']}")
        print(f"   Avg Validation Score: {self.simulation_stats['average_validation_score']:.3f}")
        
        # Print validator stats
        for validator in self.validators:
            print(f"   Validator {validator.validator_id}: "
                  f"{validator.stats['models_validated']} models, "
                  f"avg score: {validator.stats['total_score']/max(validator.stats['models_validated'], 1):.3f}")
    
    def print_final_results(self):
        """Print final simulation results"""
        total_time = time.time() - self.simulation_stats['start_time']
        
        print(f"\n🎯 SIMULATION COMPLETED")
        print(f"=" * 50)
        print(f"Total Runtime: {total_time:.1f} seconds")
        print(f"Total Prompts Generated: {self.simulation_stats['total_prompts_generated']}")
        print(f"Total Models Generated: {self.simulation_stats['total_models_generated']}")
        print(f"Total Models Validated: {self.simulation_stats['total_models_validated']}")
        print(f"Overall Average Validation Score: {self.simulation_stats['average_validation_score']:.3f}")
        print(f"Prompts per Second: {self.simulation_stats['total_prompts_generated']/total_time:.2f}")
        print(f"Models per Second: {self.simulation_stats['total_models_validated']/total_time:.2f}")
        
        # Save results to file
        results = {
            'simulation_config': self.config.__dict__,
            'simulation_stats': self.simulation_stats,
            'validator_stats': [v.stats for v in self.validators],
            'timestamp': time.time()
        }
        
        with open('simulation_results.json', 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\nResults saved to: simulation_results.json")


async def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="Validator Prompt Simulation")
    parser.add_argument("--config-file", help="Path to configuration file")
    parser.add_argument("--simulate-validators", type=int, default=3, help="Number of validators to simulate")
    parser.add_argument("--duration", type=int, default=300, help="Simulation duration in seconds")
    parser.add_argument("--prompts-per-batch", type=int, default=50, help="Prompts generated per cycle")
    
    args = parser.parse_args()
    
    # Create configuration
    config = SimulationConfig(
        num_validators=args.simulate_validators,
        simulation_duration=args.duration,
        prompts_per_batch=args.prompts_per_batch
    )
    
    # Create and run simulation
    orchestrator = SimulationOrchestrator(config)
    
    try:
        await orchestrator.run_simulation()
    except KeyboardInterrupt:
        logger.info("Simulation interrupted by user")
    except Exception as e:
        logger.error(f"Simulation error: {e}")
        raise


if __name__ == "__main__":
    asyncio.run(main())
