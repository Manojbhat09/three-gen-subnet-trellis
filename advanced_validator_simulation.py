#!/usr/bin/env python3
"""
Advanced Validator Simulation with Real Text Prompt Generator Integration

This script provides a more realistic simulation by:
1. Using the actual text-prompt-generator configuration
2. Integrating with the validation engine concepts
3. Providing detailed metrics and analysis
4. Supporting different simulation scenarios

Usage:
    python advanced_validator_simulation.py [--scenario SCENARIO] [--validators N] [--duration SECONDS]
"""

import asyncio
import json
import random
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import argparse
import logging
import aiohttp
import pybase64
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import statistics
from collections import defaultdict, deque

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class AdvancedSimulationConfig:
    """Advanced configuration for the simulation"""
    
    # Service endpoints
    get_prompts_url: str = "http://localhost:8000"
    api_key: str = "test_api_key_123"
    
    # Simulation parameters
    scenario: str = "balanced"  # balanced, high_load, stress_test, learning
    num_validators: int = 5
    simulation_duration: int = 600  # 10 minutes
    validation_interval: int = 45   # seconds between cycles
    
    # Prompt generation settings
    prompts_per_batch: int = 100
    prompt_complexity_distribution: Dict[str, float] = field(default_factory=lambda: {
        "simple": 0.3,      # 3-5 words
        "medium": 0.5,      # 6-8 words  
        "complex": 0.2      # 9+ words
    })
    
    # 3D generation simulation
    generation_success_rates: Dict[str, float] = field(default_factory=lambda: {
        "simple": 0.95,
        "medium": 0.85,
        "complex": 0.65
    })
    
    generation_time_ranges: Dict[str, Tuple[float, float]] = field(default_factory=lambda: {
        "simple": (1.0, 3.0),
        "medium": (3.0, 6.0),
        "complex": (6.0, 12.0)
    })
    
    # Validation simulation
    validation_success_rate: float = 0.92
    validation_time_range: Tuple[float, float] = (0.5, 2.5)
    
    # Quality scoring parameters
    quality_factors: Dict[str, Dict[str, Tuple[float, float]]] = field(default_factory=lambda: {
        "alignment_score": {"mean": 0.75, "std": 0.15},
        "quality_score": {"mean": 0.70, "std": 0.18},
        "ssim_score": {"mean": 0.65, "std": 0.20},
        "lpips_score": {"mean": 0.60, "std": 0.25}
    })
    
    # Learning and adaptation
    enable_learning: bool = True
    learning_rate: float = 0.1
    performance_memory_size: int = 100


class PromptComplexityAnalyzer:
    """Analyzes prompt complexity and estimates generation difficulty"""
    
    def __init__(self):
        self.complexity_patterns = {
            "simple": {
                "max_words": 5,
                "max_attributes": 2,
                "common_patterns": ["adjective noun", "noun with adjective"]
            },
            "medium": {
                "max_words": 8,
                "max_attributes": 4,
                "common_patterns": ["adjective noun with adjective", "noun with adjective and adjective"]
            },
            "complex": {
                "max_words": 15,
                "max_attributes": 6,
                "common_patterns": ["adjective adjective noun with adjective adjective", "complex descriptions"]
            }
        }
    
    def analyze_complexity(self, prompt: str) -> str:
        """Analyze prompt complexity and return category"""
        words = prompt.split()
        word_count = len(words)
        
        # Count descriptive elements
        descriptive_elements = sum(1 for word in words if word in [
            "detailed", "intricate", "elaborate", "complex", "majestic", "ethereal",
            "mechanical", "magical", "mystical", "enchanted", "steel", "chrome"
        ])
        
        if word_count <= 5 and descriptive_elements <= 2:
            return "simple"
        elif word_count <= 8 and descriptive_elements <= 4:
            return "medium"
        else:
            return "complex"
    
    def estimate_generation_difficulty(self, prompt: str) -> float:
        """Estimate generation difficulty (0.0 to 1.0)"""
        complexity = self.analyze_complexity(prompt)
        base_difficulty = {"simple": 0.2, "medium": 0.5, "complex": 0.8}[complexity]
        
        # Add randomness for realistic variation
        variation = random.uniform(-0.1, 0.1)
        return max(0.0, min(1.0, base_difficulty + variation))


class QualityScoreGenerator:
    """Generates realistic quality scores based on prompt complexity and other factors"""
    
    def __init__(self, config: AdvancedSimulationConfig):
        self.config = config
        self.quality_factors = config.quality_factors
    
    def generate_scores(self, prompt: str, complexity: str, generation_quality: float) -> Dict[str, float]:
        """Generate realistic validation scores"""
        
        # Base scores from configuration
        base_scores = {}
        for metric, params in self.quality_factors.items():
            base_score = random.normalvariate(params["mean"], params["std"])
            base_scores[metric] = max(0.0, min(1.0, base_score))
        
        # Adjust based on complexity
        complexity_multiplier = {"simple": 1.1, "medium": 1.0, "complex": 0.9}[complexity]
        
        # Adjust based on generation quality
        generation_multiplier = 0.8 + (generation_quality * 0.4)
        
        # Apply adjustments
        adjusted_scores = {}
        for metric, base_score in base_scores.items():
            adjusted_score = base_score * complexity_multiplier * generation_multiplier
            adjusted_scores[metric] = max(0.0, min(1.0, adjusted_score))
        
        # Calculate final score (weighted average)
        weights = {"alignment_score": 0.4, "quality_score": 0.3, "ssim_score": 0.2, "lpips_score": 0.1}
        final_score = sum(adjusted_scores[metric] * weight for metric, weight in weights.items())
        
        adjusted_scores["final_score"] = final_score
        
        return adjusted_scores


class AdvancedValidator:
    """Advanced validator with learning capabilities and realistic behavior"""
    
    def __init__(self, validator_id: int, config: AdvancedSimulationConfig):
        self.validator_id = validator_id
        self.config = config
        self.hotkey = f"validator_{validator_id}_hotkey"
        
        # Performance tracking
        self.stats = {
            'prompts_fetched': 0,
            'models_generated': 0,
            'models_validated': 0,
            'total_score': 0.0,
            'start_time': time.time(),
            'generation_success_rate': 0.0,
            'validation_success_rate': 0.0,
            'average_quality_score': 0.0
        }
        
        # Learning and adaptation
        self.performance_history = deque(maxlen=config.performance_memory_size)
        self.complexity_preferences = defaultdict(int)
        self.current_prompts = []
        
        # Prompt complexity analyzer
        self.complexity_analyzer = PromptComplexityAnalyzer()
        
        # Quality score generator
        self.quality_generator = QualityScoreGenerator(config)
    
    def update_performance_metrics(self, new_results: List[Dict]):
        """Update performance metrics based on new results"""
        if not new_results:
            return
        
        # Update success rates
        successful_generations = sum(1 for r in new_results if r.get('status') == 'generated')
        successful_validations = sum(1 for r in new_results if r.get('status') == 'validated')
        
        if self.stats['models_generated'] > 0:
            self.stats['generation_success_rate'] = (
                (self.stats['generation_success_rate'] * 0.9) + 
                (successful_generations / len(new_results) * 0.1)
            )
        
        if self.stats['models_validated'] > 0:
            self.stats['validation_success_rate'] = (
                (self.stats['validation_success_rate'] * 0.9) + 
                (successful_validations / len(new_results) * 0.1)
            )
        
        # Update average quality score
        quality_scores = [r.get('validation_scores', {}).get('final_score', 0) for r in new_results if r.get('validation_scores')]
        if quality_scores:
            new_avg = statistics.mean(quality_scores)
            self.stats['average_quality_score'] = (
                (self.stats['average_quality_score'] * 0.9) + (new_avg * 0.1)
            )
        
        # Store performance data for learning
        self.performance_history.append({
            'timestamp': time.time(),
            'generation_success_rate': self.stats['generation_success_rate'],
            'validation_success_rate': self.stats['validation_success_rate'],
            'average_quality_score': self.stats['average_quality_score']
        })
    
    async def fetch_prompts(self, session: aiohttp.ClientSession) -> List[str]:
        """Fetch prompts from the get-prompts service"""
        try:
            nonce = int(time.time_ns())
            message = f"{nonce}{self.hotkey}"
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
                    
                    # Analyze complexity preferences
                    for prompt in prompts:
                        complexity = self.complexity_analyzer.analyze_complexity(prompt)
                        self.complexity_preferences[complexity] += 1
                    
                    logger.info(f"Validator {self.validator_id} fetched {len(prompts)} prompts")
                    return prompts
                else:
                    logger.warning(f"Validator {self.validator_id} failed to fetch prompts: {response.status}")
                    return []
                    
        except Exception as e:
            logger.error(f"Validator {self.validator_id} error fetching prompts: {e}")
            return []
    
    async def generate_3d_models(self) -> List[Dict]:
        """Generate 3D models with realistic complexity-based behavior"""
        if not self.current_prompts:
            return []
        
        models = []
        # Limit models per cycle based on validator performance
        max_models = min(5, max(1, int(self.stats['generation_success_rate'] * 8)))
        
        for prompt in self.current_prompts[:max_models]:
            complexity = self.complexity_analyzer.analyze_complexity(prompt)
            difficulty = self.complexity_analyzer.estimate_generation_difficulty(prompt)
            
            # Success rate based on complexity and validator performance
            base_success_rate = self.config.generation_success_rates[complexity]
            validator_skill = self.stats['generation_success_rate']
            adjusted_success_rate = base_success_rate * (0.7 + validator_skill * 0.3)
            
            if random.random() < adjusted_success_rate:
                # Generation time based on complexity
                time_range = self.config.generation_time_ranges[complexity]
                generation_time = random.uniform(*time_range)
                
                # Add some randomness based on validator skill
                skill_variation = random.uniform(0.8, 1.2) if validator_skill > 0.8 else random.uniform(1.0, 1.5)
                generation_time *= skill_variation
                
                await asyncio.sleep(generation_time)
                
                # Estimate generation quality
                generation_quality = max(0.0, min(1.0, 1.0 - difficulty + (validator_skill * 0.3)))
                
                model = {
                    'id': f"model_{self.validator_id}_{int(time.time())}",
                    'prompt': prompt,
                    'validator_id': self.validator_id,
                    'complexity': complexity,
                    'difficulty': difficulty,
                    'generation_time': generation_time,
                    'generation_quality': generation_quality,
                    'status': 'generated',
                    'timestamp': time.time()
                }
                models.append(model)
                self.stats['models_generated'] += 1
                
                logger.info(f"Validator {self.validator_id} generated {complexity} model: '{prompt[:50]}...' (quality: {generation_quality:.2f})")
            else:
                logger.warning(f"Validator {self.validator_id} failed to generate {complexity} model: '{prompt[:50]}...'")
        
        return models
    
    async def validate_models(self, models: List[Dict]) -> List[Dict]:
        """Validate models with realistic quality scoring"""
        validated_models = []
        
        for model in models:
            # Validation success based on model quality and validator skill
            base_validation_success = self.config.validation_success_rate
            model_quality_factor = model.get('generation_quality', 0.5)
            validator_skill = self.stats['validation_success_rate']
            
            adjusted_validation_success = base_validation_success * (0.8 + model_quality_factor * 0.2) * (0.9 + validator_skill * 0.1)
            
            if random.random() < adjusted_validation_success:
                validation_time = random.uniform(*self.config.validation_time_range)
                await asyncio.sleep(validation_time)
                
                # Generate quality scores
                quality_scores = self.quality_generator.generate_scores(
                    model['prompt'], 
                    model['complexity'], 
                    model['generation_quality']
                )
                
                model.update({
                    'validation_scores': quality_scores,
                    'validation_time': validation_time,
                    'status': 'validated'
                })
                
                validated_models.append(model)
                self.stats['models_validated'] += 1
                self.stats['total_score'] += quality_scores['final_score']
                
                logger.info(f"Validator {self.validator_id} validated {model['complexity']} model: {quality_scores['final_score']:.3f}")
            else:
                model['status'] = 'validation_failed'
                logger.warning(f"Validator {self.validator_id} failed to validate {model['complexity']} model")
        
        return validated_models
    
    async def run_validation_cycle(self, session: aiohttp.ClientSession):
        """Run a complete validation cycle with learning"""
        logger.info(f"Validator {self.validator_id} starting validation cycle")
        
        # Fetch prompts
        prompts = await self.fetch_prompts(session)
        if not prompts:
            logger.warning(f"Validator {self.validator_id} no prompts available")
            return []
        
        # Generate 3D models
        models = await self.generate_3d_models()
        if not models:
            logger.warning(f"Validator {self.validator_id} no models generated")
            return []
        
        # Validate models
        validated_models = await self.validate_models(models)
        
        # Update performance metrics
        self.update_performance_metrics(models)
        
        # Log cycle completion
        logger.info(f"Validator {self.validator_id} completed cycle: {len(validated_models)} models validated")
        
        return validated_models


class AdvancedPromptGenerator:
    """Advanced prompt generator with category-based generation"""
    
    def __init__(self, config: AdvancedSimulationConfig):
        self.config = config
        self.generated_prompts = []
        self.prompt_categories = {
            "robots": {
                "templates": [
                    "{material} {type} robot with {feature}",
                    "{adjective} {type} android with {feature}",
                    "mechanical {type} with {material} {feature}"
                ],
                "materials": ["steel", "chrome", "copper", "titanium", "aluminum"],
                "types": ["humanoid", "quadruped", "flying", "underwater", "space"],
                "features": ["glowing eyes", "articulated joints", "solar panels", "holographic displays"]
            },
            "animals": {
                "templates": [
                    "{adjective} {animal} in {pose} pose",
                    "{animal} with {feature} in {setting}",
                    "majestic {animal} with {adjective} {feature}"
                ],
                "animals": ["lion", "eagle", "dragon", "unicorn", "phoenix", "griffin"],
                "poses": ["natural", "dramatic", "peaceful", "hunting", "flying", "resting"],
                "features": ["mane", "wings", "horns", "scales", "feathers"],
                "settings": ["forest", "mountain", "desert", "ocean", "sky"]
            },
            "fantasy": {
                "templates": [
                    "{adjective} {character} with {feature}",
                    "{character} wearing {clothing} with {accessory}",
                    "ethereal {character} with {magical} aura"
                ],
                "characters": ["elf", "wizard", "knight", "dragon rider", "mage"],
                "clothing": ["robes", "armor", "leather", "silk", "chainmail"],
                "accessories": ["staff", "sword", "crown", "amulet", "ring"],
                "magical": ["magical", "mystical", "enchanted", "ethereal", "divine"]
            }
        }
    
    def generate_prompts(self, count: int) -> List[str]:
        """Generate diverse prompts based on categories and complexity"""
        prompts = []
        
        for i in range(count):
            # Select category
            category = random.choice(list(self.prompt_categories.keys()))
            category_data = self.prompt_categories[category]
            
            # Select template
            template = random.choice(category_data["templates"])
            
            # Fill template with random values
            prompt = template
            for key, values in category_data.items():
                if key != "templates":
                    if isinstance(values, list):
                        prompt = prompt.replace(f"{{{key}}}", random.choice(values))
            
            # Ensure prompt meets complexity requirements
            complexity = self._analyze_complexity(prompt)
            target_complexity = random.choices(
                list(self.config.prompt_complexity_distribution.keys()),
                weights=list(self.config.prompt_complexity_distribution.values())
            )[0]
            
            # Adjust prompt to match target complexity
            prompt = self._adjust_complexity(prompt, target_complexity)
            
            prompts.append(prompt)
        
        self.generated_prompts.extend(prompts)
        logger.info(f"Generated {count} new prompts (total: {len(self.generated_prompts)})")
        return prompts
    
    def _analyze_complexity(self, prompt: str) -> str:
        """Analyze prompt complexity"""
        words = prompt.split()
        if len(words) <= 5:
            return "simple"
        elif len(words) <= 8:
            return "medium"
        else:
            return "complex"
    
    def _adjust_complexity(self, prompt: str, target: str) -> str:
        """Adjust prompt complexity to match target"""
        current = self._analyze_complexity(prompt)
        
        if current == target:
            return prompt
        
        if target == "simple" and current in ["medium", "complex"]:
            # Simplify by removing some adjectives
            words = prompt.split()
            if len(words) > 5:
                # Keep core structure, remove some modifiers
                core_words = [words[0], words[-1]]  # Keep first and last
                if len(words) > 2:
                    core_words.insert(1, words[len(words)//2])  # Keep middle
                return " ".join(core_words)
        
        elif target == "complex" and current in ["simple", "medium"]:
            # Add complexity
            enhancements = [
                "with intricate details",
                "featuring elaborate design",
                "showing remarkable craftsmanship",
                "displaying exceptional quality"
            ]
            return f"{prompt} {random.choice(enhancements)}"
        
        return prompt
    
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


class AdvancedSimulationOrchestrator:
    """Advanced simulation orchestrator with detailed analytics"""
    
    def __init__(self, config: AdvancedSimulationConfig):
        self.config = config
        self.prompt_generator = AdvancedPromptGenerator(config)
        self.validators = [
            AdvancedValidator(i, config) for i in range(config.num_validators)
        ]
        
        # Comprehensive statistics
        self.simulation_stats = {
            'total_prompts_generated': 0,
            'total_models_generated': 0,
            'total_models_validated': 0,
            'average_validation_score': 0.0,
            'start_time': time.time(),
            'complexity_distribution': defaultdict(int),
            'quality_trends': deque(maxlen=100),
            'performance_metrics': defaultdict(list)
        }
        
        # Scenario-specific behavior
        self._setup_scenario()
    
    def _setup_scenario(self):
        """Configure simulation based on selected scenario"""
        if self.config.scenario == "high_load":
            self.config.validation_interval = 20
            self.config.prompts_per_batch = 150
        elif self.config.scenario == "stress_test":
            self.config.validation_interval = 15
            self.config.prompts_per_batch = 200
        elif self.config.scenario == "learning":
            self.config.validation_interval = 60
            self.config.prompts_per_batch = 75
    
    async def run_prompt_generation_cycle(self, session: aiohttp.ClientSession):
        """Run advanced prompt generation cycle"""
        logger.info("Starting advanced prompt generation cycle")
        
        # Generate new prompts
        new_prompts = self.prompt_generator.generate_prompts(self.config.prompts_per_batch)
        self.simulation_stats['total_prompts_generated'] += len(new_prompts)
        
        # Analyze complexity distribution
        for prompt in new_prompts:
            complexity = self._analyze_prompt_complexity(prompt)
            self.simulation_stats['complexity_distribution'][complexity] += 1
        
        # Submit to get-prompts service
        await self.prompt_generator.submit_prompts_to_service(new_prompts, session)
        
        logger.info(f"Advanced prompt generation cycle completed: {len(new_prompts)} prompts")
    
    def _analyze_prompt_complexity(self, prompt: str) -> str:
        """Analyze prompt complexity"""
        words = prompt.split()
        if len(words) <= 5:
            return "simple"
        elif len(words) <= 8:
            return "medium"
        else:
            return "complex"
    
    async def run_validator_cycles(self, session: aiohttp.ClientSession):
        """Run advanced validation cycles"""
        logger.info("Starting advanced validator validation cycles")
        
        # Run validators concurrently
        tasks = [
            validator.run_validation_cycle(session)
            for validator in self.validators
        ]
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Process results and update comprehensive stats
        all_models = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"Validator {i} encountered error: {result}")
            elif result:
                all_models.extend(result)
                self.simulation_stats['total_models_generated'] += len(result)
                self.simulation_stats['total_models_validated'] += len(result)
        
        # Update quality trends
        if all_models:
            scores = [m.get('validation_scores', {}).get('final_score', 0) for m in all_models if m.get('validation_scores')]
            if scores:
                avg_score = statistics.mean(scores)
                self.simulation_stats['average_validation_score'] = avg_score
                self.simulation_stats['quality_trends'].append({
                    'timestamp': time.time(),
                    'average_score': avg_score,
                    'models_count': len(scores)
                })
        
        logger.info("Advanced validator cycles completed")
    
    async def run_simulation(self):
        """Run the complete advanced simulation"""
        logger.info(f"Starting advanced simulation: {self.config.scenario} scenario")
        logger.info(f"Validators: {self.config.num_validators}, Duration: {self.config.simulation_duration}s")
        
        start_time = time.time()
        cycle_count = 0
        
        async with aiohttp.ClientSession() as session:
            while time.time() - start_time < self.config.simulation_duration:
                cycle_count += 1
                logger.info(f"\n=== Advanced Simulation Cycle {cycle_count} ===")
                
                # Run prompt generation
                await self.run_prompt_generation_cycle(session)
                
                # Wait for prompts to be available
                await asyncio.sleep(5)
                
                # Run validator cycles
                await self.run_validator_cycles(session)
                
                # Wait before next cycle
                await asyncio.sleep(self.config.validation_interval)
                
                # Print detailed stats
                self.print_detailed_stats()
        
        # Print final results
        self.print_final_results()
    
    def print_detailed_stats(self):
        """Print detailed simulation statistics"""
        elapsed = time.time() - self.simulation_stats['start_time']
        
        print(f"\n📊 Advanced Simulation Statistics (Cycle {elapsed:.0f}s)")
        print(f"   Prompts Generated: {self.simulation_stats['total_prompts_generated']}")
        print(f"   Models Generated: {self.simulation_stats['total_models_generated']}")
        print(f"   Models Validated: {self.simulation_stats['total_models_validated']}")
        print(f"   Avg Validation Score: {self.simulation_stats['average_validation_score']:.3f}")
        
        # Complexity distribution
        print(f"   Complexity Distribution:")
        for complexity, count in self.simulation_stats['complexity_distribution'].items():
            print(f"     {complexity.capitalize()}: {count}")
        
        # Validator performance
        print(f"   Validator Performance:")
        for validator in self.validators:
            print(f"     Validator {validator.validator_id}: "
                  f"Gen: {validator.stats['generation_success_rate']:.2f}, "
                  f"Val: {validator.stats['validation_success_rate']:.2f}, "
                  f"Quality: {validator.stats['average_quality_score']:.3f}")
    
    def print_final_results(self):
        """Print comprehensive final results"""
        total_time = time.time() - self.simulation_stats['start_time']
        
        print(f"\n🎯 ADVANCED SIMULATION COMPLETED")
        print(f"=" * 60)
        print(f"Scenario: {self.config.scenario}")
        print(f"Total Runtime: {total_time:.1f} seconds")
        print(f"Total Prompts Generated: {self.simulation_stats['total_prompts_generated']}")
        print(f"Total Models Generated: {self.simulation_stats['total_models_generated']}")
        print(f"Total Models Validated: {self.simulation_stats['total_models_validated']}")
        print(f"Overall Average Validation Score: {self.simulation_stats['average_validation_score']:.3f}")
        
        # Performance metrics
        print(f"\n📈 Performance Metrics:")
        print(f"   Prompts per Second: {self.simulation_stats['total_prompts_generated']/total_time:.2f}")
        print(f"   Models per Second: {self.simulation_stats['total_models_validated']/total_time:.2f}")
        print(f"   Generation Success Rate: {self.simulation_stats['total_models_generated']/max(self.simulation_stats['total_models_generated'], 1):.2%}")
        print(f"   Validation Success Rate: {self.simulation_stats['total_models_validated']/max(self.simulation_stats['total_models_generated'], 1):.2%}")
        
        # Complexity analysis
        print(f"\n🔍 Complexity Analysis:")
        total_prompts = self.simulation_stats['total_prompts_generated']
        for complexity, count in self.simulation_stats['complexity_distribution'].items():
            percentage = (count / total_prompts * 100) if total_prompts > 0 else 0
            print(f"   {complexity.capitalize()}: {count} ({percentage:.1f}%)")
        
        # Save comprehensive results
        results = {
            'simulation_config': self.config.__dict__,
            'simulation_stats': dict(self.simulation_stats),
            'validator_stats': [v.stats for v in self.validators],
            'complexity_distribution': dict(self.simulation_stats['complexity_distribution']),
            'quality_trends': list(self.simulation_stats['quality_trends']),
            'timestamp': time.time()
        }
        
        with open('advanced_simulation_results.json', 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\nResults saved to: advanced_simulation_results.json")


async def main():
    """Main entry point for advanced simulation"""
    parser = argparse.ArgumentParser(description="Advanced Validator Simulation")
    parser.add_argument("--scenario", choices=["balanced", "high_load", "stress_test", "learning"], 
                       default="balanced", help="Simulation scenario")
    parser.add_argument("--validators", type=int, default=5, help="Number of validators")
    parser.add_argument("--duration", type=int, default=600, help="Simulation duration in seconds")
    parser.add_argument("--prompts-per-batch", type=int, default=100, help="Prompts per generation cycle")
    
    args = parser.parse_args()
    
    # Create advanced configuration
    config = AdvancedSimulationConfig(
        scenario=args.scenario,
        num_validators=args.validators,
        simulation_duration=args.duration,
        prompts_per_batch=args.prompts_per_batch
    )
    
    # Create and run advanced simulation
    orchestrator = AdvancedSimulationOrchestrator(config)
    
    try:
        await orchestrator.run_simulation()
    except KeyboardInterrupt:
        logger.info("Advanced simulation interrupted by user")
    except Exception as e:
        logger.error(f"Advanced simulation error: {e}")
        raise


if __name__ == "__main__":
    asyncio.run(main())
