#!/usr/bin/env python3
"""
TRELLIS Glass/Transparent Object Optimizer

This script automatically optimizes transparent object generation by:
1. Detecting transparent objects in prompts
2. Generating 3D models with different optimization strategies
3. Validating models and analyzing score breakdowns
4. Iteratively improving prompts based on validation feedback
5. Finding optimal parameters for transparent objects

Usage:
    python optimize_trellis_glass.py --prompts glass_prompts.txt --target-score 0.7
    python optimize_trellis_glass.py --single-prompt "crystal wine glass" --iterations 5
"""

import asyncio
import argparse
import json
import time
import requests
import base64
import logging
import random
import traceback
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, asdict
import csv

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('optimize_trellis_glass.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

@dataclass
class OptimizationResult:
    """Results from a single optimization attempt"""
    prompt: str
    optimized_prompt: str
    generation_time: float
    validation_score: float
    quality_score: float
    alignment_score: float
    ssim_score: float
    lpips_score: float
    transparency_analysis: Dict[str, Any]
    optimization_strategy: str
    iteration: int
    seed: int
    
    def to_dict(self):
        return asdict(self)

@dataclass
class OptimizationStrategy:
    """Configuration for different optimization approaches"""
    name: str
    prompt_modifications: List[str]
    trellis_params: Dict[str, Any]
    description: str

class TransparentObjectOptimizer:
    """Advanced optimizer for transparent objects with iterative improvement"""
    
    def __init__(self, generation_server_url: str = "http://localhost:8096", 
                 validation_server_url: str = "http://localhost:10006"):
        self.generation_server_url = generation_server_url
        self.validation_server_url = validation_server_url
        
        # Transparent object detection
        self.transparent_keywords = {
            'gemstone', 'crystal', 'diamond', 'ruby', 'sapphire', 'emerald', 
            'topaz', 'amethyst', 'quartz', 'jade', 'opal', 'pearl',
            'glass', 'glassy', 'transparent', 'translucent', 'clear',
            'see-through', 'crystalline', 'pendant', 'jewelry',
            'wine glass', 'bottle', 'vase', 'sphere', 'prism'
        }
        
        # Optimization strategies
        self.strategies = [
            OptimizationStrategy(
                name="solid_emphasis",
                prompt_modifications=[
                    "solid detailed structure",
                    "well-defined edges",
                    "opaque rendering style",
                    "clear geometric form",
                    "professional photography"
                ],
                trellis_params={
                    'guidance_scale': 4.0,
                    'ss_guidance_strength': 8.5,
                    'ss_sampling_steps': 16,
                    'slat_guidance_strength': 3.5,
                    'slat_sampling_steps': 16
                },
                description="Emphasize solid appearance and structure"
            ),
            OptimizationStrategy(
                name="material_focus",
                prompt_modifications=[
                    "detailed craftsmanship",
                    "intricate design",
                    "precise cut",
                    "sculptural form",
                    "high-quality materials"
                ],
                trellis_params={
                    'guidance_scale': 3.8,
                    'ss_guidance_strength': 9.0,
                    'ss_sampling_steps': 18,
                    'slat_guidance_strength': 4.0,
                    'slat_sampling_steps': 18
                },
                description="Focus on material properties and craftsmanship"
            ),
            OptimizationStrategy(
                name="photographic_realism",
                prompt_modifications=[
                    "studio lighting",
                    "professional product photography",
                    "sharp focus",
                    "high contrast",
                    "detailed surface texture"
                ],
                trellis_params={
                    'guidance_scale': 4.2,
                    'ss_guidance_strength': 8.0,
                    'ss_sampling_steps': 20,
                    'slat_guidance_strength': 3.8,
                    'slat_sampling_steps': 20
                },
                description="Photographic realism approach"
            ),
            OptimizationStrategy(
                name="geometric_precision",
                prompt_modifications=[
                    "precise geometry",
                    "accurate proportions",
                    "clean lines",
                    "architectural precision",
                    "technical drawing style"
                ],
                trellis_params={
                    'guidance_scale': 4.5,
                    'ss_guidance_strength': 7.5,
                    'ss_sampling_steps': 14,
                    'slat_guidance_strength': 4.2,
                    'slat_sampling_steps': 14
                },
                description="Focus on geometric accuracy and precision"
            ),
            OptimizationStrategy(
                name="adaptive_learning",
                prompt_modifications=[],  # Will be populated based on feedback
                trellis_params={},  # Will be adapted based on results
                description="Adaptive strategy based on previous results"
            )
        ]
        
        self.results_history: List[OptimizationResult] = []
        
    def is_transparent_object(self, prompt: str) -> bool:
        """Detect if prompt describes a transparent/translucent object"""
        prompt_lower = prompt.lower()
        
        for keyword in self.transparent_keywords:
            if keyword in prompt_lower:
                return True
        
        return False
    
    def build_optimized_prompt(self, original_prompt: str, strategy: OptimizationStrategy, 
                             feedback_analysis: Optional[Dict] = None) -> str:
        """Build optimized prompt using strategy and feedback"""
        if not self.is_transparent_object(original_prompt):
            return f"wbgmsst, {original_prompt}, 3D isometric object, accurate, clean, practical, white background"
        
        logger.info(f"🔮 Optimizing transparent object with strategy: {strategy.name}")
        
        # Start with base components
        prompt_parts = ["wbgmsst,", original_prompt]
        
        # Add strategy-specific modifications
        if strategy.name == "adaptive_learning" and feedback_analysis:
            # Use feedback to adapt prompt
            prompt_parts.extend(self._get_adaptive_modifications(feedback_analysis))
        else:
            prompt_parts.extend(strategy.prompt_modifications)
        
        # Add base technical descriptors
        prompt_parts.extend([
            "3D isometric object",
            "accurate proportions",
            "clean design",
            "white background"
        ])
        
        optimized_prompt = ", ".join(prompt_parts)
        logger.info(f"✨ Optimized prompt: '{optimized_prompt}'")
        return optimized_prompt
    
    def _get_adaptive_modifications(self, feedback_analysis: Dict) -> List[str]:
        """Generate adaptive prompt modifications based on feedback analysis"""
        modifications = []
        
        if feedback_analysis.get('quality_penalty', 0) > 0.2:
            modifications.extend([
                "high quality rendering",
                "detailed surface texture",
                "professional lighting"
            ])
        
        if feedback_analysis.get('alignment_penalty', 0) > 0.1:
            modifications.extend([
                "accurate representation",
                "true to description",
                "realistic proportions"
            ])
        
        if feedback_analysis.get('is_transparency_issue', False):
            modifications.extend([
                "solid appearance",
                "opaque rendering",
                "well-defined structure"
            ])
        
        return modifications
    
    async def generate_3d_model(self, prompt: str, strategy: OptimizationStrategy, 
                               seed: Optional[int] = None) -> Tuple[Optional[bytes], float]:
        """Generate 3D model using optimized prompt and strategy parameters"""
        if seed is None:
            seed = random.randint(0, 2**31 - 1)
        
        try:
            start_time = time.time()
            
            # Update server configuration with strategy parameters
            if strategy.trellis_params:
                config_response = requests.post(
                    f"{self.generation_server_url}/config/update/",
                    json=strategy.trellis_params,
                    timeout=30
                )
                if config_response.status_code != 200:
                    logger.warning(f"Failed to update config: {config_response.status_code}")
            
            # Generate model
            response = requests.post(
                f"{self.generation_server_url}/generate/",
                data={
                    'prompt': prompt,
                    'seed': seed,
                    'return_compressed': False  # Get raw PLY for validation
                },
                timeout=300
            )
            
            generation_time = time.time() - start_time
            
            if response.status_code == 200:
                logger.info(f"✅ Generation successful in {generation_time:.2f}s ({len(response.content):,} bytes)")
                return response.content, generation_time
            else:
                logger.error(f"❌ Generation failed: HTTP {response.status_code}")
                return None, generation_time
                
        except Exception as e:
            logger.error(f"❌ Generation exception: {e}")
            return None, 0.0
    
    async def validate_model(self, ply_data: bytes, prompt: str) -> Dict[str, Any]:
        """Validate generated model and return detailed scores"""
        try:
            # Encode PLY data for validation
            encoded_data = base64.b64encode(ply_data).decode('utf-8')
            
            request_data = {
                "prompt": prompt,
                "data": encoded_data,
                "compression": 0,
                "generate_preview": False,
                "preview_score_threshold": 0.8
            }
            
            response = requests.post(
                f"{self.validation_server_url}/validate_txt_to_3d_ply/",
                json=request_data,
                timeout=120
            )
            
            if response.status_code == 200:
                result = response.json()
                logger.info(f"✅ Validation completed - Score: {result.get('score', 0.0):.4f}")
                return result
            else:
                logger.error(f"❌ Validation failed: HTTP {response.status_code}")
                return {"score": 0.0, "iqa": 0.0, "alignment_score": 0.0, "ssim": 0.0, "lpips": 0.0}
                
        except Exception as e:
            logger.error(f"❌ Validation exception: {e}")
            return {"score": 0.0, "iqa": 0.0, "alignment_score": 0.0, "ssim": 0.0, "lpips": 0.0}
    
    def analyze_transparency_score_breakdown(self, validation_result: Dict, 
                                           original_prompt: str) -> Dict[str, Any]:
        """Analyze validation results for transparency-specific issues"""
        analysis = {
            'is_transparency_issue': False,
            'quality_penalty': 0.0,
            'alignment_penalty': 0.0,
            'recommendations': [],
            'optimization_suggestions': []
        }
        
        quality_score = validation_result.get('iqa', 0.0)
        alignment_score = validation_result.get('alignment_score', 0.0)
        total_score = validation_result.get('score', 0.0)
        
        # Detect transparency-related scoring issues
        if quality_score < 0.6 and alignment_score > 0.4:
            analysis['is_transparency_issue'] = True
            analysis['quality_penalty'] = 0.6 - quality_score
            analysis['recommendations'].append("Quality score low despite good alignment - likely transparency issue")
            analysis['optimization_suggestions'].append("increase_structure_emphasis")
        
        if alignment_score < 0.3:
            analysis['alignment_penalty'] = 0.3 - alignment_score
            analysis['recommendations'].append("Alignment score low - CLIP may not recognize transparent object well")
            analysis['optimization_suggestions'].append("improve_semantic_clarity")
        
        if total_score < 0.5 and quality_score > 0.4:
            analysis['recommendations'].append("Consider alternative validation metrics for transparent objects")
            analysis['optimization_suggestions'].append("adjust_validation_approach")
        
        # Specific recommendations based on object type
        if self.is_transparent_object(original_prompt):
            if 'glass' in original_prompt.lower():
                analysis['optimization_suggestions'].append("glass_specific_optimization")
            elif any(gem in original_prompt.lower() for gem in ['sapphire', 'diamond', 'crystal', 'gem']):
                analysis['optimization_suggestions'].append("gemstone_specific_optimization")
        
        return analysis
    
    async def optimize_single_prompt(self, original_prompt: str, target_score: float = 0.7, 
                                   max_iterations: int = 5) -> List[OptimizationResult]:
        """Optimize a single prompt through iterative improvement"""
        logger.info(f"🎯 Starting optimization for: '{original_prompt}' (target: {target_score})")
        
        results = []
        best_score = 0.0
        best_strategy = None
        
        for iteration in range(max_iterations):
            logger.info(f"🔄 Iteration {iteration + 1}/{max_iterations}")
            
            # Choose strategy for this iteration
            if iteration == 0:
                # Start with solid emphasis
                strategy = self.strategies[0]
            elif iteration < len(self.strategies) - 1:
                # Try different predefined strategies
                strategy = self.strategies[iteration]
            else:
                # Use adaptive strategy based on best previous results
                strategy = self.strategies[-1]  # adaptive_learning
                if results:
                    best_result = max(results, key=lambda r: r.validation_score)
                    strategy.prompt_modifications = self._get_adaptive_modifications(
                        best_result.transparency_analysis
                    )
                    strategy.trellis_params = self._adapt_trellis_params(results)
            
            # Build optimized prompt
            feedback_analysis = None
            if results and strategy.name == "adaptive_learning":
                feedback_analysis = results[-1].transparency_analysis
            
            optimized_prompt = self.build_optimized_prompt(
                original_prompt, strategy, feedback_analysis
            )
            
            # Generate 3D model
            seed = random.randint(0, 2**31 - 1)
            ply_data, generation_time = await self.generate_3d_model(
                optimized_prompt, strategy, seed
            )
            
            if ply_data is None:
                logger.error(f"❌ Generation failed for iteration {iteration + 1}")
                continue
            
            # Validate model
            validation_result = await self.validate_model(ply_data, original_prompt)
            validation_score = validation_result.get('score', 0.0)
            
            # Analyze transparency issues
            transparency_analysis = self.analyze_transparency_score_breakdown(
                validation_result, original_prompt
            )
            
            # Create result record
            result = OptimizationResult(
                prompt=original_prompt,
                optimized_prompt=optimized_prompt,
                generation_time=generation_time,
                validation_score=validation_score,
                quality_score=validation_result.get('iqa', 0.0),
                alignment_score=validation_result.get('alignment_score', 0.0),
                ssim_score=validation_result.get('ssim', 0.0),
                lpips_score=validation_result.get('lpips', 0.0),
                transparency_analysis=transparency_analysis,
                optimization_strategy=strategy.name,
                iteration=iteration + 1,
                seed=seed
            )
            
            results.append(result)
            self.results_history.append(result)
            
            logger.info(f"📊 Iteration {iteration + 1} Results:")
            logger.info(f"   Strategy: {strategy.name}")
            logger.info(f"   Score: {validation_score:.4f} (target: {target_score})")
            logger.info(f"   Quality: {result.quality_score:.4f}")
            logger.info(f"   Alignment: {result.alignment_score:.4f}")
            
            # Check if we've reached the target
            if validation_score >= target_score:
                logger.info(f"🎉 Target score achieved! {validation_score:.4f} >= {target_score}")
                break
            
            # Track best score
            if validation_score > best_score:
                best_score = validation_score
                best_strategy = strategy.name
                logger.info(f"🏆 New best score: {validation_score:.4f} with {strategy.name}")
        
        logger.info(f"✅ Optimization complete for '{original_prompt}'")
        logger.info(f"   Best score: {best_score:.4f} (strategy: {best_strategy})")
        
        return results
    
    def _adapt_trellis_params(self, results: List[OptimizationResult]) -> Dict[str, Any]:
        """Adapt TRELLIS parameters based on previous results"""
        if not results:
            return {}
        
        # Find best performing parameters
        best_result = max(results, key=lambda r: r.validation_score)
        
        # Get the strategy that worked best
        best_strategy_name = best_result.optimization_strategy
        best_strategy = next((s for s in self.strategies if s.name == best_strategy_name), None)
        
        if best_strategy and best_strategy.trellis_params:
            # Make small adjustments to the best parameters
            adapted_params = best_strategy.trellis_params.copy()
            
            # If quality is still low, increase guidance
            if best_result.quality_score < 0.6:
                adapted_params['guidance_scale'] = min(5.0, adapted_params.get('guidance_scale', 3.5) + 0.3)
                adapted_params['ss_guidance_strength'] = min(10.0, adapted_params.get('ss_guidance_strength', 7.5) + 0.5)
            
            # If alignment is low, adjust sampling steps
            if best_result.alignment_score < 0.4:
                adapted_params['ss_sampling_steps'] = min(20, adapted_params.get('ss_sampling_steps', 13) + 2)
                adapted_params['slat_sampling_steps'] = min(20, adapted_params.get('slat_sampling_steps', 14) + 2)
            
            return adapted_params
        
        return {}
    
    async def optimize_batch(self, prompts: List[str], target_score: float = 0.7, 
                           max_iterations: int = 3) -> Dict[str, List[OptimizationResult]]:
        """Optimize a batch of prompts"""
        logger.info(f"🚀 Starting batch optimization for {len(prompts)} prompts")
        
        batch_results = {}
        
        for i, prompt in enumerate(prompts):
            logger.info(f"📝 Processing prompt {i + 1}/{len(prompts)}: '{prompt}'")
            
            try:
                results = await self.optimize_single_prompt(prompt, target_score, max_iterations)
                batch_results[prompt] = results
                
                # Log best result for this prompt
                if results:
                    best_result = max(results, key=lambda r: r.validation_score)
                    logger.info(f"✅ Best result for '{prompt}': {best_result.validation_score:.4f}")
                
            except Exception as e:
                logger.error(f"❌ Failed to optimize '{prompt}': {e}")
                traceback.print_exc()
                batch_results[prompt] = []
        
        return batch_results
    
    def save_results(self, results: Dict[str, List[OptimizationResult]], output_file: str):
        """Save optimization results to JSON and CSV files"""
        # Save detailed JSON
        json_file = output_file.replace('.csv', '.json')
        with open(json_file, 'w') as f:
            json_data = {}
            for prompt, prompt_results in results.items():
                json_data[prompt] = [result.to_dict() for result in prompt_results]
            json.dump(json_data, f, indent=2)
        
        logger.info(f"📊 Detailed results saved to: {json_file}")
        
        # Save summary CSV
        with open(output_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                'Original Prompt', 'Best Score', 'Best Strategy', 'Best Quality', 
                'Best Alignment', 'Iterations', 'Improvement', 'Transparency Issue'
            ])
            
            for prompt, prompt_results in results.items():
                if prompt_results:
                    best_result = max(prompt_results, key=lambda r: r.validation_score)
                    first_result = prompt_results[0]
                    improvement = best_result.validation_score - first_result.validation_score
                    
                    writer.writerow([
                        prompt,
                        f"{best_result.validation_score:.4f}",
                        best_result.optimization_strategy,
                        f"{best_result.quality_score:.4f}",
                        f"{best_result.alignment_score:.4f}",
                        len(prompt_results),
                        f"{improvement:.4f}",
                        best_result.transparency_analysis.get('is_transparency_issue', False)
                    ])
        
        logger.info(f"📈 Summary results saved to: {output_file}")

async def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="TRELLIS Transparent Object Optimizer")
    parser.add_argument("--prompts", type=str, help="File containing prompts to optimize (one per line)")
    parser.add_argument("--single-prompt", type=str, help="Single prompt to optimize")
    parser.add_argument("--target-score", type=float, default=0.7, help="Target validation score")
    parser.add_argument("--max-iterations", type=int, default=5, help="Maximum optimization iterations")
    parser.add_argument("--output", type=str, default="glass_optimization_results.csv", help="Output file")
    parser.add_argument("--generation-server", type=str, default="http://localhost:8096", help="Generation server URL")
    parser.add_argument("--validation-server", type=str, default="http://localhost:10006", help="Validation server URL")
    
    args = parser.parse_args()
    
    # Collect prompts
    prompts = []
    if args.single_prompt:
        prompts = [args.single_prompt]
    elif args.prompts:
        with open(args.prompts, 'r') as f:
            prompts = [line.strip() for line in f if line.strip()]
    else:
        # Default transparent object prompts for testing
        prompts = [
            "large sapphire pendant hanging",
            "crystal wine glass",
            "diamond ring with emerald stone",
            "clear glass sphere",
            "translucent jade sculpture",
            "ruby gemstone with faceted cut",
            "transparent acrylic bottle",
            "amethyst crystal cluster"
        ]
    
    logger.info(f"🔮 Starting TRELLIS Glass Optimizer")
    logger.info(f"   Prompts: {len(prompts)}")
    logger.info(f"   Target score: {args.target_score}")
    logger.info(f"   Max iterations: {args.max_iterations}")
    
    # Initialize optimizer
    optimizer = TransparentObjectOptimizer(
        generation_server_url=args.generation_server,
        validation_server_url=args.validation_server
    )
    
    # Run optimization
    try:
        results = await optimizer.optimize_batch(
            prompts, 
            target_score=args.target_score, 
            max_iterations=args.max_iterations
        )
        
        # Save results
        optimizer.save_results(results, args.output)
        
        # Print summary
        logger.info("🎉 Optimization Complete!")
        logger.info("=" * 60)
        
        total_prompts = len(prompts)
        successful_optimizations = sum(1 for prompt_results in results.values() if prompt_results)
        
        logger.info(f"Total prompts processed: {total_prompts}")
        logger.info(f"Successful optimizations: {successful_optimizations}")
        
        if successful_optimizations > 0:
            all_best_scores = []
            for prompt_results in results.values():
                if prompt_results:
                    best_score = max(r.validation_score for r in prompt_results)
                    all_best_scores.append(best_score)
            
            avg_best_score = sum(all_best_scores) / len(all_best_scores)
            target_achieved = sum(1 for score in all_best_scores if score >= args.target_score)
            
            logger.info(f"Average best score: {avg_best_score:.4f}")
            logger.info(f"Target achieved: {target_achieved}/{successful_optimizations} prompts")
            
            # Show top results
            logger.info("\n🏆 Top Results:")
            sorted_results = []
            for prompt, prompt_results in results.items():
                if prompt_results:
                    best_result = max(prompt_results, key=lambda r: r.validation_score)
                    sorted_results.append((prompt, best_result))
            
            sorted_results.sort(key=lambda x: x[1].validation_score, reverse=True)
            
            for i, (prompt, result) in enumerate(sorted_results[:5]):
                logger.info(f"  {i+1}. '{prompt}': {result.validation_score:.4f} ({result.optimization_strategy})")
    
    except KeyboardInterrupt:
        logger.info("🛑 Optimization interrupted by user")
    except Exception as e:
        logger.error(f"❌ Optimization failed: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(main()) 