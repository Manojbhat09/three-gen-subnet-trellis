#!/usr/bin/env python3
"""
LoRA Benchmark Validator
Purpose: Benchmark all working LoRAs with validation prompts and get scores
"""

import json
import requests
import time
import subprocess
import statistics
from pathlib import Path
from typing import Dict, List, Any, Tuple
from dataclasses import dataclass
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('lora_benchmark_validator_15.log'),
        logging.StreamHandler()
    ]
)

@dataclass
class LoRABenchmarkResult:
    """Result for a single LoRA benchmark test"""
    lora_name: str
    model: str
    prompt: str
    optimized_prompt: str
    validation_score: float
    alignment_score: float
    quality_score: float
    generation_time: float
    file_size: int
    success: bool
    error_message: str = ""

@dataclass
class LoRABenchmarkSummary:
    """Summary of benchmark results for a LoRA"""
    lora_name: str
    model: str
    total_tests: int
    successful_tests: int
    avg_validation_score: float
    avg_alignment_score: float
    avg_quality_score: float
    best_score: float
    worst_score: float
    avg_generation_time: float
    avg_file_size: int
    success_rate: float

class LoRABenchmarker:
    """Benchmark all working LoRAs with validation prompts"""
    
    def __init__(self, trellis_server_url: str = "http://localhost:8096"):
        self.trellis_server_url = trellis_server_url
        self.logger = logging.getLogger(__name__)
        
        # Test prompts from the validation script
        self.test_prompts = [
            # "greek amphora scene detail",
            # "plastic straw of drink", 
            # "small yellow triangular wooden kitchen knife",
            # "enormous black robot with round body",
            # "rose gold locket necklace with floral"
            
            # "robot in sitting down position",
            # "mystical orb pulsating with arcane energy",
            # "small winged fairy with golden wings",
            # "parachute deployed mid-air high-speed descent",
            # "metallic robot turning right",
            # "colorful candy in clear glass bottle",
            # "black knight armored in shadow",
            # "magical lantern casting soft blue glow",
            # "purple sapphire in necklace",
            # "white pear delicate texture slightly translucent",
            
            # "rose quartz heart pendant symbolizing love",
            # "glossy blue glass candle holder elegant",
            # "orange electric sander with variable speed",
            # "polished steel drums bright and tropical",
            # "glimmering orange agate with wavy pattern",
            # "heavy-duty green plasma rifle",
            # "amethyst anklet with swirling vine-like patterns",
            # "copper measuring tape retractable",
            # "metal scissors with two sharp blades and curved shape",
            # "red triangle with black circle on it",
            # "smooth purple lacrosse stick",
            # "dark steel knife serrated edge and pointed tip",
            # "ornate bronze cannon with curved barrel",
            # "red and blue monkey with long tail",
            # "silver glowing mermaid",
<<<<<<< HEAD

=======
            "wooden desk with two chairs and laptop", 
>>>>>>> origin/multi
            "luxurious cream sedan elegant",
            "stone statue ancient warrior in battle pose"
        ]
        
        # Working LoRAs configuration
        self.working_loras = {
            'flux': {
                'isometric_3d': {
                    'name': 'Flux Isometric 3D',
                    'endpoint': '/generate/isometric_3d/',
                    'trigger_prefix': 'Isometric 3D,'
                },
                'game_assets': {
                    'name': '3D Game Assets',
                    'endpoint': '/generate/game_assets/',
                    'trigger_prefix': 'Create 3D game asset, isometric view version,'
                },
                'patched_realism': {
                    'name': 'Patched Realism',
                    'endpoint': '/generate/patched_realism/',
                    'trigger_prefix': ''
                },
                'tf2_style': {
                    'name': 'Team Fortress 2 Style',
                    'endpoint': '/generate/tf2_style/',
                    'trigger_prefix': 'tf2style,'
                },
                'baolei': {
                    'name': 'Baolei Style',
                    'endpoint': '/generate/baolei/',
                    'trigger_prefix': 'Cartoon-style design,'
                },
                'cartoon_3d': {
                    'name': 'Cartoon 3D Render',
                    'endpoint': '/generate/cartoon_3d/',
                    'trigger_prefix': ''
                },
                'cinema': {
                    'name': 'Cinema Style',
                    'endpoint': '/generate/cinema/',
                    'trigger_prefix': 'c1n3ma,'
                }
            },
            'sd15': {
                'game_icon': {
                    'name': 'Game Icon Institute',
                    'endpoint': '/generate/sd15_game_icon/',
                    'trigger_prefix': 'game icon institute,'
                }
            }
        }
        
        self.results: List[LoRABenchmarkResult] = []
        
    def test_server_health(self) -> bool:
        """Test if the server is running"""
        try:
            response = requests.get(f"{self.trellis_server_url}/health/", timeout=10)
            if response.status_code == 200:
                self.logger.info("✅ Server is healthy")
                return True
            else:
                self.logger.error(f"❌ Server health check failed: {response.status_code}")
                return False
        except Exception as e:
            self.logger.error(f"❌ Cannot connect to server: {e}")
            return False
    
    def switch_model(self, model: str) -> bool:
        """Switch to the specified model"""
        try:
            response = requests.post(
                f"{self.trellis_server_url}/config/model/",
                data={'model': model},
                timeout=10
            )
            if response.status_code == 200:
                self.logger.info(f"✅ Switched to {model.upper()} model")
                return True
            else:
                self.logger.error(f"❌ Failed to switch to {model.upper()}: {response.status_code}")
                return False
        except Exception as e:
            self.logger.error(f"❌ Error switching to {model.upper()}: {e}")
            return False
    
    def generate_with_lora(self, lora_config: Dict, prompt: str, seed: int = 42) -> Tuple[bool, bytes, float]:
        """Generate 3D model with specific LoRA"""
        start_time = time.time()
        
        try:
            # Apply trigger prefix if specified
            enhanced_prompt = prompt
            if lora_config.get('trigger_prefix'):
                enhanced_prompt = f"{lora_config['trigger_prefix']} {prompt}"
            
            response = requests.post(
                f"{self.trellis_server_url}{lora_config['endpoint']}",
                data={
                    'prompt': enhanced_prompt,
                    'seed': seed,
                    'return_compressed': True
                },
                timeout=300  # 5 minutes timeout
            )
            
            generation_time = time.time() - start_time
            
            if response.status_code == 200:
                file_size = len(response.content)
                self.logger.info(f"   ✅ Generation successful: {file_size:,} bytes in {generation_time:.2f}s")
                return True, response.content, generation_time
            else:
                self.logger.error(f"   ❌ Generation failed: HTTP {response.status_code}")
                return False, b"", generation_time
                
        except Exception as e:
            generation_time = time.time() - start_time
            self.logger.error(f"   ❌ Generation error: {e}")
            return False, b"", generation_time
    
    def validate_with_subnet_validator(self, original_prompt: str, optimized_prompt: str = None) -> Tuple[float, float, float]:
        """Run validation using subnet_accurate_validator.py"""
        try:
            self.logger.info("      🔍 Validating with subnet_accurate_validator...")
            
            # Use optimized prompt for generation if provided, otherwise use original
            if optimized_prompt and optimized_prompt != original_prompt:
                cmd = [
                    "bash", "-c",
                    f"source /home/mbhat/miniconda/bin/activate && conda activate trellis_new && python subnet_accurate_validator.py \"{original_prompt}\" \"{optimized_prompt}\""
                ]
            else:
                cmd = [
                    "bash", "-c",
                    f"source /home/mbhat/miniconda/bin/activate && conda activate trellis_new && python subnet_accurate_validator.py \"{original_prompt}\""
                ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
            
            if result.returncode != 0:
                self.logger.warning(f"      ❌ Validation failed (return code {result.returncode})")
                return 0.0, 0.0, 0.0
            
            # Parse validation results
            with open("subnet_validation_results.json", 'r') as f:
                data = json.load(f)
                score = data.get("validation_engine_score", 0.0)
                alignment_score = data.get("alignment_score", 0.0)
                quality_score = data.get("quality_score", 0.0)
                
                self.logger.info(f"      📊 Validation Score: {score:.4f}")
                self.logger.info(f"      📊 Alignment Score: {alignment_score:.4f}")
                self.logger.info(f"      📊 Quality Score: {quality_score:.4f}")
                
                return score, alignment_score, quality_score
                
        except Exception as e:
            self.logger.error(f"      ❌ Validation error: {e}")
            return 0.0, 0.0, 0.0
    
    def benchmark_single_lora(self, model: str, lora_key: str, lora_config: Dict) -> List[LoRABenchmarkResult]:
        """Benchmark a single LoRA with all test prompts"""
        results = []
        
        self.logger.info(f"\n🎨 Benchmarking {model.upper()} + {lora_config['name']}")
        self.logger.info(f"   Endpoint: {lora_config['endpoint']}")
        self.logger.info(f"   Trigger prefix: '{lora_config.get('trigger_prefix', 'None')}'")
        
        for i, prompt in enumerate(self.test_prompts, 1):
            self.logger.info(f"\n   📝 Test {i}/{len(self.test_prompts)}: '{prompt}'")
            
            # Generate with LoRA
            success, file_data, generation_time = self.generate_with_lora(lora_config, prompt)
            
            if success:
                # Save the generated file for validation
                output_dir = Path("./benchmark_outputs")
                output_dir.mkdir(exist_ok=True)
                
                filename = f"{model}_{lora_key}_{prompt.replace(' ', '_')}_{int(time.time())}.ply.spz"
                filepath = output_dir / filename
                
                with open(filepath, 'wb') as f:
                    f.write(file_data)
                
                # Apply trigger prefix for validation
                enhanced_prompt = prompt
                if lora_config.get('trigger_prefix'):
                    enhanced_prompt = f"{lora_config['trigger_prefix']} {prompt}"
                
                # Validate
                validation_score, alignment_score, quality_score = self.validate_with_subnet_validator(
                    prompt, enhanced_prompt
                )
                
                result = LoRABenchmarkResult(
                    lora_name=lora_config['name'],
                    model=model.upper(),
                    prompt=prompt,
                    optimized_prompt=enhanced_prompt,
                    validation_score=validation_score,
                    alignment_score=alignment_score,
                    quality_score=quality_score,
                    generation_time=generation_time,
                    file_size=len(file_data),
                    success=True
                )
                
            else:
                result = LoRABenchmarkResult(
                    lora_name=lora_config['name'],
                    model=model.upper(),
                    prompt=prompt,
                    optimized_prompt=prompt,
                    validation_score=0.0,
                    alignment_score=0.0,
                    quality_score=0.0,
                    generation_time=generation_time,
                    file_size=0,
                    success=False,
                    error_message="Generation failed"
                )
            
            results.append(result)
            self.results.append(result)
            
            # Brief pause between tests
            time.sleep(2)
        
        return results
    
    def benchmark_all_loras(self) -> Dict[str, LoRABenchmarkSummary]:
        """Benchmark all working LoRAs"""
        self.logger.info("🚀 Starting LoRA Benchmark")
        self.logger.info("=" * 60)
        
        if not self.test_server_health():
            raise Exception("Server is not available")
        
        summaries = {}
        
        # Test FLUX LoRAs
        self.logger.info("\n🎨 Testing FLUX LoRAs...")
        if not self.switch_model('flux'):
            raise Exception("Failed to switch to FLUX model")
        
        for lora_key, lora_config in self.working_loras['flux'].items():
            try:
                results = self.benchmark_single_lora('flux', lora_key, lora_config)
                summary = self._create_summary(results)
                summaries[f"flux_{lora_key}"] = summary
            except Exception as e:
                self.logger.error(f"❌ Error benchmarking {lora_key}: {e}")
        
        # Test SD1.5 LoRAs
        self.logger.info("\n🎨 Testing SD1.5 LoRAs...")
        if not self.switch_model('sd15'):
            raise Exception("Failed to switch to SD1.5 model")
        
        for lora_key, lora_config in self.working_loras['sd15'].items():
            try:
                results = self.benchmark_single_lora('sd15', lora_key, lora_config)
                summary = self._create_summary(results)
                summaries[f"sd15_{lora_key}"] = summary
            except Exception as e:
                self.logger.error(f"❌ Error benchmarking {lora_key}: {e}")
        
        return summaries
    
    def _create_summary(self, results: List[LoRABenchmarkResult]) -> LoRABenchmarkSummary:
        """Create summary from benchmark results"""
        successful_results = [r for r in results if r.success]
        
        if not successful_results:
            return LoRABenchmarkSummary(
                lora_name=results[0].lora_name if results else "Unknown",
                model=results[0].model if results else "Unknown",
                total_tests=len(results),
                successful_tests=0,
                avg_validation_score=0.0,
                avg_alignment_score=0.0,
                avg_quality_score=0.0,
                best_score=0.0,
                worst_score=0.0,
                avg_generation_time=0.0,
                avg_file_size=0,
                success_rate=0.0
            )
        
        validation_scores = [r.validation_score for r in successful_results]
        alignment_scores = [r.alignment_score for r in successful_results]
        quality_scores = [r.quality_score for r in successful_results]
        generation_times = [r.generation_time for r in successful_results]
        file_sizes = [r.file_size for r in successful_results]
        
        return LoRABenchmarkSummary(
            lora_name=successful_results[0].lora_name,
            model=successful_results[0].model,
            total_tests=len(results),
            successful_tests=len(successful_results),
            avg_validation_score=statistics.mean(validation_scores),
            avg_alignment_score=statistics.mean(alignment_scores),
            avg_quality_score=statistics.mean(quality_scores),
            best_score=max(validation_scores),
            worst_score=min(validation_scores),
            avg_generation_time=statistics.mean(generation_times),
            avg_file_size=int(statistics.mean(file_sizes)),
            success_rate=len(successful_results) / len(results)
        )
    
    def print_benchmark_results(self, summaries: Dict[str, LoRABenchmarkSummary]):
        """Print comprehensive benchmark results"""
        print("\n" + "="*80)
        print("📊 LoRA BENCHMARK RESULTS")
        print("="*80)
        
        # Sort by average validation score
        sorted_summaries = sorted(
            summaries.items(),
            key=lambda x: x[1].avg_validation_score,
            reverse=True
        )
        
        print(f"\n🏆 RANKED BY AVERAGE VALIDATION SCORE:")
        print("-" * 80)
        print(f"{'Rank':<4} {'LoRA':<25} {'Model':<6} {'Avg Score':<10} {'Best':<8} {'Worst':<8} {'Success':<8} {'Avg Time':<10}")
        print("-" * 80)
        
        for rank, (key, summary) in enumerate(sorted_summaries, 1):
            print(f"{rank:<4} {summary.lora_name:<25} {summary.model:<6} "
                  f"{summary.avg_validation_score:<10.4f} {summary.best_score:<8.4f} "
                  f"{summary.worst_score:<8.4f} {summary.success_rate:<8.1%} "
                  f"{summary.avg_generation_time:<10.2f}s")
        
        print(f"\n📈 DETAILED BREAKDOWN:")
        print("-" * 80)
        
        for key, summary in sorted_summaries:
            print(f"\n🎨 {summary.lora_name} ({summary.model})")
            print(f"   📊 Validation Score: {summary.avg_validation_score:.4f} (best: {summary.best_score:.4f}, worst: {summary.worst_score:.4f})")
            print(f"   📊 Alignment Score: {summary.avg_alignment_score:.4f}")
            print(f"   📊 Quality Score: {summary.avg_quality_score:.4f}")
            print(f"   ⏱️  Generation Time: {summary.avg_generation_time:.2f}s")
            print(f"   📁 File Size: {summary.avg_file_size:,} bytes")
            print(f"   ✅ Success Rate: {summary.success_rate:.1%} ({summary.successful_tests}/{summary.total_tests})")
        
        # Model comparison
        print(f"\n🏗️  MODEL COMPARISON:")
        print("-" * 80)
        
        model_stats = {}
        for key, summary in summaries.items():
            model = summary.model
            if model not in model_stats:
                model_stats[model] = []
            model_stats[model].append(summary.avg_validation_score)
        
        for model, scores in model_stats.items():
            avg_score = statistics.mean(scores)
            print(f"   {model}: {avg_score:.4f} (from {len(scores)} LoRAs)")
        
        # Save results
        self._save_results(summaries)
    
    def _save_results(self, summaries: Dict[str, LoRABenchmarkSummary]):
        """Save benchmark results to file"""
        output_file = Path("./lora_benchmark_results_2.json")
        
        # Convert to serializable format
        results_data = {
            'timestamp': time.time(),
            'test_prompts': self.test_prompts,
            'summaries': {
                key: {
                    'lora_name': summary.lora_name,
                    'model': summary.model,
                    'total_tests': summary.total_tests,
                    'successful_tests': summary.successful_tests,
                    'avg_validation_score': summary.avg_validation_score,
                    'avg_alignment_score': summary.avg_alignment_score,
                    'avg_quality_score': summary.avg_quality_score,
                    'best_score': summary.best_score,
                    'worst_score': summary.worst_score,
                    'avg_generation_time': summary.avg_generation_time,
                    'avg_file_size': summary.avg_file_size,
                    'success_rate': summary.success_rate
                }
                for key, summary in summaries.items()
            },
            'detailed_results': [
                {
                    'lora_name': result.lora_name,
                    'model': result.model,
                    'prompt': result.prompt,
                    'optimized_prompt': result.optimized_prompt,
                    'validation_score': result.validation_score,
                    'alignment_score': result.alignment_score,
                    'quality_score': result.quality_score,
                    'generation_time': result.generation_time,
                    'file_size': result.file_size,
                    'success': result.success,
                    'error_message': result.error_message
                }
                for result in self.results
            ]
        }
        
        with open(output_file, 'w') as f:
            json.dump(results_data, f, indent=2)
        
        print(f"\n💾 Results saved to: {output_file}")

def main():
    """Main benchmark function"""
    print("🚀 LoRA Benchmark Validator")
    print("=" * 60)
    print("🎯 Testing all working LoRAs with validation prompts")
    print("📊 Using subnet_accurate_validator.py for scoring")
    print("=" * 60)
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    try:
        benchmarker = LoRABenchmarker()
        summaries = benchmarker.benchmark_all_loras()
        benchmarker.print_benchmark_results(summaries)
        
        print("\n✅ Benchmark completed successfully!")
        
    except Exception as e:
        print(f"\n❌ Benchmark failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 