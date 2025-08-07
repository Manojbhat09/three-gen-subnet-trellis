#!/usr/bin/env python3
"""
Quality Optimizer for TRELLIS Generation
Purpose: Find optimal parameters for maximum validation scores (0.99+ task fidelity)

This script systematically tests different TRELLIS configurations to find
the best parameters for achieving high validation scores.
"""

import requests
import time
import json
import argparse
from typing import Dict, List, Tuple
from dataclasses import dataclass
import numpy as np

@dataclass
class QualityConfig:
    """Quality configuration for TRELLIS"""
    name: str
    guidance_scale: float
    ss_guidance_strength: float
    ss_sampling_steps: int
    slat_guidance_strength: float
    slat_sampling_steps: int
    expected_quality: str
    description: str

class QualityOptimizer:
    """Optimizer for finding best TRELLIS parameters"""
    
    def __init__(self, generation_server_url: str = "http://localhost:8096",
                 validation_server_url: str = "http://localhost:10006"):
        self.generation_url = generation_server_url
        self.validation_url = validation_server_url
        
        # Predefined quality configurations
        self.quality_configs = [
            QualityConfig(
                name="ultra_quality",
                guidance_scale=4.5,
                ss_guidance_strength=10.0,
                ss_sampling_steps=35,
                slat_guidance_strength=6.0,
                slat_sampling_steps=35,
                expected_quality="0.95+",
                description="Ultra high quality - maximum validation scores"
            ),
            QualityConfig(
                name="high_quality",
                guidance_scale=4.0,
                ss_guidance_strength=9.5,
                ss_sampling_steps=30,
                slat_guidance_strength=5.0,
                slat_sampling_steps=30,
                expected_quality="0.90+",
                description="High quality - optimized for validation"
            ),
            QualityConfig(
                name="balanced_quality",
                guidance_scale=3.8,
                ss_guidance_strength=9.0,
                ss_sampling_steps=25,
                slat_guidance_strength=4.5,
                slat_sampling_steps=25,
                expected_quality="0.85+",
                description="Balanced quality and speed"
            ),
            QualityConfig(
                name="experimental_ultra",
                guidance_scale=5.0,
                ss_guidance_strength=12.0,
                ss_sampling_steps=40,
                slat_guidance_strength=7.0,
                slat_sampling_steps=40,
                expected_quality="0.98+",
                description="Experimental ultra settings - may be unstable"
            ),
        ]
    
    def update_server_config(self, config: QualityConfig) -> bool:
        """Update server configuration"""
        try:
            response = requests.post(
                f"{self.generation_url}/config/quality/",
                data={
                    "guidance_scale": config.guidance_scale,
                    "ss_guidance_strength": config.ss_guidance_strength,
                    "ss_sampling_steps": config.ss_sampling_steps,
                    "slat_guidance_strength": config.slat_guidance_strength,
                    "slat_sampling_steps": config.slat_sampling_steps
                },
                timeout=30
            )
            return response.status_code == 200
        except Exception as e:
            print(f"❌ Failed to update server config: {e}")
            return False
    
    def generate_and_validate(self, prompt: str, seed: int = 42) -> Dict:
        """Generate model and validate it"""
        try:
            # Generate model
            print(f"🎯 Generating model with prompt: '{prompt}'")
            generation_response = requests.post(
                f"{self.generation_url}/generate/",
                data={
                    "prompt": prompt,
                    "seed": seed,
                    "return_compressed": True
                },
                timeout=300  # 5 minutes timeout
            )
            
            if generation_response.status_code != 200:
                return {"error": f"Generation failed: {generation_response.status_code}"}
            
            # Get compressed PLY data
            ply_data = generation_response.content
            
            # Validate the generation
            print("📊 Validating generation...")
            validation_response = requests.post(
                f"{self.validation_url}/validate_txt_to_3d_ply/",
                json={
                    "prompt": prompt,
                    "data": ply_data.hex(),  # Convert to hex for JSON
                    "compression": 0,
                    "generate_preview": False
                },
                timeout=120
            )
            
            if validation_response.status_code == 200:
                result = validation_response.json()
                return {
                    "success": True,
                    "validation_score": result.get("score", 0.0),
                    "quality_metrics": {
                        "iqa": result.get("iqa", 0.0),
                        "alignment": result.get("alignment_score", 0.0),
                        "ssim": result.get("ssim", 0.0),
                        "lpips": result.get("lpips", 0.0)
                    }
                }
            else:
                return {"error": f"Validation failed: {validation_response.status_code}"}
                
        except Exception as e:
            return {"error": f"Generation/validation failed: {e}"}
    
    def test_configuration(self, config: QualityConfig, test_prompts: List[str]) -> Dict:
        """Test a specific configuration"""
        print(f"\n🔧 Testing configuration: {config.name}")
        print(f"   Description: {config.description}")
        print(f"   Expected quality: {config.expected_quality}")
        print(f"   Parameters:")
        print(f"     - Guidance scale: {config.guidance_scale}")
        print(f"     - SS guidance strength: {config.ss_guidance_strength}")
        print(f"     - SS sampling steps: {config.ss_sampling_steps}")
        print(f"     - SLAT guidance strength: {config.slat_guidance_strength}")
        print(f"     - SLAT sampling steps: {config.slat_sampling_steps}")
        
        # Update server configuration
        if not self.update_server_config(config):
            return {"error": "Failed to update server configuration"}
        
        # Test with multiple prompts
        results = []
        for i, prompt in enumerate(test_prompts):
            print(f"\n   Testing prompt {i+1}/{len(test_prompts)}: '{prompt}'")
            result = self.generate_and_validate(prompt, seed=42+i)
            
            if "error" in result:
                print(f"   ❌ Failed: {result['error']}")
                results.append(result)
            else:
                score = result["validation_score"]
                print(f"   ✅ Validation score: {score:.4f}")
                results.append(result)
            
            # Wait between generations
            time.sleep(5)
        
        # Calculate statistics
        successful_results = [r for r in results if "success" in r]
        if successful_results:
            scores = [r["validation_score"] for r in successful_results]
            avg_score = np.mean(scores)
            max_score = np.max(scores)
            min_score = np.min(scores)
            
            print(f"\n📊 Configuration {config.name} Results:")
            print(f"   Average score: {avg_score:.4f}")
            print(f"   Best score: {max_score:.4f}")
            print(f"   Worst score: {min_score:.4f}")
            print(f"   Success rate: {len(successful_results)}/{len(results)}")
            
            return {
                "config": config,
                "results": results,
                "statistics": {
                    "average_score": avg_score,
                    "max_score": max_score,
                    "min_score": min_score,
                    "success_rate": len(successful_results) / len(results)
                }
            }
        else:
            print(f"\n❌ Configuration {config.name} failed all tests")
            return {
                "config": config,
                "results": results,
                "statistics": {
                    "average_score": 0.0,
                    "max_score": 0.0,
                    "min_score": 0.0,
                    "success_rate": 0.0
                }
            }
    
    def optimize(self, test_prompts: List[str], target_score: float = 0.95) -> Dict:
        """Run optimization across all configurations"""
        print("🚀 Starting Quality Optimization")
        print(f"Target score: {target_score}")
        print(f"Test prompts: {len(test_prompts)}")
        
        all_results = []
        best_config = None
        best_score = 0.0
        
        for config in self.quality_configs:
            result = self.test_configuration(config, test_prompts)
            all_results.append(result)
            
            if "statistics" in result:
                avg_score = result["statistics"]["average_score"]
                if avg_score > best_score:
                    best_score = avg_score
                    best_config = config
                
                if avg_score >= target_score:
                    print(f"\n🎉 Target score {target_score} achieved with {config.name}!")
                    print(f"Average score: {avg_score:.4f}")
                    break
        
        # Summary
        print(f"\n📈 Optimization Summary:")
        print(f"Best configuration: {best_config.name if best_config else 'None'}")
        print(f"Best average score: {best_score:.4f}")
        
        if best_config:
            print(f"\n🏆 Recommended configuration for {target_score}+ scores:")
            print(f"Configuration: {best_config.name}")
            print(f"Parameters:")
            print(f"  guidance_scale: {best_config.guidance_scale}")
            print(f"  ss_guidance_strength: {best_config.ss_guidance_strength}")
            print(f"  ss_sampling_steps: {best_config.ss_sampling_steps}")
            print(f"  slat_guidance_strength: {best_config.slat_guidance_strength}")
            print(f"  slat_sampling_steps: {best_config.slat_sampling_steps}")
        
        return {
            "best_config": best_config,
            "best_score": best_score,
            "all_results": all_results
        }

def main():
    parser = argparse.ArgumentParser(description="Quality Optimizer for TRELLIS")
    parser.add_argument("--generation-server", default="http://localhost:8096",
                       help="Generation server URL")
    parser.add_argument("--validation-server", default="http://localhost:10006",
                       help="Validation server URL")
    parser.add_argument("--target-score", type=float, default=0.95,
                       help="Target validation score")
    parser.add_argument("--prompts", nargs="+", 
                       default=["a red ceramic vase", "a wooden chair", "a metal lamp"],
                       help="Test prompts")
    
    args = parser.parse_args()
    
    # Initialize optimizer
    optimizer = QualityOptimizer(args.generation_server, args.validation_server)
    
    # Run optimization
    results = optimizer.optimize(args.prompts, args.target_score)
    
    # Save results
    with open("quality_optimization_results.json", "w") as f:
        json.dump(results, f, indent=2, default=lambda x: x.__dict__ if hasattr(x, '__dict__') else str(x))
    
    print(f"\n💾 Results saved to quality_optimization_results.json")

if __name__ == "__main__":
    main() 