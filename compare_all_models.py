#!/usr/bin/env python3
"""
Compare FLUX vs SDXL vs HunyuanDiT for Alignment Scores
Purpose: Test which text-to-image model produces the best alignment scores for TRELLIS 3D generation
"""

import requests
import time
import json
import base64
import argparse
from typing import Dict, List, Tuple
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

class ModelComparison:
    """Compare different text-to-image models for TRELLIS alignment"""
    
    def __init__(self, 
                 flux_server_url: str = "http://localhost:8096",
                 sdxl_server_url: str = "http://localhost:8097",
                 hunyuan_server_url: str = "http://localhost:8098",
                 validation_server_url: str = "http://localhost:10006"):
        self.flux_url = flux_server_url
        self.sdxl_url = sdxl_server_url
        self.hunyuan_url = hunyuan_server_url
        self.validation_url = validation_server_url
        
        # Test prompts designed to test alignment
        self.test_prompts = [
            "a red ceramic vase with gold trim",
            "a wooden chair with leather upholstery", 
            "a metal lamp with glass shade",
            "a blue ceramic bowl with white flowers",
            "a silver candlestick holder",
            "a bronze statue of a cat",
            "a glass vase with crystal patterns",
            "a wooden table with carved legs"
        ]
        
        self.model_configs = {
            "FLUX": {
                "url": self.flux_url,
                "color": "#FF6B6B",
                "description": "FLUX - Fast text-to-image"
            },
            "SDXL": {
                "url": self.sdxl_url,
                "color": "#4ECDC4",
                "description": "SDXL - High quality text-to-image"
            },
            "HunyuanDiT": {
                "url": self.hunyuan_url,
                "color": "#45B7D1",
                "description": "HunyuanDiT - Advanced text-to-image"
            }
        }
    
    def test_model_generation(self, server_url: str, prompt: str, seed: int) -> Dict:
        """Test generation with a specific model"""
        try:
            model_name = "Unknown"
            for name, config in self.model_configs.items():
                if config["url"] == server_url:
                    model_name = name
                    break
            
            print(f"🎯 Generating with {model_name}: '{prompt}'")
            
            # Generate model
            generation_response = requests.post(
                f"{server_url}/generate/",
                data={
                    "prompt": prompt,
                    "seed": seed,
                    "return_compressed": True
                },
                timeout=300
            )
            
            if generation_response.status_code != 200:
                return {"error": f"Generation failed: {generation_response.status_code}"}
            
            # Get compressed PLY data
            ply_data = generation_response.content
            
            # Validate the generation
            print("📊 Validating generation...")
            encoded_data = base64.b64encode(ply_data).decode('utf-8')
            
            validation_response = requests.post(
                f"{self.validation_url}/validate_txt_to_3d_ply/",
                json={
                    "prompt": prompt,
                    "data": encoded_data,
                    "compression": 0,
                    "generate_preview": False
                },
                timeout=120
            )
            
            if validation_response.status_code == 200:
                result = validation_response.json()
                return {
                    "success": True,
                    "model": model_name,
                    "validation_score": result.get("score", 0.0),
                    "alignment_score": result.get("alignment_score", 0.0),
                    "quality_metrics": {
                        "iqa": result.get("iqa", 0.0),
                        "alignment": result.get("alignment_score", 0.0),
                        "ssim": result.get("ssim", 0.0),
                        "lpips": result.get("lpips", 0.0)
                    },
                    "ply_size_bytes": len(ply_data)
                }
            else:
                return {"error": f"Validation failed: {validation_response.status_code}"}
                
        except Exception as e:
            return {"error": f"Generation/validation failed: {e}"}
    
    def compare_models(self, num_tests: int = 2) -> Dict:
        """Compare FLUX vs SDXL vs HunyuanDiT performance"""
        print("🚀 Starting FLUX vs SDXL vs HunyuanDiT Comparison")
        print(f"Number of tests per model: {num_tests}")
        print(f"Total prompts: {len(self.test_prompts)}")
        
        all_results = {model: [] for model in self.model_configs.keys()}
        
        for i, prompt in enumerate(self.test_prompts):
            print(f"\n{'='*80}")
            print(f"Testing Prompt {i+1}/{len(self.test_prompts)}: '{prompt}'")
            print(f"{'='*80}")
            
            prompt_results = {model: [] for model in self.model_configs.keys()}
            
            for test_num in range(num_tests):
                seed = 42 + i * 100 + test_num
                print(f"\n--- Test {test_num + 1}/{num_tests} (seed: {seed}) ---")
                
                # Test each model
                for model_name, config in self.model_configs.items():
                    result = self.test_model_generation(config["url"], prompt, seed)
                    if "success" in result:
                        score = result["validation_score"]
                        alignment = result["alignment_score"]
                        print(f"{model_name}: Score={score:.4f}, Alignment={alignment:.4f}")
                        prompt_results[model_name].append(result)
                        all_results[model_name].append(result)
                    else:
                        print(f"{model_name}: Failed - {result.get('error', 'Unknown error')}")
                    
                    # Wait between generations
                    time.sleep(3)
            
            # Calculate averages for this prompt
            print(f"\n📊 Prompt {i+1} Results:")
            for model_name in self.model_configs.keys():
                if prompt_results[model_name]:
                    scores = [r["validation_score"] for r in prompt_results[model_name]]
                    alignments = [r["alignment_score"] for r in prompt_results[model_name]]
                    avg_score = np.mean(scores)
                    avg_alignment = np.mean(alignments)
                    print(f"   {model_name}: Avg Score={avg_score:.4f}, Avg Alignment={avg_alignment:.4f}")
        
        # Overall comparison
        print(f"\n{'='*100}")
        print("📈 OVERALL COMPARISON RESULTS")
        print(f"{'='*100}")
        
        # Calculate overall statistics
        overall_stats = {}
        for model_name in self.model_configs.keys():
            if all_results[model_name]:
                scores = [r["validation_score"] for r in all_results[model_name]]
                alignments = [r["alignment_score"] for r in all_results[model_name]]
                
                overall_stats[model_name] = {
                    "avg_score": np.mean(scores),
                    "avg_alignment": np.mean(alignments),
                    "std_score": np.std(scores),
                    "std_alignment": np.std(alignments),
                    "min_score": np.min(scores),
                    "max_score": np.max(scores),
                    "zero_scores": sum(1 for s in scores if s == 0.0),
                    "total_tests": len(scores),
                    "success_rate": len(scores) / (len(self.test_prompts) * num_tests) * 100
                }
                
                print(f"\n🔍 {model_name} Results ({len(scores)} generations):")
                print(f"   Average Score: {overall_stats[model_name]['avg_score']:.4f} ± {overall_stats[model_name]['std_score']:.4f}")
                print(f"   Average Alignment: {overall_stats[model_name]['avg_alignment']:.4f} ± {overall_stats[model_name]['std_alignment']:.4f}")
                print(f"   Score Range: {overall_stats[model_name]['min_score']:.4f} - {overall_stats[model_name]['max_score']:.4f}")
                print(f"   Zero Scores: {overall_stats[model_name]['zero_scores']}/{len(scores)} ({overall_stats[model_name]['zero_scores']/len(scores)*100:.1f}%)")
                print(f"   Success Rate: {overall_stats[model_name]['success_rate']:.1f}%")
        
        # Determine winner
        print(f"\n🏆 WINNER ANALYSIS:")
        if overall_stats:
            # Find best model by average score
            best_score_model = max(overall_stats.keys(), key=lambda x: overall_stats[x]['avg_score'])
            best_alignment_model = max(overall_stats.keys(), key=lambda x: overall_stats[x]['avg_alignment'])
            best_reliability_model = min(overall_stats.keys(), key=lambda x: overall_stats[x]['zero_scores'])
            
            print(f"   Best Overall Score: {best_score_model} ({overall_stats[best_score_model]['avg_score']:.4f})")
            print(f"   Best Alignment: {best_alignment_model} ({overall_stats[best_alignment_model]['avg_alignment']:.4f})")
            print(f"   Most Reliable (fewest zeros): {best_reliability_model} ({overall_stats[best_reliability_model]['zero_scores']} zeros)")
            
            # Detailed comparison
            print(f"\n📊 Detailed Comparison:")
            for i, model1 in enumerate(overall_stats.keys()):
                for model2 in list(overall_stats.keys())[i+1:]:
                    score_diff = overall_stats[model1]['avg_score'] - overall_stats[model2]['avg_score']
                    alignment_diff = overall_stats[model1]['avg_alignment'] - overall_stats[model2]['avg_alignment']
                    
                    print(f"   {model1} vs {model2}:")
                    print(f"     Score: {model1 if score_diff > 0 else model2} wins by {abs(score_diff):.4f}")
                    print(f"     Alignment: {model1 if alignment_diff > 0 else model2} wins by {abs(alignment_diff):.4f}")
            
            # Recommendation
            print(f"\n💡 RECOMMENDATION:")
            if best_score_model == best_alignment_model == best_reliability_model:
                print(f"   Use {best_score_model} - best overall performance across all metrics")
            elif best_score_model == best_alignment_model:
                print(f"   Use {best_score_model} - best score and alignment, consider {best_reliability_model} for reliability")
            elif best_alignment_model == best_reliability_model:
                print(f"   Use {best_alignment_model} - best alignment and reliability, consider {best_score_model} for overall score")
            else:
                print(f"   Trade-off decision:")
                print(f"     - For maximum score: {best_score_model}")
                print(f"     - For best alignment: {best_alignment_model}")
                print(f"     - For reliability: {best_reliability_model}")
        
        # Create visualizations
        self.create_visualizations(overall_stats, all_results)
        
        # Save detailed results
        comparison_results = {
            "timestamp": time.time(),
            "test_config": {
                "num_tests_per_prompt": num_tests,
                "total_prompts": len(self.test_prompts),
                "models": list(self.model_configs.keys()),
                "validation_server": self.validation_url
            },
            "all_results": all_results,
            "overall_stats": overall_stats,
            "summary": {
                "best_score_model": best_score_model if overall_stats else None,
                "best_alignment_model": best_alignment_model if overall_stats else None,
                "best_reliability_model": best_reliability_model if overall_stats else None,
                "recommendation": self._generate_recommendation(overall_stats)
            }
        }
        
        with open("all_models_comparison.json", "w") as f:
            json.dump(comparison_results, f, indent=2, default=lambda x: x.__dict__ if hasattr(x, '__dict__') else str(x))
        
        print(f"\n💾 Detailed results saved to all_models_comparison.json")
        print(f"📊 Visualizations saved to comparison_charts.png")
        
        return comparison_results
    
    def create_visualizations(self, overall_stats: Dict, all_results: Dict):
        """Create visualizations of the comparison results"""
        try:
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            fig.suptitle('FLUX vs SDXL vs HunyuanDiT Comparison', fontsize=16, fontweight='bold')
            
            # Prepare data
            models = list(overall_stats.keys())
            colors = [self.model_configs[model]["color"] for model in models]
            
            # 1. Average Scores
            avg_scores = [overall_stats[model]['avg_score'] for model in models]
            score_stds = [overall_stats[model]['std_score'] for model in models]
            
            axes[0, 0].bar(models, avg_scores, yerr=score_stds, capsize=5, color=colors, alpha=0.7)
            axes[0, 0].set_title('Average Validation Scores')
            axes[0, 0].set_ylabel('Score')
            axes[0, 0].grid(True, alpha=0.3)
            
            # 2. Average Alignment Scores
            avg_alignments = [overall_stats[model]['avg_alignment'] for model in models]
            alignment_stds = [overall_stats[model]['std_alignment'] for model in models]
            
            axes[0, 1].bar(models, avg_alignments, yerr=alignment_stds, capsize=5, color=colors, alpha=0.7)
            axes[0, 1].set_title('Average Alignment Scores')
            axes[0, 1].set_ylabel('Alignment Score')
            axes[0, 1].grid(True, alpha=0.3)
            
            # 3. Zero Score Percentage
            zero_percentages = [overall_stats[model]['zero_scores']/overall_stats[model]['total_tests']*100 for model in models]
            
            axes[1, 0].bar(models, zero_percentages, color=colors, alpha=0.7)
            axes[1, 0].set_title('Zero Score Percentage')
            axes[1, 0].set_ylabel('Percentage (%)')
            axes[1, 0].grid(True, alpha=0.3)
            
            # 4. Score Distribution
            for model in models:
                scores = [r["validation_score"] for r in all_results[model]]
                axes[1, 1].hist(scores, alpha=0.6, label=model, color=self.model_configs[model]["color"], bins=10)
            
            axes[1, 1].set_title('Score Distribution')
            axes[1, 1].set_xlabel('Validation Score')
            axes[1, 1].set_ylabel('Frequency')
            axes[1, 1].legend()
            axes[1, 1].grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig('comparison_charts.png', dpi=300, bbox_inches='tight')
            plt.close()
            
        except Exception as e:
            print(f"⚠️ Visualization creation failed: {e}")
    
    def _generate_recommendation(self, overall_stats: Dict) -> str:
        """Generate a recommendation based on the results"""
        if not overall_stats:
            return "No data available for recommendation"
        
        best_score_model = max(overall_stats.keys(), key=lambda x: overall_stats[x]['avg_score'])
        best_alignment_model = max(overall_stats.keys(), key=lambda x: overall_stats[x]['avg_alignment'])
        best_reliability_model = min(overall_stats.keys(), key=lambda x: overall_stats[x]['zero_scores'])
        
        if best_score_model == best_alignment_model == best_reliability_model:
            return f"Use {best_score_model} - best overall performance"
        elif best_score_model == best_alignment_model:
            return f"Use {best_score_model} - best score and alignment"
        elif best_alignment_model == best_reliability_model:
            return f"Use {best_alignment_model} - best alignment and reliability"
        else:
            return f"Trade-off: {best_score_model} for score, {best_alignment_model} for alignment, {best_reliability_model} for reliability"

def main():
    parser = argparse.ArgumentParser(description="Compare FLUX vs SDXL vs HunyuanDiT for TRELLIS alignment")
    parser.add_argument("--flux-server", default="http://localhost:8096",
                       help="FLUX generation server URL")
    parser.add_argument("--sdxl-server", default="http://localhost:8097",
                       help="SDXL generation server URL")
    parser.add_argument("--hunyuan-server", default="http://localhost:8098",
                       help="HunyuanDiT generation server URL")
    parser.add_argument("--validation-server", default="http://localhost:10006",
                       help="Validation server URL")
    parser.add_argument("--num-tests", type=int, default=2,
                       help="Number of tests per prompt per model")
    
    args = parser.parse_args()
    
    # Initialize comparison
    comparison = ModelComparison(
        flux_server_url=args.flux_server,
        sdxl_server_url=args.sdxl_server,
        hunyuan_server_url=args.hunyuan_server,
        validation_server_url=args.validation_server
    )
    
    # Run comparison
    results = comparison.compare_models(num_tests=args.num_tests)
    
    print(f"\n✅ Comparison completed!")

if __name__ == "__main__":
    main() 