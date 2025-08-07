#!/usr/bin/env python3
"""
Compare SDXL vs FLUX for Alignment Scores
Purpose: Test which text-to-image model produces better alignment scores for TRELLIS 3D generation
"""

import requests
import time
import json
import base64
import argparse
from typing import Dict, List, Tuple
import numpy as np

class ModelComparison:
    """Compare different text-to-image models for TRELLIS alignment"""
    
    def __init__(self, 
                 flux_server_url: str = "http://localhost:8096",
                 sdxl_server_url: str = "http://localhost:8097",
                 validation_server_url: str = "http://localhost:10006"):
        self.flux_url = flux_server_url
        self.sdxl_url = sdxl_server_url
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
    
    def test_model_generation(self, server_url: str, prompt: str, seed: int) -> Dict:
        """Test generation with a specific model"""
        try:
            print(f"🎯 Generating with {server_url.split('//')[1].split(':')[0].upper()}: '{prompt}'")
            
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
    
    def compare_models(self, num_tests: int = 3) -> Dict:
        """Compare FLUX vs SDXL performance"""
        print("🚀 Starting FLUX vs SDXL Comparison")
        print(f"Number of tests per model: {num_tests}")
        print(f"Total prompts: {len(self.test_prompts)}")
        
        flux_results = []
        sdxl_results = []
        
        for i, prompt in enumerate(self.test_prompts):
            print(f"\n{'='*60}")
            print(f"Testing Prompt {i+1}/{len(self.test_prompts)}: '{prompt}'")
            print(f"{'='*60}")
            
            prompt_flux_results = []
            prompt_sdxl_results = []
            
            for test_num in range(num_tests):
                seed = 42 + i * 100 + test_num
                print(f"\n--- Test {test_num + 1}/{num_tests} (seed: {seed}) ---")
                
                # Test FLUX
                flux_result = self.test_model_generation(self.flux_url, prompt, seed)
                if "success" in flux_result:
                    score = flux_result["validation_score"]
                    alignment = flux_result["alignment_score"]
                    print(f"FLUX: Score={score:.4f}, Alignment={alignment:.4f}")
                    prompt_flux_results.append(flux_result)
                else:
                    print(f"FLUX: Failed - {flux_result.get('error', 'Unknown error')}")
                
                # Wait between generations
                time.sleep(5)
                
                # Test SDXL
                sdxl_result = self.test_model_generation(self.sdxl_url, prompt, seed)
                if "success" in sdxl_result:
                    score = sdxl_result["validation_score"]
                    alignment = sdxl_result["alignment_score"]
                    print(f"SDXL: Score={score:.4f}, Alignment={alignment:.4f}")
                    prompt_sdxl_results.append(sdxl_result)
                else:
                    print(f"SDXL: Failed - {sdxl_result.get('error', 'Unknown error')}")
                
                # Wait between generations
                time.sleep(5)
            
            # Calculate averages for this prompt
            if prompt_flux_results:
                flux_scores = [r["validation_score"] for r in prompt_flux_results]
                flux_alignments = [r["alignment_score"] for r in prompt_flux_results]
                avg_flux_score = np.mean(flux_scores)
                avg_flux_alignment = np.mean(flux_alignments)
                print(f"\n📊 FLUX Average: Score={avg_flux_score:.4f}, Alignment={avg_flux_alignment:.4f}")
                flux_results.extend(prompt_flux_results)
            
            if prompt_sdxl_results:
                sdxl_scores = [r["validation_score"] for r in prompt_sdxl_results]
                sdxl_alignments = [r["alignment_score"] for r in prompt_sdxl_results]
                avg_sdxl_score = np.mean(sdxl_scores)
                avg_sdxl_alignment = np.mean(sdxl_alignments)
                print(f"📊 SDXL Average: Score={avg_sdxl_score:.4f}, Alignment={avg_sdxl_alignment:.4f}")
                sdxl_results.extend(prompt_sdxl_results)
            
            # Compare for this prompt
            if prompt_flux_results and prompt_sdxl_results:
                if avg_sdxl_score > avg_flux_score:
                    print(f"🏆 SDXL wins for this prompt! (+{avg_sdxl_score - avg_flux_score:.4f})")
                elif avg_flux_score > avg_sdxl_score:
                    print(f"🏆 FLUX wins for this prompt! (+{avg_flux_score - avg_sdxl_score:.4f})")
                else:
                    print("🤝 Tie for this prompt")
        
        # Overall comparison
        print(f"\n{'='*80}")
        print("📈 OVERALL COMPARISON RESULTS")
        print(f"{'='*80}")
        
        if flux_results and sdxl_results:
            # Calculate overall statistics
            flux_scores = [r["validation_score"] for r in flux_results]
            flux_alignments = [r["alignment_score"] for r in flux_results]
            sdxl_scores = [r["validation_score"] for r in sdxl_results]
            sdxl_alignments = [r["alignment_score"] for r in sdxl_results]
            
            # FLUX statistics
            flux_avg_score = np.mean(flux_scores)
            flux_avg_alignment = np.mean(flux_alignments)
            flux_std_score = np.std(flux_scores)
            flux_std_alignment = np.std(flux_alignments)
            flux_min_score = np.min(flux_scores)
            flux_max_score = np.max(flux_scores)
            flux_zero_scores = sum(1 for s in flux_scores if s == 0.0)
            
            # SDXL statistics
            sdxl_avg_score = np.mean(sdxl_scores)
            sdxl_avg_alignment = np.mean(sdxl_alignments)
            sdxl_std_score = np.std(sdxl_scores)
            sdxl_std_alignment = np.std(sdxl_alignments)
            sdxl_min_score = np.min(sdxl_scores)
            sdxl_max_score = np.max(sdxl_scores)
            sdxl_zero_scores = sum(1 for s in sdxl_scores if s == 0.0)
            
            print(f"\n🔍 FLUX Results ({len(flux_results)} generations):")
            print(f"   Average Score: {flux_avg_score:.4f} ± {flux_std_score:.4f}")
            print(f"   Average Alignment: {flux_avg_alignment:.4f} ± {flux_std_alignment:.4f}")
            print(f"   Score Range: {flux_min_score:.4f} - {flux_max_score:.4f}")
            print(f"   Zero Scores: {flux_zero_scores}/{len(flux_results)} ({flux_zero_scores/len(flux_results)*100:.1f}%)")
            
            print(f"\n🔍 SDXL Results ({len(sdxl_results)} generations):")
            print(f"   Average Score: {sdxl_avg_score:.4f} ± {sdxl_std_score:.4f}")
            print(f"   Average Alignment: {sdxl_avg_alignment:.4f} ± {sdxl_std_alignment:.4f}")
            print(f"   Score Range: {sdxl_min_score:.4f} - {sdxl_max_score:.4f}")
            print(f"   Zero Scores: {sdxl_zero_scores}/{len(sdxl_results)} ({sdxl_zero_scores/len(sdxl_results)*100:.1f}%)")
            
            # Determine winner
            print(f"\n🏆 WINNER ANALYSIS:")
            score_diff = sdxl_avg_score - flux_avg_score
            alignment_diff = sdxl_avg_alignment - flux_avg_alignment
            
            if score_diff > 0:
                print(f"   SDXL wins overall score by +{score_diff:.4f}")
            else:
                print(f"   FLUX wins overall score by +{abs(score_diff):.4f}")
            
            if alignment_diff > 0:
                print(f"   SDXL wins alignment by +{alignment_diff:.4f}")
            else:
                print(f"   FLUX wins alignment by +{abs(alignment_diff):.4f}")
            
            # Zero score analysis
            if flux_zero_scores > sdxl_zero_scores:
                print(f"   SDXL has fewer zero scores: {sdxl_zero_scores} vs {flux_zero_scores}")
            elif sdxl_zero_scores > flux_zero_scores:
                print(f"   FLUX has fewer zero scores: {flux_zero_scores} vs {sdxl_zero_scores}")
            else:
                print(f"   Both models have same number of zero scores: {flux_zero_scores}")
            
            # Recommendation
            print(f"\n💡 RECOMMENDATION:")
            if sdxl_avg_score > flux_avg_score and sdxl_zero_scores < flux_zero_scores:
                print("   Use SDXL for better overall performance and fewer zero scores")
            elif flux_avg_score > sdxl_avg_score and flux_zero_scores < sdxl_zero_scores:
                print("   Use FLUX for better overall performance and fewer zero scores")
            elif sdxl_zero_scores < flux_zero_scores:
                print("   Use SDXL to avoid zero scores (better alignment)")
            elif flux_zero_scores < sdxl_zero_scores:
                print("   Use FLUX to avoid zero scores (better alignment)")
            else:
                print("   Both models perform similarly - choose based on other factors")
        
        # Save detailed results
        comparison_results = {
            "timestamp": time.time(),
            "test_config": {
                "num_tests_per_prompt": num_tests,
                "total_prompts": len(self.test_prompts),
                "flux_server": self.flux_url,
                "sdxl_server": self.sdxl_url,
                "validation_server": self.validation_url
            },
            "flux_results": flux_results,
            "sdxl_results": sdxl_results,
            "summary": {
                "flux_avg_score": flux_avg_score if flux_results else 0.0,
                "flux_avg_alignment": flux_avg_alignment if flux_results else 0.0,
                "flux_zero_scores": flux_zero_scores if flux_results else 0,
                "sdxl_avg_score": sdxl_avg_score if sdxl_results else 0.0,
                "sdxl_avg_alignment": sdxl_avg_alignment if sdxl_results else 0.0,
                "sdxl_zero_scores": sdxl_zero_scores if sdxl_results else 0,
                "winner": "SDXL" if sdxl_avg_score > flux_avg_score else "FLUX" if flux_results and sdxl_results else "Inconclusive"
            }
        }
        
        with open("sdxl_vs_flux_comparison.json", "w") as f:
            json.dump(comparison_results, f, indent=2)
        
        print(f"\n💾 Detailed results saved to sdxl_vs_flux_comparison.json")
        
        return comparison_results

def main():
    parser = argparse.ArgumentParser(description="Compare SDXL vs FLUX for TRELLIS alignment")
    parser.add_argument("--flux-server", default="http://localhost:8096",
                       help="FLUX generation server URL")
    parser.add_argument("--sdxl-server", default="http://localhost:8097",
                       help="SDXL generation server URL")
    parser.add_argument("--validation-server", default="http://localhost:10006",
                       help="Validation server URL")
    parser.add_argument("--num-tests", type=int, default=2,
                       help="Number of tests per prompt per model")
    
    args = parser.parse_args()
    
    # Initialize comparison
    comparison = ModelComparison(
        flux_server_url=args.flux_server,
        sdxl_server_url=args.sdxl_server,
        validation_server_url=args.validation_server
    )
    
    # Run comparison
    results = comparison.compare_models(num_tests=args.num_tests)
    
    print(f"\n✅ Comparison completed!")

if __name__ == "__main__":
    main() 