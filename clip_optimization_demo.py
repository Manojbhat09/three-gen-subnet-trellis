#!/usr/bin/env python3
"""
CLIP Alignment Optimization Demo
Purpose: Demonstrate the complete prompt optimization system with real examples
"""

import asyncio
import time
import requests
import json
from pathlib import Path
from typing import List, Dict, Any


class CLIPOptimizationDemo:
    """Demo class to showcase the CLIP optimization system"""
    
    def __init__(self, server_url: str = "http://localhost:8098"):
        self.server_url = server_url
        self.results = []
    
    def check_server_health(self) -> bool:
        """Check if the optimization server is running"""
        try:
            response = requests.get(f"{self.server_url}/health/", timeout=5)
            return response.status_code == 200
        except:
            return False
    
    async def demo_single_optimization(self, prompt: str, target_score: float = 0.8) -> Dict[str, Any]:
        """Demonstrate single prompt optimization"""
        print(f"\n🎯 OPTIMIZING SINGLE PROMPT")
        print(f"=" * 60)
        print(f"Original prompt: '{prompt}'")
        print(f"Target CLIP score: {target_score}")
        
        try:
            response = requests.post(
                f"{self.server_url}/optimize_prompt/",
                data={
                    'prompt': prompt,
                    'find_optimal_lora': True,
                    'target_score': target_score
                },
                timeout=300
            )
            
            if response.status_code == 200:
                result = response.json()
                
                print(f"\n📊 OPTIMIZATION RESULTS:")
                print(f"   ✅ Status: {result['status']}")
                print(f"   📝 Optimized prompt: '{result['optimized_prompt']}'")
                print(f"   📈 Original score: {result['original_score']:.4f}")
                print(f"   📈 Final score: {result['final_score']:.4f}")
                print(f"   📈 Normalized score: {result['normalized_score']:.4f}")
                print(f"   📈 Improvement: {result['improvement']:+.4f}")
                print(f"   🎭 Validation status: {result['validation_status']}")
                print(f"   🎯 Task fidelity: {result['task_fidelity']}")
                print(f"   🎨 Optimal LoRA: {result['optimal_lora']}")
                print(f"   ⏱️ Optimization time: {result['optimization_time']:.1f}s")
                
                return result
            else:
                print(f"❌ Optimization failed: HTTP {response.status_code}")
                return None
                
        except Exception as e:
            print(f"❌ Optimization error: {e}")
            return None
    
    async def demo_feedback_loop(self, prompt: str, lora_endpoint: str = "isometric_3d") -> Dict[str, Any]:
        """Demonstrate CLIP feedback loop optimization"""
        print(f"\n🔄 CLIP FEEDBACK LOOP DEMO")
        print(f"=" * 60)
        print(f"Prompt: '{prompt}'")
        print(f"LoRA endpoint: {lora_endpoint}")
        
        try:
            response = requests.post(
                f"{self.server_url}/clip_feedback_loop/",
                data={
                    'prompt': prompt,
                    'lora_endpoint': lora_endpoint,
                    'max_iterations': 3
                },
                timeout=300
            )
            
            if response.status_code == 200:
                result = response.json()
                
                print(f"\n📊 FEEDBACK LOOP RESULTS:")
                print(f"   ✅ Status: {result['status']}")
                print(f"   📝 Original prompt: '{result['original_prompt']}'")
                print(f"   📝 Optimized prompt: '{result['optimized_prompt']}'")
                print(f"   🎨 LoRA endpoint: {result['lora_endpoint']}")
                print(f"   📈 Original score: {result['original_score']:.4f} (normalized: {result['normalized_original']:.4f})")
                print(f"   📈 Optimized score: {result['optimized_score']:.4f} (normalized: {result['normalized_optimized']:.4f})")
                print(f"   📈 Improvement: {result['improvement']:+.4f}")
                print(f"   🔧 Strategy used: {result['strategy_used']}")
                print(f"   🔄 Iterations: {result['iterations']}")
                
                return result
            else:
                print(f"❌ Feedback loop failed: HTTP {response.status_code}")
                return None
                
        except Exception as e:
            print(f"❌ Feedback loop error: {e}")
            return None
    
    async def demo_optimize_and_generate(self, prompt: str, target_score: float = 0.8) -> Dict[str, Any]:
        """Demonstrate complete optimization + generation pipeline"""
        print(f"\n🚀 OPTIMIZE-AND-GENERATE DEMO")
        print(f"=" * 60)
        print(f"Prompt: '{prompt}'")
        print(f"Target score: {target_score}")
        
        try:
            response = requests.post(
                f"{self.server_url}/optimize_and_generate/",
                data={
                    'prompt': prompt,
                    'target_score': target_score,
                    'return_compressed': True
                },
                timeout=600  # 10 minutes for full pipeline
            )
            
            if response.status_code == 200:
                result = response.json()
                
                print(f"\n📊 COMPLETE PIPELINE RESULTS:")
                print(f"   ✅ Status: {result['status']}")
                print(f"   📝 Original prompt: '{result['prompt']}'")
                print(f"   📝 Optimized prompt: '{result['optimized_prompt']}'")
                print(f"   🎨 Optimal LoRA: {result['optimal_lora']}")
                print(f"   📈 Optimization improvement: {result['optimization_improvement']:+.4f}")
                print(f"   📈 Normalized score: {result['optimization_normalized_score']:.4f}")
                print(f"   🏗️ Generation time: {result['generation_time']:.1f}s")
                print(f"   📦 PLY size: {result['ply_size_bytes']:,} bytes")
                if 'compressed_size_bytes' in result:
                    print(f"   🗜️ Compressed size: {result['compressed_size_bytes']:,} bytes")
                    print(f"   🗜️ Compression ratio: {result['compression_ratio']:.2f}x")
                
                return result
            else:
                print(f"❌ Pipeline failed: HTTP {response.status_code}")
                return None
                
        except Exception as e:
            print(f"❌ Pipeline error: {e}")
            return None
    
    async def demo_batch_optimization(self, prompts: List[str]) -> List[Dict[str, Any]]:
        """Demonstrate batch optimization of multiple prompts"""
        print(f"\n📊 BATCH OPTIMIZATION DEMO")
        print(f"=" * 60)
        print(f"Optimizing {len(prompts)} prompts...")
        
        results = []
        for i, prompt in enumerate(prompts, 1):
            print(f"\n--- Prompt {i}/{len(prompts)} ---")
            result = await self.demo_single_optimization(prompt)
            if result:
                results.append(result)
                
                # Brief summary
                improvement = result['improvement']
                normalized_score = result['normalized_score']
                status = "✅" if normalized_score >= 0.8 else "🟡" if normalized_score >= 0.6 else "🟠" if normalized_score >= 0.3 else "❌"
                print(f"   {status} Score: {normalized_score:.3f} (Δ{improvement:+.3f})")
            
            # Small delay between requests
            await asyncio.sleep(1)
        
        # Summary statistics
        if results:
            improvements = [r['improvement'] for r in results]
            scores = [r['normalized_score'] for r in results]
            
            print(f"\n📈 BATCH SUMMARY:")
            print(f"   Processed: {len(results)}/{len(prompts)} prompts")
            print(f"   Average improvement: {sum(improvements)/len(improvements):+.4f}")
            print(f"   Average final score: {sum(scores)/len(scores):.4f}")
            print(f"   Best improvement: {max(improvements):+.4f}")
            print(f"   Best final score: {max(scores):.4f}")
            
            # Count by validation status
            excellent = sum(1 for s in scores if s >= 0.8)
            good = sum(1 for s in scores if 0.6 <= s < 0.8)
            poor = sum(1 for s in scores if 0.3 <= s < 0.6)
            fail = sum(1 for s in scores if s < 0.3)
            
            print(f"   Validation breakdown:")
            print(f"     ✅ Excellent (≥0.8): {excellent}/{len(results)} ({excellent/len(results)*100:.1f}%)")
            print(f"     🟡 Good (0.6-0.8): {good}/{len(results)} ({good/len(results)*100:.1f}%)")
            print(f"     🟠 Poor (0.3-0.6): {poor}/{len(results)} ({poor/len(results)*100:.1f}%)")
            print(f"     ❌ Fail (<0.3): {fail}/{len(results)} ({fail/len(results)*100:.1f}%)")
        
        return results
    
    async def demo_lora_comparison(self, prompt: str) -> Dict[str, Any]:
        """Compare optimization across different LoRA endpoints"""
        print(f"\n🎨 LORA COMPARISON DEMO")
        print(f"=" * 60)
        print(f"Prompt: '{prompt}'")
        
        lora_endpoints = [
            "isometric_3d", "live_3d", "game_assets", "patched_realism",
            "tf2_style", "baolei", "cartoon_3d", "cinema"
        ]
        
        results = {}
        for lora in lora_endpoints:
            print(f"\n--- Testing {lora} ---")
            result = await self.demo_feedback_loop(prompt, lora)
            if result:
                results[lora] = {
                    'final_score': result['optimized_score'],
                    'normalized_score': result['normalized_optimized'],
                    'improvement': result['improvement'],
                    'optimized_prompt': result['optimized_prompt']
                }
                print(f"   Score: {result['normalized_optimized']:.3f} (Δ{result['improvement']:+.3f})")
            
            await asyncio.sleep(2)  # Delay between LoRA tests
        
        if results:
            # Find best performing LoRA
            best_lora = max(results.keys(), key=lambda k: results[k]['normalized_score'])
            best_score = results[best_lora]['normalized_score']
            
            print(f"\n🏆 BEST LORA: {best_lora}")
            print(f"   Score: {best_score:.4f}")
            print(f"   Optimized prompt: '{results[best_lora]['optimized_prompt']}'")
            
            # Sort by performance
            sorted_results = sorted(results.items(), key=lambda x: x[1]['normalized_score'], reverse=True)
            print(f"\n📊 PERFORMANCE RANKING:")
            for i, (lora, data) in enumerate(sorted_results, 1):
                score = data['normalized_score']
                improvement = data['improvement']
                status = "✅" if score >= 0.8 else "🟡" if score >= 0.6 else "🟠" if score >= 0.3 else "❌"
                print(f"   {i}. {status} {lora}: {score:.3f} (Δ{improvement:+.3f})")
        
        return results
    
    def save_results(self, filename: str = "optimization_demo_results.json"):
        """Save all demo results to file"""
        with open(filename, 'w') as f:
            json.dump(self.results, f, indent=2)
        print(f"\n💾 Results saved to {filename}")


async def main():
    """Run the complete optimization demo"""
    print("🚀 CLIP ALIGNMENT OPTIMIZATION SYSTEM DEMO")
    print("=" * 80)
    
    demo = CLIPOptimizationDemo()
    
    # Check server health
    if not demo.check_server_health():
        print("❌ Optimization server is not running!")
        print("Please start the server first: python trellis_subnit_server_mix_lora_flash.py")
        return
    
    print("✅ Optimization server is running")
    
    # Demo prompts
    test_prompts = [
        "a blue ceramic vase",
        "red sports car on a racing track",
        "wooden chair with metal legs",
        "glass sphere on marble table",
        "steampunk robot with copper details",
        "medieval castle with stone walls",
        "tropical fish in coral reef",
        "vintage camera on wooden desk"
    ]
    
    # Demo 1: Single optimization
    print(f"\n{'='*20} DEMO 1: SINGLE OPTIMIZATION {'='*20}")
    result1 = await demo.demo_single_optimization(test_prompts[0])
    if result1:
        demo.results.append({'demo': 'single_optimization', 'result': result1})
    
    # Demo 2: CLIP feedback loop
    print(f"\n{'='*20} DEMO 2: CLIP FEEDBACK LOOP {'='*20}")
    result2 = await demo.demo_feedback_loop(test_prompts[1], "live_3d")
    if result2:
        demo.results.append({'demo': 'feedback_loop', 'result': result2})
    
    # Demo 3: Complete pipeline (optimize + generate)
    print(f"\n{'='*20} DEMO 3: OPTIMIZE & GENERATE {'='*20}")
    result3 = await demo.demo_optimize_and_generate(test_prompts[2])
    if result3:
        demo.results.append({'demo': 'optimize_and_generate', 'result': result3})
    
    # Demo 4: LoRA comparison
    print(f"\n{'='*20} DEMO 4: LORA COMPARISON {'='*20}")
    result4 = await demo.demo_lora_comparison(test_prompts[3])
    if result4:
        demo.results.append({'demo': 'lora_comparison', 'result': result4})
    
    # Demo 5: Batch optimization
    print(f"\n{'='*20} DEMO 5: BATCH OPTIMIZATION {'='*20}")
    batch_prompts = test_prompts[4:7]  # Use 3 prompts for batch demo
    result5 = await demo.demo_batch_optimization(batch_prompts)
    if result5:
        demo.results.append({'demo': 'batch_optimization', 'result': result5})
    
    # Save results
    demo.save_results()
    
    print(f"\n🎯 DEMO COMPLETE!")
    print("=" * 80)
    print("The optimization system demonstrates:")
    print("  1. ✅ Single prompt optimization with CLIP feedback")
    print("  2. ✅ Iterative refinement through multiple strategies")
    print("  3. ✅ Optimal LoRA endpoint selection")
    print("  4. ✅ Complete generation pipeline integration")
    print("  5. ✅ Batch processing capabilities")
    print("  6. ✅ Performance comparison across LoRA endpoints")
    print("\nNext steps:")
    print("  • Integrate with continuous simulator")
    print("  • Add image interrogator feedback")
    print("  • Implement convergence tracking")
    print("  • Add episodic memory integration")


if __name__ == "__main__":
    asyncio.run(main()) 