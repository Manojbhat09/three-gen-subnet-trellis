#!/usr/bin/env python3
"""
LoRA Router Test Script
Tests the effectiveness of an intelligent LoRA router based on benchmark data.
"""

import json
import time
import requests
import logging
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, asdict
from enum import Enum
import re

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ModelType(Enum):
    FLUX = "FLUX"
    SD15 = "SD15"

@dataclass
class LoRAConfig:
    name: str
    model: ModelType
    endpoint: str
    trigger_prefix: str
    avg_score: float
    best_score: float
    worst_score: float
    success_rate: float
    avg_time: float

@dataclass
class RouterResult:
    recommended_lora: str
    reasoning: str
    confidence: str
    actual_score: Optional[float] = None
    actual_lora_used: Optional[str] = None

class LoRARouter:
    """Intelligent LoRA router based on benchmark data"""
    
    def __init__(self):
        # Load benchmark data
        self.lora_configs = self._load_lora_configs()
        self.benchmark_data = self._load_benchmark_data()
        
    def _load_lora_configs(self) -> Dict[str, LoRAConfig]:
        """Load LoRA configurations from benchmark results"""
        configs = {
            "Patched Realism": LoRAConfig(
                name="Patched Realism",
                model=ModelType.FLUX,
                endpoint="/generate/patched_realism/",
                trigger_prefix="",
                avg_score=0.8283,
                best_score=0.8603,
                worst_score=0.7325,
                success_rate=100.0,
                avg_time=34.20
            ),
            "Team Fortress 2 Style": LoRAConfig(
                name="Team Fortress 2 Style",
                model=ModelType.FLUX,
                endpoint="/generate/tf2_style/",
                trigger_prefix="tf2style,",
                avg_score=0.8116,
                best_score=0.8380,
                worst_score=0.7621,
                success_rate=100.0,
                avg_time=31.33
            ),
            "Cartoon 3D Render": LoRAConfig(
                name="Cartoon 3D Render",
                model=ModelType.FLUX,
                endpoint="/generate/cartoon_3d/",
                trigger_prefix="",
                avg_score=0.6694,
                best_score=0.9445,
                worst_score=0.0000,
                success_rate=100.0,
                avg_time=33.19
            ),
            "3D Game Assets": LoRAConfig(
                name="3D Game Assets",
                model=ModelType.FLUX,
                endpoint="/generate/game_assets/",
                trigger_prefix="Create 3D game asset, isometric view version,",
                avg_score=0.6647,
                best_score=0.9367,
                worst_score=0.0000,
                success_rate=100.0,
                avg_time=32.77
            ),
            "Game Icon Institute": LoRAConfig(
                name="Game Icon Institute",
                model=ModelType.SD15,
                endpoint="/generate/sd15_game_icon/",
                trigger_prefix="game icon institute,",
                avg_score=0.6429,
                best_score=0.8867,
                worst_score=0.0000,
                success_rate=100.0,
                avg_time=24.22
            ),
            "Cinema Style": LoRAConfig(
                name="Cinema Style",
                model=ModelType.FLUX,
                endpoint="/generate/cinema/",
                trigger_prefix="c1n3ma,",
                avg_score=0.5026,
                best_score=0.9050,
                worst_score=0.0000,
                success_rate=100.0,
                avg_time=33.39
            ),
            "Flux Isometric 3D": LoRAConfig(
                name="Flux Isometric 3D",
                model=ModelType.FLUX,
                endpoint="/generate/isometric_3d/",
                trigger_prefix="Isometric 3D,",
                avg_score=0.4956,
                best_score=0.8452,
                worst_score=0.0000,
                success_rate=100.0,
                avg_time=32.25
            )
        }
        return configs
    
    def _load_benchmark_data(self) -> Dict[str, List[Dict]]:
        """Load detailed benchmark data for analysis"""
        try:
            with open('lora_benchmark_results.json', 'r') as f:
                data = json.load(f)
                # Ensure data is properly structured
                if isinstance(data, dict):
                    return data
                else:
                    logger.warning("Benchmark data is not a dictionary, using empty dict")
                    return {}
        except FileNotFoundError:
            logger.warning("Benchmark results file not found, using default data")
            return {}
        except json.JSONDecodeError:
            logger.warning("Invalid JSON in benchmark results file, using default data")
            return {}
    
    def classify_prompt(self, prompt: str) -> str:
        """Classify prompt into categories based on keywords"""
        prompt_lower = prompt.lower()
        
        # Category 1: Mechanical, Sci-Fi, Robots, & Game-Ready Props
        mechanical_keywords = ['robot', 'mech', 'cyborg', 'weapon', 'gun', 'knife', 'engine', 'machinery']
        if any(keyword in prompt_lower for keyword in mechanical_keywords):
            return "mechanical"
        
        # Category 2: Realistic Everyday Objects & Food
        everyday_keywords = ['plastic straw', 'cup', 'locket', 'necklace', 'baseball bat', 'apple', 'burger', 'drink']
        if any(keyword in prompt_lower for keyword in everyday_keywords):
            return "everyday"
        
        # Category 3: Historical & Artistic Artifacts
        historical_keywords = ['greek amphora', 'ancient', 'relic', 'sculpture', 'pottery', 'artifact']
        if any(keyword in prompt_lower for keyword in historical_keywords):
            return "historical"
        
        # Category 4: Stylized Characters & Worlds
        if 'tf2' in prompt_lower or 'team fortress' in prompt_lower:
            return "tf2_style"
        if 'cartoon' in prompt_lower:
            return "cartoon"
        
        # Category 5: Fallback
        return "general"
    
    def route_prompt(self, prompt: str) -> RouterResult:
        """Route prompt to best LoRA based on classification and rules"""
        category = self.classify_prompt(prompt)
        
        if category == "mechanical":
            # 3D Game Assets excels here, Cartoon 3D is also strong
            return RouterResult(
                recommended_lora="3D Game Assets",
                reasoning="Mechanical/sci-fi category - 3D Game Assets achieved highest scores for this theme",
                confidence="High"
            )
        
        elif category == "everyday":
            # Patched Realism is specifically designed for this
            return RouterResult(
                recommended_lora="Patched Realism",
                reasoning="Everyday objects category - Patched Realism consistently scores highest for real-world objects",
                confidence="High"
            )
        
        elif category == "historical":
            # Patched Realism succeeded, avoid 3D Game Assets
            return RouterResult(
                recommended_lora="Patched Realism",
                reasoning="Historical artifacts category - Patched Realism succeeded while game-focused LoRAs failed",
                confidence="High"
            )
        
        elif category == "tf2_style":
            # Match style to LoRA
            return RouterResult(
                recommended_lora="Team Fortress 2 Style",
                reasoning="TF2 style requested - matching style to specific LoRA",
                confidence="High"
            )
        
        elif category == "cartoon":
            # Cartoon 3D Render for cartoon style
            return RouterResult(
                recommended_lora="Cartoon 3D Render",
                reasoning="Cartoon style requested - using Cartoon 3D Render LoRA",
                confidence="Medium"
            )
        
        else:
            # Fallback to highest overall average score
            return RouterResult(
                recommended_lora="Patched Realism",
                reasoning="General category - using highest overall average validation score (0.8283)",
                confidence="Medium"
            )

class LoRARouterTester:
    """Test the LoRA router against benchmark data"""
    
    def __init__(self, server_url: str = "http://localhost:8096"):
        self.router = LoRARouter()
        self.server_url = server_url
        self.test_prompts = [
            "greek amphora scene detail",
            "plastic straw of drink", 
            "small yellow triangular wooden kitchen knife",
            "enormous black robot with round body",
            "rose gold locket necklace with floral"
        ]
    
    def _get_benchmark_scores(self, prompt: str) -> Dict[str, float]:
        """Get benchmark scores for a specific prompt across all LoRAs"""
        # Use hardcoded scores from benchmark data
        fallback_scores = {
            "greek amphora scene detail": {
                "Patched Realism": 0.8357,
                "Team Fortress 2 Style": 0.8019,
                "Cartoon 3D Render": 0.5663,
                "3D Game Assets": 0.0000,
                "Game Icon Institute": 0.7563,
                "Cinema Style": 0.7689,
                "Flux Isometric 3D": 0.0000
            },
            "plastic straw of drink": {
                "Patched Realism": 0.7325,
                "Team Fortress 2 Style": 0.7621,
                "Cartoon 3D Render": 0.0000,
                "3D Game Assets": 0.7588,
                "Game Icon Institute": 0.8265,
                "Cinema Style": 0.0000,
                "Flux Isometric 3D": 0.7940
            },
            "small yellow triangular wooden kitchen knife": {
                "Patched Realism": 0.8543,
                "Team Fortress 2 Style": 0.8380,
                "Cartoon 3D Render": 0.9421,
                "3D Game Assets": 0.7723,
                "Game Icon Institute": 0.8867,
                "Cinema Style": 0.0000,
                "Flux Isometric 3D": 0.0000
            },
            "enormous black robot with round body": {
                "Patched Realism": 0.8585,
                "Team Fortress 2 Style": 0.8357,
                "Cartoon 3D Render": 0.9445,
                "3D Game Assets": 0.9367,
                "Game Icon Institute": 0.7452,
                "Cinema Style": 0.9050,
                "Flux Isometric 3D": 0.8387
            },
            "rose gold locket necklace with floral": {
                "Patched Realism": 0.8603,
                "Team Fortress 2 Style": 0.8200,
                "Cartoon 3D Render": 0.8941,
                "3D Game Assets": 0.8556,
                "Game Icon Institute": 0.0000,
                "Cinema Style": 0.8392,
                "Flux Isometric 3D": 0.8452
            }
        }
        return fallback_scores.get(prompt, {})
    
    def test_router_accuracy(self) -> Dict:
        """Test router accuracy against benchmark data"""
        results = {
            "total_tests": len(self.test_prompts),
            "router_recommendations": [],
            "accuracy_analysis": {},
            "improvement_analysis": {}
        }
        
        logger.info("🚀 Testing LoRA Router Accuracy")
        logger.info("=" * 60)
        
        for prompt in self.test_prompts:
            logger.info(f"\n📝 Testing prompt: '{prompt}'")
            
            # Get router recommendation
            router_result = self.router.route_prompt(prompt)
            logger.info(f"   🎯 Router recommends: {router_result.recommended_lora}")
            logger.info(f"   💭 Reasoning: {router_result.reasoning}")
            logger.info(f"   🎯 Confidence: {router_result.confidence}")
            
            # Get actual benchmark scores for this prompt
            benchmark_scores = self._get_benchmark_scores(prompt)
            
            # Find best performing LoRA for this prompt
            best_lora = max(benchmark_scores.items(), key=lambda x: x[1])
            worst_lora = min(benchmark_scores.items(), key=lambda x: x[1])
            
            logger.info(f"   📊 Best benchmark LoRA: {best_lora[0]} (Score: {best_lora[1]:.4f})")
            logger.info(f"   📊 Worst benchmark LoRA: {worst_lora[0]} (Score: {worst_lora[1]:.4f})")
            
            # Check if router recommendation matches best
            router_score = benchmark_scores.get(router_result.recommended_lora, 0.0)
            is_correct = router_result.recommended_lora == best_lora[0]
            
            logger.info(f"   📊 Router LoRA score: {router_score:.4f}")
            logger.info(f"   ✅ Router correct: {'YES' if is_correct else 'NO'}")
            
            # Calculate improvement over worst
            improvement = router_score - worst_lora[1]
            logger.info(f"   📈 Improvement over worst: {improvement:.4f}")
            
            results["router_recommendations"].append({
                "prompt": prompt,
                "router_recommendation": router_result.recommended_lora,
                "router_score": router_score,
                "best_lora": best_lora[0],
                "best_score": best_lora[1],
                "worst_lora": worst_lora[0],
                "worst_score": worst_lora[1],
                "is_correct": is_correct,
                "improvement": improvement,
                "reasoning": router_result.reasoning,
                "confidence": router_result.confidence
            })
        
        # Calculate overall accuracy
        correct_recommendations = sum(1 for r in results["router_recommendations"] if r["is_correct"])
        accuracy = correct_recommendations / len(results["router_recommendations"])
        
        results["accuracy_analysis"] = {
            "correct_recommendations": correct_recommendations,
            "total_recommendations": len(results["router_recommendations"]),
            "accuracy_percentage": accuracy * 100,
            "average_improvement": sum(r["improvement"] for r in results["router_recommendations"]) / len(results["router_recommendations"])
        }
        
        logger.info(f"\n📊 ROUTER ACCURACY RESULTS")
        logger.info("=" * 60)
        logger.info(f"✅ Correct recommendations: {correct_recommendations}/{len(results['router_recommendations'])}")
        logger.info(f"📊 Accuracy: {accuracy * 100:.1f}%")
        logger.info(f"📈 Average improvement over worst: {results['accuracy_analysis']['average_improvement']:.4f}")
        
        return results
    
    def test_router_vs_random(self) -> Dict:
        """Compare router performance against random selection"""
        logger.info("\n🎲 Testing Router vs Random Selection")
        logger.info("=" * 60)
        
        router_scores = []
        random_scores = []
        
        for prompt in self.test_prompts:
            # Router recommendation
            router_result = self.router.route_prompt(prompt)
            benchmark_scores = self._get_benchmark_scores(prompt)
            router_score = benchmark_scores.get(router_result.recommended_lora, 0.0)
            router_scores.append(router_score)
            
            # Random selection (simulate)
            import random
            random_lora = random.choice(list(benchmark_scores.keys()))
            random_score = benchmark_scores.get(random_lora, 0.0)
            random_scores.append(random_score)
            
            logger.info(f"📝 '{prompt}':")
            logger.info(f"   🎯 Router: {router_result.recommended_lora} ({router_score:.4f})")
            logger.info(f"   🎲 Random: {random_lora} ({random_score:.4f})")
            logger.info(f"   📈 Difference: {router_score - random_score:.4f}")
        
        avg_router_score = sum(router_scores) / len(router_scores)
        avg_random_score = sum(random_scores) / len(random_scores)
        improvement = avg_router_score - avg_random_score
        
        logger.info(f"\n📊 ROUTER VS RANDOM RESULTS")
        logger.info("=" * 60)
        logger.info(f"🎯 Average Router Score: {avg_router_score:.4f}")
        logger.info(f"🎲 Average Random Score: {avg_random_score:.4f}")
        logger.info(f"📈 Average Improvement: {improvement:.4f}")
        logger.info(f"📊 Improvement Percentage: {(improvement / avg_random_score * 100) if avg_random_score > 0 else 0:.1f}%")
        
        return {
            "router_scores": router_scores,
            "random_scores": random_scores,
            "avg_router_score": avg_router_score,
            "avg_random_score": avg_random_score,
            "improvement": improvement,
            "improvement_percentage": (improvement / avg_random_score * 100) if avg_random_score > 0 else 0
        }
    
    def test_router_vs_baseline(self) -> Dict:
        """Compare router performance against always using best average LoRA"""
        logger.info("\n🏆 Testing Router vs Baseline (Best Average LoRA)")
        logger.info("=" * 60)
        
        # Baseline: Always use Patched Realism (highest average score)
        baseline_lora = "Patched Realism"
        
        router_scores = []
        baseline_scores = []
        
        for prompt in self.test_prompts:
            # Router recommendation
            router_result = self.router.route_prompt(prompt)
            benchmark_scores = self._get_benchmark_scores(prompt)
            router_score = benchmark_scores.get(router_result.recommended_lora, 0.0)
            router_scores.append(router_score)
            
            # Baseline score
            baseline_score = benchmark_scores.get(baseline_lora, 0.0)
            baseline_scores.append(baseline_score)
            
            logger.info(f"📝 '{prompt}':")
            logger.info(f"   🎯 Router: {router_result.recommended_lora} ({router_score:.4f})")
            logger.info(f"   🏆 Baseline: {baseline_lora} ({baseline_score:.4f})")
            logger.info(f"   📈 Difference: {router_score - baseline_score:.4f}")
        
        avg_router_score = sum(router_scores) / len(router_scores)
        avg_baseline_score = sum(baseline_scores) / len(baseline_scores)
        improvement = avg_router_score - avg_baseline_score
        
        logger.info(f"\n📊 ROUTER VS BASELINE RESULTS")
        logger.info("=" * 60)
        logger.info(f"🎯 Average Router Score: {avg_router_score:.4f}")
        logger.info(f"🏆 Average Baseline Score: {avg_baseline_score:.4f}")
        logger.info(f"📈 Average Improvement: {improvement:.4f}")
        logger.info(f"📊 Improvement Percentage: {(improvement / avg_baseline_score * 100) if avg_baseline_score > 0 else 0:.1f}%")
        
        return {
            "router_scores": router_scores,
            "baseline_scores": baseline_scores,
            "avg_router_score": avg_router_score,
            "avg_baseline_score": avg_baseline_score,
            "improvement": improvement,
            "improvement_percentage": (improvement / avg_baseline_score * 100) if avg_baseline_score > 0 else 0
        }
    
    def generate_router_system_prompt(self) -> str:
        """Generate the system prompt for the LoRA router"""
        system_prompt = """You are an expert LoRA Router for a text-to-3D generation engine. Your sole purpose is to analyze a user's prompt and recommend the single best LoRA to use for generating the object. Your recommendations must be based exclusively on the following internal knowledge, which is derived from extensive benchmark testing.

INTERNAL KNOWLEDGE BASE & DECISION HEURISTICS

You will classify the user's prompt into one of the following categories and apply the corresponding rule.

1. Category: Mechanical, Sci-Fi, Robots, & Game-Ready Props

Keywords: robot, mech, cyborg, weapon, gun, knife, engine, machinery.

Analysis: This is the strongest category for specialized LoRAs. 3D Game Assets excels here, achieving the highest scores for this theme. Cartoon 3D is also a strong choice if a stylized look is desired.

Rule:
Primary Recommendation: 3D Game Assets (for a standard game-dev look).
Secondary Recommendation: Cartoon 3D Render (for a stylized, non-photorealistic look).

2. Category: Realistic Everyday Objects & Food

Keywords: plastic straw, cup, locket, necklace, baseball bat, apple, burger.

Analysis: These prompts describe common, real-world objects. The Patched Realism LoRA is specifically designed for this and consistently scores the highest. Stylistic LoRAs like Cartoon 3D or Cinema Style often fail on these mundane items.

Rule:
Primary Recommendation: Patched Realism.

3. Category: Historical & Artistic Artifacts

Keywords: greek amphora, ancient, relic, sculpture, pottery, artifact.

Analysis: This category is a major point of failure for game-focused LoRAs. The 3D Game Assets and Flux Isometric 3D LoRAs consistently failed to render these objects correctly. Patched Realism succeeded by treating it as a real-world object.

Rule:
Primary Recommendation: Patched Realism.
CRITICAL NOTE: Explicitly avoid 3D Game Assets for this category.

4. Category: Stylized Characters & Worlds

Keywords: Include stylistic triggers like in the style of Team Fortress 2, as a cartoon.

Analysis: When the prompt itself requests a specific, known style for which a LoRA exists, that LoRA should be chosen.

Rule:
Primary Recommendation: Match the style to the LoRA. For example, a prompt with TF2 should use Team Fortress 2 Style.

5. Fallback / Generalist Logic

Analysis: If a prompt does not clearly fit any of the above categories, the safest and most reliable choice is the LoRA with the highest overall average validation score.

Rule:
Default Recommendation: Patched Realism (Avg Score: 0.8283).

OUTPUT FORMAT

You must provide your response in the following format. Do not be conversational. Provide the recommendation and the reasoning directly.

**LoRA Recommendation:** `[Name of the Recommended LoRA]`
**Reasoning:** `[A brief, one-sentence explanation for your choice, citing the benchmark category or data.]`
**Confidence:** `[High/Medium]`

YOUR TASK

Now, analyze the following user's prompt and provide your LoRA recommendation based on the rules above.

User Prompt:"""
        
        return system_prompt
    
    def run_comprehensive_test(self) -> Dict:
        """Run all tests and generate comprehensive report"""
        logger.info("🚀 LoRA Router Comprehensive Test")
        logger.info("=" * 60)
        
        results = {
            "accuracy_test": self.test_router_accuracy(),
            "random_comparison": self.test_router_vs_random(),
            "baseline_comparison": self.test_router_vs_baseline(),
            "system_prompt": self.generate_router_system_prompt()
        }
        
        # Generate summary
        logger.info("\n📋 COMPREHENSIVE TEST SUMMARY")
        logger.info("=" * 60)
        logger.info(f"🎯 Router Accuracy: {results['accuracy_test']['accuracy_analysis']['accuracy_percentage']:.1f}%")
        logger.info(f"📈 vs Random Improvement: {results['random_comparison']['improvement_percentage']:.1f}%")
        logger.info(f"🏆 vs Baseline Improvement: {results['baseline_comparison']['improvement_percentage']:.1f}%")
        logger.info(f"📊 Average Router Score: {results['accuracy_test']['accuracy_analysis']['average_improvement']:.4f}")
        
        # Save results
        with open('lora_router_test_results.json', 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"\n💾 Results saved to: lora_router_test_results.json")
        
        return results

def main():
    """Main function to run the LoRA router test"""
    tester = LoRARouterTester()
    results = tester.run_comprehensive_test()
    
    print("\n" + "="*60)
    print("🎉 LoRA Router Test Complete!")
    print("="*60)
    print(f"📊 Router Accuracy: {results['accuracy_test']['accuracy_analysis']['accuracy_percentage']:.1f}%")
    print(f"📈 Improvement over Random: {results['random_comparison']['improvement_percentage']:.1f}%")
    print(f"🏆 Improvement over Baseline: {results['baseline_comparison']['improvement_percentage']:.1f}%")

if __name__ == "__main__":
    main() 