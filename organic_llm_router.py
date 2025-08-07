#!/usr/bin/env python3
"""
Organic LLM-Based LoRA Router
Uses principles and patterns, not direct examples, for true generalization.
"""

import json
import time
import requests
import logging
import re
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, asdict
from enum import Enum

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

class OrganicLLMLoRARouter:
    """Organic LLM router using principles and pattern analysis"""
    
    def __init__(self, llm_endpoint: str = "http://localhost:11434/api/generate"):
        self.llm_endpoint = llm_endpoint
        self.lora_configs = self._load_lora_configs()
        self.system_prompt = self._create_organic_system_prompt()
        
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
    
    def _create_organic_system_prompt(self) -> str:
        """Create organic system prompt based on principles and patterns"""
        return """You are an expert LoRA Router for 3D generation. Analyze the user's prompt and recommend the best LoRA based on object characteristics and artistic principles.

AVAILABLE LORAS:
- Patched Realism (avg: 0.8283, range: 0.7325-0.8603) - Most consistent
- Team Fortress 2 Style (avg: 0.8116, range: 0.7621-0.8380) - Consistent stylized
- Cartoon 3D Render (avg: 0.6694, range: 0.0000-0.9445) - High variance, extreme peaks
- 3D Game Assets (avg: 0.6647, range: 0.0000-0.9367) - Game-focused, inconsistent
- Game Icon Institute (avg: 0.6429, range: 0.0000-0.8867) - Simple objects specialist
- Cinema Style (avg: 0.5026, range: 0.0000-0.9050) - Cinematic but unreliable
- Flux Isometric 3D (avg: 0.4956, range: 0.0000-0.8452) - Lowest performance

PERFORMANCE ANALYSIS PRINCIPLES:

1. CONSISTENCY vs SPECIALIZATION PATTERN:
   - High consistency (low variance) → Reliable for general use
   - High variance → Specialist that either excels or fails completely
   - Average score doesn't tell the full story - range matters

2. OBJECT COMPLEXITY INDICATORS:
   - Simple geometric forms → Look for "clean", "simple", "basic" descriptors
   - Complex detailed objects → Look for "intricate", "detailed", "ornate" descriptors
   - Material complexity → Consider "textured", "multi-material", "decorated"

3. ARTISTIC STYLE SIGNALS:
   - Realism cues → "realistic", "photorealistic", "lifelike", "accurate"
   - Stylization cues → "cartoon", "stylized", "artistic", "exaggerated"
   - Technical cues → "mechanical", "industrial", "engineered", "precise"

4. SCALE AND PROPORTION CLUES:
   - Miniature indicators → "small", "tiny", "mini", "compact"
   - Character-scale → "human-sized", "wearable", "handheld"
   - Architectural → "large", "massive", "enormous", "towering"

5. CULTURAL/TEMPORAL CONTEXT:
   - Historical → "ancient", "vintage", "classical", "traditional"
   - Modern → "contemporary", "current", "modern", "new"
   - Futuristic → "futuristic", "sci-fi", "advanced", "high-tech"

6. FUNCTIONAL CATEGORIZATION:
   - Utilitarian → "tool", "instrument", "device", "equipment"
   - Decorative → "ornament", "decoration", "accessory", "jewelry"
   - Structural → "furniture", "architecture", "building", "framework"

DECISION FRAMEWORK:
Analyze the prompt for these characteristics and match to LoRA strengths:
- High-variance LoRAs (Cartoon 3D, 3D Game Assets) → Use for their specialties
- Consistent LoRAs (Patched Realism, TF2 Style) → Use for reliability
- Specialist LoRAs (Game Icon Institute) → Use when characteristics align

REASONING PROCESS:
1. Extract key descriptors from prompt
2. Categorize object type and complexity
3. Identify artistic style requirements
4. Match to LoRA performance patterns
5. Consider risk vs reward (consistent vs specialist)

OUTPUT FORMAT:
{"recommended_lora": "LoRA Name", "reasoning": "Analysis-based explanation", "confidence": "High/Medium/Low"}

User Prompt:"""

    def query_llm(self, prompt: str) -> str:
        """Query the LLM with organic reasoning"""
        full_prompt = f"{self.system_prompt}\n\n{prompt}"
        
        payload = {
            "model": "llama3.2:3b",
            "prompt": full_prompt,
            "stream": False,
            "options": {
                "temperature": 0.3,  # Some creativity for organic thinking
                "top_p": 0.9,
                "max_tokens": 200,
                "stop": ["\n\n", "User:", "Prompt:"]
            }
        }
        
        try:
            response = requests.post(self.llm_endpoint, json=payload, timeout=25)
            if response.status_code == 200:
                result = response.json()
                return result.get("response", "").strip()
            else:
                logger.error(f"LLM request failed with status {response.status_code}")
                return self._fallback_response()
        except requests.exceptions.RequestException as e:
            logger.error(f"LLM request failed: {e}")
            return self._fallback_response()
        except Exception as e:
            logger.error(f"Unexpected error: {e}")
            return self._fallback_response()
    
    def _fallback_response(self) -> str:
        """Fallback response when LLM is unavailable"""
        return '{"recommended_lora": "Patched Realism", "reasoning": "LLM unavailable - using most consistent performer", "confidence": "Medium"}'
    
    def parse_llm_response(self, response: str) -> RouterResult:
        """Parse LLM response with multiple strategies"""
        try:
            # Strategy 1: Direct JSON parse
            data = json.loads(response.strip())
            return self._create_result_from_json(data)
            
        except json.JSONDecodeError as e:
            logger.debug(f"Direct JSON parse failed: {e}")
            try:
                # Strategy 2: Extract JSON pattern (more flexible)
                json_match = re.search(r'\{.*?"recommended_lora".*?\}', response, re.DOTALL)
                if json_match:
                    json_str = json_match.group(0)
                    data = json.loads(json_str)
                    return self._create_result_from_json(data)
                
                # Strategy 3: Find any JSON-like structure
                json_match = re.search(r'\{[^{}]*\}', response, re.DOTALL)
                if json_match:
                    json_str = json_match.group(0)
                    data = json.loads(json_str)
                    return self._create_result_from_json(data)
                
                # Strategy 4: Manual parsing with quotes
                lora_match = re.search(r'"recommended_lora":\s*"([^"]+)"', response, re.IGNORECASE)
                reasoning_match = re.search(r'"reasoning":\s*"([^"]+)"', response, re.IGNORECASE)
                confidence_match = re.search(r'"confidence":\s*"([^"]+)"', response, re.IGNORECASE)
                
                if lora_match:
                    lora = lora_match.group(1).strip()
                    reasoning = reasoning_match.group(1).strip() if reasoning_match else "Organic LLM analysis"
                    confidence = confidence_match.group(1).strip() if confidence_match else "Medium"
                    
                    return RouterResult(
                        recommended_lora=self._clean_lora_name(lora),
                        reasoning=reasoning,
                        confidence=confidence
                    )
                    
            except (json.JSONDecodeError, Exception) as e:
                logger.warning(f"JSON extraction failed: {e}")
            
            logger.error(f"Could not parse LLM response: {response[:100]}...")
            return self._create_fallback_result()
                
    def _create_result_from_json(self, data: dict) -> RouterResult:
        """Create RouterResult from parsed JSON"""
        lora = data.get("recommended_lora", "Patched Realism")
        reasoning = data.get("reasoning", "No reasoning provided")
        confidence = data.get("confidence", "Medium")
        
        return RouterResult(
            recommended_lora=self._clean_lora_name(lora),
            reasoning=reasoning,
            confidence=confidence
        )
    
    def _clean_lora_name(self, lora_name: str) -> str:
        """Clean and validate LoRA name"""
        cleaned = lora_name.strip().strip('"\'')
        
        # Check exact matches first
        for known_lora in self.lora_configs.keys():
            if cleaned.lower() == known_lora.lower():
                return known_lora
        
        # Check partial matches
        for known_lora in self.lora_configs.keys():
            if cleaned.lower() in known_lora.lower() or known_lora.lower() in cleaned.lower():
                return known_lora
        
        return cleaned
    
    def _create_fallback_result(self) -> RouterResult:
        """Create fallback result when parsing fails"""
        return RouterResult(
            recommended_lora="Patched Realism",
            reasoning="Failed to parse LLM response, using most consistent LoRA",
            confidence="Low"
        )
    
    def route_prompt(self, prompt: str) -> RouterResult:
        """Route prompt using organic LLM reasoning"""
        logger.info(f"🧠 Organic LLM analyzing: '{prompt}'")
        
        # Query LLM
        llm_response = self.query_llm(prompt)
        logger.debug(f"🧠 Raw response: {llm_response}")
        
        # Parse response
        result = self.parse_llm_response(llm_response)
        
        # Validate the LoRA name exists
        if result.recommended_lora not in self.lora_configs:
            logger.warning(f"LLM recommended unknown LoRA: {result.recommended_lora}")
            result.recommended_lora = "Patched Realism"
            result.reasoning = f"Unknown LoRA recommended, defaulting to most consistent"
            result.confidence = "Low"
        
        return result

class OrganicRouterTester:
    """Test the organic router with various prompts"""
    
    def __init__(self, llm_endpoint: str = "http://localhost:11434/api/generate"):
        self.router = OrganicLLMLoRARouter(llm_endpoint)
        
        # Original test prompts
        self.original_prompts = [
            "greek amphora scene detail",
            "plastic straw of drink", 
            "small yellow triangular wooden kitchen knife",
            "enormous black robot with round body",
            "rose gold locket necklace with floral"
        ]
        
        # Completely new test prompts to test generalization
        self.generalization_prompts = [
            "futuristic laser pistol with chrome finish",
            "tiny ceramic coffee mug with handle",
            "massive medieval castle tower",
            "delicate silver bracelet with gemstones",
            "industrial pipe wrench tool",
            "vintage brass compass with engravings",
            "cartoon monster character with big eyes",
            "sleek smartphone with glass screen",
            "ornate wooden treasure chest",
            "simple white dinner plate"
        ]
        
    def _get_benchmark_scores(self, prompt: str) -> Dict[str, float]:
        """Get benchmark scores for known prompts"""
        known_scores = {
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
        return known_scores.get(prompt, {})
    
    def test_original_prompts(self) -> Dict:
        """Test on original benchmark prompts"""
        results = {
            "total_tests": len(self.original_prompts),
            "router_recommendations": [],
            "accuracy_analysis": {},
        }
        
        logger.info("🧠 Testing Organic LLM Router on Original Prompts")
        logger.info("=" * 60)
        
        correct_count = 0
        
        for prompt in self.original_prompts:
            logger.info(f"\n📝 Testing: '{prompt}'")
            
            # Get router recommendation
            router_result = self.router.route_prompt(prompt)
            logger.info(f"   🧠 Recommends: {router_result.recommended_lora}")
            logger.info(f"   💭 Reasoning: {router_result.reasoning}")
            logger.info(f"   🎯 Confidence: {router_result.confidence}")
            
            # Get actual benchmark scores
            benchmark_scores = self._get_benchmark_scores(prompt)
            
            if benchmark_scores:
                best_lora = max(benchmark_scores.items(), key=lambda x: x[1])
                router_score = benchmark_scores.get(router_result.recommended_lora, 0.0)
                is_correct = router_result.recommended_lora == best_lora[0]
                
                if is_correct:
                    correct_count += 1
                    logger.info(f"   ✅ CORRECT ({router_score:.4f})")
                else:
                    logger.info(f"   ❌ WRONG ({router_score:.4f}) - Best: {best_lora[0]} ({best_lora[1]:.4f})")
                
                results["router_recommendations"].append({
                    "prompt": prompt,
                    "router_recommendation": router_result.recommended_lora,
                    "router_score": router_score,
                    "best_lora": best_lora[0],
                    "best_score": best_lora[1],
                    "is_correct": is_correct,
                    "reasoning": router_result.reasoning,
                    "confidence": router_result.confidence
                })
        
        accuracy = (correct_count / len(self.original_prompts) * 100)
        results["accuracy_analysis"] = {
            "correct_recommendations": correct_count,
            "total_recommendations": len(self.original_prompts),
            "accuracy_percentage": accuracy,
        }
        
        logger.info(f"\n📊 ORIGINAL PROMPTS RESULTS")
        logger.info("=" * 40)
        logger.info(f"✅ Accuracy: {accuracy:.1f}% ({correct_count}/{len(self.original_prompts)})")
        
        return results
    
    def test_generalization(self) -> Dict:
        """Test generalization on completely new prompts"""
        results = {
            "total_tests": len(self.generalization_prompts),
            "router_recommendations": [],
        }
        
        logger.info("\n🌟 Testing Generalization on New Prompts")
        logger.info("=" * 60)
        
        for prompt in self.generalization_prompts:
            logger.info(f"\n📝 New prompt: '{prompt}'")
            
            # Get router recommendation
            router_result = self.router.route_prompt(prompt)
            logger.info(f"   🧠 Recommends: {router_result.recommended_lora}")
            logger.info(f"   💭 Reasoning: {router_result.reasoning}")
            logger.info(f"   🎯 Confidence: {router_result.confidence}")
            
            results["router_recommendations"].append({
                "prompt": prompt,
                "router_recommendation": router_result.recommended_lora,
                "reasoning": router_result.reasoning,
                "confidence": router_result.confidence
            })
        
        # Analyze patterns in recommendations
        lora_usage = {}
        for rec in results["router_recommendations"]:
            lora = rec["router_recommendation"]
            lora_usage[lora] = lora_usage.get(lora, 0) + 1
        
        logger.info(f"\n📊 GENERALIZATION ANALYSIS")
        logger.info("=" * 40)
        for lora, count in sorted(lora_usage.items(), key=lambda x: x[1], reverse=True):
            percentage = (count / len(self.generalization_prompts)) * 100
            logger.info(f"📊 {lora}: {count}/{len(self.generalization_prompts)} ({percentage:.1f}%)")
        
        return results
    
    def run_comprehensive_test(self) -> Dict:
        """Run both original and generalization tests"""
        logger.info("🧠 Organic LLM Router Comprehensive Test")
        logger.info("=" * 60)
        
        original_results = self.test_original_prompts()
        generalization_results = self.test_generalization()
        
        results = {
            "original_test": original_results,
            "generalization_test": generalization_results,
            "system_prompt": self.router.system_prompt
        }
        
        # Save results
        with open('organic_llm_router_results.json', 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"\n💾 Results saved to: organic_llm_router_results.json")
        
        return results

def main():
    """Main function to run organic LLM router test"""
    
    # Check if Ollama is running
    try:
        response = requests.get("http://localhost:11434/api/tags", timeout=5)
        if response.status_code != 200:
            logger.error("Ollama is not responding properly")
            return
    except requests.exceptions.RequestException:
        logger.error("Ollama is not running. Please start Ollama first:")
        logger.error("  ollama serve")
        logger.error("  ollama pull llama3.2:3b")
        return
    
    tester = OrganicRouterTester()
    results = tester.run_comprehensive_test()
    
    print("\n" + "="*60)
    print("🎉 Organic LLM Router Test Complete!")
    print("="*60)
    if results["original_test"]["accuracy_analysis"]:
        print(f"📊 Original Test Accuracy: {results['original_test']['accuracy_analysis']['accuracy_percentage']:.1f}%")
    print(f"🌟 Generalization Test: {results['generalization_test']['total_tests']} new prompts tested")

if __name__ == "__main__":
    main() 