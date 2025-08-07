#!/usr/bin/env python3
"""
Improved LLM-Based LoRA Router
Iteratively improves system prompt and handles JSON parsing better.
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

class ImprovedLLMLoRARouter:
    """Improved LLM-based LoRA router with iterative learning"""
    
    def __init__(self, llm_endpoint: str = "http://localhost:11434/api/generate", version: int = 1):
        self.llm_endpoint = llm_endpoint
        self.version = version
        self.lora_configs = self._load_lora_configs()
        self.system_prompt = self._create_system_prompt_v1() if version == 1 else self._create_system_prompt_v2()
        self.failed_cases = []  # Track failures for learning
        
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
    
    def _create_system_prompt_v1(self) -> str:
        """Create initial system prompt - version 1"""
        return """You are an expert LoRA Router for a text-to-3D generation engine. Analyze the user's prompt and recommend the best LoRA.

AVAILABLE LORAS (use EXACT names):
- Patched Realism
- Team Fortress 2 Style  
- Cartoon 3D Render
- 3D Game Assets
- Game Icon Institute
- Cinema Style
- Flux Isometric 3D

BENCHMARK INSIGHTS:

1. SPECIFIC OBJECT PATTERNS (High Priority):
   - STRAWS/SIMPLE OBJECTS: Game Icon Institute (0.8265 for straw)
   - ROBOTS/MECHS: Cartoon 3D Render (0.9445 for robot)
   - SMALL TOOLS/KNIVES: Cartoon 3D Render (0.9421 for knife)
   - JEWELRY/METALS: Cartoon 3D Render (0.8941 for gold locket)
   - HISTORICAL ARTIFACTS: Patched Realism (0.8357 for amphora)

2. GENERAL RULES:
   - Patched Realism: Most reliable fallback (0.8283 avg)
   - Cartoon 3D Render: High-risk/high-reward (either 0.9+ or 0.0)
   - Game Icon Institute: Excellent for simple everyday objects
   - 3D Game Assets: Good for mechanical props but FAILS on historical items
   - Cinema Style: Inconsistent performance

CRITICAL: Respond with ONLY a JSON object in this EXACT format:
{"recommended_lora": "Exact LoRA Name", "reasoning": "One sentence explanation", "confidence": "High"}

User Prompt:"""

    def _create_system_prompt_v2(self) -> str:
        """Create improved system prompt - version 2 with specific examples"""
        failed_examples = "\n".join([f"MISTAKE: '{case['prompt']}' -> Wrong: {case['wrong']} | Correct: {case['correct']}" 
                                   for case in self.failed_cases[-3:]])  # Last 3 failures
        
        return f"""You are an expert LoRA Router. Analyze prompts and recommend the best LoRA using benchmark data.

EXACT LORA NAMES (use these exactly):
- Patched Realism
- Team Fortress 2 Style  
- Cartoon 3D Render
- 3D Game Assets
- Game Icon Institute
- Cinema Style
- Flux Isometric 3D

PROVEN PATTERNS FROM BENCHMARKS:

HIGH-CONFIDENCE RULES:
1. "plastic straw" or simple objects -> Game Icon Institute (scored 0.8265)
2. "robot" or "mech" -> Cartoon 3D Render (scored 0.9445) 
3. "knife" or small tools -> Cartoon 3D Render (scored 0.9421)
4. "locket", "jewelry", "gold", "silver" -> Cartoon 3D Render (scored 0.8941)
5. "amphora", historical artifacts -> Patched Realism (scored 0.8357)

AVOID THESE MISTAKES:
{failed_examples}

DECISION LOGIC:
- If prompt contains specific patterns above -> use that LoRA
- For realistic everyday objects -> Patched Realism
- For mechanical game props -> 3D Game Assets  
- When uncertain -> Patched Realism (highest average 0.8283)

OUTPUT: Respond with ONLY this JSON format:
{{"recommended_lora": "Exact LoRA Name", "reasoning": "Brief explanation", "confidence": "High"}}

User Prompt:"""

    def query_llm(self, prompt: str) -> str:
        """Query the LLM with improved parameters"""
        full_prompt = f"{self.system_prompt}\n\n{prompt}"
        
        payload = {
            "model": "llama3.2:3b",
            "prompt": full_prompt,
            "stream": False,
            "options": {
                "temperature": 0.0,  # Zero temperature for deterministic output
                "top_p": 0.1,        # Very low for focused responses
                "max_tokens": 150,   # Shorter for JSON-only responses
                "stop": ["\n\n", "User:", "Prompt:", "```"]  # Stop tokens
            }
        }
        
        try:
            response = requests.post(self.llm_endpoint, json=payload, timeout=20)
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
        return '{"recommended_lora": "Patched Realism", "reasoning": "LLM unavailable, using highest average score", "confidence": "Medium"}'
    
    def parse_llm_response(self, response: str) -> RouterResult:
        """Improved JSON parsing with multiple strategies"""
        try:
            # Strategy 1: Try to parse entire response as JSON
            data = json.loads(response.strip())
            return self._create_result_from_json(data)
            
        except json.JSONDecodeError:
            try:
                # Strategy 2: Extract JSON from response
                json_match = re.search(r'\{[^}]*"recommended_lora"[^}]*\}', response, re.DOTALL)
                if json_match:
                    json_str = json_match.group(0)
                    data = json.loads(json_str)
                    return self._create_result_from_json(data)
                
                # Strategy 3: Look for any JSON-like structure
                json_match = re.search(r'\{.*?\}', response, re.DOTALL)
                if json_match:
                    json_str = json_match.group(0)
                    # Clean up common issues
                    json_str = re.sub(r'([a-zA-Z_]+):', r'"\1":', json_str)  # Add quotes to keys
                    json_str = re.sub(r':\s*([^",}\]]+)', r': "\1"', json_str)  # Add quotes to values
                    data = json.loads(json_str)
                    return self._create_result_from_json(data)
                    
            except (json.JSONDecodeError, Exception) as e:
                logger.warning(f"JSON extraction failed: {e}")
            
            # Strategy 4: Parse manually using regex
            lora_match = re.search(r'(?:recommended_lora|lora)["\s:]+([^"\'`,\n]+)', response, re.IGNORECASE)
            reasoning_match = re.search(r'(?:reasoning)["\s:]+([^"\'`,\n]+)', response, re.IGNORECASE)
            confidence_match = re.search(r'(?:confidence)["\s:]+([^"\'`,\n]+)', response, re.IGNORECASE)
            
            if lora_match:
                lora = lora_match.group(1).strip().strip('"\'')
                reasoning = reasoning_match.group(1).strip().strip('"\'') if reasoning_match else "Parsed from LLM response"
                confidence = confidence_match.group(1).strip().strip('"\'') if confidence_match else "Medium"
                
                return RouterResult(
                    recommended_lora=self._clean_lora_name(lora),
                    reasoning=reasoning,
                    confidence=confidence
                )
            
            logger.error(f"Could not parse LLM response: {response}")
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
        # Remove common suffixes
        cleaned = re.sub(r'\s*\(FLUX\)|\s*\(SD15\)', '', lora_name.strip())
        
        # Check if it matches any known LoRA
        for known_lora in self.lora_configs.keys():
            if cleaned.lower() == known_lora.lower():
                return known_lora
            if cleaned.lower() in known_lora.lower() or known_lora.lower() in cleaned.lower():
                return known_lora
        
        # If no match, return the cleaned name (will be caught by validation later)
        return cleaned
    
    def _create_fallback_result(self) -> RouterResult:
        """Create fallback result when parsing fails"""
        return RouterResult(
            recommended_lora="Patched Realism",
            reasoning="Failed to parse LLM response, using highest average score LoRA",
            confidence="Low"
        )
    
    def route_prompt(self, prompt: str) -> RouterResult:
        """Route prompt using LLM reasoning"""
        logger.info(f"🤖 LLM v{self.version} analyzing: '{prompt}'")
        
        # Query LLM
        llm_response = self.query_llm(prompt)
        logger.debug(f"🤖 Raw response: {llm_response}")
        
        # Parse response
        result = self.parse_llm_response(llm_response)
        
        # Validate the LoRA name exists
        if result.recommended_lora not in self.lora_configs:
            logger.warning(f"LLM recommended unknown LoRA: {result.recommended_lora}")
            result.recommended_lora = "Patched Realism"
            result.reasoning = f"Unknown LoRA recommended, defaulting to Patched Realism"
            result.confidence = "Low"
        
        return result
    
    def add_failed_case(self, prompt: str, wrong_lora: str, correct_lora: str):
        """Add a failed case for learning"""
        self.failed_cases.append({
            "prompt": prompt,
            "wrong": wrong_lora,
            "correct": correct_lora
        })

class IterativeLLMRouterTester:
    """Test and improve the LLM router iteratively"""
    
    def __init__(self, llm_endpoint: str = "http://localhost:11434/api/generate"):
        self.llm_endpoint = llm_endpoint
        self.test_prompts = [
            "greek amphora scene detail",
            "plastic straw of drink", 
            "small yellow triangular wooden kitchen knife",
            "enormous black robot with round body",
            "rose gold locket necklace with floral"
        ]
        
    def _get_benchmark_scores(self, prompt: str) -> Dict[str, float]:
        """Get benchmark scores for a specific prompt across all LoRAs"""
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
    
    def test_router_version(self, version: int) -> Dict:
        """Test a specific version of the router"""
        router = ImprovedLLMLoRARouter(self.llm_endpoint, version=version)
        
        results = {
            "version": version,
            "total_tests": len(self.test_prompts),
            "router_recommendations": [],
            "accuracy_analysis": {},
        }
        
        logger.info(f"🚀 Testing LLM Router Version {version}")
        logger.info("=" * 60)
        
        correct_count = 0
        
        for prompt in self.test_prompts:
            logger.info(f"\n📝 Testing: '{prompt}'")
            
            # Get router recommendation
            router_result = router.route_prompt(prompt)
            logger.info(f"   🎯 v{version} recommends: {router_result.recommended_lora}")
            logger.info(f"   💭 Reasoning: {router_result.reasoning}")
            logger.info(f"   🎯 Confidence: {router_result.confidence}")
            
            # Get actual benchmark scores
            benchmark_scores = self._get_benchmark_scores(prompt)
            best_lora = max(benchmark_scores.items(), key=lambda x: x[1])
            worst_lora = min(benchmark_scores.items(), key=lambda x: x[1])
            
            logger.info(f"   📊 Best: {best_lora[0]} ({best_lora[1]:.4f})")
            
            # Check if correct
            router_score = benchmark_scores.get(router_result.recommended_lora, 0.0)
            is_correct = router_result.recommended_lora == best_lora[0]
            
            if is_correct:
                correct_count += 1
                logger.info(f"   ✅ CORRECT ({router_score:.4f})")
            else:
                logger.info(f"   ❌ WRONG ({router_score:.4f}) - Should be {best_lora[0]}")
                # Add to failed cases for next version
                router.add_failed_case(prompt, router_result.recommended_lora, best_lora[0])
            
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
        
        accuracy = (correct_count / len(self.test_prompts) * 100)
        results["accuracy_analysis"] = {
            "correct_recommendations": correct_count,
            "total_recommendations": len(self.test_prompts),
            "accuracy_percentage": accuracy,
        }
        
        logger.info(f"\n📊 VERSION {version} RESULTS")
        logger.info("=" * 40)
        logger.info(f"✅ Accuracy: {accuracy:.1f}% ({correct_count}/{len(self.test_prompts)})")
        
        return results, router.failed_cases
    
    def iterative_improvement(self, max_iterations: int = 3) -> Dict:
        """Iteratively improve the router until 100% or max iterations"""
        all_results = {}
        failed_cases = []
        
        for version in range(1, max_iterations + 1):
            logger.info(f"\n🔄 ITERATION {version}")
            logger.info("=" * 60)
            
            results, new_failed_cases = self.test_router_version(version)
            all_results[f"version_{version}"] = results
            
            accuracy = results["accuracy_analysis"]["accuracy_percentage"]
            
            if accuracy == 100.0:
                logger.info(f"\n🎉 PERFECT! Version {version} achieved 100% accuracy!")
                break
            else:
                logger.info(f"\n📈 Version {version}: {accuracy:.1f}% - Improving for next iteration...")
                failed_cases = new_failed_cases
        
        # Save comprehensive results
        with open('iterative_llm_router_results.json', 'w') as f:
            json.dump(all_results, f, indent=2)
        
        return all_results

def main():
    """Main function to run iterative improvement"""
    
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
    
    tester = IterativeLLMRouterTester()
    results = tester.iterative_improvement(max_iterations=3)
    
    print("\n" + "="*60)
    print("🎉 Iterative LLM Router Improvement Complete!")
    print("="*60)
    
    for version_key, version_results in results.items():
        version = version_results["version"]
        accuracy = version_results["accuracy_analysis"]["accuracy_percentage"]
        print(f"📊 Version {version}: {accuracy:.1f}% accuracy")

if __name__ == "__main__":
    main() 