#!/usr/bin/env python3
"""
LLM Close Prompt Reproducibility Test

This script implements reproducibility testing as a pre-optimization layer for the continuous orchestrator.
It finds the closest gold prompt to the original prompt and intelligently merges their components to create
a high-scoring prompt without needing full optimization.

The process is as follows:
1. Loads episodic memory to find gold standard prompts
2. For a given original prompt, finds the closest gold prompt by semantic similarity
3. Extracts components from both the original and gold prompts
4. Intelligently merges components, keeping high-scoring elements from the gold prompt
5. Reconstructs a new prompt with the merged components
6. Returns the optimized prompt for use in the continuous orchestrator

This provides a fast, deterministic way to improve prompts without full optimization.
"""
import argparse
import json
import os
import sys
import re
import subprocess
from typing import List, Dict, Any, Optional, Tuple
from difflib import SequenceMatcher

# Import episodic test prompts
from episodic_test_prompts import EPISODIC_TEST_PROMPTS

# Assuming llm_prompt_optimizer.py is in the same directory
from llm_prompt_optimizer import LLMPromptOptimizer

def extract_true_prompt(original_prompt: str) -> str:
    """Extract the true prompt from a potentially modified original prompt string."""
    # Try to find "Original Prompt: ..." at the end
    if "Original Prompt:" in original_prompt:
        return original_prompt.split("Original Prompt:")[-1].strip()
    return original_prompt

def calculate_similarity(prompt1: str, prompt2: str) -> float:
    """Calculate similarity between two prompts using sequence matching."""
    return SequenceMatcher(None, prompt1.lower(), prompt2.lower()).ratio()

class LLMClosePromptReproducibility:
    """Manages the close prompt reproducibility system."""

    def __init__(self, episodic_memory_file: str = "episodic_logs/episodic_memory.json"):
        self.episodic_memory_file = episodic_memory_file
        self.gold_standard_results = self._load_episodic_memory()
        self.optimizer = LLMPromptOptimizer()
        
        # Cache for similarity calculations
        self.similarity_cache = {}

    def _run_validator(self, original_prompt: str, optimized_prompt: str = None, endpoint: str = "generate/") -> Dict[str, Any]:
        """Runs the subnet_accurate_validator.py script to get ground truth score."""
        if optimized_prompt and optimized_prompt != original_prompt:
            print(f"  🔍 Validating optimized prompt: '{optimized_prompt[:60]}...'")
            print(f"  🎯 Computing scores against original: '{original_prompt[:60]}...'")
            cmd = [
                "bash", "-c",
                f"source /home/mbhat/miniconda/bin/activate && conda activate trellis_new && python subnet_accurate_validator.py \"{original_prompt}\" \"{optimized_prompt}\" --endpoint \"{endpoint}\""
            ]
        else:
            print(f"  🔍 Validating prompt: '{original_prompt[:60]}...'")
            cmd = [
                "bash", "-c",
                f"source /home/mbhat/miniconda/bin/activate && conda activate trellis_new && python subnet_accurate_validator.py \"{original_prompt}\" --endpoint \"{endpoint}\""
            ]
            try:
                process = subprocess.run(cmd, capture_output=True, text=True, check=True, timeout=300)
                validator_output_file = "subnet_validation_results.json"
                if os.path.exists(validator_output_file):
                    with open(validator_output_file, 'r') as f:
                        return json.load(f)
                return {"error": "subnet_validation_results.json not found."}
            except Exception as e:
                print(f"  ❌ Validation script failed for prompt: '{original_prompt[:60]}...'")
                return {"error": f"Validator script failed: {e}"}

    def _load_episodic_memory(self) -> Dict[str, Any]:
        """Loads episodic memory and extracts gold standard results."""
        if not os.path.exists(self.episodic_memory_file):
            print(f"❌ Error: Episodic memory file not found at '{self.episodic_memory_file}'.")
            return {}
        
        try:
            with open(self.episodic_memory_file, 'r') as f:
                episodic_data = json.load(f)
            
            # Extract optimization sessions
            optimization_sessions = episodic_data.get("optimization_sessions", [])
            
            # Convert to gold standard format
            gold_standard_results = {}
            
            for session in optimization_sessions:
                original_prompt = extract_true_prompt(session.get("original_prompt", ""))
                if not original_prompt:
                    continue
                
                # Find the best attempt in this session
                best_attempt = None
                best_score = 0.0
                
                for attempt in session.get("attempts", []):
                    validation_score = attempt.get("validation_score")
                    if validation_score is not None and validation_score > best_score:
                        best_score = validation_score
                        best_attempt = attempt
                
                if best_attempt:
                    # Store in format similar to experiment_results.json
                    if original_prompt not in gold_standard_results:
                        gold_standard_results[original_prompt] = {}
                    
                    # Use method_2_hybrid_example format for consistency
                    gold_standard_results[original_prompt]["method_2_hybrid_example"] = {
                        "optimized_prompt": best_attempt["optimized_prompt"],
                        "validation_results": {
                            "validation_engine_score": best_score
                        }
                    }
            
            print(f"✅ Loaded {len(gold_standard_results)} gold standard prompts from episodic memory")
            return gold_standard_results
            
        except (json.JSONDecodeError, ValueError) as e:
            print(f"❌ Error parsing episodic memory file '{self.episodic_memory_file}': {e}")
            return {}

    def _extract_clean_json(self, text: str) -> Dict[str, Any]:
        """
        More robustly extracts a JSON object from a string that might
        contain extra text or markdown.
        """
        try:
            # Find the first '{' and the last '}' to isolate the JSON object
            start_index = text.find('{')
            end_index = text.rfind('}')
            if start_index != -1 and end_index != -1:
                json_str = text[start_index:end_index+1]
                return json.loads(json_str)
            else:
                raise json.JSONDecodeError("No JSON object found.", text, 0)
        except json.JSONDecodeError as e:
            print(f"  🔧 LLM returned malformed JSON, attempting to fix... Error: {e}")
            # Try to fix common JSON formatting issues
            try:
                # Fix unquoted property names
                fixed_response = re.sub(r'(\w+):', r'"\1":', text)
                
                # Fix unquoted string values in arrays - handle multiple items
                def fix_array(match):
                    array_content = match.group(1)
                    # Split by commas and quote each item
                    items = [item.strip() for item in array_content.split(',')]
                    quoted_items = [f'"{item}"' for item in items if item]
                    return f'[{", ".join(quoted_items)}]'
                
                fixed_response = re.sub(r'\[\s*([^\]]*?)\s*\]', fix_array, fixed_response)
                
                # Fix unquoted string values for object properties
                fixed_response = re.sub(r':\s*([^",\{\}\[\]\s][^,\{\}\[\]]*?)(?=\s*[,}\]])', r': "\1"', fixed_response)
                fixed_response = re.sub(r':\s*([^",\{\}\[\]\s][^,\{\}\[\]]*?)\s*$', r': "\1"', fixed_response)
                
                # Try to extract JSON from the fixed response
                start_index = fixed_response.find('{')
                end_index = fixed_response.rfind('}')
                if start_index != -1 and end_index != -1:
                    json_str = fixed_response[start_index:end_index+1]
                    return json.loads(json_str)
                else:
                    raise json.JSONDecodeError("No JSON object found after fixing.", fixed_response, 0)
            except json.JSONDecodeError as fix_error:
                print(f"  ❌ Failed to fix JSON formatting issues: {fix_error}")
                return {"error": "Failed to extract patterns as valid JSON.", "raw_response": text[:200]}

    def extract_patterns(self, prompt_to_deconstruct: str) -> Dict[str, Any]:
        """Uses the LLM to extract patterns from a prompt."""
        system_prompt = """
You are a prompt analyst. Your task is to deconstruct the provided 3D model prompt into its core components. Identify the main subject and categorize the descriptive phrases into specific enhancement types.

Output ONLY the JSON object.

JSON Structure:
{
  "core_subject": "The main object of the prompt.",
  "enhancements": {
    "quality_adjectives": ["adjectives describing overall quality"],
    "material_details": ["phrases describing texture, materials, wear"],
    "light_interaction": ["phrases describing glows, shimmers, light"],
    "context": ["phrases describing the scene or background"]
  }
}

Example:
PROMPT: `wbgmsst, a sturdy iron pickaxe weathered to a warm, honey-brown patina, its handle worn smooth, the teeth gleaming with a subtle sheen, white background`
OUTPUT:
{
  "core_subject": "sturdy iron pickaxe",
  "enhancements": {
    "quality_adjectives": ["weathered"],
    "material_details": ["warm, honey-brown patina", "handle worn smooth"],
    "light_interaction": ["teeth gleaming with a subtle sheen"],
    "context": []
  }
}
"""
        # The _query_ollama function expects a user_prompt, so we pass the prompt there
        response_str = self.optimizer._query_ollama(system_prompt, prompt_to_deconstruct)
        
        # Use the robust JSON cleaning method
        return self._extract_clean_json(response_str)

    def find_closest_gold_prompt(self, original_prompt: str, min_similarity: float = 0.3) -> Optional[Tuple[str, float, float]]:
        """
        Find the closest gold prompt to the original prompt.
        Returns (gold_prompt, gold_score, similarity) or None if no close match found.
        """
        best_match = None
        best_similarity = 0.0
        
        for gold_original, gold_data in self.gold_standard_results.items():
            # Calculate similarity between original prompts
            similarity = calculate_similarity(original_prompt, gold_original)
            
            if similarity > best_similarity and similarity >= min_similarity:
                best_similarity = similarity
                best_run = gold_data.get("method_2_hybrid_example", {})
                if best_run and "optimized_prompt" in best_run:
                    gold_prompt = best_run["optimized_prompt"]
                    gold_score = best_run["validation_results"]["validation_engine_score"]
                    best_match = (gold_prompt, gold_score, similarity)
        
        return best_match

    def merge_components_intelligently(self, original_components: Dict[str, Any], 
                                     gold_components: Dict[str, Any], 
                                     similarity: float) -> Dict[str, Any]:
        """
        Intelligently merge components from original and gold prompts.
        Higher similarity means more components from gold prompt are preserved.
        """
        merged_components = {
            "core_subject": "",
            "enhancements": {
                "quality_adjectives": [],
                "material_details": [],
                "light_interaction": [],
                "context": []
            }
        }
        
        # Determine merge strategy based on similarity
        if similarity >= 0.8:
            # Very similar - use mostly gold components with some original elements
            gold_weight = 0.8
            original_weight = 0.2
        elif similarity >= 0.6:
            # Moderately similar - balanced merge
            gold_weight = 0.6
            original_weight = 0.4
        else:
            # Less similar - use more original components with gold enhancements
            gold_weight = 0.4
            original_weight = 0.6
        
        # Merge core subject
        if similarity >= 0.7:
            # Use gold core subject if very similar
            merged_components["core_subject"] = gold_components.get("core_subject", original_components.get("core_subject", ""))
        else:
            # Keep original core subject
            merged_components["core_subject"] = original_components.get("core_subject", "")
        
        # Merge enhancements
        for enhancement_type in ["quality_adjectives", "material_details", "light_interaction", "context"]:
            original_items = original_components.get("enhancements", {}).get(enhancement_type, [])
            gold_items = gold_components.get("enhancements", {}).get(enhancement_type, [])
            
            merged_items = []
            
            # Add gold items first (they're proven to work well)
            for item in gold_items:
                if item not in merged_items:
                    merged_items.append(item)
            
            # Add original items that don't conflict
            for item in original_items:
                if item not in merged_items:
                    # Check if this item conflicts with gold items
                    conflicts = False
                    for gold_item in gold_items:
                        if self._items_conflict(item, gold_item):
                            conflicts = True
                            break
                    
                    if not conflicts:
                        merged_items.append(item)
            
            merged_components["enhancements"][enhancement_type] = merged_items
        
        return merged_components

    def _items_conflict(self, item1: str, item2: str) -> bool:
        """Check if two enhancement items conflict with each other."""
        # Simple conflict detection - could be enhanced
        item1_lower = item1.lower()
        item2_lower = item2.lower()
        
        # Check for opposite adjectives
        opposites = [
            ("bright", "dark"), ("shiny", "matte"), ("smooth", "rough"),
            ("warm", "cool"), ("light", "heavy"), ("small", "large")
        ]
        
        for opp1, opp2 in opposites:
            if (opp1 in item1_lower and opp2 in item2_lower) or (opp2 in item1_lower and opp1 in item2_lower):
                return True
        
        return False

    def reconstruct_prompt(self, components: Dict[str, Any]) -> str:
        """Uses the LLM to reconstruct a prompt from merged components."""
        components_json = json.dumps(components, indent=2)
        system_prompt = f"""
You are a prompt assembly agent. Your task is to reconstruct a high-quality 3D model prompt from a structured set of components.

**Instructions:**
1. Start with the `core_subject`.
2. Integrate the phrases from the `enhancements` list to describe the subject.
3. Combine these components into a single, coherent, and descriptive sentence.
4. Ensure the prompt flows naturally and maintains the quality of the original components.

**Critical Constraints:**
- The final output must start with `wbgmsst,` and end with `, white background`.
- Do not invent hyper-specific details the 3D model cannot render.
- Provide only the final prompt without explanation.

**Components to use:**
{components_json}

**Reconstructed Prompt:**
"""
        # We pass an empty user prompt as the full instructions are in the system prompt
        return self.optimizer._query_ollama(system_prompt, "")

    def optimize_prompt_with_reproducibility(self, original_prompt: str, min_similarity: float = 0.3, run_validation: bool = True) -> Optional[Dict[str, Any]]:
        """
        Main function to optimize a prompt using reproducibility techniques.
        Returns optimization result or None if no suitable gold prompt found.
        """
        print(f"🔍 Finding close gold prompt for: '{original_prompt[:50]}...'")
        
        # Step 1: Find the closest gold prompt
        closest_match = self.find_closest_gold_prompt(original_prompt, min_similarity)
        if not closest_match:
            print(f"  ⚠️ No close gold prompt found (similarity threshold: {min_similarity})")
            return None
        
        gold_prompt, gold_score, similarity = closest_match
        print(f"  🏆 Found close gold prompt (similarity: {similarity:.3f}, score: {gold_score:.4f})")
        print(f"     Gold prompt: '{gold_prompt[:80]}...'")
        
        # Step 2: Extract components from original prompt
        print("  📝 Extracting components from original prompt...")
        original_components = self.extract_patterns(original_prompt)
        if "error" in original_components:
            print(f"  ❌ Failed to extract components from original prompt")
            return None
        
        # Step 3: Extract components from gold prompt
        print("  📝 Extracting components from gold prompt...")
        gold_components = self.extract_patterns(gold_prompt)
        if "error" in gold_components:
            print(f"  ❌ Failed to extract components from gold prompt")
            return None
        
        # Step 4: Intelligently merge components
        print(f"  🔄 Merging components (similarity: {similarity:.3f})...")
        merged_components = self.merge_components_intelligently(original_components, gold_components, similarity)
        
        # Step 5: Reconstruct the optimized prompt
        print("  🔧 Reconstructing optimized prompt...")
        optimized_prompt = self.reconstruct_prompt(merged_components)
        if "Error:" in optimized_prompt:
            print(f"  ❌ Failed to reconstruct prompt")
            return None
        
        print(f"  ✅ Optimization complete!")
        print(f"     Original: '{original_prompt[:80]}...'")
        print(f"     Optimized: '{optimized_prompt[:80]}...'")
        
        # Step 6: Validate the optimized prompt to get ground truth score (optional)
        if run_validation:
            print("  🎯 Running ground truth validation...")
            validation_results = self._run_validator(original_prompt, optimized_prompt)
            optimized_score = validation_results.get("validation_engine_score", 0.0)
            
            # Calculate improvement metrics
            score_improvement = optimized_score - gold_score
            improvement_percentage = ((optimized_score - gold_score) / gold_score * 100) if gold_score > 0 else 0
            
            print(f"  📊 Validation Results:")
            print(f"     Gold Score: {gold_score:.4f}")
            print(f"     Optimized Score: {optimized_score:.4f}")
            print(f"     Score Delta: {score_improvement:+.4f} ({improvement_percentage:+.1f}%)")
            
            # Determine success status
            if optimized_score >= gold_score * 0.95:
                status = "✅ SUCCESS (>= 95% of gold score)"
            elif optimized_score >= gold_score * 0.9:
                status = "⚠️ ACCEPTABLE (>= 90% of gold score)"
            else:
                status = "❌ REGRESSION (< 90% of gold score)"
            
            print(f"     Status: {status}")
        else:
            # Skip validation
            optimized_score = None
            score_improvement = None
            improvement_percentage = None
            validation_results = None
            status = "⏭️ SKIPPED (validation disabled)"
            print(f"  ⏭️ Validation skipped (use --no-validation to disable)")
        
        return {
            "original_prompt": original_prompt,
            "optimized_prompt": optimized_prompt,
            "gold_prompt": gold_prompt,
            "gold_score": gold_score,
            "optimized_score": optimized_score,
            "score_improvement": score_improvement,
            "improvement_percentage": improvement_percentage,
            "similarity": similarity,
            "original_components": original_components,
            "gold_components": gold_components,
            "merged_components": merged_components,
            "validation_results": validation_results,
            "optimization_method": "reproducibility_merge",
            "status": status
        }

def main():
    """Test the reproducibility system with sample prompts."""
    parser = argparse.ArgumentParser(
        description="Test LLM close prompt reproducibility system.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        "--episodic-memory", default="episodic_logs/episodic_memory.json",
        help="Path to episodic memory file (default: episodic_logs/episodic_memory.json)"
    )
    parser.add_argument(
        "--min-similarity", type=float, default=0.3,
        help="Minimum similarity threshold for finding close prompts (default: 0.3)"
    )
    parser.add_argument(
        "--test-prompts", nargs="+",
        help="Specific prompts to test (default: uses episodic_test_prompts)"
    )
    parser.add_argument(
        "--no-validation", action="store_true",
        help="Skip ground truth validation (faster testing)"
    )
    args = parser.parse_args()

    # Initialize the reproducibility system
    repro_system = LLMClosePromptReproducibility(args.episodic_memory)
    
    # Determine which prompts to test
    if args.test_prompts:
        test_prompts = args.test_prompts
    else:
        test_prompts = EPISODIC_TEST_PROMPTS[:5]  # Test first 5 prompts
    
    print(f"🧪 Testing reproducibility system with {len(test_prompts)} prompts...")
    print(f"   Min similarity threshold: {args.min_similarity}")
    print(f"   Gold prompts available: {len(repro_system.gold_standard_results)}")
    
    results = []
    
    for i, prompt in enumerate(test_prompts, 1):
        print(f"\n--- Test {i}/{len(test_prompts)} ---")
        
        result = repro_system.optimize_prompt_with_reproducibility(prompt, args.min_similarity, not args.no_validation)
        
        if result:
            results.append(result)
            print(f"  ✅ Success: similarity={result['similarity']:.3f}, gold_score={result['gold_score']:.4f}")
        else:
            print(f"  ❌ Failed: no suitable gold prompt found")
    
    # Summary
    print(f"\n📊 SUMMARY")
    print(f"   Total prompts tested: {len(test_prompts)}")
    print(f"   Successful optimizations: {len(results)}")
    print(f"   Success rate: {len(results)/len(test_prompts)*100:.1f}%")
    
    if results:
        avg_similarity = sum(r['similarity'] for r in results) / len(results)
        avg_gold_score = sum(r['gold_score'] for r in results) / len(results)
        
        # Only calculate validation stats if validation was run
        validated_results = [r for r in results if r['optimized_score'] is not None]
        if validated_results:
            avg_optimized_score = sum(r['optimized_score'] for r in validated_results) / len(validated_results)
            avg_improvement = sum(r['score_improvement'] for r in validated_results) / len(validated_results)
            
            # Count success categories
            successes = len([r for r in validated_results if "SUCCESS" in r['status']])
            acceptables = len([r for r in validated_results if "ACCEPTABLE" in r['status']])
            regressions = len([r for r in validated_results if "REGRESSION" in r['status']])
        else:
            avg_optimized_score = None
            avg_improvement = None
            successes = acceptables = regressions = 0
        
        print(f"   Average similarity: {avg_similarity:.3f}")
        print(f"   Average gold score: {avg_gold_score:.4f}")
        
        if validated_results:
            print(f"   Average optimized score: {avg_optimized_score:.4f}")
            print(f"   Average score improvement: {avg_improvement:+.4f}")
            print(f"   Success breakdown:")
            print(f"     ✅ SUCCESS (>= 95%): {successes}/{len(validated_results)} ({successes/len(validated_results)*100:.1f}%)")
            print(f"     ⚠️ ACCEPTABLE (>= 90%): {acceptables}/{len(validated_results)} ({acceptables/len(validated_results)*100:.1f}%)")
            print(f"     ❌ REGRESSION (< 90%): {regressions}/{len(validated_results)} ({regressions/len(validated_results)*100:.1f}%)")
            
            # Show best and worst performers
            if validated_results:
                best_result = max(validated_results, key=lambda r: r['score_improvement'])
                worst_result = min(validated_results, key=lambda r: r['score_improvement'])
                
                print(f"   Best performer:")
                print(f"     '{best_result['original_prompt'][:40]}...'")
                print(f"     Gold: {best_result['gold_score']:.4f} → Optimized: {best_result['optimized_score']:.4f} (+{best_result['score_improvement']:.4f})")
                
                print(f"   Worst performer:")
                print(f"     '{worst_result['original_prompt'][:40]}...'")
                print(f"     Gold: {worst_result['gold_score']:.4f} → Optimized: {worst_result['optimized_score']:.4f} ({worst_result['score_improvement']:+.4f})")
        else:
            print(f"   ⏭️ Validation was skipped - no score comparisons available")

if __name__ == "__main__":
    main() 