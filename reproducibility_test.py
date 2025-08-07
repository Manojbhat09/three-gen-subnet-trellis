#!/usr/bin/env python3
"""
Prompt Optimization Reproducibility Tester

This script tests the ability of an LLM to deterministically reconstruct high-scoring
prompts from a structured "recipe" of their core components.

The process is as follows:
1.  Loads a list of original prompts to test from episodic_test_prompts.py.
2.  Loads the `episodic_memory.json` to find the highest-scoring optimized
    prompt for each original prompt (the "gold standard").
3.  For each gold-standard prompt, it uses an LLM (the "Extractor") to deconstruct
    it into a structured JSON of its core subject and enhancing phrases.
4.  It then feeds this structured JSON into a second LLM call (the "Reconstructor")
    with instructions to reassemble the prompt with slight variations.
5.  It runs the newly generated prompt through the `subnet_accurate_validator.py`.
6.  Finally, it compares the new score to the original gold-standard score to
    measure how well the system reproduced a high-quality result.

This tests for both high performance and controllable, deterministic variation.
"""
import argparse
import json
import os
import subprocess
import sys
import re
from typing import List, Dict, Any

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

class ReproducibilityTester:
    """Manages the reproducibility experiment."""

    def __init__(self, results_file: str, episodic_memory_file: str = "episodic_logs/episodic_memory.json"):
        self.prompts_to_test = EPISODIC_TEST_PROMPTS
        self.episodic_memory_file = episodic_memory_file
        self.gold_standard_results = self._load_episodic_memory()
        self.reproducibility_results_file = results_file
        self.reproducibility_results = self._load_json_data(results_file, required=False)
        self.optimizer = LLMPromptOptimizer()

    def _load_episodic_memory(self) -> Dict[str, Any]:
        """Loads episodic memory and extracts gold standard results."""
        if not os.path.exists(self.episodic_memory_file):
            print(f"❌ Error: Episodic memory file not found at '{self.episodic_memory_file}'.")
            sys.exit(1)
        
        try:
            with open(self.episodic_memory_file, 'r') as f:
                episodic_data = json.load(f)
            
            # Extract optimization sessions
            optimization_sessions = episodic_data.get("optimization_sessions", [])
            
            # Convert to gold standard format similar to experiment_results.json
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
                            "validation_engine_score": best_attempt["validation_score"]
                        }
                    }
            
            return gold_standard_results
            
        except (json.JSONDecodeError, ValueError) as e:
            print(f"❌ Error parsing episodic memory file '{self.episodic_memory_file}': {e}")
            sys.exit(1)

    def _load_json_data(self, filepath: str, key: str = None, required: bool = True) -> Any:
        """Loads and validates a JSON file."""
        if not os.path.exists(filepath):
            if required:
                print(f"❌ Error: File not found at '{filepath}'.")
                sys.exit(1)
            return {} if key else {}
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)
                if key:
                    if not isinstance(data.get(key), list):
                        raise ValueError(f"JSON must contain a '{key}' key with a list.")
                    return data[key]
                return data
        except (json.JSONDecodeError, ValueError) as e:
            print(f"❌ Error parsing file '{filepath}': {e}")
            sys.exit(1)

    def _save_results(self):
        """Saves the current results to the JSON file."""
        with open(self.reproducibility_results_file, 'w') as f:
            json.dump(self.reproducibility_results, f, indent=4)

    def _run_validator(self, original_prompt: str, optimized_prompt: str = None) -> Dict[str, Any]:
        """Runs the subnet_accurate_validator.py script."""
        if optimized_prompt and optimized_prompt != original_prompt:
            print(f"  🔍 Validating optimized prompt: '{optimized_prompt[:60]}...'")
            print(f"  🎯 Computing scores against original: '{original_prompt[:60]}...'")
            cmd = [
                "bash", "-c",
                f"source /home/mbhat/miniconda/bin/activate && conda activate trellis_new && python subnet_accurate_validator.py \"{original_prompt}\" \"{optimized_prompt}\""
            ]
        else:
            print(f"  🔍 Validating prompt: '{original_prompt[:60]}...'")
            cmd = [
                "bash", "-c",
                f"source /home/mbhat/miniconda/bin/activate && conda activate trellis_new && python subnet_accurate_validator.py \"{original_prompt}\""
            ]
            process = subprocess.run(cmd, capture_output=True, text=True, check=True, timeout=300)
            validator_output_file = "subnet_validation_results.json"
            if os.path.exists(validator_output_file):
                with open(validator_output_file, 'r') as f:
                    return json.load(f)
            return {"error": "subnet_validation_results.json not found."}
        except Exception as e:
            print(f"  ❌ Validation script failed for prompt: '{prompt[:60]}...'")
            return {"error": f"Validator script failed: {e}"}

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
        """Uses the LLM to extract patterns from a high-scoring prompt."""
        print("  Extracting patterns from gold-standard prompt...")
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
        
        # Use the new robust JSON cleaning method
        return self._extract_clean_json(response_str)

    def reconstruct_prompt(self, components: Dict[str, Any]) -> str:
        """Uses the LLM to reconstruct a prompt from extracted components."""
        print("  Reconstructing new prompt with slight variations...")
        components_json = json.dumps(components, indent=2)
        system_prompt = f"""
You are a prompt assembly agent. Your task is to reconstruct a high-quality 3D model prompt from a structured set of components.

**Instructions:**
1.  Start with the `core_subject`.
2.  Integrate the phrases from the `enhancements` list to describe the subject.
3.  Combine these components into a single, coherent, and descriptive sentence.
4.  **Introduce slight, safe variations.** You can change an adjective (e.g., 'gleaming' to 'shimmering'), add a similar one, or slightly rephrase a clause, but do not change the core meaning or introduce new, complex concepts.

**Critical Constraints:**
-   The final output must start with `wbgmsst,` and end with `, white background`.
-   Do not invent hyper-specific details the 3D model cannot render.
-   Provide only the final prompt without explanation.

**Components to use:**
{components_json}

**Reconstructed Prompt:**
"""
        # We pass an empty user prompt as the full instructions are in the system prompt
        return self.optimizer._query_ollama(system_prompt, "")

    def run_test(self):
        """Runs the full reproducibility test."""
        for original_prompt in self.prompts_to_test:
            if original_prompt in self.reproducibility_results:
                print(f"\n⏭️  Skipping '{original_prompt}' as it's already in the results file.")
                continue

            print(f"\n--- Testing Reproducibility for: '{original_prompt}' ---")

            # 1. Find the best prompt from previous experiments
            if original_prompt not in self.gold_standard_results:
                print(f"  ⚠️ No previous result found for '{original_prompt}'. Skipping.")
                continue
            
            gold_data = self.gold_standard_results[original_prompt]
            # Prioritize method 2, fall back to method 1
            best_run = gold_data.get("method_2_hybrid_example", gold_data.get("method_1_strategy", {}))
            
            if not best_run or "error" in best_run.get("validation_results", {}):
                 print(f"  ⚠️ No valid gold-standard prompt found for '{original_prompt}'. Skipping.")
                 continue

            gold_prompt = best_run["optimized_prompt"]
            gold_score = best_run["validation_results"]["validation_engine_score"]
            print(f"  🏆 Gold Standard Score: {gold_score:.4f}")
            print(f"     Prompt: '{gold_prompt[:80]}...'")

            # 2. Extract patterns from the best prompt
            extracted_components = self.extract_patterns(gold_prompt)
            if "error" in extracted_components:
                self.reproducibility_results[original_prompt] = {"error": "Extraction failed"}
                self._save_results()
                continue
            
            # 3. Reconstruct a new prompt from the patterns
            reconstructed_prompt = self.reconstruct_prompt(extracted_components)
            if "Error:" in reconstructed_prompt:
                 self.reproducibility_results[original_prompt] = {"error": "Reconstruction failed"}
                 self._save_results()
                 continue

            # 4. Validate the new prompt
            new_validation_results = self._run_validator(original_prompt, reconstructed_prompt)
            new_score = new_validation_results.get("validation_engine_score", 0.0)
            
            # 5. Store and save results
            self.reproducibility_results[original_prompt] = {
                "gold_standard_prompt": gold_prompt,
                "gold_standard_score": gold_score,
                "extracted_components": extracted_components,
                "reconstructed_prompt_with_variation": reconstructed_prompt,
                "reconstructed_score": new_score,
                "validation_results": new_validation_results,
                "score_delta": new_score - gold_score
            }
            self._save_results()
            print(f"  ✨ New Score: {new_score:.4f} (Delta: {new_score - gold_score:+.4f})")

        print("\n\n✅✅ Reproducibility test finished! ✅✅")
        self.print_summary()

    def print_summary(self):
        """Prints a final summary of the test."""
        print("\n" + "="*30 + " REPRODUCIBILITY SUMMARY " + "="*30)
        success_count = 0
        total_count = 0
        deltas = []

        for original, data in self.reproducibility_results.items():
            if "error" in data or not isinstance(data.get("reconstructed_score"), float):
                print(f"\n--- Original: '{original}' ---")
                print(f"  ❌ FAILED: {data.get('error', 'Unknown Error')}")
                continue
            
            total_count += 1
            gold_score = data['gold_standard_score']
            new_score = data['reconstructed_score']
            delta = data['score_delta']
            deltas.append(delta)
            
            # Define success as being within 5% of the original high score
            is_success = new_score >= (gold_score * 0.95)
            if is_success:
                success_count += 1
            
            print(f"\n--- Original: '{original}' ---")
            print(f"  🏆 Gold Standard: {gold_score:.4f}")
            print(f"  ✨ Reconstructed: {new_score:.4f} (Delta: {delta:+.4f})")
            print(f"  Status: {'✅ SUCCESS' if is_success else '⚠️ REGRESSION'}")

        if total_count > 0:
            avg_delta = sum(deltas) / len(deltas)
            success_rate = (success_count / total_count) * 100
            print("\n--- Overall Stats ---")
            print(f"  Success Rate (score >= 95% of original): {success_rate:.2f}% ({success_count}/{total_count})")
            print(f"  Average Score Delta: {avg_delta:+.4f}")
            print("="*85)

def main():
    parser = argparse.ArgumentParser(
        description="Run prompt reproducibility experiments using episodic test prompts and memory.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        "--results", required=True,
        help="Path to JSON file to save reproducibility results."
    )
    parser.add_argument(
        "--episodic-memory", default="episodic_logs/episodic_memory.json",
        help="Path to episodic memory file (default: episodic_logs/episodic_memory.json)"
    )
    args = parser.parse_args()

    try:
        tester = ReproducibilityTester(
            results_file=args.results,
            episodic_memory_file=args.episodic_memory
        )
        tester.run_test()
    except KeyboardInterrupt:
        print("\n\n⏹️  Experiment interrupted by user. Progress has been saved.")
        sys.exit(0)

if __name__ == "__main__":
    main()

'''
<c-memory episodic_logs/episodic_memory.json --results repro_res1.json
✅ Initialized Optimizer with model: llama3.2:3b at http://localhost:11434
✅ Ollama server connection successful.

--- Testing Reproducibility for: 'small pink robot with round eyes and metal arms' ---
  🏆 Gold Standard Score: 0.8968
     Prompt: 'wbgmsst, : wbgmsst, small robot with intricate metallic details on its limbs, gl...'
  Extracting patterns from gold-standard prompt...

⏳ Querying the LLM... (This may take a moment)
  🔧 LLM returned malformed JSON, attempting to fix... Error: Expecting property name enclosed in double quotes: line 2 column 3 (char 4)
  Reconstructing new prompt with slight variations...

⏳ Querying the LLM... (This may take a moment)
  🔍 Validating prompt: 'wbgmsst, a small robot with intricate glowing metallic detai...'
  ✨ New Score: 0.8814 (Delta: -0.0154)

--- Testing Reproducibility for: 'aquamarine hairpin with spiral form' ---
  🏆 Gold Standard Score: 0.8964
     Prompt: 'wbgmsst, aquamarine hairpin with twisted, coiled, winding spiral design on white...'
  Extracting patterns from gold-standard prompt...

⏳ Querying the LLM... (This may take a moment)
  🔧 LLM returned malformed JSON, attempting to fix... Error: Expecting property name enclosed in double quotes: line 2 column 3 (char 4)
  Reconstructing new prompt with slight variations...

⏳ Querying the LLM... (This may take a moment)
  🔍 Validating prompt: 'wbgmsst A shimmering aquamarine hairpin with a twisted, coil...'
  ✨ New Score: 0.8676 (Delta: -0.0288)

--- Testing Reproducibility for: 'ivory maiden with lute' ---
  🏆 Gold Standard Score: 0.8511
     Prompt: 'wbgmsst, ancient Greek maiden with lute on a white background...'
  Extracting patterns from gold-standard prompt...

⏳ Querying the LLM... (This may take a moment)
  🔧 LLM returned malformed JSON, attempting to fix... Error: Expecting property name enclosed in double quotes: line 2 column 3 (char 4)
  Reconstructing new prompt with slight variations...

⏳ Querying the LLM... (This may take a moment)
  🔍 Validating prompt: 'wbgmsst, a shimmering ancient Greek maiden holds a gleaming ...'
  ✨ New Score: 0.8291 (Delta: -0.0220)

--- Testing Reproducibility for: 'white van cargo boxes subarban' ---
  🏆 Gold Standard Score: 0.8631
     Prompt: 'wbgmsst, urban or suburban with a white background...'
  Extracting patterns from gold-standard prompt...

⏳ Querying the LLM... (This may take a moment)
  🔧 LLM returned malformed JSON, attempting to fix... Error: Expecting property name enclosed in double quotes: line 2 column 3 (char 4)
  Reconstructing new prompt with slight variations...

⏳ Querying the LLM... (This may take a moment)
  🔍 Validating prompt: 'wbgmsst, a modern suburban home with gleaming white siding a...'
^C
'''