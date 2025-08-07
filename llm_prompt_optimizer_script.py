#!/usr/bin/env python3
"""
Automated Prompt Optimization Experiment Runner

This script automates the process of evaluating two different LLM-based prompt
optimization methods. For a given list of original prompts, it:

1.  Generates an optimized prompt using Method 1 (Strategy-Based).
2.  Generates another optimized prompt using Method 2 (Example-Based Hybrid).
3.  Runs each optimized prompt through the `subnet_accurate_validator.py` to get its score.
4.  Saves the original prompt, both optimized versions, and their full validation
    results into a single JSON file.
5.  Is designed to be pausable and resumable. If the script is stopped, it can be
    restarted and will skip any prompts already present in the results file.
6.  Finally, it prints a comparative summary of the performance of both methods.

Usage:
  python experiment_runner.py --prompts prompts_to_test.json --results experiment_results.json

python llm_prompt_optimizer_script.py --prompts zero_fidelity_prompt.json --results promptopt_v7_res.json
"""

import argparse
import json
import os
import subprocess
import sys
import re
from typing import List, Dict, Any

# Import the optimizer class from your existing script
from llm_prompt_optimizer_v7_f1 import LLMPromptOptimizer
# from llm_prompt_optimizer_v8_f1 import LLMPromptOptimizer
# from llm_prompt_optimizer_v9_f1 import LLMPromptOptimizer
# from llm_prompt_optimizer_v10_f1 import LLMPromptOptimizer
# from llm_prompt_optimizer_v11_f1 import LLMPromptOptimizer
# from llm_prompt_optimizer_v12_f1 import LLMPromptOptimizer
# from llm_prompt_optimizer_v13_f1 import LLMPromptOptimizer


class ExperimentRunner:
    """
    Manages the execution of the prompt optimization experiment.
    """

    def __init__(self, prompts_file: str, results_file: str):
        self.prompts_file = prompts_file
        self.results_file = results_file
        self.prompts_to_run = self._load_prompts()
        self.results = self._load_results()
        
        # Initialize the optimizer. You can customize the model and URL here if needed.
        self.optimizer = LLMPromptOptimizer(model="llama3.2:3b")

    def _load_prompts(self) -> List[str]:
        """Loads the list of original prompts from a JSON file."""
        try:
            with open(self.prompts_file, 'r') as f:
                data = json.load(f)
                if not isinstance(data.get('prompts'), list):
                    raise ValueError("JSON must contain a 'prompts' key with a list of strings.")
                print(f"✅ Loaded {len(data['prompts'])} prompts from '{self.prompts_file}'.")
                return data['prompts']
        except FileNotFoundError:
            print(f"❌ Error: Prompts file not found at '{self.prompts_file}'.")
            sys.exit(1)
        except (json.JSONDecodeError, ValueError) as e:
            print(f"❌ Error: Could not parse prompts file '{self.prompts_file}': {e}")
            sys.exit(1)

    def _load_results(self) -> Dict[str, Any]:
        """Loads existing results if the results file exists."""
        if os.path.exists(self.results_file):
            try:
                with open(self.results_file, 'r') as f:
                    print(f"✅ Found existing results file. Loading '{self.results_file}'.")
                    return json.load(f)
            except json.JSONDecodeError:
                print(f"⚠️ Warning: Could not parse existing results file. Starting fresh.")
                return {}
        return {}

    def _save_results(self):
        """Saves the current results to the JSON file."""
        try:
            with open(self.results_file, 'w') as f:
                json.dump(self.results, f, indent=4)
        except IOError as e:
            print(f"❌ Error: Could not write to results file '{self.results_file}': {e}")

    def _run_validator(self, original_prompt: str, optimized_prompt: str = None) -> Dict[str, Any]:
        """
        Runs the subnet_accurate_validator.py script for a given prompt.

        Args:
            original_prompt: The original prompt to compute scores against.
            optimized_prompt: The optimized prompt to use for generation (optional).

        Returns:
            A dictionary containing the parsed validation results.
        """
        if optimized_prompt and optimized_prompt != original_prompt:
            print(f"  🔍 Validating optimized prompt: '{optimized_prompt[:50]}...'")
            print(f"  🎯 Computing scores against original: '{original_prompt[:50]}...'")
            cmd = [
                "bash", "-c",
                f"source /home/mbhat/miniconda/bin/activate && conda activate trellis_new && python subnet_accurate_validator.py \"{original_prompt}\" \"{optimized_prompt}\""
            ]
        else:
            print(f"  🔍 Validating prompt: '{original_prompt[:50]}...'")
            cmd = [
                "bash", "-c",
                f"source /home/mbhat/miniconda/bin/activate && conda activate trellis_new && python subnet_accurate_validator.py \"{original_prompt}\""
            ]
            
            process = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=True, # This will raise an exception if the script returns a non-zero exit code
                timeout=300 # 5-minute timeout for safety
            )
            
            # The validator script saves its output to this file
            validator_output_file = "subnet_validation_results.json"
            if os.path.exists(validator_output_file):
                with open(validator_output_file, 'r') as f:
                    return json.load(f)
            else:
                return {"error": "subnet_validation_results.json not found after script execution."}

        except subprocess.CalledProcessError as e:
            print(f"  ❌ Validation script failed for prompt: '{prompt[:50]}...'")
            print(f"  stderr: {e.stderr}")
            return {"error": "Validator script failed to run.", "details": e.stderr}
        except subprocess.TimeoutExpired:
            print(f"  ❌ Validation script timed out for prompt: '{prompt[:50]}...'")
            return {"error": "Validator script timed out."}
        except Exception as e:
            print(f"  ❌ An unexpected error occurred during validation: {e}")
            return {"error": "An unexpected error occurred.", "details": str(e)}

    def run_experiment(self):
        """
        Executes the full experiment for all prompts.
        """
        total_prompts = len(self.prompts_to_run)
        prompts_processed = 0

        for i, original_prompt in enumerate(self.prompts_to_run):
            print("\n" + "="*70)
            print(f"🧪 Processing Prompt {i+1}/{total_prompts}: '{original_prompt}'")
            print("="*70)

            # Check if this prompt has already been processed
            if original_prompt in self.results:
                print("⏭️  Result for this prompt already exists. Skipping.")
                continue

            experiment_data = {
                "original_prompt": original_prompt,
                "method_1_strategy": {},
                "method_2_hybrid_example": {}
            }

            try:
                # --- Method 1: Strategy-Based ---
                print("\n--- Running Method 1: Strategy-Based ---")
                m1_prompt = self.optimizer.optimize_with_strategies(original_prompt)
                m1_scores = self._run_validator(original_prompt, m1_prompt)
                experiment_data["method_1_strategy"] = {
                    "optimized_prompt": m1_prompt,
                    "validation_results": m1_scores
                }
                print("--- Method 1 Complete ---")
                
                # --- Method 2: Example-Based Hybrid ---
                print("\n--- Running Method 2: Example-Based Hybrid ---")
                m2_prompt = self.optimizer.optimize_with_examples(original_prompt)
                m2_scores = self._run_validator(original_prompt, m2_prompt)
                experiment_data["method_2_hybrid_example"] = {
                    "optimized_prompt": m2_prompt,
                    "validation_results": m2_scores
                }
                print("--- Method 2 Complete ---")

                # Add the complete data for this prompt to our results
                self.results[original_prompt] = experiment_data
                prompts_processed += 1

            except Exception as e:
                print(f"❌ An unhandled error occurred while processing '{original_prompt}': {e}")
                self.results[original_prompt] = {"error": str(e)}
            
            finally:
                # Save after each prompt to ensure progress is not lost
                print("\n💾 Saving progress...")
                self._save_results()
        
        print("\n\n✅✅ Experiment finished! ✅✅")
        if prompts_processed == 0:
            print("No new prompts were processed. All results were already present.")
        else:
            print(f"Processed {prompts_processed} new prompts. Results saved to '{self.results_file}'.")
            
        self.print_summary()
        
    def print_summary(self):
        """Prints a comparative summary of the results."""
        print("\n\n" + "="*30 + " EXPERIMENT SUMMARY " + "="*30)
        
        if not self.results:
            print("No results to summarize.")
            return

        for original_prompt, data in self.results.items():
            print(f"\n--- Original: '{original_prompt}' ---")

            if "error" in data:
                print(f"  ❌ Error processing this prompt: {data['error']}")
                continue

            try:
                m1_score = data["method_1_strategy"]["validation_results"].get("validation_engine_score", "N/A")
                m2_score = data["method_2_hybrid_example"]["validation_results"].get("validation_engine_score", "N/A")
                
                m1_prompt_snip = data["method_1_strategy"]["optimized_prompt"][:60] + "..."
                m2_prompt_snip = data["method_2_hybrid_example"]["optimized_prompt"][:60] + "..."

                print(f"  📊 Method 1 (Strategies):")
                print(f"     Score: {int(m1_score):.4f}")
                print(f"     Prompt: '{m1_prompt_snip}'")

                print(f"  📊 Method 2 (Examples):")
                print(f"     Score: {int(m2_score):.4f}")
                print(f"     Prompt: '{m2_prompt_snip}'")

                # Determine the winner
                if isinstance(m1_score, float) and isinstance(m2_score, float):
                    if m1_score > m2_score:
                        print("  🏆 Winner: Method 1")
                    elif m2_score > m1_score:
                        print("  🏆 Winner: Method 2")
                    else:
                        print("  ⚖️ Result: Tie")
                else:
                    print("  ⚠️ Could not determine winner due to missing scores.")

            except (KeyError, TypeError) as e:
                print(f"  ❌ Could not parse results for this prompt. Data might be incomplete. Error: {e}")
        
        print("\n" + "="*82)

def create_prompts_file_if_not_exists(filename: str):
    """Creates a sample prompts file if one doesn't exist."""
    if not os.path.exists(filename):
        print(f"'{filename}' not found. Creating a sample file.")
        sample_prompts = {
            "prompts": [
                "a simple wooden chair",
                "sword in a stone",
                "a wizard's staff",
                "dragon guarding a treasure chest",
                "a futuristic cityscape at night",
                "an enchanted forest clearing",
                "a pirate ship on a stormy sea",
                "a robot playing chess"
            ]
        }
        with open(filename, 'w') as f:
            json.dump(sample_prompts, f, indent=4)
        print(f"Sample prompts saved to '{filename}'. You can edit this file to add your own prompts.")

def main():
    """Main function to run the experiment from the command line."""
    parser = argparse.ArgumentParser(
        description="Automated runner for prompt optimization experiments.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        "--prompts",
        type=str,
        required=True,
        help="Path to a JSON file containing a list of original prompts under the 'prompts' key."
    )
    parser.add_argument(
        "--results",
        type=str,
        required=True,
        help="Path to the JSON file where results will be saved. Can be an existing file to resume."
    )
    args = parser.parse_args()

    # Create a sample prompts file to guide the user
    create_prompts_file_if_not_exists(args.prompts)

    try:
        runner = ExperimentRunner(prompts_file=args.prompts, results_file=args.results)
        runner.run_experiment()
    except KeyboardInterrupt:
        print("\n\n⏹️  Experiment interrupted by user. Progress has been saved.")
        sys.exit(0)
    except Exception as e:
        print(f"\n\n❌ A critical error occurred: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
