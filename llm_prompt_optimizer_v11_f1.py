#!/usr/bin/env python3
"""
LLM Prompt Optimizer - Inference Script

This script uses a local LLM (via Ollama) to optimize a given prompt based on
pre-learned patterns. It offers two distinct methods for constructing the
system prompt to guide the LLM:

Method 1 (Strategy-Based): Explicitly lists the successful optimization
           strategies and asks the LLM to apply them.

Method 2 (Example-Based / Few-Shot): Provides the LLM with a list of
           original prompts and their corresponding top 3 winning optimized
           versions, asking it to infer and apply the patterns.

Usage:
  python llm_prompt_optimizer.py "your original prompt" --method 1
  python llm_prompt_optimizer.py "a wizard's staff" --method 2
"""

import argparse
import requests
import json
import sys
import re

class LLMPromptOptimizer:
    """
    Optimizes prompts using a local LLM based on learned patterns.
    """

    def __init__(self, ollama_url: str = "http://localhost:11434", model: str = "llama3.2:3b"):
        """
        Initializes the optimizer.

        Args:
            ollama_url: The URL for the Ollama API server.
            model: The name of the Ollama model to use.
        """
        self.ollama_url = ollama_url
        self.model = model
        print(f"✅ Initialized Optimizer with model: {self.model} at {self.ollama_url}")
        self._check_ollama_connection()

    def _check_ollama_connection(self):
        """Checks if the Ollama server is running and the model is available."""
        try:
            response = requests.get(self.ollama_url)
            response.raise_for_status()
            print("✅ Ollama server connection successful.")
            
            response = requests.get(f"{self.ollama_url}/api/tags")
            response.raise_for_status()
            models = response.json().get('models', [])
            if not any(m['name'].startswith(self.model) for m in models):
                print(f"⚠️ WARNING: Model '{self.model}' not found in Ollama. Please ensure it is pulled.")
                sys.exit(1)

        except requests.exceptions.RequestException as e:
            print(f"❌ Error connecting to Ollama at {self.ollama_url}: {e}")
            print("Please ensure the Ollama application is running and accessible.")
            sys.exit(1)

    def _query_ollama(self, system_prompt: str, user_prompt: str) -> str:
        """
        Sends a query to the Ollama API and gets the response.

        Args:
            system_prompt: The system-level instructions for the model.
            user_prompt: The user's original prompt to be optimized.

        Returns:
            The optimized prompt string from the model's response.
        """
        print("\n⏳ Querying the LLM... (This may take a moment)")
        full_prompt = f"{system_prompt}\n\nORIGINAL PROMPT:\n`{user_prompt}`\n\nOPTIMIZED PROMPT:"

        try:
            response = requests.post(
                f"{self.ollama_url}/api/generate",
                json={
                    "model": self.model,
                    "prompt": full_prompt,
                    "stream": False,
                    "options": {
                        "temperature": 0.7,
                        "top_p": 0.9,
                        "num_predict": 256 # Limit output length
                    }
                },
                timeout=120
            )
            response.raise_for_status()
            response_text = response.json().get("response", "").strip()
            
            # Clean the output to be just the prompt
            # The model might add extra conversational text or formatting
            cleaned_prompt = self._clean_response(response_text)

            return cleaned_prompt

        except requests.exceptions.RequestException as e:
            print(f"❌ Error during Ollama API call: {e}")
            return f"Error: Could not generate prompt. {e}"
            
    def _clean_response(self, response_text: str) -> str:
        """
        Cleans the LLM's raw output to extract only the prompt.
        """
        # Remove any leading text like "Here is the optimized prompt:"
        if "wbgmsst" in response_text:
            response_text = response_text[response_text.find("wbgmsst"):]

        # Remove explanations or text that might appear after the prompt
        response_text = response_text.split("\n\n")[0]
        
        # Remove potential markdown code blocks
        response_text = re.sub(r"```[\w\s]*", "", response_text)

        # Final cleanup
        return response_text.strip().replace('"', '')

    def optimize_with_strategies(self, original_prompt: str) -> str:
        """
        Method 1: Guides the LLM by explicitly listing optimization strategies.

        Args:
            original_prompt: The user's prompt.

        Returns:
            The optimized prompt.
        """
        print("\n🚀 Using Method 1: Strategy-Based Prompting")
        system_prompt = """
**Role:** You are a prompt optimization agent in a Reinforcement Learning loop for a 3D generative model.
**Objective:** Rewrite the user's `ORIGINAL PROMPT` to maximize the `Validation Score`.

**Analysis of Historical Data:**
High-scoring prompts consistently demonstrate the following patterns:
-   **Adjective Enhancement:** Simple nouns are upgraded with evocative, high-quality adjectives (`luxurious`, `breathtakingly`, `decadent`).
-   **Material & Light Specificity:** Tangible textures (`velvety`, `frosted`) and light interactions (`catches the light`, `ethereal glow`, `shimmer`) are specified.
-   **Contextual Framing:** The object is placed within a simple, elegant scene (`suspended over a serene lake`, `surrounded by...`).
-   **Craftsmanship Implication:** Words implying skilled creation are used (`refined`, `intricate`, `delicate`).

**Prime Example (Input -> High-Scoring Output):**
* **ORIGINAL:** `tall glass of layered lemonade`
* **OPTIMIZED (Score: 0.9443):** `wbgmsst, a slender glass of layered lemonade suspended in mid-air over a serene lake with lotus flowers and gentle ripples on the surface, white background`

**Constraint Checklist:**
1.  **Apply General Patterns:** Use the patterns above to enhance the prompt.
2.  **Avoid Over-Specificity:** Do not invent details the 3D model cannot reasonably interpret (e.g., "hand-drawn patterns," "made in the 18th century"). Stick to visual, tangible qualities.
3.  **Balance Detail and Conciseness:** The prompt should be descriptive but not excessively long.
4.  **Strict Output Format:** The final output must start with `wbgmsst,` and end with `, white background`.
5.  **No Explanation:** Do not provide any text other than the final optimized prompt.
Process the following `ORIGINAL PROMPT` according to these instructions.
"""
        print("\n--- System Prompt (Method 1) ---")
        print(system_prompt)
        print("---------------------------------")
        return self._query_ollama(system_prompt, original_prompt)

    def optimize_with_examples(self, original_prompt: str) -> str:
        """
        Method 2: Guides the LLM by providing few-shot examples.

        Args:
            original_prompt: The user's prompt.

        Returns:
            The optimized prompt.
        """
        print("\n🚀 Using Method 2: Example-Based (Few-Shot) Prompting")
        system_prompt = """
**Role:** You are an expert prompt engineer for a 3D model generator.
**Objective:** Rewrite the user's `ORIGINAL PROMPT` to maximize its `Validation Score` by transforming it into a descriptive, evocative, and thematically resonant masterpiece.

**--- The 4-Step Optimization Process ---**

**Step 1: Analyze the Object's Essence.**
First, identify the core identity of the object. Is it an object of inherent beauty (jewelry, gems), a functional tool (pickaxe, rifle), an everyday item (furniture, bottle), or something fantastical (creature, helmet)?

**Step 2: Choose a Thematically Resonant Scene.**
Based on your analysis, establish a setting that enhances the object's story.
* **For Mundane Objects:** Elevate them with a simple but imaginative context. A `pickaxe` might be in a `misty mountain terrain` or `nestled in a goblin's cave`. A `baseball bat` could be `resting on a vintage wooden bench`.
* **For Beautiful Objects:** Use a minimal, enhancing backdrop. A `gemstone` doesn't need a complex scene; its context can be `a soft, ethereal glow` or `a rich, velvety background` that makes the object itself the hero.
* **Avoid Generic Scenes:** Do not default to putting every object over a "serene lake." The context must fit the object.

**Step 3: Layer the Sensory Details.**
Enrich the prompt by describing its tangible qualities.
* **Adjectives:** Use evocative, high-quality adjectives (`luxurious`, `weathered`, `breathtakingly`).
* **Materials & Texture:** Specify textures (`velvety`, `frosted`, `worn leather`, `rich wood grain`).
* **Light Interaction:** Describe how it catches light (`shimmering`, `ethereal glow`, `catches the light`).
* **Craftsmanship:** Imply skilled creation (`refined`, `intricate`, `delicate`).

**Step 4: Refine for Brevity and Impact.**
Review your prompt. It should be dense with powerful keywords but not overly long or conversational. It should read like a high-end product description, not a story.

**--- Critical Constraints ---**

* **Avoid Unrenderable Concepts:** Do not invent details the 3D model cannot interpret (e.g., "hand-drawn patterns," body parts like "wrists," or complex scenes like "a vintage store"). Stick to visual, tangible qualities.
* **Strict Output Format:** The final output must start with `wbgmsst,` and end with `, white background`.
* **No Explanations:** Provide only the optimized prompt.

**Prime Example:**
* **ORIGINAL:** `tall glass of layered lemonade`
* **OPTIMIZED (Score: 0.9443):** `wbgmsst, a slender glass of layered lemonade suspended in mid-air over a serene lake with lotus flowers and gentle ripples on the surface, white background`

Process the following `ORIGINAL PROMPT` according to these instructions.
"""
        print("\n--- System Prompt (Method 2) ---")
        print("NOTE: The example-based prompt is very long and is not fully displayed here.")
        print("---------------------------------")
        return self._query_ollama(system_prompt, original_prompt)


def main():
    """Main function to run the optimizer from the command line."""
    parser = argparse.ArgumentParser(
        description="Optimize a prompt for a 3D model generator using a local LLM.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        "prompt",
        type=str,
        help="The original prompt to optimize."
    )
    parser.add_argument(
        "--method",
        type=int,
        choices=[1, 2],
        required=True,
        help="The optimization method to use:\n"
             "1: Strategy-Based (explicitly list patterns)\n"
             "2: Example-Based (provide few-shot examples)"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="llama3.2:3b",
        help="The Ollama model to use (default: llama3.2:3b)"
    )
    parser.add_argument(
        "--url",
        type=str,
        default="http://localhost:11434",
        help="The URL of the Ollama server (default: http://localhost:11434)"
    )

    args = parser.parse_args()

    optimizer = LLMPromptOptimizer(ollama_url=args.url, model=args.model)

    if args.method == 1:
        optimized_prompt = optimizer.optimize_with_strategies(args.prompt)
    else:
        optimized_prompt = optimizer.optimize_with_examples(args.prompt)

    print("\n" + "="*20 + " OPTIMIZATION COMPLETE " + "="*20)
    print(f"Original Prompt:  {args.prompt}")
    print(f"Optimized Prompt: {optimized_prompt}")
    print("="*63)


if __name__ == "__main__":
    main()
