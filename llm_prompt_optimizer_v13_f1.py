#!/usr/bin/env python3
"""
LLM Prompt Optimizer - Inference Script v2.0

This script uses a local LLM (via Ollama) to optimize a given prompt based on
pre-learned patterns, with a specific goal of achieving a validation score > 0.90.
It offers two distinct methods for constructing the system prompt:

Method 1 (Strategy-Based): Explicitly lists the principles of high-scoring prompts
                          and instructs the LLM to apply them to achieve a score > 0.90.

Method 2 (Example-Based / Few-Shot): Provides the LLM with a comprehensive list of
                                     original prompts and their corresponding top-scoring
                                     optimized versions, asking it to infer and apply
                                     the patterns to hit the > 0.90 target.

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
    Optimizes prompts using a local LLM with a focus on generating high-scoring outputs.
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
            requests.get(self.ollama_url, timeout=5)
            print("✅ Ollama server connection successful.")
            
            response = requests.get(f"{self.ollama_url}/api/tags", timeout=5)
            response.raise_for_status()
            models = response.json().get('models', [])
            if not any(m['name'].startswith(self.model) for m in models):
                print(f"⚠️ WARNING: Model '{self.model}' not found in Ollama. Please ensure it is pulled via `ollama pull {self.model}`.")
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
        # For Ollama, it's often better to combine system and user prompts into a single prompt field.
        full_prompt = f"{system_prompt}\n\nORIGINAL PROMPT TO TRANSFORM:\n`{user_prompt}`\n\nOPTIMIZED PROMPT:"

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
                        "num_predict": 256,  # Limit output length, 
                        "seed": 42
                    }
                },
                timeout=120
            )
            response.raise_for_status()
            response_text = response.json().get("response", "").strip()
            
            cleaned_prompt = self._clean_response(response_text)
            return cleaned_prompt

        except requests.exceptions.RequestException as e:
            print(f"❌ Error during Ollama API call: {e}")
            return f"Error: Could not generate prompt. {e}"
            
    def _clean_response(self, response_text: str) -> str:
        """
        Cleans the LLM's raw output to extract only the prompt.
        """
        # Greedily find the text between the required markers.
        match = re.search(r'(wbgmsst,.*, white background)', response_text, re.DOTALL)
        if match:
            return match.group(1).strip().replace('\n', ' ')

        # Fallback if the markers are not perfectly matched
        if "wbgmsst" in response_text:
            response_text = response_text[response_text.find("wbgmsst"):]
        
        response_text = response_text.split("\n\n")[0]
        response_text = re.sub(r"```[\w\s]*", "", response_text)
        return response_text.strip().replace('"', '')

    def optimize_with_strategies(self, original_prompt: str) -> str:
        """
        Method 1: Guides the LLM by explicitly listing high-scoring principles.
        """
        print("\n🚀 Using Method 1: Strategy-Based Prompting (Target Score > 0.90)")
        system_prompt = """You are an elite prompt engineer for a 3D generative model. Your task is to rewrite a simple user prompt into a master-level, high-scoring prompt that will achieve a validation score **above 0.90**.

**Principles for High-Scoring Prompts (>0.90):**

1.  **From Object to Scene:** Do not just describe the object. Create a complete, atmospheric scene around it. The environment is critical.
    * *Example:* `layered lemonade` -> `...suspended in mid-air over a serene lake...`

2.  **Intensify with Sensory Adjectives:** Use powerful, evocative, and sensory language. Aim for a feeling, not just a description.
    * *Example:* `cupcake with chocolate icing` -> `...a decadently indulgent cupcake with an explosively rich and intensely chocolatey icing...`

3.  **Master Light and Shadow:** This is the most important principle. Describe how light interacts with the object. Use terms like "ethereal glow," "catches the light," "subtle glow," or "interplay between light and shadow."
    * *Example:* `blue creature` -> `...surrounded by a soft, ethereal glow.`

4.  **Specify Materials and Craftsmanship:** Add details about the object's materials and imply it is well-made. Use words like "intricate," "delicate," "refined," "premium," "slender," "angular."
    * *Example:* `candle holder` -> `...a refined and premium modern candle holder...`

**Strict Formatting Rules:**
-   Your final output **MUST** start with the exact text: `wbgmsst,`
-   Your final output **MUST** end with the exact text: `, white background`
-   **DO NOT** provide any explanation, preamble, or additional text. Your entire response should be only the final, optimized prompt.
"""
        print("\n--- System Prompt (Method 1) ---")
        print("NOTE: Instructing the LLM with explicit, high-scoring principles.")
        print("---------------------------------")
        return self._query_ollama(system_prompt, original_prompt)

    def optimize_with_examples(self, original_prompt: str) -> str:
        """
        Method 2: Guides the LLM by providing few-shot examples of top-tier transformations.
        """
        print("\n🚀 Using Method 2: Example-Based Prompting (Target Score > 0.90)")
#         system_prompt = """You are a master prompt engineer for a 3D generative model. Your objective is to transform a simple `ORIGINAL PROMPT` into a highly descriptive and atmospheric prompt that will score above 0.90.

# **Core Philosophy:** Transform the ordinary into the extraordinary. Do not just describe an object; create a complete, elegant, and non-literal scene that evokes a feeling.

# **Key Principles for >0.90 Scores:**
# 1.  **Balance Object & Scene:** The final prompt must describe both the object and its environment. Mundane objects (like a tool or bottle) require a more imaginative scene, while inherently beautiful objects (like jewelry) can have a simpler context.
# 2.  **Evocative Adjectives:** Use luxurious, sensory, and powerful adjectives (e.g., `decadent`, `ethereal`, `majestic`, `weathered`, `velvety`).
# 3.  **Mastery of Light:** This is crucial. Describe how light interacts with the scene. Use phrases like `soft, ethereal glow`, `catches the light`, `shimmering`, `subtle sheen`, `interplay between light and shadow`.
# 4.  **Conciseness and Impact:** Be descriptive but avoid excessive length. Every word should add value. Overly long prompts can fail. Aim for 2-4 rich clauses.

# **Prime Example of Transformation:**
# * **Original:** `sturdy iron pickaxe worn from use`
# * **High-Scoring Optimized (Score: 0.909):** `wbgmsst, a sturdy iron pickaxe suspended over a serene lake with lotus flowers and gentle ripples on the surface, catches the light, surrounded by a soft, ethereal glow, in a refined, weathered state, white background`

# **Your Task:**
# Apply these principles to the `ORIGINAL PROMPT`. Analyze if the object is mundane or inherently beautiful to decide on the complexity of the scene.

# **Strict Output Format:**
# -   Must start with `wbgmsst,`
# -   Must end with `, white background`
# -   Provide **only** the optimized prompt. No explanations.
# """
        system_prompt ="""
**Role:** You are a prompt optimization agent in a Reinforcement Learning loop for a 3D generative model.
**Objective:** Rewrite the user's `ORIGINAL PROMPT` to maximize the `Validation Score`.

**Analysis of Historical Data:**
High-scoring prompts consistently demonstrate the following patterns:
-   **Adjective Enhancement:** Simple nouns are upgraded with evocative, high-quality adjectives (`luxurious`, `breathtakingly`, `decadent`).
-   **Material & Light Specificity:** Tangible textures (`velvety`, `frosted`) and light interactions (`catches the light`, `ethereal glow`, `shimmer`) are specified.
-   **Contextual Framing:** The object is placed within a simple, elegant scene (`suspended over a serene lake`, `surrounded by...`).
-   **Craftsmanship Implication:** Words implying skilled creation are used (`refined`, `intricate`, `delicate`).

### Case Studies: Your Thought Process
Study these examples to learn the transformation process.

---
**Case Study 1**

**Original Prompt:** `sturdy iron pickaxe worn from use`
**Thought Process:** "This is a mundane tool. The key is **Conceptual Elevation**. Putting it in a mine is boring. Placing it in a serene, beautiful context like a lake creates a powerful, artistic contrast. The juxtaposition of a rough tool with a beautiful scene is where the high score lies. I will add details about light (`ethereal glow`) and craftsmanship (`refined, weathered state`) to complete the vision."
**High-Scoring Optimized Prompt:** `wbgmsst, a sturdy iron pickaxe suspended over a serene lake with lotus flowers and gentle ripples on the surface, catches the light, surrounded by a soft, ethereal glow, in a refined, weathered state, white background`

---
**Case Study 2**

**Original Prompt:** `iridescent opal revealing spectrum of colors`
**Thought Process:** "This object is already beautiful. I don't need a complex scene to elevate it. My focus should be on **Sensory Immersion**. I will intensify the description of the opal itself, using powerful adjectives like `breathtakingly` and describing its light interaction with `kaleidoscope of shimmering hues`. The context can be simple—a `delicate, curved framework` is enough to imply craftsmanship."
**High-Scoring Optimized Prompt:** `wbgmsst, a breathtakingly iridescent opal nestled within its delicate, curved framework, slowly reveals a kaleidoscope of shimmering hues as light dances across its surface, casting a mesmerizing spectrum of colors against the surrounding white background.`

---
**Case Study 3**

**Original Prompt:** `modern plastic bottle with blue cap`
**Thought Process:** "Another mundane object. Again, **Conceptual Elevation** is the primary goal. A kitchen counter is boring. A futuristic or abstract scene is better. 'Suspended in mid-air over a minimalist cityscape at dusk' is imaginative and visually striking. I will add details about light (`neon lights`) and texture (`textured concrete`) to ground the fantastical scene."
**High-Scoring Optimized Prompt:** `wbgmsst, a sleek modern plastic bottle with a sturdy blue cap suspended in mid-air over a minimalist cityscape at dusk, surrounded by tall skyscrapers and neon lights, on a textured concrete ground, white background`


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
        print("\n--- System Prompt (Method 2) ---")
        print("NOTE: Providing the LLM with multiple high-quality examples to learn from.")
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
             "1: Strategy-Based (explicitly list principles)\n"
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