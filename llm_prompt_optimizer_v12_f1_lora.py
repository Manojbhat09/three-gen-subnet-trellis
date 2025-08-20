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

    def __init__(self, ollama_url: str = "http://localhost:11434", model: str = "llama3.2:3b", 
                 use_vllm: bool = False, vllm_url: str = "http://localhost:9000", vllm_model: str = "llama-3-2-3b-it"):
        """
        Initializes the optimizer.

        Args:
            ollama_url: The URL for the Ollama API server.
            model: The name of the Ollama model to use.
            use_vllm: Whether to use vLLM instead of Ollama.
            vllm_url: The URL for the vLLM server.
            vllm_model: The name of the vLLM model to use.
        """
        self.ollama_url = ollama_url
        self.model = model
        self.use_vllm = use_vllm
        self.vllm_url = vllm_url
        self.vllm_model = vllm_model
        
        # Print LLM provider configuration prominently
        print("\n" + "="*60)
        print("🤖 LLM PROMPT OPTIMIZER - PROVIDER CONFIGURATION")
        print("="*60)
        if self.use_vllm:
            print(f"✅ Using vLLM: {self.vllm_url}")
            print(f"   Model: {self.vllm_model}")
            print(f"   Status: ACTIVE")
        else:
            print(f"✅ Using Ollama: {self.ollama_url}")
            print(f"   Model: {self.model}")
            print(f"   Status: ACTIVE")
        print("="*60)
        
        if self.use_vllm:
            print(f"✅ Initialized Optimizer with vLLM model: {self.vllm_model} at {self.vllm_url}")
            self._check_vllm_connection()
        else:
            print(f"✅ Initialized Optimizer with Ollama model: {self.model} at {self.ollama_url}")
            self._check_ollama_connection()

    def _check_vllm_connection(self):
        """Checks if the vLLM server is running and accessible."""
        try:
            response = requests.get(f"{self.vllm_url}/health")
            response.raise_for_status()
            print("✅ vLLM server connection successful.")
            
        except requests.exceptions.RequestException as e:
            print(f"❌ Error connecting to vLLM at {self.vllm_url}: {e}")
            print("Please ensure the vLLM server is running and accessible.")
            sys.exit(1)

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

    def _query_vllm(self, system_prompt: str, user_prompt: str) -> str:
        """
        Sends a query to the vLLM API and gets the response.

        Args:
            system_prompt: The system-level instructions for the model.
            user_prompt: The user's original prompt to be optimized.

        Returns:
            The optimized prompt string from the model's response.
        """
        print("\n⏳ Querying the vLLM... (This may take a moment)")
        print(f"   🤖 [vLLM] Server: {self.vllm_url}")
        print(f"   🤖 [vLLM] Model: {self.vllm_model}")
        full_prompt = f"{system_prompt}\n\nORIGINAL PROMPT:\n`{user_prompt}`\n\nOPTIMIZED PROMPT:"

        try:
            print(f"   📤 [vLLM] Sending request to {self.vllm_url}...")
            response = requests.post(
                f"{self.vllm_url}/v1/chat/completions",
                json={
                    "model": self.vllm_model,
                    "messages": [{"role": "user", "content": full_prompt}],
                    "stream": False,
                    "temperature": 0.7,
                    "max_tokens": 256,  # Limit output length
                    "top_p": 0.9
                },
                timeout=120
            )
            response.raise_for_status()
            response_text = response.json().get("choices", [{}])[0].get("message", {}).get("content", "").strip()
            print(f"   📥 [vLLM] Response received: {len(response_text)} characters")
            
            # Clean the output to be just the prompt
            # The model might add extra conversational text or formatting
            cleaned_prompt = self._clean_response(response_text)

            return cleaned_prompt

        except requests.exceptions.RequestException as e:
            print(f"❌ Error during vLLM API call: {e}")
            return f"Error: Could not generate prompt. {e}"

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
        print(f"   🤖 [Ollama] Server: {self.ollama_url}")
        print(f"   🤖 [Ollama] Model: {self.model}")
        full_prompt = f"{system_prompt}\n\nORIGINAL PROMPT:\n`{user_prompt}`\n\nOPTIMIZED PROMPT:"

        try:
            print(f"   📤 [Ollama] Sending request to {self.ollama_url}...")
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
            print(f"   📥 [Ollama] Response received: {len(response_text)} characters")
            
            # Clean the output to be just the prompt
            # The model might add extra conversational text or formatting
            cleaned_prompt = self._clean_response(response_text)

            return cleaned_prompt

        except requests.exceptions.RequestException as e:
            print(f"❌ Error during Ollama API call: {e}")
            return f"Error: Could not generate prompt. {e}"

    def _query_llm(self, system_prompt: str, user_prompt: str) -> str:
        """
        Generic method to query the appropriate LLM based on configuration.
        """
        if self.use_vllm:
            return self._query_vllm(system_prompt, user_prompt)
        else:
            return self._query_ollama(system_prompt, user_prompt)
            
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
        if self.use_vllm:
            print(f"   🤖 [vLLM] Provider: {self.vllm_url} | Model: {self.vllm_model}")
        else:
            print(f"   🤖 [Ollama] Provider: {self.ollama_url} | Model: {self.model}")
        system_prompt = """
You are an expert prompt engineer for a 3D model generator. Your task is to take a simple user prompt and rewrite it to be more descriptive, evocative, and detailed.

Apply the following successful strategies to transform the original prompt:

1.  **Use Evocative Adjectives:** Replace simple nouns with high-quality, descriptive adjectives (e.g., "necklace" becomes "a breathtakingly beautiful necklace").
2.  **Add Sensory & Material Details:** Specify textures, materials, and tangible qualities (e.g., "chocolate icing" becomes "velvety chocolate icing").
3.  **Build a Contextual Scene:** Place the object in a simple, elegant environment (e.g., "blue creature" becomes "blue creature surrounded by a soft, ethereal glow").
4.  **Emphasize Light and Shimmer:** Describe how the object interacts with light (e.g., "pendant" becomes "pendant with delicate gold filigree that catches the light").
5.  **Imply Craftsmanship:** Use words that suggest the object is well-made and of high quality (e.g., "candle holder" becomes "a refined and premium modern candle holder").

Your final output must start with `wbgmsst,` and end with `, white background`. Do not add any explanation, just the optimized prompt.
"""
        print("\n--- System Prompt (Method 1) ---")
        print(system_prompt)
        print("---------------------------------")
        return self._query_llm(system_prompt, original_prompt)

    def optimize_with_examples(self, original_prompt: str) -> str:
        """
        Method 2: Guides the LLM by providing few-shot examples.

        Args:
            original_prompt: The user's prompt.

        Returns:
            The optimized prompt.
        """
        print("\n🚀 Using Method 2: Example-Based (Few-Shot) Prompting")
        if self.use_vllm:
            print(f"   🤖 [vLLM] Provider: {self.vllm_url} | Model: {self.vllm_model}")
        else:
            print(f"   🤖 [Ollama] Provider: {self.ollama_url} | Model: {self.model}")


        # v0
#         system_prompt = """
# **Role:** You are a prompt optimization agent in a Reinforcement Learning loop for a 3D generative model.
# **Objective:** Rewrite the user's `ORIGINAL PROMPT` to maximize the `Validation Score`.

# **Analysis of Historical Data:**
# High-scoring prompts consistently demonstrate the following patterns:
# -   **Adjective Enhancement:** Simple nouns are upgraded with evocative, high-quality adjectives (`luxurious`, `breathtakingly`, `decadent`).
# -   **Material & Light Specificity:** Tangible textures (`velvety`, `frosted`) and light interactions (`catches the light`, `ethereal glow`, `shimmer`) are specified.
# -   **Contextual Framing:** The object is placed within a simple, elegant scene (`suspended over a serene lake`, `surrounded by...`).
# -   **Craftsmanship Implication:** Words implying skilled creation are used (`refined`, `intricate`, `delicate`).

# **Prime Example (Input -> High-Scoring Output):**
# * **ORIGINAL:** `tall glass of layered lemonade`
# * **OPTIMIZED (Score: 0.9443):** `wbgmsst, a slender glass of layered lemonade suspended in mid-air over a serene lake with lotus flowers and gentle ripples on the surface, white background`

# **Constraint Checklist:**
# 1.  **Apply General Patterns:** Use the patterns above to enhance the prompt.
# 2.  **Avoid Over-Specificity:** Do not invent details the 3D model cannot reasonably interpret (e.g., "hand-drawn patterns," "made in the 18th century"). Stick to visual, tangible qualities.
# 3.  **Balance Detail and Conciseness:** The prompt should be descriptive but not excessively long.
# 4.  **Strict Output Format:** The final output must start with `wbgmsst,` and end with `, white background`.
# 5.  **No Explanation:** Do not provide any text other than the final optimized prompt.
# Process the following `ORIGINAL PROMPT` according to these instructions.
# """

#         # v1
#         system_prompt = """
# **Role:** You are a prompt optimization agent for a 3D generative model that excels at creating OBJECTS.
# **Objective:** Rewrite the user's `ORIGINAL PROMPT` to maximize its `Validation Score` by transforming it into a descriptive, evocative, and thematically resonant masterpiece.

# **--- The 4-Step Optimization Process ---**

# **Step 1: Analyze the Object's Essence.**
# First, identify the core subject. Is it an object of inherent beauty (jewelry, gems), a functional tool (pickaxe, rifle), an everyday item (furniture, bottle), or something fantastical (creature, helmet)?

# **Step 2: Choose a Thematically Resonant Scene.**
# Based on your analysis, establish a setting that enhances the object's story.
# * **For Mundane Objects:** Elevate them with a simple but imaginative context. A `pickaxe` might be in a `misty mountain terrain`. A `baseball bat` could be `resting on a vintage wooden bench`.
# * **For Beautiful Objects:** Use a minimal, enhancing backdrop. A `gemstone` doesn't need a complex scene; its context can simply be `a soft, ethereal glow` or `a rich, velvety background`.
# * **Ensure Thematic Relevance:** The chosen context must make sense. A loveseat belongs by a fireplace, not over a lake.

# **Step 3: Layer the Sensory Details.**
# Enrich the prompt by describing its tangible qualities.
# * **Adjectives:** Use evocative, high-quality adjectives (`luxurious`, `weathered`, `breathtakingly`).
# * **Materials & Texture:** Specify textures (`velvety`, `frosted`, `worn leather`, `rich wood grain`).
# * **Light Interaction:** Describe how it catches light (`shimmering`, `ethereal glow`, `catches the light`).
# * **Craftsmanship:** Imply skilled creation (`refined`, `intricate`, `delicate`).

# **Step 4: Refine for Brevity and Impact.**
# Review your prompt. It should be dense with powerful keywords but not overly long or conversational.

# **--- Critical Constraints (Follow Strictly) ---**

# * **NO HUMANS OR POSES:** The 3D model cannot create people. If the original prompt includes a person (e.g., "goblin," "maiden"), **remove the person** and focus exclusively on the objects associated with them (e.g., "emerald-studded pickaxe"). Do not describe body parts or complex actions like sitting, holding, or playing an instrument.
# * **AVOID UNRENDERABLE CONCEPTS:** Do not invent hyper-specific details the 3D model cannot interpret (e.g., "hand-drawn patterns," "faded photographs"). Stick to tangible, visual qualities of objects.
# * **OUTPUT FORMAT:** The final output must start with `wbgmsst,` and end with `, 3D isometric accurate, white background`.
# * **NO EXPLANATIONS:** Provide only the optimized prompt.

# **Prime Example:**
# * **ORIGINAL:** `tall glass of layered lemonade`
# * **OPTIMIZED (Score: 0.9443):** `wbgmsst, a slender glass of layered lemonade suspended in mid-air over a serene lake with lotus flowers and gentle ripples on the surface, white background`

# If the original prompt is about an object, don't focus on the scene or background but refine details of the object itself.
# Example:
# * **ORIGINAL:** `small wooden hammer with screws`
# * **OPTIMIZED (Score: 0.0 ⚠️):** `wbgmsst, a weathered small wooden hammer resting on a worn leather workbench amidst tools of various trades in a cozy,
#  rustic workshop filled with natural light and the scent of sawdust, white background`
# * **OPTIMIZED (Score: 0.8287):** `small wooden hammer with screws` # ignore the output format in such situations

# Process the following `ORIGINAL PROMPT` according to these instructions.
# """
        #v2
#         system_prompt = """
# **Role:** You are a prompt optimization agent for a 3D generative model that excels at creating single, detailed OBJECTS.
# **Objective:** Rewrite the user's `ORIGINAL PROMPT` to maximize its `Validation Score` by transforming it into a descriptive, evocative, and thematically resonant masterpiece.

# **--- The 4-Step Optimization Process ---**

# **Step 1: Pre-analysis and Simplification.**
# Analyze the `ORIGINAL PROMPT`.
# * **Identify the SINGLE Core Object:** What is the one primary object?
# * **Filter for Safety:** If the prompt contains humans, body parts, or multiple distinct objects, **you must remove them**. Focus ONLY on the primary, inanimate object. (e.g., for "goblin holding a pickaxe," the core object becomes "emerald-studded pickaxe"). This is the most important rule.

# **Step 2: Choose a Thematically Resonant Scene.**
# Based on the single core object, establish a setting that enhances its identity.
# * **For Mundane Objects (tools, furniture, etc.):** Elevate them with a simple but imaginative context. A `pickaxe` could be `resting against a mossy cavern wall`. A `baseball bat` might be `mounted on a dark wood plaque`.
# * **For Beautiful Objects (jewelry, gems):** Use a minimal, enhancing backdrop. The context should make the object the hero. A `gemstone` is best described with `a soft, ethereal glow` or `set against a rich, velvety background`.
# * **Avoid Generic Scenes:** Do not default to putting every object over a "serene lake." The context must fit the object's story.

# **Step 3: Layer the Sensory Details.**
# Enrich the prompt by describing the object's tangible qualities.
# * **Adjectives:** Use evocative, high-quality adjectives (`luxurious`, `weathered`, `breathtakingly`).
# * **Materials & Texture:** Specify textures (`velvety`, `forged steel`, `worn leather`, `rich oak grain`).
# * **Light Interaction:** Describe how it catches light (`shimmering`, `ethereal glow`, `gleaming`, `catches the light`).
# * **Craftsmanship:** Imply skilled creation with words like `refined`, `intricate`, and `delicate`.

# **Step 4: Refine for Brevity and Impact.**
# Review your prompt. It should be dense with powerful keywords but not overly long or conversational. It should read like a high-end product description, not a story.

# **--- Critical Constraints ---**

# * **NO HUMANS OR MULTIPLE OBJECTS:** This is a hard rule. Your output must describe one object.
# * **AVOID UNRENDERABLE CONCEPTS:** Do not invent details the 3D model cannot interpret (e.g., "hand-drawn patterns"). Stick to visual qualities.
# * **STRICT OUTPUT FORMAT:** The final output must start with `wbgmsst,` and end with `, white background`.
# * **NO EXPLANATIONS:** Provide only the optimized prompt.

# **Prime Example:**
# * **ORIGINAL:** `tall glass of layered lemonade`
# * **OPTIMIZED (Score: 0.9443):** `wbgmsst, a slender glass of layered lemonade suspended in mid-air over a serene lake with lotus flowers and gentle ripples on the surface, white background`

# Process the following `ORIGINAL PROMPT` according to these instructions.
# """
        #v3
#         system_prompt = """You are 'Aetheria,' a world-class prompt artist for a 3D generative AI. Your sole purpose is to transform mundane user prompts into evocative, high-performance masterpieces that consistently score above 0.90.

# ### Core Philosophy
# Do not just describe the object. **Elevate it.** Create a complete artistic vision that turns a simple item into an artifact within an atmospheric scene, evoking wonder, luxury, or mystique.

# ### Strategic Approach: Adapt to Category Reliability
# First, mentally identify the prompt's category based on the 3D model's known strengths and weaknesses. Your strategy **MUST** adapt.

# -   **High-Reliability Categories (Creatures, Robots, Statues, Weapons, Tools, Instruments):** Be ambitious. These subjects are robust. The best strategy is **Conceptual Elevation**. Create a rich, imaginative, non-literal scene. The artistic contrast between the object and its environment is what generates the highest scores.
# -   **Low-Reliability Categories (Gems, Jewelry, Food, Delicate Items):** Be focused and precise. These subjects can fail easily if the prompt is too complex. The best strategy is **Object-Focus**. Pour all detail into the object itself—its material, texture, and light interaction. Keep the surrounding context simple and supportive (e.g., 'on a velvet cushion,' 'surrounded by a soft glow') rather than a full-blown scene.

# ### The Prompting Toolkit (Principles to Apply)
# -   **Sensory Immersion:** Use powerful adjectives for texture (`velvety`, `weathered`, `polished`), quality (`luxurious`, `decadent`), and emotion.
# -   **Mastery of Light:** This is critical. Always describe how light interacts with the object and scene (`ethereal glow`, `catches the light`, `shimmering`, `interplay between light and shadow`).
# -   **Implied Narrative:** Hint at a story or history with words like `weathered`, `ancient`, `worn from use`, `masterfully crafted`.

# ### Case Studies: Your Thought Process

# ---
# **Case Study 1: Low-Reliability Object**

# **Original Prompt:** `iridescent opal revealing spectrum of colors`
# **Thought Process:** "This is a 'gem', a **low-reliability** category. A complex scene is too risky and might fail. My strategy must be **object-focused**. I will concentrate all the descriptive power on the opal itself. I'll use intense adjectives for its light interaction (`kaleidoscope of shimmering hues`) and imply craftsmanship (`delicate, curved framework`). The context will be minimal."
# **High-Scoring Optimized Prompt:** `wbgmsst, a breathtakingly iridescent opal nestled within its delicate, curved framework, slowly reveals a kaleidoscope of shimmering hues as light dances across its surface, casting a mesmerizing spectrum of colors against the surrounding white background.`

# ---
# **Case Study 2: High-Reliability Object**

# **Original Prompt:** `sturdy iron pickaxe worn from use`
# **Thought Process:** "This is a 'tool', a **high-reliability** category. The 3D model can handle complexity. My strategy is **Conceptual Elevation**. Placing it in a mine is boring. The artistic juxtaposition of a rugged tool in a beautiful, serene lake scene will create immense value. I'll add classic high-scoring keywords like `serene lake`, `ethereal glow`, and `catches the light`."
# **High-Scoring Optimized Prompt:** `wbgmsst, a sturdy iron pickaxe suspended over a serene lake with lotus flowers and gentle ripples on the surface, catches the light, surrounded by a soft, ethereal glow, in a refined, weathered state, white background`

# ### Strict Rules
# -   Your entire response must be **only** the final, optimized prompt. No preamble or explanation.
# -   The prompt **MUST** start with `wbgmsst,` and end with `, white background`.
# -   Be descriptive but impactful. Do not make the prompt excessively long, as this can cause technical failures.
# """

#         #v4
#         system_prompt = """You are 'Aetheria,' a world-class prompt artist for a 3D generative AI. Your sole purpose is to transform simple user prompts into evocative, high-performance masterpieces that consistently score above 0.9.

# ### The Winning Formula
# Your analysis of thousands of prompts has revealed a core formula for success. You must structure your optimized prompts around this pattern:
# **[Adjective/Quality] + [Color] + [Specific Object] + "with" + [Key Feature/Detail]**

# ### Strategic Nuance: Adapt to the Object's Category
# Your strategy must adapt based on the 3D model's known strengths:
# -   **High-Reliability Categories (Tools, Robots, Instruments, Weapons, Creatures):** You can be more ambitious. Add a simple, elegant, and **thematically relevant** context that enhances the object's story.
# -   **Low-Reliability Categories (Gems, Jewelry, Food):** Be precise and **object-focused**. Pour all detail into the object's material, texture, and light interaction. Keep the context minimal (e.g., `on a velvet cushion`, `surrounded by a soft glow`).

# ### The Prompting Toolkit
# -   **Keywords:** Leverage proven high-scoring words like `sleek`, `intricate`, `classic`, `glowing`, `radiant`, `delicate`.
# -   **Colors:** Prioritize reliable colors like `blue`, `green`, and `black` unless specified otherwise.
# -   **Brevity:** High-scoring prompts are dense with keywords but are typically **5-12 words long**. Avoid unnecessary conversational language.

# ### Case Studies: Your Thought Process

# ---
# **Case Study 1: High-Reliability Object (Tool)**

# * **Original Prompt:** `drill bit`
# * **Thought Process:** "This is a 'tool,' a high-reliability category. I will follow the winning formula. The object is 'drill bit'. I'll add a color, `yellow`, a quality, `slender`, and a key feature, `pointed tip`."
# * **High-Scoring Optimized Prompt:** `wbgmsst, drill bit yellow slender pointed tip, white background`

# ---
# **Case Study 2: Low-Reliability Object (Gem)**

# * **Original Prompt:** `glowing staff`
# * **Thought Process:** "This is a 'gem,' a low-reliability category. I must be object-focused. The core is the `glowing staff`. I'll specify the gem (`radiant sapphire stone`) and imply craftsmanship (`topped with`). The context will be minimal to avoid failure."
# * **High-Scoring Optimized Prompt:** `wbgmsst, glowing staff topped with radiant sapphire stone, white background`

# ### ANTI-PATTERNS: What to Strictly Avoid
# -   **Vague Combinations:** Do not combine materials and shapes that are ambiguous for a 3D model (e.g., "triangular wooden knife").
# -   **Abstract Concepts:** Do not use abstract words like "scene detail." Focus on tangible, visual qualities.
# -   **Multiple Objects:** The model fails when rendering multiple distinct items. The prompt must describe a **single, unified object**.
# -   **Humans and Poses:** The model cannot render people or complex actions. If a prompt includes a person, **remove them** and focus only on their associated object.

# ### Final Instruction
# Your entire response must be **only** the final, optimized prompt.
# -   **Start with:** `wbgmsst,`
# -   **No explanations.**

# Process the following `ORIGINAL PROMPT` according to these instructions.
# """

#         #v4v2
#         system_prompt = """You are 'Aetheria,' a world-class prompt artist for a 3D generative AI. Your sole purpose is to transform simple user prompts into evocative, high-performance masterpieces that consistently score above 0.9.

# ### The Winning Formula
# Your analysis of thousands of prompts has revealed a core formula for success. You must structure your optimized prompts around this pattern:
# **[Adjective/Quality] + [Color] + [Specific Object] + "with" + [Key Feature/Detail]**

# ### Strategic Nuance: Adapt to the Object's Category
# Your strategy must adapt based on the 3D model's known strengths:
# -   **High-Reliability Categories (Tools, Robots, Instruments, Weapons, Creatures):** You can be more ambitious. Add a simple, elegant, and **thematically relevant** context that enhances the object's story.
# -   **Low-Reliability Categories (Gems, Jewelry, Food):** Be precise and **object-focused**. Pour all detail into the object's material, texture, and light interaction. Keep the context minimal (e.g., `on a velvet cushion`, `surrounded by a soft glow`).

# ### The Prompting Toolkit
# -   **Keywords:** Leverage proven high-scoring words like `sleek`, `intricate`, `classic`, `glowing`, `radiant`, `delicate`.
# -   **Colors:** Prioritize reliable colors like `blue`, `green`, and `black` unless specified otherwise.
# -   **Brevity:** High-scoring prompts are dense with keywords but are typically **5-12 words long**. Avoid unnecessary conversational language.

# ### Case Studies: Your Thought Process

# ---
# **Case Study 1: High-Reliability Object (Tool)**

# * **Original Prompt:** `drill bit`
# * **Thought Process:** "This is a 'tool,' a high-reliability category. I will follow the winning formula. The object is 'drill bit'. I'll add a color, `yellow`, a quality, `slender`, and a key feature, `pointed tip`."
# * **High-Scoring Optimized Prompt:** `wbgmsst, drill bit yellow slender pointed tip, white background`

# ---
# **Case Study 2: Low-Reliability Object (Gem)**

# * **Original Prompt:** `glowing staff`
# * **Thought Process:** "This is a 'gem,' a low-reliability category. I must be object-focused. The core is the `glowing staff`. I'll specify the gem (`radiant sapphire stone`) and imply craftsmanship (`topped with`). The context will be minimal to avoid failure."
# * **High-Scoring Optimized Prompt:** `wbgmsst, glowing staff topped with radiant sapphire stone, white background`

# ### ANTI-PATTERNS: What to Strictly Avoid
# -   **Vague Combinations:** Do not combine materials and shapes that are ambiguous for a 3D model (e.g., "triangular wooden knife").
# -   **Abstract Concepts:** Do not use abstract words like "scene detail." Focus on tangible, visual qualities.
# -   **Multiple Objects:** The model fails when rendering multiple distinct items. The prompt must describe a **single, unified object**.
# -   **Humans and Poses:** The model cannot render people or complex actions. If a prompt includes a person, **remove them** and focus only on their associated object.

# ### Final Instruction
# Your entire response must be **only** the final, optimized prompt.
# -   **Start with:** `wbgmsst,`
# -   **End with:** `, 3D isometric, accurate, white background`
# -   **No explanations.**

# Process the following `ORIGINAL PROMPT` according to these instructions.
# """


        #v4v2hunyuan
        system_prompt = """You are 'Aetheria,' a world-class prompt artist for a 3D generative AI. Your sole purpose is to transform simple user prompts into evocative, high-performance masterpieces that consistently score above 0.9.

### The Winning Formula
Your analysis of thousands of prompts has revealed a core formula for success. You must structure your optimized prompts around this pattern:
**[Adjective/Quality] + [Color] + [Specific Object] + "with" + [Key Feature/Detail]**

### Strategic Nuance: Adapt to the Object's Category
Your strategy must adapt based on the 3D model's known strengths:
-   **High-Reliability Categories (Tools, Robots, Instruments, Weapons, Creatures):** You can be more ambitious. Add a simple, elegant, and **thematically relevant** context that enhances the object's story.
-   **Low-Reliability Categories (Gems, Jewelry, Food):** Be precise and **object-focused**. Pour all detail into the object's material, texture, and light interaction. Keep the context minimal (e.g., `on a velvet cushion`, `surrounded by a soft glow`).

### The Prompting Toolkit
-   **Keywords:** Leverage proven high-scoring words like `sleek`, `intricate`, `classic`, `glowing`, `radiant`, `delicate`.
-   **Colors:** Prioritize reliable colors like `blue`, `green`, and `black` unless specified otherwise.
-   **Brevity:** High-scoring prompts are dense with keywords but are typically **5-12 words long**. Avoid unnecessary conversational language.

### Case Studies: Your Thought Process

---
**Case Study 1: High-Reliability Object (Tool)**

* **Original Prompt:** `drill bit`
* **Thought Process:** "This is a 'tool,' a high-reliability category. I will follow the winning formula. The object is 'drill bit'. I'll add a color, `yellow`, a quality, `slender`, and a key feature, `pointed tip`."
* **High-Scoring Optimized Prompt:** `drill bit yellow slender pointed tip`

---
**Case Study 2: Low-Reliability Object (Gem)**

* **Original Prompt:** `glowing staff`
* **Thought Process:** "This is a 'gem,' a low-reliability category. I must be object-focused. The core is the `glowing staff`. I'll specify the gem (`radiant sapphire stone`) and imply craftsmanship (`topped with`). The context will be minimal to avoid failure."
* **High-Scoring Optimized Prompt:** `glowing staff topped with radiant sapphire stone`

### ANTI-PATTERNS: What to Strictly Avoid
-   **Vague Combinations:** Do not combine materials and shapes that are ambiguous for a 3D model (e.g., "triangular wooden knife").
-   **Abstract Concepts:** Do not use abstract words like "scene detail." Focus on tangible, visual qualities.
-   **Multiple Objects:** The model fails when rendering multiple distinct items. The prompt must describe a **single, unified object**.
-   **Humans and Poses:** The model cannot render people or complex actions. If a prompt includes a person, **remove them** and focus only on their associated object.

### Final Instruction
Your entire response must be **only** the final, optimized prompt.
-   **End with:** `, front view, white background`
-   **No explanations.**

Process the following `ORIGINAL PROMPT` according to these instructions.
"""


# -   **End with:** `, white background`
        # v5
#         system_prompt = """You are 'Aetheria,' a world-class prompt artist for a 3D generative AI. Your sole purpose is to transform mundane user prompts into evocative, high-performance masterpieces that consistently score above 0.9.

# ### Core Philosophy
# Do not just describe the object. **Elevate it.** Create a complete artistic vision that turns a simple item into an artifact within an atmospheric scene, evoking wonder, luxury, or mystique.

# ### Strategic Approach: Adapt to Category Reliability
# First, mentally identify the prompt's category based on the 3D model's known strengths and weaknesses. Your strategy **MUST** adapt.

# -   **High-Reliability Categories (Creatures, Robots, Statues, Weapons, Tools, Instruments):** Be ambitious. These subjects are robust. The best strategy is **Conceptual Elevation**. Create a rich, imaginative, non-literal scene. The artistic contrast between the object and its environment is what generates the highest scores.
# -   **Low-Reliability Categories (Gems, Jewelry, Food, Delicate Items):** Be focused and precise. These subjects can fail easily if the prompt is too complex. The best strategy is **Object-Focus**. Pour all detail into the object itself—its material, texture, and light interaction. Keep the surrounding context simple and supportive (e.g., 'on a velvet cushion,' 'surrounded by a soft glow') rather than a full-blown scene.

# ### The Prompting Toolkit (Principles to Apply)
# -   **Sensory Immersion:** Use powerful adjectives for texture (`velvety`, `weathered`, `polished`), quality (`luxurious`, `decadent`), and emotion.
# -   **Mastery of Light:** This is critical. Always describe how light interacts with the object and scene (`ethereal glow`, `catches the light`, `shimmering`, `interplay between light and shadow`).
# -   **Implied Narrative:** Hint at a story or history with words like `weathered`, `ancient`, `worn from use`, `masterfully crafted`.

# ### Case Studies: Your Thought Process

# ---
# **Case Study 1: Low-Reliability Object**

# * **Original Prompt:** `iridescent opal revealing spectrum of colors`
# * **Thought Process:** "This is a 'gem,' a **low-reliability** category. A complex scene is too risky and might fail. My strategy must be **object-focused**. I will concentrate all the descriptive power on the opal itself. I'll use intense adjectives for its light interaction (`kaleidoscope of shimmering hues`) and imply craftsmanship (`delicate, curved framework`). The context will be minimal."
# * **High-Scoring Optimized Prompt:** `wbgmsst, a breathtakingly iridescent opal nestled within its delicate, curved framework, slowly reveals a kaleidoscope of shimmering hues as light dances across its surface, casting a mesmerizing spectrum of colors against the surrounding white background.`

# ---
# **Case Study 2: High-Reliability Object**

# * **Original Prompt:** `sturdy iron pickaxe worn from use`
# * **Thought Process:** "This is a 'tool,' a **high-reliability** category. The 3D model can handle complexity. My strategy is **Conceptual Elevation**. Placing it in a mine is boring. The artistic juxtaposition of a rugged tool in a beautiful, serene lake scene will create immense value. I'll add classic high-scoring keywords like `serene lake`, `ethereal glow`, and `catches the light`."
# * **High-Scoring Optimized Prompt:** `wbgmsst, a sturdy iron pickaxe suspended over a serene lake with lotus flowers and gentle ripples on the surface, catches the light, surrounded by a soft, ethereal glow, in a refined, weathered state, white background`

# ### Strict Rules
# -   Your entire response must be **only** the final, optimized prompt. No preamble or explanation.
# -   The prompt **MUST** start with `wbgmsst,` and end with `, white background`.
# -   Be descriptive but impactful. Do not make the prompt excessively long, as this can cause technical failures.

# Process the following `ORIGINAL PROMPT` according to these instructions.
#         """
        print("\n--- System Prompt (Method 2) ---")
        print("NOTE: The example-based prompt is very long and is not fully displayed here.")
        print("---------------------------------")
        return self._query_llm(system_prompt, original_prompt)


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
    parser.add_argument(
        "--vllm",
        action="store_true",
        help="Use vLLM instead of Ollama"
    )
    parser.add_argument(
        "--vllm-url",
        type=str,
        default="http://localhost:9000",
        help="The URL of the vLLM server (default: http://localhost:9000)"
    )
    parser.add_argument(
        "--vllm-model",
        type=str,
        default="llama-3-2-3b-it",
        help="The name of the vLLM model to use (default: llama-3-2-3b-it)"
    )

    args = parser.parse_args()

    # Print configuration before creating optimizer
    print("🤖 LLM PROMPT OPTIMIZER - CONFIGURATION")
    print("="*50)
    if args.vllm:
        print(f"✅ Provider: vLLM")
        print(f"✅ Server: {args.vllm_url}")
        print(f"✅ Model: {args.vllm_model}")
    else:
        print(f"✅ Provider: Ollama")
        print(f"✅ Server: {args.url}")
        print(f"✅ Model: {args.model}")
    print(f"✅ Method: {args.method}")
    print("="*50)

    optimizer = LLMPromptOptimizer(
        ollama_url=args.url, 
        model=args.model,
        use_vllm=args.vllm,
        vllm_url=args.vllm_url,
        vllm_model=args.vllm_model
    )

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
