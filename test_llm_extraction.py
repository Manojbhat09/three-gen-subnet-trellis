#!/usr/bin/env python3
"""
Simple test script to isolate LLM component extraction failures
"""

import json
import re
from llm_prompt_optimizer_v12_f1_lora import LLMPromptOptimizer

def test_llm_extraction():
    """Test LLM component extraction with different prompts"""
    
    # Initialize optimizer
    optimizer = LLMPromptOptimizer(
        use_vllm=True,
        vllm_url="http://localhost:9002",
        vllm_model="llama-3-2-3b-it"
    )
    
    # Test prompts
    test_prompts = [
        "blue car",
        "stone-etched armor with leafy pattern", 
        "small pink umbrella on table",
        "intricate sandstone sculpture of cat lounging"
    ]
    
    for prompt in test_prompts:
        print(f"\n{'='*60}")
        print(f"TESTING PROMPT: '{prompt}'")
        print(f"{'='*60}")
        
        # Test the component extraction prompt
        system_prompt = """You are a JSON generator for 3D model prompt analysis. Your ONLY task is to output valid JSON.

**CRITICAL: RESPOND ONLY WITH VALID JSON. NO TEXT, NO EXPLANATIONS, NO COMMENTARY.**

**REQUIRED FORMAT:**
{
  "core_subject": "exact subject from prompt",
  "enhancements": {
    "quality_adjectives": ["descriptive words that add quality"],
    "material_details": ["materials, textures, properties"],
    "light_interaction": ["lighting, visual effects"],
    "context": ["background, setting, additional details"]
  }
}

**EXAMPLES:**

Input: "intricate sandstone sculpture of cat lounging"
Output: {"core_subject":"sandstone sculpture of cat lounging","enhancements":{"quality_adjectives":["intricate"],"material_details":[],"light_interaction":[],"context":[]}}

Input: "stone-etched armor with leafy pattern"
Output: {"core_subject":"stone-etched armor with leafy pattern","enhancements":{"quality_adjectives":[],"material_details":[],"light_interaction":[],"context":[]}}

Input: "glowing crystal pendant with silver chain"
Output: {"core_subject":"crystal pendant with silver chain","enhancements":{"quality_adjectives":[],"material_details":["crystal","silver"],"light_interaction":["glowing"],"context":[]}}

**RULES:**
1. Core subject includes descriptive words that are part of the object name
2. Quality adjectives are separate descriptive words that add detail
3. Material details are physical properties and textures
4. Light interaction includes visual effects and lighting
5. Context includes background and setting details

**FINAL INSTRUCTION: OUTPUT ONLY VALID JSON. NOTHING ELSE.**"""

        # Start with opening brace to cue JSON output
        user_prompt = f"{{ \"core_subject\": \"{prompt}\", \"enhancements\": {{"
        
        print(f"🔍 Sending to LLM:")
        print(f"   System prompt length: {len(system_prompt)} chars")
        print(f"   User prompt: '{user_prompt}'")
        
        try:
            response = optimizer._query_llm(system_prompt, user_prompt)
            print(f"✅ LLM Response: '{response}'")
            
            # Try to parse as JSON
            try:
                parsed = json.loads(response)
                print(f"✅ JSON parsing successful: {json.dumps(parsed, indent=2)}")
            except json.JSONDecodeError as e:
                print(f"❌ JSON parsing failed: {e}")
                print(f"🔍 Raw response analysis:")
                print(f"   Length: {len(response)} chars")
                print(f"   Starts with {{: {response.strip().startswith('{')}")
                print(f"   Ends with }}: {response.strip().endswith('}')}")
                print(f"   Contains 'core_subject': {'core_subject' in response}")
                print(f"   Contains 'enhancements': {'enhancements' in response}")
                
                # Show first and last 100 chars
                print(f"   First 100 chars: '{response[:100]}'")
                print(f"   Last 100 chars: '{response[-100:]}'")
                
        except Exception as e:
            print(f"❌ LLM query failed: {e}")
        
        print(f"{'='*60}\n")

if __name__ == "__main__":
    test_llm_extraction()
