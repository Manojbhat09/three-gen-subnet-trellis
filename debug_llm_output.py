#!/usr/bin/env python3
"""
Debug LLM Output to see what's going wrong
"""

import requests
import json

def debug_llm_response():
    system_prompt = """You are an expert LoRA Router for 3D generation. Analyze the user's prompt and recommend the best LoRA based on object characteristics and artistic principles.

AVAILABLE LORAS:
- Patched Realism (avg: 0.8283, range: 0.7325-0.8603) - Most consistent
- Team Fortress 2 Style (avg: 0.8116, range: 0.7621-0.8380) - Consistent stylized
- Cartoon 3D Render (avg: 0.6694, range: 0.0000-0.9445) - High variance, extreme peaks
- 3D Game Assets (avg: 0.6647, range: 0.0000-0.9367) - Game-focused, inconsistent
- Game Icon Institute (avg: 0.6429, range: 0.0000-0.8867) - Simple objects specialist
- Cinema Style (avg: 0.5026, range: 0.0000-0.9050) - Cinematic but unreliable
- Flux Isometric 3D (avg: 0.4956, range: 0.0000-0.8452) - Lowest performance

DECISION FRAMEWORK:
Analyze the prompt for these characteristics and match to LoRA strengths:
- High-variance LoRAs (Cartoon 3D, 3D Game Assets) → Use for their specialties
- Consistent LoRAs (Patched Realism, TF2 Style) → Use for reliability
- Specialist LoRAs (Game Icon Institute) → Use when characteristics align

You MUST respond with ONLY a JSON object in this EXACT format:
{"recommended_lora": "LoRA Name", "reasoning": "Analysis-based explanation", "confidence": "High"}

User Prompt:"""

    test_prompt = "plastic straw of drink"
    full_prompt = f"{system_prompt}\n\n{test_prompt}"
    
    payload = {
        "model": "llama3.2:3b",
        "prompt": full_prompt,
        "stream": False,
        "options": {
            "temperature": 0.1,
            "top_p": 0.9,
            "max_tokens": 150,
        }
    }
    
    try:
        response = requests.post("http://localhost:11434/api/generate", json=payload, timeout=25)
        if response.status_code == 200:
            result = response.json()
            llm_output = result.get("response", "").strip()
            print("="*60)
            print("🔍 RAW LLM OUTPUT:")
            print("="*60)
            print(repr(llm_output))
            print("\n" + "="*60)
            print("🔍 FORMATTED OUTPUT:")
            print("="*60)
            print(llm_output)
            print("\n" + "="*60)
            
            # Try to find any JSON-like patterns
            import re
            json_patterns = re.findall(r'\{[^}]*\}', llm_output)
            if json_patterns:
                print("🔍 FOUND JSON PATTERNS:")
                print("="*60)
                for i, pattern in enumerate(json_patterns):
                    print(f"Pattern {i+1}: {pattern}")
            else:
                print("❌ NO JSON PATTERNS FOUND")
                
        else:
            print(f"❌ Request failed with status {response.status_code}")
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    debug_llm_response() 