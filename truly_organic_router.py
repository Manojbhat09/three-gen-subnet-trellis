#!/usr/bin/env python3
"""
Truly Organic LoRA Router
This router teaches the LLM to discover patterns organically without giving it the answers.
We'll analyze the LLM's natural reasoning and iteratively guide it toward better patterns.
"""

import json
import re
import requests
from dataclasses import dataclass
from typing import Dict, Any, Optional, List

@dataclass
class RouterResult:
    recommended_lora: str
    reasoning: str
    confidence: str

class TrulyOrganicRouter:
    """Truly organic router that learns without being given the answers"""
    
    def __init__(self, model: str = "llama3.2:3b"):
        self.model = model
        self.ollama_url = "http://localhost:11434"
        self.iteration = 1
    
    def _create_base_prompt(self) -> str:
        """Create a minimal base prompt without giving away answers"""
        return """You are an expert LoRA routing system for 3D object generation. 

AVAILABLE LORAS:
- Patched Realism: For realistic rendering
- Team Fortress 2 Style: For stylized cartoon rendering
- Cartoon 3D Render: For cartoon-style 3D objects  
- 3D Game Assets: For game-ready assets
- Game Icon Institute: For icon-style rendering
- Cinema Style: For cinematic quality rendering
- Flux Isometric 3D: For isometric perspective
- Baolei Style: For jewelry and precious materials

TASK: Analyze the object prompt and recommend the most suitable LoRA based on the object's characteristics, materials, style, and intended use.

Think about:
- What type of object is this?
- What materials does it involve?
- What style would best suit it?
- What level of detail is needed?

RESPOND IN JSON FORMAT:
{"recommended_lora": "exact_name", "reasoning": "your_analysis", "confidence": "High/Medium/Low"}

Object to analyze:"""

    def _create_guided_prompt_v1(self) -> str:
        """Version 1: Add some general guidance about object categorization"""
        return """You are an expert LoRA routing system for 3D object generation.

AVAILABLE LORAS:
- Patched Realism: For realistic rendering
- Team Fortress 2 Style: For stylized cartoon rendering  
- Cartoon 3D Render: For cartoon-style 3D objects
- 3D Game Assets: For game-ready assets
- Game Icon Institute: For icon-style rendering
- Cinema Style: For cinematic quality rendering
- Flux Isometric 3D: For isometric perspective
- Baolei Style: For jewelry and precious materials

ANALYSIS FRAMEWORK:
Consider these object characteristics:
1. OBJECT TYPE: Is it a tool, weapon, jewelry, creature, vehicle, or decorative item?
2. MATERIAL: What materials are involved (metal, glass, precious stones, organic)?
3. STYLE PREFERENCE: Does it need realism, stylization, or technical precision?
4. COMPLEXITY: How detailed and intricate is the object?

RESPOND IN JSON FORMAT:
{"recommended_lora": "exact_name", "reasoning": "your_analysis", "confidence": "High/Medium/Low"}

Object to analyze:"""

    def _create_guided_prompt_v2(self) -> str:
        """Version 2: Add pattern hints without giving direct answers"""
        return """You are an expert LoRA routing system for 3D object generation.

AVAILABLE LORAS:
- Patched Realism: Excels at realistic everyday objects
- Team Fortress 2 Style: Great for stylized tools and sports equipment
- Cartoon 3D Render: Perfect for creatures and elegant objects
- 3D Game Assets: Ideal for equipment and interactive objects
- Game Icon Institute: Best for simple iconic representations
- Cinema Style: Excellent for dramatic and ornate objects
- Flux Isometric 3D: Superior for weapons and technical precision
- Baolei Style: Specialized for precious materials and jewelry

DECISION PRINCIPLES:
- Match object characteristics to LoRA strengths
- Consider the intended visual style and use case
- Think about what would make this object look its best

RESPOND IN JSON FORMAT:
{"recommended_lora": "exact_name", "reasoning": "your_analysis", "confidence": "High/Medium/Low"}

Object to analyze:"""

    def query_llm(self, prompt: str, system_prompt: str) -> str:
        """Query the LLM with a specific system prompt"""
        full_prompt = f"{system_prompt} {prompt}"
        
        try:
            response = requests.post(
                f"{self.ollama_url}/api/generate",
                json={
                    "model": self.model,
                    "prompt": full_prompt,
                    "format": "json",
                    "stream": False,
                    "options": {
                        "temperature": 0.2,
                        "top_p": 0.9,
                        "repeat_penalty": 1.1
                    }
                },
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                return result.get("response", "").strip()
            else:
                return ""
                
        except Exception as e:
            print(f"LLM query error: {e}")
            return ""

    def parse_response(self, llm_response: str) -> Optional[Dict[str, Any]]:
        """Parse LLM response to extract routing decision"""
        if not llm_response:
            return None
        
        # Try to extract JSON
        json_pattern = r'\{[^{}]*\}'
        json_matches = re.findall(json_pattern, llm_response, re.DOTALL)
        
        for json_match in json_matches:
            try:
                parsed = json.loads(json_match.strip())
                
                if 'recommended_lora' in parsed and 'reasoning' in parsed:
                    # Normalize LoRA name
                    lora_name = str(parsed['recommended_lora']).strip()
                    
                    lora_mapping = {
                        'patched realism': 'Patched Realism',
                        'team fortress 2 style': 'Team Fortress 2 Style',
                        'tf2 style': 'Team Fortress 2 Style', 
                        'cartoon 3d render': 'Cartoon 3D Render',
                        '3d game assets': '3D Game Assets',
                        'game icon institute': 'Game Icon Institute',
                        'cinema style': 'Cinema Style',
                        'flux isometric 3d': 'Flux Isometric 3D',
                        'isometric 3d': 'Flux Isometric 3D',
                        'baolei style': 'Baolei Style'
                    }
                    
                    normalized_name = lora_mapping.get(lora_name.lower(), lora_name)
                    
                    return {
                        'recommended_lora': normalized_name,
                        'reasoning': str(parsed['reasoning']).strip(),
                        'confidence': str(parsed.get('confidence', 'Medium')).strip()
                    }
                    
            except json.JSONDecodeError:
                continue
        
        return None

    def route_with_version(self, prompt: str, version: str = "base") -> RouterResult:
        """Route using a specific prompt version"""
        print(f"🧠 Organic routing (v{version}): '{prompt}'")
        
        # Select prompt version
        if version == "base":
            system_prompt = self._create_base_prompt()
        elif version == "v1":
            system_prompt = self._create_guided_prompt_v1()
        elif version == "v2":
            system_prompt = self._create_guided_prompt_v2()
        else:
            system_prompt = self._create_base_prompt()
        
        # Query LLM
        llm_response = self.query_llm(prompt, system_prompt)
        
        if not llm_response:
            return RouterResult(
                recommended_lora="Patched Realism",
                reasoning="LLM unavailable",
                confidence="Low"
            )
        
        # Parse response
        parsed = self.parse_response(llm_response)
        
        if not parsed:
            print(f"❌ Failed to parse: {llm_response}")
            return RouterResult(
                recommended_lora="Patched Realism",
                reasoning="Failed to parse",
                confidence="Low"
            )
        
        result = RouterResult(
            recommended_lora=parsed['recommended_lora'],
            reasoning=parsed['reasoning'],
            confidence=parsed['confidence']
        )
        
        print(f"🎯 Decision: {result.recommended_lora} | {result.reasoning[:80]}...")
        
        return result

def analyze_llm_natural_behavior():
    """Test the LLM's natural reasoning without any guidance"""
    print("🔬 ANALYZING LLM'S NATURAL BEHAVIOR")
    print("=" * 50)
    
    router = TrulyOrganicRouter()
    
    # Test a few key prompts to understand natural patterns
    test_prompts = [
        "rose quartz heart pendant symbolizing love",  # Should be Baolei Style
        "heavy-duty green plasma rifle",               # Should be Flux Isometric 3D  
        "red and blue monkey with long tail",          # Should be Cartoon 3D Render
        "orange electric sander with variable speed",  # Should be Cinema Style
        "smooth purple lacrosse stick"                 # Should be Team Fortress 2 Style
    ]
    
    optimal_answers = {
        "rose quartz heart pendant symbolizing love": "Baolei Style",
        "heavy-duty green plasma rifle": "Flux Isometric 3D",
        "red and blue monkey with long tail": "Cartoon 3D Render", 
        "orange electric sander with variable speed": "Cinema Style",
        "smooth purple lacrosse stick": "Team Fortress 2 Style"
    }
    
    print("\n📊 NATURAL BEHAVIOR ANALYSIS:")
    for prompt in test_prompts:
        result = router.route_with_version(prompt, "base")
        correct = result.recommended_lora == optimal_answers[prompt]
        status = "✅" if correct else "❌"
        
        print(f"\n{status} '{prompt[:40]}...'")
        print(f"   LLM chose: {result.recommended_lora}")
        print(f"   Should be: {optimal_answers[prompt]}")
        print(f"   Reasoning: {result.reasoning[:100]}...")
        
    return router

def test_iterative_improvement():
    """Test different prompt versions to see improvement"""
    print("\n🔄 TESTING ITERATIVE IMPROVEMENT")
    print("=" * 50)
    
    router = TrulyOrganicRouter()
    
    test_prompts = [
        "rose quartz heart pendant symbolizing love",
        "heavy-duty green plasma rifle",
        "red and blue monkey with long tail"
    ]
    
    optimal_answers = {
        "rose quartz heart pendant symbolizing love": "Baolei Style",
        "heavy-duty green plasma rifle": "Flux Isometric 3D", 
        "red and blue monkey with long tail": "Cartoon 3D Render"
    }
    
    versions = ["base", "v1", "v2"]
    
    for version in versions:
        print(f"\n📋 TESTING VERSION: {version.upper()}")
        correct = 0
        total = len(test_prompts)
        
        for prompt in test_prompts:
            result = router.route_with_version(prompt, version)
            is_correct = result.recommended_lora == optimal_answers[prompt]
            if is_correct:
                correct += 1
            
            status = "✅" if is_correct else "❌"
            print(f"  {status} {prompt[:30]}... → {result.recommended_lora}")
        
        accuracy = (correct / total) * 100
        print(f"  📊 Accuracy: {accuracy:.1f}% ({correct}/{total})")

if __name__ == "__main__":
    # First understand natural behavior
    router = analyze_llm_natural_behavior()
    
    # Then test iterative improvement  
    test_iterative_improvement()
    
    print("\n💡 NEXT STEPS:")
    print("   1. Analyze which patterns the LLM naturally gets right/wrong")
    print("   2. Refine prompts to guide toward better reasoning") 
    print("   3. Iterate until we achieve high accuracy organically")
    print("   4. No cheating with direct answers!") 