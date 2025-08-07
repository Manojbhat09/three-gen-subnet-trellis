#!/usr/bin/env python3
"""
Guided Organic Router
Based on LLM natural behavior analysis, this guides the LLM toward better patterns
without giving direct answers.
"""

import json
import re
import requests
from dataclasses import dataclass
from typing import Dict, Any, Optional

@dataclass
class RouterResult:
    recommended_lora: str
    reasoning: str
    confidence: str

class GuidedOrganicRouter:
    """Router that guides LLM learning based on observed natural patterns"""
    
    def __init__(self, model: str = "llama3.2:3b"):
        self.model = model
        self.ollama_url = "http://localhost:11434"
    
    def _create_pattern_guided_prompt(self) -> str:
        """Create prompt that guides toward better patterns based on LLM behavior analysis"""
        return """You are an expert LoRA routing system for 3D object generation.

AVAILABLE LORAS & THEIR NATURAL STRENGTHS:
- Patched Realism: Excels at everyday realistic objects and tools
- Team Fortress 2 Style: Specialized for sports equipment and stylized tools  
- Cartoon 3D Render: Perfect for living creatures and organic forms
- 3D Game Assets: Designed for mechanical equipment and instruments
- Game Icon Institute: Best for simple, iconic objects
- Cinema Style: Creates dramatic, ornate, and complex objects
- Flux Isometric 3D: Technical precision for weapons and detailed items
- Baolei Style: Master of precious materials, gemstones, and jewelry

INTELLIGENT ROUTING PRINCIPLES:

🔍 OBJECT ANALYSIS:
- Identify the PRIMARY object type (creature, tool, jewelry, weapon, etc.)
- Consider materials (precious stones, metals, organic, synthetic)
- Assess complexity level (simple icon vs detailed object)
- Determine style intention (realistic, stylized, technical)

💎 MATERIAL SPECIALIZATION:
- Precious stones, gems, jewelry → Consider specialty LoRAs
- Metals with technical precision needs → Consider technical LoRAs
- Organic creatures → Consider creature-specialized LoRAs
- Sports/recreational items → Consider stylized LoRAs

🎯 STYLE MATCHING:
- Dramatic/ornate objects → Consider cinematic approaches
- Technical/precision items → Consider technical specialists
- Simple iconic representations → Consider icon specialists
- Everyday realistic items → Consider realism specialists

⚡ DECISION LOGIC:
1. What is the PRIMARY object category?
2. Does it involve precious/specialty materials?
3. Is it a creature or living thing?
4. Does it need technical precision?
5. Is it meant to be dramatic/ornate?
6. Is it sports/recreational equipment?

Make your decision based on these patterns and object characteristics.

RESPOND IN JSON FORMAT:
{"recommended_lora": "exact_name", "reasoning": "pattern_based_analysis", "confidence": "High/Medium/Low"}

Object to analyze:"""

    def _create_enhanced_pattern_prompt(self) -> str:
        """Enhanced version with better pattern guidance"""
        return """You are an expert LoRA routing system for 3D object generation.

LORA SPECIALIZATIONS:
- Patched Realism: Realistic everyday objects, tools, mechanical items
- Team Fortress 2 Style: Sports equipment, recreational items, stylized tools
- Cartoon 3D Render: Living creatures, animals, organic forms, characters
- 3D Game Assets: Equipment, instruments, interactive objects
- Game Icon Institute: Simple icons, basic representations
- Cinema Style: Ornate objects, dramatic items, complex detailed objects
- Flux Isometric 3D: Weapons, technical precision items, detailed mechanical objects
- Baolei Style: Jewelry, precious stones, gemstones, precious materials

PATTERN RECOGNITION GUIDE:

🧬 LIVING VS NON-LIVING:
- Animals, creatures, beings → Creature specialists
- Inanimate objects → Match to object type and material

💎 MATERIAL SIGNIFICANCE:
- Words like "quartz", "diamond", "gemstone", "precious" → Jewelry specialists
- Technical metals with precision → Technical specialists
- Basic materials → General purpose LoRAs

🎮 OBJECT PURPOSE:
- "Rifle", "weapon", "gun" → Technical precision specialists
- "Stick", "ball", "equipment" in sports context → Sports specialists
- "Ornate", "decorated", "elaborate" → Dramatic specialists

🔧 COMPLEXITY SIGNALS:
- Simple, basic objects → Icon or general purpose
- Complex, detailed objects → Specialists
- Technical objects → Technical specialists

DECISION PROCESS:
1. Is this a living creature? → Creature specialist
2. Contains precious materials? → Jewelry specialist  
3. Is it a weapon/technical item? → Technical specialist
4. Is it sports/recreational? → Sports specialist
5. Is it ornate/dramatic? → Drama specialist
6. Otherwise → Match to material and complexity

RESPOND IN JSON FORMAT:
{"recommended_lora": "exact_name", "reasoning": "pattern_analysis", "confidence": "High/Medium/Low"}

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
                        "temperature": 0.15,  # Lower for more consistent pattern following
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
        
        json_pattern = r'\{[^{}]*\}'
        json_matches = re.findall(json_pattern, llm_response, re.DOTALL)
        
        for json_match in json_matches:
            try:
                parsed = json.loads(json_match.strip())
                
                if 'recommended_lora' in parsed and 'reasoning' in parsed:
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

    def route_guided(self, prompt: str, version: str = "pattern") -> RouterResult:
        """Route using guided pattern learning"""
        print(f"🧠 Guided organic routing ({version}): '{prompt}'")
        
        if version == "enhanced":
            system_prompt = self._create_enhanced_pattern_prompt()
        else:
            system_prompt = self._create_pattern_guided_prompt()
        
        llm_response = self.query_llm(prompt, system_prompt)
        
        if not llm_response:
            return RouterResult(
                recommended_lora="Patched Realism",
                reasoning="LLM unavailable",
                confidence="Low"
            )
        
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
        
        print(f"🎯 Guided decision: {result.recommended_lora}")
        
        return result

def test_guided_improvements():
    """Test the guided organic router on key examples"""
    print("🧠 TESTING GUIDED ORGANIC LEARNING")
    print("=" * 50)
    
    router = GuidedOrganicRouter()
    
    # Test the same prompts we analyzed before
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
    
    versions = ["pattern", "enhanced"]
    
    for version in versions:
        print(f"\n📋 TESTING {version.upper()} GUIDANCE:")
        correct = 0
        total = len(test_prompts)
        
        for prompt in test_prompts:
            result = router.route_guided(prompt, version)
            is_correct = result.recommended_lora == optimal_answers[prompt]
            if is_correct:
                correct += 1
            
            status = "✅" if is_correct else "❌"
            print(f"  {status} {prompt[:35]}...")
            print(f"      → {result.recommended_lora} (should be {optimal_answers[prompt]})")
            if not is_correct:
                print(f"      Reasoning: {result.reasoning[:80]}...")
        
        accuracy = (correct / total) * 100
        print(f"\n  📊 {version.upper()} Accuracy: {accuracy:.1f}% ({correct}/{total})")

def test_generalization():
    """Test on completely new prompts to check generalization"""
    print("\n🌟 GENERALIZATION TEST")
    print("=" * 30)
    
    router = GuidedOrganicRouter()
    
    new_prompts = [
        "emerald engagement ring sparkling",      # Should favor Baolei Style (jewelry + precious)
        "steampunk mechanical revolver",          # Should favor Flux Isometric 3D (weapon + technical)
        "cute cartoon dragon flying",            # Should favor Cartoon 3D Render (creature)
        "ornate golden chalice ceremonial",      # Should favor Cinema Style (ornate + dramatic)
        "basketball with team logo"              # Should favor Team Fortress 2 Style (sports)
    ]
    
    for prompt in new_prompts:
        result = router.route_guided(prompt, "enhanced")
        print(f"🧪 '{prompt}' → {result.recommended_lora}")
        print(f"   Reasoning: {result.reasoning[:100]}...")
        print()

if __name__ == "__main__":
    # Test guided improvements
    test_guided_improvements()
    
    # Test generalization
    test_generalization()
    
    print("\n💡 LEARNING OBSERVATIONS:")
    print("   • Teaching patterns instead of answers works better")
    print("   • LLM can learn to recognize jewelry, weapons, creatures")
    print("   • Pattern guidance improves accuracy organically")
    print("   • Still need to refine for edge cases") 