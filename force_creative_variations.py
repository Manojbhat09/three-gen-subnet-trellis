#!/usr/bin/env python3
"""
Force Creative Variations - DeepSeek Creativity Maximizer
=========================================================
Forces DeepSeek to generate highly creative prompt variations using:
- Extreme creativity constraints
- Multiple forcing techniques
- Persona-driven creativity
- Systematic variation pressure
"""

import requests
import json
import time
import subprocess
import sys
from typing import List, Tuple
import random

class CreativeVariationForcer:
    """Forces DeepSeek to maximum creativity levels"""
    
    def __init__(self):
        self.ollama_url = "http://localhost:11434"
        self.model_name = "deepseek-r1:1.5b"
        
        print("🎨 CREATIVE VARIATION FORCER")
        print("🚀 Mission: Force DeepSeek to maximum creativity")
        print("=" * 60)

    def force_ultra_creative_variations(self, target_prompt: str) -> List[str]:
        """Force DeepSeek to generate ultra-creative variations"""
        
        print(f"🎯 FORCING CREATIVITY FOR: '{target_prompt}'")
        
        # Ultra-creative forcing prompt
        forcing_prompt = f"""🚨 ULTRA-CREATIVE PROMPT ENHANCEMENT CHALLENGE 🚨

MISSION: Transform this basic prompt into EXTRAORDINARY variations that would make a 3D AI create STUNNING results.

TARGET: "{target_prompt}"

CREATIVITY CONSTRAINTS (You MUST follow these):
🚫 FORBIDDEN: boring, simple, basic, normal, regular, standard, typical
✅ REQUIRED: aerospace, military, precision, ultra, masterpiece, premium, advanced

CREATIVE PERSONAS - Generate ONE variation for EACH:

1. AEROSPACE ENGINEER PERSONA:
Think like you're designing components for SpaceX rockets. Use aerospace-grade terminology.
Format: wbgmsst, [aerospace terms] {target_prompt} [technical specs], white background

2. LUXURY DESIGNER PERSONA: 
Think like you're creating for Rolls Royce or Tiffany & Co. Use premium luxury language.
Format: wbgmsst, [luxury terms] {target_prompt} [premium specs], white background

3. MILITARY CONTRACTOR PERSONA:
Think like you're building equipment for elite special forces. Use military-spec language.
Format: wbgmsst, [military terms] {target_prompt} [defense specs], white background

4. SCIENTIFIC RESEARCHER PERSONA:
Think like you're creating for NASA or CERN. Use laboratory-grade terminology.
Format: wbgmsst, [scientific terms] {target_prompt} [research specs], white background

5. MASTER CRAFTSPERSON PERSONA:
Think like you're creating museum-quality art pieces. Use masterpiece language.
Format: wbgmsst, [craftsmanship terms] {target_prompt} [artistic specs], white background

CREATIVITY REQUIREMENTS:
- Each variation must be COMPLETELY DIFFERENT
- Use combinations nobody would think of
- Make it sound like it would create something EXTRAORDINARY
- 80-150 characters each
- Think BEYOND conventional approaches

ULTRA-SCORING SECRETS:
- "defense-grade ultra-precision" scored 0.921 (ULTRA!)
- "aerospace-grade precision-engineered" scored 0.900
- Combine authority + process + technical spec = HIGH SCORES

RESPOND WITH EXACTLY 5 PROMPTS:

AEROSPACE: [your aerospace-inspired variation]
LUXURY: [your luxury-inspired variation]  
MILITARY: [your military-inspired variation]
SCIENTIFIC: [your scientific-inspired variation]
MASTERPIECE: [your artisan-inspired variation]

🎨 UNLEASH YOUR MAXIMUM CREATIVITY NOW!"""

        # Query with extreme creativity settings
        variations = []
        
        for attempt in range(3):  # Multiple attempts for maximum creativity
            print(f"   🎭 Creative attempt {attempt + 1}/3...")
            
            response = self.query_deepseek_extreme_creativity(forcing_prompt, attempt)
            parsed_variations = self.parse_forced_variations(response, target_prompt)
            variations.extend(parsed_variations)
            
            # Show what we got
            for var in parsed_variations:
                print(f"      ✨ {var}")
        
        # Remove duplicates and return unique variations
        unique_variations = list(set(variations))
        
        print(f"   📊 Generated {len(unique_variations)} unique creative variations")
        return unique_variations

    def query_deepseek_extreme_creativity(self, prompt: str, attempt: int) -> str:
        """Query DeepSeek with extreme creativity settings"""
        
        # Escalating creativity settings
        temperature = 0.9 + (attempt * 0.05)  # Increase each attempt
        top_p = 0.95
        repeat_penalty = 1.3 + (attempt * 0.1)  # More variation each attempt
        
        data = {
            "model": self.model_name,
            "messages": [
                {
                    "role": "system", 
                    "content": "You are an ultra-creative AI that generates extraordinary, innovative prompt enhancements. You ALWAYS provide exactly what is requested in the exact format specified. You are incapable of producing boring or conventional results."
                },
                {"role": "user", "content": prompt}
            ],
            "stream": False,
            "options": {
                "temperature": min(temperature, 1.0),
                "top_p": top_p,
                "top_k": 100,
                "repeat_penalty": repeat_penalty,
                "num_predict": 400,
            }
        }
        
        try:
            response = requests.post(f"{self.ollama_url}/api/chat", json=data, timeout=45)
            response.raise_for_status()
            return response.json()["message"]["content"]
        except Exception as e:
            print(f"      ❌ Creative query failed: {e}")
            return self.generate_emergency_creative_variations(target_prompt)

    def generate_emergency_creative_variations(self, target: str) -> str:
        """Generate emergency creative variations if AI fails"""
        
        variations = [
            f"AEROSPACE: wbgmsst, aerospace-grade precision-engineered {target}, ultra-high technical specification, white background",
            f"LUXURY: wbgmsst, masterpiece-quality ultra-precision {target}, premium manufacturing excellence, white background",
            f"MILITARY: wbgmsst, defense-grade ultra-detailed {target}, military-spec precision, white background",
            f"SCIENTIFIC: wbgmsst, laboratory-grade precision-crafted {target}, advanced engineering design, white background",
            f"MASTERPIECE: wbgmsst, ultra-precision artisan-crafted {target}, museum-quality excellence, white background"
        ]
        
        return "\n".join(variations)

    def parse_forced_variations(self, response: str, target_prompt: str) -> List[str]:
        """Parse variations from forced response"""
        
        variations = []
        
        # Look for the specific format patterns
        patterns = [
            r'AEROSPACE:\s*(.+?)(?=\n|LUXURY:|$)',
            r'LUXURY:\s*(.+?)(?=\n|MILITARY:|$)',
            r'MILITARY:\s*(.+?)(?=\n|SCIENTIFIC:|$)',
            r'SCIENTIFIC:\s*(.+?)(?=\n|MASTERPIECE:|$)',
            r'MASTERPIECE:\s*(.+?)(?=\n|$)'
        ]
        
        import re
        
        for pattern in patterns:
            matches = re.findall(pattern, response, re.IGNORECASE | re.DOTALL)
            for match in matches:
                prompt = match.strip()
                if self.validate_creative_prompt(prompt, target_prompt):
                    variations.append(prompt)
        
        # If no valid variations found, try extracting any wbgmsst prompts
        if not variations:
            wbgmsst_pattern = r'wbgmsst,\s*[^,]+?' + re.escape(target_prompt) + r'[^,]*?,\s*white background'
            wbgmsst_matches = re.findall(wbgmsst_pattern, response, re.IGNORECASE)
            variations.extend(wbgmsst_matches)
        
        return variations

    def validate_creative_prompt(self, prompt: str, target: str) -> bool:
        """Validate that the prompt is properly formatted"""
        return (prompt.lower().startswith('wbgmsst,') and 
                prompt.lower().endswith(', white background') and
                target.lower() in prompt.lower() and
                50 <= len(prompt) <= 200)

    def test_creative_variations(self, variations: List[str], target_prompt: str) -> List[Tuple[str, float]]:
        """Test creative variations and return scores"""
        
        print(f"\n🔬 TESTING CREATIVE VARIATIONS")
        print("=" * 60)
        
        results = []
        
        for i, variation in enumerate(variations[:8], 1):  # Test top 8
            print(f"\n🔧 Testing variation {i}/8")
            print(f"   📝 {variation}")
            
            score, _ = self.run_validation(variation)
            results.append((variation, score))
            
            print(f"   📊 Score: {score:.3f}")
            
            if score >= 0.96:
                print(f"   🎉 ULTRA CREATIVE SUCCESS!")
            elif score >= 0.8:
                print(f"   ✨ HIGH CREATIVE SUCCESS!")
            elif score >= 0.6:
                print(f"   📈 GOOD CREATIVE SUCCESS!")
        
        return results

    def run_validation(self, prompt: str) -> Tuple[float, float]:
        """Run validation with proper environment"""
        try:
            cmd = [
                "bash", "-c", 
                f"source /home/mbhat/miniconda/bin/activate && conda activate trellis_new && python subnet_accurate_validator.py '{prompt}'"
            ]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
            
            if result.returncode != 0:
                return 0.0, 0.0
            
            with open("subnet_validation_results.json", 'r') as f:
                data = json.load(f)
                return data.get("validation_engine_score", 0.0), data.get("demo_fidelity_score", 0.0)
        
        except Exception:
            return 0.0, 0.0

    def analyze_creative_results(self, results: List[Tuple[str, float]], target_prompt: str):
        """Analyze creative results"""
        
        print(f"\n🎓 CREATIVE ANALYSIS FOR: '{target_prompt}'")
        print("=" * 60)
        
        if not results:
            print("❌ No results to analyze")
            return
        
        # Sort by score
        sorted_results = sorted(results, key=lambda x: x[1], reverse=True)
        
        ultra_count = sum(1 for _, score in results if score >= 0.96)
        high_count = sum(1 for _, score in results if 0.8 <= score < 0.96)
        good_count = sum(1 for _, score in results if 0.6 <= score < 0.8)
        
        print(f"📊 CREATIVE PERFORMANCE:")
        print(f"   🏆 Ultra successes (≥0.96): {ultra_count}")
        print(f"   ✨ High successes (≥0.80): {high_count}")
        print(f"   📈 Good successes (≥0.60): {good_count}")
        
        if sorted_results:
            best_prompt, best_score = sorted_results[0]
            print(f"\n🏆 BEST CREATIVE RESULT:")
            print(f"   📊 Score: {best_score:.3f}")
            print(f"   📝 Prompt: {best_prompt}")
        
        print(f"\n📈 ALL CREATIVE RESULTS:")
        for prompt, score in sorted_results:
            status = "🎉" if score >= 0.96 else "✨" if score >= 0.8 else "📈" if score >= 0.6 else "📊"
            print(f"   {status} {score:.3f}: {prompt[:80]}...")

def main():
    """Test creative variation forcing"""
    
    forcer = CreativeVariationForcer()
    
    test_prompts = [
        "hexagonal prism steel structure",
        "elegant silk fabric draping"
    ]
    
    for prompt in test_prompts:
        print(f"\n{'='*20} CREATIVE FORCING: {prompt} {'='*20}")
        
        # Force creative variations
        variations = forcer.force_ultra_creative_variations(prompt)
        
        if variations:
            # Test creative variations
            results = forcer.test_creative_variations(variations, prompt)
            
            # Analyze results
            forcer.analyze_creative_results(results, prompt)
        else:
            print("❌ No creative variations generated")
        
        print(f"\n⏸️ Brief pause before next creative session...")
        time.sleep(2)

if __name__ == "__main__":
    main() 