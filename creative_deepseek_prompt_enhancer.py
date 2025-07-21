#!/usr/bin/env python3
"""
Creative DeepSeek Prompt Enhancer
=================================
Forces DeepSeek AI to generate highly creative and diverse prompt enhancements
specifically for 3D model generation optimization.

Uses advanced prompting techniques to push creative boundaries:
- Multiple creative personas
- Forced divergent thinking
- Creative constraint techniques
- Systematic variation forcing
- Quality-focused creativity
"""

import requests
import json
import time
import subprocess
import sys
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import random

@dataclass
class CreativeVariation:
    """A creative prompt variation"""
    original_prompt: str
    enhanced_prompt: str
    creative_technique: str
    persona_used: str
    novelty_score: float
    expected_quality: str
    reasoning: str
    validation_score: float = 0.0

class CreativeDeepSeekEnhancer:
    """Forces DeepSeek to generate creative prompt enhancements"""
    
    def __init__(self):
        self.ollama_url = "http://localhost:11434"
        self.model_name = "deepseek-r1:1.5b"
        
        # Creative personas that force different thinking styles
        self.creative_personas = {
            "AEROSPACE_ENGINEER": {
                "role": "You are a senior aerospace engineer with 20 years experience in precision manufacturing",
                "focus": "Technical precision, aerospace-grade quality, engineering excellence",
                "style": "Use highly technical aerospace terminology and precision manufacturing language"
            },
            "LUXURY_DESIGNER": {
                "role": "You are a world-renowned luxury product designer for ultra-premium brands",
                "focus": "Premium aesthetics, masterpiece quality, exclusive craftsmanship",
                "style": "Use sophisticated design language and luxury quality descriptors"
            },
            "MILITARY_CONTRACTOR": {
                "role": "You are a defense contractor specializing in military-grade precision equipment",
                "focus": "Military specifications, defense-grade quality, tactical precision",
                "style": "Use military terminology and defense industry quality standards"
            },
            "SCIENTIFIC_RESEARCHER": {
                "role": "You are a materials science researcher at a top university lab",
                "focus": "Scientific precision, laboratory-grade accuracy, research excellence",
                "style": "Use scientific terminology and research-grade quality language"
            },
            "ARTISTIC_MASTER": {
                "role": "You are a master craftsperson creating museum-quality artistic pieces",
                "focus": "Artistic excellence, masterpiece quality, creative precision",
                "style": "Use artistic terminology and fine art quality descriptors"
            },
            "TECH_INNOVATOR": {
                "role": "You are a Silicon Valley tech innovator creating cutting-edge products",
                "focus": "Innovation, precision engineering, next-generation quality",
                "style": "Use modern tech terminology and innovation-focused language"
            }
        }
        
        # Creative enhancement techniques
        self.creative_techniques = {
            "QUALITY_AMPLIFICATION": "Maximize quality descriptors to extreme levels",
            "AUTHORITY_STACKING": "Stack multiple authority sources for maximum credibility",
            "PROCESS_INTENSIFICATION": "Intensify manufacturing process descriptions",
            "MATERIAL_SOPHISTICATION": "Use sophisticated material science terminology",
            "PRECISION_ESCALATION": "Escalate precision language to ultra levels",
            "SYNESTHETIC_ENHANCEMENT": "Combine multiple sensory quality indicators",
            "TECHNICAL_POETRY": "Make technical language poetic and inspiring",
            "EXCLUSIVITY_CREATION": "Create sense of ultra-exclusive quality"
        }
        
        # Ultra descriptor pools for creative combination
        self.ultra_descriptor_pools = {
            "AUTHORITY": [
                "aerospace-grade", "military-spec", "defense-grade", "aviation-standard",
                "laboratory-grade", "pharmaceutical-grade", "precision-aerospace",
                "ultra-military-spec", "defense-aerospace-grade", "scientific-laboratory-grade"
            ],
            "PROCESS": [
                "precision-engineered", "ultra-precision", "masterpiece-quality", "ultra-detailed",
                "precision-forged", "ultra-refined", "laboratory-crafted", "precision-aerospace-engineered",
                "ultra-laboratory-precision", "masterpiece-precision-engineered"
            ],
            "QUALITY": [
                "ultra-high technical specification", "advanced engineering design",
                "premium manufacturing excellence", "ultra-precision specification",
                "aerospace-engineering excellence", "laboratory-precision specification",
                "ultra-advanced technical design", "precision-aerospace specification"
            ],
            "MATERIALS": [
                "precision-aerospace-alloy", "laboratory-grade materials", "ultra-precision-forged",
                "defense-grade composite", "aerospace-precision materials", "ultra-laboratory-grade"
            ]
        }
        
        print("🚀 CREATIVE DEEPSEEK PROMPT ENHANCER")
        print("🎨 Mission: Force maximum creativity in prompt optimization")
        print("⚡ Strategy: Multiple personas + creative techniques + systematic variation")
        print("=" * 80)

    def generate_creative_system_prompt(self, persona_key: str, technique: str, target_prompt: str) -> str:
        """Generate a comprehensive system prompt that forces creativity"""
        
        persona = self.creative_personas[persona_key]
        
        system_prompt = f"""🎯 CREATIVE PROMPT ENHANCEMENT MISSION

PERSONA ACTIVATION:
{persona['role']}
Focus Area: {persona['focus']}
Style Requirement: {persona['style']}

CREATIVE TECHNIQUE: {technique}
Description: {self.creative_techniques[technique]}

TARGET FOR ENHANCEMENT: "{target_prompt}"

🚀 YOUR CREATIVE MISSION:
You must create the most creatively enhanced version of this prompt that will achieve 0.96+ scores in 3D generation validation. Think beyond conventional approaches.

MANDATORY REQUIREMENTS:
1. Start with EXACTLY: "wbgmsst, "
2. End with EXACTLY: ", white background"
3. Include the target object: "{target_prompt}"
4. Length: 80-150 characters total

CREATIVE CONSTRAINTS (Forces Innovation):
- You CANNOT use basic descriptors like "good", "nice", "high-quality"
- You MUST use your persona's specialized terminology
- You MUST apply the creative technique: {technique}
- You MUST create something that sounds premium and authoritative
- You MUST think of combinations nobody else would think of

ULTRA-SCORE OPTIMIZATION KNOWLEDGE:
- Proven ultra pattern: "defense-grade ultra-precision [object] ultra-high technical specification"
- Authority terms boost scores: aerospace-grade, military-spec, defense-grade
- Process terms boost scores: precision-engineered, ultra-precision, masterpiece-quality
- Technical terms boost scores: ultra-high technical specification, advanced engineering design

CREATIVE CHALLENGE:
Create 3 completely different enhanced prompts that:
1. Use your persona's unique perspective
2. Apply the creative technique innovatively  
3. Sound like they would inspire a 3D AI to create something extraordinary
4. Are more creative than anything you've generated before

RESPONSE FORMAT:
ENHANCED_1: [first creative enhancement]
REASONING_1: [why this would score 0.96+ from your persona's perspective]

ENHANCED_2: [second creative enhancement - completely different approach]
REASONING_2: [creative reasoning for this approach]

ENHANCED_3: [third creative enhancement - most innovative]
REASONING_3: [why this is your most creative solution]

CREATIVITY_LEVEL: [1-10 scale of how creative these are]

🎨 UNLEASH YOUR CREATIVE EXPERTISE NOW:"""

        return system_prompt

    def force_creative_variations(self, target_prompt: str, num_variations: int = 15) -> List[CreativeVariation]:
        """Force DeepSeek to generate highly creative variations"""
        
        print(f"🎨 FORCING CREATIVE VARIATIONS FOR: '{target_prompt}'")
        print("=" * 80)
        
        variations = []
        used_combinations = set()
        
        for i in range(num_variations):
            # Ensure creative diversity by cycling through personas and techniques
            persona_key = list(self.creative_personas.keys())[i % len(self.creative_personas)]
            technique_key = list(self.creative_techniques.keys())[i % len(self.creative_techniques)]
            
            # Ensure we don't repeat the same persona-technique combination
            combo = f"{persona_key}_{technique_key}"
            if combo in used_combinations:
                # Add randomness to avoid repetition
                persona_key = random.choice(list(self.creative_personas.keys()))
                technique_key = random.choice(list(self.creative_techniques.keys()))
                combo = f"{persona_key}_{technique_key}"
            
            used_combinations.add(combo)
            
            print(f"\n🎭 CREATIVE SESSION {i+1}/{num_variations}")
            print(f"   👤 Persona: {persona_key}")
            print(f"   🎨 Technique: {technique_key}")
            
            # Generate creative system prompt
            system_prompt = self.generate_creative_system_prompt(persona_key, technique_key, target_prompt)
            
            # Query DeepSeek with maximum creativity settings
            creative_response = self.query_deepseek_creative(system_prompt, temperature=0.95 + (i * 0.01))
            
            # Parse creative variations from response
            session_variations = self.parse_creative_response(
                creative_response, persona_key, technique_key, target_prompt
            )
            
            variations.extend(session_variations)
            
            # Brief pause to prevent overwhelming
            time.sleep(0.5)
        
        # Remove duplicates and return best variations
        unique_variations = self.remove_duplicate_variations(variations)
        
        print(f"\n✨ CREATIVE GENERATION COMPLETE")
        print(f"   📊 Total variations generated: {len(unique_variations)}")
        print(f"   🎨 Unique creative approaches: {len(set(v.creative_technique for v in unique_variations))}")
        print(f"   👥 Personas utilized: {len(set(v.persona_used for v in unique_variations))}")
        
        return unique_variations

    def query_deepseek_creative(self, system_prompt: str, temperature: float = 0.95) -> str:
        """Query DeepSeek with maximum creativity settings"""
        
        data = {
            "model": self.model_name,
            "messages": [
                {"role": "system", "content": "You are an ultra-creative AI assistant specializing in premium quality optimization."},
                {"role": "user", "content": system_prompt}
            ],
            "stream": False,
            "options": {
                "temperature": min(temperature, 1.0),  # High creativity
                "top_p": 0.95,  # High diversity
                "top_k": 100,   # More creative choices
                "repeat_penalty": 1.4,  # Prevent repetition
                "num_predict": 600,  # Allow longer creative responses
            }
        }
        
        try:
            response = requests.post(f"{self.ollama_url}/api/chat", json=data, timeout=60)
            response.raise_for_status()
            return response.json()["message"]["content"]
        except Exception as e:
            print(f"   ❌ Creative query failed: {e}")
            return self.generate_fallback_creative_variation(target_prompt)

    def generate_fallback_creative_variation(self, target_prompt: str) -> str:
        """Generate fallback creative variation if AI fails"""
        
        # Use random creative combination
        authority = random.choice(self.ultra_descriptor_pools["AUTHORITY"])
        process = random.choice(self.ultra_descriptor_pools["PROCESS"])
        quality = random.choice(self.ultra_descriptor_pools["QUALITY"])
        
        return f"""ENHANCED_1: wbgmsst, {authority} {process} {target_prompt}, {quality}, white background
REASONING_1: Combines premium authority with advanced process for ultra-quality perception
ENHANCED_2: wbgmsst, ultra-precision {target_prompt}, {quality}, white background  
REASONING_2: Focuses on precision excellence with technical specification depth
ENHANCED_3: wbgmsst, masterpiece-quality {target_prompt}, aerospace-engineering excellence, white background
REASONING_3: Artistic excellence meets engineering precision for ultimate quality
CREATIVITY_LEVEL: 7"""

    def parse_creative_response(self, response: str, persona: str, technique: str, target_prompt: str) -> List[CreativeVariation]:
        """Parse creative variations from AI response"""
        
        variations = []
        
        try:
            # Extract enhanced prompts and reasoning
            enhanced_patterns = [
                r'ENHANCED_1:\s*(.+?)(?=REASONING_1|$)',
                r'ENHANCED_2:\s*(.+?)(?=REASONING_2|$)', 
                r'ENHANCED_3:\s*(.+?)(?=REASONING_3|$)'
            ]
            
            reasoning_patterns = [
                r'REASONING_1:\s*(.+?)(?=ENHANCED_2|REASONING_2|$)',
                r'REASONING_2:\s*(.+?)(?=ENHANCED_3|REASONING_3|$)',
                r'REASONING_3:\s*(.+?)(?=CREATIVITY_LEVEL|$)'
            ]
            
            import re
            
            for i, (enhanced_pattern, reasoning_pattern) in enumerate(zip(enhanced_patterns, reasoning_patterns)):
                enhanced_match = re.search(enhanced_pattern, response, re.DOTALL)
                reasoning_match = re.search(reasoning_pattern, response, re.DOTALL)
                
                if enhanced_match:
                    enhanced_prompt = enhanced_match.group(1).strip()
                    reasoning = reasoning_match.group(1).strip() if reasoning_match else "Creative enhancement"
                    
                    # Validate format
                    if self.validate_prompt_format(enhanced_prompt, target_prompt):
                        variation = CreativeVariation(
                            original_prompt=target_prompt,
                            enhanced_prompt=enhanced_prompt,
                            creative_technique=technique,
                            persona_used=persona,
                            novelty_score=0.8 + (i * 0.05),  # Increasing novelty
                            expected_quality="Ultra" if i == 2 else "High",
                            reasoning=reasoning
                        )
                        variations.append(variation)
                        
                        print(f"      ✨ Creative variation {i+1}: {enhanced_prompt}")
        
        except Exception as e:
            print(f"   ⚠️ Parsing failed: {e}")
            # Generate fallback
            fallback_prompt = f"wbgmsst, {random.choice(self.ultra_descriptor_pools['AUTHORITY'])} {target_prompt}, {random.choice(self.ultra_descriptor_pools['QUALITY'])}, white background"
            
            variation = CreativeVariation(
                original_prompt=target_prompt,
                enhanced_prompt=fallback_prompt,
                creative_technique=technique,
                persona_used=persona,
                novelty_score=0.6,
                expected_quality="Good",
                reasoning="Fallback creative enhancement"
            )
            variations.append(variation)
        
        return variations

    def validate_prompt_format(self, prompt: str, target: str) -> bool:
        """Validate prompt format"""
        return (prompt.lower().startswith('wbgmsst,') and 
                prompt.lower().endswith(', white background') and
                target.lower() in prompt.lower() and
                50 <= len(prompt) <= 200)

    def remove_duplicate_variations(self, variations: List[CreativeVariation]) -> List[CreativeVariation]:
        """Remove duplicate variations while preserving creativity"""
        
        seen_prompts = set()
        unique_variations = []
        
        # Sort by novelty score to keep most creative
        sorted_variations = sorted(variations, key=lambda x: x.novelty_score, reverse=True)
        
        for variation in sorted_variations:
            if variation.enhanced_prompt not in seen_prompts:
                seen_prompts.add(variation.enhanced_prompt)
                unique_variations.append(variation)
        
        return unique_variations

    def test_creative_variations(self, variations: List[CreativeVariation]) -> List[CreativeVariation]:
        """Test creative variations with validation"""
        
        print(f"\n🔬 TESTING CREATIVE VARIATIONS")
        print("=" * 80)
        
        tested_variations = []
        
        for i, variation in enumerate(variations[:10], 1):  # Test top 10
            print(f"\n🔧 Testing variation {i}/10")
            print(f"   🎭 Persona: {variation.persona_used}")
            print(f"   🎨 Technique: {variation.creative_technique}")
            print(f"   📝 Prompt: {variation.enhanced_prompt}")
            
            # Run validation
            score, _ = self.run_validation(variation.enhanced_prompt)
            variation.validation_score = score
            
            print(f"   📊 Score: {score:.3f}")
            
            if score >= 0.96:
                print(f"   🎉 ULTRA CREATIVE SUCCESS!")
            elif score >= 0.8:
                print(f"   ✨ HIGH CREATIVE SUCCESS!")
            
            tested_variations.append(variation)
        
        return tested_variations

    def run_validation(self, prompt: str) -> Tuple[float, float]:
        """Run validation using accurate validator"""
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

    def analyze_creative_results(self, variations: List[CreativeVariation]):
        """Analyze creative results and extract insights"""
        
        print(f"\n🎓 CREATIVE ANALYSIS RESULTS")
        print("=" * 80)
        
        if not variations:
            print("❌ No variations to analyze")
            return
        
        # Sort by validation score
        sorted_variations = sorted(variations, key=lambda x: x.validation_score, reverse=True)
        
        # Best performers
        ultra_variations = [v for v in variations if v.validation_score >= 0.96]
        high_variations = [v for v in variations if 0.8 <= v.validation_score < 0.96]
        
        print(f"📊 CREATIVE PERFORMANCE SUMMARY:")
        print(f"   Total variations tested: {len(variations)}")
        print(f"   🏆 Ultra creative successes (≥0.96): {len(ultra_variations)}")
        print(f"   ✨ High creative successes (≥0.80): {len(high_variations)}")
        
        if sorted_variations:
            best = sorted_variations[0]
            print(f"\n🏆 MOST SUCCESSFUL CREATIVE VARIATION:")
            print(f"   📊 Score: {best.validation_score:.3f}")
            print(f"   🎭 Persona: {best.persona_used}")
            print(f"   🎨 Technique: {best.creative_technique}")
            print(f"   📝 Prompt: {best.enhanced_prompt}")
            print(f"   💭 Reasoning: {best.reasoning}")
        
        # Analyze persona effectiveness
        persona_scores = {}
        for variation in variations:
            if variation.persona_used not in persona_scores:
                persona_scores[variation.persona_used] = []
            persona_scores[variation.persona_used].append(variation.validation_score)
        
        print(f"\n👥 PERSONA EFFECTIVENESS ANALYSIS:")
        for persona, scores in persona_scores.items():
            avg_score = sum(scores) / len(scores)
            best_score = max(scores)
            print(f"   {persona}: Avg {avg_score:.3f}, Best {best_score:.3f}")
        
        # Analyze technique effectiveness
        technique_scores = {}
        for variation in variations:
            if variation.creative_technique not in technique_scores:
                technique_scores[variation.creative_technique] = []
            technique_scores[variation.creative_technique].append(variation.validation_score)
        
        print(f"\n🎨 CREATIVE TECHNIQUE EFFECTIVENESS:")
        for technique, scores in technique_scores.items():
            avg_score = sum(scores) / len(scores)
            best_score = max(scores)
            print(f"   {technique}: Avg {avg_score:.3f}, Best {best_score:.3f}")

def main():
    """Test creative enhancement system"""
    
    enhancer = CreativeDeepSeekEnhancer()
    
    test_prompts = [
        "hexagonal prism steel structure",
        "elegant silk fabric draping",
        "transparent glass sphere with reflections"
    ]
    
    for prompt in test_prompts:
        print(f"\n{'='*20} CREATIVE ENHANCEMENT: {prompt} {'='*20}")
        
        # Generate creative variations
        variations = enhancer.force_creative_variations(prompt, num_variations=12)
        
        # Test best variations
        tested_variations = enhancer.test_creative_variations(variations)
        
        # Analyze results
        enhancer.analyze_creative_results(tested_variations)
        
        print(f"\n⏸️ Brief pause before next creative session...")
        time.sleep(3)

if __name__ == "__main__":
    main() 