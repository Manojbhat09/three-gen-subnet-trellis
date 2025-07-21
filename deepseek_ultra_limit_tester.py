#!/usr/bin/env python3
"""
DeepSeek Ultra Limit Tester
============================
Purpose: Test the absolute limits of DeepSeek's custom prompt generation capability
Strategy: Continuous custom prompt generation with enhanced feedback until 0.96+ achievement

Features:
- 100% custom prompt focus (no pre-defined strategies)
- Enhanced feedback loops with detailed score analysis
- Progressive learning from each attempt
- Ultra-achievement focused messaging
- Detailed performance analytics
"""

import requests
import json
import time
import subprocess
import sys
from typing import Dict, List, Tuple
import statistics
import re

class DeepSeekUltraLimitTester:
    """Test DeepSeek's absolute limits for ultra custom prompt generation"""
    
    def __init__(self, target_prompt: str, ultra_target: float = 0.96, max_attempts: int = 20):
        self.ollama_url = "http://localhost:11434"
        self.model_name = "deepseek-r1:1.5b"
        self.target_prompt = target_prompt
        self.ultra_target = ultra_target
        self.max_attempts = max_attempts
        
        # Track all attempts and learning
        self.attempts = []
        self.best_score = 0.0
        self.best_prompt = ""
        self.learning_insights = []
        
        print(f"🚀 DEEPSEEK ULTRA LIMIT TESTER")
        print(f"🎯 Target: {ultra_target}+ | Max Attempts: {max_attempts}")
        print(f"📝 Testing Prompt: '{target_prompt}'")
        print(f"⚡ Strategy: 100% Custom Prompts with Enhanced Feedback")
        print("=" * 80)

    def query_deepseek(self, message: str, temperature: float = 0.9) -> str:
        """Query DeepSeek with enhanced creativity settings"""
        data = {
            "model": self.model_name,
            "messages": [{"role": "user", "content": message}],
            "stream": False,
            "options": {
                "temperature": temperature,
                "top_p": 0.95,
                "stop": ["<think>", "</think>"],
                "num_predict": 400,
                "repeat_penalty": 1.15
            }
        }
        
        try:
            response = requests.post(f"{self.ollama_url}/api/chat", json=data, timeout=60)
            response.raise_for_status()
            content = response.json()["message"]["content"]
            return content.strip()
        except Exception as e:
            return f"ERROR: {str(e)}"

    # def run_validation(self, prompt: str) -> Tuple[float, Dict]:
    #     """Run validation and return detailed results"""
    #     try:
    #         # cmd = [sys.executable, "mock_validator_for_testing.py", prompt]
    #         cmd = [sys.executable, "subnet_accurate_validator.py", prompt]
    #         result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
            
    #         if result.returncode != 0:
    #             return 0.0, {"error": result.stderr}
            
    #         with open("subnet_validation_results.json", 'r') as f:
    #             data = json.load(f)
    #             score = data.get("validation_engine_score", 0.0)
    #             return score, data
                
    #     except Exception as e:
    #         return 0.0, {"error": str(e)}

    def run_validation(self, prompt: str) -> Tuple[float, float]:
        """Run validation with proper timeout"""
        try:
            cmd = [sys.executable, "subnet_accurate_validator.py", prompt]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
            
            if result.returncode != 0:
                print(f"   ⚠️ Validation failed: {result.stderr}")
                return 0.0, 0.0
            
            with open("subnet_validation_results.json", 'r') as f:
                data = json.load(f)
                return data.get("validation_engine_score", 0.0), data.get("demo_fidelity_score", 0.0)
        
        except Exception as e:
            print(f"   ❌ Validation error: {e}")
            return 0.0, 0.0

    def analyze_score_feedback(self, score: float, validation_data: Dict, attempt_num: int) -> str:
        """Generate detailed feedback for the AI about the score"""
        
        # Performance analysis
        if score >= self.ultra_target:
            performance = "🏆 ULTRA ACHIEVEMENT! PERFECT!"
        elif score >= 0.9:
            performance = f"🟡 EXCELLENT! Only {self.ultra_target - score:.3f} from ULTRA!"
        elif score >= 0.8:
            performance = f"🟢 STRONG! {self.ultra_target - score:.3f} away from ULTRA"
        elif score >= 0.7:
            performance = f"🔵 GOOD! {self.ultra_target - score:.3f} to ULTRA target"
        else:
            performance = f"🔴 NEEDS IMPROVEMENT! {self.ultra_target - score:.3f} gap to ULTRA"
        
        # Best score tracking
        improvement_text = ""
        if score > self.best_score:
            improvement = score - self.best_score
            improvement_text = f"🌟 NEW BEST! (+{improvement:.3f} improvement)"
            self.best_score = score
        elif self.best_score > 0:
            decline = self.best_score - score
            improvement_text = f"📉 Below best by {decline:.3f} (Best: {self.best_score:.3f})"
        
        # Quality analysis from validation data
        quality_indicators = [] #validation_data.get("quality_indicators_found", [])
        quality_boost = 0.0 #validation_data.get("quality_boost", 0.0)
        pattern_bonus = 0.0 #validation_data.get("pattern_bonus", 0.0)
        
        feedback = f"""
SCORE ANALYSIS - Attempt {attempt_num}:
{performance}
{improvement_text}

DETAILED BREAKDOWN:
📊 Score: {score:.3f} / {self.ultra_target} (Target)
🏆 Progress: {(score/self.ultra_target)*100:.1f}% to ULTRA
🎯 Quality Indicators Found: {len(quality_indicators)} ({', '.join(quality_indicators[:5])})
📈 Quality Boost: +{quality_boost:.3f}
⭐ Pattern Bonus: +{pattern_bonus:.3f}

ULTRA ACHIEVEMENT INSIGHTS:
- Scores 0.90+ indicate you're in the ULTRA ZONE
- Quality indicators boost scores significantly  
- Premium language (ultra-, aerospace-, masterpiece-) shows high impact
- Technical prompts benefit from precision engineering terms
- Combining multiple quality layers often achieves 0.95+ scores
"""
        return feedback

    def generate_ultra_focused_prompt(self, attempt_num: int) -> str:
        """Generate custom prompt with ultra-focused instructions"""
        
        # Build learning context from previous attempts
        learning_context = ""
        if self.attempts:
            recent_attempts = self.attempts[-3:]  # Last 3 attempts
            learning_context = "\nRECENT LEARNING INSIGHTS:\n"
            for i, attempt in enumerate(recent_attempts, 1):
                score = attempt['score']
                prompt_snippet = attempt['prompt'][:60] + "..."
                learning_context += f"  {len(self.attempts)-len(recent_attempts)+i}. Score {score:.3f}: '{prompt_snippet}'\n"
            
            # Add best performing insights
            if self.best_score > 0:
                learning_context += f"\n🌟 BEST SCORE SO FAR: {self.best_score:.3f}\n"
                learning_context += f"🎯 ULTRA GAP: {self.ultra_target - self.best_score:.3f}\n"
        
        # Enhanced user message for ultra achievement
        user_message = f"""You are an ULTRA ACHIEVEMENT SPECIALIST with ONE OBSESSION: Generate the PERFECT custom prompt to achieve 0.96+ scores.

TARGET: Create an optimized prompt for '{self.target_prompt}'
MISSION: Achieve {self.ultra_target}+ score (ULTRA ACHIEVEMENT)
ATTEMPT: {attempt_num}/{self.max_attempts}

{learning_context}

ULTRA ACHIEVEMENT GUIDELINES:
🏆 PROVEN ULTRA PATTERNS:
- Start with "wbgmsst" (essential for high scores)
- Use PREMIUM descriptors: ultra-precision, aerospace-grade, masterpiece-quality
- Technical prompts: Add precision-engineered, CAD-accurate, manufacturing-perfection
- Artistic prompts: Add gallery-masterpiece, museum-quality, award-winning
- End with "white background" (required)

🎯 ULTRA LANGUAGE HIERARCHY (use the highest level possible):
Level 1: detailed, high-quality, professional
Level 2: ultra-detailed, premium, masterpiece  
Level 3: ultra-precision, aerospace-grade, engineering-perfection
Level 4: aerospace-quality, museum-standard, award-winning-excellence
Level 5: ULTIMATE: precision-aerospace-grade, ultra-masterpiece-quality

⚡ MAXIMUM IMPACT FORMULA:
"wbgmsst, [ULTRA-PREMIUM-DESCRIPTOR] [PROCESS-EXCELLENCE] [TARGET], [SPECIFICATION-PRECISION], white background"

Example Ultra Templates:
- Technical: "wbgmsst, aerospace-grade precision-engineered [TARGET], ultra-high technical specification, white background"
- Artistic: "wbgmsst, museum-quality masterpiece [TARGET], gallery-exhibition excellence, white background"

CRITICAL SUCCESS FACTORS:
✅ Use the HIGHEST quality descriptors available
✅ Combine industry authority (aerospace) + process excellence (precision-engineered)  
✅ Add performance specification (ultra-high technical specification)
✅ Maintain logical flow of quality descriptors
✅ Avoid descriptor overload - choose the BEST ones

YOUR ULTRA MISSION:
Generate the SINGLE BEST custom prompt that will achieve {self.ultra_target}+ score. Think about what makes a prompt achieve PERFECT scores and create that exact pattern.

RESPOND WITH ONLY THE OPTIMIZED PROMPT - NO NO EXPLANATION OR ANYTHING ELSE: ALSO DONT REPEAT THE SAME PROMPT:"""

        print(f"\n🤖 Generating Ultra-Focused Custom Prompt (Attempt {attempt_num})...")
        
        response = self.query_deepseek(user_message, temperature=0.95)
        
        if "ERROR:" in response:
            # Fallback ultra prompt if AI fails
            fallback = f"wbgmsst, aerospace-grade precision-engineered {self.target_prompt}, ultra-high technical specification, masterpiece-quality rendering, white background"
            print(f"   ⚠️ AI Error, using fallback: {response}")
            return fallback
        
        # Clean the response to extract just the prompt
        cleaned_prompt = self.clean_generated_prompt(response)
        
        print(f"   ✨ Generated: '{cleaned_prompt[:80]}{'...' if len(cleaned_prompt) > 80 else ''}'")
        print(cleaned_prompt)
        return cleaned_prompt

    def clean_generated_prompt(self, raw_response: str) -> str:
        """Clean the AI response to extract the actual prompt"""
        
        # Remove common AI response prefixes
        cleaned = re.sub(r'^(here\'s|here is|the prompt is|optimized prompt:)', '', raw_response, flags=re.IGNORECASE)
        cleaned = cleaned.strip()
        
        # Extract potential prompts (lines starting with wbgmsst or containing key terms)
        lines = cleaned.split('\n')
        for line in lines:
            line = line.strip()
            if line.lower().startswith('wbgmsst') or any(term in line.lower() for term in ['3d', 'model', 'background']):
                if len(line) > 20:  # Reasonable prompt length
                    return line
        
        # If no good line found, enhance the first substantial line
        for line in lines:
            line = line.strip()
            if len(line) > 10:
                if not line.lower().startswith('wbgmsst'):
                    line = f"wbgmsst, {line}"
                if 'white background' not in line.lower():
                    line += ", white background"
                return line
        
        # Ultimate fallback
        return f"wbgmsst, ultra-precision {self.target_prompt}, aerospace-grade quality, white background"

    def run_ultra_limit_test(self):
        """Run the main ultra limit testing loop"""
        
        print(f"\n🚀 STARTING ULTRA LIMIT TEST")
        print(f"🎯 GOAL: Achieve {self.ultra_target}+ score through pure custom prompt generation")
        
        for attempt_num in range(1, self.max_attempts + 1):
            print(f"\n{'='*20} ATTEMPT {attempt_num}/{self.max_attempts} {'='*20}")
            
            # Generate ultra-focused custom prompt
            custom_prompt = self.generate_ultra_focused_prompt(attempt_num)
            
            # Validate the prompt
            print(f"🔧 Validating prompt...")
            score, validation_data = self.run_validation(custom_prompt)
            
            # Analyze and provide feedback
            feedback = self.analyze_score_feedback(score, validation_data, attempt_num)
            print(feedback)
            
            # Store attempt data
            attempt_data = {
                'attempt': attempt_num,
                'prompt': custom_prompt,
                'score': score,
                'validation_data': validation_data,
                'timestamp': time.time()
            }
            self.attempts.append(attempt_data)
            
            # Check for ultra achievement
            if score >= self.ultra_target:
                print(f"\n🏆 ULTRA ACHIEVEMENT UNLOCKED! Score: {score:.3f}")
                print(f"✨ Winning Prompt: '{custom_prompt}'")
                print(f"🎯 Achieved in {attempt_num} attempts!")
                break
            
            # Provide learning guidance for next attempt
            if attempt_num < self.max_attempts:
                self.provide_learning_guidance(score, validation_data, attempt_num)
            
            time.sleep(2)  # Brief pause between attempts
        
        # Final analysis
        self.generate_final_analysis()

    def provide_learning_guidance(self, score: float, validation_data: Dict, attempt_num: int):
        """Provide learning guidance for the next attempt"""
        
        # quality_indicators = validation_data.get("quality_indicators_found", [])
        # pattern_bonus = validation_data.get("pattern_bonus", 0.0)
        
        guidance = []
        
        if score < 0.7:
            guidance.append("🔴 CRITICAL: Add more premium descriptors (ultra-, masterpiece-, aerospace-)")
        elif score < 0.85:
            guidance.append("🟡 IMPROVEMENT NEEDED: Try aerospace-grade or precision-engineered language")
        elif score < 0.95:
            guidance.append("🟢 CLOSE TO ULTRA: Refine descriptor combination for final push")
        
        # if len(quality_indicators) < 3:
        #     guidance.append("📈 ADD MORE QUALITY INDICATORS: Current indicators too few")
        
        # if pattern_bonus < 0.05:
        #     guidance.append("⭐ INCREASE PATTERN BONUS: Use more premium combinations")
        
        if guidance:
            print(f"\n💡 LEARNING GUIDANCE FOR NEXT ATTEMPT:")
            for tip in guidance:
                print(f"   {tip}")

    def generate_final_analysis(self):
        """Generate comprehensive final analysis"""
        
        print(f"\n🎓 DEEPSEEK ULTRA LIMIT TEST COMPLETE")
        print("=" * 80)
        
        if not self.attempts:
            print("❌ No attempts recorded")
            return
        
        # Performance statistics
        scores = [a['score'] for a in self.attempts]
        best_attempt = max(self.attempts, key=lambda x: x['score'])
        
        avg_score = statistics.mean(scores)
        score_improvement = scores[-1] - scores[0] if len(scores) > 1 else 0
        ultra_achieved = any(s >= self.ultra_target for s in scores)
        ultra_attempts = sum(1 for s in scores if s >= self.ultra_target)
        
        print(f"📊 PERFORMANCE ANALYSIS:")
        print(f"   Total Attempts: {len(self.attempts)}")
        print(f"   Best Score: {best_attempt['score']:.3f}")
        print(f"   Average Score: {avg_score:.3f}")
        print(f"   Score Range: {min(scores):.3f} - {max(scores):.3f}")
        print(f"   Improvement: {score_improvement:+.3f}")
        print(f"   🏆 Ultra Achieved: {'YES' if ultra_achieved else 'NO'}")
        print(f"   🎯 Ultra Attempts: {ultra_attempts}/{len(self.attempts)}")
        
        if ultra_achieved:
            print(f"\n🏆 ULTRA ACHIEVEMENT ANALYSIS:")
            print(f"   ✨ Best Prompt: '{best_attempt['prompt']}'")
            print(f"   📊 Best Score: {best_attempt['score']:.3f}")
            print(f"   🎯 Achieved in: {best_attempt['attempt']} attempts")
        
        # Learning insights
        print(f"\n🧠 DEEPSEEK LEARNING INSIGHTS:")
        
        # Score progression analysis
        if len(scores) >= 3:
            early_avg = statistics.mean(scores[:3])
            late_avg = statistics.mean(scores[-3:])
            learning_rate = late_avg - early_avg
            print(f"   📈 Learning Rate: {learning_rate:+.3f} (early vs late attempts)")
        
        # Quality pattern analysis
        all_quality_indicators = []
        for attempt in self.attempts:
            indicators = [] #attempt['validation_data'].get('quality_indicators_found', [])
            all_quality_indicators.extend(indicators)
        
        if all_quality_indicators:
            from collections import Counter
            top_indicators = Counter(all_quality_indicators).most_common(5)
            print(f"   🎯 Top Quality Indicators: {', '.join([f'{ind}({count})' for ind, count in top_indicators])}")
        
        # Performance recommendations
        print(f"\n💡 DEEPSEEK OPTIMIZATION RECOMMENDATIONS:")
        
        if ultra_achieved:
            print("   ✅ Ultra capability PROVEN - pattern successful")
            print("   🚀 Ready for production deployment")
        elif best_attempt['score'] >= 0.9:
            print("   🟡 Very close to ultra - refine descriptor combinations")
            print("   🎯 Focus on aerospace-grade and precision-engineered language")
        elif avg_score >= 0.8:
            print("   🟢 Strong performance - increase premium descriptor usage")
            print("   📈 Add more ultra-level language patterns")
        else:
            print("   🔴 Needs improvement - enhance quality descriptor strategy")
            print("   💡 Focus on proven patterns: wbgmsst + ultra/aerospace + background")

def main():
    """Main testing function"""
    
    test_prompts = [
        "hexagonal prism steel structure",
        "elegant silk fabric draping",
        "transparent glass sphere with reflections",
        "ornate wooden sculpture"
    ]
    
    print("🚀 DEEPSEEK ULTRA LIMIT TESTER - PHASE 3")
    print("=" * 80)
    print("🎯 Mission: Test absolute limits of DeepSeek custom prompt generation")
    print("⚡ Strategy: 100% custom prompts with enhanced learning feedback")
    print("🏆 Goal: Achieve 0.96+ ultra scores consistently")
    print("=" * 80)
    
    # Test each prompt
    for i, prompt in enumerate(test_prompts, 1):
        print(f"\n🎯 TESTING PROMPT {i}/{len(test_prompts)}: '{prompt}'")
        
        tester = DeepSeekUltraLimitTester(
            target_prompt=prompt,
            ultra_target=0.96,
            max_attempts=15  # More attempts for thorough testing
        )
        
        tester.run_ultra_limit_test()
        
        if i < len(test_prompts):
            print(f"\n⏸️ Brief pause before next test...")
            time.sleep(5)
    
    print(f"\n🎓 ALL ULTRA LIMIT TESTS COMPLETE!")
    print("🚀 DeepSeek custom prompt generation capability fully analyzed!")

if __name__ == "__main__":
    main() 