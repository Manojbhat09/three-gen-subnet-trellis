#!/usr/bin/env python3
"""
DeepSeek Conversational Optimizer
=================================
Purpose: Create a conversational feedback loop where DeepSeek can see its performance history
and continuously improve prompts through detailed feedback and motivation

Features:
- Conversational history tracking
- Performance-based feedback (rewards/penalties)
- Detailed score analysis with AI coaching
- Progressive learning through conversation
- Motivational system for score improvements
"""

import requests
import json
import time
import subprocess
import sys
from typing import Dict, List, Tuple
import statistics
import re

class DeepSeekConversationalOptimizer:
    """Conversational DeepSeek optimizer with performance tracking and feedback"""
    
    def __init__(self, target_prompt: str, ultra_target: float = 0.96, max_attempts: int = 15):
        self.ollama_url = "http://localhost:11434"
        self.model_name = "deepseek-r1:1.5b"
        self.target_prompt = target_prompt
        self.ultra_target = ultra_target
        self.max_attempts = max_attempts
        
        # Conversation and performance tracking
        self.conversation_history = []
        self.attempts = []
        self.best_score = 0.0
        self.best_prompt = ""
        self.performance_trend = []
        
        print(f"🚀 DEEPSEEK CONVERSATIONAL OPTIMIZER")
        print(f"🎯 Target: {ultra_target}+ | Max Attempts: {max_attempts}")
        print(f"📝 Testing Prompt: '{target_prompt}'")
        print(f"⚡ Strategy: Conversational learning with performance feedback")
        print("=" * 80)
        
        # Initialize conversation with the AI
        self.initialize_conversation()

    def debug_conversation_state(self, attempt_num: int):
        """Debug the current conversation state"""
        print(f"\n🔍 DEBUG: Conversation State Analysis (Attempt {attempt_num})")
        print(f"{'='*60}")
        print(f"Total messages in conversation: {len(self.conversation_history)}")
        
        for i, msg in enumerate(self.conversation_history):
            role = msg['role']
            content = msg['content']
            print(f"\nMessage {i+1} ({role}):")
            print(f"Length: {len(content)} characters")
            print(f"Preview: {content[:150]}{'...' if len(content) > 150 else ''}")
            
            # Look for prompts in assistant messages
            if role == 'assistant':
                lines = content.split('\n')
                prompt_lines = [line for line in lines if 'wbgmsst' in line.lower()]
                if prompt_lines:
                    print(f"🎯 Potential prompts found: {len(prompt_lines)}")
                    for j, prompt_line in enumerate(prompt_lines):
                        print(f"  Prompt {j+1}: {prompt_line.strip()}")
        print(f"{'='*60}")

    def query_deepseek_conversation(self, new_message: str, temperature: float = 0.9) -> str:
        """Query DeepSeek with full conversation history"""
        
        # Add new message to conversation
        self.conversation_history.append({"role": "user", "content": new_message})
        
        # DEBUG: Print the conversation state
        print(f"\n🔍 DEBUG: Conversation Length: {len(self.conversation_history)} messages")
        print(f"🔍 DEBUG: Last User Message Preview: {new_message[:100]}...")
        
        data = {
            "model": self.model_name,
            "messages": self.conversation_history,
            "stream": False,
            "options": {
                "temperature": temperature,
                "top_p": 0.95,
                # Removed stop tokens that were cutting off responses
                "num_predict": 500,
                "repeat_penalty": 1.15
            }
        }
        
        try:
            print(f"🔍 DEBUG: Sending request to DeepSeek...")
            response = requests.post(f"{self.ollama_url}/api/chat", json=data, timeout=60)
            response.raise_for_status()
            content = response.json()["message"]["content"]
            
            # DEBUG: Print AI response details
            print(f"🔍 DEBUG: AI Response Length: {len(content)} characters")
            print(f"🔍 DEBUG: AI Response Preview: {content[:200]}...")
            print(f"🔍 DEBUG: Full AI Response:")
            print(f"{'='*60}")
            print(content)
            print(f"{'='*60}")
            
            # Add AI response to conversation history
            self.conversation_history.append({"role": "assistant", "content": content})
            
            return content.strip()
        except Exception as e:
            print(f"🔍 DEBUG: ERROR in DeepSeek query: {str(e)}")
            return f"ERROR: {str(e)}"

    def initialize_conversation(self):
        """Initialize the conversation with context and goals"""
        
        init_message = f"""🎯 ULTRA PROMPT OPTIMIZATION CHALLENGE

YOU ARE A PROMPT GENERATOR. Your ONLY job is to generate PERFECT prompts.

TARGET: "{self.target_prompt}"
GOAL: Achieve 0.96+ validation score

STRICT FORMAT REQUIREMENTS - FOLLOW EXACTLY:
1. Start with EXACTLY: "wbgmsst, " (with comma and space)
2. End with EXACTLY: ", white background" 
3. Include the target: "{self.target_prompt}"
4. Add premium descriptors: aerospace-grade, ultra-precision, masterpiece-quality
5. Keep it 80-120 characters total

EXAMPLE FORMAT:
"wbgmsst, aerospace-grade ultra-precision {self.target_prompt}, masterpiece-quality rendering, white background"

RESPOND WITH ONLY THE PROMPT - NO EXPLANATION, NO ANALYSIS, NO EXTRA TEXT.

Generate your FIRST prompt now:"""

        response = self.query_deepseek_conversation(init_message)
        print(f"🤖 AI Response:\n{response}")
        
        return response

    def run_validation(self, prompt: str) -> Tuple[float, float]:
        """Run validation with proper timeout"""
        try:
            cmd = [sys.executable, "subnet_accurate_validator.py", prompt]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
            # import pdb; pdb.set_trace()
            if result.returncode != 0:
                print(f"   ⚠️ Validation failed: {result.stderr}")
                return 0.0, 0.0
            
            with open("subnet_validation_results.json", 'r') as f:
                data = json.load(f)
                return data.get("validation_engine_score", 0.0), data.get("demo_fidelity_score", 0.0)
        
        except Exception as e:
            print(f"   ❌ Validation error: {e}")
            return 0.0, 0.0

    def extract_prompt_from_response(self, response: str) -> str:
        """Extract the prompt from AI response with strict validation"""
        
        print(f"\n🔍 DEBUG: Extracting prompt from response...")
        
        # Remove thinking sections first
        clean_response = response
        if '<think>' in clean_response and '</think>' in clean_response:
            # Remove everything between <think> and </think>
            clean_response = re.sub(r'<think>.*?</think>', '', clean_response, flags=re.DOTALL)
        
        print(f"🔍 DEBUG: Cleaned response (no <think>):")
        print(f"{'='*40}")
        print(clean_response)
        print(f"{'='*40}")
        
        # Look for lines that could be prompts
        lines = clean_response.split('\n')
        print(f"🔍 DEBUG: Found {len(lines)} lines to check")
        
        for i, line in enumerate(lines):
            line = line.strip()
            print(f"🔍 DEBUG: Line {i+1}: '{line}'")
            
            # Remove quotes and common prefixes
            cleaned_line = re.sub(r'^["\']|["\']$', '', line)
            cleaned_line = re.sub(r'^(here\'s|here is|prompt:|result:|attempt \d+ prompt:)', '', cleaned_line, flags=re.IGNORECASE).strip()
            
            # Strict validation - must have correct format
            if (cleaned_line.lower().startswith('wbgmsst,') and 
                cleaned_line.lower().endswith(', white background') and 
                self.target_prompt.lower() in cleaned_line.lower() and 
                len(cleaned_line) >= 50):
                print(f"🔍 DEBUG: ✅ Found VALID prompt on line {i+1}: '{cleaned_line}'")
                return cleaned_line
        
        print(f"🔍 DEBUG: ❌ No valid prompt found! Creating corrected version...")
        
        # Try to find any line with the target and fix it
        for i, line in enumerate(lines):
            line = line.strip()
            if self.target_prompt.lower() in line.lower() and len(line) > 20:
                print(f"🔍 DEBUG: Found line with target on {i+1}: '{line}'")
                # Force correct format
                corrected = f"wbgmsst, aerospace-grade precision-engineered {self.target_prompt}, ultra-high technical specification, white background"
                print(f"🔍 DEBUG: ✅ Corrected to: '{corrected}'")
                return corrected
        
        # Ultimate fallback with correct format
        fallback = f"wbgmsst, ultra-precision {self.target_prompt}, aerospace-grade quality, white background"
        print(f"🔍 DEBUG: ⚠️ Using ultimate fallback: '{fallback}'")
        return fallback

    def generate_performance_feedback(self, score: float, attempt_num: int, prompt: str) -> str:
        """Generate detailed performance feedback for the AI"""
        
        # Calculate performance metrics
        improvement = 0.0
        trend_indicator = ""
        
        if len(self.performance_trend) > 0:
            improvement = score - self.performance_trend[-1]
            if improvement > 0:
                trend_indicator = f"📈 IMPROVING (+{improvement:.3f})"
            elif improvement < 0:
                trend_indicator = f"📉 DECLINING ({improvement:.3f})"
            else:
                trend_indicator = "➡️ STABLE (no change)"
        
        self.performance_trend.append(score)
        
        # Motivational feedback based on performance
        if score >= self.ultra_target:
            motivation = "🏆 ULTRA ACHIEVEMENT! PERFECT SCORE! You've mastered the pattern!"
            coaching = "✨ This is the gold standard! Remember this exact approach for future prompts."
        elif score >= 0.90:
            motivation = f"🟢 EXCELLENT! Only {self.ultra_target - score:.3f} points from ULTRA! You're SO close!"
            coaching = "💪 Push just a bit more with premium descriptors to reach ULTRA!"
        elif score >= 0.80:
            motivation = f"🔵 GOOD PROGRESS! {self.ultra_target - score:.3f} points to target. Getting stronger!"
            coaching = "🎯 Try combining aerospace-grade with precision-engineered for more impact."
        elif score >= 0.70:
            motivation = f"🟡 NEEDS IMPROVEMENT. {self.ultra_target - score:.3f} gap to close. Don't give up!"
            coaching = "💡 Focus on premium language: ultra-precision, masterpiece-quality, aerospace-grade."
        else:
            motivation = f"🔴 CRITICAL IMPROVEMENT NEEDED. Big gap of {self.ultra_target - score:.3f} to close."
            coaching = "⚠️ Major revision needed. Use proven pattern: aerospace-grade + precision-engineered."
        
        # Best score tracking
        best_update = ""
        if score > self.best_score:
            self.best_score = score
            self.best_prompt = prompt
            best_update = f"\n🌟 NEW PERSONAL BEST! Previous best: {self.best_score - improvement:.3f}"
        elif self.best_score > 0:
            gap = self.best_score - score
            best_update = f"\n📊 Personal best is still {self.best_score:.3f} (you're {gap:.3f} below)"
        
        # Performance analysis
        if len(self.performance_trend) >= 3:
            recent_avg = statistics.mean(self.performance_trend[-3:])
            overall_avg = statistics.mean(self.performance_trend)
            consistency = f"\n📈 Recent average: {recent_avg:.3f} | Overall average: {overall_avg:.3f}"
        else:
            consistency = ""
        
        feedback = f"""
🎯 ATTEMPT {attempt_num} PERFORMANCE REPORT:
{motivation}

📊 DETAILED ANALYSIS:
Score: {score:.3f} / {self.ultra_target} (Target)
Progress: {(score/self.ultra_target)*100:.1f}% to ULTRA
{trend_indicator}{best_update}{consistency}

🧠 COACHING FEEDBACK:
{coaching}

📝 YOUR PROMPT WAS:
"{prompt}"

WHAT'S WORKING:
✅ Format structure (wbgmsst + white background)
✅ Target inclusion ("{self.target_prompt}")

AREAS FOR IMPROVEMENT:
"""
        
        # Add specific improvement suggestions based on score range
        if score < 0.8:
            feedback += """🔧 Need more premium descriptors (aerospace-grade, ultra-precision)
🔧 Try industry authority terms (military-spec, defense-grade)
🔧 Add process excellence terms (precision-engineered, masterpiece-quality)"""
        elif score < 0.9:
            feedback += """🔧 Combine multiple quality layers
🔧 Use specification terms (ultra-high technical specification)
🔧 Consider material descriptors (aerospace-alloy, precision-forged)"""
        else:
            feedback += """🔧 You're very close! Fine-tune descriptor combinations
🔧 Consider ultimate-level language (precision-aerospace-grade)"""
        
        return feedback

    def run_conversational_optimization(self):
        """Run the main conversational optimization loop"""
        
        print(f"\n🚀 STARTING CONVERSATIONAL OPTIMIZATION")
        print(f"🎯 GOAL: Achieve {self.ultra_target}+ through AI conversation and feedback")
        
        for attempt_num in range(1, self.max_attempts + 1):
            print(f"\n{'='*20} ATTEMPT {attempt_num}/{self.max_attempts} {'='*20}")
            
            # Debug conversation state
            self.debug_conversation_state(attempt_num)
            
            # Get AI's current response (it should contain a prompt)
            if attempt_num == 1:
                # First attempt already has the initial response
                ai_response = self.conversation_history[-1]["content"]
            else: # "{self.target_prompt}"
                # Ask for next optimization
                request_message = f"""GENERATE NEXT PROMPT - ATTEMPT {attempt_num}

Previous score: {self.performance_trend[-1]:.3f} (Target: {self.ultra_target})

REQUIREMENTS:
- Start with EXACTLY: "wbgmsst, "
- End with EXACTLY: ", white background"
- Include: "{self.target_prompt}"
- Use MORE premium descriptors than before
- 80-120 characters total

RESPOND WITH ONLY THE PROMPT - NO EXPLANATION:"""
                
                print(f"🤖 Asking AI for attempt {attempt_num}...")
                ai_response = self.query_deepseek_conversation(request_message)
                print(f"🤖 AI Response:\n{ai_response}")
            
            # Extract prompt from response
            prompt = self.extract_prompt_from_response(ai_response)
            print(f"\n📝 Extracted Prompt: {prompt}")
            
            # Validate the prompt
            print("🔧 Validating prompt...")
            score, demo_score = self.run_validation(prompt)
            
            # Store attempt data
            attempt_data = {
                'attempt': attempt_num,
                'prompt': prompt,
                'score': score,
                'demo_score': demo_score,
                'ai_response': ai_response,
                'timestamp': time.time()
            }
            self.attempts.append(attempt_data)
            
            # Generate performance feedback
            feedback = self.generate_performance_feedback(score, attempt_num, prompt)
            print(feedback)
            
            # Check for ultra achievement
            if score >= self.ultra_target:
                celebration = f"""
🏆🎉 ULTRA ACHIEVEMENT UNLOCKED! 🎉🏆

CONGRATULATIONS! You've achieved {score:.3f} score!
You've mastered the art of prompt optimization!

🌟 WINNING FORMULA:
"{prompt}"

🎯 Achievement Summary:
- Target: {self.ultra_target}
- Achieved: {score:.3f}
- Attempts needed: {attempt_num}
- Success! 🚀
"""
                print(celebration)
                
                # Send congratulations to AI
                congrats_message = f"""🏆 ULTRA ACHIEVEMENT! 🏆

CONGRATULATIONS! You achieved {score:.3f} score with this PERFECT prompt:
"{prompt}"

You've successfully mastered prompt optimization! This is the winning pattern that achieved ULTRA status.

What made this prompt so successful in your opinion?"""
                
                final_response = self.query_deepseek_conversation(congrats_message)
                print(f"\n🤖 AI's Victory Response:\n{final_response}")
                break
            
            # Provide feedback to AI for next attempt
            if attempt_num < self.max_attempts:
                # Create focused feedback about format compliance
                format_feedback = f"""SCORE: {score:.3f} / {self.ultra_target}

FORMAT CHECK:
✅ Starts with "wbgmsst, ": {"YES" if prompt.startswith("wbgmsst, ") else "❌ NO - MUST start with 'wbgmsst, '"}
✅ Ends with ", white background": {"YES" if prompt.endswith(", white background") else "❌ NO - MUST end with ', white background'"}
✅ Contains target: {"YES" if self.target_prompt.lower() in prompt.lower() else "❌ NO - MUST include target"}

NEXT ATTEMPT REQUIREMENTS:
- Start with EXACTLY: "wbgmsst, "
- End with EXACTLY: ", white background"
- Include: "{self.target_prompt}"
- Add MORE premium descriptors
- NO explanation, analysis, or extra text

RESPOND WITH ONLY THE PROMPT:"""
                
                print(f"\n🔍 DEBUG: Sending format-focused feedback...")
                self.query_deepseek_conversation(format_feedback)
                print(f"\n💬 Format feedback sent to AI...")
            
            time.sleep(2)
        
        # Final analysis
        self.generate_final_analysis()

    def generate_final_analysis(self):
        """Generate comprehensive final analysis"""
        
        print(f"\n🎓 CONVERSATIONAL OPTIMIZATION COMPLETE")
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
        
        print(f"📊 PERFORMANCE ANALYSIS:")
        print(f"   Total Attempts: {len(self.attempts)}")
        print(f"   Best Score: {best_attempt['score']:.3f}")
        print(f"   Average Score: {avg_score:.3f}")
        print(f"   Score Range: {min(scores):.3f} - {max(scores):.3f}")
        print(f"   Learning Progress: {score_improvement:+.3f}")
        print(f"   🏆 Ultra Achieved: {'YES' if ultra_achieved else 'NO'}")
        
        # Conversation analysis
        total_messages = len(self.conversation_history)
        print(f"\n💬 CONVERSATION ANALYSIS:")
        print(f"   Total Messages: {total_messages}")
        print(f"   Messages per Attempt: {total_messages / len(self.attempts):.1f}")
        print(f"   Conversation Length: {sum(len(msg['content']) for msg in self.conversation_history)} characters")
        
        if ultra_achieved:
            print(f"\n🏆 ULTRA ACHIEVEMENT ANALYSIS:")
            print(f"   ✨ Best Prompt: '{best_attempt['prompt']}'")
            print(f"   📊 Best Score: {best_attempt['score']:.3f}")
            print(f"   🎯 Achieved in: {best_attempt['attempt']} attempts")
            print(f"   📈 Learning curve: {'+'.join([f'{s:.2f}' for s in scores])}")
        
        # Learning insights
        print(f"\n🧠 LEARNING INSIGHTS:")
        
        if len(scores) >= 3:
            early_avg = statistics.mean(scores[:3])
            late_avg = statistics.mean(scores[-3:])
            learning_rate = late_avg - early_avg
            print(f"   📈 Learning Rate: {learning_rate:+.3f} (early vs late)")
        
        # Show progression
        print(f"\n📈 SCORE PROGRESSION:")
        for i, attempt in enumerate(self.attempts, 1):
            trend = ""
            if i > 1:
                change = attempt['score'] - self.attempts[i-2]['score']
                trend = f" ({change:+.3f})"
            print(f"   {i}: {attempt['score']:.3f}{trend}")

def main():
    """Main testing function"""
    
    test_prompts = [
        "hexagonal prism steel structure",
        "elegant silk fabric draping",
        "transparent glass sphere with reflections",
        "ornate wooden sculpture"
    ]
    
    print("🚀 DEEPSEEK CONVERSATIONAL OPTIMIZER")
    print("=" * 80)
    print("🎯 Mission: Achieve ULTRA scores through conversational AI learning")
    print("⚡ Strategy: Performance feedback + motivation + conversation history")
    print("🏆 Goal: 0.96+ ultra scores with AI partnership")
    print("=" * 80)
    
    # Test each prompt
    for i, prompt in enumerate(test_prompts, 1):
        print(f"\n🎯 TESTING PROMPT {i}/{len(test_prompts)}: '{prompt}'")
        
        optimizer = DeepSeekConversationalOptimizer(
            target_prompt=prompt,
            ultra_target=0.96,
            max_attempts=10  # Fewer attempts since AI learns faster with feedback
        )
        
        optimizer.run_conversational_optimization()
        
        if i < len(test_prompts):
            print(f"\n⏸️ Brief pause before next test...")
            time.sleep(5)
    
    print(f"\n🎓 ALL CONVERSATIONAL TESTS COMPLETE!")
    print("🚀 DeepSeek conversational optimization fully analyzed!")

if __name__ == "__main__":
    main() 