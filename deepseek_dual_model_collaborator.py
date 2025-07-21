#!/usr/bin/env python3
"""
DeepSeek Dual-Model Collaborative Ultra Achievement System
==========================================================
Purpose: Two DeepSeek instances collaborate to find optimal prompts for ultra achievement

Collaboration Strategy:
- Model A (Generator): Creates custom prompts
- Model B (Analyzer): Analyzes results and provides improvement guidance
- Iterative feedback loop between models
- Progressive learning and strategy refinement
"""

import requests
import json
import time
import subprocess
import sys
from typing import Dict, List, Tuple
import statistics

class DeepSeekDualCollaborator:
    """Dual DeepSeek model collaboration for ultra achievement"""
    
    def __init__(self, target_prompt: str, ultra_target: float = 0.96, max_iterations: int = 15):
        self.ollama_url = "http://localhost:11434"
        self.model_name = "deepseek-r1:1.5b"
        self.target_prompt = target_prompt
        self.ultra_target = ultra_target
        self.max_iterations = max_iterations
        
        # Collaboration tracking
        self.iterations = []
        self.conversation_history = []
        self.best_score = 0.0
        self.best_prompt = ""
        
        print(f"🚀 DEEPSEEK DUAL-MODEL COLLABORATION SYSTEM")
        print(f"🤖 Model A: Custom Prompt Generator")
        print(f"🧠 Model B: Performance Analyzer & Strategy Advisor")
        print(f"🎯 Target: {ultra_target}+ | Max Iterations: {max_iterations}")
        print(f"📝 Optimizing: '{target_prompt}'")
        print("=" * 80)

    def query_model(self, message: str, role_context: str = "", temperature: float = 0.9) -> str:
        """Query DeepSeek with role-specific context"""
        
        # Add role context to the message
        if role_context:
            full_message = f"{role_context}\n\n{message}"
        else:
            full_message = message
        
        data = {
            "model": self.model_name,
            "messages": [{"role": "user", "content": full_message}],
            "stream": False,
            "options": {
                "temperature": temperature,
                "top_p": 0.95,
                "num_predict": 500,
                "repeat_penalty": 1.1
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
    #         cmd = [sys.executable, "mock_validator_for_testing.py", prompt]
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

    def model_a_generate_prompt(self, iteration: int, analyzer_feedback: str = "") -> str:
        """Model A: Generate custom prompt based on analyzer feedback"""
        
        role_context = """You are DEEPSEEK MODEL A - ULTRA PROMPT GENERATOR
Your specialization: Creating ultra-high-performance custom prompts for 3D model generation
Your mission: Generate prompts that achieve 0.96+ validation scores
Your strength: Creative optimization and premium language mastery"""

        # Build context from previous iterations
        context = ""
        if self.iterations:
            recent_scores = [iter_data['score'] for iter_data in self.iterations[-3:]]
            context += f"\nRECENT PERFORMANCE: {recent_scores}\n"
            if self.best_score > 0:
                context += f"BEST SCORE SO FAR: {self.best_score:.3f}\n"
                context += f"ULTRA GAP: {self.ultra_target - self.best_score:.3f}\n"

        # Include analyzer feedback
        feedback_section = ""
        if analyzer_feedback:
            feedback_section = f"\n🧠 MODEL B ANALYZER FEEDBACK:\n{analyzer_feedback}\n"

        message = f"""ULTRA PROMPT GENERATION - ITERATION {iteration}

TARGET: Create optimized prompt for "{self.target_prompt}"
GOAL: Achieve {self.ultra_target}+ validation score

{context}
{feedback_section}

ULTRA ACHIEVEMENT PATTERNS (proven effective):
🏆 Premium Descriptors: ultra-precision, aerospace-grade, masterpiece-quality
🔧 Technical Excellence: precision-engineered, CAD-accurate, manufacturing-perfection  
🎨 Artistic Quality: museum-quality, gallery-exhibition, award-winning
⚡ Process Excellence: ultra-detailed, flawless-execution, precision-manufacturing

PROVEN ULTRA FORMULA:
"wbgmsst, [PREMIUM-AUTHORITY] [PROCESS-EXCELLENCE] [TARGET], [SPECIFICATION-PRECISION], white background"

EXAMPLES OF ULTRA-ACHIEVING PATTERNS:
- "wbgmsst, aerospace-grade precision-engineered steel structure, ultra-high technical specification, white background"
- "wbgmsst, museum-quality masterpiece sculpture, gallery-exhibition excellence, white background"

YOUR TASK: Generate the OPTIMAL custom prompt for "{self.target_prompt}" that will achieve {self.ultra_target}+ score.

Focus on combining the highest-impact descriptors in logical sequence.

RESPOND WITH ONLY THE OPTIMIZED PROMPT:"""

        print(f"🤖 Model A: Generating custom prompt...")
        response = self.query_model(message, role_context, temperature=0.95)
        
        if "ERROR:" in response:
            # Fallback prompt
            fallback = f"wbgmsst, aerospace-grade precision-engineered {self.target_prompt}, ultra-high technical specification, white background"
            print(f"   ⚠️ Model A error, using fallback")
            return fallback
        
        # Clean response
        cleaned = self.clean_prompt_response(response)
        print(f"   ✨ Generated: '{cleaned[:70]}{'...' if len(cleaned) > 70 else ''}'")
        return cleaned

    def model_b_analyze_performance(self, iteration: int, prompt: str, score: float, validation_data: Dict) -> str:
        """Model B: Analyze performance and provide improvement guidance"""
        
        role_context = """You are DEEPSEEK MODEL B - PERFORMANCE ANALYZER & STRATEGY ADVISOR
Your specialization: Analyzing prompt performance and providing strategic improvement guidance
Your mission: Help Model A achieve 0.96+ ultra scores through expert analysis
Your strength: Pattern recognition, performance analysis, and strategic optimization"""

        # Performance context
        performance_level = ""
        if score >= self.ultra_target:
            performance_level = "🏆 ULTRA ACHIEVEMENT - MISSION ACCOMPLISHED!"
        elif score >= 0.9:
            performance_level = f"🟡 EXCELLENT - Only {self.ultra_target - score:.3f} from ULTRA!"
        elif score >= 0.8:
            performance_level = f"🟢 STRONG - {self.ultra_target - score:.3f} away from ULTRA"
        elif score >= 0.7:
            performance_level = f"🔵 GOOD - {self.ultra_target - score:.3f} to ULTRA target"
        else:
            performance_level = f"🔴 NEEDS IMPROVEMENT - {self.ultra_target - score:.3f} gap"

        # Build analysis context
        quality_indicators = [] #validation_data.get("quality_indicators_found", [])
        quality_boost = 0.0 #validation_data.get("quality_boost", 0.0)
        pattern_bonus = 0.0 #validation_data.get("pattern_bonus", 0.0)
        
        # Historical performance context
        historical_context = ""
        if len(self.iterations) > 1:
            recent_scores = [iter_data['score'] for iter_data in self.iterations[-3:]]
            trend = "improving" if len(recent_scores) > 1 and recent_scores[-1] > recent_scores[0] else "declining"
            historical_context = f"PERFORMANCE TREND: {trend} (Recent: {recent_scores})\n"

        message = f"""PERFORMANCE ANALYSIS - ITERATION {iteration}

PROMPT ANALYZED: "{prompt}"
SCORE: {score:.3f} / {self.ultra_target} (Target)
STATUS: {performance_level}

{historical_context}

DETAILED BREAKDOWN:
📊 Validation Score: {score:.3f}
🎯 Ultra Progress: {(score/self.ultra_target)*100:.1f}%
🏆 Quality Indicators: {len(quality_indicators)} found ({', '.join(quality_indicators[:5])})
📈 Quality Boost: +{quality_boost:.3f}
⭐ Pattern Bonus: +{pattern_bonus:.3f}

ANALYSIS TASK:
1. Identify what worked well in this prompt
2. Identify specific areas for improvement  
3. Recommend precise optimization strategies for Model A
4. Suggest specific premium descriptors or patterns to try

If score >= {self.ultra_target}: Analyze what made this ULTRA-achieving pattern perfect
If score < {self.ultra_target}: Provide specific guidance to bridge the gap

Focus on actionable recommendations that Model A can implement immediately.

PROVIDE YOUR EXPERT ANALYSIS AND STRATEGIC RECOMMENDATIONS:"""

        print(f"🧠 Model B: Analyzing performance...")
        response = self.query_model(message, role_context, temperature=0.7)
        
        if "ERROR:" in response:
            # Fallback analysis
            fallback = f"Score {score:.3f} analysis: Add more premium descriptors like 'aerospace-grade' and 'precision-engineered' for higher scores."
            print(f"   ⚠️ Model B error, using fallback")
            return fallback
        
        print(f"   🧠 Analysis: {response[:100]}{'...' if len(response) > 100 else ''}")
        return response

    def clean_prompt_response(self, response: str) -> str:
        """Clean Model A response to extract the prompt"""
        
        lines = response.split('\n')
        
        # Look for lines starting with wbgmsst or containing key terms
        for line in lines:
            line = line.strip()
            if line.lower().startswith('wbgmsst') and len(line) > 20:
                return line
        
        # Look for any substantial line with 3D-related terms
        for line in lines:
            line = line.strip()
            if len(line) > 20 and any(term in line.lower() for term in ['3d', 'model', 'structure', 'sculpture']):
                if not line.lower().startswith('wbgmsst'):
                    line = f"wbgmsst, {line}"
                if 'white background' not in line.lower():
                    line += ", white background"
                return line
        
        # Ultimate fallback
        return f"wbgmsst, ultra-precision {self.target_prompt}, aerospace-grade quality, white background"

    def run_collaborative_optimization(self):
        """Run the dual-model collaborative optimization"""
        
        print(f"\n🚀 STARTING DUAL-MODEL COLLABORATION")
        print(f"🎯 MISSION: Achieve {self.ultra_target}+ through model collaboration")
        
        analyzer_feedback = ""  # Initial feedback is empty
        
        for iteration in range(1, self.max_iterations + 1):
            print(f"\n{'='*25} ITERATION {iteration}/{self.max_iterations} {'='*25}")
            
            # MODEL A: Generate custom prompt
            custom_prompt = self.model_a_generate_prompt(iteration, analyzer_feedback)
            
            # Validate the prompt
            print(f"🔧 Validating collaborative prompt...")
            score, validation_data = self.run_validation(custom_prompt)
            
            # Update best score tracking
            improvement = ""
            if score > self.best_score:
                improvement = f" 🌟 NEW BEST! (+{score - self.best_score:.3f})"
                self.best_score = score
                self.best_prompt = custom_prompt
            
            print(f"📊 RESULT: {score:.3f}{improvement}")
            
            # Store iteration data
            iteration_data = {
                'iteration': iteration,
                'prompt': custom_prompt,
                'score': score,
                'validation_data': validation_data,
                'analyzer_feedback': analyzer_feedback,
                'timestamp': time.time()
            }
            self.iterations.append(iteration_data)
            
            # Check for ultra achievement
            if score >= self.ultra_target:
                print(f"\n🏆 ULTRA ACHIEVEMENT THROUGH COLLABORATION!")
                print(f"✨ Winning Prompt: '{custom_prompt}'")
                print(f"🎯 Achieved in {iteration} collaborative iterations!")
                print(f"📊 Final Score: {score:.3f}")
                
                # Final analysis of winning pattern
                final_analysis = self.model_b_analyze_performance(iteration, custom_prompt, score, validation_data)
                print(f"\n🧠 WINNING PATTERN ANALYSIS:\n{final_analysis}")
                break
            
            # MODEL B: Analyze performance and provide feedback for next iteration
            if iteration < self.max_iterations:
                analyzer_feedback = self.model_b_analyze_performance(iteration, custom_prompt, score, validation_data)
                
                # Add feedback to conversation history
                self.conversation_history.append({
                    'iteration': iteration,
                    'model_a_prompt': custom_prompt,
                    'score': score,
                    'model_b_feedback': analyzer_feedback
                })
                
                print(f"\n🔄 Model B feedback provided for next iteration")
            
            time.sleep(2)
        
        # Final collaborative analysis
        self.generate_collaboration_analysis()

    def generate_collaboration_analysis(self):
        """Generate analysis of the collaborative process"""
        
        print(f"\n🎓 DUAL-MODEL COLLABORATION ANALYSIS")
        print("=" * 80)
        
        if not self.iterations:
            print("❌ No iterations recorded")
            return
        
        # Performance statistics
        scores = [iter_data['score'] for iter_data in self.iterations]
        best_iteration = max(self.iterations, key=lambda x: x['score'])
        
        avg_score = statistics.mean(scores)
        score_improvement = scores[-1] - scores[0] if len(scores) > 1 else 0
        ultra_achieved = any(s >= self.ultra_target for s in scores)
        
        print(f"📊 COLLABORATION PERFORMANCE:")
        print(f"   Total Iterations: {len(self.iterations)}")
        print(f"   Best Score: {best_iteration['score']:.3f}")
        print(f"   Average Score: {avg_score:.3f}")
        print(f"   Score Improvement: {score_improvement:+.3f}")
        print(f"   🏆 Ultra Achieved: {'YES' if ultra_achieved else 'NO'}")
        
        if ultra_achieved:
            print(f"\n🏆 COLLABORATIVE SUCCESS ANALYSIS:")
            print(f"   ✨ Best Prompt: '{best_iteration['prompt']}'")
            print(f"   📊 Best Score: {best_iteration['score']:.3f}")
            print(f"   🤝 Achieved in: {best_iteration['iteration']} collaborative iterations")
        
        # Collaboration effectiveness analysis
        print(f"\n🤝 COLLABORATION EFFECTIVENESS:")
        
        if len(scores) >= 3:
            early_scores = scores[:len(scores)//2]
            late_scores = scores[len(scores)//2:]
            collaboration_improvement = statistics.mean(late_scores) - statistics.mean(early_scores)
            print(f"   📈 Collaboration Learning: {collaboration_improvement:+.3f}")
            
            if collaboration_improvement > 0.05:
                print("   ✅ Strong collaborative learning detected")
            elif collaboration_improvement > 0:
                print("   🟡 Moderate collaborative improvement")
            else:
                print("   🔴 Limited collaborative benefit observed")
        
        # Model interaction analysis
        print(f"\n🤖 MODEL INTERACTION INSIGHTS:")
        print(f"   🔄 Feedback Exchanges: {len(self.conversation_history)}")
        
        if ultra_achieved:
            print("   ✅ Dual-model approach successfully achieved ultra performance")
            print("   🚀 Collaboration pattern proven effective for ultra achievement")
        elif max(scores) >= 0.9:
            print("   🟡 Collaboration brought performance very close to ultra threshold")
            print("   🎯 Refinement of collaboration strategy recommended")
        else:
            print("   🔵 Collaboration showed learning but requires strategy enhancement")
        
        # Recommendations
        print(f"\n💡 COLLABORATIVE OPTIMIZATION RECOMMENDATIONS:")
        
        if ultra_achieved:
            print("   ✅ Dual-model collaboration PROVEN effective for ultra achievement")
            print("   🚀 Ready for production deployment with collaborative architecture")
        else:
            print("   📈 Increase feedback specificity between models")
            print("   🔄 Extend collaboration iterations for breakthrough performance")
            print("   🎯 Enhance Model B's analytical capability with more detailed guidance")

def main():
    """Main collaborative testing function"""
    
    test_prompts = [
        "hexagonal prism steel structure",
        "elegant silk fabric draping"
    ]
    
    print("🚀 DEEPSEEK DUAL-MODEL COLLABORATIVE SYSTEM - PHASE 3")
    print("=" * 80)
    print("🤖 Architecture: Two DeepSeek models working in collaboration")
    print("⚡ Strategy: Generator + Analyzer collaborative optimization")
    print("🏆 Goal: Achieve 0.96+ ultra scores through model collaboration")
    print("=" * 80)
    
    # Test collaborative optimization
    for i, prompt in enumerate(test_prompts, 1):
        print(f"\n🎯 COLLABORATIVE TEST {i}/{len(test_prompts)}: '{prompt}'")
        
        collaborator = DeepSeekDualCollaborator(
            target_prompt=prompt,
            ultra_target=0.96,
            max_iterations=12  # Reasonable number for collaboration
        )
        
        collaborator.run_collaborative_optimization()
        
        if i < len(test_prompts):
            print(f"\n⏸️ Brief pause before next collaborative test...")
            time.sleep(5)
    
    print(f"\n🎓 ALL COLLABORATIVE TESTS COMPLETE!")
    print("🤝 Dual-model collaboration capability fully analyzed!")

if __name__ == "__main__":
    main() 