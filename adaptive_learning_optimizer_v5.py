#!/usr/bin/env python3
"""
Adaptive Learning Optimizer v5.0 - Enhanced AI Feedback Loops
Purpose: Stronger AI decision-making with feedback loops every 3 trials, 
better insights, and more aggressive custom prompt generation.
"""
import requests
import json
import time
import subprocess
import sys
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass, asdict
from pathlib import Path
import sqlite3
import statistics
import re
import torch
seed = 42
torch.manual_seed(seed)
torch.use_deterministic_algorithms(True)
@dataclass
class AIDecision:
    """AI Decision with improved tracking"""
    attempt_number: int
    decision_type: str  # "custom_prompt", "strategy_sequence", "early_stop", "fallback"
    content: str  # Custom prompt OR strategy list
    reasoning: str
    confidence: float
    expected_improvement: float
    actual_improvement: float  # Updated after validation
    decision_success: bool  # True if led to improvement
    timestamp: float

@dataclass
class AIInsightSummary:
    """AI insights after every 3 attempts"""
    after_attempts: List[int]
    pattern_analysis: str
    strategy_effectiveness: Dict[str, str]  # strategy -> "effective"/"ineffective"/"unknown"
    next_recommendations: List[str]
    confidence_in_recommendations: float
    should_try_custom_prompt: bool
    custom_prompt_suggestion: Optional[str]
    early_termination_recommendation: bool
    timestamp: float

@dataclass
class OptimizationAttempt:
    """Enhanced attempt tracking"""
    attempt_number: int
    strategy_name: str
    optimized_prompt: str
    validation_score: float
    demo_fidelity_score: float
    score_improvement: float
    fidelity_improvement: float
    meets_minimum: bool
    meets_target: bool
    meets_ultra: bool
    ai_decision: Optional[AIDecision]
    timestamp: float

@dataclass
class OptimizationSession:
    """Session with enhanced AI tracking"""
    original_prompt: str
    category: str
    baseline_score: float
    baseline_fidelity: float
    attempts: List[OptimizationAttempt]
    ai_decisions: List[AIDecision]
    ai_insight_summaries: List[AIInsightSummary]
    best_attempt: Optional[OptimizationAttempt]
    best_score: float
    session_improvement: float
    ai_made_meaningful_decisions: bool
    ai_success_rate: float  # % of AI decisions that improved scores
    reached_minimum: bool
    reached_target: bool
    reached_ultra: bool
    timestamp: float

class EnhancedAIOptimizerV5:
    """v5.0 with stronger AI feedback loops and decision-making"""
    
    def __init__(self, max_attempts: int = 9, min_target: float = 0.6, 
                 target: float = 0.9, ultra_target: float = 0.96):
        self.ollama_url = "http://localhost:11434"
        self.model_name = "deepseek-r1:1.5b"
        self.max_attempts = max_attempts
        self.min_target = min_target
        self.target = target
        self.ultra_target = ultra_target
        
        # Strategy library
        self.strategies = {
            # "raw": "{prompt}",
            "material_focus": "wbgmsst, solid {prompt} object 3D, white background",
            "geometric_focus": "wbgmsst, {prompt} geometric 3D model, white background",
            "basic_description": "3D model of {prompt}",
            "enhanced_clarity": "wbgmsst, detailed 3D {prompt} model, accurate geometry, white background",
            "concrete_object": "wbgmsst, {prompt} as 3D object, realistic proportions, white background",
            "professional_render": "wbgmsst, professional 3D render of {prompt}, studio lighting, white background",
            "high_quality": "wbgmsst, high quality 3D model {prompt}, detailed textures, white background",
            "minimal_enhancement": "{prompt}, 3D object",
            "simplified_description": "simple 3D {prompt}",
            "artistic_focus": "wbgmsst, artistic {prompt} sculpture, clean design, white background"
        }
        
        # AI learning patterns (will be updated dynamically)
        self.learned_patterns = {
            "successful_strategies": [],
            "failed_strategies": [],
            "category_preferences": {},
            "custom_prompt_success_rate": 0.3
        }
    
    def query_deepseek(self, system_prompt: str, user_prompt: str, timeout: int = 90) -> str:
        """Enhanced DeepSeek query with better error handling"""
        data = {
            "model": self.model_name,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            "stream": False,
            "options": {
                "temperature": 0.7,
                "top_p": 0.9
            }
        }
        
        try:
            response = requests.post(f"{self.ollama_url}/api/chat", json=data, timeout=timeout)
            response.raise_for_status()
            result = response.json()
            content = result["message"]["content"]
            
            if len(content.strip()) < 10:
                return "ERROR: Response too short"
            
            return content
        except Exception as e:
            return f"ERROR: {str(e)}"
    
    def categorize_prompt(self, prompt: str) -> str:
        """Enhanced prompt categorization"""
        system_prompt = """You are an expert at categorizing 3D prompts for optimization.

Categories and their optimization characteristics:
- physical_object: Concrete items (tools, furniture) - material_focus often works
- technical_description: Engineering/geometric specs - geometric_focus preferred  
- artistic_abstract: Creative concepts - artistic_focus/enhanced_clarity work
- decorative_ornate: Decorative items - raw/simplified often best

Consider both the object type AND optimization potential.
Respond with ONLY the category name."""

        try:
            response = self.query_deepseek(system_prompt, f"Categorize: '{prompt}'")
            if "ERROR:" not in response:
                categories = ["physical_object", "technical_description", "artistic_abstract", "decorative_ornate"]
                for cat in categories:
                    if cat in response.lower():
                        return cat
            return "physical_object"
        except:
            return "physical_object"
    
    def make_ai_decision(self, prompt: str, category: str, baseline_score: float, 
                        attempts: List[OptimizationAttempt], attempt_number: int) -> AIDecision:
        """Enhanced AI decision-making with better prompts"""
        
        # Prepare detailed context
        attempts_data = []
        for attempt in attempts:
            attempts_data.append({
                "num": attempt.attempt_number,
                "strategy": attempt.strategy_name, 
                "score": f"{attempt.validation_score:.3f}",
                "improvement": f"{attempt.score_improvement:+.3f}",
                "meets_min": attempt.meets_minimum,
                "meets_target": attempt.meets_target
            })
        
        learned_info = ""
        if self.learned_patterns["successful_strategies"]:
            learned_info += f"\nSUCCESSFUL STRATEGIES: {', '.join(self.learned_patterns['successful_strategies'][-5:])}"
        if self.learned_patterns["failed_strategies"]:
            learned_info += f"\nFAILED STRATEGIES: {', '.join(self.learned_patterns['failed_strategies'][-5:])}"
        
        system_prompt = f"""You are an expert AI optimization agent with FULL AUTONOMY to make decisions.

CRITICAL MISSION: Transform this prompt to reach the targets!
- Minimum: {self.min_target} (avoid zero fidelity)
- Target: {self.target} (excellent quality)  
- Ultra: {self.ultra_target} (perfect quality)

CURRENT SITUATION:
- Prompt: "{prompt}"
- Category: {category}
- Baseline: {baseline_score:.3f}
- Attempt: {attempt_number}/{self.max_attempts}

PREVIOUS ATTEMPTS: {attempts_data}

LEARNED PATTERNS: {learned_info}

YOUR DECISION OPTIONS:

1. WRITE_CUSTOM_PROMPT: Create a completely new optimized prompt
   - Use when: strategies are failing, need major improvement
   - Example: "Create custom prompt: wbgmsst, detailed copper cylindrical pipe 5cm diameter, industrial 3D model, accurate dimensions, white background"

2. SELECT_STRATEGIES: Choose specific strategies to try next
   - Use when: some strategies show promise, want to test specific ones
   - Example: "Try strategies: enhanced_clarity,professional_render,high_quality"

3. EARLY_STOP: Stop optimization early
   - Use when: baseline already excellent OR situation hopeless
   - Example: "Stop early: baseline 0.91 already exceeds target"

4. ANALYSIS_NEEDED: Request more attempts to analyze patterns
   - Use when: unclear which approach works best
   - Example: "Continue analysis: need more data on strategy effectiveness"

RESPOND FORMAT:
DECISION: [WRITE_CUSTOM_PROMPT|SELECT_STRATEGIES|EARLY_STOP|ANALYSIS_NEEDED]
REASONING: [Your detailed analysis - think step by step]
CONFIDENCE: [0.1-1.0]
EXPECTED_IMPROVEMENT: [0.0-0.5]
CONTENT: [Custom prompt OR strategy list OR stop reason OR continue]

Think deeply about what will work for this specific prompt and situation!"""

        user_prompt = f"Make optimization decision for attempt {attempt_number}: '{prompt}'"
        
        try:
            response = self.query_deepseek(system_prompt, user_prompt)
            
            if "ERROR:" in response:
                return self._create_fallback_decision(attempt_number, f"AI error: {response}")
            
            # Parse response with flexible parsing
            decision_type = "fallback"
            reasoning = response[:150]
            confidence = 0.5
            expected_improvement = 0.1
            content = "continue"
            
            # Extract decision type
            if "WRITE_CUSTOM_PROMPT" in response.upper() or "CUSTOM PROMPT" in response.upper():
                decision_type = "custom_prompt"
                # Try to extract custom prompt
                custom_patterns = [
                    r'CONTENT:\s*(.+?)(?=\n|$)',
                    r'[Cc]ustom prompt:\s*(.+?)(?=\n|$)',
                    r'[Pp]rompt:\s*(.+?)(?=\n|$)'
                ]
                for pattern in custom_patterns:
                    match = re.search(pattern, response, re.DOTALL)
                    if match:
                        content = match.group(1).strip()
                        break
                if content == "continue":  # No custom prompt found
                    content = f"wbgmsst, optimized {prompt}, high quality 3D model, professional render, white background"
                    
            elif "SELECT_STRATEGIES" in response.upper() or "STRATEGIES" in response.upper():
                decision_type = "strategy_sequence"
                # Extract strategies mentioned
                strategies_found = []
                for strategy in self.strategies.keys():
                    if strategy in response.lower():
                        strategies_found.append(strategy)
                if strategies_found:
                    content = ",".join(strategies_found[:3])  # Limit to 3
                else:
                    content = "enhanced_clarity,professional_render"  # Fallback
                    
            elif "EARLY_STOP" in response.upper() or "STOP" in response.upper():
                decision_type = "early_stop"
                content = "early_termination"
                
            else:  # ANALYSIS_NEEDED or unclear
                decision_type = "strategy_sequence"
                content = "enhanced_clarity,professional_render"
            
            # Extract confidence and expected improvement
            conf_match = re.search(r'CONFIDENCE:\s*([0-9.]+)', response)
            if conf_match:
                confidence = min(1.0, max(0.1, float(conf_match.group(1))))
            
            exp_match = re.search(r'EXPECTED_IMPROVEMENT:\s*([0-9.]+)', response)
            if exp_match:
                expected_improvement = min(0.5, max(0.0, float(exp_match.group(1))))
            
            # Extract reasoning
            reason_match = re.search(r'REASONING:\s*(.+?)(?=CONFIDENCE:|CONTENT:|$)', response, re.DOTALL)
            if reason_match:
                reasoning = reason_match.group(1).strip()
            
            return AIDecision(
                attempt_number=attempt_number,
                decision_type=decision_type,
                content=content,
                reasoning=reasoning,
                confidence=confidence,
                expected_improvement=expected_improvement,
                actual_improvement=0.0,  # Will be updated
                decision_success=False,  # Will be updated
                timestamp=time.time()
            )
            
        except Exception as e:
            print(f"❌ AI decision error: {e}")
            return self._create_fallback_decision(attempt_number, f"Exception: {e}")
    
    def _create_fallback_decision(self, attempt_number: int, error_msg: str) -> AIDecision:
        """Create fallback decision when AI fails"""
        return AIDecision(
            attempt_number=attempt_number,
            decision_type="fallback",
            content="enhanced_clarity",
            reasoning=f"Fallback due to: {error_msg}",
            confidence=0.2,
            expected_improvement=0.05,
            actual_improvement=0.0,
            decision_success=False,
            timestamp=time.time()
        )
    
    def generate_ai_insights_summary(self, prompt: str, category: str, baseline_score: float,
                                   recent_attempts: List[OptimizationAttempt]) -> AIInsightSummary:
        """Generate AI insights every 3 attempts"""
        
        if len(recent_attempts) < 2:
            return None
        
        # Prepare detailed analysis data
        attempts_data = []
        for attempt in recent_attempts:
            attempts_data.append({
                "attempt": attempt.attempt_number,
                "strategy": attempt.strategy_name,
                "score": attempt.validation_score,
                "improvement": attempt.score_improvement,
                "meets_minimum": attempt.meets_minimum,
                "meets_target": attempt.meets_target
            })
        
        system_prompt = f"""You are an expert AI that analyzes optimization patterns and makes strategic recommendations.

ANALYSIS TASK: Review the recent optimization attempts and provide strategic insights.

PROMPT: "{prompt}" (Category: {category})
BASELINE: {baseline_score:.3f}
TARGETS: Min {self.min_target} | Target {self.target} | Ultra {self.ultra_target}

RECENT ATTEMPTS:
{json.dumps(attempts_data, indent=2)}

ANALYZE:
1. What patterns do you see in strategy effectiveness?
2. Which strategies are working/failing and why?
3. What should be tried next?
4. Should we try a custom prompt approach?
5. Are we making progress toward targets?

RESPOND IN THIS FORMAT:
PATTERN_ANALYSIS: [Your analysis of what's working/not working]
STRATEGY_EFFECTIVENESS: [List each strategy as: strategy_name=effective/ineffective/unknown]
NEXT_RECOMMENDATIONS: [List 2-3 specific strategies to try next]
CONFIDENCE: [0.1-1.0]
CUSTOM_PROMPT_NEEDED: [YES/NO]
CUSTOM_PROMPT_SUGGESTION: [If YES, provide optimized prompt]
EARLY_TERMINATION: [YES/NO - if situation is hopeless or already optimal]

Be specific and actionable in your recommendations!"""

        user_prompt = f"Analyze optimization progress for: '{prompt}'"
        
        try:
            response = self.query_deepseek(system_prompt, user_prompt)
            
            if "ERROR:" in response:
                return self._create_fallback_insights(recent_attempts)
            
            # Parse response
            pattern_analysis = "Standard analysis"
            strategy_effectiveness = {}
            next_recommendations = ["enhanced_clarity", "professional_render"]
            confidence = 0.5
            custom_prompt_needed = False
            custom_prompt_suggestion = None
            early_termination = False
            
            # Extract pattern analysis
            pattern_match = re.search(r'PATTERN_ANALYSIS:\s*(.+?)(?=STRATEGY_EFFECTIVENESS:|$)', response, re.DOTALL)
            if pattern_match:
                pattern_analysis = pattern_match.group(1).strip()
            
            # Extract strategy effectiveness
            effect_match = re.search(r'STRATEGY_EFFECTIVENESS:\s*(.+?)(?=NEXT_RECOMMENDATIONS:|$)', response, re.DOTALL)
            if effect_match:
                effect_text = effect_match.group(1).strip()
                for line in effect_text.split('\n'):
                    if '=' in line:
                        parts = line.split('=')
                        if len(parts) == 2:
                            strategy = parts[0].strip()
                            effectiveness = parts[1].strip()
                            strategy_effectiveness[strategy] = effectiveness
            
            # Extract recommendations
            rec_match = re.search(r'NEXT_RECOMMENDATIONS:\s*(.+?)(?=CONFIDENCE:|$)', response, re.DOTALL)
            if rec_match:
                rec_text = rec_match.group(1).strip()
                recommendations = [r.strip() for r in rec_text.replace('\n', ',').split(',') if r.strip()]
                if recommendations:
                    next_recommendations = recommendations[:3]
            
            # Extract custom prompt info
            if "CUSTOM_PROMPT_NEEDED: YES" in response.upper():
                custom_prompt_needed = True
                custom_match = re.search(r'CUSTOM_PROMPT_SUGGESTION:\s*(.+?)(?=EARLY_TERMINATION:|$)', response, re.DOTALL)
                if custom_match:
                    custom_prompt_suggestion = custom_match.group(1).strip()
            
            # Extract early termination
            if "EARLY_TERMINATION: YES" in response.upper():
                early_termination = True
            
            # Extract confidence
            conf_match = re.search(r'CONFIDENCE:\s*([0-9.]+)', response)
            if conf_match:
                confidence = min(1.0, max(0.1, float(conf_match.group(1))))
            
            return AIInsightSummary(
                after_attempts=[a.attempt_number for a in recent_attempts],
                pattern_analysis=pattern_analysis,
                strategy_effectiveness=strategy_effectiveness,
                next_recommendations=next_recommendations,
                confidence_in_recommendations=confidence,
                should_try_custom_prompt=custom_prompt_needed,
                custom_prompt_suggestion=custom_prompt_suggestion,
                early_termination_recommendation=early_termination,
                timestamp=time.time()
            )
            
        except Exception as e:
            print(f"❌ AI insights error: {e}")
            return self._create_fallback_insights(recent_attempts)
    
    def _create_fallback_insights(self, attempts: List[OptimizationAttempt]) -> AIInsightSummary:
        """Create fallback insights when AI analysis fails"""
        return AIInsightSummary(
            after_attempts=[a.attempt_number for a in attempts],
            pattern_analysis="Unable to analyze - AI error",
            strategy_effectiveness={},
            next_recommendations=["enhanced_clarity", "professional_render"],
            confidence_in_recommendations=0.3,
            should_try_custom_prompt=True,
            custom_prompt_suggestion=None,
            early_termination_recommendation=False,
            timestamp=time.time()
        )
    
    def execute_ai_decision(self, decision: AIDecision, prompt: str) -> Tuple[str, str]:
        """Execute AI decision and return strategy name and optimized prompt"""
        
        if decision.decision_type == "early_stop":
            return "early_stop", prompt
        
        elif decision.decision_type == "custom_prompt":
            # Use AI's custom prompt
            custom_prompt = decision.content
            if len(custom_prompt) < 10:  # Fallback if too short
                custom_prompt = f"wbgmsst, detailed 3D {prompt}, high quality model, white background"
            return "ai_custom_prompt", custom_prompt
        
        elif decision.decision_type == "strategy_sequence":
            # Use first strategy from AI's list
            strategies = [s.strip() for s in decision.content.split(',')]
            valid_strategies = [s for s in strategies if s in self.strategies]
            if valid_strategies:
                strategy = valid_strategies[0]
                return strategy, self.strategies[strategy].format(prompt=prompt)
        
        # Fallback
        return "enhanced_clarity", self.strategies["enhanced_clarity"].format(prompt=prompt)
    
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
                
        except subprocess.TimeoutExpired:
            print(f"   ⚠️ Validation timeout")
            return 0.0, 0.0
        except Exception as e:
            print(f"   ❌ Validation error: {e}")
            return 0.0, 0.0
    
    def update_learned_patterns(self, attempt: OptimizationAttempt):
        """Update learned patterns based on attempt results"""
        strategy = attempt.strategy_name
        improvement = attempt.score_improvement
        
        if improvement > 0.05:  # Significant improvement
            if strategy not in self.learned_patterns["successful_strategies"]:
                self.learned_patterns["successful_strategies"].append(strategy)
            # Remove from failed if it was there
            if strategy in self.learned_patterns["failed_strategies"]:
                self.learned_patterns["failed_strategies"].remove(strategy)
        elif improvement < -0.05:  # Significant decrease
            if strategy not in self.learned_patterns["failed_strategies"]:
                self.learned_patterns["failed_strategies"].append(strategy)
    
    def optimize_with_enhanced_feedback(self, prompt: str) -> OptimizationSession:
        """Main optimization with enhanced AI feedback loops"""
        
        print(f"\n🤖 ENHANCED AI OPTIMIZER v5.0: '{prompt}'")
        print("=" * 80)
        print(f"🎯 Targets: Min {self.min_target} | Target {self.target} | Ultra {self.ultra_target}")
        print("🧠 Features: AI insights every 3 attempts, stronger decision-making")
        
        # Setup
        category = self.categorize_prompt(prompt)
        print(f"📋 Category: {category}")
        
        baseline_score, baseline_fidelity = self.run_validation(prompt)
        print(f"📊 Baseline: {baseline_score:.3f}")
        
        # Initialize tracking
        attempts = []
        ai_decisions = []
        ai_insight_summaries = []
        best_attempt = None
        best_score = baseline_score
        ai_meaningful_decisions = 0
        ai_successful_decisions = 0
        
        # Early termination check
        if baseline_score >= self.ultra_target:
            print(f"🏆 BASELINE ALREADY ULTRA-OPTIMAL!")
            return self._create_session_summary(prompt, category, baseline_score, baseline_fidelity,
                                              attempts, ai_decisions, ai_insight_summaries, best_attempt)
        
        # Main optimization loop
        for attempt_num in range(1, self.max_attempts + 1):
            print(f"\n🔄 ATTEMPT {attempt_num}/{self.max_attempts}")
            
            # AI Decision Making
            print(f"🤖 AI analyzing situation and making decision...")
            ai_decision = self.make_ai_decision(prompt, category, baseline_score, attempts, attempt_num)
            ai_decisions.append(ai_decision)
            
            if ai_decision.decision_type != "fallback":
                ai_meaningful_decisions += 1
                print(f"   🧠 AI Decision: {ai_decision.decision_type}")
                print(f"   💭 Reasoning: {ai_decision.reasoning[:100]}...")
                print(f"   🎯 Confidence: {ai_decision.confidence:.2f}")
                print(f"   📈 Expected: +{ai_decision.expected_improvement:.3f}")
            else:
                print(f"   ⚠️ AI Decision Failed - Using Fallback")
            
            # Early termination check
            if ai_decision.decision_type == "early_stop":
                print(f"   🛑 AI Early Termination")
                break
            
            # Execute decision
            strategy_name, optimized_prompt = self.execute_ai_decision(ai_decision, prompt)
            print(f"   🔧 Executing: {strategy_name}")
            print(f"   ✨ Prompt: '{optimized_prompt[:70]}{'...' if len(optimized_prompt) > 70 else ''}'")
            
            # Validate
            val_score, val_fidelity = self.run_validation(optimized_prompt)
            score_improvement = val_score - baseline_score
            fidelity_improvement = val_fidelity - baseline_fidelity
            
            # Update AI decision outcome
            ai_decision.actual_improvement = score_improvement
            if score_improvement > 0:
                ai_decision.decision_success = True
                if ai_decision.decision_type != "fallback":
                    ai_successful_decisions += 1
            
            print(f"   📊 Result: {val_score:.3f} ({score_improvement:+.3f})")
            print(f"   🎯 Min {'✅' if val_score >= self.min_target else '❌'} | Target {'✅' if val_score >= self.target else '❌'} | Ultra {'✅' if val_score >= self.ultra_target else '❌'}")
            print(f"   🤖 AI Success: {'✅' if ai_decision.decision_success else '❌'}")
            
            # Create attempt record
            attempt = OptimizationAttempt(
                attempt_number=attempt_num,
                strategy_name=strategy_name,
                optimized_prompt=optimized_prompt,
                validation_score=val_score,
                demo_fidelity_score=val_fidelity,
                score_improvement=score_improvement,
                fidelity_improvement=fidelity_improvement,
                meets_minimum=val_score >= self.min_target,
                meets_target=val_score >= self.target,
                meets_ultra=val_score >= self.ultra_target,
                ai_decision=ai_decision,
                timestamp=time.time()
            )
            attempts.append(attempt)
            
            # Update best score
            if val_score > best_score:
                best_score = val_score
                best_attempt = attempt
                print(f"   🌟 NEW BEST SCORE!")
            
            # Update learned patterns
            self.update_learned_patterns(attempt)
            
            # AI Insights Summary every 3 attempts
            if attempt_num % 3 == 0 and attempt_num > 0:
                print(f"\n🧠 AI INSIGHTS SUMMARY (After {attempt_num} attempts)")
                recent_attempts = attempts[-3:]
                insights = self.generate_ai_insights_summary(prompt, category, baseline_score, recent_attempts)
                
                if insights:
                    ai_insight_summaries.append(insights)
                    print(f"   📊 Pattern Analysis: {insights.pattern_analysis[:80]}...")
                    print(f"   📈 Strategy Effectiveness: {len(insights.strategy_effectiveness)} strategies analyzed")
                    print(f"   🎯 Next Recommendations: {', '.join(insights.next_recommendations)}")
                    print(f"   ✨ Custom Prompt Needed: {'YES' if insights.should_try_custom_prompt else 'NO'}")
                    print(f"   🛑 Early Termination: {'YES' if insights.early_termination_recommendation else 'NO'}")
                    
                    # Apply insights for next attempts
                    if insights.early_termination_recommendation:
                        print(f"   🛑 AI recommends stopping optimization")
                        break
                    
                    if insights.should_try_custom_prompt and insights.custom_prompt_suggestion:
                        print(f"   💡 AI custom prompt ready for next attempt")
            
            # Ultra achievement check
            if val_score >= self.ultra_target:
                print(f"   🏆 ULTRA TARGET ACHIEVED!")
                break
            
            time.sleep(1)
        
        # Create final session summary
        return self._create_session_summary(prompt, category, baseline_score, baseline_fidelity,
                                          attempts, ai_decisions, ai_insight_summaries, best_attempt)
    
    def _create_session_summary(self, prompt: str, category: str, baseline_score: float, 
                               baseline_fidelity: float, attempts: List[OptimizationAttempt],
                               ai_decisions: List[AIDecision], ai_insight_summaries: List[AIInsightSummary],
                               best_attempt: Optional[OptimizationAttempt]) -> OptimizationSession:
        """Create comprehensive session summary"""
        
        best_score = best_attempt.validation_score if best_attempt else baseline_score
        session_improvement = best_score - baseline_score
        
        # Calculate AI metrics
        meaningful_decisions = sum(1 for d in ai_decisions if d.decision_type != "fallback")
        successful_decisions = sum(1 for d in ai_decisions if d.decision_success and d.decision_type != "fallback")
        ai_success_rate = (successful_decisions / meaningful_decisions) if meaningful_decisions > 0 else 0.0
        
        session = OptimizationSession(
            original_prompt=prompt,
            category=category,
            baseline_score=baseline_score,
            baseline_fidelity=baseline_fidelity,
            attempts=attempts,
            ai_decisions=ai_decisions,
            ai_insight_summaries=ai_insight_summaries,
            best_attempt=best_attempt,
            best_score=best_score,
            session_improvement=session_improvement,
            ai_made_meaningful_decisions=meaningful_decisions > 0,
            ai_success_rate=ai_success_rate,
            reached_minimum=best_score >= self.min_target,
            reached_target=best_score >= self.target,
            reached_ultra=best_score >= self.ultra_target,
            timestamp=time.time()
        )
        
        # Session summary
        print(f"\n📊 ENHANCED AI SESSION SUMMARY v5.0:")
        print(f"   📈 Best Score: {best_score:.3f} (Baseline: {baseline_score:.3f})")
        print(f"   📊 Session Improvement: {session_improvement:+.3f}")
        print(f"   🤖 AI Meaningful Decisions: {meaningful_decisions}/{len(ai_decisions)}")
        print(f"   ✅ AI Success Rate: {ai_success_rate:.1%}")
        print(f"   🧠 AI Insight Summaries: {len(ai_insight_summaries)}")
        print(f"   🎯 Targets: Min {'✅' if session.reached_minimum else '❌'} | Target {'✅' if session.reached_target else '❌'} | Ultra {'✅' if session.reached_ultra else '❌'}")
        
        return session
    
    def run_enhanced_test_suite(self, test_prompts: List[str]) -> List[OptimizationSession]:
        """Run enhanced test suite with v5.0 features"""
        
        print("🤖 ENHANCED AI OPTIMIZER v5.0 - STRONGER FEEDBACK LOOPS")
        print("=" * 80)
        print("🧠 AI Features: Insights every 3 attempts, custom prompts, pattern learning")
        print("📊 Better Decision Making: DeepSeek with enhanced prompts")
        print("🔄 Feedback Loops: Continuous learning and adaptation")
        print(f"📚 Testing {len(test_prompts)} prompts with max {self.max_attempts} attempts each")
        print("=" * 80)
        
        all_sessions = []
        
        for i, prompt in enumerate(test_prompts, 1):
            print(f"\n{'='*20} [{i}/{len(test_prompts)}] PROMPT {i} {'='*20}")
            session = self.optimize_with_enhanced_feedback(prompt)
            all_sessions.append(session)
            time.sleep(2)
        
        # Comprehensive final analysis
        self._analyze_enhanced_results(all_sessions)
        
        return all_sessions
    
    def _analyze_enhanced_results(self, sessions: List[OptimizationSession]):
        """Analyze enhanced v5.0 results"""
        
        total_sessions = len(sessions)
        meaningful_ai_sessions = sum(1 for s in sessions if s.ai_made_meaningful_decisions)
        avg_ai_success_rate = statistics.mean([s.ai_success_rate for s in sessions if s.ai_made_meaningful_decisions]) if meaningful_ai_sessions > 0 else 0.0
        
        reached_minimum = sum(1 for s in sessions if s.reached_minimum)
        reached_target = sum(1 for s in sessions if s.reached_target)
        reached_ultra = sum(1 for s in sessions if s.reached_ultra)
        
        total_ai_decisions = sum(len(s.ai_decisions) for s in sessions)
        total_meaningful_decisions = sum(len([d for d in s.ai_decisions if d.decision_type != "fallback"]) for s in sessions)
        total_successful_decisions = sum(len([d for d in s.ai_decisions if d.decision_success and d.decision_type != "fallback"]) for s in sessions)
        
        total_insights = sum(len(s.ai_insight_summaries) for s in sessions)
        
        print(f"\n🎓 ENHANCED AI OPTIMIZER v5.0 - FINAL ANALYSIS")
        print("=" * 80)
        print(f"📊 SESSION RESULTS:")
        print(f"   Total Sessions: {total_sessions}")
        print(f"   Sessions with Meaningful AI Decisions: {meaningful_ai_sessions}/{total_sessions} ({meaningful_ai_sessions/total_sessions*100:.1f}%)")
        print(f"   Average AI Success Rate: {avg_ai_success_rate:.1%}")
        print(f"   Reached Minimum (≥{self.min_target}): {reached_minimum}/{total_sessions} ({reached_minimum/total_sessions*100:.1f}%)")
        print(f"   Reached Target (≥{self.target}): {reached_target}/{total_sessions} ({reached_target/total_sessions*100:.1f}%)")
        print(f"   Reached Ultra (≥{self.ultra_target}): {reached_ultra}/{total_sessions} ({reached_ultra/total_sessions*100:.1f}%)")
        
        print(f"\n🤖 AI DECISION ANALYSIS:")
        print(f"   Total AI Decisions: {total_ai_decisions}")
        print(f"   Meaningful Decisions: {total_meaningful_decisions}")
        print(f"   Successful Decisions: {total_successful_decisions}")
        print(f"   Overall AI Decision Success Rate: {(total_successful_decisions/total_meaningful_decisions*100):.1f}%" if total_meaningful_decisions > 0 else "0.0%")
        print(f"   AI Insight Summaries Generated: {total_insights}")
        
        # Decision type breakdown
        decision_types = {}
        for session in sessions:
            for decision in session.ai_decisions:
                dt = decision.decision_type
                decision_types[dt] = decision_types.get(dt, 0) + 1
        
        print(f"\n🧠 AI DECISION TYPE BREAKDOWN:")
        for decision_type, count in decision_types.items():
            print(f"   {decision_type}: {count} times ({count/total_ai_decisions*100:.1f}%)")
        
        # Success assessment
        overall_success_rate = (total_successful_decisions / total_meaningful_decisions) if total_meaningful_decisions > 0 else 0.0
        
        if overall_success_rate >= 0.7:
            print(f"\n🎉 EXCELLENT: AI decision success rate {overall_success_rate:.1%} ≥70%!")
        elif overall_success_rate >= 0.5:
            print(f"\n🟡 GOOD: AI decision success rate {overall_success_rate:.1%} ≥50%")
        else:
            print(f"\n🔴 NEEDS IMPROVEMENT: AI decision success rate {overall_success_rate:.1%} <50%")
        
        # Save results
        results = {
            "enhanced_v5_analysis": {
                "total_sessions": total_sessions,
                "meaningful_ai_sessions": meaningful_ai_sessions,
                "avg_ai_success_rate": avg_ai_success_rate,
                "overall_ai_decision_success_rate": overall_success_rate,
                "reached_minimum": reached_minimum,
                "reached_target": reached_target,
                "reached_ultra": reached_ultra,
                "total_ai_decisions": total_ai_decisions,
                "total_meaningful_decisions": total_meaningful_decisions,
                "total_successful_decisions": total_successful_decisions,
                "total_insights": total_insights,
                "decision_type_breakdown": decision_types,
                "timestamp": time.time()
            },
            "sessions": [asdict(s) for s in sessions],
            "learned_patterns": self.learned_patterns
        }
        
        output_file = f"enhanced_ai_optimizer_v5_results_{int(time.time())}.json"
        with open(output_file, "w") as f:
            json.dump(results, f, indent=2)
        
        print(f"\n💾 Enhanced v5.0 results saved to: {output_file}")

def main():
    """Test Enhanced AI Optimizer v5.0"""
    
    test_prompts = [
        "hexagonal prism steel structure",
        "cylindrical copper pipe diameter 5cm",
        "transparent glass sphere reflection",
        "rusty metal gear mechanism",
        "elegant silk fabric draping"
    ]
    
    optimizer = EnhancedAIOptimizerV5(
        max_attempts=9,  # More attempts for better AI feedback
        min_target=0.6,
        target=0.9,
        ultra_target=0.96
    )
    
    sessions = optimizer.run_enhanced_test_suite(test_prompts)
    
    # Quick summary
    meaningful_sessions = sum(1 for s in sessions if s.ai_made_meaningful_decisions)
    avg_success_rate = statistics.mean([s.ai_success_rate for s in sessions if s.ai_made_meaningful_decisions]) if meaningful_sessions > 0 else 0.0
    
    print(f"\n🎯 ENHANCED v5.0 QUICK SUMMARY:")
    print(f"🤖 Sessions with AI Decisions: {meaningful_sessions}/{len(sessions)}")
    print(f"📈 Average AI Success Rate: {avg_success_rate:.1%}")
    print(f"🎯 Target Achievement: {sum(1 for s in sessions if s.reached_target)}/{len(sessions)}")
    print(f"🏆 Ultra Achievement: {sum(1 for s in sessions if s.reached_ultra)}/{len(sessions)}")

if __name__ == "__main__":
    main() 