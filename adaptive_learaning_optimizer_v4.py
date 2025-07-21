#!/usr/bin/env python3
"""
Adaptive Learning Optimizer v4.0 Enhanced - Best of v3 + v4 Improvements
Purpose: Combines v3's reliable AI parsing with v4's enhanced tracking and decision framework.
Features: Proper AI contribution tracking, visible decisions, learning, custom prompt autonomy.
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

@dataclass
class AIDecisionOutcome:
    """Track AI decision outcomes for proper contribution analysis"""
    decision_made: bool
    decision_type: str  # "custom_prompt", "strategy_modification", "early_termination", "continue"
    content: str  # The actual decision content
    reasoning: str
    confidence: float
    led_to_improvement: bool  # Updated after validation
    improvement_amount: float  # Score improvement achieved

@dataclass
class VisibleLearning:
    """Visible AI learning between attempts"""
    after_attempt: int
    observation: str
    strategy_effectiveness_update: Dict[str, float]
    decision_influence: str
    confidence_change: float

@dataclass
class OptimizationAttempt:
    """Enhanced attempt with AI decision tracking"""
    strategy_name: str
    optimized_prompt: str
    validation_score: float
    demo_fidelity_score: float
    score_improvement: float
    fidelity_improvement: float
    attempt_number: int
    timestamp: float
    meets_minimum_threshold: bool
    meets_target_threshold: bool
    meets_ultra_threshold: bool  # ≥0.96
    ai_decision: Optional[AIDecisionOutcome]
    visible_learning: Optional[VisibleLearning]
    is_ai_generated: bool

@dataclass
class AIInsight:
    """AI-generated insight (keeping v3's successful format)"""
    insight_type: str
    content: str
    confidence: float
    reasoning: str
    timestamp: float

@dataclass
class OptimizationSession:
    """Enhanced session with proper AI contribution tracking"""
    original_prompt: str
    prompt_category: str
    baseline_score: float
    baseline_fidelity: float
    attempts: List[OptimizationAttempt]
    ai_insights: List[AIInsight]
    best_attempt: Optional[OptimizationAttempt]
    total_attempts: int
    session_improvement: float
    reached_minimum_threshold: bool
    reached_target_threshold: bool
    reached_ultra_threshold: bool
    session_success: bool
    # Enhanced AI tracking
    ai_decisions_made: int
    ai_decisions_that_improved: int
    ai_contribution_rate: float  # % of AI decisions that led to improvement
    ai_contributed_to_best_result: bool
    visible_learning_updates: int
    timestamp: float

class EnhancedAIOptimizer:
    """Enhanced v4.0 with v3's reliability + v4's improvements"""
    
    def __init__(self, max_attempts: int = 6, min_target: float = 0.6, 
                 target: float = 0.9, ultra_target: float = 0.96):
        self.ollama_url = "http://localhost:11434"
        self.model_name = "deepseek-r1:1.5b"
        self.db_path = "enhanced_ai_optimizer_v4.db"
        self.max_attempts = max_attempts
        self.min_target = min_target
        self.target = target
        self.ultra_target = ultra_target
        
        # Strategy library (from v3, proven to work)
        self.base_strategies = {
            "raw": "{prompt}",
            "material_focus": "wbgmsst, solid {prompt} object 3D, white background",
            "geometric_focus": "wbgmsst, {prompt} geometric 3D model, white background", 
            "basic_description": "3D model of {prompt}",
            "current_production": "wbgmsst, {prompt} 3D isometric accurate, white background",
            "enhanced_clarity": "wbgmsst, detailed 3D {prompt} model, accurate geometry, white background",
            "concrete_object": "wbgmsst, {prompt} as 3D object, realistic proportions, white background",
            "minimal_enhancement": "{prompt}, 3D object",
            "simplified_description": "simple 3D {prompt}",
            "artistic_focus": "wbgmsst, artistic {prompt} sculpture, clean design, white background",
            "professional_render": "wbgmsst, professional 3D render of {prompt}, studio lighting, white background",
            "high_quality": "wbgmsst, high quality 3D model {prompt}, detailed textures, white background"
        }
        
        # AI learning state
        self.ai_strategy_effectiveness = {s: 0.5 for s in self.base_strategies.keys()}
        self.ai_custom_prompt_success_rate = 0.3
        self.ai_learned_strategies = {}
        
        self.setup_database()
    
    def setup_database(self):
        """Setup enhanced database schema"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS enhanced_sessions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                original_prompt TEXT,
                category TEXT,
                baseline_score REAL,
                best_score REAL,
                session_improvement REAL,
                total_attempts INTEGER,
                ai_decisions_made INTEGER,
                ai_decisions_that_improved INTEGER,
                ai_contribution_rate REAL,
                ai_contributed_to_best_result BOOLEAN,
                visible_learning_updates INTEGER,
                reached_minimum BOOLEAN,
                reached_target BOOLEAN,
                reached_ultra BOOLEAN,
                timestamp REAL
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS ai_decisions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id INTEGER,
                attempt_number INTEGER,
                decision_type TEXT,
                content TEXT,
                reasoning TEXT,
                confidence REAL,
                led_to_improvement BOOLEAN,
                improvement_amount REAL,
                timestamp REAL,
                FOREIGN KEY (session_id) REFERENCES enhanced_sessions (id)
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS visible_learning (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id INTEGER,
                after_attempt INTEGER,
                observation TEXT,
                decision_influence TEXT,
                confidence_change REAL,
                timestamp REAL,
                FOREIGN KEY (session_id) REFERENCES enhanced_sessions (id)
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def query_deepseek(self, system_prompt: str, user_prompt: str) -> str:
        """Query DeepSeek (using v3's reliable method)"""
        data = {
            "model": self.model_name,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            "stream": False
        }
        
        try:
            response = requests.post(f"{self.ollama_url}/api/chat", json=data, timeout=60)
            response.raise_for_status()
            result = response.json()
            return result["message"]["content"]
        except Exception as e:
            return f"ERROR: {e}"
    
    def categorize_prompt(self, prompt: str) -> str:
        """Categorize prompt (v3's working method)"""
        system_prompt = """You are an expert at categorizing 3D model generation prompts for optimization.

Categories:
- physical_object: Concrete items (bucket, chair, bottle)
- technical_description: Technical/geometric (cylindrical, angular, measurements)  
- abstract_artistic: Artistic concepts (sleek, reflecting, ethereal)
- decorative_standard: Decorative objects (ornate, elegant, gothic)

Respond with ONLY the category name."""

        try:
            response = self.query_deepseek(system_prompt, f"Categorize: '{prompt}'")
            if "ERROR:" in response:
                return "physical_object"
            
            valid_categories = ["physical_object", "technical_description", "abstract_artistic", "decorative_standard"]
            category = response.strip().lower()
            return category if category in valid_categories else "physical_object"
        except:
            return "physical_object"
    
    def make_ai_decision(self, prompt: str, category: str, baseline_score: float, 
                        attempts: List[OptimizationAttempt]) -> AIDecisionOutcome:
        """Enhanced AI decision making with visible choices"""
        
        # Prepare context
        attempts_summary = []
        for attempt in attempts:
            attempts_summary.append({
                "attempt": attempt.attempt_number,
                "strategy": attempt.strategy_name,
                "score": attempt.validation_score,
                "improvement": attempt.score_improvement,
                "ai_generated": attempt.is_ai_generated
            })
        
        context = {
            "prompt": prompt,
            "category": category,
            "baseline": baseline_score,
            "targets": {"min": self.min_target, "target": self.target, "ultra": self.ultra_target},
            "attempt_number": len(attempts) + 1,
            "max_attempts": self.max_attempts,
            "attempts": attempts_summary,
            "strategies": list(self.base_strategies.keys()),
            "ai_effectiveness": self.ai_strategy_effectiveness
        }
        
        # AI Decision System Prompt (flexible like v3, but enhanced)
        system_prompt = f"""You are an expert AI optimizer with FULL AUTONOMY over optimization decisions.

CURRENT SITUATION:
- Prompt: "{prompt}" (Category: {category})
- Baseline: {baseline_score:.3f}
- Attempt: {len(attempts) + 1}/{self.max_attempts}
- Targets: Min {self.min_target} | Target {self.target} | Ultra {self.ultra_target}

PREVIOUS ATTEMPTS: {attempts_summary}

YOUR DECISION OPTIONS:
1. WRITE_CUSTOM_PROMPT: Create your own optimized prompt
2. MODIFY_STRATEGIES: Choose specific strategies to try
3. EARLY_STOP: Stop if baseline is already optimal or situation is hopeless
4. CONTINUE_DEFAULT: Use standard optimization approach

DECISION ANALYSIS:
- If baseline ≥ {self.ultra_target}: Consider EARLY_STOP or custom enhancement
- If baseline ≥ {self.target}: Try WRITE_CUSTOM_PROMPT for ultra-optimal
- If strategies are failing: Try WRITE_CUSTOM_PROMPT
- If making progress: MODIFY_STRATEGIES or CONTINUE_DEFAULT

Make your decision and explain your reasoning. Be creative and decisive!"""

        user_prompt = f"Make optimization decision for attempt {len(attempts) + 1}: '{prompt}'"
        
        try:
            ai_response = self.query_deepseek(system_prompt, user_prompt)
            
            if "ERROR:" in ai_response:
                return AIDecisionOutcome(
                    decision_made=False,
                    decision_type="continue_default",
                    content="continue",
                    reasoning=f"AI error: {ai_response}",
                    confidence=0.1,
                    led_to_improvement=False,
                    improvement_amount=0.0
                )
            
            # Flexible parsing (like v3) - look for key indicators
            decision_type = "continue_default"
            content = "continue"
            reasoning = ai_response[:200]  # First 200 chars as reasoning
            confidence = 0.5
            
            # Parse decision type from response
            response_lower = ai_response.lower()
            if "write_custom_prompt" in response_lower or "custom prompt" in response_lower:
                decision_type = "custom_prompt"
                # Extract custom prompt if provided
                custom_match = re.search(r'(?:custom prompt|prompt):\s*["\']?([^"\'\n]+)', ai_response, re.IGNORECASE)
                if custom_match:
                    content = custom_match.group(1).strip()
                else:
                    content = "custom_prompt_request"
            elif "modify_strategies" in response_lower or "strategies" in response_lower:
                decision_type = "strategy_modification"
                # Extract strategy names mentioned
                mentioned_strategies = []
                for strategy in self.base_strategies.keys():
                    if strategy in response_lower:
                        mentioned_strategies.append(strategy)
                content = ",".join(mentioned_strategies) if mentioned_strategies else "enhanced_clarity,professional_render"
            elif "early_stop" in response_lower or "stop" in response_lower:
                decision_type = "early_termination"
                content = "early_stop"
            
            # Extract confidence if mentioned
            conf_match = re.search(r'confidence[:\s]*([0-9.]+)', response_lower)
            if conf_match:
                try:
                    confidence = float(conf_match.group(1))
                    if confidence > 1.0:  # Handle percentage
                        confidence = confidence / 100.0
                except:
                    confidence = 0.5
            
            return AIDecisionOutcome(
                decision_made=True,
                decision_type=decision_type,
                content=content,
                reasoning=reasoning,
                confidence=confidence,
                led_to_improvement=False,  # Will be updated after validation
                improvement_amount=0.0
            )
            
        except Exception as e:
            print(f"❌ AI decision error: {e}")
            return AIDecisionOutcome(
                decision_made=False,
                decision_type="continue_default",
                content="continue",
                reasoning=f"Exception: {e}",
                confidence=0.1,
                led_to_improvement=False,
                improvement_amount=0.0
            )
    
    def execute_ai_decision(self, decision: AIDecisionOutcome, prompt: str) -> Tuple[str, str]:
        """Execute AI decision and return strategy_name and optimized_prompt"""
        
        if decision.decision_type == "early_termination":
            return "early_stop", prompt
        
        elif decision.decision_type == "custom_prompt":
            if decision.content == "custom_prompt_request":
                # AI wants to write custom prompt - request it
                custom_prompt = self._request_custom_prompt(prompt)
                return "ai_custom_prompt", custom_prompt
            else:
                # AI already provided the custom prompt
                return "ai_custom_prompt", decision.content
        
        elif decision.decision_type == "strategy_modification":
            # AI specified strategies
            strategies = [s.strip() for s in decision.content.split(',')]
            valid_strategies = [s for s in strategies if s in self.base_strategies]
            if valid_strategies:
                strategy = valid_strategies[0]  # Use first strategy
                return strategy, self.base_strategies[strategy].format(prompt=prompt)
        
        # Default: use best strategy based on learned effectiveness
        best_strategy = max(self.ai_strategy_effectiveness, key=self.ai_strategy_effectiveness.get)
        return best_strategy, self.base_strategies[best_strategy].format(prompt=prompt)
    
    def _request_custom_prompt(self, original_prompt: str) -> str:
        """Request AI to write a custom prompt"""
        
        system_prompt = f"""You are an expert 3D model generation prompt engineer.

TASK: Write an optimized prompt for better 3D model generation.

ORIGINAL: "{original_prompt}"
TARGETS: Reach {self.min_target}+ (avoid zero), ideally {self.target}+ (excellent)

REQUIREMENTS:
- Keep the core concept intact
- Use 3D modeling terminology
- Add quality/detail descriptors
- Include rendering hints
- Target significant score improvement

Respond with ONLY the optimized prompt, nothing else."""

        user_prompt = f"Create optimized prompt for: '{original_prompt}'"
        
        try:
            response = self.query_deepseek(system_prompt, user_prompt)
            if "ERROR:" not in response and len(response.strip()) > 10:
                return response.strip()
        except:
            pass
        
        # Fallback to enhanced version
        return f"wbgmsst, detailed 3D {original_prompt} model, high quality rendering, professional lighting, white background"
    
    def generate_visible_learning(self, attempt: OptimizationAttempt, 
                                 all_attempts: List[OptimizationAttempt]) -> Optional[VisibleLearning]:
        """Generate visible learning after each attempt"""
        
        if not all_attempts:
            return None
        
        strategy = attempt.strategy_name
        improvement = attempt.score_improvement
        
        # Update strategy effectiveness based on result
        effectiveness_updates = {}
        if strategy in self.ai_strategy_effectiveness:
            old_effectiveness = self.ai_strategy_effectiveness[strategy]
            
            if improvement > 0:
                # Success - increase effectiveness
                new_effectiveness = min(1.0, old_effectiveness + 0.1)
                observation = f"{strategy} succeeded (+{improvement:.3f}) - effectiveness increased"
                confidence_change = +0.1
                decision_influence = f"Will prefer {strategy} and similar strategies"
            else:
                # Failure - decrease effectiveness
                new_effectiveness = max(0.0, old_effectiveness - 0.05)
                observation = f"{strategy} failed ({improvement:+.3f}) - effectiveness decreased"
                confidence_change = -0.05
                decision_influence = f"Will avoid {strategy}, try alternative approaches"
            
            self.ai_strategy_effectiveness[strategy] = new_effectiveness
            effectiveness_updates[strategy] = new_effectiveness - old_effectiveness
        else:
            observation = f"Unknown strategy {strategy} - no learning update"
            confidence_change = 0.0
            decision_influence = "No strategy preference change"
        
        return VisibleLearning(
            after_attempt=attempt.attempt_number,
            observation=observation,
            strategy_effectiveness_update=effectiveness_updates,
            decision_influence=decision_influence,
            confidence_change=confidence_change
        )
    
    def run_validation(self, prompt: str) -> Tuple[float, float]:
        """Run validation (same as v3)"""
        try:
            cmd = [sys.executable, "subnet_accurate_validator.py", prompt]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
            
            if result.returncode != 0:
                return 0.0, 0.0
            
            with open("subnet_validation_results.json", 'r') as f:
                data = json.load(f)
                return data.get("validation_engine_score", 0.0), data.get("demo_fidelity_score", 0.0)
                
        except Exception as e:
            print(f"❌ Validation error: {e}")
            return 0.0, 0.0
    
    def optimize_with_enhanced_ai(self, prompt: str) -> OptimizationSession:
        """Main optimization with enhanced AI tracking"""
        
        print(f"\n🤖 ENHANCED AI OPTIMIZER v4.0: '{prompt}'")
        print("=" * 70)
        print(f"🎯 Targets: Min {self.min_target} | Target {self.target} | Ultra {self.ultra_target}")
        
        # Initial setup
        category = self.categorize_prompt(prompt)
        print(f"📋 Category: {category}")
        
        baseline_score, baseline_fidelity = self.run_validation(prompt)
        print(f"📊 Baseline: {baseline_score:.3f}")
        
        # Initialize tracking
        attempts = []
        ai_insights = []
        best_attempt = None
        best_score = baseline_score
        ai_decisions_made = 0
        ai_decisions_that_improved = 0
        visible_learning_updates = 0
        ai_contributed_to_best = False
        
        # Goal assessment
        if baseline_score >= self.ultra_target:
            print(f"🏆 BASELINE ALREADY ULTRA-OPTIMAL!")
        elif baseline_score >= self.target:
            print(f"🎯 BASELINE GOOD - PUSHING TO ULTRA")
        else:
            print(f"🔧 BASELINE NEEDS IMPROVEMENT")
        
        # Optimization loop
        for attempt_num in range(1, self.max_attempts + 1):
            print(f"\n🔄 ATTEMPT {attempt_num}/{self.max_attempts}")
            
            # AI Decision Making (Enhanced)
            print(f"🤖 AI making optimization decision...")
            ai_decision = self.make_ai_decision(prompt, category, baseline_score, attempts)
            
            if ai_decision.decision_made:
                ai_decisions_made += 1
                print(f"   🧠 Decision: {ai_decision.decision_type}")
                print(f"   💭 Reasoning: {ai_decision.reasoning[:80]}...")
                print(f"   🎯 Confidence: {ai_decision.confidence:.2f}")
                print(f"   📝 Content: {ai_decision.content[:60]}...")
            
            # Early termination check
            if ai_decision.decision_type == "early_termination":
                print(f"   🛑 AI Early Termination: {ai_decision.reasoning}")
                break
            
            # Execute decision
            strategy_name, optimized_prompt = self.execute_ai_decision(ai_decision, prompt)
            print(f"   🔧 Executing: {strategy_name}")
            print(f"   ✨ Prompt: '{optimized_prompt[:60]}{'...' if len(optimized_prompt) > 60 else ''}'")
            
            # Validate
            val_score, val_fidelity = self.run_validation(optimized_prompt)
            score_improvement = val_score - baseline_score
            fidelity_improvement = val_fidelity - baseline_fidelity
            
            # Update AI decision outcome
            if score_improvement > 0:
                ai_decision.led_to_improvement = True
                ai_decision.improvement_amount = score_improvement
                ai_decisions_that_improved += 1
            
            print(f"   📊 Result: {val_score:.3f} ({score_improvement:+.3f})")
            print(f"   🎯 Min {'✅' if val_score >= self.min_target else '❌'} | Target {'✅' if val_score >= self.target else '❌'} | Ultra {'✅' if val_score >= self.ultra_target else '❌'}")
            print(f"   🤖 AI Improved: {'✅' if ai_decision.led_to_improvement else '❌'}")
            
            # Update best score tracking
            if val_score > best_score:
                best_score = val_score
                print(f"   🌟 NEW BEST SCORE!")
                if ai_decision.decision_made:
                    ai_contributed_to_best = True
            
            # Generate visible learning
            visible_learning = self.generate_visible_learning(OptimizationAttempt(
                strategy_name=strategy_name,
                optimized_prompt=optimized_prompt,
                validation_score=val_score,
                demo_fidelity_score=val_fidelity,
                score_improvement=score_improvement,
                fidelity_improvement=fidelity_improvement,
                attempt_number=attempt_num,
                timestamp=time.time(),
                meets_minimum_threshold=val_score >= self.min_target,
                meets_target_threshold=val_score >= self.target,
                meets_ultra_threshold=val_score >= self.ultra_target,
                ai_decision=ai_decision,
                visible_learning=None,
                is_ai_generated=strategy_name == "ai_custom_prompt"
            ), attempts)
            
            if visible_learning:
                visible_learning_updates += 1
                print(f"   📚 AI Learning: {visible_learning.observation}")
                print(f"      🔧 Next influence: {visible_learning.decision_influence}")
            
            # Create attempt record
            attempt = OptimizationAttempt(
                strategy_name=strategy_name,
                optimized_prompt=optimized_prompt,
                validation_score=val_score,
                demo_fidelity_score=val_fidelity,
                score_improvement=score_improvement,
                fidelity_improvement=fidelity_improvement,
                attempt_number=attempt_num,
                timestamp=time.time(),
                meets_minimum_threshold=val_score >= self.min_target,
                meets_target_threshold=val_score >= self.target,
                meets_ultra_threshold=val_score >= self.ultra_target,
                ai_decision=ai_decision,
                visible_learning=visible_learning,
                is_ai_generated=strategy_name == "ai_custom_prompt"
            )
            attempts.append(attempt)
            
            if attempt.best_attempt:
                best_attempt = attempt
            
            # Check for ultra achievement
            if val_score >= self.ultra_target:
                print(f"   🏆 ULTRA TARGET ACHIEVED!")
                break
            
            time.sleep(1)
        
        # Calculate AI metrics
        ai_contribution_rate = (ai_decisions_that_improved / ai_decisions_made) if ai_decisions_made > 0 else 0.0
        session_improvement = best_score - baseline_score
        session_success = (best_score >= self.min_target and session_improvement > 0) or best_score >= self.target
        
        # Create session
        session = OptimizationSession(
            original_prompt=prompt,
            prompt_category=category,
            baseline_score=baseline_score,
            baseline_fidelity=baseline_fidelity,
            attempts=attempts,
            ai_insights=ai_insights,
            best_attempt=best_attempt,
            total_attempts=len(attempts),
            session_improvement=session_improvement,
            reached_minimum_threshold=best_score >= self.min_target,
            reached_target_threshold=best_score >= self.target,
            reached_ultra_threshold=best_score >= self.ultra_target,
            session_success=session_success,
            ai_decisions_made=ai_decisions_made,
            ai_decisions_that_improved=ai_decisions_that_improved,
            ai_contribution_rate=ai_contribution_rate,
            ai_contributed_to_best_result=ai_contributed_to_best,
            visible_learning_updates=visible_learning_updates,
            timestamp=time.time()
        )
        
        # Session summary
        print(f"\n📊 ENHANCED AI SESSION SUMMARY:")
        print(f"   Best Score: {best_score:.3f} (Baseline: {baseline_score:.3f})")
        print(f"   Session Improvement: {session_improvement:+.3f}")
        print(f"   AI Decisions Made: {ai_decisions_made}")
        print(f"   AI Decisions That Improved: {ai_decisions_that_improved}")
        print(f"   🤖 AI Contribution Rate: {ai_contribution_rate:.1%}")
        print(f"   🧠 AI Contributed to Best: {'✅' if ai_contributed_to_best else '❌'}")
        print(f"   📚 Learning Updates: {visible_learning_updates}")
        print(f"   Targets: Min {'✅' if session.reached_minimum_threshold else '❌'} | Target {'✅' if session.reached_target_threshold else '❌'} | Ultra {'✅' if session.reached_ultra_threshold else '❌'}")
        
        return session
    
    def run_enhanced_test_suite(self, test_prompts: List[str]) -> List[OptimizationSession]:
        """Run enhanced test suite"""
        
        print("🤖 ENHANCED AI OPTIMIZER v4.0 - Best of v3 + v4")
        print("=" * 70)
        print("✅ Features: v3's reliable parsing + v4's enhanced tracking")
        print("🧠 AI Autonomy: Custom prompts, strategy selection, early termination")
        print("📊 Proper Contribution Tracking: Measures actual AI impact")
        print("👁️ Visible Learning: Real-time strategy effectiveness updates")
        print(f"📚 Testing {len(test_prompts)} prompts")
        print("=" * 70)
        
        all_sessions = []
        
        for i, prompt in enumerate(test_prompts, 1):
            print(f"\n[{i}/{len(test_prompts)}] Processing prompt {i}")
            session = self.optimize_with_enhanced_ai(prompt)
            all_sessions.append(session)
            time.sleep(2)
        
        # Final analysis
        total_sessions = len(all_sessions)
        ai_contributed_sessions = sum(1 for s in all_sessions if s.ai_decisions_that_improved > 0)
        avg_ai_contribution = statistics.mean([s.ai_contribution_rate for s in all_sessions])
        reached_minimum = sum(1 for s in all_sessions if s.reached_minimum_threshold)
        reached_target = sum(1 for s in all_sessions if s.reached_target_threshold)
        reached_ultra = sum(1 for s in all_sessions if s.reached_ultra_threshold)
        total_ai_decisions = sum(s.ai_decisions_made for s in all_sessions)
        total_improvements = sum(s.ai_decisions_that_improved for s in all_sessions)
        
        print(f"\n🎓 ENHANCED AI OPTIMIZER ANALYSIS")
        print("=" * 70)
        print(f"📊 RESULTS:")
        print(f"   Total Sessions: {total_sessions}")
        print(f"   🤖 Sessions with AI Improvements: {ai_contributed_sessions}/{total_sessions} ({ai_contributed_sessions/total_sessions*100:.1f}%)")
        print(f"   📈 Average AI Contribution Rate: {avg_ai_contribution:.1%}")
        print(f"   🧠 Total AI Decisions: {total_ai_decisions}")
        print(f"   ✅ AI Decisions That Improved: {total_improvements}")
        print(f"   🎯 Reached Minimum: {reached_minimum}/{total_sessions} ({reached_minimum/total_sessions*100:.1f}%)")
        print(f"   🎯 Reached Target: {reached_target}/{total_sessions} ({reached_target/total_sessions*100:.1f}%)")
        print(f"   🏆 Reached Ultra: {reached_ultra}/{total_sessions} ({reached_ultra/total_sessions*100:.1f}%)")
        
        # Success assessment
        overall_ai_success_rate = (total_improvements / total_ai_decisions) if total_ai_decisions > 0 else 0.0
        
        if overall_ai_success_rate >= 0.8:
            print(f"\n🎉 EXCELLENT: Overall AI success rate {overall_ai_success_rate:.1%} ≥80%!")
        elif overall_ai_success_rate >= 0.6:
            print(f"\n🟡 GOOD: Overall AI success rate {overall_ai_success_rate:.1%} ≥60%")
        else:
            print(f"\n🔴 IMPROVEMENT NEEDED: Overall AI success rate {overall_ai_success_rate:.1%} <60%")
        
        # Save results
        results = {
            "enhanced_ai_analysis": {
                "total_sessions": total_sessions,
                "ai_contributed_sessions": ai_contributed_sessions,
                "avg_ai_contribution_rate": avg_ai_contribution,
                "overall_ai_success_rate": overall_ai_success_rate,
                "total_ai_decisions": total_ai_decisions,
                "total_improvements": total_improvements,
                "reached_minimum": reached_minimum,
                "reached_target": reached_target,
                "reached_ultra": reached_ultra,
                "timestamp": time.time()
            },
            "sessions": [asdict(s) for s in all_sessions]
        }
        
        output_file = f"enhanced_ai_optimizer_results_{int(time.time())}.json"
        with open(output_file, "w") as f:
            json.dump(results, f, indent=2)
        
        print(f"\n💾 Results saved to: {output_file}")
        
        return all_sessions

def main():
    """Test enhanced AI optimizer - best of v3 + v4"""
    
    test_prompts = [
        "hexagonal prism steel structure",
        "cylindrical copper pipe diameter 5cm",
        "transparent glass sphere reflection", 
        "rusty metal gear mechanism",
        "elegant silk fabric draping"
    ]
    
    optimizer = EnhancedAIOptimizer(
        max_attempts=4,  # Shorter for testing
        min_target=0.6,
        target=0.9,
        ultra_target=0.96
    )
    
    sessions = optimizer.run_enhanced_test_suite(test_prompts)
    
    # Quick summary
    overall_ai_success = sum(s.ai_decisions_that_improved for s in sessions) / sum(s.ai_decisions_made for s in sessions) if sum(s.ai_decisions_made for s in sessions) > 0 else 0.0
    
    print(f"\n🎯 ENHANCED v4.0 SUMMARY:")
    print(f"🤖 Overall AI Success Rate: {overall_ai_success:.1%}")
    print(f"📈 Target Achievement: {sum(1 for s in sessions if s.reached_target_threshold)}/{len(sessions)}")
    print(f"🏆 Ultra Achievement: {sum(1 for s in sessions if s.reached_ultra_threshold)}/{len(sessions)}")

if __name__ == "__main__":
    main() 
