#!/usr/bin/env python3
"""
Adaptive Learning Optimizer v3.0 - Self-Improving AI Recommendation Engine
Purpose: AI-powered loop that learns, modifies strategies, and generates custom prompts
to reach target scores with maximum efficiency and continuous improvement.
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
class OptimizationAttempt:
    """Single optimization attempt"""
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
    is_ai_generated: bool  # True if AI generated the prompt directly

@dataclass
class AIInsight:
    """AI-generated insight for optimization"""
    insight_type: str  # "strategy_modification", "custom_prompt", "early_termination"
    content: str
    confidence: float
    reasoning: str
    timestamp: float

@dataclass
class OptimizationSession:
    """Complete optimization session with AI insights"""
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
    session_success: bool
    ai_contributed: bool  # True if AI insights contributed to success
    timestamp: float

class AIRecommendationEngine:
    """AI-powered recommendation engine that learns and improves strategies"""
    
    def __init__(self, max_attempts_per_prompt: int = 6, minimum_score_target: float = 0.6, 
                 optimal_score_target: float = 0.9, ai_confidence_threshold: float = 0.7):
        self.ollama_url = "http://localhost:11434"
        self.model_name = "deepseek-r1:1.5b"
        self.db_path = "ai_recommendation_engine.db"
        self.max_attempts = max_attempts_per_prompt
        self.minimum_score_target = minimum_score_target
        self.optimal_score_target = optimal_score_target
        self.ai_confidence_threshold = ai_confidence_threshold
        
        self.optimization_sessions: List[OptimizationSession] = []
        
        # Dynamic strategy library - AI can modify this
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
        
        # AI-learned custom strategies (will be populated during optimization)
        self.ai_learned_strategies = {}
        
        # Pattern recognition for AI insights
        self.learned_patterns = {
            "high_baseline_strategies": ["raw", "basic_description", "simplified_description"],
            "low_baseline_strategies": ["material_focus", "enhanced_clarity", "professional_render"],
            "rescue_strategies": ["material_focus", "concrete_object", "basic_description"],
            "enhancement_strategies": ["enhanced_clarity", "professional_render", "high_quality"]
        }
        
        self.setup_database()
        self.load_ai_learned_strategies()
        
    def setup_database(self):
        """Initialize database with AI insights tracking"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Existing tables...
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS optimization_sessions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                original_prompt TEXT,
                prompt_category TEXT,
                baseline_score REAL,
                baseline_fidelity REAL,
                total_attempts INTEGER,
                best_strategy TEXT,
                best_score REAL,
                best_fidelity REAL,
                session_improvement REAL,
                reached_minimum_threshold BOOLEAN,
                reached_target_threshold BOOLEAN,
                session_success BOOLEAN,
                ai_contributed BOOLEAN,
                timestamp REAL
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS optimization_attempts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id INTEGER,
                strategy_name TEXT,
                optimized_prompt TEXT,
                validation_score REAL,
                demo_fidelity_score REAL,
                score_improvement REAL,
                fidelity_improvement REAL,
                meets_minimum_threshold BOOLEAN,
                meets_target_threshold BOOLEAN,
                is_ai_generated BOOLEAN,
                attempt_number INTEGER,
                timestamp REAL,
                FOREIGN KEY (session_id) REFERENCES optimization_sessions (id)
            )
        ''')
        
        # New AI insights table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS ai_insights (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id INTEGER,
                insight_type TEXT,
                content TEXT,
                confidence REAL,
                reasoning TEXT,
                timestamp REAL,
                FOREIGN KEY (session_id) REFERENCES optimization_sessions (id)
            )
        ''')
        
        # AI-learned strategies table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS ai_learned_strategies (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                strategy_name TEXT UNIQUE,
                strategy_template TEXT,
                category TEXT,
                success_rate REAL,
                avg_score_improvement REAL,
                usage_count INTEGER,
                learned_from_prompt TEXT,
                timestamp REAL
            )
        ''')
        
        conn.commit()
        conn.close()
        
    def load_ai_learned_strategies(self):
        """Load AI-learned strategies from previous sessions"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT strategy_name, strategy_template, success_rate, avg_score_improvement
            FROM ai_learned_strategies
            WHERE success_rate > 0.5 AND usage_count >= 2
            ORDER BY success_rate DESC
        ''')
        
        for row in cursor.fetchall():
            strategy_name, template, success_rate, avg_improvement = row
            self.ai_learned_strategies[strategy_name] = template
            print(f"📚 Loaded AI-learned strategy: {strategy_name} (success: {success_rate:.1%})")
        
        conn.close()
        
    def query_deepseek(self, system_prompt: str, user_prompt: str) -> str:
        """Query DeepSeek-R1 via Ollama"""
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
            raise Exception(f"DeepSeek query failed: {e}")
    
    def categorize_prompt(self, prompt: str) -> str:
        """Enhanced prompt categorization with AI learning"""
        system_prompt = """You are an expert at categorizing 3D model generation prompts for optimization.

Target: Reach 0.6+ (avoid zero fidelity) and ideally 0.9+ (excellent performance).

Categories:
- physical_object: Concrete items (bucket, chair, bottle)
- technical_description: Technical/geometric (cylindrical, angular, measurements)  
- abstract_artistic: Artistic concepts (sleek, reflecting, ethereal)
- fashion_clothing: Clothing items (shirt, dress, shoes)
- decorative_standard: Decorative objects (ornate, elegant, gothic)

Consider optimization potential when categorizing. Respond with ONLY the category name."""

        user_prompt = f"Categorize for optimization: '{prompt}'"
        
        try:
            response = self.query_deepseek(system_prompt, user_prompt)
            category = response.strip().lower()
            
            valid_categories = ["physical_object", "technical_description", "abstract_artistic", 
                             "fashion_clothing", "decorative_standard"]
            
            return category if category in valid_categories else "physical_object"
        except:
            return "physical_object"
    
    def generate_ai_insights(self, prompt: str, category: str, baseline_score: float, 
                           previous_attempts: List[OptimizationAttempt]) -> List[AIInsight]:
        """Generate AI insights for optimization strategy"""
        
        insights = []
        
        # Prepare context for AI
        attempts_summary = []
        for attempt in previous_attempts:
            attempts_summary.append({
                "strategy": attempt.strategy_name,
                "score": attempt.validation_score,
                "improvement": attempt.score_improvement,
                "meets_target": attempt.meets_target_threshold
            })
        
        context = {
            "prompt": prompt,
            "category": category,
            "baseline_score": baseline_score,
            "target": self.optimal_score_target,
            "minimum": self.minimum_score_target,
            "attempts_so_far": attempts_summary,
            "available_strategies": list(self.base_strategies.keys()) + list(self.ai_learned_strategies.keys())
        }
        
        # AI Insight 1: Strategy modification recommendation
        try:
            strategy_insight = self._get_strategy_modification_insight(context)
            if strategy_insight:
                insights.append(strategy_insight)
        except Exception as e:
            print(f"⚠️ Strategy insight generation failed: {e}")
        
        # AI Insight 2: Custom prompt generation (if strategies aren't working)
        if len(previous_attempts) >= 2 and all(a.validation_score < self.minimum_score_target for a in previous_attempts):
            try:
                custom_prompt_insight = self._get_custom_prompt_insight(context)
                if custom_prompt_insight:
                    insights.append(custom_prompt_insight)
            except Exception as e:
                print(f"⚠️ Custom prompt insight generation failed: {e}")
        
        # AI Insight 3: Early termination recommendation
        if baseline_score >= self.optimal_score_target:
            try:
                termination_insight = self._get_early_termination_insight(context)
                if termination_insight:
                    insights.append(termination_insight)
            except Exception as e:
                print(f"⚠️ Termination insight generation failed: {e}")
        
        return insights
    
    def _get_strategy_modification_insight(self, context: dict) -> Optional[AIInsight]:
        """Get AI insight for strategy modification"""
        
        system_prompt = f"""You are an expert optimization AI that learns and improves 3D model generation strategies.

TASK: Analyze the current optimization attempt and recommend strategy modifications.

CONTEXT:
- Prompt: "{context['prompt']}"
- Category: {context['category']}
- Baseline Score: {context['baseline_score']:.3f}
- Target: {context['target']} (excellent)
- Minimum: {context['minimum']} (avoid zero fidelity)
- Previous Attempts: {context['attempts_so_far']}
- Available Strategies: {context['available_strategies']}

ANALYSIS NEEDED:
1. Why are current strategies failing/succeeding?
2. What specific modifications could help reach the target?
3. Should we try a different category of strategies?

RESPONSE FORMAT:
If you recommend strategy modifications, respond with:
CONFIDENCE: [0.0-1.0]
REASONING: [Your analysis]
RECOMMENDATION: [Specific strategy names to try next, comma-separated]

If no modifications needed, respond with: NO_MODIFICATION"""

        user_prompt = f"Analyze and recommend strategy modifications for: '{context['prompt']}'"
        
        try:
            response = self.query_deepseek(system_prompt, user_prompt)
            
            if "NO_MODIFICATION" in response:
                return None
            
            # Parse AI response
            confidence_match = re.search(r'CONFIDENCE:\s*([0-9.]+)', response)
            reasoning_match = re.search(r'REASONING:\s*(.+?)(?=RECOMMENDATION:|$)', response, re.DOTALL)
            recommendation_match = re.search(r'RECOMMENDATION:\s*(.+)', response)
            
            if confidence_match and reasoning_match and recommendation_match:
                confidence = float(confidence_match.group(1))
                reasoning = reasoning_match.group(1).strip()
                recommendation = recommendation_match.group(1).strip()
                
                if confidence >= self.ai_confidence_threshold:
                    return AIInsight(
                        insight_type="strategy_modification",
                        content=recommendation,
                        confidence=confidence,
                        reasoning=reasoning,
                        timestamp=time.time()
                    )
        except Exception as e:
            print(f"Strategy modification insight error: {e}")
        
        return None
    
    def _get_custom_prompt_insight(self, context: dict) -> Optional[AIInsight]:
        """Get AI insight for custom prompt generation"""
        
        system_prompt = f"""You are an expert 3D model generation prompt engineer.

TASK: Generate a completely custom optimized prompt for better results.

SITUATION:
- Original: "{context['prompt']}"
- Category: {context['category']}
- Baseline: {context['baseline_score']:.3f}
- Target: {context['target']} (need to reach this!)
- Previous attempts all failed to reach minimum {context['minimum']}

FAILED ATTEMPTS: {context['attempts_so_far']}

ANALYSIS: The standard strategies aren't working. You need to create a completely new prompt that:
1. Maintains the core concept of the original
2. Uses language that will generate better 3D models
3. Incorporates successful patterns from your training

RESPONSE FORMAT:
CONFIDENCE: [0.0-1.0]
REASONING: [Why this custom prompt will work better]
CUSTOM_PROMPT: [Your completely new optimized prompt]

Only respond if you're confident (>0.7) that your custom prompt will significantly improve results."""

        user_prompt = f"Generate custom optimized prompt for: '{context['prompt']}'"
        
        try:
            response = self.query_deepseek(system_prompt, user_prompt)
            
            # Parse AI response
            confidence_match = re.search(r'CONFIDENCE:\s*([0-9.]+)', response)
            reasoning_match = re.search(r'REASONING:\s*(.+?)(?=CUSTOM_PROMPT:|$)', response, re.DOTALL)
            prompt_match = re.search(r'CUSTOM_PROMPT:\s*(.+)', response, re.DOTALL)
            
            if confidence_match and reasoning_match and prompt_match:
                confidence = float(confidence_match.group(1))
                reasoning = reasoning_match.group(1).strip()
                custom_prompt = prompt_match.group(1).strip()
                
                if confidence >= self.ai_confidence_threshold:
                    return AIInsight(
                        insight_type="custom_prompt",
                        content=custom_prompt,
                        confidence=confidence,
                        reasoning=reasoning,
                        timestamp=time.time()
                    )
        except Exception as e:
            print(f"Custom prompt insight error: {e}")
        
        return None
    
    def _get_early_termination_insight(self, context: dict) -> Optional[AIInsight]:
        """Get AI insight for early termination"""
        
        if context['baseline_score'] >= context['target']:
            return AIInsight(
                insight_type="early_termination",
                content="TERMINATE_EARLY",
                confidence=0.95,
                reasoning=f"Baseline score {context['baseline_score']:.3f} already exceeds target {context['target']}",
                timestamp=time.time()
            )
        
        return None
    
    def apply_ai_insights(self, insights: List[AIInsight], prompt: str, category: str) -> Tuple[List[str], Optional[str]]:
        """Apply AI insights to modify strategy sequence or generate custom prompt"""
        
        strategy_sequence = []
        custom_prompt = None
        
        for insight in insights:
            print(f"🤖 AI Insight ({insight.insight_type}): Confidence {insight.confidence:.2f}")
            print(f"   💭 Reasoning: {insight.reasoning}")
            
            if insight.insight_type == "strategy_modification":
                # Parse recommended strategies
                recommended_strategies = [s.strip() for s in insight.content.split(',')]
                valid_strategies = []
                
                for strategy in recommended_strategies:
                    if strategy in self.base_strategies or strategy in self.ai_learned_strategies:
                        valid_strategies.append(strategy)
                
                if valid_strategies:
                    strategy_sequence = valid_strategies
                    print(f"   🔧 Modified strategy sequence: {valid_strategies}")
            
            elif insight.insight_type == "custom_prompt":
                custom_prompt = insight.content
                print(f"   ✨ AI generated custom prompt: '{custom_prompt[:80]}{'...' if len(custom_prompt) > 80 else ''}'")
            
            elif insight.insight_type == "early_termination":
                print(f"   🛑 AI recommends early termination")
                return [], None
        
        # Fallback to default strategy sequence if AI didn't provide one
        if not strategy_sequence and not custom_prompt:
            strategy_sequence = self._get_default_strategy_sequence(category, 0.0)
        
        return strategy_sequence, custom_prompt
    
    def _get_default_strategy_sequence(self, category: str, baseline_score: float) -> List[str]:
        """Get default strategy sequence when AI doesn't provide insights"""
        
        if baseline_score < self.minimum_score_target:
            if category == "decorative_standard":
                return ["raw", "basic_description", "artistic_focus", "simplified_description"]
            elif category == "technical_description":
                return ["material_focus", "enhanced_clarity", "concrete_object", "professional_render"]
            elif category == "abstract_artistic":
                return ["geometric_focus", "current_production", "artistic_focus"]
            elif category == "fashion_clothing":
                return ["basic_description", "minimal_enhancement", "simplified_description"]
            else:  # physical_object
                return ["material_focus", "concrete_object", "enhanced_clarity"]
        else:
            return ["enhanced_clarity", "professional_render", "high_quality"]
    
    def apply_strategy(self, prompt: str, strategy: str) -> str:
        """Apply strategy (from base strategies or AI-learned strategies)"""
        if strategy in self.base_strategies:
            template = self.base_strategies[strategy]
            return template.format(prompt=prompt)
        elif strategy in self.ai_learned_strategies:
            template = self.ai_learned_strategies[strategy]
            return template.format(prompt=prompt)
        else:
            return prompt
    
    def run_validation(self, prompt: str) -> Tuple[float, float]:
        """Run production-accurate validation"""
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
    
    def optimize_with_ai_loop(self, prompt: str) -> OptimizationSession:
        """Main optimization loop with AI insights integration"""
        
        print(f"\n🧠 AI-POWERED OPTIMIZATION LOOP: '{prompt}'")
        print("=" * 80)
        print(f"🎯 Goals: Minimum {self.minimum_score_target}, Target {self.optimal_score_target}")
        
        # Step 1: Initial analysis
        category = self.categorize_prompt(prompt)
        print(f"📋 Category: {category}")
        
        print(f"🧪 Getting baseline performance...")
        baseline_val, baseline_fidelity = self.run_validation(prompt)
        print(f"   📊 Baseline: Score={baseline_val:.4f}, Fidelity={baseline_fidelity:.3f}")
        
        # Initialize session tracking
        attempts = []
        ai_insights = []
        best_attempt = None
        best_score = baseline_val
        reached_minimum = baseline_val >= self.minimum_score_target
        reached_target = baseline_val >= self.optimal_score_target
        ai_contributed = False
        
        # Step 2: AI Insights Generation
        print(f"\n🤖 Generating AI insights...")
        insights = self.generate_ai_insights(prompt, category, baseline_val, attempts)
        ai_insights.extend(insights)
        
        # Step 3: Apply AI insights
        strategy_sequence, custom_prompt = self.apply_ai_insights(insights, prompt, category)
        
        # Step 4: Early termination check
        if not strategy_sequence and not custom_prompt:
            print(f"🛑 AI recommends early termination - baseline already optimal")
            session = OptimizationSession(
                original_prompt=prompt, prompt_category=category,
                baseline_score=baseline_val, baseline_fidelity=baseline_fidelity,
                attempts=attempts, ai_insights=ai_insights, best_attempt=None,
                total_attempts=0, session_improvement=0.0,
                reached_minimum_threshold=reached_minimum,
                reached_target_threshold=reached_target,
                session_success=reached_target, ai_contributed=True,
                timestamp=time.time()
            )
            return session
        
        # Step 5: Try custom prompt first if AI generated one
        attempt_num = 0
        if custom_prompt:
            attempt_num += 1
            print(f"\n[{attempt_num}/AI] Trying AI Custom Prompt")
            print(f"   ✨ AI Generated: '{custom_prompt[:80]}{'...' if len(custom_prompt) > 80 else ''}'")
            
            opt_val, opt_fidelity = self.run_validation(custom_prompt)
            score_improvement = opt_val - baseline_val
            fidelity_improvement = opt_fidelity - baseline_fidelity
            meets_minimum = opt_val >= self.minimum_score_target
            meets_target = opt_val >= self.optimal_score_target
            
            print(f"   📊 Results: Score={opt_val:.4f} ({score_improvement:+.3f}), Fidelity={opt_fidelity:.3f}")
            print(f"   🎯 Targets: Minimum={meets_minimum} ({'✅' if meets_minimum else '❌'}), Target={meets_target} ({'✅' if meets_target else '❌'})")
            
            attempt = OptimizationAttempt(
                strategy_name="ai_custom_prompt",
                optimized_prompt=custom_prompt,
                validation_score=opt_val,
                demo_fidelity_score=opt_fidelity,
                score_improvement=score_improvement,
                fidelity_improvement=fidelity_improvement,
                attempt_number=attempt_num,
                timestamp=time.time(),
                meets_minimum_threshold=meets_minimum,
                meets_target_threshold=meets_target,
                is_ai_generated=True
            )
            attempts.append(attempt)
            
            if opt_val > best_score:
                best_score = opt_val
                best_attempt = attempt
                ai_contributed = True
                print(f"   🌟 NEW BEST SCORE: {opt_val:.4f} (AI Generated!)")
            
            if meets_target:
                reached_target = True
                ai_contributed = True
                print(f"   🏆 AI CUSTOM PROMPT REACHED TARGET! ({opt_val:.3f} >= {self.optimal_score_target})")
                
                # Learn from this success
                self._learn_from_success(prompt, custom_prompt, category, opt_val)
                
                # Early exit on target achievement
                strategy_sequence = []
            elif meets_minimum and not reached_minimum:
                reached_minimum = True
                ai_contributed = True
                print(f"   🎉 AI REACHED MINIMUM THRESHOLD! ({opt_val:.3f} >= {self.minimum_score_target})")
        
        # Step 6: Try strategy sequence
        for strategy in strategy_sequence:
            if attempt_num >= self.max_attempts:
                print(f"   🛑 Reached maximum attempts ({self.max_attempts})")
                break
            
            attempt_num += 1
            print(f"\n[{attempt_num}/{len(strategy_sequence) + (1 if custom_prompt else 0)}] Trying Strategy: {strategy}")
            
            optimized_prompt = self.apply_strategy(prompt, strategy)
            print(f"   ✨ Optimized: '{optimized_prompt[:80]}{'...' if len(optimized_prompt) > 80 else ''}'")
            
            opt_val, opt_fidelity = self.run_validation(optimized_prompt)
            score_improvement = opt_val - baseline_val
            fidelity_improvement = opt_fidelity - baseline_fidelity
            meets_minimum = opt_val >= self.minimum_score_target
            meets_target = opt_val >= self.optimal_score_target
            
            print(f"   📊 Results: Score={opt_val:.4f} ({score_improvement:+.3f}), Fidelity={opt_fidelity:.3f}")
            print(f"   🎯 Targets: Minimum={meets_minimum} ({'✅' if meets_minimum else '❌'}), Target={meets_target} ({'✅' if meets_target else '❌'})")
            
            attempt = OptimizationAttempt(
                strategy_name=strategy,
                optimized_prompt=optimized_prompt,
                validation_score=opt_val,
                demo_fidelity_score=opt_fidelity,
                score_improvement=score_improvement,
                fidelity_improvement=fidelity_improvement,
                attempt_number=attempt_num,
                timestamp=time.time(),
                meets_minimum_threshold=meets_minimum,
                meets_target_threshold=meets_target,
                is_ai_generated=False
            )
            attempts.append(attempt)
            
            if opt_val > best_score:
                best_score = opt_val
                best_attempt = attempt
                print(f"   🌟 NEW BEST SCORE: {opt_val:.4f}")
            
            if meets_target and not reached_target:
                reached_target = True
                print(f"   🏆 REACHED TARGET! ({opt_val:.3f} >= {self.optimal_score_target})")
                break
            elif meets_minimum and not reached_minimum:
                reached_minimum = True
                print(f"   🎉 REACHED MINIMUM! ({opt_val:.3f} >= {self.minimum_score_target})")
            
            # AI Re-evaluation after each attempt
            if attempt_num < self.max_attempts and not reached_target:
                print(f"   🤖 AI re-evaluating after attempt {attempt_num}...")
                new_insights = self.generate_ai_insights(prompt, category, baseline_val, attempts)
                if new_insights:
                    ai_insights.extend(new_insights)
                    new_strategies, new_custom_prompt = self.apply_ai_insights(new_insights, prompt, category)
                    
                    if new_custom_prompt and new_custom_prompt != custom_prompt:
                        print(f"   💡 AI suggests new custom prompt for next iteration")
                        custom_prompt = new_custom_prompt
                    
                    if new_strategies and new_strategies != strategy_sequence[attempt_num:]:
                        print(f"   🔧 AI modified remaining strategy sequence")
                        # Update remaining strategies
                        remaining_strategies = new_strategies[:self.max_attempts - attempt_num]
                        strategy_sequence = strategy_sequence[:attempt_num] + remaining_strategies
            
            time.sleep(1)
        
        # Final session creation
        session_improvement = best_score - baseline_val if best_attempt else 0.0
        session_success = reached_target or (reached_minimum and session_improvement > 0)
        
        session = OptimizationSession(
            original_prompt=prompt,
            prompt_category=category,
            baseline_score=baseline_val,
            baseline_fidelity=baseline_fidelity,
            attempts=attempts,
            ai_insights=ai_insights,
            best_attempt=best_attempt,
            total_attempts=len(attempts),
            session_improvement=session_improvement,
            reached_minimum_threshold=reached_minimum,
            reached_target_threshold=reached_target,
            session_success=session_success,
            ai_contributed=ai_contributed,
            timestamp=time.time()
        )
        
        print(f"\n📊 AI-POWERED SESSION SUMMARY:")
        print(f"   Total Attempts: {session.total_attempts}")
        print(f"   AI Insights Generated: {len(ai_insights)}")
        print(f"   Best Strategy: {session.best_attempt.strategy_name if session.best_attempt else 'None'}")
        print(f"   Best Score: {best_score:.4f} (Baseline: {baseline_val:.4f})")
        print(f"   Session Improvement: {session.session_improvement:+.3f}")
        print(f"   Reached Minimum (≥{self.minimum_score_target}): {'✅' if session.reached_minimum_threshold else '❌'}")
        print(f"   Reached Target (≥{self.optimal_score_target}): {'✅' if session.reached_target_threshold else '❌'}")
        print(f"   AI Contributed: {'✅' if session.ai_contributed else '❌'}")
        print(f"   Overall Success: {'✅' if session.session_success else '❌'}")
        
        # Store session and learn from results
        self._store_session(session)
        
        return session
    
    def _learn_from_success(self, original_prompt: str, successful_prompt: str, category: str, score: float):
        """Learn new strategy patterns from successful AI-generated prompts"""
        
        print(f"📚 Learning from AI success: {score:.3f}")
        
        # Extract pattern from successful prompt
        try:
            learning_prompt = f"""Analyze this successful prompt optimization and extract a reusable strategy pattern.

Original: "{original_prompt}"
Successful: "{successful_prompt}"
Category: {category}
Score: {score:.3f}

Extract a template pattern that could work for similar prompts in this category.
Respond with a template using {{prompt}} placeholder, or NO_PATTERN if no clear pattern."""
            
            pattern_response = self.query_deepseek(
                "You are an expert at extracting reusable optimization patterns.",
                learning_prompt
            )
            
            if "NO_PATTERN" not in pattern_response and "{prompt}" in pattern_response:
                # Create new learned strategy
                strategy_name = f"ai_learned_{category}_{int(time.time())}"
                self.ai_learned_strategies[strategy_name] = pattern_response.strip()
                
                print(f"   💡 Learned new strategy: {strategy_name}")
                print(f"   📝 Pattern: {pattern_response.strip()}")
                
                # Store in database
                self._store_learned_strategy(strategy_name, pattern_response.strip(), category, original_prompt)
                
        except Exception as e:
            print(f"   ⚠️ Learning extraction failed: {e}")
    
    def _store_learned_strategy(self, strategy_name: str, template: str, category: str, learned_from: str):
        """Store AI-learned strategy in database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT OR REPLACE INTO ai_learned_strategies
            (strategy_name, strategy_template, category, success_rate, avg_score_improvement, 
             usage_count, learned_from_prompt, timestamp)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ''', (strategy_name, template, category, 1.0, 0.0, 1, learned_from, time.time()))
        
        conn.commit()
        conn.close()
    
    def _store_session(self, session: OptimizationSession):
        """Store complete session with AI insights"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Store session
        cursor.execute('''
            INSERT INTO optimization_sessions 
            (original_prompt, prompt_category, baseline_score, baseline_fidelity,
             total_attempts, best_strategy, best_score, best_fidelity,
             session_improvement, reached_minimum_threshold, reached_target_threshold,
             session_success, ai_contributed, timestamp)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            session.original_prompt, session.prompt_category, session.baseline_score,
            session.baseline_fidelity, session.total_attempts,
            session.best_attempt.strategy_name if session.best_attempt else None,
            session.best_attempt.validation_score if session.best_attempt else 0.0,
            session.best_attempt.demo_fidelity_score if session.best_attempt else 0.0,
            session.session_improvement, session.reached_minimum_threshold,
            session.reached_target_threshold, session.session_success,
            session.ai_contributed, session.timestamp
        ))
        
        session_id = cursor.lastrowid
        
        # Store attempts
        for attempt in session.attempts:
            cursor.execute('''
                INSERT INTO optimization_attempts
                (session_id, strategy_name, optimized_prompt, validation_score,
                 demo_fidelity_score, score_improvement, fidelity_improvement,
                 meets_minimum_threshold, meets_target_threshold, is_ai_generated,
                 attempt_number, timestamp)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                session_id, attempt.strategy_name, attempt.optimized_prompt,
                attempt.validation_score, attempt.demo_fidelity_score,
                attempt.score_improvement, attempt.fidelity_improvement,
                attempt.meets_minimum_threshold, attempt.meets_target_threshold,
                attempt.is_ai_generated, attempt.attempt_number, attempt.timestamp
            ))
        
        # Store AI insights
        for insight in session.ai_insights:
            cursor.execute('''
                INSERT INTO ai_insights
                (session_id, insight_type, content, confidence, reasoning, timestamp)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (
                session_id, insight.insight_type, insight.content,
                insight.confidence, insight.reasoning, insight.timestamp
            ))
        
        conn.commit()
        conn.close()
    
    def run_ai_learning_session(self, test_prompts: List[str]):
        """Run complete AI learning session"""
        
        print("🧠 AI-POWERED SELF-IMPROVING OPTIMIZATION ENGINE v3.0")
        print("=" * 80)
        print(f"📚 Testing {len(test_prompts)} prompts with AI loop optimization")
        print(f"🎯 Minimum Target: {self.minimum_score_target} | Optimal Target: {self.optimal_score_target}")
        print(f"🤖 AI Confidence Threshold: {self.ai_confidence_threshold}")
        print(f"🔄 Max attempts per prompt: {self.max_attempts}")
        print("=" * 80)
        
        all_sessions = []
        
        for i, prompt in enumerate(test_prompts, 1):
            print(f"\n[{i}/{len(test_prompts)}] Processing prompt {i}")
            
            session = self.optimize_with_ai_loop(prompt)
            all_sessions.append(session)
            
            time.sleep(2)
        
        # Final comprehensive analysis
        self._run_final_ai_analysis(all_sessions)
        
        return all_sessions
    
    def _run_final_ai_analysis(self, sessions: List[OptimizationSession]):
        """Comprehensive final analysis of AI-powered optimization"""
        
        print(f"\n🎓 FINAL AI-POWERED OPTIMIZATION ANALYSIS")
        print("=" * 80)
        
        total_sessions = len(sessions)
        ai_contributed_sessions = sum(1 for s in sessions if s.ai_contributed)
        reached_minimum = sum(1 for s in sessions if s.reached_minimum_threshold)
        reached_target = sum(1 for s in sessions if s.reached_target_threshold)
        total_insights = sum(len(s.ai_insights) for s in sessions)
        ai_generated_attempts = sum(1 for s in sessions for a in s.attempts if a.is_ai_generated)
        
        ai_contribution_rate = (ai_contributed_sessions / total_sessions) * 100
        minimum_rate = (reached_minimum / total_sessions) * 100
        target_rate = (reached_target / total_sessions) * 100
        
        print(f"📊 AI-POWERED SESSION STATISTICS:")
        print(f"   Total Sessions: {total_sessions}")
        print(f"   AI Contributed to Success: {ai_contributed_sessions}/{total_sessions} ({ai_contribution_rate:.1f}%)")
        print(f"   Reached Minimum (≥{self.minimum_score_target}): {reached_minimum}/{total_sessions} ({minimum_rate:.1f}%)")
        print(f"   Reached Target (≥{self.optimal_score_target}): {reached_target}/{total_sessions} ({target_rate:.1f}%)")
        print(f"   Total AI Insights Generated: {total_insights}")
        print(f"   AI-Generated Attempts: {ai_generated_attempts}")
        print(f"   AI-Learned Strategies: {len(self.ai_learned_strategies)}")
        
        # AI insights analysis
        insight_types = {}
        for session in sessions:
            for insight in session.ai_insights:
                insight_types[insight.insight_type] = insight_types.get(insight.insight_type, 0) + 1
        
        print(f"\n🤖 AI INSIGHTS BREAKDOWN:")
        for insight_type, count in insight_types.items():
            print(f"   {insight_type}: {count} times")
        
        # Success attribution
        ai_successes = sum(1 for s in sessions if s.session_success and s.ai_contributed)
        traditional_successes = sum(1 for s in sessions if s.session_success and not s.ai_contributed)
        
        print(f"\n🏆 SUCCESS ATTRIBUTION:")
        print(f"   AI-Contributed Successes: {ai_successes}")
        print(f"   Traditional Strategy Successes: {traditional_successes}")
        
        if ai_successes > 0:
            print(f"   🎉 AI is actively improving optimization results!")
        
        # Save comprehensive results
        results = {
            "ai_powered_analysis": {
                "total_sessions": total_sessions,
                "ai_contribution_rate": ai_contribution_rate,
                "minimum_rate": minimum_rate,
                "target_rate": target_rate,
                "total_insights": total_insights,
                "ai_generated_attempts": ai_generated_attempts,
                "ai_learned_strategies_count": len(self.ai_learned_strategies),
                "insight_breakdown": insight_types,
                "ai_successes": ai_successes,
                "traditional_successes": traditional_successes,
                "timestamp": time.time()
            },
            "sessions": [asdict(s) for s in sessions],
            "ai_learned_strategies": self.ai_learned_strategies
        }
        
        output_file = f"ai_powered_optimization_{int(time.time())}.json"
        with open(output_file, "w") as f:
            json.dump(results, f, indent=2)
        
        print(f"\n💾 AI optimization results saved to: {output_file}")

def main():
    """Run AI-powered self-improving optimization"""
    
    test_prompts = [
        # Known challenging prompts
        "hexagonal prism steel structure",
        "cylindrical copper pipe diameter 5cm", 
        "transparent glass sphere reflection",
        "rusty metal gear mechanism",
        "elegant silk fabric draping",
        "ornate gothic candelabra silver",
        "modern minimalist chair design",
        "abstract crystalline formation"
    ]
    
    # Initialize AI recommendation engine
    engine = AIRecommendationEngine(
        max_attempts_per_prompt=6,
        minimum_score_target=0.6,
        optimal_score_target=0.9,
        ai_confidence_threshold=0.7
    )
    
    # Run AI-powered learning session
    sessions = engine.run_ai_learning_session(test_prompts)
    
    # Final summary
    reached_target = sum(1 for s in sessions if s.reached_target_threshold)
    ai_contributed = sum(1 for s in sessions if s.ai_contributed)
    
    print(f"\n🎯 AI-POWERED OPTIMIZATION COMPLETE!")
    print(f"🏆 Reached Target (≥0.9): {reached_target}/{len(sessions)} prompts")
    print(f"🤖 AI Contributed: {ai_contributed}/{len(sessions)} sessions")
    print(f"📈 AI Success Rate: {(ai_contributed / len(sessions)) * 100:.1f}%")

if __name__ == "__main__":
    main() 
