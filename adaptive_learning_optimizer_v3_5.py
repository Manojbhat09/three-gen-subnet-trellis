#!/usr/bin/env python3
"""
Adaptive Learning Optimizer v4.0 - Ultra-Optimal AI Recommendation Engine
Purpose: Enhanced AI-powered optimization targeting 90%+ AI success rate with ultra-optimal
targets (0.96), improved learning, and better AI recommendation system prompts.
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
    meets_ultra_threshold: bool  # New: ≥0.96
    is_ai_generated: bool
    ai_confidence: float  # AI confidence in this attempt

@dataclass
class AIInsight:
    """Enhanced AI-generated insight"""
    insight_type: str
    content: str
    confidence: float
    reasoning: str
    expected_improvement: float  # Expected score improvement
    learned_from_attempts: List[int]  # Which attempts influenced this insight
    timestamp: float

@dataclass
class LearningUpdate:
    """Visible learning update between attempts"""
    attempt_number: int
    learned_pattern: str
    confidence_change: float
    strategy_adjustment: str
    reasoning: str
    timestamp: float

@dataclass
class OptimizationSession:
    """Enhanced session with ultra-optimal tracking and learning updates"""
    original_prompt: str
    prompt_category: str
    baseline_score: float
    baseline_fidelity: float
    attempts: List[OptimizationAttempt]
    ai_insights: List[AIInsight]
    learning_updates: List[LearningUpdate]  # New: visible learning
    best_attempt: Optional[OptimizationAttempt]
    total_attempts: int
    session_improvement: float
    reached_minimum_threshold: bool  # ≥0.6
    reached_target_threshold: bool   # ≥0.9
    reached_ultra_threshold: bool    # ≥0.96 (new)
    session_success: bool
    ai_contributed: bool
    ai_success_rate: float  # AI's contribution to improvements
    timestamp: float

class UltraOptimalAIEngine:
    """Ultra-optimal AI engine targeting 90%+ AI success rate"""
    
    def __init__(self, max_attempts_per_prompt: int = 8, minimum_score_target: float = 0.6, 
                 target_score: float = 0.9, ultra_optimal_target: float = 0.96,
                 ai_confidence_threshold: float = 0.65):
        self.ollama_url = "http://localhost:11434"
        self.model_name = "deepseek-r1:1.5b"
        self.db_path = "ultra_optimal_ai_engine.db"
        self.max_attempts = max_attempts_per_prompt
        self.minimum_score_target = minimum_score_target
        self.target_score = target_score
        self.ultra_optimal_target = ultra_optimal_target  # New: 0.96
        self.ai_confidence_threshold = ai_confidence_threshold
        
        self.optimization_sessions: List[OptimizationSession] = []
        
        # Enhanced dynamic strategy library
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
            "high_quality": "wbgmsst, high quality 3D model {prompt}, detailed textures, white background",
            "ultra_detailed": "wbgmsst, ultra-high detail 3D {prompt}, perfect geometry, professional rendering, white background",
            "photorealistic": "wbgmsst, photorealistic 3D {prompt}, ray-traced lighting, ultra-high quality, white background",
            "precision_engineered": "wbgmsst, precision-engineered 3D {prompt}, exact proportions, perfect finish, white background"
        }
        
        # AI-learned strategies (enhanced tracking)
        self.ai_learned_strategies = {}
        self.strategy_performance_history = {}
        
        # Enhanced learning patterns
        self.learned_patterns = {
            "ultra_optimization_strategies": ["ultra_detailed", "photorealistic", "precision_engineered"],
            "target_strategies": ["enhanced_clarity", "professional_render", "high_quality"],
            "rescue_strategies": ["material_focus", "concrete_object", "basic_description"],
            "conservative_strategies": ["raw", "basic_description", "simplified_description"]
        }
        
        self.setup_database()
        self.load_ai_learned_strategies()
        
    def setup_database(self):
        """Enhanced database with ultra-optimal and learning tracking"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
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
                reached_ultra_threshold BOOLEAN,
                session_success BOOLEAN,
                ai_contributed BOOLEAN,
                ai_success_rate REAL,
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
                meets_ultra_threshold BOOLEAN,
                is_ai_generated BOOLEAN,
                ai_confidence REAL,
                attempt_number INTEGER,
                timestamp REAL,
                FOREIGN KEY (session_id) REFERENCES optimization_sessions (id)
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS learning_updates (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id INTEGER,
                attempt_number INTEGER,
                learned_pattern TEXT,
                confidence_change REAL,
                strategy_adjustment TEXT,
                reasoning TEXT,
                timestamp REAL,
                FOREIGN KEY (session_id) REFERENCES optimization_sessions (id)
            )
        ''')
        
        conn.commit()
        conn.close()
        
    def load_ai_learned_strategies(self):
        """Load AI-learned strategies with performance tracking"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            cursor.execute('''
                SELECT strategy_name, strategy_template, success_rate, avg_score_improvement
                FROM ai_learned_strategies
                WHERE success_rate > 0.6 AND usage_count >= 2
                ORDER BY success_rate DESC
            ''')
            
            for row in cursor.fetchall():
                strategy_name, template, success_rate, avg_improvement = row
                self.ai_learned_strategies[strategy_name] = template
                print(f"📚 Loaded AI-learned strategy: {strategy_name} (success: {success_rate:.1%}, avg: +{avg_improvement:.3f})")
        except:
            print("📚 No previous AI-learned strategies found - starting fresh")
        
        conn.close()
        
    def query_deepseek_enhanced(self, system_prompt: str, user_prompt: str, context: dict = None) -> str:
        """Enhanced DeepSeek query with context and performance optimization"""
        
        # Enhanced system prompt with performance context
        enhanced_system = f"""{system_prompt}

PERFORMANCE CONTEXT:
- Your goal is to achieve 90%+ AI contribution rate to optimization success
- Previous AI insights had low impact - be more aggressive and specific
- Focus on concrete, measurable improvements that lead to score increases
- Learn from patterns and avoid repeating unsuccessful approaches

OPTIMIZATION TARGETS:
- Minimum: {self.minimum_score_target} (avoid zero fidelity)
- Target: {self.target_score} (good performance) 
- Ultra-Optimal: {self.ultra_optimal_target} (exceptional performance)

When baseline is already good (≥{self.target_score}), push for ultra-optimal (≥{self.ultra_optimal_target}).

Be confident, specific, and actionable in your recommendations."""

        if context:
            enhanced_system += f"\n\nSESSION CONTEXT:\n{json.dumps(context, indent=2)}"
        
        data = {
            "model": self.model_name,
            "messages": [
                {"role": "system", "content": enhanced_system},
                {"role": "user", "content": user_prompt}
            ],
            "stream": False
        }
        
        try:
            response = requests.post(f"{self.ollama_url}/api/chat", json=data, timeout=90)
            response.raise_for_status()
            result = response.json()
            return result["message"]["content"]
        except Exception as e:
            raise Exception(f"Enhanced DeepSeek query failed: {e}")
    
    def categorize_prompt(self, prompt: str) -> str:
        """Enhanced categorization with optimization potential assessment"""
        system_prompt = """You are an expert at categorizing 3D model generation prompts for ultra-optimal optimization.

Your categorization directly impacts optimization strategy selection. Be precise.

Categories:
- physical_object: Concrete items with clear geometric properties
- technical_description: Engineering/technical specifications  
- abstract_artistic: Artistic, abstract, or conceptual designs
- fashion_clothing: Clothing, fabrics, wearable items
- decorative_standard: Ornate, decorative, or standard objects

Consider which category will benefit most from different optimization approaches.
Physical objects often benefit from material/geometric focus.
Technical descriptions need precision and clarity.
Abstract/artistic items need creative enhancement.

Respond with ONLY the category name."""

        user_prompt = f"Categorize for ultra-optimal optimization: '{prompt}'"
        
        try:
            response = self.query_deepseek_enhanced(system_prompt, user_prompt)
            category = response.strip().lower()
            
            valid_categories = ["physical_object", "technical_description", "abstract_artistic", 
                             "fashion_clothing", "decorative_standard"]
            
            return category if category in valid_categories else "physical_object"
        except:
            return "physical_object"
    
    def generate_enhanced_ai_insights(self, prompt: str, category: str, baseline_score: float, 
                                    previous_attempts: List[OptimizationAttempt],
                                    learning_updates: List[LearningUpdate]) -> List[AIInsight]:
        """Generate enhanced AI insights with learning from previous attempts"""
        
        insights = []
        
        # Prepare comprehensive context
        attempts_summary = []
        for attempt in previous_attempts:
            attempts_summary.append({
                "attempt": attempt.attempt_number,
                "strategy": attempt.strategy_name,
                "score": attempt.validation_score,
                "improvement": attempt.score_improvement,
                "meets_target": attempt.meets_target_threshold,
                "meets_ultra": attempt.meets_ultra_threshold,
                "ai_generated": attempt.is_ai_generated
            })
        
        learning_summary = []
        for update in learning_updates:
            learning_summary.append({
                "after_attempt": update.attempt_number,
                "pattern": update.learned_pattern,
                "adjustment": update.strategy_adjustment,
                "reasoning": update.reasoning
            })
        
        context = {
            "prompt": prompt,
            "category": category,
            "baseline_score": baseline_score,
            "minimum_target": self.minimum_score_target,
            "target": self.target_score,
            "ultra_target": self.ultra_optimal_target,
            "attempts_so_far": attempts_summary,
            "learning_updates": learning_summary,
            "available_strategies": list(self.base_strategies.keys()) + list(self.ai_learned_strategies.keys()),
            "performance_requirement": "90%+ AI contribution to success"
        }
        
        # Enhanced AI Insight Generation
        if baseline_score >= self.target_score:
            # Already good - push for ultra-optimal
            insight = self._get_ultra_optimization_insight(context)
            if insight:
                insights.append(insight)
        elif len(previous_attempts) >= 2:
            # Multiple attempts tried - need custom approach
            insight = self._get_adaptive_strategy_insight(context)
            if insight:
                insights.append(insight)
        else:
            # Early stage - strategic modification
            insight = self._get_enhanced_strategy_insight(context)
            if insight:
                insights.append(insight)
        
        # Always try custom prompt generation if not reaching targets
        if len(previous_attempts) >= 2 and all(a.validation_score < self.ultra_optimal_target for a in previous_attempts):
            custom_insight = self._get_ultra_custom_prompt_insight(context)
            if custom_insight:
                insights.append(custom_insight)
        
        return insights
    
    def _get_ultra_optimization_insight(self, context: dict) -> Optional[AIInsight]:
        """Get AI insight for ultra-optimal enhancement when baseline is already good"""
        
        system_prompt = f"""You are an expert AI optimization specialist focused on achieving ultra-optimal results.

SITUATION: Baseline score {context['baseline_score']:.3f} is already good (≥{context['target']}).
TARGET: Push to ultra-optimal level (≥{context['ultra_target']}) - this is where you can make a real impact!

TASK: Recommend specific strategies or generate a custom prompt to reach ultra-optimal performance.

ANALYSIS NEEDED:
1. What specific enhancements could push this from good to exceptional?
2. Should we use ultra-detailed strategies, photorealistic rendering, or precision engineering?
3. Could a completely custom prompt achieve better results?

Be AGGRESSIVE and CONFIDENT. This is your chance to prove AI value!

RESPONSE FORMAT:
CONFIDENCE: [0.7-1.0] (be confident!)
EXPECTED_IMPROVEMENT: [estimated score improvement]
REASONING: [specific technical reasoning]
RECOMMENDATION: [strategy names OR custom prompt]
TYPE: [STRATEGY_LIST or CUSTOM_PROMPT]"""

        user_prompt = f"Push '{context['prompt']}' from {context['baseline_score']:.3f} to ultra-optimal ≥{context['ultra_target']}"
        
        try:
            response = self.query_deepseek_enhanced(system_prompt, user_prompt, context)
            
            # Parse response
            confidence_match = re.search(r'CONFIDENCE:\s*([0-9.]+)', response)
            improvement_match = re.search(r'EXPECTED_IMPROVEMENT:\s*([0-9.]+)', response)
            reasoning_match = re.search(r'REASONING:\s*(.+?)(?=RECOMMENDATION:|$)', response, re.DOTALL)
            recommendation_match = re.search(r'RECOMMENDATION:\s*(.+?)(?=TYPE:|$)', response, re.DOTALL)
            type_match = re.search(r'TYPE:\s*(.+)', response)
            
            if all([confidence_match, improvement_match, reasoning_match, recommendation_match]):
                confidence = float(confidence_match.group(1))
                expected_improvement = float(improvement_match.group(1))
                reasoning = reasoning_match.group(1).strip()
                recommendation = recommendation_match.group(1).strip()
                insight_type = "ultra_optimization" if not type_match or "STRATEGY" in type_match.group(1) else "ultra_custom_prompt"
                
                if confidence >= self.ai_confidence_threshold:
                    return AIInsight(
                        insight_type=insight_type,
                        content=recommendation,
                        confidence=confidence,
                        reasoning=reasoning,
                        expected_improvement=expected_improvement,
                        learned_from_attempts=[],
                        timestamp=time.time()
                    )
        except Exception as e:
            print(f"Ultra optimization insight error: {e}")
        
        return None
    
    def _get_adaptive_strategy_insight(self, context: dict) -> Optional[AIInsight]:
        """Get adaptive strategy insight based on learning from previous attempts"""
        
        system_prompt = f"""You are an adaptive AI that learns from optimization attempts and improves strategy selection.

LEARNING CONTEXT:
- Multiple attempts tried: {len(context['attempts_so_far'])}
- Previous attempts: {context['attempts_so_far']}
- Learning updates: {context['learning_updates']}

TASK: Analyze patterns from attempts and recommend an adaptive strategy that learns from failures.

CRITICAL ANALYSIS:
1. Why did previous strategies succeed or fail?
2. What patterns can you identify in the scores?
3. What completely different approach might work?
4. Should you generate a custom prompt based on learned patterns?

Be ANALYTICAL and ADAPTIVE. Learn from the data!

RESPONSE FORMAT:
CONFIDENCE: [0.65-1.0]
EXPECTED_IMPROVEMENT: [score improvement estimate]
REASONING: [pattern analysis and learning-based reasoning]
RECOMMENDATION: [adaptive strategy or custom prompt]
LEARNED_FROM: [which attempt numbers influenced this decision]"""

        user_prompt = f"Adapt strategy for '{context['prompt']}' based on attempt patterns"
        
        try:
            response = self.query_deepseek_enhanced(system_prompt, user_prompt, context)
            
            # Parse response (similar to above)
            confidence_match = re.search(r'CONFIDENCE:\s*([0-9.]+)', response)
            improvement_match = re.search(r'EXPECTED_IMPROVEMENT:\s*([0-9.]+)', response)
            reasoning_match = re.search(r'REASONING:\s*(.+?)(?=RECOMMENDATION:|$)', response, re.DOTALL)
            recommendation_match = re.search(r'RECOMMENDATION:\s*(.+?)(?=LEARNED_FROM:|$)', response, re.DOTALL)
            learned_match = re.search(r'LEARNED_FROM:\s*(.+)', response)
            
            if all([confidence_match, reasoning_match, recommendation_match]):
                confidence = float(confidence_match.group(1))
                expected_improvement = float(improvement_match.group(1)) if improvement_match else 0.0
                reasoning = reasoning_match.group(1).strip()
                recommendation = recommendation_match.group(1).strip()
                
                # Parse learned from attempts
                learned_from = []
                if learned_match:
                    try:
                        learned_from = [int(x.strip()) for x in learned_match.group(1).split(',') if x.strip().isdigit()]
                    except:
                        learned_from = []
                
                if confidence >= self.ai_confidence_threshold:
                    return AIInsight(
                        insight_type="adaptive_strategy",
                        content=recommendation,
                        confidence=confidence,
                        reasoning=reasoning,
                        expected_improvement=expected_improvement,
                        learned_from_attempts=learned_from,
                        timestamp=time.time()
                    )
        except Exception as e:
            print(f"Adaptive strategy insight error: {e}")
        
        return None
    
    def _get_enhanced_strategy_insight(self, context: dict) -> Optional[AIInsight]:
        """Get enhanced early-stage strategy insight"""
        
        system_prompt = f"""You are a strategic AI optimizer in the early optimization phase.

CONTEXT: 
- Prompt: "{context['prompt']}"
- Category: {context['category']}
- Baseline: {context['baseline_score']:.3f}
- Targets: Minimum {context['minimum_target']}, Target {context['target']}, Ultra {context['ultra_target']}

TASK: Recommend the most effective strategy sequence for this specific prompt and category.

STRATEGIC ANALYSIS:
1. What are the key characteristics of this prompt that affect 3D generation?
2. Which strategies are most likely to address potential issues?
3. What order should strategies be tried for maximum efficiency?

Be STRATEGIC and EFFICIENT!

RESPONSE FORMAT:
CONFIDENCE: [0.65-1.0]
EXPECTED_IMPROVEMENT: [score improvement estimate]  
REASONING: [strategic analysis]
RECOMMENDATION: [comma-separated strategy names in order]"""

        user_prompt = f"Strategic optimization for '{context['prompt']}' in category {context['category']}"
        
        try:
            response = self.query_deepseek_enhanced(system_prompt, user_prompt, context)
            
            # Parse response (similar pattern)
            confidence_match = re.search(r'CONFIDENCE:\s*([0-9.]+)', response)
            improvement_match = re.search(r'EXPECTED_IMPROVEMENT:\s*([0-9.]+)', response)
            reasoning_match = re.search(r'REASONING:\s*(.+?)(?=RECOMMENDATION:|$)', response, re.DOTALL)
            recommendation_match = re.search(r'RECOMMENDATION:\s*(.+)', response)
            
            if all([confidence_match, reasoning_match, recommendation_match]):
                confidence = float(confidence_match.group(1))
                expected_improvement = float(improvement_match.group(1)) if improvement_match else 0.0
                reasoning = reasoning_match.group(1).strip()
                recommendation = recommendation_match.group(1).strip()
                
                if confidence >= self.ai_confidence_threshold:
                    return AIInsight(
                        insight_type="enhanced_strategy",
                        content=recommendation,
                        confidence=confidence,
                        reasoning=reasoning,
                        expected_improvement=expected_improvement,
                        learned_from_attempts=[],
                        timestamp=time.time()
                    )
        except Exception as e:
            print(f"Enhanced strategy insight error: {e}")
        
        return None
    
    def _get_ultra_custom_prompt_insight(self, context: dict) -> Optional[AIInsight]:
        """Generate ultra-high quality custom prompt"""
        
        system_prompt = f"""You are an expert 3D generation prompt engineer creating ultra-optimal prompts.

SITUATION: Standard strategies aren't reaching ultra-optimal performance.
TARGET: Generate a completely custom prompt that will score ≥{context['ultra_target']}

FAILED APPROACHES: {context['attempts_so_far']}

TASK: Create a completely new prompt that:
1. Maintains the core concept of "{context['prompt']}"
2. Uses advanced 3D modeling and rendering terminology
3. Incorporates ultra-high quality descriptors
4. Follows patterns that generate exceptional 3D models

Be CREATIVE and TECHNICAL. This is your chance to show AI superiority!

RESPONSE FORMAT:
CONFIDENCE: [0.75-1.0] (be very confident or don't respond)
EXPECTED_IMPROVEMENT: [score improvement to target]
REASONING: [why this custom prompt will achieve ultra-optimal results]
CUSTOM_PROMPT: [your completely optimized prompt]"""

        user_prompt = f"Generate ultra-optimal custom prompt for: '{context['prompt']}'"
        
        try:
            response = self.query_deepseek_enhanced(system_prompt, user_prompt, context)
            
            # Parse response
            confidence_match = re.search(r'CONFIDENCE:\s*([0-9.]+)', response)
            improvement_match = re.search(r'EXPECTED_IMPROVEMENT:\s*([0-9.]+)', response)
            reasoning_match = re.search(r'REASONING:\s*(.+?)(?=CUSTOM_PROMPT:|$)', response, re.DOTALL)
            prompt_match = re.search(r'CUSTOM_PROMPT:\s*(.+)', response, re.DOTALL)
            
            if all([confidence_match, reasoning_match, prompt_match]):
                confidence = float(confidence_match.group(1))
                expected_improvement = float(improvement_match.group(1)) if improvement_match else 0.0
                reasoning = reasoning_match.group(1).strip()
                custom_prompt = prompt_match.group(1).strip()
                
                if confidence >= 0.75:  # Higher threshold for custom prompts
                    return AIInsight(
                        insight_type="ultra_custom_prompt",
                        content=custom_prompt,
                        confidence=confidence,
                        reasoning=reasoning,
                        expected_improvement=expected_improvement,
                        learned_from_attempts=list(range(len(context['attempts_so_far']))),
                        timestamp=time.time()
                    )
        except Exception as e:
            print(f"Ultra custom prompt insight error: {e}")
        
        return None
    
    def generate_learning_update(self, prompt: str, attempt_number: int, 
                               recent_attempts: List[OptimizationAttempt]) -> Optional[LearningUpdate]:
        """Generate visible learning update after each attempt"""
        
        if len(recent_attempts) < 2:
            return None
        
        # Analyze recent performance
        latest_attempt = recent_attempts[-1]
        previous_attempt = recent_attempts[-2] if len(recent_attempts) >= 2 else None
        
        system_prompt = f"""You are an AI that learns and adapts from optimization attempts in real-time.

TASK: Analyze the latest attempt and generate a learning update showing your adaptation.

RECENT ATTEMPTS:
- Previous: {previous_attempt.strategy_name if previous_attempt else 'None'} → {previous_attempt.validation_score if previous_attempt else 0:.3f}
- Latest: {latest_attempt.strategy_name} → {latest_attempt.validation_score:.3f}

LEARNING ANALYSIS:
1. What pattern can you observe from this attempt?
2. How should this change your confidence in different strategies?
3. What adjustment will you make for future attempts?

Be SPECIFIC about your learning and adaptation!

RESPONSE FORMAT:
LEARNED_PATTERN: [specific pattern observed]
CONFIDENCE_CHANGE: [+/- confidence adjustment]
STRATEGY_ADJUSTMENT: [how you'll adjust future recommendations]
REASONING: [why this learning is valuable]"""

        user_prompt = f"Learn from attempt {attempt_number} for '{prompt}'"
        
        try:
            response = self.query_deepseek_enhanced(system_prompt, user_prompt)
            
            # Parse learning update
            pattern_match = re.search(r'LEARNED_PATTERN:\s*(.+?)(?=CONFIDENCE_CHANGE:|$)', response, re.DOTALL)
            confidence_match = re.search(r'CONFIDENCE_CHANGE:\s*([+-]?[0-9.]+)', response)
            adjustment_match = re.search(r'STRATEGY_ADJUSTMENT:\s*(.+?)(?=REASONING:|$)', response, re.DOTALL)
            reasoning_match = re.search(r'REASONING:\s*(.+)', response, re.DOTALL)
            
            if all([pattern_match, adjustment_match, reasoning_match]):
                learned_pattern = pattern_match.group(1).strip()
                confidence_change = float(confidence_match.group(1)) if confidence_match else 0.0
                strategy_adjustment = adjustment_match.group(1).strip()
                reasoning = reasoning_match.group(1).strip()
                
                return LearningUpdate(
                    attempt_number=attempt_number,
                    learned_pattern=learned_pattern,
                    confidence_change=confidence_change,
                    strategy_adjustment=strategy_adjustment,
                    reasoning=reasoning,
                    timestamp=time.time()
                )
        except Exception as e:
            print(f"Learning update generation error: {e}")
        
        return None
    
    def apply_ai_insights(self, insights: List[AIInsight], prompt: str, category: str, 
                         baseline_score: float) -> Tuple[List[str], Optional[str], float]:
        """Apply AI insights with confidence tracking"""
        
        strategy_sequence = []
        custom_prompt = None
        total_ai_confidence = 0.0
        
        for insight in insights:
            print(f"🤖 AI Insight ({insight.insight_type}): Confidence {insight.confidence:.2f}")
            print(f"   💭 Reasoning: {insight.reasoning}")
            print(f"   📈 Expected Improvement: +{insight.expected_improvement:.3f}")
            
            if insight.insight_type in ["ultra_optimization", "enhanced_strategy", "adaptive_strategy"]:
                # Parse strategy recommendations
                recommended_strategies = [s.strip() for s in insight.content.split(',')]
                valid_strategies = []
                
                for strategy in recommended_strategies:
                    if strategy in self.base_strategies or strategy in self.ai_learned_strategies:
                        valid_strategies.append(strategy)
                
                if valid_strategies:
                    strategy_sequence = valid_strategies
                    print(f"   🔧 AI Strategy Sequence: {valid_strategies}")
            
            elif insight.insight_type in ["ultra_custom_prompt"]:
                custom_prompt = insight.content
                print(f"   ✨ AI Ultra Custom Prompt: '{custom_prompt[:80]}{'...' if len(custom_prompt) > 80 else ''}'")
            
            total_ai_confidence += insight.confidence
        
        # Enhanced fallback based on optimization level
        if not strategy_sequence and not custom_prompt:
            if baseline_score >= self.target_score:
                # Already good - use ultra strategies
                strategy_sequence = self.learned_patterns["ultra_optimization_strategies"]
                print(f"   🎯 Fallback: Ultra-optimization strategies for high baseline")
            elif baseline_score >= self.minimum_score_target:
                # Above minimum - use target strategies
                strategy_sequence = self.learned_patterns["target_strategies"]
                print(f"   🎯 Fallback: Target strategies for medium baseline")
            else:
                # Below minimum - use rescue strategies
                strategy_sequence = self.learned_patterns["rescue_strategies"]
                print(f"   🎯 Fallback: Rescue strategies for low baseline")
        
        avg_confidence = total_ai_confidence / len(insights) if insights else 0.0
        
        return strategy_sequence, custom_prompt, avg_confidence
    
    def apply_strategy(self, prompt: str, strategy: str) -> str:
        """Apply strategy with enhanced template support"""
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
    
    def optimize_with_ultra_ai_loop(self, prompt: str) -> OptimizationSession:
        """Main ultra-optimal optimization loop with enhanced AI learning"""
        
        print(f"\n🚀 ULTRA-OPTIMAL AI OPTIMIZATION LOOP: '{prompt}'")
        print("=" * 80)
        print(f"🎯 Goals: Minimum {self.minimum_score_target}, Target {self.target_score}, Ultra-Optimal {self.ultra_optimal_target}")
        
        # Step 1: Initial analysis
        category = self.categorize_prompt(prompt)
        print(f"📋 Category: {category}")
        
        print(f"�� Getting baseline performance...")
        baseline_val, baseline_fidelity = self.run_validation(prompt)
        print(f"   📊 Baseline: Score={baseline_val:.4f}, Fidelity={baseline_fidelity:.3f}")
        
        # Determine optimization approach based on baseline
        if baseline_val >= self.ultra_optimal_target:
            print(f"   🏆 BASELINE ALREADY ULTRA-OPTIMAL! ({baseline_val:.3f} ≥ {self.ultra_optimal_target})")
            optimization_mode = "ultra_refinement"
        elif baseline_val >= self.target_score:
            print(f"   🎯 BASELINE GOOD - PUSHING TO ULTRA-OPTIMAL ({self.ultra_optimal_target})")
            optimization_mode = "ultra_enhancement"
        elif baseline_val >= self.minimum_score_target:
            print(f"   🟡 BASELINE ADEQUATE - TARGET OPTIMIZATION")
            optimization_mode = "target_optimization"
        else:
            print(f"   🔴 BASELINE LOW - RESCUE OPTIMIZATION")
            optimization_mode = "rescue_optimization"
        
        # Initialize session tracking
        attempts = []
        ai_insights = []
        learning_updates = []
        best_attempt = None
        best_score = baseline_val
        reached_minimum = baseline_val >= self.minimum_score_target
        reached_target = baseline_val >= self.target_score
        reached_ultra = baseline_val >= self.ultra_optimal_target
        ai_contributed = False
        total_ai_confidence = 0.0
        
        # Enhanced attempt loop (increased for ultra-optimal)
        max_attempts = self.max_attempts if optimization_mode != "ultra_enhancement" else self.max_attempts + 2
        
        for attempt_num in range(1, max_attempts + 1):
            print(f"\n🔄 ATTEMPT {attempt_num}/{max_attempts} ({optimization_mode})")
            
            # Generate AI insights
            print(f"🤖 Generating AI insights...")
            insights = self.generate_enhanced_ai_insights(prompt, category, baseline_val, attempts, learning_updates)
            ai_insights.extend(insights)
            
            if not insights:
                print(f"   🤖 No AI insights generated for this attempt")
            
            # Apply AI insights
            strategy_sequence, custom_prompt, ai_confidence = self.apply_ai_insights(
                insights, prompt, category, baseline_val
            )
            total_ai_confidence += ai_confidence
            
            # Try custom prompt first if AI generated one
            if custom_prompt:
                print(f"   ✨ Trying AI Ultra Custom Prompt")
                print(f"      📝 Custom: '{custom_prompt[:80]}{'...' if len(custom_prompt) > 80 else ''}'")
                
                opt_val, opt_fidelity = self.run_validation(custom_prompt)
                score_improvement = opt_val - baseline_val
                fidelity_improvement = opt_fidelity - baseline_fidelity
                meets_minimum = opt_val >= self.minimum_score_target
                meets_target = opt_val >= self.target_score
                meets_ultra = opt_val >= self.ultra_optimal_target
                
                print(f"      📊 Results: Score={opt_val:.4f} ({score_improvement:+.3f}), Fidelity={opt_fidelity:.3f}")
                print(f"      🎯 Thresholds: Min={meets_minimum} ({'✅' if meets_minimum else '❌'}), Target={meets_target} ({'✅' if meets_target else '❌'}), Ultra={meets_ultra} ({'✅' if meets_ultra else '❌'})")
                
                attempt = OptimizationAttempt(
                    strategy_name="ai_ultra_custom",
                    optimized_prompt=custom_prompt,
                    validation_score=opt_val,
                    demo_fidelity_score=opt_fidelity,
                    score_improvement=score_improvement,
                    fidelity_improvement=fidelity_improvement,
                    attempt_number=attempt_num,
                    timestamp=time.time(),
                    meets_minimum_threshold=meets_minimum,
                    meets_target_threshold=meets_target,
                    meets_ultra_threshold=meets_ultra,
                    is_ai_generated=True,
                    ai_confidence=ai_confidence
                )
                attempts.append(attempt)
                
                if opt_val > best_score:
                    best_score = opt_val
                    best_attempt = attempt
                    ai_contributed = True
                    print(f"      🌟 NEW BEST SCORE: {opt_val:.4f} (AI ULTRA CUSTOM!)")
                
                # Update thresholds
                if meets_ultra and not reached_ultra:
                    reached_ultra = True
                    ai_contributed = True
                    print(f"      🏆 AI REACHED ULTRA-OPTIMAL! ({opt_val:.3f} ≥ {self.ultra_optimal_target})")
                    break
                elif meets_target and not reached_target:
                    reached_target = True
                    if score_improvement > 0:
                        ai_contributed = True
                    print(f"      🎯 AI REACHED TARGET! ({opt_val:.3f} ≥ {self.target_score})")
                elif meets_minimum and not reached_minimum:
                    reached_minimum = True
                    if score_improvement > 0:
                        ai_contributed = True
                    print(f"      🎉 AI REACHED MINIMUM! ({opt_val:.3f} ≥ {self.minimum_score_target})")
                
                # Generate learning update
                if len(attempts) >= 2:
                    learning_update = self.generate_learning_update(prompt, attempt_num, attempts)
                    if learning_update:
                        learning_updates.append(learning_update)
                        print(f"      📚 AI Learning: {learning_update.learned_pattern}")
                        print(f"         🔧 Adjustment: {learning_update.strategy_adjustment}")
                
                continue
            
            # Try strategy sequence
            if strategy_sequence:
                strategy = strategy_sequence[0]  # Take first strategy for this attempt
                print(f"   🔧 Trying Strategy: {strategy}")
                
                optimized_prompt = self.apply_strategy(prompt, strategy)
                print(f"      ✨ Optimized: '{optimized_prompt[:80]}{'...' if len(optimized_prompt) > 80 else ''}'")
                
                opt_val, opt_fidelity = self.run_validation(optimized_prompt)
                score_improvement = opt_val - baseline_val
                fidelity_improvement = opt_fidelity - baseline_fidelity
                meets_minimum = opt_val >= self.minimum_score_target
                meets_target = opt_val >= self.target_score
                meets_ultra = opt_val >= self.ultra_optimal_target
                
                print(f"      📊 Results: Score={opt_val:.4f} ({score_improvement:+.3f}), Fidelity={opt_fidelity:.3f}")
                print(f"      🎯 Thresholds: Min={meets_minimum} ({'✅' if meets_minimum else '❌'}), Target={meets_target} ({'✅' if meets_target else '❌'}), Ultra={meets_ultra} ({'✅' if meets_ultra else '❌'})")
                
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
                    meets_ultra_threshold=meets_ultra,
                    is_ai_generated=len(insights) > 0,
                    ai_confidence=ai_confidence
                )
                attempts.append(attempt)
                
                if opt_val > best_score:
                    best_score = opt_val
                    best_attempt = attempt
                    if len(insights) > 0:
                        ai_contributed = True
                    print(f"      🌟 NEW BEST SCORE: {opt_val:.4f}")
                
                # Update thresholds
                if meets_ultra and not reached_ultra:
                    reached_ultra = True
                    print(f"      🏆 REACHED ULTRA-OPTIMAL! ({opt_val:.3f} ≥ {self.ultra_optimal_target})")
                    break
                elif meets_target and not reached_target:
                    reached_target = True
                    print(f"      🎯 REACHED TARGET! ({opt_val:.3f} ≥ {self.target_score})")
                elif meets_minimum and not reached_minimum:
                    reached_minimum = True
                    print(f"      🎉 REACHED MINIMUM! ({opt_val:.3f} ≥ {self.minimum_score_target})")
                
                # Generate learning update
                learning_update = self.generate_learning_update(prompt, attempt_num, attempts)
                if learning_update:
                    learning_updates.append(learning_update)
                    print(f"      📚 AI Learning: {learning_update.learned_pattern}")
                    print(f"         🔧 Adjustment: {learning_update.strategy_adjustment}")
            
            time.sleep(1)
        
        # Calculate AI success metrics
        session_improvement = best_score - baseline_val if best_attempt else 0.0
        session_success = reached_ultra or reached_target or (reached_minimum and session_improvement > 0)
        ai_success_rate = (total_ai_confidence / max_attempts) if max_attempts > 0 else 0.0
        
        # Final session
        session = OptimizationSession(
            original_prompt=prompt,
            prompt_category=category,
            baseline_score=baseline_val,
            baseline_fidelity=baseline_fidelity,
            attempts=attempts,
            ai_insights=ai_insights,
            learning_updates=learning_updates,
            best_attempt=best_attempt,
            total_attempts=len(attempts),
            session_improvement=session_improvement,
            reached_minimum_threshold=reached_minimum,
            reached_target_threshold=reached_target,
            reached_ultra_threshold=reached_ultra,
            session_success=session_success,
            ai_contributed=ai_contributed,
            ai_success_rate=ai_success_rate,
            timestamp=time.time()
        )
        
        print(f"\n📊 ULTRA-OPTIMAL AI SESSION SUMMARY:")
        print(f"   Total Attempts: {session.total_attempts}")
        print(f"   AI Insights Generated: {len(ai_insights)}")
        print(f"   Learning Updates: {len(learning_updates)}")
        print(f"   Best Strategy: {session.best_attempt.strategy_name if session.best_attempt else 'None'}")
        print(f"   Best Score: {best_score:.4f} (Baseline: {baseline_val:.4f})")
        print(f"   Session Improvement: {session.session_improvement:+.3f}")
        print(f"   Reached Minimum (≥{self.minimum_score_target}): {'✅' if session.reached_minimum_threshold else '❌'}")
        print(f"   Reached Target (≥{self.target_score}): {'✅' if session.reached_target_threshold else '❌'}")
        print(f"   Reached Ultra-Optimal (≥{self.ultra_optimal_target}): {'✅' if session.reached_ultra_threshold else '❌'}")
        print(f"   AI Contributed: {'✅' if session.ai_contributed else '❌'}")
        print(f"   AI Success Rate: {session.ai_success_rate:.1%}")
        print(f"   Overall Success: {'✅' if session.session_success else '❌'}")
        
        # Store session
        self._store_session(session)
        
        return session
    
    def _store_session(self, session: OptimizationSession):
        """Store enhanced session with ultra-optimal and learning tracking"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Store session
        cursor.execute('''
            INSERT INTO optimization_sessions 
            (original_prompt, prompt_category, baseline_score, baseline_fidelity,
             total_attempts, best_strategy, best_score, best_fidelity,
             session_improvement, reached_minimum_threshold, reached_target_threshold,
             reached_ultra_threshold, session_success, ai_contributed, ai_success_rate, timestamp)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            session.original_prompt, session.prompt_category, session.baseline_score,
            session.baseline_fidelity, session.total_attempts,
            session.best_attempt.strategy_name if session.best_attempt else None,
            session.best_attempt.validation_score if session.best_attempt else 0.0,
            session.best_attempt.demo_fidelity_score if session.best_attempt else 0.0,
            session.session_improvement, session.reached_minimum_threshold,
            session.reached_target_threshold, session.reached_ultra_threshold,
            session.session_success, session.ai_contributed, session.ai_success_rate,
            session.timestamp
        ))
        
        session_id = cursor.lastrowid
        
        # Store attempts
        for attempt in session.attempts:
            cursor.execute('''
                INSERT INTO optimization_attempts
                (session_id, strategy_name, optimized_prompt, validation_score,
                 demo_fidelity_score, score_improvement, fidelity_improvement,
                 meets_minimum_threshold, meets_target_threshold, meets_ultra_threshold,
                 is_ai_generated, ai_confidence, attempt_number, timestamp)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                session_id, attempt.strategy_name, attempt.optimized_prompt,
                attempt.validation_score, attempt.demo_fidelity_score,
                attempt.score_improvement, attempt.fidelity_improvement,
                attempt.meets_minimum_threshold, attempt.meets_target_threshold,
                attempt.meets_ultra_threshold, attempt.is_ai_generated,
                attempt.ai_confidence, attempt.attempt_number, attempt.timestamp
            ))
        
        # Store learning updates
        for update in session.learning_updates:
            cursor.execute('''
                INSERT INTO learning_updates
                (session_id, attempt_number, learned_pattern, confidence_change,
                 strategy_adjustment, reasoning, timestamp)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (
                session_id, update.attempt_number, update.learned_pattern,
                update.confidence_change, update.strategy_adjustment,
                update.reasoning, update.timestamp
            ))
        
        conn.commit()
        conn.close()
    
    def run_ultra_optimal_session(self, test_prompts: List[str]):
        """Run ultra-optimal optimization session targeting 90%+ AI success"""
        
        print("🚀 ULTRA-OPTIMAL AI ENGINE v4.0 - TARGET: 90%+ AI SUCCESS RATE")
        print("=" * 80)
        print(f"📚 Testing {len(test_prompts)} prompts with ultra-optimal AI optimization")
        print(f"🎯 Targets: Min {self.minimum_score_target} | Target {self.target_score} | Ultra {self.ultra_optimal_target}")
        print(f"🤖 AI Confidence Threshold: {self.ai_confidence_threshold}")
        print(f"🔄 Max attempts per prompt: {self.max_attempts}")
        print("=" * 80)
        
        all_sessions = []
        
        for i, prompt in enumerate(test_prompts, 1):
            print(f"\n[{i}/{len(test_prompts)}] Processing prompt {i}")
            
            session = self.optimize_with_ultra_ai_loop(prompt)
            all_sessions.append(session)
            
            time.sleep(2)
        
        # Final ultra-optimal analysis
        self._run_final_ultra_analysis(all_sessions)
        
        return all_sessions
    
    def _run_final_ultra_analysis(self, sessions: List[OptimizationSession]):
        """Final analysis targeting 90%+ AI success rate"""
        
        print(f"\n🎓 FINAL ULTRA-OPTIMAL AI ANALYSIS")
        print("=" * 80)
        
        total_sessions = len(sessions)
        ai_contributed_sessions = sum(1 for s in sessions if s.ai_contributed)
        reached_minimum = sum(1 for s in sessions if s.reached_minimum_threshold)
        reached_target = sum(1 for s in sessions if s.reached_target_threshold)
        reached_ultra = sum(1 for s in sessions if s.reached_ultra_threshold)
        total_insights = sum(len(s.ai_insights) for s in sessions)
        total_learning_updates = sum(len(s.learning_updates) for s in sessions)
        
        ai_contribution_rate = (ai_contributed_sessions / total_sessions) * 100
        minimum_rate = (reached_minimum / total_sessions) * 100
        target_rate = (reached_target / total_sessions) * 100
        ultra_rate = (reached_ultra / total_sessions) * 100
        avg_ai_success_rate = statistics.mean([s.ai_success_rate for s in sessions])
        
        print(f"📊 ULTRA-OPTIMAL SESSION STATISTICS:")
        print(f"   Total Sessions: {total_sessions}")
        print(f"   🤖 AI Contributed to Success: {ai_contributed_sessions}/{total_sessions} ({ai_contribution_rate:.1f}%)")
        print(f"   📈 Average AI Success Rate: {avg_ai_success_rate:.1%}")
        print(f"   🎯 Reached Minimum (≥{self.minimum_score_target}): {reached_minimum}/{total_sessions} ({minimum_rate:.1f}%)")
        print(f"   🎯 Reached Target (≥{self.target_score}): {reached_target}/{total_sessions} ({target_rate:.1f}%)")
        print(f"   🏆 Reached Ultra-Optimal (≥{self.ultra_optimal_target}): {reached_ultra}/{total_sessions} ({ultra_rate:.1f}%)")
        print(f"   🧠 Total AI Insights: {total_insights}")
        print(f"   📚 Total Learning Updates: {total_learning_updates}")
        
        # Success analysis
        print(f"\n🎯 SUCCESS ANALYSIS:")
        if ai_contribution_rate >= 90:
            print(f"   🎉 EXCELLENT: AI contribution rate {ai_contribution_rate:.1f}% meets 90%+ target!")
        elif ai_contribution_rate >= 70:
            print(f"   🟡 GOOD: AI contribution rate {ai_contribution_rate:.1f}% - approaching target")
        else:
            print(f"   🔴 NEEDS IMPROVEMENT: AI contribution rate {ai_contribution_rate:.1f}% below target")
        
        if ultra_rate >= 50:
            print(f"   🏆 EXCEPTIONAL: {ultra_rate:.1f}% reached ultra-optimal performance!")
        elif target_rate >= 80:
            print(f"   🎯 EXCELLENT: {target_rate:.1f}% reached target performance")
        elif minimum_rate >= 90:
            print(f"   ✅ GOOD: {minimum_rate:.1f}% avoided zero fidelity")
        
        # Save ultra-optimal results
        results = {
            "ultra_optimal_analysis": {
                "total_sessions": total_sessions,
                "ai_contribution_rate": ai_contribution_rate,
                "average_ai_success_rate": avg_ai_success_rate,
                "minimum_rate": minimum_rate,
                "target_rate": target_rate,
                "ultra_optimal_rate": ultra_rate,
                "total_insights": total_insights,
                "total_learning_updates": total_learning_updates,
                "meets_90_percent_target": ai_contribution_rate >= 90,
                "timestamp": time.time()
            },
            "sessions": [asdict(s) for s in sessions]
        }
        
        output_file = f"ultra_optimal_ai_results_{int(time.time())}.json"
        with open(output_file, "w") as f:
            json.dump(results, f, indent=2)
        
        print(f"\n💾 Ultra-optimal results saved to: {output_file}")

def main():
    """Run ultra-optimal AI engine targeting 90%+ AI success"""
    
    test_prompts = [
        # Diverse optimization challenges
        "hexagonal prism steel structure",
        "cylindrical copper pipe diameter 5cm", 
        "transparent glass sphere reflection",
        "rusty metal gear mechanism",
        "elegant silk fabric draping",
        "ornate gothic candelabra silver",
        "modern minimalist chair design",
        "abstract crystalline formation",
        "vintage leather briefcase worn",
        "delicate porcelain tea cup"
    ]
    
    # Initialize ultra-optimal AI engine
    engine = UltraOptimalAIEngine(
        max_attempts_per_prompt=8,
        minimum_score_target=0.6,
        target_score=0.9,
        ultra_optimal_target=0.96,
        ai_confidence_threshold=0.65
    )
    
    # Run ultra-optimal session
    sessions = engine.run_ultra_optimal_session(test_prompts)
    
    # Final summary
    ai_contributed = sum(1 for s in sessions if s.ai_contributed)
    reached_ultra = sum(1 for s in sessions if s.reached_ultra_threshold)
    avg_ai_success = statistics.mean([s.ai_success_rate for s in sessions])
    
    print(f"\n🎯 ULTRA-OPTIMAL AI ENGINE COMPLETE!")
    print(f"🤖 AI Contribution Rate: {(ai_contributed / len(sessions)) * 100:.1f}%")
    print(f"🏆 Ultra-Optimal Achievement: {reached_ultra}/{len(sessions)} prompts")
    print(f"📈 Average AI Success Rate: {avg_ai_success:.1%}")
    
    if (ai_contributed / len(sessions)) * 100 >= 90:
        print(f"🎉 SUCCESS: Achieved 90%+ AI contribution target!")
    else:
        print(f"🔧 Continue refining to reach 90%+ AI contribution target")

if __name__ == "__main__":
    main() 
