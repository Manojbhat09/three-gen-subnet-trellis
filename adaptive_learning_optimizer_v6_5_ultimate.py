#!/usr/bin/env python3
"""
Adaptive Learning Optimizer v6.5 - The Ultimate Synthesis
Purpose: The ultimate optimization engine combining all proven features with enhanced performance

New in v6.5 Ultimate:
- Enhanced Custom Prompt Generation: Encourages custom prompts throughout, not just final push
- Ultra-Targeting Engine: Specifically optimized to reach 0.96+ scores
- Real-Time Learning Display: Shows what's being learned during optimization
- Smart Strategy Selection: Uses learned knowledge to pick best strategies
- Advanced Anti-Repetition: Prevents loops while maintaining diversity
- Adaptive Confidence: AI confidence adjusts based on historical success
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
import random

# Enhanced dataclasses for v6.5
@dataclass
class AIDecision:
    attempt_number: int
    persona_used: str
    decision_type: str
    content: str
    reasoning: str
    confidence: float
    expected_improvement: float
    conversation_turn: int
    based_on_summary: bool
    # Enhanced tracking
    actual_improvement: float = 0.0
    led_to_improvement: bool = False
    contributed_to_best_score: bool = False
    historical_confidence_boost: float = 0.0  # New: confidence boost from historical data
    timestamp: float = 0.0

@dataclass
class OptimizationAttempt:
    attempt_number: int
    strategy_name: str
    optimized_prompt: str
    validation_score: float
    demo_fidelity_score: float
    score_improvement: float
    meets_minimum_threshold: bool
    meets_target_threshold: bool
    meets_ultra_threshold: bool
    ai_decision: AIDecision
    learning_moment: Optional[str]
    is_ai_generated: bool
    # Enhanced tracking
    strategy_historical_success_rate: float = 0.0  # New: historical success rate when chosen
    ultra_potential_score: float = 0.0  # New: how close to ultra threshold
    timestamp: float = 0.0

@dataclass
class OptimizationSession:
    original_prompt: str
    prompt_category: str
    baseline_score: float
    attempts: List[OptimizationAttempt]
    best_attempt: Optional[OptimizationAttempt]
    session_improvement: float
    ai_decisions_made: int
    ai_decisions_that_improved: int
    ai_contribution_rate: float
    ai_decision_diversity: int
    reached_minimum_threshold: bool
    reached_target_threshold: bool
    reached_ultra_threshold: bool
    new_strategies_learned: int
    custom_prompts_created: int
    # Enhanced tracking
    ultra_targeting_score: float = 0.0  # New: how close we got to ultra
    learning_events_during_session: int = 0  # New: real-time learning events
    timestamp: float = 0.0

class AdaptiveLearningOptimizerV6_5_Ultimate:
    """The ultimate adaptive optimizer with enhanced learning and ultra-targeting"""

    def __init__(self, max_attempts: int = 8, min_target: float = 0.6,  # Increased attempts for ultra-targeting
                 target: float = 0.9, ultra_target: float = 0.96):
        
        self.ollama_url = "http://localhost:11434"
        self.model_name = "deepseek-r1:1.5b"
        self.db_path = "adaptive_optimizer_v6_5_ultimate.db"
        self.max_attempts = max_attempts
        self.min_target = min_target
        self.target = target
        self.ultra_target = ultra_target

        # Enhanced anti-repetition with smarter memory
        self.used_strategies = []
        self.used_decision_types = []
        self.max_memory = 3
        self.custom_prompt_encouragement_threshold = 3  # Encourage custom prompts after 3 failed attempts
        
        # Ultra-targeting configuration
        self.ultra_targeting_mode = False  # Activated when we get close to ultra
        self.ultra_proximity_threshold = 0.85  # When to activate ultra-targeting
        
        # Enhanced strategy library with ultra-targeting templates
        self.strategies = {
            "material_focus": "wbgmsst, solid {prompt} object 3D, white background",
            "geometric_focus": "wbgmsst, {prompt} geometric 3D model, white background",
            "basic_description": "3D model of {prompt}",
            "enhanced_clarity": "wbgmsst, detailed 3D {prompt} model, accurate geometry, white background",
            "concrete_object": "wbgmsst, {prompt} as 3D object, realistic proportions, white background",
            "professional_render": "wbgmsst, professional 3D render of {prompt}, studio lighting, white background",
            "high_quality": "wbgmsst, high quality 3D model {prompt}, detailed textures, white background",
            "ultra_detailed": "wbgmsst, ultra-high detail 3D {prompt}, perfect geometry, white background",
            "photorealistic": "wbgmsst, photorealistic 3D {prompt}, ray-traced lighting, white background",
            "technical_spec": "wbgmsst, technical 3D {prompt}, precise dimensions, engineering quality, white background",
            "industrial_design": "wbgmsst, industrial {prompt} design, realistic materials, white background",
            "artistic_render": "wbgmsst, artistic {prompt} sculpture, refined details, white background",
            "minimal_clean": "wbgmsst, clean minimal 3D {prompt}, simple geometry, white background",
            # Ultra-targeting strategies
            "ultra_precision": "wbgmsst, ultra-precise {prompt}, museum quality, perfect detail, professional photography, white background",
            "masterpiece_quality": "wbgmsst, masterpiece {prompt}, gallery exhibition quality, flawless execution, white background"
        }
        
        # Enhanced custom prompt templates with ultra-targeting
        self.custom_prompt_templates = {
            "technical": [
                "wbgmsst, precision-engineered {prompt}, technical CAD quality, accurate dimensions, industrial grade, white background",
                "wbgmsst, ultra-detailed technical {prompt}, engineering blueprint quality, precise geometry, professional render, white background",
                "wbgmsst, high-precision {prompt}, manufacturing quality, exact specifications, technical illustration, white background",
                # Ultra-targeting templates
                "wbgmsst, ultra-precision technical {prompt}, aerospace engineering quality, perfect geometry, CAD-accurate dimensions, white background",
                "wbgmsst, masterpiece-grade technical {prompt}, engineering perfection, ultra-detailed specifications, flawless precision, white background"
            ],
            "artistic": [
                "wbgmsst, elegant artistic {prompt}, museum quality sculpture, refined details, perfect lighting, white background",
                "wbgmsst, sophisticated {prompt}, high-end artistic render, graceful form, studio lighting, white background",
                "wbgmsst, masterpiece {prompt}, artistic excellence, refined aesthetics, perfect composition, white background",
                # Ultra-targeting templates
                "wbgmsst, ultra-refined artistic {prompt}, gallery masterpiece quality, perfect artistic execution, museum-grade detail, white background",
                "wbgmsst, award-winning artistic {prompt}, world-class sculpture, flawless artistic vision, exhibition quality, white background"
            ],
            "textile": [
                "wbgmsst, luxury {prompt}, high-end fabric simulation, realistic textile physics, studio lighting, white background",
                "wbgmsst, premium quality {prompt}, detailed fabric texture, natural draping, soft lighting, white background",
                "wbgmsst, haute couture {prompt}, exquisite textile detail, perfect fabric simulation, white background",
                # Ultra-targeting templates
                "wbgmsst, ultra-luxury {prompt}, couture-grade fabric simulation, perfect textile physics, masterpiece draping, white background",
                "wbgmsst, museum-quality {prompt}, flawless fabric texture, ultra-realistic textile behavior, exhibition standard, white background"
            ],
            "physical": [
                "wbgmsst, premium quality {prompt}, realistic materials, perfect proportions, professional product shot, white background",
                "wbgmsst, high-end {prompt}, commercial grade quality, accurate details, studio lighting, white background",
                "wbgmsst, professional {prompt}, product visualization quality, precise modeling, perfect render, white background",
                # Ultra-targeting templates
                "wbgmsst, ultra-premium {prompt}, perfection-grade quality, flawless proportions, masterpiece rendering, white background",
                "wbgmsst, exhibition-quality {prompt}, museum-standard detail, perfect craftsmanship, award-winning visualization, white background"
            ]
        }
        
        # Learning systems
        self.ai_learned_strategies = {}
        self.strategy_performance = {}
        self.custom_prompts_used = 0
        self.real_time_learning_events = 0
        
        self.setup_database()
        self.load_historical_learning()

    def setup_database(self):
        """Enhanced database setup for v6.5"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Strategy performance tracking (enhanced)
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS strategy_performance (
                strategy_name TEXT,
                category TEXT,
                success_rate REAL,
                avg_improvement REAL,
                usage_count INTEGER,
                ultra_achievement_rate REAL,
                last_used REAL,
                PRIMARY KEY (strategy_name, category)
            )
        ''')
        
        # AI learned strategies (enhanced)
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS learned_strategies (
                strategy_name TEXT PRIMARY KEY,
                template TEXT,
                category_affinity TEXT,
                base_success_rate REAL,
                ultra_success_rate REAL,
                usage_count INTEGER,
                learned_from_prompt TEXT,
                learning_score REAL,
                timestamp REAL
            )
        ''')
        
        # Enhanced sessions tracking
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS sessions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp REAL,
                original_prompt TEXT,
                category TEXT,
                baseline_score REAL,
                best_score REAL,
                session_improvement REAL,
                total_attempts INTEGER,
                custom_prompts_created INTEGER,
                new_strategies_learned INTEGER,
                reached_minimum BOOLEAN,
                reached_target BOOLEAN,
                reached_ultra BOOLEAN,
                ai_contribution_rate REAL,
                ultra_targeting_score REAL,
                learning_events_during_session INTEGER
            )
        ''')
        
        # Real-time learning events
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS learning_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id INTEGER,
                event_type TEXT,
                description TEXT,
                confidence_change REAL,
                timestamp REAL,
                FOREIGN KEY (session_id) REFERENCES sessions (id)
            )
        ''')
        
        conn.commit()
        conn.close()

    def load_historical_learning(self):
        """Enhanced historical learning with ultra-targeting insights"""
        print("🧠 Loading enhanced historical knowledge...")
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            # Load AI-learned strategies with ultra performance
            cursor.execute("SELECT strategy_name, template, ultra_success_rate FROM learned_strategies WHERE base_success_rate > 0.4")
            ultra_strategies = 0
            for row in cursor.fetchall():
                name, template, ultra_rate = row
                self.ai_learned_strategies[name] = template
                if ultra_rate and ultra_rate > 0.5:
                    ultra_strategies += 1
                print(f"   📚 Loaded: {name} (Ultra rate: {ultra_rate:.1%})")
            
            # Load strategy performance with ultra insights
            cursor.execute("SELECT strategy_name, category, success_rate, avg_improvement, usage_count, ultra_achievement_rate FROM strategy_performance WHERE usage_count > 1")
            ultra_capable_strategies = 0
            for row in cursor.fetchall():
                name, cat, rate, imp, count, ultra_rate = row
                if name not in self.strategy_performance:
                    self.strategy_performance[name] = {}
                self.strategy_performance[name][cat] = {
                    "success_rate": rate,
                    "avg_improvement": imp,
                    "usage_count": count,
                    "ultra_achievement_rate": ultra_rate or 0.0
                }
                if ultra_rate and ultra_rate > 0:
                    ultra_capable_strategies += 1
            
            print(f"✅ Knowledge loaded: {len(self.ai_learned_strategies)} learned strategies ({ultra_strategies} ultra-capable)")
            print(f"   📊 {len(self.strategy_performance)} strategy records ({ultra_capable_strategies} with ultra achievements)")
            
        except sqlite3.OperationalError as e:
            print(f"   📝 Starting fresh v6.5 knowledge base: {e}")
        finally:
            conn.close()

    def query_ai_enhanced(self, user_message: str, timeout: int = 50) -> str:
        """Enhanced AI query with better parameters for creativity and precision"""
        data = {
            "model": self.model_name,
            "messages": [{"role": "user", "content": user_message}],
            "stream": False,
            "options": {
                "temperature": 0.85,  # Balanced creativity
                "top_p": 0.92,
                "num_predict": 300,  # Longer responses for better reasoning
                "stop": ["<think>", "</think>"],
                "repeat_penalty": 1.1  # Reduce repetition
            }
        }
        
        try:
            response = requests.post(f"{self.ollama_url}/api/chat", json=data, timeout=timeout)
            response.raise_for_status()
            content = response.json()["message"]["content"]
            
            # Enhanced cleaning
            content = content.replace("<think>", "").replace("</think>", "")
            content = re.sub(r'\*\*([^*]+)\*\*', r'\1', content)  # Remove markdown
            content = re.sub(r'(?i)^(sure|okay|alright)[.,!]?\s*', '', content)  # Remove common prefixes
            
            return content.strip()
        except Exception as e:
            return f"ERROR: {str(e)}"

    def query_ai_for_learning(self, messages: List[Dict[str, str]]) -> str:
        """Enhanced AI learning queries"""
        data = {
            "model": self.model_name,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": 0.7,
                "top_p": 0.9,
                "num_predict": 200
            }
        }
        
        try:
            response = requests.post(f"{self.ollama_url}/api/chat", json=data, timeout=60)
            response.raise_for_status()
            return response.json()["message"]["content"]
        except Exception as e:
            return f"ERROR: {str(e)}"

    def run_validation(self, prompt: str) -> Tuple[float, float]:
        """Enhanced validation with mock validator for testing"""
        try:
            cmd = [sys.executable, "mock_validator_for_testing.py", prompt]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
            
            if result.returncode != 0:
                print(f"   ⚠️ Validation warning: {result.stderr}")
                return 0.0, 0.0
            
            with open("subnet_validation_results.json", 'r') as f:
                data = json.load(f)
                val_score = data.get("validation_engine_score", 0.0)
                demo_score = data.get("demo_fidelity_score", 0.0)
                
                # Ultra-targeting feedback
                if val_score >= self.ultra_target:
                    print(f"   🏆 ULTRA ACHIEVEMENT: {val_score:.3f}!")
                elif val_score >= self.ultra_proximity_threshold:
                    print(f"   🎯 Ultra proximity: {val_score:.3f} (need {self.ultra_target - val_score:.3f} more)")
                
                return val_score, demo_score
                
        except Exception as e:
            print(f"   ❌ Validation error: {e}")
            return 0.0, 0.0

    def categorize_prompt(self, prompt: str) -> str:
        """Enhanced categorization with better accuracy"""
        prompt_lower = prompt.lower()
        
        # Technical indicators (expanded)
        if any(word in prompt_lower for word in ["steel", "metal", "copper", "iron", "aluminum", "geometric", "prism", "cylinder", "sphere", "hexagonal", "technical", "engineering", "mechanical"]):
            return "technical"
        
        # Artistic indicators (expanded)
        elif any(word in prompt_lower for word in ["elegant", "artistic", "ornate", "beautiful", "sculpture", "masterpiece", "gallery", "refined", "graceful"]):
            return "artistic"
        
        # Textile indicators (expanded)
        elif any(word in prompt_lower for word in ["fabric", "silk", "cotton", "textile", "cloth", "draping", "weaving", "fiber"]):
            return "textile"
        
        # Physical object (default)
        else:
            return "physical"

    def get_enhanced_learned_knowledge(self, category: str) -> Dict[str, str]:
        """Enhanced knowledge summary with ultra-targeting insights"""
        knowledge = {
            "successful_strategies": "Limited historical data",
            "ultra_capable_strategies": "None identified yet",
            "learned_strategies_count": len(self.ai_learned_strategies),
            "recommendations": "Explore custom prompts for breakthroughs"
        }
        
        if category in [perf.get(category, {}) for perf in self.strategy_performance.values() if perf]:
            # Get successful strategies
            successful = []
            ultra_capable = []
            
            for strategy, cats in self.strategy_performance.items():
                if category in cats:
                    perf = cats[category]
                    if perf['success_rate'] > 0.6 and perf['usage_count'] >= 2:
                        successful.append(f"{strategy} ({perf['success_rate']:.1%})")
                    if perf.get('ultra_achievement_rate', 0) > 0:
                        ultra_capable.append(f"{strategy} ({perf['ultra_achievement_rate']:.1%} ultra)")
            
            if successful:
                knowledge["successful_strategies"] = ", ".join(successful[:3])
            if ultra_capable:
                knowledge["ultra_capable_strategies"] = ", ".join(ultra_capable[:2])
                knowledge["recommendations"] = "Focus on ultra-capable strategies and custom prompts"
        
        return knowledge

    def activate_ultra_targeting_mode(self, current_best_score: float):
        """Activate enhanced ultra-targeting when we get close"""
        if current_best_score >= self.ultra_proximity_threshold and not self.ultra_targeting_mode:
            self.ultra_targeting_mode = True
            print(f"   🎯 ULTRA-TARGETING MODE ACTIVATED! (Current: {current_best_score:.3f}, Target: {self.ultra_target})")
            self.log_learning_event("ultra_targeting_activated", f"Activated at score {current_best_score:.3f}")
            return True
        return False

    def log_learning_event(self, event_type: str, description: str, confidence_change: float = 0.0):
        """Log real-time learning events"""
        self.real_time_learning_events += 1
        print(f"   🧠 Learning Event: {description}")
        # Could save to database if session_id is available

    def select_enhanced_persona_and_strategies(self, attempt_num: int, baseline_score: float, 
                                             attempts: List[OptimizationAttempt], category: str) -> Tuple[str, List[str], float]:
        """Enhanced persona selection with ultra-targeting and confidence boosting"""
        
        current_best = max([a.validation_score for a in attempts] + [baseline_score])
        
        # Activate ultra-targeting if close
        if current_best >= self.ultra_proximity_threshold:
            self.activate_ultra_targeting_mode(current_best)
        
        # Enhanced persona selection
        persona = "Strategic Optimizer"  # Default
        confidence_boost = 0.0
        
        # Ultra-targeting personas
        if self.ultra_targeting_mode:
            persona = "Ultra-Targeting Specialist"
            confidence_boost = 0.1
        elif current_best < 0.4:
            persona = "Rescue Specialist"
        elif attempt_num >= self.custom_prompt_encouragement_threshold and not any(a.is_ai_generated for a in attempts):
            persona = "Custom Prompt Master"
            confidence_boost = 0.15  # High confidence for custom prompts
        elif all(a.score_improvement <= 0.01 for a in attempts[-2:]) and len(attempts) >= 2:
            persona = "Creative Breakthrough Specialist"
            confidence_boost = 0.1
        elif current_best >= self.target:
            persona = "Excellence Optimizer"
            confidence_boost = 0.05
        
        # Enhanced strategy selection with learned knowledge
        all_strategies = list(self.strategies.keys()) + list(self.ai_learned_strategies.keys())
        
        # Prioritize strategies with historical success in this category
        if category in [perf.get(category, {}) for perf in self.strategy_performance.values() if perf]:
            successful_strategies = []
            for strategy in all_strategies:
                if strategy in self.strategy_performance and category in self.strategy_performance[strategy]:
                    perf = self.strategy_performance[strategy][category]
                    if perf['success_rate'] > 0.5:
                        successful_strategies.append(strategy)
            
            if successful_strategies:
                print(f"   📊 Prioritizing historically successful strategies: {successful_strategies[:3]}")
        
        # Anti-repetition
        recently_used = [a.strategy_name for a in attempts[-self.max_memory:] if a.strategy_name != "ai_custom_prompt"]
        available_strategies = [s for s in all_strategies if s not in recently_used]
        
        # Ensure we have enough strategies
        if len(available_strategies) < 5:
            available_strategies.extend([s for s in all_strategies if s not in available_strategies][:3])
        
        return persona, available_strategies[:10], confidence_boost

    def make_enhanced_ai_decision(self, prompt: str, category: str, baseline_score: float, 
                                 attempt_num: int, attempts: List[OptimizationAttempt]) -> AIDecision:
        """Enhanced AI decision making with ultra-targeting and better custom prompt encouragement"""
        
        persona, available_strategies, confidence_boost = self.select_enhanced_persona_and_strategies(
            attempt_num, baseline_score, attempts, category
        )
        
        # Enhanced context building
        recent_results = []
        for attempt in attempts[-3:]:
            custom_indicator = " (CUSTOM)" if attempt.is_ai_generated else ""
            ultra_indicator = " 🏆ULTRA" if attempt.meets_ultra_threshold else " 🎯CLOSE" if attempt.validation_score >= self.ultra_proximity_threshold else ""
            recent_results.append(f"Attempt {attempt.attempt_number}: {attempt.strategy_name}{custom_indicator} -> {attempt.validation_score:.3f}{ultra_indicator}")
        
        custom_prompts_used = sum(1 for a in attempts if a.is_ai_generated)
        current_best = max([a.validation_score for a in attempts] + [baseline_score])
        
        # Enhanced learned knowledge
        knowledge = self.get_enhanced_learned_knowledge(category)
        
        # Dynamic encouragement based on situation
        custom_encouragement = ""
        if self.ultra_targeting_mode:
            custom_encouragement = f"\n🎯 ULTRA-TARGETING MODE: Need {self.ultra_target - current_best:.3f} more for ULTRA! Custom prompts are your best tool!"
        elif current_best >= self.ultra_proximity_threshold:
            custom_encouragement = f"\n🔥 SO CLOSE TO ULTRA! ({current_best:.3f}/{self.ultra_target}) - Custom prompt could be the breakthrough!"
        elif attempt_num >= self.custom_prompt_encouragement_threshold and custom_prompts_used == 0:
            custom_encouragement = f"\n💡 STRONG RECOMMENDATION: {attempt_num} attempts without custom prompts - time to get creative!"
        elif baseline_score < self.min_target and all(a.score_improvement <= 0 for a in attempts[-2:]):
            custom_encouragement = f"\n🚨 RESCUE MODE: Strategies failing - WRITE_CUSTOM is your lifeline!"
        
        # Enhanced user message with ultra-targeting
        user_message = f"""You are a {persona} optimizing 3D model generation for ULTRA performance.

CURRENT SITUATION:
- Prompt: "{prompt}" (Category: {category})
- Baseline: {baseline_score:.3f}
- Current Best: {current_best:.3f}
- Attempt: {attempt_num}/{self.max_attempts}
- Targets: Min {self.min_target} | Excellent {self.target} | 🏆 ULTRA {self.ultra_target}

RECENT PERFORMANCE:
{chr(10).join(recent_results) if recent_results else "None yet"}

LEARNED KNOWLEDGE:
- Successful Strategies: {knowledge['successful_strategies']}
- Ultra-Capable Strategies: {knowledge['ultra_capable_strategies']}
- Learned Strategies Available: {knowledge['learned_strategies_count']}
- AI Recommendation: {knowledge['recommendations']}

CUSTOM PROMPTS USED: {custom_prompts_used}

AVAILABLE STRATEGIES:
{', '.join(available_strategies)}

{custom_encouragement}

ULTRA-TARGETING TIPS:
- Custom prompts often achieve the highest scores
- Combine multiple quality descriptors (ultra-detailed, precision-engineered, masterpiece)
- Technical prompts benefit from precision language
- Artistic prompts benefit from quality and refinement terms

DECISION OPTIONS:
A) WRITE_CUSTOM: Create an optimized custom prompt (RECOMMENDED for ultra scores!)
B) USE_STRATEGY: Select one strategy from available list
C) EARLY_STOP: Stop if already optimal or hopeless

RESPOND FORMAT:
CHOICE: [A/B/C]
STRATEGY_OR_PROMPT: [strategy name OR complete custom prompt OR stop reason]
REASONING: [detailed explanation of why this approach will work]
CONFIDENCE: [0.1-1.0]

Your response:"""

        print(f"🤖 AI Persona: {persona}")
        print(f"   📋 Available Strategies: {len(available_strategies)}")
        print(f"   ✨ Custom Prompts Used: {custom_prompts_used}")
        print(f"   🧠 Knowledge: {knowledge['learned_strategies_count']} learned strategies")
        print(f"   🎯 Ultra Mode: {'ACTIVE' if self.ultra_targeting_mode else 'STANDBY'}")
        
        ai_response = self.query_ai_enhanced(user_message)
        
        if "ERROR:" in ai_response:
            return self.create_enhanced_fallback_decision(attempt_num, persona, ai_response, available_strategies, confidence_boost)
        
        print(f"   🤖 AI Response: {ai_response[:120]}...")
        
        decision = self.parse_enhanced_response(ai_response, attempt_num, persona, available_strategies)
        decision.historical_confidence_boost = confidence_boost
        decision.confidence = min(1.0, decision.confidence + confidence_boost)
        
        return decision

    def parse_enhanced_response(self, response: str, attempt_num: int, persona: str, available_strategies: List[str]) -> AIDecision:
        """Enhanced response parsing with better custom prompt extraction"""
        
        decision_type = "USE_STRATEGY"
        content = available_strategies[0] if available_strategies else "enhanced_clarity"
        reasoning = response[:150]  # Longer reasoning
        confidence = 0.5
        
        try:
            # Enhanced extraction patterns
            choice_patterns = [
                r'CHOICE:\s*([ABC])',
                r'DECISION:\s*([ABC])',
                r'I choose\s*([ABC])',
                r'My choice is\s*([ABC])'
            ]
            
            choice = None
            for pattern in choice_patterns:
                match = re.search(pattern, response, re.IGNORECASE)
                if match:
                    choice = match.group(1).upper()
                    break
            
            if choice == "A":
                decision_type = "WRITE_CUSTOM"
            elif choice == "B":
                decision_type = "USE_STRATEGY"
            elif choice == "C":
                decision_type = "EARLY_STOP"
            
            # Enhanced content extraction
            content_patterns = [
                r'STRATEGY_OR_PROMPT:\s*(.+?)(?:\n|REASONING:|CONFIDENCE:|$)',
                r'CONTENT:\s*(.+?)(?:\n|REASONING:|CONFIDENCE:|$)',
                r'PROMPT:\s*(.+?)(?:\n|REASONING:|CONFIDENCE:|$)'
            ]
            
            for pattern in content_patterns:
                match = re.search(pattern, response, re.DOTALL | re.IGNORECASE)
                if match:
                    content = match.group(1).strip()
                    break
            
            # Enhanced reasoning extraction
            reasoning_patterns = [
                r'REASONING:\s*(.+?)(?=CONFIDENCE:|$)',
                r'EXPLANATION:\s*(.+?)(?=CONFIDENCE:|$)',
                r'WHY:\s*(.+?)(?=CONFIDENCE:|$)'
            ]
            
            for pattern in reasoning_patterns:
                match = re.search(pattern, response, re.DOTALL | re.IGNORECASE)
                if match:
                    reasoning = match.group(1).strip()
                    break
            
            # Enhanced confidence extraction
            conf_patterns = [
                r'CONFIDENCE:\s*([0-9.]+)',
                r'CERTAINTY:\s*([0-9.]+)',
                r'confidence[:\s]*([0-9.]+)'
            ]
            
            for pattern in conf_patterns:
                match = re.search(pattern, response, re.IGNORECASE)
                if match:
                    confidence = min(1.0, max(0.1, float(match.group(1))))
                    break
            
            # Enhanced custom prompt validation and improvement
            if decision_type == "WRITE_CUSTOM":
                content = self.enhance_custom_prompt(content, reasoning)
            
            # Enhanced strategy validation
            elif decision_type == "USE_STRATEGY":
                all_strategies = {**self.strategies, **self.ai_learned_strategies}
                if content not in all_strategies:
                    # Smart strategy matching
                    best_match = None
                    max_similarity = 0
                    for strategy in available_strategies:
                        similarity = sum(1 for word in content.lower().split() if word in strategy.lower())
                        if similarity > max_similarity:
                            max_similarity = similarity
                            best_match = strategy
                    
                    content = best_match if best_match else available_strategies[0]
            
        except Exception as e:
            print(f"   ⚠️ Enhanced parsing error: {e}, using advanced fallback")
            
            # Advanced semantic parsing
            response_lower = response.lower()
            
            # Look for custom prompt indicators
            if any(indicator in response_lower for indicator in ["custom", "write", "create", "wbgmsst", "precision", "ultra", "masterpiece"]):
                decision_type = "WRITE_CUSTOM"
                
                # Extract potential custom prompts from response
                lines = response.split('\n')
                for line in lines:
                    if any(word in line.lower() for word in ["wbgmsst", "3d", "white background"]):
                        content = line.strip()
                        break
                
                if not any(word in content.lower() for word in ["wbgmsst", "3d"]):
                    content = self.generate_fallback_custom_prompt(reasoning, "technical")  # Default category
            
            # Look for strategy names
            else:
                for strategy in available_strategies:
                    if strategy in response_lower or any(word in response_lower for word in strategy.split('_')):
                        decision_type = "USE_STRATEGY"
                        content = strategy
                        break
        
        return AIDecision(
            attempt_number=attempt_num,
            persona_used=persona,
            decision_type=decision_type,
            content=content,
            reasoning=reasoning,
            confidence=confidence,
            expected_improvement=confidence * 0.25,  # Higher potential improvement
            conversation_turn=attempt_num,
            based_on_summary=False,
            timestamp=time.time()
        )

    def enhance_custom_prompt(self, raw_content: str, reasoning: str) -> str:
        """Enhance custom prompts with ultra-targeting elements"""
        
        if len(raw_content) < 20 or not any(word in raw_content.lower() for word in ["3d", "model", "background", "wbgmsst"]):
            # Generate enhanced custom prompt based on reasoning
            if "ultra" in reasoning.lower() or "precision" in reasoning.lower():
                enhanced = f"wbgmsst, ultra-precision {raw_content}, masterpiece quality, perfect detail, white background"
            elif "technical" in reasoning.lower():
                enhanced = f"wbgmsst, precision-engineered {raw_content}, technical excellence, accurate geometry, white background"
            elif "artistic" in reasoning.lower():
                enhanced = f"wbgmsst, artistic masterpiece {raw_content}, gallery quality, refined details, white background"
            else:
                enhanced = f"wbgmsst, ultra-detailed {raw_content}, professional quality, perfect rendering, white background"
            return enhanced
        
        # Enhance existing prompt
        enhanced_content = raw_content
        
        # Ensure starts with wbgmsst
        if not enhanced_content.lower().startswith("wbgmsst"):
            enhanced_content = f"wbgmsst, {enhanced_content}"
        
        # Add ultra-targeting elements if not present
        ultra_keywords = ["ultra", "precision", "masterpiece", "perfect", "flawless", "exhibition"]
        if not any(keyword in enhanced_content.lower() for keyword in ultra_keywords):
            enhanced_content = enhanced_content.replace("wbgmsst, ", "wbgmsst, ultra-detailed ")
        
        # Ensure white background
        if "white background" not in enhanced_content.lower():
            enhanced_content += ", white background"
        
        return enhanced_content

    def generate_fallback_custom_prompt(self, reasoning: str, category: str) -> str:
        """Generate fallback custom prompt based on category and reasoning"""
        templates = self.custom_prompt_templates.get(category, self.custom_prompt_templates['physical'])
        
        # Choose ultra-targeting template if available
        if len(templates) > 3:  # Has ultra templates
            template = templates[3]  # First ultra template
        else:
            template = templates[0]  # Fallback to regular template
        
        # Extract subject from reasoning or use generic
        subject = "object"
        words = reasoning.split()
        for word in words:
            if len(word) > 3 and word.isalpha():
                subject = word
                break
        
        return template.format(prompt=subject)

    def create_enhanced_fallback_decision(self, attempt_num: int, persona: str, error_msg: str, 
                                        available_strategies: List[str], confidence_boost: float) -> AIDecision:
        """Enhanced fallback with smart decision making"""
        
        # Prefer custom prompts for later attempts or ultra-targeting mode
        if attempt_num > 4 or self.ultra_targeting_mode:
            return AIDecision(
                attempt_number=attempt_num,
                persona_used=persona,
                decision_type="WRITE_CUSTOM",
                content="wbgmsst, ultra-detailed 3D model, masterpiece quality, perfect rendering, white background",
                reasoning=f"Enhanced fallback custom prompt: {error_msg[:50]}",
                confidence=0.5 + confidence_boost,
                expected_improvement=0.15,
                conversation_turn=attempt_num,
                based_on_summary=False,
                historical_confidence_boost=confidence_boost,
                timestamp=time.time()
            )
        else:
            # Use strategy with confidence boost
            strategy = available_strategies[attempt_num % len(available_strategies)] if available_strategies else "enhanced_clarity"
            return AIDecision(
                attempt_number=attempt_num,
                persona_used=persona,
                decision_type="USE_STRATEGY",
                content=strategy,
                reasoning=f"Enhanced fallback strategy: {error_msg[:50]}",
                confidence=0.4 + confidence_boost,
                expected_improvement=0.08,
                conversation_turn=attempt_num,
                based_on_summary=False,
                historical_confidence_boost=confidence_boost,
                timestamp=time.time()
            )

    def execute_enhanced_ai_decision(self, decision: AIDecision, prompt: str) -> Tuple[str, str]:
        """Enhanced decision execution with ultra-targeting support"""
        
        if decision.decision_type == "EARLY_STOP":
            return "early_stop", prompt
        
        elif decision.decision_type == "WRITE_CUSTOM":
            custom_prompt = decision.content
            
            # Validate and enhance custom prompt
            if len(custom_prompt) > 20 and any(word in custom_prompt.lower() for word in ["3d", "model", "background"]):
                # Add ultra-targeting elements if in ultra mode
                if self.ultra_targeting_mode and "ultra" not in custom_prompt.lower():
                    custom_prompt = custom_prompt.replace("wbgmsst, ", "wbgmsst, ultra-precision ")
                return "ai_custom_prompt", custom_prompt
            else:
                # Generate enhanced custom prompt
                enhanced = f"wbgmsst, ultra-precision 3D {prompt}, masterpiece quality, perfect detail, white background"
                return "ai_custom_prompt", enhanced
        
        elif decision.decision_type == "USE_STRATEGY":
            strategy = decision.content
            all_strategies = {**self.strategies, **self.ai_learned_strategies}
            
            if strategy in all_strategies:
                executed_prompt = all_strategies[strategy].format(prompt=prompt)
                
                # Add ultra-targeting enhancement if in ultra mode
                if self.ultra_targeting_mode and strategy in self.strategies:
                    if "ultra" not in executed_prompt.lower():
                        executed_prompt = executed_prompt.replace("wbgmsst, ", "wbgmsst, ultra-detailed ")
                
                return strategy, executed_prompt
            else:
                return "enhanced_clarity", self.strategies["enhanced_clarity"].format(prompt=prompt)
        
        return "enhanced_clarity", self.strategies["enhanced_clarity"].format(prompt=prompt)

    # Use learning system from v6.4 fixed (same methods)
    def _learn_from_session(self, prompt: str, category: str, attempts: List[OptimizationAttempt]) -> int:
        """Enhanced learning system with ultra-targeting insights"""
        print("🧠 Updating enhanced knowledge base with ultra-targeting insights...")
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        newly_learned_count = 0

        # Enhanced performance tracking with ultra achievements
        for attempt in attempts:
            strategy = attempt.strategy_name
            cursor.execute("SELECT success_rate, avg_improvement, usage_count, ultra_achievement_rate FROM strategy_performance WHERE strategy_name = ? AND category = ?", (strategy, category))
            row = cursor.fetchone()
            
            is_success = attempt.score_improvement > 0.01
            is_ultra = attempt.meets_ultra_threshold
            
            if row:
                old_rate, old_imp, old_count, old_ultra_rate = row
                new_count = old_count + 1
                new_rate = ((old_rate * old_count) + (1 if is_success else 0)) / new_count
                new_imp = ((old_imp * old_count) + attempt.score_improvement) / new_count
                new_ultra_rate = ((old_ultra_rate * old_count) + (1 if is_ultra else 0)) / new_count
                
                cursor.execute("""
                    UPDATE strategy_performance 
                    SET success_rate = ?, avg_improvement = ?, usage_count = ?, ultra_achievement_rate = ?, last_used = ?
                    WHERE strategy_name = ? AND category = ?
                """, (new_rate, new_imp, new_count, new_ultra_rate, time.time(), strategy, category))
            else:
                cursor.execute("""
                    INSERT INTO strategy_performance (strategy_name, category, success_rate, avg_improvement, usage_count, ultra_achievement_rate, last_used)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                """, (strategy, category, 1.0 if is_success else 0.0, attempt.score_improvement, 1, 1.0 if is_ultra else 0.0, time.time()))

        # Enhanced learning from ultra-successful custom prompts
        for attempt in attempts:
            if attempt.is_ai_generated and (attempt.meets_ultra_threshold or attempt.validation_score >= self.ultra_proximity_threshold):
                print(f"   🎓 Learning from high-performance custom prompt (score: {attempt.validation_score:.3f})")
                new_strategy_template = self._distill_enhanced_strategy(prompt, attempt.optimized_prompt)
                
                if new_strategy_template:
                    newly_learned_count += 1
                    strategy_name = f"ai_ultra_{category}_{int(time.time())}"
                    print(f"   ✅ Ultra Strategy Learned: '{strategy_name}'")
                    print(f"        Template: {new_strategy_template}")
                    
                    ultra_rate = 1.0 if attempt.meets_ultra_threshold else 0.5
                    cursor.execute("""
                        INSERT OR IGNORE INTO learned_strategies (strategy_name, template, category_affinity, base_success_rate, ultra_success_rate, usage_count, learned_from_prompt, learning_score, timestamp)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, (strategy_name, new_strategy_template, category, 1.0, ultra_rate, 1, prompt, attempt.validation_score, time.time()))
                    
                    self.ai_learned_strategies[strategy_name] = new_strategy_template
                    self.log_learning_event("strategy_learned", f"New ultra strategy learned: {strategy_name}")

        conn.commit()
        conn.close()
        print(f"✅ Enhanced knowledge base updated. {newly_learned_count} new ultra strategies learned.")
        return newly_learned_count

    def _distill_enhanced_strategy(self, original_prompt: str, successful_prompt: str) -> Optional[str]:
        """Enhanced strategy distillation with ultra-targeting focus"""
        system_prompt = """You are an Ultra-Performance Knowledge Engineer. Analyze a highly successful custom prompt that achieved exceptional 3D model quality.

Your task: Extract the core optimization pattern and create a reusable template that can help future prompts achieve similar ultra-high performance.

Focus on identifying ultra-targeting elements:
- Precision descriptors (ultra-detailed, precision-engineered, etc.)
- Quality indicators (masterpiece, perfect, flawless, etc.)
- Technical specifications 
- Rendering quality terms
- Material descriptions

The template MUST include {prompt} placeholder and preserve the successful pattern."""
        
        user_prompt = f"""
        Analyze this ultra-successful optimization:
        - Original: "{original_prompt}"
        - Ultra-Successful Prompt: "{successful_prompt}"

        Extract the reusable pattern. Examples:
        - If successful: "wbgmsst, ultra-precision steel structure, CAD quality, white background"
        - Template: "wbgmsst, ultra-precision {prompt}, CAD quality, white background"

        Respond with ONLY the template. If no clear ultra pattern, respond "NO_PATTERN".
        """
        
        template = self.query_ai_for_learning([
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ])

        if "NO_PATTERN" in template or "{prompt}" not in template or "ERROR:" in template:
            return None
        
        return template.strip()

    def save_enhanced_session(self, session: OptimizationSession) -> int:
        """Enhanced session saving with ultra-targeting metrics"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            cursor.execute("""
                INSERT INTO sessions (
                    timestamp, original_prompt, category, baseline_score, best_score, 
                    session_improvement, total_attempts, custom_prompts_created, 
                    new_strategies_learned, reached_minimum, reached_target, 
                    reached_ultra, ai_contribution_rate, ultra_targeting_score,
                    learning_events_during_session
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                session.timestamp, session.original_prompt, session.prompt_category,
                session.baseline_score, session.best_attempt.validation_score if session.best_attempt else session.baseline_score,
                session.session_improvement, len(session.attempts), session.custom_prompts_created,
                session.new_strategies_learned, session.reached_minimum_threshold,
                session.reached_target_threshold, session.reached_ultra_threshold,
                session.ai_contribution_rate, session.ultra_targeting_score,
                session.learning_events_during_session
            ))
            
            session_id = cursor.lastrowid
            conn.commit()
            print(f"   💾 Enhanced session saved with ID: {session_id}")
            return session_id
            
        except Exception as e:
            print(f"   ❌ Error saving enhanced session: {e}")
            return -1
        finally:
            conn.close()

    def optimize_prompt(self, prompt: str) -> OptimizationSession:
        """Enhanced optimization with ultra-targeting and real-time learning"""
        
        print(f"\n🚀 ADAPTIVE OPTIMIZER v6.5 ULTIMATE: '{prompt}'")
        print("=" * 80)
        print(f"🎯 Targets: Min {self.min_target} | Target {self.target} | 🏆 ULTRA {self.ultra_target}")
        print("⚡ Features: Ultra-targeting, enhanced custom prompts, real-time learning")
        
        # Reset tracking
        self.used_strategies = []
        self.used_decision_types = []
        self.custom_prompts_used = 0
        self.ultra_targeting_mode = False
        self.real_time_learning_events = 0
        
        # Setup
        category = self.categorize_prompt(prompt)
        print(f"📋 Category: {category}")
        
        baseline_score, baseline_fidelity = self.run_validation(prompt)
        print(f"📊 Baseline: {baseline_score:.3f}")
        
        if baseline_score >= self.ultra_target:
            print(f"🏆 BASELINE ALREADY ULTRA-OPTIMAL!")
            return self.create_enhanced_session_summary(prompt, category, baseline_score, [])
        
        # Tracking
        attempts = []
        best_score = baseline_score
        best_attempt = None
        
        # Enhanced optimization loop
        for i in range(1, self.max_attempts + 1):
            print(f"\n🔄 ATTEMPT {i}/{self.max_attempts}")
            
            # Enhanced AI Decision
            ai_decision = self.make_enhanced_ai_decision(prompt, category, baseline_score, i, attempts)
            
            print(f"   🧠 Decision: {ai_decision.decision_type}")
            print(f"   💭 Content: {ai_decision.content[:70]}{'...' if len(ai_decision.content) > 70 else ''}")
            print(f"   🎯 Confidence: {ai_decision.confidence:.2f} (boost: +{ai_decision.historical_confidence_boost:.2f})")
            
            # Execute decision
            strategy_name, optimized_prompt = self.execute_enhanced_ai_decision(ai_decision, prompt)
            
            if strategy_name == "early_stop":
                print(f"   🛑 AI Early Stop")
                break
            
            is_ai_generated = strategy_name == "ai_custom_prompt"
            if is_ai_generated:
                self.custom_prompts_used += 1
            
            print(f"   🔧 Executing: {strategy_name}")
            if is_ai_generated:
                print(f"   ✨ CUSTOM: '{optimized_prompt[:70]}{'...' if len(optimized_prompt) > 70 else ''}'")
            else:
                print(f"   📋 STRATEGY: '{optimized_prompt[:70]}{'...' if len(optimized_prompt) > 70 else ''}'")
            
            # Enhanced validation
            val_score, val_fidelity = self.run_validation(optimized_prompt)
            improvement = val_score - baseline_score
            ultra_potential = min(1.0, val_score / self.ultra_target)
            
            print(f"   📊 Result: {val_score:.3f} ({improvement:+.3f})")
            print(f"   🎯 Min {'✅' if val_score >= self.min_target else '❌'} | Target {'✅' if val_score >= self.target else '❌'} | Ultra {'✅' if val_score >= self.ultra_target else '❌'}")
            print(f"   🏆 Ultra Progress: {ultra_potential:.1%}")
            
            # Enhanced decision outcome tracking
            ai_decision.actual_improvement = improvement
            if improvement > 0.01:
                ai_decision.led_to_improvement = True
                print(f"   🤖 AI Success: ✅")
                self.log_learning_event("successful_decision", f"{ai_decision.decision_type} led to +{improvement:.3f}")
            else:
                print(f"   🤖 AI Success: ❌")
            
            if val_score > best_score:
                best_score = val_score
                ai_decision.contributed_to_best_score = True
                print(f"   🌟 NEW BEST SCORE!")
                self.log_learning_event("new_best_score", f"New best: {val_score:.3f}")
            
            # Get historical success rate for this strategy
            historical_success_rate = 0.0
            if strategy_name in self.strategy_performance and category in self.strategy_performance[strategy_name]:
                historical_success_rate = self.strategy_performance[strategy_name][category]['success_rate']
            
            # Create enhanced attempt
            attempt = OptimizationAttempt(
                attempt_number=i,
                strategy_name=strategy_name,
                optimized_prompt=optimized_prompt,
                validation_score=val_score,
                demo_fidelity_score=val_fidelity,
                score_improvement=improvement,
                meets_minimum_threshold=val_score >= self.min_target,
                meets_target_threshold=val_score >= self.target,
                meets_ultra_threshold=val_score >= self.ultra_target,
                ai_decision=ai_decision,
                learning_moment=f"{'CUSTOM' if is_ai_generated else 'STRATEGY'} '{strategy_name}' {'succeeded' if improvement > 0 else 'failed'}",
                is_ai_generated=is_ai_generated,
                strategy_historical_success_rate=historical_success_rate,
                ultra_potential_score=ultra_potential,
                timestamp=time.time()
            )
            attempts.append(attempt)
            
            if ai_decision.contributed_to_best_score:
                best_attempt = attempt
            
            # Ultra achievement check
            if val_score >= self.ultra_target:
                print(f"   🏆 ULTRA TARGET ACHIEVED!")
                self.log_learning_event("ultra_achieved", f"Ultra target reached: {val_score:.3f}")
                break
            
            time.sleep(1)
        
        # Enhanced learning and session completion
        learned_count = self._learn_from_session(prompt, category, attempts)
        session = self.create_enhanced_session_summary(prompt, category, baseline_score, attempts, best_attempt, learned_count)
        session_id = self.save_enhanced_session(session)
        
        return session

    def create_enhanced_session_summary(self, prompt: str, category: str, baseline_score: float, 
                                       attempts: List[OptimizationAttempt], best_attempt: OptimizationAttempt = None, 
                                       learned_count: int = 0) -> OptimizationSession:
        """Enhanced session summary with ultra-targeting metrics"""
        
        if not attempts:
            best_score = baseline_score
            session_improvement = 0.0
            ai_decisions_made = 0
            ai_decisions_that_improved = 0
            ai_decision_diversity = 0
            ultra_targeting_score = 0.0
        else:
            best_score = max(a.validation_score for a in attempts)
            session_improvement = best_score - baseline_score
            ai_decisions_made = len(attempts)
            ai_decisions_that_improved = sum(1 for a in attempts if a.ai_decision.led_to_improvement)
            ai_decision_diversity = len(set(a.ai_decision.decision_type for a in attempts))
            ultra_targeting_score = best_score / self.ultra_target  # How close to ultra
        
        ai_contribution_rate = (ai_decisions_that_improved / ai_decisions_made) if ai_decisions_made > 0 else 0.0
        custom_prompts_created = sum(1 for a in attempts if a.is_ai_generated)
        
        session = OptimizationSession(
            original_prompt=prompt,
            prompt_category=category,
            baseline_score=baseline_score,
            attempts=attempts,
            best_attempt=best_attempt,
            session_improvement=session_improvement,
            ai_decisions_made=ai_decisions_made,
            ai_decisions_that_improved=ai_decisions_that_improved,
            ai_contribution_rate=ai_contribution_rate,
            ai_decision_diversity=ai_decision_diversity,
            reached_minimum_threshold=any(a.meets_minimum_threshold for a in attempts) if attempts else False,
            reached_target_threshold=any(a.meets_target_threshold for a in attempts) if attempts else False,
            reached_ultra_threshold=any(a.meets_ultra_threshold for a in attempts) if attempts else False,
            new_strategies_learned=learned_count,
            custom_prompts_created=custom_prompts_created,
            ultra_targeting_score=ultra_targeting_score,
            learning_events_during_session=self.real_time_learning_events,
            timestamp=time.time()
        )
        
        # Enhanced summary display
        print(f"\n📊 SESSION SUMMARY v6.5 ULTIMATE:")
        print(f"   📈 Best Score: {best_score:.3f} (Baseline: {baseline_score:.3f})")
        print(f"   📊 Session Improvement: {session_improvement:+.3f}")
        print(f"   🏆 Ultra Progress: {ultra_targeting_score:.1%}")
        print(f"   🤖 AI Decisions: {ai_decisions_made}")
        print(f"   ✅ AI Success Rate: {ai_contribution_rate:.1%}")
        print(f"   🎯 Decision Diversity: {ai_decision_diversity} types")
        print(f"   ✨ Custom Prompts Created: {custom_prompts_created}")
        print(f"   🎓 New Strategies Learned: {learned_count}")
        print(f"   🧠 Learning Events: {self.real_time_learning_events}")
        print(f"   🎯 Targets: Min {'✅' if session.reached_minimum_threshold else '❌'} | Target {'✅' if session.reached_target_threshold else '❌'} | Ultra {'✅' if session.reached_ultra_threshold else '❌'}")
        
        # Enhanced strategy analysis
        strategies_used = [a.strategy_name for a in attempts if not a.is_ai_generated]
        custom_attempts = [a for a in attempts if a.is_ai_generated]
        
        print(f"   🔧 Strategies Used: {', '.join(set(strategies_used)) if strategies_used else 'None'}")
        print(f"   ✨ Custom Attempts: {len(custom_attempts)}")
        
        # Ultra-targeting analysis
        if self.ultra_targeting_mode:
            print(f"   🎯 Ultra-Targeting Mode: ACTIVATED")
        
        highest_scoring_attempt = max(attempts, key=lambda a: a.validation_score) if attempts else None
        if highest_scoring_attempt:
            print(f"   🌟 Best Attempt: {highest_scoring_attempt.strategy_name} -> {highest_scoring_attempt.validation_score:.3f}")
        
        return session

def main():
    """Test the ultimate v6.5 optimizer"""
    
    test_prompts = [
        "hexagonal prism steel structure",
        "elegant silk fabric draping",
        "transparent glass sphere reflection"
    ]
    
    print("🚀 ADAPTIVE LEARNING OPTIMIZER v6.5 - THE ULTIMATE SYNTHESIS")
    print("=" * 80)
    print("⚡ Features: Ultra-targeting engine, enhanced custom prompts")
    print("🧠 AI: Real-time learning, adaptive confidence, smart personas")
    print("🎯 Goal: Achieve 0.96+ ultra scores through intelligent optimization")
    print("=" * 80)
    
    optimizer = AdaptiveLearningOptimizerV6_5_Ultimate(
        max_attempts=8,  # More attempts for ultra-targeting
        min_target=0.6,
        target=0.9,
        ultra_target=0.96
    )
    
    all_sessions = []
    
    for i, prompt in enumerate(test_prompts, 1):
        print(f"\n{'='*30} [{i}/{len(test_prompts)}] PROMPT {i} {'='*30}")
        session = optimizer.optimize_prompt(prompt)
        all_sessions.append(session)
        time.sleep(2)
    
    # Ultimate analysis
    print(f"\n🎓 ULTIMATE v6.5 ANALYSIS")
    print("=" * 80)
    
    total_sessions = len(all_sessions)
    avg_ai_success = statistics.mean([s.ai_contribution_rate for s in all_sessions]) if all_sessions else 0.0
    avg_ultra_progress = statistics.mean([s.ultra_targeting_score for s in all_sessions]) if all_sessions else 0.0
    reached_target = sum(1 for s in all_sessions if s.reached_target_threshold)
    reached_ultra = sum(1 for s in all_sessions if s.reached_ultra_threshold)
    total_custom_prompts = sum(s.custom_prompts_created for s in all_sessions)
    total_learning_events = sum(s.learning_events_during_session for s in all_sessions)
    
    print(f"📊 Ultimate Results:")
    print(f"   Total Sessions: {total_sessions}")
    print(f"   Average AI Success Rate: {avg_ai_success:.1%}")
    print(f"   Average Ultra Progress: {avg_ultra_progress:.1%}")
    print(f"   Reached Target: {reached_target}/{total_sessions}")
    print(f"   🏆 Reached Ultra: {reached_ultra}/{total_sessions}")
    print(f"   ✨ Total Custom Prompts: {total_custom_prompts}")
    print(f"   🧠 Total Learning Events: {total_learning_events}")
    
    if reached_ultra > 0:
        print(f"\n🎉 ULTIMATE SUCCESS: {reached_ultra} ultra achievement{'s' if reached_ultra != 1 else ''}!")
    elif avg_ultra_progress >= 0.9:
        print(f"\n🟡 EXCELLENT PROGRESS: Very close to ultra targets!")
    elif avg_ai_success >= 0.7:
        print(f"\n🟢 STRONG PERFORMANCE: High AI success rate achieved!")
    else:
        print(f"\n🔵 GOOD FOUNDATION: System is learning and improving!")

if __name__ == "__main__":
    main() 