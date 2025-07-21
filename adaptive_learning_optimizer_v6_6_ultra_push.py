#!/usr/bin/env python3
"""
Adaptive Learning Optimizer v6.6 - Ultra Achievement Focus
Purpose: Specifically designed to push for 0.96+ ultra scores through aggressive custom prompt usage

Key Changes in v6.6:
- Mandatory Custom Prompts: Forces custom prompts after 2-3 strategy attempts
- Ultra-Focused AI Instructions: AI specifically trained to push for ultra scores
- Aggressive Confidence Boosting: Higher confidence for custom prompts
- Ultra Prompt Templates: Specialized templates for achieving 0.96+ scores
- Early Custom Prompt Encouragement: Starts suggesting custom prompts earlier
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

# Reuse the same dataclasses from v6.5
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
    actual_improvement: float = 0.0
    led_to_improvement: bool = False
    contributed_to_best_score: bool = False
    historical_confidence_boost: float = 0.0
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
    strategy_historical_success_rate: float = 0.0
    ultra_potential_score: float = 0.0
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
    ultra_targeting_score: float = 0.0
    learning_events_during_session: int = 0
    timestamp: float = 0.0

class AdaptiveLearningOptimizerV6_6_UltraPush:
    """Ultra Achievement Focused Optimizer - Designed to reach 0.96+ scores"""

    def __init__(self, max_attempts: int = 8, min_target: float = 0.6,
                 target: float = 0.9, ultra_target: float = 0.96):
        
        self.ollama_url = "http://localhost:11434"
        self.model_name = "deepseek-r1:1.5b"
        self.db_path = "adaptive_optimizer_v6_6_ultra.db"
        self.max_attempts = max_attempts
        self.min_target = min_target
        self.target = target
        self.ultra_target = ultra_target

        # Ultra-focused configuration
        self.force_custom_prompt_after = 2  # Force custom prompts after just 2 strategy attempts
        self.ultra_mode_threshold = 0.75  # Activate ultra mode earlier
        self.custom_prompt_confidence_boost = 0.25  # Higher confidence for custom prompts
        
        # Tracking
        self.used_strategies = []
        self.used_decision_types = []
        self.max_memory = 2  # Shorter memory for more diversity
        self.custom_prompts_used = 0
        self.ultra_targeting_mode = False
        self.real_time_learning_events = 0
        
        # Ultra-targeting strategy library (enhanced for 0.96+ scores)
        self.strategies = {
            "material_focus": "wbgmsst, solid {prompt} object 3D, white background",
            "geometric_focus": "wbgmsst, {prompt} geometric 3D model, white background",
            "basic_description": "3D model of {prompt}",
            "enhanced_clarity": "wbgmsst, detailed 3D {prompt} model, accurate geometry, white background",
            "concrete_object": "wbgmsst, {prompt} as 3D object, realistic proportions, white background",
            "professional_render": "wbgmsst, professional 3D render of {prompt}, studio lighting, white background",
            "high_quality": "wbgmsst, high quality 3D model {prompt}, detailed textures, white background",
            # Ultra-specific strategies for 0.96+ scores
            "ultra_precision": "wbgmsst, ultra-precision {prompt}, aerospace engineering quality, perfect geometry, CAD-accurate dimensions, white background",
            "masterpiece_quality": "wbgmsst, masterpiece-grade {prompt}, museum exhibition quality, flawless execution, award-winning visualization, white background",
            "ultra_technical": "wbgmsst, ultra-technical {prompt}, precision manufacturing quality, engineering perfection, ultra-detailed specifications, white background",
            "ultra_artistic": "wbgmsst, ultra-refined artistic {prompt}, gallery masterpiece quality, perfect artistic execution, museum-grade detail, white background"
        }
        
        # Ultra-achievement custom prompt templates
        self.ultra_custom_templates = {
            "technical": [
                "wbgmsst, ultra-precision aerospace-grade {prompt}, CAD-perfect geometry, engineering masterpiece, flawless technical execution, white background",
                "wbgmsst, precision-engineered {prompt}, ultra-high technical specification, manufacturing perfection, award-winning design excellence, white background",
                "wbgmsst, masterpiece-quality technical {prompt}, ultra-detailed engineering precision, perfect dimensional accuracy, exhibition-grade rendering, white background"
            ],
            "artistic": [
                "wbgmsst, ultra-refined masterpiece {prompt}, world-class artistic excellence, museum-quality sculpture, perfect aesthetic execution, white background",
                "wbgmsst, award-winning artistic {prompt}, gallery exhibition centerpiece, flawless creative vision, ultra-premium aesthetic quality, white background",
                "wbgmsst, ultra-sophisticated artistic {prompt}, masterpiece-grade refinement, perfect artistic balance, museum-standard detail, white background"
            ],
            "textile": [
                "wbgmsst, ultra-luxury couture {prompt}, masterpiece-grade fabric simulation, perfect textile physics, haute couture excellence, white background",
                "wbgmsst, museum-quality {prompt}, ultra-realistic textile behavior, perfect draping physics, couture perfection, white background",
                "wbgmsst, ultra-premium textile {prompt}, flawless fabric texture simulation, perfect material physics, exhibition standard, white background"
            ],
            "physical": [
                "wbgmsst, ultra-premium {prompt}, perfection-grade quality, flawless proportions, masterpiece rendering, award-winning visualization, white background",
                "wbgmsst, exhibition-quality {prompt}, ultra-detailed perfection, museum-standard accuracy, world-class rendering excellence, white background",
                "wbgmsst, masterpiece-grade {prompt}, ultra-high precision modeling, perfect craftsmanship execution, gallery-quality visualization, white background"
            ]
        }
        
        # Learning systems
        self.ai_learned_strategies = {}
        self.strategy_performance = {}
        
        self.setup_database()
        self.load_historical_learning()

    def setup_database(self):
        """Database setup (same as v6.5)"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
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
        """Historical learning (same as v6.5)"""
        print("🧠 Loading ultra-focused historical knowledge...")
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            cursor.execute("SELECT strategy_name, template, ultra_success_rate FROM learned_strategies WHERE base_success_rate > 0.4")
            ultra_strategies = 0
            for row in cursor.fetchall():
                name, template, ultra_rate = row
                self.ai_learned_strategies[name] = template
                if ultra_rate and ultra_rate > 0.5:
                    ultra_strategies += 1
                print(f"   📚 Loaded: {name} (Ultra rate: {ultra_rate:.1%})")
            
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
            
            print(f"✅ Ultra knowledge loaded: {len(self.ai_learned_strategies)} learned strategies ({ultra_strategies} ultra-capable)")
            print(f"   📊 {len(self.strategy_performance)} strategy records ({ultra_capable_strategies} with ultra achievements)")
            
        except sqlite3.OperationalError as e:
            print(f"   📝 Starting fresh v6.6 ultra knowledge base: {e}")
        finally:
            conn.close()

    def query_ai_enhanced(self, user_message: str, timeout: int = 50) -> str:
        """Enhanced AI query (same as v6.5)"""
        data = {
            "model": self.model_name,
            "messages": [{"role": "user", "content": user_message}],
            "stream": False,
            "options": {
                "temperature": 0.9,  # Higher creativity for ultra achievements
                "top_p": 0.95,
                "num_predict": 300,
                "stop": ["<think>", "</think>"],
                "repeat_penalty": 1.1
            }
        }
        
        try:
            response = requests.post(f"{self.ollama_url}/api/chat", json=data, timeout=timeout)
            response.raise_for_status()
            content = response.json()["message"]["content"]
            
            content = content.replace("<think>", "").replace("</think>", "")
            content = re.sub(r'\*\*([^*]+)\*\*', r'\1', content)
            content = re.sub(r'(?i)^(sure|okay|alright)[.,!]?\s*', '', content)
            
            return content.strip()
        except Exception as e:
            return f"ERROR: {str(e)}"

    def run_validation(self, prompt: str) -> Tuple[float, float]:
        """Validation (using mock validator)"""
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
                
                if val_score >= self.ultra_target:
                    print(f"   🏆 ULTRA ACHIEVEMENT: {val_score:.3f}!")
                    self.log_learning_event("ultra_achieved", f"Ultra target reached: {val_score:.3f}")
                elif val_score >= self.ultra_mode_threshold:
                    print(f"   🎯 Ultra proximity: {val_score:.3f} (need {self.ultra_target - val_score:.3f} more)")
                
                return val_score, demo_score
                
        except Exception as e:
            print(f"   ❌ Validation error: {e}")
            return 0.0, 0.0

    def categorize_prompt(self, prompt: str) -> str:
        """Categorization (same as v6.5)"""
        prompt_lower = prompt.lower()
        
        if any(word in prompt_lower for word in ["steel", "metal", "copper", "iron", "aluminum", "geometric", "prism", "cylinder", "sphere", "hexagonal", "technical", "engineering", "mechanical"]):
            return "technical"
        elif any(word in prompt_lower for word in ["elegant", "artistic", "ornate", "beautiful", "sculpture", "masterpiece", "gallery", "refined", "graceful"]):
            return "artistic"
        elif any(word in prompt_lower for word in ["fabric", "silk", "cotton", "textile", "cloth", "draping", "weaving", "fiber"]):
            return "textile"
        else:
            return "physical"

    def activate_ultra_targeting_mode(self, current_best_score: float):
        """Activate ultra-targeting mode earlier and more aggressively"""
        if current_best_score >= self.ultra_mode_threshold and not self.ultra_targeting_mode:
            self.ultra_targeting_mode = True
            print(f"   🎯 ULTRA-TARGETING MODE ACTIVATED! (Current: {current_best_score:.3f}, Target: {self.ultra_target})")
            self.log_learning_event("ultra_targeting_activated", f"Activated at score {current_best_score:.3f}")
            return True
        return False

    def log_learning_event(self, event_type: str, description: str, confidence_change: float = 0.0):
        """Log real-time learning events"""
        self.real_time_learning_events += 1
        print(f"   🧠 Learning Event: {description}")

    def should_force_custom_prompt(self, attempt_num: int, attempts: List[OptimizationAttempt], current_best: float) -> bool:
        """Determine if we should force a custom prompt"""
        
        # Force custom prompts earlier and more aggressively
        custom_prompts_used = sum(1 for a in attempts if a.is_ai_generated)
        strategy_attempts = len(attempts) - custom_prompts_used
        
        # Force custom prompt conditions
        if attempt_num > self.force_custom_prompt_after and custom_prompts_used == 0:
            return True
        
        if current_best >= self.ultra_mode_threshold and custom_prompts_used < 2:
            return True
        
        if attempt_num > 4 and custom_prompts_used < attempt_num // 2:
            return True
        
        # If last 2 attempts didn't improve, force custom
        if len(attempts) >= 2 and all(a.score_improvement <= 0.01 for a in attempts[-2:]):
            return True
        
        return False

    def make_ultra_focused_ai_decision(self, prompt: str, category: str, baseline_score: float, 
                                      attempt_num: int, attempts: List[OptimizationAttempt]) -> AIDecision:
        """Ultra-focused AI decision making with aggressive custom prompt encouragement"""
        
        current_best = max([a.validation_score for a in attempts] + [baseline_score])
        custom_prompts_used = sum(1 for a in attempts if a.is_ai_generated)
        
        # Activate ultra-targeting if close
        self.activate_ultra_targeting_mode(current_best)
        
        # Force custom prompt if conditions are met
        force_custom = self.should_force_custom_prompt(attempt_num, attempts, current_best)
        
        # Ultra-focused persona selection
        if self.ultra_targeting_mode:
            persona = "Ultra Achievement Specialist"
            confidence_boost = 0.2
        elif force_custom or current_best >= self.ultra_mode_threshold:
            persona = "Custom Prompt Ultra Master"
            confidence_boost = self.custom_prompt_confidence_boost
        elif attempt_num > self.force_custom_prompt_after:
            persona = "Custom Prompt Champion"
            confidence_boost = 0.15
        else:
            persona = "Strategic Foundation Builder"
            confidence_boost = 0.05
        
        # Enhanced context for ultra achievement
        recent_results = []
        for attempt in attempts[-3:]:
            custom_indicator = " (CUSTOM 🎨)" if attempt.is_ai_generated else ""
            ultra_indicator = " 🏆ULTRA" if attempt.meets_ultra_threshold else " 🎯ULTRA-CLOSE" if attempt.validation_score >= self.ultra_mode_threshold else ""
            recent_results.append(f"Attempt {attempt.attempt_number}: {attempt.strategy_name}{custom_indicator} -> {attempt.validation_score:.3f}{ultra_indicator}")
        
        # Available strategies (prefer ultra strategies)
        ultra_strategies = ["ultra_precision", "masterpiece_quality", "ultra_technical", "ultra_artistic"]
        regular_strategies = [k for k in self.strategies.keys() if k not in ultra_strategies]
        
        # Recent anti-repetition
        recently_used = [a.strategy_name for a in attempts[-self.max_memory:] if a.strategy_name != "ai_custom_prompt"]
        available_strategies = [s for s in ultra_strategies + regular_strategies if s not in recently_used]
        
        if len(available_strategies) < 5:
            available_strategies.extend([s for s in ultra_strategies + regular_strategies if s not in available_strategies][:3])
        
        # Ultra-focused encouragement
        ultra_distance = self.ultra_target - current_best
        custom_encouragement = ""
        
        if force_custom:
            custom_encouragement = f"""
🚨 MANDATORY CUSTOM PROMPT CREATION:
After {attempt_num} attempts, you MUST create a custom prompt! Strategies alone won't reach ULTRA.
ULTRA target: {self.ultra_target} (need +{ultra_distance:.3f} more)
Custom prompts are proven to achieve the highest scores - CREATE ONE NOW!"""
        elif self.ultra_targeting_mode:
            custom_encouragement = f"""
🎯 ULTRA-TARGETING MODE ACTIVE:
You're {ultra_distance:.3f} points away from ULTRA glory!
Custom prompts with ultra-premium language often achieve 0.95+ scores!
Use words like: ultra-precision, masterpiece-grade, aerospace-quality, museum-standard"""
        elif current_best >= self.ultra_mode_threshold:
            custom_encouragement = f"""
🔥 ULTRA ACHIEVEMENT ZONE:
You're in the ultra zone! ({current_best:.3f}/{self.ultra_target})
Custom prompts are your BEST bet for the final push to ULTRA!
Think: precision-engineered, ultra-detailed, masterpiece-quality"""
        elif custom_prompts_used == 0:
            custom_encouragement = f"""
💡 CUSTOM PROMPT OPPORTUNITY:
No custom prompts used yet - this is your advantage!
Custom prompts typically outperform strategies by 0.1-0.2 points!
Time to unleash your creative optimization power!"""
        
        # Ultra-achievement user message
        user_message = f"""You are an {persona} with ONE MISSION: Achieve 0.96+ ULTRA scores through masterful optimization.

🏆 ULTRA ACHIEVEMENT MISSION:
- Current Best: {current_best:.3f} | ULTRA Target: {self.ultra_target} | Gap: {ultra_distance:.3f}
- Attempt: {attempt_num}/{self.max_attempts} | Custom Used: {custom_prompts_used}
- Ultra Mode: {'🎯 ACTIVE' if self.ultra_targeting_mode else 'STANDBY'}

RECENT PERFORMANCE:
{chr(10).join(recent_results) if recent_results else "None yet"}

{custom_encouragement}

AVAILABLE STRATEGIES (prioritized for ULTRA):
Ultra Strategies: {', '.join([s for s in ultra_strategies if s in available_strategies])}
Regular Strategies: {', '.join([s for s in regular_strategies if s in available_strategies])}

🎯 ULTRA ACHIEVEMENT TIPS:
- Custom prompts achieve the highest scores (often 0.85-0.95+)
- Use ultra-premium language: "ultra-precision", "masterpiece-grade", "aerospace-quality"
- Technical prompts: Add CAD-quality, engineering-perfection, precision-manufacturing
- Artistic prompts: Add museum-quality, gallery-masterpiece, award-winning
- Combine multiple premium descriptors for maximum impact

DECISION OPTIONS:
A) WRITE_CUSTOM: Create ultra-optimized custom prompt ⭐ RECOMMENDED FOR ULTRA ⭐
B) USE_STRATEGY: Select ultra-focused strategy (if no custom prompts available)
C) EARLY_STOP: Only if already achieved ULTRA or completely hopeless

RESPOND FORMAT:
CHOICE: [A/B/C]
STRATEGY_OR_PROMPT: [strategy name OR complete ultra-optimized custom prompt OR stop reason]
REASONING: [detailed explanation focusing on ULTRA achievement]
CONFIDENCE: [0.1-1.0] (boost +{confidence_boost:.2f} already applied)

Your ultra-focused response:"""

        print(f"🤖 AI Persona: {persona}")
        print(f"   🎯 Force Custom: {'YES' if force_custom else 'NO'}")
        print(f"   ✨ Custom Used: {custom_prompts_used}/{attempt_num}")
        print(f"   🏆 Ultra Mode: {'ACTIVE' if self.ultra_targeting_mode else 'STANDBY'}")
        print(f"   📊 Ultra Gap: {ultra_distance:.3f}")
        
        ai_response = self.query_ai_enhanced(user_message)
        
        if "ERROR:" in ai_response:
            return self.create_ultra_fallback_decision(attempt_num, persona, ai_response, available_strategies, 
                                                     confidence_boost, force_custom, category)
        
        print(f"   🤖 AI Response: {ai_response[:150]}...")
        
        decision = self.parse_ultra_response(ai_response, attempt_num, persona, available_strategies, 
                                           force_custom, category, prompt)
        decision.historical_confidence_boost = confidence_boost
        decision.confidence = min(1.0, decision.confidence + confidence_boost)
        
        return decision

    def parse_ultra_response(self, response: str, attempt_num: int, persona: str, available_strategies: List[str],
                           force_custom: bool, category: str, original_prompt: str) -> AIDecision:
        """Enhanced parsing with ultra-focusing"""
        
        decision_type = "WRITE_CUSTOM" if force_custom else "USE_STRATEGY"
        content = self.generate_ultra_custom_prompt(original_prompt, category)  # Default ultra custom
        reasoning = response[:200]
        confidence = 0.7
        
        try:
            # Extract choice
            choice_patterns = [
                r'CHOICE:\s*([ABC])',
                r'DECISION:\s*([ABC])',
                r'I choose\s*([ABC])'
            ]
            
            choice = None
            for pattern in choice_patterns:
                match = re.search(pattern, response, re.IGNORECASE)
                if match:
                    choice = match.group(1).upper()
                    break
            
            # Force custom if required, regardless of AI choice
            if force_custom:
                choice = "A"
                decision_type = "WRITE_CUSTOM"
            elif choice == "A":
                decision_type = "WRITE_CUSTOM"
            elif choice == "B":
                decision_type = "USE_STRATEGY"
            elif choice == "C":
                decision_type = "EARLY_STOP"
            
            # Extract content
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
            
            # Extract reasoning
            reasoning_patterns = [
                r'REASONING:\s*(.+?)(?=CONFIDENCE:|$)',
                r'EXPLANATION:\s*(.+?)(?=CONFIDENCE:|$)'
            ]
            
            for pattern in reasoning_patterns:
                match = re.search(pattern, response, re.DOTALL | re.IGNORECASE)
                if match:
                    reasoning = match.group(1).strip()
                    break
            
            # Extract confidence
            conf_patterns = [
                r'CONFIDENCE:\s*([0-9.]+)',
                r'confidence[:\s]*([0-9.]+)'
            ]
            
            for pattern in conf_patterns:
                match = re.search(pattern, response, re.IGNORECASE)
                if match:
                    confidence = min(1.0, max(0.1, float(match.group(1))))
                    break
            
            # Enhance custom prompts for ultra achievement
            if decision_type == "WRITE_CUSTOM":
                content = self.enhance_ultra_custom_prompt(content, category, original_prompt, reasoning)
            
            # Validate strategies
            elif decision_type == "USE_STRATEGY":
                all_strategies = {**self.strategies, **self.ai_learned_strategies}
                if content not in all_strategies:
                    # Find best match
                    best_match = None
                    for strategy in available_strategies:
                        if strategy in content.lower() or any(word in content.lower() for word in strategy.split('_')):
                            best_match = strategy
                            break
                    
                    content = best_match if best_match else available_strategies[0]
            
        except Exception as e:
            print(f"   ⚠️ Ultra parsing error: {e}, using ultra fallback")
            
            # Advanced semantic parsing for ultra achievement
            response_lower = response.lower()
            
            if force_custom or any(indicator in response_lower for indicator in ["custom", "write", "create", "ultra", "precision", "masterpiece"]):
                decision_type = "WRITE_CUSTOM"
                content = self.generate_ultra_custom_prompt(original_prompt, category)
            else:
                decision_type = "USE_STRATEGY"
                # Prefer ultra strategies
                ultra_strats = [s for s in available_strategies if "ultra" in s]
                content = ultra_strats[0] if ultra_strats else available_strategies[0]
        
        return AIDecision(
            attempt_number=attempt_num,
            persona_used=persona,
            decision_type=decision_type,
            content=content,
            reasoning=reasoning,
            confidence=confidence,
            expected_improvement=confidence * 0.3,  # Higher expected improvement for ultra focus
            conversation_turn=attempt_num,
            based_on_summary=False,
            timestamp=time.time()
        )

    def enhance_ultra_custom_prompt(self, raw_content: str, category: str, original_prompt: str, reasoning: str) -> str:
        """Enhance custom prompts specifically for 0.96+ ultra achievement"""
        
        # If content is too short or missing key elements, generate from scratch
        if len(raw_content) < 30 or not any(word in raw_content.lower() for word in ["wbgmsst", "ultra", "precision", "masterpiece"]):
            return self.generate_ultra_custom_prompt(original_prompt, category)
        
        enhanced = raw_content
        
        # Ensure ultra-premium prefix
        if not enhanced.lower().startswith("wbgmsst"):
            enhanced = f"wbgmsst, {enhanced}"
        
        # Add ultra-achievement elements
        ultra_prefixes = ["ultra-precision", "masterpiece-grade", "aerospace-quality", "museum-standard", "precision-engineered"]
        if not any(prefix in enhanced.lower() for prefix in ultra_prefixes):
            enhanced = enhanced.replace("wbgmsst, ", f"wbgmsst, {ultra_prefixes[0]} ")
        
        # Add category-specific ultra elements
        if category == "technical" and "cad" not in enhanced.lower():
            enhanced = enhanced.replace("white background", "CAD-accurate dimensions, white background")
        elif category == "artistic" and "gallery" not in enhanced.lower():
            enhanced = enhanced.replace("white background", "gallery exhibition quality, white background")
        
        # Ensure ultra suffix
        if "white background" not in enhanced.lower():
            enhanced += ", white background"
        
        return enhanced

    def generate_ultra_custom_prompt(self, original_prompt: str, category: str) -> str:
        """Generate ultra-achievement focused custom prompt"""
        
        templates = self.ultra_custom_templates.get(category, self.ultra_custom_templates['physical'])
        
        # Randomly select an ultra template
        template = random.choice(templates)
        
        return template.format(prompt=original_prompt)

    def create_ultra_fallback_decision(self, attempt_num: int, persona: str, error_msg: str, 
                                     available_strategies: List[str], confidence_boost: float,
                                     force_custom: bool, category: str) -> AIDecision:
        """Ultra-focused fallback decision"""
        
        if force_custom or attempt_num > self.force_custom_prompt_after:
            return AIDecision(
                attempt_number=attempt_num,
                persona_used=persona,
                decision_type="WRITE_CUSTOM",
                content=self.generate_ultra_custom_prompt("object", category),
                reasoning=f"Ultra fallback custom prompt: {error_msg[:50]}",
                confidence=0.7 + confidence_boost,
                expected_improvement=0.2,
                conversation_turn=attempt_num,
                based_on_summary=False,
                historical_confidence_boost=confidence_boost,
                timestamp=time.time()
            )
        else:
            # Prefer ultra strategies
            ultra_strategies = [s for s in available_strategies if "ultra" in s]
            strategy = ultra_strategies[0] if ultra_strategies else available_strategies[0]
            
            return AIDecision(
                attempt_number=attempt_num,
                persona_used=persona,
                decision_type="USE_STRATEGY",
                content=strategy,
                reasoning=f"Ultra fallback strategy: {error_msg[:50]}",
                confidence=0.6 + confidence_boost,
                expected_improvement=0.15,
                conversation_turn=attempt_num,
                based_on_summary=False,
                historical_confidence_boost=confidence_boost,
                timestamp=time.time()
            )

    def execute_ultra_ai_decision(self, decision: AIDecision, prompt: str) -> Tuple[str, str]:
        """Execute AI decision with ultra-targeting enhancements"""
        
        if decision.decision_type == "EARLY_STOP":
            return "early_stop", prompt
        
        elif decision.decision_type == "WRITE_CUSTOM":
            custom_prompt = decision.content
            
            # Validate and enhance for ultra achievement
            if len(custom_prompt) > 30 and any(word in custom_prompt.lower() for word in ["wbgmsst", "ultra", "precision", "masterpiece"]):
                # Add ultra boost if in ultra mode
                if self.ultra_targeting_mode and "aerospace" not in custom_prompt.lower():
                    custom_prompt = custom_prompt.replace("wbgmsst, ", "wbgmsst, aerospace-grade ")
                return "ai_custom_prompt", custom_prompt
            else:
                # Generate enhanced ultra custom prompt
                category = self.categorize_prompt(prompt)
                enhanced = self.generate_ultra_custom_prompt(prompt, category)
                return "ai_custom_prompt", enhanced
        
        elif decision.decision_type == "USE_STRATEGY":
            strategy = decision.content
            all_strategies = {**self.strategies, **self.ai_learned_strategies}
            
            if strategy in all_strategies:
                executed_prompt = all_strategies[strategy].format(prompt=prompt)
                
                # Add ultra enhancement if in ultra mode
                if self.ultra_targeting_mode and strategy in self.strategies and "ultra" not in executed_prompt.lower():
                    executed_prompt = executed_prompt.replace("wbgmsst, ", "wbgmsst, ultra-premium ")
                
                return strategy, executed_prompt
            else:
                return "ultra_precision", self.strategies["ultra_precision"].format(prompt=prompt)
        
        return "ultra_precision", self.strategies["ultra_precision"].format(prompt=prompt)

    # Learning methods (reuse from v6.5 with minor enhancements)
    def _learn_from_session(self, prompt: str, category: str, attempts: List[OptimizationAttempt]) -> int:
        """Learn from session with ultra-achievement focus"""
        print("🧠 Updating ultra-focused knowledge base...")
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        newly_learned_count = 0

        # Track performance with ultra focus
        for attempt in attempts:
            strategy = attempt.strategy_name
            cursor.execute("SELECT success_rate, avg_improvement, usage_count, ultra_achievement_rate FROM strategy_performance WHERE strategy_name = ? AND category = ?", (strategy, category))
            row = cursor.fetchone()
            
            is_success = attempt.score_improvement > 0.01
            is_ultra = attempt.meets_ultra_threshold
            is_ultra_proximity = attempt.validation_score >= self.ultra_mode_threshold
            
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

        # Learn from ultra-achieving or ultra-proximity custom prompts
        for attempt in attempts:
            is_ultra_attempt = attempt.meets_ultra_threshold
            is_ultra_proximity_attempt = attempt.validation_score >= self.ultra_mode_threshold
            if attempt.is_ai_generated and (is_ultra_attempt or is_ultra_proximity_attempt):
                print(f"   🎓 Learning from ultra-proximity custom prompt (score: {attempt.validation_score:.3f})")
                new_strategy_template = self._distill_ultra_strategy(prompt, attempt.optimized_prompt)
                
                if new_strategy_template:
                    newly_learned_count += 1
                    strategy_name = f"ai_ultra_{category}_{int(time.time())}"
                    print(f"   ✅ Ultra Strategy Learned: '{strategy_name}'")
                    
                    ultra_rate = 1.0 if is_ultra_attempt else 0.7  # High rate for ultra-proximity
                    cursor.execute("""
                        INSERT OR IGNORE INTO learned_strategies (strategy_name, template, category_affinity, base_success_rate, ultra_success_rate, usage_count, learned_from_prompt, learning_score, timestamp)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, (strategy_name, new_strategy_template, category, 1.0, ultra_rate, 1, prompt, attempt.validation_score, time.time()))
                    
                    self.ai_learned_strategies[strategy_name] = new_strategy_template

        conn.commit()
        conn.close()
        print(f"✅ Ultra knowledge base updated. {newly_learned_count} new ultra strategies learned.")
        return newly_learned_count

    def _distill_ultra_strategy(self, original_prompt: str, successful_prompt: str) -> Optional[str]:
        """Distill ultra-achievement strategy"""
        system_prompt = """You are an Ultra-Achievement Knowledge Engineer. Analyze a successful custom prompt that achieved high scores (0.85+).

Extract the core ultra-optimization pattern that can be reused for similar prompts.

Focus on ultra-achievement elements:
- Premium descriptors (ultra-precision, masterpiece-grade, aerospace-quality)
- Technical excellence terms (CAD-accurate, engineering-perfection)
- Quality indicators (museum-standard, gallery-quality, award-winning)

Create a template with {prompt} placeholder that preserves the ultra-success pattern."""
        
        user_prompt = f"""
        Ultra-successful optimization analysis:
        - Original: "{original_prompt}"
        - Ultra-Success Prompt: "{successful_prompt}"

        Extract the ultra-achievement pattern. Example:
        - Success: "wbgmsst, aerospace-grade steel structure, ultra-precision CAD quality, white background"
        - Template: "wbgmsst, aerospace-grade {{prompt}}, ultra-precision CAD quality, white background"

        Respond with ONLY the template. If no clear ultra pattern, respond "NO_PATTERN".
        """
        
        try:
            template = self.query_ai_enhanced(user_prompt)
            if "NO_PATTERN" in template or "{prompt}" not in template or "ERROR:" in template:
                return None
            return template.strip()
        except:
            return None

    def save_ultra_session(self, session: OptimizationSession) -> int:
        """Save session with ultra-achievement tracking"""
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
            print(f"   💾 Ultra session saved with ID: {session_id}")
            return session_id
            
        except Exception as e:
            print(f"   ❌ Error saving ultra session: {e}")
            return -1
        finally:
            conn.close()

    def optimize_prompt(self, prompt: str) -> OptimizationSession:
        """Ultra-focused optimization with aggressive custom prompt usage"""
        
        print(f"\n🚀 ADAPTIVE OPTIMIZER v6.6 ULTRA PUSH: '{prompt}'")
        print("=" * 80)
        print(f"🏆 MISSION: Achieve {self.ultra_target}+ ULTRA scores through custom prompts")
        print(f"🎯 Targets: Min {self.min_target} | Target {self.target} | 🏆 ULTRA {self.ultra_target}")
        print("⚡ Strategy: Force custom prompts, ultra-targeting, premium language")
        
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
            return self.create_ultra_session_summary(prompt, category, baseline_score, [])
        
        # Tracking
        attempts = []
        best_score = baseline_score
        best_attempt = None
        
        # Ultra-focused optimization loop
        for i in range(1, self.max_attempts + 1):
            print(f"\n🔄 ATTEMPT {i}/{self.max_attempts}")
            
            # Ultra-focused AI Decision
            ai_decision = self.make_ultra_focused_ai_decision(prompt, category, baseline_score, i, attempts)
            
            print(f"   🧠 Decision: {ai_decision.decision_type}")
            print(f"   💭 Content: {ai_decision.content[:80]}{'...' if len(ai_decision.content) > 80 else ''}")
            print(f"   🎯 Confidence: {ai_decision.confidence:.2f} (boost: +{ai_decision.historical_confidence_boost:.2f})")
            
            # Execute decision
            strategy_name, optimized_prompt = self.execute_ultra_ai_decision(ai_decision, prompt)
            
            if strategy_name == "early_stop":
                print(f"   🛑 AI Early Stop")
                break
            
            is_ai_generated = strategy_name == "ai_custom_prompt"
            if is_ai_generated:
                self.custom_prompts_used += 1
            
            print(f"   🔧 Executing: {strategy_name}")
            if is_ai_generated:
                print(f"   ✨ CUSTOM: '{optimized_prompt[:80]}{'...' if len(optimized_prompt) > 80 else ''}'")
            else:
                print(f"   📋 STRATEGY: '{optimized_prompt[:80]}{'...' if len(optimized_prompt) > 80 else ''}'")
            
            # Validation
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
            
            # Create attempt
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
                ultra_potential_score=ultra_potential,
                timestamp=time.time()
            )
            attempts.append(attempt)
            
            if ai_decision.contributed_to_best_score:
                best_attempt = attempt
            
            # Ultra achievement check
            if val_score >= self.ultra_target:
                print(f"   🏆 ULTRA TARGET ACHIEVED!")
                break
            
            time.sleep(1)
        
        # Learning and session completion
        learned_count = self._learn_from_session(prompt, category, attempts)
        session = self.create_ultra_session_summary(prompt, category, baseline_score, attempts, best_attempt, learned_count)
        session_id = self.save_ultra_session(session)
        
        return session

    def create_ultra_session_summary(self, prompt: str, category: str, baseline_score: float, 
                                   attempts: List[OptimizationAttempt], best_attempt: OptimizationAttempt = None, 
                                   learned_count: int = 0) -> OptimizationSession:
        """Create session summary with ultra-achievement focus"""
        
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
            ultra_targeting_score = best_score / self.ultra_target
        
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
        
        # Ultra-focused summary display
        print(f"\n📊 SESSION SUMMARY v6.6 ULTRA PUSH:")
        print(f"   📈 Best Score: {best_score:.3f} (Baseline: {baseline_score:.3f})")
        print(f"   📊 Session Improvement: {session_improvement:+.3f}")
        print(f"   🏆 Ultra Achievement: {ultra_targeting_score:.1%}")
        print(f"   🎯 Ultra Gap: {self.ultra_target - best_score:.3f}")
        print(f"   🤖 AI Decisions: {ai_decisions_made}")
        print(f"   ✅ AI Success Rate: {ai_contribution_rate:.1%}")
        print(f"   🎯 Decision Diversity: {ai_decision_diversity} types")
        print(f"   ✨ Custom Prompts Created: {custom_prompts_created}")
        print(f"   🎓 New Strategies Learned: {learned_count}")
        print(f"   🧠 Learning Events: {self.real_time_learning_events}")
        print(f"   🎯 Targets: Min {'✅' if session.reached_minimum_threshold else '❌'} | Target {'✅' if session.reached_target_threshold else '❌'} | Ultra {'🏆' if session.reached_ultra_threshold else '❌'}")
        
        # Ultra analysis
        strategies_used = [a.strategy_name for a in attempts if not a.is_ai_generated]
        custom_attempts = [a for a in attempts if a.is_ai_generated]
        
        print(f"   🔧 Strategies Used: {', '.join(set(strategies_used)) if strategies_used else 'None'}")
        print(f"   ✨ Custom Attempts: {len(custom_attempts)}")
        
        if custom_attempts:
            best_custom = max(custom_attempts, key=lambda a: a.validation_score)
            print(f"   🌟 Best Custom: {best_custom.validation_score:.3f}")
        
        if session.reached_ultra_threshold:
            print(f"   🏆 ULTRA ACHIEVEMENT UNLOCKED!")
        elif ultra_targeting_score >= 0.9:
            print(f"   🟡 SO CLOSE TO ULTRA! ({ultra_targeting_score:.1%})")
        elif custom_prompts_created > 3:
            print(f"   🟢 GOOD CUSTOM USAGE: {custom_prompts_created} custom prompts")
        
        return session

def main():
    """Test the ultra-focused v6.6 optimizer"""
    
    test_prompts = [
        "hexagonal prism steel structure",
        "elegant silk fabric draping"
    ]
    
    print("🚀 ADAPTIVE LEARNING OPTIMIZER v6.6 - ULTRA ACHIEVEMENT FOCUS")
    print("=" * 80)
    print("🏆 MISSION: Achieve 0.96+ scores through aggressive custom prompt usage")
    print("⚡ Features: Forced custom prompts, ultra-targeting, premium templates")
    print("🎯 Strategy: Custom prompts are prioritized for breakthrough performance")
    print("=" * 80)
    
    optimizer = AdaptiveLearningOptimizerV6_6_UltraPush(
        max_attempts=8,
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
    
    # Ultra achievement analysis
    print(f"\n🏆 ULTRA ACHIEVEMENT ANALYSIS v6.6")
    print("=" * 80)
    
    total_sessions = len(all_sessions)
    avg_ai_success = statistics.mean([s.ai_contribution_rate for s in all_sessions]) if all_sessions else 0.0
    avg_ultra_progress = statistics.mean([s.ultra_targeting_score for s in all_sessions]) if all_sessions else 0.0
    reached_target = sum(1 for s in all_sessions if s.reached_target_threshold)
    reached_ultra = sum(1 for s in all_sessions if s.reached_ultra_threshold)
    total_custom_prompts = sum(s.custom_prompts_created for s in all_sessions)
    total_learning_events = sum(s.learning_events_during_session for s in all_sessions)
    avg_custom_per_session = total_custom_prompts / total_sessions if total_sessions > 0 else 0
    
    print(f"📊 Ultra Results:")
    print(f"   Total Sessions: {total_sessions}")
    print(f"   Average AI Success Rate: {avg_ai_success:.1%}")
    print(f"   Average Ultra Progress: {avg_ultra_progress:.1%}")
    print(f"   Reached Target: {reached_target}/{total_sessions}")
    print(f"   🏆 ULTRA ACHIEVEMENTS: {reached_ultra}/{total_sessions}")
    print(f"   ✨ Total Custom Prompts: {total_custom_prompts}")
    print(f"   📊 Avg Custom per Session: {avg_custom_per_session:.1f}")
    print(f"   🧠 Total Learning Events: {total_learning_events}")
    
    if reached_ultra > 0:
        print(f"\n🎉 ULTRA SUCCESS! {reached_ultra} ultra achievement{'s' if reached_ultra != 1 else ''}!")
        print(f"🏆 v6.6 ULTRA PUSH: MISSION ACCOMPLISHED!")
    elif avg_ultra_progress >= 0.95:
        print(f"\n🟡 ULTRA PROXIMITY: {avg_ultra_progress:.1%} - Almost there!")
    elif avg_custom_per_session >= 3:
        print(f"\n🟢 CUSTOM MASTERY: Excellent custom prompt usage!")
    elif avg_ai_success >= 0.8:
        print(f"\n🔵 STRONG AI: High success rate achieved!")
    else:
        print(f"\n🟣 LEARNING MODE: System improving with each session!")

if __name__ == "__main__":
    main() 