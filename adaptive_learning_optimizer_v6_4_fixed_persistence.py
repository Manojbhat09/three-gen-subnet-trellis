#!/usr/bin/env python3
"""
Adaptive Learning Optimizer v6.4 - Fixed Persistence
Purpose: Fixed version of v6.4 with proper database persistence and learning system

Key Fixes in v6.4 Fixed:
- Proper session data saving to database
- Learning from successful custom prompts (from v6.3)
- Strategy performance tracking persistence
- Enhanced knowledge base integration
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

# Same dataclasses as v6.4
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
    timestamp: float = 0.0

class AdaptiveLearningOptimizerV6_4_Fixed:
    """v6.4 with fixed persistence and learning integration"""

    def __init__(self, max_attempts: int = 7, min_target: float = 0.6,
                 target: float = 0.9, ultra_target: float = 0.96):
        
        self.ollama_url = "http://localhost:11434"
        self.model_name = "deepseek-r1:1.5b"
        self.db_path = "adaptive_optimizer_v6_4_fixed.db"
        self.max_attempts = max_attempts
        self.min_target = min_target
        self.target = target
        self.ultra_target = ultra_target
        
        # Anti-repetition tracking
        self.used_strategies = []
        self.used_decision_types = []
        self.max_memory = 3
        
        # Enhanced strategy library
        self.strategies = {
            # "raw": "{prompt}",  # Removed as per user's edit
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
            "minimal_clean": "wbgmsst, clean minimal 3D {prompt}, simple geometry, white background"
        }
        
        # Custom prompt templates for different categories
        self.custom_prompt_templates = {
            "technical": [
                "wbgmsst, precision-engineered {prompt}, technical CAD quality, accurate dimensions, industrial grade, white background",
                "wbgmsst, ultra-detailed technical {prompt}, engineering blueprint quality, precise geometry, professional render, white background",
                "wbgmsst, high-precision {prompt}, manufacturing quality, exact specifications, technical illustration, white background"
            ],
            "artistic": [
                "wbgmsst, elegant artistic {prompt}, museum quality sculpture, refined details, perfect lighting, white background",
                "wbgmsst, sophisticated {prompt}, high-end artistic render, graceful form, studio lighting, white background",
                "wbgmsst, masterpiece {prompt}, artistic excellence, refined aesthetics, perfect composition, white background"
            ],
            "textile": [
                "wbgmsst, luxury {prompt}, high-end fabric simulation, realistic textile physics, studio lighting, white background",
                "wbgmsst, premium quality {prompt}, detailed fabric texture, natural draping, soft lighting, white background",
                "wbgmsst, haute couture {prompt}, exquisite textile detail, perfect fabric simulation, white background"
            ],
            "physical": [
                "wbgmsst, premium quality {prompt}, realistic materials, perfect proportions, professional product shot, white background",
                "wbgmsst, high-end {prompt}, commercial grade quality, accurate details, studio lighting, white background",
                "wbgmsst, professional {prompt}, product visualization quality, precise modeling, perfect render, white background"
            ]
        }
        
        # Learning systems (restored from v6.3)
        self.ai_learned_strategies = {}
        self.strategy_performance = {}
        self.custom_prompts_used = 0
        
        self.setup_database()
        self.load_historical_learning()

    def setup_database(self):
        """Enhanced database setup with learning tables (from v6.3)"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Strategy performance tracking
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS strategy_performance (
                strategy_name TEXT,
                category TEXT,
                success_rate REAL,
                avg_improvement REAL,
                usage_count INTEGER,
                last_used REAL,
                PRIMARY KEY (strategy_name, category)
            )
        ''')
        
        # AI learned strategies
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS learned_strategies (
                strategy_name TEXT PRIMARY KEY,
                template TEXT,
                category_affinity TEXT,
                base_success_rate REAL,
                usage_count INTEGER,
                learned_from_prompt TEXT,
                timestamp REAL
            )
        ''')
        
        # Sessions tracking (FIXED: now saves data)
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
                ai_contribution_rate REAL
            )
        ''')
        
        conn.commit()
        conn.close()

    def load_historical_learning(self):
        """Load historical knowledge with enhanced error handling"""
        print("🧠 Loading historical knowledge from database...")
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            # Load AI-learned strategies
            cursor.execute("SELECT strategy_name, template FROM learned_strategies WHERE base_success_rate > 0.5")
            for row in cursor.fetchall():
                name, template = row
                self.ai_learned_strategies[name] = template
                print(f"   📚 Loaded learned strategy: {name}")
            
            # Load strategy performance
            cursor.execute("SELECT strategy_name, category, success_rate, avg_improvement, usage_count FROM strategy_performance WHERE usage_count > 1")
            for row in cursor.fetchall():
                name, cat, rate, imp, count = row
                if name not in self.strategy_performance:
                    self.strategy_performance[name] = {}
                self.strategy_performance[name][cat] = {
                    "success_rate": rate,
                    "avg_improvement": imp,
                    "usage_count": count
                }
            
            print(f"✅ Knowledge loaded: {len(self.ai_learned_strategies)} learned strategies, {len(self.strategy_performance)} performance records")
            
        except sqlite3.OperationalError as e:
            print(f"   📝 Starting fresh knowledge base: {e}")
        finally:
            conn.close()

    def query_ai_simple(self, user_message: str, timeout: int = 45) -> str:
        """Simplified AI query (same as v6.4)"""
        data = {
            "model": self.model_name,
            "messages": [{"role": "user", "content": user_message}],
            "stream": False,
            "options": {
                "temperature": 0.9,
                "top_p": 0.95,
                "num_predict": 250,
                "stop": ["<think>", "</think>"]
            }
        }
        
        try:
            response = requests.post(f"{self.ollama_url}/api/chat", json=data, timeout=timeout)
            response.raise_for_status()
            content = response.json()["message"]["content"]
            
            # Clean response
            content = content.replace("<think>", "").replace("</think>", "")
            content = re.sub(r'\*\*([^*]+)\*\*', r'\1', content)
            
            return content.strip()
        except Exception as e:
            return f"ERROR: {str(e)}"

    def query_ai_for_learning(self, messages: List[Dict[str, str]]) -> str:
        """Query AI for learning tasks (from v6.3)"""
        data = {
            "model": self.model_name,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": 0.7,
                "top_p": 0.9,
                "num_predict": 150
            }
        }
        
        try:
            response = requests.post(f"{self.ollama_url}/api/chat", json=data, timeout=60)
            response.raise_for_status()
            return response.json()["message"]["content"]
        except Exception as e:
            return f"ERROR: {str(e)}"

    def run_validation(self, prompt: str) -> Tuple[float, float]:
        """Run validation (same as before)"""
        try:
            cmd = [sys.executable, "subnet_accurate_validator.py", prompt]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
            
            if result.returncode != 0:
                return 0.0, 0.0
            
            with open("subnet_validation_results.json", 'r') as f:
                data = json.load(f)
                return data.get("validation_engine_score", 0.0), data.get("demo_fidelity_score", 0.0)
                
        except Exception as e:
            print(f"   ❌ Validation error: {e}")
            return 0.0, 0.0

    def categorize_prompt(self, prompt: str) -> str:
        """Enhanced categorization (same as v6.4)"""
        prompt_lower = prompt.lower()
        if any(word in prompt_lower for word in ["steel", "metal", "geometric", "prism", "cylinder", "technical"]):
            return "technical"
        elif any(word in prompt_lower for word in ["elegant", "artistic", "ornate", "beautiful"]):
            return "artistic"
        elif any(word in prompt_lower for word in ["fabric", "silk", "textile", "clothing"]):
            return "textile"
        else:
            return "physical"

    def get_learned_knowledge_summary(self, category: str) -> str:
        """Get summary of learned knowledge for AI decisions (from v6.3)"""
        knowledge_summary = []
        
        # Historical strategy performance
        if category in [perf.get(category, {}) for perf in self.strategy_performance.values() if perf]:
            successful_strategies = []
            for strategy, cats in self.strategy_performance.items():
                if category in cats:
                    perf = cats[category]
                    if perf['success_rate'] > 0.5 and perf['usage_count'] >= 2:
                        successful_strategies.append(f"{strategy} ({perf['success_rate']:.1%} success)")
            
            if successful_strategies:
                knowledge_summary.append(f"Successful strategies for {category}: {', '.join(successful_strategies[:3])}")
        
        # AI-learned strategies
        if self.ai_learned_strategies:
            knowledge_summary.append(f"AI-learned strategies available: {len(self.ai_learned_strategies)}")
        
        return "; ".join(knowledge_summary) if knowledge_summary else "Limited historical knowledge"

    # Use the same AI decision making methods from v6.4...
    # (keeping the same methods: select_persona_and_strategies, make_ai_decision_directive, 
    #  make_forced_custom_prompt, parse_clear_response, create_fallback_decision, execute_ai_decision)
    
    def select_persona_and_strategies(self, attempt_num: int, baseline_score: float, attempts: List[OptimizationAttempt]) -> Tuple[str, List[str]]:
        """Enhanced persona selection with custom prompt bias (same as v6.4)"""
        
        # Encourage custom prompts more often
        if baseline_score < 0.5:
            persona = "Creative Prompter"
        elif attempt_num > 2 and all(a.score_improvement <= 0.01 for a in attempts[-2:]):
            persona = "Custom Prompt Specialist"
        elif baseline_score >= self.target:
            persona = "Ultra Optimizer"
        else:
            persona = "Strategic Optimizer"
        
        # Get strategies with anti-repetition (including learned strategies)
        all_strategies = list(self.strategies.keys()) + list(self.ai_learned_strategies.keys())
        recently_used = [a.strategy_name for a in attempts[-self.max_memory:] if a.strategy_name != "ai_custom_prompt"]
        available_strategies = [s for s in all_strategies if s not in recently_used]
        
        if len(available_strategies) < 5:
            available_strategies.extend([s for s in all_strategies if s not in available_strategies][:3])
        
        return persona, available_strategies[:8]

    def make_ai_decision_directive(self, prompt: str, category: str, baseline_score: float, 
                                  attempt_num: int, attempts: List[OptimizationAttempt]) -> AIDecision:
        """Enhanced directive AI decision making with learned knowledge integration"""
        
        persona, available_strategies = self.select_persona_and_strategies(attempt_num, baseline_score, attempts)
        
        # Build context with learned knowledge
        recent_results = []
        for attempt in attempts[-3:]:
            custom_indicator = " (CUSTOM)" if attempt.is_ai_generated else ""
            recent_results.append(f"Attempt {attempt.attempt_number}: {attempt.strategy_name}{custom_indicator} -> {attempt.validation_score:.3f}")
        
        custom_prompts_used = sum(1 for a in attempts if a.is_ai_generated)
        
        # Get learned knowledge summary
        learned_knowledge = self.get_learned_knowledge_summary(category)
        
        # Enhanced encouragement based on learned knowledge
        custom_encouragement = ""
        if baseline_score < self.min_target:
            custom_encouragement = "\n🔥 STRONG RECOMMENDATION: Strategies aren't working - try WRITE_CUSTOM to break through!"
        elif baseline_score < self.target and custom_prompts_used == 0:
            custom_encouragement = "\n💡 SUGGESTION: You haven't tried a custom prompt yet - this could be the breakthrough!"
        elif all(a.score_improvement <= 0 for a in attempts[-2:]) and len(attempts) >= 2:
            custom_encouragement = "\n🚨 ALERT: Recent strategies are failing - WRITE_CUSTOM might save this session!"
        
        user_message = f"""You are a {persona} optimizing 3D model generation.

SITUATION:
- Prompt: "{prompt}" (Category: {category})
- Baseline: {baseline_score:.3f}
- Attempt: {attempt_num}/{self.max_attempts-1} (one final push remaining)
- Targets: Min {self.min_target} | Excellent {self.target} | Ultra {self.ultra_target}

RECENT RESULTS:
{chr(10).join(recent_results) if recent_results else "None yet"}

LEARNED KNOWLEDGE:
{learned_knowledge}

CUSTOM PROMPTS USED SO FAR: {custom_prompts_used}

AVAILABLE STRATEGIES:
{', '.join(available_strategies)}

{custom_encouragement}

TASK: Choose your optimization approach. Remember: custom prompts can achieve breakthroughs!

OPTIONS:
A) WRITE_CUSTOM: Create a completely new optimized prompt (RECOMMENDED for breakthroughs!)
B) USE_STRATEGY: Pick one strategy from the available list
C) EARLY_STOP: Stop if situation is hopeless or already optimal

CUSTOM PROMPT EXAMPLES for {category}:
{self.custom_prompt_templates.get(category, self.custom_prompt_templates['physical'])[0]}

RESPOND FORMAT:
CHOICE: [A/B/C]
STRATEGY_OR_PROMPT: [strategy name OR complete custom prompt OR stop reason]
REASONING: [brief explanation of why this will work]
CONFIDENCE: [0.1-1.0]

Your response:"""

        print(f"🤖 AI Persona: {persona}")
        print(f"   📋 Available Strategies: {len(available_strategies)}")
        print(f"   ✨ Custom Prompts Used: {custom_prompts_used}")
        print(f"   🧠 Learned Knowledge: {len(self.ai_learned_strategies)} learned strategies")
        
        ai_response = self.query_ai_simple(user_message)
        
        if "ERROR:" in ai_response:
            return self.create_fallback_decision(attempt_num, persona, ai_response, available_strategies)
        
        print(f"   🤖 AI Response: {ai_response[:100]}...")
        
        return self.parse_clear_response(ai_response, attempt_num, persona, available_strategies)

    def make_forced_custom_prompt(self, prompt: str, category: str, baseline_score: float, attempts: List[OptimizationAttempt]) -> AIDecision:
        """FORCED custom prompt with learned knowledge (enhanced from v6.4)"""
        
        best_score = max(a.validation_score for a in attempts) if attempts else baseline_score
        best_attempt = max(attempts, key=lambda a: a.validation_score) if attempts else None
        
        failed_strategies = [a.strategy_name for a in attempts if a.score_improvement < 0]
        successful_strategies = [a.strategy_name for a in attempts if a.score_improvement > 0]
        
        # Include learned knowledge
        learned_knowledge = self.get_learned_knowledge_summary(category)
        
        user_message = f"""You are the ULTIMATE CUSTOM PROMPT MASTER for 3D optimization.

THIS IS THE FINAL ATTEMPT - YOU MUST CREATE A CUSTOM PROMPT!

MISSION: Create the most optimized custom prompt possible to reach ULTRA target {self.ultra_target}!

ANALYSIS:
- Original: "{prompt}" (Category: {category})
- Baseline: {baseline_score:.3f}
- Current Best: {best_score:.3f}
- ULTRA TARGET: {self.ultra_target} (YOU MUST AIM FOR THIS!)

LEARNED KNOWLEDGE:
{learned_knowledge}

WHAT WORKED:
{', '.join(successful_strategies) if successful_strategies else "Nothing significant"}

WHAT FAILED:
{', '.join(failed_strategies) if failed_strategies else "Nothing major"}

BEST ATTEMPT SO FAR:
{f"{best_attempt.strategy_name} -> {best_attempt.validation_score:.3f}" if best_attempt else "Baseline"}

CUSTOM PROMPT TEMPLATES for {category}:
1. {self.custom_prompt_templates.get(category, self.custom_prompt_templates['physical'])[0]}
2. {self.custom_prompt_templates.get(category, self.custom_prompt_templates['physical'])[1] if len(self.custom_prompt_templates.get(category, self.custom_prompt_templates['physical'])) > 1 else "Template 2 unavailable"}

TASK: Write the ULTIMATE custom prompt that combines:
- The core concept of "{prompt}"
- High-quality descriptors (ultra-detailed, photorealistic, etc.)
- Professional rendering terms
- Category-specific enhancements for {category}
- Technical quality indicators
- Elements from successful strategies if any

RESPOND WITH ONLY THE CUSTOM PROMPT - NO EXPLANATION NEEDED:
wbgmsst, [your ultimate optimized prompt here]"""

        print(f"🚀 FORCED CUSTOM PROMPT GENERATION")
        print(f"   🎯 Target: {self.ultra_target}")
        print(f"   📊 Best so far: {best_score:.3f}")
        print(f"   🧠 Using learned knowledge: {learned_knowledge}")
        
        custom_prompt_response = self.query_ai_simple(user_message)
        
        # Clean and validate the custom prompt
        if "ERROR:" in custom_prompt_response:
            templates = self.custom_prompt_templates.get(category, self.custom_prompt_templates['physical'])
            custom_prompt = templates[0].format(prompt=prompt)
        else:
            custom_prompt = custom_prompt_response.strip()
            
            if not custom_prompt.lower().startswith("wbgmsst"):
                custom_prompt = f"wbgmsst, {custom_prompt}"
            
            if len(custom_prompt) < 20:
                templates = self.custom_prompt_templates.get(category, self.custom_prompt_templates['physical'])
                custom_prompt = templates[0].format(prompt=prompt)
        
        print(f"   ✨ Generated Custom Prompt: '{custom_prompt[:80]}...'")
        
        return AIDecision(
            attempt_number=self.max_attempts,
            persona_used="Ultimate Custom Prompt Master",
            decision_type="WRITE_CUSTOM",
            content=custom_prompt,
            reasoning="FORCED final custom prompt using all learned knowledge",
            confidence=0.8,
            expected_improvement=0.15,
            conversation_turn=self.max_attempts,
            based_on_summary=False,
            timestamp=time.time()
        )

    def parse_clear_response(self, response: str, attempt_num: int, persona: str, available_strategies: List[str]) -> AIDecision:
        """Enhanced parsing with learned strategy support (same logic as v6.4 but includes learned strategies)"""
        
        decision_type = "USE_STRATEGY"
        content = available_strategies[0] if available_strategies else "enhanced_clarity"
        reasoning = response[:100]
        confidence = 0.5
        
        try:
            # Extract choice
            choice_match = re.search(r'CHOICE:\s*([ABC])', response, re.IGNORECASE)
            if choice_match:
                choice = choice_match.group(1).upper()
                if choice == "A":
                    decision_type = "WRITE_CUSTOM"
                elif choice == "B":
                    decision_type = "USE_STRATEGY"
                elif choice == "C":
                    decision_type = "EARLY_STOP"
            
            # Extract strategy or prompt
            strategy_match = re.search(r'STRATEGY_OR_PROMPT:\s*(.+?)(?:\n|$)', response, re.IGNORECASE)
            if strategy_match:
                content = strategy_match.group(1).strip()
            
            # Extract reasoning
            reason_match = re.search(r'REASONING:\s*(.+?)(?=CONFIDENCE:|$)', response, re.DOTALL | re.IGNORECASE)
            if reason_match:
                reasoning = reason_match.group(1).strip()
            
            # Extract confidence
            conf_match = re.search(r'CONFIDENCE:\s*([0-9.]+)', response, re.IGNORECASE)
            if conf_match:
                confidence = min(1.0, max(0.1, float(conf_match.group(1))))
            
            # Enhanced validation for custom prompts
            if decision_type == "WRITE_CUSTOM":
                if len(content) < 15 or not any(word in content.lower() for word in ["3d", "model", "background", "wbgmsst"]):
                    if "technical" in reasoning.lower():
                        content = f"wbgmsst, precision-engineered {content}, technical quality, white background"
                    elif "artistic" in reasoning.lower():
                        content = f"wbgmsst, artistic {content}, refined details, white background"
                    else:
                        content = f"wbgmsst, detailed 3D {content}, high quality render, white background"
            
            elif decision_type == "USE_STRATEGY":
                # Check both base strategies and learned strategies
                all_strategies = {**self.strategies, **self.ai_learned_strategies}
                if content not in all_strategies:
                    for strategy in available_strategies:
                        if strategy in content.lower() or content.lower() in strategy:
                            content = strategy
                            break
                    else:
                        content = available_strategies[0] if available_strategies else "enhanced_clarity"
            
        except Exception as e:
            print(f"   ⚠️ Parsing error: {e}, using semantic parsing")
            
            response_lower = response.lower()
            
            if any(word in response_lower for word in ["custom", "write", "create", "wbgmsst"]):
                decision_type = "WRITE_CUSTOM"
                lines = response.split('\n')
                for line in lines:
                    if any(word in line.lower() for word in ["wbgmsst", "3d model"]):
                        content = line.strip()
                        break
                if not any(word in content.lower() for word in ["wbgmsst", "3d"]):
                    content = f"wbgmsst, optimized 3D {reasoning.split()[0] if reasoning else 'model'}, high quality, white background"
            
            else:
                for strategy in available_strategies:
                    if strategy in response_lower:
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
            expected_improvement=confidence * 0.2,
            conversation_turn=attempt_num,
            based_on_summary=False,
            timestamp=time.time()
        )

    def create_fallback_decision(self, attempt_num: int, persona: str, error_msg: str, available_strategies: List[str]) -> AIDecision:
        """Enhanced fallback with learned strategy support"""
        
        if attempt_num > 3:
            return AIDecision(
                attempt_number=attempt_num,
                persona_used=persona,
                decision_type="WRITE_CUSTOM",
                content=f"wbgmsst, enhanced 3D model, high quality render, white background",
                reasoning=f"Fallback custom prompt: {error_msg[:50]}",
                confidence=0.4,
                expected_improvement=0.1,
                conversation_turn=attempt_num,
                based_on_summary=False,
                timestamp=time.time()
            )
        else:
            strategy = available_strategies[attempt_num % len(available_strategies)] if available_strategies else "enhanced_clarity"
            return AIDecision(
                attempt_number=attempt_num,
                persona_used=persona,
                decision_type="USE_STRATEGY",
                content=strategy,
                reasoning=f"Fallback strategy: {error_msg[:50]}",
                confidence=0.3,
                expected_improvement=0.05,
                conversation_turn=attempt_num,
                based_on_summary=False,
                timestamp=time.time()
            )

    def execute_ai_decision(self, decision: AIDecision, prompt: str) -> Tuple[str, str]:
        """Enhanced execution with learned strategies support"""
        
        if decision.decision_type == "EARLY_STOP":
            return "early_stop", prompt
        
        elif decision.decision_type == "WRITE_CUSTOM":
            custom_prompt = decision.content
            
            if len(custom_prompt) > 15 and any(word in custom_prompt.lower() for word in ["3d", "model", "background"]):
                return "ai_custom_prompt", custom_prompt
            else:
                enhanced = f"wbgmsst, ultra-detailed 3D {prompt}, photorealistic quality, professional rendering, white background"
                return "ai_custom_prompt", enhanced
        
        elif decision.decision_type == "USE_STRATEGY":
            strategy = decision.content
            all_strategies = {**self.strategies, **self.ai_learned_strategies}
            
            if strategy in all_strategies:
                return strategy, all_strategies[strategy].format(prompt=prompt)
            else:
                return "enhanced_clarity", self.strategies["enhanced_clarity"].format(prompt=prompt)
        
        return "enhanced_clarity", self.strategies["enhanced_clarity"].format(prompt=prompt)

    # LEARNING SYSTEM INTEGRATION (from v6.3)
    def _learn_from_session(self, prompt: str, category: str, attempts: List[OptimizationAttempt]) -> int:
        """Learning system from v6.3 - distill successful custom prompts into strategies"""
        print("🧠 Updating long-term knowledge base from session results...")
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        newly_learned_count = 0

        # 1. Update performance statistics for all used strategies
        for attempt in attempts:
            strategy = attempt.strategy_name
            cursor.execute("SELECT success_rate, avg_improvement, usage_count FROM strategy_performance WHERE strategy_name = ? AND category = ?", (strategy, category))
            row = cursor.fetchone()
            
            is_success = attempt.score_improvement > 0
            
            if row:
                old_rate, old_imp, old_count = row
                new_count = old_count + 1
                new_rate = ((old_rate * old_count) + (1 if is_success else 0)) / new_count
                new_imp = ((old_imp * old_count) + attempt.score_improvement) / new_count
                cursor.execute("""
                    UPDATE strategy_performance 
                    SET success_rate = ?, avg_improvement = ?, usage_count = ?, last_used = ?
                    WHERE strategy_name = ? AND category = ?
                """, (new_rate, new_imp, new_count, time.time(), strategy, category))
            else:
                cursor.execute("""
                    INSERT INTO strategy_performance (strategy_name, category, success_rate, avg_improvement, usage_count, last_used)
                    VALUES (?, ?, ?, ?, ?, ?)
                """, (strategy, category, 1.0 if is_success else 0.0, attempt.score_improvement, 1, time.time()))

        # 2. Distill new strategies from highly successful custom prompts
        for attempt in attempts:
            if attempt.is_ai_generated and attempt.meets_ultra_threshold:
                print(f"   🎓 Found ultra-successful custom prompt (score: {attempt.validation_score:.3f}). Learning from it...")
                new_strategy_template = self._distill_new_strategy_from_success(prompt, attempt.optimized_prompt)
                
                if new_strategy_template:
                    newly_learned_count += 1
                    strategy_name = f"ai_learned_{category}_{int(time.time())}"
                    print(f"   ✅ New Strategy Learned: '{strategy_name}'")
                    print(f"        Template: {new_strategy_template}")
                    
                    cursor.execute("""
                        INSERT OR IGNORE INTO learned_strategies (strategy_name, template, category_affinity, base_success_rate, usage_count, learned_from_prompt, timestamp)
                        VALUES (?, ?, ?, ?, ?, ?, ?)
                    """, (strategy_name, new_strategy_template, category, 1.0, 1, prompt, time.time()))
                    
                    # Add to current session's knowledge
                    self.ai_learned_strategies[strategy_name] = new_strategy_template

        conn.commit()
        conn.close()
        print(f"✅ Knowledge base updated. {newly_learned_count} new strategies were learned.")
        return newly_learned_count

    def _distill_new_strategy_from_success(self, original_prompt: str, successful_prompt: str) -> Optional[str]:
        """Extract learnable patterns from successful custom prompts"""
        system_prompt = """You are a Knowledge Engineer AI. Analyze a highly successful custom prompt and create a reusable template.

Your task: Extract the core optimization pattern from this successful prompt and create a generic template that can be used for similar prompts.

The template MUST include the `{prompt}` placeholder where the original subject would go.

Focus on identifying what made this prompt successful:
- Specific descriptive terms
- Quality indicators
- Technical specifications
- Rendering instructions
- Material descriptions"""
        
        user_prompt = f"""
        Analyze this successful optimization:
        - Original: "{original_prompt}"
        - Successful Custom Prompt: "{successful_prompt}"

        Create a generic template. For example:
        - If successful prompt was "wbgmsst, ultra-realistic steel car, ray-traced lighting, white background"
        - Template would be "wbgmsst, ultra-realistic {prompt}, ray-traced lighting, white background"

        Respond with ONLY the template string. If no clear pattern, respond "NO_PATTERN".
        """
        
        template = self.query_ai_for_learning([
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ])

        if "NO_PATTERN" in template or "{prompt}" not in template or "ERROR:" in template:
            return None
        return template.strip()

    def save_session_to_database(self, session: OptimizationSession) -> int:
        """FIXED: Actually save session data to database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            cursor.execute("""
                INSERT INTO sessions (
                    timestamp, original_prompt, category, baseline_score, best_score, 
                    session_improvement, total_attempts, custom_prompts_created, 
                    new_strategies_learned, reached_minimum, reached_target, 
                    reached_ultra, ai_contribution_rate
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                session.timestamp, session.original_prompt, session.prompt_category,
                session.baseline_score, session.best_attempt.validation_score if session.best_attempt else session.baseline_score,
                session.session_improvement, len(session.attempts), session.custom_prompts_created,
                session.new_strategies_learned, session.reached_minimum_threshold,
                session.reached_target_threshold, session.reached_ultra_threshold,
                session.ai_contribution_rate
            ))
            
            session_id = cursor.lastrowid
            conn.commit()
            print(f"   💾 Session saved to database with ID: {session_id}")
            return session_id
            
        except Exception as e:
            print(f"   ❌ Error saving session: {e}")
            return -1
        finally:
            conn.close()

    def optimize_prompt(self, prompt: str) -> OptimizationSession:
        """Enhanced optimization with fixed persistence and learning"""
        
        print(f"\n🚀 ADAPTIVE OPTIMIZER v6.4 FIXED: '{prompt}'")
        print("=" * 70)
        print(f"🎯 Targets: Min {self.min_target} | Target {self.target} | Ultra {self.ultra_target}")
        print("🧠 Features: FIXED persistence, learning integration, forced custom push")
        
        # Reset tracking
        self.used_strategies = []
        self.used_decision_types = []
        self.custom_prompts_used = 0
        
        # Setup
        category = self.categorize_prompt(prompt)
        print(f"📋 Category: {category}")
        
        baseline_score, baseline_fidelity = self.run_validation(prompt)
        print(f"📊 Baseline: {baseline_score:.3f}")
        
        if baseline_score >= self.ultra_target:
            print(f"🏆 BASELINE ALREADY ULTRA-OPTIMAL!")
            return self.create_session_summary(prompt, category, baseline_score, [])
        
        # Tracking
        attempts = []
        best_score = baseline_score
        best_attempt = None
        
        # Main optimization loop (6 regular attempts)
        for i in range(1, self.max_attempts):
            print(f"\n🔄 ATTEMPT {i}/{self.max_attempts-1} (Regular)")
            
            ai_decision = self.make_ai_decision_directive(prompt, category, baseline_score, i, attempts)
            
            print(f"   🧠 Decision: {ai_decision.decision_type}")
            print(f"   💭 Content: {ai_decision.content[:60]}{'...' if len(ai_decision.content) > 60 else ''}")
            print(f"   🎯 Confidence: {ai_decision.confidence:.2f}")
            
            strategy_name, optimized_prompt = self.execute_ai_decision(ai_decision, prompt)
            
            if strategy_name == "early_stop":
                print(f"   🛑 AI Early Stop")
                break
            
            is_ai_generated = strategy_name == "ai_custom_prompt"
            if is_ai_generated:
                self.custom_prompts_used += 1
            
            print(f"   🔧 Executing: {strategy_name}")
            if is_ai_generated:
                print(f"   ✨ CUSTOM Prompt: '{optimized_prompt[:60]}{'...' if len(optimized_prompt) > 60 else ''}'")
            else:
                print(f"   📋 Strategy Prompt: '{optimized_prompt[:60]}{'...' if len(optimized_prompt) > 60 else ''}'")
            
            val_score, val_fidelity = self.run_validation(optimized_prompt)
            improvement = val_score - baseline_score
            
            print(f"   📊 Result: {val_score:.3f} ({improvement:+.3f})")
            print(f"   🎯 Min {'✅' if val_score >= self.min_target else '❌'} | Target {'✅' if val_score >= self.target else '❌'} | Ultra {'✅' if val_score >= self.ultra_target else '❌'}")
            
            # Update AI decision outcome
            ai_decision.actual_improvement = improvement
            if improvement > 0.01:
                ai_decision.led_to_improvement = True
                print(f"   🤖 AI Success: ✅")
            else:
                print(f"   🤖 AI Success: ❌")
            
            if val_score > best_score:
                best_score = val_score
                ai_decision.contributed_to_best_score = True
                print(f"   🌟 NEW BEST SCORE!")
            
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
                timestamp=time.time()
            )
            attempts.append(attempt)
            
            if ai_decision.contributed_to_best_score:
                best_attempt = attempt
            
            # Early ultra check
            if val_score >= self.ultra_target:
                print(f"   🏆 ULTRA TARGET ACHIEVED EARLY!")
                break
            
            time.sleep(1)
        
        # FORCED FINAL PUSH (same as v6.4)
        if best_score < self.ultra_target and len(attempts) < self.max_attempts:
            print(f"\n🚀🚀 FORCED FINAL PUSH ATTEMPT {self.max_attempts}/{self.max_attempts} 🚀🚀")
            print("⚡ MANDATORY CUSTOM PROMPT - NO OTHER OPTIONS")
            
            final_ai_decision = self.make_forced_custom_prompt(prompt, category, baseline_score, attempts)
            
            print(f"   🧠 Final Decision: FORCED {final_ai_decision.decision_type}")
            print(f"   💭 Final Content: {final_ai_decision.content[:80]}...")
            print(f"   🎯 Final Confidence: {final_ai_decision.confidence:.2f}")
            
            final_strategy_name, final_optimized_prompt = self.execute_ai_decision(final_ai_decision, prompt)
            
            self.custom_prompts_used += 1
            
            print(f"   🔧 Final Executing: {final_strategy_name}")
            print(f"   ✨ FINAL CUSTOM Prompt: '{final_optimized_prompt[:80]}...'")
            
            final_val_score, final_val_fidelity = self.run_validation(final_optimized_prompt)
            final_improvement = final_val_score - baseline_score
            
            print(f"   📊 Final Result: {final_val_score:.3f} ({final_improvement:+.3f})")
            print(f"   🎯 Final: Min {'✅' if final_val_score >= self.min_target else '❌'} | Target {'✅' if final_val_score >= self.target else '❌'} | Ultra {'✅' if final_val_score >= self.ultra_target else '❌'}")
            
            final_ai_decision.actual_improvement = final_improvement
            if final_improvement > 0.01:
                final_ai_decision.led_to_improvement = True
                print(f"   🤖 Final AI Success: ✅")
            else:
                print(f"   🤖 Final AI Success: ❌")
            
            if final_val_score > best_score:
                best_score = final_val_score
                final_ai_decision.contributed_to_best_score = True
                print(f"   🌟 FORCED FINAL PUSH NEW BEST SCORE!")
            
            final_attempt = OptimizationAttempt(
                attempt_number=self.max_attempts,
                strategy_name=final_strategy_name,
                optimized_prompt=final_optimized_prompt,
                validation_score=final_val_score,
                demo_fidelity_score=final_val_fidelity,
                score_improvement=final_improvement,
                meets_minimum_threshold=final_val_score >= self.min_target,
                meets_target_threshold=final_val_score >= self.target,
                meets_ultra_threshold=final_val_score >= self.ultra_target,
                ai_decision=final_ai_decision,
                learning_moment=f"FORCED FINAL CUSTOM PUSH: {'SUCCESS' if final_improvement > 0 else 'FAILED'}",
                is_ai_generated=True,
                timestamp=time.time()
            )
            attempts.append(final_attempt)
            
            if final_ai_decision.contributed_to_best_score:
                best_attempt = final_attempt
            
            if final_val_score >= self.ultra_target:
                print(f"   🏆 FORCED FINAL PUSH ACHIEVED ULTRA TARGET!")
        
        # FIXED: Learning and persistence
        learned_count = self._learn_from_session(prompt, category, attempts)
        session = self.create_session_summary(prompt, category, baseline_score, attempts, best_attempt, learned_count)
        session_id = self.save_session_to_database(session)
        
        return session

    def create_session_summary(self, prompt: str, category: str, baseline_score: float, 
                              attempts: List[OptimizationAttempt], best_attempt: OptimizationAttempt = None, 
                              learned_count: int = 0) -> OptimizationSession:
        """Enhanced session summary with learning tracking"""
        
        if not attempts:
            best_score = baseline_score
            session_improvement = 0.0
            ai_decisions_made = 0
            ai_decisions_that_improved = 0
            ai_decision_diversity = 0
        else:
            best_score = max(a.validation_score for a in attempts)
            session_improvement = best_score - baseline_score
            ai_decisions_made = len(attempts)
            ai_decisions_that_improved = sum(1 for a in attempts if a.ai_decision.led_to_improvement)
            ai_decision_diversity = len(set(a.ai_decision.decision_type for a in attempts))
        
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
            timestamp=time.time()
        )
        
        # Enhanced summary
        print(f"\n📊 SESSION SUMMARY v6.4 FIXED WITH LEARNING:")
        print(f"   📈 Best Score: {best_score:.3f} (Baseline: {baseline_score:.3f})")
        print(f"   📊 Session Improvement: {session_improvement:+.3f}")
        print(f"   🤖 AI Decisions: {ai_decisions_made}")
        print(f"   ✅ AI Success Rate: {ai_contribution_rate:.1%}")
        print(f"   🎯 Decision Diversity: {ai_decision_diversity} types")
        print(f"   ✨ Custom Prompts Created: {custom_prompts_created}")
        print(f"   🎓 New Strategies Learned: {learned_count}")
        print(f"   🎯 Targets: Min {'✅' if session.reached_minimum_threshold else '❌'} | Target {'✅' if session.reached_target_threshold else '❌'} | Ultra {'✅' if session.reached_ultra_threshold else '❌'}")
        
        # Show strategies used
        strategies_used = [a.strategy_name for a in attempts if not a.is_ai_generated]
        custom_attempts = [a for a in attempts if a.is_ai_generated]
        
        print(f"   🔧 Strategies Used: {', '.join(set(strategies_used)) if strategies_used else 'None'}")
        print(f"   ✨ Custom Attempts: {len(custom_attempts)}")
        
        # Show final push info
        final_attempts = [a for a in attempts if a.attempt_number == self.max_attempts]
        if final_attempts:
            final_attempt = final_attempts[0]
            print(f"   🚀 Forced Final Push: {final_attempt.strategy_name} -> {final_attempt.validation_score:.3f} ({'✅ SUCCESS' if final_attempt.ai_decision.led_to_improvement else '❌ FAILED'})")
        
        return session

def main():
    """Test v6.4 Fixed with persistence and learning"""
    
    test_prompts = [
        "hexagonal prism steel structure"
    ]
    
    print("🚀 ADAPTIVE LEARNING OPTIMIZER v6.4 FIXED - PERSISTENCE & LEARNING")
    print("=" * 80)
    print("🔧 Features: FIXED database persistence, learning integration")
    print("⚡ Final Attempt: MANDATORY custom prompt using learned knowledge")
    print("🎓 Learning: Successful custom prompts become new strategies")
    print("=" * 80)
    
    optimizer = AdaptiveLearningOptimizerV6_4_Fixed(
        max_attempts=7,
        min_target=0.6,
        target=0.9,
        ultra_target=0.96
    )
    
    all_sessions = []
    
    for i, prompt in enumerate(test_prompts, 1):
        print(f"\n{'='*25} [{i}/{len(test_prompts)}] PROMPT {i} {'='*25}")
        session = optimizer.optimize_prompt(prompt)
        all_sessions.append(session)
        time.sleep(2)
    
    print(f"\n🎓 PERSISTENCE & LEARNING TEST COMPLETE")
    print(f"   ✅ Data should now be saved to: {optimizer.db_path}")
    print(f"   🎓 Learning system active - ultra successful custom prompts become strategies")

if __name__ == "__main__":
    main() 