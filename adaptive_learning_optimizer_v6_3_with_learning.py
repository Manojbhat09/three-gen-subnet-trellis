#!/usr/bin/env python3
"""
Adaptive Learning Optimizer v6.3 - With Learning Integration & Final Custom Push
Purpose: Integrates the learning system and adds a final AI custom prompt attempt 
using all learned knowledge to shoot for higher scores.

Key Improvements in v6.3:
- Integrated database learning system from v6.0
- Final custom prompt attempt using learned knowledge
- Strategy performance tracking and distillation
- AI knowledge engineering for new strategy creation
- Enhanced session completion with learning updates
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

# Same dataclasses as v6.2 but with learning enhancements
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
    new_strategies_learned: int  # Added learning tracking
    timestamp: float = 0.0

class AdaptiveLearningOptimizerV6_3:
    """v6.3 with integrated learning system and final custom push"""

    def __init__(self, max_attempts: int = 7, min_target: float = 0.6,  # Increased for final push
                 target: float = 0.9, ultra_target: float = 0.96):
        
        self.ollama_url = "http://localhost:11434"
        self.model_name = "deepseek-r1:1.5b"
        self.db_path = "adaptive_optimizer_v6_3.db"
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
            # "raw": "{prompt}",
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
        
        # Learning systems
        self.ai_learned_strategies = {}
        self.strategy_performance = {}
        
        self.setup_database()
        self.load_historical_learning()

    def setup_database(self):
        """Enhanced database setup with learning tables"""
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
        
        # Sessions tracking
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
                new_strategies_learned INTEGER,
                reached_target BOOLEAN,
                reached_ultra BOOLEAN
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
        """Simplified AI query"""
        data = {
            "model": self.model_name,
            "messages": [{"role": "user", "content": user_message}],
            "stream": False,
            "options": {
                "temperature": 0.8,
                "top_p": 0.9,
                "num_predict": 200,
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
        """Query AI for learning tasks with structured messages"""
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
        """Enhanced categorization"""
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
        """Get summary of learned knowledge for AI decisions"""
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

    def select_persona_and_strategies(self, attempt_num: int, baseline_score: float, attempts: List[OptimizationAttempt], is_final_push: bool = False) -> Tuple[str, List[str]]:
        """Enhanced persona selection with final push logic"""
        
        if is_final_push:
            persona = "Knowledge Master"  # Special persona for final attempt
        elif baseline_score < 0.4:
            persona = "Rescue Specialist"
        elif attempt_num > 3 and all(a.score_improvement <= 0.01 for a in attempts[-2:]):
            persona = "Creative Breaker"
        else:
            persona = "Strategic Optimizer"
        
        # Get all available strategies (base + learned)
        all_strategies = list(self.strategies.keys()) + list(self.ai_learned_strategies.keys())
        
        # Anti-repetition: avoid recently used strategies
        recently_used = [a.strategy_name for a in attempts[-self.max_memory:]]
        available_strategies = [s for s in all_strategies if s not in recently_used]
        
        # If too few available, add some back
        if len(available_strategies) < 5:
            available_strategies.extend([s for s in all_strategies if s not in available_strategies][:3])
        
        return persona, available_strategies[:10]  # Limit for clarity

    def make_ai_decision_clear(self, prompt: str, category: str, baseline_score: float, 
                              attempt_num: int, attempts: List[OptimizationAttempt], is_final_push: bool = False) -> AIDecision:
        """Enhanced AI decision making with learning integration"""
        
        persona, available_strategies = self.select_persona_and_strategies(attempt_num, baseline_score, attempts, is_final_push)
        
        # Build enhanced context with learned knowledge
        recent_results = []
        for attempt in attempts[-3:]:
            recent_results.append(f"Attempt {attempt.attempt_number}: {attempt.strategy_name} -> {attempt.validation_score:.3f}")
        
        # Get learned knowledge
        learned_knowledge = self.get_learned_knowledge_summary(category)
        
        # Special final push prompt
        if is_final_push:
            user_message = f"""You are a Knowledge Master making the FINAL OPTIMIZATION ATTEMPT.

MISSION: Use ALL learned knowledge to create a custom prompt that reaches the ULTRA target {self.ultra_target}!

SITUATION:
- Prompt: "{prompt}" (Category: {category})
- Baseline: {baseline_score:.3f}
- Current Best: {max(a.validation_score for a in attempts) if attempts else baseline_score:.3f}
- ULTRA TARGET: {self.ultra_target} (THIS IS YOUR GOAL!)

RECENT ATTEMPTS:
{chr(10).join(recent_results) if recent_results else "None yet"}

LEARNED KNOWLEDGE:
{learned_knowledge}

AVAILABLE STRATEGIES:
{', '.join(available_strategies)}

FINAL PUSH TASK: 
This is your LAST CHANCE to reach ultra-optimal performance. You must:
1. WRITE_CUSTOM: Create a completely optimized custom prompt
2. Use everything you've learned about what works
3. Aim for {self.ultra_target}+ score

RESPOND FORMAT:
CHOICE: A
STRATEGY_OR_PROMPT: [Your ultimate custom prompt using all knowledge]
REASONING: [Why this custom prompt will achieve ultra performance]
CONFIDENCE: [0.1-1.0]

Make this count! This is the final attempt:"""

        else:
            # Regular decision prompt (same as v6.2)
            user_message = f"""You are a {persona} optimizing 3D model generation.

SITUATION:
- Prompt: "{prompt}" (Category: {category})
- Baseline: {baseline_score:.3f}
- Attempt: {attempt_num}/{self.max_attempts}
- Targets: Min {self.min_target} | Excellent {self.target} | Ultra {self.ultra_target}

RECENT RESULTS:
{chr(10).join(recent_results) if recent_results else "None yet"}

LEARNED KNOWLEDGE:
{learned_knowledge}

AVAILABLE STRATEGIES:
{', '.join(available_strategies)}

TASK: Choose your optimization approach.

OPTIONS:
A) WRITE_CUSTOM: Create a completely new optimized prompt
B) USE_STRATEGY: Pick one strategy from the available list
C) EARLY_STOP: Stop if situation is hopeless or already optimal

RESPOND FORMAT:
CHOICE: [A/B/C]
STRATEGY_OR_PROMPT: [strategy name OR custom prompt OR stop reason]
REASONING: [brief explanation]
CONFIDENCE: [0.1-1.0]

Your response:"""

        print(f"🤖 AI Persona: {persona}")
        print(f"   📋 Available Strategies: {len(available_strategies)}")
        if is_final_push:
            print(f"   🎯 FINAL PUSH MODE: Aiming for ULTRA {self.ultra_target}")
        
        ai_response = self.query_ai_simple(user_message)
        
        if "ERROR:" in ai_response:
            return self.create_fallback_decision(attempt_num, persona, ai_response, available_strategies)
        
        print(f"   🤖 AI Response: {ai_response[:100]}...")
        
        # Parse response
        return self.parse_clear_response(ai_response, attempt_num, persona, available_strategies)

    def parse_clear_response(self, response: str, attempt_num: int, persona: str, available_strategies: List[str]) -> AIDecision:
        """Enhanced parsing with learning integration (same as v6.2 but with learning context)"""
        
        # Initialize defaults
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
            
            # Validate strategy choice for USE_STRATEGY
            if decision_type == "USE_STRATEGY":
                all_strategies = {**self.strategies, **self.ai_learned_strategies}
                if content not in all_strategies:
                    # Try partial match
                    for strategy in available_strategies:
                        if strategy in content.lower() or content.lower() in strategy:
                            content = strategy
                            break
                    else:
                        content = available_strategies[0] if available_strategies else "enhanced_clarity"
            
        except Exception as e:
            print(f"   ⚠️ Parsing error: {e}, using semantic parsing")
            
            # Semantic fallback
            response_lower = response.lower()
            
            # Look for custom prompt indicators
            if any(word in response_lower for word in ["custom", "write", "create", "wbgmsst"]):
                decision_type = "WRITE_CUSTOM"
                # Extract custom prompt
                lines = response.split('\n')
                for line in lines:
                    if any(word in line.lower() for word in ["wbgmsst", "3d model", "white background"]):
                        content = line.strip()
                        break
                if not any(word in content.lower() for word in ["wbgmsst", "3d"]):
                    content = f"wbgmsst, optimized {reasoning.split()[0] if reasoning else 'model'}, ultra-high quality 3D, white background"
            
            # Look for strategy names
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
        """Enhanced fallback with learned strategies"""
        
        # Use available strategies intelligently
        all_strategies = available_strategies if available_strategies else list(self.strategies.keys())
        strategy = all_strategies[attempt_num % len(all_strategies)]
        
        return AIDecision(
            attempt_number=attempt_num,
            persona_used=persona,
            decision_type="USE_STRATEGY",
            content=strategy,
            reasoning=f"Fallback: {error_msg[:50]}",
            confidence=0.3,
            expected_improvement=0.05,
            conversation_turn=attempt_num,
            based_on_summary=False,
            timestamp=time.time()
        )

    def execute_ai_decision(self, decision: AIDecision, prompt: str) -> Tuple[str, str]:
        """Enhanced execution with learned strategies"""
        
        if decision.decision_type == "EARLY_STOP":
            return "early_stop", prompt
        
        elif decision.decision_type == "WRITE_CUSTOM":
            custom_prompt = decision.content
            if len(custom_prompt) > 10 and any(word in custom_prompt.lower() for word in ["3d", "model", "background"]):
                return "ai_custom_prompt", custom_prompt
            else:
                # Generate better custom prompt
                custom_prompt = f"wbgmsst, ultra-detailed 3D {prompt}, photorealistic quality, professional rendering, white background"
                return "ai_custom_prompt", custom_prompt
        
        elif decision.decision_type == "USE_STRATEGY":
            strategy = decision.content
            all_strategies = {**self.strategies, **self.ai_learned_strategies}
            
            if strategy in all_strategies:
                return strategy, all_strategies[strategy].format(prompt=prompt)
            else:
                return "enhanced_clarity", self.strategies["enhanced_clarity"].format(prompt=prompt)
        
        return "enhanced_clarity", self.strategies["enhanced_clarity"].format(prompt=prompt)

    # LEARNING SYSTEM INTEGRATION (from user's code)
    def _learn_from_session(self, prompt: str, category: str, attempts: List[OptimizationAttempt]) -> int:
        """
        Updates long-term DB knowledge after a session completes. This is the core of the knowledge bank.
        Returns the number of new strategies learned in this session.
        """
        print("🧠 Updating long-term knowledge base from session results...")
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        newly_learned_count = 0

        # 1. Update performance statistics for all used strategies
        for attempt in attempts:
            strategy = attempt.strategy_name
            # Fetch current stats
            cursor.execute("SELECT success_rate, avg_improvement, usage_count FROM strategy_performance WHERE strategy_name = ? AND category = ?", (strategy, category))
            row = cursor.fetchone()
            
            is_success = attempt.score_improvement > 0
            
            if row:
                # Update existing record
                old_rate, old_imp, old_count = row
                new_count = old_count + 1
                new_rate = ((old_rate * old_count) + (1 if is_success else 0)) / new_count
                new_imp = ((old_imp * old_count) + attempt.score_improvement) / new_count
                cursor.execute("""
                    UPDATE strategy_performance 
                    SET success_rate = ?, avg_improvement = ?, usage_count = ?
                    WHERE strategy_name = ? AND category = ?
                """, (new_rate, new_imp, new_count, strategy, category))
            else:
                # Create new record
                cursor.execute("""
                    INSERT INTO strategy_performance (strategy_name, category, success_rate, avg_improvement, usage_count)
                    VALUES (?, ?, ?, ?, ?)
                """, (strategy, category, 1.0 if is_success else 0.0, attempt.score_improvement, 1))

        # 2. Distill new strategies from highly successful custom prompts
        for attempt in attempts:
            if attempt.is_ai_generated and attempt.meets_ultra_threshold:
                print(f"   - Found a highly successful custom prompt (score: {attempt.validation_score:.3f}). Attempting to distill a new strategy...")
                new_strategy_template = self._distill_new_strategy_from_success(prompt, attempt.optimized_prompt)
                
                if new_strategy_template:
                    newly_learned_count += 1
                    strategy_name = f"ai_learned_{category}_{int(time.time())}"
                    print(f"   - ✅ New Strategy Distilled: '{strategy_name}'")
                    print(f"   -    Template: {new_strategy_template}")
                    # Save to DB
                    cursor.execute("""
                        INSERT OR IGNORE INTO learned_strategies (strategy_name, template, category_affinity, base_success_rate, usage_count, learned_from_prompt)
                        VALUES (?, ?, ?, ?, ?, ?)
                    """, (strategy_name, new_strategy_template, category, 1.0, 1, prompt))
                    # Add to current session's knowledge
                    self.ai_learned_strategies[strategy_name] = new_strategy_template

        conn.commit()
        conn.close()
        print(f"✅ Knowledge base updated. {newly_learned_count} new strategies were learned.")
        return newly_learned_count

    def _distill_new_strategy_from_success(self, original_prompt: str, successful_prompt: str) -> Optional[str]:
        """Uses the AI to turn a specific successful prompt into a reusable template."""
        system_prompt = """
You are a 'Knowledge Engineer' AI. Your purpose is to build a long-term knowledge bank for a 3D prompt optimization system.

You have been given a highly successful example where a creative custom prompt, written by another AI persona, achieved an ultra-high score. Your task is to analyze this success and distill its core principle into a generic, reusable strategy template. This new template will be added to the knowledge bank and used to optimize future, unseen prompts.

The template MUST include the `{prompt}` placeholder where the original subject would go. Capture the *essence* of the successful pattern (e.g., was it about lighting, texture, specific keywords, camera angle?).
"""
        user_prompt = f"""
        Analyze the following:
        - Original Prompt: "{original_prompt}"
        - Successful Custom Prompt: "{successful_prompt}"

        Now, generate a generic template based on the successful prompt. For example, if the successful prompt was "ultra-realistic photo of a red car", a good template would be "ultra-realistic photo of a {prompt}".

        Respond with ONLY the template string. If no clear pattern can be extracted, respond with "NO_PATTERN".
        """
        
        template = self.query_ai_for_learning([
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ])

        if "NO_PATTERN" in template or "{prompt}" not in template:
            return None
        return template.strip()

    def optimize_prompt(self, prompt: str) -> OptimizationSession:
        """Enhanced optimization with learning integration and final push"""
        
        print(f"\n🚀 ADAPTIVE OPTIMIZER v6.3: '{prompt}'")
        print("=" * 70)
        print(f"🎯 Targets: Min {self.min_target} | Target {self.target} | Ultra {self.ultra_target}")
        print("🧠 Features: Learning integration, final custom push")
        
        # Reset tracking
        self.used_strategies = []
        self.used_decision_types = []
        
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
        
        # Main optimization loop (regular attempts)
        for i in range(1, self.max_attempts):  # Leave room for final push
            print(f"\n🔄 ATTEMPT {i}/{self.max_attempts-1} (Regular)")
            
            # Regular AI Decision
            ai_decision = self.make_ai_decision_clear(prompt, category, baseline_score, i, attempts, is_final_push=False)
            
            print(f"   🧠 Decision: {ai_decision.decision_type}")
            print(f"   💭 Content: {ai_decision.content}")
            print(f"   🎯 Confidence: {ai_decision.confidence:.2f}")
            
            # Execute and validate
            strategy_name, optimized_prompt = self.execute_ai_decision(ai_decision, prompt)
            
            if strategy_name == "early_stop":
                print(f"   🛑 AI Early Stop")
                break
            
            self.used_strategies.append(strategy_name)
            self.used_decision_types.append(ai_decision.decision_type)
            
            is_ai_generated = strategy_name == "ai_custom_prompt"
            
            print(f"   🔧 Executing: {strategy_name}")
            print(f"   ✨ Prompt: '{optimized_prompt[:60]}{'...' if len(optimized_prompt) > 60 else ''}'")
            
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
                learning_moment=f"Strategy '{strategy_name}' {'succeeded' if improvement > 0 else 'failed'}",
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
        
        # FINAL PUSH: AI Custom Prompt using all learned knowledge
        if best_score < self.ultra_target and len(attempts) < self.max_attempts:
            print(f"\n🚀 FINAL PUSH ATTEMPT {self.max_attempts}/{self.max_attempts}")
            print("🎯 AI KNOWLEDGE MASTER - FINAL CUSTOM PROMPT FOR ULTRA TARGET")
            
            # Final push decision
            final_ai_decision = self.make_ai_decision_clear(prompt, category, baseline_score, 
                                                          self.max_attempts, attempts, is_final_push=True)
            
            print(f"   🧠 Final Decision: {final_ai_decision.decision_type}")
            print(f"   💭 Final Content: {final_ai_decision.content}")
            print(f"   🎯 Final Confidence: {final_ai_decision.confidence:.2f}")
            
            # Execute final attempt
            final_strategy_name, final_optimized_prompt = self.execute_ai_decision(final_ai_decision, prompt)
            
            if final_strategy_name != "early_stop":
                is_final_ai_generated = final_strategy_name == "ai_custom_prompt"
                
                print(f"   🔧 Final Executing: {final_strategy_name}")
                print(f"   ✨ Final Prompt: '{final_optimized_prompt[:60]}{'...' if len(final_optimized_prompt) > 60 else ''}'")
                
                final_val_score, final_val_fidelity = self.run_validation(final_optimized_prompt)
                final_improvement = final_val_score - baseline_score
                
                print(f"   📊 Final Result: {final_val_score:.3f} ({final_improvement:+.3f})")
                print(f"   🎯 Final: Min {'✅' if final_val_score >= self.min_target else '❌'} | Target {'✅' if final_val_score >= self.target else '❌'} | Ultra {'✅' if final_val_score >= self.ultra_target else '❌'}")
                
                # Update final AI decision outcome
                final_ai_decision.actual_improvement = final_improvement
                if final_improvement > 0.01:
                    final_ai_decision.led_to_improvement = True
                    print(f"   🤖 Final AI Success: ✅")
                else:
                    print(f"   🤖 Final AI Success: ❌")
                
                if final_val_score > best_score:
                    best_score = final_val_score
                    final_ai_decision.contributed_to_best_score = True
                    print(f"   🌟 FINAL PUSH NEW BEST SCORE!")
                
                # Create final attempt
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
                    learning_moment=f"FINAL PUSH: {final_strategy_name} {'succeeded' if final_improvement > 0 else 'failed'}",
                    is_ai_generated=is_final_ai_generated,
                    timestamp=time.time()
                )
                attempts.append(final_attempt)
                
                if final_ai_decision.contributed_to_best_score:
                    best_attempt = final_attempt
                
                if final_val_score >= self.ultra_target:
                    print(f"   🏆 FINAL PUSH ACHIEVED ULTRA TARGET!")
        
        # Post-session learning
        learned_count = self._learn_from_session(prompt, category, attempts)
        
        return self.create_session_summary(prompt, category, baseline_score, attempts, best_attempt, learned_count)

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
            timestamp=time.time()
        )
        
        # Enhanced summary
        print(f"\n📊 SESSION SUMMARY v6.3 WITH LEARNING:")
        print(f"   📈 Best Score: {best_score:.3f} (Baseline: {baseline_score:.3f})")
        print(f"   📊 Session Improvement: {session_improvement:+.3f}")
        print(f"   🤖 AI Decisions: {ai_decisions_made}")
        print(f"   ✅ AI Success Rate: {ai_contribution_rate:.1%}")
        print(f"   🎯 Decision Diversity: {ai_decision_diversity} types")
        print(f"   📚 New Strategies Learned: {learned_count}")
        print(f"   🎯 Targets: Min {'✅' if session.reached_minimum_threshold else '❌'} | Target {'✅' if session.reached_target_threshold else '❌'} | Ultra {'✅' if session.reached_ultra_threshold else '❌'}")
        
        # Show strategies used
        strategies_used = [a.strategy_name for a in attempts]
        unique_strategies = list(set(strategies_used))
        print(f"   🔧 Strategies Used: {', '.join(unique_strategies)}")
        
        # Show final push info
        final_attempts = [a for a in attempts if a.attempt_number == self.max_attempts]
        if final_attempts:
            final_attempt = final_attempts[0]
            print(f"   🚀 Final Push: {final_attempt.strategy_name} -> {final_attempt.validation_score:.3f}")
        
        return session

def main():
    """Test v6.3 with learning integration and final push"""
    
    test_prompts = [
        "hexagonal prism steel structure",
        "elegant silk fabric draping",
        "transparent glass sphere reflection"
    ]
    
    print("🚀 ADAPTIVE LEARNING OPTIMIZER v6.3 - WITH LEARNING & FINAL PUSH")
    print("=" * 80)
    print("🔧 Features: Integrated learning system, final AI custom push")
    print("🧠 AI: Knowledge Master final attempt using all learned knowledge")
    print("📊 Tracking: Strategy learning, performance tracking, distillation")
    print("=" * 80)
    
    optimizer = AdaptiveLearningOptimizerV6_3(
        max_attempts=7,  # 6 regular + 1 final push
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
    
    # Final analysis
    print(f"\n🎓 FINAL ANALYSIS - v6.3 WITH LEARNING")
    print("=" * 80)
    
    total_sessions = len(all_sessions)
    avg_ai_success = statistics.mean([s.ai_contribution_rate for s in all_sessions]) if all_sessions else 0.0
    avg_diversity = statistics.mean([s.ai_decision_diversity for s in all_sessions]) if all_sessions else 0.0
    reached_target = sum(1 for s in all_sessions if s.reached_target_threshold)
    reached_ultra = sum(1 for s in all_sessions if s.reached_ultra_threshold)
    total_learned = sum(s.new_strategies_learned for s in all_sessions)
    
    # Strategy diversity analysis
    all_strategies_used = set()
    for session in all_sessions:
        for attempt in session.attempts:
            all_strategies_used.add(attempt.strategy_name)
    
    print(f"📊 Results:")
    print(f"   Total Sessions: {total_sessions}")
    print(f"   Average AI Success Rate: {avg_ai_success:.1%}")
    print(f"   Average Decision Diversity: {avg_diversity:.1f}")
    print(f"   Reached Target: {reached_target}/{total_sessions}")
    print(f"   Reached Ultra: {reached_ultra}/{total_sessions}")
    print(f"   📚 Total New Strategies Learned: {total_learned}")
    print(f"   📈 Total Unique Strategies Used: {len(all_strategies_used)}")
    print(f"   🔧 Strategies: {', '.join(sorted(all_strategies_used))}")
    
    # Final push analysis
    final_push_attempts = []
    for session in all_sessions:
        final_attempts = [a for a in session.attempts if a.attempt_number == 7]  # Assuming max_attempts=7
        final_push_attempts.extend(final_attempts)
    
    if final_push_attempts:
        final_push_success = sum(1 for a in final_push_attempts if a.ai_decision.led_to_improvement)
        print(f"   🚀 Final Push Success Rate: {final_push_success}/{len(final_push_attempts)} ({final_push_success/len(final_push_attempts)*100:.1f}%)")
    
    if avg_ai_success >= 0.5 and reached_ultra > 0:
        print(f"\n🎉 EXCELLENT: Good AI performance AND ultra achievements!")
    elif len(all_strategies_used) >= 6:
        print(f"\n🟡 GOOD: Excellent strategy diversity and learning")
    else:
        print(f"\n🔴 IMPROVEMENT NEEDED: Need more strategy diversity and learning")

if __name__ == "__main__":
    main() 