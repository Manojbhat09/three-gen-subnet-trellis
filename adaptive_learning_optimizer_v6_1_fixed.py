#!/usr/bin/env python3
"""
Adaptive Learning Optimizer v6.1 - Fixed & Enhanced Synthesis Engine  
Purpose: Fixed version of v6.0 with robust parsing, anti-repetition, and improved AI decision-making

Key Improvements in v6.1:
- Robust AI response parsing with fallbacks
- Anti-repetition mechanisms to prevent loops
- Better context management and summarization
- Improved strategy execution with validation
- Enhanced persona selection logic
- More reliable database operations
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

# ==============================================================================
# 1. ENHANCED DATA STRUCTURES (v6.1)
# ==============================================================================

@dataclass
class AIDecision:
    """Enhanced AI decision with better tracking."""
    attempt_number: int
    persona_used: str
    decision_type: str
    content: str
    reasoning: str
    confidence: float
    expected_improvement: float
    conversation_turn: int
    based_on_summary: bool
    # Outcome tracking
    actual_improvement: float = 0.0
    led_to_improvement: bool = False
    contributed_to_best_score: bool = False
    timestamp: float = 0.0

@dataclass
class VisibleLearningMoment:
    """Learning moment with enhanced tracking."""
    after_attempt: int
    observation: str
    strategy_effectiveness_update: Dict[str, float]
    decision_influence: str
    timestamp: float = 0.0

@dataclass
class ConversationTurn:
    """Conversation turn with better context."""
    turn_number: int
    user_message_summary: str
    ai_persona: str
    ai_response_summary: str
    strategy_executed: str
    result_score: float
    timestamp: float = 0.0

@dataclass
class OptimizationAttempt:
    """Enhanced attempt with better validation."""
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
    learning_moment: Optional[VisibleLearningMoment]
    is_ai_generated: bool
    timestamp: float = 0.0

@dataclass
class OptimizationSession:
    """Comprehensive session tracking."""
    original_prompt: str
    prompt_category: str
    baseline_score: float
    baseline_fidelity: float
    attempts: List[OptimizationAttempt]
    conversation_history: List[ConversationTurn]
    summaries_created: int
    best_attempt: Optional[OptimizationAttempt]
    session_improvement: float
    reached_minimum_threshold: bool
    reached_target_threshold: bool
    reached_ultra_threshold: bool
    session_success: bool
    ai_decisions_made: int
    ai_decisions_that_improved: int
    ai_contribution_rate: float
    ai_decision_diversity: int
    ai_contributed_to_best_result: bool
    new_strategies_learned: int
    timestamp: float = 0.0

# ==============================================================================
# 2. ENHANCED SYNTHESIS ENGINE (v6.1)
# ==============================================================================

class AdaptiveLearningOptimizerV6_1:
    """Enhanced v6.1 with robust parsing and anti-repetition."""

    def __init__(self, max_attempts: int = 8, min_target: float = 0.6,
                 target: float = 0.9, ultra_target: float = 0.96):
        # Configuration
        self.ollama_url = "http://localhost:11434"
        self.model_name = "deepseek-r1:1.5b"
        self.db_path = "adaptive_optimizer_v6_1.db"
        self.max_attempts = max_attempts
        self.min_target = min_target
        self.target = target
        self.ultra_target = ultra_target

        # Context Management
        self.max_context_chars = 8000  # Reduced for better focus
        
        # Anti-repetition tracking
        self.recent_strategies_used = []
        self.recent_decision_types = []
        self.max_recent_memory = 4

        # Enhanced Strategy Library
        self.base_strategies = {
            "raw": "{prompt}",
            "material_focus": "wbgmsst, solid {prompt} object 3D, white background",
            "geometric_focus": "wbgmsst, {prompt} geometric 3D model, white background",
            "basic_description": "3D model of {prompt}",
            "enhanced_clarity": "wbgmsst, detailed 3D {prompt} model, accurate geometry, white background",
            "concrete_object": "wbgmsst, {prompt} as 3D object, realistic proportions, white background",
            "professional_render": "wbgmsst, professional 3D render of {prompt}, studio lighting, white background",
            "high_quality": "wbgmsst, high quality 3D model {prompt}, detailed textures, white background",
            "ultra_detailed": "wbgmsst, ultra-high detail 3D {prompt}, perfect geometry, professional rendering, white background",
            "photorealistic": "wbgmsst, photorealistic 3D {prompt}, ray-traced lighting, ultra-high quality, white background",
            "technical_spec": "wbgmsst, technical 3D {prompt}, precise dimensions, engineering quality, white background",
            "industrial_design": "wbgmsst, industrial {prompt} design, realistic materials, commercial quality, white background",
            "artistic_sculpt": "wbgmsst, artistic {prompt} sculpture, refined details, museum quality, white background"
        }

        # Learning systems
        self.ai_learned_strategies = {}
        self.strategy_performance = {}
        self.setup_database()
        self.load_historical_learning()

    def setup_database(self):
        """Enhanced database setup with better schema."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Sessions table
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
                reached_minimum BOOLEAN,
                reached_target BOOLEAN,
                reached_ultra BOOLEAN,
                session_success BOOLEAN,
                ai_contribution_rate REAL,
                ai_decision_diversity INTEGER,
                new_strategies_learned INTEGER
            )
        ''')
        
        # Attempts table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS attempts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id INTEGER,
                attempt_number INTEGER,
                strategy_name TEXT,
                validation_score REAL,
                score_improvement REAL,
                is_ai_generated BOOLEAN,
                persona_used TEXT,
                decision_type TEXT,
                FOREIGN KEY (session_id) REFERENCES sessions (id)
            )
        ''')
        
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
        
        conn.commit()
        conn.close()

    def load_historical_learning(self):
        """Load historical knowledge with error handling."""
        print("🧠 Loading historical knowledge...")
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
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
            
            print(f"✅ Loaded performance data for {len(self.strategy_performance)} strategies")
            
        except sqlite3.OperationalError as e:
            print(f"   - DB tables may be new: {e}")
        finally:
            conn.close()

    def query_ai(self, messages: List[Dict[str, str]], timeout: int = 60) -> str:
        """Enhanced AI query with better error handling."""
        data = {
            "model": self.model_name,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": 0.7,  # Good balance of creativity and consistency
                "top_p": 0.9,
                "num_predict": 400
            }
        }
        
        try:
            response = requests.post(f"{self.ollama_url}/api/chat", json=data, timeout=timeout)
            response.raise_for_status()
            content = response.json()["message"]["content"]
            
            if len(content.strip()) < 10:
                return "ERROR: Response too short"
            
            return content
        except Exception as e:
            print(f"❌ AI Query Failed: {e}")
            return f"ERROR: {str(e)}"

    def run_validation(self, prompt: str) -> Tuple[float, float]:
        """Enhanced validation with better error handling."""
        try:
            cmd = [sys.executable, "subnet_accurate_validator.py", prompt]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
            
            if result.returncode != 0:
                print(f"   ⚠️ Validation failed: {result.stderr}")
                return 0.0, 0.0
            
            with open("subnet_validation_results.json", 'r') as f:
                data = json.load(f)
                val_score = data.get("validation_engine_score", 0.0)
                demo_score = data.get("demo_fidelity_score", 0.0)
                return val_score, demo_score
                
        except subprocess.TimeoutExpired:
            print(f"   ⚠️ Validation timeout")
            return 0.0, 0.0
        except FileNotFoundError:
            print(f"   ❌ Validation script not found")
            return 0.0, 0.0
        except Exception as e:
            print(f"   ❌ Validation error: {e}")
            return 0.0, 0.0

    def categorize_prompt(self, prompt: str) -> str:
        """Simple but effective categorization."""
        prompt_lower = prompt.lower()
        
        # Technical/geometric indicators
        if any(word in prompt_lower for word in ["steel", "metal", "copper", "iron", "aluminum", "geometric", "prism", "cylinder", "sphere", "hexagonal", "diameter"]):
            return "technical_description"
        
        # Artistic/decorative indicators
        elif any(word in prompt_lower for word in ["elegant", "ornate", "artistic", "decorative", "gothic", "ornamental", "beautiful"]):
            return "artistic_abstract"
        
        # Fashion/textile indicators
        elif any(word in prompt_lower for word in ["fabric", "silk", "cotton", "dress", "shirt", "clothing", "textile"]):
            return "fashion_clothing"
        
        # Default to physical object
        else:
            return "physical_object"

    # ==============================================================================
    # 3. ENHANCED AI DECISION MAKING (v6.1)
    # ==============================================================================

    def select_ai_persona(self, attempt_num: int, baseline_score: float, attempts: List[OptimizationAttempt]) -> str:
        """Enhanced persona selection with anti-repetition."""
        
        # Check recent personas to avoid repetition
        recent_personas = [a.ai_decision.persona_used for a in attempts[-3:]]
        
        # Strategic selection based on situation
        if baseline_score < 0.3:
            if "DataAnalyst" not in recent_personas:
                return "DataAnalyst"
            elif "CreativePrompter" not in recent_personas:
                return "CreativePrompter"
        
        elif attempt_num > 3 and all(a.score_improvement <= 0.02 for a in attempts[-2:]):
            if "CreativePrompter" not in recent_personas:
                return "CreativePrompter"
            elif "StrategicPlanner" not in recent_personas:
                return "StrategicPlanner"
        
        else:
            if "StrategicPlanner" not in recent_personas:
                return "StrategicPlanner"
        
        # Fallback with randomization
        personas = ["StrategicPlanner", "CreativePrompter", "DataAnalyst"]
        available = [p for p in personas if p not in recent_personas]
        return random.choice(available) if available else random.choice(personas)

    def build_focused_context(self, session_context: dict) -> Tuple[str, bool]:
        """Build focused, actionable context for AI."""
        
        attempts = session_context["attempts"]
        
        # Recent attempts summary (last 3)
        recent_summary = []
        for attempt in attempts[-3:]:
            recent_summary.append(
                f"Attempt {attempt.attempt_number}: {attempt.strategy_name} -> "
                f"{attempt.validation_score:.3f} ({attempt.score_improvement:+.3f})"
            )
        
        # Strategy performance summary
        strategy_perf = self.get_strategy_performance_summary(session_context["category"])
        
        # Anti-repetition info
        recent_strategies = [a.strategy_name for a in attempts[-3:]]
        avoid_strategies = list(set(recent_strategies)) if len(recent_strategies) > 1 else []
        
        context = f"""OPTIMIZATION CONTEXT:
Prompt: "{session_context['prompt']}" (Category: {session_context['category']})
Baseline: {session_context['baseline_score']:.3f}
Attempt: {session_context['attempt_number']}/{self.max_attempts}
Targets: Min {self.min_target} | Target {self.target} | Ultra {self.ultra_target}

RECENT ATTEMPTS:
{chr(10).join(recent_summary) if recent_summary else "None yet"}

STRATEGY PERFORMANCE (for {session_context['category']}):
{strategy_perf}

ANTI-REPETITION:
Recently used strategies to avoid: {', '.join(avoid_strategies) if avoid_strategies else 'None'}
Recently used decision types: {', '.join(self.recent_decision_types[-2:]) if self.recent_decision_types else 'None'}"""

        # Check if context too long
        if len(context) > self.max_context_chars:
            print(f"   📊 Context too long, summarizing...")
            summary = self.create_smart_summary(context)
            return summary, True
        
        return context, False

    def get_strategy_performance_summary(self, category: str) -> str:
        """Get focused strategy performance summary."""
        if category not in [perf.get(category, {}) for perf in self.strategy_performance.values() if perf]:
            return "No historical data for this category"
        
        relevant_strategies = []
        for strategy, cats in self.strategy_performance.items():
            if category in cats:
                perf = cats[category]
                if perf['usage_count'] >= 2:
                    relevant_strategies.append((strategy, perf['avg_improvement'], perf['success_rate']))
        
        if not relevant_strategies:
            return "Limited historical data"
        
        # Sort by improvement
        sorted_strategies = sorted(relevant_strategies, key=lambda x: x[1], reverse=True)[:5]
        
        summary_lines = []
        for strategy, improvement, rate in sorted_strategies:
            summary_lines.append(f"  {strategy}: {improvement:+.3f} avg, {rate:.1%} success")
        
        return "\n".join(summary_lines)

    def create_smart_summary(self, full_context: str) -> str:
        """Create intelligent summary when context gets too long."""
        
        summary_prompt = """You are a context summarizer. Create a concise summary focusing on:
1. What strategies worked/failed recently
2. Current best score and improvement needed
3. Key patterns to continue or avoid

Keep it under 300 words and actionable."""
        
        try:
            summary = self.query_ai([
                {"role": "system", "content": summary_prompt},
                {"role": "user", "content": f"Summarize this optimization context:\n{full_context}"}
            ])
            
            if "ERROR:" not in summary:
                return f"CONTEXT SUMMARY:\n{summary}"
        except:
            pass
        
        # Fallback summary
        return f"SUMMARY: Working on prompt optimization, recent attempts show mixed results, need to reach target {self.target}"

    def make_ai_decision(self, session_context: dict) -> AIDecision:
        """Enhanced AI decision making with robust parsing."""
        
        attempt_num = session_context["attempt_number"]
        
        # 1. Select persona with anti-repetition
        persona = self.select_ai_persona(attempt_num, session_context["baseline_score"], session_context["attempts"])
        print(f"🤖 AI Persona: {persona}")
        
        # 2. Build focused context
        context_str, based_on_summary = self.build_focused_context(session_context)
        
        # 3. Get persona-specific system prompt
        system_prompt = self.get_enhanced_persona_prompt(persona, session_context, context_str)
        
        # 4. Query AI with structured request
        user_message = f"Based on the context, make your optimization decision for attempt {attempt_num}. Think about what approach would be most effective given the current situation."
        
        try:
            ai_response = self.query_ai([
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message}
            ])
            
            if "ERROR:" in ai_response:
                return self.create_fallback_decision(attempt_num, persona, session_context["turn"], based_on_summary, f"AI error: {ai_response}")
            
            # 5. Parse with robust parsing
            decision = self.parse_ai_response_robust(ai_response, attempt_num, persona, session_context["turn"], based_on_summary)
            
            # 6. Update anti-repetition tracking
            self.recent_decision_types.append(decision.decision_type)
            if len(self.recent_decision_types) > self.max_recent_memory:
                self.recent_decision_types.pop(0)
            
            return decision
            
        except Exception as e:
            print(f"❌ AI decision error: {e}")
            return self.create_fallback_decision(attempt_num, persona, session_context["turn"], based_on_summary, f"Exception: {e}")

    def get_enhanced_persona_prompt(self, persona: str, session_context: dict, context_str: str) -> str:
        """Enhanced persona prompts with better guidance."""
        
        available_strategies = list(self.base_strategies.keys()) + list(self.ai_learned_strategies.keys())
        
        base_instruction = f"""You are an expert 3D prompt optimization AI.

{context_str}

AVAILABLE STRATEGIES: {available_strategies}

Your goal is to reach the targets through smart decisions. Respond in this format:
DECISION: [CUSTOM_PROMPT|SELECT_STRATEGY|STRATEGY_SEQUENCE|EARLY_STOP]
REASONING: [Your analysis and justification]
CONFIDENCE: [0.1 to 1.0]
EXPECTED_IMPROVEMENT: [0.0 to 0.5]
CONTENT: [Custom prompt OR strategy name OR strategy list OR stop reason]

Important: Avoid repeating recently failed approaches. Be creative and strategic."""

        if persona == "StrategicPlanner":
            return f"""You are a STRATEGIC PLANNER AI focused on efficient optimization.

Your approach:
- Analyze patterns in recent attempts
- Select strategies with highest success probability
- Plan multiple steps ahead
- Balance exploration with exploitation

{base_instruction}"""

        elif persona == "CreativePrompter":
            return f"""You are a CREATIVE PROMPTER AI focused on breaking through optimization barriers.

Your approach:
- Generate completely new custom prompts when stuck
- Think outside the box with creative interpretations
- Combine concepts in novel ways
- Take calculated risks for breakthrough results

{base_instruction}"""

        elif persona == "DataAnalyst":
            return f"""You are a DATA ANALYST AI focused on evidence-based optimization.

Your approach:
- Base decisions strictly on performance data
- Identify what's working and what's not
- Choose strategies with proven track records
- Minimize risk while maximizing expected improvement

{base_instruction}"""

        return base_instruction

    def parse_ai_response_robust(self, response: str, attempt_num: int, persona: str, turn: int, based_on_summary: bool) -> AIDecision:
        """Robust AI response parsing with multiple fallback mechanisms."""
        
        # Initialize defaults
        decision_type = "SELECT_STRATEGY"
        reasoning = response[:200] if len(response) > 200 else response
        confidence = 0.5
        expected_improvement = 0.1
        content = "enhanced_clarity"
        
        try:
            # Try structured parsing first
            decision_match = re.search(r'DECISION:\s*(.+?)(?:\n|$)', response, re.IGNORECASE)
            if decision_match:
                decision_type = decision_match.group(1).strip()
            
            reason_match = re.search(r'REASONING:\s*(.+?)(?=CONFIDENCE:|EXPECTED_IMPROVEMENT:|CONTENT:|$)', response, re.DOTALL | re.IGNORECASE)
            if reason_match:
                reasoning = reason_match.group(1).strip()
            
            conf_match = re.search(r'CONFIDENCE:\s*([0-9.]+)', response, re.IGNORECASE)
            if conf_match:
                confidence = min(1.0, max(0.1, float(conf_match.group(1))))
            
            exp_match = re.search(r'EXPECTED_IMPROVEMENT:\s*([0-9.]+)', response, re.IGNORECASE)
            if exp_match:
                expected_improvement = min(0.5, max(0.0, float(exp_match.group(1))))
            
            content_match = re.search(r'CONTENT:\s*(.+)', response, re.DOTALL | re.IGNORECASE)
            if content_match:
                content = content_match.group(1).strip()
            
            # Normalize decision type
            decision_type = decision_type.upper()
            if "CUSTOM" in decision_type:
                decision_type = "CUSTOM_PROMPT"
            elif "STRATEGY_SEQUENCE" in decision_type or "SEQUENCE" in decision_type:
                decision_type = "STRATEGY_SEQUENCE"
            elif "SELECT" in decision_type or "STRATEGY" in decision_type:
                decision_type = "SELECT_STRATEGY"
            elif "STOP" in decision_type or "EARLY" in decision_type:
                decision_type = "EARLY_STOP"
            else:
                decision_type = "SELECT_STRATEGY"  # Safe fallback
            
        except Exception as e:
            print(f"   ⚠️ Structured parsing failed, using semantic parsing: {e}")
            
            # Semantic parsing fallback
            response_lower = response.lower()
            
            if any(keyword in response_lower for keyword in ["custom prompt", "write", "create prompt", "new prompt"]):
                decision_type = "CUSTOM_PROMPT"
                # Try to extract custom prompt
                lines = response.split('\n')
                for line in lines:
                    if any(word in line.lower() for word in ["wbgmsst", "3d model", "white background"]):
                        content = line.strip()
                        break
                if content == "enhanced_clarity":  # Still default
                    content = f"wbgmsst, optimized 3D {reasoning.split()[0] if reasoning else 'model'}, high quality, white background"
            
            elif any(keyword in response_lower for keyword in ["stop", "terminate", "end", "quit"]):
                decision_type = "EARLY_STOP"
                content = "AI decided to stop"
            
            else:
                decision_type = "SELECT_STRATEGY"
                # Try to find strategy names in response
                for strategy in self.base_strategies.keys():
                    if strategy in response_lower:
                        content = strategy
                        break
        
        return AIDecision(
            attempt_number=attempt_num,
            persona_used=persona,
            decision_type=decision_type,
            content=content,
            reasoning=reasoning,
            confidence=confidence,
            expected_improvement=expected_improvement,
            conversation_turn=turn,
            based_on_summary=based_on_summary,
            timestamp=time.time()
        )

    def create_fallback_decision(self, attempt_num: int, persona: str, turn: int, based_on_summary: bool, error_msg: str) -> AIDecision:
        """Create intelligent fallback decision."""
        
        # Choose fallback strategy based on attempt number
        fallback_strategies = ["enhanced_clarity", "professional_render", "high_quality", "concrete_object"]
        fallback_strategy = fallback_strategies[(attempt_num - 1) % len(fallback_strategies)]
        
        return AIDecision(
            attempt_number=attempt_num,
            persona_used=persona,
            decision_type="SELECT_STRATEGY",
            content=fallback_strategy,
            reasoning=f"Fallback decision due to: {error_msg}",
            confidence=0.3,
            expected_improvement=0.05,
            conversation_turn=turn,
            based_on_summary=based_on_summary,
            timestamp=time.time()
        )

    def execute_ai_decision(self, decision: AIDecision, prompt: str) -> Tuple[str, str]:
        """Enhanced decision execution with validation."""
        
        decision_type = decision.decision_type
        content = decision.content
        
        if decision_type == "EARLY_STOP":
            return "early_stop", prompt
        
        elif decision_type == "CUSTOM_PROMPT":
            if len(content) > 10 and content != "enhanced_clarity":
                return "ai_custom_prompt", content
            else:
                # Generate a better custom prompt
                custom_prompt = f"wbgmsst, detailed 3D {prompt}, high quality model, professional rendering, white background"
                return "ai_custom_prompt", custom_prompt
        
        elif decision_type in ["SELECT_STRATEGY", "STRATEGY_SEQUENCE"]:
            # Handle strategy selection
            all_strategies = {**self.base_strategies, **self.ai_learned_strategies}
            
            if decision_type == "STRATEGY_SEQUENCE":
                # For sequence, take first valid strategy
                strategies = [s.strip() for s in content.replace('\n', ',').split(',')]
                for strategy in strategies:
                    if strategy in all_strategies:
                        return strategy, all_strategies[strategy].format(prompt=prompt)
            else:
                # Single strategy
                if content in all_strategies:
                    return content, all_strategies[content].format(prompt=prompt)
                else:
                    # Try to find partial match
                    for strategy_name in all_strategies.keys():
                        if strategy_name in content.lower() or content.lower() in strategy_name:
                            return strategy_name, all_strategies[strategy_name].format(prompt=prompt)
        
        # Final fallback
        print(f"   ⚠️ Decision execution fallback: {decision_type} -> {content}")
        return "enhanced_clarity", self.base_strategies["enhanced_clarity"].format(prompt=prompt)

    # ==============================================================================
    # 4. MAIN OPTIMIZATION LOOP (v6.1)
    # ==============================================================================

    def optimize_prompt(self, prompt: str) -> OptimizationSession:
        """Enhanced main optimization loop."""
        
        print(f"\n🚀 ADAPTIVE OPTIMIZER v6.1: '{prompt}'")
        print("=" * 80)
        print(f"🎯 Targets: Min {self.min_target} | Target {self.target} | Ultra {self.ultra_target}")
        
        # Reset anti-repetition tracking for new session
        self.recent_strategies_used = []
        self.recent_decision_types = []
        
        # 1. Initial setup
        category = self.categorize_prompt(prompt)
        print(f"📋 Category: {category}")
        
        baseline_score, baseline_fidelity = self.run_validation(prompt)
        print(f"📊 Baseline: {baseline_score:.3f}")
        
        if baseline_score >= self.ultra_target:
            print(f"🏆 BASELINE ALREADY ULTRA-OPTIMAL!")
            return self.create_final_session(prompt, category, baseline_score, baseline_fidelity, [])
        
        # 2. Session tracking
        attempts = []
        conversation_history = []
        best_score = baseline_score
        best_attempt = None
        ai_contributed_to_best = False
        summaries_created = 0
        new_strategies_learned = 0
        
        # 3. Main optimization loop
        for i in range(1, self.max_attempts + 1):
            print(f"\n🔄 ATTEMPT {i}/{self.max_attempts}")
            
            # Build session context
            session_context = {
                "prompt": prompt,
                "category": category,
                "baseline_score": baseline_score,
                "attempt_number": i,
                "attempts": attempts,
                "turn": len(conversation_history) + 1,
                "summaries_created": summaries_created
            }
            
            # AI Decision
            print(f"🤖 AI analyzing and making decision...")
            ai_decision = self.make_ai_decision(session_context)
            
            print(f"   🧠 Decision: {ai_decision.decision_type}")
            print(f"   💭 Reasoning: {ai_decision.reasoning[:100]}...")
            print(f"   🎯 Confidence: {ai_decision.confidence:.2f}")
            print(f"   📈 Expected: +{ai_decision.expected_improvement:.3f}")
            
            # Execute decision
            strategy_name, optimized_prompt = self.execute_ai_decision(ai_decision, prompt)
            
            if strategy_name == "early_stop":
                print(f"   🛑 AI Early Stop")
                break
            
            # Update anti-repetition tracking
            self.recent_strategies_used.append(strategy_name)
            if len(self.recent_strategies_used) > self.max_recent_memory:
                self.recent_strategies_used.pop(0)
            
            is_ai_generated = strategy_name == "ai_custom_prompt"
            
            print(f"   🔧 Executing: {strategy_name}")
            print(f"   ✨ Prompt: '{optimized_prompt[:70]}{'...' if len(optimized_prompt) > 70 else ''}'")
            
            # Validate
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
                ai_contributed_to_best = True
                print(f"   🌟 NEW BEST SCORE!")
            
            # Create learning moment
            learning_moment = VisibleLearningMoment(
                after_attempt=i,
                observation=f"Strategy '{strategy_name}' {'succeeded' if improvement > 0 else 'failed'} with {improvement:+.3f} improvement",
                strategy_effectiveness_update={strategy_name: 0.1 if improvement > 0 else -0.05},
                decision_influence=f"Will {'favor' if improvement > 0 else 'avoid'} {strategy_name} in future",
                timestamp=time.time()
            )
            
            # Create attempt record
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
                learning_moment=learning_moment,
                is_ai_generated=is_ai_generated,
                timestamp=time.time()
            )
            attempts.append(attempt)
            
            if ai_decision.contributed_to_best_score:
                best_attempt = attempt
            
            # Update conversation
            turn = ConversationTurn(
                turn_number=len(conversation_history) + 1,
                user_message_summary=f"Optimize attempt {i}",
                ai_persona=ai_decision.persona_used,
                ai_response_summary=f"{ai_decision.decision_type} (conf: {ai_decision.confidence:.2f})",
                strategy_executed=strategy_name,
                result_score=val_score,
                timestamp=time.time()
            )
            conversation_history.append(turn)
            
            # Check for ultra achievement
            if val_score >= self.ultra_target:
                print(f"   🏆 ULTRA TARGET ACHIEVED!")
                break
            
            time.sleep(1)
        
        # Post-session learning
        learned_count = self.learn_from_session(prompt, category, attempts)
        
        # Create final session
        return self.create_final_session(prompt, category, baseline_score, baseline_fidelity, attempts, conversation_history, best_attempt, ai_contributed_to_best, summaries_created, learned_count)

    def learn_from_session(self, prompt: str, category: str, attempts: List[OptimizationAttempt]) -> int:
        """Enhanced learning with better tracking."""
        
        if not attempts:
            return 0
        
        print("🧠 Updating knowledge base...")
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        learned_count = 0
        
        try:
            # Update strategy performance
            for attempt in attempts:
                strategy = attempt.strategy_name
                improvement = attempt.score_improvement
                is_success = improvement > 0.01
                
                # Get current stats
                cursor.execute("SELECT success_rate, avg_improvement, usage_count FROM strategy_performance WHERE strategy_name = ? AND category = ?", (strategy, category))
                row = cursor.fetchone()
                
                if row:
                    old_rate, old_imp, old_count = row
                    new_count = old_count + 1
                    new_rate = ((old_rate * old_count) + (1 if is_success else 0)) / new_count
                    new_imp = ((old_imp * old_count) + improvement) / new_count
                    
                    cursor.execute("""
                        UPDATE strategy_performance 
                        SET success_rate = ?, avg_improvement = ?, usage_count = ?, last_used = ?
                        WHERE strategy_name = ? AND category = ?
                    """, (new_rate, new_imp, new_count, time.time(), strategy, category))
                else:
                    cursor.execute("""
                        INSERT INTO strategy_performance (strategy_name, category, success_rate, avg_improvement, usage_count, last_used)
                        VALUES (?, ?, ?, ?, ?, ?)
                    """, (strategy, category, 1.0 if is_success else 0.0, improvement, 1, time.time()))
            
            # Learn from ultra-successful custom prompts
            for attempt in attempts:
                if attempt.is_ai_generated and attempt.meets_ultra_threshold:
                    print(f"   🎓 Learning from ultra-successful custom prompt...")
                    # This could be expanded to extract patterns
                    learned_count += 1
            
            conn.commit()
            print(f"✅ Knowledge updated, {learned_count} new insights learned")
            
        except Exception as e:
            print(f"❌ Learning error: {e}")
        finally:
            conn.close()
        
        return learned_count

    def create_final_session(self, prompt: str, category: str, baseline_score: float, baseline_fidelity: float, 
                           attempts: List[OptimizationAttempt], conversation_history: List[ConversationTurn] = None,
                           best_attempt: OptimizationAttempt = None, ai_contributed_to_best: bool = False,
                           summaries_created: int = 0, new_strategies_learned: int = 0) -> OptimizationSession:
        """Create comprehensive final session."""
        
        if conversation_history is None:
            conversation_history = []
        
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
            baseline_fidelity=baseline_fidelity,
            attempts=attempts,
            conversation_history=conversation_history,
            summaries_created=summaries_created,
            best_attempt=best_attempt,
            session_improvement=session_improvement,
            reached_minimum_threshold=any(a.meets_minimum_threshold for a in attempts) if attempts else False,
            reached_target_threshold=any(a.meets_target_threshold for a in attempts) if attempts else False,
            reached_ultra_threshold=any(a.meets_ultra_threshold for a in attempts) if attempts else False,
            session_success=(session_improvement > 0.01 and any(a.meets_minimum_threshold for a in attempts)) if attempts else False,
            ai_decisions_made=ai_decisions_made,
            ai_decisions_that_improved=ai_decisions_that_improved,
            ai_contribution_rate=ai_contribution_rate,
            ai_decision_diversity=ai_decision_diversity,
            ai_contributed_to_best_result=ai_contributed_to_best,
            new_strategies_learned=new_strategies_learned,
            timestamp=time.time()
        )
        
        # Print session summary
        print(f"\n📊 SESSION SUMMARY v6.1:")
        print(f"   📈 Best Score: {best_score:.3f} (Baseline: {baseline_score:.3f})")
        print(f"   📊 Session Improvement: {session_improvement:+.3f}")
        print(f"   🤖 AI Decisions: {ai_decisions_made}")
        print(f"   ✅ AI Success Rate: {ai_contribution_rate:.1%}")
        print(f"   🎯 Decision Diversity: {ai_decision_diversity} types")
        print(f"   💭 Conversation Turns: {len(conversation_history)}")
        print(f"   📚 Strategies Learned: {new_strategies_learned}")
        print(f"   🎯 Targets: Min {'✅' if session.reached_minimum_threshold else '❌'} | Target {'✅' if session.reached_target_threshold else '❌'} | Ultra {'✅' if session.reached_ultra_threshold else '❌'}")
        
        return session

def main():
    """Enhanced main function with better testing."""
    
    test_prompts = [
        "hexagonal prism steel structure",
        "elegant silk fabric draping",
        "transparent glass sphere reflection"
    ]
    
    print("🚀 ADAPTIVE LEARNING OPTIMIZER v6.1 - ENHANCED SYNTHESIS")
    print("=" * 80)
    print("🔧 Features: Robust parsing, anti-repetition, enhanced personas")
    print("🧠 AI: Smart decision-making with memory and learning")
    print("📊 Tracking: Comprehensive metrics and performance analysis")
    print("=" * 80)
    
    optimizer = AdaptiveLearningOptimizerV6_1(
        max_attempts=6,
        min_target=0.6,
        target=0.9,
        ultra_target=0.96
    )
    
    all_sessions = []
    
    for i, prompt in enumerate(test_prompts, 1):
        print(f"\n{'='*20} [{i}/{len(test_prompts)}] PROMPT {i} {'='*20}")
        session = optimizer.optimize_prompt(prompt)
        all_sessions.append(session)
        time.sleep(2)
    
    # Final analysis
    print(f"\n🎓 FINAL ANALYSIS - v6.1 ENHANCED")
    print("=" * 80)
    
    total_sessions = len(all_sessions)
    avg_ai_success = statistics.mean([s.ai_contribution_rate for s in all_sessions]) if all_sessions else 0.0
    avg_decision_diversity = statistics.mean([s.ai_decision_diversity for s in all_sessions]) if all_sessions else 0.0
    reached_target = sum(1 for s in all_sessions if s.reached_target_threshold)
    reached_ultra = sum(1 for s in all_sessions if s.reached_ultra_threshold)
    
    print(f"📊 Results:")
    print(f"   Total Sessions: {total_sessions}")
    print(f"   Average AI Success Rate: {avg_ai_success:.1%}")
    print(f"   Average Decision Diversity: {avg_decision_diversity:.1f}")
    print(f"   Reached Target: {reached_target}/{total_sessions}")
    print(f"   Reached Ultra: {reached_ultra}/{total_sessions}")
    
    # Save results
    results = {
        "v6_1_analysis": {
            "total_sessions": total_sessions,
            "avg_ai_success_rate": avg_ai_success,
            "avg_decision_diversity": avg_decision_diversity,
            "reached_target": reached_target,
            "reached_ultra": reached_ultra,
            "timestamp": time.time()
        },
        "sessions": [asdict(s) for s in all_sessions]
    }
    
    output_file = f"adaptive_optimizer_v6_1_results_{int(time.time())}.json"
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\n💾 Results saved to: {output_file}")
    print("✅ v6.1 Enhanced Optimization Complete!")

if __name__ == "__main__":
    main() 