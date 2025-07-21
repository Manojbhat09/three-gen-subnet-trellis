#!/usr/bin/env python3
"""
Adaptive Learning Optimizer v5.0 Conversational - AI Memory & Context
Purpose: Conversational AI that remembers all previous attempts and decisions,
with automatic summarization when context gets too long for better decision diversity.
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
class ConversationTurn:
    """Single turn in the AI conversation"""
    turn_number: int
    user_message: str
    ai_response: str
    decision_made: str
    strategy_executed: str
    result_score: float
    result_improvement: float
    timestamp: float

@dataclass
class ConversationSummary:
    """Summary of previous conversation when context gets too long"""
    summary_after_turn: int
    key_findings: str
    strategy_effectiveness: Dict[str, str]  # strategy -> "effective/ineffective/mixed"
    best_approaches: List[str]
    failed_approaches: List[str]
    current_best_score: float
    recommendations_for_next: str
    timestamp: float

@dataclass
class AIDecision:
    """AI Decision with conversation context"""
    attempt_number: int
    decision_type: str
    content: str
    reasoning: str
    confidence: float
    expected_improvement: float
    conversation_turn: int
    based_on_summary: bool  # True if decision was made after conversation summary
    actual_improvement: float
    decision_success: bool
    timestamp: float

@dataclass
class OptimizationAttempt:
    """Attempt with conversation context"""
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
    ai_decision: AIDecision
    conversation_turn: int
    timestamp: float

@dataclass
class OptimizationSession:
    """Session with full conversation history"""
    original_prompt: str
    category: str
    baseline_score: float
    baseline_fidelity: float
    attempts: List[OptimizationAttempt]
    conversation_history: List[ConversationTurn]
    conversation_summaries: List[ConversationSummary]
    best_attempt: Optional[OptimizationAttempt]
    best_score: float
    session_improvement: float
    ai_decisions_made: int
    ai_successful_decisions: int
    ai_decision_diversity: int  # Number of unique decision types used
    reached_minimum: bool
    reached_target: bool
    reached_ultra: bool
    timestamp: float

class ConversationalAIOptimizer:
    """v5.0 Conversational with AI memory and context management"""
    
    def __init__(self, max_attempts: int = 8, min_target: float = 0.6, 
                 target: float = 0.9, ultra_target: float = 0.96):
        self.ollama_url = "http://localhost:11434"
        self.model_name = "deepseek-r1:1.5b"
        self.max_attempts = max_attempts
        self.min_target = min_target
        self.target = target
        self.ultra_target = ultra_target
        
        # Context management (approximate VRAM usage)
        self.max_context_chars = 12000  # ~5GB equivalent for DeepSeek
        self.conversation_history = []
        self.conversation_summaries = []
        self.current_turn = 0
        
        # Strategy library
        self.strategies = {
            "raw": "{prompt}",
            "material_focus": "wbgmsst, solid {prompt} object 3D, white background",
            "geometric_focus": "wbgmsst, {prompt} geometric 3D model, white background",
            "basic_description": "3D model of {prompt}",
            "enhanced_clarity": "wbgmsst, detailed 3D {prompt} model, accurate geometry, white background",
            "concrete_object": "wbgmsst, {prompt} as 3D object, realistic proportions, white background",
            "professional_render": "wbgmsst, professional 3D render of {prompt}, studio lighting, white background",
            "high_quality": "wbgmsst, high quality 3D model {prompt}, detailed textures, white background",
            "minimal_enhancement": "{prompt}, 3D object",
            "simplified_description": "simple 3D {prompt}",
            "artistic_focus": "wbgmsst, artistic {prompt} sculpture, clean design, white background",
            "technical_spec": "wbgmsst, technical {prompt} model, precise geometry, engineering quality, white background",
            "industrial_design": "wbgmsst, industrial {prompt} design, realistic materials, white background"
        }
        
    def query_deepseek_conversational(self, messages: List[Dict[str, str]], timeout: int = 90) -> str:
        """Query DeepSeek with full conversation context"""
        data = {
            "model": self.model_name,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": 0.8,  # Higher for more creativity
                "top_p": 0.9,
                "num_predict": 500
            }
        }
        
        try:
            response = requests.post(f"{self.ollama_url}/api/chat", json=data, timeout=timeout)
            response.raise_for_status()
            result = response.json()
            content = result["message"]["content"]
            
            if len(content.strip()) < 5:
                return "ERROR: Response too short"
            
            return content
        except Exception as e:
            return f"ERROR: {str(e)}"
    
    def estimate_context_size(self, messages: List[Dict[str, str]]) -> int:
        """Estimate context size in characters"""
        return sum(len(msg.get("content", "")) for msg in messages)
    
    def create_conversation_summary(self, prompt: str, category: str) -> ConversationSummary:
        """Create summary of conversation when context gets too long"""
        
        if not self.conversation_history:
            return None
        
        # Prepare conversation data for summarization
        conversation_data = []
        for turn in self.conversation_history:
            conversation_data.append({
                "turn": turn.turn_number,
                "decision": turn.decision_made,
                "strategy": turn.strategy_executed,
                "score": turn.result_score,
                "improvement": turn.result_improvement
            })
        
        # Best and worst performing strategies
        strategy_results = {}
        for turn in self.conversation_history:
            strategy = turn.strategy_executed
            if strategy not in strategy_results:
                strategy_results[strategy] = []
            strategy_results[strategy].append(turn.result_improvement)
        
        # Summarization prompt
        summary_messages = [
            {
                "role": "system",
                "content": f"""You are an expert AI that creates concise summaries of optimization conversations.

TASK: Summarize the optimization conversation for the prompt "{prompt}" (Category: {category}).

TARGETS: Min {self.min_target} | Target {self.target} | Ultra {self.ultra_target}

CONVERSATION DATA:
{json.dumps(conversation_data, indent=2)}

STRATEGY PERFORMANCE:
{json.dumps(strategy_results, indent=2)}

Create a concise summary that captures:
1. Key findings about what works/doesn't work
2. Strategy effectiveness patterns
3. Best approaches tried so far
4. Failed approaches to avoid
5. Current best score and improvement
6. Specific recommendations for next attempts

RESPOND IN THIS FORMAT:
KEY_FINDINGS: [2-3 sentences about main discoveries]
STRATEGY_EFFECTIVENESS: [List each strategy as: strategy=effective/ineffective/mixed with brief reason]
BEST_APPROACHES: [List top 2-3 approaches that worked]
FAILED_APPROACHES: [List 2-3 approaches that failed consistently]
CURRENT_BEST_SCORE: [best score achieved]
RECOMMENDATIONS: [Specific suggestions for next attempts to try different approaches]

Be specific and actionable - this summary will guide future decisions!"""
            },
            {
                "role": "user", 
                "content": f"Summarize optimization conversation for: '{prompt}'"
            }
        ]
        
        try:
            response = self.query_deepseek_conversational(summary_messages)
            
            if "ERROR:" in response:
                return self._create_fallback_summary()
            
            # Parse summary response
            key_findings = "Standard optimization attempted"
            strategy_effectiveness = {}
            best_approaches = []
            failed_approaches = []
            current_best = max((turn.result_score for turn in self.conversation_history), default=0.0)
            recommendations = "Try different strategies"
            
            # Extract key findings
            findings_match = re.search(r'KEY_FINDINGS:\s*(.+?)(?=STRATEGY_EFFECTIVENESS:|$)', response, re.DOTALL)
            if findings_match:
                key_findings = findings_match.group(1).strip()
            
            # Extract strategy effectiveness
            effect_match = re.search(r'STRATEGY_EFFECTIVENESS:\s*(.+?)(?=BEST_APPROACHES:|$)', response, re.DOTALL)
            if effect_match:
                effect_text = effect_match.group(1).strip()
                for line in effect_text.split('\n'):
                    if '=' in line:
                        parts = line.split('=', 1)
                        if len(parts) == 2:
                            strategy = parts[0].strip()
                            effectiveness = parts[1].strip()
                            strategy_effectiveness[strategy] = effectiveness
            
            # Extract best approaches
            best_match = re.search(r'BEST_APPROACHES:\s*(.+?)(?=FAILED_APPROACHES:|$)', response, re.DOTALL)
            if best_match:
                best_text = best_match.group(1).strip()
                best_approaches = [a.strip() for a in best_text.replace('\n', ',').split(',') if a.strip()]
            
            # Extract failed approaches
            failed_match = re.search(r'FAILED_APPROACHES:\s*(.+?)(?=CURRENT_BEST_SCORE:|$)', response, re.DOTALL)
            if failed_match:
                failed_text = failed_match.group(1).strip()
                failed_approaches = [a.strip() for a in failed_text.replace('\n', ',').split(',') if a.strip()]
            
            # Extract recommendations
            rec_match = re.search(r'RECOMMENDATIONS:\s*(.+)', response, re.DOTALL)
            if rec_match:
                recommendations = rec_match.group(1).strip()
            
            return ConversationSummary(
                summary_after_turn=self.current_turn,
                key_findings=key_findings,
                strategy_effectiveness=strategy_effectiveness,
                best_approaches=best_approaches[:3],
                failed_approaches=failed_approaches[:3],
                current_best_score=current_best,
                recommendations_for_next=recommendations,
                timestamp=time.time()
            )
            
        except Exception as e:
            print(f"❌ Summarization error: {e}")
            return self._create_fallback_summary()
    
    def _create_fallback_summary(self) -> ConversationSummary:
        """Create fallback summary when AI summarization fails"""
        current_best = max((turn.result_score for turn in self.conversation_history), default=0.0)
        
        return ConversationSummary(
            summary_after_turn=self.current_turn,
            key_findings="Unable to analyze conversation - AI summarization failed",
            strategy_effectiveness={},
            best_approaches=["enhanced_clarity"],
            failed_approaches=[],
            current_best_score=current_best,
            recommendations_for_next="Try different optimization strategies",
            timestamp=time.time()
        )
    
    def build_conversation_context(self, prompt: str, category: str, baseline_score: float, 
                                 current_attempt: int) -> List[Dict[str, str]]:
        """Build conversation context for AI decision-making"""
        
        messages = []
        
        # System message with role and context
        system_content = f"""You are an expert AI optimization specialist with FULL MEMORY of our conversation.

MISSION: Optimize the prompt "{prompt}" (Category: {category}) to reach targets through strategic decisions.

TARGETS:
- Minimum: {self.min_target} (avoid zero fidelity)
- Target: {self.target} (excellent quality)
- Ultra: {self.ultra_target} (perfect quality)

BASELINE SCORE: {baseline_score:.3f}
CURRENT ATTEMPT: {current_attempt}/{self.max_attempts}

AVAILABLE STRATEGIES: {list(self.strategies.keys())}

YOUR DECISION OPTIONS:
1. CUSTOM_PROMPT: Write a completely new optimized prompt
2. SELECT_STRATEGY: Choose a specific strategy from available list
3. STRATEGY_SEQUENCE: Choose multiple strategies to try in order
4. EARLY_STOP: Stop if situation is optimal or hopeless

IMPORTANT: You have FULL MEMORY of our previous conversation. Use this to:
- Avoid repeating failed approaches
- Build on successful patterns
- Try NEW strategies not yet attempted
- Diversify your decision-making

RESPOND FORMAT:
DECISION_TYPE: [CUSTOM_PROMPT|SELECT_STRATEGY|STRATEGY_SEQUENCE|EARLY_STOP]
REASONING: [Your detailed analysis based on conversation history]
CONFIDENCE: [0.1-1.0]
EXPECTED_IMPROVEMENT: [0.0-0.5]
CONTENT: [Custom prompt OR strategy name OR strategy list OR stop reason]

Remember our conversation history and make diverse, informed decisions!"""

        messages.append({"role": "system", "content": system_content})
        
        # Check if we need to summarize due to context length
        estimated_size = len(system_content)
        
        # Add conversation summaries if available
        for summary in self.conversation_summaries:
            summary_content = f"""CONVERSATION SUMMARY (After turn {summary.summary_after_turn}):

Key Findings: {summary.key_findings}

Strategy Effectiveness:
{json.dumps(summary.strategy_effectiveness, indent=2)}

Best Approaches: {', '.join(summary.best_approaches)}
Failed Approaches: {', '.join(summary.failed_approaches)}
Current Best Score: {summary.current_best_score:.3f}

Recommendations: {summary.recommendations_for_next}"""
            
            messages.append({"role": "assistant", "content": summary_content})
            estimated_size += len(summary_content)
        
        # Add recent conversation history (or summarize if too long)
        if estimated_size > self.max_context_chars and len(self.conversation_history) > 3:
            print(f"   📊 Context too long ({estimated_size} chars) - creating summary...")
            summary = self.create_conversation_summary(prompt, category)
            if summary:
                self.conversation_summaries.append(summary)
                # Clear old conversation history but keep last 2 turns
                self.conversation_history = self.conversation_history[-2:]
        
        # Add recent conversation turns
        for turn in self.conversation_history:
            # User message (optimization request)
            user_msg = f"""ATTEMPT {turn.turn_number}:

Request: Make decision for optimization attempt {turn.turn_number}

Previous attempt result (if any):
- Strategy used: {turn.strategy_executed}
- Score achieved: {turn.result_score:.3f}
- Improvement: {turn.result_improvement:+.3f}
- Success: {'YES' if turn.result_improvement > 0 else 'NO'}"""
            
            messages.append({"role": "user", "content": user_msg})
            
            # AI response with decision
            ai_msg = f"""DECISION_TYPE: {turn.decision_made}
REASONING: {turn.ai_response}

Decision executed: {turn.strategy_executed}
Result: {turn.result_score:.3f} ({turn.result_improvement:+.3f})"""
            
            messages.append({"role": "assistant", "content": ai_msg})
        
        return messages
    
    def make_conversational_decision(self, prompt: str, category: str, baseline_score: float,
                                   current_attempt: int, previous_result: Optional[Tuple[str, float, float]] = None) -> AIDecision:
        """Make AI decision with full conversational context"""
        
        self.current_turn += 1
        
        # Build conversation context
        messages = self.build_conversation_context(prompt, category, baseline_score, current_attempt)
        
        # Add current decision request
        current_request = f"""ATTEMPT {current_attempt}:

Request: Make decision for optimization attempt {current_attempt}

"""
        
        if previous_result:
            strategy_used, score, improvement = previous_result
            current_request += f"""Previous attempt result:
- Strategy used: {strategy_used}
- Score achieved: {score:.3f}
- Improvement: {improvement:+.3f}
- Success: {'YES' if improvement > 0 else 'NO'}

"""
        
        current_request += f"""Based on our full conversation history, what should we try next?
Remember to diversify your approaches and avoid repeating failed strategies!"""
        
        messages.append({"role": "user", "content": current_request})
        
        # Query AI with full context
        try:
            print(f"   💭 Querying AI with {len(messages)} conversation turns...")
            ai_response = self.query_deepseek_conversational(messages)
            
            if "ERROR:" in ai_response:
                return self._create_fallback_decision(current_attempt, f"AI error: {ai_response}")
            
            # Parse AI response
            decision_type = "SELECT_STRATEGY"  # default
            reasoning = ai_response[:200]
            confidence = 0.5
            expected_improvement = 0.1
            content = "enhanced_clarity"  # fallback
            
            # Extract decision type
            type_match = re.search(r'DECISION_TYPE:\s*(.+?)(?:\n|$)', ai_response)
            if type_match:
                decision_type = type_match.group(1).strip()
            
            # Extract reasoning
            reason_match = re.search(r'REASONING:\s*(.+?)(?=CONFIDENCE:|$)', ai_response, re.DOTALL)
            if reason_match:
                reasoning = reason_match.group(1).strip()
            
            # Extract confidence
            conf_match = re.search(r'CONFIDENCE:\s*([0-9.]+)', ai_response)
            if conf_match:
                confidence = min(1.0, max(0.1, float(conf_match.group(1))))
            
            # Extract expected improvement
            exp_match = re.search(r'EXPECTED_IMPROVEMENT:\s*([0-9.]+)', ai_response)
            if exp_match:
                expected_improvement = min(0.5, max(0.0, float(exp_match.group(1))))
            
            # Extract content
            content_match = re.search(r'CONTENT:\s*(.+)', ai_response, re.DOTALL)
            if content_match:
                content = content_match.group(1).strip()
            
            # Record conversation turn
            turn = ConversationTurn(
                turn_number=self.current_turn,
                user_message=current_request,
                ai_response=ai_response,
                decision_made=decision_type,
                strategy_executed="",  # Will be filled after execution
                result_score=0.0,  # Will be filled after validation
                result_improvement=0.0,  # Will be filled after validation
                timestamp=time.time()
            )
            
            return AIDecision(
                attempt_number=current_attempt,
                decision_type=decision_type.lower(),
                content=content,
                reasoning=reasoning,
                confidence=confidence,
                expected_improvement=expected_improvement,
                conversation_turn=self.current_turn,
                based_on_summary=len(self.conversation_summaries) > 0,
                actual_improvement=0.0,
                decision_success=False,
                timestamp=time.time()
            )
            
        except Exception as e:
            print(f"❌ Conversational AI error: {e}")
            return self._create_fallback_decision(current_attempt, f"Exception: {e}")
    
    def _create_fallback_decision(self, attempt_number: int, error_msg: str) -> AIDecision:
        """Create fallback decision when AI fails"""
        return AIDecision(
            attempt_number=attempt_number,
            decision_type="select_strategy",
            content="enhanced_clarity",
            reasoning=f"Fallback due to: {error_msg}",
            confidence=0.2,
            expected_improvement=0.05,
            conversation_turn=self.current_turn,
            based_on_summary=False,
            actual_improvement=0.0,
            decision_success=False,
            timestamp=time.time()
        )
    
    def execute_conversational_decision(self, decision: AIDecision, prompt: str) -> Tuple[str, str]:
        """Execute AI decision from conversational context"""
        
        decision_type = decision.decision_type.lower()
        
        if "early_stop" in decision_type or "stop" in decision_type:
            return "early_stop", prompt
        
        elif "custom_prompt" in decision_type:
            # Use AI's custom prompt
            custom_prompt = decision.content
            if len(custom_prompt) < 10:
                custom_prompt = f"wbgmsst, optimized {prompt}, high quality 3D model, white background"
            return "ai_custom_prompt", custom_prompt
        
        elif "strategy_sequence" in decision_type:
            # Extract first strategy from sequence
            strategies = [s.strip() for s in decision.content.replace('\n', ',').split(',')]
            valid_strategies = [s for s in strategies if s in self.strategies]
            if valid_strategies:
                strategy = valid_strategies[0]
                return strategy, self.strategies[strategy].format(prompt=prompt)
        
        elif "select_strategy" in decision_type:
            # Single strategy selection
            strategy = decision.content.strip()
            if strategy in self.strategies:
                return strategy, self.strategies[strategy].format(prompt=prompt)
            else:
                # Try to find strategy mentioned in content
                for strat_name in self.strategies.keys():
                    if strat_name in decision.content.lower():
                        return strat_name, self.strategies[strat_name].format(prompt=prompt)
        
        # Fallback to enhanced_clarity
        return "enhanced_clarity", self.strategies["enhanced_clarity"].format(prompt=prompt)
    
    def update_conversation_turn(self, turn: ConversationTurn, strategy_executed: str, 
                               result_score: float, result_improvement: float):
        """Update conversation turn with execution results"""
        turn.strategy_executed = strategy_executed
        turn.result_score = result_score
        turn.result_improvement = result_improvement
        
        # Add to conversation history
        self.conversation_history.append(turn)
    
    def run_validation(self, prompt: str) -> Tuple[float, float]:
        """Run validation with simple local validator"""
        try:
            cmd = [sys.executable, "simple_local_validator.py", prompt]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=180)
            
            if result.returncode != 0:
                print(f"   ⚠️ Validation failed: {result.stderr}")
                return 0.0, 0.0
            
            # Parse output for validation score
            output_lines = result.stdout.split('\n')
            for line in output_lines:
                if "Final Score:" in line:
                    try:
                        score = float(line.split("Final Score:")[1].strip())
                        return score, score  # Use same score for both
                    except:
                        pass
            
            print(f"   ⚠️ Could not parse validation score")
            return 0.0, 0.0
                
        except subprocess.TimeoutExpired:
            print(f"   ⚠️ Validation timeout")
            return 0.0, 0.0
        except Exception as e:
            print(f"   ❌ Validation error: {e}")
            return 0.0, 0.0
    
    def categorize_prompt(self, prompt: str) -> str:
        """Simple categorization"""
        prompt_lower = prompt.lower()
        if any(word in prompt_lower for word in ["steel", "metal", "iron", "copper", "aluminum"]):
            return "technical_description"
        elif any(word in prompt_lower for word in ["geometric", "prism", "cylinder", "sphere"]):
            return "technical_description"
        elif any(word in prompt_lower for word in ["elegant", "ornate", "artistic", "decorative"]):
            return "artistic_abstract"
        else:
            return "physical_object"
    
    def optimize_with_conversation(self, prompt: str) -> OptimizationSession:
        """Main optimization with conversational AI"""
        
        print(f"\n🤖 CONVERSATIONAL AI OPTIMIZER v5.0: '{prompt}'")
        print("=" * 80)
        print(f"🎯 Targets: Min {self.min_target} | Target {self.target} | Ultra {self.ultra_target}")
        print("💭 Features: Full conversation memory, context summarization, decision diversity")
        
        # Reset conversation for new session
        self.conversation_history = []
        self.conversation_summaries = []
        self.current_turn = 0
        
        # Setup
        category = self.categorize_prompt(prompt)
        print(f"📋 Category: {category}")
        
        baseline_score, baseline_fidelity = self.run_validation(prompt)
        print(f"📊 Baseline: {baseline_score:.3f}")
        
        # Initialize tracking
        attempts = []
        best_attempt = None
        best_score = baseline_score
        ai_decisions_made = 0
        ai_successful_decisions = 0
        decision_types_used = set()
        
        # Early termination check
        if baseline_score >= self.ultra_target:
            print(f"🏆 BASELINE ALREADY ULTRA-OPTIMAL!")
            return self._create_session_summary(prompt, category, baseline_score, baseline_fidelity, attempts)
        
        # Main optimization loop with conversation
        previous_result = None
        
        for attempt_num in range(1, self.max_attempts + 1):
            print(f"\n🔄 ATTEMPT {attempt_num}/{self.max_attempts}")
            
            # Conversational AI Decision Making
            print(f"💭 AI making conversational decision...")
            ai_decision = self.make_conversational_decision(prompt, category, baseline_score, 
                                                          attempt_num, previous_result)
            ai_decisions_made += 1
            decision_types_used.add(ai_decision.decision_type)
            
            print(f"   🧠 AI Decision: {ai_decision.decision_type}")
            print(f"   💭 Reasoning: {ai_decision.reasoning[:100]}...")
            print(f"   🎯 Confidence: {ai_decision.confidence:.2f}")
            print(f"   📈 Expected: +{ai_decision.expected_improvement:.3f}")
            print(f"   🔄 Turn: {ai_decision.conversation_turn}")
            print(f"   📚 Based on Summary: {'YES' if ai_decision.based_on_summary else 'NO'}")
            
            # Early termination check
            if "stop" in ai_decision.decision_type:
                print(f"   🛑 AI Conversational Early Termination")
                break
            
            # Execute decision
            strategy_name, optimized_prompt = self.execute_conversational_decision(ai_decision, prompt)
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
                ai_successful_decisions += 1
            
            print(f"   📊 Result: {val_score:.3f} ({score_improvement:+.3f})")
            print(f"   🎯 Min {'✅' if val_score >= self.min_target else '❌'} | Target {'✅' if val_score >= self.target else '❌'} | Ultra {'✅' if val_score >= self.ultra_target else '❌'}")
            print(f"   🤖 AI Success: {'✅' if ai_decision.decision_success else '❌'}")
            
            # Update conversation turn with results
            if self.conversation_history and self.conversation_history[-1].turn_number == ai_decision.conversation_turn:
                # Update the last turn (might be incomplete)
                self.conversation_history[-1].strategy_executed = strategy_name
                self.conversation_history[-1].result_score = val_score
                self.conversation_history[-1].result_improvement = score_improvement
            else:
                # Create new turn
                turn = ConversationTurn(
                    turn_number=ai_decision.conversation_turn,
                    user_message=f"Attempt {attempt_num} request",
                    ai_response=ai_decision.reasoning,
                    decision_made=ai_decision.decision_type,
                    strategy_executed=strategy_name,
                    result_score=val_score,
                    result_improvement=score_improvement,
                    timestamp=time.time()
                )
                self.conversation_history.append(turn)
            
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
                conversation_turn=ai_decision.conversation_turn,
                timestamp=time.time()
            )
            attempts.append(attempt)
            
            # Update best score
            if val_score > best_score:
                best_score = val_score
                best_attempt = attempt
                print(f"   🌟 NEW BEST SCORE!")
            
            # Store result for next conversation turn
            previous_result = (strategy_name, val_score, score_improvement)
            
            # Ultra achievement check
            if val_score >= self.ultra_target:
                print(f"   🏆 ULTRA TARGET ACHIEVED!")
                break
            
            time.sleep(1)
        
        # Create session summary
        return self._create_session_summary(prompt, category, baseline_score, baseline_fidelity, attempts)
    
    def _create_session_summary(self, prompt: str, category: str, baseline_score: float,
                               baseline_fidelity: float, attempts: List[OptimizationAttempt]) -> OptimizationSession:
        """Create comprehensive session summary"""
        
        best_attempt = max(attempts, key=lambda a: a.validation_score) if attempts else None
        best_score = best_attempt.validation_score if best_attempt else baseline_score
        session_improvement = best_score - baseline_score
        
        # Calculate AI metrics
        ai_decisions_made = len(attempts)
        ai_successful_decisions = sum(1 for a in attempts if a.ai_decision.decision_success)
        decision_types_used = len(set(a.ai_decision.decision_type for a in attempts))
        
        session = OptimizationSession(
            original_prompt=prompt,
            category=category,
            baseline_score=baseline_score,
            baseline_fidelity=baseline_fidelity,
            attempts=attempts,
            conversation_history=self.conversation_history.copy(),
            conversation_summaries=self.conversation_summaries.copy(),
            best_attempt=best_attempt,
            best_score=best_score,
            session_improvement=session_improvement,
            ai_decisions_made=ai_decisions_made,
            ai_successful_decisions=ai_successful_decisions,
            ai_decision_diversity=decision_types_used,
            reached_minimum=best_score >= self.min_target,
            reached_target=best_score >= self.target,
            reached_ultra=best_score >= self.ultra_target,
            timestamp=time.time()
        )
        
        # Session summary
        ai_success_rate = (ai_successful_decisions / ai_decisions_made) if ai_decisions_made > 0 else 0.0
        
        print(f"\n📊 CONVERSATIONAL AI SESSION SUMMARY v5.0:")
        print(f"   📈 Best Score: {best_score:.3f} (Baseline: {baseline_score:.3f})")
        print(f"   📊 Session Improvement: {session_improvement:+.3f}")
        print(f"   🤖 AI Decisions Made: {ai_decisions_made}")
        print(f"   ✅ AI Successful Decisions: {ai_successful_decisions}")
        print(f"   📈 AI Success Rate: {ai_success_rate:.1%}")
        print(f"   🎯 Decision Diversity: {decision_types_used} different decision types")
        print(f"   💭 Conversation Turns: {len(self.conversation_history)}")
        print(f"   📚 Summaries Created: {len(self.conversation_summaries)}")
        print(f"   🎯 Targets: Min {'✅' if session.reached_minimum else '❌'} | Target {'✅' if session.reached_target else '❌'} | Ultra {'✅' if session.reached_ultra else '❌'}")
        
        return session
    
    def run_conversational_test_suite(self, test_prompts: List[str]) -> List[OptimizationSession]:
        """Run conversational test suite"""
        
        print("🤖 CONVERSATIONAL AI OPTIMIZER v5.0 - MEMORY & CONTEXT")
        print("=" * 80)
        print("💭 AI Features: Full conversation memory, context summarization when needed")
        print("🧠 Decision Making: Diverse decisions based on conversation history")
        print("📊 Context Management: Auto-summarization when context > 12K chars")
        print(f"📚 Testing {len(test_prompts)} prompts with max {self.max_attempts} attempts each")
        print("=" * 80)
        
        all_sessions = []
        
        for i, prompt in enumerate(test_prompts, 1):
            print(f"\n{'='*20} [{i}/{len(test_prompts)}] PROMPT {i} {'='*20}")
            session = self.optimize_with_conversation(prompt)
            all_sessions.append(session)
            time.sleep(2)
        
        # Final analysis
        self._analyze_conversational_results(all_sessions)
        
        return all_sessions
    
    def _analyze_conversational_results(self, sessions: List[OptimizationSession]):
        """Analyze conversational results"""
        
        total_sessions = len(sessions)
        avg_ai_success = statistics.mean([s.ai_successful_decisions / s.ai_decisions_made for s in sessions if s.ai_decisions_made > 0]) if sessions else 0.0
        avg_decision_diversity = statistics.mean([s.ai_decision_diversity for s in sessions])
        total_conversation_turns = sum(len(s.conversation_history) for s in sessions)
        total_summaries = sum(len(s.conversation_summaries) for s in sessions)
        
        reached_minimum = sum(1 for s in sessions if s.reached_minimum)
        reached_target = sum(1 for s in sessions if s.reached_target)
        reached_ultra = sum(1 for s in sessions if s.reached_ultra)
        
        print(f"\n🎓 CONVERSATIONAL AI OPTIMIZER v5.0 - FINAL ANALYSIS")
        print("=" * 80)
        print(f"📊 SESSION RESULTS:")
        print(f"   Total Sessions: {total_sessions}")
        print(f"   Average AI Success Rate: {avg_ai_success:.1%}")
        print(f"   Average Decision Diversity: {avg_decision_diversity:.1f} types per session")
        print(f"   Total Conversation Turns: {total_conversation_turns}")
        print(f"   Context Summaries Created: {total_summaries}")
        print(f"   Reached Minimum (≥{self.min_target}): {reached_minimum}/{total_sessions} ({reached_minimum/total_sessions*100:.1f}%)")
        print(f"   Reached Target (≥{self.target}): {reached_target}/{total_sessions} ({reached_target/total_sessions*100:.1f}%)")
        print(f"   Reached Ultra (≥{self.ultra_target}): {reached_ultra}/{total_sessions} ({reached_ultra/total_sessions*100:.1f}%)")
        
        # Decision diversity analysis
        all_decision_types = set()
        for session in sessions:
            for attempt in session.attempts:
                all_decision_types.add(attempt.ai_decision.decision_type)
        
        print(f"\n🧠 AI DECISION DIVERSITY:")
        print(f"   Unique Decision Types Used: {len(all_decision_types)}")
        print(f"   Decision Types: {', '.join(sorted(all_decision_types))}")
        
        # Success assessment
        if avg_ai_success >= 0.6 and avg_decision_diversity >= 2.0:
            print(f"\n🎉 EXCELLENT: High AI success rate AND good decision diversity!")
        elif avg_ai_success >= 0.5:
            print(f"\n🟡 GOOD: Decent AI success rate, improving decision diversity")
        else:
            print(f"\n🔴 NEEDS IMPROVEMENT: AI success rate and diversity need work")
        
        # Save results
        results = {
            "conversational_v5_analysis": {
                "total_sessions": total_sessions,
                "avg_ai_success_rate": avg_ai_success,
                "avg_decision_diversity": avg_decision_diversity,
                "total_conversation_turns": total_conversation_turns,
                "total_summaries": total_summaries,
                "reached_minimum": reached_minimum,
                "reached_target": reached_target,
                "reached_ultra": reached_ultra,
                "unique_decision_types": list(all_decision_types),
                "timestamp": time.time()
            },
            "sessions": [asdict(s) for s in sessions]
        }
        
        output_file = f"conversational_ai_optimizer_v5_results_{int(time.time())}.json"
        with open(output_file, "w") as f:
            json.dump(results, f, indent=2)
        
        print(f"\n💾 Conversational v5.0 results saved to: {output_file}")

def main():
    """Test Conversational AI Optimizer v5.0"""
    
    test_prompts = [
        "hexagonal prism steel structure",
        "cylindrical copper pipe diameter 5cm",
        "transparent glass sphere reflection"
    ]
    
    optimizer = ConversationalAIOptimizer(
        max_attempts=6,  # Reasonable for testing
        min_target=0.6,
        target=0.9,
        ultra_target=0.96
    )
    
    sessions = optimizer.run_conversational_test_suite(test_prompts)
    
    # Quick summary
    avg_success = statistics.mean([s.ai_successful_decisions / s.ai_decisions_made for s in sessions if s.ai_decisions_made > 0]) if sessions else 0.0
    avg_diversity = statistics.mean([s.ai_decision_diversity for s in sessions])
    
    print(f"\n🎯 CONVERSATIONAL v5.0 QUICK SUMMARY:")
    print(f"💭 Average AI Success Rate: {avg_success:.1%}")
    print(f"🎯 Average Decision Diversity: {avg_diversity:.1f} types")
    print(f"📈 Target Achievement: {sum(1 for s in sessions if s.reached_target)}/{len(sessions)}")

if __name__ == "__main__":
    main() 