#!/usr/bin/env python3
"""
Adaptive Learning Optimizer v6.2 - Fixed AI Parsing & Communication
Purpose: Fixed AI communication with better parsing and clearer instructions

Key Improvements in v6.2:
- Much clearer AI instructions with examples
- Robust parsing that handles verbose AI responses
- Strong anti-repetition enforcement
- Better strategy extraction from AI responses
- Simpler, more focused prompts for AI
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

# Same dataclasses as v6.1 (keeping them concise)
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
    timestamp: float = 0.0

class AdaptiveLearningOptimizerV6_2:
    """v6.2 with fixed AI communication and parsing"""

    def __init__(self, max_attempts: int = 6, min_target: float = 0.6,
                 target: float = 0.9, ultra_target: float = 0.96):
        
        self.ollama_url = "http://localhost:11434"
        self.model_name = "deepseek-r1:1.5b"
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
            "raw": "{prompt}",
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

    def query_ai_simple(self, user_message: str, timeout: int = 45) -> str:
        """Simplified AI query focused on getting clear responses"""
        data = {
            "model": self.model_name,
            "messages": [{"role": "user", "content": user_message}],
            "stream": False,
            "options": {
                "temperature": 0.8,  # Higher creativity
                "top_p": 0.9,
                "num_predict": 200,  # Shorter responses
                "stop": ["<think>", "</think>"]  # Stop verbose thinking
            }
        }
        
        try:
            response = requests.post(f"{self.ollama_url}/api/chat", json=data, timeout=timeout)
            response.raise_for_status()
            content = response.json()["message"]["content"]
            
            # Clean up response
            content = content.replace("<think>", "").replace("</think>", "")
            content = re.sub(r'\*\*([^*]+)\*\*', r'\1', content)  # Remove markdown bold
            
            return content.strip()
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
        """Simple categorization"""
        prompt_lower = prompt.lower()
        if any(word in prompt_lower for word in ["steel", "metal", "geometric", "prism", "cylinder"]):
            return "technical"
        elif any(word in prompt_lower for word in ["elegant", "artistic", "ornate"]):
            return "artistic"
        else:
            return "physical"

    def select_persona_and_strategies(self, attempt_num: int, baseline_score: float, attempts: List[OptimizationAttempt]) -> Tuple[str, List[str]]:
        """Select persona and get available strategies with anti-repetition"""
        
        # Persona selection
        if baseline_score < 0.4:
            persona = "Rescue Specialist"
        elif attempt_num > 3 and all(a.score_improvement <= 0.01 for a in attempts[-2:]):
            persona = "Creative Breaker"
        else:
            persona = "Strategic Optimizer"
        
        # Get unused strategies
        recently_used = [a.strategy_name for a in attempts[-self.max_memory:]]
        available_strategies = [s for s in self.strategies.keys() if s not in recently_used]
        
        # If too few available, reset some
        if len(available_strategies) < 3:
            available_strategies.extend(list(self.strategies.keys())[:3])
        
        return persona, available_strategies[:8]  # Limit to 8 for clarity

    def make_ai_decision_clear(self, prompt: str, category: str, baseline_score: float, 
                              attempt_num: int, attempts: List[OptimizationAttempt]) -> AIDecision:
        """Clear, focused AI decision making"""
        
        persona, available_strategies = self.select_persona_and_strategies(attempt_num, baseline_score, attempts)
        
        # Build clear context
        recent_results = []
        for attempt in attempts[-3:]:
            recent_results.append(f"Attempt {attempt.attempt_number}: {attempt.strategy_name} -> {attempt.validation_score:.3f}")
        
        # Create focused prompt
        user_message = f"""You are a {persona} optimizing 3D model generation.

SITUATION:
- Prompt: "{prompt}" (Category: {category})
- Baseline: {baseline_score:.3f}
- Attempt: {attempt_num}/{self.max_attempts}
- Targets: Min {self.min_target} | Excellent {self.target} | Ultra {self.ultra_target}

RECENT RESULTS:
{chr(10).join(recent_results) if recent_results else "None yet"}

AVAILABLE STRATEGIES:
{', '.join(available_strategies)}

TASK: Choose your optimization approach.

OPTIONS:
A) WRITE_CUSTOM: Create a completely new optimized prompt
B) USE_STRATEGY: Pick one strategy from the available list
C) EARLY_STOP: Stop if situation is hopeless or already optimal

RESPOND FORMAT (be brief and direct):
CHOICE: [A/B/C]
STRATEGY_OR_PROMPT: [strategy name OR custom prompt OR stop reason]
REASONING: [brief explanation]
CONFIDENCE: [0.1-1.0]

Example:
CHOICE: B
STRATEGY_OR_PROMPT: material_focus
REASONING: Steel structures need material emphasis
CONFIDENCE: 0.8

Your response:"""

        print(f"🤖 AI Persona: {persona}")
        print(f"   📋 Available Strategies: {len(available_strategies)}")
        
        ai_response = self.query_ai_simple(user_message)
        
        if "ERROR:" in ai_response:
            return self.create_fallback_decision(attempt_num, persona, ai_response, available_strategies)
        
        print(f"   🤖 AI Response: {ai_response[:100]}...")
        
        # Parse the clearer response format
        return self.parse_clear_response(ai_response, attempt_num, persona, available_strategies)

    def parse_clear_response(self, response: str, attempt_num: int, persona: str, available_strategies: List[str]) -> AIDecision:
        """Parse the clearer AI response format"""
        
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
            
            # Validate strategy choice
            if decision_type == "USE_STRATEGY":
                # Check if it's a valid strategy
                if content not in self.strategies:
                    # Try to find a partial match
                    for strategy in available_strategies:
                        if strategy in content.lower() or content.lower() in strategy:
                            content = strategy
                            break
                    else:
                        # Use first available strategy as fallback
                        content = available_strategies[0] if available_strategies else "enhanced_clarity"
            
        except Exception as e:
            print(f"   ⚠️ Parsing error: {e}, using semantic parsing")
            
            # Semantic fallback parsing
            response_lower = response.lower()
            
            # Look for strategy names mentioned
            for strategy in available_strategies:
                if strategy in response_lower:
                    decision_type = "USE_STRATEGY"
                    content = strategy
                    break
            
            # Look for custom prompt indicators
            if any(word in response_lower for word in ["custom", "write", "create", "new prompt"]):
                decision_type = "WRITE_CUSTOM"
                # Try to extract a custom prompt
                lines = response.split('\n')
                for line in lines:
                    if any(word in line.lower() for word in ["wbgmsst", "3d model", "white background"]):
                        content = line.strip()
                        break
                if content == available_strategies[0]:  # Still default
                    content = f"wbgmsst, optimized {reasoning.split()[0] if reasoning else 'model'}, high quality 3D, white background"
            
            # Look for stop indicators
            elif any(word in response_lower for word in ["stop", "hopeless", "optimal", "terminate"]):
                decision_type = "EARLY_STOP"
                content = "AI decided to stop"
        
        return AIDecision(
            attempt_number=attempt_num,
            persona_used=persona,
            decision_type=decision_type,
            content=content,
            reasoning=reasoning,
            confidence=confidence,
            expected_improvement=confidence * 0.2,  # Simple estimation
            conversation_turn=attempt_num,
            based_on_summary=False,
            timestamp=time.time()
        )

    def create_fallback_decision(self, attempt_num: int, persona: str, error_msg: str, available_strategies: List[str]) -> AIDecision:
        """Create fallback decision with available strategies"""
        
        # Use round-robin through available strategies
        strategy = available_strategies[attempt_num % len(available_strategies)] if available_strategies else "enhanced_clarity"
        
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
        """Execute AI decision with validation"""
        
        if decision.decision_type == "EARLY_STOP":
            return "early_stop", prompt
        
        elif decision.decision_type == "WRITE_CUSTOM":
            custom_prompt = decision.content
            if len(custom_prompt) > 10 and any(word in custom_prompt.lower() for word in ["3d", "model", "background"]):
                return "ai_custom_prompt", custom_prompt
            else:
                # Generate better custom prompt
                custom_prompt = f"wbgmsst, detailed 3D {prompt}, high quality render, white background"
                return "ai_custom_prompt", custom_prompt
        
        elif decision.decision_type == "USE_STRATEGY":
            strategy = decision.content
            if strategy in self.strategies:
                return strategy, self.strategies[strategy].format(prompt=prompt)
            else:
                # Fallback to available strategy
                return "enhanced_clarity", self.strategies["enhanced_clarity"].format(prompt=prompt)
        
        # Final fallback
        return "enhanced_clarity", self.strategies["enhanced_clarity"].format(prompt=prompt)

    def optimize_prompt(self, prompt: str) -> OptimizationSession:
        """Main optimization with clearer AI communication"""
        
        print(f"\n🚀 ADAPTIVE OPTIMIZER v6.2: '{prompt}'")
        print("=" * 70)
        print(f"🎯 Targets: Min {self.min_target} | Target {self.target} | Ultra {self.ultra_target}")
        print("🧠 Features: Clear AI communication, robust parsing, anti-repetition")
        
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
        
        # Main optimization loop
        for i in range(1, self.max_attempts + 1):
            print(f"\n🔄 ATTEMPT {i}/{self.max_attempts}")
            
            # AI Decision
            print(f"🤖 AI making clear decision...")
            ai_decision = self.make_ai_decision_clear(prompt, category, baseline_score, i, attempts)
            
            print(f"   🧠 Decision: {ai_decision.decision_type}")
            print(f"   💭 Content: {ai_decision.content}")
            print(f"   🎯 Confidence: {ai_decision.confidence:.2f}")
            
            # Execute
            strategy_name, optimized_prompt = self.execute_ai_decision(ai_decision, prompt)
            
            if strategy_name == "early_stop":
                print(f"   🛑 AI Early Stop")
                break
            
            # Track usage for anti-repetition
            self.used_strategies.append(strategy_name)
            self.used_decision_types.append(ai_decision.decision_type)
            
            is_ai_generated = strategy_name == "ai_custom_prompt"
            
            print(f"   🔧 Executing: {strategy_name}")
            print(f"   ✨ Prompt: '{optimized_prompt[:60]}{'...' if len(optimized_prompt) > 60 else ''}'")
            
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
            
            # Ultra check
            if val_score >= self.ultra_target:
                print(f"   🏆 ULTRA TARGET ACHIEVED!")
                break
            
            time.sleep(1)
        
        return self.create_session_summary(prompt, category, baseline_score, attempts, best_attempt)

    def create_session_summary(self, prompt: str, category: str, baseline_score: float, 
                              attempts: List[OptimizationAttempt], best_attempt: OptimizationAttempt = None) -> OptimizationSession:
        """Create session summary"""
        
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
            timestamp=time.time()
        )
        
        # Print summary
        print(f"\n📊 SESSION SUMMARY v6.2:")
        print(f"   📈 Best Score: {best_score:.3f} (Baseline: {baseline_score:.3f})")
        print(f"   📊 Session Improvement: {session_improvement:+.3f}")
        print(f"   🤖 AI Decisions: {ai_decisions_made}")
        print(f"   ✅ AI Success Rate: {ai_contribution_rate:.1%}")
        print(f"   🎯 Decision Diversity: {ai_decision_diversity} types")
        print(f"   🎯 Targets: Min {'✅' if session.reached_minimum_threshold else '❌'} | Target {'✅' if session.reached_target_threshold else '❌'} | Ultra {'✅' if session.reached_ultra_threshold else '❌'}")
        
        # Show strategies used
        strategies_used = [a.strategy_name for a in attempts]
        unique_strategies = list(set(strategies_used))
        print(f"   🔧 Strategies Used: {', '.join(unique_strategies)}")
        
        return session

def main():
    """Test v6.2 with clearer AI communication"""
    
    test_prompts = [
        "hexagonal prism steel structure",
        "elegant silk fabric draping",
        "transparent glass sphere reflection"
    ]
    
    print("🚀 ADAPTIVE LEARNING OPTIMIZER v6.2 - CLEAR AI COMMUNICATION")
    print("=" * 70)
    print("🔧 Features: Clear AI instructions, robust parsing, strong anti-repetition")
    print("🧠 AI: Simplified prompts with structured responses")
    print("📊 Tracking: Strategy diversity and success metrics")
    print("=" * 70)
    
    optimizer = AdaptiveLearningOptimizerV6_2(
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
    print(f"\n🎓 FINAL ANALYSIS - v6.2 CLEAR COMMUNICATION")
    print("=" * 70)
    
    total_sessions = len(all_sessions)
    avg_ai_success = statistics.mean([s.ai_contribution_rate for s in all_sessions]) if all_sessions else 0.0
    avg_diversity = statistics.mean([s.ai_decision_diversity for s in all_sessions]) if all_sessions else 0.0
    reached_target = sum(1 for s in all_sessions if s.reached_target_threshold)
    
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
    print(f"   📈 Total Unique Strategies Used: {len(all_strategies_used)}")
    print(f"   🔧 Strategies: {', '.join(sorted(all_strategies_used))}")
    
    if avg_ai_success >= 0.5 and len(all_strategies_used) >= 5:
        print(f"\n🎉 SUCCESS: Good AI performance AND strategy diversity!")
    elif len(all_strategies_used) >= 5:
        print(f"\n🟡 GOOD: Excellent strategy diversity, improving AI success")
    else:
        print(f"\n🔴 IMPROVEMENT NEEDED: Low strategy diversity")

if __name__ == "__main__":
    main() 