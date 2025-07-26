#!/usr/bin/env python3
"""
Smart Prompt Optimizer V4 - CONVERSATIONAL RL AGENT
==================================================
🗣️ LLM as true conversational RL agent with self-strategy selection
🧠 Multi-turn reasoning with structured and unstructured parsing
⚡ Faster learning through principles instead of massive context
🎯 Agent chooses its own exploration/exploitation strategies
🔄 Continuous dialogue-based learning and self-correction

Revolutionary Improvements:
1. LLM selects its own strategies through reasoning
2. Conversational learning with structured turn-taking
3. Principle extraction instead of massive memory dumps
4. Robust structured+unstructured parsing
5. True autonomous decision making
6. Faster inference through compressed knowledge
"""

import json
import requests
import time
import sys
import random
import re

from typing import Dict, List, Optional, Tuple, Any, Union
from pathlib import Path
from dataclasses import dataclass, asdict
import statistics
import subprocess
from datetime import datetime

@dataclass
class LearnedPrinciple:
    """A learned principle extracted from experience"""
    principle_id: str
    description: str
    confidence: float
    success_rate: float
    example_transformations: List[Tuple[str, str, float]]  # (original, optimized, score)
    discovery_timestamp: float
    usage_count: int
    last_successful_use: float

@dataclass
class ConversationTurn:
    """A single turn in the agent's reasoning conversation"""
    turn_type: str  # 'strategy_selection', 'optimization', 'reflection', 'principle_extraction'
    agent_reasoning: str
    structured_output: Dict[str, Any]
    timestamp: float

@dataclass
class OptimizationSession:
    """Complete optimization session with conversational turns"""
    session_id: str
    original_prompt: str
    conversation_turns: List[ConversationTurn]
    final_optimized_prompt: str
    final_confidence: float
    validation_score: Optional[float]
    extracted_principles: List[str]  # IDs of principles learned
    success: bool
    total_time: float

class ConversationalRLAgent:
    """Conversational RL agent that reasons through optimization strategies"""
    
    def __init__(self, ollama_url: str = "http://localhost:11434",
                 memory_file: str = "conversational_rl_memory.json"):
        self.ollama_url = ollama_url
        self.model = "llama3.2:3b"
        self.memory_file = Path(memory_file)
        
        # Learning state
        self.learned_principles: Dict[str, LearnedPrinciple] = {}
        self.optimization_sessions: List[OptimizationSession] = []
        self.total_optimizations = 0
        self.successful_optimizations = 0
        
        # Conversational parameters
        self.max_conversation_turns = 5
        self.principle_confidence_threshold = 0.8
        self.principle_usage_threshold = 3
        
        self._load_memory()
        
        print("🗣️ CONVERSATIONAL RL AGENT INITIALIZED")
        print(f"   Learned principles: {len(self.learned_principles)}")
        print(f"   Past sessions: {len(self.optimization_sessions)}")
        print(f"   Success rate: {self._get_success_rate():.1%}")
    
    def _load_memory(self):
        """Load conversational memory"""
        if self.memory_file.exists():
            try:
                with open(self.memory_file, 'r') as f:
                    data = json.load(f)
                
                # Load learned principles
                principles_data = data.get('learned_principles', {})
                self.learned_principles = {
                    pid: LearnedPrinciple(**principle)
                    for pid, principle in principles_data.items()
                }
                
                # Load recent sessions (keep last 100)
                sessions_data = data.get('optimization_sessions', [])
                self.optimization_sessions = [
                    OptimizationSession(**session) for session in sessions_data[-100:]
                ]
                
                self.total_optimizations = data.get('total_optimizations', 0)
                self.successful_optimizations = data.get('successful_optimizations', 0)
                
                print(f"📚 Loaded {len(self.learned_principles)} principles and {len(self.optimization_sessions)} sessions")
                
            except Exception as e:
                print(f"⚠️ Failed to load memory: {e}")
                self._initialize_fresh()
        else:
            self._initialize_fresh()
    
    def _initialize_fresh(self):
        """Initialize fresh agent with no memory"""
        self.learned_principles = {}
        self.optimization_sessions = []
        self.total_optimizations = 0
        self.successful_optimizations = 0
        print("📄 Starting with fresh conversational memory")
    
    def _save_memory(self):
        """Save conversational memory"""
        try:
            data = {
                'learned_principles': {
                    pid: asdict(principle) for pid, principle in self.learned_principles.items()
                },
                'optimization_sessions': [
                    asdict(session) for session in self.optimization_sessions[-100:]
                ],
                'total_optimizations': self.total_optimizations,
                'successful_optimizations': self.successful_optimizations,
                'last_updated': time.time()
            }
            
            with open(self.memory_file, 'w') as f:
                json.dump(data, f, indent=2)
                
        except Exception as e:
            print(f"⚠️ Failed to save memory: {e}")
    
    def _get_success_rate(self) -> float:
        """Get current success rate"""
        if self.total_optimizations == 0:
            return 0.0
        return self.successful_optimizations / self.total_optimizations
    
    def optimize(self, prompt: str, use_validation: bool = False) -> Dict[str, Any]:
        """Main optimization through conversational reasoning"""
        session_id = f"session_{int(time.time())}_{random.randint(1000, 9999)}"
        start_time = time.time()
        
        print(f"\n🗣️ CONVERSATIONAL RL AGENT OPTIMIZING: '{prompt}'")
        print(f"   Session ID: {session_id}")
        
        # Initialize conversation session
        conversation_turns = []
        session = OptimizationSession(
            session_id=session_id,
            original_prompt=prompt,
            conversation_turns=conversation_turns,
            final_optimized_prompt="",
            final_confidence=0.0,
            validation_score=None,
            extracted_principles=[],
            success=False,
            total_time=0.0
        )
        
        try:
            # Turn 1: Strategy Selection & Analysis
            strategy_turn = self._conversation_turn_strategy_selection(prompt)
            conversation_turns.append(strategy_turn)
            
            # Turn 2: Optimization Execution
            optimization_turn = self._conversation_turn_optimization(
                prompt, strategy_turn.structured_output
            )
            conversation_turns.append(optimization_turn)
            
            # Extract results
            optimized_prompt = optimization_turn.structured_output.get('optimized_prompt', f"wbgmsst, {prompt}, white background")
            confidence = optimization_turn.structured_output.get('confidence', 0.5)
            
            # Validation if requested
            validation_score = None
            if use_validation:
                validation_score = self._validate_prompt(optimized_prompt)
            
            # Turn 3: Reflection & Learning
            reflection_turn = self._conversation_turn_reflection(
                prompt, optimized_prompt, confidence, validation_score
            )
            conversation_turns.append(reflection_turn)
            
            # Turn 4: Principle Extraction (if successful)
            success = self._evaluate_success(confidence, validation_score)
            if success:
                principle_turn = self._conversation_turn_principle_extraction(
                    prompt, optimized_prompt, validation_score or confidence
                )
                conversation_turns.append(principle_turn)
                
                # Add new principles
                new_principles = principle_turn.structured_output.get('principles', [])
                for principle_data in new_principles:
                    self._add_learned_principle(principle_data, prompt, optimized_prompt, validation_score or confidence)
            
            # Finalize session
            session.final_optimized_prompt = optimized_prompt
            session.final_confidence = confidence
            session.validation_score = validation_score
            session.success = success
            session.total_time = time.time() - start_time
            
            # Update global stats
            self.total_optimizations += 1
            if success:
                self.successful_optimizations += 1
            
            # Save session and memory
            self.optimization_sessions.append(session)
            self._save_memory()
            
            # Prepare result
            result = {
                'session_id': session_id,
                'original_prompt': prompt,
                'optimized_prompt': optimized_prompt,
                'confidence': confidence,
                'validation_score': validation_score,
                'success': success,
                'processing_time': session.total_time,
                'conversation_turns': len(conversation_turns),
                'new_principles_learned': len(session.extracted_principles),
                'total_principles': len(self.learned_principles),
                'overall_success_rate': self._get_success_rate()
            }
            
            print(f"✅ CONVERSATIONAL OPTIMIZATION COMPLETE")
            print(f"   Result: {optimized_prompt}")
            print(f"   Confidence: {confidence:.1%}")
            print(f"   Turns: {len(conversation_turns)}")
            print(f"   Success: {success}")
            if validation_score:
                print(f"   Validation: {validation_score:.3f}")
            print(f"   Time: {session.total_time:.2f}s")
            
            return result
            
        except Exception as e:
            print(f"❌ Conversational optimization failed: {e}")
            import traceback
            traceback.print_exc()
            return self._fallback_result(prompt, session_id, time.time() - start_time)
    
    def _conversation_turn_strategy_selection(self, prompt: str) -> ConversationTurn:
        """Turn 1: Agent selects strategy through reasoning"""
        
        print("   🎯 Turn 1: Strategy Selection & Analysis")
        
        # Build context with learned principles
        principles_context = self._build_principles_context()
        recent_context = self._build_recent_success_context()
        
        system_prompt = f"""You are an expert prompt optimization RL agent. You reason through strategy selection autonomously.

{principles_context}

{recent_context}

TASK: Analyze the prompt "{prompt}" and select your optimization strategy.

You must reason through:
1. What type of object/concept this is
2. What optimization approaches might work based on your learned principles
3. Whether to EXPLORE (try new approaches) or EXPLOIT (use proven methods)
4. Which specific strategy to use and why

RESPONSE FORMAT (be precise with JSON structure):
REASONING: [Your detailed reasoning process]

STRATEGY_DECISION: {{
  "selected_strategy": "[strategy name]",
  "exploration_type": "[explore/exploit]",
  "confidence_in_strategy": [0.0-1.0],
  "reasoning_summary": "[brief reason for this choice]",
  "applicable_principles": ["principle_id1", "principle_id2"],
  "expected_improvements": ["improvement1", "improvement2"]
}}"""

        try:
            response = self._query_llama(system_prompt)
            structured_output = self._parse_strategy_response(response)
            
            return ConversationTurn(
                turn_type="strategy_selection",
                agent_reasoning=response,
                structured_output=structured_output,
                timestamp=time.time()
            )
            
        except Exception as e:
            print(f"   ⚠️ Strategy selection failed: {e}")
            return ConversationTurn(
                turn_type="strategy_selection",
                agent_reasoning=f"Fallback due to error: {e}",
                structured_output={
                    "selected_strategy": "conservative_enhancement",
                    "exploration_type": "exploit",
                    "confidence_in_strategy": 0.5,
                    "reasoning_summary": "Fallback strategy",
                    "applicable_principles": [],
                    "expected_improvements": ["basic enhancement"]
                },
                timestamp=time.time()
            )
    
    def _conversation_turn_optimization(self, prompt: str, strategy_decision: Dict[str, Any]) -> ConversationTurn:
        """Turn 2: Execute optimization based on strategy"""
        
        print("   🔧 Turn 2: Optimization Execution")
        
        strategy = strategy_decision.get("selected_strategy", "conservative_enhancement")
        exploration_type = strategy_decision.get("exploration_type", "exploit")
        applicable_principles = strategy_decision.get("applicable_principles", [])
        
        # Build detailed context for this strategy
        strategy_context = self._build_strategy_context(strategy, applicable_principles)
        
        system_prompt = f"""You are executing optimization strategy: {strategy} ({exploration_type} mode)

ORIGINAL PROMPT: "{prompt}"
STRATEGY: {strategy}
EXPLORATION TYPE: {exploration_type}

{strategy_context}

TASK: Create an optimized prompt following your selected strategy.

Rules:
- Start with "wbgmsst," and end with ", white background"
- Apply the principles you identified as applicable
- Be {exploration_type}ive in your approach

RESPONSE FORMAT:
OPTIMIZATION_REASONING: [Explain your optimization approach step by step]

OPTIMIZATION_RESULT: {{
  "optimized_prompt": "[full optimized prompt]",
  "confidence": [0.0-1.0],
  "key_improvements": ["improvement1", "improvement2"],
  "principles_applied": ["principle_id1", "principle_id2"],
  "risk_assessment": "[low/medium/high]"
}}"""

        try:
            response = self._query_llama(system_prompt)
            structured_output = self._parse_optimization_response(response, prompt)
            
            return ConversationTurn(
                turn_type="optimization",
                agent_reasoning=response,
                structured_output=structured_output,
                timestamp=time.time()
            )
            
        except Exception as e:
            print(f"   ⚠️ Optimization failed: {e}")
            return ConversationTurn(
                turn_type="optimization",
                agent_reasoning=f"Fallback due to error: {e}",
                structured_output={
                    "optimized_prompt": f"wbgmsst, {prompt}, white background",
                    "confidence": 0.5,
                    "key_improvements": ["basic formatting"],
                    "principles_applied": [],
                    "risk_assessment": "low"
                },
                timestamp=time.time()
            )
    
    def _conversation_turn_reflection(self, original: str, optimized: str, 
                                    confidence: float, validation_score: Optional[float]) -> ConversationTurn:
        """Turn 3: Reflect on results and learn"""
        
        print("   🤔 Turn 3: Reflection & Learning")
        
        validation_text = f"{validation_score:.3f}" if validation_score is not None else "Not available"
        
        system_prompt = f"""You are reflecting on your optimization performance to learn and improve.

ORIGINAL: "{original}"
OPTIMIZED: "{optimized}"
YOUR CONFIDENCE: {confidence:.2f}
ACTUAL VALIDATION: {validation_text}

TASK: Reflect on this optimization attempt and identify what you learned.

RESPONSE FORMAT:
REFLECTION: [Your detailed reflection on performance]

REFLECTION_ANALYSIS: {{
  "performance_assessment": "[excellent/good/moderate/poor]",
  "confidence_accuracy": "[accurate/overconfident/underconfident]",
  "what_worked_well": ["aspect1", "aspect2"],
  "what_could_improve": ["aspect1", "aspect2"],
  "lessons_learned": ["lesson1", "lesson2"],
  "strategy_effectiveness": "[very_effective/effective/moderate/ineffective]"
}}"""

        try:
            response = self._query_llama(system_prompt)
            structured_output = self._parse_reflection_response(response)
            
            return ConversationTurn(
                turn_type="reflection",
                agent_reasoning=response,
                structured_output=structured_output,
                timestamp=time.time()
            )
            
        except Exception as e:
            print(f"   ⚠️ Reflection failed: {e}")
            return ConversationTurn(
                turn_type="reflection",
                agent_reasoning=f"Basic reflection: optimization completed",
                structured_output={
                    "performance_assessment": "moderate",
                    "confidence_accuracy": "accurate",
                    "what_worked_well": ["basic optimization"],
                    "what_could_improve": ["unknown"],
                    "lessons_learned": ["need more data"],
                    "strategy_effectiveness": "moderate"
                },
                timestamp=time.time()
            )
    
    def _conversation_turn_principle_extraction(self, original: str, optimized: str, score: float) -> ConversationTurn:
        """Turn 4: Extract new principles from successful optimizations"""
        
        print("   🎓 Turn 4: Principle Extraction")
        
        system_prompt = f"""You successfully optimized a prompt! Extract reusable principles from this success.

ORIGINAL: "{original}"
OPTIMIZED: "{optimized}"
SUCCESS SCORE: {score:.3f}

TASK: Identify specific, reusable principles that made this optimization successful.

Focus on:
- What specific changes were made and why they worked
- What patterns can be generalized to other prompts
- What rules or heuristics can be extracted

RESPONSE FORMAT:
PRINCIPLE_ANALYSIS: [Your analysis of what made this successful]

EXTRACTED_PRINCIPLES: {{
  "principles": [
    {{
      "principle_id": "[unique_id]",
      "description": "[clear description of the principle]",
      "confidence": [0.0-1.0],
      "applicability": "[when to use this principle]",
      "example_application": "[how it was applied here]"
    }}
  ]
}}"""

        try:
            response = self._query_llama(system_prompt)
            structured_output = self._parse_principle_response(response)
            
            return ConversationTurn(
                turn_type="principle_extraction",
                agent_reasoning=response,
                structured_output=structured_output,
                timestamp=time.time()
            )
            
        except Exception as e:
            print(f"   ⚠️ Principle extraction failed: {e}")
            return ConversationTurn(
                turn_type="principle_extraction",
                agent_reasoning=f"Failed to extract principles: {e}",
                structured_output={"principles": []},
                timestamp=time.time()
            )
    
    def _build_principles_context(self) -> str:
        """Build context of learned principles"""
        if not self.learned_principles:
            return "LEARNED PRINCIPLES: None yet - you're starting fresh!"
        
        # Get top principles by success rate and usage
        sorted_principles = sorted(
            self.learned_principles.values(),
            key=lambda p: (p.success_rate, p.usage_count),
            reverse=True
        )
        
        context = "LEARNED PRINCIPLES (your accumulated wisdom):\n"
        for i, principle in enumerate(sorted_principles[:5], 1):
            context += f"{i}. {principle.principle_id}: {principle.description}\n"
            context += f"   Success Rate: {principle.success_rate:.1%} | Used: {principle.usage_count} times\n"
        
        return context
    
    def _build_recent_success_context(self) -> str:
        """Build context of recent successes"""
        recent_successes = [
            session for session in self.optimization_sessions[-10:]
            if session.success and session.validation_score and session.validation_score >= 0.8
        ]
        
        if not recent_successes:
            return "RECENT SUCCESSES: None in recent memory - need to explore more!"
        
        context = "RECENT SUCCESSFUL OPTIMIZATIONS:\n"
        for session in recent_successes[-3:]:
            context += f"- '{session.original_prompt}' → Score: {session.validation_score:.3f}\n"
        
        return context
    
    def _build_strategy_context(self, strategy: str, applicable_principles: List[str]) -> str:
        """Build context for strategy execution"""
        context = f"STRATEGY DETAILS FOR {strategy}:\n"
        
        # Add principle details
        if applicable_principles:
            context += "APPLICABLE PRINCIPLES:\n"
            for principle_id in applicable_principles:
                if principle_id in self.learned_principles:
                    principle = self.learned_principles[principle_id]
                    context += f"- {principle.description}\n"
                    if principle.example_transformations:
                        example = principle.example_transformations[0]
                        context += f"  Example: '{example[0]}' → '{example[1]}' (score: {example[2]:.3f})\n"
        
        return context
    
    # Robust parsing methods with structured + unstructured fallbacks
    def _parse_strategy_response(self, response: str) -> Dict[str, Any]:
        """Parse strategy selection response with robust fallbacks"""
        try:
            # Try to extract JSON block
            json_match = re.search(r'STRATEGY_DECISION:\s*(\{.*?\})', response, re.DOTALL | re.IGNORECASE)
            if json_match:
                json_str = json_match.group(1)
                return json.loads(json_str)
        except:
            pass
        
        # Fallback: Extract key information with regex
        strategy = "conservative_enhancement"
        exploration = "exploit"
        confidence = 0.5
        
        strategy_match = re.search(r'(?:strategy|selected_strategy)[":\s]*([a-z_]+)', response, re.IGNORECASE)
        if strategy_match:
            strategy = strategy_match.group(1)
        
        explore_match = re.search(r'(explore|exploit)', response, re.IGNORECASE)
        if explore_match:
            exploration = explore_match.group(1).lower()
        
        conf_match = re.search(r'confidence["\s:]*([0-9.]+)', response, re.IGNORECASE)
        if conf_match:
            confidence = float(conf_match.group(1))
        
        return {
            "selected_strategy": strategy,
            "exploration_type": exploration,
            "confidence_in_strategy": confidence,
            "reasoning_summary": "Parsed from unstructured response",
            "applicable_principles": [],
            "expected_improvements": []
        }
    
    def _parse_optimization_response(self, response: str, original_prompt: str) -> Dict[str, Any]:
        """Parse optimization response with robust fallbacks"""
        try:
            # Try to extract JSON
            json_match = re.search(r'OPTIMIZATION_RESULT:\s*(\{.*?\})', response, re.DOTALL | re.IGNORECASE)
            if json_match:
                json_str = json_match.group(1)
                parsed = json.loads(json_str)
                if 'optimized_prompt' in parsed:
                    return parsed
        except:
            pass
        
        # Fallback: Extract optimized prompt and confidence
        optimized_prompt = f"wbgmsst, {original_prompt}, white background"
        confidence = 0.5
        
        # Look for optimized prompt
        opt_patterns = [
            r'optimized_prompt[":\s]*"([^"]+)"',
            r'wbgmsst[^"]*([^"]+(?:white background|white bg))',
            r'OPTIMIZED:\s*(.+?)(?:\n|$)'
        ]
        
        for pattern in opt_patterns:
            match = re.search(pattern, response, re.IGNORECASE | re.DOTALL)
            if match:
                optimized_prompt = match.group(1).strip()
                break
        
        # Ensure proper format
        if not optimized_prompt.startswith('wbgmsst'):
            optimized_prompt = f"wbgmsst, {optimized_prompt}"
        if not optimized_prompt.endswith('white background'):
            optimized_prompt = optimized_prompt.rstrip(', ') + ", white background"
        
        # Look for confidence
        conf_match = re.search(r'confidence["\s:]*([0-9.]+)', response, re.IGNORECASE)
        if conf_match:
            confidence = max(0.0, min(1.0, float(conf_match.group(1))))
        
        return {
            "optimized_prompt": optimized_prompt,
            "confidence": confidence,
            "key_improvements": ["structured parsing fallback"],
            "principles_applied": [],
            "risk_assessment": "medium"
        }
    
    def _parse_reflection_response(self, response: str) -> Dict[str, Any]:
        """Parse reflection with fallbacks"""
        try:
            json_match = re.search(r'REFLECTION_ANALYSIS:\s*(\{.*?\})', response, re.DOTALL | re.IGNORECASE)
            if json_match:
                return json.loads(json_match.group(1))
        except:
            pass
        
        return {
            "performance_assessment": "moderate",
            "confidence_accuracy": "accurate",
            "what_worked_well": ["optimization completed"],
            "what_could_improve": ["structure parsing"],
            "lessons_learned": ["need better structured output"],
            "strategy_effectiveness": "moderate"
        }
    
    def _parse_principle_response(self, response: str) -> Dict[str, Any]:
        """Parse principle extraction with fallbacks"""
        try:
            json_match = re.search(r'EXTRACTED_PRINCIPLES:\s*(\{.*?\})', response, re.DOTALL | re.IGNORECASE)
            if json_match:
                return json.loads(json_match.group(1))
        except:
            pass
        
        return {"principles": []}
    
    def _add_learned_principle(self, principle_data: Dict[str, Any], 
                             original: str, optimized: str, score: float):
        """Add a new learned principle"""
        principle_id = principle_data.get('principle_id', f"principle_{int(time.time())}")
        
        principle = LearnedPrinciple(
            principle_id=principle_id,
            description=principle_data.get('description', 'Unknown principle'),
            confidence=principle_data.get('confidence', 0.5),
            success_rate=1.0,  # Start at 100% since it just succeeded
            example_transformations=[(original, optimized, score)],
            discovery_timestamp=time.time(),
            usage_count=1,
            last_successful_use=time.time()
        )
        
        self.learned_principles[principle_id] = principle
        print(f"   🎓 Learned new principle: {principle_id}")
    
    def _evaluate_success(self, confidence: float, validation_score: Optional[float]) -> bool:
        """Evaluate if optimization was successful"""
        if validation_score is not None:
            return validation_score >= 0.7
        else:
            return confidence >= 0.7
    
    def _validate_prompt(self, prompt: str) -> float:
        """Run validation with conda environment"""
        try:
            print("   🔍 Running validation...")
            cmd = [
                "bash", "-c",
                f"source /home/mbhat/miniconda/bin/activate && conda activate trellis_new && python subnet_accurate_validator.py \"{prompt}\""
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
            
            if result.returncode == 0:
                with open("subnet_validation_results.json", 'r') as f:
                    data = json.load(f)
                    score = data.get("validation_engine_score", 0.0)
                    print(f"   📊 Validation score: {score:.4f}")
                    return score
            else:
                print(f"   ❌ Validation failed: {result.stderr}")
                return 0.0
        except Exception as e:
            print(f"   ❌ Validation error: {e}")
            return 0.0
    
    def _query_llama(self, prompt: str) -> str:
        """Query LLM with conversational prompt"""
        data = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "stream": False,
            "options": {
                "temperature": 0.6,  # Balanced creativity and consistency
                "num_predict": 400,  # Longer for conversational responses
                "top_p": 0.9
            }
        }
        
        response = requests.post(f"{self.ollama_url}/api/chat", json=data, timeout=60)
        response.raise_for_status()
        return response.json()["message"]["content"].strip()
    
    def _fallback_result(self, prompt: str, session_id: str, elapsed_time: float) -> Dict[str, Any]:
        """Fallback result when conversation fails"""
        return {
            'session_id': session_id,
            'original_prompt': prompt,
            'optimized_prompt': f"wbgmsst, {prompt}, white background",
            'confidence': 0.5,
            'validation_score': None,
            'success': False,
            'processing_time': elapsed_time,
            'conversation_turns': 0,
            'new_principles_learned': 0,
            'total_principles': len(self.learned_principles),
            'overall_success_rate': self._get_success_rate()
        }
    
    def get_learning_insights(self) -> Dict[str, Any]:
        """Get comprehensive learning insights"""
        if not self.optimization_sessions:
            return {"message": "No conversational learning experience yet"}
        
        recent_sessions = self.optimization_sessions[-20:]
        
        insights = {
            "total_experience": len(self.optimization_sessions),
            "overall_success_rate": self._get_success_rate(),
            "learned_principles": len(self.learned_principles),
            "recent_performance": {
                "sessions": len(recent_sessions),
                "successes": len([s for s in recent_sessions if s.success]),
                "avg_confidence": statistics.mean([s.final_confidence for s in recent_sessions]),
                "avg_conversation_turns": statistics.mean([len(s.conversation_turns) for s in recent_sessions])
            },
            "top_principles": []
        }
        
        # Add top principles
        sorted_principles = sorted(
            self.learned_principles.values(),
            key=lambda p: (p.success_rate, p.usage_count),
            reverse=True
        )
        
        for principle in sorted_principles[:5]:
            insights["top_principles"].append({
                "id": principle.principle_id,
                "description": principle.description,
                "success_rate": principle.success_rate,
                "usage_count": principle.usage_count,
                "confidence": principle.confidence
            })
        
        return insights

def main():
    """Command line interface"""
    if len(sys.argv) < 2 or sys.argv[1] in ('-h', '--help'):
        print("Usage: python smart_prompt_optimizer_v4_conversational.py \"prompt\" [--validate] [--insights]")
        print("\nCommands:")
        print("  \"prompt\"        The prompt to optimize through conversation")
        print("  dummy --insights Show conversational learning insights")
        print("\nOptions:")
        print("  --validate       Run external validation for RL learning")
        return
    
    agent = None
    try:
        if "--insights" in sys.argv:
            agent = ConversationalRLAgent()
            insights = agent.get_learning_insights()
            print("\n🗣️ CONVERSATIONAL RL AGENT INSIGHTS:")
            print("=" * 50)
            print(json.dumps(insights, indent=2))
            return
        
        user_prompt = sys.argv[1]
        use_validation = "--validate" in sys.argv
        
        print("🗣️ CONVERSATIONAL RL AGENT - AUTONOMOUS REASONING")
        print("=" * 60)
        print("✅ Agent selects own strategies through reasoning")
        print("✅ Multi-turn conversational learning")
        print("✅ Principle extraction and reuse")
        print("✅ Robust structured + unstructured parsing")
        print("=" * 60)
        
        agent = ConversationalRLAgent()
        result = agent.optimize(user_prompt, use_validation=use_validation)
        
        print("\n" + "="*20 + " CONVERSATION SUMMARY " + "="*19)
        print(f"   Original: {result['original_prompt']}")
        print(f"   Optimized: {result['optimized_prompt']}")
        print(f"   Confidence: {result['confidence']:.1%}")
        print(f"   Success: {result['success']}")
        print(f"   Conversation turns: {result['conversation_turns']}")
        print(f"   New principles: {result['new_principles_learned']}")
        print(f"   Total principles: {result['total_principles']}")
        if result['validation_score']:
            print(f"   Validation: {result['validation_score']:.3f}")
        print("="*60)
        
    except KeyboardInterrupt:
        print("\n⚠️ Interrupted by user")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if agent:
            agent._save_memory()

if __name__ == "__main__":
    main() 