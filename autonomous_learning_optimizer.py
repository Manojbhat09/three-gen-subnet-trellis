#!/usr/bin/env python3
"""
Autonomous Learning Optimizer - Complete Vision Implementation
=============================================================
Combines all your concepts:
- AI learns from validation feedback autonomously
- Builds reusable policies from successful patterns  
- Provides conversational feedback and improvement
- Generates creative variations using learned knowledge
- Stores and applies strategic patterns automatically

This is the complete implementation of your vision!
"""

import requests
import json
import time
import subprocess
import sys
import sqlite3
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, asdict
from pathlib import Path
import statistics
import re

@dataclass
class LearningEvent:
    """A learning event from validation feedback"""
    prompt: str
    score: float
    patterns_extracted: List[str]
    success_factors: List[str]
    failure_reasons: List[str]
    timestamp: float

@dataclass
class LearnedPolicy:
    """A learned policy for prompt generation"""
    policy_id: str
    pattern_template: str
    success_rate: float
    avg_score: float
    best_score: float
    usage_count: int
    applicable_categories: List[str]
    learned_from: List[str]  # Original prompts that led to this policy

@dataclass
class AIGenerationDecision:
    """AI's decision on how to generate next prompt"""
    strategy_chosen: str
    policy_applied: Optional[str]
    reasoning: str
    confidence: float
    expected_score: float
    generated_prompt: str
    actual_score: float = 0.0
    decision_success: bool = False

class AutonomousLearningOptimizer:
    """AI that learns autonomously from feedback and improves its prompt generation"""
    
    def __init__(self, ultra_target: float = 0.96):
        self.ollama_url = "http://localhost:11434"
        self.model_name = "llama3.2:3b"  # Better format compliance
        self.ultra_target = ultra_target
        
        # Learning systems
        self.learned_policies = {}
        self.learning_history = []
        self.conversation_memory = []
        self.pattern_database = {}
        
        # Initialize database
        self.setup_learning_database()
        self.load_learned_knowledge()
        
        print("🧠 AUTONOMOUS LEARNING OPTIMIZER")
        print("🎯 Vision: AI learns from feedback and improves autonomously")
        print("⚡ Features: Learning, Memory, Policy Creation, Autonomous Improvement")
        print("=" * 80)

    def setup_learning_database(self):
        """Setup database for persistent learning"""
        self.db_path = Path("autonomous_learning.db")
        
        with sqlite3.connect(self.db_path) as conn:
            # Learned policies table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS learned_policies (
                    policy_id TEXT PRIMARY KEY,
                    pattern_template TEXT,
                    success_rate REAL,
                    avg_score REAL,
                    best_score REAL,
                    usage_count INTEGER,
                    applicable_categories TEXT,
                    learned_from TEXT,
                    created_at REAL,
                    last_used REAL
                )
            """)
            
            # Learning events table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS learning_events (
                    id INTEGER PRIMARY KEY,
                    prompt TEXT,
                    score REAL,
                    patterns_extracted TEXT,
                    success_factors TEXT,
                    failure_reasons TEXT,
                    timestamp REAL
                )
            """)
            
            # AI decisions table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS ai_decisions (
                    id INTEGER PRIMARY KEY,
                    strategy_chosen TEXT,
                    policy_applied TEXT,
                    reasoning TEXT,
                    confidence REAL,
                    expected_score REAL,
                    generated_prompt TEXT,
                    actual_score REAL,
                    decision_success BOOLEAN,
                    timestamp REAL
                )
            """)

    def load_learned_knowledge(self):
        """Load previously learned knowledge"""
        with sqlite3.connect(self.db_path) as conn:
            # Load policies
            cursor = conn.execute("""
                SELECT policy_id, pattern_template, success_rate, avg_score, best_score,
                       usage_count, applicable_categories, learned_from
                FROM learned_policies ORDER BY success_rate DESC
            """)
            
            for row in cursor:
                policy = LearnedPolicy(
                    policy_id=row[0],
                    pattern_template=row[1],
                    success_rate=row[2],
                    avg_score=row[3],
                    best_score=row[4],
                    usage_count=row[5],
                    applicable_categories=json.loads(row[6]) if row[6] else [],
                    learned_from=json.loads(row[7]) if row[7] else []
                )
                self.learned_policies[policy.policy_id] = policy
        
        print(f"🧠 Loaded {len(self.learned_policies)} learned policies from previous sessions")

    def analyze_and_learn_from_result(self, prompt: str, score: float) -> LearningEvent:
        """Analyze validation result and extract learning"""
        
        print(f"🧠 AUTONOMOUS LEARNING FROM RESULT")
        print(f"   📝 Prompt: {prompt}")
        print(f"   📊 Score: {score:.3f}")
        
        # AI analyzes the result to extract learnings
        learning_request = f"""ANALYZE THIS PROMPT OPTIMIZATION RESULT FOR LEARNING:

PROMPT: "{prompt}"
VALIDATION SCORE: {score:.3f} (Target: {self.ultra_target})

YOUR TASK: Extract learnings that will help generate better prompts in the future.

ANALYSIS FRAMEWORK:
1. PATTERN EXTRACTION: What specific patterns in this prompt contributed to the score?
2. SUCCESS FACTORS: If score ≥ 0.6, what made this prompt work well?
3. FAILURE ANALYSIS: If score < 0.6, what specific issues caused the low score?
4. REUSABLE INSIGHTS: What patterns can be applied to similar prompts?

RESPOND IN THIS FORMAT:
PATTERNS: [list specific descriptor/structure patterns found]
SUCCESS_FACTORS: [what worked well - if any]  
FAILURE_REASONS: [what didn't work - if any]
REUSABLE_TEMPLATE: [template pattern for future use]
CONFIDENCE: [0.0-1.0 confidence in this analysis]

LEARNING ANALYSIS:"""

        try:
            data = {
                "model": self.model_name,
                "messages": [
                    {
                        "role": "system",
                        "content": "You are an AI learning analyst. You extract specific, actionable patterns from prompt performance data to improve future generations."
                    },
                    {"role": "user", "content": learning_request}
                ],
                "stream": False,
                "options": {"temperature": 0.3, "num_predict": 400}
            }
            
            response = requests.post(f"{self.ollama_url}/api/chat", json=data, timeout=45)
            ai_analysis = response.json()["message"]["content"]
            
            # Parse learning from AI analysis
            learning_event = self._parse_learning_analysis(ai_analysis, prompt, score)
            
            print(f"   🎯 Patterns extracted: {', '.join(learning_event.patterns_extracted)}")
            print(f"   ✅ Success factors: {', '.join(learning_event.success_factors)}")
            print(f"   ⚠️ Failure reasons: {', '.join(learning_event.failure_reasons)}")
            
            # Store learning
            self._store_learning_event(learning_event)
            
            # Update or create policies based on learning
            if score >= 0.7:  # Good enough to learn from
                self._update_policies_from_learning(learning_event)
            
            return learning_event
            
        except Exception as e:
            print(f"   ❌ Learning analysis failed: {e}")
            # Create minimal learning event
            return LearningEvent(
                prompt=prompt,
                score=score,
                patterns_extracted=["manual_fallback"],
                success_factors=[] if score < 0.6 else ["unknown_success"],
                failure_reasons=[] if score >= 0.6 else ["unknown_failure"],
                timestamp=time.time()
            )

    def _parse_learning_analysis(self, analysis: str, prompt: str, score: float) -> LearningEvent:
        """Parse AI learning analysis"""
        
        patterns = re.search(r'PATTERNS:\s*(.+?)(?=SUCCESS_FACTORS:|$)', analysis, re.DOTALL)
        success = re.search(r'SUCCESS_FACTORS:\s*(.+?)(?=FAILURE_REASONS:|$)', analysis, re.DOTALL)
        failures = re.search(r'FAILURE_REASONS:\s*(.+?)(?=REUSABLE_TEMPLATE:|$)', analysis, re.DOTALL)
        
        return LearningEvent(
            prompt=prompt,
            score=score,
            patterns_extracted=self._parse_list(patterns.group(1) if patterns else ""),
            success_factors=self._parse_list(success.group(1) if success else ""),
            failure_reasons=self._parse_list(failures.group(1) if failures else ""),
            timestamp=time.time()
        )

    def _parse_list(self, text: str) -> List[str]:
        """Parse list from text"""
        items = re.split(r'[,\n\-•]', text.strip())
        return [item.strip() for item in items if item.strip() and len(item.strip()) > 2]

    def _store_learning_event(self, event: LearningEvent):
        """Store learning event in database"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT INTO learning_events 
                (prompt, score, patterns_extracted, success_factors, failure_reasons, timestamp)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (
                event.prompt,
                event.score,
                json.dumps(event.patterns_extracted),
                json.dumps(event.success_factors),
                json.dumps(event.failure_reasons),
                event.timestamp
            ))

    def _update_policies_from_learning(self, event: LearningEvent):
        """Update or create policies based on learning"""
        
        print(f"   🔄 Updating policies from successful result...")
        
        # Extract policy template from successful prompt
        if event.score >= 0.7 and event.patterns_extracted:
            # Create policy from successful pattern
            policy_id = f"learned_{len(self.learned_policies)}"
            
            # Extract template pattern from the prompt
            template = self._extract_template_pattern(event.prompt)
            
            policy = LearnedPolicy(
                policy_id=policy_id,
                pattern_template=template,
                success_rate=1.0,  # Start with perfect rate
                avg_score=event.score,
                best_score=event.score,
                usage_count=0,
                applicable_categories=["general"],
                learned_from=[event.prompt]
            )
            
            self.learned_policies[policy_id] = policy
            self._save_policy(policy)
            
            print(f"   📚 Created new policy: {policy_id}")
            print(f"   📝 Template: {template}")

    def _extract_template_pattern(self, prompt: str) -> str:
        """Extract reusable template pattern from successful prompt"""
        
        # Find the pattern between wbgmsst and white background
        if 'wbgmsst,' in prompt and ', white background' in prompt:
            middle = prompt.split('wbgmsst,')[1].split(', white background')[0].strip()
            
            # Replace specific object with placeholder
            parts = middle.split()
            if len(parts) >= 3:
                # Assume last 2-3 words are the object
                template_parts = parts[:-2] + ["{target}"]
                return "wbgmsst, " + " ".join(template_parts) + ", white background"
        
        return "wbgmsst, learned-pattern {target}, white background"

    def _save_policy(self, policy: LearnedPolicy):
        """Save policy to database"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT OR REPLACE INTO learned_policies 
                (policy_id, pattern_template, success_rate, avg_score, best_score,
                 usage_count, applicable_categories, learned_from, created_at, last_used)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                policy.policy_id,
                policy.pattern_template,
                policy.success_rate,
                policy.avg_score,
                policy.best_score,
                policy.usage_count,
                json.dumps(policy.applicable_categories),
                json.dumps(policy.learned_from),
                time.time(),
                0
            ))

    def autonomous_prompt_generation(self, target_prompt: str) -> AIGenerationDecision:
        """AI autonomously decides how to generate the best prompt"""
        
        print(f"🤖 AUTONOMOUS PROMPT GENERATION")
        print(f"   🎯 Target: {target_prompt}")
        
        # AI makes strategic decision based on learned knowledge
        decision_request = f"""AUTONOMOUS PROMPT GENERATION DECISION

TARGET OBJECT: "{target_prompt}"
GOAL: Generate prompt that will score {self.ultra_target}+ 

YOUR LEARNED KNOWLEDGE:
{self._format_learned_knowledge()}

AVAILABLE STRATEGIES:
1. APPLY_LEARNED_POLICY: Use a proven successful pattern from your learning
2. CREATIVE_COMBINATION: Combine successful elements creatively  
3. PROVEN_PATTERN: Use the ultra-successful "defense-grade ultra-precision" pattern
4. STRATEGIC_ENHANCEMENT: Apply strategic optimization techniques

YOUR TASK: Decide the best strategy and generate the optimal prompt.

DECISION FORMAT:
STRATEGY: [chosen strategy name]
POLICY_USED: [if applying learned policy, specify which one]
REASONING: [why this approach will work best]
CONFIDENCE: [0.0-1.0 confidence level]
EXPECTED_SCORE: [predicted score]
GENERATED_PROMPT: [the actual prompt - must start with "wbgmsst," and end with ", white background"]

MAKE YOUR AUTONOMOUS DECISION:"""

        try:
            data = {
                "model": self.model_name,
                "messages": [
                    {
                        "role": "system",
                        "content": "You are an autonomous AI prompt optimizer. You make strategic decisions based on learned knowledge and generate optimal prompts. You always follow the exact format requested."
                    },
                    {"role": "user", "content": decision_request}
                ],
                "stream": False,
                "options": {"temperature": 0.5, "num_predict": 400}
            }
            
            response = requests.post(f"{self.ollama_url}/api/chat", json=data, timeout=45)
            ai_decision = response.json()["message"]["content"]
            
            print(f"   🧠 AI Decision: {ai_decision[:200]}...")
            
            # Parse AI decision
            decision = self._parse_ai_decision(ai_decision, target_prompt)
            
            print(f"   ⚡ Strategy: {decision.strategy_chosen}")
            print(f"   📝 Generated: {decision.generated_prompt}")
            print(f"   🎯 Expected Score: {decision.expected_score:.3f}")
            print(f"   💪 Confidence: {decision.confidence:.2f}")
            
            return decision
            
        except Exception as e:
            print(f"   ❌ Autonomous generation failed: {e}")
            # Fallback to proven pattern
            return self._create_fallback_decision(target_prompt)

    def _format_learned_knowledge(self) -> str:
        """Format learned knowledge for AI decision making"""
        
        if not self.learned_policies:
            return "No learned policies yet - this is your first session."
        
        knowledge = "LEARNED POLICIES:\n"
        for policy_id, policy in list(self.learned_policies.items())[:5]:  # Top 5
            knowledge += f"- {policy_id}: {policy.pattern_template} (Success: {policy.success_rate:.1%}, Avg Score: {policy.avg_score:.3f})\n"
        
        return knowledge

    def _parse_ai_decision(self, decision_text: str, target_prompt: str) -> AIGenerationDecision:
        """Parse AI's autonomous decision"""
        
        strategy = re.search(r'STRATEGY:\s*(.+?)(?=POLICY_USED:|$)', decision_text, re.DOTALL)
        policy = re.search(r'POLICY_USED:\s*(.+?)(?=REASONING:|$)', decision_text, re.DOTALL)
        reasoning = re.search(r'REASONING:\s*(.+?)(?=CONFIDENCE:|$)', decision_text, re.DOTALL)
        confidence = re.search(r'CONFIDENCE:\s*(.+?)(?=EXPECTED_SCORE:|$)', decision_text, re.DOTALL)
        expected = re.search(r'EXPECTED_SCORE:\s*(.+?)(?=GENERATED_PROMPT:|$)', decision_text, re.DOTALL)
        generated = re.search(r'GENERATED_PROMPT:\s*(.+?)$', decision_text, re.DOTALL)
        
        # Extract and validate generated prompt
        gen_prompt = generated.group(1).strip() if generated else ""
        gen_prompt = gen_prompt.strip('"\'')
        
        # If invalid, create fallback
        if not (gen_prompt.lower().startswith('wbgmsst,') and 
                gen_prompt.lower().endswith(', white background') and
                target_prompt.lower() in gen_prompt.lower()):
            gen_prompt = f"wbgmsst, defense-grade ultra-precision {target_prompt}, ultra-high technical specification, white background"
        
        return AIGenerationDecision(
            strategy_chosen=strategy.group(1).strip() if strategy else "FALLBACK",
            policy_applied=policy.group(1).strip() if policy and "none" not in policy.group(1).lower() else None,
            reasoning=reasoning.group(1).strip() if reasoning else "Autonomous generation",
            confidence=float(confidence.group(1).strip()) if confidence else 0.7,
            expected_score=float(expected.group(1).strip()) if expected else 0.8,
            generated_prompt=gen_prompt
        )

    def _create_fallback_decision(self, target_prompt: str) -> AIGenerationDecision:
        """Create fallback decision if AI fails"""
        return AIGenerationDecision(
            strategy_chosen="PROVEN_PATTERN_FALLBACK",
            policy_applied=None,
            reasoning="AI decision failed, using proven ultra pattern",
            confidence=0.8,
            expected_score=0.9,
            generated_prompt=f"wbgmsst, defense-grade ultra-precision {target_prompt}, ultra-high technical specification, white background"
        )

    def run_validation(self, prompt: str) -> float:
        """Run validation with proper environment"""
        try:
            cmd = [
                "bash", "-c", 
                f"source /home/mbhat/miniconda/bin/activate && conda activate trellis_new && python subnet_accurate_validator.py '{prompt}'"
            ]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
            
            if result.returncode != 0:
                return 0.0
            
            with open("subnet_validation_results.json", 'r') as f:
                data = json.load(f)
                return data.get("validation_engine_score", 0.0)
        
        except Exception:
            return 0.0

    def autonomous_optimization_cycle(self, target_prompt: str, max_cycles: int = 5) -> Dict:
        """Complete autonomous learning cycle"""
        
        print(f"\n🧠 AUTONOMOUS OPTIMIZATION CYCLE")
        print(f"🎯 Target: {target_prompt}")
        print(f"🔄 Max Cycles: {max_cycles}")
        print("=" * 80)
        
        best_score = 0.0
        best_prompt = ""
        learning_progression = []
        
        for cycle in range(1, max_cycles + 1):
            print(f"\n🔄 CYCLE {cycle}/{max_cycles}")
            
            # AI autonomously generates prompt
            decision = self.autonomous_prompt_generation(target_prompt)
            
            # Test the generated prompt
            print(f"   🔧 Validating autonomous generation...")
            score = self.run_validation(decision.generated_prompt)
            
            # Update decision with actual result
            decision.actual_score = score
            decision.decision_success = score >= decision.expected_score * 0.8  # Within 20% of expectation
            
            print(f"   📊 Actual Score: {score:.3f}")
            print(f"   🎯 Expected: {decision.expected_score:.3f}")
            print(f"   ✅ Decision Success: {decision.decision_success}")
            
            # Learn from this result
            learning_event = self.analyze_and_learn_from_result(decision.generated_prompt, score)
            learning_progression.append((cycle, score, decision.strategy_chosen))
            
            # Update best result
            if score > best_score:
                best_score = score
                best_prompt = decision.generated_prompt
                print(f"   🌟 NEW BEST SCORE!")
            
            # Store AI decision for analysis
            self._store_ai_decision(decision)
            
            # Check for ultra achievement
            if score >= self.ultra_target:
                print(f"   🎉 ULTRA ACHIEVED IN {cycle} CYCLES!")
                break
            
            # Brief pause for next cycle
            if cycle < max_cycles:
                print(f"   🔄 Learning applied, proceeding to next cycle...")
                time.sleep(1)
        
        # Final analysis
        print(f"\n🎓 AUTONOMOUS OPTIMIZATION COMPLETE")
        print("=" * 80)
        print(f"   🏆 Best Score: {best_score:.3f}")
        print(f"   📝 Best Prompt: {best_prompt}")
        print(f"   🧠 Learning Events: {len(learning_progression)}")
        print(f"   📚 Total Policies: {len(self.learned_policies)}")
        
        return {
            "target_prompt": target_prompt,
            "best_score": best_score,
            "best_prompt": best_prompt,
            "cycles_completed": len(learning_progression),
            "ultra_achieved": best_score >= self.ultra_target,
            "learning_progression": learning_progression
        }

    def _store_ai_decision(self, decision: AIGenerationDecision):
        """Store AI decision for analysis"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT INTO ai_decisions 
                (strategy_chosen, policy_applied, reasoning, confidence, expected_score,
                 generated_prompt, actual_score, decision_success, timestamp)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                decision.strategy_chosen,
                decision.policy_applied,
                decision.reasoning,
                decision.confidence,
                decision.expected_score,
                decision.generated_prompt,
                decision.actual_score,
                decision.decision_success,
                time.time()
            ))

def main():
    """Test autonomous learning optimization"""
    
    print("🧠 AUTONOMOUS LEARNING OPTIMIZER TEST")
    print("🎯 Vision: AI learns from feedback and improves autonomously")
    print("=" * 80)
    
    optimizer = AutonomousLearningOptimizer(ultra_target=0.96)
    
    test_prompts = [
        "hexagonal prism steel structure",
        "elegant silk fabric draping"
    ]
    
    all_results = []
    
    for prompt in test_prompts:
        print(f"\n{'='*20} AUTONOMOUS TEST: {prompt} {'='*20}")
        
        result = optimizer.autonomous_optimization_cycle(prompt, max_cycles=3)
        all_results.append(result)
        
        print(f"\n⏸️ Brief pause before next test...")
        time.sleep(2)
    
    # Overall autonomous performance analysis
    print(f"\n🎓 AUTONOMOUS LEARNING ANALYSIS")
    print("=" * 80)
    
    total_ultra = sum(1 for r in all_results if r['ultra_achieved'])
    avg_best = sum(r['best_score'] for r in all_results) / len(all_results)
    
    print(f"📊 AUTONOMOUS PERFORMANCE:")
    print(f"   Tests Completed: {len(all_results)}")
    print(f"   Ultra Achievements: {total_ultra}/{len(all_results)}")
    print(f"   Average Best Score: {avg_best:.3f}")
    print(f"   Learning Database: {len(optimizer.learned_policies)} policies")
    
    print(f"\n🧠 AI LEARNING PROGRESSION:")
    for result in all_results:
        status = "🎉 ULTRA" if result['ultra_achieved'] else "📈 LEARNING"
        print(f"   {status} {result['target_prompt']}: {result['best_score']:.3f} in {result['cycles_completed']} cycles")

if __name__ == "__main__":
    main() 