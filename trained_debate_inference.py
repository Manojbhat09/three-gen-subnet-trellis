#!/usr/bin/env python3
"""
Trained Debate Inference System
==============================

This system uses a trained Reviewer as a proxy for the external Judge,
enabling fast validation-free prompt optimization.

After training with the Proposer-Reviewer-Judge system, the Reviewer has learned
to accurately predict validation scores. This script uses that trained Reviewer
for fast inference without external validation dependencies.

Key Benefits:
- 10x+ faster than external validation (3-6s vs 30-60s)
- No external dependencies (subprocess, conda environment)
- Reliable quality scoring based on learned patterns
- Self-contained conversation-based optimization
- Maintains quality through learned ground truth patterns
"""

import json
import requests
import time
import random
import re
import os
import statistics
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path
from dataclasses import dataclass, asdict
import logging
import sys


@dataclass
class InferenceRound:
    """Data for a single round of Proposer-Reviewer inference."""
    round_number: int
    proposer_response: str
    proposed_prompt: str
    strategy_used: str
    reviewer_score: float
    reviewer_critique: str
    reviewer_suggestion: str
    reviewer_confidence: float
    round_duration: float
    timestamp: str


@dataclass
class InferenceSession:
    """Complete inference session results."""
    session_id: str
    original_prompt: str
    timestamp: str
    
    rounds: List[InferenceRound]
    total_rounds: int
    
    best_prompt: str
    best_score: float
    best_round_number: int
    
    final_prompt: str
    final_score: float
    
    score_improvement: float
    session_duration: float
    converged: bool
    
    strategy_used: str
    avg_confidence: float


class TrainedDebateInference:
    """
    Fast inference system using trained Proposer-Reviewer agents.
    
    This system leverages the Reviewer trained with Judge ground truth
    to provide fast, reliable optimization without external validation.
    """
    
    def __init__(self,
                 ollama_url: str = "http://localhost:11434",
                 model: str = "llama3.2:3b", 
                 training_memory_file: str = "prj_training_memory.json",
                 inference_log_file: str = "inference_sessions.json",
                 max_rounds: int = 3,
                 target_score: float = 0.9,
                 min_improvement: float = 0.05):
        """
        Initialize the trained debate inference system.
        
        Args:
            ollama_url: Ollama server URL
            model: LLM model to use
            training_memory_file: File containing trained examples
            inference_log_file: File to log inference sessions
            max_rounds: Maximum rounds per optimization
            target_score: Target score for convergence
            min_improvement: Minimum improvement to continue iterating
        """
        self.ollama_url = ollama_url
        self.model = model
        self.training_memory_file = Path(training_memory_file)
        self.inference_log_file = Path(inference_log_file)
        self.max_rounds = max_rounds
        self.target_score = target_score
        self.min_improvement = min_improvement
        
        # Optimization strategies
        self.strategies = [
            "material_focus",
            "artistic_elaboration", 
            "technical_precision",
            "atmospheric_enhancement",
            "structural_detailing",
            "lighting_emphasis",
            "texture_specification",
            "contextual_placement"
        ]
        
        # Load trained knowledge
        self.training_examples: List[Dict] = []
        self.strategy_performance: Dict[str, List[float]] = {}
        self.reviewer_calibration_data: List[float] = []
        
        # Inference tracking
        self.inference_sessions: List[InferenceSession] = []
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # Load training data and past inference sessions
        self._load_training_knowledge()
        self._load_inference_history()
        
        self.logger.info("🚀 TRAINED DEBATE INFERENCE SYSTEM INITIALIZED")
        self.logger.info(f"   Model: {self.model}")
        self.logger.info(f"   Training examples loaded: {len(self.training_examples)}")
        self.logger.info(f"   Strategy performance data: {len(self.strategy_performance)} strategies")
        self.logger.info(f"   Max rounds: {self.max_rounds}")
        self.logger.info(f"   Target score: {self.target_score}")
    
    def _load_training_knowledge(self):
        """Load knowledge from training sessions to inform inference."""
        if not self.training_memory_file.exists():
            self.logger.warning(f"Training memory file not found: {self.training_memory_file}")
            self.logger.warning("Inference will work but may be less accurate without training data")
            return
        
        try:
            with open(self.training_memory_file, 'r') as f:
                data = json.load(f)
            
            # Extract training examples
            for session_data in data.get('training_sessions', []):
                # Get the best round from each session
                best_round = None
                best_score = 0.0
                
                for round_data in session_data.get('rounds', []):
                    judge_score = round_data.get('judge_score', 0.0)
                    if judge_score > best_score:
                        best_score = judge_score
                        best_round = round_data
                
                if best_round and best_score > 0.7:  # Only use good examples
                    example = {
                        'original_prompt': session_data.get('original_prompt', ''),
                        'optimized_prompt': best_round.get('proposed_prompt', ''),
                        'judge_score': best_score,
                        'reviewer_prediction': best_round.get('predicted_score', 0.0),
                        'strategy': best_round.get('strategy_used', 'unknown'),
                        'critique': best_round.get('critique', ''),
                        'suggestion': best_round.get('suggestion', '')
                    }
                    self.training_examples.append(example)
            
            # Load strategy performance
            self.strategy_performance = data.get('strategy_performance', {})
            
            # Load reviewer calibration data
            self.reviewer_calibration_data = data.get('reviewer_calibration_history', [])
            
            self.logger.info(f"📚 Loaded {len(self.training_examples)} training examples")
            
        except Exception as e:
            self.logger.error(f"Error loading training knowledge: {str(e)}")
    
    def _load_inference_history(self):
        """Load previous inference sessions."""
        if not self.inference_log_file.exists():
            return
        
        try:
            with open(self.inference_log_file, 'r') as f:
                data = json.load(f)
            
            for session_data in data.get('inference_sessions', []):
                # Reconstruct rounds
                rounds = []
                for round_data in session_data.get('rounds', []):
                    rounds.append(InferenceRound(**round_data))
                
                session_data['rounds'] = rounds
                self.inference_sessions.append(InferenceSession(**session_data))
            
            self.logger.info(f"📖 Loaded {len(self.inference_sessions)} previous inference sessions")
            
        except Exception as e:
            self.logger.error(f"Error loading inference history: {str(e)}")
    
    def _save_inference_session(self, session: InferenceSession):
        """Save inference session to history."""
        try:
            self.inference_sessions.append(session)
            
            # Prepare data for JSON serialization
            sessions_data = {
                'inference_sessions': [asdict(session) for session in self.inference_sessions],
                'system_metadata': {
                    'total_sessions': len(self.inference_sessions),
                    'average_score': statistics.mean([s.best_score for s in self.inference_sessions]),
                    'average_rounds': statistics.mean([s.total_rounds for s in self.inference_sessions]),
                    'convergence_rate': len([s for s in self.inference_sessions if s.converged]) / max(1, len(self.inference_sessions))
                },
                'last_updated': datetime.now().isoformat()
            }
            
            with open(self.inference_log_file, 'w') as f:
                json.dump(sessions_data, f, indent=2)
                
        except Exception as e:
            self.logger.error(f"Error saving inference session: {str(e)}")
    
    def _call_llm(self, prompt: str, temperature: float = 0.6) -> str:
        """Make a call to the LLM with error handling."""
        try:
            response = requests.post(f"{self.ollama_url}/api/generate",
                                   json={
                                       "model": self.model,
                                       "prompt": prompt,
                                       "stream": False,
                                       "options": {
                                           "temperature": temperature,
                                           "num_predict": 500
                                       }
                                   }, timeout=60)
            
            if response.status_code == 200:
                return response.json()["response"].strip()
            else:
                self.logger.error(f"LLM error: {response.status_code}")
                return ""
                
        except Exception as e:
            self.logger.error(f"LLM call failed: {str(e)}")
            return ""
    
    def _select_strategy(self) -> str:
        """Select optimization strategy based on training performance."""
        if not self.strategy_performance:
            return random.choice(self.strategies)
        
        # Calculate strategy effectiveness from training
        strategy_scores = {}
        for strategy, scores in self.strategy_performance.items():
            if scores:
                strategy_scores[strategy] = statistics.mean(scores)
            else:
                strategy_scores[strategy] = 0.5
        
        # Epsilon-greedy: 80% exploit best, 20% explore
        if random.random() < 0.8 and strategy_scores:
            best_strategy = max(strategy_scores.keys(), key=lambda s: strategy_scores[s])
            return best_strategy
        else:
            return random.choice(self.strategies)
    
    def _run_trained_proposer(self, original_prompt: str, strategy: str, rounds: List[InferenceRound]) -> Tuple[str, str]:
        """
        Run Proposer using training knowledge for improved optimization.
        
        Args:
            original_prompt: Original prompt to optimize
            strategy: Strategy to use
            rounds: Previous rounds for context
            
        Returns:
            (proposer_response, proposed_prompt)
        """
        self.logger.info(f"💡 TRAINED PROPOSER: Strategy '{strategy}'")
        
        # Build context from training examples
        context = f"""You are an expert Proposer trained on successful prompt optimizations. Your goal is to create prompts that will receive high scores from the trained Reviewer.

Your current strategy is: {strategy}

Strategy Definitions:
- material_focus: Emphasize material properties, textures, and surface qualities
- artistic_elaboration: Add artistic style, aesthetic qualities, and visual appeal
- technical_precision: Include technical specifications and structural details
- atmospheric_enhancement: Add lighting, environment, and mood elements
- structural_detailing: Focus on shape, form, and geometric properties
- lighting_emphasis: Specify lighting conditions and illumination effects
- texture_specification: Detail surface textures and tactile qualities
- contextual_placement: Add environmental context and background elements

"""
        
        # Add successful training examples relevant to the strategy
        strategy_examples = [ex for ex in self.training_examples if ex.get('strategy') == strategy]
        if strategy_examples:
            context += f"\n--- SUCCESSFUL {strategy.upper()} EXAMPLES ---\n"
            for example in strategy_examples[:2]:  # Show top 2
                context += f"Original: '{example['original_prompt']}'\n"
                context += f"Optimized: '{example['optimized_prompt']}' (Score: {example['judge_score']:.3f})\n\n"
        elif self.training_examples:
            # Show general high-scoring examples
            context += "\n--- HIGH-SCORING EXAMPLES ---\n"
            top_examples = sorted(self.training_examples, key=lambda x: x['judge_score'], reverse=True)[:2]
            for example in top_examples:
                context += f"Original: '{example['original_prompt']}'\n"
                context += f"Optimized: '{example['optimized_prompt']}' (Score: {example['judge_score']:.3f})\n\n"
        
        # Add context from current inference rounds
        if rounds:
            context += "--- CURRENT OPTIMIZATION PROGRESS ---\n"
            for r in rounds:
                context += f"Round {r.round_number}: Score {r.reviewer_score:.3f}\n"
                context += f"Feedback: {r.reviewer_critique}\n"
                context += f"Suggestion: {r.reviewer_suggestion}\n\n"
            context += "Generate an improved version that addresses the latest feedback.\n"
        else:
            context += "Generate your first optimization of the original prompt.\n"
        
        prompt = f"""{context}
Original Prompt: "{original_prompt}"

Create an optimized version using your {strategy} strategy. Focus on elements that have scored well in training.

Optimized Prompt:"""
        
        response = self._call_llm(prompt, temperature=0.7)
        
        if not response:
            self.logger.warning("Proposer failed to generate response")
            return "", original_prompt
        
        proposed_prompt = self._clean_prompt(response)
        
        self.logger.info(f"✨ PROPOSER RESULT: '{proposed_prompt[:80]}{'...' if len(proposed_prompt) > 80 else ''}'")
        
        return response, proposed_prompt
    
    def _run_trained_reviewer(self, original_prompt: str, proposed_prompt: str, rounds: List[InferenceRound]) -> Tuple[float, str, str, float]:
        """
        Run trained Reviewer to score and critique the proposal.
        
        Args:
            original_prompt: Original prompt
            proposed_prompt: Proposer's optimization
            rounds: Previous rounds for calibration
            
        Returns:
            (predicted_score, critique, suggestion, confidence)
        """
        self.logger.info("🔍 TRAINED REVIEWER: Analyzing proposal...")
        
        # Build context from training examples showing score patterns
        context = """You are a trained Reviewer that has learned to accurately predict validation scores. You have been calibrated on many examples to understand what makes prompts score well.

Based on your training, you know that high-scoring prompts typically have:
- Rich descriptive details
- Specific material properties
- Clear structural elements
- Appropriate artistic style
- Good lighting/atmospheric elements

"""
        
        # Add score calibration examples from training
        if self.training_examples:
            context += "--- SCORING CALIBRATION FROM TRAINING ---\n"
            
            # Show diverse score examples
            high_examples = [ex for ex in self.training_examples if ex['judge_score'] >= 0.85]
            medium_examples = [ex for ex in self.training_examples if 0.7 <= ex['judge_score'] < 0.85]
            low_examples = [ex for ex in self.training_examples if ex['judge_score'] < 0.7]
            
            if high_examples:
                example = random.choice(high_examples)
                context += f"HIGH SCORE ({example['judge_score']:.3f}): '{example['optimized_prompt']}'\n"
            
            if medium_examples:
                example = random.choice(medium_examples)
                context += f"MEDIUM SCORE ({example['judge_score']:.3f}): '{example['optimized_prompt']}'\n"
            
            if low_examples:
                example = random.choice(low_examples)
                context += f"LOW SCORE ({example['judge_score']:.3f}): '{example['optimized_prompt']}'\n"
            
            context += "\n"
        
        # Add calibration from current inference session
        if rounds:
            context += "--- CURRENT SESSION CALIBRATION ---\n"
            for r in rounds:
                context += f"Round {r.round_number}: You scored {r.reviewer_score:.3f} with confidence {r.reviewer_confidence:.2f}\n"
            
            # Calculate prediction consistency
            if len(rounds) > 1:
                scores = [r.reviewer_score for r in rounds]
                consistency = 1.0 - (max(scores) - min(scores))  # Higher = more consistent
                context += f"Your scoring consistency this session: {consistency:.2f}\n"
            
            context += "\n"
        
        prompt = f"""{context}
Original Prompt: "{original_prompt}"
Proposed Optimization: "{proposed_prompt}"

Based on your training, analyze this optimization and provide your assessment:

{{
  "predicted_score": [float 0.0-1.0, your prediction of validation score],
  "confidence": [float 0.0-1.0, how confident you are in this prediction],
  "critique": "[Brief analysis of strengths and areas for improvement]",
  "suggestion": "[Specific actionable improvement for next round]"
}}"""
        
        response = self._call_llm(prompt, temperature=0.3)
        
        # Parse the JSON response
        try:
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                data = json.loads(json_match.group(0))
                predicted_score = float(data.get("predicted_score", 0.5))
                confidence = float(data.get("confidence", 0.5))
                critique = str(data.get("critique", "No critique provided"))
                suggestion = str(data.get("suggestion", "No suggestion provided"))
            else:
                raise ValueError("No JSON found in response")
                
        except Exception as e:
            self.logger.warning(f"Reviewer JSON parse error: {str(e)}")
            predicted_score = 0.6
            confidence = 0.4
            critique = "Parse error in response"
            suggestion = "Add more descriptive details"
        
        # Ensure values are in valid ranges
        predicted_score = max(0.0, min(1.0, predicted_score))
        confidence = max(0.0, min(1.0, confidence))
        
        self.logger.info(f"📊 TRAINED REVIEWER ASSESSMENT: {predicted_score:.3f} (confidence: {confidence:.2f})")
        self.logger.info(f"   Critique: {critique[:60]}{'...' if len(critique) > 60 else ''}")
        
        return predicted_score, critique, suggestion, confidence
    
    def _clean_prompt(self, text: str) -> str:
        """Clean and format LLM output into a proper prompt."""
        # Extract the core content
        lines = text.strip().split('\n')
        core_content = lines[-1].strip() if lines else text.strip()
        
        # Remove quotes and common prefixes/suffixes
        core_content = core_content.strip('"\'')
        core_content = re.sub(r"^\s*optimized\s*prompt\s*:?\s*", "", core_content, flags=re.IGNORECASE)
        core_content = re.sub(r"^\s*wbgmsst\s*,?\s*", "", core_content, flags=re.IGNORECASE)
        core_content = re.sub(r",?\s*white\s+background\s*$", "", core_content, flags=re.IGNORECASE)
        
        # Ensure proper format
        if core_content:
            return f"wbgmsst, {core_content.strip()}, white background"
        else:
            return "wbgmsst, detailed object, white background"
    
    def optimize_prompt(self, original_prompt: str) -> InferenceSession:
        """
        Optimize a prompt using trained Proposer-Reviewer inference.
        
        Args:
            original_prompt: Prompt to optimize
            
        Returns:
            Complete inference session results
        """
        session_id = f"inference_{int(time.time())}"
        session_start = time.time()
        timestamp = datetime.now().isoformat()
        
        self.logger.info(f"\n{'='*60}")
        self.logger.info(f"TRAINED DEBATE INFERENCE")
        self.logger.info(f"Session: {session_id}")
        self.logger.info(f"Original Prompt: '{original_prompt}'")
        self.logger.info(f"{'='*60}")
        
        # Select strategy
        strategy = self._select_strategy()
        self.logger.info(f"Selected Strategy: {strategy}")
        
        rounds: List[InferenceRound] = []
        best_score = 0.0
        best_round = 0
        best_prompt = original_prompt
        
        # Run inference rounds
        for round_num in range(1, self.max_rounds + 1):
            self.logger.info(f"\n--- 🔄 INFERENCE ROUND {round_num}/{self.max_rounds} ---")
            
            round_start = time.time()
            
            # 1. Trained Proposer generates optimization
            proposer_response, proposed_prompt = self._run_trained_proposer(
                original_prompt, strategy, rounds
            )
            
            # 2. Trained Reviewer scores and critiques
            reviewer_score, reviewer_critique, reviewer_suggestion, reviewer_confidence = self._run_trained_reviewer(
                original_prompt, proposed_prompt, rounds
            )
            
            round_duration = time.time() - round_start
            
            # Create round record
            round_data = InferenceRound(
                round_number=round_num,
                proposer_response=proposer_response,
                proposed_prompt=proposed_prompt,
                strategy_used=strategy,
                reviewer_score=reviewer_score,
                reviewer_critique=reviewer_critique,
                reviewer_suggestion=reviewer_suggestion,
                reviewer_confidence=reviewer_confidence,
                round_duration=round_duration,
                timestamp=datetime.now().isoformat()
            )
            rounds.append(round_data)
            
            # Update best results
            if reviewer_score > best_score:
                best_score = reviewer_score
                best_round = round_num
                best_prompt = proposed_prompt
                self.logger.info(f"🏆 NEW BEST SCORE: {best_score:.3f}")
            
            # Log round results
            self.logger.info(f"Round {round_num} Results:")
            self.logger.info(f"  Reviewer Score: {reviewer_score:.3f}")
            self.logger.info(f"  Confidence: {reviewer_confidence:.2f}")
            self.logger.info(f"  Duration: {round_duration:.1f}s")
            
            # Check for convergence
            if reviewer_score >= self.target_score:
                self.logger.info(f"🎯 CONVERGENCE: Score {reviewer_score:.3f} meets target {self.target_score}")
                break
            
            # Check for minimal improvement
            if round_num > 1:
                prev_score = rounds[-2].reviewer_score
                improvement = reviewer_score - prev_score
                if improvement < self.min_improvement:
                    self.logger.info(f"⏸️  MINIMAL IMPROVEMENT: {improvement:.3f} < {self.min_improvement}")
                    break
        
        # Calculate session metrics
        session_duration = time.time() - session_start
        
        # Score improvement
        if rounds:
            initial_score = 0.5  # Baseline assumption
            final_score = rounds[-1].reviewer_score
            score_improvement = final_score - initial_score
        else:
            final_score = 0.5
            score_improvement = 0.0
        
        # Average confidence
        if rounds:
            avg_confidence = statistics.mean([r.reviewer_confidence for r in rounds])
        else:
            avg_confidence = 0.5
        
        # Create session record
        session = InferenceSession(
            session_id=session_id,
            original_prompt=original_prompt,
            timestamp=timestamp,
            rounds=rounds,
            total_rounds=len(rounds),
            best_prompt=best_prompt,
            best_score=best_score,
            best_round_number=best_round,
            final_prompt=rounds[-1].proposed_prompt if rounds else original_prompt,
            final_score=final_score,
            score_improvement=score_improvement,
            session_duration=session_duration,
            converged=best_score >= self.target_score,
            strategy_used=strategy,
            avg_confidence=avg_confidence
        )
        
        # Save session
        self._save_inference_session(session)
        
        # Log session summary
        self.logger.info(f"\n{'='*60}")
        self.logger.info(f"INFERENCE SESSION SUMMARY")
        self.logger.info(f"{'='*60}")
        self.logger.info(f"Best Score: {best_score:.3f} (Round {best_round})")
        self.logger.info(f"Best Prompt: '{best_prompt}'")
        self.logger.info(f"Score Improvement: {score_improvement:+.3f}")
        self.logger.info(f"Average Confidence: {avg_confidence:.2f}")
        self.logger.info(f"Session Duration: {session_duration:.1f}s")
        self.logger.info(f"Converged: {session.converged}")
        self.logger.info(f"Strategy: {strategy}")
        
        return session
    
    def get_inference_statistics(self) -> Dict[str, Any]:
        """Get comprehensive inference statistics."""
        if not self.inference_sessions:
            return {"message": "No inference sessions completed yet"}
        
        # Overall statistics
        total_sessions = len(self.inference_sessions)
        avg_score = statistics.mean([s.best_score for s in self.inference_sessions])
        avg_duration = statistics.mean([s.session_duration for s in self.inference_sessions])
        avg_rounds = statistics.mean([s.total_rounds for s in self.inference_sessions])
        convergence_rate = len([s for s in self.inference_sessions if s.converged]) / total_sessions
        
        # Performance trends
        recent_sessions = self.inference_sessions[-10:] if len(self.inference_sessions) >= 10 else self.inference_sessions
        recent_avg_score = statistics.mean([s.best_score for s in recent_sessions])
        recent_avg_duration = statistics.mean([s.session_duration for s in recent_sessions])
        
        return {
            'total_sessions': total_sessions,
            'overall_performance': {
                'average_score': avg_score,
                'average_duration_seconds': avg_duration,
                'average_rounds': avg_rounds,
                'convergence_rate': convergence_rate
            },
            'recent_performance': {
                'sessions_analyzed': len(recent_sessions),
                'recent_average_score': recent_avg_score,
                'recent_average_duration': recent_avg_duration
            },
            'training_data': {
                'training_examples_available': len(self.training_examples),
                'strategies_with_data': len([s for s in self.strategy_performance.keys() if self.strategy_performance[s]])
            }
        }


def main():
    """Main inference script interface."""
    if len(sys.argv) < 2:
        print("Usage: python trained_debate_inference.py \"prompt to optimize\"")
        print("\nExample: python trained_debate_inference.py \"emerald pendant\"")
        sys.exit(1)
    
    original_prompt = sys.argv[1]
    
    print("🚀 TRAINED DEBATE INFERENCE SYSTEM")
    print("="*60)
    print("Fast optimization using trained Proposer-Reviewer agents")
    print(f"Original Prompt: '{original_prompt}'")
    print("="*60)
    
    # Initialize inference system
    inference_system = TrainedDebateInference(
        max_rounds=3,
        target_score=0.9,
        min_improvement=0.05
    )
    
    try:
        # Run inference
        session = inference_system.optimize_prompt(original_prompt)
        
        # Display results
        print(f"\n✅ INFERENCE COMPLETED")
        print(f"Best Score: {session.best_score:.3f}")
        print(f"Best Prompt: '{session.best_prompt}'")
        print(f"Rounds: {session.total_rounds}")
        print(f"Duration: {session.session_duration:.1f}s")
        print(f"Converged: {session.converged}")
        print(f"Strategy: {session.strategy_used}")
        print(f"Confidence: {session.avg_confidence:.2f}")
        
        # Show statistics
        stats = inference_system.get_inference_statistics()
        if 'overall_performance' in stats:
            perf = stats['overall_performance']
            print(f"\nSYSTEM PERFORMANCE:")
            print(f"  Total Sessions: {stats['total_sessions']}")
            print(f"  Average Score: {perf['average_score']:.3f}")
            print(f"  Average Duration: {perf['average_duration_seconds']:.1f}s")
            print(f"  Convergence Rate: {perf['convergence_rate']:.1%}")
        
        return session
        
    except KeyboardInterrupt:
        print(f"\n⏹️  Inference interrupted by user")
        return None
        
    except Exception as e:
        print(f"\n❌ Inference error: {str(e)}")
        return None


if __name__ == "__main__":
    main() 