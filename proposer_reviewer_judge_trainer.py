#!/usr/bin/env python3
"""
Proposer-Reviewer-Judge Training System
======================================

This system implements a sophisticated 3-agent training loop:

1. PROPOSER (Creative Agent): Generates optimized prompts using learned strategies
2. REVIEWER (Analytical Agent): Critiques proposals and predicts validation scores  
3. JUDGE (Ground Truth): External validator providing definitive scores for training

The key innovation: The Judge trains the Reviewer to become a reliable proxy for
the external validator, enabling fast validation-free inference later.

Training Process:
- Proposer suggests optimized prompts based on strategy and feedback
- Reviewer critiques and predicts Judge's score 
- Judge provides ground truth via external validation
- Both agents learn from Judge's feedback to improve future performance
- Successful debates are saved as training examples

After training, the Reviewer can replace the Judge for fast inference.
"""

import json
import requests
import time
import sys
import random
import re
import os
import statistics
import subprocess
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path
from dataclasses import dataclass, asdict
import logging
import textwrap


@dataclass
class DebateRound:
    """Stores complete information for a single round of Proposer-Reviewer-Judge debate."""
    round_number: int
    
    # Proposer data
    proposer_prompt: str
    proposer_response: str
    proposed_prompt: str
    strategy_used: str
    
    # Reviewer data  
    reviewer_prompt: str
    reviewer_response: str
    predicted_score: float
    critique: str
    suggestion: str
    confidence: float
    
    # Judge data (ground truth)
    judge_score: Optional[float] = None
    judge_validation_time: Optional[float] = None
    judge_full_results: Optional[Dict] = None
    
    # Round metadata
    timestamp: str = ""
    reviewer_accuracy: Optional[float] = None  # |predicted - actual|


@dataclass 
class TrainingSession:
    """Complete training session data for one original prompt."""
    session_id: str
    original_prompt: str
    timestamp: str
    
    # Debate rounds
    rounds: List[DebateRound]
    total_rounds: int
    
    # Final results
    best_prompt: str
    best_judge_score: float
    best_round_number: int
    
    # Learning metrics
    proposer_improvement: float  # Score improvement over rounds
    reviewer_accuracy_improvement: float  # Prediction accuracy improvement
    total_validation_time: float
    session_duration: float
    
    # Success indicators
    converged: bool
    met_quality_threshold: bool
    saved_to_memory: bool


class JudgeValidator:
    """
    The Judge agent that provides ground truth validation scores.
    
    This is the external validator that both Proposer and Reviewer learn from.
    During training, the Judge is the authoritative source of prompt quality.
    """
    
    def __init__(self, conda_path: str = "/home/mbhat/miniconda/bin/activate",
                 conda_env: str = "trellis_new",
                 validator_script: str = "subnet_accurate_validator.py",
                 results_file: str = "subnet_validation_results.json"):
        """
        Initialize the Judge with validation environment settings.
        
        Args:
            conda_path: Path to conda activation script
            conda_env: Conda environment name  
            validator_script: Validation script filename
            results_file: JSON results file from validator
        """
        self.conda_path = conda_path
        self.conda_env = conda_env
        self.validator_script = validator_script
        self.results_file = results_file
        
        # Validation statistics
        self.total_validations = 0
        self.total_validation_time = 0.0
        self.failed_validations = 0
        
        self.logger = logging.getLogger(__name__)
        
    def validate_prompt(self, prompt: str) -> Optional[Dict[str, Any]]:
        """
        Run external validation and return complete results.
        
        Args:
            prompt: Prompt to validate
            
        Returns:
            Complete validation results dict or None if failed
        """
        self.logger.info(f"⚖️  JUDGE: Validating prompt...")
        start_time = time.time()
        
        try:
            # Construct validation command
            cmd = [
                "bash", "-c",
                f"source {self.conda_path} && conda activate {self.conda_env} && python {self.validator_script} \"{prompt}\""
            ]
            
            # Run validation with timeout
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
            
            if result.returncode != 0:
                self.logger.error(f"Judge validation failed (Code {result.returncode})")
                self.logger.error(f"Stderr: {result.stderr[-200:]}")
                self.failed_validations += 1
                return None
            
            # Read validation results
            if not os.path.exists(self.results_file):
                self.logger.error(f"Judge results file not found: {self.results_file}")
                self.failed_validations += 1
                return None
                
            with open(self.results_file, 'r') as f:
                data = json.load(f)
            
            validation_time = time.time() - start_time
            score = data.get('validation_engine_score', 0.0)
            
            # Update statistics
            self.total_validations += 1
            self.total_validation_time += validation_time
            
            self.logger.info(f"✅ JUDGE VERDICT: {score:.4f} (in {validation_time:.1f}s)")
            
            # Add metadata to results
            data['judge_validation_time'] = validation_time
            data['judge_validation_count'] = self.total_validations
            
            return data
            
        except subprocess.TimeoutExpired:
            self.logger.error("Judge validation timed out")
            self.failed_validations += 1
            return None
        except Exception as e:
            self.logger.error(f"Judge validation error: {str(e)}")
            self.failed_validations += 1
            return None
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get Judge validation statistics."""
        avg_time = self.total_validation_time / max(1, self.total_validations)
        success_rate = (self.total_validations - self.failed_validations) / max(1, self.total_validations)
        
        return {
            'total_validations': self.total_validations,
            'failed_validations': self.failed_validations,
            'success_rate': success_rate,
            'total_validation_time': self.total_validation_time,
            'average_validation_time': avg_time
        }


class ProposerReviewerJudgeTrainer:
    """
    Main training orchestrator for the Proposer-Reviewer-Judge system.
    
    This class manages the 3-agent debate loop and learning from ground truth.
    """
    
    def __init__(self, 
                 ollama_url: str = "http://localhost:11434",
                 model: str = "llama3.2:3b",
                 memory_file: str = "prj_training_memory.json",
                 max_rounds: int = 4,
                 quality_threshold: float = 0.8,
                 convergence_threshold: float = 0.9):
        """
        Initialize the Proposer-Reviewer-Judge trainer.
        
        Args:
            ollama_url: Ollama server URL
            model: LLM model for Proposer and Reviewer
            memory_file: File to store successful training sessions
            max_rounds: Maximum debate rounds per session
            quality_threshold: Minimum score to save session to memory
            convergence_threshold: Score threshold for early stopping
        """
        self.ollama_url = ollama_url
        self.model = model
        self.memory_file = Path(memory_file)
        self.max_rounds = max_rounds
        self.quality_threshold = quality_threshold
        self.convergence_threshold = convergence_threshold
        
        # Initialize components
        self.judge = JudgeValidator()
        self.training_memory: List[TrainingSession] = []
        
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
        
        # Learning tracking
        self.strategy_performance: Dict[str, List[float]] = {s: [] for s in self.strategies}
        self.reviewer_calibration_history: List[float] = []
        
        # Setup logging
        log_dir = "training_logs"
        os.makedirs(log_dir, exist_ok=True)
        log_file = os.path.join(log_dir, f"prj_training_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        
        # Load existing memory
        self._load_training_memory()
        
        self.logger.info("🧠 PROPOSER-REVIEWER-JUDGE TRAINER INITIALIZED")
        self.logger.info(f"   Model: {self.model}")
        self.logger.info(f"   Max rounds per session: {self.max_rounds}")
        self.logger.info(f"   Quality threshold: {self.quality_threshold}")
        self.logger.info(f"   Convergence threshold: {self.convergence_threshold}")
        self.logger.info(f"   Loaded training sessions: {len(self.training_memory)}")
    
    def _load_training_memory(self):
        """Load existing training sessions from memory file."""
        if not self.memory_file.exists():
            self.logger.info("📄 Starting fresh training memory")
            return
            
        try:
            with open(self.memory_file, 'r') as f:
                data = json.load(f)
            
            # Reconstruct sessions from saved data
            self.training_memory = []
            for session_data in data.get('training_sessions', []):
                # Reconstruct rounds
                rounds = []
                for round_data in session_data.get('rounds', []):
                    rounds.append(DebateRound(**round_data))
                
                session_data['rounds'] = rounds
                self.training_memory.append(TrainingSession(**session_data))
            
            # Load strategy performance
            self.strategy_performance = data.get('strategy_performance', {s: [] for s in self.strategies})
            self.reviewer_calibration_history = data.get('reviewer_calibration_history', [])
            
            self.logger.info(f"📚 Loaded {len(self.training_memory)} training sessions")
            
        except Exception as e:
            self.logger.error(f"Error loading training memory: {str(e)}")
            self.logger.info("📄 Starting fresh training memory")
    
    def _save_training_memory(self):
        """Save current training memory and statistics."""
        try:
            memory_data = {
                'training_sessions': [asdict(session) for session in self.training_memory],
                'strategy_performance': self.strategy_performance,
                'reviewer_calibration_history': self.reviewer_calibration_history,
                'training_statistics': {
                    'total_sessions': len(self.training_memory),
                    'successful_sessions': len([s for s in self.training_memory if s.met_quality_threshold]),
                    'average_session_score': statistics.mean([s.best_judge_score for s in self.training_memory]) if self.training_memory else 0,
                    'judge_statistics': self.judge.get_statistics()
                },
                'last_updated': datetime.now().isoformat()
            }
            
            with open(self.memory_file, 'w') as f:
                json.dump(memory_data, f, indent=2)
                
            self.logger.info(f"💾 Training memory saved: {len(self.training_memory)} sessions")
            
        except Exception as e:
            self.logger.error(f"Error saving training memory: {str(e)}")
    
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
        """Select optimization strategy based on historical performance."""
        if not any(self.strategy_performance.values()):
            return random.choice(self.strategies)
        
        # Calculate strategy effectiveness
        strategy_scores = {}
        for strategy, scores in self.strategy_performance.items():
            if scores:
                strategy_scores[strategy] = statistics.mean(scores)
            else:
                strategy_scores[strategy] = 0.5  # Default for untested
        
        # Epsilon-greedy selection (75% exploit, 25% explore)
        if random.random() < 0.75 and strategy_scores:
            best_strategy = max(strategy_scores.keys(), key=lambda s: strategy_scores[s])
            return best_strategy
        else:
            return random.choice(self.strategies)
    
    def create_feedback(self, best_prompt: str, best_score: float, agent: str) -> str:
        """Generate actionable feedback for Proposer or Reviewer to focus on beating the best-so-far."""
        if agent == "proposer":
            return textwrap.dedent(f"""
                --- BEST SO FAR ---
                Prompt: '{best_prompt}'
                Judge Score: {best_score:.3f}
                Your goal is to produce a prompt that scores higher than this. BUT REMEMBER, LONGER PROMPT DOESNT MEAN HIGH SCORING OR BETTER SO OPTIMIZE. If you cannot, explain why and try a different approach or strategy. Address all Reviewer and Judge feedback directly in your next attempt.
            """)
        elif agent == "reviewer":
            return textwrap.dedent(f"""
                --- BEST SO FAR ---
                Prompt: '{best_prompt}'
                Judge Score: {best_score:.3f}
                Your feedback and suggestions should focus on how the Proposer can surpass this score. Be specific and actionable. If the new proposal is not better, explain why and suggest concrete improvements.
            """)
        else:
            return ""

    def _run_proposer(self, original_prompt: str, strategy: str, rounds: List[DebateRound]) -> Tuple[str, str, str]:
        """
        Run the Proposer agent to generate an optimized prompt.
        
        Args:
            original_prompt: Original prompt to optimize
            strategy: Strategy to use for optimization
            rounds: Previous rounds for context
            
        Returns:
            (proposer_prompt, proposer_response, proposed_prompt)
        """
        self.logger.info(f"💡 PROPOSER: Generating proposal with strategy '{strategy}'")
        
        # Build context from successful training examples
        context = f"""You are an expert Proposer in a prompt optimization debate. Your goal is to create prompts that will receive high validation scores from the Judge.

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
        
        # Add examples from successful training sessions
        if self.training_memory:
            context += "\n--- EXAMPLES OF HIGH-SCORING OPTIMIZATIONS ---\n"
            high_scoring = sorted(self.training_memory, key=lambda s: s.best_judge_score, reverse=True)[:3]
            for session in high_scoring:
                context += f"Original: '{session.original_prompt}'\n"
                context += f"Winning: '{session.best_prompt}' (Judge Score: {session.best_judge_score:.3f})\n\n"
        
        # Add context from current debate
        if rounds:
            context += "--- CURRENT DEBATE HISTORY ---\n"
            for r in rounds:
                context += f"Round {r.round_number}: Your proposal scored {r.judge_score:.3f} with the Judge.\n"
                context += f"Reviewer feedback: {r.critique}\n"
                context += f"Suggestion: {r.suggestion}\n\n"
            # Add best-so-far feedback
            if rounds:
                best_round = max(rounds, key=lambda r: r.judge_score or 0)
                context += self.create_feedback(best_round.proposed_prompt, best_round.judge_score or 0, agent="proposer")
                context += "Generate an improved proposal that addresses the Reviewer's feedback and aims for a higher Judge score.\n"
        else:
            context += "Generate your first optimization of the original prompt using your assigned strategy.\n"
        
        MAX_CONTEXT_CHARS = 9000
        if len(context) > MAX_CONTEXT_CHARS:
            self.logger.warning(f"Proposer context too long ({len(context)} chars), summarizing older examples/history.")
            context_lines = context.split('\n')
            context = '\n'.join(context_lines[-100:])
            context = "[SUMMARY: Older context omitted for brevity.]\n" + context
        
        prompt = f"""{context}
Original Prompt: "{original_prompt}"

Create an optimized version focusing on your {strategy} strategy. Return only the optimized prompt without explanations.

Optimized Prompt:"""
        
        response = self._call_llm(prompt, temperature=0.7)
        
        if not response:
            self.logger.warning("Proposer failed to generate response")
            return prompt, "", original_prompt
        
        # Clean and format the proposed prompt
        proposed_prompt = self._clean_prompt(response)
        
        # self.logger.info(f"✨ PROPOSER RESULT: '{proposed_prompt[:80]}{'...' if len(proposed_prompt) > 80 else ''}'")
        self.logger.info(f"✨ PROPOSER RESULT: '{proposed_prompt}'")
        return prompt, response, proposed_prompt
    
    def _run_reviewer(self, original_prompt: str, proposed_prompt: str, rounds: List[DebateRound]) -> Tuple[str, str, float, str, str, float]:
        """
        Run the Reviewer agent to critique and predict score.
        
        Args:
            original_prompt: Original prompt
            proposed_prompt: Proposer's suggested optimization
            rounds: Previous rounds for calibration
            
        Returns:
            (reviewer_prompt, reviewer_response, predicted_score, critique, suggestion, confidence)
        """
        self.logger.info("🔍 REVIEWER: Analyzing and scoring proposal...")
        
        # Build context from training examples to teach scoring patterns
        context = """You are an analytical Reviewer. Your goal is to accurately predict the Judge's validation score and provide constructive feedback.

The Judge scores prompts from 0.0 to 1.0 based on how well they will generate high-quality 3D objects.

"""
        
        # Add scoring examples from training memory
        if self.training_memory:
            context += "--- JUDGE'S SCORING PATTERNS ---\n"
            # Show diverse score examples
            high_score = max(self.training_memory, key=lambda s: s.best_judge_score)
            low_scores = [s for s in self.training_memory if s.best_judge_score < 0.7]
            mid_scores = [s for s in self.training_memory if 0.7 <= s.best_judge_score < 0.85]
            
            context += f"HIGH SCORE ({high_score.best_judge_score:.3f}): '{high_score.best_prompt}'\n"
            
            if mid_scores:
                mid_example = random.choice(mid_scores)
                context += f"MEDIUM SCORE ({mid_example.best_judge_score:.3f}): '{mid_example.best_prompt}'\n"
                
            if low_scores:
                low_example = min(low_scores, key=lambda s: s.best_judge_score)
                context += f"LOW SCORE ({low_example.best_judge_score:.3f}): '{low_example.best_prompt}'\n"
            
            context += "\n"
        
        # Add calibration context from current debate
        if rounds:
            context += "--- YOUR PREDICTION ACCURACY ---\n"
            total_error = 0
            for r in rounds:
                if r.judge_score is not None:
                    error = abs(r.predicted_score - r.judge_score)
                    total_error += error
                    context += f"Round {r.round_number}: You predicted {r.predicted_score:.3f}, Judge scored {r.judge_score:.3f} (error: {error:.3f})\n"
            
            if rounds:
                avg_error = total_error / len(rounds)
                context += f"Your current average prediction error: {avg_error:.3f}\n"
                context += "Calibrate your prediction based on your past accuracy.\n\n"
        
        if rounds:
            # Add best-so-far feedback for Reviewer
            best_round = max(rounds, key=lambda r: r.judge_score or 0)
            context += self.create_feedback(best_round.proposed_prompt, best_round.judge_score or 0, agent="reviewer")
        
        MAX_CONTEXT_CHARS = 9000
        if len(context) > MAX_CONTEXT_CHARS:
            self.logger.warning(f"Reviewer context too long ({len(context)} chars), summarizing older examples/history.")
            context_lines = context.split('\n')
            context = '\n'.join(context_lines[-100:])
            context = "[SUMMARY: Older context omitted for brevity.]\n" + context
        
        prompt = f"""{context}
Original Prompt: "{original_prompt}"
Proposer's Optimization: "{proposed_prompt}"

Analyze this optimization and provide your assessment in JSON format:

{{
  "predicted_score": [float 0.0-1.0],
  "confidence": [float 0.0-1.0 indicating how confident you are in your prediction],
  "critique": "[Brief analysis of strengths and weaknesses]",
  "suggestion": "[Specific actionable improvement for next round]"
}}"""
        
        response = self._call_llm(prompt, temperature=0.3)
        
        # Parse the JSON response, with retry on parse error
        for attempt in range(2):
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
                break  # Success
            except Exception as e:
                self.logger.warning(f"Reviewer JSON parse error: {str(e)} | Raw response: {response[:200]}")
                if attempt == 0:
                    self.logger.info("Retrying Reviewer LLM call due to parse error...")
                    response = self._call_llm(prompt, temperature=0.3)
                    continue
                predicted_score = 0.5
                confidence = 0.3
                critique = "Parse error in response"
                suggestion = "Add more descriptive details"
        
        predicted_score = max(0.0, min(1.0, predicted_score))
        confidence = max(0.0, min(1.0, confidence))
        
        self.logger.info(f"📊 REVIEWER ASSESSMENT: {predicted_score:.3f} (confidence: {confidence:.2f})")
        # self.logger.info(f"   Critique: {critique[:60]}{'...' if len(critique) > 60 else ''}")
        self.logger.info(f"   Critique: {critique}")
        
        return prompt, response, predicted_score, critique, suggestion, confidence
    
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
    
    def train_on_prompt(self, original_prompt: str) -> TrainingSession:
        """
        Run a complete training session on a single prompt.
        
        Args:
            original_prompt: Prompt to optimize and learn from
            
        Returns:
            Complete training session data
        """
        session_id = f"prj_session_{int(time.time())}"
        session_start = time.time()
        timestamp = datetime.now().isoformat()
        
        self.logger.info(f"\n{'='*60}")
        self.logger.info(f"STARTING PROPOSER-REVIEWER-JUDGE TRAINING")
        self.logger.info(f"Session: {session_id}")
        self.logger.info(f"Original Prompt: '{original_prompt}'")
        self.logger.info(f"{'='*60}")
        
        # Select strategy for this session
        strategy = self._select_strategy()
        self.logger.info(f"Selected Strategy: {strategy}")
        
        rounds: List[DebateRound] = []
        best_score = 0.0
        best_round = 0
        best_prompt = original_prompt
        total_validation_time = 0.0
        
        # Run debate rounds
        for round_num in range(1, self.max_rounds + 1):
            self.logger.info(f"\n--- 🏛️  DEBATE ROUND {round_num}/{self.max_rounds} ---")
            
            round_start = time.time()
            
            # 1. Proposer generates optimization
            proposer_prompt, proposer_response, proposed_prompt = self._run_proposer(
                original_prompt, strategy, rounds
            )
            
            # 2. Reviewer analyzes and predicts
            reviewer_prompt, reviewer_response, predicted_score, critique, suggestion, confidence = self._run_reviewer(
                original_prompt, proposed_prompt, rounds
            )
            
            # 3. Judge provides ground truth
            judge_results = self.judge.validate_prompt(proposed_prompt)
            if judge_results is None:
                self.logger.error("Judge validation failed - ending session")
                break
            
            judge_score = judge_results.get('validation_engine_score', 0.0)
            judge_time = judge_results.get('judge_validation_time', 0.0)
            total_validation_time += judge_time
            
            # Calculate reviewer accuracy
            reviewer_accuracy = abs(predicted_score - judge_score)
            
            # Create round record
            round_data = DebateRound(
                round_number=round_num,
                proposer_prompt=proposer_prompt,
                proposer_response=proposer_response,
                proposed_prompt=proposed_prompt,
                strategy_used=strategy,
                reviewer_prompt=reviewer_prompt,
                reviewer_response=reviewer_response,
                predicted_score=predicted_score,
                critique=critique,
                suggestion=suggestion,
                confidence=confidence,
                judge_score=judge_score,
                judge_validation_time=judge_time,
                judge_full_results=judge_results,
                timestamp=datetime.now().isoformat(),
                reviewer_accuracy=reviewer_accuracy
            )
            rounds.append(round_data)
            
            # Update best results
            if judge_score > best_score:
                best_score = judge_score
                best_round = round_num
                best_prompt = proposed_prompt
                self.logger.info(f"🏆 NEW BEST SCORE: {best_score:.3f}")
            
            # Log round results
            self.logger.info(f"Round {round_num} Results:")
            self.logger.info(f"  Judge Score: {judge_score:.3f}")
            self.logger.info(f"  Reviewer Predicted: {predicted_score:.3f}")
            self.logger.info(f"  Reviewer Accuracy: {reviewer_accuracy:.3f}")
            self.logger.info(f"  Time: {time.time() - round_start:.1f}s")
            
            # Check for convergence
            if judge_score >= self.convergence_threshold:
                self.logger.info(f"🎯 CONVERGENCE: Score {judge_score:.3f} exceeds threshold {self.convergence_threshold}")
                break
        
        # Calculate learning metrics
        session_duration = time.time() - session_start
        
        # Proposer improvement (score progression)
        if len(rounds) > 1:
            first_score = rounds[0].judge_score or 0.0
            last_score = rounds[-1].judge_score or 0.0
            proposer_improvement = last_score - first_score
        else:
            proposer_improvement = 0.0
        
        # Reviewer accuracy improvement
        if len(rounds) > 1:
            first_accuracy = rounds[0].reviewer_accuracy or 1.0
            last_accuracy = rounds[-1].reviewer_accuracy or 1.0
            reviewer_accuracy_improvement = first_accuracy - last_accuracy  # Positive = improvement
        else:
            reviewer_accuracy_improvement = 0.0
        
        # Create session record
        session = TrainingSession(
            session_id=session_id,
            original_prompt=original_prompt,
            timestamp=timestamp,
            rounds=rounds,
            total_rounds=len(rounds),
            best_prompt=best_prompt,
            best_judge_score=best_score,
            best_round_number=best_round,
            proposer_improvement=proposer_improvement,
            reviewer_accuracy_improvement=reviewer_accuracy_improvement,
            total_validation_time=total_validation_time,
            session_duration=session_duration,
            converged=best_score >= self.convergence_threshold,
            met_quality_threshold=best_score >= self.quality_threshold,
            saved_to_memory=False  # Will be updated below
        )
        
        # Update learning statistics
        self.strategy_performance[strategy].append(best_score)
        if rounds:
            avg_reviewer_accuracy = statistics.mean([r.reviewer_accuracy for r in rounds if r.reviewer_accuracy is not None])
            self.reviewer_calibration_history.append(avg_reviewer_accuracy)
        
        # Save successful sessions to memory
        if session.met_quality_threshold:
            self.training_memory.append(session)
            session.saved_to_memory = True
            self._save_training_memory()
            self.logger.info(f"✅ Session saved to memory (Score: {best_score:.3f})")
        else:
            self.logger.info(f"⚠️  Session below quality threshold (Score: {best_score:.3f})")
        
        # Log session summary
        self.logger.info(f"\n{'='*60}")
        self.logger.info(f"SESSION SUMMARY")
        self.logger.info(f"{'='*60}")
        self.logger.info(f"Best Score: {best_score:.3f} (Round {best_round})")
        self.logger.info(f"Best Prompt: '{best_prompt}'")
        self.logger.info(f"Proposer Improvement: {proposer_improvement:+.3f}")
        self.logger.info(f"Reviewer Accuracy Improvement: {reviewer_accuracy_improvement:+.3f}")
        self.logger.info(f"Total Validation Time: {total_validation_time:.1f}s")
        self.logger.info(f"Session Duration: {session_duration:.1f}s")
        self.logger.info(f"Converged: {session.converged}")
        self.logger.info(f"Saved to Memory: {session.saved_to_memory}")
        
        return session
    
    def get_training_statistics(self) -> Dict[str, Any]:
        """Get comprehensive training statistics."""
        if not self.training_memory:
            return {"message": "No training sessions completed yet"}
        
        # Overall statistics
        total_sessions = len(self.training_memory)
        successful_sessions = len([s for s in self.training_memory if s.met_quality_threshold])
        avg_score = statistics.mean([s.best_judge_score for s in self.training_memory])
        
        # Strategy performance
        strategy_stats = {}
        for strategy, scores in self.strategy_performance.items():
            if scores:
                strategy_stats[strategy] = {
                    'count': len(scores),
                    'average_score': statistics.mean(scores),
                    'best_score': max(scores)
                }
        
        # Reviewer calibration
        if self.reviewer_calibration_history:
            recent_accuracy = statistics.mean(self.reviewer_calibration_history[-10:])
            overall_accuracy = statistics.mean(self.reviewer_calibration_history)
            accuracy_trend = recent_accuracy - overall_accuracy
        else:
            recent_accuracy = overall_accuracy = accuracy_trend = 0.0
        
        # Judge statistics
        judge_stats = self.judge.get_statistics()
        
        return {
            'training_sessions': {
                'total': total_sessions,
                'successful': successful_sessions,
                'success_rate': successful_sessions / max(1, total_sessions),
                'average_score': avg_score
            },
            'strategy_performance': strategy_stats,
            'reviewer_calibration': {
                'recent_accuracy': recent_accuracy,
                'overall_accuracy': overall_accuracy,
                'accuracy_trend': accuracy_trend,
                'total_predictions': len(self.reviewer_calibration_history)
            },
            'judge_statistics': judge_stats
        }


def main():
    """Main training script interface."""
    if len(sys.argv) < 2:
        print("Usage: python proposer_reviewer_judge_trainer.py \"prompt to train on\"")
        print("\nExample: python proposer_reviewer_judge_trainer.py \"emerald pendant\"")
        sys.exit(1)
    
    original_prompt = sys.argv[1]
    
    print("🧠 PROPOSER-REVIEWER-JUDGE TRAINING SYSTEM")
    print("="*60)
    print("Training both Proposer and Reviewer using Judge ground truth")
    print(f"Original Prompt: '{original_prompt}'")
    print("="*60)
    
    # Initialize trainer
    trainer = ProposerReviewerJudgeTrainer(
        max_rounds=4,
        quality_threshold=0.8,
        convergence_threshold=0.9
    )
    
    try:
        # Run training session
        session = trainer.train_on_prompt(original_prompt)
        
        # Display results
        print(f"\n✅ TRAINING SESSION COMPLETED")
        print(f"Final Score: {session.best_judge_score:.3f}")
        print(f"Best Prompt: '{session.best_prompt}'")
        print(f"Rounds: {session.total_rounds}")
        print(f"Converged: {session.converged}")
        print(f"Saved to Memory: {session.saved_to_memory}")
        
        # Show training statistics
        stats = trainer.get_training_statistics()
        if 'training_sessions' in stats:
            ts = stats['training_sessions']
            print(f"\nTRAINING PROGRESS:")
            print(f"  Total Sessions: {ts['total']}")
            print(f"  Success Rate: {ts['success_rate']:.1%}")
            print(f"  Average Score: {ts['average_score']:.3f}")
        
        return session
        
    except KeyboardInterrupt:
        print(f"\n⏹️  Training interrupted by user")
        return None
        
    except Exception as e:
        print(f"\n❌ Training error: {str(e)}")
        return None


if __name__ == "__main__":
    main() 