#!/usr/bin/env python3
"""
Conversational Debate Optimizer (v5.0)

This system implements a sophisticated Proposer-Reviewer architecture where two specialized
LLM personas engage in structured dialogue to iteratively refine prompt optimizations.

Key Innovation: Instead of relying on external validation, we use internal conversation
between a creative Proposer and analytical Reviewer to achieve high-quality optimizations
with reliable confidence scoring.

Architecture:
- Proposer: Creative optimization agent with strategy-based generation
- Reviewer: Analytical quality assessment agent with scoring and feedback
- Debate Loop: Iterative refinement through structured conversation
- Memory Integration: Learns from historical optimization data
"""

import json
import os
import time
import requests
import statistics
from datetime import datetime
from typing import Dict, List, Any, Tuple, Optional
from pathlib import Path
import logging


class OptimizationAttempt:
    """Represents a single optimization attempt with metadata"""
    
    def __init__(self, prompt: str, strategy: str, score: float, 
                 critique: str = "", suggestion: str = ""):
        self.prompt = prompt
        self.strategy = strategy
        self.score = score
        self.critique = critique
        self.suggestion = suggestion
        self.timestamp = datetime.now().isoformat()


class ConversationalDebateOptimizer:
    """
    Advanced prompt optimizer using Proposer-Reviewer conversational debate.
    
    This system creates a structured dialogue between two specialized AI personas:
    1. Proposer: Creative optimization agent that generates improved prompts
    2. Reviewer: Analytical assessment agent that scores and critiques proposals
    
    The debate loop runs for multiple rounds until convergence or max rounds reached.
    """
    
    def __init__(self, 
                 ollama_url: str = "http://localhost:11434",
                 model: str = "llama3.2:3b",
                 memory_file: str = "debate_memory.json",
                 max_debate_rounds: int = 3,
                 target_score: float = 0.9,
                 min_improvement: float = 0.05):
        """
        Initialize the conversational debate optimizer.
        
        Args:
            ollama_url: Ollama server URL
            model: LLM model to use for both personas
            memory_file: File to store optimization history and learning
            max_debate_rounds: Maximum rounds of Proposer-Reviewer dialogue
            target_score: Target score to achieve (stops early if reached)
            min_improvement: Minimum score improvement to continue iterating
        """
        # Setup logging and logger FIRST
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        self.ollama_url = ollama_url
        self.model = model
        self.memory_file = Path(memory_file)
        self.max_debate_rounds = max_debate_rounds
        self.target_score = target_score
        self.min_improvement = min_improvement
        
        # Optimization strategies for the Proposer
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
        
        # Learning memory
        self.optimization_history: List[OptimizationAttempt] = []
        self.strategy_performance: Dict[str, List[float]] = {strategy: [] for strategy in self.strategies}
        self.quality_examples: Dict[str, List[str]] = {"high_quality": [], "low_quality": []}
        
        # Load existing memory
        self._load_memory()
        
        self.logger.info("🗣️  CONVERSATIONAL DEBATE OPTIMIZER INITIALIZED")
        self.logger.info(f"   Model: {self.model}")
        self.logger.info(f"   Max debate rounds: {self.max_debate_rounds}")
        self.logger.info(f"   Target score: {self.target_score}")
        self.logger.info(f"   Historical attempts: {len(self.optimization_history)}")
    
    def _load_memory(self):
        """Load optimization history and learning data"""
        if self.memory_file.exists():
            try:
                with open(self.memory_file, 'r') as f:
                    data = json.load(f)
                
                # Load optimization history
                history_data = data.get('optimization_history', [])
                self.optimization_history = [
                    OptimizationAttempt(**attempt) for attempt in history_data
                ]
                
                # Load strategy performance
                self.strategy_performance = data.get('strategy_performance', 
                                                   {strategy: [] for strategy in self.strategies})
                
                # Load quality examples for Reviewer training
                self.quality_examples = data.get('quality_examples', 
                                                {"high_quality": [], "low_quality": []})
                
                self.logger.info(f"📄 Loaded debate memory: {len(self.optimization_history)} attempts")
                
            except Exception as e:
                self.logger.warning(f"Could not load memory: {str(e)}")
                self._initialize_fresh_memory()
        else:
            self._initialize_fresh_memory()
    
    def _initialize_fresh_memory(self):
        """Initialize fresh memory with empty structures"""
        self.optimization_history = []
        self.strategy_performance = {strategy: [] for strategy in self.strategies}
        self.quality_examples = {"high_quality": [], "low_quality": []}
        self.logger.info("📄 Starting fresh debate memory")
    
    def _save_memory(self):
        """Save current memory state"""
        try:
            memory_data = {
                'optimization_history': [
                    {
                        'prompt': attempt.prompt,
                        'strategy': attempt.strategy, 
                        'score': attempt.score,
                        'critique': attempt.critique,
                        'suggestion': attempt.suggestion,
                        'timestamp': attempt.timestamp
                    } for attempt in self.optimization_history
                ],
                'strategy_performance': self.strategy_performance,
                'quality_examples': self.quality_examples,
                'last_updated': datetime.now().isoformat()
            }
            
            with open(self.memory_file, 'w') as f:
                json.dump(memory_data, f, indent=2)
                
        except Exception as e:
            self.logger.error(f"Could not save memory: {str(e)}")
    
    def _call_ollama(self, system_prompt: str, user_prompt: str) -> str:
        """Make a call to Ollama with error handling"""
        try:
            response = requests.post(f"{self.ollama_url}/api/generate", 
                                   json={
                                       "model": self.model,
                                       "system": system_prompt,
                                       "prompt": user_prompt,
                                       "stream": False
                                   }, timeout=30)
            
            if response.status_code == 200:
                return response.json()["response"].strip()
            else:
                self.logger.error(f"Ollama error: {response.status_code}")
                return ""
                
        except Exception as e:
            self.logger.error(f"Ollama call failed: {str(e)}")
            return ""
    
    def _select_strategy(self) -> str:
        """Select optimization strategy based on historical performance"""
        if not any(self.strategy_performance.values()):
            # No history, random selection
            import random
            return random.choice(self.strategies)
        
        # Calculate average performance per strategy
        strategy_avgs = {}
        for strategy, scores in self.strategy_performance.items():
            if scores:
                strategy_avgs[strategy] = statistics.mean(scores)
            else:
                strategy_avgs[strategy] = 0.5  # Default for untested strategies
        
        # Epsilon-greedy selection (80% exploit best, 20% explore)
        import random
        if random.random() < 0.8:
            # Exploit: choose best performing strategy
            best_strategy = max(strategy_avgs.keys(), key=lambda s: strategy_avgs[s])
            return best_strategy
        else:
            # Explore: random selection
            return random.choice(self.strategies)
    
    def _create_proposer_prompt(self, original_prompt: str, strategy: str, 
                               previous_attempt: str = "", reviewer_feedback: Dict = None) -> str:
        """Create the system and user prompts for the Proposer agent"""
        
        system_prompt = f"""You are an expert prompt optimization agent specializing in creating detailed, vivid descriptions for 3D object generation.

Your current strategy is: {strategy}

Strategy Definitions:
- material_focus: Emphasize material properties, textures, and surface qualities
- artistic_elaboration: Add artistic style, aesthetic qualities, and visual appeal
- technical_precision: Include technical specifications, measurements, and structural details
- atmospheric_enhancement: Add lighting, environment, and mood elements
- structural_detailing: Focus on shape, form, geometry, and architectural elements
- lighting_emphasis: Specify lighting conditions, shadows, and illumination
- texture_specification: Detail surface textures, patterns, and tactile qualities
- contextual_placement: Add environmental context, setting, and background elements

Your task is to transform basic prompts into rich, detailed descriptions that will generate high-quality 3D objects.

IMPORTANT: Stay focused on the core object while enhancing it with your assigned strategy. Do not completely change the object type."""
        
        if reviewer_feedback:
            # Round 2+ with feedback
            user_prompt = f"""Original Prompt: "{original_prompt}"
Your Previous Attempt: "{previous_attempt}"

The Reviewer has provided feedback on your previous attempt:
- Score: {reviewer_feedback.get('score', 0.0)}
- Critique: "{reviewer_feedback.get('critique', '')}"
- Suggestion: "{reviewer_feedback.get('suggestion', '')}"

Generate an improved version that directly addresses the Reviewer's feedback while maintaining your {strategy} strategy focus."""

        else:
            # Round 1, no previous feedback
            user_prompt = f"""Original Prompt: "{original_prompt}"

Using your {strategy} strategy, create an optimized version of this prompt that will generate a much more detailed and visually appealing 3D object.

Focus on enhancing the prompt according to your strategy while keeping the core object intact."""
        
        return system_prompt, user_prompt
    
    def _create_reviewer_prompt(self, original_prompt: str, proposed_prompt: str) -> str:
        """Create the system and user prompts for the Reviewer agent"""
        
        # Build quality examples from memory for Reviewer training
        quality_context = ""
        if self.quality_examples["high_quality"]:
            quality_context += "\nExamples of HIGH-QUALITY optimizations:\n"
            for example in self.quality_examples["high_quality"][:3]:  # Show top 3
                quality_context += f"- {example}\n"
        
        if self.quality_examples["low_quality"]:
            quality_context += "\nExamples of LOW-QUALITY optimizations:\n" 
            for example in self.quality_examples["low_quality"][:3]:  # Show worst 3
                quality_context += f"- {example}\n"
        
        system_prompt = f"""You are a strict, analytical prompt quality reviewer specializing in 3D object generation prompts.

Your goal is to ensure every optimized prompt is a significant improvement over the original that will generate higher-quality 3D objects.

Evaluation Criteria:
1. SPECIFICITY: Does it add meaningful descriptive details?
2. CLARITY: Is it clear what object should be generated?  
3. VISUAL RICHNESS: Will it create a more visually appealing result?
4. COHERENCE: Do all elements work together logically?
5. PRESERVATION: Does it maintain the core object from the original?

Scoring Guide:
- 0.9-1.0: Exceptional enhancement with rich, coherent details
- 0.8-0.9: Strong improvement with good descriptive additions
- 0.7-0.8: Moderate improvement but could be more detailed
- 0.6-0.7: Minor improvement, lacks significant enhancement
- 0.0-0.6: Poor optimization, confusing or minimal improvement

{quality_context}

RESPONSE FORMAT (JSON only):
{{
  "score": [0.0-1.0],
  "critique": "[Brief analysis of strengths and weaknesses]",
  "suggestion": "[Specific, actionable suggestion for improvement]"
}}"""
        
        user_prompt = f"""Original Prompt: "{original_prompt}"
Proposed Optimization: "{proposed_prompt}"

Evaluate this optimization and provide your assessment."""
        
        return system_prompt, user_prompt
    
    def _proposer_generate(self, original_prompt: str, strategy: str, 
                          previous_attempt: str = "", reviewer_feedback: Dict = None) -> str:
        """Generate an optimized prompt using the Proposer agent"""
        
        system_prompt, user_prompt = self._create_proposer_prompt(
            original_prompt, strategy, previous_attempt, reviewer_feedback
        )
        
        self.logger.info(f"🎨 Proposer generating with strategy: {strategy}")
        if reviewer_feedback:
            self.logger.info(f"   Addressing feedback (score: {reviewer_feedback.get('score', 0.0)})")
        
        response = self._call_ollama(system_prompt, user_prompt)
        
        if not response:
            self.logger.warning("Proposer failed to generate response")
            return original_prompt  # Fallback to original
        
        # Clean up the response
        optimized_prompt = response.replace('"', '').strip()
        
        self.logger.info(f"✨ Proposer result: '{optimized_prompt[:60]}{'...' if len(optimized_prompt) > 60 else ''}'")
        
        return optimized_prompt
    
    def _reviewer_assess(self, original_prompt: str, proposed_prompt: str) -> Dict[str, Any]:
        """Assess a proposed optimization using the Reviewer agent"""
        
        system_prompt, user_prompt = self._create_reviewer_prompt(original_prompt, proposed_prompt)
        
        self.logger.info("🔍 Reviewer assessing proposal...")
        
        response = self._call_ollama(system_prompt, user_prompt)
        
        if not response:
            self.logger.warning("Reviewer failed to generate response")
            return {"score": 0.5, "critique": "Assessment failed", "suggestion": "Try again"}
        
        # Parse JSON response
        try:
            # Extract JSON from response (handle cases where LLM adds extra text)
            json_start = response.find('{')
            json_end = response.rfind('}') + 1
            if json_start >= 0 and json_end > json_start:
                json_str = response[json_start:json_end]
                assessment = json.loads(json_str)
            else:
                # Fallback parsing
                assessment = {"score": 0.6, "critique": "Could not parse assessment", "suggestion": "Add more specific details"}
            
            # Validate required fields
            if 'score' not in assessment:
                assessment['score'] = 0.5
            if 'critique' not in assessment:
                assessment['critique'] = "No critique provided"
            if 'suggestion' not in assessment:
                assessment['suggestion'] = "No suggestion provided"
            
            # Ensure score is in valid range
            assessment['score'] = max(0.0, min(1.0, float(assessment['score'])))
            
            self.logger.info(f"📊 Reviewer assessment: {assessment['score']:.2f}")
            self.logger.info(f"   Critique: {assessment['critique'][:80]}{'...' if len(assessment['critique']) > 80 else ''}")
            
            return assessment
            
        except Exception as e:
            self.logger.warning(f"Could not parse reviewer response: {str(e)}")
            return {
                "score": 0.5, 
                "critique": f"Parse error: {str(e)}", 
                "suggestion": "Try a simpler optimization approach"
            }
    
    def _update_learning(self, strategy: str, final_score: float, optimized_prompt: str):
        """Update learning memory with optimization results"""
        
        # Update strategy performance
        self.strategy_performance[strategy].append(final_score)
        
        # Add to quality examples for Reviewer training
        if final_score >= 0.85:
            self.quality_examples["high_quality"].append(optimized_prompt)
            # Keep only top 10 high quality examples
            if len(self.quality_examples["high_quality"]) > 10:
                self.quality_examples["high_quality"] = self.quality_examples["high_quality"][-10:]
        elif final_score <= 0.6:
            self.quality_examples["low_quality"].append(optimized_prompt)
            # Keep only bottom 5 low quality examples  
            if len(self.quality_examples["low_quality"]) > 5:
                self.quality_examples["low_quality"] = self.quality_examples["low_quality"][-5:]
    
    def optimize_prompt(self, original_prompt: str) -> Dict[str, Any]:
        """
        Optimize a prompt using conversational debate between Proposer and Reviewer.
        
        Args:
            original_prompt: The original prompt to optimize
            
        Returns:
            Dictionary containing optimization results and metadata
        """
        
        self.logger.info(f"\n🗣️  STARTING CONVERSATIONAL DEBATE")
        self.logger.info(f"Original: '{original_prompt}'")
        
        start_time = time.time()
        
        # Select strategy for this optimization
        strategy = self._select_strategy()
        self.logger.info(f"Selected strategy: {strategy}")
        
        # Initialize debate variables
        current_prompt = original_prompt
        best_prompt = original_prompt
        best_score = 0.0
        debate_history = []
        
        # Run debate rounds
        for round_num in range(1, self.max_debate_rounds + 1):
            self.logger.info(f"\n--- DEBATE ROUND {round_num} ---")
            
            # Proposer generates (uses previous feedback if available)
            previous_feedback = debate_history[-1]['assessment'] if debate_history else None
            proposed_prompt = self._proposer_generate(
                original_prompt, strategy, current_prompt, previous_feedback
            )
            
            # Reviewer assesses the proposal
            assessment = self._reviewer_assess(original_prompt, proposed_prompt)
            
            # Record this round
            round_data = {
                'round': round_num,
                'proposed_prompt': proposed_prompt,
                'assessment': assessment,
                'timestamp': datetime.now().isoformat()
            }
            debate_history.append(round_data)
            
            # Update best if this is better
            if assessment['score'] > best_score:
                best_prompt = proposed_prompt
                best_score = assessment['score']
                self.logger.info(f"🏆 New best score: {best_score:.3f}")
            
            # Check for early convergence
            if assessment['score'] >= self.target_score:
                self.logger.info(f"🎯 Target score reached: {assessment['score']:.3f}")
                break
            
            # Check for minimal improvement
            if round_num > 1:
                prev_score = debate_history[-2]['assessment']['score']
                improvement = assessment['score'] - prev_score
                if improvement < self.min_improvement:
                    self.logger.info(f"⏸️  Minimal improvement: {improvement:.3f}")
                    break
            
            # Update current prompt for next round
            current_prompt = proposed_prompt
        
        duration = time.time() - start_time
        
        # Create optimization attempt record
        attempt = OptimizationAttempt(
            prompt=best_prompt,
            strategy=strategy,
            score=best_score,
            critique=debate_history[-1]['assessment']['critique'] if debate_history else "",
            suggestion=debate_history[-1]['assessment']['suggestion'] if debate_history else ""
        )
        
        # Update learning
        self.optimization_history.append(attempt)
        self._update_learning(strategy, best_score, best_prompt)
        self._save_memory()
        
        # Compile results
        result = {
            'original_prompt': original_prompt,
            'optimized_prompt': best_prompt,
            'strategy_used': strategy,
            'final_score': best_score,
            'rounds_completed': len(debate_history),
            'debate_history': debate_history,
            'duration_seconds': duration,
            'converged': best_score >= self.target_score,
            'improvement_achieved': best_score > 0.5  # Baseline assumption
        }
        
        self.logger.info(f"\n✅ DEBATE COMPLETED")
        self.logger.info(f"Final result: '{best_prompt}'")
        self.logger.info(f"Score: {best_score:.3f} (rounds: {len(debate_history)}, duration: {duration:.1f}s)")
        
        return result


def main():
    """Demo the conversational debate optimizer"""
    print("🗣️  Conversational Debate Optimizer Demo")
    print("Testing the Proposer-Reviewer debate system...")
    print()
    
    # Test prompts
    test_prompts = [
        "emerald pendant",
        "crystal wine glass", 
        "wooden chess piece",
        "silver bracelet"
    ]
    
    # Create optimizer
    optimizer = ConversationalDebateOptimizer(
        max_debate_rounds=3,
        target_score=0.9,
        min_improvement=0.05
    )
    
    # Test each prompt
    for prompt in test_prompts:
        print(f"\n{'='*60}")
        print(f"Testing: '{prompt}'")
        print('='*60)
        
        try:
            result = optimizer.optimize_prompt(prompt)
            
            print(f"\n📊 RESULTS:")
            print(f"Original: {result['original_prompt']}")
            print(f"Optimized: {result['optimized_prompt']}")
            print(f"Strategy: {result['strategy_used']}")
            print(f"Final Score: {result['final_score']:.3f}")
            print(f"Rounds: {result['rounds_completed']}")
            print(f"Duration: {result['duration_seconds']:.1f}s")
            print(f"Converged: {result['converged']}")
            
            if result['debate_history']:
                print(f"\nDEBATE PROGRESSION:")
                for round_data in result['debate_history']:
                    print(f"Round {round_data['round']}: {round_data['assessment']['score']:.3f}")
            
        except KeyboardInterrupt:
            print(f"\n⏹️  Demo interrupted")
            break
        except Exception as e:
            print(f"\n❌ Error testing '{prompt}': {str(e)}")
            continue
    
    print(f"\n✅ Demo completed!")


if __name__ == "__main__":
    main() 