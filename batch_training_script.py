#!/usr/bin/env python3
"""
Batch Training Script for Proposer-Reviewer-Judge System
=======================================================

This script trains the Proposer-Reviewer-Judge system on multiple prompts
to build a comprehensive knowledge base for reliable inference.

Features:
- Trains on diverse prompt types to build robust knowledge
- Tracks learning progress across training sessions
- Saves comprehensive training data for inference use
- Provides detailed analytics on training effectiveness
- Handles errors gracefully and continues training

Training Strategy:
1. Train on diverse object types (jewelry, furniture, food, etc.)
2. Build strategy performance knowledge
3. Calibrate Reviewer accuracy through ground truth
4. Create comprehensive examples for inference
"""

import json
import time
import sys
import random
from datetime import datetime
from typing import List, Dict, Any
from pathlib import Path
import logging

from proposer_reviewer_judge_trainer import ProposerReviewerJudgeTrainer


class BatchTrainer:
    """
    Manages batch training across multiple prompts for comprehensive learning.
    """
    
    def __init__(self, 
                 training_memory_file: str = "prj_training_memory.json",
                 max_rounds: int = 4,
                 quality_threshold: float = 0.8,
                 convergence_threshold: float = 0.9):
        """
        Initialize batch trainer.
        
        Args:
            training_memory_file: File to store all training results
            max_rounds: Maximum rounds per training session
            quality_threshold: Minimum score to save to memory
            convergence_threshold: Score for early stopping
        """
        self.training_memory_file = training_memory_file
        self.max_rounds = max_rounds
        self.quality_threshold = quality_threshold
        self.convergence_threshold = convergence_threshold
        
        # Initialize trainer
        self.trainer = ProposerReviewerJudgeTrainer(
            memory_file=training_memory_file,
            max_rounds=max_rounds,
            quality_threshold=quality_threshold,
            convergence_threshold=convergence_threshold
        )
        
        # Diverse training prompts covering various object types
        self.training_prompts = [
            # Jewelry & Accessories
            "emerald pendant",
            "silver bracelet", 
            "golden ring",
            "sapphire-studded sharp spear",
            "necklace with heart-shaped pendant made of silver and turquoise stones",
            
            # Glassware & Containers
            "crystal wine glass",
            "bottle of red wine with cork in it",
            "tall glass of layered lemonade",
            "cylindrical glass of bubbly lemonade",
            
            # Musical Instruments & Art
            "harp adorned with pearl inlays and gilded frame",
            "crystal staff with swirling light",
            
            # Furniture & Household
            "wooden chess piece",
            "ceramic vase",
            "matte black candle holder two interlocking pieces",
            
            # Food & Organic
            "cupcake with chocolate icing on top",
            
            # Cultural & Historical
            "greek kylix cup black-figure technique mythological scenes",
            
            # Fantasy & Creative
            "small round blue creature with long nose and pointed ears",
            
            # Additional diverse objects for robust training
            "ornate brass compass",
            "marble sculpture of a dove",
            "silk scarf with floral patterns",
            "copper teapot with intricate engravings",
            "leather-bound book with gold lettering",
            "porcelain figurine of a dancer",
            "stained glass window panel",
            "bamboo wind chimes",
            "ivory chess king piece",
            "pearl necklace with diamond clasp"
        ]
        
        # Batch tracking
        self.batch_results: List[Dict] = []
        self.failed_prompts: List[str] = []
        
        # Setup logging
        log_dir = "batch_training_logs"
        Path(log_dir).mkdir(exist_ok=True)
        log_file = Path(log_dir) / f"batch_training_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        
        self.logger.info("🎓 BATCH TRAINER INITIALIZED")
        self.logger.info(f"   Training prompts: {len(self.training_prompts)}")
        self.logger.info(f"   Max rounds per prompt: {max_rounds}")
        self.logger.info(f"   Quality threshold: {quality_threshold}")
        self.logger.info(f"   Results file: {training_memory_file}")
    
    def train_on_prompt_set(self, 
                          prompt_subset: List[str] = None,
                          shuffle_order: bool = True,
                          continue_on_failure: bool = True) -> Dict[str, Any]:
        """
        Train on a set of prompts with comprehensive tracking.
        
        Args:
            prompt_subset: Specific prompts to train on (None = all)
            shuffle_order: Whether to randomize training order
            continue_on_failure: Whether to continue if a prompt fails
            
        Returns:
            Comprehensive batch training results
        """
        if prompt_subset is None:
            prompts_to_train = self.training_prompts.copy()
        else:
            prompts_to_train = prompt_subset.copy()
        
        if shuffle_order:
            random.shuffle(prompts_to_train)
        
        self.logger.info(f"\n{'='*80}")
        self.logger.info(f"STARTING BATCH TRAINING")
        self.logger.info(f"{'='*80}")
        self.logger.info(f"Training prompts: {len(prompts_to_train)}")
        self.logger.info(f"Shuffle order: {shuffle_order}")
        self.logger.info(f"Continue on failure: {continue_on_failure}")
        
        batch_start_time = time.time()
        successful_sessions = 0
        total_validation_time = 0.0
        
        # Train on each prompt
        for i, prompt in enumerate(prompts_to_train, 1):
            self.logger.info(f"\n🎯 TRAINING {i}/{len(prompts_to_train)}: '{prompt}'")
            
            try:
                session_start = time.time()
                session = self.trainer.train_on_prompt(prompt)
                session_duration = time.time() - session_start
                
                # Record results
                result = {
                    'prompt': prompt,
                    'session_id': session.session_id,
                    'best_score': session.best_judge_score,
                    'rounds': session.total_rounds,
                    'converged': session.converged,
                    'saved_to_memory': session.saved_to_memory,
                    'duration': session_duration,
                    'validation_time': session.total_validation_time,
                    'proposer_improvement': session.proposer_improvement,
                    'reviewer_accuracy_improvement': session.reviewer_accuracy_improvement,
                    'timestamp': session.timestamp
                }
                self.batch_results.append(result)
                
                if session.saved_to_memory:
                    successful_sessions += 1
                
                total_validation_time += session.total_validation_time
                
                self.logger.info(f"✅ Session completed: Score {session.best_judge_score:.3f}, Rounds {session.total_rounds}")
                
                # Brief pause between training sessions
                time.sleep(2)
                
            except Exception as e:
                self.logger.error(f"❌ Training failed for '{prompt}': {str(e)}")
                self.failed_prompts.append(prompt)
                
                if not continue_on_failure:
                    self.logger.error("Stopping batch training due to failure")
                    break
                else:
                    self.logger.info("Continuing with next prompt...")
                    continue
        
        # Calculate batch statistics
        batch_duration = time.time() - batch_start_time
        
        if self.batch_results:
            avg_score = sum(r['best_score'] for r in self.batch_results) / len(self.batch_results)
            avg_rounds = sum(r['rounds'] for r in self.batch_results) / len(self.batch_results)
            convergence_rate = sum(1 for r in self.batch_results if r['converged']) / len(self.batch_results)
            success_rate = successful_sessions / len(self.batch_results)
        else:
            avg_score = avg_rounds = convergence_rate = success_rate = 0.0
        
        # Compile batch results
        batch_summary = {
            'batch_metadata': {
                'total_prompts': len(prompts_to_train),
                'completed_prompts': len(self.batch_results),
                'failed_prompts': len(self.failed_prompts),
                'successful_sessions': successful_sessions,
                'batch_duration_minutes': batch_duration / 60,
                'total_validation_time_minutes': total_validation_time / 60,
                'timestamp': datetime.now().isoformat()
            },
            'performance_metrics': {
                'average_score': avg_score,
                'average_rounds': avg_rounds,
                'convergence_rate': convergence_rate,
                'success_rate': success_rate,
                'failed_prompts': self.failed_prompts
            },
            'individual_results': self.batch_results,
            'training_effectiveness': self._analyze_training_effectiveness()
        }
        
        # Save batch results
        self._save_batch_results(batch_summary)
        
        # Log final summary
        self._log_batch_summary(batch_summary)
        
        return batch_summary
    
    def _analyze_training_effectiveness(self) -> Dict[str, Any]:
        """Analyze the effectiveness of the training process."""
        if not self.batch_results:
            return {}
        
        # Score distribution
        scores = [r['best_score'] for r in self.batch_results]
        high_scores = len([s for s in scores if s >= 0.85])
        medium_scores = len([s for s in scores if 0.7 <= s < 0.85])
        low_scores = len([s for s in scores if s < 0.7])
        
        # Rounds efficiency
        rounds = [r['rounds'] for r in self.batch_results]
        avg_rounds = sum(rounds) / len(rounds)
        efficient_sessions = len([r for r in rounds if r <= 2])  # 2 rounds or less
        
        # Learning progression (if there's enough data)
        if len(self.batch_results) >= 10:
            first_half = self.batch_results[:len(self.batch_results)//2]
            second_half = self.batch_results[len(self.batch_results)//2:]
            
            first_half_avg = sum(r['best_score'] for r in first_half) / len(first_half)
            second_half_avg = sum(r['best_score'] for r in second_half) / len(second_half)
            learning_improvement = second_half_avg - first_half_avg
        else:
            learning_improvement = 0.0
        
        # Strategy analysis (if available from trainer)
        training_stats = self.trainer.get_training_statistics()
        
        return {
            'score_distribution': {
                'high_scores_85_plus': high_scores,
                'medium_scores_70_to_85': medium_scores,
                'low_scores_below_70': low_scores,
                'score_quality_rate': high_scores / max(1, len(scores))
            },
            'efficiency_metrics': {
                'average_rounds': avg_rounds,
                'efficient_sessions_2_rounds_or_less': efficient_sessions,
                'efficiency_rate': efficient_sessions / max(1, len(rounds))
            },
            'learning_progression': {
                'learning_improvement': learning_improvement,
                'sufficient_data_for_analysis': len(self.batch_results) >= 10
            },
            'system_statistics': training_stats
        }
    
    def _save_batch_results(self, batch_summary: Dict[str, Any]):
        """Save batch training results to file."""
        try:
            results_file = Path(f"batch_training_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
            with open(results_file, 'w') as f:
                json.dump(batch_summary, f, indent=2)
            
            self.logger.info(f"💾 Batch results saved to: {results_file}")
            
        except Exception as e:
            self.logger.error(f"Error saving batch results: {str(e)}")
    
    def _log_batch_summary(self, batch_summary: Dict[str, Any]):
        """Log comprehensive batch training summary."""
        metadata = batch_summary['batch_metadata']
        metrics = batch_summary['performance_metrics']
        effectiveness = batch_summary['training_effectiveness']
        
        self.logger.info(f"\n{'='*80}")
        self.logger.info(f"BATCH TRAINING SUMMARY")
        self.logger.info(f"{'='*80}")
        
        self.logger.info(f"COMPLETION STATISTICS:")
        self.logger.info(f"  Total Prompts: {metadata['total_prompts']}")
        self.logger.info(f"  Completed: {metadata['completed_prompts']}")
        self.logger.info(f"  Failed: {metadata['failed_prompts']}")
        self.logger.info(f"  Successful Sessions: {metadata['successful_sessions']}")
        self.logger.info(f"  Success Rate: {metrics['success_rate']:.1%}")
        
        self.logger.info(f"\nPERFORMANCE METRICS:")
        self.logger.info(f"  Average Score: {metrics['average_score']:.3f}")
        self.logger.info(f"  Average Rounds: {metrics['average_rounds']:.1f}")
        self.logger.info(f"  Convergence Rate: {metrics['convergence_rate']:.1%}")
        
        self.logger.info(f"\nTIME ANALYSIS:")
        self.logger.info(f"  Batch Duration: {metadata['batch_duration_minutes']:.1f} minutes")
        self.logger.info(f"  Total Validation Time: {metadata['total_validation_time_minutes']:.1f} minutes")
        
        if 'score_distribution' in effectiveness:
            dist = effectiveness['score_distribution']
            self.logger.info(f"\nSCORE DISTRIBUTION:")
            self.logger.info(f"  High Scores (≥0.85): {dist['high_scores_85_plus']}")
            self.logger.info(f"  Medium Scores (0.70-0.85): {dist['medium_scores_70_to_85']}")
            self.logger.info(f"  Low Scores (<0.70): {dist['low_scores_below_70']}")
            self.logger.info(f"  Quality Rate: {dist['score_quality_rate']:.1%}")
        
        if effectiveness.get('learning_progression', {}).get('sufficient_data_for_analysis'):
            improvement = effectiveness['learning_progression']['learning_improvement']
            self.logger.info(f"\nLEARNING PROGRESSION:")
            self.logger.info(f"  Score Improvement: {improvement:+.3f}")
            
        if metrics['failed_prompts']:
            self.logger.info(f"\nFAILED PROMPTS:")
            for prompt in metrics['failed_prompts']:
                self.logger.info(f"  - {prompt}")
        
        self.logger.info(f"\n✅ Batch training knowledge ready for inference!")
        self.logger.info(f"Training data saved in: {self.training_memory_file}")


def main():
    """Main batch training interface."""
    print("🎓 BATCH TRAINING FOR PROPOSER-REVIEWER-JUDGE SYSTEM")
    print("="*80)
    print("Building comprehensive training knowledge for fast inference")
    print()
    
    # Configuration options
    if len(sys.argv) > 1:
        if sys.argv[1] == "--quick":
            # Quick training on subset for testing
            subset = [
                "emerald pendant",
                "crystal wine glass", 
                "wooden chess piece",
                "silver bracelet",
                "ceramic vase"
            ]
            print("🚀 QUICK TRAINING MODE")
            print(f"Training on {len(subset)} prompts for testing")
            
        elif sys.argv[1] == "--comprehensive":
            # Full comprehensive training
            subset = None
            print("🎯 COMPREHENSIVE TRAINING MODE") 
            print("Training on all available prompts for maximum knowledge")
            
        else:
            print("Usage: python batch_training_script.py [--quick|--comprehensive]")
            print("  --quick: Train on 5 prompts for testing (faster)")
            print("  --comprehensive: Train on all prompts for maximum knowledge")
            sys.exit(1)
    else:
        # Default: medium training set
        subset = None
        print("📚 STANDARD TRAINING MODE")
        print("Training on all available prompts")
    
    # Initialize batch trainer
    trainer = BatchTrainer(
        max_rounds=10,
        quality_threshold=0.8,
        convergence_threshold=0.96
    )
    
    print(f"\nConfiguration:")
    print(f"  Max rounds per prompt: {trainer.max_rounds}")
    print(f"  Quality threshold: {trainer.quality_threshold}")
    print(f"  Convergence threshold: {trainer.convergence_threshold}")
    
    try:
        # Run batch training
        print(f"\n🚀 Starting batch training...")
        results = trainer.train_on_prompt_set(
            prompt_subset=subset,
            shuffle_order=True,
            continue_on_failure=True
        )
        
        print(f"\n✅ BATCH TRAINING COMPLETED SUCCESSFULLY!")
        
        # Show key results
        metadata = results['batch_metadata']
        metrics = results['performance_metrics']
        
        print(f"\nKey Results:")
        print(f"  Prompts Trained: {metadata['completed_prompts']}/{metadata['total_prompts']}")
        print(f"  Success Rate: {metrics['success_rate']:.1%}")
        print(f"  Average Score: {metrics['average_score']:.3f}")
        print(f"  Duration: {metadata['batch_duration_minutes']:.1f} minutes")
        
        print(f"\n🎉 Training knowledge is now ready for fast inference!")
        print(f"Use 'python trained_debate_inference.py \"your prompt\"' for optimization")
        
        return results
        
    except KeyboardInterrupt:
        print(f"\n⏹️  Batch training interrupted by user")
        return None
        
    except Exception as e:
        print(f"\n❌ Batch training error: {str(e)}")
        return None


if __name__ == "__main__":
    main() 