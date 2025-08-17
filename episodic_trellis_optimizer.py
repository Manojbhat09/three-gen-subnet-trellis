#!/usr/bin/env python3
"""
Episodic TRELLIS Optimizer with Persistent Memory
================================================
🧠 Persistent memory across optimization sessions
🔄 Episodic learning with cross-session knowledge  
🎯 Force improvement over historical best scores
📊 Rich context injection to LLM with past attempts
🚫 Early termination on stuck optimization
🎓 Curriculum learning from past successes
"""

import json
import time
import random
import logging
import os
import statistics
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, asdict
from smart_prompt_optimizer_v5_rl_loop import RLLoopAgent, OptimizationAttempt, RLOptimizationSession

@dataclass
class EpisodicTrellisMemory:
    """Persistent memory entry for a prompt across TRELLIS optimization episodes"""
    original_prompt: str
    best_score: float
    best_prompt: str
    best_alignment_score: float
    best_quality_score: float
    attempt_history: List[Dict[str, Any]]
    episodes_run: int
    total_attempts: int
    first_seen: str
    last_optimized: str
    score_progression: List[float]
    alignment_progression: List[float]
    strategy_performance: Dict[str, Dict[str, Any]]
    successful_patterns: List[str]
    failed_patterns: List[str]
    curriculum_level: int  # 0=basic, 1=intermediate, 2=advanced
    mastery_achieved: bool

class EpisodicTrellisOptimizer:
    """
    Episodic TRELLIS optimizer with persistent memory and forced improvement
    """
    
    def __init__(self, 
                 num_episodes: int = 50,
                 target_score: float = 0.85,
                 max_rounds_per_episode: int = 12,
                 memory_file: str = "episodic_trellis_memory.json",
                 log_dir: str = "episodic_trellis_logs",
                 improvement_threshold: float = 0.03,
                 trellis_server_url: str = "http://localhost:8096"):
        
        self.num_episodes = num_episodes
        self.target_score = target_score
        self.max_rounds_per_episode = max_rounds_per_episode
        self.memory_file = Path(memory_file)
        self.log_dir = Path(log_dir)
        self.improvement_threshold = improvement_threshold
        self.trellis_server_url = trellis_server_url
        
        # Create log directory
        self.log_dir.mkdir(exist_ok=True)
        
        # Setup logging
        log_file = self.log_dir / f"episodic_trellis_run_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s | %(levelname)-8s | %(name)s:%(funcName)s:%(lineno)d - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        
        # Initialize RL agent with episodic memory
        self.rl_agent = RLLoopAgent(
            memory_file=str(self.log_dir / "rl_memory.json"),
            trellis_server_url_w_port=trellis_server_url
        )
        
        # Override RL parameters for episodic settings
        self.rl_agent.max_optimization_rounds = max_rounds_per_episode
        self.rl_agent.min_score_threshold = target_score
        self.rl_agent.convergence_threshold = 0.01  # Stricter convergence
        self.rl_agent.min_rounds_before_convergence = 3  # Minimum rounds
        
        # Load episodic memory
        self.episodic_memory: Dict[str, EpisodicTrellisMemory] = {}
        self._load_episodic_memory()
        
        # Episode tracking
        self.episode_results = []
        self.global_insights = []
        
        self.logger.info(f"🧠 Episodic TRELLIS Optimizer initialized")
        self.logger.info(f"   Target score: {target_score}")
        self.logger.info(f"   Episodes: {num_episodes}")
        self.logger.info(f"   Max rounds per episode: {max_rounds_per_episode}")
        self.logger.info(f"   Memory entries: {len(self.episodic_memory)}")
        self.logger.info(f"   TRELLIS server: {trellis_server_url}")
    
    def _load_episodic_memory(self):
        """Load episodic memory from file"""
        if self.memory_file.exists():
            try:
                with open(self.memory_file, 'r') as f:
                    data = json.load(f)
                
                for prompt, memory_data in data.items():
                    self.episodic_memory[prompt] = EpisodicTrellisMemory(**memory_data)
                
                self.logger.info(f"📚 Loaded episodic memory: {len(self.episodic_memory)} prompts")
            except Exception as e:
                self.logger.warning(f"⚠️ Failed to load episodic memory: {e}")
                self.episodic_memory = {}
        else:
            self.logger.info("📄 Starting fresh episodic memory")
    
    def _save_episodic_memory(self):
        """Save episodic memory to file"""
        try:
            data = {
                prompt: asdict(memory) for prompt, memory in self.episodic_memory.items()
            }
            with open(self.memory_file, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            self.logger.error(f"❌ Failed to save episodic memory: {e}")
    
    def _build_historical_context(self, prompt: str) -> str:
        """Build rich historical context for the LLM"""
        if prompt not in self.episodic_memory:
            return """HISTORICAL CONTEXT: This is the first time optimizing this prompt.
            
🎯 OPTIMIZATION GOALS:
   - Maximize validation score (combined quality + alignment)
   - Maintain alignment score > 0.3 (critical threshold)
   - Focus on quality improvements
   - Learn from score patterns"""
        
        memory = self.episodic_memory[prompt]
        
        context = f"""HISTORICAL CONTEXT FOR PROMPT: "{prompt}"
        
🎯 CURRENT BEST TO BEAT:
   Validation Score: {memory.best_score:.4f}
   Alignment Score: {memory.best_alignment_score:.4f}  
   Quality Score: {memory.best_quality_score:.4f}
   Best Prompt: "{memory.best_prompt}"
   
📊 PERFORMANCE HISTORY:
   Episodes run: {memory.episodes_run}
   Total attempts: {memory.total_attempts}
   Score progression: {[f'{s:.3f}' for s in memory.score_progression[-10:]]}
   Alignment progression: {[f'{s:.3f}' for s in memory.alignment_progression[-10:]]}
   Last optimized: {memory.last_optimized}
   Curriculum level: {memory.curriculum_level} (0=basic, 1=intermediate, 2=advanced)
   Mastery achieved: {memory.mastery_achieved}
   
🧠 LEARNED PATTERNS:
   Successful patterns: {memory.successful_patterns}
   Failed patterns: {memory.failed_patterns}
   
🎛️ STRATEGY PERFORMANCE:"""
        
        for strategy, perf in memory.strategy_performance.items():
            context += f"\n   {strategy}: avg={perf.get('avg_score', 0):.3f}, attempts={perf.get('attempts', 0)}"
        
        context += f"""
        
🚨 IMPROVEMENT REQUIREMENT:
   You MUST achieve validation score > {memory.best_score:.4f} (current best)
   Minimum improvement needed: +{self.improvement_threshold:.3f}
   Target score: {self.target_score:.3f}
   Critical: Keep alignment score ≥ 0.3 (below this = 0.0 validation score)
   
💡 RECENT INSIGHTS:
"""
        
        # Add recent attempt insights
        if memory.attempt_history:
            recent_attempts = memory.attempt_history[-5:]
            for i, attempt in enumerate(recent_attempts, 1):
                val_score = attempt.get('validation_score', 0)
                align_score = attempt.get('alignment_score', 0)
                qual_score = attempt.get('quality_score', 0)
                context += f"   {i}. Val: {val_score:.3f}, Align: {align_score:.3f}, Qual: {qual_score:.3f}, Strategy: {attempt.get('strategy', 'unknown')}\n"
                context += f"      Prompt: {attempt.get('prompt', '')[:100]}...\n"
        
        # Add curriculum guidance
        if memory.curriculum_level == 0:
            context += "\n🎓 CURRICULUM: Basic level - Focus on simple, clear object descriptions"
        elif memory.curriculum_level == 1:
            context += "\n🎓 CURRICULUM: Intermediate level - Add material properties and details"
        else:
            context += "\n🎓 CURRICULUM: Advanced level - Complex compositions and artistic elements"
        
        return context
    
    def optimize_prompt_episodically(self, prompt: str, episode_num: int) -> Dict[str, Any]:
        """Optimize a single prompt with episodic memory context"""
        episode_start_time = time.time()
        
        self.logger.info(f"\n🔄 Episode {episode_num}: Optimizing '{prompt}'")
        
        # Check if we already have memory for this prompt
        has_memory = prompt in self.episodic_memory
        memory = self.episodic_memory.get(prompt, None)
        
        if has_memory:
            self.logger.info(f"   📚 Found historical data: best score {memory.best_score:.4f}")
            self.logger.info(f"   📚 Episodes run: {memory.episodes_run}, Total attempts: {memory.total_attempts}")
            self.logger.info(f"   📚 Curriculum level: {memory.curriculum_level}, Mastery: {memory.mastery_achieved}")
            
            # Check if already achieved mastery
            if memory.mastery_achieved and memory.best_score >= self.target_score:
                self.logger.info(f"   ✅ Mastery already achieved! Skipping optimization.")
                return {
                    'episode': episode_num,
                    'original_prompt': prompt,
                    'result': 'mastery_already_achieved',
                    'best_score': memory.best_score,
                    'best_prompt': memory.best_prompt,
                    'duration': 0.0,
                    'attempts': 0
                }
        
        # Build historical context
        historical_context = self._build_historical_context(prompt)
        
        # Run RL optimization with historical context
        try:
            result = self.rl_agent.optimize_with_rl_loop(
                prompt, 
                use_validation=True, 
                prompt_with_context=historical_context
            )
            
            final_score = result.get('final_score', 0.0)
            final_alignment_score = result.get('final_alignment_score', 0.0)
            final_prompt = result.get('final_optimized_prompt', prompt)
            attempts = result.get('total_rounds', 0)
            session_duration = result.get('processing_time', 0.0)
            
            self.logger.info(f"   📊 Final validation score: {final_score:.4f}")
            self.logger.info(f"   📊 Final alignment score: {final_alignment_score:.4f}")
            self.logger.info(f"   📊 Rounds: {attempts}")
            self.logger.info(f"   📊 Duration: {session_duration:.1f}s")
            
            # Check for improvement requirement
            improvement_achieved = True
            improvement_amount = 0.0
            
            if has_memory:
                improvement_amount = final_score - memory.best_score
                improvement_achieved = improvement_amount >= self.improvement_threshold
                
                if not improvement_achieved:
                    self.logger.warning(f"   ⚠️ Insufficient improvement: {improvement_amount:.4f} < {self.improvement_threshold:.3f}")
                else:
                    self.logger.info(f"   🎉 Improvement achieved: +{improvement_amount:.4f}")
            
            # Update episodic memory
            self._update_episodic_memory(prompt, result, historical_context, episode_num)
            
            episode_result = {
                'episode': episode_num,
                'original_prompt': prompt,
                'final_score': final_score,
                'final_alignment_score': final_alignment_score,
                'final_prompt': final_prompt,
                'attempts': attempts,
                'duration': session_duration,
                'had_previous_memory': has_memory,
                'improvement_achieved': improvement_achieved,
                'improvement_amount': improvement_amount,
                'target_achieved': final_score >= self.target_score,
                'convergence_achieved': result.get('convergence_achieved', False),
                'convergence_reason': result.get('convergence_reason', 'Not converged'),
                'exploration_ratio': result.get('exploration_ratio', 0.0),
                'learned_insights': result.get('learned_insights', [])
            }
            
            return episode_result
            
        except Exception as e:
            self.logger.error(f"   ❌ Episode failed: {e}")
            return {
                'episode': episode_num,
                'original_prompt': prompt,
                'result': 'failed',
                'error': str(e),
                'duration': time.time() - episode_start_time
            }
    
    def _update_episodic_memory(self, prompt: str, rl_result: Dict[str, Any], 
                              historical_context: str, episode_num: int):
        """Update episodic memory with new optimization results"""
        
        final_score = rl_result.get('final_score', 0.0)
        final_alignment_score = rl_result.get('final_alignment_score', 0.0)
        final_prompt = rl_result.get('final_optimized_prompt', prompt)
        attempts = rl_result.get('total_rounds', 0)
        score_progression = rl_result.get('score_progression', [])
        alignment_progression = rl_result.get('alignment_progression', [])
        strategy_sequence = rl_result.get('strategy_sequence', [])
        learned_insights = rl_result.get('learned_insights', [])
        
        # Extract patterns from successful/failed attempts
        successful_patterns = []
        failed_patterns = []
        
        if score_progression and len(score_progression) > 1:
            best_score_idx = score_progression.index(max(score_progression))
            if best_score_idx < len(strategy_sequence):
                successful_patterns.append(strategy_sequence[best_score_idx])
            
            # Find consistently poor strategies
            for i, (score, strategy) in enumerate(zip(score_progression, strategy_sequence)):
                if score < statistics.mean(score_progression) * 0.8:  # 20% below average
                    failed_patterns.append(strategy)
        
        if prompt in self.episodic_memory:
            # Update existing memory
            memory = self.episodic_memory[prompt]
            
            # Update if new score is better
            if final_score > memory.best_score:
                memory.best_score = final_score
                memory.best_alignment_score = final_alignment_score
                # Extract quality score (validation score = 0.75 * quality + 0.25 * alignment)
                memory.best_quality_score = (final_score - 0.25 * final_alignment_score) / 0.75 if final_alignment_score > 0 else 0.0
                memory.best_prompt = final_prompt
            
            memory.episodes_run += 1
            memory.total_attempts += attempts
            memory.last_optimized = datetime.now().isoformat()
            memory.score_progression.extend(score_progression)
            memory.alignment_progression.extend(alignment_progression)
            
            # Keep only last 50 scores
            if len(memory.score_progression) > 50:
                memory.score_progression = memory.score_progression[-50:]
            if len(memory.alignment_progression) > 50:
                memory.alignment_progression = memory.alignment_progression[-50:]
            
            # Update curriculum level based on performance
            if memory.best_score >= self.target_score:
                memory.mastery_achieved = True
                memory.curriculum_level = 2
            elif memory.best_score >= 0.7:
                memory.curriculum_level = 2
            elif memory.best_score >= 0.5:
                memory.curriculum_level = 1
            else:
                memory.curriculum_level = 0
            
            # Update strategy performance
            for strategy in strategy_sequence:
                if strategy not in memory.strategy_performance:
                    memory.strategy_performance[strategy] = {'attempts': 0, 'total_score': 0.0}
                
                memory.strategy_performance[strategy]['attempts'] += 1
                memory.strategy_performance[strategy]['total_score'] += final_score
                memory.strategy_performance[strategy]['avg_score'] = (
                    memory.strategy_performance[strategy]['total_score'] / 
                    memory.strategy_performance[strategy]['attempts']
                )
            
            # Update patterns
            memory.successful_patterns.extend(successful_patterns)
            memory.failed_patterns.extend(failed_patterns)
            
            # Keep unique and recent patterns
            memory.successful_patterns = list(set(memory.successful_patterns))[-10:]
            memory.failed_patterns = list(set(memory.failed_patterns))[-10:]
            
        else:
            # Create new memory entry
            strategy_performance = {}
            for strategy in strategy_sequence:
                if strategy not in strategy_performance:
                    strategy_performance[strategy] = {'attempts': 0, 'total_score': 0.0}
                
                strategy_performance[strategy]['attempts'] += 1
                strategy_performance[strategy]['total_score'] += final_score
                strategy_performance[strategy]['avg_score'] = (
                    strategy_performance[strategy]['total_score'] / 
                    strategy_performance[strategy]['attempts']
                )
            
            # Determine initial curriculum level
            curriculum_level = 0
            if final_score >= 0.7:
                curriculum_level = 2
            elif final_score >= 0.5:
                curriculum_level = 1
            
            self.episodic_memory[prompt] = EpisodicTrellisMemory(
                original_prompt=prompt,
                best_score=final_score,
                best_prompt=final_prompt,
                best_alignment_score=final_alignment_score,
                best_quality_score=(final_score - 0.25 * final_alignment_score) / 0.75 if final_alignment_score > 0 else 0.0,
                attempt_history=[],
                episodes_run=1,
                total_attempts=attempts,
                first_seen=datetime.now().isoformat(),
                last_optimized=datetime.now().isoformat(),
                score_progression=score_progression,
                alignment_progression=alignment_progression,
                strategy_performance=strategy_performance,
                successful_patterns=successful_patterns,
                failed_patterns=failed_patterns,
                curriculum_level=curriculum_level,
                mastery_achieved=final_score >= self.target_score
            )
        
        # Add this attempt to history
        attempt_record = {
            'episode': episode_num,
            'validation_score': final_score,
            'alignment_score': final_alignment_score,
            'quality_score': (final_score - 0.25 * final_alignment_score) / 0.75 if final_alignment_score > 0 else 0.0,
            'prompt': final_prompt,
            'strategy': strategy_sequence[-1] if strategy_sequence else 'unknown',
            'attempts': attempts,
            'timestamp': datetime.now().isoformat(),
            'insights': learned_insights
        }
        
        self.episodic_memory[prompt].attempt_history.append(attempt_record)
        
        # Keep only last 20 attempts
        if len(self.episodic_memory[prompt].attempt_history) > 20:
            self.episodic_memory[prompt].attempt_history = self.episodic_memory[prompt].attempt_history[-20:]
    
    def run_all_episodes(self, test_prompts: List[str]) -> Dict[str, Any]:
        """Run episodic optimization on all test prompts"""
        
        overall_start_time = time.time()
        
        self.logger.info(f"\n🚀 Starting episodic TRELLIS optimization")
        self.logger.info(f"   Test prompts: {len(test_prompts)}")
        self.logger.info(f"   Episodes per prompt: {self.num_episodes}")
        self.logger.info(f"   Target score: {self.target_score}")
        
        all_results = []
        
        for episode in range(1, self.num_episodes + 1):
            self.logger.info(f"\n{'='*60}")
            self.logger.info(f"EPISODE {episode}/{self.num_episodes}")
            self.logger.info(f"{'='*60}")
            
            episode_results = []
            
            # Randomly shuffle prompts each episode for variety
            shuffled_prompts = test_prompts.copy()
            random.shuffle(shuffled_prompts)
            
            for prompt_idx, prompt in enumerate(shuffled_prompts, 1):
                self.logger.info(f"\n--- Prompt {prompt_idx}/{len(shuffled_prompts)} ---")
                
                result = self.optimize_prompt_episodically(prompt, episode)
                episode_results.append(result)
                all_results.append(result)
                
                # Save memory after each prompt
                self._save_episodic_memory()
            
            # Episode summary
            successful_episodes = [r for r in episode_results if r.get('target_achieved', False)]
            improved_episodes = [r for r in episode_results if r.get('improvement_achieved', False)]
            mastery_episodes = [r for r in episode_results if r.get('result') == 'mastery_already_achieved']
            
            self.logger.info(f"\n📊 Episode {episode} Summary:")
            self.logger.info(f"   Targets achieved: {len(successful_episodes)}/{len(episode_results)}")
            self.logger.info(f"   Improvements made: {len(improved_episodes)}/{len(episode_results)}")
            self.logger.info(f"   Mastery maintained: {len(mastery_episodes)}/{len(episode_results)}")
            
            valid_scores = [r.get('final_score', 0) for r in episode_results if 'final_score' in r]
            if valid_scores:
                avg_score = statistics.mean(valid_scores)
                self.logger.info(f"   Average score: {avg_score:.4f}")
        
        # Final analysis
        overall_duration = time.time() - overall_start_time
        
        analysis = self._analyze_episodic_learning(all_results)
        
        final_summary = {
            'total_episodes': self.num_episodes,
            'total_prompts': len(test_prompts),
            'total_duration': overall_duration,
            'all_results': all_results,
            'analysis': analysis,
            'memory_entries': len(self.episodic_memory),
            'completed_at': datetime.now().isoformat()
        }
        
        self._save_final_results(final_summary)
        self._log_final_summary(analysis, overall_duration)
        
        return final_summary
    
    def _analyze_episodic_learning(self, all_results: List[Dict]) -> Dict[str, Any]:
        """Analyze learning progress across episodes"""
        
        analysis = {
            'total_optimizations': len(all_results),
            'successful_optimizations': 0,
            'improved_optimizations': 0,
            'mastery_maintained': 0,
            'average_score': 0.0,
            'score_distribution': {},
            'curriculum_distribution': {},
            'memory_insights': {}
        }
        
        # Basic stats
        valid_results = [r for r in all_results if 'final_score' in r]
        if valid_results:
            scores = [r['final_score'] for r in valid_results]
            analysis['average_score'] = statistics.mean(scores)
            analysis['successful_optimizations'] = len([r for r in valid_results if r.get('target_achieved', False)])
            analysis['improved_optimizations'] = len([r for r in valid_results if r.get('improvement_achieved', False)])
            
            # Score distribution
            analysis['score_distribution'] = {
                'min': min(scores),
                'max': max(scores),
                'median': statistics.median(scores),
                'std_dev': statistics.stdev(scores) if len(scores) > 1 else 0.0
            }
        
        # Count mastery maintained
        analysis['mastery_maintained'] = len([r for r in all_results if r.get('result') == 'mastery_already_achieved'])
        
        # Curriculum distribution
        curriculum_counts = {0: 0, 1: 0, 2: 0}
        for memory in self.episodic_memory.values():
            curriculum_counts[memory.curriculum_level] += 1
        
        analysis['curriculum_distribution'] = {
            'basic': curriculum_counts[0],
            'intermediate': curriculum_counts[1], 
            'advanced': curriculum_counts[2]
        }
        
        # Memory insights
        total_attempts = sum(memory.total_attempts for memory in self.episodic_memory.values())
        total_episodes_run = sum(memory.episodes_run for memory in self.episodic_memory.values())
        mastery_achieved_count = sum(1 for memory in self.episodic_memory.values() if memory.mastery_achieved)
        
        analysis['memory_insights'] = {
            'unique_prompts_optimized': len(self.episodic_memory),
            'total_optimization_attempts': total_attempts,
            'total_episodes_across_prompts': total_episodes_run,
            'average_attempts_per_prompt': total_attempts / len(self.episodic_memory) if self.episodic_memory else 0,
            'prompts_achieving_target': len([m for m in self.episodic_memory.values() if m.best_score >= self.target_score]),
            'prompts_with_mastery': mastery_achieved_count
        }
        
        return analysis
    
    def _save_final_results(self, results: Dict[str, Any]):
        """Save final results to file"""
        results_file = self.log_dir / f"episodic_trellis_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        try:
            with open(results_file, 'w') as f:
                json.dump(results, f, indent=2)
            self.logger.info(f"💾 Results saved to: {results_file}")
        except Exception as e:
            self.logger.error(f"❌ Failed to save results: {e}")
    
    def _log_final_summary(self, analysis: Dict[str, Any], duration: float):
        """Log final summary"""
        self.logger.info(f"\n{'='*80}")
        self.logger.info(f"EPISODIC TRELLIS OPTIMIZATION COMPLETE")
        self.logger.info(f"{'='*80}")
        self.logger.info(f"📊 Total Duration: {duration:.1f}s ({duration/60:.1f} minutes)")
        self.logger.info(f"📊 Total Optimizations: {analysis['total_optimizations']}")
        self.logger.info(f"📊 Successful (target achieved): {analysis['successful_optimizations']}")
        self.logger.info(f"📊 Improved over previous: {analysis['improved_optimizations']}")
        self.logger.info(f"📊 Mastery maintained: {analysis['mastery_maintained']}")
        self.logger.info(f"📊 Average Score: {analysis['average_score']:.4f}")
        
        curriculum_dist = analysis['curriculum_distribution']
        self.logger.info(f"🎓 Curriculum: Basic: {curriculum_dist['basic']}, Intermediate: {curriculum_dist['intermediate']}, Advanced: {curriculum_dist['advanced']}")
        
        memory_insights = analysis['memory_insights']
        self.logger.info(f"🧠 Unique prompts optimized: {memory_insights['unique_prompts_optimized']}")
        self.logger.info(f"🧠 Total attempts: {memory_insights['total_optimization_attempts']}")
        self.logger.info(f"🧠 Prompts achieving target: {memory_insights['prompts_achieving_target']}")
        self.logger.info(f"🧠 Prompts with mastery: {memory_insights['prompts_with_mastery']}")
        
        self.logger.info(f"📚 Episodic memory entries: {len(self.episodic_memory)}")
        self.logger.info(f"💾 Memory saved to: {self.memory_file}")

def main():
    """Main function for running episodic TRELLIS optimization"""
    
    # Example test prompts
    test_prompts = [
        "luxurious cream sedan elegant",
        "ornate heart-shaped pendant",
        "small wooden hammer with screws",
        "a red car",
        "a beautiful sunset",
        "crystal wine glass",
        "vintage leather boots",
        "golden pocket watch"
    ]
    
    optimizer = EpisodicTrellisOptimizer(
        num_episodes=10,  # Start with fewer episodes for testing
        target_score=0.85,
        max_rounds_per_episode=8,
        improvement_threshold=0.03
    )
    
    try:
        results = optimizer.run_all_episodes(test_prompts)
        print(f"\n✅ Episodic TRELLIS optimization complete!")
        print(f"Results saved to: {optimizer.log_dir}")
        
    except KeyboardInterrupt:
        print("\n⚠️ Interrupted by user")
        optimizer._save_episodic_memory()
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 