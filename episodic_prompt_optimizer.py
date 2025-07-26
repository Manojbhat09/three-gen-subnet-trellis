#!/usr/bin/env python3
"""
Episodic Prompt Optimizer - Multi-Episode Learning Wrapper

This script runs multiple episodes of prompt optimization using the V4.1 RL Loop optimizer.
Each episode cycles through all test prompts, allowing the agent to learn principles and
strategies that transfer across different prompt types and episodes.

Features:
- Multi-episode learning with persistent memory
- Cross-prompt principle extraction
- Progressive strategy refinement
- Comprehensive logging and analytics
- Convergence tracking across episodes
"""

import json
import os
import time
import statistics
from datetime import datetime
from typing import List, Dict, Any, Tuple
import logging
import time
import re
from episodic_test_prompts import EPISODIC_TEST_PROMPTS
# Import the V4.1 RL Loop optimizer
# from smart_prompt_optimizer_v4_1_rl_loop import RLLoopAgent
from smart_prompt_optimizer_v5_rl_loop import RLLoopAgent

class EpisodicPromptOptimizer:
    """
    Wrapper for running multiple episodes of prompt optimization.
    
    Each episode processes all test prompts sequentially, allowing the agent
    to build up knowledge and principles that can be applied across different
    prompt types and future episodes.
    """
    
    @staticmethod
    def _get_uncommented_prompts():
        # Only use uncommented prompts from the imported list
        # (Python import will not include commented lines, but user may want to comment out in the .py file)
        # So just return the list as-is
        return [p for p in EPISODIC_TEST_PROMPTS if isinstance(p, str) and p.strip()]

    @staticmethod
    def _add_prompts_to_pyfile(new_prompts, pyfile_path="episodic_test_prompts.py"):
        # Helper to add new prompts to the .py file if not already present
        try:
            with open(pyfile_path, 'r') as f:
                lines = f.readlines()
            existing = set()
            for line in lines:
                m = re.match(r'\s*"(.*)"\s*,?\s*$', line)
                if m:
                    existing.add(m.group(1).strip())
            to_add = [p for p in new_prompts if p not in existing]
            if to_add:
                # Insert after the opening bracket (first come, first serve - most recent first)
                idx = next(i for i, l in enumerate(lines) if l.strip().endswith("["))
                for prompt in reversed(to_add):  # Reverse to maintain chronological order
                    lines.insert(idx + 1, f'    "{prompt}",\n')
                with open(pyfile_path, 'w') as f:
                    f.writelines(lines)
                print(f"[INFO] Added {len(to_add)} new prompts to {pyfile_path} (most recent first)")
        except Exception as e:
            print(f"[WARN] Could not update episodic_test_prompts.py: {e}")

    def __init__(self, 
                 num_episodes: int = 30,
                 target_score: float = 0.85,
                 max_rounds_per_prompt: int = 5,
                 log_dir: str = "episodic_logs",
                 log_path: str = "continuous_trellis.log"):
        """
        Initialize the episodic optimizer.
        
        Args:
            num_episodes: Number of episodes to run
            target_score: Target validation score for each prompt
            max_rounds_per_prompt: Maximum optimization rounds per prompt
            log_dir: Directory for storing episode logs
        """
        self.num_episodes = num_episodes
        self.target_score = target_score
        self.max_rounds_per_prompt = max_rounds_per_prompt
        self.log_dir = log_dir
        
        # Extract 0-fidelity prompts from log and add to test prompts file
        self._update_test_prompts_from_log(log_path)
        
        # Use only uncommented prompts from the imported .py file
        self.test_prompts = self._get_uncommented_prompts()
        
        # Create log directory
        os.makedirs(self.log_dir, exist_ok=True)
        
        # Setup logging
        log_file = os.path.join(self.log_dir, f"episodic_run_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        
        # Initialize the RL optimizer with episodic memory
        self.optimizer = RLLoopAgent(
            memory_file=os.path.join(self.log_dir, "episodic_memory.json")
        )
        
        # Override RL parameters to match episodic settings
        self.optimizer.max_optimization_rounds = max_rounds_per_prompt
        self.optimizer.min_score_threshold = target_score
        
        # Episode tracking
        self.episode_stats = []
        self.global_principles = []
        # Track best prompt and score for each test prompt across all episodes
        self.best_prompts = {prompt: {"score": 0.0, "prompt": prompt} for prompt in self.test_prompts}
        
    def _update_test_prompts_from_log(self, log_path: str):
        """Extract 0-fidelity prompts from log and add them to episodic_test_prompts.py"""
        try:
            zero_fid_prompts = self._extract_zero_fidelity_prompts(log_path)
            if zero_fid_prompts:
                print(f"[INFO] Found {len(zero_fid_prompts)} 0-fidelity prompts in {log_path}")
                self._add_prompts_to_pyfile(zero_fid_prompts)
            else:
                print(f"[INFO] No 0-fidelity prompts found in {log_path}")
        except Exception as e:
            print(f"[WARN] Could not update test prompts from log: {e}")
        
    def _extract_zero_fidelity_prompts(self, log_path: str) -> list:
        """Parse the log file and extract prompts with Task fidelity: 0.0000"""
        prompts = []
        try:
            with open(log_path, 'r') as f:
                lines = f.readlines()
            for i, line in enumerate(lines):
                if 'Task fidelity: 0.0000' in line:
                    # Search backwards for the 'Original:' or 'Generating 3D model:' line
                    for j in range(i-1, max(i-20, -1), -1):
                        if 'Original:' in lines[j]:
                            # Example: 'Original: spear with white head and green shaft'
                            match = re.search(r'Original:\s*(.*)', lines[j])
                            if match:
                                prompts.append(match.group(1).strip())
                                break
                        elif 'Generating 3D model:' in lines[j]:
                            # Example: 'Generating 3D model: ...'
                            match = re.search(r'Generating 3D model:\s*\'(.*?)\' \(task:', lines[j])
                            if match:
                                prompts.append(match.group(1).strip())
                                break
        except Exception as e:
            print(f"[WARN] Could not parse log for 0-fidelity prompts: {e}")
        # Return in reverse chronological order (most recent first)
        return list(reversed(prompts))

    def _check_for_new_zero_fidelity_prompts(self, log_path: str = "continuous_trellis.log"):
        """Check for new 0-fidelity prompts and update the test prompts file immediately"""
        try:
            new_prompts = self._extract_zero_fidelity_prompts(log_path)
            if new_prompts:
                # Get current prompts from the file
                current_prompts = self._get_uncommented_prompts()
                # Find truly new prompts (not already in current list)
                truly_new = [p for p in new_prompts if p not in current_prompts]
                if truly_new:
                    self._add_prompts_to_pyfile(truly_new)
                    # Reload the prompts from the updated file
                    self._reload_test_prompts()
                    self.logger.info(f"[INFO] Added {len(truly_new)} new 0-fidelity prompts during episode")
                    # Update best_prompts dict for new prompts
                    for prompt in truly_new:
                        if prompt not in self.best_prompts:
                            self.best_prompts[prompt] = {"score": 0.0, "prompt": prompt}
        except Exception as e:
            self.logger.warning(f"[WARN] Could not check for new 0-fidelity prompts: {e}")

    def _reload_test_prompts(self):
        """Reload test prompts from the updated episodic_test_prompts.py file"""
        try:
            # Reload the module to get updated prompts
            import importlib
            import episodic_test_prompts
            importlib.reload(episodic_test_prompts)
            # Update our test_prompts with the reloaded data
            self.test_prompts = [p for p in episodic_test_prompts.EPISODIC_TEST_PROMPTS if isinstance(p, str) and p.strip()]
            self.logger.info(f"[INFO] Reloaded test prompts: {len(self.test_prompts)} total prompts")
        except Exception as e:
            self.logger.warning(f"[WARN] Could not reload test prompts: {e}")
            # Fallback: manually read the file
            try:
                with open("episodic_test_prompts.py", 'r') as f:
                    content = f.read()
                # Extract prompts from the file content
                import re
                prompts = re.findall(r'"(.*?)"', content)
                self.test_prompts = [p for p in prompts if p.strip()]
                self.logger.info(f"[INFO] Manually reloaded test prompts: {len(self.test_prompts)} total prompts")
            except Exception as e2:
                self.logger.error(f"[ERROR] Failed to reload test prompts: {e2}")

    def _build_improvement_context(self, prompt: str, prompt_results: list) -> str:
        # Use best-so-far for this prompt across all episodes
        best_so_far = self.best_prompts.get(prompt, {"score": 0.0, "prompt": prompt})
        if best_so_far["score"] > 0.0:
            return (
                f"--- BEST SO FAR ---\n"
                f"Prompt: '{best_so_far['prompt']}'\n"
                f"Score: {best_so_far['score']:.3f}\n"
                "Your goal is to produce a prompt that scores higher than this. "
                "If you cannot, explain why and try a different approach or strategy. "
                "Address all feedback directly in your next attempt.\n"
            )
        else:
            return (
                "This is your first attempt for this prompt in this episode. "
                "Focus on producing the highest scoring prompt possible. "
                "If you do not succeed, analyze why and try a different approach next time.\n"
            )

    def run_single_episode(self, episode_num: int) -> Dict[str, Any]:
        """
        Run a single episode through all test prompts with dynamic updates.
        
        Args:
            episode_num: Current episode number (1-indexed)
            
        Returns:
            Dictionary containing episode statistics
        """
        self.logger.info(f"\n{'='*60}")
        self.logger.info(f"STARTING EPISODE {episode_num}/{self.num_episodes}")
        self.logger.info(f"{'='*60}")
        
        episode_start_time = time.time()
        episode_results = {
            'episode': episode_num,
            'start_time': datetime.now().isoformat(),
            'prompt_results': [],
            'episode_summary': {}
        }
        
        total_rounds = 0
        successful_optimizations = 0
        total_score_improvement = 0.0
        episode_principles = []
        
        # Track processed prompts to avoid duplicates
        processed_prompts = set()
        prompt_idx = 0
        
        # Use while loop for dynamic prompt list updates
        while True:
            # Check for new 0-fidelity prompts and update the list
            self._check_for_new_zero_fidelity_prompts()
            
            # Get current unprocessed prompts (last come, first serve)
            current_prompts = [p for p in self.test_prompts if p not in processed_prompts]
            
            if not current_prompts:
                # No more prompts to process
                break
                
            # Process the first (most recent) unprocessed prompt
            prompt = current_prompts[0]
            prompt_idx += 1
            processed_prompts.add(prompt)
            
            self.logger.info(f"\n--- Episode {episode_num}, Prompt {prompt_idx} (Total: {len(self.test_prompts)}) ---")
            self.logger.info(f"Optimizing: '{prompt}'")
            
            prompt_start_time = time.time()
            
            try:
                # Build improvement context for this prompt
                improvement_context = self._build_improvement_context(prompt, episode_results['prompt_results'])
                # Prepend context to prompt for RL agent
                prompt_with_context = f"{improvement_context}"
                max_retries = 3
                retry_count = 0
                result = None
                while retry_count < max_retries:
                    result = self.optimizer.optimize_with_rl_loop(prompt, prompt_with_context=prompt_with_context)
                    final_score = result.get('final_score', 0.0)
                    if final_score > 0.0:
                        break
                    else:
                        self.logger.warning(f"Validation score 0.0 detected (likely CUDA OOM or failure). Clearing CUDA cache and retrying ({retry_count+1}/{max_retries})...")
                        try:
                            import torch
                            if torch.cuda.is_available():
                                torch.cuda.empty_cache()
                                self.logger.info("CUDA cache cleared.")
                        except Exception as e:
                            self.logger.warning(f"Failed to clear CUDA cache: {e}")
                        retry_count += 1
                        if retry_count < max_retries:
                            time.sleep(2)
                prompt_duration = time.time() - prompt_start_time
                # Extract results
                rounds_used = result.get('total_rounds', 0)
                final_score = result.get('final_score', 0.0)
                initial_score = result.get('score_progression', [0.0])[0] if result.get('score_progression') else 0.0
                score_improvement = final_score - initial_score
                converged = result.get('convergence_achieved', False)
                # Update episode totals
                total_rounds += rounds_used
                if converged:
                    successful_optimizations += 1
                total_score_improvement += score_improvement
                # Extract any new principles learned
                if 'learned_insights' in result:
                    episode_principles.extend(result['learned_insights'])
                # Log prompt result
                prompt_result = {
                    'prompt': prompt,
                    'prompt_index': prompt_idx,
                    'rounds_used': rounds_used,
                    'initial_score': initial_score,
                    'final_score': final_score,
                    'score_improvement': score_improvement,
                    'converged': converged,
                    'duration_seconds': prompt_duration,
                    'optimized_prompt': result.get('final_optimized_prompt', prompt)
                }
                episode_results['prompt_results'].append(prompt_result)
                # Update best prompt/score for this prompt across all episodes
                if final_score > self.best_prompts[prompt]["score"]:
                    self.best_prompts[prompt]["score"] = final_score
                    self.best_prompts[prompt]["prompt"] = result.get('final_optimized_prompt', prompt)
                self.logger.info(f"Prompt optimized in {rounds_used} rounds: {initial_score:.3f} → {final_score:.3f} (+{score_improvement:.3f})")
                if converged:
                    self.logger.info("✅ Target score achieved!")
                else:
                    self.logger.info("⏰ Max rounds reached")
                    
            except Exception as e:
                self.logger.error(f"Error optimizing prompt '{prompt}': {str(e)}")
                # Add failed result
                episode_results['prompt_results'].append({
                    'prompt': prompt,
                    'prompt_index': prompt_idx,
                    'error': str(e),
                    'rounds_used': 0,
                    'converged': False
                })
        
        # Calculate episode statistics
        episode_duration = time.time() - episode_start_time
        total_prompts_processed = len(processed_prompts)
        avg_rounds = total_rounds / total_prompts_processed if total_prompts_processed > 0 else 0
        success_rate = successful_optimizations / total_prompts_processed if total_prompts_processed > 0 else 0
        avg_score_improvement = total_score_improvement / total_prompts_processed if total_prompts_processed > 0 else 0
        
        episode_summary = {
            'total_prompts': total_prompts_processed,
            'successful_optimizations': successful_optimizations,
            'success_rate': success_rate,
            'total_rounds': total_rounds,
            'avg_rounds_per_prompt': avg_rounds,
            'total_score_improvement': total_score_improvement,
            'avg_score_improvement': avg_score_improvement,
            'episode_duration_seconds': episode_duration,
            'principles_learned': episode_principles,
            'end_time': datetime.now().isoformat()
        }
        
        episode_results['episode_summary'] = episode_summary
        
        # Log episode summary
        self.logger.info(f"\n{'='*60}")
        self.logger.info(f"EPISODE {episode_num} SUMMARY")
        self.logger.info(f"{'='*60}")
        self.logger.info(f"Success Rate: {success_rate:.1%} ({successful_optimizations}/{total_prompts_processed})")
        self.logger.info(f"Average Rounds: {avg_rounds:.1f}")
        self.logger.info(f"Average Score Improvement: {avg_score_improvement:+.3f}")
        self.logger.info(f"Episode Duration: {episode_duration:.1f}s")
        self.logger.info(f"New Principles Learned: {len(episode_principles)}")
        
        # Update global principles
        self.global_principles.extend(episode_principles)
        
        return episode_results
    
    def analyze_cross_episode_learning(self) -> Dict[str, Any]:
        """
        Analyze learning patterns across episodes.
        
        Returns:
            Dictionary containing cross-episode analysis
        """
        if not self.episode_stats:
            return {}
        
        # Extract metrics across episodes
        success_rates = [ep['episode_summary']['success_rate'] for ep in self.episode_stats]
        avg_rounds = [ep['episode_summary']['avg_rounds_per_prompt'] for ep in self.episode_stats]
        avg_improvements = [ep['episode_summary']['avg_score_improvement'] for ep in self.episode_stats]
        
        # Calculate trends
        analysis = {
            'total_episodes': len(self.episode_stats),
            'success_rate_trend': {
                'first_5_episodes': statistics.mean(success_rates[:5]) if len(success_rates) >= 5 else None,
                'last_5_episodes': statistics.mean(success_rates[-5:]) if len(success_rates) >= 5 else None,
                'overall_average': statistics.mean(success_rates),
                'final_episode': success_rates[-1] if success_rates else None
            },
            'efficiency_trend': {
                'first_5_episodes_avg_rounds': statistics.mean(avg_rounds[:5]) if len(avg_rounds) >= 5 else None,
                'last_5_episodes_avg_rounds': statistics.mean(avg_rounds[-5:]) if len(avg_rounds) >= 5 else None,
                'overall_average_rounds': statistics.mean(avg_rounds),
                'final_episode_rounds': avg_rounds[-1] if avg_rounds else None
            },
            'improvement_trend': {
                'first_5_episodes_avg_improvement': statistics.mean(avg_improvements[:5]) if len(avg_improvements) >= 5 else None,
                'last_5_episodes_avg_improvement': statistics.mean(avg_improvements[-5:]) if len(avg_improvements) >= 5 else None,
                'overall_average_improvement': statistics.mean(avg_improvements),
                'final_episode_improvement': avg_improvements[-1] if avg_improvements else None
            },
            'total_principles_learned': len(self.global_principles),
            'unique_principles': len(set(self.global_principles)) if self.global_principles else 0
        }
        
        return analysis
    
    def run_all_episodes(self) -> Dict[str, Any]:
        """
        Run all episodes and return comprehensive results.
        
        Returns:
            Dictionary containing all episode results and analysis
        """
        self.logger.info(f"Starting episodic optimization: {self.num_episodes} episodes, {len(self.test_prompts)} prompts per episode")
        self.logger.info(f"Target score: {self.target_score}, Max rounds per prompt: {self.max_rounds_per_prompt}")
        
        overall_start_time = time.time()
        
        # Run each episode
        for episode_num in range(1, self.num_episodes + 1):
            try:
                episode_result = self.run_single_episode(episode_num)
                self.episode_stats.append(episode_result)
                
                # Save intermediate results
                self.save_results(episode_num)
                
                # Brief pause between episodes
                if episode_num < self.num_episodes:
                    # import time
                    time.sleep(2)
                    
            except Exception as e:
                self.logger.error(f"Error in episode {episode_num}: {str(e)}")
                continue
        
        overall_duration = time.time() - overall_start_time
        
        # Perform cross-episode analysis
        learning_analysis = self.analyze_cross_episode_learning()
        
        # Compile final results
        final_results = {
            'run_metadata': {
                'num_episodes': self.num_episodes,
                'target_score': self.target_score,
                'max_rounds_per_prompt': self.max_rounds_per_prompt,
                'total_prompts_per_episode': len(self.test_prompts),
                'total_optimizations': self.num_episodes * len(self.test_prompts),
                'overall_duration_seconds': overall_duration,
                'start_time': datetime.now().isoformat()
            },
            'episode_results': self.episode_stats,
            'learning_analysis': learning_analysis,
            'test_prompts_used': self.test_prompts
        }
        
        # Save final comprehensive results
        self.save_final_results(final_results)
        
        # Log final summary
        self.log_final_summary(learning_analysis, overall_duration)
        
        return final_results
    
    def save_results(self, episode_num: int):
        """Save intermediate results after each episode."""
        try:
            intermediate_file = os.path.join(self.log_dir, f"episodes_1_to_{episode_num}_results.json")
            with open(intermediate_file, 'w') as f:
                json.dump({
                    'episodes_completed': episode_num,
                    'episode_results': self.episode_stats,
                    'test_prompts': self.test_prompts
                }, f, indent=2)
        except Exception as e:
            self.logger.error(f"Error saving intermediate results: {str(e)}")
    
    def save_final_results(self, results: Dict[str, Any]):
        """Save final comprehensive results, including best prompts."""
        try:
            final_file = os.path.join(self.log_dir, f"final_episodic_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
            results['best_prompts'] = self.best_prompts
            with open(final_file, 'w') as f:
                json.dump(results, f, indent=2)
            self.logger.info(f"Final results saved to: {final_file}")
            # Print summary table of best prompts and scores
            self.logger.info("\nBest prompts and scores for each test prompt:")
            for prompt, data in self.best_prompts.items():
                self.logger.info(f"Prompt: {prompt}\n  Best Score: {data['score']:.3f}\n  Best Prompt: {data['prompt']}\n")
        except Exception as e:
            self.logger.error(f"Error saving final results: {str(e)}")
    
    def log_final_summary(self, learning_analysis: Dict[str, Any], overall_duration: float):
        """Log comprehensive final summary."""
        self.logger.info(f"\n{'='*80}")
        self.logger.info(f"FINAL EPISODIC OPTIMIZATION SUMMARY")
        self.logger.info(f"{'='*80}")
        
        self.logger.info(f"Total Episodes: {self.num_episodes}")
        self.logger.info(f"Total Optimizations: {self.num_episodes * len(self.test_prompts)}")
        self.logger.info(f"Overall Duration: {overall_duration:.1f}s ({overall_duration/60:.1f} minutes)")
        
        if learning_analysis:
            sr_trend = learning_analysis.get('success_rate_trend', {})
            eff_trend = learning_analysis.get('efficiency_trend', {})
            imp_trend = learning_analysis.get('improvement_trend', {})
            
            self.logger.info(f"\nLEARNING PROGRESSION:")
            
            if sr_trend.get('first_5_episodes') is not None and sr_trend.get('last_5_episodes') is not None:
                sr_change = sr_trend['last_5_episodes'] - sr_trend['first_5_episodes']
                self.logger.info(f"Success Rate: {sr_trend['first_5_episodes']:.1%} → {sr_trend['last_5_episodes']:.1%} ({sr_change:+.1%})")
            
            if eff_trend.get('first_5_episodes_avg_rounds') is not None and eff_trend.get('last_5_episodes_avg_rounds') is not None:
                rounds_change = eff_trend['last_5_episodes_avg_rounds'] - eff_trend['first_5_episodes_avg_rounds']
                self.logger.info(f"Efficiency: {eff_trend['first_5_episodes_avg_rounds']:.1f} → {eff_trend['last_5_episodes_avg_rounds']:.1f} rounds ({rounds_change:+.1f})")
            
            if imp_trend.get('first_5_episodes_avg_improvement') is not None and imp_trend.get('last_5_episodes_avg_improvement') is not None:
                imp_change = imp_trend['last_5_episodes_avg_improvement'] - imp_trend['first_5_episodes_avg_improvement']
                self.logger.info(f"Score Improvement: {imp_trend['first_5_episodes_avg_improvement']:+.3f} → {imp_trend['last_5_episodes_avg_improvement']:+.3f} ({imp_change:+.3f})")
            
            self.logger.info(f"\nPRINCIPLES LEARNED:")
            self.logger.info(f"Total Principles: {learning_analysis.get('total_principles_learned', 0)}")
            self.logger.info(f"Unique Principles: {learning_analysis.get('unique_principles', 0)}")
        
        self.logger.info(f"\n{'='*80}")


def main():
    """Main function to run the episodic optimization."""
    # Configuration
    NUM_EPISODES = 30
    TARGET_SCORE = 0.85
    MAX_ROUNDS_PER_PROMPT = 5
    
    print(f"🚀 Starting Episodic Prompt Optimization")
    print(f"Episodes: {NUM_EPISODES}")
    print(f"Target Score: {TARGET_SCORE}")
    print(f"Max Rounds per Prompt: {MAX_ROUNDS_PER_PROMPT}")
    print(f"Prompts per Episode: 13")
    print(f"Total Optimizations: {NUM_EPISODES * 13}")
    print()
    
    # Create and run the episodic optimizer
    optimizer = EpisodicPromptOptimizer(
        num_episodes=NUM_EPISODES,
        target_score=TARGET_SCORE,
        max_rounds_per_prompt=MAX_ROUNDS_PER_PROMPT
    )
    
    try:
        results = optimizer.run_all_episodes()
        print(f"\n✅ Episodic optimization completed successfully!")
        print(f"Results saved to: {optimizer.log_dir}")
        
        return results
        
    except KeyboardInterrupt:
        print(f"\n⏹️  Episodic optimization interrupted by user")
        print(f"Partial results saved to: {optimizer.log_dir}")
        return None
        
    except Exception as e:
        print(f"\n❌ Error during episodic optimization: {str(e)}")
        print(f"Partial results may be saved to: {optimizer.log_dir}")
        return None


if __name__ == "__main__":
    main() 