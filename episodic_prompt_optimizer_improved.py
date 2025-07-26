#!/usr/bin/env python3
"""
Improved Episodic Prompt Optimizer
==================================
Enhanced episodic optimization using the improved RL agent with:
- Better convergence logic (minimum rounds, adaptive thresholds)
- Enhanced exploration/exploitation balance
- Performance-based epsilon adjustment
- Adaptive convergence based on exploration success
"""

import os
import sys
import time
import logging
import argparse
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any

# Import the improved RL agent
try:
    from smart_prompt_optimizer_v4_1_rl_loop_improved import ImprovedRLLoopAgent
except ImportError:
    print("Error: Could not import ImprovedRLLoopAgent. Make sure smart_prompt_optimizer_v4_1_rl_loop_improved.py exists.")
    sys.exit(1)

class ImprovedEpisodicOptimizer:
    """Enhanced episodic optimizer with improved convergence and exploration"""
    
    def __init__(self, episodes: int = 30, target_score: float = 0.96, 
                 max_rounds: int = 15, log_dir: str = "episodic_logs_improved"):
        self.episodes = episodes
        self.target_score = target_score
        self.max_rounds = max_rounds
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)
        
        # Test prompts (same as original)
        self.test_prompts = [
            "sapphire-studded sharp spear",
            "emerald pendant",
            "bottle of red wine with cork in it",
            "crystal staff with swirling light",
            "harp adorned with pearl inlays and gilded frame",
            "necklace with heart-shaped pendant made of silver and turquoise stones",
            "bottle of red wine with cork in it",
            "cupcake with chocolate icing on top",
            "matte black candle holder two interlocking pieces",
            "greek kylix cup black-figure technique mythological scenes",
            "small round blue creature with long nose and pointed ears",
            "tall glass of layered lemonade",
            "cylindrical glass of bubbly lemonade"
        ]
        
        # Initialize improved RL agent
        self.optimizer = ImprovedRLLoopAgent(memory_file="episodic_logs_improved/episodic_memory_improved.json")
        
        # Set improved parameters
        self.optimizer.max_optimization_rounds = max_rounds
        self.optimizer.min_score_threshold = target_score
        self.optimizer.min_rounds_before_convergence = 5  # Minimum rounds before convergence
        
        # Setup logging
        self._setup_logging()
        
        # Episode tracking
        self.episode_results = []
        self.prompt_performance = {}
        
    def _setup_logging(self):
        """Setup enhanced logging for improved optimizer"""
        log_file = self.log_dir / f"improved_episodic_run_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        
        self.logger.info("🚀 IMPROVED EPISODIC OPTIMIZER INITIALIZED")
        self.logger.info(f"   Episodes: {self.episodes}")
        self.logger.info(f"   Target Score: {self.target_score}")
        self.logger.info(f"   Max Rounds per Prompt: {self.max_rounds}")
        self.logger.info(f"   Min Rounds before Convergence: {self.optimizer.min_rounds_before_convergence}")
        self.logger.info(f"   Log Directory: {self.log_dir}")
        self.logger.info(f"   Total Optimizations: {self.episodes * len(self.test_prompts)}")
        
    def run_episodic_optimization(self):
        """Run the improved episodic optimization"""
        self.logger.info("🚀 Starting improved episodic optimization...")
        self.logger.info(f"Starting improved episodic optimization: {self.episodes} episodes, {len(self.test_prompts)} prompts per episode")
        self.logger.info(f"Target score: {self.target_score}, Max rounds per prompt: {self.max_rounds}")
        
        start_time = time.time()
        
        for episode in range(1, self.episodes + 1):
            self.logger.info(f"\n{'='*60}")
            self.logger.info(f"STARTING EPISODE {episode}/{self.episodes}")
            self.logger.info(f"{'='*60}")
            
            episode_start_time = time.time()
            episode_results = []
            
            for prompt_idx, prompt in enumerate(self.test_prompts, 1):
                self.logger.info(f"\n--- Episode {episode}, Prompt {prompt_idx}/{len(self.test_prompts)} ---")
                self.logger.info(f"Optimizing: '{prompt}'")
                
                # Track initial score (0 for new prompts)
                initial_score = 0.0
                
                # Run optimization with improved RL agent
                try:
                    result = self.optimizer.optimize_with_rl_loop(prompt, use_validation=True)
                    
                    final_score = result['final_score']
                    improvement = final_score - initial_score
                    rounds_used = result['total_rounds']
                    convergence_reason = result['convergence_reason']
                    exploration_ratio = result['exploration_ratio']
                    
                    # Log results
                    self.logger.info(f"Prompt optimized in {rounds_used} rounds: {initial_score:.3f} → {final_score:.3f} (+{improvement:.3f})")
                    self.logger.info(f"Convergence: {result['convergence_achieved']} - {convergence_reason}")
                    self.logger.info(f"Exploration ratio: {exploration_ratio:.1%}")
                    
                    # Check if max rounds reached
                    if rounds_used >= self.max_rounds:
                        self.logger.info("⏰ Max rounds reached")
                    
                    # Store results
                    episode_results.append({
                        'prompt': prompt,
                        'initial_score': initial_score,
                        'final_score': final_score,
                        'improvement': improvement,
                        'rounds_used': rounds_used,
                        'convergence_achieved': result['convergence_achieved'],
                        'convergence_reason': convergence_reason,
                        'exploration_ratio': exploration_ratio,
                        'processing_time': result['processing_time'],
                        'strategy_sequence': result['strategy_sequence'],
                        'score_progression': result['score_progression']
                    })
                    
                    # Update prompt performance tracking
                    if prompt not in self.prompt_performance:
                        self.prompt_performance[prompt] = []
                    self.prompt_performance[prompt].append({
                        'episode': episode,
                        'final_score': final_score,
                        'improvement': improvement,
                        'rounds_used': rounds_used,
                        'exploration_ratio': exploration_ratio
                    })
                    
                except Exception as e:
                    self.logger.error(f"❌ Error optimizing prompt '{prompt}': {e}")
                    episode_results.append({
                        'prompt': prompt,
                        'error': str(e),
                        'final_score': 0.0,
                        'improvement': 0.0
                    })
            
            # Episode summary
            episode_duration = time.time() - episode_start_time
            successful_optimizations = [r for r in episode_results if 'error' not in r]
            
            if successful_optimizations:
                avg_score = sum(r['final_score'] for r in successful_optimizations) / len(successful_optimizations)
                avg_improvement = sum(r['improvement'] for r in successful_optimizations) / len(successful_optimizations)
                avg_rounds = sum(r['rounds_used'] for r in successful_optimizations) / len(successful_optimizations)
                avg_exploration = sum(r['exploration_ratio'] for r in successful_optimizations) / len(successful_optimizations)
                
                self.logger.info(f"\n📊 EPISODE {episode} SUMMARY:")
                self.logger.info(f"   Average final score: {avg_score:.3f}")
                self.logger.info(f"   Average improvement: {avg_improvement:.3f}")
                self.logger.info(f"   Average rounds used: {avg_rounds:.1f}")
                self.logger.info(f"   Average exploration ratio: {avg_exploration:.1%}")
                self.logger.info(f"   Episode duration: {episode_duration:.1f}s")
            
            self.episode_results.append({
                'episode': episode,
                'results': episode_results,
                'duration': episode_duration
            })
            
            # Save intermediate results
            self._save_episode_results(episode, episode_results)
        
        # Final summary
        total_duration = time.time() - start_time
        self._generate_final_summary(total_duration)
        
    def _save_episode_results(self, episode: int, results: List[Dict[str, Any]]):
        """Save episode results to file"""
        episode_file = self.log_dir / f"episode_{episode}_results.json"
        
        episode_data = {
            'episode': episode,
            'timestamp': datetime.now().isoformat(),
            'results': results,
            'summary': {
                'total_prompts': len(results),
                'successful_optimizations': len([r for r in results if 'error' not in r]),
                'average_score': sum(r['final_score'] for r in results if 'error' not in r) / len([r for r in results if 'error' not in r]) if any('error' not in r for r in results) else 0.0,
                'average_improvement': sum(r['improvement'] for r in results if 'error' not in r) / len([r for r in results if 'error' not in r]) if any('error' not in r for r in results) else 0.0,
                'average_rounds': sum(r['rounds_used'] for r in results if 'error' not in r) / len([r for r in results if 'error' not in r]) if any('error' not in r for r in results) else 0.0,
                'average_exploration': sum(r['exploration_ratio'] for r in results if 'error' not in r) / len([r for r in results if 'error' not in r]) if any('error' not in r for r in results) else 0.0
            }
        }
        
        import json
        with open(episode_file, 'w') as f:
            json.dump(episode_data, f, indent=2)
    
    def _generate_final_summary(self, total_duration: float):
        """Generate comprehensive final summary"""
        self.logger.info(f"\n{'='*60}")
        self.logger.info("🎯 IMPROVED EPISODIC OPTIMIZATION COMPLETE")
        self.logger.info(f"{'='*60}")
        
        # Overall statistics
        all_results = []
        for episode_data in self.episode_results:
            all_results.extend(episode_data['results'])
        
        successful_results = [r for r in all_results if 'error' not in r]
        
        if successful_results:
            avg_final_score = sum(r['final_score'] for r in successful_results) / len(successful_results)
            avg_improvement = sum(r['improvement'] for r in successful_results) / len(successful_results)
            avg_rounds = sum(r['rounds_used'] for r in successful_results) / len(successful_results)
            avg_exploration = sum(r['exploration_ratio'] for r in successful_results) / len(successful_results)
            
            self.logger.info(f"📊 OVERALL STATISTICS:")
            self.logger.info(f"   Total episodes: {self.episodes}")
            self.logger.info(f"   Total optimizations: {len(all_results)}")
            self.logger.info(f"   Successful optimizations: {len(successful_results)}")
            self.logger.info(f"   Average final score: {avg_final_score:.3f}")
            self.logger.info(f"   Average improvement: {avg_improvement:.3f}")
            self.logger.info(f"   Average rounds per optimization: {avg_rounds:.1f}")
            self.logger.info(f"   Average exploration ratio: {avg_exploration:.1%}")
            self.logger.info(f"   Total duration: {total_duration:.1f}s")
            
            # Learning insights
            insights = self.optimizer.get_rl_insights()
            self.logger.info(f"\n🧠 LEARNING INSIGHTS:")
            self.logger.info(f"   Current exploration rate: {insights['current_exploration_rate']:.2f}")
            self.logger.info(f"   Exploration performance: {insights['exploration_performance']:.3f}")
            self.logger.info(f"   Exploitation performance: {insights['exploitation_performance']:.3f}")
            self.logger.info(f"   Average rounds per session: {insights['average_rounds_per_session']:.1f}")
            self.logger.info(f"   Convergence rate: {insights['convergence_rate']:.1%}")
            
            # Prompt performance analysis
            self.logger.info(f"\n📈 PROMPT PERFORMANCE ANALYSIS:")
            for prompt, performances in self.prompt_performance.items():
                if len(performances) > 1:
                    scores = [p['final_score'] for p in performances]
                    avg_score = sum(scores) / len(scores)
                    best_score = max(scores)
                    worst_score = min(scores)
                    self.logger.info(f"   '{prompt[:30]}...': avg={avg_score:.3f}, best={best_score:.3f}, worst={worst_score:.3f}")
        
        # Save final summary
        final_summary = {
            'total_episodes': self.episodes,
            'total_optimizations': len(all_results),
            'successful_optimizations': len(successful_results),
            'total_duration': total_duration,
            'prompt_performance': self.prompt_performance,
            'episode_results': self.episode_results,
            'final_insights': self.optimizer.get_rl_insights()
        }
        
        import json
        with open(self.log_dir / 'final_summary.json', 'w') as f:
            json.dump(final_summary, f, indent=2)
        
        self.logger.info(f"\n💾 Results saved to: {self.log_dir}")
        self.logger.info("🎉 Improved episodic optimization complete!")

def main():
    """Main function to run improved episodic optimization"""
    parser = argparse.ArgumentParser(description='Improved Episodic Prompt Optimizer')
    parser.add_argument('--episodes', type=int, default=30, help='Number of episodes to run')
    parser.add_argument('--target', type=float, default=0.96, help='Target score threshold')
    parser.add_argument('--max-rounds', type=int, default=15, help='Maximum rounds per prompt')
    parser.add_argument('--log-dir', type=str, default='episodic_logs_improved', help='Log directory')
    
    args = parser.parse_args()
    
    print("🎯 Improved Episodic Prompt Optimization Configuration:")
    print(f"   Episodes: {args.episodes}")
    print(f"   Target Score: {args.target}")
    print(f"   Max Rounds per Prompt: {args.max_rounds}")
    print(f"   Log Directory: {args.log_dir}")
    print(f"   Total Optimizations: {args.episodes * 13}")
    print()
    
    optimizer = ImprovedEpisodicOptimizer(
        episodes=args.episodes,
        target_score=args.target,
        max_rounds=args.max_rounds,
        log_dir=args.log_dir
    )
    
    try:
        optimizer.run_episodic_optimization()
    except KeyboardInterrupt:
        print("\n⚠️ Interrupted by user")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 