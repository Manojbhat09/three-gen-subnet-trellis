#!/usr/bin/env python3
"""
Demo script for episodic prompt optimization.
Runs just 3 episodes to demonstrate the learning system quickly.
"""

from episodic_prompt_optimizer import EpisodicPromptOptimizer
import json

def main():
    print("🔬 Demo: Episodic Prompt Optimization (3 episodes)")
    print("This demo shows how the agent learns across episodes with the same prompt set.")
    print()
    
    # Run a short demo with 3 episodes
    optimizer = EpisodicPromptOptimizer(
        num_episodes=3,
        target_score=0.85,
        max_rounds_per_prompt=3,  # Shorter for demo
        log_dir="demo_episodic_logs"
    )
    
    try:
        results = optimizer.run_all_episodes()
        
        print("\n📊 DEMO RESULTS SUMMARY:")
        print("="*50)
        
        learning_analysis = results['learning_analysis']
        
        # Show episode progression
        for i, episode in enumerate(results['episode_results'], 1):
            summary = episode['episode_summary']
            print(f"Episode {i}:")
            print(f"  Success Rate: {summary['success_rate']:.1%}")
            print(f"  Avg Rounds: {summary['avg_rounds_per_prompt']:.1f}")
            print(f"  Avg Improvement: {summary['avg_score_improvement']:+.3f}")
            print(f"  Principles Learned: {len(summary.get('principles_learned', []))}")
            print()
        
        print("Learning Trends:")
        if learning_analysis.get('success_rate_trend'):
            sr_trend = learning_analysis['success_rate_trend']
            print(f"  Success Rate: Episode 1: {sr_trend.get('overall_average', 0):.1%}")
        
        print(f"  Total Principles: {learning_analysis.get('total_principles_learned', 0)}")
        print(f"  Unique Principles: {learning_analysis.get('unique_principles', 0)}")
        
        print(f"\n✅ Demo completed! Full results in: demo_episodic_logs/")
        
    except KeyboardInterrupt:
        print("\n⏹️  Demo interrupted.")
        
    except Exception as e:
        print(f"\n❌ Demo error: {str(e)}")

if __name__ == "__main__":
    main() 