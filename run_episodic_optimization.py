#!/usr/bin/env python3
"""
Simple launcher for episodic prompt optimization.

Usage:
    python run_episodic_optimization.py           # Run with default settings (30 episodes)
    python run_episodic_optimization.py --episodes 10    # Run 10 episodes
    python run_episodic_optimization.py --target 0.90    # Set target score to 0.90
"""

import argparse
import sys
from episodic_prompt_optimizer import EpisodicPromptOptimizer

def main():
    parser = argparse.ArgumentParser(description='Run episodic prompt optimization')
    parser.add_argument('--episodes', type=int, default=30, 
                       help='Number of episodes to run (default: 30)')
    parser.add_argument('--target', type=float, default=0.85,
                       help='Target validation score (default: 0.85)')
    parser.add_argument('--max-rounds', type=int, default=5,
                       help='Maximum rounds per prompt (default: 5)')
    parser.add_argument('--log-dir', type=str, default='episodic_logs',
                       help='Directory for logs (default: episodic_logs)')
<<<<<<< HEAD
    
=======
    parser.add_argument('--endpoint', type=str, default='generate/',
                       help='Endpoint path, e.g. generate/ or generate/isometric_3d/')
>>>>>>> origin/multi
    args = parser.parse_args()
    
    print(f"🎯 Episodic Prompt Optimization Configuration:")
    print(f"   Episodes: {args.episodes}")
    print(f"   Target Score: {args.target}")
    print(f"   Max Rounds per Prompt: {args.max_rounds}")
    print(f"   Log Directory: {args.log_dir}")
    print(f"   Total Optimizations: {args.episodes * 13}")
<<<<<<< HEAD
=======
    print(f"   Endpoint: {args.endpoint}")
>>>>>>> origin/multi
    print()
    
    # Confirm if running many episodes
    if args.episodes > 10:
        response = input(f"This will run {args.episodes} episodes ({args.episodes * 13} total optimizations). Continue? (y/n): ")
        if response.lower() not in ['y', 'yes']:
            print("Cancelled.")
            return
    
    # Create and run optimizer
    optimizer = EpisodicPromptOptimizer(
        num_episodes=args.episodes,
        target_score=args.target,
        max_rounds_per_prompt=args.max_rounds,
        log_dir=args.log_dir, 
<<<<<<< HEAD
        endpoint="generate/baolei/"
=======
        endpoint=args.endpoint
>>>>>>> origin/multi
    )
    
    try:
        print("🚀 Starting episodic optimization...")
        results = optimizer.run_all_episodes()
        print(f"\n✅ Completed {args.episodes} episodes successfully!")
        
    except KeyboardInterrupt:
        print(f"\n⏹️  Interrupted by user. Partial results saved.")
        
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main() 