#!/usr/bin/env python3
"""
Simple launcher for episodic prompt optimization.

Usage:
    python run_episodic_optimization.py           # Run with default settings (30 episodes)
    python run_episodic_optimization.py --episodes 10    # Run 10 episodes
    python run_episodic_optimization.py --target 0.90    # Set target score to 0.90
    python run_episodic_optimization.py --port 8097      # Use TRELLIS server on port 8097
    python run_episodic_optimization.py --vllm           # Use vLLM instead of Ollama
    python run_episodic_optimization.py --reverse        # Process prompts in reverse order
    
Full example:

4:97 2:98 6:99->exp->9002
CUDA_VISIBLE_DEVICES=4 python run_episodic_optimization.py --episodes 2 --target 0.95 --max-rounds 3 --log-dir episodic_logs_first --endpoint "generate/cinema/" --port 8097 --vllm --vllm-url http://localhost:9000 --vllm-model llama-3-2-3b-it --reverse
CUDA_VISIBLE_DEVICES=2 python run_episodic_optimization.py --episodes 2 --target 0.90 --max-rounds 10 --log-dir episodic_logs_first --endpoint "generate/cinema/" --port 8098 --disable-convergence --vllm --vllm-url http://localhost:9001 --vllm-model llama-3-2-3b-it
CUDA_VISIBLE_DEVICES=0 python run_episodic_optimization.py --episodes 2 --target 0.90 --max-rounds 3 --log-dir episodic_logs_first --endpoint "generate/cinema/" --port 8100 --disable-convergence --vllm --vllm-url http://localhost:9004 --vllm-model llama-3-2-3b-it --reverse
       
    CUDA_VISIBLE_DEVICES=2 python run_episodic_optimization.py --episodes 15 --target 0.95 --max-rounds 2 --log-dir episodic_logs_first --endpoint "generate/cinema/" --ollama-url http://localhost:11434 --port 8097 --vllm --vllm-url http://localhost:9000 --vllm-model llama-3-2-3b-it --reverse
    CUDA_VISIBLE_DEVICES=2 python run_episodic_optimization.py   --episodes 2   --target 0.95   --max-rounds 15   --log-dir episodic_logs_first   --endpoint "generate/cinema/"   --ollama-url http://localhost:11434   --port 8098 --vllm --vllm-url http://localhost:9000 --vllm-model llama-3-2-3b-it --reverse
    CUDA_VISIBLE_DEVICES=2 python run_episodic_optimization.py --episodes 5 --target 0.90 --max-rounds 10 --log-dir episodic_logs_no_conv --endpoint "generate/cinema/" --ollama-url http://localhost:11434 --port 8097 --disable-convergence
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
                       help='Maximum optimization rounds per prompt (default: 5)')
    parser.add_argument('--log-dir', type=str, default='episodic_logs',
                       help='Directory for logs (default: episodic_logs)')
    parser.add_argument('--endpoint', type=str, default='generate/',
                       help='Endpoint path, e.g. generate/ or generate/isometric_3d/')
    parser.add_argument('--ollama-url', type=str, default='http://localhost:11434',
                       help='URL for the Ollama API (default: http://localhost:11434)')
    parser.add_argument('--port', type=int, default=8096,
                       help='TRELLIS server port (default: 8096)')
    parser.add_argument('--vllm', action='store_true',
                       help='Use vLLM instead of Ollama')
    parser.add_argument('--vllm-url', type=str, default='http://localhost:9000',
                       help='URL for the vLLM server (default: http://localhost:9000)')
    parser.add_argument('--vllm-model', type=str, default='llama-3-2-3b-it',
                       help='vLLM model name (default: llama-3-2-3b-it)')
    parser.add_argument('--reverse', action='store_true',
                       help='Process prompts in reverse order (oldest first instead of newest first)')
    parser.add_argument('--disable-convergence', action='store_true',
                       help='Disable convergence checking and force running for all rounds')
    args = parser.parse_args()
    
    print(f"🎯 Episodic Prompt Optimization Configuration:")
    print(f"   Episodes: {args.episodes}")
    print(f"   Target Score: {args.target}")
    print(f"   Max Rounds per Prompt: {args.max_rounds}")
    print(f"   Log Directory: {args.log_dir}")
    print(f"   Total Optimizations: {args.episodes * 13}")
    print(f"   Endpoint: {args.endpoint}")
    print(f"   Prompt Order: {'Reverse (oldest first)' if args.reverse else 'Normal (newest first)'}")
    print(f"   Convergence: {'DISABLED - Force all rounds' if args.disable_convergence else 'ENABLED - Early stop on convergence'}")
    if args.vllm:
        print(f"   Using vLLM: {args.vllm_url} with model {args.vllm_model}")
    else:
        print(f"   Ollama URL: {args.ollama_url}")
    print(f"   Server Port: {args.port}")
    print()
    
    # Print LLM provider configuration prominently
    print("🤖 LLM PROVIDER CONFIGURATION:")
    print("="*50)
    if args.vllm:
        print(f"✅ Provider: vLLM")
        print(f"✅ Server: {args.vllm_url}")
        print(f"✅ Model: {args.vllm_model}")
        print(f"✅ Status: ACTIVE for episodic optimization")
        print(f"✅ Mode: Direct server access (no queue management)")
    else:
        print(f"✅ Provider: Ollama")
        print(f"✅ Server: {args.ollama_url}")
        print(f"✅ Status: ACTIVE for episodic optimization")
        print(f"✅ Mode: Priority-based queuing with coordinator")
    print("="*50)
    print()
    
    # Confirm if running many episodes
    if args.episodes > 10:
        response = input(f"This will run {args.episodes} episodes ({args.episodes * 13} total optimizations). Continue? (y/n): ")
        if response.lower() not in ['y', 'yes']:
            print("Cancelled.")
            return
    

    # Create and run optimizer
    print("🚀 Creating Episodic Prompt Optimizer...")
    optimizer = EpisodicPromptOptimizer(
        num_episodes=args.episodes,
        target_score=args.target,
        max_rounds_per_prompt=args.max_rounds,
        log_dir=args.log_dir, 
        endpoint=args.endpoint,
        ollama_url=args.ollama_url,
        server_url=f"http://localhost:{args.port}",
        use_vllm=args.vllm,
        vllm_url=args.vllm_url,
        vllm_model=args.vllm_model,
        reverse_prompts=args.reverse,
        disable_convergence=args.disable_convergence,
    )
    optimizer._fix_prompts_file()
    try:
        print("🚀 Starting episodic optimization...")
        if args.vllm:
            print(f"   🤖 [vLLM] Using {args.vllm_url} with model {args.vllm_model}")
        else:
            print(f"   🤖 [Ollama] Using {args.ollama_url} with priority queuing")
        print(f"   📝 [Order] Processing prompts in {'reverse (oldest first)' if args.reverse else 'normal (newest first)'} order")
        results = optimizer.run_all_episodes()
        print(f"\n✅ Completed {args.episodes} episodes successfully!")
        
    except KeyboardInterrupt:
        print(f"\n⏹️  Interrupted by user. Partial results saved.")
        
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main() 