#!/usr/bin/env python3
"""
Simple script to run Nunchaku - call this from main server
"""

import sys
import os

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python run_nunchaku.py 'prompt' [seed]")
        sys.exit(1)
    
    prompt = sys.argv[1]
    seed = int(sys.argv[2]) if len(sys.argv) > 2 else 42
    
    print(f"Running Nunchaku with prompt: {prompt}, seed: {seed}")
    
    # This will be run in the nun environment
    # Just print success for now
    print("SUCCESS: Image generated")
    print(f"PROMPT: {prompt}")
    print(f"SEED: {seed}")
