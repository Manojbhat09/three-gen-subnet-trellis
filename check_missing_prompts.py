#!/usr/bin/env python3
"""
Script to check which prompts from episodic_test_prompts.py are not present 
in the episodic_memory.json log file.
"""

import json
import re
from pathlib import Path

def load_test_prompts():
    """Load prompts from episodic_test_prompts.py"""
    prompts = []
    
    # Read the file and extract prompts from the EPISODIC_TEST_PROMPTS list
    with open('episodic_test_prompts.py', 'r') as f:
        content = f.read()
    
    # Find the EPISODIC_TEST_PROMPTS list
    match = re.search(r'EPISODIC_TEST_PROMPTS\s*=\s*\[(.*?)\]', content, re.DOTALL)
    if not match:
        raise ValueError("Could not find EPISODIC_TEST_PROMPTS list in the file")
    
    list_content = match.group(1)
    
    # Extract individual prompts (handle both quoted and commented out prompts)
    lines = list_content.split('\n')
    for line in lines:
        line = line.strip()
        if line.startswith('"') and line.endswith('",'):
            # Active prompt
            prompt = line[1:-2]  # Remove quotes and comma
            prompts.append(prompt)
        elif line.startswith('# "') and line.endswith('",'):
            # Commented out prompt - skip
            continue
        elif line.startswith('"') and line.endswith('"'):
            # Last prompt in list (no comma)
            prompt = line[1:-1]
            prompts.append(prompt)
    
    return prompts

def load_episodic_memory():
    """Load original_prompt values from episodic_memory.json"""
    prompts = set()
    
    with open('episodic_logs/episodic_memory.json', 'r') as f:
        data = json.load(f)
    
    # Recursively search for "original_prompt" fields
    def extract_prompts(obj):
        if isinstance(obj, dict):
            for key, value in obj.items():
                if key == "original_prompt" and isinstance(value, str):
                    prompts.add(value)
                elif isinstance(value, (dict, list)):
                    extract_prompts(value)
        elif isinstance(obj, list):
            for item in obj:
                extract_prompts(item)
    
    extract_prompts(data)
    return prompts

def main():
    print("Loading test prompts from episodic_test_prompts.py...")
    test_prompts = load_test_prompts()
    print(f"Found {len(test_prompts)} test prompts")
    
    print("\nLoading prompts from episodic_memory.json...")
    memory_prompts = load_episodic_memory()
    print(f"Found {len(memory_prompts)} prompts in episodic memory")
    
    # Find missing prompts
    missing_prompts = []
    for prompt in test_prompts:
        if prompt not in memory_prompts:
            missing_prompts.append(prompt)
    
    print(f"\n{'='*60}")
    print(f"RESULTS:")
    print(f"{'='*60}")
    print(f"Total test prompts: {len(test_prompts)}")
    print(f"Prompts in memory: {len(memory_prompts)}")
    print(f"Missing prompts: {len(missing_prompts)}")
    print(f"Coverage: {((len(test_prompts) - len(missing_prompts)) / len(test_prompts) * 100):.1f}%")
    
    if missing_prompts:
        print(f"\n{'='*60}")
        print(f"MISSING PROMPTS ({len(missing_prompts)}):")
        print(f"{'='*60}")
        for i, prompt in enumerate(missing_prompts, 1):
            print(f"{i:3d}. {prompt}")
    else:
        print(f"\n✅ All test prompts are present in the episodic memory!")
    
    # Also show which prompts are present (optional)
    present_prompts = [p for p in test_prompts if p in memory_prompts]
    print(f"\n{'='*60}")
    print(f"PRESENT PROMPTS ({len(present_prompts)}):")
    print(f"{'='*60}")
    for i, prompt in enumerate(present_prompts, 1):
        print(f"{i:3d}. {prompt}")

if __name__ == "__main__":
    main() 