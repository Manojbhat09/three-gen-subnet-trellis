#!/usr/bin/env python3
"""
Analyze episodic_memory_nun.json and compare with test prompts and logs
"""

import json
import os
import glob
import re
from pathlib import Path

def load_episodic_memory_nun():
    """Load and parse episodic_memory_nun.json"""
    try:
        with open('episodic_memory_nun.json', 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data
    except Exception as e:
        print(f"❌ Failed to load episodic_memory_nun.json: {e}")
        return None

def load_test_prompts():
    """Load test prompts from episodic_test_prompts.py"""
    try:
        with open('episodic_test_prompts.py', 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Extract the EPISODIC_TEST_PROMPTS list
        start = content.find('EPISODIC_TEST_PROMPTS = [')
        if start == -1:
            return []
        
        # Find the end of the list
        lines = content[start:].split('\n')
        prompts = []
        in_list = False
        
        for line in lines:
            if 'EPISODIC_TEST_PROMPTS = [' in line:
                in_list = True
                continue
            if in_list:
                if line.strip() == ']':
                    break
                if line.strip().startswith('"') and line.strip().endswith(','):
                    prompt = line.strip().strip('",')
                    prompts.append(prompt)
        
        return prompts
    except Exception as e:
        print(f"❌ Failed to load test prompts: {e}")
        return []

def extract_prompts_from_logs():
    """Extract prompts from episode logs"""
    log_dir = "episodic_logs_first"
    if not os.path.exists(log_dir):
        return {}
    
    log_files = glob.glob(os.path.join(log_dir, "episodic_run_*.log"))
    if not log_files:
        return {}
    
    all_prompts = {}
    
    for log_file in log_files:
        try:
            with open(log_file, 'r', encoding='utf-8') as f:
                content = f.read()
            
            lines = content.split('\n')
            current_prompt = None
            current_score = None
            
            for line in lines:
                # Find "Original:" lines to get the prompt
                if 'Original:' in line:
                    current_prompt = line.split('Original:')[1].strip().strip("'\"")
                    current_score = None
                
                # Find "Validation score:" lines to get the score
                elif 'Validation score:' in line:
                    score_match = re.search(r'Validation score: ([\d.]+)', line)
                    if score_match and current_prompt:
                        current_score = float(score_match.group(1))
                        
                        # Create or update prompt data
                        if current_prompt not in all_prompts:
                            all_prompts[current_prompt] = {
                                'best_score': current_score,
                                'log_file': os.path.basename(log_file)
                            }
                        else:
                            # Update with better score if found
                            if current_score > all_prompts[current_prompt]['best_score']:
                                all_prompts[current_prompt]['best_score'] = current_score
                                all_prompts[current_prompt]['log_file'] = os.path.basename(log_file)
                                
        except Exception as e:
            print(f"Error reading {log_file}: {e}")
            continue
    
    return all_prompts

def analyze_episodic_memory_nun(data):
    """Analyze the structure and content of episodic_memory_nun.json"""
    if not data:
        return {}
    
    analysis = {
        'total_sessions': 0,
        'unique_prompts': set(),
        'prompt_scores': {},
        'strategy_performance': {},
        'session_details': []
    }
    
    # Extract strategy performance
    if 'strategy_performance' in data:
        analysis['strategy_performance'] = data['strategy_performance']
    
    # Extract optimization sessions
    if 'optimization_sessions' in data:
        sessions = data['optimization_sessions']
        analysis['total_sessions'] = len(sessions)
        
        for session in sessions:
            if 'original_prompt' in session:
                prompt = session['original_prompt']
                analysis['unique_prompts'].add(prompt)
                
                # Get best score from attempts
                best_score = 0.0
                if 'attempts' in session:
                    for attempt in session['attempts']:
                        if 'validation_score' in attempt:
                            score = attempt['validation_score']
                            if score > best_score:
                                best_score = score
                
                if prompt not in analysis['prompt_scores']:
                    analysis['prompt_scores'][prompt] = best_score
                else:
                    # Keep the best score across all sessions
                    analysis['prompt_scores'][prompt] = max(analysis['prompt_scores'][prompt], best_score)
                
                # Store session details
                analysis['session_details'].append({
                    'prompt': prompt,
                    'best_score': best_score,
                    'attempts': len(session.get('attempts', [])),
                    'session_id': session.get('session_id', 'unknown')
                })
    
    return analysis

def main():
    print("🔍 Analyzing episodic_memory_nun.json vs test prompts vs logs...")
    print("="*80)
    
    # Load episodic memory nun
    print("📚 Loading episodic_memory_nun.json...")
    episodic_data = load_episodic_memory_nun()
    if not episodic_data:
        return
    
    # Analyze episodic memory nun
    print("🔍 Analyzing episodic_memory_nun.json structure...")
    analysis = analyze_episodic_memory_nun(episodic_data)
    
    print(f"📊 Episodic Memory NUN Analysis:")
    print(f"   Total optimization sessions: {analysis['total_sessions']}")
    print(f"   Unique prompts: {len(analysis['unique_prompts'])}")
    
    # Load test prompts
    print("\n📝 Loading test prompts...")
    test_prompts = load_test_prompts()
    print(f"   Test prompts loaded: {len(test_prompts)}")
    
    # Load log prompts
    print("\n📖 Loading log prompts...")
    log_prompts = extract_prompts_from_logs()
    print(f"   Log prompts loaded: {len(log_prompts)}")
    
    # Compare prompts
    print("\n🔄 COMPARISON ANALYSIS:")
    print("="*80)
    
    # Check which episodic memory prompts are in test prompts
    episodic_prompts = list(analysis['unique_prompts'])
    test_prompts_set = set(test_prompts)
    log_prompts_set = set(log_prompts.keys())
    
    # Find matches
    in_test_prompts = []
    in_logs = []
    in_both = []
    in_neither = []
    
    for prompt in episodic_prompts:
        in_test = prompt in test_prompts_set
        in_log = prompt in log_prompts_set
        
        if in_test and in_log:
            in_both.append(prompt)
        elif in_test:
            in_test_prompts.append(prompt)
        elif in_log:
            in_logs.append(prompt)
        else:
            in_neither.append(prompt)
    
    print(f"📊 Prompt Coverage Analysis:")
    print(f"   In episodic memory: {len(episodic_prompts)}")
    print(f"   In test prompts: {len(in_test_prompts)}")
    print(f"   In logs: {len(in_logs)}")
    print(f"   In both test + logs: {len(in_both)}")
    print(f"   In neither: {len(in_neither)}")
    
    # Show detailed breakdown
    print(f"\n📋 Detailed Breakdown:")
    print(f"   Only in test prompts: {len(in_test_prompts)}")
    print(f"   Only in logs: {len(in_logs)}")
    print(f"   In both test + logs: {len(in_both)}")
    print(f"   Missing from both: {len(in_neither)}")
    
    # Show sample prompts from each category
    if in_test_prompts:
        print(f"\n📝 Sample prompts only in test prompts:")
        for i, prompt in enumerate(in_test_prompts[:5], 1):
            print(f"   {i}. '{prompt}'")
        if len(in_test_prompts) > 5:
            print(f"   ... and {len(in_test_prompts) - 5} more")
    
    if in_logs:
        print(f"\n📖 Sample prompts only in logs:")
        for i, prompt in enumerate(in_logs[:5], 1):
            print(f"   {i}. '{prompt}'")
        if len(in_logs) > 5:
            print(f"   ... and {len(in_logs) - 5} more")
    
    if in_both:
        print(f"\n🔄 Sample prompts in both test + logs:")
        for i, prompt in enumerate(in_both[:5], 1):
            test_score = "N/A"
            log_score = log_prompts.get(prompt, {}).get('best_score', 'N/A')
            episodic_score = analysis['prompt_scores'].get(prompt, 'N/A')
            print(f"   {i}. '{prompt}'")
            print(f"      Test: ✓, Log score: {log_score}, Episodic score: {episodic_score}")
        if len(in_both) > 5:
            print(f"   ... and {len(in_both) - 5} more")
    
    # Check for specific prompt "leather tote bag with handles"
    target_prompt = "leather tote bag with handles"
    print(f"\n🎯 Checking for specific prompt: '{target_prompt}'")
    
    in_episodic = target_prompt in episodic_prompts
    in_test = target_prompt in test_prompts_set
    in_log = target_prompt in log_prompts_set
    
    print(f"   In episodic memory: {'✓' if in_episodic else '✗'}")
    print(f"   In test prompts: {'✓' if in_test else '✗'}")
    print(f"   In logs: {'✓' if in_log else '✗'}")
    
    if in_episodic:
        episodic_score = analysis['prompt_scores'].get(target_prompt, 'N/A')
        print(f"   Episodic best score: {episodic_score}")
    
    if in_log:
        log_score = log_prompts.get(target_prompt, {}).get('best_score', 'N/A')
        log_file = log_prompts.get(target_prompt, {}).get('log_file', 'N/A')
        print(f"   Log best score: {log_score} (from {log_file})")
    
    # Show top scoring prompts from episodic memory
    print(f"\n🏆 Top 10 scoring prompts from episodic memory:")
    sorted_prompts = sorted(analysis['prompt_scores'].items(), key=lambda x: x[1], reverse=True)
    for i, (prompt, score) in enumerate(sorted_prompts[:10], 1):
        status = []
        if prompt in test_prompts_set:
            status.append("test")
        if prompt in log_prompts_set:
            status.append("log")
        status_str = "+".join(status) if status else "neither"
        print(f"   {i:2d}. Score {score:.4f} ({status_str}): '{prompt[:60]}...'")
    
    # Show strategy performance summary
    if analysis['strategy_performance']:
        print(f"\n📈 Strategy Performance Summary:")
        for strategy, data in analysis['strategy_performance'].items():
            success_rate = (data.get('success_count', 0) / max(1, data.get('total_attempts', 1))) * 100
            avg_score = data.get('avg_score', 0.0)
            print(f"   {strategy}: {success_rate:.1f}% success, avg score: {avg_score:.4f}")

if __name__ == "__main__":
    main()
