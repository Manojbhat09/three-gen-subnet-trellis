#!/usr/bin/env python3
"""
Convergence Analysis
===================
Analyzes the current convergence behavior and compares explore vs exploit performance
to identify if convergence criteria is stopping optimization too early.
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path

def analyze_convergence_behavior(memory_file="episodic_logs/episodic_memory.json"):
    """Analyze convergence behavior and explore vs exploit performance"""
    
    # Load the JSON data
    with open(memory_file, 'r') as f:
        data = json.load(f)
    
    # Extract all attempts with convergence info
    attempts_data = []
    for session in data.get('optimization_sessions', []):
        session_id = session['session_id']
        original_prompt = session['original_prompt']
        convergence_achieved = session.get('convergence_achieved', False)
        final_best_score = session.get('final_best_score', 0.0)
        
        for attempt in session.get('attempts', []):
            attempts_data.append({
                'session_id': session_id,
                'original_prompt': original_prompt,
                'attempt_number': attempt['attempt_number'],
                'strategy_used': attempt['strategy_used'],
                'exploration_type': attempt['exploration_type'],
                'validation_score': attempt.get('validation_score', 0.0),
                'convergence_achieved': convergence_achieved,
                'final_best_score': final_best_score,
                'total_rounds': len(session.get('attempts', [])),
                'max_rounds': 5  # Current default
            })
    
    df = pd.DataFrame(attempts_data)
    
    if len(df) == 0:
        print("No data found in the memory file.")
        return
    
    print("=== CONVERGENCE BEHAVIOR ANALYSIS ===")
    print(f"Total sessions: {df['session_id'].nunique()}")
    print(f"Total attempts: {len(df)}")
    
    # Analyze convergence patterns
    convergence_stats = df.groupby('session_id').agg({
        'convergence_achieved': 'first',
        'final_best_score': 'first',
        'total_rounds': 'first',
        'max_rounds': 'first'
    })
    
    converged_sessions = convergence_stats[convergence_stats['convergence_achieved'] == True]
    non_converged_sessions = convergence_stats[convergence_stats['convergence_achieved'] == False]
    
    print(f"\n=== CONVERGENCE STATISTICS ===")
    print(f"Converged sessions: {len(converged_sessions)} ({len(converged_sessions)/len(convergence_stats)*100:.1f}%)")
    print(f"Non-converged sessions: {len(non_converged_sessions)} ({len(non_converged_sessions)/len(convergence_stats)*100:.1f}%)")
    
    if len(converged_sessions) > 0:
        print(f"Average rounds in converged sessions: {converged_sessions['total_rounds'].mean():.1f}")
        print(f"Average final score in converged sessions: {converged_sessions['final_best_score'].mean():.3f}")
    
    if len(non_converged_sessions) > 0:
        print(f"Average rounds in non-converged sessions: {non_converged_sessions['total_rounds'].mean():.1f}")
        print(f"Average final score in non-converged sessions: {non_converged_sessions['final_best_score'].mean():.3f}")
    
    # Analyze explore vs exploit performance
    print(f"\n=== EXPLORE VS EXPLOIT ANALYSIS ===")
    explore_data = df[df['exploration_type'] == 'explore']
    exploit_data = df[df['exploration_type'] == 'exploit']
    
    print(f"Explore attempts: {len(explore_data)}")
    print(f"Exploit attempts: {len(exploit_data)}")
    
    if len(explore_data) > 0:
        print(f"Explore average score: {explore_data['validation_score'].mean():.3f}")
        print(f"Explore score std: {explore_data['validation_score'].std():.3f}")
        print(f"Explore success rate (>0.7): {(explore_data['validation_score'] > 0.7).mean():.1%}")
    
    if len(exploit_data) > 0:
        print(f"Exploit average score: {exploit_data['validation_score'].mean():.3f}")
        print(f"Exploit score std: {exploit_data['validation_score'].std():.3f}")
        print(f"Exploit success rate (>0.7): {(exploit_data['validation_score'] > 0.7).mean():.1%}")
    
    # Analyze what happens after convergence
    print(f"\n=== POST-CONVERGENCE ANALYSIS ===")
    
    # Find sessions that converged early but had room for more rounds
    early_converged = converged_sessions[converged_sessions['total_rounds'] < converged_sessions['max_rounds']]
    print(f"Early converged sessions: {len(early_converged)}")
    
    if len(early_converged) > 0:
        print(f"Average rounds used in early converged: {early_converged['total_rounds'].mean():.1f}")
        print(f"Average final score in early converged: {early_converged['final_best_score'].mean():.3f}")
        
        # Check if these sessions could have benefited from more exploration
        early_converged_sessions = early_converged.index.tolist()
        early_converged_attempts = df[df['session_id'].isin(early_converged_sessions)]
        
        # Check if the last few attempts were mostly exploit
        for session_id in early_converged_sessions[:5]:  # Show first 5 examples
            session_attempts = early_converged_attempts[early_converged_attempts['session_id'] == session_id].sort_values('attempt_number')
            last_attempts = session_attempts.tail(2)  # Last 2 attempts
            exploit_count = (last_attempts['exploration_type'] == 'exploit').sum()
            print(f"Session {session_id}: Last 2 attempts - {exploit_count}/2 were exploit")
    
    # Analyze improvement patterns
    print(f"\n=== IMPROVEMENT PATTERNS ===")
    
    # Calculate improvement between consecutive attempts
    improvements = []
    for session_id in df['session_id'].unique():
        session_data = df[df['session_id'] == session_id].sort_values('attempt_number')
        for i in range(1, len(session_data)):
            improvement = session_data['validation_score'].iloc[i] - session_data['validation_score'].iloc[i-1]
            improvements.append({
                'session_id': session_id,
                'attempt_pair': f"{session_data['attempt_number'].iloc[i-1]}-{session_data['attempt_number'].iloc[i]}",
                'improvement': improvement,
                'exploration_type': session_data['exploration_type'].iloc[i],
                'strategy': session_data['strategy_used'].iloc[i]
            })
    
    if improvements:
        improvements_df = pd.DataFrame(improvements)
        positive_improvements = improvements_df[improvements_df['improvement'] > 0]
        
        print(f"Total improvement opportunities: {len(improvements_df)}")
        print(f"Positive improvements: {len(positive_improvements)} ({len(positive_improvements)/len(improvements_df)*100:.1f}%)")
        print(f"Average improvement: {improvements_df['improvement'].mean():.3f}")
        print(f"Average positive improvement: {positive_improvements['improvement'].mean():.3f}")
        
        # Analyze by exploration type
        explore_improvements = improvements_df[improvements_df['exploration_type'] == 'explore']
        exploit_improvements = improvements_df[improvements_df['exploration_type'] == 'exploit']
        
        if len(explore_improvements) > 0:
            print(f"Explore improvements: {explore_improvements['improvement'].mean():.3f} avg")
        if len(exploit_improvements) > 0:
            print(f"Exploit improvements: {exploit_improvements['improvement'].mean():.3f} avg")
    
    # Strategy-specific convergence analysis
    print(f"\n=== STRATEGY CONVERGENCE ANALYSIS ===")
    strategy_convergence = df.groupby('strategy_used').agg({
        'validation_score': ['mean', 'std', 'count'],
        'exploration_type': lambda x: (x == 'explore').mean()
    }).round(3)
    strategy_convergence.columns = ['avg_score', 'std_score', 'attempts', 'explore_ratio']
    strategy_convergence = strategy_convergence.sort_values('avg_score', ascending=False)
    
    for strategy, stats in strategy_convergence.iterrows():
        print(f"{strategy}: {stats['avg_score']:.3f} avg, {stats['explore_ratio']:.1%} explore ratio")
    
    return {
        'convergence_stats': convergence_stats,
        'explore_performance': explore_data['validation_score'].mean() if len(explore_data) > 0 else 0,
        'exploit_performance': exploit_data['validation_score'].mean() if len(exploit_data) > 0 else 0,
        'early_converged_count': len(early_converged),
        'improvements_df': improvements_df if 'improvements_df' in locals() else None
    }

if __name__ == "__main__":
    try:
        results = analyze_convergence_behavior()
        
        print(f"\n=== SUMMARY & RECOMMENDATIONS ===")
        explore_better = results['explore_performance'] > results['exploit_performance']
        print(f"Explore performs better than exploit: {explore_better}")
        print(f"Early converged sessions: {results['early_converged_count']}")
        
        if explore_better:
            print("RECOMMENDATION: Increase exploration rate and adjust convergence criteria")
            print("- Consider increasing epsilon (exploration rate)")
            print("- Relax convergence threshold (currently 0.02)")
            print("- Add minimum rounds before convergence")
            print("- Implement adaptive convergence based on exploration success")
        
    except Exception as e:
        print(f"Error analyzing convergence: {e}")
        import traceback
        traceback.print_exc() 