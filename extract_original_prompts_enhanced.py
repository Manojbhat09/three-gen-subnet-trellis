#!/usr/bin/env python3
"""
Extract Original Zero Fidelity Prompts (Enhanced)
=================================================
This script extracts original prompts that received zero fidelity,
distinguishing between prompts that failed with/without optimization.
"""

import sqlite3
import json

def extract_original_zero_fidelity_prompts(db_path="continuous_trellis_tasks.db"):
    """Extract original prompts with zero fidelity, preserving optimization info"""
    
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Check if enhanced columns exist
    cursor.execute("PRAGMA table_info(tasks)")
    columns = [row[1] for row in cursor.fetchall()]
    has_original_column = 'original_prompt' in columns
    
    if has_original_column:
        # Enhanced query with original prompt tracking
        query = """
        SELECT DISTINCT 
            original_prompt,
            prompt as final_prompt,
            optimization_applied,
            optimization_strategy,
            COUNT(*) as failure_count
        FROM tasks 
        WHERE task_fidelity_score = 0.0 
        AND feedback_received = 1
        AND original_prompt IS NOT NULL
        GROUP BY original_prompt, optimization_applied
        ORDER BY original_prompt
        """
    else:
        # Fallback to standard query
        query = """
        SELECT DISTINCT prompt, COUNT(*) as failure_count
        FROM tasks 
        WHERE task_fidelity_score = 0.0 
        AND feedback_received = 1
        GROUP BY prompt
        ORDER BY prompt
        """
    
    cursor.execute(query)
    results = cursor.fetchall()
    conn.close()
    
    if has_original_column:
        # Enhanced output
        original_prompts = []
        failed_despite_optimization = []
        failed_without_optimization = []
        
        for row in results:
            original, final, optimized, strategy, count = row
            original_prompts.append(original)
            
            if optimized:
                failed_despite_optimization.append({
                    'original_prompt': original,
                    'optimized_prompt': final,
                    'strategy': strategy,
                    'failure_count': count
                })
            else:
                failed_without_optimization.append({
                    'original_prompt': original,
                    'failure_count': count
                })
        
        # Save enhanced results
        with open('original_zero_fidelity_analysis.json', 'w') as f:
            json.dump({
                'unique_original_prompts': list(set(original_prompts)),
                'failed_despite_optimization': failed_despite_optimization,
                'failed_without_optimization': failed_without_optimization,
                'total_unique_originals': len(set(original_prompts)),
                'optimization_failure_count': len(failed_despite_optimization),
                'no_optimization_failure_count': len(failed_without_optimization)
            }, f, indent=2)
        
        print(f"📊 Enhanced Analysis Complete:")
        print(f"   Unique original prompts: {len(set(original_prompts))}")
        print(f"   Failed despite optimization: {len(failed_despite_optimization)}")
        print(f"   Failed without optimization: {len(failed_without_optimization)}")
        
    else:
        # Standard output
        prompts = [row[0] for row in results]
        
        with open('zero_fidelity_prompts_original.txt', 'w') as f:
            f.write("# Original Zero Fidelity Prompts\n")
            f.write(f"# Total: {len(prompts)}\n\n")
            f.write("original_zero_fidelity_prompts = [\n")
            for i, prompt in enumerate(prompts):
                if i == len(prompts) - 1:
                    f.write(f'    "{prompt}"\n')
                else:
                    f.write(f'    "{prompt}",\n')
            f.write("]\n")
        
        print(f"📊 Standard extraction: {len(prompts)} prompts")
    
    return results

if __name__ == "__main__":
    extract_original_zero_fidelity_prompts()
