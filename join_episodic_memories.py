#!/usr/bin/env python3
"""
Script to join multiple episodic_memory.json files into a single consolidated file.

Usage:
    python join_episodic_memories.py path1/episodic_memory.json path2/episodic_memory.json [path3/episodic_memory.json ...]

The script will:
1. Load each episodic memory file
2. Merge their contents into a single structure
3. Save the consolidated result to 'consolidated_episodic_memory.json'
4. Handle conflicts by merging strategy performance data and combining optimization sessions
5. Optionally create a gold standard format file compatible with llm_close_prompt_reproducibility_test.py
"""

import json
import sys
import os
from pathlib import Path
from typing import Dict, List, Any
import argparse


def load_episodic_memory(file_path: str) -> Dict[str, Any]:
    """Load an episodic memory JSON file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        print(f"Loaded: {file_path}")
        return data
    except FileNotFoundError:
        print(f"Error: File not found: {file_path}")
        return None
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON in {file_path}: {e}")
        return None
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None


def merge_strategy_performance(existing: Dict[str, Any], new: Dict[str, Any]) -> Dict[str, Any]:
    """Merge strategy performance data, combining metrics from multiple files."""
    merged = existing.copy()
    
    for strategy_name, strategy_data in new.items():
        if strategy_name not in merged:
            merged[strategy_name] = strategy_data.copy()
        else:
            # Merge existing strategy data
            existing_strategy = merged[strategy_name]
            
            # Combine counts
            merged[strategy_name]['success_count'] = existing_strategy.get('success_count', 0) + strategy_data.get('success_count', 0)
            merged[strategy_name]['total_attempts'] = existing_strategy.get('total_attempts', 0) + strategy_data.get('total_attempts', 0)
            
            # Combine recent scores (keep most recent ones)
            existing_scores = existing_strategy.get('recent_scores', [])
            new_scores = strategy_data.get('recent_scores', [])
            all_scores = existing_scores + new_scores
            
            # Keep the most recent 10 scores
            merged[strategy_name]['recent_scores'] = all_scores[-10:] if len(all_scores) > 10 else all_scores
            
            # Recalculate average score
            if merged[strategy_name]['recent_scores']:
                merged[strategy_name]['avg_score'] = sum(merged[strategy_name]['recent_scores']) / len(merged[strategy_name]['recent_scores'])
            
            # Update confidence and other metrics (take the higher value)
            merged[strategy_name]['confidence_in_strategy'] = max(
                existing_strategy.get('confidence_in_strategy', 0),
                strategy_data.get('confidence_in_strategy', 0)
            )
            
            # Update last_used to the most recent
            merged[strategy_name]['last_used'] = max(
                existing_strategy.get('last_used', 0),
                strategy_data.get('last_used', 0)
            )
            
            # Recalculate improvement trend based on recent scores
            if len(merged[strategy_name]['recent_scores']) >= 2:
                recent_trend = merged[strategy_name]['recent_scores'][-1] - merged[strategy_name]['recent_scores'][0]
                merged[strategy_name]['improvement_trend'] = recent_trend
    
    return merged


def merge_optimization_sessions(existing: List[Dict[str, Any]], new: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Merge optimization sessions, avoiding duplicates by session_id."""
    existing_sessions = {session['session_id']: session for session in existing}
    
    for session in new:
        session_id = session.get('session_id')
        if session_id and session_id not in existing_sessions:
            existing_sessions[session_id] = session
        elif session_id in existing_sessions:
            # If session exists, keep the one with higher final_best_score
            existing_score = existing_sessions[session_id].get('final_best_score', 0)
            new_score = session.get('final_best_score', 0)
            if new_score > existing_score:
                existing_sessions[session_id] = session
    
    return list(existing_sessions.values())


def merge_episodic_memories(memory_files: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Merge multiple episodic memory files into one."""
    if not memory_files:
        return {}
    
    # Start with the first file as the base
    merged = memory_files[0].copy()
    
    # Merge subsequent files
    for memory_data in memory_files[1:]:
        if not memory_data:
            continue
            
        # Merge strategy performance
        if 'strategy_performance' in memory_data:
            merged['strategy_performance'] = merge_strategy_performance(
                merged.get('strategy_performance', {}),
                memory_data['strategy_performance']
            )
        
        # Merge optimization sessions
        if 'optimization_sessions' in memory_data:
            merged['optimization_sessions'] = merge_optimization_sessions(
                merged.get('optimization_sessions', []),
                memory_data['optimization_sessions']
            )
        
        # Merge global insights (avoid duplicates)
        if 'global_insights' in memory_data:
            existing_insights = set(merged.get('global_insights', []))
            new_insights = set(memory_data['global_insights'])
            merged['global_insights'] = list(existing_insights.union(new_insights))
        
        # Update epsilon to the average
        if 'epsilon' in memory_data:
            existing_epsilon = merged.get('epsilon', 0)
            new_epsilon = memory_data['epsilon']
            merged['epsilon'] = (existing_epsilon + new_epsilon) / 2
        
        # Update last_updated to the most recent
        if 'last_updated' in memory_data:
            merged['last_updated'] = max(
                merged.get('last_updated', 0),
                memory_data['last_updated']
            )
    
    return merged


def convert_to_gold_standard_format(episodic_memory: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convert episodic memory to the gold standard format expected by 
    llm_close_prompt_reproducibility_test.py
    """
    gold_standard_results = {}
    
    # Extract optimization sessions
    optimization_sessions = episodic_memory.get("optimization_sessions", [])
    
    for session in optimization_sessions:
        original_prompt = session.get("original_prompt", "")
        if not original_prompt:
            continue
        
        # Find the best attempt in this session
        best_attempt = None
        best_score = 0.0
        
        for attempt in session.get("attempts", []):
            validation_score = attempt.get("validation_score")
            if validation_score is not None and validation_score > best_score:
                best_score = validation_score
                best_attempt = attempt
        
        if best_attempt:
            # Store in format expected by reproducibility test
            gold_standard_results[original_prompt] = {
                "method_2_hybrid_example": {
                    "optimized_prompt": best_attempt["optimized_prompt"],
                    "validation_results": {
                        "validation_engine_score": best_score
                    }
                }
            }
    
    return gold_standard_results


def main():
    parser = argparse.ArgumentParser(
        description="Join multiple episodic memory JSON files into a single consolidated file"
    )
    parser.add_argument(
        'files',
        nargs='+',
        help='Paths to episodic_memory.json files to join'
    )
    parser.add_argument(
        '-o', '--output',
        default='consolidated_episodic_memory.json',
        help='Output file path (default: consolidated_episodic_memory.json)'
    )
    parser.add_argument(
        '--pretty',
        action='store_true',
        help='Pretty print the output JSON with indentation'
    )
    parser.add_argument(
        '--gold-standard-only',
        action='store_true',
        help='Create only the gold standard format file (for reproducibility test)'
    )
    
    args = parser.parse_args()
    
    # Validate input files
    valid_files = []
    for file_path in args.files:
        if not os.path.exists(file_path):
            print(f"Warning: File not found: {file_path}")
            continue
        if not file_path.endswith('episodic_memory.json'):
            print(f"Warning: File doesn't end with 'episodic_memory.json': {file_path}")
            continue
        valid_files.append(file_path)
    
    if not valid_files:
        print("Error: No valid episodic memory files found.")
        sys.exit(1)
    
    print(f"Found {len(valid_files)} valid episodic memory files:")
    for f in valid_files:
        print(f"  - {f}")
    
    # Load all memory files
    memory_data_list = []
    for file_path in valid_files:
        data = load_episodic_memory(file_path)
        if data:
            memory_data_list.append(data)
    
    if not memory_data_list:
        print("Error: No episodic memory data could be loaded.")
        sys.exit(1)
    
    # Merge the memories
    print(f"\nMerging {len(memory_data_list)} episodic memory files...")
    consolidated = merge_episodic_memories(memory_data_list)
    
    # Save the consolidated result
    if args.gold_standard_only:
        # Create only gold standard format
        gold_standard_path = args.output.replace('.json', '_gold_standard.json')
        gold_standard_data = convert_to_gold_standard_format(consolidated)
        
        try:
            with open(gold_standard_path, 'w', encoding='utf-8') as f:
                if args.pretty:
                    json.dump(gold_standard_data, f, indent=2, ensure_ascii=False)
                else:
                    json.dump(gold_standard_data, f, ensure_ascii=False)
            
            print(f"\n✅ Created gold standard format file: {gold_standard_path}")
            print(f"   This file is compatible with llm_close_prompt_reproducibility_test.py")
            
            # Print summary statistics
            gold_prompts_count = len(gold_standard_data)
            print(f"\nSummary:")
            print(f"  - Gold standard prompts: {gold_prompts_count}")
            
        except Exception as e:
            print(f"Error saving gold standard file: {e}")
            sys.exit(1)
    else:
        # Create both consolidated and gold standard files
        output_path = args.output
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                if args.pretty:
                    json.dump(consolidated, f, indent=2, ensure_ascii=False)
                else:
                    json.dump(consolidated, f, ensure_ascii=False)
            
            print(f"\nSuccessfully created consolidated episodic memory file: {output_path}")
            
            # Also create gold standard format for reproducibility test
            gold_standard_path = output_path.replace('.json', '_gold_standard.json')
            gold_standard_data = convert_to_gold_standard_format(consolidated)
            
            with open(gold_standard_path, 'w', encoding='utf-8') as f:
                if args.pretty:
                    json.dump(gold_standard_data, f, indent=2, ensure_ascii=False)
                else:
                    json.dump(gold_standard_data, f, ensure_ascii=False)
            
            print(f"✅ Created gold standard format file: {gold_standard_path}")
            print(f"   This file is compatible with llm_close_prompt_reproducibility_test.py")
            
            # Print summary statistics
            strategy_count = len(consolidated.get('strategy_performance', {}))
            session_count = len(consolidated.get('optimization_sessions', []))
            insight_count = len(consolidated.get('global_insights', []))
            gold_prompts_count = len(gold_standard_data)
            
            print(f"\nSummary:")
            print(f"  - Strategies: {strategy_count}")
            print(f"  - Optimization sessions: {session_count}")
            print(f"  - Global insights: {insight_count}")
            print(f"  - Gold standard prompts: {gold_prompts_count}")
            print(f"  - Epsilon: {consolidated.get('epsilon', 'N/A')}")
            
        except Exception as e:
            print(f"Error saving consolidated file: {e}")
            sys.exit(1)


if __name__ == "__main__":
    main()
