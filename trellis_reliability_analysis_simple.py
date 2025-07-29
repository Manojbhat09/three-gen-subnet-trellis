#!/usr/bin/env python3
"""
TRELLIS Reliability Analysis - Log File Analysis
Comprehensive analysis of mining reliability statistics from actual log files
"""

import re
import math
import glob
from collections import defaultdict

def parse_log_file(log_file_path):
    """Parse a log file and extract task performance data"""
    
    task_data = []
    current_task = None
    
    try:
        with open(log_file_path, 'r', encoding='utf-8') as f:
            for line in f:
                # Look for task processing start
                processing_match = re.search(r"🔄 Processing task ([a-f0-9-]+): '([^']+)'", line)
                if processing_match:
                    task_id = processing_match.group(1)
                    prompt = processing_match.group(2)
                    current_task = {
                        'task_id': task_id,
                        'prompt': prompt,
                        'fidelity': None,
                        'success': False
                    }
                
                # Look for task fidelity score
                fidelity_match = re.search(r"Task fidelity: ([\d.]+)", line)
                if fidelity_match and current_task:
                    fidelity = float(fidelity_match.group(1))
                    current_task['fidelity'] = fidelity
                    current_task['success'] = fidelity > 0.0
                    task_data.append(current_task)
                    current_task = None
                
                # Look for task completion
                completion_match = re.search(r"✅ Task ([a-f0-9-]+) completed successfully", line)
                if completion_match and current_task:
                    # If we haven't found fidelity yet, mark as failed
                    if current_task['fidelity'] is None:
                        current_task['fidelity'] = 0.0
                        current_task['success'] = False
                        task_data.append(current_task)
                    current_task = None
    
    except Exception as e:
        print(f"Error reading {log_file_path}: {e}")
        return []
    
    return task_data

def analyze_prompt_performance(task_data):
    """Analyze prompt performance patterns"""
    
    if not task_data:
        return {}
    
    # Group by prompt type/category
    prompt_categories = defaultdict(list)
    
    for task in task_data:
        prompt = task['prompt'].lower()
        
        # Categorize prompts
        category = "other"
        if any(word in prompt for word in ['robot', 'mechanical', 'metal']):
            category = "robots/mechanical"
        elif any(word in prompt for word in ['gem', 'crystal', 'stone', 'jewel']):
            category = "gems/crystals"
        elif any(word in prompt for word in ['statue', 'sculpture', 'carving']):
            category = "statues/sculptures"
        elif any(word in prompt for word in ['tool', 'instrument', 'device']):
            category = "tools/instruments"
        elif any(word in prompt for word in ['flower', 'plant', 'nature']):
            category = "nature/plants"
        elif any(word in prompt for word in ['weapon', 'sword', 'spear', 'rifle']):
            category = "weapons"
        elif any(word in prompt for word in ['jewelry', 'necklace', 'bracelet', 'ring']):
            category = "jewelry"
        elif any(word in prompt for word in ['food', 'fruit', 'cake', 'donut']):
            category = "food"
        elif any(word in prompt for word in ['animal', 'creature', 'beast']):
            category = "animals/creatures"
        elif any(word in prompt for word in ['furniture', 'chair', 'table', 'desk']):
            category = "furniture"
        
        prompt_categories[category].append(task)
    
    # Analyze each category
    category_analysis = {}
    for category, tasks in prompt_categories.items():
        if len(tasks) < 2:  # Skip categories with too few samples
            continue
            
        successful = [t for t in tasks if t['success']]
        failed = [t for t in tasks if not t['success']]
        
        success_rate = len(successful) / len(tasks) * 100
        avg_fidelity = sum(t['fidelity'] for t in tasks) / len(tasks)
        
        # Find best and worst performing prompts
        best_task = max(tasks, key=lambda x: x['fidelity'])
        worst_task = min(tasks, key=lambda x: x['fidelity'])
        
        category_analysis[category] = {
            'total_tasks': len(tasks),
            'successful_tasks': len(successful),
            'failed_tasks': len(failed),
            'success_rate': success_rate,
            'avg_fidelity': avg_fidelity,
            'best_prompt': best_task['prompt'],
            'best_fidelity': best_task['fidelity'],
            'worst_prompt': worst_task['prompt'],
            'worst_fidelity': worst_task['fidelity'],
            'tasks': tasks
        }
    
    return category_analysis

def analyze_log_files():
    """Analyze all continuous_trellis.log* files"""
    
    # Find all log files
    log_files = glob.glob("continuous_trellis.log*")
    log_files.sort()
    
    print("=" * 80)
    print("TRELLIS LOG FILE ANALYSIS")
    print("=" * 80)
    
    all_task_data = []
    log_analysis = {}
    
    for log_file in log_files:
        print(f"\n📁 Analyzing: {log_file}")
        
        # Parse log file
        task_data = parse_log_file(log_file)
        
        if not task_data:
            print(f"   ⚠️ No task data found in {log_file}")
            continue
        
        # Calculate basic statistics
        total_tasks = len(task_data)
        successful_tasks = len([t for t in task_data if t['success']])
        failed_tasks = total_tasks - successful_tasks
        reliability = (successful_tasks / total_tasks) * 100 if total_tasks > 0 else 0
        
        # Calculate fidelity statistics
        successful_fidelities = [t['fidelity'] for t in task_data if t['success']]
        avg_fidelity = sum(successful_fidelities) / len(successful_fidelities) if successful_fidelities else 0
        max_fidelity = max(t['fidelity'] for t in task_data) if task_data else 0
        min_fidelity = min(t['fidelity'] for t in task_data) if task_data else 0
        
        # Store analysis
        log_analysis[log_file] = {
            'total_tasks': total_tasks,
            'successful_tasks': successful_tasks,
            'failed_tasks': failed_tasks,
            'reliability': reliability,
            'avg_fidelity': avg_fidelity,
            'max_fidelity': max_fidelity,
            'min_fidelity': min_fidelity,
            'task_data': task_data
        }
        
        # Analyze prompt performance
        prompt_analysis = analyze_prompt_performance(task_data)
        
        print(f"   📊 Basic Statistics:")
        print(f"      Total Tasks: {total_tasks}")
        print(f"      Successful: {successful_tasks} ({reliability:.2f}%)")
        print(f"      Failed: {failed_tasks}")
        print(f"      Avg Fidelity: {avg_fidelity:.4f}")
        print(f"      Max Fidelity: {max_fidelity:.4f}")
        print(f"      Min Fidelity: {min_fidelity:.4f}")
        
        # Show prompt category analysis
        if prompt_analysis:
            print(f"   🎯 Prompt Category Analysis:")
            for category, analysis in sorted(prompt_analysis.items(), key=lambda x: x[1]['success_rate'], reverse=True):
                print(f"      {category}: {analysis['success_rate']:.1f}% ({analysis['total_tasks']} tasks)")
                print(f"         Best: '{analysis['best_prompt'][:50]}...' ({analysis['best_fidelity']:.4f})")
                if analysis['failed_tasks'] > 0:
                    print(f"         Worst: '{analysis['worst_prompt'][:50]}...' ({analysis['worst_fidelity']:.4f})")
        
        all_task_data.extend(task_data)
    
    # Overall analysis
    print(f"\n" + "=" * 80)
    print(f"OVERALL ANALYSIS")
    print(f"=" * 80)
    
    if all_task_data:
        total_tasks = len(all_task_data)
        successful_tasks = len([t for t in all_task_data if t['success']])
        failed_tasks = total_tasks - successful_tasks
        overall_reliability = (successful_tasks / total_tasks) * 100
        
        successful_fidelities = [t['fidelity'] for t in all_task_data if t['success']]
        overall_avg_fidelity = sum(successful_fidelities) / len(successful_fidelities) if successful_fidelities else 0
        
        print(f"📊 Overall Statistics:")
        print(f"   Total Tasks: {total_tasks:,}")
        print(f"   Successful: {successful_tasks:,} ({overall_reliability:.2f}%)")
        print(f"   Failed: {failed_tasks:,}")
        print(f"   Average Fidelity: {overall_avg_fidelity:.4f}")
        
        # Overall prompt analysis
        overall_prompt_analysis = analyze_prompt_performance(all_task_data)
        
        print(f"\n🎯 Overall Prompt Category Performance:")
        for category, analysis in sorted(overall_prompt_analysis.items(), key=lambda x: x[1]['success_rate'], reverse=True):
            print(f"   {category:20} | {analysis['success_rate']:5.1f}% | {analysis['total_tasks']:3d} tasks | avg: {analysis['avg_fidelity']:.3f}")
        
        # Find best and worst performing prompts overall
        best_overall = max(all_task_data, key=lambda x: x['fidelity'])
        worst_overall = min(all_task_data, key=lambda x: x['fidelity'])
        
        print(f"\n🏆 Best Overall Performance:")
        print(f"   Prompt: '{best_overall['prompt']}'")
        print(f"   Fidelity: {best_overall['fidelity']:.4f}")
        print(f"   Log: {[k for k, v in log_analysis.items() if best_overall in v['task_data']][0]}")
        
        print(f"\n❌ Worst Overall Performance:")
        print(f"   Prompt: '{worst_overall['prompt']}'")
        print(f"   Fidelity: {worst_overall['fidelity']:.4f}")
        print(f"   Log: {[k for k, v in log_analysis.items() if worst_overall in v['task_data']][0]}")
        
        # Log file comparison
        print(f"\n📈 Log File Comparison:")
        for log_file, analysis in sorted(log_analysis.items(), key=lambda x: x[1]['reliability'], reverse=True):
            print(f"   {log_file:25} | {analysis['reliability']:5.2f}% | {analysis['total_tasks']:3d} tasks | avg: {analysis['avg_fidelity']:.3f}")
    
    return log_analysis, all_task_data

def generate_text_visualizations(log_analysis):
    """Generate text-based visualizations"""
    
    print(f"\n📊 TEXT-BASED VISUALIZATIONS")
    
    # Reliability by log file
    print(f"\n   RELIABILITY BY LOG FILE:")
    print(f"   " + "=" * 60)
    for log_file, analysis in sorted(log_analysis.items(), key=lambda x: x[1]['reliability'], reverse=True):
        bar_length = int(analysis['reliability'] / 2)  # Scale to 50 characters max
        bar = "█" * bar_length + "░" * (50 - bar_length)
        print(f"   {log_file:20} |{bar}| {analysis['reliability']:5.1f}%")
    
    # Task volume by log file
    print(f"\n   TASK VOLUME BY LOG FILE:")
    print(f"   " + "=" * 60)
    max_tasks = max(analysis['total_tasks'] for analysis in log_analysis.values())
    for log_file, analysis in sorted(log_analysis.items(), key=lambda x: x[1]['total_tasks'], reverse=True):
        volume_scale = int((analysis['total_tasks'] / max_tasks) * 40)  # Scale to 40 characters max
        volume_bar = "█" * volume_scale + "░" * (40 - volume_scale)
        print(f"   {log_file:20} |{volume_bar}| {analysis['total_tasks']:4d} tasks")

def main():
    """Main analysis function"""
    
    # Analyze all log files
    log_analysis, all_task_data = analyze_log_files()
    
    # Generate visualizations
    if log_analysis:
        generate_text_visualizations(log_analysis)
    
    print(f"\n" + "=" * 80)
    print(f"ANALYSIS COMPLETE")
    print(f"=" * 80)

if __name__ == "__main__":
    main() 