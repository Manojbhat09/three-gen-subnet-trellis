#!/usr/bin/env python3
"""
TRELLIS Detailed Reliability Analysis
Focused analysis on specific score ranges and failure patterns
"""

import re
import statistics
from collections import defaultdict
from typing import Dict, List, Any

class TrellisDetailedAnalyzer:
    """Detailed analyzer for specific reliability metrics"""
    
    def __init__(self):
        self.v0_log = "continuous_trellis_simulator.log.v0"
        self.v1_log = "continuous_trellis_simulator.log.v1"
        self.v2_log = "continuous_trellis_simulator.log.v2"
        self.v3_log = "continuous_trellis_simulator.log.v3"
        
    def parse_log_file_detailed(self, log_file_path: str) -> List[Dict[str, Any]]:
        """Parse log file with detailed metrics"""
        
        task_data = []
        current_task = None
        
        try:
            with open(log_file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    # Look for task processing start
                    processing_match = re.search(r"🔄 Processing task ([a-zA-Z0-9_]+): '([^']+)'", line)
                    if processing_match:
                        task_id = processing_match.group(1)
                        prompt = processing_match.group(2)
                        current_task = {
                            'task_id': task_id,
                            'prompt': prompt,
                            'validation_engine_score': None,
                            'alignment_score': None,
                            'quality_score': None,
                            'demo_fidelity_score': None,
                            'task_fidelity_score': None,
                            'validation_passed': False,
                            'generation_success': False,
                            'validation_success': False,
                            'validation_failed': False,
                            'cuda_oom': False,
                            'generation_time': None,
                            'validation_time': None,
                            'optimization_applied': False,
                            'reproducibility_optimization': False,
                            'traditional_optimization': False
                        }
                    
                    # Look for validation engine score
                    engine_match = re.search(r"🏆 Validation Engine Score: ([\d.]+)", line)
                    if engine_match and current_task:
                        current_task['validation_engine_score'] = float(engine_match.group(1))
                        current_task['validation_success'] = True
                    
                    # Look for alignment score
                    alignment_match = re.search(r"🤝 Alignment Score: ([\d.]+)", line)
                    if alignment_match and current_task:
                        current_task['alignment_score'] = float(alignment_match.group(1))
                    
                    # Look for quality score
                    quality_match = re.search(r"💎 Quality Score: ([\d.]+)", line)
                    if quality_match and current_task:
                        current_task['quality_score'] = float(quality_match.group(1))
                    
                    # Look for demo fidelity score
                    demo_match = re.search(r"🎭 Demo Fidelity Score: ([\d.]+)", line)
                    if demo_match and current_task:
                        current_task['demo_fidelity_score'] = float(demo_match.group(1))
                    
                    # Look for task fidelity score
                    task_match = re.search(r"🎯 Task Fidelity Score: ([\d.]+)", line)
                    if task_match and current_task:
                        current_task['task_fidelity_score'] = float(task_match.group(1))
                    
                    # Look for validation passed
                    passed_match = re.search(r"✅ Validation Passed: (True|False)", line)
                    if passed_match and current_task:
                        current_task['validation_passed'] = passed_match.group(1) == 'True'
                    
                    # Look for generation success
                    gen_success_match = re.search(r"✅ Generation successful in ([\d.]+)s", line)
                    if gen_success_match and current_task:
                        current_task['generation_success'] = True
                        current_task['generation_time'] = float(gen_success_match.group(1))
                    
                    # Look for validation completion
                    val_success_match = re.search(r"✅ Validation completed in ([\d.]+)s", line)
                    if val_success_match and current_task:
                        current_task['validation_time'] = float(val_success_match.group(1))
                    
                    # Look for validation failures
                    val_fail_match = re.search(r"❌ Validation failed", line)
                    if val_fail_match and current_task:
                        current_task['validation_failed'] = True
                    
                    # Look for CUDA OOM
                    cuda_match = re.search(r"CUDA.*out of memory|CUDA OOM", line, re.IGNORECASE)
                    if cuda_match and current_task:
                        current_task['cuda_oom'] = True
                    
                    # Look for optimization applied
                    repro_match = re.search(r"🔄 Reproducibility optimization applied", line)
                    if repro_match and current_task:
                        current_task['optimization_applied'] = True
                        current_task['reproducibility_optimization'] = True
                    
                    trad_match = re.search(r"🚀 Traditional optimization applied", line)
                    if trad_match and current_task:
                        current_task['optimization_applied'] = True
                        current_task['traditional_optimization'] = True
                    
                    # Look for task completion
                    completion_match = re.search(r"✅ Task ([a-zA-Z0-9_]+) finished processing", line)
                    if completion_match and current_task:
                        # If we haven't found validation engine score yet, mark as failed
                        if current_task['validation_engine_score'] is None:
                            current_task['validation_engine_score'] = 0.0
                            current_task['validation_failed'] = True
                        task_data.append(current_task)
                        current_task = None
                
                # Handle any remaining task
                if current_task:
                    if current_task['validation_engine_score'] is None:
                        current_task['validation_engine_score'] = 0.0
                        current_task['validation_failed'] = True
                    task_data.append(current_task)
        
        except Exception as e:
            print(f"Error reading {log_file_path}: {e}")
            return []
        
        return task_data
    
    def analyze_zero_score_prompts(self, task_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Detailed analysis of zero-score prompts"""
        
        zero_score_tasks = [task for task in task_data if task['validation_engine_score'] == 0.0]
        
        if not zero_score_tasks:
            return {'count': 0, 'tasks': []}
        
        analysis = {
            'count': len(zero_score_tasks),
            'percentage': (len(zero_score_tasks) / len(task_data)) * 100,
            'tasks': []
        }
        
        for task in zero_score_tasks:
            task_analysis = {
                'task_id': task['task_id'],
                'prompt': task['prompt'],
                'validation_engine_score': task['validation_engine_score'],
                'alignment_score': task['alignment_score'],
                'quality_score': task['quality_score'],
                'demo_fidelity_score': task['demo_fidelity_score'],
                'task_fidelity_score': task['task_fidelity_score'],
                'validation_passed': task['validation_passed'],
                'generation_success': task['generation_success'],
                'validation_failed': task['validation_failed'],
                'cuda_oom': task['cuda_oom'],
                'optimization_applied': task['optimization_applied'],
                'reproducibility_optimization': task['reproducibility_optimization'],
                'traditional_optimization': task['traditional_optimization']
            }
            analysis['tasks'].append(task_analysis)
        
        # Analyze component scores for zero-score tasks
        alignment_scores = [t['alignment_score'] for t in zero_score_tasks if t['alignment_score'] is not None]
        quality_scores = [t['quality_score'] for t in zero_score_tasks if t['quality_score'] is not None]
        demo_fidelity_scores = [t['demo_fidelity_score'] for t in zero_score_tasks if t['demo_fidelity_score'] is not None]
        task_fidelity_scores = [t['task_fidelity_score'] for t in zero_score_tasks if t['task_fidelity_score'] is not None]
        
        if alignment_scores:
            analysis['alignment_stats'] = {
                'avg': statistics.mean(alignment_scores),
                'median': statistics.median(alignment_scores),
                'min': min(alignment_scores),
                'max': max(alignment_scores),
                'count': len(alignment_scores)
            }
        
        if quality_scores:
            analysis['quality_stats'] = {
                'avg': statistics.mean(quality_scores),
                'median': statistics.median(quality_scores),
                'min': min(quality_scores),
                'max': max(quality_scores),
                'count': len(quality_scores)
            }
        
        if demo_fidelity_scores:
            analysis['demo_fidelity_stats'] = {
                'avg': statistics.mean(demo_fidelity_scores),
                'median': statistics.median(demo_fidelity_scores),
                'min': min(demo_fidelity_scores),
                'max': max(demo_fidelity_scores),
                'count': len(demo_fidelity_scores)
            }
        
        if task_fidelity_scores:
            analysis['task_fidelity_stats'] = {
                'avg': statistics.mean(task_fidelity_scores),
                'median': statistics.median(task_fidelity_scores),
                'min': min(task_fidelity_scores),
                'max': max(task_fidelity_scores),
                'count': len(task_fidelity_scores)
            }
        
        return analysis
    
    def analyze_score_ranges(self, task_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze performance in different score ranges"""
        
        ranges = {
            'zero': {'min': 0.0, 'max': 0.0, 'name': '0.0 (Failed)'},
            'low': {'min': 0.0, 'max': 0.6, 'name': '0.0-0.6'},
            'medium': {'min': 0.6, 'max': 0.8, 'name': '0.6-0.8'},
            'high': {'min': 0.8, 'max': 0.9, 'name': '0.8-0.9'},
            'excellent': {'min': 0.9, 'max': 1.0, 'name': '0.9+'}
        }
        
        analysis = {}
        
        for range_key, range_info in ranges.items():
            if range_key == 'zero':
                tasks_in_range = [t for t in task_data if t['validation_engine_score'] == 0.0]
            else:
                tasks_in_range = [t for t in task_data if range_info['min'] <= t['validation_engine_score'] < range_info['max']]
            
            if not tasks_in_range:
                analysis[range_key] = {
                    'name': range_info['name'],
                    'count': 0,
                    'percentage': 0.0,
                    'avg_engine_score': 0.0,
                    'avg_alignment': 0.0,
                    'avg_quality': 0.0,
                    'avg_demo_fidelity': 0.0,
                    'avg_task_fidelity': 0.0,
                    'tasks': []
                }
                continue
            
            # Calculate averages for each component
            engine_scores = [t['validation_engine_score'] for t in tasks_in_range]
            alignment_scores = [t['alignment_score'] for t in tasks_in_range if t['alignment_score'] is not None]
            quality_scores = [t['quality_score'] for t in tasks_in_range if t['quality_score'] is not None]
            demo_fidelity_scores = [t['demo_fidelity_score'] for t in tasks_in_range if t['demo_fidelity_score'] is not None]
            task_fidelity_scores = [t['task_fidelity_score'] for t in tasks_in_range if t['task_fidelity_score'] is not None]
            
            analysis[range_key] = {
                'name': range_info['name'],
                'count': len(tasks_in_range),
                'percentage': (len(tasks_in_range) / len(task_data)) * 100,
                'avg_engine_score': statistics.mean(engine_scores),
                'avg_alignment': statistics.mean(alignment_scores) if alignment_scores else 0.0,
                'avg_quality': statistics.mean(quality_scores) if quality_scores else 0.0,
                'avg_demo_fidelity': statistics.mean(demo_fidelity_scores) if demo_fidelity_scores else 0.0,
                'avg_task_fidelity': statistics.mean(task_fidelity_scores) if task_fidelity_scores else 0.0,
                'tasks': [{'task_id': t['task_id'], 'prompt': t['prompt'], 'score': t['validation_engine_score']} for t in tasks_in_range]
            }
        
        return analysis
    
    def analyze_low_alignment_scores(self, task_data: List[Dict[str, Any]], threshold: float = 0.3) -> Dict[str, Any]:
        """Analyze tasks with low alignment scores"""
        
        low_alignment_tasks = [task for task in task_data if task['alignment_score'] is not None and task['alignment_score'] < threshold]
        
        if not low_alignment_tasks:
            return {'count': 0, 'threshold': threshold, 'tasks': []}
        
        analysis = {
            'count': len(low_alignment_tasks),
            'threshold': threshold,
            'percentage': (len(low_alignment_tasks) / len(task_data)) * 100,
            'tasks': []
        }
        
        for task in low_alignment_tasks:
            task_analysis = {
                'task_id': task['task_id'],
                'prompt': task['prompt'],
                'validation_engine_score': task['validation_engine_score'],
                'alignment_score': task['alignment_score'],
                'quality_score': task['quality_score'],
                'demo_fidelity_score': task['demo_fidelity_score'],
                'task_fidelity_score': task['task_fidelity_score'],
                'optimization_applied': task['optimization_applied']
            }
            analysis['tasks'].append(task_analysis)
        
        # Calculate statistics
        alignment_scores = [t['alignment_score'] for t in low_alignment_tasks]
        engine_scores = [t['validation_engine_score'] for t in low_alignment_tasks]
        
        analysis['alignment_stats'] = {
            'avg': statistics.mean(alignment_scores),
            'median': statistics.median(alignment_scores),
            'min': min(alignment_scores),
            'max': max(alignment_scores)
        }
        
        analysis['engine_score_stats'] = {
            'avg': statistics.mean(engine_scores),
            'median': statistics.median(engine_scores),
            'min': min(engine_scores),
            'max': max(engine_scores)
        }
        
        return analysis
    
    def run_detailed_analysis(self):
        """Run comprehensive detailed analysis"""
        
        print("=" * 120)
        print("TRELLIS DETAILED RELIABILITY ANALYSIS - V0 vs V1 vs V2 vs V3")
        print("=" * 120)
        
        # Parse all log files
        print(f"\n📁 Parsing log files...")
        v0_data = self.parse_log_file_detailed(self.v0_log)
        v1_data = self.parse_log_file_detailed(self.v1_log)
        v2_data = self.parse_log_file_detailed(self.v2_log)
        v3_data = self.parse_log_file_detailed(self.v3_log)
        
        print(f"   V0: {len(v0_data)} tasks")
        print(f"   V1: {len(v1_data)} tasks")
        print(f"   V2: {len(v2_data)} tasks")
        print(f"   V3: {len(v3_data)} tasks")
        
        if not v0_data and not v1_data and not v2_data and not v3_data:
            print("❌ No data found in any log file")
            return
        
        # Analyze zero-score prompts
        print(f"\n" + "=" * 120)
        print(f"ZERO-SCORE PROMPT ANALYSIS")
        print(f"=" * 120)
        
        v0_zero_analysis = self.analyze_zero_score_prompts(v0_data)
        v1_zero_analysis = self.analyze_zero_score_prompts(v1_data)
        v2_zero_analysis = self.analyze_zero_score_prompts(v2_data)
        v3_zero_analysis = self.analyze_zero_score_prompts(v3_data)
        
        print(f"\n📊 ZERO-SCORE STATISTICS:")
        print(f"{'Metric':<30} {'V0':<15} {'V1':<15} {'V2':<15} {'V3':<15} {'V1-V0':<15} {'V2-V1':<15} {'V3-V2':<15}")
        print(f"{'-'*30} {'-'*15} {'-'*15} {'-'*15} {'-'*15} {'-'*15} {'-'*15}")
        
        print(f"{'Zero Score Count':<30} {v0_zero_analysis['count']:<15} {v1_zero_analysis['count']:<15} {v2_zero_analysis['count']:<15} {v3_zero_analysis['count']:<15} {v1_zero_analysis['count'] - v0_zero_analysis['count']:+<15d} {v2_zero_analysis['count'] - v1_zero_analysis['count']:+<15d} {v3_zero_analysis['count'] - v2_zero_analysis['count']:+<15d}")
        
        v0_total = len(v0_data)
        v1_total = len(v1_data)
        v2_total = len(v2_data)
        v3_total = len(v3_data)
        v0_zero_pct = (v0_zero_analysis['count'] / v0_total) * 100
        v1_zero_pct = (v1_zero_analysis['count'] / v1_total) * 100
        v2_zero_pct = (v2_zero_analysis['count'] / v2_total) * 100
        v3_zero_pct = (v3_zero_analysis['count'] / v3_total) * 100
        print(f"{'Zero Score Percentage':<30} {v0_zero_pct:<15.1f} {v1_zero_pct:<15.1f} {v2_zero_pct:<15.1f} {v3_zero_pct:<15.1f} {v1_zero_pct - v0_zero_pct:+<15.1f} {v2_zero_pct - v1_zero_pct:+<15.1f} {v3_zero_pct - v2_zero_pct:+<15.1f}")
        
        # Show zero-score prompts for V0
        if v0_zero_analysis['tasks']:
            print(f"\n❌ V0 ZERO-SCORE PROMPTS ({len(v0_zero_analysis['tasks'])}):")
            for task in v0_zero_analysis['tasks']:
                print(f"   • {task['task_id']}: '{task['prompt']}'")
                alignment = task['alignment_score'] if task['alignment_score'] is not None else 0.0
                quality = task['quality_score'] if task['quality_score'] is not None else 0.0
                demo = task['demo_fidelity_score'] if task['demo_fidelity_score'] is not None else 0.0
                task_fid = task['task_fidelity_score'] if task['task_fidelity_score'] is not None else 0.0
                print(f"     Alignment: {alignment:.4f}, Quality: {quality:.4f}, Demo: {demo:.4f}, Task: {task_fid:.4f}")
        
        # Show zero-score prompts for V1
        if v1_zero_analysis['tasks']:
            print(f"\n❌ V1 ZERO-SCORE PROMPTS ({len(v1_zero_analysis['tasks'])}):")
            for task in v1_zero_analysis['tasks']:
                print(f"   • {task['task_id']}: '{task['prompt']}'")
                alignment = task['alignment_score'] if task['alignment_score'] is not None else 0.0
                quality = task['quality_score'] if task['quality_score'] is not None else 0.0
                demo = task['demo_fidelity_score'] if task['demo_fidelity_score'] is not None else 0.0
                task_fid = task['task_fidelity_score'] if task['task_fidelity_score'] is not None else 0.0
                print(f"     Alignment: {alignment:.4f}, Quality: {quality:.4f}, Demo: {demo:.4f}, Task: {task_fid:.4f}")
        
        # Show zero-score prompts for V2
        if v2_zero_analysis['tasks']:
            print(f"\n❌ V2 ZERO-SCORE PROMPTS ({len(v2_zero_analysis['tasks'])}):")
            for task in v2_zero_analysis['tasks']:
                print(f"   • {task['task_id']}: '{task['prompt']}'")
                alignment = task['alignment_score'] if task['alignment_score'] is not None else 0.0
                quality = task['quality_score'] if task['quality_score'] is not None else 0.0
                demo = task['demo_fidelity_score'] if task['demo_fidelity_score'] is not None else 0.0
                task_fid = task['task_fidelity_score'] if task['task_fidelity_score'] is not None else 0.0
                print(f"     Alignment: {alignment:.4f}, Quality: {quality:.4f}, Demo: {demo:.4f}, Task: {task_fid:.4f}")
        
        # Show zero-score prompts for V3
        if v3_zero_analysis['tasks']:
            print(f"\n❌ V3 ZERO-SCORE PROMPTS ({len(v3_zero_analysis['tasks'])}):")
            for task in v3_zero_analysis['tasks']:
                print(f"   • {task['task_id']}: '{task['prompt']}'")
                alignment = task['alignment_score'] if task['alignment_score'] is not None else 0.0
                quality = task['quality_score'] if task['quality_score'] is not None else 0.0
                demo = task['demo_fidelity_score'] if task['demo_fidelity_score'] is not None else 0.0
                task_fid = task['task_fidelity_score'] if task['task_fidelity_score'] is not None else 0.0
                print(f"     Alignment: {alignment:.4f}, Quality: {quality:.4f}, Demo: {demo:.4f}, Task: {task_fid:.4f}")
        
        # Analyze score ranges
        print(f"\n" + "=" * 120)
        print(f"SCORE RANGE DETAILED ANALYSIS")
        print(f"=" * 120)
        
        v2_range_analysis = self.analyze_score_ranges(v2_data)
        v3_range_analysis = self.analyze_score_ranges(v3_data)
        
        ranges = ['zero', 'low', 'medium', 'high', 'excellent']
        
        for range_key in ranges:
            v2_range = v2_range_analysis[range_key]
            v3_range = v3_range_analysis[range_key]
            
            print(f"\n📈 {v2_range['name'].upper()} RANGE:")
            print(f"{'Metric':<25} {'V2':<15} {'V3':<15} {'Change':<15}")
            print(f"{'-'*25} {'-'*15} {'-'*15} {'-'*15}")
            
            print(f"{'Count':<25} {v2_range['count']:<15} {v3_range['count']:<15} {v3_range['count'] - v2_range['count']:+<15d}")
            print(f"{'Percentage':<25} {v2_range['percentage']:<15.1f} {v3_range['percentage']:<15.1f} {v3_range['percentage'] - v2_range['percentage']:+<15.1f}")
            print(f"{'Avg Engine Score':<25} {v2_range['avg_engine_score']:<15.4f} {v3_range['avg_engine_score']:<15.4f} {v3_range['avg_engine_score'] - v2_range['avg_engine_score']:+<15.4f}")
            print(f"{'Avg Alignment':<25} {v2_range['avg_alignment']:<15.4f} {v3_range['avg_alignment']:<15.4f} {v3_range['avg_alignment'] - v2_range['avg_alignment']:+<15.4f}")
            print(f"{'Avg Quality':<25} {v2_range['avg_quality']:<15.4f} {v3_range['avg_quality']:<15.4f} {v3_range['avg_quality'] - v2_range['avg_quality']:+<15.4f}")
            print(f"{'Avg Demo Fidelity':<25} {v2_range['avg_demo_fidelity']:<15.4f} {v3_range['avg_demo_fidelity']:<15.4f} {v3_range['avg_demo_fidelity'] - v2_range['avg_demo_fidelity']:+<15.4f}")
            print(f"{'Avg Task Fidelity':<25} {v2_range['avg_task_fidelity']:<15.4f} {v3_range['avg_task_fidelity']:<15.4f} {v3_range['avg_task_fidelity'] - v2_range['avg_task_fidelity']:+<15.4f}")
        
        # Analyze low alignment scores
        print(f"\n" + "=" * 120)
        print(f"LOW ALIGNMENT SCORE ANALYSIS")
        print(f"=" * 120)
        
        v2_low_alignment = self.analyze_low_alignment_scores(v2_data, threshold=0.3)
        v3_low_alignment = self.analyze_low_alignment_scores(v3_data, threshold=0.3)
        
        print(f"\n📊 LOW ALIGNMENT STATISTICS (threshold: 0.3):")
        print(f"{'Metric':<30} {'V2':<15} {'V3':<15} {'Change':<15}")
        print(f"{'-'*30} {'-'*15} {'-'*15} {'-'*15}")
        
        print(f"{'Low Alignment Count':<30} {v2_low_alignment['count']:<15} {v3_low_alignment['count']:<15} {v3_low_alignment['count'] - v2_low_alignment['count']:+<15d}")
        print(f"{'Low Alignment Percentage':<30} {v2_low_alignment['percentage']:<15.1f} {v3_low_alignment['percentage']:<15.1f} {v3_low_alignment['percentage'] - v2_low_alignment['percentage']:+<15.1f}")
        
        if v2_low_alignment['alignment_stats']:
            print(f"{'Avg Alignment Score':<30} {v2_low_alignment['alignment_stats']['avg']:<15.4f} {v3_low_alignment['alignment_stats']['avg']:<15.4f} {v3_low_alignment['alignment_stats']['avg'] - v2_low_alignment['alignment_stats']['avg']:+<15.4f}")
            print(f"{'Avg Engine Score':<30} {v2_low_alignment['engine_score_stats']['avg']:<15.4f} {v3_low_alignment['engine_score_stats']['avg']:<15.4f} {v3_low_alignment['engine_score_stats']['avg'] - v2_low_alignment['engine_score_stats']['avg']:+<15.4f}")
        
        # Show low alignment prompts for V2
        if v2_low_alignment['tasks']:
            print(f"\n⚠️ V2 LOW ALIGNMENT PROMPTS ({len(v2_low_alignment['tasks'])}):")
            for task in v2_low_alignment['tasks'][:5]:  # Show first 5
                print(f"   • {task['task_id']}: '{task['prompt']}' (Alignment: {task['alignment_score']:.4f}, Engine: {task['validation_engine_score']:.4f})")
        
        # Show low alignment prompts for V3
        if v3_low_alignment['tasks']:
            print(f"\n⚠️ V3 LOW ALIGNMENT PROMPTS ({len(v3_low_alignment['tasks'])}):")
            for task in v3_low_alignment['tasks'][:5]:  # Show first 5
                print(f"   • {task['task_id']}: '{task['prompt']}' (Alignment: {task['alignment_score']:.4f}, Engine: {task['validation_engine_score']:.4f})")
        
        # Reliability summary
        print(f"\n" + "=" * 120)
        print(f"RELIABILITY SUMMARY")
        print(f"=" * 120)
        
        # Calculate key reliability metrics
        v2_total = len(v2_data)
        v3_total = len(v3_data)
        
        v2_non_zero = len([t for t in v2_data if t['validation_engine_score'] > 0.0])
        v3_non_zero = len([t for t in v3_data if t['validation_engine_score'] > 0.0])
        
        v2_good = len([t for t in v2_data if t['validation_engine_score'] >= 0.6])
        v3_good = len([t for t in v3_data if t['validation_engine_score'] >= 0.6])
        
        v2_excellent = len([t for t in v2_data if t['validation_engine_score'] >= 0.8])
        v3_excellent = len([t for t in v3_data if t['validation_engine_score'] >= 0.8])
        
        v2_very_excellent = len([t for t in v2_data if t['validation_engine_score'] >= 0.9])
        v3_very_excellent = len([t for t in v3_data if t['validation_engine_score'] >= 0.9])
        
        print(f"\n🏆 KEY RELIABILITY METRICS:")
        print(f"{'Metric':<30} {'V2':<15} {'V3':<15} {'Change':<15}")
        print(f"{'-'*30} {'-'*15} {'-'*15} {'-'*15}")
        
        print(f"{'Total Tasks':<30} {v2_total:<15} {v3_total:<15} {v3_total - v2_total:+<15d}")
        print(f"{'Non-Zero Success Rate':<30} {(v2_non_zero/v2_total)*100:<15.1f} {(v3_non_zero/v3_total)*100:<15.1f} {((v3_non_zero/v3_total)*100 - (v2_non_zero/v2_total)*100):+<15.1f}")
        print(f"{'Good Rate (≥0.6)':<30} {(v2_good/v2_total)*100:<15.1f} {(v3_good/v3_total)*100:<15.1f} {((v3_good/v3_total)*100 - (v2_good/v2_total)*100):+<15.1f}")
        print(f"{'Excellent Rate (≥0.8)':<30} {(v2_excellent/v2_total)*100:<15.1f} {(v3_excellent/v3_total)*100:<15.1f} {((v3_excellent/v3_total)*100 - (v2_excellent/v2_total)*100):+<15.1f}")
        print(f"{'Very Excellent (≥0.9)':<30} {(v2_very_excellent/v2_total)*100:<15.1f} {(v3_very_excellent/v3_total)*100:<15.1f} {((v3_very_excellent/v3_total)*100 - (v2_very_excellent/v2_total)*100):+<15.1f}")
        
        # Overall assessment
        print(f"\n🎯 OVERALL ASSESSMENT:")
        improvements = []
        if (v3_non_zero/v3_total)*100 > (v2_non_zero/v2_total)*100:
            improvements.append("Non-zero success rate")
        if (v3_good/v3_total)*100 > (v2_good/v2_total)*100:
            improvements.append("Good quality rate")
        if (v3_excellent/v3_total)*100 > (v2_excellent/v2_total)*100:
            improvements.append("Excellent quality rate")
        if (v3_very_excellent/v3_total)*100 > (v2_very_excellent/v2_total)*100:
            improvements.append("Very excellent quality rate")
        
        if improvements:
            print(f"   ✅ V3 shows improvements in: {', '.join(improvements)}")
        else:
            print(f"   ⚠️ V3 shows no improvements in key reliability metrics")
        
        if (v3_non_zero/v3_total)*100 > (v2_non_zero/v2_total)*100 and (v3_good/v3_total)*100 > (v2_good/v2_total)*100:
            print(f"   🏆 V3 appears to be more reliable than V2")
        elif (v3_non_zero/v3_total)*100 < (v2_non_zero/v2_total)*100 and (v3_good/v3_total)*100 < (v2_good/v2_total)*100:
            print(f"   📉 V2 appears to be more reliable than V3")
        else:
            print(f"   🤔 Mixed results - V3 and V2 have different strengths")

def main():
    """Main analysis function"""
    
    analyzer = TrellisDetailedAnalyzer()
    analyzer.run_detailed_analysis()
    
    print(f"\n" + "=" * 120)
    print(f"DETAILED ANALYSIS COMPLETE")
    print(f"=" * 120)

if __name__ == "__main__":
    main() 