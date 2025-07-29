#!/usr/bin/env python3
"""
TRELLIS Version Comparison Analysis
Comprehensive comparison between v2 and v3 log files with detailed statistics
"""

import re
import math
import sqlite3
import json
import glob
from collections import defaultdict
from typing import Dict, List, Any, Tuple
import statistics

class TrellisVersionAnalyzer:
    """Analyzer for comparing TRELLIS versions"""
    
    def __init__(self):
        self.v0_log = "continuous_trellis_simulator.log.v0"
        self.v1_log = "continuous_trellis_simulator.log.v1"
        self.v2_log = "continuous_trellis_simulator.log.v2"
        self.v3_log = "continuous_trellis_simulator.log.v3"
        self.v0_db = "trellis_simulation_outputs/trellis_simulator_tasks.db.v0"
        self.v1_db = "trellis_simulation_outputs/trellis_simulator_tasks.db.v1"
        self.v2_db = "trellis_simulation_outputs/trellis_simulator_tasks.db.v2"
        self.v3_db = "trellis_simulation_outputs/trellis_simulator_tasks.db.v3"
        
    def parse_log_file_enhanced(self, log_file_path: str) -> List[Dict[str, Any]]:
        """Enhanced log parsing with validation engine scores and detailed metrics"""
        
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
    
    def get_score_distribution_stats(self, task_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate detailed score distribution statistics"""
        
        if not task_data:
            return {}
        
        # Extract validation engine scores
        scores = [task['validation_engine_score'] for task in task_data if task['validation_engine_score'] is not None]
        
        if not scores:
            return {}
        
        # Calculate score thresholds
        total_tasks = len(scores)
        zero_scores = len([s for s in scores if s == 0.0])
        low_scores = len([s for s in scores if 0.0 < s < 0.6])
        medium_scores = len([s for s in scores if 0.6 <= s < 0.8])
        high_scores = len([s for s in scores if 0.8 <= s < 0.9])
        excellent_scores = len([s for s in scores if s >= 0.9])
        
        # Calculate reliability metrics
        non_zero_rate = (total_tasks - zero_scores) / total_tasks * 100
        good_rate = (medium_scores + high_scores + excellent_scores) / total_tasks * 100
        excellent_rate = excellent_scores / total_tasks * 100
        
        # Calculate average scores for each range
        non_zero_scores = [s for s in scores if s > 0.0]
        avg_non_zero = statistics.mean(non_zero_scores) if non_zero_scores else 0.0
        
        low_range_scores = [s for s in scores if 0.0 < s < 0.6]
        avg_low = statistics.mean(low_range_scores) if low_range_scores else 0.0
        
        medium_range_scores = [s for s in scores if 0.6 <= s < 0.8]
        avg_medium = statistics.mean(medium_range_scores) if medium_range_scores else 0.0
        
        high_range_scores = [s for s in scores if 0.8 <= s < 0.9]
        avg_high = statistics.mean(high_range_scores) if high_range_scores else 0.0
        
        excellent_range_scores = [s for s in scores if s >= 0.9]
        avg_excellent = statistics.mean(excellent_range_scores) if excellent_range_scores else 0.0
        
        return {
            'total_tasks': total_tasks,
            'zero_scores': zero_scores,
            'low_scores': low_scores,
            'medium_scores': medium_scores,
            'high_scores': high_scores,
            'excellent_scores': excellent_scores,
            'non_zero_rate': non_zero_rate,
            'good_rate': good_rate,
            'excellent_rate': excellent_rate,
            'avg_non_zero': avg_non_zero,
            'avg_low': avg_low,
            'avg_medium': avg_medium,
            'avg_high': avg_high,
            'avg_excellent': avg_excellent,
            'overall_avg': statistics.mean(scores),
            'overall_median': statistics.median(scores),
            'overall_std': statistics.stdev(scores) if len(scores) > 1 else 0.0,
            'min_score': min(scores),
            'max_score': max(scores)
        }
    
    def analyze_failure_patterns(self, task_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze failure patterns and causes"""
        
        if not task_data:
            return {}
        
        failures = [task for task in task_data if task['validation_engine_score'] == 0.0]
        successes = [task for task in task_data if task['validation_engine_score'] > 0.0]
        
        failure_analysis = {
            'total_failures': len(failures),
            'total_successes': len(successes),
            'failure_rate': len(failures) / len(task_data) * 100,
            'cuda_oom_failures': len([f for f in failures if f['cuda_oom']]),
            'validation_failures': len([f for f in failures if f['validation_failed']]),
            'generation_failures': len([f for f in failures if not f['generation_success']]),
            'optimization_stats': {
                'total_optimized': len([t for t in task_data if t['optimization_applied']]),
                'reproducibility_optimized': len([t for t in task_data if t['reproducibility_optimization']]),
                'traditional_optimized': len([t for t in task_data if t['traditional_optimization']]),
                'optimized_failures': len([f for f in failures if f['optimization_applied']]),
                'unoptimized_failures': len([f for f in failures if not f['optimization_applied']])
            }
        }
        
        # Analyze failure by prompt type
        failure_prompts = [f['prompt'] for f in failures]
        success_prompts = [s['prompt'] for s in successes]
        
        # Categorize prompts
        def categorize_prompt(prompt: str) -> str:
            prompt_lower = prompt.lower()
            if any(word in prompt_lower for word in ['robot', 'mechanical', 'metal']):
                return "robots/mechanical"
            elif any(word in prompt_lower for word in ['gem', 'crystal', 'stone', 'jewel']):
                return "gems/crystals"
            elif any(word in prompt_lower for word in ['statue', 'sculpture', 'carving']):
                return "statues/sculptures"
            elif any(word in prompt_lower for word in ['tool', 'instrument', 'device']):
                return "tools/instruments"
            elif any(word in prompt_lower for word in ['weapon', 'sword', 'spear', 'rifle']):
                return "weapons"
            elif any(word in prompt_lower for word in ['jewelry', 'necklace', 'bracelet', 'ring']):
                return "jewelry"
            elif any(word in prompt_lower for word in ['food', 'fruit', 'cake', 'donut']):
                return "food"
            elif any(word in prompt_lower for word in ['animal', 'creature', 'beast']):
                return "animals/creatures"
            elif any(word in prompt_lower for word in ['furniture', 'chair', 'table', 'desk']):
                return "furniture"
            else:
                return "other"
        
        failure_categories = defaultdict(int)
        success_categories = defaultdict(int)
        
        for prompt in failure_prompts:
            failure_categories[categorize_prompt(prompt)] += 1
        
        for prompt in success_prompts:
            success_categories[categorize_prompt(prompt)] += 1
        
        failure_analysis['category_failure_rates'] = {}
        for category in set(failure_categories.keys()) | set(success_categories.keys()):
            total_in_category = failure_categories[category] + success_categories[category]
            if total_in_category > 0:
                failure_rate = failure_categories[category] / total_in_category * 100
                failure_analysis['category_failure_rates'][category] = {
                    'failures': failure_categories[category],
                    'successes': success_categories[category],
                    'total': total_in_category,
                    'failure_rate': failure_rate
                }
        
        return failure_analysis
    
    def analyze_validation_components(self, task_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze individual validation component scores"""
        
        if not task_data:
            return {}
        
        # Extract component scores
        alignment_scores = [t['alignment_score'] for t in task_data if t['alignment_score'] is not None]
        quality_scores = [t['quality_score'] for t in task_data if t['quality_score'] is not None]
        demo_fidelity_scores = [t['demo_fidelity_score'] for t in task_data if t['demo_fidelity_score'] is not None]
        task_fidelity_scores = [t['task_fidelity_score'] for t in task_data if t['task_fidelity_score'] is not None]
        
        component_analysis = {}
        
        if alignment_scores:
            component_analysis['alignment'] = {
                'avg': statistics.mean(alignment_scores),
                'median': statistics.median(alignment_scores),
                'std': statistics.stdev(alignment_scores) if len(alignment_scores) > 1 else 0.0,
                'min': min(alignment_scores),
                'max': max(alignment_scores),
                'count': len(alignment_scores)
            }
        
        if quality_scores:
            component_analysis['quality'] = {
                'avg': statistics.mean(quality_scores),
                'median': statistics.median(quality_scores),
                'std': statistics.stdev(quality_scores) if len(quality_scores) > 1 else 0.0,
                'min': min(quality_scores),
                'max': max(quality_scores),
                'count': len(quality_scores)
            }
        
        if demo_fidelity_scores:
            component_analysis['demo_fidelity'] = {
                'avg': statistics.mean(demo_fidelity_scores),
                'median': statistics.median(demo_fidelity_scores),
                'std': statistics.stdev(demo_fidelity_scores) if len(demo_fidelity_scores) > 1 else 0.0,
                'min': min(demo_fidelity_scores),
                'max': max(demo_fidelity_scores),
                'count': len(demo_fidelity_scores)
            }
        
        if task_fidelity_scores:
            component_analysis['task_fidelity'] = {
                'avg': statistics.mean(task_fidelity_scores),
                'median': statistics.median(task_fidelity_scores),
                'std': statistics.stdev(task_fidelity_scores) if len(task_fidelity_scores) > 1 else 0.0,
                'min': min(task_fidelity_scores),
                'max': max(task_fidelity_scores),
                'count': len(task_fidelity_scores)
            }
        
        return component_analysis
    
    def compare_versions(self) -> Dict[str, Any]:
        """Compare v0, v1, v2, and v3 versions comprehensively"""
        
        print("=" * 100)
        print("TRELLIS VERSION COMPARISON ANALYSIS - V0 vs V1 vs V2 vs V3")
        print("=" * 100)
        
        # Parse all log files
        print(f"\n📁 Parsing log files...")
        v0_data = self.parse_log_file_enhanced(self.v0_log)
        v1_data = self.parse_log_file_enhanced(self.v1_log)
        v2_data = self.parse_log_file_enhanced(self.v2_log)
        v3_data = self.parse_log_file_enhanced(self.v3_log)
        
        print(f"   V0: {len(v0_data)} tasks")
        print(f"   V1: {len(v1_data)} tasks")
        print(f"   V2: {len(v2_data)} tasks")
        print(f"   V3: {len(v3_data)} tasks")
        
        if not v0_data and not v1_data and not v2_data and not v3_data:
            print("❌ No data found in any log file")
            return {}
        
        comparison = {
            'v0': {
                'task_data': v0_data,
                'score_stats': self.get_score_distribution_stats(v0_data),
                'failure_analysis': self.analyze_failure_patterns(v0_data),
                'component_analysis': self.analyze_validation_components(v0_data)
            },
            'v1': {
                'task_data': v1_data,
                'score_stats': self.get_score_distribution_stats(v1_data),
                'failure_analysis': self.analyze_failure_patterns(v1_data),
                'component_analysis': self.analyze_validation_components(v1_data)
            },
            'v2': {
                'task_data': v2_data,
                'score_stats': self.get_score_distribution_stats(v2_data),
                'failure_analysis': self.analyze_failure_patterns(v2_data),
                'component_analysis': self.analyze_validation_components(v2_data)
            },
            'v3': {
                'task_data': v3_data,
                'score_stats': self.get_score_distribution_stats(v3_data),
                'failure_analysis': self.analyze_failure_patterns(v3_data),
                'component_analysis': self.analyze_validation_components(v3_data)
            }
        }
        
        # Print detailed comparison
        self.print_score_comparison(comparison)
        self.print_failure_comparison(comparison)
        self.print_component_comparison(comparison)
        self.print_reliability_summary(comparison)
        
        return comparison
    
    def print_score_comparison(self, comparison: Dict[str, Any]):
        """Print detailed score comparison"""
        
        print(f"\n" + "=" * 100)
        print(f"SCORE DISTRIBUTION COMPARISON")
        print(f"=" * 100)
        
        v2_stats = comparison['v2']['score_stats']
        v3_stats = comparison['v3']['score_stats']
        
        if not v2_stats or not v3_stats:
            print("❌ Insufficient data for comparison")
            return
        
        # Overall statistics
        print(f"\n📊 OVERALL STATISTICS:")
        print(f"{'Metric':<25} {'V0':<15} {'V1':<15} {'V2':<15} {'V3':<15} {'V1-V0':<15} {'V2-V1':<15} {'V3-V2':<15}")
        print(f"{'-'*25} {'-'*15} {'-'*15} {'-'*15} {'-'*15} {'-'*15} {'-'*15}")
        
        metrics = [
            ('Total Tasks', 'total_tasks', ''),
            ('Zero Scores', 'zero_scores', ''),
            ('Non-Zero Rate (%)', 'non_zero_rate', '.1f'),
            ('Good Rate (%)', 'good_rate', '.1f'),
            ('Excellent Rate (%)', 'excellent_rate', '.1f'),
            ('Overall Average', 'overall_avg', '.4f'),
            ('Overall Median', 'overall_median', '.4f'),
            ('Overall Std Dev', 'overall_std', '.4f'),
            ('Min Score', 'min_score', '.4f'),
            ('Max Score', 'max_score', '.4f')
        ]
        
        for metric_name, stat_key, format_spec in metrics:
            v0_val = comparison['v0']['score_stats'].get(stat_key, 0)
            v1_val = comparison['v1']['score_stats'].get(stat_key, 0)
            v2_val = comparison['v2']['score_stats'].get(stat_key, 0)
            v3_val = comparison['v3']['score_stats'].get(stat_key, 0)
            
            if format_spec:
                v0_str = f"{v0_val:{format_spec}}"
                v1_str = f"{v1_val:{format_spec}}"
                v2_str = f"{v2_val:{format_spec}}"
                v3_str = f"{v3_val:{format_spec}}"
            else:
                v0_str = f"{v0_val}"
                v1_str = f"{v1_val}"
                v2_str = f"{v2_val}"
                v3_str = f"{v3_val}"
            
            v1_v0_change = v1_val - v0_val
            v2_v1_change = v2_val - v1_val
            v3_v2_change = v3_val - v2_val
            
            if format_spec:
                v1_v0_str = f"{v1_v0_change:+{format_spec}}"
                v2_v1_str = f"{v2_v1_change:+{format_spec}}"
                v3_v2_str = f"{v3_v2_change:+{format_spec}}"
            else:
                v1_v0_str = f"{v1_v0_change:+d}"
                v2_v1_str = f"{v2_v1_change:+d}"
                v3_v2_str = f"{v3_v2_change:+d}"
            
            print(f"{metric_name:<25} {v0_str:<15} {v1_str:<15} {v2_str:<15} {v3_str:<15} {v1_v0_str:<15} {v2_v1_str:<15} {v3_v2_str:<15}")
        
        # Score range breakdown
        print(f"\n📈 SCORE RANGE BREAKDOWN:")
        print(f"{'Range':<15} {'V0 Count':<12} {'V0 %':<8} {'V1 Count':<12} {'V1 %':<8} {'V2 Count':<12} {'V2 %':<8} {'V3 Count':<12} {'V3 %':<8}")
        print(f"{'-'*15} {'-'*12} {'-'*8} {'-'*12} {'-'*8} {'-'*12} {'-'*8} {'-'*12} {'-'*8}")
        
        ranges = [
            ('0.0 (Failed)', 'zero_scores'),
            ('0.0-0.6', 'low_scores'),
            ('0.6-0.8', 'medium_scores'),
            ('0.8-0.9', 'high_scores'),
            ('0.9+', 'excellent_scores')
        ]
        
        for range_name, stat_key in ranges:
            v0_count = comparison['v0']['score_stats'].get(stat_key, 0)
            v1_count = comparison['v1']['score_stats'].get(stat_key, 0)
            v2_count = comparison['v2']['score_stats'].get(stat_key, 0)
            v3_count = comparison['v3']['score_stats'].get(stat_key, 0)
            v0_total = comparison['v0']['score_stats'].get('total_tasks', 1)
            v1_total = comparison['v1']['score_stats'].get('total_tasks', 1)
            v2_total = comparison['v2']['score_stats'].get('total_tasks', 1)
            v3_total = comparison['v3']['score_stats'].get('total_tasks', 1)
            
            v0_pct = (v0_count / v0_total) * 100
            v1_pct = (v1_count / v1_total) * 100
            v2_pct = (v2_count / v2_total) * 100
            v3_pct = (v3_count / v3_total) * 100
            
            print(f"{range_name:<15} {v0_count:<12} {v0_pct:<8.1f} {v1_count:<12} {v1_pct:<8.1f} {v2_count:<12} {v2_pct:<8.1f} {v3_count:<12} {v3_pct:<8.1f}")
    
    def print_failure_comparison(self, comparison: Dict[str, Any]):
        """Print failure pattern comparison"""
        
        print(f"\n" + "=" * 100)
        print(f"FAILURE PATTERN COMPARISON")
        print(f"=" * 100)
        
        v2_failures = comparison['v2']['failure_analysis']
        v3_failures = comparison['v3']['failure_analysis']
        
        if not v2_failures or not v3_failures:
            print("❌ Insufficient failure data for comparison")
            return
        
        # Overall failure statistics
        print(f"\n❌ FAILURE STATISTICS:")
        print(f"{'Metric':<30} {'V0':<15} {'V1':<15} {'V2':<15} {'V3':<15} {'V1-V0':<15} {'V2-V1':<15} {'V3-V2':<15}")
        print(f"{'-'*30} {'-'*15} {'-'*15} {'-'*15} {'-'*15} {'-'*15} {'-'*15}")
        
        failure_metrics = [
            ('Total Failures', 'total_failures'),
            ('Failure Rate (%)', 'failure_rate'),
            ('CUDA OOM Failures', 'cuda_oom_failures'),
            ('Validation Failures', 'validation_failures'),
            ('Generation Failures', 'generation_failures')
        ]
        
        for metric_name, stat_key in failure_metrics:
            v0_val = comparison['v0']['failure_analysis'].get(stat_key, 0)
            v1_val = comparison['v1']['failure_analysis'].get(stat_key, 0)
            v2_val = comparison['v2']['failure_analysis'].get(stat_key, 0)
            v3_val = comparison['v3']['failure_analysis'].get(stat_key, 0)
            
            v1_v0_change = v1_val - v0_val
            v2_v1_change = v2_val - v1_val
            v3_v2_change = v3_val - v2_val
            
            if 'Rate' in metric_name:
                v0_str = f"{v0_val:.1f}"
                v1_str = f"{v1_val:.1f}"
                v2_str = f"{v2_val:.1f}"
                v3_str = f"{v3_val:.1f}"
                v1_v0_str = f"{v1_v0_change:+.1f}"
                v2_v1_str = f"{v2_v1_change:+.1f}"
                v3_v2_str = f"{v3_v2_change:+.1f}"
            else:
                v0_str = f"{v0_val}"
                v1_str = f"{v1_val}"
                v2_str = f"{v2_val}"
                v3_str = f"{v3_val}"
                v1_v0_str = f"{v1_v0_change:+d}"
                v2_v1_str = f"{v2_v1_change:+d}"
                v3_v2_str = f"{v3_v2_change:+d}"
            
            print(f"{metric_name:<30} {v0_str:<15} {v1_str:<15} {v2_str:<15} {v3_str:<15} {v1_v0_str:<15} {v2_v1_str:<15} {v3_v2_str:<15}")
        
        # Optimization statistics
        print(f"\n🔧 OPTIMIZATION STATISTICS:")
        v2_opt = v2_failures.get('optimization_stats', {})
        v3_opt = v3_failures.get('optimization_stats', {})
        
        opt_metrics = [
            ('Total Optimized', 'total_optimized'),
            ('Reproducibility Optimized', 'reproducibility_optimized'),
            ('Traditional Optimized', 'traditional_optimized'),
            ('Optimized Failures', 'optimized_failures'),
            ('Unoptimized Failures', 'unoptimized_failures')
        ]
        
        for metric_name, stat_key in opt_metrics:
            v2_val = v2_opt.get(stat_key, 0)
            v3_val = v3_opt.get(stat_key, 0)
            change = v3_val - v2_val
            
            print(f"{metric_name:<30} {v2_val:<15} {v3_val:<15} {change:+<15d}")
        
        # Category failure rates
        print(f"\n🎯 CATEGORY FAILURE RATES:")
        v2_categories = v2_failures.get('category_failure_rates', {})
        v3_categories = v3_failures.get('category_failure_rates', {})
        
        all_categories = set(v2_categories.keys()) | set(v3_categories.keys())
        
        for category in sorted(all_categories):
            v2_data = v2_categories.get(category, {})
            v3_data = v3_categories.get(category, {})
            
            v2_rate = v2_data.get('failure_rate', 0)
            v3_rate = v3_data.get('failure_rate', 0)
            v2_total = v2_data.get('total', 0)
            v3_total = v3_data.get('total', 0)
            change = v3_rate - v2_rate
            
            print(f"{category:<20} | V2: {v2_rate:5.1f}% ({v2_total:2d}) | V3: {v3_rate:5.1f}% ({v3_total:2d}) | Change: {change:+6.1f}%")
    
    def print_component_comparison(self, comparison: Dict[str, Any]):
        """Print validation component comparison"""
        
        print(f"\n" + "=" * 100)
        print(f"VALIDATION COMPONENT COMPARISON")
        print(f"=" * 100)
        
        v2_components = comparison['v2']['component_analysis']
        v3_components = comparison['v3']['component_analysis']
        
        if not v2_components or not v3_components:
            print("❌ Insufficient component data for comparison")
            return
        
        components = ['alignment', 'quality', 'demo_fidelity', 'task_fidelity']
        
        for component in components:
            v2_data = v2_components.get(component, {})
            v3_data = v3_components.get(component, {})
            
            if not v2_data or not v3_data:
                continue
            
            print(f"\n🔍 {component.upper().replace('_', ' ')} COMPONENT:")
            print(f"{'Metric':<15} {'V2':<15} {'V3':<15} {'Change':<15}")
            print(f"{'-'*15} {'-'*15} {'-'*15} {'-'*15}")
            
            metrics = ['avg', 'median', 'std', 'min', 'max']
            for metric in metrics:
                v2_val = v2_data.get(metric, 0)
                v3_val = v3_data.get(metric, 0)
                change = v3_val - v2_val
                
                print(f"{metric:<15} {v2_val:<15.4f} {v3_val:<15.4f} {change:+<15.4f}")
    
    def print_reliability_summary(self, comparison: Dict[str, Any]):
        """Print overall reliability summary"""
        
        print(f"\n" + "=" * 100)
        print(f"RELIABILITY SUMMARY")
        print(f"=" * 100)
        
        v2_stats = comparison['v2']['score_stats']
        v3_stats = comparison['v3']['score_stats']
        
        if not v2_stats or not v3_stats:
            print("❌ Insufficient data for reliability summary")
            return
        
        # Calculate reliability metrics
        v2_non_zero_rate = v2_stats.get('non_zero_rate', 0)
        v3_non_zero_rate = v3_stats.get('non_zero_rate', 0)
        v2_good_rate = v2_stats.get('good_rate', 0)
        v3_good_rate = v3_stats.get('good_rate', 0)
        v2_excellent_rate = v2_stats.get('excellent_rate', 0)
        v3_excellent_rate = v3_stats.get('excellent_rate', 0)
        
        print(f"\n🏆 RELIABILITY METRICS:")
        print(f"   Non-Zero Success Rate: V2: {v2_non_zero_rate:.1f}% → V3: {v3_non_zero_rate:.1f}% ({v3_non_zero_rate - v2_non_zero_rate:+.1f}%)")
        print(f"   Good Quality Rate:     V2: {v2_good_rate:.1f}% → V3: {v3_good_rate:.1f}% ({v3_good_rate - v2_good_rate:+.1f}%)")
        print(f"   Excellent Quality Rate: V2: {v2_excellent_rate:.1f}% → V3: {v3_excellent_rate:.1f}% ({v3_excellent_rate - v2_excellent_rate:+.1f}%)")
        
        # Determine winner
        improvements = []
        if v3_non_zero_rate > v2_non_zero_rate:
            improvements.append(f"Non-zero success rate (+{v3_non_zero_rate - v2_non_zero_rate:.1f}%)")
        if v3_good_rate > v2_good_rate:
            improvements.append(f"Good quality rate (+{v3_good_rate - v2_good_rate:.1f}%)")
        if v3_excellent_rate > v2_excellent_rate:
            improvements.append(f"Excellent quality rate (+{v3_excellent_rate - v2_excellent_rate:.1f}%)")
        
        if improvements:
            print(f"\n✅ V3 IMPROVEMENTS:")
            for improvement in improvements:
                print(f"   • {improvement}")
        else:
            print(f"\n⚠️ V3 shows no improvements in reliability metrics")
        
        # Overall recommendation
        print(f"\n🎯 OVERALL ASSESSMENT:")
        if v3_non_zero_rate > v2_non_zero_rate and v3_good_rate > v2_good_rate:
            print(f"   V3 appears to be more reliable than V2")
        elif v3_non_zero_rate < v2_non_zero_rate and v3_good_rate < v2_good_rate:
            print(f"   V2 appears to be more reliable than V3")
        else:
            print(f"   Mixed results - V3 and V2 have different strengths")

def main():
    """Main analysis function"""
    
    analyzer = TrellisVersionAnalyzer()
    comparison = analyzer.compare_versions()
    
    print(f"\n" + "=" * 100)
    print(f"ANALYSIS COMPLETE")
    print(f"=" * 100)

if __name__ == "__main__":
    main() 