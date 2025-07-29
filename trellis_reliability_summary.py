#!/usr/bin/env python3
"""
TRELLIS Reliability Summary
Key statistics for comparing v2 and v3 versions
"""

import re
from collections import defaultdict
from typing import Dict, List, Any

class TrellisReliabilitySummary:
    """Summary analyzer for key reliability metrics"""
    
    def __init__(self):
        self.v0_log = "continuous_trellis_simulator.log.v0"
        self.v1_log = "continuous_trellis_simulator.log.v1"
        self.v2_log = "continuous_trellis_simulator.log.v2"
        self.v3_log = "continuous_trellis_simulator.log.v3"
        self.v4_log = "continuous_trellis_simulator.log.v4"
        self.v5_log = "continuous_trellis_simulator.log.v5"
        self.base_log = "continuous_trellis_simulator.log"
        
    def parse_log_file(self, log_file_path: str) -> List[Dict[str, Any]]:
        """Parse log file for validation engine scores"""
        
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
                            'task_fidelity_score': None
                        }
                    
                    # Look for validation engine score
                    engine_match = re.search(r"🏆 Validation Engine Score: ([\d.]+)", line)
                    if engine_match and current_task:
                        current_task['validation_engine_score'] = float(engine_match.group(1))
                    
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
                    
                    # Look for task completion
                    completion_match = re.search(r"✅ Task ([a-zA-Z0-9_]+) finished processing", line)
                    if completion_match and current_task:
                        # If we haven't found validation engine score yet, mark as failed
                        if current_task['validation_engine_score'] is None:
                            current_task['validation_engine_score'] = 0.0
                        task_data.append(current_task)
                        current_task = None
                
                # Handle any remaining task
                if current_task:
                    if current_task['validation_engine_score'] is None:
                        current_task['validation_engine_score'] = 0.0
                    task_data.append(current_task)
        
        except Exception as e:
            print(f"Error reading {log_file_path}: {e}")
            return []
        
        return task_data
    
    def calculate_reliability_metrics(self, task_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate key reliability metrics"""
        
        if not task_data:
            return {}
        
        total_tasks = len(task_data)
        scores = [task['validation_engine_score'] for task in task_data]
        
        # Calculate score ranges
        zero_scores = len([s for s in scores if s == 0.0])
        non_zero_scores = len([s for s in scores if s > 0.0])
        good_scores = len([s for s in scores if s >= 0.6])
        excellent_scores = len([s for s in scores if s >= 0.8])
        very_excellent_scores = len([s for s in scores if s >= 0.9])
        
        # Calculate rates
        non_zero_rate = (non_zero_scores / total_tasks) * 100
        good_rate = (good_scores / total_tasks) * 100
        excellent_rate = (excellent_scores / total_tasks) * 100
        very_excellent_rate = (very_excellent_scores / total_tasks) * 100
        
        # Calculate averages
        non_zero_avg = sum([s for s in scores if s > 0.0]) / non_zero_scores if non_zero_scores > 0 else 0.0
        overall_avg = sum(scores) / total_tasks
        
        return {
            'total_tasks': total_tasks,
            'zero_scores': zero_scores,
            'non_zero_scores': non_zero_scores,
            'good_scores': good_scores,
            'excellent_scores': excellent_scores,
            'very_excellent_scores': very_excellent_scores,
            'non_zero_rate': non_zero_rate,
            'good_rate': good_rate,
            'excellent_rate': excellent_rate,
            'very_excellent_rate': very_excellent_rate,
            'non_zero_avg': non_zero_avg,
            'overall_avg': overall_avg,
            'min_score': min(scores),
            'max_score': max(scores)
        }
    
    def analyze_zero_score_causes(self, task_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze causes of zero scores"""
        
        zero_score_tasks = [task for task in task_data if task['validation_engine_score'] == 0.0]
        
        if not zero_score_tasks:
            return {'count': 0, 'causes': {}}
        
        causes = {
            'low_alignment': 0,
            'low_quality': 0,
            'low_demo_fidelity': 0,
            'low_task_fidelity': 0,
            'all_components_low': 0
        }
        
        for task in zero_score_tasks:
            alignment = task['alignment_score'] if task['alignment_score'] is not None else 0.0
            quality = task['quality_score'] if task['quality_score'] is not None else 0.0
            demo = task['demo_fidelity_score'] if task['demo_fidelity_score'] is not None else 0.0
            task_fid = task['task_fidelity_score'] if task['task_fidelity_score'] is not None else 0.0
            
            if alignment < 0.3:
                causes['low_alignment'] += 1
            if quality < 0.3:
                causes['low_quality'] += 1
            if demo < 0.3:
                causes['low_demo_fidelity'] += 1
            if task_fid < 0.3:
                causes['low_task_fidelity'] += 1
            if alignment < 0.3 and quality < 0.3 and demo < 0.3 and task_fid < 0.3:
                causes['all_components_low'] += 1
        
        return {
            'count': len(zero_score_tasks),
            'causes': causes
        }
    
    def generate_summary(self):
        """Generate comprehensive reliability summary"""
        
        print("=" * 100)
        print("TRELLIS RELIABILITY SUMMARY - BASE vs V0 vs V1 vs V2 vs V3 vs V4 vs V5")
        print("=" * 100)
        
        # Parse all log files
        print(f"\n📁 Parsing log files...")
        base_data = self.parse_log_file(self.base_log)
        v0_data = self.parse_log_file(self.v0_log)
        v1_data = self.parse_log_file(self.v1_log)
        v2_data = self.parse_log_file(self.v2_log)
        v3_data = self.parse_log_file(self.v3_log)
        v4_data = self.parse_log_file(self.v4_log)
        v5_data = self.parse_log_file(self.v5_log)
        
        print(f"   BASE: {len(base_data)} tasks (NO OPTIMIZATION)")
        print(f"   V0: {len(v0_data)} tasks")
        print(f"   V1: {len(v1_data)} tasks")
        print(f"   V2: {len(v2_data)} tasks")
        print(f"   V3: {len(v3_data)} tasks")
        print(f"   V4: {len(v4_data)} tasks")
        print(f"   V5: {len(v5_data)} tasks")
        
        if not base_data and not v0_data and not v1_data and not v2_data and not v3_data and not v4_data and not v5_data:
            print("❌ No data found in any log file")
            return
        
        # Calculate metrics
        base_metrics = self.calculate_reliability_metrics(base_data)
        v0_metrics = self.calculate_reliability_metrics(v0_data)
        v1_metrics = self.calculate_reliability_metrics(v1_data)
        v2_metrics = self.calculate_reliability_metrics(v2_data)
        v3_metrics = self.calculate_reliability_metrics(v3_data)
        v4_metrics = self.calculate_reliability_metrics(v4_data)
        v5_metrics = self.calculate_reliability_metrics(v5_data)
        
        # Analyze zero score causes
        base_zero_analysis = self.analyze_zero_score_causes(base_data)
        v0_zero_analysis = self.analyze_zero_score_causes(v0_data)
        v1_zero_analysis = self.analyze_zero_score_causes(v1_data)
        v2_zero_analysis = self.analyze_zero_score_causes(v2_data)
        v3_zero_analysis = self.analyze_zero_score_causes(v3_data)
        v4_zero_analysis = self.analyze_zero_score_causes(v4_data)
        v5_zero_analysis = self.analyze_zero_score_causes(v5_data)
        
        # Print key statistics
        print(f"\n" + "=" * 100)
        print(f"KEY RELIABILITY STATISTICS")
        print(f"=" * 100)
        
        print(f"\n📊 OVERALL PERFORMANCE:")
        print(f"{'Metric':<35} {'BASE':<15} {'V0':<15} {'V1':<15} {'V2':<15} {'V3':<15} {'V4':<15} {'V5':<15} {'V0-BASE':<15} {'V1-V0':<15} {'V2-V1':<15} {'V3-V2':<15} {'V4-V3':<15} {'V5-V4':<15}")
        print(f"{'-'*35} {'-'*15} {'-'*15} {'-'*15} {'-'*15} {'-'*15} {'-'*15} {'-'*15} {'-'*15} {'-'*15}")
        
        metrics = [
            ('Total Tasks', 'total_tasks', ''),
            ('Zero Scores (Failures)', 'zero_scores', ''),
            ('Non-Zero Success Rate (%)', 'non_zero_rate', '.1f'),
            ('Good Rate ≥0.6 (%)', 'good_rate', '.1f'),
            ('Excellent Rate ≥0.8 (%)', 'excellent_rate', '.1f'),
            ('Very Excellent ≥0.9 (%)', 'very_excellent_rate', '.1f'),
            ('Overall Average Score', 'overall_avg', '.4f'),
            ('Non-Zero Average Score', 'non_zero_avg', '.4f'),
            ('Min Score', 'min_score', '.4f'),
            ('Max Score', 'max_score', '.4f')
        ]
        
        for metric_name, stat_key, format_spec in metrics:
            base_val = base_metrics.get(stat_key, 0)
            v0_val = v0_metrics.get(stat_key, 0)
            v1_val = v1_metrics.get(stat_key, 0)
            v2_val = v2_metrics.get(stat_key, 0)
            v3_val = v3_metrics.get(stat_key, 0)
            v4_val = v4_metrics.get(stat_key, 0)
            v5_val = v5_metrics.get(stat_key, 0)
            
            if format_spec:
                base_str = f"{base_val:{format_spec}}"
                v0_str = f"{v0_val:{format_spec}}"
                v1_str = f"{v1_val:{format_spec}}"
                v2_str = f"{v2_val:{format_spec}}"
                v3_str = f"{v3_val:{format_spec}}"
                v4_str = f"{v4_val:{format_spec}}"
                v5_str = f"{v5_val:{format_spec}}"
            else:
                base_str = f"{base_val}"
                v0_str = f"{v0_val}"
                v1_str = f"{v1_val}"
                v2_str = f"{v2_val}"
                v3_str = f"{v3_val}"
                v4_str = f"{v4_val}"
                v5_str = f"{v5_val}"
            
            v0_base_change = v0_val - base_val
            v1_v0_change = v1_val - v0_val
            v2_v1_change = v2_val - v1_val
            v3_v2_change = v3_val - v2_val
            v4_v3_change = v4_val - v3_val
            v5_v4_change = v5_val - v4_val
            
            if format_spec:
                v0_base_str = f"{v0_base_change:+{format_spec}}"
                v1_v0_str = f"{v1_v0_change:+{format_spec}}"
                v2_v1_str = f"{v2_v1_change:+{format_spec}}"
                v3_v2_str = f"{v3_v2_change:+{format_spec}}"
                v4_v3_str = f"{v4_v3_change:+{format_spec}}"
                v5_v4_str = f"{v5_v4_change:+{format_spec}}"
            else:
                v0_base_str = f"{v0_base_change:+d}"
                v1_v0_str = f"{v1_v0_change:+d}"
                v2_v1_str = f"{v2_v1_change:+d}"
                v3_v2_str = f"{v3_v2_change:+d}"
                v4_v3_str = f"{v4_v3_change:+d}"
                v5_v4_str = f"{v5_v4_change:+d}"
            
            print(f"{metric_name:<35} {base_str:<15} {v0_str:<15} {v1_str:<15} {v2_str:<15} {v3_str:<15} {v4_str:<15} {v5_str:<15} {v0_base_str:<15} {v1_v0_str:<15} {v2_v1_str:<15} {v3_v2_str:<15} {v4_v3_str:<15} {v5_v4_str:<15}")
        
        # Zero score analysis
        print(f"\n" + "=" * 100)
        print(f"ZERO-SCORE FAILURE ANALYSIS")
        print(f"=" * 100)
        
        print(f"\n❌ ZERO-SCORE STATISTICS:")
        print(f"{'Metric':<35} {'V0':<15} {'V1':<15} {'V2':<15} {'V3':<15} {'V4':<15} {'V1-V0':<15} {'V2-V1':<15} {'V3-V2':<15} {'V4-V3':<15}")
        print(f"{'-'*35} {'-'*15} {'-'*15} {'-'*15} {'-'*15} {'-'*15} {'-'*15} {'-'*15}")
        
        print(f"{'Zero Score Count':<35} {v0_zero_analysis['count']:<15} {v1_zero_analysis['count']:<15} {v2_zero_analysis['count']:<15} {v3_zero_analysis['count']:<15} {v4_zero_analysis['count']:<15} {v1_zero_analysis['count'] - v0_zero_analysis['count']:+<15d} {v2_zero_analysis['count'] - v1_zero_analysis['count']:+<15d} {v3_zero_analysis['count'] - v2_zero_analysis['count']:+<15d} {v4_zero_analysis['count'] - v3_zero_analysis['count']:+<15d}")
        
        v0_total = v0_metrics.get('total_tasks', 1)
        v1_total = v1_metrics.get('total_tasks', 1)
        v2_total = v2_metrics.get('total_tasks', 1)
        v3_total = v3_metrics.get('total_tasks', 1)
        v4_total = v4_metrics.get('total_tasks', 1)
        v0_zero_pct = (v0_zero_analysis['count'] / v0_total) * 100
        v1_zero_pct = (v1_zero_analysis['count'] / v1_total) * 100
        v2_zero_pct = (v2_zero_analysis['count'] / v2_total) * 100
        v3_zero_pct = (v3_zero_analysis['count'] / v3_total) * 100
        v4_zero_pct = (v4_zero_analysis['count'] / v4_total) * 100
        print(f"{'Zero Score Percentage':<35} {v0_zero_pct:<15.1f} {v1_zero_pct:<15.1f} {v2_zero_pct:<15.1f} {v3_zero_pct:<15.1f} {v4_zero_pct:<15.1f} {v1_zero_pct - v0_zero_pct:+<15.1f} {v2_zero_pct - v1_zero_pct:+<15.1f} {v3_zero_pct - v2_zero_pct:+<15.1f} {v4_zero_pct - v3_zero_pct:+<15.1f}")
        
        # Zero score causes
        print(f"\n🔍 ZERO-SCORE CAUSES:")
        v0_causes = v0_zero_analysis.get('causes', {})
        v1_causes = v1_zero_analysis.get('causes', {})
        v2_causes = v2_zero_analysis.get('causes', {})
        v3_causes = v3_zero_analysis.get('causes', {})
        
        cause_names = {
            'low_alignment': 'Low Alignment Score (<0.3)',
            'low_quality': 'Low Quality Score (<0.3)',
            'low_demo_fidelity': 'Low Demo Fidelity (<0.3)',
            'low_task_fidelity': 'Low Task Fidelity (<0.3)',
            'all_components_low': 'All Components Low (<0.3)'
        }
        
        for cause_key, cause_name in cause_names.items():
            v0_count = v0_causes.get(cause_key, 0)
            v1_count = v1_causes.get(cause_key, 0)
            v2_count = v2_causes.get(cause_key, 0)
            v3_count = v3_causes.get(cause_key, 0)
            v1_v0_change = v1_count - v0_count
            v2_v1_change = v2_count - v1_count
            v3_v2_change = v3_count - v2_count
            print(f"{cause_name:<35} {v0_count:<15} {v1_count:<15} {v2_count:<15} {v3_count:<15} {v1_v0_change:+<15d} {v2_v1_change:+<15d} {v3_v2_change:+<15d}")
        
        # Reliability assessment
        print(f"\n" + "=" * 100)
        print(f"RELIABILITY ASSESSMENT")
        print(f"=" * 100)
        
        # Calculate improvement metrics
        v0_non_zero_rate = v0_metrics.get('non_zero_rate', 0)
        v1_non_zero_rate = v1_metrics.get('non_zero_rate', 0)
        v2_non_zero_rate = v2_metrics.get('non_zero_rate', 0)
        v3_non_zero_rate = v3_metrics.get('non_zero_rate', 0)
        v0_good_rate = v0_metrics.get('good_rate', 0)
        v1_good_rate = v1_metrics.get('good_rate', 0)
        v2_good_rate = v2_metrics.get('good_rate', 0)
        v3_good_rate = v3_metrics.get('good_rate', 0)
        v0_excellent_rate = v0_metrics.get('excellent_rate', 0)
        v1_excellent_rate = v1_metrics.get('excellent_rate', 0)
        v2_excellent_rate = v2_metrics.get('excellent_rate', 0)
        v3_excellent_rate = v3_metrics.get('excellent_rate', 0)
        
        improvements = []
        if v3_non_zero_rate > v2_non_zero_rate:
            improvements.append(f"Non-zero success rate (+{v3_non_zero_rate - v2_non_zero_rate:.1f}%)")
        if v3_good_rate > v2_good_rate:
            improvements.append(f"Good quality rate (+{v3_good_rate - v2_good_rate:.1f}%)")
        if v3_excellent_rate > v2_excellent_rate:
            improvements.append(f"Excellent quality rate (+{v3_excellent_rate - v2_excellent_rate:.1f}%)")
        
        print(f"\n🏆 RELIABILITY METRICS:")
        print(f"   Non-Zero Success Rate: V0: {v0_non_zero_rate:.1f}% → V1: {v1_non_zero_rate:.1f}% → V2: {v2_non_zero_rate:.1f}% → V3: {v3_non_zero_rate:.1f}%")
        print(f"   Good Quality Rate:     V0: {v0_good_rate:.1f}% → V1: {v1_good_rate:.1f}% → V2: {v2_good_rate:.1f}% → V3: {v3_good_rate:.1f}%")
        print(f"   Excellent Quality Rate: V0: {v0_excellent_rate:.1f}% → V1: {v1_excellent_rate:.1f}% → V2: {v2_excellent_rate:.1f}% → V3: {v3_excellent_rate:.1f}%")
        
        if improvements:
            print(f"\n✅ V3 IMPROVEMENTS:")
            for improvement in improvements:
                print(f"   • {improvement}")
        else:
            print(f"\n⚠️ V3 shows no improvements in reliability metrics")
        
        # Overall recommendation
        print(f"\n🎯 OVERALL ASSESSMENT:")
        
        # Compare all four versions
        best_version = "V0"
        best_non_zero = v0_non_zero_rate
        best_good_rate = v0_good_rate
        
        if v1_non_zero_rate > best_non_zero:
            best_version = "V1"
            best_non_zero = v1_non_zero_rate
        if v2_non_zero_rate > best_non_zero:
            best_version = "V2"
            best_non_zero = v2_non_zero_rate
        if v3_non_zero_rate > best_non_zero:
            best_version = "V3"
            best_non_zero = v3_non_zero_rate
            
        if v1_good_rate > best_good_rate:
            best_good_rate = v1_good_rate
        if v2_good_rate > best_good_rate:
            best_good_rate = v2_good_rate
        if v3_good_rate > best_good_rate:
            best_good_rate = v3_good_rate
        
        print(f"   🏆 {best_version} appears to be the MOST RELIABLE version")
        print(f"   📈 Best non-zero success rate: {best_non_zero:.1f}%")
        print(f"   📈 Best good quality rate: {best_good_rate:.1f}%")
        
        # Version progression analysis
        print(f"\n📊 VERSION PROGRESSION:")
        if v1_non_zero_rate > v0_non_zero_rate and v2_non_zero_rate > v1_non_zero_rate and v3_non_zero_rate > v2_non_zero_rate:
            print(f"   ✅ Steady improvement: V0 → V1 → V2 → V3")
        elif v1_non_zero_rate > v0_non_zero_rate and v2_non_zero_rate < v1_non_zero_rate and v3_non_zero_rate > v2_non_zero_rate:
            print(f"   📈 V1 peak: V0 → V1 (improvement) → V2 (regression) → V3 (recovery)")
        elif v1_non_zero_rate > v0_non_zero_rate and v2_non_zero_rate < v1_non_zero_rate and v3_non_zero_rate < v2_non_zero_rate:
            print(f"   📉 V1 peak, then decline: V0 → V1 (improvement) → V2 (regression) → V3 (further decline)")
        elif v1_non_zero_rate < v0_non_zero_rate and v2_non_zero_rate < v1_non_zero_rate and v3_non_zero_rate > v2_non_zero_rate:
            print(f"   📉 V0-V2 decline, V3 recovery: V0 → V1 (regression) → V2 (further regression) → V3 (recovery)")
        else:
            print(f"   🤔 Mixed progression pattern")
        
        # Key takeaways
        print(f"\n📋 KEY TAKEAWAYS:")
        print(f"   • V0: {v0_metrics.get('total_tasks', 0)} tasks, V1: {v1_metrics.get('total_tasks', 0)} tasks, V2: {v2_metrics.get('total_tasks', 0)} tasks, V3: {v3_metrics.get('total_tasks', 0)} tasks")
        print(f"   • V0→V1: {v1_metrics.get('non_zero_rate', 0) - v0_metrics.get('non_zero_rate', 0):+.1f}% non-zero rate change")
        print(f"   • V1→V2: {v2_metrics.get('non_zero_rate', 0) - v1_metrics.get('non_zero_rate', 0):+.1f}% non-zero rate change")
        print(f"   • V2→V3: {v3_metrics.get('non_zero_rate', 0) - v2_metrics.get('non_zero_rate', 0):+.1f}% non-zero rate change")
        print(f"   • Zero-score failures: V0: {v0_zero_analysis['count']}, V1: {v1_zero_analysis['count']}, V2: {v2_zero_analysis['count']}, V3: {v3_zero_analysis['count']}")
        print(f"   • Overall trend: {'Improving' if v3_non_zero_rate > v0_non_zero_rate else 'Declining' if v3_non_zero_rate < v0_non_zero_rate else 'Stable'}")

def main():
    """Main function"""
    
    analyzer = TrellisReliabilitySummary()
    analyzer.generate_summary()
    
    print(f"\n" + "=" * 100)
    print(f"SUMMARY COMPLETE")
    print(f"=" * 100)

if __name__ == "__main__":
    main() 