#!/usr/bin/env python3
"""
TRELLIS Reliability Analysis
Comprehensive analysis of mining reliability statistics across multiple log files
"""

import re
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
import pandas as pd

def analyze_reliability_trends():
    """Analyze reliability trends across log files"""
    
    # Data from the provided statistics
    log_data = {
        'log.old5': {'total': 419, 'successful': 314, 'failed': 105, 'reliability': 74.94},
        'log.old4': {'total': 24, 'successful': 16, 'failed': 8, 'reliability': 66.67},
        'log.old3': {'total': 38, 'successful': 26, 'failed': 12, 'reliability': 68.42},
        'log.old2': {'total': 49, 'successful': 39, 'failed': 10, 'reliability': 79.59},
        'log.old1': {'total': 1095, 'successful': 972, 'failed': 123, 'reliability': 88.77}
    }
    
    # Convert to DataFrame for analysis
    df = pd.DataFrame.from_dict(log_data, orient='index')
    df['log_file'] = df.index
    df['failure_rate'] = 100 - df['reliability']
    
    print("=" * 80)
    print("TRELLIS MINING RELIABILITY ANALYSIS")
    print("=" * 80)
    
    # Overall Statistics
    total_tasks = df['total'].sum()
    total_successful = df['successful'].sum()
    total_failed = df['failed'].sum()
    overall_reliability = (total_successful / total_tasks) * 100
    
    print(f"\n📊 OVERALL PERFORMANCE SUMMARY")
    print(f"   Total Tasks Processed: {total_tasks:,}")
    print(f"   Total Successful: {total_successful:,} ({overall_reliability:.2f}%)")
    print(f"   Total Failed: {total_failed:,} ({100-overall_reliability:.2f}%)")
    print(f"   Overall Reliability: {overall_reliability:.2f}%")
    
    # Reliability Trends Analysis
    print(f"\n📈 RELIABILITY TRENDS ANALYSIS")
    print(f"   Best Performance: log.old1 ({df.loc['log.old1', 'reliability']:.2f}%)")
    print(f"   Worst Performance: log.old4 ({df.loc['log.old4', 'reliability']:.2f}%)")
    print(f"   Performance Range: {df['reliability'].max() - df['reliability'].min():.2f} percentage points")
    
    # Volume Analysis
    print(f"\n📦 VOLUME ANALYSIS")
    print(f"   Highest Volume: log.old1 ({df.loc['log.old1', 'total']:,} tasks)")
    print(f"   Lowest Volume: log.old4 ({df.loc['log.old4', 'total']} tasks)")
    print(f"   Average Tasks per Log: {df['total'].mean():.1f}")
    
    # Statistical Analysis
    print(f"\n🔬 STATISTICAL ANALYSIS")
    print(f"   Mean Reliability: {df['reliability'].mean():.2f}%")
    print(f"   Median Reliability: {df['reliability'].median():.2f}%")
    print(f"   Standard Deviation: {df['reliability'].std():.2f}%")
    print(f"   Coefficient of Variation: {(df['reliability'].std() / df['reliability'].mean()) * 100:.2f}%")
    
    # Performance Categories
    print(f"\n🏆 PERFORMANCE CATEGORIES")
    excellent = df[df['reliability'] >= 85]
    good = df[(df['reliability'] >= 70) & (df['reliability'] < 85)]
    poor = df[df['reliability'] < 70]
    
    print(f"   Excellent (≥85%): {len(excellent)} logs")
    print(f"   Good (70-85%): {len(good)} logs")
    print(f"   Poor (<70%): {len(poor)} logs")
    
    # Volume vs Reliability Correlation
    correlation = df['total'].corr(df['reliability'])
    print(f"\n🔗 VOLUME-RELIABILITY CORRELATION")
    print(f"   Correlation Coefficient: {correlation:.3f}")
    if correlation > 0.5:
        print(f"   Interpretation: Strong positive correlation - higher volume tends to mean higher reliability")
    elif correlation > 0.2:
        print(f"   Interpretation: Moderate positive correlation")
    elif correlation > -0.2:
        print(f"   Interpretation: Weak correlation")
    else:
        print(f"   Interpretation: Negative correlation - higher volume tends to mean lower reliability")
    
    # Failure Rate Analysis
    print(f"\n❌ FAILURE RATE ANALYSIS")
    avg_failure_rate = df['failure_rate'].mean()
    print(f"   Average Failure Rate: {avg_failure_rate:.2f}%")
    print(f"   Highest Failure Rate: {df['failure_rate'].max():.2f}% (log.old4)")
    print(f"   Lowest Failure Rate: {df['failure_rate'].min():.2f}% (log.old1)")
    
    # Recommendations
    print(f"\n💡 RECOMMENDATIONS")
    if overall_reliability >= 85:
        print(f"   ✅ Overall performance is excellent! Maintain current configuration.")
    elif overall_reliability >= 75:
        print(f"   ⚠️ Performance is good but has room for improvement.")
    else:
        print(f"   ❌ Performance needs significant improvement.")
    
    # Identify problematic logs
    problematic_logs = df[df['reliability'] < 75]
    if not problematic_logs.empty:
        print(f"   🔍 Problematic logs to investigate:")
        for log_file, row in problematic_logs.iterrows():
            print(f"      - {log_file}: {row['reliability']:.2f}% reliability ({row['total']} tasks)")
    
    # Success Rate Improvement Potential
    current_failures = total_failed
    potential_improvement = current_failures / total_tasks * 100
    print(f"\n🎯 IMPROVEMENT POTENTIAL")
    print(f"   Current failure rate: {100-overall_reliability:.2f}%")
    print(f"   Maximum potential improvement: {potential_improvement:.2f} percentage points")
    print(f"   Target reliability (90%): {overall_reliability + (90-overall_reliability):.2f}%")
    
    # Detailed Log Analysis
    print(f"\n📋 DETAILED LOG ANALYSIS")
    for log_file, data in log_data.items():
        print(f"\n   {log_file}:")
        print(f"      Tasks: {data['total']:,}")
        print(f"      Success Rate: {data['reliability']:.2f}%")
        print(f"      Success/Total: {data['successful']}/{data['total']}")
        print(f"      Failure Rate: {100-data['reliability']:.2f}%")
    
    return df

def generate_visualizations(df):
    """Generate visualizations for the reliability data"""
    
    # Create figure with subplots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('TRELLIS Mining Reliability Analysis', fontsize=16, fontweight='bold')
    
    # 1. Reliability by Log File
    ax1.bar(df.index, df['reliability'], color=['#2E8B57', '#4682B4', '#CD853F', '#DDA0DD', '#F0E68C'])
    ax1.set_title('Reliability by Log File')
    ax1.set_ylabel('Reliability (%)')
    ax1.set_ylim(0, 100)
    ax1.grid(True, alpha=0.3)
    for i, v in enumerate(df['reliability']):
        ax1.text(i, v + 1, f'{v:.1f}%', ha='center', va='bottom')
    
    # 2. Task Volume vs Reliability
    scatter = ax2.scatter(df['total'], df['reliability'], s=100, alpha=0.7, c=range(len(df)), cmap='viridis')
    ax2.set_xlabel('Number of Tasks')
    ax2.set_ylabel('Reliability (%)')
    ax2.set_title('Volume vs Reliability Correlation')
    ax2.grid(True, alpha=0.3)
    
    # Add trend line
    z = np.polyfit(df['total'], df['reliability'], 1)
    p = np.poly1d(z)
    ax2.plot(df['total'], p(df['total']), "r--", alpha=0.8)
    
    # 3. Success vs Failure Distribution
    labels = ['Successful', 'Failed']
    sizes = [df['successful'].sum(), df['failed'].sum()]
    colors = ['#2E8B57', '#DC143C']
    ax3.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
    ax3.set_title('Overall Success vs Failure Distribution')
    
    # 4. Failure Rate Trend
    ax4.plot(range(len(df)), df['failure_rate'], marker='o', linewidth=2, markersize=8, color='#DC143C')
    ax4.set_title('Failure Rate Trend Across Logs')
    ax4.set_xlabel('Log File (chronological order)')
    ax4.set_ylabel('Failure Rate (%)')
    ax4.grid(True, alpha=0.3)
    ax4.set_xticks(range(len(df)))
    ax4.set_xticklabels(df.index, rotation=45)
    
    plt.tight_layout()
    plt.savefig('trellis_reliability_analysis.png', dpi=300, bbox_inches='tight')
    print(f"\n📊 Visualizations saved to 'trellis_reliability_analysis.png'")
    
    return fig

def analyze_optimization_impact():
    """Analyze the impact of prompt optimization based on the log excerpt"""
    
    print(f"\n🔧 PROMPT OPTIMIZATION ANALYSIS")
    print(f"   Based on log excerpt analysis:")
    
    # Extract optimization data from the log excerpt
    optimization_data = {
        'reproducibility_optimizations': 2,  # From the log excerpt
        'traditional_optimizations': 0,
        'successful_optimizations': 2,
        'failed_optimizations': 0,
        'optimization_success_rate': 100.0
    }
    
    print(f"   Reproducibility Optimizations: {optimization_data['reproducibility_optimizations']}")
    print(f"   Traditional Optimizations: {optimization_data['traditional_optimizations']}")
    print(f"   Optimization Success Rate: {optimization_data['optimization_success_rate']:.1f}%")
    
    # RL Loop Analysis
    print(f"\n🤖 REINFORCEMENT LEARNING ANALYSIS")
    print(f"   RL Rounds per Optimization: 6-7 rounds")
    print(f"   Convergence Strategy: Adaptive threshold")
    print(f"   Exploration Ratio: ~66.7%")
    print(f"   Target Score Achievement: Yes (0.925+ achieved)")
    
    # Performance Metrics from RL
    print(f"\n📈 RL PERFORMANCE METRICS")
    print(f"   Best Score Achieved: 0.9347")
    print(f"   Score Improvement: 0.000 → 0.925 (+0.925)")
    print(f"   Optimization Time: ~361 seconds per prompt")
    print(f"   Strategy Effectiveness: technical_precision > material_focus")

def main():
    """Main analysis function"""
    
    # Perform reliability analysis
    df = analyze_reliability_trends()
    
    # Generate visualizations
    try:
        fig = generate_visualizations(df)
        print(f"\n✅ Analysis complete! Check the generated visualization.")
    except Exception as e:
        print(f"\n⚠️ Visualization generation failed: {e}")
        print(f"   Text analysis completed successfully.")
    
    # Analyze optimization impact
    analyze_optimization_impact()
    
    print(f"\n" + "=" * 80)
    print(f"ANALYSIS COMPLETE")
    print(f"=" * 80)

if __name__ == "__main__":
    main() 