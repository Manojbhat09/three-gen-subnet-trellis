#!/usr/bin/env python3
"""
Episodic Memory Analyzer
========================
Analyzes the episodic memory JSON file and generates comprehensive graphs
showing validation score progression over time, sessions, and attempts.
"""

import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

class EpisodicMemoryAnalyzer:
    def __init__(self, memory_file="episodic_logs/episodic_memory.json"):
        self.memory_file = Path(memory_file)
        self.data = None
        self.df = None
        self.load_data()
        
    def load_data(self):
        """Load and parse the episodic memory data"""
        if not self.memory_file.exists():
            raise FileNotFoundError(f"Memory file not found: {self.memory_file}")
            
        with open(self.memory_file, 'r') as f:
            self.data = json.load(f)
            
        # Extract all attempts into a flat structure
        attempts_data = []
        
        for session in self.data.get('optimization_sessions', []):
            session_id = session['session_id']
            original_prompt = session['original_prompt']
            session_timestamp = session.get('session_duration', 0)
            
            for attempt in session.get('attempts', []):
                attempts_data.append({
                    'session_id': session_id,
                    'original_prompt': original_prompt,
                    'attempt_number': attempt['attempt_number'],
                    'strategy_used': attempt['strategy_used'],
                    'exploration_type': attempt['exploration_type'],
                    'validation_score': attempt.get('validation_score', 0.0),
                    'predicted_confidence': attempt.get('predicted_confidence', 0.0),
                    'timestamp': attempt.get('timestamp', 0),
                    'optimized_prompt': attempt.get('optimized_prompt', ''),
                    'session_duration': session_timestamp
                })
        
        self.df = pd.DataFrame(attempts_data)
        
        # Convert timestamps to datetime
        if 'timestamp' in self.df.columns:
            self.df['datetime'] = pd.to_datetime(self.df['timestamp'], unit='s')
            self.df['date'] = self.df['datetime'].dt.date
            self.df['time'] = self.df['datetime'].dt.time
            
        print(f"Loaded {len(self.df)} attempts from {len(self.data.get('optimization_sessions', []))} sessions")
        print(f"Date range: {self.df['datetime'].min()} to {self.df['datetime'].max()}")
        
    def create_score_progression_plots(self, output_dir="episodic_analysis"):
        """Create comprehensive score progression plots"""
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        # Set up the plotting style
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
        
        # 1. Overall Score Progression Over Time
        self._plot_overall_progression(output_path)
        
        # 2. Score Progression by Prompt
        self._plot_prompt_progression(output_path)
        
        # 3. Score Progression by Strategy
        self._plot_strategy_progression(output_path)
        
        # 4. Session-based Analysis
        self._plot_session_analysis(output_path)
        
        # 5. Attempt-based Analysis
        self._plot_attempt_analysis(output_path)
        
        # 6. Strategy Performance Comparison
        self._plot_strategy_comparison(output_path)
        
        # 7. Learning Trends
        self._plot_learning_trends(output_path)
        
        print(f"All plots saved to {output_path}")
        
    def _plot_overall_progression(self, output_path):
        """Plot overall score progression over time"""
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 12))
        
        # Plot 1: Score progression over time
        self.df_sorted = self.df.sort_values('datetime')
        ax1.plot(range(len(self.df_sorted)), self.df_sorted['validation_score'], 
                alpha=0.7, linewidth=1, color='blue')
        ax1.scatter(range(len(self.df_sorted)), self.df_sorted['validation_score'], 
                   alpha=0.5, s=20, color='red')
        ax1.set_title('Overall Validation Score Progression Over Time', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Attempt Number (Chronological)')
        ax1.set_ylabel('Validation Score')
        ax1.grid(True, alpha=0.3)
        
        # Add moving average
        window = min(20, len(self.df_sorted) // 10)
        if window > 1:
            moving_avg = self.df_sorted['validation_score'].rolling(window=window).mean()
            ax1.plot(range(len(self.df_sorted)), moving_avg, 
                    color='green', linewidth=2, label=f'{window}-point Moving Average')
            ax1.legend()
        
        # Plot 2: Score distribution over time
        self.df_sorted['date_group'] = self.df_sorted['datetime'].dt.date
        daily_stats = self.df_sorted.groupby('date_group')['validation_score'].agg(['mean', 'std', 'count']).reset_index()
        
        ax2.bar(range(len(daily_stats)), daily_stats['mean'], 
               yerr=daily_stats['std'], alpha=0.7, capsize=5)
        ax2.set_title('Daily Average Validation Scores', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Date')
        ax2.set_ylabel('Average Validation Score')
        ax2.set_xticks(range(len(daily_stats)))
        ax2.set_xticklabels([str(d) for d in daily_stats['date_group']], rotation=45)
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_path / 'overall_score_progression.png', dpi=300, bbox_inches='tight')
        plt.close()
        
    def _plot_prompt_progression(self, output_path):
        """Plot score progression for each prompt"""
        unique_prompts = self.df['original_prompt'].unique()
        n_prompts = len(unique_prompts)
        
        # Create subplots
        cols = 3
        rows = (n_prompts + cols - 1) // cols
        fig, axes = plt.subplots(rows, cols, figsize=(20, 5*rows))
        if rows == 1:
            axes = axes.reshape(1, -1)
        
        for idx, prompt in enumerate(unique_prompts):
            row = idx // cols
            col = idx % cols
            ax = axes[row, col]
            
            prompt_data = self.df[self.df['original_prompt'] == prompt].sort_values('datetime')
            
            if len(prompt_data) > 0:
                # Plot score progression
                ax.plot(range(len(prompt_data)), prompt_data['validation_score'], 
                       marker='o', linewidth=2, markersize=4)
                ax.set_title(f'"{prompt[:30]}..."', fontsize=10, fontweight='bold')
                ax.set_xlabel('Attempt')
                ax.set_ylabel('Score')
                ax.grid(True, alpha=0.3)
                ax.set_ylim(0, 1)
                
                # Add trend line
                if len(prompt_data) > 1:
                    z = np.polyfit(range(len(prompt_data)), prompt_data['validation_score'], 1)
                    p = np.poly1d(z)
                    ax.plot(range(len(prompt_data)), p(range(len(prompt_data))), 
                           "r--", alpha=0.8, linewidth=1)
        
        # Hide empty subplots
        for idx in range(n_prompts, rows * cols):
            row = idx // cols
            col = idx % cols
            axes[row, col].set_visible(False)
        
        plt.suptitle('Validation Score Progression by Prompt', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(output_path / 'prompt_score_progression.png', dpi=300, bbox_inches='tight')
        plt.close()
        
    def _plot_strategy_progression(self, output_path):
        """Plot score progression by strategy"""
        strategies = self.df['strategy_used'].unique()
        n_strategies = len(strategies)
        
        fig, axes = plt.subplots(2, 4, figsize=(20, 10))
        axes = axes.flatten()
        
        for idx, strategy in enumerate(strategies):
            if idx >= len(axes):
                break
                
            ax = axes[idx]
            strategy_data = self.df[self.df['strategy_used'] == strategy].sort_values('datetime')
            
            if len(strategy_data) > 0:
                # Plot score progression
                ax.plot(range(len(strategy_data)), strategy_data['validation_score'], 
                       marker='o', linewidth=2, markersize=4, alpha=0.7)
                ax.set_title(f'{strategy.replace("_", " ").title()}', fontsize=12, fontweight='bold')
                ax.set_xlabel('Attempt')
                ax.set_ylabel('Score')
                ax.grid(True, alpha=0.3)
                ax.set_ylim(0, 1)
                
                # Add statistics
                avg_score = strategy_data['validation_score'].mean()
                ax.axhline(y=avg_score, color='red', linestyle='--', alpha=0.7, 
                          label=f'Avg: {avg_score:.3f}')
                ax.legend(fontsize=8)
        
        # Hide empty subplots
        for idx in range(n_strategies, len(axes)):
            axes[idx].set_visible(False)
        
        plt.suptitle('Validation Score Progression by Strategy', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(output_path / 'strategy_score_progression.png', dpi=300, bbox_inches='tight')
        plt.close()
        
    def _plot_session_analysis(self, output_path):
        """Plot session-based analysis"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        # Session statistics
        session_stats = self.df.groupby('session_id').agg({
            'validation_score': ['mean', 'max', 'min', 'std'],
            'attempt_number': 'max',
            'strategy_used': 'nunique'
        }).round(3)
        session_stats.columns = ['avg_score', 'max_score', 'min_score', 'std_score', 'total_attempts', 'unique_strategies']
        
        # Plot 1: Session average scores
        ax1.hist(session_stats['avg_score'], bins=20, alpha=0.7, edgecolor='black')
        ax1.set_title('Distribution of Session Average Scores', fontweight='bold')
        ax1.set_xlabel('Average Score')
        ax1.set_ylabel('Number of Sessions')
        ax1.axvline(session_stats['avg_score'].mean(), color='red', linestyle='--', 
                   label=f'Mean: {session_stats["avg_score"].mean():.3f}')
        ax1.legend()
        
        # Plot 2: Session improvement
        session_improvement = []
        for session_id in self.df['session_id'].unique():
            session_data = self.df[self.df['session_id'] == session_id].sort_values('attempt_number')
            if len(session_data) > 1:
                improvement = session_data['validation_score'].iloc[-1] - session_data['validation_score'].iloc[0]
                session_improvement.append(improvement)
        
        ax2.hist(session_improvement, bins=15, alpha=0.7, edgecolor='black')
        ax2.set_title('Distribution of Session Score Improvements', fontweight='bold')
        ax2.set_xlabel('Score Improvement')
        ax2.set_ylabel('Number of Sessions')
        ax2.axvline(np.mean(session_improvement), color='red', linestyle='--', 
                   label=f'Mean: {np.mean(session_improvement):.3f}')
        ax2.legend()
        
        # Plot 3: Attempts per session
        ax3.hist(session_stats['total_attempts'], bins=range(1, int(session_stats['total_attempts'].max()) + 2), 
                alpha=0.7, edgecolor='black')
        ax3.set_title('Distribution of Attempts per Session', fontweight='bold')
        ax3.set_xlabel('Number of Attempts')
        ax3.set_ylabel('Number of Sessions')
        
        # Plot 4: Strategies per session
        ax4.hist(session_stats['unique_strategies'], bins=range(1, int(session_stats['unique_strategies'].max()) + 2), 
                alpha=0.7, edgecolor='black')
        ax4.set_title('Distribution of Unique Strategies per Session', fontweight='bold')
        ax4.set_xlabel('Number of Unique Strategies')
        ax4.set_ylabel('Number of Sessions')
        
        plt.tight_layout()
        plt.savefig(output_path / 'session_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
        
    def _plot_attempt_analysis(self, output_path):
        """Plot attempt-based analysis"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        # Plot 1: Score by attempt number
        attempt_stats = self.df.groupby('attempt_number')['validation_score'].agg(['mean', 'std', 'count']).reset_index()
        ax1.errorbar(attempt_stats['attempt_number'], attempt_stats['mean'], 
                    yerr=attempt_stats['std'], marker='o', capsize=5, linewidth=2)
        ax1.set_title('Average Score by Attempt Number', fontweight='bold')
        ax1.set_xlabel('Attempt Number')
        ax1.set_ylabel('Average Score')
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Score improvement between consecutive attempts
        improvements = []
        for session_id in self.df['session_id'].unique():
            session_data = self.df[self.df['session_id'] == session_id].sort_values('attempt_number')
            for i in range(1, len(session_data)):
                improvement = session_data['validation_score'].iloc[i] - session_data['validation_score'].iloc[i-1]
                improvements.append(improvement)
        
        ax2.hist(improvements, bins=20, alpha=0.7, edgecolor='black')
        ax2.set_title('Distribution of Score Improvements Between Attempts', fontweight='bold')
        ax2.set_xlabel('Score Improvement')
        ax2.set_ylabel('Frequency')
        ax2.axvline(np.mean(improvements), color='red', linestyle='--', 
                   label=f'Mean: {np.mean(improvements):.3f}')
        ax2.legend()
        
        # Plot 3: Exploration vs Exploitation
        exploration_stats = self.df.groupby('exploration_type')['validation_score'].agg(['mean', 'std', 'count']).reset_index()
        bars = ax3.bar(exploration_stats['exploration_type'], exploration_stats['mean'], 
                      yerr=exploration_stats['std'], capsize=5, alpha=0.7)
        ax3.set_title('Average Score by Exploration Type', fontweight='bold')
        ax3.set_xlabel('Exploration Type')
        ax3.set_ylabel('Average Score')
        
        # Add count labels on bars
        for bar, count in zip(bars, exploration_stats['count']):
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'n={count}', ha='center', va='bottom', fontsize=10)
        
        # Plot 4: Predicted vs Actual confidence
        ax4.scatter(self.df['predicted_confidence'], self.df['validation_score'], alpha=0.6, s=20)
        ax4.plot([0, 1], [0, 1], 'r--', alpha=0.8, label='Perfect Prediction')
        ax4.set_title('Predicted vs Actual Confidence', fontweight='bold')
        ax4.set_xlabel('Predicted Confidence')
        ax4.set_ylabel('Actual Validation Score')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_path / 'attempt_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
        
    def _plot_strategy_comparison(self, output_path):
        """Plot strategy performance comparison"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        strategy_stats = self.df.groupby('strategy_used').agg({
            'validation_score': ['mean', 'std', 'count'],
            'predicted_confidence': 'mean'
        }).round(3)
        strategy_stats.columns = ['avg_score', 'std_score', 'count', 'avg_confidence']
        strategy_stats = strategy_stats.sort_values('avg_score', ascending=False)
        
        # Plot 1: Strategy average scores
        bars = ax1.bar(range(len(strategy_stats)), strategy_stats['avg_score'], 
                      yerr=strategy_stats['std_score'], capsize=5, alpha=0.7)
        ax1.set_title('Strategy Performance Comparison', fontweight='bold')
        ax1.set_xlabel('Strategy')
        ax1.set_ylabel('Average Score')
        ax1.set_xticks(range(len(strategy_stats)))
        ax1.set_xticklabels([s.replace('_', ' ').title() for s in strategy_stats.index], rotation=45)
        
        # Add count labels
        for bar, count in zip(bars, strategy_stats['count']):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'n={count}', ha='center', va='bottom', fontsize=8)
        
        # Plot 2: Strategy usage frequency
        ax2.pie(strategy_stats['count'], labels=[s.replace('_', ' ').title() for s in strategy_stats.index], 
               autopct='%1.1f%%', startangle=90)
        ax2.set_title('Strategy Usage Distribution', fontweight='bold')
        
        # Plot 3: Strategy confidence vs performance
        ax3.scatter(strategy_stats['avg_confidence'], strategy_stats['avg_score'], 
                   s=strategy_stats['count']*10, alpha=0.7)
        for strategy in strategy_stats.index:
            ax3.annotate(strategy.replace('_', ' ').title(), 
                        (strategy_stats.loc[strategy, 'avg_confidence'], 
                         strategy_stats.loc[strategy, 'avg_score']),
                        xytext=(5, 5), textcoords='offset points', fontsize=8)
        ax3.set_title('Strategy Confidence vs Performance', fontweight='bold')
        ax3.set_xlabel('Average Predicted Confidence')
        ax3.set_ylabel('Average Validation Score')
        ax3.grid(True, alpha=0.3)
        
        # Plot 4: Strategy improvement over time
        for strategy in self.df['strategy_used'].unique():
            strategy_data = self.df[self.df['strategy_used'] == strategy].sort_values('datetime')
            if len(strategy_data) > 5:
                ax4.plot(range(len(strategy_data)), strategy_data['validation_score'], 
                        marker='o', markersize=3, alpha=0.7, label=strategy.replace('_', ' ').title())
        
        ax4.set_title('Strategy Performance Over Time', fontweight='bold')
        ax4.set_xlabel('Attempt Number (Chronological)')
        ax4.set_ylabel('Validation Score')
        ax4.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_path / 'strategy_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
        
    def _plot_learning_trends(self, output_path):
        """Plot learning trends and patterns"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        # Plot 1: Learning curve (cumulative average)
        self.df_sorted = self.df.sort_values('datetime')
        cumulative_avg = self.df_sorted['validation_score'].expanding().mean()
        ax1.plot(range(len(self.df_sorted)), cumulative_avg, linewidth=2, color='blue')
        ax1.set_title('Cumulative Average Score (Learning Curve)', fontweight='bold')
        ax1.set_xlabel('Attempt Number')
        ax1.set_ylabel('Cumulative Average Score')
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Success rate over time
        window_size = min(50, len(self.df_sorted) // 10)
        if window_size > 1:
            success_rate = self.df_sorted['validation_score'].rolling(window=window_size).apply(
                lambda x: (x > 0.7).mean()
            )
            ax2.plot(range(len(self.df_sorted)), success_rate, linewidth=2, color='green')
            ax2.set_title(f'Success Rate (Score > 0.7) - {window_size}-point Window', fontweight='bold')
            ax2.set_xlabel('Attempt Number')
            ax2.set_ylabel('Success Rate')
            ax2.grid(True, alpha=0.3)
        
        # Plot 3: Strategy diversity over time
        strategy_diversity = []
        for i in range(0, len(self.df_sorted), 10):
            window_data = self.df_sorted.iloc[i:i+10]
            diversity = len(window_data['strategy_used'].unique())
            strategy_diversity.append(diversity)
        
        ax3.plot(range(0, len(self.df_sorted), 10), strategy_diversity, marker='o', linewidth=2)
        ax3.set_title('Strategy Diversity Over Time (10-attempt windows)', fontweight='bold')
        ax3.set_xlabel('Attempt Number')
        ax3.set_ylabel('Number of Unique Strategies')
        ax3.grid(True, alpha=0.3)
        
        # Plot 4: Score volatility over time
        if window_size > 1:
            score_volatility = self.df_sorted['validation_score'].rolling(window=window_size).std()
            ax4.plot(range(len(self.df_sorted)), score_volatility, linewidth=2, color='red')
            ax4.set_title(f'Score Volatility - {window_size}-point Window', fontweight='bold')
            ax4.set_xlabel('Attempt Number')
            ax4.set_ylabel('Score Standard Deviation')
            ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_path / 'learning_trends.png', dpi=300, bbox_inches='tight')
        plt.close()
        
    def generate_summary_report(self, output_dir="episodic_analysis"):
        """Generate a comprehensive summary report"""
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        report = []
        report.append("EPISODIC MEMORY ANALYSIS REPORT")
        report.append("=" * 50)
        report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("")
        
        # Overall statistics
        report.append("OVERALL STATISTICS")
        report.append("-" * 20)
        report.append(f"Total attempts: {len(self.df)}")
        report.append(f"Total sessions: {self.df['session_id'].nunique()}")
        report.append(f"Unique prompts: {self.df['original_prompt'].nunique()}")
        report.append(f"Date range: {self.df['datetime'].min()} to {self.df['datetime'].max()}")
        report.append(f"Average score: {self.df['validation_score'].mean():.3f}")
        report.append(f"Score std dev: {self.df['validation_score'].std():.3f}")
        report.append(f"Best score: {self.df['validation_score'].max():.3f}")
        report.append(f"Worst score: {self.df['validation_score'].min():.3f}")
        report.append("")
        
        # Strategy performance
        report.append("STRATEGY PERFORMANCE")
        report.append("-" * 20)
        strategy_stats = self.df.groupby('strategy_used').agg({
            'validation_score': ['mean', 'std', 'count'],
            'predicted_confidence': 'mean'
        }).round(3)
        strategy_stats.columns = ['avg_score', 'std_score', 'count', 'avg_confidence']
        strategy_stats = strategy_stats.sort_values('avg_score', ascending=False)
        
        for strategy, stats in strategy_stats.iterrows():
            report.append(f"{strategy.replace('_', ' ').title()}:")
            report.append(f"  Average score: {stats['avg_score']:.3f} ± {stats['std_score']:.3f}")
            report.append(f"  Attempts: {stats['count']}")
            report.append(f"  Avg confidence: {stats['avg_confidence']:.3f}")
            report.append("")
        
        # Session analysis
        report.append("SESSION ANALYSIS")
        report.append("-" * 20)
        session_stats = self.df.groupby('session_id').agg({
            'validation_score': ['mean', 'max', 'min'],
            'attempt_number': 'max',
            'strategy_used': 'nunique'
        }).round(3)
        session_stats.columns = ['avg_score', 'max_score', 'min_score', 'total_attempts', 'unique_strategies']
        
        report.append(f"Average attempts per session: {session_stats['total_attempts'].mean():.1f}")
        report.append(f"Average strategies per session: {session_stats['unique_strategies'].mean():.1f}")
        report.append(f"Best session average: {session_stats['avg_score'].max():.3f}")
        report.append(f"Worst session average: {session_stats['avg_score'].min():.3f}")
        report.append("")
        
        # Learning insights
        report.append("LEARNING INSIGHTS")
        report.append("-" * 20)
        
        # Calculate improvement trends
        improvements = []
        for session_id in self.df['session_id'].unique():
            session_data = self.df[self.df['session_id'] == session_id].sort_values('attempt_number')
            if len(session_data) > 1:
                improvement = session_data['validation_score'].iloc[-1] - session_data['validation_score'].iloc[0]
                improvements.append(improvement)
        
        if improvements:
            positive_improvements = [imp for imp in improvements if imp > 0]
            report.append(f"Sessions with improvement: {len(positive_improvements)}/{len(improvements)} ({len(positive_improvements)/len(improvements)*100:.1f}%)")
            report.append(f"Average improvement: {np.mean(improvements):.3f}")
            report.append(f"Best improvement: {max(improvements):.3f}")
            report.append(f"Worst decline: {min(improvements):.3f}")
        
        # Strategy effectiveness
        report.append("")
        report.append("STRATEGY EFFECTIVENESS")
        report.append("-" * 20)
        for strategy in self.df['strategy_used'].unique():
            strategy_data = self.df[self.df['strategy_used'] == strategy]
            success_rate = (strategy_data['validation_score'] > 0.7).mean()
            report.append(f"{strategy.replace('_', ' ').title()}: {success_rate:.1%} success rate")
        
        # Save report
        with open(output_path / 'analysis_report.txt', 'w') as f:
            f.write('\n'.join(report))
        
        print(f"Summary report saved to {output_path / 'analysis_report.txt'}")
        return '\n'.join(report)

def main():
    """Main function to run the analysis"""
    try:
        analyzer = EpisodicMemoryAnalyzer()
        analyzer.create_score_progression_plots()
        report = analyzer.generate_summary_report()
        print("\n" + report)
        
    except Exception as e:
        print(f"Error during analysis: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 