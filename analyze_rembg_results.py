#!/usr/bin/env python3
"""
Analyze rembg background removal results and generate insights.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from tabulate import tabulate
import sys

def load_and_clean_data(csv_path: str) -> pd.DataFrame:
    """Load CSV data and clean it for analysis."""
    df = pd.read_csv(csv_path)
    
    # Clean up the output_path column (remove line breaks)
    df['output_path'] = df['output_path'].str.replace('\n', '')
    
    # Convert elapsed_seconds to numeric, handling any non-numeric values
    df['elapsed_seconds'] = pd.to_numeric(df['elapsed_seconds'], errors='coerce')
    
    # Add model category based on naming patterns
    df['model_category'] = df['model'].apply(categorize_model)
    
    return df

def categorize_model(model_name: str) -> str:
    """Categorize models based on their naming patterns."""
    if 'u2net' in model_name:
        if model_name == 'u2net':
            return 'U2Net (Standard)'
        elif model_name == 'u2netp':
            return 'U2Net (Lite)'
        elif 'human_seg' in model_name:
            return 'U2Net (Human)'
        elif 'cloth_seg' in model_name:
            return 'U2Net (Clothing)'
        else:
            return 'U2Net (Custom)'
    elif 'birefnet' in model_name:
        return 'BirefNet'
    elif 'isnet' in model_name:
        return 'ISNet'
    elif model_name == 'sam':
        return 'SAM'
    elif model_name == 'silueta':
        return 'Silueta'
    else:
        return 'Other'

def generate_insights(df: pd.DataFrame) -> dict:
    """Generate insights from the data."""
    successful_models = df[df['status'] == 'ok']
    failed_models = df[df['status'] == 'fail']
    
    insights = {
        'total_models': len(df),
        'successful_models': len(successful_models),
        'failed_models': len(failed_models),
        'success_rate': len(successful_models) / len(df) * 100,
        'fastest_model': successful_models.loc[successful_models['elapsed_seconds'].idxmin(), 'model'],
        'fastest_time': successful_models['elapsed_seconds'].min(),
        'slowest_model': successful_models.loc[successful_models['elapsed_seconds'].idxmax(), 'model'],
        'slowest_time': successful_models['elapsed_seconds'].max(),
        'avg_time': successful_models['elapsed_seconds'].mean(),
        'median_time': successful_models['elapsed_seconds'].median(),
        'std_time': successful_models['elapsed_seconds'].std(),
        'total_processing_time': successful_models['elapsed_seconds'].sum(),
        'failed_model_names': failed_models['model'].tolist() if len(failed_models) > 0 else []
    }
    
    return insights

def create_performance_table(df: pd.DataFrame) -> str:
    """Create a nicely formatted performance table."""
    successful_models = df[df['status'] == 'ok'].copy()
    successful_models = successful_models.sort_values('elapsed_seconds')
    
    # Add ranking
    successful_models['rank'] = range(1, len(successful_models) + 1)
    
    # Format the table
    table_data = []
    for _, row in successful_models.iterrows():
        table_data.append([
            row['rank'],
            row['model'],
            row['model_category'],
            f"{row['elapsed_seconds']:.3f}s",
            f"{row['elapsed_seconds']:.1f}s"
        ])
    
    headers = ["Rank", "Model", "Category", "Time (s)", "Time (s)"]
    return tabulate(table_data, headers=headers, tablefmt="grid", numalign="right")

def create_category_summary(df: pd.DataFrame) -> str:
    """Create a summary table by model category."""
    successful_models = df[df['status'] == 'ok']
    
    category_stats = successful_models.groupby('model_category').agg({
        'elapsed_seconds': ['count', 'mean', 'min', 'max', 'std']
    }).round(3)
    
    # Flatten column names
    category_stats.columns = ['count', 'mean', 'min', 'max', 'std']
    category_stats = category_stats.sort_values('mean')
    
    # Format for display
    table_data = []
    for category, stats in category_stats.iterrows():
        table_data.append([
            category,
            int(stats['count']),
            f"{stats['mean']:.3f}s",
            f"{stats['min']:.3f}s",
            f"{stats['max']:.3f}s",
            f"{stats['std']:.3f}s"
        ])
    
    headers = ["Category", "Count", "Mean", "Min", "Max", "Std Dev"]
    return tabulate(table_data, headers=headers, tablefmt="grid", numalign="right")

def print_insights(insights: dict):
    """Print insights in a nice format."""
    print("🔍 REMBG BACKGROUND REMOVAL ANALYSIS INSIGHTS")
    print("=" * 60)
    
    print(f"📊 OVERALL STATISTICS:")
    print(f"   • Total Models Tested: {insights['total_models']}")
    print(f"   • Successful: {insights['successful_models']}")
    print(f"   • Failed: {insights['failed_models']}")
    print(f"   • Success Rate: {insights['success_rate']:.1f}%")
    print()
    
    print(f"⚡ PERFORMANCE HIGHLIGHTS:")
    print(f"   • Fastest Model: {insights['fastest_model']} ({insights['fastest_time']:.3f}s)")
    print(f"   • Slowest Model: {insights['slowest_model']} ({insights['slowest_time']:.3f}s)")
    print(f"   • Average Time: {insights['avg_time']:.3f}s")
    print(f"   • Median Time: {insights['median_time']:.3f}s")
    print(f"   • Total Processing Time: {insights['total_processing_time']:.3f}s")
    print()
    
    if insights['failed_model_names']:
        print(f"❌ FAILED MODELS:")
        for model in insights['failed_model_names']:
            print(f"   • {model}")
        print()

def main():
    csv_path = "/home/mbhat/three-gen-subnet-trellis/test_rembg_images/summary.csv"
    
    if not Path(csv_path).exists():
        print(f"Error: CSV file not found at {csv_path}")
        sys.exit(1)
    
    print("Loading data...")
    df = load_and_clean_data(csv_path)
    
    print("Generating insights...")
    insights = generate_insights(df)
    
    # Print insights
    print_insights(insights)
    
    # Print performance table
    print("🏆 PERFORMANCE RANKING (Fastest to Slowest)")
    print("=" * 60)
    print(create_performance_table(df))
    print()
    
    # Print category summary
    print("📈 PERFORMANCE BY CATEGORY")
    print("=" * 60)
    print(create_category_summary(df))
    print()
    
    # Additional insights
    print("💡 KEY INSIGHTS:")
    print("=" * 60)
    
    # Speed analysis
    fast_models = df[(df['status'] == 'ok') & (df['elapsed_seconds'] < 10)]
    medium_models = df[(df['status'] == 'ok') & (df['elapsed_seconds'] >= 10) & (df['elapsed_seconds'] < 50)]
    slow_models = df[(df['status'] == 'ok') & (df['elapsed_seconds'] >= 50)]
    
    print(f"🚀 Fast Models (<10s): {len(fast_models)}")
    for _, row in fast_models.iterrows():
        print(f"   • {row['model']}: {row['elapsed_seconds']:.3f}s")
    
    print(f"\n🐌 Slow Models (≥50s): {len(slow_models)}")
    for _, row in slow_models.iterrows():
        print(f"   • {row['model']}: {row['elapsed_seconds']:.3f}s")
    
    # Model family analysis
    print(f"\n🏗️  Model Family Distribution:")
    family_counts = df[df['status'] == 'ok']['model_category'].value_counts()
    for family, count in family_counts.items():
        print(f"   • {family}: {count} models")

if __name__ == "__main__":
    main()



