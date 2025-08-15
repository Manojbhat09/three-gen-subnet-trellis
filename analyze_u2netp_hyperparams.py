#!/usr/bin/env python3
"""
Analyze u2netp hyperparameter test results and provide insights.
"""

import pandas as pd
from pathlib import Path
import sys

def analyze_hyperparams(csv_path: str):
    """Analyze the hyperparameter test results."""
    df = pd.read_csv(csv_path)
    
    print("🔍 U2NETP HYPERPARAMETER ANALYSIS INSIGHTS")
    print("=" * 60)
    
    # Overall statistics
    print(f"📊 OVERALL STATISTICS:")
    print(f"   • Total Configurations Tested: {len(df)}")
    print(f"   • Successful: {len(df[df['status'] == 'success'])}")
    print(f"   • Failed: {len(df[df['status'] == 'failed'])}")
    print(f"   • Success Rate: {len(df[df['status'] == 'success'])/len(df)*100:.1f}%")
    print()
    
    # Performance analysis
    successful = df[df['status'] == 'success'].copy()
    successful = successful.sort_values('elapsed_seconds')
    
    print(f"⚡ PERFORMANCE RANKING (Fastest to Slowest)")
    print("=" * 60)
    
    for idx, (_, row) in enumerate(successful.iterrows(), 1):
        config_details = []
        if row['alpha_matting']:
            config_details.append("α-matting")
        if row['post_process_mask']:
            config_details.append("post-process")
        if row['only_mask']:
            config_details.append("mask-only")
        if row['putalpha']:
            config_details.append("put-alpha")
        if row['bgcolor'] != "None":
            config_details.append(f"bg:{row['bgcolor']}")
        
        details_str = ", ".join(config_details) if config_details else "default"
        
        print(f"{idx:2d}. {row['config_name']:<25} {row['elapsed_seconds']:6.3f}s  ({details_str})")
    
    print()
    
    # Speed categories
    fast = successful[successful['elapsed_seconds'] < 2.0]
    medium = successful[(successful['elapsed_seconds'] >= 2.0) & (successful['elapsed_seconds'] < 10.0)]
    slow = successful[successful['elapsed_seconds'] >= 10.0]
    
    print(f"🚀 SPEED CATEGORIES:")
    print(f"   • Fast (<2s): {len(fast)} configurations")
    print(f"   • Medium (2-10s): {len(medium)} configurations")
    print(f"   • Slow (≥10s): {len(slow)} configurations")
    print()
    
    # Feature impact analysis
    print(f"🔧 FEATURE IMPACT ANALYSIS:")
    print("=" * 60)
    
    # Alpha matting impact
    alpha_matting_configs = successful[successful['alpha_matting'] == True]
    non_alpha_configs = successful[successful['alpha_matting'] == False]
    
    if len(alpha_matting_configs) > 0 and len(non_alpha_configs) > 0:
        alpha_avg = alpha_matting_configs['elapsed_seconds'].mean()
        non_alpha_avg = non_alpha_configs['elapsed_seconds'].mean()
        alpha_impact = (alpha_avg / non_alpha_avg - 1) * 100
        
        print(f"   • Alpha Matting: {alpha_avg:.3f}s avg vs {non_alpha_avg:.3f}s avg")
        print(f"     → {alpha_impact:+.1f}% performance impact")
    
    # Post-processing impact
    post_process_configs = successful[successful['post_process_mask'] == True]
    non_post_configs = successful[successful['post_process_mask'] == False]
    
    if len(post_process_configs) > 0 and len(non_post_configs) > 0:
        post_avg = post_process_configs['elapsed_seconds'].mean()
        non_post_avg = non_post_configs['elapsed_seconds'].mean()
        post_impact = (post_avg / non_post_avg - 1) * 100
        
        print(f"   • Post-Processing: {post_avg:.3f}s avg vs {non_post_avg:.3f}s avg")
        print(f"     → {post_impact:+.1f}% performance impact")
    
    # Background color impact
    bg_color_configs = successful[successful['bgcolor'] != "None"]
    transparent_configs = successful[successful['bgcolor'] == "None"]
    
    if len(bg_color_configs) > 0 and len(transparent_configs) > 0:
        bg_avg = bg_color_configs['elapsed_seconds'].mean()
        trans_avg = transparent_configs['elapsed_seconds'].mean()
        bg_impact = (bg_avg / trans_avg - 1) * 100
        
        print(f"   • Background Color: {bg_avg:.3f}s avg vs {trans_avg:.3f}s avg")
        print(f"     → {bg_impact:+.1f}% performance impact")
    
    print()
    
    # Recommendations
    print(f"💡 RECOMMENDATIONS:")
    print("=" * 60)
    
    fastest = successful.iloc[0]
    print(f"   • Fastest Option: {fastest['config_name']} ({fastest['elapsed_seconds']:.3f}s)")
    
    # Find best quality option (alpha matting + post processing)
    quality_configs = successful[
        (successful['alpha_matting'] == True) & 
        (successful['post_process_mask'] == True)
    ]
    
    if len(quality_configs) > 0:
        best_quality = quality_configs.iloc[0]
        print(f"   • Best Quality: {best_quality['config_name']} ({best_quality['elapsed_seconds']:.3f}s)")
    
    # Find best balanced option (reasonable speed + quality)
    balanced = successful[
        (successful['elapsed_seconds'] < 5.0) & 
        (successful['alpha_matting'] == False)
    ]
    
    if len(balanced) > 0:
        best_balanced = balanced.iloc[0]
        print(f"   • Best Balanced: {best_balanced['config_name']} ({best_balanced['elapsed_seconds']:.3f}s)")
    
    print()
    
    # File size analysis
    print(f"📁 OUTPUT FILE SIZE ANALYSIS:")
    print("=" * 60)
    
    # Group by configuration type
    mask_only = successful[successful['only_mask'] == True]
    with_bg = successful[successful['bgcolor'] != "None"]
    transparent = successful[
        (successful['only_mask'] == False) & 
        (successful['bgcolor'] == "None") &
        (successful['putalpha'] == False)
    ]
    
    if len(mask_only) > 0:
        print(f"   • Mask-only outputs: {len(mask_only)} configurations")
        print(f"     → Smallest: {mask_only['elapsed_seconds'].min():.3f}s")
        print(f"     → Largest: {mask_only['elapsed_seconds'].max():.3f}s")
    
    if len(with_bg) > 0:
        print(f"   • Background color outputs: {len(with_bg)} configurations")
        print(f"     → Smallest: {with_bg['elapsed_seconds'].min():.3f}s")
        print(f"     → Largest: {with_bg['elapsed_seconds'].max():.3f}s")
    
    if len(transparent) > 0:
        print(f"   • Transparent outputs: {len(transparent)} configurations")
        print(f"     → Smallest: {transparent['elapsed_seconds'].min():.3f}s")
        print(f"     → Largest: {transparent['elapsed_seconds'].max():.3f}s")
    
    print()
    
    # Performance vs quality trade-offs
    print(f"⚖️  PERFORMANCE VS QUALITY TRADE-OFFS:")
    print("=" * 60)
    
    print(f"   • Speed-focused (1-2s): {len(fast)} configs")
    print(f"     → Use for: real-time processing, batch operations")
    print(f"     → Trade-off: basic quality, no alpha matting")
    
    print(f"   • Balanced (2-10s): {len(medium)} configs")
    print(f"     → Use for: production applications, good quality")
    print(f"     → Trade-off: moderate processing time")
    
    print(f"   • Quality-focused (10s+): {len(slow)} configs")
    print(f"     → Use for: final outputs, high-quality requirements")
    print(f"     → Trade-off: significant processing time")
    
    print()
    
    # Specific use case recommendations
    print(f"🎯 USE CASE RECOMMENDATIONS:")
    print("=" * 60)
    
    print(f"   • Real-time/Interactive: {fastest['config_name']}")
    print(f"   • Web Applications: baseline, white_background, black_background")
    print(f"   • Print/Publication: high_quality_white_bg")
    print(f"   • Video Processing: fast_processing")
    print(f"   • Mask Generation: only_mask_post_processed")
    print(f"   • Transparent Overlays: transparent_background")

def main():
    csv_path = "/home/mbhat/three-gen-subnet-trellis/test_u2netp_hyperparams/summary.csv"
    
    if not Path(csv_path).exists():
        print(f"Error: CSV file not found at {csv_path}")
        sys.exit(1)
    
    analyze_hyperparams(csv_path)

if __name__ == "__main__":
    main()



