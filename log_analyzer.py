#!/usr/bin/env python3
"""
Log Analyzer for Trellis Simulator Comparison
Parses multiple log files and creates comprehensive comparison table
"""

import re
import pandas as pd
import argparse
import sys
from datetime import datetime

def parse_log_file(file_path):
    """Parse log file and extract prompt data with scores and timing"""
    data = []
    
    with open(file_path, 'r') as f:
        content = f.read()
    
    # Split content into sections for each prompt
    # Look for the pattern that indicates a new prompt generation
    sections = re.split(r'    Original: \'', content)
    
    # Dictionary to store the latest scores for each unique prompt
    prompt_scores = {}
    
    for i, section in enumerate(sections[1:], 1):  # Skip first empty section
        prompt_data = {}
        
        # Extract original prompt (it's at the start of the section)
        original_match = re.search(r"^([^']+)'", section)
        if original_match:
            original_prompt = original_match.group(1)
            prompt_data['original_prompt'] = original_prompt
        
        # Extract optimized prompt (if exists)
        optimized_match = re.search(r"Optimized: '([^']+)'", section)
        if optimized_match:
            prompt_data['optimized_prompt'] = optimized_match.group(1)
        
        # Extract scores
        validation_match = re.search(r'🏆 Validation Engine Score: ([\d.]+)', section)
        if validation_match:
            prompt_data['validation_engine_score'] = float(validation_match.group(1))
        
        alignment_match = re.search(r'🤝 Alignment Score: ([\d.]+)', section)
        if alignment_match:
            prompt_data['alignment_score'] = float(alignment_match.group(1))
        
        quality_match = re.search(r'💎 Quality Score: ([\d.]+)', section)
        if quality_match:
            prompt_data['quality_score'] = float(quality_match.group(1))
        
        # Extract timing information
        timing_data = extract_timing_info(section)
        prompt_data.update(timing_data)
        
        # Extract timestamp
        timestamp_match = re.search(r'(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})', section)
        if timestamp_match:
            prompt_data['timestamp'] = timestamp_match.group(1)
        
        # Only process if we have all required data
        if all(key in prompt_data for key in ['original_prompt', 'validation_engine_score', 'alignment_score', 'quality_score']):
            # Store/update the latest scores for this prompt
            prompt_scores[original_prompt] = prompt_data
    
    # Convert dictionary to list
    data = list(prompt_scores.values())
    
    return data

def extract_timing_info(section):
    """Extract timing information from a log section"""
    timing_data = {}
    
    # Look for generation time patterns
    gen_time_match = re.search(r'Generation time: ([\d.]+)s', section)
    if gen_time_match:
        timing_data['generation_time'] = float(gen_time_match.group(1))
    
    # Look for validation time patterns
    val_time_match = re.search(r'Validation time: ([\d.]+)s', section)
    if val_time_match:
        timing_data['validation_time'] = float(val_time_match.group(1))
    
    # Look for total time patterns
    total_time_match = re.search(r'Total time: ([\d.]+)s', section)
    if total_time_match:
        timing_data['total_time'] = float(total_time_match.group(1))
    
    # Look for "took" patterns (e.g., "took 2.34s")
    took_match = re.search(r'took ([\d.]+)s', section)
    if took_match:
        timing_data['operation_time'] = float(took_match.group(1))
    
    # Look for timestamp differences to calculate durations
    timestamps = re.findall(r'(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3})', section)
    if len(timestamps) >= 2:
        try:
            start_time = datetime.strptime(timestamps[0], '%Y-%m-%d %H:%M:%S,%f')
            end_time = datetime.strptime(timestamps[-1], '%Y-%m-%d %H:%M:%S,%f')
            duration = (end_time - start_time).total_seconds()
            timing_data['section_duration'] = duration
        except:
            pass
    
    return timing_data

def calculate_validation_hits(scores, threshold=0.6):
    """Calculate validation hit statistics"""
    hits = [score > threshold for score in scores]
    return sum(hits), len(hits), sum(hits) / len(hits) * 100

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Compare multiple Trellis simulator log files')
    parser.add_argument('log_files', nargs='+', help='Paths to log files to compare')
    parser.add_argument('--names', nargs='*', help='Custom names for each log file (optional)')
    parser.add_argument('--output', '-o', help='Output CSV file path (default: log_comparison_table.csv)')
    parser.add_argument('--threshold', '-t', type=float, default=0.6, help='Validation hit threshold (default: 0.6)')
    
    args = parser.parse_args()
    
    if len(args.names) > 0 and len(args.names) != len(args.log_files):
        print("Error: Number of names must match number of log files")
        sys.exit(1)
    
    # Generate names if not provided
    if not args.names:
        log_names = [f"Log_{i+1}" for i in range(len(args.log_files))]
    else:
        log_names = args.names
    
    print("Parsing log files...")
    
    # Parse all log files
    all_log_data = []
    for i, log_path in enumerate(args.log_files):
        try:
            log_data = parse_log_file(log_path)
            all_log_data.append(log_data)
            print(f"{log_names[i]}: {len(log_data)} prompts found")
        except FileNotFoundError:
            print(f"Error: File not found: {log_path}")
            sys.exit(1)
        except Exception as e:
            print(f"Error parsing {log_path}: {e}")
            sys.exit(1)
    
    # Create comparison dataframe
    comparison_data = []
    
    # Get the minimum length to avoid index errors
    min_length = min(len(log_data) for log_data in all_log_data)
    
    for i in range(min_length):
        row = {'Prompt_Number': i + 1}
        
        # Add data for each log
        for j, (log_data, log_name) in enumerate(zip(all_log_data, log_names)):
            row[f'{log_name}_Original_Prompt'] = log_data[i]['original_prompt']
            row[f'{log_name}_Validation_Score'] = log_data[i]['validation_engine_score']
            row[f'{log_name}_Alignment_Score'] = log_data[i]['alignment_score']
            row[f'{log_name}_Quality_Score'] = log_data[i]['quality_score']
            row[f'{log_name}_Validation_Hit'] = log_data[i]['validation_engine_score'] > args.threshold
            
            # Add timing data if available
            if 'generation_time' in log_data[i]:
                row[f'{log_name}_Generation_Time'] = log_data[i]['generation_time']
            if 'validation_time' in log_data[i]:
                row[f'{log_name}_Validation_Time'] = log_data[i]['validation_time']
            if 'total_time' in log_data[i]:
                row[f'{log_name}_Total_Time'] = log_data[i]['total_time']
            if 'section_duration' in log_data[i]:
                row[f'{log_name}_Section_Duration'] = log_data[i]['section_duration']
        
        # Use the first log as reference for prompt text
        row['Original_Prompt'] = all_log_data[0][i]['original_prompt']
        
        # Calculate improvements relative to first log
        if len(all_log_data) > 1:
            for j in range(1, len(all_log_data)):
                log_name = log_names[j]
                ref_validation = all_log_data[0][i]['validation_engine_score']
                ref_alignment = all_log_data[0][i]['alignment_score']
                ref_quality = all_log_data[0][i]['quality_score']
                
                curr_validation = all_log_data[j][i]['validation_engine_score']
                curr_alignment = all_log_data[j][i]['alignment_score']
                curr_quality = all_log_data[j][i]['quality_score']
                
                row[f'{log_name}_Validation_Improvement'] = ((curr_validation - ref_validation) / ref_validation * 100) if ref_validation > 0 else 0
                row[f'{log_name}_Alignment_Improvement'] = ((curr_alignment - ref_alignment) / ref_alignment * 100) if ref_alignment > 0 else 0
                row[f'{log_name}_Quality_Improvement'] = ((curr_quality - ref_quality) / ref_quality * 100) if ref_quality > 0 else 0
        
        comparison_data.append(row)
    
    # Create DataFrame
    df = pd.DataFrame(comparison_data)
    
    # Calculate statistics
    print("\n" + "="*80)
    print("COMPREHENSIVE LOG COMPARISON ANALYSIS")
    print("="*80)
    
    # Validation Hit Statistics
    print(f"\nVALIDATION HIT STATISTICS (Score > {args.threshold}):")
    hit_stats = []
    for log_name in log_names:
        validation_scores = df[f'{log_name}_Validation_Score']
        hits, total, percentage = calculate_validation_hits(validation_scores, args.threshold)
        hit_stats.append((log_name, hits, total, percentage))
        print(f"{log_name}: {hits}/{total} hits ({percentage:.1f}%)")
    
    # Average Scores
    print(f"\nAVERAGE SCORES:")
    for log_name in log_names:
        validation_avg = df[f'{log_name}_Validation_Score'].mean()
        alignment_avg = df[f'{log_name}_Alignment_Score'].mean()
        quality_avg = df[f'{log_name}_Quality_Score'].mean()
        print(f"{log_name} - Validation: {validation_avg:.4f}, Alignment: {alignment_avg:.4f}, Quality: {quality_avg:.4f}")
    
    # Average Timing
    print(f"\nAVERAGE TIMING:")
    for log_name in log_names:
        timing_info = []
        if f'{log_name}_Generation_Time' in df.columns:
            gen_avg = df[f'{log_name}_Generation_Time'].mean()
            timing_info.append(f"Generation: {gen_avg:.2f}s")
        if f'{log_name}_Validation_Time' in df.columns:
            val_avg = df[f'{log_name}_Validation_Time'].mean()
            timing_info.append(f"Validation: {val_avg:.2f}s")
        if f'{log_name}_Total_Time' in df.columns:
            total_avg = df[f'{log_name}_Total_Time'].mean()
            timing_info.append(f"Total: {total_avg:.2f}s")
        if f'{log_name}_Section_Duration' in df.columns:
            section_avg = df[f'{log_name}_Section_Duration'].mean()
            timing_info.append(f"Section: {section_avg:.2f}s")
        
        if timing_info:
            print(f"{log_name} - {', '.join(timing_info)}")
        else:
            print(f"{log_name} - No timing data available")
    
    # Average Improvements (relative to first log)
    if len(all_log_data) > 1:
        print(f"\nAVERAGE IMPROVEMENTS (relative to {log_names[0]}):")
        for j in range(1, len(all_log_data)):
            log_name = log_names[j]
            validation_improvement = df[f'{log_name}_Validation_Improvement'].mean()
            alignment_improvement = df[f'{log_name}_Alignment_Improvement'].mean()
            quality_improvement = df[f'{log_name}_Quality_Improvement'].mean()
            print(f"{log_name} - Validation: {validation_improvement:+.1f}%, Alignment: {alignment_improvement:+.1f}%, Quality: {quality_improvement:+.1f}%")
    
    # Best and Worst Performers (relative to first log)
    if len(all_log_data) > 1:
        for j in range(1, len(all_log_data)):
            log_name = log_names[j]
            print(f"\n{log_name} BEST/WORST IMPROVEMENTS:")
            best_validation_idx = df[f'{log_name}_Validation_Improvement'].idxmax()
            best_quality_idx = df[f'{log_name}_Quality_Improvement'].idxmax()
            worst_validation_idx = df[f'{log_name}_Validation_Improvement'].idxmin()
            worst_quality_idx = df[f'{log_name}_Quality_Improvement'].idxmin()
            
            print(f"  Best Validation: Prompt {best_validation_idx + 1} ({df.loc[best_validation_idx, f'{log_name}_Validation_Improvement']:+.1f}%)")
            print(f"  Best Quality: Prompt {best_quality_idx + 1} ({df.loc[best_quality_idx, f'{log_name}_Quality_Improvement']:+.1f}%)")
            print(f"  Worst Validation: Prompt {worst_validation_idx + 1} ({df.loc[worst_validation_idx, f'{log_name}_Validation_Improvement']:+.1f}%)")
            print(f"  Worst Quality: Prompt {worst_quality_idx + 1} ({df.loc[worst_quality_idx, f'{log_name}_Quality_Improvement']:+.1f}%)")
    
    # Save detailed table
    output_file = args.output if args.output else 'log_comparison_table.csv'
    df.to_csv(output_file, index=False)
    print(f"\nDetailed comparison table saved to: {output_file}")
    
    # Display first few rows
    print(f"\nFIRST 10 ROWS OF COMPARISON TABLE:")
    display_cols = ['Prompt_Number', 'Original_Prompt']
    for log_name in log_names:
        display_cols.extend([f'{log_name}_Validation_Score', f'{log_name}_Validation_Hit'])
    if len(all_log_data) > 1:
        for log_name in log_names[1:]:
            display_cols.append(f'{log_name}_Validation_Improvement')
    
    print(df[display_cols].head(10).to_string(index=False))
    
    # Final verdict
    print(f"\n" + "="*80)
    print("FINAL VERDICT:")
    print("="*80)
    
    # Find winners for each metric
    validation_winner = max(hit_stats, key=lambda x: x[3])[0]  # Highest hit percentage
    quality_winner = max(log_names, key=lambda name: df[f'{name}_Quality_Score'].mean())
    alignment_winner = max(log_names, key=lambda name: df[f'{name}_Alignment_Score'].mean())
    
    print(f"Validation Hit Rate Winner: {validation_winner} ({max(hit_stats, key=lambda x: x[3])[3]:.1f}%)")
    print(f"Quality Score Winner: {quality_winner} ({df[f'{quality_winner}_Quality_Score'].mean():.4f})")
    print(f"Alignment Score Winner: {alignment_winner} ({df[f'{alignment_winner}_Alignment_Score'].mean():.4f})")
    
    # Overall winner based on validation hits and quality
    overall_scores = []
    for log_name in log_names:
        hit_pct = next(hit[3] for hit in hit_stats if hit[0] == log_name)
        quality_avg = df[f'{log_name}_Quality_Score'].mean()
        # Combined score: 60% validation hits, 40% quality
        combined_score = hit_pct * 0.6 + quality_avg * 100 * 0.4
        overall_scores.append((log_name, combined_score))
    
    overall_winner = max(overall_scores, key=lambda x: x[1])[0]
    print(f"\nOVERALL WINNER: {overall_winner} (combined score: {max(overall_scores, key=lambda x: x[1])[1]:.1f})")

if __name__ == "__main__":
    main()
