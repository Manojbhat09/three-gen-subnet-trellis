#!/usr/bin/env python3
"""
Complete Log Analyzer for Trellis Simulator Comparison
Parses multiple log files and creates comprehensive comparison table with timing analysis
"""

import re
import pandas as pd
import argparse
import sys
from datetime import datetime
import os

def parse_log_file(file_path):
    """Parse log file and extract prompt data with scores and timing"""
    print(f"Parsing: {file_path}")
    
    if not os.path.exists(file_path):
        print(f"Error: File not found: {file_path}")
        return []
    
    with open(file_path, 'r') as f:
        content = f.read()
    
    # Split content into sections for each prompt
    sections = re.split(r'    Original: \'', content)
    print(f"Found {len(sections)} sections")
    
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
    print(f"Extracted {len(data)} unique prompts")
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
    
    # Look for average validation time patterns
    avg_val_time_match = re.search(r'Average validation time: ([\d.]+)s', section)
    if avg_val_time_match:
        timing_data['average_validation_time'] = float(avg_val_time_match.group(1))
    
    # Look for total time patterns
    total_time_match = re.search(r'Total time: ([\d.]+)s', section)
    if total_time_match:
        timing_data['total_time'] = float(total_time_match.group(1))
    
    # Look for "took" patterns (e.g., "took 2.34s")
    took_match = re.search(r'took ([\d.]+)s', section)
    if took_match:
        timing_data['operation_time'] = float(took_match.group(1))
    
    # Look for elapsed time patterns
    elapsed_match = re.search(r'elapsed: ([\d.]+)s', section)
    if elapsed_match:
        timing_data['elapsed_time'] = float(elapsed_match.group(1))
    
    # Look for processing time patterns
    processing_match = re.search(r'Processing time: ([\d.]+)s', section)
    if processing_match:
        timing_data['processing_time'] = float(processing_match.group(1))
    
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
    
    # Look for specific timing patterns in the logs
    # Generation completed patterns
    gen_complete_match = re.search(r'Generation completed in ([\d.]+)s', section)
    if gen_complete_match:
        timing_data['generation_completed_time'] = float(gen_complete_match.group(1))
    
    # Validation completed patterns
    val_complete_match = re.search(r'Validation completed in ([\d.]+)s', section)
    if val_complete_match:
        timing_data['validation_completed_time'] = float(val_complete_match.group(1))
    
    # Look for inference time patterns
    inference_match = re.search(r'Inference time: ([\d.]+)s', section)
    if inference_match:
        timing_data['inference_time'] = float(inference_match.group(1))
    
    # Look for model time patterns
    model_time_match = re.search(r'Model time: ([\d.]+)s', section)
    if model_time_match:
        timing_data['model_time'] = float(model_time_match.group(1))
    
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
    
    print("="*80)
    print("TRELLIS SIMULATOR LOG COMPARISON ANALYSIS")
    print("="*80)
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
    
    if not all_log_data:
        print("No data found in any log files")
        sys.exit(1)
    
    # Create comparison dataframe
    comparison_data = []
    
    # Get the minimum length to avoid index errors
    min_length = min(len(log_data) for log_data in all_log_data)
    print(f"Comparing {min_length} prompts across {len(all_log_data)} logs")
    
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
            timing_fields = [
                'generation_time', 'validation_time', 'average_validation_time', 'total_time',
                'section_duration', 'generation_completed_time', 'validation_completed_time',
                'operation_time', 'elapsed_time', 'processing_time', 'inference_time', 'model_time'
            ]
            
            for field in timing_fields:
                if field in log_data[i]:
                    # Convert field name to column name (e.g., 'generation_time' -> 'Generation_Time')
                    column_name = field.replace('_', ' ').title().replace(' ', '_')
                    row[f'{log_name}_{column_name}'] = log_data[i][field]
        
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
        if f'{log_name}_Average_Validation_Time' in df.columns:
            avg_val_avg = df[f'{log_name}_Average_Validation_Time'].mean()
            timing_info.append(f"Avg_Validation: {avg_val_avg:.2f}s")
        if f'{log_name}_Total_Time' in df.columns:
            total_avg = df[f'{log_name}_Total_Time'].mean()
            timing_info.append(f"Total: {total_avg:.2f}s")
        if f'{log_name}_Section_Duration' in df.columns:
            section_avg = df[f'{log_name}_Section_Duration'].mean()
            timing_info.append(f"Section: {section_avg:.2f}s")
        if f'{log_name}_Generation_Completed_Time' in df.columns:
            gen_comp_avg = df[f'{log_name}_Generation_Completed_Time'].mean()
            timing_info.append(f"Gen_Completed: {gen_comp_avg:.2f}s")
        if f'{log_name}_Validation_Completed_Time' in df.columns:
            val_comp_avg = df[f'{log_name}_Validation_Completed_Time'].mean()
            timing_info.append(f"Val_Completed: {val_comp_avg:.2f}s")
        if f'{log_name}_Operation_Time' in df.columns:
            op_avg = df[f'{log_name}_Operation_Time'].mean()
            timing_info.append(f"Operation: {op_avg:.2f}s")
        if f'{log_name}_Elapsed_Time' in df.columns:
            elapsed_avg = df[f'{log_name}_Elapsed_Time'].mean()
            timing_info.append(f"Elapsed: {elapsed_avg:.2f}s")
        if f'{log_name}_Processing_Time' in df.columns:
            proc_avg = df[f'{log_name}_Processing_Time'].mean()
            timing_info.append(f"Processing: {proc_avg:.2f}s")
        if f'{log_name}_Inference_Time' in df.columns:
            inf_avg = df[f'{log_name}_Inference_Time'].mean()
            timing_info.append(f"Inference: {inf_avg:.2f}s")
        if f'{log_name}_Model_Time' in df.columns:
            model_avg = df[f'{log_name}_Model_Time'].mean()
            timing_info.append(f"Model: {model_avg:.2f}s")
        
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
    
    # Comprehensive Timing Analysis
    print(f"\n" + "="*80)
    print("COMPREHENSIVE TIMING ANALYSIS")
    print("="*80)
    
    # Create timing summary table
    timing_summary = []
    for log_name in log_names:
        timing_row = {'Method': log_name}
        
        # Get all timing columns for this log
        timing_columns = [col for col in df.columns if col.startswith(f'{log_name}_') and any(timing_field in col.lower() for timing_field in ['time', 'duration', 'elapsed', 'processing', 'inference', 'model'])]
        
        for col in timing_columns:
            if col in df.columns and not df[col].isna().all():
                avg_time = df[col].mean()
                timing_row[col.replace(f'{log_name}_', '')] = f"{avg_time:.2f}s"
        
        timing_summary.append(timing_row)
    
    # Print timing summary
    if timing_summary:
        print(f"\nDETAILED TIMING BREAKDOWN:")
        for row in timing_summary:
            print(f"\n{row['Method']}:")
            for key, value in row.items():
                if key != 'Method':
                    print(f"  {key}: {value}")
    
    # Find fastest methods for each timing metric
    print(f"\nTIMING PERFORMANCE RANKINGS:")
    
    # Get all unique timing metrics across all logs
    all_timing_metrics = set()
    for log_name in log_names:
        timing_columns = [col for col in df.columns if col.startswith(f'{log_name}_') and any(timing_field in col.lower() for timing_field in ['time', 'duration', 'elapsed', 'processing', 'inference', 'model'])]
        for col in timing_columns:
            metric_name = col.replace(f'{log_name}_', '')
            all_timing_metrics.add(metric_name)
    
    # Rank methods for each timing metric
    for metric in sorted(all_timing_metrics):
        metric_data = []
        for log_name in log_names:
            col_name = f'{log_name}_{metric}'
            if col_name in df.columns and not df[col_name].isna().all():
                avg_time = df[col_name].mean()
                metric_data.append((log_name, avg_time))
        
        if metric_data:
            # Sort by time (ascending - fastest first)
            metric_data.sort(key=lambda x: x[1])
            print(f"\n{metric.replace('_', ' ').title()}:")
            for i, (log_name, avg_time) in enumerate(metric_data, 1):
                print(f"  {i}. {log_name}: {avg_time:.2f}s")
    
    # Overall timing efficiency
    print(f"\nOVERALL TIMING EFFICIENCY:")
    efficiency_scores = []
    for log_name in log_names:
        # Calculate efficiency score based on all available timing metrics
        timing_columns = [col for col in df.columns if col.startswith(f'{log_name}_') and any(timing_field in col.lower() for timing_field in ['time', 'duration', 'elapsed', 'processing', 'inference', 'model'])]
        
        if timing_columns:
            # Calculate average of all timing metrics (lower is better)
            total_time = 0
            count = 0
            for col in timing_columns:
                if not df[col].isna().all():
                    total_time += df[col].mean()
                    count += 1
            
            if count > 0:
                avg_total_time = total_time / count
                efficiency_scores.append((log_name, avg_total_time))
    
    if efficiency_scores:
        efficiency_scores.sort(key=lambda x: x[1])  # Sort by time (ascending)
        print("Ranked by average timing efficiency (lower is better):")
        for i, (log_name, avg_time) in enumerate(efficiency_scores, 1):
            print(f"  {i}. {log_name}: {avg_time:.2f}s average")
    
    # Generation + Validation time analysis
    print(f"\nGENERATION + VALIDATION TIME ANALYSIS:")
    gen_val_times = []
    for log_name in log_names:
        gen_col = f'{log_name}_Generation_Time'
        val_col = f'{log_name}_Validation_Time'
        
        if gen_col in df.columns and val_col in df.columns:
            if not df[gen_col].isna().all() and not df[val_col].isna().all():
                gen_avg = df[gen_col].mean()
                val_avg = df[val_col].mean()
                total_avg = gen_avg + val_avg
                gen_val_times.append((log_name, gen_avg, val_avg, total_avg))
                print(f"{log_name}: Generation={gen_avg:.2f}s + Validation={val_avg:.2f}s = Total={total_avg:.2f}s")
    
    if gen_val_times:
        # Find fastest for generation + validation
        gen_val_times.sort(key=lambda x: x[3])  # Sort by total time
        fastest_gen_val = gen_val_times[0]
        print(f"\nFastest Generation + Validation: {fastest_gen_val[0]} ({fastest_gen_val[3]:.2f}s total)")

if __name__ == "__main__":
    main()
