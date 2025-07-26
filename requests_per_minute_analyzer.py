#!/usr/bin/env python3
"""
Requests Per Minute Analyzer
============================
Analyzes the episodic memory JSON to calculate average requests per minute
based on timestamps in the optimization attempts.
"""

import json
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
import numpy as np

def analyze_requests_per_minute(memory_file="episodic_logs/episodic_memory.json"):
    """Analyze requests per minute from episodic memory data"""
    
    # Load the JSON data
    with open(memory_file, 'r') as f:
        data = json.load(f)
    
    # Extract all attempts with timestamps
    attempts_data = []
    for session in data.get('optimization_sessions', []):
        for attempt in session.get('attempts', []):
            attempts_data.append({
                'session_id': session['session_id'],
                'original_prompt': session['original_prompt'],
                'attempt_number': attempt['attempt_number'],
                'timestamp': attempt.get('timestamp', 0),
                'validation_score': attempt.get('validation_score', 0.0),
                'strategy_used': attempt['strategy_used']
            })
    
    df = pd.DataFrame(attempts_data)
    
    if len(df) == 0:
        print("No data found in the memory file.")
        return
    
    # Convert timestamps to datetime
    df['datetime'] = pd.to_datetime(df['timestamp'], unit='s')
    df = df.sort_values('datetime')
    
    # Calculate time range
    start_time = df['datetime'].min()
    end_time = df['datetime'].max()
    total_duration = end_time - start_time
    
    print("=== REQUESTS PER MINUTE ANALYSIS ===")
    print(f"Start time: {start_time}")
    print(f"End time: {end_time}")
    print(f"Total duration: {total_duration}")
    print(f"Total attempts: {len(df)}")
    
    # Calculate requests per minute
    total_minutes = total_duration.total_seconds() / 60
    requests_per_minute = len(df) / total_minutes
    
    print(f"\n=== OVERALL STATISTICS ===")
    print(f"Total minutes: {total_minutes:.2f}")
    print(f"Average requests per minute: {requests_per_minute:.2f}")
    print(f"Average time between requests: {60/requests_per_minute:.2f} seconds")
    
    # Analyze by time periods
    print(f"\n=== TIME-BASED ANALYSIS ===")
    
    # Group by minute and calculate requests per minute
    df['minute'] = df['datetime'].dt.floor('min')
    requests_per_minute_data = df.groupby('minute').size()
    
    print(f"Minutes with requests: {len(requests_per_minute_data)}")
    print(f"Average requests per active minute: {requests_per_minute_data.mean():.2f}")
    print(f"Max requests in a single minute: {requests_per_minute_data.max()}")
    print(f"Min requests in a single minute: {requests_per_minute_data.min()}")
    
    # Calculate peak periods
    peak_minute = requests_per_minute_data.idxmax()
    print(f"Peak minute: {peak_minute} with {requests_per_minute_data.max()} requests")
    
    # Analyze by hour
    df['hour'] = df['datetime'].dt.hour
    hourly_stats = df.groupby('hour').size()
    print(f"\n=== HOURLY BREAKDOWN ===")
    for hour, count in hourly_stats.items():
        print(f"Hour {hour:02d}:00 - {count} requests")
    
    # Calculate moving average (5-minute window)
    print(f"\n=== MOVING AVERAGE ANALYSIS ===")
    df_sorted = df.sort_values('datetime')
    df_sorted['minute_group'] = df_sorted['datetime'].dt.floor('min')
    minute_counts = df_sorted.groupby('minute_group').size().reset_index(name='count')
    
    if len(minute_counts) > 5:
        # 5-minute moving average
        moving_avg_5min = minute_counts['count'].rolling(window=5, min_periods=1).mean()
        print(f"5-minute moving average range: {moving_avg_5min.min():.2f} - {moving_avg_5min.max():.2f}")
        print(f"Current 5-minute moving average: {moving_avg_5min.iloc[-1]:.2f}")
    
    # Analyze recent activity (last 10 minutes)
    print(f"\n=== RECENT ACTIVITY (Last 10 minutes) ===")
    ten_minutes_ago = df['datetime'].max() - timedelta(minutes=10)
    recent_activity = df[df['datetime'] >= ten_minutes_ago]
    
    if len(recent_activity) > 0:
        recent_minutes = (df['datetime'].max() - ten_minutes_ago).total_seconds() / 60
        recent_rpm = len(recent_activity) / recent_minutes
        print(f"Recent requests: {len(recent_activity)}")
        print(f"Recent requests per minute: {recent_rpm:.2f}")
    else:
        print("No activity in the last 10 minutes")
    
    # Session-based analysis
    print(f"\n=== SESSION-BASED ANALYSIS ===")
    session_stats = df.groupby('session_id').agg({
        'datetime': ['min', 'max', 'count']
    }).round(2)
    session_stats.columns = ['start_time', 'end_time', 'attempts']
    session_stats['duration_minutes'] = (session_stats['end_time'] - session_stats['start_time']).dt.total_seconds() / 60
    session_stats['rpm_per_session'] = session_stats['attempts'] / session_stats['duration_minutes']
    
    print(f"Average session duration: {session_stats['duration_minutes'].mean():.2f} minutes")
    print(f"Average requests per minute per session: {session_stats['rpm_per_session'].mean():.2f}")
    print(f"Fastest session: {session_stats['rpm_per_session'].max():.2f} requests/minute")
    print(f"Slowest session: {session_stats['rpm_per_session'].min():.2f} requests/minute")
    
    # Strategy-based timing analysis
    print(f"\n=== STRATEGY TIMING ANALYSIS ===")
    strategy_timing = df.groupby('strategy_used').agg({
        'datetime': ['min', 'max', 'count']
    }).round(2)
    strategy_timing.columns = ['first_used', 'last_used', 'total_attempts']
    strategy_timing['duration_minutes'] = (strategy_timing['last_used'] - strategy_timing['first_used']).dt.total_seconds() / 60
    strategy_timing['rpm_per_strategy'] = strategy_timing['total_attempts'] / strategy_timing['duration_minutes']
    
    for strategy, stats in strategy_timing.iterrows():
        print(f"{strategy}: {stats['rpm_per_strategy']:.2f} requests/minute over {stats['duration_minutes']:.1f} minutes")
    
    return {
        'total_requests': len(df),
        'total_minutes': total_minutes,
        'avg_requests_per_minute': requests_per_minute,
        'avg_time_between_requests': 60/requests_per_minute,
        'peak_requests_per_minute': requests_per_minute_data.max(),
        'session_stats': session_stats,
        'strategy_timing': strategy_timing
    }

if __name__ == "__main__":
    try:
        results = analyze_requests_per_minute()
        print(f"\n=== SUMMARY ===")
        print(f"Current average: {results['avg_requests_per_minute']:.2f} requests/minute")
        print(f"Peak rate: {results['peak_requests_per_minute']} requests/minute")
        print(f"Average interval: {results['avg_time_between_requests']:.1f} seconds between requests")
    except Exception as e:
        print(f"Error analyzing requests per minute: {e}")
        import traceback
        traceback.print_exc() 