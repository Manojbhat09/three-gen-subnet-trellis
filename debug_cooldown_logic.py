#!/usr/bin/env python3
"""
Debug script to analyze cooldown logic issues for UID 81
Why are violations continuously increasing without proper cooldowns?
"""

import json
import time
from datetime import datetime

def analyze_cooldown_logic():
    """Analyze the cooldown logic from the log"""
    
    print("🔍 ANALYZING COOLDOWN LOGIC FOR UID 81")
    print("="*60)
    
    # Sample log entries for UID 81 from the provided log
    log_entries = [
        {
            "timestamp": "08:33:08",
            "violations": 3,
            "cooldown_until": 1756888863,
            "status": "successful_pull"
        },
        {
            "timestamp": "08:34:13", 
            "violations": 4,
            "cooldown_until": 1756888863,
            "status": "successful_pull"
        },
        {
            "timestamp": "08:35:19",
            "violations": 5, 
            "cooldown_until": 1756888863,
            "status": "successful_pull"
        },
        {
            "timestamp": "08:36:25",
            "violations": 6,
            "cooldown_until": 1756888863, 
            "status": "successful_pull"
        },
        {
            "timestamp": "08:37:36",
            "violations": 7,
            "cooldown_until": 1756888863,
            "status": "successful_pull"
        },
        {
            "timestamp": "08:38:41",
            "violations": 8,
            "cooldown_until": 1756888863,
            "status": "successful_pull"
        },
        {
            "timestamp": "08:39:48",
            "violations": 9,
            "cooldown_until": 1756888863,
            "status": "successful_pull"
        },
        {
            "timestamp": "08:40:59",
            "violations": 10,
            "cooldown_until": 1756888863,
            "status": "successful_pull"
        }
    ]
    
    print("📊 VIOLATION PROGRESSION ANALYSIS:")
    print("-" * 40)
    
    for i, entry in enumerate(log_entries):
        print(f"Pull {i+1} at {entry['timestamp']}:")
        print(f"   Violations: {entry['violations']}")
        print(f"   Cooldown Until: {entry['cooldown_until']}")
        print(f"   Status: {entry['status']}")
        
        # Calculate time between pulls
        if i > 0:
            prev_time = datetime.strptime(log_entries[i-1]['timestamp'], "%H:%M:%S")
            curr_time = datetime.strptime(entry['timestamp'], "%H:%M:%S")
            time_diff = (curr_time - prev_time).total_seconds()
            print(f"   Time since last pull: {time_diff:.0f}s")
            
            # Check if this violates MIN_TASK_INTERVAL
            if time_diff < 35:
                print(f"   🚨 VIOLATION: {time_diff:.0f}s < 35s MIN_TASK_INTERVAL!")
            else:
                print(f"   ✅ OK: {time_diff:.0f}s >= 35s MIN_TASK_INTERVAL")
        print()
    
    print("🔍 COOLDOWN LOGIC ANALYSIS:")
    print("-" * 40)
    
    # Analyze the cooldown_until value
    cooldown_until = 1756888863
    current_time = int(time.time())
    
    print(f"Cooldown Until: {cooldown_until}")
    print(f"Current Time: {current_time}")
    print(f"Cooldown Active: {current_time < cooldown_until}")
    
    if current_time < cooldown_until:
        remaining = cooldown_until - current_time
        print(f"Remaining Cooldown: {remaining}s ({remaining/60:.1f} minutes)")
    else:
        print("Cooldown has expired")
    
    print("\n🚨 ISSUES IDENTIFIED:")
    print("-" * 40)
    print("1. UID 81 violations increase by +1 on every pull")
    print("2. All pulls show 'successful_pull' but violations increase")
    print("3. Cooldown_until remains constant (1756888863)")
    print("4. Time between pulls varies but violations still increase")
    print("5. This suggests the validator is penalizing us for something")
    
    print("\n💡 POSSIBLE CAUSES:")
    print("-" * 40)
    print("1. We're pulling too frequently (violating MIN_TASK_INTERVAL)")
    print("2. We're not respecting the validator's cooldown_until")
    print("3. There's a bug in our cooldown checking logic")
    print("4. The validator has additional restrictions we're not aware of")
    print("5. We're submitting results too quickly after pulling")
    
    print("\n🔧 RECOMMENDED FIXES:")
    print("-" * 40)
    print("1. Check if we're respecting cooldown_until from validator response")
    print("2. Ensure MIN_TASK_INTERVAL is properly enforced")
    print("3. Add more detailed logging of cooldown decisions")
    print("4. Check if we're submitting results within throttle period")
    print("5. Verify our cooldown state synchronization logic")

if __name__ == "__main__":
    analyze_cooldown_logic()

