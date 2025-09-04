#!/usr/bin/env python3
"""
Violation Analysis Script
Analyzes continuous_trellis.log to understand violation patterns and events
"""

import re
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any
from collections import defaultdict


class ViolationAnalyzer:
    """Analyzes violation patterns from log files"""

    def __init__(self, log_file: str = "continuous_trellis.log"):
        self.log_file = Path(log_file)
        self.events = []
        self.uid_events = defaultdict(list)

    def parse_logs(self) -> None:
        """Parse the log file and extract violation-related events"""
        print(f"📖 Reading log file: {self.log_file}")

        if not self.log_file.exists():
            print(f"❌ Log file not found: {self.log_file}")
            return

        with open(self.log_file, 'r') as f:
            for line_num, line in enumerate(f, 1):
                # Extract timestamp and message
                timestamp_match = re.match(r'(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3})', line)
                if not timestamp_match:
                    continue

                timestamp_str = timestamp_match.group(1)
                message = line[timestamp_match.end():].strip()

                # Parse timestamp
                try:
                    timestamp = datetime.strptime(timestamp_str, '%Y-%m-%d %H:%M:%S,%f')
                    timestamp_unix = timestamp.timestamp()
                except ValueError:
                    continue

                event = {
                    'timestamp': timestamp_unix,
                    'timestamp_readable': timestamp_str,
                    'line_number': line_num,
                    'message': message,
                    'raw_line': line.strip()
                }

                self.events.append(event)

        print(f"✅ Parsed {len(self.events)} log events")

    def extract_uid_events(self, uids: List[int]) -> Dict[int, List[Dict[str, Any]]]:
        """Extract events related to specific UIDs"""
        uid_pattern = '|'.join([f'UID {uid}' for uid in uids])
        uid_events = defaultdict(list)

        print(f"🔍 Extracting events for UIDs: {uids}")

        for event in self.events:
            if re.search(uid_pattern, event['message']):
                # Determine which UID this event belongs to
                for uid in uids:
                    if f'UID {uid}' in event['message']:
                        uid_events[uid].append(event)
                        break

        # Sort events by timestamp for each UID
        for uid in uid_events:
            uid_events[uid].sort(key=lambda x: x['timestamp'])

        print(f"📊 Found events for {len(uid_events)} UIDs:")
        for uid, events in uid_events.items():
            print(f"   UID {uid}: {len(events)} events")

        return uid_events

    def analyze_violation_patterns(self, uid_events: Dict[int, List[Dict[str, Any]]]) -> Dict[int, Dict[str, Any]]:
        """Analyze violation patterns for each UID"""
        analysis = {}

        for uid, events in uid_events.items():
            print(f"\n🔍 ANALYZING UID {uid} ({len(events)} events)")
            print("=" * 60)

            violation_events = []
            pull_events = []
            submit_events = []
            cooldown_events = []

            for event in events:
                msg = event['message']

                # Violation events
                if 'Violation +' in msg and f'UID {uid}' in msg:
                    violation_events.append(event)
                elif 'Violation -' in msg and f'UID {uid}' in msg:
                    violation_events.append(event)
                elif 'VIOLATION REPORTED: UID' in msg and f'UID {uid}' in msg:
                    violation_events.append(event)

                # Pull events
                elif 'PULL TASK SUCCESS' in msg and f'VALIDATOR UID: {uid}' in msg:
                    pull_events.append(event)

                # Submit events
                elif 'SUBMIT SUCCESS' in msg and f'VALIDATOR UID: {uid}' in msg:
                    submit_events.append(event)

                # Cooldown events
                elif 'cooldown' in msg.lower() and f'UID {uid}' in msg:
                    cooldown_events.append(event)

            analysis[uid] = {
                'violation_events': violation_events,
                'pull_events': pull_events,
                'submit_events': submit_events,
                'cooldown_events': cooldown_events,
                'total_events': len(events)
            }

            # Print summary
            print(f"Violation events: {len(violation_events)}")
            print(f"Pull events: {len(pull_events)}")
            print(f"Submit events: {len(submit_events)}")
            print(f"Cooldown events: {len(cooldown_events)}")

            # Analyze violation timeline
            if violation_events:
                print("\n🚨 VIOLATION TIMELINE:")
                for i, event in enumerate(violation_events[-10:], 1):  # Show last 10
                    print(f"   {i}. {event['timestamp_readable']}: {event['message'][:100]}...")

            # Analyze pull timeline
            if pull_events:
                print("\n📡 PULL TIMELINE:")
                for i, event in enumerate(pull_events[-5:], 1):  # Show last 5
                    print(f"   {i}. {event['timestamp_readable']}: Pull task")

                    # Look for cooldown info in this event or nearby events
                    event_idx = self.events.index(event)
                    for j in range(max(0, event_idx-2), min(len(self.events), event_idx+10)):
                        nearby_event = self.events[j]
                        if 'COOLDOWN UNTIL:' in nearby_event['message'] and f'UID {uid}' in nearby_event['message']:
                            print(f"      → {nearby_event['timestamp_readable']}: {nearby_event['message']}")
                            break

        return analysis

    def save_analysis(self, analysis: Dict[int, Dict[str, Any]], output_file: str = "violation_analysis.json") -> None:
        """Save analysis results to JSON file"""
        # Convert events to serializable format
        serializable_analysis = {}

        for uid, data in analysis.items():
            serializable_analysis[str(uid)] = {
                'total_events': data['total_events'],
                'violation_count': len(data['violation_events']),
                'pull_count': len(data['pull_events']),
                'submit_count': len(data['submit_events']),
                'cooldown_count': len(data['cooldown_events']),
                'violation_events': [
                    {
                        'timestamp': event['timestamp'],
                        'timestamp_readable': event['timestamp_readable'],
                        'message': event['message']
                    } for event in data['violation_events']
                ],
                'pull_events': [
                    {
                        'timestamp': event['timestamp'],
                        'timestamp_readable': event['timestamp_readable'],
                        'message': event['message']
                    } for event in data['pull_events']
                ]
            }

        with open(output_file, 'w') as f:
            json.dump(serializable_analysis, f, indent=2)

        print(f"💾 Analysis saved to: {output_file}")


def main():
    """Main analysis function"""
    analyzer = ViolationAnalyzer()

    # Parse logs
    analyzer.parse_logs()

    # Extract events for problematic UIDs
    target_uids = [49, 128, 142, 212]  # Based on log analysis
    uid_events = analyzer.extract_uid_events(target_uids)

    # Analyze patterns
    analysis = analyzer.analyze_violation_patterns(uid_events)

    # Save results
    analyzer.save_analysis(analysis)

    # Print summary
    print("\n" + "="*80)
    print("🎯 ANALYSIS SUMMARY")
    print("="*80)

    for uid in target_uids:
        if uid in analysis:
            data = analysis[uid]
            violations = len(data['violation_events'])
            pulls = len(data['pull_events'])
            submits = len(data['submit_events'])

            print(f"\nUID {uid}:")
            print(f"   Violations: {violations}")
            print(f"   Pull tasks: {pulls}")
            print(f"   Submissions: {submits}")
            print(f"   Ratio: {violations/pulls:.2f} violations per pull" if pulls > 0 else "   Ratio: N/A")


if __name__ == "__main__":
    main()


