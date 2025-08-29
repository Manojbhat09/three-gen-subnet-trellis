#!/usr/bin/env python3
"""
Script to parse the last 10 instances of Task fidelity: 0.0000 from the log file
and extract the original and cleaned prompts.
"""

import re
import json
from typing import List, Tuple, Optional


def extract_prompt_from_task_data(task_data_str: str) -> Optional[str]:
    """Extract prompt from task data string."""
    try:
        # Look for the prompt in the task data
        prompt_match = re.search(r"prompt='([^']*)'", task_data_str)
        if prompt_match:
            return prompt_match.group(1)
        return None
    except Exception as e:
        print(f"Error extracting prompt: {e}")
        return None


def parse_log_for_zero_fidelity(log_file_path: str, num_entries: int = 10) -> List[Tuple[str, str, str]]:
    """
    Parse log file for the last N instances of Task fidelity: 0.0000
    Returns list of tuples: (timestamp, task_id, prompt)
    """
    fidelity_entries = []

    try:
        with open(log_file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()

        # Find all lines with Task fidelity: 0.0000
        fidelity_lines = []
        for i, line in enumerate(lines):
            if "Task fidelity: 0.0000" in line:
                fidelity_lines.append(i)

        # Get the last N entries
        last_fidelity_lines = fidelity_lines[-num_entries:] if len(fidelity_lines) >= num_entries else fidelity_lines

        for line_idx in last_fidelity_lines:
            timestamp = ""
            task_id = ""
            prompt = ""

            # Extract timestamp from fidelity line
            fidelity_line = lines[line_idx].strip()
            timestamp_match = re.search(r'(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3})', fidelity_line)
            if timestamp_match:
                timestamp = timestamp_match.group(1)

            # Look backwards for the DEBUG line containing task data
            for j in range(line_idx - 1, max(-1, line_idx - 20), -1):
                line = lines[j]
                if "🔍 DEBUG: Validator submission response data:" in line:
                    # Extract task data from the line
                    task_data_match = re.search(r"'task': Task\([^)]+\)", line)
                    if task_data_match:
                        task_data_str = task_data_match.group(0)

                        # Extract task ID
                        id_match = re.search(r"id='([^']*)'", task_data_str)
                        if id_match:
                            task_id = id_match.group(1)

                        # Extract prompt
                        prompt_match = re.search(r"prompt='([^']*)'", task_data_str)
                        if prompt_match:
                            prompt = prompt_match.group(1)

                    break

            fidelity_entries.append((timestamp, task_id, prompt))

    except Exception as e:
        print(f"Error parsing log file: {e}")

    return fidelity_entries


def main():
    log_file_path = "/home/mbhat/three-gen-subnet-trellis/continuous_trellis.log"
    num_entries = 10

    print(f"Parsing the last {num_entries} instances of 'Task fidelity: 0.0000' from:")
    print(f"{log_file_path}")
    print("=" * 80)

    fidelity_entries = parse_log_for_zero_fidelity(log_file_path, num_entries)

    if not fidelity_entries:
        print("No entries found with Task fidelity: 0.0000")
        return

    for i, (timestamp, task_id, prompt) in enumerate(fidelity_entries, 1):
        print(f"\n--- Entry {i} ---")
        print(f"Timestamp: {timestamp}")
        print(f"Task ID: {task_id}")
        print(f"Prompt: {prompt}")
        print("-" * 40)

    print(f"\nTotal entries found: {len(fidelity_entries)}")


if __name__ == "__main__":
    main()



