#!/usr/bin/env python3
"""
Script to parse the last 10 instances of high Task fidelity scores (> 0.5) from the log file
and extract the original and cleaned prompts, along with CLIP similarity and fidelity scores.
"""

import re
import json
from typing import List, Tuple, Optional, Dict


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


def extract_fidelity_score(line: str) -> Optional[float]:
    """Extract fidelity score from a log line."""
    try:
        match = re.search(r"Task fidelity: ([0-9.]+)", line)
        if match:
            return float(match.group(1))
        return None
    except Exception as e:
        print(f"Error extracting fidelity score: {e}")
        return None


def parse_log_for_high_fidelity(log_file_path: str, min_score: float = 0.5, num_entries: int = 10) -> List[Dict]:
    """
    Parse log file for high fidelity scores and extract optimization details.

    Returns list of dictionaries with optimization details.
    """
    high_fidelity_entries = []

    try:
        with open(log_file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()

        # Find all lines with high Task fidelity scores
        fidelity_lines = []
        for i, line in enumerate(lines):
            if "Task fidelity:" in line:
                score = extract_fidelity_score(line)
                if score and score >= min_score:
                    fidelity_lines.append((i, score))

        # Sort by score (highest first) and get top N
        fidelity_lines.sort(key=lambda x: x[1], reverse=True)
        top_fidelity_lines = fidelity_lines[:num_entries]

        for line_idx, fidelity_score in top_fidelity_lines:
            entry = {
                'timestamp': '',
                'task_id': '',
                'original_prompt': '',
                'optimized_prompt': '',
                'cleaned_prompt': '',
                'clip_similarity': 0.0,
                'similarity_level': '',
                'fidelity_score': fidelity_score,
                'lora_used': '',
                'validator_uid': ''
            }

            # Extract timestamp from fidelity line
            fidelity_line = lines[line_idx].strip()
            timestamp_match = re.search(r'(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3})', fidelity_line)
            if timestamp_match:
                entry['timestamp'] = timestamp_match.group(1)

            # Look backwards for optimization details and task data
            original_found = False
            optimized_found = False
            cleaned_found = False
            clip_found = False

            # Search backwards from fidelity line (up to 200 lines back)
            for j in range(line_idx - 1, max(-1, line_idx - 200), -1):
                line = lines[j]

                # Extract validator UID from cooldown messages
                if "Validator UID" in line and not entry['validator_uid']:
                    uid_match = re.search(r"Validator UID (\d+)", line)
                    if uid_match:
                        entry['validator_uid'] = uid_match.group(1)

                # Extract CLIP similarity and level
                if not clip_found and "Similarity:" in line and "level:" in line:
                    similarity_match = re.search(r"Similarity: ([0-9.]+)", line)
                    level_match = re.search(r"Similarity level: ([^,]+)", line)
                    if similarity_match and level_match:
                        entry['clip_similarity'] = float(similarity_match.group(1))
                        entry['similarity_level'] = level_match.group(1).strip()
                        clip_found = True

                # Extract LoRA information
                if "Using LoRA:" in line and not entry['lora_used']:
                    lora_match = re.search(r"Using LoRA: ([^v]+)", line)
                    if lora_match:
                        entry['lora_used'] = lora_match.group(1).strip()

                # Extract prompts from FINAL OPTIMIZATION RESULT
                if "🎯 FINAL OPTIMIZATION RESULT:" in line:
                    # Look for the next few lines for prompt details
                    for k in range(j + 1, min(len(lines), j + 10)):
                        next_line = lines[k]

                        if not original_found and next_line.strip().startswith("Original:"):
                            orig_match = re.search(r"Original: '([^']*)'", next_line)
                            if orig_match:
                                entry['original_prompt'] = orig_match.group(1)
                                original_found = True

                        if not optimized_found and next_line.strip().startswith("Optimized:"):
                            opt_match = re.search(r"Optimized: '([^']*)'", next_line)
                            if opt_match:
                                entry['optimized_prompt'] = opt_match.group(1)
                                optimized_found = True

                        if not cleaned_found and next_line.strip().startswith("Cleaned:"):
                            clean_match = re.search(r"Cleaned: '([^']*)'", next_line)
                            if clean_match:
                                entry['cleaned_prompt'] = clean_match.group(1)
                                cleaned_found = True

                # Extract task data from DEBUG line
                if "🔍 DEBUG: Validator submission response data:" in line:
                    # Extract task data from the line
                    task_data_match = re.search(r"'task': Task\([^)]+\)", line)
                    if task_data_match:
                        task_data_str = task_data_match.group(0)

                        # Extract task ID
                        id_match = re.search(r"id='([^']*)'", task_data_str)
                        if id_match:
                            entry['task_id'] = id_match.group(1)

                        # Extract original prompt (should match optimized original)
                        prompt_match = re.search(r"prompt='([^']*)'", task_data_str)
                        if prompt_match and not entry['original_prompt']:
                            entry['original_prompt'] = prompt_match.group(1)

                    break  # Stop searching backwards once we find the task data

            # Only add entries that have complete information
            if entry['original_prompt'] and entry['task_id']:
                high_fidelity_entries.append(entry)

    except Exception as e:
        print(f"Error parsing log file: {e}")

    return high_fidelity_entries


def save_to_json(data: List[Dict], filename: str):
    """Save the parsed data to JSON file."""
    try:
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        print(f"✅ Data saved to {filename}")
    except Exception as e:
        print(f"❌ Error saving to JSON: {e}")


def main():
    log_file_path = "/home/mbhat/three-gen-subnet-trellis/continuous_trellis.log"
    min_score = 0.5  # Minimum fidelity score to consider "high"
    num_entries = 10

    print(f"Parsing the top {num_entries} instances of 'Task fidelity >= {min_score}' from:")
    print(f"{log_file_path}")
    print("=" * 80)

    high_fidelity_entries = parse_log_for_high_fidelity(log_file_path, min_score, num_entries)

    if not high_fidelity_entries:
        print(f"No entries found with Task fidelity >= {min_score}")
        return

    print(f"Found {len(high_fidelity_entries)} high fidelity entries:")
    print()

    for i, entry in enumerate(high_fidelity_entries, 1):
        print(f"--- High Fidelity Entry {i} ---")
        print(f"Timestamp: {entry['timestamp']}")
        print(f"Task ID: {entry['task_id']}")
        print(f"Validator UID: {entry['validator_uid']}")
        print(f"Original Prompt: {entry['original_prompt']}")
        print(f"Fidelity Score: {entry['fidelity_score']}")
        print(f"CLIP Similarity: {entry['clip_similarity']}")
        print(f"Similarity Level: {entry['similarity_level']}")
        print(f"LoRA Used: {entry['lora_used']}")
        print(f"Optimized Length: {len(entry['optimized_prompt'])} chars")
        print(f"Cleaned Length: {len(entry['cleaned_prompt'])} chars")
        print("-" * 60)

    # Save detailed data to JSON
    json_filename = f"high_fidelity_analysis_min{min_score}.json"
    save_to_json(high_fidelity_entries, json_filename)

    # Print summary statistics
    print("
📊 SUMMARY STATISTICS:"    print(f"Total high fidelity entries: {len(high_fidelity_entries)}")
    if high_fidelity_entries:
        avg_fidelity = sum(e['fidelity_score'] for e in high_fidelity_entries) / len(high_fidelity_entries)
        avg_clip = sum(e['clip_similarity'] for e in high_fidelity_entries) / len(high_fidelity_entries)
        print(".4f"        print(".4f"
        # Validator distribution
        validators = {}
        for entry in high_fidelity_entries:
            uid = entry['validator_uid']
            if uid:
                validators[uid] = validators.get(uid, 0) + 1

        print(f"Validators with high scores: {dict(validators)}")


if __name__ == "__main__":
    main()
