#!/usr/bin/env python3
"""
Extract high fidelity prompts (> 0.75) with their optimization details and fidelity scores.
"""

import re
import json
from typing import List, Dict, Optional
from collections import defaultdict


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


def extract_prompt_from_task_data(task_data_str: str) -> Optional[str]:
    """Extract prompt from task data string."""
    try:
        prompt_match = re.search(r"prompt='([^']*)'", task_data_str)
        if prompt_match:
            return prompt_match.group(1)
        return None
    except Exception as e:
        return None


def extract_optimization_details(log_lines: List[str], fidelity_line_idx: int) -> Dict:
    """Extract optimization details around a fidelity score line."""
    entry = {
        'timestamp': '',
        'task_id': '',
        'original_prompt': '',
        'optimized_prompt': '',
        'cleaned_prompt': '',
        'clip_similarity': 0.0,
        'similarity_level': '',
        'fidelity_score': 0.0,
        'lora_used': '',
        'validator_uid': '',
        'optimization_applied': False
    }

    # Extract timestamp and fidelity score
    fidelity_line = log_lines[fidelity_line_idx].strip()
    timestamp_match = re.search(r'(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3})', fidelity_line)
    fidelity_match = re.search(r'Task fidelity: ([0-9.]+)', fidelity_line)

    if timestamp_match:
        entry['timestamp'] = timestamp_match.group(1)
    if fidelity_match:
        entry['fidelity_score'] = float(fidelity_match.group(1))

    # Search backwards for optimization details (up to 150 lines back)
    search_start = max(0, fidelity_line_idx - 150)

    for i in range(fidelity_line_idx - 1, search_start - 1, -1):
        line = log_lines[i]

        # Extract validator UID
        if "Validator UID" in line and "enforced cooldown" in line and not entry['validator_uid']:
            uid_match = re.search(r"Validator UID (\d+)", line)
            if uid_match:
                entry['validator_uid'] = uid_match.group(1)

        # Extract CLIP similarity info
        if "Similarity:" in line and "level:" in line and entry['clip_similarity'] == 0.0:
            similarity_match = re.search(r"Similarity: ([0-9.]+)", line)
            level_match = re.search(r"Similarity level: ([^,]+)", line)
            if similarity_match and level_match:
                entry['clip_similarity'] = float(similarity_match.group(1))
                entry['similarity_level'] = level_match.group(1).strip()

        # Extract LoRA info
        if "Using LoRA:" in line and not entry['lora_used']:
            lora_match = re.search(r"Using LoRA: ([^v]+)", line)
            if lora_match:
                entry['lora_used'] = lora_match.group(1).strip()

        # Check if optimization was applied
        if "No optimization applied - using original prompt" in line:
            entry['optimization_applied'] = False

        # Extract task ID from processing line
        if "Processing task" in line and not entry['task_id']:
            task_match = re.search(r"Processing task ([^:]+)", line)
            if task_match:
                entry['task_id'] = task_match.group(1).strip()

        # Extract original prompt from processing line
        if "🎨 Generating 3D model:" in line and not entry['original_prompt']:
            prompt_match = re.search(r"🎨 Generating 3D model: '([^']*)'", line)
            if prompt_match:
                entry['original_prompt'] = prompt_match.group(1)

        # Extract optimized prompt
        if "Optimized:" in line and "FINAL OPTIMIZATION RESULT" not in log_lines[i-1] and not entry['optimized_prompt']:
            opt_match = re.search(r"Optimized: '([^']*)'", line)
            if opt_match:
                entry['optimized_prompt'] = opt_match.group(1)
                entry['optimization_applied'] = True

        # Extract cleaned prompt
        if "Cleaned:" in line and not entry['cleaned_prompt']:
            clean_match = re.search(r"Cleaned: '([^']*)'", line)
            if clean_match:
                entry['cleaned_prompt'] = clean_match.group(1)

        # Extract task data from DEBUG line
        if "🔍 DEBUG: Validator submission response data:" in line:
            task_data_match = re.search(r"'task': Task\([^)]+\)", line)
            if task_data_match:
                task_data_str = task_data_match.group(0)

                # Extract task ID
                id_match = re.search(r"id='([^']*)'", task_data_str)
                if id_match and not entry['task_id']:
                    entry['task_id'] = id_match.group(1)

                # Extract original prompt (should match optimized original)
                prompt_match = re.search(r"prompt='([^']*)'", task_data_str)
                if prompt_match and not entry['original_prompt']:
                    entry['original_prompt'] = prompt_match.group(1)

            break  # Stop searching once we find the task data

    return entry


def main():
    log_file = "/home/mbhat/three-gen-subnet-trellis/continuous_trellis.log"
    min_score = 0.75  # Minimum fidelity score for "high"

    print("🔍 Extracting High-Fidelity Prompts (> 0.75)")
    print("=" * 60)

    high_fidelity_prompts = []

    with open(log_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    print(f"Scanning {len(lines)} log lines...")

    # Find all lines with high Task fidelity scores
    high_fidelity_lines = []
    for i, line in enumerate(lines):
        if "Task fidelity:" in line:
            score = extract_fidelity_score(line)
            if score and score >= min_score:
                high_fidelity_lines.append((i, score))

    print(f"Found {len(high_fidelity_lines)} high fidelity entries (>= {min_score})")

    # Extract details for each high fidelity entry
    processed_tasks = set()  # Avoid duplicates

    for line_idx, fidelity_score in high_fidelity_lines:
        # Extract optimization details
        entry = extract_optimization_details(lines, line_idx)

        # Only add if we have the essential data and haven't processed this task before
        if (entry['original_prompt'] and
            entry['task_id'] and
            entry['task_id'] not in processed_tasks):

            processed_tasks.add(entry['task_id'])
            high_fidelity_prompts.append(entry)

            print(f"✅ Extracted: {entry['original_prompt'][:50]}... (Score: {entry['fidelity_score']})")

    # Save results
    if high_fidelity_prompts:
        output_file = f"high_fidelity_prompts_{min_score}.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(high_fidelity_prompts, f, indent=2, ensure_ascii=False)

        print(f"\n💾 Saved {len(high_fidelity_prompts)} high fidelity prompts to {output_file}")

        # Print summary statistics
        print("\n📊 SUMMARY STATISTICS:")
        print(f"Total high fidelity prompts: {len(high_fidelity_prompts)}")

        if high_fidelity_prompts:
            avg_fidelity = sum(p['fidelity_score'] for p in high_fidelity_prompts) / len(high_fidelity_prompts)
            print(f"Average fidelity: {avg_fidelity:.4f}")

            # Check if we have CLIP similarity data
            clip_scores = [p for p in high_fidelity_prompts if p['clip_similarity'] > 0]
            if clip_scores:
                avg_clip = sum(p['clip_similarity'] for p in clip_scores) / len(clip_scores)
                print(f"Average CLIP similarity: {avg_clip:.4f}")
            else:
                print("CLIP similarity: No data (likely unoptimized prompts)")
            # Optimization statistics
            optimized_count = sum(1 for p in high_fidelity_prompts if p['optimization_applied'])
            print(f"Optimized prompts: {optimized_count}/{len(high_fidelity_prompts)} ({optimized_count/len(high_fidelity_prompts)*100:.1f}%)")

            # Validator distribution
            validators = defaultdict(int)
            for entry in high_fidelity_prompts:
                if entry['validator_uid']:
                    validators[entry['validator_uid']] += 1

            print(f"Validators with high scores: {dict(validators)}")

            # LoRA distribution
            loras = defaultdict(int)
            for entry in high_fidelity_prompts:
                if entry['lora_used']:
                    loras[entry['lora_used']] += 1

            print(f"LoRA usage: {dict(loras)}")

    else:
        print("\n❌ No high fidelity prompts could be extracted")

    return high_fidelity_prompts


if __name__ == "__main__":
    main()
