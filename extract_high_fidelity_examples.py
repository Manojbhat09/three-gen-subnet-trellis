#!/usr/bin/env python3
"""
Extract specific high-fidelity examples with their optimization details for analysis.
"""

import re
import json
from typing import List, Dict, Optional


def extract_optimization_details_around_line(lines: List[str], fidelity_line_idx: int) -> Dict:
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
        'validator_uid': ''
    }

    # Extract timestamp and fidelity score
    fidelity_line = lines[fidelity_line_idx].strip()
    timestamp_match = re.search(r'(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3})', fidelity_line)
    fidelity_match = re.search(r'Task fidelity: ([0-9.]+)', fidelity_line)

    if timestamp_match:
        entry['timestamp'] = timestamp_match.group(1)
    if fidelity_match:
        entry['fidelity_score'] = float(fidelity_match.group(1))

    # Search backwards for optimization details (up to 150 lines back)
    search_start = max(0, fidelity_line_idx - 150)
    search_end = fidelity_line_idx

    for i in range(search_end - 1, search_start - 1, -1):
        line = lines[i]

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
        if "Optimized:" in line and "FINAL OPTIMIZATION RESULT" not in lines[i-1] and not entry['optimized_prompt']:
            # This is likely the optimized prompt line
            opt_match = re.search(r"Optimized: '([^']*)'", line)
            if opt_match:
                entry['optimized_prompt'] = opt_match.group(1)

        # Extract cleaned prompt
        if "Cleaned:" in line and not entry['cleaned_prompt']:
            clean_match = re.search(r"Cleaned: '([^']*)'", line)
            if clean_match:
                entry['cleaned_prompt'] = clean_match.group(1)

    return entry


def main():
    log_file = "/home/mbhat/three-gen-subnet-trellis/continuous_trellis.log"

    # High fidelity scores to analyze (from the grep results)
    target_scores = [
        (0.9023, "14:39:02,075"),
        (0.8834, "05:01:27,311"),  # Dragon scale that worked
        (0.8491, "14:25:57,382"),
        (0.8388, "14:20:25,794"),
        (0.8219, "14:32:38,934")
    ]

    print("🔍 Extracting High-Fidelity Prompt Optimization Examples")
    print("=" * 70)

    results = []

    with open(log_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    for score, timestamp_pattern in target_scores:
        print(f"\n🎯 Looking for fidelity score: {score}")

        # Find the line with this score
        target_line_idx = None
        for i, line in enumerate(lines):
            if f"Task fidelity: {score}" in line and timestamp_pattern in line:
                target_line_idx = i
                break

        if target_line_idx is None:
            print(f"   ❌ Could not find exact line for score {score}")
            continue

        # Extract optimization details
        entry = extract_optimization_details_around_line(lines, target_line_idx)

        if entry['original_prompt']:
            print(f"   ✅ Found complete optimization details")
            print(f"   📝 Original: {entry['original_prompt'][:60]}...")
            print(".4f"            print(f"   🎨 LoRA: {entry['lora_used']}")
            print(f"   📏 Optimized length: {len(entry['optimized_prompt'])} chars")

            results.append(entry)
        else:
            print(f"   ⚠️  Missing optimization details for score {score}")

    # Save results
    if results:
        output_file = "high_fidelity_examples.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

        print(f"\n💾 Saved {len(results)} examples to {output_file}")

        # Print summary
        print("
📊 SUMMARY:"        print(f"Total examples extracted: {len(results)}")
        if results:
            avg_fidelity = sum(r['fidelity_score'] for r in results) / len(results)
            avg_clip = sum(r['clip_similarity'] for r in results if r['clip_similarity'] > 0) / len([r for r in results if r['clip_similarity'] > 0])
            print(".4f"            print(".4f"
            # Show original prompt lengths
            lengths = [len(r['original_prompt'].split()) for r in results]
            print(f"Original prompt lengths: {lengths}")

    else:
        print("\n❌ No examples could be extracted")


if __name__ == "__main__":
    main()
