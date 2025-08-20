#!/usr/bin/env python3
import os, glob, re, json

# 1) Unique original prompts from ALL logs
log_paths = sorted(glob.glob('episodic_logs_first/episodic_run_*.log'))
unique_prompts = set()
for p in log_paths:
    try:
        with open(p, 'r', encoding='utf-8') as f:
            for line in f:
                if 'Original:' in line:
                    m = re.search(r'Original:\s*(.+)$', line)
                    if m:
                        s = m.group(1).strip()
                        # normalize outer quotes if present
                        if (s.startswith("''") and s.endswith("''")) or (s.startswith('""') and s.endswith('""')):
                            s = s[2:-2]
                        if (s.startswith("'") and s.endswith("'")) or (s.startswith('"') and s.endswith('"')):
                            s = s[1:-1]
                        unique_prompts.add(s)
    except Exception:
        pass

# 2) Prompts in episodic_memory.json
mem_file = 'episodic_logs_first/episodic_memory.json'
mem_prompts = set()
if os.path.exists(mem_file):
    try:
        with open(mem_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        if isinstance(data, dict) and 'gold_standard_results' in data and isinstance(data['gold_standard_results'], dict):
            mem_prompts = set(data['gold_standard_results'].keys())
        elif isinstance(data, dict):
            for k, v in data.items():
                if isinstance(v, dict) and ('method_2_hybrid_example' in v or 'validation_results' in v or 'optimized_prompt' in v):
                    mem_prompts.add(k)
    except Exception:
        pass

combined = set(unique_prompts) | set(mem_prompts)

print(f"Unique prompts in logs: {len(unique_prompts)}")
print(f"Prompts in episodic_memory.json: {len(mem_prompts)}")
print(f"Combined unique prompts (logs + memory): {len(combined)}")
print(f"Intersection (in both): {len(unique_prompts & mem_prompts)}")

# Show a few examples for sanity
def sample(s):
    return list(sorted(s))[:5]
print("\nExamples from logs:", sample(unique_prompts))
print("Examples from memory:", sample(mem_prompts))
