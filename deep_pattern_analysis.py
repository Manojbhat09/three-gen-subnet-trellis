#!/usr/bin/env python3

import re
from collections import Counter, defaultdict

# Read the log file
with open('continuous_trellis.log', 'r') as f:
    content = f.read()

# Extract ALL prompts with their scores
matches = re.findall(r'Processing task [^:]*: \'([^\']+)\'.*?Task fidelity: ([\d.]+)', content, re.DOTALL)

# Convert to list of tuples
all_prompts = [(prompt, float(score)) for prompt, score in matches]

# Sort by score
all_prompts.sort(key=lambda x: x[1], reverse=True)

# Get statistics
print(f'Total prompts analyzed: {len(all_prompts)}')
print(f'Score range: {all_prompts[-1][1]:.4f} - {all_prompts[0][1]:.4f}')
print()

# Get top 50 prompts
print('=== TOP 50 HIGH-PERFORMING PROMPTS ===')
for i, (prompt, score) in enumerate(all_prompts[:50], 1):
    print(f'{i:2d}. {score:.4f}: {prompt}')

# Analyze patterns in different score ranges
score_ranges = {
    '0.95+': [(p, s) for p, s in all_prompts if s >= 0.95],
    '0.93-0.949': [(p, s) for p, s in all_prompts if 0.93 <= s < 0.95],
    '0.90-0.929': [(p, s) for p, s in all_prompts if 0.90 <= s < 0.93],
    '0.85-0.899': [(p, s) for p, s in all_prompts if 0.85 <= s < 0.90],
    '0.80-0.849': [(p, s) for p, s in all_prompts if 0.80 <= s < 0.85]
}

print('\n=== SCORE DISTRIBUTION ===')
for range_name, prompts in score_ranges.items():
    print(f'{range_name}: {len(prompts)} prompts ({len(prompts)/len(all_prompts)*100:.1f}%)')

# Deep pattern analysis
print('\n=== DEEP PATTERN ANALYSIS ===')

# Analyze specific patterns in ultra-high performers (0.95+)
ultra_high = score_ranges['0.95+']
print(f'\nUltra-High Performers (0.95+): {len(ultra_high)} prompts')

# Extract common patterns
patterns = {
    'material_words': ['metal', 'wooden', 'glass', 'ceramic', 'steel', 'brass', 'copper', 'silver', 'gold', 'plastic', 'leather'],
    'color_words': ['red', 'blue', 'green', 'yellow', 'black', 'white', 'purple', 'orange', 'cyan', 'pink', 'brown', 'gray'],
    'quality_words': ['smooth', 'polished', 'glossy', 'matte', 'textured', 'shiny', 'rough', 'detailed'],
    'size_words': ['small', 'large', 'tiny', 'big', 'compact', 'miniature', 'massive'],
    'functional_words': ['handle', 'grip', 'head', 'tip', 'blade', 'edge', 'base', 'top'],
    'descriptive_words': ['elegant', 'simple', 'complex', 'ornate', 'modern', 'classic', 'vintage']
}

# Analyze pattern frequency in different score ranges
print('\n=== PATTERN FREQUENCY BY SCORE RANGE ===')
for range_name, prompts in score_ranges.items():
    if not prompts:
        continue
    print(f'\n{range_name} ({len(prompts)} prompts):')
    
    pattern_counts = defaultdict(int)
    for prompt, _ in prompts:
        prompt_lower = prompt.lower()
        for pattern_type, words in patterns.items():
            for word in words:
                if word in prompt_lower:
                    pattern_counts[pattern_type] += 1
                    break
    
    for pattern_type, count in sorted(pattern_counts.items(), key=lambda x: x[1], reverse=True):
        percentage = count / len(prompts) * 100
        print(f'  {pattern_type}: {count} ({percentage:.1f}%)')

# Find unique patterns in ultra-high performers
print('\n=== UNIQUE PATTERNS IN ULTRA-HIGH PERFORMERS ===')
ultra_high_words = []
other_words = []

for prompt, score in all_prompts:
    words = prompt.lower().split()
    if score >= 0.95:
        ultra_high_words.extend(words)
    else:
        other_words.extend(words)

ultra_high_counter = Counter(ultra_high_words)
other_counter = Counter(other_words)

# Find words that appear disproportionately in ultra-high performers
print('\nWords enriched in ultra-high performers:')
for word, ultra_count in ultra_high_counter.most_common(50):
    if len(word) <= 2:
        continue
    other_count = other_counter.get(word, 0)
    if ultra_count >= 2 and other_count > 0:
        ratio = (ultra_count / len(ultra_high)) / (other_count / (len(all_prompts) - len(ultra_high)))
        if ratio > 2.0:  # Word appears 2x more frequently in ultra-high
            print(f'  {word}: {ratio:.2f}x more frequent (appears {ultra_count} times)')

# Analyze prompt structure
print('\n=== PROMPT STRUCTURE ANALYSIS ===')
structure_patterns = {
    'simple_noun': 0,
    'adjective_noun': 0,
    'material_object': 0,
    'object_with_detail': 0,
    'complex_description': 0
}

for prompt, score in all_prompts[:50]:
    words = prompt.lower().split()
    if len(words) == 1:
        structure_patterns['simple_noun'] += 1
    elif len(words) == 2:
        structure_patterns['adjective_noun'] += 1
    elif 'with' in prompt:
        structure_patterns['object_with_detail'] += 1
    elif any(mat in prompt.lower() for mat in patterns['material_words']):
        structure_patterns['material_object'] += 1
    else:
        structure_patterns['complex_description'] += 1

print('\nTop 50 prompt structures:')
for structure, count in sorted(structure_patterns.items(), key=lambda x: x[1], reverse=True):
    print(f'  {structure}: {count} prompts')

# Extract specific high-performing patterns
print('\n=== SPECIFIC HIGH-PERFORMING PATTERNS ===')
specific_patterns = defaultdict(list)

for prompt, score in all_prompts[:30]:
    prompt_lower = prompt.lower()
    
    # Tools with specific features
    if any(tool in prompt_lower for tool in ['hammer', 'drill', 'wrench', 'screwdriver']):
        specific_patterns['tools_with_features'].append((score, prompt))
    
    # Robots with descriptors
    if 'robot' in prompt_lower:
        specific_patterns['robot_variants'].append((score, prompt))
    
    # Objects with "wand"
    if 'wand' in prompt_lower:
        specific_patterns['wand_variants'].append((score, prompt))
    
    # Sports equipment
    if any(sport in prompt_lower for sport in ['bat', 'ball', 'racket', 'club', 'stick']):
        specific_patterns['sports_equipment'].append((score, prompt))
    
    # Furniture
    if any(furn in prompt_lower for furn in ['chair', 'table', 'desk', 'stool', 'bench']):
        specific_patterns['furniture'].append((score, prompt))

for pattern_type, examples in specific_patterns.items():
    if examples:
        print(f'\n{pattern_type.upper()}:')
        for score, prompt in examples:
            print(f'  {score:.4f}: {prompt}') 