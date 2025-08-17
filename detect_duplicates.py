#!/usr/bin/env python3
"""
Script to detect duplicates in episodic_test_prompts.py
"""

import re
from collections import Counter

def extract_prompts_from_file(filename):
    """Extract prompts from the Python file, handling the syntax issues."""
    with open(filename, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Find the EPISODIC_TEST_PROMPTS list
    start_match = re.search(r'EPISODIC_TEST_PROMPTS\s*=\s*\[', content)
    if not start_match:
        print("Could not find EPISODIC_TEST_PROMPTS list")
        return []
    
    # Extract everything between the brackets
    start_pos = start_match.end() - 1  # Include the opening bracket
    bracket_count = 0
    end_pos = start_pos
    
    for i, char in enumerate(content[start_pos:], start_pos):
        if char == '[':
            bracket_count += 1
        elif char == ']':
            bracket_count -= 1
            if bracket_count == 0:
                end_pos = i
                break
    
    list_content = content[start_pos:end_pos + 1]
    
    # Extract individual prompts using regex
    # This handles both quoted strings and commented lines
    prompts = []
    
    # Split by lines and process each line
    lines = list_content.split('\n')
    for line in lines:
        line = line.strip()
        
        # Skip empty lines and comments
        if not line or line.startswith('#'):
            continue
        
        # Look for quoted strings
        matches = re.findall(r'"([^"]*)"', line)
        for match in matches:
            if match.strip():  # Only add non-empty prompts
                prompts.append(match.strip())
    
    return prompts

def analyze_duplicates(prompts):
    """Analyze the list for duplicates and provide detailed statistics."""
    print(f"Total prompts found: {len(prompts)}")
    print(f"Unique prompts: {len(set(prompts))}")
    print(f"Duplicate prompts: {len(prompts) - len(set(prompts))}")
    
    # Count occurrences of each prompt
    prompt_counts = Counter(prompts)
    
    # Find duplicates (prompts that appear more than once)
    duplicates = {prompt: count for prompt, count in prompt_counts.items() if count > 1}
    
    if duplicates:
        print(f"\nFound {len(duplicates)} unique prompts that are duplicated:")
        print("-" * 80)
        
        # Sort by count (most duplicated first)
        sorted_duplicates = sorted(duplicates.items(), key=lambda x: x[1], reverse=True)
        
        for prompt, count in sorted_duplicates:
            print(f"'{prompt}' appears {count} times")
        
        print("-" * 80)
        
        # Show total duplicate entries
        total_duplicate_entries = sum(count - 1 for count in duplicates.values())
        print(f"Total duplicate entries: {total_duplicate_entries}")
        
        # Show which prompts appear most frequently
        print(f"\nMost duplicated prompts:")
        for prompt, count in sorted_duplicates[:10]:  # Top 10
            print(f"  {count}x: {prompt}")
            
    else:
        print("\nNo duplicates found!")
    
    return duplicates

def main():
    """Main function to run the duplicate detection."""
    filename = "episodic_test_prompts.py"
    
    print("Analyzing duplicates in episodic_test_prompts.py")
    print("=" * 60)
    
    try:
        prompts = extract_prompts_from_file(filename)
        if prompts:
            analyze_duplicates(prompts)
        else:
            print("No prompts found in the file.")
            
    except FileNotFoundError:
        print(f"Error: File '{filename}' not found.")
    except Exception as e:
        print(f"Error analyzing file: {e}")

if __name__ == "__main__":
    main()
