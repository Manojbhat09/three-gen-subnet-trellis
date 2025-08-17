#!/usr/bin/env python3
"""
Script to remove duplicates from episodic_test_prompts.py and create a cleaned version
"""

import re
from collections import OrderedDict

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

def create_cleaned_file(input_filename, output_filename):
    """Create a cleaned version of the file with duplicates removed."""
    
    # Read the original file
    with open(input_filename, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Extract prompts
    prompts = extract_prompts_from_file(input_filename)
    
    # Remove duplicates while preserving order
    seen = OrderedDict()
    for prompt in prompts:
        if prompt not in seen:
            seen[prompt] = True
    
    unique_prompts = list(seen.keys())
    
    print(f"Original prompts: {len(prompts)}")
    print(f"Unique prompts: {len(unique_prompts)}")
    print(f"Duplicates removed: {len(prompts) - len(unique_prompts)}")
    
    # Create the new content
    # Find the start and end of the list
    start_match = re.search(r'(EPISODIC_TEST_PROMPTS\s*=\s*\[)', content)
    if not start_match:
        print("Could not find EPISODIC_TEST_PROMPTS list")
        return False
    
    start_pos = start_match.start()
    
    # Find the end of the list
    bracket_count = 0
    end_pos = start_match.end() - 1  # Start from the opening bracket
    
    for i, char in enumerate(content[start_match.end():], start_match.end()):
        if char == '[':
            bracket_count += 1
        elif char == ']':
            bracket_count -= 1
            if bracket_count == 0:
                end_pos = i
                break
    
    # Reconstruct the file content
    before_list = content[:start_match.end()]
    after_list = content[end_pos:]
    
    # Create the new list content
    new_list_content = '\n    '
    for i, prompt in enumerate(unique_prompts):
        if i > 0:
            new_list_content += ',\n    '
        new_list_content += f'"{prompt}"'
    
    new_list_content += '\n'
    
    # Combine all parts
    new_content = before_list + new_list_content + after_list
    
    # Write the cleaned file
    with open(output_filename, 'w', encoding='utf-8') as f:
        f.write(new_content)
    
    print(f"Cleaned file saved as: {output_filename}")
    return True

def main():
    """Main function to remove duplicates."""
    input_filename = "episodic_test_prompts.py"
    output_filename = "episodic_test_prompts_cleaned.py"
    
    print("Removing duplicates from episodic_test_prompts.py")
    print("=" * 60)
    
    try:
        success = create_cleaned_file(input_filename, output_filename)
        if success:
            print("\nDuplicate removal completed successfully!")
            print(f"Original file: {input_filename}")
            print(f"Cleaned file: {output_filename}")
            
            # Show a preview of the cleaned file
            print("\nPreview of cleaned file (first 10 unique prompts):")
            print("-" * 60)
            with open(output_filename, 'r', encoding='utf-8') as f:
                lines = f.readlines()
                for i, line in enumerate(lines):
                    if 'EPISODIC_TEST_PROMPTS' in line:
                        start_idx = i
                        break
                
                # Show the list structure
                for i in range(start_idx, min(start_idx + 15, len(lines))):
                    print(lines[i].rstrip())
                if len(lines) > start_idx + 15:
                    print("    ...")
                    print("]")
        else:
            print("Failed to create cleaned file.")
            
    except FileNotFoundError:
        print(f"Error: File '{input_filename}' not found.")
    except Exception as e:
        print(f"Error processing file: {e}")

if __name__ == "__main__":
    main()
