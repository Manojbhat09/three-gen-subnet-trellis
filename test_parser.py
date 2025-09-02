#!/usr/bin/env python3
"""
Test script to verify log parsing
"""

import re

def test_parse_log_file(file_path):
    """Test parsing of log file"""
    print(f"Testing parsing of: {file_path}")
    
    with open(file_path, 'r') as f:
        content = f.read()
    
    # Look for the pattern that indicates a new prompt generation
    sections = re.split(r'    Original: \'', content)
    print(f"Found {len(sections)} sections")
    
    data = []
    for i, section in enumerate(sections[1:6], 1):  # Test first 5 sections
        print(f"\n--- Section {i} ---")
        prompt_data = {}
        
        # Extract original prompt (it's at the start of the section)
        original_match = re.search(r"^([^']+)'", section)
        if original_match:
            prompt_data['original_prompt'] = original_match.group(1)
            print(f"Original: {prompt_data['original_prompt'][:50]}...")
        
        # Extract optimized prompt (if exists)
        optimized_match = re.search(r"Optimized: '([^']+)'", section)
        if optimized_match:
            prompt_data['optimized_prompt'] = optimized_match.group(1)
            print(f"Optimized: {prompt_data['optimized_prompt'][:50]}...")
        
        # Extract scores
        validation_match = re.search(r'🏆 Validation Engine Score: ([\d.]+)', section)
        if validation_match:
            prompt_data['validation_engine_score'] = float(validation_match.group(1))
            print(f"Validation Score: {prompt_data['validation_engine_score']}")
        
        alignment_match = re.search(r'🤝 Alignment Score: ([\d.]+)', section)
        if alignment_match:
            prompt_data['alignment_score'] = float(alignment_match.group(1))
            print(f"Alignment Score: {prompt_data['alignment_score']}")
        
        quality_match = re.search(r'💎 Quality Score: ([\d.]+)', section)
        if quality_match:
            prompt_data['quality_score'] = float(quality_match.group(1))
            print(f"Quality Score: {prompt_data['quality_score']}")
        
        # Only add if we have all required data
        if all(key in prompt_data for key in ['original_prompt', 'validation_engine_score', 'alignment_score', 'quality_score']):
            data.append(prompt_data)
            print("✅ Complete data extracted")
        else:
            print("❌ Missing data")
    
    print(f"\nSuccessfully parsed {len(data)} complete entries from first 5 sections")
    return data

if __name__ == "__main__":
    # Test both files
    log1_path = '/home/mbhat/three-gen-subnet-trellis/continuous_trellis_simulator.log.backup_20250902_022647'
    log2_path = '/home/mbhat/three-gen-subnet-trellis/continuous_trellis_simulator.log'
    
    print("="*60)
    print("TESTING LOG PARSING")
    print("="*60)
    
    print("\nTesting Log 1 (Simple optimization):")
    log1_data = test_parse_log_file(log1_path)
    
    print("\nTesting Log 2 (vLLM + CLIP optimization):")
    log2_data = test_parse_log_file(log2_path)
