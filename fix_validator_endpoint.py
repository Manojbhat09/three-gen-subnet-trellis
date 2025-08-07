#!/usr/bin/env python3
"""
Fix Validator Endpoint
Purpose: Fix the endpoint URL in subnet_accurate_validator.py
"""

import re

def fix_validator_endpoint():
    """Fix the endpoint URL in the validator script"""
    print("🔧 Fixing validator endpoint...")
    
    # Read the file
    with open('subnet_accurate_validator.py', 'r') as f:
        content = f.read()
    
    # Replace the endpoint
    old_url = "http://127.0.0.1:8096/generate_clip_optimized/"
    new_url = "http://127.0.0.1:8096/generate/"
    
    if old_url in content:
        content = content.replace(old_url, new_url)
        print(f"✅ Replaced '{old_url}' with '{new_url}'")
        
        # Write back
        with open('subnet_accurate_validator.py', 'w') as f:
            f.write(content)
        
        print("✅ Validator endpoint fixed!")
    else:
        print("❌ Endpoint not found in file")

if __name__ == "__main__":
    fix_validator_endpoint() 