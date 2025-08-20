#!/usr/bin/env python3
"""
Script to clean up episodic_test_prompts.py by removing duplicates and organizing prompts.
"""

import re
import os

def clean_prompts_file(file_path="episodic_test_prompts.py"):
    """Clean up the prompts file by removing duplicates and organizing prompts."""
    
    print(f"🔍 Reading prompts from {file_path}...")
    
    try:
        with open(file_path, 'r') as f:
            content = f.read()
    except FileNotFoundError:
        print(f"❌ File {file_path} not found!")
        return
    
    # Extract prompts using regex
    prompt_pattern = r'"([^"]+)"'
    prompts = re.findall(prompt_pattern, content)
    
    print(f"📊 Found {len(prompts)} prompts in file")
    
    # Clean and deduplicate prompts
    cleaned_prompts = []
    seen_prompts = set()
    duplicates_removed = 0
    
    for prompt in prompts:
        # Clean the prompt
        cleaned = prompt.strip()
        
        # Skip empty prompts
        if not cleaned:
            continue
            
        # Check for duplicates (case-insensitive)
        cleaned_lower = cleaned.lower()
        if cleaned_lower not in seen_prompts:
            cleaned_prompts.append(cleaned)
            seen_prompts.add(cleaned_lower)
        else:
            duplicates_removed += 1
    
    print(f"🧹 Removed {duplicates_removed} duplicate prompts")
    print(f"✅ Kept {len(cleaned_prompts)} unique prompts")
    
    # Sort prompts alphabetically for better organization
    cleaned_prompts.sort(key=str.lower)
    
    # Create backup
    backup_path = f"{file_path}.backup"
    try:
        with open(backup_path, 'w') as f:
            f.write(content)
        print(f"💾 Created backup at {backup_path}")
    except Exception as e:
        print(f"⚠️ Could not create backup: {e}")
    
    # Generate new content
    new_content = """# Episodic Test Prompts - Cleaned and Deduplicated
# This file contains unique prompts for episodic optimization testing
# Generated automatically - do not edit manually

EPISODIC_TEST_PROMPTS = [
"""
    
    # Add prompts with proper formatting
    for i, prompt in enumerate(cleaned_prompts):
        if i == len(cleaned_prompts) - 1:
            # Last prompt - no comma
            new_content += f'    "{prompt}"\n'
        else:
            new_content += f'    "{prompt}",\n'
    
    new_content += "]\n"
    
    # Write the cleaned file
    try:
        with open(file_path, 'w') as f:
            f.write(new_content)
        print(f"✅ Successfully cleaned {file_path}")
        
        # Show some statistics
        print(f"\n📈 Final Statistics:")
        print(f"   Original prompts: {len(prompts)}")
        print(f"   Duplicates removed: {duplicates_removed}")
        print(f"   Final unique prompts: {len(cleaned_prompts)}")
        print(f"   File size reduction: {len(content) - len(new_content)} characters")
        
    except Exception as e:
        print(f"❌ Error writing cleaned file: {e}")
        return False
    
    return True

def verify_cleanup(file_path="episodic_test_prompts.py"):
    """Verify that the cleanup was successful."""
    
    print(f"\n🔍 Verifying cleanup...")
    
    try:
        with open(file_path, 'r') as f:
            content = f.read()
        
        # Check if the file has the expected structure
        if 'EPISODIC_TEST_PROMPTS = [' not in content:
            print("❌ File structure is incorrect")
            return False
        
        # Extract prompts again to verify
        prompt_pattern = r'"([^"]+)"'
        prompts = re.findall(prompt_pattern, content)
        
        # Check for duplicates
        seen = set()
        duplicates = 0
        for prompt in prompts:
            if prompt.lower() in seen:
                duplicates += 1
            else:
                seen.add(prompt.lower())
        
        if duplicates == 0:
            print("✅ Verification successful - no duplicates found")
            print(f"   Total unique prompts: {len(prompts)}")
            return True
        else:
            print(f"❌ Verification failed - found {duplicates} duplicates")
            return False
            
    except Exception as e:
        print(f"❌ Verification error: {e}")
        return False

if __name__ == "__main__":
    print("🧹 Episodic Test Prompts Cleanup Tool")
    print("=" * 50)
    
    # Clean the file
    if clean_prompts_file():
        # Verify the cleanup
        verify_cleanup()
    else:
        print("❌ Cleanup failed!")
    
    print("\n✨ Cleanup process completed!")
