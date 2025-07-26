#!/usr/bin/env python3
"""
Extract Zero Fidelity Prompts from Log File
===========================================
This script extracts prompts from continuous_trellis.log that received 0.0000 task fidelity.
"""

import re
from typing import List, Dict, Set
from datetime import datetime

def extract_zero_fidelity_from_log(log_file: str = "continuous_trellis.log") -> Dict:
    """
    Extract prompts from log file entries that show 0.0000 task fidelity.
    
    Args:
        log_file: Path to the log file
        
    Returns:
        Dictionary with extracted prompts and metadata
    """
    
    print(f"🔍 Reading log file: {log_file}")
    
    try:
        with open(log_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        print(f"📖 Total lines in log: {len(lines)}")
        
        zero_fidelity_prompts = []
        zero_fidelity_entries = []
        
        # Find lines with "Task fidelity: 0.0000"
        for i, line in enumerate(lines):
            if "Task fidelity: 0.0000" in line:
                # Extract timestamp from this line
                timestamp_match = re.search(r'(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})', line)
                timestamp = timestamp_match.group(1) if timestamp_match else "Unknown"
                
                # Look backwards to find the prompt
                prompt = None
                task_id = None
                validator_uid = None
                
                # Search backwards up to 20 lines to find the prompt
                for j in range(max(0, i-20), i):
                    check_line = lines[j]
                    
                    # Look for the main pattern: "Processing task [task-id]: 'prompt'"
                    processing_match = re.search(r"Processing task ([a-f0-9-]+): '([^']+)'", check_line)
                    if processing_match:
                        task_id = processing_match.group(1)
                        prompt = processing_match.group(2)
                        break
                    
                    # Alternative pattern: "Generating 3D model: 'prompt'"
                    generating_match = re.search(r"Generating 3D model: '([^']+)'", check_line)
                    if generating_match:
                        prompt = generating_match.group(1)
                        break
                
                # Look for validator UID in surrounding lines
                for j in range(max(0, i-10), min(len(lines), i+5)):
                    check_line = lines[j]
                    if "UID" in check_line and "Submission successful" in check_line:
                        uid_match = re.search(r'UID (\d+)', check_line)
                        if uid_match:
                            validator_uid = int(uid_match.group(1))
                            break
                
                # If we found a prompt, save it
                if prompt:
                    zero_fidelity_prompts.append(prompt)
                    zero_fidelity_entries.append({
                        "prompt": prompt,
                        "timestamp": timestamp,
                        "task_id": task_id,
                        "validator_uid": validator_uid,
                        "log_line": i + 1
                    })
                    print(f"Found: {prompt}")
                else:
                    print(f"Warning: No prompt found for zero fidelity at line {i+1}")
        
        # Remove duplicates while preserving order
        unique_prompts = []
        seen = set()
        for prompt in zero_fidelity_prompts:
            if prompt not in seen:
                unique_prompts.append(prompt)
                seen.add(prompt)
        
        print(f"\n✅ Extraction complete!")
        print(f"📊 Total zero fidelity entries: {len(zero_fidelity_prompts)}")
        print(f"📝 Unique prompts: {len(unique_prompts)}")
        
        return {
            "prompts": unique_prompts,
            "all_entries": zero_fidelity_entries,
            "metadata": {
                "total_entries": len(zero_fidelity_prompts),
                "unique_prompts": len(unique_prompts),
                "extraction_date": datetime.now().isoformat(),
                "log_file": log_file
            }
        }
        
    except FileNotFoundError:
        print(f"❌ Log file not found: {log_file}")
        return {"prompts": [], "metadata": {"error": "File not found"}}
    except Exception as e:
        print(f"❌ Error processing log file: {e}")
        return {"prompts": [], "metadata": {"error": str(e)}}

def save_log_prompts(data: Dict, filename: str = "log_zero_fidelity_prompts.txt"):
    """Save prompts from log as Python list"""
    
    prompts = data["prompts"]
    print(f"💾 Saving {len(prompts)} log prompts to {filename}")
    
    with open(filename, 'w', encoding='utf-8') as f:
        f.write("# Zero Task Fidelity Prompts from Log File\n")
        f.write(f"# Extracted on: {datetime.now().isoformat()}\n")
        f.write(f"# Total prompts: {len(prompts)}\n\n")
        
        f.write("log_zero_fidelity_prompts = [\n")
        for i, prompt in enumerate(prompts):
            escaped_prompt = prompt.replace('"', '\\"').replace('\n', '\\n').replace('\r', '\\r')
            if i == len(prompts) - 1:
                f.write(f'    "{escaped_prompt}"\n')
            else:
                f.write(f'    "{escaped_prompt}",\n')
        f.write("]\n")
    
    print(f"✅ Saved to {filename}")

def load_db_prompts():
    """Load prompts from the database file"""
    try:
        with open("zero_fidelity_prompts.txt", 'r') as f:
            content = f.read()
        
        # Extract the list using regex
        match = re.search(r'zero_fidelity_prompts = \[(.*?)\]', content, re.DOTALL)
        if match:
            list_content = match.group(1)
            # Extract individual prompts
            prompts = re.findall(r'"([^"]+)"', list_content)
            return set(prompts)
        else:
            print("❌ Could not parse database prompts file")
            return set()
    except Exception as e:
        print(f"❌ Error loading database prompts: {e}")
        return set()

def compare_db_vs_log_prompts():
    """Compare prompts from database vs log file"""
    
    print("\n🔍 COMPARING DATABASE VS LOG PROMPTS")
    print("=" * 60)
    
    # Load database prompts
    db_prompts = load_db_prompts()
    if not db_prompts:
        print("❌ Could not load database prompts")
        return
    
    # Extract log prompts
    log_data = extract_zero_fidelity_from_log()
    log_prompts = set(log_data["prompts"])
    
    # Compare
    print(f"📊 Database prompts: {len(db_prompts)}")
    print(f"📊 Log prompts: {len(log_prompts)}")
    
    # Find differences
    only_in_db = db_prompts - log_prompts
    only_in_log = log_prompts - db_prompts
    common = db_prompts & log_prompts
    
    print(f"\n🔄 COMPARISON RESULTS:")
    print(f"   Common prompts: {len(common)}")
    print(f"   Only in database: {len(only_in_db)}")
    print(f"   Only in log: {len(only_in_log)}")
    
    overlap_percentage = (len(common) / len(db_prompts)) * 100 if db_prompts else 0
    print(f"   Overlap percentage: {overlap_percentage:.1f}%")
    
    if only_in_db:
        print(f"\n📋 ONLY IN DATABASE ({len(only_in_db)}):")
        for i, prompt in enumerate(sorted(only_in_db)[:10]):
            print(f"   {i+1}. {prompt}")
        if len(only_in_db) > 10:
            print(f"   ... and {len(only_in_db) - 10} more")
    
    if only_in_log:
        print(f"\n📋 ONLY IN LOG ({len(only_in_log)}):")
        for i, prompt in enumerate(sorted(only_in_log)[:10]):
            print(f"   {i+1}. {prompt}")
        if len(only_in_log) > 10:
            print(f"   ... and {len(only_in_log) - 10} more")
    
    # Show some common prompts
    if common:
        print(f"\n✅ COMMON PROMPTS (first 10 of {len(common)}):")
        for i, prompt in enumerate(sorted(common)[:10]):
            print(f"   {i+1}. {prompt}")
        if len(common) > 10:
            print(f"   ... and {len(common) - 10} more")
    
    # Save log prompts
    save_log_prompts(log_data)
    
    # Save comparison results
    comparison_results = {
        "database_prompts_count": len(db_prompts),
        "log_prompts_count": len(log_prompts),
        "common_prompts_count": len(common),
        "only_in_database_count": len(only_in_db),
        "only_in_log_count": len(only_in_log),
        "overlap_percentage": overlap_percentage,
        "common_prompts": list(common),
        "only_in_database": list(only_in_db),
        "only_in_log": list(only_in_log),
        "comparison_date": datetime.now().isoformat()
    }
    
    import json
    with open("db_vs_log_comparison.json", 'w') as f:
        json.dump(comparison_results, f, indent=2)
    
    print(f"\n💾 Saved detailed comparison to db_vs_log_comparison.json")
    
    return comparison_results

def main():
    """Main execution"""
    print("🎯 LOG VS DATABASE ZERO FIDELITY COMPARISON")
    print("=" * 60)
    
    compare_db_vs_log_prompts()

if __name__ == "__main__":
    main() 