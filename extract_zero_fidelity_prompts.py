#!/usr/bin/env python3
"""
Extract Zero Task Fidelity Prompts
==================================
This script extracts all prompts that received 0.0 task fidelity scores
from the continuous trellis tasks database and saves them as a Python list.
"""

import sqlite3
import json
from typing import List, Dict, Any
from datetime import datetime

def extract_zero_fidelity_prompts(db_path: str = "continuous_trellis_tasks.db") -> Dict[str, Any]:
    """
    Extract all prompts with 0.0 task fidelity scores from the database.
    
    Args:
        db_path: Path to the SQLite database file
        
    Returns:
        Dictionary containing the extracted data and metadata
    """
    
    print(f"🔍 Connecting to database: {db_path}")
    
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Query for all tasks with 0.0 task fidelity score
        query = """
        SELECT 
            prompt,
            task_id,
            validator_uid,
            pulled_at,
            submitted_at,
            task_fidelity_score,
            average_fidelity_score,
            validation_failed,
            generation_time,
            validation_time,
            local_validation_score
        FROM tasks 
        WHERE task_fidelity_score = 0.0 
        AND feedback_received = 1
        ORDER BY pulled_at DESC
        """
        
        print("📊 Executing query to find zero fidelity tasks...")
        cursor.execute(query)
        rows = cursor.fetchall()
        
        if not rows:
            print("❌ No tasks with 0.0 task fidelity found!")
            return {
                "prompts": [],
                "metadata": {
                    "total_count": 0,
                    "extraction_date": datetime.now().isoformat(),
                    "database_path": db_path
                }
            }
        
        print(f"✅ Found {len(rows)} tasks with 0.0 task fidelity")
        
        # Extract just the prompts and detailed data
        prompts = []
        detailed_data = []
        
        for row in rows:
            prompt = row[0]
            prompts.append(prompt)
            
            detailed_data.append({
                "prompt": prompt,
                "task_id": row[1],
                "validator_uid": row[2],
                "pulled_at": row[3],
                "submitted_at": row[4],
                "task_fidelity_score": row[5],
                "average_fidelity_score": row[6],
                "validation_failed": bool(row[7]) if row[7] is not None else None,
                "generation_time": row[8],
                "validation_time": row[9],
                "local_validation_score": row[10]
            })
        
        # Remove duplicates while preserving order
        unique_prompts = []
        seen = set()
        for prompt in prompts:
            if prompt not in seen:
                unique_prompts.append(prompt)
                seen.add(prompt)
        
        print(f"📝 Unique prompts: {len(unique_prompts)}")
        print(f"📈 Total zero fidelity tasks: {len(prompts)}")
        
        # Get some statistics
        cursor.execute("SELECT COUNT(*) FROM tasks WHERE feedback_received = 1")
        total_tasks = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(*) FROM tasks WHERE task_fidelity_score > 0.0 AND feedback_received = 1")
        successful_tasks = cursor.fetchone()[0]
        
        conn.close()
        
        return {
            "prompts": unique_prompts,
            "all_zero_fidelity_entries": detailed_data,
            "metadata": {
                "unique_prompts_count": len(unique_prompts),
                "total_zero_fidelity_tasks": len(prompts),
                "total_tasks_with_feedback": total_tasks,
                "successful_tasks": successful_tasks,
                "zero_fidelity_rate": len(prompts) / total_tasks if total_tasks > 0 else 0,
                "extraction_date": datetime.now().isoformat(),
                "database_path": db_path
            }
        }
        
    except sqlite3.Error as e:
        print(f"❌ Database error: {e}")
        return {
            "prompts": [],
            "metadata": {
                "error": str(e),
                "extraction_date": datetime.now().isoformat(),
                "database_path": db_path
            }
        }
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return {
            "prompts": [],
            "metadata": {
                "error": str(e),
                "extraction_date": datetime.now().isoformat(),
                "database_path": db_path
            }
        }

def save_as_python_list(prompts: List[str], filename: str = "zero_fidelity_prompts.txt"):
    """
    Save the prompts as a Python list format in a text file.
    
    Args:
        prompts: List of prompt strings
        filename: Output filename
    """
    
    print(f"💾 Saving {len(prompts)} prompts to {filename}")
    
    with open(filename, 'w', encoding='utf-8') as f:
        f.write("# Zero Task Fidelity Prompts\n")
        f.write(f"# Extracted on: {datetime.now().isoformat()}\n")
        f.write(f"# Total prompts: {len(prompts)}\n\n")
        
        f.write("zero_fidelity_prompts = [\n")
        for i, prompt in enumerate(prompts):
            # Escape quotes and special characters
            escaped_prompt = prompt.replace('"', '\\"').replace('\n', '\\n').replace('\r', '\\r')
            if i == len(prompts) - 1:
                f.write(f'    "{escaped_prompt}"\n')
            else:
                f.write(f'    "{escaped_prompt}",\n')
        f.write("]\n")
    
    print(f"✅ Saved prompts list to {filename}")

def save_detailed_json(data: Dict[str, Any], filename: str = "zero_fidelity_detailed_data.json"):
    """
    Save detailed data as JSON for further analysis.
    
    Args:
        data: Complete extracted data
        filename: Output JSON filename
    """
    
    print(f"💾 Saving detailed data to {filename}")
    
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    
    print(f"✅ Saved detailed data to {filename}")

def main():
    """Main execution function"""
    
    print("🎯 ZERO TASK FIDELITY PROMPT EXTRACTOR")
    print("=" * 50)
    
    # Extract the data
    data = extract_zero_fidelity_prompts()
    
    if not data["prompts"]:
        print("❌ No prompts to save")
        return
    
    # Print statistics
    metadata = data["metadata"]
    print("\n📊 EXTRACTION STATISTICS")
    print("=" * 50)
    print(f"Unique prompts with 0 fidelity: {metadata.get('unique_prompts_count', 0)}")
    print(f"Total zero fidelity tasks: {metadata.get('total_zero_fidelity_tasks', 0)}")
    print(f"Total tasks with feedback: {metadata.get('total_tasks_with_feedback', 0)}")
    print(f"Successful tasks: {metadata.get('successful_tasks', 0)}")
    print(f"Zero fidelity rate: {metadata.get('zero_fidelity_rate', 0):.2%}")
    
    # Show some example prompts
    print("\n📝 SAMPLE PROMPTS (first 10)")
    print("=" * 50)
    for i, prompt in enumerate(data["prompts"][:10]):
        print(f"{i+1:2d}. {prompt}")
    
    if len(data["prompts"]) > 10:
        print(f"... and {len(data['prompts']) - 10} more")
    
    # Save the files
    print("\n💾 SAVING FILES")
    print("=" * 50)
    save_as_python_list(data["prompts"])
    save_detailed_json(data)
    
    print("\n✅ EXTRACTION COMPLETE!")
    print(f"📁 Files created:")
    print(f"   - zero_fidelity_prompts.txt (Python list)")
    print(f"   - zero_fidelity_detailed_data.json (Detailed data)")

if __name__ == "__main__":
    main() 