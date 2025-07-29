#!/usr/bin/env python3
"""
Comprehensive script to analyze prompts across all sources:
1. episodic_test_prompts.py (test prompts)
2. episodic_logs/episodic_memory.json (episodic memory)
3. continuous_trellis_simulator.log.v1 (simulator log)
4. trellis_simulation_outputs/trellis_simulator_tasks.db.v1 (database)
"""

import json
import re
import sqlite3
from pathlib import Path
from collections import defaultdict

def load_test_prompts():
    """Load prompts from episodic_test_prompts.py"""
    prompts = []
    
    with open('episodic_test_prompts.py', 'r') as f:
        content = f.read()
    
    match = re.search(r'EPISODIC_TEST_PROMPTS\s*=\s*\[(.*?)\]', content, re.DOTALL)
    if not match:
        raise ValueError("Could not find EPISODIC_TEST_PROMPTS list in the file")
    
    list_content = match.group(1)
    lines = list_content.split('\n')
    
    for line in lines:
        line = line.strip()
        if line.startswith('"') and line.endswith('",'):
            prompt = line[1:-2]
            prompts.append(prompt)
        elif line.startswith('"') and line.endswith('"'):
            prompt = line[1:-1]
            prompts.append(prompt)
    
    return prompts

def load_episodic_memory():
    """Load original_prompt values from episodic_memory.json"""
    prompts = set()
    
    with open('episodic_logs/episodic_memory.json', 'r') as f:
        data = json.load(f)
    
    def extract_prompts(obj):
        if isinstance(obj, dict):
            for key, value in obj.items():
                if key == "original_prompt" and isinstance(value, str):
                    prompts.add(value)
                elif isinstance(value, (dict, list)):
                    extract_prompts(value)
        elif isinstance(obj, list):
            for item in obj:
                extract_prompts(item)
    
    extract_prompts(data)
    return prompts

def load_simulator_log():
    """Extract prompts from continuous simulator log"""
    prompts = set()
    
    with open('continuous_trellis_simulator.log.v1', 'r') as f:
        for line in f:
            # Look for lines that contain prompt processing
            if "Processing task sim_" in line and ":" in line:
                # Extract prompt from lines like: "Processing task sim_1: 'plastic straw of drink'"
                match = re.search(r"Processing task sim_\d+: '([^']+)'", line)
                if match:
                    prompts.add(match.group(1))
    
    return prompts

def load_database_prompts():
    """Load prompts from the SQLite database"""
    prompts = set()
    
    try:
        conn = sqlite3.connect('trellis_simulation_outputs/trellis_simulator_tasks.db.v1')
        cursor = conn.cursor()
        cursor.execute('SELECT prompt FROM tasks WHERE prompt IS NOT NULL')
        
        for row in cursor.fetchall():
            if row[0]:
                prompts.add(row[0])
        
        conn.close()
    except Exception as e:
        print(f"Warning: Could not read database: {e}")
    
    return prompts

def analyze_prompt_coverage():
    """Analyze prompt coverage across all sources"""
    
    print("🔍 Loading prompts from all sources...")
    
    # Load all sources
    test_prompts = set(load_test_prompts())
    episodic_prompts = load_episodic_memory()
    simulator_prompts = load_simulator_log()
    db_prompts = load_database_prompts()
    
    print(f"✅ Test prompts: {len(test_prompts)}")
    print(f"✅ Episodic memory: {len(episodic_prompts)}")
    print(f"✅ Simulator log: {len(simulator_prompts)}")
    print(f"✅ Database: {len(db_prompts)}")
    
    # Create comprehensive analysis
    analysis = defaultdict(set)
    
    for prompt in test_prompts:
        sources = []
        if prompt in episodic_prompts:
            sources.append("episodic")
        if prompt in simulator_prompts:
            sources.append("simulator")
        if prompt in db_prompts:
            sources.append("database")
        
        if not sources:
            analysis["missing_all"].add(prompt)
        else:
            analysis[f"in_{len(sources)}_sources"].add(prompt)
            for source in sources:
                analysis[f"in_{source}"].add(prompt)
    
    return analysis, test_prompts

def print_analysis(analysis, test_prompts):
    """Print comprehensive analysis results"""
    
    print(f"\n{'='*80}")
    print(f"📊 COMPREHENSIVE PROMPT ANALYSIS")
    print(f"{'='*80}")
    
    total_prompts = len(test_prompts)
    
    # Summary statistics
    print(f"\n📈 SUMMARY STATISTICS:")
    print(f"{'='*50}")
    print(f"Total test prompts: {total_prompts}")
    print(f"Missing from all sources: {len(analysis['missing_all'])}")
    print(f"Present in at least one source: {total_prompts - len(analysis['missing_all'])}")
    
    # Coverage by source
    print(f"\n🎯 COVERAGE BY SOURCE:")
    print(f"{'='*50}")
    for source in ["episodic", "simulator", "database"]:
        count = len(analysis[f"in_{source}"])
        percentage = (count / total_prompts) * 100
        print(f"{source.capitalize():12}: {count:3d} prompts ({percentage:5.1f}%)")
    
    # Distribution by number of sources
    print(f"\n📊 DISTRIBUTION BY SOURCE COUNT:")
    print(f"{'='*50}")
    for i in range(1, 4):
        key = f"in_{i}_sources"
        if key in analysis:
            count = len(analysis[key])
            percentage = (count / total_prompts) * 100
            print(f"In {i} source{'s' if i > 1 else ''}: {count:3d} prompts ({percentage:5.1f}%)")
    
    # Missing prompts
    if analysis["missing_all"]:
        print(f"\n❌ MISSING FROM ALL SOURCES ({len(analysis['missing_all'])}):")
        print(f"{'='*50}")
        for i, prompt in enumerate(sorted(analysis["missing_all"]), 1):
            print(f"{i:3d}. {prompt}")
    
    # Prompts in episodic but not others
    episodic_only = analysis["in_episodic"] - analysis["in_simulator"] - analysis["in_database"]
    if episodic_only:
        print(f"\n🔄 IN EPISODIC ONLY ({len(episodic_only)}):")
        print(f"{'='*50}")
        for i, prompt in enumerate(sorted(episodic_only), 1):
            print(f"{i:3d}. {prompt}")
    
    # Prompts in simulator but not episodic
    simulator_only = analysis["in_simulator"] - analysis["in_episodic"]
    if simulator_only:
        print(f"\n🚀 IN SIMULATOR ONLY ({len(simulator_only)}):")
        print(f"{'='*50}")
        for i, prompt in enumerate(sorted(simulator_only), 1):
            print(f"{i:3d}. {prompt}")
    
    # Prompts in database but not episodic
    db_only = analysis["in_database"] - analysis["in_episodic"]
    if db_only:
        print(f"\n💾 IN DATABASE ONLY ({len(db_only)}):")
        print(f"{'='*50}")
        for i, prompt in enumerate(sorted(db_only), 1):
            print(f"{i:3d}. {prompt}")

def main():
    print("🔍 Starting comprehensive prompt analysis...")
    
    try:
        analysis, test_prompts = analyze_prompt_coverage()
        print_analysis(analysis, test_prompts)
        
        # Save detailed results to file
        results = {
            "total_test_prompts": len(test_prompts),
            "missing_from_all": sorted(list(analysis["missing_all"])),
            "in_episodic": sorted(list(analysis["in_episodic"])),
            "in_simulator": sorted(list(analysis["in_simulator"])),
            "in_database": sorted(list(analysis["in_database"])),
            "episodic_only": sorted(list(analysis["in_episodic"] - analysis["in_simulator"] - analysis["in_database"])),
            "simulator_only": sorted(list(analysis["in_simulator"] - analysis["in_episodic"])),
            "db_only": sorted(list(analysis["in_database"] - analysis["in_episodic"]))
        }
        
        with open('prompt_analysis_results.json', 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\n💾 Detailed results saved to: prompt_analysis_results.json")
        
    except Exception as e:
        print(f"❌ Error during analysis: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 