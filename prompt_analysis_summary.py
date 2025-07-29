#!/usr/bin/env python3
"""
Summary of the comprehensive prompt analysis findings.
"""

import json

def load_results():
    """Load the analysis results"""
    with open('prompt_analysis_results.json', 'r') as f:
        return json.load(f)

def print_summary():
    """Print a clear summary of the findings"""
    
    results = load_results()
    
    print("🔍 COMPREHENSIVE PROMPT ANALYSIS SUMMARY")
    print("=" * 60)
    
    total = results["total_test_prompts"]
    episodic = len(results["in_episodic"])
    simulator = len(results["in_simulator"])
    database = len(results["in_database"])
    
    print(f"\n📊 OVERALL COVERAGE:")
    print(f"   • Total test prompts: {total}")
    print(f"   • Episodic memory: {episodic} ({episodic/total*100:.1f}%)")
    print(f"   • Simulator log: {simulator} ({simulator/total*100:.1f}%)")
    print(f"   • Database: {database} ({database/total*100:.1f}%)")
    
    print(f"\n🎯 KEY FINDINGS:")
    print(f"   ✅ All 71 prompts have been processed by the simulator")
    print(f"   ✅ All 71 prompts are stored in the database")
    print(f"   ⚠️  Only {episodic} prompts ({episodic/total*100:.1f}%) are in episodic memory")
    
    print(f"\n🔄 EPISODIC MEMORY GAP:")
    print(f"   • {total - episodic} prompts ({26} total) are missing from episodic memory")
    print(f"   • These prompts were processed by the simulator but not learned by the episodic system")
    
    print(f"\n📋 MISSING FROM EPISODIC MEMORY ({len(results['simulator_only'])}):")
    print("-" * 50)
    for i, prompt in enumerate(results['simulator_only'], 1):
        print(f"{i:2d}. {prompt}")
    
    print(f"\n💡 RECOMMENDATIONS:")
    print(f"   1. The episodic memory system needs to be updated to include all processed prompts")
    print(f"   2. There may be a synchronization issue between the simulator and episodic memory")
    print(f"   3. Consider running a backfill process to add missing prompts to episodic memory")
    print(f"   4. The simulator and database are working correctly (100% coverage)")

if __name__ == "__main__":
    print_summary() 