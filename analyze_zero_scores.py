#!/usr/bin/env python3
"""
Analyze zero score patterns from continuous TRELLIS mining
Helps identify problematic prompt patterns
"""

import sqlite3
import json
from pathlib import Path
from collections import Counter, defaultdict
from datetime import datetime, timedelta

def analyze_zero_scores(db_path: str = "continuous_trellis_tasks.db", hours: int = 24):
    """Analyze tasks with zero fidelity scores"""
    
    if not Path(db_path).exists():
        print(f"❌ Database not found: {db_path}")
        return
    
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Get recent tasks
    cutoff_time = datetime.now() - timedelta(hours=hours)
    cutoff_timestamp = cutoff_time.timestamp()
    
    # Fetch all tasks with scores
    cursor.execute("""
        SELECT prompt, task_fidelity_score, average_fidelity_score, 
               submission_success, validation_failed, processed_at
        FROM tasks 
        WHERE processed_at > ? AND submission_success = 1
        ORDER BY processed_at DESC
    """, (cutoff_timestamp,))
    
    tasks = cursor.fetchall()
    
    # Categorize tasks
    zero_scores = []
    low_scores = []  # 0.0 < score < 0.5
    good_scores = []  # score >= 0.5
    
    for task in tasks:
        prompt, task_score, avg_score, success, val_failed, timestamp = task
        
        if task_score == 0.0:
            zero_scores.append(prompt)
        elif task_score < 0.5:
            low_scores.append((prompt, task_score))
        else:
            good_scores.append((prompt, task_score))
    
    print(f"\n📊 Task Analysis (last {hours} hours)")
    print("=" * 60)
    print(f"Total tasks submitted: {len(tasks)}")
    print(f"Zero scores (0.0): {len(zero_scores)} ({len(zero_scores)/len(tasks)*100:.1f}%)")
    print(f"Low scores (<0.5): {len(low_scores)} ({len(low_scores)/len(tasks)*100:.1f}%)")
    print(f"Good scores (≥0.5): {len(good_scores)} ({len(good_scores)/len(tasks)*100:.1f}%)")
    
    if zero_scores:
        print(f"\n🔍 Analyzing {len(zero_scores)} zero-score prompts...")
        
        # Common words in zero scores
        word_counter = Counter()
        material_keywords = ['glass', 'crystal', 'diamond', 'emerald', 'ruby', 'sapphire',
                           'transparent', 'translucent', 'liquid', 'water', 'mirror',
                           'jewelry', 'gem', 'pendant', 'necklace', 'bracelet']
        
        material_counts = defaultdict(int)
        
        for prompt in zero_scores:
            words = prompt.lower().split()
            word_counter.update(words)
            
            # Check for problematic materials
            for keyword in material_keywords:
                if keyword in prompt.lower():
                    material_counts[keyword] += 1
        
        print("\n📌 Most common words in zero-score prompts:")
        for word, count in word_counter.most_common(20):
            if count > 1:  # Only show words that appear multiple times
                print(f"   '{word}': {count} times")
        
        if material_counts:
            print("\n⚠️  Problematic materials detected:")
            for material, count in sorted(material_counts.items(), key=lambda x: x[1], reverse=True):
                percentage = (count / len(zero_scores)) * 100
                print(f"   '{material}': {count} prompts ({percentage:.1f}%)")
        
        print("\n📝 Recent zero-score examples:")
        for prompt in zero_scores[:10]:
            print(f"   • {prompt}")
        
        # Pattern detection
        patterns = {
            'multi_object': 0,
            'with_keyword': 0,
            'liquid_related': 0,
            'transparent': 0,
            'jewelry': 0,
            'abstract': 0
        }
        
        for prompt in zero_scores:
            prompt_lower = prompt.lower()
            if any(word in prompt_lower for word in ['with', 'holding', 'beside', 'and']):
                patterns['multi_object'] += 1
            if 'with' in prompt_lower:
                patterns['with_keyword'] += 1
            if any(word in prompt_lower for word in ['water', 'liquid', 'juice', 'fluid']):
                patterns['liquid_related'] += 1
            if any(word in prompt_lower for word in ['glass', 'crystal', 'transparent', 'clear']):
                patterns['transparent'] += 1
            if any(word in prompt_lower for word in ['jewelry', 'necklace', 'bracelet', 'ring', 'pendant']):
                patterns['jewelry'] += 1
            if any(word in prompt_lower for word in ['abstract', 'contemporary', 'artistic']):
                patterns['abstract'] += 1
        
        print("\n🎯 Pattern Analysis:")
        for pattern, count in sorted(patterns.items(), key=lambda x: x[1], reverse=True):
            if count > 0:
                percentage = (count / len(zero_scores)) * 100
                print(f"   {pattern}: {count} prompts ({percentage:.1f}%)")
    
    # Successful prompts analysis
    if good_scores:
        print(f"\n✅ Top scoring prompts:")
        good_scores.sort(key=lambda x: x[1], reverse=True)
        for prompt, score in good_scores[:10]:
            print(f"   • {prompt} (score: {score:.3f})")
    
    conn.close()
    
    # Recommendations
    print("\n💡 Recommendations:")
    print("   1. Avoid transparent materials (glass, crystal, gems)")
    print("   2. Avoid multi-object scenes with 'with' or 'holding'")
    print("   3. Focus on single, solid, opaque objects")
    print("   4. Use concrete material descriptions")
    print("   5. Test problematic prompt types locally first")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Analyze zero scores from TRELLIS mining")
    parser.add_argument("--db", default="continuous_trellis_tasks.db", help="Database path")
    parser.add_argument("--hours", type=int, default=24, help="Hours to look back")
    
    args = parser.parse_args()
    analyze_zero_scores(args.db, args.hours) 