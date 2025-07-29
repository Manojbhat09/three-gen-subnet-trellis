#!/usr/bin/env python3
"""
TRELLIS Winning Prompts Analysis
Comprehensive analysis of successful prompts across all versions to understand what makes them work well.
"""

import re
import math
import statistics
from collections import defaultdict
from typing import Dict, List, Any, Tuple

class TrellisWinningPromptsAnalyzer:
    """Analyzer for winning and successful prompts across versions"""
    
    def __init__(self):
        self.v0_log = "continuous_trellis_simulator.log.v0"
        self.v1_log = "continuous_trellis_simulator.log.v1"
        self.v2_log = "continuous_trellis_simulator.log.v2"
        self.v3_log = "continuous_trellis_simulator.log.v3"
        self.v4_log = "continuous_trellis_simulator.log.v4"
        self.v5_log = "continuous_trellis_simulator.log.v5"
        self.base_log = "continuous_trellis_simulator.log"
        
    def parse_log_file_detailed(self, log_file_path: str) -> List[Dict[str, Any]]:
        """Detailed log parsing with all component scores"""
        
        task_data = []
        current_task = None
        
        try:
            with open(log_file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    # Look for task processing start
                    processing_match = re.search(r"🔄 Processing task ([a-zA-Z0-9_]+): '([^']+)'", line)
                    if processing_match:
                        task_id = processing_match.group(1)
                        prompt = processing_match.group(2)
                        current_task = {
                            'task_id': task_id,
                            'prompt': prompt,
                            'validation_engine_score': None,
                            'alignment_score': None,
                            'quality_score': None,
                            'demo_fidelity_score': None,
                            'task_fidelity_score': None,
                            'validation_passed': False,
                            'generation_success': False,
                            'validation_success': False,
                            'validation_failed': False,
                            'cuda_oom': False,
                            'generation_time': None,
                            'validation_time': None,
                            'optimization_applied': False,
                            'reproducibility_optimization': False,
                            'traditional_optimization': False
                        }
                    
                    # Look for validation engine score
                    engine_match = re.search(r"🏆 Validation Engine Score: ([\d.]+)", line)
                    if engine_match and current_task:
                        current_task['validation_engine_score'] = float(engine_match.group(1))
                        current_task['validation_success'] = True
                    
                    # Look for alignment score
                    alignment_match = re.search(r"🤝 Alignment Score: ([\d.]+)", line)
                    if alignment_match and current_task:
                        current_task['alignment_score'] = float(alignment_match.group(1))
                    
                    # Look for quality score
                    quality_match = re.search(r"💎 Quality Score: ([\d.]+)", line)
                    if quality_match and current_task:
                        current_task['quality_score'] = float(quality_match.group(1))
                    
                    # Look for demo fidelity score
                    demo_match = re.search(r"🎭 Demo Fidelity Score: ([\d.]+)", line)
                    if demo_match and current_task:
                        current_task['demo_fidelity_score'] = float(demo_match.group(1))
                    
                    # Look for task fidelity score
                    task_match = re.search(r"🎯 Task Fidelity Score: ([\d.]+)", line)
                    if task_match and current_task:
                        current_task['task_fidelity_score'] = float(task_match.group(1))
                    
                    # Look for validation passed
                    passed_match = re.search(r"✅ Validation Passed: (True|False)", line)
                    if passed_match and current_task:
                        current_task['validation_passed'] = passed_match.group(1) == 'True'
                    
                    # Look for generation success
                    gen_success_match = re.search(r"✅ Generation successful in ([\d.]+)s", line)
                    if gen_success_match and current_task:
                        current_task['generation_success'] = True
                        current_task['generation_time'] = float(gen_success_match.group(1))
                    
                    # Look for validation completion
                    val_success_match = re.search(r"✅ Validation completed in ([\d.]+)s", line)
                    if val_success_match and current_task:
                        current_task['validation_time'] = float(val_success_match.group(1))
                    
                    # Look for task completion
                    completion_match = re.search(r"✅ Task ([a-zA-Z0-9_]+) finished processing", line)
                    if completion_match and current_task:
                        # If we haven't found validation engine score yet, mark as failed
                        if current_task['validation_engine_score'] is None:
                            current_task['validation_engine_score'] = 0.0
                            current_task['validation_failed'] = True
                        task_data.append(current_task)
                        current_task = None
                        
        except Exception as e:
            print(f"Error parsing {log_file_path}: {e}")
            
        return task_data

    def categorize_prompt(self, prompt: str) -> str:
        """Categorize prompt based on content"""
        prompt_lower = prompt.lower()
        
        if any(word in prompt_lower for word in ['robot', 'mechanical', 'drone', 'cyborg']):
            return 'robots/mechanical'
        elif any(word in prompt_lower for word in ['sword', 'knife', 'rifle', 'gun', 'weapon', 'spear', 'arrow', 'crossbow', 'bayonet']):
            return 'weapons'
        elif any(word in prompt_lower for word in ['crystal', 'gem', 'diamond', 'emerald', 'sapphire', 'opal', 'amethyst', 'quartz', 'onyx', 'jasper']):
            return 'gems/crystals'
        elif any(word in prompt_lower for word in ['necklace', 'earrings', 'bracelet', 'locket', 'pendant', 'ring', 'jewelry']):
            return 'jewelry'
        elif any(word in prompt_lower for word in ['statue', 'sculpture', 'figure', 'totem']):
            return 'statues/sculptures'
        elif any(word in prompt_lower for word in ['alien', 'elf', 'troll', 'fairy', 'creature', 'dinosaur']):
            return 'creatures/characters'
        elif any(word in prompt_lower for word in ['car', 'van', 'vehicle', 'minivan']):
            return 'vehicles'
        elif any(word in prompt_lower for word in ['sofa', 'chair', 'furniture', 'lounge']):
            return 'furniture'
        elif any(word in prompt_lower for word in ['pizza', 'food', 'cupcake', 'donut', 'wine', 'juice', 'lemonade']):
            return 'food/drinks'
        elif any(word in prompt_lower for word in ['guitar', 'saxophone', 'clarinet', 'harp', 'flute', 'harmonica', 'instrument']):
            return 'musical_instruments'
        elif any(word in prompt_lower for word in ['drill', 'screwdriver', 'hammer', 'wrench', 'pliers', 'chisel', 'tool']):
            return 'tools'
        elif any(word in prompt_lower for word in ['bat', 'racket', 'club', 'sport']):
            return 'sports_equipment'
        else:
            return 'other'

    def analyze_winning_prompts(self, task_data: List[Dict[str, Any]], version: str) -> Dict[str, Any]:
        """Analyze winning prompts for a specific version"""
        
        # Define winning categories
        excellent_prompts = [t for t in task_data if t['validation_engine_score'] and t['validation_engine_score'] >= 0.9]
        good_prompts = [t for t in task_data if t['validation_engine_score'] and t['validation_engine_score'] >= 0.8]
        successful_prompts = [t for t in task_data if t['validation_engine_score'] and t['validation_engine_score'] > 0.0]
        
        analysis = {
            'version': version,
            'total_tasks': len(task_data),
            'excellent_prompts': excellent_prompts,
            'good_prompts': good_prompts,
            'successful_prompts': successful_prompts,
            'excellent_count': len(excellent_prompts),
            'good_count': len(good_prompts),
            'successful_count': len(successful_prompts),
            'excellent_rate': (len(excellent_prompts) / len(task_data)) * 100 if task_data else 0,
            'good_rate': (len(good_prompts) / len(task_data)) * 100 if task_data else 0,
            'success_rate': (len(successful_prompts) / len(task_data)) * 100 if task_data else 0
        }
        
        # Analyze excellent prompts by category
        if excellent_prompts:
            category_stats = defaultdict(list)
            for prompt in excellent_prompts:
                category = self.categorize_prompt(prompt['prompt'])
                category_stats[category].append(prompt)
            
            analysis['excellent_by_category'] = {
                category: {
                    'count': len(prompts),
                    'prompts': prompts,
                    'avg_score': statistics.mean([p['validation_engine_score'] for p in prompts]),
                    'avg_alignment': statistics.mean([p['alignment_score'] for p in prompts if p['alignment_score']]),
                    'avg_quality': statistics.mean([p['quality_score'] for p in prompts if p['quality_score']]),
                    'avg_demo_fidelity': statistics.mean([p['demo_fidelity_score'] for p in prompts if p['demo_fidelity_score']]),
                    'avg_task_fidelity': statistics.mean([p['task_fidelity_score'] for p in prompts if p['task_fidelity_score']])
                }
                for category, prompts in category_stats.items()
            }
        
        return analysis

    def find_consistent_winners(self, all_versions_data: Dict[str, List[Dict[str, Any]]]) -> Dict[str, Any]:
        """Find prompts that perform well across multiple versions"""
        
        # Get all prompts and their scores across versions
        prompt_scores = defaultdict(dict)
        
        for version, task_data in all_versions_data.items():
            for task in task_data:
                prompt = task['prompt']
                score = task['validation_engine_score'] if task['validation_engine_score'] else 0.0
                prompt_scores[prompt][version] = score
        
        # Find consistent winners (high scores across multiple versions)
        consistent_winners = []
        consistent_good = []
        consistent_successful = []
        
        for prompt, version_scores in prompt_scores.items():
            scores = list(version_scores.values())
            avg_score = statistics.mean(scores)
            min_score = min(scores)
            max_score = max(scores)
            versions_tested = len(scores)
            
            prompt_info = {
                'prompt': prompt,
                'avg_score': avg_score,
                'min_score': min_score,
                'max_score': max_score,
                'versions_tested': versions_tested,
                'version_scores': version_scores,
                'category': self.categorize_prompt(prompt)
            }
            
            # Categorize based on performance
            if avg_score >= 0.9 and min_score >= 0.8:
                consistent_winners.append(prompt_info)
            elif avg_score >= 0.8 and min_score >= 0.6:
                consistent_good.append(prompt_info)
            elif avg_score >= 0.6 and min_score > 0.0:
                consistent_successful.append(prompt_info)
        
        return {
            'consistent_winners': sorted(consistent_winners, key=lambda x: x['avg_score'], reverse=True),
            'consistent_good': sorted(consistent_good, key=lambda x: x['avg_score'], reverse=True),
            'consistent_successful': sorted(consistent_successful, key=lambda x: x['avg_score'], reverse=True)
        }

    def analyze_prompt_patterns(self, winning_prompts: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze patterns in winning prompts"""
        
        if not winning_prompts:
            return {}
        
        # Analyze prompt characteristics
        prompt_lengths = [len(p['prompt']) for p in winning_prompts]
        word_counts = [len(p['prompt'].split()) for p in winning_prompts]
        
        # Look for common words/phrases
        all_words = []
        for prompt in winning_prompts:
            words = prompt['prompt'].lower().split()
            all_words.extend(words)
        
        word_freq = defaultdict(int)
        for word in all_words:
            if len(word) > 2:  # Skip short words
                word_freq[word] += 1
        
        # Find most common words
        common_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)[:20]
        
        # Analyze color words
        color_words = ['red', 'blue', 'green', 'yellow', 'black', 'white', 'purple', 'pink', 'orange', 'gold', 'silver', 'brown', 'gray', 'grey']
        color_usage = {color: word_freq.get(color, 0) for color in color_words}
        
        # Analyze material words
        material_words = ['wooden', 'metal', 'plastic', 'glass', 'golden', 'silver', 'brass', 'leather', 'velvet', 'silk', 'crystal', 'stone']
        material_usage = {material: word_freq.get(material, 0) for material in material_words}
        
        return {
            'prompt_length_stats': {
                'avg_length': statistics.mean(prompt_lengths),
                'median_length': statistics.median(prompt_lengths),
                'min_length': min(prompt_lengths),
                'max_length': max(prompt_lengths)
            },
            'word_count_stats': {
                'avg_words': statistics.mean(word_counts),
                'median_words': statistics.median(word_counts),
                'min_words': min(word_counts),
                'max_words': max(word_counts)
            },
            'common_words': common_words,
            'color_usage': color_usage,
            'material_usage': material_usage
        }

    def run_winning_analysis(self):
        """Run comprehensive winning prompts analysis"""
        
        print("=" * 120)
        print("TRELLIS WINNING PROMPTS ANALYSIS - BASE vs V0 vs V1 vs V2 vs V3 vs V4 vs V5")
        print("=" * 120)
        
        # Parse all log files
        print(f"\n📁 Parsing log files...")
        base_data = self.parse_log_file_detailed(self.base_log)
        v0_data = self.parse_log_file_detailed(self.v0_log)
        v1_data = self.parse_log_file_detailed(self.v1_log)
        v2_data = self.parse_log_file_detailed(self.v2_log)
        v3_data = self.parse_log_file_detailed(self.v3_log)
        v4_data = self.parse_log_file_detailed(self.v4_log)
        v5_data = self.parse_log_file_detailed(self.v5_log)
        
        print(f"   BASE: {len(base_data)} tasks (NO OPTIMIZATION)")
        print(f"   V0: {len(v0_data)} tasks")
        print(f"   V1: {len(v0_data)} tasks")
        print(f"   V2: {len(v2_data)} tasks")
        print(f"   V3: {len(v3_data)} tasks")
        print(f"   V4: {len(v4_data)} tasks")
        print(f"   V5: {len(v5_data)} tasks")
        
        # Analyze winning prompts for each version
        print(f"\n" + "=" * 120)
        print(f"WINNING PROMPTS BY VERSION")
        print(f"=" * 120)
        
        all_versions_data = {
            'base': base_data,
            'v0': v0_data,
            'v1': v1_data,
            'v2': v2_data,
            'v3': v3_data,
            'v4': v4_data,
            'v5': v5_data
        }
        
        version_analyses = {}
        for version, data in all_versions_data.items():
            analysis = self.analyze_winning_prompts(data, version.upper())
            version_analyses[version] = analysis
            
            print(f"\n📊 {version.upper()} WINNING STATISTICS:")
            print(f"   Total Tasks: {analysis['total_tasks']}")
            print(f"   Excellent (≥0.9): {analysis['excellent_count']} ({analysis['excellent_rate']:.1f}%)")
            print(f"   Good (≥0.8): {analysis['good_count']} ({analysis['good_rate']:.1f}%)")
            print(f"   Successful (>0.0): {analysis['successful_count']} ({analysis['success_rate']:.1f}%)")
            
            # Show top excellent prompts
            if analysis['excellent_prompts']:
                print(f"\n🏆 TOP EXCELLENT PROMPTS ({version.upper()}):")
                sorted_excellent = sorted(analysis['excellent_prompts'], 
                                        key=lambda x: x['validation_engine_score'], reverse=True)
                for i, prompt in enumerate(sorted_excellent[:10], 1):
                    print(f"   {i:2d}. {prompt['prompt']} (Score: {prompt['validation_engine_score']:.4f})")
                    print(f"       Alignment: {prompt['alignment_score']:.4f}, Quality: {prompt['quality_score']:.4f}, Demo: {prompt['demo_fidelity_score']:.4f}, Task: {prompt['task_fidelity_score']:.4f}")
        
        # Find consistent winners across versions
        print(f"\n" + "=" * 120)
        print(f"CONSISTENT WINNERS ACROSS VERSIONS")
        print(f"=" * 120)
        
        consistent_analysis = self.find_consistent_winners(all_versions_data)
        
        print(f"\n🏆 CONSISTENT WINNERS (Avg ≥0.9, Min ≥0.8):")
        if consistent_analysis['consistent_winners']:
            for i, winner in enumerate(consistent_analysis['consistent_winners'][:15], 1):
                print(f"   {i:2d}. {winner['prompt']}")
                print(f"       Avg: {winner['avg_score']:.4f}, Min: {winner['min_score']:.4f}, Max: {winner['max_score']:.4f}")
                print(f"       Versions: {winner['versions_tested']}, Category: {winner['category']}")
                print(f"       Scores: {winner['version_scores']}")
        else:
            print("   No consistent winners found")
        
        print(f"\n✅ CONSISTENT GOOD (Avg ≥0.8, Min ≥0.6):")
        if consistent_analysis['consistent_good']:
            for i, good in enumerate(consistent_analysis['consistent_good'][:10], 1):
                print(f"   {i:2d}. {good['prompt']}")
                print(f"       Avg: {good['avg_score']:.4f}, Min: {good['min_score']:.4f}, Max: {good['max_score']:.4f}")
                print(f"       Category: {good['category']}")
        else:
            print("   No consistent good performers found")
        
        # Analyze patterns in winning prompts
        print(f"\n" + "=" * 120)
        print(f"WINNING PROMPT PATTERNS ANALYSIS")
        print(f"=" * 120)
        
        # Combine all excellent prompts across versions
        all_excellent = []
        for analysis in version_analyses.values():
            all_excellent.extend(analysis['excellent_prompts'])
        
        if all_excellent:
            patterns = self.analyze_prompt_patterns(all_excellent)
            
            print(f"\n📏 PROMPT LENGTH STATISTICS:")
            length_stats = patterns['prompt_length_stats']
            print(f"   Average Length: {length_stats['avg_length']:.1f} characters")
            print(f"   Median Length: {length_stats['median_length']:.1f} characters")
            print(f"   Range: {length_stats['min_length']} - {length_stats['max_length']} characters")
            
            print(f"\n📝 WORD COUNT STATISTICS:")
            word_stats = patterns['word_count_stats']
            print(f"   Average Words: {word_stats['avg_words']:.1f} words")
            print(f"   Median Words: {word_stats['median_words']:.1f} words")
            print(f"   Range: {word_stats['min_words']} - {word_stats['max_words']} words")
            
            print(f"\n🎨 COLOR USAGE IN WINNING PROMPTS:")
            color_usage = patterns['color_usage']
            for color, count in sorted(color_usage.items(), key=lambda x: x[1], reverse=True):
                if count > 0:
                    print(f"   {color}: {count} times")
            
            print(f"\n🔧 MATERIAL USAGE IN WINNING PROMPTS:")
            material_usage = patterns['material_usage']
            for material, count in sorted(material_usage.items(), key=lambda x: x[1], reverse=True):
                if count > 0:
                    print(f"   {material}: {count} times")
            
            print(f"\n📚 MOST COMMON WORDS IN WINNING PROMPTS:")
            common_words = patterns['common_words']
            for word, count in common_words[:15]:
                print(f"   '{word}': {count} times")
        
        # Category analysis
        print(f"\n" + "=" * 120)
        print(f"CATEGORY PERFORMANCE ANALYSIS")
        print(f"=" * 120)
        
        for version, analysis in version_analyses.items():
            if 'excellent_by_category' in analysis:
                print(f"\n📊 {version.upper()} EXCELLENT PROMPTS BY CATEGORY:")
                for category, stats in analysis['excellent_by_category'].items():
                    print(f"   {category}: {stats['count']} prompts (avg score: {stats['avg_score']:.4f})")
                    if stats['count'] <= 5:  # Show prompts for small categories
                        for prompt in stats['prompts']:
                            print(f"     • {prompt['prompt']} ({prompt['validation_engine_score']:.4f})")
        
        print(f"\n" + "=" * 120)
        print(f"WINNING ANALYSIS COMPLETE")
        print(f"=" * 120)

def main():
    """Main function"""
    analyzer = TrellisWinningPromptsAnalyzer()
    analyzer.run_winning_analysis()

if __name__ == "__main__":
    main() 