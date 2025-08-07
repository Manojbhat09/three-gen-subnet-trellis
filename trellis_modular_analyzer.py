#!/usr/bin/env python3
"""
Modular TRELLIS Prompt Analysis Tool
Analyzes winning and losing prompts for multiple versions with command-line arguments.
"""

import re
import json
import argparse
import sys
from typing import List, Dict, Any, Tuple
from collections import defaultdict

class TrellisModularAnalyzer:
    def __init__(self):
        self.versions = {}
        
    def add_version(self, name: str, log_file: str, db_file: str = None):
        """Add a version to analyze."""
        self.versions[name] = {
            'log_file': log_file,
            'db_file': db_file
        }
    
    def parse_log_file_detailed(self, log_file_path: str) -> List[Dict[str, Any]]:
        """Parse log file and extract detailed task information."""
        tasks = []
        
        try:
            with open(log_file_path, 'r') as f:
                lines = f.readlines()
            
            current_task = None
            
            for line in lines:
                # Find task start
                processing_match = re.search(r"🔄 Processing task ([a-zA-Z0-9_]+): '([^']+)'", line)
                if processing_match:
                    if current_task:
                        tasks.append(current_task)
                    
                    current_task = {
                        'task_id': processing_match.group(1),
                        'prompt': processing_match.group(2),
                        'validation_score': None,
                        'alignment_score': None,
                        'quality_score': None,
                        'demo_fidelity_score': None,
                        'task_fidelity_score': None,
                        'validation_passed': False
                    }
                
                # Find validation scores
                if current_task:
                    validation_match = re.search(r"🏆 Validation Engine Score: ([0-9.-]+)", line)
                    if validation_match:
                        current_task['validation_score'] = float(validation_match.group(1))
                    
                    alignment_match = re.search(r"🤝 Alignment Score: ([0-9.-]+)", line)
                    if alignment_match:
                        current_task['alignment_score'] = float(alignment_match.group(1))
                    
                    quality_match = re.search(r"💎 Quality Score: ([0-9.-]+)", line)
                    if quality_match:
                        current_task['quality_score'] = float(quality_match.group(1))
                    
                    demo_match = re.search(r"🎭 Demo Fidelity Score: ([0-9.-]+)", line)
                    if demo_match:
                        current_task['demo_fidelity_score'] = float(demo_match.group(1))
                    
                    task_match = re.search(r"🎯 Task Fidelity Score: ([0-9.-]+)", line)
                    if task_match:
                        current_task['task_fidelity_score'] = float(task_match.group(1))
                    
                    passed_match = re.search(r"✅ Validation Passed: (True|False)", line)
                    if passed_match:
                        current_task['validation_passed'] = passed_match.group(1) == 'True'
            
            # Add the last task
            if current_task:
                tasks.append(current_task)
                
        except Exception as e:
            print(f"Error parsing {log_file_path}: {e}")
            
        return tasks
    
    def categorize_prompt(self, prompt: str) -> str:
        """Categorize prompt based on content."""
        prompt_lower = prompt.lower()
        
        if any(word in prompt_lower for word in ['robot', 'mechanical', 'octahedral']):
            return 'robots/mechanical'
        elif any(word in prompt_lower for word in ['saxophone', 'clarinet', 'harmonica', 'flute', 'harp', 'guitar', 'bass', 'cricket', 'marimba']):
            return 'musical_instruments'
        elif any(word in prompt_lower for word in ['sword', 'rifle', 'spear', 'crossbow', 'dagger', 'bayonet', 'arrow', 'gun']):
            return 'weapons'
        elif any(word in prompt_lower for word in ['car', 'van', 'vehicle', 'minivan']):
            return 'vehicles'
        elif any(word in prompt_lower for word in ['crystal', 'gem', 'opal', 'emerald', 'sapphire', 'amethyst', 'quartz', 'onyx', 'jasper']):
            return 'gems/crystals'
        elif any(word in prompt_lower for word in ['bat', 'racket', 'club', 'golf']):
            return 'sports_equipment'
        elif any(word in prompt_lower for word in ['drill', 'screwdriver', 'wrench', 'hammer', 'chisel', 'pliers', 'knife', 'trowel', 'putty']):
            return 'tools'
        elif any(word in prompt_lower for word in ['elf', 'troll', 'alien', 'fairy', 'dinosaur', 'creature', 'maiden', 'soldier']):
            return 'creatures/characters'
        elif any(word in prompt_lower for word in ['staff', 'bow', 'vase', 'sofa', 'bucket', 'pizza', 'straw', 'cup', 'bottle', 'tray', 'candlestick', 'necklace', 'bracelet', 'locket', 'earrings']):
            return 'other'
        else:
            return 'other'
    
    def analyze_version_prompts(self, version_name: str, tasks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze prompts for a specific version."""
        if not tasks:
            return {}
        
        # Separate winning and losing prompts
        winning_prompts = [t for t in tasks if t.get('validation_score') is not None and t.get('validation_score', 0) > 0]
        losing_prompts = [t for t in tasks if t.get('validation_score') is None or t.get('validation_score', 0) == 0]
        
        # Sort by score
        winning_prompts.sort(key=lambda x: x.get('validation_score', 0), reverse=True)
        losing_prompts.sort(key=lambda x: x.get('prompt', ''))
        
        # Get max scores
        valid_scores = [t.get('validation_score', 0) for t in tasks if t.get('validation_score') is not None]
        max_score = max(valid_scores) if valid_scores else 0
        max_score_prompt = next((t for t in tasks if t.get('validation_score') is not None and t.get('validation_score', 0) == max_score), None)
        
        # Categorize prompts
        winning_by_category = defaultdict(list)
        losing_by_category = defaultdict(list)
        
        for task in winning_prompts:
            category = self.categorize_prompt(task['prompt'])
            winning_by_category[category].append(task)
        
        for task in losing_prompts:
            category = self.categorize_prompt(task['prompt'])
            losing_by_category[category].append(task)
        
        return {
            'version': version_name,
            'total_tasks': len(tasks),
            'winning_count': len(winning_prompts),
            'losing_count': len(losing_prompts),
            'success_rate': len(winning_prompts) / len(tasks) * 100 if tasks else 0,
            'max_score': max_score,
            'max_score_prompt': max_score_prompt,
            'winning_prompts': winning_prompts,
            'losing_prompts': losing_prompts,
            'winning_by_category': dict(winning_by_category),
            'losing_by_category': dict(losing_by_category),
            'excellent_prompts': [t for t in tasks if t.get('validation_score') is not None and t.get('validation_score', 0) >= 0.9],
            'good_prompts': [t for t in tasks if t.get('validation_score') is not None and t.get('validation_score', 0) >= 0.8],
            'average_score': sum(t.get('validation_score', 0) for t in tasks if t.get('validation_score') is not None) / len([t for t in tasks if t.get('validation_score') is not None]) if [t for t in tasks if t.get('validation_score') is not None] else 0
        }
    
    def generate_complete_summary(self, detailed: bool = False):
        """Generate complete summary for all versions."""
        print("=" * 120)
        print("MODULAR TRELLIS PROMPT SUMMARY - WINNING & LOSING PROMPTS BY VERSION")
        print("=" * 120)
        
        # Parse all log files
        print(f"\n📁 Parsing log files...")
        all_data = {}
        for version_name, version_info in self.versions.items():
            data = self.parse_log_file_detailed(version_info['log_file'])
            all_data[version_name] = data
            print(f"   {version_name}: {len(data)} tasks")
        
        # Analyze each version
        all_analyses = {}
        
        for version_name, data in all_data.items():
            if detailed:
                print(f"\n{'='*80}")
                print(f"📊 {version_name} COMPLETE ANALYSIS")
                print(f"{'='*80}")
            
            analysis = self.analyze_version_prompts(version_name, data)
            all_analyses[version_name] = analysis
            
            if not analysis:
                print(f"❌ No data available for {version_name}")
                continue
            
            if detailed:
                print(f"\n📈 OVERALL STATISTICS:")
                print(f"   Total Tasks: {analysis['total_tasks']}")
                print(f"   Winning Prompts: {analysis['winning_count']} ({analysis['success_rate']:.1f}%)")
                print(f"   Losing Prompts: {analysis['losing_count']}")
                print(f"   Average Score: {analysis['average_score']:.4f}")
                print(f"   Max Score: {analysis['max_score']:.4f}")
                
                if analysis['max_score_prompt']:
                    max_p = analysis['max_score_prompt']
                    print(f"   Max Score Prompt: '{max_p['prompt']}'")
                    print(f"     Alignment: {max_p.get('alignment_score', 0):.4f}")
                    print(f"     Quality: {max_p.get('quality_score', 0):.4f}")
                    print(f"     Demo Fidelity: {max_p.get('demo_fidelity_score', 0):.4f}")
                    print(f"     Task Fidelity: {max_p.get('task_fidelity_score', 0):.4f}")
                
                print(f"\n🏆 TOP 10 WINNING PROMPTS:")
                for i, task in enumerate(analysis['winning_prompts'][:10], 1):
                    score = task.get('validation_score', 0)
                    category = self.categorize_prompt(task['prompt'])
                    print(f"   {i:2d}. {task['prompt']} (Score: {score:.4f}, Category: {category})")
                
                print(f"\n❌ LOSING PROMPTS ({len(analysis['losing_prompts'])}):")
                for task in analysis['losing_prompts']:
                    category = self.categorize_prompt(task['prompt'])
                    alignment = task.get('alignment_score', 0) if task.get('alignment_score') is not None else 0.0
                    quality = task.get('quality_score', 0) if task.get('quality_score') is not None else 0.0
                    demo = task.get('demo_fidelity_score', 0) if task.get('demo_fidelity_score') is not None else 0.0
                    task_fid = task.get('task_fidelity_score', 0) if task.get('task_fidelity_score') is not None else 0.0
                    print(f"   • '{task['prompt']}' (Category: {category})")
                    print(f"     Alignment: {alignment:.4f}, Quality: {quality:.4f}, Demo: {demo:.4f}, Task: {task_fid:.4f}")
                
                print(f"\n📊 WINNING PROMPTS BY CATEGORY:")
                for category, tasks in analysis['winning_by_category'].items():
                    avg_score = sum(t.get('validation_score', 0) for t in tasks) / len(tasks)
                    print(f"   {category}: {len(tasks)} prompts (avg score: {avg_score:.4f})")
                    for task in tasks[:3]:  # Show top 3 per category
                        print(f"     • {task['prompt']} ({task.get('validation_score', 0):.4f})")
                    if len(tasks) > 3:
                        print(f"     ... and {len(tasks)-3} more")
                
                print(f"\n📊 LOSING PROMPTS BY CATEGORY:")
                for category, tasks in analysis['losing_by_category'].items():
                    print(f"   {category}: {len(tasks)} prompts")
                    for task in tasks:
                        print(f"     • {task['prompt']}")
        
        # Generate comparison summary
        print(f"\n{'='*120}")
        print(f"📊 COMPARISON SUMMARY - ALL VERSIONS")
        print(f"{'='*120}")
        
        print(f"\n📈 VERSION PERFORMANCE COMPARISON:")
        print(f"{'Version':<15} {'Total':<8} {'Winning':<8} {'Losing':<8} {'Success%':<10} {'Avg Score':<12} {'Max Score':<12} {'Max Prompt'}")
        print(f"{'-'*15} {'-'*8} {'-'*8} {'-'*8} {'-'*10} {'-'*12} {'-'*12} {'-'*50}")
        
        for version_name in self.versions.keys():
            analysis = all_analyses.get(version_name, {})
            if analysis:
                max_prompt = analysis['max_score_prompt']['prompt'][:47] + "..." if analysis['max_score_prompt'] and len(analysis['max_score_prompt']['prompt']) > 50 else analysis['max_score_prompt']['prompt'] if analysis['max_score_prompt'] else "N/A"
                print(f"{version_name:<15} {analysis['total_tasks']:<8} {analysis['winning_count']:<8} {analysis['losing_count']:<8} {analysis['success_rate']:<10.1f} {analysis['average_score']:<12.4f} {analysis['max_score']:<12.4f} {max_prompt}")
        
        # Find overall best and worst prompts
        print(f"\n🏆 OVERALL BEST PROMPTS (Across All Versions):")
        all_prompts = {}
        for version_name, analysis in all_analyses.items():
            for task in analysis.get('winning_prompts', []):
                prompt = task['prompt']
                score = task.get('validation_score', 0)
                if prompt not in all_prompts or score > all_prompts[prompt]['score']:
                    all_prompts[prompt] = {
                        'score': score,
                        'version': version_name,
                        'category': self.categorize_prompt(prompt)
                    }
        
        # Sort by score and show top 20
        sorted_prompts = sorted(all_prompts.items(), key=lambda x: x[1]['score'], reverse=True)
        print(f"\n🏆 TOP 20 OVERALL BEST PROMPTS:")
        for i, (prompt, data) in enumerate(sorted_prompts[:20], 1):
            print(f"   {i:2d}. '{prompt}' (Score: {data['score']:.4f}, Version: {data['version']}, Category: {data['category']})")
        
        # Show max scores for each version
        print(f"\n🏆 MAX SCORES BY VERSION:")
        for version_name, analysis in all_analyses.items():
            if analysis and analysis.get('max_score_prompt'):
                max_p = analysis['max_score_prompt']
                print(f"   {version_name}: {analysis['max_score']:.4f} - '{max_p['prompt']}'")
        
        # Find most problematic prompts
        print(f"\n❌ MOST PROBLEMATIC PROMPTS (Zero scores across versions):")
        zero_score_prompts = defaultdict(list)
        for version_name, analysis in all_analyses.items():
            for task in analysis.get('losing_prompts', []):
                prompt = task['prompt']
                zero_score_prompts[prompt].append(version_name)
        
        # Show prompts that failed in multiple versions
        problematic_prompts = [(prompt, versions) for prompt, versions in zero_score_prompts.items() if len(versions) > 1]
        problematic_prompts.sort(key=lambda x: len(x[1]), reverse=True)
        
        for prompt, failed_versions in problematic_prompts:
            category = self.categorize_prompt(prompt)
            print(f"   • '{prompt}' (Category: {category})")
            print(f"     Failed in: {', '.join(failed_versions)} ({len(failed_versions)} versions)")
        
        print(f"\n{'='*120}")
        print(f"MODULAR SUMMARY ANALYSIS FINISHED")
        print(f"{'='*120}")

def main():
    parser = argparse.ArgumentParser(description='Modular TRELLIS Prompt Analysis Tool')
    parser.add_argument('--versions', nargs='+', required=True, 
                       help='Version names and log files in format: name:logfile [name:logfile ...]')
    parser.add_argument('--dbs', nargs='+', 
                       help='Database files in format: name:dbfile [name:dbfile ...] (optional)')
    parser.add_argument('--detailed', action='store_true', 
                       help='Show detailed analysis for each version')
    
    args = parser.parse_args()
    
    # Parse version arguments
    analyzer = TrellisModularAnalyzer()
    
    for version_arg in args.versions:
        if ':' not in version_arg:
            print(f"Error: Version argument must be in format 'name:logfile', got: {version_arg}")
            sys.exit(1)
        
        name, log_file = version_arg.split(':', 1)
        analyzer.add_version(name, log_file)
    
    # Parse database arguments if provided
    if args.dbs:
        db_map = {}
        for db_arg in args.dbs:
            if ':' not in db_arg:
                print(f"Error: DB argument must be in format 'name:dbfile', got: {db_arg}")
                sys.exit(1)
            
            name, db_file = db_arg.split(':', 1)
            db_map[name] = db_file
        
        # Update versions with database files
        for name in analyzer.versions:
            if name in db_map:
                analyzer.versions[name]['db_file'] = db_map[name]
    
    # Generate analysis
    analyzer.generate_complete_summary(detailed=args.detailed)

if __name__ == "__main__":
    main() 