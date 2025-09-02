#!/usr/bin/env python3
"""
Script to load episodic memory, run validation for each prompt, and create comparison table.
"""

import json
import subprocess
import os
import time
import re
from typing import Dict, List, Tuple, Optional
import argparse

def load_episodic_memory(file_path: str) -> Dict:
    """Load episodic memory from JSON file."""
    try:
        with open(file_path, 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"❌ Error loading episodic memory: {e}")
        return None

def extract_prompt_data(episodic_memory: Dict) -> List[Dict]:
    """Extract original prompt, best optimized prompt, and best score from episodic memory."""
    prompt_data = []
    
    if 'optimization_sessions' not in episodic_memory:
        print("❌ No optimization sessions found in episodic memory")
        return prompt_data
    
    for session in episodic_memory['optimization_sessions']:
        session_data = {
            'session_id': session.get('session_id', 'unknown'),
            'original_prompt': session.get('original_prompt', ''),
            'final_best_prompt': session.get('final_best_prompt', ''),
            'final_best_score': session.get('final_best_score', 0.0),
            'total_rounds': session.get('total_rounds', 0)
        }
        prompt_data.append(session_data)
    
    return prompt_data

def run_validation_command(original_prompt: str, optimized_prompt: str, config: str = "GOOD short") -> Optional[Dict]:
    """Run validation command and extract results."""
    print(f"\n🔍 Running validation for:")
    print(f"   Original: {original_prompt}")
    print(f"   Optimized: {optimized_prompt}")
    print(f"   Config: {config}")
    
    # Build command
    cmd = [
        "python", "test_grid_flow_endpoint_validate.py",
        "--validate",
        "--config", config,
        "--base_prompt", f"\"{original_prompt}\"",
        "--optimized-prompt", f"\"{optimized_prompt}\""
    ]
    
    print(f"🚀 Running command: {' '.join(cmd)}")
    
    try:
        # Run validation
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=True,
            timeout=600,  # 10 minute timeout
            cwd=os.getcwd()
        )
        
        print(f"✅ Validation completed successfully!")
        
        # Parse output to extract validation scores
        validation_results = parse_validation_output(result.stdout)
        
        if validation_results:
            print(f"📊 Validation Results:")
            print(f"   Validation Engine Score: {validation_results.get('validation_engine_score', 'N/A')}")
            print(f"   Alignment Score: {validation_results.get('alignment_score', 'N/A')}")
            print(f"   Quality Score: {validation_results.get('quality_score', 'N/A')}")
            print(f"   Demo Fidelity Score: {validation_results.get('demo_fidelity_score', 'N/A')}")
            print(f"   Validation Passed: {validation_results.get('validation_passed', 'N/A')}")
        
        return validation_results
        
    except subprocess.TimeoutExpired:
        print(f"❌ Validation timed out after 10 minutes")
        return None
    except subprocess.CalledProcessError as e:
        print(f"❌ Validation failed with exit code {e.returncode}")
        print(f"   stdout: {e.stdout}")
        print(f"   stderr: {e.stderr}")
        return None
    except Exception as e:
        print(f"❌ Validation error: {e}")
        return None

def parse_validation_output(output: str) -> Optional[Dict]:
    """Parse validation output to extract scores."""
    results = {}
    
    # Extract validation engine score
    engine_match = re.search(r'🏆 Validation Engine Score: ([\d.]+)', output)
    if engine_match:
        results['validation_engine_score'] = float(engine_match.group(1))
    
    # Extract alignment score
    alignment_match = re.search(r'🤝 Alignment Score: ([\d.]+)', output)
    if alignment_match:
        results['alignment_score'] = float(alignment_match.group(1))
    
    # Extract quality score
    quality_match = re.search(r'💎 Quality Score: ([\d.]+)', output)
    if quality_match:
        results['quality_score'] = float(quality_match.group(1))
    
    # Extract demo fidelity score
    fidelity_match = re.search(r'🎭 Demo Fidelity Score: ([\d.]+)', output)
    if fidelity_match:
        results['demo_fidelity_score'] = float(fidelity_match.group(1))
    
    # Extract validation passed
    passed_match = re.search(r'✅ Validation Passed: (True|False)', output)
    if passed_match:
        results['validation_passed'] = passed_match.group(1) == 'True'
    
    return results if results else None

def create_comparison_table(prompt_data: List[Dict], validation_results: List[Dict]) -> str:
    """Create a formatted comparison table."""
    table = []
    table.append("=" * 200)
    table.append(f"{'Original Prompt':<25} {'Optimized Prompt':<40} {'Best Score':<12} {'Demo Fidelity':<15} {'Validation Engine':<18} {'Alignment':<12} {'Quality':<12} {'Status':<10}")
    table.append("=" * 200)
    
    for i, (prompt_info, validation_info) in enumerate(zip(prompt_data, validation_results)):
        original = prompt_info['original_prompt'][:22] + "..." if len(prompt_info['original_prompt']) > 25 else prompt_info['original_prompt']
        optimized = prompt_info['final_best_prompt'][:37] + "..." if len(prompt_info['final_best_prompt']) > 40 else prompt_info['final_best_prompt']
        best_score = f"{prompt_info['final_best_score']:.4f}"
        
        if validation_info:
            demo_fidelity = f"{validation_info.get('demo_fidelity_score', 0.0):.4f}"
            validation_engine = f"{validation_info.get('validation_engine_score', 0.0):.4f}"
            alignment_score = f"{validation_info.get('alignment_score', 0.0):.4f}"
            quality_score = f"{validation_info.get('quality_score', 0.0):.4f}"
            status = "✅ PASS" if validation_info.get('validation_passed', False) else "❌ FAIL"
        else:
            demo_fidelity = "N/A"
            validation_engine = "N/A"
            alignment_score = "N/A"
            quality_score = "N/A"
            status = "ERROR"
        
        table.append(f"{original:<25} {optimized:<40} {best_score:<12} {demo_fidelity:<15} {validation_engine:<18} {alignment_score:<12} {quality_score:<12} {status:<10}")
    
    table.append("=" * 200)
    return "\n".join(table)

def save_results_to_file(prompt_data: List[Dict], validation_results: List[Dict], output_file: str):
    """Save results to a JSON file."""
    results_data = []
    
    for i, (prompt_info, validation_info) in enumerate(zip(prompt_data, validation_results)):
        result_entry = {
            'session_id': prompt_info['session_id'],
            'original_prompt': prompt_info['original_prompt'],
            'final_best_prompt': prompt_info['final_best_prompt'],
            'final_best_score': prompt_info['final_best_score'],
            'total_rounds': prompt_info['total_rounds'],
            'validation_results': validation_info or {},
            'timestamp': time.time()
        }
        results_data.append(result_entry)
    
    try:
        with open(output_file, 'w') as f:
            json.dump(results_data, f, indent=2)
        print(f"💾 Results saved to: {output_file}")
    except Exception as e:
        print(f"❌ Error saving results: {e}")

def main():
    parser = argparse.ArgumentParser(description='Run validation for episodic memory prompts')
    parser.add_argument('--memory-file', default='episodic_logs_cinema/episodic_memory.json', 
                       help='Path to episodic memory JSON file')
    parser.add_argument('--config', default='GOOD short', 
                       help='Configuration to use for validation')
    parser.add_argument('--output-file', default='validation_results.json', 
                       help='Output file to save results')
    parser.add_argument('--skip-validation', action='store_true', 
                       help='Skip running validation (just load and display data)')
    
    args = parser.parse_args()
    
    print("🚀 Episodic Memory Validation Script")
    print("=" * 50)
    
    # Load episodic memory
    print(f"📁 Loading episodic memory from: {args.memory_file}")
    episodic_memory = load_episodic_memory(args.memory_file)
    
    if not episodic_memory:
        print("❌ Failed to load episodic memory")
        return
    
    # Extract prompt data
    prompt_data = extract_prompt_data(episodic_memory)
    print(f"✅ Loaded {len(prompt_data)} optimization sessions")
    
    # Display loaded data
    print(f"\n📊 Loaded Prompt Data:")
    print("=" * 80)
    for i, data in enumerate(prompt_data):
        print(f"{i+1}. Session: {data['session_id']}")
        print(f"   Original: {data['original_prompt']}")
        print(f"   Best: {data['final_best_prompt']}")
        print(f"   Score: {data['final_best_score']:.4f} (Rounds: {data['total_rounds']})")
        print()
    
    if args.skip_validation:
        print("⏭️ Skipping validation as requested")
        return
    
    # Run validation for each prompt
    print(f"\n🔍 Running validation for all prompts using config: {args.config}")
    print("=" * 80)
    
    validation_results = []
    for i, prompt_info in enumerate(prompt_data):
        print(f"\n🔄 Processing {i+1}/{len(prompt_data)}: {prompt_info['original_prompt'][:50]}...")
        
        # Run validation
        validation_result = run_validation_command(
            prompt_info['original_prompt'],
            prompt_info['final_best_prompt'],
            args.config
        )
        
        validation_results.append(validation_result)
        
        # Small delay between validations
        if i < len(prompt_data) - 1:
            time.sleep(2)
    
    # Create comparison table
    print(f"\n📋 VALIDATION COMPARISON TABLE")
    print("=" * 80)
    comparison_table = create_comparison_table(prompt_data, validation_results)
    print(comparison_table)
    
    # Save results
    save_results_to_file(prompt_data, validation_results, args.output_file)
    
    # Summary statistics
    print(f"\n📈 SUMMARY STATISTICS")
    print("=" * 50)
    
    successful_validations = sum(1 for r in validation_results if r and r.get('validation_passed', False))
    total_validations = len(validation_results)
    
    print(f"Total prompts processed: {total_validations}")
    print(f"Successful validations: {successful_validations}")
    print(f"Success rate: {successful_validations/total_validations*100:.1f}%")
    
    if validation_results:
        demo_fidelity_scores = [r.get('demo_fidelity_score', 0.0) for r in validation_results if r]
        validation_engine_scores = [r.get('validation_engine_score', 0.0) for r in validation_results if r]
        alignment_scores = [r.get('alignment_score', 0.0) for r in validation_results if r]
        quality_scores = [r.get('quality_score', 0.0) for r in validation_results if r]
        
        if demo_fidelity_scores:
            print(f"Average Demo Fidelity: {sum(demo_fidelity_scores)/len(demo_fidelity_scores):.4f}")
        if validation_engine_scores:
            print(f"Average Validation Engine: {sum(validation_engine_scores)/len(validation_engine_scores):.4f}")
        if alignment_scores:
            print(f"Average Alignment Score: {sum(alignment_scores)/len(alignment_scores):.4f}")
        if quality_scores:
            print(f"Average Quality Score: {sum(quality_scores)/len(quality_scores):.4f}")
    
    print(f"\n✅ Validation script completed!")

if __name__ == "__main__":
    main()
