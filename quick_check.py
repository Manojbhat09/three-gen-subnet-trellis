#!/usr/bin/env python3
import os, json, glob, re
from pathlib import Path

def normalize_prompt_text(text: str) -> str:
    """Normalize prompt text by removing outer quotes"""
    if not isinstance(text, str):
        return text
    s = text.strip()
    # Remove paired double/single quotes around the entire string
    if (s.startswith("''") and s.endswith("''")) or (s.startswith('""') and s.endswith('""')):
        s = s[2:-2].strip()
    if (s.startswith("'") and s.endswith("'")) or (s.startswith('"') and s.endswith('"')):
        s = s[1:-1].strip()
    return s

def extract_prompts_from_logs():
    """Extract all unique prompts from all log files"""
    log_dir = "episodic_logs_first"
    if not os.path.exists(log_dir):
        return {}
    
    log_files = glob.glob(os.path.join(log_dir, "episodic_run_*.log"))
    if not log_files:
        return {}
    
    # Sort by modification time (newest first)
    log_files.sort(key=lambda x: os.path.getmtime(x), reverse=True)
    
    all_prompts = {}
    
    for log_file in log_files:
        try:
            with open(log_file, 'r', encoding='utf-8') as f:
                content = f.read()
            
            lines = content.split('\n')
            current_prompt = None
            current_score = None
            current_optimized = None
            
            for line in lines:
                # Find "Original:" lines to get the prompt
                if 'Original:' in line:
                    current_prompt = normalize_prompt_text(line.split('Original:')[1].strip())
                    current_score = None
                    current_optimized = None
                
                # Find "Optimized:" lines to get the optimized version
                elif 'Optimized:' in line and 'wbgmsst,' in line:
                    if current_prompt:
                        optimized_text = line.split('Optimized:')[1].strip()
                        if optimized_text.startswith('wbgmsst,'):
                            current_optimized = optimized_text[8:].strip()
                        else:
                            current_optimized = optimized_text
                
                # Find "Validation score:" lines to get the score
                elif 'Validation score:' in line:
                    score_match = re.search(r'Validation score: ([\d.]+)', line)
                    if score_match and current_prompt:
                        current_score = float(score_match.group(1))
                        
                        # Create or update prompt data
                        if current_prompt not in all_prompts:
                            all_prompts[current_prompt] = {
                                'original_prompt': current_prompt,
                                'optimized_prompt': current_optimized or current_prompt,
                                'best_score': current_score,
                                'source': 'logs',
                                'log_file': os.path.basename(log_file)
                            }
                        else:
                            # Update with better score if found
                            existing_data = all_prompts[current_prompt]
                            if current_score > existing_data.get('best_score', 0.0):
                                existing_data['best_score'] = current_score
                                if current_optimized:
                                    existing_data['optimized_prompt'] = current_optimized
                                existing_data['log_file'] = os.path.basename(log_file)
                                
        except Exception as e:
            print(f"Error reading {log_file}: {e}")
            continue
    
    return all_prompts

def load_episodic_memory():
    """Load prompts from episodic memory with proper structure parsing"""
    mem_path = Path('episodic_logs_first/episodic_memory.json')
    if not mem_path.exists():
        return {}
    
    try:
        with mem_path.open('r', encoding='utf-8') as f:
            data = json.load(f)
        
        memory_prompts = {}
        
        # Handle different possible structures
        if isinstance(data, dict):
            if 'gold_standard_results' in data and isinstance(data['gold_standard_results'], dict):
                # Standard structure
                for prompt, prompt_data in data['gold_standard_results'].items():
                    if isinstance(prompt_data, dict):
                        # Extract the best score and optimized prompt
                        best_score = 0.0
                        optimized_prompt = prompt
                        
                        if 'method_2_hybrid_example' in prompt_data:
                            method_data = prompt_data['method_2_hybrid_example']
                            optimized_prompt = method_data.get('optimized_prompt', prompt)
                            best_score = method_data.get('validation_results', {}).get('validation_engine_score', 0.0)
                        elif 'validation_results' in prompt_data:
                            best_score = prompt_data['validation_results'].get('validation_engine_score', 0.0)
                            optimized_prompt = prompt_data.get('optimized_prompt', prompt)
                        
                        memory_prompts[prompt] = {
                            'original_prompt': prompt,
                            'optimized_prompt': optimized_prompt,
                            'best_score': best_score,
                            'source': 'episodic_memory'
                        }
            else:
                # Try to find prompt-like entries
                for key, value in data.items():
                    if isinstance(value, dict):
                        # Look for validation results or optimization data
                        if any(k in value for k in ['validation_results', 'method_2_hybrid_example', 'optimized_prompt']):
                            best_score = 0.0
                            optimized_prompt = key
                            
                            if 'method_2_hybrid_example' in value:
                                method_data = value['method_2_hybrid_example']
                                optimized_prompt = method_data.get('optimized_prompt', key)
                                best_score = method_data.get('validation_results', {}).get('validation_engine_score', 0.0)
                            elif 'validation_results' in value:
                                best_score = value['validation_results'].get('validation_engine_score', 0.0)
                                optimized_prompt = value.get('optimized_prompt', key)
                            
                            memory_prompts[key] = {
                                'original_prompt': key,
                                'optimized_prompt': optimized_prompt,
                                'best_score': best_score,
                                'source': 'episodic_memory'
                            }
        
        return memory_prompts
        
    except Exception as e:
        print(f"Failed to load episodic_memory.json: {e}")
        return {}

def main():
    print("🔍 Analyzing prompt counts from logs and episodic memory...")
    
    # Load from logs
    log_prompts = extract_prompts_from_logs()
    print(f"📖 Prompts from logs: {len(log_prompts)}")
    
    # Load from episodic memory
    memory_prompts = load_episodic_memory()
    print(f"📚 Prompts from episodic memory: {len(memory_prompts)}")
    
    # Combine and find best scores
    combined_prompts = {}
    
    # Start with memory prompts
    for prompt, data in memory_prompts.items():
        combined_prompts[prompt] = data.copy()
    
    # Merge with log prompts, keeping best scores
    for prompt, data in log_prompts.items():
        if prompt in combined_prompts:
            # Check if log data has better score
            existing_score = combined_prompts[prompt].get('best_score', 0.0)
            log_score = data.get('best_score', 0.0)
            
            if log_score > existing_score:
                # Update with better score from logs
                combined_prompts[prompt].update(data)
                combined_prompts[prompt]['source'] = 'logs'
        else:
            # New prompt from logs
            combined_prompts[prompt] = data
    
    print(f"🔄 Combined unique prompts: {len(combined_prompts)}")
    
    # Show score statistics
    if combined_prompts:
        scores = [data.get('best_score', 0.0) for data in combined_prompts.values()]
        avg_score = sum(scores) / len(scores)
        max_score = max(scores)
        min_score = min(scores)
        gold_count = sum(1 for s in scores if s >= 0.75)
        
        print(f"\n📊 Score Statistics:")
        print(f"   Average: {avg_score:.4f}")
        print(f"   Highest: {max_score:.4f}")
        print(f"   Lowest: {min_score:.4f}")
        print(f"   Gold prompts (≥0.75): {gold_count}")
        
        # Show top 5 prompts
        top_prompts = sorted(combined_prompts.items(), key=lambda x: x[1].get('best_score', 0.0), reverse=True)[:5]
        print(f"\n🏆 Top 5 scoring prompts:")
        for i, (prompt, data) in enumerate(top_prompts, 1):
            score = data.get('best_score', 0.0)
            source = data.get('source', 'unknown')
            print(f"   {i}. Score {score:.4f} ({source}): '{prompt[:60]}...'")

if __name__ == "__main__":
    main()
