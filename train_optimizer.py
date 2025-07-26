#!/usr/bin/env python3
"""
Training Script - Golden Examples Extractor
===========================================
🔬 Research Lab: Extracts golden examples from training runs
📊 Analyzes checkpoints and successful patterns
💎 Produces golden_examples.json for inference script

This script processes your existing training data to create the
knowledge base for the fast inference optimizer.
"""

import json
import pickle
from pathlib import Path
from typing import List, Dict
from dataclasses import asdict
import statistics

def extract_golden_examples_from_checkpoints(checkpoint_dir: str = "rl_checkpoints_v3") -> List[Dict]:
    """Extract golden examples from training checkpoints"""
    
    checkpoint_path = Path(checkpoint_dir)
    golden_examples = []
    
    print(f"🔍 Scanning checkpoints in {checkpoint_path}")
    
    if not checkpoint_path.exists():
        print(f"❌ Checkpoint directory not found: {checkpoint_path}")
        return []
    
    # Scan all checkpoint directories
    for checkpoint_folder in checkpoint_path.iterdir():
        if checkpoint_folder.is_dir():
            training_state_file = checkpoint_folder / 'training_state.json'
            agent_checkpoint_file = checkpoint_folder / 'agent_checkpoint.pth'
            
            if training_state_file.exists():
                try:
                    with open(training_state_file, 'r') as f:
                        training_data = json.load(f)
                    
                    print(f"   📂 Processing {checkpoint_folder.name}")
                    
                    # Try to load agent metadata (contains LLaMA examples)
                    try:
                        import torch
                        if agent_checkpoint_file.exists():
                            checkpoint = torch.load(agent_checkpoint_file, map_location='cpu', weights_only=False)
                            metadata = checkpoint.get('metadata', {})
                            
                            # Extract successful examples
                            successful_examples = metadata.get('llama_successful_examples', [])
                            for example in successful_examples:
                                if example.get('score', 0) >= 0.85:  # High score threshold
                                    golden_examples.append({
                                        'original': example.get('original', ''),
                                        'optimized': example.get('custom', ''),
                                        'score': example.get('score', 0),
                                        'strategy': example.get('strategy', 'unknown'),
                                        'checkpoint': checkpoint_folder.name
                                    })
                    except Exception as e:
                        print(f"     ⚠️ Could not load agent metadata: {e}")
                
                except Exception as e:
                    print(f"     ❌ Error processing {checkpoint_folder.name}: {e}")
    
    return golden_examples

def categorize_prompts(examples: List[Dict]) -> List[Dict]:
    """Categorize prompts by object type"""
    
    for example in examples:
        original = example['original'].lower()
        
        if any(word in original for word in ['glass', 'drink', 'beverage', 'lemonade', 'juice', 'wine', 'coffee', 'tea']):
            example['category'] = 'beverages'
        elif any(word in original for word in ['cupcake', 'cake', 'food', 'bread', 'cookie', 'pie']):
            example['category'] = 'food'
        elif any(word in original for word in ['pendant', 'necklace', 'ring', 'jewelry', 'gemstone', 'diamond', 'sapphire', 'emerald']):
            example['category'] = 'jewelry'
        elif any(word in original for word in ['spear', 'sword', 'weapon', 'knife', 'blade', 'staff']):
            example['category'] = 'weapons'
        elif any(word in original for word in ['creature', 'character', 'being', 'figure']):
            example['category'] = 'characters'
        elif any(word in original for word in ['fabric', 'silk', 'cloth', 'textile']):
            example['category'] = 'textiles'
        elif any(word in original for word in ['crystal', 'sphere', 'prism']):
            example['category'] = 'crystal'
        elif any(word in original for word in ['harp', 'guitar', 'instrument']):
            example['category'] = 'instruments'
        else:
            example['category'] = 'misc'
    
    return examples

def extract_optimization_principles(examples: List[Dict]) -> List[Dict]:
    """Extract optimization principles from examples"""
    
    for example in examples:
        original = example['original']
        optimized = example['optimized']
        
        # Simple principle extraction based on pattern analysis
        principle = "Enhanced with quality and detail improvements"
        
        optimized_lower = optimized.lower()
        
        if 'crystal-clear' in optimized_lower or 'pristine' in optimized_lower:
            principle = "Enhanced with clarity and transparency details"
        elif 'artisanal' in optimized_lower or 'handcrafted' in optimized_lower:
            principle = "Added artisanal craftsmanship and quality details"
        elif 'precision' in optimized_lower or 'engineered' in optimized_lower:
            principle = "Enhanced with precision and technical quality"
        elif 'luxury' in optimized_lower or 'premium' in optimized_lower:
            principle = "Added luxury materials and premium quality"
        elif 'masterwork' in optimized_lower or 'exquisite' in optimized_lower:
            principle = "Enhanced with masterwork craftsmanship details"
        
        example['principle'] = principle
    
    return examples

def filter_best_examples(examples: List[Dict], max_per_category: int = 3) -> List[Dict]:
    """Filter to get the best examples per category"""
    
    categories = {}
    for example in examples:
        category = example.get('category', 'misc')
        if category not in categories:
            categories[category] = []
        categories[category].append(example)
    
    # Sort each category by score and take top examples
    filtered = []
    for category, cat_examples in categories.items():
        sorted_examples = sorted(cat_examples, key=lambda x: x['score'], reverse=True)
        filtered.extend(sorted_examples[:max_per_category])
    
    return filtered

def create_golden_examples_file(output_file: str = "golden_examples.json"):
    """Main function to create golden examples file"""
    
    print("🔬 EXTRACTING GOLDEN EXAMPLES FROM TRAINING DATA")
    print("=" * 50)
    
    # Extract examples from checkpoints
    examples = extract_golden_examples_from_checkpoints()
    
    if not examples:
        print("❌ No examples found in checkpoints")
        print("💡 Make sure you have run training and have checkpoints in rl_checkpoints_v3/")
        return None
    
    print(f"📊 Found {len(examples)} high-scoring examples")
    
    # Categorize prompts
    examples = categorize_prompts(examples)
    
    # Extract principles
    examples = extract_optimization_principles(examples)
    
    # Filter to best examples
    golden_examples = filter_best_examples(examples, max_per_category=3)
    
    print(f"💎 Selected {len(golden_examples)} golden examples")
    
    # Show summary by category
    categories = {}
    for ex in golden_examples:
        cat = ex['category']
        if cat not in categories:
            categories[cat] = []
        categories[cat].append(ex)
    
    print("\n📈 GOLDEN EXAMPLES BY CATEGORY:")
    for category, cat_examples in categories.items():
        print(f"   {category}: {len(cat_examples)} examples (avg score: {statistics.mean([ex['score'] for ex in cat_examples]):.3f})")
    
    # Convert to the format expected by inference script
    golden_data = []
    for ex in golden_examples:
        golden_data.append({
            'original': ex['original'],
            'optimized': ex['optimized'],
            'score': ex['score'],
            'category': ex['category'],
            'principle': ex['principle']
        })
    
    # Save to file
    with open(output_file, 'w') as f:
        json.dump(golden_data, f, indent=2)
    
    print(f"\n💾 Golden examples saved to: {output_file}")
    print(f"✅ Ready for inference script!")
    
    return output_file

if __name__ == "__main__":
    create_golden_examples_file() 