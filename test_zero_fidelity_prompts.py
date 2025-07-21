#!/usr/bin/env python3
"""
Zero-Fidelity Prompt Tester
Purpose: Test a sample of prompts from the optimization framework to verify
         they still produce low validation scores with our working system.
"""
import subprocess
import sys
import json
import time
from pathlib import Path

# Sample of problematic prompts from the framework document
# Starting with a mix of Critical and Medium risk prompts
TEST_PROMPTS = [
    # Critical Risk Prompts
    "opal necklace featuring central teardrop gem",
    "necklace with heart-shaped pendant made of silver and turquoise stones",  
    "ivory-colored ornate chisel for fine woodworking",
    "golden chalice adorned with delicate filigree",
    "shiny garnet gemstone exudes rich red color",
    
    # Medium Risk Prompts  
    "crystal-clear domes reflect moonlight softly",
    "silver chalice with leafy vine pattern",
    "smooth pink gem with hole in middle",
    "metallic blue robot wearing sunflower crown",
    "crystal staff with swirling light",
    
    # Low Risk Prompts (should score better)
    "modern plastic bottle with blue cap",
    "smooth green olive in purple bowl", 
    "blue bow with gold arrows",
    "glass jug filled juice",
    
    # Zero Risk Prompts (should score well)
    "small round blue creature with long nose and pointed ears",
    "blue and white race car",
    "small yellow triangle lamp",
    "brown paper bag of groceries"
]

def run_validation(prompt):
    """Run validation on a single prompt using our working script"""
    try:
        print(f"\n🧪 Testing: '{prompt}'")
        print("-" * 60)
        
        # Run the simple_local_validator.py script
        result = subprocess.run([
            'python3', 'simple_local_validator.py', prompt
        ], capture_output=True, text=True, timeout=180)
        
        if result.returncode != 0:
            print(f"❌ Validation failed with return code {result.returncode}")
            print(f"STDERR: {result.stderr}")
            return None
        
        # Parse the output to extract the final score
        output_lines = result.stdout.split('\n')
        final_score = None
        
        for line in output_lines:
            if '🏆 Final Score:' in line:
                try:
                    score_str = line.split('🏆 Final Score:')[1].strip()
                    final_score = float(score_str)
                    break
                except (IndexError, ValueError):
                    continue
        
        if final_score is not None:
            print(f"✅ Score: {final_score:.4f}")
            if final_score > 0.75:
                print("🌟 EXCELLENT - Above optimization target!")
            elif final_score > 0.6:
                print("✅ GOOD - Above network threshold")
            elif final_score > 0.3:
                print("⚠️ FAIR - Below network threshold")
            else:
                print("❌ POOR - Very low score")
            return final_score
        else:
            print("❌ Could not parse final score from output")
            return None
            
    except subprocess.TimeoutExpired:
        print("❌ Validation timed out (>180s)")
        return None
    except Exception as e:
        print(f"❌ Validation error: {e}")
        return None

def main():
    print("🔬 Zero-Fidelity Prompt Testing")
    print("=" * 60)
    print(f"Testing {len(TEST_PROMPTS)} prompts to verify low-scoring patterns...")
    print("Target: Score > 0.75 for optimization")
    print("Network threshold: Score > 0.6")
    
    results = []
    
    for i, prompt in enumerate(TEST_PROMPTS, 1):
        print(f"\n[{i}/{len(TEST_PROMPTS)}]", end="")
        score = run_validation(prompt)
        
        result = {
            'prompt': prompt,
            'score': score,
            'timestamp': time.time()
        }
        results.append(result)
        
        # Small delay to prevent overwhelming the system
        if i < len(TEST_PROMPTS):
            print("⏳ Waiting 2s before next test...")
            time.sleep(2)
    
    # Summary Analysis
    print("\n" + "=" * 80)
    print("📊 TESTING SUMMARY")
    print("=" * 80)
    
    valid_results = [r for r in results if r['score'] is not None]
    
    if not valid_results:
        print("❌ No valid results obtained")
        return
    
    # Categorize results
    excellent = [r for r in valid_results if r['score'] > 0.75]
    good = [r for r in valid_results if 0.6 < r['score'] <= 0.75]
    fair = [r for r in valid_results if 0.3 < r['score'] <= 0.6]
    poor = [r for r in valid_results if r['score'] <= 0.3]
    
    print(f"📈 Results Distribution:")
    print(f"   🌟 Excellent (>0.75): {len(excellent)}/{len(valid_results)}")
    print(f"   ✅ Good (0.6-0.75): {len(good)}/{len(valid_results)}")
    print(f"   ⚠️ Fair (0.3-0.6): {len(fair)}/{len(valid_results)}")
    print(f"   ❌ Poor (<0.3): {len(poor)}/{len(valid_results)}")
    
    avg_score = sum(r['score'] for r in valid_results) / len(valid_results)
    print(f"📊 Average Score: {avg_score:.4f}")
    
    # Show worst performers for analysis
    if poor or fair:
        print(f"\n🔍 PROMPTS NEEDING OPTIMIZATION:")
        optimization_candidates = sorted(poor + fair, key=lambda x: x['score'])
        
        for result in optimization_candidates[:10]:  # Show top 10 worst
            score = result['score']
            prompt = result['prompt']
            print(f"   Score: {score:.4f} | Prompt: '{prompt}'")
    
    # Save results for further analysis
    output_file = "zero_fidelity_test_results.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n💾 Results saved to: {output_file}")
    print("\n🎯 Next Steps:")
    print("   1. Analyze patterns in low-scoring prompts")
    print("   2. Develop optimization strategies")
    print("   3. Implement automated prompt improvement")

if __name__ == "__main__":
    main() 