#!/usr/bin/env python3
"""
Quick Fix for Similarity Threshold Issue

This script helps you quickly fix the similarity threshold issue in your orchestrator
by showing you exactly what to change and where.
"""

import os
import re

def find_orchestrator_files():
    """Find all orchestrator files that need the threshold updated."""
    print("🔍 SEARCHING FOR ORCHESTRATOR FILES")
    print("=" * 50)
    
    orchestrator_files = []
    
    # Common orchestrator file patterns
    patterns = [
        "continuous_trellis_orchestrator*.py",
        "*orchestrator*.py"
    ]
    
    for pattern in patterns:
        import glob
        files = glob.glob(pattern)
        orchestrator_files.extend(files)
    
    # Remove duplicates and sort
    orchestrator_files = sorted(list(set(orchestrator_files)))
    
    print(f"Found {len(orchestrator_files)} orchestrator files:")
    for i, file in enumerate(orchestrator_files, 1):
        print(f"  {i}. {file}")
    
    return orchestrator_files

def check_current_thresholds(file_path: str):
    """Check what similarity thresholds are currently set in a file."""
    print(f"\n📋 CHECKING: {file_path}")
    print("-" * 40)
    
    try:
        with open(file_path, 'r') as f:
            content = f.read()
        
        # Look for similarity threshold configurations
        threshold_patterns = [
            r'reproducibility_similarity_threshold.*?(\d+\.\d+)',
            r'clip_similarity_threshold.*?(\d+\.\d+)',
            r'reproducibility-similarity.*?(\d+\.\d+)',
            r'default=(\d+\.\d+).*?reproducibility',
            r'default=(\d+\.\d+).*?similarity'
        ]
        
        found_thresholds = []
        for pattern in threshold_patterns:
            matches = re.findall(pattern, content, re.IGNORECASE)
            for match in matches:
                found_thresholds.append(float(match))
        
        if found_thresholds:
            print(f"   Current thresholds found:")
            for threshold in found_thresholds:
                print(f"     {threshold:.3f}")
        else:
            print(f"   No explicit thresholds found (using defaults)")
        
        # Check for hardcoded 0.51 values
        hardcoded_51 = content.count('0.51')
        if hardcoded_51 > 0:
            print(f"   ⚠️  Found {hardcoded_51} hardcoded 0.51 values")
        
        return found_thresholds
        
    except Exception as e:
        print(f"   ❌ Error reading file: {e}")
        return []

def show_fix_instructions():
    """Show detailed fix instructions."""
    print("\n" + "=" * 60)
    print("🔧 FIX INSTRUCTIONS")
    print("=" * 60)
    
    print("The issue is that your similarity threshold (0.51) is too high.")
    print("Based on the analysis, the optimal threshold should be 0.42.")
    print("\nHere's how to fix it:")
    
    print("\n1️⃣ COMMAND LINE ARGUMENT (Easiest):")
    print("   When running your orchestrator, use:")
    print("   --reproducibility-similarity 0.42")
    print("   --clip-similarity-threshold 0.42")
    
    print("\n2️⃣ CONFIGURATION FILE:")
    print("   Update your config dictionary:")
    print("   'reproducibility_similarity_threshold': 0.42")
    print("   'clip_similarity_threshold': 0.42")
    
    print("\n3️⃣ HARDCODED VALUES:")
    print("   Search for '0.51' in your orchestrator files and replace with '0.42'")
    
    print("\n4️⃣ RECOMMENDED THRESHOLDS:")
    print("   🎯 High precision (few false positives): 0.45-0.50")
    print("   🎯 Balanced precision/recall: 0.40-0.45")
    print("   🎯 High recall (catch more matches): 0.35-0.40")
    print("   🏆 RECOMMENDED: 0.42 (balanced approach)")
    
    print("\n5️⃣ EXPECTED RESULTS:")
    print("   Current threshold 0.51: 0% match rate")
    print("   New threshold 0.42: ~42% match rate")
    print("   This will significantly improve your reproducibility optimization!")

def create_fix_script():
    """Create a simple script to help with the fix."""
    fix_script = """#!/bin/bash
# Quick Fix Script for Similarity Threshold
# Run this to update your orchestrator configuration

echo "🔧 UPDATING SIMILARITY THRESHOLDS..."

# Find all orchestrator files
files=$(find . -name "continuous_trellis_orchestrator*.py" -o -name "*orchestrator*.py")

for file in $files; do
    if [ -f "$file" ]; then
        echo "Processing: $file"
        
        # Replace hardcoded 0.51 values with 0.42
        sed -i 's/0\.51/0.42/g' "$file"
        
        # Also replace any other common thresholds
        sed -i 's/reproducibility_similarity_threshold.*?0\.5/reproducibility_similarity_threshold: 0.42/g' "$file"
        sed -i 's/clip_similarity_threshold.*?0\.5/clip_similarity_threshold: 0.42/g' "$file"
        
        echo "  ✅ Updated $file"
    fi
done

echo "🎉 Threshold update complete!"
echo "💡 Remember to also update command line arguments when running:"
echo "   --reproducibility-similarity 0.42"
echo "   --clip-similarity-threshold 0.42"
"""
    
    with open("fix_thresholds.sh", "w") as f:
        f.write(fix_script)
    
    os.chmod("fix_thresholds.sh", 0o755)
    print(f"\n📝 Created fix script: fix_thresholds.sh")
    print(f"   Run: ./fix_thresholds.sh to automatically update thresholds")

def main():
    """Main function."""
    print("🔧 QUICK FIX FOR SIMILARITY THRESHOLD ISSUE")
    print("=" * 60)
    
    print("❌ PROBLEM IDENTIFIED:")
    print("   Your similarity threshold (0.51) is too high")
    print("   This causes 0% match rate for gold prompts")
    print("   Result: Reproducibility optimization always fails")
    
    print("\n✅ SOLUTION:")
    print("   Lower threshold to 0.42 for optimal performance")
    print("   This will give you ~42% match rate")
    
    # Find orchestrator files
    orchestrator_files = find_orchestrator_files()
    
    # Check current thresholds
    print("\n🔍 CURRENT THRESHOLD ANALYSIS")
    print("=" * 50)
    
    for file in orchestrator_files:
        check_current_thresholds(file)
    
    # Show fix instructions
    show_fix_instructions()
    
    # Create fix script
    create_fix_script()
    
    print("\n" + "=" * 60)
    print("🚀 NEXT STEPS")
    print("=" * 60)
    print("1. Run the fix script: ./fix_thresholds.sh")
    print("2. Update command line arguments when running orchestrator")
    print("3. Test with a few prompts to verify the fix")
    print("4. Monitor logs to see reproducibility optimization working")
    
    print("\n💡 The fix will enable your reproducibility system to:")
    print("   - Find close gold prompts in episodic memory")
    print("   - Use proven optimization strategies")
    print("   - Improve generation quality and consistency")
    print("   - Reduce reliance on traditional LLM optimization")

if __name__ == "__main__":
    main()
