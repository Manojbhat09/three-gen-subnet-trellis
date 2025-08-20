#!/bin/bash
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
