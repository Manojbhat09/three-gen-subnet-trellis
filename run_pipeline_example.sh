#!/bin/bash
# Example usage scripts for Multi-GPU Pipeline Wrapper

echo "🚀 Multi-GPU Pipeline Examples"
echo "=============================="

# Example 1: Image Ranking → PLY Pipeline
echo "📊 Example 1: Image Ranking → PLY Pipeline"
echo "Generates 8 images, ranks by CLIP, generates PLY from best images"
python gpu_multi_pipeline_wrapper.py \
    --prompt "a vintage red bicycle with chrome details" \
    --pipeline image_ranking \
    --num-inference-steps 25 \
    --guidance-scale 7.5

echo ""
echo "🔄 Example 2: Single Image → Multi PLY Pipeline"
echo "Finds best image, generates 8 PLY variations from it"
python gpu_multi_pipeline_wrapper.py \
    --prompt "a ceramic coffee mug with intricate patterns" \
    --pipeline single_image \
    --num-inference-steps 30 \
    --guidance-scale 8.0

echo ""
echo "🎯 Example 3: Both Pipelines"
echo "Runs both pipeline types for comparison"
python gpu_multi_pipeline_wrapper.py \
    --prompt "a wooden chess piece with detailed carving" \
    --pipeline both \
    --num-inference-steps 25 \
    --guidance-scale 7.5

echo ""
echo "🔍 Example 4: Just check GPU status"
python gpu_multi_pipeline_wrapper.py \
    --check-status-only

echo ""
echo "⚡ Example 5: Run with existing servers"
echo "Assumes GPU servers are already running"
python gpu_multi_pipeline_wrapper.py \
    --prompt "a futuristic lamp with LED accents" \
    --pipeline both \
    --skip-startup

echo ""
echo "🧪 Example 6: Run comprehensive test suite"
python test_multi_gpu_pipeline.py

echo ""
echo "✅ Examples complete! Check output directories for results:"
echo "   • ./gpu_pipeline_outputs/"
echo "   • ./test_pipeline_outputs/"
echo "   • ./performance_test_outputs/"
echo "   • ./utilization_test_outputs/"
