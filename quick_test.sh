#!/bin/bash

# Quick Test Script for GPU Servers
# Tests individual GPU endpoints with curl commands

echo "🚀 Quick Test of GPU Servers"
echo "=============================="

# Test prompts for each GPU
prompts=(
    "a pink bicycle with chrome wheels"
    "a blue ceramic vase with red trim"
    "a wooden table with four chairs"
    "a silver laptop on a desk"
    "a red sports car in a garage"
    "a green plant in a pot"
    "a black coffee mug on a saucer"
    "a white cloud in a blue sky"
)

# Test each GPU
for i in {0..7}; do
    port=$((8096 + i))
    prompt="${prompts[$i]}"
    
    echo ""
    echo "🎨 Testing GPU $i (port $port)"
    echo "   Prompt: '$prompt'"
    
    # Test generation endpoint
    response=$(curl -s -w "HTTP_STATUS:%{http_code},TIME:%{time_total}s,SIZE:%{size_download}" \
        -d "prompt=$prompt&seed=42&return_compressed=true" \
        -X POST "http://127.0.0.1:$port/generate/" \
        -o "/tmp/gpu_${i}_response.ply.spz")
    
    # Extract response info
    http_status=$(echo "$response" | grep -o 'HTTP_STATUS:[0-9]*' | cut -d: -f2)
    time_taken=$(echo "$response" | grep -o 'TIME:[0-9.]*' | cut -d: -f2)
    size_bytes=$(echo "$response" | grep -o 'SIZE:[0-9]*' | cut -d: -f2)
    
    if [ "$http_status" = "200" ]; then
        echo "   ✅ Success: HTTP $http_status"
        echo "   ⏱️  Time: ${time_taken}s"
        echo "   📦 Size: ${size_bytes} bytes"
        
        # Check if file was created
        if [ -f "/tmp/gpu_${i}_response.ply.spz" ]; then
            file_size=$(stat -c%s "/tmp/gpu_${i}_response.ply.spz")
            echo "   💾 File saved: /tmp/gpu_${i}_response.ply.spz ($file_size bytes)"
        fi
        
        # Test status endpoint
        status_response=$(curl -s "http://127.0.0.1:$port/status/" | jq -r '.status // "unknown"' 2>/dev/null || echo "unknown")
        echo "   📊 Server status: $status_response"
        
    else
        echo "   ❌ Failed: HTTP $http_status"
        echo "   ⏱️  Time: ${time_taken}s"
    fi
done

echo ""
echo "🎉 Quick test complete!"
echo ""
echo "📁 Generated files are in /tmp/gpu_*_response.ply.spz"
echo "🧹 Clean up with: rm /tmp/gpu_*_response.ply.spz"
