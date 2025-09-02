#!/usr/bin/env python3
"""
Test script for CLIP scorer functionality
Tests the CLIP scoring system independently
"""

import logging
import time
from clip_scorer import get_clip_scorer, unload_global_clip_scorer

def test_clip_scorer():
    """Test the CLIP scorer with a simple example"""
    
    print("🧪 Testing CLIP Scorer")
    print("=" * 50)
    
    # Setup logging
    logging.basicConfig(level=logging.INFO, format='%(message)s')
    
    # Get CLIP scorer
    print("🔄 Initializing CLIP scorer...")
    start_time = time.time()
    scorer = get_clip_scorer()
    init_time = time.time() - start_time
    print(f"✅ CLIP scorer initialized in {init_time:.2f}s")
    
    # Test model info
    print("\n📊 Model Information:")
    model_info = scorer.get_model_info()
    for key, value in model_info.items():
        print(f"   {key}: {value}")
    
    # Test text-to-text similarity
    print("\n🔤 Testing text-to-text similarity:")
    text1 = "a red car"
    text2 = "a red automobile"
    text3 = "a blue bicycle"
    
    start_time = time.time()
    sim1 = scorer.compute_text_to_text_similarity(text1, text2)
    sim2 = scorer.compute_text_to_text_similarity(text1, text3)
    text_time = time.time() - start_time
    
    print(f"   '{text1}' vs '{text2}': {sim1:.4f}")
    print(f"   '{text1}' vs '{text3}': {sim2:.4f}")
    print(f"   ⏱️ Text similarity computation: {text_time:.2f}s")
    
    # Test with a simple base64 image (1x1 red pixel)
    print("\n🖼️ Testing image-text similarity:")
    # Create a simple 1x1 red pixel image as base64
    import base64
    from PIL import Image
    import io
    
    # Create a simple red image
    red_image = Image.new('RGB', (224, 224), color='red')
    img_buffer = io.BytesIO()
    red_image.save(img_buffer, format='PNG')
    image_base64 = base64.b64encode(img_buffer.getvalue()).decode('utf-8')
    
    start_time = time.time()
    clip_score = scorer.compute_clip_score("a red image", image_base64)
    clip_time = time.time() - start_time
    
    print(f"   'a red image' vs red pixel image: {clip_score:.4f}")
    print(f"   ⏱️ CLIP computation: {clip_time:.2f}s")
    
    # Test batch processing
    print("\n📦 Testing batch processing:")
    prompts = ["a red car", "a blue sky", "a green tree"]
    images = [image_base64, image_base64, image_base64]  # Same image for all
    
    start_time = time.time()
    batch_scores = scorer.batch_compute_clip_scores(prompts, images)
    batch_time = time.time() - start_time
    
    for i, (prompt, score) in enumerate(zip(prompts, batch_scores)):
        print(f"   {i+1}. '{prompt}': {score:.4f}")
    print(f"   ⏱️ Batch processing: {batch_time:.2f}s")
    
    # Cleanup
    print("\n🗑️ Cleaning up...")
    cleanup_start = time.time()
    unload_global_clip_scorer()
    cleanup_time = time.time() - cleanup_start
    print(f"✅ Cleanup completed in {cleanup_time:.2f}s")
    
    print("\n🎉 CLIP scorer test completed successfully!")

if __name__ == "__main__":
    test_clip_scorer()


