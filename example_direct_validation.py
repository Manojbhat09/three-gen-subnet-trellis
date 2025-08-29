#!/usr/bin/env python3
"""
Example script demonstrating how to use the validate_prompt_direct function
for fast validation without subprocess overhead.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from subnet_accurate_validator_multigpu import validate_prompt_direct, unload_cached_models

def main():
    print("🎯 Direct Validation Example")
    print("=" * 60)

    # Example prompts - you can modify these
    original_prompt = "A sleek modern chair with clean lines"
    optimized_prompt = "A minimalist Scandinavian chair, ergonomic design, oak wood, contemporary style, photorealistic, 8k resolution"

    print(f"📝 Original Prompt: '{original_prompt}'")
    print(f"🔧 Optimized Prompt: '{optimized_prompt}'")
    print()

    try:
        # First validation call - models will be loaded
        print("🚀 First validation (models loading)...")
        result1 = validate_prompt_direct(
            original_prompt=original_prompt,
            optimized_prompt=optimized_prompt,
            endpoint="generate/",  # Use "generate/image/" for 2D validation
            port=8096
        )

        if result1:
            print("
📊 RESULTS:"            print(f"   Validation Score: {result1.get('validation_engine_score', 'N/A')}")
            print(f"   Alignment Score: {result1.get('alignment_score', 'N/A')}")
            print(f"   Method: {result1.get('validation_method', 'N/A')}")
            print(f"   Endpoint: {result1.get('endpoint_type', 'N/A')}")

        # Second validation call - models are already cached
        print("
⚡ Second validation (models cached)..."        result2 = validate_prompt_direct(
            original_prompt="A wooden table",
            optimized_prompt="A rustic oak dining table, farmhouse style, natural wood grain",
            endpoint="generate/",
            port=8096
        )

        if result2:
            print("
📊 RESULTS:"            print(f"   Validation Score: {result2.get('validation_engine_score', 'N/A')}")
            print(f"   Alignment Score: {result2.get('alignment_score', 'N/A')}")

        print("
✅ Example completed successfully!"        print("💡 Models remain cached for subsequent calls - much faster!")

    except Exception as e:
        print(f"❌ Example failed: {e}")
        import traceback
        traceback.print_exc()

    # Optional: Clean up models when done
    print("
🧹 Cleaning up cached models..."    unload_cached_models()

if __name__ == "__main__":
    main()
