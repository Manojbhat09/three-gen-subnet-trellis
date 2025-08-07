#!/usr/bin/env python3
"""
Test Quality Improvements for TRELLIS
Purpose: Test the optimized configuration and validate improvements
"""

import requests
import time
import json
import base64

def test_quality_improvements():
    """Test the quality improvements"""
    
    generation_url = "http://localhost:8096"
    validation_url = "http://localhost:10006"
    
    # Test prompt
    test_prompt = "a shiny brass candle holder with twin flames"
    
    print("🧪 Testing Quality Improvements")
    print(f"Test prompt: '{test_prompt}'")
    print(f"Generation server: {generation_url}")
    print(f"Validation server: {validation_url}")
    
    try:
        # Check server status
        print("\n📊 Checking server status...")
        status_response = requests.get(f"{generation_url}/status/")
        if status_response.status_code == 200:
            status = status_response.json()
            print(f"✅ Generation server ready: {status.get('ready', False)}")
            print(f"   Models loaded: {status.get('models_loaded', {})}")
        else:
            print(f"❌ Generation server not responding: {status_response.status_code}")
            return
        
        # Generate model with optimized settings
        print(f"\n🎯 Generating model with optimized settings...")
        generation_response = requests.post(
            f"{generation_url}/generate/",
            data={
                "prompt": test_prompt,
                "seed": 42,
                "return_compressed": True
            },
            timeout=300
        )
        
        if generation_response.status_code != 200:
            print(f"❌ Generation failed: {generation_response.status_code}")
            return
        
        print("✅ Model generated successfully")
        
        # Get the compressed PLY data
        ply_data = generation_response.content
        print(f"📦 PLY data size: {len(ply_data):,} bytes")
        
        # Validate the generation
        print("\n📊 Validating generation...")
        
        # Convert to base64 for validation
        encoded_data = base64.b64encode(ply_data).decode('utf-8')
        
        validation_response = requests.post(
            f"{validation_url}/validate_txt_to_3d_ply/",
            json={
                "prompt": test_prompt,
                "data": encoded_data,
                "compression": 0,
                "generate_preview": False
            },
            timeout=120
        )
        
        if validation_response.status_code == 200:
            result = validation_response.json()
            score = result.get("score", 0.0)
            
            print(f"✅ Validation completed!")
            print(f"📊 Final Score: {score:.4f}")
            print(f"📊 Quality Metrics:")
            print(f"   - IQA (Quality): {result.get('iqa', 0.0):.4f}")
            print(f"   - Alignment: {result.get('alignment_score', 0.0):.4f}")
            print(f"   - SSIM: {result.get('ssim', 0.0):.4f}")
            print(f"   - LPIPS: {result.get('lpips', 0.0):.4f}")
            
            # Analyze score components
            print(f"\n🔍 Score Analysis:")
            iqa = result.get('iqa', 0.0)
            alignment = result.get('alignment_score', 0.0)
            ssim = result.get('ssim', 0.0)
            lpips = result.get('lpips', 0.0)
            
            # Calculate expected score based on validation formula
            if alignment < 0.3:
                expected_score = 0.0
            else:
                # Apply sigmoid functions
                import torch
                ssim_sigmoid = 1.0 / (1.0 + torch.exp(-35 * (ssim - 0.83)))
                lpips_sigmoid = 1.0 / (1.0 + torch.exp(-30 * (lpips - 0.7)))
                
                expected_score = (
                    0.75 * iqa +
                    0.20 * alignment +
                    0.025 * ssim_sigmoid +
                    0.025 * lpips * lpips_sigmoid
                )
            
            print(f"   - Expected score: {expected_score:.4f}")
            print(f"   - Actual score: {score:.4f}")
            print(f"   - Difference: {abs(expected_score - score):.4f}")
            
            # Quality assessment
            if score >= 0.95:
                print(f"🎉 EXCELLENT! Score {score:.4f} >= 0.95")
            elif score >= 0.90:
                print(f"✅ VERY GOOD! Score {score:.4f} >= 0.90")
            elif score >= 0.85:
                print(f"👍 GOOD! Score {score:.4f} >= 0.85")
            elif score >= 0.80:
                print(f"⚠️  ACCEPTABLE! Score {score:.4f} >= 0.80")
            else:
                print(f"❌ NEEDS IMPROVEMENT! Score {score:.4f} < 0.80")
            
            # Save results
            results = {
                "timestamp": time.time(),
                "prompt": test_prompt,
                "score": score,
                "metrics": {
                    "iqa": iqa,
                    "alignment": alignment,
                    "ssim": ssim,
                    "lpips": lpips
                },
                "expected_score": expected_score,
                "ply_size_bytes": len(ply_data)
            }
            
            with open("quality_test_results.json", "w") as f:
                json.dump(results, f, indent=2)
            
            print(f"\n💾 Results saved to quality_test_results.json")
            
        else:
            print(f"❌ Validation failed: {validation_response.status_code}")
            print(f"Response: {validation_response.text}")
            
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_quality_improvements() 