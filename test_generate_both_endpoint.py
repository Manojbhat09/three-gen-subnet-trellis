#!/usr/bin/env python3
"""
Test script for the new /generate_both/ endpoint
This tests the server's ability to return both PLY and image data in a single request
"""

import asyncio
import aiohttp
import base64
import json
from PIL import Image
import io

async def test_generate_both_endpoint():
    """Test the new /generate_both/ endpoint"""
    
    # Server configuration
    server_url = "http://localhost:8099"  # Adjust port as needed
    
    # Test prompt
    test_prompt = "a blue ceramic vase with red trim"
    
    print(f"🧪 Testing /generate_both/ endpoint")
    print(f"   Server: {server_url}")
    print(f"   Prompt: '{test_prompt}'")
    print("=" * 60)
    
    try:
        # Test the basic /generate_both/ endpoint
        print("📡 Testing basic /generate_both/ endpoint...")
        
        request_data = {
            'prompt': test_prompt,
            'seed': 42,
            'num_inference_steps': 7,
            'guidance_scale': 3.5,
            'ss_sampling_steps': 12,
            'slat_sampling_steps': 21,
            'slat_guidance_strength': 3.5,
            'ss_guidance_strength': 7.5
        }
        
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{server_url}/generate_both/",
                data=request_data,
                timeout=aiohttp.ClientTimeout(total=300)
            ) as response:
                
                if response.status == 200:
                    result = await response.json()
                    
                    print("✅ Basic endpoint test successful!")
                    print(f"   Status: {result.get('status')}")
                    print(f"   Prompt: {result.get('prompt')}")
                    print(f"   Seed: {result.get('seed')}")
                    print(f"   Image size: {result.get('image_size_bytes', 0):,} bytes")
                    print(f"   PLY size: {result.get('ply_size_bytes', 0):,} bytes")
                    
                    if 'compressed_ply' in result:
                        print(f"   Compressed PLY: {result.get('compressed_size_bytes', 0):,} bytes")
                        print(f"   Compression ratio: {result.get('compression_ratio', 0):.2f}")
                    
                    # Verify we got both image and PLY data
                    if 'image' in result and ('ply_data' in result or 'compressed_ply' in result):
                        print("   ✅ Both image and PLY data received")
                        
                        # Test decoding the image
                        try:
                            image_data = base64.b64decode(result['image'])
                            image = Image.open(io.BytesIO(image_data))
                            print(f"   ✅ Image decoded successfully: {image.size} ({image.mode})")
                        except Exception as e:
                            print(f"   ❌ Image decoding failed: {e}")
                        
                        # Test decoding the PLY data
                        try:
                            if 'compressed_ply' in result:
                                ply_data = base64.b64decode(result['compressed_ply'])
                                print(f"   ✅ Compressed PLY decoded: {len(ply_data):,} bytes")
                            elif 'ply_data' in result:
                                ply_data = base64.b64decode(result['ply_data'])
                                print(f"   ✅ PLY data decoded: {len(ply_data):,} bytes")
                        except Exception as e:
                            print(f"   ❌ PLY decoding failed: {e}")
                    else:
                        print("   ❌ Missing image or PLY data")
                        
                else:
                    print(f"❌ Basic endpoint test failed: HTTP {response.status}")
                    error_text = await response.text()
                    print(f"   Error: {error_text}")
        
        print("\n" + "=" * 60)
        
        # Test the LoRA-specific endpoint
        print("📡 Testing /generate_both/cinema/ endpoint...")
        
        request_data = {
            'prompt': test_prompt,
            'seed': 42,
            'num_inference_steps': 7,
            'guidance_scale': 3.5,
            'ss_sampling_steps': 12,
            'slat_sampling_steps': 12,
            'slat_guidance_strength': 3.5,
            'ss_guidance_strength': 7.5
        }
        
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{server_url}/generate_both/cinema/",
                data=request_data,
                timeout=aiohttp.ClientTimeout(total=300)
            ) as response:
                
                if response.status == 200:
                    result = await response.json()
                    
                    print("✅ Cinema LoRA endpoint test successful!")
                    print(f"   Status: {result.get('status')}")
                    print(f"   Prompt: {result.get('prompt')}")
                    print(f"   Enhanced prompt: {result.get('enhanced_prompt', 'N/A')}")
                    print(f"   LoRA: {result.get('lora', 'N/A')}")
                    print(f"   Image size: {result.get('image_size_bytes', 0):,} bytes")
                    print(f"   PLY size: {result.get('ply_size_bytes', 0):,} bytes")
                    
                    if 'compressed_ply' in result:
                        print(f"   Compressed PLY: {result.get('compressed_size_bytes', 0):,} bytes")
                        print(f"   Compression ratio: {result.get('compression_ratio', 0):.2f}")
                        
                else:
                    print(f"❌ Cinema LoRA endpoint test failed: HTTP {response.status}")
                    error_text = await response.text()
                    print(f"   Error: {error_text}")
        
        print("\n" + "=" * 60)
        print("🎉 All tests completed!")
        
    except Exception as e:
        print(f"❌ Test failed with exception: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_generate_both_endpoint())
