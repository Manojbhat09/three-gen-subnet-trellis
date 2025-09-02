#!/usr/bin/env python3
"""
Enhanced Test script for the new comprehensive grid flow endpoint:
/generate_3d_from_prompt_grid_flow/

This endpoint follows the exact flow from test_img2img_prompt.py:
1. Generate grid image with multiple views
2. Crop grid into individual images
3. Optionally upscale images using Real-ESRGAN
4. Optionally remove backgrounds
5. Generate 3D model using TRELLIS multi-image pipeline

NEW: Saves all intermediate outputs for inspection!
"""

import requests
import json
import time
import os
from pathlib import Path

# Server configuration
SERVER_URL = "http://localhost:8097"
ENDPOINT = "/generate_3d_from_prompt_grid_flow/"

def ensure_output_dir():
    """Ensure test_outputs directory exists"""
    Path("test_outputs").mkdir(exist_ok=True)
    return "test_outputs"

def test_grid_flow_endpoint():
    """Test the comprehensive grid flow endpoint with various configurations."""
    
    print("🎯 Testing Comprehensive Grid Flow Endpoint")
    print("=" * 60)
    print(f"Server: {SERVER_URL}")
    print(f"Endpoint: {ENDPOINT}")
    print("=" * 60)
    print("💾 All intermediate outputs will be saved!")
    print("=" * 60)
    
    # Test configurations - ENHANCED with intermediate saving
    test_configs = [
        {
            "name": "SMallest",
            "params": {
                "base_prompt": "orange hut",
                "style": "cinema",
                "seed": 42,
                "num_inference_steps": 7,
                "guidance_scale": 3.5,
                "width": 256,
                "height": 256,
                "upscale": False,
                "remove_background": True,
                "return_compressed": False,  # Get uncompressed PLY
                "save_preview": True,        # Generate preview video
                "save_intermediate": True,   # Save all intermediate outputs
                "filter_low_quality": True,
                "timing": True,
                "use_short_prompt": True
            }
        },
        {
            "name": "SMallest Upscaled",
            "params": {
                "base_prompt": "orange hut",
                "style": "cinema",
                "seed": 43,
                "num_inference_steps": 7,
                "guidance_scale": 3.5,
                "width": 256,
                "height": 256,
                "upscale": True,
                "remove_background": True,
                "return_compressed": False,  # Get uncompressed PLY
                "save_preview": True,        # Generate preview video
                "save_intermediate": True,   # Save all intermediate outputs
                "filter_low_quality": True,
                "timing": True,
                "use_short_prompt": True
            }
        },
        {
            "name": "SMallest Upscaled long",
            "params": {
                "base_prompt": "orange hut",
                "style": "cinema",
                "seed": 44,
                "num_inference_steps": 7,
                "guidance_scale": 3.5,
                "width": 256,
                "height": 256,
                "upscale": True,
                "remove_background": True,
                "return_compressed": False,  # Get uncompressed PLY
                "save_preview": True,        # Generate preview video
                "save_intermediate": True,   # Save all intermediate outputs
                "filter_low_quality": True,
                "timing": True,
                "use_short_prompt": False
            }
        },
        {
            "name": "GOOD",
            "params": {
                "base_prompt": "orange hut",
                "style": "cinema",
                "seed": 45,
                "num_inference_steps": 7,
                "guidance_scale": 3.5,
                "width": 512,
                "height": 512,
                "upscale": False,
                "remove_background": True,
                "return_compressed": False,  # Get uncompressed PLY
                "save_preview": True,        # Generate preview video
                "save_intermediate": True,   # Save all intermediate outputs
                "filter_low_quality": True,
                "timing": True,
                "use_short_prompt": False
            }
        },
        {
            "name": "GOOD short",
            "params": {
                "base_prompt": "orange hut",
                "style": "cinema",
                "seed": 46,
                "num_inference_steps": 7,
                "guidance_scale": 3.5,
                "width": 512,
                "height": 512,
                "upscale": False,
                "remove_background": True,
                "return_compressed": False,  # Get uncompressed PLY
                "save_preview": True,        # Generate preview video
                "save_intermediate": True,   # Save all intermediate outputs
                "filter_low_quality": True,
                "timing": True,
                "use_short_prompt": True
            }
        },
        {
            "name": "Optimal save",
            "params": {
                "base_prompt": "orange hut",
                "style": "cinema",
                "seed": 42,
                "num_inference_steps": 7,
                "guidance_scale": 3.5,
                "width": 512,
                "height": 512,
                "upscale": False,
                "remove_background": True,
                "return_compressed": False,  # Get uncompressed PLY
                "save_preview": True,        # Generate preview video
                "save_intermediate": True,   # Save all intermediate outputs
                "filter_low_quality": True,
                "timing": True,
                "use_short_prompt": False
            }
        },
        {
            "name": "Optimal save 1024",
            "params": {
                "base_prompt": "orange hut",
                "style": "cinema",
                "seed": 42,
                "num_inference_steps": 7,
                "guidance_scale": 3.5,
                "width": 1024,
                "height": 1024,
                "upscale": False,
                "remove_background": True,
                "return_compressed": False,  # Get uncompressed PLY
                "save_preview": True,        # Generate preview video
                "save_intermediate": True,   # Save all intermediate outputs
                "filter_low_quality": True,
                "timing": True,
                "use_short_prompt": False
            }
        },
        {
            "name": "Optimal save upscale",
            "params": {
                "base_prompt": "orange hut",
                "style": "cinema",
                "seed": 42,
                "num_inference_steps": 7,
                "guidance_scale": 3.5,
                "width": 512,
                "height": 512,
                "upscale": True,
                "remove_background": True,
                "return_compressed": False,  # Get uncompressed PLY
                "save_preview": True,        # Generate preview video
                "save_intermediate": True,   # Save all intermediate outputs
                "filter_low_quality": True,
                "timing": True,
                "use_short_prompt": False
            }
        },
        {
            "name": "Optimal save short prompt",
            "params": {
                "base_prompt": "orange hut",
                "style": "cinema",
                "seed": 42,
                "num_inference_steps": 7,
                "guidance_scale": 3.5,
                "width": 512,
                "height": 512,
                "upscale": False,
                "remove_background": True,
                "return_compressed": False,  # Get uncompressed PLY
                "save_preview": True,        # Generate preview video
                "save_intermediate": True,   # Save all intermediate outputs
                "filter_low_quality": True,
                "timing": True,
                "use_short_prompt": True
            }
        },
        {
            "name": "Basic Standard Style save",
            "params": {
                "base_prompt": "robot",
                "style": "standard",
                "seed": 42,
                "num_inference_steps": 8,
                "guidance_scale": 3.5,
                "width": 1024,
                "height": 1024,
                "upscale": False,
                "remove_background": True,
                "return_compressed": False,  # Get uncompressed PLY
                "save_preview": True,        # Generate preview video
                "save_intermediate": True,   # Save all intermediate outputs
                "filter_low_quality": True,
                "timing": True,
                "use_short_prompt": True
            }
        },
        {
            "name": "Cinema Style with Upscaling save",
            "params": {
                "base_prompt": "car",
                "style": "cinema",
                "seed": 123,
                "num_inference_steps": 12,
                "guidance_scale": 4.0,
                "width": 1024,
                "height": 1024,
                "upscale": True,
                "remove_background": True,
                "return_compressed": False,  # Get uncompressed PLY
                "save_preview": True,        # Generate preview video
                "save_intermediate": True,   # Save all intermediate outputs
                "filter_low_quality": True,
                "timing": True,
                "use_short_prompt": True
            }
        },
        {
            "name": "3D Style High Quality save",
            "params": {
                "base_prompt": "spaceship",
                "style": "3d",
                "seed": 456,
                "num_inference_steps": 16,
                "guidance_scale": 5.0,
                "width": 1024,
                "height": 1024,
                "upscale": True,
                "remove_background": True,
                "ss_guidance_strength": 8.0,
                "ss_sampling_steps": 25,
                "slat_guidance_strength": 5.0,
                "slat_sampling_steps": 30,
                "return_compressed": False,  # Get uncompressed PLY
                "save_preview": True,        # Generate preview video
                "save_intermediate": True,   # Save all intermediate outputs
                "filter_low_quality": True,
                "timing": True,
                "use_short_prompt": True
            }
        },
        {
            "name": "Fast Generation (512x512) save",
            "params": {
                "base_prompt": "cat",
                "style": "standard",
                "seed": 789,
                "num_inference_steps": 4,
                "guidance_scale": 3.0,
                "width": 512,
                "height": 512,
                "upscale": False,
                "remove_background": False,
                "return_compressed": False,  # Get uncompressed PLY
                "save_preview": True,        # Generate preview video
                "save_intermediate": True,   # Save all intermediate outputs
                "filter_low_quality": True,
                "timing": True,
                "use_short_prompt": True
            }
        },
        {
            "name": "Basic Standard Style",
            "params": {
                "base_prompt": "robot",
                "style": "standard",
                "seed": 42,
                "num_inference_steps": 8,
                "guidance_scale": 3.5,
                "width": 1024,
                "height": 1024,
                "upscale": False,
                "remove_background": True,
                "timing": True
            }
        },
        {
            "name": "Cinema Style with Upscaling",
            "params": {
                "base_prompt": "car",
                "style": "cinema",
                "seed": 123,
                "num_inference_steps": 12,
                "guidance_scale": 4.0,
                "width": 1024,
                "height": 1024,
                "upscale": True,
                "remove_background": True,
                "timing": True
            }
        },
        {
            "name": "3D Style High Quality",
            "params": {
                "base_prompt": "spaceship",
                "style": "3d",
                "seed": 456,
                "num_inference_steps": 16,
                "guidance_scale": 5.0,
                "width": 1024,
                "height": 1024,
                "upscale": True,
                "remove_background": True,
                "ss_guidance_strength": 8.0,
                "ss_sampling_steps": 25,
                "slat_guidance_strength": 5.0,
                "slat_sampling_steps": 30,
                "timing": True
            }
        },
        {
            "name": "Fast Generation (512x512)",
            "params": {
                "base_prompt": "cat",
                "style": "standard",
                "seed": 789,
                "num_inference_steps": 4,
                "guidance_scale": 3.0,
                "width": 512,
                "height": 512,
                "upscale": False,
                "remove_background": False,
                "timing": False
            }
        }
    ]
    
    for i, config in enumerate(test_configs, 1):
        print(f"\n🧪 Test {i}: {config['name']}")
        print("-" * 40)
        
        # Create unique output directory for this test
        test_name = config['name'].lower().replace(' ', '_').replace('(', '').replace(')', '')
        test_output_dir = f"test_outputs/{test_name}_{config['params']['seed']}"
        Path(test_output_dir).mkdir(parents=True, exist_ok=True)
        
        print(f"📁 Output directory: {test_output_dir}")
        
        # Prepare form data
        form_data = config['params'].copy()
        
        # Convert boolean values to strings for form data
        for key, value in form_data.items():
            if isinstance(value, bool):
                form_data[key] = str(value).lower()
        
        print(f"Parameters:")
        for key, value in form_data.items():
            print(f"  {key}: {value}")
        
        try:
            # Make request to endpoint
            print(f"\n🚀 Sending request...")
            start_time = time.time()
            
            response = requests.post(
                f"{SERVER_URL}{ENDPOINT}",
                data=form_data,
                timeout=1800  # 30 minutes timeout
            )
            
            request_time = time.time() - start_time
            print(f"   Request completed in {request_time:.2f}s")
            print(f"   Status code: {response.status_code}")
            
            if response.status_code == 200:
                # Success - check response headers
                print(f"   ✅ Success!")
                
                # Extract response data from headers
                response_data = {}
                try:
                    response_data = json.loads(response.headers.get('X-Response-Data', '{}'))
                except:
                    pass
                
                # Print response info
                if response_data:
                    print(f"   📊 Response Data:")
                    print(f"     - Status: {response_data.get('status', 'unknown')}")
                    print(f"     - Pipeline: {response_data.get('pipeline', 'unknown')}")
                    print(f"     - Generation time: {response_data.get('generation_time', 0):.2f}s")
                    print(f"     - PLY size: {response_data.get('ply_size_bytes', 0):,} bytes")
                    print(f"     - Steps completed: {', '.join(response_data.get('steps_completed', []))}")
                
                # Check compression info
                compression = response.headers.get('X-Compression', 'none')
                if compression == 'spz':
                    compression_ratio = response.headers.get('X-Compression-Ratio', '0%')
                    print(f"   🗜️ Compression: SPZ ({compression_ratio})")
                else:
                    print(f"   📁 Compression: None")
                
                # Save the PLY file to test-specific directory
                filename = response.headers.get('Content-Disposition', '').split('filename=')[-1].strip('"')
                if not filename:
                    filename = f"grid_flow_{config['params']['base_prompt']}_{config['params']['seed']}.ply"
                    if compression == 'spz':
                        filename += '.spz'
                
                output_path = Path(f"{test_output_dir}/{filename}")
                
                with open(output_path, 'wb') as f:
                    f.write(response.content)
                
                print(f"   💾 File saved: {output_path}")
                print(f"   📏 File size: {len(response.content):,} bytes ({len(response.content)/1024/1024:.1f} MB)")
                
                # Save metadata
                if response_data:
                    metadata_file = Path(f"{test_output_dir}/metadata.json")
                    with open(metadata_file, 'w') as f:
                        json.dump(response_data, f, indent=2)
                    print(f"   💾 Metadata saved: {metadata_file}")
                
                print(f"   📁 Check {test_output_dir} for all intermediate outputs!")
                
            else:
                # Error response
                print(f"   ❌ Error: {response.status_code}")
                try:
                    error_detail = response.json()
                    print(f"   Error details: {error_detail}")
                except:
                    print(f"   Error text: {response.text[:200]}...")
        
        except requests.exceptions.Timeout:
            print(f"   ⏰ Request timed out after 30 minutes")
        except requests.exceptions.ConnectionError:
            print(f"   🔌 Connection error - is the server running?")
        except Exception as e:
            print(f"   💥 Unexpected error: {e}")
        
        print(f"   {'='*40}")
    
    print(f"\n🎉 All tests completed!")
    print(f"📁 Check the 'test_outputs/' directory for generated files.")
    print(f"💾 Each test has its own subdirectory with all intermediate outputs!")

def test_single_config():
    """Test a single configuration with detailed output."""
    
    print("🎯 Testing Single Configuration")
    print("=" * 40)
    
    # Single test configuration - ENHANCED with intermediate saving
    params = {
        "base_prompt": "robot",
        "style": "standard",
        "seed": 42,
        "num_inference_steps": 8,
        "guidance_scale": 3.5,
        "width": 1024,
        "height": 1024,
        "upscale": True,
        "remove_background": True,
        "ss_guidance_strength": 7.5,
        "ss_sampling_steps": 21,
        "slat_guidance_strength": 4.0,
        "slat_sampling_steps": 24,
        "return_compressed": False,  # Get uncompressed PLY
        "save_preview": True,        # Generate preview video
        "save_intermediate": True,   # Save all intermediate outputs
        "filter_low_quality": True,
        "timing": True,
        "use_short_prompt": True
    }
    
    # Create unique output directory for this test
    test_output_dir = f"test_outputs/single_test_{params['seed']}"
    Path(test_output_dir).mkdir(parents=True, exist_ok=True)
    
    print(f"📁 Output directory: {test_output_dir}")
    print(f"Parameters:")
    for key, value in params.items():
        print(f"  {key}: {value}")
    
    # Convert boolean values to strings for form data
    form_data = {k: str(v).lower() if isinstance(v, bool) else v for k, v in params.items()}
    
    try:
        print(f"\n🚀 Sending request...")
        start_time = time.time()
        
        response = requests.post(
            f"{SERVER_URL}{ENDPOINT}",
            data=form_data,
            timeout=1800
        )
        
        request_time = time.time() - start_time
        print(f"Request completed in {request_time:.2f}s")
        print(f"Status code: {response.status_code}")
        
        if response.status_code == 200:
            print(f"✅ Success!")
            
            # Extract and display response data
            response_data = {}
            try:
                response_data = json.loads(response.headers.get('X-Response-Data', '{}'))
            except:
                pass
            
            if response_data:
                print(f"\n📊 Response Data:")
                for key, value in response_data.items():
                    print(f"  {key}: {value}")
            
            # Save file to test-specific directory
            filename = response.headers.get('Content-Disposition', '').split('filename=')[-1].strip('"')
            if not filename:
                filename = f"grid_flow_{params['base_prompt']}_{params['seed']}.ply"
                if params['return_compressed']:
                    filename += '.spz'
            
            output_path = Path(f"{test_output_dir}/{filename}")
            
            with open(output_path, 'wb') as f:
                f.write(response.content)
            
            print(f"\n💾 File saved: {output_path}")
            print(f"📏 File size: {len(response.content):,} bytes ({len(response.content)/1024/1024:.1f} MB)")
            
            # Save metadata
            if response_data:
                metadata_file = Path(f"{test_output_dir}/metadata.json")
                with open(metadata_file, 'w') as f:
                    json.dump(response_data, f, indent=2)
                print(f"💾 Metadata saved: {metadata_file}")
            
            print(f"📁 Check {test_output_dir} for all intermediate outputs!")
            
        else:
            print(f"❌ Error: {response.status_code}")
            try:
                error_detail = response.json()
                print(f"Error details: {error_detail}")
            except:
                print(f"Error text: {response.text[:200]}...")
    
    except Exception as e:
        print(f"💥 Error: {e}")

if __name__ == "__main__":
    print("🚀 Enhanced FLUX + TRELLIS Grid Flow Endpoint Tester")
    print("=" * 60)
    print("💾 Saves ALL intermediate outputs for inspection!")
    print("📁 Creates organized test directories")
    print("🎬 Generates preview videos")
    print("🗜️ Provides uncompressed PLY files")
    print("=" * 60)
    
    # Ensure output directory exists
    ensure_output_dir()
    
    # Check if server is running
    try:
        health_check = requests.get(f"{SERVER_URL}/", timeout=5)
        print(f"✅ Server is running at {SERVER_URL}")
    except:
        print(f"❌ Server not accessible at {SERVER_URL}")
        print(f"   Please ensure the server is running on port 8096")
        exit(1)
    
    print("\nChoose test mode:")
    print("1. Run all test configurations (saves all outputs)")
    print("2. Run single detailed test (saves all outputs)")
    
    choice = input("\nEnter choice (1 or 2): ").strip()
    
    if choice == "1":
        test_grid_flow_endpoint()
    elif choice == "2":
        test_single_config()
    else:
        print("Invalid choice. Running all tests...")
        test_grid_flow_endpoint()
