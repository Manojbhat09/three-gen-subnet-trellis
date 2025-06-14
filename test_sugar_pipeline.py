#!/usr/bin/env python3
"""
Test script for the new Flux + Hunyuan3D + SuGaR pipeline
Tests mesh-to-Gaussian Splatting conversion and validates output format
"""

import requests
import time
import sys
import os
from pathlib import Path
import numpy as np
from plyfile import PlyData

def test_sugar_generation_server():
    """Test the SuGaR-enhanced generation server"""
    
    # Server configuration
    server_url = "http://localhost:8095"
    
    print("🧪 Testing SuGaR-Enhanced Generation Pipeline")
    print("=" * 60)
    
    # Test 1: Health check
    print("\n1. Testing server health...")
    try:
        response = requests.get(f"{server_url}/health/")
        if response.status_code == 200:
            print("✓ Server is healthy")
        else:
            print(f"❌ Server health check failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Cannot connect to server: {e}")
        return False
    
    # Test 2: Check server status
    print("\n2. Checking server status...")
    try:
        response = requests.get(f"{server_url}/status/")
        if response.status_code == 200:
            status = response.json()
            print(f"✓ Server status: {status.get('status', 'unknown')}")
            
            models_loaded = status.get('models_loaded', {})
            if isinstance(models_loaded, dict):
                print(f"  Models loaded:")
                for model, loaded in models_loaded.items():
                    print(f"    {model}: {'✓' if loaded else '❌'}")
            else:
                print(f"  Models loaded: {models_loaded}")
            
            metrics = status.get('metrics', {})
            if isinstance(metrics, dict):
                sugar_count = metrics.get('sugar_converted_count', 0)
                print(f"  SuGaR conversions: {sugar_count}")
                total_gens = metrics.get('total_generations', 0)
                success_gens = metrics.get('successful_generations', 0)
                print(f"  Total generations: {total_gens}")
                print(f"  Successful generations: {success_gens}")
            else:
                print(f"  Metrics: {metrics}")
        else:
            print(f"❌ Status check failed: {response.status_code}")
            print(f"Response: {response.text}")
    except Exception as e:
        print(f"❌ Status check error: {e}")
        import traceback
        traceback.print_exc()
    
    # Test 3: Generate a simple 3D model
    print("\n3. Testing 3D model generation with SuGaR conversion...")
    
    test_prompts = [
        "a red apple",
        "a wooden chair", 
        "a blue coffee mug"
    ]
    
    for i, prompt in enumerate(test_prompts):
        print(f"\n   Test {i+1}: '{prompt}'")
        
        try:
            # Make generation request
            data = {
                'prompt': prompt,
                'seed': 12345 + i,
                'return_compressed': False  # Get uncompressed PLY for analysis
            }
            
            print(f"   Sending request...")
            start_time = time.time()
            
            response = requests.post(f"{server_url}/generate/", data=data, timeout=300)
            
            generation_time = time.time() - start_time
            
            if response.status_code == 200:
                print(f"   ✓ Generation successful in {generation_time:.2f}s")
                
                # Check headers for metadata
                headers = response.headers
                print(f"   📊 Metadata:")
                print(f"      Generation Time: {headers.get('X-Generation-Time', 'N/A')}s")
                print(f"      Face Count: {headers.get('X-Face-Count', 'N/A')}")
                print(f"      Vertex Count: {headers.get('X-Vertex-Count', 'N/A')}")
                print(f"      SuGaR Converted: {headers.get('X-Sugar-Converted', 'N/A')}")
                print(f"      Compression Type: {headers.get('X-Compression-Type', 'N/A')}")
                
                # Analyze PLY data
                ply_data = response.content
                print(f"   📁 PLY file size: {len(ply_data):,} bytes")
                
                # Validate Gaussian Splatting format
                if validate_gaussian_splatting_ply(ply_data):
                    print(f"   ✅ Valid Gaussian Splatting PLY format!")
                else:
                    print(f"   ⚠️ PLY format validation failed")
                
                # Save for inspection
                output_file = f"test_output_{prompt.replace(' ', '_')}_{12345 + i}.ply"
                with open(output_file, 'wb') as f:
                    f.write(ply_data)
                print(f"   💾 Saved to: {output_file}")
                
            else:
                print(f"   ❌ Generation failed: {response.status_code}")
                print(f"   Error: {response.text}")
                
        except Exception as e:
            print(f"   ❌ Generation error: {e}")
    
    print("\n" + "=" * 60)
    print("🎯 SuGaR Pipeline Test Complete!")
    return True

def validate_gaussian_splatting_ply(ply_data: bytes) -> bool:
    """Validate that PLY data is in Gaussian Splatting format"""
    try:
        from io import BytesIO
        
        # Parse PLY data
        ply_buffer = BytesIO(ply_data)
        ply = PlyData.read(ply_buffer)
        
        if 'vertex' not in ply:
            print("   ❌ No vertex element found")
            return False
        
        vertices = ply['vertex']
        print(f"   📊 PLY Analysis:")
        print(f"      Vertices: {len(vertices):,}")
        
        # Check for required Gaussian Splatting attributes
        required_attrs = ['x', 'y', 'z', 'f_dc_0', 'f_dc_1', 'f_dc_2', 'opacity', 'scale_0', 'scale_1', 'scale_2']
        optional_attrs = ['nx', 'ny', 'nz', 'rot_0', 'rot_1', 'rot_2', 'rot_3']
        
        vertex_props = [prop.name for prop in vertices.properties]
        print(f"      Properties: {len(vertex_props)}")
        
        # Check required attributes
        missing_required = []
        for attr in required_attrs:
            if attr not in vertex_props:
                missing_required.append(attr)
        
        if missing_required:
            print(f"   ❌ Missing required GS attributes: {missing_required}")
            return False
        
        # Check optional attributes
        found_optional = []
        for attr in optional_attrs:
            if attr in vertex_props:
                found_optional.append(attr)
        
        print(f"      Required GS attributes: ✓ All present")
        print(f"      Optional GS attributes: {found_optional}")
        
        # Check data ranges
        try:
            opacity_data = vertices['opacity']
            scale_data = [vertices[f'scale_{i}'] for i in range(3)]
            
            print(f"      Opacity range: [{opacity_data.min():.3f}, {opacity_data.max():.3f}]")
            print(f"      Scale ranges: [{min(s.min() for s in scale_data):.3f}, {max(s.max() for s in scale_data):.3f}]")
            
            # Check for reasonable values
            if opacity_data.min() < 0 or opacity_data.max() > 1:
                print(f"   ⚠️ Opacity values outside [0,1] range")
            
        except Exception as e:
            print(f"   ⚠️ Could not analyze data ranges: {e}")
        
        return True
        
    except Exception as e:
        print(f"   ❌ PLY validation error: {e}")
        return False

def test_validation_server_compatibility():
    """Test if generated PLY files work with the validation server"""
    print("\n4. Testing validation server compatibility...")
    
    validation_url = "http://localhost:10006"
    
    # Check if validation server is running
    try:
        response = requests.get(f"{validation_url}/version/")
        if response.status_code == 200:
            print("   ✓ Validation server is running")
        else:
            print("   ⚠️ Validation server not responding")
            return
    except:
        print("   ⚠️ Validation server not available")
        return
    
    # Test with a generated PLY file
    test_files = [f for f in os.listdir('.') if f.startswith('test_output_') and f.endswith('.ply')]
    
    if not test_files:
        print("   ⚠️ No test PLY files found")
        return
    
    test_file = test_files[0]
    print(f"   Testing with: {test_file}")
    
    try:
        with open(test_file, 'rb') as f:
            ply_data = f.read()
        
        # Send to validation server
        files = {'ply_file': ('test.ply', ply_data, 'application/x-ply')}
        response = requests.post(f"{validation_url}/validate/", files=files, timeout=60)
        
        if response.status_code == 200:
            result = response.json()
            print(f"   ✅ Validation successful!")
            print(f"      Final Score: {result.get('final_score', 'N/A')}")
            print(f"      IQA Score: {result.get('iqa_score', 'N/A')}")
            print(f"      Alignment Score: {result.get('alignment_score', 'N/A')}")
        else:
            print(f"   ❌ Validation failed: {response.status_code}")
            print(f"   Response: {response.text}")
            
    except Exception as e:
        print(f"   ❌ Validation test error: {e}")

def compare_with_original_pipeline():
    """Compare SuGaR pipeline with original pipeline"""
    print("\n5. Comparing with original pipeline...")
    
    # This would require running both servers and comparing outputs
    # For now, just check if original server is available
    try:
        response = requests.get("http://localhost:8094/health/")  # Original server port
        if response.status_code == 200:
            print("   ✓ Original pipeline server available for comparison")
            # Could implement side-by-side comparison here
        else:
            print("   ⚠️ Original pipeline server not available")
    except:
        print("   ⚠️ Original pipeline server not running")

if __name__ == "__main__":
    print("🚀 Starting SuGaR Pipeline Tests")
    
    success = test_sugar_generation_server()
    
    if success:
        test_validation_server_compatibility()
        compare_with_original_pipeline()
    
    print("\n🏁 Test suite completed!") 