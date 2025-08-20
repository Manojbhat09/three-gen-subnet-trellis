#!/usr/bin/env python3
"""
Test script to debug Nunchaku API connection
"""

import requests
import json

def test_nunchaku_api():
    """Test the Nunchaku API server"""
    
    # Test 1: Health check
    print("🧪 Test 1: Health check")
    try:
        response = requests.get("http://localhost:8200/health", timeout=10)
        print(f"   Status: {response.status_code}")
        print(f"   Response: {response.text}")
    except Exception as e:
        print(f"   ❌ Failed: {e}")
    
    print()
    
    # Test 2: Generate image
    print("🧪 Test 2: Generate image")
    try:
        data = {
            "prompt": "A beautiful sunset",
            "seed": 42,
            "width": 1024,
            "height": 1024
        }
        
        print(f"   Sending data: {json.dumps(data, indent=2)}")
        
        response = requests.post(
            "http://localhost:8200/generate",
            json=data,
            timeout=60
        )
        
        print(f"   Status: {response.status_code}")
        print(f"   Response headers: {dict(response.headers)}")
        
        if response.status_code == 200:
            result = response.json()
            print(f"   ✅ Success! Image size: {len(result.get('image_base64', ''))} chars")
        else:
            print(f"   ❌ Failed: {response.text}")
            
    except Exception as e:
        print(f"   ❌ Failed: {e}")

if __name__ == "__main__":
    test_nunchaku_api()
