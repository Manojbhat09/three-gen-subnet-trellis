#!/usr/bin/env python3
"""
Simple HTTP client for Nunchaku API server
"""

import requests
import base64
import io
from PIL import Image
from typing import Optional

def generate_nunchaku_image_http(prompt: str, seed: int = 42, width: int = 1024, height: int = 1024) -> Optional[Image.Image]:
    """Generate image using Nunchaku via HTTP API"""
    try:
        print(f"🎨 Calling Nunchaku API: '{prompt}' (seed: {seed})")
        
        # Call the Nunchaku API server
        response = requests.post(
            "http://localhost:8200/generate",
            json={
                "prompt": prompt,
                "seed": seed,
                "width": width,
                "height": height
            },
            timeout=60
        )
        
        if response.status_code == 200:
            data = response.json()
            
            if data["status"] == "success":
                # Decode base64 image
                img_data = base64.b64decode(data["image_base64"])
                image = Image.open(io.BytesIO(img_data))
                print("✅ Nunchaku image generated successfully via API")
                return image
            else:
                print(f"❌ API returned error: {data.get('error', 'Unknown error')}")
                return None
        else:
            print(f"❌ API request failed: {response.status_code}")
            print(f"   Response: {response.text}")
            return None
            
    except requests.exceptions.ConnectionError:
        print("❌ Cannot connect to Nunchaku API server")
        print("   Make sure to run: conda activate nun && python nunchaku_api_server.py")
        print("   Server should be running on port 8200")
        return None
    except Exception as e:
        print(f"❌ Nunchaku API call failed: {e}")
        return None

if __name__ == "__main__":
    # Test the client
    print("🧪 Testing Nunchaku HTTP client...")
    
    # First check if server is running
    try:
        health_response = requests.get("http://localhost:8097/health", timeout=5)
        if health_response.status_code == 200:
            print("✅ Nunchaku API server is running")
            
            # Test generation
            image = generate_nunchaku_image_http("A beautiful sunset", seed=42)
            if image:
                image.save("nunchaku_api_test.png")
                print("✅ Test successful! Image saved as 'nunchaku_api_test.png'")
            else:
                print("❌ Test failed!")
        else:
            print("❌ Nunchaku API server health check failed")
    except requests.exceptions.ConnectionError:
        print("❌ Nunchaku API server is not running")
        print("   Start it with: conda activate nun && python nunchaku_api_server.py")
