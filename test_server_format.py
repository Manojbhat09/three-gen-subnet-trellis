#!/usr/bin/env python3

import requests
import sys

def test_server_format():
    print("Testing server response format...")
    
    try:
        response = requests.post(
            "http://127.0.0.1:8096/generate/",
            data={"prompt": "a_motorcycle", "return_compressed": True}
        )
        
        print(f"Status code: {response.status_code}")
        print(f"Response size: {len(response.content)} bytes")
        print(f"Content-Type: {response.headers.get('content-type', 'Not specified')}")
        
        # Check if it's JSON
        try:
            import json
            json_data = response.json()
            print("Response is JSON format")
            print(f"JSON keys: {list(json_data.keys())}")
            return json_data
        except:
            print("Response is not JSON")
            
        # Check first few bytes
        if len(response.content) > 0:
            print(f"First 50 bytes: {response.content[:50]}")
            print(f"First 50 bytes (hex): {response.content[:50].hex()}")
            
        return response.content
        
    except Exception as e:
        print(f"Error: {e}")
        return None

if __name__ == "__main__":
    test_server_format() 