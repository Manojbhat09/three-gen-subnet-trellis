#!/usr/bin/env python3
"""
Startup script for HunyuanDiT + TRELLIS server with model preloading
"""

import requests
import time
import sys
import os

def preload_models(server_url: str = "http://localhost:8098"):
    """Preload models for faster generation"""
    print("🚀 Preloading models for HunyuanDiT + TRELLIS server...")
    
    try:
        # Check if server is running
        response = requests.get(f"{server_url}/health/", timeout=10)
        if response.status_code != 200:
            print(f"❌ Server not responding at {server_url}")
            return False
        
        print("✅ Server is running")
        
        # Load models
        print("🔧 Loading HunyuanDiT pipeline...")
        response = requests.post(f"{server_url}/models/load/", timeout=300)
        if response.status_code == 200:
            result = response.json()
            print(f"✅ Models loaded: {result.get('models_loaded', {})}")
        else:
            print(f"❌ Failed to load models: {response.status_code}")
            return False
        
        # Check status
        response = requests.get(f"{server_url}/status/", timeout=10)
        if response.status_code == 200:
            status = response.json()
            models_loaded = status.get('models_loaded', {})
            print(f"📊 Server Status:")
            print(f"   HunyuanDiT: {'✅' if models_loaded.get('hunyuan_pipeline') else '❌'}")
            print(f"   TRELLIS: {'✅' if models_loaded.get('trellis_pipeline') else '❌'}")
            print(f"   GPU Memory: {status.get('gpu_memory', 0):.1f}GB free")
        
        print("🎉 Models preloaded successfully!")
        print("   Ready for fast generation!")
        return True
        
    except Exception as e:
        print(f"❌ Preloading failed: {e}")
        return False

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Preload models for HunyuanDiT server")
    parser.add_argument("--server-url", default="http://localhost:8098",
                       help="Server URL")
    
    args = parser.parse_args()
    
    success = preload_models(args.server_url)
    if success:
        print("\n💡 Tips:")
        print("   - Models will stay loaded for faster generation")
        print("   - Use /models/unload/ endpoint if you need to free GPU memory")
        print("   - Use /clear_cache/ endpoint to clear GPU cache")
        print("   - Server is ready for high-speed generation!")
    else:
        print("\n❌ Failed to preload models")
        sys.exit(1)

if __name__ == "__main__":
    main() 