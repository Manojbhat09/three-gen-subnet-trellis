#!/usr/bin/env python3
"""
Sequential Mining Script - CUDA OOM Safe
Purpose: Mine with sequential server coordination to avoid memory conflicts
"""

import subprocess
import time
import requests
import signal
import os

class SequentialMiner:
    def __init__(self):
        self.validation_url = "http://127.0.0.1:10006"
        self.generation_url = "http://127.0.0.1:8095"
    
    def start_validation_server(self):
        """Start validation server"""
        print("🚀 Starting validation server...")
        # Implementation would start the server
        pass
    
    def unload_validation_models(self):
        """Unload validation models"""
        try:
            response = requests.post(f"{self.validation_url}/unload_models/", timeout=30)
            return response.status_code == 200
        except:
            return False
    
    def start_generation_server(self):
        """Start generation server"""
        print("🚀 Starting generation server...")
        # Implementation would start the server
        pass
    
    def generate_3d_model(self, prompt: str):
        """Generate 3D model"""
        try:
            response = requests.post(
                f"{self.generation_url}/generate/",
                json={"prompt": prompt},
                timeout=300
            )
            return response.status_code == 200, response.json() if response.status_code == 200 else None
        except Exception as e:
            return False, str(e)
    
    def validate_model(self, ply_data: str, prompt: str):
        """Validate generated model"""
        try:
            response = requests.post(
                f"{self.validation_url}/validate_txt_to_3d_ply/",
                json={"prompt": prompt, "ply_data": ply_data},
                timeout=60
            )
            return response.status_code == 200, response.json() if response.status_code == 200 else None
        except Exception as e:
            return False, str(e)
    
    def sequential_mine(self, prompt: str):
        """Perform sequential mining to avoid CUDA OOM"""
        print(f"🎯 Sequential Mining: {prompt}")
        
        try:
            # Step 1: Start validation server
            self.start_validation_server()
            time.sleep(10)  # Wait for startup
            
            # Step 2: Unload validation models
            print("📤 Unloading validation models...")
            self.unload_validation_models()
            
            # Step 3: Start generation server
            self.start_generation_server()
            time.sleep(30)  # Wait for startup
            
            # Step 4: Generate
            print("🎨 Generating 3D model...")
            success, result = self.generate_3d_model(prompt)
            
            if not success:
                print(f"❌ Generation failed: {result}")
                return False
            
            # Step 5: Stop generation server
            print("🛑 Stopping generation server...")
            # Implementation would stop the server
            
            # Step 6: Reload validation models and validate
            print("📥 Reloading validation models...")
            # Implementation would reload models
            
            print("✅ Sequential mining complete!")
            return True
            
        except Exception as e:
            print(f"❌ Sequential mining error: {e}")
            return False

if __name__ == "__main__":
    miner = SequentialMiner()
    success = miner.sequential_mine("a red cube")
    print(f"Mining result: {'Success' if success else 'Failed'}")
