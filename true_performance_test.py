#!/usr/bin/env python3
"""
True Performance Test - Using a Valid Gaussian Splatting PLY File
This test uses a known-good PLY file to get a definitive performance score
from the validation server, proving the pipeline works with correct data.
"""

import asyncio
import time
import requests
import json
import os
import pybase64
from typing import Dict, List, Tuple

class TruePerformanceTest:
    """Tests the validation server with a known-valid GS PLY file."""
    
    def __init__(self, ply_path: str, prompt: str):
        self.validation_server_url = "http://localhost:10006"
        self.ply_path = ply_path
        self.prompt = prompt
        
        if not os.path.exists(self.ply_path):
            raise FileNotFoundError(f"Test PLY file not found at: {self.ply_path}")
            
        print("✅ True Performance Test Initialized")
        print(f"   - PLY File: {self.ply_path}")
        print(f"   - Prompt: '{self.prompt}'")

    async def run_test(self) -> Dict:
        """Runs the definitive validation test."""
        print("\n🚀 Running True Performance Test...")
        print("==================================================")
        
        # 1. Check if Validation Server is running
        try:
            resp = requests.get(f"{self.validation_server_url}/version/", timeout=5)
            if resp.status_code != 200:
                print("❌ ERROR: Validation server is not running or not responding.")
                return {"success": False, "error": "Validation server not found."}
            version = resp.text.strip('"')
            print(f"✅ Validation server is running (v{version})")
        except requests.ConnectionError:
            print("❌ ERROR: Cannot connect to the validation server.")
            return {"success": False, "error": "Connection to validation server failed."}

        # 2. Read and encode the PLY file
        try:
            with open(self.ply_path, "rb") as f:
                ply_content_bytes = f.read()
            
            # The API expects a base64 encoded string
            encoded_ply_data = pybase64.b64encode(ply_content_bytes).decode('utf-8')
            print(f"✅ Successfully read and encoded PLY file ({len(ply_content_bytes)} bytes)")
        except Exception as e:
            print(f"❌ ERROR: Failed to read or encode PLY file: {e}")
            return {"success": False, "error": f"File I/O error: {e}"}

        # 3. Send the data to the validation endpoint
        print("🔍 Sending data to validation server for scoring...")
        start_time = time.time()
        
        try:
            response = requests.post(
                f"{self.validation_server_url}/validate_txt_to_3d_ply/",
                json={
                    "prompt": self.prompt,
                    "data": encoded_ply_data,
                    "compression": 0,  # 0 for no compression
                    "generate_preview": True
                },
                timeout=120  # Increased timeout for potentially complex models
            )
            
            validation_time = time.time() - start_time
            
            # 4. Analyze the response
            if response.status_code == 200:
                result = response.json()
                self.print_results(result, validation_time)
                return {"success": True, "result": result, "validation_time": validation_time}
            else:
                print(f"❌ VALIDATION FAILED (HTTP {response.status_code})")
                print(f"   Response: {response.text[:500]}")
                return {"success": False, "error": f"HTTP {response.status_code}", "details": response.text}

        except requests.Timeout:
            print("❌ ERROR: The request to the validation server timed out.")
            return {"success": False, "error": "Request timeout."}
        except Exception as e:
            print(f"❌ ERROR: An unexpected error occurred: {e}")
            return {"success": False, "error": str(e)}

    def print_results(self, result: Dict, validation_time: float):
        """Prints a formatted report of the validation results."""
        score = result.get('score', 0.0)
        
        print("\n🎉 VALIDATION COMPLETE!")
        print("--------------------------------------------------")
        print(f"  Validation Time: {validation_time:.2f} seconds")
        print(f"  Final Score:     {score:.4f}")
        print("--------------------------------------------------")
        print(f"  IQA Score:       {result.get('iqa', 0.0):.4f}")
        print(f"  Alignment Score: {result.get('alignment_score', 0.0):.4f}")
        print(f"  SSIM Score:      {result.get('ssim', 0.0):.4f}")
        print(f"  LPIPS Score:     {result.get('lpips', 0.0):.4f}")
        
        if result.get('preview'):
            print("\n🖼️  Preview image was generated successfully.")
            # To save the preview image:
            # import base64
            # with open("preview.png", "wb") as f:
            #     f.write(base64.b64decode(result['preview']))
        else:
            print("\n🖼️  No preview image was generated (score might be below threshold).")
            
        print("\n🏆 ASSESSMENT:")
        if score >= 0.7:
            print("   EXCELLENT! The model is a high-quality match for the prompt.")
        elif score >= 0.6:
            print("   GOOD! This model would likely be accepted by validators.")
        elif score > 0.0:
            print("   MODERATE. The model has some quality but may not meet the threshold.")
        else:
            print("   POOR. The model has very low quality or is not a good match.")
        
        print("\nThis confirms the validation server is working correctly with valid data!")
        print("==================================================")


async def main():
    """Main function to run the test."""
    # This is the known-good test file from the validation repo
    ply_file = "validation/tests/resources/hamburger.ply"
    prompt = "a hamburger"
    
    test = TruePerformanceTest(ply_path=ply_file, prompt=prompt)
    await test.run_test()


if __name__ == "__main__":
    asyncio.run(main()) 