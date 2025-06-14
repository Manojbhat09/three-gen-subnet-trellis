#!/usr/bin/env python3
"""
Realistic Performance Test - Using Real PLY Data
Tests validation pipeline with actual 3D model data to get real scores
"""

import asyncio
import time
import requests
import json
import os
from typing import Dict, List, Tuple
import bittensor as bt

# Import our protocol
from subnet_protocol_integration import (
    MockPullTask as PullTask,
    MockSubmitResults as SubmitResults,
    Task,
    Feedback,
    MINER_LICENSE_CONSENT_DECLARATION
)

class RealisticPerformanceTest:
    """Test pipeline with realistic PLY data"""
    
    def __init__(self):
        self.validation_server_url = "http://localhost:10006"
        self.results = []
        
        # Realistic test PLY data (simple but proper geometry)
        self.test_data = [
            {
                "prompt": "a red apple",
                "ply_data": self.create_apple_ply()
            },
            {
                "prompt": "a blue chair", 
                "ply_data": self.create_chair_ply()
            },
            {
                "prompt": "a golden coin",
                "ply_data": self.create_coin_ply()
            }
        ]
        
        print("🧪 Realistic Performance Test Initialized")

    def create_apple_ply(self) -> str:
        """Create a sphere-like PLY (represents apple)"""
        import math
        
        vertices = []
        faces = []
        
        # Create icosphere vertices (20 triangular faces)
        phi = (1.0 + math.sqrt(5.0)) / 2.0  # Golden ratio
        
        # 12 vertices of icosahedron
        vertices = [
            f"0.0 {1.0/phi:.6f} {1.0:.6f}",
            f"0.0 {-1.0/phi:.6f} {1.0:.6f}",
            f"0.0 {1.0/phi:.6f} {-1.0:.6f}",
            f"0.0 {-1.0/phi:.6f} {-1.0:.6f}",
            f"{1.0/phi:.6f} {1.0:.6f} 0.0",
            f"{-1.0/phi:.6f} {1.0:.6f} 0.0",
            f"{1.0/phi:.6f} {-1.0:.6f} 0.0",
            f"{-1.0/phi:.6f} {-1.0:.6f} 0.0",
            f"{1.0:.6f} 0.0 {1.0/phi:.6f}",
            f"{-1.0:.6f} 0.0 {1.0/phi:.6f}",
            f"{1.0:.6f} 0.0 {-1.0/phi:.6f}",
            f"{-1.0:.6f} 0.0 {-1.0/phi:.6f}"
        ]
        
        # 20 triangular faces of icosahedron
        faces = [
            "3 0 8 4", "3 0 5 9", "3 0 4 5", "3 0 9 1", "3 0 1 8",
            "3 8 1 6", "3 8 6 10", "3 8 10 4", "3 4 10 2", "3 4 2 5",
            "3 5 2 11", "3 5 11 9", "3 9 11 7", "3 9 7 1", "3 1 7 6",
            "3 6 7 3", "3 6 3 10", "3 10 3 2", "3 2 3 11", "3 11 3 7"
        ]
        
        return self.create_ply_from_data(vertices, faces, "Apple-like sphere")

    def create_chair_ply(self) -> str:
        """Create a chair-like PLY"""
        # Chair with seat, back, and 4 legs
        vertices = [
            # Seat (4 corners)
            "0.0 0.5 0.0", "1.0 0.5 0.0", "1.0 0.5 1.0", "0.0 0.5 1.0",
            # Seat bottom
            "0.0 0.4 0.0", "1.0 0.4 0.0", "1.0 0.4 1.0", "0.0 0.4 1.0",
            # Back top
            "0.0 1.2 0.0", "1.0 1.2 0.0",
            # Back bottom
            "0.0 0.5 0.0", "1.0 0.5 0.0",
            # Legs bottom
            "0.1 0.0 0.1", "0.9 0.0 0.1", "0.9 0.0 0.9", "0.1 0.0 0.9"
        ]
        
        faces = [
            # Seat top
            "3 0 1 2", "3 0 2 3",
            # Seat bottom  
            "3 4 7 6", "3 4 6 5",
            # Back
            "3 8 9 1", "3 8 1 0",
            # Legs (simplified)
            "3 0 4 12", "3 1 5 13", "3 2 6 14", "3 3 7 15"
        ]
        
        return self.create_ply_from_data(vertices, faces, "Chair structure")

    def create_coin_ply(self) -> str:
        """Create a coin-like PLY (flat cylinder)"""
        import math
        
        vertices = []
        faces = []
        
        # Create cylinder with 12 sides
        num_sides = 12
        radius = 0.5
        height = 0.05
        
        # Top circle vertices
        for i in range(num_sides):
            angle = 2 * math.pi * i / num_sides
            x = radius * math.cos(angle)
            z = radius * math.sin(angle)
            vertices.append(f"{x:.6f} {height:.6f} {z:.6f}")
        
        # Bottom circle vertices
        for i in range(num_sides):
            angle = 2 * math.pi * i / num_sides
            x = radius * math.cos(angle)
            z = radius * math.sin(angle)
            vertices.append(f"{x:.6f} {-height:.6f} {z:.6f}")
        
        # Center vertices
        vertices.append(f"0.0 {height:.6f} 0.0")  # Top center
        vertices.append(f"0.0 {-height:.6f} 0.0")  # Bottom center
        
        # Top face triangles
        for i in range(num_sides):
            next_i = (i + 1) % num_sides
            faces.append(f"3 {num_sides * 2} {i} {next_i}")
        
        # Bottom face triangles
        for i in range(num_sides):
            next_i = (i + 1) % num_sides
            faces.append(f"3 {num_sides * 2 + 1} {num_sides + next_i} {num_sides + i}")
        
        # Side faces
        for i in range(num_sides):
            next_i = (i + 1) % num_sides
            faces.append(f"3 {i} {num_sides + i} {next_i}")
            faces.append(f"3 {next_i} {num_sides + i} {num_sides + next_i}")
        
        return self.create_ply_from_data(vertices, faces, "Coin cylinder")

    def create_ply_from_data(self, vertices: List[str], faces: List[str], comment: str) -> str:
        """Create PLY format string from vertex and face data"""
        ply_content = f"""ply
format ascii 1.0
comment {comment}
element vertex {len(vertices)}
property float x
property float y
property float z
element face {len(faces)}
property list uchar int vertex_indices
end_header
"""
        
        for vertex in vertices:
            ply_content += vertex + "\n"
        
        for face in faces:
            ply_content += face + "\n"
        
        return ply_content

    async def test_validation(self, prompt: str, ply_data: str) -> Dict:
        """Test validation with realistic PLY data"""
        print(f"🔍 Testing validation for: '{prompt}'")
        
        start_time = time.time()
        
        try:
            response = requests.post(
                f"{self.validation_server_url}/validate_txt_to_3d_ply/",
                json={
                    "prompt": prompt,
                    "data": ply_data,
                    "compression": 0,
                    "generate_preview": False
                },
                timeout=60
            )
            
            validation_time = time.time() - start_time
            
            if response.status_code == 200:
                result = response.json()
                score = result.get("score", 0.0)
                
                validation_data = {
                    "prompt": prompt,
                    "validation_time": validation_time,
                    "score": score,
                    "iqa": result.get("iqa", 0.0),
                    "alignment_score": result.get("alignment_score", 0.0),
                    "ssim": result.get("ssim", 0.0),
                    "lpips": result.get("lpips", 0.0),
                    "status": "success",
                    "ply_size": len(ply_data),
                    "vertex_count": ply_data.count('\n') - ply_data.find('end_header') - 10,
                    "response_details": result
                }
                
                print(f"  ✅ Score: {score:.4f} in {validation_time:.2f}s")
                print(f"     IQA: {validation_data['iqa']:.4f}")
                print(f"     Alignment: {validation_data['alignment_score']:.4f}")
                print(f"     SSIM: {validation_data['ssim']:.4f}")
                print(f"     LPIPS: {validation_data['lpips']:.4f}")
                
                return validation_data
            else:
                print(f"  ❌ Validation error: {response.status_code}")
                print(f"     Response: {response.text[:200]}")
                
                return {
                    "prompt": prompt,
                    "status": "error",
                    "error_code": response.status_code,
                    "error_text": response.text,
                    "validation_time": validation_time
                }
                
        except Exception as e:
            print(f"  ❌ Validation exception: {e}")
            return {
                "prompt": prompt,
                "status": "exception",
                "error": str(e),
                "validation_time": time.time() - start_time
            }

    async def run_performance_test(self) -> Dict:
        """Run comprehensive performance test"""
        print("🚀 REALISTIC PERFORMANCE TEST")
        print("="*60)
        
        # Check validation server
        try:
            resp = requests.get(f"{self.validation_server_url}/version/", timeout=5)
            if resp.status_code != 200:
                return {"error": "Validation server not running"}
            version_text = resp.text.strip('"')
            print(f"✅ Validation server running (v{version_text})")
        except:
            return {"error": "Cannot connect to validation server"}
        
        # Test each prompt/PLY combination
        test_start = time.time()
        results = []
        
        for i, test_case in enumerate(self.test_data, 1):
            print(f"\n🎯 Test {i}/{len(self.test_data)}")
            result = await self.test_validation(test_case["prompt"], test_case["ply_data"])
            results.append(result)
            
            # Brief pause
            await asyncio.sleep(0.5)
        
        total_time = time.time() - test_start
        
        # Generate performance report
        return self.generate_performance_report(results, total_time)

    def generate_performance_report(self, results: List[Dict], total_time: float) -> Dict:
        """Generate comprehensive performance report"""
        print("\n" + "="*70)
        print("📊 REALISTIC PERFORMANCE TEST RESULTS")
        print("="*70)
        
        successful_results = [r for r in results if r.get("status") == "success"]
        
        if not successful_results:
            print("❌ No successful validations!")
            return {"success": False, "error": "No successful validations"}
        
        # Calculate statistics
        scores = [r["score"] for r in successful_results]
        times = [r["validation_time"] for r in successful_results]
        
        avg_score = sum(scores) / len(scores)
        min_score = min(scores)
        max_score = max(scores)
        avg_time = sum(times) / len(times)
        
        print(f"\n📈 PERFORMANCE METRICS:")
        print(f"Total Test Time:       {total_time:.2f}s")
        print(f"Successful Tests:      {len(successful_results)}/{len(results)}")
        print(f"Success Rate:          {len(successful_results)/len(results)*100:.1f}%")
        print(f"Average Score:         {avg_score:.4f}")
        print(f"Score Range:           {min_score:.4f} - {max_score:.4f}")
        print(f"Average Val Time:      {avg_time:.2f}s")
        
        print(f"\n📋 DETAILED RESULTS:")
        print("-" * 70)
        for i, result in enumerate(results, 1):
            if result.get("status") == "success":
                score = result["score"]
                time_taken = result["validation_time"]
                threshold_status = "✅ PASS" if score >= 0.6 else "❌ FAIL"
                print(f"{i}. '{result['prompt'][:25]:25s}' - "
                      f"Score: {score:.4f} - "
                      f"Time: {time_taken:.2f}s - {threshold_status}")
            else:
                print(f"{i}. '{result['prompt'][:25]:25s}' - "
                      f"❌ ERROR: {result.get('error', 'Unknown')}")
        
        # Performance assessment
        print(f"\n🏆 PERFORMANCE ASSESSMENT:")
        
        if avg_score >= 0.7:
            assessment = "🎯 EXCELLENT - High validation scores!"
            readiness = "PRODUCTION READY"
        elif avg_score >= 0.6:
            assessment = "✅ GOOD - Acceptable validation scores"
            readiness = "READY FOR MINING"
        elif avg_score >= 0.4:
            assessment = "⚠️ MODERATE - Scores need improvement"
            readiness = "NEEDS TUNING"
        else:
            assessment = "❌ POOR - Low validation scores"
            readiness = "NOT READY"
        
        print(f"   {assessment}")
        print(f"   Average validation time: {avg_time:.2f}s")
        print(f"   Validation throughput: {1/avg_time:.1f} validations/second")
        
        above_threshold = len([s for s in scores if s >= 0.6])
        print(f"   Tasks above threshold (≥0.6): {above_threshold}/{len(scores)}")
        
        print(f"\n🎯 STATUS: {readiness}")
        print("="*70)
        
        return {
            "success": True,
            "total_tests": len(results),
            "successful_tests": len(successful_results),
            "avg_score": avg_score,
            "min_score": min_score,
            "max_score": max_score,
            "avg_validation_time": avg_time,
            "total_time": total_time,
            "above_threshold": above_threshold,
            "readiness": readiness,
            "assessment": assessment,
            "detailed_results": results
        }


async def main():
    """Main function"""
    test = RealisticPerformanceTest()
    
    try:
        results = await test.run_performance_test()
        
        if results.get("success"):
            print(f"\n🎉 Performance test completed!")
            print(f"📊 Average score: {results['avg_score']:.4f}")
            print(f"⏱️ Average time: {results['avg_validation_time']:.2f}s")
            print(f"🎯 Status: {results['readiness']}")
            return 0
        else:
            print(f"\n❌ Performance test failed: {results.get('error')}")
            return 1
            
    except Exception as e:
        print(f"\n💥 Test error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    import sys
    exit_code = asyncio.run(main())
    sys.exit(exit_code) 