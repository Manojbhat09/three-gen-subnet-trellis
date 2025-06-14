#!/usr/bin/env python3
"""
Local Pipeline Test - Test Generation + Validation Pipeline
Purpose: Test the complete mining pipeline with mock tasks
"""

import asyncio
import time
import requests
import json
from typing import Tuple, Optional

class LocalPipelineTester:
    """Test the complete mining pipeline locally"""
    
    def __init__(self):
        self.generation_server_url = "http://localhost:8095"
        self.validation_server_url = "http://localhost:10006"
        
        # Test prompts of varying complexity
        self.test_prompts = [
            "a red apple",
            "a blue modern chair", 
            "a purple durable robotic arm",
            "a golden treasure chest with intricate details",
            "a futuristic spaceship with glowing engines"
        ]
        
        self.results = []

    def check_services(self) -> bool:
        """Check if both services are running"""
        try:
            # Check generation server
            gen_response = requests.get(f"{self.generation_server_url}/health/", timeout=5)
            if gen_response.status_code != 200:
                print(f"❌ Generation server not available: {gen_response.status_code}")
                return False
            
            # Check validation server  
            val_response = requests.get(f"{self.validation_server_url}/version/", timeout=5)
            if val_response.status_code != 200:
                print(f"❌ Validation server not available: {val_response.status_code}")
                return False
                
            print("✅ Both services are running")
            return True
            
        except requests.exceptions.RequestException as e:
            print(f"❌ Service check failed: {e}")
            return False

    async def generate_3d_model(self, prompt: str) -> Tuple[Optional[str], float, dict]:
        """Generate 3D model and return PLY data, generation time, and metadata"""
        try:
            print(f"🎨 Generating: '{prompt}'")
            
            start_time = time.time()
            
            response = requests.post(
                f"{self.generation_server_url}/generate/",
                data={
                    "prompt": prompt,
                    "seed": 42,
                    "use_bpt": True,
                    "return_compressed": False  # Get uncompressed PLY data
                },
                timeout=300
            )
            
            generation_time = time.time() - start_time
            
            if response.status_code == 200:
                # The response is binary PLY data, not JSON
                ply_data = response.content.decode('utf-8')  # PLY is text format
                
                # Get metadata from headers
                headers = response.headers
                metadata = {
                    "generation_id": headers.get("X-Generation-ID", "unknown"),
                    "generation_time": float(headers.get("X-Generation-Time", generation_time)),
                    "face_count": int(headers.get("X-Face-Count", 0)),
                    "vertex_count": int(headers.get("X-Vertex-Count", 0)),
                    "mesh_quality_score": float(headers.get("X-Mesh-Quality-Score", 0.0)),
                    "local_validation_score": float(headers.get("X-Local-Validation-Score", 0.0)),
                    "mining_ready": headers.get("X-Mining-Ready", "false").lower() == "true",
                    "compression_type": int(headers.get("X-Compression-Type", 0)),
                    "ply_size": len(ply_data)
                }
                
                print(f"  ✅ Generated in {generation_time:.2f}s")
                print(f"     PLY size: {len(ply_data):,} chars")
                print(f"     Face count: {metadata['face_count']:,}")
                print(f"     Vertex count: {metadata['vertex_count']:,}")
                print(f"     Local validation: {metadata['local_validation_score']:.4f}")
                print(f"     Mining ready: {metadata['mining_ready']}")
                
                return ply_data, generation_time, metadata
            else:
                print(f"  ❌ Generation server error: {response.status_code}")
                print(f"     Response: {response.text[:200]}...")
                return None, generation_time, {}
                
        except Exception as e:
            print(f"  ❌ Generation error: {e}")
            return None, 0.0, {}

    async def validate_locally(self, prompt: str, ply_data: str) -> Tuple[float, float, dict]:
        """Validate the generated model locally"""
        try:
            print(f"🔍 Validating...")
            
            start_time = time.time()
            
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
                
                print(f"  ✅ Validated in {validation_time:.2f}s")
                print(f"     Score: {score:.4f}")
                print(f"     IQA: {result.get('iqa', 0.0):.4f}")
                print(f"     Alignment: {result.get('alignment_score', 0.0):.4f}")
                print(f"     SSIM: {result.get('ssim', 0.0):.4f}")
                print(f"     LPIPS: {result.get('lpips', 0.0):.4f}")
                
                validation_data = {
                    "score": score,
                    "iqa": result.get("iqa", 0.0),
                    "alignment_score": result.get("alignment_score", 0.0),
                    "ssim": result.get("ssim", 0.0),
                    "lpips": result.get("lpips", 0.0),
                    "validation_time": validation_time
                }
                
                return score, validation_time, validation_data
            else:
                print(f"  ❌ Validation server error: {response.status_code}")
                return 0.0, validation_time, {}
                
        except Exception as e:
            print(f"  ❌ Validation error: {e}")
            return 0.0, 0.0, {}

    async def test_single_prompt(self, prompt: str) -> dict:
        """Test complete pipeline for a single prompt"""
        print(f"\n{'='*50}")
        print(f"Testing: '{prompt}'")
        print(f"{'='*50}")
        
        # Generate 3D model
        ply_data, gen_time, gen_metadata = await self.generate_3d_model(prompt)
        
        if not ply_data:
            return {
                "prompt": prompt,
                "success": False,
                "error": "Generation failed",
                "generation_time": gen_time
            }
        
        # Validate locally
        score, val_time, val_data = await self.validate_locally(prompt, ply_data)
        
        # Compile results
        result = {
            "prompt": prompt,
            "success": True,
            "generation_time": gen_time,
            "validation_time": val_time,
            "total_time": gen_time + val_time,
            "validation_score": score,
            "ply_size": len(ply_data),
            "generation_metadata": gen_metadata,
            "validation_data": val_data
        }
        
        return result

    async def run_comprehensive_test(self):
        """Run comprehensive pipeline test"""
        print("🧪 LOCAL MINING PIPELINE TEST")
        print("="*60)
        
        # Check services
        if not self.check_services():
            print("❌ Services not available. Please start:")
            print("   1. Generation server: conda run -n hunyuan3d python flux_hunyuan_bpt_generation_server.py")
            print("   2. Validation server: conda run -n three-gen-validation python validation/serve.py")
            return
        
        print(f"\n🎯 Testing {len(self.test_prompts)} prompts...")
        
        # Test each prompt
        for i, prompt in enumerate(self.test_prompts, 1):
            print(f"\n⛏️ Test {i}/{len(self.test_prompts)}")
            result = await self.test_single_prompt(prompt)
            self.results.append(result)
            
            # Brief pause between tests
            await asyncio.sleep(1)
        
        # Generate comprehensive report
        self.generate_report()

    def generate_report(self):
        """Generate comprehensive performance report"""
        print("\n" + "="*80)
        print("📊 COMPREHENSIVE PIPELINE PERFORMANCE REPORT")
        print("="*80)
        
        successful_tests = [r for r in self.results if r.get("success", False)]
        failed_tests = [r for r in self.results if not r.get("success", False)]
        
        print(f"\n📈 OVERALL STATISTICS:")
        print(f"Total Tests:       {len(self.results)}")
        print(f"Successful:        {len(successful_tests)}")
        print(f"Failed:            {len(failed_tests)}")
        print(f"Success Rate:      {len(successful_tests)/len(self.results)*100:.1f}%")
        
        if successful_tests:
            # Generation performance
            gen_times = [r["generation_time"] for r in successful_tests]
            val_times = [r["validation_time"] for r in successful_tests]
            total_times = [r["total_time"] for r in successful_tests]
            scores = [r["validation_score"] for r in successful_tests]
            ply_sizes = [r["ply_size"] for r in successful_tests]
            
            print(f"\n⏱️ TIMING PERFORMANCE:")
            print(f"Avg Generation:    {sum(gen_times)/len(gen_times):.2f}s")
            print(f"Avg Validation:    {sum(val_times)/len(val_times):.2f}s")
            print(f"Avg Total:         {sum(total_times)/len(total_times):.2f}s")
            print(f"Min Generation:    {min(gen_times):.2f}s")
            print(f"Max Generation:    {max(gen_times):.2f}s")
            
            print(f"\n🎯 VALIDATION PERFORMANCE:")
            print(f"Avg Score:         {sum(scores)/len(scores):.4f}")
            print(f"Min Score:         {min(scores):.4f}")
            print(f"Max Score:         {max(scores):.4f}")
            print(f"Scores ≥ 0.6:      {len([s for s in scores if s >= 0.6])}/{len(scores)}")
            
            print(f"\n📦 MODEL SIZE ANALYSIS:")
            print(f"Avg PLY Size:      {sum(ply_sizes)/len(ply_sizes):,.0f} chars")
            print(f"Min PLY Size:      {min(ply_sizes):,} chars")
            print(f"Max PLY Size:      {max(ply_sizes):,} chars")
        
        print(f"\n📋 DETAILED RESULTS:")
        print("-" * 80)
        for i, result in enumerate(self.results, 1):
            if result.get("success"):
                print(f"{i:2d}. '{result['prompt'][:30]}...' - "
                      f"Score: {result['validation_score']:.4f} - "
                      f"Time: {result['total_time']:.1f}s - "
                      f"Size: {result['ply_size']:,}")
            else:
                print(f"{i:2d}. '{result['prompt'][:30]}...' - FAILED: {result.get('error', 'Unknown')}")
        
        if failed_tests:
            print(f"\n❌ FAILED TESTS:")
            for result in failed_tests:
                print(f"   - '{result['prompt']}': {result.get('error', 'Unknown error')}")
        
        # Performance assessment
        if successful_tests:
            avg_score = sum(scores)/len(scores)
            avg_time = sum(total_times)/len(total_times)
            
            print(f"\n🏆 PERFORMANCE ASSESSMENT:")
            if avg_score >= 0.7:
                print("   ✅ EXCELLENT - High validation scores!")
            elif avg_score >= 0.6:
                print("   ✅ GOOD - Acceptable validation scores")
            elif avg_score >= 0.4:
                print("   ⚠️ MODERATE - Validation scores need improvement")
            else:
                print("   ❌ POOR - Low validation scores")
            
            if avg_time <= 60:
                print("   ✅ FAST - Quick generation pipeline")
            elif avg_time <= 120:
                print("   ✅ ACCEPTABLE - Reasonable generation speed")
            else:
                print("   ⚠️ SLOW - Generation pipeline needs optimization")
        
        print("\n" + "="*80)


async def main():
    """Main test function"""
    tester = LocalPipelineTester()
    await tester.run_comprehensive_test()


if __name__ == "__main__":
    asyncio.run(main()) 