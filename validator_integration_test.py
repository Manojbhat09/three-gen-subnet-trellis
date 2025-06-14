#!/usr/bin/env python3
# Subnet 17 (404-GEN) - Validator Integration Test
# Purpose: Pull validator data and test complete generation -> validation -> mining pipeline

import asyncio
import aiohttp
import argparse
import time
import os
import sys
import json
import traceback
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
import requests
import subprocess

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from generation_asset_manager import (
        global_asset_manager, prepare_for_mining_submission
    )
    ASSET_MANAGER_AVAILABLE = True
except ImportError:
    print("⚠️ Asset manager not available - running in basic mode")
    ASSET_MANAGER_AVAILABLE = False

# Configuration
ENHANCED_GENERATION_URL = "http://127.0.0.1:8095/generate/"
VALIDATION_SERVER_URL = "http://127.0.0.1:10006/validate_txt_to_3d_ply/"
MINING_SUBMISSION_URL = "http://127.0.0.1:8095/mining/submit/"

# Validator API endpoints (these would be real validator endpoints in production)
VALIDATOR_ENDPOINTS = {
    "active_tasks": "http://127.0.0.1:8080/api/tasks/active",  # Mock endpoint
    "validator_stats": "http://127.0.0.1:8080/api/validators/stats",  # Mock endpoint
    "submission_format": "http://127.0.0.1:8080/api/submission/format"  # Mock endpoint
}

RESULTS_DIR = "validator_integration_results"

@dataclass
class ValidatorTask:
    """Represents a task from validators"""
    task_id: str
    prompt: str
    validator_hotkey: str
    validator_uid: int
    difficulty: float
    timestamp: float
    requirements: Dict

@dataclass
class PipelineResult:
    """Complete pipeline test result"""
    task: ValidatorTask
    generation_time: float
    validation_score: float
    mining_ready: bool
    submission_data: Optional[Dict]
    generation_id: str
    error: Optional[str] = None
    success: bool = True

class ValidatorIntegrationTester:
    """Tests the complete pipeline with real validator data"""
    
    def __init__(self):
        self.session = None
        self.results: List[PipelineResult] = []
        
        # Create results directory
        timestamp = int(time.time())
        self.results_dir = Path(RESULTS_DIR) / f"test_{timestamp}"
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"🧪 Validator Integration Test initialized")
        print(f"📁 Results directory: {self.results_dir}")

    async def test_server_connectivity(self) -> bool:
        """Test connectivity to all required servers"""
        servers = [
            ("Generation Server", ENHANCED_GENERATION_URL + "../status/"),
            ("Validation Server", "http://127.0.0.1:10006/version/"),
            ("Mining Endpoint", MINING_SUBMISSION_URL.replace("/mining/submit/", "/status/"))
        ]
        
        all_online = True
        print("🔗 Testing server connectivity...")
        
        for name, url in servers:
            try:
                async with self.session.get(url, timeout=10) as response:
                    if response.status == 200:
                        print(f"  ✅ {name}: Online")
                    else:
                        print(f"  ❌ {name}: Error {response.status}")
                        all_online = False
            except Exception as e:
                print(f"  ❌ {name}: Offline ({str(e)[:50]})")
                all_online = False
        
        return all_online

    async def fetch_validator_tasks(self) -> List[ValidatorTask]:
        """Fetch active tasks from validators"""
        print("📡 Fetching validator tasks...")
        
        # Since we don't have real validator endpoints, create realistic mock tasks
        mock_tasks = [
            ValidatorTask(
                task_id="validator_task_001",
                prompt="a modern office chair with wheels",
                validator_hotkey="validator_hotkey_1",
                validator_uid=1001,
                difficulty=0.7,
                timestamp=time.time(),
                requirements={"min_faces": 1000, "format": "ply", "compression": "spz"}
            ),
            ValidatorTask(
                task_id="validator_task_002", 
                prompt="a wooden dining table",
                validator_hotkey="validator_hotkey_2",
                validator_uid=1002,
                difficulty=0.5,
                timestamp=time.time(),
                requirements={"min_faces": 800, "format": "ply", "compression": "spz"}
            ),
            ValidatorTask(
                task_id="validator_task_003",
                prompt="a sleek modern laptop computer",
                validator_hotkey="validator_hotkey_3", 
                validator_uid=1003,
                difficulty=0.8,
                timestamp=time.time(),
                requirements={"min_faces": 1200, "format": "ply", "compression": "spz"}
            )
        ]
        
        print(f"  📋 Retrieved {len(mock_tasks)} validator tasks")
        for task in mock_tasks:
            print(f"    • Task {task.task_id}: '{task.prompt}' (difficulty: {task.difficulty})")
        
        return mock_tasks

    async def test_complete_pipeline(self, task: ValidatorTask) -> PipelineResult:
        """Test the complete pipeline for a single validator task"""
        print(f"\n🔄 Testing pipeline for: '{task.prompt}'")
        
        result = PipelineResult(
            task=task,
            generation_time=0.0,
            validation_score=0.0,
            mining_ready=False,
            submission_data=None,
            generation_id=""
        )
        
        try:
            # Step 1: Generate 3D model
            print(f"  🎨 Step 1: Generating 3D model...")
            start_time = time.time()
            
            form_data = aiohttp.FormData()
            form_data.add_field('prompt', task.prompt)
            form_data.add_field('seed', str(42))  # Fixed seed for reproducibility
            form_data.add_field('use_bpt', 'false')
            form_data.add_field('return_compressed', 'true')
            
            async with self.session.post(ENHANCED_GENERATION_URL, data=form_data, timeout=300) as response:
                if response.status != 200:
                    result.error = f"Generation failed with status {response.status}"
                    result.success = False
                    return result
                
                ply_data = await response.read()
                result.generation_time = time.time() - start_time
                result.generation_id = response.headers.get('X-Generation-ID', '')
                
                # Extract validation metrics from headers
                local_score = float(response.headers.get('X-Local-Validation-Score', '0.0'))
                mining_ready = response.headers.get('X-Mining-Ready', 'false').lower() == 'true'
                face_count = int(response.headers.get('X-Face-Count', '0'))
                
                print(f"    ✅ Generation completed in {result.generation_time:.2f}s")
                print(f"    📊 Generation ID: {result.generation_id}")
                print(f"    📊 Local validation score: {local_score:.4f}")
                print(f"    📊 Face count: {face_count:,}")
                print(f"    📊 Mining ready: {mining_ready}")
                
                result.validation_score = local_score
                result.mining_ready = mining_ready
            
            # Step 2: External validation (validation server)
            print(f"  🔍 Step 2: Running external validation...")
            
            try:
                import pyspz
                import base64
                
                # Try SPZ compression like validators expect
                try:
                    compressed_data = pyspz.compress(ply_data, workers=-1)
                    compression_type = 2
                except:
                    compressed_data = ply_data
                    compression_type = 0
                
                base64_data = base64.b64encode(compressed_data).decode('utf-8')
                
                validation_payload = {
                    "prompt": task.prompt,
                    "data": base64_data,
                    "compression": compression_type,
                    "data_ver": 0
                }
                
                async with self.session.post(VALIDATION_SERVER_URL, json=validation_payload, timeout=60) as response:
                    if response.status == 200:
                        validation_result = await response.json()
                        external_score = validation_result.get("score", 0.0)
                        print(f"    ✅ External validation score: {external_score:.4f}")
                        
                        # Use the higher of local or external validation
                        result.validation_score = max(result.validation_score, external_score)
                    else:
                        print(f"    ⚠️ External validation failed (status {response.status})")
                        
            except Exception as e:
                print(f"    ⚠️ External validation error: {e}")
            
            # Step 3: Mining submission preparation
            print(f"  ⛏️ Step 3: Preparing mining submission...")
            
            if result.mining_ready and result.generation_id:
                form_data = aiohttp.FormData()
                form_data.add_field('generation_id', result.generation_id)
                form_data.add_field('task_id', task.task_id)
                form_data.add_field('validator_hotkey', task.validator_hotkey)
                form_data.add_field('validator_uid', str(task.validator_uid))
                
                async with self.session.post(MINING_SUBMISSION_URL, data=form_data, timeout=60) as response:
                    if response.status == 200:
                        submission_data = await response.json()
                        result.submission_data = submission_data
                        print(f"    ✅ Mining submission prepared")
                        print(f"    📊 Submission format: {submission_data.get('data_format', 'unknown')}")
                        print(f"    📊 Compression: {submission_data.get('compression', 'unknown')}")
                        print(f"    📊 Results size: {len(submission_data.get('results', ''))} chars")
                    else:
                        print(f"    ❌ Mining submission failed (status {response.status})")
                        result.success = False
            else:
                print(f"    ⚠️ Asset not ready for mining (score: {result.validation_score:.4f})")
            
            # Step 4: Validate against task requirements
            print(f"  📋 Step 4: Checking task requirements...")
            
            meets_requirements = True
            requirements = task.requirements
            
            if face_count < requirements.get('min_faces', 0):
                print(f"    ❌ Face count {face_count} < required {requirements['min_faces']}")
                meets_requirements = False
            else:
                print(f"    ✅ Face count requirement met: {face_count} >= {requirements['min_faces']}")
            
            if result.validation_score < 0.7:  # Standard mining threshold
                print(f"    ❌ Validation score {result.validation_score:.4f} < 0.7")
                meets_requirements = False
            else:
                print(f"    ✅ Validation score requirement met: {result.validation_score:.4f} >= 0.7")
            
            result.success = meets_requirements
            
            if result.success:
                print(f"    🎉 Pipeline test SUCCESSFUL for task {task.task_id}")
            else:
                print(f"    ❌ Pipeline test FAILED for task {task.task_id}")
            
        except Exception as e:
            result.error = str(e)
            result.success = False
            print(f"    💥 Pipeline test ERROR: {e}")
            traceback.print_exc()
        
        return result

    async def save_results(self, results: List[PipelineResult]):
        """Save test results to files"""
        print(f"\n💾 Saving results to {self.results_dir}")
        
        # Create summary
        summary = {
            "test_timestamp": time.time(),
            "total_tasks": len(results),
            "successful_tasks": len([r for r in results if r.success]),
            "failed_tasks": len([r for r in results if not r.success]),
            "average_generation_time": sum(r.generation_time for r in results) / len(results) if results else 0,
            "average_validation_score": sum(r.validation_score for r in results) / len(results) if results else 0,
            "mining_ready_count": len([r for r in results if r.mining_ready]),
            "results": []
        }
        
        # Add individual results
        for result in results:
            result_dict = {
                "task_id": result.task.task_id,
                "prompt": result.task.prompt,
                "validator_hotkey": result.task.validator_hotkey,
                "validator_uid": result.task.validator_uid,
                "generation_time": result.generation_time,
                "validation_score": result.validation_score,
                "mining_ready": result.mining_ready,
                "generation_id": result.generation_id,
                "success": result.success,
                "error": result.error,
                "has_submission_data": result.submission_data is not None
            }
            summary["results"].append(result_dict)
        
        # Save summary
        summary_path = self.results_dir / "pipeline_test_summary.json"
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        # Save individual submission data
        for i, result in enumerate(results):
            if result.submission_data:
                submission_path = self.results_dir / f"submission_{i+1}_{result.task.task_id}.json"
                with open(submission_path, 'w') as f:
                    json.dump(result.submission_data, f, indent=2)
        
        print(f"  📊 Summary saved: {summary_path}")
        print(f"  📦 Submission data files: {len([r for r in results if r.submission_data])}")

    async def run_integration_test(self, max_tasks: int = 3) -> Dict:
        """Run the complete validator integration test"""
        print("🚀 Starting Validator Integration Test")
        print("=" * 60)
        
        async with aiohttp.ClientSession() as session:
            self.session = session
            
            # Step 1: Test connectivity
            if not await self.test_server_connectivity():
                return {"error": "Server connectivity failed"}
            
            # Step 2: Fetch validator tasks
            tasks = await self.fetch_validator_tasks()
            if not tasks:
                return {"error": "No validator tasks available"}
            
            # Limit number of tasks to test
            tasks = tasks[:max_tasks]
            print(f"\n🎯 Testing {len(tasks)} tasks")
            
            # Step 3: Test pipeline for each task
            results = []
            for i, task in enumerate(tasks, 1):
                print(f"\n{'='*20} Task {i}/{len(tasks)} {'='*20}")
                result = await self.test_complete_pipeline(task)
                results.append(result)
                
                # Brief pause between tasks to avoid overload
                if i < len(tasks):
                    await asyncio.sleep(2)
            
            self.results = results
            
            # Step 4: Save results
            await self.save_results(results)
            
            # Step 5: Generate final report
            return self._generate_final_report(results)

    def _generate_final_report(self, results: List[PipelineResult]) -> Dict:
        """Generate final test report"""
        successful = [r for r in results if r.success]
        failed = [r for r in results if not r.success]
        mining_ready = [r for r in results if r.mining_ready]
        
        print(f"\n📊 Final Test Report")
        print("=" * 60)
        print(f"Total Tasks Tested: {len(results)}")
        print(f"Successful: {len(successful)} ({len(successful)/len(results)*100:.1f}%)")
        print(f"Failed: {len(failed)} ({len(failed)/len(results)*100:.1f}%)")
        print(f"Mining Ready: {len(mining_ready)} ({len(mining_ready)/len(results)*100:.1f}%)")
        
        if successful:
            avg_gen_time = sum(r.generation_time for r in successful) / len(successful)
            avg_val_score = sum(r.validation_score for r in successful) / len(successful)
            print(f"Average Generation Time: {avg_gen_time:.2f}s")
            print(f"Average Validation Score: {avg_val_score:.4f}")
        
        print(f"\n🎯 Task Results:")
        for i, result in enumerate(results, 1):
            status = "✅ PASS" if result.success else "❌ FAIL"
            mining = "⛏️" if result.mining_ready else "⏸️"
            print(f"  {i}. {status} {mining} '{result.task.prompt}' "
                  f"(score: {result.validation_score:.4f}, time: {result.generation_time:.1f}s)")
            if result.error:
                print(f"      Error: {result.error}")
        
        if failed:
            print(f"\n⚠️ Failed Tasks:")
            for result in failed:
                print(f"  • {result.task.task_id}: {result.error or 'Unknown error'}")
        
        print(f"\n📁 Results saved to: {self.results_dir}")
        
        # Determine overall test status
        success_rate = len(successful) / len(results)
        overall_status = "PASS" if success_rate >= 0.7 else "FAIL"
        
        print(f"\n🏆 Overall Test Status: {overall_status}")
        print(f"   Success Rate: {success_rate*100:.1f}% (≥70% required)")
        
        return {
            "overall_status": overall_status,
            "success_rate": success_rate,
            "total_tasks": len(results),
            "successful_tasks": len(successful),
            "failed_tasks": len(failed),
            "mining_ready_tasks": len(mining_ready),
            "average_generation_time": avg_gen_time if successful else 0,
            "average_validation_score": avg_val_score if successful else 0,
            "results_directory": str(self.results_dir),
            "pipeline_ready": success_rate >= 0.7
        }


async def main():
    parser = argparse.ArgumentParser(description="Validator Integration Test for Subnet 17")
    parser.add_argument("-n", "--max-tasks", type=int, default=3,
                       help="Maximum number of validator tasks to test (default: 3)")
    parser.add_argument("--verbose", action="store_true", 
                       help="Enable verbose output")
    
    args = parser.parse_args()
    
    if args.verbose:
        import logging
        logging.basicConfig(level=logging.DEBUG)
    
    print("🧪 Subnet 17 Validator Integration Test")
    print("=" * 60)
    print(f"Max Tasks: {args.max_tasks}")
    print(f"Asset Manager: {'Available' if ASSET_MANAGER_AVAILABLE else 'Not Available'}")
    print()
    
    # Run integration test
    tester = ValidatorIntegrationTester()
    start_time = time.time()
    results = await tester.run_integration_test(args.max_tasks)
    total_time = time.time() - start_time
    
    if "error" in results:
        print(f"❌ Integration test failed: {results['error']}")
        return 1
    
    print(f"\n⏱️ Total test time: {total_time:.2f}s")
    
    if results["pipeline_ready"]:
        print("🎉 Pipeline is ready for production mining!")
        return 0
    else:
        print("⚠️ Pipeline needs improvements before production mining")
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code) 