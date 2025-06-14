#!/usr/bin/env python3
"""
Complete Mining Pipeline - Simulation & Real Task Handler
Purpose: Comprehensive solution for mining with real task simulation
"""

import asyncio
import time
import base64
import json
import subprocess
import requests
import os
import signal
import threading
from typing import List, Dict, Optional, Tuple
import bittensor as bt
import traceback
from pydantic import BaseModel, Field
import uuid

# Import our protocol classes
from subnet_protocol_integration import (
    MockPullTask as PullTask,
    MockSubmitResults as SubmitResults,
    Task,
    Feedback,
    MINER_LICENSE_CONSENT_DECLARATION
)

class CompleteMiningPipeline:
    """Complete mining pipeline with simulation and real task handling"""
    
    def __init__(self, wallet_name: str, hotkey_name: str):
        self.wallet_name = wallet_name
        self.hotkey_name = hotkey_name
        self.wallet = None
        self.dendrite = None
        self.metagraph = None
        self.subtensor = None
        
        # Service endpoints
        self.generation_server_url = "http://localhost:8095"
        self.validation_server_url = "http://localhost:10006"
        
        # Service processes
        self.validation_process = None
        self.generation_process = None
        
        # Performance tracking
        self.stats = {
            "real_tasks_pulled": 0,
            "simulated_tasks": 0,
            "successful_generations": 0,
            "successful_validations": 0,
            "successful_submissions": 0,
            "avg_generation_time": 0.0,
            "avg_validation_time": 0.0,
            "avg_validation_score": 0.0,
            "validation_scores": [],
            "generation_times": [],
            "validation_times": [],
            "errors": []
        }
        
        # Test prompts for simulation
        self.test_prompts = [
            "a red apple",
            "a blue chair", 
            "a golden coin",
            "a wooden box",
            "a silver spoon"
        ]
        
        print("🚀 Complete Mining Pipeline Initialized")
        print(f"💰 Wallet: {wallet_name}")
        print(f"🔑 Hotkey: {hotkey_name}")

    async def initialize_bittensor(self) -> bool:
        """Initialize Bittensor components"""
        try:
            print("🔧 Initializing Bittensor...")
            
            self.wallet = bt.wallet(name=self.wallet_name, hotkey=self.hotkey_name)
            self.subtensor = bt.subtensor()
            self.dendrite = bt.dendrite(wallet=self.wallet)
            self.metagraph = self.subtensor.metagraph(netuid=17)
            
            print(f"  ✅ Wallet: {self.wallet.hotkey.ss58_address}")
            print(f"  ✅ Network: {self.subtensor.network}")
            print(f"  ✅ Neurons: {len(self.metagraph.hotkeys)}")
            
            return True
            
        except Exception as e:
            print(f"  ❌ Bittensor initialization failed: {e}")
            self.stats["errors"].append(f"Bittensor init: {e}")
            return False

    def start_services(self) -> bool:
        """Start validation and generation servers with optimization"""
        print("\n🚀 Starting Services...")
        
        # Kill any existing processes
        self.cleanup_processes()
        
        # Clear GPU memory first
        os.system("python -c 'import torch; torch.cuda.empty_cache()' 2>/dev/null")
        
        try:
            # Start validation server
            print("  📊 Starting validation server...")
            self.validation_process = subprocess.Popen(
                ["conda", "run", "-n", "three-gen-validation", 
                 "python", "serve.py", "--host", "0.0.0.0", "--port", "10006"],
                cwd="validation",
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True
            )
            
            # Start generation server with memory optimization
            print("  🎨 Starting generation server...")
            env = os.environ.copy()
            env["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512"  # Memory optimization
            
            self.generation_process = subprocess.Popen(
                ["conda", "run", "-n", "hunyuan3d",
                 "python", "flux_hunyuan_bpt_generation_server.py"],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                env=env
            )
            
            # Wait for services to start
            print("  ⏳ Waiting for services to initialize...")
            time.sleep(30)  # Give more time for model loading
            
            # Check if services are running
            if self.check_services():
                print("  ✅ All services running!")
                return True
            else:
                print("  ❌ Service startup failed")
                return False
                
        except Exception as e:
            print(f"  ❌ Service startup error: {e}")
            self.stats["errors"].append(f"Service startup: {e}")
            return False

    def check_services(self) -> bool:
        """Check if both services are healthy"""
        try:
            # Check validation server
            val_resp = requests.get(f"{self.validation_server_url}/version/", timeout=10)
            if val_resp.status_code != 200:
                return False
            
            # Check generation server  
            gen_resp = requests.get(f"{self.generation_server_url}/health/", timeout=10)
            if gen_resp.status_code != 200:
                return False
                
            return True
        except:
            return False

    def cleanup_processes(self):
        """Clean up any existing server processes"""
        os.system("pkill -f 'serve.py' 2>/dev/null")
        os.system("pkill -f 'flux_hunyuan_bpt_generation_server' 2>/dev/null")
        time.sleep(2)

    async def generate_3d_model(self, prompt: str) -> Tuple[Optional[str], float, Dict]:
        """Generate 3D model with retry logic"""
        max_retries = 2
        
        for attempt in range(max_retries):
            try:
                print(f"🎨 Generating: '{prompt}' (attempt {attempt + 1})")
                
                start_time = time.time()
                
                response = requests.post(
                    f"{self.generation_server_url}/generate/",
                    data={
                        "prompt": prompt,
                        "seed": 42,
                        "use_bpt": False,  # Disable BPT to save memory
                        "return_compressed": False
                    },
                    timeout=180  # 3 minutes
                )
                
                generation_time = time.time() - start_time
                
                if response.status_code == 200:
                    ply_data = response.content.decode('utf-8')
                    headers = response.headers
                    
                    metadata = {
                        "generation_id": headers.get("X-Generation-ID", "unknown"),
                        "generation_time": float(headers.get("X-Generation-Time", generation_time)),
                        "face_count": int(headers.get("X-Face-Count", 0)),
                        "vertex_count": int(headers.get("X-Vertex-Count", 0)),
                        "local_validation_score": float(headers.get("X-Local-Validation-Score", 0.0)),
                        "mining_ready": headers.get("X-Mining-Ready", "false").lower() == "true",
                        "ply_size": len(ply_data)
                    }
                    
                    print(f"  ✅ Generated in {generation_time:.2f}s")
                    print(f"     Face count: {metadata['face_count']:,}")
                    print(f"     Vertex count: {metadata['vertex_count']:,}")
                    print(f"     PLY size: {metadata['ply_size']:,}")
                    
                    self.stats["successful_generations"] += 1
                    self.stats["generation_times"].append(generation_time)
                    
                    return ply_data, generation_time, metadata
                    
                elif response.status_code == 500 and ("out of memory" in response.text.lower() or "cuda" in response.text.lower()):
                    print(f"  ⚠️ Memory issue detected, restarting generation server...")
                    # Kill and restart generation server
                    os.system("pkill -f 'flux_hunyuan_bpt_generation_server' 2>/dev/null")
                    time.sleep(3)
                    
                    # Restart with lower memory usage
                    env = os.environ.copy()
                    env["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:256,garbage_collection_threshold:0.6"
                    
                    self.generation_process = subprocess.Popen(
                        ["conda", "run", "-n", "hunyuan3d",
                         "python", "flux_hunyuan_bpt_generation_server.py"],
                        stdout=subprocess.PIPE,
                        stderr=subprocess.STDOUT,
                        text=True,
                        env=env
                    )
                    
                    time.sleep(20)  # Wait for restart
                    
                    if attempt == max_retries - 1:
                        print(f"  ❌ Memory issue persists after server restart")
                        self.stats["errors"].append(f"Generation OOM after restart: {prompt}")
                        return None, generation_time, {}
                else:
                    print(f"  ❌ Generation error: {response.status_code}")
                    print(f"     Response: {response.text[:200]}...")
                    
                    if attempt == max_retries - 1:
                        self.stats["errors"].append(f"Generation failed {response.status_code}: {prompt}")
                        return None, generation_time, {}
                        
            except requests.exceptions.Timeout:
                print(f"  ⏰ Generation timeout")
                if attempt == max_retries - 1:
                    self.stats["errors"].append(f"Generation timeout: {prompt}")
                    return None, 0.0, {}
            except requests.exceptions.ConnectionError:
                print(f"  📡 Connection error - server may be down")
                if attempt == max_retries - 1:
                    self.stats["errors"].append(f"Generation connection error: {prompt}")
                    return None, 0.0, {}
            except Exception as e:
                print(f"  ❌ Generation exception: {e}")
                if attempt == max_retries - 1:
                    self.stats["errors"].append(f"Generation exception: {e}")
                    return None, 0.0, {}
        
        return None, 0.0, {}

    async def validate_locally(self, prompt: str, ply_data: str) -> Tuple[float, float, Dict]:
        """Validate generated model locally"""
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
                
                validation_data = {
                    "score": score,
                    "iqa": result.get("iqa", 0.0),
                    "alignment_score": result.get("alignment_score", 0.0),
                    "ssim": result.get("ssim", 0.0),
                    "lpips": result.get("lpips", 0.0),
                    "validation_time": validation_time
                }
                
                print(f"  ✅ Validated in {validation_time:.2f}s")
                print(f"     Score: {score:.4f}")
                print(f"     IQA: {validation_data['iqa']:.4f}")
                print(f"     Alignment: {validation_data['alignment_score']:.4f}")
                
                self.stats["successful_validations"] += 1
                self.stats["validation_scores"].append(score)
                self.stats["validation_times"].append(validation_time)
                
                return score, validation_time, validation_data
            else:
                print(f"  ❌ Validation error: {response.status_code}")
                self.stats["errors"].append(f"Validation failed: {response.status_code}")
                return 0.0, validation_time, {}
                
        except Exception as e:
            print(f"  ❌ Validation exception: {e}")
            self.stats["errors"].append(f"Validation exception: {e}")
            return 0.0, 0.0, {}

    async def get_real_task_from_validators(self) -> Optional[Dict]:
        """Try to get real task from validators"""
        if not self.metagraph:
            return None
            
        # Find active validators
        active_validators = []
        for uid, (stake, axon, hotkey) in enumerate(zip(
            self.metagraph.stake, self.metagraph.axons, self.metagraph.hotkeys
        )):
            if stake > 100 and axon.is_serving and axon.ip != '0.0.0.0':
                active_validators.append((uid, stake, axon, hotkey))
        
        active_validators.sort(key=lambda x: x[1], reverse=True)
        
        # Try validators
        for uid, stake, axon, hotkey in active_validators[:10]:
            try:
                pull_synapse = PullTask()
                
                response = await self.dendrite.call(
                    target_axon=axon,
                    synapse=pull_synapse,
                    deserialize=False,
                    timeout=15.0
                )
                
                if response and hasattr(response, 'task') and response.task:
                    print(f"🎉 REAL TASK FOUND from validator {uid}!")
                    print(f"   Task ID: {response.task.id}")
                    print(f"   Prompt: '{response.task.prompt}'")
                    print(f"   Validation threshold: {response.validation_threshold}")
                    
                    self.stats["real_tasks_pulled"] += 1
                    
                    return {
                        "task_id": response.task.id,
                        "prompt": response.task.prompt,
                        "task": response.task,
                        "validation_threshold": response.validation_threshold,
                        "validator_hotkey": hotkey,
                        "validator_uid": uid,
                        "axon": axon,
                        "is_real_task": True
                    }
                    
            except Exception as e:
                continue
        
        return None

    def create_simulated_task(self, prompt: str) -> Dict:
        """Create simulated task for testing"""
        task = Task()
        task.id = str(uuid.uuid4())
        task.prompt = prompt
        
        return {
            "task_id": task.id,
            "prompt": prompt,
            "task": task,
            "validation_threshold": 0.6,  # Standard threshold
            "validator_hotkey": "simulated_validator",
            "validator_uid": -1,
            "axon": None,
            "is_real_task": False
        }

    async def submit_results_to_validator(self, task_data: Dict, ply_data: str, 
                                        validation_score: float) -> bool:
        """Submit results to validator (real or simulated)"""
        if not task_data["is_real_task"]:
            # Simulated submission - just log success
            print(f"📤 Simulated submission - Score: {validation_score:.4f}")
            if validation_score >= task_data["validation_threshold"]:
                print(f"  ✅ Would be accepted (≥{task_data['validation_threshold']:.2f})")
                self.stats["successful_submissions"] += 1
                return True
            else:
                print(f"  ❌ Would be rejected (<{task_data['validation_threshold']:.2f})")
                return False
        
        # Real submission
        try:
            print(f"📤 Submitting to real validator {task_data['validator_uid']}")
            
            submit_time = time.time_ns()
            
            message = (
                f"{MINER_LICENSE_CONSENT_DECLARATION}"
                f"{submit_time}{task_data['prompt']}"
                f"{task_data['validator_hotkey']}{self.wallet.hotkey.ss58_address}"
            )
            signature = base64.b64encode(self.wallet.hotkey.sign(message)).decode()
            
            submit_synapse = SubmitResults()
            submit_synapse.task = task_data["task"]
            submit_synapse.results = ply_data
            submit_synapse.compression = 0
            submit_synapse.data_format = "ply"
            submit_synapse.data_ver = 0
            submit_synapse.submit_time = submit_time
            submit_synapse.signature = signature
            
            response = await self.dendrite.call(
                target_axon=task_data["axon"],
                synapse=submit_synapse,
                deserialize=False,
                timeout=300.0
            )
            
            if response and hasattr(response, 'feedback') and response.feedback:
                if not response.feedback.validation_failed:
                    print(f"  ✅ Real submission successful!")
                    print(f"     Fidelity: {response.feedback.task_fidelity_score:.4f}")
                    print(f"     Reward: {response.feedback.current_miner_reward:.4f}")
                    self.stats["successful_submissions"] += 1
                    return True
                else:
                    print(f"  ❌ Real submission failed validation")
                    return False
            else:
                print(f"  ❌ No feedback from real validator")
                return False
                
        except Exception as e:
            print(f"  ❌ Real submission error: {e}")
            self.stats["errors"].append(f"Real submission: {e}")
            return False

    async def process_single_task(self, task_data: Dict) -> Dict:
        """Process a single task (real or simulated)"""
        task_type = "REAL" if task_data["is_real_task"] else "SIMULATED"
        print(f"\n{'='*50}")
        print(f"Processing {task_type} Task: '{task_data['prompt']}'")
        print(f"{'='*50}")
        
        start_time = time.time()
        
        # Generate 3D model
        ply_data, gen_time, gen_metadata = await self.generate_3d_model(task_data["prompt"])
        
        if not ply_data:
            return {
                "task_id": task_data["task_id"],
                "success": False,
                "error": "Generation failed",
                "task_type": task_type
            }
        
        # Validate locally
        val_score, val_time, val_data = await self.validate_locally(task_data["prompt"], ply_data)
        
        # Submit results
        submission_success = await self.submit_results_to_validator(task_data, ply_data, val_score)
        
        total_time = time.time() - start_time
        
        result = {
            "task_id": task_data["task_id"],
            "prompt": task_data["prompt"],
            "task_type": task_type,
            "success": True,
            "generation_time": gen_time,
            "validation_time": val_time,
            "total_time": total_time,
            "validation_score": val_score,
            "validation_threshold": task_data["validation_threshold"],
            "submission_success": submission_success,
            "ply_size": len(ply_data),
            "generation_metadata": gen_metadata,
            "validation_data": val_data
        }
        
        print(f"\n📋 Task Summary:")
        print(f"   Type: {task_type}")
        print(f"   Generation: {gen_time:.2f}s")
        print(f"   Validation: {val_time:.2f}s")
        print(f"   Total: {total_time:.2f}s")
        print(f"   Score: {val_score:.4f} (threshold: {task_data['validation_threshold']:.2f})")
        print(f"   Submission: {'✅ SUCCESS' if submission_success else '❌ FAILED'}")
        
        return result

    async def run_comprehensive_pipeline(self, num_simulated_tasks: int = 3) -> Dict:
        """Run comprehensive mining pipeline test"""
        print("🚀 COMPLETE MINING PIPELINE")
        print("="*60)
        
        # Initialize Bittensor
        if not await self.initialize_bittensor():
            return {"error": "Bittensor initialization failed"}
        
        # Start services
        if not self.start_services():
            return {"error": "Service startup failed"}
        
        # Try to get real tasks first
        print("\n🎯 Checking for Real Tasks...")
        real_task = await self.get_real_task_from_validators()
        
        tasks_to_process = []
        
        if real_task:
            tasks_to_process.append(real_task)
        else:
            print("  ❌ No real tasks available")
        
        # Add simulated tasks
        print(f"\n🧪 Adding {num_simulated_tasks} Simulated Tasks...")
        for i, prompt in enumerate(self.test_prompts[:num_simulated_tasks]):
            simulated_task = self.create_simulated_task(prompt)
            tasks_to_process.append(simulated_task)
            self.stats["simulated_tasks"] += 1
        
        # Process all tasks
        print(f"\n⛏️ Processing {len(tasks_to_process)} Tasks...")
        results = []
        
        for i, task_data in enumerate(tasks_to_process, 1):
            print(f"\n🎯 Task {i}/{len(tasks_to_process)}")
            result = await self.process_single_task(task_data)
            results.append(result)
            
            # Brief pause between tasks
            await asyncio.sleep(1)
        
        # Generate final report
        return self.generate_final_report(results)

    def generate_final_report(self, results: List[Dict]) -> Dict:
        """Generate comprehensive final report"""
        print("\n" + "="*80)
        print("📊 COMPLETE MINING PIPELINE REPORT")
        print("="*80)
        
        successful_results = [r for r in results if r.get("success", False)]
        
        # Calculate statistics
        if self.stats["generation_times"]:
            self.stats["avg_generation_time"] = sum(self.stats["generation_times"]) / len(self.stats["generation_times"])
        
        if self.stats["validation_times"]:
            self.stats["avg_validation_time"] = sum(self.stats["validation_times"]) / len(self.stats["validation_times"])
        
        if self.stats["validation_scores"]:
            self.stats["avg_validation_score"] = sum(self.stats["validation_scores"]) / len(self.stats["validation_scores"])
        
        # Print comprehensive stats
        print(f"\n📈 OVERALL STATISTICS:")
        print(f"Real Tasks Found:      {self.stats['real_tasks_pulled']}")
        print(f"Simulated Tasks:       {self.stats['simulated_tasks']}")
        print(f"Total Tasks:           {len(results)}")
        print(f"Successful Tasks:      {len(successful_results)}")
        print(f"Failed Tasks:          {len(results) - len(successful_results)}")
        
        if len(results) > 0:
            print(f"Success Rate:          {len(successful_results)/len(results)*100:.1f}%")
        else:
            print(f"Success Rate:          0.0%")
        
        print(f"\n⚡ PERFORMANCE METRICS:")
        print(f"Avg Generation Time:   {self.stats['avg_generation_time']:.2f}s")
        print(f"Avg Validation Time:   {self.stats['avg_validation_time']:.2f}s")
        print(f"Avg Validation Score:  {self.stats['avg_validation_score']:.4f}")
        print(f"Successful Generations: {self.stats['successful_generations']}")
        print(f"Successful Validations: {self.stats['successful_validations']}")
        print(f"Successful Submissions: {self.stats['successful_submissions']}")
        
        if self.stats["validation_scores"]:
            min_score = min(self.stats["validation_scores"])
            max_score = max(self.stats["validation_scores"])
            above_threshold = len([s for s in self.stats["validation_scores"] if s >= 0.6])
            print(f"Score Range:           {min_score:.4f} - {max_score:.4f}")
            print(f"Above Threshold (≥0.6): {above_threshold}/{len(self.stats['validation_scores'])}")
        
        print(f"\n📋 DETAILED RESULTS:")
        print("-" * 80)
        for i, result in enumerate(results, 1):
            if result.get("success"):
                status = "✅ SUCCESS" if result.get("submission_success") else "⚠️ LOW SCORE"
                print(f"{i:2d}. [{result['task_type']:9s}] '{result['prompt'][:25]:25s}' - "
                      f"Score: {result['validation_score']:.4f} - "
                      f"Time: {result['total_time']:.1f}s - {status}")
            else:
                prompt = result.get('prompt', 'Unknown')[:25]
                print(f"{i:2d}. [{result.get('task_type', 'UNKNOWN'):9s}] '{prompt:25s}' - "
                      f"❌ FAILED: {result.get('error', 'Unknown')}")
        
        if self.stats["errors"]:
            print(f"\n❌ ERRORS ({len(self.stats['errors'])}):")
            # Safely handle empty error list
            recent_errors = self.stats["errors"][-10:] if self.stats["errors"] else []
            for error in recent_errors:
                print(f"   - {error}")
        
        # Performance assessment
        print(f"\n🏆 PIPELINE ASSESSMENT:")
        if len(successful_results) == 0:
            print("   ❌ CRITICAL - No successful tasks completed!")
            print("   🔧 RECOMMENDATIONS:")
            print("      - Check generation server memory configuration")
            print("      - Verify service endpoints are accessible")
            print("      - Consider using lighter models or smaller batch sizes")
        elif self.stats["avg_validation_score"] >= 0.7:
            print("   ✅ EXCELLENT - High validation scores!")
        elif self.stats["avg_validation_score"] >= 0.6:
            print("   ✅ GOOD - Acceptable validation scores")
        elif self.stats["avg_validation_score"] >= 0.4:
            print("   ⚠️ MODERATE - Validation scores need improvement")
        else:
            print("   ❌ POOR - Low validation scores")
        
        if self.stats["avg_generation_time"] > 0:
            if self.stats["avg_generation_time"] <= 30:
                print("   ✅ FAST - Quick generation pipeline")
            elif self.stats["avg_generation_time"] <= 60:
                print("   ✅ ACCEPTABLE - Reasonable generation speed")
            else:
                print("   ⚠️ SLOW - Generation pipeline needs optimization")
        
        readiness = "🎯 READY FOR REAL MINING!" if (
            len(successful_results) > 0 and 
            self.stats["avg_validation_score"] >= 0.6 and
            len([e for e in self.stats["errors"] if "OOM" not in e]) == 0
        ) else "⚠️ NEEDS OPTIMIZATION"
        
        print(f"\n{readiness}")
        print("="*80)
        
        return {
            "success": len(successful_results) > 0,
            "total_tasks": len(results),
            "successful_tasks": len(successful_results),
            "real_tasks": self.stats["real_tasks_pulled"],
            "simulated_tasks": self.stats["simulated_tasks"],
            "statistics": self.stats,
            "results": results
        }

    def cleanup(self):
        """Cleanup processes and resources"""
        print("\n🛑 Cleaning up...")
        if self.validation_process:
            self.validation_process.terminate()
        if self.generation_process:
            self.generation_process.terminate()
        self.cleanup_processes()


async def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Complete Mining Pipeline")
    parser.add_argument("--wallet", type=str, default="manbeast", help="Wallet name")
    parser.add_argument("--hotkey", type=str, default="beastman", help="Hotkey name")
    parser.add_argument("--tasks", type=int, default=3, help="Number of simulated tasks")
    
    args = parser.parse_args()
    
    pipeline = CompleteMiningPipeline(args.wallet, args.hotkey)
    
    try:
        results = await pipeline.run_comprehensive_pipeline(args.tasks)
        
        if results.get("success"):
            print(f"\n🎉 Pipeline completed successfully!")
            return 0
        else:
            print(f"\n❌ Pipeline failed: {results.get('error', 'Unknown error')}")
            return 1
            
    except KeyboardInterrupt:
        print("\n🛑 Pipeline interrupted")
        return 130
    except Exception as e:
        print(f"\n💥 Pipeline error: {e}")
        traceback.print_exc()
        return 1
    finally:
        pipeline.cleanup()


if __name__ == "__main__":
    import sys
    exit_code = asyncio.run(main())
    sys.exit(exit_code) 