#!/usr/bin/env python3
"""
Robust Mining Test - Production Ready Pipeline
Handles memory issues and provides fallback mechanisms
"""

import asyncio
import time
import base64
import json
import subprocess
import requests
import os
import uuid
from typing import List, Dict, Optional, Tuple
import bittensor as bt
import traceback

# Import our protocol classes
from subnet_protocol_integration import (
    MockPullTask as PullTask,
    MockSubmitResults as SubmitResults,
    Task,
    Feedback,
    MINER_LICENSE_CONSENT_DECLARATION
)

class RobustMiningPipeline:
    """Production-ready mining pipeline with fallback mechanisms"""
    
    def __init__(self, wallet_name: str = "manbeast", hotkey_name: str = "beastman"):
        self.wallet_name = wallet_name
        self.hotkey_name = hotkey_name
        self.wallet = None
        self.dendrite = None
        self.metagraph = None
        self.subtensor = None
        
        # Service configuration
        self.generation_server_url = "http://localhost:8095"
        self.validation_server_url = "http://localhost:10006"
        
        # Performance tracking
        self.stats = {
            "tasks_processed": 0,
            "real_tasks_found": 0,
            "successful_generations": 0,
            "successful_validations": 0,
            "successful_submissions": 0,
            "memory_errors": 0,
            "connection_errors": 0,
            "validation_scores": [],
            "generation_times": [],
            "validation_times": [],
            "total_time": 0.0
        }
        
        # Test prompts
        self.test_prompts = [
            "a red apple on a table",
            "a blue wooden chair",
            "a golden coin",
            "a small wooden box",
            "a silver spoon"
        ]
        
        print("🚀 Robust Mining Pipeline Initialized")

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
            return False

    def check_services(self) -> Dict[str, bool]:
        """Check service availability"""
        print("🔍 Checking services...")
        
        services = {
            "generation": False,
            "validation": False
        }
        
        try:
            # Check generation server
            gen_resp = requests.get(f"{self.generation_server_url}/health/", timeout=5)
            services["generation"] = gen_resp.status_code == 200
            print(f"  Generation server: {'✅ Running' if services['generation'] else '❌ Down'}")
        except:
            print(f"  Generation server: ❌ Down")
        
        try:
            # Check validation server
            val_resp = requests.get(f"{self.validation_server_url}/version/", timeout=5)
            services["validation"] = val_resp.status_code == 200
            print(f"  Validation server: {'✅ Running' if services['validation'] else '❌ Down'}")
        except:
            print(f"  Validation server: ❌ Down")
        
        return services

    def start_validation_server(self) -> bool:
        """Start validation server"""
        try:
            print("📊 Starting validation server...")
            
            # Kill existing
            os.system("pkill -f 'serve.py' 2>/dev/null")
            time.sleep(2)
            
            # Start new
            subprocess.Popen(
                ["conda", "run", "-n", "three-gen-validation", 
                 "python", "serve.py", "--host", "0.0.0.0", "--port", "10006"],
                cwd="validation",
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )
            
            time.sleep(10)
            
            # Check if running
            try:
                resp = requests.get(f"{self.validation_server_url}/version/", timeout=5)
                if resp.status_code == 200:
                    print("  ✅ Validation server started")
                    return True
            except:
                pass
            
            print("  ❌ Validation server failed to start")
            return False
            
        except Exception as e:
            print(f"  ❌ Error starting validation server: {e}")
            return False

    def generate_mock_ply_data(self, prompt: str) -> str:
        """Generate mock PLY data for testing when generation server fails"""
        # Create a simple cube PLY for testing
        vertices = [
            "0.0 0.0 0.0",
            "1.0 0.0 0.0", 
            "1.0 1.0 0.0",
            "0.0 1.0 0.0",
            "0.0 0.0 1.0",
            "1.0 0.0 1.0",
            "1.0 1.0 1.0", 
            "0.0 1.0 1.0"
        ]
        
        faces = [
            "3 0 1 2",
            "3 0 2 3",
            "3 4 7 6",
            "3 4 6 5",
            "3 0 4 5",
            "3 0 5 1",
            "3 2 6 7",
            "3 2 7 3",
            "3 0 3 7",
            "3 0 7 4",
            "3 1 5 6",
            "3 1 6 2"
        ]
        
        ply_content = f"""ply
format ascii 1.0
comment Generated for prompt: {prompt}
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

    async def generate_3d_model(self, prompt: str, use_fallback: bool = True) -> Tuple[Optional[str], float, Dict]:
        """Generate 3D model with fallback to mock data"""
        start_time = time.time()
        
        try:
            print(f"🎨 Generating: '{prompt}'")
            
            response = requests.post(
                f"{self.generation_server_url}/generate/",
                data={
                    "prompt": prompt,
                    "seed": 42,
                    "use_bpt": False,
                    "return_compressed": False
                },
                timeout=120
            )
            
            generation_time = time.time() - start_time
            
            if response.status_code == 200:
                ply_data = response.content.decode('utf-8')
                headers = response.headers
                
                metadata = {
                    "source": "real_generation",
                    "generation_time": generation_time,
                    "face_count": int(headers.get("X-Face-Count", 0)),
                    "vertex_count": int(headers.get("X-Vertex-Count", 0)),
                    "ply_size": len(ply_data)
                }
                
                print(f"  ✅ Real generation in {generation_time:.2f}s")
                print(f"     Face count: {metadata['face_count']:,}")
                print(f"     Vertex count: {metadata['vertex_count']:,}")
                
                self.stats["successful_generations"] += 1
                self.stats["generation_times"].append(generation_time)
                
                return ply_data, generation_time, metadata
            
            else:
                print(f"  ⚠️ Generation server error: {response.status_code}")
                if "memory" in response.text.lower() or "cuda" in response.text.lower():
                    self.stats["memory_errors"] += 1
                
                if use_fallback:
                    print(f"  🔄 Using fallback mock generation")
                    ply_data = self.generate_mock_ply_data(prompt)
                    generation_time = time.time() - start_time
                    
                    metadata = {
                        "source": "mock_generation",
                        "generation_time": generation_time,
                        "face_count": 12,
                        "vertex_count": 8,
                        "ply_size": len(ply_data)
                    }
                    
                    print(f"  ✅ Mock generation in {generation_time:.2f}s")
                    self.stats["generation_times"].append(generation_time)
                    return ply_data, generation_time, metadata
                
                return None, generation_time, {}
        
        except requests.exceptions.ConnectionError:
            print(f"  📡 Connection error")
            self.stats["connection_errors"] += 1
            
            if use_fallback:
                print(f"  🔄 Using fallback mock generation")
                ply_data = self.generate_mock_ply_data(prompt)
                generation_time = time.time() - start_time
                
                metadata = {
                    "source": "mock_generation_fallback",
                    "generation_time": generation_time,
                    "face_count": 12,
                    "vertex_count": 8,
                    "ply_size": len(ply_data)
                }
                
                print(f"  ✅ Mock fallback in {generation_time:.2f}s")
                return ply_data, generation_time, metadata
            
            return None, time.time() - start_time, {}
        
        except Exception as e:
            print(f"  ❌ Generation exception: {e}")
            
            if use_fallback:
                print(f"  🔄 Using fallback mock generation")
                ply_data = self.generate_mock_ply_data(prompt)
                generation_time = time.time() - start_time
                
                metadata = {
                    "source": "mock_generation_exception",
                    "generation_time": generation_time,
                    "face_count": 12,
                    "vertex_count": 8,
                    "ply_size": len(ply_data)
                }
                
                return ply_data, generation_time, metadata
            
            return None, time.time() - start_time, {}

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
                    "validation_time": validation_time
                }
                
                print(f"  ✅ Validated in {validation_time:.2f}s - Score: {score:.4f}")
                
                self.stats["successful_validations"] += 1
                self.stats["validation_scores"].append(score)
                self.stats["validation_times"].append(validation_time)
                
                return score, validation_time, validation_data
            else:
                print(f"  ❌ Validation error: {response.status_code}")
                return 0.0, validation_time, {}
                
        except Exception as e:
            print(f"  ❌ Validation exception: {e}")
            return 0.0, 0.0, {}

    async def check_for_real_tasks(self) -> Optional[Dict]:
        """Check validators for real tasks"""
        if not self.metagraph:
            return None
        
        print("🎯 Checking for real tasks...")
        
        # Find active validators
        active_validators = []
        for uid, (stake, axon, hotkey) in enumerate(zip(
            self.metagraph.stake, self.metagraph.axons, self.metagraph.hotkeys
        )):
            if stake > 100 and axon.is_serving and axon.ip != '0.0.0.0':
                active_validators.append((uid, stake, axon, hotkey))
        
        active_validators.sort(key=lambda x: x[1], reverse=True)
        
        # Try top 5 validators quickly
        for uid, stake, axon, hotkey in active_validators[:5]:
            try:
                pull_synapse = PullTask()
                
                response = await self.dendrite.call(
                    target_axon=axon,
                    synapse=pull_synapse,
                    deserialize=False,
                    timeout=10.0
                )
                
                if response and hasattr(response, 'task') and response.task:
                    print(f"🎉 REAL TASK FOUND from validator {uid}!")
                    print(f"   Task ID: {response.task.id}")
                    print(f"   Prompt: '{response.task.prompt}'")
                    
                    self.stats["real_tasks_found"] += 1
                    
                    return {
                        "task_id": response.task.id,
                        "prompt": response.task.prompt,
                        "task": response.task,
                        "validator_uid": uid,
                        "validator_hotkey": hotkey,
                        "axon": axon,
                        "validation_threshold": getattr(response, 'validation_threshold', 0.6),
                        "is_real": True
                    }
                    
            except Exception:
                continue
        
        print("  ❌ No real tasks available")
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
            "validator_uid": -1,
            "validator_hotkey": "simulated",
            "axon": None,
            "validation_threshold": 0.6,
            "is_real": False
        }

    async def process_task(self, task_data: Dict) -> Dict:
        """Process a single task"""
        task_type = "REAL" if task_data["is_real"] else "SIMULATED"
        
        print(f"\n{'='*60}")
        print(f"🎯 Processing {task_type} Task: '{task_data['prompt']}'")
        print(f"{'='*60}")
        
        start_time = time.time()
        
        # Generate 3D model
        ply_data, gen_time, gen_metadata = await self.generate_3d_model(task_data["prompt"])
        
        if not ply_data:
            return {
                "success": False,
                "error": "Generation failed",
                "task_type": task_type,
                "prompt": task_data["prompt"]
            }
        
        # Validate locally
        val_score, val_time, val_data = await self.validate_locally(task_data["prompt"], ply_data)
        
        # Check if would be accepted
        threshold = task_data["validation_threshold"]
        would_be_accepted = val_score >= threshold
        
        if would_be_accepted:
            self.stats["successful_submissions"] += 1
        
        total_time = time.time() - start_time
        
        result = {
            "success": True,
            "task_type": task_type,
            "prompt": task_data["prompt"],
            "generation_time": gen_time,
            "validation_time": val_time,
            "total_time": total_time,
            "validation_score": val_score,
            "validation_threshold": threshold,
            "would_be_accepted": would_be_accepted,
            "generation_metadata": gen_metadata,
            "validation_data": val_data
        }
        
        status = "✅ ACCEPTED" if would_be_accepted else "❌ REJECTED"
        print(f"\n📋 Task Result:")
        print(f"   Generation: {gen_time:.2f}s ({gen_metadata.get('source', 'unknown')})")
        print(f"   Validation: {val_time:.2f}s")
        print(f"   Score: {val_score:.4f} (threshold: {threshold:.2f})")
        print(f"   Status: {status}")
        
        return result

    async def run_comprehensive_test(self, num_simulated_tasks: int = 3) -> Dict:
        """Run comprehensive mining test"""
        print("🚀 ROBUST MINING PIPELINE TEST")
        print("="*70)
        
        test_start = time.time()
        
        # Initialize Bittensor
        if not await self.initialize_bittensor():
            return {"success": False, "error": "Bittensor initialization failed"}
        
        # Check and start validation server
        services = self.check_services()
        if not services["validation"]:
            if not self.start_validation_server():
                return {"success": False, "error": "Validation server failed to start"}
        
        # Try to get real task
        real_task = await self.check_for_real_tasks()
        
        # Prepare task list
        tasks = []
        if real_task:
            tasks.append(real_task)
        
        # Add simulated tasks
        for prompt in self.test_prompts[:num_simulated_tasks]:
            tasks.append(self.create_simulated_task(prompt))
        
        print(f"\n⛏️ Processing {len(tasks)} tasks...")
        
        # Process all tasks
        results = []
        for i, task_data in enumerate(tasks, 1):
            print(f"\n🎯 Task {i}/{len(tasks)}")
            result = await self.process_task(task_data)
            results.append(result)
            self.stats["tasks_processed"] += 1
            
            # Brief pause
            await asyncio.sleep(0.5)
        
        # Generate final report
        self.stats["total_time"] = time.time() - test_start
        return self.generate_report(results)

    def generate_report(self, results: List[Dict]) -> Dict:
        """Generate comprehensive test report"""
        print("\n" + "="*80)
        print("📊 ROBUST MINING PIPELINE REPORT")
        print("="*80)
        
        successful_results = [r for r in results if r.get("success", False)]
        accepted_results = [r for r in successful_results if r.get("would_be_accepted", False)]
        
        # Calculate averages
        avg_gen_time = sum(self.stats["generation_times"]) / len(self.stats["generation_times"]) if self.stats["generation_times"] else 0
        avg_val_time = sum(self.stats["validation_times"]) / len(self.stats["validation_times"]) if self.stats["validation_times"] else 0
        avg_score = sum(self.stats["validation_scores"]) / len(self.stats["validation_scores"]) if self.stats["validation_scores"] else 0
        
        print(f"\n📈 OVERALL STATISTICS:")
        print(f"Total Time:            {self.stats['total_time']:.1f}s")
        print(f"Tasks Processed:       {self.stats['tasks_processed']}")
        print(f"Real Tasks Found:      {self.stats['real_tasks_found']}")
        print(f"Successful Tasks:      {len(successful_results)}")
        print(f"Would Be Accepted:     {len(accepted_results)}")
        print(f"Success Rate:          {len(successful_results)/len(results)*100:.1f}%")
        print(f"Acceptance Rate:       {len(accepted_results)/len(results)*100:.1f}%")
        
        print(f"\n⚡ PERFORMANCE METRICS:")
        print(f"Avg Generation Time:   {avg_gen_time:.2f}s")
        print(f"Avg Validation Time:   {avg_val_time:.2f}s")
        print(f"Avg Validation Score:  {avg_score:.4f}")
        print(f"Memory Errors:         {self.stats['memory_errors']}")
        print(f"Connection Errors:     {self.stats['connection_errors']}")
        
        if self.stats["validation_scores"]:
            min_score = min(self.stats["validation_scores"])
            max_score = max(self.stats["validation_scores"])
            above_threshold = len([s for s in self.stats["validation_scores"] if s >= 0.6])
            print(f"Score Range:           {min_score:.4f} - {max_score:.4f}")
            print(f"Above Threshold:       {above_threshold}/{len(self.stats['validation_scores'])}")
        
        print(f"\n📋 DETAILED RESULTS:")
        print("-" * 80)
        for i, result in enumerate(results, 1):
            if result.get("success"):
                status = "✅ ACCEPT" if result.get("would_be_accepted") else "❌ REJECT"
                gen_source = result.get("generation_metadata", {}).get("source", "unknown")
                print(f"{i:2d}. [{result['task_type']:9s}] '{result['prompt'][:30]:30s}' - "
                      f"Score: {result['validation_score']:.4f} - "
                      f"Time: {result['total_time']:.1f}s - "
                      f"Gen: {gen_source[:4]:4s} - {status}")
            else:
                print(f"{i:2d}. [{'FAILED':9s}] '{result.get('prompt', 'Unknown')[:30]:30s}' - "
                      f"❌ {result.get('error', 'Unknown error')}")
        
        # Assessment
        print(f"\n🏆 PIPELINE ASSESSMENT:")
        
        if len(successful_results) == 0:
            print("   ❌ CRITICAL - No tasks completed successfully")
            readiness = "❌ NOT READY"
        elif len(accepted_results) == 0:
            print("   ⚠️ WARNING - No tasks meet validation threshold")
            readiness = "⚠️ NEEDS TUNING"
        elif avg_score >= 0.7:
            print("   ✅ EXCELLENT - High performance pipeline")
            readiness = "🎯 PRODUCTION READY"
        elif avg_score >= 0.6:
            print("   ✅ GOOD - Acceptable performance")
            readiness = "✅ READY FOR MINING"
        else:
            print("   ⚠️ MODERATE - Performance needs improvement")
            readiness = "⚠️ NEEDS OPTIMIZATION"
        
        print(f"\n{readiness}")
        
        if self.stats["memory_errors"] > 0:
            print(f"⚠️ Memory issues detected - consider using lighter models")
        
        if self.stats["connection_errors"] > 0:
            print(f"⚠️ Connection issues detected - verify service endpoints")
        
        print("="*80)
        
        return {
            "success": len(successful_results) > 0,
            "total_tasks": len(results),
            "successful_tasks": len(successful_results),
            "accepted_tasks": len(accepted_results),
            "real_tasks_found": self.stats["real_tasks_found"],
            "avg_generation_time": avg_gen_time,
            "avg_validation_time": avg_val_time,
            "avg_validation_score": avg_score,
            "memory_errors": self.stats["memory_errors"],
            "connection_errors": self.stats["connection_errors"],
            "readiness": readiness,
            "results": results
        }


async def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Robust Mining Pipeline Test")
    parser.add_argument("--wallet", type=str, default="manbeast", help="Wallet name")
    parser.add_argument("--hotkey", type=str, default="beastman", help="Hotkey name")
    parser.add_argument("--tasks", type=int, default=3, help="Number of simulated tasks")
    
    args = parser.parse_args()
    
    pipeline = RobustMiningPipeline(args.wallet, args.hotkey)
    
    try:
        results = await pipeline.run_comprehensive_test(args.tasks)
        
        if results.get("success"):
            print(f"\n🎉 Test completed successfully!")
            print(f"🎯 {results['accepted_tasks']}/{results['total_tasks']} tasks would be accepted")
            return 0
        else:
            print(f"\n❌ Test failed: {results.get('error', 'Unknown error')}")
            return 1
            
    except KeyboardInterrupt:
        print("\n🛑 Test interrupted")
        return 130
    except Exception as e:
        print(f"\n💥 Test error: {e}")
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    import sys
    exit_code = asyncio.run(main())
    sys.exit(exit_code) 