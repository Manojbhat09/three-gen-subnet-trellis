#!/usr/bin/env python3
"""
Real Mining Test - Integration Test with Actual Validators
Purpose: Pull real tasks, generate models, validate locally, measure performance
"""

import asyncio
import time
import base64
import json
import subprocess
import requests
import os
from typing import List, Dict, Optional, Tuple
import bittensor as bt
import traceback
from pydantic import BaseModel, Field
import uuid

# Import our protocol classes (would be from neurons.common.protocol in production)
from subnet_protocol_integration import (
    MockPullTask as PullTask,
    MockSubmitResults as SubmitResults,
    Task,
    Feedback,
    MINER_LICENSE_CONSENT_DECLARATION
)

class RealMiningTester:
    """Comprehensive mining test with real components"""
    
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
        
        # Performance tracking
        self.performance_stats = {
            "tasks_pulled": 0,
            "tasks_generated": 0,
            "tasks_validated": 0,
            "tasks_submitted": 0,
            "successful_submissions": 0,
            "total_generation_time": 0.0,
            "total_validation_time": 0.0,
            "avg_validation_score": 0.0,
            "validation_scores": []
        }
        
        print("🧪 Real Mining Test Setup")
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
            print(f"  ❌ Initialization failed: {e}")
            return False

    def check_service_status(self, url: str, service_name: str) -> bool:
        """Check if a service is running"""
        try:
            # Use different endpoints for different services
            if "10006" in url:  # Validation server
                endpoint = "/version/"
            else:  # Generation server
                endpoint = "/health/"
                
            response = requests.get(f"{url}{endpoint}", timeout=5)
            if response.status_code == 200:
                print(f"  ✅ {service_name} is running at {url}")
                return True
            else:
                print(f"  ❌ {service_name} returned status {response.status_code}")
                return False
        except requests.exceptions.RequestException as e:
            print(f"  ❌ {service_name} not available at {url}: {e}")
            return False

    def start_validation_server(self) -> subprocess.Popen:
        """Start the validation server in three-gen-validation environment"""
        print("🚀 Starting validation server...")
        try:
            # Change to validation directory and start server
            cmd = [
                "conda", "run", "-n", "three-gen-validation",
                "python", "serve.py", "--host", "0.0.0.0", "--port", "10006"
            ]
            
            process = subprocess.Popen(
                cmd,
                cwd="validation",
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True
            )
            
            # Wait a bit for server to start
            time.sleep(10)
            
            if self.check_service_status(self.validation_server_url, "Validation server"):
                return process
            else:
                process.terminate()
                return None
                
        except Exception as e:
            print(f"  ❌ Failed to start validation server: {e}")
            return None

    def start_generation_server(self) -> subprocess.Popen:
        """Start the generation server in hunyuan3d environment"""
        print("🚀 Starting generation server...")
        try:
            cmd = [
                "conda", "run", "-n", "hunyuan3d",
                "python", "flux_hunyuan_bpt_generation_server.py"
            ]
            
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True
            )
            
            # Wait a bit for server to start
            time.sleep(15)
            
            if self.check_service_status(self.generation_server_url, "Generation server"):
                return process
            else:
                process.terminate()
                return None
                
        except Exception as e:
            print(f"  ❌ Failed to start generation server: {e}")
            return None

    async def pull_task_from_validator(self, validator_uid: int) -> Optional[Dict]:
        """Pull a real task from a validator"""
        try:
            if validator_uid >= len(self.metagraph.axons):
                return None
            
            axon = self.metagraph.axons[validator_uid]
            validator_hotkey = self.metagraph.hotkeys[validator_uid]
            
            print(f"📡 Pulling task from validator {validator_uid}")
            print(f"   Hotkey: {validator_hotkey[:10]}...")
            print(f"   Axon: {axon.ip}:{axon.port}")
            
            # Create PullTask synapse
            pull_synapse = PullTask()
            
            # Make the dendrite call
            response = await self.dendrite.call(
                target_axon=axon,
                synapse=pull_synapse,
                deserialize=False,
                timeout=30.0
            )
            
            if response and hasattr(response, 'task') and response.task:
                print(f"  ✅ Task received: {response.task.id}")
                print(f"     Prompt: '{response.task.prompt}'")
                print(f"     Validation threshold: {response.validation_threshold}")
                
                self.performance_stats["tasks_pulled"] += 1
                
                return {
                    "task_id": response.task.id,
                    "prompt": response.task.prompt,
                    "task": response.task,
                    "validation_threshold": response.validation_threshold,
                    "cooldown_until": response.cooldown_until,
                    "validator_hotkey": validator_hotkey,
                    "validator_uid": validator_uid,
                    "axon": axon
                }
            else:
                print(f"  ❌ No task available from validator {validator_uid}")
                return None
                
        except Exception as e:
            print(f"  ❌ Failed to pull from validator {validator_uid}: {e}")
            return None

    async def generate_3d_model(self, prompt: str) -> Tuple[Optional[str], float]:
        """Generate 3D model using the generation server"""
        try:
            print(f"🎨 Generating 3D model for: '{prompt}'")
            
            start_time = time.time()
            
            # Use form data instead of JSON for the generation server
            response = requests.post(
                f"{self.generation_server_url}/generate/",
                data={
                    "prompt": prompt,
                    "seed": 42,
                    "use_bpt": True
                },
                timeout=300  # 5 minutes timeout
            )
            
            generation_time = time.time() - start_time
            self.performance_stats["total_generation_time"] += generation_time
            
            if response.status_code == 200:
                result = response.json()
                if result.get("success"):
                    print(f"  ✅ Generation completed in {generation_time:.2f}s")
                    print(f"     PLY size: {len(result['ply_data'])} chars")
                    
                    self.performance_stats["tasks_generated"] += 1
                    return result["ply_data"], generation_time
                else:
                    print(f"  ❌ Generation failed: {result.get('error', 'Unknown error')}")
                    return None, generation_time
            else:
                print(f"  ❌ Generation server error: {response.status_code}")
                print(f"     Response: {response.text[:200]}...")
                return None, generation_time
                
        except Exception as e:
            print(f"  ❌ Generation error: {e}")
            return None, 0.0

    async def validate_locally(self, prompt: str, ply_data: str) -> Tuple[float, float]:
        """Validate the generated model locally"""
        try:
            print(f"🔍 Validating locally...")
            
            start_time = time.time()
            
            response = requests.post(
                f"{self.validation_server_url}/validate_txt_to_3d_ply/",
                json={
                    "prompt": prompt,
                    "data": ply_data,
                    "compression": 0,  # No compression for testing
                    "generate_preview": False
                },
                timeout=60  # 1 minute timeout
            )
            
            validation_time = time.time() - start_time
            self.performance_stats["total_validation_time"] += validation_time
            
            if response.status_code == 200:
                result = response.json()
                score = result.get("score", 0.0)
                
                print(f"  ✅ Validation completed in {validation_time:.2f}s")
                print(f"     Score: {score:.4f}")
                print(f"     IQA: {result.get('iqa', 0.0):.4f}")
                print(f"     Alignment: {result.get('alignment_score', 0.0):.4f}")
                
                self.performance_stats["tasks_validated"] += 1
                self.performance_stats["validation_scores"].append(score)
                
                return score, validation_time
            else:
                print(f"  ❌ Validation server error: {response.status_code}")
                return 0.0, validation_time
                
        except Exception as e:
            print(f"  ❌ Validation error: {e}")
            return 0.0, 0.0

    async def submit_results(self, task_data: Dict, ply_data: str, 
                           validation_score: float) -> bool:
        """Submit results to the validator"""
        try:
            print(f"📤 Submitting results to validator {task_data['validator_uid']}")
            
            # Create submission timestamp and signature
            submit_time = time.time_ns()
            
            # Create signature
            message = (
                f"{MINER_LICENSE_CONSENT_DECLARATION}"
                f"{submit_time}{task_data['prompt']}"
                f"{task_data['validator_hotkey']}{self.wallet.hotkey.ss58_address}"
            )
            signature = base64.b64encode(self.wallet.hotkey.sign(message)).decode()
            
            # Create SubmitResults synapse
            submit_synapse = SubmitResults()
            submit_synapse.task = task_data["task"]
            submit_synapse.results = ply_data
            submit_synapse.compression = 0  # No compression for testing
            submit_synapse.data_format = "ply"
            submit_synapse.data_ver = 0
            submit_synapse.submit_time = submit_time
            submit_synapse.signature = signature
            
            # Make the dendrite call
            response = await self.dendrite.call(
                target_axon=task_data["axon"],
                synapse=submit_synapse,
                deserialize=False,
                timeout=300.0
            )
            
            self.performance_stats["tasks_submitted"] += 1
            
            if response and hasattr(response, 'feedback') and response.feedback:
                if not response.feedback.validation_failed:
                    print(f"  ✅ Submission successful!")
                    print(f"     Task fidelity score: {response.feedback.task_fidelity_score:.4f}")
                    print(f"     Average fidelity score: {response.feedback.average_fidelity_score:.4f}")
                    print(f"     Current miner reward: {response.feedback.current_miner_reward:.4f}")
                    
                    self.performance_stats["successful_submissions"] += 1
                    return True
                else:
                    print(f"  ❌ Submission failed validation")
                    print(f"     Task fidelity score: {response.feedback.task_fidelity_score:.4f}")
                    return False
            else:
                print(f"  ❌ No feedback received from validator")
                return False
                
        except Exception as e:
            print(f"  ❌ Submission error: {e}")
            return False

    def get_active_validators(self, limit: int = 5) -> List[int]:
        """Get active validators"""
        if not self.metagraph:
            return []
        
        active_validators = []
        for uid, (stake, axon, hotkey) in enumerate(zip(
            self.metagraph.stake, 
            self.metagraph.axons, 
            self.metagraph.hotkeys
        )):
            if (stake > 1000 and  # Minimum stake
                axon.is_serving and 
                axon.ip != "0.0.0.0" and
                len(active_validators) < limit):
                active_validators.append(uid)
                print(f"  📡 Validator {uid}: {hotkey[:10]}... (stake: {stake:.1f})")
        
        print(f"Found {len(active_validators)} active validators")
        return active_validators

    def print_performance_summary(self):
        """Print performance statistics"""
        stats = self.performance_stats
        
        print("\n" + "="*60)
        print("📊 PERFORMANCE SUMMARY")
        print("="*60)
        
        print(f"Tasks Pulled:        {stats['tasks_pulled']}")
        print(f"Tasks Generated:     {stats['tasks_generated']}")
        print(f"Tasks Validated:     {stats['tasks_validated']}")
        print(f"Tasks Submitted:     {stats['tasks_submitted']}")
        print(f"Successful Subs:     {stats['successful_submissions']}")
        
        if stats['tasks_generated'] > 0:
            avg_gen_time = stats['total_generation_time'] / stats['tasks_generated']
            print(f"Avg Generation Time: {avg_gen_time:.2f}s")
        
        if stats['tasks_validated'] > 0:
            avg_val_time = stats['total_validation_time'] / stats['tasks_validated']
            print(f"Avg Validation Time: {avg_val_time:.2f}s")
        
        if stats['validation_scores']:
            avg_score = sum(stats['validation_scores']) / len(stats['validation_scores'])
            min_score = min(stats['validation_scores'])
            max_score = max(stats['validation_scores'])
            print(f"Validation Scores:   avg={avg_score:.4f}, min={min_score:.4f}, max={max_score:.4f}")
        
        success_rate = (stats['successful_submissions'] / stats['tasks_submitted'] 
                       if stats['tasks_submitted'] > 0 else 0)
        print(f"Success Rate:        {success_rate*100:.1f}%")
        
        print("="*60)

    async def run_comprehensive_test(self, num_tasks: int = 3) -> Dict:
        """Run comprehensive mining test"""
        print("🧪 COMPREHENSIVE MINING TEST")
        print("="*50)
        
        # Initialize Bittensor
        if not await self.initialize_bittensor():
            return {"error": "Bittensor initialization failed"}
        
        # Check and start services
        print("\n🔧 Checking Services...")
        
        validation_server_proc = None
        generation_server_proc = None
        
        try:
            # Check validation server
            if not self.check_service_status(self.validation_server_url, "Validation server"):
                validation_server_proc = self.start_validation_server()
                if not validation_server_proc:
                    return {"error": "Failed to start validation server"}
            
            # Check generation server
            if not self.check_service_status(self.generation_server_url, "Generation server"):
                generation_server_proc = self.start_generation_server()
                if not generation_server_proc:
                    return {"error": "Failed to start generation server"}
            
            # Get active validators
            print("\n📡 Finding Active Validators...")
            validators = self.get_active_validators(limit=5)
            if not validators:
                return {"error": "No active validators found"}
            
            # Process tasks
            print(f"\n⛏️ Processing {num_tasks} Mining Tasks...")
            tasks_processed = 0
            
            for i in range(num_tasks):
                if tasks_processed >= num_tasks:
                    break
                    
                print(f"\n{'='*30} Task {i+1}/{num_tasks} {'='*30}")
                
                # Try to pull task from validators
                task_data = None
                for validator_uid in validators:
                    task_data = await self.pull_task_from_validator(validator_uid)
                    if task_data:
                        break
                
                if not task_data:
                    print("❌ No tasks available from validators")
                    continue
                
                # Generate 3D model
                ply_data, gen_time = await self.generate_3d_model(task_data["prompt"])
                if not ply_data:
                    print("❌ Generation failed")
                    continue
                
                # Validate locally
                validation_score, val_time = await self.validate_locally(
                    task_data["prompt"], ply_data
                )
                
                print(f"📋 Task Summary:")
                print(f"   Generation: {gen_time:.2f}s")
                print(f"   Validation: {val_time:.2f}s")
                print(f"   Score: {validation_score:.4f}")
                print(f"   Threshold: {task_data['validation_threshold']:.4f}")
                
                # Submit if score meets threshold
                if validation_score >= task_data["validation_threshold"]:
                    success = await self.submit_results(task_data, ply_data, validation_score)
                    if success:
                        print("🎉 Task completed successfully!")
                    else:
                        print("⚠️ Submission failed")
                else:
                    print(f"⚠️ Score {validation_score:.4f} below threshold {task_data['validation_threshold']:.4f}")
                    print("💡 Submitting anyway for testing...")
                    await self.submit_results(task_data, ply_data, validation_score)
                
                tasks_processed += 1
                
                # Small delay between tasks
                await asyncio.sleep(2)
            
            # Print final results
            self.print_performance_summary()
            
            return {
                "success": True,
                "tasks_processed": tasks_processed,
                "performance": self.performance_stats
            }
            
        finally:
            # Cleanup processes
            if validation_server_proc:
                print("\n🛑 Stopping validation server...")
                validation_server_proc.terminate()
            
            if generation_server_proc:
                print("🛑 Stopping generation server...")
                generation_server_proc.terminate()


async def main():
    """Main test function"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Real Mining Integration Test")
    parser.add_argument("--wallet", type=str, default="manbeast", help="Wallet name")
    parser.add_argument("--hotkey", type=str, default="beastman", help="Hotkey name")
    parser.add_argument("--tasks", type=int, default=3, help="Number of tasks to process")
    
    args = parser.parse_args()
    
    tester = RealMiningTester(args.wallet, args.hotkey)
    
    try:
        results = await tester.run_comprehensive_test(args.tasks)
        
        if results.get("success"):
            print("\n🎉 Test completed successfully!")
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