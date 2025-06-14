#!/usr/bin/env python3
"""
Production Mining Monitor - Continuous Task Monitoring
Monitors validators for real tasks and processes them immediately
"""

import asyncio
import time
import json
import subprocess
import requests
import os
import signal
from datetime import datetime
from typing import Optional, Dict, List
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

class ProductionMiningMonitor:
    """Production mining monitor for continuous operation"""
    
    def __init__(self, wallet_name: str = "manbeast", hotkey_name: str = "beastman"):
        self.wallet_name = wallet_name
        self.hotkey_name = hotkey_name
        self.wallet = None
        self.dendrite = None
        self.metagraph = None
        self.subtensor = None
        
        self.generation_server_url = "http://localhost:8095"
        self.validation_server_url = "http://localhost:10006"
        
        # Monitoring configuration
        self.check_interval = 30  # seconds between validator checks
        self.max_daily_tasks = 50  # maximum tasks per day
        self.cooldown_period = 300  # 5 minutes between tasks from same validator
        
        # Runtime statistics
        self.stats = {
            "start_time": time.time(),
            "total_checks": 0,
            "real_tasks_found": 0,
            "successful_submissions": 0,
            "failed_submissions": 0,
            "total_earned": 0.0,
            "validator_cooldowns": {},
            "last_task_time": 0,
            "daily_task_count": 0,
            "last_reset_date": datetime.now().date()
        }
        
        # Service processes
        self.validation_process = None
        self.generation_process = None
        
        # Shutdown flag
        self.shutdown_requested = False
        
        print("🚀 Production Mining Monitor Initialized")
        print(f"💰 Wallet: {wallet_name}")
        print(f"🔑 Hotkey: {hotkey_name}")
        print(f"⏱️ Check interval: {self.check_interval}s")

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

    def ensure_services_running(self) -> bool:
        """Ensure both services are running"""
        try:
            # Check validation server
            try:
                resp = requests.get(f"{self.validation_server_url}/version/", timeout=5)
                validation_running = resp.status_code == 200
            except:
                validation_running = False
            
            # Check generation server
            try:
                resp = requests.get(f"{self.generation_server_url}/health/", timeout=5)
                generation_running = resp.status_code == 200
            except:
                generation_running = False
            
            # Start validation server if needed
            if not validation_running:
                print("📊 Starting validation server...")
                os.system("pkill -f 'serve.py' 2>/dev/null")
                time.sleep(2)
                
                self.validation_process = subprocess.Popen(
                    ["conda", "run", "-n", "three-gen-validation", 
                     "python", "serve.py", "--host", "0.0.0.0", "--port", "10006"],
                    cwd="validation",
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE
                )
                time.sleep(10)
                
                # Verify it started
                try:
                    resp = requests.get(f"{self.validation_server_url}/version/", timeout=5)
                    validation_running = resp.status_code == 200
                except:
                    validation_running = False
            
            # Start generation server if needed (with memory optimization)
            if not generation_running:
                print("🎨 Starting generation server...")
                os.system("pkill -f 'flux_hunyuan_bpt_generation_server' 2>/dev/null")
                time.sleep(3)
                
                env = os.environ.copy()
                env["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512,garbage_collection_threshold:0.6"
                
                self.generation_process = subprocess.Popen(
                    ["conda", "run", "-n", "hunyuan3d",
                     "python", "flux_hunyuan_bpt_generation_server.py"],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    env=env
                )
                time.sleep(20)
                
                # Verify it started
                try:
                    resp = requests.get(f"{self.generation_server_url}/health/", timeout=5)
                    generation_running = resp.status_code == 200
                except:
                    generation_running = False
            
            services_status = f"Val: {'✅' if validation_running else '❌'} Gen: {'✅' if generation_running else '❌'}"
            print(f"🔧 Services status: {services_status}")
            
            return validation_running  # At minimum we need validation
            
        except Exception as e:
            print(f"❌ Error ensuring services: {e}")
            return False

    async def find_available_validators(self) -> List[tuple]:
        """Find validators that might have tasks available"""
        if not self.metagraph:
            return []
        
        current_time = time.time()
        active_validators = []
        
        for uid, (stake, axon, hotkey) in enumerate(zip(
            self.metagraph.stake, self.metagraph.axons, self.metagraph.hotkeys
        )):
            # Check if validator is active and not on cooldown
            if (stake > 100 and axon.is_serving and axon.ip != '0.0.0.0'):
                last_attempt = self.stats["validator_cooldowns"].get(uid, 0)
                if current_time - last_attempt > self.cooldown_period:
                    active_validators.append((uid, stake, axon, hotkey))
        
        # Sort by stake (highest first)
        active_validators.sort(key=lambda x: x[1], reverse=True)
        return active_validators

    async def check_for_real_tasks(self) -> Optional[Dict]:
        """Check validators for available tasks"""
        self.stats["total_checks"] += 1
        
        # Reset daily counter if new day
        today = datetime.now().date()
        if today != self.stats["last_reset_date"]:
            self.stats["daily_task_count"] = 0
            self.stats["last_reset_date"] = today
            print(f"📅 Daily task counter reset")
        
        # Check daily limit
        if self.stats["daily_task_count"] >= self.max_daily_tasks:
            print(f"⏸️ Daily task limit reached ({self.max_daily_tasks})")
            return None
        
        validators = await self.find_available_validators()
        if not validators:
            return None
        
        current_time = time.time()
        
        # Try up to 10 validators per check
        for uid, stake, axon, hotkey in validators[:10]:
            try:
                pull_synapse = PullTask()
                
                response = await self.dendrite.call(
                    target_axon=axon,
                    synapse=pull_synapse,
                    deserialize=False,
                    timeout=15.0
                )
                
                # Mark validator as attempted
                self.stats["validator_cooldowns"][uid] = current_time
                
                if response and hasattr(response, 'task') and response.task:
                    print(f"🎉 REAL TASK FOUND!")
                    print(f"   Validator: {uid} (stake: {stake:.0f})")
                    print(f"   Task ID: {response.task.id}")
                    print(f"   Prompt: '{response.task.prompt}'")
                    print(f"   Threshold: {getattr(response, 'validation_threshold', 0.6)}")
                    
                    self.stats["real_tasks_found"] += 1
                    self.stats["daily_task_count"] += 1
                    
                    return {
                        "task_id": response.task.id,
                        "prompt": response.task.prompt,
                        "task": response.task,
                        "validator_uid": uid,
                        "validator_hotkey": hotkey,
                        "validator_stake": stake,
                        "axon": axon,
                        "validation_threshold": getattr(response, 'validation_threshold', 0.6),
                        "found_time": current_time
                    }
                
            except Exception as e:
                continue
        
        return None

    async def process_real_task(self, task_data: Dict) -> bool:
        """Process a real task from validator"""
        print(f"\n{'='*70}")
        print(f"⛏️ PROCESSING REAL TASK")
        print(f"{'='*70}")
        print(f"📝 Prompt: {task_data['prompt']}")
        print(f"🎯 Validator: {task_data['validator_uid']} (stake: {task_data['validator_stake']:.0f})")
        print(f"📊 Threshold: {task_data['validation_threshold']:.2f}")
        
        start_time = time.time()
        
        try:
            # Generate 3D model
            print(f"\n🎨 Generating 3D model...")
            gen_start = time.time()
            
            response = requests.post(
                f"{self.generation_server_url}/generate/",
                data={
                    "prompt": task_data["prompt"],
                    "seed": 42,
                    "use_bpt": False,
                    "return_compressed": False
                },
                timeout=300  # 5 minutes
            )
            
            gen_time = time.time() - gen_start
            
            if response.status_code != 200:
                print(f"❌ Generation failed: {response.status_code}")
                self.stats["failed_submissions"] += 1
                return False
            
            ply_data = response.content.decode('utf-8')
            headers = response.headers
            
            face_count = int(headers.get("X-Face-Count", 0))
            vertex_count = int(headers.get("X-Vertex-Count", 0))
            
            print(f"✅ Generated in {gen_time:.1f}s")
            print(f"   Faces: {face_count:,}, Vertices: {vertex_count:,}")
            print(f"   PLY size: {len(ply_data):,} bytes")
            
            # Validate locally
            print(f"\n🔍 Validating locally...")
            val_start = time.time()
            
            val_response = requests.post(
                f"{self.validation_server_url}/validate_txt_to_3d_ply/",
                json={
                    "prompt": task_data["prompt"],
                    "data": ply_data,
                    "compression": 0,
                    "generate_preview": False
                },
                timeout=120
            )
            
            val_time = time.time() - val_start
            
            if val_response.status_code != 200:
                print(f"❌ Validation failed: {val_response.status_code}")
                self.stats["failed_submissions"] += 1
                return False
            
            val_result = val_response.json()
            score = val_result.get("score", 0.0)
            
            print(f"✅ Validated in {val_time:.1f}s")
            print(f"   Score: {score:.4f}")
            print(f"   IQA: {val_result.get('iqa', 0.0):.4f}")
            print(f"   Alignment: {val_result.get('alignment_score', 0.0):.4f}")
            
            # Check if meets threshold
            if score < task_data["validation_threshold"]:
                print(f"⚠️ Score {score:.4f} below threshold {task_data['validation_threshold']:.2f}")
                print(f"   Skipping submission to avoid penalty")
                self.stats["failed_submissions"] += 1
                return False
            
            # Submit to validator
            print(f"\n📤 Submitting to validator...")
            
            import base64
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
            
            total_time = time.time() - start_time
            
            if response and hasattr(response, 'feedback') and response.feedback:
                if not response.feedback.validation_failed:
                    reward = response.feedback.current_miner_reward
                    fidelity = response.feedback.task_fidelity_score
                    
                    print(f"🎉 SUBMISSION SUCCESSFUL!")
                    print(f"   Fidelity: {fidelity:.4f}")
                    print(f"   Reward: {reward:.6f} TAO")
                    print(f"   Total time: {total_time:.1f}s")
                    
                    self.stats["successful_submissions"] += 1
                    self.stats["total_earned"] += reward
                    self.stats["last_task_time"] = time.time()
                    
                    return True
                else:
                    print(f"❌ Submission failed validator check")
                    print(f"   Reason: {response.feedback.feedback_text}")
                    self.stats["failed_submissions"] += 1
                    return False
            else:
                print(f"❌ No feedback from validator")
                self.stats["failed_submissions"] += 1
                return False
                
        except Exception as e:
            print(f"❌ Task processing error: {e}")
            self.stats["failed_submissions"] += 1
            return False

    def print_status(self):
        """Print current status"""
        runtime = time.time() - self.stats["start_time"]
        hours = int(runtime // 3600)
        minutes = int((runtime % 3600) // 60)
        
        success_rate = (self.stats["successful_submissions"] / 
                       max(1, self.stats["successful_submissions"] + self.stats["failed_submissions"]) * 100)
        
        print(f"\n📊 MINING STATUS ({hours:02d}:{minutes:02d} runtime)")
        print(f"   Checks: {self.stats['total_checks']}")
        print(f"   Tasks found: {self.stats['real_tasks_found']}")
        print(f"   Successful: {self.stats['successful_submissions']}")
        print(f"   Failed: {self.stats['failed_submissions']}")
        print(f"   Success rate: {success_rate:.1f}%")
        print(f"   Total earned: {self.stats['total_earned']:.6f} TAO")
        print(f"   Daily tasks: {self.stats['daily_task_count']}/{self.max_daily_tasks}")

    async def run_monitoring_loop(self):
        """Main monitoring loop"""
        print("🔄 Starting monitoring loop...")
        
        # Initialize Bittensor
        if not await self.initialize_bittensor():
            print("❌ Failed to initialize Bittensor")
            return False
        
        # Ensure services are running
        if not self.ensure_services_running():
            print("❌ Failed to start required services")
            return False
        
        print(f"✅ Monitoring started - checking every {self.check_interval}s")
        
        while not self.shutdown_requested:
            try:
                # Check for tasks
                task_data = await self.check_for_real_tasks()
                
                if task_data:
                    # Process the task
                    success = await self.process_real_task(task_data)
                    
                    if success:
                        print(f"✅ Task completed successfully!")
                    else:
                        print(f"❌ Task failed")
                    
                    # Brief pause after task
                    await asyncio.sleep(5)
                else:
                    # No tasks found
                    if self.stats["total_checks"] % 10 == 0:  # Print status every 10 checks
                        self.print_status()
                
                # Wait before next check
                await asyncio.sleep(self.check_interval)
                
            except KeyboardInterrupt:
                print("\n🛑 Shutdown requested")
                self.shutdown_requested = True
                break
            except Exception as e:
                print(f"❌ Monitoring error: {e}")
                await asyncio.sleep(30)  # Wait longer on error
        
        print("🔄 Monitoring loop ended")
        return True

    def cleanup(self):
        """Cleanup processes and resources"""
        print("🧹 Cleaning up...")
        if self.validation_process:
            self.validation_process.terminate()
        if self.generation_process:
            self.generation_process.terminate()
        os.system("pkill -f 'serve.py' 2>/dev/null")
        os.system("pkill -f 'flux_hunyuan_bpt_generation_server' 2>/dev/null")


async def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Production Mining Monitor")
    parser.add_argument("--wallet", type=str, default="manbeast", help="Wallet name")
    parser.add_argument("--hotkey", type=str, default="beastman", help="Hotkey name")
    parser.add_argument("--interval", type=int, default=30, help="Check interval in seconds")
    parser.add_argument("--max-daily", type=int, default=50, help="Max tasks per day")
    
    args = parser.parse_args()
    
    monitor = ProductionMiningMonitor(args.wallet, args.hotkey)
    monitor.check_interval = args.interval
    monitor.max_daily_tasks = args.max_daily
    
    # Setup signal handlers
    def signal_handler(signum, frame):
        print(f"\n🛑 Received signal {signum}")
        monitor.shutdown_requested = True
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    try:
        success = await monitor.run_monitoring_loop()
        return 0 if success else 1
    except Exception as e:
        print(f"💥 Fatal error: {e}")
        traceback.print_exc()
        return 1
    finally:
        monitor.cleanup()


if __name__ == "__main__":
    import sys
    exit_code = asyncio.run(main())
    sys.exit(exit_code) 