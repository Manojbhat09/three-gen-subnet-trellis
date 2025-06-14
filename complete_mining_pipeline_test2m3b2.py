#!/usr/bin/env python3
"""
Complete Mining Pipeline Test with Memory Coordination
Test the full mining pipeline with registered miner test2m3b2/t2m3b21
"""

import asyncio
import aiohttp
import bittensor as bt
import json
import time
import requests
import subprocess
import psutil
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any
import pybase64
import zstandard

# Import real protocol classes
from neurons.common.protocol import PullTask, SubmitResults, Task, Feedback
from neurons.common.miner_license_consent_declaration import MINER_LICENSE_CONSENT_DECLARATION


class GPUMemoryCoordinator:
    """Coordinates GPU memory between validation and generation servers"""
    
    def __init__(self, validation_url: str = "http://localhost:10006", generation_url: str = "http://localhost:8095"):
        self.validation_url = validation_url
        self.generation_url = generation_url
        
    async def get_validation_gpu_status(self) -> Dict[str, Any]:
        """Get GPU status from validation server"""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(f"{self.validation_url}/gpu_status/") as response:
                    if response.status == 200:
                        return await response.json()
                    else:
                        return {"error": f"HTTP {response.status}"}
        except Exception as e:
            return {"error": str(e)}
    
    async def cleanup_validation_gpu(self) -> Dict[str, Any]:
        """Force validation server to cleanup GPU memory"""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(f"{self.validation_url}/cleanup_gpu/") as response:
                    if response.status == 200:
                        return await response.json()
                    else:
                        return {"error": f"HTTP {response.status}"}
        except Exception as e:
            return {"error": str(e)}
    
    async def check_generation_health(self) -> bool:
        """Check if generation server is healthy"""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(f"{self.generation_url}/health/") as response:
                    return response.status == 200
        except:
            return False
    
    async def unload_validation_models(self) -> Dict[str, Any]:
        """Unload validation models to free maximum GPU memory"""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(f"{self.validation_url}/unload_models/") as response:
                    if response.status == 200:
                        return await response.json()
                    else:
                        return {"error": f"HTTP {response.status}"}
        except Exception as e:
            return {"error": str(e)}
    
    async def reload_validation_models(self) -> Dict[str, Any]:
        """Reload validation models back to GPU"""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(f"{self.validation_url}/reload_models/") as response:
                    if response.status == 200:
                        return await response.json()
                    else:
                        return {"error": f"HTTP {response.status}"}
        except Exception as e:
            return {"error": str(e)}

    async def coordinate_memory_for_generation(self) -> Dict[str, Any]:
        """Coordinate GPU memory before generation with aggressive model unloading"""
        print("🧠 Coordinating GPU memory for generation...")
        
        # 1. Check current validation server GPU status
        val_status = await self.get_validation_gpu_status()
        print(f"   Validation GPU status: {val_status}")
        
        # 2. Unload validation models completely
        unload_result = await self.unload_validation_models()
        print(f"   Validation model unload result: {unload_result}")
        
        # 3. Force additional cleanup
        cleanup_result = await self.cleanup_validation_gpu()
        print(f"   Additional cleanup result: {cleanup_result}")
        
        # 4. Wait for memory to settle
        await asyncio.sleep(2)
        
        # 5. Check generation server health
        gen_healthy = await self.check_generation_health()
        print(f"   Generation server healthy: {gen_healthy}")
        
        return {
            "validation_status": val_status,
            "unload_result": unload_result,
            "cleanup_result": cleanup_result,
            "generation_healthy": gen_healthy
        }


class MiningPipelineTestCoordinated:
    """Complete mining pipeline with GPU memory coordination"""
    
    def __init__(self):
        # Registered miner credentials
        self.wallet_name = "test2m3b2"
        self.hotkey_name = "t2m3b21"
        self.netuid = 17
        
        # Server URLs
        self.validation_url = "http://localhost:10006"
        self.generation_url = "http://localhost:8095"
        
        # Initialize components
        self.coordinator = GPUMemoryCoordinator(self.validation_url, self.generation_url)
        self.wallet = None
        self.subtensor = None
        self.dendrite = None
        
        # Statistics
        self.stats = {
            "tasks_pulled": 0,
            "tasks_processed": 0,
            "successful_generations": 0,
            "successful_validations": 0,
            "successful_submissions": 0,
            "total_score": 0.0,
            "start_time": time.time()
        }
        
    def setup_bittensor(self):
        """Initialize Bittensor components"""
        print("🔧 Setting up Bittensor components...")
        
        try:
            # Load wallet
            self.wallet = bt.wallet(name=self.wallet_name, hotkey=self.hotkey_name)
            print(f"✓ Wallet loaded: {self.wallet}")
            
            # Connect to subtensor
            self.subtensor = bt.subtensor(network="finney")
            print(f"✓ Subtensor connected: {self.subtensor}")
            
            # Initialize dendrite
            self.dendrite = bt.dendrite(wallet=self.wallet)
            print(f"✓ Dendrite initialized: {self.dendrite}")
            
            # Get validator info
            metagraph = self.subtensor.metagraph(self.netuid)
            print(f"✓ Connected to subnet {self.netuid}")
            print(f"  Total neurons: {len(metagraph.n)}")
            
            # Find active validators
            validators = []
            for uid in range(len(metagraph.n)):
                if metagraph.validator_permit[uid] and metagraph.active[uid]:
                    validators.append({
                        'uid': uid,
                        'hotkey': metagraph.hotkeys[uid],
                        'stake': metagraph.total_stake[uid].item(),
                        'axon': metagraph.axons[uid]
                    })
            
            self.validators = sorted(validators, key=lambda x: x['stake'], reverse=True)
            print(f"✓ Found {len(self.validators)} active validators")
            
            for i, v in enumerate(self.validators[:3]):
                print(f"  {i+1}. UID {v['uid']}: {v['stake']:.1f} TAO - {v['hotkey'][:10]}...")
                
            return True
            
        except Exception as e:
            print(f"❌ Bittensor setup failed: {e}")
            return False
    
    async def pull_task_from_validators(self) -> Optional[Task]:
        """Pull task from validators"""
        print("\n📥 Pulling task from validators...")
        
        for validator in self.validators:
            try:
                print(f"   Trying validator UID {validator['uid']} ({validator['stake']:.1f} TAO)...")
                
                # Create PullTask synapse
                pull_request = PullTask()
                
                # Send request to validator
                response = await self.dendrite.forward(
                    axons=[validator['axon']],
                    synapse=pull_request,
                    timeout=30
                )
                
                if response and len(response) > 0:
                    resp = response[0]
                    if hasattr(resp, 'task') and resp.task:
                        print(f"✓ Task received from validator UID {validator['uid']}")
                        print(f"  Task ID: {resp.task.id}")
                        print(f"  Prompt: {resp.task.prompt}")
                        print(f"  Validation threshold: {resp.validation_threshold}")
                        print(f"  Throttle period: {resp.throttle_period}")
                        
                        self.stats["tasks_pulled"] += 1
                        return resp.task
                    else:
                        print(f"   No task available from validator UID {validator['uid']}")
                else:
                    print(f"   No response from validator UID {validator['uid']}")
                    
            except Exception as e:
                print(f"   Error with validator UID {validator['uid']}: {e}")
                continue
        
        print("❌ No tasks available from any validator")
        return None
    
    async def generate_3d_model_coordinated(self, prompt: str) -> Optional[bytes]:
        """Generate 3D model with GPU memory coordination"""
        print(f"\n🎯 Generating 3D model with coordination for: '{prompt}'")
        
        # Coordinate GPU memory before generation
        coord_result = await self.coordinator.coordinate_memory_for_generation()
        
        if not coord_result["generation_healthy"]:
            print("❌ Generation server not healthy after coordination")
            return None
        
        try:
            # Make generation request
            data = {"prompt": prompt}
            
            print("   Making generation request...")
            response = requests.post(
                f"{self.generation_url}/generate/",
                data=data,  # Form data, not JSON
                timeout=300  # 5 minutes timeout
            )
            
            if response.status_code == 200:
                ply_data = response.content
                
                # Get metadata from headers
                generation_id = response.headers.get('X-Generation-ID', 'unknown')
                face_count = response.headers.get('X-Face-Count', 'unknown')
                vertex_count = response.headers.get('X-Vertex-Count', 'unknown')
                local_score = response.headers.get('X-Local-Validation-Score', 'unknown')
                mining_ready = response.headers.get('X-Mining-Ready', 'unknown')
                
                print(f"✓ Generation successful!")
                print(f"  Generation ID: {generation_id}")
                print(f"  PLY size: {len(ply_data):,} bytes")
                print(f"  Vertices: {vertex_count}, Faces: {face_count}")
                print(f"  Local validation score: {local_score}")
                print(f"  Mining ready: {mining_ready}")
                
                self.stats["successful_generations"] += 1
                return ply_data
            else:
                print(f"❌ Generation failed with status {response.status_code}")
                print(f"   Response: {response.text}")
                return None
                
        except requests.exceptions.Timeout:
            print("❌ Generation request timed out")
            return None
        except Exception as e:
            print(f"❌ Generation failed: {e}")
            return None
    
    async def validate_locally(self, prompt: str, ply_data: bytes) -> float:
        """Validate PLY data locally using validation server"""
        print(f"\n✅ Validating locally: '{prompt}'")
        
        try:
            # First, reload validation models if they were unloaded
            print("   Reloading validation models...")
            reload_result = await self.coordinator.reload_validation_models()
            print(f"   Model reload result: {reload_result}")
            
            # Wait for models to be ready
            await asyncio.sleep(2)
            
            # Prepare request data
            encoded_data = pybase64.b64encode(ply_data).decode('utf-8')
            
            request_data = {
                "data": encoded_data,
                "prompt": prompt,
                "compression": 0,  # No compression
                "generate_preview": False,
                "preview_score_threshold": 0.5
            }
            
            # Make validation request
            response = requests.post(
                f"{self.validation_url}/validate_txt_to_3d_ply/",
                json=request_data,
                timeout=120
            )
            
            if response.status_code == 200:
                result = response.json()
                score = result.get('score', 0.0)
                
                print(f"✓ Local validation complete")
                print(f"  Final score: {score:.3f}")
                print(f"  IQA score: {result.get('iqa', 0.0):.3f}")
                print(f"  Alignment score: {result.get('alignment_score', 0.0):.3f}")
                print(f"  SSIM score: {result.get('ssim', 0.0):.3f}")
                print(f"  LPIPS score: {result.get('lpips', 0.0):.3f}")
                
                self.stats["successful_validations"] += 1
                return score
            else:
                print(f"❌ Validation failed with status {response.status_code}")
                return 0.0
                
        except Exception as e:
            print(f"❌ Validation failed: {e}")
            return 0.0
    
    def create_signature(self, task: Task, ply_data: bytes) -> str:
        """Create cryptographic signature with license consent"""
        try:
            # Create the message to sign (task ID + license declaration)
            message = f"{task.id}{MINER_LICENSE_CONSENT_DECLARATION}"
            
            # Sign with hotkey
            signature = self.wallet.hotkey.sign(message.encode()).hex()
            return signature
            
        except Exception as e:
            print(f"❌ Signature creation failed: {e}")
            return ""
    
    async def submit_results(self, task: Task, ply_data: bytes, local_score: float) -> bool:
        """Submit results to validators"""
        print(f"\n📤 Submitting results for task {task.id}")
        
        try:
            # Create signature
            signature = self.create_signature(task, ply_data)
            if not signature:
                print("❌ Failed to create signature")
                return False
            
            # Convert PLY data to base64 string
            results_data = pybase64.b64encode(ply_data).decode('utf-8')
            
            # Create SubmitResults synapse
            submit_request = SubmitResults(
                task=task,
                results=results_data,
                compression=0,  # No compression
                data_format="ply",
                submit_time=int(time.time() * 1_000_000_000),  # Nanoseconds
                signature=signature
            )
            
            # Submit to validators (try top 3 validators)
            successful_submissions = 0
            
            for validator in self.validators[:3]:
                try:
                    print(f"   Submitting to validator UID {validator['uid']}...")
                    
                    response = await self.dendrite.forward(
                        axons=[validator['axon']],
                        synapse=submit_request,
                        timeout=60
                    )
                    
                    if response and len(response) > 0:
                        resp = response[0]
                        if hasattr(resp, 'feedback') and resp.feedback:
                            feedback = resp.feedback
                            print(f"✓ Submission successful to validator UID {validator['uid']}")
                            print(f"  Validation failed: {feedback.validation_failed}")
                            print(f"  Task fidelity score: {feedback.task_fidelity_score:.3f}")
                            print(f"  Average fidelity score: {feedback.average_fidelity_score:.3f}")
                            print(f"  Generations in window: {feedback.generations_within_the_window}")
                            print(f"  Current miner reward: {feedback.current_miner_reward:.6f}")
                            
                            successful_submissions += 1
                            self.stats["total_score"] += feedback.task_fidelity_score
                        else:
                            print(f"   No feedback from validator UID {validator['uid']}")
                    else:
                        print(f"   No response from validator UID {validator['uid']}")
                        
                except Exception as e:
                    print(f"   Error submitting to validator UID {validator['uid']}: {e}")
                    continue
            
            if successful_submissions > 0:
                self.stats["successful_submissions"] += 1
                print(f"✓ Results submitted successfully to {successful_submissions} validators")
                return True
            else:
                print("❌ Failed to submit to any validator")
                return False
                
        except Exception as e:
            print(f"❌ Submission failed: {e}")
            return False
    
    async def process_task(self, task: Task) -> bool:
        """Process a single task end-to-end"""
        print(f"\n🔄 Processing task {task.id}: '{task.prompt}'")
        self.stats["tasks_processed"] += 1
        
        # Step 1: Generate 3D model with coordination
        ply_data = await self.generate_3d_model_coordinated(task.prompt)
        if not ply_data:
            print("❌ Generation failed, skipping task")
            return False
        
        # Step 2: Validate locally
        local_score = await self.validate_locally(task.prompt, ply_data)
        if local_score < 0.3:  # Minimum quality threshold
            print(f"❌ Local validation score too low ({local_score:.3f}), skipping submission")
            return False
        
        # Step 3: Submit results
        success = await self.submit_results(task, ply_data, local_score)
        
        return success
    
    def print_statistics(self):
        """Print mining statistics"""
        runtime = time.time() - self.stats["start_time"]
        
        print("\n" + "="*60)
        print("📊 MINING STATISTICS")
        print("="*60)
        print(f"Runtime: {runtime:.1f} seconds ({runtime/60:.1f} minutes)")
        print(f"Tasks pulled: {self.stats['tasks_pulled']}")
        print(f"Tasks processed: {self.stats['tasks_processed']}")
        print(f"Successful generations: {self.stats['successful_generations']}")
        print(f"Successful validations: {self.stats['successful_validations']}")
        print(f"Successful submissions: {self.stats['successful_submissions']}")
        
        if self.stats["tasks_processed"] > 0:
            success_rate = (self.stats["successful_submissions"] / self.stats["tasks_processed"]) * 100
            print(f"Success rate: {success_rate:.1f}%")
            
        if self.stats["successful_submissions"] > 0:
            avg_score = self.stats["total_score"] / self.stats["successful_submissions"]
            print(f"Average score: {avg_score:.3f}")
        
        print("="*60)
    
    async def run_mining_loop(self, max_tasks: int = 5):
        """Run the complete mining loop"""
        print("🚀 Starting coordinated mining pipeline test...")
        print(f"Wallet: {self.wallet_name}")
        print(f"Hotkey: {self.hotkey_name}")
        print(f"Subnet: {self.netuid}")
        print(f"Max tasks: {max_tasks}")
        
        # Setup Bittensor
        if not self.setup_bittensor():
            return False
        
        # Initial coordination check
        coord_result = await self.coordinator.coordinate_memory_for_generation()
        print(f"Initial coordination result: {coord_result}")
        
        tasks_completed = 0
        
        try:
            while tasks_completed < max_tasks:
                print(f"\n{'='*60}")
                print(f"MINING ITERATION {tasks_completed + 1}/{max_tasks}")
                print(f"{'='*60}")
                
                # Pull task
                task = await self.pull_task_from_validators()
                if not task:
                    print("❌ No task available, waiting 30 seconds...")
                    await asyncio.sleep(30)
                    continue
                
                # Process task
                success = await self.process_task(task)
                
                if success:
                    tasks_completed += 1
                    print(f"✅ Task {task.id} completed successfully!")
                else:
                    print(f"❌ Task {task.id} failed")
                
                # Print current statistics
                self.print_statistics()
                
                # Wait between tasks
                if tasks_completed < max_tasks:
                    print(f"\n⏳ Waiting 10 seconds before next task...")
                    await asyncio.sleep(10)
        
        except KeyboardInterrupt:
            print("\n⚠️  Mining interrupted by user")
        except Exception as e:
            print(f"\n❌ Mining loop error: {e}")
            import traceback
            traceback.print_exc()
        
        # Final statistics
        print("\n🏁 Mining test completed!")
        self.print_statistics()
        
        return tasks_completed > 0


async def main():
    """Main function"""
    print("🎯 Complete Mining Pipeline Test with GPU Memory Coordination")
    print("=" * 80)
    
    # Check server health first
    coordinator = GPUMemoryCoordinator()
    
    print("🔍 Checking server health...")
    
    # Check validation server
    val_status = await coordinator.get_validation_gpu_status()
    print(f"Validation server status: {val_status}")
    
    # Check generation server
    gen_health = await coordinator.check_generation_health()
    print(f"Generation server health: {gen_health}")
    
    if not gen_health:
        print("❌ Generation server not healthy. Please start it first:")
        print("   conda activate hunyuan3d")
        print("   python flux_hunyuan_sugar_generation_server.py")
        return
    
    # Run mining pipeline
    pipeline = MiningPipelineTestCoordinated()
    success = await pipeline.run_mining_loop(max_tasks=3)
    
    if success:
        print("✅ Mining pipeline test completed successfully!")
    else:
        print("❌ Mining pipeline test failed!")


if __name__ == "__main__":
    asyncio.run(main()) 