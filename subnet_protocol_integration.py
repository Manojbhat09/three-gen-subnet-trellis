#!/usr/bin/env python3
# Subnet 17 Protocol Integration - Real Synapse Usage
# Purpose: Show how to integrate with actual subnet protocol synapses

import asyncio
import time
import base64
from typing import List, Dict, Optional
import bittensor as bt
import traceback
from pydantic import BaseModel, Field
import uuid

# You'll need to import these from the actual subnet repository
# from subnet.protocol import PullTask, SubmitResults
# For now, we'll create mock classes to show the structure

# Miner License Consent Declaration (from neurons/common/miner_license_consent_declaration.py)
MINER_LICENSE_CONSENT_DECLARATION = (
    "I, as a miner on SN17, have obtained all licenses, rights and consents required to use, reproduce, "
    "modify, display, distribute and make available my submitted results to this subnet and its end users"
)

class Task(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))  # Unique task identifier.
    prompt: str = Field(default="")  # Prompt to use for 3D generation.


class Feedback(BaseModel):
    validation_failed: bool = False  # Set if validation failed.
    task_fidelity_score: float = 0.0  # Calculated fidelity score for the given task.
    average_fidelity_score: float = 0.0  # Average of all computed fidelity scores.
    generations_within_the_window: int = (
        0  # Total accepted generations (non-zero fidelity score) within the last 4 hours.
    )
    current_miner_reward: float = 0  # Recent miners reward value.


class MockPullTask(bt.Synapse):
    """Mock PullTask synapse - replace with real import from neurons.common.protocol"""
    
    task: Task | None = None  # Task assigned by validator to be completed by miner.

    # Minimum score required for task results to be accepted by validators.
    # Results below this threshold are rejected, penalizing the miner.
    # Miners can submit empty results to avoid penalties if unable to meet threshold.
    validation_threshold: float = 0.6

    # Minimum expected time (in seconds) for task completion.
    # Used to calculate effective cooldown:
    # - Faster completion results in longer cooldown
    # - Cooldown is reduced by actual completion time, up to throttle_period
    # Example: With 60s cooldown and 20s throttle_period:
    # - 5s completion -> 55s cooldown (reduced by 5s)
    # - 15s completion -> 45s cooldown (reduced by 15s)
    # - 20s completion -> 40s cooldown (reduced by full throttle_period)
    # - 30s completion -> 40s cooldown (still reduced by throttle_period max)
    throttle_period: int = 0

    cooldown_until: int = 0  # Unix timestamp indicating when miner can pull the next task from this validator.
    cooldown_violations: int = 0  # Count of miner's failures to respect the mandatory cooling period.


class MockSubmitResults(bt.Synapse):
    """Mock SubmitResults synapse - replace with real import from neurons.common.protocol"""
    
    task: Task = Field(default_factory=Task)  # The original task miner is submitting results for.
    results: str  # Generated assets, encoded as a string.

    data_format: str = "ply"  # Reserved for future use.
    data_ver: int = 0  # Reserved for future use.
    compression: int = 0
    # End of experiment.
    #   0 - uncompressed data (to be deprecated with the next release!)
    #   1 - zstd compression (to be deprecated with the next release!)
    #   2 - spz compression (https://github.com/404-Repo/spz) (switch to it now!)

    submit_time: int  # time.time_ns()

    # Miner signature:
    # b64encode(sign(f'{MINER_LICENSE_CONSENT_DECLARATION}{submit_time}{prompt}{validator.hotkey}{miner.hotkey}'))
    signature: str

    feedback: Feedback | None = None  # Feedback provided by a validator.
    cooldown_until: int = 0  # UTC time indicating when the miner is allowed to pull the next task from this validator.

class SubnetProtocolIntegration:
    """Integration with real subnet protocol synapses"""
    
    def __init__(self, wallet_name: str, hotkey_name: str):
        self.wallet_name = wallet_name
        self.hotkey_name = hotkey_name
        self.wallet = None
        self.dendrite = None
        self.metagraph = None
        self.subtensor = None
        
        print("🔗 Subnet Protocol Integration")
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

    async def pull_real_task_from_validator(self, validator_uid: int) -> Optional[Dict]:
        """Pull a real task using actual subnet protocol"""
        try:
            if validator_uid >= len(self.metagraph.axons):
                return None
            
            axon = self.metagraph.axons[validator_uid]
            validator_hotkey = self.metagraph.hotkeys[validator_uid]
            
            print(f"📡 Pulling task from validator {validator_uid}")
            print(f"   Hotkey: {validator_hotkey[:10]}...")
            print(f"   Axon: {axon.ip}:{axon.port}")
            
            # Create real PullTask synapse
            # In production, replace MockPullTask with: from subnet.protocol import PullTask
            pull_synapse = MockPullTask()
            
            # Make the actual dendrite call
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
                print(f"     Cooldown until: {response.cooldown_until}")
                
                return {
                    "task_id": response.task.id,
                    "prompt": response.task.prompt,
                    "task": response.task,
                    "validation_threshold": response.validation_threshold,
                    "throttle_period": response.throttle_period,
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

    async def submit_real_results_to_validator(self, task_data: Dict, results: str, 
                                             validation_score: float, compression: int) -> bool:
        """Submit results using actual subnet protocol"""
        try:
            print(f"📤 Submitting to validator {task_data['validator_uid']}")
            
            # Create submission timestamp and signature
            submit_time = time.time_ns()
            
            # Create real signature
            message = (
                f"{MINER_LICENSE_CONSENT_DECLARATION}"
                f"{submit_time}{task_data['prompt']}"
                f"{task_data['validator_hotkey']}{self.wallet.hotkey.ss58_address}"
            )
            signature = base64.b64encode(self.wallet.hotkey.sign(message)).decode()
            
            # Create real SubmitResults synapse
            # In production, replace MockSubmitResults with: from subnet.protocol import SubmitResults
            submit_synapse = MockSubmitResults()
            submit_synapse.task = task_data["task"]  # Use the Task object
            submit_synapse.results = results
            submit_synapse.compression = compression
            submit_synapse.data_format = "ply"
            submit_synapse.data_ver = 0
            submit_synapse.submit_time = submit_time
            submit_synapse.signature = signature
            
            # Note: feedback and cooldown_until are set by the validator in response
            
            # Make the actual dendrite call
            response = await self.dendrite.call(
                target_axon=task_data["axon"],
                synapse=submit_synapse,
                deserialize=False,
                timeout=300.0
            )
            
            if response and hasattr(response, 'feedback') and response.feedback:
                # Check if validation was successful (not failed)
                if not response.feedback.validation_failed:
                print(f"  ✅ Submission successful!")
                    print(f"     Task fidelity score: {response.feedback.task_fidelity_score:.4f}")
                    print(f"     Average fidelity score: {response.feedback.average_fidelity_score:.4f}")
                    print(f"     Generations in window: {response.feedback.generations_within_the_window}")
                    print(f"     Current miner reward: {response.feedback.current_miner_reward:.4f}")
                print(f"     Compression: {compression}")
                print(f"     Results size: {len(results)}")
                    print(f"     Next cooldown until: {response.cooldown_until}")
                return True
                else:
                    print(f"  ❌ Submission failed validation")
                    print(f"     Task fidelity score: {response.feedback.task_fidelity_score:.4f}")
                    return False
            else:
                print(f"  ❌ Submission failed or no feedback received")
                return False
                
        except Exception as e:
            print(f"  ❌ Submission error: {e}")
            return False

    def get_active_validators(self) -> List[int]:
        """Get active validators with proper stake and serving status"""
        if not self.metagraph:
            return []
        
        active_validators = []
        for uid, (stake, axon, hotkey) in enumerate(zip(
            self.metagraph.stake, 
            self.metagraph.axons, 
            self.metagraph.hotkeys
        )):
            # Check validator criteria
            if (stake > 1000 and  # Minimum stake (adjust as needed)
                axon.is_serving and 
                axon.ip != "0.0.0.0" and
                uid != 180):  # Exclude blacklisted
                active_validators.append(uid)
                print(f"  📡 Validator {uid}: {hotkey[:10]}... (stake: {stake:.1f})")
        
        print(f"Found {len(active_validators)} active validators")
        return active_validators

    async def run_real_mining_demo(self) -> Dict:
        """Demonstrate real mining with actual protocol"""
        print("🚀 Real Mining Demo with Subnet Protocol")
        print("=" * 50)
        
        # Initialize
        if not await self.initialize_bittensor():
            return {"error": "Initialization failed"}
        
        # Get active validators
        validators = self.get_active_validators()
        if not validators:
            return {"error": "No active validators found"}
        
        # Try to pull tasks from first few validators
        tasks = []
        for validator_uid in validators[:3]:  # Try first 3 validators
            task = await self.pull_real_task_from_validator(validator_uid)
            if task:
                tasks.append(task)
        
        if not tasks:
            return {"error": "No tasks available"}
        
        print(f"\n🎯 Processing {len(tasks)} real tasks...")
        
        # Process each task
        results = []
        for i, task in enumerate(tasks, 1):
            print(f"\n{'='*30} Task {i} {'='*30}")
            
            # Here you would integrate with your generation server
            # For demo, we'll simulate the mining process
            print(f"⛏️ Mining: '{task['prompt']}'")
            
            # Simulate generation (replace with real generation server call)
            await asyncio.sleep(2)  # Simulate generation time
            
            # Mock results (replace with real generation results)
            mock_results = f"PLY_DATA_FOR_TASK_{task['task_id']}"
            mock_score = 0.85
            mock_compression = 2
            
            print(f"  ✅ Generated 3D model")
            print(f"     Score: {mock_score:.4f}")
            print(f"     Compression: {mock_compression}")
            
            # Submit to validator
            success = await self.submit_real_results_to_validator(
                task, mock_results, mock_score, mock_compression
            )
            
            results.append({
                "task_id": task["task_id"],
                "validator_uid": task["validator_uid"],
                "success": success,
                "score": mock_score
            })
        
        # Generate report
        successful = len([r for r in results if r["success"]])
        success_rate = successful / len(results) if results else 0
        
        print(f"\n📊 Mining Session Results")
        print("=" * 50)
        print(f"Total Tasks: {len(results)}")
        print(f"Successful: {successful}/{len(results)} ({success_rate*100:.1f}%)")
        
        for result in results:
            status = "✅ SUCCESS" if result["success"] else "❌ FAILED"
            print(f"  {result['task_id']}: {status} (validator {result['validator_uid']})")
        
        return {
            "total_tasks": len(results),
            "successful": successful,
            "success_rate": success_rate,
            "results": results
        }


async def main():
    """Run the real protocol integration demo"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Subnet 17 Protocol Integration Demo")
    parser.add_argument("--wallet", type=str, required=True, help="Wallet name")
    parser.add_argument("--hotkey", type=str, required=True, help="Hotkey name")
    
    args = parser.parse_args()
    
    print("🔗 Subnet 17 Protocol Integration Demo")
    print("=" * 50)
    print("⚠️  IMPORTANT NOTES:")
    print("   1. Replace MockPullTask with: from neurons.common.protocol import PullTask")
    print("   2. Replace MockSubmitResults with: from neurons.common.protocol import SubmitResults")
    print("   3. Import license declaration: from neurons.common.miner_license_consent_declaration import MINER_LICENSE_CONSENT_DECLARATION")
    print("   4. Import Task and Feedback models: from neurons.common.protocol import Task, Feedback")
    print("   5. Integrate with your actual generation server")
    print("   6. Test with real validators on the network")
    print()
    
    integration = SubnetProtocolIntegration(args.wallet, args.hotkey)
    
    try:
        results = await integration.run_real_mining_demo()
        
        if "error" in results:
            print(f"❌ Demo failed: {results['error']}")
            return 1
        
        if results["success_rate"] >= 0.8:
            print("🎉 Protocol integration working well!")
            return 0
        else:
            print("⚠️ Protocol integration needs work")
            return 1
            
    except KeyboardInterrupt:
        print("\n🛑 Demo interrupted")
        return 130
    except Exception as e:
        print(f"\n💥 Unexpected error: {e}")
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    import sys
    exit_code = asyncio.run(main())
    sys.exit(exit_code) 