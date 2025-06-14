#!/usr/bin/env python3
"""
Pull Tasks from Validators and Save
Purpose: Pull real tasks from subnet 17 validators and save them for analysis
"""

import asyncio
import json
import time
import os
import sys
from datetime import datetime
from typing import Dict, List, Optional
from pathlib import Path
import traceback

import bittensor as bt

# Add the three-gen-subnet path for imports
sys.path.append('/home/mbhat/three-gen-subnet-trellis/three-gen-subnet/neurons')
from common.protocol import PullTask, Task

# Configuration
WALLET_NAME = "test2m3b2"
HOTKEY_NAME = "t2m3b21"
NETUID = 17
SAVE_DIR = "pulled_tasks"
MAX_VALIDATORS_TO_TRY = 20
PULL_TIMEOUT = 15.0

class TaskPuller:
    def __init__(self):
        self.wallet = None
        self.subtensor = None
        self.dendrite = None
        self.metagraph = None
        self.save_dir = Path(SAVE_DIR)
        self.save_dir.mkdir(exist_ok=True)
        
        # Statistics
        self.stats = {
            "validators_tried": 0,
            "tasks_pulled": 0,
            "failed_pulls": 0,
            "empty_responses": 0,
            "errors": 0
        }
        
    async def initialize(self) -> bool:
        """Initialize Bittensor components"""
        try:
            print("🔧 Initializing Bittensor components...")
            
            # Initialize subtensor
            self.subtensor = bt.subtensor(network="finney")
            print(f"✅ Connected to Subtensor: {self.subtensor.network}")
            
            # Initialize wallet
            self.wallet = bt.wallet(name=WALLET_NAME, hotkey=HOTKEY_NAME)
            print(f"✅ Wallet loaded: {self.wallet.name}, Hotkey: {self.wallet.hotkey_str}")
            
            # Initialize dendrite
            self.dendrite = bt.dendrite(wallet=self.wallet)
            print(f"✅ Dendrite initialized for hotkey: {self.dendrite.keypair.ss58_address}")
            
            # Get metagraph
            self.metagraph = self.subtensor.metagraph(netuid=NETUID)
            print(f"✅ Metagraph synced for NETUID {NETUID}")
            
            # Verify registration
            my_hotkey = self.wallet.hotkey.ss58_address
            if my_hotkey not in self.metagraph.hotkeys:
                print(f"❌ Hotkey {my_hotkey} is not registered on NETUID {NETUID}")
                return False
            
            my_uid = self.metagraph.hotkeys.index(my_hotkey)
            print(f"✅ Registered with UID {my_uid}")
            
            return True
            
        except Exception as e:
            print(f"❌ Initialization failed: {e}")
            traceback.print_exc()
            return False
    
    def get_active_validators(self) -> List[Dict]:
        """Get list of active validators sorted by stake"""
        validators = []
        
        for uid, (stake, axon, hotkey) in enumerate(zip(
            self.metagraph.stake, 
            self.metagraph.axons, 
            self.metagraph.hotkeys
        )):
            # Filter for active validators
            if (stake > 100 and  # Minimum stake
                axon.is_serving and 
                axon.ip != '0.0.0.0' and 
                axon.port > 0):
                
                validators.append({
                    "uid": uid,
                    "stake": float(stake),
                    "hotkey": hotkey,
                    "axon": axon,
                    "ip": axon.ip,
                    "port": axon.port
                })
        
        # Sort by stake (highest first)
        validators.sort(key=lambda x: x["stake"], reverse=True)
        
        print(f"📊 Found {len(validators)} active validators")
        for i, val in enumerate(validators[:10]):  # Show top 10
            print(f"  {i+1:2d}. UID {val['uid']:3d}: {val['hotkey'][:10]}... "
                  f"stake: {val['stake']:8.1f} at {val['ip']}:{val['port']}")
        
        return validators
    
    async def pull_task_from_validator(self, validator: Dict) -> Optional[Dict]:
        """Pull a task from a specific validator"""
        try:
            print(f"\n📡 Pulling from validator UID {validator['uid']}...")
            print(f"   Hotkey: {validator['hotkey'][:10]}...")
            print(f"   Axon: {validator['ip']}:{validator['port']}")
            print(f"   Stake: {validator['stake']:.1f}")
            
            # Create PullTask synapse
            pull_synapse = PullTask()
            
            # Make the dendrite call
            response = await self.dendrite.call(
                target_axon=validator['axon'],
                synapse=pull_synapse,
                deserialize=False,
                timeout=PULL_TIMEOUT
            )
            
            self.stats["validators_tried"] += 1
            
            if response and hasattr(response, 'task') and response.task:
                print(f"  ✅ Task received!")
                print(f"     Task ID: {response.task.id}")
                print(f"     Prompt: '{response.task.prompt}'")
                print(f"     Validation threshold: {getattr(response, 'validation_threshold', 'N/A')}")
                print(f"     Throttle period: {getattr(response, 'throttle_period', 'N/A')}")
                print(f"     Cooldown until: {getattr(response, 'cooldown_until', 'N/A')}")
                
                self.stats["tasks_pulled"] += 1
                
                # Prepare task data for saving
                task_data = {
                    "timestamp": datetime.now().isoformat(),
                    "validator_uid": validator['uid'],
                    "validator_hotkey": validator['hotkey'],
                    "validator_stake": validator['stake'],
                    "validator_ip": validator['ip'],
                    "validator_port": validator['port'],
                    "task": {
                        "id": response.task.id,
                        "prompt": response.task.prompt
                    },
                    "response_data": {
                        "validation_threshold": getattr(response, 'validation_threshold', None),
                        "throttle_period": getattr(response, 'throttle_period', None),
                        "cooldown_until": getattr(response, 'cooldown_until', None),
                        "cooldown_violations": getattr(response, 'cooldown_violations', None)
                    },
                    "dendrite_info": {
                        "status_code": response.dendrite.status_code,
                        "status_message": response.dendrite.status_message,
                        "process_time": getattr(response.dendrite, 'process_time', None)
                    }
                }
                
                return task_data
                
            else:
                print(f"  ❌ No task available")
                if response:
                    print(f"     Status: {response.dendrite.status_code} - {response.dendrite.status_message}")
                    if hasattr(response, 'cooldown_until'):
                        print(f"     Cooldown until: {response.cooldown_until}")
                
                self.stats["empty_responses"] += 1
                return None
                
        except Exception as e:
            print(f"  ❌ Error pulling from validator {validator['uid']}: {e}")
            self.stats["errors"] += 1
            return None
    
    def save_task(self, task_data: Dict) -> str:
        """Save task data to file"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"task_{task_data['validator_uid']}_{timestamp}.json"
        filepath = self.save_dir / filename
        
        try:
            with open(filepath, 'w') as f:
                json.dump(task_data, f, indent=2, default=str)
            
            print(f"  💾 Saved to: {filepath}")
            return str(filepath)
            
        except Exception as e:
            print(f"  ❌ Failed to save task: {e}")
            return ""
    
    def save_summary(self, all_tasks: List[Dict]) -> str:
        """Save summary of all pulled tasks"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        summary_file = self.save_dir / f"task_summary_{timestamp}.json"
        
        summary = {
            "pull_session": {
                "timestamp": datetime.now().isoformat(),
                "wallet": WALLET_NAME,
                "hotkey": HOTKEY_NAME,
                "netuid": NETUID
            },
            "statistics": self.stats,
            "tasks": all_tasks,
            "unique_prompts": list(set(task["task"]["prompt"] for task in all_tasks)),
            "validators_with_tasks": list(set(task["validator_uid"] for task in all_tasks))
        }
        
        try:
            with open(summary_file, 'w') as f:
                json.dump(summary, f, indent=2, default=str)
            
            print(f"\n📋 Summary saved to: {summary_file}")
            return str(summary_file)
            
        except Exception as e:
            print(f"❌ Failed to save summary: {e}")
            return ""
    
    async def pull_tasks(self) -> List[Dict]:
        """Main task pulling loop"""
        print(f"\n🎯 Starting task pulling session...")
        print(f"   Max validators to try: {MAX_VALIDATORS_TO_TRY}")
        print(f"   Pull timeout: {PULL_TIMEOUT}s")
        print(f"   Save directory: {self.save_dir}")
        
        # Get active validators
        validators = self.get_active_validators()
        if not validators:
            print("❌ No active validators found")
            return []
        
        # Limit validators to try
        validators_to_try = validators[:MAX_VALIDATORS_TO_TRY]
        print(f"\n🔄 Trying {len(validators_to_try)} validators...")
        
        all_tasks = []
        
        for i, validator in enumerate(validators_to_try, 1):
            print(f"\n--- Validator {i}/{len(validators_to_try)} ---")
            
            task_data = await self.pull_task_from_validator(validator)
            
            if task_data:
                # Save individual task
                filepath = self.save_task(task_data)
                if filepath:
                    all_tasks.append(task_data)
            
            # Small delay between pulls
            await asyncio.sleep(1.0)
        
        return all_tasks
    
    def print_final_stats(self, all_tasks: List[Dict]):
        """Print final statistics"""
        print(f"\n" + "="*60)
        print(f"📊 TASK PULLING SESSION COMPLETE")
        print(f"="*60)
        print(f"Validators tried:     {self.stats['validators_tried']}")
        print(f"Tasks pulled:         {self.stats['tasks_pulled']}")
        print(f"Empty responses:      {self.stats['empty_responses']}")
        print(f"Errors:              {self.stats['errors']}")
        print(f"Success rate:        {(self.stats['tasks_pulled']/max(1,self.stats['validators_tried']))*100:.1f}%")
        
        if all_tasks:
            print(f"\n📝 TASK ANALYSIS:")
            prompts = [task["task"]["prompt"] for task in all_tasks]
            unique_prompts = set(prompts)
            print(f"Total tasks:         {len(all_tasks)}")
            print(f"Unique prompts:      {len(unique_prompts)}")
            print(f"Validators with tasks: {len(set(task['validator_uid'] for task in all_tasks))}")
            
            print(f"\n🎨 SAMPLE PROMPTS:")
            for i, prompt in enumerate(list(unique_prompts)[:5], 1):
                print(f"  {i}. {prompt}")
            
            if len(unique_prompts) > 5:
                print(f"  ... and {len(unique_prompts) - 5} more")
        
        print(f"\n💾 Files saved in: {self.save_dir}")

async def main():
    """Main entry point"""
    print("🚀 Subnet 17 Task Puller")
    print("=" * 50)
    
    puller = TaskPuller()
    
    try:
        # Initialize
        if not await puller.initialize():
            return 1
        
        # Pull tasks
        all_tasks = await puller.pull_tasks()
        
        # Save summary
        if all_tasks:
            puller.save_summary(all_tasks)
        
        # Print final stats
        puller.print_final_stats(all_tasks)
        
        return 0
        
    except KeyboardInterrupt:
        print("\n🛑 Task pulling interrupted by user")
        return 130
    except Exception as e:
        print(f"\n💥 Unexpected error: {e}")
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    exit(asyncio.run(main())) 