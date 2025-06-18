#!/usr/bin/env python3
"""
Continuous Task Monitor - Keep checking validators for available tasks
"""

import asyncio
import time
import json
from datetime import datetime
from typing import Optional, Dict, List
import bittensor as bt
import traceback

# Import our protocol classes
from subnet_protocol_integration import (
    MockPullTask as PullTask,
    Task,
    MINER_LICENSE_CONSENT_DECLARATION
)

class ContinuousTaskMonitor:
    """Continuous task monitor for production mining"""
    
    def __init__(self, wallet_name: str = "test2m3b2", hotkey_name: str = "t2m3b21"):
        self.wallet_name = wallet_name
        self.hotkey_name = hotkey_name
        self.wallet = None
        self.dendrite = None
        self.metagraph = None
        self.subtensor = None
        
        # Monitoring configuration
        self.check_interval = 30  # seconds between checks
        self.max_validators_per_round = 10
        self.cooldown_period = 300  # 5 minutes between requests to same validator
        
        # Statistics
        self.stats = {
            "start_time": time.time(),
            "total_rounds": 0,
            "total_requests": 0,
            "tasks_found": 0,
            "validator_cooldowns": {},
            "last_task_time": 0,
            "connection_errors": 0,
            "successful_connections": 0
        }
        
        print("🔄 Continuous Task Monitor")
        print(f"💰 Wallet: {wallet_name}")
        print(f"🔑 Hotkey: {hotkey_name}")
        print(f"⏱️ Check interval: {self.check_interval}s")
        print(f"🎯 Max validators per round: {self.max_validators_per_round}")

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

    def get_available_validators(self) -> List[tuple]:
        """Get validators that are available for task requests (not on cooldown)"""
        if not self.metagraph:
            return []
        
        current_time = time.time()
        available_validators = []
        
        for uid, (stake, axon, hotkey) in enumerate(zip(
            self.metagraph.stake, 
            self.metagraph.axons, 
            self.metagraph.hotkeys
        )):
            # Check validator criteria
            if (stake > 100 and  # Minimum stake
                axon.is_serving and 
                axon.ip != "0.0.0.0"):
                
                # Check cooldown
                last_request = self.stats["validator_cooldowns"].get(uid, 0)
                if current_time - last_request > self.cooldown_period:
                    available_validators.append((uid, stake, axon, hotkey))
        
        # Sort by stake (highest first)
        available_validators.sort(key=lambda x: x[1], reverse=True)
        return available_validators

    async def pull_task_from_validator(self, validator_uid: int, axon, hotkey: str) -> Optional[Dict]:
        """Pull a task from a specific validator"""
        try:
            print(f"📡 Checking validator {validator_uid} ({hotkey[:10]}...)")
            
            # Create PullTask synapse
            pull_synapse = PullTask()
            
            # Make the dendrite call
            response = await self.dendrite.call(
                target_axon=axon,
                synapse=pull_synapse,
                deserialize=False,
                timeout=15.0
            )
            
            # Update statistics
            self.stats["total_requests"] += 1
            self.stats["validator_cooldowns"][validator_uid] = time.time()
            
            if response and hasattr(response, 'task') and response.task:
                print(f"  🎉 TASK FOUND!")
                print(f"     Task ID: {response.task.id}")
                print(f"     Prompt: '{response.task.prompt}'")
                print(f"     Threshold: {response.validation_threshold}")
                
                self.stats["tasks_found"] += 1
                self.stats["last_task_time"] = time.time()
                self.stats["successful_connections"] += 1
                
                return {
                    "task_id": response.task.id,
                    "prompt": response.task.prompt,
                    "task": response.task,
                    "validation_threshold": response.validation_threshold,
                    "throttle_period": response.throttle_period,
                    "cooldown_until": response.cooldown_until,
                    "validator_hotkey": hotkey,
                    "validator_uid": validator_uid,
                    "axon": axon,
                    "found_time": time.time()
                }
            else:
                print(f"  ❌ No task available")
                self.stats["successful_connections"] += 1
                return None
                
        except Exception as e:
            print(f"  ❌ Connection failed: {str(e)[:50]}...")
            self.stats["connection_errors"] += 1
            return None

    async def check_validators_for_tasks(self) -> List[Dict]:
        """Check multiple validators for available tasks"""
        validators = self.get_available_validators()
        
        if not validators:
            print("⏸️ No validators available (all on cooldown)")
            return []
        
        print(f"🔍 Checking {min(len(validators), self.max_validators_per_round)} validators...")
        
        tasks = []
        checked = 0
        
        for uid, stake, axon, hotkey in validators[:self.max_validators_per_round]:
            checked += 1
            task = await self.pull_task_from_validator(uid, axon, hotkey)
            if task:
                tasks.append(task)
                print(f"✅ Task collected from validator {uid}")
            
            # Small delay between requests
            await asyncio.sleep(0.5)
        
        print(f"📊 Round complete: {len(tasks)} tasks found from {checked} validators")
        return tasks

    def print_status(self):
        """Print current monitoring status"""
        runtime = time.time() - self.stats["start_time"]
        hours = int(runtime // 3600)
        minutes = int((runtime % 3600) // 60)
        
        success_rate = (self.stats["successful_connections"] / 
                       max(self.stats["total_requests"], 1)) * 100
        
        print(f"\n📊 MONITORING STATUS")
        print("=" * 50)
        print(f"⏱️ Runtime: {hours:02d}:{minutes:02d}")
        print(f"🔄 Rounds completed: {self.stats['total_rounds']}")
        print(f"📡 Total requests: {self.stats['total_requests']}")
        print(f"🎯 Tasks found: {self.stats['tasks_found']}")
        print(f"✅ Connection success rate: {success_rate:.1f}%")
        print(f"❌ Connection errors: {self.stats['connection_errors']}")
        
        if self.stats["last_task_time"] > 0:
            last_task_ago = time.time() - self.stats["last_task_time"]
            print(f"🕐 Last task: {last_task_ago/60:.1f} minutes ago")
        else:
            print(f"🕐 Last task: Never")
        
        # Show validator cooldown status
        current_time = time.time()
        on_cooldown = 0
        for uid, last_request in self.stats["validator_cooldowns"].items():
            if current_time - last_request < self.cooldown_period:
                on_cooldown += 1
        
        print(f"❄️ Validators on cooldown: {on_cooldown}")
        print()

    async def run_continuous_monitoring(self):
        """Run continuous monitoring loop"""
        print("🚀 Starting continuous monitoring...")
        print("Press Ctrl+C to stop")
        print("=" * 50)
        
        # Initialize
        if not await self.initialize_bittensor():
            print("❌ Failed to initialize Bittensor")
            return
        
        try:
            while True:
                self.stats["total_rounds"] += 1
                
                print(f"\n🔄 Round {self.stats['total_rounds']} - {datetime.now().strftime('%H:%M:%S')}")
                
                # Check for tasks
                tasks = await self.check_validators_for_tasks()
                
                # Process any found tasks
                if tasks:
                    print(f"\n🎉 FOUND {len(tasks)} TASK(S)!")
                    for i, task in enumerate(tasks, 1):
                        print(f"\nTask {i}:")
                        print(f"  ID: {task['task_id']}")
                        print(f"  Validator: {task['validator_uid']}")
                        print(f"  Prompt: '{task['prompt']}'")
                        print(f"  Threshold: {task['validation_threshold']:.2f}")
                        
                        # Here you would normally process the task
                        # For now, we'll just log it
                        print(f"  📝 Task logged for processing")
                
                # Print status every 5 rounds
                if self.stats["total_rounds"] % 5 == 0:
                    self.print_status()
                
                # Wait before next round
                print(f"⏳ Waiting {self.check_interval}s until next round...")
                await asyncio.sleep(self.check_interval)
                
        except KeyboardInterrupt:
            print("\n🛑 Monitoring stopped by user")
        except Exception as e:
            print(f"\n💥 Monitoring error: {e}")
            traceback.print_exc()
        finally:
            self.print_status()
            print("🏁 Monitoring session ended")


async def main():
    """Main function"""
    monitor = ContinuousTaskMonitor()
    await monitor.run_continuous_monitoring()


if __name__ == "__main__":
    asyncio.run(main()) 