#!/usr/bin/env python3
"""
Simple Task Pull Test Script
Tests if we can pull tasks from validators using Bittensor wallet and hotkey.
"""

import asyncio
import bittensor as bt
import time
import json
from typing import List, Dict, Any, Optional

class SimpleTaskPuller:
    def __init__(self, wallet_name: str = "test2m3b2", hotkey_name: str = "t2m3b2", netuid: int = 17):
        self.wallet_name = wallet_name
        self.hotkey_name = hotkey_name
        self.netuid = netuid
        
        # Bittensor components
        self.wallet = None
        self.subtensor = None
        self.dendrite = None
        self.metagraph = None
        
        print(f"🔧 Initializing Simple Task Puller")
        print(f"   Wallet: {wallet_name}")
        print(f"   Hotkey: {hotkey_name}")
        print(f"   NetUID: {netuid}")
    
    def setup_bittensor(self) -> bool:
        """Setup Bittensor components"""
        try:
            print("🔧 Setting up Bittensor components...")
            
            # Setup wallet
            self.wallet = bt.wallet(
                name=self.wallet_name,
                hotkey=self.hotkey_name
            )
            print(f"✅ Wallet loaded: {self.wallet.hotkey.ss58_address}")
            
            # Setup subtensor
            self.subtensor = bt.subtensor(network="finney")
            print("✅ Subtensor connected")
            
            # Setup dendrite
            self.dendrite = bt.dendrite(wallet=self.wallet)
            print("✅ Dendrite initialized")
            
            # Setup metagraph
            self.metagraph = self.subtensor.metagraph(self.netuid)
            print(f"✅ Metagraph loaded (netuid: {self.netuid})")
            
            return True
            
        except Exception as e:
            print(f"❌ Bittensor setup failed: {e}")
            return False
    
    def get_eligible_validators(self, min_stake: float = 1000.0, min_trust: float = 0.0, max_validators: int = 10) -> List[Dict[str, Any]]:
        """Get list of eligible validators"""
        try:
            print(f"🔍 Scanning for eligible validators...")
            print(f"   Min stake: {min_stake} TAO")
            print(f"   Min trust: {min_trust}")
            print(f"   Max validators: {max_validators}")
            
            eligible_validators = []
            
            for uid, neuron in enumerate(self.metagraph.neurons):
                # Check if this is a valid validator
                if not neuron.validator_permit:
                    continue
                
                stake = float(neuron.stake)
                trust = float(neuron.trust)
                consensus = float(neuron.consensus)
                
                # Apply filtering criteria
                if stake < min_stake:
                    continue
                
                if trust < min_trust:
                    continue
                
                eligible_validators.append({
                    'uid': uid,
                    'stake': stake,
                    'trust': trust,
                    'consensus': consensus,
                    'hotkey': neuron.hotkey,
                    'score': stake * trust * consensus
                })
            
            # Sort by score and take top validators
            eligible_validators.sort(key=lambda x: x['score'], reverse=True)
            eligible_validators = eligible_validators[:max_validators]
            
            print(f"✅ Found {len(eligible_validators)} eligible validators")
            for i, validator in enumerate(eligible_validators, 1):
                print(f"   {i:2d}. UID {validator['uid']:3d}: {validator['stake']:8.1f} TAO, trust: {validator['trust']:.3f}")
            
            return eligible_validators
            
        except Exception as e:
            print(f"❌ Failed to get eligible validators: {e}")
            return []
    
    async def pull_task_from_validator(self, validator: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Pull task from a specific validator"""
        try:
            uid = validator['uid']
            print(f"📡 Attempting to pull task from UID {uid}...")
            
            # Import protocol
            from neurons.common.protocol import PullTask
            
            # Create task pull request
            synapse = PullTask()
            synapse.timeout = 60  # 60 second timeout
            
            # Get neuron info
            if uid >= len(self.metagraph.neurons):
                print(f"❌ UID {uid} not found in metagraph")
                return None
            
            neuron = self.metagraph.neurons[uid]
            
            start_time = time.time()
            
            # Query the validator
            response = await self.dendrite.forward(
                axons=[neuron.axon_info],
                synapse=synapse,
                timeout=60
            )
            
            query_time = time.time() - start_time
            
            if response and len(response) > 0:
                resp = response[0]
                
                if hasattr(resp, 'task') and resp.task and resp.task.prompt:
                    print(f"✅ SUCCESS! Got task from UID {uid}")
                    print(f"   Task ID: {resp.task.id}")
                    print(f"   Prompt: '{resp.task.prompt}'")
                    print(f"   Query time: {query_time:.2f}s")
                    
                    # Check for additional fields
                    if hasattr(resp, 'validation_threshold'):
                        print(f"   Validation threshold: {resp.validation_threshold}")
                    if hasattr(resp, 'cooldown_until'):
                        print(f"   Cooldown until: {resp.cooldown_until}")
                    
                    return {
                        'task_id': resp.task.id,
                        'prompt': resp.task.prompt,
                        'validator_uid': uid,
                        'query_time': query_time,
                        'validation_threshold': getattr(resp, 'validation_threshold', None),
                        'cooldown_until': getattr(resp, 'cooldown_until', None)
                    }
                else:
                    print(f"⚠️ No task received from UID {uid}")
                    return None
            else:
                print(f"❌ No response from UID {uid}")
                return None
        
        except Exception as e:
            print(f"❌ Error pulling from UID {uid}: {e}")
            return None
    
    async def test_task_pulling(self, num_validators_to_test: int = 3):
        """Test task pulling from multiple validators"""
        print(f"\n🚀 Starting task pulling test...")
        print(f"   Testing {num_validators_to_test} validators")
        
        # Get eligible validators
        validators = self.get_eligible_validators(max_validators=num_validators_to_test)
        
        if not validators:
            print("❌ No eligible validators found")
            return
        
        # Test each validator
        successful_pulls = 0
        total_tested = 0
        
        for i, validator in enumerate(validators, 1):
            print(f"\n--- Testing Validator {i}/{len(validators)} ---")
            
            total_tested += 1
            result = await self.pull_task_from_validator(validator)
            
            if result:
                successful_pulls += 1
                print(f"✅ Task pull SUCCESSFUL from UID {validator['uid']}")
            else:
                print(f"❌ Task pull FAILED from UID {validator['uid']}")
            
            # Small delay between pulls
            if i < len(validators):
                print("   Waiting 2 seconds before next validator...")
                await asyncio.sleep(2)
        
        # Summary
        print(f"\n📊 TEST SUMMARY")
        print(f"=" * 40)
        print(f"Total validators tested: {total_tested}")
        print(f"Successful task pulls: {successful_pulls}")
        print(f"Failed task pulls: {total_tested - successful_pulls}")
        print(f"Success rate: {(successful_pulls / total_tested * 100):.1f}%")
        
        if successful_pulls > 0:
            print(f"✅ Task pulling is WORKING! You can successfully pull tasks.")
        else:
            print(f"❌ Task pulling is NOT working. Check your setup and network connection.")

async def main():
    """Main function"""
    print("🧪 SIMPLE TASK PULL TEST")
    print("=" * 50)
    
    # Create task puller
    puller = SimpleTaskPuller(
        wallet_name="test2m3b2",  # Change this to your wallet name
        hotkey_name="t2m3b2",     # Change this to your hotkey name
        netuid=17                 # Subnet 17 for 404-GEN
    )
    
    # Setup Bittensor
    if not puller.setup_bittensor():
        print("❌ Failed to setup Bittensor. Exiting.")
        return
    
    # Test task pulling
    await puller.test_task_pulling(num_validators_to_test=3)
    
    print(f"\n🏁 Test completed!")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n🛑 Test interrupted by user")
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
