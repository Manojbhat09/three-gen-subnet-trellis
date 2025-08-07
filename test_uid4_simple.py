#!/usr/bin/env python3
"""
Simple test for UID 4
"""

import asyncio

async def test_uid4():
    print("🚀 Testing UID 4...")
    
    try:
        import bittensor as bt
        print("✅ Bittensor imported")
        
        # Setup
        wallet = bt.wallet(name="manbeast3b", hotkey="m3b")
        print(f"✅ Wallet: {wallet.hotkey.ss58_address}")
        
        subtensor = bt.subtensor(network="test")
        print("✅ Connected to testnet")
        
        metagraph = subtensor.metagraph(17)
        print(f"✅ Metagraph loaded: {len(metagraph.neurons)} neurons")
        
        # Check UID 4
        target_uid = 4
        if target_uid >= len(metagraph.neurons):
            print(f"❌ UID {target_uid} does not exist!")
            return
        
        neuron = metagraph.neurons[target_uid]
        print(f"✅ UID {target_uid} exists")
        print(f"   validator_permit: {neuron.validator_permit}")
        print(f"   stake: {float(neuron.stake):.1f} TAO")
        print(f"   axon_info: {neuron.axon_info}")
        
        if not neuron.validator_permit:
            print(f"❌ UID {target_uid} is not a validator!")
            return
        
        # Try to pull task
        print("📡 Attempting to pull task...")
        
        dendrite = bt.dendrite(wallet=wallet)
        
        try:
            from neurons.common.protocol import PullTask
            print("✅ PullTask protocol imported")
            
            synapse = PullTask()
            synapse.timeout = 30
            
            print("📤 Sending request...")
            response = await dendrite.forward(
                axons=[neuron.axon_info],
                synapse=synapse,
                timeout=30
            )
            print(f"📥 Response received: {response}")
            
            if response and len(response) > 0:
                resp = response[0]
                print(f"📋 Response type: {type(resp)}")
                print(f"📋 Response attributes: {dir(resp)}")
                
                if hasattr(resp, 'task') and resp.task:
                    print(f"✅ Task received!")
                    print(f"   Task ID: {resp.task.id}")
                    print(f"   Prompt: '{resp.task.prompt}'")
                else:
                    print("⚠️ No task in response")
                    print(f"   Response: {resp}")
            else:
                print("❌ No response")
                
        except ImportError as e:
            print(f"❌ Import error: {e}")
        except Exception as e:
            print(f"❌ Error pulling task: {e}")
            import traceback
            traceback.print_exc()
    
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_uid4()) 