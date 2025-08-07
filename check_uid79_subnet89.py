#!/usr/bin/env python3
"""
Quick check for UID 79 on subnet 89
"""

import asyncio

async def check_uid79():
    print("🚀 Checking UID 79 on subnet 89...")
    
    try:
        import bittensor as bt
        
        # Setup
        wallet = bt.wallet(name="manbeast3b", hotkey="m3b")
        print(f"✅ Wallet: {wallet.hotkey.ss58_address}")
        
        subtensor = bt.subtensor(network="test")
        print("✅ Connected to testnet")
        
        # Load metagraph for subnet 89
        print("📊 Loading metagraph for subnet 89...")
        metagraph = subtensor.metagraph(89)
        print(f"✅ Metagraph loaded: {len(metagraph.neurons)} neurons")
        
        # Check UID 79
        target_uid = 79
        print(f"🔍 Checking UID {target_uid}...")
        
        if target_uid >= len(metagraph.neurons):
            print(f"❌ UID {target_uid} does not exist!")
            print(f"   Available UIDs: 0 to {len(metagraph.neurons)-1}")
            
            # Show some validators
            print("📋 Some validators on subnet 89:")
            for uid in range(min(10, len(metagraph.neurons))):
                neuron = metagraph.neurons[uid]
                if neuron.validator_permit:
                    print(f"   UID {uid}: {float(neuron.stake):.1f} TAO (trust: {float(neuron.trust):.3f})")
        else:
            neuron = metagraph.neurons[target_uid]
            print(f"✅ UID {target_uid} exists!")
            
            if neuron.validator_permit:
                print(f"✅ UID {target_uid} is a validator!")
                print(f"   Stake: {float(neuron.stake):.2f} TAO")
                print(f"   Trust: {float(neuron.trust):.3f}")
                print(f"   Consensus: {float(neuron.consensus):.3f}")
                
                # Try to pull a task
                print("📡 Attempting to pull task...")
                dendrite = bt.dendrite(wallet=wallet)
                
                try:
                    from neurons.common.protocol import PullTask
                    synapse = PullTask()
                    synapse.timeout = 30
                    
                    response = await dendrite.forward(
                        axons=[neuron.axon_info],
                        synapse=synapse,
                        timeout=30
                    )
                    
                    if response and len(response) > 0:
                        resp = response[0]
                        if hasattr(resp, 'task') and resp.task and resp.task.prompt:
                            print("🎉 SUCCESS! Task received!")
                            print(f"   Task ID: {resp.task.id}")
                            print(f"   Prompt: '{resp.task.prompt}'")
                        else:
                            print("⚠️ No task available")
                            print(f"   Response: {resp}")
                    else:
                        print("❌ No response from validator")
                        
                except Exception as e:
                    print(f"❌ Error pulling task: {e}")
            else:
                print(f"❌ UID {target_uid} is not a validator!")
        
        print("✅ Check completed!")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(check_uid79()) 