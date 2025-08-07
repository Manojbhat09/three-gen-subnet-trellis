#!/usr/bin/env python3
"""
Very simple test script
"""

print("🚀 Starting simple test...")

try:
    import bittensor as bt
    print("✅ Bittensor imported")
    
    # Create wallet
    wallet = bt.wallet(name="manbeast3b", hotkey="m3b")
    print(f"✅ Wallet: {wallet.hotkey.ss58_address}")
    
    # Connect to testnet
    subtensor = bt.subtensor(network="test")
    print("✅ Connected to testnet")
    
    # Load metagraph
    print("📊 Loading metagraph...")
    metagraph = subtensor.metagraph(89)
    print(f"✅ Metagraph loaded with {len(metagraph.neurons)} neurons")
    
    # Check UID 79
    target_uid = 79
    print(f"🔍 Checking UID {target_uid}...")
    
    if target_uid >= len(metagraph.neurons):
        print(f"❌ UID {target_uid} does not exist!")
        print(f"   Available UIDs: 0 to {len(metagraph.neurons)-1}")
        
        # Show some validators
        print("📋 Some validators:")
        for uid in range(min(10, len(metagraph.neurons))):
            neuron = metagraph.neurons[uid]
            if neuron.validator_permit:
                print(f"   UID {uid}: {float(neuron.stake):.1f} TAO")
    else:
        neuron = metagraph.neurons[target_uid]
        print(f"✅ UID {target_uid} exists!")
        
        if neuron.validator_permit:
            print(f"✅ UID {target_uid} is a validator!")
            print(f"   Stake: {float(neuron.stake):.2f} TAO")
            print(f"   Trust: {float(neuron.trust):.3f}")
        else:
            print(f"❌ UID {target_uid} is not a validator!")
    
    print("✅ Test completed successfully!")
    
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc() 