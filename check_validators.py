#!/usr/bin/env python3
"""
Check Validators for Available Tasks
"""

import bittensor as bt
import asyncio
from subnet_protocol_integration import MockPullTask as PullTask

async def check_validators_for_tasks():
    wallet = bt.wallet(name='manbeast', hotkey='beastman')
    subtensor = bt.subtensor()
    dendrite = bt.dendrite(wallet=wallet)
    metagraph = subtensor.metagraph(netuid=17)
    
    print(f'🔍 Checking {len(metagraph.hotkeys)} neurons for active validators...')
    
    # Find all active validators (broader criteria)
    active_validators = []
    for uid, (stake, axon, hotkey) in enumerate(zip(metagraph.stake, metagraph.axons, metagraph.hotkeys)):
        if (stake > 100 and axon.is_serving and axon.ip != '0.0.0.0'):  # Lower threshold
            active_validators.append((uid, stake, axon, hotkey))
    
    # Sort by stake
    active_validators.sort(key=lambda x: x[1], reverse=True)
    print(f'Found {len(active_validators)} active validators')
    
    print('\n📡 Top 15 Active Validators:')
    for i, (uid, stake, axon, hotkey) in enumerate(active_validators[:15]):
        print(f'{i+1:2d}. Validator {uid}: {hotkey[:10]}... stake: {stake:.1f} at {axon.ip}:{axon.port}')
    
    print('\n🎯 Trying to pull tasks from validators...')
    tasks_found = 0
    
    # Try to pull from top 15 validators
    for i, (uid, stake, axon, hotkey) in enumerate(active_validators[:15]):
        try:
            print(f'\n📡 Trying validator {uid} ({hotkey[:10]}...)')
            
            pull_synapse = PullTask()
            
            response = await dendrite.call(
                target_axon=axon,
                synapse=pull_synapse,
                deserialize=False,
                timeout=15.0
            )
            
            if response and hasattr(response, 'task') and response.task:
                print(f'  ✅ TASK FOUND!')
                print(f'     Task ID: {response.task.id}')
                print(f'     Prompt: "{response.task.prompt}"')
                print(f'     Validation threshold: {response.validation_threshold}')
                print(f'     Cooldown until: {response.cooldown_until}')
                tasks_found += 1
            elif response and hasattr(response, 'cooldown_until') and response.cooldown_until > 0:
                print(f'  ⏳ Cooldown until: {response.cooldown_until}')
            else:
                print(f'  ❌ No task available')
                
        except Exception as e:
            print(f'  💥 Error: {str(e)[:60]}...')
    
    print(f'\n📊 Summary: Found {tasks_found} available tasks from {len(active_validators)} validators')
    
    if tasks_found == 0:
        print('\n💡 Possible reasons for no tasks:')
        print('   - Validators currently have no tasks queued')
        print('   - Miner is on cooldown period')
        print('   - Tasks are distributed to specific miners')
        print('   - Peak hours may have more task availability')

if __name__ == "__main__":
    asyncio.run(check_validators_for_tasks()) 