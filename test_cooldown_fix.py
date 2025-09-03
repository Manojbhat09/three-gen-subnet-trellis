#!/usr/bin/env python3
"""
Test script to verify the cooldown fix prevents violations
"""

import time
import sys
import os

# Add the current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from continuous_trellis_orchestrator_working_a6000 import ValidatorState

def test_cooldown_fix():
    """Test that the cooldown fix prevents violations"""
    print("🧪 Testing cooldown fix...")
    
    # Create a validator with an active cooldown
    validator = ValidatorState(uid=212, hotkey="test_hotkey", stake=1000.0, trust=0.8, consensus=0.9)
    
    # Set a cooldown that should be active for 10 seconds
    current_time = time.time()
    validator.validator_enforced_cooldown_until = current_time + 10
    
    print(f"✅ Created validator UID {validator.uid}")
    print(f"   Cooldown until: {validator.validator_enforced_cooldown_until}")
    print(f"   Current time: {current_time}")
    print(f"   Cooldown remaining: {validator.validator_enforced_cooldown_until - current_time:.1f}s")
    
    # Test 1: Check if validator is on cooldown
    from continuous_trellis_orchestrator_working_a6000 import ContinuousTrellisOrchestrator
    
    # Create a minimal config for testing
    config = {
        'output_dir': '/tmp/test_output',
        'generation_server_url': 'http://localhost:8000',
        'validation_server_url': 'http://localhost:8001',
        'min_local_score': 0.5,
        'enable_prompt_optimization': False,
        'enable_reproducibility_optimization': False,
        'use_fixed_seed': True,
        'fixed_seed_value': 42
    }
    
    orchestrator = ContinuousTrellisOrchestrator(config)
    
    # Test the cooldown check
    is_cooldown, cooldown_type, remaining = orchestrator._is_validator_on_cooldown(validator)
    
    print(f"\n🔍 Cooldown check results:")
    print(f"   Is on cooldown: {is_cooldown}")
    print(f"   Cooldown type: {cooldown_type}")
    print(f"   Remaining: {remaining:.1f}s")
    
    if is_cooldown and cooldown_type == "validator" and remaining > 0:
        print("✅ SUCCESS: Cooldown is properly detected and active")
    else:
        print("❌ FAILURE: Cooldown not properly detected")
        return False
    
    # Test 2: Check if validator is available (should return False)
    is_available = orchestrator.is_validator_available(validator)
    
    print(f"\n🔍 Availability check results:")
    print(f"   Is available: {is_available}")
    
    if not is_available:
        print("✅ SUCCESS: Validator correctly marked as unavailable during cooldown")
    else:
        print("❌ FAILURE: Validator incorrectly marked as available during cooldown")
        return False
    
    # Test 3: Wait for cooldown to expire and check again
    print(f"\n⏳ Waiting for cooldown to expire...")
    time.sleep(1)  # Wait 1 second
    
    is_cooldown_after, cooldown_type_after, remaining_after = orchestrator._is_validator_on_cooldown(validator)
    is_available_after = orchestrator.is_validator_available(validator)
    
    print(f"🔍 After 1 second:")
    print(f"   Is on cooldown: {is_cooldown_after}")
    print(f"   Remaining: {remaining_after:.1f}s")
    print(f"   Is available: {is_available_after}")
    
    if is_cooldown_after and remaining_after < 10:
        print("✅ SUCCESS: Cooldown countdown working correctly")
    else:
        print("❌ FAILURE: Cooldown countdown not working")
        return False
    
    print("\n🎉 ALL TESTS PASSED! The cooldown fix is working correctly.")
    return True

if __name__ == "__main__":
    try:
        success = test_cooldown_fix()
        if success:
            print("\n✅ COOLDOWN FIX VERIFIED - No more violations should occur!")
        else:
            print("\n❌ COOLDOWN FIX FAILED - Issues still exist!")
            sys.exit(1)
    except Exception as e:
        print(f"\n💥 Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
