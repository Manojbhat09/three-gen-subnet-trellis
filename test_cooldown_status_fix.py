#!/usr/bin/env python3
"""
Test script to verify the cooldown status fix prevents violations
"""

import time
import sys
import os

# Add the current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from continuous_trellis_orchestrator_working_a6000 import ValidatorState

def test_cooldown_status_fix():
    """Test that the cooldown status fix prevents violations"""
    print("🧪 Testing cooldown status fix...")

    # Create a validator with an active cooldown
    validator = ValidatorState(uid=199, hotkey="test_hotkey", stake=1000.0, trust=0.8, consensus=0.9)

    # Set a cooldown that should be active for 10 seconds
    current_time = time.time()
    validator.validator_enforced_cooldown_until = current_time + 10

    print(f"✅ Created validator UID {validator.uid}")
    print(f"   Cooldown until: {validator.validator_enforced_cooldown_until}")
    print(f"   Current time: {current_time}")
    print(f"   Cooldown remaining: {validator.validator_enforced_cooldown_until - current_time:.1f}s")

    # Create a minimal orchestrator instance for testing
    from continuous_trellis_orchestrator_working_a6000 import ContinuousTrellisOrchestrator

    # Create a minimal config for testing
    class MockConfig:
        def get(self, key, default=None):
            return default

    orchestrator = ContinuousTrellisOrchestrator()
    orchestrator.config = MockConfig()
    orchestrator.logger = type('MockLogger', (), {'info': lambda x: None, 'debug': lambda x: None})()

    # Test 1: Check cooldown status
    cooldown_status = orchestrator._check_validator_cooldown_state(validator)

    print("\n🔍 Cooldown status check results:")
    print(f"   Available: {cooldown_status['available']}")
    print(f"   Reason: {cooldown_status['reason']}")
    print(f"   Remaining: {cooldown_status['remaining_time']:.1f}s")
    print(f"   Type: {cooldown_status['cooldown_type']}")

    if not cooldown_status['available'] and cooldown_status['cooldown_type'] == 'validator_enforced' and cooldown_status['remaining_time'] > 0:
        print("✅ SUCCESS: Cooldown status correctly detected and blocking")
    else:
        print("❌ FAILURE: Cooldown status not properly detected")
        return False

    # Test 2: Wait for cooldown to expire and check again
    print("\n⏳ Waiting for cooldown to expire...")
    time.sleep(1)  # Wait 1 second

    cooldown_status_after = orchestrator._check_validator_cooldown_state(validator)

    print("🔍 After 1 second:")
    print(f"   Available: {cooldown_status_after['available']}")
    print(f"   Remaining: {cooldown_status_after['remaining_time']:.1f}s")

    if cooldown_status_after['available'] and cooldown_status_after['remaining_time'] == 0:
        print("✅ SUCCESS: Cooldown expired and was cleared correctly")
    else:
        print("❌ FAILURE: Cooldown expiration handling not working")
        return False

    # Test 3: Verify cooldown was actually cleared
    print("\n🔍 Verifying cooldown was cleared:")
    print(f"   validator_enforced_cooldown_until: {validator.validator_enforced_cooldown_until}")
    print(f"   pending_cooldown_task_id: {validator.pending_cooldown_task_id}")

    if validator.validator_enforced_cooldown_until is None and validator.pending_cooldown_task_id is None:
        print("✅ SUCCESS: Cooldown fields were properly cleared")
    else:
        print("❌ FAILURE: Cooldown fields were not cleared")
        return False

    print("\n🎉 ALL TESTS PASSED! The cooldown status fix is working correctly.")
    return True

if __name__ == "__main__":
    try:
        success = test_cooldown_status_fix()
        if success:
            print("\n✅ COOLDOWN STATUS FIX VERIFIED - Should prevent violations!")
        else:
            print("\n❌ COOLDOWN STATUS FIX FAILED - Issues still exist!")
            sys.exit(1)
    except Exception as e:
        print(f"\n💥 Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
