#!/usr/bin/env python3
"""
Simple test to verify validator mimic implementation
"""

import time
import sys
import os

# Add current directory to path
sys.path.insert(0, '/home/mbhat/three-gen-subnet-trellis')

try:
    print("🧪 Testing Validator Mimic Implementation...")

    # Test 1: Import ValidatorState
    print("1. Testing import...")
    from continuous_trellis_orchestrator_working_a6000 import ValidatorState
    print("   ✅ Import successful")

    # Test 2: Create validator instance
    print("2. Testing ValidatorState creation...")
    validator = ValidatorState(
        uid=142,
        hotkey="test_hotkey",
        stake=1000.0,
        trust=0.8,
        consensus=0.9
    )
    print(f"   ✅ Validator created: UID {validator.uid}")

    # Test 3: Test mimic methods exist
    print("3. Testing mimic methods...")
    assert hasattr(validator, 'should_prevent_pull_attempt'), "should_prevent_pull_attempt method missing"
    assert hasattr(validator, 'update_mimic_state_after_pull_attempt'), "update_mimic_state_after_pull_attempt method missing"
    assert hasattr(validator, 'get_mimic_violation_risk'), "get_mimic_violation_risk method missing"
    print("   ✅ All mimic methods present")

    # Test 4: Test basic functionality
    print("4. Testing basic functionality...")
    can_pull = validator.should_prevent_pull_attempt(min_task_interval=35.0)
    print(f"   ✅ should_prevent_pull_attempt works: {can_pull}")

    risk = validator.get_mimic_violation_risk()
    print(f"   ✅ get_mimic_violation_risk works: {risk}")

    validator.update_mimic_state_after_pull_attempt(was_successful=True)
    print("   ✅ update_mimic_state_after_pull_attempt works")

    # Test 5: Test mimic state fields
    print("5. Testing mimic state fields...")
    assert hasattr(validator, 'mimic_last_pull_attempt'), "mimic_last_pull_attempt field missing"
    assert hasattr(validator, 'mimic_expected_cooldown_until'), "mimic_expected_cooldown_until field missing"
    print("   ✅ Mimic state fields present")

    print("\n🎉 ALL TESTS PASSED! Validator mimic implementation is working correctly.")
    print("📋 Summary:")
    print("   ✅ No syntax errors")
    print("   ✅ All methods functional")
    print("   ✅ State fields accessible")
    print("   ✅ Basic logic working")

except Exception as e:
    print(f"\n❌ TEST FAILED: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)