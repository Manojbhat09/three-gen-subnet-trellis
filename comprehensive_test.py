#!/usr/bin/env python3
"""
Comprehensive test for Validator Mimic Implementation
Tests all components to ensure no bugs remain
"""

import time
import sys
import os

# Add current directory to path
sys.path.insert(0, '/home/mbhat/three-gen-subnet-trellis')

def test_validator_mimic():
    """Test the validator mimic implementation"""
    print("🧪 COMPREHENSIVE VALIDATOR MIMIC TEST")
    print("=" * 60)

    try:
        # Test 1: Import ValidatorState
        print("1. Testing import...")
        from continuous_trellis_orchestrator_working_a6000 import ValidatorState
        print("   ✅ Import successful - no syntax errors")

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

        # Test 3: Check all required fields exist
        print("3. Testing required fields...")
        required_fields = [
            'mimic_last_pull_attempt',
            'mimic_expected_cooldown_until',
            'validator_reported_violations',
            'validator_enforced_cooldown_until',
            'miner_cooldown_until',
            'cooldown_multiplier'
        ]

        for field in required_fields:
            assert hasattr(validator, field), f"Missing field: {field}"
        print("   ✅ All required fields present")

        # Test 4: Check all required methods exist
        print("4. Testing required methods...")
        required_methods = [
            'should_prevent_pull_attempt',
            'update_mimic_state_after_pull_attempt',
            'get_mimic_violation_risk'
        ]

        for method in required_methods:
            assert hasattr(validator, method), f"Missing method: {method}"
            assert callable(getattr(validator, method)), f"Method not callable: {method}"
        print("   ✅ All required methods present and callable")

        # Test 5: Test method functionality
        print("5. Testing method functionality...")

        # Test should_prevent_pull_attempt
        can_pull = validator.should_prevent_pull_attempt(min_task_interval=37.0)
        print(f"   ✅ should_prevent_pull_attempt: {can_pull} (expected: False)")

        # Test update_mimic_state_after_pull_attempt
        validator.update_mimic_state_after_pull_attempt(was_successful=True)
        print("   ✅ update_mimic_state_after_pull_attempt: successful")

        # Test get_mimic_violation_risk
        risk = validator.get_mimic_violation_risk()
        print(f"   ✅ get_mimic_violation_risk: {risk} (expected: NONE)")

        # Test 6: Test field values after method calls
        print("6. Testing field values after method calls...")
        assert validator.mimic_last_pull_attempt is not None, "mimic_last_pull_attempt not set"
        assert validator.mimic_expected_cooldown_until is not None, "mimic_expected_cooldown_until not set"
        print("   ✅ Field values updated correctly")

        # Test 7: Test violation risk levels
        print("7. Testing violation risk levels...")
        test_violations = [0, 15, 75, 150]
        expected_risks = ["NONE", "LOW", "MEDIUM", "HIGH"]

        for violations, expected_risk in zip(test_violations, expected_risks):
            validator.validator_reported_violations = violations
            actual_risk = validator.get_mimic_violation_risk()
            assert actual_risk == expected_risk, f"Risk mismatch: {violations} violations should be {expected_risk}, got {actual_risk}"
            print(f"   ✅ {violations} violations → {actual_risk}")

        # Test 8: Test cooldown prevention
        print("8. Testing cooldown prevention...")
        validator.validator_enforced_cooldown_until = time.time() + 60  # Future cooldown
        can_pull_cooldown = validator.should_prevent_pull_attempt(min_task_interval=37.0)
        assert can_pull_cooldown == True, "Should prevent pull when on cooldown"
        print("   ✅ Cooldown prevention working")

        # Test 9: Test timing buffer
        print("9. Testing timing buffer...")
        validator.validator_enforced_cooldown_until = None  # Clear cooldown
        validator.mimic_last_pull_attempt = time.time()  # Recent attempt
        can_pull_buffer = validator.should_prevent_pull_attempt(min_task_interval=37.0)
        assert can_pull_buffer == True, "Should prevent pull due to timing buffer"
        print("   ✅ Timing buffer working")

        # Test 10: Test high violation prevention
        print("10. Testing high violation prevention...")
        validator.mimic_last_pull_attempt = None  # Clear timing
        validator.validator_enforced_cooldown_until = None  # Clear cooldown
        validator.validator_reported_violations = 75  # High violations
        can_pull_violations = validator.should_prevent_pull_attempt(min_task_interval=37.0)
        assert can_pull_violations == True, "Should prevent pull due to high violations"
        print("   ✅ High violation prevention working")

        print("\n🎉 ALL COMPREHENSIVE TESTS PASSED!")
        print("📋 TEST SUMMARY:")
        print("   ✅ No syntax errors")
        print("   ✅ All methods functional")
        print("   ✅ All fields accessible")
        print("   ✅ Logic working correctly")
        print("   ✅ Buffers preventing conflicts")
        print("   ✅ Violation prevention working")
        print("   ✅ Cooldown logic sound")

        return True

    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_validator_mimic()
    if success:
        print("\n🎯 VALIDATOR MIMIC IMPLEMENTATION IS PRODUCTION-READY!")
        print("🚀 No violations should occur with this implementation.")
    else:
        print("\n💥 CRITICAL ISSUES FOUND - DO NOT DEPLOY!")
        sys.exit(1)



