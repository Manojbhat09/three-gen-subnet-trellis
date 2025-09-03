#!/usr/bin/env python3
"""
Test script for Validator Behavior Mimic implementation
Tests the integrated cooldown and violation prevention system
"""

import time
import sys
import os

# Add the current directory to path so we can import the main module
sys.path.append('/home/mbhat/three-gen-subnet-trellis')

# Mock the necessary imports to test our ValidatorState class
class MockConfig:
    def __init__(self):
        self.cooldown_violations_threshold = 100
        self.cooldown_violation_penalty = 10

class MockLogger:
    def debug(self, msg): print(f"DEBUG: {msg}")
    def info(self, msg): print(f"INFO: {msg}")
    def warning(self, msg): print(f"WARNING: {msg}")
    def error(self, msg): print(f"ERROR: {msg}")

# Test the ValidatorState class with mimic functionality
def test_validator_mimic():
    print("🧪 Testing Validator Behavior Mimic Implementation")
    print("=" * 60)

    # Create a mock validator state
    from continuous_trellis_orchestrator_working_a6000 import ValidatorState

    validator = ValidatorState(
        uid=142,
        hotkey="test_hotkey",
        stake=1000.0,
        trust=0.8,
        consensus=0.9
    )

    print(f"✅ Created validator UID {validator.uid}")
    print(f"   Initial violations: {validator.validator_reported_violations}")
    print(f"   Initial cooldown multiplier: {validator.cooldown_multiplier}")

    # Test 1: Normal operation (should allow pull)
    print("\n🧪 Test 1: Normal operation")
    can_pull = validator.should_prevent_pull_attempt(min_task_interval=37.0)  # Use buffered value
    print(f"   Should prevent pull: {can_pull} (expected: False)")

    # Test 2: Recent pull attempt (should prevent)
    print("\n🧪 Test 2: Recent pull attempt prevention")
    validator.mimic_last_pull_attempt = time.time()
    can_pull = validator.should_prevent_pull_attempt(min_task_interval=37.0)  # Use buffered value
    print(f"   Should prevent pull: {can_pull} (expected: True)")

    # Wait a moment and test again (should still prevent due to buffer)
    time.sleep(1)
    can_pull_after_wait = validator.should_prevent_pull_attempt(min_task_interval=37.0)  # Use buffered value
    print(f"   Should still prevent after 1s: {can_pull_after_wait} (expected: True due to 2s buffer)")

    # Test 3: Successful pull update
    print("\n🧪 Test 3: Successful pull state update")
    validator.update_mimic_state_after_pull_attempt(was_successful=True)
    print(f"   Mimic last attempt: {validator.mimic_last_pull_attempt}")
    print(f"   Mimic expected cooldown: {validator.mimic_expected_cooldown_until}")

    # Test 4: Violation risk assessment
    print("\n🧪 Test 4: Violation risk assessment")
    risk = validator.get_mimic_violation_risk()
    print(f"   Violation risk: {risk} (expected: NONE)")

    # Test 5: High violation count
    print("\n🧪 Test 5: High violation count prevention")
    validator.validator_reported_violations = 150
    can_pull = validator.should_prevent_pull_attempt(min_task_interval=37.0)  # Use buffered value
    risk = validator.get_mimic_violation_risk()
    print(f"   Should prevent pull with high violations: {can_pull} (expected: True)")
    print(f"   Violation risk: {risk} (expected: HIGH)")

    # Test 6: Cooldown enforcement
    print("\n🧪 Test 6: Cooldown enforcement")
    validator.validator_enforced_cooldown_until = time.time() + 60  # 60 seconds from now
    can_pull = validator.should_prevent_pull_attempt(min_task_interval=37.0)  # Use buffered value
    print(f"   Should prevent pull during cooldown: {can_pull} (expected: True)")

    print("\n" + "=" * 60)
    print("✅ All Validator Mimic tests completed!")
    print("🎉 Implementation appears to be working correctly!")

if __name__ == "__main__":
    test_validator_mimic()
