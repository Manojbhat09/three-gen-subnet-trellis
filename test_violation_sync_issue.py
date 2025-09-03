#!/usr/bin/env python3
"""
Test to demonstrate and fix the violation sync issue
Shows that validator reports violations but our local tracking doesn't sync
"""

import time
from dataclasses import dataclass
from typing import Dict, Optional

@dataclass
class MockValidatorState:
    """Mock validator state"""
    uid: int
    validator_reported_violations: int = 0

@dataclass
class MockPullResponse:
    """Mock pull response showing validator's violation count"""
    status_code: int
    cooldown_violations: Optional[int] = None
    success: bool = True

class ViolationSyncTest:
    """Test violation synchronization between validator reports and local tracking"""

    def __init__(self):
        self.validators: Dict[int, MockValidatorState] = {}

    def add_validator(self, uid: int, initial_violations: int = 0):
        """Add validator with initial violation count"""
        self.validators[uid] = MockValidatorState(uid, initial_violations)
        print(f"➕ Added validator UID {uid} with {initial_violations} violations")

    def simulate_validator_response(self, uid: int, reported_violations: int) -> MockPullResponse:
        """Simulate validator reporting violation count"""
        return MockPullResponse(
            status_code=200,
            cooldown_violations=reported_violations,
            success=True
        )

    def test_current_logic(self):
        """Test current logic (BROKEN - doesn't sync violations)"""
        print("\n🔍 TESTING CURRENT LOGIC (BROKEN)")
        print("=" * 50)

        # Setup
        self.add_validator(142, 0)  # We start with 0
        validator = self.validators[142]

        print(f"Initial local violations: {validator.validator_reported_violations}")

        # Validator reports 163 violations (from log)
        response = self.simulate_validator_response(142, 163)
        print(f"Validator reports violations: {response.cooldown_violations}")

        # CURRENT BROKEN LOGIC: We don't sync the violation count!
        if hasattr(response, 'cooldown_violations'):
            reported = response.cooldown_violations
            current = validator.validator_reported_violations
            print(f"❌ BROKEN: We see validator has {reported} violations but keep local count at {current}")
            print("❌ BROKEN: No sync happens!")

        print(f"Final local violations: {validator.validator_reported_violations} (WRONG!)")

    def test_fixed_logic(self):
        """Test fixed logic (syncs violations properly)"""
        print("\n✅ TESTING FIXED LOGIC")
        print("=" * 50)

        # Setup
        self.add_validator(142, 0)  # We start with 0
        validator = self.validators[142]

        print(f"Initial local violations: {validator.validator_reported_violations}")

        # Validator reports 163 violations (from log)
        response = self.simulate_validator_response(142, 163)
        print(f"Validator reports violations: {response.cooldown_violations}")

        # FIXED LOGIC: Sync violation count from validator response
        if hasattr(response, 'cooldown_violations') and response.cooldown_violations is not None:
            old_count = validator.validator_reported_violations
            new_count = response.cooldown_violations
            validator.validator_reported_violations = new_count
            print(f"✅ FIXED: Synced local violations from {old_count} → {new_count}")

        print(f"Final local violations: {validator.validator_reported_violations} (CORRECT!)")

    def test_multiple_validators_scenario(self):
        """Test multiple validators with different violation counts"""
        print("\n🔄 TESTING MULTIPLE VALIDATORS SCENARIO")
        print("=" * 50)

        # Add validators with their actual violation counts from log
        self.add_validator(128, 0)  # Our local starts at 0
        self.add_validator(212, 0)  # Our local starts at 0

        print("\nSimulating validator responses:")

        # UID 128 reports 199 violations
        response128 = self.simulate_validator_response(128, 199)
        if hasattr(response128, 'cooldown_violations') and response128.cooldown_violations is not None:
            self.validators[128].validator_reported_violations = response128.cooldown_violations
            print(f"✅ UID 128: Synced {response128.cooldown_violations} violations")

        # UID 212 reports 163 violations
        response212 = self.simulate_validator_response(212, 163)
        if hasattr(response212, 'cooldown_violations') and response212.cooldown_violations is not None:
            self.validators[212].validator_reported_violations = response212.cooldown_violations
            print(f"✅ UID 212: Synced {response212.cooldown_violations} violations")

        print("\nFinal violation counts:")
        for uid, validator in self.validators.items():
            print(f"  UID {uid}: {validator.validator_reported_violations} violations")

    def demonstrate_real_world_impact(self):
        """Show the real-world impact of this sync issue"""
        print("\n🚨 REAL-WORLD IMPACT OF SYNC ISSUE")
        print("=" * 50)

        print("❌ WITHOUT FIX:")
        print("  - Validator has 163 violations, we think it has 0")
        print("  - We might pull from it when we shouldn't")
        print("  - Validator penalizes us for violations we didn't know about")
        print("  - Cooldown violations keep accumulating")

        print("\n✅ WITH FIX:")
        print("  - We sync to validator's actual violation count (163)")
        print("  - We respect the validator's cooldown state")
        print("  - No unexpected violations")
        print("  - Proper cooldown management")

def run_all_tests():
    """Run all violation sync tests"""
    print("🧪 VIOLATION SYNC ISSUE DEMONSTRATION")
    print("=" * 60)

    test = ViolationSyncTest()

    test.test_current_logic()
    test.test_fixed_logic()
    test.test_multiple_validators_scenario()
    test.demonstrate_real_world_impact()

    print("\n" + "=" * 60)
    print("🎯 CONCLUSION:")
    print("The issue is that we're not syncing violation counts from validator responses!")
    print("We need to update validator.validator_reported_violations = response.cooldown_violations")
    print("when the validator reports violations in the pull/submit response.")

if __name__ == "__main__":
    run_all_tests()



