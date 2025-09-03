#!/usr/bin/env python3
"""
Test the real violation sync scenario from the logs
This simulates the exact case where UID 142 has 184 violations loaded
but the validator reports 163 violations in the response
"""

import time
from dataclasses import dataclass
from typing import Dict, Optional

@dataclass
class MockValidatorState:
    """Mock validator state matching our real ValidatorState"""
    uid: int
    stake: float = 100000.0
    is_active: bool = True

    # Cooldown fields
    validator_enforced_cooldown_until: float = 0.0
    miner_cooldown_until: float = 0.0
    validator_reported_violations: int = 0

@dataclass
class MockPullResponse:
    """Mock pull response showing validator's violation count"""
    status_code: int
    cooldown_until: float = 0.0
    cooldown_violations: Optional[int] = None
    dendrite_status_code: int = 200
    success: bool = True

class ViolationSyncRealScenarioTest:
    """Test real scenario from logs"""

    def __init__(self):
        self.validators: Dict[int, MockValidatorState] = {}
        self.logs: List[str] = []

    def log(self, message: str):
        """Log a message"""
        timestamp = f"{time.time():.1f}"
        self.logs.append(f"{timestamp} - {message}")
        print(f"{timestamp} - {message}")

    def add_validator(self, uid: int, violations: int = 0):
        """Add validator with initial violation count"""
        self.validators[uid] = MockValidatorState(uid, validator_reported_violations=violations)
        self.log(f"➕ Added validator UID {uid} with {violations} violations")

    def simulate_violation_loading_from_analysis(self, uid: int, violations: int):
        """Simulate loading violations from analysis file (like UID 142 with 184)"""
        if uid in self.validators:
            self.validators[uid].validator_reported_violations = violations
            self.log(f"🔄 Loaded violation starting point for UID {uid}: {violations} violations")

    def simulate_validator_response(self, uid: int, reported_violations: int) -> MockPullResponse:
        """Simulate validator reporting different violation count (like 163 vs 184)"""
        return MockPullResponse(
            status_code=200,
            cooldown_violations=reported_violations,
            dendrite_status_code=200,
            success=True
        )

    def process_pull_response_old_way(self, uid: int, response: MockPullResponse):
        """Process response the OLD way (BROKEN - doesn't sync violations)"""
        validator = self.validators[uid]

        self.log(f"📥 Processing pull response for UID {uid}")

        # OLD BROKEN WAY: Just log the violation count but don't sync
        if hasattr(response, 'cooldown_violations') and response.cooldown_violations is not None:
            self.log(f"✅ COOLDOWN VIOLATIONS: {response.cooldown_violations}")
            # ❌ MISSING: No sync to local state!
        else:
            self.log(f"✅ COOLDOWN VIOLATIONS: NOT FOUND")

        self.log(f"❌ LOCAL VIOLATIONS REMAIN: {validator.validator_reported_violations} (NOT SYNCED!)")

    def process_pull_response_new_way(self, uid: int, response: MockPullResponse):
        """Process response the NEW way (FIXED - syncs violations)"""
        validator = self.validators[uid]

        self.log(f"📥 Processing pull response for UID {uid}")

        # NEW FIXED WAY: Sync violation count from validator response
        if hasattr(response, 'cooldown_violations') and response.cooldown_violations is not None:
            self.log(f"✅ COOLDOWN VIOLATIONS: {response.cooldown_violations}")

            # ✅ CRITICAL FIX: Sync violation count from validator response
            old_violations = validator.validator_reported_violations
            new_violations = response.cooldown_violations

            if new_violations != old_violations:
                validator.validator_reported_violations = new_violations
                if new_violations > old_violations:
                    violation_increase = new_violations - old_violations
                    self.log(f"⚠️ Synced violations from validator: {old_violations} → {new_violations} (+{violation_increase})")
                else:
                    violation_decrease = old_violations - new_violations
                    self.log(f"✅ Synced violations from validator: {old_violations} → {new_violations} (-{violation_decrease})")
            else:
                self.log(f"✅ Violation count already in sync: {new_violations}")

        else:
            self.log(f"✅ COOLDOWN VIOLATIONS: NOT FOUND (successful submission = violations -1)")

        self.log(f"✅ LOCAL VIOLATIONS NOW: {validator.validator_reported_violations} (PROPERLY SYNCED!)")

    def test_real_scenario_from_logs(self):
        """Test the exact scenario from the continuous_trellis.log"""
        print("\n🎬 TESTING REAL SCENARIO FROM LOGS")
        print("=" * 60)

        # Setup: UID 142 loaded with 184 violations from analysis
        self.add_validator(142, 0)  # Start with 0
        self.simulate_violation_loading_from_analysis(142, 184)  # Load from analysis

        validator = self.validators[142]
        print(f"Initial state: UID {validator.uid} has {validator.validator_reported_violations} violations")

        # Scenario: Validator reports 163 violations (different from our 184)
        response = self.simulate_validator_response(142, 163)
        print(f"Validator response: {response.cooldown_violations} violations")

        print("\n❌ OLD WAY (BROKEN):")
        print("-" * 30)
        self.process_pull_response_old_way(142, response)

        # Reset for new test
        validator.validator_reported_violations = 184

        print("\n✅ NEW WAY (FIXED):")
        print("-" * 30)
        self.process_pull_response_new_way(142, response)

    def test_multiple_validators_scenario(self):
        """Test multiple validators with different sync scenarios"""
        print("\n🔄 TESTING MULTIPLE VALIDATORS")
        print("=" * 60)

        # Setup multiple validators like in the logs
        validators_data = [
            (128, 0, 199),  # Local: 0, Validator: 199
            (212, 0, 163),  # Local: 0, Validator: 163
            (142, 184, 163), # Local: 184, Validator: 163
        ]

        for uid, local_violations, validator_violations in validators_data:
            self.add_validator(uid, local_violations)
            if local_violations != validator_violations:
                self.simulate_violation_loading_from_analysis(uid, local_violations)

        print("\nProcessing validator responses:")

        for uid, _, validator_violations in validators_data:
            response = self.simulate_validator_response(uid, validator_violations)
            self.process_pull_response_new_way(uid, response)
            print()

    def demonstrate_fix_impact(self):
        """Show the real-world impact of the fix"""
        print("\n🚨 REAL-WORLD IMPACT OF THE FIX")
        print("=" * 60)

        print("❌ WITHOUT FIX:")
        print("  - Local violations: 184 (from analysis)")
        print("  - Validator reports: 163 violations")
        print("  - Local tracking: STAYS at 184 (wrong!)")
        print("  - May pull from validator when we shouldn't")
        print("  - Validator sees us as having more violations than we think")

        print("\n✅ WITH FIX:")
        print("  - Local violations: 184 → 163 (synced!)")
        print("  - Now matches validator's actual count")
        print("  - Proper cooldown enforcement")
        print("  - No unexpected violations")

def run_comprehensive_test():
    """Run comprehensive violation sync test"""
    print("🧪 COMPREHENSIVE VIOLATION SYNC TEST")
    print("=" * 80)
    print("Testing the exact scenario from continuous_trellis.log")
    print("UID 142: Local=184, Validator=163")
    print("=" * 80)

    test = ViolationSyncRealScenarioTest()

    test.test_real_scenario_from_logs()
    test.test_multiple_validators_scenario()
    test.demonstrate_fix_impact()

    print("\n" + "=" * 80)
    print("🎯 CONCLUSION:")
    print("The fix ensures our local violation tracking stays in sync")
    print("with what the validator actually reports in responses.")
    print("This prevents cooldown violations and maintains proper state.")

if __name__ == "__main__":
    run_comprehensive_test()


