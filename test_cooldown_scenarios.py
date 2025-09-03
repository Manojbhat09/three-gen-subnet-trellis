#!/usr/bin/env python3
"""
Comprehensive test of cooldown violation scenarios and mimic behavior
"""

import time
import json
from dataclasses import dataclass
from typing import Dict, List, Optional, Any

# Mock constants matching our real system
MIN_TASK_INTERVAL = 35.0
THROTTLE_PERIOD = 35.0
FAILED_VALIDATOR_DELAY = 165.0
VIOLATION_THRESHOLD = 5

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

    # Mimic fields
    mimic_last_pull_attempt: float = 0.0
    mimic_expected_cooldown_until: float = 0.0

    # Performance tracking
    last_submit_time: float = 0.0
    total_tasks_received: int = 0
    total_successful_submissions: int = 0

@dataclass
class MockPullResponse:
    """Mock response from validator pull task"""
    status_code: int
    cooldown_until: float = 0.0
    cooldown_violations: Optional[int] = None
    throttle_period: float = 0.0
    task_id: str = ""
    prompt: str = ""
    success: bool = True

@dataclass
class MockSubmitResponse:
    """Mock response from validator submit result"""
    status_code: int
    cooldown_until: float = 0.0
    cooldown_violations: Optional[int] = None
    feedback_score: float = 0.8
    success: bool = True

class MockCooldownSystem:
    """Mock cooldown system that mimics our real implementation"""

    def __init__(self):
        self.current_time = 1000000000.0  # Fixed start time for testing
        self.validators: Dict[int, MockValidatorState] = {}
        self.logs: List[str] = []

    def log(self, message: str):
        """Log a message"""
        timestamp = f"{self.current_time:.1f}"
        self.logs.append(f"{timestamp} - {message}")
        print(f"{timestamp} - {message}")

    def advance_time(self, seconds: float):
        """Advance the simulation time"""
        self.current_time += seconds
        self.log(f"⏰ Advanced time by {seconds}s (now: {self.current_time:.1f})")

    def add_validator(self, uid: int, violations: int = 0, cooldown_until: float = 0.0):
        """Add a mock validator"""
        validator = MockValidatorState(
            uid=uid,
            validator_reported_violations=violations,
            validator_enforced_cooldown_until=cooldown_until
        )
        self.validators[uid] = validator
        self.log(f"➕ Added validator UID {uid} with {violations} violations")

    def should_prevent_pull_attempt(self, validator: MockValidatorState) -> bool:
        """Check if pull should be prevented (mimics real logic)"""
        current_time = self.current_time

        # Check validator enforced cooldown
        if validator.validator_enforced_cooldown_until > current_time:
            remaining = validator.validator_enforced_cooldown_until - current_time
            self.log(f"🚫 UID {validator.uid}: Validator cooldown active ({remaining:.1f}s remaining)")
            return True

        # Check miner cooldown
        if validator.miner_cooldown_until > current_time:
            remaining = validator.miner_cooldown_until - current_time
            self.log(f"🚫 UID {validator.uid}: Miner cooldown active ({remaining:.1f}s remaining)")
            return True

        # Check minimum task interval using mimic logic
        if validator.mimic_last_pull_attempt > 0:
            time_since_last_pull = current_time - validator.mimic_last_pull_attempt
            if time_since_last_pull < MIN_TASK_INTERVAL:
                remaining = MIN_TASK_INTERVAL - time_since_last_pull
                self.log(f"🚫 UID {validator.uid}: Too soon since last pull ({remaining:.1f}s remaining)")
                return True

        # Check violation threshold
        if validator.validator_reported_violations >= VIOLATION_THRESHOLD:
            self.log(f"🚫 UID {validator.uid}: Too many violations ({validator.validator_reported_violations})")
            return True

        return False

    def update_mimic_state_after_pull_attempt(self, validator: MockValidatorState, was_successful: bool = False):
        """Update mimic state after pull attempt"""
        validator.mimic_last_pull_attempt = self.current_time

        if was_successful:
            # Set expected cooldown based on MIN_TASK_INTERVAL
            validator.mimic_expected_cooldown_until = self.current_time + MIN_TASK_INTERVAL
            self.log(f"✅ UID {validator.uid}: Updated mimic state (next pull allowed at {validator.mimic_expected_cooldown_until:.1f})")
        else:
            # For failures, apply extended cooldown
            validator.mimic_expected_cooldown_until = self.current_time + FAILED_VALIDATOR_DELAY
            self.log(f"❌ UID {validator.uid}: Updated mimic state for failure (next pull allowed at {validator.mimic_expected_cooldown_until:.1f})")

    def increment_cooldown_violations(self, validator: MockValidatorState, reason: str):
        """Increment violation count"""
        validator.validator_reported_violations += 1
        self.log(f"⚠️ VIOLATION: UID {validator.uid} violations +1 → {validator.validator_reported_violations} ({reason})")

    def decrement_cooldown_violations(self, validator: MockValidatorState):
        """Decrement violation count (for successful submissions)"""
        old_count = validator.validator_reported_violations
        if old_count > 0:
            validator.validator_reported_violations = max(0, old_count - 1)
            self.log(f"✅ SUCCESS: UID {validator.uid} violations -1 → {validator.validator_reported_violations}")

    def simulate_pull_task(self, validator: MockValidatorState) -> MockPullResponse:
        """Simulate pulling a task from validator"""
        self.log(f"📥 Attempting to pull task from UID {validator.uid}")

        # Check if we should prevent the pull
        if self.should_prevent_pull_attempt(validator):
            self.increment_cooldown_violations(validator, "Attempted pull while on cooldown")
            self.update_mimic_state_after_pull_attempt(validator, was_successful=False)

            return MockPullResponse(
                status_code=408,  # Timeout (cooldown violation)
                success=False
            )

        # Simulate different scenarios
        if validator.uid == 142:  # This validator tends to timeout
            if self.current_time % 100 < 30:  # 30% chance of timeout
                self.update_mimic_state_after_pull_attempt(validator, was_successful=False)
                return MockPullResponse(
                    status_code=408,
                    success=False
                )

        # Successful pull
        task_id = f"task_{validator.uid}_{int(self.current_time)}"
        prompt = f"Generate a beautiful {['landscape', 'portrait', 'abstract art', 'realistic scene'][validator.uid % 4]} image"

        self.update_mimic_state_after_pull_attempt(validator, was_successful=True)

        return MockPullResponse(
            status_code=200,
            task_id=task_id,
            prompt=prompt,
            cooldown_until=self.current_time + MIN_TASK_INTERVAL,
            cooldown_violations=validator.validator_reported_violations,
            success=True
        )

    def simulate_submit_result(self, validator: MockValidatorState, task_id: str) -> MockSubmitResponse:
        """Simulate submitting result to validator"""
        self.log(f"📤 Submitting result for task {task_id} to UID {validator.uid}")

        # Simulate processing time
        processing_time = 8.0 + (validator.uid % 3)  # 8-10 seconds

        # Check for submission timeout (simulating network issues)
        if validator.uid == 27 and self.current_time % 150 < 20:  # Occasional failures
            self.log(f"❌ UID {validator.uid}: Submit timeout after {processing_time:.1f}s")
            return MockSubmitResponse(
                status_code=408,
                success=False
            )

        # Successful submission
        self.log(f"✅ UID {validator.uid}: Submit successful in {processing_time:.1f}s")

        # Decrement violations (successful submission)
        self.decrement_cooldown_violations(validator)

        return MockSubmitResponse(
            status_code=200,
            cooldown_until=self.current_time + MIN_TASK_INTERVAL,
            cooldown_violations=None,  # NOT FOUND = successful submission = violations -1
            feedback_score=0.7 + (validator.uid % 3) * 0.1,
            success=True
        )

    def run_scenario(self, scenario_name: str, scenario_func):
        """Run a specific test scenario"""
        self.log(f"\n🎬 STARTING SCENARIO: {scenario_name}")
        self.log("=" * 60)

        try:
            scenario_func()
            self.log(f"✅ SCENARIO COMPLETE: {scenario_name}")
        except Exception as e:
            self.log(f"❌ SCENARIO FAILED: {scenario_name} - {e}")

        self.log("=" * 60)

def test_normal_successful_cycle():
    """Test normal successful pull/submit cycle"""
    system = MockCooldownSystem()

    # Add validator
    system.add_validator(199, violations=0)

    # Pull task
    validator = system.validators[199]
    pull_response = system.simulate_pull_task(validator)

    if pull_response.success:
        system.advance_time(10)  # Simulate processing time

        # Submit result
        submit_response = system.simulate_submit_result(validator, pull_response.task_id)

        if submit_response.success:
            system.log("🎯 NORMAL CYCLE: Pull → Process → Submit successful")

def test_cooldown_violation_scenario():
    """Test cooldown violation when pulling too soon"""
    system = MockCooldownSystem()

    # Add validator with recent activity
    system.add_validator(142, violations=2)
    validator = system.validators[142]

    # First pull (should succeed)
    pull1 = system.simulate_pull_task(validator)

    # Try to pull again immediately (should fail due to cooldown)
    system.advance_time(5)  # Only 5 seconds later
    pull2 = system.simulate_pull_task(validator)

    if not pull2.success:
        system.log("🎯 COOLDOWN VIOLATION: Second pull correctly blocked")

    # Wait proper interval and try again
    system.advance_time(35)  # Wait full MIN_TASK_INTERVAL
    pull3 = system.simulate_pull_task(validator)

    if pull3.success:
        system.log("🎯 RECOVERY: Pull successful after proper cooldown")

def test_network_failure_scenario():
    """Test network failure handling"""
    system = MockCooldownSystem()

    # Add validator prone to failures
    system.add_validator(27, violations=1)
    validator = system.validators[27]

    # Simulate multiple failures
    for i in range(3):
        pull = system.simulate_pull_task(validator)
        if not pull.success:
            system.log(f"🎯 NETWORK FAILURE {i+1}: Correctly handled timeout")
        system.advance_time(10)

    # Check if violations increased appropriately
    system.log(f"🎯 FINAL VIOLATIONS: {validator.validator_reported_violations} (should be higher due to failures)")

def test_emergency_violation_scenario():
    """Test emergency response to high violation counts"""
    system = MockCooldownSystem()

    # Add validator with high violations
    system.add_validator(81, violations=8)  # Above threshold
    validator = system.validators[81]

    # Try to pull (should be blocked)
    pull = system.simulate_pull_task(validator)

    if not pull.success:
        system.log("🎯 EMERGENCY: High violation validator correctly blocked")

def test_multiple_validators_scenario():
    """Test multiple validators with different states"""
    system = MockCooldownSystem()

    # Add multiple validators with different states
    system.add_validator(142, violations=3)  # Moderate violations
    system.add_validator(199, violations=0)  # Clean record
    system.add_validator(27, violations=1)   # Low violations

    # Simulate round-robin pulls
    for i in range(3):
        system.log(f"\n🔄 ROUND {i+1}")
        for uid in [142, 199, 27]:
            validator = system.validators[uid]
            pull = system.simulate_pull_task(validator)
            if pull.success:
                system.advance_time(8)
                submit = system.simulate_submit_result(validator, pull.task_id)
                system.log(f"   UID {uid}: {'SUCCESS' if submit.success else 'FAILED'}")
            else:
                system.log(f"   UID {uid}: BLOCKED")
        system.advance_time(30)  # Wait between rounds

def run_all_scenarios():
    """Run all test scenarios"""
    print("🚀 STARTING COMPREHENSIVE COOLDOWN SCENARIO TESTS")
    print("=" * 80)

    scenarios = [
        ("Normal Successful Cycle", test_normal_successful_cycle),
        ("Cooldown Violation", test_cooldown_violation_scenario),
        ("Network Failure Handling", test_network_failure_scenario),
        ("Emergency Violation Response", test_emergency_violation_scenario),
        ("Multiple Validators", test_multiple_validators_scenario),
    ]

    for scenario_name, scenario_func in scenarios:
        system = MockCooldownSystem()
        system.run_scenario(scenario_name, scenario_func)

        # Show final state
        print(f"\n📊 FINAL STATE FOR '{scenario_name}':")
        for uid, validator in system.validators.items():
            print(f"   UID {uid}: {validator.validator_reported_violations} violations, "
                  f"cooldown: {validator.validator_enforced_cooldown_until:.1f}")

        print("-" * 80)

    print("🎉 ALL SCENARIO TESTS COMPLETE!")
    print("Check the logs above to verify cooldown logic is working correctly.")

if __name__ == "__main__":
    run_all_scenarios()


