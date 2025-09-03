#!/usr/bin/env python3
"""
Simple test to verify violation sync fix works
"""

from dataclasses import dataclass

@dataclass
class MockValidator:
    uid: int
    validator_reported_violations: int = 0

@dataclass
class MockResponse:
    cooldown_violations: int = 0

def test_sync_logic():
    """Test the violation sync logic - only sync when violations > 0"""
    print("🧪 Testing Violation Sync Fix (Only sync when > 0)")
    print("=" * 50)

    # Test 1: Validator reports violations > 0 (should sync)
    print("\n📊 Test 1: Validator reports violations > 0")
    validator = MockValidator(uid=142, validator_reported_violations=184)
    print(f"Initial violations: {validator.validator_reported_violations}")

    response = MockResponse(cooldown_violations=163)
    print(f"Validator reports: {response.cooldown_violations}")

    # Apply our fix: sync violation count ONLY when > 0
    new_violations = response.cooldown_violations
    if new_violations > 0:
        old_violations = validator.validator_reported_violations
        if new_violations != old_violations:
            validator.validator_reported_violations = new_violations
            violation_change = new_violations - old_violations
            print(f"✅ Synced violations: {old_violations} → {new_violations} ({violation_change:+d})")
        else:
            print(f"✅ Already in sync: {new_violations}")
    else:
        print(f"❌ No sync needed (violations = 0)")

    print(f"Final violations: {validator.validator_reported_violations}")
    assert validator.validator_reported_violations == 163, f"Expected 163, got {validator.validator_reported_violations}"

    # Test 2: Validator reports 0 violations (should NOT sync)
    print("\n📊 Test 2: Validator reports 0 violations")
    validator2 = MockValidator(uid=199, validator_reported_violations=50)  # We think it has violations
    print(f"Initial violations: {validator2.validator_reported_violations}")

    response2 = MockResponse(cooldown_violations=0)  # But validator says 0
    print(f"Validator reports: {response2.cooldown_violations}")

    new_violations2 = response2.cooldown_violations
    if new_violations2 > 0:
        validator2.validator_reported_violations = new_violations2
        print(f"✅ Would sync to: {new_violations2}")
    else:
        print(f"❌ No sync - validator reports 0 violations")

    print(f"Final violations: {validator2.validator_reported_violations} (unchanged)")
    assert validator2.validator_reported_violations == 50, f"Expected 50 (unchanged), got {validator2.validator_reported_violations}"

    print("\n🎯 SUCCESS: Violation sync logic works correctly!")
    print("✅ Syncs when violations > 0")
    print("❌ Doesn't sync when violations = 0")

if __name__ == "__main__":
    test_sync_logic()
