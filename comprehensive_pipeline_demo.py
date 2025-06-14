#!/usr/bin/env python3
# Subnet 17 (404-GEN) - Comprehensive Pipeline Demo
# Purpose: Demonstrate complete mining pipeline with all production features

import asyncio
import time
import json
import base64
from pathlib import Path
from typing import List, Dict, Tuple
from dataclasses import dataclass, asdict

# Critical production constants from workers.py
MINER_LICENSE_CONSENT_DECLARATION = "I_AGREE_TO_THE_SUBNET_TERMS_AND_CONDITIONS"

# Validator blacklist - UIDs to avoid
VALIDATOR_BLACKLIST = {180}  # Known problematic validators

@dataclass
class MockTask:
    """Mock mining task for demo"""
    task_id: str
    prompt: str
    validator_hotkey: str
    validator_uid: int
    difficulty: float
    synapse_uuid: str

@dataclass 
class MockResult:
    """Mock mining result for demo"""
    task: MockTask
    validation_score: float
    submission_successful: bool
    compression_type: int
    compressed_size: int
    original_size: int
    signature: str
    error: str = ""

class ComprehensivePipelineDemo:
    """Demonstrates all production pipeline features"""
    
    def __init__(self):
        self.validator_blacklist = set(VALIDATOR_BLACKLIST)
        self.validator_performance = {}
        self.results_dir = Path("demo_results") / f"demo_{int(time.time())}"
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        print("🎭 Subnet 17 Comprehensive Pipeline Demo")
        print("=" * 60)
        print(f"📁 Results directory: {self.results_dir}")
        print(f"🚫 Validator blacklist: {self.validator_blacklist}")

    def is_validator_blacklisted(self, validator_uid: int) -> bool:
        """Check if validator should be avoided"""
        return validator_uid in self.validator_blacklist

    def blacklist_validator(self, validator_uid: int, reason: str):
        """Add validator to blacklist"""
        self.validator_blacklist.add(validator_uid)
        print(f"🚫 Blacklisted validator {validator_uid}: {reason}")

    def evaluate_validator_performance(self, validator_uid: int, success: bool, response_time: float):
        """Track validator performance for blacklisting decisions"""
        if validator_uid not in self.validator_performance:
            self.validator_performance[validator_uid] = {
                'total_requests': 0,
                'successful_requests': 0,
                'average_response_time': 0.0,
                'last_success': time.time() if success else 0
            }
        
        perf = self.validator_performance[validator_uid]
        perf['total_requests'] += 1
        if success:
            perf['successful_requests'] += 1
            perf['last_success'] = time.time()
        
        # Update average response time
        perf['average_response_time'] = (
            (perf['average_response_time'] * (perf['total_requests'] - 1) + response_time) 
            / perf['total_requests']
        )
        
        # Auto-blacklist problematic validators
        success_rate = perf['successful_requests'] / perf['total_requests']
        if (perf['total_requests'] >= 3 and  # Lower threshold for demo
            (success_rate < 0.4 or perf['average_response_time'] > 30)):
            self.blacklist_validator(validator_uid, f"Auto-blacklist: {success_rate:.1%} success rate")

    async def demo_spz_compression(self, data: bytes) -> Tuple[bytes, int, float]:
        """Demonstrate SPZ compression as required by validators"""
        print(f"  📦 Testing SPZ compression on {len(data)} bytes...")
        
        try:
            import pyspz
            start_time = time.time()
            # Use compression with workers=-1 as shown in workers.py example
            compressed_data = pyspz.compress(data, workers=-1)
            compression_time = time.time() - start_time
            compression_ratio = len(compressed_data) / len(data)
            
            print(f"    ✅ SPZ compression successful:")
            print(f"       Original: {len(data):,} bytes")
            print(f"       Compressed: {len(compressed_data):,} bytes")
            print(f"       Ratio: {compression_ratio:.3f} ({compression_ratio*100:.1f}%)")
            print(f"       Time: {compression_time:.3f}s")
            
            return compressed_data, 2, compression_ratio  # Type 2 = SPZ
            
        except Exception as e:
            print(f"    ❌ SPZ compression failed: {e}")
            print(f"    ⚠️ In production, this would be MANDATORY and cause task failure")
            return data, 0, 1.0  # Fallback for demo

    def create_submission_signature(self, task: MockTask, submit_time: int, 
                                  validator_hotkey: str, miner_hotkey: str) -> str:
        """Create submission signature exactly as shown in workers.py"""
        message = (
            f"{MINER_LICENSE_CONSENT_DECLARATION}"
            f"{submit_time}{task.prompt}{validator_hotkey}{miner_hotkey}"
        )
        
        print(f"  🔐 Creating signature for message length: {len(message)} chars")
        print(f"     License declaration: {MINER_LICENSE_CONSENT_DECLARATION}")
        print(f"     Submit time: {submit_time}")
        print(f"     Prompt: {task.prompt}")
        print(f"     Validator hotkey: {validator_hotkey}")
        print(f"     Miner hotkey: {miner_hotkey}")
        
        # Mock signature for demo (in production, use wallet.keypair.sign)
        import hashlib
        signature_hash = hashlib.sha256(message.encode()).hexdigest()
        mock_signature = base64.b64encode(signature_hash.encode()).decode()[:64]
        
        print(f"     Generated signature: {mock_signature[:20]}...")
        return mock_signature

    async def demo_validation_strategy(self, ply_data: bytes, prompt: str) -> Tuple[bool, float]:
        """Demonstrate validation strategy to avoid cooldown penalties"""
        print(f"  🔍 Pre-submission validation strategy demo...")
        
        # Simulate different validation scenarios
        import random
        simulated_score = random.uniform(0.3, 0.95)
        
        validation_threshold = 0.7  # Minimum score to submit
        cooldown_threshold = 0.5    # Below this, send empty results
        
        print(f"    📊 Simulated validation score: {simulated_score:.4f}")
        print(f"    🎯 Validation threshold: {validation_threshold}")
        print(f"    ⚠️ Cooldown threshold: {cooldown_threshold}")
        
        if simulated_score >= validation_threshold:
            decision = "✅ SUBMIT - High quality results"
            should_submit = True
        elif simulated_score >= cooldown_threshold:
            decision = "⚠️ SUBMIT - Acceptable quality"
            should_submit = True
        else:
            decision = "❌ SEND EMPTY - Avoid cooldown penalty"
            should_submit = False
        
        print(f"    💡 Decision: {decision}")
        
        return should_submit, simulated_score

    async def demo_async_validator_operations(self) -> List[MockTask]:
        """Demonstrate async operations with multiple validators"""
        print(f"\n🔄 Demo: Async Validator Operations")
        print("-" * 40)
        
        # Simulate various validators
        validators = [
            {"uid": 100, "hotkey": "validator_hotkey_100", "reputation": "excellent"},
            {"uid": 101, "hotkey": "validator_hotkey_101", "reputation": "good"},
            {"uid": 102, "hotkey": "validator_hotkey_102", "reputation": "good"},
            {"uid": 180, "hotkey": "validator_hotkey_180", "reputation": "blacklisted"},  # Known WC
            {"uid": 200, "hotkey": "validator_hotkey_200", "reputation": "excellent"},
            {"uid": 201, "hotkey": "validator_hotkey_201", "reputation": "poor"},
        ]
        
        prompts = [
            "a modern ergonomic office chair",
            "a sleek gaming laptop computer",
            "a wooden dining table",
            "a comfortable leather sofa",
            "a professional camera tripod"
        ]
        
        async def pull_from_validator(validator: dict, prompt: str):
            """Simulate pulling task from a single validator"""
            uid = validator["uid"]
            
            if self.is_validator_blacklisted(uid):
                print(f"    🚫 Skipping blacklisted validator {uid}")
                return None
            
            print(f"    📡 Pulling from validator {uid} ({validator['reputation']})...")
            
            # Simulate network delay
            delay = random.uniform(0.1, 1.0)
            await asyncio.sleep(delay)
            
            # Simulate success/failure based on reputation
            success_rate = {"excellent": 0.95, "good": 0.85, "poor": 0.4}.get(validator['reputation'], 0.7)
            success = random.random() < success_rate
            
            # Track performance
            self.evaluate_validator_performance(uid, success, delay)
            
            if success:
                task = MockTask(
                    task_id=f"task_{uid}_{int(time.time())}",
                    prompt=prompt,
                    validator_hotkey=validator["hotkey"],
                    validator_uid=uid,
                    difficulty=random.uniform(0.5, 0.9),
                    synapse_uuid=f"synapse_{int(time.time())}"
                )
                print(f"    ✅ Validator {uid}: Task received - '{prompt[:30]}...'")
                return task
            else:
                print(f"    ❌ Validator {uid}: Failed to provide task")
                return None
        
        # Pull from all validators concurrently
        print(f"📡 Pulling tasks from {len(validators)} validators concurrently...")
        
        tasks = await asyncio.gather(*[
            pull_from_validator(validator, prompts[i % len(prompts)])
            for i, validator in enumerate(validators)
        ], return_exceptions=True)
        
        # Filter successful tasks
        valid_tasks = [task for task in tasks if isinstance(task, MockTask)]
        
        print(f"✅ Successfully pulled {len(valid_tasks)} tasks")
        
        # Show validator performance summary
        print(f"\n📊 Validator Performance Summary:")
        for uid, perf in self.validator_performance.items():
            success_rate = perf['successful_requests'] / perf['total_requests']
            status = "🚫 BLACKLISTED" if self.is_validator_blacklisted(uid) else "✅ ACTIVE"
            print(f"   Validator {uid}: {success_rate:.1%} success, {perf['average_response_time']:.1f}s avg ({status})")
        
        return valid_tasks

    async def demo_complete_mining_pipeline(self, task: MockTask) -> MockResult:
        """Demonstrate complete mining pipeline for a single task"""
        print(f"\n⛏️ Demo: Complete Mining Pipeline")
        print(f"   Task: {task.task_id}")
        print(f"   Prompt: '{task.prompt}'")
        print(f"   Validator: {task.validator_uid}")
        print("-" * 50)
        
        # Step 1: Generate mock PLY data
        print("🎨 Step 1: 3D Generation (simulated)")
        mock_ply_data = b"PLY_DATA_" + b"X" * 50000  # 50KB mock data
        print(f"   Generated PLY: {len(mock_ply_data)} bytes")
        
        # Step 2: Pre-submission validation
        print("\n🔍 Step 2: Pre-submission Validation")
        should_submit, score = await self.demo_validation_strategy(mock_ply_data, task.prompt)
        
        # Step 3: SPZ Compression (mandatory)
        print("\n📦 Step 3: SPZ Compression (Mandatory)")
        compressed_data, compression_type, compression_ratio = await self.demo_spz_compression(mock_ply_data)
        
        # Step 4: Create submission
        print("\n📝 Step 4: Submission Creation")
        submit_time = time.time_ns()
        miner_hotkey = "demo_miner_hotkey"
        
        signature = self.create_submission_signature(
            task, submit_time, task.validator_hotkey, miner_hotkey
        )
        
        # Step 5: Determine submission strategy
        print("\n🎯 Step 5: Submission Strategy")
        if should_submit:
            # Submit compressed results
            results = base64.b64encode(compressed_data).decode('utf-8')
            print("   ✅ Submitting high-quality compressed results")
        else:
            # Send empty results to avoid cooldown
            results = ""
            compression_type = 0
            print("   ⚠️ Sending empty results (validation failed)")
        
        # Create final submission data
        submission_data = {
            "task": {
                "task_id": task.task_id,
                "prompt": task.prompt,
                "synapse_uuid": task.synapse_uuid
            },
            "results": results,
            "compression": compression_type,
            "data_format": "ply",
            "data_ver": 0,
            "submit_time": submit_time,
            "signature": signature,
            "validator_hotkey": task.validator_hotkey,
            "validator_uid": task.validator_uid,
            "miner_hotkey": miner_hotkey,
            "local_validation_score": score
        }
        
        print(f"\n📋 Final Submission Summary:")
        print(f"   Format: {submission_data['data_format']}")
        print(f"   Compression: {submission_data['compression']} ({'SPZ' if compression_type == 2 else 'None'})")
        print(f"   Results size: {len(results)} chars")
        print(f"   Validation score: {score:.4f}")
        print(f"   Signature: {signature[:20]}...")
        
        return MockResult(
            task=task,
            validation_score=score,
            submission_successful=True,
            compression_type=compression_type,
            compressed_size=len(compressed_data),
            original_size=len(mock_ply_data),
            signature=signature
        )

    async def save_demo_results(self, results: List[MockResult]):
        """Save demo results"""
        print(f"\n💾 Saving demo results...")
        
        summary = {
            "demo_timestamp": time.time(),
            "validator_blacklist": list(self.validator_blacklist),
            "validator_performance": self.validator_performance,
            "total_tasks": len(results),
            "successful_tasks": len([r for r in results if r.submission_successful]),
            "average_score": sum(r.validation_score for r in results) / len(results) if results else 0,
            "compression_stats": {
                "spz_compressed": len([r for r in results if r.compression_type == 2]),
                "average_compression_ratio": sum(r.compressed_size / r.original_size for r in results) / len(results) if results else 1.0
            },
            "results": [asdict(r) for r in results]
        }
        
        # Save summary
        summary_path = self.results_dir / "demo_summary.json"
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"   📊 Demo summary saved: {summary_path}")
        return summary

    async def run_comprehensive_demo(self):
        """Run the complete comprehensive demo"""
        print("\n🚀 Starting Comprehensive Pipeline Demo")
        print("=" * 60)
        
        # Phase 1: Async Validator Operations
        tasks = await self.demo_async_validator_operations()
        
        # Ensure we have at least a few tasks for demo
        if len(tasks) < 2:
            print(f"⚠️ Only {len(tasks)} tasks received, generating additional demo tasks...")
            # Add some guaranteed demo tasks
            additional_tasks = [
                MockTask(
                    task_id="demo_task_1",
                    prompt="a modern ergonomic office chair",
                    validator_hotkey="validator_hotkey_100",
                    validator_uid=100,
                    difficulty=0.7,
                    synapse_uuid="demo_synapse_1"
                ),
                MockTask(
                    task_id="demo_task_2", 
                    prompt="a sleek gaming laptop computer",
                    validator_hotkey="validator_hotkey_200",
                    validator_uid=200,
                    difficulty=0.8,
                    synapse_uuid="demo_synapse_2"
                )
            ]
            tasks.extend(additional_tasks)
            print(f"✅ Added {len(additional_tasks)} guaranteed demo tasks")
        
        if not tasks:
            print("❌ No valid tasks available for demo")
            return
        
        # Phase 2: Process tasks with full pipeline
        print(f"\n🔄 Processing {len(tasks)} tasks with complete pipeline...")
        
        results = []
        for i, task in enumerate(tasks):
            print(f"\n{'='*60}")
            print(f"Processing Task {i+1}/{len(tasks)}")
            print(f"{'='*60}")
            
            result = await self.demo_complete_mining_pipeline(task)
            results.append(result)
        
        # Phase 3: Save and analyze results
        summary = await self.save_demo_results(results)
        
        # Phase 4: Final report
        print(f"\n📊 Comprehensive Demo Report")
        print("=" * 60)
        print(f"Total Tasks Processed: {summary['total_tasks']}")
        print(f"Successful Submissions: {summary['successful_tasks']}")
        print(f"Average Validation Score: {summary['average_score']:.4f}")
        print(f"SPZ Compressed Results: {summary['compression_stats']['spz_compressed']}")
        print(f"Average Compression Ratio: {summary['compression_stats']['average_compression_ratio']:.3f}")
        print(f"Validators Blacklisted: {len(self.validator_blacklist)}")
        
        print(f"\n🎯 Key Production Features Demonstrated:")
        print("   ✅ Validator blacklisting (UID 180 + auto-blacklist)")
        print("   ✅ Async validator operations")
        print("   ✅ Mandatory SPZ compression")
        print("   ✅ Pre-submission validation")
        print("   ✅ Empty results for failed validation")
        print("   ✅ Proper signature creation")
        print("   ✅ Validator performance tracking")
        
        print(f"\n📁 All results saved to: {self.results_dir}")
        return summary


async def main():
    """Run the comprehensive demo"""
    demo = ComprehensivePipelineDemo()
    await demo.run_comprehensive_demo()
    print("\n🎉 Comprehensive demo completed!")


if __name__ == "__main__":
    asyncio.run(main()) 