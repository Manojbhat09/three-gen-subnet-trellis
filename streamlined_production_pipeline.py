#!/usr/bin/env python3
# Subnet 17 (404-GEN) - Streamlined Production Pipeline  
# Purpose: Production mining pipeline using ONLY generation server validation

import asyncio
import aiohttp
import argparse
import time
import base64
from typing import List, Dict, Tuple
from dataclasses import dataclass

# Critical production constants from workers.py
MINER_LICENSE_CONSENT_DECLARATION = "I_AGREE_TO_THE_SUBNET_TERMS_AND_CONDITIONS"
VALIDATOR_BLACKLIST = {180}  # Known problematic validators

# Configuration - ONLY generation server needed
GENERATION_SERVER_URL = "http://127.0.0.1:8095/generate/"
MINING_SUBMISSION_URL = "http://127.0.0.1:8095/mining/submit/"

@dataclass
class StreamlinedTask:
    task_id: str
    prompt: str
    validator_hotkey: str
    validator_uid: int

class StreamlinedMiningPipeline:
    """Simplified pipeline using only generation server validation"""
    
    def __init__(self):
        self.validator_blacklist = set(VALIDATOR_BLACKLIST)
        print("🚀 Streamlined Mining Pipeline")
        print("✅ Using ONLY generation server validation")
        print("🚫 No redundant validation server needed")

    async def mine_single_task(self, task: StreamlinedTask) -> Dict:
        """Complete mining pipeline using only generation server"""
        
        if task.validator_uid in self.validator_blacklist:
            print(f"🚫 Skipping blacklisted validator {task.validator_uid}")
            return {"success": False, "reason": "blacklisted"}
        
        async with aiohttp.ClientSession() as session:
            try:
                print(f"\n⛏️ Mining: '{task.prompt}'")
                
                # Step 1: Generate with built-in validation
                form_data = aiohttp.FormData()
                form_data.add_field('prompt', task.prompt)
                form_data.add_field('seed', str(42))
                form_data.add_field('use_bpt', 'false')
                form_data.add_field('return_compressed', 'true')
                
                async with session.post(GENERATION_SERVER_URL, data=form_data, timeout=300) as response:
                    if response.status != 200:
                        return {"success": False, "reason": f"Generation failed: {response.status}"}
                    
                    # Generation server ALREADY provides validation!
                    generation_id = response.headers.get('X-Generation-ID', '')
                    validation_score = float(response.headers.get('X-Local-Validation-Score', '0.0'))
                    mining_ready = response.headers.get('X-Mining-Ready', 'false').lower() == 'true'
                    face_count = int(response.headers.get('X-Face-Count', '0'))
                    
                    print(f"  ✅ Generation ID: {generation_id}")
                    print(f"  📊 Validation Score: {validation_score:.4f}")
                    print(f"  ⛏️ Mining Ready: {mining_ready}")
                    print(f"  🔺 Face Count: {face_count:,}")
                
                # Step 2: Check if meets submission threshold (no external validation needed!)
                if validation_score < 0.7:
                    print(f"  ⚠️ Score {validation_score:.4f} below threshold - sending empty results")
                    return await self._submit_empty_results(session, task, generation_id)
                
                # Step 3: Direct mining submission (generation server handles compression)
                return await self._submit_mining_results(session, task, generation_id)
                
            except Exception as e:
                print(f"  ❌ Mining failed: {e}")
                return {"success": False, "reason": str(e)}

    async def _submit_mining_results(self, session: aiohttp.ClientSession, 
                                   task: StreamlinedTask, generation_id: str) -> Dict:
        """Submit mining results directly to generation server mining endpoint"""
        
        form_data = aiohttp.FormData()
        form_data.add_field('generation_id', generation_id)
        form_data.add_field('task_id', task.task_id)
        form_data.add_field('validator_hotkey', task.validator_hotkey)
        form_data.add_field('validator_uid', str(task.validator_uid))
        
        async with session.post(MINING_SUBMISSION_URL, data=form_data, timeout=60) as response:
            if response.status == 200:
                result = await response.json()
                print(f"  ✅ Mining submission successful!")
                print(f"     Compression: {result.get('compression', 'unknown')}")
                print(f"     Results size: {len(result.get('results', ''))}")
                return {"success": True, "submission_data": result}
            else:
                error_text = await response.text()
                print(f"  ❌ Mining submission failed: {response.status}")
                return {"success": False, "reason": f"Submission failed: {error_text[:100]}"}

    async def _submit_empty_results(self, session: aiohttp.ClientSession,
                                  task: StreamlinedTask, generation_id: str) -> Dict:
        """Submit empty results to avoid cooldown penalties"""
        
        submit_time = time.time_ns()
        signature = self._create_signature(task, submit_time, "demo_miner_hotkey")
        
        # Empty submission data (prevents cooldown)
        submission_data = {
            "task": {
                "task_id": task.task_id,
                "prompt": task.prompt,
                "synapse_uuid": f"empty_{int(time.time())}"
            },
            "results": "",  # Empty results
            "compression": 0,
            "data_format": "ply", 
            "submit_time": submit_time,
            "signature": signature,
            "validator_hotkey": task.validator_hotkey,
            "validator_uid": task.validator_uid,
            "miner_hotkey": "demo_miner_hotkey",
            "local_validation_score": 0.0,
            "generation_id": generation_id
        }
        
        print(f"  📤 Submitting empty results (score too low)")
        return {"success": True, "submission_data": submission_data, "empty_submission": True}

    def _create_signature(self, task: StreamlinedTask, submit_time: int, miner_hotkey: str) -> str:
        """Create signature as per workers.py"""
        message = (
            f"{MINER_LICENSE_CONSENT_DECLARATION}"
            f"{submit_time}{task.prompt}{task.validator_hotkey}{miner_hotkey}"
        )
        
        import hashlib
        signature_hash = hashlib.sha256(message.encode()).hexdigest()
        return base64.b64encode(signature_hash.encode()).decode()[:64]

    async def test_infrastructure(self) -> bool:
        """Test only the generation server (no redundant validation server)"""
        print("🔧 Testing streamlined infrastructure...")
        
        async with aiohttp.ClientSession() as session:
            try:
                # Test generation server
                async with session.get("http://127.0.0.1:8095/status/", timeout=5) as response:
                    if response.status == 200:
                        print("  ✅ Generation server: Online")
                        return True
                    else:
                        print(f"  ❌ Generation server: Error {response.status}")
                        return False
            except Exception as e:
                print(f"  ❌ Generation server: Offline ({e})")
                return False

async def main():
    """Run streamlined mining demo"""
    pipeline = StreamlinedMiningPipeline()
    
    # Test infrastructure
    if not await pipeline.test_infrastructure():
        print("❌ Infrastructure test failed")
        return 1
    
    # Demo tasks
    demo_tasks = [
        StreamlinedTask("task_1", "a modern office chair", "validator_hotkey_100", 100),
        StreamlinedTask("task_2", "a gaming laptop", "validator_hotkey_101", 101),
        StreamlinedTask("task_3", "a wooden table", "validator_hotkey_180", 180),  # Blacklisted
    ]
    
    print(f"\n🎯 Processing {len(demo_tasks)} demo tasks...")
    
    results = []
    for i, task in enumerate(demo_tasks, 1):
        print(f"\n{'='*50} Task {i}/{len(demo_tasks)} {'='*50}")
        result = await pipeline.mine_single_task(task)
        results.append(result)
    
    # Summary
    successful = len([r for r in results if r["success"]])
    print(f"\n📊 Results: {successful}/{len(results)} successful")
    print("🎉 Streamlined pipeline complete!")
    
    return 0

if __name__ == "__main__":
    exit_code = asyncio.run(main())
    exit(exit_code) 