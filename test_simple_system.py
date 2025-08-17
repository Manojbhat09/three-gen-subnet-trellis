#!/usr/bin/env python3
"""
Test Simple System
Quick test to verify the simple distributed RL system works
"""

import time
import requests
import json
from rich.console import Console

console = Console()

def test_coordinator():
    """Test coordinator functionality"""
    console.print("🧪 Testing Coordinator...")
    
    try:
        # Test health
        response = requests.get("http://localhost:8090/health", timeout=5)
        if response.status_code == 200:
            console.print("  ✅ Coordinator health check passed")
        else:
            console.print(f"  ❌ Health check failed: {response.status_code}")
            return False
        
        # Test system status
        response = requests.get("http://localhost:8090/api/system/status", timeout=5)
        if response.status_code == 200:
            status = response.json()
            console.print(f"  ✅ System status: {status['status']}")
            console.print(f"    Jobs queued: {status['jobs']['queued']}")
            console.print(f"    Jobs active: {status['jobs']['active']}")
            console.print(f"    Jobs completed: {status['jobs']['completed']}")
            return True
        else:
            console.print(f"  ❌ Status check failed: {response.status_code}")
            return False
            
    except Exception as e:
        console.print(f"  ❌ Coordinator test failed: {e}")
        return False

def test_gpu_agent(gpu_id=0):
    """Test GPU agent functionality"""
    console.print(f"🧪 Testing GPU {gpu_id} Agent...")
    
    try:
        port = 8096 + gpu_id
        
        # Test health
        response = requests.get(f"http://localhost:{port}/health", timeout=5)
        if response.status_code == 200:
            console.print(f"  ✅ GPU {gpu_id} health check passed")
        else:
            console.print(f"  ❌ Health check failed: {response.status_code}")
            return False
        
        # Test status
        response = requests.get(f"http://localhost:{port}/status", timeout=5)
        if response.status_code == 200:
            status = response.json()
            console.print(f"  ✅ GPU {gpu_id} status: {status['status']}")
            console.print(f"    Prompts processed: {status['stats']['prompts_processed']}")
            return True
        else:
            console.print(f"  ❌ Status check failed: {response.status_code}")
            return False
            
    except Exception as e:
        console.print(f"  ❌ GPU {gpu_id} test failed: {e}")
        return False

def test_single_prompt_processing():
    """Test single prompt processing"""
    console.print("🧪 Testing Single Prompt Processing...")
    
    try:
        # Test with GPU 0
        response = requests.post(
            "http://localhost:8096/test_prompt",
            json={
                "prompt": "a red sports car",
                "target_score": 0.8,
                "max_episodes": 3
            },
            timeout=30
        )
        
        if response.status_code == 200:
            result = response.json()
            prompt_result = result['result']
            console.print("  ✅ Single prompt processing successful")
            console.print(f"    Original: '{prompt_result['original_prompt']}'")
            console.print(f"    Final: '{prompt_result['final_prompt']}'")
            console.print(f"    Score: {prompt_result['final_score']:.4f}")
            console.print(f"    Episodes: {prompt_result['episodes_run']}")
            console.print(f"    Time: {prompt_result['processing_time_minutes']:.2f} minutes")
            return True
        else:
            console.print(f"  ❌ Processing failed: {response.status_code}")
            return False
            
    except Exception as e:
        console.print(f"  ❌ Single prompt test failed: {e}")
        return False

def test_job_submission():
    """Test job submission to coordinator"""
    console.print("🧪 Testing Job Submission...")
    
    try:
        # Submit job
        job_data = {
            "prompts": [
                "a blue house",
                "a green tree",
                "a yellow car"
            ],
            "target_score": 0.85,
            "max_episodes": 3
        }
        
        response = requests.post(
            "http://localhost:8090/api/jobs/submit",
            json=job_data,
            timeout=10
        )
        
        if response.status_code == 200:
            result = response.json()
            job_id = result['job_id']
            console.print(f"  ✅ Job submitted successfully: {job_id}")
            console.print(f"    Prompts: {result['prompts_count']}")
            
            # Wait a moment and check status
            console.print("  ⏳ Waiting for processing...")
            time.sleep(15)
            
            # Check job status
            response = requests.get(f"http://localhost:8090/api/jobs/{job_id}", timeout=5)
            if response.status_code == 200:
                status = response.json()
                console.print(f"  ✅ Job status: {status['status']}")
                
                if status['status'] == 'completed':
                    results = status['results']
                    console.print(f"    Average score: {results['average_score']:.4f}")
                    console.print(f"    Processing time: {results['processing_time_minutes']:.2f} minutes")
                
                return True
            else:
                console.print(f"  ⚠️  Job status check failed: {response.status_code}")
                return False
        else:
            console.print(f"  ❌ Job submission failed: {response.status_code}")
            return False
            
    except Exception as e:
        console.print(f"  ❌ Job submission test failed: {e}")
        return False

def test_cross_gpu_insights():
    """Test cross-GPU insights"""
    console.print("🧪 Testing Cross-GPU Insights...")
    
    try:
        response = requests.get("http://localhost:8090/api/insights", timeout=5)
        
        if response.status_code == 200:
            insights = response.json()
            console.print(f"  ✅ Insights retrieved")
            console.print(f"    Total insights: {insights['total_insights']}")
            console.print(f"    Strategy performance: {len(insights['strategy_performance'])} strategies")
            
            if insights['strategy_performance']:
                for strategy, perf in insights['strategy_performance'].items():
                    console.print(f"      {strategy}: {perf['avg_score']:.3f} avg, {perf['success_rate']:.1%} success")
            
            return True
        else:
            console.print(f"  ❌ Insights retrieval failed: {response.status_code}")
            return False
            
    except Exception as e:
        console.print(f"  ❌ Insights test failed: {e}")
        return False

def main():
    """Run all tests"""
    console.print("🚀 Testing Simple Distributed RL System\\n")
    
    tests = [
        ("Coordinator", test_coordinator),
        ("GPU Agent", test_gpu_agent),
        ("Single Prompt", test_single_prompt_processing),
        ("Job Submission", test_job_submission),
        ("Cross-GPU Insights", test_cross_gpu_insights)
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        console.print(f"\\n{'='*50}")
        result = test_func()
        results[test_name] = result
        console.print(f"{'='*50}")
    
    # Summary
    console.print("\\n🎯 Test Summary:")
    passed = 0
    for test_name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        console.print(f"  {test_name}: {status}")
        if result:
            passed += 1
    
    console.print(f"\\n📊 Results: {passed}/{len(tests)} tests passed")
    
    if passed == len(tests):
        console.print("\\n🎉 All tests passed! System is working correctly.", style="bold green")
    else:
        console.print(f"\\n⚠️  {len(tests) - passed} tests failed. Check system status.", style="bold yellow")

if __name__ == "__main__":
    main()




