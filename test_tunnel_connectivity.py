#!/usr/bin/env python3
"""
Test Tunnel Connectivity for Distributed RL System
This script tests the SSH tunnel connections to verify services are accessible locally.
"""

import requests
import time
import json
from typing import Dict, List, Tuple
import sys

class TunnelTester:
    def __init__(self):
        self.services = {
            'coordinator': {
                'port': 18090,
                'name': 'Coordinator API',
                'endpoints': ['/api/system/status', '/api/jobs', '/api/insights']
            },
            'dashboard': {
                'port': 18100,
                'name': 'Dashboard Frontend',
                'endpoints': ['/', '/api/status']
            },
            'gpu_agents': {
                'base_port': 18101,
                'name': 'GPU Agents',
                'endpoints': ['/status', '/test_prompt'],
                'count': 8
            }
        }
        
        self.results = {}
    
    def test_service(self, service_name: str, port: int, endpoint: str = '/') -> Tuple[bool, str]:
        """Test a single service endpoint"""
        try:
            url = f"http://localhost:{port}{endpoint}"
            response = requests.get(url, timeout=5)
            
            if response.status_code == 200:
                return True, f"✅ {response.status_code} - OK"
            else:
                return False, f"❌ {response.status_code} - {response.reason}"
                
        except requests.exceptions.ConnectionError:
            return False, "❌ Connection refused - Tunnel may not be active"
        except requests.exceptions.Timeout:
            return False, "❌ Timeout - Service may be slow"
        except Exception as e:
            return False, f"❌ Error: {str(e)}"
    
    def test_coordinator(self) -> Dict:
        """Test coordinator API endpoints"""
        print(f"\n🔍 Testing {self.services['coordinator']['name']} (Port {self.services['coordinator']['port']})")
        print("=" * 60)
        
        results = {}
        for endpoint in self.services['coordinator']['endpoints']:
            success, message = self.test_service(
                'coordinator', 
                self.services['coordinator']['port'], 
                endpoint
            )
            results[endpoint] = {'success': success, 'message': message}
            print(f"  {endpoint:<20} {message}")
        
        return results
    
    def test_dashboard(self) -> Dict:
        """Test dashboard frontend"""
        print(f"\n🔍 Testing {self.services['dashboard']['name']} (Port {self.services['dashboard']['port']})")
        print("=" * 60)
        
        results = {}
        for endpoint in self.services['dashboard']['endpoints']:
            success, message = self.test_service(
                'dashboard', 
                self.services['dashboard']['port'], 
                endpoint
            )
            results[endpoint] = {'success': success, 'message': message}
            print(f"  {endpoint:<20} {message}")
        
        return results
    
    def test_gpu_agents(self) -> Dict:
        """Test GPU agent endpoints"""
        print(f"\n🔍 Testing {self.services['gpu_agents']['name']}")
        print("=" * 60)
        
        results = {}
        base_port = self.services['gpu_agents']['base_port']
        count = self.services['gpu_agents']['count']
        
        for i in range(count):
            port = base_port + i
            print(f"\n  GPU Agent {i} (Port {port}):")
            
            agent_results = {}
            for endpoint in self.services['gpu_agents']['endpoints']:
                success, message = self.test_service('gpu_agent', port, endpoint)
                agent_results[endpoint] = {'success': success, 'message': message}
                print(f"    {endpoint:<20} {message}")
            
            results[f'gpu_{i}'] = agent_results
        
        return results
    
    def test_system_status(self) -> Dict:
        """Test the main system status endpoint"""
        print(f"\n🔍 Testing System Status via Coordinator")
        print("=" * 60)
        
        try:
            url = f"http://localhost:{self.services['coordinator']['port']}/api/system/status"
            response = requests.get(url, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                print("✅ System Status Retrieved Successfully!")
                print(f"  Active GPUs: {data.get('active_gpus', 'N/A')}")
                print(f"  Total Jobs: {data.get('total_jobs', 'N/A')}")
                print(f"  System Health: {data.get('system_health', 'N/A')}")
                
                # Test job submission
                print(f"\n  Testing Job Submission...")
                job_data = {
                    "prompt": "Test prompt for tunnel verification",
                    "target_score": 0.85,
                    "max_episodes": 3
                }
                
                job_response = requests.post(
                    f"http://localhost:{self.services['coordinator']['port']}/api/jobs/submit",
                    json=job_data,
                    timeout=10
                )
                
                if job_response.status_code in [200, 201]:
                    job_result = job_response.json()
                    print(f"  ✅ Job submitted successfully!")
                    print(f"    Job ID: {job_result.get('job_id', 'N/A')}")
                    print(f"    Status: {job_result.get('status', 'N/A')}")
                else:
                    print(f"  ❌ Job submission failed: {job_response.status_code}")
                
                return {'success': True, 'data': data}
                
            else:
                print(f"❌ Failed to get system status: {response.status_code}")
                return {'success': False, 'error': f"HTTP {response.status_code}"}
                
        except Exception as e:
            print(f"❌ Error testing system status: {str(e)}")
            return {'success': False, 'error': str(e)}
    
    def run_all_tests(self) -> Dict:
        """Run all connectivity tests"""
        print("🚀 Tunnel Connectivity Test for Distributed RL System")
        print("=" * 70)
        print(f"Testing local ports mapped to remote services...")
        print(f"Make sure SSH tunnels are active before running tests!")
        print()
        
        # Test coordinator
        self.results['coordinator'] = self.test_coordinator()
        
        # Test dashboard
        self.results['dashboard'] = self.test_dashboard()
        
        # Test GPU agents
        self.results['gpu_agents'] = self.test_gpu_agents()
        
        # Test system status
        self.results['system_status'] = self.test_system_status()
        
        return self.results
    
    def generate_summary(self) -> str:
        """Generate a summary report"""
        print(f"\n📊 Test Summary")
        print("=" * 50)
        
        total_tests = 0
        passed_tests = 0
        
        for service_name, service_results in self.results.items():
            if service_name == 'system_status':
                continue
                
            print(f"\n{service_name.upper()}:")
            
            if service_name == 'gpu_agents':
                for gpu_name, gpu_results in service_results.items():
                    print(f"  {gpu_name}:")
                    for endpoint, result in gpu_results.items():
                        total_tests += 1
                        if result['success']:
                            passed_tests += 1
                        print(f"    {endpoint}: {result['message']}")
            else:
                for endpoint, result in service_results.items():
                    total_tests += 1
                    if result['success']:
                        passed_tests += 1
                    print(f"  {endpoint}: {result['message']}")
        
        success_rate = (passed_tests / total_tests * 100) if total_tests > 0 else 0
        print(f"\n📈 Overall Results:")
        print(f"  Total Tests: {total_tests}")
        print(f"  Passed: {passed_tests}")
        print(f"  Failed: {total_tests - passed_tests}")
        print(f"  Success Rate: {success_rate:.1f}%")
        
        if success_rate >= 80:
            print(f"\n🎉 Tunnel setup is working well!")
        elif success_rate >= 50:
            print(f"\n⚠️  Tunnel setup has some issues - check SSH connections")
        else:
            print(f"\n❌ Tunnel setup has major issues - verify SSH configuration")
        
        return f"Success Rate: {success_rate:.1f}%"

def main():
    """Main function"""
    tester = TunnelTester()
    
    try:
        # Run all tests
        results = tester.run_all_tests()
        
        # Generate summary
        summary = tester.generate_summary()
        
        # Save results to file
        with open('tunnel_test_results.json', 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\n💾 Detailed results saved to: tunnel_test_results.json")
        
    except KeyboardInterrupt:
        print(f"\n\n⏹️  Testing interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Unexpected error: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()


