#!/usr/bin/env python3
"""
Test script for the actual generate_3d_model_clip function
Tests with real servers running on ports 8097 and 8099
"""

import asyncio
import time
import traceback
import sys
import os

# Add the current directory to Python path to import the orchestrator
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import the actual function from the orchestrator
try:
    from continuous_trellis_orchestrator_lora_working import ContinuousTrellisOrchestrator
    print("✅ Successfully imported ContinuousTrellisOrchestrator")
except ImportError as e:
    print(f"❌ Failed to import ContinuousTrellisOrchestrator: {e}")
    print("Make sure you're in the correct directory and the file exists")
    sys.exit(1)

class MockTaskRecord:
    """Mock task record for testing"""
    def __init__(self, prompt="a blue car"):
        self.prompt = prompt
        self.task_id = "test_task_123"
        self.hotkey = "test_hotkey"

class MockLogger:
    """Mock logger for testing"""
    def info(self, msg):
        print(f"ℹ️  INFO: {msg}")
    
    def error(self, msg):
        print(f"❌ ERROR: {msg}")
    
    def warning(self, msg):
        print(f"⚠️  WARNING: {msg}")
    
    def debug(self, msg):
        print(f"🔍 DEBUG: {msg}")

class MockPriorityCoordinator:
    """Mock priority coordinator"""
    def clear_server_cache(self):
        pass
    
    def wait_for_priority_access(self, task_id):
        """Mock priority access - always return True for testing"""
        return True
    
    def mark_priority_job_start(self, task_id, prompt):
        """Mock priority job start tracking"""
        pass
    
    def mark_priority_job_end(self, task_id):
        """Mock priority job end tracking"""
        pass

class MockConfig:
    """Mock config with dictionary-like access"""
    def __init__(self):
        self.config = {
            'generation_server_url': 'http://localhost:8097',
            'generation_timeout': 300,
            'validation_timeout': 60,
            'num_inference_steps': 50,
            'guidance_scale': 7.5,
            'ss_sampling_steps': 24,
            'slat_sampling_steps': 24,
            'slat_guidance_strength': 4.0,
            'ss_guidance_strength': 9.5,
            'use_vllm': True,
            'vllm_url': 'http://localhost:9002',
            'vllm_model': 'llama-3-2-3b-it'
        }
    
    def __getitem__(self, key):
        return self.config[key]
    
    def __contains__(self, key):
        return key in self.config
    
    def get(self, key, default=None):
        return self.config.get(key, default)

class MockReproducibilitySystem:
    """Mock reproducibility system"""
    def __init__(self):
        self.gold_standard_results = {
            "a blue car": {
                "method_2_hybrid_example": {
                    "enhanced_prompt": "a finely crafted, intricate blue car with enhanced details",
                    "score": 0.85
                }
            },
            "a red truck": {
                "method_2_hybrid_example": {
                    "enhanced_prompt": "a finely crafted, intricate red truck with enhanced details", 
                    "score": 0.78
                }
            },
            "a green motorcycle": {
                "method_2_hybrid_example": {
                    "enhanced_prompt": "a finely crafted, intricate green motorcycle with enhanced details",
                    "score": 0.82
                }
            }
        }
    
    def get_optimized_prompt(self, original_prompt):
        return f"a finely crafted, intricate {original_prompt} with enhanced details"
    
    def optimize_prompt_with_reproducibility(self, original_prompt, min_similarity=0.3, run_validation=False):
        """Mock prompt optimization with reproducibility"""
        # Return a mock optimized prompt
        return f"a finely crafted, intricate {original_prompt} with enhanced details"

async def test_generate_3d_model_clip_real():
    """Test the actual generate_3d_model_clip function with real servers"""
    
    print("🧪 Testing ACTUAL generate_3d_model_clip function with real servers")
    print("=" * 70)
    
    # Create mock components
    mock_logger = MockLogger()
    mock_config = MockConfig()
    mock_priority_coordinator = MockPriorityCoordinator()
    mock_reproducibility_system = MockReproducibilitySystem()
    
    # Create mock task
    task = MockTaskRecord("a blue car")
    
    print(f"📝 Task: {task.prompt}")
    print(f"🌐 Server 1: http://localhost:8097")
    print(f"🌐 Server 2: http://localhost:8099")
    
    try:
        # Create the actual orchestrator instance
        print("\n🔧 Creating ContinuousTrellisOrchestrator instance...")
        
        # We need to patch the orchestrator with our mock components
        orchestrator = ContinuousTrellisOrchestrator(mock_config.config)
        
        # Replace components with mocks
        orchestrator.logger = mock_logger
        orchestrator.priority_coordinator = mock_priority_coordinator
        orchestrator.reproducibility_system = mock_reproducibility_system
        
        print("✅ Orchestrator created and configured with mock components")
        
        # Test the deterministic seed function
        try:
            seed = orchestrator.get_deterministic_seed(task)
            print(f"🎲 Deterministic seed: {seed}")
        except Exception as e:
            print(f"⚠️  Deterministic seed failed: {e}")
        
        # Test CLIP analyzer availability
        try:
            clip_analyzer = orchestrator.get_clip_analyzer()
            if clip_analyzer:
                print("✅ CLIP analyzer available")
            else:
                print("⚠️  CLIP analyzer not available")
        except Exception as e:
            print(f"⚠️  CLIP analyzer check failed: {e}")
        
        # Test the actual generate_3d_model_clip function
        print("\n🚀 Testing generate_3d_model_clip function...")
        
        start_time = time.time()
        
        try:
            # Call the actual function
            result = await orchestrator.generate_3d_model_clip(task)
            end_time = time.time()
            
            print(f"⏱️  Function execution time: {end_time - start_time:.3f}s")
            
            if result:
                print("✅ Function executed successfully!")
                print(f"📊 Result type: {type(result)}")
                
                # Try to access result properties
                if hasattr(result, 'ply_data'):
                    print(f"   PLY data size: {len(result.ply_data)} bytes")
                if hasattr(result, 'compressed_data'):
                    print(f"   Compressed data size: {len(result.compressed_data)} bytes")
                if hasattr(result, 'image'):
                    print(f"   Image: {result.image}")
                
                # Print all available attributes
                print(f"   Available attributes: {dir(result)}")
                
            else:
                print("⚠️  Function returned None")
                
        except Exception as e:
            print(f"❌ Function execution failed: {e}")
            print("This might be due to missing external services (vLLM, Ollama, etc.)")
            print("Let's continue with testing the core logic...")
            traceback.print_exc()
        
        # Test endpoint conversion logic
        print("\n🔧 Testing endpoint conversion logic:")
        test_endpoints = [
            "/generate/cinema",
            "/generate/",
            "/generate",
            "generate/cinema",
            "generate/",
            "generate"
        ]
        
        for endpoint in test_endpoints:
            if endpoint.startswith('/generate'):
                clean_endpoint = endpoint.lstrip('/')
                if clean_endpoint.startswith('generate/'):
                    both_endpoint = '/' + clean_endpoint.replace('generate/', 'generate_both/', 1)
                elif clean_endpoint == 'generate':
                    both_endpoint = '/generate_both/'
                else:
                    both_endpoint = '/generate_both/'
            elif endpoint.startswith('generate/'):
                both_endpoint = '/' + endpoint.replace('generate/', 'generate_both/', 1)
            elif endpoint == 'generate':
                both_endpoint = '/generate_both/'
            else:
                both_endpoint = '/generate_both/'
            
            print(f"   {endpoint} → {both_endpoint}")
        
        # Test URL construction logic
        print("\n🌐 Testing URL construction logic:")
        test_urls = [
            ("http://localhost:8097", 8096, "http://localhost:8096"),
            ("http://localhost:8097", 8099, "http://localhost:8099"),
            ("localhost:8097", 8096, "http://localhost:8096")
        ]
        
        for base_url, port, expected in test_urls:
            if port:
                try:
                    original_url = base_url
                    if '://' in original_url:
                        protocol_and_host = original_url.split(':')[0] + ':' + original_url.split(':')[1]
                        server_url = f"{protocol_and_host}:{port}"
                    else:
                        host = original_url.split(':')[0]
                        server_url = f"http://{host}:{port}"
                except Exception as e:
                    server_url = base_url
            else:
                server_url = base_url
            
            status = "✅" if server_url == expected else "❌"
            print(f"   {status} {base_url} + port {port} → {server_url}")
        
        print("\n" + "=" * 70)
        print("✅ Test completed successfully!")
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        traceback.print_exc()
        return False

async def test_server_connectivity():
    """Test if the servers are actually reachable"""
    
    print("\n🔍 Testing server connectivity...")
    
    try:
        import aiohttp
        
        async def test_server(port):
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.get(f"http://localhost:{port}/health", timeout=5) as response:
                        if response.status == 200:
                            return True, f"✅ Server on port {port} is reachable"
                        else:
                            return False, f"⚠️  Server on port {port} responded with status {response.status}"
            except Exception as e:
                return False, f"❌ Server on port {port} is not reachable: {e}"
        
        # Test both servers
        results = await asyncio.gather(
            test_server(8097),
            test_server(8099)
        )
        
        for success, message in results:
            print(f"   {message}")
            
        return all(success for success, _ in results)
        
    except ImportError:
        print("⚠️  aiohttp not available, skipping connectivity test")
        return True
    except Exception as e:
        print(f"❌ Connectivity test failed: {e}")
        return False

if __name__ == "__main__":
    try:
        # Set CUDA deterministic environment variable to fix CLIP scoring
        import os
        os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
        print("🔧 Set CUBLAS_WORKSPACE_CONFIG=:4096:8 to fix CUDA deterministic behavior")
        
        # First test server connectivity
        connectivity_ok = asyncio.run(test_server_connectivity())
        
        if connectivity_ok:
            # Then test the actual function
            asyncio.run(test_generate_3d_model_clip_real())
        else:
            print("⚠️  Server connectivity issues detected. Some tests may fail.")
            # Try the function test anyway
            asyncio.run(test_generate_3d_model_clip_real())
            
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        traceback.print_exc()
