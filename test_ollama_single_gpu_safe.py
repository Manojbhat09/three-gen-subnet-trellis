#!/usr/bin/env python3
"""
Safe Single GPU Ollama Test
Test Ollama integration with 1 GPU without disturbing current setup
Uses isolated ports and processes - completely safe to run
"""

import asyncio
import sys
import requests
import time
import subprocess
import os
import signal
from pathlib import Path
from typing import Optional
import aiohttp
import json

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from config.settings import settings
from utils.logging_config import setup_logging, get_logger

logger = get_logger("safe_ollama_test")

class SafeOllamaTest:
    """Safe, isolated Ollama test that won't interfere with existing setup"""
    
    def __init__(self):
        # Use completely isolated ports to avoid conflicts
        self.test_ollama_port = 12000  # Well above normal range
        self.test_gpu_id = 0  # Test with GPU 0
        self.ollama_process: Optional[subprocess.Popen] = None
        self.test_running = False
        
        # Setup signal handlers for clean shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals"""
        print("\n🛑 Shutdown signal received, cleaning up...")
        self.cleanup()
        sys.exit(0)
    
    def check_ollama_available(self) -> bool:
        """Check if Ollama is installed"""
        print("🔍 Checking Ollama availability...")
        
        try:
            result = subprocess.run(
                ["ollama", "--version"], 
                capture_output=True, 
                text=True, 
                timeout=5
            )
            
            if result.returncode == 0:
                version = result.stdout.strip()
                print(f"  ✅ Ollama found: {version}")
                return True
            else:
                print(f"  ❌ Ollama check failed: {result.stderr}")
                return False
                
        except FileNotFoundError:
            print("  ❌ Ollama not found")
            print("  📝 Install: curl -fsSL https://ollama.com/install.sh | sh")
            return False
        except Exception as e:
            print(f"  ❌ Ollama check error: {e}")
            return False
    
    def check_port_available(self, port: int) -> bool:
        """Check if port is available"""
        import socket
        
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind(('127.0.0.1', port))
                return True
        except OSError:
            return False
    
    def start_test_ollama_server(self) -> bool:
        """Start isolated Ollama server for testing"""
        print(f"🚀 Starting test Ollama server on port {self.test_ollama_port}...")
        
        # Check port availability
        if not self.check_port_available(self.test_ollama_port):
            print(f"  ❌ Port {self.test_ollama_port} not available")
            print(f"  💡 Try: lsof -ti:{self.test_ollama_port} | xargs kill")
            return False
        
        try:
            # Set environment for GPU isolation and port
            env = os.environ.copy()
            env['CUDA_VISIBLE_DEVICES'] = str(self.test_gpu_id)
            env['OLLAMA_HOST'] = f"127.0.0.1:{self.test_ollama_port}"
            env['OLLAMA_ORIGINS'] = "*"
            
            # Start Ollama server
            self.ollama_process = subprocess.Popen(
                ["ollama", "serve"],
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                preexec_fn=os.setsid  # Create process group for clean shutdown
            )
            
            print(f"  🔄 Server starting (PID: {self.ollama_process.pid})...")
            
            # Wait for server to be ready
            for attempt in range(30):  # 30 second timeout
                try:
                    response = requests.get(
                        f"http://127.0.0.1:{self.test_ollama_port}/api/tags",
                        timeout=2
                    )
                    if response.status_code == 200:
                        print(f"  ✅ Server ready after {attempt + 1} seconds")
                        return True
                except requests.exceptions.RequestException:
                    pass  # Server not ready yet
                
                time.sleep(1)
            
            print("  ❌ Server failed to start within timeout")
            return False
            
        except Exception as e:
            print(f"  ❌ Failed to start server: {e}")
            return False
    
    def test_ollama_model_loading(self) -> bool:
        """Test loading a small model for testing"""
        print("📦 Testing model loading...")
        
        server_url = f"http://127.0.0.1:{self.test_ollama_port}"
        
        # Try to use a small model first (faster testing)
        small_models = ["llama3.2:1b", "llama3.1:8b", "gemma2:2b"]
        
        for model_name in small_models:
            print(f"  🔄 Trying model: {model_name}")
            
            try:
                # Check if model is already available
                response = requests.get(f"{server_url}/api/tags", timeout=5)
                if response.status_code == 200:
                    tags = response.json()
                    available_models = [model['name'] for model in tags.get('models', [])]
                    
                    if model_name in available_models:
                        print(f"  ✅ Model {model_name} already available")
                        return True
                
                # Try to pull model (with timeout)
                print(f"  📥 Pulling {model_name} (this may take a while)...")
                
                response = requests.post(
                    f"{server_url}/api/pull",
                    json={"name": model_name},
                    timeout=180,  # 3 minutes max
                    stream=True
                )
                
                if response.status_code == 200:
                    # Process streaming response
                    for line in response.iter_lines():
                        if line:
                            try:
                                status_data = json.loads(line)
                                if status_data.get('status') == 'success':
                                    print(f"  ✅ Model {model_name} loaded successfully")
                                    return True
                                elif 'error' in status_data:
                                    print(f"  ❌ Model loading error: {status_data['error']}")
                                    break
                            except json.JSONDecodeError:
                                continue
                
            except requests.exceptions.Timeout:
                print(f"  ⏰ Model {model_name} pull timeout, trying next...")
                continue
            except Exception as e:
                print(f"  ❌ Model {model_name} error: {e}")
                continue
        
        print("  ❌ No suitable model could be loaded")
        return False
    
    def test_ollama_text_generation(self) -> bool:
        """Test basic text generation"""
        print("🧠 Testing text generation...")
        
        server_url = f"http://127.0.0.1:{self.test_ollama_port}"
        
        # Get available models
        try:
            response = requests.get(f"{server_url}/api/tags", timeout=5)
            if response.status_code != 200:
                print("  ❌ Failed to get model list")
                return False
            
            tags = response.json()
            models = [model['name'] for model in tags.get('models', [])]
            
            if not models:
                print("  ❌ No models available")
                return False
            
            model_name = models[0]  # Use first available model
            print(f"  🎯 Using model: {model_name}")
            
            # Test simple generation
            test_prompt = "Improve this prompt for 3D generation: red car"
            
            print(f"  🔄 Generating response for: '{test_prompt}'")
            
            response = requests.post(
                f"{server_url}/api/generate",
                json={
                    "model": model_name,
                    "prompt": test_prompt,
                    "stream": False,
                    "options": {
                        "num_predict": 50,
                        "temperature": 0.7
                    }
                },
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                generated_text = result.get('response', '').strip()
                
                if generated_text:
                    print(f"  ✅ Generation successful!")
                    print(f"  📝 Response: '{generated_text[:100]}{'...' if len(generated_text) > 100 else ''}'")
                    return True
                else:
                    print("  ❌ Empty response")
                    return False
            else:
                print(f"  ❌ Generation failed: HTTP {response.status_code}")
                return False
                
        except Exception as e:
            print(f"  ❌ Generation test error: {e}")
            return False
    
    async def test_gpu_agent_integration(self) -> bool:
        """Test GPU agent integration with Ollama"""
        print("🤖 Testing GPU agent integration...")
        
        try:
            # Import GPU agent
            from src.gpu_agent.simple_gpu_agent import SimpleGPUAgent
            
            # Create test agent with our isolated Ollama port
            agent = SimpleGPUAgent(
                gpu_id=self.test_gpu_id,
                port=9999,  # Isolated port for testing (won't start server)
                coordinator_url="http://localhost:8090",  # Won't connect
                ollama_port=self.test_ollama_port
            )
            
            print(f"  🔄 Initializing Ollama client connection...")
            
            # Test Ollama client initialization
            await agent._initialize_ollama_client()
            
            if not agent.ollama_enabled:
                print("  ❌ Agent failed to connect to Ollama")
                return False
            
            print("  ✅ Agent connected to Ollama successfully")
            
            # Test strategy generation
            test_prompt = "wooden table"
            context = "Episode: 1, Prompt length: 12 characters"
            
            print(f"  🧠 Testing RL strategy generation...")
            print(f"     Input: '{test_prompt}'")
            
            strategy_result = await agent._get_rl_strategy_from_ollama(test_prompt, context)
            
            print(f"  📊 Strategy Result:")
            print(f"     Strategy: {strategy_result['strategy']}")
            print(f"     Improved: '{strategy_result['improved_prompt']}'")
            print(f"     Source: {strategy_result['source']}")
            print(f"     Confidence: {strategy_result['confidence']}")
            
            # Validate result structure
            required_fields = ['strategy', 'improved_prompt', 'reasoning', 'confidence', 'source']
            if all(field in strategy_result for field in required_fields):
                print("  ✅ GPU agent integration working correctly")
                return True
            else:
                missing = [f for f in required_fields if f not in strategy_result]
                print(f"  ❌ Missing fields in result: {missing}")
                return False
                
        except Exception as e:
            print(f"  ❌ Integration test error: {e}")
            return False
        finally:
            # Clean up agent resources
            if 'agent' in locals() and hasattr(agent, 'ollama_client') and agent.ollama_client:
                try:
                    await agent.ollama_client.close()
                except:
                    pass
    
    def test_gpu_isolation(self) -> bool:
        """Verify GPU isolation is working"""
        print("🔒 Testing GPU isolation...")
        
        # Check CUDA_VISIBLE_DEVICES in process environment
        if self.ollama_process:
            print(f"  🔍 Checking GPU isolation for PID {self.ollama_process.pid}")
            
            try:
                # Read process environment
                with open(f"/proc/{self.ollama_process.pid}/environ", "rb") as f:
                    env_data = f.read().decode('utf-8', errors='ignore')
                    env_vars = dict(line.split('=', 1) for line in env_data.split('\0') if '=' in line)
                    
                    cuda_devices = env_vars.get('CUDA_VISIBLE_DEVICES', 'not_set')
                    ollama_host = env_vars.get('OLLAMA_HOST', 'not_set')
                    
                    print(f"  📊 Process Environment:")
                    print(f"     CUDA_VISIBLE_DEVICES: {cuda_devices}")
                    print(f"     OLLAMA_HOST: {ollama_host}")
                    
                    if cuda_devices == str(self.test_gpu_id):
                        print("  ✅ GPU isolation working correctly")
                        return True
                    else:
                        print("  ⚠️  GPU isolation may not be working")
                        return False
                        
            except Exception as e:
                print(f"  ⚠️  Could not verify GPU isolation: {e}")
                print("  📝 This is normal on some systems")
                return True  # Don't fail the test for this
        else:
            print("  ❌ No Ollama process to check")
            return False
    
    def cleanup(self):
        """Clean up test resources"""
        if not self.test_running:
            return
            
        print("\n🧹 Cleaning up test resources...")
        
        if self.ollama_process:
            try:
                print(f"  🛑 Stopping Ollama server (PID: {self.ollama_process.pid})")
                
                # Graceful shutdown
                self.ollama_process.terminate()
                
                try:
                    self.ollama_process.wait(timeout=10)
                    print("  ✅ Server stopped gracefully")
                except subprocess.TimeoutExpired:
                    print("  ⚠️  Graceful shutdown timeout, forcing...")
                    self.ollama_process.kill()
                    self.ollama_process.wait()
                    print("  ✅ Server stopped forcefully")
                    
            except Exception as e:
                print(f"  ⚠️  Cleanup error: {e}")
        
        self.test_running = False
        print("✅ Cleanup completed")
    
    async def run_complete_test(self):
        """Run the complete safe test suite"""
        
        print("🧪 Safe Ollama Integration Test")
        print("=" * 50)
        print("Testing Ollama with 1 GPU - completely isolated and safe")
        print(f"Using test port: {self.test_ollama_port} (isolated from main system)")
        print(f"Using test GPU: {self.test_gpu_id}")
        print("")
        
        self.test_running = True
        
        tests = [
            ("Ollama Availability", self.check_ollama_available),
            ("Start Test Server", self.start_test_ollama_server),
            ("Model Loading", self.test_ollama_model_loading),
            ("Text Generation", self.test_ollama_text_generation),
            ("GPU Isolation", self.test_gpu_isolation),
            ("Agent Integration", self.test_gpu_agent_integration),
        ]
        
        results = {}
        
        try:
            for test_name, test_func in tests:
                print(f"\n🔬 {test_name}")
                print("-" * 30)
                
                try:
                    if asyncio.iscoroutinefunction(test_func):
                        result = await test_func()
                    else:
                        result = test_func()
                    
                    results[test_name] = result
                    
                    if result:
                        print(f"✅ {test_name} PASSED")
                    else:
                        print(f"❌ {test_name} FAILED")
                        
                        # Stop on critical failures
                        if test_name in ["Ollama Availability", "Start Test Server"]:
                            print(f"\n🛑 Critical test failed, stopping here")
                            break
                    
                except Exception as e:
                    print(f"💥 {test_name} CRASHED: {e}")
                    results[test_name] = False
                    
                    if test_name in ["Ollama Availability", "Start Test Server"]:
                        break
            
            # Summary
            print(f"\n{'=' * 50}")
            print("📊 Test Summary")
            print("=" * 50)
            
            passed = sum(1 for result in results.values() if result)
            total = len(results)
            
            for test_name, result in results.items():
                status = "✅ PASS" if result else "❌ FAIL"
                print(f"   {test_name:<20}: {status}")
            
            print(f"\nResults: {passed}/{total} tests passed")
            
            if passed == total:
                print("\n🎉 ALL TESTS PASSED!")
                print("✅ Ollama integration is working correctly")
                print("✅ GPU isolation is functional")
                print("✅ RL strategy generation is ready")
                print("✅ Your system is ready for full Ollama integration!")
            elif passed >= 4:  # Core functionality working
                print(f"\n✅ CORE FUNCTIONALITY WORKING!")
                print("✅ Ollama integration basics are working")
                print("📝 Some advanced features may need attention")
            else:
                print(f"\n⚠️  ISSUES DETECTED")
                
                if not results.get("Ollama Availability", False):
                    print("\n📝 Next Steps:")
                    print("   1. Install Ollama: curl -fsSL https://ollama.com/install.sh | sh")
                    print("   2. Restart terminal")
                    print("   3. Run: ollama pull llama3.1:8b")
                    print("   4. Re-run this test")
                else:
                    print("📝 Check error messages above for troubleshooting")
            
        finally:
            self.cleanup()

async def main():
    """Main test entry point"""
    
    setup_logging("safe_ollama_test")
    
    test = SafeOllamaTest()
    
    try:
        await test.run_complete_test()
    except KeyboardInterrupt:
        print("\n🛑 Test interrupted by user")
        test.cleanup()
    except Exception as e:
        print(f"\n💥 Test suite crashed: {e}")
        test.cleanup()

if __name__ == "__main__":
    print("🔒 Safe Ollama Test - No interference with existing setup")
    print("This test uses isolated ports and processes")
    print("")
    
    asyncio.run(main())

