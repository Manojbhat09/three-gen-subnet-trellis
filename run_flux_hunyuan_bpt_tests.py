#!/usr/bin/env python3
"""
Master Test Runner for Flux-Hunyuan-BPT System
Purpose: Orchestrate all tests and setup for the enhanced 3D generation system
"""

import asyncio
import subprocess
import sys
import time
import os
import signal
from pathlib import Path
from typing import List, Dict, Any, Optional
import traceback

# Configuration
SERVER_SCRIPT = "flux_hunyuan_bpt_generation_server.py"
MINER_SCRIPT = "flux_hunyuan_bpt_miner.py"
SERVER_TEST_SCRIPT = "test_flux_hunyuan_bpt_server.py"
MINER_TEST_SCRIPT = "test_flux_hunyuan_bpt_miner.py"

SERVER_PORT = 8095
SERVER_URL = f"http://127.0.0.1:{SERVER_PORT}"

class TestOrchestrator:
    def __init__(self):
        self.server_process = None
        self.test_results = {}
        
    def check_dependencies(self) -> bool:
        """Check if all required files exist."""
        print("🔍 Checking dependencies...")
        
        required_files = [
            SERVER_SCRIPT,
            SERVER_TEST_SCRIPT,
            MINER_TEST_SCRIPT,
            "Hunyuan3D-2/flux_hunyuan_bpt_demo.py"  # Original demo
        ]
        
        missing_files = []
        for file_path in required_files:
            if not Path(file_path).exists():
                missing_files.append(file_path)
        
        if missing_files:
            print("❌ Missing required files:")
            for file_path in missing_files:
                print(f"  - {file_path}")
            return False
        
        print("✅ All required files found")
        return True
    
    def check_python_packages(self) -> bool:
        """Check if required Python packages are installed."""
        print("\n🐍 Checking Python packages...")
        
        # Map of package names to their import names
        required_packages = {
            "torch": "torch",
            "diffusers": "diffusers", 
            "transformers": "transformers",
            "trimesh": "trimesh",
            "fastapi": "fastapi",
            "uvicorn": "uvicorn",
            "aiohttp": "aiohttp",
            "bittensor": "bittensor",
            "pyspz": "pyspz",
            "numpy": "numpy",
            "pillow": "PIL"  # pillow package imports as PIL
        }
        
        missing_packages = []
        for package_name, import_name in required_packages.items():
            try:
                __import__(import_name)
            except ImportError:
                missing_packages.append(package_name)
        
        if missing_packages:
            print("❌ Missing required packages:")
            for package in missing_packages:
                print(f"  - {package}")
            print("\nInstall missing packages with:")
            print(f"pip install {' '.join(missing_packages)}")
            return False
        
        print("✅ All required packages found")
        return True
    
    async def start_generation_server(self) -> bool:
        """Start the generation server."""
        print(f"\n🚀 Starting generation server on port {SERVER_PORT}...")
        
        try:
            # Start server process
            self.server_process = subprocess.Popen(
                [sys.executable, SERVER_SCRIPT],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            
            # Wait for server to start
            print("⏳ Waiting for server to initialize...")
            
            import aiohttp
            
            async def wait_for_server():
                for attempt in range(30):  # 30 attempts, 2 seconds each = 1 minute
                    try:
                        async with aiohttp.ClientSession() as session:
                            async with session.get(f"{SERVER_URL}/health/", timeout=2) as response:
                                if response.status == 200:
                                    return True
                    except:
                        pass
                    
                    await asyncio.sleep(2)
                    print(f"  Attempt {attempt + 1}/30...")
                
                return False
            
            server_ready = await wait_for_server()
            
            if server_ready:
                print("✅ Generation server started successfully")
                return True
            else:
                print("❌ Generation server failed to start within timeout")
                self.stop_generation_server()
                return False
                
        except Exception as e:
            print(f"❌ Failed to start generation server: {e}")
            return False
    
    def stop_generation_server(self):
        """Stop the generation server."""
        if self.server_process:
            print("\n🛑 Stopping generation server...")
            try:
                self.server_process.terminate()
                self.server_process.wait(timeout=10)
                print("✅ Generation server stopped")
            except subprocess.TimeoutExpired:
                print("⚠️  Server didn't stop gracefully, force killing...")
                self.server_process.kill()
                self.server_process.wait()
                print("✅ Generation server force stopped")
            except Exception as e:
                print(f"⚠️  Error stopping server: {e}")
            finally:
                self.server_process = None
    
    async def run_server_tests(self) -> bool:
        """Run server tests."""
        print("\n🧪 Running server tests...")
        
        try:
            process = await asyncio.create_subprocess_exec(
                sys.executable, SERVER_TEST_SCRIPT,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            stdout, stderr = await process.communicate()
            
            success = process.returncode == 0
            
            print("📊 Server test results:")
            if stdout:
                print(stdout.decode())
            if stderr and not success:
                print("Errors:")
                print(stderr.decode())
            
            self.test_results['server_tests'] = {
                'success': success,
                'returncode': process.returncode,
                'stdout': stdout.decode() if stdout else '',
                'stderr': stderr.decode() if stderr else ''
            }
            
            return success
            
        except Exception as e:
            print(f"❌ Failed to run server tests: {e}")
            traceback.print_exc()
            return False
    
    async def run_miner_tests(self) -> bool:
        """Run miner tests."""
        print("\n🧪 Running miner tests...")
        
        try:
            process = await asyncio.create_subprocess_exec(
                sys.executable, MINER_TEST_SCRIPT,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            stdout, stderr = await process.communicate()
            
            success = process.returncode == 0
            
            print("📊 Miner test results:")
            if stdout:
                print(stdout.decode())
            if stderr and not success:
                print("Errors:")
                print(stderr.decode())
            
            self.test_results['miner_tests'] = {
                'success': success,
                'returncode': process.returncode,
                'stdout': stdout.decode() if stdout else '',
                'stderr': stderr.decode() if stderr else ''
            }
            
            return success
            
        except Exception as e:
            print(f"❌ Failed to run miner tests: {e}")
            traceback.print_exc()
            return False
    
    def run_original_demo_test(self) -> bool:
        """Run the original Flux-Hunyuan-BPT demo as a baseline test."""
        print("\n🎯 Running original demo test...")
        
        try:
            # Change to Hunyuan3D-2 directory
            original_dir = os.getcwd()
            hunyuan_dir = Path("Hunyuan3D-2")
            
            if not hunyuan_dir.exists():
                print("❌ Hunyuan3D-2 directory not found")
                return False
            
            os.chdir(hunyuan_dir)
            
            # Create a simple test version of the demo
            test_demo_content = '''
import sys
import os

# Add parent directory to path to import from the demo
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from flux_hunyuan_bpt_demo import text_to_3d_with_bpt
    
    print("🧪 Running quick demo test...")
    print("Generating: a simple red cube")
    
    # Quick test with simple prompt
    text_to_3d_with_bpt(
        prompt="a simple red cube", 
        output_dir="demo_test_outputs", 
        seed=42, 
        use_bpt=False  # Disable BPT for faster testing
    )
    
    print("✅ Demo test completed successfully!")
    
except Exception as e:
    print(f"❌ Demo test failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
'''
            
            # Write test demo
            with open("test_demo.py", "w") as f:
                f.write(test_demo_content)
            
            # Run test demo
            result = subprocess.run(
                [sys.executable, "test_demo.py"],
                capture_output=True,
                text=True,
                timeout=300  # 5 minute timeout
            )
            
            success = result.returncode == 0
            
            print("📊 Original demo test results:")
            if result.stdout:
                print(result.stdout)
            if result.stderr and not success:
                print("Errors:")
                print(result.stderr)
            
            # Clean up
            os.chdir(original_dir)
            
            self.test_results['demo_test'] = {
                'success': success,
                'returncode': result.returncode,
                'stdout': result.stdout,
                'stderr': result.stderr
            }
            
            return success
            
        except subprocess.TimeoutExpired:
            print("❌ Demo test timed out")
            os.chdir(original_dir)
            return False
        except Exception as e:
            print(f"❌ Failed to run demo test: {e}")
            os.chdir(original_dir)
            traceback.print_exc()
            return False
    
    def print_final_summary(self):
        """Print final test summary."""
        print("\n" + "="*60)
        print("🏁 FINAL TEST SUMMARY")
        print("="*60)
        
        total_tests = len(self.test_results)
        passed_tests = sum(1 for result in self.test_results.values() if result['success'])
        
        print(f"Total Test Suites: {total_tests}")
        print(f"Passed: {passed_tests} ✅")
        print(f"Failed: {total_tests - passed_tests} ❌")
        
        # Individual test results
        for test_name, result in self.test_results.items():
            status = "✅ PASSED" if result['success'] else "❌ FAILED"
            print(f"  {test_name}: {status}")
        
        # Overall status
        overall_success = passed_tests == total_tests
        print(f"\n{'🎉 OVERALL: SUCCESS' if overall_success else '⚠️  OVERALL: ISSUES DETECTED'}")
        
        if overall_success:
            print("\n✅ All systems are ready!")
            print("\n🚀 Next Steps:")
            print("1. Start the generation server:")
            print(f"   python {SERVER_SCRIPT}")
            print("2. In another terminal, start the miner:")
            print(f"   python {MINER_SCRIPT}")
            print("3. Monitor logs in the ./logs directory")
            print("4. Check generated assets in the output directories")
        else:
            print("\n❌ Some tests failed. Please review the issues above.")
            print("\n🔧 Troubleshooting:")
            print("- Check GPU memory availability")
            print("- Ensure all dependencies are installed")
            print("- Verify model weights are downloaded")
            print("- Check network connectivity for model downloads")
    
    async def run_all_tests(self):
        """Run all tests in sequence."""
        try:
            print("🧪 Flux-Hunyuan-BPT Complete Test Suite")
            print("=" * 60)
            
            # 1. Check dependencies
            if not self.check_dependencies():
                print("❌ Dependency check failed")
                return False
            
            # 2. Check Python packages
            if not self.check_python_packages():
                print("❌ Package check failed")
                return False
            
            # 3. Run original demo test (optional, commented out for speed)
            # print("\n🎯 Step 1: Original Demo Test")
            # demo_success = self.run_original_demo_test()
            # if not demo_success:
            #     print("⚠️  Original demo test failed, but continuing...")
            
            # 4. Start generation server
            print("\n🚀 Step 1: Start Generation Server")
            if not await self.start_generation_server():
                print("❌ Failed to start generation server")
                return False
            
            # 5. Run server tests
            print("\n🧪 Step 2: Server Tests")
            server_success = await self.run_server_tests()
            
            # 6. Run miner tests
            print("\n🧪 Step 3: Miner Tests")
            miner_success = await self.run_miner_tests()
            
            # 7. Print summary
            self.print_final_summary()
            
            return server_success and miner_success
            
        except KeyboardInterrupt:
            print("\n⚠️  Tests interrupted by user")
            return False
        except Exception as e:
            print(f"\n❌ Test suite failed: {e}")
            traceback.print_exc()
            return False
        finally:
            # Always stop server
            self.stop_generation_server()

def signal_handler(signum, frame):
    """Handle shutdown signals."""
    print("\n⚠️  Received interrupt signal, shutting down...")
    sys.exit(1)

async def main():
    """Main function."""
    # Setup signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    orchestrator = TestOrchestrator()
    
    try:
        success = await orchestrator.run_all_tests()
        return success
    finally:
        # Ensure server is stopped
        orchestrator.stop_generation_server()

if __name__ == "__main__":
    print("🎬 Flux-Hunyuan-BPT Complete Test Orchestrator")
    print("This will test the entire system end-to-end\n")
    
    # Ask user for confirmation
    response = input("Continue with comprehensive testing? (y/N): ")
    if response.lower() not in ['y', 'yes']:
        print("❌ Testing cancelled by user")
        sys.exit(0)
    
    print("\n🚀 Starting comprehensive test suite...")
    
    try:
        success = asyncio.run(main())
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n⚠️  Testing interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Testing failed: {e}")
        traceback.print_exc()
        sys.exit(1) 