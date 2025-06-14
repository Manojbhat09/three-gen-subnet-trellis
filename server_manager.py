#!/usr/bin/env python3
"""
Server Manager for GPU Memory Coordination
Manages validation and generation servers with proper GPU memory coordination
"""

import subprocess
import psutil
import signal
import time
import requests
import asyncio
import aiohttp
from pathlib import Path
import sys


class ServerManager:
    """Manages validation and generation servers"""
    
    def __init__(self):
        self.validation_port = 10006
        self.generation_port = 8095
        self.validation_process = None
        self.generation_process = None
        
    def find_process_by_port(self, port: int) -> list:
        """Find processes using a specific port"""
        processes = []
        for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
            try:
                for conn in proc.connections():
                    if conn.laddr.port == port:
                        processes.append(proc)
                        break
            except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
                pass
        return processes
    
    def kill_processes_by_port(self, port: int):
        """Kill all processes using a specific port"""
        processes = self.find_process_by_port(port)
        for proc in processes:
            try:
                print(f"Killing process {proc.pid} ({proc.name()}) using port {port}")
                proc.terminate()
                proc.wait(timeout=5)
            except (psutil.NoSuchProcess, psutil.TimeoutExpired):
                try:
                    proc.kill()
                except psutil.NoSuchProcess:
                    pass
    
    def kill_python_processes(self):
        """Kill all Python processes that might be using GPU"""
        python_processes = []
        for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
            try:
                if 'python' in proc.name().lower():
                    cmdline = ' '.join(proc.cmdline()) if proc.cmdline() else ''
                    # Look for processes that might be using GPU
                    if any(keyword in cmdline.lower() for keyword in [
                        'serve.py', 'flux_hunyuan', 'generation_server', 'torch', 'cuda'
                    ]):
                        python_processes.append(proc)
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass
        
        if python_processes:
            print(f"Found {len(python_processes)} Python processes that might use GPU:")
            for proc in python_processes:
                try:
                    cmdline = ' '.join(proc.cmdline()[:3]) if proc.cmdline() else proc.name()
                    print(f"  PID {proc.pid}: {cmdline}")
                except:
                    print(f"  PID {proc.pid}: {proc.name()}")
            
            response = input("Kill these processes? [y/N]: ").strip().lower()
            if response == 'y':
                for proc in python_processes:
                    try:
                        print(f"Killing process {proc.pid}")
                        proc.terminate()
                        proc.wait(timeout=3)
                    except (psutil.NoSuchProcess, psutil.TimeoutExpired):
                        try:
                            proc.kill()
                        except psutil.NoSuchProcess:
                            pass
                
                print("Waiting for processes to clean up...")
                time.sleep(3)
    
    def clear_gpu_memory(self):
        """Clear GPU memory"""
        try:
            print("Clearing GPU memory...")
            subprocess.run(["nvidia-smi", "--gpu-reset"], check=False)
            time.sleep(2)
        except Exception as e:
            print(f"GPU reset command failed: {e}")
    
    async def check_server_health(self, url: str, endpoint: str = "") -> bool:
        """Check if a server is healthy"""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(f"{url}{endpoint}", timeout=5) as response:
                    return response.status == 200
        except:
            return False
    
    def start_validation_server(self):
        """Start the validation server"""
        print("Starting validation server...")
        
        # Kill existing processes on port
        self.kill_processes_by_port(self.validation_port)
        time.sleep(2)
        
        # Change to validation directory and start server
        validation_dir = Path("validation")
        if not validation_dir.exists():
            print("❌ validation/ directory not found")
            return False
        
        try:
            # Start in background
            self.validation_process = subprocess.Popen([
                "python", "serve.py", "--host", "0.0.0.0", "--port", str(self.validation_port)
            ], cwd=validation_dir, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            
            print(f"Validation server starting... PID: {self.validation_process.pid}")
            
            # Wait for server to start
            for i in range(30):  # 30 second timeout
                time.sleep(1)
                if self.validation_process.poll() is not None:
                    print("❌ Validation server process died")
                    return False
                
                # Check if server is responding
                try:
                    response = requests.get(f"http://localhost:{self.validation_port}/version/", timeout=2)
                    if response.status_code == 200:
                        print(f"✅ Validation server started successfully on port {self.validation_port}")
                        return True
                except:
                    pass
            
            print("❌ Validation server failed to start within timeout")
            return False
            
        except Exception as e:
            print(f"❌ Failed to start validation server: {e}")
            return False
    
    def start_generation_server(self):
        """Start the generation server"""
        print("Starting generation server...")
        
        # Kill existing processes on port
        self.kill_processes_by_port(self.generation_port)
        time.sleep(2)
        
        # Check if flux_hunyuan_sugar_generation_server.py exists
        gen_script = Path("flux_hunyuan_sugar_generation_server.py")
        if not gen_script.exists():
            print("❌ flux_hunyuan_sugar_generation_server.py not found")
            return False
        
        try:
            # Start in background
            self.generation_process = subprocess.Popen([
                "python", "flux_hunyuan_sugar_generation_server.py"
            ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            
            print(f"Generation server starting... PID: {self.generation_process.pid}")
            
            # Wait for server to start
            for i in range(60):  # 60 second timeout (model loading takes time)
                time.sleep(1)
                if self.generation_process.poll() is not None:
                    print("❌ Generation server process died")
                    return False
                
                # Check if server is responding
                try:
                    response = requests.get(f"http://localhost:{self.generation_port}/health/", timeout=2)
                    if response.status_code == 200:
                        print(f"✅ Generation server started successfully on port {self.generation_port}")
                        return True
                except:
                    pass
            
            print("❌ Generation server failed to start within timeout")
            return False
            
        except Exception as e:
            print(f"❌ Failed to start generation server: {e}")
            return False
    
    def stop_servers(self):
        """Stop both servers"""
        print("Stopping servers...")
        
        if self.validation_process:
            try:
                self.validation_process.terminate()
                self.validation_process.wait(timeout=5)
                print("✅ Validation server stopped")
            except:
                try:
                    self.validation_process.kill()
                except:
                    pass
        
        if self.generation_process:
            try:
                self.generation_process.terminate() 
                self.generation_process.wait(timeout=5)
                print("✅ Generation server stopped")
            except:
                try:
                    self.generation_process.kill()
                except:
                    pass
        
        # Also kill any remaining processes on the ports
        self.kill_processes_by_port(self.validation_port)
        self.kill_processes_by_port(self.generation_port)
    
    async def coordinate_startup(self):
        """Start servers with proper coordination"""
        print("🚀 Starting servers with GPU memory coordination")
        print("=" * 60)
        
        # Step 1: Clean up any existing processes
        print("\n1️⃣ Cleaning up existing processes...")
        self.kill_python_processes()
        
        # Step 2: Clear GPU memory
        print("\n2️⃣ Clearing GPU memory...")
        self.clear_gpu_memory()
        
        # Step 3: Start validation server first (uses less GPU memory)
        print("\n3️⃣ Starting validation server...")
        if not self.start_validation_server():
            print("❌ Failed to start validation server")
            return False
        
        # Step 4: Wait and check validation server
        print("\n4️⃣ Checking validation server health...")
        val_healthy = await self.check_server_health(f"http://localhost:{self.validation_port}", "/version/")
        if not val_healthy:
            print("❌ Validation server not healthy")
            return False
        print("✅ Validation server is healthy")
        
        # Step 5: Force validation server to clean GPU memory
        print("\n5️⃣ Forcing validation server GPU cleanup...")
        try:
            response = requests.post(f"http://localhost:{self.validation_port}/cleanup_gpu/", timeout=10)
            if response.status_code == 200:
                result = response.json()
                print(f"✅ GPU cleanup successful: {result}")
            else:
                print(f"⚠️  GPU cleanup returned status {response.status_code}")
        except Exception as e:
            print(f"⚠️  GPU cleanup failed: {e}")
        
        # Step 6: Start generation server
        print("\n6️⃣ Starting generation server...")
        if not self.start_generation_server():
            print("❌ Failed to start generation server")
            self.stop_servers()
            return False
        
        # Step 7: Final health check
        print("\n7️⃣ Final health check...")
        val_healthy = await self.check_server_health(f"http://localhost:{self.validation_port}", "/version/")
        gen_healthy = await self.check_server_health(f"http://localhost:{self.generation_port}", "/health/")
        
        if val_healthy and gen_healthy:
            print("✅ Both servers are healthy and ready!")
            print(f"   Validation server: http://localhost:{self.validation_port}")
            print(f"   Generation server: http://localhost:{self.generation_port}")
            return True
        else:
            print(f"❌ Server health check failed: val={val_healthy}, gen={gen_healthy}")
            self.stop_servers()
            return False
    
    def interactive_menu(self):
        """Interactive menu for server management"""
        while True:
            print("\n" + "=" * 50)
            print("🖥️  SERVER MANAGER")
            print("=" * 50)
            print("1. Start servers with coordination")
            print("2. Stop servers")
            print("3. Check server status")
            print("4. Kill all Python processes")
            print("5. Run coordination test")
            print("6. Run mining pipeline")
            print("0. Exit")
            print("=" * 50)
            
            choice = input("Enter choice: ").strip()
            
            if choice == "1":
                print("\nStarting servers...")
                success = asyncio.run(self.coordinate_startup())
                if success:
                    print("\n🎉 Servers started successfully!")
                else:
                    print("\n❌ Failed to start servers")
            
            elif choice == "2":
                self.stop_servers()
                print("\n✅ Servers stopped")
            
            elif choice == "3":
                print("\nChecking server status...")
                val_healthy = asyncio.run(self.check_server_health(f"http://localhost:{self.validation_port}", "/version/"))
                gen_healthy = asyncio.run(self.check_server_health(f"http://localhost:{self.generation_port}", "/health/"))
                
                print(f"Validation server (port {self.validation_port}): {'✅ Healthy' if val_healthy else '❌ Not responding'}")
                print(f"Generation server (port {self.generation_port}): {'✅ Healthy' if gen_healthy else '❌ Not responding'}")
            
            elif choice == "4":
                self.kill_python_processes()
                print("✅ Python processes killed")
            
            elif choice == "5":
                print("\nRunning coordination test...")
                subprocess.run([sys.executable, "gpu_coordination_test.py"])
            
            elif choice == "6":
                print("\nRunning mining pipeline...")
                subprocess.run([sys.executable, "complete_mining_pipeline_test2m3b2.py"])
            
            elif choice == "0":
                self.stop_servers()
                break
            
            else:
                print("Invalid choice")


def main():
    """Main function"""
    manager = ServerManager()
    
    if len(sys.argv) > 1:
        if sys.argv[1] == "start":
            success = asyncio.run(manager.coordinate_startup())
            sys.exit(0 if success else 1)
        elif sys.argv[1] == "stop":
            manager.stop_servers()
            sys.exit(0)
        elif sys.argv[1] == "status":
            val_healthy = asyncio.run(manager.check_server_health(f"http://localhost:{manager.validation_port}", "/version/"))
            gen_healthy = asyncio.run(manager.check_server_health(f"http://localhost:{manager.generation_port}", "/health/"))
            print(f"Validation: {'✅' if val_healthy else '❌'}")
            print(f"Generation: {'✅' if gen_healthy else '❌'}")
            sys.exit(0)
    else:
        manager.interactive_menu()


if __name__ == "__main__":
    main() 