# Ollama GPU Integration for Distributed RL System

## 🎯 Problem Statement

The distributed RL system needs **one Ollama instance per GPU** to avoid race conditions when multiple GPU agents simultaneously request RL optimizations. Currently, a shared Ollama server would create bottlenecks and conflicts between concurrent RL loops.

## 🏗️ Solution Architecture

### **Per-GPU Ollama Servers**

Similar to TRELLIS servers, we need dedicated Ollama instances:

```
GPU 0: Ollama Server (Port 11434) ← RL Agent (Port 8096)
GPU 1: Ollama Server (Port 11435) ← RL Agent (Port 8097)
GPU 2: Ollama Server (Port 11436) ← RL Agent (Port 8098)
...
GPU 7: Ollama Server (Port 11441) ← RL Agent (Port 8103)
```

### **Ollama GPU Selection Support**

Ollama **DOES support GPU selection** via environment variables:

```bash
# Method 1: CUDA_VISIBLE_DEVICES (Recommended)
CUDA_VISIBLE_DEVICES=0 ollama serve --host 127.0.0.1 --port 11434

# Method 2: Ollama-specific GPU selection
OLLAMA_GPU=0 ollama serve --host 127.0.0.1 --port 11434
```

## 📋 Implementation Design

### **1. Ollama Server Manager**

```python
# src/ollama_integration/ollama_server_manager.py

import subprocess
import os
import time
import requests
from dataclasses import dataclass
from typing import Dict, List, Optional
from pathlib import Path

@dataclass
class OllamaServer:
    """Represents an Ollama server instance"""
    gpu_id: int
    port: int
    process: Optional[subprocess.Popen] = None
    status: str = "idle"  # idle, starting, running, failed
    model_loaded: Optional[str] = None
    error_count: int = 0

class OllamaServerManager:
    """Manages multiple Ollama servers across GPUs"""
    
    def __init__(self, num_gpus: int = 8, base_port: int = 11434, model_name: str = "llama3.1:8b"):
        self.num_gpus = num_gpus
        self.base_port = base_port
        self.model_name = model_name
        self.servers: Dict[int, OllamaServer] = {}
        
        # Initialize server configurations
        for gpu_id in range(num_gpus):
            self.servers[gpu_id] = OllamaServer(
                gpu_id=gpu_id,
                port=base_port + gpu_id
            )
    
    def start_ollama_server(self, gpu_id: int) -> bool:
        """Start Ollama server on specific GPU"""
        server = self.servers[gpu_id]
        
        try:
            # Set environment for GPU isolation
            env = os.environ.copy()
            env['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
            env['OLLAMA_HOST'] = f"127.0.0.1:{server.port}"
            
            # Start Ollama server
            cmd = [
                "ollama", "serve"
            ]
            
            server.process = subprocess.Popen(
                cmd,
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            
            server.status = "starting"
            
            # Wait for server to be ready
            if self._wait_for_server_ready(server):
                server.status = "running"
                
                # Load model on this GPU
                if self._load_model_on_server(server):
                    server.model_loaded = self.model_name
                    return True
                else:
                    server.status = "failed"
                    return False
            else:
                server.status = "failed"
                return False
                
        except Exception as e:
            print(f"Failed to start Ollama server on GPU {gpu_id}: {e}")
            server.status = "failed"
            server.error_count += 1
            return False
    
    def _wait_for_server_ready(self, server: OllamaServer, timeout: int = 30) -> bool:
        """Wait for Ollama server to be ready"""
        for _ in range(timeout):
            try:
                response = requests.get(f"http://127.0.0.1:{server.port}/api/tags", timeout=2)
                if response.status_code == 200:
                    return True
            except:
                pass
            time.sleep(1)
        return False
    
    def _load_model_on_server(self, server: OllamaServer) -> bool:
        """Load model on specific server"""
        try:
            # Pull/load model
            response = requests.post(
                f"http://127.0.0.1:{server.port}/api/pull",
                json={"name": self.model_name},
                timeout=300  # 5 minutes for model download
            )
            return response.status_code == 200
        except Exception as e:
            print(f"Failed to load model on GPU {server.gpu_id}: {e}")
            return False
    
    def start_all_servers(self) -> Dict[int, bool]:
        """Start all Ollama servers"""
        results = {}
        
        for gpu_id in range(self.num_gpus):
            print(f"Starting Ollama server on GPU {gpu_id}...")
            results[gpu_id] = self.start_ollama_server(gpu_id)
            
            # Brief delay between starts
            time.sleep(2)
        
        return results
    
    def get_server_url(self, gpu_id: int) -> str:
        """Get server URL for specific GPU"""
        server = self.servers[gpu_id]
        return f"http://127.0.0.1:{server.port}"
    
    def stop_all_servers(self):
        """Stop all Ollama servers"""
        for gpu_id, server in self.servers.items():
            if server.process:
                server.process.terminate()
                server.process.wait()
                print(f"Stopped Ollama server on GPU {gpu_id}")
```

### **2. RL Agent Integration**

```python
# Modification to src/gpu_agent/simple_gpu_agent.py

class SimpleGPUAgent:
    def __init__(self, gpu_id: int, port: int, coordinator_url: str, ollama_port: int = None):
        self.gpu_id = gpu_id
        self.port = port
        self.coordinator_url = coordinator_url
        
        # Ollama integration
        self.ollama_port = ollama_port or (11434 + gpu_id)
        self.ollama_url = f"http://127.0.0.1:{self.ollama_port}"
        self.ollama_client = None
        
        # ... existing initialization
    
    async def _initialize_ollama_client(self):
        """Initialize connection to local Ollama server"""
        import aiohttp
        
        self.ollama_client = aiohttp.ClientSession()
        
        # Test connection
        try:
            async with self.ollama_client.get(f"{self.ollama_url}/api/tags") as response:
                if response.status == 200:
                    logger.info(f"GPU {self.gpu_id}: Connected to Ollama server on port {self.ollama_port}")
                    return True
                else:
                    logger.error(f"GPU {self.gpu_id}: Ollama server not ready on port {self.ollama_port}")
                    return False
        except Exception as e:
            logger.error(f"GPU {self.gpu_id}: Failed to connect to Ollama: {e}")
            return False
    
    async def _get_rl_strategy_from_ollama(self, prompt: str, context: str) -> Dict[str, Any]:
        """Get RL optimization strategy from local Ollama"""
        
        try:
            rl_prompt = f"""
You are an expert prompt optimizer for 3D model generation. 

Original prompt: "{prompt}"
Context: {context}

Suggest an improved version focusing on:
1. Adding descriptive adjectives
2. Improving material/texture descriptions  
3. Enhancing spatial/geometric details
4. Maintaining the core concept

Respond with:
- strategy: "creative_expansion" or "detail_enhancement" or "structure_refinement"
- improved_prompt: "your improved version"
- reasoning: "brief explanation"

Format as JSON.
"""
            
            async with self.ollama_client.post(
                f"{self.ollama_url}/api/generate",
                json={
                    "model": "llama3.1:8b",
                    "prompt": rl_prompt,
                    "stream": False,
                    "options": {
                        "temperature": 0.7,
                        "top_p": 0.9
                    }
                }
            ) as response:
                
                if response.status == 200:
                    result = await response.json()
                    
                    # Parse LLM response
                    llm_output = result.get('response', '')
                    
                    # Try to extract JSON from response
                    import json
                    try:
                        # Look for JSON in response
                        start = llm_output.find('{')
                        end = llm_output.rfind('}') + 1
                        
                        if start != -1 and end != 0:
                            json_str = llm_output[start:end]
                            parsed = json.loads(json_str)
                            
                            return {
                                "strategy": parsed.get("strategy", "creative_expansion"),
                                "improved_prompt": parsed.get("improved_prompt", prompt),
                                "reasoning": parsed.get("reasoning", "No reasoning provided"),
                                "confidence": 0.8
                            }
                    except:
                        pass
                    
                    # Fallback: simple enhancement
                    return {
                        "strategy": "creative_expansion",
                        "improved_prompt": f"detailed {prompt} with intricate textures",
                        "reasoning": "Added descriptive elements",
                        "confidence": 0.6
                    }
                else:
                    logger.error(f"GPU {self.gpu_id}: Ollama request failed: {response.status}")
                    return self._get_fallback_strategy(prompt)
                    
        except Exception as e:
            logger.error(f"GPU {self.gpu_id}: Ollama request error: {e}")
            return self._get_fallback_strategy(prompt)
    
    def _get_fallback_strategy(self, prompt: str) -> Dict[str, Any]:
        """Fallback when Ollama is unavailable"""
        strategies = ["creative_expansion", "detail_enhancement", "structure_refinement"]
        strategy = random.choice(strategies)
        
        enhancements = {
            "creative_expansion": f"artistic {prompt} with vibrant colors",
            "detail_enhancement": f"highly detailed {prompt} with intricate patterns",
            "structure_refinement": f"geometrically precise {prompt} with clean lines"
        }
        
        return {
            "strategy": strategy,
            "improved_prompt": enhancements[strategy],
            "reasoning": f"Fallback {strategy} applied",
            "confidence": 0.4
        }
```

### **3. System Startup Integration**

```python
# Modification to scripts/start_simple_system.py

from src.ollama_integration.ollama_server_manager import OllamaServerManager

class SimpleSystemManager:
    def __init__(self, num_gpus: int = 8):
        self.num_gpus = num_gpus
        # ... existing initialization
        
        # Add Ollama manager
        self.ollama_manager = OllamaServerManager(
            num_gpus=num_gpus,
            base_port=11434,
            model_name="llama3.1:8b"
        )
    
    def start_ollama_servers(self) -> bool:
        """Start Ollama servers on all GPUs"""
        console.print("🦙 Starting Ollama servers on all GPUs...")
        
        results = self.ollama_manager.start_all_servers()
        
        success_count = sum(1 for success in results.values() if success)
        
        if success_count == self.num_gpus:
            console.print(f"✅ All {self.num_gpus} Ollama servers started successfully")
            return True
        else:
            console.print(f"⚠️  {success_count}/{self.num_gpus} Ollama servers started", style="yellow")
            
            # Show failed servers
            for gpu_id, success in results.items():
                if not success:
                    console.print(f"   ❌ GPU {gpu_id} Ollama server failed", style="red")
            
            return success_count > 0  # Allow partial success
    
    def start_system(self):
        """Start complete system with Ollama integration"""
        
        try:
            # Start Redis
            if not self.start_redis():
                return False
            
            # Start Ollama servers (NEW)
            if not self.start_ollama_servers():
                console.print("⚠️  Continuing without some Ollama servers...", style="yellow")
            
            # Start coordinator
            if not self.start_coordinator():
                return False
            
            # Start GPU agents (they will connect to their local Ollama)
            if not self.start_gpu_agents():
                return False
            
            console.print("\\n🎉 Complete system started successfully!")
            console.print("\\nComponents running:")
            console.print("  📊 Redis: localhost:6379")
            console.print("  🎯 Coordinator: http://localhost:8090")
            console.print("  🦙 Ollama servers: ports 11434-11441")
            console.print("  🧠 GPU Agents: ports 8096-8103")
            
            return True
            
        except Exception as e:
            console.print(f"💥 System startup failed: {e}", style="red")
            return False
```

## 🔧 Configuration Updates

### **Settings Enhancement**

```python
# config/settings.py additions

class Settings:
    # ... existing settings
    
    # Ollama Configuration
    ollama_base_port: int = 11434
    ollama_model_name: str = "llama3.1:8b"
    ollama_enabled: bool = True
    ollama_timeout: int = 30
    
    # GPU Isolation
    use_cuda_visible_devices: bool = True
    ollama_startup_delay: int = 2  # seconds between Ollama server starts
```

## 🧪 Testing Strategy

### **Ollama Integration Test**

```python
# minimal_tests/test_ollama_integration.py

async def test_ollama_per_gpu():
    """Test Ollama servers running on each GPU"""
    
    num_gpus = 2  # Test with 2 GPUs
    
    # Start Ollama servers
    ollama_manager = OllamaServerManager(num_gpus=num_gpus)
    results = ollama_manager.start_all_servers()
    
    # Test each server independently
    for gpu_id in range(num_gpus):
        server_url = ollama_manager.get_server_url(gpu_id)
        
        # Test model loaded
        response = requests.get(f"{server_url}/api/tags")
        assert response.status_code == 200
        
        # Test generation
        response = requests.post(
            f"{server_url}/api/generate",
            json={
                "model": "llama3.1:8b",
                "prompt": "Improve this prompt: red car",
                "stream": False
            }
        )
        assert response.status_code == 200
        
        print(f"✅ GPU {gpu_id} Ollama server working on {server_url}")
    
    # Test concurrent requests (race condition test)
    import asyncio
    
    async def concurrent_request(gpu_id):
        server_url = ollama_manager.get_server_url(gpu_id)
        
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{server_url}/api/generate",
                json={
                    "model": "llama3.1:8b",
                    "prompt": f"GPU {gpu_id} test prompt",
                    "stream": False
                }
            ) as response:
                result = await response.json()
                return gpu_id, result
    
    # Send concurrent requests to different GPUs
    tasks = [concurrent_request(gpu_id) for gpu_id in range(num_gpus)]
    results = await asyncio.gather(*tasks)
    
    # Verify each GPU handled its own request
    for gpu_id, result in results:
        assert result is not None
        print(f"✅ GPU {gpu_id} handled concurrent request successfully")
    
    print("🎉 All Ollama per-GPU tests passed!")
```

## 📊 Performance Benefits

### **Isolation Advantages**
- ✅ **No race conditions**: Each GPU has dedicated Ollama instance
- ✅ **Parallel processing**: True concurrent RL optimization
- ✅ **Memory isolation**: GPU memory usage contained per GPU
- ✅ **Fault tolerance**: One GPU failure doesn't affect others

### **Resource Usage**
- **Memory per GPU**: ~4-8GB VRAM for llama3.1:8b
- **CPU overhead**: Minimal per Ollama instance
- **Network**: Localhost only, no external traffic
- **Startup time**: ~30-60 seconds per server (model loading)

## 🚀 Migration Strategy

### **Phase 1: Add Ollama Integration (Current)**
```bash
# Update startup script
python scripts/start_simple_system.py  # Now includes Ollama

# Test integration
python minimal_tests/test_ollama_integration.py
```

### **Phase 2: Enhanced RL Strategies**
- Replace simulation with real Ollama-powered RL
- Add strategy performance tracking
- Implement cross-GPU strategy learning

### **Phase 3: Advanced Features**
- Model switching per prompt type
- Dynamic model loading/unloading
- Advanced prompt engineering templates

## ✅ Implementation Checklist

- [ ] Create `OllamaServerManager` class
- [ ] Integrate Ollama client in `SimpleGPUAgent`
- [ ] Update system startup scripts
- [ ] Add Ollama configuration to settings
- [ ] Create Ollama integration tests
- [ ] Update documentation with Ollama requirements
- [ ] Test race condition scenarios
- [ ] Validate GPU memory usage

This design ensures each GPU has its own isolated Ollama instance, eliminating race conditions while enabling true parallel RL optimization across all GPUs!

