#!/usr/bin/env python3
"""
Episodic Prompt Optimizer - Multi-Episode Learning Wrapper

This script runs multiple episodes of prompt optimization using the V4.1 RL Loop optimizer.
Each episode cycles through all test prompts, allowing the agent to learn principles and
strategies that transfer across different prompt types and episodes.

Features:
- Multi-episode learning with persistent memory
- Cross-prompt principle extraction
- Progressive strategy refinement
- Comprehensive logging and analytics
- Convergence tracking across episodes
- Robust GPU server coordination with race condition prevention

Usage:
    python episodic_prompt_optimizer.py [OPTIONS]
    
Options:
    --episodes INT          Number of episodes to run (default: 30)
    --target FLOAT          Target validation score (default: 0.85)
    --max-rounds INT        Maximum optimization rounds per prompt (default: 5)
    --log-dir STR           Directory for storing episode logs (default: episodic_logs)
    --endpoint STR          TRELLIS endpoint to use (default: generate/cinema/)
    --ollama-url STR        Ollama server URL (default: http://localhost:11434)
    --port INT              TRELLIS server port (default: 8096)
    --server-buffer-time INT Buffer time between server uses in seconds (default: 30)

Example:
    CUDA_VISIBLE_DEVICES=2 python episodic_prompt_optimizer.py --episodes 15 --target 0.95 --max-rounds 2 --log-dir episodic_logs_first --endpoint "generate/cinema/" --ollama-url http://localhost:11434 --port 8097
"""

import json
import os
import time
import statistics
import threading
from datetime import datetime
from typing import List, Dict, Any, Tuple
import logging
import time
import re
import requests
from episodic_test_prompts import EPISODIC_TEST_PROMPTS
# Import the V4.1 RL Loop optimizer
# from smart_prompt_optimizer_v4_1_rl_loop import RLLoopAgent
# from smart_prompt_optimizer_v5_rl_loop import RLLoopAgent
from smart_prompt_optimizer_v5_rl_loop_lora import RLLoopAgent
# from smart_prompt_optimizer_v5_rl_loop_hunyuan import RLLoopAgent

# Import the reproducibility system
try:
    from llm_close_prompt_reproducibility_test import LLMClosePromptReproducibility
    REPRODUCIBILITY_SYSTEM_AVAILABLE = True
    print("✅ Using reproducibility system for pre-optimization")
except ImportError:
    REPRODUCIBILITY_SYSTEM_AVAILABLE = False
    print("⚠️ Reproducibility system not available")


class ServerCoordinator:
    """
    Coordinates access to the GPU server to prevent race conditions and ensure proper sequencing.
    Implements buffer times and status checking to avoid conflicts with other processes.
    """
    
    def __init__(self, server_url: str = "http://localhost:8096", 
                 buffer_time_seconds: int = 30,
                 max_wait_time_seconds: int = 300,
                 status_check_interval: int = 5):
        """
        Initialize the server coordinator.
        
        Args:
            server_url: Base URL of the GPU server
            buffer_time_seconds: Time to wait after server becomes available before using it
            max_wait_time_seconds: Maximum time to wait for server to become available
            status_check_interval: Interval between status checks
        """
        self.server_url = server_url.rstrip('/')
        self.buffer_time_seconds = buffer_time_seconds
        self.max_wait_time_seconds = max_wait_time_seconds
        self.status_check_interval = status_check_interval
        self.last_server_use_time = 0.0
        self.logger = logging.getLogger(__name__)
        
    def check_server_status(self) -> Dict[str, Any]:
        """
        Check the current status of the GPU server.
        
        Returns:
            Dictionary containing server status information
        """
        try:
            # First check health endpoint
            health_url = f"{self.server_url}/health/"
            health_resp = requests.get(health_url, timeout=5)
            if health_resp.status_code != 200:
                return {
                    "available": False,
                    "status": "unhealthy",
                    "error": f"Health check failed: HTTP {health_resp.status_code}"
                }
            
            # Check job status
            job_status_url = f"{self.server_url}/job/status/"
            job_resp = requests.get(job_status_url, timeout=5)
            if job_resp.status_code != 200:
                return {
                    "available": False,
                    "status": "unknown",
                    "error": f"Job status check failed: HTTP {job_resp.status_code}"
                }
            
            job_data = job_resp.json()
            job_status = job_data.get('status', 'unknown')
            
            # Check if server is busy
            if job_status in ('processing', 'generating', 'validating'):
                return {
                    "available": False,
                    "status": job_status,
                    "job_id": job_data.get('job_id'),
                    "prompt": job_data.get('prompt'),
                    "start_time": job_data.get('start_time')
                }
            
            # Check if enough time has passed since last use
            time_since_last_use = time.time() - self.last_server_use_time
            if time_since_last_use < self.buffer_time_seconds:
                remaining_buffer = self.buffer_time_seconds - time_since_last_use
                return {
                    "available": False,
                    "status": "buffer_time",
                    "remaining_buffer_seconds": remaining_buffer,
                    "last_use_time": self.last_server_use_time
                }
            
            # Server is available
            return {
                "available": True,
                "status": job_status,
                "job_id": job_data.get('job_id')
            }
            
        except requests.exceptions.Timeout:
            return {
                "available": False,
                "status": "timeout",
                "error": "Server status check timed out"
            }
        except requests.exceptions.ConnectionError:
            return {
                "available": False,
                "status": "connection_error",
                "error": "Cannot connect to server"
            }
        except Exception as e:
            return {
                "available": False,
                "status": "error",
                "error": str(e)
            }
    
    def wait_for_server_availability(self) -> bool:
        """
        Wait for the server to become available, respecting buffer times.
        
        Returns:
            True if server became available, False if timeout reached
        """
        start_wait_time = time.time()
        
        while time.time() - start_wait_time < self.max_wait_time_seconds:
            status = self.check_server_status()
            
            if status["available"]:
                self.logger.info(f"✅ Server is available (status: {status['status']})")
                return True
            
            # Log the current status
            if status["status"] == "buffer_time":
                remaining = status.get("remaining_buffer_seconds", 0)
                self.logger.info(f"⏳ Waiting for buffer time: {remaining:.1f}s remaining")
            elif status["status"] in ("processing", "generating", "validating"):
                job_id = status.get("job_id", "unknown")
                prompt = status.get("prompt", "unknown")
                self.logger.info(f"⏳ Server busy: {status['status']} (job: {job_id}, prompt: {prompt[:50]}...)")
            else:
                error = status.get("error", "unknown error")
                self.logger.info(f"⏳ Server unavailable: {status['status']} - {error}")
            
            # Wait before next check
            time.sleep(self.status_check_interval)
        
        self.logger.warning(f"⏰ Timeout waiting for server availability ({self.max_wait_time_seconds}s)")
        return False
    
    def mark_server_used(self):
        """Mark that the server has been used (for buffer time tracking)"""
        self.last_server_use_time = time.time()
        self.logger.info(f"📝 Marked server as used at {self.last_server_use_time}")
    
    def clear_server_cache(self) -> bool:
        """
        Clear the GPU cache on the server.
        
        Returns:
            True if successful, False otherwise
        """
        try:
            clear_url = f"{self.server_url}/clear_cache/"
            resp = requests.post(clear_url, timeout=10)
            if resp.status_code == 200:
                self.logger.info("🧹 GPU cache cleared successfully")
                return True
            else:
                self.logger.warning(f"⚠️ Failed to clear GPU cache: HTTP {resp.status_code}")
                return False
        except Exception as e:
            self.logger.warning(f"⚠️ Exception clearing GPU cache: {e}")
            return False


class OllamaCoordinator:
    """
    Coordinates access to the Ollama server with priority-based queuing.
    Prevents timeouts and race conditions when multiple RL runners share one Ollama server.
    
    Priority levels:
    - HIGH (1): Critical operations, can interrupt lower priority tasks
    - MEDIUM (2): Normal operations, wait for server to be free
    - LOW (3): Background operations, lowest priority
    """
    
    def __init__(self, ollama_url: str = "http://localhost:11434",
                 max_wait_time_seconds: int = 300,
                 status_check_interval: int = 2,
                 priority_timeout_seconds: int = 60):
        """
        Initialize the Ollama coordinator.
        
        Args:
            ollama_url: Base URL of the Ollama server
            max_wait_time_seconds: Maximum time to wait for server availability
            status_check_interval: Interval between status checks
            priority_timeout_seconds: Timeout for priority-based operations
        """
        self.ollama_url = ollama_url.rstrip('/')
        self.max_wait_time_seconds = max_wait_time_seconds
        self.status_check_interval = status_check_interval
        self.priority_timeout_seconds = priority_timeout_seconds
        self.logger = logging.getLogger(__name__)
        
        # Priority queue for managing requests
        self.request_queue = []
        self.active_requests = {}  # request_id -> request_info
        self.request_counter = 0
        self.lock = threading.Lock()
        
        # Start background queue processor
        self.running = True
        self.queue_processor = threading.Thread(target=self._process_queue, daemon=True)
        self.queue_processor.start()
    
    def _generate_request_id(self) -> str:
        """Generate unique request ID"""
        with self.lock:
            self.request_counter += 1
            return f"ollama_req_{self.request_counter}_{int(time.time())}"
    
    def check_ollama_status(self) -> Dict[str, Any]:
        """
        Check the current status of the Ollama server.
        
        Returns:
            Dictionary containing server status information
        """
        try:
            # Check if Ollama is responding
            health_url = f"{self.ollama_url}/api/tags"
            resp = requests.get(health_url, timeout=5)
            
            if resp.status_code != 200:
                return {
                    "available": False,
                    "status": "unhealthy",
                    "error": f"Health check failed: HTTP {resp.status_code}"
                }
            
            # Check if there are active requests
            with self.lock:
                active_count = len(self.active_requests)
                queue_length = len(self.request_queue)
            
            # Server is available if no active requests or queue is manageable
            if active_count == 0:
                return {
                    "available": True,
                    "status": "idle",
                    "active_requests": 0,
                    "queue_length": queue_length
                }
            elif active_count <= 2:  # Allow up to 2 concurrent requests
                return {
                    "available": True,
                    "status": "busy",
                    "active_requests": active_count,
                    "queue_length": queue_length
                }
            else:
                return {
                    "available": False,
                    "status": "overloaded",
                    "active_requests": active_count,
                    "queue_length": queue_length,
                    "error": f"Too many active requests: {active_count}"
                }
                
        except requests.exceptions.Timeout:
            return {
                "available": False,
                "status": "timeout",
                "error": "Ollama server status check timed out"
            }
        except requests.exceptions.ConnectionError:
            return {
                "available": False,
                "status": "connection_error",
                "error": "Cannot connect to Ollama server"
            }
        except Exception as e:
            return {
                "available": False,
                "status": "error",
                "error": str(e)
            }
    
    def request_access(self, priority: int = 2, description: str = "Unknown", 
                      timeout_seconds: int = None) -> str:
        """
        Request access to the Ollama server with priority-based queuing.
        
        Args:
            priority: Priority level (1=HIGH, 2=MEDIUM, 3=LOW)
            description: Description of the request for logging
            timeout_seconds: Custom timeout for this request
            
        Returns:
            Request ID that can be used to check status or cancel
        """
        if timeout_seconds is None:
            timeout_seconds = self.priority_timeout_seconds
        
        request_id = self._generate_request_id()
        request_info = {
            'id': request_id,
            'priority': priority,
            'description': description,
            'timestamp': time.time(),
            'timeout': timeout_seconds,
            'status': 'queued'
        }
        
        with self.lock:
            # Insert based on priority (lower number = higher priority)
            insert_idx = 0
            for i, existing in enumerate(self.request_queue):
                if existing['priority'] > priority:
                    insert_idx = i
                    break
                insert_idx = i + 1
            
            self.request_queue.insert(insert_idx, request_info)
            
        self.logger.info(f"📋 Queued Ollama request: {request_id} (priority: {priority}, desc: {description})")
        return request_id
    
    def wait_for_access(self, request_id: str) -> bool:
        """
        Wait for access to be granted for a specific request.
        
        Args:
            request_id: The request ID returned by request_access
            
        Returns:
            True if access granted, False if timeout or error
        """
        start_time = time.time()
        
        while time.time() - start_time < self.max_wait_time_seconds:
            with self.lock:
                # Check if request is active
                if request_id in self.active_requests:
                    self.logger.info(f"✅ Ollama access granted for request: {request_id}")
                    return True
                
                # Check if request is still in queue
                request_found = False
                for req in self.request_queue:
                    if req['id'] == request_id:
                        request_found = True
                        position = self.request_queue.index(req) + 1
                        self.logger.debug(f"⏳ Request {request_id} in queue position {position}")
                        break
                
                if not request_found:
                    self.logger.warning(f"⚠️ Request {request_id} not found in queue")
                    return False
            
            # Wait before checking again
            time.sleep(self.status_check_interval)
        
        self.logger.error(f"⏰ Timeout waiting for Ollama access: {request_id}")
        return False
    
    def _process_queue(self):
        """Background thread to process the request queue"""
        while self.running:
            try:
                with self.lock:
                    # Check if we can process more requests
                    if len(self.active_requests) >= 2:  # Max 2 concurrent
                        time.sleep(0.1)
                        continue
                    
                    # Process next request in queue
                    if self.request_queue:
                        request = self.request_queue.pop(0)
                        request['status'] = 'active'
                        self.active_requests[request['id']] = request
                        
                        self.logger.info(f"🚀 Started Ollama request: {request['id']} (priority: {request['priority']})")
                        
                        # Set up automatic cleanup after timeout (only as fallback)
                        # The timer will be cancelled when the request is manually released
                        cleanup_timer = threading.Timer(
                            request['timeout'], 
                            self._cleanup_request, 
                            args=[request['id']]
                        )
                        cleanup_timer.start()
                        # Store the timer reference so we can cancel it
                        request['cleanup_timer'] = cleanup_timer
                
                time.sleep(0.1)  # Small delay to prevent busy waiting
                
            except Exception as e:
                self.logger.error(f"Error in Ollama queue processor: {e}")
                time.sleep(1)
    
    def _cleanup_request(self, request_id: str):
        """Clean up a completed or timed out request"""
        with self.lock:
            if request_id in self.active_requests:
                request = self.active_requests.pop(request_id)
                self.logger.info(f"🧹 Cleaned up Ollama request: {request_id}")
    
    def release_access(self, request_id: str):
        """Release access for a completed request"""
        with self.lock:
            if request_id in self.active_requests:
                request = self.active_requests.pop(request_id)
                # Cancel the cleanup timer since we're manually releasing
                if 'cleanup_timer' in request and request['cleanup_timer']:
                    request['cleanup_timer'].cancel()
                self.logger.info(f"✅ Released Ollama access for request: {request_id}")
            else:
                self.logger.warning(f"⚠️ Request {request_id} not found in active requests")
    
    def cancel_request(self, request_id: str) -> bool:
        """Cancel a pending request"""
        with self.lock:
            # Remove from queue
            for i, req in enumerate(self.request_queue):
                if req['id'] == request_id:
                    self.request_queue.pop(i)
                    self.logger.info(f"❌ Cancelled Ollama request: {request_id}")
                    return True
            
            # Remove from active requests
            if request_id in self.active_requests:
                self.active_requests.pop(request_id)
                self.logger.info(f"❌ Cancelled active Ollama request: {request_id}")
                return True
            
            return False
    
    def get_queue_status(self) -> Dict[str, Any]:
        """Get current queue status"""
        with self.lock:
            return {
                'queue_length': len(self.request_queue),
                'active_requests': len(self.active_requests),
                'queue': [{'id': req['id'], 'priority': req['priority'], 'description': req['description']} 
                          for req in self.request_queue],
                'active': [{'id': req['id'], 'priority': req['priority'], 'description': req['description']} 
                           for req in self.active_requests.values()]
            }
    
    def shutdown(self):
        """Shutdown the coordinator"""
        self.running = False
        if self.queue_processor.is_alive():
            self.queue_processor.join(timeout=5)
        self.logger.info("🛑 Ollama coordinator shutdown complete")


class EpisodicPromptOptimizer:
    """
    Wrapper for running multiple episodes of prompt optimization.
    
    Each episode processes all test prompts sequentially, allowing the agent
    to build up knowledge and principles that can be applied across different
    prompt types and future episodes.
    """
    
    @staticmethod
    def _get_uncommented_prompts():
        # Only use uncommented prompts from the imported list
        # (Python import will not include commented lines, but user may want to comment out in the .py file)
        # Clean any malformed prompts that might have quotes around them
        cleaned_prompts = []
        for p in EPISODIC_TEST_PROMPTS:
            if isinstance(p, str) and p.strip():
                # Remove any surrounding quotes if they exist
                cleaned = p.strip()
                if cleaned.startswith("'") and cleaned.endswith("'"):
                    cleaned = cleaned[1:-1]
                elif cleaned.startswith('"') and cleaned.endswith('"'):
                    cleaned = cleaned[1:-1]
                
                # Only add if it's a valid prompt
                if cleaned:
                    cleaned_prompts.append(cleaned)
        
        return cleaned_prompts

    @staticmethod
    def _add_prompts_to_pyfile(new_prompts, pyfile_path="episodic_test_prompts.py"):
        # Helper to add new prompts to the .py file if not already present
        try:
            with open(pyfile_path, 'r') as f:
                lines = f.readlines()
            existing = set()
            for line in lines:
                m = re.match(r'\s*"(.*)"\s*,?\s*$', line)
                if m:
                    existing.add(m.group(1).strip())
            
            # Clean and validate new prompts before adding
            cleaned_prompts = []
            for prompt in new_prompts:
                if isinstance(prompt, str):
                    # Remove any surrounding quotes if they exist
                    cleaned = prompt.strip()
                    if cleaned.startswith("'") and cleaned.endswith("'"):
                        cleaned = cleaned[1:-1]
                    elif cleaned.startswith('"') and cleaned.endswith('"'):
                        cleaned = cleaned[1:-1]
                    
                    # Only add if it's a valid prompt and not already existing
                    if cleaned and cleaned not in existing:
                        cleaned_prompts.append(cleaned)
            
            if cleaned_prompts:
                # Insert after the opening bracket (first come, first serve - most recent first)
                idx = next(i for i, l in enumerate(lines) if l.strip().endswith("["))
                for prompt in reversed(cleaned_prompts):  # Reverse to maintain chronological order
                    lines.insert(idx + 1, f'    "{prompt}",\n')
                with open(pyfile_path, 'w') as f:
                    f.writelines(lines)
                print(f"[INFO] Added {len(cleaned_prompts)} new prompts to {pyfile_path} (most recent first)")
            else:
                print(f"[INFO] No new prompts to add to {pyfile_path}")
        except Exception as e:
            print(f"[WARN] Could not update episodic_test_prompts.py: {e}")

    def __init__(self, 
                 num_episodes: int = 30,
                 target_score: float = 0.85,
                 max_rounds_per_prompt: int = 5,
                 log_dir: str = "episodic_logs",
                 log_path: str = "continuous_trellis.log",
                 server_url: str = "http://localhost:8096",
                 server_buffer_time: int = 30,
                 endpoint: str = "generate/", 
                 ollama_url: str = "http://localhost:11434", # New parameter
                 use_vllm: bool = False,  # New parameter for vLLM
                 vllm_url: str = "http://localhost:9000",  # New parameter for vLLM URL
                 vllm_model: str = "llama-3-2-3b-it",  # New parameter for vLLM model
                 reverse_prompts: bool = False,  # New parameter for reverse prompt order
                 disable_convergence: bool = False):  # New parameter to disable convergence checking
        """
        Initialize the episodic optimizer.
        
        Args:
            num_episodes: Number of episodes to run
            target_score: Target validation score for each prompt
            max_rounds_per_prompt: Maximum optimization rounds per prompt
            log_dir: Directory for storing episode logs
            log_path: Path to the continuous trellis log file
            server_url: URL of the GPU server
            server_buffer_time: Buffer time in seconds between server uses
            use_vllm: Whether to use vLLM instead of Ollama
            vllm_url: vLLM server URL
            vllm_model: vLLM model name
            reverse_prompts: Whether to process prompts in reverse order (oldest first)
            disable_convergence: Whether to disable convergence checking and force all rounds
        """
        self.num_episodes = num_episodes
        self.target_score = target_score
        self.max_rounds_per_prompt = max_rounds_per_prompt
        self.log_dir = log_dir
        self.endpoint = endpoint
        self.use_vllm = use_vllm
        self.vllm_url = vllm_url
        self.vllm_model = vllm_model
        self.reverse_prompts = reverse_prompts
        self.disable_convergence = disable_convergence

        # Initialize server coordinator
        self.server_coordinator = ServerCoordinator(
            server_url=server_url,
            buffer_time_seconds=server_buffer_time
        )
        
        # Initialize Ollama coordinator for priority-based queuing
        # Only initialize if not using vLLM
        if not self.use_vllm:
            self.ollama_coordinator = OllamaCoordinator(
                ollama_url=ollama_url,
                max_wait_time_seconds=300,
                status_check_interval=2,
                priority_timeout_seconds=120  # Increased from 60 to 120 seconds
            )
        else:
            self.ollama_coordinator = None
        
        # Extract 0-fidelity prompts from log and add to test prompts file
        self._update_test_prompts_from_log(log_path)
        
        # Validate and clean the prompts file
        self._validate_prompts_file()
        
        # Use only uncommented prompts from the imported .py file
        self.test_prompts = self._get_uncommented_prompts()
        
        # Log prompt order configuration
        if self.reverse_prompts:
            print(f"📝 [CONFIG] Prompt processing order: REVERSE (oldest first)")
            print(f"📝 [CONFIG] First prompt to process: '{self.test_prompts[0] if self.test_prompts else 'None'}' (from beginning of list)")
            print(f"📝 [CONFIG] Last prompt to process: '{self.test_prompts[-1] if self.test_prompts else 'None'}' (from end of list)")
        else:
            print(f"📝 [CONFIG] Prompt processing order: NORMAL (newest first)")
            print(f"📝 [CONFIG] First prompt to process: '{self.test_prompts[-1] if self.test_prompts else 'None'}' (from end of list)")
            print(f"📝 [CONFIG] Last prompt to process: '{self.test_prompts[0] if self.test_prompts else 'None'}' (from beginning of list)")
        
        # Create log directory
        os.makedirs(self.log_dir, exist_ok=True)
        
        # Setup logging
        log_file = os.path.join(self.log_dir, f"episodic_run_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        
        # Initialize the RL optimizer with episodic memory and Ollama coordination
        self.optimizer = RLLoopAgent(
            memory_file=os.path.join(self.log_dir, "episodic_memory.json"),
            trellis_server_url_w_port=server_url,
            ollama_url=ollama_url, # Pass the new parameter
            ollama_coordinator=self.ollama_coordinator, # Pass the coordinator for granular control
            port=int(server_url.split(':')[-1]) if ':' in server_url else 8096, # Extract port from server URL
            use_vllm=self.use_vllm,  # Pass vLLM preference
            vllm_url=self.vllm_url,  # Pass vLLM URL
            vllm_model=self.vllm_model,  # Pass vLLM model
            disable_convergence=self.disable_convergence # Pass the new parameter
        )
        
        # Override RL parameters to match episodic settings
        self.optimizer.max_optimization_rounds = max_rounds_per_prompt
        self.optimizer.min_score_threshold = target_score
        
        # Print LLM provider information prominently
        print("\n" + "="*60)
        print("🤖 EPISODIC OPTIMIZER - LLM PROVIDER CONFIGURATION")
        print("="*60)
        if self.use_vllm:
            print(f"✅ Using vLLM: {self.vllm_url}")
            print(f"   Model: {self.vllm_model}")
            print(f"   Status: ACTIVE for episodic optimization")
        elif self.ollama_coordinator:
            print(f"✅ Using Ollama: {ollama_url}")
            print(f"   Coordinator: ACTIVE with priority queuing")
            print(f"   Status: ACTIVE for episodic optimization")
        else:
            print(f"⚠️ No LLM coordinator configured")
            print(f"   Status: INACTIVE")
        print(f"📝 Prompt Order: {'Reverse (oldest first)' if self.reverse_prompts else 'Normal (newest first)'}")
        print("="*60)
        
        # Episode tracking
        self.episode_stats = []
        self.global_principles = []
        # Track best prompt and score for each test prompt across all episodes
        self.best_prompts = {prompt: {"score": 0.0, "prompt": prompt} for prompt in self.test_prompts}
        self.enable_reproducibility_optimization = REPRODUCIBILITY_SYSTEM_AVAILABLE
        # Initialize reproducibility system
        if REPRODUCIBILITY_SYSTEM_AVAILABLE:
            self.reproducibility_system = LLMClosePromptReproducibility(
                episodic_memory_file=os.path.join(self.log_dir, "episodic_memory.json"),
                use_vllm=self.use_vllm,
                vllm_url=self.vllm_url,
                vllm_model=self.vllm_model
            )
            self.logger.info("🔄 Initialized reproducibility system for pre-optimization")
        else:
            self.reproducibility_system = None
            self.logger.info("⚠️ Reproducibility system not available")

    def _check_for_duplicate_prompts(self, new_prompts: list, pyfile_path: str = "episodic_test_prompts.py") -> list:
        """Check for duplicate prompts and return only unique ones that aren't already in the file"""
        try:
            # Get existing prompts from the file
            existing_prompts = set()
            try:
                with open(pyfile_path, 'r') as f:
                    lines = f.readlines()
                for line in lines:
                    m = re.match(r'\s*"(.*)"\s*,?\s*$', line)
                    if m:
                        existing_prompts.add(m.group(1).strip())
            except FileNotFoundError:
                # File doesn't exist yet, so no existing prompts
                pass
            
            # Also check against current test_prompts if available
            if hasattr(self, 'test_prompts'):
                existing_prompts.update(self.test_prompts)
            
            # Filter out duplicates and already existing prompts
            unique_new_prompts = []
            seen_in_new = set()
            
            for prompt in new_prompts:
                if isinstance(prompt, str):
                    cleaned = prompt.strip()
                    # Remove any surrounding quotes if they exist
                    if cleaned.startswith("'") and cleaned.endswith("'"):
                        cleaned = cleaned[1:-1]
                    elif cleaned.startswith('"') and cleaned.endswith('"'):
                        cleaned = cleaned[1:-1]
                    
                    # Only add if it's a valid prompt, not already existing, and not a duplicate within new prompts
                    if cleaned and cleaned not in existing_prompts and cleaned not in seen_in_new:
                        unique_new_prompts.append(cleaned)
                        seen_in_new.add(cleaned)
            
            if len(unique_new_prompts) < len(new_prompts):
                duplicate_count = len(new_prompts) - len(unique_new_prompts)
                print(f"[INFO] Filtered out {duplicate_count} duplicate/existing prompts")
            
            return unique_new_prompts
            
        except Exception as e:
            print(f"[WARN] Could not check for duplicate prompts: {e}")
            # Return original prompts if checking fails
            return new_prompts

    def _update_test_prompts_from_log(self, log_path: str):
        """Extract 0-fidelity prompts from log and add them to episodic_test_prompts.py"""
        try:
            # First, clean up any existing malformed prompts in the file
            self._cleanup_malformed_prompts()
            
            zero_fid_prompts = self._extract_zero_fidelity_prompts(log_path)
            if zero_fid_prompts:
                print(f"[INFO] Found {len(zero_fid_prompts)} 0-fidelity prompts in {log_path}")
                
                # Check for duplicates before adding
                unique_prompts = self._check_for_duplicate_prompts(zero_fid_prompts)
                
                if unique_prompts:
                    self._add_prompts_to_pyfile(unique_prompts)
                else:
                    print(f"[INFO] All prompts were duplicates or already exist")
            else:
                print(f"[INFO] No 0-fidelity prompts found in {log_path}")
        except Exception as e:
            print(f"[WARN] Could not update test prompts from log: {e}")
        
    def _cleanup_malformed_prompts(self):
        """Clean up any malformed prompts in the episodic_test_prompts.py file"""
        try:
            pyfile_path = "episodic_test_prompts.py"
            with open(pyfile_path, 'r') as f:
                lines = f.readlines()
            
            cleaned_lines = []
            malformed_count = 0
            
            for line in lines:
                # Check if this line contains a prompt
                if re.match(r'\s*["\'](.*)["\']\s*,?\s*$', line):
                    # Extract the prompt content
                    match = re.match(r'\s*["\'](.*)["\']\s*,?\s*$', line)
                    if match:
                        prompt_content = match.group(1)
                        
                        # Check if the prompt has quotes around it (malformed)
                        if (prompt_content.startswith("'") and prompt_content.endswith("'")) or \
                           (prompt_content.startswith('"') and prompt_content.endswith('"')):
                            # This is malformed - remove the outer quotes
                            cleaned_content = prompt_content[1:-1]
                            cleaned_lines.append(f'    "{cleaned_content}",\n')
                            malformed_count += 1
                        else:
                            # This is properly formatted
                            cleaned_lines.append(line)
                    else:
                        cleaned_lines.append(line)
                else:
                    # Not a prompt line, keep as-is
                    cleaned_lines.append(line)
            
            # Only write if we found malformed prompts
            if malformed_count > 0:
                with open(pyfile_path, 'w') as f:
                    f.writelines(cleaned_lines)
                print(f"[INFO] Cleaned up {malformed_count} malformed prompts in {pyfile_path}")
            
        except Exception as e:
            print(f"[WARN] Could not cleanup malformed prompts: {e}")

    def _validate_prompts_file(self):
        """Validate that the prompts file is properly formatted and clean any issues"""
        try:
            pyfile_path = "episodic_test_prompts.py"
            
            # First, clean up any malformed prompts
            self._cleanup_malformed_prompts()
            
            # Then validate the file structure
            with open(pyfile_path, 'r') as f:
                content = f.read()
            
            # Check if the file has the expected structure
            if 'EPISODIC_TEST_PROMPTS = [' not in content:
                print(f"[WARN] {pyfile_path} does not have expected structure")
                return
            
            # Validate that all prompts are properly formatted
            lines = content.split('\n')
            valid_prompts = []
            invalid_count = 0
            
            for line in lines:
                line = line.strip()
                if line.startswith('"') and line.endswith('",'):
                    # This looks like a valid prompt line
                    prompt_content = line[1:-2]  # Remove " and ,
                    if prompt_content and not prompt_content.startswith("'") and not prompt_content.startswith('"'):
                        valid_prompts.append(prompt_content)
                    else:
                        invalid_count += 1
                elif line.startswith('"') and line.endswith('"'):
                    # This looks like a valid prompt line (last item)
                    prompt_content = line[1:-1]  # Remove "
                    if prompt_content and not prompt_content.startswith("'") and not prompt_content.startswith('"'):
                        valid_prompts.append(prompt_content)
                    else:
                        invalid_count += 1
            
            if invalid_count > 0:
                print(f"[WARN] Found {invalid_count} invalid prompt formats in {pyfile_path}")
            
            print(f"[INFO] Validated {pyfile_path}: {len(valid_prompts)} valid prompts found")
            
        except Exception as e:
            print(f"[WARN] Could not validate prompts file: {e}")

    def _extract_zero_fidelity_prompts(self, log_path: str) -> list:
        """Parse the log file and extract prompts with Task fidelity: 0.0000"""
        prompts = []
        try:
            with open(log_path, 'r') as f:
                lines = f.readlines()
            for i, line in enumerate(lines):
                if 'Task fidelity: 0.0000' in line:
                    # Search backwards for the 'Original:' or 'Generating 3D model:' line
                    for j in range(i-1, max(i-20, -1), -1):
                        if 'Original:' in lines[j]:
                            # Example: 'Original: spear with white head and green shaft'
                            match = re.search(r'Original:\s*(.*)', lines[j])
                            if match:
                                prompts.append(match.group(1).strip())
                                break
                        elif 'Generating 3D model:' in lines[j]:
                            # Example: 'Generating 3D model: ...'
                            match = re.search(r'Generating 3D model:\s*\'(.*?)\' \(task:', lines[j])
                            if match:
                                prompts.append(match.group(1).strip())
                                break
        except Exception as e:
            print(f"[WARN] Could not parse log for 0-fidelity prompts: {e}")
        # Return in reverse chronological order (most recent first)
        return list(reversed(prompts))

    def _check_for_new_zero_fidelity_prompts(self, log_path: str = "continuous_trellis.log"):
        """Check for new 0-fidelity prompts and update the test prompts file immediately"""
        try:
            new_prompts = self._extract_zero_fidelity_prompts(log_path)
            if new_prompts:
                # Use the new duplicate checking function for consistency
                unique_new_prompts = self._check_for_duplicate_prompts(new_prompts)
                if unique_new_prompts:
                    self._add_prompts_to_pyfile(unique_new_prompts)
                    # Reload the prompts from the updated file
                    self._reload_test_prompts()
                    self.logger.info(f"[INFO] Added {len(unique_new_prompts)} new 0-fidelity prompts during episode")
                    # Update best_prompts dict for new prompts
                    for prompt in unique_new_prompts:
                        if prompt not in self.best_prompts:
                            self.best_prompts[prompt] = {"score": 0.0, "prompt": prompt}
                else:
                    self.logger.info(f"[INFO] No new unique prompts found during episode")
        except Exception as e:
            self.logger.warning(f"[WARN] Could not check for new 0-fidelity prompts: {e}")

    def _reload_test_prompts(self):
        """Reload test prompts from the updated episodic_test_prompts.py file"""
        try:
            # Reload the module to get updated prompts
            import importlib
            import episodic_test_prompts
            importlib.reload(episodic_test_prompts)
            # Update our test_prompts with the reloaded data, using the cleaned version
            self.test_prompts = self._get_uncommented_prompts()
            self.logger.info(f"[INFO] Reloaded test prompts: {len(self.test_prompts)} total prompts")
        except Exception as e:
            self.logger.warning(f"[WARN] Could not reload test prompts: {e}")
            # Fallback: manually read the file
            try:
                with open("episodic_test_prompts.py", 'r') as f:
                    content = f.read()
                # Extract prompts from the file content
                import re
                prompts = re.findall(r'"(.*?)"', content)
                self.test_prompts = [p for p in prompts if p.strip()]
                self.logger.info(f"[INFO] Manually reloaded test prompts: {len(self.test_prompts)} total prompts")
            except Exception as e2:
                self.logger.error(f"[ERROR] Failed to reload test prompts: {e2}")

    def _build_improvement_context(self, prompt: str, prompt_results: list, min_similarity: float = 0.51) -> str:
        # Use best-so-far for this prompt across all episodes
        
        if self.enable_reproducibility_optimization:
            try:
                repro_result = self.reproducibility_system.optimize_prompt_with_reproducibility(
                    prompt, min_similarity, run_validation=False
                )
                similarity = 0.0
                gold_score = 0.0
                if repro_result:
                    similarity = repro_result['similarity']
                    gold_score = repro_result['gold_score']
                    self.logger.info(f"🔄 Reproducibility optimization applied:")
                    self.logger.info(f"   Original: {prompt}")
                    self.logger.info(f"   Optimized: {repro_result['optimized_prompt']}")
                    self.logger.info(f"   Similarity: {similarity:.3f}")
                    self.logger.info(f"   Gold score: {gold_score:.4f}")
                    optimized_prompt = repro_result['optimized_prompt']
                else:
                    optimized_prompt = prompt
                    self.logger.info(f"🔄 Reproducibility optimization returned no result")
            except Exception as e:
                optimized_prompt = prompt
                self.logger.info(f"🔄 Reproducibility optimization failed: {e}")
        else:
            optimized_prompt = prompt
            self.logger.info(f"🔄 Reproducibility optimization disabled")

        best_so_far = self.best_prompts.get(prompt, {"score": 0.0, "prompt": prompt})
        if best_so_far["score"] > 0.0:
            return (
                f"--- BEST SO FAR ---\n"
                f"Prompt: '{best_so_far['prompt']}'\n"
                f"Score: {best_so_far['score']:.3f}\n"
                f"Pattern matching reconstruction of closest previously seen prompt: {optimized_prompt}\n"
                "Your goal is to produce a prompt that scores higher than this. "
                "If you cannot, explain why and try a different approach or strategy. "
                "Address all feedback directly in your next attempt.\n"
            )
        else:
            return (
                "This is your first attempt for this prompt in this episode. "
                "Focus on producing the highest scoring prompt possible. "
                "If you do not succeed, analyze why and try a different approach next time.\n"
            )

    def _wait_for_server_availability(self) -> bool:
        """
        Wait for the server to become available before starting optimization.
        
        Returns:
            True if server is available, False if timeout reached
        """
        self.logger.info("🔍 Checking server availability before optimization...")
        
        if not self.server_coordinator.wait_for_server_availability():
            self.logger.warning("⏰ Server availability timeout - skipping this prompt")
            return False
        
        # Mark server as used to start buffer time
        self.server_coordinator.mark_server_used()
        self.logger.info("✅ Server is available and marked as used")
        return True

    def _wait_for_ollama_availability(self, priority: int = 2, description: str = "RL optimization") -> Tuple[bool, str]:
        """
        Wait for Ollama server to become available with priority-based queuing.
        
        Args:
            priority: Priority level (1=HIGH, 2=MEDIUM, 3=LOW)
            description: Description of the request for logging
            
        Returns:
            Tuple of (success, request_id) where request_id is None if failed
        """
        # Skip Ollama coordination if using vLLM
        if self.use_vllm:
            self.logger.info(f"🔄 Using vLLM - skipping Ollama coordination")
            print(f"      🤖 [vLLM MODE] Skipping Ollama coordination - using vLLM directly")
            return True, "vllm_mode"
            
        self.logger.info(f"🔍 Requesting Ollama access (priority: {priority}) for: {description}")
        print(f"      🔍 [Ollama] Requesting access (priority: {priority}) for: {description}")
        
        # Request access to Ollama
        request_id = self.ollama_coordinator.request_access(
            priority=priority,
            description=description
        )
        
        # Wait for access to be granted
        if self.ollama_coordinator.wait_for_access(request_id):
            self.logger.info(f"✅ Ollama access granted for request: {request_id}")
            print(f"      ✅ [Ollama] Access granted for request: {request_id}")
            return True, request_id
        else:
            self.logger.warning(f"⏰ Ollama access timeout for request: {request_id}")
            print(f"      ⏰ [Ollama] Access timeout for request: {request_id}")
            # Cancel the request since we're not using it
            self.ollama_coordinator.cancel_request(request_id)
            return False, None

    

    def _fix_prompts_file(self):
        """Standalone function to fix malformed prompts in episodic_test_prompts.py"""
        try:
            pyfile_path = "episodic_test_prompts.py"
            
            if not os.path.exists(pyfile_path):
                print(f"[ERROR] {pyfile_path} not found")
                return False
            
            print(f"[INFO] Fixing malformed prompts in {pyfile_path}...")
            
            with open(pyfile_path, 'r') as f:
                lines = f.readlines()
            
            cleaned_lines = []
            malformed_count = 0
            
            for line in lines:
                # Check if this line contains a prompt
                if re.match(r'\s*["\'](.*)["\']\s*,?\s*$', line):
                    # Extract the prompt content
                    match = re.match(r'\s*["\'](.*)["\']\s*,?\s*$', line)
                    if match:
                        prompt_content = match.group(1)
                        
                        # Check if the prompt has quotes around it (malformed)
                        if (prompt_content.startswith("'") and prompt_content.endswith("'")) or \
                        (prompt_content.startswith('"') and prompt_content.endswith('"')):
                            # This is malformed - remove the outer quotes
                            cleaned_content = prompt_content[1:-1]
                            cleaned_lines.append(f'    "{cleaned_content}",\n')
                            malformed_count += 1
                            print(f"[FIX] Fixed: '{prompt_content}' -> '{cleaned_content}'")
                        else:
                            # This is properly formatted
                            cleaned_lines.append(line)
                    else:
                        cleaned_lines.append(line)
                else:
                    # Not a prompt line, keep as-is
                    cleaned_lines.append(line)
            
            # Write the cleaned content back
            if malformed_count > 0:
                with open(pyfile_path, 'w') as f:
                    f.writelines(cleaned_lines)
                print(f"[SUCCESS] Fixed {malformed_count} malformed prompts in {pyfile_path}")
                return True
            else:
                print(f"[INFO] No malformed prompts found in {pyfile_path}")
                return True
                
        except Exception as e:
            print(f"[ERROR] Could not fix prompts file: {e}")
            return False

    def _get_episode_prompt_order(self):
        """
        Create a stable prompt processing order for this episode.
        
        This method establishes a consistent processing order for the entire episode,
        preventing confusion from dynamic list updates during processing.
        
        Returns:
            List of prompts in the order they should be processed for this episode.
            
        Ordering logic:
        - Normal mode (reverse_prompts=False): Process prompts in order they appear in the list
          (typically newest prompts are added to the end, so this processes newest first)
        - Reverse mode (reverse_prompts=True): Process prompts in reverse order
          (oldest prompts from the beginning of the list are processed first)
          
        Example:
        If test_prompts = ["old_prompt1", "old_prompt2", "new_prompt3", "new_prompt4"]
        
        Normal mode (reverse_prompts=False):
        - Processing order: ["old_prompt1", "old_prompt2", "new_prompt3", "new_prompt4"]
        - First processed: "old_prompt1" (oldest)
        - Last processed: "new_prompt4" (newest)
        
        Reverse mode (reverse_prompts=True):
        - Processing order: ["new_prompt4", "new_prompt3", "old_prompt2", "old_prompt1"]
        - First processed: "new_prompt4" (newest)
        - Last processed: "old_prompt1" (oldest)
        """
        # Create a copy to avoid modifying the original list
        episode_prompts = self.test_prompts.copy()
        
        if self.reverse_prompts:
            # Reverse mode: process oldest first (beginning of list first)
            # This means prompts at index 0, 1, 2... will be processed first
            episode_prompts.reverse()
            self.logger.info(f"🔄 [REVERSE MODE] Prompt order established: oldest first")
            self.logger.info(f"   First prompt to process: '{episode_prompts[0] if episode_prompts else 'None'}' (was at beginning of original list)")
            self.logger.info(f"   Last prompt to process: '{episode_prompts[-1] if episode_prompts else 'None'}' (was at end of original list)")
        else:
            # Normal mode: process newest first (end of list first)
            # This means prompts at index -1, -2, -3... will be processed first
            # No need to reverse - just process in the order they appear
            self.logger.info(f"🔄 [NORMAL MODE] Prompt order established: newest first")
            self.logger.info(f"   First prompt to process: '{episode_prompts[-1] if episode_prompts else 'None'}' (newest, from end of list)")
            self.logger.info(f"   Last prompt to process: '{episode_prompts[0] if episode_prompts else 'None'}' (oldest, from beginning of list)")
        
        return episode_prompts
    
    def show_prompt_order(self):
        """
        Display the current prompt order configuration for debugging and clarity.
        """
        print(f"\n{'='*60}")
        print("📝 PROMPT ORDER CONFIGURATION")
        print(f"{'='*60}")
        print(f"Total prompts: {len(self.test_prompts)}")
        print(f"Reverse mode: {'YES' if self.reverse_prompts else 'NO'}")
        print(f"Processing order: {'Oldest first' if self.reverse_prompts else 'Newest first'}")
        print()
        
        if self.test_prompts:
            print("Prompt list (in order they appear in file):")
            for i, prompt in enumerate(self.test_prompts):
                marker = "🔄 FIRST" if i == 0 else "⏳ MIDDLE" if i < len(self.test_prompts) - 1 else "✅ LAST"
                print(f"  {i+1:2d}. {marker} - '{prompt[:60]}{'...' if len(prompt) > 60 else ''}'")
            
            print()
            print("Processing order for this episode:")
            episode_order = self._get_episode_prompt_order()
            for i, prompt in enumerate(episode_order):
                marker = "🚀 FIRST" if i == 0 else "⏳ MIDDLE" if i < len(episode_order) - 1 else "🏁 LAST"
                print(f"  {i+1:2d}. {marker} - '{prompt[:60]}{'...' if len(prompt) > 60 else ''}'")
        else:
            print("⚠️ No prompts available")
        
        print(f"{'='*60}")

    def run_single_episode(self, episode_num: int) -> Dict[str, Any]:
        """
        Run a single episode through all test prompts with dynamic updates.
        
        Args:
            episode_num: Current episode number (1-indexed)
            
        Returns:
            Dictionary containing episode statistics
        """
        self.logger.info(f"\n{'='*60}")
        self.logger.info(f"STARTING EPISODE {episode_num}/{self.num_episodes}")
        self.logger.info(f"{'='*60}")
        
        # Log prompt processing order for this episode
        if self.reverse_prompts:
            self.logger.info(f"📝 [EPISODE {episode_num}] Processing prompts in REVERSE order (oldest first)")
            self.logger.info(f"📝 [EPISODE {episode_num}] Episode prompt sequence: {len(self.test_prompts)} prompts")
        else:
            self.logger.info(f"📝 [EPISODE {episode_num}] Processing prompts in NORMAL order (newest first)")
            self.logger.info(f"📝 [EPISODE {episode_num}] Episode prompt sequence: {len(self.test_prompts)} prompts")
        
        # Check server health at the beginning of each episode
        self.logger.info("🔍 Checking server health at episode start...")
        server_status = self.server_coordinator.check_server_status()
        if not server_status.get("available", False) and server_status.get("status") != "buffer_time":
            self.logger.warning(f"⚠️ Server health check failed: {server_status.get('status', 'unknown')} - {server_status.get('error', 'unknown error')}")
            self.logger.info("⏳ Waiting for server to become healthy...")
            if not self.server_coordinator.wait_for_server_availability():
                self.logger.error("❌ Server failed to become healthy - aborting episode")
                return {
                    'episode': episode_num,
                    'start_time': datetime.now().isoformat(),
                    'prompt_results': [],
                    'episode_summary': {
                        'error': 'Server health check failed',
                        'total_prompts': 0,
                        'successful_optimizations': 0,
                        'success_rate': 0.0,
                        'total_rounds': 0,
                        'avg_rounds_per_prompt': 0,
                        'total_score_improvement': 0.0,
                        'avg_score_improvement': 0.0,
                        'episode_duration_seconds': 0.0,
                        'principles_learned': [],
                        'end_time': datetime.now().isoformat()
                    }
                }
        else:
            self.logger.info("✅ Server health check passed")
        
        episode_start_time = time.time()
        episode_results = {
            'episode': episode_num,
            'start_time': datetime.now().isoformat(),
            'prompt_results': [],
            'episode_summary': {}
        }
        
        total_rounds = 0
        successful_optimizations = 0
        total_score_improvement = 0.0
        episode_principles = []
        
        # Track processed prompts to avoid duplicates
        processed_prompts = set()
        prompt_idx = 0
        
        # Create a stable prompt processing order for this episode
        # This ensures consistent ordering even if the list changes during the episode
        episode_prompt_order = self._get_episode_prompt_order()
        
        self.logger.info(f"📝 [EPISODE {episode_num}] Prompt processing order established:")
        # for i, prompt in enumerate(episode_prompt_order):
        #     status = "✅ PROCESSED" if prompt in processed_prompts else "⏳ PENDING"
        #     self.logger.info(f"   {i+1:2d}. {status} - '{prompt[:50]}{'...' if len(prompt) > 50 else ''}'")
        
        # Process prompts in the established order
        for prompt in episode_prompt_order:
            # Skip if already processed (shouldn't happen, but safety check)
            if prompt in processed_prompts:
                self.logger.warning(f"⚠️ Prompt already processed, skipping: '{prompt}'")
                continue
                
            prompt_idx += 1
            processed_prompts.add(prompt)
            
            self.logger.info(f"\n--- Episode {episode_num}, Prompt {prompt_idx}/{len(episode_prompt_order)} ---")
            self.logger.info(f"Optimizing: '{prompt}'")
            
            # Wait for server availability before starting optimization
            if not self._wait_for_server_availability():
                # Skip this prompt if server is not available
                episode_results['prompt_results'].append({
                    'prompt': prompt,
                    'prompt_index': prompt_idx,
                    'error': 'Server availability timeout',
                    'rounds_used': 0,
                    'converged': False
                })
                continue
            
            # Note: Ollama coordination is now handled at the round level within the RL agent
            # This allows for more efficient resource utilization during validation steps
            
            prompt_start_time = time.time()
            
            try:
                # Build improvement context for this prompt
                improvement_context = self._build_improvement_context(prompt, episode_results['prompt_results'])
                # Prepend context to prompt for RL agent
                prompt_with_context = f"{improvement_context}"
                max_retries = 3
                retry_count = 0
                result = None
                while retry_count < max_retries:
                    result = self.optimizer.optimize_with_rl_loop(prompt, prompt_with_context=prompt_with_context, endpoint=self.endpoint)
                    final_score = result.get('final_score', 0.0)
                    if final_score > 0.0:
                        break
                    else:
                        self.logger.warning(f"Validation score 0.0 detected (likely CUDA OOM or failure). Clearing server cache and retrying ({retry_count+1}/{max_retries})...")
                        try:
                            # Clear server GPU cache
                            if self.server_coordinator.clear_server_cache():
                                self.logger.info("Server GPU cache cleared successfully.")
                            else:
                                self.logger.warning("Failed to clear server GPU cache.")
                            
                            # Also clear local CUDA cache if available
                            import torch
                            if torch.cuda.is_available():
                                torch.cuda.empty_cache()
                                self.logger.info("Local CUDA cache cleared.")
                        except Exception as e:
                            self.logger.warning(f"Failed to clear caches: {e}")
                        retry_count += 1
                        if retry_count < max_retries:
                            time.sleep(5)  # Longer wait after cache clearing
                prompt_duration = time.time() - prompt_start_time
                # Extract results
                rounds_used = result.get('total_rounds', 0)
                final_score = result.get('final_score', 0.0)
                initial_score = result.get('score_progression', [0.0])[0] if result.get('score_progression') else 0.0
                score_improvement = final_score - initial_score
                converged = result.get('convergence_achieved', False)
                # Update episode totals
                total_rounds += rounds_used
                if converged:
                    successful_optimizations += 1
                total_score_improvement += score_improvement
                # Extract any new principles learned
                if 'learned_insights' in result:
                    episode_principles.extend(result['learned_insights'])
                # Log prompt result
                prompt_result = {
                    'prompt': prompt,
                    'prompt_index': prompt_idx,
                    'rounds_used': rounds_used,
                    'initial_score': initial_score,
                    'final_score': final_score,
                    'score_improvement': score_improvement,
                    'converged': converged,
                    'duration_seconds': prompt_duration,
                    'optimized_prompt': result.get('final_optimized_prompt', prompt)
                }
                episode_results['prompt_results'].append(prompt_result)
                # Update best prompt/score for this prompt across all episodes
                if final_score > self.best_prompts[prompt]["score"]:
                    self.best_prompts[prompt]["score"] = final_score
                    self.best_prompts[prompt]["prompt"] = result.get('final_optimized_prompt', prompt)
                self.logger.info(f"Prompt optimized in {rounds_used} rounds: {initial_score:.3f} → {final_score:.3f} (+{score_improvement:.3f})")
                if converged:
                    self.logger.info("✅ Target score achieved!")
                else:
                    self.logger.info("⏰ Max rounds reached")
                
                # Note: Ollama access is automatically released after each round within the RL agent
                    
            except Exception as e:
                self.logger.error(f"Error optimizing prompt '{prompt}': {str(e)}")
                # Add failed result
                episode_results['prompt_results'].append({
                    'prompt': prompt,
                    'prompt_index': prompt_idx,
                    'error': str(e),
                    'rounds_used': 0,
                    'converged': False
                })
        
        # Calculate episode statistics
        episode_duration = time.time() - episode_start_time
        total_prompts_processed = len(processed_prompts)
        avg_rounds = total_rounds / total_prompts_processed if total_prompts_processed > 0 else 0
        success_rate = successful_optimizations / total_prompts_processed if total_prompts_processed > 0 else 0
        avg_score_improvement = total_score_improvement / total_prompts_processed if total_prompts_processed > 0 else 0
        
        episode_summary = {
            'total_prompts': total_prompts_processed,
            'successful_optimizations': successful_optimizations,
            'success_rate': success_rate,
            'total_rounds': total_rounds,
            'avg_rounds_per_prompt': avg_rounds,
            'total_score_improvement': total_score_improvement,
            'avg_score_improvement': avg_score_improvement,
            'episode_duration_seconds': episode_duration,
            'principles_learned': episode_principles,
            'end_time': datetime.now().isoformat()
        }
        
        episode_results['episode_summary'] = episode_summary
        
        # Log episode summary
        self.logger.info(f"\n{'='*60}")
        self.logger.info(f"EPISODE {episode_num} SUMMARY")
        self.logger.info(f"{'='*60}")
        self.logger.info(f"Success Rate: {success_rate:.1%} ({successful_optimizations}/{total_prompts_processed})")
        self.logger.info(f"Average Rounds: {avg_rounds:.1f}")
        self.logger.info(f"Average Score Improvement: {avg_score_improvement:+.3f}")
        self.logger.info(f"Episode Duration: {episode_duration:.1f}s")
        self.logger.info(f"New Principles Learned: {len(episode_principles)}")
        
        # Log Ollama queue status
        if self.ollama_coordinator:
            ollama_status = self.ollama_coordinator.get_queue_status()
            self.logger.info(f"Ollama Queue Status: {ollama_status['queue_length']} pending, {ollama_status['active_requests']} active")
            print(f"Ollama Queue Status: {ollama_status['queue_length']} pending, {ollama_status['active_requests']} active")
        elif self.use_vllm:
            print(f"vLLM Status: Direct server access - no queue management")
        else:
            print(f"LLM Status: No coordinator configured")
        
        # Update global principles
        self.global_principles.extend(episode_principles)
        
        return episode_results
    
    def analyze_cross_episode_learning(self) -> Dict[str, Any]:
        """
        Analyze learning patterns across episodes.
        
        Returns:
            Dictionary containing cross-episode analysis
        """
        if not self.episode_stats:
            return {}
        
        # Extract metrics across episodes
        success_rates = [ep['episode_summary']['success_rate'] for ep in self.episode_stats]
        avg_rounds = [ep['episode_summary']['avg_rounds_per_prompt'] for ep in self.episode_stats]
        avg_improvements = [ep['episode_summary']['avg_score_improvement'] for ep in self.episode_stats]
        
        # Calculate trends
        analysis = {
            'total_episodes': len(self.episode_stats),
            'success_rate_trend': {
                'first_5_episodes': statistics.mean(success_rates[:5]) if len(success_rates) >= 5 else None,
                'last_5_episodes': statistics.mean(success_rates[-5:]) if len(success_rates) >= 5 else None,
                'overall_average': statistics.mean(success_rates),
                'final_episode': success_rates[-1] if success_rates else None
            },
            'efficiency_trend': {
                'first_5_episodes_avg_rounds': statistics.mean(avg_rounds[:5]) if len(avg_rounds) >= 5 else None,
                'last_5_episodes_avg_rounds': statistics.mean(avg_rounds[-5:]) if len(avg_rounds) >= 5 else None,
                'overall_average_rounds': statistics.mean(avg_rounds),
                'final_episode_rounds': avg_rounds[-1] if avg_rounds else None
            },
            'improvement_trend': {
                'first_5_episodes_avg_improvement': statistics.mean(avg_improvements[:5]) if len(avg_improvements) >= 5 else None,
                'last_5_episodes_avg_improvement': statistics.mean(avg_improvements[-5:]) if len(avg_improvements) >= 5 else None,
                'overall_average_improvement': statistics.mean(avg_improvements),
                'final_episode_improvement': avg_improvements[-1] if avg_improvements else None
            },
            'total_principles_learned': len(self.global_principles),
            'unique_principles': len(set(self.global_principles)) if self.global_principles else 0
        }
        
        return analysis
    
    def run_all_episodes(self) -> Dict[str, Any]:
        """
        Run all episodes and return comprehensive results.
        
        Returns:
            Dictionary containing all episode results and analysis
        """
        self.logger.info(f"Starting episodic optimization: {self.num_episodes} episodes, {len(self.test_prompts)} prompts per episode")
        self.logger.info(f"Target score: {self.target_score}, Max rounds per prompt: {self.max_rounds_per_prompt}")
        
        overall_start_time = time.time()
        
        # Run each episode
        for episode_num in range(1, self.num_episodes + 1):
            try:
                episode_result = self.run_single_episode(episode_num)
                self.episode_stats.append(episode_result)
                
                # Save intermediate results
                self.save_results(episode_num)
                
                # Brief pause between episodes
                if episode_num < self.num_episodes:
                    # import time
                    time.sleep(2)
                    
            except Exception as e:
                self.logger.error(f"Error in episode {episode_num}: {str(e)}")
                continue
        
        overall_duration = time.time() - overall_start_time
        
        # Perform cross-episode analysis
        learning_analysis = self.analyze_cross_episode_learning()
        
        # Compile final results
        final_results = {
            'run_metadata': {
                'num_episodes': self.num_episodes,
                'target_score': self.target_score,
                'max_rounds_per_prompt': self.max_rounds_per_prompt,
                'total_prompts_per_episode': len(self.test_prompts),
                'total_optimizations': self.num_episodes * len(self.test_prompts),
                'overall_duration_seconds': overall_duration,
                'start_time': datetime.now().isoformat()
            },
            'episode_results': self.episode_stats,
            'learning_analysis': learning_analysis,
            'test_prompts_used': self.test_prompts
        }
        
        # Save final comprehensive results
        self.save_final_results(final_results)
        
        # Log final summary
        self.log_final_summary(learning_analysis, overall_duration)
        
        # Print final LLM provider summary
        print(f"\n{'='*60}")
        print("🤖 FINAL LLM PROVIDER SUMMARY")
        print(f"{'='*60}")
        if self.use_vllm:
            print(f"✅ vLLM used throughout: {self.vllm_url}")
            print(f"   Model: {self.vllm_model}")
            print(f"   Total episodes: {self.num_episodes}")
            print(f"   Total optimizations: {self.num_episodes * len(self.test_prompts)}")
        elif self.ollama_coordinator:
            print(f"✅ Ollama used throughout: {self.ollama_coordinator.ollama_url}")
            print(f"   Coordinator: ACTIVE with priority queuing")
            print(f"   Total episodes: {self.num_episodes}")
            print(f"   Total optimizations: {self.num_episodes * len(self.test_prompts)}")
        else:
            print(f"⚠️ No LLM coordinator was configured")
        print(f"{'='*60}")
        
        return final_results
    
    def save_results(self, episode_num: int):
        """Save intermediate results after each episode."""
        try:
            intermediate_file = os.path.join(self.log_dir, f"episodes_1_to_{episode_num}_results.json")
            with open(intermediate_file, 'w') as f:
                json.dump({
                    'episodes_completed': episode_num,
                    'episode_results': self.episode_stats,
                    'test_prompts': self.test_prompts
                }, f, indent=2)
        except Exception as e:
            self.logger.error(f"Error saving intermediate results: {str(e)}")
    
    def save_final_results(self, results: Dict[str, Any]):
        """Save final comprehensive results, including best prompts."""
        try:
            final_file = os.path.join(self.log_dir, f"final_episodic_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
            results['best_prompts'] = self.best_prompts
            with open(final_file, 'w') as f:
                json.dump(results, f, indent=2)
            self.logger.info(f"Final results saved to: {final_file}")
            # Print summary table of best prompts and scores
            self.logger.info("\nBest prompts and scores for each test prompt:")
            for prompt, data in self.best_prompts.items():
                self.logger.info(f"Prompt: {prompt}\n  Best Score: {data['score']:.3f}\n  Best Prompt: {data['prompt']}\n")
        except Exception as e:
            self.logger.error(f"Error saving final results: {str(e)}")
    
    def log_final_summary(self, learning_analysis: Dict[str, Any], overall_duration: float):
        """Log comprehensive final summary."""
        self.logger.info(f"\n{'='*80}")
        self.logger.info(f"FINAL EPISODIC OPTIMIZATION SUMMARY")
        self.logger.info(f"{'='*80}")
        
        self.logger.info(f"Total Episodes: {self.num_episodes}")
        self.logger.info(f"Total Optimizations: {self.num_episodes * len(self.test_prompts)}")
        self.logger.info(f"Overall Duration: {overall_duration:.1f}s ({overall_duration/60:.1f} minutes)")
        
        if learning_analysis:
            sr_trend = learning_analysis.get('success_rate_trend', {})
            eff_trend = learning_analysis.get('efficiency_trend', {})
            imp_trend = learning_analysis.get('improvement_trend', {})
            
            self.logger.info(f"\nLEARNING PROGRESSION:")
            
            if sr_trend.get('first_5_episodes') is not None and sr_trend.get('last_5_episodes') is not None:
                sr_change = sr_trend['last_5_episodes'] - sr_trend['first_5_episodes']
                self.logger.info(f"Success Rate: {sr_trend['first_5_episodes']:.1%} → {sr_trend['last_5_episodes']:.1%} ({sr_change:+.1%})")
            
            if eff_trend.get('first_5_episodes_avg_rounds') is not None and eff_trend.get('last_5_episodes_avg_rounds') is not None:
                rounds_change = eff_trend['last_5_episodes_avg_rounds'] - eff_trend['first_5_episodes_avg_rounds']
                self.logger.info(f"Efficiency: {eff_trend['first_5_episodes_avg_rounds']:.1f} → {eff_trend['last_5_episodes_avg_rounds']:.1f} rounds ({rounds_change:+.1f})")
            
            if imp_trend.get('first_5_episodes_avg_improvement') is not None and imp_trend.get('last_5_episodes_avg_improvement') is not None:
                imp_change = imp_trend['last_5_episodes_avg_improvement'] - imp_trend['first_5_episodes_avg_improvement']
                self.logger.info(f"Score Improvement: {imp_trend['first_5_episodes_avg_improvement']:+.3f} → {imp_trend['last_5_episodes_avg_improvement']:+.3f} ({imp_change:+.3f})")
            
            self.logger.info(f"\nPRINCIPLES LEARNED:")
            self.logger.info(f"Total Principles: {learning_analysis.get('total_principles_learned', 0)}")
            self.logger.info(f"Unique Principles: {learning_analysis.get('unique_principles', 0)}")
        
        self.logger.info(f"\n{'='*80}")
    
    def cleanup(self):
        """Cleanup resources and perform final server coordination"""
        self.logger.info("🧹 Performing cleanup...")
        
        # Clear server cache one final time
        try:
            if self.server_coordinator.clear_server_cache():
                self.logger.info("✅ Final server cache clear successful")
            else:
                self.logger.warning("⚠️ Final server cache clear failed")
        except Exception as e:
            self.logger.warning(f"⚠️ Exception during final cleanup: {e}")
        
        # Shutdown Ollama coordinator only if it exists
        try:
            if hasattr(self, 'ollama_coordinator') and self.ollama_coordinator:
                self.ollama_coordinator.shutdown()
                self.logger.info("✅ Ollama coordinator shutdown successful")
        except Exception as e:
            self.logger.warning(f"⚠️ Exception shutting down Ollama coordinator: {e}")
        
        # Save any pending memory
        try:
            if hasattr(self.optimizer, '_save_memory'):
                self.optimizer._save_memory()
                self.logger.info("✅ Memory saved successfully")
        except Exception as e:
            self.logger.warning(f"⚠️ Exception saving memory: {e}")
        
        self.logger.info("✅ Cleanup completed")

def main():
    """Main function to run the episodic optimization."""
    import argparse
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Episodic Prompt Optimizer")
    parser.add_argument("--episodes", type=int, default=30, help="Number of episodes to run (default: 30)")
    parser.add_argument("--target", type=float, default=0.85, help="Target validation score (default: 0.85)")
    parser.add_argument("--max-rounds", type=int, default=5, help="Maximum optimization rounds per prompt (default: 5)")
    parser.add_argument("--log-dir", type=str, default="episodic_logs", help="Directory for storing episode logs (default: episodic_logs)")
    parser.add_argument("--endpoint", type=str, default="generate/cinema/", help="TRELLIS endpoint to use (default: generate/cinema/)")
    parser.add_argument("--ollama-url", type=str, default="http://localhost:11434", help="Ollama server URL (default: http://localhost:11434)")
    parser.add_argument("--port", type=int, default=8096, help="TRELLIS server port (default: 8096)")
    parser.add_argument("--server-buffer-time", type=int, default=30, help="Buffer time between server uses in seconds (default: 30)")
    parser.add_argument("--vllm", action="store_true", help="Use vLLM instead of Ollama")
    parser.add_argument("--vllm-url", type=str, default="http://localhost:9000", help="vLLM server URL (default: http://localhost:9000)")
    parser.add_argument("--vllm-model", type=str, default="llama-3-2-3b-it", help="vLLM model name (default: llama-3-2-3b-it)")
    parser.add_argument("--reverse", action="store_true", help="Process prompts in reverse order (oldest first)")
    parser.add_argument("--fix-prompts", action="store_true", help="Fix malformed prompts in episodic_test_prompts.py and exit")
    parser.add_argument("--show-prompts", action="store_true", help="Show prompt order configuration and exit")
    
    args = parser.parse_args()
    
    # If --fix-prompts is specified, just fix the file and exit
    
    
    # Configuration from command line arguments
    NUM_EPISODES = args.episodes
    TARGET_SCORE = args.target
    MAX_ROUNDS_PER_PROMPT = args.max_rounds
    SERVER_URL = f"http://localhost:{args.port}"
    SERVER_BUFFER_TIME = args.server_buffer_time
    ENDPOINT = args.endpoint
    OLLAMA_URL = args.ollama_url
    USE_VLLM = args.vllm
    VLLM_URL = args.vllm_url
    VLLM_MODEL = args.vllm_model
    REVERSE_PROMPTS = args.reverse
    
    print(f"🚀 Starting Episodic Prompt Optimization")
    print(f"Episodes: {NUM_EPISODES}")
    print(f"Target Score: {TARGET_SCORE}")
    print(f"Max Rounds per Prompt: {MAX_ROUNDS_PER_PROMPT}")
    print(f"Server URL: {SERVER_URL}")
    print(f"Server Port: {args.port}")
    print(f"Server Buffer Time: {SERVER_BUFFER_TIME}s")
    print(f"Endpoint: {ENDPOINT}")
    if USE_VLLM:
        print(f"Using vLLM: {VLLM_URL} with model {VLLM_MODEL}")
    else:
        print(f"Ollama URL: {OLLAMA_URL}")
    print(f"Log Directory: {args.log_dir}")
    print(f"Prompts per Episode: 13")
    print(f"Total Optimizations: {NUM_EPISODES * 13}")
    print(f"Prompt Order: {'Reverse (oldest first)' if REVERSE_PROMPTS else 'Normal (newest first)'}")
    print()
    
    # Print LLM provider configuration
    print("🤖 LLM PROVIDER CONFIGURATION:")
    if USE_VLLM:
        print(f"   ✅ vLLM: {VLLM_URL}")
        print(f"   ✅ Model: {VLLM_MODEL}")
        print(f"   ✅ Status: ACTIVE")
    else:
        print(f"   ✅ Ollama: {OLLAMA_URL}")
        print(f"   ✅ Coordinator: ACTIVE with priority queuing")
        print(f"   ✅ Status: ACTIVE")
    print()
    
    # Create and run the episodic optimizer
    optimizer = EpisodicPromptOptimizer(
        num_episodes=NUM_EPISODES,
        target_score=TARGET_SCORE,
        max_rounds_per_prompt=MAX_ROUNDS_PER_PROMPT,
        log_dir=args.log_dir,
        server_url=SERVER_URL,
        server_buffer_time=SERVER_BUFFER_TIME,
        endpoint=ENDPOINT,
        ollama_url=OLLAMA_URL,
        use_vllm=USE_VLLM,
        vllm_url=VLLM_URL,
        vllm_model=VLLM_MODEL,
        reverse_prompts=REVERSE_PROMPTS
    )
    
    # Handle special arguments that don't require running episodes
    if args.fix_prompts:
        success = optimizer._fix_prompts_file()
        exit(0 if success else 1)
    
    if args.show_prompts:
        optimizer.show_prompt_order()
        exit(0)
    
    try:
        results = optimizer.run_all_episodes()
        print(f"\n✅ Episodic optimization completed successfully!")
        print(f"Results saved to: {optimizer.log_dir}")
        
        return results
        
    except KeyboardInterrupt:
        print(f"\n⏹️  Episodic optimization interrupted by user")
        print(f"Partial results saved to: {optimizer.log_dir}")
        return None
        
    except Exception as e:
        print(f"\n❌ Error during episodic optimization: {str(e)}")
        print(f"Partial results may be saved to: {optimizer.log_dir}")
        return None
    finally:
        # Always perform cleanup
        try:
            optimizer.cleanup()
        except Exception as cleanup_error:
            print(f"⚠️ Cleanup error: {cleanup_error}")


if __name__ == "__main__":
    main() 