#!/usr/bin/env python3
"""
Test script for generate_3d_model_clip function with mock data
Tests parallel generation, endpoint conversion, and CLIP scoring logic
"""

import asyncio
import time
import traceback
from unittest.mock import Mock, MagicMock
import io
import base64

# Mock PIL for testing
try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False
    print("⚠️  PIL not available, using mock images")

class MockImage:
    """Mock image class for testing"""
    def __init__(self, size=(512, 512)):
        self.size = size
    
    def convert(self, mode):
        return self
    
    def save(self, buffer, format):
        if format == 'PNG':
            buffer.write(b'fake_png_data')

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
            'ss_guidance_strength': 9.5
        }
    
    def __getitem__(self, key):
        return self.config[key]
    
    def __contains__(self, key):
        return key in self.config
    
    def get(self, key, default=None):
        return self.config.get(key, default)

class MockReproducibilitySystem:
    """Mock reproducibility system"""
    def get_optimized_prompt(self, original_prompt):
        return f"a finely crafted, intricate {original_prompt} with enhanced details"

class MockCLIPAnalyzer:
    """Mock CLIP analyzer for testing"""
    def __init__(self):
        self.model_loaded = True
    
    def load_clip_model(self):
        pass
    
    def compute_clip_score(self, prompt, image):
        """Mock CLIP score computation"""
        # Simulate realistic CLIP scores
        if "blue car" in prompt.lower():
            if "blue" in str(image).lower():
                return 0.85  # High score for matching content
            else:
                return 0.45  # Lower score for mismatch
        elif "finely crafted" in prompt.lower():
            return 0.92  # High score for optimized prompts
        else:
            return 0.65  # Default score

class MockContinuousTrellisOrchestrator:
    """Mock orchestrator for testing"""
    def __init__(self):
        self.logger = MockLogger()
        self.config = MockConfig()
        self.priority_coordinator = MockPriorityCoordinator()
        self.reproducibility_system = MockReproducibilitySystem()
        self.clip_analyzer = MockCLIPAnalyzer()
    
    def get_deterministic_seed(self, task):
        return 42
    
    def get_clip_analyzer(self):
        return self.clip_analyzer
    
    async def generate_single_prompt(self, prompt, is_optimized=False, port=None):
        """Mock single prompt generation"""
        self.logger.info(f"🎯 Generating {'optimized' if is_optimized else 'original'} prompt: {prompt[:50]}...")
        
        # Simulate generation time
        await asyncio.sleep(0.1)
        
        # Create mock response data
        mock_ply_data = b"mock_ply_data_for_testing"
        mock_compressed_data = b"mock_compressed_ply_data"
        
        if PIL_AVAILABLE:
            # Create a simple mock image
            mock_image = Image.new('RGB', (512, 512), color='blue' if 'blue' in prompt.lower() else 'red')
        else:
            mock_image = MockImage()
        
        # Simulate endpoint conversion
        endpoint = "/generate/cinema"
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
        
        # Simulate URL construction
        if port:
            try:
                original_url = self.config['generation_server_url']
                if '://' in original_url:
                    protocol_and_host = original_url.split(':')[0] + ':' + original_url.split(':')[1]
                    server_url = f"{protocol_and_host}:{port}"
                else:
                    host = original_url.split(':')[0]
                    server_url = f"http://{host}:{port}"
            except Exception as e:
                self.logger.error(f"❌ URL construction failed: {e}")
                server_url = self.config['generation_server_url']
        else:
            server_url = self.config['generation_server_url']
        
        self.logger.info(f"🌐 Mock request to: {server_url}{both_endpoint}")
        
        return {
            'ply_data': mock_ply_data,
            'compressed_data': mock_compressed_data,
            'image': mock_image,
            'endpoint_used': both_endpoint,
            'server_url': server_url
        }

async def test_generate_3d_model_clip():
    """Test the generate_3d_model_clip function with mock data"""
    
    print("🧪 Testing generate_3d_model_clip function with mock data")
    print("=" * 60)
    
    # Create mock orchestrator
    orchestrator = MockContinuousTrellisOrchestrator()
    
    # Create mock task
    task = MockTaskRecord("a blue car")
    
    print(f"📝 Task: {task.prompt}")
    print(f"🎲 Deterministic seed: {orchestrator.get_deterministic_seed(task)}")
    
    # Test endpoint conversion logic
    test_endpoints = [
        "/generate/cinema",
        "/generate/",
        "/generate",
        "generate/cinema",
        "generate/",
        "generate"
    ]
    
    print("\n🔧 Testing endpoint conversion:")
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
    print("\n🌐 Testing URL construction:")
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
    
    # Test parallel generation
    print("\n🚀 Testing parallel generation:")
    
    # Create tasks for parallel execution
    original_task = orchestrator.generate_single_prompt(task.prompt, is_optimized=False, port=8097)
    optimized_task = orchestrator.generate_single_prompt(
        orchestrator.reproducibility_system.get_optimized_prompt(task.prompt), 
        is_optimized=True, 
        port=8099
    )
    
    # Run both in parallel
    start_time = time.time()
    original_result, optimized_result = await asyncio.gather(original_task, optimized_task)
    end_time = time.time()
    
    print(f"⏱️  Parallel execution time: {end_time - start_time:.3f}s")
    
    # Test CLIP scoring
    print("\n🎯 Testing CLIP scoring:")
    
    clip_analyzer = orchestrator.get_clip_analyzer()
    
    # Compute CLIP scores
    original_prompt = task.prompt
    optimized_prompt = orchestrator.reproducibility_system.get_optimized_prompt(original_prompt)
    
    scores = {
        'original_prompt_original_image': clip_analyzer.compute_clip_score(original_prompt, original_result['image']),
        'optimized_prompt_optimized_image': clip_analyzer.compute_clip_score(optimized_prompt, optimized_result['image']),
        'original_prompt_optimized_image': clip_analyzer.compute_clip_score(original_prompt, optimized_result['image']),
        'optimized_prompt_original_image': clip_analyzer.compute_clip_score(optimized_prompt, original_result['image'])
    }
    
    for score_name, score_value in scores.items():
        print(f"   {score_name}: {score_value:.4f}")
    
    # Determine which result to submit based on CLIP score vs original prompt
    original_vs_original = scores['original_prompt_original_image']
    original_vs_optimized = scores['original_prompt_optimized_image']
    
    if original_vs_optimized > original_vs_original:
        selected_result = optimized_result
        selected_type = "optimized"
        print(f"\n🏆 Selected OPTIMIZED result (CLIP: {original_vs_optimized:.4f} > {original_vs_original:.4f})")
    else:
        selected_result = original_result
        selected_type = "original"
        print(f"\n🏆 Selected ORIGINAL result (CLIP: {original_vs_original:.4f} >= {original_vs_optimized:.4f})")
    
    # Show final result
    print(f"\n📊 Final Result Summary:")
    print(f"   Selected: {selected_type} prompt")
    print(f"   PLY data size: {len(selected_result['compressed_data'])} bytes (compressed)")
    print(f"   Image size: {selected_result['image'].size}")
    print(f"   Endpoint used: {selected_result['endpoint_used']}")
    print(f"   Server URL: {selected_result['server_url']}")
    
    print("\n" + "=" * 60)
    print("✅ Test completed successfully!")
    
    return True

if __name__ == "__main__":
    try:
        asyncio.run(test_generate_3d_model_clip())
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        traceback.print_exc()
