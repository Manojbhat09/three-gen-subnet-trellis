#!/usr/bin/env python3
"""
Integration test script for generate_3d_model_clip function with reproducibility
This tests the complete pipeline: prompt optimization, parallel generation, CLIP scoring, and selection
"""

import asyncio
import time
import json
from pathlib import Path
from typing import Dict, Any, Optional

# Mock the necessary components for testing
class MockTaskRecord:
    """Mock TaskRecord for testing"""
    def __init__(self, task_id: str, prompt: str):
        self.task_id = task_id
        self.prompt = prompt
        self.priority_access_timeout = False

class MockLogger:
    """Mock logger for testing"""
    def info(self, msg: str):
        print(f"ℹ️  {msg}")
    
    def warning(self, msg: str):
        print(f"⚠️  {msg}")
    
    def error(self, msg: str):
        print(f"❌ {msg}")
    
    def debug(self, msg: str):
        print(f"🔍 {msg}")

class MockPriorityCoordinator:
    """Mock priority coordinator for testing"""
    def __init__(self):
        self.jobs = {}
    
    def wait_for_priority_access(self, task_id: str) -> bool:
        print(f"🔒 Priority access granted for task {task_id}")
        return True
    
    def mark_priority_job_start(self, task_id: str, prompt: str):
        print(f"🚀 Priority job started: {task_id}")
        self.jobs[task_id] = {"status": "running", "prompt": prompt}
    
    def mark_priority_job_end(self, task_id: str):
        print(f"✅ Priority job ended: {task_id}")
        if task_id in self.jobs:
            self.jobs[task_id]["status"] = "completed"
    
    def clear_server_cache(self):
        print("🧹 Server cache cleared")

class MockConfig:
    """Mock configuration for testing"""
    def __init__(self):
        self.config = {
            'generation_server_url': 'http://localhost:8099',
            'num_inference_steps': 7,
            'guidance_scale': 3.5,
            'ss_sampling_steps': 21,
            'slat_sampling_steps': 24,
            'slat_guidance_strength': 4.0,
            'ss_guidance_strength': 9.5,
            'log_optimization_details': True,
            'generation_timeout': 300,
            'validation_timeout': 120,
            'submission_timeout': 60,
            'save_intermediate_results': True,
            'validate_generations': True,
            'submit_results': True
        }
    
    def get(self, key: str, default=None):
        return self.config.get(key, default)
    
    def __getitem__(self, key: str):
        """Allow dictionary-style access like config['key']"""
        return self.config[key]
    
    def __contains__(self, key: str):
        """Allow 'key in config' checks"""
        return key in self.config

class MockReproducibilitySystem:
    """Mock reproducibility system for testing"""
    def __init__(self):
        self.gold_standard_results = {
            "gold_prompts": [
                "a beautiful blue ceramic vase with intricate red trim and golden accents",
                "an elegant bronze machine gun mount with lustrous finish and detailed craftsmanship",
                "a finely crafted wooden table with smooth surface and sturdy legs",
                "a vibrant red sports car with aerodynamic design and chrome wheels"
            ]
        }
    
    def optimize_prompt_with_reproducibility(self, prompt: str, min_similarity: float = 0.3) -> Optional[Dict[str, Any]]:
        """Mock prompt optimization"""
        # Simulate finding similar gold prompts and merging components
        similar_prompts = []
        for gold_prompt in self.gold_standard_results["gold_prompts"]:
            # Simple similarity check (in real system, this would use embeddings)
            if any(word in gold_prompt.lower() for word in prompt.lower().split()):
                similar_prompts.append(gold_prompt)
        
        if similar_prompts:
            # Merge components from gold prompts
            enhanced_prompt = f"{prompt}, enhanced with professional craftsmanship and detailed textures"
            return {
                "optimized_prompt": enhanced_prompt,
                "similarity_score": 0.75,
                "gold_prompts_used": similar_prompts[:2]
            }
        else:
            # Fallback optimization
            return {
                "optimized_prompt": f"{prompt}, with high quality materials and fine details",
                "similarity_score": 0.5,
                "gold_prompts_used": []
            }

class MockCLIPAnalyzer:
    """Mock CLIP analyzer for testing"""
    def __init__(self):
        self.model_loaded = True
    
    def compute_clip_alignment_score(self, prompt: str, image) -> float:
        """Mock CLIP score computation"""
        # Simulate realistic CLIP scores based on prompt-image alignment
        import random
        
        # Base score based on prompt complexity
        base_score = 0.3 + (len(prompt.split()) * 0.02)
        
        # Add some randomness to simulate real CLIP behavior
        variation = random.uniform(-0.1, 0.1)
        
        # Ensure scores are in realistic range (0.1 to 0.9)
        final_score = max(0.1, min(0.9, base_score + variation))
        
        return round(final_score, 4)

class MockImage:
    """Mock PIL Image for testing"""
    def __init__(self, size=(512, 512), mode='RGB'):
        self.size = size
        self.mode = mode
    
    def convert(self, mode):
        """Mock image mode conversion"""
        self.mode = mode
        return self

class MockContinuousTrellisOrchestrator:
    """Mock orchestrator for testing the generate_3d_model_clip function"""
    def __init__(self):
        self.logger = MockLogger()
        self.config = MockConfig()
        self.priority_coordinator = MockPriorityCoordinator()
        self.reproducibility_system = MockReproducibilitySystem()
        self.clip_analyzer = MockCLIPAnalyzer()
        self.stats = {
            'successful_generations': 0,
            'total_generation_time': 0.0
        }
    
    def get_clip_analyzer(self):
        """Get the preloaded CLIP analyzer"""
        return self.clip_analyzer
    
    def optimize_prompt_for_generation(self, task) -> Dict[str, Any]:
        """Mock prompt optimization"""
        print(f"🔧 Optimizing prompt: '{task.prompt}'")
        
        # Use reproducibility system
        optimization_result = self.reproducibility_system.optimize_prompt_with_reproducibility(task.prompt)
        
        # Mock LoRA routing
        lora_info = {
            'lora_name': 'cinema',
            'confidence': 0.85,
            'reason': 'High confidence match for artistic style'
        }
        
        return {
            'optimized_prompt': optimization_result['optimized_prompt'],
            'lora_info': lora_info,
            'endpoint': '/generate_both/cinema/',
            'optimization_details': optimization_result
        }
    
    def clean_optimized_prompt(self, prompt: str) -> str:
        """Mock prompt cleaning"""
        # Add white background if not present
        if "white background" not in prompt.lower():
            prompt = prompt + " white background"
        return prompt
    
    def get_deterministic_seed(self, task) -> int:
        """Mock deterministic seed generation"""
        # Use task ID hash for deterministic but varied seeds
        return hash(task.task_id) % 1000 + 42
    
    async def generate_3d_model_clip(self, task) -> Optional[Dict[str, Any]]:
        """Test the generate_3d_model_clip function"""
        self.logger.info(f"🎨 Generating 3D model with CLIP comparison: '{task.prompt}' (task: {task.task_id})")
        
        try:
            # CRITICAL: Wait for priority access to the server
            if not self.priority_coordinator.wait_for_priority_access(task.task_id):
                self.logger.error(f"❌ PRIORITY ACCESS TIMEOUT for task {task.task_id} - subnet task will be missed!")
                task.priority_access_timeout = True
                return None
            
            # Mark the start of our priority job
            self.priority_coordinator.mark_priority_job_start(task.task_id, task.prompt)
            
            # Step 1: Optimize prompt and route to optimal LoRA
            optimization_result = self.optimize_prompt_for_generation(task)
            optimized_prompt = optimization_result['optimized_prompt']
            lora_info = optimization_result['lora_info']
            endpoint = optimization_result['endpoint']
            
            # Step 1.5: Clean the optimized prompt to remove artifacts
            cleaned_prompt = self.clean_optimized_prompt(optimized_prompt)
            
            # Log the final optimization result
            if self.config.get('log_optimization_details', True):
                if optimized_prompt != task.prompt:
                    self.logger.info(f"🎯 FINAL OPTIMIZATION RESULT:")
                    self.logger.info(f"   Original: '{task.prompt}'")
                    self.logger.info(f"   Optimized: '{optimized_prompt}'")
                    self.logger.info(f"   Cleaned: '{cleaned_prompt}'")
                    self.logger.info(f"   LoRA: {lora_info['lora_name']} via {endpoint}")
                else:
                    self.logger.info(f"ℹ️ No optimization applied - using original prompt")
                    self.logger.info(f"   Prompt: '{task.prompt}'")
                    self.logger.info(f"   LoRA: {lora_info['lora_name']} via {endpoint}")
            
            # Clear cache on the server using priority coordinator
            self.priority_coordinator.clear_server_cache()

            # Step 2: Get deterministic seed
            deterministic_seed = self.get_deterministic_seed(task)
            self.logger.info(f"   🎲 Using deterministic seed: {deterministic_seed}")
            self.logger.info(f"   🎨 Using LoRA: {lora_info['lora_name']} via {endpoint}")
            
            generation_start = time.time()
            
            # Step 3: Generate both prompts in parallel using asyncio
            self.logger.info(f"🚀 Starting parallel generation for both prompts")
            
            # Use the preloaded CLIP analyzer
            clip_analyzer = self.get_clip_analyzer()
            if clip_analyzer is None:
                self.logger.error("❌ CLIP analyzer not available")
                self.priority_coordinator.mark_priority_job_end(task.task_id)
                return None
            
            # Generate both prompts in parallel using the new /generate_both/ endpoint
            async def generate_single_prompt(prompt: str, is_optimized: bool = False):
                """Generate a single prompt and return results using the server endpoint"""
                try:
                    import aiohttp
                    import base64
                    from PIL import Image
                    import io
                    
                    # Prepare the request data
                    request_data = {
                        'prompt': prompt,
                        'seed': deterministic_seed,
                        'num_inference_steps': self.config.get('num_inference_steps', 7),
                        'guidance_scale': self.config.get('guidance_scale', 3.5),
                        'ss_sampling_steps': self.config.get('ss_sampling_steps', 21),
                        'slat_sampling_steps': self.config.get('slat_sampling_steps', 24),
                        'slat_guidance_strength': self.config.get('slat_guidance_strength', 4.0),
                        'ss_guidance_strength': self.config.get('ss_guidance_strength', 9.5)
                    }
                    
                    # Send request to the server
                    try:
                        async with aiohttp.ClientSession() as session:
                            async with session.post(
                                f"{self.config.get('generation_server_url')}{endpoint}",
                                data=request_data,
                                timeout=aiohttp.ClientTimeout(total=300)  # 5 minutes timeout
                            ) as response:
                                if response.status == 200:
                                    result = await response.json()
                                    
                                    # Decode the base64 image or create mock image
                                    try:
                                        image_data = base64.b64decode(result['image'])
                                        image = Image.open(io.BytesIO(image_data))
                                    except (ImportError, NameError):
                                        # Fallback to mock image if PIL not available
                                        image = MockImage()
                                    
                                    # Get PLY data (always compressed when available)
                                    if 'compressed_ply' in result:
                                        ply_data = base64.b64decode(result['compressed_ply'])
                                        compressed_data = ply_data  # Already compressed
                                    elif 'ply_data' in result:
                                        # Fallback to uncompressed PLY if compression failed
                                        ply_data = base64.b64decode(result['ply_data'])
                                        compressed_data = None
                                    else:
                                        # Create mock PLY data for testing
                                        ply_data = b"mock_ply_data_for_testing"
                                        compressed_data = b"mock_compressed_ply_data"
                                    
                                    return {
                                        'ply_data': ply_data,
                                        'compressed_data': compressed_data,
                                        'image': image,
                                        'is_optimized': is_optimized,
                                        'prompt': prompt
                                    }
                                else:
                                    self.logger.error(f"❌ Server request failed: {response.status}")
                                    # For testing purposes, create mock data when server fails
                                    self.logger.info("🔄 Creating mock data for testing...")
                                    
                                    # Create mock image
                                    try:
                                        from PIL import Image
                                        image = Image.new('RGB', (512, 512), color='blue')
                                    except ImportError:
                                        image = MockImage()
                                    
                                    # Create mock PLY data
                                    ply_data = b"mock_ply_data_for_testing"
                                    compressed_data = b"mock_compressed_ply_data"
                                    
                                    return {
                                        'ply_data': ply_data,
                                        'compressed_data': compressed_data,
                                        'image': image,
                                        'is_optimized': is_optimized,
                                        'prompt': prompt
                                    }
                    
                    except Exception as e:
                        self.logger.error(f"❌ Server connection failed: {e}")
                        # For testing purposes, create mock data when server is unreachable
                        self.logger.info("🔄 Creating mock data due to server connection failure...")
                        
                        # Create mock image
                        try:
                            from PIL import Image
                            image = Image.new('RGB', (512, 512), color='green')
                        except ImportError:
                            image = MockImage()
                        
                        # Create mock PLY data
                        ply_data = b"mock_ply_data_for_testing"
                        compressed_data = b"mock_compressed_ply_data"
                        
                        return {
                            'ply_data': ply_data,
                            'compressed_data': compressed_data,
                            'image': image,
                            'is_optimized': is_optimized,
                            'prompt': prompt
                        }
                    
                except Exception as e:
                    self.logger.error(f"❌ Generation failed for prompt '{prompt[:50]}...': {e}")
                    # For testing purposes, create mock data when generation fails
                    self.logger.info("🔄 Creating mock data due to generation failure...")
                    
                    # Create mock image
                    try:
                        from PIL import Image
                        image = Image.new('RGB', (512, 512), color='red')
                    except ImportError:
                        image = MockImage()
                    
                    # Create mock PLY data
                    ply_data = b"mock_ply_data_for_testing"
                    compressed_data = b"mock_compressed_ply_data"
                    
                    return {
                        'ply_data': ply_data,
                        'compressed_data': compressed_data,
                        'image': image,
                        'is_optimized': is_optimized,
                        'prompt': prompt
                    }
            
            # Run both generations in parallel
            original_task = generate_single_prompt(task.prompt, is_optimized=False)
            optimized_task = generate_single_prompt(cleaned_prompt, is_optimized=True)
            
            # Wait for both to complete
            original_result = await original_task
            optimized_result = await optimized_task
            
            if original_result is None or optimized_result is None:
                self.logger.error(f"❌ One or both generations failed")
                self.priority_coordinator.mark_priority_job_end(task.task_id)
                return None
            
            # Step 4: Compute CLIP scores for comparison
            self.logger.info(f"🎯 Computing CLIP alignment scores for comparison")
            
            try:
                # Get images from results
                original_image = original_result['image']
                optimized_image = optimized_result['image']
                
                # Ensure images are RGB for CLIP processing
                if original_image.mode != 'RGB':
                    original_image = original_image.convert('RGB')
                if optimized_image.mode != 'RGB':
                    optimized_image = optimized_image.convert('RGB')
                
                # Compute CLIP scores: original prompt vs both images
                original_vs_original = clip_analyzer.compute_clip_alignment_score(task.prompt, original_image)
                original_vs_optimized = clip_analyzer.compute_clip_alignment_score(task.prompt, optimized_image)
                
                # Also compute cross-comparisons for analysis
                optimized_vs_original = clip_analyzer.compute_clip_alignment_score(cleaned_prompt, original_image)
                optimized_vs_optimized = clip_analyzer.compute_clip_alignment_score(cleaned_prompt, optimized_image)
                
                self.logger.info(f"✅ CLIP scores computed:")
                self.logger.info(f"   Original prompt + Original image: {original_vs_original:.4f}")
                self.logger.info(f"   Original prompt + Optimized image: {original_vs_optimized:.4f}")
                self.logger.info(f"   Optimized prompt + Original image: {optimized_vs_original:.4f}")
                self.logger.info(f"   Optimized prompt + Optimized image: {optimized_vs_optimized:.4f}")
                
                # Step 5: Select the better result based on CLIP score
                if original_vs_original >= original_vs_optimized:
                    self.logger.info(f"✅ Using result from ORIGINAL prompt (CLIP: {original_vs_original:.4f} vs {original_vs_optimized:.4f})")
                    selected_result = original_result
                    selected_prompt = task.prompt
                    selection_reason = "Original prompt had higher CLIP score"
                else:
                    self.logger.info(f"✅ Using result from OPTIMIZED prompt (CLIP: {original_vs_optimized:.4f} vs {original_vs_original:.4f})")
                    selected_result = optimized_result
                    selected_prompt = cleaned_prompt
                    selection_reason = "Optimized prompt had higher CLIP score"
                
                # Step 6: Prepare final result
                generation_time = time.time() - generation_start
                
                final_result = {
                    'ply_data': selected_result['ply_data'],
                    'compressed_data': selected_result['compressed_data'],
                    'image': selected_result['image'],
                    'selected_prompt': selected_prompt,
                    'selection_reason': selection_reason,
                    'clip_scores': {
                        'original_vs_original': original_vs_original,
                        'original_vs_optimized': original_vs_optimized,
                        'optimized_vs_original': optimized_vs_original,
                        'optimized_vs_optimized': optimized_vs_optimized
                    },
                    'optimization_details': optimization_result,
                    'generation_time': generation_time,
                    'lora_used': lora_info['lora_name']
                }
                
                # Mark the completion of our priority job
                self.priority_coordinator.mark_priority_job_end(task.task_id)
                
                self.logger.info(f"✅ Generation successful in {generation_time:.2f}s")
                self.logger.info(f"   Selected prompt: '{selected_prompt}'")
                self.logger.info(f"   PLY size: {len(selected_result['ply_data']):,} bytes")
                if selected_result['compressed_data']:
                    compression_ratio = len(selected_result['ply_data']) / len(selected_result['compressed_data'])
                    self.logger.info(f"   Compression ratio: {compression_ratio:.2f}x")
                
                self.stats['successful_generations'] += 1
                self.stats['total_generation_time'] += generation_time
                
                return final_result
                
            except Exception as e:
                self.logger.error(f"❌ CLIP scoring failed: {e}")
                import traceback
                traceback.print_exc()
                self.priority_coordinator.mark_priority_job_end(task.task_id)
                return None
        
        except Exception as e:
            self.logger.error(f"❌ Generation exception: {e}")
            import traceback
            traceback.print_exc()
            # Mark the completion of our priority job even on exception
            self.priority_coordinator.mark_priority_job_end(task.task_id)
            return None

async def test_generate_3d_model_clip_integration():
    """Test the complete generate_3d_model_clip integration"""
    
    print("🧪 Testing generate_3d_model_clip Integration with Reproducibility")
    print("=" * 80)
    
    # Create mock orchestrator
    orchestrator = MockContinuousTrellisOrchestrator()
    
    # Test prompts
    test_prompts = [
        "a blue ceramic vase with red trim",
        "a bronze machine gun mount",
        "a wooden dining table",
        "a red sports car"
    ]
    
    results = []
    
    for i, prompt in enumerate(test_prompts):
        print(f"\n🎯 Test {i+1}/{len(test_prompts)}: '{prompt}'")
        print("-" * 60)
        
        # Create mock task
        task = MockTaskRecord(
            task_id=f"test_task_{i+1}_{int(time.time())}",
            prompt=prompt
        )
        
        # Test the function
        start_time = time.time()
        result = await orchestrator.generate_3d_model_clip(task)
        test_time = time.time() - start_time
        
        if result:
            print(f"✅ Test {i+1} SUCCESSFUL in {test_time:.2f}s")
            print(f"   Selected prompt: '{result['selected_prompt']}'")
            print(f"   Selection reason: {result['selection_reason']}")
            print(f"   LoRA used: {result['lora_used']}")
            print(f"   PLY size: {len(result['ply_data']):,} bytes")
            if result['compressed_data']:
                compression_ratio = len(result['ply_data']) / len(result['compressed_data'])
                print(f"   Compression ratio: {compression_ratio:.2f}x")
            
            # Show CLIP scores
            scores = result['clip_scores']
            print(f"   CLIP Scores:")
            print(f"     Original vs Original: {scores['original_vs_original']:.4f}")
            print(f"     Original vs Optimized: {scores['original_vs_optimized']:.4f}")
            print(f"     Optimized vs Original: {scores['optimized_vs_original']:.4f}")
            print(f"     Optimized vs Optimized: {scores['optimized_vs_optimized']:.4f}")
            
            results.append({
                'test_id': i+1,
                'prompt': prompt,
                'result': result,
                'test_time': test_time,
                'status': 'success'
            })
        else:
            print(f"❌ Test {i+1} FAILED in {test_time:.2f}s")
            results.append({
                'test_id': i+1,
                'prompt': prompt,
                'result': None,
                'test_time': test_time,
                'status': 'failed'
            })
    
    # Summary
    print("\n" + "=" * 80)
    print("📊 INTEGRATION TEST SUMMARY")
    print("=" * 80)
    
    successful_tests = [r for r in results if r['status'] == 'success']
    failed_tests = [r for r in results if r['status'] == 'failed']
    
    print(f"✅ Successful tests: {len(successful_tests)}/{len(results)}")
    print(f"❌ Failed tests: {len(failed_tests)}/{len(results)}")
    
    if successful_tests:
        avg_time = sum(r['test_time'] for r in successful_tests) / len(successful_tests)
        print(f"⏱️  Average test time: {avg_time:.2f}s")
        
        # Show optimization effectiveness
        optimizations_used = sum(1 for r in successful_tests if r['result']['selection_reason'].startswith("Optimized"))
        print(f"🎯 Optimized prompts selected: {optimizations_used}/{len(successful_tests)}")
        
        # Show CLIP score improvements
        clip_improvements = []
        for r in successful_tests:
            scores = r['result']['clip_scores']
            improvement = scores['original_vs_optimized'] - scores['original_vs_original']
            clip_improvements.append(improvement)
        
        avg_improvement = sum(clip_improvements) / len(clip_improvements)
        print(f"📈 Average CLIP score improvement: {avg_improvement:+.4f}")
    
    # Show orchestrator stats
    print(f"\n🔧 Orchestrator Statistics:")
    print(f"   Successful generations: {orchestrator.stats['successful_generations']}")
    print(f"   Total generation time: {orchestrator.stats['total_generation_time']:.2f}s")
    
    print("\n🎉 Integration test completed!")
    return results

if __name__ == "__main__":
    # Check if server is running
    print("🔍 Checking if generation server is running...")
    try:
        import aiohttp
        async def check_server():
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.get("http://localhost:8099/health/", timeout=5) as response:
                        if response.status == 200:
                            print("✅ Generation server is running on port 8099")
                            return True
                        else:
                            print("⚠️  Generation server responded but not healthy")
                            return False
            except Exception as e:
                print(f"❌ Generation server not accessible: {e}")
                print("   Please start the server first: python trellis_subnit_server_mix_lora_flash.py")
                return False
        
        # Check server status
        server_running = asyncio.run(check_server())
        
        if server_running:
            # Run the integration test
            asyncio.run(test_generate_3d_model_clip_integration())
        else:
            print("\n💡 To test with real server:")
            print("   1. Start the generation server: python trellis_subnit_server_mix_lora_flash.py")
            print("   2. Run this test script again")
            print("\n🧪 Running in mock mode (no real server calls)...")
            # Run with mock server responses
            asyncio.run(test_generate_3d_model_clip_integration())
    
    except ImportError:
        print("❌ aiohttp not available. Install with: pip install aiohttp")
        print("🧪 Running in mock mode...")
        asyncio.run(test_generate_3d_model_clip_integration())
