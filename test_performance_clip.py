#!/usr/bin/env python3
"""
CLIP Performance Test: GPU vs CPU
Purpose: Compare speed of cosine similarity computation on GPU vs CPU

Usage:
    python test_performance.py
"""

import time
import torch
import open_clip
from open_clip import CLIP
from open_clip.tokenizer import HFTokenizer
import statistics


class CLIPPerformanceTester:
    """Test CLIP performance on GPU vs CPU"""
    
    def __init__(self):
        self.model_name = "convnext_large_d"
        self.pretrained = "laion2b_s26b_b102k_augreg"
        
        # Test prompts
        self.text1 = "cream lace handkerchief pocket"
        self.text2 = "Create a highly detailed, photorealistic 3D scene depicting an elegant, cream-colored lace handkerchief pocket intricately embroidered with delicate, organic patterns of interlocking florals, leaves, and vines in a soft, muted palette. The lace should exhibit a subtle sheen and texture, with delicate threads and fibers visible to the eye. The handkerchief pocket itself should be situated on an elegant, high-collared 19th-century-style blouse, crafted from a lightweight, sheer fabric with a subtle shimmer and translucency. The surrounding fabric should be adorned with a delicate, beaded trim, adding a touch of whimsy and sophistication to the overall design. The lighting should be soft and diffused, casting a warm, golden glow over the entire scene. The camera should be positioned at a slight angle, capturing the intricate details of the lace and the folds of the fabric. The artistic style should be reminiscent of a Victorian-era front view, white background. Style the scene in a realistic, hyper-realistic manner, with attention to detail and texture, to create a visually stunning composition."
        
        print(f"🔧 CLIP Performance Tester initialized")
        print(f"   Model: {self.model_name}/{self.pretrained}")
        print(f"   Text 1: '{self.text1[:50]}...'")
        print(f"   Text 2: '{self.text2[:50]}...'")
    
    def load_model_on_device(self, device_str):
        """Load CLIP model on specified device and measure loading time"""
        # Create proper torch device object
        device = torch.device(device_str)
        print(f"🔧 Loading CLIP model on {device}...")
        
        start_time = time.time()
        try:
            model, _, _ = open_clip.create_model_and_transforms(
                self.model_name, 
                pretrained=self.pretrained,
                device=device
            )
            tokenizer = open_clip.get_tokenizer(self.model_name)
            end_time = time.time()
            loading_time = (end_time - start_time) * 1000  # Convert to milliseconds
            
            print(f"✅ CLIP model loaded successfully on {device}")
            print(f"   Loading time: {loading_time:.2f} ms")
            return model, tokenizer, loading_time, device
        except Exception as e:
            print(f"❌ Failed to load CLIP model on {device}: {e}")
            return None, None, 0, device
    
    def compute_similarity(self, model, tokenizer, device, num_runs=10):
        """Compute similarity multiple times and measure performance"""
        print(f"📊 Running {num_runs} similarity computations on {device}...")
        
        times = []
        similarities = []
        
        for i in range(num_runs):
            start_time = time.time()
            
            try:
                # Tokenize both texts
                tokenized_text1 = tokenizer(self.text1).to(device)
                tokenized_text2 = tokenizer(self.text2).to(device)
                
                # Use proper autocast for the device
                if device.type == 'cuda':
                    with torch.no_grad(), torch.amp.autocast(device_type='cuda'):
                        # Encode both texts
                        text1_features = model.encode_text(tokenized_text1)
                        text2_features = model.encode_text(tokenized_text2)
                        
                        # Normalize features
                        text1_features /= text1_features.norm(dim=-1, keepdim=True)
                        text2_features /= text2_features.norm(dim=-1, keepdim=True)
                        
                        # Compute cosine similarity
                        similarity = (text1_features @ text2_features.T).cpu().numpy()[0][0]
                        similarity = max(0.0, min(1.0, similarity))
                        
                        similarities.append(similarity)
                else:
                    with torch.no_grad():
                        # Encode both texts
                        text1_features = model.encode_text(tokenized_text1)
                        text2_features = model.encode_text(tokenized_text2)
                        
                        # Normalize features
                        text1_features /= text1_features.norm(dim=-1, keepdim=True)
                        text2_features /= text2_features.norm(dim=-1, keepdim=True)
                        
                        # Compute cosine similarity
                        similarity = (text1_features @ text2_features.T).cpu().numpy()[0][0]
                        similarity = max(0.0, min(1.0, similarity))
                        
                        similarities.append(similarity)
                
                end_time = time.time()
                run_time = (end_time - start_time) * 1000  # Convert to milliseconds
                times.append(run_time)
                
                if (i + 1) % 5 == 0:
                    print(f"   Completed {i + 1}/{num_runs} runs...")
                
            except Exception as e:
                print(f"❌ Error in run {i + 1}: {e}")
                continue
        
        return times, similarities
    
    def run_performance_test(self, num_runs=20):
        """Run performance test on both GPU and CPU"""
        print(f"\n🚀 STARTING PERFORMANCE TEST")
        print(f"=" * 60)
        print(f"Number of runs per device: {num_runs}")
        print(f"Priming runs: 5 (warm-up)")
        print(f"Measurement runs: {num_runs - 5}")
        
        results = {}
        
        # Test GPU if available
        if torch.cuda.is_available():
            print(f"\n🖥️  TESTING GPU (CUDA)")
            print(f"=" * 40)
            
            gpu_model, gpu_tokenizer, gpu_loading_time, gpu_device = self.load_model_on_device('cuda')
            if gpu_model is not None:
                # Prime the GPU with 5 warm-up runs
                print(f"🔥 Priming GPU with 5 warm-up runs...")
                self.compute_similarity(gpu_model, gpu_tokenizer, gpu_device, 5)
                
                # Now measure actual performance
                print(f"📊 Measuring GPU performance with {num_runs - 5} runs...")
                gpu_times, gpu_similarities = self.compute_similarity(gpu_model, gpu_tokenizer, gpu_device, num_runs - 5)
                
                if gpu_times:
                    results['gpu'] = {
                        'times': gpu_times,
                        'similarities': gpu_similarities,
                        'device': gpu_device,
                        'loading_time': gpu_loading_time
                    }
                    
                    # Clean up GPU memory
                    del gpu_model, gpu_tokenizer
                    torch.cuda.empty_cache()
        else:
            print(f"\n❌ GPU (CUDA) not available")
        
        # Test CPU
        print(f"\n💻 TESTING CPU")
        print(f"=" * 40)
        
        cpu_model, cpu_tokenizer, cpu_loading_time, cpu_device = self.load_model_on_device('cpu')
        if cpu_model is not None:
            # Prime the CPU with 5 warm-up runs
            print(f"🔥 Priming CPU with 5 warm-up runs...")
            self.compute_similarity(cpu_model, cpu_tokenizer, cpu_device, 5)
            
            # Now measure actual performance
            print(f"📊 Measuring CPU performance with {num_runs - 5} runs...")
            cpu_times, cpu_similarities = self.compute_similarity(cpu_model, cpu_tokenizer, cpu_device, num_runs - 5)
            
            if cpu_times:
                results['cpu'] = {
                    'times': cpu_times,
                    'similarities': cpu_similarities,
                    'device': cpu_device,
                    'loading_time': cpu_loading_time
                }
                
                # Clean up CPU memory
                del cpu_model, cpu_tokenizer
        
        return results
    
    def analyze_results(self, results):
        """Analyze and display performance results"""
        print(f"\n📊 PERFORMANCE ANALYSIS RESULTS")
        print(f"=" * 60)
        
        if not results:
            print("❌ No results to analyze")
            return
        
        # Display results for each device
        for device_name, data in results.items():
            times = data['times']
            similarities = data['similarities']
            device = data['device']
            loading_time = data['loading_time']
            
            print(f"\n🔍 {device_name.upper()} ({device}) RESULTS:")
            print(f"   Loading time: {loading_time:.2f} ms")
            print(f"   Total runs: {len(times)}")
            print(f"   Average time: {statistics.mean(times):.2f} ms")
            print(f"   Median time: {statistics.median(times):.2f} ms")
            print(f"   Min time: {min(times):.2f} ms")
            print(f"   Max time: {max(times):.2f} ms")
            if len(times) > 1:
                print(f"   Std dev: {statistics.stdev(times):.2f} ms")
            print(f"   Average similarity: {statistics.mean(similarities):.4f}")
        
        # Compare GPU vs CPU if both available
        if 'gpu' in results and 'cpu' in results:
            print(f"\n⚡ PERFORMANCE COMPARISON")
            print(f"=" * 40)
            
            gpu_avg = statistics.mean(results['gpu']['times'])
            cpu_avg = statistics.mean(results['cpu']['times'])
            
            speedup = cpu_avg / gpu_avg if gpu_avg > 0 else 0
            
            print(f"   GPU average: {gpu_avg:.2f} ms")
            print(f"   CPU average: {cpu_avg:.2f} ms")
            print(f"   GPU speedup: {speedup:.2f}x faster than CPU")
            
            if speedup > 1:
                print(f"   🚀 GPU is {speedup:.1f}x faster than CPU")
            else:
                print(f"   🐌 CPU is {1/speedup:.1f}x faster than GPU")
            
            # Similarity consistency
            gpu_sim_avg = statistics.mean(results['gpu']['similarities'])
            cpu_sim_avg = statistics.mean(results['cpu']['similarities'])
            sim_diff = abs(gpu_sim_avg - cpu_sim_avg)
            
            print(f"\n📈 SIMILARITY CONSISTENCY:")
            print(f"   GPU similarity: {gpu_sim_avg:.4f}")
            print(f"   CPU similarity: {cpu_sim_avg:.4f}")
            print(f"   Difference: {sim_diff:.4f}")
            
            if sim_diff < 0.001:
                print(f"   ✅ Excellent consistency between devices")
            elif sim_diff < 0.01:
                print(f"   🟡 Good consistency between devices")
            else:
                print(f"   ⚠️  Some variation between devices")


def main():
    print("🚀 CLIP Performance Test: GPU vs CPU")
    print("=" * 50)
    
    # Check CUDA availability
    if torch.cuda.is_available():
        print(f"✅ CUDA available: {torch.cuda.get_device_name(0)}")
        print(f"   CUDA version: {torch.version.cuda}")
        print(f"   GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    else:
        print("❌ CUDA not available")
    
    print(f"✅ PyTorch version: {torch.__version__}")
    
    # Initialize tester
    tester = CLIPPerformanceTester()
    
    try:
        # Run performance test
        results = tester.run_performance_test(num_runs=20)
        
        # Analyze results
        tester.analyze_results(results)
        
        print(f"\n🎯 PERFORMANCE TEST COMPLETE")
        print(f"=" * 50)
        
    except KeyboardInterrupt:
        print("\n⚠️  Test interrupted by user")
    except Exception as e:
        print(f"❌ Test failed: {e}")


if __name__ == "__main__":
    main()
