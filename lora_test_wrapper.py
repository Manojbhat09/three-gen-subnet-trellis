#!/usr/bin/env python3
"""
LoRA Test Wrapper Script
Purpose: Test different LoRA generations against validation prompts and compare results
"""

import os
import time
import json
import requests
import base64
import tempfile
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
import traceback

# Test prompts
TEST_PROMPTS = [
    "greek amphora scene detail",
    "plastic straw of drink", 
    "small yellow triangular wooden kitchen knife",
    "enormous black robot with round body",
    "rose gold locket necklace with floral"
]

# LoRA configurations for testing
FLUX_LORAS = {
    'isometric_3d': {
        'name': 'Flux Isometric 3D',
        'endpoint': '/generate/isometric_3d/',
        'trigger_prefix': 'Isometric 3D,',
        'description': 'Isometric 3D style LoRA for FLUX'
    },
    'live_3d': {
        'name': 'FLUX Live 3D', 
        'endpoint': '/generate/live_3d/',
        'trigger_prefix': '',
        'description': 'Live 3D style LoRA for FLUX'
    },
    'game_assets': {
        'name': '3D Game Assets',
        'endpoint': '/generate/game_assets/',
        'trigger_prefix': 'Create 3D game asset, isometric view version,',
        'description': '3D game assets style LoRA for FLUX'
    },
    'patched_realism': {
        'name': 'Patched Realism',
        'endpoint': '/generate/patched_realism/',
        'trigger_prefix': '',
        'description': 'Realism enhancement LoRA for FLUX'
    },
    'tf2_style': {
        'name': 'Team Fortress 2 Style',
        'endpoint': '/generate/tf2_style/',
        'trigger_prefix': 'tf2style,',
        'description': 'Team Fortress 2 style LoRA for FLUX'
    },
    'baolei': {
        'name': 'Baolei Style',
        'endpoint': '/generate/baolei/',
        'trigger_prefix': 'Cartoon-style design,',
        'description': 'Baolei cartoon style LoRA for FLUX'
    },
    'cartoon_3d': {
        'name': 'Cartoon 3D Render',
        'endpoint': '/generate/cartoon_3d/',
        'trigger_prefix': '',
        'description': 'Cartoon 3D render style LoRA for FLUX'
    },
    'cinema': {
        'name': 'Cinema Style',
        'endpoint': '/generate/cinema/',
        'trigger_prefix': 'c1n3ma,',
        'description': 'Cinema style LoRA for FLUX'
    }
}

SDXL_LORAS = {
    'game_icon': {
        'name': 'Game Icon Institute',
        'endpoint': '/generate/game_icon/',
        'trigger_prefix': 'game icon institute,',
        'description': 'Game icon style LoRA for SDXL'
    }
}

@dataclass
class TestResult:
    """Container for test results"""
    prompt: str
    lora_name: str
    lora_type: str  # 'flux' or 'sdxl'
    generation_time: float
    validation_score: float
    validation_details: Dict[str, Any]
    success: bool
    error_message: Optional[str] = None
    ply_size_bytes: Optional[int] = None
    compressed_size_bytes: Optional[int] = None

class LoRATestWrapper:
    """Wrapper for testing LoRA generations"""
    
    def __init__(self, flux_server_url: str = "http://127.0.0.1:8096", 
                 sdxl_server_url: str = "http://127.0.0.1:8097",
                 validation_server_url: str = "http://127.0.0.1:10006"):
        self.flux_server_url = flux_server_url
        self.sdxl_server_url = sdxl_server_url
        self.validation_server_url = validation_server_url
        self.results: List[TestResult] = []
        
        # Create output directory
        self.output_dir = Path("./lora_test_results")
        self.output_dir.mkdir(exist_ok=True)
        
    def test_flux_lora(self, lora_key: str, prompt: str, seed: int = 42) -> TestResult:
        """Test a FLUX LoRA generation"""
        lora_config = FLUX_LORAS[lora_key]
        
        print(f"🎨 Testing FLUX LoRA: {lora_config['name']}")
        print(f"   Prompt: '{prompt}'")
        print(f"   Endpoint: {lora_config['endpoint']}")
        
        start_time = time.time()
        
        try:
            # Generate 3D model
            response = requests.post(
                f"{self.flux_server_url}{lora_config['endpoint']}",
                data={
                    'prompt': prompt,
                    'seed': seed,
                    'return_compressed': True
                },
                timeout=300
            )
            
            if response.status_code != 200:
                raise Exception(f"Generation failed with status {response.status_code}: {response.text}")
            
            generation_time = time.time() - start_time
            ply_data = response.content
            
            print(f"   ✅ Generation completed in {generation_time:.2f}s")
            print(f"   📦 PLY size: {len(ply_data):,} bytes")
            
            # Validate the generation
            validation_result = self.validate_generation(prompt, ply_data)
            
            return TestResult(
                prompt=prompt,
                lora_name=lora_config['name'],
                lora_type='flux',
                generation_time=generation_time,
                validation_score=validation_result.get('validation_score', 0.0),
                validation_details=validation_result,
                success=True,
                ply_size_bytes=len(ply_data)
            )
            
        except Exception as e:
            generation_time = time.time() - start_time
            print(f"   ❌ Generation failed: {e}")
            
            return TestResult(
                prompt=prompt,
                lora_name=lora_config['name'],
                lora_type='flux',
                generation_time=generation_time,
                validation_score=0.0,
                validation_details={},
                success=False,
                error_message=str(e)
            )
    
    def test_sdxl_lora(self, lora_key: str, prompt: str, seed: int = 42) -> TestResult:
        """Test a SDXL LoRA generation"""
        lora_config = SDXL_LORAS[lora_key]
        
        print(f"🎨 Testing SDXL LoRA: {lora_config['name']}")
        print(f"   Prompt: '{prompt}'")
        print(f"   Endpoint: {lora_config['endpoint']}")
        
        start_time = time.time()
        
        try:
            # Generate 3D model
            response = requests.post(
                f"{self.sdxl_server_url}{lora_config['endpoint']}",
                data={
                    'prompt': prompt,
                    'seed': seed,
                    'return_compressed': True
                },
                timeout=300
            )
            
            if response.status_code != 200:
                raise Exception(f"Generation failed with status {response.status_code}: {response.text}")
            
            generation_time = time.time() - start_time
            ply_data = response.content
            
            print(f"   ✅ Generation completed in {generation_time:.2f}s")
            print(f"   📦 PLY size: {len(ply_data):,} bytes")
            
            # Validate the generation
            validation_result = self.validate_generation(prompt, ply_data)
            
            return TestResult(
                prompt=prompt,
                lora_name=lora_config['name'],
                lora_type='sdxl',
                generation_time=generation_time,
                validation_score=validation_result.get('validation_score', 0.0),
                validation_details=validation_result,
                success=True,
                ply_size_bytes=len(ply_data)
            )
            
        except Exception as e:
            generation_time = time.time() - start_time
            print(f"   ❌ Generation failed: {e}")
            
            return TestResult(
                prompt=prompt,
                lora_name=lora_config['name'],
                lora_type='sdxl',
                generation_time=generation_time,
                validation_score=0.0,
                validation_details={},
                success=False,
                error_message=str(e)
            )
    
    def validate_generation(self, prompt: str, ply_data: bytes) -> Dict[str, Any]:
        """Validate a generation using the validation server"""
        try:
            print(f"   🔍 Validating generation...")
            
            # Encode PLY data
            encoded_data = base64.b64encode(ply_data).decode('utf-8')
            
            # Prepare validation request
            request_data = {
                "prompt": prompt,
                "data": encoded_data,
                "compression": 0,
                "generate_preview": False,
                "preview_score_threshold": 0.8
            }
            
            # Submit for validation
            response = requests.post(
                f"{self.validation_server_url}/validate_txt_to_3d_ply/",
                json=request_data,
                timeout=120
            )
            
            if response.status_code == 200:
                result = response.json()
                validation_score = result.get("score", 0.0)
                print(f"   ✅ Validation completed! Score: {validation_score:.4f}")
                return result
            else:
                print(f"   ⚠️ Validation failed: {response.status_code}")
                return {"score": 0.0, "error": f"HTTP {response.status_code}"}
                
        except Exception as e:
            print(f"   ⚠️ Validation failed: {e}")
            return {"score": 0.0, "error": str(e)}
    
    def run_all_tests(self, test_flux: bool = True, test_sdxl: bool = True, 
                     specific_loras: Optional[List[str]] = None) -> List[TestResult]:
        """Run all LoRA tests"""
        print("🚀 Starting LoRA Test Suite")
        print("=" * 80)
        
        all_results = []
        
        # Test FLUX LoRAs
        if test_flux:
            print("\n🎨 Testing FLUX LoRAs")
            print("-" * 40)
            
            flux_loras_to_test = specific_loras if specific_loras else list(FLUX_LORAS.keys())
            
            for lora_key in flux_loras_to_test:
                if lora_key not in FLUX_LORAS:
                    print(f"⚠️ LoRA '{lora_key}' not found in FLUX LoRAs, skipping...")
                    continue
                
                for prompt in TEST_PROMPTS:
                    result = self.test_flux_lora(lora_key, prompt)
                    all_results.append(result)
                    print()  # Empty line for readability
        
        # Test SDXL LoRAs
        if test_sdxl:
            print("\n🎨 Testing SDXL LoRAs")
            print("-" * 40)
            
            sdxl_loras_to_test = specific_loras if specific_loras else list(SDXL_LORAS.keys())
            
            for lora_key in sdxl_loras_to_test:
                if lora_key not in SDXL_LORAS:
                    print(f"⚠️ LoRA '{lora_key}' not found in SDXL LoRAs, skipping...")
                    continue
                
                for prompt in TEST_PROMPTS:
                    result = self.test_sdxl_lora(lora_key, prompt)
                    all_results.append(result)
                    print()  # Empty line for readability
        
        self.results = all_results
        return all_results
    
    def generate_report(self) -> Dict[str, Any]:
        """Generate a comprehensive test report"""
        if not self.results:
            return {"error": "No test results available"}
        
        # Calculate statistics
        successful_results = [r for r in self.results if r.success]
        failed_results = [r for r in self.results if not r.success]
        
        # Group by LoRA
        lora_stats = {}
        for result in successful_results:
            lora_name = result.lora_name
            if lora_name not in lora_stats:
                lora_stats[lora_name] = {
                    'total_tests': 0,
                    'successful_tests': 0,
                    'failed_tests': 0,
                    'average_score': 0.0,
                    'best_score': 0.0,
                    'worst_score': float('inf'),
                    'average_generation_time': 0.0,
                    'results': []
                }
            
            lora_stats[lora_name]['total_tests'] += 1
            lora_stats[lora_name]['successful_tests'] += 1
            lora_stats[lora_name]['results'].append(result)
            
            # Update statistics
            current_avg = lora_stats[lora_name]['average_score']
            current_count = lora_stats[lora_name]['successful_tests']
            lora_stats[lora_name]['average_score'] = (current_avg * (current_count - 1) + result.validation_score) / current_count
            
            lora_stats[lora_name]['best_score'] = max(lora_stats[lora_name]['best_score'], result.validation_score)
            lora_stats[lora_name]['worst_score'] = min(lora_stats[lora_name]['worst_score'], result.validation_score)
            
            # Update generation time
            current_time_avg = lora_stats[lora_name]['average_generation_time']
            lora_stats[lora_name]['average_generation_time'] = (current_time_avg * (current_count - 1) + result.generation_time) / current_count
        
        # Add failed tests to stats
        for result in failed_results:
            lora_name = result.lora_name
            if lora_name not in lora_stats:
                lora_stats[lora_name] = {
                    'total_tests': 0,
                    'successful_tests': 0,
                    'failed_tests': 0,
                    'average_score': 0.0,
                    'best_score': 0.0,
                    'worst_score': float('inf'),
                    'average_generation_time': 0.0,
                    'results': []
                }
            
            lora_stats[lora_name]['total_tests'] += 1
            lora_stats[lora_name]['failed_tests'] += 1
            lora_stats[lora_name]['results'].append(result)
        
        # Fix worst_score for LoRAs with no successful tests
        for lora_name, stats in lora_stats.items():
            if stats['successful_tests'] == 0:
                stats['worst_score'] = 0.0
        
        # Overall statistics
        total_tests = len(self.results)
        total_successful = len(successful_results)
        total_failed = len(failed_results)
        
        if successful_results:
            overall_avg_score = sum(r.validation_score for r in successful_results) / len(successful_results)
            overall_best_score = max(r.validation_score for r in successful_results)
            overall_avg_time = sum(r.generation_time for r in successful_results) / len(successful_results)
        else:
            overall_avg_score = 0.0
            overall_best_score = 0.0
            overall_avg_time = 0.0
        
        # Create report
        report = {
            "test_summary": {
                "total_tests": total_tests,
                "successful_tests": total_successful,
                "failed_tests": total_failed,
                "success_rate": (total_successful / total_tests * 100) if total_tests > 0 else 0,
                "overall_average_score": overall_avg_score,
                "overall_best_score": overall_best_score,
                "overall_average_generation_time": overall_avg_time
            },
            "lora_statistics": lora_stats,
            "test_prompts": TEST_PROMPTS,
            "timestamp": time.time(),
            "test_duration": sum(r.generation_time for r in self.results)
        }
        
        return report
    
    def save_report(self, filename: Optional[str] = None) -> str:
        """Save the test report to a JSON file"""
        if filename is None:
            timestamp = int(time.time())
            filename = f"lora_test_report_{timestamp}.json"
        
        filepath = self.output_dir / filename
        
        report = self.generate_report()
        
        with open(filepath, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        print(f"📊 Report saved to: {filepath}")
        return str(filepath)
    
    def print_summary(self):
        """Print a summary of the test results"""
        if not self.results:
            print("No test results available")
            return
        
        report = self.generate_report()
        summary = report["test_summary"]
        
        print("\n" + "=" * 80)
        print("📊 LoRA Test Summary")
        print("=" * 80)
        print(f"Total Tests: {summary['total_tests']}")
        print(f"Successful: {summary['successful_tests']}")
        print(f"Failed: {summary['failed_tests']}")
        print(f"Success Rate: {summary['success_rate']:.1f}%")
        print(f"Overall Average Score: {summary['overall_average_score']:.4f}")
        print(f"Overall Best Score: {summary['overall_best_score']:.4f}")
        print(f"Average Generation Time: {summary['overall_average_generation_time']:.2f}s")
        
        print("\n🎨 LoRA Performance:")
        print("-" * 40)
        
        # Sort LoRAs by average score
        lora_stats = report["lora_statistics"]
        sorted_loras = sorted(
            lora_stats.items(),
            key=lambda x: x[1]['average_score'],
            reverse=True
        )
        
        for lora_name, stats in sorted_loras:
            if stats['successful_tests'] > 0:
                print(f"{lora_name}:")
                print(f"  Average Score: {stats['average_score']:.4f}")
                print(f"  Best Score: {stats['best_score']:.4f}")
                print(f"  Success Rate: {stats['successful_tests']}/{stats['total_tests']} ({stats['successful_tests']/stats['total_tests']*100:.1f}%)")
                print(f"  Average Time: {stats['average_generation_time']:.2f}s")
                print()

def main():
    """Main function to run the LoRA test suite"""
    import argparse
    
    parser = argparse.ArgumentParser(description="LoRA Test Wrapper")
    parser.add_argument("--flux-server", default="http://127.0.0.1:8096", help="FLUX server URL")
    parser.add_argument("--sdxl-server", default="http://127.0.0.1:8097", help="SDXL server URL")
    parser.add_argument("--validation-server", default="http://127.0.0.1:10006", help="Validation server URL")
    parser.add_argument("--test-flux", action="store_true", default=True, help="Test FLUX LoRAs")
    parser.add_argument("--test-sdxl", action="store_true", default=True, help="Test SDXL LoRAs")
    parser.add_argument("--loras", nargs="+", help="Specific LoRAs to test")
    parser.add_argument("--output", help="Output filename for report")
    
    args = parser.parse_args()
    
    # Create test wrapper
    wrapper = LoRATestWrapper(
        flux_server_url=args.flux_server,
        sdxl_server_url=args.sdxl_server,
        validation_server_url=args.validation_server
    )
    
    # Run tests
    results = wrapper.run_all_tests(
        test_flux=args.test_flux,
        test_sdxl=args.test_sdxl,
        specific_loras=args.loras
    )
    
    # Print summary
    wrapper.print_summary()
    
    # Save report
    report_path = wrapper.save_report(args.output)
    
    print(f"\n✅ LoRA test suite completed!")
    print(f"📊 Report saved to: {report_path}")

if __name__ == "__main__":
    main() 