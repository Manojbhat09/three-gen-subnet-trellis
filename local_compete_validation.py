#!/usr/bin/env python3
# Subnet 17 (404-GEN) - Local Competitive Validation
# Purpose: Run multiple generations with different seeds to find the best model for a prompt

import asyncio
import aiohttp
import argparse
import time
import os
import sys
import json
from pathlib import Path
from typing import List, Dict, Tuple
from dataclasses import dataclass, asdict

# Add current directory to path for asset manager import
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from generation_asset_manager import (
        global_asset_manager, AssetType, GenerationStatus,
        prepare_for_mining_submission
    )
    ASSET_MANAGER_AVAILABLE = True
except ImportError:
    print("⚠️ Asset manager not available - running in basic mode")
    ASSET_MANAGER_AVAILABLE = False

# Configuration
ENHANCED_GENERATION_URL = "http://127.0.0.1:8095/generate/"
GENERATION_SERVER_URL = "http://127.0.0.1:8093/generate/"
VALIDATION_SERVER_URL = "http://127.0.0.1:10006/validate_txt_to_3d_ply/"
COMPETITION_DIR = "competition_results"

@dataclass
class CompetitionResult:
    """Single competition result"""
    seed: int
    generation_time: float
    validation_time: float
    validation_score: float
    ply_size: int
    face_count: int = 0
    compression_ratio: float = 1.0
    generation_id: str = ""
    error: str = ""
    success: bool = True

class CompetitionRunner:
    """Manages competitive validation runs"""
    
    def __init__(self, prompt: str, num_variants: int = 5, use_enhanced: bool = True):
        self.prompt = prompt
        self.num_variants = num_variants
        self.use_enhanced = use_enhanced
        self.results: List[CompetitionResult] = []
        self.session = None

        # Create competition directory
        timestamp = int(time.time())
        safe_prompt = "".join(c for c in prompt if c.isalnum() or c in (' ', '-', '_')).rstrip()
        safe_prompt = safe_prompt.replace(' ', '_')[:50]
        
        self.competition_dir = Path(COMPETITION_DIR) / f"{safe_prompt}_{timestamp}"
        self.competition_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"🏁 Competition initialized: {self.competition_dir}")

    async def test_server_connectivity(self) -> bool:
        """Test if servers are available"""
        servers = [
            ("Enhanced Generation", ENHANCED_GENERATION_URL + "../health/"),
            ("Generation", GENERATION_SERVER_URL.replace("/generate/", "/health/")),
            ("Validation", "http://127.0.0.1:8094/version/")
        ]
        
        all_online = True
        for name, url in servers:
            try:
                async with self.session.get(url, timeout=5) as response:
                    if response.status == 200:
                        print(f"✓ {name} server: Online")
                    else:
                        print(f"❌ {name} server: Error {response.status}")
                        all_online = False
            except Exception as e:
                print(f"❌ {name} server: Offline ({e})")
                all_online = False
        
        return all_online

    async def run_single_generation(self, seed: int) -> CompetitionResult:
        """Run a single generation and validation"""
        print(f"🎲 Running seed {seed}...")
        
        result = CompetitionResult(
            seed=seed,
            generation_time=0.0,
            validation_time=0.0,
            validation_score=0.0,
            ply_size=0
        )
        
        try:
            # Step 1: Generate
            start_time = time.time()
            
            if self.use_enhanced:
                ply_bytes, gen_metadata = await self._run_enhanced_generation(seed)
            else:
                ply_bytes, gen_metadata = await self._run_basic_generation(seed)
            
            result.generation_time = time.time() - start_time
            
            if ply_bytes is None:
                result.error = "Generation failed"
                result.success = False
                return result
            
            result.ply_size = len(ply_bytes)
            result.face_count = gen_metadata.get('face_count', 0)
            result.compression_ratio = gen_metadata.get('compression_ratio', 1.0)
            result.generation_id = gen_metadata.get('generation_id', '')
            
            # Step 2: Validate
            start_time = time.time()
            score = await self._run_validation(ply_bytes)
            result.validation_time = time.time() - start_time
            result.validation_score = score
            
            if score < 0:
                result.error = "Validation failed"
                result.success = False
                return result
            
            # Step 3: Save PLY for successful generations
            if score > 0.5:  # Only save decent results
                await self._save_result_ply(ply_bytes, seed, score)
            
            print(f"  ✓ Seed {seed}: Score {score:.4f}, Time {result.generation_time:.1f}s+{result.validation_time:.1f}s")
            
        except Exception as e:
            result.error = str(e)
            result.success = False
            print(f"  ❌ Seed {seed}: {e}")
        
        return result

    async def _run_enhanced_generation(self, seed: int) -> Tuple[bytes, dict]:
        """Run enhanced generation"""
        form_data = aiohttp.FormData()
        form_data.add_field('prompt', self.prompt)
        form_data.add_field('seed', str(seed))
        form_data.add_field('use_bpt', 'false')
        form_data.add_field('return_compressed', 'true')
        
        async with self.session.post(ENHANCED_GENERATION_URL, data=form_data, timeout=300) as response:
            if response.status == 200:
                ply_bytes = await response.read()
                metadata = {
                    'generation_id': response.headers.get('X-Generation-ID', ''),
                    'compression_ratio': float(response.headers.get('X-Compression-Ratio', '1.0')),
                    'face_count': int(response.headers.get('X-Face-Count', '0'))
                }
                return ply_bytes, metadata
            else:
                return None, {}

    async def _run_basic_generation(self, seed: int) -> Tuple[bytes, dict]:
        """Run basic generation"""
        payload = {"prompt": self.prompt, "seed": seed}
        async with self.session.post(GENERATION_SERVER_URL, data=payload, timeout=300) as response:
            if response.status == 200:
                ply_bytes = await response.read()
                return ply_bytes, {}
            else:
                return None, {}

    async def _run_validation(self, ply_bytes: bytes) -> float:
        """Run validation"""
        try:
            import pyspz
            import base64
            
            # Try compression
            try:
                compressed_data = pyspz.compress(ply_bytes, 1, 1)
                compression_type = 2
            except:
                compressed_data = ply_bytes
                compression_type = 0
            
            base64_data = base64.b64encode(compressed_data).decode('utf-8')
            
            payload = {
                "prompt": self.prompt,
                "data": base64_data,
                "compression": compression_type,
                "data_ver": 0
            }
            
            async with self.session.post(VALIDATION_SERVER_URL, json=payload, timeout=60) as response:
                if response.status == 200:
                    result = await response.json()
                    return result.get("score", 0.0)
                else:
                    return -1.0
        except:
            return -1.0

    async def _save_result_ply(self, ply_bytes: bytes, seed: int, score: float):
        """Save PLY file for a result"""
        filename = f"seed_{seed}_score_{score:.4f}.ply"
        filepath = self.competition_dir / filename
        
        with open(filepath, 'wb') as f:
            f.write(ply_bytes)

    async def run_competition(self) -> Dict:
        """Run the full competition"""
        print(f"🏆 Starting competition for: '{self.prompt}'")
        print(f"📊 Running {self.num_variants} variants")
        print(f"🎯 Server mode: {'Enhanced' if self.use_enhanced else 'Basic'}")
        print()
        
        async with aiohttp.ClientSession() as session:
            self.session = session
            
            # Test connectivity
            if not await self.test_server_connectivity():
                print("❌ Server connectivity issues detected!")
                return {"error": "Server connectivity failed"}
            
            print()
            
            # Generate seeds (use time-based with spacing for variety)
            base_seed = int(time.time()) % 10000
            seeds = [base_seed + i * 1000 for i in range(self.num_variants)]
            
            # Run all generations
            tasks = [self.run_single_generation(seed) for seed in seeds]
            self.results = await asyncio.gather(*tasks)
            
            # Analyze results
            return await self._analyze_results()

    async def _analyze_results(self) -> Dict:
        """Analyze competition results"""
        print(f"\n📈 Competition Analysis")
        print("=" * 60)
        
        # Filter successful results
        successful_results = [r for r in self.results if r.success and r.validation_score > 0]
        
        if not successful_results:
            print("❌ No successful generations!")
            return {"error": "No successful generations"}
        
        # Find best result
        best_result = max(successful_results, key=lambda x: x.validation_score)
        
        # Calculate statistics
        scores = [r.validation_score for r in successful_results]
        times = [r.generation_time + r.validation_time for r in successful_results]
        
        avg_score = sum(scores) / len(scores)
        avg_time = sum(times) / len(times)
        
        # Print detailed results
        print(f"Results: {len(successful_results)}/{len(self.results)} successful")
        print(f"Best Score: {best_result.validation_score:.4f} (seed {best_result.seed})")
        print(f"Average Score: {avg_score:.4f}")
        print(f"Average Time: {avg_time:.2f}s")
        print()
        
        # Print all results sorted by score
        print("🏅 Leaderboard:")
        sorted_results = sorted(successful_results, key=lambda x: x.validation_score, reverse=True)
        
        for i, result in enumerate(sorted_results[:10]):  # Top 10
            medal = ["🥇", "🥈", "🥉"][i] if i < 3 else f"{i+1:2d}."
            print(f"  {medal} Seed {result.seed:5d}: {result.validation_score:.4f} "
                  f"({result.generation_time:.1f}s+{result.validation_time:.1f}s) "
                  f"[{result.ply_size:,} bytes, {result.face_count:,} faces]")
        
        # Save results
        await self._save_competition_results(best_result, successful_results, avg_score, avg_time)
        
        # Test mining integration with best result
        if ASSET_MANAGER_AVAILABLE and best_result.generation_id:
            await self._test_mining_integration(best_result)
        
        return {
            "best_result": asdict(best_result),
            "successful_count": len(successful_results),
            "total_count": len(self.results),
            "average_score": avg_score,
            "average_time": avg_time,
            "competition_dir": str(self.competition_dir)
        }

    async def _save_competition_results(self, best_result: CompetitionResult, 
                                      all_results: List[CompetitionResult], 
                                      avg_score: float, avg_time: float):
        """Save competition summary"""
        summary = {
            "prompt": self.prompt,
            "num_variants": self.num_variants,
            "server_mode": "enhanced" if self.use_enhanced else "basic",
            "timestamp": time.time(),
            "best_result": asdict(best_result),
            "average_score": avg_score,
            "average_time": avg_time,
            "successful_count": len(all_results),
            "total_count": len(self.results),
            "all_results": [asdict(r) for r in self.results]
        }
        
        # Save summary
        summary_path = self.competition_dir / "competition_summary.json"
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        # Save best PLY as separate file
        if best_result.generation_id and self.use_enhanced:
            await self._copy_best_result_assets(best_result)
        
        print(f"💾 Results saved to: {self.competition_dir}")

    async def _copy_best_result_assets(self, best_result: CompetitionResult):
        """Copy additional assets for the best result"""
        if not best_result.generation_id:
            return
        
        try:
            # Download additional assets for the best result
            asset_types = ['original_image', 'background_removed_image', 'initial_mesh_glb']
            
            for asset_type in asset_types:
                url = f"http://127.0.0.1:8095/generate/{best_result.generation_id}/download/{asset_type}"
                try:
                    async with self.session.get(url, timeout=30) as response:
                        if response.status == 200:
                            data = await response.read()
                            
                            # Determine extension
                            if 'image' in asset_type:
                                ext = '.png'
                            elif 'glb' in asset_type:
                                ext = '.glb'
                            else:
                                ext = '.bin'
                            
                            filename = f"best_{asset_type}{ext}"
                            filepath = self.competition_dir / filename
                            
                            with open(filepath, 'wb') as f:
                                f.write(data)
                            
                            print(f"💎 Saved best {asset_type}: {len(data)} bytes")
                except Exception as e:
                    print(f"⚠️ Could not download {asset_type}: {e}")
                    
        except Exception as e:
            print(f"⚠️ Error copying best result assets: {e}")

    async def _test_mining_integration(self, best_result: CompetitionResult):
        """Test mining integration with the best result"""
        if not best_result.generation_id:
            return
        
        print(f"\n⛏️ Testing mining integration with best result...")
        
        try:
            asset = global_asset_manager.get_asset(best_result.generation_id)
            if asset:
                submission_data = prepare_for_mining_submission(
                    asset,
                    task_id="competition_best",
                    validator_hotkey="competition_validator",
                    validator_uid=1000
                )
                
                print(f"✓ Mining submission prepared for best result:")
                print(f"  - Score: {submission_data.get('local_validation_score', 'N/A')}")
                print(f"  - Compressed PLY: {len(submission_data.get('compressed_ply_b64', ''))} chars")
                print(f"  - Face count: {submission_data.get('face_count', 'N/A')}")
            else:
                print(f"⚠️ Asset {best_result.generation_id} not found in asset manager")
                
        except Exception as e:
            print(f"❌ Mining integration test failed: {e}")


async def main():
    parser = argparse.ArgumentParser(description="Competitive validation for Subnet 17")
    parser.add_argument("prompt", help="Text prompt for 3D generation")
    parser.add_argument("-n", "--num-variants", type=int, default=5, 
                       help="Number of variants to generate (default: 5)")
    parser.add_argument("--server", choices=['enhanced', 'basic'], default='enhanced',
                       help="Server type to use (default: enhanced)")
    parser.add_argument("--max-variants", type=int, default=20,
                       help="Maximum number of variants allowed (default: 20)")
    
    args = parser.parse_args()

    # Validate arguments
    if args.num_variants > args.max_variants:
        print(f"❌ Too many variants requested. Maximum: {args.max_variants}")
        return 1
    
    if args.num_variants < 2:
        print(f"❌ Minimum 2 variants required for competition")
        return 1
    
    print("🚀 Subnet 17 Competitive Validation")
    print("=" * 60)
    print(f"Prompt: {args.prompt}")
    print(f"Variants: {args.num_variants}")
    print(f"Server: {args.server}")
    print(f"Asset Manager: {'Available' if ASSET_MANAGER_AVAILABLE else 'Not Available'}")
    print()
    
    # Run competition
    runner = CompetitionRunner(
        prompt=args.prompt,
        num_variants=args.num_variants,
        use_enhanced=(args.server == 'enhanced')
    )
    
    start_time = time.time()
    results = await runner.run_competition()
    total_time = time.time() - start_time
    
    if "error" in results:
        print(f"❌ Competition failed: {results['error']}")
        return 1
    
    print(f"\n🎯 Competition completed in {total_time:.2f}s")
    print(f"🏆 Best score: {results['best_result']['validation_score']:.4f}")
    print(f"📁 Results directory: {results['competition_dir']}")
    
    return 0


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code) 