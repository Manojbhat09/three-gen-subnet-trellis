#!/usr/bin/env python3
"""
Continuous TRELLIS Orchestrator (Test, LoRA)
- Mines on testnet (netuid 89)
- Generates via LoRA server endpoint: /generate_image/baolei/
- Uses separate outputs/logging to avoid mixing with production
- VALIDATOR BLACKLISTING to skip problematic validators (e.g., UID 180 WC)
- FULLY ASYNCHRONOUS operations for concurrent task pulling and processing
- Configurable concurrency limits for optimal performance

Run with the test runner:
  ./run_trellis_mining_test.sh --continuous --harvest --submit --start-server
"""

import asyncio
import argparse
import logging
import sys
import time
from typing import Any, Dict, Optional, List
from concurrent.futures import ThreadPoolExecutor

import requests
import base64

# Set CUDA deterministic environment variable before any CUDA operations
import os
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
import math

from continuous_trellis_orchestrator_lora import (
    ContinuousTrellisOrchestrator,
    logger as base_logger,
    PriorityServerCoordinator,
    TaskDatabase,
    TaskRecord,
)

# Optional local production-accurate validator
try:
    from subnet_accurate_validator import validate_with_production_logic as local_prod_validate
except Exception:
    local_prod_validate = None

# Optional rich table for pretty comparison
try:
    from rich.console import Console
    from rich.table import Table
    _rich_available = True
    _console = Console()
except Exception:
    _rich_available = False
    _console = None

# Reconfigure logging globally (root) to test-specific file
for h in list(base_logger.handlers):
    base_logger.removeHandler(h)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('continuous_trellis_test.log'),
        logging.StreamHandler(sys.stdout),
    ],
    force=True,  # ensure we override any prior basicConfig
)
logger = logging.getLogger(__name__)


class LenientPriorityServerCoordinator(PriorityServerCoordinator):
    def check_server_status(self) -> Dict[str, Any]:
        """Lenient check: only ping /health with longer timeout; never block pulls."""
        try:
            health_url = f"{self.server_url}/health/"
            resp = requests.get(health_url, timeout=8)
            if resp.status_code == 200:
                return {"available": True, "status": "healthy"}
            return {"available": True, "status": f"http_{resp.status_code}"}
        except Exception as e:
            # Don't block harvesting on status errors in test mode
            return {"available": True, "status": "unknown", "error": str(e)}

    def wait_for_priority_access(self, task_id: str = None) -> bool:
        # In test mode, grant priority immediately after a quick health check
        _ = self.check_server_status()
        return True


class ContinuousTrellisOrchestratorLoRATest(ContinuousTrellisOrchestrator):
    def _get_default_config(self) -> Dict[str, Any]:
        cfg = super()._get_default_config()
        cfg.update({
            'wallet_name': 'manbeast3b',
            'hotkey_name': 'm3b',
            'netuid': 89,  # testnet mirror of SN17
            'min_validator_stake': 1.0,
            'generation_server_url': 'http://localhost:8096',
            'validation_server_url': 'http://localhost:10006',
            'output_dir': './continuous_trellis_outputs_lora_test',
            # Snappier timeouts to avoid long waits on bad validators
            'submission_timeout': 20,
            'generation_timeout': 240,
            'task_pull_interval': 0,
            'ignore_cooldown': True,
            'dual_validation': False,
            
            # Validator blacklisting
            'validator_blacklist': [180],   # UIDs to blacklist (e.g., 180 is a WC)
            'enable_validator_blacklisting': True,
            
            # Asynchronous operation settings
            'max_concurrent_tasks': 5,      # Maximum number of concurrent task processing
            'max_concurrent_pulls': 10,     # Maximum number of concurrent validator pulls
            'enable_concurrent_processing': True,  # Enable fully asynchronous operations
        })
        return cfg

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        # Use test-specific DB
        try:
            self.db = TaskDatabase(db_path="continuous_trellis_tasks_test.db")
            self.logger.info("🧪 Using test database: continuous_trellis_tasks_test.db")
        except Exception as e:
            self.logger.warning(f"⚠️ Failed to set test DB, using default. Error: {e}")
        # Replace with lenient coordinator for test to avoid blocking on status timeouts
        self.priority_coordinator = LenientPriorityServerCoordinator(
            server_url=self.config.get('generation_server_url', 'http://localhost:8096'),
            max_wait_time_seconds=30,
            status_check_interval=1,
            priority_timeout=15,
        )
        # In-memory rows for rich table comparison
        self._dual_rows: List[Dict[str, Any]] = []

    def refresh_validators(self):
        """Refresh and then restrict active validators to allowed list if provided."""
        super().refresh_validators()
        allowed: List[int] = self.config.get('allowed_validator_uids', []) or []
        if allowed:
            for uid, state in list(self.validators.items()):
                state.is_active = uid in allowed
            active = [v for v in self.validators.values() if v.is_active]
            self.logger.info(f"🧪 Restricting to validators: {sorted(allowed)} (active: {len(active)})")

    def is_validator_available(self, validator) -> bool:
        if self.config.get('ignore_cooldown', False):
            return True
        return super().is_validator_available(validator)

    async def generate_3d_model(self, task) -> Optional[Dict[str, Any]]:
        """Generate 3D model using LoRA server with configurable LoRA and parameters."""
        lora_name = self.config.get('lora_name')
        if lora_name:
            self.logger.info(f"🎨 [LoRA] Generating 3D model via '{lora_name}': '{task.prompt}' (task: {task.task_id})")
        else:
            self.logger.info(f"🎨 Generating 3D model via default endpoint: '{task.prompt}' (task: {task.task_id})")
        try:
            # Ensure server priority access like base class
            if not self.priority_coordinator.wait_for_priority_access(task.task_id):
                self.logger.error(f"❌ PRIORITY ACCESS TIMEOUT for task {task.task_id}")
                task.priority_access_timeout = True
                return None
            self.priority_coordinator.mark_priority_job_start(task.task_id, task.prompt)

            # Optional: clear cache for clean memory
            try:
                self.priority_coordinator.clear_server_cache()
            except Exception:
                pass

            # Deterministic seed
            deterministic_seed = self.get_deterministic_seed(task)
            self.logger.info(f"   🎲 Using deterministic seed: {deterministic_seed}")

            # Map LoRA names to endpoints (matching main orchestrator and server)
            if lora_name:
                lora_endpoints = {
                    'patched_realism': '/generate/',
                    'tf2_style': '/generate/tf2_style/',
                    'cartoon_3d': '/generate/cartoon_3d/',
                    'game_assets': '/generate/game_assets/',
                    'sd15_game_icon': '/generate/sd15_game_icon/',
                    'cinema': '/generate/cinema/',
                    'isometric_3d': '/generate/isometric_3d/',
                    'baolei': '/generate/baolei/',
                    'live_3d': '/generate/live_3d/',
                    'necklace': '/generate/necklace/'
                }
                
                endpoint_path = lora_endpoints.get(lora_name, '/generate/')
                endpoint = self.config['generation_server_url'].rstrip('/') + endpoint_path
                self.logger.info(f"   🔗 LoRA: {lora_name} -> Endpoint: {endpoint}")
            else:
                # Use default endpoint when no LoRA specified
                endpoint = self.config['generation_server_url'].rstrip('/') + '/generate/'
                self.logger.info(f"   🔗 Using default endpoint: {endpoint}")

            # Build request data with all generation parameters
            request_data = {
                'prompt': task.prompt,
                'seed': deterministic_seed,
                'return_compressed': True
            }
            
            # Add optional generation parameters if configured
            if self.config.get('num_inference_steps') is not None:
                request_data['num_inference_steps'] = self.config['num_inference_steps']
            if self.config.get('guidance_scale') is not None:
                request_data['guidance_scale'] = self.config['guidance_scale']
            if self.config.get('ss_sampling_steps') is not None:
                request_data['ss_sampling_steps'] = self.config['ss_sampling_steps']
            if self.config.get('slat_sampling_steps') is not None:
                request_data['slat_sampling_steps'] = self.config['slat_sampling_steps']
            if self.config.get('slat_guidance_strength') is not None:
                request_data['slat_guidance_strength'] = self.config['slat_guidance_strength']
            if self.config.get('ss_guidance_strength') is not None:
                request_data['ss_guidance_strength'] = self.config['ss_guidance_strength']

            self.logger.info(f"   🔧 Generation parameters: {request_data}")

            start = time.time()
            response = requests.post(
                endpoint,
                data=request_data,
                timeout=self.config.get('generation_timeout', 300)
            )
            gen_time = time.time() - start
            task.generation_time = gen_time

            if response.status_code != 200:
                self.logger.error(f"❌ Generation failed: HTTP {response.status_code}")
                if response.text:
                    self.logger.error(f"   Response: {response.text[:200]}")
                self.priority_coordinator.mark_priority_job_end(task.task_id)
                return None

            ply_data = response.content
            compression_ratio = response.headers.get('X-Compression-Ratio', 'unknown')
            self.logger.info(f"✅ Generation successful in {gen_time:.2f}s ({len(ply_data):,} bytes, compression: {compression_ratio})")

            # Save compressed output if configured
            if self.config.get('save_intermediate_results', True):
                import os
                from pathlib import Path
                Path(self.output_dir).mkdir(exist_ok=True)
                ts = int(time.time())
                out_path = Path(self.output_dir) / f"task_{task.task_id}_{ts}.ply.spz"
                try:
                    with open(out_path, 'wb') as f:
                        f.write(ply_data)
                    task.compressed_file_path = str(out_path)
                    self.logger.info(f"   💾 Saved to: {out_path}")
                except Exception as e:
                    self.logger.warning(f"⚠️ Failed to save compressed output: {e}")

            self.priority_coordinator.mark_priority_job_end(task.task_id)
            return {'ply_data': ply_data, 'compression_ratio': compression_ratio}

        except Exception as e:
            self.logger.error(f"❌ Generation exception: {e}")
            try:
                self.priority_coordinator.mark_priority_job_end(task.task_id)
            except Exception:
                pass
            return None

    async def validate_model(self, task, ply_data: bytes) -> Optional[float]:
        """Validate using SPZ-compressed payload (compression=2) sent to validator."""
        if not self.config.get('validate_generations', True):
            return None
        try:
            self.logger.info(f"📊 Validating model (SPZ path): '{task.prompt[:50]}...'")
            validation_start = time.time()

            # Encode SPZ-compressed bytes directly
            encoded_data = base64.b64encode(ply_data).decode('utf-8')
            request_data = {
                "prompt": task.prompt,
                "data": encoded_data,
                "compression": 2,  # SPZ
                "generate_preview": False,
                "preview_score_threshold": 0.8,
            }

            response = requests.post(
                f"{self.config['validation_server_url']}/validate_txt_to_3d_ply/",
                json=request_data,
                timeout=self.config.get('validation_timeout', 120),
            )

            validation_time = time.time() - validation_start
            task.validation_time = validation_time

            if response.status_code == 200:
                result = response.json()
                score = float(result.get("score", 0.0))
                task.local_validation_score = score
                self.logger.info(f"✅ Validation completed in {validation_time:.2f}s (score={score:.4f})")
                return score
            else:
                self.logger.error(f"❌ Validation failed: HTTP {response.status_code}")
                return None

        except Exception as e:
            self.logger.error(f"❌ Validation exception: {e}")
            return None

    async def submit_result(self, task: TaskRecord, generation_result: Dict[str, Any]) -> bool:
        """Submit result and optionally perform dual validation comparison"""
        # Perform standard submission first
        success = await super().submit_result(task, generation_result)
        
        # Dual validation: compare submission feedback with local production validation
        if self.config.get('dual_validation', False) and success:
            spz_bytes = generation_result.get('ply_data')
            
            # Initialize variables for comparison
            remote_score = getattr(task, 'task_fidelity_score', None)
            remote_align = None
            remote_iqa = None
            remote_ssim = None
            remote_lpips = None
            remote_val_s = None
            
            local_score = None
            local_align = None
            local_iqa = None
            local_ssim = None
            local_lpips = None
            local_val_s = None
            
            local_raw_score = None
            local_raw_align = None
            local_raw_iqa = None
            local_raw_ssim = None
            local_raw_lpips = None
            local_raw_val_s = None
            compressed_size = len(spz_bytes) if spz_bytes else None
            raw_size = None

            try:
                self.logger.info("🔬 Running dual validation comparison...")
                self.logger.info(f"   Remote feedback score: {remote_score:.4f}" if remote_score is not None else "   Remote feedback score: N/A")
                
                # Local production-accurate validation on SPZ (compression=2)
                if local_prod_validate is not None and spz_bytes:
                    self.logger.info("🔬 Local validation (SPZ, compression=2)...")
                    t1 = time.time()
                    local_res = local_prod_validate(spz_bytes, task.prompt)
                    local_score = float(local_res.get('validation_engine_score', 0.0))
                    local_align = float(local_res.get('alignment_score', 0.0))
                    local_iqa = float(local_res.get('quality_score', 0.0))
                    local_ssim = float(local_res.get('ssim_score', 0.0))
                    local_lpips = float(local_res.get('lpips_score', 0.0))
                    task.local_validation_score = local_score
                    local_val_s = time.time() - t1
                    self.logger.info(f"   Local (SPZ) scores: score={local_score:.4f}, align={local_align:.4f}, iqa={local_iqa:.4f}, ssim={local_ssim:.4f}, lpips={local_lpips:.4f}")
                
                # Local validation on raw decompressed PLY (compression=0)
                if spz_bytes:
                    try:
                        import pyspz
                        self.logger.info("🔬 Local validation (RAW, compression=0)...")
                        raw_ply = pyspz.decompress(spz_bytes, include_normals=False)
                        raw_size = len(raw_ply)
                        from subnet_accurate_validator import validate_with_production_logic_raw
                        t2 = time.time()
                        raw_res = validate_with_production_logic_raw(raw_ply, task.prompt)
                        local_raw_score = float(raw_res.get('validation_engine_score', 0.0))
                        local_raw_align = float(raw_res.get('alignment_score', 0.0))
                        local_raw_iqa = float(raw_res.get('quality_score', 0.0))
                        local_raw_ssim = float(raw_res.get('ssim_score', 0.0))
                        local_raw_lpips = float(raw_res.get('lpips_score', 0.0))
                        local_raw_val_s = time.time() - t2
                        self.logger.info(f"   Local (RAW) scores: score={local_raw_score:.4f}, align={local_raw_align:.4f}, iqa={local_raw_iqa:.4f}, ssim={local_raw_ssim:.4f}, lpips={local_raw_lpips:.4f}")
                    except Exception as e:
                        self.logger.warning(f"⚠️ RAW validation failed: {e}")
                
                # Summary comparison
                if remote_score is not None and local_score is not None:
                    delta = local_score - remote_score
                    self.logger.info(f"   📊 SPZ vs Remote: Remote={remote_score:.4f}, Local(SPZ)={local_score:.4f}, Δ={delta:+.4f}")
                if local_raw_score is not None and local_score is not None:
                    delta_raw = local_raw_score - local_score
                    self.logger.info(f"   📊 RAW vs SPZ: Local(RAW)={local_raw_score:.4f}, Local(SPZ)={local_score:.4f}, Δ={delta_raw:+.4f}")

            except Exception as e:
                self.logger.error(f"❌ Dual validation failed: {e}")

            # Helper for sigmoid
            def _sigmoid(x: float, slope: float, shift: float) -> float:
                try:
                    return 1.0 / (1.0 + math.exp(-slope * (x - shift)))
                except OverflowError:
                    return 0.0 if (slope * (x - shift)) < 0 else 1.0

            def _estimate_iqa(final_score: Optional[float], align: Optional[float], ssim: Optional[float], lpips: Optional[float]) -> Optional[float]:
                if final_score is None or align is None or ssim is None or lpips is None:
                    return None
                if align < 0.3:
                    return 0.0
                contrib = 0.2 * align + 0.025 * _sigmoid(ssim, 35.0, 0.83) + 0.025 * lpips * _sigmoid(lpips, 30.0, 0.7)
                iqa = (final_score - contrib) / 0.75
                return max(0.0, min(1.0, iqa))

            def _estimate_align(final_score: Optional[float], iqa: Optional[float], ssim: Optional[float], lpips: Optional[float]) -> Optional[float]:
                if final_score is None or iqa is None or ssim is None or lpips is None:
                    return None
                contrib = 0.75 * iqa + 0.025 * _sigmoid(ssim, 35.0, 0.83) + 0.025 * lpips * _sigmoid(lpips, 30.0, 0.7)
                align = (final_score - contrib) / 0.2
                return max(0.0, min(1.0, align))

            # Four estimates
            est_raw_iqa = _estimate_iqa(remote_score, local_raw_align, local_raw_ssim, local_raw_lpips)
            est_raw_align = _estimate_align(remote_score, local_raw_iqa, local_raw_ssim, local_raw_lpips)
            est_spz_iqa = _estimate_iqa(remote_score, local_align, local_ssim, local_lpips)
            est_spz_align = _estimate_align(remote_score, local_iqa, local_ssim, local_lpips)

            # Append and render comparison table (two local rows: SPZ and RAW)
            rows = []
            base = {
                'prompt': task.prompt,
                'gen_s': getattr(task, 'generation_time', None),
            }
            rows.append({
                **base,
                'source': 'Remote',
                'score': remote_score,
                'align': remote_align,
                'iqa': remote_iqa,
                'ssim': remote_ssim,
                'lpips': remote_lpips,
                'val_s': remote_val_s,
            })
            rows.append({
                **base,
                'source': 'Local (SPZ)',
                'score': local_score,
                'align': local_align,
                'iqa': local_iqa,
                'ssim': local_ssim,
                'lpips': local_lpips,
                'val_s': local_val_s,
            })
            # Commented out: Local (RAW) row since scores match SPZ; kept for future use
            # rows.append({
            #     **base,
            #     'source': 'Local (RAW)',
            #     'score': local_raw_score,
            #     'align': local_raw_align,
            #     'iqa': local_raw_iqa,
            #     'ssim': local_raw_ssim,
            #     'lpips': local_raw_lpips,
            #     'val_s': local_raw_val_s,
            # })
            # Commented out: RAW-based estimation rows; kept for future use
            # rows.append({
            #     **base,
            #     'source': 'Est (RAW IQA)',
            #     'score': remote_score,
            #     'align': local_raw_align,
            #     'iqa': est_raw_iqa,
            #     'ssim': local_raw_ssim,
            #     'lpips': local_raw_lpips,
            #     'val_s': None,
            # })
            # rows.append({
            #     **base,
            #     'source': 'Est (RAW Align)',
            #     'score': remote_score,
            #     'align': est_raw_align,
            #     'iqa': local_raw_iqa,
            #     'ssim': local_raw_ssim,
            #     'lpips': local_raw_lpips,
            #     'val_s': None,
            # })
            rows.append({
                **base,
                'source': 'Est (SPZ IQA)',
                'score': remote_score,
                'align': local_align,
                'iqa': est_spz_iqa,
                'ssim': local_ssim,
                'lpips': local_lpips,
                'val_s': None,
            })
            rows.append({
                **base,
                'source': 'Est (SPZ Align)',
                'score': remote_score,
                'align': est_spz_align,
                'iqa': local_iqa,
                'ssim': local_ssim,
                'lpips': local_lpips,
                'val_s': None,
            })

            # Print sizes header
            self._print_size_header(compressed_size, raw_size)
            self._render_dual_table_rows(rows)
        
        return success

    def _print_size_header(self, compressed_size: Optional[int], raw_size: Optional[int]):
        try:
            comp = f"{compressed_size:,} bytes" if compressed_size is not None else "?"
            raw = f"{raw_size:,} bytes" if raw_size is not None else "?"
            msg = f"Compressed: {comp} | Raw: {raw}"
            if _rich_available and _console:
                _console.print(f"[bold]{msg}[/bold]")
            else:
                self.logger.info(msg)
        except Exception:
            pass

    def _render_dual_table_rows(self, rows: List[Dict[str, Any]]):
        if _rich_available and _console:
            table = Table(title="Dual Validation Comparison (Remote vs Local SPZ)")
            table.add_column("Prompt", overflow="fold", max_width=60)
            table.add_column("Source", justify="left")
            table.add_column("Score", justify="right")
            table.add_column("Align", justify="right")
            table.add_column("IQA", justify="right")
            table.add_column("SSIM", justify="right")
            table.add_column("LPIPS", justify="right")
            table.add_column("Gen(s)", justify="right")
            table.add_column("Val(s)", justify="right")
            def f(x):
                return f"{x:.4f}" if isinstance(x, (int, float)) and x is not None else ("" if x is None else str(x))
            def fs(x):
                return f"{x:.2f}" if isinstance(x, (int, float)) and x is not None else ("" if x is None else str(x))
            for r in rows:
                table.add_row(
                    (r['prompt'] or '')[:200],
                    r.get('source', ''),
                    f(r.get('score')),
                    f(r.get('align')),
                    f(r.get('iqa')),
                    f(r.get('ssim')),
                    f(r.get('lpips')),
                    fs(r.get('gen_s')),
                    fs(r.get('val_s')),
                )
            _console.print(table)
        else:
            for r in rows:
                self.logger.info(
                    f"[Dual] '{(r['prompt'] or '')[:80]}' {r.get('source','')}: "
                    f"score={r.get('score')} align={r.get('align')} iqa={r.get('iqa')} "
                    f"ssim={r.get('ssim')} lpips={r.get('lpips')} gen_s={r.get('gen_s')} val_s={r.get('val_s')}"
                )


async def main():
    parser = argparse.ArgumentParser(description="Continuous TRELLIS Orchestrator (LoRA Test)")
    parser.add_argument("--no-harvest", action="store_true", help="Disable task harvesting")
    parser.add_argument("--no-validate", action="store_true", help="Disable validation")
    parser.add_argument("--no-submit", action="store_true", help="Disable result submission")
    parser.add_argument("--dual-validation", action="store_true", help="Submit first, then run local production-accurate validation for comparison")
    parser.add_argument("--generation-server", default="http://localhost:8096", help="TRELLIS generation server URL")
    parser.add_argument("--validation-server", default="http://localhost:10006", help="Validation server URL")
    parser.add_argument("--output-dir", default="./continuous_trellis_outputs_lora_test", help="Output directory (test)")
    parser.add_argument("--min-score", type=float, default=0.3, help="Minimum local validation score")
    parser.add_argument("--seed", type=int, default=42, help="Fixed seed")
    parser.add_argument("--validators", type=str, default="", help="Comma-separated list of validator UIDs to restrict to (e.g., '79' or '79,1')")
    parser.add_argument("--respect-cooldown", action="store_true", help="Respect validator cooldowns and pull intervals (default: ignore for tests)")
    
    # Async operations arguments
    parser.add_argument("--no-async", action="store_true", help="Disable concurrent processing (use sequential mode)")
    parser.add_argument("--max-concurrent-tasks", type=int, default=5, help="Maximum number of concurrent tasks (default: 5)")
    parser.add_argument("--max-concurrent-pulls", type=int, default=10, help="Maximum number of concurrent validator pulls (default: 10)")
    
    # Validator blacklisting arguments  
    parser.add_argument("--blacklist", type=int, nargs="*", default=[180], help="Validator UIDs to blacklist (default: [180])")
    parser.add_argument("--no-blacklist", action="store_true", help="Disable validator blacklisting")
    
    # LoRA and generation parameters
    parser.add_argument("--lora", type=str, default="baolei", help="LoRA to use for generation (default: baolei). Options: default, patched_realism, tf2_style, cartoon_3d, game_assets, sd15_game_icon, cinema, isometric_3d, baolei, live_3d, necklace")
    parser.add_argument("--num-inference-steps", type=int, default=None, help="Number of inference steps for generation")
    parser.add_argument("--guidance-scale", type=float, default=None, help="Guidance scale for generation")
    parser.add_argument("--ss-sampling-steps", type=int, default=None, help="SS sampling steps for TRELLIS generation")
    parser.add_argument("--slat-sampling-steps", type=int, default=None, help="SLAT sampling steps for TRELLIS generation")
    parser.add_argument("--slat-guidance-strength", type=float, default=None, help="SLAT guidance strength for TRELLIS generation")
    parser.add_argument("--ss-guidance-strength", type=float, default=None, help="SS guidance strength for TRELLIS generation")

    args = parser.parse_args()

    config: Dict[str, Any] = {}
    if args.no_harvest:
        config['harvest_tasks'] = False
    if args.no_validate:
        config['validate_generations'] = False
    if args.no_submit:
        config['submit_results'] = False
    if args.dual_validation:
        config['dual_validation'] = True

    config['generation_server_url'] = args.generation_server
    config['validation_server_url'] = args.validation_server
    config['output_dir'] = args.output_dir
    config['min_local_score'] = args.min_score
    config['use_fixed_seed'] = True
    config['fixed_seed_value'] = args.seed
    if args.validators:
        try:
            config['allowed_validator_uids'] = [int(x.strip()) for x in args.validators.split(',') if x.strip()]
        except Exception:
            config['allowed_validator_uids'] = []
    config['ignore_cooldown'] = not args.respect_cooldown
    
    # Async operations configuration
    if args.no_async:
        config['enable_concurrent_processing'] = False
    config['max_concurrent_tasks'] = args.max_concurrent_tasks
    config['max_concurrent_pulls'] = args.max_concurrent_pulls
    
    # Validator blacklisting configuration
    if args.no_blacklist:
        config['enable_validator_blacklisting'] = False
    if args.blacklist is not None:
        config['validator_blacklist'] = args.blacklist
    
    # LoRA and generation parameters configuration
    if args.lora != 'default':
        config['lora_name'] = args.lora
    else:
        config['lora_name'] = None  # Will use default /generate/ endpoint
    if args.num_inference_steps is not None:
        config['num_inference_steps'] = args.num_inference_steps
    if args.guidance_scale is not None:
        config['guidance_scale'] = args.guidance_scale
    if args.ss_sampling_steps is not None:
        config['ss_sampling_steps'] = args.ss_sampling_steps
    if args.slat_sampling_steps is not None:
        config['slat_sampling_steps'] = args.slat_sampling_steps
    if args.slat_guidance_strength is not None:
        config['slat_guidance_strength'] = args.slat_guidance_strength
    if args.ss_guidance_strength is not None:
        config['ss_guidance_strength'] = args.ss_guidance_strength

    orch = ContinuousTrellisOrchestratorLoRATest(config)
    try:
        await orch.continuous_mining_loop()
    except Exception as e:
        logger.error(f"❌ LoRA Test orchestrator failed: {e}")
        raise


if __name__ == "__main__":
    asyncio.run(main())




