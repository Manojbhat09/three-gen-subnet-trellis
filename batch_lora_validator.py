#!/usr/bin/env python3
"""
Batch LoRA Validator

Generates 3D models for multiple prompts across all available LoRAs using the
FLUX+TRELLIS server, then validates each result using the production-accurate
validator logic to obtain scores that match production.

Requirements:
- trellis_subnit_server_mix_lora_flash.py server running (default: http://127.0.0.1:8096)
- subnet_accurate_validator.py available in the same workspace and its deps installed
- GPU with sufficient memory for generation

Usage examples:
  # Use built-in Python list of prompts
  python batch_lora_validator.py --use-python-list

  # Provide repeated flags
  python batch_lora_validator.py \
    --prompt "a blue ceramic vase with red trim" \
    --prompt "a low-poly wooden chair" \
    --prompt "a futuristic sci-fi helmet"

  # Provide a JSON or comma-separated list
  python batch_lora_validator.py \
    --server-url http://127.0.0.1:8096 \
    --timeout 900 \
    --seed 42 \
    --output-json lora_scores.json \
    --output-csv lora_scores.csv \
    --prompts "[\"a fantasy treasure chest\", \"a cartoonish yellow duck\", \"a medieval iron sword\"]"

  python batch_lora_validator.py \
    --prompts "a chair, a table, a lamp"
"""

import argparse
import csv
import json
import sys
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import requests

# Import production-accurate validator utilities
# Note: this module performs environment checks on import.
try:
    from subnet_accurate_validator import validate_with_production_logic
except SystemExit:
    print("subnet_accurate_validator import failed. Ensure its dependencies are installed and available.")
    raise


# Built-in Python list of prompts (used when --use-python-list is passed)
# DEFAULT_PROMPTS: List[str] = [
#     "a blue ceramic vase with red trim",
#     "a low-poly wooden chair",
#     "a futuristic sci-fi helmet",
# ]
# DEFAULT_PROMPTS: List[str] = [
#     "silver robot wearing green scarf",
#     "warm yellow woolen scarf with fringed edges",
#     "ruby bracelet adorned with faceted stones",
# ]

DEFAULT_PROMPTS: List[str] = [
    # "large dark purple pyramid shaped gemstone",
    # "steel long-handled spade"
    # "robot that is orange and has pointed head"
    # "cyan robot with orange arms"
    # "fern plant with delicate fronds green"
    " heavy-duty green crossbow"
]
# LoRA generation endpoints exposed by trellis_subnit_server_mix_lora_flash.py
# These endpoints automatically load/apply the corresponding LoRA and return
# SPZ-compressed PLY data (if compression succeeds).
LORA_ENDPOINTS: List[str] = [
    "isometric_3d",
    "live_3d",
    "game_assets",
    "patched_realism",
    "tf2_style",
    "baolei",
    "cartoon_3d",
    "cinema",
    "sd15_game_icon",
    "necklace",
]


@dataclass
class ValidationResult:
    lora: str
    prompt: str
    seed: int
    compressed: bool
    ply_size_bytes: int
    validation_engine_score: float
    alignment_score: float
    quality_score: float
    ssim_score: float
    lpips_score: float
    demo_fidelity_score: float
    task_fidelity_score: float
    validation_passed: bool
    alignment_threshold_passed: bool
    generation_time_s: Optional[float]
    validation_time_s: Optional[float]
    total_time_s: Optional[float]


def request_generation(
    server_url: str,
    lora_key: str,
    prompt: str,
    seed: int,
    timeout_s: int,
) -> Tuple[bytes, Dict[str, str]]:
    """
    Calls the specific LoRA endpoint to generate a 3D model and returns raw bytes
    plus response headers.
    """
    url = f"{server_url.rstrip('/')}/generate/{lora_key}/"
    data = {"prompt": prompt, "seed": str(seed), "return_compressed": "true"}
    resp = requests.post(url, data=data, timeout=timeout_s)
    resp.raise_for_status()
    return resp.content, resp.headers  # bytes + headers


def ensure_spz_compressed(raw_bytes: bytes, headers: Dict[str, str]) -> Tuple[bytes, bool]:
    """
    Ensure data is SPZ-compressed. If response indicates compression is not SPZ,
    attempt local compression as a fallback.
    Returns (bytes, compressed_flag).
    """
    compression_header = headers.get("X-Compression") or headers.get("x-compression") or ""
    if compression_header.lower() == "spz":
        return raw_bytes, True

    # Fallback: try local compression
    try:
        import pyspz  # type: ignore

        compressed = pyspz.compress(raw_bytes, workers=-1)
        return compressed, True
    except Exception:
        # Return original if compression fails; validator expects SPZ though
        return raw_bytes, False


def validate_bytes_with_production(ply_bytes: bytes, prompt: str) -> Dict[str, any]:
    """
    Use production-accurate validation logic to compute scores.
    """
    return validate_with_production_logic(ply_bytes, prompt)


def _parse_prompts_list(single_string: Optional[str]) -> List[str]:
    """Parse --prompts into a list. Accepts JSON array or comma/pipe-separated string."""
    if not single_string:
        return []
    s = single_string.strip()
    if not s:
        return []
    # Try JSON first
    try:
        obj = json.loads(s)
        if isinstance(obj, list):
            return [str(x).strip() for x in obj if str(x).strip()]
    except Exception:
        pass
    # Fallback: split by comma or pipe
    parts = [p.strip() for p in s.replace("|", ",").split(",")]
    return [p for p in parts if p]


def resolve_prompts(
    use_python_list: bool,
    repeated_prompts: Optional[List[str]],
    list_string: Optional[str],
) -> List[str]:
    """Resolve final prompt list given CLI flags."""
    if use_python_list:
        return list(DEFAULT_PROMPTS)

    merged: List[str] = []
    if repeated_prompts:
        merged.extend([p for p in repeated_prompts if p and p.strip()])
    parsed = _parse_prompts_list(list_string)
    if parsed:
        merged.extend(parsed)

    if not merged:
        # Fallback to defaults when nothing provided
        return list(DEFAULT_PROMPTS)
    return merged


def main():
    parser = argparse.ArgumentParser(description="Batch validate LoRA generations for multiple prompts")
    parser.add_argument("--server-url", default="http://127.0.0.1:8096", help="Generation server URL")
    parser.add_argument("--seed", type=int, default=42, help="Deterministic seed for generations")
    parser.add_argument("--timeout", type=int, default=1200, help="Per-request timeout in seconds")
    parser.add_argument("--output-json", default="lora_scores.json", help="Path to write JSON results")
    parser.add_argument("--output-csv", default="lora_scores.csv", help="Path to write CSV results")

    # Use the hardcoded Python list of prompts
    parser.add_argument(
        "--use-python-list",
        action="store_true",
        help="Use the built-in Python list of prompts defined in this script (overrides other prompt flags).",
    )

    # Repeated flag: --prompt "text" (can be used multiple times)
    parser.add_argument(
        "--prompt",
        action="append",
        dest="prompts",
        help="Prompt to evaluate (can be provided multiple times).",
    )

    # Single flag containing multiple prompts: JSON array or comma-separated
    parser.add_argument(
        "--prompts",
        dest="prompts_list",
        type=str,
        default=None,
        help="Prompts as JSON array or comma/pipe-separated string.",
    )

    args = parser.parse_args()

    prompts = resolve_prompts(
        use_python_list=args.use_python_list,
        repeated_prompts=args.prompts,
        list_string=args.prompts_list,
    )

    print(f"Server: {args.server_url}")
    print(f"LoRAs: {', '.join(LORA_ENDPOINTS)}")
    print(f"Prompts ({len(prompts)}):")
    for i, p in enumerate(prompts, 1):
        print(f"  {i}. {p}")
    print("")

    all_results: List[ValidationResult] = []

    for lora_key in LORA_ENDPOINTS:
        print(f"=== LoRA: {lora_key} ===")
        for prompt in prompts:
            print(f"→ Generating for prompt: '{prompt}' (seed={args.seed})")
            t0 = time.time()
            try:
                raw_bytes, headers = request_generation(
                    server_url=args.server_url,
                    lora_key=lora_key,
                    prompt=prompt,
                    seed=args.seed,
                    timeout_s=args.timeout,
                )
            except Exception as e:
                print(f"  Generation failed: {e}")
                continue

            ply_size = len(raw_bytes)
            spz_bytes, is_spz = ensure_spz_compressed(raw_bytes, headers)
            if not is_spz:
                print("  Warning: Data is not SPZ-compressed; attempting validation anyway (may fail)")

            print(f"  Bytes: {ply_size:,} (compressed={is_spz})")

            # Validate with production logic
            try:
                validation_start = time.time()
                scores = validate_bytes_with_production(spz_bytes, prompt)
                validation_end = time.time()
            except Exception as e:
                print(f"  Validation failed: {e}")
                continue

            gen_time = headers.get("X-Generation-Time")  # not always present
            try:
                gen_time_s = float(gen_time) if gen_time is not None else None
            except Exception:
                gen_time_s = None

            result = ValidationResult(
                lora=lora_key,
                prompt=prompt,
                seed=args.seed,
                compressed=is_spz,
                ply_size_bytes=ply_size,
                validation_engine_score=float(scores.get("validation_engine_score", 0.0)),
                alignment_score=float(scores.get("alignment_score", 0.0)),
                quality_score=float(scores.get("quality_score", 0.0)),
                ssim_score=float(scores.get("ssim_score", 0.0)),
                lpips_score=float(scores.get("lpips_score", 0.0)),
                demo_fidelity_score=float(scores.get("demo_fidelity_score", 0.0)),
                task_fidelity_score=float(scores.get("task_fidelity_score", 0.0)),
                validation_passed=bool(scores.get("validation_passed", False)),
                alignment_threshold_passed=bool(scores.get("alignment_threshold_passed", False)),
                generation_time_s=gen_time_s,
                validation_time_s=(validation_end - validation_start),
                total_time_s=(time.time() - t0),
            )

            all_results.append(result)

            print(
                f"  Scores: score={result.validation_engine_score:.4f}, "
                f"align={result.alignment_score:.4f}, iqa={result.quality_score:.4f}, "
                f"ssim={result.ssim_score:.4f}, lpips={result.lpips_score:.4f}"
            )
        print("")

    if not all_results:
        print("No results collected.")
        sys.exit(1)

    # Print compact summary per LoRA
    print("=== Summary (avg score per LoRA) ===")
    by_lora: Dict[str, List[ValidationResult]] = {}
    for r in all_results:
        by_lora.setdefault(r.lora, []).append(r)

    summary_rows = []
    for lora_key, rows in by_lora.items():
        avg_score = sum(r.validation_engine_score for r in rows) / max(1, len(rows))
        avg_align = sum(r.alignment_score for r in rows) / max(1, len(rows))
        avg_iqa = sum(r.quality_score for r in rows) / max(1, len(rows))
        summary_rows.append((lora_key, avg_score, avg_align, avg_iqa, len(rows)))

    summary_rows.sort(key=lambda x: x[1], reverse=True)
    for (lora_key, avg_score, avg_align, avg_iqa, n) in summary_rows:
        print(f"{lora_key:18} | n={n:2d} | avg_score={avg_score:.4f} | avg_align={avg_align:.4f} | avg_iqa={avg_iqa:.4f}")

    # Write JSON
    with open(args.output_json, "w") as f:
        json.dump(
            [
                {
                    "lora": r.lora,
                    "prompt": r.prompt,
                    "seed": r.seed,
                    "compressed": r.compressed,
                    "ply_size_bytes": r.ply_size_bytes,
                    "validation_engine_score": r.validation_engine_score,
                    "alignment_score": r.alignment_score,
                    "quality_score": r.quality_score,
                    "ssim_score": r.ssim_score,
                    "lpips_score": r.lpips_score,
                    "demo_fidelity_score": r.demo_fidelity_score,
                    "task_fidelity_score": r.task_fidelity_score,
                    "validation_passed": r.validation_passed,
                    "alignment_threshold_passed": r.alignment_threshold_passed,
                    "generation_time_s": r.generation_time_s,
                    "validation_time_s": r.validation_time_s,
                    "total_time_s": r.total_time_s,
                }
                for r in all_results
            ],
            f,
            indent=2,
        )
    print(f"\nJSON results written to {args.output_json}")

    # Write CSV
    with open(args.output_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "lora",
                "prompt",
                "seed",
                "compressed",
                "ply_size_bytes",
                "validation_engine_score",
                "alignment_score",
                "quality_score",
                "ssim_score",
                "lpips_score",
                "demo_fidelity_score",
                "task_fidelity_score",
                "validation_passed",
                "alignment_threshold_passed",
                "generation_time_s",
                "validation_time_s",
                "total_time_s",
            ]
        )
        for r in all_results:
            writer.writerow(
                [
                    r.lora,
                    r.prompt,
                    r.seed,
                    int(r.compressed),
                    r.ply_size_bytes,
                    f"{r.validation_engine_score:.6f}",
                    f"{r.alignment_score:.6f}",
                    f"{r.quality_score:.6f}",
                    f"{r.ssim_score:.6f}",
                    f"{r.lpips_score:.6f}",
                    f"{r.demo_fidelity_score:.6f}",
                    f"{r.task_fidelity_score:.6f}",
                    int(r.validation_passed),
                    int(r.alignment_threshold_passed),
                    f"{r.generation_time_s:.3f}" if r.generation_time_s is not None else "",
                    f"{r.validation_time_s:.3f}" if r.validation_time_s is not None else "",
                    f"{r.total_time_s:.3f}" if r.total_time_s is not None else "",
                ]
            )
    print(f"CSV results written to {args.output_csv}")


if __name__ == "__main__":
    main() 