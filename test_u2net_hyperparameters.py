#!/usr/bin/env python3
"""
Test u2net hyperparameters and their impact on performance and output quality.
"""

import argparse
import csv
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple, Any
from dataclasses import dataclass

from PIL import Image
from rembg import new_session, remove


@dataclass
class TestConfig:
    """Configuration for a single test run."""
    name: str
    alpha_matting: bool = False
    alpha_matting_foreground_threshold: int = 240
    alpha_matting_background_threshold: int = 10
    alpha_matting_erode_size: int = 10
    post_process_mask: bool = False
    only_mask: bool = False
    bgcolor: Tuple[int, int, int, int] = None
    putalpha: bool = False
    force_return_bytes: bool = False


def create_test_configs() -> List[TestConfig]:
    """Create various test configurations to explore hyperparameter space."""
    configs = []
    
    # Baseline configuration
    configs.append(TestConfig(
        name="baseline"
    ))
    
    # Alpha matting variations
    configs.append(TestConfig(
        name="alpha_matting_aggressive",
        alpha_matting=True,
        alpha_matting_foreground_threshold=200,  # Lower threshold = more aggressive
        alpha_matting_background_threshold=50,   # Higher threshold = more aggressive
        alpha_matting_erode_size=5
    ))
    
    configs.append(TestConfig(
        name="alpha_matting_conservative",
        alpha_matting=True,
        alpha_matting_foreground_threshold=250,  # Higher threshold = more conservative
        alpha_matting_background_threshold=5,    # Lower threshold = more conservative
        alpha_matting_erode_size=15
    ))
    
    configs.append(TestConfig(
        name="alpha_matting_balanced",
        alpha_matting=True,
        alpha_matting_foreground_threshold=240,
        alpha_matting_background_threshold=10,
        alpha_matting_erode_size=10
    ))
    
    # Post-processing variations
    configs.append(TestConfig(
        name="post_process_mask",
        post_process_mask=True
    ))
    
    configs.append(TestConfig(
        name="post_process_alpha_matting",
        alpha_matting=True,
        post_process_mask=True
    ))
    
    # Background color variations
    configs.append(TestConfig(
        name="white_background",
        bgcolor=(255, 255, 255, 255)
    ))
    
    configs.append(TestConfig(
        name="black_background",
        bgcolor=(0, 0, 0, 255)
    ))
    
    configs.append(TestConfig(
        name="transparent_background",
        putalpha=True
    ))
    
    # Mask-only variations
    configs.append(TestConfig(
        name="only_mask",
        only_mask=True
    ))
    
    configs.append(TestConfig(
        name="only_mask_post_processed",
        only_mask=True,
        post_process_mask=True
    ))
    
    # Combined effects
    configs.append(TestConfig(
        name="high_quality_white_bg",
        alpha_matting=True,
        alpha_matting_foreground_threshold=235,
        alpha_matting_background_threshold=15,
        alpha_matting_erode_size=8,
        post_process_mask=True,
        bgcolor=(255, 255, 255, 255)
    ))
    
    configs.append(TestConfig(
        name="fast_processing",
        alpha_matting=False,
        post_process_mask=False,
        putalpha=False
    ))
    
    return configs


def run_single_test(
    config: TestConfig,
    image_bytes: bytes,
    output_dir: Path,
    image_stem: str
) -> Dict[str, Any]:
    """Run a single test configuration and return results."""
    print(f"Testing: {config.name}")
    
    start_time = time.time()
    
    try:
        session = new_session("u2net")  # Changed from u2netp to u2net
        
        # Build kwargs from config
        kwargs = {
            "alpha_matting": config.alpha_matting,
            "alpha_matting_foreground_threshold": config.alpha_matting_foreground_threshold,
            "alpha_matting_background_threshold": config.alpha_matting_background_threshold,
            "alpha_matting_erode_size": config.alpha_matting_erode_size,
            "post_process_mask": config.post_process_mask,
            "only_mask": config.only_mask,
            "putalpha": config.putalpha,
            "force_return_bytes": config.force_return_bytes,
        }
        
        if config.bgcolor:
            kwargs["bgcolor"] = config.bgcolor
        
        # Remove None values
        kwargs = {k: v for k, v in kwargs.items() if v is not None}
        
        result = remove(image_bytes, session=session, **kwargs)
        
        # Save result
        if config.only_mask:
            output_filename = f"{image_stem}.u2net.{config.name}.mask.png"
        else:
            output_filename = f"{image_stem}.u2net.{config.name}.png"
        
        output_path = output_dir / output_filename
        
        if isinstance(result, (bytes, bytearray)):
            output_path.write_bytes(result)
        else:
            # Handle PIL Image or numpy array
            from io import BytesIO
            bio = BytesIO()
            if hasattr(result, 'save'):
                result.save(bio, "PNG")
                output_path.write_bytes(bio.getvalue())
            else:
                # Convert numpy array to PIL and save
                Image.fromarray(result).save(output_path, "PNG")
        
        elapsed = time.time() - start_time
        
        print(f"  ✓ Saved: {output_filename} ({elapsed:.3f}s)")
        
        return {
            "config_name": config.name,
            "status": "success",
            "elapsed_seconds": elapsed,
            "output_path": str(output_path),
            "error": "",
            "alpha_matting": config.alpha_matting,
            "post_process_mask": config.post_process_mask,
            "only_mask": config.only_mask,
            "putalpha": config.putalpha,
            "bgcolor": str(config.bgcolor) if config.bgcolor else "None"
        }
        
    except Exception as exc:
        elapsed = time.time() - start_time
        error_msg = f"{type(exc).__name__}: {exc}"
        print(f"  ✗ Failed: {error_msg} ({elapsed:.3f}s)")
        
        return {
            "config_name": config.name,
            "status": "failed",
            "elapsed_seconds": elapsed,
            "output_path": "",
            "error": error_msg,
            "alpha_matting": config.alpha_matting,
            "post_process_mask": config.post_process_mask,
            "only_mask": config.only_mask,
            "putalpha": config.putalpha,
            "bgcolor": str(config.bgcolor) if config.bgcolor else "None"
        }


def main():
    parser = argparse.ArgumentParser(
        description="Test u2net hyperparameters and their impact on performance and quality."
    )
    
    parser.add_argument(
        "image",
        nargs="?",
        default="/home/mbhat/three-gen-subnet-trellis/flux_image_chair.png",
        help="Path to input image (default: %(default)s)"
    )
    
    parser.add_argument(
        "output_dir",
        nargs="?",
        default="/home/mbhat/three-gen-subnet-trellis/test_u2net_hyperparams",
        help="Directory to save outputs (default: %(default)s)"
    )
    
    parser.add_argument(
        "--configs",
        type=str,
        default=None,
        help="Comma-separated list of specific configs to test (default: all)"
    )
    
    parser.add_argument(
        "--summary-csv",
        type=str,
        default=None,
        help="Path to save CSV summary (default: output_dir/summary.csv)"
    )
    
    args = parser.parse_args()
    
    image_path = Path(str(args.image)).expanduser().resolve()
    output_directory = Path(str(args.output_dir)).expanduser().resolve()
    
    if not image_path.exists():
        print(f"Error: Input image not found: {image_path}", file=sys.stderr)
        return 1
    
    # Create output directory
    output_directory.mkdir(parents=True, exist_ok=True)
    
    # Load image
    try:
        image_bytes = image_path.read_bytes()
        image_stem = image_path.stem
    except Exception as exc:
        print(f"Error: Failed to read image: {exc}", file=sys.stderr)
        return 1
    
    # Get test configurations
    all_configs = create_test_configs()
    
    if args.configs:
        selected_names = [name.strip() for name in args.configs.split(",")]
        configs_to_run = [c for c in all_configs if c.name in selected_names]
        if not configs_to_run:
            print(f"Error: No valid configs found in: {args.configs}", file=sys.stderr)
            return 1
    else:
        configs_to_run = all_configs
    
    print(f"Testing u2net with {len(configs_to_run)} configurations")
    print(f"Input: {image_path}")
    print(f"Output: {output_directory}")
    print("=" * 60)
    
    # Run tests
    results = []
    for config in configs_to_run:
        result = run_single_test(config, image_bytes, output_directory, image_stem)
        results.append(result)
        print()
    
    # Generate summary
    successful = [r for r in results if r["status"] == "success"]
    failed = [r for r in results if r["status"] == "failed"]
    
    print("=" * 60)
    print("📊 TEST SUMMARY")
    print("=" * 60)
    print(f"Total Configurations: {len(configs_to_run)}")
    print(f"Successful: {len(successful)}")
    print(f"Failed: {len(failed)}")
    print(f"Success Rate: {len(successful)/len(configs_to_run)*100:.1f}%")
    
    if successful:
        fastest = min(successful, key=lambda x: x["elapsed_seconds"])
        slowest = max(successful, key=lambda x: x["elapsed_seconds"])
        avg_time = sum(r["elapsed_seconds"] for r in successful) / len(successful)
        
        print(f"\n⚡ PERFORMANCE:")
        print(f"Fastest: {fastest['config_name']} ({fastest['elapsed_seconds']:.3f}s)")
        print(f"Slowest: {slowest['config_name']} ({slowest['elapsed_seconds']:.3f}s)")
        print(f"Average: {avg_time:.3f}s")
    
    if failed:
        print(f"\n❌ FAILED CONFIGURATIONS:")
        for result in failed:
            print(f"  • {result['config_name']}: {result['error']}")
    
    # Save CSV summary
    csv_path = args.summary_csv or output_directory / "summary.csv"
    try:
        with open(csv_path, 'w', newline='') as f:
            if results:
                fieldnames = results[0].keys()
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(results)
        print(f"\n📄 CSV Summary saved: {csv_path}")
    except Exception as exc:
        print(f"Warning: Failed to save CSV: {exc}", file=sys.stderr)
    
    # Print configuration details
    print(f"\n🔧 CONFIGURATION DETAILS:")
    print("=" * 60)
    for result in results:
        if result["status"] == "success":
            config_details = []
            if result["alpha_matting"]:
                config_details.append("α-matting")
            if result["post_process_mask"]:
                config_details.append("post-process")
            if result["only_mask"]:
                config_details.append("mask-only")
            if result["putalpha"]:
                config_details.append("put-alpha")
            if result["bgcolor"] != "None":
                config_details.append(f"bg:{result['bgcolor']}")
            
            details_str = ", ".join(config_details) if config_details else "default"
            print(f"  • {result['config_name']}: {details_str}")
    
    return 0 if not failed else 2


if __name__ == "__main__":
    sys.exit(main())



