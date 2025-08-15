import argparse
import csv
import json
import sys
import time
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

from PIL import Image

from rembg import new_session, remove
from rembg.sessions import sessions_names

try:
    # Optional utility for pre-downloading models
    from rembg.bg import download_models as rembg_download_models  # type: ignore
except Exception:  # noqa: BLE001
    rembg_download_models = None  # type: ignore


def ensure_output_directory_exists(output_directory: Path) -> None:
    output_directory.mkdir(parents=True, exist_ok=True)


def build_sam_prompt(image_size: Tuple[int, int]) -> Dict[str, List[Dict[str, object]]]:
    width, height = image_size
    center_x = max(0, width // 2)
    center_y = max(0, height // 2)
    return {
        "sam_prompt": [
            {"type": "point", "data": [center_x, center_y], "label": 1},
        ]
    }


def build_sam_prompt_from_points(points: List[Tuple[int, int, int]]) -> Dict[str, List[Dict[str, object]]]:
    return {
        "sam_prompt": [
            {"type": "point", "data": [x, y], "label": int(label)} for x, y, label in points
        ]
    }


def parse_points(values: Iterable[str]) -> List[Tuple[int, int, int]]:
    points: List[Tuple[int, int, int]] = []
    for v in values:
        try:
            x_str, y_str, label_str = v.split(",")
            points.append((int(x_str), int(y_str), int(label_str)))
        except Exception as exc:  # noqa: BLE001
            raise argparse.ArgumentTypeError(
                f"Invalid point '{v}'. Expected 'x,y,label'"
            ) from exc
    return points


def parse_models_arg(models_arg: Optional[str]) -> Optional[List[str]]:
    if not models_arg:
        return None
    # Accept comma or space-separated
    raw = [m.strip() for chunk in models_arg.split(",") for m in chunk.split()]  # type: ignore
    return [m for m in raw if m]


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Run rembg background removal across multiple models and save outputs."
        )
    )

    parser.add_argument(
        "image",
        nargs="?",
        default="/home/mbhat/three-gen-subnet-trellis/flux_image_chair.png",
        help="Path to input image (default: %(default)s)",
    )
    parser.add_argument(
        "output_dir",
        nargs="?",
        default="/home/mbhat/three-gen-subnet-trellis/test_rembg_images",
        help="Directory to save outputs (default: %(default)s)",
    )
    parser.add_argument(
        "--models",
        type=str,
        default=None,
        help="Comma or space-separated list of models to run. Defaults to all available.",
    )
    parser.add_argument(
        "--exclude-models",
        type=str,
        default=None,
        help="Comma or space-separated list of models to skip.",
    )
    parser.add_argument(
        "--retries",
        type=int,
        default=1,
        help="Number of retries per model on failure (default: %(default)s)",
    )
    parser.add_argument(
        "--retry-sleep",
        type=float,
        default=1.0,
        help="Seconds to sleep between retries (default: %(default)s)",
    )
    parser.add_argument(
        "--predownload",
        action="store_true",
        help="Pre-download selected models before running.",
    )
    parser.add_argument(
        "--alpha-matting",
        action="store_true",
        help="Enable alpha matting cutout for all models.",
    )
    parser.add_argument(
        "--alpha-fg-th",
        type=int,
        default=240,
        help="Alpha matting foreground threshold (default: %(default)s)",
    )
    parser.add_argument(
        "--alpha-bg-th",
        type=int,
        default=10,
        help="Alpha matting background threshold (default: %(default)s)",
    )
    parser.add_argument(
        "--alpha-erode",
        type=int,
        default=10,
        help="Alpha matting erode size (default: %(default)s)",
    )
    parser.add_argument(
        "--post-process-mask",
        action="store_true",
        help="Apply mask post-processing before cutout.",
    )
    parser.add_argument(
        "--bgcolor",
        type=str,
        default=None,
        help="RGBA background color as JSON array, e.g. '[255,255,255,255]'.",
    )
    parser.add_argument(
        "--sam-center",
        action="store_true",
        help="Use a centered positive point for SAM (default behavior if no --sam-point provided).",
    )
    parser.add_argument(
        "--sam-point",
        action="append",
        default=None,
        help="Provide a SAM point as 'x,y,label'. Repeatable.",
    )
    parser.add_argument(
        "--summary-csv",
        type=str,
        default=None,
        help="Optional path to write a CSV summary of results.",
    )

    args = parser.parse_args()

    image_path = Path(str(args.image)).expanduser().resolve()
    output_directory = Path(str(args.output_dir)).expanduser().resolve()

    if not image_path.exists():
        print(f"Input image not found: {image_path}", file=sys.stderr)
        return 1

    ensure_output_directory_exists(output_directory)

    # Read once to bytes for remove(), and open once to get dimensions (for SAM prompt)
    try:
        image_bytes = image_path.read_bytes()
    except Exception as exc:  # noqa: BLE001
        print(f"Failed to read image bytes: {exc}", file=sys.stderr)
        return 1

    with Image.open(str(image_path)) as pil_image:
        image_width, image_height = pil_image.size

    available_models = sorted(list(sessions_names))

    selected_models = parse_models_arg(args.models)
    excluded_models = set(parse_models_arg(args.exclude_models) or [])

    if selected_models is None:
        models_to_run = [m for m in available_models if m not in excluded_models]
    else:
        models_to_run = [m for m in selected_models if m in available_models and m not in excluded_models]

    print(
        f"Discovered models ({len(available_models)}): {', '.join(available_models)}"
    )
    print(f"Models to run ({len(models_to_run)}): {', '.join(models_to_run)}")

    # Optional pre-download
    if args.predownload and models_to_run:
        if rembg_download_models is not None:
            try:
                print("Pre-downloading selected models...")
                rembg_download_models(tuple(models_to_run))
            except Exception as exc:  # noqa: BLE001
                print(f"Warning: pre-download failed: {exc}", file=sys.stderr)
        else:
            print("Warning: pre-download function not available in this rembg version.")

    # SAM prompt config
    sam_kwargs: Dict[str, object] = {}
    user_points: List[Tuple[int, int, int]] = []
    if args.sam_point:
        user_points = parse_points(args.sam_point)
    if user_points:
        sam_kwargs.update(build_sam_prompt_from_points(user_points))
    else:
        # Default to center point unless explicitly disabled by providing empty list
        if args.sam_center or args.sam_point is None:
            sam_kwargs.update(build_sam_prompt((image_width, image_height)))

    # Optional background color
    bgcolor_value: Optional[Tuple[int, int, int, int]] = None
    if args.bgcolor:
        try:
            parsed = json.loads(args.bgcolor)
            if (
                isinstance(parsed, list)
                and len(parsed) == 4
                and all(isinstance(v, int) for v in parsed)
            ):
                bgcolor_value = tuple(parsed)  # type: ignore[assignment]
            else:
                raise ValueError("bgcolor must be a JSON array of 4 integers")
        except Exception as exc:  # noqa: BLE001
            print(f"Invalid --bgcolor value: {exc}", file=sys.stderr)
            return 1

    # Results tracking
    successes: List[str] = []
    failures: List[Tuple[str, str]] = []
    records: List[Dict[str, object]] = []

    for idx, model_name in enumerate(models_to_run, start=1):
        print(f"\n[{idx}/{len(models_to_run)}] Running model: {model_name}")
        attempt = 0
        model_start_time = time.time()
        output_path: Optional[Path] = None
        last_error: Optional[str] = None

        while attempt <= max(0, args.retries):
            attempt += 1
            try:
                session = new_session(model_name)

                extra_kwargs: Dict[str, object] = {
                    "alpha_matting": bool(args.alpha_matting),
                    "alpha_matting_foreground_threshold": int(args.alpha_fg_th),
                    "alpha_matting_background_threshold": int(args.alpha_bg_th),
                    "alpha_matting_erode_size": int(args.alpha_erode),
                    "post_process_mask": bool(args.post_process_mask),
                }

                if bgcolor_value is not None:
                    extra_kwargs["bgcolor"] = bgcolor_value

                if model_name == "sam":
                    extra_kwargs.update(sam_kwargs)

                result_bytes = remove(image_bytes, session=session, **extra_kwargs)

                output_filename = f"{image_path.stem}.{model_name}.png"
                output_path = output_directory / output_filename
                if isinstance(result_bytes, (bytes, bytearray)):
                    output_path.write_bytes(result_bytes)
                else:
                    # Fallback: ensure we serialize to bytes if unexpected type
                    from io import BytesIO  # local import to avoid overhead

                    bio = BytesIO()
                    if hasattr(result_bytes, "save"):
                        result_bytes.save(bio, "PNG")  # type: ignore[attr-defined]
                        output_path.write_bytes(bio.getvalue())
                    else:
                        raise TypeError(
                            f"Unexpected result type: {type(result_bytes)}"
                        )

                elapsed = time.time() - model_start_time
                print(f"Saved: {output_path} ({elapsed:.2f}s)")
                successes.append(model_name)
                records.append(
                    {
                        "model": model_name,
                        "status": "ok",
                        "elapsed_seconds": round(elapsed, 3),
                        "output_path": str(output_path),
                        "error": "",
                    }
                )
                break
            except Exception as exc:  # noqa: BLE001
                last_error = f"{type(exc).__name__}: {exc}"
                if attempt <= max(0, args.retries):
                    print(
                        f"Attempt {attempt} failed for {model_name}: {last_error}. Retrying...",
                        file=sys.stderr,
                    )
                    time.sleep(max(0.0, float(args.retry_sleep)))
                else:
                    elapsed = time.time() - model_start_time
                    print(
                        f"Failed: {model_name} after {elapsed:.2f}s -> {last_error}",
                        file=sys.stderr,
                    )
                    failures.append((model_name, last_error))
                    records.append(
                        {
                            "model": model_name,
                            "status": "fail",
                            "elapsed_seconds": round(elapsed, 3),
                            "output_path": str(output_path) if output_path else "",
                            "error": last_error,
                        }
                    )
                    break

    # Optional CSV summary
    if args.summary_csv:
        csv_path = Path(str(args.summary_csv)).expanduser().resolve()
        try:
            with csv_path.open("w", newline="") as f:
                writer = csv.DictWriter(
                    f,
                    fieldnames=[
                        "model",
                        "status",
                        "elapsed_seconds",
                        "output_path",
                        "error",
                    ],
                )
                writer.writeheader()
                writer.writerows(records)
            print(f"Wrote summary CSV: {csv_path}")
        except Exception as exc:  # noqa: BLE001
            print(f"Warning: failed to write CSV summary: {exc}", file=sys.stderr)

    print("\n--- Summary ---")
    print(f"Succeeded: {len(successes)} -> {', '.join(successes) if successes else '-'}")
    if failures:
        print(f"Failed: {len(failures)}")
        for failed_model, msg in failures:
            print(f"  - {failed_model}: {msg}")
    else:
        print("Failed: 0")

    return 0 if not failures else 2


if __name__ == "__main__":
    sys.exit(main())


