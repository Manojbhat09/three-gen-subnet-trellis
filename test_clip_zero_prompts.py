import re
import io
import os
import base64
import json
import time
import argparse
import logging
import importlib.util
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Any
from tqdm import tqdm
import asyncio
import requests
import torch
import torch.nn.functional as F
import open_clip
import numpy as np
from torchvision import transforms
from PIL import Image

# Defaults (can be overridden via CLI)
DEFAULT_LOG_PATH = "/home/mbhat/three-gen-subnet-trellis/continuous_trellis_test.log"
DEFAULT_OUTPUT_DIR = "/home/mbhat/three-gen-subnet-trellis/continuous_trellis_outputs_lora_test"

# Validator CLIP configuration (MUST MATCH validator)
MODEL_NAME = "convnext_large_d"
PRETRAINED = "laion2b_s26b_b102k_augreg"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def setup_logger(log_file: Optional[str] = None, verbose: bool = True) -> logging.Logger:
    logger = logging.getLogger("clip_zero_analysis")
    logger.setLevel(logging.DEBUG)
    logger.handlers.clear()

    fmt = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    if verbose:
        sh = logging.StreamHandler()
        sh.setLevel(logging.INFO)
        sh.setFormatter(fmt)
        logger.addHandler(sh)
    if log_file:
        fh = logging.FileHandler(log_file)
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(fmt)
        logger.addHandler(fh)
    return logger


def load_validator_clip(logger: logging.Logger):
    logger.info(f"Loading validator CLIP: {MODEL_NAME}/{PRETRAINED} on {DEVICE}")
    model, _, _ = open_clip.create_model_and_transforms(
        MODEL_NAME, pretrained=PRETRAINED, device=DEVICE
    )
    tokenizer = open_clip.get_tokenizer(MODEL_NAME)
    model.eval()
    mean = torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(1, 3, 1, 1) * 3
    std = torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(1, 3, 1, 1) * 3
    normalize = transforms.Normalize(mean, std)
    logger.info("Validator CLIP loaded")
    return model, tokenizer, normalize


def encode_text(model, tokenizer, text: str):
    tokens = tokenizer(text).to(DEVICE)
    with torch.no_grad(), torch.amp.autocast(DEVICE.type):
        feats = model.encode_text(tokens)
        feats = feats / feats.norm(dim=-1, keepdim=True)
    return feats


def encode_image(model, normalize, img: Image.Image, res: int = 224):
    t = torch.tensor(np.array(img)).float() / 255.0
    if t.ndim == 3:
        t = t.permute(2, 0, 1)
    t = t.unsqueeze(0).to(DEVICE)
    t = F.interpolate(t, size=(res, res), mode="bicubic", align_corners=False)
    t = normalize(t)
    with torch.no_grad(), torch.amp.autocast(DEVICE.type):
        feats = model.encode_image(t)
        feats = feats / feats.norm(dim=-1, keepdim=True)
    return feats


def text_text_clip_sim(model, tokenizer, a: str, b: str) -> float:
    fa = encode_text(model, tokenizer, a)
    fb = encode_text(model, tokenizer, b)
    sim = (fa @ fb.T).float().item()
    return float(np.clip(sim, 0, 1))


def text_image_clip_sim(model, tokenizer, normalize, text: str, img: Image.Image) -> float:
    tf = encode_text(model, tokenizer, text)
    vf = encode_image(model, normalize, img)
    sim = (vf @ tf.T).float().cpu().numpy()[0][0]
    return float(np.clip(sim, 0, 1))


def requests_session(timeout: int) -> requests.Session:
    s = requests.Session()
    s.request = _with_timeout(s.request, timeout)  # type: ignore
    return s


def _with_timeout(func, timeout):
    def wrapper(method, url, **kwargs):
        if "timeout" not in kwargs:
            kwargs["timeout"] = timeout
        return func(method, url, **kwargs)
    return wrapper


def gen_image_with_lora(session: requests.Session, base_url: str, lora: str, prompt: str, seed: int, steps: int, guidance: float,
                        logger: logging.Logger) -> Optional[Image.Image]:
    prompt = prompt + ", front view, accurate, complete, white background"
    endpoint = f"{base_url.rstrip('/')}/generate_image/{lora.strip('/')}/"
    data = {
        "prompt": prompt,
        "seed": seed,
        "num_inference_steps": steps,
        "guidance_scale": guidance,
    }
    try:
        r = session.post(endpoint, data=data)
        if r.status_code != 200:
            logger.warning(f"Generation failed (HTTP {r.status_code}) for '{prompt[:60]}...' @ {endpoint}")
            return None
        js = r.json()
        b64 = js.get("image")
        if not b64:
            logger.warning("No image field in response")
            return None
        img = Image.open(io.BytesIO(base64.b64decode(b64))).convert("RGB")
        return img
    except Exception as e:
        logger.warning(f"Generation exception: {e}")
        return None


def parse_entries(log_text: str, only_zero: bool = True) -> List[Dict[str, Any]]:
    entries: List[Dict[str, Any]] = []
    current: Optional[Dict[str, Any]] = None
    for line in log_text.splitlines():
        m_proc = re.search(r"Processing task ([0-9a-f-]+): '(.+?)'", line)
        if m_proc:
            if current and current.get("zero_fidelity"):
                entries.append(current)
            elif current and not only_zero:
                entries.append(current)
            current = {
                "task_id": m_proc.group(1),
                "original_prompt": m_proc.group(2),
                "optimized_prompt": None,
                # local SPZ
                "local_score": None,
                "local_align": None,
                "local_iqa": None,
                # remote
                "remote_score": None,
                "zero_fidelity": False,
            }
            continue

        if current:
            m_opt = re.search(r"^\d{4}-\d{2}-\d{2} .*? INFO -\s+Optimized:\s(.+)$", line)
            if m_opt and not current.get("optimized_prompt"):
                current["optimized_prompt"] = m_opt.group(1).strip()

            # Local (SPZ) block
            m_loc_spz = re.search(r"Local \(SPZ\) scores: score=([0-9.]+), align=([0-9.]+), iqa=([0-9.]+)", line)
            if m_loc_spz:
                current["local_score"] = float(m_loc_spz.group(1))
                current["local_align"] = float(m_loc_spz.group(2))
                current["local_iqa"] = float(m_loc_spz.group(3))

            # If only RAW present, capture it as local
            m_loc_raw = re.search(r"Local \(RAW\) scores: score=([0-9.]+), align=([0-9.]+), iqa=([0-9.]+)", line)
            if (m_loc_raw and current.get("local_score") is None):
                current["local_score"] = float(m_loc_raw.group(1))
                current["local_align"] = float(m_loc_raw.group(2))
                current["local_iqa"] = float(m_loc_raw.group(3))

            if "Task fidelity: 0.0000" in line:
                current["zero_fidelity"] = True

            m_remote = re.search(r"Remote feedback score: ([0-9.]+)", line)
            if m_remote:
                current["remote_score"] = float(m_remote.group(1))

            if "completed successfully" in line:
                should_add = bool(current.get("zero_fidelity")) or (not only_zero)
                if should_add:
                entries.append(current)
                current = None

    if current and (current.get("zero_fidelity") or not only_zero):
        entries.append(current)
    # Dedup by task_id
    dedup: List[Dict[str, Any]] = []
    seen = set()
    for e in entries:
        if e.get("task_id") in seen:
            continue
        seen.add(e.get("task_id"))
        dedup.append(e)
    return dedup


def load_prompts_from_file(path: str, logger: logging.Logger) -> List[Dict[str, Any]]:
    p = Path(path)
    if not p.exists():
        logger.error(f"Prompt file not found: {path}")
        return []
    spec = importlib.util.spec_from_file_location(p.stem, p.resolve())
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)
    prompts = getattr(module, "EPISODIC_TEST_PROMPTS", None)
    if prompts is None or not isinstance(prompts, list):
        logger.error("EPISODIC_TEST_PROMPTS list not found in prompt file")
        return []

    results: List[Dict[str, Any]] = []
    for item in prompts:
        if isinstance(item, str):
            results.append({"original_prompt": item, "optimized_prompt": None})
        elif isinstance(item, dict):
            orig = item.get("original") or item.get("prompt") or item.get("original_prompt")
            opt = item.get("optimized") or item.get("optimized_prompt")
            if orig:
                results.append({"original_prompt": orig, "optimized_prompt": opt})
    logger.info(f"Loaded {len(results)} prompts from file")
    return results


def sanitize_filename(text: str) -> str:
    text = re.sub(r"[^a-zA-Z0-9._-]+", "_", text)
    return text.strip("_")[:100]


def run_analysis(
    logger: logging.Logger,
    gen_server: str,
    lora: str,
    seed: int,
    steps: int,
    guidance: float,
    timeout: int,
    rows_inputs: List[Dict[str, Any]],
    save_csv: str,
    image_dir: Optional[str] = None,
    retries: int = 1,
    resume: bool = True,
    non_zero_cols: bool = False,
    enable_optimizer: bool = False,
    optimizer_find_lora: bool = False,
    optimizer_max_iters: int = 3,
    optimizer_target: float = 0.8,
    append_suffix: str = "front view, accurate, complete, white background",
    compute_3d_align: bool = False,
    validator_cmd_template: Optional[str] = None,
):
    session = requests_session(timeout)
    model, tokenizer, normalize = load_validator_clip(logger)

    headers = [
        "source",
        "task_id",
        "original_prompt",
        "optimized_prompt",
        "text_text_clip",
        "orig_to_image_clip",
        "local_align_in_log",
        "image_path",
    ]

    if non_zero_cols:
        headers[headers.index("local_align_in_log")] = "local_align"
        headers.extend(["remote_score", "local_score", "local_iqa"])

    if enable_optimizer:
        headers.extend([
            "opt_new_prompt",
            "tt_new",
            "ti_new",
            "new_3d_align",
        ])

    out_rows: List[Dict[str, Any]] = []
    try:
        os.makedirs(os.path.dirname(save_csv), exist_ok=True)
    except Exception as e:
        logger.warning(f"Failed to create output directory: {e}")
    try:
        if image_dir:
            os.makedirs(image_dir, exist_ok=True)
    except Exception as e:
        logger.warning(f"Failed to create image directory: {e}")

    # Load existing keys if resuming
    existing_keys: set[str] = set()
    csv_exists = os.path.exists(save_csv)
    if resume and csv_exists:
        try:
            with open(save_csv, "r", encoding="utf-8") as f:
                # skip header
                header = f.readline()
                for line in f:
                    parts = line.rstrip("\n").split(",")
                    # Expect columns order per headers; task_id at index 1
                    if len(parts) >= 2:
                        existing_keys.add(parts[1])
        except Exception as ex:
            logger.warning(f"Failed to read existing CSV for resume: {ex}")

    # Open CSV for appending or create new and write header
    need_header = not csv_exists or os.path.getsize(save_csv) == 0
    csv_file = open(save_csv, "a", encoding="utf-8")
    try:
        if need_header:
            csv_file.write(",".join(headers) + "\n")
            csv_file.flush()
    except Exception as ex:
        logger.warning(f"Failed to write CSV header: {ex}")

    logger.info(f"Analyzing {len(rows_inputs)} entries using LoRA='{lora}' @ {gen_server}")
    for idx, e in tqdm(enumerate(rows_inputs, start=1)):
        task_id = e.get("task_id", f"manual_{idx}")
        orig = e.get("original_prompt")
        opt = e.get("optimized_prompt")
        local_align = e.get("local_align")
        source = e.get("source", "log")

        if not orig:
            logger.warning(f"Skipping entry without original prompt: {e}")
            continue

        # text-text similarity if optimized available
        txt_txt = None
        if opt:
            try:
                txt_txt = text_text_clip_sim(model, tokenizer, orig, opt)
            except Exception as ex:
                logger.warning(f"text-text CLIP failed for '{orig[:40]}...': {ex}")

        # generate image
        gen_prompt = opt or orig
        if append_suffix:
            gen_prompt = f"{gen_prompt}, {append_suffix}".strip().strip(',')
        img = None
        for attempt in range(1, max(1, retries) + 1):
            img = gen_image_with_lora(session, gen_server, lora, gen_prompt, seed, steps, guidance, logger)
            if img is not None:
                break
            logger.info(f"Retry {attempt}/{retries} for '{gen_prompt[:50]}...'")

        txt_img = None
        image_path = None
        if img is not None:
            try:
                txt_img = text_image_clip_sim(model, tokenizer, normalize, orig, img)
                if image_dir:
                    fname = f"{sanitize_filename(orig[:60])}_{seed}.jpg"
                    image_path = os.path.join(image_dir, fname)
                    img.save(image_path, format="JPEG", quality=95)
            except Exception as ex:
                logger.warning(f"text-image CLIP failed for '{orig[:40]}...': {ex}")

        row = {
            "source": source,
            "task_id": task_id,
            "original_prompt": orig,
            "optimized_prompt": (opt or "N/A"),
            "text_text_clip": (None if txt_txt is None else round(float(txt_txt), 4)),
            "orig_to_image_clip": (None if txt_img is None else round(float(txt_img), 4)),
            "local_align_in_log": (None if local_align is None else round(float(local_align), 4)),
            "image_path": (image_path or ""),
        }

        if non_zero_cols:
            row["remote_score"] = e.get("remote_score")
            row["local_score"] = e.get("local_score")
            row["local_iqa"] = e.get("local_iqa")

        # Optional optimization block
        if enable_optimizer:
            try:
                from prompt_optimization_engine import CLIPAlignmentOptimizer
                async def _do_opt(prompt: str) -> Tuple[str, float]:
                    # Use the same generation server and share logger
                    optimizer = CLIPAlignmentOptimizer(hunyuan_server_url=gen_server, logger_instance=logger)
                    # Keep it fast: don't scan LoRAs unless requested
                    session = await optimizer.optimize_prompt_comprehensive(
                        prompt, seed=seed, find_optimal_lora=optimizer_find_lora
                    )
                    return session.final_prompt, session.final_score

                t0 = time.time()
                new_prompt, new_score = asyncio.run(_do_opt(orig))
                logger.info(f"Optimizer produced new prompt in {int(1000*(time.time()-t0))} ms: '{new_prompt}'")
                row["opt_new_prompt"] = new_prompt

                # Text-Text vs new
                try:
                    row["tt_new"] = round(text_text_clip_sim(model, tokenizer, orig, new_prompt), 4)
                except Exception as ex:
                    logger.warning(f"tt_new failed: {ex}")

                # New image vs original text
                gen2 = new_prompt
                if append_suffix:
                    gen2 = f"{gen2}, {append_suffix}".strip().strip(',')
                img2 = gen_image_with_lora(session, gen_server, lora, gen2, seed, steps, guidance, logger)
                if img2 is not None:
                    try:
                        row["ti_new"] = round(text_image_clip_sim(model, tokenizer, normalize, orig, img2), 4)
                    except Exception as ex:
                        logger.warning(f"ti_new failed: {ex}")

                # New 3D align via /generate/baolei/ + optional validator cmd
                if compute_3d_align and validator_cmd_template:
                    try:
                        # Use external validator command; user supplies a template with {original} and {optimized}
                        # Ensure prompts are shell-escaped within the template which should contain quotes around placeholders
                        escaped_orig = orig.replace('"', '\\"')
                        escaped_opt = new_prompt.replace('"', '\\"')
                        cmd = validator_cmd_template.format(original=escaped_orig, optimized=escaped_opt)
                        logger.info(f"Running validator command: {cmd}")
                        import subprocess
                        res = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=600, cwd="/home/mbhat/three-gen-subnet-trellis")
                        if res.returncode != 0:
                            logger.warning(f"Validator command failed (rc={res.returncode}): {res.stderr.strip()}")
                        # Attempt to read results file regardless of stdout
                        try:
                            with open("/home/mbhat/three-gen-subnet-trellis/subnet_validation_results.json", "r") as jf:
                                jd = json.load(jf)
                                row["new_3d_align"] = jd.get("alignment_score")
                        except Exception as ex_file:
                            # Fallback: try parse JSON from stdout
                            try:
                                jd = json.loads(res.stdout.strip())
                                row["new_3d_align"] = jd.get("alignment_score")
                            except Exception:
                                logger.warning(f"Could not extract alignment score from validator output: {ex_file}")
                                row["new_3d_align"] = None
                    except Exception as ex:
                        logger.warning(f"3D align computation failed: {ex}")
            except Exception as ex:
                logger.warning(f"Optimizer integration failed: {ex}")

        out_rows.append(row)

        # Write incrementally to CSV to avoid data loss on interruptions
        try:
            # Skip writing duplicates if resuming and key is seen
            if not (resume and task_id in existing_keys):
                vals = [str(row.get(h, "")).replace("\n", " ").replace(",", " ") for h in headers]
                csv_file.write(",".join(vals) + "\n")
                csv_file.flush()
                existing_keys.add(task_id)
        except Exception as ex:
            logger.warning(f"Failed to append CSV for task {task_id}: {ex}")

    # Emit table to stdout (tab-separated)
    logger.info("Results (validator CLIP)")
    print("\t".join(headers))
    for r in out_rows:
        print("\t".join(str(r.get(h, "")) for h in headers))

    # CSV already appended incrementally
    try:
        csv_file.close()
    except Exception:
        pass
    logger.info(f"CSV updated: {save_csv}")


def main():
    parser = argparse.ArgumentParser(description="Analyze prompts with validator CLIP and LoRA generation")
    parser.add_argument("--log-path", default=DEFAULT_LOG_PATH, help="Path to continuous log to parse scores from")
    parser.add_argument("--promptfile", help="Python file exporting EPISODIC_TEST_PROMPTS list (overrides log parsing)")
    parser.add_argument("--gen-server", default=os.environ.get("GEN_SERVER_URL", "http://localhost:8096"), help="Generation server base URL")
    parser.add_argument("--lora", default="baolei", help="LoRA endpoint name (e.g., baolei)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--steps", type=int, default=7)
    parser.add_argument("--guidance", type=float, default=4.0)
    parser.add_argument("--timeout", type=int, default=90, help="HTTP timeout per request in seconds")
    parser.add_argument("--retries", type=int, default=1, help="Retries for image generation")
    parser.add_argument("--image-dir", help="Directory to save generated images")
    parser.add_argument("--out-csv", default=os.path.join(DEFAULT_OUTPUT_DIR, "zero_fidelity_clip_analysis.csv"))
    parser.add_argument("--log-file", help="Save detailed logs to this file")
    parser.add_argument("--quiet", action="store_true", help="Reduce console logging")
    parser.add_argument("--non-zero", action="store_true", help="Include remote/local task fidelity and local align/IQA columns")
    parser.add_argument("--full", action="store_true", help="Consider all prompts from the log, not just zero-fidelity")
    parser.add_argument("--append-suffix", default="front view, accurate, complete, white background", help="Suffix to append to generation prompts")
    # Optimizer controls
    parser.add_argument("--enable-optimizer", action="store_true", help="Run CLIPAlignmentOptimizer to produce a new prompt")
    parser.add_argument("--optimizer-find-lora", action="store_true", help="Allow optimizer to search optimal LoRA (slower)")
    parser.add_argument("--optimizer-max-iters", type=int, default=10, help="Max optimizer iterations (if supported)")
    parser.add_argument("--optimizer-target", type=float, default=0.8, help="Target CLIP score for optimizer")
    # 3D alignment measurement via external validator
    parser.add_argument("--compute-3d-align", action="store_true", help="Compute new 3D alignment using validator command")
    parser.add_argument("--validator-cmd-template", help="Shell command template with {original} and {optimized} placeholders")

    args = parser.parse_args()
    logger = setup_logger(args.log_file, verbose=not args.quiet)

    # Build input rows
    inputs: List[Dict[str, Any]] = []
    if args.promptfile:
        # Load from prompt file
        pf_entries = load_prompts_from_file(args.promptfile, logger)
        for e in pf_entries:
            e["task_id"] = "manual"
            e["source"] = "promptfile"
        inputs = pf_entries
        if not inputs:
            logger.error("No prompts loaded from file; exiting")
            return
    else:
        # Parse from log
        try:
            with open(args.log_path, "r", encoding="utf-8") as f:
                log_text = f.read()
        except Exception as ex:
            logger.error(f"Failed to read log file: {ex}")
            return
        entries = parse_entries(log_text, only_zero=(not args.full))
        if not entries:
            logger.info("No entries found in log with given filters")
            return
        for e in entries:
            e["source"] = "log"
        inputs = entries
        logger.info(f"Found {len(inputs)} entries in log (full={args.full})")

    # In full mode, include remote/local columns by default for complete view
    if args.full and not args.non_zero:
        args.non_zero = True
        logger.info("Full mode: including remote/local score columns by default (--non-zero implied)")

    if args.compute_3d_align and args.validator_cmd_template is None:
        args.validator_cmd_template = "bash -c 'source /home/mbhat/miniconda/bin/activate && conda activate trellis_new && python subnet_accurate_validator.py {original} {optimized}'"
        logger.info(f"Using default validator command template: {args.validator_cmd_template}") 

    run_analysis(
        logger=logger,
        gen_server=args.gen_server,
        lora=args.lora,
        seed=args.seed,
        steps=args.steps,
        guidance=args.guidance,
        timeout=args.timeout,
        rows_inputs=inputs,
        save_csv=args.out_csv,
        image_dir=args.image_dir,
        retries=args.retries,
        non_zero_cols=args.non_zero,
        enable_optimizer=args.enable_optimizer,
        optimizer_find_lora=args.optimizer_find_lora,
        optimizer_max_iters=args.optimizer_max_iters,
        optimizer_target=args.optimizer_target,
        append_suffix=args.append_suffix,
        compute_3d_align=args.compute_3d_align,
        validator_cmd_template=args.validator_cmd_template,
    )


if __name__ == "__main__":
    main()
