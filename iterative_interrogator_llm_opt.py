#!/usr/bin/env python3
"""
Iterative Interrogator + LLM Optimizer

Flow:
0) Baseline: generate image from original prompt (no suffix), measure:
   - text–image CLIP: original prompt vs baseline image
   - (no optimized yet) text–text CLIP N/A

Loop for N iterations:
1) Interrogate last image (CLIP Interrogator, fast mode)
2) Query local LLM (Ollama) to rewrite a new optimized prompt that aligns to the original,
   using history of attempts and scores for self-improvement
3) Generate image from the new optimized prompt
4) Measure scores:
   - tt_clip: original prompt vs new optimized prompt (text–text CLIP)
   - ti_clip: original prompt vs new image (text–image CLIP)
   Persist one CSV row per iteration; save images; JSONL optional

Convergence:
- Stop early if raw CLIP (ti_clip) reaches target or no improvement for K rounds
- Exploration/exploitation: a portion of iterations ask LLM to be exploratory

Uses validator CLIP (convnext_large_d/laion2b_s26b_b102k_augreg) for all scoring.
"""

import os
import base64
import io
import re
import json
import time
import math
import random
import argparse
import logging
from typing import List, Dict, Any, Optional, Tuple

import requests
import torch
import torch.nn.functional as F
import open_clip
import numpy as np
from PIL import Image
from torchvision import transforms

# Reuse the interrogator implementation
from prompt_optimization_engine import ImageInterrogatorInterface


MODEL_NAME = "convnext_large_d"
PRETRAINED = "laion2b_s26b_b102k_augreg"


def setup_logger(log_file: Optional[str], verbose: bool = True) -> logging.Logger:
    logger = logging.getLogger("iterative_interrogator_llm")
    logger.setLevel(logging.DEBUG)
    logger.handlers.clear()
    logger.propagate = False
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


def load_validator_clip(device: torch.device, logger: logging.Logger):
    logger.info(f"Loading validator CLIP: {MODEL_NAME}/{PRETRAINED} on {device}")
    model, _, _ = open_clip.create_model_and_transforms(MODEL_NAME, pretrained=PRETRAINED, device=device)
    tokenizer = open_clip.get_tokenizer(MODEL_NAME)
    model.eval()
    mean = torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(1, 3, 1, 1) * 3
    std = torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(1, 3, 1, 1) * 3
    normalize = transforms.Normalize(mean, std)
    logger.info("Validator CLIP loaded")
    return model, tokenizer, normalize


def encode_text(model, tokenizer, device: torch.device, text: str) -> torch.Tensor:
    tokens = tokenizer(text).to(device)
    with torch.no_grad(), torch.amp.autocast(device.type):
        feats = model.encode_text(tokens)
        feats = feats / feats.norm(dim=-1, keepdim=True)
    return feats


def encode_image(model, normalize, device: torch.device, img: Image.Image, res: int = 224) -> torch.Tensor:
    t = torch.tensor(np.array(img)).float() / 255.0
    if t.ndim == 3:
        t = t.permute(2, 0, 1)
    t = t.unsqueeze(0).to(device)
    t = F.interpolate(t, size=(res, res), mode="bicubic", align_corners=False)
    t = normalize(t)
    with torch.no_grad(), torch.amp.autocast(device.type):
        feats = model.encode_image(t)
        feats = feats / feats.norm(dim=-1, keepdim=True)
    return feats


def clip_text_text(model, tokenizer, device, a: str, b: str) -> float:
    fa = encode_text(model, tokenizer, device, a)
    fb = encode_text(model, tokenizer, device, b)
    sim = (fa @ fb.T).float().item()
    return float(np.clip(sim, 0, 1))


def clip_text_image(model, tokenizer, normalize, device, text: str, img: Image.Image) -> float:
    tf = encode_text(model, tokenizer, device, text)
    vf = encode_image(model, normalize, device, img)
    sim = (vf @ tf.T).float().cpu().numpy()[0][0]
    return float(np.clip(sim, 0, 1))


def gen_image(gen_server: str, lora: str, prompt: str, seed: int, steps: int, guidance: float, timeout: int, logger: logging.Logger) -> Optional[Image.Image]:
    endpoint = f"{gen_server.rstrip('/')}/generate_image/{lora.strip('/')}/"
    data = {
        "prompt": prompt,
        "seed": seed,
        "num_inference_steps": steps,
        "guidance_scale": guidance,
    }
    try:
        r = requests.post(endpoint, data=data, timeout=timeout)
        if r.status_code != 200:
            logger.warning(f"Generation failed (HTTP {r.status_code}) @ {endpoint}")
            return None
        js = r.json()
        b64 = js.get("image")
        if not b64:
            logger.warning("No image in response")
            return None
        return Image.open(io.BytesIO(base64.b64decode(b64))).convert("RGB")
    except Exception as e:
        logger.warning(f"Generation exception: {e}")
        return None


def query_ollama_for_prompt(ollama_url: str, model: str, system_prompt: str, user_prompt: str, timeout: int, logger: logging.Logger) -> Optional[str]:
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "stream": False,
        "options": {"temperature": 0.4, "top_p": 0.9, "num_predict": 256},
    }
    try:
        r = requests.post(f"{ollama_url.rstrip('/')}/api/chat", json=payload, timeout=timeout)
        if not r.ok:
            logger.warning(f"Ollama call failed: HTTP {r.status_code}")
            return None
        content = r.json().get("message", {}).get("content", "").strip()
        # Remove code blocks and artifacts
        content = re.sub(r"```[\s\S]*?```", "", content).strip()
        # Pick the first non-empty line
        candidate = None
        for line in content.splitlines():
            txt = line.strip()
            if txt:
                candidate = txt
                break
        candidate = candidate or content
        # Clean common prefixes like "prompt:" and surrounding quotes
        candidate = re.sub(r"^prompt\s*:\s*", "", candidate, flags=re.IGNORECASE).strip()
        if candidate.startswith("'") and candidate.endswith("'") and len(candidate) > 2:
            candidate = candidate[1:-1]
        if candidate.startswith('"') and candidate.endswith('"') and len(candidate) > 2:
            candidate = candidate[1:-1]
        # Remove markdown emphasis like **text** and stray asterisks
        candidate = re.sub(r"\*\*(.*?)\*\*", r"\1", candidate).strip().strip('*').strip()
        return candidate
    except Exception as e:
        logger.warning(f"Ollama exception: {e}")
        return None


def prompt_similarity(prompt1: str, prompt2: str) -> float:
    """Compute similarity between two prompts using simple word overlap."""
    words1 = set(prompt1.lower().split())
    words2 = set(prompt2.lower().split())
    if not words1 or not words2:
        return 0.0
    intersection = words1.intersection(words2)
    union = words1.union(words2)
    return len(intersection) / len(union) if union else 0.0


def build_llm_system_prompt() -> str:
    return (
        "You improve image-generation prompts for a 3D-friendly pipeline. Keep the object's semantics faithful to the original prompt. "
        "Favor clear product render guidance (front view, centered, white background) without adding new unrelated nouns. "
        "Use crisp, concrete terms."
    )


def build_llm_user_prompt(original: str,
                          interrogated: str,
                          history: List[Dict[str, Any]],
                          best_ti: float,
                          target: float,
                          mode: str,
                          plateau_note: Optional[str] = None,
                          best_prompt: Optional[str] = None,
                          best_iter: Optional[int] = None) -> str:
    parts = []
    parts.append(f"ORIGINAL PROMPT:\n{original}")
    parts.append(f"INTERROGATED CAPTION:\n{interrogated}")
    parts.append(f"CURRENT BEST ti_clip (raw): {best_ti:.4f} | TARGET: {target:.4f}")
    if best_prompt and best_iter:
        parts.append(f"BEST PERFORMING PROMPT (iter {best_iter}): '{best_prompt[:200]}'")
        parts.append(f"BEST SCORE: {best_ti:.4f}")
    parts.append(f"MODE: {mode}")
    if history:
        parts.append("HISTORY (most recent first):")
        for h in history[-8:][::-1]:
            parts.append(
                f"- iter {h['iter']}: tt_clip={h['tt']:.4f}, ti_clip={h['ti']:.4f}, prompt='{h['prompt'][:160]}'"
            )
        parts.append("FEEDBACK TASK: Identify what increased ti_clip in prior attempts and what reduced it. Avoid adding unrelated nouns. Increase ti_clip while staying faithful to the original semantics.")
    if plateau_note:
        parts.append(plateau_note)
    parts.append(
        "Output only the new prompt (single line, no commentary)."
    )
    return "\n\n".join(parts)


def save_image(img: Image.Image, path: str):
    try:
        img.save(path, format="JPEG", quality=95)
    except Exception:
        pass


def main():
    parser = argparse.ArgumentParser(description="Iterative Interrogator + LLM prompt optimizer loop")
    parser.add_argument("prompt", type=str, help="Original prompt")
    parser.add_argument("--gen-server", default=os.environ.get("GEN_SERVER_URL", "http://localhost:8096"))
    parser.add_argument("--lora", default="baolei")
    parser.add_argument("--steps", type=int, default=7)
    parser.add_argument("--guidance", type=float, default=4.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--target-clip-raw", type=float, default=0.28)
    parser.add_argument("--min-abs-improve", type=float, default=0.003)
    parser.add_argument("--patience", type=int, default=3, help="Stop if no improvement for this many rounds")
    parser.add_argument("--epsilon", type=float, default=0.3, help="Exploration probability for LLM")
    parser.add_argument("--force-first-explore", type=int, default=2, help="Force exploration for the first K iterations")
    parser.add_argument("--repeat-epsilon", type=float, default=1e-4, help="Plateau detection tolerance for repeating ti_clip values")
    parser.add_argument("--repeat-window", type=int, default=3, help="Rounds to consider for repetition/plateau detection")
    parser.add_argument("--prompt-similarity-threshold", type=float, default=0.8, help="Threshold for detecting too similar prompts (0.0-1.0)")
    parser.add_argument("--drift-threshold", type=float, default=0.3, help="Threshold for semantic drift detection (0.0-1.0, higher = more drift allowed)")
    parser.add_argument("--append-suffix", default="", help="Suffix to append to optimized prompts (e.g., 'front view, accurate, complete, white background')")
    parser.add_argument("--append-to-baseline", action="store_true", help="Also append suffix to baseline generation")
    # Optional final 3D comparison using validator
    parser.add_argument("--final-3d-compare", action="store_true", help="Run subnet validator for original vs best optimized prompts")
    parser.add_argument("--generate-endpoint", default="generate/baolei/", help="3D generation endpoint for validator (e.g., generate/baolei/ or generate/)")
    parser.add_argument("--validator-cmd-template", help="Shell template with {original} {optimized} {endpoint}; runs in repo root")
    parser.add_argument("--image-dir", default="iter_images")
    parser.add_argument("--out-csv", default="iter_optimization.csv")
    parser.add_argument("--out-jsonl", default="iter_optimization.jsonl")
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument("--ollama-url", default="http://localhost:11434")
    parser.add_argument("--ollama-model", default="llama3.2:3b")
    parser.add_argument("--log-file")
    parser.add_argument("--quiet", action="store_true")

    args = parser.parse_args()
    logger = setup_logger(args.log_file, verbose=not args.quiet)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, tokenizer, normalize = load_validator_clip(device, logger)

    os.makedirs(args.image_dir, exist_ok=True)
    # Prepare CSV
    need_header = not os.path.exists(args.out_csv) or os.path.getsize(args.out_csv) == 0
    csv_f = open(args.out_csv, "a", encoding="utf-8")
    if need_header:
        csv_f.write(
            ",".join([
                "iter", "orig_prompt", "optimized_prompt", "tt_clip", "ti_clip", "image_path"
            ]) + "\n"
        )
        csv_f.flush()
    jsonl_f = open(args.out_jsonl, "a", encoding="utf-8")

    # Baseline
    logger.info("Generating baseline image...")
    base_prompt = args.prompt
    if args.append_suffix and args.append_to_baseline:
        suf = args.append_suffix.strip().strip(',')
        if suf and suf.lower() not in base_prompt.lower():
            base_prompt = f"{base_prompt}, {suf}".strip().strip(',')
    base_img = gen_image(args.gen_server, args.lora, base_prompt, args.seed, args.steps, args.guidance, args.timeout, logger)
    if base_img is None:
        logger.error("Baseline generation failed; aborting")
        return
    base_img_path = os.path.join(args.image_dir, f"iter_0.jpg")
    save_image(base_img, base_img_path)
    base_ti = clip_text_image(model, tokenizer, normalize, device, args.prompt, base_img)
    logger.info(f"Baseline ti_clip={base_ti:.4f}")
    csv_f.write(
        ",".join([
            "0", args.prompt.replace(",", " "), "N/A", "", f"{base_ti:.4f}", base_img_path
        ]) + "\n"
    )
    csv_f.flush()
    jsonl_f.write(json.dumps({
        "iter": 0,
        "orig_prompt": args.prompt,
        "optimized_prompt": None,
        "tt_clip": None,
        "ti_clip": base_ti,
        "image_path": base_img_path
    }) + "\n")
    jsonl_f.flush()

    # Iterative loop
    interrogator = ImageInterrogatorInterface(clip_model_name=f"{MODEL_NAME}/{PRETRAINED}", caption_model_name="blip-large")
    # Interrogate baseline once
    baseline_caption = interrogator.interrogate_image(base_img, style_focus="clip_optimized") or ""
    logger.info(f"Baseline interrogation: '{baseline_caption[:180]}'")
    history: List[Dict[str, Any]] = []
    best_ti = base_ti
    best_img = base_img
    last_improve_round = 0
    force_explore_next = False  # Flag to force exploration on next iteration
    force_modify_best = False # Flag to force modification of best prompt

    for itr in range(1, args.iterations + 1):
        logger.info(f"\n▶ Iteration {itr}/{args.iterations}")
        # Plateau detection on recent ti values (from previous iterations)
        plateau_note: Optional[str] = None
        recent = [h["ti"] for h in history[-args.repeat_window:]] if history else []
        repeating = False
        
        if len(recent) >= args.repeat_window:
            max_ti = max(recent)
            min_ti = min(recent)
            repeating = (max_ti - min_ti) <= args.repeat_epsilon
            if repeating:
                plateau_note = (
                    f"PLATEAU: ti_clip has repeated within ±{args.repeat_epsilon:.4f} over the last {args.repeat_window} rounds. "
                    "Propose a meaning-preserving but structurally different phrasing: tighten nouns, remove ambiguous adjectives, "
                    "prefer concrete object descriptors (material, count, view), and avoid photography jargon."
                )
                logger.info(
                    f"   Plateau detected (ti range {min_ti:.4f}..{max_ti:.4f} ≤ eps={args.repeat_epsilon:.4f}); forcing explore"
                )

        # Interrogate
        caption = interrogator.interrogate_image(best_img, style_focus="clip_optimized")
        caption = caption or ""
        logger.info(f"   Interrogated: '{caption[:180]}'")

        # Build LLM prompt (explore vs exploit)
        sys_prompt = build_llm_system_prompt()
        # Exploration schedule: force first K rounds, then epsilon-greedy
        explore = True if itr <= args.force_first_explore else (random.random() < args.epsilon)
        if repeating or force_explore_next:
            explore = True
        hist_for_llm = history.copy()
        mode = "explore" if explore else "exploit"
        if explore:
            if repeating or force_explore_next:
                sys_prompt += " CRITICAL: You are stuck in a repetitive pattern. Make a SUBSTANTIAL change to the prompt structure. Consider: different object descriptions, alternative materials, new spatial arrangements, or completely different rendering styles. Avoid minor adjective changes."
            else:
                sys_prompt += " Try exactly one bold variation that might improve alignment."
        # Reset the flag after using it
        force_explore_next = False

        # Find best prompt so far
        best_prompt_info = None
        best_iter_info = None
        if history:
            best_entry = max(history, key=lambda h: h.get("ti", 0.0))
            best_prompt_info = best_entry.get("prompt")
            best_iter_info = best_entry.get("iter")

        user_prompt = build_llm_user_prompt(
            args.prompt, caption, hist_for_llm, best_ti, args.target_clip_raw, mode, 
            plateau_note=plateau_note, best_prompt=best_prompt_info, best_iter=best_iter_info
        )

        # If drift was detected in previous iteration, instruct the LLM to modify the best prompt
        if force_modify_best and best_prompt_info:
            sys_prompt += f" CRITICAL: High semantic drift detected in previous iteration. Modify the best performing prompt '{best_prompt_info[:100]}...' to reduce drift while maintaining performance. Focus on making it more faithful to the original semantics."

        optimized = query_ollama_for_prompt(args.ollama_url, args.ollama_model, sys_prompt, user_prompt, args.timeout, logger)
        if not optimized:
            optimized = f"{args.prompt}, front view, centered, white background"
        logger.info(f"   LLM optimized prompt: '{optimized}'")

        # Check semantic drift from original prompt AFTER generating the prompt
        drift_detected = False
        if history:
            # Compute drift between original and current optimized prompt
            drift_score = 1.0 - clip_text_text(model, tokenizer, device, args.prompt, optimized)
            if drift_score > args.drift_threshold:
                drift_detected = True
                logger.info(f"   High semantic drift detected ({drift_score:.3f} > {args.drift_threshold:.3f}); will modify best prompt next iteration")
                force_explore_next = True
                # Store drift info for next iteration
                force_modify_best = True
            else:
                force_modify_best = False

        # Check if the new prompt is too similar to recent ones
        # prompt_too_similar = False
        # if history:
        #     recent_prompts = [h["prompt"] for h in history[-3:]]  # Check last 3 prompts
        #     for recent_prompt in recent_prompts:
        #         similarity = prompt_similarity(optimized, recent_prompt)
        #         if similarity > args.prompt_similarity_threshold:  # If more than threshold similar
        #             prompt_too_similar = True
        #             logger.info(f"   Prompt too similar to previous (similarity={similarity:.3f}); forcing aggressive exploration next iteration")
        #             force_explore_next = True
        #             break

        # Generate and score
        gen_prompt = optimized
        if args.append_suffix:
            suf = args.append_suffix.strip().strip(',')
            if suf and suf.lower() not in gen_prompt.lower():
                gen_prompt = f"{gen_prompt}, {suf}".strip().strip(',')
        img = gen_image(args.gen_server, args.lora, gen_prompt, args.seed, args.steps, args.guidance, args.timeout, logger)
        if img is None:
            logger.info("   Generation failed; skipping iteration")
            continue
        img_path = os.path.join(args.image_dir, f"iter_{itr}.jpg")
        save_image(img, img_path)

        tt = clip_text_text(model, tokenizer, device, args.prompt, optimized)
        ti = clip_text_image(model, tokenizer, normalize, device, args.prompt, img)
        # Interrogate this generated image once, compute tt2 (original vs caption)
        gen_caption = interrogator.interrogate_image(img, style_focus="clip_optimized") or ""
        tt2 = clip_text_text(model, tokenizer, device, args.prompt, gen_caption) if gen_caption else 0.0
        logger.info(f"   Scores: tt_clip={tt:.4f}, ti_clip={ti:.4f}, tt2_caption={tt2:.4f}")

        # Check for exact score matches in history AFTER computing current scores
        exact_match = False
        if history:
            all_previous_scores = [h["ti"] for h in history]
            if ti in all_previous_scores:
                exact_match = True
                logger.info(f"   Exact score match detected (ti={ti:.4f}) - this score appeared in a previous iteration")
                # Force exploration for next iteration if we detect exact match
                force_explore_next = True

        # Persist
        csv_f.write(
            ",".join([
                str(itr), args.prompt.replace(",", " "), optimized.replace(",", " "), f"{tt:.4f}", f"{ti:.4f}", img_path
            ]) + "\n"
        )
        csv_f.flush()
        jsonl_f.write(json.dumps({
            "iter": itr,
            "orig_prompt": args.prompt,
            "optimized_prompt": optimized,
            "tt_clip": tt,
            "ti_clip": ti,
            "tt2_caption": tt2,
            "gen_caption": gen_caption,
            "baseline_caption": baseline_caption,
            "image_path": img_path
        }) + "\n")
        jsonl_f.flush()

        history.append({"iter": itr, "prompt": optimized, "tt": tt, "ti": ti, "tt2": tt2, "cap": gen_caption})

        # Update best and check convergence
        improved = ti - best_ti
        if improved > args.min_abs_improve:
            best_ti = ti
            best_img = img
            last_improve_round = itr
            logger.info(f"   ✓ Improvement: +{improved:.4f} (best={best_ti:.4f})")
        else:
            logger.info("   ✖ No significant improvement")

        if best_ti >= args.target_clip_raw:
            logger.info("   🎯 Target raw CLIP reached; stopping early")
            break
        if itr - last_improve_round >= args.patience:
            logger.info("   🛑 Patience exceeded without improvement; stopping")
            break

    csv_f.close()
    jsonl_f.close()

    # Optional final validator comparison for original vs best optimized
    if args.final_3d_compare:
        tmpl = args.validator_cmd_template or (
            "bash -c 'source /home/mbhat/miniconda/bin/activate && conda activate trellis_new && "
            "python subnet_accurate_validator.py \"{original}\" \"{optimized}\" --endpoint \"{endpoint}\"'"
        )
        escaped_orig = args.prompt.replace('"', '\\"')
        # Use the last best optimized from history
        best_opt = None
        if history:
            best_opt = max(history, key=lambda h: h.get("ti", 0.0)).get("prompt")
        if best_opt:
            escaped_opt = best_opt.replace('"', '\\"')
            endpoint = args.generate_endpoint
            cmd = tmpl.format(original=escaped_orig, optimized=escaped_opt, endpoint=endpoint)
            logger.info(f"Running final validator: {cmd}")
            import subprocess
            res = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=900, cwd="/home/mbhat/three-gen-subnet-trellis")
            if res.returncode != 0:
                logger.warning(f"Validator failed rc={res.returncode}: {res.stderr.strip()}")
            final_json = None
            try:
                with open("/home/mbhat/three-gen-subnet-trellis/subnet_validation_results.json", "r") as jf:
                    final_json = json.load(jf)
            except Exception:
                try:
                    final_json = json.loads(res.stdout.strip()) if res.stdout.strip() else None
                except Exception:
                    final_json = None
            if final_json:
                logger.info(
                    f"Final 3D validation: score={final_json.get('validation_engine_score')} align={final_json.get('alignment_score')} "
                    f"iqa={final_json.get('quality_score')} ssim={final_json.get('ssim_score')} lpips={final_json.get('lpips_score')}"
                )
            else:
                logger.warning("Could not parse final validator results")

    logger.info("Done.")


if __name__ == "__main__":
    main()

