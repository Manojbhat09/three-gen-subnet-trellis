#!/usr/bin/env python3
"""
Offline CLIP Optimization → DPO Dataset Builder (single fixed LoRA)

Usage examples:
  # Basic: optimize prompts.txt with baolei LoRA against local server and build datasets
  python offline_clip_optimization_to_dpo.py \
    --prompts_file prompts.txt \
    --server_url http://localhost:8096 \
    --lora baolei \
    --out_dir offline_outputs/baolei \
    --seed 42

  # Try multiple seeds and limit prompts
  python offline_clip_optimization_to_dpo.py \
    --prompts_file prompts.txt \
    --server_url http://localhost:8096 \
    --lora baolei \
    --out_dir offline_outputs/baolei \
    --seed 42 --extra_seeds 77,123 \
    --limit 200

This script:
  - Loads prompts from a file (one per line)
  - Runs the CLIP feedback optimizer OFFLINE, but using your running image endpoints, for a single fixed LoRA
  - Collects (original, optimized, score) per prompt (and per optional seed)
  - Writes JSONL with examples and builds SFT and DPO datasets via finetune/build_dpo_dataset.py

Notes:
  - Uses the same CLIP model and preprocessing as production (convnext_large_d/laion2b_s26b_b102k_augreg)
  - No LoRA search; strictly uses the provided --lora endpoint
  - Expects your generation server at --server_url to expose /generate_image/<lora>/
"""

import argparse
import json
import os
import sys
import time
from typing import List, Dict

# Ensure local imports work
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.append(SCRIPT_DIR)

from prompt_optimization_engine import CLIPAlignmentOptimizer  # noqa: E402
from finetune.build_dpo_dataset import build_sft_and_dpo  # noqa: E402


def read_prompts(path: str, limit: int | None) -> List[str]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Prompts file not found: {path}")
    prompts: List[str] = []
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            p = line.strip()
            if not p:
                continue
            prompts.append(p)
            if limit is not None and len(prompts) >= limit:
                break
    return prompts


def save_jsonl(rows: List[Dict], path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def optimize_single_prompt(
    optimizer: CLIPAlignmentOptimizer,
    original_prompt: str,
    lora: str,
    seed: int,
) -> Dict:
    """Run the single-LoRA optimization and return a result row.
    Returns keys: original, optimized, score, original_score, lora, seed, duration_sec
    """
    t0 = time.time()
    try:
        result = optimizer.optimize_for_lora_endpoint(original_prompt, lora, seed)
        duration = time.time() - t0
        row = {
            "original": original_prompt,
            "optimized": result.optimized_prompt,
            "score": float(result.optimized_score),
            "original_score": float(result.original_score),
            "lora": lora,
            "seed": seed,
            "duration_sec": round(duration, 3),
            "strategy_used": result.strategy_used,
            "iteration": int(result.iteration),
        }
        return row
    except Exception as e:
        duration = time.time() - t0
        return {
            "original": original_prompt,
            "optimized": original_prompt,
            "score": 0.0,
            "original_score": 0.0,
            "lora": lora,
            "seed": seed,
            "duration_sec": round(duration, 3),
            "error": str(e),
        }


def main():
    parser = argparse.ArgumentParser(description="Offline CLIP optimization to build DPO datasets (single fixed LoRA)")
    parser.add_argument("--prompts_file", type=str, required=True, help="Path to a text file with one prompt per line")
    parser.add_argument("--server_url", type=str, default="http://localhost:8096", help="Base URL of the generation server")
    parser.add_argument("--lora", type=str, default="baolei", help="Fixed LoRA endpoint key (e.g., baolei, isometric_3d)")
    parser.add_argument("--out_dir", type=str, default="offline_outputs/baolei", help="Output directory for datasets and logs")
    parser.add_argument("--seed", type=int, default=42, help="Base seed")
    parser.add_argument("--extra_seeds", type=str, default="", help="Comma-separated extra seeds, e.g., '77,123'")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of prompts to process")
    parser.add_argument("--pos_threshold", type=float, default=0.8, help="Positive threshold for DPO pair building")
    parser.add_argument("--neg_threshold", type=float, default=0.3, help="Negative threshold for DPO pair building")
    parser.add_argument("--top_k", type=int, default=2, help="Top-k positives per original for DPO")
    parser.add_argument("--bottom_k", type=int, default=2, help="Bottom-k negatives per original for DPO")
    parser.add_argument("--max_pairs_per_original", type=int, default=8, help="Max DPO pairs per original prompt")
    parser.add_argument("--validate_topk", type=int, default=2, help="Validate top-K CLIP winners per original with production validator")
    parser.add_argument("--validate_fraction", type=float, default=0.2, help="Fraction of originals to run production validation on [0,1]")
    parser.add_argument("--run_training", action="store_true", help="Run DPO warmup-on-silver then finish-on-gold after dataset build")
    parser.add_argument("--base_model", type=str, default="unsloth/Llama-3.2-3B-Instruct", help="Base model for DPO training")
    parser.add_argument("--out_train_dir", type=str, default="finetune_outputs/baolei", help="Training output directory root")

    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    examples_path = os.path.join(args.out_dir, "offline_examples.jsonl")
    sft_path = os.path.join(args.out_dir, "sft.jsonl")
    dpo_path = os.path.join(args.out_dir, "dpo.jsonl")

    prompts = read_prompts(args.prompts_file, args.limit)
    print(f"Loaded {len(prompts)} prompts from {args.prompts_file}")

    # Initialize optimizer with the server URL that exposes /generate_image/<lora>/
    optimizer = CLIPAlignmentOptimizer(hunyuan_server_url=args.server_url)
    # Load CLIP scorer once for speed
    optimizer.load_clip_model()

    # Prepare seeds
    seeds: List[int] = [int(args.seed)]
    if args.extra_seeds.strip():
        try:
            seeds.extend([int(s.strip()) for s in args.extra_seeds.split(",") if s.strip()])
        except Exception:
            print("⚠️  Failed to parse --extra_seeds; ignoring.")

    # Iterate and optimize
    rows: List[Dict] = []
    total = len(prompts) * len(seeds)
    n = 0
    t_start = time.time()
    for p in prompts:
        for s in seeds:
            n += 1
            print(f"[{n}/{total}] Optimizing (lora={args.lora}, seed={s}): '{p[:80]}'")
            row = optimize_single_prompt(optimizer, p, args.lora, s)
            rows.append(row)

            # Flush periodically
            if n % 20 == 0:
                save_jsonl(rows, examples_path)

    # Final save
    save_jsonl(rows, examples_path)

    # Build SFT and DPO datasets
    # Convert to examples expected by builder: {original, optimized, score}
    examples = [{"original": r["original"], "optimized": r["optimized"], "score": float(r.get("score", 0.0))} for r in rows]

    # Optional: Production validator spot-check on top-K winners per original
    # Select a random subset of originals based on --validate_fraction
    diagnostics_path = os.path.join(args.out_dir, "validation_diagnostics.jsonl")
    if args.validate_topk > 0 and 0.0 < args.validate_fraction <= 1.0:
        import random
        from collections import defaultdict
        grouped: Dict[str, List[Dict]] = defaultdict(list)
        for e in rows:
            grouped[e["original"]].append(e)
        originals = list(grouped.keys())
        random.shuffle(originals)
        sample_count = max(1, int(len(originals) * args.validate_fraction))
        sampled_originals = set(originals[:sample_count])
        diag_rows: List[Dict] = []
        # We will run the local production-accurate validator script for spot checks
        import subprocess
        for orig in sampled_originals:
            candidates = sorted(grouped[orig], key=lambda x: x.get("score", -1e9), reverse=True)[: args.validate_topk]
            for c in candidates:
                # Call: python subnet_accurate_validator.py "<original>" "<optimized>" "generate/<lora>/"
                endpoint = f"generate/{args.lora}/"
                cmd = [
                    sys.executable,
                    os.path.join(SCRIPT_DIR, "subnet_accurate_validator.py"),
                    orig,
                    c["optimized"],
                    endpoint,
                ]
                try:
                    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
                    # Attempt to read results file if produced
                    result_file = os.path.join(SCRIPT_DIR, "subnet_validation_results.json")
                    result_json = None
                    if os.path.exists(result_file):
                        with open(result_file, "r") as rf:
                            result_json = json.load(rf)
                    diag = {
                        "original": orig,
                        "optimized": c["optimized"],
                        "clip_score": float(c.get("score", 0.0)),
                        "validator_rc": proc.returncode,
                        "validator_stdout_tail": proc.stdout[-4000:],
                        "validator_stderr_tail": proc.stderr[-4000:],
                        "production_validation": result_json,
                    }
                    diag_rows.append(diag)
                except Exception as e:
                    diag_rows.append({
                        "original": orig,
                        "optimized": c["optimized"],
                        "clip_score": float(c.get("score", 0.0)),
                        "error": str(e),
                    })
        save_jsonl(diag_rows, diagnostics_path)
        print(f"   Diagnostics saved to: {diagnostics_path}")

    print("Building SFT and DPO datasets...")
    _ = build_sft_and_dpo(
        examples=examples,
        out_sft_jsonl=sft_path,
        out_dpo_jsonl=dpo_path,
        pos_threshold=args.pos_threshold,
        neg_threshold=args.neg_threshold,
        top_k=args.top_k,
        bottom_k=args.bottom_k,
        pair_strategy="cross",
        max_pairs_per_original=args.max_pairs_per_original,
    )

    elapsed = time.time() - t_start
    print(f"\n✅ Done. Examples: {len(rows)} | Time: {elapsed/60:.1f} min")
    print(f"   Saved examples to: {examples_path}")
    print(f"   SFT dataset:       {sft_path}")
    print(f"   DPO dataset:       {dpo_path}")
    if os.path.exists(diagnostics_path):
        print(f"   Prod-validation diagnostics: {diagnostics_path}")

    # Optional: run DPO warmup-on-silver then finish-on-gold (two-phase training)
    if args.run_training:
        print("\n🚀 Starting two-phase DPO training: warmup on silver, finish on gold")
        from datasets import Dataset
        from finetune.utils import (
            ModelConfig,
            load_base_model_and_tokenizer,
            apply_peft_lora,
            build_dpo_trainer,
            set_seed,
        )
        set_seed(21)
        # Load base policy
        cfg = ModelConfig(base_model=args.base_model, max_seq_length=2048, dtype=None, load_in_4bit=True)
        policy_model, tokenizer = load_base_model_and_tokenizer(cfg)
        policy_model = apply_peft_lora(policy_model, r=16, lora_alpha=16, lora_dropout=0.0)
        ref_model = None  # Let TRL create frozen copy internally

        # Build silver and gold datasets
        # Silver: use all CLIP-only examples (DPO pairs constructed already in dpo.jsonl)
        # Gold: filter diagnostics with production_validation present and high score
        def load_jsonl_rows(path: str) -> List[Dict]:
            if not os.path.exists(path):
                return []
            out = []
            with open(path, "r") as f:
                for line in f:
                    if line.strip():
                        out.append(json.loads(line))
            return out

        # Build DPO datasets as HF Dataset objects with prompt/chosen/rejected
        # Reuse builder outputs directly when possible
        # Here, we reconstruct minimal DPO format from our examples by pairing top vs bottom within each original
        from collections import defaultdict
        by_orig: Dict[str, List[Dict]] = defaultdict(list)
        for e in examples:
            by_orig[e["original"]].append(e)
        dpo_pairs = []
        for orig, lst in by_orig.items():
            lst_sorted = sorted(lst, key=lambda x: x.get("score", -1e9))
            # bottom
            negs = lst_sorted[: max(1, args.bottom_k)]
            # top
            poss = list(reversed(lst_sorted))[: max(1, args.top_k)]
            for p in poss:
                for n in negs:
                    if p["optimized"] != n["optimized"]:
                        dpo_pairs.append({
                            "prompt": orig,
                            "chosen": p["optimized"],
                            "rejected": n["optimized"],
                        })
        silver_ds = Dataset.from_list(dpo_pairs)

        # Gold: from diagnostics file, pick entries with production validation score present and > pos_threshold
        gold_rows = load_jsonl_rows(diagnostics_path)
        gold_pairs = []
        for r in gold_rows:
            prod = r.get("production_validation")
            if not prod:
                continue
            val_score = prod.get("validation_engine_score", 0.0)
            if val_score >= args.pos_threshold:
                # For gold pairs, use chosen = this optimized; rejected = original or lowest-CLIP variant
                gold_pairs.append({
                    "prompt": r["original"],
                    "chosen": r["optimized"],
                    "rejected": r["original"],
                })
        if not gold_pairs:
            print("⚠️  No gold pairs found in diagnostics; finishing-on-gold will be skipped.")
        gold_ds = Dataset.from_list(gold_pairs) if gold_pairs else None

        # Phase A: warmup on silver
        out_silver = os.path.join(args.out_train_dir, "dpo_silver")
        os.makedirs(out_silver, exist_ok=True)
        trainer_silver = build_dpo_trainer(
            model=policy_model,
            ref_model=ref_model,
            tokenizer=tokenizer,
            train_dataset=silver_ds,
            eval_dataset=None,
            output_dir=out_silver,
            max_seq_length=2048,
            beta=0.1,
            per_device_train_batch_size=2,
            gradient_accumulation_steps=4,
            warmup_steps=10,
            learning_rate=1e-5,
            logging_steps=5,
            max_steps=300,
            save_steps=100,
            eval_steps=100,
            report_to="none",
            run_name="dpo_silver_warmup",
        )
        trainer_silver.train()
        # Save adapters
        trainer_silver.save_model()

        # Phase B: finish on gold (if available)
        if gold_ds is not None and len(gold_ds) > 0:
            out_gold = os.path.join(args.out_train_dir, "dpo_gold")
            os.makedirs(out_gold, exist_ok=True)
            trainer_gold = build_dpo_trainer(
                model=policy_model,
                ref_model=ref_model,
                tokenizer=tokenizer,
                train_dataset=gold_ds,
                eval_dataset=None,
                output_dir=out_gold,
                max_seq_length=2048,
                beta=0.1,
                per_device_train_batch_size=2,
                gradient_accumulation_steps=4,
                warmup_steps=10,
                learning_rate=5e-6,
                logging_steps=5,
                max_steps=150,
                save_steps=75,
                eval_steps=75,
                report_to="none",
                run_name="dpo_gold_finish",
            )
            trainer_gold.train()
            trainer_gold.save_model()
            print(f"   ✅ Finished on gold. Saved to: {out_gold}")
        else:
            print("   ⚠️  Skipped gold finishing; none available.")

        print(f"   ✅ Warmup on silver complete. Saved to: {out_silver}")


if __name__ == "__main__":
    main() 