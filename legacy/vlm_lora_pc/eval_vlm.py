"""Sharded evaluation of a (LoRA-adapted or zero-shot) VLM on a PGM split with PER-CANDIDATE scoring, + merge of the shards.

  # one shard (Slurm array task i of N), same file sharding as pgm_infer.py:
  python eval_vlm.py --model_id Qwen/Qwen3-VL-2B-Instruct --adapter runs/qwen_pc_neutral/best \
      --npz_dir .../test_full/neutral --out_dir results/qwen_pc_neutral --shard_idx $i --num_shards 28
  # zero-shot baseline: omit --adapter (the base model scored with the same Yes/No readout)
  # merge all shards -> summary.json and a printed table:
  python eval_vlm.py --merge_dir results/qwen_pc_neutral

Per problem: the 8 scores logit(Yes) - logit(No), one per completed matrix; the choice is their argmax (always valid).
Shards resume: re-running a shard skips the problems it already finished.
"""
import argparse
import glob
import json
import os
import time

import torch
from torch.utils.data import DataLoader

from evalcore import evaluate_batches, summarize
from model_utils import DTYPES, ScoreTokens, load_adapter, load_base
from pgm_vlm_data import CandidateCollator, PGMStream, list_npz, shard_files


def merge(merge_dir, expected_n=0):
    recs, shards = [], sorted(glob.glob(os.path.join(merge_dir, "results_shard*.json")))
    for p in shards:
        recs += json.load(open(p))
    errs = [r for r in recs if "error" in r]
    recs = [r for r in recs if "error" not in r]
    seen = {r["file"] for r in recs}
    summ = summarize(recs)
    summ.update(shards=len(shards), errors=len(errs), unique_files=len(seen), duplicates=len(recs) - len(seen))
    if expected_n:
        summ["expected_n"], summ["complete"] = expected_n, len(seen) == expected_n
    meta = os.path.join(merge_dir, "eval_config.json")
    if os.path.exists(meta):
        summ["config"] = json.load(open(meta))
    json.dump(summ, open(os.path.join(merge_dir, "summary.json"), "w"), indent=1)
    print(f"{merge_dir}: {summ['n']} problems from {len(shards)} shards ({len(errs)} errors, {summ['duplicates']} duplicates)")
    if "scores" in summ:
        d = summ["scores"]
        print(f"  acc {100*d['acc']:.2f}%  (95% CI {100*d['ci95'][0]:.2f}-{100*d['ci95'][1]:.2f})")
        print(f"  predicted-position histogram {d['pred_hist']}")
        print(f"  answer scored Yes>No {100*d['answer_says_yes']:.1f}% | distractors scored Yes>No {100*d['distractor_says_yes']:.1f}%")
    return summ


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--merge_dir", default=None)
    ap.add_argument("--expected_n", type=int, default=0, help="with --merge_dir: e.g. 200000 for a full PGM test regime")
    ap.add_argument("--model_id", default="Qwen/Qwen3-VL-2B-Instruct")
    ap.add_argument("--family", default=None)
    ap.add_argument("--adapter", default=None, help="LoRA adapter dir; omit for the zero-shot model")
    ap.add_argument("--npz_dir")
    ap.add_argument("--out_dir")
    ap.add_argument("--file_cache_dir", default=None)
    ap.add_argument("--shard_idx", type=int, default=0)
    ap.add_argument("--num_shards", type=int, default=1)
    ap.add_argument("--max_examples", type=int, default=0)
    ap.add_argument("--context_mode", default="normal", choices=["normal", "blank", "shuffle"],
                    help="shortcut check: blank/shuffled context panels should push accuracy to ~12.5%")
    ap.add_argument("--batch_size", type=int, default=4, help="problems (x8 sequences)")
    ap.add_argument("--dtype", default="bf16", choices=list(DTYPES))
    ap.add_argument("--attn", default="sdpa")
    ap.add_argument("--crop_to_patches", type=int, default=1)
    ap.add_argument("--panel_px", type=int, default=160, help="must match training")
    ap.add_argument("--merge_adapter", action="store_true", help="fold LoRA into the weights (faster, tiny bf16 rounding)")
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--flush_every", type=int, default=10, help="batches")
    args = ap.parse_args()

    if args.merge_dir:
        merge(args.merge_dir, args.expected_n)
        return
    assert args.npz_dir and args.out_dir, "--npz_dir and --out_dir are required"
    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = DTYPES[args.dtype]
    out_path = os.path.join(args.out_dir, f"results_shard{args.shard_idx}.json")

    files = shard_files(list_npz(args.npz_dir, os.path.join(args.file_cache_dir, "eval_files.txt") if args.file_cache_dir else None),
                        args.shard_idx, args.num_shards, args.max_examples)
    done = json.load(open(out_path)) if os.path.exists(out_path) else []
    done_files = {r["file"] for r in done if "error" not in r}
    todo = [f for f in files if f not in done_files]
    print(f"[shard {args.shard_idx}/{args.num_shards}] {len(files)} problems, {len(done_files)} already done, {len(todo)} to go")

    model, processor, family = load_base(args.model_id, args.dtype, args.attn, device, args.family)
    st = ScoreTokens(processor.tokenizer)
    if args.adapter:
        model = load_adapter(model, args.adapter)
        if args.merge_adapter:
            model = model.merge_and_unload()
    model.eval()
    json.dump({"model_id": args.model_id, "adapter": args.adapter, "family": family, "npz_dir": args.npz_dir,
               "context_mode": args.context_mode, "crop_to_patches": args.crop_to_patches, "panel_px": args.panel_px,
               "num_shards": args.num_shards}, open(os.path.join(args.out_dir, "eval_config.json"), "w"), indent=1)

    ds = PGMStream(args.npz_dir, todo, seed=0, shuffle_candidates=False, context_mode=args.context_mode)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers,
                        collate_fn=CandidateCollator(processor, family, args.crop_to_patches, cell_size=args.panel_px))
    results, t0, nb = list(done), time.time(), 0

    def flush():
        tmp = out_path + ".tmp"
        json.dump(results, open(tmp, "w"))
        os.replace(tmp, out_path)

    def on_batch(recs):
        nonlocal nb
        results.extend(recs)
        nb += 1
        if nb % args.flush_every == 0:
            flush()
            print(f"[shard {args.shard_idx}] {len(results)}/{len(files)} done, {(time.time()-t0)/max(1,len(results)-len(done)):.3f}s/problem", flush=True)

    for _ in evaluate_batches(model, loader, st, device, dtype, on_batch):
        pass
    flush()
    s = summarize(results)
    print(f"[shard {args.shard_idx}] finished: " + (f"acc {100*s['scores']['acc']:.2f}%" if "scores" in s else "nothing to score"))


if __name__ == "__main__":
    main()
