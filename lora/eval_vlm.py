"""Sharded evaluation of a (LoRA-adapted or zero-shot) VLM on a PGM split with PER-CANDIDATE scoring, + merge of the shards.

  # one shard (Slurm array task i of N), same file sharding as pgm_infer.py:
  python lora/eval_vlm.py --model_id Qwen/Qwen3-VL-2B-Instruct --adapter runs/qwen_pc_neutral/best \
      --npz_dir .../test_full/neutral --out_dir results/qwen_pc_neutral --shard_idx $i --num_shards 28
  # zero-shot baseline: omit --adapter (the base model scored with the same Yes/No readout)
  # merge all shards -> summary.json and a printed table:
  python lora/eval_vlm.py --merge_dir results/qwen_pc_neutral

Per problem: the 8 scores logit(Yes) - logit(No), one per completed matrix; the choice is their argmax (always valid).
Shards resume: re-running a shard skips the problems it already finished.
"""
import argparse
import sys
import glob
import json
import os
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repo root

import torch
from torch.utils.data import DataLoader

from pgmvlm.evalcore import evaluate_batches, summarize
from pgmvlm.models import DTYPES, load_adapter, load_base
from pgmvlm.data import shard_files


from pgmvlm.config import parse_configured
from pgmvlm.protocols import add_protocol_args, build_protocol
from pgmvlm.datasets import files_for, make_stream
from pgmvlm.extensions import add_extension_args, attach_extension, check_checkpoint_protocol, set_extension_enabled
from pgmvlm.runtime import eval_contract
from pgmvlm.checkpoints import checkpoint_reader


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
        summ["expected_n"], summ["complete"] = expected_n, len(seen) == expected_n and len(recs) == len(seen) and not errs
    meta = os.path.join(merge_dir, "eval_config.json")
    if os.path.exists(meta):
        summ["config"] = json.load(open(meta))
    json.dump(summ, open(os.path.join(merge_dir, "summary.json"), "w"), indent=1)
    print(f"{merge_dir}: {summ['n']} problems from {len(shards)} shards ({len(errs)} errors, {summ['duplicates']} duplicates)")
    if "scores" in summ:
        d = summ["scores"]
        print(f"  acc {100*d['acc']:.2f}%  (95% CI {100*d['ci95'][0]:.2f}-{100*d['ci95'][1]:.2f})")
        print(f"  predicted-position histogram {d['pred_hist']}")
        if "answer_says_yes" in d:
            print(f"  answer scored Yes>No {100*d['answer_says_yes']:.1f}% | distractors scored Yes>No {100*d['distractor_says_yes']:.1f}%")
    return summ


def main():
    ap = argparse.ArgumentParser()
    add_protocol_args(ap)
    add_extension_args(ap)
    ap.add_argument("--disable_extension", action="store_true", help="Paired bypass ablation; retain identical trained LoRA and inputs")
    ap.add_argument('--revision', default=None)
    ap.add_argument('--manifest', default=None)
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
    args = parse_configured(ap)

    if args.merge_dir:
        merge(args.merge_dir, args.expected_n)
        return
    if args.adapter:
        with checkpoint_reader(args.adapter) as checkpoint:
            args.adapter = str(checkpoint)
            return evaluate(args)
    return evaluate(args)


def evaluate(args):
    assert (args.npz_dir or args.manifest) and args.out_dir, "data and --out_dir are required"
    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = DTYPES[args.dtype]
    out_path = os.path.join(args.out_dir, f"results_shard{args.shard_idx}.json")

    if args.num_shards < 1 or not 0 <= args.shard_idx < args.num_shards:
        raise ValueError('Invalid shard index/count')
    files = shard_files(files_for(args, 'eval'), args.shard_idx, args.num_shards, args.max_examples)
    eval_contract(args, files_for(args, 'eval'))
    done = json.load(open(out_path)) if os.path.exists(out_path) else []
    done_names = [r['file'] for r in done if 'error' not in r]
    if len(done_names) != len(set(done_names)) or not set(done_names).issubset(files):
        raise ValueError('Existing predictions have duplicate or out-of-shard IDs')
    done_files = {r["file"] for r in done if "error" not in r}
    todo = [f for f in files if f not in done_files]
    print(f"[shard {args.shard_idx}/{args.num_shards}] {len(files)} problems, {len(done_files)} already done, {len(todo)} to go")

    model, processor, family = load_base(args.model_id, args.dtype, args.attn, device, args.family,
                                         **({'revision': args.revision} if args.revision else {}))
    st = build_protocol(args, processor, family)
    if args.adapter:
        check_checkpoint_protocol(args.adapter, args)
        if os.path.isfile(os.path.join(args.adapter, 'adapter_config.json')):
            model = load_adapter(model, args.adapter)
        if args.merge_adapter and hasattr(model, 'merge_and_unload'):
            model = model.merge_and_unload()
    controller = attach_extension(model, processor, args, args.adapter if args.extension != 'none' else None)
    if controller is not None and args.readout == 'llm':
        set_extension_enabled(model, controller, not args.disable_extension)
    elif args.disable_extension:
        raise ValueError('Bypass requires an LLM-readout extension')
    model.eval()
    json.dump({"model_id": args.model_id, "adapter": args.adapter, "family": family, "npz_dir": args.npz_dir,
               "context_mode": args.context_mode, "crop_to_patches": args.crop_to_patches, "panel_px": args.panel_px,
               "num_shards": args.num_shards, "protocol": args.protocol, "pgm_presentation": args.pgm_presentation, "disable_extension": args.disable_extension, "answer_loss_scope": args.answer_loss_scope, "manifest": args.manifest}, open(os.path.join(args.out_dir, "eval_config.json"), "w"), indent=1)

    ds = make_stream(args, 'eval', todo, context_mode=args.context_mode)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers,
                        collate_fn=st.collator)
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
