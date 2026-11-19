"""Sharded evaluation of a Stage-1/2 checkpoint on one source, plus merge.

  python eval.py --ckpt runs/s1_qwen/best.pt --source pgm=matrix:$DATA/test_full/neutral \
      --out_json out/shard0.json --shard_idx 0 --num_shards 28
  python eval.py --merge_dir out --expected_n 200000

--context_mode blank|shuffle (matrix sources) are the shortcut checks: accuracy should fall to ~12.5%.
--readout aux skips the LLM and scores with the aux head only.
"""
import argparse
import glob
import json
import math
import os

import torch
from torch.utils.data import DataLoader

from backbones import DTYPES
from data import Collator, FixedSet, build_sources, parse_mix
from factory import adapter_lora_cfg, build, load_peft_adapter, restore
from train_stage1 import evaluate


def wilson(k, n, z=1.96):
    if n == 0:
        return [float("nan")] * 2
    p, d = k / n, 1 + z * z / n
    c, h = (p + z * z / (2 * n)) / d, z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [c - h, c + h]


def summarize(records):
    out = {"n": len(records)}
    for key in ("llm", "aux"):
        rs = [r for r in records if f"{key}_pred" in r]
        if rs:
            k = sum(r[f"{key}_pred"] == r["label"] for r in rs)
            out[key] = {"acc": k / len(rs), "ci95": wilson(k, len(rs)), "n": len(rs)}
            if all(r["kind"] == "cand" for r in rs):
                out[key]["pred_hist"] = [sum(r[f"{key}_pred"] == i for r in rs) for i in range(8)]
    by_src = {}
    for r in records:
        by_src.setdefault(r["source"], []).append(r)
    if len(by_src) > 1:
        out["by_source"] = {s: summarize(v) for s, v in by_src.items()}
    return out


def merge(d, expected_n=0):
    recs = []
    for f in sorted(glob.glob(os.path.join(d, "*shard*.json"))):
        recs += json.load(open(f))["records"]
    ids = [r["id"] for r in recs]
    s = summarize(recs)
    s.update(shards=len(glob.glob(os.path.join(d, "*shard*.json"))), unique=len(set(ids)), duplicates=len(ids) - len(set(ids)))
    if expected_n and len(set(ids)) != expected_n:
        print(f"[warn] {len(set(ids))} unique examples, expected {expected_n}")
    json.dump(s, open(os.path.join(d, "merged.json"), "w"), indent=1)
    print(json.dumps({k: v for k, v in s.items() if k != "by_source"}, indent=1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt")
    ap.add_argument("--source", help="name=type:path")
    ap.add_argument("--out_json")
    ap.add_argument("--shard_idx", type=int, default=0)
    ap.add_argument("--num_shards", type=int, default=1)
    ap.add_argument("--max_examples", type=int, default=0)
    ap.add_argument("--context_mode", default="normal", choices=["normal", "blank", "shuffle"])
    ap.add_argument("--readout", default="llm", choices=["llm", "aux"])
    ap.add_argument("--branch_off", action="store_true",
                    help="ablation: cut the branch's input to the LLM (fusion: lambda = 1; interleave: zero relation "
                         "token) to see what LoRA learned on its own")
    ap.add_argument("--rel_ablate", default="none", choices=["none", "zero", "drop", "shuffle_cand", "shuffle_puzzle", "permute"],
                    help="interleave only: ablate the relation tokens at evaluation (see RelationalVLM._ablate_tokens)")
    ap.add_argument("--lora_from", default=None,
                    help="score with this PEFT adapter (e.g. a vlm_lora_pc run's best/) as the LoRA; with a fusion "
                         "checkpoint and --branch_off this is LoRA alone, read through this code")
    ap.add_argument("--val_list", default=None, help="file of .npz names (first tab column): evaluate only these")
    ap.add_argument("--panel_px", type=int, default=None, help="default: the training value")
    ap.add_argument("--file_cache_dir", default=None)
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--dtype", default="bf16", choices=list(DTYPES))
    ap.add_argument("--merge_dir", default=None)
    ap.add_argument("--expected_n", type=int, default=0)
    args = ap.parse_args()
    if args.merge_dir:
        return merge(args.merge_dir, args.expected_n)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = DTYPES[args.dtype]
    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    targs = ck.get("args", {})
    lora_cfg = ck.get("lora_cfg") or (adapter_lora_cfg(args.lora_from) if args.lora_from else None)
    rv, bb, st = build(ck["model_id"], ck["slot_ckpt"], ck["cfg"], device, args.dtype, targs.get("attn", "sdpa"),
                       lora_cfg)
    restore(rv, bb, ck)
    if args.lora_from:
        n = load_peft_adapter(bb.model, args.lora_from, lora_cfg)
        print(f"[eval] LoRA from {args.lora_from} ({n} tensors" + (", replacing the checkpoint's own)" if ck.get("lora") else ")"))
    if args.rel_ablate != "none":
        assert rv.interleave and not args.branch_off, "--rel_ablate is for interleave checkpoints, without --branch_off"
        rv.rel_ablate = args.rel_ablate
        print(f"[eval] relation tokens ablated: {args.rel_ablate}")
    if args.branch_off:
        rv.branch_off = True                         # fusion: native visual tokens; interleave: zero relation token
        print("[eval] branch off: the LLM gets nothing from the branch")
    panel_px = args.panel_px or targs.get("panel_px", 160)
    (name, spec, _), = parse_mix([args.source])
    src = build_sources([(name, spec, 1)], panel_px, args.file_cache_dir, args.context_mode)[0]
    idx = list(range(len(src)))
    if args.val_list and hasattr(src, "files"):
        keep = {l.split("\t")[0].strip() for l in open(args.val_list) if l.strip() and not l.startswith("#")}
        idx = [j for j, f in enumerate(src.files) if f in keep]
        print(f"[eval] {len(idx)} problems from {args.val_list}")
    idx = idx[args.shard_idx::args.num_shards]
    if args.max_examples:
        idx = idx[: args.max_examples]
    collate = Collator(bb.processor, bb.processor_kwargs(), **rv.collate_kwargs())
    loader = DataLoader(FixedSet(src, idx, name), batch_size=args.batch_size, num_workers=args.num_workers,
                        collate_fn=collate)
    records = []
    res = evaluate(rv, bb, st, loader, device, dtype, use_llm=args.readout == "llm", on_record=records.append)
    print(f"[eval] {name} shard {args.shard_idx}/{args.num_shards} ({args.context_mode}): {res}")
    os.makedirs(os.path.dirname(os.path.abspath(args.out_json)), exist_ok=True)
    json.dump({"summary": summarize(records), "config": vars(args), "ckpt_step": ck.get("step"),
               "lambda": float(rv.lam()) if rv.fuse else None, "records": records}, open(args.out_json, "w"))


if __name__ == "__main__":
    main()
