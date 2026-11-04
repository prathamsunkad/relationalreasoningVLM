"""Baselines on the same sources and readout as eval.py: the plain VLM (zero-shot) or the VLM + a vlm_lora_pc LoRA adapter.
No relational branch. Output records/summary/merge are identical to eval.py, so the tables line up.

  python eval_baseline.py --model_id Qwen/Qwen3-VL-2B-Instruct --source cart=jsonl:$D/clevr_art_ood/rmts_cand.jsonl \
      --out_json out/shard0.json [--adapter ~/scratch/vlm_lora_pc_runs/qwen_pc_recipe_e1/best] [--shard_idx 0 --num_shards 4]
  python eval.py --merge_dir out
"""
import argparse
import json
import os

import torch
from torch.utils.data import DataLoader

from backbones import DTYPES, Backbone, load_vlm
from data import Collator, FixedSet, build_sources, parse_mix
from eval import summarize
from relvlm import ScoreTokens, example_logits


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_id", required=True)
    ap.add_argument("--adapter", default=None, help="PEFT adapter dir (e.g. vlm_lora_pc_runs/<tag>/best); none = zero-shot")
    ap.add_argument("--source", required=True, help="name=type:path")
    ap.add_argument("--out_json", required=True)
    ap.add_argument("--shard_idx", type=int, default=0)
    ap.add_argument("--num_shards", type=int, default=1)
    ap.add_argument("--max_examples", type=int, default=0)
    ap.add_argument("--context_mode", default="normal", choices=["normal", "blank", "shuffle"])
    ap.add_argument("--panel_px", type=int, default=160)
    ap.add_argument("--crop_to_patches", type=int, default=1,
                    help="InternVL tiling, as the processor does by default (the LoRA baseline was trained with 1)")
    ap.add_argument("--file_cache_dir", default=None)
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--dtype", default="bf16", choices=list(DTYPES))
    ap.add_argument("--attn", default="sdpa")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = DTYPES[args.dtype]
    model, processor, family = load_vlm(args.model_id, args.dtype, args.attn, device)
    model.requires_grad_(False).eval()
    bb = Backbone(model, processor, family)
    if args.adapter:
        from peft import PeftModel
        PeftModel.from_pretrained(model, args.adapter)          # injects LoRA layers into the same modules bb uses
    st = ScoreTokens(processor.tokenizer)
    kw = {"crop_to_patches": bool(args.crop_to_patches)} if family == "internvl" else {}

    (name, spec, _), = parse_mix([args.source])
    src = build_sources([(name, spec, 1)], args.panel_px, args.file_cache_dir, args.context_mode)[0]
    idx = list(range(len(src)))[args.shard_idx::args.num_shards]
    if args.max_examples:
        idx = idx[: args.max_examples]
    loader = DataLoader(FixedSet(src, idx, name), batch_size=args.batch_size, num_workers=args.num_workers,
                        collate_fn=Collator(processor, kw, rel_tokens=0))
    records = []
    ac = torch.autocast("cuda", dtype=dtype) if device.type == "cuda" and dtype != torch.float32 \
        else torch.autocast("cpu", enabled=False)
    for batch in loader:
        inputs = bb.model_inputs(batch, device, dtype)
        with ac:
            h = bb.core(**inputs, use_cache=False).last_hidden_state[:, -1].float()
        for ex, r in zip(batch["examples"], example_logits(h, None, batch, bb.model, st)):
            x = r["llm"].float().cpu()
            pred = int(x.argmax()) if r["loss"] == "ce" else int(float(x.reshape(())) > 0)
            records.append(dict(id=ex["id"], source=ex["source"], kind=r["kind"], label=r["label"], llm_pred=pred,
                                llm_scores=[round(float(v), 4) for v in x.reshape(-1)]))
    s = summarize(records)
    print(f"[baseline] {name} shard {args.shard_idx}/{args.num_shards} adapter={args.adapter}: "
          f"{json.dumps({k: v for k, v in s.items() if k != 'by_source'})}")
    os.makedirs(os.path.dirname(os.path.abspath(args.out_json)), exist_ok=True)
    json.dump({"summary": s, "config": vars(args), "records": records}, open(args.out_json, "w"))


if __name__ == "__main__":
    main()
