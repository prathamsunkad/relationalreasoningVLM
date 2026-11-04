"""Stage 1 (and Stage 2): train the relational branch against the frozen VLM.

Stage 1   trainable: Abstractor + relation-token projector and/or fusion projector + lambda (+ aux head).
          Frozen: ViT, merger/projector, LLM, slot attention (Stage 0 checkpoint).
Stage 2   = Stage 1 + --lora_r 16 (LoRA on the LLM, the baseline's targets), usually --init_from a Stage-1 best.pt.

  python train_stage1.py --model_id Qwen/Qwen3-VL-2B-Instruct --slot_ckpt runs/stage0_qwen/best.pt \
      --out_dir runs/s1_qwen_interleave --inject interleave \
      --train pgm=matrix:$DATA/train_full/neutral --val pgm=matrix:$DATA/val_full/neutral --epochs 1

--objective llm   loss on the LLM's own readout (Yes/No or letter logits)                         [default]
            aux   loss on the aux head only (no LLM forward: cheap Abstractor pre-training, the paper's Stage 1a)
            both  llm + --aux_weight * aux
--batch_size counts EXAMPLES (a PGM problem = 8 sequences). Resumes from <out_dir>/state.pt.
"""
import argparse
import json
import math
import os
import time
from collections import defaultdict

import torch
from torch.utils.data import DataLoader

from backbones import DTYPES
from common import TimeBudget, log_jsonl, lr_at, save_ckpt, seed_everything
from data import Collator, FixedSet, MixtureStream, build_sources, parse_mix, subset
from factory import build, checkpoint_payload, restore, trainable_lora
from relvlm import example_logits, loss_and_acc


def parse():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_id", default="Qwen/Qwen3-VL-2B-Instruct")
    ap.add_argument("--slot_ckpt", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--train", action="append", required=True, help="name=type:path[:weight]; the FIRST is the primary")
    ap.add_argument("--val", action="append", default=[], help="name=type:path; best.pt follows the first")
    ap.add_argument("--val_samples", type=int, default=1000)
    ap.add_argument("--fit_eval_samples", type=int, default=0,
                    help="overfit check: also score the first N TRAINING problems of the primary source at every "
                         "validation (listed first, so best.pt follows it). With --train_samples N this should climb "
                         "towards 100%%; staying near chance means the mode cannot learn through the frozen LLM")
    ap.add_argument("--val_list", default=None, help="file of .npz names (first tab column) for the first --val, "
                                                     "e.g. vlm_lora_pc_runs/filecache/neutral/val1000_shared.tsv")
    ap.add_argument("--file_cache_dir", default=None)
    ap.add_argument("--train_samples", type=int, default=0, help="nested subset of the primary source (0 = all)")
    ap.add_argument("--epochs", type=float, default=1.0, help="passes over the primary source (if --max_steps 0)")
    ap.add_argument("--max_steps", type=int, default=0)
    ap.add_argument("--panel_px", type=int, default=160)
    # branch
    ap.add_argument("--inject", default="interleave", choices=["interleave", "fuse"],
                    help="two separate training regimes: relation tokens in the prompt vs lambda-fusion into visual tokens")
    ap.add_argument("--rel_tokens", type=int, default=1, help="M relation tokens per sequence (interleave)")
    ap.add_argument("--fuse_lambda", type=float, default=0.9, help="initial lambda (weight of the native visual token)")
    ap.add_argument("--fixed_lambda", action="store_true", help="do not learn lambda")
    ap.add_argument("--layers", type=int, default=24)
    ap.add_argument("--heads", type=int, default=8)
    ap.add_argument("--head_dim", type=int, default=32)
    ap.add_argument("--mlp_dim", type=int, default=512)
    ap.add_argument("--objective", default="llm", choices=["llm", "aux", "both"])
    ap.add_argument("--aux_weight", type=float, default=0.5)
    ap.add_argument("--init_from", default=None, help="Stage-1 checkpoint to start the branch from (Stage 2)")
    # LoRA (Stage 2)
    ap.add_argument("--lora_r", type=int, default=0)
    ap.add_argument("--lora_alpha", type=int, default=32)
    ap.add_argument("--lora_dropout", type=float, default=0.05)
    ap.add_argument("--lora_lr", type=float, default=1e-4)
    # optimisation
    ap.add_argument("--batch_size", type=int, default=2)
    ap.add_argument("--grad_accum", type=int, default=8, help="global batch = batch_size x grad_accum examples (16)")
    ap.add_argument("--lr", type=float, default=1e-4, help="Abstractor")
    ap.add_argument("--proj_lr", type=float, default=3e-4, help="projectors, lambda, aux head")
    ap.add_argument("--lr_schedule", default="cosine", choices=["constant", "cosine"])
    ap.add_argument("--warmup_steps", type=int, default=500)
    ap.add_argument("--weight_decay", type=float, default=0.01)
    ap.add_argument("--max_grad_norm", type=float, default=1.0)
    ap.add_argument("--dtype", default="bf16", choices=list(DTYPES))
    ap.add_argument("--attn", default="sdpa")
    ap.add_argument("--grad_ckpt", action="store_true")
    ap.add_argument("--num_workers", type=int, default=6)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--time_limit_min", type=float, default=0)
    ap.add_argument("--log_every", type=int, default=10)
    ap.add_argument("--val_every", type=int, default=1000)
    ap.add_argument("--val_batch_size", type=int, default=4)
    ap.add_argument("--save_every", type=int, default=500)
    return ap.parse_args()


def branch_cfg(args):
    return dict(inject=args.inject, rel_tokens=args.rel_tokens, fuse_lambda=args.fuse_lambda,
                learn_lambda=not args.fixed_lambda,
                abstractor_cfg=dict(layers=args.layers, heads=args.heads, head_dim=args.head_dim, mlp_dim=args.mlp_dim))


@torch.no_grad()
def evaluate(rv, bb, st, loader, device, dtype, use_llm, on_record=None):
    """Accuracy of the LLM readout and of the aux head. on_record(rec) receives every per-example record."""
    rv.eval()
    tot = defaultdict(float)
    for batch in loader:
        h, aux = rv(batch, device, dtype, llm=use_llm)
        recs = example_logits(h, aux, batch, bb.model, st)
        for ex, r in zip(batch["examples"], recs):
            rec = dict(id=ex["id"], source=ex["source"], kind=r["kind"], label=r["label"])
            for key in ("llm", "aux"):
                x = r[key]
                if x is None:
                    continue
                x = x.float().cpu()
                pred = int(x.argmax()) if r["loss"] == "ce" else int(float(x.reshape(())) > 0)
                rec[f"{key}_pred"], rec[f"{key}_scores"] = pred, [round(float(v), 4) for v in x.reshape(-1)]
                tot[f"{key}_correct"] += pred == r["label"]
                tot[f"{key}_n"] += 1
            if on_record:
                on_record(rec)
    out = {f"{k}_acc": tot[f"{k}_correct"] / tot[f"{k}_n"] for k in ("llm", "aux") if tot[f"{k}_n"]}
    out["n"] = int(max(tot["llm_n"], tot["aux_n"]))
    return out


def set_modes(rv, bb, args):
    rv.train()
    bb.model.eval()                                       # frozen VLM behaves as at inference
    if args.lora_r or args.grad_ckpt:
        bb.lm.train()                                     # LoRA dropout / HF checkpointing only run in train mode


def main():
    args = parse()
    os.makedirs(args.out_dir, exist_ok=True)
    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = DTYPES[args.dtype]
    metrics = os.path.join(args.out_dir, "metrics.jsonl")
    state_path = os.path.join(args.out_dir, "state.pt")
    use_llm = args.objective != "aux"

    lora = dict(r=args.lora_r, alpha=args.lora_alpha, dropout=args.lora_dropout) if args.lora_r else None
    rv, bb, st = build(args.model_id, args.slot_ckpt, branch_cfg(args), device, args.dtype, args.attn, lora)
    if args.grad_ckpt:
        bb.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    if args.init_from and not os.path.exists(state_path):
        ck = torch.load(args.init_from, map_location="cpu", weights_only=False)
        restore(rv, bb, {"branch": ck["branch"]}, strict=False)
        print(f"[stage1] branch initialised from {args.init_from}")

    # ---------------- data
    mix = parse_mix(args.train)
    srcs = build_sources(mix, args.panel_px, args.file_cache_dir)
    if args.train_samples and hasattr(srcs[0], "files"):
        srcs[0].files = subset(srcs[0].files, args.train_samples, args.seed)
    weights = [w for *_, w in mix]
    glob_bs = args.batch_size * args.grad_accum
    frac0 = weights[0] / sum(weights)
    max_steps = args.max_steps or max(1, math.ceil(args.epochs * len(srcs[0]) / (glob_bs * frac0)))
    stream = MixtureStream(srcs, weights, [n for n, *_ in mix], max_steps * glob_bs, args.seed)
    for (name, spec, w), s in zip(mix, srcs):
        print(f"[stage1] train source {name}: {spec} | {len(s)} items | weight {w / sum(weights):.2f}")
    print(f"[stage1] global batch {glob_bs} examples | {max_steps} steps | inject={args.inject} objective={args.objective}")
    collate = Collator(bb.processor, bb.processor_kwargs(), rel_tokens=args.rel_tokens if rv.interleave else 0)

    vals = []
    if args.fit_eval_samples > 0:      # training problems in their original order (the stream is just a reordering)
        vals.append(("fit", DataLoader(FixedSet(srcs[0], range(min(args.fit_eval_samples, len(srcs[0]))), "fit"),
                                       batch_size=args.val_batch_size, num_workers=min(2, args.num_workers),
                                       collate_fn=collate)))
    for i, (vname, vspec, _) in enumerate(parse_mix(args.val)):
        vsrc = build_sources([(vname, vspec, 1)], args.panel_px, args.file_cache_dir)[0]
        if i == 0 and args.val_list and hasattr(vsrc, "files"):
            keep = {l.split("\t")[0].strip() for l in open(args.val_list) if l.strip() and not l.startswith("#")}
            idx = [j for j, f in enumerate(vsrc.files) if f in keep]
        else:
            idx = subset(list(range(len(vsrc))), args.val_samples, 1234)
        vals.append((vname, DataLoader(FixedSet(vsrc, idx, vname), batch_size=args.val_batch_size,
                                       num_workers=min(2, args.num_workers), collate_fn=collate)))

    # ---------------- optimiser
    groups = [{"params": list(rv.abstractor.parameters()), "lr": args.lr, "base": args.lr},
              {"params": [p for n, p in rv.named_parameters() if not n.startswith("abstractor.") and p.requires_grad],
               "lr": args.proj_lr, "base": args.proj_lr}]
    if lora:
        groups.append({"params": list(trainable_lora(bb.model).values()), "lr": args.lora_lr, "base": args.lora_lr})
    params = [p for g in groups for p in g["params"]]
    opt = torch.optim.AdamW(groups, weight_decay=args.weight_decay)
    n_branch = sum(p.numel() for p in rv.parameters() if p.requires_grad)
    n_lora = sum(p.numel() for p in trainable_lora(bb.model).values())
    print(f"[stage1] trainable: branch {n_branch/1e6:.2f}M + LoRA {n_lora/1e6:.2f}M")

    step, best = 0, -1.0
    if os.path.exists(state_path):
        ck = torch.load(state_path, map_location="cpu", weights_only=False)
        restore(rv, bb, ck)
        opt.load_state_dict(ck["opt"])
        step, best = ck["step"], ck["best"]
        print(f"[stage1] resumed at step {step} (best {best:.4f})")
    if step >= max_steps:
        print("already finished"); return
    meta = dict(model_id=args.model_id, slot_ckpt=os.path.abspath(args.slot_ckpt), cfg=branch_cfg(args),
                lora_cfg=lora, args=vars(args))
    with open(os.path.join(args.out_dir, "args.json"), "w") as f:
        json.dump({**vars(args), "max_steps": max_steps, "branch_params": n_branch, "lora_params": n_lora}, f, indent=1)

    loader = DataLoader(stream, batch_size=args.batch_size, sampler=range(step * glob_bs, len(stream)),
                        num_workers=args.num_workers, drop_last=True, collate_fn=collate,
                        pin_memory=device.type == "cuda", prefetch_factor=4 if args.num_workers else None)
    it, budget = iter(loader), TimeBudget(args.time_limit_min)

    def save(tag, with_opt):
        extra = dict(opt=opt.state_dict()) if with_opt else {}
        save_ckpt(os.path.join(args.out_dir, tag), **checkpoint_payload(rv, bb, meta), step=done, best=best, **extra)

    def validate():
        nonlocal best
        for k, (vname, vl) in enumerate(vals):
            res = evaluate(rv, bb, st, vl, device, dtype, use_llm)
            log_jsonl(metrics, {"step": done, "val": vname, **res})
            lam = f" lambda {float(rv.lam()):.3f}" if rv.fuse else ""
            print(f"[val] step {done} {vname}: " + " ".join(f"{a} {b:.4f}" for a, b in res.items() if a != "n")
                  + f" (n={res['n']}){lam}", flush=True)
            key = "llm_acc" if use_llm else "aux_acc"
            if k == 0 and res.get(key, -1) > best:
                best = res[key]
                save("best.pt", False)
                with open(os.path.join(args.out_dir, "best.json"), "w") as f:
                    json.dump({"step": done, key: best, "val": vname}, f)
        set_modes(rv, bb, args)

    set_modes(rv, bb, args)
    agg, n_agg, t0, seen, done = defaultdict(float), 0, time.time(), 0, step
    for step in range(step, max_steps):
        for g in opt.param_groups:
            g["lr"] = lr_at(step, g["base"], args.warmup_steps, max_steps, schedule=args.lr_schedule)
        for _ in range(args.grad_accum):
            batch = next(it)
            h, aux = rv(batch, device, dtype, llm=use_llm)
            recs = example_logits(h, aux, batch, bb.model, st)
            l_llm, a_llm, _ = loss_and_acc(recs, "llm") if use_llm else (None, 0.0, 0)
            l_aux, a_aux, _ = loss_and_acc(recs, "aux")
            if args.objective == "llm":
                loss = l_llm
            elif args.objective == "aux":
                loss = l_aux
            else:
                loss = l_llm + args.aux_weight * l_aux if l_aux is not None else l_llm
            (loss / args.grad_accum).backward()
            agg["loss"] += float(loss) / args.grad_accum
            agg["llm_acc"] += a_llm / args.grad_accum
            agg["aux_acc"] += a_aux / args.grad_accum
            seen += len(batch["examples"])
        gn = torch.nn.utils.clip_grad_norm_(params, args.max_grad_norm)
        if not math.isfinite(float(gn)):
            raise RuntimeError(f"non-finite gradient norm at step {step}")
        opt.step()
        opt.zero_grad(set_to_none=True)
        done, n_agg = step + 1, n_agg + 1
        if done % args.log_every == 0 or done == max_steps:
            rec = {"step": done, **{k: v / n_agg for k, v in agg.items()}, "grad_norm": float(gn),
                   "lr": opt.param_groups[0]["lr"], "ex_per_s": seen / (time.time() - t0),
                   "mem_gb": torch.cuda.max_memory_allocated() / 2**30 if device.type == "cuda" else 0.0}
            if rv.fuse:
                rec["lambda"] = float(rv.lam())
            if rv.interleave:
                rec["tok_gain"] = float(rv.tok_gain)
            log_jsonl(metrics, rec)
            print(" ".join(f"{k} {v:.4g}" if isinstance(v, float) else f"{k} {v}" for k, v in rec.items()), flush=True)
            agg.clear(); n_agg = 0
        out_of_time = budget.exhausted()
        if vals and (done % args.val_every == 0 or done == max_steps):
            validate()
        if done % args.save_every == 0 or done == max_steps or out_of_time:
            save("state.pt", True)
        if out_of_time and done < max_steps:
            print(f"time budget reached at step {done}; resubmit the same command to continue")
            return
    save("final.pt", False)
    if not vals:
        save("best.pt", False)
    print("[stage1] finished: best.pt / final.pt / state.pt")


if __name__ == "__main__":
    main()
