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
import random
import time
from collections import defaultdict

import torch
from torch.utils.data import DataLoader

from backbones import DTYPES
from common import TimeBudget, log_jsonl, lr_at, save_ckpt, seed_everything
from data import Collator, FixedSet, MixtureStream, build_sources, parse_mix, subset
from factory import build, checkpoint_payload, load_peft_adapter, restore, trainable_lora
from relvlm import example_logits, loss_and_acc


def parse():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_id", default="Qwen/Qwen3-VL-2B-Instruct")
    ap.add_argument("--slot_ckpt", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--train", action="append", required=True, help="name=type:path[:weight]; the FIRST is the primary")
    ap.add_argument("--val", action="append", default=[], help="name=type:path; best.pt follows --best_on")
    ap.add_argument("--best_on", default="first", choices=["first", "family"],
                    help="first: the first validation set. family: mean accuracy over task families (the name before "
                         "'_': art_sd and art_rmts are one family), for mixtures")
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
    ap.add_argument("--rel_tokens", type=int, default=1, help="M relation tokens per sequence (interleave, --rel_mode pool)")
    ap.add_argument("--rel_mode", default="pool", choices=["pool", "panel", "slot"],
                    help="interleave: tokens from the pooled r, one token per panel, or one token per slot")
    ap.add_argument("--no_relational", action="store_true", help="same-size control: ordinary attention instead of the relational bottleneck")
    ap.add_argument("--fuse_lambda", type=float, default=0.9, help="initial lambda (weight of the native visual token)")
    ap.add_argument("--fixed_lambda", action="store_true", help="do not learn lambda")
    ap.add_argument("--no_slots", action="store_true",
                    help="control: a fixed grid of ViT locations replaces the slots; --slot_ckpt still gives the feature stats")
    ap.add_argument("--layers", type=int, default=24)
    ap.add_argument("--heads", type=int, default=8)
    ap.add_argument("--head_dim", type=int, default=32)
    ap.add_argument("--mlp_dim", type=int, default=512)
    ap.add_argument("--objective", default="llm", choices=["llm", "aux", "both"])
    ap.add_argument("--aux_weight", type=float, default=0.5)
    ap.add_argument("--loss_norm", default="none", choices=["none", "chance"],
                    help="chance: divide each example's loss by its chance-level loss (use for mixtures)")
    ap.add_argument("--init_from", default=None, help="Stage-1 checkpoint to start the branch from (Stage 2)")
    ap.add_argument("--freeze_abstractor", action="store_true",
                    help="Stage 1 alignment: keep the (warm-started) Abstractor and aux head fixed; only the connector "
                         "into the LLM trains")
    ap.add_argument("--branch_dropout", type=float, default=0.0,
                    help="Stage 2: per micro-batch, with this probability the LLM gets nothing from the branch (fusion: "
                         "native visual tokens; interleave: zero relation token), so LoRA must also learn from the image "
                         "instead of only copying the branch. Validation has the branch on, except at 1.0: the "
                         "branch-always-off control, validated with the branch off too")
    ap.add_argument("--finetune_slots", action="store_true", help="also train slot attention on the task loss (decoder excluded)")
    ap.add_argument("--slot_lr", type=float, default=3e-5, help="slot attention rate with --finetune_slots")
    # LoRA (Stage 2)
    ap.add_argument("--lora_r", type=int, default=0)
    ap.add_argument("--lora_alpha", type=int, default=32)
    ap.add_argument("--lora_dropout", type=float, default=0.05)
    ap.add_argument("--lora_lr", type=float, default=1e-4)
    ap.add_argument("--lora_from", default=None,
                    help="start LoRA from a trained PEFT adapter dir (e.g. a vlm_lora_pc run's best/), same r and alpha; "
                         "ignored on resume (state.pt has this run's own LoRA)")
    ap.add_argument("--freeze_lora_steps", type=int, default=0,
                    help="keep the LoRA rate at 0 for the first N steps (the connector adapts to a trained LoRA first)")
    # optimisation
    ap.add_argument("--batch_size", type=int, default=2)
    ap.add_argument("--grad_accum", type=int, default=8, help="global batch = batch_size x grad_accum examples (16)")
    ap.add_argument("--lr", type=float, default=1e-4, help="Abstractor")
    ap.add_argument("--proj_lr", type=float, default=3e-4, help="projectors, lambda, aux head")
    ap.add_argument("--lr_schedule", default="cosine", choices=["constant", "cosine"])
    ap.add_argument("--warmup_steps", type=int, default=500)
    ap.add_argument("--decay_start", type=int, default=0,
                    help="with --lr_schedule cosine: hold the LR until this step, then decay (the LoRA baseline's last epoch)")
    ap.add_argument("--rewarm_from", type=int, default=0,
                    help="continuing a finished run past its old end: from this step, ramp every group's LR linearly "
                         "from 10%% of its schedule back up to 100%% over --rewarm_steps (0 = off)")
    ap.add_argument("--rewarm_steps", type=int, default=2000)
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
                learn_lambda=not args.fixed_lambda, no_slots=args.no_slots, rel_mode=args.rel_mode,
                finetune_slots=args.finetune_slots,
                abstractor_cfg=dict(layers=args.layers, heads=args.heads, head_dim=args.head_dim, mlp_dim=args.mlp_dim,
                                    relational=not args.no_relational))


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


def selection_score(results, key, mode):
    """results: [(val name, metrics)]. The score best.pt follows; see --best_on."""
    if mode == "first":
        return results[0][1].get(key, -1)
    families = defaultdict(list)
    for name, res in results:
        if name != "fit" and key in res:
            families[name.split("_")[0]].append(res[key])
    return sum(sum(v) / len(v) for v in families.values()) / max(1, len(families))


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
    if os.environ.get("SLURM_GPUS_ON_NODE") and device.type != "cuda":
        raise SystemExit("Slurm allocated a GPU but torch cannot see it: stopping instead of training on the CPU")
    dtype = DTYPES[args.dtype]
    metrics = os.path.join(args.out_dir, "metrics.jsonl")
    state_path = os.path.join(args.out_dir, "state.pt")
    use_llm = args.objective != "aux"

    lora = dict(r=args.lora_r, alpha=args.lora_alpha, dropout=args.lora_dropout) if args.lora_r else None
    rv, bb, st = build(args.model_id, args.slot_ckpt, branch_cfg(args), device, args.dtype, args.attn, lora)
    if args.grad_ckpt:
        bb.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    if args.init_from:
        ck0 = torch.load(args.init_from, map_location="cpu", weights_only=False)
        if ck0.get("slot_ae"):        # slot attention fine-tuned in that run: use it here too, and save it with
            rv.slot_ae.load_state_dict(ck0["slot_ae"])     # this run's checkpoints (resume, later warm starts).
            rv.save_slots = True      # On resume, restore() below then loads this run's own saved copy.
            print(f"[stage1] slot attention loaded from {args.init_from} (fine-tuned there; "
                  f"{'trained further' if args.finetune_slots else 'frozen'} here)")
        del ck0
    if args.init_from and not os.path.exists(state_path):
        ck = torch.load(args.init_from, map_location="cpu", weights_only=False)
        missing, unexpected = rv.load_state_dict(ck["branch"], strict=False)
        print(f"[stage1] branch initialised from {args.init_from} | new modules: "
              f"{sorted({k.split('.')[0] for k in missing}) or 'none'} | not used here: "
              f"{sorted({k.split('.')[0] for k in unexpected}) or 'none'}")
    if args.lora_from and not os.path.exists(state_path):
        if not lora:
            raise SystemExit("--lora_from needs --lora_r (and --lora_alpha) matching the adapter")
        n = load_peft_adapter(bb.model, args.lora_from, lora)
        print(f"[stage1] LoRA initialised from {args.lora_from} ({n} tensors, r={args.lora_r} alpha={args.lora_alpha:g})")
    if args.freeze_abstractor:        # alignment: only the connector (tok_proj + gain, or fuse_proj + lambda) trains
        for p in list(rv.abstractor.parameters()) + list(rv.aux_head.parameters()):
            p.requires_grad_(False)
        print("[stage1] Abstractor and aux head frozen")

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
    collate = Collator(bb.processor, bb.processor_kwargs(), **rv.collate_kwargs())

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
    ck = torch.load(state_path, map_location="cpu", weights_only=False) if os.path.exists(state_path) else None
    slot_params = [p for p in rv.slot_ae.parameters() if p.requires_grad]
    slot_group = {"params": slot_params, "lr": args.slot_lr, "base": args.slot_lr}
    # a state saved before fine-tuning started has no slot group: load it as saved, then add the group
    add_slots_after_load = bool(slot_params) and ck is not None and len(ck["opt"]["param_groups"]) == len(groups)
    if slot_params and not add_slots_after_load:
        groups.append(slot_group)
    params = [p for g in groups for p in g["params"]] + (slot_params if add_slots_after_load else [])
    lora_ids = {id(p) for p in trainable_lora(bb.model).values()} if lora else set()
    opt = torch.optim.AdamW(groups, weight_decay=args.weight_decay)
    n_branch = sum(p.numel() for p in rv.parameters() if p.requires_grad)
    n_lora = sum(p.numel() for p in trainable_lora(bb.model).values())
    n_slots = sum(p.numel() for p in slot_params)
    print(f"[stage1] trainable: branch {n_branch/1e6:.2f}M + LoRA {n_lora/1e6:.2f}M + slot attention {n_slots/1e6:.2f}M")

    step, best = 0, -1.0
    if ck is not None:
        restore(rv, bb, ck)
        opt.load_state_dict(ck["opt"])
        step, best = ck["step"], ck["best"]
        print(f"[stage1] resumed at step {step} (best {best:.4f})")
    if add_slots_after_load:
        opt.add_param_group(slot_group)
        print("[stage1] slot attention group added to the resumed optimiser")
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
        key = "llm_acc" if use_llm else "aux_acc"
        results = []
        for vname, vl in vals:
            rv.branch_off = args.branch_dropout >= 1.0          # the branch-always-off control is scored without it
            res = evaluate(rv, bb, st, vl, device, dtype, use_llm)
            rv.branch_off = False
            log_jsonl(metrics, {"step": done, "val": vname, **res})
            lam = f" lambda {float(rv.lam()):.3f}" if rv.fuse else ""
            print(f"[val] step {done} {vname}: " + " ".join(f"{a} {b:.4f}" for a, b in res.items() if a != "n")
                  + f" (n={res['n']}){lam}", flush=True)
            results.append((vname, res))
        score = selection_score(results, key, args.best_on)
        if args.best_on != "first":
            log_jsonl(metrics, {"step": done, "val": "selection", key: score})
            print(f"[val] step {done} selection ({args.best_on}): {score:.4f}", flush=True)
        if score > best:
            best = score
            save("best.pt", False)
            with open(os.path.join(args.out_dir, "best.json"), "w") as f:
                json.dump({"step": done, key: best, "val": vals[0][0] if args.best_on == "first" else args.best_on}, f)
        set_modes(rv, bb, args)

    set_modes(rv, bb, args)
    agg, n_agg, t0, seen, done = defaultdict(float), 0, time.time(), 0, step
    drop_rng, split = random.Random(args.seed * 7919 + step), defaultdict(float)    # branch dropout draws, on/off acc
    if args.branch_dropout >= 1.0:
        print("[stage1] control: branch always off, in training and validation")
    elif args.branch_dropout:
        print(f"[stage1] branch dropout {args.branch_dropout:g} per micro-batch (validation keeps the branch on)")
    if args.freeze_lora_steps:
        print(f"[stage1] LoRA frozen (rate 0) for the first {args.freeze_lora_steps} steps")
    for step in range(step, max_steps):
        ramp = 1.0
        if args.rewarm_from and args.rewarm_from <= step < args.rewarm_from + args.rewarm_steps:
            ramp = 0.1 + 0.9 * (step - args.rewarm_from + 1) / max(1, args.rewarm_steps)
        for g in opt.param_groups:
            g["lr"] = ramp * lr_at(step, g["base"], args.warmup_steps, max_steps, schedule=args.lr_schedule,
                                   decay_start=args.decay_start)
            if step < args.freeze_lora_steps and g["params"] and id(g["params"][0]) in lora_ids:
                g["lr"] = 0.0
        for _ in range(args.grad_accum):
            batch = next(it)
            off = bool(args.branch_dropout) and drop_rng.random() < args.branch_dropout
            rv.branch_off = off
            h, aux = rv(batch, device, dtype, llm=use_llm)
            recs = example_logits(h, aux, batch, bb.model, st)
            cn = args.loss_norm == "chance"
            l_llm, a_llm, _ = loss_and_acc(recs, "llm", cn) if use_llm else (None, 0.0, 0)
            l_aux, a_aux, _ = loss_and_acc(recs, "aux", cn)
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
            if args.branch_dropout:
                k = "off" if off else "on"
                split[f"n_{k}"] += 1
                split[f"acc_{k}"] += a_llm
        rv.branch_off = False
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
            if args.branch_dropout:                    # train accuracy with the branch on / cut, since the last log
                for k in ("on", "off"):
                    if split[f"n_{k}"]:
                        rec[f"llm_acc_{k}"] = split[f"acc_{k}"] / split[f"n_{k}"]
                split.clear()
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
