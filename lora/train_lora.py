"""LoRA fine-tuning of a VLM (Qwen3-VL / InternVL3.5-HF) on PGM with PER-CANDIDATE scoring, single GPU, resumable.

  python lora/train_lora.py --model_id Qwen/Qwen3-VL-2B-Instruct --train_dir .../train_full/neutral \
      --val_dir .../val_full/neutral --out_dir runs/qwen_pc_neutral --train_samples 100000 --epochs 1

Each problem becomes 8 sequences (the matrix completed with candidate k, "Answer Yes or No"); score(k) = logit(Yes) - logit(No).
--loss ce8    softmax over the 8 scores + cross-entropy with the correct choice (the paper's loss)   [default]
--loss bce    each candidate independently: Yes for the answer, No for the others
--loss both   ce8 + bce
--batch_size counts PROBLEMS per micro-batch (8 sequences each).
"""
import argparse
import sys
import json
import math
import os
import time
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repo root

import numpy as np
import torch
from torch.utils.data import DataLoader

from pgmvlm.common import TimeBudget, count_params, log_jsonl, lr_at, lr_restart, save_ckpt, seed_everything
from pgmvlm.evalcore import evaluate_batches, summarize
from pgmvlm.models import (DTYPES, ScoreTokens, add_lora, candidate_losses, candidate_scores, load_adapter, load_base,
                         param_report, set_train_mode)
from pgmvlm.data import CandidateCollator, PGMStream, list_npz, make_order, subset_files


from pgmvlm.protocols import add_protocol_args, build_protocol
from pgmvlm.datasets import files_for, select_files, make_stream


def parse():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_id", default="Qwen/Qwen3-VL-2B-Instruct")
    ap.add_argument("--family", default=None, help="qwen3vl | internvl (auto-detected from config)")
    add_protocol_args(ap)
    ap.add_argument('--revision', default=None)
    ap.add_argument('--train_manifest', default=None)
    ap.add_argument('--val_manifest', default=None)
    ap.add_argument('--train_list', default=None, help='Exact ordered filename list (.txt, .tsv, or .gz)')
    ap.add_argument("--train_dir", default=None)
    ap.add_argument("--val_dir", default=None)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--file_cache_dir", default=None, help="where to cache the sorted file listing (1.2M files)")
    # data budget
    ap.add_argument("--train_samples", type=int, default=0, help="0 = all training problems; else a nested random subset")
    ap.add_argument("--train_fraction", type=float, default=0.0,
                    help="alternative to --train_samples: e.g. 0.2 = 20%% of the training problems found in --train_dir "
                         "(the paper's Table 5 uses 20%% vs 100%%). Subsets are nested: 20%% is inside 50%%")
    ap.add_argument("--epochs", type=float, default=1.0, help="passes over the (sub)set; ignored if --max_steps > 0")
    ap.add_argument("--max_steps", type=int, default=0)
    ap.add_argument("--shuffle_candidates", type=int, default=0, help="irrelevant for per-candidate scoring; kept for ablations")
    # optimisation
    ap.add_argument("--batch_size", type=int, default=2, help="PROBLEMS per forward pass (x8 sequences)")
    ap.add_argument("--grad_accum", type=int, default=8, help="effective batch = batch_size * grad_accum problems (16)")
    # defaults = the teammate recipe that converged on PGM (LFM2.5-VL 43.8% after 1 epoch on the same 240k set)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--lr_schedule", default="constant", choices=["constant", "cosine", "restart"],
                    help="constant: linear warmup then flat (teammate recipe); cosine: decay to 0.1 x lr at the end")
    ap.add_argument("--warmup_steps", type=int, default=20)
    ap.add_argument("--decay_start", type=int, default=0,
                    help="with --lr_schedule cosine: hold --lr until this step, then cosine-decay to 0.1 x lr by the last "
                         "step of this run (e.g. 45000 = start of epoch 4 at 15k steps/epoch)")
    ap.add_argument("--cycle_start", type=int, default=0,
                    help="--lr_schedule restart: step where the new cycle begins (normally the step the run resumes at)")
    ap.add_argument("--cycle_warmup", type=int, default=500, help="--lr_schedule restart: updates to rise from --cycle_from to --lr")
    ap.add_argument("--cycle_from", type=float, default=1e-5, help="--lr_schedule restart: LR at the start of the warmup")
    ap.add_argument("--min_lr", type=float, default=1e-5, help="--lr_schedule restart: LR at the last step of the run")
    ap.add_argument("--reset_schedule", action="store_true",
                    help="on resume, recompute the LR schedule for the new total steps (default: keep the schedule the run "
                         "was started with, so extending to more epochs continues at the LR where it stopped, no jump)")
    ap.add_argument("--weight_decay", type=float, default=0.01)
    ap.add_argument("--max_grad_norm", type=float, default=1.0)
    ap.add_argument("--loss", default=None, choices=["ce8", "bce", "both", "causal_ce"])
    # LoRA
    ap.add_argument("--lora_r", type=int, default=16)
    ap.add_argument("--lora_alpha", type=int, default=32)
    ap.add_argument("--lora_dropout", type=float, default=0.05)
    ap.add_argument("--lora_vision", action="store_true", help="also put LoRA on the vision tower")
    ap.add_argument("--lora_projector", action="store_true", help="also put LoRA on the merger / projector")
    # system
    ap.add_argument("--dtype", default="bf16", choices=list(DTYPES))
    ap.add_argument("--attn", default="sdpa")
    ap.add_argument("--grad_ckpt", action="store_true")
    ap.add_argument("--crop_to_patches", type=int, default=1, help="InternVL only: tile images (see README)")
    ap.add_argument("--panel_px", type=int, default=160, help="size each 160px panel is drawn at (bigger = more visual tokens)")
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--time_limit_min", type=float, default=0, help="stop cleanly after this many minutes (Slurm)")
    # logging / validation
    ap.add_argument("--log_every", type=int, default=10)
    ap.add_argument("--val_every", type=int, default=1000, help="steps")
    ap.add_argument("--val_samples", type=int, default=1000)
    ap.add_argument("--val_list", default=None,
                    help="file whose first tab-separated column lists the validation .npz names to use (e.g. the 1000 shared "
                         "validation problems of the 20%% dataset); overrides the random --val_samples subset")
    ap.add_argument("--val_batch_size", type=int, default=4, help="problems")
    ap.add_argument("--fit_eval_samples", type=int, default=0,
                    help="also measure accuracy on the first N TRAINING problems at every validation (overfit test: "
                         "with --train_samples N this should climb towards 100%%)")
    ap.add_argument("--save_every", type=int, default=500, help="steps")
    args = ap.parse_args()
    args.loss = args.loss or ('ce8' if args.protocol == 'candidate' else 'causal_ce')
    if (args.protocol == 'answer_letter') != (args.loss == 'causal_ce'):
        ap.error('answer_letter uses causal_ce; candidate uses ce8/bce/both')
    return args


def make_val_loader(args, processor, family):
    if (not args.val_dir and not args.val_manifest) or args.val_samples <= 0:
        return None
    available = files_for(args, 'val')
    files = select_files(available, args.val_list, 0 if args.val_list else args.val_samples, seed=1234)
    ds = make_stream(args, 'val', files)
    return DataLoader(ds, batch_size=args.val_batch_size, shuffle=False, num_workers=min(2, args.num_workers),
                      collate_fn=build_protocol(args, processor, family).collator)


def main():
    args = parse()
    os.makedirs(args.out_dir, exist_ok=True)
    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = DTYPES[args.dtype]
    metrics_path = os.path.join(args.out_dir, "metrics.jsonl")
    state_path = os.path.join(args.out_dir, "state.pt")
    last_dir, best_dir = os.path.join(args.out_dir, "last"), os.path.join(args.out_dir, "best")

    # ---------------- data
    cache = os.path.join(args.file_cache_dir, "train_files.txt") if args.file_cache_dir else None
    all_files = files_for(args, 'train')
    n_sub = args.train_samples or (int(round(args.train_fraction * len(all_files))) if args.train_fraction > 0 else 0)
    files = select_files(all_files, args.train_list, n_sub, seed=args.seed)
    print(f"train dir has {len(all_files)} problems; using {len(files)} ({100*len(files)/len(all_files):.1f}%)")
    glob_bs = args.batch_size * args.grad_accum
    max_steps = args.max_steps or max(1, int(math.ceil(args.epochs * len(files) / glob_bs)))
    order = make_order(len(files), max_steps * glob_bs, args.seed)
    print(f"train problems {len(files)} | global batch {glob_bs} | steps {max_steps} | epochs {max_steps * glob_bs / len(files):.2f}")

    # ---------------- model
    model, processor, family = load_base(args.model_id, args.dtype, args.attn, device, args.family,
                                         **({'revision': args.revision} if args.revision else {}))
    st = build_protocol(args, processor, family)
    print(f"family={family} | score tokens: {st.describe()}")
    start_step, best_val, resumed = 0, -1.0, False
    if os.path.exists(state_path) and os.path.isdir(last_dir):
        model = load_adapter(model, last_dir, trainable=True, grad_ckpt=args.grad_ckpt)
        resumed = True
    else:
        model = add_lora(model, family, args.lora_r, args.lora_alpha, args.lora_dropout, args.lora_vision,
                         args.lora_projector, args.grad_ckpt)
    for p in model.parameters():                       # LoRA weights in fp32 (stable), base stays in `dtype`
        if p.requires_grad:
            p.data = p.data.float()
    model.config.use_cache = False
    tr, tot = param_report(model)
    print(f"trainable {tr/1e6:.2f}M / {tot/1e6:.1f}M ({100*tr/tot:.2f}%)")
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=args.lr, betas=(0.9, 0.999), weight_decay=args.weight_decay)
    # The LR schedule length is fixed when a run starts and stored in state.pt. Extending a finished run to more epochs
    # keeps it: steps past its end stay at the final (minimum) LR instead of the cosine being redrawn over the new total.
    schedule_steps = max_steps
    if resumed:
        ckpt = torch.load(state_path, map_location="cpu", weights_only=False)
        opt.load_state_dict(ckpt["opt"])
        start_step, best_val = ckpt["step"], ckpt["best_val"]
        schedule_steps = (max_steps if (args.reset_schedule or args.decay_start or args.lr_schedule == "restart")
                          else ckpt.get("schedule_steps", max_steps))
        print(f"resumed from step {start_step} (best val so far {best_val:.4f}); LR schedule over {schedule_steps} steps"
              + (f" -> continuing at the final LR {lr_at(start_step, args.lr, args.warmup_steps, schedule_steps, schedule=args.lr_schedule, decay_start=args.decay_start):.3g} "
                 f"up to step {max_steps}" if max_steps > schedule_steps else ""))
    if args.lr_schedule == "cosine" and args.decay_start:
        print(f"LR schedule: {args.lr:.3g} held until step {args.decay_start}, then cosine decay to {0.1 * args.lr:.3g} "
              f"at step {max_steps} (now at step {start_step}: "
              f"{lr_at(start_step, args.lr, args.warmup_steps, schedule_steps, schedule='cosine', decay_start=args.decay_start):.3g})")
    def lr_for(step):
        if args.lr_schedule == "restart":
            return lr_restart(step, args.lr, args.cycle_start, args.cycle_warmup, max_steps, args.cycle_from, args.min_lr)
        return lr_at(step, args.lr, args.warmup_steps, schedule_steps, schedule=args.lr_schedule, decay_start=args.decay_start)

    if args.lr_schedule == "restart":
        if args.cycle_start <= 0:
            raise SystemExit("--lr_schedule restart needs --cycle_start (the step the new cycle begins)")
        print(f"LR schedule: warm restart at step {args.cycle_start}: {args.cycle_from:.3g} -> {args.lr:.3g} over "
              f"{args.cycle_warmup} updates, then cosine decay to {args.min_lr:.3g} at step {max_steps} "
              f"(now at step {start_step}: {lr_for(start_step):.3g})")
    if start_step >= max_steps:
        print("already finished"); return
    with open(os.path.join(args.out_dir, "args.json"), "w") as f:
        json.dump({**vars(args), "family": family, "trainable_params": tr, "total_params": tot,
                   "max_steps": max_steps, "schedule_steps": schedule_steps}, f, indent=1)

    stream = make_stream(args, 'train', files, order=order, seed=args.seed, shuffle_candidates=bool(args.shuffle_candidates))
    loader = DataLoader(stream, batch_size=args.batch_size, sampler=range(start_step * glob_bs, len(order)),
                        num_workers=args.num_workers, drop_last=True, pin_memory=device.type == "cuda",
                        collate_fn=st.collator,
                        prefetch_factor=4 if args.num_workers > 0 else None)
    val_loader = make_val_loader(args, processor, family)
    fit_loader = None
    if args.fit_eval_samples > 0:                      # training problems in their original candidate order
        fit_loader = DataLoader(make_stream(args, 'train', files[:args.fit_eval_samples], seed=args.seed),
                                batch_size=args.val_batch_size, shuffle=False, num_workers=min(2, args.num_workers),
                                collate_fn=st.collator)
    budget, it = TimeBudget(args.time_limit_min), iter(loader)

    def save_state(step):
        model.save_pretrained(last_dir)
        save_ckpt(state_path, opt=opt.state_dict(), step=step, best_val=best_val, schedule_steps=schedule_steps)

    def validate(step):
        nonlocal best_val
        if fit_loader is not None:
            fit = summarize(list(evaluate_batches(model, fit_loader, st, device, dtype)))
            log_jsonl(metrics_path, {"step": step, "train_fit_acc": fit["scores"]["acc"], "train_fit_n": fit["n"],
                                     "train_fit_pred_hist": fit["scores"]["pred_hist"]})
            print(f"[fit] step {step}: acc on {fit['n']} training problems {fit['scores']['acc']:.4f} "
                  f"| predicted positions {fit['scores']['pred_hist']}")
        if val_loader is None:
            return
        s = summarize(list(evaluate_batches(model, val_loader, st, device, dtype)))
        acc, hist = s["scores"]["acc"], s["scores"]["pred_hist"]
        log_jsonl(metrics_path, {"step": step, "val_acc": acc, "val_n": s["n"], "val_pred_hist": hist,
                                 **({'val_loss': s['scores']['loss'], 'val_answer_loss': s['scores']['answer_loss']}
                                    if args.protocol == 'answer_letter' else {
                                        'val_answer_says_yes': s['scores']['answer_says_yes'],
                                        'val_distractor_says_yes': s['scores']['distractor_says_yes']})})
        # one position getting nearly every prediction = the model ignores the images and outputs the label prior
        collapsed = "  <- COLLAPSED to one answer" if max(hist) > 0.8 * s["n"] else ""
        print(f"[val] step {step}: acc {acc:.4f} on {s['n']} problems (chance 0.125) "
              f"| predicted positions {hist}{collapsed}")
        if acc > best_val:
            best_val = acc
            model.save_pretrained(best_dir)
            with open(os.path.join(args.out_dir, "best.json"), "w") as f:
                json.dump({"step": step, "val_acc": acc}, f)

    set_train_mode(model)
    t0, seen, agg, n_agg = time.time(), 0, defaultdict(float), 0
    step = start_step
    for step in range(start_step, max_steps):
        for g in opt.param_groups:
            g["lr"] = lr_for(step)
        for _ in range(args.grad_accum):
            batch = next(it)
            targets = batch["targets"].to(device)
            metrics = st.training(model, batch, device, dtype, args.loss)
            loss = metrics['loss']
            if not torch.isfinite(loss):
                raise RuntimeError(f'Nonfinite training loss at step {step}')
            (loss / args.grad_accum).backward()
            for k, v in metrics.items():
                agg[k] += float(v) / args.grad_accum
            seen += targets.numel()
        gn = torch.nn.utils.clip_grad_norm_(params, args.max_grad_norm)
        opt.step()
        opt.zero_grad(set_to_none=True)
        if not math.isfinite(float(gn)):
            raise RuntimeError(f"non-finite gradient norm at step {step}")
        done = step + 1
        n_agg += 1
        if done % args.log_every == 0 or done == max_steps:
            rec = {"step": done, **{k: v / n_agg for k, v in agg.items()}, "grad_norm": float(gn),
                   "lr": opt.param_groups[0]["lr"], "samples_per_s": seen / (time.time() - t0),
                   "mem_gb": torch.cuda.max_memory_allocated() / 2**30 if device.type == "cuda" else 0.0}
            log_jsonl(metrics_path, rec)
            print(" ".join(f"{k} {v:.4g}" if isinstance(v, float) else f"{k} {v}" for k, v in rec.items()), flush=True)
            agg.clear()
            n_agg = 0
        out_of_time = budget.exhausted()
        if done % args.val_every == 0 or done == max_steps:
            validate(done); set_train_mode(model)
        if done % args.save_every == 0 or done == max_steps or out_of_time:
            save_state(done)
        if out_of_time and done < max_steps:
            print(f"time budget reached at step {done}; resubmit the same command to continue")
            return
    model.save_pretrained(os.path.join(args.out_dir, "final"))
    print("finished; adapters: last/ best/ final/")


if __name__ == "__main__":
    main()
