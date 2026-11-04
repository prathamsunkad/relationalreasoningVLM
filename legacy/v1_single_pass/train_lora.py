"""LoRA fine-tuning of a VLM (Qwen3-VL / InternVL3.5-HF) on PGM, single GPU, resumable across Slurm jobs.

  python train_lora.py --model_id Qwen/Qwen3-VL-2B-Instruct --train_dir .../train_full/neutral \
      --val_dir .../val_full/neutral --out_dir runs/qwen_neutral_100k --train_samples 100000 --epochs 1

Loss: cross-entropy on the answer tokens only ("Answer: k<|im_end|>"); the prompt and images are not supervised.
--loss lm       standard SFT (all 5 answer tokens; 4 of them are trivial format tokens, so the answer digit gets 1/5 of the loss)
--loss digit8   8-way softmax over the digit-token logits only (the paper's candidate-scoring CE)  [default]
--loss both     lm + digit8
"""
import argparse
import json
import math
import os
import time
from collections import defaultdict

import numpy as np
import torch
from torch.utils.data import DataLoader

from common import TimeBudget, count_params, log_jsonl, lr_at, save_ckpt, seed_everything
from evalcore import evaluate_batches, summarize
from model_utils import (DTYPES, AnswerTokens, add_lora, load_adapter, load_base, lm_and_digit_losses, param_report,
                         teacher_forced_logits)
from pgm_vlm_data import PGMStream, VLMCollator, list_npz, make_order, shard_files, subset_files


def parse():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_id", default="Qwen/Qwen3-VL-2B-Instruct")
    ap.add_argument("--family", default=None, help="qwen3vl | internvl (auto-detected from config)")
    ap.add_argument("--train_dir", required=True)
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
    ap.add_argument("--shuffle_candidates", type=int, default=1)
    # optimisation
    ap.add_argument("--batch_size", type=int, default=8, help="micro-batch per forward pass")
    ap.add_argument("--grad_accum", type=int, default=4, help="effective batch = batch_size * grad_accum")
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--warmup_steps", type=int, default=50)
    ap.add_argument("--weight_decay", type=float, default=0.0)
    ap.add_argument("--max_grad_norm", type=float, default=1.0)
    ap.add_argument("--loss", default="digit8", choices=["lm", "digit8", "both"])
    # LoRA
    ap.add_argument("--lora_r", type=int, default=64)
    ap.add_argument("--lora_alpha", type=int, default=128)
    ap.add_argument("--lora_dropout", type=float, default=0.05)
    ap.add_argument("--lora_vision", action="store_true", help="also put LoRA on the vision tower")
    ap.add_argument("--lora_projector", action="store_true", help="also put LoRA on the merger / projector")
    # system
    ap.add_argument("--dtype", default="bf16", choices=list(DTYPES))
    ap.add_argument("--attn", default="sdpa")
    ap.add_argument("--grad_ckpt", action="store_true")
    ap.add_argument("--crop_to_patches", type=int, default=1, help="InternVL only: tile non-square images (see README)")
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--time_limit_min", type=float, default=0, help="stop cleanly after this many minutes (Slurm)")
    # logging / validation
    ap.add_argument("--log_every", type=int, default=10)
    ap.add_argument("--val_every", type=int, default=500, help="steps")
    ap.add_argument("--val_samples", type=int, default=1000)
    ap.add_argument("--val_batch_size", type=int, default=16)
    ap.add_argument("--fit_eval_samples", type=int, default=0,
                    help="also measure accuracy on the first N TRAINING problems at every validation (overfit test: "
                         "with --train_samples N this should climb towards 100%%; if it stays at 12.5%% the model cannot "
                         "tell the inputs apart)")
    ap.add_argument("--save_every", type=int, default=250, help="steps")
    return ap.parse_args()


def make_val_loader(args, processor, family):
    if not args.val_dir or args.val_samples <= 0:
        return None
    cache = os.path.join(args.file_cache_dir, "val_files.txt") if args.file_cache_dir else None
    files = subset_files(list_npz(args.val_dir, cache), args.val_samples, seed=1234)
    ds = PGMStream(args.val_dir, files, seed=args.seed, shuffle_candidates=False)
    return DataLoader(ds, batch_size=args.val_batch_size, shuffle=False, num_workers=min(2, args.num_workers),
                      collate_fn=VLMCollator(processor, family, args.crop_to_patches))


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
    all_files = list_npz(args.train_dir, cache)
    n_sub = args.train_samples or (int(round(args.train_fraction * len(all_files))) if args.train_fraction > 0 else 0)
    files = subset_files(all_files, n_sub, seed=args.seed)
    print(f"train dir has {len(all_files)} problems; using {len(files)} ({100*len(files)/len(all_files):.1f}%)")
    glob_bs = args.batch_size * args.grad_accum
    max_steps = args.max_steps or max(1, int(math.ceil(args.epochs * len(files) / glob_bs)))
    order = make_order(len(files), max_steps * glob_bs, args.seed)
    print(f"train problems {len(files)} | global batch {glob_bs} | steps {max_steps} | epochs {max_steps * glob_bs / len(files):.2f}")

    # ---------------- model
    model, processor, family = load_base(args.model_id, args.dtype, args.attn, device, args.family)
    answer = AnswerTokens(processor.tokenizer)
    print(f"family={family} | {answer.describe()}")
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
    if resumed:
        st = torch.load(state_path, map_location="cpu", weights_only=False)
        opt.load_state_dict(st["opt"])
        start_step, best_val = st["step"], st["best_val"]
        print(f"resumed from step {start_step} (best val so far {best_val:.4f})")
    if start_step >= max_steps:
        print("already finished"); return
    with open(os.path.join(args.out_dir, "args.json"), "w") as f:
        json.dump({**vars(args), "family": family, "trainable_params": tr, "total_params": tot}, f, indent=1)

    stream = PGMStream(args.train_dir, files, order=order, seed=args.seed, shuffle_candidates=bool(args.shuffle_candidates))
    loader = DataLoader(stream, batch_size=args.batch_size, sampler=range(start_step * glob_bs, len(order)),
                        num_workers=args.num_workers, drop_last=True, pin_memory=device.type == "cuda",
                        collate_fn=VLMCollator(processor, family, args.crop_to_patches),
                        prefetch_factor=4 if args.num_workers > 0 else None)
    val_loader = make_val_loader(args, processor, family)
    fit_loader = None
    if args.fit_eval_samples > 0:                      # training problems in their original candidate order
        fit_loader = DataLoader(PGMStream(args.train_dir, files[:args.fit_eval_samples], seed=args.seed),
                                batch_size=args.val_batch_size, shuffle=False, num_workers=min(2, args.num_workers),
                                collate_fn=VLMCollator(processor, family, args.crop_to_patches))
    budget, it = TimeBudget(args.time_limit_min), iter(loader)

    def save_state(step):
        model.save_pretrained(last_dir)
        save_ckpt(state_path, opt=opt.state_dict(), step=step, best_val=best_val)

    def validate(step):
        nonlocal best_val
        if fit_loader is not None:
            fit = summarize(list(evaluate_batches(model, processor, fit_loader, answer, device, dtype, modes=("logits",))))
            log_jsonl(metrics_path, {"step": step, "train_fit_acc_logits": fit["logits"]["acc"], "train_fit_n": fit["n"],
                                     "train_fit_pred_hist": fit["logits"]["pred_hist"]})
            print(f"[fit] step {step}: acc on {fit['n']} training problems {fit['logits']['acc']:.4f} "
                  f"| predicted positions {fit['logits']['pred_hist']}")
        if val_loader is None:
            return
        s = summarize(list(evaluate_batches(model, processor, val_loader, answer, device, dtype, modes=("logits",))))
        acc, hist = s["logits"]["acc"], s["logits"]["pred_hist"]
        log_jsonl(metrics_path, {"step": step, "val_acc_logits": acc, "val_n": s["n"], "val_pred_hist": hist})
        # one position getting nearly every prediction = the model ignores the images and outputs the label prior
        collapsed = "  <- COLLAPSED to one answer" if max(hist) > 0.8 * s["n"] else ""
        print(f"[val] step {step}: restricted-readout acc {acc:.4f} on {s['n']} problems (chance 0.125) "
              f"| predicted positions {hist}{collapsed}")
        if acc > best_val:
            best_val = acc
            model.save_pretrained(best_dir)
            with open(os.path.join(args.out_dir, "best.json"), "w") as f:
                json.dump({"step": step, "val_acc_logits": acc}, f)

    model.train()
    t0, seen, agg, n_agg = time.time(), 0, defaultdict(float), 0
    step = start_step
    for step in range(start_step, max_steps):
        for g in opt.param_groups:
            g["lr"] = lr_at(step, args.lr, args.warmup_steps, max_steps)
        for _ in range(args.grad_accum):
            batch = next(it)
            logits, resp, targets = teacher_forced_logits(model, batch, answer, device, dtype)
            lm, d8, acc = lm_and_digit_losses(logits, resp, targets, answer)
            loss = {"lm": lm, "digit8": d8, "both": lm + d8}[args.loss]
            (loss / args.grad_accum).backward()
            for k, v in (("loss", loss), ("lm", lm), ("digit8", d8), ("train_acc", acc)):
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
            validate(done); model.train()
        if done % args.save_every == 0 or done == max_steps or out_of_time:
            save_state(done)
        if out_of_time and done < max_steps:
            print(f"time budget reached at step {done}; resubmit the same command to continue")
            return
    model.save_pretrained(os.path.join(args.out_dir, "final"))
    print("finished; adapters: last/ best/ final/")


if __name__ == "__main__":
    main()
