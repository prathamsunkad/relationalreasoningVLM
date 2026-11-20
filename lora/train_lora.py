"""Shared resumable VLM adaptation, with candidate scoring as the default protocol.

Candidate mode scores eight completed matrices using Yes-minus-No logits and
ce8/bce/both losses. Answer-letter mode uses one contact sheet, native chat
encoding and masked causal CE. The same loop supports language LoRA, frozen-base
extension training and LoRA plus extensions; torchrun is optional.

See docs/VLM_INTEGRATION.md and configs/vlm/ for executable recipes.
"""
import argparse
from contextlib import ExitStack
import sys
import json
import math
import os
import time
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repo root

import numpy as np
import torch
from torch.utils.data import DataLoader

from pgmvlm.common import TimeBudget, log_jsonl, lr_at, lr_restart, save_ckpt, seed_everything
from pgmvlm.evalcore import evaluate_batches, summarize
from pgmvlm.models import DTYPES, add_lora, load_adapter, load_base, param_report


from pgmvlm.config import parse_configured
from pgmvlm.protocols import add_protocol_args, build_protocol
from pgmvlm.datasets import files_for, select_files, make_stream, training_order
from pgmvlm.extensions import add_extension_args, attach_extension, extension_module, save_model, check_checkpoint_protocol
from pgmvlm.extensions import set_train_mode as set_shared_train_mode
from pgmvlm.runtime import run_contract, capture_rng, restore_rng, model_digest, preserved_rng
from pgmvlm.diagnostics import GradientRecorder
from pgmvlm.distributed import DistributedContext, rank_positions
from pgmvlm.checkpoints import RunLock, commit, pointer_for, resume_checkpoint, resolve_checkpoint, rewind_logs, checkpoint_reader
from pgmvlm.shutdown import StopController


def parse():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_id", default="Qwen/Qwen3-VL-2B-Instruct")
    ap.add_argument("--family", default=None, help="Auto-detected model family; normally leave unset")
    add_protocol_args(ap)
    add_extension_args(ap, training=True)
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
    ap.add_argument("--batch_size", type=int, default=2, help="Problems per microbatch; candidate mode creates 8 sequences per problem")
    ap.add_argument('--global_batch', type=int, default=None, help='Fix global batch across torchrun ranks; overrides grad_accum')
    ap.add_argument("--grad_accum", type=int, default=8, help="Accumulation per rank; global batch = batch_size * grad_accum * world_size")
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
    ap.add_argument('--data_order', choices=['position', 'python_epoch', 'nested_first'], default='position')
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--time_limit_min", type=float, default=0, help="stop cleanly after this many minutes (Slurm)")
    # logging / validation
    ap.add_argument('--diagnostics_every', type=int, default=0, help='Record activation and parameter gradients every N updates')
    ap.add_argument('--select_by', choices=['accuracy', 'accuracy_loss', 'loss'], default=None)
    ap.add_argument('--early_stop_patience', type=int, default=0, help='Completed validation checks without loss improvement; 0 disables')
    ap.add_argument('--early_stop_min_delta', type=float, default=0.)
    ap.add_argument('--deterministic', action='store_true')
    ap.add_argument('--checkpoint_reserve_seconds', type=float, default=120, help='Time reserved before the Slurm deadline for a committed save')
    ap.add_argument('--check_lustre_quota', action='store_true', help='Check personal and filesystem quota before checkpoints')
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
    args = parse_configured(ap)
    args.select_by = args.select_by or ('accuracy' if args.protocol == 'candidate' else 'accuracy_loss')
    args.loss = args.loss or ('ce8' if args.protocol == 'candidate' else 'causal_ce')
    if (args.protocol == 'answer_letter') != (args.loss == 'causal_ce'):
        ap.error('answer_letter uses causal_ce; candidate uses ce8/bce/both')
    if (args.extension == 'none') != (args.train_mode == 'lora'):
        ap.error('Use --train_mode module or lora_module when selecting an extension')
    if args.readout != 'llm' and args.train_mode != 'module':
        ap.error('Classifier diagnostics keep the entire VLM frozen: use --train_mode module')
    if args.extension != 'none' and args.grad_ckpt:
        ap.error('Gradient checkpointing is not audited with the stateful extension hooks')
    if args.extension != 'none' and args.protocol != 'answer_letter':
        ap.error('LFM extension hooks require the answer_letter protocol')
    if args.protocol == 'candidate' and (args.select_by != 'accuracy' or args.early_stop_patience):
        ap.error('Validation-loss selection/stopping currently requires answer_letter')
    if min(args.batch_size, args.grad_accum, args.val_every, args.save_every, args.log_every) < 1:
        ap.error('Batch sizes and recording intervals must be positive')
    if args.lr_schedule == 'restart' and args.cycle_start <= 0:
        ap.error('--lr_schedule restart needs --cycle_start (the step the new cycle begins)')
    if args.checkpoint_reserve_seconds < 0:
        ap.error('Checkpoint reserve must be nonnegative')
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
    ctx = DistributedContext()
    try:
        with RunLock(args.out_dir, ctx.primary), StopController(args.time_limit_min, args.checkpoint_reserve_seconds) as stop, ExitStack() as imports:
            train(args, ctx, stop, imports)
    finally:
        ctx.close()


def train(args, ctx, stop, imports):
    args.world_size = ctx.world
    if args.global_batch:
        if args.global_batch % (args.batch_size * ctx.world):
            raise ValueError('global_batch must be divisible by microbatch times world size')
        args.grad_accum = args.global_batch // (args.batch_size * ctx.world)
    os.makedirs(args.out_dir, exist_ok=True)
    seed_everything(args.seed)
    if args.deterministic:
        os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
        torch.use_deterministic_algorithms(True)
    device = ctx.device
    dtype = DTYPES[args.dtype]
    metrics_path = os.path.join(args.out_dir, "metrics.jsonl")
    if ctx.primary:
        resume_checkpoint(args.out_dir)  # validate/recover and repair convenience paths under the run lock
    ctx.barrier()
    last_dir = str(resolve_checkpoint(os.path.join(args.out_dir, 'last')))
    state_path = os.path.join(last_dir, 'state.pt') if os.path.exists(os.path.join(last_dir, 'state.pt')) else os.path.join(args.out_dir, 'state.pt')
    best_dir = os.path.join(args.out_dir, 'best')

    # ---------------- data
    all_files = files_for(args, 'train')
    n_sub = args.train_samples or (int(round(args.train_fraction * len(all_files))) if args.train_fraction > 0 else 0)
    files = select_files(all_files, args.train_list, n_sub, seed=args.seed)
    print(f"train dir has {len(all_files)} problems; using {len(files)} ({100*len(files)/len(all_files):.1f}%)")
    glob_bs = args.batch_size * args.grad_accum * ctx.world
    max_steps = args.max_steps or max(1, int(math.ceil(args.epochs * len(files) / glob_bs)))
    order = training_order(len(files), max_steps * glob_bs, args.seed, args.data_order)
    print(f"train problems {len(files)} | global batch {glob_bs} | steps {max_steps} | epochs {max_steps * glob_bs / len(files):.2f}")

    validation_files = []
    if (args.val_dir or args.val_manifest) and args.val_samples > 0:
        validation_files = select_files(files_for(args, 'val'), args.val_list,
                                         0 if args.val_list else args.val_samples, seed=1234)
    if ctx.primary:
        run_contract(args, files, validation_files)
    ctx.barrier()

    # ---------------- model
    model, processor, family = load_base(args.model_id, args.dtype, args.attn, device, args.family,
                                         **({'revision': args.revision} if args.revision else {}))
    st = build_protocol(args, processor, family)
    print(f"family={family} | score tokens: {st.describe()}")
    start_step, best_val, resumed = 0, -1.0, False
    best_loss, plateau_loss, stale_checks, stop_training = float('inf'), float('inf'), 0, False
    if os.path.exists(state_path) and os.path.isdir(last_dir):
        check_checkpoint_protocol(last_dir, args)
        if os.path.isfile(os.path.join(last_dir, 'adapter_config.json')):
            model = load_adapter(model, last_dir, trainable=args.train_mode != 'module', grad_ckpt=args.grad_ckpt)
        else:
            model.requires_grad_(False)
        resumed = True
    elif args.init_adapter:
        args.init_adapter = str(imports.enter_context(checkpoint_reader(args.init_adapter)))
        if os.path.exists(os.path.join(args.init_adapter, 'extension.json')) and (not args.init_extension or resolve_checkpoint(args.init_extension) != Path(args.init_adapter)):
            raise ValueError('A joint checkpoint needs both --init_adapter and --init_extension pointing to it')
        # A joint converted checkpoint can initialize both components explicitly.
        check_checkpoint_protocol(args.init_adapter, args, allow_new_extension=True)
        model = load_adapter(model, args.init_adapter, trainable=args.train_mode != 'module', grad_ckpt=args.grad_ckpt)
    elif args.train_mode == 'module':
        model.requires_grad_(False)
    else:
        model = add_lora(model, family, args.lora_r, args.lora_alpha, args.lora_dropout, args.lora_vision,
                         args.lora_projector, args.grad_ckpt)
    if args.init_extension and not resumed:
        args.init_extension = str(imports.enter_context(checkpoint_reader(args.init_extension)))
        if os.path.exists(os.path.join(args.init_extension, "adapter_config.json")) and not args.init_adapter:
            raise ValueError("Joint checkpoint initialization also requires --init_adapter to preserve inherited LoRA")
        check_checkpoint_protocol(args.init_extension, args)
    controller = attach_extension(model, processor, args,
                                  last_dir if resumed and args.extension != 'none' else args.init_extension)
    for p in model.parameters():                       # LoRA weights in fp32 (stable), base stays in `dtype`
        if p.requires_grad:
            p.data = p.data.float()
    model.config.use_cache = False
    tr, tot = param_report(model)
    print(f"trainable {tr/1e6:.2f}M / {tot/1e6:.1f}M ({100*tr/tot:.2f}%)")
    params = [p for p in model.parameters() if p.requires_grad]
    module = extension_module(model)
    ext_ids = {id(p) for p in module.parameters()} if module is not None else set()
    groups = []
    for is_ext, base_lr in ((False, args.lr), (True, args.extension_lr or args.lr)):
        group = [p for p in params if (id(p) in ext_ids) == is_ext]
        if group:
            groups.append(dict(params=group, lr=base_lr, lr_ratio=base_lr / args.lr))
    opt = torch.optim.AdamW(groups, lr=args.lr, betas=(0.9, 0.999), weight_decay=args.weight_decay)
    # The LR schedule length is fixed when a run starts and stored in state.pt. Extending a finished run to more epochs
    # keeps it: steps past its end stay at the final (minimum) LR instead of the cosine being redrawn over the new total.
    schedule_steps = max_steps
    if resumed:
        ckpt = torch.load(state_path, map_location="cpu", weights_only=False)
        if ckpt.get('model_sha256') and ckpt['model_sha256'] != model_digest(last_dir):
            raise ValueError('Checkpoint weights and optimizer state do not match')
        lr_ratios = [group["lr_ratio"] for group in groups]
        opt.load_state_dict(ckpt["opt"])
        for group, ratio in zip(opt.param_groups, lr_ratios):
            group["lr_ratio"] = ratio
        if ctx.primary:
            rewind_logs(args.out_dir, ckpt['step'])
        start_step, best_val = ckpt["step"], ckpt["best_val"]
        best_loss = ckpt.get('best_loss', float('inf'))
        plateau_loss, stale_checks = ckpt.get('plateau_loss', float('inf')), ckpt.get('stale_checks', 0)
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
        if ctx.primary and not os.path.isdir(os.path.join(args.out_dir, 'final')):
            commit(args.out_dir, 'final', lambda path: save_model(model, path, args))
        print("already finished"); return
    if args.early_stop_patience and stale_checks >= args.early_stop_patience:
        print('already stopped by validation plateau'); return
    if ctx.primary:
        with open(os.path.join(args.out_dir, "args.json"), "w") as f:
            json.dump({**vars(args), "family": family, "trainable_params": tr, "total_params": tot,
                       "max_steps": max_steps, "schedule_steps": schedule_steps}, f, indent=1)

    stream = make_stream(args, 'train', files, order=order, seed=args.seed, shuffle_candidates=bool(args.shuffle_candidates))
    loader = DataLoader(stream, batch_size=args.batch_size, sampler=list(rank_positions(start_step * glob_bs, len(order), args.batch_size, ctx.world, ctx.rank)),
                        num_workers=args.num_workers, drop_last=True, pin_memory=device.type == "cuda",
                        collate_fn=st.collator,
                        prefetch_factor=4 if args.num_workers > 0 else None)
    val_loader = make_val_loader(args, processor, family)
    fit_loader = None
    if args.fit_eval_samples > 0:                      # training problems in their original candidate order
        fit_loader = DataLoader(make_stream(args, 'train', files[:args.fit_eval_samples], seed=args.seed),
                                batch_size=args.val_batch_size, shuffle=False, num_workers=min(2, args.num_workers),
                                collate_fn=st.collator)
    it = iter(loader)
    if resumed and ckpt.get('rng_by_rank'):
        if len(ckpt['rng_by_rank']) != ctx.world:
            raise ValueError('Exact optimizer continuation requires the same world size')
        restore_rng(ckpt['rng_by_rank'][ctx.rank])
    elif resumed and ckpt.get('rng'):
        restore_rng(ckpt['rng'])
    elif ctx.world > 1:
        seed_everything(args.seed + ctx.rank)

    def save_state(step):
        rngs = ctx.gather(capture_rng())
        if ctx.primary:
            state = dict(opt=opt.state_dict(), step=step, best_val=best_val, schedule_steps=schedule_steps,
                         rng_by_rank=rngs, best_loss=best_loss, plateau_loss=plateau_loss,
                         stale_checks=stale_checks, best_checkpoint=pointer_for(args.out_dir, 'best'))
            commit(args.out_dir, 'last', lambda path: save_model(model, path, args), state=state)
        ctx.barrier()

    def bounded_validation(loader):
        records = []
        for batch in loader:
            if stop.exhausted():
                return None
            records.extend(evaluate_batches(model, [batch], st, device, dtype))
        return summarize(records)

    def validate(step):
        nonlocal best_val, best_loss, plateau_loss, stale_checks, stop_training
        if not ctx.primary:
            return
        if fit_loader is not None:
            fit = bounded_validation(fit_loader)
            if fit is None:
                return
            log_jsonl(metrics_path, {"step": step, "train_fit_acc": fit["scores"]["acc"], "train_fit_n": fit["n"],
                                     "train_fit_pred_hist": fit["scores"]["pred_hist"]})
            print(f"[fit] step {step}: acc on {fit['n']} training problems {fit['scores']['acc']:.4f} "
                  f"| predicted positions {fit['scores']['pred_hist']}")
        if val_loader is None:
            return
        s = bounded_validation(val_loader)
        if s is None:
            print('Validation interrupted; partial results excluded from checkpoint selection')
            return
        acc, hist = s["scores"]["acc"], s["scores"]["pred_hist"]
        log_jsonl(metrics_path, {"step": step, "val_acc": acc, "val_n": s["n"], "val_pred_hist": hist,
                                 **({'val_loss': s['scores']['loss'], 'val_answer_loss': s['scores']['answer_loss']}
                                    if args.protocol == 'answer_letter' else {
                                        'val_answer_says_yes': s['scores']['answer_says_yes'],
                                        'val_distractor_says_yes': s['scores']['distractor_says_yes']})})
        # one position getting nearly every prediction = the model ignores the images and outputs the label prior
        collapsed = "  <- COLLAPSED to one answer" if max(hist) > 0.8 * s["n"] else ""
        print(f"[val] step {step}: acc {acc:.4f} on {s['n']} problems (chance {1/len(hist):.3g}) "
              f"| predicted positions {hist}{collapsed}")
        val_loss = s['scores'].get('loss', float('inf'))
        better = {'accuracy': acc > best_val, 'loss': val_loss < best_loss,
                  'accuracy_loss': (acc, -val_loss) > (best_val, -best_loss)}[args.select_by]
        if args.early_stop_patience:
            if val_loss < plateau_loss - args.early_stop_min_delta:
                plateau_loss, stale_checks = val_loss, 0
            else:
                stale_checks += 1
            stop_training = stale_checks >= args.early_stop_patience
        if better:
            best_val, best_loss = acc, val_loss
            selection = dict(step=step, val_acc=acc, val_loss=val_loss if math.isfinite(val_loss) else None, select_by=args.select_by)
            commit(args.out_dir, 'best', lambda path: save_model(model, path, args), selection=selection)

    set_shared_train_mode(model, args)
    recorder = GradientRecorder(model) if args.diagnostics_every else None
    t0, seen, agg, n_agg = time.time(), 0, defaultdict(float), 0
    step = start_step
    if ctx.any(stop.externally_stopped()):
        save_state(start_step)
        print('Stop requested before training; saved initialized/resumed state')
        return
    for step in range(start_step, max_steps):
        capture = recorder is not None and (step + 1) % args.diagnostics_every == 0
        if capture:
            recorder.start()
        for g in opt.param_groups:
            g["lr"] = lr_for(step) * g.get("lr_ratio", 1.)
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
        ctx.gradients(params)
        if capture and ctx.primary:
            log_jsonl(os.path.join(args.out_dir, 'gradients.jsonl'), dict(step=step + 1, **recorder.finish(model)))
        elif capture:
            recorder.finish(model)
        if any(p.grad is not None for p in model.parameters() if not p.requires_grad):
            raise RuntimeError('A frozen base parameter received a gradient')
        gn = torch.nn.utils.clip_grad_norm_(params, args.max_grad_norm)
        if not math.isfinite(float(gn)):
            raise RuntimeError(f"non-finite gradient norm at step {step}")
        opt.step()
        opt.zero_grad(set_to_none=True)
        done = step + 1
        n_agg += 1
        if done % args.log_every == 0 or done == max_steps:
            metrics = ctx.metrics({k: v / n_agg for k, v in agg.items()})
            rec = {"step": done, **metrics, "grad_norm": float(gn),
                   "lr": opt.param_groups[0]["lr"], "samples_per_s": seen * ctx.world / (time.time() - t0), "presentations": done * glob_bs,
                   "mem_gb": torch.cuda.max_memory_allocated() / 2**30 if device.type == "cuda" else 0.0}
            if ctx.primary:
                log_jsonl(metrics_path, rec)
            print(" ".join(f"{k} {v:.4g}" if isinstance(v, float) else f"{k} {v}" for k, v in rec.items()), flush=True)
            agg.clear()
            n_agg = 0
        out_of_time = ctx.any(stop.exhausted())
        if not out_of_time and (done % args.val_every == 0 or done == max_steps):
            with preserved_rng():
                validate(done)
            set_shared_train_mode(model, args)
        out_of_time = ctx.any(stop.exhausted())
        stop_training = ctx.any(stop_training)
        if done % args.save_every == 0 or done == max_steps or out_of_time or stop_training:
            save_state(done)
        if stop_training:
            print(f'validation loss plateau: stopped at update {done} after {stale_checks} checks')
            break
        if out_of_time and done < max_steps:
            print(f"time budget reached at step {done}; resubmit the same command to continue")
            return
    if ctx.primary:
        commit(args.out_dir, 'final', lambda path: save_model(model, path, args))
    ctx.barrier()
    print("finished; adapters: last/ best/ final/")


if __name__ == "__main__":
    main()
