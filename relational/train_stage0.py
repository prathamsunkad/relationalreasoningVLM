"""Stage 0: train slot attention on the VLM's own frozen ViT features (no labels; one model per backbone).

Each image's pre-merge patch tokens are ROI-sampled per panel to a P x P grid (the same crops Stage 1 uses), and a
slot autoencoder reconstructs them (MSE to LayerNormed features). The decoder is thrown away afterwards; Stage 1 loads
best.pt with slot attention frozen.

  python train_stage0.py --model_id Qwen/Qwen3-VL-2B-Instruct --out_dir runs/stage0_qwen \
      --train pgm=matrix:$DATA/train_full/neutral:0.6 --train clevr=jsonl:$MIX/clevr_art_train.jsonl:0.4 \
      --val pgm=matrix:$DATA/val_full/neutral

WATCH attn_entropy vs ln K in the log: slot attention sits on a plateau with uniform attention (entropy = ln K) for a
long time. Judge a run by val MSE AND entropy / slots_used / the mask pictures, and finally by Stage-1 accuracy.
Resumes from <out_dir>/last.pt; stops cleanly at --time_limit_min.
"""
import argparse
import json
import math
import os
import time

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader

from backbones import DTYPES, Backbone, load_vlm
from common import TimeBudget, log_jsonl, lr_at, save_ckpt, seed_everything
from data import Collator, FixedSet, MixtureStream, build_sources, parse_mix, subset
from relvlm import roi_sample
from slots import SlotAutoencoder, attn_diagnostics


def parse():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_id", default="Qwen/Qwen3-VL-2B-Instruct")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--train", action="append", required=True, help="name=type:path[:weight], repeatable")
    ap.add_argument("--val", default=None, help="name=type:path (first --val_samples items)")
    ap.add_argument("--val_samples", type=int, default=200)
    ap.add_argument("--train_samples", type=int, default=0,
                    help="0 = every item of the first --train source; else the same nested seeded subset Stage 1 and "
                         "the LoRA baseline use (data.subset), e.g. 60000 = the 5% PGM problems")
    ap.add_argument("--file_cache_dir", default=None)
    ap.add_argument("--seqs_per_example", type=int, default=1,
                    help="a PGM problem has 8 near-identical sequences; use this many random ones per problem")
    ap.add_argument("--panel_px", type=int, default=160)
    # slot model (paper PGM: slot_dim 32, K 16, T 3)
    ap.add_argument("--grid", type=int, default=12, help="P: each panel is sampled to P x P ViT positions")
    ap.add_argument("--slot_dim", type=int, default=32)
    ap.add_argument("--num_slots", type=int, default=16)
    ap.add_argument("--iters", type=int, default=3)
    ap.add_argument("--decoder_hidden", type=int, default=512)
    ap.add_argument("--feat_norm", default="channel", choices=["channel", "layernorm"],
                    help="channel: standardise each ViT channel with dataset stats (default; see slots.py). "
                         "layernorm: per-token LayerNorm (plateaus on Qwen: two huge channels dominate)")
    ap.add_argument("--stats_batches", type=int, default=50, help="batches used to measure the channel statistics")
    # optimisation
    ap.add_argument("--batch_size", type=int, default=16, help="examples per step")
    ap.add_argument("--lr", type=float, default=4e-4)
    ap.add_argument("--warmup_steps", type=int, default=10000)
    ap.add_argument("--max_steps", type=int, default=100000)
    ap.add_argument("--clip", type=float, default=1.0)
    ap.add_argument("--dtype", default="bf16", choices=list(DTYPES))
    ap.add_argument("--attn", default="sdpa")
    ap.add_argument("--num_workers", type=int, default=6)
    ap.add_argument("--log_every", type=int, default=100)
    ap.add_argument("--val_every", type=int, default=2000)
    ap.add_argument("--save_every", type=int, default=1000)
    ap.add_argument("--snapshot_every", type=int, default=25000,
                    help="keep step<N>.pt snapshots: pick the Stage 0 checkpoint later by Stage-1a accuracy, not MSE alone")
    ap.add_argument("--time_limit_min", type=float, default=0)
    ap.add_argument("--seed", type=int, default=0)
    # continue a finished run until convergence: warm up to --restart_lr, hold it until the best val MSE improves by
    # less than --plateau_rel over --plateau_steps, then cosine to 10% over --anneal_steps and stop (done.json).
    # --max_steps caps it: the anneal starts by max_steps - anneal_steps at the latest.
    ap.add_argument("--restart_step", type=int, default=0, help="step the continuation starts at (0 = normal run)")
    ap.add_argument("--restart_lr", type=float, default=1e-4)
    ap.add_argument("--restart_warmup", type=int, default=2000)
    ap.add_argument("--plateau_steps", type=int, default=20000)
    ap.add_argument("--plateau_rel", type=float, default=0.005)
    ap.add_argument("--anneal_steps", type=int, default=20000)
    return ap.parse_args()


class FewSeqs(torch.utils.data.Dataset):
    def __init__(self, ds, k, seed):
        self.ds, self.k, self.seed = ds, k, seed

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, i):
        ex = self.ds[i]
        if len(ex["seqs"]) > self.k:
            pick = np.random.default_rng([self.seed, 99, i]).choice(len(ex["seqs"]), self.k, replace=False)
            ex["seqs"] = [ex["seqs"][j] for j in sorted(pick)]
        return ex


class PanelFeatures:
    """batch -> (n_panels, P*P, C) ViT features of every panel, via a hook on the vision tower."""

    def __init__(self, bb, grid):
        self.bb, self.grid, self._maps = bb, grid, None
        bb.vision.register_forward_hook(self._hook)
        self._grids = None

    def _hook(self, module, args, out):
        if self._grids is not None:
            self._maps = self.bb.feature_maps(out, self._grids)

    @torch.no_grad()
    def __call__(self, batch, device, dtype):
        inputs = self.bb.model_inputs(batch, device, dtype)
        self._grids = self.bb.grids(inputs, len(batch["boxes"]))
        try:
            with torch.autocast(device.type, dtype=dtype, enabled=device.type == "cuda" and dtype != torch.float32):
                self.bb.run_vision(inputs)
        finally:
            self._grids = None
        return torch.cat([roi_sample(m, b.to(device), self.grid) for m, b in zip(self._maps, batch["boxes"])], 0)


def save_viz(batch, attn, P, path, n=6, cell=96):
    """Rows: panel crop | masks of slot 1..K. Slots should isolate objects, not stay blurry."""
    crops = []
    for img_i, b in enumerate(batch["boxes"]):
        for box in b.tolist():
            crops.append((img_i, box))
    rows = []
    K = attn.shape[1]
    for i in range(min(n, len(crops))):
        img_i, (x0, y0, x1, y1) = crops[i]
        img = batch["_images"][img_i]
        W, H = img.size
        tiles = [np.array(img.crop((x0 * W, y0 * H, x1 * W, y1 * H)).resize((cell, cell)).convert("L"))]
        for k in range(K):
            mk = attn[i, k].reshape(P, P).cpu().numpy()
            mk = (255 * (mk - mk.min()) / (mk.max() - mk.min() + 1e-8)).astype(np.uint8)
            tiles.append(np.array(Image.fromarray(mk).resize((cell, cell), Image.NEAREST)))
        rows.append(np.concatenate(tiles, 1))
    Image.fromarray(np.concatenate(rows, 0)).save(path)


class KeepImages(Collator):
    def __call__(self, items):
        b = super().__call__(items)
        b["_images"] = [s["image"] for ex in items for s in ex["seqs"]]
        return b


def main():
    args = parse()
    os.makedirs(args.out_dir, exist_ok=True)
    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = DTYPES[args.dtype]
    log_path = os.path.join(args.out_dir, "log.jsonl")
    done_path, anneal_path = os.path.join(args.out_dir, "done.json"), os.path.join(args.out_dir, "anneal.json")
    if os.path.exists(done_path):
        print(f"[stage0] already finished ({done_path})")
        return
    anneal = json.load(open(anneal_path))["start"] if os.path.exists(anneal_path) else None

    model, processor, family = load_vlm(args.model_id, args.dtype, args.attn, device)
    model.eval().requires_grad_(False)
    bb = Backbone(model, processor, family)
    feats = PanelFeatures(bb, args.grid)
    collate = KeepImages(processor, bb.processor_kwargs(), rel_tokens=0)

    mix = parse_mix(args.train)
    srcs = build_sources(mix, args.panel_px, args.file_cache_dir)
    if args.train_samples and hasattr(srcs[0], "files"):
        n_all = len(srcs[0].files)
        srcs[0].files = subset(srcs[0].files, args.train_samples, args.seed)
        print(f"[stage0] training on {len(srcs[0].files)} of {n_all} {mix[0][0]} items (--train_samples, seed {args.seed})")
    for (name, spec, w), s in zip(mix, srcs):
        print(f"[stage0] source {name}: {spec} ({len(s)} items, weight {w})")
    stream = MixtureStream(srcs, [w for *_, w in mix], [n for n, *_ in mix], args.max_steps * args.batch_size, args.seed)
    ae = SlotAutoencoder(bb.vit_dim, args.slot_dim, args.num_slots, args.iters, args.decoder_hidden, grid=args.grid,
                         feat_norm=args.feat_norm).to(device)
    opt = torch.optim.Adam(ae.parameters(), lr=args.lr)
    step, best = 0, float("inf")
    last = os.path.join(args.out_dir, "last.pt")
    if os.path.exists(last):
        ck = torch.load(last, map_location="cpu", weights_only=False)
        ae.load_state_dict(ck["model"]); opt.load_state_dict(ck["opt"])
        step, best = ck["step"], ck["best"]
        print(f"[stage0] resumed at step {step} (best val mse {best:.5f})")
    loader = DataLoader(FewSeqs(stream, args.seqs_per_example, args.seed), batch_size=args.batch_size,
                        sampler=range(step * args.batch_size, len(stream)), num_workers=args.num_workers,
                        collate_fn=collate, drop_last=True, prefetch_factor=4 if args.num_workers else None)
    val_loader = None
    if args.val:
        vmix = parse_mix([args.val])
        vname, vsrc = vmix[0][0], build_sources(vmix, args.panel_px, args.file_cache_dir)[0]
        val_loader = DataLoader(FewSeqs(FixedSet(vsrc, subset(range(len(vsrc)), args.val_samples, 1234), vname), 1, 0),
                                batch_size=args.batch_size, num_workers=min(2, args.num_workers), collate_fn=collate)

    def ckpt(tag):
        save_ckpt(os.path.join(args.out_dir, f"{tag}.pt"), model=ae.state_dict(), opt=opt.state_dict(), config=ae.cfg,
                  step=step, best=best, args=vars(args), backbone=args.model_id, vit_dim=bb.vit_dim)

    @torch.no_grad()
    def validate():
        ae.eval()
        tot, n, ents, used, first = 0.0, 0, [], [], None
        for b in val_loader:
            x = feats(b, device, dtype)
            loss, _ = ae(x)
            _, _, attn = ae.extract(x)
            e, emax, u = attn_diagnostics(attn)
            tot, n = tot + loss.item(), n + 1
            ents.append(e); used.append(u)
            if first is None:
                first = (b, attn)
        ae.train()
        return tot / max(1, n), float(np.mean(ents)), emax, float(np.mean(used)), first

    if step == 0 and args.feat_norm == "channel":
        # Per-channel mean / std of the panel features, from the first stats_batches batches of the training stream
        # (a separate loader, so the training order is untouched). Stored as buffers in every checkpoint.
        stats_loader = DataLoader(FewSeqs(stream, 1, args.seed + 5), batch_size=args.batch_size,
                                  sampler=range(0, min(len(stream), args.stats_batches * args.batch_size)),
                                  num_workers=args.num_workers, collate_fn=collate)
        s1 = s2 = None
        n = 0
        for b in stats_loader:
            x = feats(b, device, dtype).float().reshape(-1, bb.vit_dim)
            s1 = x.sum(0) if s1 is None else s1 + x.sum(0)
            s2 = (x * x).sum(0) if s2 is None else s2 + (x * x).sum(0)
            n += x.shape[0]
        mean = s1 / n
        std = (s2 / n - mean * mean).clamp_min(0).sqrt()
        ae.set_feature_stats(mean, std)
        top = std.topk(3)
        print(f"[stage0] channel stats from {n} positions: std median {float(std.median()):.3f}, largest "
              f"{[round(v, 1) for v in top.values.tolist()]} (channels {top.indices.tolist()})", flush=True)

    def start_anneal(at, reason):
        nonlocal anneal
        anneal = at
        with open(anneal_path, "w") as f:
            json.dump({"start": at, "reason": reason, "best_val_mse": best}, f)
        print(f"[stage0] annealing from step {at} for {args.anneal_steps} steps ({reason})", flush=True)

    def lr_now(s):
        if not args.restart_step:
            return lr_at(s, args.lr, args.warmup_steps, args.max_steps)
        if anneal is not None and s >= anneal:
            p = min(1.0, (s - anneal) / max(1, args.anneal_steps))
            return args.restart_lr * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * p)))
        return args.restart_lr * min(1.0, (s - args.restart_step + 1) / max(1, args.restart_warmup))

    def plateaued():
        """Best val MSE improved by less than plateau_rel over the last plateau_steps (read back from log.jsonl)."""
        if step - args.restart_step < args.plateau_steps:
            return False
        past = [r["best_val_mse"] for r in map(json.loads, open(log_path)) if "best_val_mse" in r
                and r["step"] <= step - args.plateau_steps]
        return bool(past) and (past[-1] - best) / past[-1] < args.plateau_rel

    budget, t0, run, rn = TimeBudget(args.time_limit_min), time.time(), 0.0, 0
    ae.train()
    for batch in loader:
        if step >= args.max_steps or (anneal is not None and step >= anneal + args.anneal_steps):
            break
        if args.restart_step and anneal is None and step >= args.max_steps - args.anneal_steps:
            start_anneal(step, "step cap reached")
        x = feats(batch, device, dtype)
        for g in opt.param_groups:
            g["lr"] = lr_now(step)
        loss, _ = ae(x)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(ae.parameters(), args.clip)
        opt.step()
        step, run, rn = step + 1, run + loss.item(), rn + 1
        if step % args.log_every == 0:
            rec = {"step": step, "train_mse": run / rn, "lr": opt.param_groups[0]["lr"], "panels_per_step": x.shape[0],
                   "sec_per_step": (time.time() - t0) / rn}
            print(rec, flush=True); log_jsonl(log_path, rec)
            run, rn, t0 = 0.0, 0, time.time()
        if val_loader is not None and step % args.val_every == 0:
            vm, ent, emax, used, first = validate()
            if vm < best:
                best = vm
                ckpt("best")
            save_viz(first[0], first[1], args.grid, os.path.join(args.out_dir, f"slots_step{step}.png"))
            rec = {"step": step, "val_mse": vm, "best_val_mse": best, "attn_entropy": ent, "attn_entropy_max": emax,
                   "slots_used": used}
            print(rec, flush=True); log_jsonl(log_path, rec)
            if step >= 0.25 * args.max_steps and ent > 0.95 * emax:
                print("[stage0] WARNING: slot attention is still ~uniform (entropy ~ ln K): slots have not specialised.")
            if args.restart_step and anneal is None and plateaued():
                start_anneal(step, f"best val MSE improved < {args.plateau_rel:.1%} over {args.plateau_steps} steps")
            ckpt("last")
        elif step % args.save_every == 0:
            ckpt("last")
        if args.snapshot_every and step % args.snapshot_every == 0:
            ckpt(f"step{step}")
        if budget.exhausted():
            ckpt("last")
            print("[stage0] time budget reached; resubmit to continue")
            return
    ckpt("last")
    if val_loader is None:
        ckpt("best")
    with open(done_path, "w") as f:
        json.dump({"step": step, "best_val_mse": best, "anneal_start": anneal}, f)
    print(f"[stage0] done at step {step}")


if __name__ == "__main__":
    main()
