"""Reconstruction quality of finished Stage 0 models on any dataset (held-out splits, transfer sets). Trains nothing.

  python tools/eval_stage0.py --model_id Qwen/Qwen3-VL-2B-Instruct --out_dir <dir> \
      --ckpt pgm32=<run>/best.pt --ckpt mix256=<run>/best.pt \
      --source pgm=matrix:<dir> --source clevr=jsonl:<file> [--samples 1000]

Every model sees the same images. One row per (model, source) is appended to <out_dir>/results.jsonl as it finishes:
  mse      the Stage 0 loss: mean squared error in that model's own standardised feature space
  dataset_var  the MSE of predicting this dataset's per-channel mean in the same space
  r2       1 - mse / dataset_var. Use this to compare datasets or models, whose standardisations differ
  entropy, slots_used, panels_per_example, and a mask picture in <out_dir>/viz/
"""
import argparse
import json
import os
import sys
import time

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from backbones import DTYPES, Backbone, load_vlm  # noqa: E402
from common import seed_everything  # noqa: E402
from data import FixedSet, build_sources, parse_mix, subset  # noqa: E402
from slots import attn_diagnostics, load_slot_ae  # noqa: E402
from train_stage0 import FewSeqs, KeepImages, PanelFeatures, save_viz  # noqa: E402


def parse():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_id", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--ckpt", action="append", required=True, help="name=path to a Stage 0 best.pt, repeatable")
    ap.add_argument("--source", action="append", required=True, help="name=type:path, repeatable")
    ap.add_argument("--samples", type=int, default=1000, help="examples per source (fewer if the source is smaller)")
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--panel_px", type=int, default=160)
    ap.add_argument("--file_cache_dir", default=None)
    ap.add_argument("--dtype", default="bf16", choices=list(DTYPES))
    ap.add_argument("--attn", default="sdpa")
    ap.add_argument("--num_workers", type=int, default=6)
    ap.add_argument("--seed", type=int, default=0)
    return ap.parse_args()


class Stats:
    """Running totals of one model on one source."""

    def __init__(self):
        self.sse, self.elems, self.pos, self.s1, self.s2, self.ent, self.used = 0.0, 0, 0, None, None, [], []

    def add(self, mse, z, attn):
        n, c = z.shape[0], z.shape[1]
        self.sse, self.elems, self.pos = self.sse + mse * n * c, self.elems + n * c, self.pos + n
        s1, s2 = z.sum(0), (z * z).sum(0)
        self.s1 = s1 if self.s1 is None else self.s1 + s1
        self.s2 = s2 if self.s2 is None else self.s2 + s2
        e, _, u = attn_diagnostics(attn)
        self.ent.append(e)
        self.used.append(u)

    def row(self):
        var = float((self.s2 - self.s1 * self.s1 / self.pos).sum() / self.elems)
        mse = self.sse / self.elems
        return dict(mse=mse, dataset_var=var, r2=1 - mse / var, entropy=sum(self.ent) / len(self.ent),
                    slots_used=sum(self.used) / len(self.used))


@torch.no_grad()
def evaluate(name, spec, args, feats, collate, aes, device, dtype, out):
    mix = parse_mix([f"{name}={spec}"])
    src = build_sources(mix, args.panel_px, args.file_cache_dir)[0]
    n = min(args.samples, len(src))
    ds = FewSeqs(FixedSet(src, subset(range(len(src)), n, 1234), name), 1, 0)
    loader = DataLoader(ds, batch_size=args.batch_size, num_workers=args.num_workers, collate_fn=collate)
    stats = {nm: Stats() for nm, _ in aes}
    panels, images, first = 0, 0, {}
    for i, batch in enumerate(loader):
        x = feats(batch, device, dtype)
        panels, images = panels + x.shape[0], images + len(batch["boxes"])
        for nm, ae in aes:
            torch.manual_seed(args.seed + i)              # the same slot-init noise for every model
            loss, _ = ae(x)
            z = ae.standardize(x).reshape(-1, x.shape[-1]).double()
            _, _, attn = ae.extract(x)
            stats[nm].add(loss.item(), z, attn)
            first.setdefault(nm, (batch, attn))
    rows = []
    for nm, ae in aes:
        row = dict(model=nm, source=name, examples=n, panels_per_example=panels / max(1, images), **stats[nm].row())
        save_viz(first[nm][0], first[nm][1], ae.cfg["grid"], os.path.join(out, "viz", f"{nm}__{name}.png"))
        rows.append(row)
    return rows


def main():
    args = parse()
    os.makedirs(os.path.join(args.out_dir, "viz"), exist_ok=True)
    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if os.environ.get("SLURM_GPUS_ON_NODE") and device.type != "cuda":
        raise SystemExit("Slurm allocated a GPU but torch cannot see it: stopping instead of running on the CPU")
    dtype = DTYPES[args.dtype]
    model, processor, family = load_vlm(args.model_id, args.dtype, args.attn, device)
    model.eval().requires_grad_(False)
    bb = Backbone(model, processor, family)
    aes = []
    for c in args.ckpt:
        nm, path = c.split("=", 1)
        aes.append((nm, load_slot_ae(path, device)[0]))
    grids = {ae.cfg["grid"] for _, ae in aes}
    assert len(grids) == 1, f"models use different grids: {grids}"
    feats = PanelFeatures(bb, grids.pop())
    collate = KeepImages(processor, bb.processor_kwargs(), rel_tokens=0)
    results = os.path.join(args.out_dir, "results.jsonl")
    for s in args.source:
        name, spec = s.split("=", 1)
        t0 = time.time()
        try:
            rows = evaluate(name, spec, args, feats, collate, aes, device, dtype, args.out_dir)
        except Exception as e:                            # one unreadable dataset must not lose the others
            print(f"[eval_stage0] {name}: FAILED: {type(e).__name__}: {e}", flush=True)
            with open(results, "a") as f:
                f.write(json.dumps(dict(source=name, backbone=args.model_id, error=f"{type(e).__name__}: {e}")) + "\n")
            continue
        with open(results, "a") as f:
            for r in rows:
                f.write(json.dumps(dict(backbone=args.model_id, **r)) + "\n")
                print(f"[eval_stage0] {r['model']:12s} {name:12s} n={r['examples']:5d} panels/ex={r['panels_per_example']:.1f} "
                      f"mse={r['mse']:.4f} var={r['dataset_var']:.4f} r2={r['r2']:.4f} ent={r['entropy']:.3f} "
                      f"used={r['slots_used']:.1f}", flush=True)
        print(f"[eval_stage0] {name} done in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
