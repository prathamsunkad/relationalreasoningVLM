"""How much of a panel's ViT feature variance is spatial (differs between positions) vs shared by all its positions?
If the per-token-LayerNormed features are nearly identical across a panel, a single averaged slot reconstructs the
panel and slot attention has no reason to specialise (uniform-attention plateau with a tiny MSE).

  python tools/feature_stats.py --model_id Qwen/Qwen3-VL-2B-Instruct --train pgm=matrix:<dir> [--n 32]
Prints, for raw / per-token LayerNorm / per-panel-centred features: spatial variance fraction, and the MSE a
constant (panel-mean) reconstruction already achieves under each normalisation.
"""
import argparse
import os
import sys

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from backbones import DTYPES, Backbone, load_vlm  # noqa: E402
from data import Collator, FixedSet, build_sources, parse_mix  # noqa: E402
from train_stage0 import FewSeqs, PanelFeatures  # noqa: E402


def report(name, x):
    """x (n_panels, N, C)."""
    mean = x.mean(1, keepdim=True)
    spatial = (x - mean).pow(2).sum(-1).mean()
    total = x.pow(2).sum(-1).mean()
    const_mse = (x - mean).pow(2).mean()
    print(f"{name:28s} spatial/total energy {float(spatial / total):.4f} | MSE of a constant per-panel reconstruction "
          f"{float(const_mse):.5f} | per-element variance {float(x.var()):.5f}", flush=True)


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_id", default="Qwen/Qwen3-VL-2B-Instruct")
    ap.add_argument("--train", required=True)
    ap.add_argument("--n", type=int, default=32)
    ap.add_argument("--grid", type=int, default=12)
    args = ap.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, processor, family = load_vlm(args.model_id, "bf16", "sdpa", device)
    model.requires_grad_(False).eval()
    bb = Backbone(model, processor, family)
    feats = PanelFeatures(bb, args.grid)
    mix = parse_mix([args.train])
    src = build_sources(mix, 160, None)[0]
    loader = DataLoader(FewSeqs(FixedSet(src, list(range(args.n)), "x"), 1, 0), batch_size=8,
                        collate_fn=Collator(processor, bb.processor_kwargs()))
    x = torch.cat([feats(b, device, torch.bfloat16).float().cpu() for b in loader], 0)     # (n_panels, P*P, C)
    print(f"{family}: {x.shape[0]} panels x {x.shape[1]} positions x {x.shape[2]} channels")
    top = x.abs().mean((0, 1)).topk(5)
    print(f"largest mean |activation| channels {top.indices.tolist()} values {[round(v, 1) for v in top.values.tolist()]}"
          f" (median channel {float(x.abs().mean((0, 1)).median()):.2f})")
    report("raw", x)
    ln = F.layer_norm(x, x.shape[-1:])
    report("per-token LayerNorm (now)", ln)
    c = x - x.mean(1, keepdim=True)
    report("raw, panel-centred", c)
    report("LN of panel-centred", F.layer_norm(c, c.shape[-1:]))
    z = (x - x.mean((0, 1))) / x.std((0, 1)).clamp_min(1e-6)
    report("per-channel standardised", z)


if __name__ == "__main__":
    main()
