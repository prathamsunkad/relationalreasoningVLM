"""Load a few examples of every source of a mixture and print how often each source is seen in one pass of the first.
  python datasets/check_mix.py --train pgm=matrix:/path:0.5 --train clevr=jsonl:/x.jsonl:0.5 --file_cache_dir <dir>"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data import build_sources, parse_mix

ap = argparse.ArgumentParser()
ap.add_argument("--train", action="append", required=True)
ap.add_argument("--file_cache_dir", default=None)
ap.add_argument("--samples", type=int, default=3)
a = ap.parse_args()
mix = parse_mix(a.train)
srcs = build_sources(mix, 160, a.file_cache_dir)
W = sum(w for *_, w in mix)
print(f"total weight {W:.3f}")
total = len(srcs[0]) / (mix[0][2] / W)     # examples drawn in one pass of the primary source
rows = []
for (name, spec, w), s in zip(mix, srcs):
    rng = np.random.default_rng(0)
    exs = [s.get(int(i), rng) for i in rng.integers(0, len(s), a.samples)]
    assert all(e["seqs"][0]["image"].size[0] > 0 for e in exs), name
    rows.append((name, len(s), w / W, total * w / W / len(s), len(exs[0]["seqs"]), exs[0]["kind"]))
cost = sum(r[2] * r[4] for r in rows)
print(f"{'source':12s} {'items':>8s} {'weight':>7s} {'passes':>7s} {'seqs':>5s} {'compute':>8s}  kind")
for name, n, f, passes, seqs, kind in rows:
    print(f"{name:12s} {n:8d} {f:7.2f} {passes:7.2f} {seqs:5d} {f * seqs / cost:8.2f}  {kind}")
