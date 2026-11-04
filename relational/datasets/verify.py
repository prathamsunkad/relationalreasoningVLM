"""Load every converted split through relvlm's own loaders: counts, kinds, label balance, image sizes, panel boxes,
and a preview sheet per split (first examples, panel boxes drawn in red) under <data>/_previews/.

  python datasets/verify.py [--n 6] [glob ...]      e.g.  python datasets/verify.py art/*/test.jsonl
"""
import argparse
import collections
import glob
import json
import os
import sys

import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data import JsonlSource  # noqa: E402
from packing import ROOT  # noqa: E402


def preview(src, n, path, thumb=200):
    tiles = []
    for i in range(min(n, len(src))):
        ex = src.get(i, np.random.default_rng(0))
        row = []
        for s in ex["seqs"][:4]:
            im = s["image"].copy()
            d = ImageDraw.Draw(im)
            W, H = im.size
            for x0, y0, x1, y1 in s["boxes"]:
                d.rectangle([x0 * W, y0 * H, x1 * W - 1, y1 * H - 1], outline=(255, 0, 0), width=max(1, W // 160))
            im.thumbnail((thumb, thumb))
            canvas = Image.new("RGB", (thumb, thumb), (230, 230, 230))
            canvas.paste(im, (0, 0))
            if ex["kind"] == "cand" and len(row) == ex["label"]:
                ImageDraw.Draw(canvas).rectangle([0, 0, thumb - 1, thumb - 1], outline=(0, 160, 0), width=4)
            row.append(canvas)
        while len(row) < 4:
            row.append(Image.new("RGB", (thumb, thumb), (255, 255, 255)))
        tiles.append(np.concatenate([np.asarray(t) for t in row], 1))
    Image.fromarray(np.concatenate(tiles, 0)).save(path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("globs", nargs="*", default=["**/*.jsonl"])
    ap.add_argument("--n", type=int, default=6)
    ap.add_argument("--full", action="store_true", help="decode every image (slow), not just the previewed ones")
    args = ap.parse_args()
    os.makedirs(os.path.join(ROOT, "_previews"), exist_ok=True)
    report = {}
    files = sorted({f for g in args.globs for f in glob.glob(os.path.join(ROOT, g), recursive=True)
                    if "/_" not in f[len(ROOT):]})
    for f in files:
        rel = os.path.relpath(f, ROOT)
        src = JsonlSource(f)
        kinds, labels = collections.Counter(), collections.Counter()
        n_seq = collections.Counter()
        for r in src.rows:
            k = "cand" if "candidates" in r else ("image" if "answer" not in r else
                                                   ("letter" if r.get("choices") not in (None, ["Yes", "No"]) else "binary"))
            kinds[k] += 1
            labels[str(r.get("answer"))] += 1
            n_seq[len(r.get("candidates", [1]))] += 1
        check = range(len(src)) if args.full else range(min(args.n, len(src)))
        sizes = collections.Counter()
        for i in check:
            ex = src.get(i, np.random.default_rng(0))
            for s in ex["seqs"]:
                sizes[s["image"].size] += 1
                assert all(0 <= v <= 1 for b in s["boxes"] for v in b), (rel, i, s["boxes"])
        prev = os.path.join(ROOT, "_previews", rel.replace("/", "__").replace(".jsonl", ".png"))
        preview(src, args.n, prev)
        report[rel] = {"rows": len(src), "kinds": dict(kinds), "seqs_per_example": dict(n_seq),
                       "answers": dict(labels.most_common(8)), "image_sizes": {str(k): v for k, v in sizes.most_common(3)}}
        print(f"{rel}: {json.dumps(report[rel])}", flush=True)
    with open(os.path.join(ROOT, "_previews", "report.json"), "w") as fh:
        json.dump(report, fh, indent=1)


if __name__ == "__main__":
    main()
