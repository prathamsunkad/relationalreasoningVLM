#!/usr/bin/env python
"""
Infer layout.json (panel boxes) for PGM contact sheets from the images alone.

Idea: across many puzzles, the inside of every panel changes from sheet to sheet,
while the background, borders, the "?" cell, the A-H labels and the prompt never
change. The per-pixel standard deviation over N sheets is therefore > 0 exactly
inside the 8 context panels and 8 choice panels. Row/column bands of that
"varying" mask give the boxes; the missing cell is the empty slot of the 3x3 grid.

Expects the usual arrangement: 3x3 matrix above 2 rows of 4 choices (A-D, E-H).
If the sheet is laid out differently it stops, saves the variation map, and you
write layout.json by hand.

  python infer_layout.py --data $DATA --jsonl $DATA/validation.jsonl --out layout.json

Always open the saved overlay (layout_check.png) before using the layout.
"""
import argparse
import json
import os

import numpy as np
from PIL import Image, ImageDraw


def image_path(row, key):
    p = row[key]
    return p[0] if isinstance(p, list) else p


def runs(profile, thr, merge_gap, min_len):
    on = profile > thr
    out, start = [], None
    for i, v in enumerate(np.append(on, False)):
        if v and start is None:
            start = i
        elif not v and start is not None:
            out.append([start, i])
            start = None
    merged = []
    for r in out:
        if merged and r[0] - merged[-1][1] <= merge_gap:
            merged[-1][1] = r[1]
        else:
            merged.append(r)
    return [r for r in merged if r[1] - r[0] >= min_len]


def boxes_from_bands(row_bands, col_bands, size=None):
    """Equal-size boxes centred on each band intersection."""
    h = size[1] if size else int(np.median([b - a for a, b in row_bands]))
    w = size[0] if size else int(np.median([b - a for a, b in col_bands]))
    out = []
    for ra, rb in row_bands:
        cy = (ra + rb) / 2
        for ca, cb in col_bands:
            cx = (ca + cb) / 2
            out.append([int(round(cx - w / 2)), int(round(cy - h / 2)),
                        int(round(cx + w / 2)), int(round(cy + h / 2))])
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", "--root", dest="root", default=os.environ.get("DATA"),
                    help="dataset root, image paths are relative to it (default: $DATA)")
    ap.add_argument("--jsonl", help="default: $DATA/validation.jsonl")
    ap.add_argument("--image_key", default="images")
    ap.add_argument("--n", type=int, default=300, help="sheets to use (more = cleaner map)")
    ap.add_argument("--thr", type=float, default=0.5, help="std threshold for 'varying' pixels")
    ap.add_argument("--min_frac", type=float, default=0.02, help="min fraction of varying pixels per row/col")
    ap.add_argument("--merge_gap", type=int, default=3, help="merge bands separated by <= this many px")
    ap.add_argument("--min_len", type=int, default=20)
    ap.add_argument("--pad", type=int, default=0, help="grow every box by this many px")
    ap.add_argument("--out", default="layout.json")
    args = ap.parse_args()
    if not args.root:
        ap.error("give the dataset root with --data $DATA (or export DATA=...)")
    args.jsonl = args.jsonl or os.path.join(args.root, "validation.jsonl")

    rows = []
    with open(args.jsonl) as fh:
        for line in fh:
            if line.strip():
                rows.append(json.loads(line))
            if len(rows) >= args.n:
                break
    sheets = [np.asarray(Image.open(os.path.join(args.root, image_path(r, args.image_key)))
                         .convert("RGB").convert("L"), dtype=np.float32) for r in rows]
    if len({s.shape for s in sheets}) != 1:
        raise SystemExit(f"Sheets differ in size: {sorted({s.shape for s in sheets})}")
    stack = np.stack(sheets)
    hgt, wid = stack.shape[1:]
    std = stack.std(0)
    mask = std > args.thr
    out_dir = os.path.dirname(os.path.abspath(args.out))
    vis = (255 * std / max(std.max(), 1e-6)).astype(np.uint8)
    Image.fromarray(vis).save(os.path.join(out_dir, "layout_variation.png"))
    print(f"{len(sheets)} sheets of {wid}x{hgt}; varying pixels: {mask.mean():.1%}")

    row_bands = runs(mask.mean(1), args.min_frac, args.merge_gap, args.min_len)
    print("row bands:", row_bands)
    if len(row_bands) != 5:
        raise SystemExit(f"Expected 5 row bands (3 matrix + 2 choice rows), found {len(row_bands)}. "
                         f"Look at layout_variation.png; adjust --thr/--merge_gap or write layout.json by hand.")
    m_rows, c_rows = row_bands[:3], row_bands[3:]
    m_cols = runs(mask[m_rows[0][0]:m_rows[1][1]].mean(0), args.min_frac, args.merge_gap, args.min_len)
    c_cols = runs(mask[c_rows[0][0]:c_rows[1][1]].mean(0), args.min_frac, args.merge_gap, args.min_len)
    print("matrix column bands:", m_cols, "\nchoice column bands:", c_cols)
    if len(m_cols) != 3 or len(c_cols) != 4:
        raise SystemExit("Expected 3 matrix columns and 4 choice columns. Look at layout_variation.png.")

    matrix = boxes_from_bands(m_rows, m_cols)
    choices = boxes_from_bands(c_rows, c_cols)
    if args.pad:
        grow = lambda b: [b[0] - args.pad, b[1] - args.pad, b[2] + args.pad, b[3] + args.pad]
        matrix, choices = [grow(b) for b in matrix], [grow(b) for b in choices]
    clip = lambda b: [max(0, b[0]), max(0, b[1]), min(wid, b[2]), min(hgt, b[3])]
    matrix, choices = [clip(b) for b in matrix], [clip(b) for b in choices]

    # the missing cell should be the one matrix slot with no variation
    var_in = [mask[b[1]:b[3], b[0]:b[2]].mean() for b in matrix]
    missing = int(np.argmin(var_in))
    print("varying fraction per matrix cell:", [f"{v:.2f}" for v in var_in], "-> missing =", missing)
    if missing != 8:
        print("NOTE: the empty cell is not bottom-right; check the overlay.")

    layout = {"sheet_size": [wid, hgt], "matrix": matrix, "missing": missing, "choices": choices}
    with open(args.out, "w") as fh:
        json.dump(layout, fh, indent=1)

    img = Image.fromarray(sheets[0].astype(np.uint8)).convert("RGB")
    d = ImageDraw.Draw(img)
    for i, b in enumerate(matrix):
        d.rectangle(b, outline=(40, 160, 40) if i == missing else (220, 40, 40), width=2)
    for k, b in enumerate(choices):
        d.rectangle(b, outline=(40, 90, 220), width=2)
        d.text((b[0] + 3, b[1] + 3), "ABCDEFGH"[k], fill=(40, 90, 220))
    img.save(os.path.join(out_dir, "layout_check.png"))
    print(f"wrote {args.out}, layout_check.png and layout_variation.png in {out_dir}")


if __name__ == "__main__":
    main()
