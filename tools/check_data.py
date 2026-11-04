"""Sanity-check the PGM files BEFORE blaming the model: value range / dtype of the panels, target balance, and PNGs of
what the model actually receives. Run (no GPU needed):
  python tools/check_data.py --dir $DATA/val_full/neutral --n 2000 --out_dir $RUNS/check
Look at the PNGs: the context panels must show visible shapes/lines on a white-ish background."""
import argparse
import os
import sys
from collections import Counter

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pgmvlm.data import filled_matrices, list_npz, load_pgm_instance, panels_to_grid_image  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--n", type=int, default=2000)
    ap.add_argument("--out_dir", default="check")
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)
    files = list_npz(a.dir)[: a.n]
    print(f"{len(files)} files checked from {a.dir}")
    z = np.load(os.path.join(a.dir, files[0]))
    for k in z.files:
        v = z[k]
        if v.dtype.kind in "iufb":
            print(f"key {k!r}: dtype={v.dtype} shape={v.shape} min={v.min()} max={v.max()} mean={float(np.mean(v)):.3f}")
        else:
            print(f"key {k!r}: dtype={v.dtype} shape={v.shape} (non-numeric, skipped)")
    raw = z["image"].shape
    print(f"stored image shape {raw}: " + ("declared (160,160,16) but stored panel-major -> reshape(16,160,160) on load (never transpose)"
          if raw == (160, 160, 16) else "(16,160,160) layout" if raw == (16, 160, 160) else "UNEXPECTED layout"))
    tg, mins, maxs, means, nuniq = Counter(), [], [], [], []
    for f in files:
        ctx, cand, t = load_pgm_instance(os.path.join(a.dir, f))
        tg[t] += 1
        mins.append(float(ctx.min())); maxs.append(float(ctx.max())); means.append(float(ctx.mean()))
        nuniq.append(len(np.unique(ctx)))
    print(f"panel value range over files: min {min(mins)}  max {max(maxs)}  mean-of-means {np.mean(means):.2f}  "
          f"distinct grey levels per problem ~{np.mean(nuniq):.1f}")
    print("target histogram (index 0-7):", [tg[i] for i in range(8)], " <- should be roughly uniform")
    if max(maxs) <= 1.0:
        print("!! WARNING: panel values are <= 1.0, but panels_to_grid_image casts with astype(uint8): the images would be "
              "(almost) BLACK. Scale by 255 first.")
    if np.mean(nuniq) < 2:
        print("!! WARNING: panels are constant - the model sees blank images.")
    for i in range(2):
        ctx, cand, t = load_pgm_instance(os.path.join(a.dir, files[i]))
        panels_to_grid_image(ctx, (3, 3)).save(os.path.join(a.out_dir, f"p{i}_context.png"))
        panels_to_grid_image(cand, (2, 4), add_labels=True).save(os.path.join(a.out_dir, f"p{i}_candidates.png"))
        for k, im in enumerate(filled_matrices(ctx, cand)):          # what the model actually scores, one image per candidate
            im.save(os.path.join(a.out_dir, f"p{i}_filled_{k + 1}{'_ANSWER' if k == t else ''}.png"))
        print(f"problem {i}: {files[i]} target={t} (candidate number {t + 1})")
    print(f"PNGs written to {a.out_dir}")


if __name__ == "__main__":
    main()
