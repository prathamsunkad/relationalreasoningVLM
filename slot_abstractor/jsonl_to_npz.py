"""
Option B helper: convert the JSONL + contact-sheet package into the original PGM NPZ
format, so the authors' train/test scripts can run on it UNCHANGED.

Each puzzle becomes <out>/PGM_neutral20_<split>_<n>.npz with
    image  : uint8 (16, 160, 160)  8 context panels (row-major) then choices A-H
    target : int                   answer index
which is exactly what `dataset` in train_slot_abstractor_pgm.py reads
(np.load -> data["image"].reshape(16, 160, 160), data["target"]). The original loader
picks files by the substrings "train" / "val" / "test" in the file name.

Cost: one file per puzzle (241k for train+validation, +200k for test) and roughly
15-30 KB each with compression. Check your inode and disk quota first.

  python jsonl_to_npz.py --root $DATA --layout layout.json --out pgm_datasets/neutral20/ \
      --splits train validation test --workers 8
"""
import argparse
import os
from multiprocessing import Pool

import numpy as np

from pgm_jsonl_data import JsonlPGM

SPLIT_TAG = {"train": "train", "validation": "val", "test": "test"}


def work(job):
    root, split, layout, out, lo, hi = job
    ds = JsonlPGM(root, f"{split}.jsonl", layout, 80, n=hi)
    for i in range(lo, hi):
        f = os.path.join(out, f"PGM_neutral20_{SPLIT_TAG[split]}_{i}.npz")
        if os.path.exists(f):
            continue
        img = np.stack([np.asarray(p, np.uint8) for p in ds.panels(i)])
        np.savez_compressed(f, image=img, target=np.int64(ds.targets[i]), id=ds.ids[i])
    return hi - lo


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--layout", required=True)
    ap.add_argument("--out", default="pgm_datasets/neutral20/")
    ap.add_argument("--splits", nargs="+", default=["train", "validation", "test"])
    ap.add_argument("--n", type=int, default=0, help="first N rows per split (0 = all)")
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    for split in a.splits:
        total = len(JsonlPGM(a.root, f"{split}.jsonl", a.layout, 80, n=a.n))
        step = 2000
        jobs = [(a.root, split, a.layout, a.out, lo, min(lo + step, total)) for lo in range(0, total, step)]
        done = 0
        with Pool(a.workers) as pool:
            for k in pool.imap_unordered(work, jobs):
                done += k
                print(f"\r{split}: {done}/{total}", end="", flush=True)
        print()
    print(f"wrote NPZ files to {a.out}")
