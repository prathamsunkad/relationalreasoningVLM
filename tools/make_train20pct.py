"""Turn pgm_data/train_full/neutral into exactly the shared 20% selection (240,000 problems), keeping the .npz format.

  1. delete every .npz in train_full/neutral that is not selected (except the files in --protect, still in use by jobs)
  2. extract the selected files that are missing from neutral.tar.gz (one streaming pass, by member name)
  3. verify: all 240,000 present, each target equals the dataset's answer letter, nothing unselected left (except protected)
  4. write the sorted file listing for vlm_lora_pc's file cache (so training does not relist 240k files)
"""
import argparse
import os
import subprocess
import sys
import time

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--selection", required=True, help="tsv: file, target_0based, original_order")
    ap.add_argument("--train_dir", required=True)
    ap.add_argument("--tar", required=True)
    ap.add_argument("--protect", default=None, help="file names (one per line) not to delete yet")
    ap.add_argument("--cache_out", default=None, help="write the sorted selected listing here")
    ap.add_argument("--work", required=True, help="directory for the member list")
    a = ap.parse_args()

    sel = {}
    for line in open(a.selection):
        if line.startswith("#") or not line.strip():
            continue
        f, t, _ = line.rstrip("\n").split("\t")
        sel[f] = int(t)
    assert len(sel) == 240000, len(sel)
    protect = set(l.strip() for l in open(a.protect)) if a.protect else set()

    # 1. delete unselected
    t0 = time.time()
    present = [f for f in os.listdir(a.train_dir)]
    gone = kept_protected = 0
    for f in present:
        if f in sel:
            continue
        if f in protect:
            kept_protected += 1
            continue
        os.remove(os.path.join(a.train_dir, f))
        gone += 1
        if gone % 50000 == 0:
            print(f"  deleted {gone} ...", flush=True)
    print(f"[1] {len(present)} files found; deleted {gone} unselected; kept {kept_protected} protected; "
          f"{time.time() - t0:.0f}s", flush=True)

    # 2. extract the missing selected files
    have = set(os.listdir(a.train_dir))
    missing = sorted(f for f in sel if f not in have)
    print(f"[2] selected already present {len(sel) - len(missing)}, missing {len(missing)}", flush=True)
    if missing:
        members = os.path.join(a.work, "missing_members.txt")
        with open(members, "w") as fh:
            fh.write("\n".join(f"neutral/{f}" for f in missing) + "\n")
        t0 = time.time()
        r = subprocess.run(["tar", "-xzf", a.tar, "-C", a.train_dir, "--strip-components=1", "-T", members])
        print(f"    tar exit {r.returncode} after {time.time() - t0:.0f}s", flush=True)
        if r.returncode != 0:
            sys.exit("tar failed")

    # 3. verify
    have = set(os.listdir(a.train_dir))
    lacking = [f for f in sel if f not in have]
    extra = [f for f in have if f not in sel and f not in protect]
    bad = 0
    for i, f in enumerate(sorted(sel)):
        if f in have and int(np.load(os.path.join(a.train_dir, f))["target"]) != sel[f]:
            bad += 1
        if (i + 1) % 40000 == 0:
            print(f"  verified {i + 1} ...", flush=True)
    print(f"[3] selected present {len(sel) - len(lacking)}/240000 | target mismatches {bad} | "
          f"unexpected extra files {len(extra)}", flush=True)
    if lacking or bad or extra:
        print("    first lacking:", lacking[:5], " first extra:", extra[:5])
        sys.exit("VERIFY FAILED")

    # 4. file cache
    if a.cache_out:
        os.makedirs(os.path.dirname(a.cache_out), exist_ok=True)
        with open(a.cache_out, "w") as fh:
            fh.write("\n".join(sorted(sel)))
        print(f"[4] wrote {a.cache_out}")
    print("DONE: train_dir now holds exactly the 240,000 selected problems (+ protected files, removed later)")


if __name__ == "__main__":
    main()
