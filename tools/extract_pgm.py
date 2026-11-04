"""Create the train_full/ val_full/ test_full/ layout that the Slurm scripts expect, from the raw PGM download.

  # from the original tarball (one streaming pass; extracts train AND val together):
  python tools/extract_pgm.py --tar /path/neutral.tar.gz --regime neutral --out_root $DATA --splits train,val
  # from a folder where train/val/test files are mixed (or in sub-folders); links by default, --copy to copy:
  python tools/extract_pgm.py --src_dir /path/neutral --regime neutral --out_root $DATA --splits train,val
  # only keep a random subset of train problems (deterministic per file name, and nested: 20% is inside 50%):
  python tools/extract_pgm.py --tar ... --splits train --train_fraction 0.2

Result: $DATA/train_full/<regime>/*.npz, $DATA/val_full/<regime>/*.npz (same layout as your test_full/<regime>).
The split of a file is read from its name/path: the first of the words train|val|test delimited by _ - . or /
(e.g. PGM_neutral_train_123.npz, neutral/train/123.npz). ALWAYS run with --dry_run first: it prints how many files
it would put in each split and the first names it matched, so you can confirm the naming assumption.
Already-existing output files are skipped, so an interrupted extraction can simply be restarted.
"""
import argparse
import hashlib
import os
import re
import shutil
import sys
import tarfile
from collections import Counter

SPLIT_RE = re.compile(r"(?:^|[_/.\-])(train|val|test)(?=[_/.\-]|$)")


def split_of(name):
    m = SPLIT_RE.search(name)
    return m.group(1) if m else None


def keep(name, fraction, seed):
    """Deterministic pseudo-random subset: a file is kept iff hash(seed, basename) < fraction. Nested across fractions."""
    if fraction >= 1.0:
        return True
    h = int(hashlib.md5(f"{seed}:{os.path.basename(name)}".encode()).hexdigest()[:12], 16) / float(16 ** 12)
    return h < fraction


def iter_tar(path):
    with tarfile.open(path, "r|*") as tf:                 # streaming: works for .tar and .tar.gz without random access
        for m in tf:
            if m.isfile() and m.name.endswith(".npz"):
                yield m.name, (lambda tf=tf, m=m: tf.extractfile(m))
    return


def iter_dir(path):
    for root, _, files in os.walk(path):
        for f in files:
            if f.endswith(".npz"):
                full = os.path.join(root, f)
                yield os.path.relpath(full, path), full


def main():
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--tar")
    g.add_argument("--src_dir")
    ap.add_argument("--regime", required=True, help="output sub-folder name, e.g. neutral")
    ap.add_argument("--out_root", required=True, help="the pgm_data folder")
    ap.add_argument("--splits", default="train,val")
    ap.add_argument("--train_fraction", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--copy", action="store_true", help="--src_dir mode: copy instead of symlink")
    ap.add_argument("--dry_run", action="store_true")
    ap.add_argument("--limit", type=int, default=0, help="stop after N matched files per split (testing)")
    args = ap.parse_args()
    want = set(args.splits.split(","))
    outs = {s: os.path.join(args.out_root, f"{s}_full", args.regime) for s in want}
    if not args.dry_run:
        for d in outs.values():
            os.makedirs(d, exist_ok=True)

    seen, written, skipped, unmatched, examples = Counter(), Counter(), Counter(), 0, {}
    src = iter_tar(args.tar) if args.tar else iter_dir(args.src_dir)
    for name, handle in src:
        sp = split_of(name)
        if sp is None:
            unmatched += 1
            continue
        seen[sp] += 1
        if sp not in want:
            continue
        if sp == "train" and not keep(name, args.train_fraction, args.seed):
            continue
        if args.limit and written[sp] + skipped[sp] >= args.limit:
            continue
        examples.setdefault(sp, []).append(name) if len(examples.get(sp, [])) < 3 else None
        dst = os.path.join(outs[sp], os.path.basename(name))
        if os.path.exists(dst):
            skipped[sp] += 1
            continue
        written[sp] += 1
        if args.dry_run:
            continue
        if args.tar:
            with open(dst + ".part", "wb") as f:
                shutil.copyfileobj(handle(), f)
            os.replace(dst + ".part", dst)
        elif args.copy:
            shutil.copyfile(handle, dst)
        else:
            os.symlink(os.path.abspath(handle), dst)
    verb = "would write" if args.dry_run else "wrote"
    print(f"found per split: {dict(seen)} | unmatched names (no train/val/test word): {unmatched}")
    for s in sorted(want):
        print(f"  {s}: {verb} {written[s]} files, {skipped[s]} already present -> {outs[s]}   e.g. {examples.get(s, [])}")
    if not seen:
        sys.exit("no .npz files found")
    if unmatched and not sum(seen.values()):
        sys.exit("no file name contained train/val/test; adapt SPLIT_RE")


if __name__ == "__main__":
    main()
