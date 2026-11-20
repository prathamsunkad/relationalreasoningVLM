#!/usr/bin/env python
"""Learning curve and test accuracy of the Slot Abstractor runs, for results/ and RESULTS.md.

  python summarize_runs.py                                  # curve from run_logs/*.log -> results/val_curve.csv
  python summarize_runs.py --csv weights/<name>_log.csv     # or from the run's own per-epoch CSV
  python summarize_runs.py --eval_glob "weights/*best.pth.tar.seed*.eval.json"   # test: mean +- sd over seeds

The training logs print one line per epoch:  "epoch E: train acc A  val acc V  best B ...". A resumed job can repeat
an epoch line only if that epoch was redone, so the last line seen for each epoch is kept.
"""
import argparse
import csv
import glob
import json
import os
import re
import statistics

HERE = os.path.dirname(os.path.abspath(__file__))
LINE = re.compile(r"epoch (\d+): train acc ([\d.]+)\s+val acc ([\d.]+)\s+best ([\d.]+)")


def from_logs(pattern):
    rows = {}
    for path in sorted(glob.glob(pattern)):
        for line in open(path, errors="replace"):
            m = LINE.search(line)
            if m:
                e = int(m.group(1))
                rows[e] = dict(epoch=e + 1, train_acc=float(m.group(2)), val_acc=float(m.group(3)),
                               best_val_acc=float(m.group(4)), log=os.path.basename(path))
    return [rows[e] for e in sorted(rows)]


def from_csv(path):
    return [dict(epoch=int(r["epoch"]) + 1, train_acc=float(r["train_acc"]), val_acc=float(r["val_acc"]),
                 best_val_acc=float(r["best_val_acc"]), log=os.path.basename(path)) for r in csv.DictReader(open(path))]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--logs", default=os.path.join(HERE, "run_logs", "*.log"))
    ap.add_argument("--csv", default=None)
    ap.add_argument("--out", default=os.path.join(HERE, "results", "val_curve.csv"))
    ap.add_argument("--eval_glob", default=None)
    a = ap.parse_args()
    if a.eval_glob:
        runs = [json.load(open(p)) for p in sorted(glob.glob(a.eval_glob))]
        if not runs:
            raise SystemExit(f"no files match {a.eval_glob}")
        for split in ("validation", "test"):
            acc = [r[split]["accuracy"] for r in runs if split in r]
            if acc:
                sd = statistics.stdev(acc) if len(acc) > 1 else 0.0
                print(f"{split}: {statistics.mean(acc):.2f}% +- {sd:.2f} over {len(acc)} seeds "
                      f"({', '.join(f'{x:.2f}' for x in acc)}), n={runs[0][split]['n']}")
        return
    rows = from_csv(a.csv) if a.csv else from_logs(a.logs)
    if not rows:
        raise SystemExit("no epoch lines found")
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    best = max(rows, key=lambda r: r["val_acc"])
    print(f"{len(rows)} epochs -> {a.out}")
    print(f"best validation {best['val_acc']:.2f}% after epoch {best['epoch']} (train {best['train_acc']:.2f}%); "
          f"last: epoch {rows[-1]['epoch']}, train {rows[-1]['train_acc']:.2f}%, val {rows[-1]['val_acc']:.2f}%")
    for e in (1, 10, 20, 30, 40, 50, 60, 70):
        r = next((x for x in rows if x["epoch"] == e), None)
        if r:
            print(f"  epoch {e:3d}: train {r['train_acc']:6.2f}  val {r['val_acc']:6.2f}")


if __name__ == "__main__":
    main()
