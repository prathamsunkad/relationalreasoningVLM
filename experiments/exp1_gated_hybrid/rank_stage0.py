#!/usr/bin/env python
"""
Rank stage-0 ablation runs (successive halving) by held-out validation loss.

  python rank_stage0.py ckpt/sh              # table of all runs under ckpt/sh
  python rank_stage0.py ckpt/sh --keep 3     # also print which runs continue to the next rung
  python rank_stage0.py ckpt/sh --choose     # write the best run's settings to stage0_choice.txt

Ranks by the best validation loss each run has reached (lower is better; ln 8 = 2.079 is chance).
Loss separates runs earlier than accuracy, which barely moves until a model leaves the plateau.
Accuracy is shown with a 95% interval; runs whose best accuracy falls inside the leader's interval
are flagged as not separable.
"""
import argparse
import glob
import json
import math
import os

LN8 = math.log(8)


def wilson(acc, n, z=1.96):
    den = 1 + z * z / n
    c = (acc + z * z / (2 * n)) / den
    h = z * math.sqrt(acc * (1 - acc) / n + z * z / (4 * n * n)) / den
    return c - h, c + h


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("root", help="folder holding one sub-folder per run")
    ap.add_argument("--keep", type=int, default=0, help="how many runs advance to the next rung")
    ap.add_argument("--choose", action="store_true", help="write the best run's flags to stage0_choice.txt")
    args = ap.parse_args()

    rows = []
    for p in sorted(glob.glob(os.path.join(args.root, "*", "summary.json"))):
        s = json.load(open(p))
        hist = s.get("history") or []
        if not hist:
            continue
        name = os.path.basename(os.path.dirname(p))
        best_loss = min(hist, key=lambda h: h["loss"])
        best_acc = max(hist, key=lambda h: h["acc"])
        last = hist[-1]
        rows.append(dict(name=name, step=s.get("step", last["step"]), n=last["n"], best_loss=best_loss["loss"],
                         best_loss_step=best_loss["step"], best_acc=best_acc["acc"], last_loss=last["loss"],
                         last_acc=last["acc"], cfg=s.get("fusion_config") or {}, recon=s.get("recon_weight", 0.0),
                         val_set=s.get("val_set", "?")))
    if not rows:
        raise SystemExit(f"no finished runs (summary.json with history) under {args.root}")
    rows.sort(key=lambda r: r["best_loss"])
    lead_lo, _ = wilson(rows[0]["best_acc"], rows[0]["n"])
    print(f"\nValidation: {rows[0]['val_set']} (n={rows[0]['n']}). Chance: loss {LN8:.3f}, accuracy 12.5%.\n")
    print("| rank | run | steps | best val loss (at step) | latest val loss | best val acc [95% CI] | latest acc |")
    print("|---|---|---|---|---|---|---|")
    for i, r in enumerate(rows, 1):
        lo, hi = wilson(r["best_acc"], r["n"])
        flag = " *" if i > 1 and r["best_acc"] >= lead_lo else ""
        print(f"| {i} | {r['name']} | {r['step']} | {r['best_loss']:.4f} ({r['best_loss_step']}) | "
              f"{r['last_loss']:.4f} | {r['best_acc']:.1%} [{lo:.1%}, {hi:.1%}]{flag} | {r['last_acc']:.1%} |")
    left = [r["name"] for r in rows if r["best_loss"] < LN8 - 0.01]
    print(f"\nRuns clearly below chance loss (by > 0.01): {', '.join(left) if left else 'none yet'}")
    if any(i > 0 and r["best_acc"] >= lead_lo for i, r in enumerate(rows)):
        print("* best accuracy inside the leader's 95% interval: not separable on accuracy alone")
    if not left:
        print("No run has left the plateau: train longer before cutting (a cut now would be random).")
    if args.keep:
        print(f"\nAdvance to the next rung: {', '.join(r['name'] for r in rows[:args.keep])}")
    if args.choose:
        r = rows[0]
        c = r["cfg"]
        flags = (f"--slot_source vit --slot_layer {c.get('slot_layer', -1)}" if c.get("source", "vit") == "vit"
                 else f"--slot_source {c['source']}") + f" --recon_weight {r['recon']}"
        open("stage0_choice.txt", "w").write(flags + "\n")
        print(f"\nChosen: {r['name']}. Wrote stage0_choice.txt: {flags}")


if __name__ == "__main__":
    main()
