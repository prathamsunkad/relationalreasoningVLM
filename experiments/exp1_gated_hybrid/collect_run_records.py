#!/usr/bin/env python
"""
Status of every Exp1 run, and copies of their small log files for git (the checkpoints stay out).

  python collect_run_records.py                 # table of every run under ckpt/ (read-only)
  python collect_run_records.py --copy          # also copy train_log.jsonl + summary.json into run_records/
  python collect_run_records.py --copy --plot   # and run_records/collapse_evidence.png (needs matplotlib)

Each run folder writes train_log.jsonl: one line per log step with "spread" (mean within-puzzle spread of the 8
candidate scores; 0 means every candidate got the same score = collapse), "ce", "acc", "attn_max", ..., and one
{"val": {...}} line per validation. ckpt/ is git-ignored because it also holds the .pt checkpoints; run_records/
keeps only the logs, so the collapse and depth results can be checked from the repo.
"""
import argparse
import json
import os
import shutil

COLLAPSED = 0.01   # median spread over the last 10% of logged steps below this = collapsed (all 8 scored alike)


def read(path):
    out = []
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if line:
                try:
                    out.append(json.loads(line))
                except json.JSONDecodeError:
                    pass       # a line cut off by a killed job
    return out


def describe(run_dir):
    recs = read(os.path.join(run_dir, "train_log.jsonl"))
    tr = [r for r in recs if "step" in r and "val" not in r and "event" not in r]
    vals = [r["val"] for r in recs if "val" in r]
    sp = [r["spread"] for r in tr if "spread" in r]
    best = min(vals, key=lambda v: v["loss"]) if vals else None
    tail = sorted(sp[-max(1, len(sp) // 10):])         # the last 10% of logged steps
    med = tail[len(tail) // 2] if tail else None
    return {
        "finished": os.path.exists(os.path.join(run_dir, "summary.json")),
        "step": max([r["step"] for r in tr] + [v.get("step", 0) for v in vals] + [0]),
        "of": tr[-1].get("of") if tr else None,
        "best": best,
        "spread_first": sp[0] if sp else None,
        "spread_last": sp[-1] if sp else None,
        "spread_end": med,
        "collapsed": med is not None and med < COLLAPSED,
        "n_vals": len(vals),
    }


def fmt(x, f="{:.3f}"):
    return "" if x is None else f.format(x)


# ---- the C6 figure: collapse before the fix vs no collapse after it
FIX, NOFIX, INK, MUTED, GRID = "#257de0", "#f47230", "#1f1f1f", "#6b6b6b", "#e6e6e6"
CHANCE = 2.0794   # ln 8: the held-out loss of uniform scores


def _curves(run_dir, smooth=9):
    recs = read(os.path.join(run_dir, "train_log.jsonl"))
    tr = [r for r in recs if "spread" in r and "val" not in r]
    xs, ys = [r["step"] for r in tr], [r["spread"] for r in tr]
    k = smooth // 2                                    # running median: shows the level, not the step-to-step noise
    ys = [sorted(ys[max(0, i - k):i + k + 1])[len(ys[max(0, i - k):i + k + 1]) // 2] for i in range(len(ys))]
    vals = [r["val"] for r in recs if "val" in r]
    return xs, ys, [v["step"] for v in vals], [v["loss"] for v in vals]


def plot_collapse(ckpt, out, check=(("collapse/cnn_old", "no qk_norm", NOFIX, "-"),
                                    ("collapse/cnn_fix_r01", "qk_norm, recon 0.1", NOFIX, "--"),
                                    ("collapse/cnn_fix", "qk_norm", FIX, "-")),
                  before="sh/seed0", after="sh_v2/seed0", upto=20000):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 9, "text.color": INK, "axes.labelcolor": INK, "xtick.color": MUTED,
                         "ytick.color": MUTED, "axes.edgecolor": GRID, "axes.titlesize": 10})
    fig, ax = plt.subplots(2, 2, figsize=(10, 6.2), sharex="col")
    for a in ax.flat:
        a.grid(True, color=GRID, linewidth=0.6)
        a.set_axisbelow(True)
        for side in ("top", "right"):
            a.spines[side].set_visible(False)
        a.axhline(0, color=GRID, linewidth=0.8) if a in ax[0] else None

    # left: the 6k-step collapse check, three named runs, labelled at their line ends
    ax[0, 0].set_title("A. Collapse check (CNN slots, 6k steps)", loc="left")
    for name, label, color, ls in check:
        d = os.path.join(ckpt, name)
        if not os.path.exists(os.path.join(d, "train_log.jsonl")):
            continue
        xs, ys, vx, vy = _curves(d)
        ax[0, 0].plot(xs, ys, color=color, linestyle=ls, linewidth=2)
        ax[1, 0].plot(vx, vy, color=color, linestyle=ls, linewidth=2, marker="o", markersize=4, label=label)

    # right: the slot-source search, 7 sources before the fix vs 7 after, first `upto` steps
    ax[0, 1].set_title(f"B. Slot-source search, first {upto // 1000}k steps (7 sources each)", loc="left")
    for group, color, ls, label in ((before, NOFIX, "--", "before the fix"),
                                    (after, FIX, "-", "after the fix (qk_norm, lr 1e-4)")):
        gdir = os.path.join(ckpt, group)
        first = True
        for run in sorted(os.listdir(gdir)) if os.path.isdir(gdir) else []:
            d = os.path.join(gdir, run)
            if not os.path.exists(os.path.join(d, "train_log.jsonl")):
                continue
            xs, ys, vx, vy = _curves(d)
            keep = [i for i, x in enumerate(xs) if x <= upto]
            vkeep = [i for i, x in enumerate(vx) if x <= upto]
            kw = dict(color=color, linestyle=ls, linewidth=1.2, alpha=0.85, label=label if first else None)
            ax[0, 1].plot([xs[i] for i in keep], [ys[i] for i in keep], **kw)
            ax[1, 1].plot([vx[i] for i in vkeep], [vy[i] for i in vkeep], marker="o", markersize=3, **kw)
            first = False
    for a in ax[1]:   # one legend per column, in the loss panel's empty lower left (lines are the same above)
        a.legend(frameon=True, framealpha=0.92, edgecolor=GRID, loc="lower left", fontsize=8)

    for a in ax[1]:
        a.axhline(CHANCE, color=MUTED, linewidth=1, linestyle=":")
        a.annotate("chance (ln 8)", (0, CHANCE), xycoords=("axes fraction", "data"), xytext=(2, 3),
                   textcoords="offset points", ha="left", va="bottom", fontsize=8, color=MUTED)
        a.set_xlabel("training step")
    ax[0, 0].set_ylabel("spread of the 8 candidate scores\n(0 = all scored alike: collapse)")
    ax[1, 0].set_ylabel("held-out loss (10,000 puzzles)")
    fig.suptitle("Exp1 stage 0: the score collapse and its fix", x=0.01, ha="left", fontsize=11, color=INK)
    fig.text(0.01, 0.005, "Spread: running median over 9 log steps. Held-out = last 10,000 rows of train.jsonl. "
             "Source: run_records/ (train_log.jsonl of each run).", fontsize=7, color=MUTED)
    fig.tight_layout(rect=(0, 0.02, 1, 0.97))
    fig.savefig(out, dpi=170, facecolor="white")
    print(f"wrote {out}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", default="ckpt")
    ap.add_argument("--out", default="run_records")
    ap.add_argument("--copy", action="store_true", help="copy the logs into --out and write --out/README.md")
    ap.add_argument("--plot", action="store_true",
                    help="also draw the collapse figure (--out/collapse_evidence.png)")
    ap.add_argument("--skip", default="pipeline_smoke", help="comma-separated folder names to leave out")
    a = ap.parse_args()
    skip = set(filter(None, a.skip.split(",")))

    runs = []
    for root, dirs, files in os.walk(a.ckpt):
        dirs[:] = sorted(d for d in dirs if d not in skip)
        if "train_log.jsonl" in files:
            runs.append((os.path.relpath(root, a.ckpt), describe(root)))
    if not runs:
        raise SystemExit(f"no train_log.jsonl under {a.ckpt}/ (run from experiments/exp1_gated_hybrid)")

    lines = ["| run | state | step / total | best held-out loss (step) | acc at best | spread at end | collapsed |",
             "|---|---|---|---|---|---|---|"]
    for name, d in runs:
        b = d["best"]
        lines.append(
            f"| `{name}` | {'finished' if d['finished'] else 'partial'} | {d['step']} / {d['of'] or '?'} | "
            + (f"{b['loss']:.4f} ({b.get('step', '?')})" if b else "") + " | "
            + (f"{b['acc']:.1%}" if b and "acc" in b else "") + " | "
            + f"{fmt(d['spread_end'])} | {'**yes**' if d['collapsed'] else ('no' if d['spread_end'] is not None else 'not logged')} |")
    table = "\n".join(lines)
    print(table)
    print(f"\n{len(runs)} runs. Chance: loss ln 8 = 2.079, accuracy 12.5%. Spread at end = median over the last 10% "
          f"of logged steps; collapsed = below {COLLAPSED}.")

    if not a.copy:
        return
    os.makedirs(a.out, exist_ok=True)
    total = 0
    for name, _ in runs:
        src = os.path.join(a.ckpt, name)
        dst = os.path.join(a.out, name)
        os.makedirs(dst, exist_ok=True)
        for f in ("train_log.jsonl", "summary.json"):
            if os.path.exists(os.path.join(src, f)):
                shutil.copy2(os.path.join(src, f), os.path.join(dst, f))
                total += os.path.getsize(os.path.join(dst, f))
    with open(os.path.join(a.out, "README.md"), "w") as fh:
        fh.write("# Exp1 run records\n\nThe logs of every Exp1 run (`train_log.jsonl`, and `summary.json` when the run "
                 "finished), copied from `ckpt/` by `collect_run_records.py`. Checkpoints are not included.\n\n"
                 "`spread` = mean within-puzzle spread of the 8 candidate scores (0 = every candidate scored alike, "
                 f"i.e. collapse). Spread at end = its median over the last 10% of logged steps; collapsed = below "
                 f"{COLLAPSED}. Runs without a spread column predate its logging. "
                 "Held-out = the last 10,000 rows of `train.jsonl`, never trained on.\n\n" + table + "\n")
    print(f"\ncopied the logs of {len(runs)} runs into {a.out}/ ({total / 1e6:.1f} MB) and wrote {a.out}/README.md")

    if a.plot:
        plot_collapse(a.ckpt, os.path.join(a.out, "collapse_evidence.png"))


if __name__ == "__main__":
    main()
