"""Plot validation accuracy, validation loss and training loss over epochs for a run (per-candidate LoRA on PGM).

  python tools/plot_curves.py --run_dir $RUNS/ivl1b_pc_recipe_e1 --label "InternVL3.5-1B" \
      --test 1=0.4023,2=0.4853,3=0.5231 --val_loss val_loss_ivl.json \
      --phases "3-4:cosine decay 1e-4 -> 1e-5;4-6:warm restart 1e-4 -> 1e-5" --out internvl_curves.png
--val_loss is the JSON written by tools/val_loss.py (validation loss of the saved end-of-epoch checkpoints); without
it the figure has two panels (validation accuracy, training loss).
"""
import argparse
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

PHASE_COLORS = ["#f2c94c", "#9ad0c2", "#c3b1e1"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", required=True)
    ap.add_argument("--label", default="")
    ap.add_argument("--steps_per_epoch", type=int, default=15000)
    ap.add_argument("--window", type=int, default=250, help="updates averaged per training-loss point")
    ap.add_argument("--test", default="", help="epoch=accuracy pairs, e.g. 1=0.40,2=0.49")
    ap.add_argument("--val_loss", default=None, help="JSON from tools/val_loss.py")
    ap.add_argument("--phases", default="", help='shaded LR phases: "start-end:label;start-end:label" in epochs')
    ap.add_argument("--decay_from_epoch", type=int, default=0, help="(older option) same as --phases 'N-end:cosine decay'")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    train, val = {}, {}
    for line in open(f"{a.run_dir}/metrics.jsonl"):
        r = json.loads(line)
        if "ce8" in r:
            train[r["step"]] = r          # later entries win: steps redone after a resume count once
        if "val_acc" in r:
            val[r["step"]] = r["val_acc"]
    E = a.steps_per_epoch
    last = max(train)
    n_ep = round(last / E)
    xs, loss = [], []
    for lo in range(0, last, a.window):
        w = [train[s]["ce8"] for s in train if lo < s <= lo + a.window]
        if w:
            xs.append((lo + a.window) / E)
            loss.append(sum(w) / len(w))
    vx = [s / E for s in sorted(val)]
    vy = [100 * val[s] for s in sorted(val)]
    test = {int(k): 100 * float(v) for k, v in (p.split("=") for p in a.test.split(",") if p)}
    phases = []
    for p in (x for x in a.phases.split(";") if x.strip()):
        span, lab = p.split(":", 1)
        s0, s1 = span.split("-")
        phases.append((float(s0), float(s1), lab.strip().replace("->", "→")))
    if a.decay_from_epoch and not phases:
        phases.append((a.decay_from_epoch, n_ep, "cosine LR decay 1e-4 → 1e-5"))
    vl = {int(k): v for k, v in json.load(open(a.val_loss)).items()} if a.val_loss else {}

    n_panels = 3 if vl else 2
    fig, axs = plt.subplots(1, n_panels, figsize=(6.6 * n_panels, 5.0), dpi=150)
    ax1, ax_tr = axs[0], axs[-1]

    ax1.plot(vx, vy, "-o", ms=3, lw=1.8, color="#2b6cb0", label="Validation accuracy (1,000 problems, every 1k updates)")
    if test:
        ax1.plot(list(test), list(test.values()), "D", ms=8, color="#d9822b", label="Test accuracy (200k problems)")
        for e, v in test.items():
            ax1.annotate(f"{v:.1f}%", (e, v), textcoords="offset points", xytext=(-8, 9), ha="right", fontsize=9, color="#d9822b")
    ax1.axhline(12.5, ls=":", color="grey", lw=1)
    ax1.text(0.05, 13.3, "chance 12.5%", fontsize=8, color="grey")
    ax1.set_ylabel("Accuracy (%)")
    ax1.set_ylim(10, max(vy + list(test.values())) + 6)
    ax1.set_title("Validation accuracy")
    ax1.legend(loc="lower right", fontsize=8)

    if vl:
        ax2 = axs[1]
        ep = sorted(vl)
        ax2.plot(ep, [vl[e]["val_loss"] for e in ep], "-s", ms=7, lw=1.8, color="#2b6cb0",
                 label="Validation loss (saved end-of-epoch checkpoints)")
        tr_end = []
        for e in ep:
            w = [train[s]["ce8"] for s in train if e * E - 1000 < s <= e * E]
            tr_end.append(sum(w) / len(w) if w else float("nan"))
        ax2.plot(ep, tr_end, "--o", ms=5, lw=1.2, color="#718096", label="Training loss (last 1k updates of the epoch)")
        for e in ep:
            ax2.annotate(f"{vl[e]['val_loss']:.3f}", (e, vl[e]["val_loss"]), textcoords="offset points", xytext=(0, 9),
                         ha="center", fontsize=8, color="#2b6cb0")
        ax2.axhline(2.079, ls="--", color="#c53030", lw=1)
        ax2.text(0.05, 2.06, "random guessing (ln 8 = 2.079)", fontsize=8, color="#c53030", va="top")
        ax2.set_ylabel("8-way cross-entropy")
        ax2.set_title("Validation loss vs training loss")
        ax2.set_ylim(min(tr_end + [vl[e]["val_loss"] for e in ep]) - 0.15, 2.2)
        ax2.legend(loc="lower left", fontsize=8)

    ax_tr.plot(xs, loss, lw=1.6, color="#2b6cb0", label=f"Training loss (mean of every {a.window} updates)")
    ax_tr.axhline(2.079, ls="--", color="#c53030", lw=1)
    ax_tr.text(0.05, 2.06, "random guessing (ln 8 = 2.079)", fontsize=8, color="#c53030", va="top")
    ax_tr.set_ylabel("8-way cross-entropy")
    ax_tr.set_title("Training loss")
    ax_tr.legend(loc="lower left", fontsize=8)

    for ax in axs:
        for i, (s0, s1, lab) in enumerate(phases):
            ax.axvspan(s0, s1, color=PHASE_COLORS[i % len(PHASE_COLORS)], alpha=0.2, lw=0)
            ax.text(s0 + 0.04, 0.5, lab.replace(" 1e", "\n1e", 1), transform=ax.get_xaxis_transform(),
                    va="top", fontsize=7.5, color="#5a4a00")
        for e in range(1, n_ep):
            ax.axvline(e, ls="--", color="lightgrey", lw=1)
        ax.set_xlim(0, n_ep + 0.1)
        ax.set_xticks(range(n_ep + 1))
        ax.set_xlabel(f"Epoch ({E:,} updates = 240k problems)")
        ax.grid(alpha=0.25)
    first = phases[0][0] if phases else n_ep
    lr_note = f"constant LR 1e-4 for epochs 1-{int(first)}" + ("; shaded: LR schedule changes" if phases else "")
    fig.suptitle(f"{a.label} — per-candidate LoRA on PGM neutral ({lr_note})", fontsize=12)
    fig.tight_layout()
    fig.savefig(a.out)
    print("saved", a.out)


if __name__ == "__main__":
    main()
