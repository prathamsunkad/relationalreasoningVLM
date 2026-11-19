"""Draw the relational-branch figures (H1-H13, A2) from the records in results/ only.

    python tools/plot_relational.py            # writes results/plots/*.png
Needs numpy + matplotlib (the PACE container has both): see slurm/plots.sbatch.
"""
import glob
import gzip
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RR = f"{ROOT}/results/relational/run_records"
LR = f"{ROOT}/results/lora/run_records"
T5 = f"{ROOT}/results/relational/test5000"
OUT = f"{ROOT}/results/plots"
os.makedirs(OUT, exist_ok=True)
plt.rcParams.update({"figure.dpi": 130, "font.size": 9, "axes.grid": True, "grid.alpha": 0.3})


def rows(run, base=RR, fname="metrics.jsonl.gz"):
    p = f"{base}/{run}/{fname}"
    if not os.path.exists(p):
        return []
    with gzip.open(p, "rt") as f:
        return [json.loads(line) for line in f if line.strip()]


def val(run, src=None, key=None):
    """Validation curve of a relational run: (steps, accuracy %). key None = llm_acc if logged, else aux_acc."""
    out = []
    for r in rows(run):
        if "val" not in r or (src and r["val"] != src):
            continue
        k = key or ("llm_acc" if "llm_acc" in r else "aux_acc")
        if k in r:
            out.append((r["step"], 100 * r[k]))
    return [s for s, _ in out], [a for _, a in out]


def lora_val(run):
    out = [(r["step"], 100 * r["val_acc"]) for r in rows(run, LR) if "val_acc" in r]
    return [s for s, _ in out], [a for _, a in out]


def curve(ax, xy, label, **kw):
    x, y = xy
    if x:
        ax.plot([s / 1000 for s in x], y, label=label, **kw)


def finish(fig, axes, name, title, chance=True):
    for ax in axes:
        ax.set_xlabel("training step (thousands)")
        ax.set_ylabel("accuracy (%)")
        if chance:
            ax.axhline(12.5, color="grey", lw=0.8, ls=":")
        ax.legend(fontsize=7)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(f"{OUT}/{name}.png")
    plt.close(fig)
    print("wrote", name)


# H1 + H3: memorisation of 500 puzzles (scored on the puzzles trained on)
fig, axs = plt.subplots(1, 2, figsize=(11, 4))
for r in sorted(glob.glob(f"{RR}/overfit*_*")):
    r = os.path.basename(r)
    curve(axs[0], val(r, "fit"), r)
axs[0].set_title("H1: frozen LLM, 24-layer branch (rounds 1 and 2)")
for r in ["llm_ovf_q_int_lr3e-4", "llm_ovf_q_int_lr1e-3", "llm_ovf_q_fuse_lr3e-4", "llm_ovf_q_fuse_lr1e-3"]:
    curve(axs[1], val(r, "fit"), r)
curve(axs[1], val("aux_overfit_qwen_L6_lr3e-4", "fit", "aux_acc"), "branch alone, L6 3e-4 (no LLM)", ls="--", color="k")
axs[1].set_title("H3: frozen LLM, 6-layer branch")
finish(fig, axs, "H1_H3_memorisation", "Memorising 500 PGM puzzles (accuracy on those puzzles)")

# H2 + H6: branch trained through the frozen LLM, against the branch alone
fig, axs = plt.subplots(1, 2, figsize=(11, 4))
for r in ["s1_qwen_interleave", "s1_qwen_fuse", "s1_ivl1b_interleave", "s1_ivl1b_fuse"]:
    curve(axs[0], val(r, "pgm"), r)
axs[0].set_title("H2: 1 epoch, 24 layers, untuned")
for r in ["s1_q_int_pool_lr3e-4_ep10", "s1_q_int_pool_lr1e-3_ep10", "s1_q_fuse_lr3e-4_ep10", "s1_q_fuse_lr1e-3_ep10"]:
    curve(axs[1], val(r, "pgm"), r)
curve(axs[1], val("aux_full_qwen_L6_lr3e-4_ep10", "pgm", "aux_acc"), "branch alone, frozen slots", ls="--", color="k")
curve(axs[1], val("aux_ft_q_L6_lr3e-4_ep10", "pgm", "aux_acc"), "branch alone, fine-tuned slots", ls="--", color="tab:red")
axs[1].set_title("H6: toward 10 epochs, 6 layers, 256-dim (stopped)")
finish(fig, axs, "H2_H6_frozen_llm", "Branch trained through a frozen LLM (1,000 validation puzzles)")

# H4: full-data hybrids with LoRA from scratch
fig, ax = plt.subplots(figsize=(7.5, 4.5))
curve(ax, lora_val("qwen_pc_recipe_e1"), "LoRA alone (full data)", color="k", lw=2)
for r in ["lora_q_int_pool_lr3e-4", "lora_q_int_panel_lr3e-4", "lora_q_int_slot_lr3e-4", "lora_q_int_slot_lr1e-3",
          "lora_q_fuse_lr3e-4", "lora_q_fuse_lr1e-3", "lora_q_int_slot_control_lr3e-4"]:
    curve(ax, val(r, "pgm"), r, ls="--" if "control" in r else "-")
finish(fig, [ax], "H4_full_data_hybrids", "H4: full-data hybrids vs LoRA alone (validation)")

# H5: mixture hybrids, per task
fig, axs = plt.subplots(1, 3, figsize=(13, 4))
for i, src in enumerate(["pgm", "clevr", "vsr"]):
    for r in ["lora_mix30_q_int_slot", "lora_mix30_q_fuse", "lora_mix30_q_int_slot_control"]:
        curve(axs[i], val(r, src), r)
    axs[i].set_title(src.upper())
finish(fig, axs, "H5_mixture_hybrids", "H5: mixture hybrids with LoRA (validation by task)")

# H7 / H10 / H12 / H13 alignments: the frozen LLM reads the branch
fig, ax = plt.subplots(figsize=(8, 4.5))
for tag, ref, col in [("L6ft", 37.1, "tab:blue"), ("L6ft50", 40.9, "tab:orange"), ("L6ft276k", 47.6, "tab:green"),
                      ("L6ftc60e", 28.9, "tab:red")]:
    for m, ls in [("int", "-"), ("fuse", "--")]:
        curve(ax, val(f"align_q_{m}_{tag}_lr3e-4", "pgm"), f"align {m} {tag}", color=col, ls=ls)
    ax.axhline(ref, color=col, lw=0.8, ls=":")
finish(fig, [ax], "H7_alignment", "Alignment (connector only): dotted line = the branch's own accuracy")

# H8 / H9 / H10: Stage 2 at 5% data
fig, axs = plt.subplots(1, 2, figsize=(11, 4))
for i, m in enumerate(["int", "fuse"]):
    curve(axs[i], lora_val("qwen_pc_neutral5pct_recipe"), "LoRA alone 5%", color="k", lw=2)
    curve(axs[i], val(f"lora_w5_q_{m}_L6ft", "pgm"), "H8 no dropout")
    curve(axs[i], val(f"lora_w5d_q_{m}_L6ft", "pgm"), "H9 dropout 0.5")
    curve(axs[i], val(f"lora_w5d_q_{m}_L6ft50", "pgm"), "H10 dropout 0.5, step-204k branch")
    axs[i].set_title({"int": "interleave", "fuse": "fusion"}[m])
finish(fig, axs, "H8_H9_H10_stage2", "Stage 2 at 5% data (validation)")

# H11 / H12 / H13: LoRA-first
fig, axs = plt.subplots(1, 2, figsize=(11, 4))
for i, m in enumerate(["int", "fuse"]):
    curve(axs[i], val(f"lora_lf_q_{m}_ft50_ctrl", "pgm"), "control (branch off)", color="k", lw=2)
    curve(axs[i], val(f"lora_lf_q_{m}_ft50", "pgm"), "H11 step-204k branch")
    curve(axs[i], val(f"lora_lf_q_{m}_ft276k", "pgm"), "H12 step-276k branch")
    curve(axs[i], val(f"lora_lf_q_{m}_c60e", "pgm"), "H13 60k-only branch (step 43k)")
    axs[i].set_title({"int": "interleave", "fuse": "fusion"}[m])
finish(fig, axs, "H11_H12_H13_lora_first", "LoRA-first from the LoRA-alone adapter (validation)", chance=False)

# H13: Stage 2 recipes on the 60k-only branch
fig, axs = plt.subplots(1, 2, figsize=(11, 4))
for i, m in enumerate(["int", "fuse"]):
    curve(axs[i], lora_val("qwen_pc_neutral5pct_recipe"), "LoRA alone 5%", color="k", lw=2)
    curve(axs[i], val(f"lora_w5_q_{m}_c60e", "pgm"), "H8 recipe")
    curve(axs[i], val(f"lora_w5d_q_{m}_c60e", "pgm"), "H9 recipe")
    axs[i].set_title({"int": "interleave", "fuse": "fusion"}[m])
finish(fig, axs, "H13_stage2_60k_branch", "H13: Stage 2 recipes with the 60k-only branch (validation)")

# Branch alone (Stage 1a) runs behind the hybrids
fig, ax = plt.subplots(figsize=(8, 4.5))
for r in ["aux_full_qwen_L6_lr3e-4_ep10", "aux_ft_q_L6_lr3e-4_ep10", "aux_ft_q_L6_lr3e-4_ep50", "aux_ft_q_L24_lr8e-5_ep50",
          "aux_ft_q_L6_lr3e-4_settle", "aux_ft60k_q_L6_lr3e-4", "aux_ft60k_q_L6_lr1e-3"]:
    curve(ax, val(r, "pgm", "aux_acc"), r)
finish(fig, [ax], "branch_alone_curves", "Branch alone (linear head, no LLM): validation")

# Stage 0 slot models: reconstruction error
fig, ax = plt.subplots(figsize=(7.5, 4))
for d in sorted(glob.glob(f"{RR}/s0_*")):
    r = os.path.basename(d)
    pts = [(x["step"], x["val_mse"]) for x in rows(r, fname="log.jsonl.gz") if "val_mse" in x]
    if pts:
        ax.plot([p[0] / 1000 for p in pts], [p[1] for p in pts], label=r)
ax.set_xlabel("training step (thousands)")
ax.set_ylabel("validation reconstruction MSE")
ax.legend(fontsize=6)
ax.set_title("Stage 0 slot attention: reconstruction (lower is better)")
fig.tight_layout()
fig.savefig(f"{OUT}/stage0_slots.png")
plt.close(fig)


# A2: 5,000-puzzle test accuracy with 95% intervals
def t5(name):
    p = f"{T5}/{name}.json.gz"
    if not os.path.exists(p):
        return None
    with gzip.open(p, "rt") as f:
        s = json.load(f)["summary"]
    m = s["llm"] if s.get("llm") and s["llm"].get("acc") is not None else s["aux"]
    return 100 * m["acc"], [100 * c for c in m["ci95"]]


GROUPS = [("LoRA alone", ["lora5"]), ("branch alone", ["br10", "br204k", "brset", "br276k", "br60e"]),
          ("H8", ["w5_int", "w5_fuse"]), ("H9", ["w5d_int", "w5d_fuse"]), ("H10", ["w5d50_int", "w5d50_fuse"]),
          ("H11 controls", ["lf_int_ctrl", "lf_fuse_ctrl"]), ("H11", ["lf_int", "lf_fuse"]), ("H12", ["lf276_int", "lf276_fuse"]),
          ("H13 H8-recipe", ["c60e_w5_int", "c60e_w5_fuse"]), ("H13 H9-recipe", ["c60e_w5d_int", "c60e_w5d_fuse"]),
          ("H13 H12-recipe", ["c60e_lf_int", "c60e_lf_fuse"])]
labels, accs, errs, cols = [], [], [], []
palette = [plt.get_cmap("tab20")(i) for i in range(0, 20, 2)] + [plt.get_cmap("tab20")(i) for i in range(1, 20, 2)]
for gi, (g, names) in enumerate(GROUPS):
    for n in names:
        r = t5(n)
        if r:
            labels.append(f"{g}: {n}")
            accs.append(r[0])
            errs.append([r[0] - r[1][0], r[1][1] - r[0]])
            cols.append(palette[gi % len(palette)])
fig, ax = plt.subplots(figsize=(8, 0.28 * len(labels) + 1.5))
y = list(range(len(labels)))[::-1]
ax.barh(y, accs, xerr=list(zip(*errs)), color=cols, capsize=2)
ax.set_yticks(y)
ax.set_yticklabels(labels, fontsize=7)
lora = t5("lora5")
if lora:
    ax.axvline(lora[0], color="k", ls="--", lw=1)
ax.set_xlabel("test accuracy on 5,000 PGM puzzles (%), 95% interval")
ax.set_title("A2: test scores (dashed = LoRA alone at 5% data)")
fig.tight_layout()
fig.savefig(f"{OUT}/A2_test5000.png")
plt.close(fig)

# Gain over the LoRA-first control vs the branch's own test accuracy
pts = [("br60e", "c60e_lf", "60k-only, step 43k"), ("br204k", "lf", "step 204k"), ("br276k", "lf276", "step 276k")]
fig, ax = plt.subplots(figsize=(6, 4))
for m, mk in [("int", "o"), ("fuse", "s")]:
    xs, ys = [], []
    for br, h, _ in pts:
        b, a, c = t5(br), t5(f"{h}_{m}"), t5(f"lf_{m}_ctrl")
        if b and a and c:
            xs.append(b[0])
            ys.append(a[0] - c[0])
    ax.plot(xs, ys, marker=mk, label=f"LoRA-first {m}")
for br, h, lab in pts:
    b, a, c = t5(br), t5(f"{h}_fuse"), t5("lf_fuse_ctrl")
    if b and a and c:
        ax.annotate(lab, (b[0], a[0] - c[0]), textcoords="offset points", xytext=(0, 10), ha="center", fontsize=7)
ax.axhline(0, color="grey", lw=0.8)
ax.set_xlabel("branch alone, test accuracy (%)")
ax.set_ylabel("hybrid minus control (points)")
ax.set_title("The gain grows with the branch's own strength")
ax.legend(fontsize=7)
fig.tight_layout()
fig.savefig(f"{OUT}/gain_vs_branch_strength.png")
plt.close(fig)
print("all figures in", OUT)
