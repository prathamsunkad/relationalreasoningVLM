"""A3: do the interleave models actually use their relation tokens? Reads <run>__<condition>.json (eval.py per-puzzle
records on the 5,000 test puzzles) in this folder and writes report.md: per run, accuracy under each condition, the
change from the unablated score, and the exact McNemar p-value on the same puzzles.

    python3 analyze_ablate.py [--dir DIR]
"""
import argparse
import glob
import json
import math
import os
from collections import defaultdict

ap = argparse.ArgumentParser()
ap.add_argument("--dir", default=os.path.dirname(os.path.abspath(__file__)))
D = ap.parse_args().dir

EXP = {  # run -> (experiment, short description)
    "lora_q_int_pool_lr3e-4": ("H4", "full data, LoRA from scratch, 1 token"),
    "lora_q_int_panel_lr3e-4": ("H4", "full data, 9 tokens (per panel)"),
    "lora_q_int_slot_lr3e-4": ("H4", "full data, 144 tokens (per slot)"),
    "lora_q_int_slot_lr1e-3": ("H4", "full data, 144 tokens, Abstractor lr 1e-3"),
    "lora_q_int_slot_control_lr3e-4": ("H4", "full data, 144 tokens, non-relational control"),
    "align_q_int_L6ft_lr3e-4": ("H7", "alignment, frozen LLM, 10-epoch branch, lr 3e-4"),
    "align_q_int_L6ft_lr1e-3": ("H7", "alignment, frozen LLM, 10-epoch branch, lr 1e-3"),
    "lora_w5_q_int_L6ft": ("H8", "Stage 2 at 5%, no dropout"),
    "lora_w5d_q_int_L6ft": ("H9", "Stage 2 at 5%, dropout 0.5"),
    "align_q_int_L6ft50_lr3e-4": ("H10", "alignment, frozen LLM, step-204k branch"),
    "lora_w5d_q_int_L6ft50": ("H10", "Stage 2 at 5%, dropout 0.5, step-204k branch"),
    "lora_lf_q_int_ft50": ("H11", "LoRA-first, step-204k branch"),
    "align_q_int_L6ft276k_lr3e-4": ("H12", "alignment, frozen LLM, step-276k branch"),
    "lora_lf_q_int_ft276k": ("H12", "LoRA-first, step-276k branch"),
}
CONDS = ["normal", "zero", "drop", "shuffle_cand", "shuffle_puzzle", "permute"]
ORDER = ["H4", "H7", "H8", "H9", "H10", "H11", "H12"]


def load(path):
    js = json.load(open(path))
    return {r["id"]: int(r["llm_pred"] == r["label"]) for r in js["records"]}, js.get("ckpt_step")


def mcnemar(a, b):
    """Exact two-sided McNemar p on paired 0/1 outcomes (dicts keyed by puzzle id)."""
    ids = a.keys() & b.keys()
    n01 = sum(1 for i in ids if a[i] and not b[i])
    n10 = sum(1 for i in ids if b[i] and not a[i])
    n = n01 + n10
    if n == 0:
        return 1.0, n01, n10
    k = min(n01, n10)
    p = 2 * sum(math.comb(n, j) for j in range(k + 1)) / 2 ** n
    return min(1.0, p), n01, n10


res = defaultdict(dict)
steps = {}
for f in sorted(glob.glob(f"{D}/*__*.json")):
    run, cond = os.path.basename(f)[:-5].split("__")
    res[run][cond], steps[run] = load(f)

L = ["# A3: do the interleave models use their relation tokens?", "",
     "5,000 test puzzles (`test5000.tsv`, as in A2); every condition of a run scored in the same job on one GPU type.",
     "Chance = 12.5%. Change = condition minus normal; p = exact McNemar on the same puzzles.", "",
     "| Condition | What it does |", "|---|---|",
     "| zero | relation tokens set to 0, placeholders kept (= `--branch_off`) |",
     "| drop | placeholders removed from the prompt |",
     "| shuffle_cand | each candidate gets another candidate's tokens from the same puzzle |",
     "| shuffle_puzzle | each puzzle gets another puzzle's tokens (same candidate slot) |",
     "| permute | the 9 / 144 tokens of a sequence in a fixed random order |", "",
     "## Summary (accuracy %)", "",
     "| Exp. | Run | " + " | ".join(CONDS) + " |", "|---|---|" + "---|" * len(CONDS)]
runs = sorted(res, key=lambda r: (ORDER.index(EXP.get(r, ("H99",))[0]) if EXP.get(r, ("H99",))[0] in ORDER else 99, r))
for run in runs:
    exp = EXP.get(run, ("?", ""))[0]
    cells = []
    for c in CONDS:
        if c in res[run]:
            acc = 100 * sum(res[run][c].values()) / len(res[run][c])
            if c != "normal" and "normal" in res[run]:
                base = 100 * sum(res[run]["normal"].values()) / len(res[run]["normal"])
                cells.append(f"{acc:.1f} ({acc - base:+.1f})")
            else:
                cells.append(f"**{acc:.1f}**")
        else:
            cells.append("")
    L.append(f"| {exp} | `{run}` | " + " | ".join(cells) + " |")
L += ["", "## Paired tests against the unablated score", "",
      "| Exp. | Run | Condition | Normal | Ablated | Change | Only normal right | Only ablated right | p |",
      "|---|---|---|---|---|---|---|---|---|"]
for run in runs:
    if "normal" not in res[run]:
        continue
    exp = EXP.get(run, ("?", ""))[0]
    a = res[run]["normal"]
    an = 100 * sum(a.values()) / len(a)
    for c in CONDS[1:]:
        if c not in res[run]:
            continue
        b = res[run][c]
        bn = 100 * sum(b.values()) / len(b)
        p, n01, n10 = mcnemar(a, b)
        L.append(f"| {exp} | `{run}` | {c} | {an:.1f} | {bn:.1f} | **{bn - an:+.1f}** | {n01} | {n10} | {p:.2g} |")
L += ["", "## Runs", "", "| Exp. | Run | What it is | Checkpoint step |", "|---|---|---|---|"]
for run in runs:
    exp, what = EXP.get(run, ("?", ""))
    L.append(f"| {exp} | `{run}` | {what} | {steps.get(run)} |")
open(f"{D}/report.md", "w").write("\n".join(L) + "\n")
print(f"report.md: {len(runs)} runs, {sum(len(v) for v in res.values())} scored conditions")
