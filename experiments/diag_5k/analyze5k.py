"""Phase 1, step 4: the finished runs on 5,000 unseen PGM test problems (diag_5k.sbatch writes <name>.json here).
Writes report.md: each model's accuracy with a 95% interval; paired comparisons (problem by problem, exact McNemar),
most importantly each hybrid against its own branch-off control; the late-fusion reference (log-softmax average of
LoRA and the settled branch, w = 0.5, untuned); and, if rules.json exists, the hybrid-vs-control gap per attribute.
Usage: python3 analyze5k.py [--dir D]   (numpy only; runs on the login node)"""
import argparse
import glob
import json
import math
import os

import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--dir", default=os.path.dirname(os.path.abspath(__file__)))
D = ap.parse_args().dir

WHAT = {
    "lora5": "LoRA alone, 5% data, best (vlm_lora_pc adapter)",
    "lf_int": "LoRA first + branch, interleave",
    "lf_int_ctrl": "LoRA first, branch always off (control for lf_int)",
    "lf_int_off": "lf_int scored with the branch off",
    "lf_fuse": "LoRA first + branch, fusion",
    "lf_fuse_ctrl": "LoRA first, branch always off (control for lf_fuse)",
    "lf_fuse_off": "lf_fuse scored with the branch off",
    "brset": "settled branch alone (aux head, no LLM)",
    "w5_int": "Stage 2, 10-epoch branch, interleave, no dropout",
    "w5_fuse": "Stage 2, 10-epoch branch, fusion, no dropout",
    "w5d_int": "Stage 2, 10-epoch branch, interleave, dropout 0.5",
    "w5d_fuse": "Stage 2, 10-epoch branch, fusion, dropout 0.5",
    "w5d_int_off": "w5d_int scored with the branch off",
    "w5d_fuse_off": "w5d_fuse scored with the branch off",
    "w5d50_int": "Stage 2, 50-epoch branch (204k), interleave, dropout 0.5",
    "w5d50_int_off": "w5d50_int scored with the branch off",
    "w5d50_fuse": "Stage 2, 50-epoch branch (204k), fusion, dropout 0.5",
    "w5d50_fuse_off": "w5d50_fuse scored with the branch off",
    "lf276_int": "LoRA first + step-276k branch, interleave",
    "lf276_int_off": "lf276_int scored with the branch off",
    "lf276_fuse": "LoRA first + step-276k branch, fusion",
    "lf276_fuse_off": "lf276_fuse scored with the branch off",
    "c60_lf_int": "Clean 60k: LoRA-first, interleave (H12 recipe, 60k-only branch)",
    "c60_lf_int_off": "c60_lf_int scored with the branch off",
    "c60_lf_fuse": "Clean 60k: LoRA-first, fusion",
    "c60_lf_fuse_off": "c60_lf_fuse scored with the branch off",
    "c60_w5_int": "Clean 60k: Stage 2, no dropout, interleave (H8 recipe)",
    "c60_w5_int_off": "c60_w5_int scored with the branch off",
    "c60_w5_fuse": "Clean 60k: Stage 2, no dropout, fusion",
    "c60_w5_fuse_off": "c60_w5_fuse scored with the branch off",
    "c60_w5d_int": "Clean 60k: Stage 2, dropout 0.5, interleave (H9 recipe)",
    "c60_w5d_int_off": "c60_w5d_int scored with the branch off",
    "c60_w5d_fuse": "Clean 60k: Stage 2, dropout 0.5, fusion",
    "c60_w5d_fuse_off": "c60_w5d_fuse scored with the branch off",
    "br60": "60k-only branch alone (used by c60_*)",
    "c60e_lf_int": "Early 60k (branch step 43k): LoRA-first, interleave (H12 recipe, 60k-only branch)",
    "c60e_lf_int_off": "c60e_lf_int scored with the branch off",
    "c60e_lf_fuse": "Early 60k (branch step 43k): LoRA-first, fusion",
    "c60e_lf_fuse_off": "c60e_lf_fuse scored with the branch off",
    "c60e_w5_int": "Early 60k (branch step 43k): Stage 2, no dropout, interleave (H8 recipe)",
    "c60e_w5_int_off": "c60e_w5_int scored with the branch off",
    "c60e_w5_fuse": "Early 60k (branch step 43k): Stage 2, no dropout, fusion",
    "c60e_w5_fuse_off": "c60e_w5_fuse scored with the branch off",
    "c60e_w5d_int": "Early 60k (branch step 43k): Stage 2, dropout 0.5, interleave (H9 recipe)",
    "c60e_w5d_int_off": "c60e_w5d_int scored with the branch off",
    "c60e_w5d_fuse": "Early 60k (branch step 43k): Stage 2, dropout 0.5, fusion",
    "c60e_w5d_fuse_off": "c60e_w5d_fuse scored with the branch off",
    "br60e": "early 60k-only branch alone (step 43k, used by c60e_*)",
    "br10": "10-epoch branch alone (used by w5_*, w5d_*)",
    "br204k": "50-epoch branch at step 204k alone (used by lf_*, w5d50_*)",
    "br276k": "50-epoch branch at step 276k alone (its best so far)",
}
PAIRS = [  # (A, B, question): positive gap = A better
    ("lf_int", "lf_int_ctrl", "Does the branch help? (interleave, vs its control)"),
    ("lf_fuse", "lf_fuse_ctrl", "Does the branch help? (fusion, vs its control)"),
    ("lf_int_ctrl", "lora5", "Do 3,000 more LoRA steps help on their own?"),
    ("lf_fuse_ctrl", "lora5", "Same, fusion control"),
    ("lf_int_off", "lf_int_ctrl", "Did the hybrid's own LoRA keep up? (interleave)"),
    ("lf_fuse_off", "lf_fuse_ctrl", "Did the hybrid's own LoRA keep up? (fusion)"),
    ("lf_int", "lora5", "Interleave hybrid vs LoRA alone"),
    ("lf_fuse", "lora5", "Fusion hybrid vs LoRA alone"),
    ("w5d_int", "lora5", "Stage 2 dropout interleave vs LoRA alone"),
    ("w5d_fuse", "lora5", "Stage 2 dropout fusion vs LoRA alone"),
    ("w5_int", "lora5", "Stage 2 no-dropout interleave vs LoRA alone"),
    ("w5_fuse", "lora5", "Stage 2 no-dropout fusion vs LoRA alone"),
    ("w5d_int", "w5d_int_off", "Stage 2 dropout interleave: branch on vs off in the same model"),
    ("w5d_fuse", "w5d_fuse_off", "Stage 2 dropout fusion: branch on vs off in the same model"),
    ("w5d_int_off", "lora5", "Stage 2 dropout interleave's own LoRA vs LoRA alone"),
    ("w5d_fuse_off", "lora5", "Stage 2 dropout fusion's own LoRA vs LoRA alone"),
    ("w5d50_int", "lora5", "Stage 2 dropout interleave, stronger branch, vs LoRA alone"),
    ("w5d50_fuse", "lora5", "Stage 2 dropout fusion, stronger branch, vs LoRA alone"),
    ("w5d50_int", "w5d_int", "Does the stronger branch help in Stage 2? (interleave)"),
    ("w5d50_fuse", "w5d_fuse", "Does the stronger branch help in Stage 2? (fusion)"),
    ("w5d50_int", "w5d50_int_off", "Stronger-branch interleave: branch on vs off in the same model"),
    ("w5d50_fuse", "w5d50_fuse_off", "Stronger-branch fusion: branch on vs off in the same model"),
    ("lf_int", "w5d_int", "LoRA-first interleave vs Stage 2 dropout interleave"),
    ("lf276_int", "lf_int_ctrl", "Step-276k branch, LoRA-first interleave, vs the control"),
    ("lf276_fuse", "lf_fuse_ctrl", "Step-276k branch, LoRA-first fusion, vs the control"),
    ("lf276_int", "lora5", "Step-276k LoRA-first interleave vs LoRA alone"),
    ("lf276_fuse", "lora5", "Step-276k LoRA-first fusion vs LoRA alone"),
    ("lf276_int", "lf_int", "Does the stronger branch help LoRA-first? (interleave)"),
    ("lf276_fuse", "lf_fuse", "Does the stronger branch help LoRA-first? (fusion)"),
    ("lf276_int_off", "lf_int_ctrl", "Step-276k interleave: did its own LoRA keep up?"),
    ("lf276_fuse_off", "lf_fuse_ctrl", "Step-276k fusion: did its own LoRA keep up?"),
    ("c60_lf_int", "lf_int_ctrl", "Clean 60k LoRA-first interleave vs the control"),
    ("c60_lf_fuse", "lf_fuse_ctrl", "Clean 60k LoRA-first fusion vs the control"),
    ("c60_lf_int", "lora5", "Clean 60k LoRA-first interleave vs LoRA alone"),
    ("c60_lf_fuse", "lora5", "Clean 60k LoRA-first fusion vs LoRA alone"),
    ("c60_lf_int_off", "lf_int_ctrl", "Clean 60k LoRA-first interleave: did its own LoRA keep up?"),
    ("c60_lf_fuse_off", "lf_fuse_ctrl", "Clean 60k LoRA-first fusion: did its own LoRA keep up?"),
    ("c60_w5d_int", "lora5", "Clean 60k Stage 2 dropout interleave vs LoRA alone"),
    ("c60_w5d_fuse", "lora5", "Clean 60k Stage 2 dropout fusion vs LoRA alone"),
    ("c60_w5d_int", "c60_w5d_int_off", "Clean 60k Stage 2 dropout interleave: branch on vs off"),
    ("c60_w5d_fuse", "c60_w5d_fuse_off", "Clean 60k Stage 2 dropout fusion: branch on vs off"),
    ("c60_w5_int", "lora5", "Clean 60k Stage 2 no-dropout interleave vs LoRA alone"),
    ("c60_w5_fuse", "lora5", "Clean 60k Stage 2 no-dropout fusion vs LoRA alone"),
    ("c60_w5_int", "c60_w5_int_off", "Clean 60k Stage 2 no-dropout interleave: branch on vs off"),
    ("c60_w5_fuse", "c60_w5_fuse_off", "Clean 60k Stage 2 no-dropout fusion: branch on vs off"),
    ("c60_lf_int", "lf276_int", "60k-only branch vs 240k step-276k branch (LoRA-first, interleave)"),
    ("c60_lf_fuse", "lf276_fuse", "60k-only branch vs 240k step-276k branch (LoRA-first, fusion)"),
    ("c60e_lf_int", "lf_int_ctrl", "Early 60k LoRA-first interleave vs the control"),
    ("c60e_lf_fuse", "lf_fuse_ctrl", "Early 60k LoRA-first fusion vs the control"),
    ("c60e_lf_int", "lora5", "Early 60k LoRA-first interleave vs LoRA alone"),
    ("c60e_lf_fuse", "lora5", "Early 60k LoRA-first fusion vs LoRA alone"),
    ("c60e_lf_int_off", "lf_int_ctrl", "Early 60k LoRA-first interleave: did its own LoRA keep up?"),
    ("c60e_lf_fuse_off", "lf_fuse_ctrl", "Early 60k LoRA-first fusion: did its own LoRA keep up?"),
    ("c60e_w5d_int", "lora5", "Early 60k Stage 2 dropout interleave vs LoRA alone"),
    ("c60e_w5d_fuse", "lora5", "Early 60k Stage 2 dropout fusion vs LoRA alone"),
    ("c60e_w5d_int", "c60e_w5d_int_off", "Early 60k Stage 2 dropout interleave: branch on vs off"),
    ("c60e_w5d_fuse", "c60e_w5d_fuse_off", "Early 60k Stage 2 dropout fusion: branch on vs off"),
    ("c60e_w5_int", "lora5", "Early 60k Stage 2 no-dropout interleave vs LoRA alone"),
    ("c60e_w5_fuse", "lora5", "Early 60k Stage 2 no-dropout fusion vs LoRA alone"),
    ("c60e_w5_int", "c60e_w5_int_off", "Early 60k Stage 2 no-dropout interleave: branch on vs off"),
    ("c60e_w5_fuse", "c60e_w5_fuse_off", "Early 60k Stage 2 no-dropout fusion: branch on vs off"),
    ("c60_lf_int", "c60e_lf_int", "Finished vs early 60k branch (LoRA-first, interleave)"),
    ("c60_lf_fuse", "c60e_lf_fuse", "Finished vs early 60k branch (LoRA-first, fusion)"),
]


def load(name):
    js = json.load(open(f"{D}/{name}.json"))
    key = "aux" if name.startswith("br") else "llm"
    step = js.get("ckpt_step")
    lf = (js.get("config") or {}).get("lora_from")
    if lf and os.path.exists(lf.rstrip("/") + ".json"):
        step = json.load(open(lf.rstrip("/") + ".json")).get("step")
    recs = {r["id"]: (r["label"], np.array(r[f"{key}_scores"], dtype=np.float64)) for r in js["records"]
            if f"{key}_scores" in r}
    return recs, step


def logsm(x):
    x = x - x.max(-1, keepdims=True)
    return x - np.log(np.exp(x).sum(-1, keepdims=True))


def mcnemar(a, b):
    n01, n10 = int((~a & b).sum()), int((a & ~b).sum())
    n = n01 + n10
    if n == 0:
        return 1.0
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(min(n01, n10) + 1)) / 2 ** n)


def ci(c):
    p, n = c.mean(), len(c)
    return 100 * 1.96 * math.sqrt(p * (1 - p) / n)


names = [os.path.basename(f)[:-5] for f in sorted(glob.glob(f"{D}/*.json")) if not f.endswith("rules.json")]
names = sorted(names, key=lambda n: list(WHAT).index(n) if n in WHAT else 99)
M = {n: load(n) for n in names}
ids = sorted(set.intersection(*[set(m[0]) for m in M.values()]))
N = len(ids)
y = np.array([M[names[0]][0][i][0] for i in ids])
S = {n: np.stack([M[n][0][i][1] for i in ids]) for n in names}
C = {n: S[n].argmax(1) == y for n in names}
rules = json.load(open(f"{D}/rules.json")) if os.path.exists(f"{D}/rules.json") else {}

L = ["# Phase 1 runs on 5,000 unseen PGM test problems\n",
     f"{N} problems scored by every model below (`test5000.tsv`, never used for training or picking checkpoints). "
     f"Chance is 12.5%. With {N} problems one model's accuracy is good to about +/-{98 / math.sqrt(N):.1f} points; two models compared "
     "problem by problem are more precise than that, which is what the McNemar p measures.\n",
     "## 1. Accuracy\n", "| Model | What it is | Checkpoint step | Accuracy (95% interval) |", "|---|---|---|---|"]
for n in names:
    L.append(f"| `{n}` | {WHAT.get(n, '')} | {M[n][1]} | {100 * C[n].mean():.1f}% (+/-{ci(C[n]):.1f}) |")

L += ["\n## 2. Paired comparisons\n", "| Question | A | B | A | B | Gap | Only A right | Only B right | McNemar p |",
      "|---|---|---|---|---|---|---|---|---|"]
for a, b, q in PAIRS:
    if a in C and b in C:
        ca, cb = C[a], C[b]
        L.append(f"| {q} | `{a}` | `{b}` | {100 * ca.mean():.1f}% | {100 * cb.mean():.1f}% | "
                 f"**{100 * (ca.mean() - cb.mean()):+.1f}** | {int((ca & ~cb).sum())} | {int((~ca & cb).sum())} | "
                 f"{mcnemar(ca, cb):.3g} |")

BR = [n for n in names if n.startswith("br")]
if BR:
    L += ["\n## 3. Late-fusion reference (average of LoRA's and the branch's answers, w = 0.5, untuned)\n",
          "What simply averaging two separately trained models gets; the trained hybrids should match or beat it.\n",
          "| LoRA | + branch | Branch alone | LoRA alone | Averaged | Gain | McNemar p |", "|---|---|---|---|---|---|---|"]
    for a in ("lora5", "lf_int_ctrl", "lf_fuse_ctrl"):
        for b in (BR if a == "lora5" else [n for n in BR if n == "brset"]):
            if a in C:
                comb = (0.5 * logsm(S[a]) + 0.5 * logsm(S[b])).argmax(1) == y
                L.append(f"| `{a}` | `{b}` | {100 * C[b].mean():.1f}% | {100 * C[a].mean():.1f}% | {100 * comb.mean():.1f}% | "
                         f"{100 * (comb.mean() - C[a].mean()):+.1f} | {mcnemar(comb, C[a]):.3g} |")

if rules:
    for a, b in (("lf_int", "lf_int_ctrl"), ("lf_fuse", "lf_fuse_ctrl"), ("lf276_int", "lf_int_ctrl"), ("lf276_fuse", "lf_fuse_ctrl"),
                 ("c60_lf_int", "lf_int_ctrl"), ("c60_lf_fuse", "lf_fuse_ctrl"),
                 ("c60e_lf_int", "lf_int_ctrl"), ("c60e_lf_fuse", "lf_fuse_ctrl")):
        if a not in C or b not in C:
            continue
        L += [f"\n## 4. By attribute: `{a}` vs `{b}`\n", "A problem counts once in each attribute it has rules on.\n",
              "| Attribute | n | Hybrid | Control | Gap |" + (" Branch alone |" if "brset" in C else ""),
              "|---|---|---|---|---|" + ("---|" if "brset" in C else "")]
        groups = {}
        for k, i in enumerate(ids):
            for g in {r[1] for r in rules.get(i, [])}:
                groups.setdefault(g, []).append(k)
        for g, ks in sorted(groups.items(), key=lambda kv: -len(kv[1])):
            ks = np.array(ks)
            row = (f"| {g} | {len(ks)} | {100 * C[a][ks].mean():.1f}% | {100 * C[b][ks].mean():.1f}% | "
                   f"{100 * (C[a][ks].mean() - C[b][ks].mean()):+.1f} |")
            L.append(row + (f" {100 * C['brset'][ks].mean():.1f}% |" if "brset" in C else ""))

open(f"{D}/report.md", "w").write("\n".join(L) + "\n")
print("\n".join(L))
