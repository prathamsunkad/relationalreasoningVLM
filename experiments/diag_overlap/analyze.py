"""Overlap check (Phase 1): does the branch know anything LoRA alone does not?

Reads the per-problem eval records in this folder (LoRA alone: lora*.json, llm readout; branch alone: br*.json, aux
readout) for the same 1,000 shared PGM validation problems, plus rules.json, and writes report.md:
  1. accuracy of each model (sanity check against the training logs);
  2. per LoRA x branch pair: the 2x2 right/wrong table, the 'only the branch' count against what independence would
     give, the either-right ceiling, and the combined model: average of the two log-softmaxed 8-way scores
     (w = 0.5, untuned) and a weight picked by 2-fold cross-validation (100 random splits), with an exact McNemar test
     against LoRA alone;
  3. per rule (relation, attribute, object): LoRA, branch and combined accuracy.
Usage: python analyze.py [--dir D]   (numpy only; runs on the login node)"""
import argparse
import glob
import json
import math
import os

import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--dir", default=os.path.dirname(os.path.abspath(__file__)))
args = ap.parse_args()
D = args.dir


def load(name):
    js = json.load(open(f"{D}/{name}.json"))
    key = "aux" if name.startswith("br") else "llm"
    lf = (js.get("config") or {}).get("lora_from")
    if lf and os.path.exists(lf.rstrip("/") + ".json"):      # LoRA alone: the adapter's own step, not the host ckpt's
        js["ckpt_step"] = json.load(open(lf.rstrip("/") + ".json")).get("step")
    out = {}
    for r in js["records"]:
        if f"{key}_scores" in r:
            out[r["id"]] = (r["label"], np.array(r[f"{key}_scores"], dtype=np.float64))
    return out, js.get("ckpt_step")


def logsm(x):
    x = x - x.max(-1, keepdims=True)
    return x - np.log(np.exp(x).sum(-1, keepdims=True))


def mcnemar(a, b):
    """Exact two-sided McNemar p for paired correctness vectors a, b."""
    n01, n10 = int(((~a) & b).sum()), int((a & ~b).sum())
    n = n01 + n10
    if n == 0:
        return 1.0, n01, n10
    k = min(n01, n10)
    p = 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n
    return min(1.0, p), n01, n10


def pct(x):
    return f"{100 * x:.1f}"


names = sorted(os.path.basename(f)[:-5] for f in glob.glob(f"{D}/*.json") if os.path.basename(f) != "rules.json")
models = {n: load(n) for n in names}
loras = [n for n in names if n.startswith("lora")]
branches = [n for n in names if n.startswith("br")]
ids = sorted(set.intersection(*[set(m[0]) for m in models.values()]))
N = len(ids)
y = np.array([models[names[0]][0][i][0] for i in ids])
S = {n: np.stack([models[n][0][i][1] for i in ids]) for n in names}
LP = {n: logsm(S[n]) for n in names}
C = {n: S[n].argmax(1) == y for n in names}
rules = json.load(open(f"{D}/rules.json")) if os.path.exists(f"{D}/rules.json") else {}

L = [f"# Overlap check: LoRA alone vs the branch alone\n",
     f"{N} problems (the shared validation set), the same for every model. Chance is 12.5%.\n",
     "## 1. Accuracy of each model\n", "| Model | Checkpoint step | Accuracy |", "|---|---|---|"]
for n in names:
    L.append(f"| `{n}` | {models[n][1]} | {pct(C[n].mean())}% |")

rng = np.random.default_rng(0)
W = np.round(np.arange(0, 1.0001, 0.05), 2)
L += ["\n## 2. Pairs\n",
      "Combined = w x LoRA log-probabilities + (1 - w) x branch log-probabilities, argmax over the 8 candidates. "
      "'w = 0.5' is untuned. 'CV' picks w on a random half and scores the other half (100 splits); its gain is "
      "against LoRA alone on the same halves. McNemar compares the w = 0.5 combination with LoRA alone, problem by "
      "problem (p < 0.05: unlikely to be noise).\n",
      "| LoRA | Branch | Both right | Only LoRA | Only branch (if independent) | Neither | Either right | "
      "Combined w=0.5 | Gain | McNemar p | CV gain (w picked) |", "|---|---|---|---|---|---|---|---|---|---|---|"]
summary = []
for a in loras:
    for b in branches:
        ca, cb = C[a], C[b]
        both, oa, ob, nei = int((ca & cb).sum()), int((ca & ~cb).sum()), int((~ca & cb).sum()), int((~ca & ~cb).sum())
        exp_ob = N * cb.mean() * (1 - ca.mean())
        comb = (0.5 * LP[a] + 0.5 * LP[b]).argmax(1) == y
        p, _, _ = mcnemar(ca, comb)
        gains, ws = [], []
        for _ in range(100):
            perm = rng.permutation(N)
            h1, h2 = perm[: N // 2], perm[N // 2:]
            for tr, te in ((h1, h2), (h2, h1)):
                accs = [((w * LP[a][tr] + (1 - w) * LP[b][tr]).argmax(1) == y[tr]).mean() for w in W]
                w = W[int(np.argmax(accs))]
                ws.append(w)
                gains.append(((w * LP[a][te] + (1 - w) * LP[b][te]).argmax(1) == y[te]).mean() - ca[te].mean())
        g = 100 * np.mean(gains)
        L.append(f"| `{a}` | `{b}` | {both} | {oa} | **{ob}** ({exp_ob:.0f}) | {nei} | {pct((ca | cb).mean())}% | "
                 f"{pct(comb.mean())}% | {100 * (comb.mean() - ca.mean()):+.1f} | {p:.3g} | {g:+.1f} (w~{np.median(ws):.2f}) |")
        summary.append((a, b, comb.mean() - ca.mean(), g, p))

if rules and loras and branches:
    a = loras[0] if "lora5" not in loras else "lora5"
    b = max(branches, key=lambda n: C[n].mean())
    comb = (0.5 * LP[a] + 0.5 * LP[b]).argmax(1) == y
    L += [f"\n## 3. By rule (`{a}` vs `{b}`, combined at w = 0.5)\n",
          "A problem has 1-4 rules and counts once in each group it has. Small groups are noisy: with n = 100, "
          "a difference under ~10 points can be chance.\n"]
    for col, title in ((2, "Relation"), (1, "Attribute"), (0, "Object")):
        groups = {}
        for k, i in enumerate(ids):
            for g in {r[col] for r in rules.get(i, [])}:
                groups.setdefault(g, []).append(k)
        L += [f"\n**{title}**\n", f"| {title} | n | LoRA | Branch | Combined | Branch - LoRA |", "|---|---|---|---|---|---|"]
        for g, ks in sorted(groups.items(), key=lambda kv: -len(kv[1])):
            ks = np.array(ks)
            L.append(f"| {g} | {len(ks)} | {pct(C[a][ks].mean())}% | {pct(C[b][ks].mean())}% | {pct(comb[ks].mean())}% | "
                     f"{100 * (C[b][ks].mean() - C[a][ks].mean()):+.1f} |")
    nr = {}
    for k, i in enumerate(ids):
        nr.setdefault(len(rules.get(i, [])), []).append(k)
    L += ["\n**Number of rules in the problem**\n", "| Rules | n | LoRA | Branch | Combined |", "|---|---|---|---|---|"]
    for m, ks in sorted(nr.items()):
        ks = np.array(ks)
        L.append(f"| {m} | {len(ks)} | {pct(C[a][ks].mean())}% | {pct(C[b][ks].mean())}% | {pct(comb[ks].mean())}% |")

open(f"{D}/report.md", "w").write("\n".join(L) + "\n")
print("\n".join(L))
