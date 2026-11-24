#!/usr/bin/env python
"""
Progress and results of an Exp1 pipeline run (pipeline.sh). CPU only, no torch.

  python pipeline_report.py status <RUN>     # per step: step / total, ETA, best held-out validation
  python pipeline_report.py compare <RUN>    # test table + paired comparisons -> <RUN>/report_test.md

What the hybrid has to show (all on the same test puzzles, paired):
  1. s2 > ctrl        it beats LoRA trained for the same number of steps without the branch
  2. s2 > s2 shuffled  it uses the relational features of *its own* puzzle (not just extra capacity)
  3. s2 > ensemble     it beats simply adding the branch's 8-way scores to the control's answer (no fusion needed)
"""
import argparse
import json
import math
import os

STEPS = ["s0A", "s0B", "base", "s1", "s2", "ctrl"]
NAMES = {"s0A": "stage 0, half A (branch alone)", "s0B": "stage 0, half B (branch alone)",
         "base": "LoRA baseline on the crops", "s1": "stage 1: alignment", "s2": "stage 2: hybrid",
         "ctrl": "control: LoRA, equal steps"}


def wilson(k, n, z=1.96):
    if n == 0:
        return 0.0, 0.0, 0.0
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return p, c - h, c + h


def read_jsonl(path):
    with open(path) as fh:
        return [json.loads(line) for line in fh if line.strip()]


def status(run, recipe):
    print(f"{run}  (recipe {recipe})\n")
    print("| step | what | state | step / total | puzzles/s | ETA | best held-out loss | acc at best |")
    print("|---|---|---|---|---|---|---|---|")
    for s in STEPS:
        if s == "ctrl" and recipe == "joint":
            continue
        d = os.path.join(run, s)
        logf = os.path.join(d, "train_log.jsonl")
        if not os.path.exists(logf):
            print(f"| {s} | {NAMES[s]} | not started | | | | | |")
            continue
        recs = read_jsonl(logf)
        tr = [r for r in recs if "step" in r and "loss" in r and "val" not in r]
        vals = [r["val"] for r in recs if "val" in r]
        done = os.path.exists(os.path.join(d, "summary.json"))
        last = tr[-1] if tr else {}
        stepn = max([last.get("step", 0)] + [v.get("step", 0) for v in vals])
        total = last.get("of")
        eta = ""
        timed = [r for r in tr if r.get("t")][-2:]   # wall-clock rate between the last two log lines
        if not done and total and len(timed) == 2 and timed[1]["step"] > timed[0]["step"] and timed[1]["t"] > timed[0]["t"]:
            rate = (timed[1]["step"] - timed[0]["step"]) / (timed[1]["t"] - timed[0]["t"])
            eta = f"{(total - stepn) / rate / 3600:.1f} h of GPU time"
        if done:
            sm = json.load(open(os.path.join(d, "summary.json")))
            state = "finished" + (" (early stop)" if sm.get("stopped_early") else "") + (
                f" (plateau at {sm['plateau_at']}, then LR decay)" if sm.get("plateau_at") else "")
        else:
            state = "running / paused"
        b = min(vals, key=lambda v: v["loss"]) if vals else None
        print(f"| {s} | {NAMES[s]} | {state} | {stepn} / {total or '?'} | {last.get('puzzles_per_s', '')} | {eta} | "
              f"{b['loss']:.4f} (step {b['step']}) | {b['acc']:.2%} |" if b else
              f"| {s} | {NAMES[s]} | {state} | {stepn} / {total or '?'} | {last.get('puzzles_per_s', '')} | {eta} | | |")
    rep = os.path.join(run, "report_test.md")
    print(f"\nTest report: {'written: ' + rep if os.path.exists(rep) else 'not yet (eval step)'}")
    print("Chance: loss ln 8 = 2.079 (stage 0); accuracy 12.5%. Held-out = last rows of train.jsonl.")


def mcnemar(a, b):
    """a, b: lists of bools on the same puzzles. Returns (a-only, b-only, two-sided p, normal approximation)."""
    x = sum(1 for i, j in zip(a, b) if i and not j)
    y = sum(1 for i, j in zip(a, b) if j and not i)
    if x + y == 0:
        return x, y, 1.0
    z = (abs(x - y) - 1) / math.sqrt(x + y)
    return x, y, math.erfc(max(z, 0) / math.sqrt(2))


def compare(run, recipe):
    def load(step, tag=""):
        p = os.path.join(run, step, f"eval_test{tag}.jsonl")
        return {r["id"]: r for r in read_jsonl(p)} if os.path.exists(p) else None

    ctrl_step = "base" if recipe == "joint" else "ctrl"
    runs = {"base": load("base"), "ctrl": load(ctrl_step), "s1": load("s1"), "s2": load("s2"),
            "s2_nofusion": load("s2", "_nofusion"), "s2_shuffled": load("s2", "_shuffled")}
    missing = [k for k, v in runs.items() if v is None]
    if missing:
        raise SystemExit(f"missing evaluations: {missing} (run: bash pipeline.sh run eval)")
    ids = [i for i in runs["s2"] if all(i in v for v in runs.values())]
    n = len(ids)
    corr = {k: [v[i]["correct"] for i in ids] for k, v in runs.items()}

    def argmax(xs):
        return max(range(len(xs)), key=lambda j: xs[j])

    ans = [runs["s2"][i]["answer"] for i in ids]
    L = "ABCDEFGH"
    # the branch alone: the stage-1 checkpoint holds the two stage-0 branches (averaged), unchanged by stage 2
    corr["branch"] = [L[argmax(runs["s1"][i]["candidate_logprobs"])] == a for i, a in zip(ids, ans)]
    corr["branch_s2"] = [L[argmax(runs["s2"][i]["candidate_logprobs"])] == a for i, a in zip(ids, ans)]
    # post-hoc ensemble: the control's letter log-probs + the branch's candidate log-probs (no fusion, no training)
    corr["ensemble"] = [L[argmax([x + y for x, y in zip(runs["ctrl"][i]["letter_logprobs"],
                                                         runs["s1"][i]["candidate_logprobs"])])] == a
                        for i, a in zip(ids, ans)]
    corr["s2_internal_ens"] = [L[argmax([x + y for x, y in zip(runs["s2"][i]["letter_logprobs"],
                                                                runs["s2"][i]["candidate_logprobs"])])] == a
                               for i, a in zip(ids, ans)]
    ctrl_label = "LoRA baseline (= the control in the joint recipe)" if recipe == "joint" else \
        "control: base LoRA + the same extra steps, no branch"
    rows = [("base", "LoRA baseline on the 16 crops"),
            ("ctrl", ctrl_label),
            ("branch", "relational branch alone (stage 0, both halves averaged)"),
            ("ensemble", "post-hoc ensemble: control + branch scores"),
            ("s1", "stage 1: frozen LLM" + (" + base LoRA" if recipe == "lorafirst" else "") + " + aligned branch"),
            ("s2", "**stage 2: hybrid**"),
            ("s2_nofusion", "hybrid, branch switched off"),
            ("s2_shuffled", "hybrid, another puzzle's relational features"),
            ("branch_s2", "hybrid's score head alone"),
            ("s2_internal_ens", "hybrid + its own score head")]
    out = [f"# Exp1 test results: {os.path.basename(os.path.normpath(run))}", "",
           f"{n} test puzzles (the same for every row), chance 12.5%. 95% Wilson intervals.", "",
           "| model | accuracy | 95% CI |", "|---|---|---|"]
    for k, label in rows:
        p, lo, hi = wilson(sum(corr[k]), n)
        out.append(f"| {label} | {p:.2%} | [{lo:.2%}, {hi:.2%}] |")
    out += ["", "## Paired comparisons (McNemar, same puzzles)", "",
            "| comparison | difference | only first right | only second right | p |", "|---|---|---|---|---|"]
    verdict = {}
    for a, b, what in [("s2", "ctrl", "beats LoRA with equal steps"),
                       ("s2", "s2_shuffled", "uses its own puzzle's relational features"),
                       ("s2", "ensemble", "beats the post-hoc ensemble"),
                       ("s2", "s2_nofusion", "relies on the branch at test time"),
                       ("s2_nofusion", "ctrl", "LLM alone after hybrid training vs control")]:
        x, y, p = mcnemar(corr[a], corr[b])
        d = (sum(corr[a]) - sum(corr[b])) / n
        verdict[(a, b)] = d > 0 and p < 0.05
        out.append(f"| {a} vs {b}: {what} | {d:+.2%} | {x} | {y} | {p:.2g} |")
    ok = [verdict[("s2", "ctrl")], verdict[("s2", "s2_shuffled")], verdict[("s2", "ensemble")]]
    out += ["", "## Reading", "",
            f"- Hybrid beats the equal-step LoRA control: **{'yes' if ok[0] else 'no'}**",
            f"- Hybrid uses its own puzzle's relational features (beats shuffled): **{'yes' if ok[1] else 'no'}**",
            f"- Hybrid beats adding the branch's scores after the fact: **{'yes' if ok[2] else 'no'}**",
            "",
            "All three yes = the fused relational features help the LLM reason. Yes to the first only = the gain "
            "may be extra capacity or an ensemble effect.",
            ""]
    for s in STEPS:
        sp = os.path.join(run, s, "summary.json")
        if os.path.exists(sp):
            b = json.load(open(sp)).get("best") or {}
            if b:
                out.append(f"- {s}: selected at step {b.get('step')} (held-out loss {b.get('loss', 0):.4f}, "
                           f"acc {b.get('acc', 0):.2%})")
    text = "\n".join(out) + "\n"
    with open(os.path.join(run, "report_test.md"), "w") as fh:
        fh.write(text)
    print(text)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["status", "compare"])
    ap.add_argument("run")
    ap.add_argument("--recipe", default="lorafirst", choices=["lorafirst", "joint"])
    a = ap.parse_args()
    (status if a.cmd == "status" else compare)(a.run, a.recipe)


if __name__ == "__main__":
    main()
