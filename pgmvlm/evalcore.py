"""Evaluation loop shared by train_lora.py (quick validation) and eval_vlm.py (full sharded test runs)."""
import math

import torch

from model_utils import candidate_scores


@torch.no_grad()
def evaluate_batches(model, loader, st, device, dtype, on_batch=None):
    """Yields one record per problem: the 8 candidate scores, the argmax choice and whether it is correct."""
    model.eval()
    for batch in loader:
        scores = candidate_scores(model, batch, st, device, dtype).cpu()
        recs = []
        for f, t, s in zip(batch["files"], batch["targets"].tolist(), scores):
            p = int(s.argmax())
            recs.append(dict(file=f, target=t, pred=p, correct=p == t, scores=[round(float(x), 4) for x in s]))
        if on_batch:
            on_batch(recs)
        yield from recs


def wilson(k, n, z=1.96):
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (c - h, c + h)


def summarize(records):
    """Accuracy with 95% Wilson interval + diagnostics: predicted-position histogram (a collapse to one position shows
    up here), accuracy per answer position, and how often the correct candidate gets a positive score (Yes > No)."""
    rs = [r for r in records if "correct" in r]
    out = {"n": len(rs)}
    if not rs:
        return out
    k = sum(r["correct"] for r in rs)
    lo, hi = wilson(k, len(rs))
    out["scores"] = {
        "acc": k / len(rs), "ci95": [lo, hi], "n": len(rs),
        "pred_hist": [sum(r["pred"] == i for r in rs) for i in range(8)],
        "target_hist": [sum(r["target"] == i for r in rs) for i in range(8)],
        "acc_by_target_position": [sum(r["correct"] for r in rs if r["target"] == i) / max(1, sum(r["target"] == i for r in rs))
                                   for i in range(8)],
        "answer_says_yes": sum(r["scores"][r["target"]] > 0 for r in rs) / len(rs),
        "distractor_says_yes": sum(sum(s > 0 for j, s in enumerate(r["scores"]) if j != r["target"]) for r in rs) / (7 * len(rs)),
    }
    return out
