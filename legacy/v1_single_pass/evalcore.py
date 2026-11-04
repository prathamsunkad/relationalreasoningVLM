"""Evaluation loop shared by train_lora.py (quick validation) and eval_vlm.py (full sharded test runs)."""
import math

import numpy as np
import torch

from model_utils import generate_text, restricted_logits
from pgm_vlm_data import extract_answer


@torch.no_grad()
def evaluate_batches(model, processor, loader, answer, device, dtype, modes=("logits",), max_new_tokens=16,
                     on_batch=None):
    """Yields one record per problem. modes: 'logits' (restricted 8-way readout) and/or 'generate' (free greedy text)."""
    model.eval()
    for batch in loader:
        targets = batch["targets"].tolist()
        recs = [dict(file=f, target=t) for f, t in zip(batch["files"], targets)]
        if "logits" in modes:
            lg = restricted_logits(model, batch, answer, device, dtype)
            probs = lg.softmax(-1).cpu()
            pred = probs.argmax(-1).tolist()
            for r, p, pr in zip(recs, pred, probs):
                r.update(pred_logits=p, correct_logits=bool(p == r["target"]), probs=[round(float(x), 4) for x in pr])
        if "generate" in modes:
            texts = generate_text(model, processor, batch, device, dtype, max_new_tokens, stop_ids=[answer.eos_id])
            for r, t in zip(recs, texts):
                p = extract_answer(t)
                r.update(pred_gen=p, correct_gen=bool(p == r["target"]), raw_response=t)
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
    """Accuracy with 95% Wilson interval for both readouts + diagnostics for shortcuts / format failures."""
    out = {"n": len(records)}
    for key, name in (("correct_logits", "logits"), ("correct_gen", "generate")):
        rs = [r for r in records if key in r]
        if not rs:
            continue
        k = sum(r[key] for r in rs)
        lo, hi = wilson(k, len(rs))
        d = {"acc": k / len(rs), "ci95": [lo, hi], "n": len(rs)}
        pk = "pred_logits" if name == "logits" else "pred_gen"
        if name == "generate":
            d["invalid_rate"] = sum(r[pk] is None for r in rs) / len(rs)
        d["pred_hist"] = [sum(r[pk] == i for r in rs) for i in range(8)]
        d["target_hist"] = [sum(r["target"] == i for r in rs) for i in range(8)]
        d["acc_by_target_position"] = [
            (sum(r[key] for r in rs if r["target"] == i) / max(1, sum(r["target"] == i for r in rs))) for i in range(8)]
        out[name] = d
    return out
