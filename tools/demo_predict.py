"""Show exactly how a fine-tuned adapter answers ONE PGM test problem, the same way it was trained and evaluated:
8 completed matrices (candidate k in the bottom-right cell), one sequence each, score = logit(Yes) - logit(No),
softmax over the 8 scores, argmax = the model's answer. Also prints each sequence's free-text reply.

  python tools/demo_predict.py --npz_dir .../test_full/neutral --seed 7 \
      --model "Qwen3-VL-2B (epoch 5)=Qwen/Qwen3-VL-2B-Instruct@runs/qwen_pc_recipe_e1/final" \
      --model "InternVL3.5-1B (epoch 6)=OpenGVLab/InternVL3_5-1B-HF@runs/ivl1b_pc_recipe_e1/final" --out_dir demo
"""
import argparse
import json
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch

from pgmvlm.models import DTYPES, ScoreTokens, candidate_scores, load_adapter, load_base, to_model_inputs
from pgmvlm.data import PROMPT_TEXT, CandidateCollator, PGMStream, filled_matrices, list_npz


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz_dir", required=True)
    ap.add_argument("--file_cache", default=None, help="sorted test file list (avoids listing 200k files)")
    ap.add_argument("--file", default=None, help="a specific .npz name; default: a random one from --seed")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--model", action="append", required=True, help='"label=model_id@adapter_dir"')
    ap.add_argument("--dtype", default="bf16", choices=list(DTYPES))
    ap.add_argument("--out_dir", required=True)
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = DTYPES[a.dtype]

    fname = a.file or random.Random(a.seed).choice(list_npz(a.npz_dir, a.file_cache))
    item = PGMStream(a.npz_dir, [fname])[0]
    target = item["target"]
    print(f"problem {fname}: correct answer = candidate {target + 1}")
    print("prompt for every candidate:\n  " + PROMPT_TEXT.replace("\n", "\n  "))
    report = {"file": fname, "correct": target + 1, "models": {}}

    for spec in a.model:
        label, rest = spec.split("=", 1)
        model_id, adapter = rest.split("@", 1)
        model, proc, family = load_base(model_id, a.dtype, "sdpa", device)
        model = load_adapter(model, adapter).eval()
        st = ScoreTokens(proc.tokenizer)
        batch = CandidateCollator(proc, family, crop_to_patches=True)([item])
        with torch.no_grad():
            scores = candidate_scores(model, batch, st, device, dtype)[0].cpu()
            probs = scores.softmax(-1)
            # free-text reply of each of the 8 sequences (greedy, a few tokens)
            inputs = to_model_inputs(batch, device, dtype)
            try:
                with torch.autocast("cuda", dtype=dtype, enabled=device.type == "cuda"):
                    out = model.generate(**inputs, max_new_tokens=3, do_sample=False)
                replies = [t.strip() for t in proc.tokenizer.batch_decode(out[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)]
            except Exception as e:                                    # scoring above is what training/eval use
                replies = [f"(generate failed: {type(e).__name__})"] * 8
        pred = int(scores.argmax())
        print(f"\n=== {label}  [{model_id} + {adapter}]")
        print(f"{'cand':>4} {'score Yes-No':>13} {'prob':>7}  reply")
        for k in range(8):
            mark = ("  <- model's answer" if k == pred else "") + ("  (correct)" if k == target else "")
            print(f"{k + 1:>4} {scores[k]:>13.3f} {probs[k]:>7.1%}  {replies[k]!r}{mark}")
        print(f"model answers candidate {pred + 1} ({probs[pred]:.1%}); correct is {target + 1} -> "
              f"{'RIGHT' if pred == target else 'WRONG'}")
        report["models"][label] = {"scores": [round(float(x), 4) for x in scores], "probs": [round(float(x), 4) for x in probs],
                                   "replies": replies, "pred": pred + 1, "right": pred == target}
        del model
        torch.cuda.empty_cache()

    json.dump(report, open(os.path.join(a.out_dir, "demo.json"), "w"), indent=1)

    # figure: the 8 model inputs, annotated with each model's probability
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    mats = filled_matrices(item["context"], item["candidates"])
    fig, axs = plt.subplots(2, 4, figsize=(16, 9.2), dpi=110)
    labels = list(report["models"])
    for k, ax in enumerate(axs.ravel()):
        ax.imshow(mats[k]); ax.set_xticks([]); ax.set_yticks([])
        lines = [f"candidate {k + 1}" + ("  ✓ correct" if k == target else "")]
        for lab in labels:
            m = report["models"][lab]
            lines.append(f"{lab.split(' ')[0]}: p={m['probs'][k]:.1%}, says {m['replies'][k] or '?'}" + ("  ◀ pick" if m["pred"] == k + 1 else ""))
        ax.set_title("\n".join(lines), fontsize=9, color="green" if k == target else "black", loc="left")
        if k == target:
            for s in ax.spines.values():
                s.set_edgecolor("green"); s.set_linewidth(3)
    fig.suptitle(f"PGM test problem {fname}: the 8 inputs (candidate k in the bottom-right cell), one sequence each", fontsize=12)
    fig.tight_layout()
    fig.savefig(os.path.join(a.out_dir, "demo_inputs_scored.png"))
    print(f"\nsaved {a.out_dir}/demo_inputs_scored.png and demo.json")


if __name__ == "__main__":
    main()
