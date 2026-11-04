"""Verify that this installation reproduces the original results.

Scores the sample problems in sample_problems/ with the fine-tuned adapters and compares every candidate score with
the values recorded in the original training environment (sample_problems/expected_scores.json). Same predicted
answers + scores within tolerance = the setup is equivalent to ours.

  bash run.sh check                    (normally via: sbatch check.sbatch)
  python check.py --bundle . --all     every adapter, not just the final ones
  python check.py --bundle . --record --base qwen=Qwen/Qwen3-VL-2B-Instruct --base internvl=OpenGVLab/InternVL3_5-1B-HF
                                        (maintainers: write expected_scores.json from the original environment)
"""
import argparse
import glob
import json
import os
import platform
import sys

import torch

from pgmvlm.data import CandidateCollator, PGMStream
from pgmvlm.models import ScoreTokens, candidate_scores, load_adapter, load_base

MODELS = {"qwen": ("Qwen3-VL-2B-Instruct", "qwen3vl2b_pgm_ep"), "internvl": ("InternVL3_5-1B-HF", "internvl35_1b_pgm_ep")}
FINAL = {"qwen": 5, "internvl": 6}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bundle", required=True)
    ap.add_argument("--all", action="store_true", help="check every adapter in adapters/, not only the final ones")
    ap.add_argument("--tol", type=float, default=0.3, help="max allowed |score difference| (bf16 on different GPUs)")
    ap.add_argument("--record", action="store_true", help="write expected_scores.json instead of checking")
    ap.add_argument("--base", action="append", default=[], help="model=path_or_id override for the base models")
    a = ap.parse_args()
    B = os.path.abspath(a.bundle)
    base = {m: os.path.join(B, "models", d) for m, (d, _) in MODELS.items()}
    base.update(dict(x.split("=", 1) for x in a.base))
    sample_dir = os.path.join(B, "sample_problems")
    files = sorted(os.path.basename(f) for f in glob.glob(os.path.join(sample_dir, "*.npz")))
    exp_path = os.path.join(sample_dir, "expected_scores.json")
    expected = {} if a.record else json.load(open(exp_path))

    runs = []
    for m, (_, prefix) in MODELS.items():
        epochs = sorted(int(d.rsplit("ep", 1)[1]) for d in os.listdir(os.path.join(B, "adapters"))
                        if d.startswith(prefix) and d.rsplit("ep", 1)[1].isdigit())
        runs += [(m, e) for e in epochs if a.all or a.record or e == FINAL[m]]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"python {platform.python_version()} | torch {torch.__version__} | GPU {torch.cuda.get_device_name(0) if device.type == 'cuda' else 'none (CPU)'}")
    import peft, transformers
    print(f"transformers {transformers.__version__} | peft {peft.__version__}\n")
    if device.type != "cuda":
        print("WARNING: no GPU; scores are computed in bf16 on GPU in the reference, so CPU results may differ more")

    out, ok_all = {}, True
    for m, e in runs:
        key = f"{m}_ep{e}"
        model, proc, family = load_base(base[m], "bf16", "sdpa", device)
        model = load_adapter(model, os.path.join(B, "adapters", f"{MODELS[m][1]}{e}")).eval()
        st, col = ScoreTokens(proc.tokenizer), CandidateCollator(proc, family, crop_to_patches=True)
        ds = PGMStream(sample_dir, files)
        res = {}
        with torch.no_grad():
            for i in range(len(ds)):
                item = ds[i]
                s = candidate_scores(model, col([item]), st, device, torch.bfloat16)[0].cpu()
                res[item["file"]] = {"scores": [round(float(x), 4) for x in s], "pred": int(s.argmax()) + 1,
                                     "correct": item["target"] + 1}
        out[key] = res
        acc = sum(r["pred"] == r["correct"] for r in res.values()) / len(res)
        if a.record:
            print(f"{key}: recorded {len(res)} problems (accuracy on the samples {acc:.0%})")
        else:
            ref = expected[key]
            same = sum(res[f]["pred"] == ref[f]["pred"] for f in res)
            diff = max(max(abs(x - y) for x, y in zip(res[f]["scores"], ref[f]["scores"])) for f in res)
            ok = same == len(res) and diff <= a.tol
            ok_all &= ok
            print(f"{key}: {'PASS' if ok else 'FAIL'} | same answers {same}/{len(res)} | max score difference {diff:.3f} "
                  f"(tolerance {a.tol}) | sample accuracy {acc:.0%}")
        del model
        torch.cuda.empty_cache()

    if a.record:
        json.dump(out, open(exp_path, "w"), indent=1)
        print(f"\nwrote {exp_path}")
        return
    print("\nALL CHECKS PASSED: this setup reproduces the original results." if ok_all else
          "\nCHECK FAILED: compare the versions printed above with requirements-lock.txt and README.md (Troubleshooting).")
    sys.exit(0 if ok_all else 1)


if __name__ == "__main__":
    main()
