"""Validation loss and accuracy of saved checkpoints (validation loss was not logged during training).

Scores each adapter on the same validation problems the training run checked (the 1,000 shared ones), with the same
per-candidate readout, and reports the 8-way cross-entropy (the training loss) and the accuracy.

  python tools/val_loss.py --model_id Qwen/Qwen3-VL-2B-Instruct --val_dir .../val_full/neutral \
      --val_list .../val1000_shared.tsv --ckpt 1=runs/qwen/final_e1 --ckpt 2=runs/qwen/final_e2 --out val_loss_qwen.json
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from model_utils import DTYPES, ScoreTokens, candidate_scores, load_adapter, load_base
from pgm_vlm_data import CandidateCollator, PGMStream


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_id", required=True)
    ap.add_argument("--val_dir", required=True)
    ap.add_argument("--val_list", required=True)
    ap.add_argument("--ckpt", action="append", required=True, help="epoch=adapter_dir")
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--dtype", default="bf16", choices=list(DTYPES))
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = DTYPES[a.dtype]
    files = [l.split("\t")[0].strip() for l in open(a.val_list) if l.strip() and not l.startswith("#")]

    out = {}
    for spec in a.ckpt:
        epoch, path = spec.split("=", 1)
        model, proc, family = load_base(a.model_id, a.dtype, "sdpa", device)     # fresh base per adapter
        model = load_adapter(model, path).eval()
        st = ScoreTokens(proc.tokenizer)
        loader = DataLoader(PGMStream(a.val_dir, files), batch_size=a.batch_size, shuffle=False, num_workers=4,
                            collate_fn=CandidateCollator(proc, family, crop_to_patches=True))
        tot, correct, n = 0.0, 0, 0
        with torch.no_grad():
            for batch in loader:
                s = candidate_scores(model, batch, st, device, dtype)
                t = batch["targets"].to(s.device)
                tot += F.cross_entropy(s, t, reduction="sum").item()
                correct += (s.argmax(-1) == t).sum().item()
                n += t.numel()
        out[epoch] = {"adapter": path, "n": n, "val_loss": tot / n, "val_acc": correct / n}
        print(f"epoch {epoch}: val loss {tot / n:.4f}  val acc {correct / n:.4f}  (n={n}, {path})", flush=True)
        del model
        torch.cuda.empty_cache()
    json.dump(out, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
