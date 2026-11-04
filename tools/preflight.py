"""RUN THIS FIRST on the cluster with the real checkpoint (1 GPU, a few minutes). It prints everything this repo
had to assume, then does a real forward/backward so you see memory and speed before committing a long job.

  python tools/preflight.py --model_id Qwen/Qwen3-VL-2B-Instruct        --npz_dir .../train_full/neutral
  python tools/preflight.py --model_id OpenGVLab/InternVL3_5-1B-HF      --npz_dir .../train_full/neutral
Optional: --batch_size 2 (problems = 8 sequences each) --grad_ckpt --lora_vision --lora_projector --panel_px 224
"""
import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from pgmvlm.models import (DTYPES, ScoreTokens, add_lora, candidate_losses, candidate_scores, load_base, param_report,
                         set_train_mode)
from pgmvlm.data import CandidateCollator, PGMStream, list_npz


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_id", required=True)
    ap.add_argument("--npz_dir", required=True)
    ap.add_argument("--batch_size", type=int, default=2, help="problems")
    ap.add_argument("--steps", type=int, default=3)
    ap.add_argument("--dtype", default="bf16", choices=list(DTYPES))
    ap.add_argument("--attn", default="sdpa")
    ap.add_argument("--crop_to_patches", type=int, default=1)
    ap.add_argument("--panel_px", type=int, default=160)
    ap.add_argument("--lora_r", type=int, default=16)
    ap.add_argument("--lora_vision", action="store_true")
    ap.add_argument("--lora_projector", action="store_true")
    ap.add_argument("--grad_ckpt", action="store_true")
    args = ap.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = DTYPES[args.dtype]

    model, proc, family = load_base(args.model_id, args.dtype, args.attn, device)
    tok = proc.tokenizer
    st = ScoreTokens(tok)
    print(f"== family {family} | dtype {args.dtype} | attn {args.attn}")
    print(f"== score tokens: {st.describe()}")

    files = list_npz(args.npz_dir)[: args.batch_size]
    ds = PGMStream(args.npz_dir, files)
    batch = CandidateCollator(proc, family, args.crop_to_patches, cell_size=args.panel_px)([ds[i] for i in range(len(files))])
    ids = batch["input_ids"][0].tolist()
    img_id = proc.image_token_id
    shown, run = [], 0
    for t in ids + [None]:
        if t == img_id:
            run += 1
            continue
        if run:
            shown.append(f"<{run} image tokens>")
            run = 0
        if t is not None:
            shown.append(tok.decode([t]))
    print("== what the model sees for ONE candidate (image runs collapsed):\n" + "".join(shown))
    print(f"== {batch['input_ids'].shape[0]} sequences for {len(files)} problems | sequence length {len(ids)} | "
          f"image tokens {sum(t == img_id for t in ids)} | "
          + " ".join(f"{k}{tuple(v.shape)}" for k, v in batch.items() if hasattr(v, "shape") and k != "input_ids"))

    model.eval()
    with torch.no_grad():
        sc = candidate_scores(model, batch, st, device, dtype)
    print("== zero-shot scores logit(Yes)-logit(No), one row per problem:")
    for row, t in zip(sc.tolist(), batch["targets"].tolist()):
        print("   " + " ".join(f"{x:+.2f}" for x in row) + f"   -> pick {int(torch.tensor(row).argmax()) + 1}, answer {t + 1}")

    model = add_lora(model, family, args.lora_r, 2 * args.lora_r, 0.05, args.lora_vision, args.lora_projector, args.grad_ckpt)
    for p in model.parameters():
        if p.requires_grad:
            p.data = p.data.float()
    model.config.use_cache = False
    tr, tot = param_report(model)
    print(f"== LoRA r={args.lora_r}: trainable {tr/1e6:.1f}M of {tot/1e6:.0f}M ({100*tr/tot:.2f}%)")
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-4)
    set_train_mode(model)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    for s in range(args.steps):
        t0 = time.time()
        ce8, bce, acc = candidate_losses(candidate_scores(model, batch, st, device, dtype), batch["targets"])
        ce8.backward()
        opt.step(); opt.zero_grad(set_to_none=True)
        if device.type == "cuda":
            torch.cuda.synchronize()
        print(f"== step {s}: ce8 {ce8.item():.3f} bce {bce.item():.3f} | {time.time()-t0:.2f}s for {len(files)} problems")
    if device.type == "cuda":
        print(f"== peak GPU memory {torch.cuda.max_memory_allocated()/2**30:.1f} GiB at {len(files)} problems per micro-batch "
              f"(raise/lower --batch_size accordingly; --grad_ckpt trades speed for memory)")
    print("== preflight OK")


if __name__ == "__main__":
    main()
