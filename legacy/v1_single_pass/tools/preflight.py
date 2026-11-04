"""RUN THIS FIRST on the cluster with the real checkpoint (1 GPU, a few minutes). It prints everything this repo
had to assume, then does a real forward/backward so you see memory and speed before committing a 5h job.

  python tools/preflight.py --model_id Qwen/Qwen3-VL-2B-Instruct        --npz_dir .../train_full/neutral
  python tools/preflight.py --model_id OpenGVLab/InternVL3_5-2B-HF       --npz_dir .../train_full/neutral
Optional: --batch_size 8 --grad_ckpt --lora_vision --lora_projector --crop_to_patches 0
"""
import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from model_utils import (DTYPES, AnswerTokens, add_lora, generate_text, lm_and_digit_losses, load_base, param_report,
                         restricted_logits, teacher_forced_logits)
from pgm_vlm_data import PGMStream, VLMCollator, extract_answer, list_npz


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_id", required=True)
    ap.add_argument("--npz_dir", required=True)
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--steps", type=int, default=3)
    ap.add_argument("--dtype", default="bf16", choices=list(DTYPES))
    ap.add_argument("--attn", default="sdpa")
    ap.add_argument("--crop_to_patches", type=int, default=1)
    ap.add_argument("--lora_r", type=int, default=64)
    ap.add_argument("--lora_vision", action="store_true")
    ap.add_argument("--lora_projector", action="store_true")
    ap.add_argument("--grad_ckpt", action="store_true")
    args = ap.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = DTYPES[args.dtype]

    model, proc, family = load_base(args.model_id, args.dtype, args.attn, device)
    tok = proc.tokenizer
    answer = AnswerTokens(tok)
    print(f"== family {family} | dtype {args.dtype} | attn {args.attn}")
    print(f"== answer tokens: {answer.describe()}   (digit ids {answer.digit_ids})")
    print(f"== end-of-turn id {answer.eos_id} = {tok.convert_ids_to_tokens(answer.eos_id)!r}; "
          f"generation_config eos = {getattr(model.generation_config, 'eos_token_id', None)}")

    files = list_npz(args.npz_dir)[: args.batch_size]
    ds = PGMStream(args.npz_dir, files)
    col = VLMCollator(proc, family, args.crop_to_patches)
    batch = col([ds[i] for i in range(len(files))])
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
    print("== what the model sees (image runs collapsed):\n" + "".join(shown))
    print(f"== sequence length {len(ids)} | image tokens {sum(t == img_id for t in ids)} | "
          + " ".join(f"{k}{tuple(v.shape)}" for k, v in batch.items() if hasattr(v, "shape") and k != "input_ids"))

    model.eval()
    lg = restricted_logits(model, batch, answer, device, dtype)
    texts = generate_text(model, proc, batch, device, dtype, 16, stop_ids=[answer.eos_id])
    print("== zero-shot restricted preds (1-based):", (lg.argmax(-1) + 1).tolist(), "| targets:", (batch["targets"] + 1).tolist())
    for t in texts:
        print(f"   generate -> {t!r}  parsed={extract_answer(t)}")

    model = add_lora(model, family, args.lora_r, 2 * args.lora_r, 0.05, args.lora_vision, args.lora_projector, args.grad_ckpt)
    for p in model.parameters():
        if p.requires_grad:
            p.data = p.data.float()
    model.config.use_cache = False
    tr, tot = param_report(model)
    print(f"== LoRA r={args.lora_r}: trainable {tr/1e6:.1f}M of {tot/1e6:.0f}M ({100*tr/tot:.2f}%)")
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-4)
    model.train()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    for s in range(args.steps):
        t0 = time.time()
        logits, resp, targets = teacher_forced_logits(model, batch, answer, device, dtype)
        lm, d8, acc = lm_and_digit_losses(logits, resp, targets, answer)
        lm.backward()
        opt.step(); opt.zero_grad(set_to_none=True)
        if device.type == "cuda":
            torch.cuda.synchronize()
        print(f"== step {s}: lm {lm.item():.3f} digit8 {d8.item():.3f} | {time.time()-t0:.2f}s for {len(files)} samples")
    if device.type == "cuda":
        print(f"== peak GPU memory {torch.cuda.max_memory_allocated()/2**30:.1f} GiB at batch {len(files)} "
              f"(micro-batch {len(files)}: raise/lower --batch_size accordingly; --grad_ckpt trades speed for memory)")
    print("== preflight OK")


if __name__ == "__main__":
    main()
