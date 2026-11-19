"""Show exactly what goes into the model for ONE PGM problem, stage by stage, with shapes and sizes, and save figures.

  python tools/visualize_input.py --dir $DATA/val_full/neutral --index 0 --out_dir $RUNS/viz                      # images only
  python tools/visualize_input.py --dir $DATA/val_full/neutral --index 0 --out_dir $RUNS/viz --model_id Qwen/Qwen3-VL-2B-Instruct
(the second form also runs the real processor: needs the model in the offline cache, no GPU)

Output: printed shapes/sizes at every stage, and PNGs
  1_all16_panels.png     the 16 panels as stored in the file (context / candidates, answer marked)
  2_model_images.png     the 8 completed matrices the model scores (one sequence each)
  3_token_grid.png       (with --model_id, Qwen only) the image as resized by the processor with one cell per visual token
  filled_<k>.png          each completed matrix on its own (the answer is marked _ANSWER)
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pgm_vlm_data import PROMPT_TEXT, filled_matrices, list_npz, to_panels  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", default=None, help="one .npz file")
    ap.add_argument("--dir", default=None, help="folder of .npz files (use with --index)")
    ap.add_argument("--index", type=int, default=0)
    ap.add_argument("--model_id", default=None, help="also run the real processor (Qwen3-VL or InternVL3.5-HF)")
    ap.add_argument("--out_dir", default="viz")
    ap.add_argument("--panel_px", type=int, default=160)
    a = ap.parse_args()
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(a.out_dir, exist_ok=True)
    path = a.npz or os.path.join(a.dir, list_npz(a.dir)[a.index])
    print("=" * 78 + f"\nFILE {path}")

    # ---- stage 1: the file
    z = np.load(path)
    img = z["image"]
    print("\n[1] .npz contents")
    for k in z.files:
        v = z[k]
        if v.dtype.kind in "iufb":
            print(f"    {k:<28} dtype={str(v.dtype):<7} shape={str(v.shape):<14} min={v.min()} max={v.max()}")
    target = int(z["target"])

    # ---- stage 2: panels
    panels = to_panels(img)
    ctx, cand = panels[:8], panels[8:]
    print(f"\n[2] panels: stored {img.shape} -> reshape -> {panels.shape} (16 panels of 160x160 grey values 0-255)")
    print(f"    context panels  = panels[0:8]   shape {ctx.shape}")
    print(f"    candidate panels = panels[8:16] shape {cand.shape}")
    print(f"    correct answer: candidate number {target + 1} (0-based index {target})")

    # ---- figure 1: all 16 panels
    fig, axs = plt.subplots(4, 4, figsize=(9, 9))
    for i, ax in enumerate(axs.ravel()):
        ax.imshow(panels[i], cmap="gray", vmin=0, vmax=255)
        ax.set_xticks([]); ax.set_yticks([])
        if i < 8:
            ax.set_title(f"context {i + 1}", fontsize=9)
        else:
            ok = (i - 8) == target
            ax.set_title(f"candidate {i - 7}" + ("  <- ANSWER" if ok else ""), fontsize=9, color="green" if ok else "black")
            if ok:
                for s in ax.spines.values():
                    s.set_edgecolor("green"); s.set_linewidth(3)
    fig.suptitle("The 16 panels in the file (160x160 each)\nrows 1-2: context (3x3 matrix, 9th cell is the missing one) | rows 3-4: candidates")
    fig.tight_layout(); fig.savefig(os.path.join(a.out_dir, "1_all16_panels.png"), dpi=110); plt.close(fig)

    # ---- stage 3: the 8 completed matrices (one model input each)
    mats = filled_matrices(ctx, cand, a.panel_px)
    print(f"\n[3] the 8 images given to the processor, one sequence each (candidate k in the bottom-right cell, no labels)")
    print(f"    each image: {mats[0].size} (W x H) mode {mats[0].mode}; panels drawn at {a.panel_px}px with a 10 px white gap")
    for k, im in enumerate(mats):
        im.save(os.path.join(a.out_dir, f"filled_{k + 1}{'_ANSWER' if k == target else ''}.png"))
    fig, axs = plt.subplots(2, 4, figsize=(14, 7.5))
    for k, ax in enumerate(axs.ravel()):
        ax.imshow(mats[k]); ax.set_xticks([]); ax.set_yticks([])
        ok = k == target
        ax.set_title(f"candidate {k + 1}" + ("  <- ANSWER (should score Yes)" if ok else ""), fontsize=9,
                     color="green" if ok else "black")
    fig.suptitle("The 8 model inputs for this problem: score(k) = logit(Yes) - logit(No), softmax over k")
    fig.tight_layout(); fig.savefig(os.path.join(a.out_dir, "2_model_images.png"), dpi=100); plt.close(fig)

    print("\n[4] text prompt (user turn), after the image:\n    " + PROMPT_TEXT.replace("\n", "\n    "))
    print('    the score is read at the first assistant position: logit("Yes") - logit("No")')

    if not a.model_id:
        print(f"\nPNGs written to {a.out_dir}   (add --model_id to also see the tokenised input)")
        return

    # ---- stage 5: the real processor
    import re
    import torch
    from model_utils import ScoreTokens, detect_family
    from pgm_vlm_data import CandidateCollator
    from transformers import AutoProcessor
    fam = detect_family(a.model_id)
    proc = AutoProcessor.from_pretrained(a.model_id)
    batch = CandidateCollator(proc, fam, crop_to_patches=True, cell_size=a.panel_px)(
        [dict(context=ctx, candidates=cand, target=target, file=os.path.basename(path))])
    ids = batch["input_ids"][0]
    tok = proc.tokenizer
    img_id = getattr(proc, "image_token_id", None)
    if img_id is None:
        img_id = tok.convert_tokens_to_ids(getattr(proc, "image_token", "<IMG_CONTEXT>"))
    n_img = int((ids == img_id).sum())
    print(f"\n[5] after the processor ({a.model_id}, family {fam}): {batch['input_ids'].shape[0]} sequences for 1 problem")
    for k, v in batch.items():
        if torch.is_tensor(v):
            extra = f" min={v.min().item():.3g} max={v.max().item():.3g}" if v.dtype.is_floating_point else ""
            print(f"    {k:<20} shape={tuple(v.shape)!s:<16} dtype={v.dtype}{extra}")
    print(f"    sequence length {ids.numel()} tokens, of which {n_img} are image tokens")
    if "image_grid_thw" in batch:
        t, h, w = batch["image_grid_thw"][0].tolist()
        print(f"    image resized to {w * 16}x{h * 16} px -> {h * w // 4} tokens ({h // 2} x {w // 2} token grid, 32x32 px each)")
    text = re.sub(rf"({re.escape(tok.decode([img_id]))})+", lambda m: f"<{len(m.group(0)) // max(1, len(tok.decode([img_id])))} image tokens>", tok.decode(ids))
    print("\n    decoded input of candidate 1 (image tokens collapsed):\n    " + text.replace("\n", "\n    "))
    print(f"\n    score tokens: {ScoreTokens(tok).describe()}")

    if "image_grid_thw" in batch:                    # Qwen: draw one cell per visual token on the resized image
        t, h, w = batch["image_grid_thw"][0].tolist()
        fig, ax = plt.subplots(figsize=(7, 7))
        ax.imshow(mats[0].resize((w * 16, h * 16)))
        for x in range(0, w * 16 + 1, 32):
            ax.axvline(x - 0.5, color="red", lw=0.4, alpha=0.6)
        for y in range(0, h * 16 + 1, 32):
            ax.axhline(y - 0.5, color="red", lw=0.4, alpha=0.6)
        ax.set_title(f"as the model sees it: {w * 16}x{h * 16} px = {h * w // 4} tokens (each red cell = one visual token)", fontsize=9)
        ax.set_xticks([]); ax.set_yticks([])
        fig.tight_layout(); fig.savefig(os.path.join(a.out_dir, "3_token_grid.png"), dpi=110); plt.close(fig)
    print(f"\nPNGs written to {a.out_dir}")


if __name__ == "__main__":
    main()
