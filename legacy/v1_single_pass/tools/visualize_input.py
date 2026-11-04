"""Show exactly what goes into the model for ONE PGM problem, stage by stage, with shapes and sizes, and save figures.

  python tools/visualize_input.py --dir $DATA/val_full/neutral --index 0 --out_dir $RUNS/viz                      # images only
  python tools/visualize_input.py --dir $DATA/val_full/neutral --index 0 --out_dir $RUNS/viz --model_id Qwen/Qwen3-VL-2B-Instruct
(the second form also runs the real processor: needs the model in the offline cache, no GPU)

Output: printed shapes/sizes at every stage, and PNGs
  1_all16_panels.png     the 16 panels as stored in the file (context / candidates, answer marked)
  2_model_images.png     the two images the model receives (exact pixel sizes)
  3_token_grid.png       (with --model_id, Qwen only) the image as resized by the processor with one cell per visual token
  context.png, candidates.png   the two images on their own
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pgm_vlm_data import PROMPT_TEXT, list_npz, panels_to_grid_image, to_panels  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", default=None, help="one .npz file")
    ap.add_argument("--dir", default=None, help="folder of .npz files (use with --index)")
    ap.add_argument("--index", type=int, default=0)
    ap.add_argument("--model_id", default=None, help="also run the real processor (Qwen3-VL or InternVL3.5-HF)")
    ap.add_argument("--out_dir", default="viz")
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

    # ---- stage 3: the two images
    im_ctx = panels_to_grid_image(ctx, (3, 3), add_labels=False)
    im_cand = panels_to_grid_image(cand, (2, 4), add_labels=True, label_offset=1)
    print(f"\n[3] the two PIL images given to the processor")
    print(f"    image 1 (context, 3x3 canvas, 8 panels + empty cell): size {im_ctx.size} (W x H) mode {im_ctx.mode}")
    print(f"    image 2 (candidates, 2x4 canvas, yellow number tags): size {im_cand.size} (W x H) mode {im_cand.mode}")
    print("    each panel is pasted at 160x160 with a 10 px white gap; panel pixels are grey (R=G=B)")
    im_ctx.save(os.path.join(a.out_dir, "context.png")); im_cand.save(os.path.join(a.out_dir, "candidates.png"))
    fig, axs = plt.subplots(1, 2, figsize=(13, 5), gridspec_kw={"width_ratios": [im_ctx.size[0], im_cand.size[0]]})
    axs[0].imshow(im_ctx); axs[0].set_title(f"image 1: context  {im_ctx.size[0]}x{im_ctx.size[1]} px")
    axs[1].imshow(im_cand); axs[1].set_title(f"image 2: candidates  {im_cand.size[0]}x{im_cand.size[1]} px  (answer = {target + 1})")
    for ax in axs:
        ax.set_xticks([]); ax.set_yticks([])
    fig.tight_layout(); fig.savefig(os.path.join(a.out_dir, "2_model_images.png"), dpi=110); plt.close(fig)

    print("\n[4] text prompt (user turn), after the two images:\n    " + PROMPT_TEXT.replace("\n", "\n    "))
    print('    the model must then continue with:  "Answer: <digit>"  followed by the end-of-turn token')

    if not a.model_id:
        print(f"\nPNGs written to {a.out_dir}   (add --model_id to also see the tokenised input)")
        return

    # ---- stage 5: the real processor
    import torch
    from model_utils import AnswerTokens, detect_family
    from pgm_vlm_data import VLMCollator
    from transformers import AutoProcessor
    fam = detect_family(a.model_id)
    proc = AutoProcessor.from_pretrained(a.model_id)
    col = VLMCollator(proc, fam, crop_to_patches=True)
    batch = col([dict(context=ctx, candidates=cand, target=target, file=os.path.basename(path))])
    ids = batch["input_ids"][0]
    tok = proc.tokenizer
    img_id = getattr(proc, "image_token_id", None)
    if img_id is None:
        img_id = tok.convert_tokens_to_ids(getattr(proc, "image_token", "<IMG_CONTEXT>"))
    n_img = int((ids == img_id).sum())
    print(f"\n[5] after the processor ({a.model_id}, family {fam})")
    for k, v in batch.items():
        if torch.is_tensor(v):
            extra = f" min={v.min().item():.3g} max={v.max().item():.3g}" if v.dtype.is_floating_point else ""
            print(f"    {k:<20} shape={tuple(v.shape)!s:<16} dtype={v.dtype}{extra}")
    print(f"    sequence length {ids.numel()} tokens, of which {n_img} are image tokens (the rest: chat template + prompt text)")
    if "image_grid_thw" in batch:
        g = batch["image_grid_thw"]
        print(f"    image_grid_thw (frames, patch rows, patch cols; patch = 16 px): {g.tolist()}")
        for j, (t, h, w) in enumerate(g.tolist()):
            print(f"      image {j + 1}: resized to {w * 16}x{h * 16} px -> {h * w} patches -> {h * w // 4} tokens "
                  f"after 2x2 merge ({h // 2} x {w // 2} token grid, each token covers 32x32 px)")
    text = tok.decode(ids)
    import re
    text = re.sub(rf"({re.escape(tok.decode([img_id]))})+", lambda m: f"<{len(m.group(0)) // max(1, len(tok.decode([img_id])))} image tokens>", text)
    print("\n    decoded input (image tokens collapsed):\n    " + text.replace("\n", "\n    "))
    ans = AnswerTokens(tok)
    print(f"\n    answer tokens appended during training: {tok.convert_ids_to_tokens(ans.full[target].tolist())}  (digit is token #{ans.digit_pos + 1})")

    if "image_grid_thw" in batch:                    # Qwen: draw one cell per visual token on the resized image
        fig, axs = plt.subplots(1, 2, figsize=(14, 5.5))
        for j, (im, (t, h, w)) in enumerate(zip((im_ctx, im_cand), batch["image_grid_thw"].tolist())):
            rs = im.resize((w * 16, h * 16))
            axs[j].imshow(rs)
            for x in range(0, w * 16 + 1, 32):
                axs[j].axvline(x - 0.5, color="red", lw=0.4, alpha=0.6)
            for y in range(0, h * 16 + 1, 32):
                axs[j].axhline(y - 0.5, color="red", lw=0.4, alpha=0.6)
            axs[j].set_title(f"image {j + 1} as the model sees it: {w * 16}x{h * 16} px = {h // 2}x{w // 2} = {h * w // 4} tokens\n"
                             f"(each red cell = one visual token = 32x32 px; a 160 px panel is ~5x5 tokens)", fontsize=9)
            axs[j].set_xticks([]); axs[j].set_yticks([])
        fig.tight_layout(); fig.savefig(os.path.join(a.out_dir, "3_token_grid.png"), dpi=110); plt.close(fig)
    print(f"\nPNGs written to {a.out_dir}")


if __name__ == "__main__":
    main()
