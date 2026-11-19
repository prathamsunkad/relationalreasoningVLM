"""PGM -> VLM inputs, PER-CANDIDATE version (like the Slot Abstractor / WReN papers).

For every problem the model gets 8 separate sequences, one per answer choice k:
  image : the full 3x3 matrix, 8 context panels + candidate k in the bottom-right cell (no labels, no number tags)
  prompt: "is this matrix correct? Answer Yes or No"
The score of candidate k is logit(Yes) - logit(No) at the answer position (see model_utils.candidate_scores); the 8 scores
go through a softmax and cross-entropy with the correct choice. The model never has to point at a tagged panel: each
sequence asks one local question, and the 8 near-identical matrices give a direct contrast between right and wrong.
"""
import os

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from torch.utils.data import Dataset

PROMPT_TEXT = (
    "The image shows a 3x3 abstract reasoning matrix. In a correct matrix, shape type, size, color, count, and position, "
    "and line type and color, follow consistent rules across rows and columns (constant, progression, XOR, OR, AND, or "
    "consistent union).\n\n"
    "Is the bottom-right panel the correct completion of this matrix? Answer Yes or No."
)
N_CAND = 8


# ----------------------------------------------------------------------------- files / sharding
def list_npz(npz_dir, cache_path=None):
    """Sorted *.npz names (sorted so sharding matches pgm_infer.py). cache_path avoids re-listing 1.2M files."""
    if cache_path and os.path.exists(cache_path):
        with open(cache_path) as f:
            return [l.rstrip("\n") for l in f if l.strip()]
    files = sorted(f for f in os.listdir(npz_dir) if f.endswith(".npz"))
    if cache_path:
        os.makedirs(os.path.dirname(os.path.abspath(cache_path)), exist_ok=True)
        with open(cache_path, "w") as f:
            f.write("\n".join(files))
    return files


def shard_files(files, shard_idx=0, num_shards=1, max_examples=0):
    """Same scheme as pgm_infer.py: every num_shards-th file of the sorted list."""
    out = files[shard_idx::num_shards]
    return out[:max_examples] if max_examples > 0 else out


def subset_files(files, n, seed=0):
    """First n of one fixed seeded permutation, so the 1% subset is contained in the 5% subset, which is contained in the 20% subset (nested)."""
    if n <= 0 or n >= len(files):
        return list(files)
    perm = np.random.default_rng(seed).permutation(len(files))[:n]
    return [files[i] for i in sorted(perm)]


# ----------------------------------------------------------------------------- images
def to_panels(img):
    """(16, 160, 160) panels from a stored PGM image. The released files declare shape (160, 160, 16), but the bytes are
    panel-major, so a plain reshape is correct. Do NOT transpose (2, 0, 1): that scrambles every panel into stripes
    (checked by rendering both on neutral train/val/test files)."""
    img = np.asarray(img)
    if img.size != 16 * 160 * 160:
        raise ValueError(f"unexpected PGM image shape {img.shape}; expected 16*160*160 values")
    return img.reshape(16, 160, 160)


def load_pgm_instance(npz_path, image_key="image", target_key="target"):
    data = np.load(npz_path)
    panels = to_panels(data[image_key])
    return panels[:8], panels[8:], int(data[target_key])


_FONT = None


def _font():
    global _FONT
    if _FONT is None:
        try:
            _FONT = ImageFont.truetype("DejaVuSans-Bold.ttf", 24)
        except OSError:
            _FONT = ImageFont.load_default()
    return _FONT


def panels_to_grid_image(panels, grid_shape, cell_size=160, pad=10, add_labels=False, label_offset=1):
    """Panels pasted on a white canvas. cell_size != 160 resizes each panel (more visual tokens per panel)."""
    rows, cols = grid_shape
    width = cols * cell_size + (cols + 1) * pad
    height = rows * cell_size + (rows + 1) * pad
    canvas = Image.new("RGB", (width, height), color=(255, 255, 255))
    draw = ImageDraw.Draw(canvas)
    font = _font() if add_labels else None
    for i, panel in enumerate(panels):
        row, col = divmod(i, cols)
        panel_img = Image.fromarray(np.asarray(panel).astype(np.uint8), mode="L").convert("RGB")
        if panel_img.size != (cell_size, cell_size):
            panel_img = panel_img.resize((cell_size, cell_size), Image.BICUBIC)
        x = pad + col * (cell_size + pad)
        y = pad + row * (cell_size + pad)
        canvas.paste(panel_img, (x, y))
        if add_labels:
            draw.rectangle([x, y, x + 28, y + 28], fill=(255, 255, 0))
            draw.text((x + 5, y + 2), str(i + label_offset), fill=(0, 0, 0), font=font)
    return canvas


def filled_matrices(context, candidates, cell_size=160):
    """The 8 complete 3x3 matrices: context panels + candidate k in the bottom-right cell, k = 0..7."""
    return [panels_to_grid_image(list(context) + [candidates[k]], (3, 3), cell_size=cell_size) for k in range(len(candidates))]


# ----------------------------------------------------------------------------- dataset
class PGMStream(Dataset):
    """Indexed by *position in the training stream* (not file index). `order[pos]` says which file that position
    shows, so the data order is a pure function of (seed, pos): a run resumed at step s sees exactly the data an
    uninterrupted run would have seen. Candidate shuffling (answer remapped) is also a function of (seed, pos); with
    per-candidate scoring the order of the 8 candidates does not change what the model sees, so it is off by default.

    context_mode (evaluation sanity checks): normal | blank (context panels zeroed) | shuffle (context panels permuted).
    """

    def __init__(self, npz_dir, files, order=None, seed=0, shuffle_candidates=False, context_mode="normal"):
        self.dir, self.files = npz_dir, list(files)
        self.order = np.arange(len(self.files)) if order is None else np.asarray(order)
        self.seed, self.shuffle_candidates, self.context_mode = seed, shuffle_candidates, context_mode

    def __len__(self):
        return len(self.order)

    def __getitem__(self, pos):
        fname = self.files[int(self.order[pos])]
        ctx, cand, target = load_pgm_instance(os.path.join(self.dir, fname))
        rng = np.random.default_rng([self.seed, int(pos)])
        if self.shuffle_candidates:
            perm = rng.permutation(8)
            cand = cand[perm]
            target = int(np.where(perm == target)[0][0])
        if self.context_mode == "blank":
            ctx = np.zeros_like(ctx)
        elif self.context_mode == "shuffle":
            ctx = ctx[rng.permutation(8)]
        return dict(context=ctx, candidates=cand, target=target, file=fname)


def make_order(n_files, total_positions, seed):
    """Concatenated per-epoch permutations, truncated to total_positions."""
    chunks, have, epoch = [], 0, 0
    while have < total_positions:
        chunks.append(np.random.default_rng([seed, 10_000 + epoch]).permutation(n_files))
        have += n_files
        epoch += 1
    return np.concatenate(chunks)[:total_positions]


class CandidateCollator:
    """B problems -> 8*B sequences, problem-major (problem 0 candidates 0..7, problem 1 candidates 0..7, ...), so the
    model's 8*B scores reshape to (B, 8). All images have the same size, so sequences have equal length (left padding
    is still set)."""

    def __init__(self, processor, family, crop_to_patches=True, prompt=PROMPT_TEXT, cell_size=160):
        self.processor, self.family, self.cell_size = processor, family, cell_size
        self.processor.tokenizer.padding_side = "left"
        messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt}]}]
        self.text = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        # InternVL's processor forces crop_to_patches=True unless told otherwise (its preprocessor_config says False,
        # the processor default overrides it), so we always pass it explicitly.
        self.extra = {"crop_to_patches": bool(crop_to_patches)} if family == "internvl" else {}

    def __call__(self, items):
        images = []
        for it in items:
            images += filled_matrices(it["context"], it["candidates"], self.cell_size)
        batch = self.processor(text=[self.text] * len(images), images=images, padding=True, return_tensors="pt", **self.extra)
        batch = dict(batch)
        batch["targets"] = torch.tensor([it["target"] for it in items], dtype=torch.long)
        batch["files"] = [it["file"] for it in items]
        return batch
