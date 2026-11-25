"""Data: every task becomes one or more (image, prompt) SEQUENCES read out with the VLM's own LM head.

Example kinds
  cand    K sequences, one per candidate (image = the problem completed with candidate k), score_k = logit(Yes) - logit(No),
          softmax over K + CE.  PGM / I-RAVEN / RAVEN-FAIR / V-PROM style matrices (same format as the LoRA baseline).
  binary  1 sequence, Yes/No, BCE on logit(Yes) - logit(No).  VSR, SVRT, CLEVR yes/no, ART same/different.
  letter  1 sequence, CE over the logits of the choice letters at the answer position.  Contact-sheet multiple choice
          (the CLEVR-ART OOD package), What'sUp, CV-Bench, BLINK.
Every image carries PANEL BOXES (normalised x0, y0, x1, y1): the regions the relational branch runs slot attention on.
PGM: the 9 cells of the matrix. A single natural image: [[0, 0, 1, 1]].

Sources
  matrix:<dir|zip>  PGM / RAVEN .npz files (16 x 160 x 160 panels + target), rendered exactly like vlm_lora_pc; a .zip of
                    .npz members works too (packed datasets: scratch has a 1M file-count quota)
  jsonl:<file>   one example per line. Image refs are paths relative to the file's directory, or
                 "zip:<archive>::<member>" (archive relative to the file's directory) for packed images:
       letter : {"images": [p], "prompt": str, "choices": ["A", "B", ...], "answer": "B"}            (CLEVR-ART package)
       binary : {"images": [p], "prompt": str, "answer": "Yes" | "No"}
       cand   : {"candidates": [p0, p1, ...], "prompt": str, "answer": 2}   (optional "prompts": one per candidate)
       image  : {"images": [p]}  (no answer: unlabelled, Stage 0 only)
       optional "panels": [[x0, y0, x1, y1], ...] (shared by all images of the example)
A training run mixes sources with weights:  --train name=matrix:/path/train_full/neutral:0.7 --train name=jsonl:x.jsonl:0.3
"""
import hashlib
import io
import json
import os
import zipfile

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from backbones import REL_TOKEN

PGM_PROMPT = (
    "The image shows a 3x3 abstract reasoning matrix. In a correct matrix, shape type, size, color, count, and position, "
    "and line type and color, follow consistent rules across rows and columns (constant, progression, XOR, OR, AND, or "
    "consistent union).\n\n"
    "Is the bottom-right panel the correct completion of this matrix? Answer Yes or No."
)   # identical to vlm_lora_pc, so scores are comparable with the LoRA baseline
RAVEN_PROMPT = (
    "The image shows a 3x3 abstract reasoning matrix. In a correct matrix, the number, position, type, size and color "
    "of the objects follow consistent rules across each row (constant, progression, arithmetic, or distribute three).\n\n"
    "Is the bottom-right panel the correct completion of this matrix? Answer Yes or No."
)
WHOLE = [[0.0, 0.0, 1.0, 1.0]]


# ----------------------------------------------------------------------------- files
_ZIPS = {}   # per-process open archives (each DataLoader worker opens its own)


def _zip(path):
    key = (os.getpid(), path)
    if key not in _ZIPS:
        _ZIPS[key] = zipfile.ZipFile(path)
    return _ZIPS[key]


def read_bytes(ref, root=""):
    """'zip:<archive>::<member>' or a plain path; relative paths are taken from root."""
    if ref.startswith("zip:"):
        arc, member = ref[4:].split("::", 1)
        return _zip(arc if os.path.isabs(arc) else os.path.join(root, arc)).read(member)
    with open(ref if os.path.isabs(ref) else os.path.join(root, ref), "rb") as f:
        return f.read()


def load_image(ref, root=""):
    return Image.open(io.BytesIO(read_bytes(ref, root))).convert("RGB")


def list_npz(npz_dir, cache_path=None):
    if npz_dir.endswith(".zip"):
        return sorted(n for n in zipfile.ZipFile(npz_dir).namelist() if n.endswith(".npz"))
    if cache_path and os.path.exists(cache_path):
        with open(cache_path) as f:
            return [l.rstrip("\n") for l in f if l.strip()]
    files = sorted(f for f in os.listdir(npz_dir) if f.endswith(".npz"))
    if cache_path:
        os.makedirs(os.path.dirname(os.path.abspath(cache_path)), exist_ok=True)
        with open(cache_path, "w") as f:
            f.write("\n".join(files))
    return files


def subset(items, n, seed=0):
    """First n of one fixed seeded permutation (nested: the 1% subset is inside the 5% subset)."""
    if n <= 0 or n >= len(items):
        return list(items)
    perm = np.random.default_rng(seed).permutation(len(items))[:n]
    return [items[i] for i in sorted(perm)]


# ----------------------------------------------------------------------------- matrices
def to_panels(img):
    """(16, 160, 160). PGM files declare (160, 160, 16) but are panel-major: reshape, never transpose."""
    img = np.asarray(img)
    if img.size != 16 * 160 * 160:
        raise ValueError(f"unexpected matrix image shape {img.shape}")
    return img.reshape(16, 160, 160)


def grid_layout(rows, cols, cell=160, pad=10):
    W, H = cols * cell + (cols + 1) * pad, rows * cell + (rows + 1) * pad
    boxes = []
    for i in range(rows * cols):
        r, c = divmod(i, cols)
        x, y = pad + c * (cell + pad), pad + r * (cell + pad)
        boxes.append([x / W, y / H, (x + cell) / W, (y + cell) / H])
    return (W, H), boxes


def render_grid(panels, rows=3, cols=3, cell=160, pad=10):
    """Same canvas as vlm_lora_pc.panels_to_grid_image (white background, 10 px gaps). Returns (image, boxes)."""
    (W, H), boxes = grid_layout(rows, cols, cell, pad)
    canvas = Image.new("RGB", (W, H), (255, 255, 255))
    for i, p in enumerate(panels):
        im = Image.fromarray(np.asarray(p).astype(np.uint8), mode="L").convert("RGB")
        if im.size != (cell, cell):
            im = im.resize((cell, cell), Image.BICUBIC)
        r, c = divmod(i, cols)
        canvas.paste(im, (pad + c * (cell + pad), pad + r * (cell + pad)))
    return canvas, boxes


class MatrixSource:
    """PGM or RAVEN .npz: each problem -> 8 sequences (context + candidate k in the bottom-right cell)."""

    def __init__(self, npz_dir, files=None, prompt=PGM_PROMPT, cell=160, context_mode="normal", file_cache=None):
        self.dir, self.prompt, self.cell, self.context_mode = npz_dir, prompt, cell, context_mode
        self.files = list(files) if files is not None else list_npz(npz_dir, file_cache)
        self.row_swap = 0.0      # training augmentation (set by train_stage1 --row_swap_p): swap context rows 1 and 2

    def __len__(self):
        return len(self.files)

    def get(self, i, rng):
        f = self.files[i]
        if self.dir.endswith(".zip"):
            d = np.load(io.BytesIO(_zip(self.dir).read(f)))
        else:
            d = np.load(os.path.join(self.dir, f))
        p = to_panels(d["image"])
        ctx, cand, target = p[:8], p[8:], int(d["target"])
        if self.context_mode == "blank":
            ctx = np.zeros_like(ctx)
        elif self.context_mode == "shuffle":
            ctx = ctx[rng.permutation(8)]
        if self.row_swap and rng.random() < self.row_swap:   # rows are rule-independent: each obeys the rules alone
            ctx = ctx[[3, 4, 5, 0, 1, 2, 6, 7]]
        seqs = []
        for k in range(8):
            img, boxes = render_grid(list(ctx) + [cand[k]], cell=self.cell)
            seqs.append(dict(image=img, boxes=boxes, prompt=self.prompt))
        return dict(kind="cand", seqs=seqs, label=target, id=f)


class JsonlSource:
    def __init__(self, path, prompt_override=None):
        self.root = os.path.dirname(os.path.abspath(path))
        with open(path) as f:
            self.rows = [json.loads(l) for l in f if l.strip()]
        self.prompt_override = prompt_override

    def __len__(self):
        return len(self.rows)

    def _img(self, p):
        return load_image(p, self.root)

    def get(self, i, rng):
        r = self.rows[i]
        prompt = self.prompt_override or r.get("prompt")
        boxes = r.get("panels", WHOLE)
        rid = r.get("id", str(i))
        if "candidates" in r:
            prompts = r.get("prompts") or [prompt] * len(r["candidates"])    # per-candidate prompts (What'sUp)
            seqs = [dict(image=self._img(p), boxes=boxes, prompt=q) for p, q in zip(r["candidates"], prompts)]
            return dict(kind="cand", seqs=seqs, label=int(r["answer"]), id=rid)
        seq = [dict(image=self._img(r["images"][0]), boxes=boxes, prompt=prompt or "")]
        if "answer" not in r:
            return dict(kind="image", seqs=seq, label=None, id=rid)
        ans = str(r["answer"])
        if ans.lower() in ("yes", "no", "true", "false") and r.get("choices") in (None, ["Yes", "No"]):
            return dict(kind="binary", seqs=seq, label=int(ans.lower() in ("yes", "true")), id=rid)
        return dict(kind="letter", seqs=seq, label=r["choices"].index(ans), choices=list(r["choices"]), id=rid)


def make_source(spec, **kw):
    """'matrix:<dir>' | 'raven:<dir>' | 'jsonl:<file>'."""
    typ, path = spec.split(":", 1)
    if typ == "matrix":
        return MatrixSource(path, **kw)
    if typ == "raven":
        return MatrixSource(path, prompt=RAVEN_PROMPT, **kw)
    if typ == "jsonl":
        return JsonlSource(path)
    raise ValueError(f"unknown source type {typ!r} in {spec!r}")


def build_sources(mix, panel_px=160, file_cache_dir=None, context_mode="normal"):
    """Sources for parse_mix() output. Matrix listings (1.2M files) are cached in file_cache_dir/<name>_files.txt."""
    srcs = []
    for name, spec, _ in mix:
        kw = {}
        if spec.split(":", 1)[0] in ("matrix", "raven"):
            tag = hashlib.md5(os.path.abspath(spec.split(":", 1)[1]).encode()).hexdigest()[:8]   # train/val dirs differ
            kw = dict(cell=panel_px, context_mode=context_mode,
                      file_cache=os.path.join(file_cache_dir, f"{name}_{tag}_files.txt") if file_cache_dir else None)
        srcs.append(make_source(spec, **kw))
    return srcs


def parse_mix(specs):
    """['name=matrix:/dir:0.7', ...] -> [(name, 'matrix:/dir', 0.7)]. Weight optional (default 1)."""
    out = []
    for s in specs:
        name, rest = s.split("=", 1)
        typ, tail = rest.split(":", 1)
        w = 1.0
        if ":" in tail and _is_float(tail.rsplit(":", 1)[1]):
            tail, w = tail.rsplit(":", 1)[0], float(tail.rsplit(":", 1)[1])
        out.append((name, f"{typ}:{tail}", w))
    return out


def _is_float(s):
    try:
        float(s)
        return True
    except ValueError:
        return False


class MixtureStream(Dataset):
    """Indexed by POSITION in the training stream: the (source, item) at every position is fixed up front from the seed,
    so a run resumed at step s sees exactly the data an uninterrupted run would. Sources are drawn by weight; within a
    source items run through concatenated per-epoch permutations."""

    def __init__(self, sources, weights, names, total, seed=0):
        self.sources, self.names, self.seed = sources, names, seed
        rng = np.random.default_rng([seed, 7])
        p = np.asarray(weights, dtype=np.float64)
        self.src = rng.choice(len(sources), size=total, p=p / p.sum())
        self.item = np.zeros(total, dtype=np.int64)
        for s, src in enumerate(sources):
            pos = np.nonzero(self.src == s)[0]
            n, chunks, have, ep = len(src), [], 0, 0
            while have < len(pos):
                chunks.append(np.random.default_rng([seed, 1000 + s, ep]).permutation(n))
                have, ep = have + n, ep + 1
            self.item[pos] = np.concatenate(chunks)[: len(pos)] if chunks else []

    def __len__(self):
        return len(self.src)

    def __getitem__(self, pos):
        s, i = int(self.src[pos]), int(self.item[pos])
        ex = self.sources[s].get(i, np.random.default_rng([self.seed, int(pos)]))
        ex["source"] = self.names[s]
        return ex


class FixedSet(Dataset):
    """A plain list of items of one source (validation / test, sharded)."""

    def __init__(self, source, indices, name="eval", seed=0):
        self.source, self.idx, self.name, self.seed = source, list(indices), name, seed

    def __len__(self):
        return len(self.idx)

    def __getitem__(self, j):
        ex = self.source.get(self.idx[j], np.random.default_rng([self.seed, int(self.idx[j])]))
        ex["source"] = self.name
        return ex


# ----------------------------------------------------------------------------- collator
class Collator:
    """B examples -> all their sequences in one processor call (left padding), plus the per-image panel boxes and the
    example structure. rel_tokens > 0 inserts <|rel|> placeholders right after the image: rel_tokens of them (rel_mode pool),
    one per panel (panel), or one per slot (slot)."""

    def __init__(self, processor, processor_kwargs=None, rel_tokens=0, rel_mode="pool", num_slots=0):
        self.processor, self.kw, self.M = processor, processor_kwargs or {}, rel_tokens
        self.rel_mode, self.K = rel_mode, num_slots
        processor.tokenizer.padding_side = "left"

    def text(self, prompt, n_panels=1):
        if self.M > 0:
            n = {"pool": self.M, "panel": n_panels, "slot": n_panels * self.K}[self.rel_mode]
            prompt = "Relational summary: " + REL_TOKEN * n + "\n" + prompt
        msgs = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt}]}]
        return self.processor.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)

    def __call__(self, items):
        texts, images, boxes, examples = [], [], [], []
        for ex in items:
            start = len(images)
            for s in ex["seqs"]:
                texts.append(self.text(s["prompt"], len(s["boxes"])))
                images.append(s["image"])
                boxes.append(torch.tensor(s["boxes"], dtype=torch.float32))
            examples.append(dict(kind=ex["kind"], label=ex["label"], choices=ex.get("choices"), id=ex["id"],
                                 source=ex.get("source", ""), seq_idx=list(range(start, len(images)))))
        batch = dict(self.processor(text=texts, images=images, padding=True, return_tensors="pt", **self.kw))
        batch["boxes"], batch["examples"] = boxes, examples
        return batch
