#!/usr/bin/env python
"""
Perception-vs-reasoning gap experiments for VLMs on PGM contact sheets.

Works with any Hugging Face image-text-to-text checkpoint that loads through
AutoProcessor + AutoModelForImageTextToText, e.g.
    Qwen/Qwen3-VL-2B-Instruct
    OpenGVLab/InternVL3_5-1B-HF        (the "-HF" transformers-native release)
and with your own fine-tuned checkpoints saved in the same format.

Stages (run them in this order; each writes per-puzzle JSONL so runs can resume
and be sharded across GPUs):

  image     End-to-end baseline: the dataset's exact prompt + contact sheet,
            scored by the log-probability of the letters A-H.
              --variant crops : the same puzzle as 16 separate upscaled panels.
                                The REASONING test paired with `match --variant crops`.
  match     PERCEPTION CONTROLS (no reasoning needed; chance = 12.5%).
              --variant sheet : the correct answer is pasted into the "?" cell
                                of the original sheet; "which choice is identical
                                to that panel?" Same pixels/resolution as `image`.
              --variant crops : reference panel + the 8 choices as 9 separate,
                                upscaled images. Same question.
  samediff  PURE DISCRIMINATION: two panels, "same or not?" (Yes/No). Pairs are the
            puzzle's answer vs each distractor (different) and each choice vs itself
            (same); every image is shifted by up to --jitter px. Scored as AUC, with
            a breakdown by how many pixels differ (a psychometric curve).
  probe     ENCODED vs REPORTED: a linear same/different probe on the frozen vision
            features of the same pairs (trained on --probe_train training puzzles),
            plus the same probe on raw 32x32 pixels as a reference.
  describe  Each of the 16 panels is cropped, upscaled and described alone as
            structured JSON (cached by pixel hash). Reports parse rate and
            whether the descriptions still tell the 8 choices apart.
  text      TEXT REASONING: the model (or any text LLM via --causal) gets only
            the 16 descriptions and picks A-H. --rules adds the puzzle's
            relation structure from the source metadata as a hint.
  solve     INFORMATION SUFFICIENCY: a small relational network is trained on
            the descriptions of training puzzles and tested on validation.
            A choices-only network (no context) is trained as a shortcut check.
            CPU is enough; no VLM is loaded.
  report    One table per model with 95% Wilson intervals.

Typical run (one model; repeat with the other --model):

  export DATA=/path/to/pgm-neutral-train20pct-seed20261007-v1
  (--data can be left out when DATA is exported; run_gap.sh does all of this for both models)
  python infer_layout.py --data $DATA --jsonl $DATA/validation.jsonl --out layout.json
  M=Qwen/Qwen3-VL-2B-Instruct
  python pgm_gap.py image    --data $DATA --layout layout.json --model $M --split validation
  python pgm_gap.py match    --data $DATA --layout layout.json --model $M --split validation --variant sheet
  python pgm_gap.py match    --data $DATA --layout layout.json --model $M --split validation --variant crops
  python pgm_gap.py image    --data $DATA --layout layout.json --model $M --split validation --variant crops --batch_size 4
  python pgm_gap.py samediff --data $DATA --layout layout.json --model $M --split validation
  python pgm_gap.py probe    --data $DATA --layout layout.json --model $M --split validation
  python pgm_gap.py describe --data $DATA --layout layout.json --model $M --split validation
  python pgm_gap.py describe --data $DATA --layout layout.json --model $M --split train --limit 20000
  python pgm_gap.py text     --model $M --split validation
  python pgm_gap.py text     --model $M --split validation --rules
  python pgm_gap.py solve    --model $M
  python pgm_gap.py report

Outputs go to --out (default runs/pgm_gap/<model-tag>/). Use --dump 5 on any
image stage to save the crops/inputs of the first 5 puzzles for a visual check;
do that before any long run.

Sharding: add --shard i --num_shards n (e.g. one SLURM array task per GPU).
Each shard writes its own file; `report`, `text` and `solve` merge them.
"""
import argparse
import glob
import hashlib
import json
import math
import os
import re
import sys
import time
from collections import Counter

import numpy as np
from PIL import Image, ImageDraw

LETTERS = "ABCDEFGH"
POS9 = ["top-left", "top-center", "top-right",
        "middle-left", "center", "middle-right",
        "bottom-left", "bottom-center", "bottom-right"]
SHAPES = ["triangle", "square", "pentagon", "hexagon", "circle"]
LINE_KINDS = ["diagonal", "horizontal", "vertical", "curved", "circle", "other"]
CELL_NAMES = ["top-left", "top-middle", "top-right", "middle-left", "center",
              "middle-right", "bottom-left", "bottom-middle", "bottom-right"]
DESCRIBE_VERSION = "v1"

DESCRIBE_PROMPT = """This image is a single panel from an abstract visual reasoning puzzle (Procedurally Generated Matrices). Describe exactly what is drawn in it as JSON, using this schema:

{"shapes": [{"position": P, "type": T, "size": S, "color": C}], "lines": [{"kind": K, "color": C}]}

- position: where the shape sits on an imaginary 3x3 grid, one of: top-left, top-center, top-right, middle-left, center, middle-right, bottom-left, bottom-center, bottom-right. If there is only one shape in the middle, use center.
- type: one of triangle, square, pentagon, hexagon, circle.
- size: integer from 1 (smallest) to 10 (largest), relative to the panel.
- color: fill darkness as an integer from 0 (white) to 9 (black).
- lines: line drawings that are not closed shapes; kind is one of diagonal, horizontal, vertical, curved, circle, other; color is 0 (white) to 9 (black).
- List every shape and every line. Use [] when there are none.

Output only the JSON, with no explanation."""

TEXT_INTRO = """You are solving an abstract visual reasoning puzzle (Procedurally Generated Matrices) that is given to you as text.

The puzzle is a 3x3 grid of panels, read row by row. One panel is missing, marked "?". The panels are related by one or more rules that hold the same way in every row (for example, an attribute stays constant, increases step by step, or is combined across the first two panels to give the third). Each panel is described by its shapes (position on a 3x3 layout inside the panel, type, size from 1 to 10, color from 0 = white to 9 = black) and its lines (kind and color).
"""


# ----------------------------------------------------------------------------
# data
# ----------------------------------------------------------------------------

def _first(row, keys):
    for k in keys:
        if k in row and row[k] not in (None, "", []):
            return row[k]
    return None


def _messages(row):
    for k in ("messages", "conversations", "conversation"):
        if isinstance(row.get(k), list):
            return row[k]
    return None


def _content_text(c):
    if isinstance(c, str):
        return c
    if isinstance(c, list):
        return "".join(x.get("text", "") for x in c if isinstance(x, dict))
    return str(c)


def normalize_answer(a):
    if a is None:
        return None
    if isinstance(a, (int, np.integer)):
        return LETTERS[int(a)]
    m = re.search(r"\b([A-H])\b", str(a).strip()) or re.match(r"\s*([A-H])", str(a).strip())
    return m.group(1) if m else None


class Puzzle:
    __slots__ = ("id", "prompt", "answer", "image", "rules")

    def __init__(self, **kw):
        for k, v in kw.items():
            setattr(self, k, v)


def find_rules(row, key=None):
    """Relation structure from the source metadata, as a readable string (or None)."""
    if key:
        obj = row
        for part in key.split("."):
            obj = obj.get(part) if isinstance(obj, dict) else None
        return _fmt_rules(obj) if obj is not None else None
    stack = [row]
    while stack:
        obj = stack.pop()
        if isinstance(obj, dict):
            for k, v in obj.items():
                if "relation_structure" in k and "encoded" not in k:
                    return _fmt_rules(v)
                stack.append(v)
    return None


def _fmt_rules(v):
    def clean(x):
        s = str(x)
        s = re.sub(r"^b['\"](.*)['\"]$", r"\1", s)
        return s.strip()
    if isinstance(v, str):
        try:
            v = json.loads(v)
        except Exception:
            return re.sub(r"b'([^']*)'", r"\1", v)
    if isinstance(v, list):
        if v and isinstance(v[0], list):
            return "; ".join(" ".join(clean(t) for t in triple) for triple in v)
        return " ".join(clean(t) for t in v)
    return clean(v)


def load_puzzles(args, split=None, limit=None, shard=True):
    split = split or args.split
    limit = args.limit if limit is None else limit
    path = os.path.join(args.root, f"{split}.jsonl")
    out = []
    with open(path) as fh:
        for i, line in enumerate(fh):
            if not line.strip():
                continue
            if i < args.start:
                continue
            if limit and len(out) >= limit:
                break
            row = json.loads(line)
            msgs = _messages(row)
            pid = _first(row, [args.id_key] if args.id_key else ["id", "question_id", "qid", "uid", "sample_id"])
            prompt = _first(row, [args.prompt_key] if args.prompt_key else ["prompt", "question", "query", "instruction", "text"])
            answer = _first(row, [args.answer_key] if args.answer_key else ["answer", "correct_answer", "label", "target", "solution"])
            if msgs and prompt is None:
                um = next((m for m in msgs if m.get("role", m.get("from")) in ("user", "human")), None)
                if um:
                    prompt = _content_text(um.get("content", um.get("value")))
            if msgs and answer is None:
                am = [m for m in msgs if m.get("role", m.get("from")) in ("assistant", "gpt")]
                if am:
                    answer = _content_text(am[-1].get("content", am[-1].get("value")))
            img = row[args.image_key]
            img = img[0] if isinstance(img, list) else img
            if isinstance(img, dict):
                img = img.get("path") or img.get("image")
            ans = normalize_answer(answer)
            if ans is None:
                raise SystemExit(f"Could not read the answer from row {i}: keys={list(row)}; pass --answer_key.")
            if prompt is None:
                raise SystemExit(f"Could not read the prompt from row {i}: keys={list(row)}; pass --prompt_key.")
            prompt = re.sub(r"<image>\s*", "", str(prompt)).strip()
            out.append(Puzzle(id=str(pid if pid is not None else f"{split}-{i}"), prompt=prompt, answer=ans,
                              image=os.path.join(args.root, img), rules=find_rules(row, args.rules_key)))
    if shard and args.num_shards > 1:
        out = out[args.shard::args.num_shards]
    return out


def load_layout(path):
    with open(path) as fh:
        lay = json.load(fh)
    assert len(lay["matrix"]) == 9 and len(lay["choices"]) == 8, "layout.json needs 9 matrix and 8 choice boxes"
    return lay


def open_sheet(path):
    return Image.open(path).convert("RGB")   # exactly recovers the model-input pixels


def crop_panels(sheet, lay):
    """Returns (context: 8 crops in reading order, choices: 8 crops A-H)."""
    ctx = [sheet.crop(tuple(b)) for i, b in enumerate(lay["matrix"]) if i != lay["missing"]]
    ch = [sheet.crop(tuple(b)) for b in lay["choices"]]
    return ctx, ch


def upscale(img, size):
    w, h = img.size
    s = size / max(w, h)
    return img.resize((max(1, round(w * s)), max(1, round(h * s))), Image.BICUBIC)


def panel_hash(img):
    return hashlib.sha1(img.tobytes() + str(img.size).encode()).hexdigest()


def paste_answer(sheet, lay, answer):
    sh = sheet.copy()
    _, ch = crop_panels(sheet, lay)
    b = lay["matrix"][lay["missing"]]
    sh.paste(ch[LETTERS.index(answer)].resize((b[2] - b[0], b[3] - b[1]), Image.BICUBIC), (b[0], b[1]))
    return sh


# ----------------------------------------------------------------------------
# model
# ----------------------------------------------------------------------------

def model_tag(name):
    """Results folder name: the HF repo name, or parent_folder for a local checkpoint
    (so two checkpoints both called e.g. checkpoint-5000 do not collide)."""
    if os.path.isdir(name):
        p = os.path.abspath(name)
        tag = f"{os.path.basename(os.path.dirname(p))}_{os.path.basename(p)}"
    else:
        tag = name.rstrip("/").split("/")[-1]
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", tag)


class VLM:
    """Thin wrapper: batched letter scoring and greedy generation."""

    def __init__(self, name, dtype="bfloat16", attn="sdpa", causal=False, device=None):
        import torch
        from transformers import AutoProcessor, AutoTokenizer, AutoModelForImageTextToText, AutoModelForCausalLM
        self.torch = torch
        self.causal = causal
        td = getattr(torch, dtype)
        device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        kw = dict(dtype=td, attn_implementation=attn)
        if causal:
            self.processor = None
            self.tok = AutoTokenizer.from_pretrained(name)
            self.model = AutoModelForCausalLM.from_pretrained(name, **kw).to(device).eval()
        else:
            self.processor = AutoProcessor.from_pretrained(name)
            self.tok = self.processor.tokenizer
            self.model = AutoModelForImageTextToText.from_pretrained(name, **kw).to(device).eval()
        self.tok.padding_side = "left"
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token
        self.device = device
        self.letter_ids = self.option_ids([[L, " " + L] for L in LETTERS])

    def option_ids(self, groups):
        """For each answer option (a list of spellings), the single-token ids that spell it."""
        out = []
        for spellings in groups:
            toks = [self.tok.encode(s, add_special_tokens=False) for s in spellings]
            ids = {t[0] for t in toks if len(t) == 1}
            if not ids:   # no single-token spelling: fall back to the first token of each spelling
                ids = {t[0] for t in toks if t}
                print(f"  note: {spellings[0]!r} is not a single token; scoring its first token", flush=True)
            out.append(sorted(ids))
        for i in range(len(out)):
            for j in range(i + 1, len(out)):
                if set(out[i]) & set(out[j]):
                    raise RuntimeError(f"options {groups[i][0]!r} and {groups[j][0]!r} share a first token")
        return out

    def _inputs(self, convs, images, prefix=""):
        """convs: list of message lists; images: list (per sample) of PIL lists, in order of appearance."""
        if self.causal:
            texts = []
            for c in convs:
                flat = [{"role": m["role"], "content": _content_text(m["content"])} for m in c]
                try:
                    t = self.tok.apply_chat_template(flat, tokenize=False, add_generation_prompt=True,
                                                     enable_thinking=False)
                except TypeError:
                    t = self.tok.apply_chat_template(flat, tokenize=False, add_generation_prompt=True)
                texts.append(t + prefix)
            enc = self.tok(texts, return_tensors="pt", padding=True, add_special_tokens=False)
        else:
            texts = [self.processor.apply_chat_template(c, tokenize=False, add_generation_prompt=True) + prefix
                     for c in convs]
            flat = [im for ims in images for im in ims]
            enc = self.processor(text=texts, images=flat or None, return_tensors="pt", padding=True)
        return enc.to(self.device)

    def score_options(self, convs, images, prefix="", option_ids=None):
        """Log-probabilities of each option at the first answer token, renormalised over the options,
        and the total probability mass the model put on them."""
        torch = self.torch
        option_ids = option_ids or self.letter_ids
        enc = self._inputs(convs, images, prefix)
        with torch.inference_mode():
            logits = self.model(**enc, logits_to_keep=1).logits[:, -1].float()
        lp = torch.log_softmax(logits, -1)
        per = torch.stack([torch.logsumexp(lp[:, ids], -1) for ids in option_ids], -1)   # [B, n_options]
        mass = per.exp().sum(-1)
        norm = per - torch.logsumexp(per, -1, keepdim=True)
        return norm.cpu().numpy(), mass.cpu().numpy()

    def score_letters(self, convs, images, prefix=""):
        return self.score_options(convs, images, prefix, self.letter_ids)

    def layer_features(self, images, layers, grid=2):
        """InternVL only: features after each requested InternViT layer (0 = patch embeddings), one vector
        per image, average-pooled to grid x grid cells. Returns {layer: [n_images, C*grid*grid]}."""
        torch = self.torch
        m, ip = self.model, self.processor.image_processor
        if "qwen" in m.config.model_type or not hasattr(m.model, "vision_tower"):
            raise RuntimeError("--probe_layers is implemented for InternVL checkpoints only")
        try:
            enc = ip(images=images, return_tensors="pt", crop_to_patches=False)
        except TypeError:
            enc = ip(images=images, return_tensors="pt")
        pv = enc["pixel_values"].to(self.device, dtype=m.dtype)
        with torch.inference_mode():
            hs = m.model.vision_tower(pixel_values=pv, output_hidden_states=True, return_dict=True).hidden_states
        out = {}
        for L in layers:
            f = hs[min(L, len(hs) - 1) if L >= 0 else L].float()   # layer numbers past the end = last layer
            n = f.shape[1]
            s = math.isqrt(n)
            if s * s != n:                     # drop the CLS token
                f, s = f[:, 1:], math.isqrt(n - 1)
            g = f.reshape(len(f), s, s, -1).permute(0, 3, 1, 2)
            out[L] = torch.nn.functional.adaptive_avg_pool2d(g, grid).permute(0, 2, 3, 1).reshape(len(f), -1).half().cpu()
        return out

    def panel_features(self, images, grid=2):
        """Frozen vision features (the tokens the LLM receives), one vector per image:
        the token grid average-pooled to grid x grid cells and concatenated."""
        torch = self.torch
        m, ip = self.model, self.processor.image_processor
        cfg = m.config
        if "qwen" in cfg.model_type:
            enc = ip(images=images, return_tensors="pt")
            pv = enc["pixel_values"].to(self.device)
            thw = enc["image_grid_thw"].to(self.device)
            with torch.inference_mode():
                out = m.get_image_features(pv, thw)
            feats = out.pooler_output if hasattr(out, "pooler_output") else out[0]
            merge = getattr(cfg.vision_config, "spatial_merge_size", 2)
            if torch.is_tensor(feats):
                feats = torch.split(feats, (thw.prod(-1) // merge ** 2).tolist())
            shapes = [(int(h) // merge, int(w) // merge) for _, h, w in thw.tolist()]
        else:   # InternVL-style: one 448x448 tile per image, no tiling
            try:
                enc = ip(images=images, return_tensors="pt", crop_to_patches=False)
            except TypeError:
                enc = ip(images=images, return_tensors="pt")
            pv = enc["pixel_values"].to(self.device, dtype=m.dtype)
            if pv.shape[0] != len(images):
                raise RuntimeError(f"expected one tile per image, got {pv.shape[0]} for {len(images)} images")
            with torch.inference_mode():
                out = m.get_image_features(pixel_values=pv, vision_feature_layer=cfg.vision_feature_layer,
                                           vision_feature_select_strategy=cfg.vision_feature_select_strategy)
            feats = out.pooler_output if hasattr(out, "pooler_output") else out
            feats = list(feats)
            side = int(round(feats[0].shape[0] ** 0.5))
            shapes = [(side, side)] * len(feats)
        vecs = []
        for f, (h, w) in zip(feats, shapes):
            f = f.float()
            if h * w == f.shape[0] and grid > 1:
                g = f.reshape(h, w, -1).permute(2, 0, 1)[None]
                v = torch.nn.functional.adaptive_avg_pool2d(g, grid)[0].permute(1, 2, 0).reshape(-1)
            else:
                v = f.mean(0)
            vecs.append(v.half().cpu())
        return torch.stack(vecs)

    def generate(self, convs, images, max_new_tokens=384):
        enc = self._inputs(convs, images)
        with self.torch.inference_mode():
            out = self.model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False,
                                      temperature=None, top_p=None, top_k=None)
        new = out[:, enc["input_ids"].shape[1]:]
        return self.tok.batch_decode(new, skip_special_tokens=True)


def user(content):
    return [{"role": "user", "content": content}]


def load_model(args):
    print(f"loading {args.model} ...", flush=True)
    return VLM(args.model, dtype=args.dtype, attn=args.attn, causal=getattr(args, "causal", False))


# ----------------------------------------------------------------------------
# io helpers
# ----------------------------------------------------------------------------

def out_dir(args, tag=None):
    d = os.path.join(args.out, tag or model_tag(args.model))
    os.makedirs(d, exist_ok=True)
    return d


def shard_suffix(args):
    return f".shard{args.shard}of{args.num_shards}" if args.num_shards > 1 else ""


def done_ids(path):
    s = set()
    if os.path.exists(path):
        with open(path) as fh:
            for line in fh:
                try:
                    s.add(json.loads(line)["id"])
                except Exception:
                    pass   # a line cut off by an interrupted run
    return s


def read_merged(d, stem):
    rows = {}
    for p in sorted(glob.glob(os.path.join(d, stem + ".jsonl")) + glob.glob(os.path.join(d, stem + ".shard*.jsonl"))):
        with open(p) as fh:
            for line in fh:
                try:
                    r = json.loads(line)
                except Exception:
                    continue
                rows[r["id"]] = r
    return rows


def wilson(k, n, z=1.96):
    if n == 0:
        return (float("nan"),) * 3
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return p, c - h, c + h


def batched(xs, n):
    for i in range(0, len(xs), n):
        yield xs[i:i + n]


def run_scoring(args, stem, build):
    """Generic loop for the multiple-choice stages. build(puzzle) -> (conv, images, prefix)."""
    puzzles = load_puzzles(args)
    d = out_dir(args)
    path = os.path.join(d, f"{stem}{shard_suffix(args)}.jsonl")
    skip = done_ids(path)
    todo = [p for p in puzzles if p.id not in skip]
    print(f"{stem}: {len(todo)} to do, {len(skip)} already in {path}", flush=True)
    if not todo:
        return path
    vlm = load_model(args)
    t0, n = time.time(), 0
    with open(path, "a") as fh:
        for chunk in batched(todo, args.batch_size):
            built = [build(p) for p in chunk]
            lp, mass = vlm.score_letters([b[0] for b in built], [b[1] for b in built], built[0][2])
            for p, l, m in zip(chunk, lp, mass):
                pred = LETTERS[int(np.argmax(l))]
                fh.write(json.dumps({"id": p.id, "answer": p.answer, "pred": pred, "correct": pred == p.answer,
                                     "letter_logprobs": [round(float(x), 4) for x in l],
                                     "letter_mass": round(float(m), 4)}) + "\n")
            fh.flush()
            n += len(chunk)
            if n % (args.batch_size * 20) < args.batch_size:
                print(f"  {n}/{len(todo)}  {n / (time.time() - t0):.1f} puzzles/s", flush=True)
    summarize_scoring(path)
    return path


def summarize_scoring(path):
    rows = [json.loads(l) for l in open(path) if l.strip()]
    k = sum(r["correct"] for r in rows)
    p, lo, hi = wilson(k, len(rows))
    mass = np.mean([r["letter_mass"] for r in rows])
    preds = Counter(r["pred"] for r in rows)
    print(f"{os.path.basename(path)}: acc {p:.2%} [{lo:.2%}, {hi:.2%}] on {len(rows)}; "
          f"mean letter mass {mass:.2f}; predictions {dict(sorted(preds.items()))}")
    if mass < 0.5:
        print("  WARNING: under half of the next-token probability is on A-H; the model probably wants to "
              "write something else first. Check --answer_prefix (e.g. 'Answer: ') or the chat template.")


# ----------------------------------------------------------------------------
# stages: image, match
# ----------------------------------------------------------------------------

def dump(args, p, name, img):
    if args.dump:
        d = os.path.join(out_dir(args), "debug")
        os.makedirs(d, exist_ok=True)
        safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", p.id)   # IDs like pgm/neutral/validation/11723
        img.save(os.path.join(d, f"{safe}_{name}.png"))


CROPS_PUZZLE_INTRO = ("This is an abstract visual reasoning puzzle. A 3x3 grid of panels is shown one panel at a time, "
                      "row by row; the {cell} panel is missing. The panels follow rules that hold the same way in "
                      "every row. After the grid come eight answer choices, A to H.\n")
CROPS_PUZZLE_Q = "\nWhich choice completes the grid? Answer with a single letter from A to H."


def stage_image(args):
    seen = []
    lay = load_layout(args.layout) if args.variant == "crops" else None

    def build(p):
        sheet = open_sheet(p.image)
        if args.variant == "sheet":
            if len(seen) < args.dump:
                seen.append(p.id)
                dump(args, p, "sheet", sheet)
            return user([{"type": "image"}, {"type": "text", "text": p.prompt}]), [sheet], args.answer_prefix
        # the same puzzle as 16 separate upscaled panels, so perception is not limited by the sheet
        ctx, ch = crop_panels(sheet, lay)
        ctx = [upscale(c, args.panel_size) for c in ctx]
        ch = [upscale(c, args.panel_size) for c in ch]
        content, imgs, k = [{"type": "text", "text": CROPS_PUZZLE_INTRO.format(cell=CELL_NAMES[lay["missing"]])}], [], 0
        for cell in range(9):
            label = f"\nRow {cell // 3 + 1}, panel {cell % 3 + 1}:"
            if cell == lay["missing"]:
                content.append({"type": "text", "text": label + " ? (missing)"})
            else:
                content += [{"type": "text", "text": label}, {"type": "image"}]
                imgs.append(ctx[k])
                k += 1
        for L, im in zip(LETTERS, ch):
            content += [{"type": "text", "text": f"\nChoice {L}:"}, {"type": "image"}]
            imgs.append(im)
        content.append({"type": "text", "text": CROPS_PUZZLE_Q})
        if len(seen) < args.dump:
            seen.append(p.id)
            for i, im in enumerate(imgs):
                dump(args, p, f"puzzle_crop{i:02d}", im)
        return user(content), imgs, args.answer_prefix
    stem = f"image_{args.split}" if args.variant == "sheet" else f"image_crops_{args.split}"
    run_scoring(args, stem, build)


MATCH_SHEET_Q = ("This image shows a 3x3 grid of panels and, below it, eight answer choices labelled A to H. "
                 "The {cell} cell of the grid contains a panel that is an exact copy of one of the eight choices. "
                 "Which choice is identical to the {cell} panel? Answer with a single letter from A to H.")
MATCH_CROPS_Q = ("The first image is a reference panel. The next eight images are choices A to H. Exactly one "
                 "choice is identical to the reference panel. Which one? Answer with a single letter from A to H.")


def stage_match(args):
    lay = load_layout(args.layout)
    cell = CELL_NAMES[lay["missing"]]
    seen = []

    def build(p):
        sheet = open_sheet(p.image)
        if args.variant == "sheet":
            img = paste_answer(sheet, lay, p.answer)
            if len(seen) < args.dump:
                seen.append(p.id)
                dump(args, p, "match_sheet", img)
            return (user([{"type": "image"}, {"type": "text", "text": MATCH_SHEET_Q.format(cell=cell)}]),
                    [img], args.answer_prefix)
        _, ch = crop_panels(sheet, lay)
        ch = [upscale(c, args.panel_size) for c in ch]
        ref = ch[LETTERS.index(p.answer)]
        content = [{"type": "text", "text": "Reference panel:"}, {"type": "image"}]
        for L in LETTERS:
            content += [{"type": "text", "text": f"\nChoice {L}:"}, {"type": "image"}]
        content.append({"type": "text", "text": "\n" + MATCH_CROPS_Q})
        if len(seen) < args.dump:
            seen.append(p.id)
            for name, im in [("ref", ref)] + list(zip(LETTERS, ch)):
                dump(args, p, f"match_crop_{name}", im)
        return user(content), [ref] + ch, args.answer_prefix
    run_scoring(args, f"match_{args.variant}_{args.split}", build)


# ----------------------------------------------------------------------------
# same/different perception test and frozen-encoder probe
# ----------------------------------------------------------------------------
#
# Trials, built per puzzle from its 8 answer choices:
#   same : a choice paired with itself              (8 per puzzle)
#   diff : the correct answer paired with a distractor (up to 7 per puzzle)
# These are exactly the discriminations the puzzle needs. With --jitter J > 0, every image is
# shifted by up to J pixels independently, so "same" means same content rather than identical
# pixels (otherwise a frozen-feature probe could detect any pixel change trivially).
# Each diff trial records how far apart the two panels are in pixels (pix_frac: fraction of
# pixels whose gray level differs by more than 16), which gives a psychometric curve.

SAMEDIFF_Q = "Are the contents of these two images exactly the same? Answer Yes or No."
SAMEDIFF_JITTER_NOTE = " The whole picture may be shifted by a few pixels; ignore that shift."


def jitter(img, j, rng):
    if j <= 0:
        return img
    dx, dy = (int(v) for v in rng.integers(-j, j + 1, 2))
    a = np.asarray(img)
    edge = np.concatenate([a[0], a[-1], a[:, 0], a[:, -1]])
    vals, counts = np.unique(edge.reshape(-1, a.shape[-1]), axis=0, return_counts=True)
    out = Image.new(img.mode, img.size, tuple(int(x) for x in vals[counts.argmax()]))
    out.paste(img, (dx, dy))
    return out


def make_trials(p, lay, args):
    """Deterministic for a given puzzle, --seed and --jitter, so samediff and probe use identical trials."""
    rng = np.random.default_rng(int(hashlib.sha1(f"{args.seed}:{p.id}".encode()).hexdigest()[:12], 16))
    _, ch = crop_panels(open_sheet(p.image), lay)
    gray = [np.asarray(c.convert("L"), np.float32) for c in ch]
    a = LETTERS.index(p.answer)
    trials = []
    for k in range(8):
        trials.append({"id": f"{p.id}#s{LETTERS[k]}", "puzzle": p.id, "kind": "same", "pair": [LETTERS[k]] * 2,
                       "pix_frac": 0.0, "pix_mad": 0.0,
                       "imgs": (jitter(ch[k], args.jitter, rng), jitter(ch[k], args.jitter, rng))})
    for k in range(8):
        if k == a:
            continue
        d = np.abs(gray[a] - gray[k])
        frac = float((d > 16).mean())
        if frac == 0:
            continue   # distractor identical to the answer in pixels: not a valid trial
        x, y = (a, k) if rng.random() < 0.5 else (k, a)
        trials.append({"id": f"{p.id}#d{LETTERS[k]}", "puzzle": p.id, "kind": "diff", "pair": [LETTERS[x], LETTERS[y]],
                       "pix_frac": round(frac, 5), "pix_mad": round(float(d.mean() / 255), 5),
                       "imgs": (jitter(ch[x], args.jitter, rng), jitter(ch[y], args.jitter, rng))})
    if args.pairs:
        same = [t for t in trials if t["kind"] == "same"]
        diff = [t for t in trials if t["kind"] == "diff"]
        pick = lambda xs: [xs[i] for i in sorted(rng.choice(len(xs), min(args.pairs, len(xs)), replace=False))]
        trials = pick(same) + pick(diff)
    return trials


def auc(scores, labels):
    """Area under the ROC curve (tie-aware Mann-Whitney). labels: True = different."""
    s, y = np.asarray(scores, float), np.asarray(labels, bool)
    npos, nneg = int(y.sum()), int((~y).sum())
    if npos == 0 or nneg == 0:
        return float("nan")
    order = np.argsort(s, kind="mergesort")
    _, inv, counts = np.unique(s[order], return_inverse=True, return_counts=True)
    avg = np.cumsum(counts) - (counts - 1) / 2
    ranks = np.empty(len(s))
    ranks[order] = avg[inv]
    return float((ranks[y].sum() - npos * (npos + 1) / 2) / (npos * nneg))


def samediff_metrics(rows, n_boot=200, n_bins=5):
    """rows: dicts with puzzle, kind, pix_frac and score (higher = 'different')."""
    rows = [r for r in rows if np.isfinite(r["score"])]
    if not rows:
        return {}
    lab = np.array([r["kind"] == "diff" for r in rows])
    sc = np.array([r["score"] for r in rows])
    said_diff = sc > 0
    res = {"n_same": int((~lab).sum()), "n_diff": int(lab.sum()), "auc": auc(sc, lab),
           "balanced_acc": float((said_diff[lab].mean() + (~said_diff[~lab]).mean()) / 2),
           "says_diff_on_diff": float(said_diff[lab].mean()), "says_diff_on_same": float(said_diff[~lab].mean())}
    # 95% interval by resampling puzzles (trials from one puzzle are correlated)
    by = {}
    for i, r in enumerate(rows):
        by.setdefault(r["puzzle"], []).append(i)
    groups = list(by.values())
    rng = np.random.default_rng(0)
    boots = []
    for _ in range(n_boot):
        idx = np.concatenate([groups[g] for g in rng.integers(0, len(groups), len(groups))])
        boots.append(auc(sc[idx], lab[idx]))
    res["auc_ci"] = [float(np.nanpercentile(boots, 2.5)), float(np.nanpercentile(boots, 97.5))]
    # psychometric curve: diff trials binned by pixel difference, each bin against all same trials
    pf = np.array([r["pix_frac"] for r in rows])
    edges = np.quantile(pf[lab], np.linspace(0, 1, n_bins + 1))
    bins = []
    for b in range(n_bins):
        lo, hi = edges[b], edges[b + 1]
        inb = lab & (pf >= lo) & ((pf <= hi) if b == n_bins - 1 else (pf < hi))
        if inb.sum() == 0:
            continue
        sel = inb | ~lab
        bins.append({"pix_frac": [float(lo), float(hi)], "n": int(inb.sum()),
                     "auc": auc(sc[sel], lab[sel]), "says_diff": float(said_diff[inb].mean())})
    res["bins"] = bins
    return res


def stage_samediff(args):
    lay = load_layout(args.layout)
    puzzles = load_puzzles(args)
    d = out_dir(args)
    path = os.path.join(d, f"samediff_{args.split}{shard_suffix(args)}.jsonl")
    skip = done_ids(path)
    done_p = {i.rsplit("#", 1)[0] for i in skip}
    todo_p = [p for p in puzzles if p.id not in done_p]
    print(f"samediff: {len(todo_p)} puzzles to do (jitter {args.jitter}px)", flush=True)
    if todo_p:
        vlm = load_model(args)
        yes_no = vlm.option_ids([["Yes", "yes", " Yes", " yes"], ["No", "no", " No", " no"]])
        q = SAMEDIFF_Q + (SAMEDIFF_JITTER_NOTE if args.jitter > 0 else "")
        dumped = 0
        with open(path, "a") as fh:
            for pchunk in batched(todo_p, max(1, args.batch_size // 8)):
                trials = [t for p in pchunk for t in make_trials(p, lay, args)]
                for tchunk in batched(trials, args.batch_size):
                    imgs = [[upscale(t["imgs"][0], args.panel_size), upscale(t["imgs"][1], args.panel_size)]
                            for t in tchunk]
                    convs = [user([{"type": "text", "text": "Image 1:"}, {"type": "image"},
                                   {"type": "text", "text": "\nImage 2:"}, {"type": "image"},
                                   {"type": "text", "text": "\n" + q}]) for _ in tchunk]
                    lp, mass = vlm.score_options(convs, imgs, args.answer_prefix, yes_no)
                    for t, l, m, im in zip(tchunk, lp, mass, imgs):
                        score = float(l[1] - l[0])           # log P(No) - log P(Yes): higher = "different"
                        rec = {k: v for k, v in t.items() if k != "imgs"}
                        rec.update({"score": round(score, 4), "pred": "diff" if score > 0 else "same",
                                    "correct": (score > 0) == (t["kind"] == "diff"), "option_mass": round(float(m), 4)})
                        fh.write(json.dumps(rec) + "\n")
                        if dumped < args.dump * 15:
                            dumped += 1
                            safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", t["id"])
                            dd = os.path.join(d, "debug")
                            os.makedirs(dd, exist_ok=True)
                            for j, x in enumerate(im):
                                x.save(os.path.join(dd, f"samediff_{safe}_{j + 1}.png"))
                fh.flush()
    rows = list(read_merged(d, f"samediff_{args.split}").values())
    print_samediff(f"samediff/{args.split} (VLM answers)", samediff_metrics(rows))
    mass = np.mean([r["option_mass"] for r in rows])
    if mass < 0.5:
        print(f"  WARNING: only {mass:.2f} of the next-token probability is on Yes/No; try --answer_prefix.")


def print_samediff(name, m):
    if not m:
        return
    print(f"{name}: AUC {m['auc']:.3f} [{m['auc_ci'][0]:.3f}, {m['auc_ci'][1]:.3f}], balanced acc "
          f"{m['balanced_acc']:.1%}, says 'different' on {m['says_diff_on_diff']:.1%} of different and "
          f"{m['says_diff_on_same']:.1%} of same pairs (n={m['n_diff']}+{m['n_same']})")
    for b in m["bins"]:
        print(f"    pixels differing {b['pix_frac'][0]:.4f}-{b['pix_frac'][1]:.4f}: AUC {b['auc']:.3f}, "
              f"says 'different' {b['says_diff']:.1%} (n={b['n']})")


def _trial_features(vlm, puzzles, lay, args, kinds):
    """Per trial: {kind: (features of image 1, features of image 2)} and metadata.
    kinds: 'encoder' (tokens the LLM receives), 'pixels' (raw 32x32), 'layer<L>' (InternViT layer L)."""
    import torch
    layers = [int(k[5:]) for k in kinds if k.startswith("layer")]
    F = {k: ([], []) for k in kinds}
    meta = []
    for pchunk in batched(puzzles, 16):
        trials = [t for p in pchunk for t in make_trials(p, lay, args)]
        flat = [im for t in trials for im in t["imgs"]]
        got = {}
        if "encoder" in kinds:
            got["encoder"] = torch.cat([vlm.panel_features([upscale(x, args.panel_size) for x in b], args.probe_grid)
                                        for b in batched(flat, args.batch_size * 2)])
        if "pixels" in kinds:
            got["pixels"] = torch.tensor(np.stack([np.asarray(x.convert("L").resize((32, 32), Image.BILINEAR),
                                                              np.float32).reshape(-1) / 255 for x in flat])).half()
        if layers:
            parts = [vlm.layer_features([upscale(x, args.panel_size) for x in b], layers, args.probe_grid)
                     for b in batched(flat, args.batch_size * 2)]
            for L in layers:
                got[f"layer{L}"] = torch.cat([pt[L] for pt in parts])
        for k in kinds:
            F[k][0].append(got[k][0::2])
            F[k][1].append(got[k][1::2])
        meta += [{k: v for k, v in t.items() if k != "imgs"} for t in trials]
    return {k: (torch.cat(a), torch.cat(b)) for k, (a, b) in F.items()}, meta


def _fit_probe(tr, te, device):
    """Logistic regression on [|f1-f2|, f1*f2] (symmetric in the pair). Weight decay picked on 10% of the
    training puzzles. Returns test scores (logits, higher = different)."""
    import torch
    (A1, A2, ma), (B1, B2, mb) = tr, te
    pair = lambda x1, x2: torch.cat([(x1 - x2).abs(), x1 * x2], -1)
    ya = torch.tensor([m["kind"] == "diff" for m in ma], dtype=torch.float32)
    mu, sd = None, None
    for idx in torch.arange(len(ya)).split(4096):
        f = pair(A1[idx].float(), A2[idx].float())
        mu = f.sum(0) if mu is None else mu + f.sum(0)
        sd = (f ** 2).sum(0) if sd is None else sd + (f ** 2).sum(0)
    mu = mu / len(ya)
    sd = (sd / len(ya) - mu ** 2).clamp_min(1e-8).sqrt()
    feat = lambda x1, x2, idx: ((pair(x1[idx].float(), x2[idx].float()) - mu) / sd).to(device)
    puzzles = sorted({m["puzzle"] for m in ma})
    rng = np.random.default_rng(0)
    hold_p = set(rng.choice(puzzles, max(1, len(puzzles) // 10), replace=False).tolist())
    hold = torch.tensor([m["puzzle"] in hold_p for m in ma])
    tr_idx, ho_idx = torch.where(~hold)[0], torch.where(hold)[0]

    def train(wd, idx_all):
        torch.manual_seed(0)
        lin = torch.nn.Linear(mu.numel(), 1).to(device)
        opt = torch.optim.AdamW(lin.parameters(), lr=1e-3, weight_decay=wd)
        for _ in range(30):
            for idx in idx_all[torch.randperm(len(idx_all))].split(512):
                loss = torch.nn.functional.binary_cross_entropy_with_logits(
                    lin(feat(A1, A2, idx)).squeeze(-1), ya[idx].to(device))
                opt.zero_grad()
                loss.backward()
                opt.step()
        return lin

    def scores(lin, X1, X2):
        with torch.no_grad():
            return torch.cat([lin(feat(X1, X2, idx)).squeeze(-1).cpu()
                              for idx in torch.arange(len(X1)).split(4096)]).numpy()

    best = None
    for wd in (1e-4, 1e-3, 1e-2, 1e-1):
        lin = train(wd, tr_idx)
        a = auc(scores(lin, A1[ho_idx], A2[ho_idx]), ya[ho_idx].numpy().astype(bool))
        print(f"    weight decay {wd:g}: held-out AUC {a:.3f}", flush=True)
        if best is None or a > best[0]:
            best = (a, wd, lin)
    return scores(best[2], B1, B2), best[1]


def stage_probe(args):
    import torch
    lay = load_layout(args.layout)
    tr_p = load_puzzles(args, split=args.train_split, limit=args.probe_train, shard=False)
    te_p = load_puzzles(args, shard=False)
    te_ids = {p.id for p in te_p}
    tr_p = [p for p in tr_p if p.id not in te_ids]
    print(f"probe: {len(tr_p)} training puzzles ({args.train_split}), {len(te_p)} test puzzles ({args.split}), "
          f"jitter {args.jitter}px, encoder features pooled to {args.probe_grid}x{args.probe_grid}", flush=True)
    vlm = load_model(args)
    dev = vlm.device
    path = os.path.join(out_dir(args), f"probe_{args.split}.json")
    res = json.load(open(path)) if os.path.exists(path) else {}
    res.update({"n_train_puzzles": len(tr_p), "n_test_puzzles": len(te_p), "jitter": args.jitter,
                "grid": args.probe_grid})
    layers = [int(x) for x in args.probe_layers.split(",")] if args.probe_layers else []
    passes = [["encoder", "pixels"]] if not (layers and args.layers_only) else []
    if layers:
        passes.append([f"layer{L}" for L in layers])
    for kinds in passes:
        print(f"  features: {', '.join(kinds)} ...", flush=True)
        tr, mtr = _trial_features(vlm, tr_p, lay, args, kinds)
        te, mte = _trial_features(vlm, te_p, lay, args, kinds)
        for k in kinds:
            sc, wd = _fit_probe((*tr[k], mtr), (*te[k], mte), dev)
            rows = [dict(m, score=float(s)) for m, s in zip(mte, sc)]
            res[k] = samediff_metrics(rows)
            res[k]["weight_decay"] = wd
            res[k]["dim"] = int(tr[k][0].shape[1])
            print_samediff(f"probe/{k}", res[k])
        del tr, te
    with open(path, "w") as fh:
        json.dump(res, fh, indent=1)


# ----------------------------------------------------------------------------
# describe
# ----------------------------------------------------------------------------

def _num(x, lo, hi):
    try:
        v = float(re.search(r"-?\d+(\.\d+)?", str(x)).group())
    except Exception:
        return None
    return int(round(min(hi, max(lo, v))))


def _pos(s):
    s = re.sub(r"[\s_]+", "-", str(s).strip().lower()).replace("centre", "center")
    s = s.replace("upper", "top").replace("lower", "bottom")
    if s in POS9:
        return s
    parts = set(s.split("-"))
    if not parts & {"top", "bottom", "left", "right", "center", "middle"}:
        return None
    v = "top" if "top" in parts else "bottom" if "bottom" in parts else "middle"
    h = "left" if "left" in parts else "right" if "right" in parts else "center"
    if v == "middle":
        return "center" if h == "center" else f"middle-{h}"
    return f"{v}-{h}"


def _shape(s):
    s = str(s).lower()
    for t in SHAPES:
        if t in s or (len(s) >= 3 and t.startswith(s[:3])):
            return t
    if "rect" in s or "diamond" in s:
        return "square"
    if "oval" in s or "ellip" in s or "ring" in s:
        return "circle"
    return None


def _line(s):
    s = str(s).lower()
    for k in LINE_KINDS[:-1]:
        if k in s:
            return k
    if "arc" in s or "curve" in s or "wav" in s:
        return "curved"
    return "other"


def parse_description(text):
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        return None
    raw = m.group()
    obj = None
    for attempt in (raw, re.sub(r",\s*([}\]])", r"\1", raw)):
        try:
            obj = json.loads(attempt)
            break
        except Exception:
            continue
    if not isinstance(obj, dict):
        return None
    shapes, lines = [], []
    for s in obj.get("shapes") or []:
        if not isinstance(s, dict):
            continue
        shapes.append({"position": _pos(s.get("position", "")), "type": _shape(s.get("type", "")),
                       "size": _num(s.get("size"), 1, 10), "color": _num(s.get("color"), 0, 9)})
    for l in obj.get("lines") or []:
        if not isinstance(l, dict):
            continue
        lines.append({"kind": _line(l.get("kind", l.get("type", ""))), "color": _num(l.get("color"), 0, 9)})
    shapes.sort(key=lambda s: (POS9.index(s["position"]) if s["position"] else 9, str(s["type"])))
    lines.sort(key=lambda l: (l["kind"], l["color"] if l["color"] is not None else -1))
    return {"shapes": shapes, "lines": lines}


def canonical(parsed, raw):
    if parsed is None:
        return "UNPARSEABLE: " + " ".join(raw.split())[:200]
    parts = []
    for s in parsed["shapes"]:
        parts.append(f"{s['type'] or '?'} at {s['position'] or '?'}, size {s['size'] if s['size'] is not None else '?'}, "
                     f"color {s['color'] if s['color'] is not None else '?'}")
    sh = "; ".join(parts) if parts else "none"
    ln = "; ".join(f"{l['kind']} line, color {l['color'] if l['color'] is not None else '?'}"
                   for l in parsed["lines"]) or "none"
    return f"shapes: {sh} | lines: {ln}"


def stage_describe(args):
    lay = load_layout(args.layout)
    puzzles = load_puzzles(args)
    d = out_dir(args)
    cache_path = os.path.join(d, f"panel_cache_{DESCRIBE_VERSION}_{args.panel_size}{shard_suffix(args)}.jsonl")
    cache = {}
    for p in glob.glob(os.path.join(d, f"panel_cache_{DESCRIBE_VERSION}_{args.panel_size}*.jsonl")):
        for line in open(p):
            try:
                r = json.loads(line)
                cache[r["hash"]] = r
            except Exception:
                pass
    path = os.path.join(d, f"describe_{args.split}{shard_suffix(args)}.jsonl")
    skip = done_ids(path)
    todo = [p for p in puzzles if p.id not in skip]
    print(f"describe/{args.split}: {len(todo)} puzzles to do, {len(skip)} done, {len(cache)} cached panels", flush=True)
    if not todo:
        summarize_describe(d, args.split)
        return
    vlm = load_model(args)
    t0, ngen = time.time(), 0
    with open(path, "a") as fh, open(cache_path, "a") as cfh:
        for ci, chunk in enumerate(batched(todo, args.chunk)):
            per_puzzle, need = [], {}
            for p in chunk:
                ctx, ch = crop_panels(open_sheet(p.image), lay)
                hs = []
                for k, im in enumerate(ctx + ch):
                    h = panel_hash(im)
                    hs.append(h)
                    if h not in cache and h not in need:
                        need[h] = upscale(im, args.panel_size)
                    if ci == 0 and args.dump and len(per_puzzle) < args.dump:
                        dump(args, p, f"panel{k:02d}", upscale(im, args.panel_size))
                per_puzzle.append((p, hs))
            items = list(need.items())
            for b in batched(items, args.batch_size):
                outs = vlm.generate([user([{"type": "image"}, {"type": "text", "text": DESCRIBE_PROMPT}]) for _ in b],
                                    [[im] for _, im in b], max_new_tokens=args.max_new_tokens)
                for (h, _), o in zip(b, outs):
                    parsed = parse_description(o)
                    rec = {"hash": h, "raw": o, "parsed": parsed, "text": canonical(parsed, o)}
                    cache[h] = rec
                    cfh.write(json.dumps(rec) + "\n")
                ngen += len(b)
            cfh.flush()
            for p, hs in per_puzzle:
                recs = [cache[h] for h in hs]
                fh.write(json.dumps({
                    "id": p.id, "answer": p.answer, "rules": p.rules, "missing": lay["missing"],
                    "context": [r["text"] for r in recs[:8]], "choices": [r["text"] for r in recs[8:]],
                    "context_parsed": [r["parsed"] for r in recs[:8]], "choices_parsed": [r["parsed"] for r in recs[8:]],
                    "hashes": hs}) + "\n")
            fh.flush()
            el = time.time() - t0
            print(f"  {min((ci + 1) * args.chunk, len(todo))}/{len(todo)} puzzles, {ngen} panels generated, "
                  f"{ngen / max(el, 1e-6):.1f} panels/s", flush=True)
    summarize_describe(d, args.split)


def summarize_describe(d, split):
    rows = list(read_merged(d, f"describe_{split}").values())
    if not rows:
        return {}
    n = len(rows)
    parsed = sum(x is not None for r in rows for x in r["context_parsed"] + r["choices_parsed"])
    all_distinct = sum(len(set(r["choices"])) == 8 for r in rows)
    ans_unique = sum(Counter(r["choices"])[r["choices"][LETTERS.index(r["answer"])]] == 1 for r in rows)
    pix_unique = sum(Counter(r["hashes"][8:])[r["hashes"][8 + LETTERS.index(r["answer"])]] == 1 for r in rows)
    mean_unique = np.mean([len(set(r["choices"])) for r in rows])
    s = {"n": n, "panel_parse_rate": parsed / (16 * n), "all_8_choices_distinct": all_distinct / n,
         "answer_description_unique": ans_unique / n, "answer_pixels_unique": pix_unique / n,
         "mean_distinct_choice_descriptions": float(mean_unique)}
    print(f"describe/{split}: " + ", ".join(f"{k}={v:.3f}" if isinstance(v, float) else f"{k}={v}" for k, v in s.items()))
    with open(os.path.join(d, f"describe_{split}_summary.json"), "w") as fh:
        json.dump(s, fh, indent=1)
    return s


# ----------------------------------------------------------------------------
# text reasoning
# ----------------------------------------------------------------------------

def text_prompt(r, with_rules):
    ctx = list(r["context"])
    grid = ctx[:r["missing"]] + ["?"] + ctx[r["missing"]:]
    lines = [TEXT_INTRO]
    for row in range(3):
        lines.append(f"Row {row + 1}:")
        for col in range(3):
            lines.append(f"  Panel {col + 1}: {grid[3 * row + col]}")
    lines.append("\nChoices:")
    for L, c in zip(LETTERS, r["choices"]):
        lines.append(f"  {L}: {c}")
    if with_rules and r.get("rules"):
        lines.append(f"\nHint - the rules of this puzzle (object, attribute, relation): {r['rules']}")
    lines.append("\nWhich choice completes the grid? Answer with a single letter from A to H.")
    return "\n".join(lines)


def stage_text(args):
    src = os.path.join(args.out, args.desc_from or model_tag(args.model))
    rows = list(read_merged(src, f"describe_{args.split}").values())
    if not rows:
        raise SystemExit(f"No describe_{args.split}*.jsonl in {src}. Run the describe stage first.")
    if args.rules and not any(r.get("rules") for r in rows):
        raise SystemExit("--rules: no relation structure was found in the source metadata; pass --rules_key "
                         "(dotted path into the JSONL row) and rerun describe, or drop --rules.")
    if args.num_shards > 1:
        rows = rows[args.shard::args.num_shards]
    d = out_dir(args)
    stem = f"text{'_rules' if args.rules else ''}_{args.split}"
    if args.desc_from:
        stem += f"_desc-{args.desc_from}"
    path = os.path.join(d, f"{stem}{shard_suffix(args)}.jsonl")
    skip = done_ids(path)
    todo = [r for r in rows if r["id"] not in skip]
    print(f"{stem}: {len(todo)} to do", flush=True)
    if args.dump and todo:
        print("---- example prompt ----\n" + text_prompt(todo[0], args.rules) + "\n------------------------")
    if todo:
        vlm = load_model(args)
        with open(path, "a") as fh:
            for chunk in batched(todo, args.batch_size):
                convs = [user([{"type": "text", "text": text_prompt(r, args.rules)}]) for r in chunk]
                lp, mass = vlm.score_letters(convs, [[] for _ in chunk], args.answer_prefix)
                for r, l, m in zip(chunk, lp, mass):
                    pred = LETTERS[int(np.argmax(l))]
                    uniq = Counter(r["choices"])[r["choices"][LETTERS.index(r["answer"])]] == 1
                    fh.write(json.dumps({"id": r["id"], "answer": r["answer"], "pred": pred,
                                         "correct": pred == r["answer"], "answer_desc_unique": uniq,
                                         "letter_logprobs": [round(float(x), 4) for x in l],
                                         "letter_mass": round(float(m), 4)}) + "\n")
                fh.flush()
    summarize_scoring(path)
    rs = [json.loads(l) for l in open(path) if l.strip()]
    sub = [r for r in rs if r["answer_desc_unique"]]
    if sub:
        p, lo, hi = wilson(sum(r["correct"] for r in sub), len(sub))
        print(f"  on puzzles whose answer description is unique among the choices: {p:.2%} [{lo:.2%}, {hi:.2%}] (n={len(sub)})")


# ----------------------------------------------------------------------------
# solver on descriptions (information sufficiency)
# ----------------------------------------------------------------------------

def featurize(parsed):
    """Fixed-length vector for one panel description."""
    f = np.zeros(9 * 8 + 1 + 5 + 2 + 1 + len(LINE_KINDS) + 1 + 1, np.float32)
    if parsed is None:
        f[-1] = 1.0
        return f
    sizes, colors = [], []
    for s in parsed["shapes"]:
        t = SHAPES.index(s["type"]) if s["type"] in SHAPES else None
        if s["position"] in POS9:
            o = POS9.index(s["position"]) * 8
            f[o] = 1.0
            if t is not None:
                f[o + 1 + t] = 1.0
            if s["size"] is not None:
                f[o + 6] = s["size"] / 10
            if s["color"] is not None:
                f[o + 7] = s["color"] / 9
        if t is not None:
            f[73 + t] += 1 / 9
        if s["size"] is not None:
            sizes.append(s["size"] / 10)
        if s["color"] is not None:
            colors.append(s["color"] / 9)
    o = 72
    f[o] = len(parsed["shapes"]) / 9
    f[78] = np.mean(sizes) if sizes else 0
    f[79] = np.mean(colors) if colors else 0
    f[80] = len(parsed["lines"]) / 6
    for l in parsed["lines"]:
        f[81 + LINE_KINDS.index(l["kind"])] += 1 / 6
    lc = [l["color"] / 9 for l in parsed["lines"] if l["color"] is not None]
    f[81 + len(LINE_KINDS)] = np.mean(lc) if lc else 0
    return f


def build_arrays(rows):
    X, y = [], []
    for r in rows:
        ctx = np.stack([featurize(p) for p in r["context_parsed"]])
        ch = np.stack([featurize(p) for p in r["choices_parsed"]])
        X.append(np.concatenate([ctx, ch]))
        y.append(LETTERS.index(r["answer"]))
    return np.stack(X), np.array(y), int(rows[0]["missing"])


def train_solver(Xtr, ytr, Xte, yte, missing, choices_only, seed, epochs, device):
    import torch
    import torch.nn as nn
    torch.manual_seed(seed)
    np.random.seed(seed)
    F = Xtr.shape[-1]

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.enc = nn.Sequential(nn.Linear(F, 128), nn.ReLU(), nn.Linear(128, 64), nn.ReLU())
            self.row = nn.Sequential(nn.Linear(3 * 64, 256), nn.ReLU(), nn.Linear(256, 128), nn.ReLU())
            self.head = nn.Sequential(nn.Linear(6 * 128, 256), nn.ReLU(), nn.Dropout(0.1), nn.Linear(256, 1))
            self.solo = nn.Sequential(nn.Linear(64, 128), nn.ReLU(), nn.Linear(128, 1))

        def forward(self, x):                     # x: [B, 16, F]
            e = self.enc(x)                       # [B, 16, 64]
            ctx, ch = e[:, :8], e[:, 8:]
            if choices_only:
                return self.solo(ch).squeeze(-1)
            B = x.shape[0]
            scores = []
            for k in range(8):
                grid = torch.cat([ctx[:, :missing], ch[:, k:k + 1], ctx[:, missing:]], 1).view(B, 3, 3, 64)
                rows = self.row(grid.reshape(B, 3, 192))
                cols = self.row(grid.transpose(1, 2).reshape(B, 3, 192))
                scores.append(self.head(torch.cat([rows, cols], 1).reshape(B, -1)))
            return torch.cat(scores, -1)

    net = Net().to(device)
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-4)
    xt, yt = torch.tensor(Xtr, device=device), torch.tensor(ytr, device=device)
    xv, yv = torch.tensor(Xte, device=device), torch.tensor(yte, device=device)
    # hold out 5% of training puzzles to pick the epoch
    perm = torch.randperm(len(xt), device=device)
    nh = max(1, len(xt) // 20)
    hold, tr = perm[:nh], perm[nh:]
    best, best_state = -1, None
    for ep in range(epochs):
        net.train()
        for idx in tr[torch.randperm(len(tr), device=device)].split(256):
            loss = nn.functional.cross_entropy(net(xt[idx]), yt[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
        net.eval()
        with torch.no_grad():
            acc = (net(xt[hold]).argmax(-1) == yt[hold]).float().mean().item()
        if acc > best:
            best, best_state = acc, {k: v.clone() for k, v in net.state_dict().items()}
    net.load_state_dict(best_state)
    with torch.no_grad():
        pred = net(xv).argmax(-1)
    return (pred == yv).float().mean().item(), best


def stage_solve(args):
    import torch
    d = os.path.join(args.out, args.desc_from or model_tag(args.model))
    tr = list(read_merged(d, f"describe_{args.train_split}").values())
    te = list(read_merged(d, f"describe_{args.split}").values())
    if not tr or not te:
        raise SystemExit(f"Need describe_{args.train_split} and describe_{args.split} in {d}.")
    te_ids = {r["id"] for r in te}
    tr = [r for r in tr if r["id"] not in te_ids]
    Xtr, ytr, miss = build_arrays(tr)
    Xte, yte, _ = build_arrays(te)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    res = {"n_train": len(tr), "n_test": len(te)}
    for name, co in (("relational", False), ("choices_only", True)):
        accs = []
        for seed in range(args.seeds):
            acc, hold = train_solver(Xtr, ytr, Xte, yte, miss, co, seed, args.epochs, dev)
            accs.append(acc)
            print(f"  {name} seed {seed}: test {acc:.2%} (held-out train {hold:.2%})", flush=True)
        res[name] = {"mean": float(np.mean(accs)), "std": float(np.std(accs)), "runs": accs}
        print(f"solver/{name}: {np.mean(accs):.2%} ± {np.std(accs):.2%} over {args.seeds} seeds "
              f"(train {len(tr)}, test {len(te)})")
    with open(os.path.join(d, f"solve_{args.split}.json"), "w") as fh:
        json.dump(res, fh, indent=1)


# ----------------------------------------------------------------------------
# report
# ----------------------------------------------------------------------------

def stage_report(args):
    tags = sorted(t for t in os.listdir(args.out) if os.path.isdir(os.path.join(args.out, t)))
    sp = args.split
    cols = [("image", f"image_{sp}"), ("image: crops", f"image_crops_{sp}"),
            ("match: sheet", f"match_sheet_{sp}"), ("match: crops", f"match_crops_{sp}"),
            ("text", f"text_{sp}"), ("text + rules", f"text_rules_{sp}")]
    lines = ["| model | " + " | ".join(c[0] for c in cols) +
             " | desc parse | answer desc unique | solver | choices-only |",
             "|---" * (len(cols) + 5) + "|"]
    for t in tags:
        d = os.path.join(args.out, t)
        cells = []
        for _, stem in cols:
            rows = list(read_merged(d, stem).values())
            if rows:
                p, lo, hi = wilson(sum(r["correct"] for r in rows), len(rows))
                cells.append(f"{p:.1%} [{lo:.1%}, {hi:.1%}] n={len(rows)}")
            else:
                cells.append("-")
        s = summarize_describe(d, sp) if read_merged(d, f"describe_{sp}") else {}
        cells.append(f"{s['panel_parse_rate']:.1%}" if s else "-")
        cells.append(f"{s['answer_description_unique']:.1%}" if s else "-")
        sol = os.path.join(d, f"solve_{sp}.json")
        if os.path.exists(sol):
            j = json.load(open(sol))
            cells += [f"{j['relational']['mean']:.1%} ± {j['relational']['std']:.1%}",
                      f"{j['choices_only']['mean']:.1%} ± {j['choices_only']['std']:.1%}"]
        else:
            cells += ["-", "-"]
        lines.append(f"| {t} | " + " | ".join(cells) + " |")
    extra = sorted({re.sub(r"\.shard\d+of\d+", "", os.path.basename(p))[:-6] + " (" + os.path.basename(os.path.dirname(p)) + ")"
                    for p in glob.glob(os.path.join(args.out, "*", f"text*_{sp}_desc-*.jsonl"))})
    for e in extra:
        stem, t = e.rsplit(" (", 1)
        rows = list(read_merged(os.path.join(args.out, t[:-1]), stem).values())
        p, lo, hi = wilson(sum(r["correct"] for r in rows), len(rows))
        lines.append(f"\nCross-model text run {stem} by {t[:-1]}: {p:.1%} [{lo:.1%}, {hi:.1%}] n={len(rows)}")
    lines += ["", "Chance is 12.5% for every accuracy column above.", ""] + perception_tables(args.out, tags, sp)
    table = "\n".join(lines)
    print()
    print(table)
    with open(os.path.join(args.out, f"report_{sp}.md"), "w") as fh:
        fh.write(table + "\n")


def perception_tables(out, tags, sp):
    """Same/different (VLM answers), encoder probe and pixel probe: AUC 0.5 = chance, 1.0 = perfect."""
    lines = ["Perception: same/different on the puzzle's own answer-vs-distractor pairs (AUC: 0.5 chance, 1.0 perfect)", "",
             "| model | VLM answers: AUC | VLM balanced acc | VLM says 'different' (diff / same pairs) "
             "| encoder probe AUC | pixel probe AUC |", "|---|---|---|---|---|---|"]
    curves = []
    fmt = lambda m: f"{m['auc']:.3f} [{m['auc_ci'][0]:.3f}, {m['auc_ci'][1]:.3f}]" if m else "-"
    for t in tags:
        d = os.path.join(out, t)
        vm = samediff_metrics(list(read_merged(d, f"samediff_{sp}").values()))
        pj = os.path.join(d, f"probe_{sp}.json")
        pr = json.load(open(pj)) if os.path.exists(pj) else {}
        if not vm and not pr:
            continue
        lines.append(f"| {t} | {fmt(vm)} | " + (f"{vm['balanced_acc']:.1%}" if vm else "-") + " | " +
                     (f"{vm['says_diff_on_diff']:.1%} / {vm['says_diff_on_same']:.1%}" if vm else "-") +
                     f" | {fmt(pr.get('encoder'))} | {fmt(pr.get('pixels'))} |")
        ref = vm or pr.get("encoder") or pr.get("pixels")
        if ref and ref.get("bins"):
            curves += ["", f"{t}: AUC by how different the two panels are (fraction of pixels that differ)", "",
                       "| pixels differing | n | VLM answers | encoder probe | pixel probe |", "|---|---|---|---|---|"]
            get = lambda m, i: f"{m['bins'][i]['auc']:.3f}" if m and i < len(m.get("bins", [])) else "-"
            for i, b in enumerate(ref["bins"]):
                curves.append(f"| {b['pix_frac'][0]:.4f} to {b['pix_frac'][1]:.4f} | {b['n']} | {get(vm, i)} | "
                              f"{get(pr.get('encoder'), i)} | {get(pr.get('pixels'), i)} |")
    layer_rows = []
    for t in tags:
        pj = os.path.join(out, t, f"probe_{sp}.json")
        pr = json.load(open(pj)) if os.path.exists(pj) else {}
        ks = sorted((k for k in pr if k.startswith("layer")), key=lambda k: int(k[5:]))
        if ks:
            if not layer_rows:
                layer_rows = ["", "Probe AUC by InternViT layer (overall / on the most similar fifth of pairs)", ""]
            layer_rows += [f"| {t} | " + " | ".join(k for k in ks) + " | tokens to LLM |",
                           "|---" * (len(ks) + 2) + "|",
                           "| AUC | " + " | ".join(f"{pr[k]['auc']:.3f} / {pr[k]['bins'][0]['auc']:.3f}" for k in ks) +
                           " | " + (f"{pr['encoder']['auc']:.3f} / {pr['encoder']['bins'][0]['auc']:.3f}"
                                    if pr.get("encoder") else "-") + " |"]
    out_lines = lines + curves if len(lines) > 4 else []
    return out_lines + layer_rows


# ----------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stage", choices=["image", "match", "samediff", "probe", "describe", "text", "solve", "report"])
    ap.add_argument("--data", "--root", dest="root", default=os.environ.get("DATA"),
                    help="extracted dataset root (default: $DATA)")
    ap.add_argument("--layout", default="layout.json")
    ap.add_argument("--model", default="Qwen/Qwen3-VL-2B-Instruct",
                    help="HF id or local checkpoint; for InternVL use OpenGVLab/InternVL3_5-1B-HF")
    ap.add_argument("--causal", action="store_true", help="text stage only: load a text-only LLM")
    ap.add_argument("--desc_from", help="text/solve: use descriptions made by this model tag")
    ap.add_argument("--split", default="validation")
    ap.add_argument("--train_split", default="train", help="solve: split whose descriptions train the solver")
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--limit", type=int, default=0, help="0 = all rows")
    ap.add_argument("--out", default="runs/pgm_gap")
    ap.add_argument("--variant", choices=["sheet", "crops"], default="sheet",
                    help="image/match: the contact sheet, or the panels as separate upscaled images")
    ap.add_argument("--jitter", type=int, default=2,
                    help="samediff/probe: shift every image by up to this many pixels (0 = exact copies)")
    ap.add_argument("--pairs", type=int, default=0,
                    help="samediff/probe: same and different pairs per puzzle (0 = all: 8 same, up to 7 different)")
    ap.add_argument("--probe_train", type=int, default=1000, help="probe: training-split puzzles for the probe")
    ap.add_argument("--probe_grid", type=int, default=2, help="probe: pool vision tokens to an N x N grid")
    ap.add_argument("--probe_layers", default="",
                    help="probe, InternVL: also probe these InternViT layers, e.g. 0,6,12,18,24 (0 = patch embeddings)")
    ap.add_argument("--layers_only", action="store_true", help="probe: skip the encoder/pixel probes (already done)")
    ap.add_argument("--seed", type=int, default=0, help="samediff/probe: seed for pair order and jitter")
    ap.add_argument("--rules", action="store_true", help="text: add relation structure as a hint")
    ap.add_argument("--rules_key", help="dotted path to the relation structure in a JSONL row (auto-detected)")
    ap.add_argument("--image_key", default="images")
    ap.add_argument("--id_key"), ap.add_argument("--prompt_key"), ap.add_argument("--answer_key")
    ap.add_argument("--panel_size", type=int, default=448, help="crops are upscaled so the long side is this")
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--chunk", type=int, default=64, help="describe: puzzles per cache/flush chunk")
    ap.add_argument("--max_new_tokens", type=int, default=384)
    ap.add_argument("--answer_prefix", default="", help="text appended after the assistant turn starts")
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--attn", default="sdpa", help="sdpa | flash_attention_2 | eager")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num_shards", type=int, default=1)
    ap.add_argument("--dump", type=int, default=0, help="save inputs of the first N puzzles to <out>/<tag>/debug")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--seeds", type=int, default=3)
    args = ap.parse_args()

    if args.stage in ("image", "match", "samediff", "probe", "describe") and not args.root:
        ap.error("give the dataset root with --data $DATA (or export DATA=...)")
    if args.stage == "text" and args.desc_from is None and args.causal:
        ap.error("--causal needs --desc_from <tag of the VLM that wrote the descriptions>")
    {"image": stage_image, "match": stage_match, "samediff": stage_samediff, "probe": stage_probe,
     "describe": stage_describe, "text": stage_text, "solve": stage_solve, "report": stage_report}[args.stage](args)


if __name__ == "__main__":
    main()
