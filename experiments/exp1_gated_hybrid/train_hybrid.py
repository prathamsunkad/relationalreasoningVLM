#!/usr/bin/env python
"""
Train and evaluate the gated relational hybrid on InternVL (and the matching LoRA baseline).

Every mode uses the same input: each puzzle is 16 separate panels (8 context panels in reading
order, then choices A-H) cut from the contact sheet with layout.json, one 448x448 tile per panel,
with the same prompt as `pgm_gap.py image --variant crops`. The objective is cross-entropy on the
answer letter (the first assistant token). So `baseline` and `stage2` differ only in the module.

Modes
  stage0    Train the relational branch alone: frozen InternViT -> slots -> abstractor -> 8-way
            candidate score. No LLM. Use --half A / --half B for cross-fitting (see below).
  stage1    Alignment: load the stage-0 branch(es), freeze everything except the fusion's up-projection,
            gate and norms; train on the answer loss through the frozen LLM.
  stage2    Joint: LoRA on the LLM + fusion head + (low LR) relational branch + auxiliary score loss.
  baseline  LoRA on the LLM, no fusion module. Same data, input, objective, schedule and selection.
  cache     Crop every puzzle's 16 panels once into <cache>/<split>_panels.npy (uint8). Training with
            --cache <dir> then skips PNG decoding/cropping/resizing on the CPU and resizes on the GPU.
  eval      Score a checkpoint on a split; writes per-puzzle predictions (pgm_gap.py format) and
            accuracy of the LLM, the branch's own score head, and their ensemble.
            Controls: --no_fusion (relational path off), --shuffle_relational (other puzzles' features).

Cross-fitting: a branch trained on puzzle X is over-confident on X. `stage0 --half A` and
`stage0 --half B` train two branches on disjoint halves of the pool; give both to stage1 with
`--branches runA runB`, and every training puzzle is routed to the branch that did not see it.
At evaluation the two branches are averaged.

Architecture search (successive halving): --schedule constant --max_steps N keeps the learning rate
flat after --warmup_steps, so a run can be extended later by re-running it with a larger --max_steps
(it resumes from last.pt). --val_from_train N holds out the LAST N rows of train.jsonl as the validation
set (excluded from training) for a less noisy comparison; your 1,000 validation puzzles stay untouched.

Long runs in 6-hour jobs: --max_hours H saves last.pt and exits cleanly after H hours (no summary.json yet), so
the same command in the next job continues. --patience N stops after N validations without a new best; with
--decay_steps K (constant schedule) it first decays the learning rate linearly to 0 over K more steps, then stops.
Without a plateau the decay is the last K steps before --max_steps.

The recipe (LoRA first, then the branch): `baseline` trains LoRA on the crops; `stage1 --lora_from <baseline>`
aligns the branch with that LoRA frozen in place; `stage2` then continues LoRA + branch with --branch_dropout p
(each puzzle has its relational path switched off with probability p, so the LLM keeps solving on its own); the
equal-step control is `baseline --init <baseline>` for the same epochs. pipeline.sh runs all of it.

Data-scaling: --train_limit N uses the first N training rows. The file order is your fixed nested
ranking, so smaller limits are subsets of larger ones.

Example (paths are placeholders):
  D=/path/to/pgm-neutral-train20pct-seed20261007-v1; M=OpenGVLab/InternVL3_5-1B-HF
  python train_hybrid.py stage0 --data $D --model $M --half A --out ckpt/s0A
  python train_hybrid.py stage0 --data $D --model $M --half B --out ckpt/s0B
  python train_hybrid.py stage1 --data $D --model $M --branches ckpt/s0A ckpt/s0B --out ckpt/s1
  python train_hybrid.py stage2 --data $D --model $M --init ckpt/s1 --out ckpt/s2
  python train_hybrid.py baseline --data $D --model $M --out ckpt/base
  python train_hybrid.py eval --data $D --model $M --init ckpt/s2 --split test
"""
import argparse
import hashlib
import json
import math
import os
import random
import sys
import time
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.environ.get("PROBES") or os.path.join(REPO, "probes"))   # pgm_gap.py lives in probes/
sys.path.insert(0, HERE)
import pgm_gap as G                      # noqa: E402  data loading, layout, prompt pieces
import relational_fusion as RF           # noqa: E402

LETTERS = G.LETTERS
# The puzzle-from-crops prompt. Kept here as well so training does not depend on the pgm_gap.py version;
# it must stay identical to pgm_gap.py's `image --variant crops` prompt.
CROPS_PUZZLE_INTRO = getattr(G, "CROPS_PUZZLE_INTRO", (
    "This is an abstract visual reasoning puzzle. A 3x3 grid of panels is shown one panel at a time, "
    "row by row; the {cell} panel is missing. The panels follow rules that hold the same way in "
    "every row. After the grid come eight answer choices, A to H.\n"))
CROPS_PUZZLE_Q = getattr(G, "CROPS_PUZZLE_Q", "\nWhich choice completes the grid? Answer with a single letter from A to H.")


# ----------------------------------------------------------------------------- data

def half_of(pid):
    return "A" if int(hashlib.sha1(pid.encode()).hexdigest(), 16) % 2 == 0 else "B"


def load_split(args, split, limit=0):
    ns = SimpleNamespace(root=args.data, split=split, start=0, limit=limit, id_key=None, prompt_key=None,
                         answer_key=None, image_key=args.image_key, rules_key=None, num_shards=1, shard=0)
    return G.load_puzzles(ns)


def prompt_content(lay):
    content = [{"type": "text", "text": CROPS_PUZZLE_INTRO.format(cell=G.CELL_NAMES[lay["missing"]])}]
    for cell in range(9):
        label = f"\nRow {cell // 3 + 1}, panel {cell % 3 + 1}:"
        if cell == lay["missing"]:
            content.append({"type": "text", "text": label + " ? (missing)"})
        else:
            content += [{"type": "text", "text": label}, {"type": "image"}]
    for L in LETTERS:
        content += [{"type": "text", "text": f"\nChoice {L}:"}, {"type": "image"}]
    content.append({"type": "text", "text": CROPS_PUZZLE_Q})
    return content


class PuzzleSet(torch.utils.data.Dataset):
    def __init__(self, puzzles, lay):
        self.p, self.lay = puzzles, lay

    def __len__(self):
        return len(self.p)

    def __getitem__(self, i):
        p = self.p[i]
        ctx, ch = G.crop_panels(G.open_sheet(p.image), self.lay)
        return ctx + ch, LETTERS.index(p.answer), p.id


class Collate:
    def __init__(self, processor, text, vision_only):
        self.proc, self.text, self.vision_only = processor, text, vision_only

    def __call__(self, batch):
        imgs = [im for b in batch for im in b[0]]
        if self.vision_only:
            try:
                enc = self.proc.image_processor(images=imgs, return_tensors="pt", crop_to_patches=False)
            except TypeError:
                enc = self.proc.image_processor(images=imgs, return_tensors="pt")
            enc = {"pixel_values": enc["pixel_values"]}
        else:
            try:
                enc = self.proc(text=[self.text] * len(batch), images=imgs, return_tensors="pt", padding=True,
                                crop_to_patches=False)
            except TypeError:
                enc = self.proc(text=[self.text] * len(batch), images=imgs, return_tensors="pt", padding=True)
            enc = dict(enc)
        if enc["pixel_values"].shape[0] != 16 * len(batch):
            raise RuntimeError(f"got {enc['pixel_values'].shape[0]} tiles for {16 * len(batch)} panels; this "
                               "transformers version ignores crop_to_patches=False. Upgrade transformers.")
        return enc, torch.tensor([b[1] for b in batch]), [b[2] for b in batch]


# ----------------------------------------------------------------------------- panel cache

def _panel_size(lay):
    boxes = [b for i, b in enumerate(lay["matrix"]) if i != lay["missing"]] + list(lay["choices"])
    return max(b[2] - b[0] for b in boxes), max(b[3] - b[1] for b in boxes)


def _crop_rows(job):
    """Worker: crop the 16 panels of each sheet. Returns uint8 [n, 16, H, W] (or [n, 16, 3, H, W])."""
    paths, lay, (W, H), rgb = job
    from PIL import Image
    out = np.zeros((len(paths), 16) + ((3, H, W) if rgb else (H, W)), np.uint8)
    for i, path in enumerate(paths):
        ctx, ch = G.crop_panels(G.open_sheet(path), lay)
        for k, im in enumerate(ctx + ch):
            if im.size != (W, H):
                im = im.resize((W, H), Image.BICUBIC)
            a = np.asarray(im)
            out[i, k] = a.transpose(2, 0, 1) if rgb else a[..., 0]
    return out


def build_cache(args):
    import multiprocessing as mp
    import shutil
    lay = G.load_layout(args.layout)
    W, H = _panel_size(lay)
    os.makedirs(args.cache, exist_ok=True)
    for split in args.splits:
        puzzles = load_split(args, split, args.limit)
        rgb = False
        for p in puzzles[:20]:                     # the dataset is grayscale; store 1 channel if so
            a = np.asarray(G.open_sheet(p.image))
            rgb |= not ((a[..., 0] == a[..., 1]).all() and (a[..., 1] == a[..., 2]).all())
        shape = (len(puzzles), 16) + ((3, H, W) if rgb else (H, W))
        need = int(np.prod(shape))
        free = shutil.disk_usage(args.cache).free
        print(f"{split}: {len(puzzles)} puzzles -> {shape} uint8 = {need / 1e9:.1f} GB "
              f"({free / 1e9:.0f} GB free in {args.cache})", flush=True)
        if need > free:
            raise SystemExit("not enough disk space for the cache")
        final = os.path.join(args.cache, f"{split}_panels.npy")
        tmp = os.path.join(args.cache, f"{split}_panels.partial.npy")
        mm = np.lib.format.open_memmap(tmp, mode="w+", dtype=np.uint8, shape=shape)
        chunk = 64
        jobs = [([p.image for p in puzzles[i:i + chunk]], lay, (W, H), rgb) for i in range(0, len(puzzles), chunk)]
        t0, done = time.time(), 0
        with mp.get_context("fork").Pool(args.workers) as pool:
            for j, arr in enumerate(pool.imap(_crop_rows, jobs)):
                mm[j * chunk: j * chunk + len(arr)] = arr
                done += len(arr)
                if j % 200 == 0 or done == len(puzzles):
                    print(f"  {done}/{len(puzzles)}  {done / (time.time() - t0):.0f} puzzles/s", flush=True)
        mm.flush()
        del mm
        os.replace(tmp, final)
        json.dump({"split": split, "ids": [p.id for p in puzzles], "answers": [p.answer for p in puzzles],
                   "shape": list(shape), "rgb": rgb, "layout": lay, "data": os.path.abspath(args.data)},
                  open(os.path.join(args.cache, f"{split}_meta.json"), "w"))
        # spot-check: cached panels equal freshly cropped ones
        arr = np.load(final, mmap_mode="r")
        for i in sorted({0, len(puzzles) // 2, len(puzzles) - 1}):
            ref = _crop_rows(([puzzles[i].image], lay, (W, H), rgb))[0]
            assert np.array_equal(arr[i], ref), f"cache row {i} differs from a fresh crop"
        print(f"  wrote {final} (spot-check passed)", flush=True)
    if args.model:
        check_cache_pixels(args, lay)


class CachedPuzzleSet(torch.utils.data.Dataset):
    """Panels from the cache; each item is uint8 [16, H, W] (or [16, 3, H, W])."""

    def __init__(self, puzzles, cache_dir, split, lay=None):
        self.p, self.dir, self.split = puzzles, cache_dir, split
        mpath = os.path.join(cache_dir, f"{split}_meta.json")
        if not os.path.exists(mpath):
            raise SystemExit(f"no cache for split '{split}' in {cache_dir}: run "
                             f"`python train_hybrid.py cache --data $DATA --cache {cache_dir} --splits {split}`")
        meta = json.load(open(mpath))
        if lay is not None and (meta["layout"]["matrix"] != lay["matrix"] or meta["layout"]["choices"] != lay["choices"]):
            raise SystemExit(f"{mpath} was built with a different layout.json; rebuild the cache")
        row = {pid: i for i, pid in enumerate(meta["ids"])}
        missing = [p.id for p in puzzles if p.id not in row]
        if missing:
            raise SystemExit(f"{len(missing)} puzzles are not in the {split} cache (e.g. {missing[0]}); rebuild it")
        self.rows = [row[p.id] for p in puzzles]
        self.mm = None

    def __len__(self):
        return len(self.p)

    def __getitem__(self, i):
        if self.mm is None:      # opened lazily, once per worker
            self.mm = np.load(os.path.join(self.dir, f"{self.split}_panels.npy"), mmap_mode="r")
        p = self.p[i]
        return torch.from_numpy(np.array(self.mm[self.rows[i]])), LETTERS.index(p.answer), p.id


class CachedCollate:
    def __init__(self, text_inputs=None):
        self.text = text_inputs

    def __call__(self, batch):
        panels = torch.stack([b[0] for b in batch])
        enc = {"panels": panels.reshape(-1, *panels.shape[2:])}
        if self.text:
            enc.update({k: v.repeat(len(batch), 1) for k, v in self.text.items()})
        return enc, torch.tensor([b[1] for b in batch]), [b[2] for b in batch]


class PixelSpec:
    """The image processor's resize + rescale + normalise, done on the GPU."""

    def __init__(self, proc):
        ip = proc.image_processor
        sz = ip.size
        get = lambda k: (sz.get(k) if isinstance(sz, dict) else getattr(sz, k, None))
        if isinstance(sz, int):
            self.hw = (sz, sz)
        else:
            self.hw = (int(get("height") or get("shortest_edge")), int(get("width") or get("shortest_edge")))
        self.mean = torch.tensor(ip.image_mean).view(1, 3, 1, 1)
        self.std = torch.tensor(ip.image_std).view(1, 3, 1, 1)
        self.rescale = float(getattr(ip, "rescale_factor", 1 / 255))

    def __call__(self, panels, dev):
        x = panels.to(dev, non_blocking=True).float()
        if x.dim() == 3:
            x = x[:, None].expand(-1, 3, -1, -1)
        # Reproduces the image processor's uint8 resize (antialiased bicubic, PIL's kernel): width pass,
        # round to 8 bits, height pass, round. Matches it to within 2 grey levels on ~0.06% of pixels.
        h, w = x.shape[-2:]
        x = F.interpolate(x, size=(h, self.hw[1]), mode="bicubic", align_corners=False, antialias=True)
        x = F.interpolate(x.round().clamp_(0, 255), size=self.hw, mode="bicubic", align_corners=False, antialias=True)
        x = x.round().clamp_(0, 255)
        return (x * self.rescale - self.mean.to(dev)) / self.std.to(dev)


def text_inputs(proc, text, lay):
    """The tokenised prompt is identical for every puzzle: build it once with dummy panels."""
    from PIL import Image
    W, H = _panel_size(lay)
    dummy = [Image.new("RGB", (W, H))] * 16
    try:
        enc = proc(text=[text], images=dummy, return_tensors="pt", crop_to_patches=False)
    except TypeError:
        enc = proc(text=[text], images=dummy, return_tensors="pt")
    extra = set(enc) - {"input_ids", "attention_mask", "pixel_values"}
    if extra:
        raise SystemExit(f"this processor returns {sorted(extra)}; run without --cache")
    if enc["pixel_values"].shape[0] != 16:
        raise SystemExit("the processor does not give one tile per panel; upgrade transformers or run without --cache")
    return {"input_ids": enc["input_ids"], "attention_mask": enc["attention_mask"]}


def check_cache_pixels(args, lay):
    """Compare GPU preprocessing of cached panels with the processor on the original crops."""
    from transformers import AutoProcessor
    proc = AutoProcessor.from_pretrained(args.model)
    split = args.splits[0]
    puzzles = load_split(args, split, 2)
    ds = CachedPuzzleSet(puzzles, args.cache, split, lay)
    spec = PixelSpec(proc)
    diffs = []
    for i, p in enumerate(puzzles):
        ctx, ch = G.crop_panels(G.open_sheet(p.image), lay)
        try:
            ref = proc.image_processor(images=ctx + ch, return_tensors="pt", crop_to_patches=False)["pixel_values"]
        except TypeError:
            ref = proc.image_processor(images=ctx + ch, return_tensors="pt")["pixel_values"]
        got = spec(ds[i][0], "cpu")
        diffs.append((got - ref.float()).abs())
    d = torch.cat([x.flatten() for x in diffs])
    level = (1 / 255) / float(spec.std.max())
    print(f"GPU preprocessing vs processor: mean |diff| {d.mean():.4f}, max {d.max():.4f} "
          f"(one grey level = {level:.4f} in normalised units)", flush=True)


def loader(ds, coll, batch_size, workers, order=None):
    sub = ds if order is None else torch.utils.data.Subset(ds, order)
    return torch.utils.data.DataLoader(sub, batch_size=batch_size, shuffle=False, num_workers=workers,
                                       collate_fn=coll, persistent_workers=False, pin_memory=True)


# ----------------------------------------------------------------------------- model

def lora_config(args):
    from peft import LoraConfig
    return LoraConfig(r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=args.lora_dropout, bias="none",
                      target_modules=r".*language_model.*\.(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)")


def build(args, mode, lay):
    """Returns (model, processor, fusion or None, meta)."""
    from transformers import AutoProcessor, AutoModelForImageTextToText
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = getattr(torch, args.dtype)
    proc = AutoProcessor.from_pretrained(args.model)
    proc.tokenizer.padding_side = "left"
    model = AutoModelForImageTextToText.from_pretrained(args.model, dtype=dtype, attn_implementation=args.attn).to(dev)
    for p in model.parameters():
        p.requires_grad_(False)

    init = load_ckpt(args.init) if args.init else None
    meta = dict(init["meta"]) if init else {}
    fusion = None
    hybrid = mode in ("stage0", "stage1", "stage2") or (mode == "eval" and init and init.get("fusion"))
    if hybrid:
        if mode == "stage1":       # architecture comes from the stage-0 branches
            cfgs = [load_ckpt(p)["meta"]["fusion_config"] for p in args.branches]
            key = lambda c: {k: c.get(k) for k in ("source", "slot_layer", "rel_dim", "n_slots", "abs_layers")}
            if any(key(c) != key(cfgs[0]) for c in cfgs):
                raise SystemExit(f"the stage-0 branches differ in architecture: {[key(c) for c in cfgs]}")
            cfg = dict(cfgs[0], n_branches=len(args.branches), gate=args.gate)
        else:
            cfg = meta.get("fusion_config") or dict(
                rel_dim=args.rel_dim, n_slots=args.n_slots, slot_iters=args.slot_iters, abs_layers=args.abs_layers,
                gate=args.gate, n_branches=1, source=args.slot_source, slot_layer=args.slot_layer,
                recon_res=args.recon_res, recon_panels=args.recon_panels, qk_norm=bool(args.qk_norm))
        cfg = {k: v for k, v in cfg.items() if k not in ("missing", "panels")}
        cfg.setdefault("qk_norm", False)       # checkpoints from before the fix had no normalisation
        fusion = RF.install(model, missing=lay["missing"], **cfg)
        if init and init.get("fusion"):
            fusion.load_state_dict(init["fusion"], strict=False)
        if mode == "stage1" and args.branches:
            for i, path in enumerate(args.branches):
                st = load_ckpt(path)["fusion"]
                br = {k[len("branches.0."):]: v for k, v in st.items() if k.startswith("branches.0.")}
                fusion.branches[i].load_state_dict(br)
                print(f"branch {i} <- {path}")
            meta["branch_halves"] = [load_ckpt(p)["meta"].get("half", "all") for p in args.branches]
        meta["fusion_config"] = dict(fusion.config)
    # LoRA weights come from --init (stage2 after a LoRA-first stage1, a continued baseline, eval) or from
    # --lora_from (stage1/stage2: a trained baseline's LoRA). Stage 1 and eval keep them frozen.
    lora_src = init if (init and init.get("lora")) else None
    if lora_src is None and getattr(args, "lora_from", None):
        lora_src = load_ckpt(args.lora_from)
        if not lora_src.get("lora"):
            raise SystemExit(f"--lora_from {args.lora_from}: that checkpoint has no LoRA weights")
        meta["lora_from"] = os.path.abspath(args.lora_from)
    if mode in ("stage2", "baseline") or lora_src is not None:
        from peft import get_peft_model, set_peft_model_state_dict
        cfg = lora_config(SimpleNamespace(**lora_src["meta"]["lora"])) if lora_src else lora_config(args)
        model = get_peft_model(model, cfg)
        if lora_src:
            set_peft_model_state_dict(model, lora_src["lora"])
        meta["lora"] = dict(lora_r=cfg.r, lora_alpha=cfg.lora_alpha, lora_dropout=cfg.lora_dropout)
        for n, p in model.named_parameters():
            p.requires_grad_("lora_" in n and mode in ("stage2", "baseline"))
    if fusion is not None and mode != "eval":
        fusion.float()   # trainable fusion weights in fp32; autocast handles bf16 activations
        fusion.projector.to(dtype)
    model.eval()
    return model, proc, fusion, meta


def load_ckpt(path):
    f = path if path.endswith(".pt") else os.path.join(path, "best.pt")
    return torch.load(f, map_location="cpu", weights_only=False)


def save_ckpt(path, model, fusion, meta, extra=None):
    st = {"meta": meta}
    if fusion is not None:
        st["fusion"] = RF.fusion_state(fusion)
    if hasattr(model, "peft_config"):
        from peft import get_peft_model_state_dict
        st["lora"] = {k: v.detach().cpu() for k, v in get_peft_model_state_dict(model).items()}
    if extra:
        st.update(extra)
    tmp = path + ".tmp"
    torch.save(st, tmp)
    os.replace(tmp, path)


def vision_inner(model):
    m = model.get_base_model() if hasattr(model, "get_base_model") else model
    return m.model if hasattr(m.model, "get_image_features") else m


def letter_token_ids(tok):
    ids = []
    for L in LETTERS:
        t = tok.encode(L, add_special_tokens=False)
        assert len(t) == 1, f"letter {L} is not one token"
        ids.append(t[0])
    return torch.tensor(ids)


def vit_prefix(tower, pixel_values, L):
    """Run only InternViT's patch embedding and its first L layers (0 = embeddings only). Stage 0 with slots
    from layer L needs nothing above it, so this gives the same features as the full pass, much faster."""
    h = tower.embeddings(pixel_values)
    h = h[0] if isinstance(h, tuple) else h
    for layer in tower.encoder.layer[:L]:
        h = layer(h)
        h = h[0] if isinstance(h, tuple) else h
    return h


def forward(model, fusion, enc, mode, dev, dtype, pix=None):
    """Returns (letter logits over the full vocabulary at the answer position or None, candidate logits or None)."""
    if "panels" in enc:
        enc = dict(enc)
        enc["pixel_values"] = pix(enc.pop("panels"), dev)
    enc = {k: (v.to(dev) if torch.is_tensor(v) else v) for k, v in enc.items()}
    ac = torch.autocast(device_type="cuda" if dev == "cuda" else "cpu", dtype=dtype, enabled=dtype != torch.float32)
    with ac:
        if mode == "stage0":
            if fusion.config["source"] in ("cnn", "resnet"):      # the ViT is not needed
                fusion._pixels = enc["pixel_values"].to(dtype)
                fusion.stage0(None)
            elif fusion.config["slot_layer"] is not None and fusion.config["slot_layer"] >= 0:
                with torch.no_grad():                               # frozen ViT, stopped at the slot layer
                    fusion._layer_feats = vit_prefix(vision_inner(model).vision_tower,
                                                     enc["pixel_values"].to(dtype), fusion.config["slot_layer"])
                fusion._pixels = enc["pixel_values"].to(dtype)
                fusion.stage0(None)
            else:
                inner = vision_inner(model)
                cfg = inner.config
                fusion.stage0_only = True
                inner.get_image_features(pixel_values=enc["pixel_values"].to(dtype),
                                         vision_feature_layer=cfg.vision_feature_layer,
                                         vision_feature_select_strategy=cfg.vision_feature_select_strategy)
                fusion.stage0_only = False
            return None, fusion.last["candidate_logits"].float()
        out = model(**enc, logits_to_keep=1)
    cand = fusion.last.get("candidate_logits") if (fusion is not None and fusion.enabled) else None
    return out.logits[:, -1].float(), (cand.float() if cand is not None else None)


# ----------------------------------------------------------------------------- evaluation

@torch.no_grad()
def evaluate(args, mode, model, fusion, val_ds, coll, letter_ids, dev, dtype, write=None, pix=None):
    model.eval()
    if fusion is not None:
        fusion.route = None
        fusion.drop = None
    rows, tot_loss, n = [], 0.0, 0
    for enc, y, ids in loader(val_ds, coll, args.eval_batch_size, args.workers):
        logits, cand = forward(model, fusion, enc, "stage0" if mode == "stage0" else "lm", dev, dtype, pix)
        y = y.to(dev)
        if mode == "stage0":
            tot_loss += F.cross_entropy(cand, y, reduction="sum").item()
            lp = cand.log_softmax(-1)
            mass = torch.ones(len(y))
        else:
            tot_loss += F.cross_entropy(logits, letter_ids.to(dev)[y], reduction="sum").item()
            full = logits.log_softmax(-1)[:, letter_ids.to(dev)]
            mass = full.exp().sum(-1).cpu()
            lp = full - full.logsumexp(-1, keepdim=True)
        n += len(y)
        for i in range(len(y)):
            r = {"id": ids[i], "answer": LETTERS[y[i]], "pred": LETTERS[int(lp[i].argmax())],
                 "letter_logprobs": [round(float(x), 4) for x in lp[i]], "letter_mass": round(float(mass[i]), 4)}
            r["correct"] = r["pred"] == r["answer"]
            if cand is not None and mode != "stage0":
                r["candidate_logprobs"] = [round(float(x), 4) for x in cand[i].log_softmax(-1)]
            rows.append(r)
    res = {"loss": tot_loss / n, "acc": float(np.mean([r["correct"] for r in rows])), "n": n}
    if rows and "candidate_logprobs" in rows[0]:
        cl = np.array([r["candidate_logprobs"] for r in rows])
        ll = np.array([r["letter_logprobs"] for r in rows])
        ans = np.array([LETTERS.index(r["answer"]) for r in rows])
        res["score_head_acc"] = float((cl.argmax(1) == ans).mean())
        res["ensemble_acc"] = float(((cl + ll).argmax(1) == ans).mean())
    if write:
        with open(write, "w") as fh:
            for r in rows:
                fh.write(json.dumps(r) + "\n")
    return res


# ----------------------------------------------------------------------------- training

def param_groups(args, mode, model, fusion):
    groups = []
    if mode in ("stage2", "baseline"):
        groups.append({"params": [p for n, p in model.named_parameters() if "lora_" in n], "lr": args.lr_lora,
                       "name": "lora"})
    if fusion is not None:
        head = [p for n, p in fusion.named_parameters()
                if n.split(".")[0] in ("up", "gate", "nv", "nu")]
        rel = [p for n, p in fusion.named_parameters() if n.startswith("branches.") and ".trunk." not in n
               and (mode == "stage0" or ".decoder." not in n)]
        if mode in ("stage1", "stage2"):
            groups.append({"params": head, "lr": args.lr_head, "name": "fusion_head"})
        if mode == "stage0":
            groups.append({"params": rel, "lr": args.lr_rel, "name": "relational"})
        elif mode == "stage2" and args.lr_rel > 0:
            groups.append({"params": rel, "lr": args.lr_rel, "name": "relational"})
    for g in groups:
        for p in g["params"]:
            p.requires_grad_(True)
    return [g for g in groups if g["params"]]


def train(args, mode):
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    lay = G.load_layout(args.layout)
    os.makedirs(args.out, exist_ok=True)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = getattr(torch, args.dtype)
    model, proc, fusion, meta = build(args, mode, lay)
    letter_ids = letter_token_ids(proc.tokenizer)
    text = proc.apply_chat_template([{"role": "user", "content": prompt_content(lay)}], tokenize=False,
                                    add_generation_prompt=True)
    if args.val_from_train:
        pool = load_split(args, "train", 0)
        val_p, val_split = pool[-args.val_from_train:], "train"
        held = {p.id for p in val_p}
        train_p = [p for p in (pool[:args.train_limit] if args.train_limit else pool) if p.id not in held]
        if args.val_limit:
            val_p = val_p[:args.val_limit]
    else:
        train_p = load_split(args, "train", args.train_limit)
        val_p, val_split = load_split(args, "validation", args.val_limit), "validation"
    if mode == "stage0" and args.half != "all":
        train_p = [p for p in train_p if half_of(p.id) == args.half]
    pix = None
    if args.cache:
        pix = PixelSpec(proc)
        coll = CachedCollate(None if mode == "stage0" else text_inputs(proc, text, lay))
        ds = CachedPuzzleSet(train_p, args.cache, "train", lay)
        val_ds = CachedPuzzleSet(val_p, args.cache, val_split, lay)
    else:
        coll = Collate(proc, text, vision_only=(mode == "stage0"))
        ds, val_ds = PuzzleSet(train_p, lay), PuzzleSet(val_p, lay)
    route_of = None
    halves = meta.get("branch_halves") or []
    if fusion is not None and mode in ("stage1", "stage2") and len(fusion.branches) > 1 and sorted(halves) == ["A", "B"]:
        # a puzzle from half A goes to the branch trained on half B, and vice versa
        route_of = lambda pid: halves.index("B" if half_of(pid) == "A" else "A")

    groups = param_groups(args, mode, model, fusion)
    if fusion is not None:
        fusion.compute_recon = mode == "stage0" and args.recon_weight > 0
        meta["slot"] = f"{fusion.config['source']}" + (f" layer {fusion.config['slot_layer']}"
                                                       if fusion.config["source"] == "vit" else "")
    n_trainable = sum(p.numel() for g in groups for p in g["params"])
    print(f"{mode}: {len(train_p)} training puzzles, {len(val_ds)} validation "
          f"({'last ' + str(len(val_p)) + ' rows of train.jsonl' if args.val_from_train else 'validation.jsonl'}"
          f"{', cached panels' if args.cache else ''}); trainable "
          f"{n_trainable / 1e6:.2f}M in {[g['name'] for g in groups]}", flush=True)
    opt = torch.optim.AdamW(groups, weight_decay=args.weight_decay)
    per_epoch = math.ceil(len(ds) / (args.batch_size * args.grad_accum))
    total = args.max_steps if args.max_steps else math.ceil(args.epochs * per_epoch)
    warm = args.warmup_steps if args.warmup_steps else int(args.warmup * total)
    plan = {"total": total, "stopped": False}   # --patience can move the end earlier (kept in last.pt)
    if args.schedule == "constant":
        ws = max(1, args.warmup_steps or 500)
        K = args.decay_steps

        def lr_lambda(s):
            f = min(1.0, (s + 1) / ws)
            if K and s >= plan["total"] - K:        # linear decay to 0 over the last K steps
                f *= max(0.0, (plan["total"] - s) / K)
            return f
        sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)
    else:
        from transformers import get_cosine_schedule_with_warmup
        sched = get_cosine_schedule_with_warmup(opt, warm, total)
    if args.grad_ckpt and mode != "stage0":
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()

    step, best, history = 0, None, []
    last_path = os.path.join(args.out, "last.pt")
    if os.path.exists(last_path) and not args.no_resume:
        st = torch.load(last_path, map_location="cpu", weights_only=False)
        if fusion is not None and st.get("fusion"):
            fusion.load_state_dict(st["fusion"], strict=False)
        if st.get("lora"):
            from peft import set_peft_model_state_dict
            set_peft_model_state_dict(model, st["lora"])
        opt.load_state_dict(st["opt"])
        sched.load_state_dict(st["sched"])
        step, best, history = st["step"], st["best"], st["history"]
        if st.get("plan") and (st["plan"]["stopped"] or st["plan"]["total"] < total):
            plan.update(st["plan"])               # an early stop / decay decided by --patience stays decided
        total = plan["total"]
        print(f"resumed from step {step} (training to step {total})", flush=True)
    if step >= total:
        print(f"already at step {step} >= {total}; nothing to train (raise --max_steps to continue)", flush=True)
    meta.update({"mode": mode, "half": args.half if mode == "stage0" else None, "model": args.model,
                 "train_puzzles": len(train_p), "args": vars(args)})

    log = open(os.path.join(args.out, "train_log.jsonl"), "a")
    t0, seen = time.time(), 0
    deadline = t0 + args.max_hours * 3600 if args.max_hours else None
    timed_out = False
    drop_p = args.branch_dropout if (mode == "stage2" and fusion is not None) else 0.0

    def save_last():
        save_ckpt(last_path, model, fusion, meta, {"opt": opt.state_dict(), "sched": sched.state_dict(),
                                                   "step": step, "best": best, "history": history, "plan": plan})
    while step < total and not timed_out:
        epoch, pos = divmod(step, per_epoch)
        order = list(range(len(ds)))
        if args.shuffle:
            random.Random(args.seed + epoch).shuffle(order)
        order = order[pos * args.batch_size * args.grad_accum:]
        model.train()
        it = iter(loader(ds, coll, args.batch_size, args.workers, order))
        while step < total:
            opt.zero_grad(set_to_none=True)
            stats, got = {"loss": 0.0, "ce": 0.0, "acc": 0.0, "aux": 0.0}, 0
            for _ in range(args.grad_accum):
                try:
                    enc, y, ids = next(it)
                except StopIteration:
                    break
                got += 1
                if route_of is not None:
                    fusion.route = [route_of(i) for i in ids]
                if drop_p > 0:            # reproducible after a resume: depends on seed, step and puzzle
                    fusion.drop = [random.Random(f"{args.seed}/{step}/{i}").random() < drop_p for i in ids]
                logits, cand = forward(model, fusion, enc, "stage0" if mode == "stage0" else "lm", dev, dtype, pix)
                y = y.to(dev)
                if mode == "stage0":
                    loss = F.cross_entropy(cand, y)
                    stats["ce"] += loss.item() / args.grad_accum
                    stats["acc"] += (cand.argmax(-1) == y).float().mean().item() / args.grad_accum
                    stats["spread"] = stats.get("spread", 0.0) + cand.std(-1).mean().item() / args.grad_accum
                    rl = fusion.last.get("recon_loss")
                    if rl is not None:
                        stats["recon"] = stats.get("recon", 0.0) + rl.item() / args.grad_accum
                        loss = loss + args.recon_weight * rl
                else:
                    loss = F.cross_entropy(logits, letter_ids.to(dev)[y])
                    stats["ce"] += loss.item() / args.grad_accum
                    stats["acc"] += (logits[:, letter_ids.to(dev)].argmax(-1) == y).float().mean().item() / args.grad_accum
                    if mode == "stage2" and cand is not None and args.aux_weight > 0:
                        aux = F.cross_entropy(cand, y)
                        stats["aux"] += aux.item() / args.grad_accum
                        loss = loss + args.aux_weight * aux
                (loss / args.grad_accum).backward()
                stats["loss"] += loss.item() / args.grad_accum
                seen += len(y)
            if got == 0:
                break
            torch.nn.utils.clip_grad_norm_([p for g in groups for p in g["params"]], args.max_grad_norm)
            opt.step()
            sched.step()
            step += 1
            if step % args.log_every == 0 or step == 1:
                rec = {"step": step, "of": total, "epoch": round(step / per_epoch, 3), "lr": sched.get_last_lr()[0],
                       **{k: round(v, 4) for k, v in stats.items()},
                       "puzzles_per_s": round(seen / (time.time() - t0), 2), "t": round(time.time(), 1)}
                if fusion is not None and fusion.last and mode != "stage0":
                    rec["gate_mean"] = round(float(fusion.last["gate_mean"]), 4)
                if fusion is not None:
                    rec.update(fusion.diagnostics())
                print(json.dumps(rec), flush=True)
                log.write(json.dumps(rec) + "\n")
                log.flush()
            if step % args.eval_every == 0 or step == total:
                res = evaluate(args, mode, model, fusion, val_ds, coll, letter_ids, dev, dtype, pix=pix)
                res.update({"step": step, "epoch": round(step / per_epoch, 3)})
                history.append(res)
                print("VAL " + json.dumps(res), flush=True)
                log.write(json.dumps({"val": res}) + "\n")
                log.flush()
                key = res["loss"] if args.select == "loss" else -res["acc"]
                if best is None or key < best["key"]:
                    best = {"key": key, **res}
                    save_ckpt(os.path.join(args.out, "best.pt"), model, fusion, dict(meta, selected=res))
                    print(f"  new best at step {step}", flush=True)
                since = sum(1 for h in history if h["step"] > best["step"])
                decaying = bool(args.schedule == "constant" and args.decay_steps
                                and step >= plan["total"] - args.decay_steps)
                if args.patience and since >= args.patience and step < total and not decaying:   # a decay runs to its end
                    if args.schedule == "constant" and args.decay_steps:
                        plan["total"] = total = step + args.decay_steps
                        plan["plateau_at"] = step
                        msg = f"plateau ({since} validations without a new best): decaying the LR to 0 by step {total}"
                    else:
                        plan["total"], plan["stopped"], total = step, True, step
                        msg = f"early stop: {since} validations without a new best"
                    print("  " + msg, flush=True)
                    log.write(json.dumps({"event": msg, "step": step}) + "\n")
                    log.flush()
                model.train()
            if step % args.save_every == 0 or step == total:
                save_last()
            if deadline and time.time() > deadline and step < total:
                save_last()
                timed_out = True
                print(f"time budget of {args.max_hours} h reached at step {step}/{total}: saved last.pt; "
                      "run the same command again to continue", flush=True)
                break
        if fusion is not None:
            fusion.route = None
            fusion.drop = None
    log.close()
    if timed_out:
        return
    json.dump({"best": best, "stopped_early": plan["stopped"], "plateau_at": plan.get("plateau_at"), "history": history, "step": step, "slot": meta.get("slot"),
               "fusion_config": meta.get("fusion_config"), "recon_weight": args.recon_weight,
               "val_set": f"last {len(val_p)} of train.jsonl" if args.val_from_train else "validation.jsonl"},
              open(os.path.join(args.out, "summary.json"), "w"), indent=1)
    print(f"done; best by validation {args.select}: {json.dumps(best)}")


def run_eval(args):
    lay = G.load_layout(args.layout)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = getattr(torch, args.dtype)
    model, proc, fusion, meta = build(args, "eval", lay)
    if fusion is not None:
        fusion.enabled = not args.no_fusion
        fusion.shuffle = args.shuffle_relational
        if args.shuffle_relational and args.eval_batch_size < 2:
            raise SystemExit("--shuffle_relational needs --eval_batch_size >= 2")
    text = proc.apply_chat_template([{"role": "user", "content": prompt_content(lay)}], tokenize=False,
                                    add_generation_prompt=True)
    puzzles = load_split(args, args.split, args.limit)
    pix = None
    if args.cache and os.path.exists(os.path.join(args.cache, f"{args.split}_meta.json")):
        pix = PixelSpec(proc)
        coll, ds = CachedCollate(text_inputs(proc, text, lay)), CachedPuzzleSet(puzzles, args.cache, args.split, lay)
    else:
        coll, ds = Collate(proc, text, vision_only=False), PuzzleSet(puzzles, lay)
    tag = args.tag or ("_nofusion" if args.no_fusion else "") + ("_shuffled" if args.shuffle_relational else "")
    folder = args.init if os.path.isdir(args.init) else os.path.dirname(os.path.abspath(args.init))
    out = os.path.join(folder, f"eval_{args.split}{tag}.jsonl")
    res = evaluate(args, "eval", model, fusion, ds, coll, letter_token_ids(proc.tokenizer), dev, dtype, write=out,
                   pix=pix)
    p, lo, hi = G.wilson(round(res["acc"] * res["n"]), res["n"])
    print(f"{args.split}{tag}: accuracy {p:.2%} [{lo:.2%}, {hi:.2%}] n={res['n']}, loss {res['loss']:.4f}")
    if "score_head_acc" in res:
        print(f"  relational score head alone {res['score_head_acc']:.2%}; ensemble (LLM + score head) "
              f"{res['ensemble_acc']:.2%}. The hybrid should beat the ensemble.")
    print(f"  predictions: {out}")
    json.dump(res, open(out.replace(".jsonl", "_summary.json"), "w"), indent=1)


# ----------------------------------------------------------------------------- cli

DEFAULTS = {   # mode -> (epochs, lr_lora, lr_head, lr_rel, batch_size, grad_accum)
    "stage0": (3, 0, 0, 3e-4, 16, 1),
    "stage1": (0.25, 0, 1e-3, 0, 4, 4),
    "stage2": (6, 1e-4, 1e-4, 3e-5, 4, 4),
    "baseline": (6, 1e-4, 0, 0, 4, 4),
}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mode", choices=["stage0", "stage1", "stage2", "baseline", "eval", "cache"])
    ap.add_argument("--data", default=os.environ.get("DATA"), help="dataset root (default $DATA)")
    ap.add_argument("--layout", default=os.environ.get("LAYOUT") or os.path.join(REPO, "probes", "layout.json"),
                    help="panel boxes (default: $LAYOUT, else probes/layout.json)")
    ap.add_argument("--model", default="OpenGVLab/InternVL3_5-1B-HF")
    ap.add_argument("--out", default=None, help="checkpoint folder for this run")
    ap.add_argument("--init", help="checkpoint folder (or .pt) to start from / to evaluate")
    ap.add_argument("--branches", nargs="+", help="stage1: stage-0 checkpoint folder(s); two = cross-fitting")
    ap.add_argument("--half", choices=["all", "A", "B"], default="all", help="stage0: train on this half of the pool")
    ap.add_argument("--image_key", default="images")
    ap.add_argument("--train_limit", type=int, default=0, help="use only the first N training rows (0 = all)")
    ap.add_argument("--val_limit", type=int, default=0, help="validation rows used for selection (0 = all 1000)")
    ap.add_argument("--split", default="test", help="eval: split to score")
    ap.add_argument("--limit", type=int, default=0, help="eval: first N rows")
    ap.add_argument("--tag", default="", help="eval: suffix for the output file")
    ap.add_argument("--no_fusion", action="store_true", help="eval control: relational path off")
    ap.add_argument("--shuffle_relational", action="store_true", help="eval control: other puzzles' relational features")
    ap.add_argument("--epochs", type=float)
    ap.add_argument("--lr_lora", type=float), ap.add_argument("--lr_head", type=float), ap.add_argument("--lr_rel", type=float)
    ap.add_argument("--batch_size", type=int), ap.add_argument("--grad_accum", type=int)
    ap.add_argument("--eval_batch_size", type=int, default=8)
    ap.add_argument("--aux_weight", type=float, default=0.2, help="stage2: weight of the score-head loss")
    ap.add_argument("--branch_dropout", type=float, default=0.0,
                    help="stage2: probability of switching a puzzle's relational path off (0.5 avoided the shortcut)")
    ap.add_argument("--lora_from", help="stage1/stage2: start from this checkpoint's LoRA (a trained baseline)")
    ap.add_argument("--max_hours", type=float, default=0,
                    help="save last.pt and exit after this many hours (0 = no limit); rerun to continue")
    ap.add_argument("--patience", type=int, default=0,
                    help="validations without a new best before stopping (or decaying, see --decay_steps); 0 = off")
    ap.add_argument("--decay_steps", type=int, default=0,
                    help="constant schedule: decay the LR linearly to 0 over the last N steps (or after a plateau)")
    ap.add_argument("--warmup", type=float, default=0.03, help="cosine: warmup as a fraction of the run")
    ap.add_argument("--warmup_steps", type=int, default=0, help="warmup in optimizer steps (constant default: 500)")
    ap.add_argument("--schedule", choices=["cosine", "constant"], default="cosine",
                    help="constant = warmup then flat; lets you extend a run with a larger --max_steps")
    ap.add_argument("--max_steps", type=int, default=0, help="train to this many optimizer steps (overrides --epochs)")
    ap.add_argument("--val_from_train", type=int, default=0,
                    help="validate on the last N rows of train.jsonl (excluded from training) instead of validation.jsonl")
    ap.add_argument("--cache", default=None, help="panel cache folder (built with the `cache` mode)")
    ap.add_argument("--splits", nargs="+", default=["train", "validation"], help="cache: splits to build")
    ap.add_argument("--weight_decay", type=float, default=0.0)
    ap.add_argument("--max_grad_norm", type=float, default=1.0)
    ap.add_argument("--select", choices=["loss", "acc"], default="loss", help="checkpoint selection on validation")
    ap.add_argument("--shuffle", action="store_true", help="shuffle each epoch (default: file order, as your baselines)")
    ap.add_argument("--lora_r", type=int, default=16), ap.add_argument("--lora_alpha", type=int, default=32)
    ap.add_argument("--lora_dropout", type=float, default=0.05)
    ap.add_argument("--rel_dim", type=int, default=256), ap.add_argument("--n_slots", type=int, default=10)
    ap.add_argument("--slot_iters", type=int, default=3), ap.add_argument("--abs_layers", type=int, default=2)
    ap.add_argument("--gate", choices=["channel", "token"], default="channel")
    ap.add_argument("--qk_norm", type=int, default=1,
                    help="stage0: LayerNorm slot features before abstractor attention (1, the fix) or not (0, old)")
    ap.add_argument("--slot_source", choices=["vit", "cnn", "resnet"], default="vit",
                    help="stage0: features for slot attention (later stages inherit it)")
    ap.add_argument("--slot_layer", type=int, default=-1,
                    help="stage0, vit: InternViT layer for the slots (0 = patch embeddings, -1 = last)")
    ap.add_argument("--recon_weight", type=float, default=0.0,
                    help="stage0: weight of the slot reconstruction loss (0 = off)")
    ap.add_argument("--recon_res", type=int, default=64)
    ap.add_argument("--recon_panels", type=int, default=32, help="panels per step used for reconstruction")
    ap.add_argument("--eval_every", type=int, default=1000, help="optimizer steps between validations")
    ap.add_argument("--save_every", type=int, default=500, help="optimizer steps between resumable saves")
    ap.add_argument("--log_every", type=int, default=50)
    ap.add_argument("--grad_ckpt", type=int, default=1)
    ap.add_argument("--no_resume", action="store_true")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--attn", default="sdpa")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    if not args.data:
        ap.error("give the dataset root with --data (or export DATA=...)")
    if args.mode == "cache":
        if not args.cache:
            ap.error("cache needs --cache <folder>")
        if args.workers < 2:
            args.workers = 2
        return build_cache(args)
    if args.mode == "eval":
        if not args.init:
            ap.error("eval needs --init <checkpoint>")
        if args.no_fusion and args.shuffle_relational:
            ap.error("pick one control at a time")
        return run_eval(args)
    if not args.out:
        ap.error("--out is required for training")
    if args.mode == "stage1" and not args.branches:
        ap.error("stage1 needs --branches <stage-0 checkpoint(s)>")
    if args.mode == "stage2" and not args.init:
        ap.error("stage2 needs --init <stage-1 checkpoint>")
    if args.lora_from and args.mode not in ("stage1", "stage2"):
        ap.error("--lora_from is for stage1/stage2 (use --init to continue a baseline)")
    if args.branch_dropout and args.mode != "stage2":
        ap.error("--branch_dropout is for stage2")
    for k, v in zip(["epochs", "lr_lora", "lr_head", "lr_rel", "batch_size", "grad_accum"], DEFAULTS[args.mode]):
        if getattr(args, k) is None:
            setattr(args, k, v)
    train(args, args.mode)


if __name__ == "__main__":
    main()
