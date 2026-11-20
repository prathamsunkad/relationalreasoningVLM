#!/usr/bin/env python
"""
Attribute recognition vs rule inference on PGM-style puzzles.

Hypothesis under test: a VLM fails PGM mainly because it has to acquire PGM's attribute concepts
(grey levels, size steps, counts, grid positions, line types of abstract shapes) before it can
induce rules over them; given the attributes, it would infer the rules easily.

Real PGM has no per-panel attribute labels (the release stores only the answer, relation_structure
and meta_target), so this script generates PGM-style puzzles whose every panel attribute is known:
the same objects (shapes, lines), attributes (shape type, colour, size, number, position; line type,
line colour) and relations (progression, consistent union, XOR, OR, AND) as PGM, one rule per puzzle,
answer in the bottom-right cell, 8 choices. Every puzzle is checked to have exactly one valid choice.

Stages
  synth    Generate puzzles: panel PNGs, puzzles.jsonl with every panel's attributes, the rule and the
           answer. With --template_jsonl, also a contact sheet per puzzle in YOUR sheet format (the
           panels pasted into a real sheet using layout.json), so sheet-trained fine-tuned models can be
           tested with their own prompt.
  attrs    ATTRIBUTE RECOGNITION, no rules involved. Per panel: shape type (5-way), number (9-way),
           position occupancy (yes/no, balanced), line type present (yes/no, balanced). Per pair of
           panels from the same puzzle: which is darker / larger / has darker lines (2-way), recorded
           with the size of the difference. Scored on option log-probabilities (no parsing).
  infer    RULE INFERENCE under controlled inputs (same puzzles, same question, chance 12.5%):
             images          the 16 panels as images
             gt_text         ground-truth attribute descriptions only, no images
             images_gt_text  images + ground-truth descriptions
             images_rule     images + which rule applies (object, attribute, relation)
             gt_text_rule    ground-truth descriptions + which rule applies
             sheet           the contact sheet with your dataset prompt (needs synth --template_jsonl)
           Every condition lists the possible rule types, so `_rule` adds only WHICH rule applies.
  report   Tables per model: attribute accuracy, puzzle accuracy per condition and per rule attribute,
           and the link between recognising an attribute and the gain from being given it.

Reading the results (the hypothesis predicts the first row):
  attributes poorly recognised, gt_text >> images, images_rule ~ images   -> perception/attribute bottleneck
  attributes well recognised, gt_text ~ images (both low)                  -> rule-induction bottleneck
  gt_text low but gt_text_rule high                                         -> identifying the rule is hard,
                                                                               applying it is easy

  python pgm_attr_probe.py synth  --out probe --n 1000 [--template_jsonl $DATA/validation.jsonl --data $DATA --layout layout.json]
  python pgm_attr_probe.py attrs  --out probe --model OpenGVLab/InternVL3_5-1B-HF
  python pgm_attr_probe.py infer  --out probe --model OpenGVLab/InternVL3_5-1B-HF --conditions images gt_text images_gt_text images_rule gt_text_rule
  python pgm_attr_probe.py report --out probe

Results go to <out>/<model-tag>/ ; every stage resumes where it stopped (per-question JSONL).
Run attrs/infer once for the base model and once for each fine-tuned checkpoint (HF format).
"""
import argparse
import hashlib
import json
import math
import os
import random
import sys
from collections import defaultdict
from types import SimpleNamespace

import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pgm_gap as G   # noqa: E402  VLM wrapper, cropping/upscaling, Wilson intervals

LETTERS = "ABCDEFGHI"
SHAPES = ["triangle", "square", "pentagon", "hexagon", "circle"]
SIDES = {"triangle": 3, "square": 4, "pentagon": 5, "hexagon": 6}
LINE_TYPES = ["diagonal down", "diagonal up", "horizontal", "vertical", "square frame", "circle"]
POS = G.POS9
N_SIZE = 6

RULES = [("shape", "type", "progression"), ("shape", "type", "consistent union"),
         ("shape", "colour", "progression"), ("shape", "colour", "consistent union"),
         ("shape", "size", "progression"), ("shape", "size", "consistent union"),
         ("shape", "number", "progression"), ("shape", "number", "consistent union"),
         ("shape", "position", "XOR"), ("shape", "position", "OR"), ("shape", "position", "AND"),
         ("line", "type", "XOR"), ("line", "type", "OR"), ("line", "type", "AND"),
         ("line", "colour", "progression"), ("line", "colour", "consistent union")]
SET_OPS = {"XOR": lambda a, b: a ^ b, "OR": lambda a, b: a | b, "AND": lambda a, b: a & b}
DOMAIN = {("shape", "type"): list(range(5)), ("shape", "colour"): list(range(10)),
          ("shape", "size"): list(range(1, N_SIZE + 1)), ("shape", "number"): list(range(1, 10)),
          ("line", "colour"): list(range(1, 10))}
# which attribute probe measures the attribute a rule is about
PROBE_FOR = {("shape", "type"): "type", ("shape", "number"): "number", ("shape", "position"): "position",
             ("shape", "colour"): "colour_rel", ("shape", "size"): "size_rel",
             ("line", "type"): "line_type", ("line", "colour"): "line_colour_rel"}

SCALES = ("Panels are described by their shapes (type; color from 0 = white to 9 = black; size from 1 = smallest "
          f"to {N_SIZE} = largest; positions on a 3x3 layout inside the panel) and their lines (types; line color "
          "from 1 = light grey to 9 = black).")
RULE_MENU = ("Each puzzle follows one rule, applied the same way in every row, to one attribute of the shapes or "
             "lines: 'progression' (the value increases or decreases by a constant step along the row), "
             "'consistent union' (every row contains the same three values, in some order), or a set rule on "
             "shape positions or line types: 'XOR', 'OR', 'AND' (the third panel's set is the symmetric "
             "difference, union or intersection of the first two panels' sets). Attributes not governed by the "
             "rule are constant or irrelevant.")
REL_DEF = {"progression": "changes by a constant step along each row",
           "consistent union": "takes the same three values in every row, in some order",
           "XOR": "of the third panel is the XOR (symmetric difference) of the first two panels' sets",
           "OR": "of the third panel is the union of the first two panels' sets",
           "AND": "of the third panel is the intersection of the first two panels' sets"}


# ============================================================================ generator

def _row_values(rule, rng):
    obj, attr, rel = rule
    if rel == "progression":
        d = DOMAIN[(obj, attr)]
        lo, hi = d[0], d[-1]
        s = rng.choice([x for x in (1, -1, 2, -2) if hi - lo >= 2 * abs(x)])
        return [[a, a + s, a + 2 * s] for a in (rng.choice([x for x in d if lo <= x + 2 * s <= hi])
                                                for _ in range(3))]
    if rel == "consistent union":
        vals = rng.sample(DOMAIN[(obj, attr)], 3)
        return [rng.sample(vals, 3) for _ in range(3)]
    U, maxk, maxc = (9, 5, 9) if attr == "position" else (len(LINE_TYPES), 3, 4)
    rows = []
    for _ in range(3):
        while True:
            a = frozenset(rng.sample(range(U), rng.randint(1, maxk)))
            b = frozenset(rng.sample(range(U), rng.randint(1, maxk)))
            c = SET_OPS[rel](a, b)
            if c and a != b and len(c) <= maxc:
                break
        rows.append([a, b, c])
    return rows


def rule_value(spec, rule):
    obj, attr, _ = rule
    o = spec["shapes"] if obj == "shape" else spec["lines"]
    if o is None:
        return None
    if attr == "position":
        return frozenset(o["positions"])
    if attr == "type" and obj == "line":
        return frozenset(o["types"])
    if attr == "number":
        return len(o["positions"])
    return o[attr]


def rule_holds(rule, rows):
    rel = rule[2]
    if any(v is None for r in rows for v in r):
        return False
    if rel == "progression":
        steps = {(r[1] - r[0], r[2] - r[1]) for r in rows}
        return len(steps) == 1 and next(iter(steps))[0] == next(iter(steps))[1] != 0
    if rel == "consistent union":
        sets = {frozenset(r) for r in rows}
        return len(sets) == 1 and all(len(set(r)) == 3 for r in rows)
    return all(r[2] == SET_OPS[rel](r[0], r[1]) for r in rows)


def _set_rule_attr(spec, rule, v, rng):
    obj, attr, _ = rule
    if obj == "shape":
        sh = spec["shapes"]
        if attr == "position":
            sh["positions"] = sorted(v)
        elif attr == "number":
            sh["positions"] = sorted(rng.sample(range(9), v))
        else:
            sh[attr] = v
    else:
        if attr == "type":
            spec["lines"]["types"] = sorted(v)
        else:
            spec["lines"]["colour"] = v


def _make_panel(rule, v, base, rng, distract):
    sh = dict(base["shape"]) if base["shape"] else None
    ln = dict(base["line"]) if base["line"] else None
    if distract:      # non-rule scalar attributes vary randomly per panel
        if sh:
            sh.update(type=rng.randrange(5), colour=rng.randrange(10), size=rng.randint(1, N_SIZE))
        if ln:
            ln.update(colour=rng.randint(1, 9), types=sorted(rng.sample(range(len(LINE_TYPES)), rng.randint(1, 2))))
    if sh:            # positions vary per panel unless they are the rule attribute
        sh["positions"] = sorted(rng.sample(range(9), base["shape"]["number"]))
    spec = {"shapes": sh, "lines": ln}
    _set_rule_attr(spec, rule, v, rng)
    if sh:
        sh.pop("number", None)
    return spec


def _perturb(rule, v, rng):
    obj, attr, rel = rule
    if rel in SET_OPS:
        U = 9 if attr == "position" else len(LINE_TYPES)
        s = set(v)
        if rng.random() < 0.7:
            s ^= {rng.randrange(U)}
        else:
            s = set(rng.sample(range(U), rng.randint(1, min(U, len(v) + 1))))
        return frozenset(s) if s and frozenset(s) != v else None
    d = [x for x in DOMAIN[(obj, attr)] if x != v]
    near = [x for x in d if abs(x - v) <= 2]
    return rng.choice(near if near and rng.random() < 0.7 else d)


def _key(spec):
    return json.dumps(spec, sort_keys=True)


def generate_puzzle(rng, rule, p_other, distract, p_extra=0.3):
    obj = rule[0]
    shape = {"type": rng.randrange(5), "colour": rng.randrange(10), "size": rng.randint(1, N_SIZE),
             "number": rng.randint(1, 4)}
    line = {"types": sorted(rng.sample(range(len(LINE_TYPES)), rng.randint(1, 2))), "colour": rng.randint(1, 9)}
    if obj == "shape" and rng.random() >= p_other:
        line = None
    if obj == "line" and rng.random() >= p_other:
        shape = None
    base = {"shape": shape, "line": line}
    rows = _row_values(rule, rng)
    grid = [_make_panel(rule, rows[r][c], base, rng, distract) for r in range(3) for c in range(3)]
    answer = grid[8]
    choices, keys = [answer], {_key(answer)}
    tries = 0
    while len(choices) < 8:
        tries += 1
        if tries > 500:
            return None
        nv = _perturb(rule, rule_value(answer, rule), rng)
        if nv is None:
            continue
        cand = json.loads(json.dumps(answer))
        _set_rule_attr(cand, rule, nv, rng)
        if rng.random() < p_extra:             # sometimes also change a second attribute
            o = cand["shapes"] if cand["shapes"] else cand["lines"]
            k = rng.choice([k for k in ("colour", "size", "type") if k in o and not isinstance(o[k], list)
                            and (obj, k) != rule[:2]] or [None])
            if k == "colour":
                o[k] = rng.choice([x for x in (range(10) if cand["shapes"] is o else range(1, 10)) if x != o[k]])
            elif k == "size":
                o[k] = rng.choice([x for x in range(1, N_SIZE + 1) if x != o[k]])
            elif k == "type":
                o[k] = rng.choice([x for x in range(5) if x != o[k]])
        rows9 = [[rule_value(s, rule) for s in grid[i * 3:i * 3 + 3]] for i in range(3)]
        rows9[2][2] = rule_value(cand, rule)
        if rule_holds(rule, rows9) or _key(cand) in keys:
            continue
        keys.add(_key(cand))
        choices.append(cand)
    rows9 = [[rule_value(s, rule) for s in grid[i * 3:i * 3 + 3]] for i in range(3)]
    assert rule_holds(rule, rows9), "generated grid violates its rule"
    order = list(range(8))
    rng.shuffle(order)
    choices = [choices[i] for i in order]
    return {"rule": list(rule), "context": grid[:8], "choices": choices, "answer": LETTERS[order.index(0)]}


def grey(c):
    return int(round(255 - c * 255 / 9))


def render(spec, size=160, ss=3):
    W = size * ss
    im = Image.new("L", (W, W), 255)
    d = ImageDraw.Draw(im)
    ln = spec["lines"]
    if ln:
        g, lw = grey(ln["colour"]), 3 * ss
        for t in ln["types"]:
            name = LINE_TYPES[t]
            if name == "diagonal down":
                d.line([(0, 0), (W, W)], fill=g, width=lw)
            elif name == "diagonal up":
                d.line([(0, W), (W, 0)], fill=g, width=lw)
            elif name == "horizontal":
                d.line([(0, W // 2), (W, W // 2)], fill=g, width=lw)
            elif name == "vertical":
                d.line([(W // 2, 0), (W // 2, W)], fill=g, width=lw)
            elif name == "square frame":
                m = int(W * 0.06)
                d.rectangle([m, m, W - m, W - m], outline=g, width=lw)
            else:
                m = int(W * 0.10)
                d.ellipse([m, m, W - m, W - m], outline=g, width=lw)
    sh = spec["shapes"]
    if sh:
        cell = W / 3
        r = cell / 2 * (0.30 + 0.58 * (sh["size"] - 1) / (N_SIZE - 1))
        f, t = grey(sh["colour"]), SHAPES[sh["type"]]
        for p in sh["positions"]:
            cx, cy = (p % 3 + 0.5) * cell, (p // 3 + 0.5) * cell
            if t == "circle":
                d.ellipse([cx - r, cy - r, cx + r, cy + r], fill=f, outline=0, width=2 * ss)
            else:
                n = SIDES[t]
                off = math.pi / 4 if n == 4 else 0.0          # squares axis-aligned, others point up
                pts = [(cx + r * math.sin(2 * math.pi * k / n + off), cy - r * math.cos(2 * math.pi * k / n + off))
                       for k in range(n)]
                d.polygon(pts, fill=f, outline=0, width=2 * ss)
    return im.resize((size, size), Image.LANCZOS).convert("RGB")


def describe(spec):
    sh, ln = spec["shapes"], spec["lines"]
    if sh:
        t, n = SHAPES[sh["type"]], len(sh["positions"])
        s = (f"{n} {t}{'s' if n > 1 else ''}, color {sh['colour']}, size {sh['size']}, at "
             + ", ".join(POS[p] for p in sh["positions"]))
    else:
        s = "no shapes"
    l = (f"lines: {', '.join(LINE_TYPES[t] for t in ln['types'])}, line color {ln['colour']}" if ln else "no lines")
    return f"{s} | {l}"


def stage_synth(args):
    rng = random.Random(args.seed)
    pdir = os.path.join(args.out, "panels")
    os.makedirs(pdir, exist_ok=True)
    template = None
    if args.template_jsonl:
        ns = SimpleNamespace(root=args.data, split=os.path.basename(args.template_jsonl)[:-6], start=0, limit=1,
                             id_key=None, prompt_key=None, answer_key=None, image_key=args.image_key, rules_key=None,
                             num_shards=1, shard=0)
        tp = G.load_puzzles(ns)[0]
        lay = G.load_layout(args.layout)
        if lay["missing"] != 8:
            raise SystemExit("this generator puts the answer bottom-right; your layout's missing cell is not 8")
        template = (G.open_sheet(tp.image), lay, tp.prompt)
        os.makedirs(os.path.join(args.out, "sheets"), exist_ok=True)
    counts = defaultdict(int)
    with open(os.path.join(args.out, "puzzles.jsonl"), "w") as fh:
        i = 0
        while i < args.n:
            rule = RULES[i % len(RULES)] if args.balanced else rng.choice(RULES)
            pz = generate_puzzle(rng, rule, args.p_other, args.distract)
            if pz is None:
                continue
            pid = f"synth-{i:06d}"
            paths = []
            for k, spec in enumerate(pz["context"] + pz["choices"]):
                p = os.path.join("panels", f"{pid}_{k:02d}.png")
                render(spec, args.render_size).save(os.path.join(args.out, p))
                paths.append(p)
            pz.update(id=pid, missing=8, panels=paths, descriptions=[describe(s) for s in pz["context"] + pz["choices"]])
            if template:
                sheet, lay, prompt = template
                sheet = sheet.copy()
                boxes = [b for j, b in enumerate(lay["matrix"]) if j != 8] + list(lay["choices"])
                for p, b in zip(paths, boxes):
                    im = Image.open(os.path.join(args.out, p)).convert("RGB")
                    sheet.paste(im.resize((b[2] - b[0], b[3] - b[1]), Image.LANCZOS), (b[0], b[1]))
                sp = os.path.join("sheets", f"{pid}.png")
                sheet.save(os.path.join(args.out, sp))
                pz.update(sheet=sp, prompt=prompt)
            fh.write(json.dumps(pz) + "\n")
            counts[" ".join(rule)] += 1
            i += 1
    json.dump(vars(args), open(os.path.join(args.out, "synth_args.json"), "w"), indent=1)
    print(f"wrote {args.n} puzzles to {args.out}/puzzles.jsonl; rules: {dict(counts)}")
    if template:
        print(f"contact sheets in {args.out}/sheets/: open one and check the panels sit in the right boxes "
              "and the A-H labels are still visible")


# ============================================================================ probing

def load_synth(args):
    rows = [json.loads(l) for l in open(os.path.join(args.out, "puzzles.jsonl"))]
    return rows[:args.limit] if args.limit else rows


def panel_img(args, pz, k):
    return G.upscale(Image.open(os.path.join(args.out, pz["panels"][k])).convert("RGB"), args.panel_size)


def letter_opts(vlm, n, cache={}):
    if n not in cache:
        cache[n] = vlm.option_ids([[L, " " + L] for L in LETTERS[:n]])
    return cache[n]


def build_attr_questions(pz, seed):
    """Deterministic per puzzle. Returns dicts: kind, panels (indices 0-15), text, answer letter, n, extra."""
    rng = random.Random(int(hashlib.sha1(f"{seed}:{pz['id']}".encode()).hexdigest()[:12], 16))
    specs = pz["context"] + pz["choices"]
    qs = []
    with_sh = [k for k, s in enumerate(specs) if s["shapes"]]
    with_ln = [k for k, s in enumerate(specs) if s["lines"]]
    one = "This image is one panel of an abstract visual reasoning puzzle. "
    for k in rng.sample(with_sh, min(2, len(with_sh))):
        opts = "\n".join(f"{L}. {s}" for L, s in zip(LETTERS, SHAPES))
        qs.append(dict(kind="type", panels=[k], n=5, answer=LETTERS[specs[k]["shapes"]["type"]],
                       text=one + f"What shape is drawn in it?\n{opts}\nAnswer with a single letter."))
    for k in rng.sample(with_sh, min(2, len(with_sh))):
        n = len(specs[k]["shapes"]["positions"])
        opts = "\n".join(f"{L}. {j + 1}" for j, L in enumerate(LETTERS))
        qs.append(dict(kind="number", panels=[k], n=9, answer=LETTERS[n - 1], extra={"value": n},
                       text=one + f"How many shapes does it contain?\n{opts}\nAnswer with a single letter."))
    yn = "\nA. yes\nB. no\nAnswer with a single letter."
    for k in rng.sample(with_sh, min(1, len(with_sh))):
        occ = specs[k]["shapes"]["positions"]
        empty = [p for p in range(9) if p not in occ]
        for p, ans in [(rng.choice(occ), "A")] + ([(rng.choice(empty), "B")] if empty else []):
            qs.append(dict(kind="position", panels=[k], n=2, answer=ans,
                           text=one + f"Imagine the panel divided into a 3x3 grid. Is there a shape in the "
                                      f"{POS[p]} position?" + yn))
    for k in rng.sample(with_ln, min(1, len(with_ln))):
        have = specs[k]["lines"]["types"]
        absent = [t for t in range(len(LINE_TYPES)) if t not in have]
        for t, ans in [(rng.choice(have), "A")] + ([(rng.choice(absent), "B")] if absent else []):
            qs.append(dict(kind="line_type", panels=[k], n=2, answer=ans,
                           text=one + f"Does it contain a {LINE_TYPES[t]} line?" + yn))
    two = "These two images are panels of an abstract visual reasoning puzzle. "
    rel = [("colour_rel", "shapes", "colour", "In which image are the shapes darker (filled with a darker grey)?"),
           ("size_rel", "shapes", "size", "In which image are the shapes larger?"),
           ("line_colour_rel", "lines", "colour", "In which image are the lines darker?")]
    for kind, obj, attr, q in rel:
        pairs = [(a, b) for a in range(16) for b in range(a + 1, 16) if specs[a][obj] and specs[b][obj]
                 and specs[a][obj][attr] != specs[b][obj][attr]]
        for a, b in rng.sample(pairs, min(2, len(pairs))):
            if rng.random() < 0.5:
                a, b = b, a
            va, vb = specs[a][obj][attr], specs[b][obj][attr]
            qs.append(dict(kind=kind, panels=[a, b], n=2, answer="A" if va > vb else "B",
                           extra={"diff": abs(va - vb)},
                           text=two + q + "\nA. image 1\nB. image 2\nAnswer with a single letter."))
    for j, q in enumerate(qs):
        q["id"] = f"{pz['id']}#{q['kind']}#{j}"
        q["puzzle"] = pz["id"]
        q["rule"] = pz["rule"]
    return qs


def model_dir(args):
    d = os.path.join(args.out, G.model_tag(args.model))
    os.makedirs(d, exist_ok=True)
    return d


def stage_attrs(args):
    puzzles = load_synth(args)
    path = os.path.join(model_dir(args), "attrs.jsonl")
    done = G.done_ids(path)
    by_kind = defaultdict(list)
    for pz in puzzles:
        for q in build_attr_questions(pz, args.seed):
            if q["id"] not in done:
                by_kind[q["kind"]].append((pz, q))
    todo = sum(len(v) for v in by_kind.values())
    print(f"attrs: {todo} questions to do ({len(done)} done) from {len(puzzles)} puzzles", flush=True)
    if todo:
        vlm = G.load_model(args)
        with open(path, "a") as fh:
            for kind, items in by_kind.items():
                ids = letter_opts(vlm, items[0][1]["n"])
                for chunk in G.batched(items, args.batch_size):
                    convs, imgs = [], []
                    for pz, q in chunk:
                        if len(q["panels"]) == 1:
                            content = [{"type": "image"}, {"type": "text", "text": q["text"]}]
                        else:
                            content = [{"type": "text", "text": "Image 1:"}, {"type": "image"},
                                       {"type": "text", "text": "\nImage 2:"}, {"type": "image"},
                                       {"type": "text", "text": "\n" + q["text"]}]
                        convs.append(G.user(content))
                        imgs.append([panel_img(args, pz, k) for k in q["panels"]])
                    lp, mass = vlm.score_options(convs, imgs, args.answer_prefix, ids)
                    for (pz, q), l, m in zip(chunk, lp, mass):
                        pred = LETTERS[int(np.argmax(l))]
                        fh.write(json.dumps({k: q[k] for k in ("id", "puzzle", "kind", "rule", "answer", "n")}
                                            | {"extra": q.get("extra", {}), "pred": pred, "correct": pred == q["answer"],
                                               "option_mass": round(float(m), 4)}) + "\n")
                    fh.flush()
                print(f"  {kind}: {len(items)} questions", flush=True)
    summarize_attrs(model_dir(args))


def summarize_attrs(d):
    rows = list(G.read_merged(d, "attrs").values())
    if not rows:
        return {}
    out = {}
    for kind in sorted({r["kind"] for r in rows}):
        rs = [r for r in rows if r["kind"] == kind]
        p, lo, hi = G.wilson(sum(r["correct"] for r in rs), len(rs))
        out[kind] = {"acc": p, "lo": lo, "hi": hi, "n": len(rs), "chance": 1 / rs[0]["n"],
                     "mass": float(np.mean([r["option_mass"] for r in rs]))}
        if kind.endswith("_rel"):
            byd = defaultdict(list)
            for r in rs:
                byd[min(r["extra"]["diff"], 3)].append(r["correct"])
            out[kind]["by_diff"] = {f"{k}{'+' if k == 3 else ''}": (float(np.mean(v)), len(v)) for k, v in sorted(byd.items())}
    return out


COND_ALL = ["images", "gt_text", "images_gt_text", "images_rule", "gt_text_rule", "sheet"]


def hint_text(pz):
    obj, attr, rel = pz["rule"]
    what = {"position": "set of positions", "colour": "color",
            "type": "set of line types" if obj == "line" else "type"}.get(attr, attr)
    return f"\nHint: in every row, the {obj} {what} {REL_DEF[rel]}."


def puzzle_conv(args, pz, cond):
    if cond == "sheet":
        img = Image.open(os.path.join(args.out, pz["sheet"])).convert("RGB")
        return G.user([{"type": "image"}, {"type": "text", "text": pz["prompt"]}]), [img]
    use_img, use_txt, hint = cond.startswith("images"), "gt_text" in cond, cond.endswith("_rule")
    intro = ("This is an abstract visual reasoning puzzle: a 3x3 grid of panels, read row by row, with the "
             "bottom-right panel missing, followed by eight answer choices A to H. "
             + ("The panels are given as text descriptions. " if not use_img else "")
             + (SCALES + " " if use_txt else "") + RULE_MENU + "\n")
    content, imgs = [{"type": "text", "text": intro}], []
    labels = [f"Row {c // 3 + 1}, panel {c % 3 + 1}" for c in range(8)] + [f"Choice {L}" for L in LETTERS[:8]]
    for k, lab in enumerate(labels):
        if k == 8:
            content.append({"type": "text", "text": "\nRow 3, panel 3: ? (missing)"})
        content.append({"type": "text", "text": f"\n{lab}:" + ("" if use_img else f" {pz['descriptions'][k]}")})
        if use_img:
            content.append({"type": "image"})
            imgs.append(panel_img(args, pz, k))
            if use_txt:
                content.append({"type": "text", "text": f" ({pz['descriptions'][k]})"})
    q = (hint_text(pz) if hint else "") + "\nWhich choice completes the grid? Answer with a single letter from A to H."
    content.append({"type": "text", "text": q})
    if not use_img:     # one text block
        content = [{"type": "text", "text": "".join(c["text"] for c in content)}]
    return G.user(content), imgs


def stage_infer(args):
    puzzles = load_synth(args)
    conds = args.conditions
    if "sheet" in conds and not puzzles[0].get("sheet"):
        print("no contact sheets (run synth with --template_jsonl): skipping 'sheet'")
        conds = [c for c in conds if c != "sheet"]
    vlm = None
    d = model_dir(args)
    for cond in conds:
        path = os.path.join(d, f"infer_{cond}.jsonl")
        done = G.done_ids(path)
        todo = [p for p in puzzles if p["id"] not in done]
        print(f"infer/{cond}: {len(todo)} to do", flush=True)
        if args.dump and todo:
            conv, _ = puzzle_conv(args, todo[0], cond)
            print("---- example ----\n" + "".join(c.get("text", "<image>") for c in conv[0]["content"]) + "\n---------")
        if not todo:
            continue
        vlm = vlm or G.load_model(args)
        bs = args.batch_size if cond.startswith("gt_text") or cond == "sheet" else max(1, args.batch_size // 4)
        with open(path, "a") as fh:
            for chunk in G.batched(todo, bs):
                built = [puzzle_conv(args, p, cond) for p in chunk]
                lp, mass = vlm.score_letters([b[0] for b in built], [b[1] for b in built], args.answer_prefix)
                for p, l, m in zip(chunk, lp, mass):
                    pred = LETTERS[int(np.argmax(l))]
                    fh.write(json.dumps({"id": p["id"], "rule": p["rule"], "answer": p["answer"], "pred": pred,
                                         "correct": pred == p["answer"], "letter_mass": round(float(m), 4)}) + "\n")
                fh.flush()
        G.summarize_scoring(path)


# ============================================================================ report

def stage_report(args):
    tags = sorted(t for t in os.listdir(args.out) if os.path.isdir(os.path.join(args.out, t))
                  and t not in ("panels", "sheets"))
    L = ["# Attribute recognition vs rule inference", ""]
    fmt = lambda p, lo, hi, n: f"{p:.1%} [{lo:.1%}, {hi:.1%}] n={n}"
    for t in tags:
        d = os.path.join(args.out, t)
        L += [f"## {t}", "", "### Attribute recognition (no rules involved)", "",
              "| attribute | accuracy [95% CI] | chance | by difference (1 / 2 / 3+ levels) |", "|---|---|---|---|"]
        at = summarize_attrs(d)
        for k, v in at.items():
            bd = " / ".join(f"{a:.0%} (n={n})" for a, n in v.get("by_diff", {}).values()) if "by_diff" in v else ""
            L.append(f"| {k} | {fmt(v['acc'], v['lo'], v['hi'], v['n'])} | {v['chance']:.0%} | {bd} |")
        res = {c: list(G.read_merged(d, f"infer_{c}").values()) for c in COND_ALL}
        res = {c: r for c, r in res.items() if r}
        if res:
            L += ["", "### Puzzle accuracy by input (chance 12.5%)", "", "| condition | accuracy [95% CI] |", "|---|---|"]
            for c, rs in res.items():
                L.append(f"| {c} | {fmt(*G.wilson(sum(r['correct'] for r in rs), len(rs)), len(rs))} |")
            keys = sorted({tuple(r["rule"][:2]) for rs in res.values() for r in rs})
            L += ["", "### By rule attribute: recognising the attribute vs gain from being given it", "",
                  "| rule attribute | its recognition probe | " + " | ".join(res) + " | gain gt_text - images |",
                  "|---" * (len(res) + 3) + "|"]
            for k in keys:
                cells = {c: [r["correct"] for r in rs if tuple(r["rule"][:2]) == k] for c, rs in res.items()}
                acc = {c: (np.mean(v) if v else float("nan")) for c, v in cells.items()}
                pr = at.get(PROBE_FOR.get(k), {})
                gain = acc.get("gt_text", float("nan")) - acc.get("images", float("nan"))
                L.append(f"| {k[0]} {k[1]} | " + (f"{pr['acc']:.0%}" if pr else "-") + " | "
                         + " | ".join(f"{acc[c]:.0%} (n={len(cells[c])})" for c in res) + f" | {gain:+.0%} |")
        L.append("")
    text = "\n".join(L)
    print(text)
    open(os.path.join(args.out, "report.md"), "w").write(text + "\n")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stage", choices=["synth", "attrs", "infer", "report"])
    ap.add_argument("--out", default="attr_probe", help="folder for puzzles and results")
    ap.add_argument("--n", type=int, default=1000, help="synth: number of puzzles")
    ap.add_argument("--balanced", type=int, default=1, help="synth: equal numbers of each of the 16 rules")
    ap.add_argument("--p_other", type=float, default=0.5, help="synth: chance the non-rule object is also drawn")
    ap.add_argument("--distract", action="store_true", help="synth: non-rule attributes vary randomly per panel")
    ap.add_argument("--render_size", type=int, default=160, help="synth: panel resolution (PGM panels are 160)")
    ap.add_argument("--template_jsonl", help="synth: a real split file whose first sheet is the template")
    ap.add_argument("--data", default=os.environ.get("DATA"), help="synth: dataset root for the template sheet")
    ap.add_argument("--layout", default="layout.json")
    ap.add_argument("--image_key", default="images")
    ap.add_argument("--model", default="OpenGVLab/InternVL3_5-1B-HF")
    ap.add_argument("--conditions", nargs="+", default=["images", "gt_text", "images_gt_text", "images_rule",
                                                        "gt_text_rule", "sheet"], choices=COND_ALL)
    ap.add_argument("--limit", type=int, default=0, help="attrs/infer: first N puzzles (0 = all)")
    ap.add_argument("--panel_size", type=int, default=448, help="panels are upscaled so the long side is this")
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--answer_prefix", default="")
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--attn", default="sdpa")
    ap.add_argument("--dump", type=int, default=0, help="infer: print one example prompt per condition")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    args.causal = False
    {"synth": stage_synth, "attrs": stage_attrs, "infer": stage_infer, "report": stage_report}[args.stage](args)


if __name__ == "__main__":
    main()
