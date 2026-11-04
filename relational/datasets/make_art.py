"""ART (Abstract Reasoning Tasks; Webb et al. 2021, used by OCRA and Slot Abstractors) -> packed jsonl/zip.

Faithful to the official code:
  * shapes: the 100 `resizedcropped<i>.npy` unicode glyphs (32x32) from github.com/Shanka123/Slot-Abstractor/art/imgs
  * trials: the original ESBN task generators (github.com/taylorwwebb/emergent_symbols/tasks), n=100 shapes, m=95 held
    out: train uses 5 shapes (SD 40 / RMTS 480 / Dist3 360 / ID 8640 trials), test the other 95 (10,000 trials each;
    identity rules and RMTS tests use ESBN's 'subsample' mode: the paper's script does this for identity rules, and
    'full_space' enumerates every 95-shape RMTS trial first, which needs far more than 64 GB)
  * layout: 160x160 white canvas, glyph centres as in Slot-Abstractor's train_slot_abstractor_{sd,rmts,dist3}.py, each
    glyph jittered by U{-5..4} px
Our format (per-candidate, like PGM): every candidate gets its own completed image and a Yes/No question.
  sd      binary   two glyphs side by side: "are they the same?"
  rmts    cand x2  source pair on top, candidate pair below
  dist3   cand x4  top row A B C, bottom row two glyphs + candidate
  idrules cand x4  top row shows ABA / AAB / ABB / AAA, bottom row two glyphs + candidate
Panel boxes = one box per glyph cell. Labels are recomputed from the shape indices and checked against ESBN's y.
Train sets are tiny, so each train trial is rendered `--train_copies` times with fresh jitter.

  python datasets/make_art.py [--scale 2] [--train_copies auto]
"""
import argparse
import importlib
import os
import sys
import types
import urllib.request

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from packing import ROOT, Packer, boxes_from_centers

SHAPES_URL = "https://raw.githubusercontent.com/Shanka123/Slot-Abstractor/HEAD/art/imgs/resizedcropped{}.npy"
ESBN_URL = "https://raw.githubusercontent.com/taylorwwebb/emergent_symbols/HEAD/tasks/{}.py"
TASKS = {  # name -> (ESBN module, glyph centres (row, col) on the 160 canvas: context..., then the candidate slot(s))
    "sd": ("same_diff", [(80, 40), (80, 120)]),
    "rmts": ("RMTS", [(40, 40), (40, 120), (120, 40), (120, 120)]),
    "dist3": ("dist3", [(40, 40), (40, 88), (40, 136), (120, 40), (120, 88), (120, 136)]),
    "idrules": ("identity_rules", [(40, 40), (40, 88), (40, 136), (120, 40), (120, 88), (120, 136)]),
}
PROMPTS = {
    "sd": "The image shows two shapes. Are the two shapes the same? Answer Yes or No.",
    "rmts": ("The top two shapes are either the same as each other or different from each other. Do the bottom two "
             "shapes have the same relation (same or different) as the top two shapes? Answer Yes or No."),
    "dist3": ("The top row shows three different shapes. Does the bottom row contain the same three shapes, in some "
              "order? Answer Yes or No."),
    "idrules": ("The top row of three shapes follows an identity pattern (for example: first and third the same, last "
                "two the same, or all three the same). Does the bottom row follow the same identity pattern as the top "
                "row? Answer Yes or No."),
}


def fetch(src_dir):
    os.makedirs(src_dir, exist_ok=True)
    shapes_path = os.path.join(src_dir, "shapes.npy")
    if not os.path.exists(shapes_path):
        tmp = os.path.join(src_dir, "_glyph.npy")
        glyphs = []
        for i in range(100):
            urllib.request.urlretrieve(SHAPES_URL.format(i), tmp)
            glyphs.append(np.load(tmp))
        os.remove(tmp)
        np.save(shapes_path, np.stack(glyphs))
    for mod in {m for m, _ in TASKS.values()}:
        p = os.path.join(src_dir, f"{mod}.py")
        if not os.path.exists(p):
            urllib.request.urlretrieve(ESBN_URL.format(mod), p)
    return np.load(shapes_path)


def esbn_module(src_dir, name):
    """Import an original ESBN task generator (it needs `util.log` and numpy's removed np.int alias)."""
    if "util" not in sys.modules:
        util = types.ModuleType("util")
        util.log = types.SimpleNamespace(info=lambda *a, **k: None)
        sys.modules["util"] = util
    if not hasattr(np, "int") or np.int is not int:
        np.int = int
    sys.path.insert(0, src_dir)
    try:
        return importlib.import_module(name)
    finally:
        sys.path.remove(src_dir)


def answer_of(task, s):
    """Recompute the correct answer from shape indices (independent check of ESBN's labels)."""
    if task == "sd":
        return int(s[0] == s[1])                       # 1 = same
    if task == "rmts":
        src = s[0] == s[1]
        return [k for k in range(2) if (s[2 + 2 * k] == s[3 + 2 * k]) == src]
    top = s[:3]
    if task == "dist3":
        return [k for k in range(4) if sorted([s[3], s[4], s[5 + k]]) == sorted(top)]
    pat = lambda a, b, c: (a == b, a == c, b == c)
    return [k for k in range(4) if pat(s[3], s[4], s[5 + k]) == pat(*top)]


def render(glyphs, idx, centres, scale, rng):
    S = 160 * scale
    canvas = Image.new("L", (S, S), 255)
    for g, (r, c) in zip(idx, centres):
        im = Image.fromarray(glyphs[g].astype(np.uint8)).resize((32 * scale, 32 * scale), Image.BICUBIC)
        dr, dc = rng.integers(-5, 5, 2) * scale
        canvas.paste(im, (int(c * scale + dc - 16 * scale), int(r * scale + dr - 16 * scale)))
    return canvas


def build_split(task, glyphs, trials, out_dir, split, scale, copies, seed):
    mod_centres = TASKS[task][1]
    S = 160 * scale
    boxes = boxes_from_centers([(r * scale, c * scale) for r, c in mod_centres], 24 * scale, S, S)
    rng = np.random.default_rng(seed)
    pk = Packer(out_dir, split)
    n = 0
    for copy in range(copies):
        for i, s in enumerate(trials):
            s = [int(v) for v in s]
            rid = f"art/{task}/{split}/{i}" + (f"/c{copy}" if copies > 1 else "")
            if task == "sd":
                ref = pk.add_image(f"{n:07d}.png", render(glyphs, s, mod_centres, scale, rng))
                pk.add_row({"id": rid, "images": [ref], "prompt": PROMPTS[task], "panels": boxes,
                            "answer": "Yes" if answer_of(task, s) else "No"})
            else:
                n_ctx = len(mod_centres) - (2 if task == "rmts" else 1)
                ctx = s[:n_ctx]
                n_cand = 2 if task == "rmts" else 4
                cands = []
                for k in range(n_cand):
                    cand = s[n_ctx + 2 * k: n_ctx + 2 * k + 2] if task == "rmts" else [s[n_ctx + k]]
                    cands.append(pk.add_image(f"{n:07d}_{k}.png", render(glyphs, ctx + cand, mod_centres, scale, rng)))
                ans = answer_of(task, s)
                assert len(ans) == 1, (task, s, ans)
                pk.add_row({"id": rid, "candidates": cands, "prompt": PROMPTS[task], "panels": boxes, "answer": ans[0]})
            n += 1
    return pk.close({"task": task, "trials": len(trials), "copies": copies, "scale": scale})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(ROOT, "art"))
    ap.add_argument("--tasks", default="sd,rmts,dist3,idrules")
    ap.add_argument("--scale", type=int, default=2, help="render at 160*scale px (bigger glyphs = more ViT patches)")
    ap.add_argument("--min_train", type=int, default=4000, help="render train trials repeatedly up to this many rows")
    ap.add_argument("--val", type=int, default=500)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    glyphs = fetch(os.path.join(args.out, "_src"))
    for task in args.tasks.split(","):
        mod = esbn_module(os.path.join(args.out, "_src"), TASKS[task][0])
        np.random.seed(args.seed)                                    # ESBN uses the global numpy / random state
        import random
        random.seed(args.seed)
        shapes = np.arange(100)
        np.random.shuffle(shapes)
        train_shapes, test_shapes = shapes[95:], shapes[:95]
        g = types.SimpleNamespace(n_shapes=100, m_holdout=95, train_set_size=10000, test_set_size=10000,
                                  train_proportion=0.95, train_gen_method="full_space",
                                  test_gen_method="subsample" if task in ("idrules", "rmts") else "full_space")
        _, tr, te = mod.create_task(g, train_shapes, test_shapes)
        for name, d in (("train", tr), ("test", te)):                # ESBN's labels must agree with ours
            for s, y in zip(d["seq_ind"], d["y"]):
                a = answer_of(task, [int(v) for v in s])
                a = a if task == "sd" else a[0]
                if task == "sd":
                    assert int(y) in (0, 1)
                else:
                    assert int(y) == a, (task, name, s, y, a)
        if task == "sd":    # ESBN's sd label convention, recorded for reference
            print(f"[art] sd: ESBN y==1 means {'same' if all((int(y) == 1) == (s[0] == s[1]) for s, y in zip(tr['seq_ind'], tr['y'])) else 'different'}")
        out = os.path.join(args.out, task)
        copies = max(1, int(np.ceil(args.min_train / len(tr["seq_ind"]))))
        print(f"[art] {task}: train {len(tr['seq_ind'])} trials x{copies}, test {len(te['seq_ind'])}", flush=True)
        build_split(task, glyphs, tr["seq_ind"], out, "train", args.scale, copies, args.seed + 1)
        vi = np.random.default_rng(args.seed).choice(len(tr["seq_ind"]), min(args.val, len(tr["seq_ind"])), replace=False)
        build_split(task, glyphs, tr["seq_ind"][vi], out, "val", args.scale, 1, args.seed + 2)
        build_split(task, glyphs, te["seq_ind"], out, "test", args.scale, 1, args.seed + 3)


if __name__ == "__main__":
    main()
