"""CLEVR-ART OOD test package (~/scratch/clevr_art_data/clevr-art-ood-test-seed1-v1) -> two relvlm formats.

  <out>/{rmts,idrules}_sheet.jsonl  letter: the package's own composite image + exact prompt (comparable with the
                                    zero-shot protocol); images referenced in place, nothing copied
  <out>/{rmts,idrules}_cand.{jsonl,zip}  cand: each 320x240 candidate scene cropped out of the sheet, one Yes/No question
                                    per scene (the same per-candidate format as PGM / ART and the rendered train split)
Scene rectangles are found from the sheet itself (scenes are the large non-white blocks); every sheet must yield
exactly 2 (RMTS) or 4 (identity rules) blocks of the expected source size, otherwise the script stops.
Labels come from the package manifests; they are cross-checked with the official symbolic npz (y) via metadata.
"""
import argparse
import json
import os
import sys

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from packing import ROOT, Packer

PKG = os.path.expanduser("~/scratch/clevr_art_data/clevr-art-ood-test-seed1-v1")
CAND_PROMPT = {
    "rmts": ("The scene shows a source pair of objects in the back row and a candidate pair in the front row. Object "
             "identity is determined by color, shape, size, and material. Does the front pair have the same "
             "same-or-different relation as the back pair? Answer Yes or No."),
    "idrules": ("The scene shows two rows of three objects. The back row demonstrates an identity pattern (for example "
                "first equals third, last two equal, or all three equal). Object identity is determined by color, "
                "shape, size, and material. Does the front row follow the same identity pattern as the back row? "
                "Answer Yes or No."),
}


def runs(mask_1d, min_len):
    out, start = [], None
    for i, v in enumerate(list(mask_1d) + [False]):
        if v and start is None:
            start = i
        elif not v and start is not None:
            if i - start >= min_len:
                out.append((start, i))
            start = None
    return out


def scene_boxes(img, w=320, h=240, tol=6):
    """Pixel boxes (x0, y0, x1, y1) of the rendered scenes, in reading order."""
    a = np.asarray(img.convert("L")).astype(np.int32)
    nonwhite = a < 235
    boxes = []
    for y0, y1 in runs(nonwhite.sum(1) > 150, h - tol):                      # scene rows
        for x0, x1 in runs(nonwhite[y0:y1].mean(0) > 0.9, w - tol):          # scene columns inside that band
            boxes.append((x0, y0, x1, y1))
    return boxes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pkg", default=PKG)
    ap.add_argument("--out", default=os.path.join(ROOT, "clevr_art_ood"))
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    for task, n_c in (("rmts", 2), ("idrules", 4)):
        man = os.path.join(args.pkg, f"clevr-art-{task}-test-full.jsonl")
        rows = [json.loads(l) for l in open(man)]
        with open(os.path.join(args.out, f"{task}_sheet.jsonl"), "w") as f:
            for r in rows:
                f.write(json.dumps({"id": r["id"], "images": [os.path.join(args.pkg, r["images"][0])], "prompt": r["prompt"],
                                    "choices": r["choices"], "answer": r["answer"]}) + "\n")
        # The layout is fixed per task (constant sheet size): take the scene boxes most sheets agree on, then crop
        # every sheet with them (a very bright scene can defeat the per-sheet detector).
        votes = {}
        for r in rows[:200]:
            bx = tuple(scene_boxes(Image.open(os.path.join(args.pkg, r["images"][0]))))
            if len(bx) == n_c:
                votes[bx] = votes.get(bx, 0) + 1
        layout, n_agree = max(votes.items(), key=lambda kv: kv[1])
        sheet_size = Image.open(os.path.join(args.pkg, rows[0]["images"][0])).size
        print(f"[clevr_art_ood] {task}: layout {layout} (agreed by {n_agree}/200 sheets), sheet {sheet_size}", flush=True)
        assert n_agree >= 150, votes
        pk = Packer(args.out, f"{task}_cand")
        for i, r in enumerate(rows):
            img = Image.open(os.path.join(args.pkg, r["images"][0])).convert("RGB")
            assert img.size == sheet_size, (r["id"], img.size)
            cands = [pk.add_image(f"{i:05d}_{k}.png", img.crop(b)) for k, b in enumerate(layout)]
            pk.add_row({"id": r["id"], "candidates": cands, "prompt": CAND_PROMPT[task],
                        "answer": r["choices"].index(r["answer"])})
        pk.close({"source": man, "scene_size": [320, 240]})


if __name__ == "__main__":
    main()
