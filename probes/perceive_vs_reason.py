#!/usr/bin/env python
"""
Simple perception vs reasoning probe on real PGM puzzles.

Every puzzle is asked twice, with the same panels and the same eight choices:

  PERCEIVE  the correct answer is already shown in the bottom-right cell:
            "which choice is identical to the bottom-right panel?"
            -> needs only telling the answer apart from its 7 near-identical distractors (no rule).
  REASON    the normal puzzle, bottom-right cell empty ("?"):
            "which choice completes the grid?"
            -> needs perception AND finding/applying the rule.

Both are 8-way questions (chance 12.5%), scored from the model's probabilities for the letters A-H.
Because each puzzle gets both questions, the report shows a 2x2 table per model and the key number:

  reasoning accuracy on puzzles the model PERCEIVED correctly
  = how well it reasons when perception is not the problem.

Reading it:
  PERCEIVE high, REASON low          -> reasoning gap (it sees the differences but cannot use the rule)
  PERCEIVE low                        -> perception gap (it cannot even tell the answer from the distractors)
  REASON | perceived  >> REASON | not -> perception failures drag reasoning down

Input format (--input):
  crops  the 16 panels as separate images (cut out with layout.json), labelled by row/column and A-H
  sheet  your contact sheet; PERCEIVE pastes the answer into the "?" cell, REASON uses your dataset prompt
         (use this for models fine-tuned on contact sheets)

  python perceive_vs_reason.py run    --data $DATA --split validation --model OpenGVLab/InternVL3_5-1B-HF
  python perceive_vs_reason.py run    --data $DATA --split validation --model /path/to/finetuned --input sheet
  python perceive_vs_reason.py report
Results: pvr/<model-tag>/{perceive,reason}_<input>_<split>.jsonl (resumable) and pvr/report_<split>.md
"""
import argparse
import json
import os
import sys
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pgm_gap as G   # noqa: E402  (VLM wrapper, cropping, Wilson intervals)

LETTERS = "ABCDEFGH"
SETUP = ("This is an abstract visual reasoning puzzle: a 3x3 grid of panels, shown one panel at a time, row by row, "
         "followed by eight answer choices A to H.\n")
Q = {"perceive": "\nThe bottom-right panel of the grid is an exact copy of one of the choices. Which choice is "
                 "identical to the bottom-right panel? Answer with a single letter from A to H.",
     "reason": "\nThe bottom-right panel is missing. The panels follow rules that hold the same way in every row. "
               "Which choice completes the grid? Answer with a single letter from A to H."}
SHEET_PERCEIVE = ("This image shows a 3x3 grid of panels and eight answer choices labelled A to H. The bottom-right "
                  "cell of the grid contains an exact copy of one of the choices. Which choice is identical to it? "
                  "Answer with a single letter from A to H.")


def build(p, lay, task, mode, size):
    """Returns (message content, images) for one puzzle and task."""
    sheet = G.open_sheet(p.image)
    if mode == "sheet":
        if task == "perceive":
            return [{"type": "image"}, {"type": "text", "text": SHEET_PERCEIVE}], [G.paste_answer(sheet, lay, p.answer)]
        return [{"type": "image"}, {"type": "text", "text": p.prompt}], [sheet]
    ctx, ch = G.crop_panels(sheet, lay)
    answer_panel = ch[LETTERS.index(p.answer)]
    content, imgs, k = [{"type": "text", "text": SETUP}], [], 0
    for cell in range(9):
        label = f"\nRow {cell // 3 + 1}, panel {cell % 3 + 1}:"
        if cell == lay["missing"]:
            if task == "perceive":              # the only difference between the two tasks
                content += [{"type": "text", "text": label}, {"type": "image"}]
                imgs.append(G.upscale(answer_panel, size))
            else:
                content.append({"type": "text", "text": label + " ?"})
        else:
            content += [{"type": "text", "text": label}, {"type": "image"}]
            imgs.append(G.upscale(ctx[k], size))
            k += 1
    for L, im in zip(LETTERS, ch):
        content += [{"type": "text", "text": f"\nChoice {L}:"}, {"type": "image"}]
        imgs.append(G.upscale(im, size))
    content.append({"type": "text", "text": Q[task]})
    return content, imgs


def run(args):
    lay = G.load_layout(args.layout)
    if lay["missing"] != 8:
        raise SystemExit("expected the missing cell bottom-right (index 8) in layout.json")
    puzzles = G.load_puzzles(SimpleNamespace(root=args.data, split=args.split, start=0, limit=args.limit, id_key=None,
                                             prompt_key=None, answer_key=None, image_key=args.image_key,
                                             rules_key=None, num_shards=1, shard=0))
    d = os.path.join(args.out, G.model_tag(args.model))
    os.makedirs(d, exist_ok=True)
    vlm = None
    for task in ("perceive", "reason"):
        path = os.path.join(d, f"{task}_{args.input}_{args.split}.jsonl")
        done = G.done_ids(path)
        todo = [p for p in puzzles if p.id not in done]
        print(f"{task}/{args.input}: {len(todo)} puzzles to do ({len(done)} done)", flush=True)
        if args.dump and todo:
            os.makedirs(os.path.join(d, "debug"), exist_ok=True)
            _, imgs = build(todo[0], lay, task, args.input, args.panel_size)
            for i, im in enumerate(imgs):
                im.save(os.path.join(d, "debug", f"{task}_{args.input}_{i:02d}.png"))
        if not todo:
            continue
        vlm = vlm or G.load_model(args)
        bs = args.batch_size if args.input == "sheet" else max(1, args.batch_size // 4)
        with open(path, "a") as fh:
            for chunk in G.batched(todo, bs):
                built = [build(p, lay, task, args.input, args.panel_size) for p in chunk]
                lp, mass = vlm.score_letters([G.user(c) for c, _ in built], [im for _, im in built], args.answer_prefix)
                for p, l, m in zip(chunk, lp, mass):
                    pred = LETTERS[int(np.argmax(l))]
                    fh.write(json.dumps({"id": p.id, "answer": p.answer, "pred": pred, "correct": pred == p.answer,
                                         "letter_mass": round(float(m), 4)}) + "\n")
                fh.flush()
    print(summary(d, args.input, args.split)[1])


def summary(d, mode, split):
    P = G.read_merged(d, f"perceive_{mode}_{split}")
    R = G.read_merged(d, f"reason_{mode}_{split}")
    ids = sorted(set(P) & set(R))
    if not ids:
        return None, f"{d}: no paired results for input '{mode}' yet"
    p = np.array([P[i]["correct"] for i in ids])
    r = np.array([R[i]["correct"] for i in ids])
    ci = lambda x: G.wilson(int(x.sum()), len(x)) if len(x) else (float("nan"),) * 3
    fmt = lambda x: f"{ci(x)[0]:.1%} [{ci(x)[1]:.1%}, {ci(x)[2]:.1%}] n={len(x)}" if len(x) else "– (n=0)"
    mass = np.mean([P[i]["letter_mass"] for i in ids] + [R[i]["letter_mass"] for i in ids])
    res = {"n": len(ids), "perceive": float(p.mean()), "reason": float(r.mean()),
           "reason_given_perceived": float(r[p].mean()) if p.any() else None,
           "reason_given_not_perceived": float(r[~p].mean()) if (~p).any() else None,
           "table": {"both": int((p & r).sum()), "perceive_only": int((p & ~r).sum()),
                     "reason_only": int((~p & r).sum()), "neither": int((~p & ~r).sum())}}
    t = res["table"]
    text = "\n".join([
        f"### {os.path.basename(d)} ({mode} input, {len(ids)} puzzles, chance 12.5%)", "",
        "| measure | accuracy [95% CI] |", "|---|---|",
        f"| PERCEIVE (answer shown, find the identical choice) | {fmt(p)} |",
        f"| REASON (normal puzzle) | {fmt(r)} |",
        f"| REASON on puzzles it perceived correctly | {fmt(r[p])} |",
        f"| REASON on puzzles it did not perceive | {fmt(r[~p])} |", "",
        "| | reasoned right | reasoned wrong |", "|---|---|---|",
        f"| perceived right | {t['both']} | {t['perceive_only']} |",
        f"| perceived wrong | {t['reason_only']} | {t['neither']} |", "",
        f"Mean probability on the letters A-H: {mass:.2f}"
        + ("  (low: the model may want to answer in another format; try --answer_prefix 'Answer: ')" if mass < 0.5 else ""),
        ""])
    return res, text


def report(args):
    parts = [f"# Perception vs reasoning ({args.split})", ""]
    results = {}
    for tag in sorted(os.listdir(args.out)):
        d = os.path.join(args.out, tag)
        if not os.path.isdir(d):
            continue
        for mode in ("crops", "sheet"):
            res, text = summary(d, mode, args.split)
            if res:
                parts.append(text)
                results[f"{tag}/{mode}"] = res
    text = "\n".join(parts)
    print(text)
    open(os.path.join(args.out, f"report_{args.split}.md"), "w").write(text + "\n")
    json.dump(results, open(os.path.join(args.out, f"report_{args.split}.json"), "w"), indent=1)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stage", choices=["run", "report"])
    ap.add_argument("--data", default=os.environ.get("DATA"), help="dataset root (default $DATA)")
    ap.add_argument("--split", default="validation")
    ap.add_argument("--layout", default="layout.json")
    ap.add_argument("--model", default="OpenGVLab/InternVL3_5-1B-HF")
    ap.add_argument("--input", choices=["crops", "sheet"], default="crops")
    ap.add_argument("--out", default="pvr")
    ap.add_argument("--limit", type=int, default=0, help="first N puzzles (0 = all)")
    ap.add_argument("--image_key", default="images")
    ap.add_argument("--panel_size", type=int, default=448, help="crops: panels upscaled to this size")
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--answer_prefix", default="")
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--attn", default="sdpa")
    ap.add_argument("--dump", action="store_true", help="save the images of the first puzzle to <out>/<tag>/debug")
    args = ap.parse_args()
    args.causal = False
    if args.stage == "run":
        if not args.data:
            ap.error("give the dataset root with --data $DATA (or export DATA=...)")
        run(args)
    else:
        report(args)


if __name__ == "__main__":
    main()
