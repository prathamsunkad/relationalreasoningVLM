#!/usr/bin/env python
"""
Attribute perception vs rule inference on REAL PGM puzzles.

Real PGM stores no per-panel attribute values, but every puzzle comes with its rules
(`relation_structure`: triples like  shape / color / progression) and the answer. This script
uses only that ground truth, so nothing is guessed from pixels:

  rules    Get each puzzle's rule triples. Read from your JSONL if they are there; otherwise fetch
           only the id / relation_structure / target columns of the pinned HuggingFaceM4/PGM revision
           (a few MB, not the 44 GB of images) and join them to your puzzles. The join is checked: the
           PGM answer must equal your JSONL answer for >= 98% of puzzles, or the stage stops.
  order    ATTRIBUTE PERCEPTION. A progression rule guarantees that, in every row, the governed attribute
           changes step by step, so the middle panel of a row has the intermediate value. The three panels
           of a row are shown in random order: "which image has the intermediate darkness / size / number of
           shapes / number of sides / line darkness?" (chance 33%). No rule has to be found; the model only
           has to read and compare one attribute.
  ruleid   RULE IDENTIFICATION on the completed grid (the answer filled in), puzzles with one rule:
             attribute question: which attribute follows a pattern?             (7 options, chance 14%)
             relation question:  given that attribute, which relation is it?    (5 options, chance 20%)
  infer    SOLVING with and without being told the rule (chance 12.5%):
             crops        the 16 panels as images (as in pgm_gap.py image --variant crops)
             crops_rule   the same + the puzzle's rule(s)
             sheet        your contact sheet and dataset prompt (what the fine-tuned models were trained on)
             sheet_rule   the same + the rule(s)
  report   Tables per model, and per rule attribute: perception of that attribute vs solving vs the gain
           from being told the rule.

How the pieces separate the hypothesis (attributes must be learned before rules can be induced):
  order low for an attribute            -> the model cannot read/compare that attribute in PGM panels
  ruleid-attribute low, relation high    -> it cannot tell WHICH attribute varies, but knows the relation types
  crops_rule ~ crops (both low)          -> knowing the rule does not help: it cannot read the values
  crops_rule >> crops                    -> it can read the values; finding the rule was the problem
The complementary condition "given ground-truth attribute values" needs per-panel labels, which PGM lacks;
pgm_attr_probe.py provides it on generated PGM-style puzzles.

  python pgm_real_probe.py rules  --data $DATA --split validation
  python pgm_real_probe.py order  --data $DATA --split validation --model OpenGVLab/InternVL3_5-1B-HF
  python pgm_real_probe.py ruleid --data $DATA --split validation --model OpenGVLab/InternVL3_5-1B-HF
  python pgm_real_probe.py infer  --data $DATA --split validation --model OpenGVLab/InternVL3_5-1B-HF
  python pgm_real_probe.py report --out real_probe
Results go to <out>/<model-tag>/ ; every stage resumes where it stopped.
"""
import argparse
import hashlib
import json
import os
import random
import re
import sys
from collections import Counter, defaultdict
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pgm_gap as G   # noqa: E402

LETTERS = "ABCDEFGH"
HF_REPO, HF_REV = "HuggingFaceM4/PGM", "195a03a7e3bc6d6a349b55986afba5319929c3bf"
ATTR_OPTS = [("shape", "type"), ("shape", "color"), ("shape", "size"), ("shape", "number"), ("shape", "position"),
             ("line", "type"), ("line", "color")]
ATTR_TEXT = {("shape", "type"): "the type of the shapes", ("shape", "color"): "the color (darkness) of the shapes",
             ("shape", "size"): "the size of the shapes", ("shape", "number"): "the number of shapes",
             ("shape", "position"): "the positions of the shapes", ("line", "type"): "the types of the lines",
             ("line", "color"): "the color (darkness) of the lines"}
RELS = ["progression", "XOR", "OR", "AND", "consistent union"]
REL_TEXT = {"progression": "progression: the value increases or decreases step by step along the row",
            "XOR": "XOR: the third panel contains what is in exactly one of the first two panels",
            "OR": "OR: the third panel contains what is in either of the first two panels",
            "AND": "AND: the third panel contains what is in both of the first two panels",
            "consistent union": "consistent union: every row contains the same three values, in some order"}
ORDER_PHRASE = {("shape", "color"): "shapes of intermediate darkness (neither the lightest nor the darkest fill)",
                ("shape", "size"): "shapes of intermediate size (neither the smallest nor the largest)",
                ("shape", "number"): "the intermediate number of shapes (neither the fewest nor the most)",
                ("shape", "type"): "shapes with the intermediate number of sides (neither the fewest nor the most)",
                ("line", "color"): "lines of intermediate darkness (neither the lightest nor the darkest)"}


# ============================================================================ data

def ns(args, split=None, limit=None):
    return SimpleNamespace(root=args.data, split=split or args.split, start=0,
                           limit=args.limit if limit is None else limit, id_key=None, prompt_key=None,
                           answer_key=None, image_key=args.image_key, rules_key=args.rules_key, num_shards=1, shard=0)


def norm_triple(t):
    clean = lambda x: re.sub(r"^b['\"](.*)['\"]$", r"\1", str(x).strip()).strip().lower().replace("_", " ")
    o, a, r = (clean(x) for x in t)
    o = {"shapes": "shape", "lines": "line"}.get(o, o)
    a = {"colour": "color"}.get(a, a)
    r = {"xor": "XOR", "or": "OR", "and": "AND"}.get(r, r)
    return [o, a, r]


def parse_rule_string(s):
    """pgm_gap.find_rules output, e.g. "shapes color progression; line type XOR"."""
    out = []
    for part in s.split(";"):
        toks = part.split()
        if len(toks) >= 3:
            out.append(norm_triple([toks[0], toks[1], " ".join(toks[2:])]))
    return out


def rules_path(args, split=None):
    return os.path.join(args.out, f"rules_{split or args.split}.json")


def load_rules(args, split=None):
    p = rules_path(args, split)
    if not os.path.exists(p):
        raise SystemExit(f"no {p}: run `python pgm_real_probe.py rules --data $DATA --split {split or args.split}` first")
    return json.load(open(p))


def int_fields(row, prefix=""):
    """All integer-like values in a JSONL row (nested), as {dotted path: int}; plus the trailing number of the id."""
    out = {}
    if isinstance(row, dict):
        for k, v in row.items():
            path = f"{prefix}{k}"
            if isinstance(v, (dict, list)) and not (isinstance(v, list) and len(v) > 50):
                out.update(int_fields(v, path + "."))
            elif isinstance(v, bool):
                continue
            elif isinstance(v, int):
                out[path] = v
            elif isinstance(v, str) and re.fullmatch(r"\d+", v):
                out[path] = int(v)
            if k in ("id", "question_id", "qid", "uid") and isinstance(v, str) and re.search(r"(\d+)$", v):
                out[path + "#trailing_number"] = int(re.search(r"(\d+)$", v).group(1))
    elif isinstance(row, list):
        for i, v in enumerate(row[:50]):
            if isinstance(v, (dict, list)):
                out.update(int_fields(v, f"{prefix}{i}."))
    return out


def fetch_hf_columns(split, cols=("id", "relation_structure", "target")):
    """Read only the needed columns of the pinned PGM revision (range requests; no images downloaded)."""
    from huggingface_hub import HfFileSystem
    import pyarrow.parquet as pq
    fs = HfFileSystem()
    root = f"datasets/{HF_REPO}@{HF_REV}"
    files = sorted(f for f in fs.find(root) if f.endswith(".parquet")
                   and (os.path.basename(f).startswith(split + "-") or f"/{split}/" in f or f"/{split}-" in f))
    if not files:
        raise SystemExit(f"no parquet files for split '{split}' in {root}; files: "
                         f"{[f for f in fs.find(root) if f.endswith('.parquet')][:10]}")
    rows = []
    for i, f in enumerate(files):
        with fs.open(f, "rb") as fh:
            rows.extend(pq.ParquetFile(fh).read(columns=list(cols)).to_pylist())
        print(f"  {split}: {i + 1}/{len(files)} files, {len(rows)} rows", flush=True)
    return rows


def triples_from_hf(rs):
    if rs is None:
        return []
    if isinstance(rs, (bytes, str)):
        return parse_rule_string(rs.decode() if isinstance(rs, bytes) else rs)
    out = []
    for t in rs:
        t = [x.decode() if isinstance(x, bytes) else x for x in t]
        if len(t) == 3 and all(str(x).strip() for x in t):
            out.append(norm_triple(t))
    return out


def stage_rules(args):
    os.makedirs(args.out, exist_ok=True)
    puzzles = G.load_puzzles(ns(args, limit=0))
    have = {p.id: parse_rule_string(p.rules) for p in puzzles if p.rules}
    if len(have) == len(puzzles):
        res = {"source": "jsonl", "rules": have}
        print(f"rules for all {len(puzzles)} puzzles found in {args.split}.jsonl")
    else:
        print(f"{args.split}.jsonl has rules for {len(have)}/{len(puzzles)} puzzles; fetching them from "
              f"{HF_REPO}@{HF_REV[:8]} (columns id, relation_structure, target only)", flush=True)
        hf = fetch_hf_columns(args.hf_split or args.split)
        by_id = {r["id"]: r for r in hf}
        by_pos = dict(enumerate(hf))
        raw = []
        with open(os.path.join(args.data, f"{args.split}.jsonl")) as fh:
            for line in fh:
                if line.strip():
                    raw.append(json.loads(line))
        ans = {p.id: LETTERS.index(p.answer) for p in puzzles}
        pid_of = lambda row: str(row.get("id", row.get("question_id", "")))
        sample = raw[:500]
        cands = Counter(k for row in sample for k in int_fields(row))
        best = None
        for field in [k for k, n in cands.items() if n == len(sample)]:
            for name, table in (("id", by_id), ("position", by_pos)):
                hits = [table.get(int_fields(row).get(field)) for row in sample]
                agree = np.mean([h is not None and int(h["target"]) == ans.get(pid_of(row), -1)
                                 for h, row in zip(hits, sample)])
                if best is None or agree > best[0]:
                    best = (agree, field, name)
        if best is None or best[0] < 0.98:
            raise SystemExit(f"could not join your puzzles to PGM: best match {best}. Tell me which JSONL field holds "
                             "the original PGM question id (e.g. source.row_index).")
        agree, field, name = best
        table = by_id if name == "id" else by_pos
        rules, bad = {}, 0
        for row in raw:
            h = table.get(int_fields(row).get(field))
            pid = pid_of(row)
            if h is None or int(h["target"]) != ans.get(pid, -1):
                bad += 1
                continue
            rules[pid] = triples_from_hf(h["relation_structure"])
        print(f"joined on JSONL field '{field}' = PGM {name} (answer agreement {agree:.1%} on the first 500); "
              f"{len(rules)} puzzles with rules, {bad} skipped (no match or answer mismatch)")
        res = {"source": f"hf:{HF_REPO}@{HF_REV}", "join": [field, name], "rules": rules}
    json.dump(res, open(rules_path(args), "w"))
    c = Counter(" ".join(t) for r in res["rules"].values() for t in r)
    n = Counter(len(r) for r in res["rules"].values())
    print(f"wrote {rules_path(args)}\nrules per puzzle: {dict(sorted(n.items()))}\nrule types: {dict(c.most_common())}")


def puzzles_with_rules(args):
    rules = load_rules(args)["rules"]
    return [(p, rules[p.id]) for p in G.load_puzzles(ns(args)) if p.id in rules]


def grid_panels(p, lay):
    """The completed 3x3 grid (answer filled in) as 9 crops, plus the 8 context and 8 choice crops."""
    ctx, ch = G.crop_panels(G.open_sheet(p.image), lay)
    ans = ch[LETTERS.index(p.answer)]
    m = lay["missing"]
    cells = [ctx[c] if c < m else (ans if c == m else ctx[c - 1]) for c in range(9)]
    return cells, ctx, ch


def rng_for(args, key):
    return random.Random(int(hashlib.sha1(f"{args.seed}:{key}".encode()).hexdigest()[:12], 16))


def model_dir(args):
    d = os.path.join(args.out, G.model_tag(args.model))
    os.makedirs(d, exist_ok=True)
    return d


def letter_ids(vlm, n, cache={}):
    if n not in cache:
        cache[n] = vlm.option_ids([[L, " " + L] for L in "ABCDEFGHI"[:n]])
    return cache[n]


def run_questions(args, stem, items, n_options):
    """items: list of (record, content, images). Scores options A.. by log-probability; resumes by id."""
    path = os.path.join(model_dir(args), f"{stem}.jsonl")
    done = G.done_ids(path)
    todo = [it for it in items if it[0]["id"] not in done]
    print(f"{stem}: {len(todo)} questions to do ({len(done)} done)", flush=True)
    if not todo:
        return path
    vlm = G.load_model(args)
    ids = letter_ids(vlm, n_options)
    with open(path, "a") as fh:
        for chunk in G.batched(todo, args.batch_size):
            lp, mass = vlm.score_options([G.user(c) for _, c, _ in chunk], [im for _, _, im in chunk],
                                         args.answer_prefix, ids)
            for (rec, _, _), l, m in zip(chunk, lp, mass):
                pred = "ABCDEFGHI"[int(np.argmax(l))]
                fh.write(json.dumps(dict(rec, pred=pred, correct=pred == rec["answer"],
                                         option_mass=round(float(m), 4))) + "\n")
            fh.flush()
    return path


# ============================================================================ stages

def stage_order(args):
    lay = G.load_layout(args.layout)
    items = []
    for p, rules in puzzles_with_rules(args):
        prog = [tuple(t[:2]) for t in rules if t[2] == "progression" and tuple(t[:2]) in ORDER_PHRASE]
        if not prog:
            continue
        cells, _, _ = grid_panels(p, lay)
        for oa in prog:
            for row in (0, 1, 2)[:args.rows]:
                rng = rng_for(args, f"{p.id}:{oa}:{row}")
                order = [0, 1, 2]
                rng.shuffle(order)          # order[i] = which column is shown as image i+1
                imgs = [G.upscale(cells[row * 3 + c], args.panel_size) for c in order]
                content = []
                for i in range(3):
                    content += [{"type": "text", "text": f"{'' if i == 0 else chr(10)}Image {i + 1}:"}, {"type": "image"}]
                content.append({"type": "text", "text": "\nThese three images are panels of an abstract visual reasoning "
                                f"puzzle. Which image shows {ORDER_PHRASE[oa]}?\nA. image 1\nB. image 2\nC. image 3\n"
                                "Answer with a single letter."})
                rec = {"id": f"{p.id}#{oa[0]}-{oa[1]}#row{row + 1}", "puzzle": p.id, "attr": list(oa),
                       "row": row + 1, "answer": "ABC"[order.index(1)], "n_rules": len(rules)}
                items.append((rec, content, imgs))
    print(f"order: {len(items)} questions from progression rules on {sorted({tuple(r['attr']) for r, _, _ in items})}")
    run_questions(args, f"order_{args.split}", items, 3)
    summarize(model_dir(args), f"order_{args.split}", "attr")


def stage_ruleid(args):
    lay = G.load_layout(args.layout)
    a_items, r_items = [], []
    for p, rules in puzzles_with_rules(args):
        if len(rules) != 1 or tuple(rules[0][:2]) not in ATTR_OPTS or rules[0][2] not in RELS:
            continue
        o, a, rel = rules[0]
        cells, _, _ = grid_panels(p, lay)
        grid = []
        imgs = [G.upscale(c, args.panel_size) for c in cells]
        for c in range(9):
            grid += [{"type": "text", "text": f"\nRow {c // 3 + 1}, panel {c % 3 + 1}:"}, {"type": "image"}]
        intro = [{"type": "text", "text": "This is a completed abstract visual reasoning puzzle: a 3x3 grid of panels, "
                  "shown one panel at a time, row by row. One attribute follows the same rule in every row."}]
        opts = "\n".join(f"{L}. {ATTR_TEXT[oa]}" for L, oa in zip(LETTERS, ATTR_OPTS))
        a_items.append(({"id": f"{p.id}#attribute", "puzzle": p.id, "attr": [o, a], "relation": rel,
                         "answer": LETTERS[ATTR_OPTS.index((o, a))]},
                        intro + grid + [{"type": "text", "text": "\nWhich attribute follows a rule across the rows?\n"
                                         f"{opts}\nAnswer with a single letter."}], imgs))
        ropts = "\n".join(f"{L}. {REL_TEXT[r]}" for L, r in zip(LETTERS, RELS))
        r_items.append(({"id": f"{p.id}#relation", "puzzle": p.id, "attr": [o, a], "relation": rel,
                         "answer": LETTERS[RELS.index(rel)]},
                        intro + grid + [{"type": "text", "text": f"\nIn every row, {ATTR_TEXT[(o, a)]} follows a rule. "
                                         f"Which rule?\n{ropts}\nAnswer with a single letter."}], imgs))
    print(f"ruleid: {len(a_items)} single-rule puzzles")
    bs = args.batch_size
    args.batch_size = max(1, bs // 4)
    run_questions(args, f"ruleid_attribute_{args.split}", a_items, len(ATTR_OPTS))
    run_questions(args, f"ruleid_relation_{args.split}", r_items, len(RELS))
    args.batch_size = bs
    summarize(model_dir(args), f"ruleid_attribute_{args.split}", "attr")
    summarize(model_dir(args), f"ruleid_relation_{args.split}", "attr")


def hint(rules):
    parts = [f"{ATTR_TEXT.get(tuple(t[:2]), t[0] + ' ' + t[1])} follows the rule '{t[2]}' ({REL_TEXT.get(t[2], t[2])})"
             for t in rules]
    return "Hint: in every row, " + "; and ".join(parts) + ".\n"


def stage_infer(args):
    lay = G.load_layout(args.layout)
    for cond in args.conditions:
        items = []
        for p, rules in puzzles_with_rules(args):
            rec = {"id": p.id, "puzzle": p.id, "answer": p.answer, "rules": rules,
                   "attr": list(rules[0][:2]) if len(rules) == 1 else ["multiple", "rules"]}
            h = hint(rules) if cond.endswith("_rule") else ""
            if cond.startswith("sheet"):
                items.append((rec, [{"type": "image"}, {"type": "text", "text": h + p.prompt}], [G.open_sheet(p.image)]))
                continue
            _, ctx, ch = grid_panels(p, lay)
            content, imgs, k = [{"type": "text", "text": G.CROPS_PUZZLE_INTRO.format(cell=G.CELL_NAMES[lay["missing"]])}], [], 0
            for cell in range(9):
                lab = f"\nRow {cell // 3 + 1}, panel {cell % 3 + 1}:"
                if cell == lay["missing"]:
                    content.append({"type": "text", "text": lab + " ? (missing)"})
                else:
                    content += [{"type": "text", "text": lab}, {"type": "image"}]
                    imgs.append(G.upscale(ctx[k], args.panel_size))
                    k += 1
            for L, im in zip(LETTERS, ch):
                content += [{"type": "text", "text": f"\nChoice {L}:"}, {"type": "image"}]
                imgs.append(G.upscale(im, args.panel_size))
            content.append({"type": "text", "text": "\n" + h + G.CROPS_PUZZLE_Q.strip()})
            items.append((rec, content, imgs))
        bs = args.batch_size
        args.batch_size = bs if cond.startswith("sheet") else max(1, bs // 4)
        path = run_questions(args, f"infer_{cond}_{args.split}", items, 8)
        args.batch_size = bs
        summarize(model_dir(args), f"infer_{cond}_{args.split}", "attr")


# ============================================================================ report

def summarize(d, stem, by):
    rows = list(G.read_merged(d, stem).values())
    if not rows:
        return {}
    out = {}
    groups = defaultdict(list)
    for r in rows:
        groups[" ".join(r[by]) if isinstance(r[by], list) else r[by]].append(r["correct"])
    for g, v in sorted(groups.items()):
        out[g] = G.wilson(sum(v), len(v)) + (len(v),)
    allv = [r["correct"] for r in rows]
    out["all"] = G.wilson(sum(allv), len(allv)) + (len(allv),)
    print(f"{stem}: " + ", ".join(f"{g} {a:.1%} (n={n})" for g, (a, lo, hi, n) in out.items()))
    return out


def stage_report(args):
    sp = args.split
    tags = sorted(t for t in os.listdir(args.out) if os.path.isdir(os.path.join(args.out, t))
                  and any(f.endswith(f"_{sp}.jsonl") for f in os.listdir(os.path.join(args.out, t))))
    f = lambda x: f"{x[0]:.1%} [{x[1]:.1%}, {x[2]:.1%}] n={x[3]}" if x else "-"
    L = ["# Real PGM: attribute perception vs rule inference", ""]
    for t in tags:
        d = os.path.join(args.out, t)
        order = summarize(d, f"order_{sp}", "attr")
        ra = summarize(d, f"ruleid_attribute_{sp}", "attr")
        rr = summarize(d, f"ruleid_relation_{sp}", "attr")
        inf = {c: summarize(d, f"infer_{c}_{sp}", "attr") for c in ("crops", "crops_rule", "sheet", "sheet_rule")}
        inf = {c: v for c, v in inf.items() if v}
        L += [f"## {t}", "", "| measure | chance | accuracy [95% CI] |", "|---|---|---|",
              f"| order: pick the intermediate panel (attribute perception) | 33% | {f(order.get('all'))} |",
              f"| ruleid: which attribute follows a rule | 14% | {f(ra.get('all'))} |",
              f"| ruleid: which relation, given the attribute | 20% | {f(rr.get('all'))} |"]
        for c, v in inf.items():
            L.append(f"| solve: {c} | 12.5% | {f(v.get('all'))} |")
        keys = sorted((set(order) | set(ra) | {k for v in inf.values() for k in v}) - {"all", "multiple rules"})
        if keys:
            L += ["", "Per rule attribute (single-rule puzzles for ruleid and solving):", "",
                  "| attribute | order | ruleid attribute | ruleid relation | " + " | ".join(inf) + " | gain crops_rule - crops |",
                  "|---" * (len(inf) + 5) + "|"]
            for k in keys:
                cells = [order.get(k), ra.get(k), rr.get(k)] + [inf[c].get(k) for c in inf]
                g = (inf["crops_rule"][k][0] - inf["crops"][k][0]) if ("crops" in inf and "crops_rule" in inf
                                                                      and k in inf["crops"] and k in inf["crops_rule"]) else None
                L.append(f"| {k} | " + " | ".join(f"{x[0]:.0%} (n={x[3]})" if x else "-" for x in cells)
                         + f" | {'-' if g is None else f'{g:+.0%}'} |")
        L.append("")
    text = "\n".join(L)
    print("\n" + text)
    open(os.path.join(args.out, f"report_{sp}.md"), "w").write(text + "\n")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stage", choices=["rules", "order", "ruleid", "infer", "report"])
    ap.add_argument("--data", default=os.environ.get("DATA"), help="dataset root (default $DATA)")
    ap.add_argument("--split", default="validation")
    ap.add_argument("--hf_split", default=None, help="rules: PGM split name if it differs from --split")
    ap.add_argument("--layout", default="layout.json")
    ap.add_argument("--out", default="real_probe")
    ap.add_argument("--model", default="OpenGVLab/InternVL3_5-1B-HF")
    ap.add_argument("--conditions", nargs="+", default=["crops", "crops_rule", "sheet", "sheet_rule"],
                    choices=["crops", "crops_rule", "sheet", "sheet_rule"])
    ap.add_argument("--limit", type=int, default=0, help="first N puzzles of the split (0 = all)")
    ap.add_argument("--rows", type=int, default=3, help="order: rows used per progression rule (1-3)")
    ap.add_argument("--image_key", default="images")
    ap.add_argument("--rules_key", default=None, help="dotted path to relation_structure in your JSONL, if present")
    ap.add_argument("--panel_size", type=int, default=448)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--answer_prefix", default="")
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--attn", default="sdpa")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    args.causal = False
    if args.stage != "report" and not args.data:
        ap.error("give the dataset root with --data $DATA (or export DATA=...)")
    {"rules": stage_rules, "order": stage_order, "ruleid": stage_ruleid, "infer": stage_infer,
     "report": stage_report}[args.stage](args)


if __name__ == "__main__":
    main()
