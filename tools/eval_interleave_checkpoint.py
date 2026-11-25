"""Score a saved InternVL interleave checkpoint (tools/train_*_interleave.py) on validation or test, with ablations.

Added Nov 25 for the missing H15-on-InternVL comparisons (branch off, ablations, test-set scores). It reuses the training
code's own pieces unchanged: load_base/load_adapter, CachedInterleave + its saved connector, the same Collator and
render_grid, candidate_scores, and FrozenRelations (the frozen Slot Abstractor, same package, hashes and seed).

  --split val   rows of plan['val_manifest'] (contact sheets, as in training validation)
  --split test  puzzles from a list of PGM .npz names (e.g. results/splits/test5000.tsv). Each puzzle's 16 panels are
                pasted into a blank 700 x 1016 sheet at exactly the positions pgmvlm.panels.split_composite crops, so the
                cropped panels (for InternVL and for the Slot Abstractor) are the .npz panels pixel for pixel.

Ablations (the relation token only; the image and prompt are unchanged):
  none            as trained
  zero            branch off: the relation token is zeros (the hook's branch_off), placeholder kept
  shuffle_cand    each candidate gets another candidate's relations from the same puzzle (cyclic shift by one)
  shuffle_puzzle  each puzzle gets the previous puzzle's relations (same candidate slots)

Writes one JSON with per-puzzle scores (same record format as the training validation files) and a summary.
"""
from __future__ import annotations
import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'relational'))
import numpy as np
import torch
from PIL import Image
from safetensors.torch import load_file
from backbones import Backbone
from cached_interleave import CachedInterleave
from data import Collator, render_grid
from pgmvlm.data import PROMPT_TEXT
from pgmvlm.datasets import read_manifest
from pgmvlm.models import load_base, load_adapter, candidate_scores, ScoreTokens
from pgmvlm.panels import split_composite
from pgmvlm.pretrained_relations import FrozenRelations
from tools.internvl_connector_pilot import candidate_item
from tools.input_representation_pilot import resolve_image

SHEET = (700, 1016)
POS = [(98 + (i % 3) * 172, 42 + (i // 3) * 172) for i in range(8)] + \
      [(12 + (i % 4) * 172, 630 + (i // 4) * 202) for i in range(8)]


def npz_question(path):
    d = np.load(path)
    panels = d['image'].reshape(16, 160, 160).astype(np.uint8)      # panel-major bytes (see pgmvlm.data.to_panels)
    sheet = Image.new('RGB', SHEET, (255, 255, 255))
    for p, xy in zip(panels, POS):
        sheet.paste(Image.fromarray(p, mode='L').convert('RGB'), xy)
    back = np.stack([np.asarray(c.convert('L')) for c in split_composite(sheet)])
    if not np.array_equal(back, panels):
        raise RuntimeError(f'{path}: sheet round trip is not lossless')
    return dict(context=panels[:8], candidates=panels[8:], target=int(d['target'])), sheet


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--plan', required=True, help="the run's plan json (teacher hashes/seed, manifests)")
    ap.add_argument('--checkpoint', required=True, help='a saved folder: adapter_model.safetensors + connector.safetensors')
    ap.add_argument('--model', default='OpenGVLab/InternVL3_5-1B-HF')
    ap.add_argument('--teacher_package', required=True, help='the Slot Abstractor package (same files/hashes as the plan)')
    ap.add_argument('--split', choices=['val', 'test'], required=True)
    ap.add_argument('--npz_root', help='test: folder with the .npz files')
    ap.add_argument('--list', help='test: list of .npz names (first column; # lines skipped)')
    ap.add_argument('--ablate', default='none', choices=['none', 'zero', 'shuffle_cand', 'shuffle_puzzle'])
    ap.add_argument('--limit', type=int, default=0)
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    plan = json.load(open(a.plan))
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    device = torch.device('cuda')
    model, processor, family = load_base(a.model, device=device)
    if family != 'internvl':
        raise ValueError('Expected InternVL3.5 HF')
    model = load_adapter(model, a.checkpoint, trainable=False)
    model.eval()
    bb = Backbone(model.get_base_model(), processor, family)
    st = ScoreTokens(processor.tokenizer)
    conn = CachedInterleave(bb).to(device=device, dtype=torch.float32)
    conn.load_state_dict(load_file(str(Path(a.checkpoint) / 'connector.safetensors')), strict=True)
    conn.eval()
    conn.branch_off = a.ablate == 'zero'
    collator = Collator(processor, {'crop_to_patches': True}, rel_tokens=1, rel_mode='pool', num_slots=16)
    teacher = FrozenRelations(a.teacher_package, plan['teacher_hashes'], device=device)

    if a.split == 'val':
        rows = read_manifest(plan['val_manifest'])
        def load(i):
            row = rows[i]
            item = candidate_item(row, plan)
            with Image.open(resolve_image(row, plan)) as im:
                sheet = im.convert('RGB')
            return row['id'], item, sheet
        n = len(rows)
    else:
        names = [l.split('\t')[0].strip() for l in open(a.list) if l.strip() and not l.startswith('#')]
        def load(i):
            item, sheet = npz_question(Path(a.npz_root) / names[i])
            return 'pgm/neutral/test/' + names[i].split('_')[-1].split('.')[0], item, sheet
        n = len(names)
    if a.limit:
        n = min(n, a.limit)

    records, prev_bank, t0 = [], None, time.time()
    for i in range(n):
        qid, item, sheet = load(i)
        bank = teacher.extract(sheet, qid, plan['teacher_seed'], verify=not teacher.parity_verified)[0].float()
        use = bank
        if a.ablate == 'shuffle_cand':
            use = bank[[(k + 1) % 8 for k in range(8)]]
        elif a.ablate == 'shuffle_puzzle':
            use = prev_bank if prev_bank is not None else bank[[(k + 1) % 8 for k in range(8)]]
        prev_bank = bank
        seqs = [dict(image=render_grid(list(item['context']) + [p])[0], boxes=render_grid(list(item['context']) + [p])[1],
                     prompt=PROMPT_TEXT) for p in item['candidates']]
        batch = collator([dict(kind='cand', seqs=seqs, label=item['target'], id=qid)])
        batch['targets'] = torch.tensor([item['target']])
        with torch.no_grad(), conn.question(use.to(device).contiguous(), batch['input_ids'].eq(bb.rel_id).to(device)):
            s = candidate_scores(model, batch, st, device, torch.bfloat16)
        target = torch.tensor([item['target']], device=device)
        loss = float(torch.nn.functional.cross_entropy(s.float(), target))
        records.append(dict(index=i, id=qid, scores=s[0].float().tolist(), target=item['target'],
                            prediction=int(s.argmax()), correct=int(s.argmax()) == item['target'], loss=loss))
        if (i + 1) % 250 == 0:
            acc = sum(r['correct'] for r in records) / len(records)
            print(f'{i + 1}/{n} acc {acc:.4f} ({time.time() - t0:.0f}s)', flush=True)
    acc = sum(r['correct'] for r in records) / len(records)
    summary = dict(split=a.split, ablate=a.ablate, checkpoint=str(a.checkpoint), count=len(records), accuracy=acc,
                   loss=sum(r['loss'] for r in records) / len(records), seconds=time.time() - t0)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(dict(summary=summary, predictions=records), open(a.out, 'w'))
    print(json.dumps(summary))


if __name__ == '__main__':
    main()
