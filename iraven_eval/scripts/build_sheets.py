"""Render packed I-RAVEN test puzzles as model-ready contact sheets in the exact PGM
presentation ("matrix-3x3-options-2x4-v1") the adapters were trained and tested on.

The template is a real PGM contact sheet with its 16 panel slots blanked, so labels, the
"?" cell and all spacing are pixel-identical to the PGM inputs; only the panels differ.
Output: one tar of PNGs per configuration (few files, for the inode quota), test.jsonl in
the same schema as the PGM manifest, README.md and dataset-description.json.
"""
import argparse, hashlib, io, json, tarfile
from pathlib import Path
import numpy as np
from PIL import Image

SLOTS = [(98 + 172 * c, 42 + 172 * r) for r in range(3) for c in range(3)][:8] + \
        [(12 + 172 * c, 630 + 202 * r) for r in range(2) for c in range(4)]
LETTERS = 'ABCDEFGH'


def render(template, panels):
    """panels: uint8 array (16,160,160): 8 context panels (row-major) then 8 choices A-H."""
    assert panels.shape == (16, 160, 160) and panels.dtype == np.uint8
    sheet = template.copy()
    for (x, y), p in zip(SLOTS, panels):
        sheet.paste(Image.fromarray(p, 'L'), (x, y))
    return sheet


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--raw', type=Path, required=True)
    ap.add_argument('--template', type=Path, required=True)
    ap.add_argument('--prompts', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--generator-commit', default='unknown')
    ap.add_argument('--seed', type=int, default=1234)
    a = ap.parse_args()
    template = Image.open(a.template); assert template.mode == 'L' and template.size == (700, 1016)
    prompt = json.loads(a.prompts.read_text())['pgm']
    a.out.mkdir(parents=True, exist_ok=True)
    counts, archives, rows = {}, [], []
    for raw in sorted(a.raw.glob('iraven_test_*.npz')):
        cfg = raw.stem[len('iraven_test_'):]
        d = np.load(raw)
        arc = 'test-%s.tar' % cfg
        with tarfile.open(a.out / arc, 'w') as tar:
            for i in range(len(d['k'])):
                k, target = int(d['k'][i]), int(d['target'][i])
                sheet = render(template, d['image'][i])
                buf = io.BytesIO(); sheet.save(buf, 'PNG'); png = buf.getvalue()
                rel = 'images/%s/%05d.png' % (cfg, k)
                info = tarfile.TarInfo(rel); info.size = len(png); info.mode = 0o644
                tar.addfile(info, io.BytesIO(png))
                rows.append({'answer': LETTERS[target], 'choices': list(LETTERS), 'dataset': 'i-raven',
                             'id': 'i-raven/%s/test/%d' % (cfg, k), 'images': [rel],
                             'metadata': {'archive': arc, 'configuration': cfg, 'generator_index': k,
                                          'generator': 'husheng12345/SRAN I-RAVEN', 'generator_commit': a.generator_commit,
                                          'generator_seed': a.seed, 'contact_sheet_sha256': hashlib.sha256(png).hexdigest(),
                                          'contact_sheet_size': [700, 1016],
                                          'decoded_rgb_sha256': hashlib.sha256(sheet.convert('RGB').tobytes()).hexdigest(),
                                          'presentation': 'matrix-3x3-options-2x4-v1', 'source_image_shape': [16, 160, 160],
                                          'meta_target': d['meta_target'][i].tolist()},
                             'prompt': prompt, 'regime': cfg, 'split': 'test', 'task': 'matrix-completion'})
        counts[cfg] = int(len(d['k'])); archives.append(arc)
        print(cfg, counts[cfg], 'sheets ->', arc, flush=True)
    with (a.out / 'test.jsonl').open('w') as f:
        for r in rows: f.write(json.dumps(r, sort_keys=True) + '\n')
    desc = {'dataset': 'i-raven', 'split': 'test', 'counts': counts, 'total': len(rows), 'archives': archives,
            'presentation': 'matrix-3x3-options-2x4-v1', 'generator': 'https://github.com/husheng12345/SRAN (I-RAVEN/)',
            'generator_commit': a.generator_commit, 'generator_seed': a.seed,
            'split_rule': 'generator default 6:2:2 by index (index % 10 in {8, 9} is test)'}
    (a.out / 'dataset-description.json').write_text(json.dumps(desc, indent=2) + '\n')
    (a.out / 'README.md').write_text(
        '# I-RAVEN test split as PGM-style contact sheets\n\n'
        '%d test puzzles (%s), generated with the I-RAVEN authors\' unmodified generator '
        '(husheng12345/SRAN, commit %s) under Python 2.7 with its pinned requirements and default '
        'settings (seed %d, 10,000 puzzles per configuration, 6:2:2 split; only the test split is kept).\n\n'
        'Each puzzle is one 700x1016 grayscale PNG in the same presentation as the PGM contact sheets '
        '(3x3 matrix with a missing cell, eight choices A-H); the template is a real PGM sheet with the panels '
        'replaced. Load with PIL.Image.open(...).convert(\'RGB\'). test.jsonl uses the PGM manifest schema; '
        '`regime` holds the I-RAVEN configuration. PNGs are stored in one tar per configuration because the '
        'scratch file-count quota is full; `tar -xf test-<config>.tar` in this folder makes the manifest paths '
        'resolve directly, and eval_adapter.py also reads them straight from the tars.\n'
        % (len(rows), ', '.join('%s: %d' % kv for kv in counts.items()), a.generator_commit, a.seed))
    print('total', len(rows))


if __name__ == '__main__':
    main()
