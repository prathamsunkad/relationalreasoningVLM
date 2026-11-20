"""Evaluate one adapter from the bundle on a contact-sheet manifest (PGM or I-RAVEN).

Inference mirrors the bundle's load_adapter.py exactly: same loader (pinned base revision,
checksum-verified adapter), same chat template call, greedy decoding, max_new_tokens=16,
one image per prompt. Predictions are appended to a JSONL file so an interrupted run resumes.
"""
import argparse, io, json, math, os, sys, tarfile, time
from collections import Counter, defaultdict
from pathlib import Path

LETTERS = 'ABCDEFGH'


def wilson(k, n, z=1.959963984540054):
    if n == 0: return [0.0, 0.0]
    p = k / n; d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d; h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [c - h, c + h]


def parse(text):
    t = text.strip().strip('.').strip().upper()
    return t if len(t) == 1 and t in LETTERS else None


def summarize(rows, preds, entry, tag, elapsed):
    groups = defaultdict(lambda: [0, 0, 0])
    lat = []
    for r in rows:
        p = preds[r['id']]
        for g in ('__all__', r.get('regime', 'unknown')):
            groups[g][0] += 1; groups[g][1] += int(p['prediction'] == r['answer']); groups[g][2] += int(p['prediction'] is None)
        lat.append(p['latency_seconds'])
    def block(v):
        n, k, inv = v
        return {'evaluated': n, 'correct': k, 'accuracy': k / n, 'wilson_95_interval': wilson(k, n),
                'invalid_outputs': inv, 'invalid_rate': inv / n}
    out = {'key': entry['key'], 'base_model': entry['base_model'], 'base_revision': entry['base_revision'],
           'manifest': tag, 'chance_accuracy': 0.125, 'overall': block(groups.pop('__all__')),
           'by_regime': {g: block(v) for g, v in sorted(groups.items())},
           'prediction_histogram': dict(sorted(Counter(str(preds[r['id']]['prediction']) for r in rows).items())),
           'answer_histogram': dict(sorted(Counter(r['answer'] for r in rows).items())),
           'mean_latency_seconds': sum(lat) / len(lat), 'wall_seconds_this_run': elapsed}
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--bundle', type=Path, default=Path.home() / 'scratch' / 'pgm-best-lora-2026-11-01-v1')
    ap.add_argument('--key', required=True)
    ap.add_argument('--manifest', type=Path, required=True)
    ap.add_argument('--root', type=Path, help='dataset root (default: the manifest folder)')
    ap.add_argument('--tag', required=True, help='short name for the output files')
    ap.add_argument('--limit', type=int, default=0)
    ap.add_argument('--out-dir', type=Path, required=True)
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--dtype', default='bfloat16')
    a = ap.parse_args()
    if Path('/storage/ice1').exists() and not os.environ.get('SLURM_JOB_ID'):
        raise RuntimeError('On ICE, run inference inside an allocated compute job, not a login node.')
    root = a.root or a.manifest.parent
    rows = [json.loads(line) for line in a.manifest.open()]
    if a.limit: rows = rows[:a.limit]
    a.out_dir.mkdir(parents=True, exist_ok=True)
    pred_path = a.out_dir / ('%s__%s.predictions.jsonl' % (a.key, a.tag))
    preds = {}
    if pred_path.exists():
        for line in pred_path.open():
            p = json.loads(line); preds[p['id']] = p
    todo = [r for r in rows if r['id'] not in preds]
    print('%d rows, %d already done, %d to run' % (len(rows), len(rows) - len(todo), len(todo)), flush=True)

    sys.path.insert(0, str(a.bundle))
    from load_adapter import load_adapter
    import torch
    from PIL import Image
    model, processor, entry = load_adapter(a.key, a.bundle, a.device, a.dtype)
    print('loaded', entry['key'], entry['base_model'], 'on', torch.cuda.get_device_name(0) if a.device == 'cuda' else a.device, flush=True)

    tars = {}
    def load_image(r):
        rel = r['images'][0]
        if (root / rel).exists():
            with Image.open(root / rel) as im: return im.convert('RGB')
        arc = r['metadata']['archive']
        if arc not in tars: tars[arc] = tarfile.open(root / arc)
        with Image.open(io.BytesIO(tars[arc].extractfile(rel).read())) as im: return im.convert('RGB')

    t0 = time.perf_counter()
    with pred_path.open('a') as f:
        for n, r in enumerate(todo, 1):
            image = load_image(r)
            messages = [{'role': 'user', 'content': [{'type': 'image', 'image': image}, {'type': 'text', 'text': r['prompt']}]}]
            inputs = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=True,
                                                   return_dict=True, return_tensors='pt', truncation=False)
            inputs = inputs.to(device=model.device, dtype=model.dtype)
            t = time.perf_counter()
            with torch.inference_mode():
                output = model.generate(**inputs, do_sample=False, num_beams=1, max_new_tokens=16,
                                        use_cache=True, return_dict_in_generate=False)
            text = processor.decode(output[0, inputs['input_ids'].shape[1]:], skip_special_tokens=True).strip()
            p = {'id': r['id'], 'answer': r['answer'], 'raw_output': text, 'prediction': parse(text),
                 'latency_seconds': time.perf_counter() - t}
            preds[r['id']] = p; f.write(json.dumps(p) + '\n')
            if n % 200 == 0 or n == len(todo):
                f.flush()
                done = [preds[x['id']] for x in rows if x['id'] in preds]
                acc = sum(x['prediction'] == x['answer'] for x in done) / len(done)
                print('%d/%d  running accuracy %.4f  %.1f items/s' % (n, len(todo), acc, n / (time.perf_counter() - t0)), flush=True)
    summary = summarize(rows, preds, entry, a.tag, time.perf_counter() - t0)
    (a.out_dir / ('%s__%s.summary.json' % (a.key, a.tag))).write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
