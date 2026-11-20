"""Bounded, matched PGM input screen: frozen vision features, trainable heads.

Run only inside a GPU allocation. Uses the existing selected-rows.json and never
reads test questions. Cache files are disposable and remain on node-local disk.
"""
from __future__ import annotations

import argparse
from collections import Counter
import copy
import fcntl
import hashlib
import json
import os
from pathlib import Path
import random
import shutil
import signal
import subprocess
import sys
import tempfile
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from PIL import Image
import torch
from torch.nn import functional as F
from pgmvlm.input_probe import (ARMS, CACHE_KEYS, InputProbe, cache_key, cache_representations,
                              context_normalize, feature_statistics, make_input_arms)
from pgmvlm.extensions.probe_diagnostics import make_arms, tensor_digest
from pgmvlm.models import load_base
from pgmvlm.panels import split_composite
from pgmvlm.storage import require_headroom

STOP = False


def stopped(signum, frame):
    global STOP
    STOP = True


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()


def atomic_json(path, value):
    temp = path.with_name(path.name + '.tmp')
    with temp.open('w') as f:
        json.dump(value, f, indent=2, allow_nan=False)
        f.write('\n'); f.flush(); os.fsync(f.fileno())
    temp.replace(path)


def checkpoint(path, value):
    temp = path.with_name(path.name + '.tmp')
    torch.save(value, temp)
    with temp.open('rb') as f:
        os.fsync(f.fileno())
    temp.replace(path)
    atomic_json(path.with_suffix('.sha256.json'), dict(sha256=digest(path)))


def append(path, row):
    with path.open('a') as f:
        f.write(json.dumps(row, allow_nan=False) + '\n')


def resolve_image(row, plan):
    rel = Path(row['images'][0])
    if rel.is_absolute() or '..' in rel.parts:
        raise ValueError('Expected a relative data image path')
    folder = Path(plan['data_root'])
    path = folder / rel
    if not path.is_file() and row['split'] == 'validation':
        path = Path(plan['validation_image_root']) / rel.name
    if not path.is_file() or digest(path) != row['metadata']['contact_sheet_sha256']:
        raise ValueError('Missing image or checksum mismatch: ' + row['id'])
    return path


@torch.no_grad()
def image_features(model, processor, images, panels=False):
    kwargs = dict(images=images, return_tensors='pt')
    # Each original 160px panel stays one image; the whole-sheet arm keeps native tiling.
    if panels:
        kwargs.update(do_image_splitting=False)
    batch = processor.image_processor(**kwargs)
    inputs = {k: batch[k].to('cuda') for k in ('pixel_values', 'spatial_shapes', 'pixel_attention_mask')}
    inputs['pixel_values'] = inputs['pixel_values'].to(torch.bfloat16)
    values = model.model.get_image_features(**inputs, return_dict=True).pooler_output
    factor = int(model.config.downsample_factor)
    shapes = [((int(h) + factor - 1)//factor, (int(w) + factor - 1)//factor)
              for h, w in inputs['spatial_shapes'].tolist()]
    if len(values) != len(shapes) or any(len(v) != h*w for v, (h, w) in zip(values, shapes)):
        raise RuntimeError('Projected grid geometry does not match native model outputs')
    if panels and len(values) != len(images):
        raise RuntimeError('Panel image unexpectedly expanded into multiple tiles')
    return values, shapes


def local_cache_parent(required):
    candidates = list(dict.fromkeys([os.environ.get('SLURM_TMPDIR'), os.environ.get('TMPDIR'), '/tmp']))
    for parent in candidates:
        if not parent or not Path(parent).is_dir():
            continue
        fs = subprocess.run(['findmnt', '-T', parent, '-n', '-o', 'FSTYPE'], capture_output=True, text=True)
        if fs.returncode or fs.stdout.strip() in ('tmpfs', 'ramfs', 'lustre', 'nfs', 'nfs4'):
            continue
        if shutil.disk_usage(parent).free >= required:
            return parent
    raise RuntimeError('No verified node-local disk with enough cache headroom; refusing RAM/shared-storage cache')


def load_selection(plan):
    if digest(plan['selected_rows']) != plan['selected_rows_sha256']:
        raise ValueError('Historical question selection changed')
    selected = json.loads(Path(plan['selected_rows']).read_text())
    train, val = selected['train']['4096'], selected['validation']
    if len(train) != 4096 or len(val) != 1000:
        raise ValueError('Wrong split size')
    if len({r['id'] for r in train + val}) != 5096:
        raise ValueError('Duplicate or overlapping train/validation questions')
    for rows, split in ((train, 'train'), (val, 'validation')):
        if any(r['split'] != split or r['dataset'] != 'pgm' or r['regime'] != 'neutral'
               or r['choices'] != list('ABCDEFGH') or r['answer'] not in 'ABCDEFGH' for r in rows):
            raise ValueError('Unexpected dataset, split, task or label')
    monitor = selected['training_monitors']['4096']
    lookup = {r['id']: i for i, r in enumerate(train)}
    monitor = [lookup[r['id'] if isinstance(r, dict) else r] for r in monitor]
    if len(monitor) != 512:
        raise ValueError('Wrong historical training monitor')
    return train, val, monitor


def evaluate(arm, cache, labels, indices, batch_size=32, shuffle_images=False):
    arm.eval(); losses, predictions = [], []
    image_indices = indices[1:] + indices[:1] if shuffle_images else indices
    with torch.no_grad():
        for begin in range(0, len(indices), batch_size):
            target_idx = indices[begin:begin+batch_size]
            source_idx = image_indices[begin:begin+batch_size]
            x = torch.from_numpy(np.array(cache[source_idx], copy=True)).to('cuda')
            y = torch.tensor(labels[target_idx], device='cuda')
            logits = arm(x)
            losses.extend(F.cross_entropy(logits, y, reduction='none').cpu().tolist())
            predictions.extend(logits.argmax(-1).cpu().tolist())
    return dict(count=len(indices), answer_ce=float(np.mean(losses)),
                accuracy=float(np.mean(np.asarray(predictions) == labels[indices])),
                prediction_histogram=dict(Counter(str(p) for p in predictions)),
                predictions=predictions, losses=losses, image_indices=image_indices)


def run(plan, output, deadline):
    if not os.environ.get('SLURM_JOB_ID') or not torch.cuda.is_available():
        raise RuntimeError('GPU allocation required; never run on a login node')
    output.mkdir(parents=True, exist_ok=True)
    with (output/'.run.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (output/'config.json').exists():
            raise RuntimeError('Existing pilot output; do not overwrite or silently resume')
        atomic_json(output/'config.json', plan)
        atomic_json(output/'storage-preflight.json', require_headroom(output))
        torch.set_num_threads(2)
        torch.manual_seed(plan['initialization_seed']); np.random.seed(plan['training_seed'])
        torch.use_deterministic_algorithms(True)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        for sig in (signal.SIGUSR1, signal.SIGTERM, signal.SIGINT):
            signal.signal(sig, stopped)
        train, val, monitor = load_selection(plan)
        rows = train + val
        atomic_json(output/'question_ids.json', dict(train=[r['id'] for r in train],
                    validation=[r['id'] for r in val], monitor=[train[i]['id'] for i in monitor]))
        labels = np.asarray([ord(r['answer'])-65 for r in rows], dtype=np.int64)
        validation_indices = list(range(len(train), len(rows)))
        model, processor, _ = load_base('lfm25', revision=plan['revision'], device='cuda')
        model.eval().requires_grad_(False)
        base_hash = tensor_digest(model.named_parameters())
        dimension = model.config.text_config.hidden_size
        needed = len(rows) * 128 * dimension * 4 * len(CACHE_KEYS) + 4 * 1024**3
        parent = local_cache_parent(needed)
        prepared = time.time()
        with tempfile.TemporaryDirectory(prefix='pgm-input-'+os.environ['SLURM_JOB_ID']+'-', dir=parent) as cache_dir:
            caches = {key: np.lib.format.open_memmap(Path(cache_dir)/(key+'.npy'), mode='w+',
                      dtype=np.float32, shape=(len(rows), 128, dimension)) for key in CACHE_KEYS}
            stats = []
            for i, row in enumerate(rows):
                if STOP or time.time() > deadline - plan['reserve_seconds']:
                    raise RuntimeError('Preparation reached allocation deadline; no training result claimed')
                with Image.open(resolve_image(row, plan)) as raw:
                    image = raw.convert('RGB')
                sheet, sheet_shapes = image_features(model, processor, [image])
                panels, panel_shapes = image_features(model, processor, split_composite(image), panels=True)
                pooled = cache_representations(sheet, panels, panel_shapes)
                for key in CACHE_KEYS:
                    if not torch.isfinite(pooled[key]).all():
                        raise RuntimeError('Nonfinite frozen features')
                    caches[key][i] = pooled[key].cpu().numpy()
                if i < 8:
                    entry = dict(id=row['id'], sheet_shapes=sheet_shapes, panel_shapes=panel_shapes,
                                 features={key: feature_statistics(x) for key, x in pooled.items()})
                    entry['features']['panels_grid_context'] = feature_statistics(context_normalize(pooled['panels_grid']))
                    stats.append(entry)
                if i == 0:
                    # Validate cache equivalence on actual native features, not just synthetic tensors.
                    original = make_arms(plan['initialization_seed'], input_dim=dimension)['classifier_d2'].cuda()
                    cached_arm = InputProbe(original).cuda()
                    with torch.no_grad():
                        a = original(torch.cat(sheet).unsqueeze(0)); b = cached_arm(pooled['sheet_flat'].unsqueeze(0))
                    torch.testing.assert_close(a, b, atol=2e-6, rtol=1e-4)
                    atomic_json(output/'pilot-result.json', dict(status='passed', cache_logit_max_difference=float((a-b).abs().max()),
                        panel_count=16, cached_tokens=128, gpu=torch.cuda.get_device_name(),
                        frozen_parameters=sum(p.numel() for p in model.parameters())))
                    del original, cached_arm
                if (i+1) % 64 == 0 or i+1 == len(rows):
                    progress = dict(phase='features', completed=i+1, total=len(rows), seconds=time.time()-prepared,
                                    projected_total_seconds=(time.time()-prepared)*len(rows)/(i+1))
                    atomic_json(output/'progress.json', progress); print(json.dumps(progress), flush=True)
            for cache in caches.values(): cache.flush()
            final_hash = tensor_digest(model.named_parameters())
            if base_hash != final_hash or any(p.grad is not None for p in model.parameters()):
                raise RuntimeError('Frozen VLM unexpectedly changed or received parameter gradients')
            atomic_json(output/'features.json', dict(base_unchanged=True, base_sha256=base_hash,
                        cache_shape=[len(rows), 128, dimension], feature_seconds=time.time()-prepared,
                        cache_bytes=needed-4*1024**3, diagnostics=stats))
            del model, processor, sheet, panels, pooled
            torch.cuda.empty_cache()
            arms = {name: arm.cuda() for name, arm in make_input_arms(plan['initialization_seed'], dimension).items()}
            optimizers = {name: torch.optim.AdamW([p for p in arm.parameters() if p.requires_grad],
                          lr=plan['lr'], weight_decay=plan['weight_decay']) for name, arm in arms.items()}
            best, last = {}, {}; updates = 0
            started = time.time()
            for epoch in range(plan['max_epochs']+1):
                if epoch:
                    order = list(range(len(train)))
                    random.Random(plan['training_seed']+epoch-1).shuffle(order)
                    for begin in range(0, len(order), plan['global_batch']):
                        if STOP or time.time() > deadline-plan['reserve_seconds']:
                            break
                        indices = order[begin:begin+plan['global_batch']]
                        y = torch.tensor(labels[indices], device='cuda')
                        # All arms see the identical next batch before any arm advances.
                        names = list(ARMS); rotate = updates % len(names); names = names[rotate:]+names[:rotate]
                        for name in names:
                            arm, opt = arms[name], optimizers[name]
                            arm.train(); opt.zero_grad(set_to_none=True)
                            for group in opt.param_groups:
                                group['lr'] = plan['lr']*min(1., (updates+1)/plan['warmup_updates'])
                            x = torch.from_numpy(np.array(caches[cache_key(name)][indices], copy=True)).to('cuda')
                            logits = arm(x); loss = F.cross_entropy(logits, y)
                            if not torch.isfinite(loss): raise RuntimeError('Nonfinite training loss')
                            loss.backward()
                            gradient_norm = torch.nn.utils.clip_grad_norm_(arm.parameters(), plan['clip_norm'], error_if_nonfinite=True)
                            opt.step()
                            if updates % 64 == 0:
                                append(output/'learning-curve.jsonl', dict(arm=name, update=updates+1,
                                    epoch=(updates+1)*plan['global_batch']/len(train), answer_ce=float(loss.detach()),
                                    accuracy=float((logits.argmax(-1)==y).float().mean()), preclip_gradient_norm=float(gradient_norm),
                                    q_gradient_rms_postclip={n:float(p.grad.float().square().mean().sqrt()) for n,p in arm.named_parameters()
                                                    if '.relation.q.weight' in n and p.grad is not None}))
                        updates += 1
                        if updates % 64 == 0:
                            atomic_json(output/'progress.json', dict(phase='training', updates_per_arm=updates,
                                epochs_per_arm=updates*plan['global_batch']/len(train), seconds=time.time()-started))
                fraction = updates*plan['global_batch']/len(train)
                for name, arm in arms.items():
                    scores = evaluate(arm, caches[cache_key(name)], labels, validation_indices)
                    fit = evaluate(arm, caches[cache_key(name)], labels, monitor)
                    last[name] = dict(arm=name, epoch=fraction, updates=updates,
                        validation={k:v for k,v in scores.items() if k not in ('predictions','losses','image_indices')},
                        training_monitor={k:v for k,v in fit.items() if k not in ('predictions','losses','image_indices')})
                    append(output/'evaluation-curve.jsonl', last[name])
                    print(json.dumps(last[name]), flush=True)
                    if name not in best or scores['answer_ce'] < best[name]['validation']['answer_ce']:
                        best[name] = copy.deepcopy(last[name])
                        checkpoint(output/(name+'-best.pt'), dict(model=arm.state_dict(), epoch=fraction, update=updates,
                                   plan=plan, model_kind='input_probe', arm=name))
                        atomic_json(output/(name+'-best-validation.json'), dict(ids=[rows[i]['id'] for i in validation_indices], **scores))
                    if epoch:
                        checkpoint(output/(name+'-latest.pt'), dict(model=arm.state_dict(), optimizer=optimizers[name].state_dict(),
                                   epoch=fraction, update=updates, plan=plan, arm=name))
                if STOP or time.time() > deadline-plan['reserve_seconds']:
                    break
            shuffled = {}
            for name, arm in arms.items():
                result = evaluate(arm, caches[cache_key(name)], labels, validation_indices, shuffle_images=True)
                shuffled[name] = {k:v for k,v in result.items() if k not in ('predictions','losses','image_indices')}
                atomic_json(output/(name+'-final-shuffle.json'), dict(target_ids=[rows[i]['id'] for i in validation_indices],
                            source_ids=[rows[i]['id'] for i in result['image_indices']], **result))
            summary = dict(status='completed' if updates == plan['max_epochs']*len(train)//plan['global_batch'] else 'time_limited',
                job_id=os.environ['SLURM_JOB_ID'], base_unchanged=True, epochs_per_arm=updates*plan['global_batch']/len(train),
                updates_per_arm=updates, unique_training_questions=min(len(train),updates*plan['global_batch']),
                presentations_per_arm=updates*plan['global_batch'], validation_questions=len(val),
                best_by_validation_loss=best, final=last, final_shuffled_images=shuffled,
                trainable_parameters={n:sum(p.numel() for p in a.parameters() if p.requires_grad) for n,a in arms.items()},
                limitation='Direct classifier input screen. Not a Slot Abstractor reproduction or a frozen-LLM benchmark. One seed.',
                feature_and_training_seconds=time.time()-prepared)
            atomic_json(output/'summary.json', summary)
            print(json.dumps(summary), flush=True)
            for cache in caches.values(): cache._mmap.close()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--plan', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--deadline', type=float, required=True)
    args = p.parse_args()
    try:
        run(json.loads(args.plan.read_text()), args.output, args.deadline)
    except Exception as exc:
        if args.output.is_dir():
            atomic_json(args.output/'failure.json', dict(error=repr(exc), time=time.time()))
        raise


if __name__ == '__main__':
    main()
