"""Inference-only control for the cached-teacher InternVL interleaving campaign.

The primary rule follows experiments/diag_overlap/analyze.py: equal weights on
log-softmax scores. Arithmetic probability averaging is also recorded, without
selecting a weight or temperature on validation. Neither rule is calibrated.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import time
import traceback

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def log_softmax(scores):
    scores = np.asarray(scores, dtype=np.float64)
    if scores.ndim != 2 or scores.shape[1] != 8 or not np.isfinite(scores).all():
        raise ValueError('Expected finite question-by-eight scores')
    shifted = scores - scores.max(axis=1, keepdims=True)
    return shifted - np.log(np.exp(shifted).sum(axis=1, keepdims=True))


def paired_counts(first, second):
    return dict(both_correct=int((first & second).sum()),
                first_only=int((first & ~second).sum()),
                second_only=int((~first & second).sum()),
                neither_correct=int((~first & ~second).sum()))


def summarize(records, hybrid):
    ids = [r['id'] for r in records]
    references = hybrid['predictions']
    if not records or len(set(ids)) != len(ids):
        raise ValueError('Empty or duplicate baseline IDs')
    ref_ids = [r['id'] for r in references]
    if (not hybrid.get('complete') or hybrid.get('count') != len(records)
            or len(ref_ids) != len(records) or len(set(ref_ids)) != len(ref_ids)
            or set(ref_ids) != set(ids)):
        raise ValueError('Hybrid must cover exactly the same complete validation IDs')
    lookup = {r['id']: r for r in references}
    targets = np.asarray([r['target'] for r in records])
    if targets.dtype.kind not in 'iu' or not ((targets >= 0) & (targets < 8)).all():
        raise ValueError('Invalid candidate labels')
    if any(lookup[r['id']]['target'] != r['target'] for r in records):
        raise ValueError('Hybrid/baseline labels differ')
    lora = log_softmax([r['lora_scores'] for r in records])
    abstractor = log_softmax([r['abstractor_scores'] for r in records])
    scores = dict(lora=lora, abstractor=abstractor,
                  average_log_probability=log_softmax(.5 * (lora + abstractor)),
                  average_probability=np.logaddexp(lora, abstractor) - np.log(2.),
                  hybrid=log_softmax([lookup[key]['scores'] for key in ids]))
    metrics, correct = {}, {}
    for name, lp in scores.items():
        predictions = lp.argmax(axis=1)
        correct[name] = predictions == targets
        metrics[name] = dict(count=len(ids), correct=int(correct[name].sum()),
                            accuracy=float(correct[name].mean()),
                            loss=float(-lp[np.arange(len(ids)), targets].mean()))
    return dict(complete=True, split='validation', benchmark_test=False,
                count=len(ids), metrics=metrics, hybrid_update=hybrid['update'],
                primary_rule='argmax(0.5*log_softmax(lora)+0.5*log_softmax(abstractor))',
                secondary_rule='argmax(0.5*softmax(lora)+0.5*softmax(abstractor))',
                tuned_weights=False, calibrated_temperatures=False,
                overlap_lora_abstractor=paired_counts(correct['lora'], correct['abstractor']),
                overlap_average_hybrid=paired_counts(correct['average_log_probability'], correct['hybrid']),
                hybrid_minus_average_percentage_points=float(
                    100 * (correct['hybrid'].mean() - correct['average_log_probability'].mean())),
                limitation='Same validation IDs used to select the hybrid; not a held-out test result.')


def run(plan, output):
    import torch
    from PIL import Image
    from peft import get_peft_model_state_dict
    from safetensors.torch import load_file
    from pgmvlm.checkpoints import atomic_json
    from pgmvlm.data import CandidateCollator, PROMPT_TEXT
    from pgmvlm.datasets import read_manifest
    from pgmvlm.extensions.probe_diagnostics import tensor_digest
    from pgmvlm.models import load_base, load_adapter, ScoreTokens, candidate_scores
    from pgmvlm.pretrained_relations import FrozenRelations, checked, sha256
    from pgmvlm.storage import require_headroom
    from tools.input_representation_pilot import resolve_image
    from tools.internvl_connector_pilot import candidate_item
    from tools.train_pretrained_connectors import Banks

    if not os.environ.get('SLURM_JOB_ID') or not torch.cuda.is_available():
        raise RuntimeError('Requires a GPU allocation')
    if plan['validation_questions'] != 1000 or plan['benchmark_test'] is not False:
        raise ValueError('This campaign authorizes only the matched 1,000-question validation control')
    if (output / 'config.json').exists():
        raise ValueError('Refusing to overwrite an existing evaluation')
    atomic_json(output / 'config.json', plan)
    atomic_json(output / 'storage.json', require_headroom(output, reserve_files=50))
    start = time.monotonic()
    deadline = start + plan['evaluation_seconds']
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.manual_seed(plan['teacher_seed'])
    torch.cuda.manual_seed_all(plan['teacher_seed'])
    for split in ('train', 'val'):
        checked(plan[f'{split}_manifest'], plan[f'{split}_manifest_sha256'])
    train = read_manifest(plan['train_manifest'])
    rows = read_manifest(plan['val_manifest'])
    if len(train) != 60000 or len(rows) != 1000 or len({r['id'] for r in train + rows}) != 61000:
        raise ValueError('Changed parent manifests or duplicate/overlapping IDs')
    if any(r['dataset'] != 'pgm' or r['regime'] != 'neutral' or r['split'] != 'validation'
           or r['choices'] != list('ABCDEFGH') for r in rows):
        raise ValueError('Unexpected validation task or candidate order')
    checked(plan['hybrid_reference'], plan['hybrid_reference_sha256'])
    hybrid = json.loads(Path(plan['hybrid_reference']).read_text())
    banks = Banks(plan['cache_output'], train + rows, plan, capacity=8)
    teacher = FrozenRelations(plan['teacher_package'], plan['teacher_hashes'])
    teacher_params = list(teacher.encoder.named_parameters()) + list(teacher.scorer.named_parameters())
    teacher_before = tensor_digest(teacher_params)
    # Test the saved features against the original reference implementation,
    # including its original classifier and candidate-permutation checks.
    first = rows[0]
    checked(resolve_image(first, plan), first['metadata']['contact_sheet_sha256'])
    with Image.open(resolve_image(first, plan)) as image:
        fresh = teacher.extract(image, first['id'], plan['teacher_seed'], verify=True)
    cached = banks[first['id']]
    torch.testing.assert_close(fresh, cached, rtol=1e-5, atol=1e-5)
    head = teacher.scorer.abstractors.mlp_head
    with torch.no_grad():
        torch.testing.assert_close(head(fresh.cuda().mean(2)), head(cached.cuda().mean(2)),
                                   rtol=1e-5, atol=1e-5)
    for name, expected in plan['adapter_hashes'].items():
        checked(Path(plan['adapter']) / name, expected)
    device = torch.device('cuda')
    model, processor, family = load_base(plan['model'], device=device)
    if family != 'internvl':
        raise ValueError('Expected InternVL')
    model = load_adapter(model, plan['adapter'], trainable=False)
    # Match the interleaving run's FP32 adapter arithmetic, without its connector.
    for name, parameter in model.named_parameters():
        if 'lora_' in name:
            parameter.data = parameter.data.float()
    model.eval().requires_grad_(False)
    saved = load_file(str(Path(plan['adapter']) / 'adapter_model.safetensors'))
    loaded = get_peft_model_state_dict(model)
    if set(saved) != set(loaded):
        raise ValueError('Adapter tensor keys differ')
    for name, value in saved.items():
        torch.testing.assert_close(value.to(loaded[name]), loaded[name], rtol=0, atol=0)
    del saved, loaded
    before = tensor_digest(model.named_parameters())
    collator = CandidateCollator(processor, family, crop_to_patches=True)
    tokens = ScoreTokens(processor.tokenizer)
    import peft, transformers
    atomic_json(output / 'backend.json', dict(torch=torch.__version__, peft=peft.__version__,
        transformers=transformers.__version__, gpu=torch.cuda.get_device_name(),
        prompt=PROMPT_TEXT, relation_tokens=0, connector_installed=False,
        visual_input='8 completed 520x520 matrices; native crop_to_patches=True',
        scoring='FP32 Yes-minus-No from final hidden states',
        cache_ready_sha256=sha256(Path(plan['cache_output']) / 'READY.json'),
        model_revision=plan['model_revision'], base_and_lora_sha256=before,
        teacher_sha256=teacher_before))
    records = []
    with torch.no_grad(), (output / 'predictions.jsonl').open('x') as stream:
        for index, row in enumerate(rows):
            if time.monotonic() >= deadline:
                raise TimeoutError('Inference budget expired; partial predictions are not a full result')
            item = candidate_item(row, plan)
            batch = collator([item])
            lora = candidate_scores(model, batch, tokens, device, torch.bfloat16)
            abstractor = head(banks[row['id']].to(device).mean(2)).squeeze(-1)
            if lora.shape != (1, 8) or abstractor.shape != (1, 8):
                raise ValueError('Unexpected candidate score shape')
            if not torch.isfinite(lora).all() or not torch.isfinite(abstractor).all():
                raise RuntimeError('Nonfinite scores')
            if index == 0:
                permutation = [6, 2, 0, 7, 3, 5, 1, 4]
                permuted = collator([{**item, 'candidates': item['candidates'][permutation]}])
                actual = candidate_scores(model, permuted, tokens, device, torch.bfloat16)
                torch.testing.assert_close(actual, lora[:, permutation], rtol=1e-5, atol=1e-5)
                atomic_json(output / 'checks.json', dict(passed=True, original_adapter_exact=True,
                    teacher_reference_parity=True, cache_feature_parity=True,
                    original_teacher_head_reconstruction=True, candidate_permutation=True))
            record = dict(id=row['id'], target=item['target'], lora_scores=lora[0].tolist(),
                          abstractor_scores=abstractor[0].tolist())
            records.append(record)
            stream.write(json.dumps(record, allow_nan=False) + '\n')
            if (index + 1) % 50 == 0:
                stream.flush()
                progress = dict(count=index + 1, total=len(rows), complete=False,
                                seconds=time.monotonic() - start)
                atomic_json(output / 'progress.json', progress)
                print(json.dumps(progress), flush=True)
        stream.flush()
        os.fsync(stream.fileno())
    if before != tensor_digest(model.named_parameters()) or teacher_before != tensor_digest(teacher_params):
        raise RuntimeError('Frozen weights changed')
    if any(p.requires_grad or p.grad is not None for p in model.parameters()):
        raise RuntimeError('VLM was not frozen')
    result = summarize(records, hybrid)
    result.update(seconds=time.monotonic() - start, frozen_weights_unchanged=True,
                  predictions_sha256=sha256(output / 'predictions.jsonl'),
                  hybrid_reference_sha256=plan['hybrid_reference_sha256'])
    atomic_json(output / 'summary.json', result)
    print(json.dumps(result), flush=True)


def main():
    from pgmvlm.checkpoints import RunLock, atomic_json
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    output = Path(args.output)
    with RunLock(output):
        try:
            run(json.loads(Path(args.plan).read_text()), output)
        except Exception:
            atomic_json(output / 'failure.json', dict(complete=False, error=traceback.format_exc()))
            raise


if __name__ == '__main__':
    main()
