"""Bounded correctness pilot for frozen InternVL + external pretrained relations.

Reuses the verified teacher cache and the existing candidate-scoring pipeline.
No scheduler operations, full benchmark, or automatic larger training run.
"""
from __future__ import annotations
import argparse
import copy
import json
import os
from pathlib import Path
import sys
import time
import traceback

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from PIL import Image
from pgmvlm.checkpoints import RunLock, atomic_json
from pgmvlm.data import CandidateCollator, PROMPT_TEXT
from pgmvlm.datasets import read_manifest
from pgmvlm.extensions.internvl_connectors import install
from pgmvlm.extensions.pretrained_connectors import DESIGNS
from pgmvlm.extensions.probe_diagnostics import tensor_digest
from pgmvlm.models import load_base, load_adapter, ScoreTokens, candidate_scores
from pgmvlm.panels import split_composite
from pgmvlm.pretrained_relations import checked, sha256
from pgmvlm.storage import require_headroom
from tools import pretrained_connector_pilot as common
from tools.train_pretrained_connectors import Banks
from tools.input_representation_pilot import resolve_image


def candidate_item(row, plan):
    path = resolve_image(row, plan)
    checked(path, row['metadata']['contact_sheet_sha256'])
    with Image.open(path) as image:
        panels = np.stack([np.asarray(p.convert('L')) for p in split_composite(image)])
    return dict(context=panels[:8], candidates=panels[8:],
                target=ord(row['answer']) - ord('A'), file=row['id'])


def run(plan, out, deadline):
    if not os.environ.get('SLURM_JOB_ID') or not torch.cuda.is_available():
        raise RuntimeError('Requires a GPU allocation')
    if plan['train_questions'] != 8 or plan['validation_questions'] != 4 or plan['max_updates'] != 8:
        raise ValueError('This entry point only authorizes the 8/4-question correctness pilot')
    if plan['designs'] != list(DESIGNS):
        raise ValueError('Expected all three connectors')
    if (out / 'config.json').exists():
        raise ValueError('Refuse to restart an existing pilot')
    atomic_json(out / 'config.json', plan)
    atomic_json(out / 'storage.json', require_headroom(out, reserve_files=2000))
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    device = torch.device('cuda')
    for split in ('train', 'val'):
        checked(plan[f'{split}_manifest'], plan[f'{split}_manifest_sha256'])
    train = read_manifest(plan['train_manifest'])
    val = read_manifest(plan['val_manifest'])
    if len(train) != 60000 or len(val) != 1000:
        raise ValueError('Changed parent subset')
    banks = Banks(plan['cache_output'], train + val, plan, capacity=4)
    train, val = train[:8], val[:4]
    if set(r['id'] for r in train) & set(r['id'] for r in val):
        raise ValueError('Train/validation overlap')
    rows = train + val
    atomic_json(out / 'question_ids.json', dict(train=[r['id'] for r in train],
        validation=[r['id'] for r in val], parent_train=plan['train_manifest_sha256'],
        parent_validation=plan['val_manifest_sha256']))
    items = {r['id']: candidate_item(r, plan) for r in rows}
    memories = {r['id']: banks[r['id']][0].to(device) for r in rows}
    for name, expected in plan['adapter_hashes'].items():
        checked(Path(plan['adapter']) / name, expected)
    model, processor, family = load_base(plan['model'], device=device)
    if family != 'internvl':
        raise ValueError('Wrong base model')
    model = load_adapter(model, plan['adapter'], trainable=False)
    model.eval().requires_grad_(False)
    base = model.get_base_model()
    from peft import get_peft_model_state_dict
    from safetensors.torch import load_file
    stored = load_file(str(Path(plan['adapter']) / 'adapter_model.safetensors'))
    loaded = get_peft_model_state_dict(model)
    if set(stored) != set(loaded):
        raise ValueError('Adapter tensor keys differ')
    for key, value in stored.items():
        torch.testing.assert_close(value.to(loaded[key]), loaded[key], rtol=0, atol=0)
    del stored, loaded
    frozen = tensor_digest(common.base_params(model))
    collator = CandidateCollator(processor, family, crop_to_patches=True)
    st = ScoreTokens(processor.tokenizer)
    batches = {r['id']: collator([items[r['id']]]) for r in rows}
    import transformers, peft
    atomic_json(out / 'backend.json', dict(torch=torch.__version__, transformers=transformers.__version__,
        peft=peft.__version__, gpu=torch.cuda.get_device_name(), prompt=PROMPT_TEXT,
        visual_input='8 completed 520x520 matrices; native processor crop_to_patches=True',
        objective='8-way CE over FP32 Yes-minus-No scores from final hidden states',
        teacher_input='cached 8x9x32 features; one matching 9x32 bank per candidate',
        model_revision=plan['model_revision'], base_and_lora_sha256=frozen,
        cache_ready_sha256=sha256(Path(plan['cache_output']) / 'READY.json')))

    def native_scores(batch):
        if time.monotonic() >= deadline:
            raise TimeoutError('Pilot time limit')
        return candidate_scores(model, batch, st, device, torch.bfloat16)

    first = train[0]['id']
    with torch.no_grad(): baseline = native_scores(batches[first]).detach()
    summary = dict(passed=False, arms={})
    atomic_json(out / 'summary.json', summary)
    for design in DESIGNS:
        print(f'Starting {design}', flush=True)
        arm = out / design
        arm.mkdir()
        torch.manual_seed(plan['initialization_seed'])
        torch.cuda.manual_seed_all(plan['initialization_seed'])
        ctl = install(base, design, **plan['connector'])
        ctl.module.train()
        initial = copy.deepcopy(ctl.module.state_dict())
        atomic_json(arm / 'config.json', dict(spec=ctl.module.spec,
            trainable_parameters=sum(p.numel() for p in ctl.module.parameters())))

        def scores(key):
            with ctl.question(memories[key]):
                return native_scores(batches[key])

        def step(key, opt):
            opt.zero_grad(set_to_none=True)
            score = scores(key)
            loss = torch.nn.functional.cross_entropy(score, batches[key]['targets'].to(device))
            if not torch.isfinite(loss):
                raise RuntimeError('Nonfinite candidate CE')
            loss.backward()
            gradients = common.gradient_report(ctl.module)
            norm = torch.nn.utils.clip_grad_norm_(ctl.module.parameters(), 1., error_if_nonfinite=True)
            opt.step()
            return dict(loss=float(loss.detach()), correct=int(score.argmax() == items[key]['target']),
                gradient_rms=gradients, gradient_norm=float(norm))

        @torch.no_grad()
        def evaluate(selected):
            records = []
            for row in selected:
                key = row['id']
                score = scores(key)
                loss = torch.nn.functional.cross_entropy(score, batches[key]['targets'].to(device))
                if not torch.isfinite(loss) or not torch.isfinite(score).all():
                    raise RuntimeError('Nonfinite evaluation')
                records.append(dict(id=key, scores=score[0].tolist(), prediction=int(score.argmax()),
                    target=items[key]['target'], loss=float(loss), correct=int(score.argmax()) == items[key]['target']))
            return dict(count=len(records), accuracy=sum(r['correct'] for r in records)/len(records),
                loss=sum(r['loss'] for r in records)/len(records), predictions=records)

        try:
            with torch.no_grad():
                ctl.set_enabled(False)
                torch.testing.assert_close(scores(first), baseline, rtol=0, atol=0)
                ctl.set_enabled(True)
                if design != 'tokens':
                    torch.testing.assert_close(scores(first), baseline, rtol=0, atol=0)
            opt = torch.optim.AdamW(ctl.module.parameters(), lr=plan['lr'], weight_decay=plan['weight_decay'])
            for _ in range(3): audit = step(first, opt)
            if audit['gradient_rms'].get('input.weight', 0.) <= 0:
                raise RuntimeError('No gradient to the relation input projector after 3 updates')
            common.save(arm / 'replay.pt', ctl, opt, plan, 3)
            with torch.no_grad(): expected = scores(first).clone()
            step(first, opt)
            updated = copy.deepcopy(ctl.module.state_dict())
            # Reinstall hooks as well as reloading disk weights and optimizer.
            ctl.remove()
            ctl = install(base, design, **plan['connector'])
            opt = torch.optim.AdamW(ctl.module.parameters(), lr=plan['lr'], weight_decay=plan['weight_decay'])
            common.restore(arm / 'replay.pt', ctl, opt, plan)
            with torch.no_grad(): torch.testing.assert_close(scores(first), expected, rtol=0, atol=0)
            step(first, opt)
            for key, value in ctl.module.state_dict().items():
                torch.testing.assert_close(value, updated[key], rtol=0, atol=0)
            # Rebuild pixels AND features after permutation; no candidate-index shortcuts.
            permutation = [6, 2, 0, 7, 3, 5, 1, 4]
            item = {**items[first], 'candidates': items[first]['candidates'][permutation]}
            permuted = collator([item])
            with torch.no_grad():
                reference = scores(first)[:, permutation]
                with ctl.question(memories[first][permutation]): actual = native_scores(permuted)
                torch.testing.assert_close(actual, reference, rtol=1e-5, atol=1e-5)
            ctl.set_enabled(False)
            with torch.no_grad(): torch.testing.assert_close(scores(first), baseline, rtol=0, atol=0)
            ctl.set_enabled(True)
            common.check_frozen(model, frozen)
            atomic_json(arm / 'checks.json', dict(passed=True, disabled_exact=True, reload_exact=True,
                next_update_exact=True, candidate_permutation=True, deep_gradient=True,
                frozen_weights_unchanged=True, audit_gradient_rms=audit['gradient_rms']))
            ctl.module.load_state_dict(initial)
            opt = torch.optim.AdamW(ctl.module.parameters(), lr=plan['lr'], weight_decay=plan['weight_decay'])
            before = evaluate(val)
            atomic_json(arm / 'initial-validation.json', before)
            for update, row in enumerate(train, 1):
                result = step(row['id'], opt)
                with (arm / 'curve.jsonl').open('a') as stream:
                    stream.write(json.dumps(dict(update=update, id=row['id'], **result)) + '\n')
            train_result, after = evaluate(train), evaluate(val)
            atomic_json(arm / 'train-result.json', train_result)
            atomic_json(arm / 'validation.json', after)
            common.save(arm / 'connector.pt', ctl, opt, plan, len(train))
            common.check_frozen(model, frozen)
            summary['arms'][design] = dict(passed=True, updates=len(train), train=train_result,
                validation_before=before, validation_after=after)
            atomic_json(out / 'summary.json', summary)
        finally:
            ctl.remove()
            torch.cuda.empty_cache()
    summary['passed'] = len(summary['arms']) == len(DESIGNS)
    atomic_json(out / 'summary.json', summary)
    print('ALL THREE INTERNVL CONNECTOR PILOTS PASSED', flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--plan', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--seconds', type=int, default=1500)
    args = p.parse_args()
    if not 0 < args.seconds <= 1500:
        p.error('Pilot cap is 25 minutes inside one 30-minute allocation')
    output = Path(args.output)
    with RunLock(output):
        try:
            run(json.loads(Path(args.plan).read_text()), output, time.monotonic() + args.seconds)
        except Exception as exc:
            atomic_json(output / 'failure.json', dict(error=str(exc), traceback=traceback.format_exc()))
            raise
