"""Two-hour paired native-LFM pilot; no new jobs or automatic continuation.

Both arms advance on the same question batch, with independent identical
Abstractors/optimizers and one shared frozen VLM. Full evaluation is reserved
at the end; partial evaluations and partial epochs are explicitly recorded.
"""
from __future__ import annotations

import argparse
from collections import Counter
import copy
import fcntl
import json
import os
from pathlib import Path
import random
import signal
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from PIL import Image
from pgmvlm.extensions.relational import install
from pgmvlm.extensions.probe_diagnostics import tensor_digest
from pgmvlm.models import load_base, ScoreTokens
from pgmvlm.protocol_probe import (candidate_images, candidate_ce, score_matrices,
                                   question_result, paired_order)
from pgmvlm.storage import require_headroom
from tools.input_representation_pilot import (load_selection, resolve_image, atomic_json,
                                             checkpoint, append)

STOP = False
NAMES = ('answer_letter', 'candidate')


def stop(signum, frame):
    global STOP
    STOP = True


def base_parameters(model):
    return [(n, p) for n, p in model.named_parameters() if not n.startswith('relation_probe.')]


def activate(model, controller, arms, name):
    model.relation_probe = arms[name]
    controller.probe = arms[name]
    controller.set_enabled(True)
    arms[name].train()


def read_image(row, plan):
    with Image.open(resolve_image(row, plan)) as im:
        return im.convert('RGB')


def evaluate_pair(model, processor, controller, arms, tokens, letter_ids, rows,
                  plan, output, label, updates, hard_deadline, generate=False):
    results = {name: [] for name in NAMES}
    started = time.time()
    # Pair every evaluated question too: a deadline cannot give unequal ID sets.
    with torch.no_grad():
        for row in rows:
            if STOP or time.time() > hard_deadline - 30:
                break
            image = read_image(row, plan)
            for name in paired_order(NAMES, len(results[NAMES[0]])):
                activate(model, controller, arms, name)
                arms[name].eval()
                result = question_result(model, processor, image, ord(row['answer'])-65,
                                         name, tokens, letter_ids, generate=generate)
                result['loss'] = float(result['loss'])
                result.update(id=row['id'], target=ord(row['answer'])-65)
                results[name].append(result)
    metrics = {}
    for name, records in results.items():
        n = len(records)
        if not n:
            metrics[name] = dict(count=0, complete=False, planned=len(rows))
            continue
        value = dict(count=n, complete=n==len(rows), planned=len(rows),
                     accuracy=sum(r['pred']==r['target'] for r in records)/n,
                     loss=sum(r['loss'] for r in records)/n,
                     prediction_histogram=dict(Counter(str(r['pred']) for r in records)))
        if name == 'answer_letter':
            value.update(answer_loss=sum(r['answer_loss'] for r in records)/n,
                         first_token_accuracy=sum(r['first_token_pred']==r['target'] for r in records)/n,
                         invalid=sum(r['pred']==-1 for r in records), greedy=generate)
        metrics[name] = value
        atomic_json(output/f'{label}-{name}-{updates:05d}.json',
                    dict(summary=value, predictions=records))
    entry = dict(label=label, updates=updates, epoch=updates*plan['global_batch']/4096,
                 seconds=time.time()-started, metrics=metrics)
    append(output/'evaluation-curve.jsonl', entry)
    print(json.dumps(entry), flush=True)
    return entry


def save_pair(output, arms, optimizers, updates, plan, tag):
    require_headroom(output)
    for name in NAMES:
        checkpoint(output/f'{name}-{tag}.pt', dict(
            module=arms[name].state_dict(), optimizer=optimizers[name].state_dict(),
            updates=updates, presentations=updates*plan['global_batch'],
            config=plan, protocol=name, torch_rng=torch.get_rng_state(),
            cuda_rng=torch.cuda.get_rng_state_all()))


def pilot_checks(model, processor, controller, arms, tokens, letter_ids, row, plan, output):
    initial = copy.deepcopy(arms[NAMES[0]].state_dict())
    image = read_image(row, plan)
    target = ord(row['answer'])-65
    checks = []
    for name in NAMES:
        activate(model, controller, arms, name)
        started = time.time()
        with torch.no_grad():
            controller.set_enabled(False)
            native = question_result(model, processor, image, target, name, tokens, letter_ids, generate=True)
            controller.set_enabled(True)
            zero = question_result(model, processor, image, target, name, tokens, letter_ids, generate=True)
        torch.testing.assert_close(native['loss'], zero['loss'], rtol=0, atol=0)
        if native['pred'] != zero['pred'] or native.get('raw_output') != zero.get('raw_output'):
            raise RuntimeError('Initial module changes the native prediction')
        opt = torch.optim.AdamW(arms[name].parameters(), lr=plan['lr'], weight_decay=plan['weight_decay'])
        for step in range(2):
            opt.zero_grad(set_to_none=True)
            result = question_result(model, processor, image, target, name, tokens, letter_ids)
            result['loss'].backward()
            norm = torch.nn.utils.clip_grad_norm_(arms[name].parameters(), plan['clip_norm'],
                                                 error_if_nonfinite=True)
            if not norm > 0:
                raise RuntimeError('Disconnected or zero module gradients')
            opt.step()
        deep = arms[name].content_in[1].weight.grad
        if deep is None or not torch.isfinite(deep).all() or not deep.abs().max() > 0:
            raise RuntimeError('No content-projection gradient after opening writeback')
        if name == 'candidate':
            matrices = candidate_images(image)
            permutation = [3, 7, 0, 6, 2, 1, 5, 4]
            with torch.no_grad():
                a = score_matrices(model, processor, matrices, tokens)
                b = score_matrices(model, processor, [matrices[i] for i in permutation], tokens)
            torch.testing.assert_close(b, a[permutation], rtol=0, atol=1e-5)
            torch.testing.assert_close(candidate_ce(a, target), candidate_ce(b, permutation.index(target)),
                                       rtol=0, atol=1e-5)
        checks.append(dict(protocol=name, initial_noop=True, finite_nonzero_gradients=True,
                           seconds=time.time()-started,
                           content_gradient_rms=float(deep.float().square().mean().sqrt())))
        arms[name].load_state_dict(initial)
        arms[name].zero_grad(set_to_none=True)
    if any(p.requires_grad or p.grad is not None for _, p in base_parameters(model)):
        raise RuntimeError('A frozen VLM parameter was made trainable or received a gradient')
    atomic_json(output/'pilot-result.json', dict(status='passed', checks=checks,
                candidate_permutation_equivariance=True, pilot_updates_discarded=True,
                initial_module_sha256=tensor_digest(arms[NAMES[0]].named_parameters())))


def run(plan, output, deadline):
    if not os.environ.get('SLURM_JOB_ID') or not torch.cuda.is_available():
        raise RuntimeError('Run only inside an allocated GPU job')
    output.mkdir(parents=True, exist_ok=True)
    with (output/'.run.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (output/'config.json').exists():
            raise RuntimeError('Existing output; this pilot does not silently restart or overwrite')
        atomic_json(output/'config.json', plan)
        atomic_json(output/'storage-preflight.json', require_headroom(output))
        for sig in (signal.SIGUSR1, signal.SIGTERM, signal.SIGINT): signal.signal(sig, stop)
        torch.set_num_threads(2)
        torch.manual_seed(plan['initialization_seed'])
        torch.use_deterministic_algorithms(True)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        train, val, monitor = load_selection(plan)
        # Prefixes preserve the earlier monitor; no selection depends on outcomes.
        screen = val[:plan['screen_questions']]
        monitor_pool = [train[i] for i in monitor]
        atomic_json(output/'question_ids.json', dict(train=[r['id'] for r in train],
            validation=[r['id'] for r in val], screen=[r['id'] for r in screen],
            monitor_pool=[r['id'] for r in monitor_pool],
            monitor_selection='First at most 128 historical monitor IDs actually seen during training'))
        model, processor, family = load_base('lfm25', revision=plan['revision'], device='cuda')
        model.eval().requires_grad_(False)
        before_hash = tensor_digest(base_parameters(model))
        tokens = ScoreTokens(processor.tokenizer)
        if not tokens.single: raise ValueError('Yes and No must each be one native token')
        letters = [processor.tokenizer.encode(x, add_special_tokens=False) for x in 'ABCDEFGH']
        if any(len(x)!=1 for x in letters): raise ValueError('Letter tokenization is not single-token')
        letter_ids = [x[0] for x in letters]
        torch.manual_seed(plan['initialization_seed'])
        controller = install(model, processor, 'late', width=256, heads=4, depth=2, symbols=128)
        arms = {'answer_letter': model.relation_probe, 'candidate': copy.deepcopy(model.relation_probe)}
        atomic_json(output/'backend.json', dict(torch=torch.__version__, gpu=torch.cuda.get_device_name(),
            family=family, yes_id=tokens.yes_id, no_id=tokens.no_id, letter_ids=letter_ids,
            trainable_parameters={n:sum(p.numel() for p in a.parameters()) for n,a in arms.items()},
            base_parameters=sum(p.numel() for _,p in base_parameters(model))))
        pilot_checks(model, processor, controller, arms, tokens, letter_ids, train[0], plan, output)
        optimizers = {n: torch.optim.AdamW(a.parameters(), lr=plan['lr'], weight_decay=plan['weight_decay'])
                      for n,a in arms.items()}
        # Initial full native generation also measures a conservative final-eval reserve.
        initial = evaluate_pair(model, processor, controller, arms, tokens, letter_ids, screen,
                                plan, output, 'initial-validation-screen', 0, deadline, generate=True)
        if not all(v.get('complete') for v in initial['metrics'].values()):
            raise RuntimeError('Initial screen incomplete; no training comparison claimed')
        reserve = max(plan['minimum_reserve_seconds'],
                      initial['seconds']/len(screen)*(len(val)+plan['train_monitor_questions'])*1.5 + 180)
        atomic_json(output/'budget.json', dict(measured_final_eval_reserve_seconds=reserve,
                    initial_screen_seconds=initial['seconds'], deadline=deadline))
        updates = 0
        presentation_ids = set()
        latest_screen = initial
        pair_times = []
        timing = {n:0. for n in NAMES}
        stop_reason = 'epoch_limit'
        for epoch in range(plan['max_epochs']):
            order = list(range(len(train)))
            random.Random(plan['training_seed']+epoch).shuffle(order)
            for begin in range(0, len(order), plan['global_batch']):
                # Budget before a complete paired update, never between the two arms.
                safety = max(60., max(pair_times[-4:], default=0.)*1.5)
                if STOP or time.time()+safety > deadline-reserve:
                    stop_reason = 'signal' if STOP else 'time_budget'
                    break
                rows = [train[i] for i in order[begin:begin+plan['global_batch']]]
                images = [read_image(r, plan) for r in rows]
                paired_start = time.time()
                for name in paired_order(NAMES, updates):
                    activate(model, controller, arms, name)
                    opt = optimizers[name]
                    opt.zero_grad(set_to_none=True)
                    lr = plan['lr']*min(1., (updates+1)/plan['warmup_updates'])
                    for group in opt.param_groups: group['lr'] = lr
                    losses, answer_losses, correct = [], [], 0
                    started = time.time()
                    for image,row in zip(images, rows):
                        target = ord(row['answer'])-65
                        result = question_result(model, processor, image, target, name, tokens, letter_ids)
                        loss = result['loss']
                        if not torch.isfinite(loss): raise RuntimeError('Nonfinite loss')
                        (loss/len(rows)).backward()
                        losses.append(float(loss.detach()))
                        if 'answer_loss' in result: answer_losses.append(result['answer_loss'])
                        correct += result['pred']==target
                    grad = torch.nn.utils.clip_grad_norm_(arms[name].parameters(), plan['clip_norm'],
                                                         error_if_nonfinite=True)
                    opt.step()
                    elapsed = time.time()-started
                    timing[name] += elapsed
                    append(output/'learning-curve.jsonl', dict(protocol=name, update=updates+1,
                        presentations=(updates+1)*plan['global_batch'], epoch=(updates+1)*plan['global_batch']/4096,
                        mean_loss=sum(losses)/len(losses), accuracy=correct/len(rows),
                        answer_loss=sum(answer_losses)/len(answer_losses) if answer_losses else None,
                        learning_rate=lr, gradient_norm=float(grad), seconds=elapsed,
                        question_ids=[r['id'] for r in rows]))
                updates += 1
                presentation_ids.update(r['id'] for r in rows)
                pair_times.append(time.time()-paired_start)
                if updates % 16 == 0:
                    progress = dict(phase='training', updates=updates, presentations=updates*plan['global_batch'],
                        epochs=updates*plan['global_batch']/4096, unique_questions=len(presentation_ids),
                        training_seconds=timing, remaining_seconds=deadline-time.time())
                    atomic_json(output/'progress.json', progress); print(json.dumps(progress), flush=True)
                if updates % plan['screen_every_updates'] == 0:
                    save_pair(output, arms, optimizers, updates, plan, 'latest')
                    # Every screen uses the same paired IDs; it does not select winners.
                    if time.time()+initial['seconds']*1.2 < deadline-reserve:
                        latest_screen = evaluate_pair(model, processor, controller, arms, tokens, letter_ids,
                            screen, plan, output, 'validation-screen', updates, deadline-reserve, generate=False)
            if stop_reason != 'epoch_limit': break
        save_pair(output, arms, optimizers, updates, plan, 'latest')
        final = evaluate_pair(model, processor, controller, arms, tokens, letter_ids, val,
                              plan, output, 'final-validation', updates, deadline, generate=True)
        monitor = [r for r in monitor_pool if r['id'] in presentation_ids][:plan['train_monitor_questions']]
        training = evaluate_pair(model, processor, controller, arms, tokens, letter_ids, monitor,
                                 plan, output, 'final-train-monitor', updates, deadline, generate=True)
        after_hash = tensor_digest(base_parameters(model))
        if before_hash != after_hash or any(p.grad is not None for _,p in base_parameters(model)):
            raise RuntimeError('Frozen original VLM changed')
        complete = all(v.get('complete') for v in final['metrics'].values())
        status = ('completed' if complete else 'partial_evaluation') if updates else 'no_training_budget'
        summary = dict(status=status, stop_reason=stop_reason,
            job_id=os.environ['SLURM_JOB_ID'], updates_per_arm=updates,
            presentations_per_arm=updates*plan['global_batch'], epochs_per_arm=updates*plan['global_batch']/4096,
            unique_training_questions=len(presentation_ids), training_pool_questions=4096,
            validation_questions=1000, base_bitwise_unchanged=True, base_sha256=after_hash,
            benchmark_test=False, initial_screen=initial, final_validation=final,
            final_training_monitor=training, training_seconds=timing,
            checkpoint_selection='final common update; no cross-protocol loss selection',
            limitation='One seed; protocol changes image composition, prompt and objective together. '
                       'Question exposure matched, not FLOPs; partial epochs possible in two hours.')
        atomic_json(output/'summary.json', summary)
        atomic_json(output/'progress.json', summary)
        print(json.dumps(summary), flush=True)


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--plan', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--deadline', type=float, required=True)
    a=p.parse_args()
    try:
        run(json.loads(a.plan.read_text()), a.output, a.deadline)
    except Exception as exc:
        a.output.mkdir(parents=True, exist_ok=True)
        atomic_json(a.output/'failure.json', dict(type=type(exc).__name__, message=str(exc)))
        raise


if __name__=='__main__': main()
