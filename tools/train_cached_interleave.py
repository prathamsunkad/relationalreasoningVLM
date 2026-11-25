"""Exact repository interleave, existing InternVL LoRA, on-demand frozen features.

One pass over a fixed 240k subset. The 60k cached prefix is visited first; missing
features are computed only when a question is reached and committed in shards.
No scheduler actions. Pilot and training use the same optimizer/checkpoint path.
"""
from __future__ import annotations
import argparse
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import sys
import time
import traceback
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'relational'))
import numpy as np
import torch
from PIL import Image
from safetensors.torch import save_file, load_file
from peft import get_peft_model_state_dict, set_peft_model_state_dict
from backbones import Backbone
from cached_interleave import CachedInterleave
from data import Collator, render_grid
from pgmvlm.checkpoints import RunLock, atomic_json, commit, resume_checkpoint, pointer_for
from pgmvlm.data import CandidateCollator, PROMPT_TEXT
from pgmvlm.datasets import read_manifest
from pgmvlm.incremental_relations import IncrementalRelations
from pgmvlm.models import load_base, load_adapter, candidate_scores, ScoreTokens, set_train_mode
from pgmvlm.pretrained_relations import FrozenRelations, checked
from pgmvlm.extensions.probe_diagnostics import tensor_digest
from pgmvlm.storage import require_headroom
from tools.internvl_connector_pilot import candidate_item
from tools.train_pretrained_connectors import Banks, Images, lr_at
from tools.pretrained_connector_pilot import contract

STOP = False


def stop(*_):
    global STOP
    STOP = True


def append(path, row):
    with Path(path).open('a') as f:
        f.write(json.dumps(row, allow_nan=False) + '\n')
        f.flush()


def ordered_rows(plan):
    for key in ('train_manifest', 'val_manifest', 'full_train_manifest'):
        checked(plan[key], plan[key + '_sha256'])
    old = read_manifest(plan['train_manifest'])
    val = read_manifest(plan['val_manifest'])
    full = read_manifest(plan['full_train_manifest'])
    if (len(old), len(val), len(full)) != (60000, 1000, 240000):
        raise ValueError('This campaign requires the verified 60k/240k/1k manifests')
    old_ids = {r['id'] for r in old}
    if len(old_ids) != 60000 or not old_ids.issubset({r['id'] for r in full}):
        raise ValueError('Cached prefix is outside the 20-percent subset')
    if len({r['id'] for r in full + val}) != 241000:
        raise ValueError('Duplicate or overlapping training/validation IDs')
    # Preserve the same order across checkpoint experiments and every resume.
    first = [r for r in full if r['id'] in old_ids]
    rest = [r for r in full if r['id'] not in old_ids]
    rng = np.random.default_rng(plan['initialization_seed'])
    order = [first[i] for i in rng.permutation(len(first))] + [rest[i] for i in rng.permutation(len(rest))]
    return old, val, order


class ZeroBanks:
    """Control (plan['control'] == 'branch_off', added Nov 25): no Slot Abstractor features are read, computed or
    cached; every candidate gets a zero bank and the connector's branch_off zeroes the relation token anyway."""
    extracted, identity, blocks = 0, 'control-branch_off', []

    def __getitem__(self, key):
        return torch.zeros(1, 8, 9, 32)

    def flush(self):
        pass

    def committed_count(self):
        return 0


def is_control(plan):
    control = plan.get('control')
    if control not in (None, 'branch_off'):
        raise ValueError('control must be absent or "branch_off"')
    return control == 'branch_off'


class Run:
    def __init__(self, plan, out, mode, deadline, *, banks_factory=IncrementalRelations):
        self.plan, self.out, self.mode, self.deadline = plan, out, mode, deadline
        self.device = torch.device('cuda', torch.cuda.current_device())
        self.teacher = None
        self.old, self.val, self.order = ordered_rows(plan)
        self.rows = {r['id']: r for r in self.order + self.val}
        self.images = Images(self.order + self.val, plan, capacity=8)
        self.control = is_control(plan)
        if self.control:
            self.old_banks, self.banks = None, ZeroBanks()
        else:
            self.old_banks = Banks(plan['cache_output'], self.old + self.val, plan, capacity=4)
            identity = contract({k: plan[k] for k in ('teacher_hashes', 'teacher_seed', 'full_train_manifest_sha256', 'val_manifest_sha256')})
            self.banks = banks_factory(plan['incremental_cache'], identity, self.order + self.val,
                self.old_banks, {r['id'] for r in self.old + self.val}, self.extract,
                headroom=lambda root, **_: require_headroom(root, reserve_bytes=8*1024**3, reserve_files=300))
        if mode == 'pilot':
            offset = plan.get('pilot_uncached_start', 60000)
            self.train = self.order[:128] + self.order[offset:offset+128]
            self.validation = self.val[:32]
            self.total, self.frozen_steps, self.warmup = 16, 4, 2
        else:
            self.train, self.validation = self.order, self.val
            self.total, self.frozen_steps, self.warmup = 15000, 500, 50
        for name, digest in plan['adapter_hashes'].items():
            checked(Path(plan['adapter']) / name, digest)
        model, processor, family = load_base(plan['model'], device=self.device)
        if family != 'internvl':
            raise ValueError('Expected InternVL3.5 HF')
        self.model = load_adapter(model, plan['adapter'], trainable=True)
        self.lora = [(n, p) for n, p in self.model.named_parameters() if 'lora_' in n]
        if not self.lora or any(p.requires_grad for n, p in self.model.named_parameters() if 'lora_' not in n):
            raise RuntimeError('Unexpected model trainability')
        for _, p in self.lora:
            p.data = p.data.float()
        self.processor = processor
        self.bb = Backbone(self.model.get_base_model(), processor, family)
        self.st = ScoreTokens(processor.tokenizer)
        self.connector = CachedInterleave(self.bb).to(device=self.device, dtype=torch.float32)
        self.connector.branch_off = self.control      # control: the relation token is always zero
        self.native = CandidateCollator(processor, family, crop_to_patches=True)
        self.single_tile = CandidateCollator(processor, family, crop_to_patches=False)
        # Keep the pretrained checkpoint's crop setting. Interleave has no
        # spatial-map constraint and does not use Backbone.processor_kwargs().
        self.collator = Collator(processor, {'crop_to_patches': True}, rel_tokens=1, rel_mode='pool', num_slots=16)
        self.frozen = [(n, p) for n, p in self.model.named_parameters() if 'lora_' not in n]
        self.frozen_hash = tensor_digest(self.frozen)
        groups = []
        for kind, named in [('connector', list(self.connector.named_parameters())), ('lora', self.lora)]:
            for decay in (True, False):
                ps = [p for n, p in named if (p.ndim > 1 and not n.endswith('bias')) == decay]
                if ps:
                    groups.append(dict(params=ps, kind=kind, lr=0., weight_decay=.01 if decay else 0.))
        self.opt = torch.optim.AdamW(groups, betas=(.9, .999), eps=1e-8)
        self.progress = dict(update=0, examples=0, best_loss=None, best_checkpoint=None, pending_validation=True)
        self.identity = contract(dict(plan=plan, mode=mode))
        self.all_params = list(self.connector.parameters()) + [p for _, p in self.lora]
        self.initial_lora_hash = tensor_digest(self.lora)
        self.initial_connector_hash = tensor_digest(list(self.connector.named_parameters()))

    def extract(self, key):
        if self.teacher is None:
            # Lazy model construction must not perturb LoRA's dropout RNG.
            with torch.random.fork_rng(devices=[self.device.index or 0]):
                self.teacher = FrozenRelations(self.plan['teacher_package'], self.plan['teacher_hashes'], device=self.device)
        return self.teacher.extract(self.images[key], key, self.plan['teacher_seed'], verify=not self.teacher.parity_verified)

    def batch(self, row, permutation=None):
        item = candidate_item(row, self.plan)
        if permutation is not None:
            item['candidates'] = item['candidates'][permutation]
            item['target'] = list(permutation).index(item['target'])
        seqs = []
        for panel in item['candidates']:
            image, boxes = render_grid(list(item['context']) + [panel])
            seqs.append(dict(image=image, boxes=boxes, prompt=PROMPT_TEXT))
        batch = self.collator([dict(kind='cand', seqs=seqs, label=item['target'], id=row['id'])])
        batch['targets'] = torch.tensor([item['target']])
        return batch, item

    def scores(self, row, permutation=None):
        batch, _ = self.batch(row, permutation)
        bank = self.banks[row['id']][0].to(self.device)
        if permutation is not None:
            bank = bank[permutation]
        with self.connector.question(bank, batch['input_ids'].eq(self.bb.rel_id).to(self.device)):
            return candidate_scores(self.model, batch, self.st, self.device, torch.bfloat16), batch['targets'].to(self.device)

    def phase(self, update):
        joint = update >= self.frozen_steps
        for _, p in self.lora:
            p.requires_grad_(joint)
        if joint:
            set_train_mode(self.model)
        else:
            self.model.eval()
        self.connector.train()
        # Same schedule family as teammate H14/H15: connector warms early,
        # LoRA is zero until attachment ends; both decay after that boundary.
        decay = .1 + .9 * .5 * (1 + math.cos(math.pi * max(0, update-self.frozen_steps) / max(1, self.total-self.frozen_steps-1)))
        for g in self.opt.param_groups:
            peak = 1e-4 if g['kind'] == 'connector' else 2e-5
            g['lr'] = peak * min(1., (update+1)/self.warmup) * decay
            if g['kind'] == 'lora' and not joint:
                g['lr'] = 0.
        return joint

    def step(self, update):
        joint = self.phase(update)
        self.opt.zero_grad(set_to_none=True)
        start, misses = time.monotonic(), self.banks.extracted
        loss_sum, correct = 0., 0
        for row in self.train[update*16:(update+1)*16]:
            scores, target = self.scores(row)
            loss = torch.nn.functional.cross_entropy(scores, target)
            if not torch.isfinite(loss):
                raise RuntimeError('Nonfinite eight-way CE')
            (loss / 16).backward()
            loss_sum += float(loss.detach())
            correct += int(scores.argmax() == target)
        grads = {}
        for label, ps in [('connector', self.connector.parameters()), ('lora', (p for _, p in self.lora))]:
            values = [float(p.grad.detach().float().square().sum()) for p in ps if p.grad is not None]
            grads[label] = math.sqrt(sum(values))
        if (grads['connector'] <= 0 and not self.control) or (joint and grads['lora'] <= 0):
            raise RuntimeError('Missing connector/LoRA gradient')
        norm = torch.nn.utils.clip_grad_norm_(self.all_params, 1., error_if_nonfinite=True)
        self.opt.step()
        if not joint and tensor_digest(self.lora) != self.initial_lora_hash:
            raise RuntimeError('LoRA changed during connector-only warmup')
        return dict(update=update+1, examples=(update+1)*16, phase='joint' if joint else 'connector_only',
                    loss=loss_sum/16, accuracy=correct/16, gradient_norm=float(norm), gradient_groups=grads,
                    gain=float(self.connector.tok_gain.detach()), seconds=time.monotonic()-start,
                    cache_misses=self.banks.extracted-misses, lr={g['kind']:g['lr'] for g in self.opt.param_groups})

    @torch.no_grad()
    def evaluate(self):
        self.model.eval(); self.connector.eval()
        records, start = [], time.monotonic()
        for row in self.validation:
            if STOP or time.monotonic() > self.deadline - 120:
                break
            score, target = self.scores(row)
            loss = torch.nn.functional.cross_entropy(score, target)
            records.append(dict(id=row['id'], scores=score[0].tolist(), target=int(target), prediction=int(score.argmax()),
                                correct=int(score.argmax()) == int(target), loss=float(loss)))
        count = len(records)
        return dict(count=count, expected=len(self.validation), complete=count==len(self.validation),
                    loss=sum(x['loss'] for x in records)/count if count else None,
                    accuracy=sum(x['correct'] for x in records)/count if count else None,
                    seconds=time.monotonic()-start, predictions=records, update=self.progress['update'])

    def save(self, tag='last'):
        self.banks.flush()
        require_headroom(self.out, reserve_bytes=8*1024**3, reserve_files=300)
        def weights(folder):
            self.model.save_pretrained(folder, safe_serialization=True)
            save_file({n: t.detach().cpu().contiguous() for n,t in self.connector.state_dict().items()}, str(folder/'connector.safetensors'))
            atomic_json(folder/'connector.json', dict(reference='relational/relvlm.py', mode='pool', tokens=1, input_dim=32))
        state = dict(identity=self.identity, step=self.progress['update'], progress=copy.deepcopy(self.progress), optimizer=self.opt.state_dict(),
                     torch_rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state_all(), best_checkpoint=self.progress['best_checkpoint'])
        return commit(self.out, tag, weights, state=state if tag=='last' else None,
                      selection=dict(update=self.progress['update'], validation_loss=self.progress['best_loss']))

    def restore(self):
        result = resume_checkpoint(self.out)
        if result is None:
            raise RuntimeError('No complete checkpoint to resume')
        folder, state_path = result
        state = torch.load(state_path, map_location='cpu', weights_only=True)
        if state['identity'] != self.identity:
            raise RuntimeError('Changed training/resume contract')
        set_peft_model_state_dict(self.model, load_file(str(folder/'adapter_model.safetensors')))
        self.connector.load_state_dict(load_file(str(folder/'connector.safetensors')), strict=True)
        self.opt.load_state_dict(state['optimizer'])
        self.progress = state['progress']
        torch.set_rng_state(state['torch_rng']); torch.cuda.set_rng_state_all(state['cuda_rng'])

    @torch.no_grad()
    def checks(self):
        self.model.eval(); self.connector.eval()
        row = self.train[0]
        batch, item = self.batch(row)
        native = self.native([item]); single = self.single_tile([item])
        pixel_equal = native['pixel_values'].shape == single['pixel_values'].shape and torch.equal(native['pixel_values'], single['pixel_values'])
        native_scores = candidate_scores(self.model, native, self.st, self.device, torch.bfloat16)
        # Adding a placeholder token to the vocabulary must not change the
        # original candidate-scoring path; there is no active relation context.
        reference_collator = Collator(self.processor, {'crop_to_patches':True}, rel_tokens=0)
        seqs = []
        for panel in item['candidates']:
            image, boxes = render_grid(list(item['context']) + [panel])
            seqs.append(dict(image=image, boxes=boxes, prompt=PROMPT_TEXT))
        ref = reference_collator([dict(kind='cand',seqs=seqs,label=item['target'],id=row['id'])])
        ref['targets'] = native['targets']
        for key in ('input_ids','attention_mask','pixel_values'):
            torch.testing.assert_close(ref[key], native[key], rtol=0, atol=0)
        torch.testing.assert_close(candidate_scores(self.model, ref, self.st, self.device, torch.bfloat16),native_scores,rtol=0,atol=0)
        perm = [6,2,0,7,3,5,1,4]
        original, _ = self.scores(row)
        permuted, _ = self.scores(row, perm)
        torch.testing.assert_close(permuted, original[:,perm], rtol=1e-5, atol=1e-5)
        if not self.control:          # control: no teacher features exist to compare
            fresh = self.extract(row['id'])
            torch.testing.assert_close(fresh.mean(2), self.banks[row['id']].mean(2), rtol=1e-5, atol=1e-5)
        return dict(native_protocol_exact=True, candidate_permutation=True, teacher_cached_pool_parity=True,
                    crop_true_false_pixels_identical=pixel_equal, crop_true_shape=list(native['pixel_values'].shape),
                    crop_false_shape=list(single['pixel_values'].shape), preserved_crop_to_patches=True,
                    reference_connector=True, original_token_embedding_path=True)


def main(plan, out, mode, seconds, resume):
    if not os.environ.get('SLURM_JOB_ID') or not torch.cuda.is_available():
        raise RuntimeError('Requires a GPU allocation')
    if mode == 'pilot' and resume:
        raise ValueError('Use a fresh output for a new pilot; training resumes use mode=train')
    if mode == 'train':
        gate = json.loads((Path(plan['pilot_output'])/'summary.json').read_text())
        if not gate.get('passed') or gate.get('plan_sha256') != contract(plan):
            raise RuntimeError('Exact pilot must pass before full training')
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    torch.manual_seed(plan['initialization_seed']); torch.cuda.manual_seed_all(plan['initialization_seed'])
    deadline = time.monotonic() + seconds
    require_headroom(out, reserve_bytes=8*1024**3, reserve_files=2000)
    old_config = out/'config.json'
    cfg = dict(plan=plan,mode=mode)
    if old_config.exists():
        if not resume or json.loads(old_config.read_text()) != cfg:
            raise RuntimeError('Explicit identical resume required')
    else:
        atomic_json(old_config,cfg)
    with RunLock(plan['incremental_cache']):
        run = Run(plan,out,mode,deadline)
        if resume:
            run.restore()
            for name in ('learning-curve.jsonl','validation-curve.jsonl'):
                f = out/name
                if f.exists():
                    kept=[]
                    for line in f.read_text().splitlines():
                        try: r=json.loads(line)
                        except json.JSONDecodeError: continue
                        if r['update'] <= run.progress['update']: kept.append(line)
                    f.write_text(''.join(x+'\n' for x in kept))
        if mode == 'pilot' and not resume:
            atomic_json(out/'checks.json',run.checks())
        metrics=[]
        while True:
            if run.progress['pending_validation']:
                val=run.evaluate()
                atomic_json(out/'validation-latest.json',val)
                if val['complete']:
                    append(out/'validation-curve.jsonl',{k:v for k,v in val.items() if k!='predictions'})
                    run.progress['pending_validation']=False
                    if run.progress['best_loss'] is None or val['loss'] < run.progress['best_loss']:
                        run.progress['best_loss']=val['loss']
                        atomic_json(out/'validation-best.json',val)
                        run.progress['best_checkpoint']=run.save('best')
                run.save()
                if not val['complete']: break
            if run.progress['update'] >= run.total: break
            if STOP or time.monotonic() > deadline-300: break
            u=run.progress['update']; result=run.step(u)
            # Replay a real joint update through disk, optimizer and dropout RNG.
            if mode=='pilot' and u==run.frozen_steps:
                # Save after this update, replay the following update twice.
                run.progress.update(update=u+1, examples=(u+1)*16)
                run.save()
                nxt=run.step(u+1)
                expected={n:t.detach().clone() for n,t in get_peft_model_state_dict(run.model).items()}
                expected_c=copy.deepcopy(run.connector.state_dict())
                run.restore(); replay=run.step(u+1)
                for n,t in get_peft_model_state_dict(run.model).items():
                    torch.testing.assert_close(t,expected[n],rtol=0,atol=0)
                for n,t in run.connector.state_dict().items():
                    torch.testing.assert_close(t,expected_c[n],rtol=0,atol=0)
                checks=json.loads((out/'checks.json').read_text());checks['joint_next_update_replay_exact']=True
                atomic_json(out/'checks.json',checks)
                append(out/'learning-curve.jsonl',result);metrics.append(result)
                u+=1;result=replay
            run.progress.update(update=u+1, examples=(u+1)*16)
            append(out/'learning-curve.jsonl',result); metrics.append(result)
            if run.progress['update'] % (4 if mode=='pilot' else 1500)==0 or run.progress['update']==run.total:
                run.progress['pending_validation']=True
            if run.progress['update'] % (4 if mode=='pilot' else 100)==0:
                run.save()
            atomic_json(out/'progress.json',dict(**run.progress,cache_extracted_this_allocation=run.banks.extracted,
                        peak_gpu_bytes=torch.cuda.max_memory_allocated(),last_update=result))
        run.save()
        frozen_unchanged=tensor_digest(run.frozen)==run.frozen_hash
        if not frozen_unchanged or any(p.grad is not None or p.requires_grad for _,p in run.frozen):
            raise RuntimeError('Frozen backbone changed or received parameter gradients')
        complete=run.progress['update']==run.total and not run.progress['pending_validation']
        if mode=='pilot' and complete:
            checks=json.loads((out/'checks.json').read_text())
            for key in ('joint_next_update_replay_exact','native_protocol_exact','candidate_permutation','teacher_cached_pool_parity'):
                if checks.get(key) is not True: raise RuntimeError('Missing pilot gate: '+key)
            if tensor_digest(run.lora)==run.initial_lora_hash or tensor_digest(list(run.connector.named_parameters()))==run.initial_connector_hash:
                raise RuntimeError('Pilot parameters did not update')
        summary=dict(complete=complete,passed=complete,mode=mode,plan_sha256=contract(plan),
                     **run.progress,frozen_weights_unchanged=frozen_unchanged,
                     peak_gpu_bytes=torch.cuda.max_memory_allocated(),cache_extracted_this_allocation=run.banks.extracted)
        if mode=='pilot':
            for label,select in [('cached_joint',lambda m:m['phase']=='joint' and m['cache_misses']==0),
                                 ('uncached_joint',lambda m:m['phase']=='joint' and m['cache_misses']>0)]:
                chosen=[m for m in metrics if select(m)]
                summary[label+'_questions_per_second']=16*len(chosen)/sum(m['seconds'] for m in chosen) if chosen else None
        if complete and mode=='train' and not run.control:
            cached=sum(len(run.banks._read_pointer(i)['ids']) if run.banks._read_pointer(i) else 0 for i in range(len(run.banks.blocks)))
            if cached!=180000: raise RuntimeError('Completed epoch has an incomplete incremental feature cache')
            atomic_json(Path(plan['incremental_cache'])/'READY.json',dict(complete=True,new_training_questions=cached,
                        reused_training_questions=60000,validation_questions=1000,total_questions=241000,
                        identity=run.banks.identity,old_cache=plan['cache_output']))
        atomic_json(out/'summary.json',summary)
        if not complete: raise SystemExit(75)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--plan',required=True);p.add_argument('--output',required=True)
    p.add_argument('--mode',choices=['pilot','train'],required=True);p.add_argument('--seconds',type=int,required=True)
    p.add_argument('--resume',action='store_true');args=p.parse_args()
    if args.seconds<300 or (args.mode=='pilot' and args.seconds>1500):p.error('Invalid runtime cap')
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGUSR1,stop)
    output=Path(args.output)
    with RunLock(output):
        try:main(json.loads(Path(args.plan).read_text()),output,args.mode,args.seconds,args.resume)
        except Exception as e:
            atomic_json(output/'failure.json',dict(error=str(e),traceback=traceback.format_exc()));raise
