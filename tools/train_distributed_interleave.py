"""Four-GPU, global-batch16 interleave with exact resume and cautious stopping."""
from __future__ import annotations
import argparse
import copy
from datetime import timedelta
import json
import os
from pathlib import Path
import signal
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'relational'))
import torch
import torch.distributed as dist
from peft import get_peft_model_state_dict, set_peft_model_state_dict
from safetensors.torch import load_file, save_file
from tools.train_cached_interleave import Run, append, contract
from pgmvlm.checkpoints import RunLock, atomic_json, commit, resume_checkpoint
from pgmvlm.distributed_relations import ConcurrentRelations
from pgmvlm.distributed_training import question_rng, sum_gradients, stop_vote
from pgmvlm.extensions.probe_diagnostics import tensor_digest
from pgmvlm.plateau_stop import ConservativePlateau
from pgmvlm.storage import require_headroom

STOP = False


def stop(*_):
    global STOP
    STOP = True


def broadcast(value):
    values = [value if dist.get_rank() == 0 else None]
    dist.broadcast_object_list(values, src=0)
    return values[0]


class DistributedRun(Run):
    def __init__(self, plan, out, mode, deadline):
        self.rank, self.world = dist.get_rank(), dist.get_world_size()
        if self.world != 4:
            raise ValueError('This approved campaign requires exactly four GPU ranks')
        super().__init__(plan, out, mode, deadline, banks_factory=ConcurrentRelations)
        self.stopper = ConservativePlateau(**plan['early_stopping'])
        self.progress['early_stopped'] = False
        self.identity = contract(dict(plan=plan, mode=mode, world_size=self.world, runner='distributed-v1'))
        self.pilot_checks = {}

    def stopped(self, reserve=300):
        return stop_vote(STOP or time.monotonic() > self.deadline-reserve, self.device)

    def _accumulate(self, update, indices):
        loss_sum, correct = 0., 0
        for i in indices:
            row = self.train[update*16+i]
            with question_rng(row['id'], update, self.plan['initialization_seed'], self.device):
                scores, target = self.scores(row)
                loss = torch.nn.functional.cross_entropy(scores, target)
                if not torch.isfinite(loss):
                    raise RuntimeError('Nonfinite candidate CE')
                (loss/16).backward()
            loss_sum += float(loss.detach())
            correct += int(scores.argmax() == target)
        return loss_sum, correct

    def step(self, update, verify_serial=False):
        joint = self.phase(update)
        self.opt.zero_grad(set_to_none=True)
        start, misses = time.monotonic(), self.banks.extracted
        loss_sum, correct = self._accumulate(update, range(self.rank, 16, self.world))
        sum_gradients(self.all_params)  # Local losses already divide by global16.
        serial_expected = None
        serial_result = None
        if verify_serial and self.rank == 0:
            distributed_grads = [None if p.grad is None else p.grad.detach().clone() for p in self.all_params]
            before = [p.detach().clone() for p in self.all_params]
            optimizer_before = copy.deepcopy(self.opt.state_dict())
            self.opt.zero_grad(set_to_none=True)
            self._accumulate(update, range(16))
            max_gradient_error = 0.
            for p, expected in zip(self.all_params, distributed_grads):
                if expected is None:
                    if p.grad is not None:
                        raise RuntimeError('Serial/distributed gradient presence mismatch')
                else:
                    torch.testing.assert_close(p.grad, expected, rtol=1e-5, atol=1e-6)
                    max_gradient_error = max(max_gradient_error, float((p.grad-expected).abs().max()))
            torch.nn.utils.clip_grad_norm_(self.all_params, 1., error_if_nonfinite=True)
            self.opt.step()
            serial_expected = [p.detach().clone() for p in self.all_params]
            with torch.no_grad():
                for p, old in zip(self.all_params, before):
                    p.copy_(old)
            self.opt.load_state_dict(optimizer_before)
            for p, grad in zip(self.all_params, distributed_grads):
                p.grad = grad
            serial_result = dict(max_gradient_absolute_error=max_gradient_error,
                                 gradient_atol=1e-6, gradient_rtol=1e-5, update_atol=1e-6)
        if verify_serial:
            dist.barrier()
        gradients = {}
        for label, ps in [('connector', self.connector.parameters()), ('lora', (p for _,p in self.lora))]:
            gradients[label] = sum(float(p.grad.detach().float().square().sum()) for p in ps if p.grad is not None)**.5
        if gradients['connector'] <= 0 or (joint and gradients['lora'] <= 0):
            raise RuntimeError('Missing trainable gradient')
        norm = torch.nn.utils.clip_grad_norm_(self.all_params, 1., error_if_nonfinite=True)
        self.opt.step()
        if not joint and tensor_digest(self.lora) != self.initial_lora_hash:
            raise RuntimeError('LoRA changed while frozen')
        if verify_serial:
            if self.rank == 0:
                max_update_error = 0.
                for p, expected in zip(self.all_params, serial_expected):
                    torch.testing.assert_close(p, expected, rtol=0, atol=1e-6)
                    max_update_error = max(max_update_error, float((p-expected).abs().max()))
                serial_result['max_update_absolute_error'] = max_update_error
                serial_result['global_batch16_serial_equivalence'] = True
            self.pilot_checks.update(broadcast(serial_result))
        stats = torch.tensor([loss_sum, correct, self.banks.extracted-misses], dtype=torch.float64, device=self.device)
        dist.all_reduce(stats, op=dist.ReduceOp.SUM)
        duration = torch.tensor(time.monotonic()-start, dtype=torch.float64, device=self.device)
        dist.all_reduce(duration, op=dist.ReduceOp.MAX)
        return dict(update=update+1, examples=(update+1)*16, phase='joint' if joint else 'connector_only',
                    loss=float(stats[0])/16, accuracy=float(stats[1])/16, cache_misses=int(stats[2]),
                    seconds=float(duration), gradient_norm=float(norm), gradient_groups=gradients,
                    gain=float(self.connector.tok_gain.detach()), world_size=self.world,
                    serial_verification=verify_serial, lr={g['kind']:g['lr'] for g in self.opt.param_groups})

    @torch.no_grad()
    def evaluate(self):
        self.model.eval(); self.connector.eval()
        local, start = [], time.monotonic()
        for offset in range(0, len(self.validation), self.world):
            if self.stopped(120):
                break
            i = offset+self.rank
            if i < len(self.validation):
                row = self.validation[i]
                scores, target = self.scores(row)
                loss = torch.nn.functional.cross_entropy(scores, target)
                if not torch.isfinite(loss):
                    raise RuntimeError('Nonfinite validation loss')
                local.append(dict(index=i, id=row['id'], scores=scores[0].tolist(), target=int(target),
                                  prediction=int(scores.argmax()), correct=int(scores.argmax())==int(target), loss=float(loss)))
        groups = [None]*self.world
        dist.all_gather_object(groups, local)
        records = sorted([r for group in groups for r in group], key=lambda r:r['index'])
        if len({r['id'] for r in records}) != len(records):
            raise RuntimeError('Duplicated validation IDs')
        count = len(records)
        return dict(count=count, expected=len(self.validation), complete=count==len(self.validation),
                    loss=sum(r['loss'] for r in records)/count if count else None,
                    accuracy=sum(r['correct'] for r in records)/count if count else None,
                    seconds=time.monotonic()-start, predictions=records, update=self.progress['update'])

    def assert_synchronized(self):
        own = tensor_digest(list(self.connector.named_parameters())+self.lora)
        values = [None]*self.world
        dist.all_gather_object(values, own)
        if len(set(values)) != 1:
            raise RuntimeError('Trainable weights differ across GPU ranks')

    def save(self, tag='last'):
        self.banks.flush()
        dist.barrier()
        local_rng = dict(cpu=torch.get_rng_state(), cuda=torch.cuda.get_rng_state(self.device))
        states = [None]*self.world
        dist.all_gather_object(states, local_rng)
        pointer = None
        if self.rank == 0:
            require_headroom(self.out, reserve_bytes=8*1024**3, reserve_files=300)
            def weights(folder):
                self.model.save_pretrained(folder, safe_serialization=True)
                save_file({n:t.detach().cpu().contiguous() for n,t in self.connector.state_dict().items()}, str(folder/'connector.safetensors'))
                atomic_json(folder/'connector.json',dict(reference='relational/relvlm.py',mode='pool',tokens=1,input_dim=32))
            state = dict(identity=self.identity, world_size=self.world, step=self.progress['update'],
                         progress=copy.deepcopy(self.progress), optimizer=self.opt.state_dict(), rng_by_rank=states,
                         stopper=self.stopper.state_dict(), best_checkpoint=self.progress['best_checkpoint'])
            pointer = commit(self.out, tag, weights, state=state if tag=='last' else None,
                             selection=dict(update=self.progress['update'],validation_loss=self.progress['best_loss']))
        return broadcast(pointer)

    def restore(self):
        result = resume_checkpoint(self.out) if self.rank == 0 else None
        result = broadcast(tuple(str(p) for p in result) if result else None)
        if result is None:
            raise RuntimeError('No complete distributed checkpoint to resume')
        folder, state_path = map(Path, result)
        state = torch.load(state_path, map_location='cpu', weights_only=True)
        if state['identity'] != self.identity or state['world_size'] != self.world:
            raise RuntimeError('Changed resume contract or process count')
        set_peft_model_state_dict(self.model, load_file(str(folder/'adapter_model.safetensors')))
        self.connector.load_state_dict(load_file(str(folder/'connector.safetensors')), strict=True)
        self.opt.load_state_dict(state['optimizer'])
        self.progress = state['progress']
        self.stopper = ConservativePlateau.from_state_dict(state['stopper'])
        torch.set_rng_state(state['rng_by_rank'][self.rank]['cpu'])
        torch.cuda.set_rng_state(state['rng_by_rank'][self.rank]['cuda'], self.device)
        self.assert_synchronized()


def run(plan, out, mode, seconds, resume):
    rank = dist.get_rank()
    if mode=='pilot' and resume:
        raise ValueError('Retry a pilot in a fresh directory')
    if rank==0:
        require_headroom(out, reserve_bytes=8*1024**3, reserve_files=2000)
        if mode=='train':
            gate=json.loads((Path(plan['pilot_output'])/'summary.json').read_text())
            if not gate.get('passed') or gate.get('plan_sha256')!=contract(plan):
                raise RuntimeError('Matching four-GPU pilot must pass first')
        config=out/'config.json';value=dict(plan=plan,mode=mode)
        if config.exists():
            if not resume or json.loads(config.read_text())!=value:
                raise RuntimeError('Explicit identical resume required')
        else:
            atomic_json(config,value)
    dist.barrier()
    deadline=time.monotonic()+seconds
    runner=DistributedRun(plan,out,mode,deadline)
    runner.assert_synchronized()
    if resume:
        runner.restore()
        if rank==0:
            for name in ('learning-curve.jsonl','validation-curve.jsonl','stopping-curve.jsonl'):
                p=out/name
                if p.exists():
                    kept=[]
                    for line in p.read_text().splitlines():
                        try:r=json.loads(line)
                        except json.JSONDecodeError:continue
                        if r['update']<=runner.progress['update']:kept.append(line)
                    p.write_text(''.join(line+'\n' for line in kept))
        dist.barrier()
    if mode=='pilot':
        checks=runner.checks() if rank==0 else None
        runner.pilot_checks.update(broadcast(checks))
        if rank==0:atomic_json(out/'checks.json',runner.pilot_checks)
    metrics=[]
    while True:
        if runner.progress['pending_validation']:
            val=runner.evaluate()
            if rank==0:atomic_json(out/'validation-latest.json',val)
            if val['complete']:
                if rank==0:append(out/'validation-curve.jsonl',{k:v for k,v in val.items() if k!='predictions'})
                runner.progress['pending_validation']=False
                if runner.progress['best_loss'] is None or val['loss']<runner.progress['best_loss']:
                    runner.progress['best_loss']=val['loss']
                    if rank==0:atomic_json(out/'validation-best.json',val)
                    runner.progress['best_checkpoint']=runner.save('best')
                if mode=='train':
                    decision=runner.stopper.update(update=val['update'],loss=val['loss'],accuracy=val['accuracy'],count=val['count'])
                    runner.progress['early_stopped']=decision['stop']
                    if rank==0:append(out/'stopping-curve.jsonl',decision)
            runner.save()
            if not val['complete']:break
        if runner.progress['early_stopped'] or runner.progress['update']>=runner.total:break
        if runner.stopped():break
        u=runner.progress['update']
        result=runner.step(u,verify_serial=mode=='pilot' and u==runner.frozen_steps)
        if mode=='pilot' and u==runner.frozen_steps:
            runner.progress.update(update=u+1,examples=(u+1)*16)
            runner.save()
            runner.step(u+1)
            expected={n:t.detach().clone() for n,t in get_peft_model_state_dict(runner.model).items()}
            expected_c=copy.deepcopy(runner.connector.state_dict())
            runner.restore();replay=runner.step(u+1)
            for n,t in get_peft_model_state_dict(runner.model).items():torch.testing.assert_close(t,expected[n],rtol=0,atol=0)
            for n,t in runner.connector.state_dict().items():torch.testing.assert_close(t,expected_c[n],rtol=0,atol=0)
            runner.pilot_checks['joint_next_update_replay_exact']=True
            runner.assert_synchronized()
            if rank==0:
                atomic_json(out/'checks.json',runner.pilot_checks)
                append(out/'learning-curve.jsonl',result)
            metrics.append(result);u+=1;result=replay
        runner.progress.update(update=u+1,examples=(u+1)*16)
        if rank==0:append(out/'learning-curve.jsonl',result)
        metrics.append(result)
        if runner.progress['update']%(4 if mode=='pilot' else 500)==0 or runner.progress['update']==runner.total:
            runner.progress['pending_validation']=True
        if runner.progress['update']%(4 if mode=='pilot' else 100)==0:runner.save()
        if rank==0:atomic_json(out/'progress.json',dict(**runner.progress,peak_gpu_bytes=torch.cuda.max_memory_allocated(),last_update=result))
    runner.save();runner.assert_synchronized()
    unchanged=tensor_digest(runner.frozen)==runner.frozen_hash
    invalid=not unchanged or any(p.requires_grad or p.grad is not None for _,p in runner.frozen)
    if stop_vote(invalid,runner.device):raise RuntimeError('Frozen backbone changed or received gradients')
    complete=(runner.progress['update']==runner.total or runner.progress['early_stopped']) and not runner.progress['pending_validation']
    if mode=='pilot' and complete:
        for key in ('global_batch16_serial_equivalence','joint_next_update_replay_exact','native_protocol_exact','candidate_permutation','teacher_cached_pool_parity'):
            if runner.pilot_checks.get(key) is not True:raise RuntimeError('Missing pilot check '+key)
        if tensor_digest(runner.lora)==runner.initial_lora_hash or tensor_digest(list(runner.connector.named_parameters()))==runner.initial_connector_hash:
            raise RuntimeError('Pilot trainable parameters did not update')
    extracted=torch.tensor(runner.banks.extracted,device=runner.device);dist.all_reduce(extracted,op=dist.ReduceOp.SUM)
    summary=dict(complete=complete,passed=complete,mode=mode,plan_sha256=contract(plan),world_size=4,
                 **runner.progress,frozen_weights_unchanged=True,cache_extracted_this_allocation=int(extracted),
                 allocation_seconds=seconds-max(0,deadline-time.monotonic()),
                 stop_reason='validation_plateau' if runner.progress['early_stopped'] else ('one_epoch' if complete else 'allocation_boundary'))
    if mode=='pilot':
        for label,condition in [('cached_joint',lambda m:m['phase']=='joint' and m['cache_misses']==0 and not m['serial_verification']),
                                ('uncached_joint',lambda m:m['phase']=='joint' and m['cache_misses']>0 and not m['serial_verification'])]:
            selected=[m for m in metrics if condition(m)]
            summary[label+'_questions_per_second']=16*len(selected)/sum(m['seconds'] for m in selected) if selected else None
    if rank==0:
        summary['new_cache_committed_questions']=runner.banks.committed_count()
        if runner.progress['update']==runner.total and mode=='train':
            if summary['new_cache_committed_questions']!=180000:raise RuntimeError('Incomplete feature cache after full epoch')
            atomic_json(Path(plan['incremental_cache'])/'READY.json',dict(complete=True,new_training_questions=180000,reused_training_questions=60000,validation_questions=1000,total_questions=241000,identity=runner.banks.identity,old_cache=plan['cache_output']))
        atomic_json(out/'summary.json',summary)
        atomic_json(out/'progress.json',dict(**runner.progress,stop_reason=summary['stop_reason']))
    dist.barrier()
    return complete


def main():
    p=argparse.ArgumentParser();p.add_argument('--plan',required=True);p.add_argument('--output',required=True)
    p.add_argument('--mode',choices=['pilot','train'],required=True);p.add_argument('--seconds',type=int,required=True);p.add_argument('--resume',action='store_true')
    args=p.parse_args()
    if not os.environ.get('SLURM_JOB_ID') or not torch.cuda.is_available():raise RuntimeError('Requires GPU allocation')
    if args.seconds<300:raise ValueError('Insufficient runtime reserve')
    rank=int(os.environ['RANK']);local=int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(local)
    dist.init_process_group('nccl',timeout=timedelta(minutes=10),device_id=torch.device('cuda',local))
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGUSR1,stop)
    torch.set_num_threads(2);torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    plan=json.loads(Path(args.plan).read_text());torch.manual_seed(plan['initialization_seed']);torch.cuda.manual_seed(plan['initialization_seed'])
    out=Path(args.output)
    try:
        with RunLock(out,enabled=rank==0),RunLock(plan['incremental_cache'],enabled=rank==0):
            dist.barrier()
            complete=run(plan,out,args.mode,args.seconds,args.resume)
        dist.destroy_process_group()
        # A clean allocation boundary is successful for Slurm dependency chains.
        if not complete:
            if args.mode=='pilot':raise SystemExit(75)
            if rank==0:print('CHECKPOINTED_NEEDS_RESUME',flush=True)
    except Exception as error:
        out.mkdir(parents=True,exist_ok=True)
        atomic_json(out/f'failure-rank{rank}.json',dict(error=str(error),traceback=traceback.format_exc()))
        raise


if __name__=='__main__':main()
