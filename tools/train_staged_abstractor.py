"""Single-design frozen Case-A training, gated on the seven-arm GPU pilot.

No scheduler submission. No automatic restart. Native/model/scoring code must
match the passing pilot; only the training pool, schedule and duration expand.
"""
from __future__ import annotations
import argparse
from collections import OrderedDict
import fcntl
import json
import math
import os
from pathlib import Path
import signal
import sys
import time

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from PIL import Image
from pgmvlm.datasets import read_manifest,training_order
from pgmvlm.runtime import capture_rng,restore_rng
from pgmvlm.extensions.staged_abstractor import DESIGNS,install
from pgmvlm.models import load_base,ScoreTokens
from pgmvlm.protocol_probe import candidate_images,candidate_ce
from pgmvlm.storage import require_headroom
from tools.input_representation_pilot import load_selection,resolve_image,digest,atomic_json,checkpoint,append
from tools.staged_abstractor_pilot import scores_for,base_parameters,frozen_check,lr_at,contract_hash
from pgmvlm.extensions.probe_diagnostics import tensor_digest

STOP=False
CRITICAL_FILES=('pgmvlm/extensions/staged_abstractor.py','pgmvlm/extensions/abstractor_plugin.py',
    'pgmvlm/extensions/relational.py','pgmvlm/models.py','pgmvlm/protocol_probe.py',
    'pgmvlm/panels.py','pgmvlm/data.py','tools/staged_abstractor_pilot.py')


def validate_pilot(summary,config):
    if summary.get('status')!='passed' or not summary.get('base_bitwise_unchanged'):
        raise ValueError('A complete passing frozen-model GPU pilot is required')
    if set(summary.get('results',{}))!=set(DESIGNS):
        raise ValueError('Pilot does not cover exactly the seven designs')
    for name,result in summary['results'].items():
        checks=result.get('checks',{})
        if (result.get('status')!='passed' or result.get('updates')!=config['max_updates']
            or result.get('validation',{}).get('count')!=config['validation_questions']
            or not all(result.get(k) for k in ('base_bitwise_unchanged','checkpoint_reload_exact','next_update_replay_exact'))
            or not all(checks.get(k) for k in ('zero_share_native_exact','candidate_permutation','deep_gradients','added_weights_update'))):
            raise ValueError('Pilot gate failed: '+name)


def verified_gate(plan,repo):
    path=Path(plan['pilot_output'])
    config=json.loads((path/'config.json').read_text())
    summary=json.loads((path/'summary.json').read_text())
    validate_pilot(summary,config)
    if plan['module']!=config['module'] or plan['revision']!=config['revision']:
        raise ValueError('Architecture/base revision differs from passing pilot')
    for name in CRITICAL_FILES:
        if digest(repo/name)!=digest(Path(plan['pilot_source'])/name):
            raise ValueError('Re-pilot changed architecture/scoring source: '+name)
    return dict(job_id=summary['job_id'],summary_sha256=digest(path/'summary.json'),
                pilot_source_commit=config['source_commit'],passed_designs=list(DESIGNS))


class Images:
    """Bounded decoded-image cache; no full training set is materialized in RAM."""
    def __init__(self,rows,plan,capacity=32):
        self.rows={r['id']:r for r in rows};self.plan=plan;self.capacity=capacity;self.cache=OrderedDict()
    def __getitem__(self,key):
        if key in self.cache:
            self.cache.move_to_end(key);return self.cache[key]
        with Image.open(resolve_image(self.rows[key],self.plan)) as image:
            result=candidate_images(image.convert('RGB'))
        self.cache[key]=result
        if len(self.cache)>self.capacity: self.cache.popitem(last=False)
        return result


def selected_rows(plan):
    small,val,_=load_selection(plan)
    if plan['train_questions']==4096:
        train=small
    elif plan['train_questions']==240000:
        if digest(plan['train_manifest'])!=plan['train_manifest_sha256']:
            raise ValueError('Fixed 20% manifest changed')
        train=read_manifest(plan['train_manifest'])
    else:
        raise ValueError('Select either the historical 4096 or the fixed 240000 pool')
    if len(train)!=plan['train_questions'] or len(val)!=1000:
        raise ValueError('Split size mismatch')
    if len({r['id'] for r in train+val})!=len(train)+len(val):
        raise ValueError('Duplicate or overlapping train/validation IDs')
    for rows,split in ((train,'train'),(val,'validation')):
        if any(r['split']!=split or r['dataset']!='pgm' or r['regime']!='neutral'
               or r['choices']!=list('ABCDEFGH') for r in rows):
            raise ValueError('Unexpected dataset, regime or split')
    return train,val


def evaluate(model,processor,ctl,tokens,rows,images,deadline):
    ctl.module.eval();records=[]
    with torch.no_grad():
        for row in rows:
            if STOP or time.time()>deadline-30: break
            scores=scores_for(model,processor,images[row['id']],tokens);target=ord(row['answer'])-65
            records.append(dict(id=row['id'],target=target,pred=int(scores.argmax()),
                                loss=float(candidate_ce(scores,target)),scores=scores.cpu().tolist()))
    return dict(count=len(records),planned=len(rows),complete=len(records)==len(rows),
        loss=sum(r['loss'] for r in records)/len(records) if records else None,
        accuracy=sum(r['pred']==r['target'] for r in records)/len(records) if records else None,predictions=records)


def train_batch(model,processor,ctl,tokens,opt,rows,images,lr,clip):
    ctl.module.train();opt.zero_grad(set_to_none=True)
    for group in opt.param_groups:group['lr']=lr
    loss_sum=0.;correct=0;started=time.time()
    for row in rows:
        scores=scores_for(model,processor,images[row['id']],tokens);target=ord(row['answer'])-65
        loss=candidate_ce(scores,target)
        if not torch.isfinite(loss):raise RuntimeError('Nonfinite candidate loss')
        (loss/len(rows)).backward();loss_sum+=float(loss.detach());correct+=int(scores.detach().argmax())==target
    # Nonzero gradients are verified in the pilot; a converged batch may have zero
    # gradient. Finiteness and base freezing remain mandatory throughout training.
    norm=torch.nn.utils.clip_grad_norm_(ctl.module.parameters(),clip,error_if_nonfinite=True)
    keys=('content.1.weight','layers.0.relation.q.weight',
          f"layers.{ctl.module.spec['depth']-1}.relation.q.weight",'fusion.projection.1.weight','fusion.logit')
    grads={n:float(p.grad.float().square().mean().sqrt()) if p.grad is not None else None
           for n,p in ctl.module.named_parameters() if n in keys}
    if any(p.grad is not None for _,p in base_parameters(model)):
        raise RuntimeError('A frozen VLM parameter received gradients')
    opt.step()
    return dict(loss=loss_sum/len(rows),accuracy=correct/len(rows),gradient_norm=float(norm),
                gradient_rms=grads,fusion=dict(ctl.module.fusion.stats),lr=lr,seconds=time.time()-started)


def save(path,ctl,opt,progress,plan,ids_hash):
    require_headroom(path.parent)
    checkpoint(path,dict(format_version=1,module=ctl.module.state_dict(),spec=ctl.module.spec,
        optimizer=opt.state_dict(),progress=progress,contract=contract_hash(plan),ids_hash=ids_hash,rng=capture_rng()))


def restore(path,ctl,opt,plan,ids_hash):
    if digest(path)!=json.loads(path.with_suffix('.sha256.json').read_text())['sha256']:
        raise ValueError('Checkpoint checksum mismatch')
    state=torch.load(path,map_location='cpu',weights_only=False)
    if state['contract']!=contract_hash(plan) or state['ids_hash']!=ids_hash or state['spec']!=ctl.module.spec:
        raise ValueError('Checkpoint model/data/recipe/source identity changed')
    ctl.module.load_state_dict(state['module'],strict=True);opt.load_state_dict(state['optimizer']);restore_rng(state['rng'])
    return state['progress']


def run(plan,design,output,deadline,resume=False):
    if not os.environ.get('SLURM_JOB_ID') or not torch.cuda.is_available():raise RuntimeError('GPU allocation required')
    if plan['case']!='A' or plan['lora'] or not plan['pretrained_vlm_frozen'] or design not in DESIGNS:
        raise ValueError('Only fresh-branch frozen Case A is authorized by this runner')
    plan={**plan,'design':design}
    output.mkdir(parents=True,exist_ok=True)
    with (output/'.run.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if (output/'config.json').exists():
            if not resume or json.loads((output/'config.json').read_text())!=plan:raise ValueError('Explicit same-contract resume required')
        else:atomic_json(output/'config.json',plan)
        repo=Path(__file__).resolve().parents[1]
        atomic_json(output/'pilot-gate.json',verified_gate(plan,repo))
        atomic_json(output/'storage-preflight.json',require_headroom(output))
        def stop(*args):
            global STOP
            STOP=True
        for sig in (signal.SIGUSR1,signal.SIGTERM,signal.SIGINT):signal.signal(sig,stop)
        torch.set_num_threads(2);torch.use_deterministic_algorithms(True)
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        train,val=selected_rows(plan);batch=plan['global_batch']
        if len(train)%batch:raise ValueError('Dataset must divide into full global batches')
        ids=dict(train=[r['id'] for r in train],validation=[r['id'] for r in val]);ids_hash=contract_hash(ids)
        atomic_json(output/'question_ids.json',ids)
        images=Images(train+val,plan)
        total=plan['max_epochs']*len(train)//batch
        order=training_order(len(train),total*batch,plan['training_seed'],'python_epoch')
        model,processor,family=load_base('lfm25',revision=plan['revision'],device='cuda')
        model.eval().requires_grad_(False);base_hash=tensor_digest(base_parameters(model))
        tokens=ScoreTokens(processor.tokenizer)
        if not tokens.single:raise ValueError('Yes/No must be single native tokens')
        torch.manual_seed(plan['initialization_seed']);ctl=install(model,processor,design,**plan['module'])
        opt=torch.optim.AdamW(ctl.module.parameters(),lr=plan['lr'],weight_decay=plan['weight_decay'])
        atomic_json(output/'backend.json',dict(family=family,gpu=torch.cuda.get_device_name(),torch=torch.__version__,
            base_sha256=base_hash,module=ctl.module.spec,trainable_parameters=sum(p.numel() for p in ctl.module.parameters())))
        progress=dict(updates=0,best_loss=None,best_update=None,bad_validations=0,last_validation_update=-1,
                      evaluation_seconds=None,stop_reason=None,plateau_reference_loss=None)
        if resume and (output/'latest.pt').exists():progress=restore(output/'latest.pt',ctl,opt,plan,ids_hash)
        def validation():
            started=time.time();result=evaluate(model,processor,ctl,tokens,val,images,deadline)
            elapsed=time.time()-started;u=progress['updates']
            atomic_json(output/f'validation-{u:06d}.json',result)
            append(output/'validation-curve.jsonl',dict(update=u,epoch=u*batch/len(train),seconds=elapsed,
                **{k:v for k,v in result.items() if k!='predictions'}))
            if result['complete']:
                progress['evaluation_seconds']=elapsed;progress['last_validation_update']=u
                previous=progress['best_loss']
                improved=previous is None or result['loss']<previous
                anchor=progress['plateau_reference_loss']
                meaningful=anchor is None or result['loss']<anchor-plan['early_stop_min_delta']
                if meaningful:progress['plateau_reference_loss']=result['loss']
                progress['bad_validations']=0 if meaningful else progress['bad_validations']+1
                if improved:
                    progress['best_loss']=result['loss'];progress['best_update']=u
                    save(output/'best.pt',ctl,opt,progress,plan,ids_hash)
                    atomic_json(output/'best-validation.json',result)
            return result
        if progress['evaluation_seconds'] is None:
            initial=validation()
            if not initial['complete']:raise RuntimeError('Initial validation incomplete; do not claim training results')
        save(output/'latest.pt',ctl,opt,progress,plan,ids_hash)
        times=[];last=None
        while progress['updates']<total:
            reserve=max(plan['minimum_reserve_seconds'],progress['evaluation_seconds']*1.5+120)
            if STOP or time.time()+max(60,max(times[-4:],default=0)*1.5)>deadline-reserve:
                progress['stop_reason']='signal' if STOP else 'wall_time';break
            u=progress['updates'];rows=[train[i] for i in order[u*batch:(u+1)*batch]]
            info=train_batch(model,processor,ctl,tokens,opt,rows,images,
                lr_at(u,total,plan['lr'],plan['warmup_updates'],plan['min_lr_ratio']),plan['clip_norm'])
            progress['updates']+=1;times.append(info['seconds']);u=progress['updates']
            info.update(update=u,epoch=u*batch/len(train),presentations=u*batch,question_ids=[r['id'] for r in rows])
            append(output/'learning-curve.jsonl',info)
            if u%plan['checkpoint_every']==0:save(output/'latest.pt',ctl,opt,progress,plan,ids_hash)
            if u%16==0:
                atomic_json(output/'progress.json',dict(**progress,epochs=u*batch/len(train),
                    unique_questions=min(len(train),u*batch),remaining_seconds=deadline-time.time()))
            if u%plan['validation_every']==0:
                last=validation();save(output/'latest.pt',ctl,opt,progress,plan,ids_hash)
                if (plan['early_stop_patience']>0 and u*batch/len(train)>=plan['early_stop_min_epochs']
                        and progress['bad_validations']>=plan['early_stop_patience']):
                    progress['stop_reason']='validation_plateau';break
        if progress['stop_reason'] is None:progress['stop_reason']='epoch_limit'
        if progress['last_validation_update']!=progress['updates']:last=validation()
        save(output/'latest.pt',ctl,opt,progress,plan,ids_hash)
        frozen_check(model,base_hash)
        summary=dict(status='stopped',**progress,epochs=progress['updates']*batch/len(train),
            unique_questions=min(len(train),progress['updates']*batch),presentations=progress['updates']*batch,
            fraction_of_official_train=min(len(train),progress['updates']*batch)/1200000,
            train_pool=len(train),validation_pool=len(val),base_bitwise_unchanged=True,benchmark_test=False,
            final_validation_complete=progress['last_validation_update']==progress['updates'],
            selected_checkpoint='best.pt',selection='minimum complete 1000-question validation CE',
            job_id=os.environ['SLURM_JOB_ID'],source_commit=plan['source_commit'])
        atomic_json(output/'summary.json',summary);print(json.dumps(summary),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--plan',type=Path,required=True);p.add_argument('--design',choices=DESIGNS,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--deadline',type=float,required=True)
    p.add_argument('--resume',action='store_true');a=p.parse_args()
    try:run(json.loads(a.plan.read_text()),a.design,a.output,a.deadline,a.resume)
    except Exception as exc:
        a.output.mkdir(parents=True,exist_ok=True);atomic_json(a.output/'failure.json',dict(error=repr(exc)));raise


if __name__=='__main__':main()
