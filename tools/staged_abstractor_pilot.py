"""Bounded seven-design Case-A pilot. No job submission or automatic continuation."""
from __future__ import annotations

import argparse
import copy
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import random
import signal
import sys
import time

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from PIL import Image
from pgmvlm.data import PROMPT_TEXT
from pgmvlm.panels import messages
from pgmvlm.models import load_base, ScoreTokens
from pgmvlm.protocol_probe import candidate_images, candidate_ce, yes_no_score
from pgmvlm.extensions.staged_abstractor import DESIGNS, install
from pgmvlm.extensions.probe_diagnostics import tensor_digest
from pgmvlm.storage import require_headroom
from tools.input_representation_pilot import load_selection, resolve_image, atomic_json, checkpoint, append, digest

STOP=False


def stopped(signum,frame):
    global STOP
    STOP=True


def lr_at(update,total,peak,warmup,minimum=.1):
    if update<warmup:
        return peak*(update+1)/max(warmup,1)
    fraction=(update-warmup)/max(total-warmup-1,1)
    return peak*(minimum+(1-minimum)*.5*(1+math.cos(math.pi*min(fraction,1.))))


def encode(processor,image,device):
    batch=processor.apply_chat_template(messages(image,PROMPT_TEXT,'composite'),
        add_generation_prompt=True,tokenize=True,return_dict=True,return_tensors='pt',
        truncation=False,do_image_splitting=False)
    result={k:v.to(device=device,dtype=torch.bfloat16 if v.is_floating_point() else v.dtype)
            for k,v in batch.items() if isinstance(v,torch.Tensor)}
    if tuple(result['spatial_shapes'].shape)!=(1,2):
        raise ValueError('Processor did not honor one-whole-image-tile policy')
    return result


def scores_for(model,processor,matrices,tokens):
    values=[]
    for matrix in matrices:
        batch=encode(processor,matrix,next(model.parameters()).device)
        result=model(**batch,use_cache=False,logits_to_keep=1,output_hidden_states=True)
        values.append(yes_no_score(result.hidden_states[-1][:,-1],model.get_output_embeddings(),
                                   tokens.yes_id,tokens.no_id))
    return torch.stack(values)


def base_parameters(model):
    return [(n,p) for n,p in model.named_parameters() if not n.startswith('staged_abstractor.')]


def frozen_check(model,expected):
    params=base_parameters(model)
    if any(p.requires_grad or p.grad is not None for _,p in params):
        raise RuntimeError('Original VLM became trainable or received parameter gradients')
    actual=tensor_digest(params)
    if actual!=expected:
        raise RuntimeError('Frozen base weights changed')
    return actual


def train_step(model,processor,controller,tokens,optimizer,rows,images,lr,clip):
    controller.module.train()
    optimizer.zero_grad(set_to_none=True)
    for group in optimizer.param_groups: group['lr']=lr
    started=time.time()
    loss_sum,correct=0.,0
    for row in rows:
        scores=scores_for(model,processor,images[row['id']],tokens)
        target=ord(row['answer'])-65
        loss=candidate_ce(scores,target)
        if not torch.isfinite(loss): raise RuntimeError('Nonfinite candidate CE')
        (loss/len(rows)).backward()
        loss_sum+=float(loss.detach())
        correct+=int(scores.detach().argmax())==target
    gradients={n:float(p.grad.float().square().mean().sqrt()) for n,p in controller.module.named_parameters()
               if p.grad is not None}
    for name in ('content.1.weight','layers.0.relation.q.weight',
                 f"layers.{controller.module.spec['depth']-1}.relation.q.weight",'fusion.projection.1.weight'):
        if not math.isfinite(gradients.get(name,float('nan'))) or not gradients[name]>0:
            raise RuntimeError('Missing/nonfinite/zero branch gradient: '+name)
    norm=torch.nn.utils.clip_grad_norm_(controller.module.parameters(),clip,error_if_nonfinite=True)
    optimizer.step()
    return dict(loss=loss_sum/len(rows),accuracy=correct/len(rows),gradient_norm=float(norm),
                gradient_rms=gradients,seconds=time.time()-started,lr=lr,
                fusion=copy.deepcopy(controller.module.fusion.stats))


def evaluate(model,processor,controller,tokens,rows,images):
    controller.module.eval()
    records=[]
    with torch.no_grad():
        for row in rows:
            scores=scores_for(model,processor,images[row['id']],tokens)
            target=ord(row['answer'])-65
            records.append(dict(id=row['id'],target=target,pred=int(scores.argmax()),
                loss=float(candidate_ce(scores,target)),scores=scores.cpu().tolist()))
    return dict(count=len(records),loss=sum(r['loss'] for r in records)/len(records),
                accuracy=sum(r['target']==r['pred'] for r in records)/len(records),predictions=records)


def contract_hash(plan):
    return hashlib.sha256(json.dumps(plan,sort_keys=True,separators=(',',':')).encode()).hexdigest()


def save_state(path,controller,optimizer,updates,plan,train_ids,val_ids):
    require_headroom(path.parent)
    checkpoint(path,dict(format_version=1,spec=controller.module.spec,module=controller.module.state_dict(),
        optimizer=optimizer.state_dict(),updates=updates,contract=contract_hash(plan),
        train_ids=train_ids,validation_ids=val_ids,torch_rng=torch.get_rng_state(),
        cuda_rng=torch.cuda.get_rng_state_all()))


def restore_state(path,controller,optimizer,plan,train_ids,val_ids):
    meta=json.loads(path.with_suffix('.sha256.json').read_text())
    if digest(path)!=meta['sha256']: raise ValueError('Checkpoint checksum mismatch')
    state=torch.load(path,map_location=next(controller.module.parameters()).device,weights_only=False)
    if (state['contract']!=contract_hash(plan) or state['spec']!=controller.module.spec
            or state['train_ids']!=train_ids or state['validation_ids']!=val_ids):
        raise ValueError('Cannot resume with changed model, recipe, code revision or data IDs')
    controller.module.load_state_dict(state['module'],strict=True)
    optimizer.load_state_dict(state['optimizer'])
    torch.set_rng_state(state['torch_rng'].cpu())
    torch.cuda.set_rng_state_all([x.cpu() for x in state['cuda_rng']])
    return state['updates']


def correctness_checks(model,processor,controller,tokens,row,images,plan):
    module=controller.module
    initial=copy.deepcopy(module.state_dict())
    matrices=images[row['id']]
    target=ord(row['answer'])-65
    with torch.no_grad():
        controller.set_enabled(False)
        native=scores_for(model,processor,matrices,tokens)
        controller.set_enabled(True)
        module.fusion.force_share=0.
        bypass=scores_for(model,processor,matrices,tokens)
        torch.testing.assert_close(native,bypass,rtol=0,atol=0)
        module.fusion.force_share=None
        initial_scores=scores_for(model,processor,matrices,tokens)
        # Deliberately change candidate order: neither encoding nor hooks sees labels.
        permutation=[3,7,0,6,2,1,5,4]
        reordered=scores_for(model,processor,[matrices[i] for i in permutation],tokens)
        torch.testing.assert_close(reordered,initial_scores[permutation],rtol=0,atol=1e-5)
        torch.testing.assert_close(candidate_ce(initial_scores,target),
            candidate_ce(reordered,permutation.index(target)),rtol=0,atol=1e-5)
    opt=torch.optim.AdamW(module.parameters(),lr=plan['lr'],weight_decay=plan['weight_decay'])
    gradient_check=train_step(model,processor,controller,tokens,opt,[row],images,plan['lr'],plan['clip_norm'])
    changed=tensor_digest(module.named_parameters())
    module.load_state_dict(initial); module.zero_grad(set_to_none=True)
    if changed==tensor_digest(module.named_parameters()): raise RuntimeError('Optimizer did not change added weights')
    return dict(zero_share_native_exact=True,candidate_permutation=True,deep_gradients=True,
                added_weights_update=True,initial_score_shift=float((initial_scores-native).abs().max()),
                gradient_check=gradient_check,check_updates_discarded=True)


def run(plan,output,deadline,resume=False):
    if not os.environ.get('SLURM_JOB_ID') or not torch.cuda.is_available():
        raise RuntimeError('Run inside a GPU allocation, never on the login node')
    if tuple(plan['designs'])!=DESIGNS or plan['case']!='A' or plan['lora'] or not plan['pretrained_vlm_frozen']:
        raise ValueError('This pilot requires all seven Case-A frozen-VLM designs')
    output.mkdir(parents=True,exist_ok=True)
    with (output/'.run.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if (output/'config.json').exists():
            if not resume or json.loads((output/'config.json').read_text())!=plan:
                raise ValueError('Existing output: explicit same-contract resume is required')
        else:
            atomic_json(output/'config.json',plan)
        atomic_json(output/'storage-preflight.json',require_headroom(output))
        for sig in (signal.SIGUSR1,signal.SIGTERM,signal.SIGINT): signal.signal(sig,stopped)
        torch.set_num_threads(2);torch.use_deterministic_algorithms(True)
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        train,val,_=load_selection(plan)
        random.Random(plan['training_seed']).shuffle(train)
        train=train[:plan['train_questions']];val=val[:plan['validation_questions']]
        if len(train)%plan['global_batch']: raise ValueError('Use full batches for this pilot')
        train_ids=[r['id'] for r in train];val_ids=[r['id'] for r in val]
        atomic_json(output/'question_ids.json',dict(train=train_ids,validation=val_ids,
             parent_selection_sha256=plan['selected_rows_sha256']))
        images={}
        for row in train+val:
            with Image.open(resolve_image(row,plan)) as image:
                images[row['id']]=candidate_images(image.convert('RGB'))
        model,processor,family=load_base('lfm25',revision=plan['revision'],device='cuda')
        model.eval().requires_grad_(False)
        before=tensor_digest(base_parameters(model))
        tokens=ScoreTokens(processor.tokenizer)
        if not tokens.single: raise ValueError('Yes and No must each be a single native token')
        atomic_json(output/'backend.json',dict(torch=torch.__version__,family=family,
            gpu=torch.cuda.get_device_name(),base_sha256=before,prompt=PROMPT_TEXT,
            processor_policy=dict(do_image_splitting=False),yes_id=tokens.yes_id,no_id=tokens.no_id,
            source_commit=plan.get('source_commit'),base_parameters=sum(p.numel() for p in model.parameters())))
        results={}
        for design in plan['designs']:
            folder=output/design;folder.mkdir(exist_ok=True)
            if (folder/'summary.json').exists():
                existing=json.loads((folder/'summary.json').read_text())
                if existing.get('status')=='passed': results[design]=existing;continue
            if STOP or time.time()>deadline-plan['reserve_seconds']:
                results[design]=dict(status='not_started',reason='time_or_signal');continue
            ctl=None
            try:
                torch.manual_seed(plan['initialization_seed'])
                ctl=install(model,processor,design,**plan['module'])
                opt=torch.optim.AdamW(ctl.module.parameters(),lr=plan['lr'],weight_decay=plan['weight_decay'])
                atomic_json(folder/'spec.json',dict(**ctl.module.spec,trainable_parameters=sum(p.numel() for p in ctl.module.parameters())))
                state_path=folder/'latest.pt'
                if resume and state_path.exists():
                    updates=restore_state(state_path,ctl,opt,plan,train_ids,val_ids)
                    checks=json.loads((folder/'checks.json').read_text())
                else:
                    updates=0
                    checks=correctness_checks(model,processor,ctl,tokens,train[0],images,plan)
                    atomic_json(folder/'checks.json',checks)
                    initial=evaluate(model,processor,ctl,tokens,val,images)
                    atomic_json(folder/'initial-validation.json',initial)
                    save_state(state_path,ctl,opt,0,plan,train_ids,val_ids)
                batches=[train[i:i+plan['global_batch']] for i in range(0,len(train),plan['global_batch'])]
                times=[]
                while updates<plan['max_updates']:
                    if STOP or time.time()+max(60,max(times[-3:],default=0)*1.5)>deadline-plan['reserve_seconds']:
                        break
                    rows=batches[updates%len(batches)]
                    info=train_step(model,processor,ctl,tokens,opt,rows,images,
                        lr_at(updates,plan['max_updates'],plan['lr'],plan['warmup_updates'],plan['min_lr_ratio']),plan['clip_norm'])
                    updates+=1;times.append(info['seconds'])
                    info.update(update=updates,presentations=updates*plan['global_batch'],
                        epochs=updates*plan['global_batch']/len(train),ids=[r['id'] for r in rows])
                    append(folder/'learning-curve.jsonl',info)
                    atomic_json(output/'progress.json',dict(design=design,update=updates,total_updates=plan['max_updates']))
                    if updates%plan['checkpoint_every']==0:
                        save_state(state_path,ctl,opt,updates,plan,train_ids,val_ids)
                save_state(state_path,ctl,opt,updates,plan,train_ids,val_ids)
                if STOP or updates<plan['max_updates']:
                    results[design]=dict(status='time_limited',updates=updates)
                else:
                    # Verify serialized reload and the NEXT optimizer update, then
                    # restore the selected pilot state; replay does not count as training.
                    with torch.no_grad(): before_scores=scores_for(model,processor,images[train[0]['id']],tokens)
                    restore_state(state_path,ctl,opt,plan,train_ids,val_ids)
                    with torch.no_grad(): after_scores=scores_for(model,processor,images[train[0]['id']],tokens)
                    torch.testing.assert_close(before_scores,after_scores,atol=0,rtol=0)
                    replay_rows=batches[0]
                    train_step(model,processor,ctl,tokens,opt,replay_rows,images,plan['lr'],plan['clip_norm'])
                    next_digest=tensor_digest(ctl.module.named_parameters())
                    restore_state(state_path,ctl,opt,plan,train_ids,val_ids)
                    train_step(model,processor,ctl,tokens,opt,replay_rows,images,plan['lr'],plan['clip_norm'])
                    if next_digest!=tensor_digest(ctl.module.named_parameters()): raise RuntimeError('Next-update checkpoint replay differs')
                    restore_state(state_path,ctl,opt,plan,train_ids,val_ids)
                    fit=evaluate(model,processor,ctl,tokens,train[:plan['train_monitor_questions']],images)
                    validation=evaluate(model,processor,ctl,tokens,val,images)
                    atomic_json(folder/'final-training.json',fit);atomic_json(folder/'final-validation.json',validation)
                    results[design]=dict(status='passed',updates=updates,unique_questions=min(len(train),updates*plan['global_batch']),
                        presentations=updates*plan['global_batch'],epochs=updates*plan['global_batch']/len(train),
                        training={k:v for k,v in fit.items() if k!='predictions'},
                        validation={k:v for k,v in validation.items() if k!='predictions'},
                        checkpoint_reload_exact=True,next_update_replay_exact=True,
                        examples_per_second=plan['global_batch']/(sum(times)/len(times)) if times else None,
                        checks=checks,benchmark_test=False)
                frozen_check(model,before)
                results[design]['base_bitwise_unchanged']=True
            except Exception as exc:
                results[design]=dict(status='failed',error=repr(exc))
                atomic_json(folder/'failure.json',results[design])
                # Corrupting the original model must abort the complete pilot.
                frozen_check(model,before)
            finally:
                if ctl is not None: ctl.remove()
                model.zero_grad(set_to_none=True)
                torch.cuda.empty_cache()
            atomic_json(folder/'summary.json',results[design])
            print(json.dumps(dict(design=design,**results[design])),flush=True)
        frozen_check(model,before)
        summary=dict(status='passed' if all(r['status']=='passed' for r in results.values()) else 'incomplete_or_failed',
            results=results,job_id=os.environ['SLURM_JOB_ID'],base_bitwise_unchanged=True,
            benchmark_test=False,larger_runs_submitted=False,
            limitation='Correctness/throughput pilot on 64 questions, not evidence of convergence or architecture quality.')
        atomic_json(output/'summary.json',summary)
        print(json.dumps(summary),flush=True)
        return summary


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--deadline',type=float,required=True);parser.add_argument('--resume',action='store_true')
    args=parser.parse_args()
    try:
        result=run(json.loads(args.plan.read_text()),args.output,args.deadline,args.resume)
    except Exception as exc:
        args.output.mkdir(parents=True,exist_ok=True)
        atomic_json(args.output/'failure.json',dict(error=repr(exc)))
        raise
    if result['status']!='passed': raise SystemExit(2)


if __name__=='__main__': main()
