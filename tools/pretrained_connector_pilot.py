"""One bounded shared GPU pilot; never submits follow-up jobs.

Existing pretrained VLM+LoRA and reference Slot Abstractor stay frozen.
Only connectors update. No standalone benchmark, overlap or ensemble evaluation.
"""
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
import traceback

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from PIL import Image
from pgmvlm.models import load_base,load_adapter
from pgmvlm.protocols import encode_answer,answer_losses,parse_answer
from pgmvlm.manifests import PGM_PROMPT
from pgmvlm.extensions.pretrained_connectors import DESIGNS,install
from pgmvlm.extensions.probe_diagnostics import tensor_digest
from pgmvlm.pretrained_relations import FrozenRelations,checked
from pgmvlm.storage import require_headroom
from tools.input_representation_pilot import load_selection,resolve_image,atomic_json,checkpoint,append,digest

STOP=False

def stopped(*args):
    global STOP
    STOP=True


def contract(plan):
    return hashlib.sha256(json.dumps(plan,sort_keys=True,separators=(',',':')).encode()).hexdigest()


def base_params(model):
    return [(n,p) for n,p in model.named_parameters() if 'pretrained_connector.' not in n]


def check_frozen(model,expected):
    params=base_params(model)
    if any(p.requires_grad or p.grad is not None for _,p in params):
        raise RuntimeError('Frozen VLM/LoRA received trainable weights or parameter gradients')
    if tensor_digest(params)!=expected: raise RuntimeError('Frozen VLM/LoRA weights changed')


def gradient_report(module):
    result={n:float(p.grad.float().square().mean().sqrt()) for n,p in module.named_parameters() if p.grad is not None}
    if not result or not all(math.isfinite(x) for x in result.values()):
        raise RuntimeError('Absent or nonfinite connector gradients')
    return result


def encode(processor,image,answer,device):
    return encode_answer(processor,image,PGM_PROMPT,answer,device,torch.bfloat16)


def losses(model,ctl,processor,row,images,banks):
    _,full,start=encode(processor,images[row['id']],row['answer'],next(model.parameters()).device)
    with ctl.question(banks[row['id']].to(next(model.parameters()).device)):
        suffix,answer,logits=answer_losses(model,full,start)
    return suffix,answer,(logits.argmax(-1)==full['input_ids'][:,start]).float().mean()


def step(model,ctl,processor,optimizer,rows,images,banks,lr,clip=1.):
    optimizer.zero_grad(set_to_none=True)
    for group in optimizer.param_groups: group['lr']=lr
    total=[0.,0.,0.];started=time.time()
    for row in rows:
        suffix,answer,accuracy=losses(model,ctl,processor,row,images,banks)
        if not torch.isfinite(suffix): raise RuntimeError('Nonfinite answer/suffix loss')
        (suffix/len(rows)).backward()
        for i,value in enumerate((suffix,answer,accuracy)): total[i]+=float(value.detach())/len(rows)
    gradients=gradient_report(ctl.module)
    norm=torch.nn.utils.clip_grad_norm_(ctl.module.parameters(),clip,error_if_nonfinite=True)
    optimizer.step()
    return dict(suffix_loss=total[0],answer_loss=total[1],first_token_accuracy=total[2],
                gradient_rms=gradients,gradient_norm=float(norm),lr=lr,seconds=time.time()-started)


@torch.no_grad()
def generate(model,ctl,processor,row,image,bank):
    prefix,_,_=encode(processor,image,row['answer'],next(model.parameters()).device)
    count=prefix['input_ids'].shape[1]
    # Explicit full-prefix decoding avoids hidden cache differences across interfaces.
    # Preserve native multi-tile image tensors and the native EOS IDs.
    eos=model.generation_config.eos_token_id or processor.tokenizer.eos_token_id
    eos={eos} if isinstance(eos,int) else set(eos)
    with ctl.question(bank.to(next(model.parameters()).device)):
        for _ in range(16):
            next_token=model(**prefix,use_cache=False,logits_to_keep=1).logits[:,-1].argmax(-1,keepdim=True)
            prefix['input_ids']=torch.cat((prefix['input_ids'],next_token),1)
            prefix['attention_mask']=torch.ones_like(prefix['input_ids'])
            if int(next_token) in eos: break
    raw=processor.tokenizer.decode(prefix['input_ids'][0,count:],skip_special_tokens=True).strip()
    return raw,parse_answer(raw,list('ABCDEFGH'))


@torch.no_grad()
def evaluate(model,ctl,processor,rows,images,banks,deadline):
    records=[];ctl.module.eval()
    for row in rows:
        if STOP or time.time()>=deadline: break
        suffix,answer,accuracy=losses(model,ctl,processor,row,images,banks)
        raw,pred=generate(model,ctl,processor,row,images[row['id']],banks[row['id']])
        records.append(dict(id=row['id'],answer=row['answer'],prediction=pred,raw=raw,
                            correct=pred==row['answer'],suffix_loss=float(suffix),answer_loss=float(answer)))
    ctl.module.train()
    n=len(records)
    return dict(count=n,expected=len(rows),complete=n==len(rows),
                accuracy=sum(r['correct'] for r in records)/n if n else None,
                suffix_loss=sum(r['suffix_loss'] for r in records)/n if n else None,
                answer_loss=sum(r['answer_loss'] for r in records)/n if n else None,predictions=records)


def save(path,ctl,optimizer,plan,updates):
    require_headroom(path.parent,reserve_bytes=2*1024**3,reserve_files=500)
    checkpoint(path,dict(version=1,contract=contract(plan),spec=ctl.module.spec,
        module=ctl.module.state_dict(),optimizer=optimizer.state_dict(),updates=updates,
        torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all()))


def restore(path,ctl,optimizer,plan):
    checked(path,json.loads(path.with_suffix('.sha256.json').read_text())['sha256'])
    state=torch.load(path,map_location=next(ctl.module.parameters()).device,weights_only=True)
    if state['contract']!=contract(plan) or state['spec']!=ctl.module.spec:
        raise ValueError('Checkpoint contract changed')
    ctl.module.load_state_dict(state['module'],strict=True);optimizer.load_state_dict(state['optimizer'])
    torch.set_rng_state(state['torch_rng'].cpu())
    torch.cuda.set_rng_state_all([t.cpu() for t in state['cuda_rng']])
    return state['updates']


def correctness(model,base,ctl,processor,row,images,banks,plan,out):
    initial=copy.deepcopy(ctl.module.state_dict())
    _,full,start=encode(processor,images[row['id']],row['answer'],next(model.parameters()).device)
    bank=banks[row['id']].to(next(model.parameters()).device)
    def logits():
        with ctl.question(bank): return model(**full,use_cache=False,logits_to_keep=4).logits.detach()
    with torch.no_grad():
        ctl.set_enabled(False);disabled=logits()
        ctl.set_enabled(True);first=logits()
        if ctl.module.design!='tokens': torch.testing.assert_close(disabled,first,rtol=0,atol=0)
    opt=torch.optim.AdamW(ctl.module.parameters(),lr=plan['lr'],weight_decay=plan['weight_decay'])
    checks=[]
    for _ in range(3): checks.append(step(model,ctl,processor,opt,[row],images,banks,plan['lr']))
    if not checks[-1]['gradient_rms'].get('input.weight',0)>0:
        raise RuntimeError('Deep connector projection gradient did not open after three updates')
    save(out/'replay.pt',ctl,opt,plan,3)
    expected=logits()
    step(model,ctl,processor,opt,[row],images,banks,plan['lr'])
    next_state=copy.deepcopy(ctl.module.state_dict())
    restore(out/'replay.pt',ctl,opt,plan)
    torch.testing.assert_close(expected,logits(),rtol=0,atol=0)
    step(model,ctl,processor,opt,[row],images,banks,plan['lr'])
    for key,value in ctl.module.state_dict().items():
        torch.testing.assert_close(next_state[key],value,rtol=0,atol=0)
    ctl.module.load_state_dict(initial);ctl.module.zero_grad(set_to_none=True)
    # Disable path is exact, including removal/reinstallation (native tests).
    ctl.set_enabled(False)
    torch.testing.assert_close(disabled,logits(),rtol=0,atol=0)
    # One-question inference plumbing check, not a standalone evaluation.
    prefix,_,_=encode(processor,images[row['id']],row['answer'],next(model.parameters()).device)
    with torch.no_grad():
        native=model.generate(**prefix,max_new_tokens=16,do_sample=False,use_cache=False)
        manual,_=generate(model,ctl,processor,row,images[row['id']],banks[row['id']])
    raw=processor.tokenizer.decode(native[0,prefix['input_ids'].shape[1]:],skip_special_tokens=True).strip()
    if manual!=raw: raise RuntimeError('Manual/native greedy decode mismatch')
    ctl.set_enabled(True)
    return dict(passed=True,deep_gradient=True,reload_exact=True,next_update_exact=True,
                disabled_exact=True,manual_native_greedy_parity=True,
                initial_native_max_logit_shift=float((first-disabled).abs().max()),checks=checks)


def run(plan,output,deadline):
    if not os.environ.get('SLURM_JOB_ID') or not torch.cuda.is_available():
        raise RuntimeError('Run inside a GPU allocation, never on a login node')
    if plan['designs']!=list(DESIGNS) or plan['train_questions']>64 or plan['validation_questions']>32:
        raise ValueError('This entry point is restricted to the shared correctness pilot')
    if plan['max_updates']>16 or plan['train_questions']%plan['global_batch']:
        raise ValueError('Invalid pilot budget/batching')
    output.mkdir(parents=True,exist_ok=True)
    with (output/'.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if (output/'config.json').exists(): raise ValueError('Existing output: refuse accidental restart')
        atomic_json(output/'config.json',plan)
        atomic_json(output/'storage.json',require_headroom(output,reserve_bytes=8*1024**3,reserve_files=2000))
        for sig in (signal.SIGUSR1,signal.SIGTERM,signal.SIGINT): signal.signal(sig,stopped)
        torch.set_num_threads(2);torch.use_deterministic_algorithms(True)
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        train,val,_=load_selection(plan)
        random.Random(plan['training_seed']).shuffle(train)
        train=train[:plan['train_questions']];val=val[:plan['validation_questions']]
        atomic_json(output/'question_ids.json',dict(train=[r['id'] for r in train],validation=[r['id'] for r in val],
                 parent_sha256=plan['selected_rows_sha256']))
        images={}
        for row in train+val:
            with Image.open(resolve_image(row,plan)) as image: images[row['id']]=image.convert('RGB')
        print('Loading frozen reference teacher',flush=True)
        teacher=FrozenRelations(plan['teacher_package'],plan['teacher_hashes'])
        teacher_before=tensor_digest(list(teacher.encoder.named_parameters())+list(teacher.scorer.named_parameters()))
        banks={};started=time.time()
        for i,row in enumerate(train+val):
            if STOP or time.time()>=deadline-plan['reserve_seconds']: raise TimeoutError('Teacher cache time limit')
            banks[row['id']]=teacher.extract(images[row['id']],row['id'],plan['teacher_seed'],verify=i==0)
            if (i+1)%16==0: print(f'Cached {i+1}/{len(train+val)} questions',flush=True)
        teacher_after=tensor_digest(list(teacher.encoder.named_parameters())+list(teacher.scorer.named_parameters()))
        if teacher_before!=teacher_after: raise RuntimeError('Frozen teacher changed')
        checkpoint(output/'relation-cache.pt',dict(contract=contract(plan),banks=banks))
        atomic_json(output/'teacher-checks.json',dict(reference_features_exact=True,head_reconstruction=True,
             candidate_permutation=True,frozen_weights_sha256=teacher_after,seconds=time.time()-started,
             shape=[1,8,9,32],pooling='mean 16 post-Abstractor slots within each panel'))
        del teacher;torch.cuda.empty_cache()
        for name,digest_value in plan['adapter_hashes'].items(): checked(Path(plan['adapter'])/name,digest_value)
        print('Loading best frozen PGM LFM and LoRA',flush=True)
        model,processor,family=load_base('lfm25',revision=plan['revision'],device='cuda')
        model=load_adapter(model,plan['adapter'],trainable=False)
        model.eval().requires_grad_(False);base=model.get_base_model()
        from peft import get_peft_model_state_dict
        from safetensors.torch import load_file
        expected_adapter=load_file(str(Path(plan['adapter'])/'adapter_model.safetensors'))
        loaded_adapter=get_peft_model_state_dict(model)
        if set(expected_adapter)!=set(loaded_adapter): raise RuntimeError('Adapter tensor keys changed on load')
        for name,value in expected_adapter.items():
            torch.testing.assert_close(value.to(device=loaded_adapter[name].device,dtype=loaded_adapter[name].dtype),
                                       loaded_adapter[name],rtol=0,atol=0)
        del expected_adapter,loaded_adapter
        frozen=tensor_digest(base_params(model))
        import transformers,peft,torchvision
        atomic_json(output/'backend.json',dict(torch=torch.__version__,transformers=transformers.__version__,
             peft=peft.__version__,torchvision=torchvision.__version__,gpu=torch.cuda.get_device_name(),
             family=family,prompt=PGM_PROMPT,base_and_lora_sha256=frozen,native_processor=processor.to_dict(),
             inference='full-prefix greedy, no KV cache, 16 tokens',teacher_features_sha256=digest(output/'relation-cache.pt')))
        summaries={}
        for design in DESIGNS:
            out=output/design;out.mkdir(exist_ok=True)
            if STOP or time.time()>=deadline-plan['reserve_seconds']:
                summaries[design]=dict(status='not_started_time_limit');continue
            torch.manual_seed(plan['initialization_seed']);torch.cuda.manual_seed_all(plan['initialization_seed'])
            ctl=install(base,design,**plan['connector']);ctl.module.train()
            try:
                atomic_json(out/'config.json',dict(plan=plan,spec=ctl.module.spec,
                     trainable_parameters=sum(p.numel() for p in ctl.module.parameters())))
                checks=correctness(model,base,ctl,processor,train[0],images,banks,plan,out)
                atomic_json(out/'checks.json',checks);check_frozen(model,frozen)
                opt=torch.optim.AdamW(ctl.module.parameters(),lr=plan['lr'],weight_decay=plan['weight_decay'])
                initial=evaluate(model,ctl,processor,val,images,banks,deadline)
                atomic_json(out/'initial-validation.json',initial)
                save(out/'latest.pt',ctl,opt,plan,0)
                updates=0
                for u in range(plan['max_updates']):
                    if STOP or time.time()>=deadline-plan['reserve_seconds']: break
                    batch=train[u*plan['global_batch']:(u+1)*plan['global_batch']]
                    warmup=plan['warmup_updates']
                    fraction=max(0,u-warmup)/max(1,plan['max_updates']-warmup-1)
                    lr=plan['lr']*((u+1)/warmup if u<warmup else .1+.9*.5*(1+math.cos(math.pi*fraction)))
                    record=step(model,ctl,processor,opt,batch,images,banks,lr)
                    updates=u+1
                    record.update(update=updates,presentations=updates*len(batch))
                    append(out/'learning-curve.jsonl',record)
                    if updates%4==0: save(out/'latest.pt',ctl,opt,plan,updates)
                    print(f'{design} {updates}/{plan["max_updates"]}: suffix CE {record["suffix_loss"]:.4f}',flush=True)
                save(out/'latest.pt',ctl,opt,plan,updates)
                final=evaluate(model,ctl,processor,val,images,banks,deadline)
                monitor=evaluate(model,ctl,processor,train[:16],images,banks,deadline)
                atomic_json(out/'final-validation.json',final);atomic_json(out/'train-monitor.json',monitor)
                check_frozen(model,frozen)
                complete=updates==plan['max_updates'] and initial['complete'] and final['complete'] and monitor['complete']
                summaries[design]=dict(status='passed' if complete else 'incomplete_time_limit',updates=updates,
                    presentations=updates*plan['global_batch'],validation_before={k:v for k,v in initial.items() if k!='predictions'},
                    validation_after={k:v for k,v in final.items() if k!='predictions'},frozen_weights_unchanged=True,
                    checkpoint='latest.pt',checkpoint_policy='final common update, no validation selection in correctness pilot')
            except Exception as exc:
                summaries[design]=dict(status='failed',error=str(exc),traceback=traceback.format_exc())
                print(summaries[design]['traceback'],flush=True)
                # Fail closed on any possible base mutation before trying another connector.
                check_frozen(model,frozen)
            finally:
                ctl.remove();torch.cuda.empty_cache()
            atomic_json(out/'summary.json',summaries[design])
            atomic_json(output/'summary.json',dict(arms=summaries,complete=False))
        passed=len(summaries)==3 and all(v['status']=='passed' for v in summaries.values())
        atomic_json(output/'summary.json',dict(arms=summaries,complete=True,passed=passed,
                   benchmark_test=False,larger_runs_submitted=False))
        if not passed: raise RuntimeError('At least one connector did not complete/pass; inspect summary')


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--plan',required=True);parser.add_argument('--output',required=True)
    parser.add_argument('--deadline',required=True,type=float)
    args=parser.parse_args();out=Path(args.output)
    try: run(json.loads(Path(args.plan).read_text()),out,args.deadline)
    except Exception as exc:
        if out.exists(): atomic_json(out/'failure.json',dict(error=str(exc),traceback=traceback.format_exc()))
        raise

if __name__=='__main__': main()
