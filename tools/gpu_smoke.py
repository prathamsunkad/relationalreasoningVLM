"""Bounded real-model GPU checks through the public train/eval entry points.

Requires a Slurm GPU allocation and small, disjoint portable manifests. This is
pipeline verification, not an accuracy benchmark. Does not submit jobs or fetch
weights; pre-populate HF_HOME and set HF_HUB_OFFLINE=1.
"""
import argparse
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import socket
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
CASES = {
    'lfm25_lora': ('lfm25', 'lora', 'none'),
    'smolvlm2_lora': ('smolvlm2', 'lora', 'none'),
    'spatialevo_lora': ('spatialevo', 'lora', 'none'),
    'lfm25_plugin_frozen': ('lfm25', 'module', 'rca_plugin'),
    'lfm25_late_joint': ('lfm25', 'lora_module', 'late'),
    'lfm25_memory_frozen': ('lfm25', 'module', 'rca_memory'),
}


def write_json(path, value):
    path=Path(path)
    tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(value, indent=2)+'\n')
    tmp.replace(path)


def compare_weights(left, right):
    import torch
    from safetensors.torch import load_file
    names={p.name for p in Path(left).glob('*.safetensors')}
    if not names or names != {p.name for p in Path(right).glob('*.safetensors')}:
        raise AssertionError('Trainable checkpoint files differ')
    maximum=0.
    exact=True
    changed=0
    tensors=0
    for filename in sorted(names):
        a,b=load_file(str(Path(left)/filename)),load_file(str(Path(right)/filename))
        if a.keys()!=b.keys():raise AssertionError('Checkpoint tensor keys differ')
        for name,x in a.items():
            y=b[name]
            if not torch.isfinite(x).all() or not torch.isfinite(y).all():
                raise AssertionError('Nonfinite checkpoint weights')
            difference=float((x.float()-y.float()).abs().max()) if x.numel() else 0.
            maximum=max(maximum,difference)
            tensors+=1
            changed+=int(not torch.equal(x,y))
            exact &= torch.equal(x,y)
    return dict(exact=exact,max_abs_difference=maximum,changed_tensors=changed,tensors=tensors)


def compare_predictions(left,right):
    a=json.loads(Path(left).read_text());b=json.loads(Path(right).read_text())
    if len(a)!=len(b) or not a:raise AssertionError('Evaluation sizes differ or are empty')
    maximum=0.
    for x,y in zip(a,b):
        for k in ('file','target','pred','raw_output','choices'):
            if x.get(k)!=y.get(k):raise AssertionError(f'Reloaded evaluation differs: {k}')
        for k in ('loss','answer_loss','suffix_loss'):
            if not math.isfinite(x[k]) or not math.isfinite(y[k]):raise AssertionError('Nonfinite evaluation loss')
            maximum=max(maximum,abs(x[k]-y[k]))
    if maximum>1e-6:raise AssertionError(f'Reloaded loss differs by {maximum}')
    return dict(questions=len(a),predictions_identical=True,max_loss_difference=maximum)


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--train_manifest',required=True,type=Path)
    ap.add_argument('--val_manifest',required=True,type=Path)
    ap.add_argument('--output',required=True,type=Path)
    ap.add_argument('--cases',nargs='+',choices=list(CASES),default=list(CASES))
    ap.add_argument('--seconds',type=int,default=1080)
    args=ap.parse_args()
    if not os.environ.get('SLURM_JOB_ID'):raise SystemExit('Run inside a Slurm GPU allocation')
    import torch
    if not torch.cuda.is_available():raise SystemExit('Allocated GPU is not visible to PyTorch')
    from pgmvlm.datasets import read_manifest,ManifestStream
    train,val=read_manifest(args.train_manifest),read_manifest(args.val_manifest)
    if not 4<=len(train)<=32 or not 2<=len(val)<=8:
        raise ValueError('Smoke manifests must contain 4–32 training and 2–8 validation questions')
    if {r['id'] for r in train}&{r['id'] for r in val}:raise ValueError('Train/validation overlap')
    for path,rows in [(args.train_manifest,train),(args.val_manifest,val)]:
        stream=ManifestStream(path,[r['id'] for r in rows])
        for i in range(len(stream)):stream[i]  # verify all PNGs before loading large models
    out=args.output.resolve()
    out.mkdir(parents=True,exist_ok=True)
    if (out/'summary.json').exists():raise ValueError('Use a fresh smoke output directory')
    started=time.time();deadline=started+args.seconds
    result=dict(status='running',job_id=os.environ['SLURM_JOB_ID'],hostname=socket.gethostname(),
        gpu=torch.cuda.get_device_name(),cuda=torch.version.cuda,started_unix=started,
        versions={k:importlib.metadata.version(k) for k in ['torch','transformers','peft','accelerate','safetensors']},
        source_commit=os.environ.get('VLM_SOURCE_COMMIT'),
        data={str(path.name):dict(count=len(rows),sha256=hashlib.sha256(path.read_bytes()).hexdigest(),ids=[r['id'] for r in rows])
              for path,rows in [(args.train_manifest,train),(args.val_manifest,val)]},
        cases={},commands=[])
    write_json(out/'summary.json',result)
    def run(name,script,flags):
        left=deadline-time.time()
        if left<30:raise TimeoutError('Smoke time reserve reached')
        cmd=[sys.executable,'-u',str(ROOT/script),*map(str,flags)]
        log=out/(name+'.log')
        begin=time.time()
        with log.open('w') as stream:
            proc=subprocess.run(cmd,cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT,timeout=left)
        result['commands'].append(dict(name=name,argv=cmd,returncode=proc.returncode,seconds=time.time()-begin))
        write_json(out/'summary.json',result)
        if proc.returncode:raise RuntimeError(f'{name} failed (exit {proc.returncode}); see {log.name}')
        return log.read_text()
    try:
        for case in args.cases:
            model,mode,extension=CASES[case]
            directory=out/case
            directory.mkdir()
            common=['--model_id',model,'--protocol','answer_letter','--dtype','bf16','--attn','sdpa',
                '--num_workers','0','--max_new_tokens','16','--extension',extension]
            if extension in ('rca_plugin','rca_memory'):common+=['--correction_scale','0.01']
            train_flags=common+['--train_manifest',args.train_manifest,'--val_manifest',args.val_manifest,
                '--train_mode',mode,'--batch_size','1','--global_batch','2','--max_steps','2','--warmup_steps','0',
                '--lr','0.0001','--lora_r','16','--lora_alpha','32','--lora_dropout','0.05',
                '--val_samples',str(len(val)),'--val_batch_size','1','--val_every','2','--save_every','1',
                '--log_every','1','--seed','20261119','--deterministic','--check_lustre_quota']
            resumed=directory/'resumed';reference=directory/'reference'
            log=run(case+'-interrupt','lora/train_lora.py',train_flags+['--out_dir',resumed,'--time_limit_min','0.000001'])
            if 'time budget reached at step 1' not in log:raise AssertionError('Expected a controlled stop at update 1')
            # Preserve just the first trainable weights to prove an actual subsequent update.
            import shutil
            first=directory/'first-update';first.mkdir()
            for file in (resumed/'last').glob('*.safetensors'):shutil.copy2(file,first/file.name)
            log=run(case+'-resume','lora/train_lora.py',train_flags+['--out_dir',resumed])
            if 'resumed from step 1' not in log:raise AssertionError('Optimizer continuation did not occur')
            run(case+'-reference','lora/train_lora.py',train_flags+['--out_dir',reference])
            replay=compare_weights(resumed/'final',reference/'final')
            if not replay['exact']:raise AssertionError(f'{case}: replay weights differ: {replay}')
            update=compare_weights(first,resumed/'final')
            if not update['changed_tensors']:raise AssertionError('No parameters changed on update 2')
            if mode=='module' and (resumed/'final/adapter_config.json').exists():
                raise AssertionError('Frozen-module case unexpectedly saved LoRA')
            for repeat in ('a','b'):
                ev=directory/('eval-'+repeat)
                run(case+'-eval-'+repeat,'lora/eval_vlm.py',common+['--adapter',resumed/'best',
                    '--manifest',args.val_manifest,'--out_dir',ev,'--batch_size','1','--flush_every','1'])
                run(case+'-merge-'+repeat,'lora/eval_vlm.py',['--merge_dir',ev,'--expected_n',str(len(val))])
                if not json.loads((ev/'summary.json').read_text())['complete']:raise AssertionError('Incomplete evaluation')
            reload=compare_predictions(directory/'eval-a/results_shard0.json',directory/'eval-b/results_shard0.json')
            training_args=json.loads((resumed/'args.json').read_text())
            evaluation=json.loads((directory/'eval-a/summary.json').read_text())
            result['cases'][case]=dict(status='passed',model=model,train_mode=mode,extension=extension,
                updates=2,presentations=4,replay=replay,update=update,reload=reload,
                trainable_params=training_args['trainable_params'],total_params=training_args['total_params'],
                validation_accuracy=evaluation['scores']['acc'],validation_loss=evaluation['scores']['loss'])
            write_json(out/'summary.json',result)
            print(json.dumps(dict(case=case,status='passed')),flush=True)
        result['status']='passed'
    except Exception as exc:
        result.update(status='failed',error=f'{type(exc).__name__}: {exc}')
        raise
    finally:
        result['elapsed_seconds']=time.time()-started
        write_json(out/'summary.json',result)

if __name__=='__main__':main()
