"""Gated 60k/one-epoch connector training and shared frozen-teacher cache.

No scheduler operations. Both pretrained models stay fixed. Immutable committed
checkpoints resume the same shuffled pass; no extra epoch or baseline evaluation.
"""
from __future__ import annotations
import argparse,copy,json,math,os,random,signal,sys,time,traceback
from collections import OrderedDict
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from PIL import Image
from safetensors.torch import save_file,load_file
from pgmvlm.checkpoints import RunLock,atomic_json,commit,pointer_for,resume_checkpoint
from pgmvlm.datasets import read_manifest
from pgmvlm.extensions.pretrained_connectors import DESIGNS,install
from pgmvlm.extensions.probe_diagnostics import tensor_digest
from pgmvlm.pretrained_relations import FrozenRelations,checked
from pgmvlm.models import load_base,load_adapter
from pgmvlm.storage import require_headroom
from tools.input_representation_pilot import digest,resolve_image,checkpoint,append
from tools import pretrained_connector_pilot as common

CRITICAL_FILES=('pgmvlm/extensions/pretrained_connectors.py','pgmvlm/pretrained_relations.py',
 'tools/pretrained_connector_pilot.py','pgmvlm/models.py','pgmvlm/protocols.py','pgmvlm/panels.py','pgmvlm/manifests.py')


def validate_pilot(summary,checks,config):
 if summary.get('passed') is not True or summary.get('complete') is not True or set(summary.get('arms',{}))!=set(DESIGNS):
  raise ValueError('All three pilot arms must pass')
 for design in DESIGNS:
  arm=summary['arms'][design];check=checks[design]
  if arm.get('status')!='passed' or arm.get('updates')!=config['max_updates'] or not arm.get('frozen_weights_unchanged'):
   raise ValueError('Incomplete or unfrozen pilot: '+design)
  for k in ('validation_before','validation_after'):
   v=arm.get(k,{})
   if not v.get('complete') or v.get('count')!=config['validation_questions']:raise ValueError('Partial pilot validation')
  for k in ('passed','deep_gradient','reload_exact','next_update_exact','disabled_exact','manual_native_greedy_parity'):
   if check.get(k) is not True:raise ValueError('Missing pilot check: '+design+'/'+k)


def pilot_gate(plan,repo):
 root=Path(plan['pilot_output'])
 cfg=json.loads((root/'config.json').read_text());summary=json.loads((root/'summary.json').read_text())
 checks={d:json.loads((root/d/'checks.json').read_text()) for d in DESIGNS}
 validate_pilot(summary,checks,cfg)
 teacher=json.loads((root/'teacher-checks.json').read_text())
 if any(teacher.get(k) is not True for k in ('reference_features_exact','head_reconstruction','candidate_permutation')):
  raise ValueError('Reference-teacher checks did not pass')
 for name in ('revision','adapter_hashes','teacher_hashes','teacher_seed','connector'):
  if plan[name]!=cfg[name]:raise ValueError('Changed frozen identity or architecture: '+name)
 for rel in CRITICAL_FILES:
  if digest(repo/rel)!=digest(Path(plan['pilot_source'])/rel):raise ValueError('Changed critical source requires a new pilot: '+rel)
 return dict(passed=True,summary_sha256=digest(root/'summary.json'),pilot_source_commit=cfg['source_commit'])


def selected_rows(plan):
 checked(plan['train_manifest'],plan['train_manifest_sha256']);checked(plan['val_manifest'],plan['val_manifest_sha256'])
 train=read_manifest(plan['train_manifest']);val=read_manifest(plan['val_manifest'])
 if len(train)!=60000 or len(val)!=1000 or plan['max_epochs']!=1:raise ValueError('Requires exactly 60k/1k and one epoch')
 if len({r['id'] for r in train+val})!=61000:raise ValueError('Overlapping or duplicate questions')
 for rows,split in ((train,'train'),(val,'validation')):
  if any(r['split']!=split or r['dataset']!='pgm' or r['regime']!='neutral' or r['choices']!=list('ABCDEFGH') for r in rows):
   raise ValueError('Unexpected split/task')
 return train,val


class Images:
 def __init__(self,rows,plan,capacity=16):
  self.rows={r['id']:r for r in rows};self.plan=plan;self.capacity=capacity;self.cache=OrderedDict()
 def __getitem__(self,key):
  if key not in self.cache:
   with Image.open(resolve_image(self.rows[key],self.plan)) as img:self.cache[key]=img.convert('RGB')
   if len(self.cache)>self.capacity:self.cache.popitem(last=False)
  self.cache.move_to_end(key);return self.cache[key]


def cache_contract(plan):
 return common.contract({k:plan[k] for k in ('teacher_hashes','teacher_seed','train_manifest_sha256','val_manifest_sha256')})


def prepare_cache(plan,output,deadline,resume):
 with RunLock(output):
  gate=pilot_gate(plan,Path(__file__).resolve().parents[1]);train,val=selected_rows(plan);rows=train+val
  identity=cache_contract(plan)
  if (output/'READY.json').exists():
   if json.loads((output/'READY.json').read_text())['contract']!=identity:raise ValueError('Wrong cache identity')
   return
  if (output/'config.json').exists():
   if not resume or json.loads((output/'config.json').read_text())!=plan:raise ValueError('Explicit identical cache resume required')
  else:atomic_json(output/'config.json',plan)
  atomic_json(output/'pilot-gate.json',gate);require_headroom(output)
  teacher=FrozenRelations(plan['teacher_package'],plan['teacher_hashes'])
  params=list(teacher.encoder.named_parameters())+list(teacher.scorer.named_parameters());before=tensor_digest(params)
  images=Images(rows,plan);shards=[];processed=0;start=time.time();chunk=256
  for index,offset in enumerate(range(0,len(rows),chunk)):
   part=rows[offset:offset+chunk];path=output/f'bank-{index:04d}.pt';expected=[r['id'] for r in part]
   if path.exists() and path.with_suffix('.sha256.json').exists():
    checked(path,json.loads(path.with_suffix('.sha256.json').read_text())['sha256'])
    old=torch.load(path,map_location='cpu',weights_only=True)
    if old['contract']!=identity or old['ids']!=expected or tuple(old['bank'].shape)!=(len(part),8,9,32):raise ValueError('Bad cache shard')
   else:
    # Do not strand a half-shard if the allocation is almost over.
    if common.STOP or time.time()>=deadline-plan['reserve_seconds']:break
    banks=[]
    for row in part:
     if common.STOP or time.time()>=deadline-60:break
     banks.append(teacher.extract(images[row['id']],row['id'],plan['teacher_seed'],verify=not teacher.parity_verified)[0])
    if len(banks)!=len(part):break
    checkpoint(path,dict(contract=identity,ids=expected,bank=torch.stack(banks)))
   shards.append(dict(file=path.name,sha256=digest(path),count=len(part)))
   processed+=len(part)
   atomic_json(output/'progress.json',dict(processed=processed,total=len(rows),seconds=time.time()-start,complete=False))
  after=tensor_digest(params)
  if after!=before:raise RuntimeError('Teacher weights changed')
  result=dict(contract=identity,processed=processed,total=len(rows),complete=processed==len(rows),
              teacher_weights_sha256=after,shards=shards,seconds=time.time()-start)
  atomic_json(output/('READY.json' if result['complete'] else 'incomplete.json'),result)
  if not result['complete']:raise SystemExit(75)


class Banks:
 def __init__(self,root,rows,plan,capacity=256):
  self.root=Path(root);ready=json.loads((self.root/'READY.json').read_text())
  if not ready.get('complete') or ready['contract']!=cache_contract(plan) or ready['processed']!=len(rows):raise ValueError('Incomplete/wrong frozen cache')
  self.lookup={};self.files={};self.cache=OrderedDict();self.capacity=capacity;self.contract=ready['contract']
  offset=0
  for item in ready['shards']:
   self.files[item['file']]=item
   for i,row in enumerate(rows[offset:offset+item['count']]):self.lookup[row['id']]=(item['file'],i)
   offset+=item['count']
  if offset!=len(rows):raise ValueError('Cache inventory mismatch')
 def __getitem__(self,key):
  name,index=self.lookup[key]
  if name not in self.cache:
   checked(self.root/name,self.files[name]['sha256'])
   shard=torch.load(self.root/name,map_location='cpu',weights_only=True)
   if shard['contract']!=self.contract:raise ValueError('Cache identity changed')
   self.cache[name]=shard
   if len(self.cache)>self.capacity:self.cache.popitem(last=False)
  self.cache.move_to_end(name);shard=self.cache[name]
  if shard['ids'][index]!=key:raise ValueError('Question/cache index mismatch')
  return shard['bank'][index:index+1]


def lr_at(update,total,peak,warmup,minimum=.1):
 if update<warmup:return peak*(update+1)/max(1,warmup)
 phase=(update-warmup)/max(1,total-warmup-1)
 return peak*(minimum+(1-minimum)*.5*(1+math.cos(math.pi*min(1.,phase))))


def save_training(root,ctl,opt,progress,plan,tag='last'):
 require_headroom(root,reserve_bytes=2*1024**3,reserve_files=500)
 def weights(folder):
  save_file({n:t.detach().cpu().contiguous() for n,t in ctl.module.state_dict().items()},str(folder/'connector.safetensors'))
  atomic_json(folder/'connector.json',ctl.module.spec)
 state=dict(contract=common.contract(plan),step=progress['updates'],progress=copy.deepcopy(progress),optimizer=opt.state_dict(),
            best_checkpoint=progress.get('best_checkpoint'),torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all())
 return commit(root,tag,weights,state=state,selection=dict(update=progress['updates'],validation_loss=progress.get('best_loss')))


def restore_training(root,ctl,opt,plan):
 result=resume_checkpoint(root)
 if result is None:raise ValueError('No committed checkpoint to resume')
 folder,state_path=result;state=torch.load(state_path,map_location='cpu',weights_only=True)
 if state['contract']!=common.contract(plan) or json.loads((folder/'connector.json').read_text())!=ctl.module.spec:raise ValueError('Resume contract changed')
 ctl.module.load_state_dict(load_file(str(folder/'connector.safetensors')),strict=True)
 opt.load_state_dict(state['optimizer']);torch.set_rng_state(state['torch_rng'])
 torch.cuda.set_rng_state_all(state['cuda_rng'])
 for filename in ('learning-curve.jsonl','validation-curve.jsonl'):
  path=Path(root)/filename
  if path.exists():
   retained=[]
   for line in path.read_text().splitlines():
    try:value=json.loads(line)
    except json.JSONDecodeError:continue
    if value['update']<=state['step']:retained.append(line)
   temp=path.with_suffix('.tmp');temp.write_text(''.join(line+'\n' for line in retained));temp.replace(path)
 return state['progress']


def train(plan,design,output,deadline,resume):
 if design not in DESIGNS:raise ValueError('Unknown design')
 plan={**plan,'design':design}
 with RunLock(output):
  gate=pilot_gate(plan,Path(__file__).resolve().parents[1]);rows,val=selected_rows(plan)
  if (output/'config.json').exists():
   if not resume or json.loads((output/'config.json').read_text())!=plan:raise ValueError('Explicit identical training resume required')
  else:atomic_json(output/'config.json',plan)
  atomic_json(output/'pilot-gate.json',gate);require_headroom(output)
  images=Images(rows+val,plan);banks=Banks(plan['cache_output'],rows+val,plan)
  order=list(range(len(rows)));random.Random(plan['training_seed']).shuffle(order)
  if len(rows)%plan['global_batch']:raise ValueError('Use complete batches')
  total=len(rows)//plan['global_batch']
  for name,value in plan['adapter_hashes'].items():checked(Path(plan['adapter'])/name,value)
  model,processor,family=load_base('lfm25',revision=plan['revision'],device='cuda')
  model=load_adapter(model,plan['adapter'],trainable=False);model.eval().requires_grad_(False)
  frozen=tensor_digest(common.base_params(model))
  torch.manual_seed(plan['initialization_seed']);torch.cuda.manual_seed_all(plan['initialization_seed'])
  ctl=install(model.get_base_model(),design,**plan['connector'])
  opt=torch.optim.AdamW(ctl.module.parameters(),lr=plan['lr'],weight_decay=plan['weight_decay'])
  progress=dict(updates=0,best_loss=None,best_update=None,best_checkpoint=None,last_validation=-1,validation_seconds=None)
  if resume:progress=restore_training(output,ctl,opt,plan)
  atomic_json(output/'backend.json',dict(family=family,gpu=torch.cuda.get_device_name(),torch=torch.__version__,
      base_and_lora_sha256=frozen,trainable_parameters=sum(p.numel() for p in ctl.module.parameters()),spec=ctl.module.spec))
  def validation():
   start=time.time();result=common.evaluate(model,ctl,processor,val,images,banks,deadline-60)
   u=progress['updates'];atomic_json(output/f'validation-{u:06d}.json',result)
   append(output/'validation-curve.jsonl',dict(update=u,unique_questions=u*plan['global_batch'],epoch=u/total,
          **{k:v for k,v in result.items() if k!='predictions'}))
   if result['complete']:
    progress['last_validation']=u;progress['validation_seconds']=time.time()-start
    if progress['best_loss'] is None or result['suffix_loss']<progress['best_loss']:
     progress['best_loss']=result['suffix_loss'];progress['best_update']=u
     progress['best_checkpoint']=save_training(output,ctl,opt,progress,plan,tag='best')
     atomic_json(output/'best-validation.json',result)
   return result['complete']
  # No fresh full baseline evaluation: first full validation follows 15k training questions.
  save_training(output,ctl,opt,progress,plan)
  times=[]
  while progress['updates']<total:
   reserve=max(plan['reserve_seconds'],(progress['validation_seconds'] or plan['reserve_seconds'])*1.5+120)
   if common.STOP or time.time()+max(60,max(times[-4:],default=0)*1.5)>=deadline-reserve:break
   u=progress['updates'];batch=[rows[i] for i in order[u*plan['global_batch']:(u+1)*plan['global_batch']]]
   info=common.step(model,ctl,processor,opt,batch,images,banks,lr_at(u,total,plan['lr'],plan['warmup_updates']))
   progress['updates']+=1;times.append(info['seconds']);u=progress['updates']
   info.update(update=u,unique_questions=u*len(batch),epoch=u/total)
   append(output/'learning-curve.jsonl',info)
   if u%25==0:atomic_json(output/'progress.json',dict(**progress,epoch=u/total,unique_questions=u*len(batch),
       mean_update_seconds=sum(times[-25:])/len(times[-25:]),remaining_training_seconds=(total-u)*sum(times[-25:])/len(times[-25:])))
   if u%plan['checkpoint_every']==0:save_training(output,ctl,opt,progress,plan)
   if u%plan['validation_every']==0:
    validation();save_training(output,ctl,opt,progress,plan)
  if not common.STOP and progress['last_validation']!=progress['updates']:validation()
  save_training(output,ctl,opt,progress,plan)
  common.check_frozen(model,frozen)
  complete=progress['updates']==total and progress['last_validation']==total
  if complete:save_training(output,ctl,opt,progress,plan,tag='final')
  atomic_json(output/'summary.json',dict(status='completed' if complete else 'incomplete',**progress,
    epoch=progress['updates']/total,unique_questions=progress['updates']*plan['global_batch'],
    fraction_of_official_train=progress['updates']*plan['global_batch']/1200000,train_pool=60000,validation_pool=1000,
    frozen_weights_unchanged=True,benchmark_test=False,stop_reason='one_epoch' if complete else 'allocation_limit_or_signal'))
  ctl.remove()


def main():
 p=argparse.ArgumentParser();p.add_argument('--mode',choices=['cache','train'],required=True)
 p.add_argument('--design',choices=DESIGNS);p.add_argument('--plan',required=True);p.add_argument('--output',type=Path,required=True)
 p.add_argument('--deadline',type=float,required=True);p.add_argument('--resume',action='store_true');a=p.parse_args()
 if not os.environ.get('SLURM_JOB_ID') or not torch.cuda.is_available():raise RuntimeError('GPU allocation required')
 torch.set_num_threads(2);torch.use_deterministic_algorithms(True)
 torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
 for sig in (signal.SIGUSR1,signal.SIGTERM,signal.SIGINT):signal.signal(sig,common.stopped)
 plan=json.loads(Path(a.plan).read_text())
 try:
  if a.mode=='cache':prepare_cache(plan,a.output,a.deadline,a.resume)
  else:train(plan,a.design,a.output,a.deadline,a.resume)
 except Exception as exc:
  if a.output.exists():atomic_json(a.output/'failure.json',dict(error=str(exc),traceback=traceback.format_exc()))
  raise

if __name__=='__main__':main()
