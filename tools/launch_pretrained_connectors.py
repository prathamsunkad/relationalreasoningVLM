"""Idempotent hourly advance: passing pilot -> cache -> one epoch per connector.

Run on the login node with the frozen source snapshot and resolved plan. Only
clean allocation-limit continuations are automatic; material failures stop.
"""
import argparse,fcntl,hashlib,json,math,os,subprocess,time
from pathlib import Path

DESIGNS=('reader','kv_memory','tokens')
CRITICAL=('pgmvlm/extensions/pretrained_connectors.py','pgmvlm/pretrained_relations.py',
 'tools/pretrained_connector_pilot.py','pgmvlm/models.py','pgmvlm/protocols.py','pgmvlm/panels.py','pgmvlm/manifests.py')

def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
 return h.hexdigest()

def read(p):return json.loads(Path(p).read_text())

def write(p,value):
 temp=p.with_suffix('.tmp');temp.write_text(json.dumps(value,indent=2)+'\n');temp.replace(p)

def job_state(job):
 result=subprocess.run(['sacct','-j',str(job),'--format=JobIDRaw,State','-n','-P'],check=True,capture_output=True,text=True)
 for line in result.stdout.splitlines():
  parts=line.split('|')
  if parts[0]==str(job):return parts[1].split()[0]
 return 'UNKNOWN'

def pilot_passed(plan,repo):
 state=job_state(plan['pilot_job_id'])
 if state!='COMPLETED':return False,'pilot '+state
 root=Path(plan['pilot_output'])
 if not (root/'summary.json').exists():return False,'pilot lacks summary'
 summary=read(root/'summary.json');cfg=read(root/'config.json')
 if not summary.get('passed') or not summary.get('complete') or set(summary.get('arms',{}))!=set(DESIGNS):return False,'pilot failed/incomplete'
 for d in DESIGNS:
  arm=summary['arms'][d];checks=read(root/d/'checks.json')
  if arm.get('status')!='passed' or arm.get('updates')!=cfg['max_updates'] or not arm.get('frozen_weights_unchanged'):return False,'pilot arm incomplete'
  for k in ('validation_before','validation_after'):
   if not arm[k]['complete'] or arm[k]['count']!=cfg['validation_questions']:return False,'partial pilot validation'
  if not all(checks.get(k) is True for k in ('passed','deep_gradient','reload_exact','next_update_exact','disabled_exact','manual_native_greedy_parity')):return False,'pilot checks failed'
 teacher=read(root/'teacher-checks.json')
 if not all(teacher.get(k) is True for k in ('reference_features_exact','head_reconstruction','candidate_permutation')):return False,'teacher checks failed'
 for k in ('revision','adapter_hashes','teacher_hashes','teacher_seed','connector'):
  if cfg[k]!=plan[k]:raise ValueError('Pilot/model mismatch: '+k)
 for rel in CRITICAL:
  if sha(repo/rel)!=sha(Path(plan['pilot_source'])/rel):raise ValueError('Changed critical source: '+rel)
 return True,'passed'

def minutes_for(seconds):
 return max(60,min(960,math.ceil(seconds/900)*15))

def estimates(plan):
 root=Path(plan['pilot_output']);teacher=read(root/'teacher-checks.json');cfg=read(root/'config.json')
 values={'cache':minutes_for(teacher['seconds']/(cfg['train_questions']+cfg['validation_questions'])*61000*1.3+900)}
 for d in DESIGNS:
  curve=[json.loads(x) for x in (root/d/'learning-curve.jsonl').read_text().splitlines() if x]
  per_question=max(x['seconds']/cfg['global_batch'] for x in curve[2:])
  # Reserve for four complete 1k validation passes and I/O/load variance.
  values[d]=minutes_for(per_question*60000*1.3+7200)
 return values

def advance(plan_path,repo,root):
 plan=read(plan_path)
 if plan['max_epochs']!=1 or plan['train_questions']!=60000 or plan['validation_questions']!=1000:raise ValueError('Unauthorized training scope')
 root.mkdir(parents=True,exist_ok=True)
 with (root/'.launch.lock').open('a') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  passed,why=pilot_passed(plan,repo)
  if not passed:
   print(json.dumps({'action':'wait','reason':why}));return
  if sha(plan['train_manifest'])!=plan['train_manifest_sha256'] or sha(plan['val_manifest'])!=plan['val_manifest_sha256']:raise ValueError('Data manifest changed')
  jobs=read(root/'jobs.json') if (root/'jobs.json').exists() else {'source_commit':plan['source_commit'],'jobs':[]}
  estimates_by_arm=estimates(plan)
  def enqueue(mode,design=None,resume=False):
   name='cache' if mode=='cache' else design
   prior=[j for j in jobs['jobs'] if j['name']==name]
   intent=root/f'submission-{name}-{len(prior)+1:02d}.json'
   if intent.exists():raise RuntimeError('Unresolved submission intent; inspect scheduler before retrying: '+str(intent))
   minutes=estimates_by_arm[name]
   output=Path(plan['cache_output']) if mode=='cache' else root/design
   env=os.environ.copy();env.update(CONNECTOR_REPO=str(repo),CONNECTOR_PLAN=str(plan_path),CONNECTOR_OUTPUT=str(output),
    CONNECTOR_MODE=mode,CONNECTOR_DESIGN=design or '',CONNECTOR_RESUME=str(int(resume)),CONNECTOR_SECONDS=str(minutes*60),
    VLM_PYTHON=plan['python'],HF_HOME=plan['hf_home'],PYTHONPATH=plan['runtime'])
   cmd=['sbatch','--parsable','--export=ALL','--job-name=lfm-connect60k-'+name,'--time='+str(minutes),
        '--output='+str(root/(name+'-%j.out')),'--error='+str(root/(name+'-%j.err')),
        str(repo/'slurm_portable/slurm/pretrained_connectors_train.sbatch')]
   write(intent,dict(command=cmd,unix=time.time()))
   result=subprocess.run(cmd,env=env,check=True,capture_output=True,text=True)
   job=result.stdout.strip().split(';')[0]
   if not job.isdigit():raise RuntimeError('Unknown sbatch response')
   receipt=dict(job_id=job,name=name,mode=mode,minutes=minutes,resume=resume,output=str(output),unix=time.time())
   jobs['jobs'].append(receipt);write(root/'jobs.json',jobs);print(json.dumps({'action':'submitted',**receipt}))
  cache=Path(plan['cache_output'])
  if not (cache/'READY.json').exists():
   existing=[j for j in jobs['jobs'] if j['name']=='cache']
   if not existing:enqueue('cache');return
   state=job_state(existing[-1]['job_id'])
   if state in ('PENDING','RUNNING','CONFIGURING','COMPLETING','UNKNOWN'):
    print(json.dumps({'action':'wait','reason':'shared teacher cache '+state}));return
   if (cache/'failure.json').exists():raise RuntimeError('Shared cache failed; inspect before resubmission')
   if (cache/'incomplete.json').exists() and state in ('FAILED','COMPLETED','TIMEOUT'):
    enqueue('cache',resume=True);return
   raise RuntimeError('Shared cache did not publish a clean resumable state')
  ready=read(cache/'READY.json')
  if not ready.get('complete') or ready.get('processed')!=61000:raise ValueError('Shared cache incomplete')
  for design in DESIGNS:
   out=root/design;existing=[j for j in jobs['jobs'] if j['name']==design]
   if not existing:enqueue('train',design);continue
   state=job_state(existing[-1]['job_id'])
   if state in ('PENDING','RUNNING','CONFIGURING','COMPLETING','UNKNOWN'):continue
   if (out/'failure.json').exists():raise RuntimeError(design+' failed; inspect before resubmission')
   summary=read(out/'summary.json') if (out/'summary.json').exists() else {}
   if summary.get('status')=='completed':continue
   if summary.get('status')=='incomplete' and state in ('COMPLETED','TIMEOUT'):
    enqueue('train',design,resume=True);continue
   raise RuntimeError(design+' has no clean resumable summary')
  print(json.dumps({'action':'checked','jobs':jobs['jobs']}))

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--plan',type=Path,required=True);p.add_argument('--repo',type=Path,required=True);p.add_argument('--root',type=Path,required=True)
 a=p.parse_args();advance(a.plan,a.repo,a.root)
