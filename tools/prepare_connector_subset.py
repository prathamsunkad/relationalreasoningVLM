"""Create the nested 5% manifest without copying images or touching test data."""
import argparse,hashlib,json,os
from pathlib import Path

def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
 return h.hexdigest()

def prepare(parent,selection,output,parent_sha,selection_sha):
 if sha(parent)!=parent_sha or sha(selection)!=selection_sha:raise ValueError('Source manifest/selection checksum changed')
 if output.exists():raise ValueError('Refuse to replace an existing fixed subset')
 history=json.loads(selection.read_text());validation=history['validation']
 if len(validation)!=1000:raise ValueError('Expected original 1000 validation IDs')
 lines=[];ids=set();previous=None;count=0
 with parent.open() as f:
  for line in f:
   row=json.loads(line);original=row['metadata']['source_original_id']
   key=hashlib.sha256(f'20261007\0train\0{original}'.encode()).digest()
   if previous is not None and key<previous:raise ValueError('Parent no longer follows original seeded rank order')
   if row['split']!='train' or row['dataset']!='pgm' or row['regime']!='neutral':raise ValueError('Unexpected training split')
   previous=key;count+=1
   if count<=60000:
    if row['id'] in ids:raise ValueError('Duplicate training ID')
    ids.add(row['id']);lines.append(line)
 if count!=240000 or len(ids)!=60000:raise ValueError('Wrong parent/subset size')
 val_ids=[r['id'] for r in validation]
 if len(set(val_ids))!=1000 or set(val_ids)&ids or any(r['split']!='validation' for r in validation):raise ValueError('Invalid validation IDs')
 output.mkdir(parents=True)
 (output/'train.jsonl').write_text(''.join(lines))
 (output/'validation.jsonl').write_text(''.join(json.dumps(r,sort_keys=True)+'\n' for r in validation))
 result=dict(train_count=60000,validation_count=1000,official_train_count=1200000,fraction=.05,
   parent_train_sha256=parent_sha,historical_selection_sha256=selection_sha,
   selection='first 60000 of the original smallest SHA256(20261007 NUL train NUL original_id) ranking',
   train_manifest=str(output/'train.jsonl'),train_manifest_sha256=sha(output/'train.jsonl'),
   val_manifest=str(output/'validation.jsonl'),val_manifest_sha256=sha(output/'validation.jsonl'))
 (output/'READY.json').write_text(json.dumps(result,indent=2)+'\n')
 return result

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--parent',type=Path,required=True);p.add_argument('--selection',type=Path,required=True)
 p.add_argument('--output',type=Path,required=True);p.add_argument('--parent-sha',required=True);p.add_argument('--selection-sha',required=True)
 a=p.parse_args();print(json.dumps(prepare(a.parent,a.selection,a.output,a.parent_sha,a.selection_sha),indent=2))
