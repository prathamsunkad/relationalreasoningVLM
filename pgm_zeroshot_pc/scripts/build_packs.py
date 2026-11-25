"""PGM neutral test (HuggingFaceM4/PGM mirror, pinned revision) -> 28 shard packs for the pgm-lora-v1 bundle's evaluator.

Each row of the mirror holds 8 context panels and 8 choice panels as PNGs, the target and the original test id. Pack k
holds the problems that the bundle's own sharding gives shard k: file names PGM_neutral_test_<id>.npz sorted as strings,
every 28th one (pgmvlm/data.py: list_npz + shard_files). Arrays: image uint8 (N,16,160,160), target uint8, name (file names).
Checks: 200,000 unique ids; the 10 original DeepMind files in the bundle's sample_problems/ are pixel- and target-identical;
targets agree with the contact-sheet test.jsonl (answer letter, matched by mirror_sample_id).
"""
import argparse, glob, io, json, os, sys
import numpy as np
import pyarrow.parquet as pq
from PIL import Image

ap = argparse.ArgumentParser()
ap.add_argument('--parquet_dir', required=True)
ap.add_argument('--out', required=True)
ap.add_argument('--samples', required=True)
ap.add_argument('--sheet_jsonl', required=True)
ap.add_argument('--num_shards', type=int, default=28)
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)

def png(d):
    im = Image.open(io.BytesIO(d['bytes']))
    arr = np.asarray(im.convert('L') if im.mode != 'L' else im)
    assert arr.shape == (160, 160), arr.shape
    return arr

files = sorted(glob.glob(os.path.join(a.parquet_dir, 'test-*.parquet')))
assert len(files) == 9, files
# pass 1: ids -> (file index, row) so packs can be written in the bundle's shard order without holding 82 GB in memory
loc = {}
for fi, f in enumerate(files):
    ids = pq.read_table(f, columns=['id']).column('id').to_pylist()
    for r, i in enumerate(ids):
        assert i not in loc, f'duplicate id {i}'
        loc[i] = (fi, r)
print('rows', len(loc), 'id range', min(loc), max(loc), flush=True)
assert len(loc) == 200000

names = sorted(f'PGM_neutral_test_{i}.npz' for i in loc)
ans = {}
for line in open(a.sheet_jsonl):
    r = json.loads(line)
    ans[r['metadata']['mirror_sample_id']] = ord(r['answer']) - 65
assert len(ans) == 200000

samples = {os.path.basename(p): np.load(p) for p in glob.glob(os.path.join(a.samples, '*.npz'))}
# keep the PNG bytes in memory (~6 GB), then decode one shard at a time (~3 GB) and write it compressed
raw = {}
for f in files:
    pf = pq.ParquetFile(f)
    for batch in pf.iter_batches(batch_size=512, columns=['id', 'panels', 'choices', 'target']):
        d = batch.to_pydict()
        for i, pn, ch, t in zip(d['id'], d['panels'], d['choices'], d['target']):
            assert len(pn) == 8 and len(ch) == 8
            raw[f'PGM_neutral_test_{i}.npz'] = (i, [x['bytes'] for x in pn] + [x['bytes'] for x in ch], int(t))
    print(f'read {os.path.basename(f)}', flush=True)
mism = sum(ans[i] != t for i, _, t in raw.values())
checked = 0
for name, s in samples.items():
    i, pngs, t = raw[name]
    img = np.stack([png({'bytes': b}) for b in pngs])
    assert np.array_equal(np.asarray(s['image']).reshape(16, 160, 160), img), f'pixel mismatch {name}'
    assert int(s['target']) == t, f'target mismatch {name}'
    checked += 1
print(f'sample_problems identical: {checked}/{len(samples)}; targets differing from test.jsonl: {mism}', flush=True)
assert checked == len(samples) == 10 and mism == 0
for k in range(a.num_shards):
    fs = names[k::a.num_shards]
    img = np.empty((len(fs), 16, 160, 160), np.uint8)
    for j, n in enumerate(fs):
        img[j] = np.stack([png({'bytes': b}) for b in raw[n][1]])
    np.savez_compressed(os.path.join(a.out, f'pack{k:02d}.npz'), image=img,
                        target=np.array([raw[n][2] for n in fs], np.uint8), name=np.array(fs))
    print(f'pack{k:02d}: {len(fs)} problems', flush=True)
json.dump({'source': 'HuggingFaceM4/PGM test', 'revision': '195a03a7e3bc6d6a349b55986afba5319929c3bf', 'n': 200000,
           'num_shards': a.num_shards, 'sample_problems_identical': checked, 'target_mismatches_vs_sheet_jsonl': mism},
          open(os.path.join(a.out, 'packs.json'), 'w'), indent=1)
print('done')
