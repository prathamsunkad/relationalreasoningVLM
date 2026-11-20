"""Export selected historical scalar records; never copy weights or datasets."""
import argparse
import csv
import gzip
import hashlib
import json
import re
from pathlib import Path

CAMPAIGNS = [
    'lfm-lora-idrules-2026-10-07', 'adaptation-repair-2026-10-07',
    'pgm-coverage-16gpuh-2026-10-08', 'pgm-coverage-eval-2026-10-09',
    'lfm-pgm-20pct-2026-10-10', 'pgm-best-test-2026-10-27',
    'pgm-lora-six-epochs-2026-10-28', 'pgm-six-epoch-test-2026-11-01',
    'lfm-pgm-seven-epochs-2026-11-02', 'lfm-pgm-epoch673-test-2026-11-05',
    'lfm-relational-probes-2026-10-11', 'lfm-relational-frozen-2026-10-14',
    'lfm-abstractor-fresh-epoch1-2026-11-10',
    'lfm-abstractor-overfit-2026-11-16', 'lfm-abstractor-plugin-first-2026-11-16',
    'lfm-abstractor-plugin-4k-2026-11-17',
]
ROOTS = ['.local/campaigns/' + n for n in CAMPAIGNS] + [
    '.local/campaigns/adaptation-controls-2026-10-07/lfm25/rmts',
    '.local/progress-checks/2026-10-19-lora-continued',
    '.local/progress-checks/2026-10-22-target-revision',
    '.local/progress-checks/2026-11-19-abstractor-diagnostics',
    'reports/lora-reproduction-2026-10-14',
    'reports/abstractor-capacity-gradient-audit-2026-11-18',
    'reports/abstractor-diagnostics-2026-11-19',
]
NAMES = {
    'summary.json', 'config.json', 'args.json', 'plan.json', 'progress.json',
    'best.json', 'latest.json', 'COMPLETE.json', 'pilot-result.json',
    'backend.json', 'adapter-provenance.json', 'comparison.json',
    'matched-comparison.json', 'convergence-comparison.json',
    'analysis.json', 'gradient-analysis.json', 'common-config.json',
    'models.json', 'prompts.json', 'selected-validation.json',
}
CURVES = {'learning-curve.jsonl', 'metrics.jsonl', 'training.jsonl', 'gradients.jsonl'}
OMIT = {'records', 'predictions', 'examples', 'example_ids', 'rows', 'matched_ids',
        'train_ids', 'val_ids', 'validation_ids', 'test_ids'}
SECRET = re.compile(rb'(?:gh[pousr]_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{30,}|hf_[A-Za-z0-9]{25,}|-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----)')

def digest(b):
    return hashlib.sha256(b).hexdigest()

def strip_records(value, removed, path=''):
    if isinstance(value, dict):
        out = {}
        for k, v in value.items():
            key = path + '/' + k
            if k in OMIT and isinstance(v, (list, dict)):
                removed.append({'key': key, 'count': len(v)})
            else:
                out[k] = strip_records(v, removed, key)
        return out
    if isinstance(value, list):
        return [strip_records(x, removed, path + '/' + str(i)) for i, x in enumerate(value)]
    return value

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--archive-root', type=Path, required=True, help='Original benchmarks directory')
    ap.add_argument('--output', type=Path, required=True)
    a = ap.parse_args()
    a.output.mkdir(parents=True, exist_ok=False)
    records, unique, missing = [], {}, []
    selected = set()
    for rel in ROOTS:
        root = a.archive_root / rel
        if not root.is_dir():
            missing.append(rel)
            continue
        for p in root.rglob('*'):
            if not p.is_file() or p.is_symlink():
                continue
            if any(x in p.relative_to(root).parts for x in ('source', 'frozen-source', 'source-frozen', '.git', 'checkpoints', 'pilot')):
                continue
            name = p.name
            chosen = name in NAMES or name in CURVES or name.startswith('validation-') and p.suffix == '.json' or name.endswith(('-validation.json', '-summary.json'))
            chosen |= rel.startswith('reports/') and p.suffix in ('.csv', '.png')
            if chosen:
                selected.add(p)
    for p in sorted(selected):
        src = p.read_bytes()
        if SECRET.search(src):
            raise ValueError('Potential credential in ' + str(p))
        removed, lines = [], None
        if p.suffix == '.json':
            obj = strip_records(json.loads(src), removed)
            raw = (json.dumps(obj, indent=2, ensure_ascii=False) + '\n').encode() if removed else src
        elif p.suffix == '.jsonl':
            for i, line in enumerate(src.splitlines(), 1):
                if line.strip():
                    json.loads(line)
            lines = sum(bool(line.strip()) for line in src.splitlines())
            raw = src
        else:
            raw = src
        rel = p.relative_to(a.archive_root).as_posix()
        target = rel.removeprefix('.local/')
        compressed = p.suffix == '.jsonl' or len(raw) > 200_000 and p.suffix == '.json'
        payload = gzip.compress(raw, compresslevel=9, mtime=0) if compressed else raw
        if compressed:
            target += '.gz'
        key = (digest(raw), p.suffix)
        if key in unique:
            target = unique[key]
        else:
            unique[key] = target
            out = a.output / target
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_bytes(payload)
        records.append({
            'source': rel, 'source_sha256': digest(src), 'source_bytes': len(src),
            'artifact': target, 'artifact_sha256': digest(payload), 'artifact_bytes': len(payload),
            'content_sha256': digest(raw), 'content_bytes': len(raw),
            'transform': 'omit_per_question_arrays' if removed else 'verbatim',
            'omitted_arrays': removed, 'jsonl_records': lines,
        })
    manifest = {'schema_version': 1, 'created_date': '2026-11-19',
        'source_code_freeze': 'd9822b520448d816b9a883ae7a1378eb8918c55e',
        'source_root_description': 'Original benchmarks archive; all source paths are relative to it.',
        'selection_roots': ROOTS, 'missing_roots': missing, 'records': records}
    (a.output / 'MANIFEST.json').write_text(json.dumps(manifest, indent=2) + '\n')
    fields = ['source', 'artifact', 'transform', 'source_bytes', 'artifact_bytes', 'jsonl_records', 'source_sha256', 'artifact_sha256']
    with (a.output / 'INDEX.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fields, extrasaction='ignore', lineterminator='\n')
        writer.writeheader()
        writer.writerows(records)
    artifacts = {x['artifact']: x['artifact_bytes'] for x in records}
    print(json.dumps({'source_records': len(records), 'unique_artifacts': len(artifacts),
                      'artifact_bytes': sum(artifacts.values()), 'missing_roots': missing}))

if __name__ == '__main__':
    main()
