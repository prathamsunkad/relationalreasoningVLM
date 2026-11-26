"""Verify archived hashes and recompute the paired validation comparison (standard library only)."""
from pathlib import Path
import gzip
import hashlib
import json
import math

ROOT = Path(__file__).resolve().parent


def log_softmax(scores):
    m = max(scores)
    z = m + math.log(sum(math.exp(x - m) for x in scores))
    return [x - z for x in scores]


def log_average(x, y):
    m = max(x, y)
    return m + math.log(math.exp(x - m) + math.exp(y - m)) - math.log(2)


def main():
    provenance = json.loads((ROOT / 'PROVENANCE.json').read_text())
    for item in provenance['files']:
        raw = (ROOT / item['path']).read_bytes()
        assert hashlib.sha256(raw).hexdigest() == item['sha256'], item['path']
        original = gzip.decompress(raw) if item['encoding'] == 'gzip' else raw
        for key in ['original_sha256', 'remote_sha256']:
            if item.get(key):
                assert hashlib.sha256(original).hexdigest() == item[key], (item['path'], key)
    records = [json.loads(line) for line in gzip.decompress(
        (ROOT / 'records/output/predictions.jsonl.gz').read_bytes()).decode().splitlines() if line]
    ref = json.loads(gzip.decompress((ROOT / 'records/hybrid-reference.json.gz').read_bytes()))
    summary = json.loads((ROOT / 'records/output/summary.json').read_text())
    checks = json.loads((ROOT / 'records/output/checks.json').read_text())
    assert checks['passed'] and summary['complete'] and summary['frozen_weights_unchanged']
    assert summary['split'] == 'validation' and summary['benchmark_test'] is False
    assert ref['update'] == 3500 and ref['complete'] and len(records) == 1000
    hybrid = {r['id']: r for r in ref['predictions']}
    assert len(hybrid) == len(ref['predictions']) == len({r['id'] for r in records}) == len(records)
    assert set(hybrid) == {r['id'] for r in records}
    outcomes = {key: [] for key in summary['metrics']}
    losses = {key: [] for key in summary['metrics']}
    for row in records:
        h = hybrid[row['id']]
        assert row['target'] == h['target']
        a, b = log_softmax(row['lora_scores']), log_softmax(row['abstractor_scores'])
        scores = dict(lora=a, abstractor=b,
                      average_log_probability=log_softmax([(x+y)/2 for x,y in zip(a,b)]),
                      average_probability=[log_average(x,y) for x,y in zip(a,b)],
                      hybrid=log_softmax(h['scores']))
        for key, s in scores.items():
            assert len(s) == 8 and all(math.isfinite(x) for x in s)
            outcomes[key].append(max(range(8), key=s.__getitem__) == row['target'])
            losses[key].append(-s[row['target']])
    for key, expected in summary['metrics'].items():
        correct, count = sum(outcomes[key]), len(records)
        loss = sum(losses[key])/count
        assert correct == expected['correct'] and count == expected['count']
        assert abs(correct/count-expected['accuracy']) < 1e-12
        assert abs(loss-expected['loss']) < 1e-9, (key,loss,expected['loss'])
        print(f'{key}: {correct}/{count} = {100*correct/count:.1f}%, CE={loss:.8f}')
    a, b = outcomes['hybrid'], outcomes['average_log_probability']
    wins = sum(x and not y for x,y in zip(a,b))
    losses_count = sum(y and not x for x,y in zip(a,b))
    n = wins + losses_count
    p = min(1.0, 2*sum(math.comb(n,k) for k in range(min(wins,losses_count)+1))/2**n) if n else 1.0
    assert (wins,losses_count) == (32,31) and p == 1.0
    print(f'Hybrid vs primary average: discordant {wins}/{losses_count}, exact McNemar p={p}')
    print(f'PASS: {len(provenance["files"])} exported files, 1,000 matched IDs/labels, metrics and paired comparison verified.')


if __name__ == '__main__':
    main()
