"""Verify the committed historical record bundle with Python's standard library."""
import argparse
import csv
import gzip
import hashlib
import importlib.util
import json
from pathlib import Path

def sha(data):
    return hashlib.sha256(data).hexdigest()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--source-root', type=Path, help='Optional original benchmarks archive')
    args = ap.parse_args()
    root = Path(__file__).resolve().parent
    manifest = json.loads((root / 'MANIFEST.json').read_text())
    spec = importlib.util.spec_from_file_location('export_records', root / 'export_records.py')
    exporter = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(exporter)
    unique, total_lines = {}, 0
    for record in manifest['records']:
        relative = Path(record['artifact'])
        assert not relative.is_absolute() and '..' not in relative.parts, relative
        path = root / relative
        assert not path.is_symlink(), path
        if record['artifact'] not in unique:
            payload = path.read_bytes()
            assert len(payload) == record['artifact_bytes'] and sha(payload) == record['artifact_sha256'], path
            content = gzip.decompress(payload) if path.suffix == '.gz' else payload
            assert len(content) == record['content_bytes'] and sha(content) == record['content_sha256'], path
            if record['jsonl_records'] is not None:
                lines = [line for line in content.splitlines() if line.strip()]
                assert len(lines) == record['jsonl_records'], path
                for line in lines:
                    json.loads(line)
                total_lines += len(lines)
            elif '.json' in path.suffixes:
                json.loads(content)
            assert not exporter.SECRET.search(content), path
            unique[record['artifact']] = (content, record['artifact_sha256'])
        content, artifact_sha = unique[record['artifact']]
        assert artifact_sha == record['artifact_sha256'], record['artifact']
        if record['transform'] == 'verbatim':
            assert sha(content) == record['source_sha256'], record['source']
        else:
            assert record['transform'] == 'omit_per_question_arrays' and record['omitted_arrays'], record['source']
        if args.source_root:
            source = args.source_root / record['source']
            original = source.read_bytes()
            assert len(original) == record['source_bytes'] and sha(original) == record['source_sha256'], source
            if record['transform'] != 'verbatim':
                removed = []
                stripped = exporter.strip_records(json.loads(original), removed)
                assert stripped == json.loads(content) and removed == record['omitted_arrays'], source
    with (root / 'INDEX.csv').open(newline='') as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == len(manifest['records'])
    for row, record in zip(rows, manifest['records']):
        for key, value in row.items():
            assert value == ('' if record[key] is None else str(record[key])), (record['source'], key)
    print(f"Verified {len(manifest['records'])} source records, {len(unique)} unique artifacts, {total_lines} JSONL records.")
    if args.source_root:
        print('Original source hashes and every aggregate-only transformation also verified.')

if __name__ == '__main__':
    main()
