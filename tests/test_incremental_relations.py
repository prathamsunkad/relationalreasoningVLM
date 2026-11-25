"""CPU-only integrity and crash-recovery tests for the lazy teacher cache."""
import json
from pathlib import Path

import pytest
import torch

import pgmvlm.incremental_relations as module
from pgmvlm.checkpoints import sha256
from pgmvlm.incremental_relations import IncrementalRelations


def feature(key):
    return torch.full((1, 8, 9, 32), float(int(key.removeprefix('q'))))


@pytest.fixture
def make_cache(tmp_path):
    calls = []
    guards = []

    def extract(key):
        calls.append(key)
        return feature(key)

    def guard(path, **kwargs):
        guards.append((Path(path), kwargs))

    def make(**overrides):
        options = dict(root=tmp_path / 'cache', identity='teacher-seed-images-hash',
                       original_rows=[dict(id=f'q{i}', image=f'{i}.png') for i in range(12)],
                       old_banks=None, old_ids=set(), extract=extract,
                       capacity=1, chunk_size=4, headroom=guard)
        options.update(overrides)
        return IncrementalRelations(**options)

    make.calls, make.guards = calls, guards
    return make


def test_only_requested_question_is_extracted_and_results_are_isolated(make_cache):
    cache = make_cache(extract=lambda key: (
        make_cache.calls.append(key) or feature(key).to(torch.float64).requires_grad_()))
    assert make_cache.calls == []
    result = cache['q2']
    assert make_cache.calls == ['q2']
    assert result.shape == (1, 8, 9, 32)
    assert result.dtype == torch.float32 and result.device.type == 'cpu'
    assert not result.requires_grad
    result.fill_(999)
    torch.testing.assert_close(cache['q2'], feature('q2'), rtol=0, atol=0)
    assert make_cache.calls == ['q2']
    with pytest.raises(KeyError):
        cache['unknown']
    assert make_cache.calls == ['q2']


def test_old_banks_are_reused_before_new_cache_and_excluded_from_blocks(make_cache):
    class OldBanks:
        def __init__(self):
            self.calls = []

        def __getitem__(self, key):
            self.calls.append(key)
            return feature(key)

    old = OldBanks()
    cache = make_cache(old_banks=old, old_ids={'q0', 'q2', 'q7', 'q11'})
    assert cache.blocks[0] == ['q1', 'q3', 'q4', 'q5']
    torch.testing.assert_close(cache['q7'], feature('q7'))
    assert old.calls == ['q7'] and make_cache.calls == []
    cache.flush()
    assert list(cache.root.glob('block-*')) == []


def test_partial_eviction_and_resume_preserve_question_alignment(make_cache):
    cache = make_cache()
    # Nonmonotonic access also exercises canonical ordering within each block.
    for key in ('q3', 'q1', 'q9', 'q4'):
        torch.testing.assert_close(cache[key], feature(key), rtol=0, atol=0)
    assert len(cache.cache) == 1
    cache.flush()
    assert json.loads((cache.root / 'block-00000.json').read_text())['ids'] == ['q1', 'q3']
    resumed = make_cache()
    for key in ('q1', 'q9', 'q3', 'q4'):
        torch.testing.assert_close(resumed[key], feature(key), rtol=0, atol=0)
    assert make_cache.calls == ['q3', 'q1', 'q9', 'q4']
    resumed['q0']
    assert make_cache.calls[-1] == 'q0'
    resumed.flush()
    assert json.loads((cache.root / 'block-00000.json').read_text())['ids'] == ['q0', 'q1', 'q3']


def test_unflushed_loss_recomputes_only_requested_missing_question(make_cache):
    cache = make_cache()
    cache['q0']
    cache.flush()
    cache['q1']  # Simulate a process loss without its explicit flush.
    resumed = make_cache()
    resumed['q0']
    assert make_cache.calls == ['q0', 'q1']
    resumed['q1']
    assert make_cache.calls == ['q0', 'q1', 'q1']
    assert 'q2' not in make_cache.calls


def test_interrupted_pointer_commit_preserves_prior_generation(make_cache, monkeypatch):
    cache = make_cache()
    cache['q0']
    cache.flush()
    pointer_path = cache.root / 'block-00000.json'
    original_pointer = json.loads(pointer_path.read_text())
    cache['q1']
    real_atomic_json = module.atomic_json

    def interrupted(path, value):
        if path.name == 'block-00000.json':
            raise OSError('simulated process loss before pointer commit')
        return real_atomic_json(path, value)

    monkeypatch.setattr(module, 'atomic_json', interrupted)
    with pytest.raises(OSError, match='process loss'):
        cache.flush()
    assert json.loads(pointer_path.read_text()) == original_pointer
    assert len(list(cache.root.glob('*.pt'))) == 2
    monkeypatch.setattr(module, 'atomic_json', real_atomic_json)
    resumed = make_cache()
    assert len(list(cache.root.glob('*.pt'))) == 1
    resumed['q0']
    assert make_cache.calls == ['q0', 'q1']
    resumed['q1']
    resumed.flush()
    assert make_cache.calls == ['q0', 'q1', 'q1']
    assert len(list(cache.root.glob('*.pt'))) == 1


def test_interruption_after_pointer_commit_recovers_new_generation(make_cache, monkeypatch):
    cache = make_cache()
    cache['q0']
    cache.flush()
    previous = next(cache.root.glob('*.pt'))
    cache['q1']
    real_unlink = Path.unlink

    def interrupted(path, *args, **kwargs):
        if path == previous:
            raise OSError('simulated process loss during old-generation cleanup')
        return real_unlink(path, *args, **kwargs)

    monkeypatch.setattr(Path, 'unlink', interrupted)
    with pytest.raises(OSError, match='process loss'):
        cache.flush()
    assert len(list(cache.root.glob('*.pt'))) == 2
    monkeypatch.setattr(Path, 'unlink', real_unlink)
    resumed = make_cache()
    resumed['q0']
    resumed['q1']
    assert make_cache.calls == ['q0', 'q1']
    assert len(list(cache.root.glob('*.pt'))) == 1


@pytest.mark.parametrize('corruption', ['checksum', 'wrong_id'])
def test_existing_bank_integrity_errors_propagate_without_extraction(make_cache, tmp_path, corruption):
    # Use the unchanged production reader, not a stub, to verify that adding the
    # incremental layer cannot silently replace corrupt historical features.
    from tools.train_pretrained_connectors import Banks, cache_contract

    plan = dict(teacher_hashes={'weights': 'frozen'}, teacher_seed=123,
                train_manifest_sha256='old-train', val_manifest_sha256='old-val')
    identity = cache_contract(plan)
    old_root = tmp_path / 'old'
    old_root.mkdir()
    path = old_root / 'bank-0000.pt'
    torch.save(dict(contract=identity, ids=['q9' if corruption == 'wrong_id' else 'q0'],
                    bank=feature('q0')), path)
    checksum = sha256(path)
    if corruption == 'checksum':
        with path.open('ab') as stream:
            stream.write(b'corrupt old bank')
    (old_root / 'READY.json').write_text(json.dumps(dict(
        complete=True, contract=identity, processed=1,
        shards=[dict(file=path.name, sha256=checksum, count=1)])))
    reader = Banks(old_root, [dict(id='q0')], plan)
    cache = make_cache(old_banks=reader, old_ids={'q0'})
    with pytest.raises(ValueError, match='(Checksum mismatch|Question/cache index mismatch)'):
        cache['q0']
    assert make_cache.calls == []
    cache.flush()
    assert list(cache.root.glob('block-*')) == []


def test_complete_cache_is_reused_across_epochs_without_extra_files(make_cache):
    cache = make_cache()
    for key in cache.lookup:
        cache[key]
    cache.flush()
    assert len(make_cache.calls) == 12
    committed = {path.name: sha256(path) for path in cache.root.iterdir()}
    assert len(committed) == 1 + 2 * 3
    for _ in range(3):
        cache = make_cache()
        for key in cache.lookup:
            torch.testing.assert_close(cache[key], feature(key), rtol=0, atol=0)
        cache.flush()
    assert len(make_cache.calls) == 12
    assert {path.name: sha256(path) for path in cache.root.iterdir()} == committed


@pytest.mark.parametrize('change', ['identity', 'order', 'image', 'old_ids', 'chunk_size'])
def test_cache_contract_changes_are_rejected(make_cache, change):
    make_cache()
    override = {}
    if change == 'identity':
        override['identity'] = 'changed-teacher-or-seed'
    elif change in ('order', 'image'):
        rows = [dict(id=f'q{i}', image=f'{i}.png') for i in range(12)]
        if change == 'order':
            rows.reverse()
        else:
            rows[0]['image'] = 'changed.png'
        override['original_rows'] = rows
    elif change == 'old_ids':
        override.update(old_ids={'q0'}, old_banks={'q0': feature('q0')})
    else:
        override['chunk_size'] = 3
    with pytest.raises(ValueError, match='manifest/identity changed'):
        make_cache(**override)
    assert make_cache.calls == []


def test_checksum_corruption_is_rejected_without_teacher_fallback(make_cache):
    cache = make_cache()
    cache['q1']
    cache.flush()
    path = next(cache.root.glob('*.pt'))
    with path.open('ab') as stream:
        stream.write(b'corrupted')
    resumed = make_cache()
    with pytest.raises(ValueError, match='checksum mismatch'):
        resumed['q1']
    assert make_cache.calls == ['q1']


@pytest.mark.parametrize('corruption', ['wrong_id', 'wrong_block', 'nonfinite', 'wrong_shape', 'wrong_dtype'])
def test_checksummed_invalid_payload_is_rejected(make_cache, corruption):
    cache = make_cache()
    cache['q1']
    cache.flush()
    pointer_path = cache.root / 'block-00000.json'
    pointer = json.loads(pointer_path.read_text())
    path = cache.root / pointer['file']
    payload = torch.load(path, weights_only=True)
    if corruption == 'wrong_id':
        payload['ids'] = ['q2']
    elif corruption == 'wrong_block':
        payload['block'] = 1
    elif corruption == 'nonfinite':
        payload['bank'][0, 0, 0, 0] = float('nan')
    elif corruption == 'wrong_shape':
        payload['bank'] = payload['bank'][:, :, :8]
    else:
        payload['bank'] = payload['bank'].to(torch.float64)
    torch.save(payload, path)
    pointer['sha256'] = sha256(path)
    pointer_path.write_text(json.dumps(pointer))
    resumed = make_cache()
    with pytest.raises(ValueError, match='(question/identity mismatch|feature tensor)'):
        resumed['q1']
    assert make_cache.calls == ['q1']


def test_pointer_cannot_assign_question_to_another_logical_block(make_cache):
    cache = make_cache()
    cache['q1']
    cache.flush()
    pointer_path = cache.root / 'block-00000.json'
    pointer = json.loads(pointer_path.read_text())
    pointer['ids'] = ['q9']
    pointer_path.write_text(json.dumps(pointer))
    with pytest.raises(ValueError, match='question/block index mismatch'):
        make_cache()
    assert make_cache.calls == ['q1']


@pytest.mark.parametrize('invalid', [torch.ones(8, 9, 32), torch.ones(1, 8, 9, 32, dtype=torch.int64),
                                   torch.full((1, 8, 9, 32), float('inf'))])
def test_invalid_extraction_is_not_cached(make_cache, invalid):
    cache = make_cache(extract=lambda key: invalid)
    with pytest.raises(ValueError, match='(teacher feature|Nonfinite)'):
        cache['q0']
    cache.flush()
    assert list(cache.root.glob('block-*')) == []
    assert cache.extracted == 0


def test_storage_guard_blocks_publication_and_allows_safe_retry(make_cache):
    cache = make_cache()
    cache['q1']

    def full(path, **kwargs):
        raise RuntimeError('quota unavailable')

    cache.headroom = full
    with pytest.raises(RuntimeError, match='quota unavailable'):
        cache.flush()
    assert list(cache.root.glob('block-*')) == []
    cache.headroom = lambda path, **kwargs: None
    cache.flush()
    resumed = make_cache()
    resumed['q1']
    assert make_cache.calls == ['q1']
