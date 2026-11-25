"""Question-local frozen-teacher cache with crash-safe, bounded shard storage.

The runner must hold ``RunLock(root)`` for this object's entire lifetime and call
``flush()`` before committing training progress. ``original_rows`` is the exact,
fixed training order followed by validation rows. Rows already present in
``old_banks`` are excluded from new blocks, and remain in that read-only cache.

``identity`` must freeze the teacher hashes/seed and source-image identity. The
ordered full manifest and block size are additionally frozen here. The callback
``extract(identifier)`` is only invoked for the requested, missing question, so it
can instantiate the teacher lazily. An interrupted process may recompute only
unflushed features; it never consumes an uncommitted shard.
"""
from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass, field
import hashlib
import json
import os
from pathlib import Path
import re
import uuid

import torch

from pgmvlm.checkpoints import atomic_json, fsync_dir, sha256
from pgmvlm.storage import require_headroom


_SHAPE = (8, 9, 32)
_GENERATION = re.compile(r"block-(\d{5,})-([0-9a-f]{32})\.pt\Z")


@dataclass
class _Block:
    values: dict = field(default_factory=dict)
    dirty: bool = False


class IncrementalRelations:
    """Mapping from question ID to a detached CPU float32 [1, 8, 9, 32] tensor.

    ``capacity`` bounds resident blocks, not questions. Partial blocks are flushed
    on LRU eviction and by explicit ``flush()``. No destructor performs I/O. New
    generations are immutable; a checksummed JSON pointer is their commit point.
    Superseded and orphan generations are removed under the caller's writer lock,
    leaving two files per populated block plus one shared manifest.
    """

    def __init__(self, root, identity: str, original_rows: list, old_banks,
                 old_ids: set, extract, capacity=4, chunk_size=256, *,
                 headroom=require_headroom):
        if not isinstance(identity, str) or not identity:
            raise ValueError('A frozen cache identity is required')
        if type(capacity) is not int or capacity < 1:
            raise ValueError('Cache capacity must be a positive integer')
        if type(chunk_size) is not int or chunk_size < 1:
            raise ValueError('Cache chunk size must be a positive integer')
        if not callable(extract) or not callable(headroom):
            raise TypeError('Extraction and storage guards must be callable')
        ids = [row['id'] for row in original_rows]
        if any(not isinstance(key, str) or not key for key in ids):
            raise ValueError('Question IDs must be nonempty strings')
        if len(ids) != len(set(ids)):
            raise ValueError('Duplicate question IDs in cache manifest')
        self.old_ids = frozenset(old_ids)
        if not self.old_ids.issubset(ids):
            raise ValueError('Old cache IDs are outside the ordered manifest')
        if self.old_ids and old_banks is None:
            raise ValueError('Old cache IDs require an old bank reader')
        manifest_hash = hashlib.sha256(json.dumps(
            original_rows, sort_keys=True, separators=(',', ':'),
            ensure_ascii=True, allow_nan=False).encode()).hexdigest()
        self.root = Path(root)
        self.identity, self.extract = identity, extract
        self.old_banks, self.capacity = old_banks, capacity
        self.chunk_size, self.headroom = chunk_size, headroom
        new_ids = [key for key in ids if key not in self.old_ids]
        self.blocks = [new_ids[start:start + chunk_size]
                       for start in range(0, len(new_ids), chunk_size)]
        self.lookup = {key: index for index, keys in enumerate(self.blocks)
                       for key in keys}
        self.cache = OrderedDict()
        self.extracted = 0
        self.manifest = dict(schema=1, identity=identity,
                             original_rows_sha256=manifest_hash,
                             old_ids_sha256=hashlib.sha256(json.dumps(
                                 sorted(self.old_ids), separators=(',', ':')).encode()).hexdigest(),
                             questions=len(ids), old_questions=len(self.old_ids),
                             new_questions=len(new_ids), chunk_size=chunk_size,
                             shape=list(_SHAPE), dtype='float32')
        self.manifest_digest = hashlib.sha256(json.dumps(
            self.manifest, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
        path = self.root / 'manifest.json'
        if path.exists():
            if path.is_symlink() or json.loads(path.read_text()) != self.manifest:
                raise ValueError('Incremental cache manifest/identity changed')
        else:
            if self.root.exists() and any(self.root.glob('block-*')):
                raise ValueError('Incremental cache has shards but no manifest')
            self._require_headroom()
            self.root.mkdir(parents=True, exist_ok=True)
            atomic_json(path, self.manifest)
        self._remove_orphans()

    def _require_headroom(self):
        # A full block is about 2.3 MiB at the production chunk size. Keep a
        # larger working reserve for checkpoint coexistence and atomic saves.
        self.headroom(self.root, reserve_bytes=2 * 1024**3,
                      reserve_files=32)

    def _pointer_path(self, index):
        return self.root / f'block-{index:05d}.json'

    def _read_pointer(self, index):
        path = self._pointer_path(index)
        if not path.exists():
            return None
        if path.is_symlink():
            raise ValueError('Unsafe incremental cache pointer')
        pointer = json.loads(path.read_text())
        if (not isinstance(pointer, dict) or pointer.get('schema') != 1
                or pointer.get('identity') != self.identity
                or pointer.get('manifest_sha256') != self.manifest_digest
                or pointer.get('block') != index):
            raise ValueError('Incremental cache block identity changed')
        name = pointer.get('file')
        match = _GENERATION.fullmatch(name) if isinstance(name, str) else None
        if not match or int(match.group(1)) != index:
            raise ValueError('Invalid incremental cache generation name')
        digest = pointer.get('sha256')
        if not isinstance(digest, str) or not re.fullmatch(r'[0-9a-f]{64}', digest):
            raise ValueError('Invalid incremental cache checksum')
        keys = pointer.get('ids')
        if (not isinstance(keys, list) or not keys
                or any(not isinstance(key, str) for key in keys)
                or len(keys) != len(set(keys))):
            raise ValueError('Invalid incremental cache question IDs')
        expected = self.blocks[index]
        included = set(keys)
        if keys != [key for key in expected if key in included]:
            raise ValueError('Incremental cache question/block index mismatch')
        if pointer.get('count') != len(keys):
            raise ValueError('Incremental cache count mismatch')
        return pointer

    def _remove_orphans(self):
        """Reclaim only our unpublished generations, after validating pointers."""
        live = set()
        for index in range(len(self.blocks)):
            pointer = self._read_pointer(index)
            if pointer:
                path = self.root / pointer['file']
                if path.is_symlink() or not path.is_file():
                    raise ValueError('Missing or unsafe incremental cache generation')
                live.add(pointer['file'])
        changed = False
        for path in self.root.iterdir():
            match = _GENERATION.fullmatch(path.name)
            # Hidden stage files were never published. Restrict cleanup to the
            # exact format this module creates, preserving unrelated files.
            temporary = (re.fullmatch(r'\.block-\d{5,}-[0-9a-f]{32}\.pt\.tmp', path.name)
                         or re.fullmatch(r'\.(?:block-\d{5,}|manifest)\.json-[0-9a-f]{32}',
                                         path.name))
            if ((match and int(match.group(1)) < len(self.blocks)
                 and path.name not in live) or temporary):
                if path.is_symlink() or not path.is_file():
                    raise ValueError('Unsafe incremental cache orphan')
                path.unlink()
                changed = True
        if changed:
            fsync_dir(self.root)

    def _load(self, index):
        pointer = self._read_pointer(index)
        if pointer is None:
            return _Block()
        path = self.root / pointer['file']
        if path.is_symlink() or not path.is_file():
            raise ValueError('Missing or unsafe incremental cache generation')
        if sha256(path) != pointer['sha256']:
            raise ValueError('Incremental cache checksum mismatch')
        shard = torch.load(path, map_location='cpu', weights_only=True)
        if (not isinstance(shard, dict) or shard.get('schema') != 1
                or shard.get('identity') != self.identity
                or shard.get('manifest_sha256') != self.manifest_digest
                or shard.get('block') != index
                or shard.get('ids') != pointer['ids']):
            raise ValueError('Incremental cache question/identity mismatch')
        bank = shard.get('bank')
        if (not isinstance(bank, torch.Tensor) or bank.dtype != torch.float32
                or bank.layout != torch.strided
                or tuple(bank.shape) != (len(pointer['ids']), *_SHAPE)
                or not torch.isfinite(bank).all()):
            raise ValueError('Invalid incremental cache feature tensor')
        return _Block({key: bank[offset:offset + 1]
                       for offset, key in enumerate(pointer['ids'])})

    @staticmethod
    def _feature(value):
        if (not isinstance(value, torch.Tensor)
                or value.layout != torch.strided
                or not value.is_floating_point()
                or tuple(value.shape) != (1, *_SHAPE)):
            raise ValueError('Expected one teacher feature of shape [1, 8, 9, 32]')
        value = value.detach().to(device='cpu', dtype=torch.float32).contiguous()
        if not torch.isfinite(value).all():
            raise ValueError('Nonfinite teacher features')
        return value.clone()

    def __getitem__(self, key):
        if key in self.old_ids:
            return self._feature(self.old_banks[key])
        index = self.lookup[key]  # Unknown IDs must never reach the teacher.
        if index not in self.cache:
            if len(self.cache) >= self.capacity:
                oldest = next(iter(self.cache))
                self._flush_block(oldest, self.cache[oldest])
                del self.cache[oldest]
            self.cache[index] = self._load(index)
        self.cache.move_to_end(index)
        block = self.cache[index]
        if key not in block.values:
            block.values[key] = self._feature(self.extract(key))
            block.dirty = True
            self.extracted += 1
        # Callers cannot mutate the authoritative in-memory cache via a view.
        return block.values[key].clone()

    def _flush_block(self, index, block):
        if not block.dirty:
            return
        self._require_headroom()
        keys = [key for key in self.blocks[index] if key in block.values]
        name = f'block-{index:05d}-{uuid.uuid4().hex}.pt'
        temp, final = self.root / ('.' + name + '.tmp'), self.root / name
        shard = dict(schema=1, identity=self.identity,
                     manifest_sha256=self.manifest_digest, block=index,
                     ids=keys, bank=torch.cat([block.values[key] for key in keys]))
        try:
            with temp.open('xb') as stream:
                torch.save(shard, stream)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temp, final)
            fsync_dir(self.root)
            pointer = dict(schema=1, identity=self.identity,
                           manifest_sha256=self.manifest_digest, block=index,
                           file=name, sha256=sha256(final), ids=keys, count=len(keys))
            atomic_json(self._pointer_path(index), pointer)
            block.dirty = False
            # Only the pointer's generation is live; no checkpoint refers to a
            # cache generation, since question features are immutable by ID.
            for path in self.root.glob(f'block-{index:05d}-*.pt'):
                if path.name != name and _GENERATION.fullmatch(path.name):
                    if path.is_symlink() or not path.is_file():
                        raise ValueError('Unsafe superseded cache generation')
                    path.unlink()
            fsync_dir(self.root)
        finally:
            temp.unlink(missing_ok=True)

    def flush(self):
        """Durably publish all extracted features before checkpointing progress."""
        for index, block in self.cache.items():
            self._flush_block(index, block)
