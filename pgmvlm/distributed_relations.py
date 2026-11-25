"""Concurrent readers/writers for the existing frozen question-feature cache.

Only short disk transactions are serialized. Missing teacher features are
computed outside the lock, so ranks can extract different questions in parallel.
The format, identity and immutable generations are exactly IncrementalRelations'.
The training job still needs its own lifetime RunLock, held by rank zero, to keep
unrelated training jobs from racing; this adapter's transaction lock is separate.
"""
from __future__ import annotations

from contextlib import contextmanager
import fcntl
from pathlib import Path

import torch

from pgmvlm.incremental_relations import IncrementalRelations


class ConcurrentRelations(IncrementalRelations):
    """IncrementalRelations with locked reload/merge/publish transactions.

    The lock is process-safe and reentrant within one instance; callers must not
    share an instance between threads. A crashed writer releases its OS lock.
    Restart cleans only unpublished generations under that same lock. Different
    process counts and rank assignments reuse the same cache without conversion.
    """

    def __init__(self, root, *args, **kwargs):
        root = Path(root)
        if root.is_symlink():
            raise ValueError('Unsafe concurrent cache root')
        root.mkdir(parents=True, exist_ok=True)
        self._transaction_path = root / '.distributed-cache.lock'
        self._transaction_depth = 0
        with self._transaction():
            super().__init__(root, *args, **kwargs)

    @contextmanager
    def _transaction(self):
        if self._transaction_depth:
            self._transaction_depth += 1
            try:
                yield
            finally:
                self._transaction_depth -= 1
            return
        if self._transaction_path.is_symlink():
            raise ValueError('Unsafe concurrent cache lock')
        with self._transaction_path.open('a+b') as stream:
            fcntl.flock(stream, fcntl.LOCK_EX)
            self._transaction_depth = 1
            try:
                yield
            finally:
                self._transaction_depth = 0
                fcntl.flock(stream, fcntl.LOCK_UN)

    def _remove_orphans(self):
        with self._transaction():
            super()._remove_orphans()

    def _load(self, index):
        with self._transaction():
            block = super()._load(index)
            pointer = self._read_pointer(index)
            block._generation = pointer['file'] if pointer else None
            return block

    @staticmethod
    def _merge(local, committed):
        for key in local.keys() & committed.keys():
            if not torch.equal(local[key], committed[key]):
                raise ValueError('Conflicting immutable teacher features for ID: ' + key)
        return {**committed, **local}

    def __getitem__(self, key):
        if key in self.old_ids:
            return self._feature(self.old_banks[key])
        index = self.lookup[key]
        if index not in self.cache:
            if len(self.cache) >= self.capacity:
                oldest = next(iter(self.cache))
                self._flush_block(oldest, self.cache[oldest])
                del self.cache[oldest]
            self.cache[index] = self._load(index)
        self.cache.move_to_end(index)
        block = self.cache[index]
        if key not in block.values:
            # A different rank may have published this question since our
            # block was loaded. Reload only when the pointer changed.
            with self._transaction():
                pointer = self._read_pointer(index)
                generation = pointer['file'] if pointer else None
                if generation != getattr(block, '_generation', None):
                    committed = super()._load(index)
                    block.values = self._merge(block.values, committed.values)
                    block._generation = generation
            if key not in block.values:
                # No filesystem lock is held over the expensive GPU forward.
                block.values[key] = self._feature(self.extract(key))
                block.dirty = True
                self.extracted += 1
        return block.values[key].clone()

    def _flush_block(self, index, block):
        if not block.dirty:
            return
        with self._transaction():
            committed = super()._load(index)
            block.values = self._merge(block.values, committed.values)
            # Reload+merge+atomic pointer publication+old-generation cleanup
            # share one transaction, preventing lost writes or read/unlink races.
            super()._flush_block(index, block)
            block._generation = self._read_pointer(index)['file']

    def committed_count(self):
        """Count durably published new questions; excludes old immutable banks."""
        with self._transaction():
            count = 0
            for index in range(len(self.blocks)):
                pointer = self._read_pointer(index)
                count += len(pointer['ids']) if pointer else 0
            return count
