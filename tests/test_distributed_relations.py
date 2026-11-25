"""Real multiprocessing cache publication, restart and corruption regressions."""
import multiprocessing as mp
import os
from pathlib import Path
import queue
import tempfile
import unittest

import torch

from pgmvlm.distributed_relations import ConcurrentRelations
from pgmvlm.incremental_relations import IncrementalRelations


ROWS = [{'id': f'q{i}', 'image': f'images/{i}.png'} for i in range(12)]
IDENTITY = 'same-teacher-manifest-seed'


def feature(key):
    return torch.full((1, 8, 9, 32), float(key[1:]))


def headroom(*args, **kwargs):
    return {}


def cache(root, extract=feature, cls=ConcurrentRelations, **kwargs):
    return cls(root, IDENTITY, ROWS, None, set(), extract,
               chunk_size=8, headroom=headroom, **kwargs)


def parallel_writer(root, rank, barrier, errors):
    try:
        torch.set_num_threads(1)
        def extract(key):
            # All ranks must enter teacher extraction concurrently. Holding
            # the cache lock during extraction would deadlock this barrier.
            barrier.wait(timeout=15)
            return feature(key)
        banks = cache(root, extract)
        banks[f'q{rank}']
        banks.flush()
        errors.put(None)
    except BaseException as exc:
        errors.put(repr(exc))
        raise


def interrupted_writer(root):
    import pgmvlm.incremental_relations as implementation
    original = implementation.atomic_json
    def crash_before_pointer(path, value):
        if Path(path).name.startswith('block-'):
            os._exit(37)
        return original(path, value)
    implementation.atomic_json = crash_before_pointer
    banks = cache(root)
    banks['q1']
    banks.flush()


def paused_writer(root, ready, release, errors):
    try:
        import pgmvlm.incremental_relations as implementation
        original = implementation.atomic_json
        def pause_before_pointer(path, value):
            if Path(path).name.startswith('block-'):
                ready.set()
                if not release.wait(15):
                    raise TimeoutError('Reader/publisher synchronization timed out')
            return original(path, value)
        implementation.atomic_json = pause_before_pointer
        banks = cache(root)
        banks['q1']
        banks.flush()
        errors.put(None)
    except BaseException as exc:
        errors.put(repr(exc))
        raise


def concurrent_constructor(root, attempted, finished, errors):
    try:
        attempted.set()
        banks = cache(root)
        torch.testing.assert_close(banks['q1'], feature('q1'))
        finished.set()
        errors.put(None)
    except BaseException as exc:
        errors.put(repr(exc))
        raise


class ConcurrentRelationsTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name) / 'cache'
        self.ctx = mp.get_context('spawn')

    def join(self, process, expected=0):
        process.join(30)
        if process.is_alive():
            process.terminate()
            process.join(5)
            self.fail('Cache process deadlocked')
        self.assertEqual(process.exitcode, expected)

    def test_parallel_extraction_and_partial_flush_do_not_lose_rows(self):
        barrier = self.ctx.Barrier(4)
        errors = self.ctx.Queue()
        processes = [self.ctx.Process(target=parallel_writer,
                     args=(str(self.root), rank, barrier, errors)) for rank in range(4)]
        for process in processes:
            process.start()
        for process in processes:
            self.join(process)
        self.assertEqual([errors.get(timeout=3) for _ in processes], [None] * 4)
        banks = cache(self.root, extract=lambda _: self.fail('Lost a committed question'))
        self.assertEqual(banks.committed_count(), 4)
        for rank in range(4):
            torch.testing.assert_close(banks[f'q{rank}'], feature(f'q{rank}'))
        self.assertEqual(banks.extracted, 0)
        # Latest pointer plus a single tensor generation, no per-rank shards.
        self.assertEqual(len(list(self.root.glob('block-*.pt'))), 1)

    def test_reuses_existing_single_writer_format_and_remains_compatible(self):
        old = cache(self.root, cls=IncrementalRelations)
        old['q0']; old.flush()
        newer = cache(self.root)
        torch.testing.assert_close(newer['q0'], feature('q0'))
        self.assertEqual(newer.extracted, 0)
        newer['q1']; newer.flush()
        reopened = cache(self.root, cls=IncrementalRelations,
                         extract=lambda _: self.fail('Format migration lost data'))
        torch.testing.assert_close(reopened['q0'], feature('q0'))
        torch.testing.assert_close(reopened['q1'], feature('q1'))

    def test_stale_reader_refreshes_without_reextracting(self):
        first = cache(self.root)
        first['q0']; first.flush()
        stale = cache(self.root, extract=lambda _: self.fail('Published feature re-extracted'))
        stale['q0']
        first['q1']; first.flush()
        torch.testing.assert_close(stale['q1'], feature('q1'))
        self.assertEqual(stale.extracted, 0)

    def test_conflicting_duplicate_is_rejected_without_overwriting(self):
        one = cache(self.root)
        two = cache(self.root, extract=lambda key: feature(key) + 1)
        one['q0']; two['q0']
        one.flush()
        with self.assertRaisesRegex(ValueError, 'Conflicting immutable teacher'):
            two.flush()
        correct = cache(self.root)
        torch.testing.assert_close(correct['q0'], feature('q0'))

    def test_killed_publication_preserves_last_commit_and_releases_lock(self):
        first = cache(self.root)
        first['q0']; first.flush()
        writer = self.ctx.Process(target=interrupted_writer, args=(str(self.root),))
        writer.start(); self.join(writer, expected=37)
        self.assertEqual(len(list(self.root.glob('block-*.pt'))), 2)
        reopened = cache(self.root)
        self.assertEqual(reopened.committed_count(), 1)
        self.assertEqual(len(list(self.root.glob('block-*.pt'))), 1)
        torch.testing.assert_close(reopened['q0'], feature('q0'))
        reopened['q1']; reopened.flush()
        self.assertEqual(reopened.committed_count(), 2)

    def test_constructor_cannot_delete_an_active_unpublished_generation(self):
        first = cache(self.root)
        first['q0']; first.flush()
        ready, release = self.ctx.Event(), self.ctx.Event()
        attempted, finished = self.ctx.Event(), self.ctx.Event()
        errors = self.ctx.Queue()
        writer = self.ctx.Process(target=paused_writer,
                                  args=(str(self.root), ready, release, errors))
        writer.start()
        self.assertTrue(ready.wait(20))
        reader = self.ctx.Process(target=concurrent_constructor,
                                  args=(str(self.root), attempted, finished, errors))
        reader.start()
        self.assertTrue(attempted.wait(20))
        self.assertFalse(finished.wait(.3))
        release.set()
        self.join(writer); self.join(reader)
        self.assertEqual([errors.get(timeout=3) for _ in range(2)], [None, None])
        self.assertEqual(cache(self.root).committed_count(), 2)

    def test_checksum_corruption_and_unknown_ids_fail_closed(self):
        first = cache(self.root)
        first['q0']; first.flush()
        generation = next(self.root.glob('block-*.pt'))
        generation.write_bytes(b'corrupted')
        reopened = cache(self.root, extract=lambda _: self.fail('Corruption triggered extraction'))
        with self.assertRaisesRegex(ValueError, 'checksum mismatch'):
            reopened['q0']
        with self.assertRaises(KeyError):
            reopened['not-in-manifest']

    def test_lru_eviction_merges_and_returned_values_do_not_mutate_cache(self):
        first = cache(self.root, capacity=1)
        second = cache(self.root, capacity=1)
        first['q0'].fill_(999)
        second['q1']; second.flush()
        first['q8']; first.flush()
        reopened = cache(self.root)
        self.assertEqual(reopened.committed_count(), 3)
        for key in ('q0', 'q1', 'q8'):
            torch.testing.assert_close(reopened[key], feature(key))


if __name__ == '__main__':
    unittest.main()
