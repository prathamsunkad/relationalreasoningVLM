"""Transactional adapter checkpoints: immutable generations and atomic pointers.

Only generations created here are pruned. Historical directories are preserved.
The pointer, not the convenience symlinks, is authoritative for resume.
"""
from contextlib import AbstractContextManager
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import uuid
import warnings

import torch


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def fsync_dir(path):
    fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def atomic_json(path, value):
    path = Path(path)
    tmp = path.with_name('.' + path.name + '-' + uuid.uuid4().hex)
    try:
        with tmp.open('x') as f:
            json.dump(value, f, indent=2, allow_nan=False)
            f.write('\n')
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
        fsync_dir(path.parent)
    finally:
        tmp.unlink(missing_ok=True)


class RunLock(AbstractContextManager):
    """Prevent two allocations from writing the same run; kernel releases on exit."""
    def __init__(self, root, enabled=True):
        self.root, self.enabled, self.file = Path(root), enabled, None

    def __enter__(self):
        if self.enabled:
            self.root.mkdir(parents=True, exist_ok=True)
            self.file = (self.root / '.run.lock').open('a+')
            try:
                fcntl.flock(self.file, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                self.file.close()
                raise RuntimeError('Another process is writing this run directory') from None
        return self

    def __exit__(self, *exc):
        if self.file:
            self.file.close()


def checked_generation(root, pointer):
    store = Path(root) / '.checkpoints'
    name = pointer.get('directory', '')
    if Path(name).name != name or not name or not name.startswith(('last-', 'best-', 'final-')):
        raise ValueError('Invalid checkpoint generation name')
    folder = store / name
    if folder.is_symlink() or not folder.is_dir():
        raise ValueError('Missing checkpoint generation')
    manifest_path = folder / 'checkpoint.json'
    if manifest_path.is_symlink() or sha256(manifest_path) != pointer.get('manifest_sha256'):
        raise ValueError('Checkpoint manifest checksum mismatch')
    manifest = json.loads(manifest_path.read_text())
    if manifest.get('schema') != 1:
        raise ValueError('Unsupported checkpoint schema')
    for rel, digest in manifest['files'].items():
        path = folder / rel
        if Path(rel).is_absolute() or '..' in Path(rel).parts or path.is_symlink() or not path.is_file() or not path.resolve().is_relative_to(folder.resolve()):
            raise ValueError('Unsafe or missing checkpoint member')
        if sha256(path) != digest:
            raise ValueError(f'Checkpoint checksum mismatch: {rel}')
    actual = {str(p.relative_to(folder)) for p in folder.rglob('*') if p.is_file() and p != manifest_path}
    if actual != set(manifest['files']):
        raise ValueError('Checkpoint file inventory changed')
    return folder


def _link(root, name, target):
    path = root / name
    if path.exists() and not path.is_symlink():
        # Preserve earlier shared-trainer checkpoints; never delete user files.
        path.rename(root / '.checkpoints' / ('legacy-' + name + '-' + uuid.uuid4().hex))
    temp = root / ('.' + name + '-' + uuid.uuid4().hex)
    temp.symlink_to(os.path.relpath(target, root))
    os.replace(temp, path)
    fsync_dir(root)


def publish(root, tag, pointer):
    root = Path(root)
    folder = checked_generation(root, pointer)
    atomic_json(root / '.checkpoints' / (tag + '.json'), pointer)
    _link(root, tag, folder)
    if tag == 'last':
        _link(root, 'state.pt', folder / 'state.pt')
    if tag == 'best':
        atomic_json(root / 'best.json', json.loads((folder / 'selection.json').read_text()))


def pointer_for(root, tag):
    path = Path(root) / '.checkpoints' / (tag + '.json')
    return json.loads(path.read_text()) if path.exists() else None


def commit(root, tag, write_model, state=None, selection=None):
    """Make model + optimizer + RNG visible together; retain prior complete saves."""
    if tag not in ('last', 'best', 'final'):
        raise ValueError('Unsupported checkpoint tag')
    root = Path(root)
    store = root / '.checkpoints'
    store.mkdir(parents=True, exist_ok=True)
    name = tag + '-' + uuid.uuid4().hex
    stage, final = store / ('.' + name), store / name
    stage.mkdir()
    try:
        write_model(stage)
        if state is not None:
            torch.save(state, stage / 'state.pt')
        if selection is not None:
            atomic_json(stage / 'selection.json', selection)
        files = {str(p.relative_to(stage)): sha256(p) for p in stage.rglob('*') if p.is_file()}
        if not any(p.endswith('.safetensors') for p in files):
            raise ValueError('Checkpoint has no trainable weights')
        atomic_json(stage / 'checkpoint.json', dict(schema=1, files=files))
        for path in stage.rglob('*'):
            if path.is_file():
                with path.open('rb') as f:
                    os.fsync(f.fileno())
        fsync_dir(stage)
        stage.rename(final)
        fsync_dir(store)
        old = pointer_for(root, tag)
        pointer = dict(directory=name, manifest_sha256=sha256(final / 'checkpoint.json'))
        if tag == 'last' and old:
            atomic_json(store / 'last.previous.json', old)
        publish(root, tag, pointer)
        prune(root)
        return pointer
    finally:
        if stage.exists():
            shutil.rmtree(stage)


def prune(root):
    """Keep current/previous resume saves and their selected best; never scan outside this store."""
    root = Path(root)
    store = root / '.checkpoints'
    keep = set()
    # A damaged pointer is grounds to preserve everything, not guess at deletion.
    try:
        for filename in ('last.json', 'last.previous.json', 'best.json', 'final.json'):
            path = store / filename
            if not path.exists():
                continue
            pointer = json.loads(path.read_text())
            folder = checked_generation(root, pointer)
            keep.add(folder.name)
            if (folder / 'state.pt').exists():
                state = torch.load(folder / 'state.pt', map_location='cpu', weights_only=False)
                best = state.get('best_checkpoint')
                if best:
                    keep.add(checked_generation(root, best).name)
    except (ValueError, OSError, KeyError):
        return
    for folder in store.iterdir():
        if (folder.is_dir() and not folder.is_symlink() and folder.name not in keep
                and folder.name.startswith(('last-', 'best-', 'final-'))
                and (folder / 'checkpoint.json').is_file()):
            leases = store / '.leases'
            leases.mkdir(exist_ok=True)
            with (leases / (folder.name + '.lock')).open('a+') as lease:
                try:
                    fcntl.flock(lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError:
                    continue
                shutil.rmtree(folder)


def resolve_checkpoint(path):
    """Resolve one immutable generation before evaluation (no mixed live saves)."""
    path = Path(path).absolute()
    if path.name in ('last', 'best', 'final'):
        pointer = pointer_for(path.parent, path.name)
        if pointer:
            return checked_generation(path.parent, pointer)
    return path


def resume_checkpoint(root):
    root = Path(root)
    store = root / '.checkpoints'
    current, previous = store / 'last.json', store / 'last.previous.json'
    if not current.exists() and not previous.exists():
        if (root/'last').is_symlink() or (root/'state.pt').is_symlink():
            raise ValueError('Missing committed checkpoint pointers; inspect the generation store before resuming')
        if (root / 'state.pt').exists() and (root / 'last').is_dir():
            return root / 'last', root / 'state.pt'
        if (root / 'state.pt').exists() or (root / 'last').exists():
            raise ValueError('Incomplete legacy checkpoint: both last/ and state.pt are required')
        return None
    errors = []
    for path in (current, previous):
        try:
            pointer = json.loads(path.read_text())
            folder = checked_generation(root, pointer)
            if not (folder / 'state.pt').is_file():
                raise ValueError('Resume checkpoint lacks optimizer state')
            if errors:
                warnings.warn('Recovering previous committed checkpoint: ' + '; '.join(errors))
            publish(root, 'last', pointer)
            state = torch.load(folder / 'state.pt', map_location='cpu', weights_only=False)
            if 'best_checkpoint' in state:
                if state['best_checkpoint']:
                    publish(root, 'best', state['best_checkpoint'])
                else:
                    # A best chosen after this resume point cannot remain selected.
                    for p in (root/'best', root/'best.json', store/'best.json'):
                        if p.is_symlink() or p.is_file():
                            p.unlink()
            rewind_logs(root, state['step'])
            return folder, folder / 'state.pt'
        except (ValueError, OSError, KeyError) as error:
            errors.append(f'{path.name}: {error}')
    raise ValueError('No complete resume checkpoint: ' + '; '.join(errors))


def rewind_logs(root, step):
    for name in ('metrics.jsonl', 'gradients.jsonl'):
        path = Path(root) / name
        if not path.exists():
            continue
        lines = path.read_text().splitlines()
        retained = []
        for index, line in enumerate(lines):
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                if index != len(lines)-1:
                    raise ValueError(f'Corrupt log record in {name}') from None
                continue
            if row.get('step', 0) <= step:
                retained.append(line)
        if retained != lines:
            tmp = path.with_suffix('.rewind.tmp')
            with tmp.open('w') as f:
                f.write(''.join(line+'\n' for line in retained))
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp, path)


from contextlib import contextmanager

@contextmanager
def checkpoint_reader(path):
    """Hold a shared generation lease so training retention cannot remove a live reader."""
    path = Path(path).absolute()
    lease = None
    try:
        for attempt in range(3):
            folder = resolve_checkpoint(path)
            if folder.parent.name != '.checkpoints':
                yield folder
                return
            leases = folder.parent / '.leases'
            leases.mkdir(exist_ok=True)
            lease = (leases / (folder.name + '.lock')).open('a+')
            fcntl.flock(lease, fcntl.LOCK_SH)
            if folder.exists():
                yield folder
                return
            lease.close(); lease = None
        raise ValueError('Checkpoint changed repeatedly before it could be opened')
    finally:
        if lease:
            lease.close()
