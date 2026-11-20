"""Shared split selection for raw PGM files and portable contact-sheet manifests."""
import gzip
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image
from torch.utils.data import Dataset

from .data import PGMStream, list_npz, subset_files


def read_file_list(path):
    opener = gzip.open if str(path).endswith('.gz') else open
    with opener(path, 'rt') as stream:
        files = [line.split('\t')[0].strip() for line in stream if line.strip() and not line.startswith('#')]
    if not files or len(files) != len(set(files)):
        raise ValueError('File list must be nonempty and contain unique IDs')
    return files


def read_manifest(path):
    with open(path) as stream:
        rows = [json.loads(line) for line in stream if line.strip()]
    if not rows or len({r['id'] for r in rows}) != len(rows):
        raise ValueError('Manifest must be nonempty with unique IDs')
    for row in rows:
        if len(row['images']) != 1 or row['answer'] not in row['choices'] or not row['prompt']:
            raise ValueError('Expected one contact sheet, prompt and a valid answer per row')
        if row['choices'] not in [list('AB'), list('ABCD'), list('ABCDEFGH')]:
            raise ValueError('Unsupported answer vocabulary')
    return rows


class ManifestStream(Dataset):
    def __init__(self, manifest, files, order=None):
        self.root = Path(manifest).resolve().parent
        lookup = {r['id']: r for r in read_manifest(manifest)}
        self.rows = [lookup[f] for f in files]
        self.order = np.arange(len(files)) if order is None else np.asarray(order)
    def __len__(self):
        return len(self.order)
    def __getitem__(self, pos):
        row = self.rows[int(self.order[pos])]
        path = (self.root / row['images'][0]).resolve()
        if not path.is_relative_to(self.root):
            raise ValueError('Image path escapes manifest directory')
        digest = row.get('metadata', {}).get('contact_sheet_sha256')
        if digest and hashlib.sha256(path.read_bytes()).hexdigest() != digest:
            raise ValueError(f'Image checksum mismatch: {row["id"]}')
        with Image.open(path) as image:
            image = image.convert('RGB')
        return dict(file=row['id'], target=row['choices'].index(row['answer']), image=image,
                    prompt=row['prompt'], choices=row['choices'])


def manifest_for(args, split):
    return getattr(args, {'train': 'train_manifest', 'val': 'val_manifest', 'eval': 'manifest'}[split], None)


def files_for(args, split):
    manifest = manifest_for(args, split)
    if manifest:
        if args.protocol != 'answer_letter':
            raise ValueError('Contact-sheet manifests require --protocol answer_letter')
        return [r['id'] for r in read_manifest(manifest)]
    directory = getattr(args, {'train': 'train_dir', 'val': 'val_dir', 'eval': 'npz_dir'}[split])
    if not directory:
        raise ValueError(f'{split} needs a raw PGM directory or a contact-sheet manifest')
    cache_dir = getattr(args, 'file_cache_dir', None)
    cache = str(Path(cache_dir) / f'{split}_files.txt') if cache_dir else None
    return list_npz(directory, cache)


def select_files(available, list_path=None, count=0, seed=0):
    if not list_path:
        return subset_files(available, count, seed)
    files = read_file_list(list_path)
    missing = set(files) - set(available)
    if missing:
        raise ValueError(f'Explicit split contains unavailable files: {sorted(missing)[:3]}')
    if count and count != len(files):
        raise ValueError('Explicit file list and requested sample count disagree')
    return files


def make_stream(args, split, files, order=None, seed=0, shuffle_candidates=False, context_mode='normal'):
    manifest = manifest_for(args, split)
    if manifest:
        if shuffle_candidates or context_mode != 'normal':
            raise ValueError('Candidate/context permutation is unsupported for pre-rendered contact sheets')
        return ManifestStream(manifest, files, order)
    directory = getattr(args, {'train': 'train_dir', 'val': 'val_dir', 'eval': 'npz_dir'}[split])
    return PGMStream(directory, files, order, seed, shuffle_candidates, context_mode)


def training_order(n_files, total_positions, seed, policy='position'):
    if policy == 'position':
        from .data import make_order
        return make_order(n_files, total_positions, seed)
    if policy not in ('python_epoch', 'nested_first'):
        raise ValueError('Unknown data order')
    import random
    result = []
    epoch = 0
    while len(result) < total_positions:
        order = list(range(n_files))
        if policy == 'python_epoch' or epoch:
            random.Random(seed + epoch).shuffle(order)
        result.extend(order)
        epoch += 1
    return np.asarray(result[:total_positions])
