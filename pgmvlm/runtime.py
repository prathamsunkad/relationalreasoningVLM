"""Run identity and checkpoint integrity for the shared training loop."""
import hashlib
import json
import random
from pathlib import Path

import numpy as np
import torch


def digest_json(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def atomic_json(path, value):
    path = Path(path)
    tmp = path.with_name(path.name + '.tmp')
    tmp.write_text(json.dumps(value, indent=2) + '\n')
    tmp.replace(path)


def run_contract(args, train_files, val_files):
    train_ids = train_files if getattr(args, 'train_manifest', None) else [str((Path(args.train_dir) / f).resolve()) for f in train_files]
    val_ids = val_files if getattr(args, 'val_manifest', None) else [str((Path(args.val_dir) / f).resolve()) for f in val_files]
    if set(train_ids) & set(val_ids):
        raise ValueError('Training and validation IDs overlap')
    keys = ('model_id', 'revision', 'protocol', 'answer_loss_scope', 'loss', 'seed',
            'batch_size', 'grad_accum', 'lora_r', 'lora_alpha', 'lora_dropout',
            'train_mode', 'extension', 'extension_width', 'extension_depth', 'extension_heads',
            'extension_symbols', 'correction_scale', 'correction_cap', 'init_adapter', 'init_extension',
            'dtype', 'attn', 'select_by', 'early_stop_patience', 'early_stop_min_delta', 'deterministic', 'readout', 'insertion_layer', 'world_size', 'data_order', 'pgm_presentation',
            'lr', 'extension_lr', 'weight_decay', 'max_grad_norm', 'lora_vision', 'lora_projector',
            'grad_ckpt', 'panel_px', 'crop_to_patches', 'max_new_tokens')
    result = {k: getattr(args, k, None) for k in keys}
    result.update(train_count=len(train_files), train_ids_sha256=digest_json(train_ids),
                  val_count=len(val_files), val_ids_sha256=digest_json(val_ids))
    for key in ('train_manifest', 'val_manifest'):
        path = getattr(args, key, None)
        result[key + '_sha256'] = hashlib.sha256(Path(path).read_bytes()).hexdigest() if path else None
    path = Path(args.out_dir) / 'contract.json'
    if path.exists():
        previous = json.loads(path.read_text())
        # Upgrade pre-transaction shared runs only when their recorded args establish the missing settings.
        old_args_path = Path(args.out_dir) / 'args.json'
        old_args = json.loads(old_args_path.read_text()) if old_args_path.exists() else {}
        for key in set(result) - set(previous):
            if key == 'pgm_presentation':
                previous[key] = 'composite'
            elif key in old_args:
                previous[key] = old_args[key]
        # An explicit schedule restart/reset intentionally permits a new base LR.
        if getattr(args, 'reset_schedule', False) or getattr(args, 'lr_schedule', None) == 'restart':
            previous['lr'] = result['lr']
        if previous != result:
            raise ValueError('Run protocol, dataset IDs or training configuration changed; use a new output directory')
    if not path.exists() and (Path(args.out_dir) / 'state.pt').exists() and args.protocol != 'candidate':
        raise ValueError('Cannot resume answer-letter optimizer state without its run contract')
    atomic_json(path, result)
    return result


def capture_rng():
    return dict(python=random.getstate(), numpy=np.random.get_state(), torch=torch.get_rng_state(),
                cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [])


def restore_rng(state):
    random.setstate(state['python'])
    np.random.set_state(state['numpy'])
    torch.set_rng_state(state['torch'])
    if state['cuda']:
        torch.cuda.set_rng_state_all(state['cuda'])


def model_digest(directory):
    files = sorted(Path(directory).glob('*.safetensors'))
    if not files:
        raise ValueError('No trainable weights found in checkpoint')
    h = hashlib.sha256()
    for path in files:
        h.update(path.name.encode())
        h.update(path.read_bytes())
    return h.hexdigest()


def eval_contract(args, files):
    from .extensions import extension_spec
    contract = {k: getattr(args, k) for k in ('model_id', 'revision', 'protocol', 'answer_loss_scope',
                'max_new_tokens', 'num_shards', 'max_examples', 'dtype', 'attn', 'context_mode', 'panel_px', 'crop_to_patches', 'family', 'merge_adapter')}
    contract.update(extension_spec(args))
    contract['pgm_presentation'] = getattr(args, 'pgm_presentation', 'composite')
    contract['disable_extension'] = getattr(args, 'disable_extension', False)
    if args.adapter:
        contract['adapter_metadata'] = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in Path(args.adapter).glob('*.json')}
    contract.update(files_sha256=digest_json(files), data=str(Path(args.manifest or args.npz_dir).resolve()),
                    adapter_sha256=model_digest(args.adapter) if args.adapter else None)
    if args.manifest:
        contract['manifest_sha256'] = hashlib.sha256(Path(args.manifest).read_bytes()).hexdigest()
    path = Path(args.out_dir) / 'eval_contract.json'
    # Exclusive creation prevents two starting array workers from replacing each other's identity.
    import os
    import time
    try:
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError:
        for attempt in range(20):
            try:
                old = json.loads(path.read_text())
                break
            except json.JSONDecodeError:
                time.sleep(0.05)
        else:
            raise ValueError('Incomplete evaluation contract; inspect before resuming')
        if old != contract:
            raise ValueError('Evaluation model/protocol/data/sharding changed; choose a new output directory')
    else:
        with os.fdopen(fd, 'w') as stream:
            json.dump(contract, stream, indent=2)


from contextlib import contextmanager

@contextmanager
def preserved_rng():
    state = capture_rng()
    try:
        yield
    finally:
        restore_rng(state)
