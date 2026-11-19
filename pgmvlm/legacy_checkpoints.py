"""Read historical weights without importing their campaign-specific optimizers.

Conversion is offline, non-destructive, and uses torch's weights-only loader.
The caller supplies the model/protocol/extension recipe; recorded metadata and
all tensor keys/shapes are checked before a destination is published.
"""
import json
from pathlib import Path
import re
import shutil
import uuid

import torch
from safetensors.torch import load_file, save_file
from .checkpoints import atomic_json, sha256, fsync_dir
from .extensions import extension_spec
from .registry import MODEL_REVISIONS


def member(root, name):
    path = root / name
    if Path(name).is_absolute() or '..' in Path(name).parts or path.is_symlink() or not path.is_file() or not path.resolve().is_relative_to(root.resolve()):
        raise ValueError(f'Unsafe/missing legacy checkpoint member: {name}')
    return path


def locate(source, pointer):
    source = Path(source).resolve()
    evidence = {}
    if source.is_file():
        return source, evidence
    if (source / (pointer + '.json')).exists():
        path = source / (pointer + '.json')
        receipt = json.loads(path.read_text())
        if 'directory' not in receipt:
            raise ValueError('Pointer does not identify a checkpoint directory')
        name = receipt['directory']
        if Path(name).name != name or (source / name).is_symlink():
            raise ValueError('Unsafe historical checkpoint pointer')
        evidence[str(path)] = sha256(path)
        source = source / name
        if receipt.get('manifest_sha256') != (sha256(source/'manifest.json') if (source/'manifest.json').exists() else None):
            raise ValueError('Historical pointer/manifest checksum mismatch')
    if not source.is_dir():
        raise ValueError('Checkpoint does not exist')
    for name in ('manifest.json', 'checksums.json'):
        path = source / name
        if not path.exists():
            continue
        data = json.loads(path.read_text())
        hashes = data['files'] if name == 'manifest.json' else data
        for rel, digest in hashes.items():
            file = member(source, rel)
            if sha256(file) != digest:
                raise ValueError(f'Historical checksum mismatch: {rel}')
            evidence[str(file)] = digest
        evidence[str(path)] = sha256(path)
    return source, evidence


def build_module(args):
    if args.extension == 'none':
        return None
    if args.extension in ('rca_plugin', 'rca_memory'):
        from .extensions.abstractor_plugin import AbstractorPlugin
        from .extensions.abstractor_memory import AbstractorMemory
        cls = AbstractorPlugin if args.extension == 'rca_plugin' else AbstractorMemory
        return cls(args.text_dim, args.extension_width, args.extension_heads, args.extension_depth)
    if args.readout != 'llm':
        from .extensions.probe_diagnostics import DiagnosticArm
        name = 'classifier_only' if args.readout == 'classifier_only' else f'classifier_d{args.extension_depth}'
        return DiagnosticArm(name, input_dim=args.text_dim, width=args.extension_width, bins=args.extension_symbols)
    from .extensions.relational import RelationalProbe
    return RelationalProbe(args.extension, vision_dim=args.vision_dim, text_dim=args.text_dim,
                           width=args.extension_width, heads=args.extension_heads,
                           depth=args.extension_depth, symbols=args.extension_symbols)


def metadata(source):
    values = []
    root = source if source.is_dir() else source.parent
    for parent in (root, root.parent):
        for filename in ('manifest.json', 'config.json'):
            path = parent / filename
            if path.is_file():
                values.append(json.loads(path.read_text()))
    return values


def check_metadata(values, args):
    """Validate information that survived archival; never infer scale from shapes."""
    expected_model, pinned = MODEL_REVISIONS.get(args.model_id, (args.model_id, args.revision))
    revision = args.revision or pinned
    candidates = []
    for value in values:
        candidates.append(value)
        identity = value.get('identity', {})
        candidates.extend((identity, identity.get('plan', {}), value.get('plan', {})))
    for value in candidates:
        if not isinstance(value, dict):
            continue
        old_model = value.get('model_id', value.get('model'))
        if old_model and isinstance(old_model, str):
            old_model = MODEL_REVISIONS.get(old_model, (old_model, None))[0]
            if old_model != expected_model:
                raise ValueError('Historical base model differs from conversion recipe')
        if value.get('revision') and value['revision'] != revision:
            raise ValueError('Historical base revision differs from conversion recipe')
        if value.get('variant') and value['variant'] != args.extension:
            raise ValueError('Historical architecture differs from conversion recipe')
        for key in ('probe', 'plugin'):
            spec = value.get(key, {})
            if isinstance(spec, dict):
                for old, new in (('width','extension_width'),('depth','extension_depth'),('heads','extension_heads'),('symbols','extension_symbols')):
                    if old in spec and spec[old] != getattr(args, new):
                        raise ValueError(f'Historical {key} {old} differs from recipe')
        for old, new in (('max_correction_ratio', 'correction_cap'), ('correction_cap', 'correction_cap')):
            if old in value and value[old] != getattr(args, new):
                raise ValueError('Historical correction bound differs from recipe')
        if 'plugin_output_scale' in value and value['plugin_output_scale'] != args.correction_scale:
            raise ValueError('Historical plugin output scale differs from recipe')
    return candidates


def convert(source, output, args, pointer='best', adapter_config=None):
    output = Path(output).absolute()
    if output.exists():
        raise ValueError('Destination already exists; conversion never overwrites checkpoints')
    folder, evidence = locate(source, pointer)
    values = metadata(folder)
    metadata_root = folder if folder.is_dir() else folder.parent
    for parent in (metadata_root, metadata_root.parent):
        for name in ('manifest.json', 'config.json'):
            if (parent/name).is_file():
                evidence[str(parent/name)] = sha256(parent/name)
    raw, module_state, adapter_state, config = None, None, None, None
    if folder.is_file():
        payload = torch.load(folder, map_location='cpu', weights_only=True)
        evidence[str(folder)] = sha256(folder)
        if not isinstance(payload, dict) or 'arm' not in payload:
            raise ValueError('Expected an archived diagnostic arm checkpoint')
        module_state = payload['arm']
        values.append(payload.get('identity', {}))
        values.append(payload.get('plan', {}))
    elif (folder/'weights.pt').exists():
        raw = torch.load(folder/'weights.pt', map_location='cpu', weights_only=True)
        evidence[str(folder/'weights.pt')] = sha256(folder/'weights.pt')
    else:
        probe = folder/'probe.safetensors'
        if probe.exists():
            module_state = load_file(str(probe)); evidence[str(probe)] = sha256(probe)
        adapter = folder/'adapter' if (folder/'adapter').is_dir() else folder
        if (adapter/'adapter_model.safetensors').exists():
            adapter_state = load_file(str(adapter/'adapter_model.safetensors'))
            config = json.loads((adapter/'adapter_config.json').read_text())
            for name in ('adapter_model.safetensors', 'adapter_config.json'):
                evidence[str(adapter/name)] = sha256(adapter/name)
    candidates = check_metadata(values, args)
    if raw is not None:
        if not isinstance(raw, dict) or not raw or not all(isinstance(t, torch.Tensor) for t in raw.values()):
            raise ValueError('Invalid historical tensor dictionary')
        module_state, adapter_state = {}, {}
        for name, tensor in raw.items():
            match = re.search(r'(?:^|\.)(abstractor_plugin|relation_probe)\.(.+)', name)
            if match:
                module_state[match[2]] = tensor
            elif '.lora_A.' in name or '.lora_B.' in name:
                key = name.replace('.lora_A.default.', '.lora_A.').replace('.lora_B.default.', '.lora_B.')
                if key in adapter_state:
                    raise ValueError('Duplicate LoRA tensor after name conversion')
                adapter_state[key] = tensor
            else:
                raise ValueError(f'Unrecognized historical tensor: {name}')
        module_state = module_state or None
        adapter_state = adapter_state or None
    module = build_module(args)
    if (module is None) != (module_state is None):
        raise ValueError('Recipe and checkpoint disagree about presence of architectural weights')
    if module is not None:
        expected = module.state_dict()
        if set(module_state) != set(expected) and all(k.startswith('probe.') for k in module_state):
            module_state = {k.removeprefix('probe.'): v for k,v in module_state.items()}
        module.load_state_dict(module_state, strict=True)
        # load_state_dict casts dtypes; keep original tensors after checking shapes.
        if any(not torch.isfinite(t).all() for t in module_state.values()):
            raise ValueError('Nonfinite architectural weights')
    if adapter_state:
        if adapter_config:
            supplied = json.loads(Path(adapter_config).read_text())
            if config is not None and supplied != config:
                raise ValueError('Supplied adapter config differs from saved configuration')
            config = supplied
            evidence[str(Path(adapter_config).resolve())] = sha256(adapter_config)
        if config is None:
            lora = next((v['lora'] for v in candidates if isinstance(v.get('lora'),dict)), None)
            if not lora or not {'r','lora_alpha','lora_dropout'} <= lora.keys():
                raise ValueError('Packed LoRA needs --adapter-config from the original adapter or recorded plan.lora')
            from peft import LoraConfig
            targets = {key.removeprefix('base_model.model.').split('.lora_')[0] for key in adapter_state}
            config = LoraConfig(r=lora['r'], lora_alpha=lora['lora_alpha'], lora_dropout=lora['lora_dropout'],
                                bias='none', target_modules=targets).to_dict()
            config['target_modules'] = sorted(targets)
            config['base_model_name_or_path'] = MODEL_REVISIONS.get(args.model_id,(args.model_id,None))[0]
        saved_base = config.get('base_model_name_or_path')
        expected_base = MODEL_REVISIONS.get(args.model_id, (args.model_id, None))[0]
        if saved_base and saved_base != expected_base:
            raise ValueError('Saved adapter base differs from conversion model_id')
        if config.get('peft_type', 'LORA') != 'LORA' or config.get('bias', 'none') != 'none':
            raise ValueError('Only plain bias-free LoRA checkpoints are supported')
        if config.get('rank_pattern') or config.get('alpha_pattern') or config.get('use_dora'):
            raise ValueError('Historical conversion currently supports uniform plain LoRA only')
        for key, tensor in adapter_state.items():
            partner = key.replace('.lora_A.', '.lora_B.') if '.lora_A.' in key else key.replace('.lora_B.', '.lora_A.')
            if (not re.search(r'\.lora_[AB]\.weight$', key) or tensor.ndim != 2 or partner not in adapter_state
                    or tensor.shape[0 if '.lora_A.' in key else 1] != config['r'] or not torch.isfinite(tensor).all()):
                raise ValueError(f'Invalid or incomplete LoRA weights: {key}')
    if not module_state and not adapter_state:
        raise ValueError('No recognized checkpoint weights found')
    output.parent.mkdir(parents=True, exist_ok=True)
    stage = output.with_name('.'+output.name+'-'+uuid.uuid4().hex)
    stage.mkdir()
    try:
        if adapter_state:
            save_file({k:v.contiguous() for k,v in adapter_state.items()}, str(stage/'adapter_model.safetensors'))
            atomic_json(stage/'adapter_config.json', config)
        if module_state:
            save_file({k:v.contiguous() for k,v in module_state.items()}, str(stage/'extension.safetensors'))
            atomic_json(stage/'extension.json', extension_spec(args))
        atomic_json(stage/'protocol.json', dict(protocol=args.protocol, answer_loss_scope=args.answer_loss_scope,
                                                model_id=args.model_id, revision=args.revision))
        result = dict(schema=1, source=str(Path(source).resolve()), resolved_source=str(folder),
                      source_sha256=evidence, extension=extension_spec(args),
                      optimizer_imported=False, continuation='weights only; use a new output directory and optimizer',
                      files={p.name:sha256(p) for p in stage.iterdir() if p.is_file()})
        atomic_json(stage/'conversion.json', result)
        for p in stage.iterdir():
            with p.open('rb') as f:
                import os
                os.fsync(f.fileno())
        fsync_dir(stage)
        stage.rename(output)
        fsync_dir(output.parent)
        return result
    finally:
        if stage.exists():
            shutil.rmtree(stage)
