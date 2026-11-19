"""Attach experimental LFM modules to the shared trainer; keep base weights frozen."""
import json
import math
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

NAMES = ('none', 'early', 'late', 'joint', 'memory', 'dual', 'rca_plugin', 'rca_memory')


def add_extension_args(parser, training=False):
    parser.add_argument('--extension', choices=NAMES, default='none')
    parser.add_argument('--readout', choices=['llm', 'classifier', 'classifier_only'], default='llm')
    parser.add_argument('--insertion_layer', type=int, default=None, help='late probe: decoder block index instead of projector')
    parser.add_argument('--extension_width', type=int, default=256)
    parser.add_argument('--extension_depth', type=int, default=2)
    parser.add_argument('--extension_heads', type=int, default=4)
    parser.add_argument('--extension_symbols', type=int, default=128)
    parser.add_argument('--correction_cap', type=float, default=None, help='Per-example correction/native RMS bound')
    parser.add_argument('--correction_scale', type=float, default=1.)
    if training:
        parser.add_argument('--train_mode', choices=['lora', 'module', 'lora_module'], default='lora')
        parser.add_argument('--extension_lr', type=float, default=None)
        parser.add_argument('--init_adapter', default=None, help='Initialize from existing LoRA weights; new optimizer')


def base_model(model):
    return model.get_base_model() if hasattr(model, 'get_base_model') else model


def extension_module(model):
    base = base_model(model)
    return getattr(base, 'relation_probe', getattr(base, 'abstractor_plugin', None))


def bounded_correction(delta, native, cap):
    if cap is None:
        return delta
    if not math.isfinite(cap) or cap <= 0:
        raise ValueError('Correction cap must be finite and positive')
    dims = tuple(range(1, delta.ndim))
    limit = cap * native.detach().float().square().mean(dims, keepdim=True).sqrt()
    raw = delta.float()
    scale = limit / (limit.square() + raw.square().mean(dims, keepdim=True)).clamp_min(1e-30).sqrt()
    return (raw * scale).to(delta.dtype)


def extension_spec(args):
    return {key: getattr(args, key) for key in ('extension', 'extension_width', 'extension_depth',
            'extension_heads', 'extension_symbols', 'correction_cap', 'correction_scale', 'readout', 'insertion_layer')}


def attach_extension(model, processor, args, checkpoint=None):
    """Called after native device/dtype conversion and optional PEFT wrapping."""
    if args.extension == 'none':
        return None
    if args.protocol != 'answer_letter':
        raise ValueError('The LFM modules were audited with the serial answer-letter protocol')
    base = base_model(model)
    if getattr(base.config, 'model_type', None) != 'lfm2_vl':
        raise ValueError('These extension hooks require native LFM2-VL')
    if args.readout != 'llm':
        if args.extension != 'late' or args.insertion_layer is not None:
            raise ValueError('Classifier readout uses the late/projector diagnostic only')
        from types import SimpleNamespace
        from .probe_diagnostics import DiagnosticArm
        name = 'classifier_only' if args.readout == 'classifier_only' else f'classifier_d{args.extension_depth}'
        module = DiagnosticArm(name, input_dim=base.config.text_config.hidden_size,
                               width=args.extension_width, bins=args.extension_symbols)
        module.to(device=next(base.parameters()).device, dtype=torch.float32)
        base.add_module('relation_probe', module)
        base._extension_readout = args.readout
        ctl = SimpleNamespace(probe=module, handles=[])
    elif args.insertion_layer is not None:
        if args.extension != 'late':
            raise ValueError('Block insertion currently uses the late probe only')
        from .locations import LocationController
        ctl = LocationController(base, args.insertion_layer, args.extension_width,
                                  args.extension_heads, args.extension_depth, args.extension_symbols)
    elif args.extension in ('rca_plugin', 'rca_memory'):
        from . import abstractor_plugin, abstractor_memory
        source = abstractor_plugin if args.extension == 'rca_plugin' else abstractor_memory
        ctl = source.install(base, processor, width=args.extension_width,
                             heads=args.extension_heads, depth=args.extension_depth)
    else:
        from . import relational
        ctl = relational.install(base, processor, args.extension, width=args.extension_width,
                                 heads=args.extension_heads, depth=args.extension_depth,
                                 symbols=args.extension_symbols)
    module = extension_module(model)
    if not math.isfinite(args.correction_scale) or args.correction_scale < 0:
        raise ValueError('Correction scale must be finite and nonnegative')
    if args.correction_cap is not None or args.correction_scale != 1.:
        if args.readout != 'llm':
            raise ValueError('Correction limits apply to LLM writeback, not classifier scores')
        if args.extension == 'memory':
            raise ValueError('Correction limits currently apply to visual/prefix branches, not memory readers')
        residual_output = args.extension not in ('rca_plugin', 'rca_memory')
        def bound(mod, inputs, output):
            native = inputs[0]
            delta = output - native if residual_output else output
            delta = bounded_correction(delta * args.correction_scale, native, args.correction_cap)
            return native + delta if residual_output else delta
        ctl.handles.append(module.register_forward_hook(bound))
    if checkpoint:
        path = Path(checkpoint)
        spec = json.loads((path / 'extension.json').read_text())
        if spec != extension_spec(args):
            raise ValueError('Extension checkpoint configuration does not match this run')
        module.load_state_dict(load_file(str(path / 'extension.safetensors')), strict=True)
    return ctl


def save_model(model, path, args):
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    if getattr(args, 'check_lustre_quota', False):
        from pgmvlm.storage import require_headroom
        n = sum(p.numel()*p.element_size() for p in model.parameters() if p.requires_grad)
        require_headroom(path, reserve_bytes=2*1024**3 + 3*n)
    if hasattr(model, 'peft_config'):
        model.save_pretrained(path, safe_serialization=True, save_embedding_layers=False)
    elif args.extension == 'none':
        raise ValueError('Refusing to save full base weights as an adapter')
    module = extension_module(model)
    if module is not None:
        save_file({k: v.detach().cpu().contiguous() for k, v in module.state_dict().items()}, str(path / 'extension.safetensors'))
        (path / 'extension.json').write_text(json.dumps(extension_spec(args), indent=2) + '\n')
    (path / 'protocol.json').write_text(json.dumps(dict(protocol=args.protocol, answer_loss_scope=args.answer_loss_scope,
                                                       model_id=args.model_id, revision=args.revision), indent=2) + '\n')


def check_checkpoint_protocol(path, args):
    path = Path(path)
    if (path / 'protocol.json').exists():
        stored = json.loads((path / 'protocol.json').read_text())
        for key in ('protocol', 'answer_loss_scope', 'model_id', 'revision'):
            if stored[key] != getattr(args, key):
                raise ValueError(f'Checkpoint {key} differs from evaluation/training configuration')
    if (path / 'extension.json').exists():
        stored = json.loads((path / 'extension.json').read_text())
        if stored != extension_spec(args):
            raise ValueError('Specify the saved extension configuration when loading this checkpoint')
    elif args.extension != 'none':
        raise ValueError('Requested an extension but checkpoint contains none')


def set_train_mode(model, args):
    if args.train_mode == 'module':
        model.eval()
        extension_module(model).train()
    else:
        from pgmvlm.models import set_train_mode as original
        original(model)
