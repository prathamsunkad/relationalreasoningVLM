"""Token-aligned Abstractor plugin for the pinned native LFM2-VL model.

This is original relational cross-attention: visual content supplies Q/K and
address symbols (then abstract states) supply V. There is no Slot Attention,
token pooling, content-valued attention, or change to the native attention.
The plugin runs alongside the first language full-attention block and adds a
zero-initialized correction to its visual-token outputs only.

The audited interface deliberately supports one unpadded example at a time,
native image tiling, and ordinary cached one-token decoding. Gradient
checkpointing and concurrent/reentrant forwards are unsupported. Install after
moving/casting the native model; plugin parameters and arithmetic stay FP32.
"""

import math
from dataclasses import dataclass

import torch
from torch import nn


_INTEGER_DTYPES = (torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8)


def projected_addresses(spatial_shapes, downsample_factor=2, *, max_tiles=16,
                        max_rows=64, max_cols=64, image_token_count=None,
                        pixel_attention_mask=None):
    """Return ``[tile, row, column]`` in native projected-image token order.

    Native LFM2-VL flattens each downsampled image grid row-major, concatenates
    tiles in processor order, then scatters into image-token positions. Patch
    padding must be a contiguous suffix and each valid grid must be complete.
    Shape metadata is tiny; validate it on CPU once per full image prefill.
    """
    if not isinstance(spatial_shapes, torch.Tensor) or spatial_shapes.ndim != 2:
        raise ValueError('spatial_shapes must be an integer tensor of shape [tiles, 2]')
    if spatial_shapes.shape[1] != 2 or spatial_shapes.dtype not in _INTEGER_DTYPES:
        raise ValueError('spatial_shapes must contain integer [height, width] pairs')
    if not isinstance(downsample_factor, int) or isinstance(downsample_factor, bool) or downsample_factor < 1:
        raise ValueError('downsample_factor must be a positive integer')
    shapes = spatial_shapes.detach().cpu().tolist()
    if not 1 <= len(shapes) <= max_tiles:
        raise ValueError(f'Expected 1..{max_tiles} image tiles, got {len(shapes)}')
    patch_mask = None
    if pixel_attention_mask is not None:
        if not isinstance(pixel_attention_mask, torch.Tensor) or pixel_attention_mask.ndim != 2:
            raise ValueError('pixel_attention_mask must have shape [tiles, padded_patches]')
        if pixel_attention_mask.shape[0] != len(shapes):
            raise ValueError('Patch-mask tile count disagrees with spatial_shapes')
        patch_mask = pixel_attention_mask.detach().cpu()
        if not bool(((patch_mask == 0) | (patch_mask == 1)).all()):
            raise ValueError('pixel_attention_mask must be binary')
    addresses = []
    for tile, (height, width) in enumerate(shapes):
        if height < 1 or width < 1 or height % downsample_factor or width % downsample_factor:
            raise ValueError('Each native image grid must be positive and divisible by downsample_factor')
        rows, cols = height // downsample_factor, width // downsample_factor
        if rows > max_rows or cols > max_cols:
            raise ValueError(f'Projected image grid exceeds supported {max_rows}x{max_cols} address space')
        if patch_mask is not None:
            count = height * width
            row = patch_mask[tile]
            if count > row.numel() or not bool(row[:count].all()) or bool(row[count:].any()):
                raise ValueError('Patch mask must contain exactly height*width leading valid patches')
        addresses.extend((tile, row, col) for row in range(rows) for col in range(cols))
    if image_token_count is not None and len(addresses) != image_token_count:
        raise ValueError(f'Native image token count {image_token_count} != projected grid count {len(addresses)}')
    return torch.tensor(addresses, dtype=torch.long, device=spatial_shapes.device)


class RelationalCrossAttention(nn.Module):
    """softmax(Q(content) K(content)^T / sqrt(d)) V(abstract_states)."""

    def __init__(self, width, heads):
        super().__init__()
        if width < 1 or heads < 1 or width % heads:
            raise ValueError('width must be positive and divisible by positive heads')
        self.heads = heads
        self.width = width
        self.q, self.k, self.v, self.out = [nn.Linear(width, width, bias=False) for _ in range(4)]
        self.capture = False
        self.last_diagnostics = {}

    def forward(self, content, abstract_states):
        if content.ndim != 3 or content.shape != abstract_states.shape or content.shape[-1] != self.width:
            raise ValueError('RCA content and abstract states must have matching [batch, tokens, width] shapes')
        batch, tokens, width = content.shape

        def split(x):
            return x.reshape(batch, tokens, self.heads, width // self.heads).transpose(1, 2)

        query = split(self.q(content))
        key = split(self.k(content))
        value = split(self.v(abstract_states))
        weights = (query @ key.transpose(-1, -2) / math.sqrt(width // self.heads)).softmax(-1)
        result = self.out((weights @ value).transpose(1, 2).reshape(batch, tokens, width))
        if self.capture:
            with torch.no_grad():
                probabilities = weights.detach()
                entropy = -(probabilities * probabilities.clamp_min(1e-30).log()).sum(-1).mean()
                self.last_diagnostics = {'attention_entropy': float(entropy.cpu())}
        return result


class AbstractorLayer(nn.Module):
    """Fixed content Q/K; recurrent abstract-state V, residual, normalization/FFN."""

    def __init__(self, width, heads):
        super().__init__()
        self.content_norm = nn.LayerNorm(width)
        self.value_norm = nn.LayerNorm(width)
        self.relation = RelationalCrossAttention(width, heads)
        self.ff_norm = nn.LayerNorm(width)
        self.ff = nn.Sequential(nn.Linear(width, 2 * width), nn.GELU(), nn.Linear(2 * width, width))

    def forward(self, content, abstract_states):
        abstract_states = abstract_states + self.relation(self.content_norm(content), self.value_norm(abstract_states))
        return abstract_states + self.ff(self.ff_norm(abstract_states))


class AbstractorPlugin(nn.Module):
    """Return a visual-token correction; never add visual content to symbol V."""

    def __init__(self, input_dim, width=256, heads=4, depth=2, *, max_tiles=16, max_rows=64, max_cols=64):
        super().__init__()
        if input_dim < 1 or width < 1 or heads < 1 or width % heads or depth < 1:
            raise ValueError('Invalid plugin dimensions, head count, or depth')
        self.spec = dict(architecture='original_rca_parallel_visual_plugin_v1', input_dim=input_dim,
                         width=width, heads=heads, depth=depth, max_tiles=max_tiles,
                         max_rows=max_rows, max_cols=max_cols, symbols='tile+projected_row+projected_column',
                         token_pooling=False, internal_dtype='float32', gate='sigmoid(learned_scalar)',
                         output_initialization='zero', symbol_self_attention=False)
        self.content_in = nn.Sequential(nn.LayerNorm(input_dim), nn.Linear(input_dim, width))
        self.tile_symbols = nn.Embedding(max_tiles, width)
        self.row_symbols = nn.Embedding(max_rows, width)
        self.col_symbols = nn.Embedding(max_cols, width)
        for embedding in (self.tile_symbols, self.row_symbols, self.col_symbols):
            nn.init.normal_(embedding.weight, std=0.02)
        self.layers = nn.ModuleList([AbstractorLayer(width, heads) for _ in range(depth)])
        self.output_norm = nn.LayerNorm(width)
        self.output = nn.Linear(width, input_dim, bias=False)
        nn.init.zeros_(self.output.weight)
        self.gate = nn.Parameter(torch.zeros(()))  # sigmoid(0) = 0.5, not a second zero.
        self.capture = False
        self.last_diagnostics = {}

    def set_capture_diagnostics(self, enabled=True):
        self.capture = bool(enabled)
        self.clear_diagnostics()
        for layer in self.layers:
            layer.relation.capture = self.capture

    def clear_diagnostics(self):
        self.last_diagnostics = {}
        for layer in self.layers:
            layer.relation.last_diagnostics = {}

    def diagnostics(self, include_gradients=False):
        """Explicit snapshots contain numbers only, never graph-bearing tensors."""
        result = dict(self.last_diagnostics)
        if include_gradients:
            result['gradient_norms'] = {
                name: None if parameter.grad is None else float(parameter.grad.detach().float().norm().cpu())
                for name, parameter in self.named_parameters()
            }
        return result

    def forward(self, visual_content, addresses):
        if visual_content.ndim != 3 or visual_content.shape[0] != 1 or visual_content.shape[-1] != self.spec['input_dim']:
            raise ValueError('Plugin supports [1, native_visual_tokens, input_dim] content only')
        if addresses.shape != (visual_content.shape[1], 3) or addresses.dtype != torch.long:
            raise ValueError('Expected one integer [tile,row,column] address per visual token')
        if self.output.weight.dtype != torch.float32:
            raise ValueError('Plugin parameters must remain FP32; install after casting the native model')
        addresses = addresses.to(device=visual_content.device)
        with torch.autocast(device_type=visual_content.device.type, enabled=False):
            content = self.content_in(visual_content.float())
            abstract_states = (self.tile_symbols(addresses[:, 0]) + self.row_symbols(addresses[:, 1])
                               + self.col_symbols(addresses[:, 2])).unsqueeze(0)
            for layer in self.layers:
                # Q/K content is fixed across layers; only the symbol stream evolves.
                abstract_states = layer(content, abstract_states)
            delta = self.gate.sigmoid() * self.output(self.output_norm(abstract_states))
            if self.capture:
                with torch.no_grad():
                    input_rms = visual_content.detach().float().square().mean().sqrt()
                    delta_rms = delta.detach().square().mean().sqrt()
                    self.last_diagnostics = dict(
                        visual_tokens=visual_content.shape[1], gate=float(self.gate.detach().sigmoid().cpu()),
                        input_rms=float(input_rms.cpu()), delta_rms=float(delta_rms.cpu()),
                        delta_to_input_rms=float((delta_rms / input_rms.clamp_min(1e-12)).cpu()),
                        abstract_rms=float(abstract_states.detach().square().mean().sqrt().cpu()),
                        attention_entropy=[layer.relation.last_diagnostics['attention_entropy'] for layer in self.layers])
        return delta.to(dtype=visual_content.dtype)


@dataclass
class _Prefill:
    positions: torch.Tensor
    addresses: torch.Tensor
    sequence_length: int
    applied: bool = False


class Controller:
    """Serial native-model hooks with per-forward context and exact bypass."""

    def __init__(self, base, plugin, layer_index):
        self.base = base
        self.plugin = plugin
        self.layer_index = layer_index
        self.spec = dict(plugin.spec, layer_index=layer_index, insertion='self_attn_output_parallel_visual_only',
                         content_source='self_attn_input_visual_positions', image_token_id=int(base.config.image_token_id),
                         downsample_factor=int(base.config.downsample_factor))
        self._enabled = True
        self._removed = False
        self._context = None
        self._has_prefill = False
        self._active = False
        attention = base.model.language_model.layers[layer_index].self_attn
        self.handles = [base.register_forward_pre_hook(self._before, with_kwargs=True),
                        attention.register_forward_hook(self._attention, with_kwargs=True),
                        base.register_forward_hook(self._after, with_kwargs=True, always_call=True)]

    @property
    def enabled(self):
        return self._enabled

    @enabled.setter
    def enabled(self, enabled):
        self.set_enabled(enabled)

    def set_enabled(self, enabled):
        if self._removed:
            raise RuntimeError('The Abstractor plugin has been removed')
        self._enabled = bool(enabled)
        self._context = None
        self._has_prefill = False
        self.plugin.clear_diagnostics()

    def capture_diagnostics(self, enabled=True):
        self.plugin.set_capture_diagnostics(enabled)

    def diagnostics(self, include_gradients=False):
        return self.plugin.diagnostics(include_gradients=include_gradients)

    def _before(self, module, args, kwargs):
        self.plugin.clear_diagnostics()
        self._context = None
        if not self.enabled:
            self._has_prefill = False
            return
        if self._active:
            raise ValueError('Concurrent or reentrant forwards are unsupported')
        self._active = True
        ids = kwargs.get('input_ids', args[0] if args else None)
        if not isinstance(ids, torch.Tensor) or ids.ndim != 2 or ids.shape[0] != 1:
            raise ValueError('Plugin supports exactly one example per forward, with input_ids')
        if ids.dtype not in _INTEGER_DTYPES or ids.shape[1] < 1 or kwargs.get('inputs_embeds') is not None:
            raise ValueError('Integer input_ids are required; inputs_embeds is unsupported')
        if bool(getattr(module, 'is_gradient_checkpointing', False)):
            raise ValueError('Gradient checkpointing is unsupported by the stateful plugin hooks')
        pixels = kwargs.get('pixel_values')
        if pixels is None:
            if not self._has_prefill or kwargs.get('past_key_values') is None or ids.shape[1] != 1:
                raise ValueError('Text-only call must be a cached one-token continuation after image prefill')
            if bool((ids == self.spec['image_token_id']).any()):
                raise ValueError('A cached continuation cannot introduce a new image token')
            return

        self._has_prefill = False
        mask = kwargs.get('attention_mask')
        if mask is not None and (mask.shape != ids.shape or not bool((mask == 1).all())):
            raise ValueError('Padded or nonstandard language attention masks are unsupported')
        cache_position = kwargs.get('cache_position')
        if cache_position is not None:
            expected = torch.arange(ids.shape[1], device=cache_position.device)
            if cache_position.ndim != 1 or not torch.equal(cache_position, expected):
                raise ValueError('Image prefill must start at cache position zero')
        cache = kwargs.get('past_key_values')
        if cache is not None:
            if not hasattr(cache, 'get_seq_length') or cache.get_seq_length() != 0:
                raise ValueError('Image prefill cannot extend a populated or unsupported cache')
        shapes = kwargs.get('spatial_shapes')
        patch_mask = kwargs.get('pixel_attention_mask')
        if not isinstance(pixels, torch.Tensor) or pixels.ndim != 3 or patch_mask is None:
            raise ValueError('Expected native packed image patches and pixel_attention_mask')
        if not isinstance(patch_mask, torch.Tensor) or tuple(patch_mask.shape) != tuple(pixels.shape[:2]):
            raise ValueError('pixel_attention_mask must match native packed image patches')
        positions = (ids[0] == self.spec['image_token_id']).nonzero(as_tuple=False).flatten()
        addresses = projected_addresses(shapes, self.spec['downsample_factor'],
                                        max_tiles=self.spec['max_tiles'], max_rows=self.spec['max_rows'],
                                        max_cols=self.spec['max_cols'], image_token_count=positions.numel(),
                                        pixel_attention_mask=patch_mask)
        if pixels.shape[0] != shapes.shape[0]:
            raise ValueError('Packed image tile count disagrees with spatial_shapes')
        self._context = _Prefill(positions, addresses, ids.shape[1])

    def _attention(self, module, args, kwargs, result):
        if not self.enabled:
            return result
        if not self._active:
            raise ValueError('Self-attention was invoked outside the audited native forward')
        context = self._context
        if context is None:  # Cached decode: the transformed visual prefix is already in downstream caches.
            return result
        if context.applied:
            raise ValueError('The selected self-attention block was invoked more than once in one prefill')
        hidden = kwargs.get('hidden_states', args[0] if args else None)
        if not isinstance(hidden, torch.Tensor) or tuple(hidden.shape) != (1, context.sequence_length, self.spec['input_dim']):
            raise ValueError('Self-attention hidden-state shape disagrees with prefill input_ids')
        if not isinstance(result, tuple) or not result or not isinstance(result[0], torch.Tensor) or result[0].shape != hidden.shape:
            raise ValueError('Expected native self-attention (output, ...) tuple with unchanged token shape')
        positions = context.positions.to(hidden.device)
        visual = hidden.index_select(1, positions)
        delta = self.plugin(visual, context.addresses)
        native = result[0]
        updated = native.clone()
        updated[:, positions, :] = native.index_select(1, positions) + delta.to(dtype=native.dtype)
        context.applied = True
        return (updated, *result[1:])

    def _after(self, module, args, kwargs, result):
        missing_hook = self._context is not None and result is not None and not self._context.applied
        if self._context is not None and result is not None and not missing_hook:
            self._has_prefill = True
        if result is None or missing_hook:
            self._has_prefill = False
            self.plugin.clear_diagnostics()
        self._context = None
        self._active = False
        if missing_hook:
            raise ValueError('Native image prefill did not invoke the selected self-attention block')

    def remove(self):
        if self._removed:
            return
        for handle in self.handles:
            handle.remove()
        self.handles.clear()
        self._context = None
        self._has_prefill = False
        self._active = False
        self.plugin.clear_diagnostics()
        if getattr(self.base, 'abstractor_plugin', None) is self.plugin:
            delattr(self.base, 'abstractor_plugin')
        if getattr(self.base, '_abstractor_plugin_controller', None) is self:
            delattr(self.base, '_abstractor_plugin_controller')
        self._enabled = False
        self._removed = True


def install(base, processor=None, layer_index=2, width=256, heads=4, depth=2):
    """Install on a native LFM2-VL base (``peft_model.get_base_model()``).

    ``processor`` is accepted for compatibility but no tokenizer or answer text
    is consulted. Existing native and LoRA parameters/requires_grad are intact.
    """
    del processor
    if hasattr(base, 'abstractor_plugin') or hasattr(base, '_abstractor_plugin_controller'):
        raise ValueError('Abstractor plugin already installed')
    if getattr(base.config, 'model_type', None) != 'lfm2_vl':
        raise ValueError('Only audited native LFM2-VL models are supported')
    layer_types = base.config.text_config.layer_types
    if not isinstance(layer_index, int) or layer_index < 0 or layer_index >= len(layer_types):
        raise ValueError('Invalid insertion layer index')
    if layer_types[layer_index] != 'full_attention' or layer_index != layer_types.index('full_attention'):
        raise ValueError('Plugin must attach to the first native full_attention layer')
    layer = base.model.language_model.layers[layer_index]
    if not isinstance(getattr(layer, 'self_attn', None), nn.Module):
        raise ValueError('Selected native layer has no self_attn module')
    plugin = AbstractorPlugin(base.config.text_config.hidden_size, width, heads, depth)
    plugin.to(device=next(base.parameters()).device, dtype=torch.float32)
    base.add_module('abstractor_plugin', plugin)
    try:
        controller = Controller(base, plugin, layer_index)
    except Exception:
        delattr(base, 'abstractor_plugin')
        raise
    base._abstractor_plugin_controller = controller
    return controller
