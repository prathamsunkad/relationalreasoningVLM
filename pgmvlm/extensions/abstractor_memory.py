"""Original-Abstractor relational memory read by a frozen native LFM2-VL.

Image hidden states supply content Q/K to the same RCA implementation used by
the visual plugin; address symbols supply V. A separate cross-attention reader
lets *subsequent* language tokens read those relational states. The image-token
stream and the native answer head are unchanged. There are no slots or pooling.

The serial, batch-one interface follows ``abstractor_plugin``. Memory is built
from the first full-attention layer's input image positions, before any reader
correction. Restricting readers to positions after the final image token avoids
future-image leakage into preceding text. Native tiling inserts markers between
image-token runs, so image positions need not be contiguous. Cached one-token
inference reuses detached memory only with its own successful prefill cache.
"""

import math

import torch
from torch import nn

from .abstractor_plugin import AbstractorLayer, Controller as VisualController


class AbstractorMemory(nn.Module):
    """Build symbol-valued relational memory and return a language correction."""

    def __init__(self, input_dim, width=256, heads=4, depth=2, *, max_tiles=16,
                 max_rows=64, max_cols=64):
        super().__init__()
        if input_dim < 1 or width < 1 or heads < 1 or width % heads or depth < 1:
            raise ValueError('Invalid memory dimensions, head count, or depth')
        self.spec = dict(
            architecture='original_rca_language_memory_v1', input_dim=input_dim,
            width=width, heads=heads, depth=depth, max_tiles=max_tiles,
            max_rows=max_rows, max_cols=max_cols,
            symbols='tile+projected_row+projected_column', token_pooling=False,
            internal_dtype='float32', gate='sigmoid(learned_scalar)',
            output_initialization='zero', symbol_self_attention=False,
            memory_values='address_symbols_then_abstract_states',
            reader='language_Q_relational_memory_KV',
            reader_positions='strictly_after_final_image_token')
        self.content_in = nn.Sequential(nn.LayerNorm(input_dim), nn.Linear(input_dim, width))
        self.tile_symbols = nn.Embedding(max_tiles, width)
        self.row_symbols = nn.Embedding(max_rows, width)
        self.col_symbols = nn.Embedding(max_cols, width)
        for embedding in (self.tile_symbols, self.row_symbols, self.col_symbols):
            nn.init.normal_(embedding.weight, std=0.02)
        self.layers = nn.ModuleList([AbstractorLayer(width, heads) for _ in range(depth)])
        self.reader_query = nn.Sequential(nn.LayerNorm(input_dim), nn.Linear(input_dim, width))
        self.memory_norm = nn.LayerNorm(width)
        self.reader_key = nn.Linear(width, width, bias=False)
        self.reader_value = nn.Linear(width, width, bias=False)
        self.reader_out = nn.Linear(width, width, bias=False)
        self.output_norm = nn.LayerNorm(width)
        self.output = nn.Linear(width, input_dim, bias=False)
        nn.init.zeros_(self.output.weight)
        self.gate = nn.Parameter(torch.zeros(()))
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
        result = dict(self.last_diagnostics)
        if include_gradients:
            result['gradient_norms'] = {
                name: None if parameter.grad is None else float(parameter.grad.detach().float().norm().cpu())
                for name, parameter in self.named_parameters()
            }
        return result

    def _validate_content(self, content):
        if content.ndim != 3 or content.shape[0] != 1 or content.shape[-1] != self.spec['input_dim']:
            raise ValueError('Memory supports [1, tokens, input_dim] hidden states only')
        if self.output.weight.dtype != torch.float32:
            raise ValueError('Memory parameters must remain FP32; install after casting the native model')

    def build_memory(self, visual_content, addresses):
        """Visual content affects Q/K only; no language/answer input is accepted."""
        self._validate_content(visual_content)
        if visual_content.shape[1] < 1:
            raise ValueError('Relational memory requires at least one native image token')
        if addresses.shape != (visual_content.shape[1], 3) or addresses.dtype != torch.long:
            raise ValueError('Expected one integer [tile,row,column] address per visual token')
        addresses = addresses.to(device=visual_content.device)
        with torch.autocast(device_type=visual_content.device.type, enabled=False):
            content = self.content_in(visual_content.float())
            states = (self.tile_symbols(addresses[:, 0]) + self.row_symbols(addresses[:, 1])
                      + self.col_symbols(addresses[:, 2])).unsqueeze(0)
            for layer in self.layers:
                states = layer(content, states)
            if self.capture:
                with torch.no_grad():
                    entropy = [layer.relation.last_diagnostics['attention_entropy'] for layer in self.layers]
                    self.last_diagnostics.update(
                        visual_tokens=visual_content.shape[1],
                        memory_tokens=states.shape[1],
                        abstract_rms=float(states.detach().square().mean().sqrt().cpu()),
                        memory_attention_entropy=entropy,
                        attention_entropy=entropy)
        return states

    def forward(self, language_queries, memory):
        """Return only the additive correction, allowing an external scale hook."""
        self._validate_content(language_queries)
        if (memory.ndim != 3 or memory.shape[0] != 1 or memory.shape[1] < 1
                or memory.shape[2] != self.spec['width']):
            raise ValueError('Expected [1, image_tokens, width] relational memory')
        if language_queries.shape[1] < 1 or memory.device != language_queries.device:
            raise ValueError('At least one language query and colocated memory are required')
        heads, width = self.spec['heads'], self.spec['width']

        def split(value):
            return value.reshape(1, value.shape[1], heads, width // heads).transpose(1, 2)

        with torch.autocast(device_type=language_queries.device.type, enabled=False):
            query = split(self.reader_query(language_queries.float()))
            normalized = self.memory_norm(memory.float())
            key = split(self.reader_key(normalized))
            value = split(self.reader_value(normalized))
            weights = (query @ key.transpose(-1, -2) / math.sqrt(width // heads)).softmax(-1)
            readout = (weights @ value).transpose(1, 2).reshape(1, language_queries.shape[1], width)
            delta = self.gate.sigmoid() * self.output(self.output_norm(self.reader_out(readout)))
            if self.capture:
                with torch.no_grad():
                    input_rms = language_queries.detach().float().square().mean().sqrt()
                    delta_rms = delta.detach().square().mean().sqrt()
                    probabilities = weights.detach()
                    entropy = -(probabilities * probabilities.clamp_min(1e-30).log()).sum(-1).mean()
                    self.last_diagnostics.update(
                        query_tokens=language_queries.shape[1], memory_tokens=memory.shape[1],
                        gate=float(self.gate.detach().sigmoid().cpu()), input_rms=float(input_rms.cpu()),
                        delta_rms=float(delta_rms.cpu()),
                        delta_to_input_rms=float((delta_rms / input_rms.clamp_min(1e-12)).cpu()),
                        reader_attention_entropy=float(entropy.cpu()))
        return delta.to(dtype=language_queries.dtype)


class Controller(VisualController):
    """Native hooks with cache-bound relational memory and exact bypass."""

    def __init__(self, base, plugin, layer_index):
        self._memory = None
        self._cache = None
        self._cached_length = None
        self._pending_memory = None
        self._decode_applied = False
        super().__init__(base, plugin, layer_index)
        self.spec.update(insertion='self_attn_output_parallel_language_memory',
                         cached_memory='detached_bound_to_exact_prefill_cache',
                         original_visual_outputs='unchanged')

    def _invalidate_memory(self):
        self._memory = None
        self._cache = None
        self._cached_length = None
        self._pending_memory = None
        self._has_prefill = False

    def set_enabled(self, enabled):
        self._invalidate_memory()
        super().set_enabled(enabled)

    def _before(self, module, args, kwargs):
        self._decode_applied = False
        self._pending_memory = None
        if not self.enabled or kwargs.get('pixel_values') is not None:
            self._invalidate_memory()
        # Reuse native shape/address, padding, checkpointing and prefill checks.
        super()._before(module, args, kwargs)
        if not self.enabled or self._context is not None:
            return
        cache = kwargs.get('past_key_values')
        if self._memory is None or cache is not self._cache:
            raise ValueError('Cached continuation must use the exact cache associated with relational memory')
        if torch.is_grad_enabled():
            raise ValueError('Cached memory decoding is inference-only; train with use_cache=False')
        if kwargs.get('use_cache') is False:
            raise ValueError('Cached memory continuation requires use_cache=True')
        if int(cache.get_seq_length()) != self._cached_length:
            raise ValueError('Cached prefix length changed outside the relational-memory forward')
        mask = kwargs.get('attention_mask')
        if mask is not None and (tuple(mask.shape) != (1, self._cached_length + 1)
                                 or not bool((mask == 1).all())):
            raise ValueError('Cached memory requires an unpadded full-prefix attention mask')
        cache_position = kwargs.get('cache_position')
        if cache_position is not None and (cache_position.numel() != 1
                                           or int(cache_position.item()) != self._cached_length):
            raise ValueError('Cached continuation position disagrees with relational-memory prefix')

    def _attention(self, module, args, kwargs, result):
        if not self.enabled:
            return result
        if not self._active:
            raise ValueError('Self-attention was invoked outside the audited native forward')
        context = self._context
        if (context is not None and context.applied) or self._decode_applied:
            raise ValueError('The selected self-attention block was invoked more than once')
        hidden = kwargs.get('hidden_states', args[0] if args else None)
        length = context.sequence_length if context is not None else 1
        if not isinstance(hidden, torch.Tensor) or tuple(hidden.shape) != (1, length, self.spec['input_dim']):
            raise ValueError('Self-attention hidden-state shape disagrees with input_ids')
        if not isinstance(result, tuple) or not result or not isinstance(result[0], torch.Tensor) or result[0].shape != hidden.shape:
            raise ValueError('Expected native self-attention (output, ...) tuple with unchanged token shape')
        if context is not None:
            positions = context.positions.to(hidden.device)
            memory = self.plugin.build_memory(hidden.index_select(1, positions), context.addresses)
            self._pending_memory = memory
            # Earlier text/markers must not see images that are in their future.
            readers = torch.arange(int(positions[-1]) + 1, length, device=hidden.device)
            context.applied = True
        else:
            memory = self._memory
            readers = torch.arange(length, device=hidden.device)
            self._decode_applied = True
        if readers.numel() == 0:
            return result
        delta = self.plugin(hidden.index_select(1, readers), memory)
        native = result[0]
        updated = native.clone()
        updated[:, readers, :] = native.index_select(1, readers) + delta.to(dtype=native.dtype)
        return (updated, *result[1:])

    @staticmethod
    def _output_cache(result):
        if hasattr(result, 'past_key_values'):
            return result.past_key_values
        if isinstance(result, tuple):
            return next((item for item in result if hasattr(item, 'get_seq_length')), None)
        return None

    def _after(self, module, args, kwargs, result):
        context = self._context
        missing = self.enabled and result is not None and (
            (context is not None and not context.applied) or (context is None and not self._decode_applied))
        invalid_cache = False
        if not self.enabled or result is None or missing:
            self._invalidate_memory()
            if result is None or missing:
                self.plugin.clear_diagnostics()
        else:
            cache = self._output_cache(result)
            if cache is None:
                self._invalidate_memory()
            elif context is not None:
                # The forward result retains its graph; lifecycle state never does.
                self._memory = self._pending_memory.detach()
                self._cache = cache
                self._cached_length = int(cache.get_seq_length())
                self._has_prefill = True
            elif cache is self._cache:
                self._cached_length = int(cache.get_seq_length())
            else:
                self._invalidate_memory()
                invalid_cache = True
        self._pending_memory = None
        self._context = None
        self._active = False
        self._decode_applied = False
        if missing:
            raise ValueError('Native forward did not invoke the selected relational-memory hook')
        if invalid_cache:
            raise ValueError('Native decoder replaced the cache associated with relational memory')

    def remove(self):
        self._invalidate_memory()
        super().remove()


def install(base, processor=None, layer_index=2, width=256, heads=4, depth=2):
    """Install after native casting; preserve all original weights/trainability."""
    del processor
    if hasattr(base, 'abstractor_plugin') or hasattr(base, '_abstractor_plugin_controller'):
        raise ValueError('Abstractor plugin or memory already installed')
    if getattr(base.config, 'model_type', None) != 'lfm2_vl':
        raise ValueError('Only audited native LFM2-VL models are supported')
    layer_types = base.config.text_config.layer_types
    if not isinstance(layer_index, int) or layer_index < 0 or layer_index >= len(layer_types):
        raise ValueError('Invalid insertion layer index')
    if layer_types[layer_index] != 'full_attention' or layer_index != layer_types.index('full_attention'):
        raise ValueError('Memory must attach to the first native full_attention layer')
    layer = base.model.language_model.layers[layer_index]
    if not isinstance(getattr(layer, 'self_attn', None), nn.Module):
        raise ValueError('Selected native layer has no self_attn module')
    plugin = AbstractorMemory(base.config.text_config.hidden_size, width, heads, depth)
    plugin.to(device=next(base.parameters()).device, dtype=torch.float32)
    base.add_module('abstractor_plugin', plugin)
    try:
        controller = Controller(base, plugin, layer_index)
    except Exception:
        delattr(base, 'abstractor_plugin')
        raise
    base._abstractor_plugin_controller = controller
    return controller
