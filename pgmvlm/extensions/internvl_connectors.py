"""Frozen InternVL3.5/Qwen3 connectors for the native candidate-scoring protocol.

Each completed matrix receives its own nine panel summaries. Candidate indices
are deliberately not embedded: reordering choices must only reorder scores.
No labels, teacher scores, or classifier weights enter a connector.
"""
import torch
from torch import nn
from torch.nn import functional as F

from pgmvlm.extensions.pretrained_connectors import Connector, Controller, bound


class CandidateConnector(Connector):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        del self.candidate
        self.spec.update(family='internvl_qwen3', memory_shape=[9, 32],
                         protocol='candidate_scoring', candidate_index_embedding=False)

    def memory(self, bank):
        if bank.ndim != 3 or tuple(bank.shape[1:]) != (9, 32) or bank.requires_grad:
            raise ValueError('Expected detached [candidate_batch, 9, 32] relations')
        panels = torch.arange(9, device=bank.device)
        return self.input(self.norm(bank.float())) + self.panel(panels)[None]


class InternVLController(Controller):
    """Same three interfaces as LFM, using native Qwen3 normalization and RoPE."""

    def prepare(self, model, args, kwargs):
        self.context = None
        if not self.enabled:
            return
        ids = kwargs.get('input_ids')
        if args or ids is None or ids.ndim != 2:
            raise ValueError('Expected keyword input_ids and unpadded candidate sequences')
        if kwargs.get('use_cache') is not False or kwargs.get('past_key_values') is not None:
            raise ValueError('Use full-prefix use_cache=False')
        if any(kwargs.get(k) is not None for k in ('labels', 'position_ids', 'inputs_embeds')):
            raise ValueError('Use native positions/embeddings and external candidate CE')
        mask = kwargs.get('attention_mask')
        if mask is not None and (mask.shape != ids.shape or not bool(mask.all())):
            raise ValueError('Padded candidate sequences are unsupported')
        if self.bank is None or self.bank.shape[0] != ids.shape[0]:
            raise ValueError('Supply one aligned relation bank per candidate sequence')
        positions = [(row == self.base.config.image_token_id).nonzero().flatten() for row in ids]
        if any(len(p) == 0 for p in positions):
            raise ValueError('Each candidate must contain image tokens')
        starts = [int(p[-1]) + 1 for p in positions]
        if len(set(starts)) != 1 or starts[0] >= ids.shape[1]:
            raise ValueError('Candidates must have equal image spans followed by the prompt')
        self.context = dict(start=starts[0], memory=self.module.memory(self.bank))

    def kv(self, attn, args, kwargs, result):
        if not self.enabled:
            return result
        from transformers.models.qwen3.modeling_qwen3 import apply_rotary_pos_emb, repeat_kv
        hidden = kwargs.get('hidden_states', args[0] if args else None)
        if hidden is None or kwargs.get('past_key_values') is not None:
            raise ValueError('Unsupported native Qwen3 attention invocation')
        shape = (*hidden.shape[:-1], -1, attn.head_dim)
        q = attn.q_norm(attn.q_proj(hidden).view(shape)).transpose(1, 2)
        k = attn.k_norm(attn.k_proj(hidden).view(shape)).transpose(1, 2)
        v = attn.v_proj(hidden).view(shape).transpose(1, 2)
        q, k = apply_rotary_pos_emb(q, k, *kwargs['position_embeddings'])
        k = repeat_kv(k, attn.num_key_value_groups)
        v = repeat_kv(v, attn.num_key_value_groups)
        start = self.context['start']
        q = q[:, :, start:]
        mask = kwargs.get('attention_mask')
        if mask is None:
            qi = torch.arange(start, hidden.shape[1], device=hidden.device)[:, None]
            ki = torch.arange(hidden.shape[1], device=hidden.device)[None, :]
            mask = torch.zeros((1, 1, q.shape[2], hidden.shape[1]), device=hidden.device, dtype=q.dtype)
            mask = mask.masked_fill(ki > qi, torch.finfo(q.dtype).min)
        else:
            mask = mask[:, :, start:, :k.shape[2]]
            if mask.dtype == torch.bool:
                mask = torch.zeros_like(mask, dtype=q.dtype).masked_fill(~mask, torch.finfo(q.dtype).min)
        spec = self.module.spec
        batch, count = self.context['memory'].shape[:2]
        mk, mv = self.module.kv(self.context['memory']).view(
            batch, count, 2, spec['kv_heads'], spec['head_dim']).unbind(2)
        mk = repeat_kv(mk.transpose(1, 2).to(q.dtype), attn.num_key_value_groups)
        mv = repeat_kv(mv.transpose(1, 2).to(q.dtype), attn.num_key_value_groups)
        native = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, dropout_p=0., scale=attn.scaling)
        joint = F.scaled_dot_product_attention(q, torch.cat((k, mk), 2), torch.cat((v, mv), 2),
            attn_mask=F.pad(mask, (0, count), value=0.), dropout_p=0., scale=attn.scaling)
        delta = (joint - native).transpose(1, 2).reshape(batch, q.shape[2], -1)
        delta = attn.o_proj(delta)
        if getattr(attn.o_proj, 'bias', None) is not None:
            delta = delta - attn.o_proj.bias
        delta = delta * torch.tanh(self.module.gate).to(delta.dtype)
        output = result[0]
        changed = torch.cat((output[:, :start], output[:, start:] + bound(
            delta, output[:, start:], self.module.cap)), 1)
        return (changed,) + result[1:]

    def insert(self, model, args, kwargs):
        if not self.enabled:
            return
        if args or kwargs.get('inputs_embeds') is None or kwargs.get('past_key_values') is not None:
            raise ValueError('Expected merged InternVL embeddings without cache')
        x, start = kwargs['inputs_embeds'], self.context['start']
        tokens = self.module.output(self.context['memory']).to(x.dtype)
        scale = x[:, :start].detach().float().square().mean((1, 2), keepdim=True).sqrt()
        rms = tokens.float().square().mean(-1, keepdim=True).clamp_min(1e-24).sqrt()
        tokens = tokens * (scale / rms).clamp(max=1).to(tokens.dtype)
        kw = dict(kwargs)
        kw['inputs_embeds'] = torch.cat((x[:, :start], tokens, x[:, start:]), 1)
        kw['input_ids'] = None
        kw['attention_mask'] = torch.ones(kw['inputs_embeds'].shape[:2], device=x.device, dtype=torch.long)
        kw['position_ids'] = None
        kw.pop('cache_position', None)
        return args, kw

    def strip(self, model, args, result):
        if not self.enabled:
            return result
        start = self.context['start']
        def strip(x):
            return torch.cat((x[:, :start], x[:, start + 9:]), 1)
        result.last_hidden_state = strip(result.last_hidden_state)
        if result.hidden_states is not None:
            result.hidden_states = tuple(strip(h) for h in result.hidden_states)
        return result


def install(base, design, layer_index=8, **options):
    cfg = base.config
    if cfg.model_type != 'internvl' or cfg.text_config.model_type != 'qwen3':
        raise ValueError('Requires native InternVL with Qwen3 language backbone')
    if any(p.requires_grad for p in base.parameters()):
        raise ValueError('Freeze the entire VLM and existing LoRA first')
    if any(hasattr(base, n) for n in ('pretrained_connector', 'staged_abstractor', 'relation_probe', 'abstractor_plugin')):
        raise ValueError('Remove existing extensions first')
    text = cfg.text_config
    if not 0 <= layer_index < text.num_hidden_layers:
        raise ValueError('Invalid layer index')
    if text.layer_types[layer_index] != 'full_attention':
        raise ValueError('Requires a full-attention layer')
    module = CandidateConnector(design, text.hidden_size, text.num_key_value_heads,
        text.head_dim, **options).to(device=next(base.parameters()).device, dtype=torch.float32)
    module.spec['layer_index'] = layer_index if design != 'tokens' else None
    base.add_module('pretrained_connector', module)
    return InternVLController(base, module, layer_index)
