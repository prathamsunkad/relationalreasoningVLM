"""Repository interleaving with frozen, externally cached relational states.

The projection and hook are the same methods used by RelationalVLM. Only the
source of relations changes: a detached bank replaces the live vision branch.
"""
from contextlib import contextmanager

import torch
from torch import nn

from relvlm import RelationalVLM, _rms


class CachedInterleave(nn.Module):
    relation_tokens = RelationalVLM.relation_tokens
    _lm_hook = RelationalVLM._lm_hook

    def __init__(self, backbone, input_dim=32):
        super().__init__()
        object.__setattr__(self, 'bb', backbone)
        self.d = int(backbone.d_llm)
        self.input_dim = input_dim
        self.rel_mode, self.M = 'pool', 1
        self.branch_off, self.rel_ablate = False, None
        self.tok_proj = nn.Sequential(nn.Linear(input_dim, self.d), nn.GELU(), nn.Linear(self.d, self.d))
        self.tok_gain = nn.Parameter(torch.ones(()))
        emb = backbone.model.get_input_embeddings().weight
        self.register_buffer('emb_rms', _rms(emb.detach()).mean().reshape(()), persistent=False)
        self._ctx, self._res = None, None
        self._handle = backbone.lm.register_forward_pre_hook(self._lm_hook, with_kwargs=True)

    @contextmanager
    def question(self, bank, rel_mask):
        if self._ctx is not None:
            raise RuntimeError('Nested/stale relation context')
        if (bank.requires_grad or bank.dtype != torch.float32 or bank.ndim != 3
                or tuple(bank.shape[1:]) != (9, self.input_dim) or not torch.isfinite(bank).all()):
            raise ValueError('Expected finite detached FP32 candidate x 9 x 32 bank')
        if (rel_mask.dtype != torch.bool or rel_mask.ndim != 2
                or rel_mask.shape[0] != bank.shape[0] or not bool((rel_mask.sum(1) == 1).all())):
            raise ValueError('Exactly one relation placeholder is required per candidate')
        self._ctx = dict(llm=True, rel_mask=rel_mask)
        self._res = dict(r=bank.mean(1))
        try:
            yield self
        finally:
            self._ctx, self._res = None, None

    def remove(self):
        self._handle.remove()
        self._ctx, self._res = None, None
