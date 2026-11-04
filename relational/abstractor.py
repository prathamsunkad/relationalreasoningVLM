"""Abstractor with relational cross-attention (Slot Abstractors, Algorithm 1 / Eqs. 3-5).

    s^0 = m + panel embedding
    per layer:  s = RCA(z, s) + s ; s = FF(s) + s ; s = SelfAttn(s) + s ; s = FF(s) + s
RCA takes queries AND keys from the slot feature embeddings z but values from s (position codes at layer 0): appearance
only decides who attends to whom, and what flows forward is relational. Wq and Wk start equal (paper Sec. 4.3).

Our additions (not in the paper text): pre-LayerNorm sub-layers + final LayerNorm; a learned panel-index embedding
(the paper's position code only locates a slot *within* its panel); a padding mask so sequences with different panel
counts (PGM 9 cells, a single natural image = 1 panel) share one batch.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


def _ff(dim, hidden, dropout):
    return nn.Sequential(nn.Linear(dim, hidden), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden, dim))


class _Attn(nn.Module):
    def __init__(self, dim, heads, head_dim, dropout, relational, tie_qk_init=True):
        super().__init__()
        inner = heads * head_dim
        self.heads, self.head_dim, self.dropout, self.relational = heads, head_dim, dropout, relational
        self.q = nn.Linear(dim, inner, bias=False)
        self.k = nn.Linear(dim, inner, bias=False)
        self.v = nn.Linear(dim, inner, bias=False)
        self.o = nn.Linear(inner, dim)
        if relational and tie_qk_init:
            with torch.no_grad():
                self.k.weight.copy_(self.q.weight)

    def forward(self, z, s, mask):
        B, S, _ = s.shape
        src = z if self.relational else s
        sh = lambda t: t.view(B, S, self.heads, self.head_dim).transpose(1, 2)
        out = F.scaled_dot_product_attention(sh(self.q(src)), sh(self.k(src)), sh(self.v(s)), attn_mask=mask,
                                             dropout_p=self.dropout if self.training else 0.0)
        return self.o(out.transpose(1, 2).reshape(B, S, -1))


class AbstractorLayer(nn.Module):
    def __init__(self, dim, heads, head_dim, mlp_dim, dropout=0.0):
        super().__init__()
        self.n1, self.n2, self.n3, self.n4 = (nn.LayerNorm(dim) for _ in range(4))
        self.rca = _Attn(dim, heads, head_dim, dropout, relational=True)
        self.ff1 = _ff(dim, mlp_dim, dropout)
        self.sa = _Attn(dim, heads, head_dim, dropout, relational=False)
        self.ff2 = _ff(dim, mlp_dim, dropout)

    def forward(self, z, s, mask):
        s = s + self.rca(z, self.n1(s), mask)
        s = s + self.ff1(self.n2(s))
        s = s + self.sa(None, self.n3(s), mask)
        return s + self.ff2(self.n4(s))


class RelationalAbstractor(nn.Module):
    """Slots of the panels of one sequence -> relation embeddings per slot + one pooled relation vector.

    z, m: (n_seq, P_max, K, D); panel_mask: (n_seq, P_max) bool (True = real panel).
    Returns s (n_seq, P_max*K, D) and r (n_seq, D) = mean of s over real slots (the paper's pooled embedding).
    """

    def __init__(self, dim=32, heads=8, head_dim=32, layers=24, mlp_dim=512, dropout=0.0, max_panels=16):
        super().__init__()
        self.cfg = dict(dim=dim, heads=heads, head_dim=head_dim, layers=layers, mlp_dim=mlp_dim, dropout=dropout,
                        max_panels=max_panels)
        self.z_norm = nn.LayerNorm(dim)
        self.panel_emb = nn.Embedding(max_panels, dim)
        nn.init.normal_(self.panel_emb.weight, std=0.02)
        self.layers = nn.ModuleList([AbstractorLayer(dim, heads, head_dim, mlp_dim, dropout) for _ in range(layers)])
        self.out_norm = nn.LayerNorm(dim)

    def forward(self, z, m, panel_mask):
        n, P, K, D = z.shape
        s = m + self.panel_emb(torch.arange(P, device=z.device))[None, :, None, :]
        z = self.z_norm(z.reshape(n, P * K, D))
        s = s.reshape(n, P * K, D)
        tok_mask = panel_mask[:, :, None].expand(n, P, K).reshape(n, P * K)
        attn_mask = tok_mask[:, None, None, :]                       # keys only; padded queries are ignored below
        for layer in self.layers:
            s = layer(z, s, attn_mask)
        s = self.out_norm(s)
        w = tok_mask.float().unsqueeze(-1)
        r = (s * w).sum(1) / w.sum(1).clamp_min(1.0)
        return s, r
