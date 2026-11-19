"""Slot attention over frozen VLM patch features (DINOSAUR-style) + feature-reconstruction decoder (Stage 0).

Follows the Slot Abstractors paper (Mondal, Cohen, Webb 2024) for the object-centric part: slots drawn from a learned
Gaussian, T iterations with softmax over slots (slots compete), GRU + residual MLP; a spatial-broadcast decoder
reconstructs the input. After training, per-slot feature embeddings z_k and position embeddings m_k are attention-
weighted means of the (pre-position) features and of the position code (paper Eqs. 1-2): the Abstractor's inputs.
Input here is one panel's P x P grid of ViT features (ROI-sampled from the image), so the decoder reconstructs
ViT features instead of pixels (the paper's V-PROM / DINOSAUR variant).

Feature normalisation (feat_norm="channel", default): every ViT channel is standardised with dataset statistics
measured at the start of Stage 0 and stored in the checkpoint. Qwen3-VL's ViT has two channels (201, 62) with mean
|activation| ~1000 vs a median of ~5; after per-token LayerNorm they dominate every token, only ~1% of a panel's
energy is spatial, and a constant (panel-mean) reconstruction already reaches the training MSE, so slots never
specialise (measured by tools/feature_stats.py). Per-channel standardisation leaves ~90% of the energy spatial.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class SoftPositionEmbed(nn.Module):
    """Linear map of [x, 1-x, y, 1-y] (paper Sec. 2.1); (h*w, dim) in raster order."""

    def __init__(self, dim):
        super().__init__()
        self.proj = nn.Linear(4, dim)

    def forward(self, h, w, device=None):
        ys = torch.linspace(0.0, 1.0, h, device=device)
        xs = torch.linspace(0.0, 1.0, w, device=device)
        gy, gx = torch.meshgrid(ys, xs, indexing="ij")
        return self.proj(torch.stack([gx, 1 - gx, gy, 1 - gy], -1).reshape(h * w, 4))


class SlotAttention(nn.Module):
    def __init__(self, num_slots, dim, iters=3, eps=1e-8):
        super().__init__()
        self.num_slots, self.dim, self.iters, self.eps = num_slots, dim, iters, eps
        self.scale = dim ** -0.5
        self.slots_mu = nn.Parameter(torch.randn(1, 1, dim))
        self.slots_logsigma = nn.Parameter(torch.zeros(1, 1, dim))
        nn.init.xavier_uniform_(self.slots_logsigma)
        self.to_q = nn.Linear(dim, dim, bias=False)
        self.to_k = nn.Linear(dim, dim, bias=False)
        self.to_v = nn.Linear(dim, dim, bias=False)
        self.gru = nn.GRUCell(dim, dim)
        self.mlp = nn.Sequential(nn.Linear(dim, 2 * dim), nn.ReLU(inplace=True), nn.Linear(2 * dim, dim))
        self.norm_in, self.norm_slots, self.norm_mlp = nn.LayerNorm(dim), nn.LayerNorm(dim), nn.LayerNorm(dim)

    def forward(self, inputs, generator=None, noise=None):
        """inputs (B, N, D) -> slots (B, K, D), attn_w (B, K, N) (renormalised over N), attn_soft (B, K, N) (softmax
        over slots = per-position masks). Always fp32 (slot attention is touchy under bf16).
        noise (B, K, D): slot-init noise; drawn here if None."""
        with torch.autocast(device_type=inputs.device.type, enabled=False):
            x = inputs.float()
            B, N, D = x.shape
            mu = self.slots_mu.expand(B, self.num_slots, -1)
            if noise is None:
                noise = torch.randn(mu.shape, device=x.device, generator=generator)
            slots = mu + self.slots_logsigma.exp() * noise
            x = self.norm_in(x)
            k, v = self.to_k(x), self.to_v(x)
            for _ in range(self.iters):
                prev = slots
                q = self.to_q(self.norm_slots(slots))
                attn_soft = torch.einsum("bkd,bnd->bkn", q, k).mul(self.scale).softmax(dim=1) + self.eps
                attn_w = attn_soft / attn_soft.sum(-1, keepdim=True)
                updates = torch.einsum("bkn,bnd->bkd", attn_w, v)
                slots = self.gru(updates.reshape(-1, D), prev.reshape(-1, D)).reshape(B, -1, D)
                slots = slots + self.mlp(self.norm_mlp(slots))
        return slots, attn_w, attn_soft


class SpatialBroadcastDecoder(nn.Module):
    def __init__(self, slot_dim, out_dim, hidden=512, layers=3):
        super().__init__()
        mods, d = [], slot_dim
        for _ in range(layers - 1):
            mods += [nn.Linear(d, hidden), nn.ReLU(inplace=True)]
            d = hidden
        mods.append(nn.Linear(d, out_dim + 1))
        self.mlp, self.out_dim = nn.Sequential(*mods), out_dim

    def forward(self, slots, pos):
        x = self.mlp(slots[:, :, None, :] + pos[None, None])          # (B, K, N, out+1)
        alpha = x[..., self.out_dim].float().softmax(dim=1)             # masks, softmax over slots
        recon = (x[..., : self.out_dim].float() * alpha.unsqueeze(-1)).sum(1)
        return recon, alpha


class SlotAutoencoder(nn.Module):
    """Stage 0 model. `extract()` is what Stage 1 calls (frozen)."""

    def __init__(self, in_dim, slot_dim=32, num_slots=16, iters=3, decoder_hidden=512, decoder_layers=3, grid=12,
                 feat_norm="channel"):
        super().__init__()
        assert feat_norm in ("channel", "layernorm")
        self.cfg = dict(in_dim=in_dim, slot_dim=slot_dim, num_slots=num_slots, iters=iters,
                        decoder_hidden=decoder_hidden, decoder_layers=decoder_layers, grid=grid, feat_norm=feat_norm)
        self.grid, self.feat_norm = grid, feat_norm
        self.register_buffer("feat_mean", torch.zeros(in_dim))   # set by set_feature_stats() before Stage 0 training
        self.register_buffer("feat_std", torch.ones(in_dim))
        self.in_norm = nn.LayerNorm(in_dim)
        self.in_proj = nn.Linear(in_dim, slot_dim)
        self.pos = SoftPositionEmbed(slot_dim)
        self.enc_norm = nn.LayerNorm(slot_dim)
        self.enc_mlp = nn.Sequential(nn.Linear(slot_dim, slot_dim), nn.ReLU(inplace=True), nn.Linear(slot_dim, slot_dim))
        self.slot_attn = SlotAttention(num_slots, slot_dim, iters)
        self.dec_pos = SoftPositionEmbed(slot_dim)
        self.decoder = SpatialBroadcastDecoder(slot_dim, in_dim, decoder_hidden, decoder_layers)

    def set_feature_stats(self, mean, std):
        self.feat_mean.copy_(mean)
        self.feat_std.copy_(std.clamp_min(1e-4))

    def standardize(self, feats):
        x = feats.float()
        if self.feat_norm == "channel":
            return (x - self.feat_mean) / self.feat_std
        return F.layer_norm(x, x.shape[-1:])

    def _embed(self, feats):
        feat = self.in_proj(self.in_norm(self.standardize(feats)))
        pos = self.pos(self.grid, self.grid, device=feats.device)
        return feat, pos, self.enc_mlp(self.enc_norm(feat + pos))

    def extract(self, feats, generator=None, noise=None):
        """feats (n, P*P, in_dim) -> z, m (n, K, D) and attn (n, K, P*P) slot masks (softmax over slots)."""
        feat, pos, inputs = self._embed(feats)
        _, attn_w, attn_soft = self.slot_attn(inputs, generator, noise)
        z = torch.einsum("bkn,bnd->bkd", attn_w, feat.float())
        m = torch.einsum("bkn,nd->bkd", attn_w, pos.float())
        return z, m, attn_soft

    def forward(self, feats):
        _, _, inputs = self._embed(feats)
        slots, _, _ = self.slot_attn(inputs)
        recon, alpha = self.decoder(slots, self.dec_pos(self.grid, self.grid, device=feats.device))
        return F.mse_loss(recon, self.standardize(feats).detach()), alpha


def attn_diagnostics(attn):
    """attn (n, K, N) softmax over slots -> (mean entropy, ln K, #slots winning >= 2% of positions).
    Entropy near ln K means the slots have NOT specialised (the known plateau)."""
    K = attn.shape[1]
    ent = -(attn.clamp_min(1e-9).log() * attn).sum(1).mean().item()
    win = torch.bincount(attn.argmax(1).reshape(-1), minlength=K).float() / attn[:, 0].numel()
    return ent, math.log(K), int((win >= 0.02).sum().item())


def load_slot_ae(path, device):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    ae = SlotAutoencoder(**ck["config"])
    ae.load_state_dict(ck["model"])
    ae.to(device).eval()
    for p in ae.parameters():
        p.requires_grad_(False)
    return ae, ck
