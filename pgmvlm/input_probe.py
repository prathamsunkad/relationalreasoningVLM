"""Controlled input-representation screen for a frozen LFM vision encoder.

This is a direct eight-class readout diagnostic, not a hybrid VLM benchmark.
Caching pools *non-affine* LayerNorm features. The learned affine and linear
projection commute with fixed normalized pooling, preserving the old sheet arm.
"""
from __future__ import annotations

import copy
import torch
from torch import nn
from torch.nn import functional as F

from .extensions.relational import ordered_pool
from .extensions.probe_diagnostics import make_arms

ARMS = ("sheet_flat", "panels_bands", "panels_grid", "panels_grid_context", "grid_context_classifier")
CACHE_KEYS = ("sheet_flat", "panels_bands", "panels_grid")


def normalize_tokens(x):
    return F.layer_norm(x.float(), (x.shape[-1],), eps=1e-5)


def grid_pool(x, height, width, rows=2, cols=4):
    """Nonoverlapping spatial cells, raster order; never cross panel boundaries."""
    if x.ndim != 2 or len(x) != height * width or height < rows or width < cols:
        raise ValueError("Invalid feature grid")
    y = torch.arange(height, device=x.device) * rows // height
    z = torch.arange(width, device=x.device) * cols // width
    cell = (y[:, None] * cols + z[None, :]).flatten()
    weights = (cell[None] == torch.arange(rows * cols, device=x.device)[:, None]).float()
    return weights.div(weights.sum(-1, keepdim=True)) @ x.float()


def cache_representations(sheet_tiles, panel_features, panel_grids):
    if len(panel_features) != 16 or len(panel_grids) != 16:
        raise ValueError("Require exactly 8 context and 8 candidate panels")
    if len({tuple(g) for g in panel_grids}) != 1:
        raise ValueError("Equal-sized original panels must have equal feature grids")
    sheet = ordered_pool(normalize_tokens(torch.cat(sheet_tiles)).unsqueeze(0), 128)[0]
    panels = [normalize_tokens(x) for x in panel_features]
    bands = torch.cat([ordered_pool(x.unsqueeze(0), 8)[0] for x in panels])
    grid = torch.cat([grid_pool(x, *shape) for x, shape in zip(panels, panel_grids)])
    return dict(sheet_flat=sheet, panels_bands=bands, panels_grid=grid)


def context_normalize(x, minimum_std=1e-4):
    """Per-question/channel normalization over tokens; never fits on validation.

This is analogous to upstream context normalization, but uses our fixed grid
features over all 16 panels, not upstream's nine-panel candidate/slot sequence.
"""
    return (x - x.mean(-2, keepdim=True)) / x.std(-2, unbiased=True, keepdim=True).clamp_min(minimum_std)


class PooledAffine(nn.Module):
    def __init__(self, layer_norm):
        super().__init__()
        self.weight = nn.Parameter(layer_norm.weight.detach().clone())
        self.bias = nn.Parameter(layer_norm.bias.detach().clone())

    def forward(self, normalized_pool):
        return normalized_pool * self.weight + self.bias


class InputProbe(nn.Module):
    def __init__(self, source, use_context=False):
        super().__init__()
        self.arm = copy.deepcopy(source)
        self.use_context = use_context
        self.classifier_only = source.name == "classifier_only"
        projection = self.arm.content_in if self.classifier_only else self.arm.probe.content_in
        projection[0] = PooledAffine(projection[0])

    def forward(self, cached):
        x = context_normalize(cached.float()) if self.use_context else cached.float()
        if self.classifier_only:
            representation = self.arm.content_in(x)
        else:
            probe = self.arm.probe
            content = probe.content_in(x)
            representation = probe.symbols.expand(x.shape[0], -1, -1)
            for layer in probe.layers:
                representation = layer(content, representation)
        return self.arm.head(representation)


def make_input_arms(seed=20261118, input_dim=2048, width=256, bins=128):
    originals = make_arms(seed, input_dim=input_dim, width=width, bins=bins)
    arms = {name: InputProbe(originals["classifier_d2"], name == "panels_grid_context")
            for name in ARMS[:-1]}
    arms[ARMS[-1]] = InputProbe(originals["classifier_only"], True)
    return arms


def cache_key(arm):
    return arm if arm in CACHE_KEYS else "panels_grid"


def feature_statistics(x):
    x = x.float()
    normalized = F.normalize(x, dim=-1)
    similarity = normalized @ normalized.transpose(-1, -2)
    n = x.shape[-2]
    off_diagonal = ~torch.eye(n, dtype=torch.bool, device=x.device)
    variance = (x - x.mean(-2, keepdim=True)).square().mean()
    return dict(rms=float(x.square().mean().sqrt()),
                token_variation_energy_fraction=float(variance / x.square().mean().clamp_min(1e-20)),
                mean_token_cosine=float(similarity[..., off_diagonal].mean()),
                low_variance_channel_fraction=float((x.std(-2) < 1e-4).float().mean()))
