"""Scientific invariants for the panel input diagnostic."""
import copy
import unittest
import numpy as np
import torch
from torch.nn import functional as F
from pgmvlm.input_probe import (ARMS, InputProbe, cache_representations, context_normalize,
                              grid_pool, make_input_arms, normalize_tokens)
from pgmvlm.extensions.relational import ordered_pool
from pgmvlm.extensions.probe_diagnostics import make_arms
from pgmvlm.manifests import render_pgm
from pgmvlm.panels import split_composite


class InputProbeTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(13)

    def test_panel_pixels_and_order(self):
        panels = np.random.default_rng(8).integers(0, 256, (16, 160, 160), dtype=np.uint8)
        recovered = split_composite(render_pgm(panels))
        for original, image in zip(panels, recovered):
            np.testing.assert_array_equal(original, np.asarray(image)[..., 0])

    def test_grid_preserves_spatial_regions(self):
        x = torch.arange(8).reshape(2, 4).repeat_interleave(4, 0).repeat_interleave(2, 1)
        result = grid_pool(x.reshape(-1, 1), 8, 8)
        torch.testing.assert_close(result[:, 0], torch.arange(8).float())

    def test_panel_identity_never_mixed(self):
        feats = [torch.randn(64, 8) for _ in range(16)]
        cached = cache_representations(feats[:2], feats, [(8, 8)] * 16)
        self.assertEqual(cached['panels_grid'].shape, (128, 8))
        for i, feature in enumerate(feats):
            torch.testing.assert_close(cached['panels_grid'][8*i:8*i+8],
                                       grid_pool(normalize_tokens(feature), 8, 8))

    def test_cached_sheet_preserves_logits_gradients_and_update(self):
        original = make_arms(9, input_dim=8, width=8, bins=128)['classifier_d2']
        with torch.no_grad():
            original.probe.content_in[0].weight.uniform_(0.7, 1.3)
            original.probe.content_in[0].bias.uniform_(-0.2, 0.2)
        cached_model = InputProbe(original)
        x = torch.randn(2, 257, 8)
        pooled = ordered_pool(normalize_tokens(x), 128)
        torch.testing.assert_close(original(x), cached_model(pooled), atol=1e-6, rtol=1e-5)
        y = torch.tensor([2, 7])
        opt_a = torch.optim.AdamW([p for p in original.parameters() if p.requires_grad], lr=1e-4)
        opt_b = torch.optim.AdamW([p for p in cached_model.parameters() if p.requires_grad], lr=1e-4)
        F.cross_entropy(original(x), y).backward()
        F.cross_entropy(cached_model(pooled), y).backward()
        for name, p in original.named_parameters():
            q = dict(cached_model.arm.named_parameters())[name]
            if p.grad is not None:
                torch.testing.assert_close(p.grad, q.grad, atol=1e-7, rtol=2e-4)
        opt_a.step(); opt_b.step()
        torch.testing.assert_close(original(x), cached_model(pooled), atol=5e-6, rtol=1e-4)

    def test_common_initialization_and_context_invariance(self):
        arms = make_input_arms(input_dim=8, width=8)
        initial = arms[ARMS[0]].state_dict()
        for name in ARMS[1:4]:
            self.assertTrue(all(torch.equal(v, arms[name].state_dict()[k]) for k, v in initial.items()))
        x = torch.randn(2, 128, 8)
        scale, offset = torch.rand(8) + 0.5, torch.randn(8)
        torch.testing.assert_close(context_normalize(x), context_normalize(x * scale + offset),
                                   atol=1e-6, rtol=1e-5)
        self.assertTrue(torch.isfinite(context_normalize(torch.ones_like(x))).all())

    def test_all_arms_receive_finite_gradients(self):
        for arm in make_input_arms(input_dim=8, width=8).values():
            F.cross_entropy(arm(torch.randn(2, 128, 8)), torch.tensor([1, 5])).backward()
            grads = [p.grad for p in arm.parameters() if p.requires_grad]
            self.assertTrue(all(g is not None and torch.isfinite(g).all() for g in grads))
            self.assertGreater(sum(g.abs().sum().item() for g in grads), 0)


if __name__ == '__main__':
    unittest.main()
