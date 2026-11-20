"""Torch-only correctness tests; native HF integration lives in a separate file."""

import math
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest

import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pgmvlm.extensions.abstractor_plugin import AbstractorPlugin, RelationalCrossAttention, install, projected_addresses


class FakeAttention(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.projection = nn.Linear(width, width)
        self.extra = object()

    def forward(self, hidden_states, **kwargs):
        return self.projection(hidden_states), self.extra, 'native remainder'


class FakeLayer(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.self_attn = FakeAttention(width)


class FakeBase(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(model_type='lfm2_vl', image_token_id=30, downsample_factor=2,
                                      text_config=SimpleNamespace(hidden_size=12, layer_types=['conv', 'conv', 'full_attention']))
        self.embedding = nn.Embedding(64, 12)
        self.model = nn.Module()
        self.model.language_model = nn.Module()
        self.model.language_model.layers = nn.ModuleList([nn.Identity(), nn.Identity(), FakeLayer(12)])
        self.is_gradient_checkpointing = False

    def forward(self, input_ids, **kwargs):
        return self.model.language_model.layers[2].self_attn(hidden_states=self.embedding(input_ids))


def example():
    return dict(input_ids=torch.tensor([[1, 30, 30, 30, 30, 2, 30, 30, 30, 30, 3, 4, 5]]),
                attention_mask=torch.ones(1, 13, dtype=torch.long),
                pixel_values=torch.randn(2, 16, 3),
                pixel_attention_mask=torch.ones(2, 16, dtype=torch.long),
                spatial_shapes=torch.tensor([[4, 4], [4, 4]]))


class RelationalFormulaTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(12)

    def test_original_rca_formula_and_symbol_only_values(self):
        attention = RelationalCrossAttention(4, 2)
        with torch.no_grad():
            for projection in (attention.q, attention.k, attention.v, attention.out):
                projection.weight.copy_(torch.eye(4))
        content = torch.randn(1, 3, 4)
        symbols = torch.randn(1, 3, 4)
        q = content.reshape(1, 3, 2, 2).transpose(1, 2)
        v = symbols.reshape(1, 3, 2, 2).transpose(1, 2)
        weights = (q @ q.transpose(-1, -2) / math.sqrt(2)).softmax(-1)
        expected = (weights @ v).transpose(1, 2).reshape(1, 3, 4)
        torch.testing.assert_close(attention(content, symbols), expected, rtol=0, atol=0)
        torch.testing.assert_close(attention(content, torch.zeros_like(symbols)), torch.zeros_like(content), rtol=0, atol=0)
        self.assertFalse(torch.equal(attention(content, symbols), attention(content, symbols + 1)))

    def test_projected_geometry_order_and_count(self):
        shapes = torch.tensor([[4, 6], [2, 4]])
        patch_mask = torch.zeros(2, 24, dtype=torch.long)
        patch_mask[0] = 1
        patch_mask[1, :8] = 1
        addresses = projected_addresses(shapes, 2, image_token_count=8, pixel_attention_mask=patch_mask)
        self.assertEqual(addresses.tolist(), [[0, 0, 0], [0, 0, 1], [0, 0, 2],
                                             [0, 1, 0], [0, 1, 1], [0, 1, 2], [1, 0, 0], [1, 0, 1]])
        with self.assertRaisesRegex(ValueError, 'token count'):
            projected_addresses(shapes, image_token_count=7)
        with self.assertRaisesRegex(ValueError, 'divisible'):
            projected_addresses(torch.tensor([[3, 4]]))
        with self.assertRaisesRegex(ValueError, 'integer'):
            projected_addresses(shapes.float())
        with self.assertRaisesRegex(ValueError, 'tiles'):
            projected_addresses(shapes, max_tiles=1)
        with self.assertRaisesRegex(ValueError, 'address space'):
            projected_addresses(shapes, max_cols=2)
        patch_mask[1, 0], patch_mask[1, 9] = 0, 1
        with self.assertRaisesRegex(ValueError, 'leading valid'):
            projected_addresses(shapes, pixel_attention_mask=patch_mask)

    def test_zero_output_opens_gradients_after_update(self):
        plugin = AbstractorPlugin(12, width=8, heads=2, depth=2)
        content = torch.randn(1, 4, 12)
        addresses = projected_addresses(torch.tensor([[4, 4]]))
        target = torch.randn_like(content)
        first = plugin(content, addresses)
        self.assertTrue(torch.equal(first, torch.zeros_like(content)))
        optimizer = torch.optim.SGD(plugin.parameters(), lr=0.1)
        (first - target).square().mean().backward()
        self.assertGreater(float(plugin.output.weight.grad.norm()), 0)
        self.assertEqual(float(plugin.layers[0].relation.q.weight.grad.norm()), 0)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        (plugin(content, addresses) - target).square().mean().backward()
        self.assertGreater(float(plugin.layers[0].relation.q.weight.grad.norm()), 0)
        self.assertGreater(float(plugin.tile_symbols.weight.grad.norm()), 0)
        self.assertGreater(float(plugin.gate.grad.abs()), 0)

    def test_address_symbols_are_content_independent(self):
        plugin = AbstractorPlugin(12, width=8, heads=2, depth=2)
        captured_values = []
        captured_content = []
        def capture(module, args):
            captured_content.append(args[0].detach().clone())
            captured_values.append(args[1].detach().clone())
        hook = plugin.layers[0].relation.register_forward_pre_hook(capture)
        addresses = projected_addresses(torch.tensor([[4, 4]]))
        plugin(torch.randn(1, 4, 12), addresses)
        plugin(torch.randn(1, 4, 12), addresses)
        hook.remove()
        self.assertTrue(torch.equal(*captured_values))
        self.assertFalse(torch.equal(*captured_content))

    def test_state_dict_roundtrip_and_explicit_diagnostics(self):
        plugin = AbstractorPlugin(12, width=8, heads=2, depth=2)
        other = AbstractorPlugin(12, width=8, heads=2, depth=2)
        with torch.no_grad():
            plugin.output.weight.normal_(std=0.03)
        other.load_state_dict(plugin.state_dict(), strict=True)
        content = torch.randn(1, 4, 12)
        addresses = projected_addresses(torch.tensor([[4, 4]]))
        self.assertTrue(torch.equal(plugin(content, addresses), other(content, addresses)))
        self.assertEqual(plugin.diagnostics(), {})
        plugin.set_capture_diagnostics()
        plugin(content, addresses).sum().backward()
        diagnostics = plugin.diagnostics(include_gradients=True)
        self.assertEqual(diagnostics['visual_tokens'], 4)
        self.assertEqual(len(diagnostics['attention_entropy']), 2)
        self.assertIsInstance(diagnostics['delta_rms'], float)
        self.assertIsInstance(diagnostics['gradient_norms']['output.weight'], float)
        plugin.set_capture_diagnostics(False)
        self.assertEqual(plugin.diagnostics(), {})


class ControllerTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(18)
        self.base = FakeBase()
        self.inputs = example()

    def test_initial_exactness_disabled_path_and_native_preservation(self):
        native_state = {name: value.detach().clone() for name, value in self.base.state_dict().items()}
        trainability = {name: value.requires_grad for name, value in self.base.named_parameters()}
        reference = self.base(**self.inputs)
        controller = install(self.base, width=8, heads=2)
        result = self.base(**self.inputs)
        self.assertTrue(torch.equal(reference[0], result[0]))
        self.assertIs(reference[1], result[1])
        self.assertEqual(reference[2], result[2])
        with torch.no_grad():
            controller.plugin.output.weight.normal_()
        controller.enabled = False
        self.assertTrue(torch.equal(reference[0], self.base(**self.inputs)[0]))
        # Disabled is the exact native path, even for otherwise unsupported text-only batches.
        text_ids = torch.tensor([[1, 2], [3, 4]])
        expected = self.base.model.language_model.layers[2].self_attn(self.base.embedding(text_ids))[0]
        self.assertTrue(torch.equal(expected, self.base(text_ids)[0]))
        controller.remove()
        controller.remove()  # Idempotent teardown.
        self.assertFalse(hasattr(self.base, 'abstractor_plugin'))
        self.assertEqual(set(native_state), set(self.base.state_dict()))
        for name, value in self.base.state_dict().items():
            self.assertTrue(torch.equal(native_state[name], value))
        self.assertEqual(trainability, {name: value.requires_grad for name, value in self.base.named_parameters()})
        with self.assertRaises(RuntimeError):
            controller.enabled = True

    def test_visual_only_input_source_cached_decode_and_context_reset(self):
        native = self.base(**self.inputs)[0]
        controller = install(self.base, width=8, heads=2)
        with torch.no_grad():
            controller.plugin.output.weight.normal_(std=0.03)
        captured = []
        hook = controller.plugin.register_forward_pre_hook(lambda module, args: captured.append(args[0].detach().clone()))
        result = self.base(**self.inputs)[0]
        positions = self.inputs['input_ids'][0] == 30
        self.assertTrue(torch.equal(result[:, ~positions], native[:, ~positions]))
        self.assertFalse(torch.equal(result[:, positions], native[:, positions]))
        self.assertTrue(torch.equal(captured[0], self.base.embedding(self.inputs['input_ids'])[:, positions]))
        self.assertIsNone(controller._context)
        # Neither text positions nor cached next-token calls enter the plugin.
        decode_ids = torch.tensor([[6]])
        expected = self.base.model.language_model.layers[2].self_attn.projection(self.base.embedding(decode_ids))
        decoded = self.base(decode_ids, past_key_values=object())[0]
        self.assertTrue(torch.equal(expected, decoded))
        self.assertEqual(len(captured), 1)
        next_inputs = example()
        next_inputs.update(input_ids=torch.tensor([[30, 30, 30, 30, 1, 2]]),
                           attention_mask=torch.ones(1, 6, dtype=torch.long),
                           pixel_values=next_inputs['pixel_values'][:1],
                           pixel_attention_mask=next_inputs['pixel_attention_mask'][:1],
                           spatial_shapes=next_inputs['spatial_shapes'][:1])
        self.base(**next_inputs)
        self.assertEqual(captured[-1].shape[1], 4)
        hook.remove()
        controller.remove()

    def test_fail_closed_and_recover_after_invalid_request(self):
        controller = install(self.base, width=8, heads=2)
        malformed = dict(self.inputs, spatial_shapes=torch.tensor([[2, 4], [4, 4]]))
        with self.assertRaises(ValueError):
            self.base(**malformed)
        with self.assertRaisesRegex(ValueError, 'cached one-token'):
            self.base(torch.tensor([[1]]), past_key_values=object())
        self.base(**self.inputs)  # Failed prefill does not poison a fresh request.
        padded = dict(self.inputs, attention_mask=self.inputs['attention_mask'].clone())
        padded['attention_mask'][0, -1] = 0
        with self.assertRaisesRegex(ValueError, 'Padded'):
            self.base(**padded)
        with self.assertRaisesRegex(ValueError, 'cached one-token'):
            self.base(torch.tensor([[1]]), past_key_values=object())
        self.base.is_gradient_checkpointing = True
        with self.assertRaisesRegex(ValueError, 'checkpointing'):
            self.base(**self.inputs)
        self.base.is_gradient_checkpointing = False
        self.base(**self.inputs)
        with self.assertRaisesRegex(ValueError, 'already installed'):
            install(self.base)
        controller.remove()

    def test_no_stale_diagnostics_on_decode_or_disabled_forward(self):
        controller = install(self.base, width=8, heads=2)
        controller.capture_diagnostics(True)
        self.base(**self.inputs)
        self.assertEqual(controller.diagnostics()['visual_tokens'], 8)
        self.base(torch.tensor([[1]]), past_key_values=object())
        self.assertEqual(controller.diagnostics(), {})
        self.base(**self.inputs)
        controller.enabled = False
        self.assertEqual(controller.diagnostics(), {})
        self.base(**self.inputs)
        self.assertEqual(controller.diagnostics(), {})
        controller.remove()


if __name__ == '__main__':
    unittest.main()
